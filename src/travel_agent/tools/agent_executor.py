from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from travel_agent.agents.actions import CallToolAction
from travel_agent.domain.models import TripSpec
from travel_agent.domain.tool_models import (
    POIFacts,
    POISearchQuery,
    RouteMode,
    RouteResult,
    ToolCallContext,
    ToolStatus,
    UnknownFactPolicy,
)
from travel_agent.evidence.models import (
    AgentObservation,
    EvidenceKind,
    EvidenceRecord,
    ObservationKind,
    evidence_content_hash,
)
from travel_agent.execution.context import (
    consume_evidence_record,
    current_run_context,
    record_agent_event,
)
from travel_agent.execution.models import TraceEventType
from travel_agent.domain.optimization_models import OptimizationBudget
from travel_agent.planning.defaults import POIDefaultPolicy
from travel_agent.planning.optimization import (
    collect_route_matrix_queries,
    select_optimization_pois,
)
from travel_agent.planning.policy import PlanningPolicy
from travel_agent.planning.search_plan import build_search_plan, select_search_candidates
from travel_agent.planning.stay import derive_day_boundaries, resolve_stay_anchor
from travel_agent.identity.models import Principal
from travel_agent.memory.models import AgentRole
from travel_agent.memory.service import PreferenceMemoryService
from travel_agent.tools.gateway import ToolGateway
from travel_agent.tools.registry import ToolRegistry
from travel_agent.weather.gateway import WeatherToolGateway
from travel_agent.weather.policy import classify_forecast


logger = logging.getLogger(__name__)


class POISearchToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=1, max_length=100)
    keywords: tuple[str, ...] = Field(
        default=(), max_length=12,
        description="兼容字段；可省略或填写景点，服务端通用检索固定使用景点和类型110000。",
    )
    limit: int = Field(default=10, ge=1, le=25)


class RouteMatrixToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: RouteMode = RouteMode.DRIVING
    strategy: int = Field(default=32, ge=0, le=100)


class AnchorResolveToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roles: tuple[Literal["arrival", "departure", "accommodation"], ...] = Field(
        min_length=1, max_length=3
    )


class RouteDeltaToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    affected_dates: tuple[str, ...] = Field(min_length=1, max_length=2)


class WeatherSnapshotToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    destination: str = Field(min_length=1, max_length=100)
    start_date: str
    end_date: str


class PreferenceRetrieveToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categories: tuple[str, ...] = Field(default=(), max_length=20)
    destination: str | None = Field(default=None, max_length=100)


TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "poi.search": POISearchToolInput,
    "anchor.resolve": AnchorResolveToolInput,
    "route.build_matrix": RouteMatrixToolInput,
    "route.load_delta": RouteDeltaToolInput,
    "weather.snapshot": WeatherSnapshotToolInput,
    "preference.retrieve": PreferenceRetrieveToolInput,
}


class ToolActionExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: AgentObservation
    evidence: tuple[EvidenceRecord, ...] = ()
    poi_facts: tuple[POIFacts, ...] = ()
    route_results: dict[str, RouteResult] = Field(default_factory=dict)


AgentToolHandler = Callable[
    [BaseModel, CallToolAction, TripSpec, str], Awaitable[ToolActionExecution]
]


class AgentToolExecutor:
    """将 Planner 的粗粒度 Tool Action 映射到现有可靠 Tool Gateway。"""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        gateway: ToolGateway,
        handlers: dict[str, AgentToolHandler] | None = None,
        defaults: POIDefaultPolicy | None = None,
        planning_policy: PlanningPolicy = PlanningPolicy(),
        optimization_budget: OptimizationBudget = OptimizationBudget(),
        max_route_batch_size: int | None = None,
        weather_gateway: WeatherToolGateway | None = None,
        preference_service: PreferenceMemoryService | None = None,
    ) -> None:
        self.registry = registry
        self.gateway = gateway
        self.handlers = dict(handlers or {})
        self.defaults = defaults or POIDefaultPolicy(
            UnknownFactPolicy.ASSUME_WITH_WARNING
        )
        self.planning_policy = planning_policy
        self.optimization_budget = optimization_budget
        self.max_route_batch_size = max_route_batch_size
        self.weather_gateway = weather_gateway
        self.preference_service = preference_service

    async def execute(
        self,
        action: CallToolAction,
        *,
        trip: TripSpec,
        thread_id: str,
        run_id: str,
        poi_facts: tuple[POIFacts, ...] = (),
    ) -> ToolActionExecution:
        descriptor = self.registry.get(action.tool_name)
        schema = TOOL_INPUT_MODELS.get(action.tool_name)
        if descriptor is None or schema is None:
            return self._failure(action, "unknown_tool", retryable=False)
        try:
            parsed = schema.model_validate(action.arguments)
        except ValidationError:
            return self._failure(action, "invalid_tool_arguments", retryable=False)

        if action.tool_name == "poi.search":
            result = await self._search_pois(
                parsed, action, trip, thread_id, run_id
            )
        elif action.tool_name == "route.build_matrix":
            result = await self._build_route_matrix(
                parsed, action, trip, thread_id, run_id, poi_facts
            )
        elif action.tool_name == "anchor.resolve":
            result = await self._resolve_anchor(
                parsed, action, trip, run_id, poi_facts
            )
        elif action.tool_name == "weather.snapshot":
            result = await self._weather_snapshot(
                parsed, action, trip, thread_id, run_id
            )
        elif action.tool_name == "preference.retrieve":
            result = await self._retrieve_preferences(
                parsed, action, trip, run_id
            )
        elif action.tool_name in self.handlers:
            result = await self.handlers[action.tool_name](
                parsed, action, trip, run_id
            )
        else:
            result = self._failure(action, "tool_handler_unavailable", retryable=False)

        for record in result.evidence:
            consume_evidence_record()
            record_agent_event(
                TraceEventType.EVIDENCE_RECORDED,
                status="recorded",
                operation="evidence.record",
                attributes={
                    "evidence_id": record.evidence_id,
                    "evidence_kind": record.kind.value,
                    "tool_name": action.tool_name,
                },
            )
        return result

    async def _search_pois(
        self,
        parsed: BaseModel,
        action: CallToolAction,
        trip: TripSpec,
        thread_id: str,
        run_id: str,
    ) -> ToolActionExecution:
        data = POISearchToolInput.model_validate(parsed)
        try:
            queries = build_search_plan(trip, per_query_limit=data.limit, max_queries=self.planning_policy.poi_max_queries,
                                        city=data.city)
        except ValueError:
            return self._failure(action, "must_visit_query_budget_exceeded", retryable=False)
        context = ToolCallContext(thread_id=thread_id)
        required = [query for query in queries if query.exact_match]
        ordinary = [query for query in queries if not query.exact_match]
        results = []
        for query in required:
            result = (await self.gateway.search_pois([query], context))[0]
            if result.status is ToolStatus.FAILED:
                error = result.error
                return self._failure(action, error.code if error else "poi_search_failed", retryable=bool(error and error.retryable))
            results.append(result)
        # 通用搜索固定为景点 + 110000，不把自然/人文等偏好当作地点名称搜索。
        if ordinary:
            logger.info("poi.general_search | city=%s keyword=景点 types=110000", data.city)
        # 普通搜索在所有必去查询完成后执行，仍由 Gateway 统一限速和缓存。
        results.extend(await self.gateway.search_pois(ordinary, context))
        queries = [*required, *ordinary]
        failures = [item for item in results if item.status is ToolStatus.FAILED]
        if failures:
            error = failures[0].error
            return self._failure(
                action,
                error.code if error else "poi_search_failed",
                retryable=bool(error and error.retryable),
            )
        facts = tuple(select_search_candidates(queries, results, max_candidates=self.planning_policy.poi_candidate_limit))
        summary = f"获得 {len(facts)} 个标准化 POI 候选，覆盖 {len(queries)} 个查询"
        evidence = self._evidence(
            run_id=run_id,
            kind=EvidenceKind.POI,
            subject_key="poi.candidates.interests",
            summary=summary,
            provider=results[0].provider if results else None,
            confidence=min((item.data[0].data_confidence for item in results if item.data), default=0.8),
        )
        return ToolActionExecution(
            observation=AgentObservation(
                kind=ObservationKind.TOOL_RESULT,
                action_id=action.action_id,
                summary=summary,
                evidence_ids=(evidence.evidence_id,),
            ),
            evidence=(evidence,),
            poi_facts=facts,
        )

    async def _build_route_matrix(
        self,
        parsed: BaseModel,
        action: CallToolAction,
        trip: TripSpec,
        thread_id: str,
        run_id: str,
        poi_facts: tuple[POIFacts, ...],
    ) -> ToolActionExecution:
        data = RouteMatrixToolInput.model_validate(parsed)
        planning_pois = [
            resolution.poi
            for facts in poi_facts
            if (resolution := self.defaults.resolve(facts, trip)).poi is not None
        ]
        optimization_pois = select_optimization_pois(
            trip, planning_pois, self.optimization_budget
        )
        stay = resolve_stay_anchor(trip, planning_pois)
        boundaries = derive_day_boundaries(trip, stay)
        modes = tuple(
            dict.fromkeys((data.mode, *self.planning_policy.route_modes))
        )
        queries = collect_route_matrix_queries(
            trip,
            optimization_pois,
            modes=modes,
            # Provider 路线策略属于服务端安全策略；模型参数只作为受校验的意图，
            # 不能令 Solve 与 Tool 阶段使用不同的路线键。
            strategy=self.planning_policy.route_strategy,
            max_walking_leg_meters=self.planning_policy.max_walking_leg_meters,
            stay_resolution=stay,
            day_boundaries=boundaries,
        )
        descriptor = self.registry.get(action.tool_name)
        batch_limit = self.max_route_batch_size or (
            descriptor.max_batch_size if descriptor is not None else 256
        )
        if len(queries) > batch_limit:
            logger.warning(
                "agent_tool.route_batch_rejected | thread_id=%s query_count=%s "
                "batch_limit=%s",
                thread_id,
                len(queries),
                batch_limit,
            )
            return self._failure(action, "route_batch_limit_exceeded", retryable=False)
        if not queries:
            return self._failure(action, "route_points_missing", retryable=False)
        results = await self.gateway.get_routes(
            queries, ToolCallContext(thread_id=thread_id)
        )
        failures = [item for item in results.values() if item.status is ToolStatus.FAILED]
        if failures:
            error = next((item.error for item in failures if item.error), None)
            return self._failure(
                action,
                error.code if error else "route_matrix_failed",
                retryable=bool(error and error.retryable),
            )
        successful = {
            key: item.data
            for key, item in results.items()
            if item.status is ToolStatus.SUCCESS and item.data is not None
        }
        if not successful:
            error = next((item.error for item in results.values() if item.error), None)
            return self._failure(
                action,
                error.code if error else "route_matrix_failed",
                retryable=bool(error and error.retryable),
            )
        summary = (
            f"完整规划路线矩阵已获得 {len(successful)}/{len(queries)} 条标准化路线，"
            f"覆盖 {len(optimization_pois)} 个候选 POI"
        )
        evidence = self._evidence(
            run_id=run_id,
            kind=EvidenceKind.ROUTE,
            subject_key="route.matrix.required",
            summary=summary,
            provider=next(iter(successful.values())).provider,
            confidence=min(item.data_confidence for item in successful.values()),
        )
        return ToolActionExecution(
            observation=AgentObservation(
                kind=ObservationKind.TOOL_RESULT,
                action_id=action.action_id,
                summary=summary,
                evidence_ids=(evidence.evidence_id,),
            ),
            evidence=(evidence,),
            route_results=successful,
        )

    async def _resolve_anchor(
        self,
        parsed: BaseModel,
        action: CallToolAction,
        trip: TripSpec,
        run_id: str,
        poi_facts: tuple[POIFacts, ...],
    ) -> ToolActionExecution:
        data = AnchorResolveToolInput.model_validate(parsed)
        planning_pois = [
            resolution.poi
            for facts in poi_facts
            if (resolution := self.defaults.resolve(facts, trip)).poi is not None
        ]
        stay = resolve_stay_anchor(trip, planning_pois)
        values = {
            "arrival": trip.arrival.name,
            "departure": trip.departure.name,
            "accommodation": (
                stay.anchor.name if stay.anchor is not None else "unresolved"
            ),
        }
        summary = "；".join(f"{role}={values[role]}" for role in data.roles)
        summary = f"锚点解析完成：{summary}；stay_mode={stay.mode.value}"
        evidence = self._evidence(
            run_id=run_id,
            kind=EvidenceKind.USER_CONSTRAINT,
            subject_key="trip.anchor.resolved",
            summary=summary,
            provider="planning_policy",
            confidence=stay.confidence if "accommodation" in data.roles else 1.0,
        )
        return ToolActionExecution(
            observation=AgentObservation(
                kind=ObservationKind.TOOL_RESULT,
                action_id=action.action_id,
                summary=summary,
                evidence_ids=(evidence.evidence_id,),
            ),
            evidence=(evidence,),
        )

    async def _weather_snapshot(
        self,
        parsed: BaseModel,
        action: CallToolAction,
        trip: TripSpec,
        thread_id: str,
        run_id: str,
    ) -> ToolActionExecution:
        data = WeatherSnapshotToolInput.model_validate(parsed)
        if self.weather_gateway is None:
            return self._failure(action, "weather_handler_unavailable", retryable=False)
        if (
            data.destination.casefold() != trip.destination.casefold()
            or data.start_date != trip.start_date.isoformat()
            or data.end_date != trip.end_date.isoformat()
        ):
            return self._failure(action, "weather_scope_mismatch", retryable=False)
        context = ToolCallContext(thread_id=thread_id)
        location = await self.weather_gateway.resolve_location(
            trip.destination, context
        )
        if location.status is ToolStatus.FAILED or location.data is None:
            error = location.error
            return self._failure(
                action,
                error.code if error else "weather_location_failed",
                retryable=bool(error and error.retryable),
            )
        forecast = await self.weather_gateway.get_forecast(
            location.data,
            start_date=trip.start_date,
            end_date=trip.end_date,
            context=context,
        )
        if forecast.status is ToolStatus.FAILED or forecast.data is None:
            error = forecast.error
            return self._failure(
                action,
                error.code if error else "weather_forecast_failed",
                retryable=bool(error and error.retryable),
            )
        risks = classify_forecast(forecast.data.days)
        warning_days = [item for item in risks if item.level.value != "normal"]
        summary = (
            f"天气快照覆盖 {len(forecast.data.days)} 天，"
            f"风险日期 {len(warning_days)} 天："
            + ",".join(
                f"{item.date.isoformat()}={item.level.value}"
                for item in warning_days
            )
        )[:800]
        evidence = self._evidence(
            run_id=run_id,
            kind=EvidenceKind.WEATHER,
            subject_key="weather.snapshot.trip_range",
            summary=summary,
            provider=forecast.provider,
            confidence=1.0,
        )
        return ToolActionExecution(
            observation=AgentObservation(
                kind=ObservationKind.TOOL_RESULT,
                action_id=action.action_id,
                summary=summary,
                evidence_ids=(evidence.evidence_id,),
            ),
            evidence=(evidence,),
        )

    async def _retrieve_preferences(
        self,
        parsed: BaseModel,
        action: CallToolAction,
        trip: TripSpec,
        run_id: str,
    ) -> ToolActionExecution:
        data = PreferenceRetrieveToolInput.model_validate(parsed)
        if self.preference_service is None:
            return self._failure(
                action, "preference_handler_unavailable", retryable=False
            )
        if data.destination and data.destination.casefold() != trip.destination.casefold():
            return self._failure(
                action, "preference_scope_mismatch", retryable=False
            )
        run = current_run_context()
        principal = Principal(
            tenant_id=run.record.tenant_id if run is not None else "local",
            user_id=run.record.user_id if run is not None else "demo",
        )
        context = await self.preference_service.context_for_trip(
            principal,
            trip=trip,
            draft=None,
            agent_role=AgentRole.PLANNER,
        )
        requested = set(data.categories)
        summaries = [
            item
            for item in context.summaries
            if not requested or item.category.value in requested
        ]
        rendered = ", ".join(
            f"{item.category.value}={item.value}" for item in summaries
        )
        summary = (
            f"检索到 {len(summaries)} 条已确认且作用域匹配的偏好"
            + (f"：{rendered}" if rendered else "")
        )[:800]
        evidence = self._evidence(
            run_id=run_id,
            kind=EvidenceKind.PREFERENCE,
            subject_key="preference.confirmed.relevant",
            summary=summary,
            provider="preference_memory",
            confidence=(
                min((item.confidence for item in summaries), default=1.0)
            ),
        )
        return ToolActionExecution(
            observation=AgentObservation(
                kind=ObservationKind.TOOL_RESULT,
                action_id=action.action_id,
                summary=summary,
                evidence_ids=(evidence.evidence_id,),
            ),
            evidence=(evidence,),
        )

    @staticmethod
    def _failure(
        action: CallToolAction, code: str, *, retryable: bool
    ) -> ToolActionExecution:
        return ToolActionExecution(
            observation=AgentObservation(
                kind=ObservationKind.TOOL_FAILURE,
                action_id=action.action_id,
                summary="领域工具未能产生可用证据",
                error_code=code,
                retryable=retryable,
            )
        )

    @staticmethod
    def _evidence(
        *,
        run_id: str,
        kind: EvidenceKind,
        subject_key: str,
        summary: str,
        provider: str | None,
        confidence: float,
    ) -> EvidenceRecord:
        now = datetime.now(timezone.utc)
        return EvidenceRecord(
            run_id=run_id,
            kind=kind,
            subject_key=subject_key,
            summary=summary,
            provider=provider,
            observed_at=now,
            confidence=confidence,
            payload_ref=f"run:{run_id}:{subject_key}",
            content_hash=evidence_content_hash(
                kind=kind, subject_key=subject_key, summary=summary
            ),
        )


def tool_arguments_hash(action: CallToolAction) -> str:
    encoded = json.dumps(
        action.arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
