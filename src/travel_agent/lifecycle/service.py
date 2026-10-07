from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from travel_agent.domain.lifecycle_models import (
    ActionReceipt,
    LifecycleResumeRequest,
    PlanDiff,
    PlanningSnapshot,
    PlanSessionRecord,
    PlanSessionResponse,
    PlanSessionStatus,
)
from travel_agent.domain.models import PlanningRequest, PlanningResponse
from travel_agent.domain.weather_models import (
    RefreshWeatherAction,
    WeatherEventView,
    WeatherRefreshRequest,
    WeatherStateView,
)
from travel_agent.graph.workflow import run_planning
from travel_agent.lifecycle.errors import (
    LifecycleActionError,
    LifecycleConflictError,
    LifecycleNotFoundError,
)
from travel_agent.lifecycle.repository import PlanRepository
from travel_agent.lifecycle.workflow import (
    response_from_session,
    resume_lifecycle,
    start_lifecycle,
)
from travel_agent.requirements.models import (
    ClarificationResumeRequest,
    NaturalPlanningRequest,
)
from travel_agent.requirements.workflow import resume_natural_planning, run_natural_planning
from travel_agent.weather.persistence import weather_state_view


logger = logging.getLogger(__name__)


class PlanLifecycleService:
    def __init__(
        self,
        *,
        repository: PlanRepository,
        planning_workflow: CompiledStateGraph,
        planning_runner: Callable[[PlanningRequest, str], Awaitable[PlanningResponse]] | None = None,
        lifecycle_workflow: CompiledStateGraph,
        requirement_workflow: CompiledStateGraph | None,
        weather_stale_max_seconds: int = 21_600,
    ) -> None:
        self._repository = repository
        self._planning_workflow = planning_workflow
        self._planning_runner = planning_runner
        self._lifecycle_workflow = lifecycle_workflow
        self._requirement_workflow = requirement_workflow
        self._weather_stale_max_seconds = weather_stale_max_seconds
        self._locks: dict[str, asyncio.Lock] = {}

    async def create(
        self,
        request: PlanningRequest,
        *,
        session_id: str | None = None,
        tenant_id: str = "local",
        user_id: str = "demo",
    ) -> PlanSessionResponse:
        resolved_id = session_id or str(uuid4())
        planning_thread_id = f"planning:{resolved_id}"
        response = (
            await self._planning_runner(request, planning_thread_id)
            if self._planning_runner is not None
            else await run_planning(
                self._planning_workflow, request, thread_id=planning_thread_id
            )
        )
        if response.status != "completed" or response.selected_plan is None:
            raise LifecycleActionError(
                resolved_id,
                "planning_not_completed",
                response.message or "规划没有产生可进入生命周期的候选",
            )
        snapshot = await self._planning_snapshot(planning_thread_id)
        session = PlanSessionRecord(
            session_id=resolved_id,
            tenant_id=tenant_id,
            user_id=user_id,
            lifecycle_thread_id=f"lifecycle:{resolved_id}",
            status=PlanSessionStatus.AWAITING_CANDIDATE_SELECTION,
            snapshot=snapshot,
        )
        await self._repository.create(session)
        return await start_lifecycle(
            self._lifecycle_workflow,
            self._repository,
            session,
            weather_stale_max_seconds=self._weather_stale_max_seconds,
        )

    async def create_from_checkpoint(
        self,
        thread_id: str,
        *,
        session_id: str,
        tenant_id: str,
        user_id: str,
    ) -> PlanSessionResponse:
        """把已完成的自然语言规划接入选择 Interrupt，不重复调用模型或工具。"""
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            try:
                existing = await self._repository.get(session_id)
            except LifecycleNotFoundError:
                existing = None
            if existing is not None:
                if existing.tenant_id != tenant_id or existing.user_id != user_id:
                    raise LifecycleConflictError(session_id, code="session_owner_mismatch")
                logger.info("lifecycle.selection_session.reused | session_id=%s", session_id)
                return await self.get(session_id)
            if self._requirement_workflow is None:
                raise RuntimeError("natural-language planning is not configured")
            state = await self._requirement_workflow.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            values = state.values
            if (
                values.get("status") != "completed"
                or values.get("tenant_id") != tenant_id
                or values.get("user_id") != user_id
            ):
                raise LifecycleActionError(
                    session_id, "planning_not_completed", "没有可用于选择的已完成规划，请重新生成计划"
                )
            session = PlanSessionRecord(
                session_id=session_id,
                lifecycle_thread_id=f"lifecycle:{session_id}",
                intake_thread_id=thread_id,
                status=PlanSessionStatus.AWAITING_CANDIDATE_SELECTION,
                tenant_id=tenant_id,
                user_id=user_id,
                snapshot=await self._planning_snapshot(thread_id),
            )
            await self._repository.create(session)
            logger.info(
                "lifecycle.selection_session.created | session_id=%s source_thread_id=%s candidate_count=%s",
                session_id, thread_id, len(session.snapshot.candidates),
            )
            return await start_lifecycle(
                self._lifecycle_workflow,
                self._repository,
                session,
                weather_stale_max_seconds=self._weather_stale_max_seconds,
            )

    async def create_from_text(
        self,
        request: NaturalPlanningRequest,
        *,
        session_id: str | None = None,
        tenant_id: str = "local",
        user_id: str = "demo",
    ) -> PlanSessionResponse:
        if self._requirement_workflow is None:
            raise RuntimeError("natural-language planning is not configured")
        resolved_id = session_id or str(uuid4())
        intake_thread_id = f"intake:{resolved_id}"
        response = await run_natural_planning(
            self._requirement_workflow,
            request,
            thread_id=intake_thread_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        if response.status == "needs_clarification":
            session = PlanSessionRecord(
                session_id=resolved_id,
                lifecycle_thread_id=f"lifecycle:{resolved_id}",
                intake_thread_id=intake_thread_id,
                status=PlanSessionStatus.NEEDS_REQUIREMENT_CLARIFICATION,
                tenant_id=tenant_id,
                user_id=user_id,
                external_interrupt=(
                    response.interrupt.model_dump(mode="json")
                    if response.interrupt is not None
                    else None
                ),
            )
            await self._repository.create(session)
            return response_from_session(
                session,
                message=response.message,
                weather_stale_max_seconds=self._weather_stale_max_seconds,
            )
        if response.status != "completed" or response.planning is None:
            raise LifecycleActionError(
                resolved_id,
                "planning_not_completed",
                response.message or "规划没有产生可进入生命周期的候选",
            )
        snapshot = await self._planning_snapshot(intake_thread_id)
        session = PlanSessionRecord(
            session_id=resolved_id,
            lifecycle_thread_id=f"lifecycle:{resolved_id}",
            intake_thread_id=intake_thread_id,
            status=PlanSessionStatus.AWAITING_CANDIDATE_SELECTION,
            tenant_id=tenant_id,
            user_id=user_id,
            snapshot=snapshot,
        )
        await self._repository.create(session)
        return await start_lifecycle(
            self._lifecycle_workflow,
            self._repository,
            session,
            weather_stale_max_seconds=self._weather_stale_max_seconds,
        )

    async def resume(
        self, session_id: str, request: LifecycleResumeRequest
    ) -> PlanSessionResponse:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = await self._repository.get(session_id)
            receipt = session.receipts.get(str(request.request_id))
            if receipt is not None:
                if session.status is PlanSessionStatus.NEEDS_REQUIREMENT_CLARIFICATION:
                    return response_from_session(
                        session,
                        weather_stale_max_seconds=self._weather_stale_max_seconds,
                    )
                return await resume_lifecycle(
                    self._lifecycle_workflow,
                    self._repository,
                    request,
                    session_id=session_id,
                    weather_stale_max_seconds=self._weather_stale_max_seconds,
                )
            if session.status is PlanSessionStatus.NEEDS_REQUIREMENT_CLARIFICATION:
                return await self._resume_requirement(session, request)
            return await resume_lifecycle(
                self._lifecycle_workflow,
                self._repository,
                request,
                session_id=session_id,
                weather_stale_max_seconds=self._weather_stale_max_seconds,
            )

    async def get(self, session_id: str) -> PlanSessionResponse:
        session = await self._repository.get(session_id)
        if session.external_interrupt is not None:
            return response_from_session(
                session,
                weather_stale_max_seconds=self._weather_stale_max_seconds,
            )
        snapshot = await self._lifecycle_workflow.aget_state(
            {"configurable": {"thread_id": session.lifecycle_thread_id}}
        )
        interruptions = tuple(
            value
            for task in snapshot.tasks
            for value in getattr(task, "interrupts", ())
        )
        return response_from_session(
            session,
            interruptions=interruptions,
            weather_stale_max_seconds=self._weather_stale_max_seconds,
        )

    async def refresh_weather(
        self, session_id: str, request: WeatherRefreshRequest
    ) -> PlanSessionResponse:
        current = await self.get(session_id)
        if current.interrupt is None:
            raise LifecycleConflictError(session_id, code="weather_refresh_not_resumable")
        return await self.resume(
            session_id,
            LifecycleResumeRequest(
                interrupt_id=current.interrupt.id,
                request_id=request.request_id,
                expected_active_version_id=request.expected_active_version_id,
                expected_session_revision=request.expected_session_revision,
                action=RefreshWeatherAction(),
            ),
        )

    async def weather(self, session_id: str) -> WeatherStateView:
        session = await self._repository.get(session_id)
        return weather_state_view(
            session, stale_max_seconds=self._weather_stale_max_seconds
        )

    async def weather_events(self, session_id: str) -> tuple[WeatherEventView, ...]:
        session = await self._repository.get(session_id)
        return tuple(
            WeatherEventView(
                event=event,
                receipt=session.weather_monitor.event_receipts.get(event.event_id),
            )
            for event in sorted(
                session.weather_events.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
        )

    async def weather_event(
        self, session_id: str, event_id: str
    ) -> WeatherEventView:
        session = await self._repository.get(session_id)
        event = session.weather_events.get(event_id)
        if event is None:
            raise LifecycleActionError(
                session_id, "weather_event_not_found", "天气事件不存在"
            )
        return WeatherEventView(
            event=event,
            receipt=session.weather_monitor.event_receipts.get(event_id),
        )

    async def versions(self, session_id: str):
        session = await self._repository.get(session_id)
        return tuple(sorted(session.versions.values(), key=lambda item: item.number))

    async def version(self, session_id: str, version_id: str):
        session = await self._repository.get(session_id)
        version = session.versions.get(version_id)
        if version is None:
            raise LifecycleActionError(session_id, "plan_version_not_found", "计划版本不存在")
        return version

    async def diff(self, session_id: str, from_id: str, to_id: str) -> PlanDiff:
        session = await self._repository.get(session_id)
        if to_id.startswith("P"):
            preview = session.previews.get(to_id)
            if preview is None or preview.base_version_id != from_id:
                raise LifecycleActionError(session_id, "plan_diff_not_found", "计划差异不存在")
            return preview.diff
        target = session.versions.get(to_id)
        if target is None or target.parent_version_id != from_id:
            raise LifecycleActionError(session_id, "plan_diff_not_found", "计划差异不存在")
        preview = next(
            (
                item
                for item in session.previews.values()
                if item.status.value == "approved"
                and item.base_version_id == from_id
                and item.candidate == target.candidate
            ),
            None,
        )
        if preview is None:
            raise LifecycleActionError(session_id, "plan_diff_not_found", "计划差异不存在")
        return preview.diff.model_copy(update={"to_id": to_id})

    async def _resume_requirement(
        self, session: PlanSessionRecord, request: LifecycleResumeRequest
    ) -> PlanSessionResponse:
        if request.action.kind != "clarify_requirement":
            raise LifecycleActionError(session.session_id, "invalid_action", "当前正在等待需求澄清")
        if session.external_interrupt is None or request.interrupt_id != session.external_interrupt.get("id"):
            raise LifecycleConflictError(session.session_id, code="stale_interrupt")
        assert self._requirement_workflow is not None
        response = await resume_natural_planning(
            self._requirement_workflow,
            ClarificationResumeRequest(
                interrupt_id=request.interrupt_id,
                request_id=request.request_id,
                answer=request.action.answer,
            ),
            thread_id=session.intake_thread_id or "",
            tenant_id=session.tenant_id,
            user_id=session.user_id,
        )
        previous = session.session_revision
        session.session_revision += 1
        session.receipts[str(request.request_id)] = ActionReceipt(
            request_id=str(request.request_id),
            action_kind=request.action.kind,
            resulting_revision=session.session_revision,
        )
        if response.status == "needs_clarification":
            session.external_interrupt = (
                response.interrupt.model_dump(mode="json")
                if response.interrupt is not None
                else None
            )
            await self._repository.save(session, expected_revision=previous)
            return response_from_session(
                session,
                message=response.message,
                weather_stale_max_seconds=self._weather_stale_max_seconds,
            )
        if response.status != "completed" or response.planning is None:
            raise LifecycleActionError(
                session.session_id,
                "planning_not_completed",
                response.message or "规划没有产生可进入生命周期的候选",
            )
        session.snapshot = await self._planning_snapshot(session.intake_thread_id or "")
        session.status = PlanSessionStatus.AWAITING_CANDIDATE_SELECTION
        session.external_interrupt = None
        await self._repository.save(session, expected_revision=previous)
        return await start_lifecycle(
            self._lifecycle_workflow,
            self._repository,
            session,
            weather_stale_max_seconds=self._weather_stale_max_seconds,
        )

    async def _planning_snapshot(self, thread_id: str) -> PlanningSnapshot:
        state = await self._planning_workflow.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        values = state.values
        selected = values.get("selected_plan")
        if selected is None:
            raise RuntimeError("completed planning checkpoint has no selected plan")
        kernel_snapshot = values.get("kernel_snapshot")
        if kernel_snapshot is not None:
            return PlanningSnapshot(
                trip=values["trip"],
                candidates=tuple(values["candidates"]),
                recommended_candidate_id=selected.id,
                candidate_drafts=tuple(kernel_snapshot.candidate_drafts),
                planning_pois=tuple(kernel_snapshot.planning_pois),
                route_results=dict(kernel_snapshot.route_results),
                critic_status=values.get("critic_status", "not_run"),
            )
        return PlanningSnapshot(
            trip=values["trip"],
            candidates=tuple(values["candidates"]),
            recommended_candidate_id=selected.id,
            candidate_drafts=tuple(values["candidate_drafts"]),
            planning_pois=tuple(values["planning_pois"]),
            route_results=dict(values["route_results"]),
            critic_status=values["critic_status"],
        )
