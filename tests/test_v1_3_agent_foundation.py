from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from travel_agent.agents.action_policy import ActionPolicyContext, validate_action
from travel_agent.agents.actions import (
    AGENT_ACTION_ADAPTER,
    AgentPhase,
    CallToolAction,
    FinishAction,
    SolveAction,
    action_fingerprint,
)
from travel_agent.evidence.models import (
    EvidenceGapStatus,
    EvidenceKind,
    EvidenceRecord,
    evidence_content_hash,
)
from travel_agent.evidence.policy import derive_evidence_gaps
from travel_agent.evidence.repository import InMemoryEvidenceRepository
from travel_agent.tools.registry import default_agent_tool_registry
from travel_agent.tools.agent_executor import AgentToolExecutor
from travel_agent.agents.planner.gateway import PlannerGateway
from travel_agent.agents.planner.providers.mock import MockPlannerModel
from travel_agent.agents.replanner.gateway import ReplannerGateway
from travel_agent.agents.replanner.providers.mock import MockReplannerModel
from travel_agent.domain.models import (
    PlanningRequest,
    ValidationResult,
    Violation,
    ViolationSeverity,
)
from travel_agent.graph.agentic_workflow import build_agentic_workflow, run_agentic_planning
from travel_agent.planning.defaults import POIDefaultPolicy
from travel_agent.planning.kernel import (
    PlanningKernelService,
    PlanningKernelSolveResult,
)
from travel_agent.domain.tool_models import ToolResult, UnknownFactPolicy
from travel_agent.config import AgentMode, Settings
from travel_agent.runtime import PlanningRuntime
from travel_agent.execution.models import TraceEventType
from travel_agent.agents.replanner.models import ModelRepairProposal
from travel_agent.agents.replanner.policy import RepairPolicyContext, validate_repair_proposal
from travel_agent.domain.repair_models import RepairAction, RepairActionKind
from travel_agent.critique.gateway import CriticGateway
from travel_agent.critique.quality import CriticPolicy
from travel_agent.domain.critique_models import (
    CriticStatus,
    DimensionCritique,
    SoftCritique,
    SoftCriticProviderOutput,
    SoftDimension,
    SuggestedActionKind,
    SuggestedSoftAction,
)
from travel_agent.planning.quality import QualityReviewService
from travel_agent.domain.weather_models import (
    DailyWeather,
    WeatherForecast,
    WeatherLocation,
    WeatherPhenomenon,
)
from travel_agent.identity.models import Principal
from travel_agent.memory.models import MemoryCategory, PreferenceCreateRequest
from travel_agent.memory.repository import InMemoryPreferenceRepository
from travel_agent.memory.service import PreferenceMemoryService


def _evidence(run_id: str, kind: EvidenceKind, subject_key: str) -> EvidenceRecord:
    summary = f"{subject_key} available"
    return EvidenceRecord(
        run_id=run_id,
        kind=kind,
        subject_key=subject_key,
        summary=summary,
        confidence=1.0,
        observed_at=datetime.now(timezone.utc),
        content_hash=evidence_content_hash(
            kind=kind, subject_key=subject_key, summary=summary
        ),
    )


def test_agent_action_is_discriminated_and_fingerprint_is_stable():
    payload = {
        "kind": "call_tool",
        "reason_code": "poi_missing",
        "decision_summary": "需要候选兴趣点",
        "tool_name": "poi.search",
        "arguments": {"city": "杭州", "keywords": ["自然"]},
        "evidence_goal": "补齐 POI",
    }
    first = AGENT_ACTION_ADAPTER.validate_python(payload)
    second = AGENT_ACTION_ADAPTER.validate_python(payload)

    assert isinstance(first, CallToolAction)
    assert action_fingerprint(first, AgentPhase.RESEARCHING) == action_fingerprint(
        second, AgentPhase.RESEARCHING
    )
    with pytest.raises(ValidationError):
        AGENT_ACTION_ADAPTER.validate_python({**payload, "provider": "amap"})


def test_action_guard_rejects_unknown_evidence_forbidden_identity_and_early_finish():
    registry = default_agent_tool_registry()
    gaps = ()
    base = ActionPolicyContext(
        phase=AgentPhase.RESEARCHING,
        run_id="run-a",
        evidence_ids=frozenset({"ev-1"}),
        evidence_gaps=gaps,
        action_fingerprints=(),
        registry=registry,
    )
    unknown_evidence = SolveAction(
        reason_code="ready",
        decision_summary="solve",
        evidence_ids=("ev-unknown",),
    )
    forbidden_identity = CallToolAction(
        reason_code="need_preferences",
        decision_summary="retrieve",
        tool_name="preference.retrieve",
        arguments={"user_id": "other-user"},
        evidence_goal="preferences",
    )
    early_finish = FinishAction(
        reason_code="finish",
        decision_summary="finish",
        candidate_id="candidate-a",
    )

    assert validate_action(unknown_evidence, base).reason_code == "unknown_evidence_reference"
    assert validate_action(forbidden_identity, base).reason_code == "forbidden_tool_argument"
    assert validate_action(early_finish, base).reason_code == "action_not_allowed_in_phase"

    invalid_arguments = CallToolAction(
        reason_code="bad_route",
        decision_summary="bad route schema",
        tool_name="route.build_matrix",
        arguments={"mode": "flying"},
        evidence_goal="route",
    )
    assert validate_action(invalid_arguments, base).reason_code == "invalid_tool_arguments"


def test_repair_policy_rejects_must_visit_before_side_effect():
    day = date(2026, 10, 2)
    proposal = ModelRepairProposal(
        target_candidate_id="candidate-a",
        source_violation_fingerprints=("violation-a",),
        actions=(RepairAction(
            kind=RepairActionKind.REMOVE_OPTIONAL_POI,
            source_violation_type="daily_time",
            poi_id="lingyin",
            from_day=day,
            reason="remove",
            expected_effect="reduce duration",
        ),),
        affected_days=(day,),
        expected_effect_codes=("time_reduced",),
        summary="unsafe proposal",
    )
    result = validate_repair_proposal(
        proposal,
        RepairPolicyContext(
            candidate_ids=frozenset({"candidate-a"}),
            poi_ids=frozenset({"lingyin"}),
            evidence_ids=frozenset(),
            violation_fingerprints=frozenset({"violation-a"}),
            allowed_dates=frozenset({day}),
            must_visit_poi_ids=frozenset({"lingyin"}),
        ),
    )
    assert result.allowed is False
    assert result.reason_code == "must_visit_removal"


def test_evidence_policy_derives_open_then_satisfied_gaps(hangzhou_trip):
    empty = derive_evidence_gaps(hangzhou_trip, ())
    assert next(g for g in empty if g.key == "poi.candidates.interests").status is EvidenceGapStatus.OPEN

    records = (
        _evidence("run-a", EvidenceKind.POI, "poi.candidates.interests"),
        _evidence("run-a", EvidenceKind.ROUTE, "route.matrix.required"),
    )
    satisfied = derive_evidence_gaps(hangzhou_trip, records)
    assert next(g for g in satisfied if g.key == "poi.candidates.interests").status is EvidenceGapStatus.SATISFIED
    assert next(g for g in satisfied if g.key == "route.matrix.required").status is EvidenceGapStatus.SATISFIED


@pytest.mark.asyncio
async def test_evidence_repository_is_run_scoped_and_content_idempotent():
    repository = InMemoryEvidenceRepository(max_records_per_run=2)
    record = _evidence("run-a", EvidenceKind.POI, "poi.candidates.interests")

    assert await repository.put(record) == record
    assert await repository.put(record) == record
    assert await repository.get("run-a", record.evidence_id) == record
    assert await repository.get("run-b", record.evidence_id) is None
    assert len(await repository.list_for_run("run-a")) == 1


@pytest.mark.asyncio
async def test_dynamic_executor_wires_anchor_weather_and_preference_handlers(
    workflow_harness, hangzhou_trip
):
    class WeatherGatewayFixture:
        async def resolve_location(self, destination, _context):
            return ToolResult.success(
                data=WeatherLocation(
                    city_name=destination,
                    adcode="330100",
                    provider="fixture",
                ),
                provider="fixture",
            )

        async def get_forecast(
            self, location, *, start_date, end_date, context
        ):
            del end_date, context
            return ToolResult.success(
                data=WeatherForecast(
                    location=location,
                    provider="fixture",
                    days=(
                        DailyWeather(
                            date=start_date,
                            day_phenomenon=WeatherPhenomenon.RAIN,
                            night_phenomenon=WeatherPhenomenon.CLOUDY,
                        ),
                    ),
                ),
                provider="fixture",
            )

    memory = PreferenceMemoryService(InMemoryPreferenceRepository())
    await memory.create_explicit(
        Principal(tenant_id="local", user_id="demo"),
        PreferenceCreateRequest(category=MemoryCategory.PACE, value="relaxed"),
    )
    executor = AgentToolExecutor(
        registry=default_agent_tool_registry(),
        gateway=workflow_harness.gateway,
        weather_gateway=WeatherGatewayFixture(),
        preference_service=memory,
    )
    anchor = await executor.execute(
        CallToolAction(
            reason_code="anchor_check",
            decision_summary="解析住宿锚点",
            tool_name="anchor.resolve",
            arguments={"roles": ["arrival", "departure", "accommodation"]},
            evidence_goal="补齐锚点证据",
        ),
        trip=hangzhou_trip,
        thread_id="handler-wiring",
        run_id="handler-wiring",
    )
    weather = await executor.execute(
        CallToolAction(
            reason_code="weather_check",
            decision_summary="读取天气",
            tool_name="weather.snapshot",
            arguments={
                "destination": hangzhou_trip.destination,
                "start_date": hangzhou_trip.start_date.isoformat(),
                "end_date": hangzhou_trip.end_date.isoformat(),
            },
            evidence_goal="补齐天气证据",
        ),
        trip=hangzhou_trip,
        thread_id="handler-wiring",
        run_id="handler-wiring",
    )
    preference = await executor.execute(
        CallToolAction(
            reason_code="preference_check",
            decision_summary="读取偏好",
            tool_name="preference.retrieve",
            arguments={"categories": ["pace"], "destination": "杭州"},
            evidence_goal="补齐偏好证据",
        ),
        trip=hangzhou_trip,
        thread_id="handler-wiring",
        run_id="handler-wiring",
    )

    assert anchor.evidence[0].kind is EvidenceKind.USER_CONSTRAINT
    assert weather.evidence[0].kind is EvidenceKind.WEATHER
    assert preference.evidence[0].kind is EvidenceKind.PREFERENCE


@pytest.mark.asyncio
async def test_dynamic_agent_uses_tool_observations_before_solve_and_finish(
    workflow_harness, hangzhou_trip
):
    registry = default_agent_tool_registry()
    evidence_repository = InMemoryEvidenceRepository()
    workflow = build_agentic_workflow(
        planning_kernel=PlanningKernelService(
            gateway=workflow_harness.gateway,
            defaults=POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING),
        ),
        planner_gateway=PlannerGateway(model=MockPlannerModel()),
        tool_executor=AgentToolExecutor(
            registry=registry,
            gateway=workflow_harness.gateway,
        ),
        evidence_repository=evidence_repository,
    )

    response = await run_agentic_planning(
        workflow,
        PlanningRequest(trip=hangzhou_trip),
        thread_id="v1-3-dynamic-happy-path",
    )

    assert response.status == "completed"
    assert response.selected_plan is not None
    assert response.selected_plan.validation is not None
    assert response.selected_plan.validation.valid is True
    assert response.agent_mode == "dynamic_planner"
    assert response.action_summary == ("call_tool", "call_tool", "solve", "finish")
    assert response.evidence_summary["poi"] == 1
    assert response.evidence_summary["route"] == 1
    assert response.evidence_summary["validation"] == len(response.candidates)


@pytest.mark.asyncio
async def test_planning_kernel_applies_model_patch_and_hard_revalidates(
    workflow_harness, hangzhou_trip
):
    registry = default_agent_tool_registry()
    defaults = POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING)
    executor = AgentToolExecutor(
        registry=registry,
        gateway=workflow_harness.gateway,
        defaults=defaults,
    )
    poi_action = CallToolAction(
        reason_code="poi_evidence_missing",
        decision_summary="load pois",
        tool_name="poi.search",
        arguments={
            "city": hangzhou_trip.destination,
            "keywords": tuple(hangzhou_trip.must_visit + hangzhou_trip.interests),
        },
        evidence_goal="poi evidence",
    )
    poi_execution = await executor.execute(
        poi_action,
        trip=hangzhou_trip,
        thread_id="kernel-model-repair",
        run_id="kernel-model-repair",
    )
    route_action = CallToolAction(
        reason_code="route_evidence_missing",
        decision_summary="load complete routes",
        tool_name="route.build_matrix",
        arguments={"mode": "driving", "strategy": 32},
        evidence_goal="route evidence",
    )
    route_execution = await executor.execute(
        route_action,
        trip=hangzhou_trip,
        thread_id="kernel-model-repair",
        run_id="kernel-model-repair",
        poi_facts=poi_execution.poi_facts,
    )
    kernel = PlanningKernelService(
        gateway=workflow_harness.gateway,
        defaults=defaults,
    )
    solved = await kernel.solve(
        hangzhou_trip,
        poi_facts=poi_execution.poi_facts,
        route_results=route_execution.route_results,
        thread_id="kernel-model-repair",
    )
    candidate = solved.snapshot.candidates[0]
    removable = next(
        (day.date, item.poi_id)
        for day in candidate.days
        for item in day.items
        if item.poi_id is not None
        and not any(
            required.casefold() in item.name.casefold()
            for required in hangzhou_trip.must_visit
        )
    )
    affected_day, poi_id = removable
    synthetic_violation = Violation(
        type="activity_time_limit",
        severity=ViolationSeverity.ERROR,
        message="synthetic repair trajectory violation",
        day=affected_day,
        entity_ids=[poi_id],
    )
    invalid_candidate = candidate.model_copy(
        update={"validation": ValidationResult.from_violations([synthetic_violation])}
    )
    invalid_snapshot = replace(
        solved.snapshot,
        candidates=tuple(
            invalid_candidate if item.id == candidate.id else item
            for item in solved.snapshot.candidates
        ),
    )
    proposal = ModelRepairProposal(
        target_candidate_id=candidate.id,
        source_violation_fingerprints=("synthetic-violation",),
        actions=(
            RepairAction(
                kind=RepairActionKind.REMOVE_OPTIONAL_POI,
                source_violation_type="activity_time_limit",
                poi_id=poi_id,
                from_day=affected_day,
                reason="remove optional activity",
                expected_effect="reduce activity duration",
            ),
        ),
        affected_days=(affected_day,),
        expected_effect_codes=("activity_time_reduced",),
        summary="minimal model patch",
    )

    repaired = await kernel.apply_repair(
        hangzhou_trip,
        invalid_snapshot,
        proposal,
        thread_id="kernel-model-repair",
    )

    assert repaired.attempt.round == 1
    assert repaired.attempt.loaded_route_count >= 0
    assert repaired.selected_plan is not None
    assert repaired.selected_plan.validation is not None
    assert repaired.selected_plan.validation.valid is True
    assert repaired.selected_plan.id.endswith("repair-r1")


@pytest.mark.asyncio
async def test_dynamic_agent_routes_invalid_candidate_through_model_repair(
    workflow_harness, hangzhou_trip
):
    class InvalidFirstKernel(PlanningKernelService):
        async def solve(self, *args, **kwargs):
            solved = await super().solve(*args, **kwargs)
            candidate = next(
                item
                for item in solved.snapshot.candidates
                if any(
                    plan_item.poi_id is not None
                    and not any(
                        required.casefold() in plan_item.name.casefold()
                        for required in hangzhou_trip.must_visit
                    )
                    for day in item.days
                    for plan_item in day.items
                )
            )
            affected_day, poi_id = next(
                (day.date, plan_item.poi_id)
                for day in candidate.days
                for plan_item in day.items
                if plan_item.poi_id is not None
                and not any(
                    required.casefold() in plan_item.name.casefold()
                    for required in hangzhou_trip.must_visit
                )
            )
            violation = Violation(
                type="activity_time_limit",
                severity=ViolationSeverity.ERROR,
                message="force the dynamic repair branch",
                day=affected_day,
                entity_ids=[poi_id],
            )
            invalid = candidate.model_copy(
                update={"validation": ValidationResult.from_violations([violation])}
            )
            draft = next(
                item
                for item in solved.snapshot.candidate_drafts
                if item.id == candidate.id
            )
            snapshot = replace(
                solved.snapshot,
                candidates=(invalid,),
                candidate_drafts=(draft,),
            )
            return PlanningKernelSolveResult(
                snapshot=snapshot,
                selected_plan=None,
                degraded_reasons=solved.degraded_reasons,
            )

    registry = default_agent_tool_registry()
    defaults = POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING)
    workflow = build_agentic_workflow(
        planning_kernel=InvalidFirstKernel(
            gateway=workflow_harness.gateway,
            defaults=defaults,
        ),
        planner_gateway=PlannerGateway(model=MockPlannerModel()),
        replanner_gateway=ReplannerGateway(model=MockReplannerModel()),
        tool_executor=AgentToolExecutor(
            registry=registry,
            gateway=workflow_harness.gateway,
            defaults=defaults,
        ),
        evidence_repository=InMemoryEvidenceRepository(),
    )

    response = await run_agentic_planning(
        workflow,
        PlanningRequest(trip=hangzhou_trip),
        thread_id="v1-3-dynamic-model-repair",
    )

    assert response.status == "completed"
    assert response.selected_plan is not None
    assert response.selected_plan.validation is not None
    assert response.selected_plan.validation.valid is True
    assert response.selected_plan.id.endswith("repair-r1")
    assert response.iterations == 1
    assert response.action_summary == (
        "call_tool",
        "call_tool",
        "solve",
        "repair",
        "finish",
    )


@pytest.mark.asyncio
async def test_dynamic_agent_runs_grounded_critic_and_accepts_soft_repair(
    workflow_harness, hangzhou_trip
):
    class ImprovingCritic:
        name = "dynamic-soft-repair-fixture"
        model = "dynamic-soft-repair-v1"
        prompt_version = "test-v1"

        async def critique(self, request):
            critiques = []
            for digest in request.digests:
                optional = next(
                    (
                        item
                        for item in digest.evidence
                        if item.field == "must_visit"
                        and item.value is False
                        and item.day is not None
                    ),
                    None,
                )
                reference = optional.id if optional else digest.evidence[0].id
                repaired = "-soft-r" in digest.candidate_id
                dimensions = []
                for dimension in SoftDimension:
                    suggestion = None
                    if not repaired and dimension is SoftDimension.PACE and optional:
                        suggestion = SuggestedSoftAction(
                            kind=SuggestedActionKind.REMOVE_OPTIONAL_POI,
                            poi_id=optional.entity_id,
                            from_day=optional.day,
                            evidence_ids=(reference,),
                            expected_dimension=dimension,
                        )
                    dimensions.append(
                        DimensionCritique(
                            dimension=dimension,
                            score=90 if repaired else 50,
                            summary="动态软质量测试",
                            evidence_ids=(reference,),
                            suggested_action=suggestion,
                        )
                    )
                critiques.append(
                    SoftCritique(
                        candidate_id=digest.candidate_id,
                        dimensions=tuple(dimensions),
                        overall_summary="动态 Critic 测试",
                        tradeoff_evidence_ids=(reference,),
                    )
                )
            return SoftCriticProviderOutput(critiques=tuple(critiques))

    defaults = POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING)
    workflow = build_agentic_workflow(
        planning_kernel=PlanningKernelService(
            gateway=workflow_harness.gateway,
            defaults=defaults,
        ),
        planner_gateway=PlannerGateway(model=MockPlannerModel()),
        tool_executor=AgentToolExecutor(
            registry=default_agent_tool_registry(),
            gateway=workflow_harness.gateway,
            defaults=defaults,
        ),
        evidence_repository=InMemoryEvidenceRepository(),
        quality_service=QualityReviewService(
            critic_gateway=CriticGateway(
                model=ImprovingCritic(),
                timeout_seconds=1,
                max_attempts=1,
                base_delay_seconds=0,
                max_delay_seconds=0,
            ),
            policy=CriticPolicy(),
        ),
    )

    response = await run_agentic_planning(
        workflow,
        PlanningRequest(trip=hangzhou_trip),
        thread_id="v1-3-dynamic-soft-repair",
    )

    assert response.status == "completed"
    assert response.critic_status is CriticStatus.SUCCESS
    assert response.soft_iterations == 1
    assert response.selected_plan is not None
    assert "-soft-r1" in response.selected_plan.id
    assert response.selected_plan.validation is not None
    assert response.selected_plan.validation.valid is True
    assert response.grounded_explanation is not None
    assert response.evidence_summary["quality"] == 1


@pytest.mark.asyncio
async def test_runtime_feature_flag_selects_dynamic_graph_and_records_decision_trace(
    hangzhou_trip,
):
    runtime = await PlanningRuntime.create(
        Settings(agent_mode=AgentMode.DYNAMIC_PLANNER)
    )
    try:
        result = await runtime.execute_plan(
            PlanningRequest(trip=hangzhou_trip),
            thread_id="v1-3-runtime-dynamic",
        )
        assert result.payload.status == "completed"
        assert result.payload.agent_mode == "dynamic_planner"
        assert result.run is not None
        trace = await runtime.get_agent_trace(result.run.run_id, limit=500)
    finally:
        await runtime.close()

    event_types = {item.event_type for item in trace}
    assert TraceEventType.AGENT_DECISION_STARTED in event_types
    assert TraceEventType.AGENT_ACTION_DISPATCHED in event_types
    assert TraceEventType.AGENT_OBSERVATION_RECORDED in event_types
    assert TraceEventType.FINAL_GUARD_COMPLETED in event_types


@pytest.mark.asyncio
async def test_lifecycle_creation_reuses_dynamic_runner_snapshot(hangzhou_trip):
    runtime = await PlanningRuntime.create(Settings())
    try:
        result = await runtime.execute_create_plan_session(
            PlanningRequest(trip=hangzhou_trip),
            session_id="dynamic-lifecycle-session",
        )
        assert result.run is not None
        session = await runtime.plan_repository.get("dynamic-lifecycle-session")
        trace = await runtime.get_agent_trace(result.run.run_id, limit=500)
    finally:
        await runtime.close()

    assert runtime.agent_mode is AgentMode.DYNAMIC_PLANNER
    assert session.snapshot is not None
    assert session.snapshot.recommended_candidate_id in {
        item.id for item in session.snapshot.candidates
    }
    assert session.snapshot.critic_status is CriticStatus.SUCCESS
    assert any(
        item.event_type is TraceEventType.NODE_STARTED and item.graph == "agentic"
        for item in trace
    )
    assert not any(
        item.event_type is TraceEventType.NODE_STARTED
        and item.node == "build_search_plan"
        for item in trace
    )


@pytest.mark.asyncio
async def test_shadow_mode_records_one_decision_without_dispatching_shadow_tool(hangzhou_trip):
    runtime = await PlanningRuntime.create(
        Settings(agent_mode=AgentMode.SHADOW_DYNAMIC_PLANNER)
    )
    try:
        result = await runtime.execute_plan(
            PlanningRequest(trip=hangzhou_trip), thread_id="v1-3-shadow"
        )
        trace = await runtime.get_agent_trace(result.run.run_id, limit=500)
    finally:
        await runtime.close()

    assert result.payload.status == "completed"
    assert result.payload.agent_mode == "shadow_dynamic_planner"
    assert result.payload.decision_count == 1
    assert result.payload.action_summary == ("call_tool",)
    assert not any(
        item.event_type is TraceEventType.AGENT_ACTION_DISPATCHED for item in trace
    )
