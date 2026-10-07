from pathlib import Path

import pytest

from travel_agent.config import Settings
from travel_agent.domain.models import PlanningRequest, TripSpec
from travel_agent.execution.budget import ExecutionLedger
from travel_agent.execution.checkpoints import CheckpointInjectedError
from travel_agent.execution.coordinator import RunCoordinator
from travel_agent.execution.faults import FaultMode, FaultPlan, FaultPoint, FaultRule
from travel_agent.execution.models import (
    ExecutionBudget,
    RunKind,
    RunTerminalReason,
    TraceEventType,
    TraceStatus,
)
from travel_agent.execution.repository import InMemoryRunRepository
from travel_agent.execution.tracing import TraceRecorder
from travel_agent.lifecycle.repository import PlanRepositoryInjectedError
from travel_agent.requirements.models import NaturalPlanningRequest
from travel_agent.runtime import PlanningRuntime


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_trace_sink_fault_degrades_observability_without_losing_domain_result():
    repository = InMemoryRunRepository()
    coordinator = RunCoordinator(repository, ExecutionBudget())
    result = await coordinator.execute(
        RunKind.STRUCTURED_PLAN,
        lambda: _completed_payload(),
        fault_plan=FaultPlan(
            rules=(
                FaultRule(
                    point=FaultPoint.TRACE_SINK,
                    mode=FaultMode.WRITE_FAILURE,
                ),
            )
        ),
    )

    assert result.payload.status == "completed"
    assert result.run is not None
    assert result.run.trace_status is TraceStatus.DEGRADED
    assert "trace_sink_failure" in result.run.degraded_reasons


@pytest.mark.asyncio
async def test_checkpoint_fault_is_classified_separately_from_business_infeasible():
    runtime = await PlanningRuntime.create(Settings.from_env({}))
    thread_id = "fault-checkpoint"
    try:
        with pytest.raises(CheckpointInjectedError) as raised:
            await runtime.execute_plan_from_text(
                NaturalPlanningRequest(
                    text="2026年10月2日到10月4日去杭州，3个人。",
                    reference_date="2026-08-23",
                ),
                thread_id=thread_id,
                fault_plan=FaultPlan(
                    rules=(
                        FaultRule(
                            point=FaultPoint.CHECKPOINT_READ,
                            mode=FaultMode.CONNECTION_ERROR,
                            operation="aget_tuple",
                        ),
                    )
                ),
            )
        runs = await runtime.get_thread_runs(thread_id, limit=1)
        trace = await runtime.get_agent_trace(runs[0].run_id)
        assert raised.value.agent_run_id == runs[0].run_id
        assert runs[0].terminal_reason is RunTerminalReason.CHECKPOINT_FAILURE
        assert any(
            event.event_type is TraceEventType.CHECKPOINT_FAILED for event in trace
        )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_plan_repository_fault_has_repository_terminal_reason():
    runtime = await PlanningRuntime.create(Settings.from_env({}))
    trip = TripSpec.model_validate_json(
        (ROOT / "evals" / "repairs" / "base_trip.json").read_text(
            encoding="utf-8"
        )
    )
    session_id = "fault-plan-repository"
    try:
        with pytest.raises(PlanRepositoryInjectedError):
            await runtime.execute_create_plan_session(
                PlanningRequest(trip=trip),
                session_id=session_id,
                fault_plan=FaultPlan(
                    rules=(
                        FaultRule(
                            point=FaultPoint.PLAN_REPOSITORY,
                            mode=FaultMode.WRITE_FAILURE,
                            operation="create",
                        ),
                    )
                ),
            )
        runs = await runtime.get_session_runs(session_id, limit=1)
        assert runs[0].terminal_reason is RunTerminalReason.REPOSITORY_FAILURE
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_empty_poi_result_is_business_outcome_not_provider_failure():
    # 该断言验证旧固定基线的业务失败分类，而不是生产默认模式。
    runtime = await PlanningRuntime.create(
        Settings.from_env({"AGENT_MODE": "fixed_workflow"})
    )
    trip = TripSpec.model_validate_json(
        (ROOT / "evals" / "repairs" / "base_trip.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        result = await runtime.execute_plan(
            PlanningRequest(trip=trip),
            thread_id="fault-empty-poi",
            fault_plan=FaultPlan(
                rules=(
                    FaultRule(
                        point=FaultPoint.POI_PROVIDER,
                        mode=FaultMode.EMPTY_BUSINESS_RESULT,
                        times=100,
                    ),
                )
            ),
        )
        assert result.run is not None
        assert result.run.terminal_reason is RunTerminalReason.BUSINESS_INFEASIBLE
    finally:
        await runtime.close()


def test_trace_recorder_drops_unknown_secret_attributes_and_truncates_values():
    ledger = ExecutionLedger("safe-trace", ExecutionBudget())
    recorder = TraceRecorder("safe-trace", ledger, attribute_max_chars=8)
    recorder.record(
        TraceEventType.TOOL_STARTED,
        status="started",
        attributes={
            "provider": "provider-name",
            "api_key": "secret-value",
            "prompt": "raw prompt",
        },
    )

    assert recorder.events[0].attributes == {"provider": "provider"}


class _Payload:
    status = "completed"


async def _completed_payload() -> _Payload:
    return _Payload()
