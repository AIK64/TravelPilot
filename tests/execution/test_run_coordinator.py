from __future__ import annotations

import asyncio

from pydantic import BaseModel
import pytest

from travel_agent.execution.context import current_run_context
from travel_agent.execution.coordinator import RunCoordinator
from travel_agent.execution.models import (
    ExecutionBudget,
    RunKind,
    RunStatus,
    RunTerminalReason,
    TraceEventType,
)
from travel_agent.execution.repository import InMemoryRunRepository


class Payload(BaseModel):
    status: str
    interrupt: dict | None = None


@pytest.mark.asyncio
async def test_coordinator_persists_completed_run_and_trace():
    repository = InMemoryRunRepository()
    coordinator = RunCoordinator(repository, ExecutionBudget())

    result = await coordinator.execute(
        RunKind.STRUCTURED_PLAN,
        lambda: _payload("completed"),
        thread_id="thread-1",
    )

    assert result.run is not None
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.terminal_reason is RunTerminalReason.PLAN_COMPLETED
    events = await repository.trace(result.run.run_id)
    assert events[0].event_type is TraceEventType.RUN_STARTED
    assert events[-1].event_type is TraceEventType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_coordinator_persists_trace_before_run_completes():
    repository = InMemoryRunRepository()
    coordinator = RunCoordinator(repository, ExecutionBudget())
    node_recorded = asyncio.Event()
    release = asyncio.Event()

    async def blocked_payload() -> Payload:
        context = current_run_context()
        assert context is not None
        context.trace.record(
            TraceEventType.NODE_STARTED,
            status="running",
            graph="planning",
            node="generate_candidates",
        )
        node_recorded.set()
        await release.wait()
        return Payload(status="completed")

    execution = asyncio.create_task(
        coordinator.execute(
            RunKind.STRUCTURED_PLAN,
            blocked_payload,
            thread_id="thread-live",
            run_id="run-live",
        )
    )
    await asyncio.wait_for(node_recorded.wait(), timeout=1)

    live_events = ()
    for _ in range(20):
        live_events = await repository.trace("run-live")
        if len(live_events) >= 2:
            break
        await asyncio.sleep(0)

    assert not execution.done()
    assert [event.event_type for event in live_events[:2]] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.NODE_STARTED,
    ]

    release.set()
    await execution


@pytest.mark.asyncio
async def test_coordinator_marks_interrupt_and_idempotent_replay():
    repository = InMemoryRunRepository()
    coordinator = RunCoordinator(repository, ExecutionBudget())
    first = await coordinator.execute(
        RunKind.CLARIFICATION_RESUME,
        lambda: _payload("needs_clarification", interrupt={"id": "i1"}),
        thread_id="thread-2",
        request_id="request-1",
    )
    second = await coordinator.execute(
        RunKind.CLARIFICATION_RESUME,
        lambda: _payload("needs_clarification", interrupt={"id": "i1"}),
        thread_id="thread-2",
        request_id="request-1",
    )

    assert first.run is not None and second.run is not None
    assert first.run.status is RunStatus.INTERRUPTED
    assert second.run.status is RunStatus.REPLAYED
    assert second.run.replay_of_run_id == first.run.run_id


async def _payload(status: str, interrupt: dict | None = None) -> Payload:
    return Payload(status=status, interrupt=interrupt)
