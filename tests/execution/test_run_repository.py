from __future__ import annotations

from datetime import datetime, timezone

import pytest

from travel_agent.execution.models import (
    AgentRunRecord,
    ExecutionBudget,
    RunKind,
    RunStatus,
    TraceEvent,
    TraceEventType,
)
from travel_agent.execution.repository import SQLiteRunRepository


@pytest.mark.asyncio
async def test_sqlite_run_repository_survives_reopen(tmp_path):
    path = tmp_path / "runs.sqlite3"
    first = SQLiteRunRepository(str(path))
    run = AgentRunRecord(
        run_id="sqlite-run",
        run_kind=RunKind.STRUCTURED_PLAN,
        status=RunStatus.RUNNING,
        thread_id="thread-sqlite",
        budget=ExecutionBudget(),
        started_at=datetime.now(timezone.utc),
        config_fingerprint="config",
    )
    await first.create(run)
    event = TraceEvent(
        event_id="event-1",
        run_id=run.run_id,
        sequence=1,
        event_type=TraceEventType.RUN_STARTED,
        timestamp=datetime.now(timezone.utc),
        monotonic_offset_ms=0,
        status="running",
    )
    await first.append_trace_event(event)
    assert (await first.trace(run.run_id))[0].event_id == "event-1"
    completed = run.model_copy(update={"status": RunStatus.COMPLETED})
    await first.finalize(completed, (event,))
    await first.close()

    second = SQLiteRunRepository(str(path))
    restored = await second.get("sqlite-run")
    listed = await second.list_for_thread("thread-sqlite")
    restored_events = await second.trace("sqlite-run")
    await second.close()

    assert restored.status is RunStatus.COMPLETED
    assert listed[0].run_id == "sqlite-run"
    assert restored_events[0].event_id == "event-1"
