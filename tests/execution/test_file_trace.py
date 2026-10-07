from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from travel_agent.config import Settings
from travel_agent.domain.models import PlanningRequest
from travel_agent.execution.context import current_run_context
from travel_agent.execution.coordinator import RunCoordinator
from travel_agent.execution.file_trace import FileTraceSink
from travel_agent.execution.file_trace import bounded_action_details, MAX_ACTION_IO_CHARS
from travel_agent.execution.models import ExecutionBudget, RunKind, TraceStatus, TraceEventType
from travel_agent.execution.repository import InMemoryRunRepository
from travel_agent.runtime import PlanningRuntime


def read_events(path: Path):
    return [record for record in read_records(path) if record.get("record_type") != "action_io"]


def read_records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_file_trace_is_live_ordered_and_sanitized(tmp_path):
    repository = InMemoryRunRepository()
    coordinator = RunCoordinator(repository, ExecutionBudget(), trace_file_dir=str(tmp_path))
    release = asyncio.Event()
    path = tmp_path / "live-run.log"

    async def call():
        context = current_run_context()
        context.trace.record(
            TraceEventType.NODE_STARTED, status="running", node="规划节点",
            attributes={"reason": "中文原因", "api_key": "do-not-write", "prompt": "secret-prompt"},
        )
        await release.wait()
        return None

    task = asyncio.create_task(coordinator.execute(RunKind.STRUCTURED_PLAN, call, run_id="live-run"))
    try:
        for _ in range(100):
            if path.exists() and len(path.read_text(encoding="utf-8").splitlines()) >= 2:
                break
            await asyncio.sleep(0.01)
        assert not task.done()
        events = read_events(path)
        assert [event["sequence"] for event in events] == [1, 2]
        assert events[1]["attributes"] == {"reason": "中文原因"}
        assert "do-not-write" not in path.read_text(encoding="utf-8")
        assert "secret-prompt" not in path.read_text(encoding="utf-8")
    finally:
        release.set()
        await task
    events = read_events(path)
    assert events[-1]["event_type"] == "run.completed"
    assert events == [event.model_dump(mode="json") for event in await repository.trace("live-run")]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_and_cancelled_runs_flush_terminal_event(tmp_path, cancelled):
    repository = InMemoryRunRepository()
    coordinator = RunCoordinator(repository, ExecutionBudget(), trace_file_dir=str(tmp_path))
    error_type = asyncio.CancelledError if cancelled else ValueError

    async def fail():
        raise error_type("private exception text")

    with pytest.raises(error_type):
        await coordinator.execute(RunKind.STRUCTURED_PLAN, fail, run_id="failed-run")
    events = read_events(tmp_path / "failed-run.log")
    assert events[-1]["event_type"] == ("run.cancelled" if cancelled else "run.failed")
    assert "private exception text" not in (tmp_path / "failed-run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_parallel_runs_have_separate_files(tmp_path):
    coordinator = RunCoordinator(InMemoryRunRepository(), ExecutionBudget(), trace_file_dir=str(tmp_path))

    async def call():
        await asyncio.sleep(0)

    await asyncio.gather(*(
        coordinator.execute(RunKind.STRUCTURED_PLAN, call, run_id=run_id)
        for run_id in ("first", "second")
    ))
    for run_id in ("first", "second"):
        events = read_events(tmp_path / f"{run_id}.log")
        assert {event["run_id"] for event in events} == {run_id}
        assert [event["sequence"] for event in events] == [1, 2]


@pytest.mark.asyncio
async def test_file_failure_does_not_abort_run_or_database_trace(tmp_path, caplog):
    blocked_directory = tmp_path / "blocked"
    blocked_directory.write_text("not a directory", encoding="utf-8")
    repository = InMemoryRunRepository()
    coordinator = RunCoordinator(repository, ExecutionBudget(), trace_file_dir=str(blocked_directory))

    async def call():
        return None

    result = await coordinator.execute(RunKind.STRUCTURED_PLAN, call, run_id="file-failure")
    assert result.run.status.value == "completed"
    assert result.run.trace_status is TraceStatus.DEGRADED
    assert (await repository.trace("file-failure"))[-1].event_type is TraceEventType.RUN_COMPLETED
    assert "agent_run.file_trace_write_failed" in caplog.text


@pytest.mark.asyncio
async def test_database_append_failure_does_not_skip_text_output(tmp_path):
    class BrokenLiveRepository(InMemoryRunRepository):
        async def append_trace_event(self, event):
            raise OSError("append unavailable")

    coordinator = RunCoordinator(BrokenLiveRepository(), ExecutionBudget(), trace_file_dir=str(tmp_path))

    async def call():
        return None

    result = await coordinator.execute(RunKind.STRUCTURED_PLAN, call, run_id="database-failure")
    assert result.run.trace_status is TraceStatus.DEGRADED
    assert [event["sequence"] for event in read_events(tmp_path / "database-failure.log")] == [1, 2]


def test_file_trace_settings_and_path_confinement(tmp_path):
    settings = Settings.from_env({})
    assert settings.trace_file_enabled
    assert settings.trace_file_dir == ".data/traces"
    settings = Settings.from_env({"TRACE_FILE_ENABLED": "false", "TRACE_FILE_DIR": ""})
    assert not settings.trace_file_enabled
    with pytest.raises(ValueError, match="TRACE_FILE_DIR"):
        Settings.from_env({"TRACE_FILE_DIR": " "})
    sink = FileTraceSink(tmp_path, "../../outside")
    assert sink.path.parent == tmp_path
    assert len(sink.path.stem) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_mock_planning_runtime_obeys_file_trace_setting(tmp_path, hangzhou_trip, enabled):
    runtime = await PlanningRuntime.create(Settings.from_env({
        "TRACE_FILE_ENABLED": "true" if enabled else "false",
        "TRACE_FILE_DIR": str(tmp_path),
    }))
    try:
        result = await runtime.execute_plan(
            PlanningRequest(trip=hangzhou_trip), thread_id="file-trace-demo", run_id="planning-demo"
        )
        assert result.payload.status == "completed"
        path = tmp_path / "planning-demo.log"
        assert path.exists() == enabled
        if enabled:
            events = read_events(path)
            assert events[0]["event_type"] == "run.started"
            assert events[-1]["event_type"] == "run.completed"
            assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
            assert any(event["event_type"].startswith("tool.") for event in events)
            details = [record for record in read_records(path) if record.get("record_type") == "action_io"]
            searches = [record for record in details if record["operation"] == "poi.search" and "result" in record["details"]]
            assert searches
            search = searches[0]
            assert search["details"]["request"]["city"] == "杭州"
            assert search["details"]["request"]["keyword"]
            assert search["details"]["result"]["data"][0]["name"]
            by_id = {event["event_id"]: event for event in events}
            assert by_id[search["trace_event_id"]]["sequence"] == search["trace_sequence"]
            assert search["parent_event_id"] in by_id
            assert all("details" not in event for event in await runtime.run_repository.trace("planning-demo", limit=500))
    finally:
        await runtime.close()


def test_debug_details_redact_nested_secrets_and_bound_large_results():
    safe = bounded_action_details({"request": {
        "apiKey": "private-key", "Authorization": "Bearer private-token",
        "nested": {"approval_token": "private-approval", "user_id": "private-user"},
        "url": "https://example.test/search?key=private-url-key&keyword=杭州",
    }, "result": [{"name": "灵隐寺", "password": "private-password"}]})
    text = json.dumps(safe, ensure_ascii=False)
    assert "private-" not in text
    assert "灵隐寺" in text and "杭州" in text
    assert safe["request"]["apiKey"] == "[REDACTED]"
    large = bounded_action_details({"result": "x" * (MAX_ACTION_IO_CHARS + 10)})
    assert large["truncated"]
    assert len(large["preview"]) == MAX_ACTION_IO_CHARS


@pytest.mark.asyncio
async def test_failed_query_and_cache_hit_have_correlated_results(tmp_path, workflow_harness, monkeypatch):
    from travel_agent.domain.tool_models import POISearchQuery, ToolCallContext, ToolErrorCategory
    from travel_agent.tools.errors import ToolProviderError

    gateway = workflow_harness.gateway
    repository = InMemoryRunRepository()
    coordinator = RunCoordinator(repository, ExecutionBudget(), trace_file_dir=str(tmp_path))
    query = POISearchQuery(city="杭州", keyword="灵隐寺")

    async def call():
        return await gateway.search_pois([query], ToolCallContext(thread_id="query-debug"))

    await coordinator.execute(RunKind.STRUCTURED_PLAN, call, run_id="first-query")
    await coordinator.execute(RunKind.STRUCTURED_PLAN, call, run_id="cached-query")
    cached = next(record for record in read_records(tmp_path / "cached-query.log")
                  if record.get("record_type") == "action_io" and "result" in record["details"])
    assert cached["details"]["result"]["cache_hit"]
    assert cached["details"]["result"]["data"]

    async def fail(_query):
        raise ToolProviderError(category=ToolErrorCategory.INVALID_REQUEST, code="test_failure",
                                operation="poi.search", retryable=False, safe_message="查询失败")

    monkeypatch.setattr(workflow_harness.poi_provider, "search_pois", fail)
    query = POISearchQuery(city="杭州", keyword="未缓存查询")
    await coordinator.execute(RunKind.STRUCTURED_PLAN, call, run_id="failed-query")
    failed = next(record for record in read_records(tmp_path / "failed-query.log")
                  if record.get("record_type") == "action_io" and "result" in record["details"])
    assert failed["details"]["request"]["keyword"] == "未缓存查询"
    assert failed["details"]["result"]["status"] == "failed"
    assert failed["details"]["result"]["error"]["code"] == "test_failure"


@pytest.mark.asyncio
async def test_bad_debug_serialization_does_not_abort_planning(tmp_path, caplog):
    from pydantic import BaseModel

    class BrokenDebugModel(BaseModel):
        def model_dump(self, **kwargs):
            raise ValueError("private-serialization-error")

    coordinator = RunCoordinator(InMemoryRunRepository(), ExecutionBudget(), trace_file_dir=str(tmp_path))

    async def call():
        context = current_run_context()
        context.trace.record(TraceEventType.NODE_COMPLETED, status="completed", node="debug-test",
                             file_details={"result": BrokenDebugModel()})

    result = await coordinator.execute(RunKind.STRUCTURED_PLAN, call, run_id="bad-debug")
    assert result.run.status.value == "completed"
    assert result.run.trace_status is TraceStatus.DEGRADED
    assert "trace_detail_serialization_failure" in result.run.degraded_reasons
    assert read_events(tmp_path / "bad-debug.log")[-1]["event_type"] == "run.completed"
    assert "private-serialization-error" not in caplog.text
