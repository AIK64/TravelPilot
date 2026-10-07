from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Generic, TypeVar
from uuid import uuid4
import asyncio

from travel_agent.execution.budget import ExecutionLedger
from travel_agent.execution.context import RunContext, bind_run_context
from travel_agent.execution.errors import ExecutionBudgetExceeded
from travel_agent.execution.faults import FaultInjector, FaultPlan
from travel_agent.execution.faults import FaultMode, FaultPoint
from travel_agent.execution.models import (
    AgentRunRecord,
    ExecutionBudget,
    RunKind,
    RunStatus,
    RunTerminalReason,
    TraceEventType,
    TraceEvent,
    safe_status_value,
)
from travel_agent.execution.repository import RunRepository
from travel_agent.execution.file_trace import FileTraceSink
from travel_agent.execution.tracing import TraceRecorder


T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionResult(Generic[T]):
    payload: T
    run: AgentRunRecord | None


@dataclass(frozen=True, slots=True)
class _FileTraceDetails:
    event: TraceEvent
    details: dict[str, object]


class _LiveTraceWriter:
    """按 sequence 串行追加事件，避免同步的 Node 埋点阻塞 Graph。"""

    def __init__(self, repository: RunRepository, run_id: str, trace_file_dir: str | None = None) -> None:
        self.repository = repository
        self.run_id = run_id
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._stop = object()
        self._failed = False
        self._file_sink = FileTraceSink(trace_file_dir, run_id) if trace_file_dir else None
        self._task = asyncio.create_task(
            self._write_events(), name=f"live-trace:{run_id}"
        )

    @property
    def failed(self) -> bool:
        return self._failed

    def publish(self, event) -> None:
        self._queue.put_nowait(event)

    def publish_details(self, event: TraceEvent, details: dict[str, object]) -> None:
        if self._file_sink is not None:
            self._queue.put_nowait(_FileTraceDetails(event, details))

    async def close(self) -> None:
        self._queue.put_nowait(self._stop)
        await self._task

    async def _write_events(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is self._stop:
                    return
                if not isinstance(event, _FileTraceDetails):
                    try:
                        await self.repository.append_trace_event(event)
                    except Exception:
                        self._failed = True
                        logger.exception(
                            "agent_run.live_trace_write_failed | run_id=%s", self.run_id
                        )
                # 两个输出独立失败；文件错误不终止规划或影响后续数据库事件。
                if self._file_sink is not None:
                    try:
                        if isinstance(event, _FileTraceDetails):
                            await asyncio.to_thread(self._file_sink.append_details, event.event, event.details)
                        else:
                            await asyncio.to_thread(self._file_sink.append, event)
                    except Exception:
                        self._failed = True
                        logger.exception(
                            "agent_run.file_trace_write_failed | run_id=%s", self.run_id
                        )
                        self._file_sink = None
            finally:
                self._queue.task_done()


class RunCoordinator:
    def __init__(
        self,
        repository: RunRepository,
        budget: ExecutionBudget,
        *,
        trace_attribute_max_chars: int = 256,
        trace_file_dir: str | None = None,
        config_values: dict[str, object] | None = None,
    ) -> None:
        self.repository = repository
        self.budget = budget
        self.trace_attribute_max_chars = trace_attribute_max_chars
        self.trace_file_dir = trace_file_dir
        encoded = json.dumps(
            config_values or {"budget": budget.model_dump(mode="json")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.config_fingerprint = hashlib.sha256(encoded).hexdigest()[:24]

    async def execute(
        self,
        kind: RunKind,
        call: Callable[[], Awaitable[T]],
        *,
        thread_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        parent_run_id: str | None = None,
        causation_id: str | None = None,
        fault_plan: FaultPlan | None = None,
        budget: ExecutionBudget | None = None,
        run_id: str | None = None,
        tenant_id: str = "local",
        user_id: str = "demo",
        precreated: bool = False,
    ) -> ExecutionResult[T]:
        run_id = run_id or str(uuid4())
        active_budget = budget or self.budget
        previous = None
        if request_id:
            previous = await self.repository.find_request(
                request_id=request_id,
                session_id=session_id,
                thread_id=thread_id,
            )
        if precreated:
            record = await self.repository.get(run_id)
            if record.status is not RunStatus.RUNNING:
                raise ValueError("precreated run is not in running status")
            active_budget = record.budget
        else:
            record = self.new_record(
                kind,
                run_id=run_id,
                thread_id=thread_id,
                session_id=session_id,
                request_id=request_id,
                parent_run_id=parent_run_id,
                replay_of_run_id=previous.run_id if previous else None,
                causation_id=causation_id,
                budget=active_budget,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            await self.repository.create(record)
        ledger = ExecutionLedger(run_id, active_budget)
        live_trace = _LiveTraceWriter(self.repository, run_id, self.trace_file_dir)
        trace = TraceRecorder(
            run_id,
            ledger,
            attribute_max_chars=self.trace_attribute_max_chars,
            event_sink=live_trace.publish,
            file_detail_sink=live_trace.publish_details if self.trace_file_dir else None,
        )
        context = RunContext(record, ledger, trace, FaultInjector(fault_plan))
        trace.record(
            TraceEventType.RUN_STARTED,
            status="running",
            terminal=True,
            attributes={"run_kind": kind.value},
        )
        try:
            with bind_run_context(context):
                payload = await call()
            if previous is not None:
                status = RunStatus.REPLAYED
                reason = RunTerminalReason.IDEMPOTENT_REPLAY
                trace.record(
                    TraceEventType.RUN_REPLAYED,
                    status=status.value,
                    terminal=True,
                    attributes={"replay_of_run_id": previous.run_id},
                )
            else:
                status, reason = classify_payload(payload)
                trace.record(
                    TraceEventType.RUN_INTERRUPTED
                    if status is RunStatus.INTERRUPTED
                    else TraceEventType.RUN_FAILED
                    if status is RunStatus.FAILED
                    else TraceEventType.RUN_COMPLETED,
                    status=status.value,
                    terminal=True,
                    attributes={"terminal_reason": reason.value},
                )
        except BaseException as error:
            reason = classify_error(error)
            try:
                setattr(error, "agent_run_id", run_id)
            except (AttributeError, TypeError):
                pass
            if isinstance(error, ExecutionBudgetExceeded):
                trace.record(
                    TraceEventType.BUDGET_EXCEEDED,
                    status="failed",
                    terminal=True,
                    attributes={
                        "terminal_reason": reason.value,
                        "error_type": type(error).__name__,
                        "limit": error.limit,
                    },
                )
            cancelled = isinstance(error, asyncio.CancelledError)
            trace.record(
                TraceEventType.RUN_CANCELLED if cancelled else TraceEventType.RUN_FAILED,
                status="cancelled" if cancelled else "failed",
                terminal=True,
                attributes={"terminal_reason": reason.value},
            )
            await live_trace.close()
            if live_trace.failed:
                trace.add_degradation("live_trace_sink_failure")
                trace.mark_degraded()
            failed = self._final_record(
                record,
                ledger=ledger,
                trace=trace,
                status=RunStatus.CANCELLED if cancelled else RunStatus.FAILED,
                reason=reason,
            )
            await self._persist(failed, trace)
            raise
        trace_fault = context.faults.match(
            FaultPoint.TRACE_SINK, operation="finalize", attempt=1
        )
        if trace_fault in {
            FaultMode.WRITE_FAILURE,
            FaultMode.CONNECTION_ERROR,
            FaultMode.TIMEOUT,
        }:
            trace.add_degradation("trace_sink_failure")
            trace.mark_degraded()
        await live_trace.close()
        if live_trace.failed:
            trace.add_degradation("live_trace_sink_failure")
            trace.mark_degraded()
        completed = self._final_record(
            record,
            ledger=ledger,
            trace=trace,
            status=status,
            reason=reason,
            plan_version_id=_plan_version_id(payload),
        )
        await self._persist(completed, trace)
        logger.info(
            "agent_run.completed | run_id=%s kind=%s status=%s reason=%s steps=%s "
            "tool_calls=%s llm_calls=%s elapsed_ms=%s",
            completed.run_id,
            completed.run_kind.value,
            completed.status.value,
            completed.terminal_reason.value if completed.terminal_reason else None,
            completed.usage.graph_steps,
            completed.usage.tool_calls,
            completed.usage.llm_calls,
            completed.elapsed_ms,
        )
        return ExecutionResult(payload=payload, run=completed)

    def new_record(
        self,
        kind: RunKind,
        *,
        run_id: str,
        thread_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        parent_run_id: str | None = None,
        replay_of_run_id: str | None = None,
        causation_id: str | None = None,
        budget: ExecutionBudget | None = None,
        tenant_id: str = "local",
        user_id: str = "demo",
    ) -> AgentRunRecord:
        return AgentRunRecord(
            run_id=run_id,
            run_kind=kind,
            status=RunStatus.RUNNING,
            thread_id=thread_id,
            session_id=session_id,
            request_id=request_id,
            parent_run_id=parent_run_id,
            replay_of_run_id=replay_of_run_id,
            causation_id=causation_id,
            budget=budget or self.budget,
            started_at=datetime.now(timezone.utc),
            config_fingerprint=self.config_fingerprint,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    async def reserve(
        self,
        kind: RunKind,
        *,
        run_id: str,
        thread_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        parent_run_id: str | None = None,
        causation_id: str | None = None,
        budget: ExecutionBudget | None = None,
        tenant_id: str = "local",
        user_id: str = "demo",
    ) -> AgentRunRecord:
        record = self.new_record(
            kind,
            run_id=run_id,
            thread_id=thread_id,
            session_id=session_id,
            request_id=request_id,
            parent_run_id=parent_run_id,
            causation_id=causation_id,
            budget=budget,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        await self.repository.create(record)
        return record

    def _final_record(
        self,
        record: AgentRunRecord,
        *,
        ledger: ExecutionLedger,
        trace: TraceRecorder,
        status: RunStatus,
        reason: RunTerminalReason,
        plan_version_id: str | None = None,
    ) -> AgentRunRecord:
        ended = datetime.now(timezone.utc)
        return record.model_copy(
            update={
                "status": status,
                "terminal_reason": reason,
                "usage": ledger.snapshot(),
                "trace_status": trace.status,
                "degraded_reasons": tuple(trace.degraded_reasons),
                "ended_at": ended,
                "elapsed_ms": max(
                    0, round((ended - record.started_at).total_seconds() * 1000)
                ),
                "plan_version_id": plan_version_id,
            }
        )

    async def _persist(self, record: AgentRunRecord, trace: TraceRecorder) -> None:
        try:
            await self.repository.finalize(record, trace.events)
        except BaseException:
            trace.mark_degraded()
            logger.exception("agent_run.persistence_failed | run_id=%s", record.run_id)
            raise


def classify_payload(payload: object) -> tuple[RunStatus, RunTerminalReason]:
    status = safe_status_value(getattr(payload, "status", None))
    if status == "failed":
        planning = getattr(payload, "planning", None) or payload
        reason = getattr(planning, "terminal_reason", None)
        try:
            terminal_reason = RunTerminalReason(reason)
        except (ValueError, TypeError):
            terminal_reason = RunTerminalReason.INVALID_AGENT_DECISION
        return RunStatus.FAILED, terminal_reason
    interrupt = getattr(payload, "interrupt", None)
    if status in {"needs_clarification", "needs_requirement_clarification"}:
        return RunStatus.INTERRUPTED, RunTerminalReason.NEEDS_CLARIFICATION
    if status == "awaiting_candidate_selection":
        return RunStatus.INTERRUPTED, RunTerminalReason.AWAITING_CANDIDATE_SELECTION
    if status == "awaiting_change_approval":
        return RunStatus.INTERRUPTED, RunTerminalReason.AWAITING_APPROVAL
    if status in {"active", "needs_edit_clarification"} and interrupt is not None:
        return RunStatus.INTERRUPTED, RunTerminalReason.AWAITING_USER_ACTION
    if status == "requires_new_plan":
        return RunStatus.INTERRUPTED, RunTerminalReason.REQUIRES_NEW_PLAN
    if status == "infeasible":
        return RunStatus.COMPLETED, RunTerminalReason.BUSINESS_INFEASIBLE
    return RunStatus.COMPLETED, RunTerminalReason.PLAN_COMPLETED


def classify_error(error: BaseException) -> RunTerminalReason:
    if isinstance(error, asyncio.CancelledError):
        return RunTerminalReason.USER_CANCELLED
    if isinstance(error, ExecutionBudgetExceeded):
        return (
            RunTerminalReason.DEADLINE_EXCEEDED
            if error.deadline
            else RunTerminalReason.EXECUTION_BUDGET_EXHAUSTED
        )
    module = type(error).__module__
    name = type(error).__name__
    if "tools" in module or name == "WeatherUnavailableError":
        return RunTerminalReason.EXTERNAL_TOOL_FAILURE
    if name in {
        "RequirementUnavailableError",
        "EditUnavailableError",
        "CriticUnavailableError",
    }:
        return RunTerminalReason.LLM_PROVIDER_FAILURE
    if "Checkpoint" in name:
        return RunTerminalReason.CHECKPOINT_FAILURE
    if "Lifecycle" in name or "Repository" in name:
        return RunTerminalReason.REPOSITORY_FAILURE
    return RunTerminalReason.INVALID_INTERNAL_STATE


def _plan_version_id(payload: object) -> str | None:
    active = getattr(payload, "active_version", None)
    return str(getattr(active, "version_id", "")) or None
