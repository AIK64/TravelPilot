from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from travel_agent.execution.budget import ExecutionLedger
from travel_agent.execution.faults import FaultInjector, FaultMode, FaultPoint
from travel_agent.execution.models import AgentRunRecord, TraceEventType
from travel_agent.execution.tracing import TraceRecorder


@dataclass(slots=True)
class RunContext:
    record: AgentRunRecord
    ledger: ExecutionLedger
    trace: TraceRecorder
    faults: FaultInjector


_CURRENT_RUN: ContextVar[RunContext | None] = ContextVar(
    "travel_agent_current_run", default=None
)


def current_run_context() -> RunContext | None:
    return _CURRENT_RUN.get()


@contextmanager
def bind_run_context(context: RunContext) -> Iterator[RunContext]:
    token = _CURRENT_RUN.set(context)
    try:
        yield context
    finally:
        _CURRENT_RUN.reset(token)


def current_run_id() -> str | None:
    context = current_run_context()
    return context.record.run_id if context is not None else None


def effective_timeout(component_timeout: float) -> float:
    context = current_run_context()
    return (
        context.ledger.effective_timeout(component_timeout)
        if context is not None
        else component_timeout
    )


def match_fault(
    point: FaultPoint, *, operation: str | None = None, attempt: int = 1
) -> FaultMode | None:
    context = current_run_context()
    if context is None:
        return None
    return context.faults.match(point, operation=operation, attempt=attempt)


def begin_tool(operation: str, *, provider: str, thread_id: str, request: object = None) -> str | None:
    context = current_run_context()
    if context is None:
        return None
    context.ledger.consume_tool_call()
    event = context.trace.record(
        TraceEventType.TOOL_STARTED,
        status="started",
        operation=operation,
        attributes={"provider": provider, "thread_id": thread_id},
        file_details={"request": request} if request is not None else None,
    )
    return event.event_id if event else None


def begin_tool_attempt(
    operation: str, *, provider: str, attempt: int, parent_event_id: str | None
) -> None:
    context = current_run_context()
    if context is None:
        return
    context.ledger.consume_provider_attempt()
    context.trace.record(
        TraceEventType.TOOL_PROVIDER_ATTEMPT,
        status="started",
        operation=operation,
        attempt=attempt,
        parent_event_id=parent_event_id,
        attributes={"provider": provider},
    )


def tool_retry(
    operation: str,
    *,
    provider: str,
    attempt: int,
    category: str,
    code: str,
    parent_event_id: str | None,
) -> None:
    context = current_run_context()
    if context is not None:
        context.trace.record(
            TraceEventType.TOOL_RETRY,
            status="scheduled",
            operation=operation,
            attempt=attempt,
            parent_event_id=parent_event_id,
            attributes={"provider": provider, "category": category, "code": code},
        )


def finish_tool(
    operation: str,
    *,
    provider: str,
    status: str,
    cache_hit: bool,
    attempt_count: int,
    elapsed_ms: float | None,
    error_code: str | None,
    parent_event_id: str | None,
    request: object = None,
    result: object = None,
) -> None:
    context = current_run_context()
    if context is None:
        return
    if cache_hit:
        context.ledger.note_cache_hit()
    event_type = (
        TraceEventType.TOOL_CACHE_HIT
        if cache_hit
        else TraceEventType.TOOL_COMPLETED
        if status == "success"
        else TraceEventType.TOOL_FAILED
    )
    context.trace.record(
        event_type,
        status=status,
        operation=operation,
        duration_ms=round(elapsed_ms or 0),
        parent_event_id=parent_event_id,
        attributes={
            "provider": provider,
            "cache_hit": cache_hit,
            "attempt_count": attempt_count,
            "error_code": error_code,
        },
        file_details={"request": request, "result": result} if result is not None else None,
    )


def begin_llm(
    operation: str,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    input_chars: int,
) -> str | None:
    context = current_run_context()
    if context is None:
        return None
    context.ledger.consume_llm_call(input_chars)
    event = context.trace.record(
        TraceEventType.LLM_STARTED,
        status="started",
        operation=operation,
        attributes={
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "input_chars": input_chars,
        },
    )
    return event.event_id if event else None


def begin_llm_attempt(
    operation: str,
    *,
    provider: str,
    model: str,
    attempt: int,
    parent_event_id: str | None,
) -> None:
    context = current_run_context()
    if context is None:
        return
    context.ledger.consume_llm_attempt()
    context.trace.record(
        TraceEventType.LLM_ATTEMPT,
        status="started",
        operation=operation,
        attempt=attempt,
        parent_event_id=parent_event_id,
        attributes={"provider": provider, "model": model},
    )


def llm_retry(
    operation: str,
    *,
    attempt: int,
    category: str,
    code: str,
    parent_event_id: str | None,
) -> None:
    context = current_run_context()
    if context is not None:
        context.trace.record(
            TraceEventType.LLM_RETRY,
            status="scheduled",
            operation=operation,
            attempt=attempt,
            parent_event_id=parent_event_id,
            attributes={"category": category, "code": code},
        )


def finish_llm(
    operation: str,
    *,
    provider: str,
    model: str,
    status: str,
    attempt_count: int,
    elapsed_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
    error_code: str | None,
    parent_event_id: str | None,
) -> None:
    context = current_run_context()
    if context is None:
        return
    if status == "success":
        context.ledger.complete_llm(input_tokens, output_tokens)
    context.trace.record(
        TraceEventType.LLM_COMPLETED if status == "success" else TraceEventType.LLM_FAILED,
        status=status,
        operation=operation,
        duration_ms=round(elapsed_ms),
        parent_event_id=parent_event_id,
        attributes={
            "provider": provider,
            "model": model,
            "attempt_count": attempt_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error_code": error_code,
        },
    )


def record_checkpoint(operation: str, *, success: bool, write: bool) -> None:
    context = current_run_context()
    if context is None:
        return
    if write:
        context.ledger.consume_checkpoint_write()
    context.trace.record(
        (
            TraceEventType.CHECKPOINT_WRITTEN
            if write and success
            else TraceEventType.CHECKPOINT_READ
            if success
            else TraceEventType.CHECKPOINT_FAILED
        ),
        status="success" if success else "failed",
        operation=operation,
    )


def record_repository(operation: str, *, status: str, code: str | None = None) -> None:
    context = current_run_context()
    if context is None:
        return
    event_type = {
        "create": TraceEventType.REPOSITORY_CREATED,
        "get": TraceEventType.REPOSITORY_READ,
        "save": TraceEventType.REPOSITORY_SAVED,
        "cas_conflict": TraceEventType.REPOSITORY_CAS_CONFLICT,
    }.get(operation, TraceEventType.REPOSITORY_FAILED)
    context.trace.record(
        event_type,
        status=status,
        operation=operation,
        attributes={"code": code},
    )


def record_degradation(reason: str) -> None:
    context = current_run_context()
    if context is not None:
        context.trace.add_degradation(reason)
        context.trace.record(
            TraceEventType.DEGRADATION_APPLIED,
            status="degraded",
            attributes={"reason": reason},
        )


def record_stay_resolution(
    *, mode: str, confidence: float, confirmed: bool, candidate_count: int
) -> None:
    context = current_run_context()
    if context is not None:
        context.trace.record(
            TraceEventType.STAY_ANCHOR_RESOLVED,
            status="resolved",
            operation="resolve_stay_anchor",
            attributes={
                "mode": mode,
                "confidence": confidence,
                "confirmed": confirmed,
                "candidate_count": candidate_count,
            },
        )


def record_day_boundaries(
    *, day_count: int, start_role: str, end_role: str
) -> None:
    context = current_run_context()
    if context is not None:
        context.trace.record(
            TraceEventType.DAY_BOUNDARIES_RESOLVED,
            status="resolved",
            operation="derive_day_boundaries",
            attributes={
                "day_count": day_count,
                "start_role": start_role,
                "end_role": end_role,
            },
        )


def record_memory_event(
    event_type: TraceEventType,
    *,
    status: str,
    operation: str,
    attributes: dict[str, str | int | float | bool | None] | None = None,
) -> None:
    context = current_run_context()
    if context is not None:
        context.trace.record(
            event_type,
            status=status,
            operation=operation,
            attributes=attributes,
        )


def record_agent_event(
    event_type: TraceEventType,
    *,
    status: str,
    operation: str,
    attributes: dict[str, str | int | float | bool | None] | None = None,
    file_details: dict[str, object] | None = None,
) -> None:
    """记录经过白名单裁剪的动态 Agent 事件，不写入 Prompt 或正文。"""
    context = current_run_context()
    if context is not None:
        context.trace.record(
            event_type,
            status=status,
            operation=operation,
            attributes=attributes,
            file_details=file_details,
            terminal=event_type
            in {
                TraceEventType.FINAL_GUARD_COMPLETED,
                TraceEventType.AGENT_NO_PROGRESS,
            },
        )


def consume_agent_decision() -> None:
    context = current_run_context()
    if context is not None:
        context.ledger.consume_agent_decision()


def consume_invalid_action() -> None:
    context = current_run_context()
    if context is not None:
        context.ledger.consume_invalid_action()


def consume_evidence_record() -> None:
    context = current_run_context()
    if context is not None:
        context.ledger.consume_evidence_record()
