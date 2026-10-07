from __future__ import annotations

import inspect
from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar

from langgraph.errors import GraphInterrupt

from travel_agent.execution.context import current_run_context
from travel_agent.execution.models import TraceEventType


F = TypeVar("F", bound=Callable[..., Any])


def instrument_node(graph: str, node: str, function: F, *, terminal: bool = False) -> F:
    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
            context = current_run_context()
            started = perf_counter()
            parent = None
            if context is not None:
                context.ledger.consume_graph_step(terminal=terminal)
                if node in _REPAIR_NODES:
                    context.ledger.consume_repair_round()
                event = context.trace.record(
                    TraceEventType.NODE_STARTED,
                    status="started",
                    graph=graph,
                    node=node,
                )
                parent = event.event_id if event else None
            try:
                result = await function(*args, **kwargs)
            except GraphInterrupt:
                if context is not None:
                    context.ledger.consume_interrupt()
                    context.trace.record(
                        TraceEventType.INTERRUPT_CREATED,
                        status="interrupted",
                        graph=graph,
                        node=node,
                        duration_ms=round((perf_counter() - started) * 1000),
                        parent_event_id=parent,
                    )
                raise
            except BaseException as error:
                if context is not None:
                    context.trace.record(
                        TraceEventType.NODE_FAILED,
                        status="failed",
                        graph=graph,
                        node=node,
                        duration_ms=round((perf_counter() - started) * 1000),
                        parent_event_id=parent,
                        attributes={"error_type": type(error).__name__},
                    )
                raise
            if context is not None:
                context.trace.record(
                    TraceEventType.NODE_COMPLETED,
                    status="completed",
                    graph=graph,
                    node=node,
                    duration_ms=round((perf_counter() - started) * 1000),
                    parent_event_id=parent,
                )
                _record_domain_event(context, graph, node, result)
            return result
        return async_wrapped  # type: ignore[return-value]

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        context = current_run_context()
        started = perf_counter()
        parent = None
        if context is not None:
            context.ledger.consume_graph_step(terminal=terminal)
            if node in _REPAIR_NODES:
                context.ledger.consume_repair_round()
            event = context.trace.record(
                TraceEventType.NODE_STARTED,
                status="started",
                graph=graph,
                node=node,
            )
            parent = event.event_id if event else None
        try:
            result = function(*args, **kwargs)
        except GraphInterrupt:
            if context is not None:
                context.ledger.consume_interrupt()
                context.trace.record(
                    TraceEventType.INTERRUPT_CREATED,
                    status="interrupted",
                    graph=graph,
                    node=node,
                    duration_ms=round((perf_counter() - started) * 1000),
                    parent_event_id=parent,
                )
            raise
        except BaseException as error:
            if context is not None:
                context.trace.record(
                    TraceEventType.NODE_FAILED,
                    status="failed",
                    graph=graph,
                    node=node,
                    duration_ms=round((perf_counter() - started) * 1000),
                    parent_event_id=parent,
                    attributes={"error_type": type(error).__name__},
                )
            raise
        if context is not None:
            context.trace.record(
                TraceEventType.NODE_COMPLETED,
                status="completed",
                graph=graph,
                node=node,
                duration_ms=round((perf_counter() - started) * 1000),
                parent_event_id=parent,
            )
            _record_domain_event(context, graph, node, result)
        return result
    return wrapped  # type: ignore[return-value]


def instrument_route(graph: str, node: str, function: F) -> F:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        target = function(*args, **kwargs)
        context = current_run_context()
        if context is not None:
            context.trace.record(
                TraceEventType.ROUTE_DECIDED,
                status="decided",
                graph=graph,
                node=node,
                attributes={"target": str(target), "reason_code": _route_reason(args)},
            )
        return target
    return wrapped  # type: ignore[return-value]


def execution_budget_guard(_state: dict[str, Any]) -> dict[str, Any]:
    context = current_run_context()
    if context is None:
        return {"execution": None}
    context.ledger.ensure_can_continue()
    usage = context.ledger.snapshot()
    context.trace.record(
        TraceEventType.BUDGET_UPDATED,
        status="within_budget",
        operation="execution_budget_guard",
        attributes={
            "graph_steps": usage.graph_steps,
            "tool_calls": usage.tool_calls,
            "llm_calls": usage.llm_calls,
            "repair_rounds": usage.repair_rounds,
        },
    )
    return {
        "execution": {
            "run_id": context.record.run_id,
            "budget_profile": context.record.budget.profile,
            "usage": usage.model_dump(mode="json"),
            "execution_terminal_reason": None,
        }
    }


def _route_reason(args: tuple[Any, ...]) -> str:
    if not args or not isinstance(args[0], dict):
        return "state_route"
    state = args[0]
    for key in (
        "repair_terminal_reason",
        "soft_repair_terminal_reason",
        "weather_decision",
        "status",
    ):
        value = state.get(key)
        if value is not None:
            return str(getattr(value, "value", value))[:128]
    return "state_route"


_REPAIR_NODES = {
    "apply_repair",
    "apply_local_repair",
    "apply_soft_repair",
    "build_local_preview",
    "build_weather_repair_plan",
}


def _record_domain_event(context, graph: str, node: str, result: Any) -> None:
    if node in {"validate_candidates", "build_local_preview"}:
        status = result.get("status", "completed") if isinstance(result, dict) else "completed"
        context.trace.record(
            TraceEventType.VALIDATION_COMPLETED,
            status=str(status),
            graph=graph,
            node=node,
        )
    elif node in {"build_repair_plan", "compile_soft_repair_plan", "build_weather_repair_plan"}:
        plan = None
        if isinstance(result, dict):
            plan = (
                result.get("repair_plan")
                or result.get("soft_repair_plan")
                or result.get("weather_repair_plan")
            )
        fingerprint = getattr(plan, "action_fingerprint", None)
        if fingerprint:
            context.ledger.note_action_fingerprint(str(fingerprint))
        context.trace.record(
            TraceEventType.REPAIR_PLANNED,
            status="planned",
            graph=graph,
            node=node,
        )
    elif node in {"apply_repair", "apply_local_repair", "apply_soft_repair"}:
        context.trace.record(
            TraceEventType.REPAIR_APPLIED,
            status="applied",
            graph=graph,
            node=node,
        )
