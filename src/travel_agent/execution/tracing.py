from __future__ import annotations

from datetime import datetime, timezone
import logging
from time import monotonic
from typing import Callable
from uuid import uuid4

from travel_agent.execution.budget import ExecutionLedger
from travel_agent.execution.file_trace import bounded_action_details
from travel_agent.execution.models import JsonScalar, TraceEvent, TraceEventType, TraceStatus


logger = logging.getLogger(__name__)


_SAFE_ATTRIBUTE_KEYS = {
    "action_kind",
    "action_fingerprint",
    "tool_name",
    "schema_name",
    "arguments_hash",
    "evidence_id",
    "evidence_kind",
    "evidence_count",
    "gap_count",
    "required_gap_count",
    "candidate_id",
    "violation_fingerprint",
    "decision_round",
    "guard_result",
    "terminal_reason",
    "evidence_count",
    "extracted_count",
    "accepted_count",
    "rejected_count",
    "deduplicated_count",
    "proposal_count",
    "reason_code",
    "policy_version",
    "mode",
    "confidence",
    "confirmed",
    "candidate_count",
    "day_count",
    "start_role",
    "end_role",
    "provider",
    "thread_id",
    "category",
    "code",
    "cache_hit",
    "attempt_count",
    "error_code",
    "model",
    "prompt_version",
    "input_chars",
    "input_tokens",
    "output_tokens",
    "reason",
    "run_kind",
    "replay_of_run_id",
    "terminal_reason",
    "error_type",
    "limit",
    "target",
    "reason_code",
    "graph_steps",
    "tool_calls",
    "llm_calls",
    "repair_rounds",
    "agent_role",
    "handoff_id",
    "input_schema",
    "output_schema",
    "context_characters",
    "output_characters",
    "memory_id",
    "proposal_id",
    "namespace_hash",
    "role",
    "retrieved_count",
    "selected_count",
    "excluded_count",
    "conflict_count",
    "estimated_tokens",
    "context_id",
    "chain_position",
    "from_provider",
    "to_provider",
    "budget_profile",
    "execution",
    "run_id",
}


class TraceRecorder:
    """Run 内同步收集安全事件，并把每个新事件交给实时 Trace Sink。"""

    def __init__(
        self,
        run_id: str,
        ledger: ExecutionLedger,
        *,
        attribute_max_chars: int = 256,
        utcnow: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] = monotonic,
        id_factory: Callable[[], str] | None = None,
        event_sink: Callable[[TraceEvent], None] | None = None,
        file_detail_sink: Callable[[TraceEvent, dict[str, object]], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.ledger = ledger
        self.attribute_max_chars = attribute_max_chars
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._clock = monotonic_clock
        self._started = monotonic_clock()
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._event_sink = event_sink
        self._file_detail_sink = file_detail_sink
        self._events: list[TraceEvent] = []
        self.degraded_reasons: list[str] = []
        self._degraded = False
        self._drop_noted = False

    @property
    def status(self) -> TraceStatus:
        return TraceStatus.DEGRADED if self._degraded else TraceStatus.COMPLETE

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        event_type: TraceEventType,
        *,
        status: str,
        terminal: bool = False,
        graph: str | None = None,
        node: str | None = None,
        operation: str | None = None,
        duration_ms: int | None = None,
        parent_event_id: str | None = None,
        attempt: int | None = None,
        plan_version_id: str | None = None,
        attributes: dict[str, JsonScalar] | None = None,
        file_details: dict[str, object] | None = None,
    ) -> TraceEvent | None:
        budget = self.ledger.budget
        normal_limit = budget.max_trace_events - budget.terminal_trace_reserve
        if not terminal and len(self._events) >= normal_limit:
            self._degraded = True
            self._drop_noted = True
            self.add_degradation("trace_event_limit")
            return None
        if len(self._events) >= budget.max_trace_events:
            self._degraded = True
            self.add_degradation("trace_event_limit")
            return None
        safe_attributes = {
            str(key)[:64]: self._sanitize(value)
            for key, value in (attributes or {}).items()
            if str(key) in _SAFE_ATTRIBUTE_KEYS
        }
        event = TraceEvent(
            event_id=self._id_factory(),
            run_id=self.run_id,
            sequence=len(self._events) + 1,
            event_type=event_type,
            timestamp=self._utcnow(),
            monotonic_offset_ms=max(0, round((self._clock() - self._started) * 1000)),
            duration_ms=duration_ms,
            graph=graph,
            node=node,
            operation=operation,
            status=status,
            parent_event_id=parent_event_id,
            attempt=attempt,
            plan_version_id=plan_version_id,
            attributes=safe_attributes,
        )
        self._events.append(event)
        self.ledger.note_trace_event()
        if self._event_sink is not None:
            self._event_sink(event)
        if self._file_detail_sink is not None and file_details is not None:
            try:
                self._file_detail_sink(event, bounded_action_details(file_details))
            except Exception as error:
                self.mark_degraded()
                self.add_degradation("trace_detail_serialization_failure")
                logger.warning(
                    "agent_run.trace_details_failed | run_id=%s event_id=%s error_type=%s",
                    self.run_id, event.event_id, type(error).__name__,
                )
        return event

    def mark_degraded(self) -> None:
        self._degraded = True

    def add_degradation(self, reason: str) -> None:
        if reason not in self.degraded_reasons:
            self.degraded_reasons.append(reason)

    def _sanitize(self, value: JsonScalar) -> JsonScalar:
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return str(value)[: self.attribute_max_chars]
