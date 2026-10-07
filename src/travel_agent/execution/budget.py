from __future__ import annotations

from threading import RLock
from time import monotonic
from typing import Callable

from travel_agent.execution.errors import ExecutionBudgetExceeded
from travel_agent.execution.models import ExecutionBudget, ExecutionUsage


class ExecutionLedger:
    """并发安全的 Run 级计量器；所有检查都在副作用发生前完成。"""

    def __init__(
        self,
        run_id: str,
        budget: ExecutionBudget,
        *,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        self.run_id = run_id
        self.budget = budget
        self._clock = monotonic_clock
        self._started = monotonic_clock()
        self._lock = RLock()
        self._values = {
            "graph_steps": 0,
            "tool_calls": 0,
            "provider_attempts": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "llm_attempts": 0,
            "llm_input_chars": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "repair_rounds": 0,
            "interrupts": 0,
            "checkpoint_writes": 0,
            "trace_events": 0,
            "agent_decisions": 0,
            "invalid_actions": 0,
            "evidence_records": 0,
            "observations": 0,
        }
        self._token_usage_complete = True
        self._last_action_fingerprint: str | None = None
        self._repeated_fingerprint_count = 0

    def elapsed_ms(self) -> int:
        return max(0, round((self._clock() - self._started) * 1000))

    def remaining_seconds(self) -> float:
        remaining = (self.budget.deadline_ms - self.elapsed_ms()) / 1000
        if remaining <= 0:
            raise self._exceeded("deadline_ms", self.elapsed_ms(), self.budget.deadline_ms, True)
        return remaining

    def effective_timeout(self, component_timeout: float) -> float:
        return max(0.001, min(component_timeout, self.remaining_seconds()))

    def ensure_can_continue(self, *, reserve_terminal_steps: bool = True) -> None:
        with self._lock:
            self.remaining_seconds()
            maximum = self.budget.max_graph_steps
            if reserve_terminal_steps:
                maximum -= self.budget.terminal_step_reserve
            if self._values["graph_steps"] >= maximum:
                raise self._exceeded(
                    "max_graph_steps",
                    self._values["graph_steps"],
                    self.budget.max_graph_steps,
                )

    def consume_graph_step(self, *, terminal: bool = False) -> None:
        maximum = self.budget.max_graph_steps
        if not terminal:
            maximum -= self.budget.terminal_step_reserve
        self._consume("graph_steps", 1, maximum, "max_graph_steps")

    def consume_tool_call(self) -> None:
        self._consume("tool_calls", 1, self.budget.max_tool_calls, "max_tool_calls")

    def consume_provider_attempt(self) -> None:
        self._consume(
            "provider_attempts",
            1,
            self.budget.max_provider_attempts,
            "max_provider_attempts",
        )

    def note_cache_hit(self) -> None:
        self._increment("cache_hits")

    def consume_llm_call(self, input_chars: int) -> None:
        with self._lock:
            self.remaining_seconds()
            if self._values["llm_calls"] + 1 > self.budget.max_llm_calls:
                raise self._exceeded(
                    "max_llm_calls", self._values["llm_calls"], self.budget.max_llm_calls
                )
            if self._values["llm_input_chars"] + input_chars > self.budget.max_llm_input_chars:
                raise self._exceeded(
                    "max_llm_input_chars",
                    self._values["llm_input_chars"],
                    self.budget.max_llm_input_chars,
                )
            self._values["llm_calls"] += 1
            self._values["llm_input_chars"] += input_chars

    def consume_llm_attempt(self) -> None:
        self._consume(
            "llm_attempts", 1, self.budget.max_llm_attempts, "max_llm_attempts"
        )

    def complete_llm(self, input_tokens: int | None, output_tokens: int | None) -> None:
        with self._lock:
            if input_tokens is None or output_tokens is None:
                self._token_usage_complete = False
                return
            if self._values["input_tokens"] + input_tokens > self.budget.max_input_tokens:
                raise self._exceeded(
                    "max_input_tokens",
                    self._values["input_tokens"] + input_tokens,
                    self.budget.max_input_tokens,
                )
            if self._values["output_tokens"] + output_tokens > self.budget.max_output_tokens:
                raise self._exceeded(
                    "max_output_tokens",
                    self._values["output_tokens"] + output_tokens,
                    self.budget.max_output_tokens,
                )
            self._values["input_tokens"] += input_tokens
            self._values["output_tokens"] += output_tokens

    def consume_repair_round(self) -> None:
        self._consume(
            "repair_rounds", 1, self.budget.max_repair_rounds, "max_repair_rounds"
        )

    def consume_interrupt(self) -> None:
        self._consume("interrupts", 1, self.budget.max_interrupts, "max_interrupts")

    def consume_checkpoint_write(self) -> None:
        self._consume(
            "checkpoint_writes",
            1,
            self.budget.max_checkpoint_writes,
            "max_checkpoint_writes",
        )

    def note_trace_event(self) -> None:
        self._increment("trace_events")

    def consume_agent_decision(self) -> None:
        self._consume(
            "agent_decisions",
            1,
            self.budget.max_agent_decisions,
            "max_agent_decisions",
        )

    def consume_invalid_action(self) -> None:
        self._consume(
            "invalid_actions",
            1,
            self.budget.max_invalid_actions,
            "max_invalid_actions",
        )

    def consume_evidence_record(self) -> None:
        self._consume(
            "evidence_records",
            1,
            self.budget.max_evidence_records,
            "max_evidence_records",
        )

    def consume_observation(self) -> None:
        self._consume(
            "observations",
            1,
            self.budget.max_observation_history,
            "max_observation_history",
        )

    def note_action_fingerprint(self, fingerprint: str) -> None:
        """在执行重复动作前阻断无进展循环；不同动作会重置连续计数。"""
        with self._lock:
            if fingerprint == self._last_action_fingerprint:
                self._repeated_fingerprint_count += 1
            else:
                self._last_action_fingerprint = fingerprint
                self._repeated_fingerprint_count = 1
            maximum = self.budget.max_repeated_fingerprint_count
            if self._repeated_fingerprint_count > maximum:
                raise self._exceeded(
                    "max_repeated_fingerprint_count",
                    self._repeated_fingerprint_count - 1,
                    maximum,
                )

    def snapshot(self) -> ExecutionUsage:
        with self._lock:
            values = dict(self._values)
            if not self._token_usage_complete:
                values["input_tokens"] = None
                values["output_tokens"] = None
            return ExecutionUsage(**values)

    def _consume(self, key: str, amount: int, maximum: int, limit: str) -> None:
        with self._lock:
            self.remaining_seconds()
            value = self._values[key] + amount
            if value > maximum:
                raise self._exceeded(limit, self._values[key], maximum)
            self._values[key] = value

    def _increment(self, key: str) -> None:
        with self._lock:
            self._values[key] += 1

    def _exceeded(
        self,
        limit: str,
        used: int | float,
        maximum: int | float,
        deadline: bool = False,
    ) -> ExecutionBudgetExceeded:
        return ExecutionBudgetExceeded(
            run_id=self.run_id,
            limit=limit,
            used=used,
            maximum=maximum,
            deadline=deadline,
        )
