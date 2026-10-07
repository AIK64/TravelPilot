from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from travel_agent.agents.actions import PlannerDecision
from travel_agent.agents.context import DynamicPlannerContext
from travel_agent.agents.planner.errors import (
    PlannerProviderError,
    PlannerUnavailableError,
)
from travel_agent.agents.planner.protocols import PlannerModel
from travel_agent.execution.context import (
    begin_llm,
    begin_llm_attempt,
    consume_agent_decision,
    effective_timeout,
    finish_llm,
    llm_retry,
    record_agent_event,
)
from travel_agent.execution.models import TraceEventType


logger = logging.getLogger(__name__)


class PlannerGateway:
    def __init__(
        self,
        *,
        model: PlannerModel,
        timeout_seconds: float = 20.0,
        max_attempts: int = 2,
        base_delay_seconds: float = 0.25,
    ) -> None:
        if timeout_seconds <= 0 or max_attempts < 1 or base_delay_seconds < 0:
            raise ValueError("invalid planner gateway limits")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds

    async def decide(
        self, context: DynamicPlannerContext, *, thread_id: str, decision_round: int
    ) -> PlannerDecision:
        input_chars = len(context.model_dump_json())
        consume_agent_decision()
        record_agent_event(
            TraceEventType.AGENT_DECISION_STARTED,
            status="started",
            operation="planner.decide",
            attributes={"decision_round": decision_round},
            file_details={"request": context},
        )
        parent_event_id = begin_llm(
            "planner.decide",
            provider=self.model.name,
            model=self.model.model,
            prompt_version=self.model.prompt_version,
            input_chars=input_chars,
        )
        started = perf_counter()
        last_error: PlannerProviderError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                begin_llm_attempt(
                    "planner.decide",
                    provider=self.model.name,
                    model=self.model.model,
                    attempt=attempt,
                    parent_event_id=parent_event_id,
                )
                decision = await asyncio.wait_for(
                    self.model.decide(context),
                    timeout=effective_timeout(self.timeout_seconds),
                )
            except TimeoutError:
                last_error = PlannerProviderError("timeout", "动态规划决策超时")
            except PlannerProviderError as error:
                last_error = error
            else:
                elapsed_ms = round((perf_counter() - started) * 1000, 2)
                finish_llm(
                    "planner.decide",
                    provider=self.model.name,
                    model=self.model.model,
                    status="success",
                    attempt_count=attempt,
                    elapsed_ms=elapsed_ms,
                    input_tokens=decision.input_tokens,
                    output_tokens=decision.output_tokens,
                    error_code=None,
                    parent_event_id=parent_event_id,
                )
                record_agent_event(
                    TraceEventType.AGENT_DECISION_COMPLETED,
                    status="completed",
                    operation="planner.decide",
                    attributes={
                        "decision_round": decision_round,
                        "action_kind": decision.action.kind,
                        "reason_code": decision.action.reason_code,
                    },
                    file_details={"result": decision},
                )
                logger.info(
                    "agent.decision.completed thread_id=%s round=%s action=%s reason=%s",
                    thread_id,
                    decision_round,
                    decision.action.kind,
                    decision.action.reason_code,
                )
                return decision
            assert last_error is not None
            if not last_error.retryable or attempt == self.max_attempts:
                break
            llm_retry(
                "planner.decide",
                attempt=attempt,
                category="invalid_or_transient",
                code=last_error.code,
                parent_event_id=parent_event_id,
            )
            if self.base_delay_seconds:
                await asyncio.sleep(self.base_delay_seconds * attempt)

        assert last_error is not None
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        finish_llm(
            "planner.decide",
            provider=self.model.name,
            model=self.model.model,
            status="failed",
            attempt_count=attempt,
            elapsed_ms=elapsed_ms,
            input_tokens=None,
            output_tokens=None,
            error_code=last_error.code,
            parent_event_id=parent_event_id,
        )
        record_agent_event(
            TraceEventType.AGENT_DECISION_FAILED,
            status="failed",
            operation="planner.decide",
            attributes={"decision_round": decision_round, "reason_code": last_error.code},
        )
        raise PlannerUnavailableError(
            provider=self.model.name,
            model=self.model.model,
            code=last_error.code,
            attempt_count=attempt,
        )
