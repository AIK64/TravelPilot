from __future__ import annotations

import asyncio
from time import perf_counter

from travel_agent.agents.context import ReplannerContext
from travel_agent.agents.replanner.errors import ReplannerProviderError
from travel_agent.agents.replanner.models import ReplannerOutput
from travel_agent.agents.replanner.protocols import ReplannerModel
from travel_agent.execution.context import begin_llm, begin_llm_attempt, effective_timeout, finish_llm, llm_retry, record_agent_event
from travel_agent.execution.models import TraceEventType


class ReplannerGateway:
    def __init__(self, *, model: ReplannerModel, timeout_seconds: float = 20, max_attempts: int = 2) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    async def propose(self, context: ReplannerContext) -> ReplannerOutput:
        parent = begin_llm(
            "replanner.propose",
            provider=self.model.name,
            model=self.model.model,
            prompt_version=self.model.prompt_version,
            input_chars=len(context.model_dump_json()),
        )
        started = perf_counter()
        last_error: ReplannerProviderError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                begin_llm_attempt("replanner.propose", provider=self.model.name, model=self.model.model, attempt=attempt, parent_event_id=parent)
                output = await asyncio.wait_for(self.model.propose(context), timeout=effective_timeout(self.timeout_seconds))
            except TimeoutError:
                last_error = ReplannerProviderError("timeout", "修复模型调用超时")
            except ReplannerProviderError as error:
                last_error = error
            else:
                finish_llm(
                    "replanner.propose", provider=self.model.name, model=self.model.model,
                    status="success", attempt_count=attempt,
                    elapsed_ms=round((perf_counter() - started) * 1000, 2),
                    input_tokens=output.input_tokens, output_tokens=output.output_tokens,
                    error_code=None, parent_event_id=parent,
                )
                record_agent_event(
                    TraceEventType.REPLANNER_PROPOSAL_CREATED,
                    status="created",
                    operation="replanner.propose",
                    attributes={"candidate_id": output.proposal.target_candidate_id, "proposal_count": 1},
                )
                return output
            assert last_error is not None
            if not last_error.retryable or attempt == self.max_attempts:
                break
            llm_retry("replanner.propose", attempt=attempt, category="transient", code=last_error.code, parent_event_id=parent)
        assert last_error is not None
        finish_llm(
            "replanner.propose", provider=self.model.name, model=self.model.model,
            status="failed", attempt_count=attempt,
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
            input_tokens=None, output_tokens=None, error_code=last_error.code,
            parent_event_id=parent,
        )
        raise last_error
