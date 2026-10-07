from __future__ import annotations

import json
from typing import Any

from travel_agent.agents.actions import PlannerDecision
from travel_agent.agents.context import DynamicPlannerContext
from travel_agent.agents.planner.errors import PlannerProviderError
from travel_agent.agents.planner.prompts import PLANNER_PROMPT_VERSION, PLANNER_SYSTEM_PROMPT
from travel_agent.agents.planner.providers._compat import map_provider_error, usage_value


class OpenAIPlannerModel:
    name = "openai"
    prompt_version = PLANNER_PROMPT_VERSION

    def __init__(self, *, client: Any, model: str) -> None:
        self._client = client
        self.model = model

    async def decide(self, context: DynamicPlannerContext) -> PlannerDecision:
        try:
            response = await self._client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": _context_json(context)},
                ],
                text_format=PlannerDecision,
                store=False,
            )
        except PlannerProviderError:
            raise
        except Exception as error:
            raise map_provider_error(error) from None
        if getattr(response, "status", None) == "incomplete":
            raise PlannerProviderError("incomplete", "动态规划决策输出不完整")
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise PlannerProviderError("missing_structured_output", "动态规划决策缺少结构化输出")
        try:
            decision = parsed if isinstance(parsed, PlannerDecision) else PlannerDecision.model_validate(parsed)
        except Exception:
            raise PlannerProviderError("invalid_schema", "动态规划决策结构无效") from None
        usage = getattr(response, "usage", None)
        return decision.model_copy(update={
            "input_tokens": usage_value(usage, "input_tokens"),
            "output_tokens": usage_value(usage, "output_tokens"),
        })


def _context_json(context: DynamicPlannerContext) -> str:
    return json.dumps(context.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
