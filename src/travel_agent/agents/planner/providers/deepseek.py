from __future__ import annotations

import json
from typing import Any

from travel_agent.agents.actions import PlannerDecision
from travel_agent.agents.context import DynamicPlannerContext
from travel_agent.agents.planner.errors import PlannerProviderError
from travel_agent.agents.planner.prompts import PLANNER_PROMPT_VERSION, PLANNER_SYSTEM_PROMPT
from travel_agent.agents.planner.providers._compat import map_provider_error, usage_value


class DeepSeekPlannerModel:
    name = "deepseek"
    prompt_version = PLANNER_PROMPT_VERSION

    def __init__(self, *, client: Any, model: str, max_tokens: int = 1200) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens

    async def decide(self, context: DynamicPlannerContext) -> PlannerDecision:
        schema = json.dumps(PlannerDecision.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{PLANNER_SYSTEM_PROMPT}\nPlannerDecision JSON Schema:\n{schema}"},
                    {"role": "user", "content": json.dumps(context.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))},
                ],
                response_format={"type": "json_object"},
                max_tokens=self._max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except PlannerProviderError:
            raise
        except Exception as error:
            raise map_provider_error(error) from None
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise PlannerProviderError("missing_choice", "动态规划决策缺少模型输出")
        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise PlannerProviderError("incomplete", "动态规划决策输出不完整")
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if not isinstance(content, str) or not content.strip():
            raise PlannerProviderError("empty_content", "动态规划决策缺少 JSON 输出")
        try:
            decision = PlannerDecision.model_validate_json(content, strict=True)
        except Exception:
            raise PlannerProviderError("invalid_schema", "动态规划决策结构无效") from None
        usage = getattr(response, "usage", None)
        return decision.model_copy(update={
            "input_tokens": usage_value(usage, "prompt_tokens"),
            "output_tokens": usage_value(usage, "completion_tokens"),
        })
