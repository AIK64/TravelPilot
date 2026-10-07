from __future__ import annotations

import json
from typing import Any

from travel_agent.agents.context import ReplannerContext
from travel_agent.agents.replanner.errors import ReplannerProviderError
from travel_agent.agents.replanner.models import ReplannerOutput
from travel_agent.agents.replanner.prompts import REPLANNER_PROMPT_VERSION, REPLANNER_SYSTEM_PROMPT
from travel_agent.agents.replanner.providers._compat import map_provider_error, usage_value


class DeepSeekReplannerModel:
    name = "deepseek"
    prompt_version = REPLANNER_PROMPT_VERSION

    def __init__(self, *, client: Any, model: str, max_tokens: int = 1600) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens

    async def propose(self, context: ReplannerContext) -> ReplannerOutput:
        schema = json.dumps(ReplannerOutput.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": f"{REPLANNER_SYSTEM_PROMPT}\nReplannerOutput JSON Schema:\n{schema}"}, {"role": "user", "content": context.model_dump_json()}],
                response_format={"type": "json_object"}, max_tokens=self._max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as error:
            raise map_provider_error(error) from None
        choices = getattr(response, "choices", None)
        content = getattr(getattr(choices[0], "message", None), "content", None) if isinstance(choices, list) and choices else None
        if not isinstance(content, str) or not content.strip():
            raise ReplannerProviderError("invalid_schema", "修复模型缺少 JSON 输出")
        try:
            output = ReplannerOutput.model_validate_json(content, strict=True)
        except Exception:
            raise ReplannerProviderError("invalid_schema", "修复模型输出结构无效") from None
        usage = getattr(response, "usage", None)
        return output.model_copy(update={"input_tokens": usage_value(usage, "prompt_tokens"), "output_tokens": usage_value(usage, "completion_tokens")})
