from __future__ import annotations

from typing import Any

from travel_agent.agents.context import ReplannerContext
from travel_agent.agents.replanner.errors import ReplannerProviderError
from travel_agent.agents.replanner.models import ReplannerOutput
from travel_agent.agents.replanner.prompts import REPLANNER_PROMPT_VERSION, REPLANNER_SYSTEM_PROMPT
from travel_agent.agents.replanner.providers._compat import map_provider_error, usage_value


class OpenAIReplannerModel:
    name = "openai"
    prompt_version = REPLANNER_PROMPT_VERSION

    def __init__(self, *, client: Any, model: str) -> None:
        self._client = client
        self.model = model

    async def propose(self, context: ReplannerContext) -> ReplannerOutput:
        try:
            response = await self._client.responses.parse(
                model=self.model,
                input=[{"role": "system", "content": REPLANNER_SYSTEM_PROMPT}, {"role": "user", "content": context.model_dump_json()}],
                text_format=ReplannerOutput,
                store=False,
            )
        except Exception as error:
            raise map_provider_error(error) from None
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ReplannerProviderError("invalid_schema", "修复模型缺少结构化输出")
        output = parsed if isinstance(parsed, ReplannerOutput) else ReplannerOutput.model_validate(parsed)
        usage = getattr(response, "usage", None)
        return output.model_copy(update={"input_tokens": usage_value(usage, "input_tokens"), "output_tokens": usage_value(usage, "output_tokens")})
