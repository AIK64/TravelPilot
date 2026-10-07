from __future__ import annotations

from typing import Protocol, runtime_checkable

from travel_agent.agents.context import ReplannerContext
from travel_agent.agents.replanner.models import ReplannerOutput


@runtime_checkable
class ReplannerModel(Protocol):
    name: str
    model: str
    prompt_version: str

    async def propose(self, context: ReplannerContext) -> ReplannerOutput: ...
