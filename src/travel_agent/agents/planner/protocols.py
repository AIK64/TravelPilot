from __future__ import annotations

from typing import Protocol, runtime_checkable

from travel_agent.agents.actions import PlannerDecision
from travel_agent.agents.context import DynamicPlannerContext


@runtime_checkable
class PlannerModel(Protocol):
    name: str
    model: str
    prompt_version: str

    async def decide(self, context: DynamicPlannerContext) -> PlannerDecision: ...
