from __future__ import annotations

import asyncio

import pytest

from travel_agent.agents import AgentBudget, HandoffReason, SpecialistExecutor
from travel_agent.agents.context import PlannerContext
from travel_agent.agents.errors import SpecialistContextRejected, SpecialistTimeout
from travel_agent.memory.models import AgentRole
from travel_agent.planning.search_plan import build_search_plan
from travel_agent.config import AgentMode, Settings
from travel_agent.domain.models import PlanningRequest
from travel_agent.execution.models import TraceEventType
from travel_agent.identity.models import Principal
from travel_agent.runtime import PlanningRuntime


@pytest.mark.asyncio
async def test_planner_specialist_only_receives_projected_context(hangzhou_trip):
    executor = SpecialistExecutor(max_handoffs=2)
    context = PlannerContext(
        trip=hangzhou_trip,
        poi_query_limit=4,
        max_queries=6,
    )

    queries, summary = await executor.invoke(
        role=AgentRole.PLANNER,
        reason=HandoffReason.PLAN,
        context=context,
        expected_output_schema="list[POISearchQuery]",
        operation=lambda: build_search_plan(
            context.trip,
            per_query_limit=context.poi_query_limit,
            max_queries=context.max_queries,
        ),
    )

    assert queries
    assert summary.role is AgentRole.PLANNER
    assert summary.output_hash
    dumped = context.model_dump_json()
    assert "critic_report" not in dumped
    assert "repair_history" not in dumped
    assert "raw_provider" not in dumped


@pytest.mark.asyncio
async def test_specialist_rejects_context_over_budget(hangzhou_trip):
    executor = SpecialistExecutor()
    context = PlannerContext(
        trip=hangzhou_trip,
        poi_query_limit=4,
        max_queries=6,
    )

    with pytest.raises(SpecialistContextRejected) as captured:
        await executor.invoke(
            role=AgentRole.PLANNER,
            reason=HandoffReason.PLAN,
            context=context,
            expected_output_schema="list",
            operation=lambda: [],
            budget=AgentBudget(max_context_characters=128),
        )

    assert captured.value.code == "context_budget_exceeded"


@pytest.mark.asyncio
async def test_specialist_enforces_deadline(hangzhou_trip):
    executor = SpecialistExecutor()
    context = PlannerContext(
        trip=hangzhou_trip,
        poi_query_limit=4,
        max_queries=6,
    )

    async def slow_operation():
        await asyncio.sleep(0.2)
        return []

    with pytest.raises(SpecialistTimeout):
        await executor.invoke(
            role=AgentRole.PLANNER,
            reason=HandoffReason.PLAN,
            context=context,
            expected_output_schema="list",
            operation=slow_operation,
            budget=AgentBudget(deadline_ms=100),
        )


@pytest.mark.asyncio
async def test_specialist_rejects_large_output_and_propagates_failure(hangzhou_trip):
    executor = SpecialistExecutor()
    context = PlannerContext(trip=hangzhou_trip, poi_query_limit=4, max_queries=6)

    with pytest.raises(SpecialistContextRejected) as output_error:
        await executor.invoke(
            role=AgentRole.PLANNER,
            reason=HandoffReason.PLAN,
            context=context,
            expected_output_schema="str",
            operation=lambda: "x" * 500,
            budget=AgentBudget(max_output_characters=128),
        )
    assert output_error.value.code == "output_budget_exceeded"

    def fail():
        raise RuntimeError("specialist-test-failure")

    with pytest.raises(RuntimeError, match="specialist-test-failure"):
        await executor.invoke(
            role=AgentRole.PLANNER,
            reason=HandoffReason.PLAN,
            context=context,
            expected_output_schema="str",
            operation=fail,
        )


@pytest.mark.asyncio
async def test_legacy_specialist_mode_migrates_to_dynamic_planner(hangzhou_trip):
    runtime = await PlanningRuntime.create(
        Settings(agent_mode=AgentMode.SPECIALIST_SUBAGENTS, agent_max_handoffs=8)
    )
    principal = Principal(tenant_id="tenant-a", user_id="user-a")
    try:
        result = await runtime.execute_plan(
            PlanningRequest(trip=hangzhou_trip),
            thread_id="specialist-trace",
            principal=principal,
        )
        assert result.run is not None
        trace = await runtime.get_agent_trace(result.run.run_id, limit=500)
    finally:
        await runtime.close()

    handoffs = [
        event
        for event in trace
        if event.event_type is TraceEventType.AGENT_HANDOFF_STARTED
    ]
    decisions = [
        event
        for event in trace
        if event.event_type is TraceEventType.AGENT_DECISION_COMPLETED
    ]
    assert runtime.agent_mode is AgentMode.DYNAMIC_PLANNER
    assert result.payload.agent_mode == "dynamic_planner"
    assert handoffs == []
    assert decisions
