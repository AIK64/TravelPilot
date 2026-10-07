from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from travel_agent.domain.models import PlanningRequest, ValidationStatus
from travel_agent.domain.optimization_models import OptimizationSolveStatus
from travel_agent.domain.tool_models import RouteMode, UnknownFactPolicy
from travel_agent.graph.workflow import build_workflow, run_planning
from travel_agent.planning.defaults import POIDefaultPolicy
from travel_agent.planning.policy import PlanningPolicy
from travel_agent.planning.poi_identity import normalize_poi_name


@pytest.mark.asyncio
@pytest.mark.parametrize("harness_name", ["workflow_harness", "fallback_workflow_harness"])
async def test_alias_attractions_are_not_repeated_in_materialized_plans(
    hangzhou_trip, harness_name, request, monkeypatch
):
    harness = request.getfixturevalue(harness_name)
    original_search = harness.poi_provider.search_pois

    async def duplicate_search(query):
        facts = await original_search(query)
        return [
            *facts,
            *(fact.model_copy(update={"id": f"alias-{fact.id}"}) for fact in facts),
        ]

    monkeypatch.setattr(harness.poi_provider, "search_pois", duplicate_search)
    response = await run_planning(
        harness.workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=2),
        thread_id=f"unique-attractions-{harness_name}",
    )
    assert response.status == "completed"
    assert len(response.candidates) == 3
    for candidate in response.candidates:
        names = [
            normalize_poi_name(item.name)
            for day in candidate.days
            for item in day.items
            if item.type.value == "activity"
        ]
        assert len(names) == len(set(names))
        assert names.count("灵隐寺") == 1
    assert response.selected_plan.validation.valid


@pytest.mark.asyncio
async def test_optimizer_builds_three_explainable_valid_variants(
    hangzhou_trip,
    workflow_harness,
):
    response = await run_planning(
        workflow_harness.workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=2),
        thread_id="optimization-three-variants",
    )
    snapshot = await workflow_harness.workflow.aget_state(
        {"configurable": {"thread_id": "optimization-three-variants"}}
    )
    state = snapshot.values
    result = state["optimization_result"]

    assert response.status == "completed"
    assert response.iterations == 0
    assert result.status in {
        OptimizationSolveStatus.OPTIMAL,
        OptimizationSolveStatus.FEASIBLE,
    }
    assert result.degraded_reason is None
    assert len(result.solutions) == 3
    assert {solution.style.value for solution in result.solutions} == {
        "relaxed",
        "balanced",
        "exploration",
    }
    assert all(solution.objective_breakdown.travel_minutes >= 0 for solution in result.solutions)
    assert all(candidate.validation.valid for candidate in response.candidates)
    assert all("-opt-r0" in candidate.id for candidate in response.candidates)
    assert state["optimization_problem"].pois
    assert state["optimization_problem"].route_matrix
    assert {query.mode for query in state["route_queries"]} == {
        RouteMode.DRIVING,
        RouteMode.WALKING,
    }
    assert state["route_matrix_provider_calls"] == len(state["route_queries"])
    assert state["route_matrix_cache_hits"] == 0


@pytest.mark.asyncio
async def test_route_matrix_is_reused_across_threads(
    hangzhou_trip,
    workflow_harness,
):
    await run_planning(
        workflow_harness.workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=0),
        thread_id="optimization-cache-first",
    )
    await run_planning(
        workflow_harness.workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=0),
        thread_id="optimization-cache-second",
    )
    snapshot = await workflow_harness.workflow.aget_state(
        {"configurable": {"thread_id": "optimization-cache-second"}}
    )
    state = snapshot.values

    assert state["route_matrix_cache_hits"] == len(state["route_queries"])
    assert state["route_matrix_provider_calls"] == 0


@pytest.mark.asyncio
async def test_optimizer_timeout_is_visible_and_falls_back_deterministically(
    hangzhou_trip,
    fallback_workflow_harness,
    caplog,
):
    caplog.set_level(logging.INFO, logger="travel_agent.graph.workflow")
    response = await run_planning(
        fallback_workflow_harness.workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=2),
        thread_id="optimization-degraded",
    )
    snapshot = await fallback_workflow_harness.workflow.aget_state(
        {"configurable": {"thread_id": "optimization-degraded"}}
    )
    result = snapshot.values["optimization_result"]

    assert response.status == "completed"
    assert result.status is OptimizationSolveStatus.DEGRADED
    assert result.degraded_reason == "optimizer_timeout"
    assert result.solver == "deterministic-nearest-neighbor-v0.5"
    assert all("-opt-r0" not in candidate.id for candidate in response.candidates)
    assert any(
        record.getMessage().startswith("optimization.degraded")
        and "reason=optimizer_timeout" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_optimizer_avoids_repair_round_for_budget_constraint(
    hangzhou_trip,
    workflow_harness,
    fallback_workflow_harness,
):
    trip = hangzhou_trip.model_copy(update={"total_budget": Decimal("300")})
    optimized = await run_planning(
        workflow_harness.workflow,
        PlanningRequest(trip=trip, max_replan_rounds=2),
        thread_id="optimization-budget",
    )
    heuristic = await run_planning(
        fallback_workflow_harness.workflow,
        PlanningRequest(trip=trip, max_replan_rounds=2),
        thread_id="heuristic-budget",
    )

    assert optimized.status == heuristic.status == "completed"
    assert optimized.iterations == 0
    assert heuristic.iterations == 1
    assert optimized.selected_plan is not None
    assert optimized.selected_plan.metrics.known_estimated_cost <= Decimal("300")


@pytest.mark.asyncio
async def test_real_walking_ablation_removes_distance_estimation_warning(
    hangzhou_trip,
    gateway_factory,
):
    actual_workflow = build_workflow(
        gateway_factory(),
        POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING),
        PlanningPolicy(use_real_walking_routes=True),
    )
    estimated_workflow = build_workflow(
        gateway_factory(),
        POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING),
        PlanningPolicy(use_real_walking_routes=False),
    )

    actual = await run_planning(
        actual_workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=0),
        thread_id="walking-actual",
    )
    estimated = await run_planning(
        estimated_workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=0),
        thread_id="walking-estimated",
    )

    assert actual.selected_plan is not None
    assert estimated.selected_plan is not None
    assert actual.selected_plan.validation.status is ValidationStatus.VALID
    assert estimated.selected_plan.validation.status is ValidationStatus.VALID_WITH_WARNINGS
    assert not any(
        item.walking_distance_estimated
        for day in actual.selected_plan.days
        for item in day.items
    )
    assert any(
        item.walking_distance_estimated
        for day in estimated.selected_plan.days
        for item in day.items
    )
