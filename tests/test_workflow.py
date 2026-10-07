from __future__ import annotations

from decimal import Decimal

import pytest

from travel_agent.domain.models import (
    PlanningRequest,
    ValidationResult,
    ValidationStatus,
    Violation,
    ViolationSeverity,
)
from travel_agent.graph.workflow import run_planning, select_best_candidate


@pytest.mark.asyncio
async def test_workflow_builds_valid_plan(hangzhou_trip, mock_workflow):
    response = await run_planning(
        mock_workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=2),
        thread_id="test-valid-plan",
    )

    assert response.status == "completed"
    assert response.selected_plan is not None
    assert response.selected_plan.validation is not None
    assert response.selected_plan.validation.valid
    assert response.selected_plan.validation.status is ValidationStatus.VALID
    assert all(
        not item.walking_distance_estimated
        for day in response.selected_plan.days
        for item in day.items
    )
    scheduled_names = {
        item.name
        for day in response.selected_plan.days
        for item in day.items
    }
    assert "灵隐寺" in scheduled_names


@pytest.mark.asyncio
async def test_workflow_returns_infeasible_when_required_poi_exceeds_budget(
    hangzhou_trip,
    mock_workflow,
):
    constrained = hangzhou_trip.model_copy(
        update={"total_budget": Decimal("10")}
    )
    response = await run_planning(
        mock_workflow,
        PlanningRequest(trip=constrained, max_replan_rounds=1),
        thread_id="test-infeasible-plan",
    )

    assert response.status == "infeasible"
    assert response.selected_plan is None
    assert response.iterations == 0


@pytest.mark.asyncio
async def test_fully_valid_candidate_ranks_ahead_of_higher_scored_warning_candidate(
    hangzhou_trip,
    mock_workflow,
):
    response = await run_planning(
        mock_workflow,
        PlanningRequest(trip=hangzhou_trip, max_replan_rounds=0),
        thread_id="test-validation-rank",
    )
    base = response.candidates[0]
    warning_candidate = base.model_copy(
        update={
            "id": "warning-high-score",
            "score": 1.0,
            "validation": ValidationResult.from_violations(
                [
                    Violation(
                        type="unverified",
                        severity=ViolationSeverity.WARNING,
                        message="事实待确认",
                    )
                ]
            ),
        }
    )
    valid_candidate = base.model_copy(
        update={
            "id": "valid-low-score",
            "score": -1.0,
            "validation": ValidationResult.from_violations([]),
        }
    )

    selected = select_best_candidate([warning_candidate, valid_candidate])

    assert selected.id == "valid-low-score"


def test_graph_exposes_tool_use_and_replan_nodes(mock_workflow):
    graph = mock_workflow.get_graph()
    node_names = set(graph.nodes)
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert {
        "build_search_plan",
        "load_pois",
            "resolve_poi_facts",
            "resolve_stay_anchor",
            "derive_day_boundaries",
        "build_route_matrix",
        "build_optimization_problem",
        "solve_candidate_variants",
        "materialize_optimized_candidates",
        "materialize_candidates",
        "validate_candidates",
        "select_repair_target",
        "analyze_violations",
        "build_repair_plan",
        "apply_local_repair",
        "collect_delta_routes",
        "load_delta_routes",
        "select_best",
        "mark_infeasible",
        "prepare_critic_context",
        "soft_constraint_critic",
        "validate_critic_evidence",
        "quality_gate",
        "compile_soft_repair_plan",
        "apply_soft_repair",
        "collect_soft_delta_routes",
        "load_soft_delta_routes",
        "materialize_soft_candidate",
        "restore_soft_baseline",
        "compare_soft_repair",
        "select_by_quality",
        "explain_selection",
    } <= node_names
    assert {
        ("__start__", "execution_budget_guard"),
        ("execution_budget_guard", "build_search_plan"),
        ("build_search_plan", "load_pois"),
        ("load_pois", "resolve_poi_facts"),
            ("resolve_poi_facts", "resolve_stay_anchor"),
            ("resolve_stay_anchor", "derive_day_boundaries"),
            ("derive_day_boundaries", "build_route_matrix"),
        ("build_route_matrix", "build_optimization_problem"),
        ("build_optimization_problem", "solve_candidate_variants"),
        ("solve_candidate_variants", "materialize_optimized_candidates"),
        ("materialize_optimized_candidates", "validate_candidates"),
        ("materialize_candidates", "validate_candidates"),
        ("select_repair_target", "analyze_violations"),
        ("apply_local_repair", "collect_delta_routes"),
        ("load_delta_routes", "materialize_candidates"),
        ("select_best", "__end__"),
        ("mark_infeasible", "__end__"),
        ("prepare_critic_context", "soft_constraint_critic"),
        ("soft_constraint_critic", "validate_critic_evidence"),
        ("apply_soft_repair", "collect_soft_delta_routes"),
        ("load_soft_delta_routes", "materialize_soft_candidate"),
        ("materialize_soft_candidate", "validate_candidates"),
        ("restore_soft_baseline", "select_by_quality"),
        ("compare_soft_repair", "select_by_quality"),
        ("select_by_quality", "explain_selection"),
        ("explain_selection", "__end__"),
    } <= edges
    assert "replan" not in node_names

