from datetime import date, timedelta
from decimal import Decimal

import pytest

from travel_agent.domain.models import PlanStyle
from travel_agent.domain.optimization_models import (
    OptimizationBudget, OptimizationPOI, OptimizationProblem, RouteMatrixEntry,
)
from travel_agent.domain.tool_models import RouteMode
from travel_agent.planning.optimization import ORToolsOptimizationSolver, STYLE_WEIGHTS


def _problem(*, daily_minutes=300, separate_openings=False, asymmetric=False):
    dates = (date(2026, 10, 2), date(2026, 10, 3))
    ids = ("a", "b") if daily_minutes == 130 or separate_openings else ("a", "b", "c", "d")
    pois = tuple(OptimizationPOI(
        id=identifier, name=identifier, categories=("scenic",),
        duration_minutes=120, party_cost=Decimal("45"), preference_value=100,
        data_confidence=1, must_visit=True,
        available_days=(dates[index],) if separate_openings else dates,
    ) for index, identifier in enumerate(ids))
    routes = []
    for origin in ("anchor", *ids):
        for destination in ids:
            if origin == destination:
                continue
            minutes = 10 if origin == "anchor" else (
                2 if {origin, destination} <= {"a", "b"}
                or {origin, destination} <= {"c", "d"} else 60
            )
            if asymmetric and origin == "b" and destination == "a":
                minutes = 45
            routes.append(RouteMatrixEntry(
                origin_id=origin, destination_id=destination,
                duration_minutes=minutes, distance_meters=minutes * 100,
                mode=RouteMode.DRIVING, provider="fixture", data_confidence=1,
            ))
    return OptimizationProblem(
        id="proximity-fixture", dates=dates, anchor_id="anchor", pois=pois,
        route_matrix=tuple(routes), max_daily_activity_minutes=daily_minutes,
        max_daily_walking_meters=5000, max_walking_leg_meters=1500,
        available_minutes_by_day={day: daily_minutes for day in dates},
        weights_by_style=STYLE_WEIGHTS,
        budget=OptimizationBudget(max_solve_ms=3000),
    )


def test_nearby_pairs_are_assigned_together_across_styles():
    result = ORToolsOptimizationSolver().solve(_problem())
    assert len(result.solutions) == len(PlanStyle)
    for solution in result.solutions:
        assert {frozenset(day.poi_ids) for day in solution.days} == {
            frozenset(("a", "b")), frozenset(("c", "d")),
        }
        assert solution.objective_breakdown.proximity_score == 56


@pytest.mark.parametrize("kwargs", [{"daily_minutes": 130}, {"separate_openings": True}])
def test_proximity_cannot_override_daily_capacity_or_opening_dates(kwargs):
    result = ORToolsOptimizationSolver().solve(_problem(**kwargs))
    assert len(result.solutions) == 3
    for solution in result.solutions:
        assert all(len(day.poi_ids) == 1 for day in solution.days)
        assert solution.objective_breakdown.proximity_score == 0


def test_one_way_short_route_does_not_receive_proximity_reward():
    result = ORToolsOptimizationSolver().solve(_problem(asymmetric=True))
    assert len(result.solutions) == 3
    assert all(solution.objective_breakdown.proximity_score == 28 for solution in result.solutions)


def test_fallback_groups_nearby_pois_instead_of_round_robin(hangzhou_trip):
    from travel_agent.domain.models import PlanningPOI, TimeWindow
    from travel_agent.domain.tool_models import POIFacts
    from travel_agent.planning.drafts import prepare_candidate_drafts

    trip = hangzhou_trip.model_copy(update={
        "end_date": hangzhou_trip.start_date + timedelta(days=1), "must_visit": [],
    })
    dates = [trip.start_date, trip.end_date]
    pois = [PlanningPOI(
        facts=POIFacts(
            id=identifier, name=f"景点{identifier}", city="杭州",
            coordinate={"longitude": longitude, "latitude": 30},
            categories=[], provider="fixture", fetched_at=trip.arrival.at,
            data_confidence=1,
        ),
        opening_windows={day: TimeWindow(start="08:00", end="18:00") for day in dates},
        duration_minutes=120, party_cost=Decimal("45"), data_confidence=1,
    ) for identifier, longitude in [("a", 120), ("b", 121), ("c", 120.001), ("d", 121.001)]]
    for draft in prepare_candidate_drafts(trip, pois, replan_round=0):
        assert {frozenset(day.poi_ids) for day in draft.days} == {
            frozenset(("a", "c")), frozenset(("b", "d")),
        }
