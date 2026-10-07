from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import combinations
import logging
from time import perf_counter
from typing import Protocol

from ortools.sat.python import cp_model

from travel_agent.domain.models import (
    Coordinate,
    DayBoundary,
    PlanStyle,
    PlanningPOI,
    StayAnchorResolution,
    TripSpec,
)
from travel_agent.domain.optimization_models import (
    ObjectiveBreakdown,
    ObjectiveWeights,
    OptimizationBudget,
    OptimizationDayAssignment,
    OptimizationPOI,
    OptimizationProblem,
    OptimizationResult,
    OptimizationSolution,
    OptimizationSolveStatus,
    RouteMatrixEntry,
)
from travel_agent.domain.tool_models import RouteMode, RouteQuery, RouteResult, route_key
from travel_agent.planning.drafts import CandidateDraft, DraftDay
from travel_agent.planning.poi_identity import unique_planning_pois
from travel_agent.planning.routing import haversine_distance_meters
from travel_agent.planning.stay import derive_day_boundaries, resolve_stay_anchor


ANCHOR_ID = "__trip_anchor__"
logger = logging.getLogger(__name__)
PROXIMITY_WINDOW_MINUTES = 30
STYLE_ACTIVITY_LIMITS = {
    PlanStyle.RELAXED: 2,
    PlanStyle.BALANCED: 3,
    PlanStyle.EXPLORATION: 4,
}
STYLE_WEIGHTS = {
    PlanStyle.RELAXED: ObjectiveWeights(
        preference=10,
        diversity=2,
        travel=4,
        cost=2,
    ),
    PlanStyle.BALANCED: ObjectiveWeights(
        preference=10,
        diversity=5,
        travel=2,
        cost=1,
    ),
    PlanStyle.EXPLORATION: ObjectiveWeights(
        preference=8,
        diversity=10,
        travel=1,
        cost=0,
    ),
}


class OptimizationTimeoutError(TimeoutError):
    """求解器未在显式预算内给出可行解。"""


class OptimizationSolver(Protocol):
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        raise NotImplementedError


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _is_required(poi: PlanningPOI, trip: TripSpec) -> bool:
    name = _normalize(poi.facts.name)
    return any(
        _normalize(required) in name or name in _normalize(required)
        for required in trip.must_visit
    )


def _preference_value(poi: PlanningPOI, trip: TripSpec) -> int:
    categories = {_normalize(value) for value in poi.facts.categories}
    interests = {_normalize(value) for value in trip.interests}
    avoid = {_normalize(value) for value in trip.avoid}
    return (
        100
        + 400 * len(categories & interests)
        - 500 * len(categories & avoid)
        + round(100 * poi.data_confidence)
    )


def select_optimization_pois(
    trip: TripSpec,
    pois: list[PlanningPOI],
    budget: OptimizationBudget,
) -> list[PlanningPOI]:
    ranked = sorted(
        pois,
        key=lambda poi: (
            0 if _is_required(poi, trip) else 1,
            -_preference_value(poi, trip),
            -poi.data_confidence,
            poi.party_cost if poi.party_cost is not None else Decimal("Infinity"),
            poi.facts.id,
        ),
    )
    return unique_planning_pois(ranked)[: budget.candidate_limit]


def _anchor(
    trip: TripSpec,
    stay_resolution: StayAnchorResolution | None = None,
) -> Coordinate:
    if trip.day_count == 1:
        return trip.arrival.coordinate
    stay = stay_resolution or resolve_stay_anchor(trip, [])
    return stay.anchor.coordinate if stay.anchor is not None else trip.arrival.coordinate


def collect_route_matrix_queries(
    trip: TripSpec,
    pois: list[PlanningPOI],
    *,
    modes: tuple[RouteMode, ...],
    strategy: int,
    max_walking_leg_meters: int,
    stay_resolution: StayAnchorResolution | None = None,
    day_boundaries: tuple[DayBoundary, ...] | None = None,
) -> list[RouteQuery]:
    queries: list[RouteQuery] = []
    seen: set[str] = set()
    stay = stay_resolution or resolve_stay_anchor(trip, pois)
    boundaries = day_boundaries or derive_day_boundaries(trip, stay)
    start_coordinates = {
        (boundary.start_anchor.coordinate.longitude, boundary.start_anchor.coordinate.latitude):
        boundary.start_anchor.coordinate
        for boundary in boundaries
    }.values()
    end_coordinates = {
        (boundary.end_anchor.coordinate.longitude, boundary.end_anchor.coordinate.latitude):
        boundary.end_anchor.coordinate
        for boundary in boundaries
    }.values()

    def append(query: RouteQuery) -> None:
        key = route_key(query)
        if key not in seen:
            seen.add(key)
            queries.append(query)

    for anchor in start_coordinates:
      for destination in pois:
        for mode in modes:
            if (
                mode is RouteMode.WALKING
                and haversine_distance_meters(
                    anchor,
                    destination.facts.coordinate,
                )
                > max_walking_leg_meters * 1.25
            ):
                continue
            append(
                RouteQuery(
                    origin=anchor,
                    destination=destination.facts.coordinate,
                    destination_poi_id=destination.facts.id,
                    mode=mode,
                    strategy=strategy if mode is RouteMode.DRIVING else 0,
                )
            )
    for origin in pois:
        for destination in pois:
            if origin.facts.id == destination.facts.id:
                continue
            for mode in modes:
                if (
                    mode is RouteMode.WALKING
                    and haversine_distance_meters(
                        origin.facts.coordinate,
                        destination.facts.coordinate,
                    )
                    > max_walking_leg_meters * 1.25
                ):
                    continue
                append(
                    RouteQuery(
                        origin=origin.facts.coordinate,
                        destination=destination.facts.coordinate,
                        origin_poi_id=origin.facts.id,
                        destination_poi_id=destination.facts.id,
                        mode=mode,
                        strategy=strategy if mode is RouteMode.DRIVING else 0,
                    )
                )
    for origin in pois:
        for anchor in end_coordinates:
            for mode in modes:
                if (
                    mode is RouteMode.WALKING
                    and haversine_distance_meters(
                        origin.facts.coordinate, anchor
                    ) > max_walking_leg_meters * 1.25
                ):
                    continue
                append(
                    RouteQuery(
                        origin=origin.facts.coordinate,
                        destination=anchor,
                        origin_poi_id=origin.facts.id,
                        mode=mode,
                        strategy=strategy if mode is RouteMode.DRIVING else 0,
                    )
                )
    return queries


def _available_minutes(trip: TripSpec, day: date) -> int:
    timezone = trip.arrival.at.tzinfo
    start = datetime.combine(day, trip.daily_start, tzinfo=timezone)
    end = datetime.combine(day, trip.daily_end, tzinfo=timezone)
    if day == trip.arrival.at.date():
        start = max(start, trip.arrival.at + timedelta(minutes=60))
    if day == trip.departure.at.date():
        end = min(end, trip.departure.at - timedelta(minutes=90))
    return max(0, round((end - start).total_seconds() / 60))


def build_optimization_problem(
    trip: TripSpec,
    pois: list[PlanningPOI],
    routes: dict[str, RouteResult],
    budget: OptimizationBudget,
    *,
    modes: tuple[RouteMode, ...],
    strategy: int,
    max_walking_leg_meters: int,
    stay_resolution: StayAnchorResolution | None = None,
    day_boundaries: tuple[DayBoundary, ...] | None = None,
) -> OptimizationProblem:
    dates = tuple(
        trip.start_date + timedelta(days=index) for index in range(trip.day_count)
    )
    optimization_pois = tuple(
        OptimizationPOI(
            id=poi.facts.id,
            name=poi.facts.name,
            categories=tuple(sorted(set(poi.facts.categories))),
            duration_minutes=poi.duration_minutes,
            party_cost=poi.party_cost,
            preference_value=_preference_value(poi, trip),
            data_confidence=poi.data_confidence,
            must_visit=_is_required(poi, trip),
            available_days=tuple(
                day
                for day in dates
                if day in poi.opening_windows
                and _available_minutes(trip, day) >= poi.duration_minutes
            ),
        )
        for poi in pois
    )
    entries: list[RouteMatrixEntry] = []
    anchor = _anchor(trip, stay_resolution)
    logical_queries: list[RouteQuery] = []
    for destination in pois:
        for mode in modes:
            if mode is RouteMode.WALKING and haversine_distance_meters(
                anchor, destination.facts.coordinate
            ) > max_walking_leg_meters * 1.25:
                continue
            logical_queries.append(RouteQuery(
                origin=anchor,
                destination=destination.facts.coordinate,
                destination_poi_id=destination.facts.id,
                mode=mode,
                strategy=strategy if mode is RouteMode.DRIVING else 0,
            ))
    for origin in pois:
        for destination in pois:
            if origin.facts.id == destination.facts.id:
                continue
            for mode in modes:
                if mode is RouteMode.WALKING and haversine_distance_meters(
                    origin.facts.coordinate, destination.facts.coordinate
                ) > max_walking_leg_meters * 1.25:
                    continue
                logical_queries.append(RouteQuery(
                    origin=origin.facts.coordinate,
                    destination=destination.facts.coordinate,
                    origin_poi_id=origin.facts.id,
                    destination_poi_id=destination.facts.id,
                    mode=mode,
                    strategy=strategy if mode is RouteMode.DRIVING else 0,
                ))
    for query in logical_queries:
        result = routes[route_key(query)]
        entries.append(
            RouteMatrixEntry(
                origin_id=query.origin_poi_id or ANCHOR_ID,
                destination_id=query.destination_poi_id or ANCHOR_ID,
                duration_minutes=result.duration_minutes,
                distance_meters=result.distance_meters,
                mode=result.mode,
                provider=result.provider,
                data_confidence=result.data_confidence,
            )
        )
    return OptimizationProblem(
        id=f"{trip.destination}-{trip.start_date.isoformat()}-{len(pois)}",
        dates=dates,
        anchor_id=ANCHOR_ID,
        pois=optimization_pois,
        route_matrix=tuple(entries),
        total_budget=trip.total_budget,
        max_daily_activity_minutes=trip.mobility.max_daily_activity_minutes,
        max_daily_walking_meters=trip.mobility.max_daily_walking_meters,
        max_walking_leg_meters=max_walking_leg_meters,
        available_minutes_by_day={day: _available_minutes(trip, day) for day in dates},
        weights_by_style=STYLE_WEIGHTS,
        budget=budget,
    )


def _ordered_day(
    poi_ids: list[str],
    problem: OptimizationProblem,
    matrix: dict[tuple[str, str], RouteMatrixEntry],
) -> tuple[str, ...]:
    remaining = set(poi_ids)
    ordered: list[str] = []
    current = problem.anchor_id
    while remaining:
        next_id = min(
            remaining,
            key=lambda poi_id: (
                matrix[(current, poi_id)].duration_minutes,
                matrix[(current, poi_id)].distance_meters,
                poi_id,
            ),
        )
        ordered.append(next_id)
        remaining.remove(next_id)
        current = next_id
    return tuple(ordered)


class ORToolsOptimizationSolver:
    name = "ortools-cp-sat-assignment-v1"

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        started = perf_counter()
        raw_matrix = {
            (entry.origin_id, entry.destination_id, entry.mode): entry
            for entry in problem.route_matrix
        }
        pairs = {
            (entry.origin_id, entry.destination_id)
            for entry in problem.route_matrix
        }
        matrix: dict[tuple[str, str], RouteMatrixEntry] = {}
        for origin_id, destination_id in pairs:
            walking = raw_matrix.get(
                (origin_id, destination_id, RouteMode.WALKING)
            )
            driving = raw_matrix.get(
                (origin_id, destination_id, RouteMode.DRIVING)
            )
            if (
                walking is not None
                and walking.distance_meters <= problem.max_walking_leg_meters
            ):
                matrix[(origin_id, destination_id)] = walking
            elif driving is not None:
                matrix[(origin_id, destination_id)] = driving
            elif walking is not None:
                matrix[(origin_id, destination_id)] = walking
            else:
                raise ValueError("route matrix pair has no supported mode")
        poi_by_id = {poi.id: poi for poi in problem.pois}
        # 双向均便捷才奖励同天；这是分组评分，不是实际路线总时间。
        proximity_pairs = {}
        for left, right in combinations(problem.pois, 2):
            forward = matrix.get((left.id, right.id))
            backward = matrix.get((right.id, left.id))
            if forward is None or backward is None:
                continue
            score = max(0, PROXIMITY_WINDOW_MINUTES - max(
                forward.duration_minutes, backward.duration_minutes,
            ))
            if score:
                proximity_pairs[(left.id, right.id)] = score
        logger.info(
            "optimization.proximity_pairs | problem_id=%s pair_count=%s window_minutes=%s",
            problem.id, len(proximity_pairs), PROXIMITY_WINDOW_MINUTES,
        )
        styles = tuple(PlanStyle)[: problem.budget.variant_count]
        solutions: list[OptimizationSolution] = []
        statuses: list[int] = []
        search_states = 0
        per_variant_seconds = (
            problem.budget.max_solve_ms / 1000 / max(1, len(styles))
        )

        for style_index, style in enumerate(styles):
            model = cp_model.CpModel()
            assignments = {
                (poi.id, day): model.new_bool_var(f"x_{poi_index}_{day_index}")
                for poi_index, poi in enumerate(problem.pois)
                for day_index, day in enumerate(problem.dates)
            }
            for poi in problem.pois:
                selected = [assignments[(poi.id, day)] for day in problem.dates]
                model.add(sum(selected) == 1 if poi.must_visit else sum(selected) <= 1)
                for day in problem.dates:
                    if day not in poi.available_days:
                        model.add(assignments[(poi.id, day)] == 0)

            all_assignments = list(assignments.values())
            model.add(sum(all_assignments) >= 1)
            for day in problem.dates:
                day_vars = [assignments[(poi.id, day)] for poi in problem.pois]
                model.add(sum(day_vars) <= STYLE_ACTIVITY_LIMITS[style])
                model.add(
                    sum(
                        (
                            poi.duration_minutes
                            + matrix[(problem.anchor_id, poi.id)].duration_minutes
                        )
                        * assignments[(poi.id, day)]
                        for poi in problem.pois
                    )
                    <= min(
                        problem.max_daily_activity_minutes,
                        problem.available_minutes_by_day[day],
                    )
                )
                model.add(
                    sum(
                        (
                            matrix[(problem.anchor_id, poi.id)].distance_meters
                            if matrix[(problem.anchor_id, poi.id)].mode
                            is RouteMode.WALKING
                            else 0
                        )
                        * assignments[(poi.id, day)]
                        for poi in problem.pois
                    )
                    <= problem.max_daily_walking_meters
                )

            if problem.total_budget is not None:
                budget_cents = round(problem.total_budget * 100)
                model.add(
                    sum(
                        round((poi.party_cost or Decimal("0")) * 100)
                        * assignments[(poi.id, day)]
                        for poi in problem.pois
                        for day in problem.dates
                    )
                    <= budget_cents
                )

            categories = sorted(
                {category for poi in problem.pois for category in poi.categories}
            )
            category_vars = {}
            for day_index, day in enumerate(problem.dates):
                for category_index, category in enumerate(categories):
                    category_var = model.new_bool_var(
                        f"category_{day_index}_{category_index}"
                    )
                    matching = [
                        assignments[(poi.id, day)]
                        for poi in problem.pois
                        if category in poi.categories
                    ]
                    if matching:
                        model.add(category_var <= sum(matching))
                    else:
                        model.add(category_var == 0)
                    category_vars[(day, category)] = category_var

            weights = problem.weights_by_style[style]
            same_day_vars = {}
            for (left, right), score in proximity_pairs.items():
                for day_index, day in enumerate(problem.dates):
                    together = model.new_bool_var(f"together_{left}_{right}_{day_index}")
                    left_var = assignments[(left, day)]
                    right_var = assignments[(right, day)]
                    model.add(together <= left_var)
                    model.add(together <= right_var)
                    model.add(together >= left_var + right_var - 1)
                    same_day_vars[(left, right, day)] = together
            proximity_term = weights.proximity * sum(
                proximity_pairs[(left, right)] * together
                for (left, right, _day), together in same_day_vars.items()
            )
            preference_term = sum(
                weights.preference
                * poi.preference_value
                * assignments[(poi.id, day)]
                for poi in problem.pois
                for day in problem.dates
            )
            diversity_term = weights.diversity * sum(category_vars.values())
            travel_term = sum(
                weights.travel
                * matrix[(problem.anchor_id, poi.id)].duration_minutes
                * assignments[(poi.id, day)]
                for poi in problem.pois
                for day in problem.dates
            )
            cost_term = sum(
                weights.cost
                * round(poi.party_cost or Decimal("0"))
                * assignments[(poi.id, day)]
                for poi in problem.pois
                for day in problem.dates
            )
            model.maximize(preference_term + diversity_term + proximity_term - travel_term - cost_term)

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = per_variant_seconds
            solver.parameters.max_number_of_conflicts = (
                problem.budget.max_search_states // max(1, len(styles))
            )
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = style_index
            status = solver.solve(model)
            statuses.append(status)
            search_states += solver.num_branches
            if status == cp_model.UNKNOWN:
                raise OptimizationTimeoutError(
                    f"{style.value} optimization exceeded solve budget"
                )
            if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                continue

            day_assignments: list[OptimizationDayAssignment] = []
            selected_ids: list[str] = []
            total_travel = 0
            total_walking = 0
            for day in problem.dates:
                day_ids = [
                    poi.id
                    for poi in problem.pois
                    if solver.value(assignments[(poi.id, day)])
                ]
                ordered = _ordered_day(day_ids, problem, matrix)
                day_assignments.append(
                    OptimizationDayAssignment(date=day, poi_ids=ordered)
                )
                selected_ids.extend(ordered)
                current = problem.anchor_id
                for poi_id in ordered:
                    route = matrix[(current, poi_id)]
                    total_travel += route.duration_minutes
                    if route.mode is RouteMode.WALKING:
                        total_walking += route.distance_meters
                    current = poi_id

            selected = [poi_by_id[poi_id] for poi_id in selected_ids]
            proximity_score = sum(
                proximity_pairs[(left, right)] * solver.value(together)
                for (left, right, _day), together in same_day_vars.items()
            )
            logger.info(
                "optimization.proximity_assignment | problem_id=%s style=%s score=%s days=%s",
                problem.id, style.value, proximity_score,
                [(str(day.date), day.poi_ids) for day in day_assignments],
            )
            solutions.append(
                OptimizationSolution(
                    style=style,
                    days=tuple(day_assignments),
                    objective_value=round(solver.objective_value, 3),
                    objective_breakdown=ObjectiveBreakdown(
                        proximity_score=proximity_score,
                        preference_value=sum(poi.preference_value for poi in selected),
                        diversity_count=len(
                            {category for poi in selected for category in poi.categories}
                        ),
                        travel_minutes=total_travel,
                        walking_meters=total_walking,
                        known_cost=sum(
                            (poi.party_cost or Decimal("0") for poi in selected),
                            start=Decimal("0"),
                        ),
                    ),
                )
            )

        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        status = (
            OptimizationSolveStatus.OPTIMAL
            if solutions and all(item == cp_model.OPTIMAL for item in statuses)
            else OptimizationSolveStatus.FEASIBLE
            if solutions
            else OptimizationSolveStatus.INFEASIBLE
        )
        return OptimizationResult(
            status=status,
            solver=self.name,
            solutions=tuple(solutions),
            elapsed_ms=elapsed_ms,
            search_states=search_states,
        )


def drafts_from_optimization(result: OptimizationResult) -> list[CandidateDraft]:
    return [
        CandidateDraft(
            id=f"{solution.style.value}-opt-r0",
            style=solution.style,
            days=tuple(
                DraftDay(date=day.date, poi_ids=day.poi_ids) for day in solution.days
            ),
        )
        for solution in result.solutions
    ]


def degraded_result(
    drafts: list[CandidateDraft],
    *,
    reason: str,
    elapsed_ms: float,
) -> OptimizationResult:
    return OptimizationResult(
        status=OptimizationSolveStatus.DEGRADED,
        solver="deterministic-nearest-neighbor-v0.5",
        solutions=tuple(
            OptimizationSolution(
                style=draft.style,
                days=tuple(
                    OptimizationDayAssignment(date=day.date, poi_ids=day.poi_ids)
                    for day in draft.days
                ),
                objective_value=0,
                objective_breakdown=ObjectiveBreakdown(
                    preference_value=0,
                    diversity_count=0,
                    travel_minutes=0,
                    walking_meters=0,
                    known_cost=Decimal("0"),
                ),
            )
            for draft in drafts
        ),
        elapsed_ms=elapsed_ms,
        search_states=0,
        degraded_reason=reason,
    )
