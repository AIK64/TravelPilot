from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
from statistics import mean

from travel_agent.domain.models import (
    Coordinate,
    DayBoundary,
    DayPlan,
    ItemType,
    PlanCandidate,
    PlanItem,
    PlanMetrics,
    PlanRouteLeg,
    PlanningAssumption,
    PlanningPOI,
    StayAnchorResolution,
    TripSpec,
)
from travel_agent.domain.tool_models import (
    RouteMode,
    RouteQuery,
    RouteResult,
    ValueSource,
    route_key,
)
from travel_agent.planning.drafts import (
    CandidateDraft,
    DraftDay,
    MissingPlanningPOI,
    collect_route_queries,
)
from travel_agent.planning.routing import haversine_distance_meters
from travel_agent.planning.stay import (
    derive_day_boundaries,
    resolve_stay_anchor,
    stay_assumption,
)


def _normalize(value: str) -> str:
    return value.strip().casefold()


class MissingRouteResult(LookupError):
    """物化日程所需的标准化路线结果不存在。"""

    def __init__(self, missing_route_key: str) -> None:
        self.route_key = missing_route_key
        super().__init__(f"missing route result: {missing_route_key}")


def _route_query(
    origin: Coordinate,
    destination: Coordinate,
    origin_poi_id: str | None,
    destination_poi_id: str | None,
    route_strategy: int,
    route_mode: RouteMode,
) -> RouteQuery:
    return RouteQuery(
        origin=origin,
        destination=destination,
        origin_poi_id=origin_poi_id,
        destination_poi_id=destination_poi_id,
        mode=route_mode,
        strategy=route_strategy if route_mode is RouteMode.DRIVING else 0,
    )


def _append_unique_assumptions(
    target: list[PlanningAssumption],
    additions: list[PlanningAssumption],
) -> None:
    for assumption in additions:
        if assumption not in target:
            target.append(assumption)


def _route_provenance_fact(
    routes: list[RouteResult],
    total_travel_minutes: int,
) -> str:
    if not routes:
        return f"路线来源 无（未调用路线工具），交通 {total_travel_minutes} 分钟"
    providers = sorted({route.provider for route in routes})
    provider_text = "/".join(providers)
    if providers == ["mock"]:
        result_kind = "本地估算"
    elif "mock" in providers:
        result_kind = "含 mock 本地估算的 Provider 标准化结果"
    else:
        result_kind = "Provider 标准化结果"
    confidence = mean(route.data_confidence for route in routes)
    return (
        f"路线来源 {provider_text}（{result_kind}），"
        f"路线置信度 {confidence:.0%}，交通 {total_travel_minutes} 分钟"
    )


def _materialize_day(
    trip: TripSpec,
    draft_day: DraftDay,
    poi_by_id: dict[str, PlanningPOI],
    routes: dict[str, RouteResult],
    route_strategy: int,
    route_mode: RouteMode,
    route_modes: tuple[RouteMode, ...] | None,
    max_walking_leg_meters: int,
    boundary: DayBoundary,
) -> tuple[DayPlan, list[PlanningPOI], list[RouteResult]]:
    current_date = draft_day.date
    timezone = trip.arrival.at.tzinfo
    day_start = datetime.combine(current_date, trip.daily_start, tzinfo=timezone)
    day_end = datetime.combine(current_date, trip.daily_end, tzinfo=timezone)
    if current_date == trip.arrival.at.date():
        day_start = max(day_start, trip.arrival.at + timedelta(minutes=60))
    if current_date == trip.departure.at.date():
        day_end = min(day_end, trip.departure.at - timedelta(minutes=90))

    current_coordinate = boundary.start_anchor.coordinate
    current_poi_id: str | None = None
    current_time = day_start
    items: list[PlanItem] = []
    scheduled_pois: list[PlanningPOI] = []
    used_routes: list[RouteResult] = []
    known_cost = Decimal("0")
    unknown_cost_count = 0
    travel_minutes_total = 0
    walking_total = 0
    final_route: RouteResult | None = None
    final_walking_meters = 0
    final_walking_estimated = False
    for poi_id in draft_day.poi_ids:
        poi = poi_by_id.get(poi_id)
        if poi is None:
            raise MissingPlanningPOI(poi_id)
        active_modes = route_modes or (route_mode,)
        route_by_mode: dict[RouteMode, RouteResult] = {}
        for active_mode in active_modes:
            if (
                active_mode is RouteMode.WALKING
                and haversine_distance_meters(
                    current_coordinate,
                    poi.facts.coordinate,
                )
                > max_walking_leg_meters * 1.25
            ):
                continue
            query = _route_query(
                current_coordinate,
                poi.facts.coordinate,
                current_poi_id,
                poi_id,
                route_strategy,
                active_mode,
            )
            key = route_key(query)
            result = routes.get(key)
            if result is None:
                raise MissingRouteResult(key)
            route_by_mode[active_mode] = result

        walking_route = route_by_mode.get(RouteMode.WALKING)
        driving_route = route_by_mode.get(RouteMode.DRIVING)
        if (
            walking_route is not None
            and driving_route is not None
            and (
                walking_route.distance_meters > max_walking_leg_meters
                or walking_total + walking_route.distance_meters
                > trip.mobility.max_daily_walking_meters
            )
        ):
            route = driving_route
            walking_meters = 0
            walking_is_estimated = False
        elif walking_route is not None:
            route = walking_route
            walking_meters = route.distance_meters
            walking_is_estimated = False
        else:
            assert driving_route is not None
            route = driving_route
            hybrid_routes_enabled = (
                route_modes is not None and RouteMode.WALKING in route_modes
            )
            walking_meters = (
                0
                if hybrid_routes_enabled
                else min(round(route.distance_meters * 0.12), 2_000)
            )
            walking_is_estimated = not hybrid_routes_enabled

        arrival_time = current_time + timedelta(minutes=route.duration_minutes)
        opening_window = poi.opening_windows[current_date]
        opening = datetime.combine(
            current_date, opening_window.start, tzinfo=timezone
        )
        closing = datetime.combine(
            current_date, opening_window.end, tzinfo=timezone
        )
        start_at = max(arrival_time, opening)
        end_at = start_at + timedelta(minutes=poi.duration_minutes)
        if end_at > day_end or end_at > closing:
            break

        # 只有在最后一站仍能按时到达当日日终锚点时，才接受该活动。
        end_routes: dict[RouteMode, RouteResult] = {}
        for active_mode in active_modes:
            if (
                active_mode is RouteMode.WALKING
                and haversine_distance_meters(
                    poi.facts.coordinate, boundary.end_anchor.coordinate
                ) > max_walking_leg_meters * 1.25
            ):
                continue
            end_query = _route_query(
                poi.facts.coordinate,
                boundary.end_anchor.coordinate,
                poi.facts.id,
                None,
                route_strategy,
                active_mode,
            )
            end_key = route_key(end_query)
            end_result = routes.get(end_key)
            if end_result is None:
                raise MissingRouteResult(end_key)
            end_routes[active_mode] = end_result
        end_walking = end_routes.get(RouteMode.WALKING)
        end_driving = end_routes.get(RouteMode.DRIVING)
        if (
            end_walking is not None
            and end_driving is not None
            and (
                end_walking.distance_meters > max_walking_leg_meters
                or walking_total + walking_meters + end_walking.distance_meters
                > trip.mobility.max_daily_walking_meters
            )
        ):
            candidate_end_route = end_driving
            candidate_end_walking = 0
            candidate_end_estimated = False
        elif end_walking is not None:
            candidate_end_route = end_walking
            candidate_end_walking = end_walking.distance_meters
            candidate_end_estimated = False
        else:
            assert end_driving is not None
            candidate_end_route = end_driving
            hybrid_routes_enabled = (
                route_modes is not None and RouteMode.WALKING in route_modes
            )
            candidate_end_walking = (
                0 if hybrid_routes_enabled
                else min(round(end_driving.distance_meters * 0.12), 2_000)
            )
            candidate_end_estimated = not hybrid_routes_enabled
        if end_at + timedelta(minutes=candidate_end_route.duration_minutes) > day_end:
            break

        items.append(
            PlanItem(
                type=ItemType.ACTIVITY,
                name=poi.facts.name,
                poi_id=poi.facts.id,
                coordinate=poi.facts.coordinate,
                start_at=start_at,
                end_at=end_at,
                travel_from_previous_minutes=route.duration_minutes,
                distance_from_previous_meters=route.distance_meters,
                estimated_cost=poi.party_cost,
                walking_distance_estimated=walking_is_estimated,
            )
        )
        scheduled_pois.append(poi)
        used_routes.append(route)
        current_time = end_at
        current_coordinate = poi.facts.coordinate
        current_poi_id = poi.facts.id
        travel_minutes_total += route.duration_minutes
        walking_total += walking_meters
        final_route = candidate_end_route
        final_walking_meters = candidate_end_walking
        final_walking_estimated = candidate_end_estimated
        if poi.party_cost is None:
            unknown_cost_count += 1
        else:
            known_cost += poi.party_cost

    end_leg = None
    if final_route is not None and scheduled_pois:
        used_routes.append(final_route)
        travel_minutes_total += final_route.duration_minutes
        walking_total += final_walking_meters
        end_leg = PlanRouteLeg(
            origin_name=scheduled_pois[-1].facts.name,
            destination_name=boundary.end_anchor.name,
            duration_minutes=final_route.duration_minutes,
            distance_meters=final_route.distance_meters,
            mode=final_route.mode.value,
            provider=final_route.provider,
            data_confidence=final_route.data_confidence,
            walking_distance_meters=final_walking_meters,
            walking_distance_estimated=final_walking_estimated,
        )

    categories = [
        category
        for poi in scheduled_pois
        for category in poi.facts.categories
    ]
    theme = (
        " / ".join(name for name, _ in Counter(categories).most_common(2))
        or "自由活动"
    )
    activity_minutes = sum(
        round((item.end_at - item.start_at).total_seconds() / 60)
        for item in items
    )
    fatigue = min(
        1.0,
        (activity_minutes + travel_minutes_total)
        / trip.mobility.max_daily_activity_minutes,
    )
    return (
        DayPlan(
            date=current_date,
            theme=theme,
            primary_area=trip.destination if items else "未安排",
            items=items,
            known_estimated_cost=known_cost,
            unknown_cost_item_count=unknown_cost_count,
            total_travel_minutes=travel_minutes_total,
            walking_distance_meters=walking_total,
            fatigue_score=round(fatigue, 3),
            start_anchor=boundary.start_anchor,
            end_anchor=boundary.end_anchor,
            end_leg=end_leg,
        ),
        scheduled_pois,
        used_routes,
    )


def _materialize_candidate(
    trip: TripSpec,
    draft: CandidateDraft,
    poi_by_id: dict[str, PlanningPOI],
    routes: dict[str, RouteResult],
    route_strategy: int,
    route_mode: RouteMode,
    route_modes: tuple[RouteMode, ...] | None,
    max_walking_leg_meters: int,
    stay_resolution: StayAnchorResolution,
    day_boundaries: tuple[DayBoundary, ...],
    all_pois: list[PlanningPOI],
) -> PlanCandidate:
    days: list[DayPlan] = []
    scheduled_pois: list[PlanningPOI] = []
    used_routes: list[RouteResult] = []
    assumptions: list[PlanningAssumption] = []
    walking_dates: list[date] = []
    boundary_by_date = {boundary.date: boundary for boundary in day_boundaries}
    stay_note = stay_assumption(trip, stay_resolution, all_pois)
    if stay_note is not None:
        assumptions.append(stay_note)
    for draft_day in draft.days:
        day, day_pois, day_routes = _materialize_day(
            trip,
            draft_day,
            poi_by_id,
            routes,
            route_strategy,
            route_mode,
            route_modes,
            max_walking_leg_meters,
            boundary_by_date[draft_day.date],
        )
        days.append(day)
        scheduled_pois.extend(day_pois)
        used_routes.extend(day_routes)
        if day.items:
            walking_dates.append(day.date)
        for poi in day_pois:
            _append_unique_assumptions(assumptions, poi.assumptions)

    estimated_walking_routes = (
        [route for route in used_routes if route.mode is not RouteMode.WALKING]
        if RouteMode.WALKING not in (route_modes or (route_mode,))
        else []
    )
    if estimated_walking_routes:
        _append_unique_assumptions(
            assumptions,
            [
                PlanningAssumption(
                    field="walking_distance",
                    value="min(round(driving_distance_meters * 0.12), 2000)",
                    reason=(
                        "基于路线 Provider 标准化距离派生接驳步行估算，"
                        "非步行路线事实"
                    ),
                    source=ValueSource.DEFAULT,
                    affected_dates=walking_dates,
                    created_at=min(
                        route.fetched_at for route in estimated_walking_routes
                    ),
                )
            ],
        )

    interests = {_normalize(value) for value in trip.interests}
    matching = sum(
        bool(
            interests
            & {
                _normalize(value)
                for value in poi.facts.categories
            }
        )
        for poi in scheduled_pois
    )
    preference_match = matching / len(scheduled_pois) if scheduled_pois else 0.0
    category_count = len(
        {
            category
            for poi in scheduled_pois
            for category in poi.facts.categories
        }
    )
    diversity = min(1.0, category_count / 5)
    confidence_values = [poi.data_confidence for poi in scheduled_pois]
    confidence_values.extend(route.data_confidence for route in used_routes)
    confidence = mean(confidence_values) if confidence_values else 0.0
    total_travel = sum(day.total_travel_minutes for day in days)
    walking = sum(day.walking_distance_meters for day in days)
    known_cost = sum(
        (day.known_estimated_cost for day in days), start=Decimal("0")
    )
    unknown_cost_count = sum(day.unknown_cost_item_count for day in days)
    fatigue = mean([day.fatigue_score for day in days]) if days else 0.0
    metrics = PlanMetrics(
        preference_match=round(preference_match, 3),
        diversity=round(diversity, 3),
        data_confidence=round(confidence, 3),
        total_travel_minutes=total_travel,
        walking_distance_meters=walking,
        known_estimated_cost=known_cost,
        unknown_cost_item_count=unknown_cost_count,
        fatigue_score=round(fatigue, 3),
    )
    budget_risk = (
        min(1.0, float(known_cost / trip.total_budget))
        if trip.total_budget
        else 0.0
    )
    warning_types = {
        assumption.field
        for assumption in assumptions
        if assumption.source is ValueSource.DEFAULT
    }
    warning_risk = min(
        1.0,
        (len(warning_types) + int(unknown_cost_count > 0)) / 4,
    )
    score = (
        0.35 * metrics.preference_match
        + 0.20 * metrics.diversity
        + 0.15 * metrics.data_confidence
        - 0.10 * min(1.0, total_travel / max(1, trip.day_count * 180))
        - 0.10 * metrics.fatigue_score
        - 0.10 * budget_risk
        - 0.10 * warning_risk
    )
    return PlanCandidate(
        id=draft.id,
        style=draft.style,
        days=days,
        metrics=metrics,
        score=round(score, 4),
        reason_facts=[
            f"兴趣匹配度 {metrics.preference_match:.0%}",
            _route_provenance_fact(used_routes, metrics.total_travel_minutes),
            f"已知预计费用 {metrics.known_estimated_cost} 元",
            f"未知费用活动 {metrics.unknown_cost_item_count} 个",
            f"数据置信度 {metrics.data_confidence:.0%}",
        ],
        assumptions=assumptions,
    )


def materialize_candidates(
    trip: TripSpec,
    drafts: list[CandidateDraft],
    pois: list[PlanningPOI],
    routes: dict[str, RouteResult],
    route_strategy: int = 32,
    route_mode: RouteMode = RouteMode.DRIVING,
    route_modes: tuple[RouteMode, ...] | None = None,
    max_walking_leg_meters: int = 1_500,
    stay_resolution: StayAnchorResolution | None = None,
    day_boundaries: tuple[DayBoundary, ...] | None = None,
) -> list[PlanCandidate]:
    """Phase 2：只消费标准化 RouteResult 物化日程与指标。"""
    resolution = stay_resolution or resolve_stay_anchor(trip, pois)
    boundaries = day_boundaries or derive_day_boundaries(trip, resolution)
    poi_by_id = {poi.facts.id: poi for poi in pois}
    for query in collect_route_queries(
        trip,
        drafts,
        pois,
        route_strategy=route_strategy,
        route_mode=route_mode,
        route_modes=route_modes,
        max_walking_leg_meters=max_walking_leg_meters,
        stay_resolution=resolution,
        day_boundaries=boundaries,
    ):
        key = route_key(query)
        if key not in routes:
            raise MissingRouteResult(key)
    return [
        _materialize_candidate(
            trip,
            draft,
            poi_by_id,
            routes,
            route_strategy,
            route_mode,
            route_modes,
            max_walking_leg_meters,
            resolution,
            boundaries,
            pois,
        )
        for draft in drafts
    ]

