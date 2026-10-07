from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from math import inf
import logging

from pydantic import BaseModel, ConfigDict

from travel_agent.domain.models import (
    Coordinate,
    DayBoundary,
    PlanStyle,
    PlanningPOI,
    POI,
    StayAnchorResolution,
    TripSpec,
)
from travel_agent.domain.tool_models import POIFacts, RouteMode, RouteQuery, route_key
from travel_agent.planning.routing import haversine_distance_meters
from travel_agent.planning.poi_identity import unique_planning_pois
from travel_agent.planning.stay import derive_day_boundaries, resolve_stay_anchor


STYLE_ACTIVITY_LIMITS = {
    PlanStyle.RELAXED: 2,
    PlanStyle.BALANCED: 3,
    PlanStyle.EXPLORATION: 4,
}
logger = logging.getLogger(__name__)


class MissingPlanningPOI(LookupError):
    """Draft 引用了当前规划上下文中不存在的 POI。"""

    def __init__(self, poi_id: str) -> None:
        self.poi_id = poi_id
        super().__init__(f"missing planning POI: {poi_id}")


class DraftDay(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    poi_ids: tuple[str, ...]


class CandidateDraft(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    style: PlanStyle
    days: tuple[DraftDay, ...]


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _facts(poi: PlanningPOI | POI) -> POIFacts | POI:
    return poi.facts if isinstance(poi, PlanningPOI) else poi


def _party_cost(poi: PlanningPOI | POI) -> Decimal | None:
    if isinstance(poi, PlanningPOI):
        return poi.party_cost
    return poi.estimated_cost


def _confidence(poi: PlanningPOI | POI) -> float:
    return poi.data_confidence


def _is_must_visit(poi: PlanningPOI | POI, trip: TripSpec) -> bool:
    name = _normalize(_facts(poi).name)
    return any(
        _normalize(required) in name or name in _normalize(required)
        for required in trip.must_visit
    )


def _poi_preference_score(
    poi: PlanningPOI | POI,
    trip: TripSpec,
    replan_round: int,
) -> float:
    facts = _facts(poi)
    interests = {_normalize(value) for value in trip.interests}
    avoid = {_normalize(value) for value in trip.avoid}
    categories = {_normalize(value) for value in facts.categories}
    tags = {
        _normalize(value)
        for value in getattr(facts, "suitability_tags", [])
    }
    score = 1.0
    score += 4.0 * len(interests & (categories | tags))
    score -= 5.0 * len(avoid & (categories | tags))
    if _is_must_visit(poi, trip):
        score += 100.0
    if trip.mobility.needs_frequent_rest and "适老" in categories | tags:
        score += 2.0
    cost = _party_cost(poi)
    if replan_round and cost is not None:
        score -= float(cost / Decimal("100")) * replan_round
    return score


def _select_pois(
    trip: TripSpec,
    pois: list[PlanningPOI],
    style: PlanStyle,
    replan_round: int,
) -> list[PlanningPOI]:
    per_day_limit = max(1, STYLE_ACTIVITY_LIMITS[style] - replan_round)
    total_limit = min(len(pois), trip.day_count * per_day_limit)

    def rank_key(poi: PlanningPOI) -> tuple[float, float, float, str]:
        cost = _party_cost(poi)
        return (
            -_poi_preference_score(poi, trip, replan_round),
            -_confidence(poi),
            float(cost) if cost is not None else inf,
            poi.facts.id,
        )

    return unique_planning_pois(sorted(pois, key=rank_key))[:total_limit]


def _order_nearest(
    start: Coordinate,
    pois: list[PlanningPOI],
    trip: TripSpec,
) -> list[PlanningPOI]:
    ordered: list[PlanningPOI] = []
    current = start
    for required_layer in (True, False):
        remaining = [
            poi
            for poi in pois
            if _is_must_visit(poi, trip) is required_layer
        ]
        while remaining:
            next_poi = min(
                remaining,
                key=lambda poi: (
                    haversine_distance_meters(current, poi.facts.coordinate),
                    poi.facts.id,
                ),
            )
            ordered.append(next_poi)
            remaining.remove(next_poi)
            current = next_poi.facts.coordinate
    return ordered


def prepare_candidate_drafts(
    trip: TripSpec,
    pois: list[PlanningPOI],
    replan_round: int,
    *,
    day_boundaries: tuple[DayBoundary, ...] | None = None,
) -> list[CandidateDraft]:
    """Phase 1：确定性选择与排序，不生成任何路线时间或道路距离。"""
    boundaries = day_boundaries or derive_day_boundaries(
        trip, resolve_stay_anchor(trip, pois)
    )
    drafts: list[CandidateDraft] = []
    for style in PlanStyle:
        selected = _select_pois(trip, pois, style, replan_round)
        day_buckets: list[list[PlanningPOI]] = [
            [] for _ in range(trip.day_count)
        ]
        # 保留原有每日数量配额，按邻近关系填充，避免轮流分配拆散相邻地点。
        remaining = list(selected)
        for day_index, bucket in enumerate(day_buckets):
            quota = len(selected) // trip.day_count + (day_index < len(selected) % trip.day_count)
            if not quota:
                continue
            bucket.append(remaining.pop(0))
            while len(bucket) < quota:
                nearest = min(remaining, key=lambda poi: (
                    min(haversine_distance_meters(member.facts.coordinate, poi.facts.coordinate)
                        for member in bucket),
                    selected.index(poi),
                ))
                bucket.append(nearest)
                remaining.remove(nearest)
        logger.info(
            "planning.fallback_proximity_assignment | style=%s round=%s buckets=%s",
            style.value, replan_round,
            [[poi.facts.id for poi in bucket] for bucket in day_buckets],
        )
        days = tuple(
            DraftDay(
                date=trip.start_date + timedelta(days=day_index),
                poi_ids=tuple(
                    poi.facts.id
                    for poi in _order_nearest(
                        boundaries[day_index].start_anchor.coordinate,
                        day_buckets[day_index],
                        trip,
                    )
                ),
            )
            for day_index in range(trip.day_count)
        )
        drafts.append(
            CandidateDraft(
                id=f"{style.value}-r{replan_round}",
                style=style,
                days=days,
            )
        )
    return drafts


def collect_route_queries(
    trip: TripSpec,
    drafts: list[CandidateDraft],
    pois: list[PlanningPOI],
    route_strategy: int = 32,
    route_mode: RouteMode = RouteMode.DRIVING,
    route_modes: tuple[RouteMode, ...] | None = None,
    max_walking_leg_meters: int = 1_500,
    stay_resolution: StayAnchorResolution | None = None,
    day_boundaries: tuple[DayBoundary, ...] | None = None,
) -> list[RouteQuery]:
    """收集首见优先、方向敏感且模式隔离的路线查询。"""
    poi_by_id = {poi.facts.id: poi for poi in pois}
    resolution = stay_resolution or resolve_stay_anchor(trip, pois)
    boundaries = day_boundaries or derive_day_boundaries(trip, resolution)
    boundary_by_date = {boundary.date: boundary for boundary in boundaries}
    queries: list[RouteQuery] = []
    seen_keys: set[str] = set()
    for draft in drafts:
        for day in draft.days:
            boundary = boundary_by_date[day.date]
            previous_coordinate = boundary.start_anchor.coordinate
            previous_poi_id: str | None = None
            for poi_id in day.poi_ids:
                poi = poi_by_id.get(poi_id)
                if poi is None:
                    raise MissingPlanningPOI(poi_id)
                for active_mode in route_modes or (route_mode,):
                    if (
                        active_mode is RouteMode.WALKING
                        and haversine_distance_meters(
                            previous_coordinate,
                            poi.facts.coordinate,
                        )
                        > max_walking_leg_meters * 1.25
                    ):
                        continue
                    query = RouteQuery(
                        origin=previous_coordinate,
                        destination=poi.facts.coordinate,
                        origin_poi_id=previous_poi_id,
                        destination_poi_id=poi_id,
                        mode=active_mode,
                        strategy=(route_strategy if active_mode is RouteMode.DRIVING else 0),
                    )
                    key = route_key(query)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        queries.append(query)
                previous_coordinate = poi.facts.coordinate
                previous_poi_id = poi_id
            # 每一个可能成为当天最后一站的 POI 都预取返回日终锚点的路线，
            # 这样物化阶段即使因营业时间截断行程，也不需要临时调用工具。
            for poi_id in day.poi_ids:
                poi = poi_by_id.get(poi_id)
                if poi is None:
                    raise MissingPlanningPOI(poi_id)
                for active_mode in route_modes or (route_mode,):
                    if (
                        active_mode is RouteMode.WALKING
                        and haversine_distance_meters(
                            poi.facts.coordinate,
                            boundary.end_anchor.coordinate,
                        )
                        > max_walking_leg_meters * 1.25
                    ):
                        continue
                    query = RouteQuery(
                        origin=poi.facts.coordinate,
                        destination=boundary.end_anchor.coordinate,
                        origin_poi_id=poi_id,
                        mode=active_mode,
                        strategy=(
                            route_strategy if active_mode is RouteMode.DRIVING else 0
                        ),
                    )
                    key = route_key(query)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        queries.append(query)
    return queries
