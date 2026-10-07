from __future__ import annotations

from datetime import timedelta

from travel_agent.domain.models import (
    Coordinate,
    DayBoundary,
    LocationAnchor,
    PlanningAssumption,
    PlanningPOI,
    StayAnchorMode,
    StayAnchorResolution,
    TripSpec,
)
from travel_agent.domain.tool_models import ValueSource
from travel_agent.planning.routing import haversine_distance_meters


def resolve_stay_anchor(
    trip: TripSpec,
    pois: list[PlanningPOI],
    *,
    candidate_limit: int = 3,
) -> StayAnchorResolution:
    """解析规划住宿基点，不把推荐值写回用户提供的 TripSpec。"""
    if trip.accommodation is not None:
        return StayAnchorResolution(
            mode=StayAnchorMode.PROVIDED,
            anchor=trip.accommodation,
            confidence=1.0,
            confirmed=True,
            reason_codes=("user_provided",),
        )
    if trip.day_count == 1:
        return StayAnchorResolution(
            mode=StayAnchorMode.NOT_REQUIRED,
            confidence=1.0,
            confirmed=True,
            reason_codes=("single_day_trip",),
        )

    if not pois:
        midpoint = Coordinate(
            longitude=(
                trip.arrival.coordinate.longitude
                + trip.departure.coordinate.longitude
            )
            / 2,
            latitude=(
                trip.arrival.coordinate.latitude
                + trip.departure.coordinate.latitude
            )
            / 2,
        )
        return StayAnchorResolution(
            mode=StayAnchorMode.UNRESOLVED,
            anchor=LocationAnchor(
                name=f"{trip.destination}住宿区域（待确认）",
                coordinate=midpoint,
            ),
            confidence=0.2,
            confirmed=False,
            reason_codes=("insufficient_poi_context",),
        )

    weighted = [(poi, _poi_weight(poi, trip)) for poi in pois]
    ranked = sorted(
        (
            (
                _candidate_score(candidate, weighted, trip),
                candidate.facts.id,
                candidate,
            )
            for candidate, _ in weighted
        ),
        key=lambda item: (item[0], item[1]),
    )[: max(1, candidate_limit)]
    best_score, _, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else best_score
    separation = (
        max(0.0, min(1.0, (second_score - best_score) / second_score))
        if second_score > 0
        else 1.0
    )
    provider_confidence = sum(
        poi.data_confidence * weight for poi, weight in weighted
    ) / sum(weight for _, weight in weighted)
    confidence = max(
        0.35,
        min(0.9, provider_confidence * (0.65 + 0.35 * separation)),
    )
    total_weight = sum(weight for _, weight in weighted)
    planning_center = Coordinate(
        longitude=sum(
            poi.facts.coordinate.longitude * weight for poi, weight in weighted
        ) / total_weight,
        latitude=sum(
            poi.facts.coordinate.latitude * weight for poi, weight in weighted
        ) / total_weight,
    )
    return StayAnchorResolution(
        mode=StayAnchorMode.RECOMMENDED,
        anchor=LocationAnchor(
            name=f"建议住宿区域：{best.facts.name}附近",
            coordinate=planning_center,
        ),
        confidence=round(confidence, 3),
        confirmed=False,
        reference_poi_ids=tuple(item[2].facts.id for item in ranked),
        reason_codes=(
            "weighted_poi_medoid",
            "arrival_departure_access_considered",
        ),
    )


def derive_day_boundaries(
    trip: TripSpec,
    stay: StayAnchorResolution,
) -> tuple[DayBoundary, ...]:
    arrival = LocationAnchor(
        name=trip.arrival.name,
        coordinate=trip.arrival.coordinate,
    )
    departure = LocationAnchor(
        name=trip.departure.name,
        coordinate=trip.departure.coordinate,
    )
    if trip.day_count == 1:
        return (
            DayBoundary(
                date=trip.start_date,
                start_role="arrival",
                start_anchor=arrival,
                end_role="departure",
                end_anchor=departure,
            ),
        )
    if stay.anchor is None:
        raise ValueError("multi-day planning requires a resolved stay anchor")

    boundaries: list[DayBoundary] = []
    for index in range(trip.day_count):
        day = trip.start_date + timedelta(days=index)
        first = index == 0
        last = index == trip.day_count - 1
        boundaries.append(
            DayBoundary(
                date=day,
                start_role="arrival" if first else "stay",
                start_anchor=arrival if first else stay.anchor,
                end_role="departure" if last else "stay",
                end_anchor=departure if last else stay.anchor,
            )
        )
    return tuple(boundaries)


def stay_assumption(
    trip: TripSpec,
    stay: StayAnchorResolution,
    pois: list[PlanningPOI],
) -> PlanningAssumption | None:
    if stay.mode not in {StayAnchorMode.RECOMMENDED, StayAnchorMode.UNRESOLVED}:
        return None
    return PlanningAssumption(
        field="stay_anchor",
        value=stay.anchor.name if stay.anchor is not None else "未解析",
        reason="用户未提供住宿，系统仅使用建议住宿区域估算每日路线",
        source=ValueSource.DERIVED,
        affected_dates=[
            trip.start_date + timedelta(days=index)
            for index in range(trip.day_count)
        ],
        # 规划假设属于本次 Trip，而不是某次 Provider 拉取；使用行程锚点保证回放稳定。
        created_at=trip.arrival.at,
    )


def _poi_weight(poi: PlanningPOI, trip: TripSpec) -> float:
    normalized_name = poi.facts.name.strip().casefold()
    must_visit = any(
        required.strip().casefold() in normalized_name
        or normalized_name in required.strip().casefold()
        for required in trip.must_visit
    )
    categories = {item.strip().casefold() for item in poi.facts.categories}
    interests = {item.strip().casefold() for item in trip.interests}
    return 3.0 if must_visit else 2.0 if categories & interests else 1.0


def _candidate_score(
    candidate: PlanningPOI,
    weighted: list[tuple[PlanningPOI, float]],
    trip: TripSpec,
) -> float:
    coordinate = candidate.facts.coordinate
    poi_distance = sum(
        weight * haversine_distance_meters(coordinate, poi.facts.coordinate)
        for poi, weight in weighted
    )
    access_distance = 0.35 * (
        haversine_distance_meters(trip.arrival.coordinate, coordinate)
        + haversine_distance_meters(coordinate, trip.departure.coordinate)
    )
    return poi_distance + access_distance
