from __future__ import annotations

from datetime import date, datetime, timedelta
import logging
from decimal import Decimal

from travel_agent.domain.models import (
    ItemType,
    PlanCandidate,
    PlanningPOI,
    POI,
    TimeWindow,
    TripSpec,
    ValidationResult,
    Violation,
    ViolationSeverity,
)
from travel_agent.domain.tool_models import ValueSource
from travel_agent.planning.poi_identity import normalize_poi_name, poi_names_overlap


logger = logging.getLogger(__name__)


_ASSUMPTION_WARNINGS = {
    "opening_window": (
        "opening_hours_unverified",
        "营业时间来自默认假设，尚未由 Provider 验证",
    ),
    "duration_minutes": (
        "duration_unverified",
        "游览时长来自默认假设，尚未由 Provider 验证",
    ),
    "walking_distance": (
        "walking_distance_estimated",
        "步行距离包含估算路线，尚未由 Provider 验证",
    ),
    "walking_distance_meters": (
        "walking_distance_estimated",
        "步行距离包含估算路线，尚未由 Provider 验证",
    ),
    "party_cost": (
        "cost_unverified",
        "部分地点费用按默认值估算，尚未由 Provider 验证",
    ),
}


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _assumption_warnings(
    candidate: PlanCandidate,
    *,
    opening_hours_unverified: bool = False,
) -> list[Violation]:
    """把默认事实收敛为候选计划级告警，避免每条日程重复报告。"""
    warnings_by_type: dict[str, Violation] = {}
    for assumption in candidate.assumptions:
        if assumption.field == "stay_anchor":
            warnings_by_type.setdefault(
                "stay_anchor_unconfirmed",
                Violation(
                    type="stay_anchor_unconfirmed",
                    severity=ViolationSeverity.WARNING,
                    message="住宿位置未确认，当前路线基于建议住宿区域估算",
                ),
            )
            continue
        if assumption.source is not ValueSource.DEFAULT:
            continue
        violation_type, message = _ASSUMPTION_WARNINGS.get(
            assumption.field,
            (f"{assumption.field}_unverified", f"{assumption.field} 来自默认假设，尚未验证"),
        )
        warnings_by_type.setdefault(
            violation_type,
            Violation(
                type=violation_type,
                severity=ViolationSeverity.WARNING,
                message=message,
            ),
        )
    if opening_hours_unverified:
        warnings_by_type.setdefault(
            "opening_hours_unverified",
            Violation(
                type="opening_hours_unverified",
                severity=ViolationSeverity.WARNING,
                message="部分日程营业时间来自默认假设，尚未由 Provider 验证",
            ),
        )
    return [warnings_by_type[key] for key in sorted(warnings_by_type)]


def _poi_id(poi: POI | PlanningPOI) -> str:
    return poi.id if isinstance(poi, POI) else poi.facts.id


def _opening_window_for_day(
    poi: POI | PlanningPOI,
    day: date,
) -> tuple[TimeWindow | None, ValueSource]:
    if isinstance(poi, POI):
        return poi.opening_window, ValueSource.PROVIDER
    return (
        poi.opening_windows.get(day),
        poi.opening_window_sources.get(day, ValueSource.DEFAULT),
    )


def validate_candidate(
    trip: TripSpec,
    candidate: PlanCandidate,
    pois: list[POI | PlanningPOI],
) -> ValidationResult:
    violations: list[Violation] = []
    poi_by_id = {_poi_id(poi): poi for poi in pois}
    opening_hours_unverified = False
    all_items = [item for day in candidate.days for item in day.items]
    activities = [item for item in all_items if item.type == ItemType.ACTIVITY]

    visited_ids: dict[str, date] = {}
    visited_names: dict[str, date] = {}
    for day in sorted(candidate.days, key=lambda value: value.date):
        for item in sorted(day.items, key=lambda value: value.start_at):
            if item.type is not ItemType.ACTIVITY:
                continue
            poi = poi_by_id.get(item.poi_id) if item.poi_id else None
            canonical_name = (
                poi.facts.name if isinstance(poi, PlanningPOI)
                else poi.name if isinstance(poi, POI)
                else item.name
            )
            name = normalize_poi_name(canonical_name)
            previous_day = visited_ids.get(item.poi_id) if item.poi_id else None
            if previous_day is None and name:
                previous_day = visited_names.get(name)
                if previous_day is None:
                    previous_day = next(
                        (seen_day for seen_name, seen_day in visited_names.items()
                         if poi_names_overlap(name, seen_name)),
                        None,
                    )
            if previous_day is not None:
                violations.append(
                    Violation(
                        type="duplicate_poi",
                        severity=ViolationSeverity.ERROR,
                        day=day.date,
                        entity_ids=[item.poi_id] if item.poi_id else [],
                        message=f"景点重复安排：{item.name}，已在 {previous_day} 安排",
                        repair_hint="保留一次访问，删除重复项或替换为尚未安排的景点，并重新计算路线",
                    )
                )
                logger.warning(
                    "validation.duplicate_poi candidate_id=%s poi_id=%s day=%s first_day=%s",
                    candidate.id, item.poi_id, day.date, previous_day,
                )
            if item.poi_id:
                visited_ids.setdefault(item.poi_id, day.date)
            if name:
                visited_names.setdefault(name, day.date)

    if not activities:
        violations.append(
            Violation(
                type="empty_plan",
                severity=ViolationSeverity.ERROR,
                message="计划中没有任何活动",
                repair_hint="增加可用时间、预算或候选地点",
            )
        )

    scheduled_names = {_normalize(item.name) for item in activities}
    for required in trip.must_visit:
        normalized = _normalize(required)
        if not any(normalized in name or name in normalized for name in scheduled_names):
            violations.append(
                Violation(
                    type="missing_must_visit",
                    severity=ViolationSeverity.ERROR,
                    message=f"必去地点未安排：{required}",
                    repair_hint="调整每日活动上限或放宽其他约束",
                )
            )

    arrival_buffer = trip.arrival.at + timedelta(minutes=60)
    departure_buffer = trip.departure.at - timedelta(minutes=90)
    trip_timezone = trip.arrival.at.tzinfo

    for day in candidate.days:
        ordered = sorted(day.items, key=lambda item: item.start_at)
        if ordered and day.end_anchor is not None and day.end_leg is None:
            violations.append(
                Violation(
                    type="missing_day_end_route",
                    severity=ViolationSeverity.ERROR,
                    day=day.date,
                    message="当天最后一个活动缺少前往日终锚点的路线",
                    repair_hint="补充最后一个 POI 到住宿点或离开地点的路线",
                )
            )
        previous = None
        for item in ordered:
            if previous and item.start_at < previous.end_at:
                violations.append(
                    Violation(
                        type="time_overlap",
                        severity=ViolationSeverity.ERROR,
                        day=day.date,
                        entity_ids=[value for value in [previous.poi_id, item.poi_id] if value],
                        message=f"{previous.name} 与 {item.name} 时间重叠",
                        repair_hint="重新排序或减少活动",
                    )
                )
            previous = item

            if item.type is ItemType.ACTIVITY:
                daily_start = datetime.combine(
                    day.date,
                    trip.daily_start,
                    tzinfo=trip_timezone,
                )
                daily_end = datetime.combine(
                    day.date,
                    trip.daily_end,
                    tzinfo=trip_timezone,
                )
                start_at = item.start_at.astimezone(trip_timezone)
                end_at = item.end_at.astimezone(trip_timezone)
                if start_at < daily_start or end_at > daily_end:
                    violations.append(
                        Violation(
                            type="outside_daily_window",
                            severity=ViolationSeverity.ERROR,
                            day=day.date,
                            entity_ids=[item.poi_id] if item.poi_id else [],
                            message=f"{item.name} 的访问时间不在每日时间窗内",
                            repair_hint="调整访问时间或移动到其他日期",
                        )
                    )

            if day.date == trip.arrival.at.date() and item.start_at < arrival_buffer:
                violations.append(
                    Violation(
                        type="arrival_buffer",
                        severity=ViolationSeverity.ERROR,
                        day=day.date,
                        entity_ids=[item.poi_id] if item.poi_id else [],
                        message=f"{item.name} 安排在到达缓冲时间之前",
                    )
                )
            if day.date == trip.departure.at.date() and item.end_at > departure_buffer:
                violations.append(
                    Violation(
                        type="departure_buffer",
                        severity=ViolationSeverity.ERROR,
                        day=day.date,
                        entity_ids=[item.poi_id] if item.poi_id else [],
                        message=f"{item.name} 结束时间侵占返程缓冲",
                    )
                )

            if item.poi_id and item.poi_id in poi_by_id:
                poi = poi_by_id[item.poi_id]
                opening_window, opening_source = _opening_window_for_day(poi, day.date)
                if opening_source is not ValueSource.PROVIDER:
                    opening_hours_unverified = True
                elif opening_window is not None:
                    timezone = item.start_at.tzinfo
                    opening = datetime.combine(day.date, opening_window.start, tzinfo=timezone)
                    closing = datetime.combine(day.date, opening_window.end, tzinfo=timezone)
                    poi_name = poi.name if isinstance(poi, POI) else poi.facts.name
                    poi_id = _poi_id(poi)
                    if item.start_at < opening or item.end_at > closing:
                        violations.append(
                            Violation(
                                type="outside_opening_hours",
                                severity=ViolationSeverity.ERROR,
                                day=day.date,
                                entity_ids=[poi_id],
                                message=f"{poi_name} 的访问时间不在营业时间内",
                                repair_hint="调整访问时间或移动到其他日期",
                            )
                        )

        if day.walking_distance_meters > trip.mobility.max_daily_walking_meters:
            violations.append(
                Violation(
                    type="walking_limit",
                    severity=ViolationSeverity.ERROR,
                    day=day.date,
                    message=(
                        f"预计步行 {day.walking_distance_meters} 米，超过上限 "
                        f"{trip.mobility.max_daily_walking_meters} 米"
                    ),
                    repair_hint="减少跨区域活动或降低每日活动数量",
                )
            )

        if ordered and day.end_leg is not None:
            day_deadline = datetime.combine(
                day.date, trip.daily_end, tzinfo=trip_timezone
            )
            if day.date == trip.departure.at.date():
                day_deadline = min(day_deadline, departure_buffer)
            end_arrival = ordered[-1].end_at + timedelta(
                minutes=day.end_leg.duration_minutes
            )
            if end_arrival > day_deadline:
                violations.append(
                    Violation(
                        type="day_end_route_late",
                        severity=ViolationSeverity.ERROR,
                        day=day.date,
                        entity_ids=(
                            [ordered[-1].poi_id] if ordered[-1].poi_id else []
                        ),
                        message="当天最后一个活动结束后无法按时到达日终锚点",
                        repair_hint="提前结束当天行程或移除最后一个活动",
                    )
                )

        activity_minutes = sum(
            int((item.end_at - item.start_at).total_seconds() / 60)
            for item in ordered
            if item.type == ItemType.ACTIVITY
        )
        if activity_minutes > trip.mobility.max_daily_activity_minutes:
            violations.append(
                Violation(
                    type="activity_time_limit",
                    severity=ViolationSeverity.ERROR,
                    day=day.date,
                    message=(
                        f"活动时间 {activity_minutes} 分钟，超过每日上限 "
                        f"{trip.mobility.max_daily_activity_minutes} 分钟"
                    ),
                    repair_hint="减少活动或移动到其他日期",
                )
            )

    known_cost = sum(
        (item.estimated_cost for item in all_items if item.estimated_cost is not None),
        start=Decimal("0"),
    )
    unknown_cost_item_count = sum(
        item.estimated_cost is None for item in all_items
    )
    if trip.total_budget is not None:
        if unknown_cost_item_count:
            violations.append(
                Violation(
                    type="budget_unverified",
                    severity=ViolationSeverity.WARNING,
                    message=(
                        f"有 {unknown_cost_item_count} 个日程费用未知；已知费用 "
                        f"{known_cost} 元，无法验证总预算 {trip.total_budget} 元"
                    ),
                    repair_hint="补充地点、交通或餐饮费用后重新验证预算",
                )
            )
        if known_cost > trip.total_budget:
            violations.append(
                Violation(
                    type="budget_exceeded",
                    severity=ViolationSeverity.ERROR,
                    message=f"已知预计费用 {known_cost} 元，超过预算 {trip.total_budget} 元",
                    repair_hint="减少收费活动或提高预算",
                )
            )

    violations.extend(
        _assumption_warnings(
            candidate,
            opening_hours_unverified=opening_hours_unverified,
        )
    )
    return ValidationResult.from_violations(violations)
