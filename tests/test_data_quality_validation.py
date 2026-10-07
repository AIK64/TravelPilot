from datetime import datetime, time, timedelta
from decimal import Decimal
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from travel_agent.domain.models import (
    DayPlan,
    Coordinate,
    ItemType,
    PlanCandidate,
    PlanItem,
    PlanMetrics,
    PlanStyle,
    PlanningAssumption,
    PlanningPOI,
    TimeWindow,
    ValidationResult,
    ValidationStatus,
    Violation,
    ViolationSeverity,
)
from travel_agent.domain.tool_models import POIFacts, ValueSource
from travel_agent.planning.validator import validate_candidate


@pytest.mark.parametrize(
    ("first_id", "second_id", "second_name", "day_offset"),
    [
        ("lingyin", "lingyin", "灵隐寺另一名称", 1),
        ("lingyin", "alias", " 灵 隐 寺 ", 1),
        (None, None, "灵隐寺", 1),
        ("lingyin", "lingyin", "灵隐寺", 0),
        ("lingyin", "building", "灵隐寺-大雄宝殿", 1),
    ],
)
def test_repeated_attraction_fails_hard_validation(
    hangzhou_trip, first_id, second_id, second_name, day_offset, caplog
):
    candidate = _candidate_with_activity(hangzhou_trip, Decimal("0"))
    first = candidate.days[0].items[0].model_copy(update={"poi_id": first_id})
    candidate.days[0].items = [first]
    repeated = first.model_copy(update={
        "poi_id": second_id,
        "name": second_name,
        "start_at": first.start_at + timedelta(days=day_offset, hours=2),
        "end_at": first.end_at + timedelta(days=day_offset, hours=2),
    })
    if day_offset:
        candidate.days.append(candidate.days[0].model_copy(update={
            "date": candidate.days[0].date + timedelta(days=day_offset),
            "items": [repeated],
        }))
    else:
        candidate.days[0].items.append(repeated)
    result = validate_candidate(hangzhou_trip, candidate, [])
    duplicates = [violation for violation in result.violations if violation.type == "duplicate_poi"]
    assert not result.valid
    assert len(duplicates) == 1
    assert duplicates[0].severity is ViolationSeverity.ERROR
    assert duplicates[0].day == candidate.days[-1].date
    assert "validation.duplicate_poi" in caplog.text


def test_repeated_non_activity_is_not_a_duplicate_attraction(hangzhou_trip):
    candidate = _candidate_with_activity(hangzhou_trip, Decimal("0"))
    # 重复住宿/休息记录不属于景点访问。
    resting = candidate.days[0].items[0].model_copy(update={"type": ItemType.REST})
    candidate.days[0].items.append(resting)
    assert not any(
        violation.type == "duplicate_poi"
        for violation in validate_candidate(hangzhou_trip, candidate, []).violations
    )


def test_domain_models_resolve_provenance_types_without_defaults_import():
    """防止 domain 模型依赖规划模块的偶然导入顺序才能生成 schema。"""
    source_root = Path(__file__).parents[1] / "src"
    environment = {**os.environ, "PYTHONPATH": str(source_root)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from travel_agent.domain.models import PlanCandidate, PlanningAssumption; "
                "PlanCandidate.model_json_schema(); PlanningAssumption.model_json_schema()"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def _candidate_with_activity(
    hangzhou_trip,
    estimated_cost: Decimal | None,
    assumptions: list[PlanningAssumption] | None = None,
) -> PlanCandidate:
    """构造不触发其他硬约束的单活动候选计划。"""
    start_at = hangzhou_trip.arrival.at + timedelta(minutes=90)
    item = PlanItem(
        type=ItemType.ACTIVITY,
        name="灵隐寺",
        start_at=start_at,
        end_at=start_at + timedelta(minutes=60),
        estimated_cost=estimated_cost,
    )
    known_cost = estimated_cost if estimated_cost is not None else Decimal("0")
    return PlanCandidate(
        id="quality-check",
        style=PlanStyle.RELAXED,
        days=[
            DayPlan(
                date=hangzhou_trip.start_date,
                theme="人文",
                primary_area="杭州",
                items=[item],
                estimated_cost=known_cost,
                unknown_cost_item_count=int(estimated_cost is None),
            )
        ],
        metrics=PlanMetrics(
            preference_match=1,
            diversity=1,
            data_confidence=1,
            total_travel_minutes=0,
            walking_distance_meters=0,
            estimated_cost=known_cost,
            unknown_cost_item_count=int(estimated_cost is None),
            fatigue_score=0,
        ),
        assumptions=assumptions or [],
    )


def test_warning_only_result_is_valid_with_warnings():
    """防止未知数据被错误路由为不可行计划。"""
    result = ValidationResult.from_violations(
        [
            Violation(
                type="opening_hours_unverified",
                severity=ViolationSeverity.WARNING,
                message="营业时间来自默认假设",
            )
        ]
    )

    assert result.status is ValidationStatus.VALID_WITH_WARNINGS
    assert result.valid is True
    assert result.model_dump()["valid"] is True


def test_error_result_is_invalid():
    """防止真实硬约束冲突被降级为可选择的告警。"""
    result = ValidationResult.from_violations(
        [Violation(type="time_conflict", severity=ViolationSeverity.ERROR, message="冲突")]
    )

    assert result.status is ValidationStatus.INVALID
    assert result.valid is False
    assert result.model_dump()["valid"] is False


@pytest.mark.parametrize(
    ("violations", "expected_status"),
    [
        ([], ValidationStatus.VALID),
        (
            [Violation(type="uncertain", severity=ViolationSeverity.WARNING, message="待确认")],
            ValidationStatus.VALID_WITH_WARNINGS,
        ),
        (
            [Violation(type="conflict", severity=ViolationSeverity.ERROR, message="冲突")],
            ValidationStatus.INVALID,
        ),
        (
            [
                Violation(type="uncertain", severity=ViolationSeverity.WARNING, message="待确认"),
                Violation(type="conflict", severity=ViolationSeverity.ERROR, message="冲突"),
            ],
            ValidationStatus.INVALID,
        ),
    ],
)
def test_validation_result_status_truth_table_round_trips(violations, expected_status):
    """防止序列化后的 status 与 violations 关系被破坏，导致条件边路由错误。"""
    result = ValidationResult.from_violations(violations)

    assert result.status is expected_status
    assert ValidationResult.model_validate(result.model_dump()) == result


@pytest.mark.parametrize(
    ("status", "violations"),
    [
        (
            ValidationStatus.VALID,
            [Violation(type="conflict", severity=ViolationSeverity.ERROR, message="冲突")],
        ),
        (
            ValidationStatus.VALID_WITH_WARNINGS,
            [],
        ),
        (
            ValidationStatus.INVALID,
            [Violation(type="uncertain", severity=ViolationSeverity.WARNING, message="待确认")],
        ),
    ],
)
def test_validation_result_rejects_status_inconsistent_with_violations(status, violations):
    """防止外部直接构造矛盾结果绕过 from_violations。"""
    with pytest.raises(ValidationError, match="status"):
        ValidationResult(status=status, violations=violations)


def test_unknown_cost_warns_without_claiming_budget_is_feasible(hangzhou_trip):
    """防止未知费用按零元累计后错误宣称预算通过。"""
    result = validate_candidate(
        hangzhou_trip,
        _candidate_with_activity(hangzhou_trip, estimated_cost=None),
        pois=[],
    )

    assert result.status is ValidationStatus.VALID_WITH_WARNINGS
    assert [(item.type, item.severity) for item in result.violations] == [
        ("budget_unverified", ViolationSeverity.WARNING)
    ]


def test_known_cost_over_budget_is_invalid(hangzhou_trip):
    """防止真实已知费用超预算时只产生数据质量告警。"""
    result = validate_candidate(
        hangzhou_trip,
        _candidate_with_activity(hangzhou_trip, estimated_cost=Decimal("1501")),
        pois=[],
    )

    assert result.status is ValidationStatus.INVALID
    assert [(item.type, item.severity) for item in result.violations] == [
        ("budget_exceeded", ViolationSeverity.ERROR)
    ]


@pytest.mark.parametrize(
    ("daily_start", "daily_end", "start_delta", "end_delta"),
    [
        (time(12, 30), time(20, 0), 90, 150),
        (time(9, 0), time(12, 30), 90, 150),
    ],
)
def test_activity_outside_daily_window_is_a_hard_error(
    hangzhou_trip,
    daily_start,
    daily_end,
    start_delta,
    end_delta,
):
    """防止活动越过用户每日时间窗后仍被条件边视为可交付候选。"""
    trip = hangzhou_trip.model_copy(
        update={"daily_start": daily_start, "daily_end": daily_end}
    )
    candidate = _candidate_with_activity(
        trip,
        estimated_cost=Decimal("100"),
    )
    start_at = trip.arrival.at + timedelta(minutes=start_delta)
    candidate.days[0].items[0] = candidate.days[0].items[0].model_copy(
        update={
            "start_at": start_at,
            "end_at": trip.arrival.at + timedelta(minutes=end_delta),
        }
    )

    result = validate_candidate(trip, candidate, pois=[])

    assert result.status is ValidationStatus.INVALID
    assert [
        (item.type, item.severity, item.day)
        for item in result.violations
        if item.type == "outside_daily_window"
    ] == [
        (
            "outside_daily_window",
            ViolationSeverity.ERROR,
            trip.start_date,
        )
    ]


def test_default_assumptions_emit_one_warning_per_field(hangzhou_trip):
    """防止每段路线重复告警，或把默认事实当作 Provider 事实。"""
    assumptions = [
        PlanningAssumption(
            field="opening_window",
            value="10:00-16:00",
            reason="默认营业时间",
            source=ValueSource.DEFAULT,
            created_at=hangzhou_trip.arrival.at,
        ),
        PlanningAssumption(
            field="opening_window",
            value="10:00-16:00",
            reason="另一日期的默认营业时间",
            source=ValueSource.DEFAULT,
            created_at=hangzhou_trip.arrival.at,
        ),
        PlanningAssumption(
            field="duration_minutes",
            value="90",
            reason="默认游览时长",
            source=ValueSource.DEFAULT,
            created_at=hangzhou_trip.arrival.at,
        ),
        PlanningAssumption(
            field="walking_distance",
            value="estimated",
            reason="路线距离来自估算",
            source=ValueSource.DEFAULT,
            created_at=hangzhou_trip.arrival.at,
        ),
    ]

    result = validate_candidate(
        hangzhou_trip,
        _candidate_with_activity(
            hangzhou_trip,
            estimated_cost=Decimal("100"),
            assumptions=assumptions,
        ),
        pois=[],
    )

    assert result.status is ValidationStatus.VALID_WITH_WARNINGS
    assert {(item.type, item.severity) for item in result.violations} == {
        ("opening_hours_unverified", ViolationSeverity.WARNING),
        ("duration_unverified", ViolationSeverity.WARNING),
        ("walking_distance_estimated", ViolationSeverity.WARNING),
    }


def test_opening_hours_use_daily_provenance_for_hard_validation(hangzhou_trip):
    """防止默认营业时间被当作 Provider 事实，从而误路由到不可行分支。"""
    first_day = hangzhou_trip.start_date
    second_day = first_day + timedelta(days=1)
    provider_poi = PlanningPOI(
        facts=POIFacts(
            id="lingyin",
            name="灵隐寺",
            city="杭州",
            coordinate=Coordinate(longitude=120.1, latitude=30.2),
            categories=["人文"],
            provider="mock",
            fetched_at=hangzhou_trip.arrival.at,
        ),
        opening_windows={
            first_day: TimeWindow(start="13:00", end="18:00"),
            second_day: TimeWindow(start="13:00", end="18:00"),
        },
        duration_minutes=60,
        party_cost=Decimal("100"),
        opening_window_sources={
            first_day: ValueSource.PROVIDER,
            second_day: ValueSource.DEFAULT,
        },
        data_confidence=1,
    )
    timezone = hangzhou_trip.arrival.at.tzinfo
    first_start = datetime.combine(first_day, TimeWindow(start="12:00", end="12:01").start, tzinfo=timezone)
    second_start = datetime.combine(second_day, TimeWindow(start="12:00", end="12:01").start, tzinfo=timezone)
    candidate = _candidate_with_activity(hangzhou_trip, estimated_cost=Decimal("200"))
    candidate = candidate.model_copy(
        update={
            "days": [
                DayPlan(
                    date=first_day,
                    theme="人文",
                    primary_area="杭州",
                    items=[
                        PlanItem(
                            type=ItemType.ACTIVITY,
                            name="灵隐寺",
                            poi_id="lingyin",
                            start_at=first_start,
                            end_at=first_start + timedelta(minutes=60),
                            estimated_cost=Decimal("100"),
                        )
                    ],
                ),
                DayPlan(
                    date=second_day,
                    theme="人文",
                    primary_area="杭州",
                    items=[
                        PlanItem(
                            type=ItemType.ACTIVITY,
                            name="灵隐寺",
                            poi_id="lingyin",
                            start_at=second_start,
                            end_at=second_start + timedelta(minutes=60),
                            estimated_cost=Decimal("100"),
                        )
                    ],
                ),
            ]
        }
    )

    result = validate_candidate(hangzhou_trip, candidate, pois=[provider_poi])

    assert [(item.type, item.day) for item in result.violations if item.severity is ViolationSeverity.ERROR] == [
        ("duplicate_poi", second_day),
        ("outside_opening_hours", first_day)
    ]
    assert [(item.type, item.severity) for item in result.violations if item.severity is ViolationSeverity.WARNING] == [
        ("opening_hours_unverified", ViolationSeverity.WARNING)
    ]
