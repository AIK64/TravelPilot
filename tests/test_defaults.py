from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal

import pytest

from travel_agent.domain.models import Coordinate, TimeWindow
from travel_agent.domain.tool_models import POIFacts, UnknownFactPolicy, ValueSource
from travel_agent.planning.defaults import POIDefaultPolicy


@pytest.fixture
def poi_facts() -> POIFacts:
    return POIFacts(
        id="lingyin",
        name="灵隐寺",
        city="杭州",
        coordinate=Coordinate(longitude=120.101, latitude=30.237),
        categories=["景点", "人文"],
        provider="mock",
        fetched_at=datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc),
        data_confidence=0.9,
    )


def test_assume_policy_marks_missing_hours_duration_and_cost(hangzhou_trip, poi_facts):
    """防止 assume 模式默默伪造未知事实而没有标记。"""
    resolution = POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING).resolve(
        poi_facts, hangzhou_trip
    )

    assert resolution.poi is not None
    assert resolution.poi.opening_windows[hangzhou_trip.start_date] == TimeWindow(
        start=time(8), end=time(18)
    )
    assert resolution.poi.duration_minutes == 120
    assert resolution.poi.party_cost == Decimal("45") * hangzhou_trip.travelers
    assert {item.field for item in resolution.poi.assumptions} == {
        "opening_window",
        "duration_minutes",
        "party_cost",
    }
    assert resolution.poi.field_sources == {
        "duration_minutes": ValueSource.DEFAULT,
        "party_cost": ValueSource.DEFAULT,
        "data_confidence": ValueSource.DERIVED,
    }
    assert resolution.poi.opening_window_sources == {
        date(2026, 10, 2): ValueSource.DEFAULT,
        date(2026, 10, 3): ValueSource.DEFAULT,
        date(2026, 10, 4): ValueSource.DEFAULT,
    }
    assumptions_by_field = {item.field: item for item in resolution.poi.assumptions}
    assert {item.source for item in assumptions_by_field.values()} == {ValueSource.DEFAULT}
    assert assumptions_by_field["opening_window"].source is ValueSource.DEFAULT
    assert assumptions_by_field["opening_window"].value == "08:00-18:00"
    assert assumptions_by_field["duration_minutes"].value == "120"
    assert assumptions_by_field["opening_window"].affected_dates == [
        date(2026, 10, 2),
        date(2026, 10, 3),
        date(2026, 10, 4),
    ]
    assert assumptions_by_field["duration_minutes"].affected_dates == []
    assert assumptions_by_field["party_cost"].affected_dates == []
    assert assumptions_by_field["party_cost"].value == "45"
    assert resolution.poi.data_confidence == pytest.approx(0.45)


def test_strict_policy_rejects_missing_critical_facts(hangzhou_trip, poi_facts):
    """防止 strict 模式以不完整的 POI 继续后续规划。"""
    resolution = POIDefaultPolicy(UnknownFactPolicy.STRICT).resolve(poi_facts, hangzhou_trip)

    assert resolution.poi is None
    assert resolution.missing_fields == [
        "opening_window",
        "duration_minutes",
        "party_cost",
    ]


def test_free_provider_price_is_not_replaced_by_default(hangzhou_trip, poi_facts):
    facts = poi_facts.model_copy(update={"average_cost_per_person": Decimal("0")})
    resolution = POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING).resolve(
        facts, hangzhou_trip
    )
    assert resolution.poi.party_cost == Decimal("0")
    assert "party_cost" not in {item.field for item in resolution.poi.assumptions}


def test_policy_resolves_hours_by_trip_date_and_only_uses_matching_today_value(hangzhou_trip, poi_facts):
    """防止把抓取当天的营业时间错误套用到其它出行日期。"""
    facts = poi_facts.model_copy(
        update={
            "opening_windows_by_weekday": {
                4: TimeWindow(start=time(9), end=time(18)),
                6: TimeWindow(start=time(10), end=time(15)),
            },
            "today_opening_date": date(2026, 10, 3),
            "today_opening_window": TimeWindow(start=time(11), end=time(17)),
            "suggested_duration_minutes": 120,
            "average_cost_per_person": Decimal("50"),
        }
    )

    resolution = POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING).resolve(
        facts, hangzhou_trip
    )

    assert resolution.poi is not None
    assert resolution.poi.opening_windows == {
        date(2026, 10, 2): TimeWindow(start=time(9), end=time(18)),
        date(2026, 10, 3): TimeWindow(start=time(11), end=time(17)),
        date(2026, 10, 4): TimeWindow(start=time(10), end=time(15)),
    }
    assert resolution.poi.duration_minutes == 120
    assert resolution.poi.party_cost == Decimal("150")
    assert resolution.poi.assumptions == []
    assert resolution.poi.field_sources == {
        "duration_minutes": ValueSource.PROVIDER,
        "party_cost": ValueSource.DERIVED,
        "data_confidence": ValueSource.DERIVED,
    }
    assert resolution.poi.opening_window_sources == {
        date(2026, 10, 2): ValueSource.PROVIDER,
        date(2026, 10, 3): ValueSource.PROVIDER,
        date(2026, 10, 4): ValueSource.PROVIDER,
    }
    assert resolution.poi.data_confidence == 0.9
