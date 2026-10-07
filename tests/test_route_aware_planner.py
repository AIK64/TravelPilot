from __future__ import annotations

from datetime import time, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from travel_agent.domain.models import (
    PlanStyle,
    PlanningAssumption,
    PlanningPOI,
    TimeWindow,
)
from travel_agent.domain.tool_models import (
    POIFacts,
    RouteMode,
    RouteResult,
    ValueSource,
    route_key,
)
from travel_agent.planning.drafts import (
    CandidateDraft,
    DraftDay,
    MissingPlanningPOI,
    collect_route_queries,
    prepare_candidate_drafts,
)
from travel_agent.planning.planner import MissingRouteResult, materialize_candidates
from travel_agent.domain.optimization_models import OptimizationBudget
from travel_agent.planning.optimization import select_optimization_pois
from travel_agent.planning.poi_identity import normalize_poi_name


@pytest.fixture
def planning_pois(hangzhou_trip) -> list[PlanningPOI]:
    poi_specs = [
        ("lingyin", "灵隐寺", 120.1017, 30.2404, ["人文", "寺庙"], "75", 120),
        ("west_lake", "西湖", 120.1487, 30.2448, ["自然"], "0", 120),
        ("hefang", "河坊街", 120.1714, 30.2371, ["美食", "人文"], "60", 90),
        ("museum", "浙江省博物馆", 120.1420, 30.2537, ["人文", "博物馆"], "0", 90),
        ("wetland", "西溪湿地", 120.0624, 30.2668, ["自然"], "80", 180),
    ]
    trip_dates = [
        hangzhou_trip.start_date + timedelta(days=index)
        for index in range(hangzhou_trip.day_count)
    ]
    window = TimeWindow(start="08:00", end="21:00")
    return [
        PlanningPOI(
            facts=POIFacts(
                id=poi_id,
                name=name,
                city="杭州",
                coordinate={"longitude": longitude, "latitude": latitude},
                categories=categories,
                provider="fixture",
                fetched_at=hangzhou_trip.arrival.at,
                data_confidence=0.9,
            ),
            opening_windows={trip_date: window for trip_date in trip_dates},
            duration_minutes=duration,
            party_cost=Decimal(cost),
            data_confidence=0.9,
        )
        for poi_id, name, longitude, latitude, categories, cost, duration in poi_specs
    ]


@pytest.fixture
def candidate_drafts(hangzhou_trip, planning_pois) -> list[CandidateDraft]:
    return prepare_candidate_drafts(hangzhou_trip, planning_pois, replan_round=0)


@pytest.fixture
def single_draft(hangzhou_trip, planning_pois) -> CandidateDraft:
    return CandidateDraft(
        id="single",
        style=PlanStyle.RELAXED,
        days=tuple(
            DraftDay(
                date=hangzhou_trip.start_date + timedelta(days=index),
                poi_ids=(planning_pois[0].facts.id,) if index == 0 else (),
            )
            for index in range(hangzhou_trip.day_count)
        ),
    )


def _route_results(
    hangzhou_trip,
    queries,
    *,
    confidence=0.9,
    provider="fixture",
):
    return {
        route_key(query): RouteResult(
            distance_meters=4200 + index * 100,
            duration_minutes=18 + index,
            provider=provider,
            data_confidence=confidence,
            fetched_at=hangzhou_trip.arrival.at,
        )
        for index, query in enumerate(queries)
    }


def _forbid_route_estimate(*_args):
    raise AssertionError("route estimate used")


@pytest.mark.parametrize("same_id", [True, False])
def test_duplicate_pois_do_not_take_cross_day_slots(hangzhou_trip, planning_pois, same_id):
    original = planning_pois[0]
    duplicate = original.model_copy(update={
        "facts": original.facts.model_copy(update={
            "id": original.facts.id if same_id else "lingyin-alias",
            "name": " 灵 隐 寺 ",
        }),
    })
    pois = [original, duplicate, *planning_pois[1:]]
    selected = select_optimization_pois(
        hangzhou_trip, pois, OptimizationBudget(candidate_limit=2)
    )
    assert len(selected) == 2
    assert len({normalize_poi_name(poi.facts.name) for poi in selected}) == 2

    by_id = {poi.facts.id: poi for poi in pois}
    for draft in prepare_candidate_drafts(hangzhou_trip, pois, replan_round=0):
        ids = [poi_id for day in draft.days for poi_id in day.poi_ids]
        names = [normalize_poi_name(by_id[poi_id].facts.name) for poi_id in ids]
        assert len(ids) == len(set(ids))
        assert len(names) == len(set(names))
        assert len(ids) == len(planning_pois)


def test_drafts_use_haversine_only_for_ordering(
    hangzhou_trip, planning_pois, monkeypatch
):
    monkeypatch.setattr(
        "travel_agent.planning.routing.estimate_route",
        _forbid_route_estimate,
    )
    drafts = prepare_candidate_drafts(hangzhou_trip, planning_pois, replan_round=0)

    assert {draft.style for draft in drafts} == set(PlanStyle)
    assert all(len(draft.days) == hangzhou_trip.day_count for draft in drafts)
    assert all(
        set(draft.model_dump()) == {"id", "style", "days"}
        for draft in drafts
    )
    assert all(
        set(day.model_dump()) == {"date", "poi_ids"}
        for draft in drafts
        for day in draft.days
    )


def test_replan_density_is_bounded_and_keeps_must_visit(
    hangzhou_trip, planning_pois
):
    drafts = prepare_candidate_drafts(
        hangzhou_trip, planning_pois, replan_round=5
    )

    assert all(
        sum(len(day.poi_ids) for day in draft.days) == hangzhou_trip.day_count
        for draft in drafts
    )
    assert all(
        "lingyin" in {poi_id for day in draft.days for poi_id in day.poi_ids}
        for draft in drafts
    )


def test_must_visit_precedes_nearer_infeasible_optional_and_route_order_matches(
    hangzhou_trip, planning_pois
):
    trip = hangzhou_trip.model_copy(
        update={
            "end_date": hangzhou_trip.start_date,
            "departure": hangzhou_trip.departure.model_copy(
                update={
                    "at": hangzhou_trip.arrival.at.replace(hour=19, minute=0)
                }
            ),
        }
    )
    must_visit = planning_pois[0]
    nearer_optional = planning_pois[1].model_copy(
        update={
            "opening_windows": {
                trip.start_date: TimeWindow(start="09:00", end="11:00")
            }
        }
    )
    pois = [must_visit, nearer_optional]
    draft = next(
        item
        for item in prepare_candidate_drafts(trip, pois, replan_round=0)
        if item.style is PlanStyle.RELAXED
    )
    queries = collect_route_queries(trip, [draft], pois)

    candidate = materialize_candidates(
        trip,
        [draft],
        pois,
        _route_results(trip, queries),
    )[0]

    assert [
        query.destination_poi_id
        for query in queries
        if query.destination_poi_id is not None
    ] == list(
        draft.days[0].poi_ids
    )
    assert must_visit.facts.id in {
        item.poi_id for item in candidate.days[0].items
    }


def test_non_required_pois_keep_nearest_neighbor_order(
    hangzhou_trip, planning_pois
):
    trip = hangzhou_trip.model_copy(
        update={
            "end_date": hangzhou_trip.start_date,
            "must_visit": [],
        }
    )
    west_lake = planning_pois[1]
    museum = planning_pois[3]
    draft = next(
        item
        for item in prepare_candidate_drafts(
            trip, [museum, west_lake], replan_round=0
        )
        if item.style is PlanStyle.RELAXED
    )

    assert draft.days[0].poi_ids == (museum.facts.id, west_lake.facts.id)


def test_candidate_drafts_are_immutable(hangzhou_trip):
    day = DraftDay(date=hangzhou_trip.start_date, poi_ids=("lingyin",))
    draft = CandidateDraft(id="relaxed-r0", style=PlanStyle.RELAXED, days=(day,))

    with pytest.raises(ValidationError, match="frozen"):
        draft.id = "changed"


def test_route_queries_are_directional_and_deduplicated(
    hangzhou_trip, candidate_drafts, planning_pois
):
    queries = collect_route_queries(hangzhou_trip, candidate_drafts, planning_pois)
    keys = [route_key(query) for query in queries]

    assert len(keys) == len(set(keys))
    assert all(query.mode is RouteMode.DRIVING for query in queries)
    assert all(query.strategy == 32 for query in queries)
    assert any(query.destination_poi_id is None for query in queries)


def test_route_queries_keep_opposite_directions(hangzhou_trip, planning_pois):
    first, second = planning_pois[:2]
    drafts = [
        CandidateDraft(
            id="directional",
            style=PlanStyle.RELAXED,
            days=(
                DraftDay(
                    date=hangzhou_trip.start_date,
                    poi_ids=(first.facts.id, second.facts.id),
                ),
                DraftDay(
                    date=hangzhou_trip.start_date + timedelta(days=1),
                    poi_ids=(second.facts.id, first.facts.id),
                ),
            ),
        )
    ]

    queries = collect_route_queries(hangzhou_trip, drafts, planning_pois)
    poi_pairs = [
        (query.origin_poi_id, query.destination_poi_id)
        for query in queries
        if query.origin_poi_id is not None
    ]

    assert (first.facts.id, second.facts.id) in poi_pairs
    assert (second.facts.id, first.facts.id) in poi_pairs


def test_route_queries_reject_unknown_draft_poi_id(hangzhou_trip, planning_pois):
    draft = CandidateDraft(
        id="broken",
        style=PlanStyle.RELAXED,
        days=(DraftDay(date=hangzhou_trip.start_date, poi_ids=("missing",)),),
    )

    with pytest.raises(MissingPlanningPOI, match="missing"):
        collect_route_queries(hangzhou_trip, [draft], planning_pois)


def test_materialization_uses_provider_route_values(
    hangzhou_trip, single_draft, planning_pois
):
    query = collect_route_queries(
        hangzhou_trip, [single_draft], planning_pois
    )[0]
    queries = collect_route_queries(hangzhou_trip, [single_draft], planning_pois)
    routes = _route_results(hangzhou_trip, queries)
    routes[route_key(query)] = RouteResult(
            distance_meters=4200,
            duration_minutes=18,
            provider="fixture",
            data_confidence=0.9,
            fetched_at=hangzhou_trip.arrival.at,
        )

    candidate = materialize_candidates(
        hangzhou_trip, [single_draft], planning_pois, routes
    )[0]

    first = candidate.days[0].items[0]
    assert first.distance_from_previous_meters == 4200
    assert first.travel_from_previous_minutes == 18
    assert first.coordinate == planning_pois[0].facts.coordinate
    assert candidate.model_dump(mode="json")["days"][0]["items"][0]["coordinate"] == {
        "longitude": 120.1017, "latitude": 30.2404,
    }
    assert first.walking_distance_estimated is True
    assert candidate.days[0].walking_distance_meters >= 504
    assert candidate.days[0].end_leg is not None


def test_materialization_rejects_missing_route_result(
    hangzhou_trip, single_draft, planning_pois
):
    query = collect_route_queries(
        hangzhou_trip, [single_draft], planning_pois
    )[0]

    with pytest.raises(MissingRouteResult) as error:
        materialize_candidates(
            hangzhou_trip, [single_draft], planning_pois, routes={}
        )

    assert error.value.route_key == route_key(query)


def test_materialization_checks_all_draft_segments_before_scheduling(
    hangzhou_trip, planning_pois
):
    first_date = hangzhou_trip.start_date
    closes_before_arrival = planning_pois[1].model_copy(
        update={
            "opening_windows": {
                **planning_pois[1].opening_windows,
                first_date: TimeWindow(start="09:00", end="11:00"),
            }
        }
    )
    second = planning_pois[2]
    pois = [planning_pois[0], closes_before_arrival, second, *planning_pois[3:]]
    draft = CandidateDraft(
        id="preflight",
        style=PlanStyle.RELAXED,
        days=(
            DraftDay(
                date=first_date,
                poi_ids=(closes_before_arrival.facts.id, second.facts.id),
            ),
        ),
    )
    queries = collect_route_queries(hangzhou_trip, [draft], pois)
    only_first_route = _route_results(hangzhou_trip, queries[:1])

    with pytest.raises(MissingRouteResult) as error:
        materialize_candidates(
            hangzhou_trip, [draft], pois, only_first_route
        )

    assert error.value.route_key == route_key(queries[1])


def test_materialization_preserves_assumptions_and_marks_derived_walking_once(
    hangzhou_trip, single_draft, planning_pois
):
    upstream = PlanningAssumption(
        field="duration_minutes",
        value="120",
        reason="Provider 未返回游览时长",
        source=ValueSource.DEFAULT,
        affected_dates=[hangzhou_trip.start_date],
        created_at=hangzhou_trip.arrival.at,
    )
    pois = [
        planning_pois[0].model_copy(update={"assumptions": [upstream]}),
        *planning_pois[1:],
    ]
    queries = collect_route_queries(hangzhou_trip, [single_draft], pois)

    candidate = materialize_candidates(
        hangzhou_trip,
        [single_draft],
        pois,
        _route_results(hangzhou_trip, queries),
    )[0]

    assert upstream in candidate.assumptions
    walking_assumptions = [
        assumption
        for assumption in candidate.assumptions
        if assumption.field == "walking_distance"
    ]
    assert len(walking_assumptions) == 1
    assert walking_assumptions[0].source is ValueSource.DEFAULT
    assert "路线 Provider 标准化距离" in walking_assumptions[0].reason


def test_materialization_uses_daily_window_and_tracks_unknown_party_cost(
    hangzhou_trip, planning_pois
):
    first_date = hangzhou_trip.start_date
    first = planning_pois[0].model_copy(
        update={
            "opening_windows": {
                **planning_pois[0].opening_windows,
                first_date: TimeWindow(start="15:00", end="18:00"),
            }
        }
    )
    second = planning_pois[1].model_copy(update={"party_cost": None})
    pois = [first, second, *planning_pois[2:]]
    draft = CandidateDraft(
        id="cost-and-window",
        style=PlanStyle.BALANCED,
        days=(DraftDay(date=first_date, poi_ids=(first.facts.id, second.facts.id)),),
    )
    queries = collect_route_queries(hangzhou_trip, [draft], pois)

    candidate = materialize_candidates(
        hangzhou_trip,
        [draft],
        pois,
        _route_results(hangzhou_trip, queries),
    )[0]

    day = candidate.days[0]
    assert day.items[0].start_at.hour == 15
    assert day.known_estimated_cost == first.party_cost
    assert day.unknown_cost_item_count == 1
    assert candidate.metrics.known_estimated_cost == first.party_cost
    assert candidate.metrics.unknown_cost_item_count == 1


def test_materialization_does_not_schedule_must_visit_outside_daily_window(
    hangzhou_trip, single_draft, planning_pois
):
    """防止必去地点绕过每日结束时间或 Provider 营业结束时间。"""
    trip = hangzhou_trip.model_copy(update={"daily_end": time(12, 0)})
    queries = collect_route_queries(trip, [single_draft], planning_pois)

    candidate = materialize_candidates(
        trip,
        [single_draft],
        planning_pois,
        _route_results(trip, queries),
    )[0]

    assert candidate.days[0].items == []
    assert candidate.metrics.total_travel_minutes == 0


def test_route_confidence_changes_candidate_confidence_and_score(
    hangzhou_trip, single_draft, planning_pois
):
    queries = collect_route_queries(
        hangzhou_trip, [single_draft], planning_pois
    )
    high = materialize_candidates(
        hangzhou_trip,
        [single_draft],
        planning_pois,
        _route_results(hangzhou_trip, queries, confidence=0.9),
    )[0]
    low = materialize_candidates(
        hangzhou_trip,
        [single_draft],
        planning_pois,
        _route_results(hangzhou_trip, queries, confidence=0.1),
    )[0]

    assert high.metrics.data_confidence == 0.9
    assert low.metrics.data_confidence < high.metrics.data_confidence
    assert high.score > low.score


def test_mock_route_provenance_is_explicitly_a_local_estimate(
    hangzhou_trip, single_draft, planning_pois
):
    """防止默认 Mock 的本地估算在候选解释中被表述成真实路线事实。"""
    queries = collect_route_queries(hangzhou_trip, [single_draft], planning_pois)
    candidate = materialize_candidates(
        hangzhou_trip,
        [single_draft],
        planning_pois,
        _route_results(
            hangzhou_trip,
            queries,
            confidence=0.65,
            provider="mock",
        ),
    )[0]
    public_text = "\n".join(
        [
            *candidate.reason_facts,
            *(assumption.reason for assumption in candidate.assumptions),
        ]
    )

    assert "路线来源 mock（本地估算）" in public_text
    assert "路线置信度 65%" in public_text
    assert "真实驾车路线" not in public_text
    assert "真实驾车距离" not in public_text


def test_amap_route_provenance_is_provider_normalized_without_overclaiming(
    hangzhou_trip, single_draft, planning_pois
):
    """防止真实 Provider 标准化结果被描述为超出适配器证据强度的事实。"""
    queries = collect_route_queries(hangzhou_trip, [single_draft], planning_pois)
    candidate = materialize_candidates(
        hangzhou_trip,
        [single_draft],
        planning_pois,
        _route_results(
            hangzhou_trip,
            queries,
            confidence=0.95,
            provider="amap",
        ),
    )[0]
    public_text = "\n".join(candidate.reason_facts)

    assert "路线来源 amap（Provider 标准化结果）" in public_text
    assert "路线置信度 95%" in public_text
    assert "本地估算" not in public_text
    assert "真实驾车路线" not in public_text


def test_default_assumption_increases_warning_risk(
    hangzhou_trip, single_draft, planning_pois
):
    queries = collect_route_queries(
        hangzhou_trip, [single_draft], planning_pois
    )
    routes = _route_results(hangzhou_trip, queries)
    baseline = materialize_candidates(
        hangzhou_trip, [single_draft], planning_pois, routes
    )[0]
    duration_assumption = PlanningAssumption(
        field="duration_minutes",
        value="120",
        reason="Provider 未返回游览时长",
        source=ValueSource.DEFAULT,
        affected_dates=[hangzhou_trip.start_date],
        created_at=hangzhou_trip.arrival.at,
    )
    assumed_pois = [
        planning_pois[0].model_copy(
            update={"assumptions": [duration_assumption]}
        ),
        *planning_pois[1:],
    ]
    warned = materialize_candidates(
        hangzhou_trip, [single_draft], assumed_pois, routes
    )[0]

    assert warned.metrics == baseline.metrics
    assert warned.score < baseline.score
