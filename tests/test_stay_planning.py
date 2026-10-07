from __future__ import annotations

from decimal import Decimal

import pytest

from travel_agent.domain.models import PlanningPOI, StayAnchorMode, TimeWindow
from travel_agent.domain.tool_models import POIFacts, RouteMode
from travel_agent.planning.drafts import collect_route_queries, prepare_candidate_drafts
from travel_agent.planning.stay import derive_day_boundaries, resolve_stay_anchor


@pytest.fixture
def planning_pois(hangzhou_trip) -> list[PlanningPOI]:
    window = TimeWindow(start="08:00", end="21:00")
    dates = [
        hangzhou_trip.start_date,
        hangzhou_trip.start_date.replace(day=hangzhou_trip.start_date.day + 1),
        hangzhou_trip.end_date,
    ]
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
            opening_windows={day: window for day in dates},
            duration_minutes=90,
            party_cost=Decimal("0"),
            data_confidence=0.9,
        )
        for poi_id, name, longitude, latitude, categories in [
            ("west_lake", "西湖", 120.1487, 30.2448, ["自然"]),
            ("lingyin", "灵隐寺", 120.1017, 30.2404, ["人文"]),
        ]
    ]


def test_missing_accommodation_recommends_unconfirmed_planning_base(
    hangzhou_trip, planning_pois
):
    trip = hangzhou_trip.model_copy(update={"accommodation": None})

    first = resolve_stay_anchor(trip, planning_pois)
    second = resolve_stay_anchor(trip, planning_pois)

    assert trip.accommodation is None
    assert first == second
    assert first.mode is StayAnchorMode.RECOMMENDED
    assert first.anchor is not None
    assert first.confirmed is False
    assert first.reference_poi_ids


def test_multi_day_boundaries_use_arrival_stay_and_departure(
    hangzhou_trip, planning_pois
):
    trip = hangzhou_trip.model_copy(update={"accommodation": None})
    stay = resolve_stay_anchor(trip, planning_pois)

    boundaries = derive_day_boundaries(trip, stay)

    assert [(item.start_role, item.end_role) for item in boundaries] == [
        ("arrival", "stay"),
        ("stay", "stay"),
        ("stay", "departure"),
    ]
    assert boundaries[0].start_anchor.coordinate == trip.arrival.coordinate
    assert boundaries[-1].end_anchor.coordinate == trip.departure.coordinate


def test_route_queries_include_directional_day_end_legs(
    hangzhou_trip, planning_pois
):
    trip = hangzhou_trip.model_copy(update={"accommodation": None})
    stay = resolve_stay_anchor(trip, planning_pois)
    boundaries = derive_day_boundaries(trip, stay)
    drafts = prepare_candidate_drafts(
        trip, planning_pois, 0, day_boundaries=boundaries
    )

    queries = collect_route_queries(
        trip,
        drafts,
        planning_pois,
        route_modes=(RouteMode.DRIVING,),
        stay_resolution=stay,
        day_boundaries=boundaries,
    )

    assert any(query.destination_poi_id is not None for query in queries)
    assert any(query.origin_poi_id is not None and query.destination_poi_id is None for query in queries)


def test_single_day_does_not_invent_a_stay(hangzhou_trip, planning_pois):
    trip = hangzhou_trip.model_copy(
        update={
            "end_date": hangzhou_trip.start_date,
            "departure": hangzhou_trip.departure.model_copy(
                update={"at": hangzhou_trip.arrival.at.replace(hour=19)}
            ),
            "accommodation": None,
        }
    )

    stay = resolve_stay_anchor(trip, planning_pois)
    boundary = derive_day_boundaries(trip, stay)[0]

    assert stay.mode is StayAnchorMode.NOT_REQUIRED
    assert stay.anchor is None
    assert (boundary.start_role, boundary.end_role) == ("arrival", "departure")
