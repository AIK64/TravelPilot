"""基于现有确定性数据集的显式 Mock Provider 实现。"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

from travel_agent.domain.tool_models import (
    POIFacts,
    POISearchQuery,
    RouteMode,
    RouteQuery,
    RouteResult,
    ValueSource,
)
from travel_agent.planning.mock_data import get_mock_pois
from travel_agent.planning.routing import estimate_route
from travel_agent.planning.routing import haversine_distance_meters


def _normalize(value: str) -> str:
    return value.strip().casefold()


class MockPOIProvider:
    """将 Hangzhou Mock POI 数据集适配为供应商无关事实。"""

    name = "mock"

    async def search_pois(self, query: POISearchQuery) -> list[POIFacts]:
        keyword = _normalize(query.keyword)
        matches = []
        for index, poi in enumerate(get_mock_pois(query.city)):
            scenic_query = query.types == "110000" and keyword == "景点"
            if scenic_query and not set(poi.categories).intersection(
                {"自然", "人文", "历史", "街区", "公园", "博物馆", "寺庙"}
            ):
                continue
            searchable = [_normalize(poi.name)]
            searchable.extend(_normalize(value) for value in poi.categories)
            searchable.extend(_normalize(value) for value in poi.suitability_tags)
            if not scenic_query and keyword and not any(keyword in value for value in searchable):
                continue
            matches.append((index, poi))

        if query.exact_match and keyword:
            matches.sort(
                key=lambda item: (
                    0 if _normalize(item[1].name) == keyword else 1,
                    0 if keyword in _normalize(item[1].name) else 1,
                    item[0],
                )
            )

        fetched_at = datetime.now(timezone.utc)
        return [self._to_facts(poi, fetched_at) for _, poi in matches[: query.limit]]

    def _to_facts(self, poi, fetched_at: datetime) -> POIFacts:
        provider_source = ValueSource.PROVIDER
        return POIFacts(
            id=poi.id,
            name=poi.name,
            city=poi.city,
            coordinate=poi.coordinate,
            categories=list(poi.categories),
            opening_windows_by_weekday={
                weekday: poi.opening_window for weekday in range(7)
            },
            today_opening_window=poi.opening_window,
            today_opening_date=date.today(),
            average_cost_per_person=poi.estimated_cost,
            suggested_duration_minutes=poi.estimated_duration_minutes,
            provider=self.name,
            fetched_at=fetched_at,
            data_confidence=poi.data_confidence,
            field_sources={
                "id": provider_source,
                "name": provider_source,
                "city": provider_source,
                "coordinate": provider_source,
                "categories": provider_source,
                "opening_windows_by_weekday": provider_source,
                "today_opening_window": provider_source,
                "average_cost_per_person": provider_source,
                "suggested_duration_minutes": provider_source,
                "data_confidence": provider_source,
            },
        )


class MockRouteProvider:
    """将确定性路线估算器适配为显式低置信度的本地估算结果。"""

    name = "mock"

    async def get_driving_route(self, query: RouteQuery) -> RouteResult:
        distance_meters, duration_minutes, _ = estimate_route(
            query.origin, query.destination
        )
        return RouteResult(
            distance_meters=distance_meters,
            duration_minutes=duration_minutes,
            mode=query.mode,
            provider=self.name,
            data_confidence=0.65,
            fetched_at=datetime.now(timezone.utc),
        )

    async def get_walking_route(self, query: RouteQuery) -> RouteResult:
        distance_meters = max(
            1,
            round(haversine_distance_meters(query.origin, query.destination) * 1.08),
        )
        return RouteResult(
            distance_meters=distance_meters,
            duration_minutes=max(1, math.ceil(distance_meters / 75)),
            mode=RouteMode.WALKING,
            provider=self.name,
            data_confidence=0.75,
            fetched_at=datetime.now(timezone.utc),
        )
