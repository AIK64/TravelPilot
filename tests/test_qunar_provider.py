from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest

from travel_agent.domain.models import Coordinate
from travel_agent.domain.tool_models import POIFacts, POISearchQuery, ValueSource
from travel_agent.tools.providers.qunar import QunarScenicPOIProvider


def fact(name, cost=None):
    return POIFacts(id=name, name=name, city="杭州", coordinate=Coordinate(
        longitude=120.1, latitude=30.2), categories=["风景名胜"],
        average_cost_per_person=cost, provider="amap",
        fetched_at=datetime.now(timezone.utc))


class MapProvider:
    name = "amap"

    def __init__(self):
        self.queries = []

    async def search_pois(self, query):
        self.queries.append(query)
        if query.exact_match:
            return [fact("灵隐寺")] if query.keyword == "灵隐寺" else [fact("无关地点")]
        return [fact("西湖", Decimal("12"))]


@pytest.mark.asyncio
async def test_scenic_merge_verifies_names_and_preserves_free_price():
    rows = [{"sightName": "西湖", "free": True},
            {"sightName": "灵隐寺", "qunarPrice": "75"},
            {"sightName": "不存在", "qunarPrice": "20"}]

    def handler(request):
        assert request.url.params["keyword"] == "杭州"
        assert "key" not in request.url.params
        return httpx.Response(200, json={"data": {"sightList": rows}})

    primary = MapProvider()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await QunarScenicPOIProvider(primary, client).search_pois(
            POISearchQuery(city="杭州", keyword="景点", types="110000"))
    assert [p.name for p in result] == ["西湖", "灵隐寺"]
    assert [p.average_cost_per_person for p in result] == [Decimal("0"), Decimal("75")]
    assert result[1].field_sources["average_cost_per_person"] is ValueSource.PROVIDER
    assert result[1].id == "灵隐寺"  # 沿用地图身份，路线工具无需供应商转换。


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"data": {"sightList": []}}, {},
                                       {"data": {"sightList": "bad"}}])
async def test_unavailable_supplement_keeps_map_results(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload))) as client:
        result = await QunarScenicPOIProvider(MapProvider(), client).search_pois(
            POISearchQuery(city="杭州", keyword="景点", types="110000"))
    assert len(result) == 1
    assert result[0].average_cost_per_person == Decimal("12")


@pytest.mark.asyncio
async def test_anchor_lookup_does_not_request_qunar():
    def unexpected(_):
        pytest.fail("精确地点查询不得发起城市景点搜索")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        result = await QunarScenicPOIProvider(MapProvider(), client).search_pois(
            POISearchQuery(city="杭州", keyword="灵隐寺", exact_match=True))
    assert result[0].name == "灵隐寺"


@pytest.mark.asyncio
async def test_missing_ticket_does_not_become_free():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
        200, json={"data": {"sightList": [{"sightName": "灵隐寺"}]}}))) as client:
        result = await QunarScenicPOIProvider(MapProvider(), client).search_pois(
            POISearchQuery(city="杭州", keyword="景点", types="110000"))
    assert next(p for p in result if p.name == "灵隐寺").average_cost_per_person is None
