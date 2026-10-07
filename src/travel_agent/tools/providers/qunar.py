"""去哪儿景点发现与票价补充；高德负责确认位置和统一 POI 身份。"""

from __future__ import annotations

import asyncio
import logging
import re
from decimal import Decimal, InvalidOperation

import httpx

from travel_agent.domain.tool_models import POIFacts, POISearchQuery, ValueSource
from travel_agent.tools.errors import ToolProviderError
from travel_agent.tools.protocols import POIProvider

logger = logging.getLogger(__name__)


def _name(value: str) -> str:
    return re.sub(r"\s+", "", value.split("(")[0].split("（")[0]).casefold()


def _price(row: dict) -> Decimal | None:
    if row.get("free") is True or row.get("free") == 1:
        return Decimal("0")
    value = row.get("qunarPrice")
    if value is None or value == "":
        return None
    try:
        price = Decimal(str(value))
    except InvalidOperation:
        return None
    return price if price.is_finite() and price >= 0 else None


class QunarScenicPOIProvider:
    """在景点分类查询中合并去哪儿候选，失败时保留高德结果。

    只读取第一页，最多核实5个新地点；不返回未经地图核实的坐标。
    外部原始响应不进入 Graph，来源通过标准化 facts 传递。
    """

    name = "amap+qunar"

    def __init__(self, primary: POIProvider, client: httpx.AsyncClient) -> None:
        self.primary = primary
        self.client = client

    async def _scenic(self, city: str, limit: int) -> list[dict]:
        try:
            response = await self.client.get(
                "https://piao.qunar.com/ticket/list.json",
                params={"keyword": city, "from": "mpl_search_suggest", "page": 1},
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=1.5,
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload["data"]["sightList"]
            if not isinstance(rows, list):
                raise ValueError("invalid scenic list")
            return [row for row in rows if isinstance(row, dict)
                    and isinstance(row.get("sightName"), str)
                    and row["sightName"].strip()][:limit]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            # 补充源不可用是工具降级，不是业务不可行；不打印原始响应。
            logger.warning("qunar.scenic.unavailable city=%s fallback=amap", city)
            return []

    async def search_pois(self, query: POISearchQuery) -> list[POIFacts]:
        if query.exact_match or query.types != "110000":
            return await self.primary.search_pois(query)
        primary, rows = await asyncio.gather(
            self.primary.search_pois(query), self._scenic(query.city, query.limit)
        )
        if not rows:
            return primary
        try:
            return await asyncio.wait_for(self._merge(query, primary, rows), timeout=2.0)
        except TimeoutError:
            logger.warning("qunar.scenic.resolve_timeout city=%s fallback=amap", query.city)
            return primary

    async def _merge(
        self, query: POISearchQuery, primary: list[POIFacts], rows: list[dict]
    ) -> list[POIFacts]:
        facts_by_id = {fact.id: fact for fact in primary}
        preferred: list[str] = []
        resolutions = 0
        for row in rows:
            name = row["sightName"].strip()
            matches = [fact for fact in facts_by_id.values() if _name(fact.name) == _name(name)]
            if not matches and resolutions < 5:
                resolutions += 1
                try:
                    candidates = await self.primary.search_pois(POISearchQuery(
                        city=query.city, keyword=name, types="110000", exact_match=True, limit=3
                    ))
                except ToolProviderError:
                    logger.warning("qunar.scenic.resolve_failed city=%s", query.city)
                    continue
                matches = [fact for fact in candidates if _name(fact.name) == _name(name)
                           and fact.city.rstrip("市") == query.city.rstrip("市")]
            price = _price(row)
            for fact in matches:
                if price is not None:
                    fact = fact.model_copy(update={
                        "average_cost_per_person": price,
                        "provider": "amap+qunar",
                        "field_sources": {**fact.field_sources,
                                          "average_cost_per_person": ValueSource.PROVIDER},
                    })
                facts_by_id[fact.id] = fact
                if fact.id not in preferred:
                    preferred.append(fact.id)
        ids = preferred + [key for key in facts_by_id if key not in preferred]
        result = [facts_by_id[key] for key in ids[:query.limit]]
        logger.info("qunar.scenic.merged city=%s amap_count=%s qunar_count=%s verified_count=%s result_count=%s",
                    query.city, len(primary), len(rows), len(preferred), len(result))
        return result
