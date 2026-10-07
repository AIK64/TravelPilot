"""将用户偏好转换为稳定、可审计的 POI 检索意图。"""

import logging

from travel_agent.domain.models import TripSpec
from travel_agent.domain.tool_models import POIFacts, POISearchQuery, ToolResult
from travel_agent.planning.poi_identity import normalize_poi_name, poi_names_overlap


logger = logging.getLogger(__name__)


def build_search_plan(
    trip: TripSpec,
    per_query_limit: int = 10,
    max_queries: int = 12,
    *,
    city: str | None = None,
) -> list[POISearchQuery]:
    """必去地点逐个查询；通用检索固定使用景点关键词和景点类型。"""
    if not 1 <= max_queries <= 100:
        raise ValueError("max_queries must be between 1 and 100")
    seen: set[str] = set()
    queries: list[POISearchQuery] = []
    required = tuple({normalize_poi_name(name): name.strip() for name in trip.must_visit if name.strip()}.values())
    if len(required) > max_queries:
        raise ValueError("must_visit query count exceeds search budget")
    candidates = [*((name, True, 100) for name in required),
                  ("景点", False, 50)]

    for keyword, exact_match, priority in candidates:
        normalized = normalize_poi_name(keyword)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        queries.append(
            POISearchQuery(
                city=city or trip.destination,
                keyword=keyword.strip(),
                exact_match=exact_match,
                limit=per_query_limit,
                priority=priority,
                types="" if exact_match else "110000",
            )
        )
        if len(queries) == max_queries:
            break

    if not queries:
        queries.append(
            POISearchQuery(city=trip.destination, keyword="景点", limit=per_query_limit, priority=10)
        )
    return queries


def select_search_candidates(
    queries: list[POISearchQuery], results: list[ToolResult[list[POIFacts]]],
    *, max_candidates: int,
) -> list[POIFacts]:
    """必去查询只取首条且校验名称；普通结果判重后补充候选。"""
    candidates: list[POIFacts] = []
    ordinary: list[POIFacts] = []
    for query, result in zip(queries, results, strict=True):
        facts = result.data or []
        if query.exact_match:
            first = facts[0] if facts else None
            if first is None or not poi_names_overlap(query.keyword, first.name):
                logger.warning("poi.must_visit_not_found keyword=%s returned_count=%s", query.keyword, len(facts))
                continue
            candidates.append(first)
            logger.info("poi.must_visit_selected keyword=%s poi_id=%s", query.keyword, first.id)
        else:
            ordinary.extend(facts)
    unique: list[POIFacts] = []
    for fact in [*candidates, *ordinary]:
        if any(fact.id == previous.id or poi_names_overlap(fact.name, previous.name) for previous in unique):
            logger.info("poi.search_duplicate_excluded poi_id=%s", fact.id)
            continue
        # 必去目标不因普通候选数量上限被截断。
        if len(unique) >= max(max_candidates, len(candidates)):
            break
        unique.append(fact)
    return unique
