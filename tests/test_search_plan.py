from __future__ import annotations

import pytest

from travel_agent.planning.search_plan import build_search_plan


def test_must_visit_queries_precede_interests(hangzhou_trip):
    """防止必去地点失去最高检索优先级。"""
    queries = build_search_plan(hangzhou_trip, per_query_limit=10)

    assert queries[0].keyword == "灵隐寺"
    assert queries[0].exact_match is True
    assert [query.keyword for query in queries[1:]] == ["景点"]
    assert queries[1].types == "110000"
    assert queries[0].types == ""


def test_empty_preferences_use_scenic_default(hangzhou_trip):
    """防止没有偏好时返回空的工具检索意图。"""
    trip = hangzhou_trip.model_copy(update={"must_visit": [], "interests": []})

    assert [query.keyword for query in build_search_plan(trip)] == ["景点"]


def test_duplicate_or_blank_terms_do_not_create_duplicate_queries(hangzhou_trip):
    """防止同一关键词重复消耗外部检索预算。"""
    trip = hangzhou_trip.model_copy(
        update={
            "must_visit": [" 灵隐寺 ", "灵隐寺", ""],
            "interests": ["美食", " 美食 ", ""],
        }
    )

    queries = build_search_plan(trip, per_query_limit=7)

    assert [(query.keyword, query.exact_match, query.priority, query.limit) for query in queries] == [
        ("灵隐寺", True, 100, 7),
        ("景点", False, 50, 7),
    ]


def test_search_plan_caps_queries_after_stable_must_visit_first_deduplication(
    hangzhou_trip,
):
    """防止大量偏好越过总 Tool Use 预算，并保证必去地点先占预算。"""
    trip = hangzhou_trip.model_copy(
        update={
            "must_visit": [" 必去甲 ", "必去乙", "必去甲", "必去丙"],
            "interests": ["兴趣甲", "兴趣乙", "兴趣丙"],
        }
    )

    # 不静默丢弃必去目标；预算不足时明确拒绝。
    with pytest.raises(ValueError, match="must_visit query count"):
        build_search_plan(trip, per_query_limit=1, max_queries=2)

    queries = build_search_plan(trip, per_query_limit=1, max_queries=4)

    assert [query.keyword for query in queries] == ["必去甲", "必去乙", "必去丙", "景点"]
    assert all(query.exact_match for query in queries[:3])
    assert not queries[3].exact_match
    assert all(query.limit == 1 for query in queries)
