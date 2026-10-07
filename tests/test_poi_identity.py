import pytest

from travel_agent.domain.tool_models import POIFacts, UnknownFactPolicy
from travel_agent.planning.defaults import POIDefaultPolicy
from travel_agent.planning.poi_identity import poi_names_overlap, unique_planning_pois


@pytest.fixture
def planning_pois(hangzhou_trip):
    facts = POIFacts(
        id="original", name="灵隐寺", city="杭州",
        coordinate=hangzhou_trip.arrival.coordinate, categories=["人文"],
        provider="mock", fetched_at=hangzhou_trip.arrival.at,
    )
    return [POIDefaultPolicy(UnknownFactPolicy.ASSUME_WITH_WARNING).resolve(
        facts, hangzhou_trip).poi]


@pytest.mark.parametrize("first,second,expected", [
    ("鲁迅故里", "鲁迅故里景区", True),
    ("沈园", "沈园之夜", True),
    ("泉州大开元寺", "泉州大开元寺普贤楼", True),
    ("开元寺（泉州）", "开元寺", True),
    (" 西 湖 ", "西湖风景名胜区", True),
    ("浙江省博物馆", "中国茶叶博物馆", False),
    ("开元寺", "承天寺", False),
    ("山", "山海公园", False),
    ("", "开元寺", False),
    # 规则不推断别名或父子归属；名称没有包含关系时不自行合并。
    ("泉州大开元寺", "泉州开元寺-仁寿塔", False),
])
def test_keyword_overlap(first, second, expected):
    assert poi_names_overlap(first, second) is expected
    assert poi_names_overlap(second, first) is expected


def test_related_candidates_do_not_consume_multiple_slots(planning_pois, caplog):
    original = planning_pois[0]

    def renamed(poi_id, name):
        return original.model_copy(update={
            "facts": original.facts.model_copy(update={"id": poi_id, "name": name})
        })

    candidates = [renamed("main", "泉州大开元寺"),
                  renamed("building", "泉州大开元寺普贤楼"),
                  renamed("alias", "泉州大开元寺（泉州）"),
                  renamed("other", "清源山")]
    result = unique_planning_pois(candidates)
    assert [poi.facts.id for poi in result] == ["main", "other"]
    assert "kept_poi_id=main" in caplog.text
    assert result[0] is candidates[0]  # 不将子景点的坐标或费用拼到代表上。


def test_parent_name_appearing_late_still_connects_related_names(planning_pois):
    original = planning_pois[0]
    candidates = [original.model_copy(update={"facts": original.facts.model_copy(
        update={"id": str(index), "name": name})}) for index, name in enumerate(
            ["开元寺-仁寿塔", "开元寺普贤楼", "开元寺"])]
    assert unique_planning_pois(candidates) == [candidates[0]]
