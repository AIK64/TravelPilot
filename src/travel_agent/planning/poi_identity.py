"""单城市行程的访问目标去重：身份、名称归一化和关键词包含关系。"""

import logging
import re
import unicodedata

from travel_agent.domain.models import PlanningPOI


logger = logging.getLogger(__name__)


def normalize_poi_name(name: str) -> str:
    return "".join(unicodedata.normalize("NFKC", name).casefold().split())


def normalize_poi_match_name(name: str) -> str:
    """对齐参考项目：移除括号备注及末尾景区类通用后缀。"""
    normalized = normalize_poi_name(name)
    core = re.sub(r"\([^)]*\)", "", normalized)
    core = re.sub(
        r"(风景名胜区|风景区|旅游区|度假区|历史街区|景区|街区|公园|博物馆|纪念馆)$",
        "", core,
    )
    return core or normalized


def poi_names_overlap(first: str, second: str) -> bool:
    """完整同名或归一化关键词互为子串时判重；包含匹配要求至少2字。"""
    left, right = normalize_poi_name(first), normalize_poi_name(second)
    if left and left == right:
        return True
    left, right = normalize_poi_match_name(first), normalize_poi_match_name(second)
    return min(len(left), len(right)) >= 2 and (left in right or right in left)


def unique_planning_pois(pois: list[PlanningPOI]) -> list[PlanningPOI]:
    """每个重复组保留输入排序最优代表，避免父景点晚出现时漏掉关联项。

    输入由上层按必去、偏好及置信度排序；仅剔除记录，不拼接坐标或费用。
    """
    parents = list(range(len(pois)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, poi in enumerate(pois):
        for previous in range(index):
            other = pois[previous]
            if poi.facts.id == other.facts.id or poi_names_overlap(poi.facts.name, other.facts.name):
                first, second = root(previous), root(index)
                parents[max(first, second)] = min(first, second)

    unique: list[PlanningPOI] = []
    for index, poi in enumerate(pois):
        representative = root(index)
        if representative != index:
            logger.info(
                "planning.poi_duplicate_excluded poi_id=%s kept_poi_id=%s reason=identity_or_keyword_overlap",
                poi.facts.id, pois[representative].facts.id,
            )
        else:
            unique.append(poi)
    return unique
