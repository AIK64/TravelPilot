from __future__ import annotations

from enum import StrEnum
import re
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.memory.errors import MemoryPolicyError
from travel_agent.memory.models import (
    MemoryCategory,
    MemoryProposal,
    MemorySource,
    PreferenceLearningStatus,
    PreferenceScope,
    PreferenceValue,
)
from travel_agent.memory.policy import normalize_preference_value
if TYPE_CHECKING:
    from travel_agent.requirements.models import NaturalPlanningRequest, RequirementDraft


class PersistenceIntent(StrEnum):
    EXPLICIT_LONG_TERM = "explicit_long_term"
    AMBIGUOUS = "ambiguous"
    CURRENT_TRIP_ONLY = "current_trip_only"


class PreferenceEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    source: Literal["initial_request", "clarification"]
    category: MemoryCategory
    value: PreferenceValue
    summary: str = Field(min_length=1, max_length=300)
    persistence_intent: PersistenceIntent


class ExtractedPreferenceCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    category: MemoryCategory
    value: PreferenceValue
    scope: PreferenceScope = PreferenceScope.GLOBAL
    scope_key: str | None = None
    source: MemorySource
    confidence: float = Field(ge=0, le=1)
    evidence_ids: tuple[str, ...]
    persistence_intent: PersistenceIntent
    reason: str = Field(min_length=1, max_length=500)


class RejectedPreferenceCandidate(BaseModel):
    candidate_id: str
    category: MemoryCategory
    reason_code: str


class PreferenceLearningResult(BaseModel):
    status: PreferenceLearningStatus = PreferenceLearningStatus.NOT_RUN
    proposals: tuple[MemoryProposal, ...] = ()
    extracted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    deduplicated_count: int = Field(default=0, ge=0)
    message: str | None = None


_LONG_TERM_CUES = ("以后", "今后", "每次旅行", "一直", "通常", "平时", "记住")
_CURRENT_TRIP_CUES = ("这次", "本次", "此次", "这趟", "本趟", "这个行程")


def prepare_preference_evidence(
    request: NaturalPlanningRequest,
    draft: RequirementDraft,
    *,
    clarification_answer: str | None = None,
) -> list[PreferenceEvidence]:
    """仅从用户提供的需求字段生成证据，不从 Agent 生成的计划反推偏好。"""
    text = " ".join(
        value for value in (request.text, clarification_answer) if value
    )
    intent = _persistence_intent(text)
    source: Literal["initial_request", "clarification"] = (
        "clarification" if clarification_answer else "initial_request"
    )
    rows: list[tuple[MemoryCategory, PreferenceValue, str]] = []
    if draft.interests:
        rows.append(
            (
                MemoryCategory.PREFERRED_CATEGORIES,
                draft.interests,
                "用户明确表达了偏好的旅行内容类别",
            )
        )
    if draft.avoid:
        rows.append(
            (
                MemoryCategory.AVOIDED_CATEGORIES,
                draft.avoid,
                "用户明确表达了希望避开的旅行内容类别",
            )
        )
    if draft.pace is not None:
        rows.append((MemoryCategory.PACE, draft.pace.value, "用户明确表达了行程节奏"))
    walking = _walking_tolerance(text)
    if walking is not None:
        rows.append(
            (
                MemoryCategory.WALKING_TOLERANCE,
                walking,
                "用户明确给出了步行距离上限",
            )
        )
    transport = _preferred_transport(text)
    if transport:
        rows.append(
            (
                MemoryCategory.PREFERRED_TRANSPORT,
                transport,
                "用户明确表达了市内交通偏好",
            )
        )
    schedule: dict[str, str | bool] = {}
    if draft.daily_start is not None:
        schedule["earliest_start"] = draft.daily_start.isoformat(timespec="minutes")
    if draft.daily_end is not None:
        schedule["latest_end"] = draft.daily_end.isoformat(timespec="minutes")
    if schedule:
        rows.append(
            (
                MemoryCategory.SCHEDULE_PREFERENCES,
                schedule,
                "用户明确表达了每日行程时间偏好",
            )
        )
    budget_style = _budget_style(text)
    if budget_style is not None:
        rows.append(
            (MemoryCategory.BUDGET_STYLE, budget_style, "用户明确表达了消费风格")
        )
    accessibility = _accessibility_needs(text)
    if accessibility:
        rows.append(
            (
                MemoryCategory.ACCESSIBILITY_NEEDS,
                accessibility,
                "用户明确表达了无障碍或行动支持需求",
            )
        )
    return [
        PreferenceEvidence(
            source=source,
            category=category,
            value=value,
            summary=summary,
            persistence_intent=intent,
        )
        for category, value, summary in rows[:5]
    ]


def extract_preference_candidates(
    evidence: list[PreferenceEvidence],
) -> list[ExtractedPreferenceCandidate]:
    candidates: list[ExtractedPreferenceCandidate] = []
    for item in evidence[:5]:
        explicit_long_term = (
            item.persistence_intent is PersistenceIntent.EXPLICIT_LONG_TERM
        )
        candidates.append(
            ExtractedPreferenceCandidate(
                category=item.category,
                value=item.value,
                source=(
                    MemorySource.EXPLICIT_USER
                    if explicit_long_term
                    else MemorySource.MODEL_INFERENCE
                ),
                confidence=0.95 if explicit_long_term else 0.72,
                evidence_ids=(item.evidence_id,),
                persistence_intent=item.persistence_intent,
                reason=item.summary,
            )
        )
    return candidates


def validate_preference_candidates(
    candidates: list[ExtractedPreferenceCandidate],
) -> tuple[list[ExtractedPreferenceCandidate], list[RejectedPreferenceCandidate]]:
    accepted: list[ExtractedPreferenceCandidate] = []
    rejected: list[RejectedPreferenceCandidate] = []
    for candidate in candidates[:5]:
        if candidate.persistence_intent is PersistenceIntent.CURRENT_TRIP_ONLY:
            rejected.append(
                RejectedPreferenceCandidate(
                    candidate_id=candidate.candidate_id,
                    category=candidate.category,
                    reason_code="current_trip_only",
                )
            )
            continue
        try:
            normalized = normalize_preference_value(
                candidate.category, candidate.value
            )
        except MemoryPolicyError as error:
            rejected.append(
                RejectedPreferenceCandidate(
                    candidate_id=candidate.candidate_id,
                    category=candidate.category,
                    reason_code=error.code,
                )
            )
            continue
        accepted.append(candidate.model_copy(update={"value": normalized}))
    return accepted, rejected


def _persistence_intent(text: str) -> PersistenceIntent:
    if any(cue in text for cue in _LONG_TERM_CUES):
        return PersistenceIntent.EXPLICIT_LONG_TERM
    if any(cue in text for cue in _CURRENT_TRIP_CUES):
        return PersistenceIntent.CURRENT_TRIP_ONLY
    return PersistenceIntent.AMBIGUOUS


def _walking_tolerance(text: str) -> int | None:
    match = re.search(
        r"(?:步行|走路)[^，。；;]{0,12}?(?:最多|不超过|少于|控制在)?\s*"
        r"(\d+(?:\.\d+)?)\s*(公里|千米|米)",
        text,
    )
    if match is None:
        return None
    value = float(match.group(1))
    return round(value * 1000) if match.group(2) in {"公里", "千米"} else round(value)


def _preferred_transport(text: str) -> list[str]:
    aliases = {
        "地铁": "subway",
        "公交": "bus",
        "打车": "taxi",
        "出租车": "taxi",
        "步行": "walking",
        "骑行": "cycling",
        "自驾": "driving",
    }
    return [
        value
        for marker, value in aliases.items()
        if re.search(rf"(?:优先|喜欢|尽量|通常)[^，。；;]{{0,8}}{marker}", text)
    ]


def _budget_style(text: str) -> str | None:
    if any(marker in text for marker in ("穷游", "省钱", "经济型", "性价比")):
        return "economy"
    if any(marker in text for marker in ("舒适优先", "品质优先", "预算宽松")):
        return "comfort"
    return None


def _accessibility_needs(text: str) -> list[str]:
    values: list[str] = []
    if "轮椅" in text:
        values.append("wheelchair_accessible")
    if any(marker in text for marker in ("行动不便", "无障碍")):
        values.append("step_free_access")
    if any(marker in text for marker in ("需要频繁休息", "经常休息")):
        values.append("frequent_rest")
    return values
