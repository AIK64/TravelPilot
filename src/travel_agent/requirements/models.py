from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from travel_agent.domain.models import (
    Coordinate,
    MobilityConstraints,
    Pace,
    PlanningResponse,
    TripSpec,
)
from travel_agent.memory.models import (
    MemoryProposal,
    PreferenceContext,
    PreferenceLearningStatus,
)


class RequirementProviderMode(StrEnum):
    MOCK = "mock"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class RequirementIssueCode(StrEnum):
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"
    INVALID = "invalid"


class RequirementModelStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class RequirementOperation(StrEnum):
    INITIAL_PARSE = "initial_parse"
    CLARIFICATION_PATCH = "clarification_patch"


class AnchorDraft(BaseModel):
    """LLM 从用户原文抽取的地点名称与时间，不包含推测坐标。"""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    at: datetime | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class RequirementDraft(BaseModel):
    """允许字段缺失的语义抽取结果；不能直接作为规划输入。"""

    model_config = ConfigDict(extra="forbid")

    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    travelers: int | None = None
    arrival: AnchorDraft | None = None
    departure: AnchorDraft | None = None
    accommodation_name: str | None = None
    total_budget: Decimal | None = None
    interests: list[str] = Field(default_factory=list, max_length=100)
    avoid: list[str] = Field(default_factory=list, max_length=100)
    must_visit: list[str] = Field(default_factory=list, max_length=100)
    pace: Pace | None = None
    mobility: MobilityConstraints | None = None
    daily_start: time | None = None
    daily_end: time | None = None

    @field_validator("destination", "accommodation_name")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("interests", "avoid", "must_visit")
    @classmethod
    def normalize_terms(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            item = value.strip()
            key = item.casefold()
            if not item or key in seen:
                continue
            seen.add(key)
            normalized.append(item)
        return normalized


class RequirementPatch(BaseModel):
    """补充回答的候选字段补丁；最终可写字段仍由确定性白名单控制。"""

    model_config = ConfigDict(extra="forbid")

    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    travelers: int | None = None
    arrival: AnchorDraft | None = None
    departure: AnchorDraft | None = None
    accommodation_name: str | None = None
    total_budget: Decimal | None = None
    interests: list[str] | None = None
    avoid: list[str] | None = None
    must_visit: list[str] | None = None
    pace: Pace | None = None
    mobility: MobilityConstraints | None = None
    daily_start: time | None = None
    daily_end: time | None = None

    @field_validator("destination", "accommodation_name")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("interests", "avoid", "must_visit")
    @classmethod
    def normalize_optional_terms(
        cls,
        values: list[str] | None,
    ) -> list[str] | None:
        if values is None:
            return None
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            item = value.strip()
            key = item.casefold()
            if not item or key in seen:
                continue
            seen.add(key)
            normalized.append(item)
        return normalized


class RequirementIssue(BaseModel):
    code: RequirementIssueCode
    field: str
    message: str
    question: str | None = None
    blocking: bool = True


class AnchorResolution(BaseModel):
    role: Literal["arrival", "departure", "accommodation"]
    query_name: str
    resolved_name: str
    poi_id: str
    coordinate: Coordinate
    provider: str
    data_confidence: float = Field(ge=0, le=1)


class RequirementProviderOutput(BaseModel):
    draft: RequirementDraft
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ClarificationModelInput(BaseModel):
    answer: str = Field(min_length=1, max_length=2_000)
    current_draft: RequirementDraft
    target_fields: list[str] = Field(min_length=1)
    issues: list[RequirementIssue] = Field(min_length=1)
    reference_date: date
    timezone: str


class RequirementPatchProviderOutput(BaseModel):
    patch: RequirementPatch
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class RequirementExecutionSummary(BaseModel):
    provider: str
    model: str
    prompt_version: str
    operation: RequirementOperation = RequirementOperation.INITIAL_PARSE
    status: RequirementModelStatus
    attempt_count: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class RequirementModelResult(BaseModel):
    draft: RequirementDraft
    summary: RequirementExecutionSummary


class RequirementPatchModelResult(BaseModel):
    patch: RequirementPatch
    summary: RequirementExecutionSummary


class ClarificationInterruptPayload(BaseModel):
    kind: Literal["requirement_clarification"] = "requirement_clarification"
    round: int = Field(ge=1)
    max_rounds: int = Field(ge=1)
    target_fields: list[str] = Field(min_length=1)
    issues: list[RequirementIssue] = Field(min_length=1)
    questions: list[str] = Field(min_length=1)


class ClarificationInterrupt(BaseModel):
    id: str
    payload: ClarificationInterruptPayload


class ClarificationResumeRequest(BaseModel):
    interrupt_id: str = Field(min_length=1, max_length=256)
    request_id: UUID
    answer: str = Field(min_length=1, max_length=2_000)

    @field_validator("interrupt_id", "answer")
    @classmethod
    def normalize_resume_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class ClarificationResumeValue(BaseModel):
    interrupt_id: str = Field(min_length=1, max_length=256)
    request_id: UUID
    answer: str = Field(min_length=1, max_length=2_000)

    @field_validator("interrupt_id", "answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer must not be blank")
        return normalized


class NaturalPlanningRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)
    reference_date: date = Field(default_factory=date.today)
    timezone: str = "Asia/Shanghai"
    max_replan_rounds: int = Field(default=2, ge=0, le=5)
    max_clarification_rounds: int = Field(default=3, ge=1, le=5)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        normalized = value.strip()
        try:
            ZoneInfo(normalized)
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a valid IANA timezone") from error
        return normalized


class NaturalPlanningResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "infeasible", "needs_clarification", "failed"]
    trip: TripSpec | None = None
    issues: list[RequirementIssue] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    clarification_round: int = Field(default=0, ge=0)
    can_resume: bool = False
    interrupt: ClarificationInterrupt | None = None
    planning: PlanningResponse | None = None
    preference_context: PreferenceContext | None = None
    personalized_fields: list[str] = Field(default_factory=list)
    preference_learning_status: PreferenceLearningStatus = (
        PreferenceLearningStatus.NOT_RUN
    )
    memory_proposals: list[MemoryProposal] = Field(default_factory=list)
    preference_extracted_count: int = Field(default=0, ge=0)
    preference_rejected_count: int = Field(default=0, ge=0)
    preference_deduplicated_count: int = Field(default=0, ge=0)
    message: str | None = None
