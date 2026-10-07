from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


PreferenceScalar = str | int | float | bool
PreferenceValue = (
    PreferenceScalar
    | list[PreferenceScalar]
    | dict[str, PreferenceScalar | list[PreferenceScalar]]
)


class MemoryCategory(StrEnum):
    PACE = "pace"
    PREFERRED_CATEGORIES = "preferred_categories"
    AVOIDED_CATEGORIES = "avoided_categories"
    WALKING_TOLERANCE = "walking_tolerance"
    PREFERRED_TRANSPORT = "preferred_transport"
    FOOD_PREFERENCES = "food_preferences"
    SCHEDULE_PREFERENCES = "schedule_preferences"
    ACCESSIBILITY_NEEDS = "accessibility_needs"
    BUDGET_STYLE = "budget_style"


class MemorySource(StrEnum):
    EXPLICIT_USER = "explicit_user"
    REPEATED_BEHAVIOR = "repeated_behavior"
    SINGLE_ACTION = "single_action"
    MODEL_INFERENCE = "model_inference"
    IMPORT = "import"


class ConfirmationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class PreferenceScope(StrEnum):
    GLOBAL = "global"
    DESTINATION = "destination"
    TRAVEL_PARTY = "travel_party"


class ProposalStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PreferenceLearningStatus(StrEnum):
    NOT_RUN = "not_run"
    SKIPPED = "skipped"
    COMPLETED = "completed"
    DEGRADED = "degraded"


class AgentRole(StrEnum):
    PLANNER = "planner"
    CRITIC = "critic"
    REPLANNER = "replanner"


class PreferenceMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["preference-memory-v1"] = "preference-memory-v1"
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    category: MemoryCategory
    value: PreferenceValue
    scope: PreferenceScope = PreferenceScope.GLOBAL
    scope_key: str | None = Field(default=None, max_length=256)
    source: MemorySource
    source_run_id: str | None = Field(default=None, max_length=128)
    confidence: float = Field(ge=0, le=1)
    confirmation_status: ConfirmationStatus
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    content_hash: str = Field(min_length=16, max_length=128)

    @model_validator(mode="after")
    def validate_scope_and_confirmation(self) -> "PreferenceMemory":
        if self.scope is PreferenceScope.GLOBAL and self.scope_key is not None:
            raise ValueError("global preference must not define scope_key")
        if self.scope is not PreferenceScope.GLOBAL and not self.scope_key:
            raise ValueError("scoped preference requires scope_key")
        if (
            self.source is MemorySource.MODEL_INFERENCE
            and self.confirmation_status is ConfirmationStatus.CONFIRMED
        ):
            raise ValueError("model inference cannot be directly confirmed")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        return self

    def active_at(self, now: datetime | None = None) -> bool:
        current = now or utcnow()
        return (
            self.confirmation_status is ConfirmationStatus.CONFIRMED
            and self.revoked_at is None
            and (self.expires_at is None or self.expires_at > current)
        )


class MemoryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["memory-proposal-v1"] = "memory-proposal-v1"
    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    category: MemoryCategory
    value: PreferenceValue
    scope: PreferenceScope = PreferenceScope.GLOBAL
    scope_key: str | None = Field(default=None, max_length=256)
    source: MemorySource
    source_run_id: str | None = Field(default=None, max_length=128)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    resolved_at: datetime | None = None
    memory_id: str | None = None
    content_hash: str = Field(min_length=16, max_length=128)

    @model_validator(mode="after")
    def validate_scope(self) -> "MemoryProposal":
        if self.scope is PreferenceScope.GLOBAL and self.scope_key is not None:
            raise ValueError("global proposal must not define scope_key")
        if self.scope is not PreferenceScope.GLOBAL and not self.scope_key:
            raise ValueError("scoped proposal requires scope_key")
        return self

    def pending_at(self, now: datetime | None = None) -> bool:
        current = now or utcnow()
        return (
            self.status is ProposalStatus.PENDING
            and (self.expires_at is None or self.expires_at > current)
        )


class MemoryConflict(BaseModel):
    conflict_id: str = Field(default_factory=lambda: str(uuid4()))
    category: MemoryCategory
    current_memory_id: str
    conflicting_memory_id: str
    resolution: Literal["current_request_wins", "user_confirmation_required"]
    detected_at: datetime = Field(default_factory=utcnow)


class PreferenceSummary(BaseModel):
    memory_id: str
    category: MemoryCategory
    value: PreferenceValue
    confidence: float = Field(ge=0, le=1)
    source: MemorySource
    reason: str = Field(min_length=1, max_length=300)
    score: float


class ContextManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_role: AgentRole
    policy_version: str = "memory-context-v1"
    selected_memory_ids: tuple[str, ...] = ()
    selected_categories: tuple[MemoryCategory, ...] = ()
    excluded_count: int = Field(default=0, ge=0)
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)
    estimated_tokens: int = Field(default=0, ge=0)
    character_count: int = Field(default=0, ge=0)
    max_tokens: int = Field(ge=1)
    max_characters: int = Field(ge=1)
    overridden_memory_ids: tuple[str, ...] = ()
    content_hash: str = Field(min_length=16, max_length=128)


class PreferenceContext(BaseModel):
    summaries: tuple[PreferenceSummary, ...] = ()
    manifest: ContextManifest
    conflicts: tuple[MemoryConflict, ...] = ()


class PersonalizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    user_id: str
    enabled: bool = True
    revision: int = Field(default=1, ge=1)
    updated_at: datetime = Field(default_factory=utcnow)


class PreferenceCreateRequest(BaseModel):
    category: MemoryCategory
    value: PreferenceValue
    scope: PreferenceScope = PreferenceScope.GLOBAL
    scope_key: str | None = Field(default=None, max_length=256)
    expires_at: datetime | None = None


class PreferenceUpdateRequest(BaseModel):
    value: PreferenceValue | None = None
    expires_at: datetime | None = None
    expected_revision: int = Field(ge=1)


class MemoryProposalRequest(BaseModel):
    category: MemoryCategory
    value: PreferenceValue
    scope: PreferenceScope = PreferenceScope.GLOBAL
    scope_key: str | None = Field(default=None, max_length=256)
    source: MemorySource = MemorySource.MODEL_INFERENCE
    source_run_id: str | None = Field(default=None, max_length=128)
    confidence: float = Field(default=0.5, ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    expires_at: datetime | None = None


class ProposalDecisionRequest(BaseModel):
    request_id: UUID


class PersonalizationUpdateRequest(BaseModel):
    enabled: bool
    expected_revision: int | None = Field(default=None, ge=1)


class PreferenceList(BaseModel):
    items: tuple[PreferenceMemory, ...]
    personalization: PersonalizationSettings


class PreferenceExport(BaseModel):
    schema_version: Literal["preference-export-v1"] = "preference-export-v1"
    exported_at: datetime = Field(default_factory=utcnow)
    items: tuple[PreferenceMemory, ...]
    personalization: PersonalizationSettings


TokenBudget = Annotated[int, Field(ge=32, le=100_000)]
