from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceKind(StrEnum):
    POI = "poi"
    ROUTE = "route"
    WEATHER = "weather"
    USER_CONSTRAINT = "user_constraint"
    PREFERENCE = "preference"
    VALIDATION = "validation"
    QUALITY = "quality"


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["evidence-record-v1"] = "evidence-record-v1"
    evidence_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    run_id: str = Field(min_length=1, max_length=128)
    kind: EvidenceKind
    subject_key: str = Field(min_length=1, max_length=256)
    summary: str = Field(min_length=1, max_length=800)
    provider: str | None = Field(default=None, max_length=80)
    observed_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    confidence: float = Field(ge=0, le=1)
    payload_ref: str | None = Field(default=None, max_length=512)
    content_hash: str = Field(min_length=16, max_length=128)

    @model_validator(mode="after")
    def validate_expiry(self) -> "EvidenceRecord":
        if self.expires_at is not None and self.expires_at <= self.observed_at:
            raise ValueError("expires_at must be later than observed_at")
        return self

    def active_at(self, now: datetime | None = None) -> bool:
        current = now or utcnow()
        return self.expires_at is None or self.expires_at > current


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    kind: EvidenceKind
    subject_key: str
    summary: str = Field(max_length=800)
    confidence: float = Field(ge=0, le=1)
    expires_at: datetime | None = None

    @classmethod
    def from_record(cls, record: EvidenceRecord) -> "EvidenceSummary":
        return cls(**record.model_dump(include=set(cls.model_fields)))


class EvidenceGapStatus(StrEnum):
    OPEN = "open"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"
    USER_REQUIRED = "user_required"


class EvidenceGap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, max_length=256)
    kind: EvidenceKind
    required: bool
    status: EvidenceGapStatus
    reason_code: str = Field(min_length=1, max_length=64)
    satisfied_by: tuple[str, ...] = Field(default=(), max_length=64)


class ObservationKind(StrEnum):
    TOOL_RESULT = "tool_result"
    TOOL_FAILURE = "tool_failure"
    USER_INPUT = "user_input"
    ACTION_REJECTED = "action_rejected"
    SOLVE_RESULT = "solve_result"
    VALIDATION_RESULT = "validation_result"
    REPAIR_RESULT = "repair_result"


class AgentObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["agent-observation-v1"] = "agent-observation-v1"
    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    kind: ObservationKind
    action_id: str | None = None
    summary: str = Field(min_length=1, max_length=1_000)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=64)
    error_code: str | None = Field(default=None, max_length=128)
    retryable: bool = False
    created_at: datetime = Field(default_factory=utcnow)


def evidence_content_hash(
    *, kind: EvidenceKind, subject_key: str, summary: str
) -> str:
    encoded = json.dumps(
        {"kind": kind.value, "subject_key": subject_key, "summary": summary},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
