from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class AgentPhase(StrEnum):
    RESEARCHING = "researching"
    SOLVED = "solved"
    INVALID = "invalid"
    REPAIRING = "repairing"
    VALIDATED = "validated"
    AWAITING_USER = "awaiting_user"
    TERMINAL = "terminal"


class ActionBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str = Field(default_factory=lambda: str(uuid4()))
    reason_code: str = Field(min_length=1, max_length=64)
    decision_summary: str = Field(min_length=1, max_length=512)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)


class ClarifyAction(ActionBase):
    kind: Literal["clarify"] = "clarify"
    fields: tuple[str, ...] = Field(min_length=1, max_length=3)
    question: str = Field(min_length=1, max_length=512)


class CallToolAction(ActionBase):
    kind: Literal["call_tool"] = "call_tool"
    tool_name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, object] = Field(max_length=20)
    evidence_goal: str = Field(min_length=1, max_length=256)


class SolveAction(ActionBase):
    kind: Literal["solve"] = "solve"
    strategy: Literal["relaxed", "balanced", "exploration", "auto"] = "auto"


class RepairAction(ActionBase):
    kind: Literal["repair"] = "repair"
    target_candidate_id: str = Field(min_length=1, max_length=128)
    target_violation_fingerprints: tuple[str, ...] = Field(
        min_length=1, max_length=16
    )


class FinishAction(ActionBase):
    kind: Literal["finish"] = "finish"
    candidate_id: str = Field(min_length=1, max_length=128)


class EscalateAction(ActionBase):
    kind: Literal["escalate"] = "escalate"
    issue_code: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=512)


AgentAction = Annotated[
    Union[
        ClarifyAction,
        CallToolAction,
        SolveAction,
        RepairAction,
        FinishAction,
        EscalateAction,
    ],
    Field(discriminator="kind"),
]
AGENT_ACTION_ADAPTER = TypeAdapter(AgentAction)


class PlannerDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["planner-decision-v1"] = "planner-decision-v1"
    action: AgentAction
    confidence: float = Field(ge=0, le=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ActionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str
    action_fingerprint: str = Field(min_length=16, max_length=128)
    kind: str
    phase: AgentPhase
    status: Literal["validated", "rejected", "dispatched", "completed"]
    reason_code: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def action_fingerprint(action: AgentAction, phase: AgentPhase) -> str:
    value = action.model_dump(mode="json", exclude={"action_id", "decision_summary"})
    value["phase"] = phase.value
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
