from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.domain.repair_models import RepairAction


class ModelRepairProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["model-repair-proposal-v1"] = "model-repair-proposal-v1"
    target_candidate_id: str
    source_violation_fingerprints: tuple[str, ...] = Field(min_length=1, max_length=16)
    actions: tuple[RepairAction, ...] = Field(min_length=1, max_length=3)
    affected_days: tuple[date, ...] = Field(max_length=2)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)
    expected_effect_codes: tuple[str, ...] = Field(min_length=1, max_length=8)
    summary: str = Field(min_length=1, max_length=512)


class ReplannerOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal: ModelRepairProposal
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


def proposal_fingerprint(proposal: ModelRepairProposal) -> str:
    encoded = json.dumps(
        proposal.model_dump(mode="json", exclude={"summary"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
