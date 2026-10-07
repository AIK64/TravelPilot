from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict

from travel_agent.agents.replanner.models import ModelRepairProposal, proposal_fingerprint
from travel_agent.domain.repair_models import RepairActionKind


class RepairPolicyResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason_code: str
    proposal_fingerprint: str


@dataclass(frozen=True, slots=True)
class RepairPolicyContext:
    candidate_ids: frozenset[str]
    poi_ids: frozenset[str]
    evidence_ids: frozenset[str]
    violation_fingerprints: frozenset[str]
    allowed_dates: frozenset[date]
    must_visit_poi_ids: frozenset[str] = frozenset()
    locked_poi_ids: frozenset[str] = frozenset()
    previous_proposal_fingerprints: frozenset[str] = frozenset()
    max_affected_days: int = 2


def validate_repair_proposal(
    proposal: ModelRepairProposal, context: RepairPolicyContext
) -> RepairPolicyResult:
    fingerprint = proposal_fingerprint(proposal)

    def reject(code: str) -> RepairPolicyResult:
        return RepairPolicyResult(
            allowed=False, reason_code=code, proposal_fingerprint=fingerprint
        )

    if proposal.target_candidate_id not in context.candidate_ids:
        return reject("unknown_candidate")
    if not set(proposal.source_violation_fingerprints).issubset(
        context.violation_fingerprints
    ):
        return reject("unknown_violation")
    if not set(proposal.evidence_ids).issubset(context.evidence_ids):
        return reject("unknown_evidence_reference")
    if fingerprint in context.previous_proposal_fingerprints:
        return reject("repeated_repair_proposal")
    if len(set(proposal.affected_days)) > context.max_affected_days:
        return reject("repair_scope_exceeded")
    if not set(proposal.affected_days).issubset(context.allowed_dates):
        return reject("repair_date_out_of_range")
    if not proposal.expected_effect_codes:
        return reject("expected_effect_missing")
    for action in proposal.actions:
        if action.poi_id not in context.poi_ids:
            return reject("unknown_poi")
        if action.poi_id in context.locked_poi_ids:
            return reject("locked_item_mutation")
        if (
            action.kind is RepairActionKind.REMOVE_OPTIONAL_POI
            and action.poi_id in context.must_visit_poi_ids
        ):
            return reject("must_visit_removal")
        action_dates = {value for value in (action.from_day, action.to_day) if value}
        if not action_dates.issubset(context.allowed_dates):
            return reject("repair_date_out_of_range")
    return RepairPolicyResult(
        allowed=True,
        reason_code="repair_proposal_allowed",
        proposal_fingerprint=fingerprint,
    )
