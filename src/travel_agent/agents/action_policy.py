from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from travel_agent.agents.actions import (
    AgentAction,
    AgentPhase,
    CallToolAction,
    ClarifyAction,
    FinishAction,
    RepairAction,
    SolveAction,
    action_fingerprint,
)
from travel_agent.evidence.models import EvidenceGap, EvidenceGapStatus
from travel_agent.memory.models import AgentRole
from travel_agent.tools.registry import ToolRegistry
from travel_agent.tools.agent_executor import TOOL_INPUT_MODELS


_ALLOWED_ACTIONS: dict[AgentPhase, frozenset[str]] = {
    AgentPhase.RESEARCHING: frozenset({"clarify", "call_tool", "solve", "escalate"}),
    AgentPhase.SOLVED: frozenset({"call_tool", "repair", "finish", "escalate"}),
    AgentPhase.INVALID: frozenset({"call_tool", "repair", "escalate"}),
    AgentPhase.REPAIRING: frozenset({"call_tool", "repair", "escalate"}),
    AgentPhase.VALIDATED: frozenset({"finish", "repair", "escalate"}),
    AgentPhase.AWAITING_USER: frozenset(),
    AgentPhase.TERMINAL: frozenset(),
}


class ActionPolicyResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason_code: str
    action_fingerprint: str = Field(min_length=16, max_length=128)
    validation_feedback: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActionPolicyContext:
    phase: AgentPhase
    run_id: str
    evidence_ids: frozenset[str]
    evidence_gaps: tuple[EvidenceGap, ...]
    action_fingerprints: tuple[str, ...]
    registry: ToolRegistry
    hard_validated_candidate_ids: frozenset[str] = frozenset()
    known_candidate_ids: frozenset[str] = frozenset()
    unresolved_violation_fingerprints: frozenset[str] = frozenset()
    asked_clarification_fields: frozenset[str] = frozenset()
    budget_allows_side_effect: bool = True


def validate_action(
    action: AgentAction, context: ActionPolicyContext
) -> ActionPolicyResult:
    fingerprint = action_fingerprint(action, context.phase)

    def reject(code: str, feedback: tuple[str, ...] = ()) -> ActionPolicyResult:
        return ActionPolicyResult(
            allowed=False, reason_code=code, action_fingerprint=fingerprint, validation_feedback=feedback
        )

    if action.kind not in _ALLOWED_ACTIONS[context.phase]:
        return reject("action_not_allowed_in_phase")
    if not set(action.evidence_ids).issubset(context.evidence_ids):
        return reject("unknown_evidence_reference")
    if fingerprint in context.action_fingerprints:
        return reject("repeated_action")
    if isinstance(action, CallToolAction):
        descriptor = context.registry.get(action.tool_name)
        if descriptor is None:
            return reject("unknown_tool")
        if AgentRole.PLANNER not in descriptor.allowed_roles:
            return reject("tool_role_not_allowed")
        if context.phase not in descriptor.allowed_phases:
            return reject("tool_phase_not_allowed")
        if descriptor.forbidden_argument_names.intersection(action.arguments):
            return reject("forbidden_tool_argument")
        input_model = TOOL_INPUT_MODELS.get(action.tool_name)
        if input_model is None:
            return reject("tool_schema_unavailable")
        try:
            input_model.model_validate(action.arguments)
        except ValidationError as error:
            feedback = tuple(
                f"{'.'.join(str(value) for value in item['loc'])}: {item['type']}"
                for item in error.errors(include_input=False, include_url=False)[:10]
            )
            return reject("invalid_tool_arguments", feedback)
        if not context.budget_allows_side_effect:
            return reject("action_budget_exhausted")
    if isinstance(action, SolveAction):
        unresolved = [
            gap
            for gap in context.evidence_gaps
            if gap.required and gap.status is not EvidenceGapStatus.SATISFIED
        ]
        if unresolved:
            return reject("required_evidence_missing")
    if isinstance(action, RepairAction):
        if action.target_candidate_id not in context.known_candidate_ids:
            return reject("unknown_candidate")
        if not set(action.target_violation_fingerprints).issubset(
            context.unresolved_violation_fingerprints
        ):
            return reject("unknown_violation")
    if isinstance(action, FinishAction):
        if action.candidate_id not in context.hard_validated_candidate_ids:
            return reject("candidate_not_hard_validated")
        if any(
            gap.required and gap.status is not EvidenceGapStatus.SATISFIED
            for gap in context.evidence_gaps
        ):
            return reject("required_evidence_missing")
    if isinstance(action, ClarifyAction):
        if set(action.fields).issubset(context.asked_clarification_fields):
            return reject("repeated_clarification")
        if not any(
            gap.required and gap.status is EvidenceGapStatus.USER_REQUIRED
            for gap in context.evidence_gaps
        ):
            return reject("clarification_not_required")
    return ActionPolicyResult(
        allowed=True, reason_code="action_allowed", action_fingerprint=fingerprint
    )
