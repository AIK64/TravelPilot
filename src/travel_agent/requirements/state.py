from __future__ import annotations

from typing import NotRequired, TypedDict

from travel_agent.domain.models import PlanningResponse, TripSpec
from travel_agent.domain.tool_models import POIFacts, ToolExecutionSummary, ToolResult
from travel_agent.requirements.anchors import AnchorSearchIntent
from travel_agent.requirements.models import (
    AnchorResolution,
    ClarificationResumeValue,
    NaturalPlanningRequest,
    RequirementDraft,
    RequirementExecutionSummary,
    RequirementIssue,
    RequirementPatch,
)
from travel_agent.memory.extraction import (
    ExtractedPreferenceCandidate,
    PreferenceEvidence,
    RejectedPreferenceCandidate,
)
from travel_agent.memory.models import (
    MemoryProposal,
    PreferenceContext,
    PreferenceLearningStatus,
)


class RequirementState(TypedDict):
    execution: NotRequired[dict | None]
    thread_id: str
    tenant_id: str
    user_id: str
    natural_request: NaturalPlanningRequest
    requirement_draft: RequirementDraft | None
    requirement_issues: list[RequirementIssue]
    clarification_questions: list[str]
    clarification_round: int
    max_clarification_rounds: int
    clarification_target_fields: list[str]
    clarification_input: ClarificationResumeValue | None
    clarification_patch: RequirementPatch | None
    changed_fields: list[str]
    rejected_patch_fields: list[str]
    clarification_exhausted: bool
    anchor_search_plan: list[AnchorSearchIntent]
    anchor_results: list[ToolResult[list[POIFacts]]]
    anchor_resolutions: dict[str, AnchorResolution]
    reused_anchor_roles: list[str]
    invalidated_anchor_roles: list[str]
    llm_summaries: list[RequirementExecutionSummary]
    tool_summaries: list[ToolExecutionSummary]
    trip: TripSpec | None
    preference_context: PreferenceContext | None
    personalized_fields: list[str]
    planning_response: PlanningResponse | None
    preference_evidence: list[PreferenceEvidence]
    extracted_preference_candidates: list[ExtractedPreferenceCandidate]
    accepted_preference_candidates: list[ExtractedPreferenceCandidate]
    rejected_preference_candidates: list[RejectedPreferenceCandidate]
    memory_proposals: list[MemoryProposal]
    preference_learning_status: PreferenceLearningStatus
    preference_deduplicated_count: int
    status: str
    message: str | None
