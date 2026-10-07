from __future__ import annotations

from typing import NotRequired, TypedDict

from travel_agent.agents.actions import ActionRecord, AgentAction, AgentPhase, PlannerDecision
from travel_agent.agents.context import DynamicPlannerContext
from travel_agent.domain.models import PlanCandidate, PlanningRequest, TripSpec
from travel_agent.domain.critique_models import (
    CandidateEvidenceDigest,
    CriticExecutionSummary,
    CriticStatus,
    GroundedExplanation,
    SoftCritique,
    SoftRepairAttempt,
    SoftRepairPlan,
)
from travel_agent.domain.tool_models import POIFacts, RouteResult
from travel_agent.evidence.models import AgentObservation, EvidenceGap, EvidenceRecord
from travel_agent.planning.kernel import PlanningKernelSnapshot


class AgenticTravelState(TypedDict):
    execution: NotRequired[dict | None]
    thread_id: str
    run_id: str
    request: PlanningRequest
    trip: TripSpec
    phase: AgentPhase
    evidence_gaps: list[EvidenceGap]
    evidence_records: list[EvidenceRecord]
    recent_observations: list[AgentObservation]
    planner_context: DynamicPlannerContext | None
    last_decision: PlannerDecision | None
    last_action: AgentAction | None
    last_action_allowed: bool
    last_action_fingerprint: str | None
    action_history: list[ActionRecord]
    decision_round: int
    invalid_action_count: int
    repair_proposal_fingerprints: list[str]
    asked_clarification_fields: list[str]
    poi_facts: list[POIFacts]
    route_results: dict[str, RouteResult]
    candidates: list[PlanCandidate]
    selected_plan: PlanCandidate | None
    kernel_snapshot: PlanningKernelSnapshot | None
    iterations: int
    critic_evidence_digests: list[CandidateEvidenceDigest]
    soft_critiques: list[SoftCritique]
    critic_execution_summary: CriticExecutionSummary | None
    critic_status: CriticStatus
    critic_grounding_attempts: int
    critic_grounding_errors: list[str]
    quality_scores: dict[str, float]
    quality_review_complete: bool
    grounded_explanation: GroundedExplanation | None
    soft_repair_plan: SoftRepairPlan | None
    soft_repair_history: list[SoftRepairAttempt]
    soft_iterations: int
    soft_baseline_snapshot: PlanningKernelSnapshot | None
    soft_baseline_evidence_digests: list[CandidateEvidenceDigest]
    soft_baseline_critiques: list[SoftCritique]
    soft_baseline_quality_scores: dict[str, float]
    soft_baseline_critic_execution_summary: CriticExecutionSummary | None
    soft_baseline_critic_status: CriticStatus
    soft_repaired_candidate_id: str | None
    soft_repair_reused_routes: int
    soft_repair_loaded_routes: int
    agent_mode: str
    terminal_reason: str | None
    degraded_reasons: list[str]
    status: str
    message: str | None
