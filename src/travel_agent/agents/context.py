from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.domain.critique_models import SoftCriticRequest
from travel_agent.domain.models import PlanCandidate, PlanningPOI, TripSpec
from travel_agent.domain.repair_models import CriticReport, RepairAttempt
from travel_agent.agents.actions import AgentPhase
from travel_agent.evidence.models import AgentObservation, EvidenceGap, EvidenceSummary
from travel_agent.execution.models import ExecutionUsage
from travel_agent.memory.models import PreferenceSummary
from travel_agent.tools.registry import ToolDescriptor
from travel_agent.planning.drafts import CandidateDraft


class PlannerContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trip: TripSpec
    poi_query_limit: int = Field(ge=1, le=100)
    max_queries: int = Field(ge=1, le=100)


class CandidateSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    hard_valid: bool
    day_count: int = Field(ge=0)
    total_cost: str | None = None
    quality_reviewed: bool = False
    quality_score: float | None = Field(default=None, ge=0, le=100)


class ViolationSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint: str
    code: str
    severity: str
    candidate_id: str


class PlannerContextManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: str = "dynamic-planner-context-v1"
    evidence_ids: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    observation_count: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    character_count: int = Field(default=0, ge=0)


class DynamicPlannerContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: TripSpec
    phase: AgentPhase
    evidence_gaps: tuple[EvidenceGap, ...]
    evidence_catalog: tuple[EvidenceSummary, ...]
    recent_observations: tuple[AgentObservation, ...]
    candidate_summaries: tuple[CandidateSummary, ...] = ()
    violation_summaries: tuple[ViolationSummary, ...] = ()
    tool_manifest: tuple[ToolDescriptor, ...]
    memory_summaries: tuple[PreferenceSummary, ...] = ()
    budget_remaining: ExecutionUsage
    context_manifest: PlannerContextManifest


class CriticContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: SoftCriticRequest
    candidate_ids: tuple[str, ...]


class ReplannerContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trip: TripSpec
    candidate: PlanCandidate
    draft: CandidateDraft
    planning_pois: tuple[PlanningPOI, ...]
    critic_report: CriticReport
    repair_round: int = Field(ge=1)
    previous_action_fingerprints: frozenset[str] = frozenset()
