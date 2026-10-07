from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunKind(StrEnum):
    STRUCTURED_PLAN = "structured_plan"
    NATURAL_PLAN = "natural_plan"
    CLARIFICATION_RESUME = "clarification_resume"
    LIFECYCLE_CREATE = "lifecycle_create"
    LIFECYCLE_CREATE_FROM_TEXT = "lifecycle_create_from_text"
    LIFECYCLE_RESUME = "lifecycle_resume"
    WEATHER_REFRESH = "weather_refresh"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    REPLAYED = "replayed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunTerminalReason(StrEnum):
    PLAN_COMPLETED = "plan_completed"
    BUSINESS_INFEASIBLE = "business_infeasible"
    NEEDS_CLARIFICATION = "needs_clarification"
    AWAITING_CANDIDATE_SELECTION = "awaiting_candidate_selection"
    AWAITING_USER_ACTION = "awaiting_user_action"
    AWAITING_APPROVAL = "awaiting_approval"
    REQUIRES_NEW_PLAN = "requires_new_plan"
    IDEMPOTENT_REPLAY = "idempotent_replay"
    EXECUTION_BUDGET_EXHAUSTED = "execution_budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    EXTERNAL_TOOL_FAILURE = "external_tool_failure"
    LLM_PROVIDER_FAILURE = "llm_provider_failure"
    CHECKPOINT_FAILURE = "checkpoint_failure"
    REPOSITORY_FAILURE = "repository_failure"
    INVALID_INTERNAL_STATE = "invalid_internal_state"
    USER_CANCELLED = "user_cancelled"
    INVALID_AGENT_DECISION = "invalid_agent_decision"
    AGENT_NO_PROGRESS = "agent_no_progress"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    ACTION_POLICY_REJECTED = "action_policy_rejected"


class TraceStatus(StrEnum):
    COMPLETE = "complete"
    DEGRADED = "degraded"


class TraceEventType(StrEnum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_INTERRUPTED = "run.interrupted"
    RUN_FAILED = "run.failed"
    RUN_REPLAYED = "run.replayed"
    RUN_CANCELLED = "run.cancelled"
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    ROUTE_DECIDED = "route.decided"
    TOOL_STARTED = "tool.started"
    TOOL_PROVIDER_ATTEMPT = "tool.provider_attempt"
    TOOL_CACHE_HIT = "tool.cache_hit"
    TOOL_RETRY = "tool.retry"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    LLM_STARTED = "llm.started"
    LLM_ATTEMPT = "llm.attempt"
    LLM_RETRY = "llm.retry"
    LLM_COMPLETED = "llm.completed"
    LLM_FAILED = "llm.failed"
    VALIDATION_COMPLETED = "validation.completed"
    REPAIR_PLANNED = "repair.planned"
    REPAIR_APPLIED = "repair.applied"
    REPAIR_NO_PROGRESS = "repair.no_progress"
    INTERRUPT_CREATED = "interrupt.created"
    INTERRUPT_RESUMED = "interrupt.resumed"
    CHECKPOINT_READ = "checkpoint.read"
    CHECKPOINT_WRITTEN = "checkpoint.written"
    CHECKPOINT_FAILED = "checkpoint.failed"
    REPOSITORY_CREATED = "repository.created"
    REPOSITORY_READ = "repository.read"
    REPOSITORY_SAVED = "repository.saved"
    REPOSITORY_CAS_CONFLICT = "repository.cas_conflict"
    REPOSITORY_FAILED = "repository.failed"
    BUDGET_UPDATED = "budget.updated"
    BUDGET_EXCEEDED = "budget.exceeded"
    DEGRADATION_APPLIED = "degradation.applied"
    STAY_ANCHOR_RESOLVED = "stay_anchor.resolved"
    DAY_BOUNDARIES_RESOLVED = "day_boundaries.resolved"
    MEMORY_NAMESPACE_RESOLVED = "memory.namespace_resolved"
    MEMORY_RETRIEVE_STARTED = "memory.retrieve_started"
    MEMORY_RETRIEVE_COMPLETED = "memory.retrieve_completed"
    MEMORY_EXTRACTION_STARTED = "memory.extraction_started"
    MEMORY_EXTRACTION_COMPLETED = "memory.extraction_completed"
    MEMORY_CANDIDATE_REJECTED = "memory.candidate_rejected"
    MEMORY_CANDIDATE_DEDUPLICATED = "memory.candidate_deduplicated"
    MEMORY_EXTRACTION_DEGRADED = "memory.extraction_degraded"
    MEMORY_CONFLICT_DETECTED = "memory.conflict_detected"
    CONTEXT_COMPOSED = "context.composed"
    MEMORY_PROPOSAL_CREATED = "memory.proposal_created"
    MEMORY_CONFIRMATION_REQUIRED = "memory.confirmation_required"
    MEMORY_PERSISTED = "memory.persisted"
    MEMORY_REVOKED = "memory.revoked"
    AGENT_HANDOFF_STARTED = "agent.handoff_started"
    AGENT_HANDOFF_COMPLETED = "agent.handoff_completed"
    AGENT_HANDOFF_REJECTED = "agent.handoff_rejected"
    PROVIDER_ATTEMPT_STARTED = "provider.attempt_started"
    PROVIDER_ATTEMPT_COMPLETED = "provider.attempt_completed"
    PROVIDER_FALLBACK_SELECTED = "provider.fallback_selected"
    PROVIDER_CHAIN_EXHAUSTED = "provider.chain_exhausted"
    AGENT_DECISION_STARTED = "agent.decision_started"
    AGENT_DECISION_COMPLETED = "agent.decision_completed"
    AGENT_DECISION_FAILED = "agent.decision_failed"
    AGENT_ACTION_VALIDATED = "agent.action_validated"
    AGENT_ACTION_REJECTED = "agent.action_rejected"
    AGENT_ACTION_DISPATCHED = "agent.action_dispatched"
    AGENT_OBSERVATION_RECORDED = "agent.observation_recorded"
    EVIDENCE_GAP_DERIVED = "evidence.gap_derived"
    EVIDENCE_RECORDED = "evidence.recorded"
    EVIDENCE_EXPIRED = "evidence.expired"
    REPLANNER_PROPOSAL_CREATED = "replanner.proposal_created"
    REPLANNER_PROPOSAL_REJECTED = "replanner.proposal_rejected"
    FINAL_GUARD_COMPLETED = "final_guard.completed"
    AGENT_NO_PROGRESS = "agent.no_progress"


class ExecutionBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: str = "default-v1"
    max_graph_steps: int = Field(default=120, ge=4, le=10_000)
    max_tool_calls: int = Field(default=160, ge=1, le=10_000)
    max_provider_attempts: int = Field(default=240, ge=1, le=20_000)
    max_llm_calls: int = Field(default=8, ge=1, le=1_000)
    max_llm_attempts: int = Field(default=12, ge=1, le=2_000)
    max_llm_input_chars: int = Field(default=80_000, ge=1, le=10_000_000)
    max_input_tokens: int = Field(default=40_000, ge=1, le=10_000_000)
    max_output_tokens: int = Field(default=12_000, ge=1, le=10_000_000)
    max_repair_rounds: int = Field(default=4, ge=0, le=100)
    max_interrupts: int = Field(default=1, ge=0, le=20)
    max_checkpoint_writes: int = Field(default=160, ge=1, le=20_000)
    max_trace_events: int = Field(default=512, ge=8, le=100_000)
    max_repeated_fingerprint_count: int = Field(default=2, ge=1, le=100)
    deadline_ms: int = Field(default=120_000, ge=100, le=3_600_000)
    max_estimated_cost_microunits: int | None = Field(default=None, ge=1)
    terminal_step_reserve: int = Field(default=2, ge=1, le=100)
    terminal_trace_reserve: int = Field(default=4, ge=2, le=100)
    max_agent_decisions: int = Field(default=12, ge=1, le=1_000)
    max_invalid_actions: int = Field(default=2, ge=0, le=100)
    max_evidence_records: int = Field(default=256, ge=1, le=100_000)
    max_observation_history: int = Field(default=8, ge=1, le=1_000)
    max_planner_context_tokens: int = Field(default=6_000, ge=128, le=1_000_000)
    max_replanner_context_tokens: int = Field(default=5_000, ge=128, le=1_000_000)

    @model_validator(mode="after")
    def validate_reserves(self) -> "ExecutionBudget":
        if self.terminal_step_reserve >= self.max_graph_steps:
            raise ValueError("terminal_step_reserve must be below max_graph_steps")
        if self.terminal_trace_reserve >= self.max_trace_events:
            raise ValueError("terminal_trace_reserve must be below max_trace_events")
        if self.max_provider_attempts < self.max_tool_calls:
            raise ValueError("max_provider_attempts must cover max_tool_calls")
        if self.max_llm_attempts < self.max_llm_calls:
            raise ValueError("max_llm_attempts must cover max_llm_calls")
        return self


class ExecutionUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    graph_steps: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    provider_attempts: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    llm_attempts: int = Field(default=0, ge=0)
    llm_input_chars: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=0, ge=0)
    output_tokens: int | None = Field(default=0, ge=0)
    repair_rounds: int = Field(default=0, ge=0)
    interrupts: int = Field(default=0, ge=0)
    checkpoint_writes: int = Field(default=0, ge=0)
    trace_events: int = Field(default=0, ge=0)
    estimated_cost_microunits: int | None = Field(default=None, ge=0)
    agent_decisions: int = Field(default=0, ge=0)
    invalid_actions: int = Field(default=0, ge=0)
    evidence_records: int = Field(default=0, ge=0)
    observations: int = Field(default=0, ge=0)


JsonScalar = str | int | float | bool | None


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "trace-event-v1"
    event_id: str
    run_id: str
    sequence: int = Field(ge=1)
    event_type: TraceEventType
    timestamp: datetime
    monotonic_offset_ms: int = Field(ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    graph: str | None = None
    node: str | None = None
    operation: str | None = None
    status: str
    parent_event_id: str | None = None
    attempt: int | None = Field(default=None, ge=1)
    plan_version_id: str | None = None
    attributes: dict[str, JsonScalar] = Field(default_factory=dict)


class AgentRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "agent-run-v1"
    run_id: str
    run_kind: RunKind
    status: RunStatus
    terminal_reason: RunTerminalReason | None = None
    thread_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    parent_run_id: str | None = None
    replay_of_run_id: str | None = None
    causation_id: str | None = None
    plan_version_id: str | None = None
    budget: ExecutionBudget
    usage: ExecutionUsage = Field(default_factory=ExecutionUsage)
    degraded_reasons: tuple[str, ...] = ()
    trace_status: TraceStatus = TraceStatus.COMPLETE
    started_at: datetime
    ended_at: datetime | None = None
    elapsed_ms: int | None = Field(default=None, ge=0)
    config_fingerprint: str
    tenant_id: str = "local"
    user_id: str = "demo"


class TracePage(BaseModel):
    run_id: str
    events: tuple[TraceEvent, ...]
    next_sequence: int | None = None


class RunList(BaseModel):
    runs: tuple[AgentRunRecord, ...]


def safe_status_value(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)
