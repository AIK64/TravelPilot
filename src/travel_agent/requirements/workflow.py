from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Literal, cast
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from travel_agent.domain.models import PlanningRequest
from travel_agent.domain.tool_models import (
    ToolCallContext,
    ToolExecutionSummary,
    ToolStatus,
)
from travel_agent.graph.workflow import _CHECKPOINT_ALLOWED_TYPES, run_planning
from travel_agent.execution.checkpoints import ObservedCheckpointSaver
from travel_agent.execution.instrumentation import (
    execution_budget_guard,
    instrument_node,
    instrument_route,
)
from travel_agent.execution.context import (
    current_run_id,
    record_degradation,
    record_memory_event,
)
from travel_agent.execution.models import TraceEventType
from travel_agent.requirements.anchors import (
    AnchorRole,
    build_anchor_search_plan,
    resolve_anchor_search_results,
)
from travel_agent.requirements.clarification import (
    clarification_target_fields,
    invalidated_anchor_roles,
    merge_requirement_patch,
)
from travel_agent.requirements.errors import (
    ClarificationResumeConflictError,
    ClarificationThreadNotFoundError,
)
from travel_agent.requirements.gateway import RequirementGateway
from travel_agent.requirements.models import (
    ClarificationInterrupt,
    ClarificationInterruptPayload,
    ClarificationModelInput,
    ClarificationResumeRequest,
    ClarificationResumeValue,
    NaturalPlanningRequest,
    NaturalPlanningResponse,
)
from travel_agent.requirements.state import RequirementState
from travel_agent.requirements.validation import (
    assemble_trip_spec as create_trip_spec,
    validate_requirement as check_requirement,
)
from travel_agent.identity.models import Principal
from travel_agent.memory.extraction import (
    extract_preference_candidates as extract_memory_candidates,
    prepare_preference_evidence as build_preference_evidence,
    validate_preference_candidates as check_preference_candidates,
)
from travel_agent.memory.models import (
    AgentRole,
    MemoryProposalRequest,
    PreferenceContext,
    PreferenceLearningStatus,
)
from travel_agent.memory.policy import preference_content_hash
from travel_agent.memory.service import PreferenceMemoryService
from travel_agent.tools.errors import ToolUnavailableError
from travel_agent.tools.gateway import ToolGateway


logger = logging.getLogger(__name__)
_REQUIREMENT_CHECKPOINT_ALLOWED_TYPES = (
    *_CHECKPOINT_ALLOWED_TYPES,
    ("travel_agent.domain.models", "PlanningResponse"),
    ("travel_agent.domain.tool_models", "ToolErrorInfo"),
    ("travel_agent.domain.tool_models", "ToolResult"),
    ("travel_agent.requirements.anchors", "AnchorSearchIntent"),
    ("travel_agent.requirements.models", "AnchorDraft"),
    ("travel_agent.requirements.models", "AnchorResolution"),
    ("travel_agent.requirements.models", "ClarificationResumeValue"),
    ("travel_agent.requirements.models", "NaturalPlanningRequest"),
    ("travel_agent.requirements.models", "RequirementDraft"),
    ("travel_agent.requirements.models", "RequirementExecutionSummary"),
    ("travel_agent.requirements.models", "RequirementIssue"),
    ("travel_agent.requirements.models", "RequirementIssueCode"),
    ("travel_agent.requirements.models", "RequirementModelStatus"),
    ("travel_agent.requirements.models", "RequirementOperation"),
    ("travel_agent.requirements.models", "RequirementPatch"),
    ("travel_agent.memory.models", "AgentRole"),
    ("travel_agent.memory.models", "ContextManifest"),
    ("travel_agent.memory.models", "MemoryCategory"),
    ("travel_agent.memory.models", "MemoryConflict"),
    ("travel_agent.memory.models", "MemorySource"),
    ("travel_agent.memory.models", "MemoryProposal"),
    ("travel_agent.memory.models", "PreferenceLearningStatus"),
    ("travel_agent.memory.models", "PreferenceScope"),
    ("travel_agent.memory.models", "ProposalStatus"),
    ("travel_agent.memory.models", "PreferenceContext"),
    ("travel_agent.memory.models", "PreferenceSummary"),
    ("travel_agent.memory.extraction", "ExtractedPreferenceCandidate"),
    ("travel_agent.memory.extraction", "PersistenceIntent"),
    ("travel_agent.memory.extraction", "PreferenceEvidence"),
    ("travel_agent.memory.extraction", "RejectedPreferenceCandidate"),
)


def requirement_checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=_REQUIREMENT_CHECKPOINT_ALLOWED_TYPES
    )


def _log_node(state: RequirementState, phase: str, node: str, status: str) -> None:
    logger.info(
        "requirement.node.%s | thread_id=%s node=%s status=%s",
        phase,
        state["thread_id"],
        node,
        status,
    )


def _tool_summary(result) -> ToolExecutionSummary:
    return ToolExecutionSummary(
        provider=result.provider,
        operation="anchor.resolve",
        status=result.status,
        cache_hit=result.cache_hit,
        attempt_count=result.attempt_count,
    )


def build_requirement_workflow(
    *,
    requirement_gateway: RequirementGateway,
    tool_gateway: ToolGateway,
    planning_workflow: CompiledStateGraph,
    planning_runner: Callable[[PlanningRequest, str], Awaitable[object]] | None = None,
    memory_service: PreferenceMemoryService | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """构建可中断、可恢复的需求澄清与规划流程。"""

    async def parse_requirement(state: RequirementState) -> dict:
        _log_node(state, "started", "parse_requirement", state["status"])
        result = await requirement_gateway.parse(
            state["natural_request"],
            thread_id=state["thread_id"],
        )
        _log_node(state, "completed", "parse_requirement", "parsed")
        return {
            "requirement_draft": result.draft,
            "llm_summaries": [*state["llm_summaries"], result.summary],
            "status": "parsed",
        }

    def validate_requirement(state: RequirementState) -> dict:
        _log_node(state, "started", "validate_requirement", state["status"])
        draft = state["requirement_draft"]
        assert draft is not None
        issues = check_requirement(
            draft,
            timezone_name=state["natural_request"].timezone,
        )
        logger.info(
            "requirement.validated | thread_id=%s issue_count=%s blocking_count=%s",
            state["thread_id"],
            len(issues),
            sum(issue.blocking for issue in issues),
        )
        _log_node(state, "completed", "validate_requirement", "validated")
        return {
            "requirement_issues": issues,
            "clarification_questions": [],
            "clarification_target_fields": [],
            "status": "validated",
        }

    def route_requirement(
        state: RequirementState,
    ) -> Literal["request_clarification", "resolve_anchors"]:
        next_node = (
            "request_clarification"
            if any(issue.blocking for issue in state["requirement_issues"])
            else "resolve_anchors"
        )
        logger.info(
            "requirement.routing_decision | thread_id=%s after=validate_requirement "
            "next=%s issue_count=%s",
            state["thread_id"],
            next_node,
            len(state["requirement_issues"]),
        )
        return next_node

    async def resolve_anchors(state: RequirementState) -> dict:
        _log_node(state, "started", "resolve_anchors", state["status"])
        draft = state["requirement_draft"]
        assert draft is not None
        required_roles = _required_anchor_roles(draft)
        cached_roles = set(state["anchor_resolutions"]) & required_roles
        roles_to_load = required_roles - cached_roles
        plan = build_anchor_search_plan(draft, roles=roles_to_load)
        results = []
        if plan:
            results = await tool_gateway.search_pois(
                [intent.query for intent in plan],
                ToolCallContext(thread_id=state["thread_id"]),
            )
        for result in results:
            if result.status is ToolStatus.FAILED:
                raise ToolUnavailableError.from_result(result, state["thread_id"])
        reused = sorted(cached_roles)
        logger.info(
            "anchor.resolution.plan | thread_id=%s requested_roles=%s "
            "reused_roles=%s query_count=%s",
            state["thread_id"],
            ",".join(sorted(roles_to_load)) or "none",
            ",".join(reused) or "none",
            len(plan),
        )
        _log_node(state, "completed", "resolve_anchors", "anchors_loaded")
        return {
            "anchor_search_plan": plan,
            "anchor_results": results,
            "reused_anchor_roles": reused,
            "tool_summaries": [
                *state["tool_summaries"],
                *[_tool_summary(result) for result in results],
            ],
            "status": "anchors_loaded",
        }

    def evaluate_anchors(state: RequirementState) -> dict:
        _log_node(state, "started", "evaluate_anchors", state["status"])
        new_resolutions, issues = resolve_anchor_search_results(
            state["anchor_search_plan"],
            state["anchor_results"],
        )
        resolutions = {**state["anchor_resolutions"], **new_resolutions}
        logger.info(
            "anchors.resolved | thread_id=%s resolution_count=%s "
            "new_resolution_count=%s issue_count=%s",
            state["thread_id"],
            len(resolutions),
            len(new_resolutions),
            len(issues),
        )
        _log_node(state, "completed", "evaluate_anchors", "anchors_evaluated")
        return {
            "anchor_resolutions": resolutions,
            "requirement_issues": issues,
            "status": "anchors_evaluated",
        }

    def route_anchors(
        state: RequirementState,
    ) -> Literal["request_clarification", "assemble_trip_spec"]:
        next_node = (
            "request_clarification"
            if any(issue.blocking for issue in state["requirement_issues"])
            else "assemble_trip_spec"
        )
        logger.info(
            "requirement.routing_decision | thread_id=%s after=evaluate_anchors "
            "next=%s issue_count=%s",
            state["thread_id"],
            next_node,
            len(state["requirement_issues"]),
        )
        return next_node

    def request_clarification(state: RequirementState) -> dict:
        _log_node(state, "started", "request_clarification", state["status"])
        questions = list(
            dict.fromkeys(
                issue.question or issue.message
                for issue in state["requirement_issues"]
                if issue.blocking
            )
        )
        targets = clarification_target_fields(state["requirement_issues"])
        if state["clarification_round"] >= state["max_clarification_rounds"]:
            logger.warning(
                "clarification.exhausted | thread_id=%s round=%s max_rounds=%s",
                state["thread_id"],
                state["clarification_round"],
                state["max_clarification_rounds"],
            )
            return {
                "clarification_questions": questions,
                "clarification_target_fields": targets,
                "clarification_exhausted": True,
                "status": "clarification_exhausted_pending",
            }

        next_round = state["clarification_round"] + 1
        logger.info(
            "clarification.prepared | thread_id=%s round=%s max_rounds=%s "
            "issue_count=%s target_fields=%s",
            state["thread_id"],
            next_round,
            state["max_clarification_rounds"],
            len(state["requirement_issues"]),
            ",".join(targets),
        )
        logger.info(
            "requirement.clarification.required | thread_id=%s issue_count=%s "
            "question_count=%s round=%s",
            state["thread_id"],
            len(state["requirement_issues"]),
            len(questions),
            next_round,
        )
        _log_node(
            state,
            "completed",
            "request_clarification",
            "needs_clarification",
        )
        return {
            "clarification_questions": questions,
            "clarification_round": next_round,
            "clarification_target_fields": targets,
            "clarification_input": None,
            "clarification_patch": None,
            "changed_fields": [],
            "rejected_patch_fields": [],
            "status": "needs_clarification",
            "message": "需要补充或确认旅行需求后才能继续规划",
        }

    def route_clarification_request(
        state: RequirementState,
    ) -> Literal["await_clarification", "clarification_exhausted"]:
        return (
            "clarification_exhausted"
            if state["clarification_exhausted"]
            else "await_clarification"
        )

    def await_clarification(state: RequirementState) -> dict:
        payload = ClarificationInterruptPayload(
            round=state["clarification_round"],
            max_rounds=state["max_clarification_rounds"],
            target_fields=state["clarification_target_fields"],
            issues=state["requirement_issues"],
            questions=state["clarification_questions"],
        )
        resume_value = ClarificationResumeValue.model_validate(
            interrupt(payload.model_dump(mode="json"))
        )
        logger.info(
            "clarification.resumed | thread_id=%s round=%s request_id=%s "
            "answer_chars=%s",
            state["thread_id"],
            state["clarification_round"],
            resume_value.request_id,
            len(resume_value.answer),
        )
        return {
            "clarification_input": resume_value,
            "status": "clarification_received",
            "message": None,
        }

    async def parse_clarification_patch(state: RequirementState) -> dict:
        _log_node(
            state,
            "started",
            "parse_clarification_patch",
            state["status"],
        )
        draft = state["requirement_draft"]
        resume_value = state["clarification_input"]
        assert draft is not None
        assert resume_value is not None
        result = await requirement_gateway.parse_clarification(
            ClarificationModelInput(
                answer=resume_value.answer,
                current_draft=draft,
                target_fields=state["clarification_target_fields"],
                issues=state["requirement_issues"],
                reference_date=state["natural_request"].reference_date,
                timezone=state["natural_request"].timezone,
            ),
            thread_id=state["thread_id"],
        )
        _log_node(
            state,
            "completed",
            "parse_clarification_patch",
            "clarification_parsed",
        )
        return {
            "clarification_patch": result.patch,
            "llm_summaries": [*state["llm_summaries"], result.summary],
            "status": "clarification_parsed",
        }

    def apply_clarification_patch(state: RequirementState) -> dict:
        _log_node(
            state,
            "started",
            "apply_clarification_patch",
            state["status"],
        )
        draft = state["requirement_draft"]
        patch = state["clarification_patch"]
        assert draft is not None
        assert patch is not None
        merged, changed, rejected = merge_requirement_patch(
            draft,
            patch,
            allowed_fields=state["clarification_target_fields"],
        )
        invalidated = invalidated_anchor_roles(changed)
        resolutions = dict(state["anchor_resolutions"])
        for role in invalidated:
            resolutions.pop(role, None)
        logger.info(
            "clarification.patch.applied | thread_id=%s round=%s "
            "changed_fields=%s rejected_fields=%s invalidated_roles=%s",
            state["thread_id"],
            state["clarification_round"],
            ",".join(changed) or "none",
            ",".join(rejected) or "none",
            ",".join(invalidated) or "none",
        )
        _log_node(
            state,
            "completed",
            "apply_clarification_patch",
            "patch_applied" if changed else "patch_ignored",
        )
        return {
            "requirement_draft": merged,
            "changed_fields": changed,
            "rejected_patch_fields": rejected,
            "invalidated_anchor_roles": invalidated,
            "anchor_resolutions": resolutions,
            "clarification_input": None,
            "clarification_patch": None,
            "status": "patch_applied" if changed else "patch_ignored",
        }

    def route_after_patch(
        state: RequirementState,
    ) -> Literal["validate_requirement", "request_clarification"]:
        next_node = (
            "validate_requirement"
            if state["changed_fields"]
            else "request_clarification"
        )
        logger.info(
            "clarification.routing_decision | thread_id=%s round=%s next=%s "
            "changed_count=%s",
            state["thread_id"],
            state["clarification_round"],
            next_node,
            len(state["changed_fields"]),
        )
        return next_node

    def clarification_exhausted(state: RequirementState) -> dict:
        _log_node(
            state,
            "completed",
            "clarification_exhausted",
            "needs_clarification",
        )
        return {
            "status": "needs_clarification",
            "message": "已达到需求澄清轮次上限，请创建新的规划请求",
        }

    def assemble_trip_spec(state: RequirementState) -> dict:
        _log_node(state, "started", "assemble_trip_spec", state["status"])
        draft = state["requirement_draft"]
        assert draft is not None
        trip = create_trip_spec(
            draft,
            state["anchor_resolutions"],
            timezone_name=state["natural_request"].timezone,
        )
        logger.info(
            "trip_spec.assembled | thread_id=%s destination=%s days=%s "
            "travelers=%s",
            state["thread_id"],
            trip.destination,
            trip.day_count,
            trip.travelers,
        )
        _log_node(state, "completed", "assemble_trip_spec", "trip_assembled")
        return {"trip": trip, "status": "trip_assembled"}

    async def retrieve_preferences(state: RequirementState) -> dict:
        _log_node(state, "started", "retrieve_relevant_preferences", state["status"])
        trip = state["trip"]
        draft = state["requirement_draft"]
        assert trip is not None
        assert draft is not None
        assert memory_service is not None
        context = await memory_service.context_for_trip(
            Principal(
                tenant_id=state["tenant_id"],
                user_id=state["user_id"],
                scopes=frozenset({"preferences:read"}),
            ),
            trip=trip,
            draft=draft,
            agent_role=AgentRole.PLANNER,
        )
        personalized = memory_service.apply_to_trip(
            trip,
            draft=draft,
            context=context,
        )
        changed = [
            field
            for field in ("pace", "mobility", "interests", "avoid")
            if getattr(personalized, field) != getattr(trip, field)
        ]
        logger.info(
            "memory.applied | thread_id=%s context_id=%s selected=%s "
            "personalized_fields=%s",
            state["thread_id"],
            context.manifest.context_id,
            len(context.summaries),
            ",".join(changed) or "none",
        )
        _log_node(
            state,
            "completed",
            "retrieve_relevant_preferences",
            "preferences_composed",
        )
        return {
            "trip": personalized,
            "preference_context": context,
            "personalized_fields": changed,
            "status": "preferences_composed",
        }

    async def execute_planning(state: RequirementState) -> dict:
        _log_node(state, "started", "execute_planning", state["status"])
        trip = state["trip"]
        assert trip is not None
        planning_request = PlanningRequest(
            trip=trip,
            max_replan_rounds=state["natural_request"].max_replan_rounds,
        )
        response = (
            await planning_runner(planning_request, state["thread_id"])
            if planning_runner is not None
            else await run_planning(
                planning_workflow,
                planning_request,
                thread_id=state["thread_id"],
            )
        )
        _log_node(state, "completed", "execute_planning", response.status)
        return {
            "planning_response": response,
            "status": response.status,
            "message": response.message,
        }

    def prepare_memory_evidence(state: RequirementState) -> dict:
        _log_node(state, "started", "prepare_memory_evidence", state["status"])
        record_memory_event(
            TraceEventType.MEMORY_EXTRACTION_STARTED,
            status="started",
            operation="memory.extract_preferences",
            attributes={"policy_version": "preference-extraction-v1"},
        )
        draft = state["requirement_draft"]
        assert draft is not None
        clarification = state["clarification_input"]
        try:
            evidence = build_preference_evidence(
                state["natural_request"],
                draft,
                clarification_answer=clarification.answer if clarification else None,
            )
        except Exception:
            logger.exception(
                "memory.evidence_preparation_failed | thread_id=%s",
                state["thread_id"],
            )
            record_degradation("preference_evidence_failure")
            record_memory_event(
                TraceEventType.MEMORY_EXTRACTION_DEGRADED,
                status="degraded",
                operation="memory.prepare_evidence",
                attributes={"reason_code": "evidence_preparation_failure"},
            )
            return {
                "preference_evidence": [],
                "preference_learning_status": PreferenceLearningStatus.DEGRADED,
            }
        logger.info(
            "memory.extraction_evidence_prepared | thread_id=%s evidence_count=%s",
            state["thread_id"],
            len(evidence),
        )
        _log_node(state, "completed", "prepare_memory_evidence", state["status"])
        return {
            "preference_evidence": evidence,
            "preference_learning_status": (
                PreferenceLearningStatus.COMPLETED
                if evidence
                else PreferenceLearningStatus.SKIPPED
            ),
        }

    def extract_preference_candidates_node(state: RequirementState) -> dict:
        _log_node(
            state, "started", "extract_preference_candidates", state["status"]
        )
        try:
            candidates = extract_memory_candidates(state["preference_evidence"])
        except Exception:
            logger.exception(
                "memory.extraction_failed | thread_id=%s", state["thread_id"]
            )
            record_degradation("preference_extraction_failure")
            record_memory_event(
                TraceEventType.MEMORY_EXTRACTION_DEGRADED,
                status="degraded",
                operation="memory.extract_preferences",
                attributes={"reason_code": "extractor_failure"},
            )
            return {
                "preference_learning_status": PreferenceLearningStatus.DEGRADED
            }
        logger.info(
            "memory.extraction_completed | thread_id=%s candidate_count=%s",
            state["thread_id"],
            len(candidates),
        )
        record_memory_event(
            TraceEventType.MEMORY_EXTRACTION_COMPLETED,
            status="success",
            operation="memory.extract_preferences",
            attributes={
                "evidence_count": len(state["preference_evidence"]),
                "extracted_count": len(candidates),
            },
        )
        _log_node(
            state, "completed", "extract_preference_candidates", state["status"]
        )
        return {"extracted_preference_candidates": candidates}

    def validate_preference_candidates_node(state: RequirementState) -> dict:
        _log_node(
            state, "started", "validate_preference_candidates", state["status"]
        )
        if state["preference_learning_status"] is PreferenceLearningStatus.DEGRADED:
            return {}
        accepted, rejected = check_preference_candidates(
            state["extracted_preference_candidates"]
        )
        logger.info(
            "memory.candidates_validated | thread_id=%s accepted=%s rejected=%s "
            "reason_codes=%s",
            state["thread_id"],
            len(accepted),
            len(rejected),
            ",".join(sorted({item.reason_code for item in rejected})) or "none",
        )
        if rejected:
            record_memory_event(
                TraceEventType.MEMORY_CANDIDATE_REJECTED,
                status="rejected",
                operation="memory.validate_candidates",
                attributes={
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "reason_code": ",".join(
                        sorted({item.reason_code for item in rejected})
                    ),
                },
            )
        _log_node(
            state, "completed", "validate_preference_candidates", state["status"]
        )
        return {
            "accepted_preference_candidates": accepted,
            "rejected_preference_candidates": rejected,
        }

    async def create_memory_proposals(state: RequirementState) -> dict:
        _log_node(state, "started", "create_memory_proposals", state["status"])
        assert memory_service is not None
        if state["preference_learning_status"] is PreferenceLearningStatus.DEGRADED:
            return {}
        principal = Principal(
            tenant_id=state["tenant_id"],
            user_id=state["user_id"],
            scopes=frozenset({"preferences:read", "preferences:write"}),
        )
        proposals = []
        deduplicated = 0
        try:
            for candidate in state["accepted_preference_candidates"][:5]:
                content_hash = preference_content_hash(
                    category=candidate.category,
                    value=candidate.value,
                    scope=candidate.scope,
                    scope_key=candidate.scope_key,
                )
                confirmed = await memory_service.repository.find_content_hash(
                    principal.tenant_id, principal.user_id, content_hash
                )
                if confirmed is not None:
                    deduplicated += 1
                    continue
                pending = await memory_service.repository.find_pending_proposal(
                    principal.tenant_id, principal.user_id, content_hash
                )
                if pending is not None:
                    deduplicated += 1
                    proposals.append(pending)
                    continue
                proposal = await memory_service.propose(
                    principal,
                    MemoryProposalRequest(
                        category=candidate.category,
                        value=candidate.value,
                        scope=candidate.scope,
                        scope_key=candidate.scope_key,
                        source=candidate.source,
                        source_run_id=current_run_id() or state["thread_id"],
                        confidence=candidate.confidence,
                        reason=candidate.reason,
                    ),
                )
                proposals.append(proposal)
        except Exception:
            logger.exception(
                "memory.proposal_pipeline_failed | thread_id=%s",
                state["thread_id"],
            )
            record_degradation("preference_proposal_failure")
            record_memory_event(
                TraceEventType.MEMORY_EXTRACTION_DEGRADED,
                status="degraded",
                operation="memory.create_proposals",
                attributes={"reason_code": "proposal_persistence_failure"},
            )
            return {
                "memory_proposals": proposals,
                "preference_deduplicated_count": deduplicated,
                "preference_learning_status": PreferenceLearningStatus.DEGRADED,
            }
        status = (
            PreferenceLearningStatus.COMPLETED
            if state["preference_evidence"]
            else PreferenceLearningStatus.SKIPPED
        )
        logger.info(
            "memory.proposals_ready | thread_id=%s proposal_count=%s "
            "deduplicated_count=%s",
            state["thread_id"],
            len(proposals),
            deduplicated,
        )
        if deduplicated:
            record_memory_event(
                TraceEventType.MEMORY_CANDIDATE_DEDUPLICATED,
                status="deduplicated",
                operation="memory.create_proposals",
                attributes={"deduplicated_count": deduplicated},
            )
        _log_node(state, "completed", "create_memory_proposals", state["status"])
        return {
            "memory_proposals": proposals,
            "preference_deduplicated_count": deduplicated,
            "preference_learning_status": status,
        }

    def finalize_response(state: RequirementState) -> dict:
        _log_node(state, "completed", "finalize_response", state["status"])
        return {}

    builder = StateGraph(RequirementState)

    def add_node(name: str, function, *, terminal: bool = False) -> None:
        builder.add_node(
            name, instrument_node("requirements", name, function, terminal=terminal)
        )

    add_node("execution_budget_guard", execution_budget_guard)
    add_node("parse_requirement", parse_requirement)
    add_node("validate_requirement", validate_requirement)
    add_node("resolve_anchors", resolve_anchors)
    add_node("evaluate_anchors", evaluate_anchors)
    add_node("request_clarification", request_clarification)
    add_node("await_clarification", await_clarification, terminal=True)
    add_node("parse_clarification_patch", parse_clarification_patch)
    add_node("apply_clarification_patch", apply_clarification_patch)
    add_node("clarification_exhausted", clarification_exhausted, terminal=True)
    add_node("assemble_trip_spec", assemble_trip_spec)
    if memory_service is not None:
        add_node("retrieve_relevant_preferences", retrieve_preferences)
    add_node("execute_planning", execute_planning, terminal=memory_service is None)
    if memory_service is not None:
        add_node("prepare_memory_evidence", prepare_memory_evidence)
        add_node("extract_preference_candidates", extract_preference_candidates_node)
        add_node("validate_preference_candidates", validate_preference_candidates_node)
        add_node("create_memory_proposals", create_memory_proposals)
        add_node("finalize_response", finalize_response, terminal=True)
    builder.add_edge(START, "execution_budget_guard")
    builder.add_edge("execution_budget_guard", "parse_requirement")
    builder.add_edge("parse_requirement", "validate_requirement")
    builder.add_conditional_edges(
        "validate_requirement",
        instrument_route("requirements", "validate_requirement", route_requirement),
    )
    builder.add_edge("resolve_anchors", "evaluate_anchors")
    builder.add_conditional_edges(
        "evaluate_anchors",
        instrument_route("requirements", "evaluate_anchors", route_anchors),
    )
    builder.add_conditional_edges(
        "request_clarification",
        instrument_route(
            "requirements", "request_clarification", route_clarification_request
        ),
    )
    builder.add_edge("await_clarification", "parse_clarification_patch")
    builder.add_edge("parse_clarification_patch", "apply_clarification_patch")
    builder.add_conditional_edges(
        "apply_clarification_patch",
        instrument_route(
            "requirements", "apply_clarification_patch", route_after_patch
        ),
    )
    builder.add_edge("clarification_exhausted", END)
    if memory_service is None:
        builder.add_edge("assemble_trip_spec", "execute_planning")
    else:
        builder.add_edge("assemble_trip_spec", "retrieve_relevant_preferences")
        builder.add_edge("retrieve_relevant_preferences", "execute_planning")
    if memory_service is None:
        builder.add_edge("execute_planning", END)
    else:
        builder.add_edge("execute_planning", "prepare_memory_evidence")
        builder.add_edge("prepare_memory_evidence", "extract_preference_candidates")
        builder.add_edge(
            "extract_preference_candidates", "validate_preference_candidates"
        )
        builder.add_edge("validate_preference_candidates", "create_memory_proposals")
        builder.add_edge("create_memory_proposals", "finalize_response")
        builder.add_edge("finalize_response", END)
    resolved_checkpointer = checkpointer or ObservedCheckpointSaver(
        InMemorySaver(serde=requirement_checkpoint_serializer())
    )
    return builder.compile(checkpointer=resolved_checkpointer)


def initial_requirement_state(
    request: NaturalPlanningRequest,
    thread_id: str,
    *,
    tenant_id: str = "local",
    user_id: str = "demo",
) -> RequirementState:
    return {
        "thread_id": thread_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "natural_request": request,
        "requirement_draft": None,
        "requirement_issues": [],
        "clarification_questions": [],
        "clarification_round": 0,
        "max_clarification_rounds": request.max_clarification_rounds,
        "clarification_target_fields": [],
        "clarification_input": None,
        "clarification_patch": None,
        "changed_fields": [],
        "rejected_patch_fields": [],
        "clarification_exhausted": False,
        "anchor_search_plan": [],
        "anchor_results": [],
        "anchor_resolutions": {},
        "reused_anchor_roles": [],
        "invalidated_anchor_roles": [],
        "llm_summaries": [],
        "tool_summaries": [],
        "trip": None,
        "preference_context": None,
        "personalized_fields": [],
        "planning_response": None,
        "preference_evidence": [],
        "extracted_preference_candidates": [],
        "accepted_preference_candidates": [],
        "rejected_preference_candidates": [],
        "memory_proposals": [],
        "preference_learning_status": PreferenceLearningStatus.NOT_RUN,
        "preference_deduplicated_count": 0,
        "status": "started",
        "message": None,
    }


def response_from_requirement_state(
    state: RequirementState,
    *,
    interruptions: tuple[object, ...] = (),
) -> NaturalPlanningResponse:
    interrupt_view: ClarificationInterrupt | None = None
    if interruptions:
        if len(interruptions) != 1:
            raise RuntimeError("only one clarification interrupt is supported")
        item = interruptions[0]
        interrupt_view = ClarificationInterrupt(
            id=str(getattr(item, "id")),
            payload=ClarificationInterruptPayload.model_validate(
                getattr(item, "value")
            ),
        )
    return NaturalPlanningResponse(
        thread_id=state["thread_id"],
        status=cast(
            Literal["completed", "infeasible", "needs_clarification", "failed"],
            state["status"],
        ),
        trip=state["trip"],
        issues=state["requirement_issues"],
        clarification_questions=state["clarification_questions"],
        clarification_round=state["clarification_round"],
        can_resume=interrupt_view is not None,
        interrupt=interrupt_view,
        planning=state["planning_response"],
        preference_context=state["preference_context"],
        personalized_fields=state["personalized_fields"],
        preference_learning_status=state["preference_learning_status"],
        memory_proposals=state["memory_proposals"],
        preference_extracted_count=len(state["extracted_preference_candidates"]),
        preference_rejected_count=len(state["rejected_preference_candidates"]),
        preference_deduplicated_count=state["preference_deduplicated_count"],
        message=state["message"],
    )


async def run_natural_planning(
    workflow: CompiledStateGraph,
    request: NaturalPlanningRequest,
    *,
    thread_id: str | None = None,
    tenant_id: str = "local",
    user_id: str = "demo",
) -> NaturalPlanningResponse:
    run_thread_id = thread_id or str(uuid4())
    started = perf_counter()
    logger.info(
        "natural_planning.started | thread_id=%s input_chars=%s "
        "reference_date=%s timezone=%s max_clarification_rounds=%s",
        run_thread_id,
        len(request.text),
        request.reference_date,
        request.timezone,
        request.max_clarification_rounds,
    )
    try:
        result = await workflow.ainvoke(
            initial_requirement_state(
                request,
                run_thread_id,
                tenant_id=tenant_id,
                user_id=user_id,
            ),
            config=_graph_config(run_thread_id),
        )
    except Exception:
        logger.exception("natural_planning.failed | thread_id=%s", run_thread_id)
        raise
    return _finalize_response(result, run_thread_id, started)


async def resume_natural_planning(
    workflow: CompiledStateGraph,
    request: ClarificationResumeRequest,
    *,
    thread_id: str,
    tenant_id: str | None = None,
    user_id: str | None = None,
) -> NaturalPlanningResponse:
    config = _graph_config(thread_id)
    snapshot = await workflow.aget_state(config)
    if not snapshot.values:
        raise ClarificationThreadNotFoundError(thread_id=thread_id)
    if tenant_id is not None and snapshot.values.get("tenant_id") != tenant_id:
        raise ClarificationThreadNotFoundError(thread_id=thread_id)
    if user_id is not None and snapshot.values.get("user_id") != user_id:
        raise ClarificationThreadNotFoundError(thread_id=thread_id)
    interruptions = tuple(
        interruption
        for task in snapshot.tasks
        for interruption in getattr(task, "interrupts", ())
    )
    retry_failed_resume = False
    if not interruptions:
        stored_resume = snapshot.values.get("clarification_input")
        if stored_resume is not None and snapshot.next:
            stored_resume = ClarificationResumeValue.model_validate(stored_resume)
            retry_failed_resume = (
                stored_resume.interrupt_id == request.interrupt_id
                and stored_resume.request_id == request.request_id
                and stored_resume.answer == request.answer
            )
        if not retry_failed_resume:
            raise ClarificationResumeConflictError(thread_id=thread_id)
    elif request.interrupt_id not in {
        str(getattr(item, "id", "")) for item in interruptions
    }:
        raise ClarificationResumeConflictError(
            thread_id=thread_id,
            code="stale_interrupt",
        )

    started = perf_counter()
    logger.info(
        "natural_planning.resume_started | thread_id=%s interrupt_id=%s "
        "request_id=%s answer_chars=%s",
        thread_id,
        request.interrupt_id,
        request.request_id,
        len(request.answer),
    )
    try:
        graph_input = None
        if not retry_failed_resume:
            graph_input = Command(
                resume=ClarificationResumeValue(
                    interrupt_id=request.interrupt_id,
                    request_id=request.request_id,
                    answer=request.answer,
                ).model_dump(mode="json")
            )
        else:
            logger.info(
                "clarification.resume_retry | thread_id=%s interrupt_id=%s "
                "request_id=%s",
                thread_id,
                request.interrupt_id,
                request.request_id,
            )
        result = await workflow.ainvoke(graph_input, config=config)
    except Exception:
        logger.exception("natural_planning.resume_failed | thread_id=%s", thread_id)
        raise
    return _finalize_response(result, thread_id, started)


def _required_anchor_roles(draft) -> set[AnchorRole]:
    roles: set[AnchorRole] = set()
    if draft.arrival and draft.arrival.name:
        roles.add("arrival")
    if draft.departure and draft.departure.name:
        roles.add("departure")
    if draft.accommodation_name:
        roles.add("accommodation")
    return roles


def _graph_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }


def _finalize_response(
    result: dict,
    thread_id: str,
    started: float,
) -> NaturalPlanningResponse:
    interruptions = tuple(result.get("__interrupt__", ()))
    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    if interruptions:
        logger.info(
            "clarification.interrupted | thread_id=%s round=%s "
            "interrupt_id=%s elapsed_ms=%s",
            thread_id,
            result["clarification_round"],
            getattr(interruptions[0], "id", "unknown"),
            elapsed_ms,
        )
    else:
        logger.info(
            "natural_planning.completed | thread_id=%s status=%s elapsed_ms=%s",
            thread_id,
            result["status"],
            elapsed_ms,
        )
    return response_from_requirement_state(
        cast(RequirementState, result),
        interruptions=interruptions,
    )
