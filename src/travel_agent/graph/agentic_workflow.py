from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import logging
from typing import Literal, cast
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from travel_agent.agents.action_policy import ActionPolicyContext, validate_action
from travel_agent.agents.actions import (
    ActionRecord,
    AgentPhase,
    CallToolAction,
    ClarifyAction,
    EscalateAction,
    FinishAction,
    RepairAction,
    SolveAction,
    action_fingerprint,
)
from travel_agent.agents.context import (
    CandidateSummary,
    DynamicPlannerContext,
    PlannerContextManifest,
    ViolationSummary,
)
from travel_agent.agents.planner.gateway import PlannerGateway
from travel_agent.agents.replanner.gateway import ReplannerGateway
from travel_agent.agents.replanner.models import ModelRepairProposal
from travel_agent.agents.replanner.policy import RepairPolicyContext, validate_repair_proposal
from travel_agent.agents.context import ReplannerContext
from travel_agent.domain.critique_models import CriticStatus, SoftRepairAttempt
from travel_agent.domain.models import PlanningRequest, PlanningResponse
from travel_agent.evidence.models import (
    AgentObservation,
    EvidenceKind,
    EvidenceRecord,
    EvidenceSummary,
    ObservationKind,
    evidence_content_hash,
)
from travel_agent.evidence.policy import derive_evidence_gaps
from travel_agent.evidence.repository import EvidenceRepository
from travel_agent.execution.context import (
    consume_invalid_action,
    current_run_context,
    current_run_id,
    record_agent_event,
)
from travel_agent.execution.instrumentation import execution_budget_guard, instrument_node, instrument_route
from travel_agent.execution.models import ExecutionUsage, TraceEventType
from travel_agent.graph.agentic_state import AgenticTravelState
from travel_agent.memory.models import AgentRole
from travel_agent.planning.critic import analyze_candidate
from travel_agent.planning.kernel import (
    PlanningEvidenceMissingError,
    PlanningKernelService,
)
from travel_agent.planning.quality import QualityReviewResult, QualityReviewService
from travel_agent.planning.repair import build_repair_plan
from travel_agent.tools.agent_executor import AgentToolExecutor


logger = logging.getLogger(__name__)


def final_guard(state: AgenticTravelState, candidate_id: str) -> tuple[bool, str]:
    candidate = next(
        (item for item in state["candidates"] if item.id == candidate_id), None
    )
    if candidate is None:
        return False, "candidate_not_found"
    if candidate.validation is None or not candidate.validation.valid:
        return False, "candidate_not_hard_validated"
    required_open = [
        gap
        for gap in state["evidence_gaps"]
        if gap.required and gap.status.value != "satisfied"
    ]
    if required_open:
        return False, "required_evidence_missing"
    if state["kernel_snapshot"] is None:
        return False, "planning_snapshot_missing"
    if not state["quality_review_complete"]:
        return False, "quality_review_incomplete"
    return True, "final_guard_passed"


def build_agentic_workflow(
    *,
    planning_kernel: PlanningKernelService,
    planner_gateway: PlannerGateway,
    tool_executor: AgentToolExecutor,
    evidence_repository: EvidenceRepository,
    max_observation_history: int = 8,
    max_invalid_actions: int = 2,
    replanner_gateway: ReplannerGateway | None = None,
    quality_service: QualityReviewService | None = None,
    max_planner_context_tokens: int = 6_000,
) -> CompiledStateGraph:
    if max_observation_history < 1:
        raise ValueError("max_observation_history must be positive")
    quality_service = quality_service or QualityReviewService(critic_gateway=None)

    async def derive_gaps(state: AgenticTravelState) -> dict:
        records = await evidence_repository.list_for_run(state["run_id"])
        gaps = derive_evidence_gaps(
            state["trip"], records, candidate=state["selected_plan"]
        )
        record_agent_event(
            TraceEventType.EVIDENCE_GAP_DERIVED,
            status="derived",
            operation="evidence.derive_gaps",
            attributes={
                "gap_count": len(gaps),
                "required_gap_count": sum(
                    gap.required and gap.status.value != "satisfied" for gap in gaps
                ),
            },
        )
        return {"evidence_records": list(records), "evidence_gaps": list(gaps)}

    def compose_context(state: AgenticTravelState) -> dict:
        run = current_run_context()
        usage = run.ledger.snapshot() if run is not None else ExecutionUsage()
        candidate_summaries = tuple(
            CandidateSummary(
                candidate_id=item.id,
                hard_valid=bool(item.validation and item.validation.valid),
                day_count=len(item.days),
                total_cost=str(item.metrics.known_estimated_cost),
                quality_reviewed=state["quality_review_complete"],
                quality_score=state["quality_scores"].get(item.id),
            )
            for item in state["candidates"]
        )
        violations = tuple(
            ViolationSummary(
                fingerprint=_violation_fingerprint(item.id, violation),
                code=violation.type,
                severity=violation.severity.value,
                candidate_id=item.id,
            )
            for item in state["candidates"]
            for violation in (item.validation.violations if item.validation else [])
            if violation.severity.value == "error"
        )
        manifest_tools = tool_executor.registry.manifest(
            role=AgentRole.PLANNER, phase=state["phase"]
        )
        summaries = tuple(
            EvidenceSummary.from_record(item) for item in state["evidence_records"]
        )
        recent = tuple(state["recent_observations"][-max_observation_history:])
        draft = DynamicPlannerContext(
            goal=state["trip"],
            phase=state["phase"],
            evidence_gaps=tuple(state["evidence_gaps"]),
            evidence_catalog=summaries,
            recent_observations=recent,
            candidate_summaries=candidate_summaries,
            violation_summaries=violations,
            tool_manifest=manifest_tools,
            budget_remaining=usage,
            context_manifest=PlannerContextManifest(
                evidence_ids=tuple(item.evidence_id for item in summaries),
                tool_names=tuple(item.name for item in manifest_tools),
                observation_count=len(recent),
                estimated_tokens=0,
                character_count=0,
            ),
        )
        serialized = draft.model_dump_json()
        context_character_limit = max_planner_context_tokens * 4
        while len(serialized) > context_character_limit and recent:
            recent = recent[1:]
            draft = draft.model_copy(update={"recent_observations": recent})
            serialized = draft.model_dump_json()
        while len(serialized) > context_character_limit and summaries:
            summaries = summaries[1:]
            draft = draft.model_copy(update={"evidence_catalog": summaries})
            serialized = draft.model_dump_json()
        if len(serialized) > context_character_limit:
            raise RuntimeError("planner context exceeds configured token budget")
        context = draft.model_copy(
            update={
                "context_manifest": draft.context_manifest.model_copy(
                    update={
                        "evidence_ids": tuple(item.evidence_id for item in summaries),
                        "observation_count": len(recent),
                        "estimated_tokens": max(1, len(serialized) // 4),
                        "character_count": len(serialized),
                    }
                )
            }
        )
        return {"planner_context": context}

    async def decide(state: AgenticTravelState) -> dict:
        context = state["planner_context"]
        assert context is not None
        decision_round = state["decision_round"] + 1
        decision = await planner_gateway.decide(
            context, thread_id=state["thread_id"], decision_round=decision_round
        )
        return {
            "last_decision": decision,
            "last_action": decision.action,
            "decision_round": decision_round,
        }

    def guard_action(state: AgenticTravelState) -> dict:
        action = state["last_action"]
        assert action is not None
        candidate_ids = frozenset(item.id for item in state["candidates"])
        validated_ids = frozenset(
            item.id
            for item in state["candidates"]
            if item.validation is not None and item.validation.valid
        )
        violations = frozenset(
            _violation_fingerprint(item.id, violation)
            for item in state["candidates"]
            for violation in (item.validation.violations if item.validation else [])
            if violation.severity.value == "error"
        )
        result = validate_action(
            action,
            ActionPolicyContext(
                phase=state["phase"],
                run_id=state["run_id"],
                evidence_ids=frozenset(
                    item.evidence_id for item in state["evidence_records"]
                ),
                evidence_gaps=tuple(state["evidence_gaps"]),
                action_fingerprints=tuple(
                    item.action_fingerprint for item in state["action_history"]
                ),
                registry=tool_executor.registry,
                hard_validated_candidate_ids=validated_ids,
                known_candidate_ids=candidate_ids,
                unresolved_violation_fingerprints=violations,
                asked_clarification_fields=frozenset(
                    state["asked_clarification_fields"]
                ),
            ),
        )
        status: Literal["validated", "rejected"] = (
            "validated" if result.allowed else "rejected"
        )
        record = ActionRecord(
            action_id=action.action_id,
            action_fingerprint=result.action_fingerprint,
            kind=action.kind,
            phase=state["phase"],
            status=status,
            reason_code=result.reason_code,
        )
        event_type = (
            TraceEventType.AGENT_ACTION_VALIDATED
            if result.allowed
            else TraceEventType.AGENT_ACTION_REJECTED
        )
        record_agent_event(
            event_type,
            status=status,
            operation="agent.validate_action",
            attributes={
                "action_kind": action.kind,
                "action_fingerprint": result.action_fingerprint,
                "reason_code": result.reason_code,
                "guard_result": status,
            },
        )
        updates: dict = {
            "last_action_allowed": result.allowed,
            "last_action_fingerprint": result.action_fingerprint,
            "action_history": [*state["action_history"], record],
        }
        if not result.allowed:
            consume_invalid_action()
            observation = AgentObservation(
                kind=ObservationKind.ACTION_REJECTED,
                action_id=action.action_id,
                summary="Action Guard 拒绝了模型动作" + (
                    "；工具参数错误：" + "；".join(result.validation_feedback)
                    if result.validation_feedback else ""
                ),
                error_code=result.reason_code,
            )
            updates.update(
                {
                    "invalid_action_count": state["invalid_action_count"] + 1,
                    "recent_observations": _append_observation(state, observation),
                }
            )
            _trace_observation(observation)
        return updates

    def route_action(state: AgenticTravelState) -> str:
        if not state["last_action_allowed"]:
            if state["invalid_action_count"] >= max_invalid_actions:
                return "mark_no_progress"
            return "derive_evidence_gaps"
        action = state["last_action"]
        assert action is not None
        record_agent_event(
            TraceEventType.AGENT_ACTION_DISPATCHED,
            status="dispatched",
            operation="agent.dispatch_action",
            attributes={
                "action_kind": action.kind,
                "action_fingerprint": state["last_action_fingerprint"],
            },
            file_details={"request": action},
        )
        return {
            "call_tool": "execute_tool_action",
            "solve": "solve_plan",
            "finish": "final_guard",
            "repair": "apply_repair",
            "clarify": "request_clarification",
            "escalate": "request_clarification",
        }[action.kind]

    async def execute_tool(state: AgenticTravelState) -> dict:
        action = state["last_action"]
        assert isinstance(action, CallToolAction)
        result = await tool_executor.execute(
            action,
            trip=state["trip"],
            thread_id=state["thread_id"],
            run_id=state["run_id"],
            poi_facts=tuple(state["poi_facts"]),
        )
        records = list(state["evidence_records"])
        for item in result.evidence:
            stored = await evidence_repository.put(item)
            if stored.evidence_id not in {record.evidence_id for record in records}:
                records.append(stored)
        _trace_observation(result.observation, file_details={"request": action, "result": result})
        return {
            "evidence_records": records,
            "recent_observations": _append_observation(state, result.observation),
            "poi_facts": list(result.poi_facts) or state["poi_facts"],
            "route_results": {**state["route_results"], **result.route_results},
        }

    async def solve_plan(state: AgenticTravelState) -> dict:
        action = state["last_action"]
        assert isinstance(action, SolveAction)
        try:
            result = await planning_kernel.solve(
                state["trip"],
                poi_facts=tuple(state["poi_facts"]),
                route_results=state["route_results"],
                thread_id=state["thread_id"],
                strategy=action.strategy,
            )
        except PlanningEvidenceMissingError as error:
            observation = AgentObservation(
                kind=ObservationKind.SOLVE_RESULT,
                action_id=action.action_id,
                summary="求解前证据完整性检查失败，动态运行已安全终止",
                error_code=f"missing_{error.evidence_kind}_evidence",
            )
            _trace_observation(observation)
            return {
                "recent_observations": _append_observation(state, observation),
                "phase": AgentPhase.TERMINAL,
                "status": "failed",
                "terminal_reason": "evidence_unavailable",
                "message": str(error),
            }

        snapshot = result.snapshot
        candidate = result.selected_plan
        valid = candidate is not None
        record_agent_event(
            TraceEventType.VALIDATION_COMPLETED,
            status="validated" if valid else "invalid",
            operation="planning_kernel.validate",
            attributes={
                "candidate_count": len(snapshot.candidates),
                "deliverable_count": sum(
                    bool(item.validation and item.validation.valid)
                    for item in snapshot.candidates
                ),
            },
        )
        summary = f"约束求解生成 {len(snapshot.candidates)} 个候选，硬合法={valid}"
        observation = AgentObservation(
            kind=ObservationKind.SOLVE_RESULT,
            action_id=action.action_id,
            summary=summary,
        )
        records = list(state["evidence_records"])
        validation_ids: list[str] = []
        for validated_candidate in snapshot.candidates:
            candidate_valid = bool(
                validated_candidate.validation
                and validated_candidate.validation.valid
            )
            validation_summary = (
                f"候选 {validated_candidate.id} 硬约束校验 "
                f"{'通过' if candidate_valid else '未通过'}"
            )
            evidence = EvidenceRecord(
                run_id=state["run_id"],
                kind=EvidenceKind.VALIDATION,
                subject_key=f"candidate.validation:{validated_candidate.id}",
                summary=validation_summary,
                observed_at=datetime.now(timezone.utc),
                confidence=1.0,
                payload_ref=(
                    f"run:{state['run_id']}:candidate:{validated_candidate.id}"
                ),
                content_hash=evidence_content_hash(
                    kind=EvidenceKind.VALIDATION,
                    subject_key=f"candidate.validation:{validated_candidate.id}",
                    summary=validation_summary,
                ),
            )
            evidence = await evidence_repository.put(evidence)
            if evidence.evidence_id not in {item.evidence_id for item in records}:
                records.append(evidence)
            validation_ids.append(evidence.evidence_id)
            record_agent_event(
                TraceEventType.EVIDENCE_RECORDED,
                status="recorded",
                operation="evidence.record",
                attributes={
                    "evidence_id": evidence.evidence_id,
                    "evidence_kind": evidence.kind.value,
                    "candidate_id": validated_candidate.id,
                },
            )
        observation = observation.model_copy(
            update={"evidence_ids": tuple(validation_ids)}
        )
        _trace_observation(observation)
        return {
            "kernel_snapshot": snapshot,
            "candidates": list(snapshot.candidates),
            "selected_plan": candidate,
            "phase": AgentPhase.SOLVED if valid else AgentPhase.INVALID,
            "evidence_records": records,
            "recent_observations": _append_observation(state, observation),
            "route_results": snapshot.route_results,
            "iterations": snapshot.iterations,
            "critic_status": CriticStatus.NOT_RUN,
            "critic_execution_summary": None,
            "critic_evidence_digests": [],
            "soft_critiques": [],
            "critic_grounding_attempts": 0,
            "critic_grounding_errors": [],
            "quality_scores": {},
            "quality_review_complete": False,
            "grounded_explanation": None,
            "degraded_reasons": [
                *state["degraded_reasons"],
                *result.degraded_reasons,
            ],
        }

    def guard_finish(state: AgenticTravelState) -> dict:
        action = state["last_action"]
        assert isinstance(action, FinishAction)
        passed, reason = final_guard(state, action.candidate_id)
        record_agent_event(
            TraceEventType.FINAL_GUARD_COMPLETED,
            status="passed" if passed else "rejected",
            operation="agent.final_guard",
            attributes={"candidate_id": action.candidate_id, "reason_code": reason},
        )
        if passed:
            return {
                "phase": AgentPhase.TERMINAL,
                "status": "completed",
                "terminal_reason": "plan_completed",
                "message": f"已选择并通过硬约束校验的方案 {action.candidate_id}",
            }
        observation = AgentObservation(
            kind=ObservationKind.ACTION_REJECTED,
            action_id=action.action_id,
            summary="Final Guard 拒绝提前交付",
            error_code=reason,
        )
        _trace_observation(observation)
        return {
            "recent_observations": _append_observation(state, observation),
            "last_action_allowed": False,
        }

    def route_after_final_guard(state: AgenticTravelState) -> str:
        return "mark_terminal" if state["phase"] is AgentPhase.TERMINAL else "derive_evidence_gaps"

    def route_after_solve(state: AgenticTravelState) -> str:
        if state["phase"] is AgentPhase.TERMINAL:
            return "mark_terminal"
        if state["phase"] is AgentPhase.SOLVED:
            return "review_quality"
        return "derive_evidence_gaps"

    async def apply_repair(state: AgenticTravelState) -> dict:
        action = state["last_action"]
        assert isinstance(action, RepairAction)
        snapshot = state["kernel_snapshot"]
        candidate = next(
            (
                item
                for item in state["candidates"]
                if item.id == action.target_candidate_id
            ),
            None,
        )
        proposal_fingerprints = list(state["repair_proposal_fingerprints"])
        if snapshot is None or candidate is None or candidate.validation is None:
            raise RuntimeError("repair requires a solved kernel snapshot")
        if state["iterations"] >= state["request"].max_replan_rounds:
            observation = AgentObservation(
                kind=ObservationKind.REPAIR_RESULT,
                action_id=action.action_id,
                summary="局部修复预算已耗尽",
                error_code="repair_budget_exhausted",
            )
            _trace_observation(observation)
            return {
                "recent_observations": _append_observation(state, observation),
                "phase": AgentPhase.TERMINAL,
                "status": "infeasible",
                "terminal_reason": "business_infeasible",
                "message": "在局部修复预算内没有找到合法方案。",
            }

        draft = next(item for item in snapshot.candidate_drafts if item.id == candidate.id)
        report = analyze_candidate(
            candidate, state["trip"], list(snapshot.planning_pois)
        ).model_copy(
            update={
                "violation_fingerprint": action.target_violation_fingerprints[0]
            }
        )
        context = ReplannerContext(
            trip=state["trip"],
            candidate=candidate,
            draft=draft,
            planning_pois=snapshot.planning_pois,
            critic_report=report,
            repair_round=state["iterations"] + 1,
            previous_action_fingerprints=frozenset(proposal_fingerprints),
        )
        if replanner_gateway is not None:
            output = await replanner_gateway.propose(context)
            proposal = output.proposal
            repair_mode = "model"
        else:
            deterministic, terminal_reason = build_repair_plan(
                context.trip,
                context.candidate,
                context.draft,
                list(context.planning_pois),
                context.critic_report,
                repair_round=context.repair_round,
            )
            if deterministic is None:
                observation = AgentObservation(
                    kind=ObservationKind.REPAIR_RESULT,
                    action_id=action.action_id,
                    summary="确定性 Replanner 无法生成安全 Patch",
                    error_code=terminal_reason or "repair_unavailable",
                )
                _trace_observation(observation)
                return {
                    "recent_observations": _append_observation(state, observation),
                    "phase": AgentPhase.TERMINAL,
                    "status": "infeasible",
                    "terminal_reason": "business_infeasible",
                    "message": "没有找到不破坏硬约束的局部修复。",
                }
            if len(deterministic.actions) > 3 or len(deterministic.affected_days) > 2:
                observation = AgentObservation(
                    kind=ObservationKind.REPAIR_RESULT,
                    action_id=action.action_id,
                    summary="确定性兼容 Replanner 的修复范围超过动态 Patch 上限",
                    error_code="repair_scope_exceeded",
                )
                _trace_observation(observation)
                return {
                    "recent_observations": _append_observation(state, observation),
                    "phase": AgentPhase.TERMINAL,
                    "status": "infeasible",
                    "terminal_reason": "repair_policy_rejected",
                    "message": "安全策略拒绝了影响范围过大的修复。",
                }
            proposal = ModelRepairProposal(
                target_candidate_id=deterministic.target_candidate_id,
                source_violation_fingerprints=tuple(
                    action.target_violation_fingerprints
                ),
                actions=deterministic.actions,
                affected_days=deterministic.affected_days,
                expected_effect_codes=deterministic.source_violation_types,
                summary="确定性兼容 Replanner 生成的受控局部修复",
            )
            repair_mode = "deterministic"

        known_pois = frozenset(item.facts.id for item in snapshot.planning_pois)
        locked_pois = frozenset(
            item.poi_id
            for day in candidate.days
            for item in day.items
            if item.poi_id and item.locked
        )
        must_visit_pois = frozenset(
            item.facts.id
            for item in snapshot.planning_pois
            if any(
                required.casefold() in item.facts.name.casefold()
                or item.facts.name.casefold() in required.casefold()
                for required in state["trip"].must_visit
            )
        )
        policy_result = validate_repair_proposal(
            proposal,
            RepairPolicyContext(
                candidate_ids=frozenset({candidate.id}),
                poi_ids=known_pois,
                evidence_ids=frozenset(
                    item.evidence_id for item in state["evidence_records"]
                ),
                violation_fingerprints=frozenset(
                    action.target_violation_fingerprints
                ),
                allowed_dates=frozenset(day.date for day in candidate.days),
                must_visit_poi_ids=must_visit_pois,
                locked_poi_ids=locked_pois,
                previous_proposal_fingerprints=frozenset(proposal_fingerprints),
            ),
        )
        proposal_fingerprints.append(policy_result.proposal_fingerprint)
        if not policy_result.allowed:
            record_agent_event(
                TraceEventType.REPLANNER_PROPOSAL_REJECTED,
                status="rejected",
                operation="replanner.validate_proposal",
                attributes={
                    "candidate_id": candidate.id,
                    "reason_code": policy_result.reason_code,
                },
            )
            observation = AgentObservation(
                kind=ObservationKind.ACTION_REJECTED,
                action_id=action.action_id,
                summary="Repair Policy 拒绝了 Replanner Proposal",
                error_code=policy_result.reason_code,
            )
            _trace_observation(observation)
            return {
                "recent_observations": _append_observation(state, observation),
                "repair_proposal_fingerprints": proposal_fingerprints,
                "phase": AgentPhase.INVALID,
            }

        repaired = await planning_kernel.apply_repair(
            state["trip"], snapshot, proposal, thread_id=state["thread_id"]
        )
        repaired_candidate = next(
            item
            for item in repaired.snapshot.candidates
            if item.id.endswith(f"repair-r{repaired.attempt.round}")
        )
        valid = bool(
            repaired_candidate.validation and repaired_candidate.validation.valid
        )
        record_agent_event(
            TraceEventType.VALIDATION_COMPLETED,
            status="validated" if valid else "invalid",
            operation="planning_kernel.revalidate",
            attributes={
                "candidate_id": repaired_candidate.id,
                "repair_round": repaired.attempt.round,
                "repair_outcome": repaired.attempt.outcome.value,
            },
        )
        validation_summary = (
            f"修复候选 {repaired_candidate.id} 硬约束校验 "
            f"{'通过' if valid else '未通过'}"
        )
        validation_evidence = EvidenceRecord(
            run_id=state["run_id"],
            kind=EvidenceKind.VALIDATION,
            subject_key=f"candidate.validation:{repaired_candidate.id}",
            summary=validation_summary,
            observed_at=datetime.now(timezone.utc),
            confidence=1.0,
            payload_ref=(
                f"run:{state['run_id']}:candidate:{repaired_candidate.id}"
            ),
            content_hash=evidence_content_hash(
                kind=EvidenceKind.VALIDATION,
                subject_key=f"candidate.validation:{repaired_candidate.id}",
                summary=validation_summary,
            ),
        )
        validation_evidence = await evidence_repository.put(validation_evidence)
        records = [
            item
            for item in state["evidence_records"]
            if not item.subject_key.startswith(f"candidate.validation:{candidate.id}")
        ]
        records.append(validation_evidence)
        observation = AgentObservation(
            kind=ObservationKind.REPAIR_RESULT,
            action_id=action.action_id,
            summary=(
                f"{repair_mode} Replanner Patch 已应用并完成硬复验，"
                f"outcome={repaired.attempt.outcome.value}"
            ),
            evidence_ids=(validation_evidence.evidence_id,),
            error_code=repaired.terminal_reason,
        )
        _trace_observation(observation)
        return {
            "kernel_snapshot": repaired.snapshot,
            "candidates": list(repaired.snapshot.candidates),
            "selected_plan": repaired.selected_plan,
            "route_results": repaired.snapshot.route_results,
            "iterations": repaired.snapshot.iterations,
            "evidence_records": records,
            "recent_observations": _append_observation(state, observation),
            "repair_proposal_fingerprints": proposal_fingerprints,
            "phase": (
                AgentPhase.SOLVED
                if repaired.selected_plan is not None
                else AgentPhase.INVALID
            ),
            "critic_status": CriticStatus.NOT_RUN,
            "critic_execution_summary": None,
            "critic_evidence_digests": [],
            "soft_critiques": [],
            "critic_grounding_attempts": 0,
            "critic_grounding_errors": [],
            "quality_scores": {},
            "quality_review_complete": False,
            "grounded_explanation": None,
            "status": "running",
        }

    def _quality_review(state: AgenticTravelState) -> QualityReviewResult:
        summary = state["critic_execution_summary"]
        if summary is None:
            raise RuntimeError("quality review summary is unavailable")
        return QualityReviewResult(
            status=state["critic_status"],
            evidence_digests=tuple(state["critic_evidence_digests"]),
            critiques=tuple(state["soft_critiques"]),
            execution_summary=summary,
            quality_scores=dict(state["quality_scores"]),
            grounding_errors=tuple(state["critic_grounding_errors"]),
            grounding_attempt=state["critic_grounding_attempts"],
        )

    async def _quality_evidence(
        state: AgenticTravelState,
        *,
        candidate_id: str,
        status: CriticStatus,
        score: float | None,
    ) -> tuple[list[EvidenceRecord], EvidenceRecord]:
        summary = (
            f"候选 {candidate_id} 软质量评审已完成，"
            f"critic_status={status.value}，quality_score="
            f"{score if score is not None else 'deterministic_fallback'}"
        )
        record = EvidenceRecord(
            run_id=state["run_id"],
            kind=EvidenceKind.QUALITY,
            subject_key=f"candidate.quality_review:{candidate_id}",
            summary=summary,
            observed_at=datetime.now(timezone.utc),
            confidence=1.0 if status is CriticStatus.SUCCESS else 0.7,
            payload_ref=f"run:{state['run_id']}:quality:{candidate_id}",
            content_hash=evidence_content_hash(
                kind=EvidenceKind.QUALITY,
                subject_key=f"candidate.quality_review:{candidate_id}",
                summary=summary,
            ),
        )
        stored = await evidence_repository.put(record)
        records = [
            item
            for item in state["evidence_records"]
            if item.kind is not EvidenceKind.QUALITY
        ]
        records.append(stored)
        record_agent_event(
            TraceEventType.EVIDENCE_RECORDED,
            status="recorded",
            operation="quality.review",
            attributes={
                "evidence_id": stored.evidence_id,
                "candidate_id": candidate_id,
                "critic_status": status.value,
            },
        )
        return records, stored

    async def review_quality(state: AgenticTravelState) -> dict:
        snapshot = state["kernel_snapshot"]
        if snapshot is None or state["selected_plan"] is None:
            raise RuntimeError("quality review requires a hard-valid planning snapshot")
        attempt = state["critic_grounding_attempts"] + 1
        target_ids = (
            (state["soft_repaired_candidate_id"],)
            if state["soft_baseline_snapshot"] is not None
            and state["soft_repaired_candidate_id"] is not None
            else ()
        )
        review = await quality_service.review(
            state["trip"],
            snapshot,
            thread_id=state["thread_id"],
            grounding_attempt=attempt,
            grounding_feedback=tuple(state["critic_grounding_errors"]),
            candidate_ids=target_ids,
        )
        observation = AgentObservation(
            kind=ObservationKind.VALIDATION_RESULT,
            action_id=(state["last_action"].action_id if state["last_action"] else None),
            summary=(
                f"Grounded Soft Critic 完成，status={review.status.value}，"
                f"candidate_count={len(review.evidence_digests)}"
            ),
            error_code=(
                review.grounding_errors[0] if review.grounding_errors else None
            ),
        )
        _trace_observation(observation)
        return {
            "critic_evidence_digests": list(review.evidence_digests),
            "soft_critiques": list(review.critiques),
            "critic_execution_summary": review.execution_summary,
            "critic_status": review.status,
            "critic_grounding_attempts": review.grounding_attempt,
            "critic_grounding_errors": list(review.grounding_errors),
            "quality_scores": review.quality_scores,
            "recent_observations": _append_observation(state, observation),
        }

    def route_after_quality_review(state: AgenticTravelState) -> str:
        if (
            state["critic_status"] is CriticStatus.INVALID_GROUNDING
            and state["critic_grounding_attempts"]
            < quality_service.policy.grounding_max_attempts
        ):
            return "review_quality"
        if state["soft_baseline_snapshot"] is not None:
            return "compare_soft_repair"
        return "quality_gate"

    async def quality_gate(state: AgenticTravelState) -> dict:
        snapshot = state["kernel_snapshot"]
        if snapshot is None:
            raise RuntimeError("quality gate requires a planning snapshot")
        review = _quality_review(state)
        selected = quality_service.select(snapshot, review)
        plan, reason = quality_service.compile_improvement(
            state["trip"],
            snapshot,
            review,
            soft_iterations=state["soft_iterations"],
        )
        logger.info(
            "agentic.quality_gate | thread_id=%s candidate_id=%s status=%s "
            "score=%s repair=%s reason=%s",
            state["thread_id"],
            selected.id,
            review.status.value,
            review.quality_scores.get(selected.id),
            bool(plan),
            reason or "none",
        )
        if plan is not None:
            observation = AgentObservation(
                kind=ObservationKind.REPAIR_RESULT,
                action_id=(
                    state["last_action"].action_id if state["last_action"] else None
                ),
                summary=(
                    f"软质量低于阈值，已编译受控改进动作 "
                    f"{plan.action.kind.value}"
                ),
            )
            _trace_observation(observation)
            return {
                "selected_plan": selected,
                "soft_repair_plan": plan,
                "soft_baseline_snapshot": snapshot,
                "soft_baseline_evidence_digests": list(review.evidence_digests),
                "soft_baseline_critiques": list(review.critiques),
                "soft_baseline_quality_scores": dict(review.quality_scores),
                "soft_baseline_critic_execution_summary": review.execution_summary,
                "soft_baseline_critic_status": review.status,
                "phase": AgentPhase.REPAIRING,
                "recent_observations": _append_observation(state, observation),
            }

        explanation = quality_service.explain(selected, review)
        records, evidence = await _quality_evidence(
            state,
            candidate_id=selected.id,
            status=review.status,
            score=review.quality_scores.get(selected.id),
        )
        observation = AgentObservation(
            kind=ObservationKind.VALIDATION_RESULT,
            action_id=(state["last_action"].action_id if state["last_action"] else None),
            summary=f"质量门禁完成，选择候选 {selected.id}",
            evidence_ids=(evidence.evidence_id,),
        )
        _trace_observation(observation)
        return {
            "selected_plan": selected,
            "grounded_explanation": explanation,
            "quality_review_complete": True,
            "phase": AgentPhase.VALIDATED,
            "evidence_records": records,
            "recent_observations": _append_observation(state, observation),
        }

    def route_after_quality_gate(state: AgenticTravelState) -> str:
        return (
            "apply_soft_repair"
            if state["soft_repair_plan"] is not None
            and not state["quality_review_complete"]
            else "derive_evidence_gaps"
        )

    async def apply_soft_repair(state: AgenticTravelState) -> dict:
        snapshot = state["kernel_snapshot"]
        plan = state["soft_repair_plan"]
        if snapshot is None or plan is None:
            raise RuntimeError("soft repair requires snapshot and compiled plan")
        result = await planning_kernel.apply_soft_repair(
            state["trip"], snapshot, plan, thread_id=state["thread_id"]
        )
        candidate = result.repaired_candidate
        valid = bool(candidate.validation and candidate.validation.valid)
        record_agent_event(
            TraceEventType.VALIDATION_COMPLETED,
            status="validated" if valid else "invalid",
            operation="planning_kernel.soft_revalidate",
            attributes={
                "candidate_id": candidate.id,
                "soft_repair_round": plan.round,
            },
        )
        return {
            "kernel_snapshot": result.snapshot,
            "candidates": list(result.snapshot.candidates),
            "selected_plan": candidate if valid else None,
            "route_results": result.snapshot.route_results,
            "soft_repaired_candidate_id": candidate.id,
            "soft_repair_reused_routes": result.reused_route_count,
            "soft_repair_loaded_routes": result.loaded_route_count,
            "critic_status": CriticStatus.NOT_RUN,
            "critic_execution_summary": None,
            "critic_evidence_digests": [],
            "soft_critiques": [],
            "critic_grounding_attempts": 0,
            "critic_grounding_errors": [],
            "quality_scores": {},
            "phase": AgentPhase.SOLVED if valid else AgentPhase.INVALID,
        }

    def route_after_soft_repair(state: AgenticTravelState) -> str:
        return (
            "review_quality"
            if state["selected_plan"] is not None
            else "restore_soft_baseline"
        )

    async def _finish_soft_repair(
        state: AgenticTravelState,
        *,
        accepted: bool,
        hard_validation_passed: bool,
        after_score: float | None,
        reason: str,
    ) -> dict:
        baseline = state["soft_baseline_snapshot"]
        plan = state["soft_repair_plan"]
        if baseline is None or plan is None:
            raise RuntimeError("soft repair comparison requires baseline state")
        before_score = state["soft_baseline_quality_scores"][plan.target_candidate_id]
        attempt = SoftRepairAttempt(
            round=plan.round,
            before_quality_score=before_score,
            after_quality_score=after_score,
            hard_validation_passed=hard_validation_passed,
            accepted=accepted,
            reused_route_count=state["soft_repair_reused_routes"],
            loaded_route_count=state["soft_repair_loaded_routes"],
            terminal_reason=reason,
        )
        if accepted:
            snapshot = state["kernel_snapshot"]
            assert snapshot is not None
            repaired_id = state["soft_repaired_candidate_id"]
            digests = [
                item
                for item in state["soft_baseline_evidence_digests"]
                if item.candidate_id != plan.target_candidate_id
            ] + list(state["critic_evidence_digests"])
            critiques = [
                item
                for item in state["soft_baseline_critiques"]
                if item.candidate_id != plan.target_candidate_id
            ] + list(state["soft_critiques"])
            scores = {
                key: value
                for key, value in state["soft_baseline_quality_scores"].items()
                if key != plan.target_candidate_id
            }
            scores.update(state["quality_scores"])
            summary = state["critic_execution_summary"]
            status = state["critic_status"]
            assert summary is not None and repaired_id is not None
            review = QualityReviewResult(
                status=status,
                evidence_digests=tuple(digests),
                critiques=tuple(critiques),
                execution_summary=summary,
                quality_scores=scores,
                grounding_errors=tuple(state["critic_grounding_errors"]),
                grounding_attempt=state["critic_grounding_attempts"],
            )
        else:
            snapshot = baseline
            summary = state["soft_baseline_critic_execution_summary"]
            status = state["soft_baseline_critic_status"]
            if summary is None:
                raise RuntimeError("soft baseline critic summary is unavailable")
            review = QualityReviewResult(
                status=status,
                evidence_digests=tuple(state["soft_baseline_evidence_digests"]),
                critiques=tuple(state["soft_baseline_critiques"]),
                execution_summary=summary,
                quality_scores=dict(state["soft_baseline_quality_scores"]),
                grounding_errors=(),
                grounding_attempt=summary.grounding_attempt_count,
            )
        selected = quality_service.select(snapshot, review)
        explanation = quality_service.explain(selected, review)
        records, evidence = await _quality_evidence(
            state,
            candidate_id=selected.id,
            status=review.status,
            score=review.quality_scores.get(selected.id),
        )
        observation = AgentObservation(
            kind=ObservationKind.REPAIR_RESULT,
            action_id=(state["last_action"].action_id if state["last_action"] else None),
            summary=(
                f"软质量改进{'已接受' if accepted else '已回退'}，reason={reason}"
            ),
            evidence_ids=(evidence.evidence_id,),
        )
        _trace_observation(observation)
        return {
            "kernel_snapshot": snapshot,
            "candidates": list(snapshot.candidates),
            "selected_plan": selected,
            "route_results": snapshot.route_results,
            "critic_evidence_digests": list(review.evidence_digests),
            "soft_critiques": list(review.critiques),
            "critic_execution_summary": review.execution_summary,
            "critic_status": review.status,
            "critic_grounding_attempts": review.grounding_attempt,
            "critic_grounding_errors": list(review.grounding_errors),
            "quality_scores": dict(review.quality_scores),
            "grounded_explanation": explanation,
            "quality_review_complete": True,
            "soft_iterations": plan.round,
            "soft_repair_history": [*state["soft_repair_history"], attempt],
            "soft_repair_plan": None,
            "soft_baseline_snapshot": None,
            "soft_baseline_evidence_digests": [],
            "soft_baseline_critiques": [],
            "soft_baseline_quality_scores": {},
            "soft_baseline_critic_execution_summary": None,
            "soft_repaired_candidate_id": None,
            "phase": AgentPhase.VALIDATED,
            "evidence_records": records,
            "recent_observations": _append_observation(state, observation),
        }

    async def restore_soft_baseline(state: AgenticTravelState) -> dict:
        return await _finish_soft_repair(
            state,
            accepted=False,
            hard_validation_passed=False,
            after_score=None,
            reason="hard_validation_failed",
        )

    async def compare_soft_repair(state: AgenticTravelState) -> dict:
        plan = state["soft_repair_plan"]
        if plan is None:
            raise RuntimeError("soft repair comparison requires a plan")
        before = state["soft_baseline_quality_scores"][plan.target_candidate_id]
        repaired_id = state["soft_repaired_candidate_id"]
        after = state["quality_scores"].get(repaired_id or "")
        accepted = (
            state["critic_status"] is CriticStatus.SUCCESS
            and after is not None
            and after - before >= quality_service.policy.min_improvement
        )
        reason = (
            "quality_improved"
            if accepted
            else (
                "critic_unavailable_after_repair"
                if state["critic_status"] is not CriticStatus.SUCCESS
                else "quality_improvement_below_threshold"
            )
        )
        return await _finish_soft_repair(
            state,
            accepted=accepted,
            hard_validation_passed=True,
            after_score=after,
            reason=reason,
        )

    def request_clarification(state: AgenticTravelState) -> dict:
        action = state["last_action"]
        assert isinstance(action, (ClarifyAction, EscalateAction))
        fields = list(action.fields) if isinstance(action, ClarifyAction) else []
        return {
            "asked_clarification_fields": [*state["asked_clarification_fields"], *fields],
            "phase": AgentPhase.TERMINAL,
            "status": "needs_clarification",
            "terminal_reason": "needs_clarification",
            "message": action.question,
        }

    def mark_terminal(state: AgenticTravelState) -> dict:
        return {}

    def route_after_repair(state: AgenticTravelState) -> str:
        if state["phase"] is AgentPhase.TERMINAL:
            return "mark_terminal"
        if state["phase"] is AgentPhase.SOLVED:
            return "review_quality"
        return "derive_evidence_gaps"

    def mark_no_progress(state: AgenticTravelState) -> dict:
        record_agent_event(
            TraceEventType.AGENT_NO_PROGRESS,
            status="failed",
            operation="agent.no_progress",
            attributes={
                "reason_code": "invalid_action_limit",
                "action_fingerprint": state["last_action_fingerprint"],
            },
        )
        return {
            "phase": AgentPhase.TERMINAL,
            "status": "failed",
            "terminal_reason": "agent_no_progress",
            "message": "动态规划连续产生无效或重复动作，已在预算内终止。",
        }

    builder = StateGraph(AgenticTravelState)
    builder.add_node("execution_budget_guard", instrument_node("agentic", "execution_budget_guard", execution_budget_guard))
    builder.add_node("derive_evidence_gaps", instrument_node("agentic", "derive_evidence_gaps", derive_gaps))
    builder.add_node("compose_planner_context", instrument_node("agentic", "compose_planner_context", compose_context))
    builder.add_node("planner_decide", instrument_node("agentic", "planner_decide", decide))
    builder.add_node("validate_action", instrument_node("agentic", "validate_action", guard_action))
    builder.add_node("execute_tool_action", instrument_node("agentic", "execute_tool_action", execute_tool))
    builder.add_node("solve_plan", instrument_node("agentic", "solve_plan", solve_plan))
    builder.add_node("final_guard", instrument_node("agentic", "final_guard", guard_finish))
    builder.add_node("apply_repair", instrument_node("agentic", "apply_repair", apply_repair))
    builder.add_node("review_quality", instrument_node("agentic", "review_quality", review_quality))
    builder.add_node("quality_gate", instrument_node("agentic", "quality_gate", quality_gate))
    builder.add_node("apply_soft_repair", instrument_node("agentic", "apply_soft_repair", apply_soft_repair))
    builder.add_node("restore_soft_baseline", instrument_node("agentic", "restore_soft_baseline", restore_soft_baseline))
    builder.add_node("compare_soft_repair", instrument_node("agentic", "compare_soft_repair", compare_soft_repair))
    builder.add_node("request_clarification", instrument_node("agentic", "request_clarification", request_clarification))
    builder.add_node("mark_terminal", instrument_node("agentic", "mark_terminal", mark_terminal, terminal=True))
    builder.add_node("mark_no_progress", instrument_node("agentic", "mark_no_progress", mark_no_progress, terminal=True))
    builder.add_edge(START, "execution_budget_guard")
    builder.add_edge("execution_budget_guard", "derive_evidence_gaps")
    builder.add_edge("derive_evidence_gaps", "compose_planner_context")
    builder.add_edge("compose_planner_context", "planner_decide")
    builder.add_edge("planner_decide", "validate_action")
    builder.add_conditional_edges("validate_action", instrument_route("agentic", "dispatch_action", route_action))
    builder.add_edge("execute_tool_action", "derive_evidence_gaps")
    builder.add_conditional_edges(
        "solve_plan",
        instrument_route("agentic", "solve_plan", route_after_solve),
    )
    builder.add_conditional_edges("final_guard", instrument_route("agentic", "final_guard", route_after_final_guard))
    builder.add_conditional_edges(
        "apply_repair",
        instrument_route("agentic", "apply_repair", route_after_repair),
    )
    builder.add_conditional_edges(
        "review_quality",
        instrument_route("agentic", "review_quality", route_after_quality_review),
    )
    builder.add_conditional_edges(
        "quality_gate",
        instrument_route("agentic", "quality_gate", route_after_quality_gate),
    )
    builder.add_conditional_edges(
        "apply_soft_repair",
        instrument_route("agentic", "apply_soft_repair", route_after_soft_repair),
    )
    builder.add_edge("restore_soft_baseline", "derive_evidence_gaps")
    builder.add_edge("compare_soft_repair", "derive_evidence_gaps")
    builder.add_edge("request_clarification", "mark_terminal")
    builder.add_edge("mark_no_progress", "mark_terminal")
    builder.add_edge("mark_terminal", END)
    return builder.compile(checkpointer=InMemorySaver())


def initial_agentic_state(
    request: PlanningRequest, thread_id: str, *, run_id: str | None = None
) -> AgenticTravelState:
    return {
        "thread_id": thread_id,
        "run_id": run_id or current_run_id() or thread_id,
        "request": request,
        "trip": request.trip,
        "phase": AgentPhase.RESEARCHING,
        "evidence_gaps": [],
        "evidence_records": [],
        "recent_observations": [],
        "planner_context": None,
        "last_decision": None,
        "last_action": None,
        "last_action_allowed": False,
        "last_action_fingerprint": None,
        "action_history": [],
        "decision_round": 0,
        "invalid_action_count": 0,
        "repair_proposal_fingerprints": [],
        "asked_clarification_fields": [],
        "poi_facts": [],
        "route_results": {},
        "candidates": [],
        "selected_plan": None,
        "kernel_snapshot": None,
        "iterations": 0,
        "critic_evidence_digests": [],
        "soft_critiques": [],
        "critic_execution_summary": None,
        "critic_status": CriticStatus.NOT_RUN,
        "critic_grounding_attempts": 0,
        "critic_grounding_errors": [],
        "quality_scores": {},
        "quality_review_complete": False,
        "grounded_explanation": None,
        "soft_repair_plan": None,
        "soft_repair_history": [],
        "soft_iterations": 0,
        "soft_baseline_snapshot": None,
        "soft_baseline_evidence_digests": [],
        "soft_baseline_critiques": [],
        "soft_baseline_quality_scores": {},
        "soft_baseline_critic_execution_summary": None,
        "soft_baseline_critic_status": CriticStatus.NOT_RUN,
        "soft_repaired_candidate_id": None,
        "soft_repair_reused_routes": 0,
        "soft_repair_loaded_routes": 0,
        "agent_mode": "dynamic_planner",
        "terminal_reason": None,
        "degraded_reasons": [],
        "status": "running",
        "message": None,
    }


async def run_agentic_planning(
    workflow: CompiledStateGraph,
    request: PlanningRequest,
    *,
    thread_id: str | None = None,
) -> PlanningResponse:
    resolved_thread_id = thread_id or str(uuid4())
    result = cast(
        AgenticTravelState,
        await workflow.ainvoke(
            initial_agentic_state(request, resolved_thread_id),
            config={"configurable": {"thread_id": resolved_thread_id}, "recursion_limit": 80},
        ),
    )
    snapshot = result["kernel_snapshot"]
    return PlanningResponse(
        status=result["status"],
        terminal_reason=result["terminal_reason"],
        selected_plan=result["selected_plan"],
        candidates=result["candidates"],
        iterations=result["iterations"],
        message=result["message"],
        critic_status=result["critic_status"],
        critic_summary=result["critic_execution_summary"],
        candidate_critiques=(
            result["soft_critiques"]
            if result["critic_status"] is CriticStatus.SUCCESS
            else []
        ),
        grounded_explanation=result["grounded_explanation"],
        soft_iterations=result["soft_iterations"],
        stay_resolution=snapshot.stay_resolution if snapshot else None,
        agent_mode="dynamic_planner",
        decision_count=result["decision_round"],
        action_summary=tuple(item.kind for item in result["action_history"]),
        evidence_summary={
            kind.value: sum(item.kind is kind for item in result["evidence_records"])
            for kind in EvidenceKind
            if any(item.kind is kind for item in result["evidence_records"])
        },
        degraded_reasons=tuple(result["degraded_reasons"]),
    )


def _append_observation(
    state: AgenticTravelState, observation: AgentObservation
) -> list[AgentObservation]:
    return [*state["recent_observations"], observation][-8:]


def _trace_observation(observation: AgentObservation, *, file_details: dict[str, object] | None = None) -> None:
    record_agent_event(
        TraceEventType.AGENT_OBSERVATION_RECORDED,
        status="recorded",
        operation="agent.record_observation",
        attributes={
            "reason_code": observation.error_code,
            "evidence_count": len(observation.evidence_ids),
        },
        file_details=file_details,
    )


def _violation_fingerprint(candidate_id: str, violation: object) -> str:
    value = "|".join(
        (
            candidate_id,
            str(getattr(violation, "type", "unknown")),
            str(getattr(violation, "day", "")),
            ",".join(getattr(violation, "entity_ids", ()) or ()),
        )
    )
    return sha256(value.encode("utf-8")).hexdigest()
