from __future__ import annotations

import logging
from time import perf_counter
from typing import Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from travel_agent.agents.context import CriticContext, PlannerContext, ReplannerContext
from travel_agent.agents.contracts import HandoffReason
from travel_agent.agents.orchestrator import SpecialistExecutor
from travel_agent.config import AgentMode
from travel_agent.domain.models import (
    DayBoundary,
    PlanCandidate,
    PlanningRequest,
    PlanningResponse,
    POIResolutionIssue,
    StayAnchorResolution,
    TripSpec,
    ValidationStatus,
)
from travel_agent.critique.evidence import EvidenceBudget, build_evidence_digests
from travel_agent.critique.errors import CriticUnavailableError
from travel_agent.critique.explanation import build_grounded_explanation
from travel_agent.critique.gateway import CriticGateway
from travel_agent.critique.grounding import validate_critique_grounding
from travel_agent.critique.quality import (
    CriticPolicy,
    quality_scores as calculate_quality_scores,
    select_by_quality as choose_by_quality,
)
from travel_agent.domain.critique_models import (
    CriticExecutionSummary,
    CriticStatus,
    SoftCriticRequest,
    SoftRepairAttempt,
)
from travel_agent.domain.optimization_models import (
    OptimizationBudget,
    OptimizationProblem,
    OptimizationResult,
    OptimizationSolveStatus,
)
from travel_agent.domain.repair_models import RepairAttempt, RepairOutcome
from travel_agent.domain.tool_models import (
    POIFacts,
    RouteQuery,
    ToolCallContext,
    ToolExecutionSummary,
    ToolResult,
    ToolStatus,
)
from travel_agent.graph.state import TravelState
from travel_agent.graph.evaluation import PlanningEvaluationOverrides
from travel_agent.planning.defaults import POIDefaultPolicy
from travel_agent.planning.drafts import (
    CandidateDraft,
    prepare_candidate_drafts,
)
from travel_agent.execution.checkpoints import ObservedCheckpointSaver
from travel_agent.execution.instrumentation import (
    execution_budget_guard,
    instrument_node,
    instrument_route,
)
from travel_agent.execution.context import (
    record_day_boundaries,
    record_degradation,
    record_stay_resolution,
)
from travel_agent.execution.errors import ExecutionBudgetExceeded
from travel_agent.planning.critic import (
    analyze_candidate,
    error_violations,
    select_repair_target as choose_repair_target,
    violation_fingerprint,
)
from travel_agent.planning.impact import collect_route_delta, day_fingerprint
from travel_agent.planning.planner import materialize_candidates
from travel_agent.planning.optimization import (
    ORToolsOptimizationSolver,
    OptimizationSolver,
    OptimizationTimeoutError,
    build_optimization_problem as create_optimization_problem,
    collect_route_matrix_queries,
    degraded_result,
    drafts_from_optimization,
    select_optimization_pois,
)
from travel_agent.planning.policy import PlanningPolicy
from travel_agent.planning.repair import apply_repair_plan, build_repair_plan
from travel_agent.planning.soft_repair import (
    apply_soft_repair as apply_soft_repair_to_draft,
    compile_soft_repair_plan as create_soft_repair_plan,
)
from travel_agent.planning.search_plan import build_search_plan as create_search_plan
from travel_agent.planning.search_plan import select_search_candidates
from travel_agent.planning.stay import derive_day_boundaries, resolve_stay_anchor
from travel_agent.planning.validator import validate_candidate
from travel_agent.tools.errors import ToolUnavailableError
from travel_agent.tools.gateway import ToolGateway
from travel_agent.memory.models import AgentRole


logger = logging.getLogger(__name__)
_DELIVERABLE_STATUSES = {
    ValidationStatus.VALID,
    ValidationStatus.VALID_WITH_WARNINGS,
}
_CHECKPOINT_ALLOWED_TYPES = (
    ("travel_agent.domain.models", "DayBoundary"),
    ("travel_agent.domain.models", "ItemType"),
    ("travel_agent.domain.models", "Pace"),
    ("travel_agent.domain.models", "PlanCandidate"),
    ("travel_agent.domain.models", "PlanRouteLeg"),
    ("travel_agent.domain.models", "PlanStyle"),
    ("travel_agent.domain.models", "PlanningPOI"),
    ("travel_agent.domain.models", "StayAnchorMode"),
    ("travel_agent.domain.models", "StayAnchorResolution"),
    ("travel_agent.domain.models", "POIResolutionIssue"),
    ("travel_agent.domain.models", "TripSpec"),
    ("travel_agent.domain.models", "ValidationStatus"),
    ("travel_agent.domain.models", "ViolationSeverity"),
    ("travel_agent.domain.critique_models", "CandidateEvidenceDigest"),
    ("travel_agent.domain.critique_models", "CriticExecutionSummary"),
    ("travel_agent.domain.critique_models", "CriticStatus"),
    ("travel_agent.domain.critique_models", "DimensionCritique"),
    ("travel_agent.domain.critique_models", "EvidenceItem"),
    ("travel_agent.domain.critique_models", "EvidenceKind"),
    ("travel_agent.domain.critique_models", "GroundedExplanation"),
    ("travel_agent.domain.critique_models", "GroundedStatement"),
    ("travel_agent.domain.critique_models", "SoftCritique"),
    ("travel_agent.domain.critique_models", "SoftDimension"),
    ("travel_agent.domain.critique_models", "SoftRepairAttempt"),
    ("travel_agent.domain.critique_models", "SoftRepairPlan"),
    ("travel_agent.domain.critique_models", "SuggestedActionKind"),
    ("travel_agent.domain.critique_models", "SuggestedSoftAction"),
    ("travel_agent.domain.optimization_models", "ObjectiveBreakdown"),
    ("travel_agent.domain.optimization_models", "ObjectiveWeights"),
    ("travel_agent.domain.optimization_models", "OptimizationBudget"),
    ("travel_agent.domain.optimization_models", "OptimizationDayAssignment"),
    ("travel_agent.domain.optimization_models", "OptimizationPOI"),
    ("travel_agent.domain.optimization_models", "OptimizationProblem"),
    ("travel_agent.domain.optimization_models", "OptimizationResult"),
    ("travel_agent.domain.optimization_models", "OptimizationSolution"),
    ("travel_agent.domain.optimization_models", "OptimizationSolveStatus"),
    ("travel_agent.domain.optimization_models", "RouteMatrixEntry"),
    ("travel_agent.domain.repair_models", "CriticReport"),
    ("travel_agent.domain.repair_models", "RepairAction"),
    ("travel_agent.domain.repair_models", "RepairActionKind"),
    ("travel_agent.domain.repair_models", "RepairAttempt"),
    ("travel_agent.domain.repair_models", "RepairOutcome"),
    ("travel_agent.domain.repair_models", "RepairPlan"),
    ("travel_agent.domain.tool_models", "POIFacts"),
    ("travel_agent.domain.tool_models", "POISearchQuery"),
    ("travel_agent.domain.tool_models", "RouteMode"),
    ("travel_agent.domain.tool_models", "RouteQuery"),
    ("travel_agent.domain.tool_models", "RouteResult"),
    ("travel_agent.domain.tool_models", "ToolExecutionSummary"),
    ("travel_agent.domain.tool_models", "ToolStatus"),
    ("travel_agent.domain.tool_models", "ValueSource"),
    ("travel_agent.planning.drafts", "CandidateDraft"),
)


def _planning_round(state: TravelState) -> int:
    pending_round = state["pending_replan_round"]
    return pending_round if pending_round is not None else state["iterations"]


def _log_node_started(state: TravelState, node: str) -> None:
    logger.info(
        "node.started | thread_id=%s node=%s iteration=%s status=%s",
        state["thread_id"],
        node,
        state["iterations"],
        state["status"],
    )


def _log_node_completed(
    state: TravelState,
    node: str,
    next_status: str,
    **details: object,
) -> None:
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    logger.info(
        "node.completed | thread_id=%s node=%s iteration=%s status=%s%s",
        state["thread_id"],
        node,
        state["iterations"],
        next_status,
        f" {detail_text}" if detail_text else "",
    )


def _log_candidates(state: TravelState, candidates: list[PlanCandidate]) -> None:
    phase = "initial" if _planning_round(state) == 0 else "replan"
    for candidate in candidates:
        activity_count = sum(len(day.items) for day in candidate.days)
        logger.info(
            "candidate.generated | thread_id=%s phase=%s candidate_id=%s "
            "style=%s activities=%s cost=%s travel_minutes=%s score=%s",
            state["thread_id"],
            phase,
            candidate.id,
            candidate.style.value,
            activity_count,
            candidate.metrics.estimated_cost,
            candidate.metrics.total_travel_minutes,
            candidate.score,
        )
        for day in candidate.days:
            poi_names = ",".join(item.name for item in day.items) or "none"
            timeline = ",".join(
                f"{item.name}[{item.start_at:%H:%M}-{item.end_at:%H:%M}]"
                for item in day.items
            ) or "none"
            logger.debug(
                "candidate.schedule | thread_id=%s candidate_id=%s style=%s "
                "day=%s poi_names=%s timeline=%s cost=%s travel_minutes=%s "
                "walking_meters=%s fatigue=%s",
                state["thread_id"],
                candidate.id,
                candidate.style.value,
                day.date,
                poi_names,
                timeline,
                day.estimated_cost,
                day.total_travel_minutes,
                day.walking_distance_meters,
                day.fatigue_score,
            )


def _tool_summary(
    result: ToolResult[object],
    operation: str,
) -> ToolExecutionSummary:
    return ToolExecutionSummary(
        provider=result.provider,
        operation=operation,
        status=result.status,
        cache_hit=result.cache_hit,
        attempt_count=result.attempt_count,
    )


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _is_required(facts: POIFacts, trip: TripSpec) -> bool:
    poi_name = _normalize(facts.name)
    return any(
        _normalize(required) in poi_name or poi_name in _normalize(required)
        for required in trip.must_visit
    )


def _deliverable_candidates(state: TravelState) -> list[PlanCandidate]:
    return [
        candidate
        for candidate in state["candidates"]
        if candidate.validation is not None
        and candidate.validation.status in _DELIVERABLE_STATUSES
    ]


def select_best_candidate(candidates: list[PlanCandidate]) -> PlanCandidate:
    """优先选择完全合法候选，再在同一验证等级内比较分数。"""
    return min(
        candidates,
        key=lambda candidate: (
            0
            if candidate.validation is not None
            and candidate.validation.status is ValidationStatus.VALID
            else 1,
            -(
                candidate.score
                if candidate.score is not None
                else float("-inf")
            ),
            candidate.id,
        ),
    )


def route_after_validation(
    state: TravelState,
) -> Literal[
    "prepare_critic_context",
    "select_best",
    "select_repair_target",
    "mark_infeasible",
    "restore_soft_baseline",
]:
    deliverable = _deliverable_candidates(state)
    if state["pending_soft_replan_round"] is not None and not deliverable:
        next_node = "restore_soft_baseline"
    elif deliverable:
        next_node = "prepare_critic_context"
    elif (
        state["repair_terminal_reason"] is None
        and state["planning_pois"]
        and state["iterations"] < state["max_replan_rounds"]
    ):
        next_node = "select_repair_target"
    else:
        next_node = "mark_infeasible"
    logger.info(
        "routing.decision | thread_id=%s after=validate_candidates next=%s "
        "iteration=%s deliverable_candidates=%s max_replan_rounds=%s",
        state["thread_id"],
        next_node,
        state["iterations"],
        len(deliverable),
        state["max_replan_rounds"],
    )
    return next_node


def _route_after_validation_without_soft_critic(
    state: TravelState,
) -> Literal[
    "select_best",
    "select_repair_target",
    "mark_infeasible",
    "restore_soft_baseline",
]:
    """评测消融仍保留 Hard Validator，只移除 Soft Critic 路径。"""
    deliverable = _deliverable_candidates(state)
    if state["pending_soft_replan_round"] is not None and not deliverable:
        return "restore_soft_baseline"
    if deliverable:
        return "select_best"
    if (
        state["repair_terminal_reason"] is None
        and state["planning_pois"]
        and state["iterations"] < state["max_replan_rounds"]
    ):
        return "select_repair_target"
    return "mark_infeasible"


def route_after_critic(
    state: TravelState,
) -> Literal["build_repair_plan", "mark_infeasible"]:
    report = state["critic_report"]
    return (
        "build_repair_plan"
        if report is not None and report.repairable
        else "mark_infeasible"
    )


def route_after_repair_plan(
    state: TravelState,
) -> Literal["apply_local_repair", "mark_infeasible"]:
    return (
        "apply_local_repair"
        if state["repair_plan"] is not None
        and state["repair_terminal_reason"] is None
        else "mark_infeasible"
    )


def route_after_delta_routes(
    state: TravelState,
) -> Literal["load_delta_routes", "materialize_candidates"]:
    return (
        "load_delta_routes"
        if state["delta_route_queries"]
        else "materialize_candidates"
    )


def route_after_grounding(
    state: TravelState,
) -> Literal["soft_constraint_critic", "quality_gate"]:
    if (
        state["critic_grounding_errors"]
        and state["critic_grounding_attempts"]
        < state["critic_grounding_max_attempts"]
        and state["critic_status"] is CriticStatus.INVALID_GROUNDING
    ):
        return "soft_constraint_critic"
    return "quality_gate"


def route_after_quality_gate(
    state: TravelState,
) -> Literal["compare_soft_repair", "compile_soft_repair_plan", "select_by_quality"]:
    if state["pending_soft_replan_round"] is not None:
        return "compare_soft_repair"
    if state["critic_status"] is CriticStatus.SUCCESS and state["quality_scores"]:
        return "compile_soft_repair_plan"
    return "select_by_quality"


def route_after_soft_repair_plan(
    state: TravelState,
) -> Literal["apply_soft_repair", "select_by_quality"]:
    return "apply_soft_repair" if state["soft_repair_plan"] is not None else "select_by_quality"


def route_after_soft_delta_routes(
    state: TravelState,
) -> Literal["load_soft_delta_routes", "materialize_soft_candidate"]:
    return "load_soft_delta_routes" if state["delta_route_queries"] else "materialize_soft_candidate"


def build_workflow(
    gateway: ToolGateway,
    defaults: POIDefaultPolicy,
    policy: PlanningPolicy = PlanningPolicy(),
    *,
    optimizer: OptimizationSolver | None = None,
    optimization_budget: OptimizationBudget = OptimizationBudget(),
    critic_gateway: CriticGateway | None = None,
    critic_policy: CriticPolicy = CriticPolicy(),
    evidence_budget: EvidenceBudget = EvidenceBudget(),
    evaluation_overrides: PlanningEvaluationOverrides | None = None,
    agent_mode: AgentMode = AgentMode.SINGLE_GRAPH,
    specialist_executor: SpecialistExecutor | None = None,
) -> CompiledStateGraph:
    """构建只通过注入 ToolGateway 获取外部事实的异步规划图。"""

    eval_overrides = evaluation_overrides or PlanningEvaluationOverrides()
    use_specialists = agent_mode in {
        AgentMode.SPECIALIST_SUBAGENTS,
        AgentMode.SHADOW_SUBAGENTS,
    }
    specialists = (
        specialist_executor or SpecialistExecutor() if use_specialists else None
    )

    async def build_search_plan(state: TravelState) -> dict:
        _log_node_started(state, "build_search_plan")
        context = PlannerContext(
            trip=state["trip"],
            poi_query_limit=policy.poi_query_limit,
            max_queries=policy.poi_max_queries,
        )
        if use_specialists:
            assert specialists is not None
            queries, _ = await specialists.invoke(
                role=AgentRole.PLANNER,
                reason=HandoffReason.PLAN,
                context=context,
                expected_output_schema="list[POISearchQuery]",
                operation=lambda: create_search_plan(
                    context.trip,
                    per_query_limit=context.poi_query_limit,
                    max_queries=context.max_queries,
                ),
            )
        else:
            queries = create_search_plan(
                context.trip,
                per_query_limit=context.poi_query_limit,
                max_queries=context.max_queries,
            )
        logger.info(
            "search_plan.created | thread_id=%s query_count=%s priorities=%s",
            state["thread_id"],
            len(queries),
            ",".join(str(query.priority) for query in queries) or "none",
        )
        _log_node_completed(
            state,
            "build_search_plan",
            "search_planned",
            query_count=len(queries),
        )
        return {
            "search_queries": queries,
            "max_soft_replan_rounds": critic_policy.max_soft_replan_rounds,
            "critic_grounding_max_attempts": critic_policy.grounding_max_attempts,
            "status": "search_planned",
        }

    async def load_pois(state: TravelState) -> dict:
        _log_node_started(state, "load_pois")
        context = ToolCallContext(thread_id=state["thread_id"])
        required = [query for query in state["search_queries"] if query.exact_match]
        ordinary = [query for query in state["search_queries"] if not query.exact_match]
        results = []
        for query in required:
            result = (await gateway.search_pois([query], context))[0]
            if result.status is ToolStatus.FAILED:
                raise ToolUnavailableError.from_result(result, state["thread_id"])
            results.append(result)
        results.extend(await gateway.search_pois(ordinary, context))
        for result in results:
            if result.status is ToolStatus.FAILED:
                raise ToolUnavailableError.from_result(result, state["thread_id"])

        poi_facts = select_search_candidates([*required, *ordinary], results, max_candidates=policy.poi_candidate_limit)
        summaries = [
            _tool_summary(result, "poi.search")
            for result in results
        ]
        logger.info(
            "poi_facts.loaded | thread_id=%s fact_count=%s",
            state["thread_id"],
            len(poi_facts),
        )
        _log_node_completed(
            state,
            "load_pois",
            "poi_facts_loaded",
            fact_count=len(poi_facts),
        )
        return {
            "poi_facts": poi_facts,
            "tool_summaries": [*state["tool_summaries"], *summaries],
            "status": "poi_facts_loaded",
        }

    def resolve_poi_facts(state: TravelState) -> dict:
        _log_node_started(state, "resolve_poi_facts")
        planning_pois = []
        issues = []
        for facts in state["poi_facts"]:
            resolution = defaults.resolve(facts, state["trip"])
            if resolution.poi is not None:
                planning_pois.append(resolution.poi)
            else:
                issues.append(
                    POIResolutionIssue(
                        poi_id=facts.id,
                        poi_name=facts.name,
                        missing_fields=resolution.missing_fields,
                        required=_is_required(facts, state["trip"]),
                    )
                )
        logger.info(
            "poi_context.loaded | thread_id=%s fact_count=%s planning_poi_count=%s "
            "resolution_issue_count=%s",
            state["thread_id"],
            len(state["poi_facts"]),
            len(planning_pois),
            len(issues),
        )
        _log_node_completed(
            state,
            "resolve_poi_facts",
            "poi_context_loaded",
            poi_count=len(planning_pois),
        )
        return {
            "planning_pois": planning_pois,
            "poi_resolution_issues": issues,
            "status": "poi_context_loaded",
        }

    def resolve_stay_anchor_node(state: TravelState) -> dict:
        _log_node_started(state, "resolve_stay_anchor")
        resolution = resolve_stay_anchor(state["trip"], state["planning_pois"])
        record_stay_resolution(
            mode=resolution.mode.value,
            confidence=resolution.confidence,
            confirmed=resolution.confirmed,
            candidate_count=len(resolution.reference_poi_ids),
        )
        logger.info(
            "stay_anchor.resolved | thread_id=%s mode=%s confidence=%s "
            "confirmed=%s candidate_count=%s",
            state["thread_id"], resolution.mode.value, resolution.confidence,
            resolution.confirmed, len(resolution.reference_poi_ids),
        )
        _log_node_completed(
            state, "resolve_stay_anchor", "stay_anchor_resolved",
            mode=resolution.mode.value,
        )
        return {"stay_resolution": resolution, "status": "stay_anchor_resolved"}

    def derive_day_boundaries_node(state: TravelState) -> dict:
        _log_node_started(state, "derive_day_boundaries")
        resolution = state["stay_resolution"]
        if resolution is None:
            raise RuntimeError("day boundaries require stay resolution")
        boundaries = derive_day_boundaries(state["trip"], resolution)
        record_day_boundaries(
            day_count=len(boundaries),
            start_role=boundaries[0].start_role,
            end_role=boundaries[-1].end_role,
        )
        logger.info(
            "day_boundaries.resolved | thread_id=%s day_count=%s "
            "start_role=%s end_role=%s",
            state["thread_id"], len(boundaries), boundaries[0].start_role,
            boundaries[-1].end_role,
        )
        _log_node_completed(
            state, "derive_day_boundaries", "day_boundaries_resolved",
            day_count=len(boundaries),
        )
        return {"day_boundaries": boundaries, "status": "day_boundaries_resolved"}

    class _ForcedHeuristicOptimizer:
        name = "evaluation-forced-heuristic"

        def solve(self, _problem):
            raise OptimizationTimeoutError("evaluation optimizer ablation")

    active_optimizer = (
        _ForcedHeuristicOptimizer()
        if eval_overrides.force_heuristic_optimizer
        else optimizer or ORToolsOptimizationSolver()
    )

    async def build_route_matrix(state: TravelState) -> dict:
        _log_node_started(state, "build_route_matrix")
        optimization_pois = select_optimization_pois(
            state["trip"],
            state["planning_pois"],
            optimization_budget,
        )
        queries = collect_route_matrix_queries(
            state["trip"],
            optimization_pois,
            modes=policy.route_modes,
            strategy=policy.route_strategy,
            max_walking_leg_meters=policy.max_walking_leg_meters,
            stay_resolution=state["stay_resolution"],
            day_boundaries=state["day_boundaries"],
        )
        results = await gateway.get_routes(
            queries,
            ToolCallContext(thread_id=state["thread_id"]),
        )
        for result in results.values():
            if result.status is ToolStatus.FAILED:
                raise ToolUnavailableError.from_result(result, state["thread_id"])
        routes = {}
        for key, result in results.items():
            assert result.data is not None
            routes[key] = result.data
        summaries = [
            _tool_summary(
                result,
                f"route.get_{result.data.mode.value}" if result.data else "route.get",
            )
            for result in results.values()
        ]
        cache_hits = sum(result.cache_hit for result in results.values())
        provider_calls = len(results) - cache_hits
        logger.info(
            "route_matrix.loaded | thread_id=%s mode=%s poi_count=%s "
            "query_count=%s route_count=%s cache_hits=%s provider_calls=%s",
            state["thread_id"],
            "+".join(mode.value for mode in policy.route_modes),
            len(optimization_pois),
            len(queries),
            len(routes),
            cache_hits,
            provider_calls,
        )
        _log_node_completed(
            state,
            "build_route_matrix",
            "route_matrix_loaded",
            route_count=len(routes),
        )
        return {
            "optimization_pois": optimization_pois,
            "route_queries": queries,
            "delta_route_queries": [],
            "route_results": routes,
            "reused_route_keys": [],
            "last_route_loaded_count": len(routes),
            "last_route_reused_count": 0,
            "route_matrix_cache_hits": cache_hits,
            "route_matrix_provider_calls": provider_calls,
            "tool_summaries": [*state["tool_summaries"], *summaries],
            "status": "route_matrix_loaded",
        }

    def build_optimization_problem(state: TravelState) -> dict:
        _log_node_started(state, "build_optimization_problem")
        problem = create_optimization_problem(
            state["trip"],
            state["optimization_pois"],
            state["route_results"],
            optimization_budget,
            modes=policy.route_modes,
            strategy=policy.route_strategy,
            max_walking_leg_meters=policy.max_walking_leg_meters,
            stay_resolution=state["stay_resolution"],
            day_boundaries=state["day_boundaries"],
        )
        logger.info(
            "optimization.problem_built | thread_id=%s problem_id=%s "
            "poi_count=%s route_count=%s day_count=%s mode=%s",
            state["thread_id"],
            problem.id,
            len(problem.pois),
            len(problem.route_matrix),
            len(problem.dates),
            "+".join(mode.value for mode in policy.route_modes),
        )
        _log_node_completed(
            state,
            "build_optimization_problem",
            "optimization_problem_built",
            poi_count=len(problem.pois),
        )
        return {
            "optimization_problem": problem,
            "status": "optimization_problem_built",
        }

    def solve_candidate_variants(state: TravelState) -> dict:
        _log_node_started(state, "solve_candidate_variants")
        problem = state["optimization_problem"]
        if problem is None:
            raise RuntimeError("optimization requires a problem")
        logger.info(
            "optimization.started | thread_id=%s problem_id=%s solver=%s "
            "max_solve_ms=%s max_search_states=%s variants=%s",
            state["thread_id"],
            problem.id,
            getattr(active_optimizer, "name", type(active_optimizer).__name__),
            problem.budget.max_solve_ms,
            problem.budget.max_search_states,
            problem.budget.variant_count,
        )
        degraded_reason: str | None = None
        started = perf_counter()
        try:
            result = active_optimizer.solve(problem)
            if not result.solutions:
                degraded_reason = "optimizer_infeasible"
        except OptimizationTimeoutError:
            result = None
            degraded_reason = "optimizer_timeout"
        if degraded_reason is not None:
            fallback_drafts = prepare_candidate_drafts(
                state["trip"],
                state["optimization_pois"],
                replan_round=0,
                day_boundaries=state["day_boundaries"],
            )
            elapsed_ms = round((perf_counter() - started) * 1000, 2)
            result = degraded_result(
                fallback_drafts,
                reason=degraded_reason,
                elapsed_ms=elapsed_ms,
            )
            drafts = fallback_drafts
            logger.warning(
                "optimization.degraded | thread_id=%s problem_id=%s reason=%s "
                "fallback=%s elapsed_ms=%s",
                state["thread_id"],
                problem.id,
                degraded_reason,
                result.solver,
                result.elapsed_ms,
            )
        else:
            assert result is not None
            drafts = drafts_from_optimization(result)
            objective_summary = ",".join(
                f"{solution.style.value}:{solution.objective_value}"
                for solution in result.solutions
            )
            logger.info(
                "optimization.completed | thread_id=%s problem_id=%s status=%s "
                "solver=%s solution_count=%s search_states=%s elapsed_ms=%s "
                "objectives=%s",
                state["thread_id"],
                problem.id,
                result.status.value,
                result.solver,
                len(result.solutions),
                result.search_states,
                result.elapsed_ms,
                objective_summary or "none",
            )
        _log_node_completed(
            state,
            "solve_candidate_variants",
            "optimization_solved",
            solution_count=len(drafts),
            degraded=str(result.status is OptimizationSolveStatus.DEGRADED).lower(),
        )
        return {
            "optimization_result": result,
            "candidate_drafts": drafts,
            "status": "optimization_solved",
        }

    def _materialize(state: TravelState, node_name: str) -> dict:
        _log_node_started(state, node_name)
        candidates = materialize_candidates(
            state["trip"],
            state["candidate_drafts"],
            state["planning_pois"],
            state["route_results"],
            route_strategy=policy.route_strategy,
            route_modes=policy.route_modes,
            max_walking_leg_meters=policy.max_walking_leg_meters,
            stay_resolution=state["stay_resolution"],
            day_boundaries=state["day_boundaries"],
        )
        _log_candidates(state, candidates)
        _log_node_completed(
            state,
            node_name,
            "candidates_materialized",
            candidate_count=len(candidates),
        )
        return {"candidates": candidates, "status": "candidates_materialized"}

    def materialize_optimized(state: TravelState) -> dict:
        return _materialize(state, "materialize_optimized_candidates")

    def materialize_repaired(state: TravelState) -> dict:
        return _materialize(state, "materialize_candidates")

    def validate(state: TravelState) -> dict:
        _log_node_started(state, "validate_candidates")
        validated = []
        for candidate in state["candidates"]:
            result = validate_candidate(
                state["trip"], candidate, state["planning_pois"]
            )
            validated_candidate = candidate.model_copy(
                update={"validation": result}
            )
            validated.append(validated_candidate)
            violation_types = ",".join(
                item.type for item in result.violations
            ) or "none"
            logger.info(
                "candidate.validated | thread_id=%s candidate_id=%s style=%s "
                "status=%s valid=%s violation_count=%s violation_types=%s",
                state["thread_id"],
                candidate.id,
                candidate.style.value,
                result.status.value,
                str(result.valid).lower(),
                len(result.violations),
                violation_types,
            )
            for violation in result.violations:
                logger.debug(
                    "candidate.violation | thread_id=%s candidate_id=%s type=%s "
                    "severity=%s day=%s message=%s repair_hint=%s",
                    state["thread_id"],
                    candidate.id,
                    violation.type,
                    violation.severity.value,
                    violation.day or "none",
                    violation.message,
                    violation.repair_hint or "none",
                )
        deliverable_count = sum(
            candidate.validation is not None
            and candidate.validation.status in _DELIVERABLE_STATUSES
            for candidate in validated
        )
        _log_node_completed(
            state,
            "validate_candidates",
            "validated",
            candidate_count=len(validated),
            deliverable_count=deliverable_count,
        )
        update: dict[str, object] = {
            "candidates": validated,
            "status": "validated",
        }
        pending_soft_round = state["pending_soft_replan_round"]
        if pending_soft_round is not None and deliverable_count:
            if len(validated) != 1:
                raise RuntimeError("soft repair validation requires one candidate")
            repaired_soft = validated[0]
            soft_day_by_date = {
                day.date.isoformat(): day for day in repaired_soft.days
            }
            for day_text, expected_fingerprint in state[
                "preserved_day_hashes"
            ].items():
                day = soft_day_by_date.get(day_text)
                if day is None or day_fingerprint(day) != expected_fingerprint:
                    raise RuntimeError(
                        f"soft repair changed preserved day: {day_text}"
                    )
            logger.info(
                "soft_repair.validation_delta | thread_id=%s round=%s "
                "candidate_id=%s hard_validation_passed=true preserved_days=%s",
                state["thread_id"],
                pending_soft_round,
                repaired_soft.id,
                len(state["preserved_day_hashes"]),
            )
        pending_round = state["pending_replan_round"]
        if pending_round is not None:
            report = state["critic_report"]
            plan = state["repair_plan"]
            if report is None or plan is None or len(validated) != 1:
                raise RuntimeError("repair validation requires one candidate and repair context")
            repaired = validated[0]
            day_by_date = {day.date.isoformat(): day for day in repaired.days}
            for day_text, expected_fingerprint in state[
                "preserved_day_hashes"
            ].items():
                day = day_by_date.get(day_text)
                if day is None or day_fingerprint(day) != expected_fingerprint:
                    raise RuntimeError(
                        f"local repair changed preserved day: {day_text}"
                    )

            after_fingerprint = violation_fingerprint(repaired)
            after_errors = error_violations(repaired)
            after_error_types = {violation.type for violation in after_errors}
            resolved = repaired.validation.status in _DELIVERABLE_STATUSES
            repeated_fingerprint = (
                after_fingerprint == report.violation_fingerprint
                or any(
                    after_fingerprint
                    in {
                        attempt.before_violation_fingerprint,
                        attempt.after_violation_fingerprint,
                    }
                    for attempt in state["repair_history"]
                )
            )
            source_resolved = bool(
                set(plan.source_violation_types) - after_error_types
            )
            improved = (
                len(after_errors) < report.error_count
                or (
                    source_resolved
                    and len(after_errors) <= report.error_count
                )
            )
            terminal_reason: str | None = None
            if resolved:
                outcome = RepairOutcome.RESOLVED
            elif repeated_fingerprint:
                outcome = RepairOutcome.NO_PROGRESS
                terminal_reason = "repeated_violation_fingerprint"
            elif improved:
                outcome = RepairOutcome.IMPROVED
            else:
                outcome = RepairOutcome.NO_PROGRESS
                terminal_reason = "repair_no_progress"

            attempt = RepairAttempt(
                round=pending_round,
                target_candidate_id=plan.target_candidate_id,
                before_violation_fingerprint=report.violation_fingerprint,
                after_violation_fingerprint=after_fingerprint,
                before_error_count=report.error_count,
                after_error_count=len(after_errors),
                action_fingerprint=plan.action_fingerprint,
                action_kinds=tuple(action.kind for action in plan.actions),
                outcome=outcome,
                affected_days=plan.affected_days,
                preserved_day_count=len(plan.preserved_days),
                reused_route_count=state["last_route_reused_count"],
                loaded_route_count=state["last_route_loaded_count"],
                terminal_reason=terminal_reason,
            )
            update.update(
                {
                    "iterations": pending_round,
                    "pending_replan_round": None,
                    "repair_history": [*state["repair_history"], attempt],
                    "repair_terminal_reason": terminal_reason,
                }
            )
            logger.info(
                "repair.validation.delta | thread_id=%s round=%s candidate_id=%s "
                "before_errors=%s after_errors=%s outcome=%s affected_days=%s "
                "preserved_days=%s reused_routes=%s loaded_routes=%s",
                state["thread_id"],
                pending_round,
                repaired.id,
                report.error_count,
                len(after_errors),
                outcome.value,
                len(plan.affected_days),
                len(plan.preserved_days),
                state["last_route_reused_count"],
                state["last_route_loaded_count"],
            )
            logger.info(
                "replan.completed | thread_id=%s round=%s status=validated "
                "candidate_count=%s deliverable_count=%s",
                state["thread_id"],
                pending_round,
                len(validated),
                deliverable_count,
            )
            if resolved or terminal_reason is not None:
                logger.info(
                    "repair.terminated | thread_id=%s round=%s outcome=%s reason=%s",
                    state["thread_id"],
                    pending_round,
                    outcome.value,
                    terminal_reason or "resolved",
                )
        return update

    def select_repair_target(state: TravelState) -> dict:
        _log_node_started(state, "select_repair_target")
        target = choose_repair_target(
            state["candidates"], state["trip"], state["planning_pois"]
        )
        logger.info(
            "repair.target.selected | thread_id=%s candidate_id=%s "
            "candidate_count=%s score=%s",
            state["thread_id"],
            target.id,
            len(state["candidates"]),
            target.score,
        )
        _log_node_completed(
            state,
            "select_repair_target",
            "repair_target_selected",
            candidate_id=target.id,
        )
        return {
            "repair_target_candidate_id": target.id,
            "critic_report": None,
            "repair_plan": None,
            "status": "repair_target_selected",
        }

    def analyze_violations(state: TravelState) -> dict:
        _log_node_started(state, "analyze_violations")
        target_id = state["repair_target_candidate_id"]
        candidate = next(
            candidate
            for candidate in state["candidates"]
            if candidate.id == target_id
        )
        report = analyze_candidate(
            candidate, state["trip"], state["planning_pois"]
        )
        logger.info(
            "critic.completed | thread_id=%s candidate_id=%s errors=%s "
            "warnings=%s violation_types=%s affected_days=%s repairable=%s "
            "terminal_reason=%s",
            state["thread_id"],
            candidate.id,
            report.error_count,
            report.warning_count,
            ",".join(report.violation_types) or "none",
            len(report.affected_days),
            str(report.repairable).lower(),
            report.terminal_reason or "none",
        )
        _log_node_completed(
            state,
            "analyze_violations",
            "violations_analyzed",
            repairable=str(report.repairable).lower(),
        )
        return {
            "critic_report": report,
            "repair_terminal_reason": report.terminal_reason,
            "status": "violations_analyzed",
        }

    async def create_repair_plan(state: TravelState) -> dict:
        _log_node_started(state, "build_repair_plan")
        target_id = state["repair_target_candidate_id"]
        candidate = next(
            candidate
            for candidate in state["candidates"]
            if candidate.id == target_id
        )
        draft = next(
            draft
            for draft in state["candidate_drafts"]
            if draft.id == target_id
        )
        next_round = state["iterations"] + 1
        report = state["critic_report"]
        if report is None:
            raise RuntimeError("repair plan requires critic report")
        context = ReplannerContext(
            trip=state["trip"],
            candidate=candidate,
            draft=draft,
            planning_pois=tuple(state["planning_pois"]),
            critic_report=report,
            repair_round=next_round,
            previous_action_fingerprints=frozenset(
                attempt.action_fingerprint for attempt in state["repair_history"]
            ),
        )
        if use_specialists:
            assert specialists is not None
            repair_result, _ = await specialists.invoke(
                role=AgentRole.REPLANNER,
                reason=HandoffReason.HARD_REPAIR,
                context=context,
                expected_output_schema="tuple[RepairPlan | None, str | None]",
                operation=lambda: build_repair_plan(
                    context.trip,
                    context.candidate,
                    context.draft,
                    list(context.planning_pois),
                    context.critic_report,
                    repair_round=context.repair_round,
                ),
            )
            plan, terminal_reason = repair_result
        else:
            plan, terminal_reason = build_repair_plan(
                context.trip,
                context.candidate,
                context.draft,
                list(context.planning_pois),
                context.critic_report,
                repair_round=context.repair_round,
            )
        if plan is not None and plan.action_fingerprint in context.previous_action_fingerprints:
            terminal_reason = "repeated_repair_action"
            plan = None
        logger.info(
            "repair.plan.created | thread_id=%s round=%s candidate_id=%s "
            "action_count=%s action_types=%s affected_days=%s status=%s reason=%s",
            state["thread_id"],
            next_round,
            candidate.id,
            len(plan.actions) if plan else 0,
            (
                ",".join(action.kind.value for action in plan.actions)
                if plan
                else "none"
            ),
            len(plan.affected_days) if plan else 0,
            "ready" if plan else "unavailable",
            terminal_reason or "none",
        )
        _log_node_completed(
            state,
            "build_repair_plan",
            "repair_plan_ready" if plan else "repair_unavailable",
        )
        return {
            "repair_plan": plan,
            "repair_terminal_reason": terminal_reason,
            "status": "repair_plan_ready" if plan else "repair_unavailable",
        }

    def apply_local_repair(state: TravelState) -> dict:
        _log_node_started(state, "apply_local_repair")
        target_id = state["repair_target_candidate_id"]
        candidate = next(
            candidate
            for candidate in state["candidates"]
            if candidate.id == target_id
        )
        draft = next(
            draft
            for draft in state["candidate_drafts"]
            if draft.id == target_id
        )
        plan = state["repair_plan"]
        if plan is None:
            raise RuntimeError("local repair requires repair plan")
        repaired_draft, applied_plan = apply_repair_plan(
            state["trip"],
            draft,
            state["planning_pois"],
            plan,
            route_strategy=policy.route_strategy,
            route_modes=policy.route_modes,
            max_walking_leg_meters=policy.max_walking_leg_meters,
        )
        preserved_hashes = {
            day.date.isoformat(): day_fingerprint(day)
            for day in candidate.days
            if day.date in applied_plan.preserved_days
        }
        logger.info(
            "repair.action.applied | thread_id=%s round=%s candidate_id=%s "
            "repaired_candidate_id=%s action_count=%s affected_days=%s "
            "preserved_days=%s",
            state["thread_id"],
            applied_plan.round,
            candidate.id,
            repaired_draft.id,
            len(applied_plan.actions),
            len(applied_plan.affected_days),
            len(applied_plan.preserved_days),
        )
        logger.info(
            "repair.routes.invalidated | thread_id=%s round=%s route_count=%s",
            state["thread_id"],
            applied_plan.round,
            len(applied_plan.invalidated_route_keys),
        )
        _log_node_completed(
            state,
            "apply_local_repair",
            "local_repair_applied",
            candidate_id=repaired_draft.id,
        )
        return {
            "pending_replan_round": applied_plan.round,
            "candidate_drafts": [repaired_draft],
            "route_queries": [],
            "delta_route_queries": [],
            "reused_route_keys": [],
            "candidates": [],
            "selected_plan": None,
            "repair_plan": applied_plan,
            "preserved_day_hashes": preserved_hashes,
            "last_route_loaded_count": 0,
            "last_route_reused_count": 0,
            "status": "local_repair_applied",
            "message": (
                f"第 {applied_plan.round} 轮局部修复："
                f"影响 {len(applied_plan.affected_days)} 个日期"
            ),
        }

    def collect_delta_routes(state: TravelState) -> dict:
        _log_node_started(state, "collect_delta_routes")
        draft = state["candidate_drafts"][0]
        reusable_routes = {} if eval_overrides.full_replan else state["route_results"]
        delta = collect_route_delta(
            state["trip"],
            draft,
            state["planning_pois"],
            reusable_routes,
            route_strategy=policy.route_strategy,
            route_modes=policy.route_modes,
            max_walking_leg_meters=policy.max_walking_leg_meters,
        )
        logger.info(
            "repair.routes.reused | thread_id=%s round=%s required=%s "
            "reused=%s missing=%s",
            state["thread_id"],
            state["pending_replan_round"],
            len(delta.required_queries),
            len(delta.reused_route_keys),
            len(delta.missing_queries),
        )
        _log_node_completed(
            state,
            "collect_delta_routes",
            "delta_routes_collected",
            reused=len(delta.reused_route_keys),
            missing=len(delta.missing_queries),
        )
        return {
            "route_queries": list(delta.required_queries),
            "delta_route_queries": list(delta.missing_queries),
            "reused_route_keys": list(delta.reused_route_keys),
            "last_route_reused_count": len(delta.reused_route_keys),
            "last_route_loaded_count": 0,
            "status": "delta_routes_collected",
        }

    async def load_delta_routes(state: TravelState) -> dict:
        _log_node_started(state, "load_delta_routes")
        queries: list[RouteQuery] = state["delta_route_queries"]
        results = await gateway.get_routes(
            queries,
            ToolCallContext(thread_id=state["thread_id"]),
        )
        for result in results.values():
            if result.status is ToolStatus.FAILED:
                raise ToolUnavailableError.from_result(result, state["thread_id"])
        loaded = {}
        for key, result in results.items():
            assert result.data is not None
            loaded[key] = result.data
        summaries = [
            _tool_summary(
                result,
                f"route.get_{result.data.mode.value}" if result.data else "route.get",
            )
            for result in results.values()
        ]
        logger.info(
            "repair.routes.loaded | thread_id=%s round=%s query_count=%s "
            "route_count=%s",
            state["thread_id"],
            state["pending_replan_round"],
            len(queries),
            len(loaded),
        )
        _log_node_completed(
            state,
            "load_delta_routes",
            "delta_routes_loaded",
            route_count=len(loaded),
        )
        return {
            "route_results": {**state["route_results"], **loaded},
            "last_route_loaded_count": len(loaded),
            "tool_summaries": [*state["tool_summaries"], *summaries],
            "status": "delta_routes_loaded",
        }

    def prepare_critic_context(state: TravelState) -> dict:
        _log_node_started(state, "prepare_critic_context")
        deliverable = _deliverable_candidates(state)
        digests = build_evidence_digests(
            state["trip"],
            deliverable,
            state["planning_pois"],
            evidence_budget,
        )
        if not digests:
            raise RuntimeError("soft critic requires at least one evidence digest")
        logger.info(
            "critic.context_prepared | thread_id=%s candidate_count=%s "
            "evidence_count=%s input_chars=%s truncated_count=%s",
            state["thread_id"],
            len(digests),
            sum(len(digest.evidence) for digest in digests),
            sum(digest.input_chars for digest in digests),
            sum(digest.truncated for digest in digests),
        )
        _log_node_completed(
            state,
            "prepare_critic_context",
            "critic_context_prepared",
            candidate_count=len(digests),
        )
        return {
            "critic_evidence_digests": list(digests),
            "soft_critiques": [],
            "critic_grounding_attempts": 0,
            "critic_grounding_errors": [],
            "quality_scores": {},
            "critic_status": CriticStatus.PENDING,
            "status": "critic_context_prepared",
        }

    async def soft_constraint_critic(state: TravelState) -> dict:
        _log_node_started(state, "soft_constraint_critic")
        semantic_attempt = state["critic_grounding_attempts"] + 1
        if critic_gateway is None:
            summary = CriticExecutionSummary(
                provider="disabled",
                model="disabled",
                prompt_version="disabled",
                status=CriticStatus.DISABLED,
                attempt_count=0,
                grounding_attempt_count=0,
                elapsed_ms=0,
                input_chars=sum(
                    digest.input_chars for digest in state["critic_evidence_digests"]
                ),
            )
            logger.info(
                "critic.failed | thread_id=%s provider=disabled model=disabled "
                "category=disabled code=disabled",
                state["thread_id"],
            )
            return {
                "critic_status": CriticStatus.DISABLED,
                "critic_execution_summary": summary,
                "soft_critiques": [],
                "critic_grounding_attempts": 0,
                "status": "critic_disabled",
            }
        request = SoftCriticRequest(
            digests=tuple(state["critic_evidence_digests"]),
            grounding_feedback=tuple(state["critic_grounding_errors"]),
        )
        try:
            context = CriticContext(
                request=request,
                candidate_ids=tuple(
                    digest.candidate_id for digest in state["critic_evidence_digests"]
                ),
            )
            if use_specialists:
                assert specialists is not None
                result, _ = await specialists.invoke(
                    role=AgentRole.CRITIC,
                    reason=HandoffReason.SOFT_CRITIQUE,
                    context=context,
                    expected_output_schema="SoftCriticResult",
                    operation=lambda: critic_gateway.critique(
                        context.request,
                        thread_id=state["thread_id"],
                        grounding_attempt=semantic_attempt,
                    ),
                )
            else:
                result = await critic_gateway.critique(
                    context.request,
                    thread_id=state["thread_id"],
                    grounding_attempt=semantic_attempt,
                )
        except ExecutionBudgetExceeded:
            record_degradation("soft_critic_budget_exhausted")
            summary = CriticExecutionSummary(
                provider=critic_gateway.provider,
                model=critic_gateway.model,
                prompt_version=critic_gateway.prompt_version,
                status=CriticStatus.UNAVAILABLE,
                attempt_count=0,
                grounding_attempt_count=semantic_attempt,
                elapsed_ms=0,
                input_chars=request.input_chars,
                error_category="execution_budget_exhausted",
            )
            return {
                "critic_status": CriticStatus.UNAVAILABLE,
                "critic_execution_summary": summary,
                "soft_critiques": [],
                "critic_grounding_attempts": semantic_attempt,
                "critic_grounding_errors": [],
                "status": "critic_budget_exhausted",
            }
        except CriticUnavailableError as error:
            summary = CriticExecutionSummary(
                provider=error.provider,
                model=error.model,
                prompt_version=critic_gateway.prompt_version,
                status=CriticStatus.UNAVAILABLE,
                attempt_count=error.attempt_count,
                grounding_attempt_count=semantic_attempt,
                elapsed_ms=error.elapsed_ms,
                input_chars=request.input_chars,
                error_category=error.category.value,
            )
            return {
                "critic_status": CriticStatus.UNAVAILABLE,
                "critic_execution_summary": summary,
                "soft_critiques": [],
                "critic_grounding_attempts": semantic_attempt,
                "critic_grounding_errors": [],
                "status": "critic_unavailable",
            }
        _log_node_completed(
            state,
            "soft_constraint_critic",
            "critic_completed",
            critique_count=len(result.critiques),
        )
        return {
            "soft_critiques": list(result.critiques),
            "critic_execution_summary": result.summary,
            "critic_status": CriticStatus.SUCCESS,
            "critic_grounding_attempts": semantic_attempt,
            "status": "critic_completed",
        }

    def validate_critic_evidence(state: TravelState) -> dict:
        _log_node_started(state, "validate_critic_evidence")
        if state["critic_status"] is not CriticStatus.SUCCESS:
            return {"critic_grounding_errors": [], "status": "critic_degraded"}
        errors = validate_critique_grounding(
            tuple(state["critic_evidence_digests"]),
            tuple(state["soft_critiques"]),
        )
        if errors:
            status = CriticStatus.INVALID_GROUNDING
            logger.warning(
                "critic.grounding_rejected | thread_id=%s attempt=%s "
                "error_count=%s error_codes=%s",
                state["thread_id"],
                state["critic_grounding_attempts"],
                len(errors),
                ",".join(errors),
            )
        else:
            status = CriticStatus.SUCCESS
            logger.info(
                "critic.grounding_validated | thread_id=%s attempt=%s "
                "candidate_count=%s evidence_reference_count=%s",
                state["thread_id"],
                state["critic_grounding_attempts"],
                len(state["soft_critiques"]),
                sum(
                    len(dimension.evidence_ids)
                    for critique in state["soft_critiques"]
                    for dimension in critique.dimensions
                ),
            )
        summary = state["critic_execution_summary"]
        if summary is not None and status is not summary.status:
            summary = summary.model_copy(update={"status": status})
        _log_node_completed(
            state,
            "validate_critic_evidence",
            "grounding_valid" if not errors else "grounding_invalid",
            error_count=len(errors),
        )
        return {
            "critic_status": status,
            "critic_execution_summary": summary,
            "critic_grounding_errors": list(errors),
            "status": "grounding_valid" if not errors else "grounding_invalid",
        }

    def quality_gate(state: TravelState) -> dict:
        _log_node_started(state, "quality_gate")
        scores = (
            calculate_quality_scores(tuple(state["soft_critiques"]), critic_policy)
            if state["critic_status"] is CriticStatus.SUCCESS
            else {}
        )
        decision = "deterministic_fallback"
        if scores:
            best_score = max(scores.values())
            decision = (
                "acceptable"
                if best_score >= critic_policy.quality_threshold
                else "repairable_check"
            )
        else:
            best_score = None
        logger.info(
            "quality_gate.decision | thread_id=%s critic_status=%s "
            "decision=%s best_score=%s threshold=%s soft_round=%s",
            state["thread_id"],
            state["critic_status"].value,
            decision,
            best_score,
            critic_policy.quality_threshold,
            state["pending_soft_replan_round"] or 0,
        )
        _log_node_completed(state, "quality_gate", "quality_scored")
        return {"quality_scores": scores, "status": "quality_scored"}

    def compile_soft_repair_plan(state: TravelState) -> dict:
        _log_node_started(state, "compile_soft_repair_plan")
        deliverable = _deliverable_candidates(state)
        selected = choose_by_quality(deliverable, state["quality_scores"])
        score = state["quality_scores"].get(selected.id)
        if (
            score is None
            or score >= critic_policy.quality_threshold
            or state["soft_iterations"] >= critic_policy.max_soft_replan_rounds
        ):
            plan, reason = None, "quality_acceptable_or_budget_exhausted"
        else:
            critique = next(
                item for item in state["soft_critiques"] if item.candidate_id == selected.id
            )
            draft = next(
                item for item in state["candidate_drafts"] if item.id == selected.id
            )
            plan, reason = create_soft_repair_plan(
                state["trip"],
                selected,
                draft,
                state["planning_pois"],
                critique,
                repair_round=state["soft_iterations"] + 1,
            )
        logger.info(
            "soft_repair.plan_created | thread_id=%s candidate_id=%s "
            "score=%s action=%s status=%s reason=%s",
            state["thread_id"],
            selected.id,
            score,
            plan.action.kind.value if plan else "none",
            "ready" if plan else "skipped",
            reason or "none",
        )
        return {
            "soft_repair_plan": plan,
            "soft_repair_terminal_reason": reason,
            "soft_baseline_candidates": list(state["candidates"]) if plan else [],
            "soft_baseline_candidate_drafts": list(state["candidate_drafts"]) if plan else [],
            "soft_baseline_critiques": list(state["soft_critiques"]) if plan else [],
            "soft_baseline_evidence_digests": list(state["critic_evidence_digests"]) if plan else [],
            "soft_baseline_quality_scores": dict(state["quality_scores"]) if plan else {},
            "soft_baseline_critic_execution_summary": state["critic_execution_summary"] if plan else None,
            "status": "soft_repair_ready" if plan else "soft_repair_skipped",
        }

    def apply_soft_repair(state: TravelState) -> dict:
        _log_node_started(state, "apply_soft_repair")
        plan = state["soft_repair_plan"]
        if plan is None:
            raise RuntimeError("soft repair requires a compiled plan")
        baseline_candidate = next(
            item for item in state["soft_baseline_candidates"] if item.id == plan.target_candidate_id
        )
        baseline_draft = next(
            item for item in state["soft_baseline_candidate_drafts"] if item.id == plan.target_candidate_id
        )
        repaired = apply_soft_repair_to_draft(baseline_draft, plan)
        preserved_hashes = {
            day.date.isoformat(): day_fingerprint(day)
            for day in baseline_candidate.days
            if day.date in plan.preserved_days
        }
        logger.info(
            "soft_repair.action_applied | thread_id=%s round=%s candidate_id=%s "
            "repaired_candidate_id=%s action=%s affected_days=%s preserved_days=%s",
            state["thread_id"],
            plan.round,
            plan.target_candidate_id,
            repaired.id,
            plan.action.kind.value,
            len(plan.affected_days),
            len(plan.preserved_days),
        )
        return {
            "pending_soft_replan_round": plan.round,
            "candidate_drafts": [repaired],
            "candidates": [],
            "selected_plan": None,
            "route_queries": [],
            "delta_route_queries": [],
            "reused_route_keys": [],
            "preserved_day_hashes": preserved_hashes,
            "critic_status": CriticStatus.PENDING,
            "soft_critiques": [],
            "quality_scores": {},
            "last_route_loaded_count": 0,
            "last_route_reused_count": 0,
            "status": "soft_repair_applied",
        }

    def collect_soft_delta_routes(state: TravelState) -> dict:
        _log_node_started(state, "collect_soft_delta_routes")
        draft = state["candidate_drafts"][0]
        reusable_routes = {} if eval_overrides.full_replan else state["route_results"]
        delta = collect_route_delta(
            state["trip"],
            draft,
            state["planning_pois"],
            reusable_routes,
            route_strategy=policy.route_strategy,
            route_modes=policy.route_modes,
            max_walking_leg_meters=policy.max_walking_leg_meters,
        )
        logger.info(
            "soft_repair.routes_reused | thread_id=%s round=%s required=%s "
            "reused=%s missing=%s",
            state["thread_id"],
            state["pending_soft_replan_round"],
            len(delta.required_queries),
            len(delta.reused_route_keys),
            len(delta.missing_queries),
        )
        return {
            "route_queries": list(delta.required_queries),
            "delta_route_queries": list(delta.missing_queries),
            "reused_route_keys": list(delta.reused_route_keys),
            "last_route_reused_count": len(delta.reused_route_keys),
            "last_route_loaded_count": 0,
            "status": "soft_delta_routes_collected",
        }

    async def load_soft_delta_routes(state: TravelState) -> dict:
        _log_node_started(state, "load_soft_delta_routes")
        queries = state["delta_route_queries"]
        results = await gateway.get_routes(
            queries,
            ToolCallContext(thread_id=state["thread_id"]),
        )
        for result in results.values():
            if result.status is ToolStatus.FAILED:
                raise ToolUnavailableError.from_result(result, state["thread_id"])
        loaded = {}
        for key, result in results.items():
            assert result.data is not None
            loaded[key] = result.data
        summaries = [
            _tool_summary(
                result,
                f"route.get_{result.data.mode.value}" if result.data else "route.get",
            )
            for result in results.values()
        ]
        logger.info(
            "soft_repair.routes_loaded | thread_id=%s round=%s route_count=%s",
            state["thread_id"],
            state["pending_soft_replan_round"],
            len(loaded),
        )
        return {
            "route_results": {**state["route_results"], **loaded},
            "last_route_loaded_count": len(loaded),
            "tool_summaries": [*state["tool_summaries"], *summaries],
            "status": "soft_delta_routes_loaded",
        }

    def materialize_soft_candidate(state: TravelState) -> dict:
        return _materialize(state, "materialize_soft_candidate")

    def _restore_soft_values(
        state: TravelState,
        *,
        reason: str,
        hard_validation_passed: bool,
        after_score: float | None,
    ) -> dict:
        plan = state["soft_repair_plan"]
        if plan is None:
            raise RuntimeError("soft baseline restore requires a repair plan")
        before_score = state["soft_baseline_quality_scores"][plan.target_candidate_id]
        attempt = SoftRepairAttempt(
            round=plan.round,
            before_quality_score=before_score,
            after_quality_score=after_score,
            hard_validation_passed=hard_validation_passed,
            accepted=False,
            reused_route_count=state["last_route_reused_count"],
            loaded_route_count=state["last_route_loaded_count"],
            terminal_reason=reason,
        )
        logger.info(
            "soft_repair.rejected | thread_id=%s round=%s reason=%s "
            "before_score=%s after_score=%s hard_validation_passed=%s",
            state["thread_id"],
            plan.round,
            reason,
            before_score,
            after_score,
            str(hard_validation_passed).lower(),
        )
        return {
            "candidates": list(state["soft_baseline_candidates"]),
            "candidate_drafts": list(state["soft_baseline_candidate_drafts"]),
            "soft_critiques": list(state["soft_baseline_critiques"]),
            "critic_evidence_digests": list(state["soft_baseline_evidence_digests"]),
            "quality_scores": dict(state["soft_baseline_quality_scores"]),
            "critic_execution_summary": state["soft_baseline_critic_execution_summary"],
            "critic_status": CriticStatus.SUCCESS,
            "critic_grounding_errors": [],
            "pending_soft_replan_round": None,
            "soft_iterations": plan.round,
            "soft_repair_history": [*state["soft_repair_history"], attempt],
            "soft_repair_terminal_reason": reason,
            "soft_baseline_candidates": [],
            "soft_baseline_candidate_drafts": [],
            "soft_baseline_critiques": [],
            "soft_baseline_evidence_digests": [],
            "soft_baseline_quality_scores": {},
            "soft_baseline_critic_execution_summary": None,
            "status": "soft_baseline_restored",
        }

    def restore_soft_baseline(state: TravelState) -> dict:
        return _restore_soft_values(
            state,
            reason="hard_validation_failed",
            hard_validation_passed=False,
            after_score=None,
        )

    def compare_soft_repair(state: TravelState) -> dict:
        _log_node_started(state, "compare_soft_repair")
        plan = state["soft_repair_plan"]
        if plan is None or state["pending_soft_replan_round"] is None:
            raise RuntimeError("soft comparison requires pending repair context")
        before = state["soft_baseline_quality_scores"][plan.target_candidate_id]
        after = next(iter(state["quality_scores"].values()), None)
        accepted = (
            state["critic_status"] is CriticStatus.SUCCESS
            and after is not None
            and after - before >= critic_policy.min_improvement
        )
        if not accepted:
            reason = (
                "critic_unavailable_after_repair"
                if state["critic_status"] is not CriticStatus.SUCCESS
                else "quality_improvement_below_threshold"
            )
            return _restore_soft_values(
                state,
                reason=reason,
                hard_validation_passed=True,
                after_score=after,
            )
        attempt = SoftRepairAttempt(
            round=plan.round,
            before_quality_score=before,
            after_quality_score=after,
            hard_validation_passed=True,
            accepted=True,
            reused_route_count=state["last_route_reused_count"],
            loaded_route_count=state["last_route_loaded_count"],
            terminal_reason="quality_improved",
        )
        logger.info(
            "soft_repair.accepted | thread_id=%s round=%s before_score=%s "
            "after_score=%s improvement=%s",
            state["thread_id"],
            plan.round,
            before,
            after,
            round(after - before, 2),
        )
        repaired_candidates = list(state["candidates"])
        repaired_drafts = list(state["candidate_drafts"])
        repaired_critiques = list(state["soft_critiques"])
        repaired_digests = list(state["critic_evidence_digests"])
        repaired_scores = dict(state["quality_scores"])
        merged_candidates = [
            item
            for item in state["soft_baseline_candidates"]
            if item.id != plan.target_candidate_id
        ] + repaired_candidates
        merged_drafts = [
            item
            for item in state["soft_baseline_candidate_drafts"]
            if item.id != plan.target_candidate_id
        ] + repaired_drafts
        merged_critiques = [
            item
            for item in state["soft_baseline_critiques"]
            if item.candidate_id != plan.target_candidate_id
        ] + repaired_critiques
        merged_digests = [
            item
            for item in state["soft_baseline_evidence_digests"]
            if item.candidate_id != plan.target_candidate_id
        ] + repaired_digests
        merged_scores = {
            candidate_id: score
            for candidate_id, score in state[
                "soft_baseline_quality_scores"
            ].items()
            if candidate_id != plan.target_candidate_id
        }
        merged_scores.update(repaired_scores)
        return {
            "candidates": merged_candidates,
            "candidate_drafts": merged_drafts,
            "soft_critiques": merged_critiques,
            "critic_evidence_digests": merged_digests,
            "quality_scores": merged_scores,
            "pending_soft_replan_round": None,
            "soft_iterations": plan.round,
            "soft_repair_history": [*state["soft_repair_history"], attempt],
            "soft_repair_terminal_reason": "quality_improved",
            "soft_baseline_candidates": [],
            "soft_baseline_candidate_drafts": [],
            "soft_baseline_critiques": [],
            "soft_baseline_evidence_digests": [],
            "soft_baseline_quality_scores": {},
            "soft_baseline_critic_execution_summary": None,
            "status": "soft_repair_accepted",
        }

    def select_by_quality(state: TravelState) -> dict:
        _log_node_started(state, "select_by_quality")
        deliverable = _deliverable_candidates(state)
        selected = (
            choose_by_quality(deliverable, state["quality_scores"])
            if state["quality_scores"]
            else select_best_candidate(deliverable)
        )
        logger.info(
            "plan.selected | thread_id=%s candidate_id=%s style=%s "
            "validation_status=%s deterministic_score=%s soft_score=%s critic_status=%s",
            state["thread_id"],
            selected.id,
            selected.style.value,
            selected.validation.status.value if selected.validation else "none",
            selected.score,
            state["quality_scores"].get(selected.id),
            state["critic_status"].value,
        )
        return {"selected_plan": selected, "status": "quality_selected"}

    def explain_selection(state: TravelState) -> dict:
        _log_node_started(state, "explain_selection")
        selected = state["selected_plan"]
        if selected is None:
            raise RuntimeError("selection explanation requires a selected plan")
        digest = next(
            (
                item
                for item in state["critic_evidence_digests"]
                if item.candidate_id == selected.id
            ),
            None,
        )
        critique = next(
            (
                item
                for item in state["soft_critiques"]
                if item.candidate_id == selected.id
            ),
            None,
        )
        explanation = build_grounded_explanation(
            selected,
            status=state["critic_status"],
            digest=digest,
            critique=critique,
        )
        logger.info(
            "explanation.generated | thread_id=%s candidate_id=%s "
            "critic_status=%s highlight_count=%s tradeoff_count=%s",
            state["thread_id"],
            selected.id,
            state["critic_status"].value,
            len(explanation.highlights),
            len(explanation.tradeoffs),
        )
        _log_node_completed(state, "explain_selection", "completed")
        return {
            "grounded_explanation": explanation,
            "status": "completed",
            "message": f"已选择 {selected.style.value} 方案",
        }

    def select_best(state: TravelState) -> dict:
        _log_node_started(state, "select_best")
        deliverable = _deliverable_candidates(state)
        selected = select_best_candidate(deliverable)
        logger.info(
            "plan.selected | thread_id=%s candidate_id=%s style=%s "
            "validation_status=%s score=%s cost=%s travel_minutes=%s",
            state["thread_id"],
            selected.id,
            selected.style.value,
            selected.validation.status.value if selected.validation else "none",
            selected.score,
            selected.metrics.estimated_cost,
            selected.metrics.total_travel_minutes,
        )
        _log_node_completed(state, "select_best", "completed", selected=selected.id)
        return {
            "selected_plan": selected,
            "status": "completed",
            "message": f"已选择 {selected.style.value} 方案",
        }

    def mark_infeasible(state: TravelState) -> dict:
        _log_node_started(state, "mark_infeasible")
        logger.warning(
            "planning.infeasible | thread_id=%s iterations=%s candidate_count=%s "
            "reason=%s",
            state["thread_id"],
            state["iterations"],
            len(state["candidates"]),
            state["repair_terminal_reason"] or "budget_or_candidates_exhausted",
        )
        _log_node_completed(state, "mark_infeasible", "infeasible")
        return {
            "selected_plan": None,
            "status": "infeasible",
            "message": (
                "在当前约束和局部修复预算内没有找到合法方案"
                f"（{state['repair_terminal_reason']}）"
                if state["repair_terminal_reason"]
                else "在当前约束和局部修复预算内没有找到合法方案"
            ),
        }

    def select_unvalidated_for_evaluation(state: TravelState) -> dict:
        """评测沙箱专用：暴露无 Validator 的不安全基线，绝不用于生产 API。"""
        _log_node_started(state, "select_unvalidated_for_evaluation")
        if not state["candidates"]:
            raise RuntimeError("unvalidated evaluation requires candidates")
        selected = min(
            state["candidates"],
            key=lambda candidate: (
                -(candidate.score if candidate.score is not None else float("-inf")),
                candidate.id,
            ),
        )
        logger.warning(
            "evaluation.unsafe_validator_bypass | thread_id=%s candidate_id=%s",
            state["thread_id"],
            selected.id,
        )
        _log_node_completed(
            state, "select_unvalidated_for_evaluation", "completed"
        )
        return {
            "selected_plan": selected,
            "status": "completed",
            "message": "evaluation-only unvalidated candidate",
        }

    builder = StateGraph(TravelState)

    def add_node(name: str, function, *, terminal: bool = False) -> None:
        builder.add_node(
            name, instrument_node("planning", name, function, terminal=terminal)
        )

    add_node("execution_budget_guard", execution_budget_guard)
    add_node("build_search_plan", build_search_plan)
    add_node("load_pois", load_pois)
    add_node("resolve_poi_facts", resolve_poi_facts)
    add_node("resolve_stay_anchor", resolve_stay_anchor_node)
    add_node("derive_day_boundaries", derive_day_boundaries_node)
    add_node("build_route_matrix", build_route_matrix)
    add_node("build_optimization_problem", build_optimization_problem)
    add_node("solve_candidate_variants", solve_candidate_variants)
    add_node("materialize_optimized_candidates", materialize_optimized)
    add_node("materialize_candidates", materialize_repaired)
    add_node("validate_candidates", validate, terminal=True)
    add_node("select_repair_target", select_repair_target)
    add_node("analyze_violations", analyze_violations)
    add_node("build_repair_plan", create_repair_plan)
    add_node("apply_local_repair", apply_local_repair)
    add_node("collect_delta_routes", collect_delta_routes)
    add_node("load_delta_routes", load_delta_routes)
    add_node("prepare_critic_context", prepare_critic_context)
    add_node("soft_constraint_critic", soft_constraint_critic)
    add_node("validate_critic_evidence", validate_critic_evidence)
    add_node("quality_gate", quality_gate)
    add_node("compile_soft_repair_plan", compile_soft_repair_plan)
    add_node("apply_soft_repair", apply_soft_repair)
    add_node("collect_soft_delta_routes", collect_soft_delta_routes)
    add_node("load_soft_delta_routes", load_soft_delta_routes)
    add_node("materialize_soft_candidate", materialize_soft_candidate)
    add_node("restore_soft_baseline", restore_soft_baseline)
    add_node("compare_soft_repair", compare_soft_repair)
    add_node("select_by_quality", select_by_quality)
    add_node("explain_selection", explain_selection, terminal=True)
    add_node("select_best", select_best, terminal=True)
    add_node("mark_infeasible", mark_infeasible, terminal=True)
    if eval_overrides.skip_validator:
        add_node(
            "select_unvalidated_for_evaluation",
            select_unvalidated_for_evaluation,
            terminal=True,
        )

    builder.add_edge(START, "execution_budget_guard")
    builder.add_edge("execution_budget_guard", "build_search_plan")
    builder.add_edge("build_search_plan", "load_pois")
    builder.add_edge("load_pois", "resolve_poi_facts")
    builder.add_edge("resolve_poi_facts", "resolve_stay_anchor")
    builder.add_edge("resolve_stay_anchor", "derive_day_boundaries")
    builder.add_edge("derive_day_boundaries", "build_route_matrix")
    builder.add_edge("build_route_matrix", "build_optimization_problem")
    builder.add_edge("build_optimization_problem", "solve_candidate_variants")
    builder.add_edge(
        "solve_candidate_variants", "materialize_optimized_candidates"
    )
    initial_validation_target = (
        "select_unvalidated_for_evaluation"
        if eval_overrides.skip_validator
        else "validate_candidates"
    )
    builder.add_edge("materialize_optimized_candidates", initial_validation_target)
    builder.add_edge("materialize_candidates", initial_validation_target)
    builder.add_conditional_edges(
        "validate_candidates",
        instrument_route(
            "planning",
            "validate_candidates",
            (
                _route_after_validation_without_soft_critic
                if eval_overrides.skip_soft_critic
                else route_after_validation
            ),
        ),
    )
    builder.add_edge("select_repair_target", "analyze_violations")
    builder.add_conditional_edges(
        "analyze_violations",
        instrument_route("planning", "analyze_violations", route_after_critic),
    )
    builder.add_conditional_edges(
        "build_repair_plan",
        instrument_route("planning", "build_repair_plan", route_after_repair_plan),
    )
    builder.add_edge("apply_local_repair", "collect_delta_routes")
    builder.add_conditional_edges(
        "collect_delta_routes",
        instrument_route("planning", "collect_delta_routes", route_after_delta_routes),
    )
    builder.add_edge("load_delta_routes", "materialize_candidates")
    builder.add_edge("prepare_critic_context", "soft_constraint_critic")
    builder.add_edge("soft_constraint_critic", "validate_critic_evidence")
    builder.add_conditional_edges(
        "validate_critic_evidence",
        instrument_route(
            "planning", "validate_critic_evidence", route_after_grounding
        ),
    )
    builder.add_conditional_edges(
        "quality_gate",
        instrument_route("planning", "quality_gate", route_after_quality_gate),
    )
    builder.add_conditional_edges(
        "compile_soft_repair_plan",
        instrument_route(
            "planning", "compile_soft_repair_plan", route_after_soft_repair_plan
        ),
    )
    builder.add_edge("apply_soft_repair", "collect_soft_delta_routes")
    builder.add_conditional_edges(
        "collect_soft_delta_routes",
        instrument_route(
            "planning", "collect_soft_delta_routes", route_after_soft_delta_routes
        ),
    )
    builder.add_edge("load_soft_delta_routes", "materialize_soft_candidate")
    builder.add_edge("materialize_soft_candidate", "validate_candidates")
    builder.add_edge("restore_soft_baseline", "select_by_quality")
    builder.add_edge("compare_soft_repair", "select_by_quality")
    builder.add_edge("select_by_quality", "explain_selection")
    builder.add_edge("explain_selection", END)
    builder.add_edge("select_best", END)
    builder.add_edge("mark_infeasible", END)
    if eval_overrides.skip_validator:
        builder.add_edge("select_unvalidated_for_evaluation", END)
    checkpoint_serde = JsonPlusSerializer(
        allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES
    )
    return builder.compile(
        checkpointer=ObservedCheckpointSaver(InMemorySaver(serde=checkpoint_serde))
    )


def initial_state(request: PlanningRequest, thread_id: str) -> TravelState:
    return {
        "thread_id": thread_id,
        "trip": request.trip,
        "search_queries": [],
        "poi_facts": [],
        "planning_pois": [],
        "stay_resolution": None,
        "day_boundaries": (),
        "optimization_pois": [],
        "optimization_problem": None,
        "optimization_result": None,
        "route_matrix_cache_hits": 0,
        "route_matrix_provider_calls": 0,
        "poi_resolution_issues": [],
        "candidate_drafts": [],
        "route_queries": [],
        "delta_route_queries": [],
        "route_results": {},
        "reused_route_keys": [],
        "tool_summaries": [],
        "candidates": [],
        "selected_plan": None,
        "iterations": 0,
        "pending_replan_round": None,
        "max_replan_rounds": request.max_replan_rounds,
        "repair_target_candidate_id": None,
        "critic_report": None,
        "repair_plan": None,
        "repair_history": [],
        "preserved_day_hashes": {},
        "repair_terminal_reason": None,
        "last_route_loaded_count": 0,
        "last_route_reused_count": 0,
        "critic_evidence_digests": [],
        "soft_critiques": [],
        "critic_execution_summary": None,
        "critic_status": CriticStatus.NOT_RUN,
        "critic_grounding_attempts": 0,
        "critic_grounding_max_attempts": 2,
        "critic_grounding_errors": [],
        "quality_scores": {},
        "soft_baseline_candidates": [],
        "soft_baseline_candidate_drafts": [],
        "soft_baseline_critiques": [],
        "soft_baseline_evidence_digests": [],
        "soft_baseline_quality_scores": {},
        "soft_baseline_critic_execution_summary": None,
        "soft_repair_plan": None,
        "soft_repair_history": [],
        "soft_iterations": 0,
        "max_soft_replan_rounds": 1,
        "pending_soft_replan_round": None,
        "soft_repair_terminal_reason": None,
        "grounded_explanation": None,
        "status": "started",
        "message": None,
    }


def response_from_state(state: TravelState) -> PlanningResponse:
    return PlanningResponse(
        status=state["status"],
        selected_plan=state["selected_plan"],
        candidates=state["candidates"],
        iterations=state["iterations"],
        message=state["message"],
        critic_status=state["critic_status"],
        critic_summary=state["critic_execution_summary"],
        candidate_critiques=(
            state["soft_critiques"]
            if state["critic_status"] is CriticStatus.SUCCESS
            else []
        ),
        grounded_explanation=state["grounded_explanation"],
        soft_iterations=state["soft_iterations"],
        stay_resolution=state["stay_resolution"],
    )


async def run_planning(
    workflow: CompiledStateGraph,
    request: PlanningRequest,
    thread_id: str | None = None,
) -> PlanningResponse:
    run_thread_id = thread_id or str(uuid4())
    started_at = perf_counter()
    logger.info(
        "planning.started | thread_id=%s destination=%s days=%s travelers=%s "
        "max_replan_rounds=%s",
        run_thread_id,
        request.trip.destination,
        request.trip.day_count,
        request.trip.travelers,
        request.max_replan_rounds,
    )
    try:
        recursion_limit = max(32, 24 + request.max_replan_rounds * 8)
        result = await workflow.ainvoke(
            initial_state(request, run_thread_id),
            config={
                "configurable": {"thread_id": run_thread_id},
                "recursion_limit": recursion_limit,
            },
        )
    except Exception:
        logger.exception("planning.failed | thread_id=%s", run_thread_id)
        raise
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    selected_id = result["selected_plan"].id if result["selected_plan"] else "none"
    logger.info(
        "planning.completed | thread_id=%s status=%s selected=%s "
        "iterations=%s elapsed_ms=%s",
        run_thread_id,
        result["status"],
        selected_id,
        result["iterations"],
        elapsed_ms,
    )
    return response_from_state(result)
