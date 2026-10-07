from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter

from travel_agent.agents.replanner.models import ModelRepairProposal, proposal_fingerprint
from travel_agent.domain.models import (
    DayBoundary,
    PlanCandidate,
    PlanStyle,
    PlanningPOI,
    POIResolutionIssue,
    StayAnchorResolution,
    TripSpec,
    ValidationStatus,
)
from travel_agent.domain.critique_models import SoftRepairPlan
from travel_agent.domain.optimization_models import (
    OptimizationBudget,
    OptimizationResult,
)
from travel_agent.domain.repair_models import (
    RepairAttempt,
    RepairOutcome,
    RepairPlan,
)
from travel_agent.domain.tool_models import (
    POIFacts,
    RouteQuery,
    RouteResult,
    ToolCallContext,
    ToolStatus,
    route_key,
)
from travel_agent.planning.critic import error_violations, violation_fingerprint
from travel_agent.planning.defaults import POIDefaultPolicy
from travel_agent.planning.drafts import CandidateDraft, prepare_candidate_drafts
from travel_agent.planning.impact import collect_route_delta, day_fingerprint
from travel_agent.planning.optimization import (
    ORToolsOptimizationSolver,
    OptimizationSolver,
    OptimizationTimeoutError,
    build_optimization_problem,
    collect_route_matrix_queries,
    degraded_result,
    drafts_from_optimization,
    select_optimization_pois,
)
from travel_agent.planning.planner import materialize_candidates
from travel_agent.planning.policy import PlanningPolicy
from travel_agent.planning.repair import apply_repair_plan
from travel_agent.planning.soft_repair import apply_soft_repair as apply_soft_repair_to_draft
from travel_agent.planning.stay import derive_day_boundaries, resolve_stay_anchor
from travel_agent.planning.validator import validate_candidate
from travel_agent.tools.errors import ToolUnavailableError
from travel_agent.tools.gateway import ToolGateway


logger = logging.getLogger(__name__)

_DELIVERABLE_STATUSES = {
    ValidationStatus.VALID,
    ValidationStatus.VALID_WITH_WARNINGS,
}


class PlanningEvidenceMissingError(RuntimeError):
    """动态 Planner 在证据不完整时尝试进入确定性求解。"""

    def __init__(self, evidence_kind: str, missing_count: int) -> None:
        super().__init__(f"missing {missing_count} {evidence_kind} evidence records")
        self.evidence_kind = evidence_kind
        self.missing_count = missing_count


@dataclass(frozen=True, slots=True)
class PlanningKernelSnapshot:
    """一次动态规划的领域快照；不包含 Graph、Prompt 或 Provider 原始响应。"""

    planning_pois: tuple[PlanningPOI, ...]
    poi_resolution_issues: tuple[POIResolutionIssue, ...]
    stay_resolution: StayAnchorResolution
    day_boundaries: tuple[DayBoundary, ...]
    optimization_pois: tuple[PlanningPOI, ...]
    optimization_result: OptimizationResult
    candidate_drafts: tuple[CandidateDraft, ...]
    route_results: dict[str, RouteResult]
    candidates: tuple[PlanCandidate, ...]
    iterations: int = 0
    repair_history: tuple[RepairAttempt, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanningKernelSolveResult:
    snapshot: PlanningKernelSnapshot
    selected_plan: PlanCandidate | None
    degraded_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanningKernelRepairResult:
    snapshot: PlanningKernelSnapshot
    selected_plan: PlanCandidate | None
    attempt: RepairAttempt
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PlanningKernelSoftRepairResult:
    snapshot: PlanningKernelSnapshot
    repaired_candidate: PlanCandidate
    reused_route_count: int
    loaded_route_count: int


class PlanningKernelService:
    """为动态 Agent 提供无 LangGraph 依赖的求解、物化、校验与局部修复能力。"""

    def __init__(
        self,
        *,
        gateway: ToolGateway,
        defaults: POIDefaultPolicy,
        policy: PlanningPolicy = PlanningPolicy(),
        optimizer: OptimizationSolver | None = None,
        optimization_budget: OptimizationBudget = OptimizationBudget(),
    ) -> None:
        self.gateway = gateway
        self.defaults = defaults
        self.policy = policy
        self.optimizer = optimizer or ORToolsOptimizationSolver()
        self.optimization_budget = optimization_budget

    async def solve(
        self,
        trip: TripSpec,
        *,
        poi_facts: tuple[POIFacts, ...],
        route_results: dict[str, RouteResult],
        thread_id: str,
        strategy: str = "auto",
    ) -> PlanningKernelSolveResult:
        """消费 Planner 已收集的 Evidence，生成并硬校验候选，不执行固定 Graph。"""

        planning_pois, issues = self._resolve_pois(trip, poi_facts)
        if not planning_pois:
            raise PlanningEvidenceMissingError("poi", 1)

        stay = resolve_stay_anchor(trip, planning_pois)
        boundaries = derive_day_boundaries(trip, stay)
        optimization_pois = select_optimization_pois(
            trip, planning_pois, self.optimization_budget
        )
        required_queries = collect_route_matrix_queries(
            trip,
            optimization_pois,
            modes=self.policy.route_modes,
            strategy=self.policy.route_strategy,
            max_walking_leg_meters=self.policy.max_walking_leg_meters,
            stay_resolution=stay,
            day_boundaries=boundaries,
        )
        missing_keys = tuple(
            key
            for key in (route_key(query) for query in required_queries)
            if key not in route_results
        )
        if missing_keys:
            logger.warning(
                "planning_kernel.solve_evidence_missing | thread_id=%s kind=route "
                "missing_count=%s available_count=%s",
                thread_id,
                len(missing_keys),
                len(route_results),
            )
            raise PlanningEvidenceMissingError("route", len(missing_keys))

        relevant_routes = {
            key: route_results[key] for key in (route_key(item) for item in required_queries)
        }
        problem = build_optimization_problem(
            trip,
            optimization_pois,
            relevant_routes,
            self.optimization_budget,
            modes=self.policy.route_modes,
            strategy=self.policy.route_strategy,
            max_walking_leg_meters=self.policy.max_walking_leg_meters,
            stay_resolution=stay,
            day_boundaries=boundaries,
        )
        started = perf_counter()
        degraded_reasons: list[str] = []
        try:
            optimization_result = self.optimizer.solve(problem)
            degraded_reason = (
                "optimizer_infeasible" if not optimization_result.solutions else None
            )
        except OptimizationTimeoutError:
            optimization_result = None
            degraded_reason = "optimizer_timeout"

        if degraded_reason is not None:
            drafts = prepare_candidate_drafts(
                trip,
                optimization_pois,
                replan_round=0,
                day_boundaries=boundaries,
            )
            optimization_result = degraded_result(
                drafts,
                reason=degraded_reason,
                elapsed_ms=round((perf_counter() - started) * 1000, 2),
            )
            degraded_reasons.append(degraded_reason)
        else:
            assert optimization_result is not None
            drafts = drafts_from_optimization(optimization_result)

        drafts = self._filter_strategy(drafts, strategy)
        candidates = materialize_candidates(
            trip,
            drafts,
            planning_pois,
            relevant_routes,
            route_strategy=self.policy.route_strategy,
            route_modes=self.policy.route_modes,
            max_walking_leg_meters=self.policy.max_walking_leg_meters,
            stay_resolution=stay,
            day_boundaries=boundaries,
        )
        validated = tuple(self._validate(trip, candidate, planning_pois) for candidate in candidates)
        selected = self._select_deliverable(validated)
        snapshot = PlanningKernelSnapshot(
            planning_pois=tuple(planning_pois),
            poi_resolution_issues=tuple(issues),
            stay_resolution=stay,
            day_boundaries=boundaries,
            optimization_pois=tuple(optimization_pois),
            optimization_result=optimization_result,
            candidate_drafts=tuple(drafts),
            route_results=dict(relevant_routes),
            candidates=validated,
        )
        logger.info(
            "planning_kernel.solved | thread_id=%s solver=%s status=%s "
            "candidate_count=%s deliverable_count=%s strategy=%s",
            thread_id,
            optimization_result.solver,
            optimization_result.status.value,
            len(validated),
            sum(self._deliverable(item) for item in validated),
            strategy,
        )
        return PlanningKernelSolveResult(
            snapshot=snapshot,
            selected_plan=selected,
            degraded_reasons=tuple(degraded_reasons),
        )

    async def apply_repair(
        self,
        trip: TripSpec,
        snapshot: PlanningKernelSnapshot,
        proposal: ModelRepairProposal,
        *,
        thread_id: str,
    ) -> PlanningKernelRepairResult:
        """应用已通过 Repair Policy 的模型 Patch，并完成路线增量和硬复验。"""

        candidate = next(
            item for item in snapshot.candidates if item.id == proposal.target_candidate_id
        )
        draft = next(
            item for item in snapshot.candidate_drafts if item.id == candidate.id
        )
        affected_days = tuple(sorted(set(proposal.affected_days)))
        preserved_days = tuple(
            day.date for day in draft.days if day.date not in set(affected_days)
        )
        repair_round = snapshot.iterations + 1
        plan = RepairPlan(
            round=repair_round,
            target_candidate_id=candidate.id,
            source_violation_types=tuple(
                sorted({action.source_violation_type for action in proposal.actions})
            ),
            actions=proposal.actions,
            affected_days=affected_days,
            preserved_days=preserved_days,
            expected_effects=tuple(action.expected_effect for action in proposal.actions),
            action_fingerprint=proposal_fingerprint(proposal),
        )
        repaired_draft, applied_plan = apply_repair_plan(
            trip,
            draft,
            list(snapshot.planning_pois),
            plan,
            route_strategy=self.policy.route_strategy,
            route_modes=self.policy.route_modes,
            max_walking_leg_meters=self.policy.max_walking_leg_meters,
        )
        preserved_hashes = {
            day.date.isoformat(): day_fingerprint(day)
            for day in candidate.days
            if day.date in set(applied_plan.preserved_days)
        }
        reusable_routes = {
            key: value
            for key, value in snapshot.route_results.items()
            if key not in set(applied_plan.invalidated_route_keys)
        }
        delta = collect_route_delta(
            trip,
            repaired_draft,
            list(snapshot.planning_pois),
            reusable_routes,
            route_strategy=self.policy.route_strategy,
            route_modes=self.policy.route_modes,
            max_walking_leg_meters=self.policy.max_walking_leg_meters,
        )
        loaded = await self._load_routes(
            delta.missing_queries,
            thread_id=thread_id,
        )
        routes = {**reusable_routes, **loaded}
        repaired = materialize_candidates(
            trip,
            [repaired_draft],
            list(snapshot.planning_pois),
            routes,
            route_strategy=self.policy.route_strategy,
            route_modes=self.policy.route_modes,
            max_walking_leg_meters=self.policy.max_walking_leg_meters,
            stay_resolution=snapshot.stay_resolution,
            day_boundaries=snapshot.day_boundaries,
        )[0]
        repaired = self._validate(trip, repaired, list(snapshot.planning_pois))
        repaired_days = {day.date.isoformat(): day for day in repaired.days}
        for day_text, expected in preserved_hashes.items():
            day = repaired_days.get(day_text)
            if day is None or day_fingerprint(day) != expected:
                raise RuntimeError(f"model repair changed preserved day: {day_text}")

        before_errors = error_violations(candidate)
        after_errors = error_violations(repaired)
        before_fingerprint = violation_fingerprint(candidate)
        after_fingerprint = violation_fingerprint(repaired)
        repeated = before_fingerprint == after_fingerprint or any(
            after_fingerprint
            in {item.before_violation_fingerprint, item.after_violation_fingerprint}
            for item in snapshot.repair_history
        )
        if self._deliverable(repaired):
            outcome = RepairOutcome.RESOLVED
            terminal_reason = None
        elif len(after_errors) < len(before_errors):
            outcome = RepairOutcome.IMPROVED
            terminal_reason = None
        elif repeated:
            outcome = RepairOutcome.NO_PROGRESS
            terminal_reason = "repeated_violation_fingerprint"
        else:
            outcome = RepairOutcome.NO_PROGRESS
            terminal_reason = "repair_no_progress"
        attempt = RepairAttempt(
            round=repair_round,
            target_candidate_id=candidate.id,
            before_violation_fingerprint=before_fingerprint,
            after_violation_fingerprint=after_fingerprint,
            before_error_count=len(before_errors),
            after_error_count=len(after_errors),
            action_fingerprint=applied_plan.action_fingerprint,
            action_kinds=tuple(action.kind for action in applied_plan.actions),
            outcome=outcome,
            affected_days=applied_plan.affected_days,
            preserved_day_count=len(applied_plan.preserved_days),
            reused_route_count=len(delta.reused_route_keys),
            loaded_route_count=len(loaded),
            terminal_reason=terminal_reason,
        )
        candidates = tuple(
            repaired if item.id == candidate.id else item
            for item in snapshot.candidates
        )
        if all(item.id != repaired.id for item in candidates):
            candidates = tuple(item for item in candidates if item.id != candidate.id) + (
                repaired,
            )
        selected = self._select_deliverable(candidates)
        updated = PlanningKernelSnapshot(
            planning_pois=snapshot.planning_pois,
            poi_resolution_issues=snapshot.poi_resolution_issues,
            stay_resolution=snapshot.stay_resolution,
            day_boundaries=snapshot.day_boundaries,
            optimization_pois=snapshot.optimization_pois,
            optimization_result=snapshot.optimization_result,
            candidate_drafts=tuple(
                repaired_draft if item.id == draft.id else item
                for item in snapshot.candidate_drafts
            ),
            route_results=routes,
            candidates=candidates,
            iterations=repair_round,
            repair_history=(*snapshot.repair_history, attempt),
        )
        logger.info(
            "planning_kernel.repaired | thread_id=%s round=%s candidate_id=%s "
            "outcome=%s before_errors=%s after_errors=%s reused_routes=%s loaded_routes=%s",
            thread_id,
            repair_round,
            candidate.id,
            outcome.value,
            len(before_errors),
            len(after_errors),
            len(delta.reused_route_keys),
            len(loaded),
        )
        return PlanningKernelRepairResult(
            snapshot=updated,
            selected_plan=selected,
            attempt=attempt,
            terminal_reason=terminal_reason,
        )

    async def apply_soft_repair(
        self,
        trip: TripSpec,
        snapshot: PlanningKernelSnapshot,
        plan: SoftRepairPlan,
        *,
        thread_id: str,
    ) -> PlanningKernelSoftRepairResult:
        """应用 Grounded Critic 编译出的单步软修复，并重新完成路线和硬校验。"""

        candidate = next(
            item for item in snapshot.candidates if item.id == plan.target_candidate_id
        )
        draft = next(
            item
            for item in snapshot.candidate_drafts
            if item.id == plan.target_candidate_id
        )
        repaired_draft = apply_soft_repair_to_draft(draft, plan)
        preserved_hashes = {
            day.date.isoformat(): day_fingerprint(day)
            for day in candidate.days
            if day.date in set(plan.preserved_days)
        }
        delta = collect_route_delta(
            trip,
            repaired_draft,
            list(snapshot.planning_pois),
            snapshot.route_results,
            route_strategy=self.policy.route_strategy,
            route_modes=self.policy.route_modes,
            max_walking_leg_meters=self.policy.max_walking_leg_meters,
        )
        loaded = await self._load_routes(delta.missing_queries, thread_id=thread_id)
        routes = {**snapshot.route_results, **loaded}
        repaired = materialize_candidates(
            trip,
            [repaired_draft],
            list(snapshot.planning_pois),
            routes,
            route_strategy=self.policy.route_strategy,
            route_modes=self.policy.route_modes,
            max_walking_leg_meters=self.policy.max_walking_leg_meters,
            stay_resolution=snapshot.stay_resolution,
            day_boundaries=snapshot.day_boundaries,
        )[0]
        repaired = self._validate(trip, repaired, list(snapshot.planning_pois))
        repaired_days = {day.date.isoformat(): day for day in repaired.days}
        for day_text, expected in preserved_hashes.items():
            day = repaired_days.get(day_text)
            if day is None or day_fingerprint(day) != expected:
                raise RuntimeError(f"soft repair changed preserved day: {day_text}")

        candidates = tuple(
            repaired if item.id == candidate.id else item
            for item in snapshot.candidates
        )
        if all(item.id != repaired.id for item in candidates):
            candidates = tuple(
                item for item in candidates if item.id != candidate.id
            ) + (repaired,)
        drafts = tuple(
            repaired_draft if item.id == draft.id else item
            for item in snapshot.candidate_drafts
        )
        if all(item.id != repaired_draft.id for item in drafts):
            drafts = tuple(item for item in drafts if item.id != draft.id) + (
                repaired_draft,
            )
        updated = PlanningKernelSnapshot(
            planning_pois=snapshot.planning_pois,
            poi_resolution_issues=snapshot.poi_resolution_issues,
            stay_resolution=snapshot.stay_resolution,
            day_boundaries=snapshot.day_boundaries,
            optimization_pois=snapshot.optimization_pois,
            optimization_result=snapshot.optimization_result,
            candidate_drafts=drafts,
            route_results=routes,
            candidates=candidates,
            iterations=snapshot.iterations,
            repair_history=snapshot.repair_history,
        )
        logger.info(
            "planning_kernel.soft_repaired | thread_id=%s round=%s "
            "candidate_id=%s repaired_candidate_id=%s hard_valid=%s "
            "reused_routes=%s loaded_routes=%s",
            thread_id,
            plan.round,
            candidate.id,
            repaired.id,
            bool(repaired.validation and repaired.validation.valid),
            len(delta.reused_route_keys),
            len(loaded),
        )
        return PlanningKernelSoftRepairResult(
            snapshot=updated,
            repaired_candidate=repaired,
            reused_route_count=len(delta.reused_route_keys),
            loaded_route_count=len(loaded),
        )

    def _resolve_pois(
        self, trip: TripSpec, facts: tuple[POIFacts, ...]
    ) -> tuple[list[PlanningPOI], list[POIResolutionIssue]]:
        planning_pois: list[PlanningPOI] = []
        issues: list[POIResolutionIssue] = []
        for item in facts:
            resolution = self.defaults.resolve(item, trip)
            if resolution.poi is not None:
                planning_pois.append(resolution.poi)
                continue
            normalized = item.name.strip().casefold()
            issues.append(
                POIResolutionIssue(
                    poi_id=item.id,
                    poi_name=item.name,
                    missing_fields=resolution.missing_fields,
                    required=any(
                        required.strip().casefold() in normalized
                        or normalized in required.strip().casefold()
                        for required in trip.must_visit
                    ),
                )
            )
        return planning_pois, issues

    async def _load_routes(
        self, queries: tuple[RouteQuery, ...], *, thread_id: str
    ) -> dict[str, RouteResult]:
        if not queries:
            return {}
        results = await self.gateway.get_routes(
            list(queries), ToolCallContext(thread_id=thread_id)
        )
        loaded: dict[str, RouteResult] = {}
        for key, result in results.items():
            if result.status is ToolStatus.FAILED:
                raise ToolUnavailableError.from_result(result, thread_id)
            assert result.data is not None
            loaded[key] = result.data
        return loaded

    @staticmethod
    def _filter_strategy(
        drafts: list[CandidateDraft], strategy: str
    ) -> list[CandidateDraft]:
        if strategy == "auto":
            return drafts
        requested = PlanStyle(strategy)
        selected = [item for item in drafts if item.style is requested]
        return selected or drafts

    @staticmethod
    def _validate(
        trip: TripSpec, candidate: PlanCandidate, pois: list[PlanningPOI]
    ) -> PlanCandidate:
        return candidate.model_copy(
            update={"validation": validate_candidate(trip, candidate, pois)}
        )

    @staticmethod
    def _deliverable(candidate: PlanCandidate) -> bool:
        return bool(
            candidate.validation
            and candidate.validation.status in _DELIVERABLE_STATUSES
        )

    @classmethod
    def _select_deliverable(
        cls, candidates: tuple[PlanCandidate, ...]
    ) -> PlanCandidate | None:
        deliverable = [item for item in candidates if cls._deliverable(item)]
        if not deliverable:
            return None
        return min(
            deliverable,
            key=lambda item: (
                0
                if item.validation
                and item.validation.status is ValidationStatus.VALID
                else 1,
                -(item.score if item.score is not None else float("-inf")),
                item.id,
            ),
        )
