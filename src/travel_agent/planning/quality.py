from __future__ import annotations

from dataclasses import dataclass
import logging

from travel_agent.critique.errors import CriticUnavailableError
from travel_agent.critique.evidence import EvidenceBudget, build_evidence_digests
from travel_agent.critique.explanation import build_grounded_explanation
from travel_agent.critique.gateway import CriticGateway
from travel_agent.critique.grounding import validate_critique_grounding
from travel_agent.critique.quality import (
    CriticPolicy,
    quality_scores,
    select_by_quality,
)
from travel_agent.domain.critique_models import (
    CandidateEvidenceDigest,
    CriticExecutionSummary,
    CriticStatus,
    GroundedExplanation,
    SoftCriticRequest,
    SoftCritique,
    SoftRepairPlan,
)
from travel_agent.domain.models import PlanCandidate, TripSpec, ValidationStatus
from travel_agent.execution.context import record_degradation
from travel_agent.execution.errors import ExecutionBudgetExceeded
from travel_agent.planning.kernel import PlanningKernelSnapshot
from travel_agent.planning.soft_repair import compile_soft_repair_plan


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QualityReviewResult:
    """一次可审计的 Grounded Soft Critic 结果。"""

    status: CriticStatus
    evidence_digests: tuple[CandidateEvidenceDigest, ...]
    critiques: tuple[SoftCritique, ...]
    execution_summary: CriticExecutionSummary
    quality_scores: dict[str, float]
    grounding_errors: tuple[str, ...]
    grounding_attempt: int


class QualityReviewService:
    """为动态 Graph 提供无 Graph 依赖的软质量评审和安全改进编译能力。"""

    def __init__(
        self,
        *,
        critic_gateway: CriticGateway | None,
        policy: CriticPolicy = CriticPolicy(),
        evidence_budget: EvidenceBudget = EvidenceBudget(),
    ) -> None:
        self.critic_gateway = critic_gateway
        self.policy = policy
        self.evidence_budget = evidence_budget

    async def review(
        self,
        trip: TripSpec,
        snapshot: PlanningKernelSnapshot,
        *,
        thread_id: str,
        grounding_attempt: int,
        grounding_feedback: tuple[str, ...] = (),
        candidate_ids: tuple[str, ...] = (),
    ) -> QualityReviewResult:
        deliverable = tuple(
            candidate
            for candidate in snapshot.candidates
            if self._deliverable(candidate)
            and (not candidate_ids or candidate.id in candidate_ids)
        )
        digests = build_evidence_digests(
            trip,
            list(deliverable),
            list(snapshot.planning_pois),
            self.evidence_budget,
        )
        if not digests:
            raise RuntimeError("soft critic requires at least one deliverable candidate")
        request = SoftCriticRequest(
            digests=tuple(digests),
            grounding_feedback=grounding_feedback,
        )
        if self.critic_gateway is None:
            summary = CriticExecutionSummary(
                provider="disabled",
                model="disabled",
                prompt_version="disabled",
                status=CriticStatus.DISABLED,
                attempt_count=0,
                grounding_attempt_count=0,
                elapsed_ms=0,
                input_chars=request.input_chars,
            )
            return QualityReviewResult(
                status=CriticStatus.DISABLED,
                evidence_digests=tuple(digests),
                critiques=(),
                execution_summary=summary,
                quality_scores={},
                grounding_errors=(),
                grounding_attempt=0,
            )
        try:
            result = await self.critic_gateway.critique(
                request,
                thread_id=thread_id,
                grounding_attempt=grounding_attempt,
            )
        except ExecutionBudgetExceeded:
            record_degradation("soft_critic_budget_exhausted")
            summary = CriticExecutionSummary(
                provider=self.critic_gateway.provider,
                model=self.critic_gateway.model,
                prompt_version=self.critic_gateway.prompt_version,
                status=CriticStatus.UNAVAILABLE,
                attempt_count=0,
                grounding_attempt_count=grounding_attempt,
                elapsed_ms=0,
                input_chars=request.input_chars,
                error_category="execution_budget_exhausted",
            )
            return QualityReviewResult(
                status=CriticStatus.UNAVAILABLE,
                evidence_digests=tuple(digests),
                critiques=(),
                execution_summary=summary,
                quality_scores={},
                grounding_errors=(),
                grounding_attempt=grounding_attempt,
            )
        except CriticUnavailableError as error:
            summary = CriticExecutionSummary(
                provider=error.provider,
                model=error.model,
                prompt_version=self.critic_gateway.prompt_version,
                status=CriticStatus.UNAVAILABLE,
                attempt_count=error.attempt_count,
                grounding_attempt_count=grounding_attempt,
                elapsed_ms=error.elapsed_ms,
                input_chars=request.input_chars,
                error_category=error.category.value,
            )
            return QualityReviewResult(
                status=CriticStatus.UNAVAILABLE,
                evidence_digests=tuple(digests),
                critiques=(),
                execution_summary=summary,
                quality_scores={},
                grounding_errors=(),
                grounding_attempt=grounding_attempt,
            )

        errors = validate_critique_grounding(tuple(digests), result.critiques)
        status = CriticStatus.INVALID_GROUNDING if errors else CriticStatus.SUCCESS
        summary = result.summary.model_copy(update={"status": status})
        scores = (
            quality_scores(result.critiques, self.policy)
            if status is CriticStatus.SUCCESS
            else {}
        )
        logger.info(
            "quality_review.completed | thread_id=%s status=%s candidate_count=%s "
            "grounding_attempt=%s error_count=%s",
            thread_id,
            status.value,
            len(digests),
            grounding_attempt,
            len(errors),
        )
        return QualityReviewResult(
            status=status,
            evidence_digests=tuple(digests),
            critiques=result.critiques,
            execution_summary=summary,
            quality_scores=scores,
            grounding_errors=errors,
            grounding_attempt=grounding_attempt,
        )

    def compile_improvement(
        self,
        trip: TripSpec,
        snapshot: PlanningKernelSnapshot,
        review: QualityReviewResult,
        *,
        soft_iterations: int,
    ) -> tuple[SoftRepairPlan | None, str | None]:
        selected = self.select(snapshot, review)
        score = review.quality_scores.get(selected.id)
        if review.status is not CriticStatus.SUCCESS or score is None:
            return None, "critic_unavailable_or_invalid"
        if score >= self.policy.quality_threshold:
            return None, "quality_acceptable"
        if soft_iterations >= self.policy.max_soft_replan_rounds:
            return None, "soft_repair_budget_exhausted"
        critique = next(
            item for item in review.critiques if item.candidate_id == selected.id
        )
        draft = next(
            item for item in snapshot.candidate_drafts if item.id == selected.id
        )
        return compile_soft_repair_plan(
            trip,
            selected,
            draft,
            list(snapshot.planning_pois),
            critique,
            repair_round=soft_iterations + 1,
        )

    def select(
        self,
        snapshot: PlanningKernelSnapshot,
        review: QualityReviewResult,
    ) -> PlanCandidate:
        deliverable = [
            item for item in snapshot.candidates if self._deliverable(item)
        ]
        if not deliverable:
            raise RuntimeError("quality selection requires a deliverable candidate")
        if review.quality_scores:
            return select_by_quality(deliverable, review.quality_scores)
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

    def explain(
        self,
        candidate: PlanCandidate,
        review: QualityReviewResult,
    ) -> GroundedExplanation:
        digest = next(
            (
                item
                for item in review.evidence_digests
                if item.candidate_id == candidate.id
            ),
            None,
        )
        critique = next(
            (item for item in review.critiques if item.candidate_id == candidate.id),
            None,
        )
        return build_grounded_explanation(
            candidate,
            status=review.status,
            digest=digest,
            critique=critique,
        )

    @staticmethod
    def _deliverable(candidate: PlanCandidate) -> bool:
        return bool(candidate.validation and candidate.validation.valid)
