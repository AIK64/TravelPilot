from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from travel_agent import __version__
from travel_agent.config import Settings
from travel_agent.evaluation.loader import load_jsonl, load_manifest
from travel_agent.evaluation.models import (
    AblationEvalReport,
    EvaluationVariant,
    EvidenceLevel,
    GateResult,
    ReleaseCaseResult,
    VariantSummary,
)
from travel_agent.evaluation.runner import _build_provenance, _run_case
from travel_agent.graph.evaluation import PlanningEvaluationOverrides
from travel_agent.runtime import PlanningRuntime


_VARIANTS = (
    EvaluationVariant.FULL,
    EvaluationVariant.NO_VALIDATOR,
    EvaluationVariant.NO_OPTIMIZER,
    EvaluationVariant.NO_SOFT_CRITIC,
    EvaluationVariant.FULL_REPLAN,
    EvaluationVariant.CACHE_OFF,
)


async def run_ablation_evaluation(
    manifest_path: Path,
    *,
    settings: Settings | None = None,
    random_seed: int = 0,
) -> AblationEvalReport:
    """实际执行六套隔离工作流；消融开关不会进入生产配置。"""
    manifest = load_manifest(manifest_path)
    root = manifest_path.resolve().parents[2]
    dataset_path = root / manifest.base_dataset
    sources = load_jsonl(dataset_path)
    # v1.0 消融指标以固定工作流为实验对象。生产默认已迁移到动态 Planner，
    # 因此这里必须显式锁定历史基线，避免“默认模式变化”污染纵向对比。
    active_settings = settings or Settings.from_env(
        {"AGENT_MODE": "fixed_workflow"}
    )
    started = datetime.now(timezone.utc)
    results: list[ReleaseCaseResult] = []

    for variant in _VARIANTS:
        runtime = await PlanningRuntime.create(
            active_settings,
            evaluation_overrides=_overrides_for(variant),
            tool_cache_enabled=variant is not EvaluationVariant.CACHE_OFF,
        )
        try:
            for source in sources:
                results.append(
                    await _run_case(
                        runtime,
                        source,
                        variant=variant,
                        evidence_level=EvidenceLevel.WORKFLOW_EXECUTION,
                    )
                )
        finally:
            await runtime.close()

    ended = datetime.now(timezone.utc)
    cases = tuple(results)
    summaries = tuple(_summarize(variant, cases) for variant in _VARIANTS)
    gates = _comparison_gates(cases, summaries)
    return AblationEvalReport(
        dataset_version=manifest.dataset_version,
        project_version=__version__,
        started_at=started,
        ended_at=ended,
        case_count=len(cases),
        workflow_execution_count=len(cases),
        variants=summaries,
        cases=cases,
        comparison_gates=gates,
        gate_passed=all(gate.passed for gate in gates),
        provenance=_build_provenance(
            root=root,
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            settings=active_settings,
            random_seed=random_seed,
            pricing_registry=None,
        ),
    )


def _overrides_for(variant: EvaluationVariant) -> PlanningEvaluationOverrides:
    return PlanningEvaluationOverrides(
        skip_validator=variant is EvaluationVariant.NO_VALIDATOR,
        force_heuristic_optimizer=variant is EvaluationVariant.NO_OPTIMIZER,
        skip_soft_critic=variant is EvaluationVariant.NO_SOFT_CRITIC,
        full_replan=variant is EvaluationVariant.FULL_REPLAN,
    )


def _summarize(
    variant: EvaluationVariant,
    cases: tuple[ReleaseCaseResult, ...],
) -> VariantSummary:
    selected = [case for case in cases if case.variant is variant]
    completed = [
        case for case in selected if case.terminal_reason == "plan_completed"
    ]
    known_hard = [
        case.hard_constraints_satisfied
        for case in completed
        if case.hard_constraints_satisfied is not None
    ]
    return VariantSummary(
        variant=variant,
        case_count=len(selected),
        completed_plan_count=len(completed),
        hard_constraint_satisfaction_rate=_rate(known_hard),
        unsafe_delivery_count=sum(
            case.hard_constraints_satisfied is False for case in completed
        ),
        trace_completeness_rate=_rate(case.trace_complete for case in selected),
        bounded_termination_rate=_rate(case.bounded for case in selected),
        total_tool_calls=sum(case.tool_calls for case in selected),
        total_provider_attempts=sum(case.provider_attempts for case in selected),
        total_cache_hits=sum(case.cache_hits for case in selected),
        total_llm_calls=sum(case.llm_calls for case in selected),
        latency_p50_ms=_percentile([case.elapsed_ms for case in selected], 0.5),
        latency_p95_ms=_percentile([case.elapsed_ms for case in selected], 0.95),
    )


def _comparison_gates(
    cases: tuple[ReleaseCaseResult, ...],
    summaries: tuple[VariantSummary, ...],
) -> tuple[GateResult, ...]:
    by_variant = {summary.variant: summary for summary in summaries}
    full = by_variant[EvaluationVariant.FULL]
    cache_off = by_variant[EvaluationVariant.CACHE_OFF]
    no_soft = by_variant[EvaluationVariant.NO_SOFT_CRITIC]
    no_optimizer = by_variant[EvaluationVariant.NO_OPTIMIZER]
    full_replan = by_variant[EvaluationVariant.FULL_REPLAN]
    fingerprints_match = _fingerprints(cases, EvaluationVariant.FULL) == _fingerprints(
        cases, EvaluationVariant.CACHE_OFF
    )
    return (
        GateResult(
            name="all_variants_bounded",
            passed=all(item.bounded_termination_rate == 1.0 for item in summaries),
            actual=min(item.bounded_termination_rate for item in summaries),
            expected=1.0,
        ),
        GateResult(
            name="all_traces_complete",
            passed=all(item.trace_completeness_rate == 1.0 for item in summaries),
            actual=min(item.trace_completeness_rate for item in summaries),
            expected=1.0,
        ),
        GateResult(
            name="full_hard_constraints",
            passed=full.hard_constraint_satisfaction_rate == 1.0,
            actual=full.hard_constraint_satisfaction_rate,
            expected=1.0,
        ),
        GateResult(
            name="soft_critic_no_hard_regression",
            passed=no_soft.hard_constraint_satisfaction_rate >= full.hard_constraint_satisfaction_rate,
            actual=no_soft.hard_constraint_satisfaction_rate,
            expected=f">={full.hard_constraint_satisfaction_rate}",
        ),
        GateResult(
            name="heuristic_no_hard_regression",
            passed=no_optimizer.hard_constraint_satisfaction_rate >= full.hard_constraint_satisfaction_rate,
            actual=no_optimizer.hard_constraint_satisfaction_rate,
            expected=f">={full.hard_constraint_satisfaction_rate}",
        ),
        GateResult(
            name="cache_result_equivalence",
            passed=fingerprints_match,
            actual="equal" if fingerprints_match else "different",
            expected="equal",
        ),
        GateResult(
            name="cache_reduces_provider_attempts",
            passed=full.total_provider_attempts <= cache_off.total_provider_attempts,
            actual=full.total_provider_attempts,
            expected=f"<={cache_off.total_provider_attempts}",
        ),
        GateResult(
            name="local_replan_not_more_tool_calls",
            passed=full.total_tool_calls <= full_replan.total_tool_calls,
            actual=full.total_tool_calls,
            expected=f"<={full_replan.total_tool_calls}",
        ),
    )


def _fingerprints(
    cases: tuple[ReleaseCaseResult, ...], variant: EvaluationVariant
) -> tuple[tuple[str, str | None], ...]:
    return tuple(
        (case.source_case_id, case.result_fingerprint)
        for case in cases
        if case.variant is variant
    )


def _rate(values) -> float:
    items = list(values)
    return round(sum(bool(value) for value in items) / len(items), 4) if items else 1.0


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if quantile == 0.5:
        return round(median(ordered))
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[index]
