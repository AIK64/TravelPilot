from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import sys

from travel_agent import __version__
from travel_agent.config import Settings
from travel_agent.evaluation.loader import load_jsonl, load_manifest
from travel_agent.evaluation.models import (
    EvaluationVariant,
    EvidenceLevel,
    GateResult,
    ReportProvenance,
    ReleaseCaseResult,
    ReleaseEvalReport,
)
from travel_agent.evaluation.pricing import PricingRegistry
from travel_agent.evaluation.trajectory import trace_is_complete
from travel_agent.execution.errors import ExecutionBudgetExceeded
from travel_agent.execution.faults import (
    FaultMode,
    FaultPlan,
    FaultPoint,
    FaultRule,
)
from travel_agent.execution.models import ExecutionBudget, RunStatus
from travel_agent.requirements.errors import RequirementUnavailableError
from travel_agent.requirements.models import NaturalPlanningRequest
from travel_agent.planning.validator import validate_candidate
from travel_agent.runtime import PlanningRuntime


async def run_release_evaluation(
    manifest_path: Path,
    *,
    settings: Settings | None = None,
    pricing_registry: PricingRegistry | None = None,
    random_seed: int = 0,
) -> ReleaseEvalReport:
    manifest = load_manifest(manifest_path)
    root = manifest_path.resolve().parents[2]
    dataset_path = root / manifest.base_dataset
    source_cases = load_jsonl(dataset_path)
    # Release 数据集及既有阈值来自固定工作流；生产默认切换后仍显式复现实验基线。
    active_settings = settings or Settings.from_env(
        {"AGENT_MODE": "fixed_workflow"}
    )
    random.seed(random_seed)
    runtime = await PlanningRuntime.create(active_settings)
    started = datetime.now(timezone.utc)
    results: list[ReleaseCaseResult] = []
    try:
        for matrix in manifest.matrix:
            for source in source_cases:
                results.append(
                    await _run_case(
                        runtime,
                        source,
                        variant=matrix.variant,
                        evidence_level=matrix.evidence_level,
                    )
                )
    finally:
        await runtime.close()
    ended = datetime.now(timezone.utc)
    return _build_report(
        manifest.dataset_version,
        started,
        ended,
        tuple(results),
        minimum_case_count=manifest.minimum_case_count,
        minimum_workflow_count=manifest.minimum_workflow_execution_count,
        provenance=_build_provenance(
            root=root,
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            settings=active_settings,
            random_seed=random_seed,
            pricing_registry=pricing_registry,
        ),
        pricing_registry=pricing_registry,
    )


async def _run_case(
    runtime: PlanningRuntime,
    source: dict,
    *,
    variant: EvaluationVariant,
    evidence_level: EvidenceLevel,
) -> ReleaseCaseResult:
    source_id = str(source["id"])
    thread_id = f"v1-eval:{variant.value}:{source_id}"
    request = NaturalPlanningRequest(
        text=(
            source["text"]
            + " 预算1500元，喜欢自然和美食，灵隐寺必须去，不想太累。"
        ),
        reference_date=source["reference_date"],
    )
    fault_plan = None
    budget = None
    if variant is EvaluationVariant.LLM_FAULT:
        fault_plan = FaultPlan(
            rules=(
                FaultRule(
                    point=FaultPoint.REQUIREMENT_LLM,
                    mode=FaultMode.TIMEOUT,
                    trigger_attempt=1,
                    times=2,
                ),
            )
        )
    elif variant is EvaluationVariant.BUDGET_FAULT:
        base = runtime.run_coordinator.budget if runtime.run_coordinator else ExecutionBudget()
        budget = base.model_copy(
            update={
                "profile": "eval-budget-fault-v1",
                "max_graph_steps": 4,
                "terminal_step_reserve": 2,
            }
        )
    payload = None
    try:
        execution = await runtime.execute_plan_from_text(
            request,
            thread_id=thread_id,
            fault_plan=fault_plan,
            budget=budget,
        )
        payload = execution.payload
        run = execution.run
    except (ExecutionBudgetExceeded, RequirementUnavailableError):
        runs = await runtime.get_thread_runs(thread_id, limit=1)
        run = runs[0]
    assert run is not None
    trace = await runtime.get_agent_trace(run.run_id, limit=run.budget.max_trace_events)
    expected = _expected_behavior(source, variant, run.terminal_reason.value)
    hard_valid = _hard_valid(
        payload,
        evaluate_unvalidated=variant is EvaluationVariant.NO_VALIDATOR,
    )
    return ReleaseCaseResult(
        case_id=f"{variant.value}:{source_id}",
        source_case_id=source_id,
        category=str(source.get("category", "unknown")),
        variant=variant,
        evidence_level=evidence_level,
        run_id=run.run_id,
        run_status=run.status.value,
        terminal_reason=run.terminal_reason.value,
        expected_behavior_met=expected,
        hard_constraints_satisfied=hard_valid,
        trace_complete=trace_is_complete(trace),
        bounded=run.status is not RunStatus.RUNNING,
        graph_steps=run.usage.graph_steps,
        tool_calls=run.usage.tool_calls,
        provider_attempts=run.usage.provider_attempts,
        cache_hits=run.usage.cache_hits,
        llm_calls=run.usage.llm_calls,
        input_tokens=run.usage.input_tokens,
        output_tokens=run.usage.output_tokens,
        elapsed_ms=run.elapsed_ms or 0,
        result_fingerprint=_result_fingerprint(payload),
    )


def _expected_behavior(source: dict, variant: EvaluationVariant, terminal: str) -> bool:
    if variant is EvaluationVariant.BUDGET_FAULT:
        return terminal == "execution_budget_exhausted"
    if variant is EvaluationVariant.LLM_FAULT:
        return terminal == "llm_provider_failure"
    expected_clarification = bool(source.get("expected_needs_clarification", False))
    return (
        terminal == "needs_clarification"
        if expected_clarification
        else terminal in {"plan_completed", "business_infeasible"}
    )


def _hard_valid(
    payload: object | None, *, evaluate_unvalidated: bool = False
) -> bool | None:
    if payload is None or getattr(payload, "status", None) != "completed":
        return None
    planning = getattr(payload, "planning", None)
    selected = getattr(planning, "selected_plan", None)
    validation = getattr(selected, "validation", None)
    if evaluate_unvalidated and selected is not None:
        trip = getattr(payload, "trip", None)
        if trip is None:
            return False
        return validate_candidate(trip, selected, []).valid
    return bool(validation and validation.valid)


def _result_fingerprint(payload: object | None) -> str | None:
    if payload is None:
        return None
    if hasattr(payload, "model_dump"):
        value = payload.model_dump(mode="json")
    else:
        value = str(payload)
    value = _stable_domain_value(value)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _stable_domain_value(value: object) -> object:
    """移除线程 ID、耗时和缓存等观测字段，只比较领域结果。"""
    volatile = {
        "thread_id",
        "message",
        "elapsed_ms",
        "fetched_at",
        "created_at",
        "resolved_at",
        "expires_at",
        "cache_hit",
        "attempt_count",
        "interrupt",
        "proposal_id",
        "memory_id",
        "source_run_id",
    }
    if isinstance(value, dict):
        return {
            key: _stable_domain_value(item)
            for key, item in value.items()
            if key not in volatile
        }
    if isinstance(value, list):
        return [_stable_domain_value(item) for item in value]
    return value


def _build_report(
    dataset_version: str,
    started: datetime,
    ended: datetime,
    cases: tuple[ReleaseCaseResult, ...],
    *,
    minimum_case_count: int,
    minimum_workflow_count: int,
    provenance: ReportProvenance,
    pricing_registry: PricingRegistry | None,
) -> ReleaseEvalReport:
    workflow_count = sum(
        case.evidence_level is EvidenceLevel.WORKFLOW_EXECUTION for case in cases
    )
    completed_hard = [
        case.hard_constraints_satisfied
        for case in cases
        if case.hard_constraints_satisfied is not None
    ]
    rate = lambda values: round(sum(bool(value) for value in values) / len(values), 4) if values else 1.0
    hard_rate = rate(completed_hard)
    bounded_rate = rate([case.bounded for case in cases])
    classification_rate = rate([case.expected_behavior_met for case in cases])
    trace_rate = rate([case.trace_complete for case in cases])
    unsafe_delivery_count = sum(
        case.run_status == "completed"
        and case.terminal_reason == "plan_completed"
        and case.hard_constraints_satisfied is not True
        for case in cases
    )
    failure_misclassification_count = sum(
        case.variant in {EvaluationVariant.BUDGET_FAULT, EvaluationVariant.LLM_FAULT}
        and case.terminal_reason == "business_infeasible"
        for case in cases
    )
    total_input_tokens = _optional_sum(case.input_tokens for case in cases)
    total_output_tokens = _optional_sum(case.output_tokens for case in cases)
    total_cost = _estimate_report_cost(
        pricing_registry,
        provenance.provider_models,
        total_input_tokens,
        total_output_tokens,
    )
    gates = (
        GateResult(name="case_count", passed=len(cases) >= minimum_case_count, actual=len(cases), expected=f">={minimum_case_count}"),
        GateResult(name="workflow_execution_count", passed=workflow_count >= minimum_workflow_count, actual=workflow_count, expected=f">={minimum_workflow_count}"),
        GateResult(name="completed_hard_constraints", passed=hard_rate == 1.0, actual=hard_rate, expected=1.0),
        GateResult(name="bounded_termination", passed=bounded_rate == 1.0, actual=bounded_rate, expected=1.0),
        GateResult(name="failure_classification", passed=classification_rate == 1.0, actual=classification_rate, expected=1.0),
        GateResult(name="trace_completeness", passed=trace_rate == 1.0, actual=trace_rate, expected=1.0),
        GateResult(name="unsafe_delivery", passed=unsafe_delivery_count == 0, actual=unsafe_delivery_count, expected=0),
        GateResult(name="failure_not_infeasible", passed=failure_misclassification_count == 0, actual=failure_misclassification_count, expected=0),
        GateResult(name="report_provenance", passed=bool(provenance.reproducibility_fingerprint), actual="complete" if provenance.reproducibility_fingerprint else "missing", expected="complete"),
    )
    return ReleaseEvalReport(
        dataset_version=dataset_version,
        project_version=__version__,
        provider_profile="mock",
        started_at=started,
        ended_at=ended,
        case_count=len(cases),
        workflow_execution_count=workflow_count,
        completed_count=sum(case.run_status == "completed" for case in cases),
        interrupted_count=sum(case.run_status == "interrupted" for case in cases),
        failed_count=sum(case.run_status == "failed" for case in cases),
        completed_hard_constraint_satisfaction_rate=hard_rate,
        bounded_termination_rate=bounded_rate,
        failure_classification_accuracy=classification_rate,
        trace_completeness_rate=trace_rate,
        total_graph_steps=sum(case.graph_steps for case in cases),
        total_tool_calls=sum(case.tool_calls for case in cases),
        total_provider_attempts=sum(case.provider_attempts for case in cases),
        total_cache_hits=sum(case.cache_hits for case in cases),
        total_llm_calls=sum(case.llm_calls for case in cases),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_estimated_cost_microunits=total_cost,
        unsafe_delivery_count=unsafe_delivery_count,
        external_failure_misclassified_as_infeasible_count=failure_misclassification_count,
        provenance=provenance,
        cases=cases,
        gates=gates,
        gate_passed=all(gate.passed for gate in gates),
    )


def _optional_sum(values) -> int | None:
    items = list(values)
    return sum(items) if all(item is not None for item in items) else None


def _build_provenance(
    *,
    root: Path,
    manifest_path: Path,
    dataset_path: Path,
    settings: Settings,
    random_seed: int,
    pricing_registry: PricingRegistry | None,
) -> ReportProvenance:
    manifest_hash = _file_sha256(manifest_path)
    dataset_hash = _file_sha256(dataset_path)
    safe_config = {
        key: value
        for key, value in asdict(settings).items()
        if "api_key" not in key
    }
    config_hash = _json_sha256(safe_config)
    fixture_hash = _json_sha256(
        {"manifest": manifest_hash, "dataset": dataset_hash}
    )
    git_commit, git_dirty = _git_state(root)
    provider_models = (
        f"requirement:{settings.requirement_provider.value}/{settings.requirement_model}",
        f"critic:{settings.critic_provider.value}/{settings.critic_model}",
        f"edit:{settings.edit_provider.value}/{settings.edit_model}",
        f"travel:{settings.provider.value}",
        f"weather:{settings.weather_provider.value}",
    )
    prompt_versions = (
        "requirement-extraction-v1",
        "clarification-patch-v1",
        "soft-critic-v1",
        "plan-edit-v1",
    )
    reproducibility = _json_sha256(
        {
            "commit": git_commit,
            "manifest": manifest_hash,
            "dataset": dataset_hash,
            "config": config_hash,
            "seed": random_seed,
            "providers": provider_models,
            "prompts": prompt_versions,
        }
    )
    return ReportProvenance(
        git_commit=git_commit,
        git_dirty=git_dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        manifest_sha256=manifest_hash,
        dataset_sha256=dataset_hash,
        config_fingerprint=config_hash[:24],
        fixture_fingerprint=fixture_hash[:24],
        random_seed=random_seed,
        provider_models=provider_models,
        prompt_versions=prompt_versions,
        usage_kind=("mock_unknown" if settings.provider.value == "mock" else "live"),
        pricing_registry_version=(
            pricing_registry.version if pricing_registry is not None else None
        ),
        reproducibility_fingerprint=reproducibility[:24],
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        common = ["git", "-c", f"safe.directory={root.as_posix()}"]
        commit = subprocess.run(
            [*common, "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [*common, "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit or None, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _estimate_report_cost(
    registry: PricingRegistry | None,
    provider_models: tuple[str, ...],
    input_tokens: int | None,
    output_tokens: int | None,
) -> int | None:
    if registry is None or input_tokens is None or output_tokens is None:
        return None
    requirement = provider_models[0].split(":", 1)[1]
    provider, model = requirement.split("/", 1)
    return registry.estimate_microunits(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
