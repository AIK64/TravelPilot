from __future__ import annotations

from pathlib import Path

import pytest

from travel_agent.planning.repair_evaluation import (
    evaluate_repair_cases,
    load_repair_cases,
    repair_report_as_json,
)


DATASET = Path("evals/repairs/cases.jsonl")


def test_repair_benchmark_dataset_covers_core_routes():
    cases = load_repair_cases(DATASET)

    assert len(cases) >= 8
    assert {case.category for case in cases} >= {
        "no_repair_needed",
        "budget_repair",
        "hard_conflict",
        "missing_must_visit",
        "missing_poi_facts",
        "budget_exhausted",
        "activity_limit",
        "walking_limit",
    }


@pytest.mark.asyncio
async def test_mock_repair_benchmark_is_reproducible(
    hangzhou_trip,
    fallback_workflow,
):
    cases = load_repair_cases(DATASET)

    report = await evaluate_repair_cases(fallback_workflow, hangzhou_trip, cases)

    assert report.case_count == len(cases)
    assert report.execution_failure_count == 0
    assert report.exact_case_accuracy == 1.0, [
        (
            case.id,
            case.actual_status,
            case.actual_rounds,
            [kind.value for kind in case.actual_action_kinds],
            case.terminal_reason,
        )
        for case in report.cases
        if not case.exact
    ]
    # 完整计入日终返程后，两类约束确实不可通过删除活动安全修复。
    assert report.repair_success_rate == 0.75
    assert report.hard_constraint_satisfaction_rate == 1.0
    assert report.bounded_termination_rate == 1.0
    assert report.replanning_locality == 1.0
    assert report.route_reuse_rate > 0.5
    assert '"exact_case_accuracy": 1.0' in repair_report_as_json(report)
