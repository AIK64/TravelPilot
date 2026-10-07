from __future__ import annotations

from travel_agent.agents.actions import (
    CallToolAction,
    ClarifyAction,
    EscalateAction,
    FinishAction,
    PlannerDecision,
    RepairAction,
    SolveAction,
)
from travel_agent.agents.context import DynamicPlannerContext
from travel_agent.agents.planner.prompts import PLANNER_PROMPT_VERSION
from travel_agent.evidence.models import EvidenceGapStatus


class MockPlannerModel:
    name = "mock"
    model = "mock-dynamic-planner-v1"
    prompt_version = PLANNER_PROMPT_VERSION

    async def decide(self, context: DynamicPlannerContext) -> PlannerDecision:
        user_gap = next(
            (
                gap
                for gap in context.evidence_gaps
                if gap.required and gap.status is EvidenceGapStatus.USER_REQUIRED
            ),
            None,
        )
        if user_gap is not None:
            action = ClarifyAction(
                reason_code=user_gap.reason_code,
                decision_summary="缺少只能由用户确认的规划信息",
                fields=(user_gap.key,),
                question="请确认多日行程每天往返的住宿区域或住宿地点。",
            )
        else:
            open_keys = {
                gap.key
                for gap in context.evidence_gaps
                if gap.required and gap.status is EvidenceGapStatus.OPEN
            }
            if "poi.candidates.interests" in open_keys:
                action = CallToolAction(
                    reason_code="poi_evidence_missing",
                    decision_summary="需要先获取兴趣点候选",
                    tool_name="poi.search",
                    arguments={
                        "city": context.goal.destination,
                        "keywords": ("景点",),
                    },
                    evidence_goal="补齐 POI 候选证据",
                )
            elif "route.matrix.required" in open_keys:
                action = CallToolAction(
                    reason_code="route_evidence_missing",
                    decision_summary="POI 已具备，需要构建路线矩阵",
                    tool_name="route.build_matrix",
                    arguments={"mode": "driving", "strategy": 32},
                    evidence_goal="补齐候选点路线矩阵",
                    evidence_ids=tuple(
                        item.evidence_id
                        for item in context.evidence_catalog
                        if item.kind.value == "poi"
                    ),
                )
            elif not context.candidate_summaries:
                action = SolveAction(
                    reason_code="evidence_ready",
                    decision_summary="规划证据已满足，可以执行约束求解",
                    evidence_ids=tuple(item.evidence_id for item in context.evidence_catalog),
                    strategy="auto",
                )
            else:
                valid = next(
                    (item for item in context.candidate_summaries if item.hard_valid),
                    None,
                )
                if valid is not None:
                    action = FinishAction(
                        reason_code="hard_validation_passed",
                        decision_summary="候选已经通过硬约束校验",
                        candidate_id=valid.candidate_id,
                        evidence_ids=tuple(
                            item.evidence_id
                            for item in context.evidence_catalog
                            if item.kind.value == "validation"
                        ),
                    )
                elif context.violation_summaries:
                    target = context.violation_summaries[0]
                    action = RepairAction(
                        reason_code="hard_violation_detected",
                        decision_summary="候选存在硬约束违规，需要受控修复",
                        target_candidate_id=target.candidate_id,
                        target_violation_fingerprints=tuple(
                            item.fingerprint for item in context.violation_summaries
                        ),
                    )
                else:
                    action = EscalateAction(
                        reason_code="evidence_unavailable",
                        decision_summary="缺少可继续执行的证据和候选",
                        issue_code="evidence_unavailable",
                        question="当前证据不足，是否允许调整规划约束？",
                    )
        return PlannerDecision(action=action, confidence=1.0)
