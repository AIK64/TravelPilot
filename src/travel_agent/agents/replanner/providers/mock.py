from __future__ import annotations

from travel_agent.agents.context import ReplannerContext
from travel_agent.agents.replanner.models import ModelRepairProposal, ReplannerOutput
from travel_agent.agents.replanner.prompts import REPLANNER_PROMPT_VERSION
from travel_agent.domain.repair_models import RepairAction, RepairActionKind


class MockReplannerModel:
    name = "mock"
    model = "mock-model-replanner-v1"
    prompt_version = REPLANNER_PROMPT_VERSION

    async def propose(self, context: ReplannerContext) -> ReplannerOutput:
        affected_poi = next(iter(context.critic_report.affected_poi_ids), None)
        affected_day = next(iter(context.critic_report.affected_days), None)
        if affected_poi is None or affected_day is None:
            raise ValueError("mock replanner requires an affected POI and day")
        violation = context.critic_report.violation_fingerprint
        proposal = ModelRepairProposal(
            target_candidate_id=context.candidate.id,
            source_violation_fingerprints=(violation,),
            actions=(
                RepairAction(
                    kind=RepairActionKind.REMOVE_OPTIONAL_POI,
                    source_violation_type=next(iter(context.critic_report.violation_types), "hard_violation"),
                    poi_id=affected_poi,
                    from_day=affected_day,
                    reason="移除导致硬约束违规的可选景点",
                    expected_effect="降低当日时长、费用或步行负担",
                ),
            ),
            affected_days=(affected_day,),
            expected_effect_codes=("hard_error_reduced",),
            summary="对违规日期执行最小范围局部修复",
        )
        return ReplannerOutput(proposal=proposal)
