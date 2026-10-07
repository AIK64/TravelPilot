from datetime import datetime, timezone

import pytest

from travel_agent.agents.actions import AgentPhase, CallToolAction, PlannerDecision
from travel_agent.agents.planner.providers.mock import MockPlannerModel
from travel_agent.config import Settings
from travel_agent.domain.tool_models import POIFacts, ToolResult
from travel_agent.execution.models import RunStatus, RunTerminalReason
from travel_agent.memory.models import AgentRole
from travel_agent.planning.search_plan import build_search_plan, select_search_candidates
from travel_agent.requirements.models import NaturalPlanningRequest
from travel_agent.runtime import PlanningRuntime
from travel_agent.tools.agent_executor import AgentToolExecutor, POISearchToolInput, TOOL_INPUT_MODELS
from travel_agent.tools.registry import default_agent_tool_registry


def fact(poi_id, name):
    return POIFacts(id=poi_id, name=name, city="泉州", coordinate={"longitude": 118.6, "latitude": 24.9},
                    categories=["人文"], provider="fixture", fetched_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test_must_visit_searches_finish_before_ordinary_and_keep_one_each(hangzhou_trip):
    trip = hangzhou_trip.model_copy(update={"destination": "泉州", "must_visit": ["大开元寺", "清净寺"],
                                          "interests": ["自然", "人文"]})

    class Gateway:
        def __init__(self):
            self.calls = []

        async def search_pois(self, queries, context):
            self.calls.append([query.keyword for query in queries])
            assert all(query.city == "泉州" for query in queries)
            assert all(query.types == "110000" for query in queries if not query.exact_match)
            results = []
            for query in queries:
                if query.exact_match:
                    data = [fact(query.keyword, query.keyword), fact("extra-" + query.keyword, query.keyword + "周边")]
                else:
                    data = [fact("alias", "大开元寺"), fact("scenic", "清源山")]
                results.append(ToolResult.success(data=data, provider="fixture"))
            return results

    gateway = Gateway()
    executor = AgentToolExecutor(registry=default_agent_tool_registry(), gateway=gateway)
    action = CallToolAction(reason_code="poi_evidence_missing", decision_summary="搜索候选",
                            tool_name="poi.search", arguments={"city": "泉州", "keywords": ["自然", "人文"]},
                            evidence_goal="获取 POI")
    result = await executor.execute(action, trip=trip, thread_id="search-order", run_id="search-order")
    # 即使旧 Planner 仍传自然、人文，实际通用查询也必须是景点 + 110000。
    assert gateway.calls == [["大开元寺"], ["清净寺"], ["景点"]]
    assert [poi.id for poi in result.poi_facts] == ["大开元寺", "清净寺", "scenic"]
    assert result.evidence


def test_required_first_result_is_checked_and_required_slots_are_reserved(hangzhou_trip, caplog):
    trip = hangzhou_trip.model_copy(update={"must_visit": ["灵隐寺", "西湖"], "interests": ["自然"]})
    queries = build_search_plan(trip)
    results = [ToolResult.success(provider="fixture", data=[fact("temple", "灵隐寺"), fact("unused", "灵隐寺周边")]),
               ToolResult.success(provider="fixture", data=[fact("lake", "西湖")]),
               ToolResult.success(provider="fixture", data=[fact("ordinary", "清源山")])]
    assert [poi.id for poi in select_search_candidates(queries, results, max_candidates=1)] == ["temple", "lake"]
    wrong = [ToolResult.success(provider="fixture", data=[fact("wrong", "不相关地点")])]
    assert select_search_candidates(queries[:1], wrong, max_candidates=12) == []
    assert "poi.must_visit_not_found" in caplog.text
    with pytest.raises(ValueError, match="must_visit"):
        build_search_plan(trip, max_queries=1)


def test_planner_manifest_contains_executable_schema_and_example():
    tools = default_agent_tool_registry().manifest(role=AgentRole.PLANNER, phase=AgentPhase.RESEARCHING)
    search = next(tool for tool in tools if tool.name == "poi.search")
    assert search.input_schema == POISearchToolInput.model_json_schema()
    assert search.input_schema["required"] == ["city"]
    assert search.input_schema["additionalProperties"] is False
    assert set(search.input_example) == {"city", "keywords", "limit"}
    for tool in tools:
        if tool.input_example:
            TOOL_INPUT_MODELS[tool.name].model_validate(tool.input_example)


TEXT = "2026年10月2日到10月4日去杭州，3个人，预算1500元，住西湖东侧，喜欢自然和人文，2日10:30到杭州东站，4日19:00从杭州东站离开，灵隐寺必须去，不想太累。"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_arguments", [True, False])
async def test_bad_planner_arguments_receive_feedback_and_failure_does_not_crash(repair_arguments):
    class Planner(MockPlannerModel):
        calls = 0
        feedback_seen = False

        async def decide(self, context):
            self.calls += 1
            if self.calls > 1:
                observation = context.recent_observations[-1]
                if observation.error_code == "invalid_tool_arguments":
                    assert "city: missing" in observation.summary
                    assert "destination: extra_forbidden" in observation.summary
                    self.feedback_seen = True
            if self.calls == 1 or not repair_arguments:
                return PlannerDecision(confidence=0.9, action=CallToolAction(
                    reason_code="poi_evidence_missing", decision_summary="查询候选",
                    tool_name="poi.search", arguments={"destination": "杭州", "interests": ["自然"]},
                    evidence_goal="获取 POI"))
            return await super().decide(context)

    runtime = await PlanningRuntime.create(Settings.from_env({"TRACE_FILE_ENABLED": "false"}))
    planner = Planner()
    runtime.planner_gateway.model = planner
    try:
        result = await runtime.execute_plan_from_text(NaturalPlanningRequest(text=TEXT), thread_id=f"required-search-{repair_arguments}")
        assert planner.feedback_seen
        if repair_arguments:
            assert result.payload.status == "completed"
            assert result.payload.planning.selected_plan.validation.valid
        else:
            assert result.payload.status == "failed"
            assert result.run.status is RunStatus.FAILED
            assert result.run.terminal_reason is RunTerminalReason.AGENT_NO_PROGRESS
            assert (await runtime.run_repository.trace(result.run.run_id, limit=500))[-1].event_type.value == "run.failed"
    finally:
        await runtime.close()
