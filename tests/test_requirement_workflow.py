from __future__ import annotations

from datetime import date
import logging

import pytest

from travel_agent.identity.models import Principal
from travel_agent.domain.tool_models import POIFacts, POISearchQuery
from travel_agent.requirements.gateway import RequirementGateway
from travel_agent.requirements.errors import (
    RequirementErrorCategory,
    RequirementProviderError,
    RequirementUnavailableError,
)
from travel_agent.requirements.models import (
    ClarificationResumeRequest,
    NaturalPlanningRequest,
    RequirementIssueCode,
)
from travel_agent.requirements.providers.mock import MockRequirementModel
from travel_agent.requirements.workflow import (
    build_requirement_workflow,
    resume_natural_planning,
    run_natural_planning,
)
from travel_agent.tools.errors import ToolProviderError, ToolUnavailableError
from travel_agent.tools.providers.mock import MockRouteProvider
from travel_agent.memory.models import PreferenceLearningStatus
from travel_agent.memory.repository import InMemoryPreferenceRepository
from travel_agent.memory.service import PreferenceMemoryService


COMPLETE_TEXT = (
    "2026年10月2日到10月4日去杭州，3个人，预算1500元，住西湖东侧，"
    "喜欢自然和美食，2日10:30到杭州东站，4日19:00从杭州东站离开，"
    "灵隐寺必须去，不想太累。"
)


def _requirement_gateway() -> RequirementGateway:
    return RequirementGateway(
        model=MockRequirementModel(),
        timeout_seconds=1,
        max_attempts=1,
        base_delay_seconds=0,
        max_delay_seconds=0,
    )


class FailOnceClarificationModel(MockRequirementModel):
    def __init__(self) -> None:
        self.clarification_calls = 0

    async def parse_clarification(self, request):
        self.clarification_calls += 1
        if self.clarification_calls == 1:
            raise RequirementProviderError(
                category=RequirementErrorCategory.TIMEOUT,
                code="timeout",
                retryable=True,
                safe_message="需求解析服务暂时超时",
            )
        return await super().parse_clarification(request)


def _event_names(records, thread_id: str) -> list[str]:
    return [
        record.getMessage().split(maxsplit=1)[0]
        for record in records
        if f"thread_id={thread_id}" in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_complete_requirement_trajectory_reaches_existing_planning_graph(
    workflow_harness,
    caplog,
):
    thread_id = "natural-complete"
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=workflow_harness.gateway,
        planning_workflow=workflow_harness.workflow,
    )
    caplog.set_level(logging.INFO)

    response = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text=COMPLETE_TEXT,
            reference_date=date(2026, 8, 23),
        ),
        thread_id=thread_id,
    )

    assert response.status == "completed"
    assert response.trip is not None
    assert response.trip.arrival.name == "杭州东站"
    assert response.trip.arrival.coordinate.longitude == pytest.approx(120.212)
    assert response.planning is not None
    assert response.planning.selected_plan is not None
    events = _event_names(caplog.records, thread_id)
    for expected in [
        "requirement.parse.completed",
        "requirement.validated",
        "requirement.routing_decision",
        "anchors.resolved",
        "trip_spec.assembled",
        "search_plan.created",
        "candidate.validated",
        "plan.selected",
        "natural_planning.completed",
    ]:
        assert expected in events

    snapshot = await workflow.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    assert snapshot.values["requirement_draft"].destination == "杭州"
    assert snapshot.values["anchor_resolutions"]["arrival"].provider == "mock"
    assert snapshot.values["llm_summaries"][0].provider == "mock"
    assert snapshot.values["planning_response"].status == "completed"


@pytest.mark.asyncio
async def test_completed_plan_creates_confirmable_memory_proposals(
    workflow_harness,
):
    memory_service = PreferenceMemoryService(InMemoryPreferenceRepository())
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=workflow_harness.gateway,
        planning_workflow=workflow_harness.workflow,
        memory_service=memory_service,
    )

    response = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text=COMPLETE_TEXT,
            reference_date=date(2026, 8, 23),
        ),
        thread_id="natural-memory-extraction",
        tenant_id="tenant-a",
        user_id="user-a",
    )

    assert response.status == "completed"
    assert response.preference_learning_status is PreferenceLearningStatus.COMPLETED
    assert response.memory_proposals
    assert (await memory_service.list(
        Principal(tenant_id="tenant-a", user_id="user-a")
    )).items == ()


class FailingPreferenceRepository(InMemoryPreferenceRepository):
    async def find_content_hash(
        self, tenant_id: str, user_id: str, content_hash: str
    ):
        raise RuntimeError("simulated memory repository failure")


@pytest.mark.asyncio
async def test_preference_learning_failure_does_not_fail_completed_plan(
    workflow_harness,
):
    memory_service = PreferenceMemoryService(FailingPreferenceRepository())
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=workflow_harness.gateway,
        planning_workflow=workflow_harness.workflow,
        memory_service=memory_service,
    )

    response = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text=COMPLETE_TEXT,
            reference_date=date(2026, 8, 23),
        ),
        thread_id="natural-memory-degraded",
        tenant_id="tenant-a",
        user_id="user-a",
    )

    assert response.status == "completed"
    assert response.planning is not None
    assert response.preference_learning_status is PreferenceLearningStatus.DEGRADED
    assert response.memory_proposals == []


@pytest.mark.asyncio
async def test_missing_requirement_routes_to_clarification_without_tool_use(
    workflow_harness,
    caplog,
):
    thread_id = "natural-missing"
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=workflow_harness.gateway,
        planning_workflow=workflow_harness.workflow,
    )
    caplog.set_level(logging.INFO)

    response = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text=(
                "2026年10月2日到10月4日去杭州，3个人，预算1500元，"
                "住西湖东侧，喜欢自然和美食，2日10:30到杭州东站，"
                "灵隐寺必须去，不想太累。"
            ),
            reference_date=date(2026, 8, 23),
        ),
        thread_id=thread_id,
    )

    assert response.status == "needs_clarification"
    assert response.planning is None
    assert response.trip is None
    assert {issue.field for issue in response.issues} == {
        "departure.name",
        "departure.at",
    }
    assert workflow_harness.poi_provider.calls == []
    assert workflow_harness.route_provider.calls == []
    events = _event_names(caplog.records, thread_id)
    assert "requirement.clarification.required" in events
    assert "search_plan.created" not in events


class EmptyPOIProvider:
    name = "empty"

    def __init__(self) -> None:
        self.calls: list[POISearchQuery] = []

    async def search_pois(self, query: POISearchQuery) -> list[POIFacts]:
        self.calls.append(query)
        return []


class FailingPOIProvider:
    name = "failing"

    def __init__(self) -> None:
        self.calls: list[POISearchQuery] = []

    async def search_pois(self, query: POISearchQuery) -> list[POIFacts]:
        self.calls.append(query)
        raise ToolProviderError.timeout("poi.search")


class AmbiguousDeparturePOIProvider:
    name = "ambiguous-departure"

    def __init__(self) -> None:
        self.calls: list[POISearchQuery] = []

    async def search_pois(self, query: POISearchQuery) -> list[POIFacts]:
        self.calls.append(query)
        from travel_agent.tools.providers.mock import MockPOIProvider

        provider = MockPOIProvider()
        if query.keyword == "杭州火车站":
            east = await provider.search_pois(
                query.model_copy(update={"keyword": "杭州东站"})
            )
            assert east
            return [
                east[0].model_copy(
                    update={"id": "ambiguous-a", "name": "杭州火车站东广场"}
                ),
                east[0].model_copy(
                    update={"id": "ambiguous-b", "name": "杭州火车站西广场"}
                ),
            ]
        return await provider.search_pois(query)


@pytest.mark.asyncio
async def test_successful_empty_anchor_tool_result_requests_clarification(
    gateway_factory,
    workflow_harness,
):
    provider = EmptyPOIProvider()
    tool_gateway = gateway_factory(
        poi_provider=provider,
        route_provider=MockRouteProvider(),
    )
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=tool_gateway,
        planning_workflow=workflow_harness.workflow,
    )

    response = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text=COMPLETE_TEXT,
            reference_date=date(2026, 8, 23),
        ),
        thread_id="natural-anchor-empty",
    )

    assert response.status == "needs_clarification"
    assert response.planning is None
    assert provider.calls
    assert all(
        issue.code is RequirementIssueCode.NOT_FOUND for issue in response.issues
    )


@pytest.mark.asyncio
async def test_anchor_tool_failure_stays_unavailable_instead_of_infeasible(
    gateway_factory,
    workflow_harness,
):
    provider = FailingPOIProvider()
    tool_gateway = gateway_factory(
        poi_provider=provider,
        route_provider=MockRouteProvider(),
    )
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=tool_gateway,
        planning_workflow=workflow_harness.workflow,
    )

    with pytest.raises(ToolUnavailableError) as caught:
        await run_natural_planning(
            workflow,
            NaturalPlanningRequest(
                text=COMPLETE_TEXT,
                reference_date=date(2026, 8, 23),
            ),
            thread_id="natural-anchor-failed",
        )

    assert caught.value.result.error is not None
    assert caught.value.result.error.category.value == "timeout"
    assert provider.calls
    assert workflow_harness.route_provider.calls == []


def test_requirement_graph_exposes_intake_and_anchor_routing_nodes(
    workflow_harness,
):
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=workflow_harness.gateway,
        planning_workflow=workflow_harness.workflow,
    )

    assert {
        "parse_requirement",
        "validate_requirement",
        "resolve_anchors",
        "evaluate_anchors",
        "assemble_trip_spec",
        "execute_planning",
        "request_clarification",
        "await_clarification",
        "parse_clarification_patch",
        "apply_clarification_patch",
        "clarification_exhausted",
    } <= set(workflow.get_graph().nodes)


@pytest.mark.asyncio
async def test_missing_requirement_interrupts_and_resumes_same_thread(
    workflow_harness,
):
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=workflow_harness.gateway,
        planning_workflow=workflow_harness.workflow,
    )
    thread_id = "natural-resume"

    interrupted = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text=(
                "2026年10月2日到10月4日去杭州，3个人，预算1500元，"
                "住西湖东侧，喜欢自然和美食，2日10:30到杭州东站，"
                "灵隐寺必须去，不想太累。"
            ),
            reference_date=date(2026, 8, 23),
        ),
        thread_id=thread_id,
    )

    assert interrupted.status == "needs_clarification"
    assert interrupted.can_resume is True
    assert interrupted.interrupt is not None
    assert interrupted.interrupt.payload.target_fields == [
        "departure.name",
        "departure.at",
    ]
    assert workflow_harness.poi_provider.calls == []

    completed = await resume_natural_planning(
        workflow,
        ClarificationResumeRequest(
            interrupt_id=interrupted.interrupt.id,
            request_id="3eed20d6-9abc-4c11-9fea-ec5f3f4bc4fe",
            answer="10月4日19:00从杭州东站离开。",
        ),
        thread_id=thread_id,
    )

    assert completed.status == "completed"
    assert completed.can_resume is False
    assert completed.interrupt is None
    assert completed.trip is not None
    assert completed.trip.departure.name == "杭州东站"
    snapshot = await workflow.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    assert [summary.operation.value for summary in snapshot.values["llm_summaries"]] == [
        "initial_parse",
        "clarification_patch",
    ]


@pytest.mark.asyncio
async def test_clarification_loop_stops_at_explicit_budget(workflow_harness):
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=workflow_harness.gateway,
        planning_workflow=workflow_harness.workflow,
    )
    thread_id = "natural-resume-budget"
    response = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text="2026年10月2日到10月4日去杭州，2日10:30到杭州东站。",
            reference_date=date(2026, 8, 23),
            max_clarification_rounds=2,
        ),
        thread_id=thread_id,
    )

    for request_id in (
        "a59924b8-305d-4e3f-b88a-c36b015283d2",
        "8fbf0604-8c84-49ce-b898-4692a23f40fd",
    ):
        assert response.interrupt is not None
        response = await resume_natural_planning(
            workflow,
            ClarificationResumeRequest(
                interrupt_id=response.interrupt.id,
                request_id=request_id,
                answer="我还不确定。",
            ),
            thread_id=thread_id,
        )

    assert response.status == "needs_clarification"
    assert response.clarification_round == 2
    assert response.can_resume is False
    assert response.interrupt is None
    assert workflow_harness.poi_provider.calls == []


@pytest.mark.asyncio
async def test_anchor_clarification_reuses_unaffected_resolutions(
    gateway_factory,
    workflow_harness,
):
    provider = AmbiguousDeparturePOIProvider()
    tool_gateway = gateway_factory(
        poi_provider=provider,
        route_provider=MockRouteProvider(),
    )
    workflow = build_requirement_workflow(
        requirement_gateway=_requirement_gateway(),
        tool_gateway=tool_gateway,
        planning_workflow=workflow_harness.workflow,
    )
    thread_id = "natural-anchor-resume"

    interrupted = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text=(
                "2026年10月2日到10月4日去杭州，3个人，预算1500元，"
                "住西湖东侧，喜欢自然和美食，2日10:30到杭州东站，"
                "4日19:00从杭州火车站离开，灵隐寺必须去，不想太累。"
            ),
            reference_date=date(2026, 8, 23),
        ),
        thread_id=thread_id,
    )

    assert interrupted.interrupt is not None
    assert interrupted.interrupt.payload.target_fields == ["departure.name"]
    first_snapshot = await workflow.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    assert set(first_snapshot.values["anchor_resolutions"]) == {
        "arrival",
        "accommodation",
    }

    completed = await resume_natural_planning(
        workflow,
        ClarificationResumeRequest(
            interrupt_id=interrupted.interrupt.id,
            request_id="c96f9f31-97bd-4b71-a7d3-35217c62e658",
            answer="10月4日19:00从杭州东站离开。",
        ),
        thread_id=thread_id,
    )

    assert completed.status == "completed"
    final_snapshot = await workflow.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    assert final_snapshot.values["reused_anchor_roles"] == [
        "accommodation",
        "arrival",
    ]
    assert len(final_snapshot.values["anchor_search_plan"]) == 1
    assert final_snapshot.values["anchor_search_plan"][0].roles == ["departure"]


@pytest.mark.asyncio
async def test_same_resume_request_retries_failed_patch_node(workflow_harness):
    model = FailOnceClarificationModel()
    workflow = build_requirement_workflow(
        requirement_gateway=RequirementGateway(
            model=model,
            timeout_seconds=1,
            max_attempts=1,
            base_delay_seconds=0,
            max_delay_seconds=0,
        ),
        tool_gateway=workflow_harness.gateway,
        planning_workflow=workflow_harness.workflow,
    )
    thread_id = "natural-resume-provider-retry"
    interrupted = await run_natural_planning(
        workflow,
        NaturalPlanningRequest(
            text=(
                "2026年10月2日到10月4日去杭州，3个人，预算1500元，"
                "住西湖东侧，喜欢自然和美食，2日10:30到杭州东站，"
                "灵隐寺必须去，不想太累。"
            ),
            reference_date=date(2026, 8, 23),
        ),
        thread_id=thread_id,
    )
    assert interrupted.interrupt is not None
    resume_request = ClarificationResumeRequest(
        interrupt_id=interrupted.interrupt.id,
        request_id="5721a98e-7d40-4668-8d5c-2b414636bb04",
        answer="10月4日19:00从杭州东站离开。",
    )

    with pytest.raises(RequirementUnavailableError):
        await resume_natural_planning(
            workflow,
            resume_request,
            thread_id=thread_id,
        )

    completed = await resume_natural_planning(
        workflow,
        resume_request,
        thread_id=thread_id,
    )

    assert completed.status == "completed"
    assert model.clarification_calls == 2
