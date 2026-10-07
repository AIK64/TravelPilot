from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from datetime import date
from decimal import Decimal
import logging
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from travel_agent import __version__
from travel_agent.app import create_app
from travel_agent.config import Settings
from travel_agent.critique.gateway import CriticGateway
from travel_agent.critique.providers.mock import MockCriticModel
from travel_agent.critique.providers.deepseek import DeepSeekCriticModel
from travel_agent.edits.providers.deepseek import DeepSeekEditModel
from travel_agent.edits.providers.mock import MockEditModel
from travel_agent.edits.providers.openai import OpenAIEditModel
from travel_agent.domain.models import PlanningResponse
from travel_agent.domain.tool_models import (
    ProviderMode,
    ToolErrorCategory,
    ToolErrorInfo,
    ToolResult,
)
from travel_agent.runtime import PlanningRuntime
from travel_agent.requirements.errors import (
    RequirementErrorCategory,
    RequirementUnavailableError,
)
from travel_agent.requirements.providers.openai import OpenAIRequirementModel
from travel_agent.requirements.providers.deepseek import DeepSeekRequirementModel
from travel_agent.tools.providers.amap import AMapPOIProvider, AMapRouteProvider
from travel_agent.tools.providers.mock import MockPOIProvider, MockRouteProvider
from travel_agent.weather.errors import WeatherUnavailableError


RUNTIME_OWNED_FIELDS = {
    "poi_provider",
    "route_provider",
    "gateway",
    "workflow",
    "client",
    "agent_mode",
    "agentic_workflow",
    "planner_gateway",
    "planner_model_client",
    "tool_registry",
    "replanner_gateway",
    "replanner_model_client",
    "auxiliary_client",
    "weather_provider",
    "weather_gateway",
    "requirement_model",
    "requirement_gateway",
    "requirement_workflow",
    "model_client",
    "critic_model",
    "critic_gateway",
    "critic_model_client",
    "edit_model",
    "edit_gateway",
    "edit_model_client",
    "plan_repository",
    "lifecycle_workflow",
    "lifecycle_service",
    "checkpoint_context",
    "repository_context",
    "run_repository",
    "run_coordinator",
    "run_repository_context",
    "preference_repository",
    "preference_service",
    "preference_repository_context",
    "specialist_executor",
    "resume_locks",
}


COMPLETE_NATURAL_TEXT = (
    "2026年10月2日到10月4日去杭州，3个人，预算1500元，住西湖东侧，"
    "喜欢自然和美食，2日10:30到杭州东站，4日19:00从杭州东站离开，"
    "灵隐寺必须去，不想太累。"
)


@dataclass
class RuntimeProbe:
    plan_calls: list[tuple[object, str]]
    closed: bool = False

    async def plan(self, request, thread_id: str) -> PlanningResponse:
        self.plan_calls.append((request, thread_id))
        return PlanningResponse(
            status="infeasible",
            selected_plan=None,
            candidates=[],
            iterations=0,
        )

    async def close(self) -> None:
        self.closed = True


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_version_matches_package_version(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert __version__ == "1.3.0"
    assert response.json()["info"]["version"] == __version__


def test_create_plan(client, hangzhou_trip):
    response = client.post(
        "/api/v1/plans",
        json={
            "trip": hangzhou_trip.model_dump(mode="json"),
            "max_replan_rounds": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["selected_plan"]["validation"]["valid"] is True


def test_agent_run_header_and_trace_api_are_correlated(client, hangzhou_trip):
    response = client.post(
        "/api/v1/plans",
        json={"trip": hangzhou_trip.model_dump(mode="json")},
    )
    run_id = response.headers["x-agent-run-id"]

    run = client.get(f"/api/v1/runs/{run_id}")
    trace = client.get(f"/api/v1/runs/{run_id}/trace?limit=500")

    assert response.headers["x-agent-trace-status"] == "complete"
    assert run.status_code == 200
    assert run.json()["run_id"] == run_id
    assert run.json()["status"] == "completed"
    events = trace.json()["events"]
    assert events[0]["event_type"] == "run.started"
    assert events[-1]["event_type"] == "run.completed"
    serialized = str(events)
    assert "approval_token" not in serialized.casefold()
    assert "api_key" not in serialized.casefold()


def test_missing_agent_run_returns_safe_404(client):
    response = client.get("/api/v1/runs/missing-run")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "agent_run_not_found"


def test_create_plan_from_complete_natural_language(client):
    response = client.post(
        "/api/v1/plans/from-text",
        json={
            "text": COMPLETE_NATURAL_TEXT,
            "reference_date": "2026-08-23",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["trip"]["arrival"]["name"] == "杭州东站"
    assert body["planning"]["selected_plan"] is not None
    assert body["thread_id"]
    assert body["preference_learning_status"] == "completed"
    assert body["memory_proposals"]
    assert body["preference_extracted_count"] >= len(body["memory_proposals"])

    run_id = response.headers["x-agent-run-id"]
    trace = client.get(f"/api/v1/runs/{run_id}/trace?limit=500")
    assert trace.status_code == 200
    event_types = {event["event_type"] for event in trace.json()["events"]}
    assert "memory.extraction_started" in event_types
    assert "memory.extraction_completed" in event_types
    assert "memory.proposal_created" in event_types


def test_incomplete_natural_language_returns_structured_clarification(client):
    response = client.post(
        "/api/v1/plans/from-text",
        json={
            "text": "2026年10月2日到10月4日去杭州，2日10:30到杭州东站。",
            "reference_date": "2026-08-23",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_clarification"
    assert body["planning"] is None
    assert {issue["field"] for issue in body["issues"]} == {
        "departure.name",
        "departure.at",
    }
    assert body["clarification_questions"]


def test_natural_language_clarification_can_resume_same_thread(client):
    initial = client.post(
        "/api/v1/plans/from-text",
        json={
            "text": (
                "2026年10月2日到10月4日去杭州，3个人，预算1500元，"
                "住西湖东侧，喜欢自然和美食，2日10:30到杭州东站，"
                "灵隐寺必须去，不想太累。"
            ),
            "reference_date": "2026-08-23",
        },
    )

    assert initial.status_code == 200
    interrupted = initial.json()
    assert interrupted["status"] == "needs_clarification"
    assert interrupted["can_resume"] is True
    assert interrupted["interrupt"]["payload"]["target_fields"] == [
        "departure.name",
        "departure.at",
    ]

    resumed = client.post(
        f"/api/v1/plans/from-text/{interrupted['thread_id']}/resume",
        json={
            "interrupt_id": interrupted["interrupt"]["id"],
            "request_id": "e90bc26b-2ab0-4fe6-b733-df8f04081a14",
            "answer": "10月4日19:00从杭州东站离开。",
        },
    )

    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] == "completed"
    assert body["can_resume"] is False
    assert body["interrupt"] is None
    assert body["trip"]["departure"]["name"] == "杭州东站"


def test_resume_rejects_unknown_and_stale_threads(client):
    payload = {
        "interrupt_id": "stale-interrupt",
        "request_id": "d5130656-7be3-4c4d-91b4-51601279e21c",
        "answer": "10月4日19:00从杭州东站离开。",
    }

    missing = client.post(
        "/api/v1/plans/from-text/missing-thread/resume",
        json=payload,
    )

    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "clarification_thread_not_found"

    initial = client.post(
        "/api/v1/plans/from-text",
        json={
            "text": "2026年10月2日到10月4日去杭州，2日10:30到杭州东站。",
            "reference_date": "2026-08-23",
        },
    ).json()
    stale = client.post(
        f"/api/v1/plans/from-text/{initial['thread_id']}/resume",
        json=payload,
    )

    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "stale_interrupt"


def test_requirement_model_failure_returns_503_with_safe_detail(monkeypatch):
    class FailedNaturalRuntime:
        async def plan_from_text(self, _request, thread_id):
            raise RequirementUnavailableError(
                provider="openai",
                model="configured-model",
                category=RequirementErrorCategory.TIMEOUT,
                code="timeout",
                retryable=True,
                safe_message="需求解析服务暂时超时",
                thread_id=thread_id,
                attempt_count=2,
            )

        async def close(self):
            return None

    async def runtime_factory(_settings):
        return FailedNaturalRuntime()

    monkeypatch.setattr("travel_agent.api.routes.uuid4", lambda: "model-failed")
    application = create_app(
        Settings.from_env({}), runtime_factory=runtime_factory
    )

    with TestClient(application) as test_client:
        response = test_client.post(
            "/api/v1/plans/from-text",
            json={"text": COMPLETE_NATURAL_TEXT, "reference_date": "2026-08-23"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "timeout",
            "provider": "openai",
            "model": "configured-model",
            "category": "timeout",
            "retryable": True,
            "thread_id": "model-failed",
            "message": "需求解析服务暂时超时",
        }
    }


def test_weather_failure_returns_503_without_exposing_provider_payload():
    class LocalPlanRepository:
        async def get(self, _session_id):
            return type(
                "OwnedPlan",
                (),
                {"tenant_id": "local", "user_id": "demo"},
            )()

    class FailedWeatherRuntime:
        plan_repository = LocalPlanRepository()

        async def execute_refresh_plan_weather(
            self, _request, *, session_id, principal=None
        ):
            result = ToolResult[object].failed(
                provider="amap",
                error=ToolErrorInfo(
                    category=ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                    code="weather_upstream_unavailable",
                    operation="weather.get_forecast",
                    retryable=True,
                    safe_message="天气服务暂时不可用",
                ),
                attempt_count=3,
            )
            raise WeatherUnavailableError.from_result(result, session_id=session_id)

        async def close(self):
            return None

    async def runtime_factory(_settings):
        return FailedWeatherRuntime()

    application = create_app(
        Settings.from_env({}), runtime_factory=runtime_factory
    )
    with TestClient(application) as test_client:
        response = test_client.post(
            "/api/v1/plan-sessions/weather-failed/weather/refresh",
            json={
                "request_id": "e90bc26b-2ab0-4fe6-b733-df8f04081a14",
                "expected_session_revision": 0,
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "session_id": "weather-failed",
            "provider": "amap",
            "operation": "weather.get_forecast",
            "category": "upstream_unavailable",
            "code": "weather_upstream_unavailable",
            "retryable": True,
            "message": "天气服务暂时不可用",
        }
    }


def test_mock_api_exposes_route_provider_confidence_and_estimate_kind(
    client, hangzhou_trip
):
    """防止 API 丢失 Mock 路线 provenance，让调用方误认为是真实驾车路线。"""
    response = client.post(
        "/api/v1/plans",
        json={"trip": hangzhou_trip.model_dump(mode="json")},
    )

    assert response.status_code == 200
    facts = response.json()["selected_plan"]["reason_facts"]
    public_text = "\n".join(facts)
    assert "路线来源 mock（本地估算）" in public_text
    assert "路线置信度 65%" in public_text
    assert "真实驾车路线" not in public_text


def test_json_response_declares_utf8_for_legacy_windows_clients(
    client, hangzhou_trip
):
    response = client.post(
        "/api/v1/plans",
        json={
            "trip": hangzhou_trip.model_dump(mode="json"),
            "max_replan_rounds": 2,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].lower() == "application/json; charset=utf-8"


@pytest.mark.asyncio
async def test_mock_runtime_contains_only_mock_providers():
    runtime = await PlanningRuntime.create(Settings.from_env({}))
    try:
        assert set(runtime.__slots__) == RUNTIME_OWNED_FIELDS
        assert not hasattr(runtime, "settings")
        assert isinstance(runtime.poi_provider, MockPOIProvider)
        assert isinstance(runtime.route_provider, MockRouteProvider)
        assert isinstance(runtime.critic_model, MockCriticModel)
        assert isinstance(runtime.critic_gateway, CriticGateway)
        assert isinstance(runtime.edit_model, MockEditModel)
        assert runtime.critic_model_client is None
        assert runtime.client is None
        assert not hasattr(runtime, "fallback_provider")
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_amap_runtime_contains_only_amap_providers_and_closes_client():
    settings = Settings.from_env(
        {"TRAVEL_PROVIDER": "amap", "AMAP_API_KEY": "test-key"}
    )
    runtime = await PlanningRuntime.create(settings)
    client = runtime.client

    assert settings.provider is ProviderMode.AMAP
    assert set(runtime.__slots__) == RUNTIME_OWNED_FIELDS
    assert not hasattr(runtime, "settings")
    assert isinstance(runtime.poi_provider, AMapPOIProvider)
    assert isinstance(runtime.route_provider, AMapRouteProvider)
    assert client is not None
    assert not client.is_closed
    assert not hasattr(runtime, "fallback_provider")

    await runtime.close()

    assert client.is_closed


@pytest.mark.asyncio
async def test_deepseek_critic_runtime_is_independent_and_closes_client(monkeypatch):
    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        async def close(self):
            self.closed = True

    clients = []

    def create_client(**kwargs):
        client = FakeOpenAIClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=create_client),
    )
    runtime = await PlanningRuntime.create(
        Settings.from_env(
            {
                "CRITIC_PROVIDER": "deepseek",
                "CRITIC_MODEL": "deepseek-explicit-model",
                "DEEPSEEK_API_KEY": "secret",
            }
        )
    )
    try:
        assert isinstance(runtime.critic_model, DeepSeekCriticModel)
        assert runtime.critic_model_client is clients[0]
        assert runtime.model_client is None
        assert clients[0].kwargs["base_url"] == "https://api.deepseek.com"
        assert clients[0].kwargs["max_retries"] == 0
    finally:
        await runtime.close()
    assert clients[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,model,key_name,model_type,uses_base_url",
    [
        ("openai", "openai-edit-model", "OPENAI_API_KEY", OpenAIEditModel, False),
        ("deepseek", "deepseek-edit-model", "DEEPSEEK_API_KEY", DeepSeekEditModel, True),
    ],
)
async def test_external_edit_runtime_is_explicit_and_closes_client(
    monkeypatch, provider, model, key_name, model_type, uses_base_url
):
    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        async def close(self):
            self.closed = True

    clients = []

    def create_client(**kwargs):
        client = FakeOpenAIClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(AsyncOpenAI=create_client)
    )
    runtime = await PlanningRuntime.create(
        Settings.from_env(
            {
                "EDIT_PROVIDER": provider,
                "EDIT_MODEL": model,
                key_name: "secret-key",
            }
        )
    )
    try:
        assert isinstance(runtime.edit_model, model_type)
        assert runtime.edit_model_client is clients[0]
        assert clients[0].kwargs["max_retries"] == 0
        assert ("base_url" in clients[0].kwargs) is uses_base_url
    finally:
        await runtime.close()
    assert clients[0].closed


@pytest.mark.asyncio
async def test_openai_requirement_runtime_is_explicit_and_closes_client(monkeypatch):
    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        async def close(self):
            self.closed = True

    clients = []

    def create_client(**kwargs):
        client = FakeOpenAIClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=create_client),
    )
    settings = Settings.from_env(
        {
            "REQUIREMENT_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai-key",
            "REQUIREMENT_MODEL": "configured-model",
        }
    )

    runtime = await PlanningRuntime.create(settings)
    try:
        assert isinstance(runtime.requirement_model, OpenAIRequirementModel)
        assert runtime.requirement_model.model == "configured-model"
        assert runtime.model_client is clients[0]
        assert clients[0].kwargs["max_retries"] == 0
        assert "test-openai-key" not in repr(runtime.requirement_model)
    finally:
        await runtime.close()

    assert clients[0].closed


@pytest.mark.asyncio
async def test_deepseek_requirement_runtime_is_explicit_and_closes_client(
    monkeypatch,
):
    class FakeDeepSeekClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        async def close(self):
            self.closed = True

    clients = []

    def create_client(**kwargs):
        client = FakeDeepSeekClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=create_client),
    )
    settings = Settings.from_env(
        {
            "REQUIREMENT_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "deepseek-test-key",
            "DEEPSEEK_MODEL": "deepseek-v4-flash",
        }
    )

    runtime = await PlanningRuntime.create(settings)
    try:
        assert isinstance(runtime.requirement_model, DeepSeekRequirementModel)
        assert runtime.requirement_model.model == "deepseek-v4-flash"
        assert runtime.model_client is clients[0]
        assert clients[0].kwargs["base_url"] == "https://api.deepseek.com"
        assert clients[0].kwargs["max_retries"] == 0
        assert "deepseek-test-key" not in repr(runtime.requirement_model)
    finally:
        await runtime.close()

    assert clients[0].closed


@pytest.mark.asyncio
async def test_amap_runtime_closes_client_when_assembly_fails(monkeypatch):
    class ClientProbe:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    client = ClientProbe()

    def fail_workflow(*_args, **_kwargs):
        raise RuntimeError("workflow assembly failed")

    monkeypatch.setattr(
        "travel_agent.runtime.httpx.AsyncClient", lambda **_kwargs: client
    )
    monkeypatch.setattr("travel_agent.runtime.build_workflow", fail_workflow)
    settings = Settings.from_env(
        {"TRAVEL_PROVIDER": "amap", "AMAP_API_KEY": "test-key"}
    )

    with pytest.raises(RuntimeError, match="workflow assembly failed"):
        await PlanningRuntime.create(settings)

    assert client.closed


def test_amap_app_creation_without_key_fails(monkeypatch):
    monkeypatch.setenv("TRAVEL_PROVIDER", "amap")
    monkeypatch.delenv("AMAP_API_KEY", raising=False)

    with pytest.raises(ValueError, match="AMAP_API_KEY"):
        create_app()


def test_lifespan_stores_one_runtime_and_closes_it():
    settings = Settings.from_env({})
    runtime = RuntimeProbe(plan_calls=[])
    factory_settings = []

    async def runtime_factory(received_settings):
        factory_settings.append(received_settings)
        return runtime

    application = create_app(settings=settings, runtime_factory=runtime_factory)

    with TestClient(application):
        assert application.state.planning_runtime is runtime
        assert factory_settings == [settings]
        assert not runtime.closed

    assert runtime.closed


def test_async_endpoint_awaits_runtime_from_lifespan(monkeypatch, hangzhou_trip):
    runtime = RuntimeProbe(plan_calls=[])

    async def runtime_factory(_settings):
        return runtime

    monkeypatch.setattr("travel_agent.api.routes.uuid4", lambda: "api-runtime")
    application = create_app(
        settings=Settings.from_env({}), runtime_factory=runtime_factory
    )

    with TestClient(application) as test_client:
        response = test_client.post(
            "/api/v1/plans",
            json={"trip": hangzhou_trip.model_dump(mode="json")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "infeasible"
    assert len(runtime.plan_calls) == 1
    assert runtime.plan_calls[0][1] == "api-runtime"
    assert runtime.closed


def test_business_infeasible_still_returns_200(client, hangzhou_trip):
    low_budget_trip = hangzhou_trip.model_copy(
        update={"total_budget": Decimal("10")}
    )

    response = client.post(
        "/api/v1/plans",
        json={
            "trip": low_budget_trip.model_dump(mode="json"),
            "max_replan_rounds": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "infeasible"


def test_daily_window_api_response_is_satisfied_by_initial_optimization(
    client, hangzhou_trip
):
    """优化器在首轮满足时间窗和必去项，无需消耗修复预算。"""
    constrained = hangzhou_trip.model_copy(update={"daily_end": time(12, 0)})

    response = client.post(
        "/api/v1/plans",
        json={
            "trip": constrained.model_dump(mode="json"),
            "max_replan_rounds": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["iterations"] == 0
    assert body["selected_plan"]["validation"]["valid"] is True
    assert "灵隐寺" in {
        item["name"]
        for day in body["selected_plan"]["days"]
        for item in day["items"]
    }


def test_invalid_request_keeps_fastapi_4xx_semantics(client):
    response = client.post("/api/v1/plans", json={})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "missing"


def test_tool_failure_returns_503_with_only_safe_detail(
    tool_failure_client, hangzhou_trip
):
    response = tool_failure_client.post(
        "/api/v1/plans",
        json={"trip": hangzhou_trip.model_dump(mode="json")},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "tool_unavailable",
            "provider": "amap",
            "operation": "search_pois",
            "category": "timeout",
            "retryable": True,
            "thread_id": "api-tool-failure",
            "message": "地图服务暂时不可用，请稍后重试",
        }
    }


def test_tool_failure_logs_error_with_traceback(
    tool_failure_client, hangzhou_trip, caplog
):
    caplog.set_level(logging.ERROR, logger="travel_agent.api.errors")

    response = tool_failure_client.post(
        "/api/v1/plans",
        json={"trip": hangzhou_trip.model_dump(mode="json")},
    )

    assert response.status_code == 503
    records = [
        record
        for record in caplog.records
        if record.name == "travel_agent.api.errors"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is not None
