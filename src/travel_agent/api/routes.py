from __future__ import annotations

from typing import Annotated
import inspect
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import APIRouter, Depends, Query, Response

from travel_agent import __version__
from travel_agent.api.dependencies import (
    get_application_service,
    get_principal,
    get_runtime,
)
from travel_agent.application.service import TravelApplicationService
from travel_agent.application.errors import ApplicationConflictError
from travel_agent.identity.models import Principal
from travel_agent.domain.models import PlanningRequest, PlanningResponse
from travel_agent.domain.lifecycle_models import (
    LifecycleResumeRequest,
    PlanDiff,
    PlanSessionResponse,
    PlanVersion,
)
from travel_agent.runtime import PlanningRuntime
from travel_agent.execution.models import AgentRunRecord, RunKind, RunStatus, TracePage
from travel_agent.domain.weather_models import (
    WeatherEventView,
    WeatherRefreshRequest,
    WeatherStateView,
)
from travel_agent.requirements.models import (
    ClarificationResumeRequest,
    NaturalPlanningRequest,
    NaturalPlanningResponse,
)


router = APIRouter()


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.post("/api/v1/plans", response_model=PlanningResponse, tags=["planning"])
async def create_plan(
    request: PlanningRequest,
    response: Response,
    runtime: Annotated[PlanningRuntime, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> PlanningResponse:
    thread_id = str(uuid4())
    return await _execute_with_run(
        runtime,
        "execute_plan",
        "plan",
        response,
        request,
        thread_id=thread_id,
        principal=principal,
    )


@router.post(
    "/api/v1/plans/from-text",
    response_model=NaturalPlanningResponse,
    tags=["planning"],
)
async def create_plan_from_text(
    request: NaturalPlanningRequest,
    response: Response,
    runtime: Annotated[PlanningRuntime, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> NaturalPlanningResponse:
    thread_id = str(uuid4())
    return await _execute_with_run(
        runtime,
        "execute_plan_from_text",
        "plan_from_text",
        response,
        request,
        thread_id=thread_id,
        principal=principal,
    )


@router.post(
    "/api/v1/plans/from-text/{thread_id}/resume",
    response_model=NaturalPlanningResponse,
    tags=["planning"],
)
async def resume_plan_from_text(
    thread_id: str,
    request: ClarificationResumeRequest,
    response: Response,
    runtime: Annotated[PlanningRuntime, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> NaturalPlanningResponse:
    return await _execute_with_run(
        runtime,
        "execute_resume_from_text",
        "resume_from_text",
        response,
        request,
        thread_id=thread_id,
        principal=principal,
    )


@router.post(
    "/api/v1/plan-sessions",
    response_model=PlanSessionResponse,
    tags=["plan-lifecycle"],
)
async def create_plan_session(
    request: PlanningRequest,
    response: Response,
    runtime: Annotated[PlanningRuntime, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> PlanSessionResponse:
    session_id = str(uuid4())
    return await _execute_with_run(
        runtime,
        "execute_create_plan_session",
        "create_plan_session",
        response,
        request,
        session_id=session_id,
        principal=principal,
    )


@router.post(
    "/api/v1/plan-sessions/from-text",
    response_model=PlanSessionResponse,
    tags=["plan-lifecycle"],
)
async def create_plan_session_from_text(
    request: NaturalPlanningRequest,
    response: Response,
    runtime: Annotated[PlanningRuntime, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> PlanSessionResponse:
    return await _execute_with_run(
        runtime,
        "execute_create_plan_session_from_text",
        "create_plan_session_from_text",
        response,
        request,
        session_id=str(uuid4()),
        principal=principal,
    )


@router.post(
    "/api/v1/plan-sessions/{session_id}/resume",
    response_model=PlanSessionResponse,
    tags=["plan-lifecycle"],
)
async def resume_plan_session(
    session_id: str,
    request: LifecycleResumeRequest,
    response: Response,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> PlanSessionResponse:
    result = await service.execute_resume_plan_session(
        session_id, request, principal=principal
    )
    _apply_run_headers(response, result)
    return result.payload


@router.post(
    "/api/v1/runs/{run_id}/plan-session",
    response_model=PlanSessionResponse,
    tags=["plan-lifecycle"],
)
async def create_selection_session_from_run(
    run_id: str,
    response: Response,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> PlanSessionResponse:
    """复用当前用户已完成的规划候选；重复请求返回同一选择会话。"""
    record = await service.get_run(run_id, principal=principal)
    if (
        record.run_kind not in {RunKind.NATURAL_PLAN, RunKind.CLARIFICATION_RESUME}
        or record.status is not RunStatus.COMPLETED
        or record.thread_id is None
    ):
        raise ApplicationConflictError("该运行没有可供确认的已完成自然语言规划")
    session_id = str(uuid5(NAMESPACE_URL, f"travel-agent:selection:{record.thread_id}"))
    result = await service.runtime.execute_create_plan_session_from_checkpoint(
        thread_id=record.thread_id,
        session_id=session_id,
        source_run_id=run_id,
        principal=principal,
    )
    _apply_run_headers(response, result)
    return result.payload


@router.get(
    "/api/v1/plan-sessions/{session_id}",
    response_model=PlanSessionResponse,
    tags=["plan-lifecycle"],
)
async def get_plan_session(
    session_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> PlanSessionResponse:
    return await service.get_plan_session(session_id, principal=principal)


@router.get(
    "/api/v1/plan-sessions/{session_id}/versions",
    response_model=list[PlanVersion],
    tags=["plan-lifecycle"],
)
async def get_plan_versions(
    session_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
):
    return await service.get_plan_versions(session_id, principal=principal)


@router.get(
    "/api/v1/plan-sessions/{session_id}/versions/{version_id}",
    response_model=PlanVersion,
    tags=["plan-lifecycle"],
)
async def get_plan_version(
    session_id: str,
    version_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
):
    return await service.get_plan_version(
        session_id, version_id, principal=principal
    )


@router.get(
    "/api/v1/plan-sessions/{session_id}/diff",
    response_model=PlanDiff,
    tags=["plan-lifecycle"],
)
async def get_plan_diff(
    session_id: str,
    from_id: str,
    to_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
):
    return await service.get_plan_diff(
        session_id, from_id, to_id, principal=principal
    )


@router.post(
    "/api/v1/plan-sessions/{session_id}/weather/refresh",
    response_model=PlanSessionResponse,
    tags=["weather-lifecycle"],
)
async def refresh_plan_weather(
    session_id: str,
    request: WeatherRefreshRequest,
    response: Response,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> PlanSessionResponse:
    result = await service.refresh_plan_weather(
        session_id, request, principal=principal
    )
    _apply_run_headers(response, result)
    return result.payload


@router.get(
    "/api/v1/plan-sessions/{session_id}/weather",
    response_model=WeatherStateView,
    tags=["weather-lifecycle"],
)
async def get_plan_weather(
    session_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> WeatherStateView:
    return await service.get_plan_weather(session_id, principal=principal)


@router.get(
    "/api/v1/plan-sessions/{session_id}/weather/events",
    response_model=list[WeatherEventView],
    tags=["weather-lifecycle"],
)
async def get_plan_weather_events(
    session_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
):
    return await service.get_plan_weather_events(session_id, principal=principal)


@router.get(
    "/api/v1/plan-sessions/{session_id}/weather/events/{event_id}",
    response_model=WeatherEventView,
    tags=["weather-lifecycle"],
)
async def get_plan_weather_event(
    session_id: str,
    event_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
):
    return await service.get_plan_weather_event(
        session_id, event_id, principal=principal
    )


@router.get(
    "/api/v1/runs/{run_id}",
    response_model=AgentRunRecord,
    tags=["agent-runs"],
)
async def get_agent_run(
    run_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> AgentRunRecord:
    return await service.get_run(run_id, principal=principal)


@router.get(
    "/api/v1/runs/{run_id}/trace",
    response_model=TracePage,
    tags=["agent-runs"],
)
async def get_agent_trace(
    run_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> TracePage:
    events = await service.get_trace(
        run_id,
        principal=principal,
        after_sequence=after_sequence,
        limit=limit,
    )
    return TracePage(
        run_id=run_id,
        events=events,
        next_sequence=events[-1].sequence if len(events) == limit else None,
    )


@router.get(
    "/api/v1/plan-sessions/{session_id}/runs",
    response_model=list[AgentRunRecord],
    tags=["agent-runs"],
)
async def get_session_runs(
    session_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
    limit: int = Query(default=50, ge=1, le=100),
):
    return await service.get_session_runs(
        session_id, principal=principal, limit=limit
    )


@router.get(
    "/api/v1/requirement-threads/{thread_id}/runs",
    response_model=list[AgentRunRecord],
    tags=["agent-runs"],
)
async def get_thread_runs(
    thread_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
    limit: int = Query(default=50, ge=1, le=100),
):
    return await service.get_thread_runs(
        thread_id, principal=principal, limit=limit
    )


async def _execute_with_run(
    runtime: object,
    execute_method: str,
    legacy_method: str,
    response: Response,
    *args,
    **kwargs,
):
    execute = getattr(runtime, execute_method, None)
    target = execute or getattr(runtime, legacy_method)
    parameters = inspect.signature(target).parameters.values()
    accepts_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters
    )
    if not accepts_kwargs:
        accepted = {item.name for item in parameters}
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    if execute is None:
        return await target(*args, **kwargs)
    result = await target(*args, **kwargs)
    _apply_run_headers(response, result)
    return result.payload


def _apply_run_headers(response: Response, result: object) -> None:
    run = getattr(result, "run", None)
    if run is not None:
        response.headers["X-Agent-Run-Id"] = run.run_id
        response.headers["X-Agent-Trace-Status"] = run.trace_status.value

