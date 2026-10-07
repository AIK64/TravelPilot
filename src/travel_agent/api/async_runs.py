from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from travel_agent.api.dependencies import (
    get_application_service,
    get_principal,
    get_runtime,
)
from travel_agent.application.models import RunHandle, TripRecord
from travel_agent.application.service import TravelApplicationService
from travel_agent.domain.models import PlanningRequest
from travel_agent.execution.models import AgentRunRecord, RunKind, RunStatus
from travel_agent.identity.models import Principal
from travel_agent.requirements.models import (
    ClarificationResumeRequest,
    NaturalPlanningRequest,
)
from travel_agent.runtime import PlanningRuntime


router = APIRouter(prefix="/api/v1", tags=["async-agent-runs"])
logger = logging.getLogger(__name__)
_TERMINAL = {
    RunStatus.COMPLETED,
    RunStatus.INTERRUPTED,
    RunStatus.REPLAYED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}


@router.post("/trips", response_model=TripRecord, status_code=status.HTTP_201_CREATED)
async def create_trip(
    request: PlanningRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> TripRecord:
    return await service.create_trip(request, principal=principal)


@router.get("/trips/{trip_id}", response_model=TripRecord)
async def get_trip(
    trip_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> TripRecord:
    return await service.get_trip(trip_id, principal=principal)


@router.post(
    "/trips/{trip_id}/runs",
    response_model=RunHandle,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_trip_run(
    trip_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
    request_id: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> RunHandle:
    return await service.start_trip_run(
        trip_id, principal=principal, request_id=request_id
    )


@router.post("/plans/from-text/stream")
async def stream_plan_from_text(
    payload: NaturalPlanningRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    runtime: Annotated[PlanningRuntime, Depends(get_runtime)],
) -> StreamingResponse:
    """立即返回 SSE，在自然语言规划执行期间逐条推送 Trace。"""
    run_id = str(uuid4())
    thread_id = str(uuid4())
    await runtime.reserve_run(
        RunKind.NATURAL_PLAN,
        run_id=run_id,
        thread_id=thread_id,
        principal=principal,
    )

    async def execute():
        return await runtime.execute_plan_from_text(
            payload,
            thread_id=thread_id,
            principal=principal,
            run_id=run_id,
            precreated=True,
        )

    return _execution_stream(
        runtime,
        execute,
        run_id=run_id,
        thread_id=thread_id,
    )


@router.post("/plans/from-text/{thread_id}/resume/stream")
async def stream_resume_plan_from_text(
    thread_id: str,
    payload: ClarificationResumeRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    runtime: Annotated[PlanningRuntime, Depends(get_runtime)],
) -> StreamingResponse:
    """以同一协议实时推送 clarification resume 的 Trace 与结果。"""
    run_id = str(uuid4())
    await runtime.reserve_run(
        RunKind.CLARIFICATION_RESUME,
        run_id=run_id,
        thread_id=thread_id,
        principal=principal,
        request_id=str(payload.request_id),
        causation_id=payload.interrupt_id,
    )

    async def execute():
        return await runtime.execute_resume_from_text(
            payload,
            thread_id=thread_id,
            principal=principal,
            run_id=run_id,
            precreated=True,
        )

    return _execution_stream(
        runtime,
        execute,
        run_id=run_id,
        thread_id=thread_id,
    )


@router.post("/runs/{run_id}/cancel", response_model=AgentRunRecord)
async def cancel_run(
    run_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
) -> AgentRunRecord:
    return await service.cancel_run(run_id, principal=principal)


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
    service: Annotated[TravelApplicationService, Depends(get_application_service)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    after_sequence: int = Query(default=0, ge=0),
) -> StreamingResponse:
    cursor = after_sequence
    if last_event_id and last_event_id.isdigit():
        cursor = max(cursor, int(last_event_id))

    async def events():
        nonlocal cursor
        while True:
            if await request.is_disconnected():
                return
            values = await service.get_trace(
                run_id,
                principal=principal,
                after_sequence=cursor,
                limit=200,
            )
            for event in values:
                cursor = event.sequence
                payload = json.dumps(
                    event.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                yield f"id: {event.sequence}\nevent: trace\ndata: {payload}\n\n"
            record = await service.get_run(run_id, principal=principal)
            if record.status in _TERMINAL and not values:
                yield "event: end\ndata: {}\n\n"
                return
            if not values:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _execution_stream(
    runtime: PlanningRuntime,
    execute: Callable[[], Awaitable[object]],
    *,
    run_id: str,
    thread_id: str,
) -> StreamingResponse:
    async def events():
        cursor = 0
        task = asyncio.create_task(execute(), name=f"streaming-run:{run_id}")
        try:
            while True:
                values = await runtime.get_agent_trace(
                    run_id, after_sequence=cursor, limit=200
                )
                for event in values:
                    cursor = event.sequence
                    yield _sse_frame(
                        "trace",
                        event.model_dump(mode="json"),
                        event_id=str(event.sequence),
                    )
                if task.done() and not values:
                    try:
                        result = task.result()
                        payload = getattr(result, "payload", result)
                        body = (
                            payload.model_dump(mode="json")
                            if hasattr(payload, "model_dump")
                            else payload
                        )
                        yield _sse_frame("result", body)
                    except asyncio.CancelledError:
                        yield _sse_frame(
                            "error",
                            {
                                "code": "run_cancelled",
                                "message": "Agent 运行已取消",
                            },
                        )
                    except Exception as error:
                        logger.error(
                            "agent_run.stream_failed | run_id=%s error_type=%s",
                            run_id,
                            type(error).__name__,
                            exc_info=(type(error), error, error.__traceback__),
                        )
                        yield _sse_frame("error", _safe_stream_error(error))
                    yield _sse_frame("end", {})
                    return
                if not values:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.1)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Agent-Run-Id": run_id,
            "X-Agent-Thread-Id": thread_id,
        },
    )


def _sse_frame(event: str, payload: object, *, event_id: str | None = None) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), default=str
    )
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {encoded}\n\n"


def _safe_stream_error(error: Exception) -> dict[str, object]:
    category = getattr(error, "category", None)
    return {
        "code": getattr(error, "code", "agent_run_failed"),
        "message": getattr(
            error, "safe_message", "Agent 运行失败，请稍后重试"
        ),
        "retryable": bool(getattr(error, "retryable", False)),
        **(
            {"category": getattr(category, "value", str(category))}
            if category is not None
            else {}
        ),
    }
