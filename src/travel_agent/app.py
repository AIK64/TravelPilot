from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from travel_agent import __version__
from travel_agent.api.errors import (
    UTF8JSONResponse,
    clarification_conflict_exception_handler,
    clarification_not_found_exception_handler,
    tool_unavailable_exception_handler,
    requirement_unavailable_exception_handler,
    edit_unavailable_exception_handler,
    lifecycle_action_exception_handler,
    lifecycle_conflict_exception_handler,
    lifecycle_not_found_exception_handler,
    weather_unavailable_exception_handler,
    execution_budget_exception_handler,
    run_not_found_exception_handler,
    memory_conflict_exception_handler,
    memory_forbidden_exception_handler,
    memory_not_found_exception_handler,
    memory_policy_exception_handler,
    application_exception_handler,
)
from travel_agent.api.routes import router
from travel_agent.api.preferences import router as preference_router
from travel_agent.api.async_runs import router as async_run_router
from travel_agent.application.errors import ApplicationError
from travel_agent.application.service import TravelApplicationService
from travel_agent.config import AsyncExecutionBackend, Settings
from travel_agent.logging_config import configure_logging
from travel_agent.runtime import PlanningRuntime
from travel_agent.tools.errors import ToolUnavailableError
from travel_agent.requirements.errors import RequirementUnavailableError
from travel_agent.requirements.errors import (
    ClarificationResumeConflictError,
    ClarificationThreadNotFoundError,
)
from travel_agent.edits.errors import EditUnavailableError
from travel_agent.lifecycle.errors import (
    LifecycleActionError,
    LifecycleConflictError,
    LifecycleNotFoundError,
)
from travel_agent.weather.errors import WeatherUnavailableError
from travel_agent.execution.errors import ExecutionBudgetExceeded, RunNotFoundError
from travel_agent.memory.errors import (
    MemoryConflictError,
    MemoryForbiddenError,
    MemoryNotFoundError,
    MemoryPolicyError,
)
from travel_agent.identity.models import Principal
from travel_agent.mcp_server.server import build_http_mcp_app
from travel_agent.infrastructure.queue import open_run_queue


RuntimeFactory = Callable[[Settings], Awaitable[PlanningRuntime]]


def create_app(
    settings: Settings | None = None,
    runtime_factory: RuntimeFactory = PlanningRuntime.create,
) -> FastAPI:
    configure_logging()
    resolved_settings = settings or Settings.from_env()
    resolved_settings.validate()
    application_holder: dict[str, TravelApplicationService] = {}
    mcp_default_principal = Principal(
        tenant_id=resolved_settings.dev_tenant_id or "unauthenticated",
        user_id=resolved_settings.dev_user_id or "unauthenticated",
        scopes=frozenset({"read:data"}),
        authentication_method="mcp_http_default",
    )
    mcp_server, mcp_http_app = build_http_mcp_app(
        lambda: application_holder["service"],
        default_principal=mcp_default_principal,
        allow_default_identity=resolved_settings.dev_identity_enabled,
        allowed_hosts=resolved_settings.mcp_allowed_hosts,
        allowed_origins=resolved_settings.mcp_allowed_origins,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime = await runtime_factory(resolved_settings)
        queue_context = None
        run_queue = None
        if resolved_settings.async_execution_backend is AsyncExecutionBackend.REDIS:
            queue_context = open_run_queue(resolved_settings.redis_url)
            run_queue = await queue_context.__aenter__()
        application_service = TravelApplicationService(runtime, run_queue=run_queue)
        application.state.planning_runtime = runtime
        application.state.travel_application_service = application_service
        application_holder["service"] = application_service
        application.state.settings = resolved_settings
        try:
            async with mcp_server.session_manager.run():
                yield
        finally:
            await application_service.close()
            await runtime.close()
            if queue_context is not None:
                await queue_context.__aexit__(None, None, None)
            application_holder.clear()

    app = FastAPI(
        title="Constraint-Aware Travel Agent",
        version=__version__,
        default_response_class=UTF8JSONResponse,
        description=(
            "A grounded natural-language and structured-input travel agent with "
            "explicit Parse-Tool-Validate-Replan trajectories."
        ),
        lifespan=lifespan,
    )
    app.add_exception_handler(
        ToolUnavailableError,
        tool_unavailable_exception_handler,
    )
    app.add_exception_handler(
        RequirementUnavailableError,
        requirement_unavailable_exception_handler,
    )
    app.add_exception_handler(
        ClarificationThreadNotFoundError,
        clarification_not_found_exception_handler,
    )
    app.add_exception_handler(
        ClarificationResumeConflictError,
        clarification_conflict_exception_handler,
    )
    app.add_exception_handler(EditUnavailableError, edit_unavailable_exception_handler)
    app.add_exception_handler(
        LifecycleNotFoundError, lifecycle_not_found_exception_handler
    )
    app.add_exception_handler(
        LifecycleConflictError, lifecycle_conflict_exception_handler
    )
    app.add_exception_handler(
        LifecycleActionError, lifecycle_action_exception_handler
    )
    app.add_exception_handler(
        WeatherUnavailableError, weather_unavailable_exception_handler
    )
    app.add_exception_handler(
        ExecutionBudgetExceeded, execution_budget_exception_handler
    )
    app.add_exception_handler(RunNotFoundError, run_not_found_exception_handler)
    app.add_exception_handler(
        MemoryNotFoundError, memory_not_found_exception_handler
    )
    app.add_exception_handler(
        MemoryConflictError, memory_conflict_exception_handler
    )
    app.add_exception_handler(MemoryPolicyError, memory_policy_exception_handler)
    app.add_exception_handler(
        MemoryForbiddenError, memory_forbidden_exception_handler
    )
    app.add_exception_handler(ApplicationError, application_exception_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=[
            "Content-Type",
            "Idempotency-Key",
            "Last-Event-ID",
            "X-Tenant-Id",
            "X-User-Id",
            "X-Scopes",
        ],
        expose_headers=[
            "X-Agent-Run-Id",
            "X-Agent-Thread-Id",
            "X-Agent-Trace-Status",
        ],
    )
    app.include_router(router)
    app.include_router(preference_router)
    app.include_router(async_run_router)
    app.mount("/mcp", mcp_http_app, name="travel-mcp")
    return app


app = create_app()
