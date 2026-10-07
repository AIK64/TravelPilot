"""将 Provider 调用编排为可缓存、可重试且可观察的工具结果。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, TypeVar

from travel_agent.config import Settings
from travel_agent.domain.tool_models import (
    POIFacts,
    POISearchQuery,
    RouteMode,
    RouteQuery,
    RouteResult,
    ToolCallContext,
    ToolErrorInfo,
    ToolResult,
    ToolStatus,
    route_key,
)
from travel_agent.tools.cache import AsyncTTLCache, CacheLookup
from travel_agent.tools.errors import ToolRetryExhausted
from travel_agent.tools.errors import ToolProviderError
from travel_agent.execution.context import (
    begin_tool,
    begin_tool_attempt,
    effective_timeout,
    finish_tool,
    match_fault,
    tool_retry,
)
from travel_agent.execution.faults import FaultMode, FaultPoint
from travel_agent.tools.protocols import POIProvider, RouteProvider
from travel_agent.tools.retry import RetryEvent, RetryPolicy


T = TypeVar("T")
SafeLogScalar = str | int | float | bool | None
logger = logging.getLogger("travel_agent.tools.gateway")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _log_event(event: str, **fields: SafeLogScalar) -> None:
    """只允许经过选择的标量字段进入工具日志，避免记录参数或原始响应。"""
    if not all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in fields.values()
    ):
        raise TypeError("tool log fields must be safe scalars")
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    logger.info("%s %s", event, details)


class ToolGateway:
    """以统一可靠性语义执行供应商无关的 POI 与路线工具调用。"""

    def __init__(
        self,
        *,
        poi_provider: POIProvider,
        route_provider: RouteProvider,
        cache: AsyncTTLCache[ToolResult[Any]],
        retry: RetryPolicy,
        semaphore: asyncio.Semaphore,
        poi_cache_ttl_seconds: int,
        route_cache_ttl_seconds: int,
        timeout_seconds: float = 5.0,
        cache_enabled: bool = True,
        utcnow: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._poi_provider = poi_provider
        self._route_provider = route_provider
        self._cache = cache
        self._retry = retry
        self._semaphore = semaphore
        self._poi_cache_ttl_seconds = poi_cache_ttl_seconds
        self._route_cache_ttl_seconds = route_cache_ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._cache_enabled = cache_enabled
        self._utcnow = utcnow

    async def search_pois(
        self,
        queries: list[POISearchQuery],
        context: ToolCallContext,
    ) -> list[ToolResult[list[POIFacts]]]:
        provider = self._poi_provider.name
        return await asyncio.gather(
            *(
                self._execute(
                    cache_key=(
                        f"{provider}|poi|{query.city.casefold()}|"
                        f"{query.keyword.casefold()}|{query.exact_match}|{query.limit}|{query.types}"
                    ),
                    ttl_seconds=self._poi_cache_ttl_seconds,
                    provider=provider,
                    operation="poi.search",
                    context=context,
                    request=query,
                    call=lambda query=query: self._poi_provider.search_pois(query),
                )
                for query in queries
            )
        )

    async def get_routes(
        self,
        queries: list[RouteQuery],
        context: ToolCallContext,
    ) -> dict[str, ToolResult[RouteResult]]:
        provider = self._route_provider.name
        unique_queries = {route_key(query): query for query in queries}
        keys = list(unique_queries)
        results = await asyncio.gather(
            *(
                self._execute(
                    cache_key=f"{provider}|route|{key}",
                    ttl_seconds=self._route_cache_ttl_seconds,
                    provider=provider,
                    operation=f"route.get_{query.mode.value}",
                    context=context,
                    request=query,
                    call=(
                        (lambda query=query: self._route_provider.get_walking_route(query))
                        if query.mode is RouteMode.WALKING
                        else (
                            lambda query=query: self._route_provider.get_driving_route(
                                query
                            )
                        )
                    ),
                )
                for key, query in unique_queries.items()
            )
        )
        return dict(zip(keys, results, strict=True))

    async def _execute(
        self,
        *,
        cache_key: str,
        ttl_seconds: int,
        provider: str,
        operation: str,
        context: ToolCallContext,
        call: Callable[[], Awaitable[T]],
        request: object = None,
    ) -> ToolResult[T]:
        parent_event_id = begin_tool(
            operation, provider=provider, thread_id=context.thread_id, request=request
        )
        self._log_started(context, provider, operation)

        async def load() -> ToolResult[T]:
            started = perf_counter()
            attempt = 0

            async def one_attempt() -> T:
                nonlocal attempt
                attempt += 1
                begin_tool_attempt(
                    operation,
                    provider=provider,
                    attempt=attempt,
                    parent_event_id=parent_event_id,
                )
                injected = match_fault(
                    _fault_point(operation), operation=operation, attempt=attempt
                )
                if injected is not None:
                    if (
                        injected is FaultMode.EMPTY_BUSINESS_RESULT
                        and operation == "poi.search"
                    ):
                        return []  # type: ignore[return-value]
                    raise _injected_tool_error(injected, operation)
                async with self._semaphore:
                    try:
                        return await asyncio.wait_for(
                            call(), timeout=effective_timeout(self._timeout_seconds)
                        )
                    except TimeoutError as error:
                        raise ToolProviderError.timeout(operation) from error

            try:
                outcome = await self._retry.execute(
                    one_attempt,
                    on_retry=lambda event: self._log_retry(
                        event, context, provider, operation, parent_event_id
                    ),
                )
            except ToolRetryExhausted as exhausted:
                return ToolResult.failed(
                    provider=provider,
                    error=ToolErrorInfo.from_provider_error(exhausted.last_error),
                    attempt_count=exhausted.attempts,
                    elapsed_ms=round((perf_counter() - started) * 1000, 2),
                )

            fetched_at = self._utcnow()
            return ToolResult.success(
                data=outcome.value,
                provider=provider,
                fetched_at=fetched_at,
                expires_at=fetched_at + timedelta(seconds=ttl_seconds),
                cache_hit=False,
                attempt_count=outcome.attempts,
                elapsed_ms=round((perf_counter() - started) * 1000, 2),
            )

        if self._cache_enabled:
            lookup = await self._cache.get_or_load(
                cache_key,
                ttl_seconds,
                load,
                should_cache=lambda result: result.status is ToolStatus.SUCCESS,
            )
        else:
            lookup = CacheLookup(
                value=await load(),
                hit=False,
                expires_at_monotonic=0.0,
            )
        result = lookup.value.model_copy(
            update={
                "cache_hit": lookup.hit,
                "attempt_count": 0 if lookup.hit else lookup.value.attempt_count,
                "elapsed_ms": 0.0 if lookup.hit else lookup.value.elapsed_ms,
            }
        )
        self._log_result(result, context, provider, operation)
        finish_tool(
            operation,
            provider=provider,
            status=result.status.value,
            cache_hit=result.cache_hit,
            attempt_count=result.attempt_count,
            elapsed_ms=result.elapsed_ms,
            error_code=result.error.code if result.error is not None else None,
            parent_event_id=parent_event_id,
            request=request,
            result=result,
        )
        return result

    def _log_started(
        self,
        context: ToolCallContext,
        provider: str,
        operation: str,
        parent_event_id: str | None = None,
    ) -> None:
        _log_event(
            "tool.started",
            thread_id=context.thread_id,
            provider=provider,
            operation=operation,
            attempt=1,
        )

    async def _log_retry(
        self,
        event: RetryEvent,
        context: ToolCallContext,
        provider: str,
        operation: str,
        parent_event_id: str,
    ) -> None:
        _log_event(
            "tool.retry_scheduled",
            thread_id=context.thread_id,
            provider=provider,
            operation=operation,
            attempt=event.attempt,
            next_attempt=event.next_attempt,
            delay_seconds=round(event.delay_seconds, 3),
            category=event.error.category.value,
            code=event.error.code,
            retryable=event.error.retryable,
        )
        tool_retry(
            operation,
            provider=provider,
            attempt=event.attempt,
            category=event.error.category.value,
            code=event.error.code,
            parent_event_id=parent_event_id,
        )

    def _log_result(
        self,
        result: ToolResult[Any],
        context: ToolCallContext,
        provider: str,
        operation: str,
    ) -> None:
        if result.cache_hit:
            _log_event(
                "tool.cache_hit",
                thread_id=context.thread_id,
                provider=provider,
                operation=operation,
                attempt_count=result.attempt_count,
                cache_hit="true",
                elapsed_ms=result.elapsed_ms,
            )
        elif result.status is ToolStatus.SUCCESS:
            _log_event(
                "tool.completed",
                thread_id=context.thread_id,
                provider=provider,
                operation=operation,
                attempt_count=result.attempt_count,
                cache_hit="false",
                elapsed_ms=result.elapsed_ms,
            )
        else:
            error = result.error
            assert error is not None
            _log_event(
                "tool.failed",
                thread_id=context.thread_id,
                provider=provider,
                operation=operation,
                attempt_count=result.attempt_count,
                cache_hit="false",
                elapsed_ms=result.elapsed_ms,
                category=error.category.value,
                code=error.code,
                retryable=error.retryable,
            )


def build_gateway(
    settings: Settings,
    poi_provider: POIProvider,
    route_provider: RouteProvider,
    *,
    cache_enabled: bool = True,
) -> ToolGateway:
    """将显式选定的 Provider 与配置装配为一个共享可靠性边界。"""
    return ToolGateway(
        poi_provider=poi_provider,
        route_provider=route_provider,
        cache=AsyncTTLCache(max_entries=settings.tool_cache_max_entries),
        retry=RetryPolicy(
            max_attempts=settings.tool_max_attempts,
            base_delay_seconds=settings.tool_backoff_base_seconds,
            max_delay_seconds=settings.tool_max_backoff_seconds,
        ),
        semaphore=asyncio.Semaphore(settings.tool_max_concurrency),
        poi_cache_ttl_seconds=settings.poi_cache_ttl_seconds,
        route_cache_ttl_seconds=settings.route_cache_ttl_seconds,
        timeout_seconds=settings.tool_timeout_seconds,
        cache_enabled=cache_enabled,
    )


def _fault_point(operation: str) -> FaultPoint:
    return FaultPoint.ROUTE_PROVIDER if operation.startswith("route.") else FaultPoint.POI_PROVIDER


def _injected_tool_error(mode: FaultMode, operation: str) -> ToolProviderError:
    mapping = {
        FaultMode.TIMEOUT: ("timeout", True),
        FaultMode.RATE_LIMIT: ("rate_limit", True),
        FaultMode.AUTH_ERROR: ("authentication", False),
        FaultMode.CONNECTION_ERROR: ("connection", True),
        FaultMode.INVALID_SCHEMA: ("invalid_response", False),
        FaultMode.EMPTY_BUSINESS_RESULT: ("empty_business_result", False),
        FaultMode.WRITE_FAILURE: ("upstream_unavailable", True),
    }
    code, retryable = mapping[mode]
    from travel_agent.domain.tool_models import ToolErrorCategory

    category = {
        "timeout": ToolErrorCategory.TIMEOUT,
        "rate_limit": ToolErrorCategory.RATE_LIMIT,
        "authentication": ToolErrorCategory.AUTHENTICATION,
        "connection": ToolErrorCategory.CONNECTION,
        "invalid_response": ToolErrorCategory.INVALID_RESPONSE,
        "empty_business_result": ToolErrorCategory.INVALID_RESPONSE,
        "upstream_unavailable": ToolErrorCategory.UPSTREAM_UNAVAILABLE,
    }[code]
    return ToolProviderError(
        category=category,
        code=f"injected_{code}",
        operation=operation,
        retryable=retryable,
        safe_message="Injected tool failure for evaluation",
    )
