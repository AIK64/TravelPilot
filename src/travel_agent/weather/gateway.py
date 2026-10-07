from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta, timezone
from time import perf_counter
from typing import Any, TypeVar

from travel_agent.domain.tool_models import (
    ToolCallContext,
    ToolErrorInfo,
    ToolResult,
    ToolStatus,
)
from travel_agent.domain.weather_models import WeatherForecast, WeatherLocation
from travel_agent.tools.cache import AsyncTTLCache
from travel_agent.tools.errors import ToolRetryExhausted
from travel_agent.tools.errors import ToolProviderError
from travel_agent.tools.retry import RetryEvent, RetryPolicy
from travel_agent.weather.protocols import WeatherProvider
from travel_agent.config import Settings
from travel_agent.execution.context import (
    begin_tool,
    begin_tool_attempt,
    effective_timeout,
    finish_tool,
    match_fault,
    tool_retry,
)
from travel_agent.execution.faults import FaultPoint
from travel_agent.tools.gateway import _injected_tool_error


T = TypeVar("T")
logger = logging.getLogger("travel_agent.weather.gateway")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WeatherToolGateway:
    def __init__(
        self,
        *,
        provider: WeatherProvider,
        cache: AsyncTTLCache[ToolResult[Any]],
        retry: RetryPolicy,
        semaphore: asyncio.Semaphore,
        location_cache_ttl_seconds: int = 86_400,
        forecast_cache_ttl_seconds: int = 1_800,
        timeout_seconds: float = 5.0,
        utcnow: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._retry = retry
        self._semaphore = semaphore
        self._location_ttl = location_cache_ttl_seconds
        self._forecast_ttl = forecast_cache_ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._utcnow = utcnow

    async def resolve_location(
        self, destination: str, context: ToolCallContext
    ) -> ToolResult[WeatherLocation]:
        normalized = destination.strip().casefold()
        return await self._execute(
            cache_key=f"{self._provider.name}|weather-location|{normalized}",
            ttl_seconds=self._location_ttl,
            operation="weather.resolve_location",
            context=context,
            request={"destination": destination},
            call=lambda: self._provider.resolve_location(destination),
        )

    async def get_forecast(
        self,
        location: WeatherLocation,
        *,
        start_date: date,
        end_date: date,
        context: ToolCallContext,
    ) -> ToolResult[WeatherForecast]:
        return await self._execute(
            cache_key=(
                f"{self._provider.name}|weather-forecast|{location.adcode}|"
                f"{start_date.isoformat()}|{end_date.isoformat()}|daily-v1"
            ),
            ttl_seconds=self._forecast_ttl,
            operation="weather.get_forecast",
            context=context,
            request={"location": location, "start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
            call=lambda: self._provider.get_forecast(
                location, start_date=start_date, end_date=end_date
            ),
        )

    async def _execute(
        self,
        *,
        cache_key: str,
        ttl_seconds: int,
        operation: str,
        context: ToolCallContext,
        call: Callable[[], Awaitable[T]],
        request: object = None,
    ) -> ToolResult[T]:
        provider = self._provider.name
        parent_event_id = begin_tool(
            operation, provider=provider, thread_id=context.thread_id, request=request
        )
        logger.info(
            "weather.tool.started | thread_id=%s provider=%s operation=%s",
            context.thread_id,
            provider,
            operation,
        )

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
                    FaultPoint.WEATHER_PROVIDER,
                    operation=operation,
                    attempt=attempt,
                )
                if injected is not None:
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
                attempt_count=outcome.attempts,
                elapsed_ms=round((perf_counter() - started) * 1000, 2),
            )

        lookup = await self._cache.get_or_load(
            cache_key,
            ttl_seconds,
            load,
            should_cache=lambda result: result.status is ToolStatus.SUCCESS,
        )
        result = lookup.value.model_copy(
            update={
                "cache_hit": lookup.hit,
                "attempt_count": 0 if lookup.hit else lookup.value.attempt_count,
                "elapsed_ms": 0.0 if lookup.hit else lookup.value.elapsed_ms,
            }
        )
        if result.status is ToolStatus.SUCCESS:
            logger.info(
                "weather.tool.completed | thread_id=%s provider=%s operation=%s "
                "cache_hit=%s attempt_count=%s elapsed_ms=%s",
                context.thread_id,
                provider,
                operation,
                result.cache_hit,
                result.attempt_count,
                result.elapsed_ms,
            )
        else:
            assert result.error is not None
            logger.warning(
                "weather.tool.failed | thread_id=%s provider=%s operation=%s "
                "category=%s code=%s retryable=%s attempt_count=%s",
                context.thread_id,
                provider,
                operation,
                result.error.category.value,
                result.error.code,
                result.error.retryable,
                result.attempt_count,
            )
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

    async def _log_retry(
        self,
        event: RetryEvent,
        context: ToolCallContext,
        provider: str,
        operation: str,
        parent_event_id: str | None = None,
    ) -> None:
        logger.info(
            "weather.tool.retry_scheduled | thread_id=%s provider=%s "
            "operation=%s attempt=%s next_attempt=%s delay_seconds=%s "
            "category=%s code=%s",
            context.thread_id,
            provider,
            operation,
            event.attempt,
            event.next_attempt,
            round(event.delay_seconds, 3),
            event.error.category.value,
            event.error.code,
        )
        tool_retry(
            operation,
            provider=provider,
            attempt=event.attempt,
            category=event.error.category.value,
            code=event.error.code,
            parent_event_id=parent_event_id,
        )


def build_weather_gateway(
    settings: Settings, provider: WeatherProvider
) -> WeatherToolGateway:
    return WeatherToolGateway(
        provider=provider,
        cache=AsyncTTLCache(max_entries=settings.tool_cache_max_entries),
        retry=RetryPolicy(
            max_attempts=settings.tool_max_attempts,
            base_delay_seconds=settings.tool_backoff_base_seconds,
            max_delay_seconds=settings.tool_max_backoff_seconds,
        ),
        semaphore=asyncio.Semaphore(settings.tool_max_concurrency),
        forecast_cache_ttl_seconds=settings.weather_cache_ttl_seconds,
        timeout_seconds=settings.tool_timeout_seconds,
    )
