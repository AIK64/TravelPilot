from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

from travel_agent.domain.models import Coordinate, TimeWindow

if TYPE_CHECKING:
    from travel_agent.tools.errors import ToolProviderError


T = TypeVar("T")


class ProviderMode(StrEnum):
    MOCK = "mock"
    AMAP = "amap"
    BAIDU = "baidu"


class UnknownFactPolicy(StrEnum):
    ASSUME_WITH_WARNING = "assume_with_warning"
    STRICT = "strict"


class RouteMode(StrEnum):
    DRIVING = "driving"
    WALKING = "walking"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class ValueSource(StrEnum):
    PROVIDER = "provider"
    DERIVED = "derived"
    DEFAULT = "default"
    USER_CONFIRMED = "user_confirmed"


class ToolErrorCategory(StrEnum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"


class POISearchQuery(BaseModel):
    city: str
    keyword: str
    types: str = ""
    exact_match: bool = False
    limit: int = Field(default=10, ge=1, le=25)
    priority: int = 0


class POIFacts(BaseModel):
    id: str
    name: str
    city: str
    coordinate: Coordinate
    categories: list[str]
    opening_windows_by_weekday: dict[int, TimeWindow] = Field(default_factory=dict)
    today_opening_window: TimeWindow | None = None
    today_opening_date: date | None = None
    average_cost_per_person: Decimal | None = Field(default=None, ge=0)
    suggested_duration_minutes: int | None = Field(default=None, gt=0)
    provider: str
    fetched_at: datetime
    data_confidence: float = Field(default=1.0, ge=0, le=1)
    field_sources: dict[str, ValueSource] = Field(default_factory=dict)


class RouteQuery(BaseModel):
    origin: Coordinate
    destination: Coordinate
    origin_poi_id: str | None = None
    destination_poi_id: str | None = None
    mode: RouteMode = RouteMode.DRIVING
    strategy: int = 32


class RouteResult(BaseModel):
    distance_meters: int = Field(ge=0)
    duration_minutes: int = Field(ge=0)
    mode: RouteMode = RouteMode.DRIVING
    provider: str
    data_confidence: float = Field(ge=0, le=1)
    fetched_at: datetime


class ToolErrorInfo(BaseModel):
    category: ToolErrorCategory
    code: str
    operation: str
    retryable: bool
    safe_message: str

    @classmethod
    def from_provider_error(cls, error: ToolProviderError) -> "ToolErrorInfo":
        return cls(
            category=error.category,
            code=error.code,
            operation=error.operation,
            retryable=error.retryable,
            safe_message=error.safe_message,
        )


class ToolResult(BaseModel, Generic[T]):
    status: ToolStatus
    data: T | None = None
    provider: str
    fetched_at: datetime | None = None
    expires_at: datetime | None = None
    cache_hit: bool = False
    attempt_count: int = Field(default=0, ge=0)
    elapsed_ms: float | None = Field(default=None, ge=0)
    error: ToolErrorInfo | None = None

    @model_validator(mode="after")
    def validate_result_shape(self) -> "ToolResult[T]":
        if self.status is ToolStatus.SUCCESS and self.data is None:
            raise ValueError("data is required when status is success")
        if self.status is ToolStatus.FAILED and self.error is None:
            raise ValueError("error is required when status is failed")
        return self

    @classmethod
    def success(
        cls,
        *,
        data: T,
        provider: str,
        fetched_at: datetime | None = None,
        expires_at: datetime | None = None,
        cache_hit: bool = False,
        attempt_count: int = 0,
        elapsed_ms: float | None = None,
    ) -> "ToolResult[T]":
        return cls(
            status=ToolStatus.SUCCESS,
            data=data,
            provider=provider,
            fetched_at=fetched_at,
            expires_at=expires_at,
            cache_hit=cache_hit,
            attempt_count=attempt_count,
            elapsed_ms=elapsed_ms,
        )

    @classmethod
    def failed(
        cls,
        *,
        provider: str,
        error: ToolErrorInfo,
        fetched_at: datetime | None = None,
        expires_at: datetime | None = None,
        cache_hit: bool = False,
        attempt_count: int = 0,
        elapsed_ms: float | None = None,
    ) -> "ToolResult[T]":
        return cls(
            status=ToolStatus.FAILED,
            provider=provider,
            fetched_at=fetched_at,
            expires_at=expires_at,
            cache_hit=cache_hit,
            attempt_count=attempt_count,
            elapsed_ms=elapsed_ms,
            error=error,
        )


class ToolCallContext(BaseModel):
    thread_id: str


class ToolExecutionSummary(BaseModel):
    provider: str
    operation: str
    status: ToolStatus
    cache_hit: bool
    attempt_count: int


def route_key(query: RouteQuery) -> str:
    """构造含坐标版本、方向、模式和策略的稳定路线键。"""
    return (
        f"route:v1|{query.origin.longitude:.6f},{query.origin.latitude:.6f}"
        f"->{query.destination.longitude:.6f},{query.destination.latitude:.6f}"
        f"|{query.mode.value}|{query.strategy}"
    )


# 支持先导入 tool_models 的调用方，避免 provenance schema 受导入顺序影响。
from travel_agent.domain.models import rebuild_provenance_models

rebuild_provenance_models(poi_facts=POIFacts, value_source=ValueSource)
