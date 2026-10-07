"""高德地图 HTTP 契约到供应商无关事实的适配。"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation

import httpx

from travel_agent.domain.models import Coordinate, TimeWindow
from travel_agent.domain.tool_models import (
    POIFacts,
    POISearchQuery,
    RouteMode,
    RouteQuery,
    RouteResult,
    ToolErrorCategory,
    ValueSource,
)
from travel_agent.tools.errors import ToolProviderError


logger = logging.getLogger(__name__)


RETRYABLE_AMAP_CODES = {
    "10004",
    "10015",
    "10016",
    "10017",
    "10019",
    "10020",
    "10021",
}
AUTH_CODES = {"10001"}
PERMISSION_CODES = {"10002", "10005", "10009", "10012"}
RATE_LIMIT_CODES = {"10003", "10004", "10014", "10019", "10020", "10021"}

_WEEKDAY = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
}
_WEEKLY_HOURS = re.compile(
    r"^周(?P<start>[一二三四五六日])至周(?P<end>[一二三四五六日]):"
    r"(?P<hours>\d{2}:\d{2}-\d{2}:\d{2})$"
)


class AMapClient:
    """共享的安全高德 JSON client；按接口限速，缓存与重试由 Gateway 管理。"""

    REQUEST_INTERVAL_SECONDS = 0.4

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._client = client
        self.__api_key = api_key
        self.timeout_seconds = timeout_seconds
        # Runtime 内所有规划任务共享此 client；不同接口使用独立的请求节奏。
        self._request_locks: dict[str, asyncio.Lock] = {}
        self._last_request_at: dict[str, float] = {}

    async def _wait_for_request_slot(self, operation: str, path: str) -> None:
        lock = self._request_locks.setdefault(path, asyncio.Lock())
        async with lock:
            loop = asyncio.get_running_loop()
            last_request_at = self._last_request_at.get(path)
            if last_request_at is not None:
                delay = self.REQUEST_INTERVAL_SECONDS - (loop.time() - last_request_at)
                if delay > 0:
                    logger.info(
                        "amap.request.throttled operation=%s wait_seconds=%.3f",
                        operation,
                        delay,
                    )
                    # 等待被取消时不预占后续槽位，也不会发出 HTTP 请求。
                    while delay > 0:
                        await asyncio.sleep(delay)
                        delay = self.REQUEST_INTERVAL_SECONDS - (
                            loop.time() - last_request_at
                        )
            self._last_request_at[path] = loop.time()

    async def request_json(
        self,
        operation: str,
        path: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        # 每次真实请求（包括 Gateway 重试）都经过同一接口的限速器。
        await self._wait_for_request_slot(operation, path)
        error: ToolProviderError | None = None
        response: httpx.Response | None = None
        try:
            response = await self._client.get(
                path,
                params={**params, "key": self.__api_key},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            error = ToolProviderError.timeout(operation)
        except httpx.ConnectError:
            error = ToolProviderError(
                category=ToolErrorCategory.CONNECTION,
                code="connection",
                operation=operation,
                retryable=True,
                safe_message="The provider connection failed. Please try again.",
            )
        except httpx.HTTPStatusError as exc:
            error = self._http_error(
                operation,
                exc.response.status_code,
                retry_after_seconds=_parse_retry_after_seconds(
                    exc.response.headers.get("Retry-After")
                ),
            )
        except httpx.RequestError:
            error = ToolProviderError(
                category=ToolErrorCategory.CONNECTION,
                code="request_error",
                operation=operation,
                retryable=True,
                safe_message="The provider request failed. Please try again.",
            )
        if error is not None:
            raise error
        assert response is not None

        try:
            payload = response.json()
        except ValueError:
            error = self._invalid_response(operation)
        if error is not None:
            raise error

        if not isinstance(payload, dict):
            raise self._invalid_response(operation)
        if not all(
            isinstance(payload.get(field), str)
            for field in ("status", "info", "infocode")
        ):
            raise self._invalid_response(operation)

        status = payload["status"]
        infocode = payload["infocode"]
        if status == "1":
            return payload
        if status == "0":
            raise self._amap_error(operation, infocode)
        raise self._invalid_response(operation)

    @staticmethod
    def _http_error(
        operation: str,
        status_code: int,
        retry_after_seconds: float | None = None,
    ) -> ToolProviderError:
        if status_code == 429:
            return ToolProviderError(
                category=ToolErrorCategory.RATE_LIMIT,
                code="http_429",
                operation=operation,
                retryable=True,
                safe_message="The provider rate limit was reached. Please try again.",
                retry_after_seconds=retry_after_seconds,
            )
        if status_code in {502, 503, 504}:
            return ToolProviderError(
                category=ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                code=f"http_{status_code}",
                operation=operation,
                retryable=True,
                safe_message="The provider is temporarily unavailable. Please try again.",
                retry_after_seconds=retry_after_seconds,
            )
        if status_code == 401:
            return ToolProviderError.authentication(operation)
        if status_code == 403:
            return ToolProviderError(
                category=ToolErrorCategory.PERMISSION,
                code="http_403",
                operation=operation,
                retryable=False,
                safe_message="The provider request is not permitted.",
            )
        return ToolProviderError(
            category=ToolErrorCategory.INVALID_REQUEST,
            code=f"http_{status_code}",
            operation=operation,
            retryable=False,
            safe_message="The provider rejected the request.",
        )

    @staticmethod
    def _invalid_response(operation: str) -> ToolProviderError:
        return ToolProviderError(
            category=ToolErrorCategory.INVALID_RESPONSE,
            code="invalid_response",
            operation=operation,
            retryable=False,
            safe_message="The provider returned an invalid response.",
        )

    @staticmethod
    def _amap_error(operation: str, infocode: str) -> ToolProviderError:
        if infocode in AUTH_CODES:
            return ToolProviderError.authentication(operation)
        if infocode in PERMISSION_CODES:
            return ToolProviderError(
                category=ToolErrorCategory.PERMISSION,
                code=infocode,
                operation=operation,
                retryable=False,
                safe_message="The provider request is not permitted.",
            )
        if infocode in RATE_LIMIT_CODES:
            return ToolProviderError(
                category=ToolErrorCategory.RATE_LIMIT,
                code=infocode,
                operation=operation,
                retryable=infocode in RETRYABLE_AMAP_CODES,
                safe_message="The provider rate limit was reached. Please try again.",
            )
        if infocode in RETRYABLE_AMAP_CODES:
            return ToolProviderError(
                category=ToolErrorCategory.UPSTREAM_UNAVAILABLE,
                code=infocode,
                operation=operation,
                retryable=True,
                safe_message="The provider is temporarily unavailable. Please try again.",
            )
        return ToolProviderError(
            category=ToolErrorCategory.INVALID_REQUEST,
            code=infocode,
            operation=operation,
            retryable=False,
            safe_message="The provider rejected the request.",
        )


class AMapPOIProvider:
    """将高德文字搜索结果归一化为 POI facts。"""

    name = "amap"

    def __init__(self, amap_client: AMapClient) -> None:
        self._amap_client = amap_client

    async def search_pois(self, query: POISearchQuery) -> list[POIFacts]:
        payload = await self._amap_client.request_json(
            "poi.search",
            "/v5/place/text",
            {
                "keywords": query.keyword,
                "region": query.city,
                "city_limit": "true",
                "show_fields": "business",
                "page_size": query.limit,
                **({"types": query.types} if query.types else {}),
            },
        )
        raw_pois = payload.get("pois")
        if not isinstance(raw_pois, list):
            raise AMapClient._invalid_response("poi.search")

        fetched_at = datetime.now(timezone.utc)
        error: ToolProviderError | None = None
        try:
            facts = [self._normalize_poi(raw_poi, fetched_at) for raw_poi in raw_pois]
        except (KeyError, TypeError, ValueError, InvalidOperation):
            error = AMapClient._invalid_response("poi.search")
        if error is not None:
            raise error
        return facts

    def _normalize_poi(self, raw_poi: object, fetched_at: datetime) -> POIFacts:
        if not isinstance(raw_poi, dict):
            raise ValueError("poi must be an object")
        poi_id = _required_text(raw_poi, "id")
        name = _required_text(raw_poi, "name")
        city = _required_text(raw_poi, "cityname")
        coordinate = _parse_coordinate(_required_text(raw_poi, "location"))
        categories = _split_categories(raw_poi.get("type"))
        business = raw_poi.get("business", {})
        if not isinstance(business, dict):
            raise ValueError("business must be an object")

        today_window = _parse_time_window(business.get("opentime_today"))
        weekly_windows = _parse_weekly_windows(business.get("opentime_week"))
        cost = _parse_cost(business.get("cost"))
        sources = {
            "id": ValueSource.PROVIDER,
            "name": ValueSource.PROVIDER,
            "city": ValueSource.PROVIDER,
            "coordinate": ValueSource.PROVIDER,
            "categories": ValueSource.PROVIDER,
        }
        if today_window is not None:
            sources["today_opening_window"] = ValueSource.PROVIDER
        if weekly_windows:
            sources["opening_windows_by_weekday"] = ValueSource.PROVIDER
        if cost is not None:
            sources["average_cost_per_person"] = ValueSource.PROVIDER

        return POIFacts(
            id=poi_id,
            name=name,
            city=city,
            coordinate=coordinate,
            categories=categories,
            opening_windows_by_weekday=weekly_windows,
            today_opening_window=today_window,
            today_opening_date=date.today() if today_window is not None else None,
            average_cost_per_person=cost,
            suggested_duration_minutes=None,
            provider=self.name,
            fetched_at=fetched_at,
            field_sources=sources,
        )


class AMapRouteProvider:
    """将高德驾车/步行路线归一化为供应商无关路线事实。"""

    name = "amap"

    def __init__(self, amap_client: AMapClient) -> None:
        self._amap_client = amap_client

    async def get_driving_route(self, query: RouteQuery) -> RouteResult:
        params: dict[str, object] = {
            "origin": _format_coordinate(query.origin),
            "destination": _format_coordinate(query.destination),
            "strategy": query.strategy,
            "show_fields": "cost",
        }
        if query.origin_poi_id is not None:
            params["origin_id"] = query.origin_poi_id
        if query.destination_poi_id is not None:
            params["destination_id"] = query.destination_poi_id

        payload = await self._amap_client.request_json(
            "route.driving",
            "/v5/direction/driving",
            params,
        )
        return self._normalize_route(payload, RouteMode.DRIVING, "route.driving")

    async def get_walking_route(self, query: RouteQuery) -> RouteResult:
        params: dict[str, object] = {
            "origin": _format_coordinate(query.origin),
            "destination": _format_coordinate(query.destination),
            "show_fields": "cost",
        }
        if query.origin_poi_id is not None:
            params["origin_id"] = query.origin_poi_id
        if query.destination_poi_id is not None:
            params["destination_id"] = query.destination_poi_id

        payload = await self._amap_client.request_json(
            "route.walking",
            "/v5/direction/walking",
            params,
        )
        return self._normalize_route(payload, RouteMode.WALKING, "route.walking")

    @staticmethod
    def _normalize_route(
        payload: dict[str, object],
        mode: RouteMode,
        operation: str,
    ) -> RouteResult:
        error: ToolProviderError | None = None
        try:
            route = payload.get("route")
            if not isinstance(route, dict):
                raise ValueError("route must be an object")
            paths = route.get("paths")
            if not isinstance(paths, list) or not paths:
                raise ValueError("paths must be a non-empty list")
            path = paths[0]
            if not isinstance(path, dict):
                raise ValueError("path must be an object")
            cost = path.get("cost")
            if not isinstance(cost, dict):
                raise ValueError("cost must be an object")
            distance = _parse_positive_int(path.get("distance"))
            duration_seconds = _parse_positive_int(cost.get("duration"))
            return RouteResult(
                distance_meters=distance,
                duration_minutes=math.ceil(duration_seconds / 60),
                mode=mode,
                provider="amap",
                data_confidence=0.95,
                fetched_at=datetime.now(timezone.utc),
            )
        except (TypeError, ValueError):
            error = AMapClient._invalid_response(operation)
        if error is not None:
            raise error
        raise AssertionError("unreachable")


def _required_text(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {field}")
    return value.strip()


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _parse_coordinate(value: str) -> Coordinate:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("invalid location")
    longitude, latitude = (float(part.strip()) for part in parts)
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise ValueError("invalid location")
    return Coordinate(longitude=longitude, latitude=latitude)


def _format_coordinate(coordinate: Coordinate) -> str:
    return f"{coordinate.longitude:.6f},{coordinate.latitude:.6f}"


def _parse_positive_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("value must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip().isdigit():
        result = int(value.strip())
    else:
        raise ValueError("value must be an integer")
    if result <= 0:
        raise ValueError("value must be positive")
    return result


def _split_categories(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [category.strip() for category in value.split(";") if category.strip()]


def _parse_cost(value: object) -> Decimal | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        cost = Decimal(value)
    except InvalidOperation:
        return None
    if not cost.is_finite() or cost < 0:
        return None
    return cost


def _parse_time_window(value: object) -> TimeWindow | None:
    if not isinstance(value, str):
        return None
    try:
        start_text, end_text = value.split("-", maxsplit=1)
        return TimeWindow(
            start=time.fromisoformat(start_text),
            end=time.fromisoformat(end_text),
        )
    except ValueError:
        return None


def _parse_weekly_windows(value: object) -> dict[int, TimeWindow]:
    if not isinstance(value, str):
        return {}
    match = _WEEKLY_HOURS.fullmatch(value)
    if match is None:
        return {}
    window = _parse_time_window(match.group("hours"))
    if window is None:
        return {}
    start = _WEEKDAY[match.group("start")]
    end = _WEEKDAY[match.group("end")]
    weekdays = (
        range(start, end + 1)
        if start <= end
        else (*range(start, 7), *range(0, end + 1))
    )
    return {weekday: window for weekday in weekdays}
