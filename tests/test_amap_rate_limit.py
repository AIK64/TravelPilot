"""验证真实 HTTP 出口节奏，避免并发与重试绕过高德接口配额。"""

import asyncio

import httpx
import pytest

from travel_agent.tools.providers.amap import AMapClient
from travel_agent.tools.retry import RetryPolicy


SUCCESS = {"status": "1", "info": "OK", "infocode": "10000"}
DRIVING = "/v5/direction/driving"
WALKING = "/v5/direction/walking"


@pytest.mark.asyncio
async def test_concurrent_requests_share_endpoint_spacing():
    starts: list[float] = []

    def handler(request):
        starts.append(asyncio.get_running_loop().time())
        return httpx.Response(200, json=SUCCESS)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://restapi.amap.com"
    ) as http:
        client = AMapClient(http, api_key="test-key")
        await asyncio.gather(
            *(client.request_json("route.driving", DRIVING, {}) for _ in range(3))
        )

    assert len(starts) == 3
    assert all(b - a >= 0.395 for a, b in zip(starts, starts[1:]))


@pytest.mark.asyncio
async def test_endpoints_have_independent_request_slots():
    paths: list[str] = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=SUCCESS)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://restapi.amap.com"
    ) as http:
        client = AMapClient(http, api_key="test-key")
        await client.request_json("route.driving", DRIVING, {})
        await asyncio.gather(
            client.request_json("route.driving", DRIVING, {}),
            client.request_json("route.walking", WALKING, {}),
        )

    assert paths == [DRIVING, WALKING, DRIVING]


@pytest.mark.asyncio
async def test_rate_limit_retry_obeys_same_request_spacing(caplog):
    starts: list[float] = []

    def handler(request):
        starts.append(asyncio.get_running_loop().time())
        payload = (
            {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10021"}
            if len(starts) == 1
            else SUCCESS
        )
        return httpx.Response(200, json=payload)

    caplog.set_level("INFO", logger="travel_agent.tools.providers.amap")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://restapi.amap.com"
    ) as http:
        client = AMapClient(http, api_key="test-key")
        outcome = await RetryPolicy(
            max_attempts=2, base_delay_seconds=0, jitter=lambda: 0
        ).execute(lambda: client.request_json("route.driving", DRIVING, {}))

    assert outcome.attempts == 2
    assert outcome.value == SUCCESS
    assert starts[1] - starts[0] >= 0.395
    assert "amap.request.throttled" in caplog.text
    assert "test-key" not in caplog.text
