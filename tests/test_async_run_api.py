from __future__ import annotations

import time

from fastapi.testclient import TestClient

from travel_agent.app import create_app
from travel_agent.config import Settings


OWNER = {"X-Tenant-Id": "tenant-a", "X-User-Id": "user-a"}
OUTSIDER = {"X-Tenant-Id": "tenant-a", "X-User-Id": "user-b"}


def test_async_run_api_owner_trace_sse_and_completed_cancel(hangzhou_trip) -> None:
    with TestClient(create_app(Settings.from_env({}))) as client:
        trip = client.post(
            "/api/v1/trips",
            headers=OWNER,
            json={"trip": hangzhou_trip.model_dump(mode="json")},
        )
        assert trip.status_code == 201
        trip_id = trip.json()["trip_id"]

        own_trip = client.get(f"/api/v1/trips/{trip_id}", headers=OWNER)
        forbidden_trip = client.get(
            f"/api/v1/trips/{trip_id}", headers=OUTSIDER
        )
        assert own_trip.status_code == 200
        assert forbidden_trip.status_code == 403

        started = client.post(
            f"/api/v1/trips/{trip_id}/runs",
            headers={**OWNER, "Idempotency-Key": "async-api-case-1"},
        )
        assert started.status_code == 202
        handle = started.json()
        run_id = handle["run_id"]

        record = None
        for _ in range(200):
            response = client.get(f"/api/v1/runs/{run_id}", headers=OWNER)
            assert response.status_code == 200
            record = response.json()
            if record["status"] != "running":
                break
            time.sleep(0.01)
        assert record is not None
        assert record["status"] in {"completed", "interrupted"}
        assert client.get(
            f"/api/v1/runs/{run_id}", headers=OUTSIDER
        ).status_code == 403

        trace = client.get(
            f"/api/v1/runs/{run_id}/trace?limit=500", headers=OWNER
        )
        assert trace.status_code == 200
        assert trace.json()["events"]

        stream = client.get(
            f"/api/v1/runs/{run_id}/events",
            headers={**OWNER, "Last-Event-ID": "0"},
        )
        assert stream.status_code == 200
        assert "event: trace" in stream.text
        assert "event: end" in stream.text

        cancelled = client.post(
            f"/api/v1/runs/{run_id}/cancel", headers=OWNER
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == record["status"]


def test_natural_plan_stream_emits_trace_before_result() -> None:
    complete_text = (
        "2026年10月2日到10月4日去杭州，3个人，预算1500元，住西湖东侧，"
        "喜欢自然和美食，2日10:30到杭州东站，4日19:00从杭州东站离开，"
        "灵隐寺必须去，不想太累。"
    )
    with TestClient(create_app(Settings.from_env({}))) as client:
        response = client.post(
            "/api/v1/plans/from-text/stream",
            headers={**OWNER, "Accept": "text/event-stream"},
            json={"text": complete_text, "reference_date": "2026-08-23"},
        )

        assert response.status_code == 200
        assert response.headers["x-agent-run-id"]
        assert response.headers["x-agent-thread-id"]
        assert response.text.index("event: trace") < response.text.index(
            "event: result"
        )
        assert "event: end" in response.text
        assert '"event_type":"run.started"' in response.text
        assert '"status":"completed"' in response.text
