from uuid import uuid4


OWNER = {"X-Tenant-Id": "tenant-a", "X-User-Id": "user-a"}
OUTSIDER = {"X-Tenant-Id": "tenant-a", "X-User-Id": "user-b"}
TEXT = (
    "2026年10月2日到10月4日去杭州，3个人，预算1500元，住西湖东侧，"
    "喜欢自然和美食，2日10:30到杭州东站，4日19:00从杭州东站离开，"
    "灵隐寺必须去，不想太累。"
)


def test_completed_run_candidates_are_reused_and_user_choice_is_persisted(client, monkeypatch):
    planned = client.post("/api/v1/plans/from-text", headers=OWNER, json={"text": TEXT})
    assert planned.status_code == 200
    planning = planned.json()["planning"]
    assert len(planning["candidates"]) == 3
    source_run_id = planned.headers["x-agent-run-id"]
    endpoint = f"/api/v1/runs/{source_run_id}/plan-session"

    async def no_replanning(*args, **kwargs):
        raise AssertionError("创建选择会话不应重新执行规划或需求模型")

    runtime = client.app.state.planning_runtime
    monkeypatch.setattr(runtime.lifecycle_service, "_planning_runner", no_replanning)
    monkeypatch.setattr(runtime.requirement_workflow, "ainvoke", no_replanning)

    assert client.post(endpoint, headers=OUTSIDER).status_code == 403
    created = client.post(endpoint, headers=OWNER)
    assert created.status_code == 200
    session = created.json()
    assert session["status"] == "awaiting_candidate_selection"
    assert session["active_version"] is None
    assert session["candidates"] == planning["candidates"]
    assert session["interrupt"]["payload"]["recommended_candidate_id"] == planning["selected_plan"]["id"]

    # 连接选择会话只读取 Checkpoint 并创建 Interrupt，无新模型或工具调用。
    trace = client.get(f"/api/v1/runs/{created.headers['x-agent-run-id']}/trace?limit=500", headers=OWNER).json()["events"]
    assert not {"tool.started", "llm.started"} & {event["event_type"] for event in trace}
    assert client.post(endpoint, headers=OWNER).json()["session_id"] == session["session_id"]

    choice = next(
        candidate for candidate in session["candidates"]
        if candidate["validation"]["valid"] and candidate["id"] != planning["selected_plan"]["id"]
    )
    confirmed = client.post(
        f"/api/v1/plan-sessions/{session['session_id']}/resume",
        headers=OWNER,
        json={
            "interrupt_id": session["interrupt"]["id"],
            "request_id": str(uuid4()),
            "expected_session_revision": session["session_revision"],
            "action": {"kind": "select_candidate", "candidate_id": choice["id"]},
        },
    )
    assert confirmed.status_code == 200
    active = confirmed.json()
    assert active["status"] == "active"
    assert active["active_version"]["selected_candidate_id"] == choice["id"]
    assert active["active_version"]["version_id"] == "V1"
    assert "select_candidate" not in active["allowed_actions"]
    assert client.get(f"/api/v1/plan-sessions/{session['session_id']}", headers=OWNER).json()["active_version"] == active["active_version"]
    assert client.post(endpoint, headers=OWNER).json()["active_version"] == active["active_version"]


def test_incomplete_run_cannot_create_selection_session(client):
    response = client.post("/api/v1/plans/from-text", headers=OWNER, json={"text": "去杭州"})
    assert response.json()["status"] == "needs_clarification"
    endpoint = f"/api/v1/runs/{response.headers['x-agent-run-id']}/plan-session"
    assert client.post(endpoint, headers=OWNER).status_code == 409


def test_completed_clarification_can_enter_candidate_selection(client):
    initial = client.post(
        "/api/v1/plans/from-text", headers=OWNER,
        json={"text": TEXT.replace("4日19:00从杭州东站离开，", "")},
    ).json()
    assert initial["status"] == "needs_clarification"
    resumed = client.post(
        f"/api/v1/plans/from-text/{initial['thread_id']}/resume",
        headers=OWNER,
        json={
            "interrupt_id": initial["interrupt"]["id"],
            "request_id": str(uuid4()),
            "answer": "4日19:00从杭州东站离开。",
        },
    )
    assert resumed.json()["status"] == "completed"
    connected = client.post(
        f"/api/v1/runs/{resumed.headers['x-agent-run-id']}/plan-session", headers=OWNER,
    )
    assert connected.status_code == 200
    assert connected.json()["status"] == "awaiting_candidate_selection"
    assert connected.json()["candidates"] == resumed.json()["planning"]["candidates"]
