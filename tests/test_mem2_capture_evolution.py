from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.storage import MilvusLiteFulltextStore

ROOT = Path(__file__).parents[1]
CAPTURE_HOOK = ROOT / "plugins" / "personal-agent-memory" / "scripts" / "capture.mjs"


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir, memory_kernel="mem2"))


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _region(
    client: TestClient,
    headers: dict[str, str],
    kind: str,
    region_id: str,
    *,
    session_id: str | None = None,
) -> None:
    payload: dict[str, object] = {"kind": kind, "name": region_id, "region_id": region_id}
    if session_id is not None:
        payload["session_id"] = session_id
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json=payload,
    )
    assert response.status_code == 201, response.text


def _write_region(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    content: str,
) -> str:
    response = client.post(
        "/api/v1/memories",
        headers=headers,
        json={
            "region_id": region_id,
            "content": content,
            "memory_type": "domain_fact",
            "source_references": ["test:r2-f1"],
            "actor": "test",
            "idempotency_key": f"r2-f1-{region_id}-{content}",
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _bind(client: TestClient, headers: dict[str, str], root: Path, region_id: str) -> None:
    response = client.post(
        "/api/v1/project-bindings",
        headers=headers,
        json={"project_root": str(root), "region_id": region_id},
    )
    assert response.status_code == 201, response.text


def _event(
    event_id: str,
    session_id: str,
    turn_id: str,
    kind: str,
    content: str,
    cwd: Path,
    *,
    occurred_at: str = "2026-10-24T00:00:00Z",
    topic_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": event_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "event_kind": kind,
        "content": content,
        "occurred_at": occurred_at,
        "cwd": str(cwd),
    }
    if topic_id is not None:
        payload["topic_id"] = topic_id
    return payload


def _capture(
    client: TestClient, headers: dict[str, str], payload: dict[str, object]
) -> dict[str, object]:
    response = client.post("/api/v1/capture/events", headers=headers, json=payload)
    assert response.status_code == 202, response.text
    return response.json()


def _search(
    client: TestClient,
    headers: dict[str, str],
    query: str,
    *,
    cwd: Path,
    session_id: str | None = None,
    topic_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"query": query, "cwd": str(cwd), "limit": 20}
    if session_id is not None:
        payload["session_id"] = session_id
    if topic_id is not None:
        payload["topic_id"] = topic_id
    response = client.post("/api/v1/search", headers=headers, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _all_units(client: TestClient) -> list[object]:
    kernel = client.app.state.memory_kernel
    bundle = kernel._require_bundle()
    return [
        unit
        for scope in bundle.kv_store.scopes()
        for unit in kernel._units_in_scope(bundle, scope)
    ]


def _fail_fulltext_insert_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original_insert = MilvusLiteFulltextStore.insert
    failed = False

    def fail_once(
        store: MilvusLiteFulltextStore, scope: object, docs: object
    ) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated fulltext index failure")
        original_insert(store, scope, docs)

    monkeypatch.setattr(MilvusLiteFulltextStore, "insert", fail_once)


def test_capture_routes_to_project_topic_and_isolated_chat_regions(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    other_root = tmp_path / "other"
    project_root.mkdir()
    other_root.mkdir()

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "project-one")
        _region(client, headers, "topic", "rust")
        _bind(client, headers, project_root, "project-one")

        project = _capture(
            client,
            headers,
            _event(
                "project-event",
                "project-session",
                "project-turn",
                "user",
                "PROJECT-ONLY-MARKER",
                project_root,
                topic_id="rust",
            ),
        )
        topic = _capture(
            client,
            headers,
            _event(
                "topic-event",
                "  topic-session  ",
                "topic-turn",
                "user",
                "TOPIC-ONLY-MARKER",
                other_root,
                topic_id="  rust  ",
            ),
        )
        chat_one = _capture(
            client,
            headers,
            _event("chat-one", "chat-session-one", "one", "user", "ALPHAUNIQUE", other_root),
        )
        chat_two = _capture(
            client,
            headers,
            _event("chat-two", "chat-session-two", "two", "user", "BETAUNIQUE", other_root),
        )

        assert project["region_id"] == "project-one"
        assert project["space"] == "project:project-one"
        assert project["agent"] == "codex"
        assert topic["region_id"] == "rust"
        assert topic["space"] == "topic:rust"
        assert topic["agent"] == "codex"
        assert topic["session_id"] == "topic-session"
        assert chat_one["space"] == chat_two["space"] == "chat"
        assert chat_one["agent"] == chat_two["agent"] == "codex"
        assert chat_one["region_id"] != chat_two["region_id"]

        project_result = _search(
            client,
            headers,
            "PROJECT-ONLY-MARKER",
            cwd=project_root,
            session_id="project-session",
            topic_id="rust",
        )
        assert project_result["scope"]["kind"] == "project_cwd"
        assert {item["region_id"] for item in project_result["results"]} == {"project-one"}

        topic_result = _search(
            client,
            headers,
            "TOPIC-ONLY-MARKER",
            cwd=other_root,
            session_id="  topic-session  ",
            topic_id="  rust  ",
        )
        assert topic_result["scope"]["kind"] == "topic_session"
        assert {item["region_id"] for item in topic_result["results"]} == {"rust"}

        first_chat = _search(
            client, headers, "ALPHAUNIQUE", cwd=other_root, session_id="chat-session-one"
        )
        second_chat = _search(
            client, headers, "BETAUNIQUE", cwd=other_root, session_id="chat-session-two"
        )
        assert {item["region_id"] for item in first_chat["results"]} == {
            str(chat_one["region_id"])
        }
        assert {item["region_id"] for item in second_chat["results"]} == {
            str(chat_two["region_id"])
        }
        assert _search(
            client, headers, "ALPHAUNIQUE", cwd=other_root, session_id="chat-session-two"
        )["results"] == []

        duplicate = _capture(
            client,
            headers,
            _event(
                "project-event",
                "project-session",
                "project-turn",
                "user",
                "PROJECT-ONLY-MARKER",
                project_root,
                topic_id="rust",
            ),
        )
        assert duplicate["duplicate"] is True
        assert duplicate["memory_id"] == project["memory_id"]
        conflict = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json=_event(
                "project-event",
                "project-session",
                "project-turn",
                "user",
                "CHANGED-CONTENT",
                project_root,
            ),
        )
        assert conflict.status_code == 409

        events = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": "chat-session-one"},
        ).json()
        assert [item["event_id"] for item in events] == ["chat-one"]
        assert events[0]["library_id"] == chat_one["region_id"]


def test_non_project_default_recall_is_user_plus_current_chat_only(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "user", "user")
        _region(client, headers, "topic", "course")
        _region(client, headers, "project", "project-one")
        _bind(client, headers, project_root, "project-one")
        for region_id, content in (
            ("user", "DEFAULT-RECALL user"),
            ("course", "DEFAULT-RECALL topic"),
            ("project-one", "DEFAULT-RECALL project"),
        ):
            response = client.post(
                "/api/v1/memories",
                headers=headers,
                json={
                    "region_id": region_id,
                    "content": content,
                    "memory_type": "domain_fact",
                    "source_references": ["test:default-recall"],
                    "actor": "test",
                    "idempotency_key": f"recall-{region_id}",
                },
            )
            assert response.status_code == 201, response.text
        _capture(
            client,
            headers,
            _event(
                "chat-default",
                "chat-session",
                "turn",
                "user",
                "DEFAULT-RECALL chat",
                other_root,
            ),
        )

        result = _search(
            client, headers, "DEFAULT-RECALL", cwd=other_root, session_id="chat-session"
        )
        assert result["scope"]["kind"] == "user_and_session"
        assert "cross_region" not in result["budget"]
        assert {item["region_id"] for item in result["results"]} == {
            "user",
            _capture_region_id(client, headers, "chat-session"),
        }
        assert all("project" not in str(item["content"]) for item in result["results"])
        assert all("topic" not in str(item["content"]) for item in result["results"])


def test_explicit_session_region_search_uses_region_session_scope(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(
            client,
            headers,
            "session",
            "session-region",
            session_id="bound-session",
        )
        memory_id = _write_region(
            client, headers, "session-region", "R2F1-SESSION-REGION"
        )
        cross_region_attempt = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_ids": ["session-region"],
                "query": "R2F1-SESSION-REGION",
            },
        )
        assert cross_region_attempt.status_code == 422
        for selector in (
            {"region_id": "session-region"},
            {
                "region_id": "session-region",
                "session_id": "bound-session",
            },
        ):
            response = client.post(
                "/api/v1/search",
                headers=headers,
                json={**selector, "query": "R2F1-SESSION-REGION"},
            )
            assert response.status_code == 200, response.text
            assert [item["document_id"] for item in response.json()["results"]] == [
                memory_id
            ]


def test_explicit_project_and_topic_search_with_session_keeps_base_memory(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "project-base")
        _region(client, headers, "topic", "topic-base")
        project_id = _write_region(
            client, headers, "project-base", "R2F1-PROJECT-BASE"
        )
        topic_id = _write_region(client, headers, "topic-base", "R2F1-TOPIC-BASE")
        response = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_ids": ["project-base", "topic-base"],
                "session_id": "unrelated-session",
                "query": "R2F1",
            },
        )
        assert response.status_code == 200, response.text
        assert {item["document_id"] for item in response.json()["results"]} == {
            project_id,
            topic_id,
        }


def _capture_region_id(
    client: TestClient, headers: dict[str, str], session_id: str
) -> str:
    events = client.get(
        "/api/v1/capture/events", headers=headers, params={"session_id": session_id}
    ).json()
    return str(events[0]["region_id"])


def test_evolve_consolidates_capture_without_candidate_queue_and_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    other_root = tmp_path / "other"
    other_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _capture(
            client,
            headers,
            _event(
                "evolve-user",
                "evolve-session",
                "turn-one",
                "user",
                "EVOLVE-SOURCE user fact",
                other_root,
            ),
        )
        _capture(
            client,
            headers,
            _event(
                "evolve-assistant",
                "evolve-session",
                "turn-one",
                "assistant",
                "EVOLVE-SOURCE assistant conclusion",
                other_root,
            ),
        )
        evolved = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "evolve-session"},
        )
        assert evolved.status_code == 202, evolved.text
        assert evolved.json() == {"queued": 1}
        repeated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "evolve-session"},
        )
        assert repeated.json() == {"queued": 0}

        rounds = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={"session_id": "evolve-session"},
        ).json()
        assert rounds[0]["status"] == "done"
        assert rounds[0]["candidate_id"] is None

        fulltext_calls: list[str] = []
        fulltext_limits: list[int] = []
        original_search = MilvusLiteFulltextStore.search

        def tracked_search(
            store: MilvusLiteFulltextStore, scope: object, query: object
        ) -> object:
            fulltext_calls.append(str(getattr(query, "text", "")))
            fulltext_limits.append(int(getattr(query, "top_k", 0)))
            return original_search(store, scope, query)

        monkeypatch.setattr(MilvusLiteFulltextStore, "search", tracked_search)
        result = _search(
            client, headers, "助手", cwd=other_root, session_id="evolve-session"
        )
        assert "助手" in fulltext_calls
        assert fulltext_limits and all(limit <= 20 for limit in fulltext_limits)
        evolved_items = [
            item
            for item in result["results"]
            if item.get("user_metadata", {}).get("memory_type") == "evolved"
        ]
        assert len(evolved_items) == 1
        assert "用户: EVOLVE-SOURCE user fact" in str(evolved_items[0]["content"])
        assert "助手: EVOLVE-SOURCE assistant conclusion" in str(evolved_items[0]["content"])
        assert evolved_items[0]["user_metadata"]["provenance"]

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        after_restart = _search(
            client, headers, "助手", cwd=other_root, session_id="evolve-session"
        )
        assert any(
            item.get("user_metadata", {}).get("memory_type") == "evolved"
            for item in after_restart["results"]
        )


def test_evolve_record_failure_restart_reconciles_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    other_root = tmp_path / "other"
    other_root.mkdir()
    app = _app(state_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = _headers(state_dir)
        _capture(
            client,
            headers,
            _event("r1f1-user", "r1f1", "turn", "user", "R1F1 user fact", other_root),
        )
        _capture(
            client,
            headers,
            _event(
                "r1f1-assistant",
                "r1f1",
                "turn",
                "assistant",
                "R1F1 assistant conclusion",
                other_root,
            ),
        )
        kernel = app.state.memory_kernel

        def fail_completion(evolve_key: str, memory_id: str) -> None:
            del evolve_key, memory_id
            raise RuntimeError("simulated evolution record failure")

        monkeypatch.setattr(kernel, "_complete_evolution", fail_completion)
        failed = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r1f1"},
        )
        assert failed.status_code == 500

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        retried = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r1f1"},
        )
        assert retried.status_code == 202, retried.text
        assert retried.json() == {"queued": 0}
        result = _search(client, headers, "助手", cwd=other_root, session_id="r1f1")
        evolved = [
            item
            for item in result["results"]
            if item.get("user_metadata", {}).get("memory_type") == "evolved"
        ]
        assert len(evolved) == 1
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            rows = connection.execute(
                "SELECT status, memory_id FROM capture_evolutions"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "completed"
        assert rows[0][1] == evolved[0]["document_id"]


def test_blank_topic_and_session_identifiers_return_422(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        event = _event("blank", "session", "turn", "user", "content", project_root)
        blank_topic = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={**event, "topic_id": "   "},
        )
        blank_session = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={**event, "session_id": "   "},
        )
        blank_consolidation = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "   "},
        )
        blank_search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(project_root), "query": "content", "session_id": "   "},
        )
        blank_topic_search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(project_root), "query": "content", "topic_id": "   "},
        )
        assert blank_topic.status_code == 422
        assert blank_session.status_code == 422
        assert blank_consolidation.status_code == 422
        assert blank_search.status_code == 422
        assert blank_topic_search.status_code == 422


@pytest.mark.parametrize(
    "occurred_at",
    [
        "not-a-timestamp",
        "2026-10-24T00:00:00",
        "2026-10-24T00:00:00+25:00",
    ],
)
def test_capture_rejects_invalid_or_naive_occurred_at(
    tmp_path: Path, occurred_at: str
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    with TestClient(_app(state_dir)) as client:
        response = client.post(
            "/api/v1/capture/events",
            headers=_headers(state_dir),
            json=_event(
                "invalid-time",
                "session",
                "turn",
                "user",
                "content",
                root,
                occurred_at=occurred_at,
            ),
        )
        assert response.status_code == 422


def test_session_id_over_index_limit_is_rejected_without_capture(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    session_id = "s" * 257
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        capture = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json=_event(
                "oversized-session",
                session_id,
                "turn",
                "user",
                "R3F2 content",
                root,
            ),
        )
        region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={
                "kind": "session",
                "name": "oversized",
                "region_id": "oversized-session",
                "session_id": session_id,
            },
        )
        assert capture.status_code == 422
        assert region.status_code == 422
        assert client.get("/api/v1/capture/events", headers=headers).json() == []


def test_capture_retry_after_index_failure_does_not_duplicate_kv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    app = _app(state_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = _headers(state_dir)
        payload = _event(
            "r3f2-capture",
            "s" * 256,
            "turn",
            "user",
            "R3F2 retry marker",
            root,
        )
        _fail_fulltext_insert_once(monkeypatch)
        failed = client.post("/api/v1/capture/events", headers=headers, json=payload)
        assert failed.status_code in {500, 503}
        retried = client.post("/api/v1/capture/events", headers=headers, json=payload)
        assert retried.status_code == 202, retried.text
        events = client.get(
            "/api/v1/capture/events", headers=headers, params={"session_id": "s" * 256}
        ).json()
        assert len(events) == 1
        units = [
            unit
            for unit in _all_units(client)
            if unit.user_metadata.get("capture_event_id") == "r3f2-capture"
        ]
        assert len(units) == 1


def test_evolve_index_failure_retry_and_restart_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    app = _app(state_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = _headers(state_dir)
        for event_id, kind, content in (
            ("r3f1-user", "user", "R3F1MARK " + "U" * 40_000),
            ("r3f1-assistant", "assistant", "A" * 40_000),
        ):
            _capture(
                client,
                headers,
                _event(event_id, "r3f1", "turn", kind, content, root),
            )
        _fail_fulltext_insert_once(monkeypatch)
        failed = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r3f1"},
        )
        assert failed.status_code in {500, 503}
        assert [
            unit
            for unit in _all_units(client)
            if unit.user_metadata.get("memory_type") == "evolved"
        ] == []
        retried = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r3f1"},
        )
        assert retried.status_code == 202, retried.text
        assert retried.json() == {"queued": 1}
        evolved = [
            unit
            for unit in _all_units(client)
            if unit.user_metadata.get("memory_type") == "evolved"
        ]
        assert len(evolved) == 2
        assert all(unit.scope.agent == "codex" for unit in evolved)
        result = _search(client, headers, "用户", cwd=root, session_id="r3f1")
        assert any(
            item.get("user_metadata", {}).get("memory_type") == "evolved"
            for item in result["results"]
        )

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        repeated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r3f1"},
        )
        assert repeated.json() == {"queued": 0}
        assert (
            len(
                [
                    unit
                    for unit in _all_units(client)
                    if unit.user_metadata.get("memory_type") == "evolved"
                ]
            )
            == 2
        )
        result = _search(client, headers, "用户", cwd=root, session_id="r3f1")
        assert any(
            item.get("user_metadata", {}).get("memory_type") == "evolved"
            for item in result["results"]
        )


def test_evolve_restart_cleans_unindexed_residual_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    app = _app(state_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = _headers(state_dir)
        _capture(
            client,
            headers,
            _event("r3f1-restart-user", "r3f1-restart", "turn", "user", "fact", root),
        )
        _capture(
            client,
            headers,
            _event(
                "r3f1-restart-assistant",
                "r3f1-restart",
                "turn",
                "assistant",
                "conclusion",
                root,
            ),
        )
        kernel = app.state.memory_kernel
        _fail_fulltext_insert_once(monkeypatch)

        def fail_cleanup(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("simulated interrupted cleanup")

        monkeypatch.setattr(kernel, "_purge_evolution_units", fail_cleanup)
        failed = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r3f1-restart"},
        )
        assert failed.status_code in {500, 503}
        assert any(
            unit.user_metadata.get("memory_type") == "evolved"
            for unit in _all_units(client)
        )

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        retried = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r3f1-restart"},
        )
        assert retried.status_code == 202, retried.text
        assert retried.json() == {"queued": 1}
        result = _search(client, headers, "助手", cwd=root, session_id="r3f1-restart")
        assert any(
            item.get("user_metadata", {}).get("memory_type") == "evolved"
            for item in result["results"]
        )


def test_codex_agent_scope_and_non_codex_compatibility(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "topic", "agent-topic")
        personal_id = _write_region(
            client, headers, "agent-topic", "R3F4 PERSONAL"
        )
        captured = _capture(
            client,
            headers,
            _event(
                "r3f4-capture",
                "r3f4-session",
                "turn",
                "user",
                "R3F4 CODEX",
                root,
                topic_id="agent-topic",
            ),
        )
        assert captured["agent"] == "codex"
        result = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_ids": ["agent-topic"],
                "session_id": "r3f4-session",
                "query": "R3F4",
            },
        )
        assert result.status_code == 200, result.text
        assert {item["document_id"] for item in result.json()["results"]} == {
            personal_id,
            captured["memory_id"],
        }
        agents = {
            unit.scope.agent
            for unit in _all_units(client)
            if "R3F4" in unit.content
        }
        assert agents == {"personal-agent-memory", "codex"}


def test_session_retention_policy_archives_captures_but_keeps_evolved_units(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    other_root = tmp_path / "other"
    other_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _capture(
            client,
            headers,
            _event(
                "old-capture",
                "retention-session",
                "turn-old",
                "user",
                "RETENTION-TRANSIENT marker",
                other_root,
                occurred_at="2020-01-01T00:00:00Z",
            ),
        )
        evolved = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "retention-session"},
        )
        assert evolved.json() == {"queued": 1}
        region_id = _capture_region_id(client, headers, "retention-session")

        policy = client.put(
            f"/api/v1/libraries/{region_id}/retention-policy",
            headers=headers,
            json={
                "inbox_days": 0,
                "candidate_days": 90,
                "recycle_days": 30,
                "early_inbox_cleanup": True,
                "diagnostics_enabled": False,
                "diagnostic_days": 7,
            },
        )
        assert policy.status_code == 200, policy.text
        assert policy.json()["inbox_days"] == 0

        preview = client.get(
            f"/api/v1/libraries/{region_id}/retention-cleanup/preview",
            headers=headers,
        ).json()
        assert preview["counts"]["captures_to_archive"] == 1
        assert preview["counts"]["evolved_units_protected"] == 1

        cleaned = client.post(
            f"/api/v1/libraries/{region_id}/retention-cleanup",
            headers=headers,
        )
        assert cleaned.status_code == 200, cleaned.text
        assert cleaned.json()["counts"]["captures_archived"] == 1
        repeated_preview = client.get(
            f"/api/v1/libraries/{region_id}/retention-cleanup/preview",
            headers=headers,
        ).json()
        assert repeated_preview["counts"]["captures_to_archive"] == 0
        repeated = client.post(
            f"/api/v1/libraries/{region_id}/retention-cleanup",
            headers=headers,
        ).json()
        assert repeated["counts"]["captures_archived"] == 0
        assert client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": "retention-session"},
        ).json() == []

        result = _search(
            client, headers, "RETENTION-TRANSIENT", cwd=other_root, session_id="retention-session"
        )
        assert result["results"]
        assert all(
            item.get("user_metadata", {}).get("memory_type") == "evolved"
            for item in result["results"]
        )

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        persisted = client.get(
            f"/api/v1/libraries/{region_id}/retention-policy", headers=headers
        ).json()
        assert persisted["inbox_days"] == 0


def test_capture_hook_spools_and_exits_zero_when_daemon_is_unavailable(tmp_path: Path) -> None:
    plugin_data = tmp_path / "plugin-data"
    event = {
        "session_id": "offline-session",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Keep this while the daemon is offline.",
    }
    result = subprocess.run(
        ["node", str(CAPTURE_HOOK)],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        timeout=4,
        check=False,
        env={
            **os.environ,
            "PERSONAL_AGENT_MEMORY_URL": "http://127.0.0.1:1",
            "PERSONAL_AGENT_MEMORY_API_KEY": "offline-test-key",
            "PERSONAL_AGENT_MEMORY_TIMEOUT_MS": "100",
            "PLUGIN_DATA": str(plugin_data),
        },
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert list((plugin_data / "capture" / "spool").glob("event-*.json"))


def test_capture_retry_recovers_when_cleanup_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "unbound"
    root.mkdir()
    payload = _event(
        "r4f1-capture",
        "r4f1-session",
        "turn",
        "user",
        "R4F1 recoverable capture",
        root,
    )
    app = _app(state_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = _headers(state_dir)
        kernel = app.state.memory_kernel
        original_purge = kernel._purge_capture_units
        failed = False

        def fail_cleanup_once(*args: object, **kwargs: object) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("simulated cleanup failure")
            original_purge(*args, **kwargs)

        _fail_fulltext_insert_once(monkeypatch)
        monkeypatch.setattr(kernel, "_purge_capture_units", fail_cleanup_once)
        first = client.post("/api/v1/capture/events", headers=headers, json=payload)
        assert first.status_code in {500, 503}

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        retried = None
        for _ in range(3):
            retried = client.post("/api/v1/capture/events", headers=headers, json=payload)
            assert retried.status_code != 409, retried.text
            if retried.status_code == 202:
                break
        assert retried is not None
        assert retried.status_code == 202, retried.text
        events = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": "r4f1-session"},
        ).json()
        assert len(events) == 1
        units = [
            unit
            for unit in _all_units(client)
            if unit.user_metadata.get("capture_event_id") == "r4f1-capture"
        ]
        assert len(units) == 1


def test_evolve_late_same_turn_event_replaces_without_duplicate_content(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "unbound"
    root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _capture(
            client,
            headers,
            _event(
                "r4f2-user",
                "r4f2-session",
                "shared-turn",
                "user",
                "LATE-FACT user statement",
                root,
            ),
        )
        first = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r4f2-session"},
        )
        assert first.json() == {"queued": 1}

        _capture(
            client,
            headers,
            _event(
                "r4f2-assistant",
                "r4f2-session",
                "shared-turn",
                "assistant",
                "LATE-ANSWER assistant conclusion",
                root,
            ),
        )
        second = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r4f2-session"},
        )
        assert second.json() == {"queued": 1}

        evolved = [
            unit
            for unit in _all_units(client)
            if unit.user_metadata.get("memory_type") == "evolved"
        ]
        assert len(evolved) == 1
        assert evolved[0].content.count("LATE-FACT") == 1
        assert evolved[0].content.count("LATE-ANSWER") == 1

        repeated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "r4f2-session"},
        )
        assert repeated.json() == {"queued": 0}
        result = _search(
            client,
            headers,
            "用户",
            cwd=root,
            session_id="r4f2-session",
        )
        assert any(
            item.get("user_metadata", {}).get("memory_type") == "evolved"
            for item in result["results"]
        )


def test_extreme_timezone_timestamp_is_rejected_before_retention(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "unbound"
    root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _capture(
            client,
            headers,
            _event(
                "r4f3-valid",
                "r4f3-session",
                "turn",
                "user",
                "R4F3 valid capture",
                root,
            ),
        )
        invalid = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json=_event(
                "r4f3-invalid",
                "r4f3-session",
                "turn",
                "user",
                "R4F3 invalid capture",
                root,
                occurred_at="0001-01-01T00:00:00+23:59",
            ),
        )
        assert invalid.status_code == 422
        region_id = _capture_region_id(client, headers, "r4f3-session")
        preview = client.get(
            f"/api/v1/libraries/{region_id}/retention-cleanup/preview",
            headers=headers,
        )
        assert preview.status_code == 200, preview.text
        cleanup = client.post(
            f"/api/v1/libraries/{region_id}/retention-cleanup",
            headers=headers,
        )
        assert cleanup.status_code == 200, cleanup.text


def test_capture_with_non_topic_topic_id_returns_422(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "unbound"
    root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "r4f4-project")
        response = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json=_event(
                "r4f4-capture",
                "r4f4-session",
                "turn",
                "user",
                "R4F4 invalid topic",
                root,
                topic_id="r4f4-project",
            ),
        )
        assert response.status_code == 422
        assert client.get("/api/v1/capture/events", headers=headers).json() == []
