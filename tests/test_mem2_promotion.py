from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir, memory_kernel="mem2"))


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _region(client: TestClient, headers: dict[str, str], kind: str, region_id: str) -> None:
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json={"kind": kind, "name": region_id, "region_id": region_id},
    )
    assert response.status_code == 201, response.text


def _write(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    content: str,
    *,
    key: str,
    memory_type: str = "domain_fact",
) -> dict[str, object]:
    response = client.post(
        "/api/v1/memories",
        headers=headers,
        json={
            "region_id": region_id,
            "content": content,
            "memory_type": memory_type,
            "source_references": [f"source:{key}"],
            "actor": "test",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _wait_promotion(
    client: TestClient,
    headers: dict[str, str],
    job_id: str,
    *,
    timeout: float = 15.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/promotions/{job_id}", headers=headers)
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"promotion job {job_id} did not finish")


def _search(
    client: TestClient,
    headers: dict[str, str],
    *,
    region_id: str,
    query: str,
) -> dict[str, object]:
    response = client.post(
        "/api/v1/search",
        headers=headers,
        json={
            "region_id": region_id,
            "query": query,
            "limit": 20,
            "token_budget": 10_000,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_promotion_promotes_user_and_topic_targets_and_keeps_project_fact(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        _region(client, headers, "topic", "rust")
        user_source = _write(
            client,
            headers,
            "demo",
            "I prefer concise answers",
            key="promotion-user",
            memory_type="preference",
        )
        topic_source = _write(
            client,
            headers,
            "demo",
            "Rust ownership is a memory-safety principle",
            key="promotion-topic",
        )
        project_source = _write(
            client,
            headers,
            "demo",
            "This repository uses pnpm for builds",
            key="promotion-project",
        )

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={},
        )
        assert started.status_code == 202, started.text
        job_id = str(started.json()["job_id"])
        status = _wait_promotion(client, headers, job_id)
        assert status["status"] == "completed", status
        summary = status["summary"]
        assert isinstance(summary, dict)
        assert summary["candidates"] == 3
        assert summary["promoted"] == 2
        assert summary["skipped"] == 1
        assert summary["failed"] == 0

        user_results = _search(
            client,
            headers,
            region_id="user",
            query="concise",
        )["results"]
        promoted_user = next(
            item for item in user_results if item["content"] == "I prefer concise answers"
        )
        assert promoted_user["user_metadata"]["memory_type"] == "promoted"
        assert promoted_user["user_metadata"]["source_unit_ids"] == [user_source["id"]]
        assert f"unit:{user_source['id']}" in promoted_user["user_metadata"]["source_references"]
        assert "source:promotion-user" in promoted_user["user_metadata"]["source_references"]
        assert promoted_user["user_metadata"]["provenance"] == [user_source["id"]]
        inspected = client.get(
            "/api/v1/libraries/user/document",
            headers=headers,
            params={"path": promoted_user["document_id"]},
        )
        assert inspected.status_code == 200, inspected.text
        assert inspected.json()["provenance"] == [user_source["id"]]

        topic_results = _search(
            client,
            headers,
            region_id="rust",
            query="ownership",
        )["results"]
        promoted_topic = next(
            item
            for item in topic_results
            if item["content"] == "Rust ownership is a memory-safety principle"
        )
        assert promoted_topic["user_metadata"]["memory_type"] == "promoted"
        assert promoted_topic["user_metadata"]["source_unit_ids"] == [topic_source["id"]]

        project_results = _search(
            client,
            headers,
            region_id="demo",
            query="pnpm",
        )["results"]
        assert any(item["document_id"] == project_source["id"] for item in project_results)
        assert all(
            item["user_metadata"].get("memory_type") != "promoted" for item in project_results
        )
        source_user_results = _search(
            client,
            headers,
            region_id="demo",
            query="concise",
        )["results"]
        assert any(item["document_id"] == user_source["id"] for item in source_user_results)
        source_topic_results = _search(
            client,
            headers,
            region_id="demo",
            query="memory-safety",
        )["results"]
        assert any(item["document_id"] == topic_source["id"] for item in source_topic_results)

        history = client.get(
            "/api/v1/libraries/user/history",
            headers=headers,
        )
        assert history.status_code == 200, history.text
        promotion_audits = [event for event in history.json() if event["action"] == "promote"]
        assert len(promotion_audits) == 1
        assert promotion_audits[0]["actor"] == "platform:auto-promotion"
        assert promotion_audits[0]["target_id"] == promoted_user["document_id"]
        topic_history = client.get(
            "/api/v1/libraries/rust/history",
            headers=headers,
        ).json()
        assert len([event for event in topic_history if event["action"] == "promote"]) == 1

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        persisted = client.get(f"/api/v1/promotions/{job_id}", headers=headers)
        assert persisted.status_code == 200, persisted.text
        assert persisted.json()["status"] == "completed"


def test_repeated_promotion_runs_are_idempotent_and_audit_once(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        _write(
            client,
            headers,
            "demo",
            "I prefer dark mode in every editor",
            key="promotion-idempotent",
            memory_type="preference",
        )

        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1

        second = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert second.status_code == 202, second.text
        second_status = _wait_promotion(client, headers, str(second.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 0
        assert second_status["summary"]["skipped"] == 1

        results = _search(
            client,
            headers,
            region_id="user",
            query="dark",
        )["results"]
        promoted = [
            item for item in results if item["content"] == "I prefer dark mode in every editor"
        ]
        assert len(promoted) == 1
        history = client.get(
            "/api/v1/libraries/user/history",
            headers=headers,
        ).json()
        assert len([event for event in history if event["action"] == "promote"]) == 1

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        records = connection.execute(
            "SELECT target_unit_id, audit_event_id FROM promotion_records"
        ).fetchall()
    assert len(records) == 1
    assert records[0][0]
    assert int(records[0][1]) > 0


def test_session_capture_evolve_then_promotion_preserves_source(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    other_root = tmp_path / "other"
    other_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        captured = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "promotion-user-event",
                "session_id": "promotion-session",
                "turn_id": "promotion-turn",
                "event_kind": "user",
                "content": "I prefer detailed explanations",
                "occurred_at": "2026-10-24T00:00:00Z",
                "cwd": str(other_root),
            },
        )
        assert captured.status_code == 202, captured.text
        capture_payload = captured.json()
        source_region_id = str(capture_payload["region_id"])
        assistant = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "promotion-assistant-event",
                "session_id": "promotion-session",
                "turn_id": "promotion-turn",
                "event_kind": "assistant",
                "content": "Noted",
                "occurred_at": "2026-10-24T00:00:01Z",
                "cwd": str(other_root),
            },
        )
        assert assistant.status_code == 202, assistant.text

        evolved = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "promotion-session"},
        )
        assert evolved.status_code == 202, evolved.text
        assert evolved.json() == {"queued": 1}

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": [source_region_id]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["status"] == "completed", status
        assert status["summary"]["promoted"] == 1

        user_results = _search(
            client,
            headers,
            region_id="user",
            query="detailed",
        )["results"]
        promoted = next(item for item in user_results if "detailed explanations" in item["content"])
        provenance = promoted["user_metadata"]["source_unit_ids"]
        assert provenance
        assert any(
            ref.startswith(f"capture:{source_region_id}:")
            for ref in promoted["user_metadata"]["source_references"]
        )

        source_results = _search(
            client,
            headers,
            region_id=source_region_id,
            query="detailed",
        )["results"]
        assert any("detailed explanations" in item["content"] for item in source_results)


def test_promotion_rejects_invalid_sources_and_missing_jobs(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")

        duplicate = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo", "demo"]},
        )
        assert duplicate.status_code == 422, duplicate.text
        unknown = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["missing"]},
        )
        assert unknown.status_code == 404, unknown.text
        missing_job = client.get(
            "/api/v1/promotions/promotion-missing",
            headers=headers,
        )
        assert missing_job.status_code == 404, missing_job.text


def test_injected_classifier_can_route_generalizable_knowledge_to_topic(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        _region(client, headers, "topic", "rust")
        _write(
            client,
            headers,
            "demo",
            "Ownership keeps shared state memory safe",
            key="promotion-classified-topic",
        )
        kernel = client.app.state.memory_kernel
        calls: list[dict[str, object]] = []

        def classify(content: str, context: dict[str, object]) -> dict[str, object]:
            calls.append({"content": content, "context": context})
            return {
                "category": "topic",
                "confidence": 0.99,
                "explicit": True,
                "topic_id": "rust",
                "reason": "deterministic classifier",
            }

        kernel._promotion_classifier = classify
        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["promoted"] == 1
        assert len(calls) == 1
        assert calls[0]["content"] == "Ownership keeps shared state memory safe"
        results = _search(
            client,
            headers,
            region_id="rust",
            query="ownership",
        )["results"]
        assert any(
            item["content"] == "Ownership keeps shared state memory safe"
            and item["user_metadata"]["memory_type"] == "promoted"
            for item in results
        )


def test_repeated_promotion_repairs_record_and_audit_after_partial_persistence(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    content = "I prefer keyboard shortcuts for navigation"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        _write(
            client,
            headers,
            "demo",
            content,
            key="promotion-repair",
            memory_type="preference",
        )

        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1

        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            row = connection.execute(
                "SELECT target_unit_id, audit_event_id FROM promotion_records"
            ).fetchone()
            assert row is not None
            target_unit_id = str(row[0])
            audit_event_id = int(row[1])
            connection.execute("DELETE FROM promotion_records")
            connection.execute(
                """UPDATE governance_events
                   SET unit_id = 'deleted-unit', after_content = 'stale'
                   WHERE id = ?""",
                (audit_event_id,),
            )

        second = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert second.status_code == 202, second.text
        second_status = _wait_promotion(client, headers, str(second.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 0
        assert second_status["summary"]["skipped"] == 1

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        repaired = connection.execute(
            "SELECT target_unit_id, audit_event_id FROM promotion_records"
        ).fetchone()
        assert repaired is not None
        assert str(repaired[0]) == target_unit_id
        assert int(repaired[1]) == audit_event_id
        audit = connection.execute(
            "SELECT unit_id, after_content FROM governance_events WHERE id = ?",
            (audit_event_id,),
        ).fetchone()
    assert audit == (target_unit_id, content)


def test_promotion_requires_high_confidence_classification(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    content = "I prefer short status reports"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        source = _write(
            client,
            headers,
            "demo",
            content,
            key="promotion-low-confidence",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel

        def classify(_: str, __: dict[str, object]) -> dict[str, object]:
            return {
                "category": "user",
                "confidence": 0.89,
                "explicit": True,
                "topic_id": None,
                "reason": "below the automatic promotion threshold",
            }

        kernel._promotion_classifier = classify
        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["candidates"] == 1
        assert status["summary"]["promoted"] == 0
        assert status["summary"]["skipped"] == 1

        source_results = _search(
            client,
            headers,
            region_id="demo",
            query="status reports",
        )["results"]
        assert any(item["document_id"] == source["id"] for item in source_results)
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_records"
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM governance_events WHERE action = 'promote'"
            ).fetchone() == (0,)
