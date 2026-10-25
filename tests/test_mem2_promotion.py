from __future__ import annotations

import json
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


def _inspect_document(
    client: TestClient,
    headers: dict[str, str],
    *,
    region_id: str,
    path: str,
) -> dict[str, object]:
    response = client.get(
        f"/api/v1/libraries/{region_id}/document",
        headers=headers,
        params={"path": path},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _update_document(
    client: TestClient,
    headers: dict[str, str],
    *,
    region_id: str,
    path: str,
    content: str,
    revision: str,
    mode: str,
    operation_id: str,
) -> dict[str, object]:
    response = client.put(
        f"/api/v1/libraries/{region_id}/document",
        headers=headers,
        json={
            "path": path,
            "content": content,
            "expected_source_version": revision,
            "operation_id": operation_id,
            "actor_type": "user",
            "source": "test-promotion-governance",
            "mode": mode,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _delete_document(
    client: TestClient,
    headers: dict[str, str],
    *,
    region_id: str,
    path: str,
    revision: str,
    mode: str,
    operation_id: str,
) -> dict[str, str]:
    preview = client.post(
        f"/api/v1/libraries/{region_id}/document/delete-preview",
        headers=headers,
        json={
            "path": path,
            "expected_source_version": revision,
            "operation_id": f"{operation_id}-preview",
        },
    )
    assert preview.status_code == 200, preview.text
    token = str(preview.json()["preview_token"])
    response = client.request(
        "DELETE",
        f"/api/v1/libraries/{region_id}/document",
        headers=headers,
        json={
            "path": path,
            "expected_source_version": revision,
            "preview_token": token,
            "operation_id": operation_id,
            "actor_type": "user",
            "source": "test-promotion-governance",
            "mode": mode,
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
        history = client.get(
            "/api/v1/libraries/demo/history",
            headers=headers,
        ).json()
        skip_audits = [item for item in history if item["action"] == "promotion_skip"]
        assert len(skip_audits) == 1
        assert "insufficient independent evidence" in skip_audits[0]["detail"]


def test_multi_source_evidence_promotes_once_and_merges_provenance(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    content = "I prefer compact diffs"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        first = _write(
            client,
            headers,
            "alpha",
            content,
            key="multi-source-alpha",
            memory_type="preference",
        )
        second = _write(
            client,
            headers,
            "beta",
            content,
            key="multi-source-beta",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = lambda content, context: {
            "category": "user",
            "confidence": 0.5,
            "explicit": False,
            "topic_id": None,
            "claim_subject": "user:preference",
            "claim_value": "compact diffs",
            "reason": "single-source confidence is intentionally low",
        }

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        summary = status["summary"]
        assert status["status"] == "completed"
        assert summary["candidates"] == 2
        assert summary["promoted"] == 1
        assert summary["skipped"] == 0
        assert summary["failed"] == 0
        assert summary["conflict_groups"] == 0
        assert summary["evidence_groups"] == 1

        user_results = _search(client, headers, region_id="user", query="compact")["results"]
        promoted = next(item for item in user_results if item["content"] == content)
        metadata = promoted["user_metadata"]
        assert metadata["promotion_evidence_kind"] == "multi_source"
        assert metadata["promotion_evidence_count"] == 2
        assert set(metadata["source_unit_ids"]) == {first["id"], second["id"]}
        assert set(metadata["source_region_ids"]) == {"alpha", "beta"}
        assert set(metadata["provenance"]) == {first["id"], second["id"]}
        for region_id, source in (("alpha", first), ("beta", second)):
            source_results = _search(
                client,
                headers,
                region_id=region_id,
                query="compact",
            )["results"]
            assert any(item["document_id"] == source["id"] for item in source_results)

        repeated = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert repeated.status_code == 202, repeated.text
        repeated_status = _wait_promotion(client, headers, str(repeated.json()["job_id"]))
        assert repeated_status["summary"]["promoted"] == 0
        assert repeated_status["summary"]["skipped"] == 1

        history = client.get("/api/v1/libraries/user/history", headers=headers).json()
        assert len([item for item in history if item["action"] == "promote"]) == 1

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        records = connection.execute(
            "SELECT evidence_kind, source_unit_ids FROM promotion_records"
        ).fetchall()
    assert len(records) == 1
    assert records[0][0] == "multi_source"
    assert set(json.loads(records[0][1])) == {first["id"], second["id"]}


def test_later_source_merges_into_existing_promotion(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    content = "I prefer compact diffs"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        first = _write(
            client,
            headers,
            "alpha",
            content,
            key="later-source-alpha",
            memory_type="preference",
        )
        first_started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert first_started.status_code == 202, first_started.text
        first_status = _wait_promotion(client, headers, str(first_started.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1

        second = _write(
            client,
            headers,
            "beta",
            content,
            key="later-source-beta",
            memory_type="preference",
        )
        second_started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["beta"]},
        )
        assert second_started.status_code == 202, second_started.text
        second_status = _wait_promotion(client, headers, str(second_started.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 0
        assert second_status["summary"]["skipped"] == 1

        user_results = _search(client, headers, region_id="user", query="compact")["results"]
        promoted = next(item for item in user_results if item["content"] == content)
        metadata = promoted["user_metadata"]
        assert set(metadata["source_unit_ids"]) == {first["id"], second["id"]}
        assert set(metadata["source_region_ids"]) == {"alpha", "beta"}
        assert set(metadata["provenance"]) == {first["id"], second["id"]}
        assert metadata["promotion_evidence_kind"] == "multi_source"

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        record = connection.execute(
            "SELECT source_unit_ids, source_region_ids FROM promotion_records"
        ).fetchone()
    assert record is not None
    assert set(json.loads(record[0])) == {first["id"], second["id"]}
    assert set(json.loads(record[1])) == {"alpha", "beta"}


def test_cross_run_conflicting_claim_is_blocked_and_audited(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        _write(
            client,
            headers,
            "alpha",
            "I prefer dark mode",
            key="cross-run-alpha",
            memory_type="preference",
        )
        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1

        _write(
            client,
            headers,
            "beta",
            "I prefer light mode",
            key="cross-run-beta",
            memory_type="preference",
        )
        second = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["beta"]},
        )
        assert second.status_code == 202, second.text
        second_status = _wait_promotion(client, headers, str(second.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 0
        assert second_status["summary"]["skipped"] == 1
        assert second_status["summary"]["conflict_groups"] == 1

        user_results = _search(client, headers, region_id="user", query="mode")["results"]
        contents = {item["content"] for item in user_results}
        assert "I prefer dark mode" in contents
        assert "I prefer light mode" not in contents
        history = client.get("/api/v1/libraries/beta/history", headers=headers).json()
        conflicts = [item for item in history if item["action"] == "conflict"]
        assert len(conflicts) == 1
        assert "dark mode" in conflicts[0]["detail"]
        assert "light mode" in conflicts[0]["detail"]


def test_conflicting_sources_skip_promotion_and_write_conflict_audit(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        _write(
            client,
            headers,
            "alpha",
            "I prefer dark mode",
            key="conflict-alpha",
            memory_type="preference",
        )
        _write(
            client,
            headers,
            "beta",
            "I prefer light mode",
            key="conflict-beta",
            memory_type="preference",
        )
        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        summary = status["summary"]
        assert summary["candidates"] == 2
        assert summary["promoted"] == 0
        assert summary["skipped"] == 2
        assert summary["conflict_groups"] == 1

        repeated = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert repeated.status_code == 202, repeated.text
        repeated_status = _wait_promotion(client, headers, str(repeated.json()["job_id"]))
        assert repeated_status["summary"]["conflict_groups"] == 1
        assert repeated_status["summary"]["promoted"] == 0

        for region_id, query in (("alpha", "dark"), ("beta", "light")):
            history = client.get(
                f"/api/v1/libraries/{region_id}/history",
                headers=headers,
            ).json()
            conflicts = [item for item in history if item["action"] == "conflict"]
            assert len(conflicts) == 1
            assert "dark mode" in conflicts[0]["detail"]
            assert "light mode" in conflicts[0]["detail"]
            source_results = _search(
                client,
                headers,
                region_id=region_id,
                query=query,
            )["results"]
            assert any(query in item["content"] for item in source_results)

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM promotion_records").fetchone() == (0,)


def test_complementary_claims_are_not_treated_as_conflict(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        _region(client, headers, "topic", "rust")
        _write(
            client,
            headers,
            "alpha",
            "Rust is fast",
            key="complementary-alpha",
        )
        _write(
            client,
            headers,
            "beta",
            "Rust is memory-safe",
            key="complementary-beta",
        )

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["conflict_groups"] == 0
        assert status["summary"]["promoted"] == 2

        fast_results = _search(client, headers, region_id="rust", query="fast")["results"]
        assert any(
            item["content"] == "Rust is fast"
            and item["user_metadata"]["memory_type"] == "promoted"
            for item in fast_results
        )
        safe_results = _search(client, headers, region_id="rust", query="memory")["results"]
        assert any(
            item["content"] == "Rust is memory-safe"
            and item["user_metadata"]["memory_type"] == "promoted"
            for item in safe_results
        )


def test_model_claim_complementary_values_are_not_treated_as_conflict(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        _region(client, headers, "topic", "rust")
        _write(
            client,
            headers,
            "alpha",
            "Rust is fast",
            key="model-complementary-alpha",
        )
        _write(
            client,
            headers,
            "beta",
            "Rust is memory-safe",
            key="model-complementary-beta",
        )
        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = lambda content, context: {
            "category": "topic",
            "confidence": 0.95,
            "explicit": True,
            "topic_id": "rust",
            "claim_subject": "rust",
            "claim_value": "fast" if "fast" in content else "memory-safe",
            "reason": "model-provided complementary claim",
        }

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["conflict_groups"] == 0
        assert status["summary"]["promoted"] == 2


def test_existing_manual_target_fact_is_checked_for_conflict(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "user", "user")
        _region(client, headers, "project", "alpha")
        _write(
            client,
            headers,
            "user",
            "I prefer light mode",
            key="manual-user-light",
            memory_type="preference",
        )
        _write(
            client,
            headers,
            "alpha",
            "I prefer dark mode",
            key="manual-user-dark",
            memory_type="preference",
        )

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["promoted"] == 0
        assert status["summary"]["conflict_groups"] == 1

        user_results = _search(client, headers, region_id="user", query="mode")["results"]
        contents = {item["content"] for item in user_results}
        assert "I prefer light mode" in contents
        assert "I prefer dark mode" not in contents
        history = client.get("/api/v1/libraries/alpha/history", headers=headers).json()
        assert len([item for item in history if item["action"] == "conflict"]) == 1


def test_model_generic_preference_conflicts_with_manual_target(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "user", "user")
        _region(client, headers, "project", "alpha")
        _write(
            client,
            headers,
            "user",
            "I prefer light mode",
            key="generic-manual-light",
            memory_type="preference",
        )
        _write(
            client,
            headers,
            "alpha",
            "I prefer dark mode",
            key="generic-project-dark",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = lambda content, context: {
            "category": "user",
            "confidence": 0.95,
            "explicit": True,
            "topic_id": None,
            "claim_subject": "user:preference",
            "claim_value": content.removeprefix("I prefer "),
            "reason": "generic preference claim",
        }

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["promoted"] == 0
        assert status["summary"]["conflict_groups"] == 1
        user_results = _search(client, headers, region_id="user", query="mode")["results"]
        contents = {item["content"] for item in user_results}
        assert "I prefer light mode" in contents
        assert "I prefer dark mode" not in contents


def test_corrected_target_rejects_late_source_without_merging(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        alpha_source = _write(
            client,
            headers,
            "alpha",
            "I prefer blue highlights",
            key="corrected-alpha",
            memory_type="preference",
        )
        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1
        target = next(
            item
            for item in _search(
                client,
                headers,
                region_id="user",
                query="blue",
            )["results"]
            if item["content"] == "I prefer blue highlights"
        )
        target_id = str(target["document_id"])
        revision = str(
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=target_id,
            )["version"]
        )
        _update_document(
            client,
            headers,
            region_id="user",
            path=target_id,
            content="I prefer green highlights",
            revision=revision,
            mode="overwrite",
            operation_id="correct-before-late-source",
        )
        beta_source = _write(
            client,
            headers,
            "beta",
            "I prefer blue highlights",
            key="corrected-beta",
            memory_type="preference",
        )

        second = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["beta"]},
        )
        assert second.status_code == 202, second.text
        second_status = _wait_promotion(client, headers, str(second.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 0
        assert second_status["summary"]["conflict_groups"] == 1

        corrected = next(
            item
            for item in _search(
                client,
                headers,
                region_id="user",
                query="green",
            )["results"]
            if item["content"] == "I prefer green highlights"
        )
        assert corrected["user_metadata"]["source_unit_ids"] == [alpha_source["id"]]
        assert beta_source["id"] not in corrected["user_metadata"]["source_unit_ids"]
        history = client.get("/api/v1/libraries/beta/history", headers=headers).json()
        assert len([item for item in history if item["action"] == "conflict"]) == 1


def test_model_generic_preference_corrected_target_still_writes_conflict(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        _write(
            client,
            headers,
            "alpha",
            "I prefer blue highlights",
            key="generic-corrected-alpha",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = lambda content, context: {
            "category": "user",
            "confidence": 0.95,
            "explicit": True,
            "topic_id": None,
            "claim_subject": "user:preference",
            "claim_value": content.removeprefix("I prefer "),
            "reason": "generic preference claim",
        }
        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1
        target = next(
            item
            for item in _search(
                client,
                headers,
                region_id="user",
                query="blue",
            )["results"]
            if item["content"] == "I prefer blue highlights"
        )
        target_id = str(target["document_id"])
        revision = str(
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=target_id,
            )["version"]
        )
        _update_document(
            client,
            headers,
            region_id="user",
            path=target_id,
            content="I prefer green highlights",
            revision=revision,
            mode="overwrite",
            operation_id="generic-correct-before-late-source",
        )
        _write(
            client,
            headers,
            "beta",
            "I prefer blue highlights",
            key="generic-corrected-beta",
            memory_type="preference",
        )

        second = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["beta"]},
        )
        assert second.status_code == 202, second.text
        second_status = _wait_promotion(client, headers, str(second.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 0
        assert second_status["summary"]["conflict_groups"] == 1
        history = client.get("/api/v1/libraries/beta/history", headers=headers).json()
        assert len([item for item in history if item["action"] == "conflict"]) == 1


def test_same_region_duplicates_are_not_independent_evidence(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    content = "I prefer compact diffs"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _write(
            client,
            headers,
            "alpha",
            content,
            key="same-region-a",
            memory_type="preference",
        )
        _write(
            client,
            headers,
            "alpha",
            content,
            key="same-region-b",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = lambda content, context: {
            "category": "user",
            "confidence": 0.5,
            "explicit": False,
            "topic_id": None,
            "claim_subject": "user:prefer:diffs",
            "claim_value": "compact diffs",
            "reason": "same-region duplicate is not independent",
        }

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["candidates"] == 2
        assert status["summary"]["promoted"] == 0

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM promotion_records").fetchone() == (0,)


def test_multi_source_groups_by_claim_after_punctuation_difference(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        first = _write(
            client,
            headers,
            "alpha",
            "I prefer compact diffs",
            key="claim-group-alpha",
            memory_type="preference",
        )
        second = _write(
            client,
            headers,
            "beta",
            "I prefer compact diffs.",
            key="claim-group-beta",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = lambda content, context: {
            "category": "user",
            "confidence": 0.5,
            "explicit": False,
            "topic_id": None,
            "claim_subject": "user:preference",
            "claim_value": "compact diffs",
            "reason": "same claim with punctuation variation",
        }

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["promoted"] == 1
        assert status["summary"]["evidence_groups"] == 1

        user_results = _search(client, headers, region_id="user", query="compact")["results"]
        promoted = next(
            item for item in user_results if item["content"].startswith("I prefer compact diffs")
        )
        assert set(promoted["user_metadata"]["source_unit_ids"]) == {first["id"], second["id"]}


def test_indentation_preference_conflict_is_detected(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        _write(
            client,
            headers,
            "alpha",
            "I prefer tabs",
            key="indentation-alpha",
            memory_type="preference",
        )
        _write(
            client,
            headers,
            "beta",
            "I prefer spaces",
            key="indentation-beta",
            memory_type="preference",
        )

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["promoted"] == 0
        assert status["summary"]["conflict_groups"] == 1

        for region_id in ("alpha", "beta"):
            history = client.get(
                f"/api/v1/libraries/{region_id}/history",
                headers=headers,
            ).json()
            assert len([item for item in history if item["action"] == "conflict"]) == 1


def test_sensitive_content_is_not_promoted_and_writes_audit(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    content = "password='correct-horse-battery-staple'"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        source = _write(
            client,
            headers,
            "demo",
            content,
            key="sensitive-promotion",
            memory_type="preference",
        )

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["promoted"] == 0
        assert status["summary"]["skipped"] == 1

        history = client.get("/api/v1/libraries/demo/history", headers=headers).json()
        sensitive = [item for item in history if item["action"] == "sensitive"]
        assert len(sensitive) == 1
        assert "sensitive content requires manual review" in sensitive[0]["detail"]
        source_results = _search(client, headers, region_id="demo", query="password")["results"]
        assert any(item["document_id"] == source["id"] for item in source_results)

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM promotion_records").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM governance_events WHERE action = 'promote'"
        ).fetchone() == (0,)


def test_promoted_unit_can_be_corrected_downweighted_and_forgotten(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    contents = {
        "blue": "I prefer blue highlights",
        "large": "I prefer large fonts",
        "tab": "I prefer tab indentation",
    }
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        sources = {
            query: _write(
                client,
                headers,
                "demo",
                content,
                key=f"revoke-{query}",
                memory_type="preference",
            )
            for query, content in contents.items()
        }
        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["promoted"] == 3

        targets = {
            query: next(
                item
                for item in _search(
                    client,
                    headers,
                    region_id="user",
                    query=query,
                )["results"]
                if item["content"] == contents[query]
            )
            for query in contents
        }
        corrected_id = str(targets["blue"]["document_id"])
        corrected_revision = str(
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=corrected_id,
            )["version"]
        )
        corrected = _update_document(
            client,
            headers,
            region_id="user",
            path=corrected_id,
            content="I prefer green highlights",
            revision=corrected_revision,
            mode="overwrite",
            operation_id="promotion-correct",
        )
        assert corrected["id"] == corrected_id
        assert (
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=corrected_id,
            )["content"]
            == "I prefer green highlights"
        )

        downweighted_id = str(targets["large"]["document_id"])
        downweighted_revision = str(
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=downweighted_id,
            )["version"]
        )
        downweighted = _delete_document(
            client,
            headers,
            region_id="user",
            path=downweighted_id,
            revision=downweighted_revision,
            mode="downweight",
            operation_id="promotion-downweight",
        )
        assert downweighted["status"] == "downweight"
        assert (
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=downweighted_id,
            )["lifecycle"]
            == "active"
        )

        forgotten_id = str(targets["tab"]["document_id"])
        forgotten_revision = str(
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=forgotten_id,
            )["version"]
        )
        forgotten = _delete_document(
            client,
            headers,
            region_id="user",
            path=forgotten_id,
            revision=forgotten_revision,
            mode="forget",
            operation_id="promotion-forget",
        )
        assert forgotten["status"] == "forget"
        assert (
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=forgotten_id,
            )["lifecycle"]
            == "forgotten"
        )

        for query, source in sources.items():
            source_results = _search(
                client,
                headers,
                region_id="demo",
                query=query,
            )["results"]
            assert any(item["document_id"] == source["id"] for item in source_results)

        history = client.get("/api/v1/libraries/user/history", headers=headers).json()
        actions = {item["action"] for item in history}
        assert {"overwrite", "delete:downweight", "delete:forget"}.issubset(actions)


def test_promotion_schema_migrates_existing_ticket08_database(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    maintenance = state_dir / "mem2" / "maintenance.sqlite3"
    maintenance.parent.mkdir(parents=True)
    (maintenance.parent / "VERSION").write_text(
        "jiuwen-memory-0.2.0\n", encoding="utf-8"
    )
    with sqlite3.connect(maintenance) as connection:
        connection.execute(
            """CREATE TABLE promotion_records (
               promotion_key TEXT PRIMARY KEY,
               job_id TEXT NOT NULL,
               source_unit_id TEXT NOT NULL,
               source_region_id TEXT NOT NULL,
               source_region_kind TEXT NOT NULL,
               target_region_id TEXT NOT NULL,
               target_region_kind TEXT NOT NULL,
               target_unit_id TEXT NOT NULL,
               classification TEXT NOT NULL,
               confidence REAL NOT NULL,
               audit_event_id INTEGER,
               created_at TEXT NOT NULL
           )"""
        )
        connection.execute(
            """INSERT INTO promotion_records
               (promotion_key, job_id, source_unit_id, source_region_id,
                source_region_kind, target_region_id, target_region_kind,
                target_unit_id, classification, confidence, audit_event_id,
                created_at)
               VALUES ('old-key', 'old-job', 'old-unit', 'old-region',
                       'project', 'user', 'user', 'old-target', 'user', 0.95,
                       NULL, '2026-10-01T00:00:00Z')"""
        )

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        _write(
            client,
            headers,
            "demo",
            "I prefer migration checks",
            key="promotion-migration",
            memory_type="preference",
        )
        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["status"] == "completed", status
        assert status["summary"]["promoted"] == 1

    with sqlite3.connect(maintenance) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(promotion_records)").fetchall()
        }
        assert {"evidence_kind", "source_unit_ids", "source_region_ids"}.issubset(columns)
        record = connection.execute(
            """SELECT evidence_kind, source_unit_ids, source_region_ids
               FROM promotion_records WHERE promotion_key = 'old-key'"""
        ).fetchone()
    assert record is not None
    assert record[0] == "explicit"
    assert json.loads(record[1]) == ["old-unit"]
    assert json.loads(record[2]) == ["old-region"]
