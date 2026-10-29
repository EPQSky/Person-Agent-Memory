from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.mem2_kernel import Mem2MemoryKernel


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir))


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


def _memories(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
) -> list[dict[str, object]]:
    response = client.get(
        f"/api/v1/libraries/{region_id}/memories?limit=100",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return list(response.json()["items"])


def _detail(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    memory_id: str,
) -> dict[str, object]:
    response = client.get(
        f"/api/v1/libraries/{region_id}/memories/{memory_id}",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


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
        assert promoted_user["user_metadata"]["memory_type"] == "preference"
        assert promoted_user["user_metadata"]["promotion_marker"] is True
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
        assert promoted_topic["user_metadata"]["memory_type"] == "domain_fact"
        assert promoted_topic["user_metadata"]["promotion_marker"] is True
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


def test_repeated_promotion_does_not_resurrect_forgotten_target(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    content = "I prefer durable archive navigation"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "demo")
        _write(
            client,
            headers,
            "demo",
            content,
            key="promotion-forgotten-target",
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
        user_results = _search(client, headers, region_id="user", query="archive")["results"]
        target = next(item for item in user_results if item["content"] == content)
        target_id = str(target["document_id"])
        target_revision = str(
            _inspect_document(
                client,
                headers,
                region_id="user",
                path=target_id,
            )["version"]
        )
        _delete_document(
            client,
            headers,
            region_id="user",
            path=target_id,
            revision=target_revision,
            mode="forget",
            operation_id="promotion-forget-target",
        )
        assert client.get(
            "/api/v1/libraries/user/memories",
            headers=headers,
        ).json()["items"] == []

        repeated = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["demo"]},
        )
        assert repeated.status_code == 202, repeated.text
        repeated_status = _wait_promotion(client, headers, str(repeated.json()["job_id"]))
        assert repeated_status["status"] == "completed", repeated_status
        assert repeated_status["summary"]["promoted"] == 0
        assert repeated_status["summary"]["skipped"] == 1
        assert client.get(
            "/api/v1/libraries/user/memories",
            headers=headers,
        ).json()["items"] == []
        tombstone = client.get(
            f"/api/v1/libraries/user/memories/{target_id}",
            headers=headers,
        )
        assert tombstone.status_code == 404, tombstone.text

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        record = connection.execute(
            "SELECT target_unit_id FROM promotion_records"
        ).fetchone()
    assert record == (target_id,)


def test_evolved_units_stay_diagnostic_and_do_not_promote(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    other_root = tmp_path / "other"
    other_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "user", "user")
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
        assert status["summary"]["candidates"] == 0
        assert status["summary"]["promoted"] == 0

        user_page = client.get(
            "/api/v1/libraries/user/memories",
            headers=headers,
            params={"q": "detailed"},
        )
        assert user_page.status_code == 200, user_page.text
        assert user_page.json()["items"] == []

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            rounds = client.get(
                "/api/v1/capture/rounds",
                headers=headers,
                params={"session_id": "promotion-session"},
            )
            assert rounds.status_code == 200, rounds.text
            if rounds.json() and rounds.json()[0]["status"] == "done":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("capture consolidation did not complete")

        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        evolved_units = [
            unit
            for scope in bundle.kv_store.scopes()
            for unit in kernel._units_in_scope(bundle, scope)
            if unit.user_metadata.get("memory_type") == "evolved"
            and unit.user_metadata.get("region_id") == source_region_id
        ]
        assert len(evolved_units) == 1
        assert "detailed explanations" in evolved_units[0].content


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
            and item["user_metadata"].get("promotion_marker") is True
            and item["user_metadata"]["memory_type"] != "promoted"
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
        source = _write(
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

        user_page = client.get(
            "/api/v1/libraries/user/memories",
            headers=headers,
            params={"q": content},
        )
        assert user_page.status_code == 200, user_page.text
        assert [item["id"] for item in user_page.json()["items"]] == [target_unit_id]
        detail = client.get(
            f"/api/v1/libraries/user/memories/{target_unit_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["user_metadata"]["source_unit_ids"] == [source["id"]]
        assert detail.json()["provenance"] == [source["id"]]

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
            "alpha",
            content,
            key="multi-source-alpha-two",
            memory_type="preference",
        )
        third = _write(
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
        assert summary["candidates"] == 3
        assert summary["promoted"] == 1
        assert summary["skipped"] == 0
        assert summary["failed"] == 0
        assert summary["conflict_groups"] == 0
        assert summary["evidence_groups"] == 1

        user_page = client.get(
            "/api/v1/libraries/user/memories",
            headers=headers,
            params={"q": "compact", "memory_type": "preference"},
        )
        assert user_page.status_code == 200, user_page.text
        promoted = next(
            item for item in user_page.json()["items"] if item["summary"] == content
        )
        promoted_detail = client.get(
            f"/api/v1/libraries/user/memories/{promoted['id']}",
            headers=headers,
        )
        assert promoted_detail.status_code == 200, promoted_detail.text
        metadata = promoted_detail.json()["user_metadata"]
        assert metadata["promotion_evidence_kind"] == "multi_source"
        assert metadata["promotion_evidence_count"] == 3
        assert set(metadata["source_unit_ids"]) == {
            first["id"],
            second["id"],
            third["id"],
        }
        assert set(metadata["source_region_ids"]) == {"alpha", "beta"}
        assert set(metadata["provenance"]) == {
            first["id"],
            second["id"],
            third["id"],
        }
        source_unit_regions = json.loads(metadata["source_unit_regions"])
        assert {
            (item["unit_id"], item["region_id"]) for item in source_unit_regions
        } == {
            (first["id"], "alpha"),
            (second["id"], "alpha"),
            (third["id"], "beta"),
        }
        detail = client.get(
            f"/api/v1/libraries/user/memories/{promoted['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        evidence = {item["id"]: item for item in detail.json()["evidence"]}
        assert evidence[first["id"]]["region_id"] == "alpha"
        assert evidence[second["id"]]["region_id"] == "alpha"
        assert evidence[third["id"]]["region_id"] == "beta"
        assert all(evidence[source["id"]]["resolved"] is True for source in (first, second, third))
        for region_id, source in (("alpha", first), ("beta", third)):
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
    assert set(json.loads(records[0][1])) == {
        first["id"],
        second["id"],
        third["id"],
    }


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
            and item["user_metadata"].get("promotion_marker") is True
            and item["user_metadata"]["memory_type"] != "promoted"
            for item in fast_results
        )
        safe_results = _search(client, headers, region_id="rust", query="memory")["results"]
        assert any(
            item["content"] == "Rust is memory-safe"
            and item["user_metadata"].get("promotion_marker") is True
            and item["user_metadata"]["memory_type"] != "promoted"
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


def test_single_value_package_manager_conflict_preserves_first_target_and_audits_once(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        _write(
            client,
            headers,
            "alpha",
            "The build uses pnpm.",
            key="package-manager-pnpm",
            memory_type="constraint",
        )
        _write(
            client,
            headers,
            "beta",
            "The build uses npm.",
            key="package-manager-npm",
            memory_type="constraint",
        )
        kernel = client.app.state.memory_kernel

        def classify(content: str, context: dict[str, object]) -> dict[str, object]:
            del context
            value = "pnpm" if "pnpm" in content else "npm"
            return {
                "category": "user",
                "confidence": 0.99,
                "explicit": True,
                "topic_id": None,
                "claim_subject": "build package manager",
                "claim_value": value,
                "reason": "single-valued build tooling choice",
            }

        kernel._promotion_classifier = classify
        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1

        second = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["beta"]},
        )
        assert second.status_code == 202, second.text
        second_status = _wait_promotion(client, headers, str(second.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 0
        assert second_status["summary"]["conflict_groups"] == 1

        active_user = [
            item
            for item in _memories(client, headers, "user")
            if item["lifecycle"] == "active"
        ]
        assert [item["summary"] for item in active_user] == ["The build uses pnpm."]
        assert "The build uses npm." not in {item["summary"] for item in active_user}

        repeated = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["beta"]},
        )
        assert repeated.status_code == 202, repeated.text
        repeated_status = _wait_promotion(client, headers, str(repeated.json()["job_id"]))
        assert repeated_status["summary"]["promoted"] == 0
        assert repeated_status["summary"]["conflict_groups"] == 1

        history = client.get("/api/v1/libraries/beta/history", headers=headers).json()
        conflicts = [item for item in history if item["action"] == "conflict"]
        assert len(conflicts) == 1
        assert "pnpm" in conflicts[0]["detail"]
        assert "npm" in conflicts[0]["detail"]


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


def test_equivalent_preference_values_merge_across_sources_and_repeat_idempotently(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        dark = _write(
            client,
            headers,
            "alpha",
            "I prefer dark.",
            key="equivalent-dark",
            memory_type="preference",
        )
        dark_mode = _write(
            client,
            headers,
            "beta",
            "I prefer dark mode.",
            key="equivalent-dark-mode",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = lambda content, context: {
            "category": "user",
            "confidence": 0.99,
            "explicit": True,
            "topic_id": None,
            "claim_subject": "user:preference:theme",
            "claim_value": "dark mode" if "dark mode" in content else "dark",
            "reason": "equivalent theme preference values",
        }

        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1
        assert first_status["summary"]["conflict_groups"] == 0

        promoted = [
            item
            for item in _search(client, headers, region_id="user", query="dark")["results"]
        ]
        assert len(promoted) == 1
        metadata = promoted[0]["user_metadata"]
        assert set(metadata["source_unit_ids"]) == {dark["id"], dark_mode["id"]}
        assert metadata["promotion_evidence_kind"] == "multi_source"

        second = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert second.status_code == 202, second.text
        second_status = _wait_promotion(client, headers, str(second.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 0
        assert second_status["summary"]["conflict_groups"] == 0

        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_records"
            ).fetchone() == (1,)
            assert connection.execute(
                """SELECT COUNT(*) FROM governance_events
                   WHERE actor = 'platform:auto-promotion' AND action = 'promote'"""
            ).fetchone() == (1,)


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


def test_promotion_schema_migration_is_bounded_and_batched(tmp_path: Path) -> None:
    maintenance = tmp_path / "state" / "mem2" / "maintenance.sqlite3"
    maintenance.parent.mkdir(parents=True)
    with sqlite3.connect(maintenance) as connection:
        connection.execute(
            """CREATE TABLE memory_browser_index (
               region_id TEXT NOT NULL,
               unit_id TEXT NOT NULL,
               PRIMARY KEY (region_id, unit_id)
               )"""
        )
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
               evidence_kind TEXT NOT NULL DEFAULT 'explicit',
               source_unit_ids TEXT NOT NULL DEFAULT '[]',
               source_region_ids TEXT NOT NULL DEFAULT '[]',
               created_at TEXT NOT NULL
               )"""
        )
        records: list[tuple[object, ...]] = []
        indexes: list[tuple[str, str]] = []
        for index in range(250):
            unit_id = f"source-{index:03d}"
            region_id = "alpha" if index % 2 == 0 else "beta"
            records.append(
                (
                    f"promotion-{index:03d}",
                    "job",
                    unit_id,
                    region_id,
                    "topic",
                    "user",
                    "user",
                    f"target-{index:03d}",
                    "user",
                    0.95,
                    None,
                    "explicit",
                    json.dumps([unit_id]),
                    json.dumps([region_id]),
                    "2026-10-01T00:00:00Z",
                )
            )
            indexes.append((region_id, unit_id))
        connection.executemany(
            """INSERT INTO promotion_records
               (promotion_key, job_id, source_unit_id, source_region_id,
                source_region_kind, target_region_id, target_region_kind,
                target_unit_id, classification, confidence, audit_event_id,
                evidence_kind, source_unit_ids, source_region_ids, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            records,
        )
        connection.executemany(
            "INSERT INTO memory_browser_index (region_id, unit_id) VALUES (?, ?)",
            indexes,
        )
        connection.commit()

        batches = Mem2MemoryKernel._ensure_promotion_schema(
            connection,
            batch_size=100,
        )

        assert batches == [100, 100, 50]
        assert max(batches) <= 100
        rows = connection.execute(
            """SELECT source_unit_id, source_region_id, source_unit_regions
               FROM promotion_records ORDER BY promotion_key"""
        ).fetchall()
    assert len(rows) == 250
    for unit_id, region_id, raw_mapping in rows:
        mapping = json.loads(str(raw_mapping))
        assert mapping == [{"unit_id": unit_id, "region_id": region_id}]


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


def test_same_claim_identity_merges_different_wording_into_one_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        alpha_source = _write(
            client,
            headers,
            "alpha",
            "I prefer dark mode.",
            key="claim-identity-alpha",
            memory_type="preference",
        )

        def classify(content: str, context: dict[str, object]) -> dict[str, object]:
            del content, context
            return {
                "category": "user",
                "confidence": 0.99,
                "explicit": True,
                "topic_id": None,
                "claim_subject": "user:preference:theme",
                "claim_value": "dark",
                "reason": "same normalized claim identity",
            }

        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = classify
        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 1

        maintenance = state_dir / "mem2" / "maintenance.sqlite3"
        with sqlite3.connect(maintenance) as connection:
            first_record = connection.execute(
                """SELECT promotion_key, target_unit_id, audit_event_id
                   FROM promotion_records"""
            ).fetchone()
            assert first_record is not None
            original_key = str(first_record[0])
            original_target_id = str(first_record[1])
            original_audit_id = int(first_record[2])
            connection.execute("DELETE FROM promotion_records")

        captured_outbox: dict[str, object] = {}
        original_complete = kernel._complete_promotion_outbox

        def capture_and_complete(promotion_key: str) -> None:
            with sqlite3.connect(maintenance) as connection:
                row = connection.execute(
                    "SELECT payload FROM promotion_outbox WHERE promotion_key = ?",
                    (promotion_key,),
                ).fetchone()
            if row is not None:
                captured_outbox["payload"] = json.loads(str(row[0]))
            original_complete(promotion_key)

        monkeypatch.setattr(
            kernel,
            "_complete_promotion_outbox",
            capture_and_complete,
        )

        beta_source = _write(
            client,
            headers,
            "beta",
            "The user prefers dark mode.",
            key="claim-identity-beta",
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

        user_results = _search(client, headers, region_id="user", query="dark")["results"]
        promoted = [
            item
            for item in user_results
            if item["content"] in {"I prefer dark mode.", "The user prefers dark mode."}
        ]
        assert len(promoted) == 1
        assert promoted[0]["document_id"] == original_target_id
        metadata = promoted[0]["user_metadata"]
        assert set(metadata["source_unit_ids"]) == {
            alpha_source["id"],
            beta_source["id"],
        }
        assert set(metadata["source_region_ids"]) == {"alpha", "beta"}
        history = client.get("/api/v1/libraries/user/history", headers=headers).json()
        promote_audits = [item for item in history if item["action"] == "promote"]
        assert len(promote_audits) == 1
        assert promote_audits[0]["id"] == f"governance-{original_audit_id}"

    with sqlite3.connect(maintenance) as connection:
        repaired = connection.execute(
            """SELECT promotion_key, target_unit_id, audit_event_id
               FROM promotion_records"""
        ).fetchone()
        assert repaired is not None
        repaired_key = str(repaired[0])
        assert str(repaired[1]) == original_target_id
        assert int(repaired[2]) == original_audit_id
        assert repaired_key == original_key
        assert connection.execute("SELECT COUNT(*) FROM promotion_records").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM promotion_outbox").fetchone() == (0,)

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        detail = client.get(
            f"/api/v1/libraries/user/memories/{original_target_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["user_metadata"]["promotion_key"] == repaired_key

    payload = captured_outbox.get("payload")
    assert isinstance(payload, dict)
    target = payload.get("target")
    assert isinstance(target, dict)
    target_metadata = target.get("user_metadata")
    assert isinstance(target_metadata, dict)
    assert target_metadata["promotion_key"] == repaired_key


def test_superseded_promotion_target_is_repaired_before_merging_same_claim(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "alpha")
        _region(client, headers, "project", "beta")
        alpha_source = _write(
            client,
            headers,
            "alpha",
            "I prefer dark mode.",
            key="supersede-claim-alpha",
            memory_type="preference",
        )

        def classify(content: str, context: dict[str, object]) -> dict[str, object]:
            del context
            value = "light" if "light" in content else "dark"
            return {
                "category": "user",
                "confidence": 0.99,
                "explicit": True,
                "topic_id": None,
                "claim_subject": "user:preference:theme",
                "claim_value": value,
                "reason": "theme preference",
            }

        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = classify
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
            for item in _search(client, headers, region_id="user", query="dark")["results"]
            if item["content"] == "I prefer dark mode."
        )
        old_target_id = str(target["document_id"])
        inspected = _inspect_document(
            client,
            headers,
            region_id="user",
            path=old_target_id,
        )
        corrected = client.put(
            "/api/v1/libraries/user/document",
            headers=headers,
            json={
                "path": old_target_id,
                "content": "I prefer light mode.",
                "expected_source_version": inspected["version"],
                "operation_id": "supersede-promotion-target",
                "actor_type": "user",
                "source": "manual-supersede-test",
            },
        )
        assert corrected.status_code == 200, corrected.text
        corrected_id = str(corrected.json()["id"])
        assert corrected_id != old_target_id

        beta_source = _write(
            client,
            headers,
            "beta",
            "I prefer light mode.",
            key="supersede-claim-beta",
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

        active = [
            item
            for item in _memories(client, headers, "user")
            if item["lifecycle"] == "active"
        ]
        assert [item["id"] for item in active] == [corrected_id]
        detail = _detail(client, headers, "user", corrected_id)
        assert detail["content"] == "I prefer light mode."
        assert set(detail["user_metadata"]["source_unit_ids"]) == {
            alpha_source["id"],
            beta_source["id"],
        }

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        record = connection.execute(
            """SELECT promotion_key, target_unit_id, audit_event_id
               FROM promotion_records"""
        ).fetchone()
        assert record is not None
        assert str(record[1]) == corrected_id
        audit = connection.execute(
            "SELECT unit_id, after_content FROM governance_events WHERE id = ?",
            (int(record[2]),),
        ).fetchone()
        assert audit == (corrected_id, "I prefer light mode.")


def test_explicit_preference_subject_conflicts_with_ordinary_user_memory(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "user", "user")
        _region(client, headers, "project", "alpha")
        _write(
            client,
            headers,
            "user",
            "I prefer dark mode.",
            key="ordinary-compatible-dark",
            memory_type="preference",
        )
        source = _write(
            client,
            headers,
            "alpha",
            "I prefer light mode.",
            key="ordinary-compatible-light",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        kernel._promotion_classifier = lambda content, context: {
            "category": "user",
            "confidence": 0.99,
            "explicit": True,
            "topic_id": None,
            "claim_subject": "user:preference:theme",
            "claim_value": "light",
            "reason": "compatible preference subject",
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

        active = [
            item
            for item in _memories(client, headers, "user")
            if item["lifecycle"] == "active"
        ]
        assert [item["summary"] for item in active] == ["I prefer dark mode."]
        assert source["id"] not in " ".join(str(item["id"]) for item in active)
        history = client.get("/api/v1/libraries/alpha/history", headers=headers).json()
        assert len([item for item in history if item["action"] == "conflict"]) == 1
