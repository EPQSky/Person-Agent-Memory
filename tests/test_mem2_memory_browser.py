from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from jiuwen_memory.api import Modality, Scope
from jiuwen_memory.common.audit.base import AuditEvent
from jiuwen_memory.common.type_def import LifecycleState, memory_key
from jiuwen_memory.common.type_def.memory_codec import dumps
from jiuwen_memory.storage.kv import load_units

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.mem2_kernel import _scope_for_region, _security_for
from personal_agent_memory.memory_kernel import MemoryKernelUnavailableError
from personal_agent_memory.storage.audit import SQLiteAuditLogger


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir))


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _region(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    *,
    kind: str = "topic",
    session_id: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "kind": kind,
        "name": region_id,
        "region_id": region_id,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json=payload,
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
    source_references: list[str] | None = None,
) -> dict[str, object]:
    response = client.post(
        "/api/v1/memories",
        headers=headers,
        json={
            "region_id": region_id,
            "content": content,
            "memory_type": memory_type,
            "source_references": source_references or [f"source:{key}"],
            "actor": "test",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _direct_write(
    client: TestClient,
    region_id: str,
    content: str,
    *,
    memory_type: str = "domain_fact",
    source_references: list[str] | None = None,
    updated_at: datetime | None = None,
    lifecycle: LifecycleState = LifecycleState.ACTIVE,
    provenance: list[str] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> str:
    kernel = client.app.state.memory_kernel
    bundle = kernel._require_bundle()
    region = kernel._regions.require_region(region_id)
    scope = _scope_for_region(region)
    units = bundle.runtime.api.add(
        content,
        scope,
        Modality.TEXT,
        security=_security_for(scope),
        user_metadata={
            "memory_type": memory_type,
            "source_references": source_references or ["source:direct"],
            **dict(extra_metadata or {}),
        },
    )
    assert units
    unit = units[0]
    unit.lifecycle = lifecycle
    if updated_at is not None:
        unit.temporal.t_ingest = updated_at
    if provenance is not None:
        unit.provenance = provenance
        unit.user_metadata["source_unit_ids"] = provenance
    bundle.kv_store.update(scope, memory_key(unit.id), dumps(unit))
    kernel._refresh_browser_index_ids(bundle, region, scope, [unit.id])
    return unit.id


def _capture(
    client: TestClient,
    headers: dict[str, str],
    content: str,
    *,
    event_id: str,
    topic_id: str = "browser",
) -> str:
    response = client.post(
        "/api/v1/capture/events",
        headers=headers,
        json={
            "event_id": event_id,
            "session_id": "browser-session",
            "turn_id": event_id,
            "event_kind": "user",
            "content": content,
            "occurred_at": "2026-10-27T00:00:00Z",
            "cwd": "/tmp/browser-project",
            "topic_id": topic_id,
        },
    )
    assert response.status_code == 202, response.text
    payload = response.json()
    return str(payload.get("memory_id") or payload.get("id"))


def _list(
    client: TestClient,
    headers: dict[str, str],
    region_id: str = "browser",
    **params: str,
) -> dict[str, object]:
    response = client.get(
        f"/api/v1/libraries/{region_id}/memories",
        headers=headers,
        params=params,
    )
    assert response.status_code == 200, response.text
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


def _forget(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    memory_id: str,
) -> None:
    detail = client.get(
        f"/api/v1/libraries/{region_id}/memories/{memory_id}",
        headers=headers,
    )
    assert detail.status_code == 200, detail.text
    revision = str(detail.json()["source_version"])
    preview = client.post(
        f"/api/v1/libraries/{region_id}/document/delete-preview",
        headers=headers,
        json={
            "path": memory_id,
            "expected_source_version": revision,
            "operation_id": f"{memory_id}-preview",
        },
    )
    assert preview.status_code == 200, preview.text
    deleted = client.request(
        "DELETE",
        f"/api/v1/libraries/{region_id}/document",
        headers=headers,
        json={
            "path": memory_id,
            "expected_source_version": revision,
            "preview_token": preview.json()["preview_token"],
            "operation_id": f"{memory_id}-forget",
            "actor_type": "user",
            "source": "test-browser-view",
            "mode": "forget",
        },
    )
    assert deleted.status_code == 200, deleted.text


def test_formal_memory_list_excludes_capture_evolved_and_forgotten(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        formal = _write(client, headers, "browser", "浏览器可见的正式记忆", key="formal")
        capture_id = _capture(
            client,
            headers,
            "原始用户提问不应出现在正式列表",
            event_id="capture-1",
        )
        _direct_write(client, "browser", "旧 evolved 原样拼接不应出现", memory_type="evolved")
        forgotten = _write(client, headers, "browser", "即将被遗忘的记忆", key="forgotten")
        _forget(client, headers, "browser", str(forgotten["id"]))

        page = _list(client, headers)
        ids = {str(item["id"]) for item in page["items"]}
        assert ids == {str(formal["id"])}
        assert capture_id not in ids
        assert forgotten["id"] not in ids
        assert page["next_cursor"] is None
        assert page["has_more"] is False
        assert page["page_size"] == 50


def test_forgotten_evidence_hides_source_content_in_target_detail(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        source_id = _direct_write(
            client,
            "browser",
            "FORGOTTEN-EVIDENCE-BODY-MUST-NOT-LEAK",
            memory_type="preference",
        )
        target_id = _direct_write(
            client,
            "browser",
            "目标记忆仍需展示来源元数据",
            memory_type="preference",
            provenance=[source_id],
            extra_metadata={
                "source_region_ids": ["browser"],
                "source_unit_regions": json.dumps(
                    [{"unit_id": source_id, "region_id": "browser"}]
                ),
            },
        )
        _forget(client, headers, "browser", source_id)

        detail = client.get(
            f"/api/v1/libraries/browser/memories/{target_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        evidence = next(item for item in detail.json()["evidence"] if item["id"] == source_id)
        assert evidence["resolved"] is True
        assert evidence["lifecycle"] == "forgotten"
        assert evidence["redacted"] is True
        assert evidence["content"] == ""
        assert evidence["region_id"] == "browser"
        assert evidence["deleted_at"]
        assert "FORGOTTEN-EVIDENCE-BODY-MUST-NOT-LEAK" not in json.dumps(
            evidence,
            ensure_ascii=False,
        )


def test_memory_list_pagination_is_stable_at_50_and_handles_ties_and_duplicates(
    tmp_path: Path, monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        base = datetime(2026, 10, 27, tzinfo=UTC)
        created: list[tuple[str, datetime]] = []
        for index in range(55):
            timestamp = base + timedelta(seconds=index)
            if index in {20, 21}:
                timestamp = base + timedelta(seconds=20, milliseconds=500)
            unit_id = _direct_write(
                client,
                "browser",
                f"分页记忆 {index:02d}",
                updated_at=timestamp,
                source_references=[f"source:{index}"],
            )
            created.append((unit_id, timestamp))

        kernel = client.app.state.memory_kernel
        original_pairs = kernel._region_unit_pairs

        def duplicate_pairs(bundle, region):
            pairs = original_pairs(bundle, region)
            return [*pairs, *pairs[:3]]

        monkeypatch.setattr(kernel, "_region_unit_pairs", duplicate_pairs)
        first = _list(client, headers)
        assert len(first["items"]) == 50
        assert len({item["id"] for item in first["items"]}) == 50
        assert first["has_more"] is True
        assert first["next_cursor"]

        first_ids = [str(item["id"]) for item in first["items"]]
        repeated = _list(client, headers, cursor=str(first["next_cursor"]))
        repeated_again = _list(client, headers, cursor=str(first["next_cursor"]))
        assert [item["id"] for item in repeated["items"]] == [
            item["id"] for item in repeated_again["items"]
        ]
        assert set(first_ids).isdisjoint({str(item["id"]) for item in repeated["items"]})
        assert len(repeated["items"]) == 5
        assert repeated["has_more"] is False
        assert repeated["next_cursor"] is None

        ordered_times = [str(item["updated_at"]) for item in first["items"]]
        assert ordered_times == sorted(ordered_times, reverse=True)
        tied = [
            str(item["id"])
            for item in first["items"]
            if item["updated_at"] == (base + timedelta(seconds=20, milliseconds=500)).isoformat()
        ]
        assert len(tied) == 2
        assert tied == sorted(tied, reverse=True)


def test_browser_index_rebuild_keeps_distinct_session_regions(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "session-one", kind="session", session_id="session-one")
        _region(client, headers, "session-two", kind="session", session_id="session-two")
        first = _write(
            client,
            headers,
            "session-one",
            "第一个会话区域的正式记忆",
            key="session-one-memory",
        )
        second = _write(
            client,
            headers,
            "session-two",
            "第二个会话区域的正式记忆",
            key="session-two-memory",
        )
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            connection.execute(
                "DELETE FROM memory_browser_index_meta WHERE key = 'schema_version'"
            )
            connection.commit()
        kernel._ensure_browser_index(bundle)

        first_page = _list(client, headers, region_id="session-one")
        second_page = _list(client, headers, region_id="session-two")
        assert [item["id"] for item in first_page["items"]] == [first["id"]]
        assert [item["id"] for item in second_page["items"]] == [second["id"]]


def test_memory_filters_summary_and_detail_contract(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        evidence_id = _capture(
            client,
            headers,
            "用户原文证据：项目使用 uv。",
            event_id="evidence-1",
        )
        long_content = "项目坚持使用 uv 管理依赖。 " * 30
        unit_id = _direct_write(
            client,
            "browser",
            long_content,
            memory_type="decision",
            source_references=["session:evidence-1", "project:README"],
            provenance=[evidence_id],
        )
        _write(
            client,
            headers,
            "browser",
            "用户偏好简体中文回复",
            key="preference",
            memory_type="preference",
        )
        _direct_write(
            client,
            "browser",
            "已经归档的约束",
            memory_type="constraint",
            lifecycle=LifecycleState.ARCHIVED,
        )

        filtered = _list(client, headers, q="uv", memory_type="decision", lifecycle="active")
        assert [item["id"] for item in filtered["items"]] == [unit_id]
        item = filtered["items"][0]
        assert item["summary"].endswith("…")
        assert len(item["summary"]) <= 161
        assert item["memory_type"] == "decision"
        assert item["lifecycle"] == "active"
        assert item["source_count"] == 2
        assert item["updated_at"]
        assert item["version"]

        preferences = _list(client, headers, memory_type="preference")
        assert [item["memory_type"] for item in preferences["items"]] == ["preference"]
        archived = _list(client, headers, lifecycle="archived")
        assert [item["lifecycle"] for item in archived["items"]] == ["archived"]

        detail = client.get(
            f"/api/v1/libraries/browser/memories/{unit_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        assert payload["content"] == long_content
        assert payload["region"] == {
            "region_id": "browser",
            "name": "browser",
            "kind": "topic",
        }
        assert payload["memory_type"] == "decision"
        assert payload["lifecycle"] == "active"
        assert payload["source_version"]
        assert payload["source_count"] == 2
        assert payload["source_references"] == ["session:evidence-1", "project:README"]
        assert payload["evidence"][0]["id"] == evidence_id
        assert "用户原文证据" in payload["evidence"][0]["content"]

        capture_detail = client.get(
            f"/api/v1/libraries/browser/memories/{evidence_id}",
            headers=headers,
        )
        assert capture_detail.status_code == 404


def test_memory_governance_updates_list_and_detail_audit_entry(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(
            client,
            headers,
            "browser",
            "更正前内容",
            key="governance-view",
            memory_type="constraint",
        )
        old_id = str(created["id"])
        detail = client.get(
            f"/api/v1/libraries/browser/memories/{old_id}",
            headers=headers,
        ).json()
        updated = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": old_id,
                "content": "更正后内容",
                "expected_source_version": detail["source_version"],
                "operation_id": "browser-view-update",
                "actor_type": "user",
                "source": "test-browser-view",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        new_id = str(updated.json()["id"])

        page = _list(client, headers)
        by_id = {str(item["id"]): item for item in page["items"]}
        assert by_id[old_id]["lifecycle"] == "superseded"
        assert by_id[new_id]["lifecycle"] == "active"
        new_detail = client.get(
            f"/api/v1/libraries/browser/memories/{new_id}",
            headers=headers,
        )
        assert new_detail.status_code == 200, new_detail.text
        assert new_detail.json()["content"] == "更正后内容"
        old_detail = client.get(
            f"/api/v1/libraries/browser/memories/{old_id}",
            headers=headers,
        )
        assert old_detail.status_code == 200, old_detail.text
        assert old_detail.json()["lifecycle"] == "superseded"
        assert old_detail.json()["content"] == "更正前内容"
        assert isinstance(old_detail.json()["evidence"], list)
        superseded_page = _list(client, headers, lifecycle="superseded")
        assert [item["id"] for item in superseded_page["items"]] == [old_id]

        history = client.get("/api/v1/libraries/browser/history", headers=headers)
        assert history.status_code == 200, history.text
        assert any(
            event["action"] == "supersede" and event["target_id"] in {old_id, new_id}
            for event in history.json()
        )


def test_forgetting_version_chain_redacts_history_and_hides_all_versions(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    secret = "SECRET-ORIGINAL-VERSION-CHAIN"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(client, headers, "browser", secret, key="chain-secret")
        first_id = str(created["id"])
        first_revision = str(
            client.get(
                f"/api/v1/libraries/browser/memories/{first_id}",
                headers=headers,
            ).json()["source_version"]
        )
        updated = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": first_id,
                "content": "SECOND-VERSION-BODY",
                "expected_source_version": first_revision,
                "operation_id": "chain-supersede",
                "actor_type": "user",
                "source": "chain-test",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        second_id = str(updated.json()["id"])
        _forget(client, headers, "browser", second_id)

        assert _list(client, headers)["items"] == []
        forgotten = client.get(
            "/api/v1/libraries/browser/forgotten",
            headers=headers,
        )
        assert forgotten.status_code == 200, forgotten.text
        assert [item["id"] for item in forgotten.json()] == [second_id]
        for version_id in (first_id, second_id):
            detail = client.get(
                f"/api/v1/libraries/browser/memories/{version_id}",
                headers=headers,
            )
            assert detail.status_code == 404, detail.text
            assert secret not in detail.text

        expected_actions = {"add", "supersede", "delete:forget"}
        for target_id in (first_id, second_id):
            response = client.get(
                "/api/v1/libraries/browser/history",
                headers=headers,
                params={"target_id": target_id, "limit": 100},
            )
            assert response.status_code == 200, response.text
            assert secret not in response.text
            events = response.json()
            assert {event["action"] for event in events} >= expected_actions
            assert {
                event["action"]
                for event in events
                if event.get("status") == "committed"
            } >= expected_actions
            assert all(
                not event.get("before_content") and not event.get("after_content")
                for event in events
            )

        regional = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"limit": 100},
        )
        assert regional.status_code == 200, regional.text
        assert secret not in regional.text
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            rows = connection.execute(
                """SELECT unit_id, logical_memory_id, lifecycle
                   FROM memory_browser_index WHERE region_id = 'browser'"""
            ).fetchall()
        assert {str(row[0]) for row in rows} == {first_id, second_id}
        assert len({str(row[1]) for row in rows}) == 1
        assert {str(row[2]) for row in rows} == {"forgotten"}


def test_archived_memory_detail_is_readable_and_filter_consistent(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(
            client,
            headers,
            "browser",
            "ARCHIVED-READ-ONLY-CONTENT",
            key="archived-read-only",
        )
        memory_id = str(created["id"])
        revision = str(
            client.get(
                f"/api/v1/libraries/browser/memories/{memory_id}",
                headers=headers,
            ).json()["source_version"]
        )
        preview = client.post(
            "/api/v1/libraries/browser/document/delete-preview",
            headers=headers,
            json={
                "path": memory_id,
                "expected_source_version": revision,
                "operation_id": "archived-read-only-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        archived = client.request(
            "DELETE",
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": memory_id,
                "expected_source_version": revision,
                "preview_token": preview.json()["preview_token"],
                "operation_id": "archived-read-only",
                "actor_type": "user",
                "source": "archived-read-only-test",
                "mode": "archive",
            },
        )
        assert archived.status_code == 200, archived.text
        detail = client.get(
            f"/api/v1/libraries/browser/memories/{memory_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["lifecycle"] == "archived"
        assert detail.json()["content"] == "ARCHIVED-READ-ONLY-CONTENT"
        assert isinstance(detail.json()["evidence"], list)
        filtered = _list(client, headers, lifecycle="archived")
        assert [item["id"] for item in filtered["items"]] == [memory_id]


def test_forget_uses_index_lineage_when_kv_metadata_is_missing(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    secret = "LEGACY-LINEAGE-SECRET"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(client, headers, "browser", secret, key="legacy-lineage")
        first_id = str(created["id"])
        first_revision = str(
            client.get(
                f"/api/v1/libraries/browser/memories/{first_id}",
                headers=headers,
            ).json()["source_version"]
        )
        updated = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": first_id,
                "content": "LEGACY-LINEAGE-NEW-BODY",
                "expected_source_version": first_revision,
                "operation_id": "legacy-lineage-supersede",
                "actor_type": "user",
                "source": "legacy-lineage",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        second_id = str(updated.json()["id"])
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        region = kernel._regions.require_region("browser")
        scope = _scope_for_region(region)
        units = list(load_units(bundle.kv_store, scope, [first_id, second_id]))
        assert len(units) == 2
        for unit in units:
            unit.user_metadata.pop("logical_memory_id", None)
            bundle.kv_store.update(scope, memory_key(unit.id), dumps(unit))
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            index_lineages = {
                str(row[0])
                for row in connection.execute(
                    """SELECT logical_memory_id FROM memory_browser_index
                       WHERE region_id = 'browser' AND unit_id IN (?, ?)""",
                    (first_id, second_id),
                ).fetchall()
            }
        assert len(index_lineages) == 1
        assert next(iter(index_lineages))

        authoritative = client.get(
            "/api/v1/libraries/browser/document",
            headers=headers,
            params={"path": second_id},
        )
        assert authoritative.status_code == 200, authoritative.text
        revision = str(authoritative.json()["version"])
        preview = client.post(
            "/api/v1/libraries/browser/document/delete-preview",
            headers=headers,
            json={
                "path": second_id,
                "expected_source_version": revision,
                "operation_id": "legacy-lineage-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["affected_units"] == 2
        forgotten = client.request(
            "DELETE",
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": second_id,
                "expected_source_version": revision,
                "preview_token": preview.json()["preview_token"],
                "operation_id": "legacy-lineage-forget",
                "actor_type": "user",
                "source": "legacy-lineage",
                "mode": "forget",
            },
        )
        assert forgotten.status_code == 200, forgotten.text
        assert forgotten.json()["affected_units"] == "2"
        assert _list(client, headers)["items"] == []
        for version_id in (first_id, second_id):
            detail = client.get(
                f"/api/v1/libraries/browser/memories/{version_id}",
                headers=headers,
            )
            assert detail.status_code == 404, detail.text
        history = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"target_id": second_id, "limit": 100},
        )
        assert history.status_code == 200, history.text
        assert secret not in history.text
        assert "LEGACY-LINEAGE-NEW-BODY" not in history.text
        assert {item["action"] for item in history.json()} >= {
            "add",
            "supersede",
            "delete:forget",
        }


def test_audit_uses_lineage_without_region_scan_or_blocking_event_loop(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    secret = "SECRET-NO-SCAN-HISTORY"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(client, headers, "browser", secret, key="no-scan-secret")
        first_id = str(created["id"])
        first_revision = str(
            client.get(
                f"/api/v1/libraries/browser/memories/{first_id}",
                headers=headers,
            ).json()["source_version"]
        )
        updated = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": first_id,
                "content": "SECOND-NO-SCAN-BODY",
                "expected_source_version": first_revision,
                "operation_id": "no-scan-supersede",
                "actor_type": "user",
                "source": "no-scan-test",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        second_id = str(updated.json()["id"])
        _forget(client, headers, "browser", second_id)

    app = _app(state_dir)

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            headers = _headers(state_dir)
            kernel = app.state.memory_kernel

            def forbidden_scan(*_args, **_kwargs):
                raise AssertionError("audit must not scan or load the whole region")

            kernel._region_unit_pairs = forbidden_scan
            kernel._units_in_scope = forbidden_scan
            original_audit = kernel.audit
            entered = threading.Event()

            def slow_audit(request):
                entered.set()
                time.sleep(0.8)
                return original_audit(request)

            kernel.audit = slow_audit
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                history_task = asyncio.create_task(
                    client.get(
                        "/api/v1/libraries/browser/history",
                        headers=headers,
                        params={"target_id": first_id, "limit": 100},
                    )
                )
                assert await asyncio.to_thread(entered.wait, 2.0)
                started = time.monotonic()
                live = await client.get("/health/live", headers=headers)
                elapsed = time.monotonic() - started
                history = await history_task

            assert live.status_code == 200, live.text
            assert elapsed < 0.4, f"audit blocked the event loop for {elapsed:.3f}s"
            assert history.status_code == 200, history.text
            assert secret not in history.text
            assert {event["action"] for event in history.json()} >= {
                "add",
                "supersede",
                "delete:forget",
            }

    asyncio.run(exercise())


def test_lineage_audit_migration_is_batched_and_uses_index(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(client, headers, "browser", "迁移正文一", key="lineage-migration")
        first_id = str(created["id"])
        revision = str(
            client.get(
                f"/api/v1/libraries/browser/memories/{first_id}",
                headers=headers,
            ).json()["source_version"]
        )
        updated = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": first_id,
                "content": "迁移正文二",
                "expected_source_version": revision,
                "operation_id": "lineage-migration-supersede",
                "actor_type": "user",
                "source": "lineage-migration",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        kernel = client.app.state.memory_kernel
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            connection.execute("UPDATE governance_events SET logical_memory_id = ''")
            connection.commit()
        batches = kernel._backfill_governance_lineage(batch_size=1)
        assert batches
        assert max(batches) == 1
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            lineages = {
                str(row[0])
                for row in connection.execute(
                    """SELECT logical_memory_id FROM governance_events
                       WHERE logical_memory_id != ''"""
                ).fetchall()
            }
        assert len(lineages) == 1


def test_memory_list_reports_empty_auth_failure_and_invalid_cursor(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "empty")
        empty = _list(client, headers, region_id="empty")
        assert empty == {
            "items": [],
            "next_cursor": None,
            "has_more": False,
            "page_size": 50,
        }

        unauthorized = client.get("/api/v1/libraries/empty/memories")
        assert unauthorized.status_code == 401

        invalid = client.get(
            "/api/v1/libraries/empty/memories",
            headers=headers,
            params={"cursor": "not-a-cursor"},
        )
        assert invalid.status_code == 422
        assert invalid.json()["detail"] == "invalid memory cursor"


def test_overwrite_and_downweight_refresh_updated_at_order(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        old_time = datetime(2025, 1, 1, tzinfo=UTC)
        memory_id = _direct_write(
            client,
            "browser",
            "覆盖前更新时间",
            updated_at=old_time,
        )
        detail = client.get(
            f"/api/v1/libraries/browser/memories/{memory_id}",
            headers=headers,
        ).json()
        initial = _list(client, headers)["items"][0]
        assert initial["updated_at"] == old_time.isoformat()

        updated = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": memory_id,
                "content": "覆盖后更新时间",
                "expected_source_version": detail["source_version"],
                "operation_id": "overwrite-time",
                "actor_type": "user",
                "source": "test-browser-view",
                "mode": "overwrite",
            },
        )
        assert updated.status_code == 200, updated.text
        after_overwrite = _list(client, headers)["items"][0]
        assert after_overwrite["updated_at"] != old_time.isoformat()
        assert after_overwrite["updated_at"] > old_time.isoformat()

        revision = client.get(
            f"/api/v1/libraries/browser/memories/{memory_id}",
            headers=headers,
        ).json()["source_version"]
        preview = client.post(
            "/api/v1/libraries/browser/document/delete-preview",
            headers=headers,
            json={
                "path": memory_id,
                "expected_source_version": revision,
                "operation_id": "downweight-time-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        downweighted = client.request(
            "DELETE",
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": memory_id,
                "expected_source_version": revision,
                "preview_token": preview.json()["preview_token"],
                "operation_id": "downweight-time",
                "actor_type": "user",
                "source": "test-browser-view",
                "mode": "downweight",
            },
        )
        assert downweighted.status_code == 200, downweighted.text
        after_downweight = _list(client, headers)["items"][0]
        assert after_downweight["updated_at"] > after_overwrite["updated_at"]


def test_large_library_list_and_detail_do_not_scan_all_units(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        real_id = _write(
            client,
            headers,
            "browser",
            "真实详情目标",
            key="large-real",
        )["id"]
        kernel = client.app.state.memory_kernel
        rows: list[tuple[object, ...]] = []
        for index in range(1_000):
            unit_id = f"bulk-{index:04d}"
            updated = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index)
            rows.append(
                (
                    "browser",
                    unit_id,
                        unit_id,
                    "browser",
                    "topic",
                    "local",
                    "topic:browser",
                    "local",
                    "personal-agent-memory",
                    "",
                    f"批量索引记忆 {index}",
                    f"批量索引记忆 {index}",
                    "domain_fact",
                    "native",
                    "active",
                    updated.isoformat(),
                    f"version-{index}",
                    "",
                    1,
                    '["source:bulk"]',
                    "[]",
                    "{}",
                )
            )
        kernel._write_browser_index_rows("browser", rows, [])

        def forbidden_scan(*_args, **_kwargs):
            raise AssertionError("browse endpoint performed a full region scan")

        monkeypatch.setattr(kernel, "_region_unit_pairs", forbidden_scan)
        monkeypatch.setattr(kernel, "_units_in_scope", forbidden_scan)
        first = _list(client, headers, limit="50")
        second = _list(client, headers, limit="50", cursor=str(first["next_cursor"]))
        assert len(first["items"]) == 50
        assert len(second["items"]) == 50
        detail = client.get(
            f"/api/v1/libraries/browser/memories/{real_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["content"] == "真实详情目标"



def test_memory_list_q_limit_and_unknown_parameters(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        for index in range(3):
            _write(
                client,
                headers,
                "browser",
                f"NEEDLE-{index} 可检索契约",
                key=f"contract-{index}",
            )
        contract = client.get(
            "/api/v1/libraries/browser/memories",
            headers=headers,
            params={"q": "NEEDLE", "limit": 1},
        )
        assert contract.status_code == 200, contract.text
        payload = contract.json()
        assert payload["page_size"] == 1
        assert len(payload["items"]) == 1
        assert payload["has_more"] is True
        assert "NEEDLE" in payload["items"][0]["summary"]

        for params in ({"limit": 0}, {"limit": 101}, {"query": "old"}, {"unknown": "x"}):
            invalid = client.get(
                "/api/v1/libraries/browser/memories",
                headers=headers,
                params=params,
            )
            assert invalid.status_code == 422, (params, invalid.text)


def test_promotion_preserves_semantic_type_and_cross_region_capture_evidence(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "promo-source")
        _region(client, headers, "user", kind="user")
        capture_id = _capture(
            client,
            headers,
            "I prefer concise release notes",
            event_id="promotion-capture",
            topic_id="promo-source",
        )
        source_id = _direct_write(
            client,
            "promo-source",
            "I prefer concise release notes",
            memory_type="preference",
            source_references=["source:release-notes"],
            provenance=[capture_id],
        )
        legacy_id = _direct_write(
            client,
            "user",
            "I prefer concise release notes",
            memory_type="promoted",
            provenance=[source_id],
            extra_metadata={"source_region_ids": ["promo-source"]},
        )
        unresolved_id = _direct_write(
            client,
            "user",
            "Legacy promoted without source",
            memory_type="promoted",
            provenance=["missing-source"],
            extra_metadata={"source_region_ids": ["promo-source"]},
        )

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["promo-source"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["status"] == "completed", status
        assert status["summary"]["promoted"] == 1

        promoted = _list(client, headers, region_id="user", memory_type="preference")
        promoted_ids = {item["id"] for item in promoted["items"]}
        assert legacy_id in promoted_ids
        assert all(item["memory_type"] != "promoted" for item in promoted["items"])
        assert unresolved_id not in promoted_ids
        new_promoted = next(
            item for item in promoted["items"] if item["id"] not in {legacy_id, unresolved_id}
        )

        detail = client.get(
            f"/api/v1/libraries/user/memories/{new_promoted['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        detail_payload = detail.json()
        assert detail_payload["memory_type"] == "preference"
        evidence = detail_payload["evidence"]
        capture_evidence = next(item for item in evidence if item["id"] == capture_id)
        assert capture_evidence["region_id"] == "promo-source"
        assert capture_evidence["session_id"] == "browser-session"
        assert capture_evidence["turn_id"] == "promotion-capture"
        assert capture_evidence["event_id"] == "promotion-capture"
        assert "I prefer concise release notes" in capture_evidence["content"]

        unresolved = _list(client, headers, region_id="user", q="Legacy promoted without source")
        assert unresolved["items"] == []


def test_legacy_promotion_source_regions_recover_explicit_cross_region_mapping(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "alpha")
        _region(client, headers, "beta")
        _region(client, headers, "user", kind="user")
        first = _write(
            client,
            headers,
            "alpha",
            "跨区域证据一",
            key="legacy-alpha-one",
            memory_type="preference",
        )
        second = _write(
            client,
            headers,
            "alpha",
            "跨区域证据二",
            key="legacy-alpha-two",
            memory_type="preference",
        )
        third = _write(
            client,
            headers,
            "beta",
            "跨区域证据三",
            key="legacy-beta-one",
            memory_type="preference",
        )
        source_ids = [str(first["id"]), str(second["id"]), str(third["id"])]
        target_id = _direct_write(
            client,
            "user",
            "跨区域偏好",
            memory_type="preference",
            provenance=source_ids,
            extra_metadata={
                "promotion_key": "legacy-cross-region",
                "promotion_marker": True,
                "source_region_ids": ["alpha", "beta"],
                "source_unit_regions": "[]",
            },
        )
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        target_scope = _scope_for_region(kernel._regions.require_region("user"))
        units = load_units(bundle.kv_store, target_scope, [target_id])
        assert units
        units[0].user_metadata.pop("source_unit_regions", None)
        bundle.kv_store.update(target_scope, memory_key(target_id), dumps(units[0]))

    maintenance = state_dir / "mem2" / "maintenance.sqlite3"
    with sqlite3.connect(maintenance) as connection:
        connection.execute("DROP TABLE promotion_records")
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
        connection.execute(
            """INSERT INTO promotion_records
               (promotion_key, job_id, source_unit_id, source_region_id,
                source_region_kind, target_region_id, target_region_kind,
                target_unit_id, classification, confidence, audit_event_id,
                evidence_kind, source_unit_ids, source_region_ids, created_at)
               VALUES ('legacy-cross-region', 'legacy-job', ?, 'alpha', 'topic',
                       'user', 'user', ?, 'user', 0.95, NULL, 'multi_source', ?,
                       ?, '2026-10-01T00:00:00Z')""",
            (
                source_ids[0],
                target_id,
                json.dumps(source_ids),
                json.dumps(["alpha", "beta"]),
            ),
        )
        connection.execute("DELETE FROM memory_browser_index")
        connection.execute(
            "DELETE FROM memory_browser_index_meta "
            "WHERE key IN ('schema_version', 'promotion_schema_version')"
        )
        connection.commit()

    with TestClient(_app(state_dir)) as restarted:
        headers = _headers(state_dir)
        page = _list(restarted, headers, region_id="user", memory_type="preference")
        assert [item["id"] for item in page["items"]] == [target_id]
        detail = restarted.get(
            f"/api/v1/libraries/user/memories/{target_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        evidence = {item["id"]: item for item in detail.json()["evidence"]}
        assert evidence[source_ids[0]]["region_id"] == "alpha"
        assert evidence[source_ids[1]]["region_id"] == "alpha"
        assert evidence[source_ids[2]]["region_id"] == "beta"
        assert all(evidence[source_id]["resolved"] is True for source_id in source_ids)

    with sqlite3.connect(maintenance) as connection:
        row = connection.execute(
            """SELECT source_unit_regions FROM promotion_records
               WHERE promotion_key = 'legacy-cross-region'"""
        ).fetchone()
    assert row is not None
    mapping = {
        (item["unit_id"], item["region_id"]) for item in json.loads(str(row[0]))
    }
    assert mapping == {
        (source_ids[0], "alpha"),
        (source_ids[1], "alpha"),
        (source_ids[2], "beta"),
    }


def test_target_audit_filter_finds_event_outside_latest_50(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        kernel = client.app.state.memory_kernel
        with kernel._maintenance_db() as connection, connection:
            connection.execute(
                """INSERT INTO governance_events
                   (region_id, unit_id, action, actor, source, operation_id,
                    before_content, after_content, detail, status, created_at)
                   VALUES ('browser', 'old-target', 'update', 'user', 'test',
                           'old-target-event', 'before', 'after', '{}', 'committed', ?)""",
                ("2026-01-01T00:00:00+00:00",),
            )
            connection.executemany(
                """INSERT INTO governance_events
                   (region_id, unit_id, action, actor, source, operation_id,
                    before_content, after_content, detail, status, created_at)
                   VALUES ('browser', ?, 'update', 'user', 'test', ?, '', '', '{}',
                           'committed', ?)""",
                [
                    (
                        f"newer-{index}",
                        f"newer-event-{index}",
                        f"2026-02-01T00:00:{index:02d}+00:00",
                    )
                    for index in range(60)
                ],
            )
        recent = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"limit": 50},
        )
        assert recent.status_code == 200, recent.text
        assert "old-target" not in {item["target_id"] for item in recent.json()}
        targeted = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"limit": 50, "target_id": "old-target"},
        )
        assert targeted.status_code == 200, targeted.text
        assert [item["target_id"] for item in targeted.json()] == ["old-target"]


def test_browser_index_survives_restart_and_governance_updates(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(client, headers, "browser", "重启前内容", key="restart-source")
        revision = client.get(
            f"/api/v1/libraries/browser/memories/{created['id']}",
            headers=headers,
        ).json()["source_version"]
        updated = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": created["id"],
                "content": "重启后应可见的更正",
                "expected_source_version": revision,
                "operation_id": "restart-update",
                "actor_type": "user",
                "source": "test-browser-view",
            },
        )
        assert updated.status_code == 200, updated.text
        new_id = str(updated.json()["id"])

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        page = _list(client, headers)
        by_id = {item["id"]: item for item in page["items"]}
        assert by_id[str(created["id"])]["lifecycle"] == "superseded"
        assert by_id[new_id]["lifecycle"] == "active"
        detail = client.get(
            f"/api/v1/libraries/browser/memories/{new_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["content"] == "重启后应可见的更正"


def test_restore_paths_refresh_browser_index(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        forgotten = _write(
            client,
            headers,
            "browser",
            "遗忘后可恢复",
            key="restore-forgotten",
        )
        _forget(client, headers, "browser", str(forgotten["id"]))
        assert _list(client, headers)["items"] == []

        restored = client.post(
            "/api/v1/libraries/browser/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": forgotten["id"],
                "commit": forgotten["id"],
                "operation_id": "restore-forgotten-index",
                "actor_type": "user",
                "source": "test-browser-view",
            },
        )
        assert restored.status_code == 200, restored.text
        restored_id = str(restored.json()["id"])
        assert {item["id"] for item in _list(client, headers)["items"]} == {restored_id}

        current = _write(
            client,
            headers,
            "browser",
            "当前版本",
            key="restore-history",
        )
        current_id = str(current["id"])
        revision = client.get(
            f"/api/v1/libraries/browser/memories/{current_id}",
            headers=headers,
        ).json()["source_version"]
        superseded = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": current_id,
                "content": "后续版本",
                "expected_source_version": revision,
                "operation_id": "history-before-restore",
                "actor_type": "user",
                "source": "test-browser-view",
            },
        )
        assert superseded.status_code == 200, superseded.text
        superseded_id = str(superseded.json()["id"])
        superseded_revision = client.get(
            f"/api/v1/libraries/browser/memories/{superseded_id}",
            headers=headers,
        ).json()["source_version"]
        history_restored = client.post(
            "/api/v1/libraries/browser/history/restore",
            headers=headers,
            json={
                "path": superseded_id,
                "commit": current_id,
                "expected_source_version": superseded_revision,
                "operation_id": "restore-history-index",
                "actor_type": "user",
                "source": "test-browser-view",
            },
        )
        assert history_restored.status_code == 200, history_restored.text
        restored_history_id = str(history_restored.json()["id"])
        by_id = {item["id"]: item for item in _list(client, headers)["items"]}
        assert by_id[restored_history_id]["lifecycle"] == "active"
        assert by_id[restored_history_id]["summary"] == "当前版本"


def test_restored_active_version_is_readable_with_forgotten_lineage_history(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(
            client,
            headers,
            "browser",
            "RESTORED-ACTIVE-CONTENT",
            key="restored-active",
        )
        original_id = str(created["id"])
        _forget(client, headers, "browser", original_id)
        restored = client.post(
            "/api/v1/libraries/browser/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": original_id,
                "commit": original_id,
                "operation_id": "restored-active-restore",
                "actor_type": "user",
                "source": "restored-active-test",
            },
        )
        assert restored.status_code == 200, restored.text
        restored_id = str(restored.json()["id"])

        page = _list(client, headers)
        assert [item["id"] for item in page["items"]] == [restored_id]
        active_detail = client.get(
            f"/api/v1/libraries/browser/memories/{restored_id}",
            headers=headers,
        )
        assert active_detail.status_code == 200, active_detail.text
        assert active_detail.json()["content"] == "RESTORED-ACTIVE-CONTENT"
        assert isinstance(active_detail.json()["evidence"], list)
        old_detail = client.get(
            f"/api/v1/libraries/browser/memories/{original_id}",
            headers=headers,
        )
        assert old_detail.status_code == 404, old_detail.text

        history = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"target_id": restored_id, "limit": 100},
        )
        assert history.status_code == 200, history.text
        assert {item["action"] for item in history.json()} >= {
            "add",
            "delete:forget",
            "restore:forgotten",
        }

        _forget(client, headers, "browser", restored_id)
        assert _list(client, headers)["items"] == []
        for version_id in (original_id, restored_id):
            response = client.get(
                f"/api/v1/libraries/browser/memories/{version_id}",
                headers=headers,
            )
            assert response.status_code == 404, response.text


def test_target_audit_filter_applies_before_runtime_limit(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        scope = Scope(
            org="local",
            space="topic:browser",
            user="local",
            agent="personal-agent-memory",
            session="",
        )
        logger = SQLiteAuditLogger(state_dir / "mem2" / "audit.sqlite3")
        try:
            logger.record(
                AuditEvent(
                    id="old-runtime-target",
                    actor=scope,
                    target=scope,
                    action="update",
                    target_id="old-runtime-target",
                    decision="allow",
                    occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            for index in range(60):
                logger.record(
                    AuditEvent(
                        id=f"new-runtime-{index}",
                        actor=scope,
                        target=scope,
                        action="write",
                        target_id=f"new-target-{index}",
                        decision="allow",
                        occurred_at=datetime(2026, 2, 1, tzinfo=UTC),
                    )
                )
        finally:
            logger.close()

        recent = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"limit": 50},
        )
        assert recent.status_code == 200, recent.text
        assert "old-runtime-target" not in {
            item["target_id"] for item in recent.json()
        }
        targeted = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"limit": 50, "target_id": "old-runtime-target"},
        )
        assert targeted.status_code == 200, targeted.text
        assert [item["target_id"] for item in targeted.json()] == [
            "old-runtime-target"
        ]


def test_target_audit_returns_latest_window_for_same_target(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        scope = Scope(
            org="local",
            space="topic:browser",
            user="local",
            agent="personal-agent-memory",
            session="",
        )
        logger = SQLiteAuditLogger(state_dir / "mem2" / "audit.sqlite3")
        try:
            for index in range(60):
                logger.record(
                    AuditEvent(
                        id=f"same-target-{index:02d}",
                        actor=scope,
                        target=scope,
                        action="update",
                        target_id="same-target",
                        decision="allow",
                        occurred_at=datetime(2026, 2, 1, tzinfo=UTC)
                        + timedelta(seconds=index),
                    )
                )
        finally:
            logger.close()

        targeted = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"limit": 10, "target_id": "same-target"},
        )
        assert targeted.status_code == 200, targeted.text
        assert [item["id"] for item in targeted.json()] == [
            f"same-target-{index:02d}" for index in range(59, 49, -1)
        ]


def test_promotion_index_failure_does_not_leave_skipped_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "promo-source")
        _region(client, headers, "user", kind="user")
        source = _write(
            client,
            headers,
            "promo-source",
            "I prefer durable promotion recovery",
            key="promotion-index-failure",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        target_scope = _scope_for_region(kernel._regions.require_region("user"))
        original_index = kernel._index_browser_units

        def fail_index(*_args, **_kwargs):
            raise MemoryKernelUnavailableError("forced browser index failure")

        monkeypatch.setattr(kernel, "_index_browser_units", fail_index)
        first = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["promo-source"]},
        )
        assert first.status_code == 202, first.text
        first_status = _wait_promotion(client, headers, str(first.json()["job_id"]))
        assert first_status["summary"]["promoted"] == 0
        assert first_status["summary"]["failed"] == 1
        targets = [
            unit
            for unit in kernel._units_in_scope(bundle, target_scope)
            if unit.user_metadata.get("promotion_key")
        ]
        assert len(targets) == 1
        target_id = targets[0].id
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_records"
            ).fetchone()[0] == 0

        monkeypatch.setattr(kernel, "_index_browser_units", original_index)
        second = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["promo-source"]},
        )
        assert second.status_code == 202, second.text
        second_status = _wait_promotion(client, headers, str(second.json()["job_id"]))
        assert second_status["summary"]["promoted"] == 1, second_status["summary"]
        assert second_status["summary"]["skipped"] == 0, second_status["summary"]
        promoted = _list(
            client,
            headers,
            region_id="user",
            memory_type="preference",
        )
        assert [item["id"] for item in promoted["items"]] == [target_id]
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            record = connection.execute(
                """SELECT source_unit_ids FROM promotion_records
                   WHERE target_unit_id = ?""",
                (target_id,),
            ).fetchone()
            outbox_count = connection.execute(
                "SELECT COUNT(*) FROM promotion_outbox"
            ).fetchone()[0]
        assert record is not None
        assert json.loads(str(record[0])) == [source["id"]]
        assert outbox_count == 0


def test_promotion_outbox_recovers_written_target_after_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "promo-source")
        _region(client, headers, "user", kind="user")
        source = _write(
            client,
            headers,
            "promo-source",
            "I prefer restart-safe promotion recovery",
            key="promotion-restart-recovery",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel

        def fail_index(*_args, **_kwargs):
            raise MemoryKernelUnavailableError("forced restart recovery index failure")

        monkeypatch.setattr(kernel, "_index_browser_units", fail_index)
        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["promo-source"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["failed"] == 1
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_outbox"
            ).fetchone()[0] == 1
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_records"
            ).fetchone()[0] == 0

    with TestClient(_app(state_dir)) as restarted:
        headers = _headers(state_dir)
        page = _list(restarted, headers, region_id="user", memory_type="preference")
        assert len(page["items"]) == 1
        target_id = str(page["items"][0]["id"])
        detail = restarted.get(
            f"/api/v1/libraries/user/memories/{target_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["source_count"] == page["items"][0]["source_count"]
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_outbox"
            ).fetchone()[0] == 0
            record = connection.execute(
                """SELECT target_unit_id, source_unit_ids FROM promotion_records"""
            ).fetchone()
        assert record is not None
        assert record[0] == target_id
        assert json.loads(str(record[1])) == [source["id"]]


def test_promotion_merge_index_failure_recovers_record_and_source_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    content = "I prefer consistent merge recovery"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "alpha")
        _region(client, headers, "beta")
        _region(client, headers, "user", kind="user")
        first_source = _write(
            client,
            headers,
            "alpha",
            content,
            key="merge-recovery-alpha",
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
        before_merge = _list(
            client,
            headers,
            region_id="user",
            memory_type="preference",
        )
        target_id = str(before_merge["items"][0]["id"])
        source_count_before_merge = int(before_merge["items"][0]["source_count"])

        second_source = _write(
            client,
            headers,
            "beta",
            content,
            key="merge-recovery-beta",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel
        original_index = kernel._index_browser_units
        failed_once = False

        def fail_merge_index(*args, **kwargs):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise MemoryKernelUnavailableError("forced merge index failure")
            return original_index(*args, **kwargs)

        monkeypatch.setattr(kernel, "_index_browser_units", fail_merge_index)
        failed = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert failed.status_code == 202, failed.text
        failed_status = _wait_promotion(
            client,
            headers,
            str(failed.json()["job_id"]),
        )
        assert failed_status["summary"]["failed"] == 1
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_outbox"
            ).fetchone()[0] == 1

        monkeypatch.setattr(kernel, "_index_browser_units", original_index)
        repeated = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert repeated.status_code == 202, repeated.text
        repeated_status = _wait_promotion(
            client,
            headers,
            str(repeated.json()["job_id"]),
        )
        assert repeated_status["status"] == "completed", repeated_status
        page = _list(
            client,
            headers,
            region_id="user",
            q="consistent merge recovery",
            memory_type="preference",
        )
        assert [item["id"] for item in page["items"]] == [target_id]
        assert page["items"][0]["source_count"] > source_count_before_merge
        detail = client.get(
            f"/api/v1/libraries/user/memories/{target_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["source_count"] == page["items"][0]["source_count"]
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            record = connection.execute(
                """SELECT source_unit_ids, source_region_ids
                   FROM promotion_records WHERE target_unit_id = ?""",
                (target_id,),
            ).fetchone()
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_outbox"
            ).fetchone()[0] == 0
        assert record is not None
        assert set(json.loads(str(record[0]))) == {
            first_source["id"],
            second_source["id"],
        }
        assert set(json.loads(str(record[1]))) == {"alpha", "beta"}


def test_promotion_outbox_recovers_sources_before_target_merge_is_applied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    content = "I prefer recovery before target merge"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "alpha")
        _region(client, headers, "beta")
        _region(client, headers, "user", kind="user")
        alpha_source = _write(
            client,
            headers,
            "alpha",
            content,
            key="early-merge-alpha",
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
        before = _list(client, headers, region_id="user", q="recovery before target")
        target_id = str(before["items"][0]["id"])

        beta_source = _write(
            client,
            headers,
            "beta",
            content,
            key="early-merge-beta",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel

        def fail_before_merge(*_args, **_kwargs):
            raise MemoryKernelUnavailableError("forced failure before target merge")

        monkeypatch.setattr(kernel, "_merge_promotion_sources", fail_before_merge)
        failed = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["alpha", "beta"]},
        )
        assert failed.status_code == 202, failed.text
        failed_status = _wait_promotion(client, headers, str(failed.json()["job_id"]))
        assert failed_status["summary"]["failed"] == 1
        detail_before = client.get(
            f"/api/v1/libraries/user/memories/{target_id}",
            headers=headers,
        )
        assert detail_before.status_code == 200, detail_before.text
        assert detail_before.json()["user_metadata"]["source_unit_ids"] == [alpha_source["id"]]
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_outbox"
            ).fetchone()[0] == 1

    with TestClient(_app(state_dir)) as restarted:
        headers = _headers(state_dir)
        page = _list(restarted, headers, region_id="user", q="recovery before target")
        assert [item["id"] for item in page["items"]] == [target_id]
        detail = restarted.get(
            f"/api/v1/libraries/user/memories/{target_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        expected_sources = {str(alpha_source["id"]), str(beta_source["id"])}
        assert set(payload["user_metadata"]["source_unit_ids"]) == expected_sources
        assert set(payload["provenance"]) == expected_sources
        assert {
            item["id"] for item in payload["evidence"] if item["resolved"] is True
        } >= expected_sources
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            record = connection.execute(
                """SELECT source_unit_ids, source_region_ids
                   FROM promotion_records WHERE target_unit_id = ?""",
                (target_id,),
            ).fetchone()
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_outbox"
            ).fetchone()[0] == 0
        assert record is not None
        assert set(json.loads(str(record[0]))) == expected_sources
        assert set(json.loads(str(record[1]))) == {"alpha", "beta"}


def test_browser_index_pending_recovers_create_update_and_delete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        kernel = client.app.state.memory_kernel
        original_index = kernel._index_browser_units

        def fail_index(*_args, **_kwargs):
            raise MemoryKernelUnavailableError("forced ordinary index failure")

        monkeypatch.setattr(kernel, "_index_browser_units", fail_index)
        create_failed = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "region_id": "browser",
                "content": "CREATE-PENDING-BODY",
                "memory_type": "domain_fact",
                "source_references": ["source:create-pending"],
                "actor": "test",
                "idempotency_key": "create-pending",
            },
        )
        assert create_failed.status_code == 201, create_failed.text
        assert create_failed.json()["browser_index_sync"] == "pending"
        pending_created_id = str(create_failed.json()["memory_id"])
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_browser_index_pending"
            ).fetchone()[0] == 1

        monkeypatch.setattr(kernel, "_index_browser_units", original_index)
        created_page = _list(client, headers, q="CREATE-PENDING-BODY")
        assert [item["id"] for item in created_page["items"]] == [pending_created_id]
        created_id = str(created_page["items"][0]["id"])
        repeated_create = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "region_id": "browser",
                "content": "CREATE-PENDING-BODY",
                "memory_type": "domain_fact",
                "source_references": ["source:create-pending"],
                "actor": "test",
                "idempotency_key": "create-pending",
            },
        )
        assert repeated_create.status_code == 201, repeated_create.text
        assert repeated_create.json()["status"] == "existing"
        recovered_items = _list(client, headers, q="CREATE-PENDING-BODY")["items"]
        assert [item["id"] for item in recovered_items] == [pending_created_id]
        created_detail = client.get(
            f"/api/v1/libraries/browser/memories/{created_id}",
            headers=headers,
        )
        assert created_detail.status_code == 200, created_detail.text

        revision = str(created_detail.json()["source_version"])
        monkeypatch.setattr(kernel, "_index_browser_units", fail_index)
        update_failed = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": created_id,
                "content": "UPDATE-PENDING-BODY",
                "expected_source_version": revision,
                "operation_id": "update-pending",
                "actor_type": "user",
                "source": "test-pending-index",
                "mode": "supersede",
            },
        )
        assert update_failed.status_code == 200, update_failed.text
        assert update_failed.json()["browser_index_sync"] == "pending"
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            update_event = connection.execute(
                """SELECT unit_id, status FROM governance_events
                   WHERE operation_id = 'update-pending' AND action = 'supersede'"""
            ).fetchone()
        assert update_event is not None
        assert update_event[1] == "committed"
        updated_authoritative_id = str(update_event[0])
        assert updated_authoritative_id != created_id
        authoritative_old = client.get(
            "/api/v1/libraries/browser/document",
            headers=headers,
            params={"path": created_id},
        )
        assert authoritative_old.status_code == 200, authoritative_old.text
        assert authoritative_old.json()["lifecycle"] == "superseded"

        monkeypatch.setattr(kernel, "_index_browser_units", original_index)
        updated_page = _list(client, headers, q="UPDATE-PENDING-BODY")
        assert len(updated_page["items"]) == 1
        updated_id = str(updated_page["items"][0]["id"])
        updated_detail = client.get(
            f"/api/v1/libraries/browser/memories/{updated_id}",
            headers=headers,
        )
        assert updated_detail.status_code == 200, updated_detail.text
        assert updated_detail.json()["content"] == "UPDATE-PENDING-BODY"
        assert updated_id == updated_authoritative_id
        update_history = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"target_id": updated_id},
        )
        assert update_history.status_code == 200, update_history.text
        assert any(
            item["action"] == "supersede" and item["status"] == "committed"
            for item in update_history.json()
        )
        assert updated_id not in {
            item["id"] for item in _list(client, headers, q="CREATE-PENDING-BODY")["items"]
        }

        delete_revision = str(updated_detail.json()["source_version"])
        preview = client.post(
            "/api/v1/libraries/browser/document/delete-preview",
            headers=headers,
            json={
                "path": updated_id,
                "expected_source_version": delete_revision,
                "operation_id": "delete-pending-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        monkeypatch.setattr(kernel, "_index_browser_units", fail_index)
        delete_failed = client.request(
            "DELETE",
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": updated_id,
                "expected_source_version": delete_revision,
                "preview_token": preview.json()["preview_token"],
                "operation_id": "delete-pending",
                "actor_type": "user",
                "source": "test-pending-index",
                "mode": "forget",
            },
        )
        assert delete_failed.status_code == 200, delete_failed.text
        assert delete_failed.json()["browser_index_sync"] == "pending"
        assert delete_failed.json()["affected_units"] == "2"
        assert set(delete_failed.json()["affected_unit_ids"]) == {
            created_id,
            updated_id,
        }
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            delete_event = connection.execute(
                """SELECT unit_id, status FROM governance_events
                   WHERE operation_id = 'delete-pending'"""
            ).fetchone()
        assert delete_event == (updated_id, "committed")
        delete_history = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"target_id": updated_id},
        )
        assert delete_history.status_code == 200, delete_history.text
        assert any(
            item["action"] == "delete:forget" and item["status"] == "committed"
            for item in delete_history.json()
        )

        monkeypatch.setattr(kernel, "_index_browser_units", original_index)
        assert _list(client, headers, q="UPDATE-PENDING-BODY")["items"] == []
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_browser_index_pending"
            ).fetchone()[0] == 0


def test_browser_index_pending_recovers_update_after_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(
            client,
            headers,
            "browser",
            "RESTART-PENDING-BEFORE",
            key="restart-pending",
        )
        detail = client.get(
            f"/api/v1/libraries/browser/memories/{created['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        kernel = client.app.state.memory_kernel

        def fail_index(*_args, **_kwargs):
            raise MemoryKernelUnavailableError("forced restart pending index failure")

        monkeypatch.setattr(kernel, "_index_browser_units", fail_index)
        failed = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": created["id"],
                "content": "RESTART-PENDING-AFTER",
                "expected_source_version": detail.json()["source_version"],
                "operation_id": "restart-pending-update",
                "actor_type": "user",
                "source": "test-pending-index",
                "mode": "supersede",
            },
        )
        assert failed.status_code == 200, failed.text
        assert failed.json()["browser_index_sync"] == "pending"
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_browser_index_pending"
            ).fetchone()[0] == 1

    with TestClient(_app(state_dir)) as restarted:
        headers = _headers(state_dir)
        page = _list(restarted, headers, q="RESTART-PENDING-AFTER")
        assert len(page["items"]) == 1
        detail = restarted.get(
            f"/api/v1/libraries/browser/memories/{page['items'][0]['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["content"] == "RESTART-PENDING-AFTER"
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_browser_index_pending"
            ).fetchone()[0] == 0


def test_scope_pending_recovers_new_supersede_target_before_id_registration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(
            client,
            headers,
            "browser",
            "SUPERSEDE-PENDING-BEFORE",
            key="supersede-pending",
        )
        detail = client.get(
            f"/api/v1/libraries/browser/memories/{created['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        kernel = client.app.state.memory_kernel

        def fail_id_registration(*_args, **_kwargs):
            raise MemoryKernelUnavailableError("forced failure before new id registration")

        monkeypatch.setattr(kernel, "_update_browser_index_pending", fail_id_registration)
        failed = client.put(
            "/api/v1/libraries/browser/document",
            headers=headers,
            json={
                "path": created["id"],
                "content": "SUPERSEDE-PENDING-AFTER",
                "expected_source_version": detail.json()["source_version"],
                "operation_id": "supersede-pending-update",
                "actor_type": "user",
                "source": "test-pending-index",
                "mode": "supersede",
            },
        )
        assert failed.status_code == 200, failed.text
        assert failed.json()["browser_index_sync"] == "pending"
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            pending = connection.execute(
                "SELECT unit_ids FROM memory_browser_index_pending"
            ).fetchone()
        assert pending == ("[]",)

    with TestClient(_app(state_dir)) as restarted:
        headers = _headers(state_dir)
        page = _list(restarted, headers, q="SUPERSEDE-PENDING-AFTER")
        assert len(page["items"]) == 1
        detail = restarted.get(
            f"/api/v1/libraries/browser/memories/{page['items'][0]['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["content"] == "SUPERSEDE-PENDING-AFTER"
        assert detail.json()["lifecycle"] == "active"


def test_delete_waits_for_index_sync_before_list_or_detail_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(
            client,
            headers,
            "browser",
            "CONCURRENT-DELETE-SECRET-BODY",
            key="concurrent-delete",
        )
        detail = client.get(
            f"/api/v1/libraries/browser/memories/{created['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        revision = str(detail.json()["source_version"])
        preview = client.post(
            "/api/v1/libraries/browser/document/delete-preview",
            headers=headers,
            json={
                "path": created["id"],
                "expected_source_version": revision,
                "operation_id": "concurrent-delete-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        kernel = client.app.state.memory_kernel
        index_entered = threading.Event()
        release_index = threading.Event()
        original_refresh = kernel._refresh_browser_index_ids

        def pause_after_kv_delete(*args, **kwargs):
            index_entered.set()
            assert release_index.wait(timeout=5.0)
            return original_refresh(*args, **kwargs)

        monkeypatch.setattr(kernel, "_refresh_browser_index_ids", pause_after_kv_delete)
        with ThreadPoolExecutor(max_workers=3) as executor:
            delete_future = executor.submit(
                client.request,
                "DELETE",
                "/api/v1/libraries/browser/document",
                headers=headers,
                json={
                    "path": created["id"],
                    "expected_source_version": revision,
                    "preview_token": preview.json()["preview_token"],
                    "operation_id": "concurrent-delete",
                    "actor_type": "user",
                    "source": "test-concurrent-delete",
                    "mode": "forget",
                },
            )
            assert index_entered.wait(timeout=5.0)
            list_future = executor.submit(
                client.get,
                "/api/v1/libraries/browser/memories",
                headers=headers,
            )
            detail_future = executor.submit(
                client.get,
                f"/api/v1/libraries/browser/memories/{created['id']}",
                headers=headers,
            )
            time.sleep(0.1)
            assert list_future.done() is False
            assert detail_future.done() is False
            release_index.set()
            deleted = delete_future.result(timeout=10.0)
            listed = list_future.result(timeout=10.0)
            read_detail = detail_future.result(timeout=10.0)

        assert deleted.status_code == 200, deleted.text
        assert listed.status_code == 200, listed.text
        assert [item for item in listed.json()["items"] if item["id"] == created["id"]] == []
        assert "CONCURRENT-DELETE-SECRET-BODY" not in listed.text
        assert read_detail.status_code == 404, read_detail.text
        assert "CONCURRENT-DELETE-SECRET-BODY" not in read_detail.text


def test_outbox_without_target_id_preserves_forgotten_promotion_tombstone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "promo-source")
        _region(client, headers, "user", kind="user")
        _write(
            client,
            headers,
            "promo-source",
            "I prefer tombstone-safe outbox recovery",
            key="outbox-forgotten",
            memory_type="preference",
        )
        kernel = client.app.state.memory_kernel

        def fail_target_registration(*_args, **_kwargs):
            raise MemoryKernelUnavailableError("forced outbox target registration failure")

        monkeypatch.setattr(kernel, "_set_promotion_outbox_target", fail_target_registration)
        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["promo-source"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["summary"]["failed"] == 1

        bundle = kernel._require_bundle()
        user_region = kernel._regions.require_region("user")
        user_scope = _scope_for_region(user_region)
        targets = [
            unit
            for unit in kernel._units_in_scope(bundle, user_scope)
            if unit.user_metadata.get("promotion_key")
        ]
        assert len(targets) == 1
        target_id = targets[0].id
        assert targets[0].lifecycle.value == "active"
        kernel._refresh_browser_index_ids(bundle, user_region, user_scope, [target_id])
        _forget(client, headers, "user", target_id)
        assert _list(client, headers, region_id="user")["items"] == []
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            outbox = connection.execute(
                "SELECT target_unit_id FROM promotion_outbox"
            ).fetchone()
        assert outbox == ("",)

    with TestClient(_app(state_dir)) as restarted:
        headers = _headers(state_dir)
        assert _list(restarted, headers, region_id="user")["items"] == []
        tombstone = restarted.get(
            "/api/v1/libraries/user/document",
            headers=headers,
            params={"path": target_id},
        )
        assert tombstone.status_code == 200, tombstone.text
        assert tombstone.json()["lifecycle"] == "forgotten"
        kernel = restarted.app.state.memory_kernel
        bundle = kernel._require_bundle()
        user_scope = _scope_for_region(kernel._regions.require_region("user"))
        promoted = [
            unit
            for unit in kernel._units_in_scope(bundle, user_scope)
            if unit.user_metadata.get("promotion_key")
        ]
        assert len(promoted) == 1
        assert promoted[0].id == target_id
        assert promoted[0].lifecycle.value == "forgotten"
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM promotion_outbox"
            ).fetchone()[0] == 0
            record = connection.execute(
                "SELECT target_unit_id FROM promotion_records"
            ).fetchone()
        assert record == (target_id,)


def test_pending_index_retry_is_scoped_to_current_region(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "region-a")
        _region(client, headers, "region-b")
        memory_a = _write(client, headers, "region-a", "REGION-A-BEFORE", key="region-a-memory")
        memory_b = _write(client, headers, "region-b", "REGION-B-BODY", key="region-b-memory")
        kernel = client.app.state.memory_kernel
        original_index = kernel._index_browser_units

        def fail_region_a(bundle, region, scope, units):
            if region.region_id == "region-a":
                raise MemoryKernelUnavailableError("forced region-a index failure")
            return original_index(bundle, region, scope, units)

        detail = client.get(
            f"/api/v1/libraries/region-a/memories/{memory_a['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        monkeypatch.setattr(kernel, "_index_browser_units", fail_region_a)
        updated = client.put(
            "/api/v1/libraries/region-a/document",
            headers=headers,
            json={
                "path": memory_a["id"],
                "content": "REGION-A-AFTER",
                "expected_source_version": detail.json()["source_version"],
                "operation_id": "region-a-pending-update",
                "actor_type": "user",
                "source": "test-region-scope",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["browser_index_sync"] == "pending"

        page_b = _list(client, headers, region_id="region-b")
        assert [item["id"] for item in page_b["items"]] == [memory_b["id"]]
        detail_b = client.get(
            f"/api/v1/libraries/region-b/memories/{memory_b['id']}",
            headers=headers,
        )
        assert detail_b.status_code == 200, detail_b.text
        assert detail_b.json()["content"] == "REGION-B-BODY"

        list_a = client.get("/api/v1/libraries/region-a/memories", headers=headers)
        assert list_a.status_code == 503, list_a.text
        monkeypatch.setattr(kernel, "_index_browser_units", original_index)
        recovered_a = _list(client, headers, region_id="region-a", q="REGION-A-AFTER")
        assert len(recovered_a["items"]) == 1


def test_session_region_audit_filters_target_session_within_shared_space(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "session-one", kind="session", session_id="session-one")
        _region(client, headers, "session-two", kind="session", session_id="session-two")
        first = _write(client, headers, "session-one", "SESSION-ONE-AUDIT", key="session-one")
        second = _write(client, headers, "session-two", "SESSION-TWO-AUDIT", key="session-two")

        history_one = client.get(
            "/api/v1/libraries/session-one/history",
            headers=headers,
            params={"limit": 100},
        )
        history_two = client.get(
            "/api/v1/libraries/session-two/history",
            headers=headers,
            params={"limit": 100},
        )
        assert history_one.status_code == 200, history_one.text
        assert history_two.status_code == 200, history_two.text
        assert len(history_one.json()) == 2
        assert len(history_two.json()) == 2
        assert {item["target_id"] for item in history_one.json()} == {"", first["id"]}
        assert {item["target_id"] for item in history_two.json()} == {"", second["id"]}


def test_new_memory_has_committed_targeted_add_audit(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(client, headers, "browser", "TARGETED-ADD-AUDIT", key="targeted-add")
        targeted = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"target_id": created["id"]},
        )
        assert targeted.status_code == 200, targeted.text
        assert any(
            item["action"] == "add"
            and item["target_id"] == created["id"]
            and item["status"] == "committed"
            for item in targeted.json()
        )
        regional = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"limit": 100},
        )
        assert regional.status_code == 200, regional.text
        assert any(
            item["action"] == "add" and item["target_id"] == created["id"]
            for item in regional.json()
        )


def test_update_and_promotion_concurrent_locks_complete_without_deadlock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "race", kind="project")
        created = _write(
            client,
            headers,
            "race",
            "I prefer deadlock-free promotions",
            key="deadlock-update",
            memory_type="preference",
        )
        detail = client.get(
            f"/api/v1/libraries/race/memories/{created['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        kernel = client.app.state.memory_kernel
        entered = threading.Event()
        release = threading.Event()
        original_begin = kernel._begin_browser_index_pending

        def delayed_begin(*args, **kwargs):
            entered.set()
            assert release.wait(timeout=5.0)
            return original_begin(*args, **kwargs)

        monkeypatch.setattr(kernel, "_begin_browser_index_pending", delayed_begin)
        with ThreadPoolExecutor(max_workers=2) as executor:
            update_future = executor.submit(
                client.put,
                "/api/v1/libraries/race/document",
                headers=headers,
                json={
                    "path": created["id"],
                    "content": "I prefer lock-order-safe promotions",
                    "expected_source_version": detail.json()["source_version"],
                    "operation_id": "deadlock-update",
                    "actor_type": "user",
                    "source": "test-deadlock",
                    "mode": "supersede",
                },
            )
            assert entered.wait(timeout=5.0)
            promotion_future = executor.submit(
                client.post,
                "/api/v1/promotions",
                headers=headers,
                json={"source_region_ids": ["race"]},
            )
            time.sleep(0.2)
            release.set()
            update_response = update_future.result(timeout=5.0)
            promotion_response = promotion_future.result(timeout=5.0)

        assert update_response.status_code == 200, update_response.text
        assert promotion_response.status_code == 202, promotion_response.text
        promotion_status = _wait_promotion(
            client,
            headers,
            str(promotion_response.json()["job_id"]),
            timeout=5.0,
        )
        assert promotion_status["status"] in {"completed", "failed"}


def test_list_and_detail_concurrent_with_graph_rebuild_complete_without_deadlock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(client, headers, "browser", "REBUILD-LOCK-BODY", key="rebuild-lock")
        kernel = client.app.state.memory_kernel
        entered = threading.Event()
        release = threading.Event()
        original_retry = kernel._retry_pending_browser_index

        def delayed_retry(*args, **kwargs):
            entered.set()
            assert release.wait(timeout=5.0)
            return original_retry(*args, **kwargs)

        monkeypatch.setattr(kernel, "_retry_pending_browser_index", delayed_retry)
        with ThreadPoolExecutor(max_workers=3) as executor:
            list_future = executor.submit(
                client.get,
                "/api/v1/libraries/browser/memories",
                headers=headers,
            )
            detail_future = executor.submit(
                client.get,
                f"/api/v1/libraries/browser/memories/{created['id']}",
                headers=headers,
            )
            assert entered.wait(timeout=5.0)
            rebuild_future = executor.submit(kernel._rebuild_graph_serialized, "browser")
            time.sleep(0.2)
            release.set()
            listed = list_future.result(timeout=5.0)
            read_detail = detail_future.result(timeout=5.0)
            rebuilt = rebuild_future.result(timeout=5.0)

        assert listed.status_code == 200, listed.text
        assert read_detail.status_code == 200, read_detail.text
        assert rebuilt["library_id"] == "browser"


def test_forgotten_target_audit_redacts_all_content_but_keeps_metadata(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    secret = "FORGOTTEN-AUDIT-SECRET-BODY"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        created = _write(client, headers, "browser", secret, key="forgotten-audit")
        _forget(client, headers, "browser", str(created["id"]))

        targeted = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"target_id": created["id"], "limit": 100},
        )
        regional = client.get(
            "/api/v1/libraries/browser/history",
            headers=headers,
            params={"limit": 100},
        )
        assert targeted.status_code == 200, targeted.text
        assert regional.status_code == 200, regional.text
        assert secret not in targeted.text
        assert secret not in regional.text
        actions = {(item["action"], item["status"]) for item in targeted.json()}
        assert ("add", "committed") in actions
        assert ("delete:forget", "committed") in actions
        assert all(
            item.get("content_redacted") == "true"
            for item in targeted.json()
            if item.get("before_content") or item.get("after_content")
        ) or all(
            not item.get("before_content") and not item.get("after_content")
            for item in targeted.json()
        )


def test_pending_recovery_drains_more_than_one_batch_before_list_read(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "browser")
        kernel = client.app.state.memory_kernel
        scope = _scope_for_region(kernel._regions.require_region("browser"))
        maintenance = state_dir / "mem2" / "maintenance.sqlite3"
        with sqlite3.connect(maintenance) as connection:
            for index in range(101):
                unit_id = f"forgotten-pending-{index:03d}"
                connection.execute(
                    """INSERT INTO memory_browser_index
                       (region_id, unit_id, region_name, region_kind,
                        scope_org, scope_space, scope_user, scope_agent, scope_session,
                        content, summary, memory_type, memory_type_resolution, lifecycle,
                        updated_at, version, supersedes, source_count, source_references,
                        provenance, user_metadata)
                       VALUES (?, ?, 'browser', 'topic', ?, ?, ?, ?, ?, ?, ?, 'domain_fact',
                               'native', 'active', ?, ?, '', 1, '[]', '[]', '{}')""",
                    (
                        "browser",
                        unit_id,
                        scope.org,
                        scope.space,
                        scope.user,
                        scope.agent,
                        scope.session,
                        f"FORGOTTEN-PENDING-SECRET-{index:03d}",
                        f"FORGOTTEN-PENDING-SECRET-{index:03d}",
                        f"2026-10-27T00:00:{index % 60:02d}+00:00",
                        f"version-{index:03d}",
                    ),
                )
                connection.execute(
                    """INSERT INTO memory_browser_index_pending
                       (token, region_id, scope_org, scope_space, scope_user,
                        scope_agent, scope_session, unit_ids, created_at)
                       VALUES (?, 'browser', ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"pending-{index:03d}",
                        scope.org,
                        scope.space,
                        scope.user,
                        scope.agent,
                        scope.session,
                        json.dumps([unit_id]),
                        f"2026-10-27T00:01:{index % 60:02d}+00:00",
                    ),
                )
            connection.commit()

        page = _list(client, headers, region_id="browser")
        assert page["items"] == []
        assert "FORGOTTEN-PENDING-SECRET" not in json.dumps(page, ensure_ascii=False)
        with sqlite3.connect(maintenance) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_browser_index_pending"
            ).fetchone()[0] == 0


def test_browser_index_migration_is_single_pass_and_not_rebuilt_every_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "alpha")
        _region(client, headers, "beta")
        _write(client, headers, "alpha", "alpha durable memory", key="alpha-bulk")
        _write(client, headers, "beta", "beta durable memory", key="beta-bulk")
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        calls: list[str] = []
        original_units_in_scope = kernel._units_in_scope

        def count_scope(bundle_arg, scope):
            calls.append(scope.space)
            return original_units_in_scope(bundle_arg, scope)

        monkeypatch.setattr(kernel, "_units_in_scope", count_scope)
        monkeypatch.setattr(
            kernel,
            "_region_unit_pairs",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("index migration must not traverse by region")
            ),
        )
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            connection.execute(
                "DELETE FROM memory_browser_index_meta WHERE key = 'schema_version'"
            )
            connection.commit()
        kernel._ensure_browser_index(bundle)
        assert calls
        assert len(calls) == len(list(bundle.kv_store.scopes()))
        assert len(calls) == len(set(calls))

        kernel_type = type(kernel)
        monkeypatch.setattr(
            kernel_type,
            "_rebuild_browser_index",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("index must not rebuild when schema version is current")
            ),
        )

    with TestClient(_app(state_dir)) as restarted:
        headers = _headers(state_dir)
        assert _list(restarted, headers, region_id="alpha")["items"]
