from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.memory_kernel import (
    DeleteMemoryRequest,
    DocumentUpdateRequest,
    MemoryLifecycleConflictError,
    MemoryRevisionConflictError,
    RestoreForgottenRequest,
)


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir))


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _region(client: TestClient, headers: dict[str, str], region_id: str = "governance") -> None:
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json={"kind": "topic", "name": "治理测试", "region_id": region_id},
    )
    assert response.status_code == 201, response.text


def _write(
    client: TestClient,
    headers: dict[str, str],
    content: str,
    *,
    region_id: str = "governance",
    key: str = "write-1",
) -> dict[str, object]:
    response = client.post(
        "/api/v1/memories",
        headers=headers,
        json={
            "region_id": region_id,
            "content": content,
            "memory_type": "domain_fact",
            "source_references": ["source:governance-test"],
            "actor": "user",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _inspect(client: TestClient, headers: dict[str, str], unit_id: str) -> dict[str, object]:
    response = client.get(
        "/api/v1/libraries/governance/document",
        headers=headers,
        params={"path": unit_id},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _update(
    client: TestClient,
    headers: dict[str, str],
    unit_id: str,
    content: str,
    revision: str,
    *,
    mode: str,
    operation_id: str = "update-1",
) -> dict[str, object]:
    response = client.put(
        "/api/v1/libraries/governance/document",
        headers=headers,
        json={
            "path": unit_id,
            "content": content,
            "expected_source_version": revision,
            "operation_id": operation_id,
            "actor_type": "user",
            "source": "test-governance",
            "mode": mode,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _preview_delete(
    client: TestClient, headers: dict[str, str], unit_id: str, revision: str
) -> dict[str, object]:
    response = client.post(
        "/api/v1/libraries/governance/document/delete-preview",
        headers=headers,
        json={
            "path": unit_id,
            "expected_source_version": revision,
            "operation_id": "delete-preview-1",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _delete(
    client: TestClient,
    headers: dict[str, str],
    unit_id: str,
    revision: str,
    token: str,
    *,
    mode: str,
    operation_id: str = "delete-1",
) -> dict[str, str]:
    response = client.request(
        "DELETE",
        "/api/v1/libraries/governance/document",
        headers=headers,
        json={
            "path": unit_id,
            "expected_source_version": revision,
            "preview_token": token,
            "operation_id": operation_id,
            "actor_type": "user",
            "source": "test-governance",
            "mode": mode,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _documents(client: TestClient, headers: dict[str, str]) -> list[dict[str, object]]:
    response = client.get("/api/v1/libraries/governance/documents", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _history(client: TestClient, headers: dict[str, str]) -> list[dict[str, str]]:
    response = client.get("/api/v1/libraries/governance/history", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_list_inspect_and_markdown_export_show_region_memory(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "治理可见内容")
        unit_id = str(created["id"])

        page = client.get("/")
        assert page.status_code == 200
        for element in ("region-list", "document-list", "document-content", "history-list"):
            assert f'id="{element}"' in page.text

        regions = client.get("/api/v1/regions", headers=headers)
        assert regions.status_code == 200
        assert any(item["region_id"] == "governance" for item in regions.json())
        assert client.get("/api/v1/libraries", headers=headers).status_code == 404

        documents = _documents(client, headers)
        assert len(documents) == 1
        assert documents[0]["id"] == unit_id
        assert documents[0]["content"] == "治理可见内容"
        assert documents[0]["lifecycle"] == "active"
        assert documents[0]["source_references"] == ["source:governance-test"]
        assert documents[0]["space"] == "topic:governance"
        assert documents[0]["version"]

        inspected = _inspect(client, headers, unit_id)
        assert inspected["id"] == unit_id
        assert inspected["content"] == "治理可见内容"
        assert inspected["metadata"]["user"]["source_references"] == [
            "source:governance-test"
        ]
        assert inspected["history"][0]["id"] == unit_id

        exported = client.get(
            "/api/v1/libraries/governance/export.md", headers=headers
        )
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith("text/markdown")
        assert "治理可见内容" in exported.text
        assert unit_id in exported.text
        assert client.post(
            "/api/v1/libraries/governance/export.md", headers=headers
        ).status_code == 405


def test_supersede_creates_version_chain_and_persists_audit(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "第一版内容")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])

        updated = _update(
            client,
            headers,
            first_id,
            "第二版内容",
            first_revision,
            mode="supersede",
        )
        second_id = str(updated["id"])
        assert second_id != first_id

        second = _inspect(client, headers, second_id)
        assert second["content"] == "第二版内容"
        assert second["supersedes"] == first_id
        assert {item["id"] for item in second["history"]} == {first_id, second_id}
        assert _inspect(client, headers, first_id)["lifecycle"] == "superseded"

        history = _history(client, headers)
        event = next(item for item in history if item["action"] == "supersede")
        assert event["before_content"] == "第一版内容"
        assert event["after_content"] == "第二版内容"
        assert event["target_id"] == second_id

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        assert _inspect(client, headers, second_id)["content"] == "第二版内容"
        assert any(
            item["action"] == "supersede" for item in _history(client, headers)
        )


def test_overwrite_reuses_identity_and_audit_keeps_old_content(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "覆盖前内容")
        unit_id = str(first["id"])
        revision = str(_inspect(client, headers, unit_id)["version"])

        result = _update(
            client,
            headers,
            unit_id,
            "覆盖后内容",
            revision,
            mode="overwrite",
        )
        assert result["id"] == unit_id
        inspected = _inspect(client, headers, unit_id)
        assert inspected["content"] == "覆盖后内容"
        assert inspected["lifecycle"] == "active"

        event = next(item for item in _history(client, headers) if item["action"] == "overwrite")
        assert event["before_content"] == "覆盖前内容"
        assert event["after_content"] == "覆盖后内容"


@pytest.mark.parametrize(
    ("mode", "expected_lifecycle", "present_after"),
    [
        ("forget", "forgotten", True),
        ("archive", "archived", True),
        ("downweight", "active", True),
        ("purge", None, False),
    ],
)
def test_delete_semantics(
    tmp_path: Path, mode: str, expected_lifecycle: str | None, present_after: bool
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, f"删除前内容-{mode}")
        unit_id = str(created["id"])
        revision = str(_inspect(client, headers, unit_id)["version"])
        preview = _preview_delete(client, headers, unit_id, revision)
        assert preview["preview_token"] == revision

        deleted = _delete(
            client,
            headers,
            unit_id,
            revision,
            str(preview["preview_token"]),
            mode=mode,
        )
        assert deleted["status"] == mode

        documents = _documents(client, headers)
        ids = {str(item["id"]) for item in documents}
        if present_after:
            assert unit_id in ids
            assert _inspect(client, headers, unit_id)["lifecycle"] == expected_lifecycle
        else:
            assert unit_id not in ids

        history = _history(client, headers)
        event = next(item for item in history if item["action"] == f"delete:{mode}")
        if mode == "forget":
            assert event["before_content"] == ""
            assert event["content_redacted"] == "true"
            forgotten = client.get(
                "/api/v1/libraries/governance/forgotten", headers=headers
            )
            assert forgotten.status_code == 200
            assert {item["id"] for item in forgotten.json()} == {unit_id}
        else:
            assert event["before_content"] == f"删除前内容-{mode}"


def test_restore_old_content_creates_new_version(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "可恢复旧内容")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "当前内容",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        second_revision = str(_inspect(client, headers, second_id)["version"])

        restored = client.post(
            "/api/v1/libraries/governance/history/restore",
            headers=headers,
            json={
                "path": second_id,
                "commit": first_id,
                "expected_source_version": second_revision,
                "operation_id": "restore-1",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert restored.status_code == 200, restored.text
        third_id = restored.json()["id"]
        assert third_id not in {first_id, second_id}

        third = _inspect(client, headers, third_id)
        assert third["content"] == "可恢复旧内容"
        assert third["supersedes"] == second_id
        assert {item["id"] for item in third["history"]} == {
            first_id,
            second_id,
            third_id,
        }
        event = next(item for item in _history(client, headers) if item["action"] == "restore")
        assert event["before_content"] == "当前内容"
        assert event["after_content"] == "可恢复旧内容"


def test_governance_errors_are_mapped_and_stale_revision_is_rejected(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "版本冲突内容")
        unit_id = str(created["id"])
        revision = str(_inspect(client, headers, unit_id)["version"])
        _update(
            client,
            headers,
            unit_id,
            "新内容",
            revision,
            mode="supersede",
        )

        stale = client.put(
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": unit_id,
                "content": "过期修改",
                "expected_source_version": revision,
                "operation_id": "stale-update",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "supersede",
            },
        )
        assert stale.status_code == 409
        missing = client.get(
            "/api/v1/libraries/governance/document",
            headers=headers,
            params={"path": "missing-unit"},
        )
        assert missing.status_code == 404
        unknown_region = client.get(
            "/api/v1/libraries/missing-region/documents", headers=headers
        )
        assert unknown_region.status_code == 404


def test_restore_forgotten_creates_active_copy(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "可恢复的遗忘内容")
        unit_id = str(created["id"])
        revision = str(_inspect(client, headers, unit_id)["version"])
        preview = _preview_delete(client, headers, unit_id, revision)
        _delete(
            client,
            headers,
            unit_id,
            revision,
            str(preview["preview_token"]),
            mode="forget",
        )

        restored = client.post(
            "/api/v1/libraries/governance/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": unit_id,
                "commit": unit_id,
                "operation_id": "restore-forgotten-1",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert restored.status_code == 200, restored.text
        restored_id = restored.json()["id"]
        assert restored_id != unit_id
        inspected = _inspect(client, headers, restored_id)
        assert inspected["content"] == "可恢复的遗忘内容"
        assert inspected["lifecycle"] == "active"
        assert inspected["user_metadata"]["restored_from"] == unit_id
        event = next(
            item
            for item in _history(client, headers)
            if item["action"] == "restore:forgotten"
        )
        assert event["target_id"] == restored_id


def test_inspect_returns_provenance_chain_for_evolved_memory(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "unbound"
    root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        for event_id, kind, content in (
            ("trace-user", "user", "TRACE-FACT"),
            ("trace-assistant", "assistant", "TRACE-ANSWER"),
        ):
            response = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json={
                    "event_id": event_id,
                    "session_id": "trace-session",
                    "turn_id": "trace-turn",
                    "event_kind": kind,
                    "content": content,
                    "occurred_at": "2026-10-24T00:00:00Z",
                    "cwd": str(root),
                    "topic_id": "governance",
                },
            )
            assert response.status_code == 202, response.text
        evolved = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "trace-session"},
        )
        assert evolved.status_code == 202, evolved.text
        assert evolved.json() == {"queued": 1}

        documents = _documents(client, headers)
        evolved_document = next(
            item
            for item in documents
            if item["user_metadata"].get("memory_type") == "evolved"
        )
        inspected = _inspect(client, headers, str(evolved_document["id"]))
        assert inspected["provenance"]
        assert {item["content"] for item in inspected["provenance_chain"]} == {
            "TRACE-FACT",
            "TRACE-ANSWER",
        }


def test_management_contract_exposes_source_version_preview_diff_and_controls(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "管理台旧内容")
        unit_id = str(created["id"])
        inspected = _inspect(client, headers, unit_id)
        assert inspected["source_version"] == inspected["version"]

        preview = client.post(
            "/api/v1/libraries/governance/document/preview",
            headers=headers,
            json={
                "path": unit_id,
                "content": "管理台新内容",
                "expected_source_version": inspected["source_version"],
            },
        )
        assert preview.status_code == 200, preview.text
        assert "-管理台旧内容" in preview.json()["diff"]
        assert "+管理台新内容" in preview.json()["diff"]

        page = client.get("/")
        assert page.status_code == 200
        for control in ("save-document", "delete-document", "load-history", "promotion-button"):
            assert f'id="{control}"' in page.text
        assert "/api/v1/candidates" not in page.text

        history = _history(client, headers)
        assert history
        for entry in history:
            assert entry["commit"]
            assert entry["subject"]
            assert "authored_at" in entry
            assert "actor_type" in entry
            assert "source" in entry


def test_forgotten_history_exposes_deleted_at_and_version_chain(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "遗忘版本一")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "遗忘版本二",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        second_revision = str(_inspect(client, headers, second_id)["version"])
        preview = _preview_delete(client, headers, second_id, second_revision)
        _delete(
            client,
            headers,
            second_id,
            second_revision,
            str(preview["preview_token"]),
            mode="forget",
        )

        forgotten = client.get(
            "/api/v1/libraries/governance/forgotten", headers=headers
        )
        assert forgotten.status_code == 200, forgotten.text
        item = next(entry for entry in forgotten.json() if entry["id"] == second_id)
        assert item["deleted_at"]
        assert {version["id"] for version in item["history"]} == {first_id, second_id}


def test_restore_forgotten_selected_version_creates_linked_new_version(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "恢复目标旧内容")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "遗忘前当前内容",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        second_revision = str(_inspect(client, headers, second_id)["version"])
        preview = _preview_delete(client, headers, second_id, second_revision)
        _delete(
            client,
            headers,
            second_id,
            second_revision,
            str(preview["preview_token"]),
            mode="forget",
        )

        restored = client.post(
            "/api/v1/libraries/governance/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": second_id,
                "commit": first_id,
                "operation_id": "restore-selected-version",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert restored.status_code == 200, restored.text
        restored_id = restored.json()["id"]
        assert restored_id not in {first_id, second_id}
        restored_document = _inspect(client, headers, restored_id)
        assert restored_document["content"] == "恢复目标旧内容"
        assert restored_document["supersedes"] == second_id
        assert _inspect(client, headers, second_id)["lifecycle"] == "forgotten"
        diff = client.get(
            f"/api/v1/libraries/governance/history/{first_id}/diff",
            headers=headers,
        )
        assert diff.status_code == 200, diff.text
        assert diff.json()["current"] == restored_id


def test_restore_document_rejects_cross_document_history(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "文档甲旧内容", key="write-a")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        _update(
            client,
            headers,
            first_id,
            "文档甲新内容",
            first_revision,
            mode="supersede",
        )
        other = _write(client, headers, "文档乙内容", key="write-b")
        other_id = str(other["id"])
        other_revision = str(_inspect(client, headers, other_id)["version"])

        rejected = client.post(
            "/api/v1/libraries/governance/history/restore",
            headers=headers,
            json={
                "path": other_id,
                "commit": first_id,
                "expected_source_version": other_revision,
                "operation_id": "cross-document-restore",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert rejected.status_code == 404, rejected.text
        assert _inspect(client, headers, other_id)["content"] == "文档乙内容"


def test_governance_rejects_illegal_lifecycle_without_partial_write(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "生命周期一", key="lifecycle-a")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "生命周期二",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        assert _inspect(client, headers, first_id)["lifecycle"] == "superseded"
        superseded_revision = str(_inspect(client, headers, first_id)["version"])

        stale_update = client.put(
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": first_id,
                "content": "不应写入",
                "expected_source_version": superseded_revision,
                "operation_id": "illegal-update",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "supersede",
            },
        )
        assert stale_update.status_code == 409, stale_update.text
        superseded_preview = _preview_delete(
            client, headers, first_id, superseded_revision
        )
        stale_archive = client.request(
            "DELETE",
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": first_id,
                "expected_source_version": superseded_revision,
                "preview_token": superseded_preview["preview_token"],
                "operation_id": "illegal-archive",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "archive",
            },
        )
        assert stale_archive.status_code == 409, stale_archive.text
        assert {item["id"] for item in _documents(client, headers)} == {
            first_id,
            second_id,
        }
        assert _inspect(client, headers, first_id)["lifecycle"] == "superseded"

        active = _write(client, headers, "待归档内容", key="lifecycle-b")
        active_id = str(active["id"])
        active_revision = str(_inspect(client, headers, active_id)["version"])
        preview = _preview_delete(client, headers, active_id, active_revision)
        archived = _delete(
            client,
            headers,
            active_id,
            active_revision,
            str(preview["preview_token"]),
            mode="archive",
        )
        assert archived["status"] == "archive"
        archived_document = _inspect(client, headers, active_id)
        assert archived_document["lifecycle"] == "archived"
        illegal_update = client.put(
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": active_id,
                "content": "归档后不应更新",
                "expected_source_version": archived_document["version"],
                "operation_id": "illegal-archived-update",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "supersede",
            },
        )
        assert illegal_update.status_code == 409, illegal_update.text
        assert _inspect(client, headers, active_id)["content"] == "待归档内容"


def test_old_capture_schema_migration_creates_governance_audit(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    mem2_dir = state_dir / "mem2"
    mem2_dir.mkdir(parents=True)
    (mem2_dir / "VERSION").write_text("jiuwen-memory-0.2.0\n", encoding="utf-8")
    with sqlite3.connect(mem2_dir / "maintenance.sqlite3") as connection:
        connection.executescript(
            """CREATE TABLE capture_evolutions (
                evolve_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                region_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "旧表升级内容")
        unit_id = str(created["id"])
        revision = str(_inspect(client, headers, unit_id)["version"])
        updated = _update(
            client,
            headers,
            unit_id,
            "旧表升级后内容",
            revision,
            mode="supersede",
        )
        assert updated["id"] != unit_id
        assert any(
            event["action"] == "supersede" for event in _history(client, headers)
        )


def test_forgotten_deleted_at_tracks_deletion_time(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "删除时间内容")
        unit_id = str(created["id"])
        before = _inspect(client, headers, unit_id)
        time.sleep(0.05)
        revision = str(before["version"])
        preview = _preview_delete(client, headers, unit_id, revision)
        _delete(
            client,
            headers,
            unit_id,
            revision,
            str(preview["preview_token"]),
            mode="forget",
        )
        forgotten = next(
            item
            for item in client.get(
                "/api/v1/libraries/governance/forgotten", headers=headers
            ).json()
            if item["id"] == unit_id
        )
        assert forgotten["deleted_at"] != forgotten["created_at"]
        assert forgotten["deleted_at"] == forgotten["updated_at"]

        forgotten_revision = str(_inspect(client, headers, unit_id)["version"])
        second_preview = _preview_delete(
            client, headers, unit_id, forgotten_revision
        )
        repeated = client.request(
            "DELETE",
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": unit_id,
                "expected_source_version": forgotten_revision,
                "preview_token": second_preview["preview_token"],
                "operation_id": "repeat-forget",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "forget",
            },
        )
        assert repeated.status_code == 409, repeated.text
        repeated_item = next(
            item
            for item in client.get(
                "/api/v1/libraries/governance/forgotten", headers=headers
            ).json()
            if item["id"] == unit_id
        )
        assert repeated_item["deleted_at"] == forgotten["deleted_at"]


def test_history_and_export_reject_missing_region(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        for path in (
            "/api/v1/libraries/missing-region/history",
            "/api/v1/libraries/missing-region/export.md",
        ):
            response = client.get(path, headers=headers)
            assert response.status_code == 404, (path, response.text)
            assert "memory region not found" in response.json()["detail"]


def test_audit_mixed_timestamp_formats_include_latest_governance_event(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "审计排序内容")
        unit_id = str(created["id"])
        kernel = app.state.memory_kernel
        with closing(kernel._maintenance_db()) as connection, connection:
            connection.execute(
                """INSERT INTO governance_events
                   (region_id, unit_id, action, actor, source, operation_id,
                    before_content, after_content, detail, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "governance",
                    unit_id,
                    "future-event",
                    "user",
                    "test",
                    "future-operation",
                    "before",
                    "after",
                    "{}",
                    "committed",
                    "2099-01-01 00:00:00",
                ),
            )

        history = client.get(
            "/api/v1/libraries/governance/history",
            headers=headers,
            params={"limit": 1},
        )
        assert history.status_code == 200, history.text
        assert history.json()[0]["action"] == "future-event"


def test_governance_audit_is_persisted_before_destructive_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "ORIGINAL-SECRET", key="audit-before-write")
        unit_id = str(created["id"])
        revision = str(_inspect(client, headers, unit_id)["version"])
        kernel = app.state.memory_kernel

        def fail_audit(*_args: object, **_kwargs: object) -> int:
            raise OSError("audit storage unavailable")

        monkeypatch.setattr(kernel, "_begin_governance_event", fail_audit)
        failed_update = client.put(
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": unit_id,
                "content": "REPLACED-CONTENT",
                "expected_source_version": revision,
                "operation_id": "overwrite-with-failed-audit",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "overwrite",
            },
        )
        assert failed_update.status_code == 500, failed_update.text
        unchanged = _inspect(client, headers, unit_id)
        assert unchanged["content"] == "ORIGINAL-SECRET"
        assert unchanged["version"] == revision

        preview = _preview_delete(client, headers, unit_id, revision)
        failed_purge = client.request(
            "DELETE",
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": unit_id,
                "expected_source_version": revision,
                "preview_token": preview["preview_token"],
                "operation_id": "purge-with-failed-audit",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "purge",
            },
        )
        assert failed_purge.status_code == 500, failed_purge.text
        assert _inspect(client, headers, unit_id)["content"] == "ORIGINAL-SECRET"


def test_governance_audit_completion_failure_keeps_pending_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        created = _write(client, headers, "PENDING-BEFORE", key="pending-audit")
        unit_id = str(created["id"])
        revision = str(_inspect(client, headers, unit_id)["version"])
        kernel = app.state.memory_kernel
        monkeypatch.setattr(kernel, "_finish_governance_event", lambda *args, **kwargs: None)

        updated = client.put(
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": unit_id,
                "content": "PENDING-AFTER",
                "expected_source_version": revision,
                "operation_id": "pending-overwrite",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "overwrite",
            },
        )
        assert updated.status_code == 200, updated.text
        assert _inspect(client, headers, unit_id)["content"] == "PENDING-AFTER"
        event = next(
            item for item in _history(client, headers) if item["action"] == "overwrite"
        )
        assert event["status"] == "pending"
        assert event["before_content"] == "PENDING-BEFORE"
        assert event["after_content"] == "PENDING-AFTER"


def test_concurrent_governance_operations_are_serialized(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        kernel = app.state.memory_kernel

        def attempt(operation):
            try:
                return "ok", operation()
            except MemoryRevisionConflictError:
                return "conflict", None

        supersede_base = _write(client, headers, "并发基线", key="concurrent-supersede")
        supersede_id = str(supersede_base["id"])
        supersede_revision = str(_inspect(client, headers, supersede_id)["version"])
        supersede_requests = [
            DocumentUpdateRequest(
                library_id="governance",
                path=supersede_id,
                content=f"并发版本-{index}",
                expected_revision=supersede_revision,
                operation_id=f"concurrent-supersede-{index}",
                actor="user",
                source="test-governance",
                mode="supersede",
            )
            for index in range(2)
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            supersede_results = list(
                executor.map(
                    lambda request: attempt(
                        lambda: kernel._update_document_serialized(request)
                    ),
                    supersede_requests,
                )
            )
        assert sorted(result[0] for result in supersede_results) == ["conflict", "ok"]
        supersede_documents = _documents(client, headers)
        successors = [
            item for item in supersede_documents if item["supersedes"] == supersede_id
        ]
        assert len(successors) == 1

        overwrite_base = _write(client, headers, "覆盖基线", key="concurrent-overwrite")
        overwrite_id = str(overwrite_base["id"])
        overwrite_revision = str(_inspect(client, headers, overwrite_id)["version"])
        overwrite_requests = [
            DocumentUpdateRequest(
                library_id="governance",
                path=overwrite_id,
                content=f"覆盖结果-{index}",
                expected_revision=overwrite_revision,
                operation_id=f"concurrent-overwrite-{index}",
                actor="user",
                source="test-governance",
                mode="overwrite",
            )
            for index in range(2)
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            overwrite_results = list(
                executor.map(
                    lambda request: attempt(
                        lambda: kernel._update_document_serialized(request)
                    ),
                    overwrite_requests,
                )
            )
        assert sorted(result[0] for result in overwrite_results) == ["conflict", "ok"]
        overwrite_events = [
            item
            for item in _history(client, headers)
            if item["action"] == "overwrite" and item["target_id"] == overwrite_id
        ]
        assert len(overwrite_events) == 1

        forget_base = _write(client, headers, "并发遗忘", key="concurrent-forget")
        forget_id = str(forget_base["id"])
        forget_revision = str(_inspect(client, headers, forget_id)["version"])
        preview = _preview_delete(client, headers, forget_id, forget_revision)
        forget_requests = [
            DeleteMemoryRequest(
                library_id="governance",
                path=forget_id,
                expected_revision=forget_revision,
                operation_id=f"concurrent-forget-{index}",
                actor="user",
                source="test-governance",
                preview_token=str(preview["preview_token"]),
                mode="forget",
            )
            for index in range(2)
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            forget_results = list(
                executor.map(
                    lambda request: attempt(
                        lambda: kernel._delete_document_serialized(request)
                    ),
                    forget_requests,
                )
            )
        assert sorted(result[0] for result in forget_results) == ["conflict", "ok"]
        forget_events = [
            item
            for item in _history(client, headers)
            if item["action"] == "delete:forget" and item["target_id"] == forget_id
        ]
        assert len(forget_events) == 1


def test_post_mutation_metadata_failures_do_not_return_false_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        forget_base = _write(client, headers, "遗忘元数据失败", key="forget-meta-failure")
        forget_id = str(forget_base["id"])
        forget_revision = str(_inspect(client, headers, forget_id)["version"])
        preview = _preview_delete(client, headers, forget_id, forget_revision)
        monkeypatch.setattr(
            "personal_agent_memory.mem2_kernel.dumps",
            lambda _unit: (_ for _ in ()).throw(OSError("kv write failed")),
        )
        forgotten = client.request(
            "DELETE",
            "/api/v1/libraries/governance/document",
            headers=headers,
            json={
                "path": forget_id,
                "expected_source_version": forget_revision,
                "preview_token": preview["preview_token"],
                "operation_id": "forget-meta-failure",
                "actor_type": "user",
                "source": "test-governance",
                "mode": "forget",
            },
        )
        assert forgotten.status_code == 200, forgotten.text
        assert _inspect(client, headers, forget_id)["lifecycle"] == "forgotten"
        forget_event = next(
            item
            for item in _history(client, headers)
            if item["action"] == "delete:forget" and item["target_id"] == forget_id
        )
        assert '"deleted_at_persist_failed": true' in forget_event["detail"]


def test_restore_forgotten_rolls_back_when_version_link_cannot_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "恢复旧内容", key="restore-link-base")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "当前内容",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        second_revision = str(_inspect(client, headers, second_id)["version"])
        preview = _preview_delete(client, headers, second_id, second_revision)
        _delete(
            client,
            headers,
            second_id,
            second_revision,
            str(preview["preview_token"]),
            mode="forget",
        )
        monkeypatch.setattr(
            "personal_agent_memory.mem2_kernel.dumps",
            lambda _unit: (_ for _ in ()).throw(OSError("kv write failed")),
        )
        restored = client.post(
            "/api/v1/libraries/governance/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": second_id,
                "commit": first_id,
                "operation_id": "restore-link-failure",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert restored.status_code == 503, restored.text
        documents = _documents(client, headers)
        assert all(item["lifecycle"] != "active" for item in documents)
        assert {item["id"] for item in documents} == {first_id, second_id}


def test_restore_forgotten_rejects_duplicate_active_copy(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "重复恢复旧内容", key="duplicate-restore-base")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "重复恢复当前内容",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        second_revision = str(_inspect(client, headers, second_id)["version"])
        preview = _preview_delete(client, headers, second_id, second_revision)
        _delete(
            client,
            headers,
            second_id,
            second_revision,
            str(preview["preview_token"]),
            mode="forget",
        )
        kernel = app.state.memory_kernel
        request = RestoreForgottenRequest(
            library_id="governance",
            tombstone_id=second_id,
            commit=first_id,
            operation_id="duplicate-restore",
            actor="user",
            source="test-governance",
        )
        first_result = kernel._restore_forgotten_serialized(request)
        assert first_result["supersedes"] == second_id
        try:
            kernel._restore_forgotten_serialized(request)
        except MemoryLifecycleConflictError:
            pass
        else:
            raise AssertionError("duplicate forgotten restore was accepted")
        active = [
            item for item in _documents(client, headers) if item["lifecycle"] == "active"
        ]
        assert len(active) == 1
        restore_events = [
            item
            for item in _history(client, headers)
            if item["action"] == "restore:forgotten"
        ]
        assert len(restore_events) == 1


def test_concurrent_restore_forgotten_creates_one_active_copy(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "并发恢复旧内容", key="concurrent-restore-base")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "并发恢复当前内容",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        second_revision = str(_inspect(client, headers, second_id)["version"])
        preview = _preview_delete(client, headers, second_id, second_revision)
        _delete(
            client,
            headers,
            second_id,
            second_revision,
            str(preview["preview_token"]),
            mode="forget",
        )
        kernel = app.state.memory_kernel
        requests = [
            RestoreForgottenRequest(
                library_id="governance",
                tombstone_id=second_id,
                commit=first_id,
                operation_id=f"concurrent-restore-{index}",
                actor="user",
                source="test-governance",
            )
            for index in range(2)
        ]

        def attempt(request: RestoreForgottenRequest) -> str:
            try:
                kernel._restore_forgotten_serialized(request)
                return "ok"
            except (MemoryRevisionConflictError, MemoryLifecycleConflictError):
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(attempt, requests))
        assert sorted(results) == ["conflict", "ok"]
        active = [
            item for item in _documents(client, headers) if item["lifecycle"] == "active"
        ]
        assert len(active) == 1
        assert len(
            [
                item
                for item in _history(client, headers)
                if item["action"] == "restore:forgotten"
            ]
        ) == 1


def test_restore_forgotten_double_failure_returns_503_with_pending_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    app = _app(state_dir)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "双失败旧内容", key="double-failure-base")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "双失败当前内容",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        second_revision = str(_inspect(client, headers, second_id)["version"])
        preview = _preview_delete(client, headers, second_id, second_revision)
        _delete(
            client,
            headers,
            second_id,
            second_revision,
            str(preview["preview_token"]),
            mode="forget",
        )
        kernel = app.state.memory_kernel
        bundle = kernel._require_bundle()
        monkeypatch.setattr(
            "personal_agent_memory.mem2_kernel.dumps",
            lambda _unit: (_ for _ in ()).throw(OSError("kv write failed")),
        )

        def fail_delete(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("rollback unavailable")

        monkeypatch.setattr(bundle.runtime.api, "delete", fail_delete)
        restored = client.post(
            "/api/v1/libraries/governance/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": second_id,
                "commit": first_id,
                "operation_id": "double-failure-restore",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert restored.status_code == 503, restored.text
        event = next(
            item
            for item in _history(client, headers)
            if item["action"] == "restore:forgotten"
        )
        assert event["status"] == "pending"
        assert "version_link_pending" in event["detail"]
        active = [
            item for item in _documents(client, headers) if item["lifecycle"] == "active"
        ]
        assert len(active) == 1
        assert active[0]["user_metadata"]["restore_pending"] is True
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": "governance",
                "query": "双失败旧内容",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["results"] == []
        repeated = client.post(
            "/api/v1/libraries/governance/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": second_id,
                "commit": first_id,
                "operation_id": "double-failure-retry",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert repeated.status_code == 409, repeated.text


def test_restore_forgotten_uses_latest_tombstone_in_multi_forget_cycle(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(client, headers, "多周期版本一", key="multi-forget-base")
        first_id = str(first["id"])
        first_revision = str(_inspect(client, headers, first_id)["version"])
        second = _update(
            client,
            headers,
            first_id,
            "多周期版本二",
            first_revision,
            mode="supersede",
        )
        second_id = str(second["id"])
        second_revision = str(_inspect(client, headers, second_id)["version"])
        preview = _preview_delete(client, headers, second_id, second_revision)
        _delete(
            client,
            headers,
            second_id,
            second_revision,
            str(preview["preview_token"]),
            mode="forget",
        )
        restored = client.post(
            "/api/v1/libraries/governance/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": second_id,
                "commit": first_id,
                "operation_id": "multi-forget-restore-1",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert restored.status_code == 200, restored.text
        third_id = restored.json()["id"]
        third_revision = str(_inspect(client, headers, third_id)["version"])
        third_preview = _preview_delete(client, headers, third_id, third_revision)
        _delete(
            client,
            headers,
            third_id,
            third_revision,
            str(third_preview["preview_token"]),
            mode="forget",
        )

        forgotten = client.get(
            "/api/v1/libraries/governance/forgotten", headers=headers
        )
        assert forgotten.status_code == 200, forgotten.text
        assert {item["id"] for item in forgotten.json()} == {third_id}
        old_tombstone = client.post(
            "/api/v1/libraries/governance/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": second_id,
                "commit": first_id,
                "operation_id": "multi-forget-old-tombstone",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert old_tombstone.status_code == 409, old_tombstone.text
        current_tombstone = client.post(
            "/api/v1/libraries/governance/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": third_id,
                "commit": first_id,
                "operation_id": "multi-forget-current-tombstone",
                "actor_type": "user",
                "source": "test-governance",
            },
        )
        assert current_tombstone.status_code == 200, current_tombstone.text
        fourth_id = current_tombstone.json()["id"]
        assert _inspect(client, headers, fourth_id)["supersedes"] == third_id
        assert _inspect(client, headers, second_id)["supersedes"] == first_id
        assert _inspect(client, headers, third_id)["supersedes"] == second_id
        diff = client.get(
            f"/api/v1/libraries/governance/history/{first_id}/diff",
            headers=headers,
        )
        assert diff.status_code == 200, diff.text
        assert diff.json()["current"] == fourth_id
