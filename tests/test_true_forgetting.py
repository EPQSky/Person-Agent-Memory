from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NoReturn, cast

from fastapi.testclient import TestClient as FastAPITestClient
from pytest import MonkeyPatch, mark, raises, skip

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.git_history import GitRepository
from personal_agent_memory.graph_adapter import (
    GraphAdapterError,
    GraphExpansion,
    GraphPurge,
    GraphSourceDocument,
    JiuwenMilvusGraphAdapter,
)
from personal_agent_memory.model_client import ModelEndpoint, OpenAICompatibleClient
from personal_agent_memory.state import MemoryMutationError, PlatformState


class TestClient(FastAPITestClient):
    def request(self, method: str, url: str, **kwargs: object):
        payload = kwargs.get("json")
        if (
            method.upper() == "DELETE"
            and url.endswith("/document")
            and isinstance(payload, dict)
            and "preview_token" not in payload
        ):
            preview = super().request(
                "POST",
                f"{url}/delete-preview",
                headers=kwargs.get("headers"),
                json={
                    "path": payload["path"],
                    "expected_source_version": payload["expected_source_version"],
                    "operation_id": f"preview-{uuid.uuid4()}",
                },
            )
            if preview.is_success:
                kwargs["json"] = {**payload, "preview_token": preview.json()["preview_token"]}
        return super().request(method, url, **kwargs)


class _FailingPurgeGraphAdapter:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[GraphSourceDocument, ...]] = {}

    def stage_rebuild(
        self,
        library_id: str,
        documents: tuple[GraphSourceDocument, ...],
        *,
        cleanup_id: str | None = None,
    ) -> GraphPurge:
        del documents, cleanup_id
        raise GraphAdapterError(f"cannot purge {library_id}")

    def stage_purge(self, library_id: str, *, cleanup_id: str | None = None) -> GraphPurge:
        del cleanup_id
        raise GraphAdapterError(f"cannot purge {library_id}")

    def reconcile_staged_purge(
        self, library_id: str, cleanup_id: str, *, committed: bool
    ) -> None:
        del library_id, cleanup_id, committed

    async def rebuild(self, library_id: str, documents: tuple[GraphSourceDocument, ...]) -> None:
        self.documents[library_id] = documents

    async def expand(
        self,
        library_id: str,
        seed_document_ids: tuple[str, ...],
        *,
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        return []


class _RecordingGraphPurge:
    def __init__(
        self,
        adapter: _RecordingGraphAdapter,
        library_id: str,
        previous: tuple[GraphSourceDocument, ...],
    ) -> None:
        self.adapter = adapter
        self.library_id = library_id
        self.previous = previous

    def commit(self) -> None:
        return

    def rollback(self) -> None:
        self.adapter.documents[self.library_id] = self.previous


class _RecordingGraphAdapter:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[GraphSourceDocument, ...]] = {}

    def stage_rebuild(
        self,
        library_id: str,
        documents: tuple[GraphSourceDocument, ...],
        *,
        cleanup_id: str | None = None,
    ) -> GraphPurge:
        del cleanup_id
        previous = self.documents.get(library_id, ())
        self.documents[library_id] = documents
        return _RecordingGraphPurge(self, library_id, previous)

    def stage_purge(self, library_id: str, *, cleanup_id: str | None = None) -> GraphPurge:
        del cleanup_id
        previous = self.documents.pop(library_id, ())
        return _RecordingGraphPurge(self, library_id, previous)

    def reconcile_staged_purge(
        self, library_id: str, cleanup_id: str, *, committed: bool
    ) -> None:
        del library_id, cleanup_id, committed

    async def rebuild(self, library_id: str, documents: tuple[GraphSourceDocument, ...]) -> None:
        self.documents[library_id] = documents

    async def expand(
        self,
        library_id: str,
        seed_document_ids: tuple[str, ...],
        *,
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        return []


class _BlockingRebuildGraphAdapter(_RecordingGraphAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    async def rebuild(
        self, library_id: str, documents: tuple[GraphSourceDocument, ...]
    ) -> None:
        self.started.set()
        await asyncio.to_thread(self.release.wait)
        self.documents[library_id] = documents


class _BlockingEmbeddingClient(OpenAICompatibleClient):
    def __init__(self) -> None:
        super().__init__(
            ModelEndpoint(base_url="http://127.0.0.1:1", model="blocking-embedding"),
            None,
        )
        self.started = threading.Event()
        self.release = threading.Event()

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.started.set()
        self.release.wait()
        return [[float(index + 1), 0.5] for index, _ in enumerate(texts)]


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _register(
    client: TestClient, headers: dict[str, str], root: Path, kind: str = "project"
) -> str:
    response = client.post(
        "/api/v1/libraries", headers=headers, json={"path": str(root), "kind": kind}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _deletion_preview_token(
    client: TestClient,
    headers: dict[str, str],
    library_id: str,
    path: str,
    expected_source_version: str,
) -> str:
    response = client.post(
        f"/api/v1/libraries/{library_id}/document/delete-preview",
        headers=headers,
        json={
            "path": path,
            "expected_source_version": expected_source_version,
            "operation_id": f"preview-{uuid.uuid4()}",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["preview_token"])


def _delete_for_restore_test(
    client: TestClient,
    headers: dict[str, str],
    library_id: str,
    content: str,
    *,
    operation_id: str,
) -> tuple[str, dict[str, str]]:
    initial_commit = str(
        client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
            "commit"
        ]
    )
    deleted = client.request(
        "DELETE",
        f"/api/v1/libraries/{library_id}/document",
        headers=headers,
        json={
            "path": "forgotten.md",
            "expected_source_version": hashlib.sha256(content.encode()).hexdigest(),
            "operation_id": operation_id,
            "actor_type": "user",
            "source": "test",
        },
    )
    assert deleted.status_code == 200, deleted.text
    return initial_commit, cast(dict[str, str], deleted.json())


def test_delete_uses_preview_key_during_concurrent_tombstone_rotation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Stable key\n\nDelete and restore across key rotation.\n"
    path = root / "stable.md"
    path.write_text(content, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        source_version = hashlib.sha256(content.encode()).hexdigest()
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": content,
                "source_references": ["codex-session:rotation#assistant-final"],
                "creator": "codex",
                "idempotency_key": "rotation-candidate",
            },
        )
        assert candidate.status_code == 201, candidate.text
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute(
                "INSERT INTO capture_rounds "
                "(session_id, turn_id, library_id, status, candidate_id) "
                "VALUES ('rotation-session', 'turn-1', ?, 'done', ?)",
                (library_id, candidate.json()["id"]),
            )
            connection.execute(
                "INSERT INTO capture_inbox "
                "(event_id, request_hash, session_id, project_id, library_id, turn_id, "
                "event_kind, content, occurred_at) "
                "VALUES ('rotation-event', 'rotation-request', 'rotation-session', "
                "'project', ?, 'turn-1', 'assistant', ?, '2026-09-06T00:00:00Z')",
                (library_id, content),
            )
            connection.commit()
        original_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        preview_token = _deletion_preview_token(
            client, headers, library_id, "stable.md", source_version
        )
        preview_key_id = preview_token.split(":")[1]
        semantic_content = content

        fingerprint_keys: list[str] = []
        match_keys: list[str] = []
        source_scope_keys: list[str] = []
        original_fingerprint = state._tombstone_fingerprint
        original_match = state._tombstone_match_fingerprint
        original_source_scope = state._tombstone_source_scope

        def rotate_after_first_fingerprint(
            target_library_id: str, target_content: str, key_id: str
        ) -> str:
            fingerprint_keys.append(key_id)
            result = original_fingerprint(target_library_id, target_content, key_id)
            if len(fingerprint_keys) == 1:
                rotation = threading.Thread(target=state.rotate_tombstone_key)
                rotation.start()
                rotation.join(timeout=5)
                assert not rotation.is_alive()
            return result

        def record_match_key(
            target_library_id: str,
            target_content: str,
            key_id: str,
            *,
            reject_invalid_formal: bool = False,
        ) -> str:
            match_keys.append(key_id)
            return original_match(
                target_library_id,
                target_content,
                key_id,
                reject_invalid_formal=reject_invalid_formal,
            )

        def record_source_scope_key(
            target_library_id: str,
            target_path: str,
            authoritative_paths: tuple[str, ...],
            candidate_snapshot: object,
            key_id: str,
        ) -> dict[str, object]:
            source_scope_keys.append(key_id)
            return original_source_scope(
                target_library_id,
                target_path,
                authoritative_paths,
                cast(Any, candidate_snapshot),
                key_id,
            )

        monkeypatch.setattr(state, "_tombstone_fingerprint", rotate_after_first_fingerprint)
        monkeypatch.setattr(state, "_tombstone_match_fingerprint", record_match_key)
        monkeypatch.setattr(state, "_tombstone_source_scope", record_source_scope_key)

        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "stable.md",
                "expected_source_version": source_version,
                "operation_id": "delete-during-key-rotation",
                "actor_type": "user",
                "source": "test",
                "preview_token": preview_token,
            },
        )
        assert deleted.status_code == 200, deleted.text
        assert state.active_tombstone_key_id != preview_key_id
        assert set(fingerprint_keys + match_keys + source_scope_keys) == {preview_key_id}

        tombstone_id = deleted.json()["tombstone_id"]
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            fingerprint, match_fingerprint, key_id, source_scope_json, marker_path = (
                connection.execute(
                "SELECT fingerprint, match_fingerprint, key_id, source_scope_json, marker_path "
                "FROM memory_tombstones WHERE id = ?",
                (tombstone_id,),
                ).fetchone()
            )
        marker = json.loads((root / marker_path).read_text(encoding="utf-8"))
        assert key_id == marker["key_id"] == preview_key_id
        assert json.loads(source_scope_json) == marker["source_scope"]
        scope_material = b"personal-agent-memory:tombstone-source:v1\0" + json.dumps(
            {
                "library_id": library_id,
                "session_id": "rotation-session",
                "turn_id": "turn-1",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert marker["source_scope"]["capture_ranges"] == [
            {
                "scope_id": "hmac-sha256:"
                + hmac.new(
                    state.tombstone_keys[preview_key_id], scope_material, hashlib.sha256
                ).hexdigest(),
                "event_count": 1,
            }
        ]
        assert fingerprint == marker["fingerprint"] == original_fingerprint(
            library_id, content, preview_key_id
        )
        assert match_fingerprint == marker["match_fingerprint"] == original_match(
            library_id, semantic_content, preview_key_id
        )

        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": tombstone_id,
                "commit": original_commit,
                "operation_id": "restore-after-key-rotation",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert restored.status_code == 200, restored.text
        assert path.read_text(encoding="utf-8") == content


def test_forgetting_removes_every_projection_blocks_replay_and_restores_atomically(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    content = "# Runtime\n\nUse Firebird 5 for the audit database.\n"
    replacement = "# Runtime\n\nUse SQLite for the local scratch database.\n"
    path = first_root / "runtime.md"
    path.write_text(content, encoding="utf-8")
    (first_root / "replacement.md").write_text(content, encoding="utf-8")
    (second_root / "same.md").write_text(content, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, first_root)
        other_library_id = _register(client, headers, second_root, "user")
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "runtime.md"},
        ).json()
        initial_history = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        initial_commit = initial_history[0]["commit"]
        replacement_document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "replacement.md"},
        ).json()
        replacement_edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "replacement.md",
                "content": replacement,
                "expected_source_version": replacement_document["source_version"],
                "operation_id": "prepare-tombstone-bypass-regressions",
                "actor_type": "user",
                "source": "test-setup",
            },
        )
        assert replacement_edit.status_code == 200, replacement_edit.text

        database = state_dir / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            chunk_id = connection.execute(
                "SELECT id FROM memory_chunks WHERE library_id = ?", (library_id,)
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO memory_chunk_vectors
                   (chunk_id, library_id, source_version, model, vector_json)
                   VALUES (?, ?, ?, 'test', '[1.0]')""",
                (chunk_id, library_id, document["source_version"]),
            )
            connection.execute(
                """INSERT INTO memory_graph_documents
                   (library_id, document_id, path, source_version)
                   SELECT library_id, id, path, source_version FROM memory_documents
                   WHERE library_id = ?""",
                (library_id,),
            )
            connection.commit()

        preview_payload = {
            "path": "runtime.md",
            "expected_source_version": document["source_version"],
            "operation_id": "unused-preview-operation",
        }
        preview = client.post(
            f"/api/v1/libraries/{library_id}/document/delete-preview",
            headers=headers,
            json=preview_payload,
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["content"] == content
        assert preview.json()["derived"] == {
            "summary": False,
            "full_text_chunks": 1,
            "full_text_entries": 1,
            "embeddings": 1,
            "graph_documents": 1,
            "graph_relationships": "all relationships sourced by this document",
            "candidate_records": 0,
            "candidate_audit_records": 0,
            "capture_events": 0,
        }
        assert preview.json()["supersession_chain"][0]["content"] == content

        deletion_payload = {
            **preview_payload,
            "preview_token": preview.json()["preview_token"],
            "operation_id": "forget-runtime",
            "actor_type": "user",
            "source": "web-delete",
        }
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json=deletion_payload,
        )
        assert deleted.status_code == 200, deleted.text
        assert (
            client.request(
                "DELETE",
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json=deletion_payload,
            ).json()
            == deleted.json()
        )
        tombstone_id = deleted.json()["tombstone_id"]
        assert not path.exists()

        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_documents WHERE library_id = ? AND path = ?",
                (library_id, "runtime.md"),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE library_id = ? AND path = ?",
                (library_id, "runtime.md"),
            ).fetchone() == (0,)
            assert connection.execute(
                """SELECT COUNT(*) FROM memory_chunk_vectors AS vector
                   JOIN memory_chunks AS chunk ON chunk.id = vector.chunk_id
                   WHERE vector.library_id = ? AND chunk.path = ?""",
                (library_id, "runtime.md"),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_graph_documents WHERE library_id = ? AND path = ?",
                (library_id, "runtime.md"),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_versions WHERE library_id = ? AND path = ?",
                (library_id, "runtime.md"),
            ).fetchone() == (0,)
            tombstone = connection.execute(
                """SELECT fingerprint, key_id, source_scope_json, deleted_at, marker_path
                   FROM memory_tombstones WHERE id = ?""",
                (tombstone_id,),
            ).fetchone()
            git_dir = connection.execute(
                "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
                (library_id,),
            ).fetchone()[0]
        assert tombstone[0].startswith("hmac-sha256:")
        assert content not in json.dumps(tombstone)
        marker = first_root / tombstone[4]
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        assert set(marker_payload) == {
            "deleted_at",
            "fingerprint",
            "format",
            "key_id",
            "match_fingerprint",
            "source_scope",
        }
        assert content not in marker.read_text(encoding="utf-8")
        tracked = subprocess.run(
            ["git", f"--git-dir={git_dir}", "ls-tree", "-r", "--name-only", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        assert tracked == [tombstone[4], ".personal-agent-memory.json", "replacement.md"]

        for include_history in (False, True):
            search = client.post(
                "/api/v1/search",
                headers=headers,
                json={
                    "library_id": library_id,
                    "query": "Firebird",
                    "include_history": include_history,
                },
            )
            assert search.status_code == 200
            assert search.json()["results"] == []
        hidden_diff = client.get(
            f"/api/v1/libraries/{library_id}/history/{initial_commit}/diff",
            headers=headers,
        ).json()
        assert hidden_diff["diff"].startswith("[redacted:")
        assert content not in json.dumps(hidden_diff)

        blocked = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": content,
                "source_references": ["codex-session:old#assistant-final"],
                "creator": "codex",
                "idempotency_key": "old-replay",
            },
        )
        assert blocked.status_code == 422
        assert blocked.json()["detail"] == "candidate matches forgotten memory"
        isolated = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": other_library_id,
                "suggested_type": "decision",
                "body": content,
                "source_references": ["codex-session:new#assistant-final"],
                "creator": "codex",
                "idempotency_key": "other-library",
            },
        )
        assert isolated.status_code == 201

        replacement_document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "replacement.md"},
        ).json()
        blocked_edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "replacement.md",
                "content": content,
                "expected_source_version": replacement_document["source_version"],
                "operation_id": "blocked-forgotten-edit",
                "actor_type": "user",
                "source": "web",
            },
        )
        assert blocked_edit.status_code == 422
        assert blocked_edit.json()["detail"] == "content matches forgotten memory"
        blocked_restore = client.post(
            f"/api/v1/libraries/{library_id}/history/restore",
            headers=headers,
            json={
                "path": "replacement.md",
                "commit": initial_commit,
                "expected_source_version": replacement_document["source_version"],
                "operation_id": "blocked-forgotten-generic-restore",
                "actor_type": "user",
                "source": "web-history",
            },
        )
        assert blocked_restore.status_code == 422
        assert blocked_restore.json()["detail"] == "content matches forgotten memory"

        old_key_id = tombstone[1]
        client.app.state.platform_state.rotate_tombstone_key()
        assert client.app.state.platform_state.active_tombstone_key_id != old_key_id
        assert (
            client.post(
                "/mcp/candidates",
                headers=headers,
                json={
                    "library_id": library_id,
                    "suggested_type": "decision",
                    "body": content,
                    "source_references": ["codex-session:rotated#assistant-final"],
                    "creator": "codex",
                    "idempotency_key": "old-after-key-rotation",
                },
            ).status_code
            == 422
        )

        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": tombstone_id,
                "commit": initial_commit,
                "operation_id": "restore-runtime",
                "actor_type": "user",
                "source": "web-forgotten-history",
            },
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["commit"] not in {initial_commit, deleted.json()["commit"]}
        assert path.read_text(encoding="utf-8") == content
        assert client.get(f"/api/v1/libraries/{library_id}/forgotten", headers=headers).json() == []
        assert not marker.exists()
        history = client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()
        assert history[0]["kind"] == "restore"
        assert history[1]["kind"] == "delete"
        assert (
            client.post(
                "/mcp/candidates",
                headers=headers,
                json={
                    "library_id": library_id,
                    "suggested_type": "decision",
                    "body": content,
                    "source_references": ["codex-session:after-restore#assistant-final"],
                    "creator": "codex",
                    "idempotency_key": "after-restore",
                },
            ).status_code
            == 201
        )


def test_forgotten_restore_commit_then_raise_returns_durable_success(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    real_connect = sqlite3.connect

    class RaiseAfterDurableRestore(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if type(self).injected:
                return
            operation = self.execute(
                "SELECT 1 FROM memory_operations WHERE operation_id = 'durable-forgotten-restore'"
            ).fetchone()
            if operation is not None:
                type(self).injected = True
                raise sqlite3.OperationalError("injected durable restore commit ambiguity")

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = RaiseAfterDurableRestore
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Durable restore\n\nRestoreCommitAmbiguityNeedle\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id="delete-before-durable-restore",
        )
        restore_payload = {
            "tombstone_id": deleted["tombstone_id"],
            "commit": initial_commit,
            "operation_id": "durable-forgotten-restore",
            "actor_type": "user",
            "source": "test",
        }
        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json=restore_payload,
        )
        assert restored.status_code == 200, restored.text
        restored_payload = restored.json()
        assert (root / "forgotten.md").read_text(encoding="utf-8") == content
        assert client.get(
            f"/api/v1/libraries/{library_id}/forgotten", headers=headers
        ).json() == []

    with TestClient(create_app(settings)) as restarted:
        headers = _headers(state_dir)
        assert restarted.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json=restore_payload,
        ).json() == restored_payload
        assert restarted.get(
            f"/api/v1/libraries/{library_id}/forgotten", headers=headers
        ).json() == []
        replay = restarted.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": content,
                "source_references": ["session:after-durable-restore"],
                "creator": "codex",
                "idempotency_key": "after-durable-restore",
            },
        )
        assert replay.status_code == 201, replay.text


def test_inconsistent_commit_then_raise_restores_tombstone_and_replay_protection(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    real_connect = sqlite3.connect

    class CorruptAfterRestoreCommit(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if type(self).injected:
                return
            operation = self.execute(
                "SELECT 1 FROM memory_operations WHERE operation_id = 'inconsistent-restore'"
            ).fetchone()
            if operation is not None:
                type(self).injected = True
                self.execute(
                    "UPDATE memory_documents SET source_version = 'inconsistent' "
                    "WHERE path = 'forgotten.md'"
                )
                super().commit()
                raise MemoryMutationError("injected inconsistent durable restore")

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = CorruptAfterRestoreCommit
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    project = tmp_path / "project"
    root.mkdir()
    project.mkdir()
    content = "# Forgotten\n\nCompensatedRestoreReplayNeedle\n"
    retained = "# Retained\n\nKeep this editable.\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    (root / "retained.md").write_text(retained, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        assert client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library_id},
        ).status_code == 201
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id="delete-before-inconsistent-restore",
        )
        delete_commit = deleted["commit"]
        marker = root / ".personal-agent-memory-tombstones" / f"{deleted['tombstone_id']}.json"
        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted["tombstone_id"],
                "commit": initial_commit,
                "operation_id": "inconsistent-restore",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert restored.status_code == 422, restored.text
        assert restored.json()["detail"] == "injected inconsistent durable restore"
        assert not (root / "forgotten.md").exists()
        assert marker.is_file()
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"] == delete_commit

    with TestClient(create_app(settings)) as restarted:
        headers = _headers(state_dir)
        forgotten = restarted.get(
            f"/api/v1/libraries/{library_id}/forgotten", headers=headers
        ).json()
        assert [item["id"] for item in forgotten] == [deleted["tombstone_id"]]
        retained_document = restarted.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "retained.md"},
        ).json()
        ordinary = restarted.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "retained.md",
                "content": content,
                "expected_source_version": retained_document["source_version"],
                "operation_id": "blocked-ordinary-replay-after-compensation",
                "actor_type": "user",
                "source": "test",
            },
        )
        candidate = restarted.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": content,
                "source_references": ["session:compensated-replay"],
                "creator": "codex",
                "idempotency_key": "compensated-candidate-replay",
            },
        )
        capture = restarted.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "compensated-capture-replay",
                "session_id": "compensated-session",
                "turn_id": "turn-1",
                "event_kind": "assistant",
                "content": content,
                "occurred_at": "2026-09-06T12:00:00Z",
                "cwd": str(project),
            },
        )
        assert ordinary.status_code == candidate.status_code == capture.status_code == 422
        assert ordinary.json()["detail"] == "content matches forgotten memory"
        assert candidate.json()["detail"] == "candidate matches forgotten memory"
        assert capture.json()["detail"] == "capture event matches forgotten memory"


@mark.parametrize("corruption", ("fts", "version", "job"))
def test_restore_commit_ambiguity_rejects_incomplete_index_projection(
    tmp_path: Path, monkeypatch: MonkeyPatch, corruption: str
) -> None:
    real_connect = sqlite3.connect
    operation_id = f"corrupt-{corruption}-restore"

    class CorruptProjectionAfterRestoreCommit(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if type(self).injected or self.execute(
                "SELECT 1 FROM memory_operations WHERE operation_id = ?", (operation_id,)
            ).fetchone() is None:
                return
            type(self).injected = True
            if corruption == "fts":
                self.execute(
                    "UPDATE memory_chunk_search SET content = 'tampered FTS projection' "
                    "WHERE path = 'forgotten.md'"
                )
            elif corruption == "version":
                self.execute(
                    "UPDATE memory_versions SET state = 'conditional', "
                    "condition_text = 'tampered version projection' "
                    "WHERE path = 'forgotten.md' AND state = 'current'"
                )
            else:
                self.execute(
                    "UPDATE background_jobs SET attempts = 999 WHERE id = ("
                    "SELECT MAX(id) FROM background_jobs WHERE kind = 'vector_rebuild')"
                )
            super().commit()
            raise MemoryMutationError(f"injected {corruption} projection corruption")

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = CorruptProjectionAfterRestoreCommit
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = f"# Forgotten\n\nCorrupt{corruption.title()}ProjectionNeedle\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id=f"delete-before-{corruption}-corruption",
        )
        if corruption == "job":
            with real_connect(state_dir / "platform.sqlite3") as connection:
                connection.execute(
                    "INSERT INTO background_jobs (kind, payload) "
                    "VALUES ('vector_rebuild', ?)",
                    (json.dumps({"library_id": library_id}),),
                )
        delete_commit = deleted["commit"]
        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted["tombstone_id"],
                "commit": initial_commit,
                "operation_id": operation_id,
                "actor_type": "user",
                "source": "test",
            },
        )
        assert restored.status_code == 422, restored.text
        assert restored.json()["detail"] == f"injected {corruption} projection corruption"
        assert not (root / "forgotten.md").exists()
        assert client.get(
            f"/api/v1/libraries/{library_id}/forgotten", headers=headers
        ).json()[0]["id"] == deleted["tombstone_id"]
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"] == delete_commit
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_documents WHERE library_id = ? "
                "AND path = 'forgotten.md'",
                (library_id,),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_versions WHERE library_id = ? "
                "AND path = 'forgotten.md'",
                (library_id,),
            ).fetchone() == (0,)
            if corruption == "job":
                assert connection.execute(
                    "SELECT attempts FROM background_jobs WHERE kind = 'vector_rebuild' "
                    "AND json_extract(payload, '$.library_id') = ?",
                    (library_id,),
                ).fetchone() == (0,)


@mark.parametrize("git_corruption", ("content", "marker", "parent"))
def test_restore_commit_ambiguity_rejects_counterfeit_git_transition(
    tmp_path: Path, monkeypatch: MonkeyPatch, git_corruption: str
) -> None:
    real_connect = sqlite3.connect
    real_commit = GitRepository.commit
    operation_id = "counterfeit-git-restore"

    class RaiseAfterCounterfeitRestore(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if type(self).injected or self.execute(
                "SELECT 1 FROM memory_operations WHERE operation_id = ?", (operation_id,)
            ).fetchone() is None:
                return
            type(self).injected = True
            raise MemoryMutationError("injected counterfeit Git restore")

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = RaiseAfterCounterfeitRestore
        return real_connect(*args, **kwargs)

    def counterfeit_commit(
        repository: GitRepository, contents: dict[str, bytes | None], message: str
    ) -> str:
        if not message.startswith("Restore forgotten memory"):
            return real_commit(repository, contents, message)
        counterfeit = dict(contents)
        document_path = next(path for path, body in contents.items() if body is not None)
        if git_corruption == "content":
            counterfeit[document_path] = b"# Counterfeit\n\nWrong restored Git blob.\n"
        elif git_corruption == "marker":
            marker_path = next(path for path, body in contents.items() if body is None)
            del counterfeit[marker_path]
        else:
            real_commit(
                repository,
                {"intervening.md": b"# Intervening\n\nUnexpected parent.\n"},
                "Unexpected commit before restore",
            )
        return real_commit(repository, counterfeit, message)

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(GitRepository, "commit", counterfeit_commit)
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = f"# Forgotten\n\nCounterfeit{git_corruption.title()}GitRestoreNeedle\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id="delete-before-counterfeit-git",
        )
        delete_commit = deleted["commit"]
        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted["tombstone_id"],
                "commit": initial_commit,
                "operation_id": operation_id,
                "actor_type": "user",
                "source": "test",
            },
        )
        assert restored.status_code == 422, restored.text
        assert restored.json()["detail"] in {
            "memory history commit content does not match",
            "memory history commit changed unexpected paths",
            "memory history commit has an unexpected parent",
        }
        assert (root / "forgotten.md").exists()
        history_head = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        assert history_head != delete_commit
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_restore_intents WHERE operation_id = ?",
                (operation_id,),
            ).fetchone() == (1,)


def test_restore_commit_then_raise_is_verified_and_compensated(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    real_commit = GitRepository.commit

    def commit_then_raise(
        repository: GitRepository, contents: dict[str, bytes | None], message: str
    ) -> str:
        committed = real_commit(repository, contents, message)
        if message.startswith("Restore forgotten memory"):
            raise MemoryMutationError("injected failure after durable Git restore")
        return committed

    monkeypatch.setattr(GitRepository, "commit", commit_then_raise)
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Forgotten\n\nCommitThenRaiseRestoreNeedle\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id="delete-before-commit-then-raise",
        )
        delete_commit = deleted["commit"]
        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted["tombstone_id"],
                "commit": initial_commit,
                "operation_id": "commit-then-raise-restore",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert restored.status_code == 422, restored.text
        assert restored.json()["detail"] == "injected failure after durable Git restore"
        assert not (root / "forgotten.md").exists()
        assert client.get(
            f"/api/v1/libraries/{library_id}/forgotten", headers=headers
        ).json()[0]["id"] == deleted["tombstone_id"]
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"] == delete_commit
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_restore_intents"
            ).fetchone() == (0,)

    monkeypatch.setattr(GitRepository, "commit", real_commit)
    for _ in range(2):
        with TestClient(create_app(settings)) as restarted:
            headers = _headers(state_dir)
            assert restarted.get(
                f"/api/v1/libraries/{library_id}/history", headers=headers
            ).json()[0]["commit"] == delete_commit


def test_restore_git_rollback_failure_stops_compensation_until_restart(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    real_commit = GitRepository.commit
    real_rollback = GitRepository.rollback_commit
    rollback_attempted = False

    def commit_then_raise(
        repository: GitRepository, contents: dict[str, bytes | None], message: str
    ) -> str:
        committed = real_commit(repository, contents, message)
        if message.startswith("Restore forgotten memory"):
            raise MemoryMutationError("injected failure after durable Git restore")
        return committed

    def fail_first_rollback(
        repository: GitRepository, commit: str, previous_head: str | None
    ) -> None:
        nonlocal rollback_attempted
        if not rollback_attempted:
            rollback_attempted = True
            raise MemoryMutationError("injected transient Git rollback failure")
        real_rollback(repository, commit, previous_head)

    monkeypatch.setattr(GitRepository, "commit", commit_then_raise)
    monkeypatch.setattr(GitRepository, "rollback_commit", fail_first_rollback)
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Forgotten\n\nTransientGitRollbackNeedle\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id="delete-before-transient-git-rollback",
        )
        previous_head = deleted["commit"]
        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted["tombstone_id"],
                "commit": initial_commit,
                "operation_id": "transient-git-rollback-restore",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert restored.status_code == 422, restored.text
        assert "injected transient Git rollback failure" in restored.json()["detail"]
        assert (root / "forgotten.md").read_text(encoding="utf-8") == content
        assert not next(
            (root / ".personal-agent-memory-tombstones").glob("*.json"), None
        )
        current_head = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        assert current_head != previous_head
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT recovery_phase FROM memory_restore_intents"
            ).fetchone() == ("rollback",)

    monkeypatch.setattr(GitRepository, "commit", real_commit)
    monkeypatch.setattr(GitRepository, "rollback_commit", real_rollback)
    for _ in range(2):
        with TestClient(create_app(settings)) as restarted:
            headers = _headers(state_dir)
            assert restarted.get(
                f"/api/v1/libraries/{library_id}/history", headers=headers
            ).json()[0]["commit"] == previous_head
            assert not (root / "forgotten.md").exists()
            forgotten = restarted.get(
                f"/api/v1/libraries/{library_id}/forgotten", headers=headers
            ).json()
            assert forgotten[0]["id"] == deleted["tombstone_id"]
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_restore_intents"
            ).fetchone() == (0,)


def test_restore_commit_failure_before_persistence_restores_original_forgotten_state(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    real_connect = sqlite3.connect

    class FailBeforeRestoreCommit(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            if not type(self).injected and self.execute(
                "SELECT 1 FROM memory_operations WHERE operation_id = 'uncommitted-restore'"
            ).fetchone() is not None:
                type(self).injected = True
                raise MemoryMutationError("injected failure before restore persistence")
            super().commit()

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = FailBeforeRestoreCommit
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Forgotten\n\nUncommittedRestoreNeedle\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id="delete-before-uncommitted-restore",
        )
        delete_commit = deleted["commit"]
        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted["tombstone_id"],
                "commit": initial_commit,
                "operation_id": "uncommitted-restore",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert restored.status_code == 422, restored.text
        assert restored.json()["detail"] == "injected failure before restore persistence"
        assert not (root / "forgotten.md").exists()
        assert client.get(
            f"/api/v1/libraries/{library_id}/forgotten", headers=headers
        ).json()[0]["id"] == deleted["tombstone_id"]
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"] == delete_commit

    with TestClient(create_app(settings)) as restarted:
        headers = _headers(state_dir)
        assert restarted.get(
            f"/api/v1/libraries/{library_id}/forgotten", headers=headers
        ).json()[0]["id"] == deleted["tombstone_id"]


@mark.parametrize(
    ("crash_stage", "completed"),
    (
        ("intent", False),
        ("document", False),
        ("marker", False),
        ("git", False),
        ("sqlite", True),
    ),
)
def test_restart_reconciles_interrupted_forgotten_restore(
    tmp_path: Path, crash_stage: str, completed: bool
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    sentinel = f"RestoreCrash{crash_stage.title()}Needle"
    content = f"# Interrupted restore\n\n{sentinel}\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))
    operation_id = f"crash-{crash_stage}-restore"

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id=f"delete-before-{crash_stage}-restore",
        )
        delete_commit = deleted["commit"]

    payload = {
        "tombstone_id": deleted["tombstone_id"],
        "commit": initial_commit,
        "operation_id": operation_id,
        "actor_type": "user",
        "source": "test",
    }
    crash_script = f"""
import os
import sqlite3
from fastapi.testclient import TestClient
from pathlib import Path
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.git_history import GitRepository
from personal_agent_memory.state import PlatformState

stage = {crash_stage!r}
if stage == "intent":
    original = PlatformState._begin_memory_restore_intent
    def crash(self, **kwargs):
        original(self, **kwargs)
        os._exit(91)
    PlatformState._begin_memory_restore_intent = crash
elif stage == "document":
    def crash(self, operation_id, identity):
        os._exit(91)
    PlatformState._record_memory_restore_document_identity = crash
elif stage == "marker":
    original = GitRepository.commit
    def crash(self, contents, message):
        if message.startswith("Restore forgotten memory"):
            os._exit(91)
        return original(self, contents, message)
    GitRepository.commit = crash
elif stage == "git":
    original = GitRepository.commit
    def crash(self, contents, message):
        commit = original(self, contents, message)
        if message.startswith("Restore forgotten memory"):
            os._exit(91)
        return commit
    GitRepository.commit = crash
else:
    real_connect = sqlite3.connect
    class CrashAfterDurableRestore(sqlite3.Connection):
        def commit(self):
            super().commit()
            if self.execute(
                "SELECT 1 FROM memory_operations WHERE operation_id = ?",
                ({operation_id!r},),
            ).fetchone() is not None:
                os._exit(91)
    def connect(*args, **kwargs):
        kwargs["factory"] = CrashAfterDurableRestore
        return real_connect(*args, **kwargs)
    sqlite3.connect = connect

settings = Settings(
    state_dir=Path({str(state_dir)!r}),
    library_roots=(Path({str(tmp_path)!r}),),
)
with TestClient(create_app(settings)) as client:
    response = client.post(
        "/api/v1/libraries/{library_id}/forgotten/restore",
        headers={headers!r},
        json={payload!r},
    )
    raise SystemExit(response.status_code)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert crashed.returncode == 91, crashed.stderr
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        intent = connection.execute(
            "SELECT * FROM memory_restore_intents WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        assert intent is not None
        assert sentinel not in json.dumps(intent)

    for _ in range(2):
        with TestClient(create_app(settings)) as restarted:
            headers = _headers(state_dir)
            forgotten = restarted.get(
                f"/api/v1/libraries/{library_id}/forgotten", headers=headers
            ).json()
            history_head = restarted.get(
                f"/api/v1/libraries/{library_id}/history", headers=headers
            ).json()[0]["commit"]
            if completed:
                assert forgotten == []
                assert (root / "forgotten.md").read_text(encoding="utf-8") == content
                assert history_head not in {initial_commit, delete_commit}
                assert restarted.post(
                    f"/api/v1/libraries/{library_id}/forgotten/restore",
                    headers=headers,
                    json=payload,
                ).status_code == 200
            else:
                assert forgotten[0]["id"] == deleted["tombstone_id"]
                assert not (root / "forgotten.md").exists()
                assert history_head == delete_commit
            with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
                assert connection.execute(
                    "SELECT COUNT(*) FROM memory_restore_intents"
                ).fetchone() == (0,)


def _crash_forgotten_restore_before_git_commit(tmp_path: Path) -> dict[str, object]:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Interrupted restore\n\nRestoreRecoveryAttackNeedle\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))
    operation_id = "crash-before-restore-git"
    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id="delete-before-restore-recovery-attack",
        )
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            git_dir = Path(
                str(
                    connection.execute(
                        "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
                        (library_id,),
                    ).fetchone()[0]
                )
            )
    payload = {
        "tombstone_id": deleted["tombstone_id"],
        "commit": initial_commit,
        "operation_id": operation_id,
        "actor_type": "user",
        "source": "test",
    }
    crash_script = f"""
import os
from fastapi.testclient import TestClient
from pathlib import Path
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.git_history import GitRepository

original = GitRepository.commit
def crash(self, contents, message):
    if message.startswith("Restore forgotten memory"):
        os._exit(92)
    return original(self, contents, message)
GitRepository.commit = crash
settings = Settings(
    state_dir=Path({str(state_dir)!r}),
    library_roots=(Path({str(tmp_path)!r}),),
)
with TestClient(create_app(settings)) as client:
    client.post(
        "/api/v1/libraries/{library_id}/forgotten/restore",
        headers={headers!r},
        json={payload!r},
    )
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert crashed.returncode == 92, crashed.stderr
    return {
        "content": content,
        "git_dir": git_dir,
        "library_id": library_id,
        "operation_id": operation_id,
        "root": root,
        "settings": settings,
        "state_dir": state_dir,
    }


@mark.parametrize("replacement", ("fifo", "inode", "mode"))
def test_restart_rejects_changed_restore_target_without_clearing_intent(
    tmp_path: Path, replacement: str
) -> None:
    crashed = _crash_forgotten_restore_before_git_commit(tmp_path)
    root = cast(Path, crashed["root"])
    target = root / "forgotten.md"
    content = cast(str, crashed["content"])
    if replacement == "fifo":
        target.unlink()
        os.mkfifo(target, 0o600)
    elif replacement == "inode":
        target.rename(root / "pinned.md")
        target.write_text(content, encoding="utf-8")
        target.chmod(0o600)
    else:
        target.chmod(0o644)
    started = time.monotonic()
    startup = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fastapi.testclient import TestClient; "
                "from pathlib import Path; "
                "from personal_agent_memory.app import create_app; "
                "from personal_agent_memory.config import Settings; "
                f"s=Settings(state_dir=Path({str(crashed['state_dir'])!r}), "
                f"library_roots=(Path({str(tmp_path)!r}),)); "
                "TestClient(create_app(s)).__enter__()"
            ),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert startup.returncode != 0
    assert time.monotonic() - started < 5
    with sqlite3.connect(cast(Path, crashed["state_dir"]) / "platform.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_restore_intents WHERE operation_id = ?",
            (crashed["operation_id"],),
        ).fetchone() == (1,)


def test_restart_clears_restore_intent_after_completed_rollback_crashes(
    tmp_path: Path,
) -> None:
    crashed = _crash_forgotten_restore_before_git_commit(tmp_path)
    state_dir = cast(Path, crashed["state_dir"])
    root = cast(Path, crashed["root"])
    crash_script = f"""
import os
from fastapi.testclient import TestClient
from pathlib import Path
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.state import PlatformState

original = PlatformState._complete_memory_restore_intent
def crash(self, operation_id):
    phase = self.connection_or_raise.execute(
        "SELECT recovery_phase FROM memory_restore_intents WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()
    if phase == ("rollback",):
        os._exit(93)
    original(self, operation_id)
PlatformState._complete_memory_restore_intent = crash
settings = Settings(
    state_dir=Path({str(state_dir)!r}),
    library_roots=(Path({str(tmp_path)!r}),),
)
TestClient(create_app(settings)).__enter__()
"""
    first_restart = subprocess.run(
        [sys.executable, "-c", crash_script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert first_restart.returncode == 93, first_restart.stderr
    assert not (root / "forgotten.md").exists()
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        row = connection.execute(
            "SELECT recovery_phase FROM memory_restore_intents WHERE operation_id = ?",
            (crashed["operation_id"],),
        ).fetchone()
        assert row == ("rollback",)
        marker_path = str(
            connection.execute(
                "SELECT marker_path FROM memory_restore_intents WHERE operation_id = ?",
                (crashed["operation_id"],),
            ).fetchone()[0]
        )
    restored_marker = root / marker_path
    assert restored_marker.is_file()

    race_script = f"""
import os
from fastapi.testclient import TestClient
from pathlib import Path
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings

marker = Path({str(restored_marker)!r})
real_read = os.read
replaced = False
def racing_read(descriptor, size):
    global replaced
    data = real_read(descriptor, size)
    if data and not replaced:
        try:
            opened = Path(os.readlink(f"/proc/self/fd/{{descriptor}}"))
        except OSError:
            opened = Path()
        if opened == marker:
            replacement = marker.with_suffix(".race")
            replacement.write_bytes(data)
            replacement.chmod(0o600)
            os.replace(replacement, marker)
            replaced = True
    return data
os.read = racing_read
settings = Settings(
    state_dir=Path({str(state_dir)!r}),
    library_roots=(Path({str(tmp_path)!r}),),
)
TestClient(create_app(settings)).__enter__()
"""
    raced_restart = subprocess.run(
        [sys.executable, "-c", race_script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert raced_restart.returncode != 0
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        assert connection.execute(
            "SELECT recovery_phase FROM memory_restore_intents WHERE operation_id = ?",
            (crashed["operation_id"],),
        ).fetchone() == ("rollback",)
    restored_identity = (restored_marker.stat().st_dev, restored_marker.stat().st_ino)

    settings = cast(Settings, crashed["settings"])
    for _ in range(2):
        with TestClient(create_app(settings)):
            pass
        assert not (root / "forgotten.md").exists()
        assert restored_marker.is_file()
        assert (restored_marker.stat().st_dev, restored_marker.stat().st_ino) == restored_identity
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_restore_intents"
            ).fetchone() == (0,)


def test_request_compensation_advances_restore_journal_before_side_effects(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Request compensation\n\nCompensationJournalCrashNeedle\n"
    (root / "forgotten.md").write_text(content, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))
    operation_id = "request-compensation-crash"
    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit, deleted = _delete_for_restore_test(
            client,
            headers,
            library_id,
            content,
            operation_id="delete-before-request-compensation",
        )
    payload = {
        "tombstone_id": deleted["tombstone_id"],
        "commit": initial_commit,
        "operation_id": operation_id,
        "actor_type": "user",
        "source": "test",
    }
    crash_script = f"""
import os
from fastapi.testclient import TestClient
from pathlib import Path
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.git_history import GitRepository
from personal_agent_memory.state import MemoryMutationError, PlatformState

real_commit = GitRepository.commit
def fail_restore_commit(self, contents, message):
    if message.startswith("Restore forgotten memory"):
        raise MemoryMutationError("injected restore failure")
    return real_commit(self, contents, message)
GitRepository.commit = fail_restore_commit
real_complete = PlatformState._complete_memory_restore_intent
def crash_before_clear(self, operation_id):
    phase = self.connection_or_raise.execute(
        "SELECT recovery_phase FROM memory_restore_intents WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()
    if phase == ("rollback",):
        os._exit(94)
    real_complete(self, operation_id)
PlatformState._complete_memory_restore_intent = crash_before_clear
settings = Settings(
    state_dir=Path({str(state_dir)!r}),
    library_roots=(Path({str(tmp_path)!r}),),
)
with TestClient(create_app(settings)) as client:
    client.post(
        "/api/v1/libraries/{library_id}/forgotten/restore",
        headers={headers!r},
        json={payload!r},
    )
"""
    compensated = subprocess.run(
        [sys.executable, "-c", crash_script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert compensated.returncode == 94, compensated.stderr
    assert not (root / "forgotten.md").exists()
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        assert connection.execute(
            "SELECT recovery_phase FROM memory_restore_intents WHERE operation_id = ?",
            (operation_id,),
        ).fetchone() == ("rollback",)

    for _ in range(2):
        with TestClient(create_app(settings)) as restarted:
            headers = _headers(state_dir)
            forgotten = restarted.get(
                f"/api/v1/libraries/{library_id}/forgotten", headers=headers
            ).json()
            assert forgotten[0]["id"] == deleted["tombstone_id"]
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_restore_intents"
            ).fetchone() == (0,)


def test_restart_migrates_restore_journal_without_recovery_phase(tmp_path: Path) -> None:
    crashed = _crash_forgotten_restore_before_git_commit(tmp_path)
    state_dir = cast(Path, crashed["state_dir"])
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        connection.execute("ALTER TABLE memory_restore_intents DROP COLUMN recovery_phase")
        connection.commit()

    with TestClient(create_app(cast(Settings, crashed["settings"]))):
        pass
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        columns = {
            str(row[1]): row for row in connection.execute(
                "PRAGMA table_info(memory_restore_intents)"
            )
        }
        assert columns["recovery_phase"][4] == "'started'"
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_restore_intents"
        ).fetchone() == (0,)


@mark.parametrize("replacement", ("same-digest", "wrong-directory", "wrong-mode"))
def test_restart_rejects_replaced_restore_marker_before_rollback_phase(
    tmp_path: Path, replacement: str
) -> None:
    crashed = _crash_forgotten_restore_before_git_commit(tmp_path)
    state_dir = cast(Path, crashed["state_dir"])
    root = cast(Path, crashed["root"])
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        marker_path = str(
            connection.execute(
                "SELECT marker_path FROM memory_restore_intents WHERE operation_id = ?",
                (crashed["operation_id"],),
            ).fetchone()[0]
        )
    marker = subprocess.run(
        ["git", f"--git-dir={crashed['git_dir']}", "show", f"HEAD:{marker_path}"],
        capture_output=True,
        check=True,
    ).stdout
    target = root / marker_path
    if replacement == "wrong-directory":
        moved = root / f"{target.parent.name}.moved"
        target.parent.rename(moved)
        target.parent.mkdir(mode=0o700)
    target.write_bytes(marker)
    target.chmod(0o644 if replacement == "wrong-mode" else 0o600)

    startup = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fastapi.testclient import TestClient; "
                "from pathlib import Path; "
                "from personal_agent_memory.app import create_app; "
                "from personal_agent_memory.config import Settings; "
                f"s=Settings(state_dir=Path({str(state_dir)!r}), "
                f"library_roots=(Path({str(tmp_path)!r}),)); "
                "TestClient(create_app(s)).__enter__()"
            ),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert startup.returncode != 0
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        assert connection.execute(
            "SELECT recovery_phase FROM memory_restore_intents WHERE operation_id = ?",
            (crashed["operation_id"],),
        ).fetchone() == ("started",)


def test_restart_rejects_counterfeit_restore_child_without_changing_history(
    tmp_path: Path,
) -> None:
    crashed = _crash_forgotten_restore_before_git_commit(tmp_path)
    git_dir = cast(Path, crashed["git_dir"])
    previous_head = subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    with sqlite3.connect(cast(Path, crashed["state_dir"]) / "platform.sqlite3") as connection:
        marker_path = str(
            connection.execute(
                "SELECT marker_path FROM memory_restore_intents WHERE operation_id = ?",
                (crashed["operation_id"],),
            ).fetchone()[0]
        )
    root = cast(Path, crashed["root"])
    index = tmp_path / "counterfeit-restore.index"
    environment = {
        **os.environ,
        "GIT_INDEX_FILE": str(index),
        "GIT_WORK_TREE": str(root),
    }
    subprocess.run(
        ["git", f"--git-dir={git_dir}", "read-tree", previous_head],
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            "update-index",
            "--remove",
            "--",
            marker_path,
        ],
        env=environment,
        check=True,
    )
    blob = subprocess.run(
        ["git", f"--git-dir={git_dir}", "hash-object", "-w", "--stdin"],
        input=b"# Counterfeit\n\nWrong restore body.\n",
        capture_output=True,
        check=True,
    ).stdout.decode().strip()
    subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            "update-index",
            "--add",
            "--cacheinfo",
            "100644",
            blob,
            "forgotten.md",
        ],
        env=environment,
        check=True,
    )
    tree = subprocess.run(
        ["git", f"--git-dir={git_dir}", "write-tree"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    counterfeit = subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            "commit-tree",
            tree,
            "-p",
            previous_head,
            "-m",
            "Counterfeit restore commit",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    reference = subprocess.run(
        ["git", f"--git-dir={git_dir}", "symbolic-ref", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", f"--git-dir={git_dir}", "update-ref", reference, counterfeit, previous_head],
        check=True,
    )
    startup = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fastapi.testclient import TestClient; "
                "from pathlib import Path; "
                "from personal_agent_memory.app import create_app; "
                "from personal_agent_memory.config import Settings; "
                f"s=Settings(state_dir=Path({str(crashed['state_dir'])!r}), "
                f"library_roots=(Path({str(tmp_path)!r}),)); "
                "TestClient(create_app(s)).__enter__()"
            ),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert startup.returncode != 0
    assert subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip() == counterfeit
    with sqlite3.connect(cast(Path, crashed["state_dir"]) / "platform.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_restore_intents WHERE operation_id = ?",
            (crashed["operation_id"],),
        ).fetchone() == (1,)


def test_tombstone_fingerprint_is_library_scoped(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    roots = (tmp_path / "one", tmp_path / "two")
    content = "# Preference\n\nKeep the terminal quiet.\n"
    for root in roots:
        root.mkdir()
        (root / "preference.md").write_text(content, encoding="utf-8")
    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        fingerprints = []
        for index, root in enumerate(roots):
            library_id = _register(client, headers, root)
            source_version = hashlib.sha256(content.encode()).hexdigest()
            response = client.request(
                "DELETE",
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": "preference.md",
                    "expected_source_version": source_version,
                    "operation_id": f"forget-{index}",
                },
            )
            assert response.status_code == 200, response.text
            marker = next((root / ".personal-agent-memory-tombstones").glob("*.json"))
            fingerprints.append(json.loads(marker.read_text())["fingerprint"])
        assert fingerprints[0] != fingerprints[1]


def test_graph_purge_failure_rolls_back_forgetting(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    path = root / "retained.md"
    content = "# Retained\n\nForget only after the graph is clean.\n"
    path.write_text(content, encoding="utf-8")
    adapter = _FailingPurgeGraphAdapter()

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        state.graph_adapter = adapter
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        source_version = hashlib.sha256(content.encode()).hexdigest()
        before = client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
            "commit"
        ]

        response = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "retained.md",
                "expected_source_version": source_version,
                "operation_id": "purge-must-succeed",
            },
        )

        assert response.status_code == 422
        assert "cannot purge" in response.json()["detail"]
        assert path.read_text(encoding="utf-8") == content
        assert (
            client.get(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                params={"path": "retained.md"},
            ).status_code
            == 200
        )
        assert client.get(f"/api/v1/libraries/{library_id}/forgotten", headers=headers).json() == []
        assert (
            client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
                "commit"
            ]
            == before
        )


def _require_retained_mode(path: Path, mode: int) -> None:
    path.chmod(mode)
    if path.stat().st_mode & 0o7777 != mode:
        skip(f"filesystem does not retain mode {mode:04o}")


def test_forgetting_accepts_regular_documents_with_special_modes(tmp_path: Path) -> None:
    for mode in (0o755, 0o4755, 0o2755, 0o1755):
        case = tmp_path / f"mode-{mode:04o}"
        root = case / "memory"
        root.mkdir(parents=True)
        path = root / "special.md"
        content = f"# Special mode {mode:04o}\n\nDelete this regular Markdown.\n"
        path.write_text(content, encoding="utf-8")
        _require_retained_mode(path, mode)
        state_dir = case / "state"

        with TestClient(
            create_app(Settings(state_dir=state_dir, library_roots=(case,)))
        ) as client:
            headers = _headers(state_dir)
            library_id = _register(client, headers, root)
            document = client.get(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                params={"path": "special.md"},
            ).json()
            response = client.request(
                "DELETE",
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": "special.md",
                    "expected_source_version": document["source_version"],
                    "operation_id": f"forget-mode-{mode:04o}",
                },
            )

        assert response.status_code == 200, response.text
        assert not path.exists()


def test_forgetting_compensation_restores_special_modes(tmp_path: Path) -> None:
    for mode in (0o755, 0o4755, 0o2755, 0o1755):
        case = tmp_path / f"mode-{mode:04o}"
        root = case / "memory"
        root.mkdir(parents=True)
        path = root / "retained.md"
        content = f"# Retained mode {mode:04o}\n\nRestore this regular Markdown.\n"
        path.write_text(content, encoding="utf-8")
        _require_retained_mode(path, mode)
        state_dir = case / "state"

        with TestClient(
            create_app(Settings(state_dir=state_dir, library_roots=(case,)))
        ) as client:
            state = cast(PlatformState, client.app.state.platform_state)
            state.graph_adapter = _FailingPurgeGraphAdapter()
            headers = _headers(state_dir)
            library_id = _register(client, headers, root)
            document = client.get(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                params={"path": "retained.md"},
            ).json()
            response = client.request(
                "DELETE",
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": "retained.md",
                    "expected_source_version": document["source_version"],
                    "operation_id": f"rollback-mode-{mode:04o}",
                },
            )

        assert response.status_code == 422
        assert path.read_text(encoding="utf-8") == content
        assert path.stat().st_mode & 0o7777 == mode


def test_forgetting_compensation_retains_journal_when_special_mode_is_masked(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    path = root / "retained.md"
    content = "# Retained mode\n\nDo not accept a masked recovery mode.\n"
    path.write_text(content, encoding="utf-8")
    _require_retained_mode(path, 0o4755)

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        state.graph_adapter = _FailingPurgeGraphAdapter()
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "retained.md"},
        ).json()
        real_fchmod = os.fchmod

        def mask_special_mode(descriptor: int, mode: int) -> None:
            real_fchmod(descriptor, mode & 0o777)

        with monkeypatch.context() as patch:
            patch.setattr(os, "fchmod", mask_special_mode)
            response = client.request(
                "DELETE",
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": "retained.md",
                    "expected_source_version": document["source_version"],
                    "operation_id": "masked-mode-compensation",
                },
            )

        assert response.status_code == 422
        assert "memory document mode could not be restored" in response.json()["detail"]
        assert path.read_text(encoding="utf-8") == content
        assert path.stat().st_mode & 0o7777 == 0o755
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM graph_cleanup_intents WHERE library_id = ?",
                (library_id,),
            ).fetchone() == (1,)


def test_failure_after_graph_purge_restores_every_store(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Retained\n\nGraph compensation must restore this projection.\n"
    path = root / "retained.md"
    path.write_text(content, encoding="utf-8")
    adapter = _RecordingGraphAdapter()

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        state.graph_adapter = adapter
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        source_version = hashlib.sha256(content.encode()).hexdigest()
        database = state_dir / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            document_id = connection.execute(
                "SELECT id FROM memory_documents WHERE library_id = ?", (library_id,)
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO memory_graph_documents
                   (library_id, document_id, path, source_version) VALUES (?, ?, ?, ?)""",
                (library_id, document_id, "retained.md", source_version),
            )
            connection.commit()
        projected = (GraphSourceDocument(str(document_id), "retained.md", source_version, content),)
        adapter.documents[library_id] = projected
        before = client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
            "commit"
        ]

        def fail_operation_insert(*args: object, **kwargs: object) -> NoReturn:
            del args, kwargs
            raise MemoryMutationError("injected operation insert failure")

        monkeypatch.setattr(state, "_insert_memory_operation", fail_operation_insert)
        response = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "retained.md",
                "expected_source_version": source_version,
                "operation_id": "fail-after-graph-purge",
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "injected operation insert failure"
        assert path.read_text(encoding="utf-8") == content
        assert adapter.documents[library_id] == projected
        assert client.get(f"/api/v1/libraries/{library_id}/forgotten", headers=headers).json() == []
        assert (
            client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
                "commit"
            ]
            == before
        )
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_documents WHERE library_id = ?", (library_id,)
            ).fetchone() == (1,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_graph_documents WHERE library_id = ?",
                (library_id,),
            ).fetchone() == (1,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_tombstones WHERE library_id = ?", (library_id,)
            ).fetchone() == (0,)


def test_forgotten_restore_rejects_nested_parent_symlink(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    nested = root / "nested"
    outside = tmp_path / "outside"
    nested.mkdir(parents=True)
    outside.mkdir()
    content = "# Nested\n\nNever restore outside the library.\n"
    path = nested / "private.md"
    path.write_text(content, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "nested/private.md",
                "expected_source_version": hashlib.sha256(content.encode()).hexdigest(),
                "operation_id": "delete-nested-before-symlink",
            },
        )
        assert deleted.status_code == 200, deleted.text
        nested.rmdir()
        nested.symlink_to(outside, target_is_directory=True)

        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted.json()["tombstone_id"],
                "commit": initial_commit,
                "operation_id": "reject-parent-symlink",
            },
        )

        assert restored.status_code == 422
        assert "opened safely" in restored.json()["detail"]
        assert not (outside / "private.md").exists()
        assert (
            client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
                "commit"
            ]
            == deleted.json()["commit"]
        )
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_tombstones WHERE id = ?",
                (deleted.json()["tombstone_id"],),
            ).fetchone() == (1,)


def test_forget_rejects_preexisting_tombstone_marker_symlink(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Private\n\nNever write a tombstone outside the library.\n"
    document_path = root / "private.md"
    document_path.write_text(content, encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("outside remains unchanged\n", encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        before = client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
            "commit"
        ]
        fixed_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        marker_directory = root / ".personal-agent-memory-tombstones"
        marker_directory.mkdir()
        (marker_directory / f"{fixed_id.hex}.json").symlink_to(outside)
        monkeypatch.setattr("personal_agent_memory.state.uuid.uuid4", lambda: fixed_id)

        response = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "private.md",
                "expected_source_version": hashlib.sha256(content.encode()).hexdigest(),
                "operation_id": "reject-marker-symlink",
                "actor_type": "user",
                "source": "test",
            },
        )

        assert response.status_code == 422
        assert "tombstone marker cannot be created safely" in response.json()["detail"]
        assert outside.read_text(encoding="utf-8") == "outside remains unchanged\n"
        assert document_path.read_text(encoding="utf-8") == content
        assert (
            client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
                "commit"
            ]
            == before
        )
        assert client.get(f"/api/v1/libraries/{library_id}/forgotten", headers=headers).json() == []


def test_forget_compensates_concurrent_tombstone_directory_replacement(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    content = "# Private\n\nRestore this document when the marker directory moves.\n"
    document_path = root / "private.md"
    document_path.write_text(content, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        before = client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
            "commit"
        ]
        original_verify = state._verify_bound_tombstone_marker
        calls = 0
        detached = root / "detached-tombstones"

        def replace_directory(*args: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                (root / ".personal-agent-memory-tombstones").rename(detached)
                (root / ".personal-agent-memory-tombstones").symlink_to(
                    outside, target_is_directory=True
                )
            original_verify(*args)  # type: ignore[arg-type]

        monkeypatch.setattr(state, "_verify_bound_tombstone_marker", replace_directory)
        response = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "private.md",
                "expected_source_version": hashlib.sha256(content.encode()).hexdigest(),
                "operation_id": "reject-marker-directory-race",
                "actor_type": "user",
                "source": "test",
            },
        )

        assert response.status_code == 422
        assert "tombstone directory changed" in response.json()["detail"]
        assert calls == 2
        assert document_path.read_text(encoding="utf-8") == content
        assert list(outside.iterdir()) == []
        assert list(detached.iterdir()) == []
        assert (
            client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
                "commit"
            ]
            == before
        )
        assert client.get(f"/api/v1/libraries/{library_id}/forgotten", headers=headers).json() == []
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_documents WHERE library_id = ?",
                (library_id,),
            ).fetchone() == (1,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_tombstones WHERE library_id = ?",
                (library_id,),
            ).fetchone() == (0,)


def test_forgotten_restore_compensates_concurrent_parent_replacement(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    nested = root / "nested"
    detached = root / "detached"
    outside = tmp_path / "outside"
    nested.mkdir(parents=True)
    outside.mkdir()
    content = "# Nested\n\nThe bound parent must remain authoritative.\n"
    (nested / "private.md").write_text(content, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "nested/private.md",
                "expected_source_version": hashlib.sha256(content.encode()).hexdigest(),
                "operation_id": "delete-nested-before-race",
            },
        )
        assert deleted.status_code == 200, deleted.text
        original_scan = state._trusted_library_scan_for_creation

        def replace_parent(*args: object, **kwargs: object):
            scan = original_scan(*args, **kwargs)
            nested.rename(detached)
            nested.symlink_to(outside, target_is_directory=True)
            return scan

        monkeypatch.setattr(state, "_trusted_library_scan_for_creation", replace_parent)
        restored = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted.json()["tombstone_id"],
                "commit": initial_commit,
                "operation_id": "reject-parent-replacement",
            },
        )

        assert restored.status_code == 422
        assert "document path changed" in restored.json()["detail"]
        assert not (outside / "private.md").exists()
        assert not (detached / "private.md").exists()
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_tombstones WHERE id = ?",
                (deleted.json()["tombstone_id"],),
            ).fetchone() == (1,)


def test_tombstone_fingerprint_preserves_distinct_heading_only_bodies(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    first = "# Alpha\n"
    second = "# Beta\n"
    path = root / "heading.md"
    path.write_text(first, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        first_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        edited = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "heading.md",
                "content": second,
                "expected_source_version": hashlib.sha256(first.encode()).hexdigest(),
                "operation_id": "replace-heading-only-body",
                "actor_type": "user",
                "source": "web",
            },
        )
        assert edited.status_code == 200, edited.text
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "heading.md",
                "expected_source_version": edited.json()["source_version"],
                "operation_id": "forget-beta-heading",
            },
        )
        assert deleted.status_code == 200, deleted.text

        wrong_restore = client.post(
            f"/api/v1/libraries/{library_id}/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": deleted.json()["tombstone_id"],
                "commit": first_commit,
                "operation_id": "do-not-authorize-alpha",
            },
        )
        assert wrong_restore.status_code == 422
        assert wrong_restore.json()["detail"] == (
            "selected history does not match forgotten memory"
        )
        distinct = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": first,
                "source_references": ["codex-session:distinct-heading#assistant-final"],
                "creator": "codex",
                "idempotency_key": "distinct-heading-body",
            },
        )
        assert distinct.status_code == 201, distinct.text
        equivalent = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "# BETA   \n",
                "source_references": ["codex-session:equivalent-heading#assistant-final"],
                "creator": "codex",
                "idempotency_key": "equivalent-heading-body",
            },
        )
        assert equivalent.status_code == 422
        assert equivalent.json()["detail"] == "candidate matches forgotten memory"


def test_tombstone_source_scope_never_copies_arbitrary_candidate_references(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    arbitrary_reference = "project:internal roadmap sentence that must not be retained"

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "# Decision\n\nUse the stable release channel.\n",
                "source_references": [arbitrary_reference],
                "creator": "codex",
                "idempotency_key": "arbitrary-source-reference",
            },
        ).json()
        approved = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json={
                "operator": "epq",
                "reason": "Confirmed durable decision",
                "operation_id": "approve-arbitrary-source",
            },
        )
        assert approved.status_code == 200, approved.text
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": approved.json()["published_path"]},
        ).json()
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": document["path"],
                "expected_source_version": document["source_version"],
                "operation_id": "forget-arbitrary-source",
            },
        )
        assert deleted.status_code == 200, deleted.text

        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            source_scope, marker_path, git_dir = connection.execute(
                """SELECT tombstone.source_scope_json, tombstone.marker_path, repo.git_dir
                   FROM memory_tombstones AS tombstone
                   JOIN library_git_repositories AS repo
                     ON repo.library_id = tombstone.library_id
                   WHERE tombstone.id = ?""",
                (deleted.json()["tombstone_id"],),
            ).fetchone()
        assert json.loads(source_scope) == {
            "capture_ranges": [],
            "document_path": document["path"],
        }
        marker = (root / marker_path).read_text(encoding="utf-8")
        committed_marker = subprocess.run(
            ["git", f"--git-dir={git_dir}", "show", f"HEAD:{marker_path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert arbitrary_reference not in source_scope
        assert arbitrary_reference not in marker
        assert arbitrary_reference not in committed_marker


def test_partial_graph_quarantine_cleanup_rolls_back_complete_projection(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    graph_root = tmp_path / "graph"
    root.mkdir()
    content = "# Retain\n\nGraph cleanup must be complete before forgetting succeeds.\n"
    path = root / "forget.md"
    path.write_text(content, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        state.graph_adapter = JiuwenMilvusGraphAdapter(
            graph_root, cast(OpenAICompatibleClient, object())
        )
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": content,
                "source_references": ["codex-session:graph-cleanup#assistant-final"],
                "creator": "codex",
                "idempotency_key": "candidate-before-graph-cleanup-failure",
            },
        )
        assert candidate.status_code == 201, candidate.text
        database = state_dir / "platform.sqlite3"
        with sqlite3.connect(database) as setup:
            setup.execute(
                """INSERT INTO capture_rounds
                   (session_id, turn_id, library_id, status, candidate_id)
                   VALUES ('graph-cleanup', 'turn-1', ?, 'done', ?)""",
                (library_id, candidate.json()["id"]),
            )
            setup.execute(
                """INSERT INTO capture_inbox
                   (event_id, request_hash, session_id, project_id, library_id, turn_id,
                    event_kind, content, occurred_at)
                   VALUES ('graph-cleanup-event', 'request-hash', 'graph-cleanup', 'project',
                           ?, 'turn-1', 'assistant', ?, '2026-09-06T00:00:00Z')""",
                (library_id, content),
            )
            setup.commit()
        projection = graph_root / library_id
        projection.mkdir(parents=True)
        (projection / "graph.db").write_text("forgotten graph body", encoding="utf-8")
        (projection / "projection.ready").write_text("ready\n", encoding="ascii")
        stale_a = graph_root / f".{library_id}.replacement-a"
        stale_b = graph_root / f".{library_id}.replacement-b"
        stale_a.mkdir()
        stale_b.mkdir()
        (stale_a / "stale-a").write_text("disposable", encoding="ascii")
        (stale_b / "stale-b").write_text("restore until retry", encoding="ascii")
        initial_head = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        with sqlite3.connect(database) as connection:
            before = {
                "candidate": connection.execute(
                    "SELECT * FROM candidate_memories WHERE id = ?", (candidate.json()["id"],)
                ).fetchall(),
                "governance": connection.execute(
                    "SELECT * FROM candidate_governance WHERE candidate_id = ?",
                    (candidate.json()["id"],),
                ).fetchall(),
                "audit": connection.execute(
                    "SELECT * FROM candidate_audit WHERE candidate_id = ?",
                    (candidate.json()["id"],),
                ).fetchall(),
                "round": connection.execute(
                    "SELECT * FROM capture_rounds WHERE session_id = 'graph-cleanup'"
                ).fetchall(),
                "event": connection.execute(
                    "SELECT * FROM capture_inbox WHERE event_id = 'graph-cleanup-event'"
                ).fetchall(),
                "documents": connection.execute(
                    "SELECT * FROM memory_documents WHERE library_id = ?", (library_id,)
                ).fetchall(),
                "index": connection.execute(
                    "SELECT * FROM memory_graph_indexes WHERE library_id = ?", (library_id,)
                ).fetchall(),
            }
        real_rmtree = shutil.rmtree

        def partial_cleanup(directory: Path, *args: object, **kwargs: object) -> None:
            directory = Path(directory)
            if (directory / "stale-b").exists():
                os.unlink(directory / "stale-b")
                raise OSError("injected cleanup interruption")
            real_rmtree(directory, *args, **kwargs)

        monkeypatch.setattr("personal_agent_memory.graph_adapter.shutil.rmtree", partial_cleanup)
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "forget.md",
                "expected_source_version": hashlib.sha256(content.encode()).hexdigest(),
                "operation_id": "partial-quarantine-cleanup",
            },
        )

        assert deleted.status_code == 422, deleted.text
        assert "graph projection cleanup failed" in deleted.json()["detail"]
        assert path.read_text(encoding="utf-8") == content
        assert (projection / "graph.db").read_text(encoding="utf-8") == "forgotten graph body"
        assert (projection / "projection.ready").read_text(encoding="ascii") == "ready\n"
        assert not list(graph_root.glob(f".{library_id}.purge-*"))
        assert client.get(f"/api/v1/libraries/{library_id}/forgotten", headers=headers).json() == []
        assert not stale_a.exists()
        assert (stale_b / "stale-b").read_text(encoding="ascii") == "restore until retry"
        assert (
            client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
                "commit"
            ]
            == initial_head
        )
        with sqlite3.connect(database) as connection:
            after = {
                "candidate": connection.execute(
                    "SELECT * FROM candidate_memories WHERE id = ?", (candidate.json()["id"],)
                ).fetchall(),
                "governance": connection.execute(
                    "SELECT * FROM candidate_governance WHERE candidate_id = ?",
                    (candidate.json()["id"],),
                ).fetchall(),
                "audit": connection.execute(
                    "SELECT * FROM candidate_audit WHERE candidate_id = ?",
                    (candidate.json()["id"],),
                ).fetchall(),
                "round": connection.execute(
                    "SELECT * FROM capture_rounds WHERE session_id = 'graph-cleanup'"
                ).fetchall(),
                "event": connection.execute(
                    "SELECT * FROM capture_inbox WHERE event_id = 'graph-cleanup-event'"
                ).fetchall(),
                "documents": connection.execute(
                    "SELECT * FROM memory_documents WHERE library_id = ?", (library_id,)
                ).fetchall(),
                "index": connection.execute(
                    "SELECT * FROM memory_graph_indexes WHERE library_id = ?", (library_id,)
                ).fetchall(),
            }
            assert after == before
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_tombstones WHERE library_id = ?", (library_id,)
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_operations WHERE operation_id = ?",
                ("partial-quarantine-cleanup",),
            ).fetchone() == (0,)
        assert client.get("/api/v1/capture/events", headers=headers).json()[0]["content"] == content

        monkeypatch.setattr("personal_agent_memory.graph_adapter.shutil.rmtree", real_rmtree)
        retried = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "forget.md",
                "expected_source_version": hashlib.sha256(content.encode()).hexdigest(),
                "operation_id": "partial-quarantine-cleanup",
            },
        )
        assert retried.status_code == 200, retried.text
        assert not projection.exists()
        assert not list(graph_root.glob(f".{library_id}.replacement-*"))
        assert not list(graph_root.glob(f".{library_id}.purge-*"))


def test_replacement_rollback_restores_old_projection_before_partial_cleanup(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    graph_root = tmp_path / "graph"
    library_id = "library-one"
    projection = graph_root / library_id
    projection.mkdir(parents=True)
    old_files = {
        "graph.db": b"old graph includes the retained authoritative projection",
        "projection.ready": b"ready\n",
        "sidecar": b"old sidecar",
    }
    for name, payload in old_files.items():
        (projection / name).write_bytes(payload)
    adapter = JiuwenMilvusGraphAdapter(graph_root, cast(OpenAICompatibleClient, object()))

    def build_replacement(
        temporary_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
        graph_library_id: str | None = None,
    ) -> None:
        del extracted, graph_library_id
        temporary = graph_root / temporary_id
        temporary.mkdir(parents=True)
        (temporary / "graph.db").write_bytes(b"surviving projection only")
        (temporary / "projection.ready").write_text("ready\n", encoding="ascii")

    monkeypatch.setattr(adapter, "_extract_documents", lambda documents: [])
    monkeypatch.setattr(adapter, "_replace_projection", build_replacement)
    replacement = adapter.stage_rebuild(library_id, ())

    real_rmtree = shutil.rmtree

    def partial_cleanup(directory: Path) -> None:
        if ".rollback-" in directory.name:
            os.unlink(directory / "graph.db")
            raise OSError("injected replacement cleanup interruption")
        real_rmtree(directory)

    monkeypatch.setattr("personal_agent_memory.graph_adapter.shutil.rmtree", partial_cleanup)
    try:
        replacement.rollback()
    except GraphAdapterError as error:
        assert "graph projection cleanup failed" in str(error)
    else:
        raise AssertionError("partial replacement cleanup must be reported")

    assert {path.name: path.read_bytes() for path in projection.iterdir()} == old_files
    quarantines = list(graph_root.glob(f".{library_id}.purge-*"))
    assert quarantines == []
    detached = list(graph_root.glob(f".{library_id}.rollback-*"))
    assert len(detached) == 1
    assert b"retained authoritative" not in b"".join(
        path.read_bytes() for path in detached[0].rglob("*") if path.is_file()
    )

    monkeypatch.setattr("personal_agent_memory.graph_adapter.shutil.rmtree", real_rmtree)
    replacement.rollback()
    assert {path.name: path.read_bytes() for path in projection.iterdir()} == old_files
    assert not list(graph_root.glob(f".{library_id}.rollback-*"))


def test_formal_memory_replay_and_preexisting_candidate_reapproval_are_blocked(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    body = "# Decision\n\nKeep the audit database on Firebird 5.\n"

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        original = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": body,
                "source_references": ["codex-session:original#assistant-final"],
                "creator": "codex",
                "idempotency_key": "formal-replay-original",
            },
        ).json()
        approved = client.post(
            f"/api/v1/candidates/{original['id']}/approve",
            headers=headers,
            json={
                "operator": "epq",
                "reason": "Confirmed",
                "operation_id": "approve-formal-replay-original",
            },
        )
        assert approved.status_code == 200, approved.text
        pending = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": body,
                "source_references": ["codex-session:pending#assistant-final"],
                "creator": "codex",
                "idempotency_key": "formal-replay-pending",
            },
        )
        assert pending.status_code == 201, pending.text
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": approved.json()["published_path"]},
        ).json()
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": document["path"],
                "expected_source_version": document["source_version"],
                "operation_id": "delete-formal-replay-original",
            },
        )
        assert deleted.status_code == 200, deleted.text

        replay_id = str(uuid.uuid4())
        formal_replay = "\n".join(
            (
                "---",
                f"memory_id: {json.dumps(replay_id)}",
                'memory_type: "decision"',
                f"candidate_id: {json.dumps(replay_id)}",
                'created_by: "different-creator"',
                'approved_by: "different-operator"',
                'approval_reason: "different reason"',
                "source_references:",
                '  - "codex-session:replayed#assistant-final"',
                "---",
                "",
                body.rstrip(),
                "",
            )
        )
        replay = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": formal_replay,
                "source_references": ["codex-session:new-envelope#assistant-final"],
                "creator": "codex",
                "idempotency_key": "formal-replay-new-envelope",
            },
        )
        assert replay.status_code == 422
        assert replay.json()["detail"] == "candidate matches forgotten memory"
        reordered = "\r\n".join(
            (
                "---",
                'approved_by : "different-operator"',
                f"candidate_id: {json.dumps(replay_id)}",
                "format_version: 2",
                'approval_reason: "different reason"',
                'memory_type: "decision"',
                "source_references:",
                '    - "codex-session:reordered#assistant-final"',
                f"memory_id: {json.dumps(replay_id)}",
                'created_by: "different-creator"',
                "---",
                "",
                body.rstrip(),
                "",
            )
        )
        reordered_replay = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": reordered,
                "source_references": ["codex-session:reordered-envelope#assistant-final"],
                "creator": "codex",
                "idempotency_key": "formal-replay-reordered-envelope",
            },
        )
        assert reordered_replay.status_code == 422
        assert reordered_replay.json()["detail"] == "candidate matches forgotten memory"

        duplicate_key = formal_replay.replace(
            f"memory_id: {json.dumps(replay_id)}",
            f"memory_id: {json.dumps(replay_id)}\nmemory_id: {json.dumps(str(uuid.uuid4()))}",
            1,
        )
        ambiguous = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": duplicate_key,
                "source_references": ["codex-session:ambiguous#assistant-final"],
                "creator": "codex",
                "idempotency_key": "formal-replay-ambiguous-envelope",
            },
        )
        assert ambiguous.status_code == 422
        assert ambiguous.json()["detail"] == "candidate has invalid formal memory metadata"

        nested_extra = reordered.replace("format_version: 2", 'format_version: {"major": 2}')
        malicious = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": nested_extra,
                "source_references": ["codex-session:nested#assistant-final"],
                "creator": "codex",
                "idempotency_key": "formal-replay-nested-envelope",
            },
        )
        assert malicious.status_code == 422
        assert malicious.json()["detail"] == "candidate has invalid formal memory metadata"

        project = tmp_path / "project"
        project.mkdir()
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library_id},
        )
        assert binding.status_code == 201, binding.text
        state = cast(PlatformState, client.app.state.platform_state)
        state.model_client.extract_candidate = lambda _: {
            "eligible": True,
            "suggested_type": "decision",
            "body": reordered,
        }
        for event_id, kind, content in (
            ("formal-replay-user", "user", "Remember the deleted decision."),
            ("formal-replay-assistant", "assistant", "The decision is confirmed."),
        ):
            captured = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json={
                    "event_id": event_id,
                    "session_id": "formal-replay-capture",
                    "turn_id": "turn-1",
                    "event_kind": kind,
                    "content": content,
                    "occurred_at": "2026-09-05T05:00:00Z",
                    "cwd": str(project),
                },
            )
            assert captured.status_code == 202, captured.text
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rounds = client.get(
                "/api/v1/capture/rounds",
                headers=headers,
                params={"session_id": "formal-replay-capture"},
            ).json()
            if rounds and rounds[0]["status"] == "done":
                break
            time.sleep(0.05)
        assert rounds[0]["status"] == "done"
        assert rounds[0]["candidate_id"] is None

        ordinary_markdown = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "---\ntitle: Ordinary note\npriority: 2\n---\n\n# Different\n\nKeep it.\n",
                "source_references": ["codex-session:ordinary#assistant-final"],
                "creator": "codex",
                "idempotency_key": "ordinary-frontmatter-is-not-formal-memory",
            },
        )
        assert ordinary_markdown.status_code == 201, ordinary_markdown.text
        reapprove = client.post(
            f"/api/v1/candidates/{pending.json()['id']}/approve",
            headers=headers,
            json={
                "operator": "epq",
                "reason": "Must remain forgotten",
                "operation_id": "approve-preexisting-replay",
            },
        )
        assert reapprove.status_code == 404
        assert reapprove.json()["detail"] == "candidate memory not found"


def test_forgetting_removes_equivalent_candidates_audit_and_capture_links(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    body = "# Durable deletion sentinel\n\nForget CandidateAuditNeedle forever.\n"
    (root / "forgotten.md").write_text(body, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        platform_id = str(uuid.uuid4())
        equivalent_bodies = (
            body,
            "---\ntitle: Rewrapped note\npriority: 7\n---\n\n" + body,
            "\n".join(
                (
                    "---",
                    f"memory_id: {json.dumps(platform_id)}",
                    'memory_type: "decision"',
                    f"candidate_id: {json.dumps(platform_id)}",
                    'created_by: "codex"',
                    'approved_by: "epq"',
                    'approval_reason: "confirmed"',
                    "source_references:",
                    '  - "codex-session:equivalent#assistant-final"',
                    "---",
                    "",
                    body.rstrip(),
                    "",
                )
            ),
        )
        candidates = []
        for index, candidate_body in enumerate(equivalent_bodies):
            response = client.post(
                "/mcp/candidates",
                headers=headers,
                json={
                    "library_id": library_id,
                    "suggested_type": "decision",
                    "body": candidate_body,
                    "source_references": [f"codex-session:equivalent-{index}#assistant-final"],
                    "creator": "codex",
                    "idempotency_key": f"equivalent-before-forget-{index}",
                },
            )
            assert response.status_code == 201, response.text
            candidates.append(response.json())

        audit_only = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": body,
                "source_references": ["codex-session:audit-only#assistant-final"],
                "creator": "codex",
                "idempotency_key": "audit-only-before-forget",
            },
        )
        assert audit_only.status_code == 201, audit_only.text
        edited = client.put(
            f"/api/v1/candidates/{audit_only.json()['id']}",
            headers=headers,
            json={
                "body": (
                    "# Unrelated current body\n\n"
                    "This candidate is retained only in old audit text.\n"
                ),
                "operator": "epq",
                "reason": "change current body",
            },
        )
        assert edited.status_code == 200, edited.text
        candidates.append(audit_only.json())

        unrelated = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "domain_fact",
                "body": "# Keep\n\nThis candidate is unrelated.\n",
                "source_references": ["codex-session:unrelated#assistant-final"],
                "creator": "codex",
                "idempotency_key": "unrelated-before-forget",
            },
        )
        assert unrelated.status_code == 201, unrelated.text

        other_root = tmp_path / "other"
        other_root.mkdir()
        other_library_id = _register(client, headers, other_root, "user")
        other_library = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": other_library_id,
                "suggested_type": "decision",
                "body": body,
                "source_references": ["codex-session:other-library#assistant-final"],
                "creator": "codex",
                "idempotency_key": "same-body-other-library",
            },
        )
        assert other_library.status_code == 201, other_library.text

        database = state_dir / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute(
                """INSERT INTO capture_rounds
                   (session_id, turn_id, library_id, status, candidate_id)
                   VALUES ('forgotten-capture', 'turn-1', ?, 'done', ?)""",
                (library_id, candidates[0]["id"]),
            )
            connection.commit()

        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "forgotten.md"},
        ).json()
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "forgotten.md",
                "expected_source_version": document["source_version"],
                "operation_id": "forget-equivalent-candidates",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert deleted.status_code == 200, deleted.text

        forgotten_ids = tuple(str(candidate["id"]) for candidate in candidates)
        with sqlite3.connect(database) as connection:
            placeholders = ",".join("?" for _ in forgotten_ids)
            assert connection.execute(
                f"SELECT COUNT(*) FROM candidate_memories WHERE id IN ({placeholders})",  # noqa: S608
                forgotten_ids,
            ).fetchone() == (0,)
            assert connection.execute(
                f"SELECT COUNT(*) FROM candidate_governance WHERE candidate_id IN ({placeholders})",  # noqa: S608
                forgotten_ids,
            ).fetchone() == (0,)
            assert connection.execute(
                f"SELECT COUNT(*) FROM candidate_audit WHERE candidate_id IN ({placeholders})",  # noqa: S608
                forgotten_ids,
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT candidate_id FROM capture_rounds "
                "WHERE session_id = 'forgotten-capture' AND turn_id = 'turn-1'"
            ).fetchone() == (None,)
            assert connection.execute(
                "SELECT COUNT(*) FROM candidate_memories WHERE id = ?",
                (unrelated.json()["id"],),
            ).fetchone() == (1,)
            assert connection.execute(
                "SELECT COUNT(*) FROM candidate_memories WHERE id = ?",
                (other_library.json()["id"],),
            ).fetchone() == (1,)


def test_forgetting_removes_all_same_library_authoritative_copies(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    other_root = tmp_path / "other"
    root.mkdir()
    other_root.mkdir()
    body = "# Durable deletion sentinel\n\nForget AuthorityCopyNeedle-19d8 forever.\n"
    ordinary = "---\ntitle: Rewrapped copy\npriority: 3\n---\n\n" + body
    platform_id = str(uuid.uuid4())
    platform = "\n".join(
        (
            "---",
            f"memory_id: {json.dumps(platform_id)}",
            'memory_type: "decision"',
            f"candidate_id: {json.dumps(platform_id)}",
            'created_by: "codex"',
            'approved_by: "epq"',
            'approval_reason: "confirmed"',
            "source_references:",
            '  - "codex-session:copy#assistant-final"',
            "---",
            "",
            body.rstrip(),
            "",
        )
    )
    copies = {
        "forgotten.md": body,
        "ordinary.md": ordinary,
        "platform.md": platform,
    }
    for name, content in copies.items():
        (root / name).write_text(content, encoding="utf-8")
    unrelated = "# Durable deletion sentinel\n\nKeep this distinct content.\n"
    (root / "unrelated.md").write_text(unrelated, encoding="utf-8")
    (other_root / "same.md").write_text(body, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        other_library_id = _register(client, headers, other_root, "user")
        initial_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "forgotten.md"},
        ).json()
        preview = client.post(
            f"/api/v1/libraries/{library_id}/document/delete-preview",
            headers=headers,
            json={
                "path": "forgotten.md",
                "expected_source_version": document["source_version"],
                "operation_id": "preview-authoritative-copies",
            },
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["authoritative_copies"] == [
            {
                "path": name,
                "source_version": hashlib.sha256(content.encode()).hexdigest(),
            }
            for name, content in sorted(copies.items())
        ]

        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "forgotten.md",
                "expected_source_version": document["source_version"],
                "operation_id": "forget-authoritative-copies",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert deleted.status_code == 200, deleted.text
        assert all(not (root / name).exists() for name in copies)
        assert (root / "unrelated.md").read_text(encoding="utf-8") == unrelated
        assert (other_root / "same.md").read_text(encoding="utf-8") == body
        for include_history in (False, True):
            results = client.post(
                "/api/v1/search",
                headers=headers,
                json={
                    "library_id": library_id,
                    "query": "AuthorityCopyNeedle",
                    "include_history": include_history,
                },
            )
            assert results.status_code == 200, results.text
            assert results.json()["results"] == []
        initial_diff = client.get(
            f"/api/v1/libraries/{library_id}/history/{initial_commit}/diff",
            headers=headers,
        )
        assert initial_diff.status_code == 200, initial_diff.text
        assert initial_diff.json()["diff"].startswith("[redacted:")
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            placeholders = ",".join("?" for _ in copies)
            parameters = (library_id, *copies)
            for table in (
                "memory_documents",
                "memory_chunks",
                "memory_versions",
                "memory_graph_documents",
            ):
                assert connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE library_id = ? "
                    f"AND path IN ({placeholders})",  # noqa: S608
                    parameters,
                ).fetchone() == (0,)
            scope = json.loads(
                connection.execute(
                    "SELECT source_scope_json FROM memory_tombstones WHERE id = ?",
                    (deleted.json()["tombstone_id"],),
                ).fetchone()[0]
            )
            other_count = connection.execute(
                "SELECT COUNT(*) FROM memory_documents WHERE library_id = ? AND path = 'same.md'",
                (other_library_id,),
            ).fetchone()
        assert scope["authoritative_paths"] == sorted(copies)
        assert other_count == (1,)

    with TestClient(create_app(settings)) as restarted:
        headers = _headers(state_dir)
        for name in copies:
            assert (
                restarted.get(
                    f"/api/v1/libraries/{library_id}/document",
                    headers=headers,
                    params={"path": name},
                ).status_code
                == 404
            )
        assert (
            restarted.get(
                f"/api/v1/libraries/{other_library_id}/document",
                headers=headers,
                params={"path": "same.md"},
            ).status_code
            == 200
        )


def test_delete_requires_current_complete_preview_snapshot(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    other_root = tmp_path / "other"
    project = tmp_path / "project"
    root.mkdir()
    other_root.mkdir()
    project.mkdir()
    body = "# Preview-bound forgetting\n\nForget PreviewSnapshotNeedle-72c1 completely.\n"
    distinct = "# Distinct\n\nKeep this document.\n"
    (root / "target.md").write_text(body, encoding="utf-8")
    (root / "later-copy.md").write_text(distinct, encoding="utf-8")
    (root / "other-document.md").write_text(distinct + "Still distinct.\n", encoding="utf-8")
    (other_root / "same.md").write_text(body, encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        other_library_id = _register(client, headers, other_root, "user")
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library_id},
        )
        assert binding.status_code == 201, binding.text
        target = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "target.md"},
        ).json()
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": body,
                "source_references": ["session:preview-snapshot#assistant-final"],
                "creator": "codex",
                "idempotency_key": "preview-snapshot-candidate",
            },
        )
        assert candidate.status_code == 201, candidate.text
        candidate_id = candidate.json()["id"]
        captured = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "preview-snapshot-event",
                "session_id": "preview-snapshot-session",
                "turn_id": "turn-1",
                "event_kind": "assistant",
                "content": "PreviewCaptureNeedle-55ee",
                "occurred_at": "2026-09-06T01:00:00Z",
                "cwd": str(project),
            },
        )
        assert captured.status_code == 202, captured.text
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute(
                """UPDATE capture_rounds SET status = 'done', candidate_id = ?
                   WHERE session_id = 'preview-snapshot-session' AND turn_id = 'turn-1'""",
                (candidate_id,),
            )
            connection.commit()

        def preview(
            preview_library_id: str, path: str, version: str
        ) -> dict[str, object]:
            response = client.post(
                f"/api/v1/libraries/{preview_library_id}/document/delete-preview",
                headers=headers,
                json={
                    "path": path,
                    "expected_source_version": version,
                    "operation_id": f"preview-{path}-{uuid.uuid4()}",
                },
            )
            assert response.status_code == 200, response.text
            return cast(dict[str, object], response.json())

        initial_preview = preview(library_id, "target.md", target["source_version"])
        initial_token = str(initial_preview["preview_token"])
        assert initial_token.startswith("pam-delete-preview-v1:")
        assert body not in initial_token
        assert initial_preview["authoritative_copies"] == [
            {"path": "target.md", "source_version": target["source_version"]}
        ]
        initial_candidates = cast(list[dict[str, object]], initial_preview["candidate_impacts"])
        assert initial_candidates[0]["id"] == candidate_id
        assert initial_candidates[0]["body"] == body
        assert initial_candidates[0]["source_references"] == [
            "session:preview-snapshot#assistant-final"
        ]
        assert len(cast(list[object], initial_candidates[0]["audit"])) == 1
        assert initial_preview["capture_impacts"] == [
            {
                "event_id": "preview-snapshot-event",
                "session_id": "preview-snapshot-session",
                "turn_id": "turn-1",
                "event_kind": "assistant",
                "content": "PreviewCaptureNeedle-55ee",
                "occurred_at": "2026-09-06T01:00:00Z",
                "candidate_id": candidate_id,
            }
        ]

        later_copy = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "later-copy.md"},
        ).json()
        changed = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "later-copy.md",
                "content": body,
                "expected_source_version": later_copy["source_version"],
                "operation_id": "make-later-authoritative-copy",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert changed.status_code == 200, changed.text

        def delete_with(
            token: str,
            *,
            path: str = "target.md",
            version: str = target["source_version"],
        ) -> object:
            return client.request(
                "DELETE",
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": path,
                    "expected_source_version": version,
                    "preview_token": token,
                    "operation_id": f"delete-{path}-{uuid.uuid4()}",
                    "actor_type": "user",
                    "source": "test",
                },
            )

        stale_copy = delete_with(initial_token)
        assert stale_copy.status_code == 409
        assert stale_copy.json()["detail"] == "deletion preview is stale; request a new preview"
        assert (root / "target.md").exists()
        assert (root / "later-copy.md").exists()
        assert client.get(f"/mcp/candidates/{candidate_id}", headers=headers).status_code == 200
        assert any(
            event["event_id"] == "preview-snapshot-event"
            for event in client.get("/api/v1/capture/events", headers=headers).json()
        )

        copy_preview = preview(library_id, "target.md", target["source_version"])
        copy_token = str(copy_preview["preview_token"])
        edited_candidate = client.put(
            f"/api/v1/candidates/{candidate_id}",
            headers=headers,
            json={
                "body": "---\ntitle: Equivalent candidate wrapper\n---\n\n" + body,
                "operator": "epq",
                "reason": "Preserve an audit version",
            },
        )
        assert edited_candidate.status_code == 200, edited_candidate.text
        stale_candidate = delete_with(copy_token)
        assert stale_candidate.status_code == 409
        assert stale_candidate.json()["detail"] == (
            "deletion preview is stale; request a new preview"
        )

        current_preview = preview(library_id, "target.md", target["source_version"])
        current_token = str(current_preview["preview_token"])
        assert [
            item["path"]
            for item in cast(list[dict[str, object]], current_preview["authoritative_copies"])
        ] == ["later-copy.md", "target.md"]
        candidate_impact = cast(
            list[dict[str, object]], current_preview["candidate_impacts"]
        )[0]
        assert len(cast(list[object], candidate_impact["audit"])) == 2

        tampered = current_token[:-1] + ("0" if current_token[-1] != "0" else "1")
        assert delete_with(tampered).status_code == 409

        other_document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "other-document.md"},
        ).json()
        other_document_preview = preview(
            library_id, "other-document.md", other_document["source_version"]
        )
        assert (
            delete_with(
                str(other_document_preview["preview_token"]),
                version=target["source_version"],
            ).status_code
            == 409
        )

        other_library_document = client.get(
            f"/api/v1/libraries/{other_library_id}/document",
            headers=headers,
            params={"path": "same.md"},
        ).json()
        cross_library_preview = preview(
            other_library_id, "same.md", other_library_document["source_version"]
        )
        assert delete_with(str(cross_library_preview["preview_token"])).status_code == 409

        deleted = delete_with(current_token)
        assert deleted.status_code == 200, deleted.text
        assert not (root / "target.md").exists()
        assert not (root / "later-copy.md").exists()
        assert (root / "other-document.md").exists()
        assert (other_root / "same.md").exists()
        assert client.get(f"/mcp/candidates/{candidate_id}", headers=headers).status_code == 404
        assert all(
            event["event_id"] != "preview-snapshot-event"
            for event in client.get("/api/v1/capture/events", headers=headers).json()
        )


def test_raw_equivalent_capture_invalidates_preview_and_is_forgotten(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    project = tmp_path / "project"
    root.mkdir()
    project.mkdir()
    body = "# Raw capture forgetting\n\nForget RawCaptureNeedle-509d completely.\n"
    (root / "target.md").write_text(body, encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library_id},
        )
        assert binding.status_code == 201, binding.text
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "target.md"},
        ).json()
        stale_token = _deletion_preview_token(
            client, headers, library_id, "target.md", document["source_version"]
        )
        captured = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "raw-equivalent-event",
                "session_id": "raw-equivalent-session",
                "turn_id": "turn-1",
                "event_kind": "assistant",
                "content": body,
                "occurred_at": "2026-09-06T02:00:00Z",
                "cwd": str(project),
            },
        )
        assert captured.status_code == 202, captured.text

        stale_delete = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "target.md",
                "expected_source_version": document["source_version"],
                "preview_token": stale_token,
                "operation_id": "stale-raw-capture-delete",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert stale_delete.status_code == 409, stale_delete.text
        assert (root / "target.md").exists()

        preview = client.post(
            f"/api/v1/libraries/{library_id}/document/delete-preview",
            headers=headers,
            json={
                "path": "target.md",
                "expected_source_version": document["source_version"],
                "operation_id": "current-raw-capture-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["capture_impacts"] == [
            {
                "event_id": "raw-equivalent-event",
                "session_id": "raw-equivalent-session",
                "turn_id": "turn-1",
                "event_kind": "assistant",
                "content": body,
                "occurred_at": "2026-09-06T02:00:00Z",
                "candidate_id": None,
            }
        ]
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "target.md",
                "expected_source_version": document["source_version"],
                "preview_token": preview.json()["preview_token"],
                "operation_id": "forget-raw-capture",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert deleted.status_code == 200, deleted.text
        assert all(
            event["event_id"] != "raw-equivalent-event"
            for event in client.get("/api/v1/capture/events", headers=headers).json()
        )


def test_deletion_during_capture_extraction_does_not_recreate_candidate(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    project = tmp_path / "project"
    root.mkdir()
    project.mkdir()
    body = "# Extraction race\n\nForget ExtractionRaceNeedle-a44b completely.\n"
    (root / "target.md").write_text(body, encoding="utf-8")
    extraction_started = threading.Event()
    release_extraction = threading.Event()

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library_id},
        )
        assert binding.status_code == 201, binding.text
        state = client.app.state.platform_state

        def extract(_: str) -> dict[str, object]:
            extraction_started.set()
            assert release_extraction.wait(5)
            return {
                "eligible": True,
                "suggested_type": "decision",
                "body": body,
                "confidence": 0.99,
                "durability": "durable",
                "scope": "project",
                "explicit_confirmation": True,
            }

        state.model_client.extract_candidate = extract
        for event_id, event_kind, content in (
            ("extraction-race-user", "user", body),
            ("extraction-race-assistant", "assistant", "Confirmed."),
        ):
            captured = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json={
                    "event_id": event_id,
                    "session_id": "extraction-race-session",
                    "turn_id": "turn-1",
                    "event_kind": event_kind,
                    "content": content,
                    "occurred_at": "2026-09-06T03:00:00Z",
                    "cwd": str(project),
                },
            )
            assert captured.status_code == 202, captured.text
        assert extraction_started.wait(5)

        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "target.md"},
        ).json()
        token = _deletion_preview_token(
            client, headers, library_id, "target.md", document["source_version"]
        )
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "target.md",
                "expected_source_version": document["source_version"],
                "preview_token": token,
                "operation_id": "delete-during-extraction",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert deleted.status_code == 200, deleted.text
        release_extraction.set()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rounds = client.get(
                "/api/v1/capture/rounds",
                headers=headers,
                params={"session_id": "extraction-race-session"},
            ).json()
            if rounds and rounds[0]["status"] == "done":
                break
            time.sleep(0.02)
        assert rounds[0]["status"] == "done"
        assert rounds[0]["candidate_id"] is None
        assert client.get(
            "/api/v1/candidates", headers=headers, params={"library_id": library_id}
        ).json() == []


def test_delete_lock_serializes_same_library_writers_but_not_other_libraries(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    other_root = tmp_path / "other"
    project = tmp_path / "project"
    root.mkdir()
    other_root.mkdir()
    project.mkdir()
    body = "# Locked forgetting\n\nForget LockedWriterNeedle-6f81 completely.\n"
    (root / "target.md").write_text(body, encoding="utf-8")

    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))
    with (
        TestClient(create_app(settings)) as deleting_client,
        TestClient(create_app(settings)) as creating_client,
        TestClient(create_app(settings)) as editing_client,
        TestClient(create_app(settings)) as capturing_client,
        TestClient(create_app(settings)) as other_client,
    ):
        headers = _headers(state_dir)
        library_id = _register(deleting_client, headers, root)
        other_library_id = _register(deleting_client, headers, other_root, "user")
        binding = deleting_client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library_id},
        )
        assert binding.status_code == 201, binding.text
        retained = deleting_client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "# Keep\n\nUnrelated pending candidate.\n",
                "source_references": ["session:retained#assistant-final"],
                "creator": "codex",
                "idempotency_key": "retained-during-forget",
            },
        )
        assert retained.status_code == 201, retained.text
        document = deleting_client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "target.md"},
        ).json()
        token = _deletion_preview_token(
            deleting_client,
            headers,
            library_id,
            "target.md",
            document["source_version"],
        )
        state = deleting_client.app.state.platform_state
        original_token = state._document_deletion_preview_token
        delete_has_lock = threading.Event()
        release_delete = threading.Event()
        paused = False

        def pause_delete(impact: object) -> str:
            nonlocal paused
            generated = original_token(impact)
            if not paused:
                paused = True
                delete_has_lock.set()
                assert release_delete.wait(5)
            return generated

        state._document_deletion_preview_token = pause_delete

        def delete() -> object:
            return deleting_client.request(
                "DELETE",
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": "target.md",
                    "expected_source_version": document["source_version"],
                    "preview_token": token,
                    "operation_id": "locked-delete",
                    "actor_type": "user",
                    "source": "test",
                },
            )

        def create_same_library() -> object:
            return creating_client.post(
                "/mcp/candidates",
                headers=headers,
                json={
                    "library_id": library_id,
                    "suggested_type": "decision",
                    "body": body,
                    "source_references": ["session:blocked-create#assistant-final"],
                    "creator": "codex",
                    "idempotency_key": "blocked-create-after-delete",
                },
            )

        def edit_same_library() -> object:
            return editing_client.put(
                f"/api/v1/candidates/{retained.json()['id']}",
                headers=headers,
                json={"body": body, "operator": "epq", "reason": "race edit"},
            )

        def capture_same_library() -> object:
            return capturing_client.post(
                "/api/v1/capture/events",
                headers=headers,
                json={
                    "event_id": "blocked-capture-after-delete",
                    "session_id": "blocked-capture-session",
                    "turn_id": "turn-1",
                    "event_kind": "assistant",
                    "content": body,
                    "occurred_at": "2026-09-06T04:00:00Z",
                    "cwd": str(project),
                },
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            deleting = executor.submit(delete)
            assert delete_has_lock.wait(5)
            creating = executor.submit(create_same_library)
            editing = executor.submit(edit_same_library)
            capturing = executor.submit(capture_same_library)
            other = executor.submit(
                other_client.post,
                "/mcp/candidates",
                headers=headers,
                json={
                    "library_id": other_library_id,
                    "suggested_type": "decision",
                    "body": body,
                    "source_references": ["session:other-library#assistant-final"],
                    "creator": "codex",
                    "idempotency_key": "other-library-during-delete",
                },
            )
            assert other.result(timeout=2).status_code == 201
            time.sleep(0.05)
            assert not creating.done()
            assert not editing.done()
            assert not capturing.done()
            release_delete.set()
            assert deleting.result(timeout=5).status_code == 200
            created_response = creating.result(timeout=5)
            edited_response = editing.result(timeout=5)
            captured_response = capturing.result(timeout=5)

        assert created_response.status_code == 422, created_response.text
        assert created_response.json()["detail"] == "candidate matches forgotten memory"
        assert edited_response.status_code == 422, edited_response.text
        assert edited_response.json()["detail"] == "candidate matches forgotten memory"
        assert captured_response.status_code == 422, captured_response.text
        assert captured_response.json()["detail"] == "capture event matches forgotten memory"


def test_busy_forgetting_checkpoint_compensates_and_same_operation_retries(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    graph_root = tmp_path / "graph"
    root = tmp_path / "memory"
    root.mkdir()
    sentinel = "PinnedCheckpointForgetNeedle-92ad0ca10de142bda733"
    body = f"# Forget\n\n{sentinel}\n"
    document_path = root / "forgotten.md"
    document_path.write_text(body, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))
    operation = {
        "path": "forgotten.md",
        "operation_id": "retry-after-busy-checkpoint",
        "actor_type": "user",
        "source": "test",
    }

    with TestClient(create_app(settings)) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        state.graph_adapter = JiuwenMilvusGraphAdapter(
            graph_root, cast(OpenAICompatibleClient, object())
        )
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": body,
                "source_references": ["codex-session:pinned-reader#assistant-final"],
                "creator": "codex",
                "idempotency_key": "candidate-before-pinned-reader",
            },
        )
        assert candidate.status_code == 201, candidate.text
        database = state_dir / "platform.sqlite3"
        with sqlite3.connect(database) as setup:
            setup.execute(
                """INSERT INTO capture_rounds
                   (session_id, turn_id, library_id, status, candidate_id)
                   VALUES ('pinned-capture', 'turn-1', ?, 'done', ?)""",
                (library_id, candidate.json()["id"]),
            )
            setup.execute(
                """INSERT INTO capture_inbox
                   (event_id, request_hash, session_id, project_id, library_id, turn_id,
                    event_kind, content, occurred_at)
                   VALUES ('pinned-event', 'request-hash', 'pinned-capture', 'project', ?,
                           'turn-1', 'assistant', ?, '2026-09-06T00:00:00Z')""",
                (library_id, body),
            )
            document_row = setup.execute(
                "SELECT id, path, source_version FROM memory_documents WHERE library_id = ?",
                (library_id,),
            ).fetchone()
            setup.execute(
                """INSERT INTO memory_graph_documents
                   (library_id, document_id, path, source_version)
                   VALUES (?, ?, ?, ?)""",
                (library_id, *document_row),
            )
            setup.execute(
                "UPDATE memory_graph_indexes SET status = 'ready', total_documents = 1, "
                "projected_documents = 1, last_error = '' WHERE library_id = ?",
                (library_id,),
            )
            setup.commit()
        projection = graph_root / library_id
        projection.mkdir(parents=True)
        (projection / "graph.db").write_text(body, encoding="utf-8")
        (projection / "projection.ready").write_text("ready\n", encoding="ascii")
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "forgotten.md"},
        ).json()
        operation["expected_source_version"] = document["source_version"]
        initial_head = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]

        reader = sqlite3.connect(database)
        try:
            reader.execute("BEGIN")
            assert reader.execute(
                "SELECT body FROM candidate_memories WHERE id = ?",
                (candidate.json()["id"],),
            ).fetchone() == (body,)
            failed = client.request(
                "DELETE",
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json=operation,
            )
            assert failed.status_code == 422, failed.text
            assert failed.json()["detail"] == "SQLite WAL checkpoint is busy"
        finally:
            reader.rollback()
            reader.close()

        assert document_path.read_text(encoding="utf-8") == body
        assert (
            client.get(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                params={"path": "forgotten.md"},
            ).status_code
            == 200
        )
        assert (
            client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()[0][
                "commit"
            ]
            == initial_head
        )
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT body FROM candidate_memories WHERE id = ?",
                (candidate.json()["id"],),
            ).fetchone() == (body,)
            assert connection.execute(
                "SELECT COUNT(*) FROM candidate_governance WHERE candidate_id = ?",
                (candidate.json()["id"],),
            ).fetchone() == (1,)
            assert connection.execute(
                "SELECT COUNT(*) FROM candidate_audit WHERE candidate_id = ? AND body = ?",
                (candidate.json()["id"], body),
            ).fetchone() == (1,)
            assert connection.execute(
                "SELECT candidate_id FROM capture_rounds "
                "WHERE session_id = 'pinned-capture' AND turn_id = 'turn-1'"
            ).fetchone() == (candidate.json()["id"],)
            assert connection.execute(
                "SELECT content FROM capture_inbox WHERE event_id = 'pinned-event'"
            ).fetchone() == (body,)
            assert connection.execute(
                "SELECT status, total_documents, projected_documents "
                "FROM memory_graph_indexes WHERE library_id = ?",
                (library_id,),
            ).fetchone() == ("ready", 1, 1)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_tombstones WHERE library_id = ?",
                (library_id,),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_operations WHERE operation_id = ?",
                (operation["operation_id"],),
            ).fetchone() == (0,)
        assert (projection / "graph.db").read_text(encoding="utf-8") == body
        assert not list(graph_root.glob(f".{library_id}.purge-*"))
        capture_events = client.get("/api/v1/capture/events", headers=headers)
        assert capture_events.status_code == 200, capture_events.text
        assert capture_events.json()[0]["content"] == body

        retried = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json=operation,
        )
        assert retried.status_code == 200, retried.text
        assert not document_path.exists()
        assert not projection.exists()
        assert not list(graph_root.glob(f".{library_id}.purge-*"))

        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM candidate_memories WHERE id = ?",
                (candidate.json()["id"],),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM capture_inbox WHERE event_id = 'pinned-event'"
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT status, total_documents, projected_documents "
                "FROM memory_graph_indexes WHERE library_id = ?",
                (library_id,),
            ).fetchone() == ("ready", 0, 0)
        assert client.get("/api/v1/capture/events", headers=headers).json() == []
        for suffix in ("", "-wal", "-shm"):
            persisted_path = Path(f"{database}{suffix}")
            if persisted_path.exists():
                assert sentinel.encode() not in persisted_path.read_bytes()

    with TestClient(create_app(settings)) as restarted:
        headers = _headers(state_dir)
        assert (
            restarted.get(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                params={"path": "forgotten.md"},
            ).status_code
            == 404
        )
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM candidate_memories WHERE id = ?",
                (candidate.json()["id"],),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM capture_inbox WHERE event_id = 'pinned-event'"
            ).fetchone() == (0,)
        for suffix in ("", "-wal", "-shm"):
            persisted_path = Path(f"{state_dir / 'platform.sqlite3'}{suffix}")
            if persisted_path.exists():
                assert sentinel.encode() not in persisted_path.read_bytes()


def test_restart_finishes_committed_graph_cleanup_after_process_exit(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    graph_root = state_dir / "graphs"
    root.mkdir()
    sentinel = "CrashWindowGraphNeedle-91f335e2"
    body = f"# Forget after crash\n\n{sentinel}\n"
    (root / "forgotten.md").write_text(body, encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        with sqlite3.connect(state_dir / "platform.sqlite3") as setup:
            setup.execute("DELETE FROM background_jobs WHERE kind = 'graph_rebuild'")
            setup.execute(
                "UPDATE memory_graph_indexes SET status = 'building' WHERE library_id = ?",
                (library_id,),
            )
            setup.commit()
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "forgotten.md"},
        ).json()
        preview_token = _deletion_preview_token(
            client, headers, library_id, "forgotten.md", document["source_version"]
        )

    projection = graph_root / library_id
    projection.mkdir(parents=True)
    (projection / "graph.db").write_text(body, encoding="utf-8")
    (projection / "projection.ready").write_text("ready\n", encoding="ascii")
    request_payload = {
        "path": "forgotten.md",
        "expected_source_version": document["source_version"],
        "operation_id": "crash-after-durable-delete",
        "actor_type": "user",
        "source": "test",
        "preview_token": preview_token,
    }
    crash_script = f"""
import os
from fastapi.testclient import TestClient
from personal_agent_memory.app import create_app
from pathlib import Path

from personal_agent_memory.config import Settings

settings = Settings(
    state_dir=Path({str(state_dir)!r}),
    library_roots=(Path({str(tmp_path)!r}),),
)
with TestClient(create_app(settings)) as client:
    state = client.app.state.platform_state
    state._checkpoint_forgotten_plaintext = lambda bodies: os._exit(86)
    response = client.request(
        "DELETE",
        "/api/v1/libraries/{library_id}/document",
        headers={headers!r},
        json={request_payload!r},
    )
    raise SystemExit(response.status_code)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script],
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert crashed.returncode == 86

    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        cleanup_id, stored_library_id = connection.execute(
            "SELECT cleanup_id, library_id FROM graph_cleanup_intents"
        ).fetchone()
        assert stored_library_id == library_id
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_tombstones WHERE id = ? AND library_id = ?",
            (cleanup_id, library_id),
        ).fetchone() == (1,)
    quarantines = list(graph_root.glob(f".{library_id}.purge-{cleanup_id}-*"))
    assert len(quarantines) == 1
    assert sentinel in (quarantines[0] / "graph.db").read_text(encoding="utf-8")
    assert not projection.exists()

    unrelated_cleanup_id = "f" * 32
    unrelated = graph_root / f".{library_id}.purge-{unrelated_cleanup_id}-unrelated"
    unrelated.mkdir()
    (unrelated / "graph.db").write_text("unrelated quarantine", encoding="utf-8")
    other_library = graph_root / "other-library"
    other_library.mkdir()
    (other_library / "graph.db").write_text("live other projection", encoding="utf-8")

    with TestClient(create_app(settings)) as restarted:
        assert not quarantines[0].exists()
        retry_quarantine = graph_root / (
            f".{library_id}.purge-{cleanup_id}-disposable-idempotent-retry"
        )
        retry_quarantine.mkdir()
        (retry_quarantine / "graph.db").write_text(body, encoding="utf-8")
        with sqlite3.connect(state_dir / "platform.sqlite3") as setup:
            setup.execute(
                "INSERT INTO graph_cleanup_intents (cleanup_id, library_id) VALUES (?, ?)",
                (cleanup_id, library_id),
            )
            setup.commit()
        retried = restarted.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json=request_payload,
        )
        assert retried.status_code == 200, retried.text
        assert retried.json()["operation_id"] == request_payload["operation_id"]
        assert not retry_quarantine.exists()

    assert unrelated.exists()
    assert (other_library / "graph.db").read_text(encoding="utf-8") == "live other projection"
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM graph_cleanup_intents").fetchone() == (0,)
    for suffix in ("", "-wal", "-shm"):
        persisted = Path(f"{state_dir / 'platform.sqlite3'}{suffix}")
        if persisted.exists():
            assert sentinel.encode() not in persisted.read_bytes()


def _crash_after_forget_graph_stage(
    tmp_path: Path,
    *,
    documents: dict[str, tuple[str, int]],
) -> dict[str, Any]:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    graph_root = state_dir / "graphs"
    root.mkdir()
    for path, (body, mode) in documents.items():
        document_path = root / path
        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(body, encoding="utf-8")
        document_path.chmod(mode)
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))

    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        with sqlite3.connect(state_dir / "platform.sqlite3") as setup:
            git_dir = Path(
                str(
                    setup.execute(
                        "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
                        (library_id,),
                    ).fetchone()[0]
                )
            )
            setup.execute("DELETE FROM background_jobs WHERE kind = 'graph_rebuild'")
            setup.execute(
                "UPDATE memory_graph_indexes SET status = 'building' WHERE library_id = ?",
                (library_id,),
            )
            setup.commit()
        initial_head = subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "forgotten.md"},
        ).json()
        preview_token = _deletion_preview_token(
            client, headers, library_id, "forgotten.md", document["source_version"]
        )

    projection = graph_root / library_id
    projection.mkdir(parents=True)
    (projection / "graph.db").write_text(
        "".join(body for body, _mode in documents.values()), encoding="utf-8"
    )
    (projection / "projection.ready").write_text("ready\n", encoding="ascii")
    request_payload = {
        "path": "forgotten.md",
        "expected_source_version": document["source_version"],
        "operation_id": "crash-after-graph-stage-before-database-commit",
        "actor_type": "user",
        "source": "test",
        "preview_token": preview_token,
    }
    crash_script = f"""
import os
from fastapi.testclient import TestClient
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from pathlib import Path

settings = Settings(
    state_dir=Path({str(state_dir)!r}),
    library_roots=(Path({str(tmp_path)!r}),),
)
with TestClient(create_app(settings)) as client:
    state = client.app.state.platform_state
    adapter = state.graph_adapter
    assert adapter is not None
    original = adapter.stage_purge
    def crash_after_stage(library_id, *, cleanup_id=None):
        original(library_id, cleanup_id=cleanup_id)
        os._exit(87)
    adapter.stage_purge = crash_after_stage
    response = client.request(
        "DELETE",
        "/api/v1/libraries/{library_id}/document",
        headers={headers!r},
        json={request_payload!r},
    )
    raise SystemExit(response.status_code)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script],
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert crashed.returncode == 87
    assert all(not (root / path).exists() for path in documents)
    interrupted_head = subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert interrupted_head != initial_head
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        intent = connection.execute(
            """SELECT cleanup_id, previous_head, document_paths_json, marker_path,
                      marker_digest, document_modes_json
               FROM graph_cleanup_intents WHERE library_id = ?""",
            (library_id,),
        ).fetchone()
        assert intent is not None
        cleanup_id = str(intent[0])
        assert intent[1] == initial_head
        assert set(json.loads(str(intent[2]))) == set(documents)
        assert intent[3] == f".personal-agent-memory-tombstones/{cleanup_id}.json"
        assert len(str(intent[4])) == 64
        assert json.loads(str(intent[5])) == {
            path: mode for path, (_body, mode) in documents.items()
        }
        assert all(body not in json.dumps(intent) for body, _mode in documents.values())
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_documents WHERE library_id = ? AND path = ?",
            (library_id, "forgotten.md"),
        ).fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM memory_tombstones").fetchone() == (0,)
    assert not projection.exists()
    assert list(graph_root.glob(f".{library_id}.purge-{cleanup_id}-official-*"))
    return {
        "cleanup_id": cleanup_id,
        "document": document,
        "git_dir": git_dir,
        "graph_root": graph_root,
        "headers": headers,
        "initial_head": initial_head,
        "interrupted_head": interrupted_head,
        "library_id": library_id,
        "projection": projection,
        "request_payload": request_payload,
        "root": root,
        "settings": settings,
        "state_dir": state_dir,
    }


def test_restart_rolls_back_forgetting_after_post_stage_process_exit(tmp_path: Path) -> None:
    sentinel = "PreCommitCrashNeedle-22a8c14f"
    body = f"# Interrupted forgetting\n\n{sentinel}\n"
    documents = {
        "forgotten.md": (body, 0o644),
        "nested/copy.md": (body, 0o664),
    }
    crashed = _crash_after_forget_graph_stage(tmp_path, documents=documents)
    state_dir = cast(Path, crashed["state_dir"])
    root = cast(Path, crashed["root"])
    graph_root = cast(Path, crashed["graph_root"])
    projection = cast(Path, crashed["projection"])
    settings = cast(Settings, crashed["settings"])
    headers = cast(dict[str, str], crashed["headers"])
    library_id = cast(str, crashed["library_id"])
    cleanup_id = cast(str, crashed["cleanup_id"])
    initial_head = cast(str, crashed["initial_head"])
    interrupted_head = cast(str, crashed["interrupted_head"])
    git_dir = cast(Path, crashed["git_dir"])

    for _restart in range(2):
        with TestClient(create_app(settings)) as recovered:
            headers = _headers(state_dir)
            restored = recovered.get(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                params={"path": "forgotten.md"},
            )
            assert restored.status_code == 200, restored.text
            assert restored.json()["content"] == body
            assert recovered.get(
                f"/api/v1/libraries/{library_id}/forgotten", headers=headers
            ).json() == []
            history = recovered.get(
                f"/api/v1/libraries/{library_id}/history", headers=headers
            ).json()
            assert history[0]["commit"] == initial_head
            assert all(item["commit"] != interrupted_head for item in history)
            unreachable = recovered.get(
                f"/api/v1/libraries/{library_id}/history/{interrupted_head}/diff",
                headers=headers,
            )
            assert unreachable.status_code == 422
            assert sentinel not in unreachable.text
        for path, (expected_body, expected_mode) in documents.items():
            restored_path = root / path
            assert restored_path.read_text(encoding="utf-8") == expected_body
            assert restored_path.stat().st_mode & 0o777 == expected_mode
        assert sentinel in (projection / "graph.db").read_text(encoding="utf-8")
        assert subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip() == initial_head
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM graph_cleanup_intents"
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT source_version FROM memory_documents WHERE library_id = ? AND path = ?",
                (library_id, "forgotten.md"),
            ).fetchone() == (hashlib.sha256(body.encode()).hexdigest(),)
        assert not list(graph_root.glob(f".{library_id}.purge-{cleanup_id}-*"))
        assert not (root / ".personal-agent-memory-tombstones" / f"{cleanup_id}.json").exists()


def test_restart_restores_complete_document_modes_after_interrupted_forgetting(
    tmp_path: Path,
) -> None:
    modes = {
        "forgotten.md": 0o000,
        "executable.md": 0o755,
        "setuid.md": 0o4755,
        "setgid.md": 0o2755,
        "sticky.md": 0o1755,
    }
    for name, mode in modes.items():
        probe = tmp_path / f".{name}.mode-probe"
        probe.touch()
        _require_retained_mode(probe, mode)
        probe.unlink()
    body = "# Restore modes\n\nPreserve every mode after restart.\n"
    documents = {
        path: (body, 0o644 if mode == 0 else mode)
        for path, mode in modes.items()
    }
    crashed = _crash_after_forget_graph_stage(tmp_path, documents=documents)
    state_dir = cast(Path, crashed["state_dir"])
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        stored_modes = json.loads(
            str(
                connection.execute(
                    "SELECT document_modes_json FROM graph_cleanup_intents"
                ).fetchone()[0]
            )
        )
        assert stored_modes == {
            path: configured_mode for path, (_body, configured_mode) in documents.items()
        }
        stored_modes["forgotten.md"] = 0o000
        connection.execute(
            "UPDATE graph_cleanup_intents SET document_modes_json = ?",
            (json.dumps(stored_modes, sort_keys=True),),
        )
        connection.commit()

    with TestClient(create_app(cast(Settings, crashed["settings"]))):
        pass

    root = cast(Path, crashed["root"])
    for path, expected_mode in modes.items():
        restored = root / path
        assert restored.stat().st_mode & 0o7777 == expected_mode
        restored.chmod(0o600)
        assert restored.read_text(encoding="utf-8") == documents[path][0]
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM graph_cleanup_intents").fetchone() == (0,)


def test_restart_recovery_is_idempotent_after_restoring_mode_zero_document(
    tmp_path: Path,
) -> None:
    body = "# Repeated recovery\n\nA second crash must not strand a mode-zero document.\n"
    crashed = _crash_after_forget_graph_stage(
        tmp_path,
        documents={
            "forgotten.md": (body, 0o600),
            "second.md": (body, 0o644),
        },
    )
    state_dir = cast(Path, crashed["state_dir"])
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        intent = connection.execute(
            "SELECT cleanup_id, document_modes_json FROM graph_cleanup_intents"
        ).fetchone()
        assert intent is not None
        cleanup_id = str(intent[0])
        modes = json.loads(
            str(
                intent[1]
            )
        )
        modes["forgotten.md"] = 0o000
        connection.execute(
            "UPDATE graph_cleanup_intents SET document_modes_json = ?",
            (json.dumps(modes, sort_keys=True),),
        )
        connection.commit()

    crash_script = f"""
import os
from fastapi.testclient import TestClient
from pathlib import Path
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.state import PlatformState

settings = Settings(
    state_dir=Path({str(state_dir)!r}),
    library_roots=(Path({str(tmp_path)!r}),),
)
original = PlatformState._restore_interrupted_document
calls = 0
def crash_after_first(root, path, content, mode):
    global calls
    original(root, path, content, mode)
    calls += 1
    if calls == 1:
        os._exit(89)
PlatformState._restore_interrupted_document = staticmethod(crash_after_first)
with TestClient(create_app(settings)):
    raise SystemExit("startup unexpectedly completed")
"""
    interrupted = subprocess.run(
        [sys.executable, "-c", crash_script],
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert interrupted.returncode == 89
    root = cast(Path, crashed["root"])
    first = root / "forgotten.md"
    assert first.stat().st_mode & 0o7777 == 0o000
    assert not (root / "second.md").exists()
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM graph_cleanup_intents").fetchone() == (1,)

    with TestClient(create_app(cast(Settings, crashed["settings"]))) as recovered:
        assert first.stat().st_mode & 0o7777 == 0o000
        first.chmod(0o600)
        headers = _headers(state_dir)
        response = recovered.get(
            f"/api/v1/libraries/{crashed['library_id']}/document",
            headers=headers,
            params={"path": "forgotten.md"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == body

    assert first.read_text(encoding="utf-8") == body
    second = root / "second.md"
    assert second.read_text(encoding="utf-8") == body
    assert second.stat().st_mode & 0o7777 == 0o644
    projection = cast(Path, crashed["projection"])
    assert body in (projection / "graph.db").read_text(encoding="utf-8")
    assert not list(cast(Path, crashed["graph_root"]).glob(f".*.purge-{cleanup_id}-*"))
    marker = root / ".personal-agent-memory-tombstones" / f"{cleanup_id}.json"
    assert not marker.exists()
    assert subprocess.run(
        ["git", f"--git-dir={crashed['git_dir']}", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip() == crashed["initial_head"]
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM graph_cleanup_intents").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_documents WHERE library_id = ?",
            (crashed["library_id"],),
        ).fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM memory_tombstones").fetchone() == (0,)


def test_interrupted_recovery_rejects_final_directory_entry_replacement(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    target = root / "target.md"
    expected = b"# Expected\n\nRecovery body.\n"
    target.write_bytes(expected)
    target.chmod(0)
    real_open = os.open
    swapped = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and path == "target.md" and flags & os.O_PATH:
            swapped = True
            assert dir_fd is not None
            os.rename(
                "target.md", "pinned.md", src_dir_fd=dir_fd, dst_dir_fd=dir_fd
            )
            replacement = real_open(
                "target.md",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0,
                dir_fd=dir_fd,
            )
            os.write(replacement, b"attacker\n")
            os.close(replacement)
        return descriptor

    monkeypatch.setattr(os, "open", racing_open)
    with raises(
        MemoryMutationError, match="target changed during recovery"
    ):
        PlatformState._restore_interrupted_document(root, "target.md", expected, 0)
    monkeypatch.setattr(os, "open", real_open)
    target.chmod(0o600)
    (root / "pinned.md").chmod(0o600)
    assert target.read_bytes() == b"attacker\n"
    assert (root / "pinned.md").read_bytes() == expected


def _replace_interrupted_head_with_counterfeit(
    crashed: dict[str, Any], *, retained_body: bytes | None, marker: bytes
) -> str:
    git_dir = cast(Path, crashed["git_dir"])
    initial_head = cast(str, crashed["initial_head"])
    interrupted_head = cast(str, crashed["interrupted_head"])
    cleanup_id = cast(str, crashed["cleanup_id"])
    root = cast(Path, crashed["root"])
    index = git_dir.parent / f"counterfeit-{uuid.uuid4().hex}.index"
    environment = {
        **os.environ,
        "GIT_INDEX_FILE": str(index),
        "GIT_WORK_TREE": str(root),
    }
    try:
        subprocess.run(
            ["git", f"--git-dir={git_dir}", "read-tree", initial_head],
            env=environment,
            check=True,
        )
        if retained_body is None:
            subprocess.run(
                ["git", f"--git-dir={git_dir}", "update-index", "--remove", "--", "forgotten.md"],
                env=environment,
                check=True,
            )
        else:
            document_blob = subprocess.run(
                ["git", f"--git-dir={git_dir}", "hash-object", "-w", "--stdin"],
                env=environment,
                input=retained_body,
                capture_output=True,
                check=True,
            ).stdout.decode().strip()
            subprocess.run(
                [
                    "git",
                    f"--git-dir={git_dir}",
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    "100644",
                    document_blob,
                    "forgotten.md",
                ],
                env=environment,
                check=True,
            )
        marker_blob = subprocess.run(
            ["git", f"--git-dir={git_dir}", "hash-object", "-w", "--stdin"],
            env=environment,
            input=marker,
            capture_output=True,
            check=True,
        ).stdout.decode().strip()
        marker_path = f".personal-agent-memory-tombstones/{cleanup_id}.json"
        subprocess.run(
            [
                "git",
                f"--git-dir={git_dir}",
                "update-index",
                "--add",
                "--cacheinfo",
                "100644",
                marker_blob,
                marker_path,
            ],
            env=environment,
            check=True,
        )
        tree = subprocess.run(
            ["git", f"--git-dir={git_dir}", "write-tree"],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        counterfeit = subprocess.run(
            [
                "git",
                f"--git-dir={git_dir}",
                "commit-tree",
                tree,
                "-p",
                initial_head,
                "-m",
                "Counterfeit forget commit",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        reference = subprocess.run(
            ["git", f"--git-dir={git_dir}", "symbolic-ref", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        subprocess.run(
            [
                "git",
                f"--git-dir={git_dir}",
                "update-ref",
                reference,
                counterfeit,
                interrupted_head,
            ],
            check=True,
        )
        return counterfeit
    finally:
        index.unlink(missing_ok=True)


def test_restart_rejects_same_parent_same_paths_counterfeit_forget_commit(
    tmp_path: Path,
) -> None:
    body = "# Counterfeit recovery\n\nOriginal body must survive.\n"
    for variant in ("marker", "body"):
        case = tmp_path / variant
        case.mkdir()
        crashed = _crash_after_forget_graph_stage(
            case, documents={"forgotten.md": (body, 0o644)}
        )
        root = cast(Path, crashed["root"])
        cleanup_id = cast(str, crashed["cleanup_id"])
        marker_path = root / ".personal-agent-memory-tombstones" / f"{cleanup_id}.json"
        genuine_marker = marker_path.read_bytes()
        counterfeit = _replace_interrupted_head_with_counterfeit(
            crashed,
            retained_body=b"# Counterfeit\n\nWrong body.\n" if variant == "body" else None,
            marker=b'{"counterfeit":true}\n' if variant == "marker" else genuine_marker,
        )
        startup = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from fastapi.testclient import TestClient; "
                    "from pathlib import Path; "
                    "from personal_agent_memory.app import create_app; "
                    "from personal_agent_memory.config import Settings; "
                    f"s=Settings(state_dir=Path({str(crashed['state_dir'])!r}), "
                    f"library_roots=(Path({str(case)!r}),)); "
                    "TestClient(create_app(s)).__enter__()"
                ),
            ],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert startup.returncode != 0
        git_dir = cast(Path, crashed["git_dir"])
        assert subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip() == counterfeit
        with sqlite3.connect(cast(Path, crashed["state_dir"]) / "platform.sqlite3") as db:
            assert db.execute(
                "SELECT COUNT(*) FROM graph_cleanup_intents WHERE cleanup_id = ?",
                (cleanup_id,),
            ).fetchone() == (1,)
        assert not (root / "forgotten.md").exists()
        assert marker_path.read_bytes() == genuine_marker


def test_restart_rejects_fifo_restore_target_without_blocking(tmp_path: Path) -> None:
    body = "# FIFO recovery\n\nStartup must fail closed.\n"
    crashed = _crash_after_forget_graph_stage(
        tmp_path, documents={"forgotten.md": (body, 0o664)}
    )
    root = cast(Path, crashed["root"])
    fifo = root / "forgotten.md"
    os.mkfifo(fifo, 0o600)
    started = time.monotonic()
    startup = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fastapi.testclient import TestClient; "
                "from pathlib import Path; "
                "from personal_agent_memory.app import create_app; "
                "from personal_agent_memory.config import Settings; "
                f"s=Settings(state_dir=Path({str(crashed['state_dir'])!r}), "
                f"library_roots=(Path({str(tmp_path)!r}),)); "
                "TestClient(create_app(s)).__enter__()"
            ),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert startup.returncode != 0
    assert time.monotonic() - started < 5
    cleanup_id = cast(str, crashed["cleanup_id"])
    with sqlite3.connect(cast(Path, crashed["state_dir"]) / "platform.sqlite3") as db:
        assert db.execute(
            "SELECT COUNT(*) FROM graph_cleanup_intents WHERE cleanup_id = ?",
            (cleanup_id,),
        ).fetchone() == (1,)
    assert fifo.is_fifo()


def test_restart_migrates_pre_mode_cleanup_journal(tmp_path: Path) -> None:
    body = "# Legacy recovery journal\n\nRestore without plaintext metadata.\n"
    crashed = _crash_after_forget_graph_stage(
        tmp_path, documents={"forgotten.md": (body, 0o664)}
    )
    state_dir = cast(Path, crashed["state_dir"])
    with sqlite3.connect(state_dir / "platform.sqlite3") as db:
        db.execute("UPDATE graph_cleanup_intents SET document_modes_json = NULL")
        db.commit()
    with TestClient(create_app(cast(Settings, crashed["settings"]))) as recovered:
        response = recovered.get(
            f"/api/v1/libraries/{crashed['library_id']}/document",
            headers=cast(dict[str, str], crashed["headers"]),
            params={"path": "forgotten.md"},
        )
        assert response.status_code == 200
        assert response.json()["content"] == body
    restored = cast(Path, crashed["root"]) / "forgotten.md"
    assert restored.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(state_dir / "platform.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM graph_cleanup_intents").fetchone() == (0,)


def test_reserved_formal_metadata_fails_closed_across_authoritative_writes(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    forgotten_body = "# Decision\n\nKeep the audit database on Firebird 5.\n"
    (root / "forgotten.md").write_text(forgotten_body, encoding="utf-8")
    (root / "live.md").write_text("# Live\n\nCurrent content.\n", encoding="utf-8")
    nested_history = (
        '---\n"memory_id": {"value": "legacy"}\ntitle: Legacy\n---\n\n'
        "# Legacy\n\nMalformed platform metadata.\n"
    )
    (root / "history.md").write_text(nested_history, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        history_document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "history.md"},
        ).json()
        history_edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "history.md",
                "content": "# History\n\nSafe current content.\n",
                "expected_source_version": history_document["source_version"],
                "operation_id": "prepare-malformed-history-restore",
                "actor_type": "user",
                "source": "test-setup",
            },
        )
        assert history_edit.status_code == 200, history_edit.text
        pending = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "# Pending\n\nSafe before legacy metadata is injected.\n",
                "source_references": ["codex-session:legacy#assistant-final"],
                "creator": "codex",
                "idempotency_key": "reserved-key-pending-candidate",
            },
        )
        assert pending.status_code == 201, pending.text
        forgotten_document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "forgotten.md"},
        ).json()
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "forgotten.md",
                "expected_source_version": forgotten_document["source_version"],
                "operation_id": "forget-before-reserved-key-regressions",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert deleted.status_code == 200, deleted.text

        replay_id = str(uuid.uuid4())
        quoted_replay = "\r\n".join(
            (
                "---",
                f'"memory_id": {json.dumps(replay_id)}',
                "'memory_type': decision",
                f"'candidate_id': {json.dumps(replay_id)}",
                '"created_by": "other"',
                "'approved_by': 'other'",
                '"approval_reason": "other reason"',
                "'source_references':",
                "  - 'codex-session:quoted#assistant-final'",
                "format_version: 2",
                "---",
                "",
                forgotten_body.rstrip(),
                "",
            )
        )
        replay = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": quoted_replay,
                "source_references": ["codex-session:quoted-envelope#assistant-final"],
                "creator": "codex",
                "idempotency_key": "quoted-reserved-key-replay",
            },
        )
        assert replay.status_code == 422
        assert replay.json()["detail"] == "candidate matches forgotten memory"

        duplicate_reserved = (
            f'---\n"memory_id": {json.dumps(replay_id)}\n'
            f"'memory_id': {json.dumps(str(uuid.uuid4()))}\n---\n\n# Duplicate\n"
        )
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute(
                "UPDATE candidate_memories SET body = ? WHERE id = ?",
                (duplicate_reserved, pending.json()["id"]),
            )
            connection.commit()
        blocked_approval = client.post(
            f"/api/v1/candidates/{pending.json()['id']}/approve",
            headers=headers,
            json={
                "operator": "epq",
                "reason": "Reject ambiguous metadata",
                "operation_id": "approve-duplicate-reserved-key",
            },
        )
        assert blocked_approval.status_code == 422
        assert blocked_approval.json()["detail"] == ("candidate has invalid formal memory metadata")

        live_document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "live.md"},
        ).json()
        wrong_type_edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "live.md",
                "content": "---\n'memory_id': 7\ntitle: Wrong type\n---\n\n# Invalid\n",
                "expected_source_version": live_document["source_version"],
                "operation_id": "edit-wrong-reserved-key-type",
                "actor_type": "user",
                "source": "web",
            },
        )
        assert wrong_type_edit.status_code == 422
        assert wrong_type_edit.json()["detail"] == ("content has invalid formal memory metadata")

        history_document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "history.md"},
        ).json()
        blocked_history_restore = client.post(
            f"/api/v1/libraries/{library_id}/history/restore",
            headers=headers,
            json={
                "path": "history.md",
                "commit": initial_commit,
                "expected_source_version": history_document["source_version"],
                "operation_id": "restore-nested-reserved-key",
                "actor_type": "user",
                "source": "web-history",
            },
        )
        assert blocked_history_restore.status_code == 422
        assert blocked_history_restore.json()["detail"] == (
            "content has invalid formal memory metadata"
        )

        ordinary_edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "live.md",
                "content": (
                    '---\n"title": "Ordinary note"\npriority: 2\n---\n\n'
                    "# Live\n\nOrdinary frontmatter remains valid.\n"
                ),
                "expected_source_version": live_document["source_version"],
                "operation_id": "edit-ordinary-frontmatter",
                "actor_type": "user",
                "source": "web",
            },
        )
        assert ordinary_edit.status_code == 200, ordinary_edit.text


def test_ordinary_frontmatter_replay_is_blocked_across_authoritative_writes(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    body = "# Decision\n\nKeep the audit database on Firebird 5.\n"
    wrapped = "---\ntitle: Different title\npriority: 9\n---\n\n" + body
    (root / "forgotten.md").write_text(body, encoding="utf-8")
    (root / "history.md").write_text(wrapped, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        history = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "history.md"},
        ).json()
        history_edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "history.md",
                "content": "# Current\n\nSafe current content.\n",
                "expected_source_version": history["source_version"],
                "operation_id": "prepare-ordinary-frontmatter-history",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert history_edit.status_code == 200, history_edit.text
        approval_candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "# Pending\n\nSafe before approval.\n",
                "source_references": ["codex-session:approval#assistant-final"],
                "creator": "codex",
                "idempotency_key": "ordinary-frontmatter-approval",
            },
        )
        edit_candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "# Pending\n\nSafe before edit.\n",
                "source_references": ["codex-session:edit#assistant-final"],
                "creator": "codex",
                "idempotency_key": "ordinary-frontmatter-edit",
            },
        )
        assert approval_candidate.status_code == 201, approval_candidate.text
        assert edit_candidate.status_code == 201, edit_candidate.text
        forgotten = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "forgotten.md"},
        ).json()
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "forgotten.md",
                "expected_source_version": forgotten["source_version"],
                "operation_id": "forget-before-ordinary-frontmatter-replay",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert deleted.status_code == 200, deleted.text

        created = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": wrapped,
                "source_references": ["codex-session:create#assistant-final"],
                "creator": "codex",
                "idempotency_key": "ordinary-frontmatter-create-replay",
            },
        )
        assert created.status_code == 422
        assert created.json()["detail"] == "candidate matches forgotten memory"

        edited = client.put(
            f"/api/v1/candidates/{edit_candidate.json()['id']}",
            headers=headers,
            json={"body": wrapped, "operator": "epq", "reason": "replay"},
        )
        assert edited.status_code == 422
        assert edited.json()["detail"] == "candidate matches forgotten memory"

        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute(
                "UPDATE candidate_memories SET body = ? WHERE id = ?",
                (wrapped, approval_candidate.json()["id"]),
            )
            connection.commit()
        approved = client.post(
            f"/api/v1/candidates/{approval_candidate.json()['id']}/approve",
            headers=headers,
            json={
                "operator": "epq",
                "reason": "replay",
                "operation_id": "ordinary-frontmatter-approval-replay",
            },
        )
        assert approved.status_code == 422
        assert approved.json()["detail"] == "candidate matches forgotten memory"

        restored = client.post(
            f"/api/v1/libraries/{library_id}/history/restore",
            headers=headers,
            json={
                "path": "history.md",
                "commit": initial_commit,
                "expected_source_version": history_edit.json()["source_version"],
                "operation_id": "ordinary-frontmatter-history-replay",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert restored.status_code == 422
        assert restored.json()["detail"] == "content matches forgotten memory"

        assert client.portal is not None
        try:
            client.portal.call(
                state._publish_document,
                library_id,
                "internal-replay.md",
                wrapped,
                "ordinary-frontmatter-internal-replay",
                "epq",
                "test",
                "Reject internal replay",
            )
        except MemoryMutationError as error:
            assert str(error) == "content matches forgotten memory"
        else:
            raise AssertionError("internal publish accepted forgotten content")

        project = tmp_path / "project"
        project.mkdir()
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library_id},
        )
        assert binding.status_code == 201, binding.text
        state.model_client.extract_candidate = lambda _: {
            "eligible": True,
            "suggested_type": "decision",
            "body": wrapped,
        }
        for event_id, kind in (("ordinary-user", "user"), ("ordinary-assistant", "assistant")):
            captured = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json={
                    "event_id": event_id,
                    "session_id": "ordinary-frontmatter-capture",
                    "turn_id": "turn-1",
                    "event_kind": kind,
                    "content": "Remember this decision.",
                    "occurred_at": "2026-09-05T06:00:00Z",
                    "cwd": str(project),
                },
            )
            assert captured.status_code == 202, captured.text
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rounds = client.get(
                "/api/v1/capture/rounds",
                headers=headers,
                params={"session_id": "ordinary-frontmatter-capture"},
            ).json()
            if rounds and rounds[0]["status"] == "done":
                break
            time.sleep(0.05)
        assert rounds[0]["status"] == "done"
        assert rounds[0]["candidate_id"] is None


def test_deleting_one_document_atomically_preserves_surviving_graph_projection(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    forgotten = "# Forgotten\n\nRemove this graph source.\n"
    surviving = "# Surviving\n\nKeep this graph source queryable.\n"
    (root / "forgotten.md").write_text(forgotten, encoding="utf-8")
    (root / "surviving.md").write_text(surviving, encoding="utf-8")
    adapter = _RecordingGraphAdapter()

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        state = cast(PlatformState, client.app.state.platform_state)
        state.graph_adapter = adapter
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        database = state_dir / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT id, path, source_version FROM memory_documents "
                "WHERE library_id = ? ORDER BY path",
                (library_id,),
            ).fetchall()
            projected = tuple(
                GraphSourceDocument(
                    str(document_id),
                    str(path),
                    str(source_version),
                    forgotten if path == "forgotten.md" else surviving,
                )
                for document_id, path, source_version in rows
            )
            connection.executemany(
                "INSERT INTO memory_graph_documents "
                "(library_id, document_id, path, source_version) VALUES (?, ?, ?, ?)",
                [
                    (library_id, item.document_id, item.path, item.source_version)
                    for item in projected
                ],
            )
            connection.execute(
                "UPDATE memory_graph_indexes SET status = 'ready', total_documents = 2, "
                "projected_documents = 2, last_error = '' WHERE library_id = ?",
                (library_id,),
            )
            connection.commit()
        adapter.documents[library_id] = projected
        forgotten_document = next(item for item in projected if item.path == "forgotten.md")

        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": forgotten_document.path,
                "expected_source_version": forgotten_document.source_version,
                "operation_id": "delete-one-graph-document",
            },
        )
        assert deleted.status_code == 200, deleted.text
        assert tuple(item.path for item in adapter.documents[library_id]) == ("surviving.md",)
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT path FROM memory_graph_documents WHERE library_id = ?",
                (library_id,),
            ).fetchall() == [("surviving.md",)]
            assert connection.execute(
                "SELECT status, total_documents, projected_documents FROM memory_graph_indexes "
                "WHERE library_id = ?",
                (library_id,),
            ).fetchone() == ("ready", 1, 1)
            assert connection.execute(
                "SELECT COUNT(*) FROM background_jobs WHERE kind = 'graph_rebuild' "
                "AND json_extract(payload, '$.library_id') = ? AND status != 'done'",
                (library_id,),
            ).fetchone() == (0,)


def test_in_flight_graph_rebuild_cannot_restore_forgotten_projection(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    forgotten = "# Decision\n\nUse Firebird 5 for the audit database.\n"
    (root / "target.md").write_text(forgotten, encoding="utf-8")
    adapter = _BlockingRebuildGraphAdapter()

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "target.md"},
        ).json()
        state = cast(PlatformState, client.app.state.platform_state)
        state.graph_adapter = adapter
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute(
                "UPDATE memory_graph_indexes SET status = 'ready', total_documents = 1, "
                "projected_documents = 1 WHERE library_id = ?",
                (library_id,),
            )
            connection.commit()

        errors: list[BaseException] = []

        def rebuild() -> None:
            worker = PlatformState(state_dir / "platform.sqlite3", graph_adapter=adapter)
            worker.connection = sqlite3.connect(state_dir / "platform.sqlite3")
            try:
                asyncio.run(worker._rebuild_graph(library_id))
            except BaseException as error:
                errors.append(error)
            finally:
                worker.connection.close()

        thread = threading.Thread(target=rebuild)
        thread.start()
        assert adapter.started.wait(5)
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "target.md",
                "expected_source_version": document["source_version"],
                "operation_id": "delete-during-graph-rebuild",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert deleted.status_code == 200, deleted.text
        assert not adapter.documents.get(library_id)

        adapter.release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert not errors
        assert not any(
            "Firebird 5" in document.content
            for documents in adapter.documents.values()
            for document in documents
        )


def test_in_flight_vector_rebuild_cannot_restore_forgotten_vectors(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    forgotten = "# Decision\n\nUse Firebird 5 for the audit database.\n"
    (root / "target.md").write_text(forgotten, encoding="utf-8")
    embedding = _BlockingEmbeddingClient()

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        document = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "target.md"},
        ).json()
        errors: list[BaseException] = []

        def rebuild() -> None:
            worker = PlatformState(
                state_dir / "platform.sqlite3",
                model_client=embedding,
            )
            worker.connection = sqlite3.connect(state_dir / "platform.sqlite3")
            try:
                asyncio.run(worker._rebuild_vectors(library_id))
            except BaseException as error:
                errors.append(error)
            finally:
                worker.connection.close()

        thread = threading.Thread(target=rebuild)
        thread.start()
        assert embedding.started.wait(5)
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "target.md",
                "expected_source_version": document["source_version"],
                "operation_id": "delete-during-vector-rebuild",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert deleted.status_code == 200, deleted.text

        embedding.release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert not errors
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_chunk_vectors WHERE library_id = ?",
                (library_id,),
            ).fetchone() == (0,)


def test_invalid_library_lock_ids_cannot_create_files_outside_lock_directory(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    absolute_escape = tmp_path / "absolute-escape"
    relative_escape = tmp_path / "relative-escape.lock"

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        for library_id in (str(absolute_escape), "../../relative-escape"):
            response = client.post(
                "/mcp/candidates",
                headers=headers,
                json={
                    "library_id": library_id,
                    "suggested_type": "decision",
                    "body": "# Candidate\n\nThis must not create a lock file.",
                    "source_references": ["test:invalid-library-lock"],
                    "creator": "test",
                    "idempotency_key": (
                        f"invalid-lock-{hashlib.sha256(library_id.encode()).hexdigest()}"
                    ),
                },
            )
            assert response.status_code == 422, response.text
            assert response.json()["detail"] == "invalid memory library identifier"

    assert not Path(f"{absolute_escape}.lock").exists()
    assert not relative_escape.exists()
    assert not (state_dir / "locks").exists()


def test_history_diff_redacts_non_ascii_forgotten_path_for_create_and_delete(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    content = "# Private\n\nThis non-ASCII path must never leak from history.\n"
    path = root / "记忆.md"
    path.write_text(content, encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        initial_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        deleted = client.request(
            "DELETE",
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "记忆.md",
                "expected_source_version": hashlib.sha256(content.encode()).hexdigest(),
                "operation_id": "delete-non-ascii-history",
            },
        )
        assert deleted.status_code == 200, deleted.text
        for commit in (initial_commit, deleted.json()["commit"]):
            history_diff = client.get(
                f"/api/v1/libraries/{library_id}/history/{commit}/diff",
                headers=headers,
            )
            assert history_diff.status_code == 200, history_diff.text
            assert history_diff.json()["diff"].startswith("[redacted:")
            assert content not in history_diff.text
