from __future__ import annotations

import asyncio
import errno
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory import markdown_index
from personal_agent_memory import state as state_module
from personal_agent_memory.app import capture_request_budget_seconds, create_app
from personal_agent_memory.config import ConfigurationError, Settings
from personal_agent_memory.git_history import GitHistoryError, GitRepository
from personal_agent_memory.markdown_index import (
    MarkdownChunk,
    StoredChunk,
    chunk_markdown,
    match_chunk_ids,
)
from personal_agent_memory.security import ManagementSessionStore
from personal_agent_memory.state import MemoryMutationError, PlatformState


def auth_headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, 0.25),
        ("1000100", 0.1),
        ("1000250", 0.25),
        ("999999999999999999999999", 0.25),
        ("0", 0.0),
        ("-20", 0.0),
        ("not-a-deadline", 0.0),
    ],
)
def test_capture_request_budget_is_fail_closed_and_never_extends_default(
    header: str | None, expected: float
) -> None:
    assert capture_request_budget_seconds(
        header, now_epoch_seconds=1000.0
    ) == pytest.approx(expected)


def test_capture_deadline_header_rejects_invalid_values_and_clamps_large_values(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project-memory"
    project_root = tmp_path / "project"
    library_path.mkdir(parents=True)
    project_root.mkdir()
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        bound = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": library["id"]},
        )
        assert bound.status_code == 201, bound.text
        event = {
            "event_id": "capture-budget-invalid",
            "session_id": "capture-budget-session",
            "turn_id": "turn-one",
            "event_kind": "user",
            "content": "Capture only with a bounded valid persistence deadline.",
            "occurred_at": "2026-09-07T00:00:00Z",
            "cwd": str(project_root),
        }
        for value in ("not-an-integer", "-1", "0"):
            response = client.post(
                "/api/v1/capture/events",
                headers={
                    **headers,
                    "X-Personal-Agent-Memory-Persistence-Deadline-Ms": value,
                },
                json=event,
            )
            assert response.status_code == 503, response.text

        accepted = client.post(
            "/api/v1/capture/events",
            headers={
                **headers,
                "X-Personal-Agent-Memory-Persistence-Deadline-Ms": "999999999999999999999999",
            },
            json={**event, "event_id": "capture-budget-clamped"},
        )
        assert accepted.status_code == 202, accepted.text
        stored = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": event["session_id"]},
        ).json()
        assert [item["event_id"] for item in stored] == ["capture-budget-clamped"]


def test_sensitive_capture_deadline_rolls_back_slow_quarantine_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    app = create_app(Settings(state_dir=state_dir))
    with TestClient(app) as client:
        headers = {
            **auth_headers(state_dir),
            "X-Personal-Agent-Memory-Persistence-Deadline-Ms": str(
                int(time.time() * 1000) + 100
            ),
        }
        state = app.state.platform_state
        actual_prune = state._prune_sensitive_quarantine

        def slow_prune(connection: sqlite3.Connection) -> None:
            time.sleep(0.12)
            actual_prune(connection)

        monkeypatch.setattr(state, "_prune_sensitive_quarantine", slow_prune)
        started = time.monotonic()
        response = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "sensitive-capture-deadline",
                "session_id": "sensitive-capture-session",
                "turn_id": "turn-one",
                "event_kind": "user",
                "content": "password=DeadlineSensitiveSecretValue123456",
                "occurred_at": "2026-09-07T00:00:00Z",
                "cwd": str(project_root),
            },
        )
        assert response.status_code == 503, response.text
        assert time.monotonic() - started < 0.25
        assert client.portal is not None
        assert client.portal.call(
            lambda: state.connection_or_raise.execute(
                "SELECT COUNT(*) FROM sensitive_quarantine WHERE source_kind = 'capture'"
            ).fetchone()
        ) == (0,)


def test_capture_consolidation_is_fail_fast_when_sqlite_writer_is_busy(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project-memory"
    library_path.mkdir(parents=True)
    app = create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    with TestClient(app) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        database = state_dir / "platform.sqlite3"
        with sqlite3.connect(database, timeout=0) as writer:
            writer.execute(
                "INSERT INTO capture_rounds (session_id, turn_id, library_id) VALUES (?, ?, ?)",
                ("busy-consolidation-session", "busy-turn", library["id"]),
            )
            writer.commit()
            writer.execute("BEGIN IMMEDIATE")
            started = time.monotonic()
            response = client.post(
                "/api/v1/capture/consolidate",
                headers={
                    **headers,
                    "X-Personal-Agent-Memory-Persistence-Deadline-Ms": str(
                        int(time.time() * 1000) + 250
                    ),
                },
                json={"session_id": "busy-consolidation-session"},
            )
            elapsed = time.monotonic() - started
            assert response.status_code == 503, response.text
            assert elapsed < 0.2
            assert writer.execute(
                "SELECT COUNT(*) FROM background_jobs WHERE kind = 'capture_consolidation'"
            ).fetchone() == (0,)


def test_capture_consolidation_does_not_report_timeout_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project-memory"
    library_path.mkdir(parents=True)
    app = create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    with TestClient(app) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        database = state_dir / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute(
                "INSERT INTO capture_rounds (session_id, turn_id, library_id) VALUES (?, ?, ?)",
                ("commit-boundary-session", "commit-boundary-turn", library["id"]),
            )

        actual_connect = state_module.sqlite3.connect
        commit_started = threading.Event()

        class SlowCommitConnection:
            def __init__(self, connection: sqlite3.Connection) -> None:
                self._connection = connection

            def __getattr__(self, name: str) -> object:
                return getattr(self._connection, name)

            def commit(self) -> None:
                commit_started.set()
                time.sleep(0.2)
                self._connection.commit()

        def slow_connect(*args: object, **kwargs: object) -> SlowCommitConnection:
            return SlowCommitConnection(actual_connect(*args, **kwargs))

        monkeypatch.setattr(state_module.sqlite3, "connect", slow_connect)
        deadline = int(time.time() * 1000) + 100
        with ThreadPoolExecutor(max_workers=1) as executor:
            started = time.monotonic()
            request = executor.submit(
                client.post,
                "/api/v1/capture/consolidate",
                headers={
                    **headers,
                    "X-Personal-Agent-Memory-Persistence-Deadline-Ms": str(deadline),
                },
                json={"session_id": "commit-boundary-session"},
            )
            assert commit_started.wait(timeout=1)
            health_started = time.monotonic()
            health = client.get("/health/live", headers=headers)
            health_elapsed = time.monotonic() - health_started
            response = request.result(timeout=1)
            elapsed = time.monotonic() - started

        assert response.status_code == 202, response.text
        assert response.json() == {"queued": 1}
        assert elapsed >= 0.2
        assert health.status_code == 200
        assert health_elapsed < 0.1
        with actual_connect(database) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM background_jobs WHERE kind = 'capture_consolidation'"
            ).fetchone() == (1,)


def import_external_changes(
    client: TestClient,
    headers: dict[str, str],
    library_id: str,
    root: Path,
) -> dict[str, int | str]:
    scanned = client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
    assert scanned.status_code == 200, scanned.text
    changes = client.get(
        f"/api/v1/libraries/{library_id}/out-of-band-changes", headers=headers
    ).json()
    for change in changes:
        payload: dict[str, str] = {
            "action": "import",
            "operation_id": f"test-import-{uuid.uuid4()}",
        }
        if change["status"] == "conflict" or change["external_withheld"]:
            path = change["external_path"] or change["base_path"]
            payload["final_content"] = (root / path).read_text(encoding="utf-8")
        resolved = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{change['id']}/resolve",
            headers=headers,
            json=payload,
        )
        assert resolved.status_code == 200, resolved.text
    return scanned.json()


def require_relative_worktrees(repository: Path) -> None:
    help_result = subprocess.run(
        ["git", "-C", str(repository), "worktree", "add", "-h"],
        check=False,
        capture_output=True,
        text=True,
    )
    if "--[no-]relative-paths" not in help_result.stdout + help_result.stderr:
        pytest.skip("installed Git does not support worktree add --relative-paths")


def test_library_lock_context_does_not_grant_reentry_to_child_task(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    state = PlatformState(tmp_path / "state" / "platform.db")

    async def exercise() -> None:
        child_entered = asyncio.Event()

        async def child() -> None:
            async with state.library_write_lock("library-one"):
                child_entered.set()

        async with state.library_write_lock("library-one"):
            with state._library_lock("library-one"):
                pass
            child_task = asyncio.create_task(child())
            await asyncio.sleep(0.05)
            assert not child_entered.is_set()
        await asyncio.wait_for(child_task, timeout=1)
        assert child_entered.is_set()

    asyncio.run(exercise())


def test_waiting_library_edit_does_not_block_another_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    first_library = library_root / "first"
    second_library = library_root / "second"
    first_library.mkdir(parents=True)
    second_library.mkdir()
    first_note = first_library / "note.md"
    preview_note = first_library / "preview.md"
    second_note = second_library / "note.md"
    first_note.write_text("# First\n\nBaseline A.\n", encoding="utf-8")
    preview_note.write_text("# Preview\n\nDeletion preview baseline.\n", encoding="utf-8")
    second_note.write_text("# Second\n\nBaseline B.\n", encoding="utf-8")

    app = create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    with TestClient(app) as client:
        headers = auth_headers(state_dir)
        first = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(first_library), "kind": "project"},
        ).json()
        second = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(second_library), "kind": "project"},
        ).json()
        first_loaded = client.get(
            f"/api/v1/libraries/{first['id']}/document",
            headers=headers,
            params={"path": first_note.name},
        ).json()
        second_loaded = client.get(
            f"/api/v1/libraries/{second['id']}/document",
            headers=headers,
            params={"path": second_note.name},
        ).json()
        preview_loaded = client.get(
            f"/api/v1/libraries/{first['id']}/document",
            headers=headers,
            params={"path": preview_note.name},
        ).json()
        first_payload = {
            "path": first_note.name,
            "content": "# First\n\nUpdated A.\n",
            "expected_source_version": first_loaded["source_version"],
            "operation_id": "async-library-lock-first",
            "actor_type": "user",
            "source": "test",
        }
        second_payload = {
            "path": second_note.name,
            "content": "# Second\n\nUpdated B.\n",
            "expected_source_version": second_loaded["source_version"],
            "operation_id": "async-library-lock-second",
            "actor_type": "user",
            "source": "test",
        }
        candidate_payload = {
            "library_id": first["id"],
            "suggested_type": "decision",
            "body": "# Decision\n\nSerialized after the document edit.\n",
            "source_references": ["test-session:lock-order#assistant-final"],
            "creator": "test",
            "idempotency_key": "async-library-lock-candidate",
        }
        approval_candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                **candidate_payload,
                "body": "# Decision\n\nApprove after the held library lock.\n",
                "idempotency_key": "async-library-lock-approval-candidate",
            },
        )
        assert approval_candidate.status_code == 201, approval_candidate.text
        approval_payload = {
            "operator": "test-operator",
            "reason": "Verify cross-library independence for governance writes.",
            "operation_id": "async-library-lock-approve",
        }

        lock_path = state_dir / "locks" / f"{uuid.UUID(str(first['id']))}.lock"
        original_flock = fcntl.flock
        first_lock_attempted = threading.Event()

        def observed_flock(lock: object, operation: int) -> None:
            if operation & fcntl.LOCK_NB:
                first_lock_attempted.set()
            original_flock(lock, operation)

        delete_preview_payload = {
            "path": preview_note.name,
            "expected_source_version": preview_loaded["source_version"],
            "operation_id": "async-library-lock-delete-preview",
        }

        with lock_path.open("rb") as held_lock, ThreadPoolExecutor(max_workers=6) as executor:
            original_flock(held_lock, fcntl.LOCK_EX)
            monkeypatch.setattr(state_module.fcntl, "flock", observed_flock)
            first_future = executor.submit(
                client.put,
                f"/api/v1/libraries/{first['id']}/document",
                headers=headers,
                json=first_payload,
            )
            try:
                assert first_lock_attempted.wait(5)
                with pytest.raises(FutureTimeoutError):
                    first_future.result(timeout=0.1)
                candidate_future = executor.submit(
                    client.post,
                    "/mcp/candidates",
                    headers=headers,
                    json=candidate_payload,
                )
                with pytest.raises(FutureTimeoutError):
                    candidate_future.result(timeout=0.1)
                approval_future = executor.submit(
                    client.post,
                    f"/api/v1/candidates/{approval_candidate.json()['id']}/approve",
                    headers=headers,
                    json=approval_payload,
                )
                with pytest.raises(FutureTimeoutError):
                    approval_future.result(timeout=0.1)
                delete_preview_future = executor.submit(
                    client.post,
                    f"/api/v1/libraries/{first['id']}/document/delete-preview",
                    headers=headers,
                    json=delete_preview_payload,
                )
                with pytest.raises(FutureTimeoutError):
                    delete_preview_future.result(timeout=0.1)
                retention_preview_future = executor.submit(
                    client.get,
                    f"/api/v1/libraries/{first['id']}/retention-cleanup/preview",
                    headers=headers,
                )
                with pytest.raises(FutureTimeoutError):
                    retention_preview_future.result(timeout=0.1)
                second_future = executor.submit(
                    client.put,
                    f"/api/v1/libraries/{second['id']}/document",
                    headers=headers,
                    json=second_payload,
                )
                second_response = second_future.result(timeout=5)
                assert second_response.status_code == 200, second_response.text
                assert not first_future.done()
                assert not candidate_future.done()
                assert not approval_future.done()
                assert not delete_preview_future.done()
                assert not retention_preview_future.done()
            finally:
                original_flock(held_lock, fcntl.LOCK_UN)
            first_response = first_future.result(timeout=5)
            candidate_response = candidate_future.result(timeout=5)
            approval_response = approval_future.result(timeout=5)
            delete_preview_response = delete_preview_future.result(timeout=5)
            retention_preview_response = retention_preview_future.result(timeout=5)

        assert first_response.status_code == 200, first_response.text
        assert candidate_response.status_code == 201, candidate_response.text
        assert approval_response.status_code == 200, approval_response.text
        assert delete_preview_response.status_code == 200, delete_preview_response.text
        assert retention_preview_response.status_code == 200, retention_preview_response.text
        assert first_note.read_text(encoding="utf-8") == first_payload["content"]
        assert second_note.read_text(encoding="utf-8") == second_payload["content"]


def test_capture_binding_revalidation_is_bounded_and_retries_a_single_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    project_root = tmp_path / "project"
    first_library = library_root / "first"
    second_library = library_root / "second"
    first_library.mkdir(parents=True)
    second_library.mkdir()
    project_root.mkdir()

    app = create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    with TestClient(app) as client:
        headers = auth_headers(state_dir)
        libraries = [
            client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(path), "kind": "project"},
            ).json()
            for path in (first_library, second_library)
        ]
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": libraries[0]["id"]},
        ).json()
        state = app.state.platform_state
        actual_resolve = state.resolve_project_binding

        def resolved(library_index: int) -> dict[str, str]:
            return {
                "status": "bound",
                "project_id": binding["project_id"],
                "binding_id": binding["id"],
                "library_id": libraries[library_index]["id"],
            }

        changing_once = iter((resolved(0), resolved(1), resolved(1), resolved(1)))

        def resolve_after_one_change(cwd: str) -> dict[str, str]:
            try:
                return next(changing_once)
            except StopIteration:
                return resolved(1)

        monkeypatch.setattr(state, "resolve_project_binding", resolve_after_one_change)
        accepted = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "capture-binding-one-change",
                "session_id": "capture-binding-session",
                "turn_id": "turn-one",
                "event_kind": "user",
                "content": "Persist after one binding change.",
                "occurred_at": "2026-09-07T00:00:00Z",
                "cwd": str(project_root),
            },
        )
        assert accepted.status_code == 202, accepted.text
        stored = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": "capture-binding-session"},
        ).json()
        assert len(stored) == 1
        assert stored[0]["library_id"] == libraries[1]["id"]

        calls = 0

        def continuously_rebound(cwd: str) -> dict[str, str]:
            nonlocal calls
            result = resolved(calls % 2)
            calls += 1
            return result

        monkeypatch.setattr(state, "resolve_project_binding", continuously_rebound)
        started = time.monotonic()
        rejected = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "capture-binding-continuous-change",
                "session_id": "capture-binding-session",
                "turn_id": "turn-two",
                "event_kind": "user",
                "content": "This event must spool instead of reaching the wrong library.",
                "occurred_at": "2026-09-07T00:00:01Z",
                "cwd": str(project_root),
            },
        )
        elapsed = time.monotonic() - started
        assert rejected.status_code == 503, rejected.text
        assert calls == 24
        assert elapsed < 1
        assert client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": "capture-binding-session"},
        ).json() == stored
        monkeypatch.setattr(state, "resolve_project_binding", actual_resolve)

        lock_path = state_dir / "locks" / f"{uuid.UUID(str(libraries[0]['id']))}.lock"
        with lock_path.open("rb") as held_lock, ThreadPoolExecutor(max_workers=1) as executor:
            fcntl.flock(held_lock, fcntl.LOCK_EX)
            blocked = executor.submit(
                client.post,
                "/api/v1/capture/events",
                headers=headers,
                json={
                    "event_id": "capture-binding-held-lock",
                    "session_id": "capture-binding-session",
                    "turn_id": "turn-three",
                    "event_kind": "user",
                    "content": "Timeout safely while the target library is unavailable.",
                    "occurred_at": "2026-09-07T00:00:02Z",
                    "cwd": str(project_root),
                },
            )
            blocked_response = blocked.result(timeout=1)
            assert blocked_response.status_code == 503, blocked_response.text
            fcntl.flock(held_lock, fcntl.LOCK_UN)
        time.sleep(0.1)
        assert client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": "capture-binding-session"},
        ).json() == stored


def test_capture_sqlite_write_lock_returns_503_without_late_persistence(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project-memory"
    project_root = tmp_path / "project"
    library_path.mkdir(parents=True)
    project_root.mkdir()

    app = create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    with TestClient(app) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        bound = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": library["id"]},
        )
        assert bound.status_code == 201, bound.text
        event = {
            "event_id": "capture-sqlite-busy",
            "session_id": "capture-sqlite-busy-session",
            "turn_id": "turn-one",
            "event_kind": "user",
            "content": "Spool this event while SQLite is busy.",
            "occurred_at": "2026-09-07T00:00:00Z",
            "cwd": str(project_root),
        }
        database_path = state_dir / "platform.sqlite3"

        with sqlite3.connect(database_path, check_same_thread=False) as blocker:
            blocker.execute("BEGIN IMMEDIATE")

            def delayed_release() -> None:
                time.sleep(0.5)
                blocker.rollback()

            release = threading.Thread(target=delayed_release)
            release.start()
            started = time.monotonic()
            response = client.post(
                "/api/v1/capture/events", headers=headers, json=event
            )
            elapsed = time.monotonic() - started
            assert response.status_code == 503, response.text
            assert elapsed < 0.25
            release.join(timeout=1)
            assert not release.is_alive()

        time.sleep(0.1)
        assert client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": event["session_id"]},
        ).json() == []
        def capture_connection_state() -> tuple[bool, tuple[int] | None]:
            state_connection = app.state.platform_state.connection_or_raise
            return (
                state_connection.in_transaction,
                state_connection.execute("PRAGMA busy_timeout").fetchone(),
            )

        assert client.portal is not None
        assert client.portal.call(capture_connection_state) == (False, (5000,))

        with sqlite3.connect(database_path) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            started = time.monotonic()
            response = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json={**event, "event_id": "capture-sqlite-still-busy"},
            )
            assert response.status_code == 503, response.text
            assert time.monotonic() - started < 0.25
            health_started = time.monotonic()
            health = client.get("/health/live", headers=headers)
            assert health.status_code == 200, health.text
            assert time.monotonic() - health_started < 0.25
            blocker.rollback()

        accepted = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={**event, "event_id": "capture-sqlite-normal"},
        )
        duplicate = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={**event, "event_id": "capture-sqlite-normal"},
        )
        assert accepted.status_code == duplicate.status_code == 202
        assert accepted.json()["duplicate"] is False
        assert duplicate.json()["duplicate"] is True


def test_capture_large_tombstone_set_stays_within_deadline_and_is_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    first_library_path = library_root / "first"
    second_library_path = library_root / "second"
    project_root = tmp_path / "project"
    first_library_path.mkdir(parents=True)
    second_library_path.mkdir()
    project_root.mkdir()

    app = create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    with TestClient(app) as client:
        headers = auth_headers(state_dir)
        libraries = [
            client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(path), "kind": "project"},
            ).json()
            for path in (first_library_path, second_library_path)
        ]
        bound = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": libraries[0]["id"]},
        )
        assert bound.status_code == 201, bound.text

        def seed_tombstones() -> str:
            state = app.state.platform_state
            connection = state.connection_or_raise
            key_id = state.active_tombstone_key_id
            connection.executemany(
                """INSERT INTO memory_tombstones
                   (id, library_id, path, fingerprint, match_fingerprint, key_id,
                    source_scope_json, deleted_at, marker_path)
                   VALUES (?, ?, ?, ?, ?, ?, '{}', CURRENT_TIMESTAMP, ?)""",
                (
                    (
                        f"large-tombstone-{index}",
                        libraries[0]["id"],
                        f"forgotten-{index}.md",
                        f"hmac-sha256:{index:064x}",
                        f"hmac-sha256:{index + 1:064x}",
                        key_id,
                        f".memory-tombstones/large-{index}.json",
                    )
                    for index in range(4000)
                ),
            )
            connection.commit()
            query_plan = connection.execute(
                """EXPLAIN QUERY PLAN SELECT 1 FROM memory_tombstones
                   WHERE library_id = ? AND key_id = ? AND match_fingerprint = ? LIMIT 1""",
                (libraries[0]["id"], key_id, "not-present"),
            ).fetchall()
            return " ".join(str(row) for row in query_plan)

        assert client.portal is not None
        assert "memory_tombstone_match" in client.portal.call(seed_tombstones)
        large_content = "# Large capture\n\n" + "DeadlineSafeCaptureBody " * 2600
        event = {
            "event_id": "capture-large-tombstone-set",
            "session_id": "capture-large-tombstone-session",
            "turn_id": "turn-one",
            "event_kind": "user",
            "content": large_content,
            "occurred_at": "2026-09-07T00:00:00Z",
            "cwd": str(project_root),
        }
        started = time.monotonic()
        response = client.post("/api/v1/capture/events", headers=headers, json=event)
        elapsed = time.monotonic() - started
        assert response.status_code in {202, 503}, response.text
        assert elapsed < 0.25
        time.sleep(0.35)
        stored = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": event["session_id"]},
        ).json()
        if response.status_code == 503:
            assert stored == []
        else:
            assert [item["event_id"] for item in stored] == [event["event_id"]]

        state = app.state.platform_state
        actual_normalize = state._normalized_tombstone_match_content

        def slow_normalize(content: str, *, reject_invalid_formal: bool) -> str:
            time.sleep(0.21)
            return actual_normalize(
                content, reject_invalid_formal=reject_invalid_formal
            )

        monkeypatch.setattr(state, "_normalized_tombstone_match_content", slow_normalize)
        delayed_event = {
            **event,
            "event_id": "capture-expired-before-write",
            "turn_id": "turn-two",
        }
        started = time.monotonic()
        delayed = client.post(
            "/api/v1/capture/events", headers=headers, json=delayed_event
        )
        assert delayed.status_code == 503, delayed.text
        assert time.monotonic() - started < 0.25
        time.sleep(0.25)
        after_deadline = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"session_id": event["session_id"]},
        ).json()
        assert all(item["event_id"] != delayed_event["event_id"] for item in after_deadline)

        health_started = time.monotonic()
        health = client.get("/health/live", headers=headers)
        assert health.status_code == 200, health.text
        assert time.monotonic() - health_started < 0.25


def test_first_start_creates_protected_key_and_authenticated_surfaces(tmp_path: Path) -> None:
    app = create_app(Settings(state_dir=tmp_path))

    with TestClient(app) as client:
        key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        assert key
        assert stat.S_IMODE((tmp_path / "api-key").stat().st_mode) == 0o600
        assert client.get("/health/live").status_code == 401
        assert client.get("/api/v1/status").status_code == 401
        assert (
            client.get("/mcp/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        )

        headers = {"Authorization": f"Bearer {key}"}
        assert client.get("/health/live", headers=headers).json() == {"status": "ok"}
        status = client.get("/api/v1/status", headers=headers)
        assert status.status_code == 200
        body = status.json()
        assert body["status"] == "ready"
        assert body["database"] == "ok"
        assert body["background_worker"] == "running"
        assert body["api_key_fingerprint"] != key
        assert key not in json.dumps(body)
        assert client.get("/mcp/health", headers=headers).json()["status"] == "ready"


def test_empty_existing_key_is_replaced_before_accepting_requests(tmp_path: Path) -> None:
    key_path = tmp_path / "api-key"
    key_path.write_text("\n", encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=tmp_path))) as client:
        generated_key = key_path.read_text(encoding="utf-8").strip()
        assert generated_key
        assert client.get("/api/v1/status", headers={"Authorization": "Bearer "}).status_code == 401
        assert (
            client.get(
                "/api/v1/status", headers={"Authorization": f"Bearer {generated_key}"}
            ).status_code
            == 200
        )


@pytest.mark.parametrize("mutation", ["empty", "delete"])
def test_runtime_key_loss_fails_closed(tmp_path: Path, mutation: str) -> None:
    with TestClient(create_app(Settings(state_dir=tmp_path))) as client:
        key_path = tmp_path / "api-key"
        key = key_path.read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}
        assert client.get("/api/v1/status", headers=headers).status_code == 200

        if mutation == "empty":
            key_path.write_text("\n", encoding="utf-8")
        else:
            key_path.unlink()

        for path in ("/health/live", "/api/v1/status", "/mcp/health"):
            response = client.get(path, headers=headers)
            assert response.status_code == 401
            assert response.json()["detail"] == "invalid API key"


def test_key_rotation_immediately_invalidates_old_key(tmp_path: Path) -> None:
    app = create_app(Settings(state_dir=tmp_path))

    with TestClient(app) as client:
        old_key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        old_headers = {"Authorization": f"Bearer {old_key}"}
        response = client.post("/api/v1/auth/rotate", headers=old_headers)
        assert response.status_code == 200
        assert "api_key" not in response.json()
        new_key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        assert new_key != old_key
        assert client.get("/api/v1/status", headers=old_headers).status_code == 401
        new_status = client.get("/api/v1/status", headers={"Authorization": f"Bearer {new_key}"})
        assert new_status.status_code == 200
        assert new_status.json()["api_key_fingerprint"] == response.json()["fingerprint"]


def test_management_session_is_cookie_only_persistent_and_expires_at_24_hours(
    tmp_path: Path,
) -> None:
    start = "2026-09-25T12:00:00Z"
    login_settings = Settings(state_dir=tmp_path, session_now=start)
    with TestClient(create_app(login_settings)) as client:
        key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        login = client.post("/api/v1/auth/session", json={"api_key": key})
        assert login.status_code == 200, login.text
        assert login.json() == {
            "expires_at": "2026-09-26T12:00:00+00:00",
            "expires_in_seconds": 86_400,
        }
        cookie = client.cookies.get("pam_management_session")
        assert cookie
        set_cookie = login.headers["set-cookie"]
        assert "HttpOnly" in set_cookie
        assert "SameSite=strict" in set_cookie
        assert "Max-Age=86400" in set_cookie
        assert "Path=/" in set_cookie
        assert client.get("/api/v1/status").status_code == 200
        session_file = tmp_path / "management-sessions.json"
        stored = session_file.read_text(encoding="utf-8")
        assert cookie not in stored
        assert key not in stored

    restarted = TestClient(
        create_app(Settings(state_dir=tmp_path, session_now="2026-09-26T11:59:59Z"))
    )
    with restarted:
        restarted.cookies.set("pam_management_session", cookie)
        assert restarted.get("/api/v1/status").status_code == 200

    expired = TestClient(
        create_app(Settings(state_dir=tmp_path, session_now="2026-09-26T12:00:00Z"))
    )
    with expired:
        expired.cookies.set("pam_management_session", cookie)
        response = expired.get("/api/v1/status")
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid API key"
        assert json.loads((tmp_path / "management-sessions.json").read_text()) == []


def test_management_session_logout_and_key_rotation_revoke_cookie_access(
    tmp_path: Path,
) -> None:
    with TestClient(
        create_app(Settings(state_dir=tmp_path, session_now="2026-09-25T12:00:00Z"))
    ) as client:
        old_key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        login = client.post("/api/v1/auth/login", json={"api_key": old_key})
        assert login.status_code == 200, login.text
        cookie = client.cookies.get("pam_management_session")
        assert cookie
        assert client.post("/api/v1/auth/logout").json() == {"status": "logged_out"}
        assert client.get("/api/v1/status").status_code == 401

        login = client.post("/api/v1/auth/session", json={"api_key": old_key})
        assert login.status_code == 200, login.text
        rotated_cookie = client.cookies.get("pam_management_session")
        assert rotated_cookie and rotated_cookie != cookie
        rotated = client.post(
            "/api/v1/auth/rotate",
            headers={"Authorization": f"Bearer {old_key}"},
        )
        assert rotated.status_code == 200, rotated.text
        assert client.get("/api/v1/status").status_code == 401


def test_management_session_records_are_bounded_and_bearer_remains_supported(
    tmp_path: Path,
) -> None:
    with TestClient(
        create_app(Settings(state_dir=tmp_path, session_now="2026-09-25T12:00:00Z"))
    ) as client:
        key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        for _ in range(80):
            response = client.post("/api/v1/auth/session", json={"api_key": key})
            assert response.status_code == 200
        records = json.loads((tmp_path / "management-sessions.json").read_text())
        assert len(records) <= 64
        assert client.get(
            "/api/v1/status", headers={"Authorization": f"Bearer {key}"}
        ).status_code == 200


def test_management_session_concurrent_creation_is_serialized_and_bounded(
    tmp_path: Path,
) -> None:
    store = ManagementSessionStore(
        tmp_path / "management-sessions.json",
        now=lambda: datetime.fromisoformat("2026-09-25T12:00:00+00:00"),
    )

    with ThreadPoolExecutor(max_workers=32) as executor:
        tokens = list(executor.map(lambda _: store.create("concurrent-key")[0], range(200)))

    assert len(tokens) == 200
    records = json.loads((tmp_path / "management-sessions.json").read_text(encoding="utf-8"))
    assert len(records) <= store.max_records
    assert not list(tmp_path.glob("*.tmp"))
    valid_tokens = sum(store.verify(token, "concurrent-key") for token in tokens)
    assert valid_tokens == len(records)


def test_restart_reports_previous_clean_shutdown(tmp_path: Path) -> None:
    headers: dict[str, str]
    with TestClient(create_app(Settings(state_dir=tmp_path))) as client:
        key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}
        assert client.get("/api/v1/status", headers=headers).json()["startup_count"] == 1

    with TestClient(create_app(Settings(state_dir=tmp_path))) as client:
        body = client.get("/api/v1/status", headers=headers).json()
        assert body["startup_count"] == 2
        assert body["previous_shutdown_clean"] is True


def test_background_worker_executes_persisted_job(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = PlatformState(tmp_path / "platform.sqlite3")
        await state.start()
        job_id = state.enqueue_job("foundation-check", "{}")
        for _ in range(20):
            if state.job_status(job_id) == "done":
                break
            await asyncio.sleep(0.02)
        assert state.job_status(job_id) == "done"
        await state.close()

        restarted = PlatformState(tmp_path / "platform.sqlite3")
        await restarted.start()
        assert restarted.job_status(job_id) == "done"
        await restarted.close()

    asyncio.run(scenario())


def test_non_loopback_binding_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="loopback"):
        Settings(state_dir=tmp_path, host="0.0.0.0")


def test_corrupt_state_database_fails_startup(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "platform.sqlite3").write_bytes(b"not sqlite")

    with (
        pytest.raises(sqlite3.DatabaseError),
        TestClient(create_app(Settings(state_dir=tmp_path))),
    ):
        pass


def test_management_interface_is_available(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(state_dir=tmp_path))) as client:
        key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        response = client.get("/")
        assert response.status_code == 200
        assert "Personal Agent Memory" in response.text
        assert key not in response.text
        fingerprint = f"sha256:{hashlib.sha256(key.encode()).hexdigest()[:12]}"
        assert fingerprint not in response.text
        assert str(tmp_path) not in response.text
        assert "localStorage" not in response.text
        assert "sessionStorage" not in response.text
        assert "URLSearchParams" not in response.text
        assert "Authorization" not in response.text
        assert "Bearer" not in response.text
        assert "/api/v1/auth/session" in response.text
        assert "/api/v1/auth/logout" in response.text
        assert "credentials: 'same-origin'" in response.text
        assert "会话有效期 24 小时" in response.text
        assert "会话已过期，请重新登录" in response.text
        assert "退出" in response.text
        assert 'lang="zh-CN"' in response.text
        assert "个人智能体记忆" in response.text
        assert "服务状态" in response.text
        assert "记忆库" in response.text
        assert "注册记忆库" in response.text
        assert "项目绑定" in response.text
        assert "绑定项目" in response.text
        assert "关联 Worktree" in response.text
        assert "记忆检索" in response.text
        assert "/api/v1/search" in response.text
        assert "historyDocument.id = 'history-document'" in response.text
        assert "historyMessage.id = 'history-message'" in response.text
        assert "retentionPreviewContext" in response.text
        assert "bindingIntents" in response.text
        assert "networkError: true" in response.text
        for label in ("决策", "约束", "偏好", "领域事实", "流程", "经验"):
            assert label in response.text

        assert client.get("/api/v1/status").status_code == 401
        authenticated = client.get("/api/v1/status", headers={"Authorization": f"Bearer {key}"})
        assert authenticated.status_code == 200
        assert authenticated.json()["status"] == "ready"


def test_registers_new_and_existing_libraries_without_rewriting_markdown(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    existing = allowed_root / "existing"
    existing.mkdir(parents=True)
    note = existing / "notes.md"
    original = "# Existing\n\nKeep formatting.\n"
    note.write_text(original, encoding="utf-8")

    settings = Settings(state_dir=state_dir, library_roots=(allowed_root,))
    with TestClient(create_app(settings)) as client:
        headers = auth_headers(state_dir)
        created = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(allowed_root / "created"), "kind": "user"},
        )
        adopted = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(existing), "kind": "project"},
        )

        assert created.status_code == 201
        assert Path(created.json()["canonical_path"]).is_dir()
        assert created.json()["kind"] == "user"
        assert created.json()["availability"] == "available"
        assert created.json()["sync_status"] == "ready"
        assert adopted.status_code == 201
        assert adopted.json()["kind"] == "project"
        assert note.read_text(encoding="utf-8") == original
        assert {item["id"] for item in client.get("/api/v1/libraries", headers=headers).json()} == {
            created.json()["id"],
            adopted.json()["id"],
        }


def test_canonical_path_cannot_be_registered_twice(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    target = allowed_root / "target"
    target.mkdir(parents=True)
    alias = allowed_root / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(allowed_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        first = client.post(
            "/api/v1/libraries", headers=headers, json={"path": str(target), "kind": "project"}
        )
        duplicate = client.post(
            "/api/v1/libraries", headers=headers, json={"path": str(alias), "kind": "user"}
        )
        monkeypatch.chdir(allowed_root)
        relative_duplicate = client.post(
            "/api/v1/libraries", headers=headers, json={"path": "target", "kind": "project"}
        )

        assert first.status_code == 201
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"]["library_id"] == first.json()["id"]
        assert "already registered" in duplicate.json()["detail"]["reason"]
        assert relative_duplicate.status_code == 409
        assert relative_duplicate.json()["detail"]["library_id"] == first.json()["id"]


def test_initial_symlink_registration_stores_real_target_and_stays_available(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    target = allowed_root / "target"
    target.mkdir(parents=True)
    alias = allowed_root / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(allowed_root,)))
    ) as client:
        response = client.post(
            "/api/v1/libraries",
            headers=auth_headers(state_dir),
            json={"path": str(alias), "kind": "project"},
        )
        assert response.status_code == 201
        assert response.json()["canonical_path"] == str(target)
        assert response.json()["availability"] == "available"
        listed = client.get("/api/v1/libraries", headers=auth_headers(state_dir)).json()[0]
        assert listed["canonical_path"] == str(target)
        assert listed["availability"] == "available"


def test_registration_rejects_out_of_boundary_and_unwritable_paths(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    allowed_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = allowed_root / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    unwritable = allowed_root / "unwritable"
    unwritable.mkdir()
    unwritable.chmod(0o500)

    try:
        with TestClient(
            create_app(Settings(state_dir=state_dir, library_roots=(allowed_root,)))
        ) as client:
            headers = auth_headers(state_dir)
            escaped = client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(escape), "kind": "project"},
            )
            denied = client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(unwritable), "kind": "project"},
            )
            assert escaped.status_code == 422
            assert "allowed roots" in escaped.json()["detail"]
            assert denied.status_code == 422
            assert "writable" in denied.json()["detail"]
    finally:
        unwritable.chmod(0o700)


def test_library_health_recovers_without_losing_identity(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    library_path = allowed_root / "project"
    settings = Settings(state_dir=state_dir, library_roots=(allowed_root,))

    with TestClient(create_app(settings)) as client:
        headers = auth_headers(state_dir)
        registered = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        shutil.rmtree(library_path)
        missing = client.get("/api/v1/libraries", headers=headers).json()[0]
        assert missing["id"] == registered["id"]
        assert missing["availability"] == "missing"
        assert not library_path.exists()

        library_path.mkdir()
        library_path.chmod(0o000)
        unavailable = client.get("/api/v1/libraries", headers=headers).json()[0]
        assert unavailable["id"] == registered["id"]
        assert unavailable["availability"] == "unavailable"
        library_path.chmod(0o700)
        recovered = client.get("/api/v1/libraries", headers=headers).json()[0]
        assert recovered["id"] == registered["id"]
        assert recovered["availability"] == "available"

    with TestClient(create_app(settings)) as restarted:
        restored = restarted.get("/mcp/libraries", headers=headers).json()[0]
        sync = restarted.get(
            f"/mcp/libraries/{registered['id']}/sync-status", headers=headers
        ).json()
        assert restored["id"] == registered["id"]
        assert restored["kind"] == "project"
        assert sync == {
            "library_id": registered["id"],
            "availability": "available",
            "sync_status": "ready",
        }
        assert "api_key" not in json.dumps(restored)


def test_library_health_rejects_replacement_symlink_outside_allowed_roots(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    library_path = allowed_root / "project"
    outside = tmp_path / "outside"
    outside.mkdir()

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(allowed_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        registered = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        library_path.rmdir()
        library_path.symlink_to(outside, target_is_directory=True)

        listed = client.get("/api/v1/libraries", headers=headers).json()[0]
        sync = client.get(f"/mcp/libraries/{registered['id']}/sync-status", headers=headers).json()
        assert listed["id"] == registered["id"]
        assert listed["canonical_path"] == str(library_path)
        assert listed["availability"] == "unavailable"
        assert sync["availability"] == "unavailable"


def test_library_health_rejects_replacement_symlink_to_registered_library(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    first_path = allowed_root / "first"
    second_path = allowed_root / "second"

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(allowed_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        first = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(first_path), "kind": "project"},
        ).json()
        second = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(second_path), "kind": "project"},
        ).json()
        first_path.rmdir()
        first_path.symlink_to(second_path, target_is_directory=True)

        listed = {
            item["id"]: item for item in client.get("/api/v1/libraries", headers=headers).json()
        }
        sync = client.get(f"/mcp/libraries/{first['id']}/sync-status", headers=headers).json()
        assert listed[first["id"]]["canonical_path"] == str(first_path)
        assert listed[first["id"]]["availability"] == "unavailable"
        assert sync["availability"] == "unavailable"
        assert listed[second["id"]]["canonical_path"] == str(second_path)
        assert listed[second["id"]]["availability"] == "available"


def test_library_health_rejects_replaced_ancestor_alias_to_registered_library(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    source_parent = allowed_root / "source-parent"
    source_path = source_parent / "project"
    target_parent = allowed_root / "target-parent"
    target_path = target_parent / "project"

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(allowed_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        source = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(source_path), "kind": "project"},
        ).json()
        target = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(target_path), "kind": "project"},
        ).json()
        source_path.rmdir()
        source_parent.rmdir()
        source_parent.symlink_to(target_parent, target_is_directory=True)

        listed = {
            item["id"]: item for item in client.get("/api/v1/libraries", headers=headers).json()
        }
        sync = client.get(f"/mcp/libraries/{source['id']}/sync-status", headers=headers).json()
        assert listed[source["id"]]["canonical_path"] == str(source_path)
        assert listed[source["id"]]["availability"] == "unavailable"
        assert sync["availability"] == "unavailable"
        assert listed[target["id"]]["canonical_path"] == str(target_path)
        assert listed[target["id"]]["availability"] == "available"


def test_parent_permission_error_is_unavailable_and_recovers(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    protected_parent = allowed_root / "protected"
    library_path = protected_parent / "project"
    library_path.mkdir(parents=True)
    settings = Settings(state_dir=state_dir, library_roots=(allowed_root,))

    with TestClient(create_app(settings)) as client:
        headers = auth_headers(state_dir)
        registered = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        protected_parent.chmod(0o000)
        try:
            unavailable = client.get("/api/v1/libraries", headers=headers).json()[0]
            assert unavailable["id"] == registered["id"]
            assert unavailable["availability"] == "unavailable"
        finally:
            protected_parent.chmod(0o700)

        recovered = client.get("/api/v1/libraries", headers=headers).json()[0]
        assert recovered["id"] == registered["id"]
        assert recovered["availability"] == "available"


def test_unresolvable_symlink_is_rejected_with_actionable_error(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    allowed_root = tmp_path / "libraries"
    allowed_root.mkdir()
    loop = allowed_root / "loop"
    loop.symlink_to(loop)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(allowed_root,)))
    ) as client:
        response = client.post(
            "/api/v1/libraries",
            headers=auth_headers(state_dir),
            json={"path": str(loop), "kind": "project"},
        )
        assert response.status_code == 422
        assert "cannot be resolved" in response.json()["detail"]


def test_project_binding_crud_and_user_library_rejection_are_atomic(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    project_root = tmp_path / "projects" / "alpha"
    project_root.mkdir(parents=True)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        project_library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "project"), "kind": "project"},
        ).json()
        replacement_library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "replacement"), "kind": "project"},
        ).json()
        user_library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "user"), "kind": "user"},
        ).json()

        rejected = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": user_library["id"]},
        )
        assert rejected.status_code == 422
        assert "project memory library" in rejected.json()["detail"]
        assert client.get("/api/v1/project-bindings", headers=headers).json() == []

        created = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": project_library["id"]},
        )
        assert created.status_code == 201
        binding = created.json()
        assert binding["project_id"] == binding["id"]
        assert binding["kind"] == "project"
        assert binding["canonical_root"] == str(project_root)
        assert binding["library_id"] == project_library["id"]
        assert binding["availability"] == "available"

        updated = client.put(
            f"/api/v1/project-bindings/{binding['id']}",
            headers=headers,
            json={"library_id": replacement_library["id"]},
        )
        assert updated.status_code == 200
        assert updated.json()["library_id"] == replacement_library["id"]

        rejected_update = client.put(
            f"/api/v1/project-bindings/{binding['id']}",
            headers=headers,
            json={"library_id": user_library["id"]},
        )
        assert rejected_update.status_code == 422
        listed = client.get("/api/v1/project-bindings", headers=headers).json()
        assert listed[0]["library_id"] == replacement_library["id"]

        deleted = client.delete(f"/api/v1/project-bindings/{binding['id']}", headers=headers)
        assert deleted.status_code == 204
        assert client.get("/api/v1/project-bindings", headers=headers).json() == []


def test_cwd_resolution_is_explicit_path_based_and_most_specific(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    projects = tmp_path / "projects"
    first = projects / "one" / "service"
    second = projects / "two" / "service"
    nested = first / "packages" / "nested"
    unbound_same_name = projects / "three" / "service"
    for path in (
        first / "src",
        nested / "src",
        second / "src",
        unbound_same_name / "src",
    ):
        path.mkdir(parents=True)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)

        def library(name: str) -> dict[str, str]:
            response = client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(library_root / name), "kind": "project"},
            )
            assert response.status_code == 201
            return response.json()

        first_library = library("first")
        second_library = library("second")
        nested_library = library("nested")

        bindings = []
        for root, library_id in (
            (first, first_library["id"]),
            (second, second_library["id"]),
            (nested, nested_library["id"]),
        ):
            response = client.post(
                "/api/v1/project-bindings",
                headers=headers,
                json={"project_root": str(root), "library_id": library_id},
            )
            assert response.status_code == 201
            bindings.append(response.json())

        first_result = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(first / "src")},
        ).json()
        second_result = client.post(
            "/mcp/project-bindings/resolve",
            headers=headers,
            json={"cwd": str(second / "src")},
        ).json()
        nested_result = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(nested / "src")},
        ).json()
        unbound_result = client.post(
            "/mcp/project-bindings/resolve",
            headers=headers,
            json={"cwd": str(unbound_same_name / "src")},
        ).json()

        assert first_result == {
            "status": "bound",
            "project_id": bindings[0]["id"],
            "binding_id": bindings[0]["id"],
            "binding_kind": "project",
            "project_root": str(first),
            "library_id": first_library["id"],
        }
        assert second_result["project_id"] == bindings[1]["id"]
        assert second_result["library_id"] == second_library["id"]
        assert nested_result["project_id"] == bindings[2]["id"]
        assert nested_result["library_id"] == nested_library["id"]
        assert unbound_result == {"status": "unbound", "cwd": str(unbound_same_name / "src")}


def test_worktree_inheritance_requires_explicit_association(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    main_root = tmp_path / "main"
    worktree_root = tmp_path / "worktrees" / "feature"
    (main_root / "src").mkdir(parents=True)
    (worktree_root / "src").mkdir(parents=True)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        first_library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "first"), "kind": "project"},
        ).json()
        second_library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "second"), "kind": "project"},
        ).json()
        main_binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(main_root), "library_id": first_library["id"]},
        ).json()

        before = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(worktree_root / "src")},
        ).json()
        assert before["status"] == "unbound"

        associated = client.post(
            "/api/v1/project-bindings/worktrees",
            headers=headers,
            json={
                "worktree_root": str(worktree_root),
                "main_project_binding_id": main_binding["id"],
            },
        )
        assert associated.status_code == 201
        association = associated.json()
        assert association["kind"] == "worktree"
        assert association["project_id"] == main_binding["id"]
        assert association["library_id"] == first_library["id"]

        inherited = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(worktree_root / "src")},
        ).json()
        assert inherited["project_id"] == main_binding["id"]
        assert inherited["binding_id"] == association["id"]
        assert inherited["binding_kind"] == "worktree"
        assert inherited["library_id"] == first_library["id"]

        client.put(
            f"/api/v1/project-bindings/{main_binding['id']}",
            headers=headers,
            json={"library_id": second_library["id"]},
        )
        updated = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(worktree_root / "src")},
        ).json()
        assert updated["library_id"] == second_library["id"]


def test_embedded_real_git_worktree_is_unbound_until_explicitly_associated(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    main_root = tmp_path / "main"
    subprocess.run(["git", "init", "-q", str(main_root)], check=True)
    subprocess.run(
        ["git", "-C", str(main_root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(main_root), "config", "user.name", "Test"], check=True
    )
    (main_root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(main_root), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(main_root), "commit", "-qm", "initial"], check=True
    )
    require_relative_worktrees(main_root)
    embedded_root = main_root / ".worktrees" / "feature"
    embedded_root.parent.mkdir()
    subprocess.run(
        [
            "git",
            "-C",
            str(main_root),
            "worktree",
            "add",
            "--relative-paths",
            "-qb",
            "feature",
            str(embedded_root),
        ],
        check=True,
    )
    (embedded_root / "src").mkdir()
    (main_root / "ordinary" / "src").mkdir(parents=True)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "project"), "kind": "project"},
        ).json()
        main_binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(main_root), "library_id": library["id"]},
        ).json()

        ordinary = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(main_root / "ordinary" / "src")},
        ).json()
        before = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(embedded_root / "src")},
        ).json()
        assert ordinary["project_id"] == main_binding["id"]
        assert before == {"status": "unbound", "cwd": str(embedded_root / "src")}

        association = client.post(
            "/api/v1/project-bindings/worktrees",
            headers=headers,
            json={
                "worktree_root": str(embedded_root),
                "main_project_binding_id": main_binding["id"],
            },
        ).json()
        after = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(embedded_root / "src")},
        ).json()
        assert after["binding_id"] == association["id"]
        assert after["project_id"] == main_binding["id"]
        assert after["library_id"] == library["id"]


def test_bound_worktree_root_detects_its_own_nested_worktree(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    primary_root = tmp_path / "primary"
    subprocess.run(["git", "init", "-q", str(primary_root)], check=True)
    subprocess.run(
        ["git", "-C", str(primary_root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(primary_root), "config", "user.name", "Test"], check=True
    )
    (primary_root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(primary_root), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(primary_root), "commit", "-qm", "initial"], check=True
    )
    require_relative_worktrees(primary_root)
    bound_root = primary_root / ".worktrees" / "bound"
    bound_root.parent.mkdir()
    subprocess.run(
        [
            "git",
            "-C",
            str(primary_root),
            "worktree",
            "add",
            "--relative-paths",
            "-qb",
            "bound",
            str(bound_root),
        ],
        check=True,
    )
    nested_root = bound_root / ".worktrees" / "nested"
    nested_root.parent.mkdir()
    subprocess.run(
        [
            "git",
            "-C",
            str(bound_root),
            "worktree",
            "add",
            "--relative-paths",
            "-qb",
            "nested",
            str(nested_root),
        ],
        check=True,
    )
    (nested_root / "src").mkdir()

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "project"), "kind": "project"},
        ).json()
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(bound_root), "library_id": library["id"]},
        ).json()

        before = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(nested_root / "src")},
        ).json()
        assert before == {"status": "unbound", "cwd": str(nested_root / "src")}

        association = client.post(
            "/api/v1/project-bindings/worktrees",
            headers=headers,
            json={
                "worktree_root": str(nested_root),
                "main_project_binding_id": binding["id"],
            },
        ).json()
        after = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(nested_root / "src")},
        ).json()
        assert after["binding_id"] == association["id"]
        assert after["project_id"] == binding["id"]


def test_project_root_git_file_is_not_mistaken_for_nested_worktree(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    project_root = tmp_path / "project"
    (project_root / "src").mkdir(parents=True)
    (project_root / ".git").write_text("gitdir: external\n", encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "project"), "kind": "project"},
        ).json()
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": library["id"]},
        ).json()

        resolved = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(project_root / "src")},
        ).json()
        assert resolved["project_id"] == binding["id"]


@pytest.mark.parametrize(
    "marker",
    [
        "gitdir: external\n",
        "gitdir: /tmp/not-main-metadata\n",
        "gitdir: /tmp/one\ngitdir: /tmp/two\n",
        "not a git marker\n",
        "x" * 4097,
    ],
)
def test_fake_nested_git_marker_does_not_cut_project_binding(
    tmp_path: Path, marker: str
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    project_root = tmp_path / "project"
    nested = project_root / "ordinary"
    (nested / "src").mkdir(parents=True)
    (nested / ".git").write_text(marker, encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "project"), "kind": "project"},
        ).json()
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": library["id"]},
        ).json()

        resolved = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(nested / "src")},
        ).json()
        assert resolved["project_id"] == binding["id"]


def test_moved_or_replaced_project_binding_is_diagnostic_and_never_rebound(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    project_root = tmp_path / "projects" / "service"
    project_root.mkdir(parents=True)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root / "project"), "kind": "project"},
        ).json()
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "library_id": library["id"]},
        ).json()

        moved = project_root.with_name("moved-service")
        project_root.rename(moved)
        project_root.mkdir()

        listed = client.get("/api/v1/project-bindings", headers=headers).json()[0]
        assert listed["id"] == binding["id"]
        assert listed["canonical_root"] == str(project_root)
        assert listed["availability"] == "replaced"
        replacement_result = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(project_root)},
        ).json()
        moved_result = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(moved)},
        ).json()
        assert replacement_result["status"] == "unbound"
        assert moved_result["status"] == "unbound"

        project_root.rmdir()
        listed_missing = client.get("/api/v1/project-bindings", headers=headers).json()[0]
        assert listed_missing["availability"] == "missing"


def test_markdown_chunking_preserves_headings_paragraphs_and_complete_fences() -> None:
    content = """# Guide

First paragraph.

## Install

```python
print("first")

print("second")
```

Final paragraph.
"""

    chunks = chunk_markdown(content)

    assert [(chunk.heading, chunk.start_line, chunk.end_line) for chunk in chunks] == [
        ("Guide", 3, 3),
        ("Guide > Install", 7, 11),
        ("Guide > Install", 13, 13),
    ]
    assert chunks[1].content.count("```") == 2
    assert 'print("first")\n\nprint("second")' in chunks[1].content


def test_fenced_code_obeys_commonmark_closing_and_container_rules() -> None:
    indented_close = chunk_markdown("```\nline\n    ```\nstill\n```\n")
    assert len(indented_close) == 1
    assert indented_close[0].kind == "code"
    assert indented_close[0].start_line == 1
    assert indented_close[0].end_line == 5
    assert "    ```\nstill" in indented_close[0].content

    listed = chunk_markdown("- item\n  ```python\n  print('list')\n  ```\n")
    assert [(chunk.kind, chunk.content) for chunk in listed] == [
        ("paragraph", "- item"),
        ("code", "  ```python\n  print('list')\n  ```"),
    ]
    quoted = chunk_markdown("> ```sh\n> echo quote\n> ```\n")
    assert [(chunk.kind, chunk.content) for chunk in quoted] == [
        ("code", "> ```sh\n> echo quote\n> ```")
    ]

    longer_marker = chunk_markdown("````\n~~~\n```\nbody\n`````\n")
    assert len(longer_marker) == 1
    assert longer_marker[0].kind == "code"
    assert longer_marker[0].content == "````\n~~~\n```\nbody\n`````"


def test_markdown_chunking_supports_setext_headings_and_accurate_locations() -> None:
    chunks = chunk_markdown(
        """Main title
==========

Intro paragraph.

Child title
-----------

Child paragraph.
"""
    )

    locations = [
        (chunk.heading, chunk.start_line, chunk.end_line, chunk.content) for chunk in chunks
    ]
    assert locations == [
        ("Main title", 4, 4, "Intro paragraph."),
        ("Main title > Child title", 9, 9, "Child paragraph."),
    ]


def test_setext_does_not_consume_lists_quotes_indented_code_or_thematic_breaks() -> None:
    chunks = chunk_markdown(
        """- list item
---

> quoted text
---

    indented code
---

---

Ordinary paragraph.
"""
    )
    assert [(chunk.kind, chunk.heading, chunk.content, chunk.start_line) for chunk in chunks] == [
        ("paragraph", None, "- list item", 1),
        ("thematic_break", None, "---", 2),
        ("paragraph", None, "> quoted text", 4),
        ("thematic_break", None, "---", 5),
        ("code", None, "    indented code", 7),
        ("thematic_break", None, "---", 8),
        ("thematic_break", None, "---", 10),
        ("paragraph", None, "Ordinary paragraph.", 12),
    ]


def test_legal_multiline_setext_heading_keeps_following_source_location() -> None:
    chunks = chunk_markdown("First title line\nsecond title line\n---\n\nBody.\n")
    locations = [
        (chunk.heading, chunk.content, chunk.start_line, chunk.end_line) for chunk in chunks
    ]
    assert locations == [("First title line second title line", "Body.", 5, 5)]


def test_markdown_chunking_preserves_authoritative_raw_blocks() -> None:
    chunks = chunk_markdown(
        """# Section

<section data-key="HtmlNeedle42">
raw html
</section>

[doc]: https://example.test "ReferenceNeedle43"

> ## QuoteHeadingNeedle44

# HeadingOnlyNeedle45
"""
    )

    assert [(chunk.kind, chunk.start_line, chunk.end_line, chunk.content) for chunk in chunks] == [
        (
            "html",
            3,
            5,
            '<section data-key="HtmlNeedle42">\nraw html\n</section>',
        ),
        (
            "reference_definition",
            7,
            7,
            '[doc]: https://example.test "ReferenceNeedle43"',
        ),
        ("heading", 9, 9, "> ## QuoteHeadingNeedle44"),
        ("heading", 11, 11, "# HeadingOnlyNeedle45"),
    ]


def test_markdown_chunking_preserves_every_duplicate_reference_definition() -> None:
    chunks = chunk_markdown(
        """# References

[same]: https://example.test/first "FirstReferenceNeedle"
[same]:
  https://example.test/second
  "SecondReferenceNeedle"
"""
    )

    assert [(chunk.kind, chunk.start_line, chunk.end_line, chunk.content) for chunk in chunks] == [
        ("heading", 1, 1, "# References"),
        (
            "reference_definition",
            3,
            3,
            '[same]: https://example.test/first "FirstReferenceNeedle"',
        ),
        (
            "reference_definition",
            4,
            6,
            '[same]:\n  https://example.test/second\n  "SecondReferenceNeedle"',
        ),
    ]


def test_sidecar_chunk_matching_survives_heading_rename_and_duplicate_prepend() -> None:
    old = [
        StoredChunk("first-id", "paragraph", "First.", "Old", 3, 3),
        StoredChunk("repeat-one", "paragraph", "Repeated.", "Old", 5, 5),
        StoredChunk("repeat-two", "paragraph", "Repeated.", "Old", 7, 7),
    ]
    renamed = chunk_markdown("# New\n\nFirst.\n\nRepeated.\n\nRepeated.\n")
    assert match_chunk_ids(old, renamed) == ["first-id", "repeat-one", "repeat-two"]

    prepended = chunk_markdown("# Old\n\nFirst.\n\nRepeated.\n\nRepeated.\n\nRepeated.\n")
    prepended_ids = match_chunk_ids(old, prepended)
    assert prepended_ids[0] == "first-id"
    assert prepended_ids[1:3] == ["repeat-one", "repeat-two"]
    assert prepended_ids[3] not in {chunk.id for chunk in old}


def test_sidecar_chunk_matching_follows_content_through_delete_and_reorder() -> None:
    old = [
        StoredChunk("a-id", "paragraph", "Alpha.", "Stable", 3, 3),
        StoredChunk("b-id", "paragraph", "Beta.", "Stable", 5, 5),
        StoredChunk("c-id", "paragraph", "Gamma.", "Stable", 7, 7),
    ]
    reordered = chunk_markdown("# Stable\n\nGamma.\n\nAlpha.\n")
    assert match_chunk_ids(old, reordered) == ["c-id", "a-id"]



def test_duplicate_chunk_matching_minimizes_displacement_for_edit_matrix() -> None:
    def current(*positions: int) -> list[MarkdownChunk]:
        return [MarkdownChunk("paragraph", "Stable", line, line, "Same.") for line in positions]

    old = [
        StoredChunk("one", "paragraph", "Same.", "Stable", 5, 5),
        StoredChunk("two", "paragraph", "Same.", "Stable", 7, 7),
    ]
    front_insert = match_chunk_ids(old, current(1, 5, 7))
    middle_insert = match_chunk_ids(old, current(5, 6, 7))
    back_insert = match_chunk_ids(old, current(5, 7, 11))
    assert front_insert[1:] == ["one", "two"]
    assert front_insert[0] not in {"one", "two"}
    assert middle_insert[0] == "one"
    assert middle_insert[2] == "two"
    assert middle_insert[1] not in {"one", "two"}
    assert back_insert[:2] == ["one", "two"]
    assert back_insert[2] not in {"one", "two"}

    with_edges = [
        StoredChunk("front", "paragraph", "Same.", "Stable", 1, 1),
        *old,
        StoredChunk("back", "paragraph", "Same.", "Stable", 11, 11),
    ]
    assert match_chunk_ids(with_edges, current(5, 7, 11)) == ["one", "two", "back"]
    assert match_chunk_ids(with_edges, current(1, 5, 7)) == ["front", "one", "two"]

    anchored_old = [
        StoredChunk("anchor", "paragraph", "Anchor.", "Stable", 3, 3),
        StoredChunk("one", "paragraph", "Same.", "Stable", 5, 5),
        StoredChunk("two", "paragraph", "Same.", "Stable", 7, 7),
    ]
    anchored_front_insert = [
        MarkdownChunk("paragraph", "Stable", 3, 3, "Same."),
        MarkdownChunk("paragraph", "Stable", 5, 5, "Anchor."),
        MarkdownChunk("paragraph", "Stable", 7, 7, "Same."),
        MarkdownChunk("paragraph", "Stable", 9, 9, "Same."),
    ]
    anchored_ids = match_chunk_ids(anchored_old, anchored_front_insert)
    assert anchored_ids[1:] == ["anchor", "one", "two"]
    assert anchored_ids[0] not in {"anchor", "one", "two"}


def test_duplicate_matching_uses_real_coordinates_and_heading_context() -> None:
    original = chunk_markdown(
        """# Alpha

Same.

Same.

# Beta

Same.

Anchor.

Same.
"""
    )
    old = [
        StoredChunk(
            f"old-{index}",
            chunk.kind,
            chunk.content,
            chunk.heading,
            chunk.start_line,
            chunk.end_line,
        )
        for index, chunk in enumerate(original)
    ]

    front_insert = chunk_markdown(
        """# Alpha

Same.

Same.

Same.

# Beta

Same.

Anchor.

Same.
"""
    )
    front_ids = match_chunk_ids(old, front_insert)
    assert front_ids[1:3] == ["old-0", "old-1"]
    assert front_ids[3:] == ["old-2", "old-3", "old-4"]
    assert front_ids[0] not in {chunk.id for chunk in old}

    extra_heading = chunk_markdown(
        """# Gamma

Same.

# Alpha

Same.

Same.

# Beta

Same.

Anchor.

Same.
"""
    )
    extra_ids = match_chunk_ids(old, extra_heading)
    assert extra_ids[1:] == ["old-0", "old-1", "old-2", "old-3", "old-4"]
    assert extra_ids[0] not in {chunk.id for chunk in old}


def test_duplicate_matching_allows_section_rename_after_context_anchor() -> None:
    original = chunk_markdown("# Old name\n\nAnchor.\n\nSame.\n\nSame.\n")
    old = [
        StoredChunk(
            f"old-{index}",
            chunk.kind,
            chunk.content,
            chunk.heading,
            chunk.start_line,
            chunk.end_line,
        )
        for index, chunk in enumerate(original)
    ]
    renamed = chunk_markdown("# New name\n\nAnchor.\n\nSame.\n\nSame.\n")
    assert match_chunk_ids(old, renamed) == ["old-0", "old-1", "old-2"]


def test_duplicate_matching_does_not_reassign_deleted_cross_heading_id() -> None:
    old = [
        StoredChunk("a-id", "paragraph", "Same.", "A", 3, 3),
        StoredChunk("b-id", "paragraph", "Same.", "B", 7, 7),
    ]
    current = [
        MarkdownChunk("paragraph", "X", 3, 3, "Same."),
        MarkdownChunk("paragraph", "A", 7, 7, "Same."),
    ]

    matched = match_chunk_ids(old, current)

    assert matched[1] == "a-id"
    assert matched[0] not in {"a-id", "b-id"}


def test_duplicate_matching_is_bounded_for_thousands_of_identical_chunks() -> None:
    count = 3_200
    old_duplicates = [
        StoredChunk(f"id-{index}", "paragraph", "Same.", "Bulk", index * 2 + 1, index * 2 + 1)
        for index in range(count)
    ]
    old = [
        *old_duplicates,
        StoredChunk("tail", "paragraph", "Tail anchor.", "Bulk", count * 2 + 1, count * 2 + 1),
    ]
    current_duplicates = [
        MarkdownChunk("paragraph", "Bulk", index * 2 + 1, index * 2 + 1, "Same.")
        for index in range(count + 1)
    ]
    current = [
        *current_duplicates,
        MarkdownChunk("paragraph", "Bulk", count * 2 + 3, count * 2 + 3, "Tail anchor."),
    ]

    started = time.monotonic()
    matched = match_chunk_ids(old, current)
    elapsed = time.monotonic() - started

    assert matched[1:-1] == [chunk.id for chunk in old_duplicates]
    assert matched[-1] == "tail"
    assert matched[0] not in {chunk.id for chunk in old}
    assert elapsed < 2.0


def test_duplicate_matching_uses_bounded_fallback_for_large_delta() -> None:
    old_count = 10_000
    new_count = 20_000
    old = [
        StoredChunk(f"id-{index}", "paragraph", "Same.", "Bulk", index * 2 + 1, index * 2 + 1)
        for index in range(old_count)
    ]
    current = [
        MarkdownChunk("paragraph", "Bulk", index * 2 + 1, index * 2 + 1, "Same.")
        for index in range(new_count)
    ]

    started = time.monotonic()
    matched = match_chunk_ids(old, current)
    elapsed = time.monotonic() - started

    assert matched[:old_count] == [chunk.id for chunk in old]
    assert not ({chunk.id for chunk in old} & set(matched[old_count:]))
    assert elapsed < 5.0


def test_legacy_chunk_kind_migration_preserves_code_id_after_heading_change(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "legacy"
    library_path.mkdir(parents=True)
    source = library_path / "code.md"
    original = "# Old heading\n\n```python\nprint('kept')\n```\n"
    source.write_text(original, encoding="utf-8")
    database_path = state_dir / "platform.sqlite3"
    state_dir.mkdir()
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE memory_libraries (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, canonical_path TEXT NOT NULL UNIQUE,
            sync_status TEXT NOT NULL, created_at TEXT NOT NULL, ignore_patterns TEXT NOT NULL
        );
        CREATE TABLE memory_documents (
            id TEXT PRIMARY KEY, library_id TEXT NOT NULL, path TEXT NOT NULL,
            source_version TEXT NOT NULL, UNIQUE(library_id, path)
        );
        CREATE TABLE memory_chunks (
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL, library_id TEXT NOT NULL,
            path TEXT NOT NULL, heading TEXT, start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL, content TEXT NOT NULL, source_version TEXT NOT NULL
        );
        CREATE TABLE memory_versions (
            version_id TEXT PRIMARY KEY, library_id TEXT NOT NULL, path TEXT NOT NULL,
            source_version TEXT NOT NULL, content TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'current', supersedes_version_id TEXT,
            commit_id TEXT, effective_at TEXT, condition_text TEXT, candidate_id TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(library_id, path, source_version)
        );
        CREATE TABLE library_git_repositories (
            library_id TEXT PRIMARY KEY, git_dir TEXT NOT NULL UNIQUE,
            mode TEXT NOT NULL, created_at TEXT
        );
        """
    )
    version = hashlib.sha256(original.encode()).hexdigest()
    connection.execute(
        "INSERT INTO memory_libraries VALUES (?, 'project', ?, 'ready', CURRENT_TIMESTAMP, '[]')",
        ("legacy-library", str(library_path)),
    )
    connection.execute(
        "INSERT INTO memory_documents VALUES (?, ?, 'code.md', ?)",
        ("legacy-document", "legacy-library", version),
    )
    connection.execute(
        """INSERT INTO memory_chunks VALUES
           (?, ?, ?, 'code.md', 'Old heading', 3, 5, ?, ?)""",
        (
            "legacy-code-id",
            "legacy-document",
            "legacy-library",
            "```python\nprint('kept')\n```",
            version,
        ),
    )
    connection.execute(
        """INSERT INTO memory_versions
           (version_id, library_id, path, source_version, content)
           VALUES ('legacy-version', 'legacy-library', 'code.md', ?, ?)""",
        (version, original),
    )
    repository = GitRepository(state_dir / "git" / "legacy-library.git", library_path)
    repository.initialize("legacy-library", {"code.md": original.encode()})
    connection.execute(
        """INSERT INTO library_git_repositories (library_id, git_dir, mode)
           VALUES ('legacy-library', ?, 'sidecar')""",
        (str(repository.git_dir),),
    )
    connection.commit()
    connection.close()

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        source.write_text(original.replace("Old heading", "New heading"), encoding="utf-8")
        import_external_changes(client, headers, "legacy-library", library_path)
        migrated = sqlite3.connect(database_path).execute(
            "SELECT id, kind, heading FROM memory_chunks"
        ).fetchone()
        assert migrated == ("legacy-code-id", "code", "New heading")


def test_markdown_import_ignore_rules_symlink_safety_and_stable_sidecar_ids(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project"
    outside = tmp_path / "outside.md"
    (library_path / "notes").mkdir(parents=True)
    (library_path / ".git").mkdir()
    (library_path / "node_modules").mkdir()
    (library_path / "notes" / "keep.md").write_text(
        "# Stable\n\nRepeated fact.\n\nRepeated fact.\n", encoding="utf-8"
    )
    (library_path / "private.md").write_text("private keyword", encoding="utf-8")
    (library_path / ".git" / "hidden.md").write_text("git secret", encoding="utf-8")
    (library_path / "node_modules" / "hidden.md").write_text(
        "dependency secret", encoding="utf-8"
    )
    outside.write_text("outside secret", encoding="utf-8")
    (library_path / "escape.md").symlink_to(outside)

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        initial = connection.execute(
            "SELECT id, path, source_version FROM memory_documents WHERE library_id = ?",
            (library["id"],),
        ).fetchall()
        initial_chunks = connection.execute(
            "SELECT id FROM memory_chunks WHERE library_id = ? ORDER BY start_line",
            (library["id"],),
        ).fetchall()
        assert {row[1] for row in initial} == {"notes/keep.md", "private.md"}
        assert len(initial_chunks) == 3
        assert len(set(initial_chunks)) == 3

        unchanged = client.post(
            f"/api/v1/libraries/{library['id']}/scan", headers=headers
        ).json()
        assert unchanged["detected"] == 0
        assert unchanged["pending"] == 0
        assert connection.execute(
            "SELECT id FROM memory_chunks WHERE library_id = ? ORDER BY start_line",
            (library["id"],),
        ).fetchall() == initial_chunks

        rules = client.put(
            f"/api/v1/libraries/{library['id']}/ignore-rules",
            headers=headers,
            json={"patterns": ["private.md"]},
        )
        assert rules.status_code == 200
        assert rules.json()["patterns"] == ["private.md"]
        assert connection.execute(
            "SELECT path FROM memory_documents WHERE library_id = ?",
            (library["id"],),
        ).fetchall() == [("notes/keep.md",)]
        connection.close()


def test_incomplete_markdown_scan_preserves_old_index_and_reports_error(
    tmp_path: Path, monkeypatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project"
    nested = library_path / "nested"
    nested.mkdir(parents=True)
    source = nested / "knowledge.md"
    source.write_text("# Stable\n\nPreservedNeedle42.\n", encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        project = tmp_path / "project"
        project.mkdir()
        client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library["id"]},
        )
        database = sqlite3.connect(state_dir / "platform.sqlite3")
        before = database.execute(
            "SELECT id, source_version FROM memory_documents WHERE library_id = ?",
            (library["id"],),
        ).fetchall()

        real_scandir = markdown_index.os.scandir

        def failing_scandir(path):
            if Path(path) == nested:
                raise PermissionError("temporary directory read failure")
            return real_scandir(path)

        monkeypatch.setattr(markdown_index.os, "scandir", failing_scandir)
        failed = client.post(
            f"/api/v1/libraries/{library['id']}/scan",
            headers=headers,
        )
        assert failed.status_code == 422
        assert "scan incomplete" in failed.json()["detail"]
        assert database.execute(
            "SELECT id, source_version FROM memory_documents WHERE library_id = ?",
            (library["id"],),
        ).fetchall() == before
        assert client.get(
            f"/mcp/libraries/{library['id']}/sync-status", headers=headers
        ).json()["sync_status"] == "error"
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(project), "query": "PreservedNeedle42"},
        ).json()["results"]

        monkeypatch.setattr(markdown_index.os, "scandir", real_scandir)
        recovered = client.post(
            f"/api/v1/libraries/{library['id']}/scan", headers=headers
        )
        assert recovered.status_code == 200
        assert recovered.json()["detected"] == 0
        database.close()


def test_invalid_utf8_scan_is_incomplete_and_does_not_replace_old_document(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    source = library_root / "project" / "knowledge.md"
    source.parent.mkdir(parents=True)
    source.write_text("Valid original.", encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(source.parent), "kind": "project"},
        ).json()
        source.write_bytes(b"\xff\xfe")
        response = client.post(
            f"/api/v1/libraries/{library['id']}/scan", headers=headers
        )
        assert response.status_code == 422
        assert "not valid UTF-8" in response.json()["detail"]
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        assert connection.execute(
            "SELECT content FROM memory_chunks WHERE library_id = ?", (library["id"],)
        ).fetchall() == [("Valid original.",)]
        assert connection.execute(
            "SELECT sync_status FROM memory_libraries WHERE id = ?", (library["id"],)
        ).fetchone() == ("error",)
        connection.close()


def test_html_block_is_indexed_in_sqlite_and_searchable_over_rest_and_mcp(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project"
    project = tmp_path / "project"
    library_path.mkdir(parents=True)
    project.mkdir()
    (library_path / "html.md").write_text(
        "# HTML\n\n<div>HtmlSearchNeedle42</div>\n", encoding="utf-8"
    )

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library["id"]},
        )

        stored = sqlite3.connect(state_dir / "platform.sqlite3").execute(
            "SELECT kind, heading, start_line, end_line, content FROM memory_chunks"
        ).fetchone()
        assert stored == (
            "html",
            "HTML",
            3,
            3,
            "<div>HtmlSearchNeedle42</div>",
        )

        for endpoint in ("/api/v1/search", "/mcp/search"):
            response = client.post(
                endpoint,
                headers=headers,
                json={"cwd": str(project), "query": "HtmlSearchNeedle42"},
            )
            assert response.status_code == 200
            hit = response.json()["results"][0]
            assert hit["content"] == "<div>HtmlSearchNeedle42</div>"
            assert hit["start_line"] == hit["end_line"] == 3


def test_duplicate_reference_definitions_are_searchable_over_rest_and_mcp(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project"
    project = tmp_path / "project"
    library_path.mkdir(parents=True)
    project.mkdir()
    (library_path / "references.md").write_text(
        """# References

[same]: https://example.test/first "FirstReferenceNeedle"
[same]: https://example.test/second "SecondReferenceNeedle"
""",
        encoding="utf-8",
    )

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": library["id"]},
        )

        stored = sqlite3.connect(state_dir / "platform.sqlite3").execute(
            """SELECT kind, start_line, content FROM memory_chunks
               WHERE kind = 'reference_definition' ORDER BY start_line"""
        ).fetchall()
        assert stored == [
            (
                "reference_definition",
                3,
                '[same]: https://example.test/first "FirstReferenceNeedle"',
            ),
            (
                "reference_definition",
                4,
                '[same]: https://example.test/second "SecondReferenceNeedle"',
            ),
        ]

        for endpoint in ("/api/v1/search", "/mcp/search"):
            response = client.post(
                endpoint,
                headers=headers,
                json={"cwd": str(project), "query": "SecondReferenceNeedle"},
            )
            assert response.status_code == 200
            assert response.json()["results"][0]["start_line"] == 4


def test_sqlite_rescan_keeps_only_exact_heading_duplicate_id(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project"
    library_path.mkdir(parents=True)
    source = library_path / "same.md"
    source.write_text("# A\n\nSame.\n\n# B\n\nSame.\n", encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_path), "kind": "project"},
        ).json()
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        old_ids = dict(
            connection.execute(
                "SELECT heading, id FROM memory_chunks WHERE library_id = ?",
                (library["id"],),
            ).fetchall()
        )

        source.write_text("# X\n\nSame.\n\n# A\n\nSame.\n", encoding="utf-8")
        import_external_changes(client, headers, library["id"], library_path)
        new_ids = dict(
            connection.execute(
                "SELECT heading, id FROM memory_chunks WHERE library_id = ?",
                (library["id"],),
            ).fetchall()
        )

        assert new_ids["A"] == old_ids["A"]
        assert new_ids["X"] not in set(old_ids.values())
        assert old_ids["B"] not in set(new_ids.values())
        connection.close()


def test_direct_search_is_project_scoped_source_attributed_and_incremental(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    first_library = library_root / "first"
    second_library = library_root / "second"
    first_project = tmp_path / "projects" / "first"
    second_project = tmp_path / "projects" / "second"
    unbound_project = tmp_path / "projects" / "unbound"
    for path in (first_library, second_library, first_project, second_project, unbound_project):
        path.mkdir(parents=True)
    source = first_library / "knowledge.md"
    source.write_text("# 决策\n\nProject alpha uses ExactNeedle42.\n", encoding="utf-8")
    (second_library / "private.md").write_text(
        "# Other\n\nExactNeedle42 belongs elsewhere.\n", encoding="utf-8"
    )

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)

        def register(path: Path) -> dict[str, str]:
            return client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(path), "kind": "project"},
            ).json()

        first = register(first_library)
        second = register(second_library)
        for project, library in ((first_project, first), (second_project, second)):
            response = client.post(
                "/api/v1/project-bindings",
                headers=headers,
                json={"project_root": str(project), "library_id": library["id"]},
            )
            assert response.status_code == 201

        rest = client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(first_project), "query": "ExactNeedle42"},
        )
        assert rest.status_code == 200
        result = rest.json()
        assert result["library_id"] == first["id"]
        assert len(result["results"]) == 1
        hit = result["results"][0]
        assert hit["content"] == "Project alpha uses ExactNeedle42."
        assert hit["path"] == "knowledge.md"
        assert hit["heading"] == "决策"
        assert hit["start_line"] == hit["end_line"] == 3
        assert hit["source_type"] == "markdown"
        assert hit["classification"] == "direct"
        assert len(hit["document_id"]) == 36
        assert len(hit["chunk_id"]) == 36
        assert len(hit["source_version"]) == 64
        chinese = client.post(
            "/mcp/search",
            headers=headers,
            json={"cwd": str(first_project), "query": "决策"},
        ).json()
        assert chinese["results"][0]["heading"] == "决策"

        other = client.post(
            "/mcp/search",
            headers=headers,
            json={"cwd": str(second_project), "query": "ExactNeedle42"},
        ).json()
        assert other["library_id"] == second["id"]
        assert len(other["results"]) == 1
        assert other["results"][0]["path"] == "private.md"
        unbound = client.post(
            "/mcp/search",
            headers=headers,
            json={"cwd": str(unbound_project), "query": "ExactNeedle42"},
        ).json()
        assert unbound["status"] == "unbound"
        assert unbound["query"] == "ExactNeedle42"
        assert unbound["results"] == []
        assert unbound["scope"]["kind"] == "project_cwd"

        original_chunk_id = hit["chunk_id"]
        source.write_text("# 新决策\n\nProject alpha uses ExactNeedle42.\n", encoding="utf-8")
        import_external_changes(client, headers, first["id"], first_library)
        renamed = client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(first_project), "query": "ExactNeedle42"},
        ).json()["results"][0]
        assert renamed["chunk_id"] == original_chunk_id
        assert renamed["heading"] == "新决策"

        source.write_text("# 决策\n\nProject alpha uses UpdatedNeedle77.\n", encoding="utf-8")
        import_external_changes(client, headers, first["id"], first_library)
        updated = client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(first_project), "query": "UpdatedNeedle77"},
        ).json()["results"][0]
        assert updated["chunk_id"] != original_chunk_id
        assert updated["source_version"] != hit["source_version"]
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(first_project), "query": "ExactNeedle42"},
        ).json()["results"] == []

        source.unlink()
        import_external_changes(client, headers, first["id"], first_library)
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(first_project), "query": "UpdatedNeedle77"},
        ).json()["results"] == []


def test_web_document_edit_preview_history_and_restore_are_atomic(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    original = "# Decision\n\nUse old value.\n\n```python\nprint('old')\n```\n"
    updated = "# Decision\n\nUse new value.\n\n```python\nprint('new')\n```\n\n" + (
        "Long paragraph content. " * 500
    )
    document = library / "decisions.md"
    document.write_text(original, encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        registered = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library), "kind": "project"},
        ).json()
        library_id = registered["id"]
        tree = client.get(
            f"/api/v1/libraries/{library_id}/documents", headers=headers
        ).json()
        assert tree == [
            {
                "library_id": library_id,
                "path": "decisions.md",
                "source_version": hashlib.sha256(original.encode()).hexdigest(),
            }
        ]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "decisions.md"},
        ).json()
        preview = client.post(
            f"/api/v1/libraries/{library_id}/document/preview",
            headers=headers,
            json={
                "path": "decisions.md",
                "content": updated + "\n<script>alert(1)</script>\n",
                "expected_source_version": loaded["source_version"],
            },
        )
        assert preview.status_code == 200
        assert "-Use old value." in preview.json()["diff"]
        assert "+Use new value." in preview.json()["diff"]
        assert "<pre><code class=\"language-python\">" in preview.json()["rendered_html"]
        assert "<script>" not in preview.json()["rendered_html"]

        edit_payload = {
            "path": "decisions.md",
            "content": updated,
            "expected_source_version": loaded["source_version"],
            "operation_id": "edit-1",
            "actor_type": "user",
            "source": "web",
        }
        edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json=edit_payload,
        )
        assert edit.status_code == 200
        assert document.read_text(encoding="utf-8") == updated
        assert client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json=edit_payload,
        ).json() == edit.json()
        reused = dict(edit_payload, content="different")
        assert client.put(
            f"/api/v1/libraries/{library_id}/document", headers=headers, json=reused
        ).status_code == 422

        history = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        assert len(history) == 2
        assert history[0]["commit"] == edit.json()["commit"]
        assert history[0]["author_name"] == "Personal Agent Memory"
        assert history[0]["author_email"] == "memory-platform@localhost"
        assert history[0]["actor_type"] == "user"
        assert history[0]["source"] == "web"
        diff = client.get(
            f"/api/v1/libraries/{library_id}/history/{history[0]['commit']}/diff",
            headers=headers,
        ).json()
        assert {line["kind"] for line in diff["lines"]} >= {"addition", "deletion", "hunk"}

        restore = client.post(
            f"/api/v1/libraries/{library_id}/history/restore",
            headers=headers,
            json={
                "path": "decisions.md",
                "commit": history[1]["commit"],
                "expected_source_version": edit.json()["source_version"],
                "operation_id": "restore-1",
                "actor_type": "user",
                "source": "web-history",
            },
        )
        assert restore.status_code == 200
        assert restore.json()["commit"] != history[0]["commit"]
        assert document.read_text(encoding="utf-8") == original
        restored_history = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        assert len(restored_history) == 3
        assert restored_history[0]["kind"] == "restore"
        assert restored_history[0]["source"] == "web-history"

        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        git_dir, mode = connection.execute(
            "SELECT git_dir, mode FROM library_git_repositories WHERE library_id = ?",
            (library_id,),
        ).fetchone()
        assert mode == "sidecar"
        assert Path(git_dir).is_relative_to(state_dir / "git")
        tracked = subprocess.run(
            ["git", f"--git-dir={git_dir}", "ls-tree", "-r", "--name-only", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        assert tracked == [".personal-agent-memory.json", "decisions.md"]
        assert not (library / ".git").exists()
        assert subprocess.run(
            ["git", f"--git-dir={git_dir}", "remote"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout == ""
        operations = connection.execute(
            "SELECT kind, actor_type, source FROM memory_operations ORDER BY created_at, rowid"
        ).fetchall()
        assert operations == [("edit", "user", "web"), ("restore", "user", "web-history")]
        connection.close()


def test_sidecar_history_never_touches_containing_project_repository(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    project = tmp_path / "project"
    library = project / "memory"
    library.mkdir(parents=True)
    (project / ".gitignore").write_text("/memory/\n", encoding="utf-8")
    document = library / "memory.md"
    document.write_text("# Fact\n\nBefore.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main", str(project)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(project), "add", ".gitignore"], check=True)
    subprocess.run(
        [
            "git", "-C", str(project), "-c", "user.name=Owner", "-c",
            "user.email=owner@example.test", "commit", "-m", "project baseline",
        ],
        check=True,
        capture_output=True,
    )

    def project_fingerprint() -> tuple[str, str, bytes, str]:
        head = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout
        branch = subprocess.run(
            ["git", "-C", str(project), "branch", "--show-current"], check=True,
            capture_output=True, text=True,
        ).stdout
        status_output = subprocess.run(
            ["git", "-C", str(project), "status", "--porcelain=v2"], check=True,
            capture_output=True, text=True,
        ).stdout
        return head, branch, (project / ".git" / "index").read_bytes(), status_output

    before = project_fingerprint()
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(project,)))
    ) as client:
        headers = auth_headers(state_dir)
        registered = client.post(
            "/api/v1/libraries", headers=headers,
            json={"path": str(library), "kind": "project"},
        ).json()
        loaded = client.get(
            f"/api/v1/libraries/{registered['id']}/document", headers=headers,
            params={"path": "memory.md"},
        ).json()
        response = client.put(
            f"/api/v1/libraries/{registered['id']}/document", headers=headers,
            json={
                "path": "memory.md", "content": "# Fact\n\nAfter.\n",
                "expected_source_version": loaded["source_version"],
                "operation_id": "project-contained-edit", "actor_type": "user",
                "source": "web",
            },
        )
        assert response.status_code == 200
    assert project_fingerprint() == before


def test_existing_git_repository_requires_explicit_reuse_authorization(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    default_library = library_root / "default"
    authorized_library = library_root / "authorized"
    for library in (default_library, authorized_library):
        library.mkdir(parents=True)
        (library / "note.md").write_text("# Note\n\nOriginal.\n", encoding="utf-8")
        subprocess.run(["git", "init", "-b", "main", str(library)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(library), "add", "note.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )
        default_head = subprocess.run(
        ["git", "-C", str(default_library), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    default_status = subprocess.run(
        ["git", "-C", str(default_library), "status", "--porcelain=v2"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    default_index = (default_library / ".git" / "index").read_bytes()

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        default = client.post(
            "/api/v1/libraries", headers=headers,
            json={"path": str(default_library), "kind": "project"},
        )
        assert default.status_code == 201
        assert subprocess.run(
            ["git", "-C", str(default_library), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip() == default_head
        assert subprocess.run(
            ["git", "-C", str(default_library), "status", "--porcelain=v2"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout == default_status
        assert (default_library / ".git" / "index").read_bytes() == default_index
        assert not (default_library / ".personal-agent-memory.json").exists()

        authorized = client.post(
            "/api/v1/libraries", headers=headers,
            json={
                "path": str(authorized_library), "kind": "project",
                "reuse_existing_git": True,
            },
        )
        assert authorized.status_code == 201
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        git_dir, mode = connection.execute(
            "SELECT git_dir, mode FROM library_git_repositories WHERE library_id = ?",
            (authorized.json()["id"],),
        ).fetchone()
        assert Path(git_dir) == authorized_library / ".git"
        assert mode == "authorized_existing"
        assert subprocess.run(
            ["git", "-C", str(authorized_library), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout == ""
        connection.close()


def test_failed_git_commit_restores_authoritative_file_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    document = library / "fact.md"
    original = "# Fact\n\nOriginal searchable value.\n"
    document.write_text(original, encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        registered = client.post(
            "/api/v1/libraries", headers=headers,
            json={"path": str(library), "kind": "project"},
        ).json()
        library_id = registered["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document", headers=headers,
            params={"path": "fact.md"},
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()

        def fail_commit(
            self: GitRepository,
            contents: dict[str, bytes | None],
            message: str,
        ) -> str:
            raise GitHistoryError("injected Git failure")

        monkeypatch.setattr(GitRepository, "commit", fail_commit)
        failed = client.put(
            f"/api/v1/libraries/{library_id}/document", headers=headers,
            json={
                "path": "fact.md", "content": "# Fact\n\nBroken partial value.\n",
                "expected_source_version": loaded["source_version"],
                "operation_id": "failing-edit", "actor_type": "user", "source": "test",
            },
        )
        assert failed.status_code == 422
        assert failed.json()["detail"] == "injected Git failure"
        assert document.read_text(encoding="utf-8") == original
        assert client.get(
            f"/api/v1/libraries/{library_id}/document", headers=headers,
            params={"path": "fact.md"},
        ).json()["source_version"] == loaded["source_version"]
        assert client.post(
            "/api/v1/search", headers=headers,
            json={"cwd": str(tmp_path), "query": "partial"},
        ).json()["results"] == []
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_operations WHERE operation_id = 'failing-edit'"
        ).fetchone() == (0,)
        connection.close()


def test_scan_failure_after_git_commit_rolls_back_every_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    document = library / "fact.md"
    original = "# Fact\n\nOriginal scan value.\n"
    document.write_text(original, encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library_id = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library), "kind": "project"},
        ).json()["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "fact.md"},
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        original_scan = PlatformState.scan_library

        def fail_transactional_scan(
            state: PlatformState,
            target_library_id: str,
            *,
            participate_in_transaction: bool = False,
            frozen_scan=None,
        ):
            if participate_in_transaction:
                state.connection_or_raise.execute(
                    "UPDATE memory_libraries SET sync_status = 'error' WHERE id = ?",
                    (target_library_id,),
                )
                raise MemoryMutationError("injected transactional scan failure")
            return original_scan(
                state,
                target_library_id,
                participate_in_transaction=participate_in_transaction,
                frozen_scan=frozen_scan,
            )

        monkeypatch.setattr(PlatformState, "scan_library", fail_transactional_scan)
        failed = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "fact.md",
                "content": "# Fact\n\nUncommitted scan value.\n",
                "expected_source_version": loaded["source_version"],
                "operation_id": "scan-failure",
                "actor_type": "user",
                "source": "fault-injection",
            },
        )
        assert failed.status_code == 422
        assert document.read_text(encoding="utf-8") == original
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        assert connection.execute(
            "SELECT sync_status FROM memory_libraries WHERE id = ?", (library_id,)
        ).fetchone() == ("ready",)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_operations WHERE operation_id = 'scan-failure'"
        ).fetchone() == (0,)
        connection.close()


def test_temporary_real_directory_swap_during_mutation_scan_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    section = library / "section"
    section.mkdir(parents=True)
    document = section / "fact.md"
    original = "# Fact\n\nOriginal authoritative value.\n"
    document.write_text(original, encoding="utf-8")
    injected = library / "injected-section"
    injected.mkdir()
    (injected / "fact.md").write_text(
        "# Fact\n\nInjected temporary value.\n", encoding="utf-8"
    )
    held = library / "held-section"

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library_id = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library), "kind": "project"},
        ).json()["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "section/fact.md"},
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        before_database = sqlite3.connect(state_dir / "platform.sqlite3")
        before_index = (
            before_database.execute(
                "SELECT * FROM memory_documents ORDER BY id"
            ).fetchall(),
            before_database.execute(
                "SELECT * FROM memory_chunks ORDER BY id"
            ).fetchall(),
            before_database.execute(
                "SELECT * FROM memory_chunk_search ORDER BY chunk_id"
            ).fetchall(),
        )
        before_database.close()
        original_scan = PlatformState._scan_bound_library
        calls = 0

        def swapped_scan(state, root, bound, target_library_id):
            nonlocal calls
            calls += 1
            if calls != 2:
                return original_scan(state, root, bound, target_library_id)
            section.rename(held)
            injected.rename(section)
            try:
                return original_scan(state, root, bound, target_library_id)
            finally:
                section.rename(injected)
                held.rename(section)

        monkeypatch.setattr(PlatformState, "_scan_bound_library", swapped_scan)
        failed = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "section/fact.md",
                "content": "# Fact\n\nAttempted platform edit.\n",
                "expected_source_version": loaded["source_version"],
                "operation_id": "temporary-directory-swap",
                "actor_type": "user",
                "source": "race-test",
            },
        )
        assert failed.status_code == 422
        assert document.read_text(encoding="utf-8") == original
        assert (injected / "fact.md").read_text(encoding="utf-8") == (
            "# Fact\n\nInjected temporary value.\n"
        )
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        indexed = connection.execute(
            "SELECT content FROM memory_chunks WHERE library_id = ? AND path = ? "
            "ORDER BY start_line",
            (library_id, "section/fact.md"),
        ).fetchall()
        assert all("Injected temporary" not in str(row[0]) for row in indexed)
        assert any("Original authoritative" in str(row[0]) for row in indexed)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_operations WHERE operation_id = ?",
            ("temporary-directory-swap",),
        ).fetchone() == (0,)
        assert (
            connection.execute("SELECT * FROM memory_documents ORDER BY id").fetchall(),
            connection.execute("SELECT * FROM memory_chunks ORDER BY id").fetchall(),
            connection.execute(
                "SELECT * FROM memory_chunk_search ORDER BY chunk_id"
            ).fetchall(),
        ) == before_index
        connection.close()


@pytest.mark.parametrize("unrelated_kind", ["regular", "symlink"])
def test_transient_unrelated_entry_swap_during_mutation_scan_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unrelated_kind: str,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    document = library / "fact.md"
    original = "# Fact\n\nOriginal authoritative value.\n"
    document.write_text(original, encoding="utf-8")
    unrelated = library / ("notes.txt" if unrelated_kind == "regular" else "current")
    replacement = library / "replacement"
    if unrelated_kind == "regular":
        unrelated.write_text("trusted auxiliary data\n", encoding="utf-8")
        replacement.write_text("injected auxiliary data\n", encoding="utf-8")
    else:
        trusted_target = library / "trusted-target"
        injected_target = library / "injected-target"
        trusted_target.mkdir()
        injected_target.mkdir()
        unrelated.symlink_to(trusted_target.name, target_is_directory=True)
        replacement.symlink_to(injected_target.name, target_is_directory=True)
    held = library / "held-unrelated"

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library_id = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library), "kind": "project"},
        ).json()["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "fact.md"},
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        original_scan = PlatformState._scan_bound_library
        calls = 0

        def swapped_scan(state, root, bound, target_library_id):
            nonlocal calls
            calls += 1
            if calls != 2:
                return original_scan(state, root, bound, target_library_id)
            unrelated.rename(held)
            replacement.rename(unrelated)
            try:
                return original_scan(state, root, bound, target_library_id)
            finally:
                unrelated.rename(replacement)
                held.rename(unrelated)

        monkeypatch.setattr(PlatformState, "_scan_bound_library", swapped_scan)
        failed = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "fact.md",
                "content": "# Fact\n\nAttempted platform edit.\n",
                "expected_source_version": loaded["source_version"],
                "operation_id": f"transient-unrelated-{unrelated_kind}",
                "actor_type": "user",
                "source": "race-test",
            },
        )
        assert failed.status_code == 422
        assert document.read_text(encoding="utf-8") == original
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_operations WHERE operation_id = ?",
            (f"transient-unrelated-{unrelated_kind}",),
        ).fetchone() == (0,)
        connection.close()


@pytest.mark.parametrize(
    "repository_mode", ["sidecar", "authorized-head", "authorized-unborn"]
)
def test_git_mapping_trigger_failure_rolls_back_registration_and_history(
    tmp_path: Path,
    repository_mode: str,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / repository_mode
    library.mkdir(parents=True)
    reuse = repository_mode.startswith("authorized")
    if repository_mode != "authorized-unborn":
        (library / "note.md").write_text("# Note\n\nOriginal.\n", encoding="utf-8")
    if reuse:
        subprocess.run(
            ["git", "init", "-b", "main", str(library)],
            check=True,
            capture_output=True,
        )
    if repository_mode == "authorized-head":
        subprocess.run(["git", "-C", str(library), "add", "note.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )

    before_head = None
    before_index = None
    if reuse:
        head = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        before_head = head.stdout.strip() if head.returncode == 0 else None
        index = library / ".git" / "index"
        before_index = index.read_bytes() if index.exists() else None

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        trigger_connection = sqlite3.connect(state_dir / "platform.sqlite3")
        trigger_connection.execute(
            """CREATE TRIGGER fail_memory_git_mapping
               BEFORE INSERT ON library_git_repositories
               BEGIN SELECT RAISE(FAIL, 'injected mapping failure'); END"""
        )
        trigger_connection.commit()
        trigger_connection.close()
        failed = client.post(
            "/api/v1/libraries",
            headers=auth_headers(state_dir),
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": reuse,
            },
        )
        assert failed.status_code == 422
        assert "injected mapping failure" in failed.json()["detail"]

    database = sqlite3.connect(state_dir / "platform.sqlite3")
    assert database.execute("SELECT COUNT(*) FROM memory_libraries").fetchone() == (0,)
    assert database.execute("SELECT COUNT(*) FROM memory_documents").fetchone() == (0,)
    assert database.execute("SELECT COUNT(*) FROM memory_chunks").fetchone() == (0,)
    assert database.execute("SELECT COUNT(*) FROM library_git_repositories").fetchone() == (0,)
    database.close()
    assert not (library / ".personal-agent-memory.json").exists()
    if repository_mode == "sidecar":
        assert not list((state_dir / "git").glob("*.git"))
    else:
        head = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert (head.stdout.strip() if head.returncode == 0 else None) == before_head
        index = library / ".git" / "index"
        assert (index.read_bytes() if index.exists() else None) == before_index


@pytest.mark.parametrize(
    "repository_mode", ["sidecar", "authorized-head", "authorized-unborn"]
)
def test_post_head_initialization_failure_is_fully_compensated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_mode: str,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / repository_mode
    library.mkdir(parents=True)
    reuse = repository_mode.startswith("authorized")
    if repository_mode != "authorized-unborn":
        (library / "note.md").write_text("# Note\n\nOriginal.\n", encoding="utf-8")
    if reuse:
        subprocess.run(
            ["git", "init", "-b", "main", str(library)],
            check=True,
            capture_output=True,
        )
    if repository_mode == "authorized-head":
        subprocess.run(["git", "-C", str(library), "add", "note.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )
    before_head = None
    before_index = None
    if reuse:
        result = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        before_head = result.stdout.strip() if result.returncode == 0 else None
        index = library / ".git" / "index"
        before_index = index.read_bytes() if index.exists() else None
    original_commit = GitRepository.commit

    def fail_after_head_update(repository, contents, message):
        original_commit(repository, contents, message)
        raise GitHistoryError("injected failure after HEAD update")

    monkeypatch.setattr(GitRepository, "commit", fail_after_head_update)
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        failed = client.post(
            "/api/v1/libraries",
            headers=auth_headers(state_dir),
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": reuse,
            },
        )
        assert failed.status_code == 422
        assert "injected failure after HEAD update" in failed.json()["detail"]

    connection = sqlite3.connect(state_dir / "platform.sqlite3")
    assert connection.execute("SELECT COUNT(*) FROM memory_libraries").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_documents").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_chunks").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_chunk_search").fetchone() == (0,)
    assert connection.execute(
        "SELECT COUNT(*) FROM library_git_repositories"
    ).fetchone() == (0,)
    connection.close()
    assert not (library / ".personal-agent-memory.json").exists()
    if repository_mode == "sidecar":
        assert not list((state_dir / "git").glob("*.git"))
    else:
        result = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert (result.stdout.strip() if result.returncode == 0 else None) == before_head
        index = library / ".git" / "index"
        assert (index.read_bytes() if index.exists() else None) == before_index


@pytest.mark.parametrize("reuse_existing_git", [False, True])
def test_database_commit_failure_after_persisting_registration_is_compensated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_existing_git: bool,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    (library / "note.md").write_text("# Note\n\nOriginal.\n", encoding="utf-8")
    before_head = None
    before_index = None
    if reuse_existing_git:
        subprocess.run(
            ["git", "init", "-b", "main", str(library)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(library), "add", "note.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )
        before_head = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        before_index = (library / ".git" / "index").read_bytes()

    real_connect = sqlite3.connect

    class FailAfterRegistrationCommit(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if self.injected:
                return
            table = self.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'library_git_repositories'"
            ).fetchone()
            if table is not None and self.execute(
                "SELECT COUNT(*) FROM library_git_repositories"
            ).fetchone() != (0,):
                self.injected = True
                raise sqlite3.OperationalError(
                    "injected database commit failure after persistence"
                )

    def connect_with_failure(*args, **kwargs):
        kwargs["factory"] = FailAfterRegistrationCommit
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect_with_failure)
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        failed = client.post(
            "/api/v1/libraries",
            headers=auth_headers(state_dir),
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": reuse_existing_git,
            },
        )
        assert failed.status_code == 422
        assert "injected database commit failure" in failed.json()["detail"]

    connection = real_connect(state_dir / "platform.sqlite3")
    assert connection.execute("SELECT COUNT(*) FROM memory_libraries").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_documents").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_chunks").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_chunk_search").fetchone() == (0,)
    assert connection.execute(
        "SELECT COUNT(*) FROM library_git_repositories"
    ).fetchone() == (0,)
    connection.close()
    assert not (library / ".personal-agent-memory.json").exists()
    if reuse_existing_git:
        assert subprocess.run(
            ["git", "-C", str(library), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == before_head
        assert (library / ".git" / "index").read_bytes() == before_index
    else:
        assert not list((state_dir / "git").glob("*.git"))


@pytest.mark.parametrize("reuse_existing_git", [False, True])
@pytest.mark.parametrize("mutation", ["add", "delete", "replace"])
def test_registration_rejects_tree_changes_after_initial_git_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_existing_git: bool,
    mutation: str,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    document = library / "note.md"
    original = "# Note\n\nOriginal registration snapshot.\n"
    document.write_text(original, encoding="utf-8")
    before_head = None
    before_index = None
    if reuse_existing_git:
        subprocess.run(
            ["git", "init", "-b", "main", str(library)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(library), "add", "note.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )
        before_head = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        before_index = (library / ".git" / "index").read_bytes()
    original_initialize = GitRepository.initialize
    raced = False

    def racing_initialize(repository, library_id, documents, verify):
        def race_before_verification():
            nonlocal raced
            if not raced:
                raced = True
                if mutation == "add":
                    (library / "added.md").write_text(
                        "# Added\n\nMust not be silently omitted.\n", encoding="utf-8"
                    )
                elif mutation == "delete":
                    document.unlink()
                else:
                    replacement = library / "replacement.md"
                    replacement.write_text(
                        "# Note\n\nReplacement during registration.\n",
                        encoding="utf-8",
                    )
                    replacement.replace(document)
            assert verify is not None
            verify()

        return original_initialize(
            repository, library_id, documents, race_before_verification
        )

    monkeypatch.setattr(GitRepository, "initialize", racing_initialize)
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        failed = client.post(
            "/api/v1/libraries",
            headers=auth_headers(state_dir),
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": reuse_existing_git,
            },
        )
        assert failed.status_code == 422
        assert "memory library changed during registration" in failed.json()["detail"]

    assert raced
    connection = sqlite3.connect(state_dir / "platform.sqlite3")
    assert connection.execute("SELECT COUNT(*) FROM memory_libraries").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_documents").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_chunks").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_chunk_search").fetchone() == (0,)
    assert connection.execute(
        "SELECT COUNT(*) FROM library_git_repositories"
    ).fetchone() == (0,)
    connection.close()
    assert not (library / ".personal-agent-memory.json").exists()
    if mutation == "add":
        assert (library / "added.md").is_file()
        assert document.read_text(encoding="utf-8") == original
    elif mutation == "delete":
        assert not document.exists()
    else:
        assert document.read_text(encoding="utf-8") == (
            "# Note\n\nReplacement during registration.\n"
        )
    if reuse_existing_git:
        assert subprocess.run(
            ["git", "-C", str(library), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == before_head
        assert (library / ".git" / "index").read_bytes() == before_index
    else:
        assert not list((state_dir / "git").glob("*.git"))


@pytest.mark.parametrize("reuse_existing_git", [False, True])
def test_registration_rejects_markdown_added_while_database_commit_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_existing_git: bool,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    (library / "note.md").write_text("# Note\n\nOriginal.\n", encoding="utf-8")
    before_head = None
    before_index = None
    if reuse_existing_git:
        subprocess.run(
            ["git", "init", "-b", "main", str(library)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(library), "add", "note.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )
        before_head = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        before_index = (library / ".git" / "index").read_bytes()
    real_connect = sqlite3.connect

    class AddMarkdownAfterCommit(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if self.injected:
                return
            table = self.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'library_git_repositories'"
            ).fetchone()
            if table is not None and self.execute(
                "SELECT COUNT(*) FROM library_git_repositories"
            ).fetchone() != (0,):
                self.injected = True
                (library / "late.md").write_text(
                    "# Late\n\nMust not be omitted after commit.\n", encoding="utf-8"
                )

    def connect_with_race(*args, **kwargs):
        kwargs["factory"] = AddMarkdownAfterCommit
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect_with_race)
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        failed = client.post(
            "/api/v1/libraries",
            headers=auth_headers(state_dir),
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": reuse_existing_git,
            },
        )
        assert failed.status_code == 422
        assert "memory library changed during registration" in failed.json()["detail"]

    assert (library / "late.md").read_text(encoding="utf-8") == (
        "# Late\n\nMust not be omitted after commit.\n"
    )
    connection = real_connect(state_dir / "platform.sqlite3")
    assert connection.execute("SELECT COUNT(*) FROM memory_libraries").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_documents").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_chunks").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM memory_chunk_search").fetchone() == (0,)
    assert connection.execute(
        "SELECT COUNT(*) FROM library_git_repositories"
    ).fetchone() == (0,)
    connection.close()
    assert not (library / ".personal-agent-memory.json").exists()
    if reuse_existing_git:
        assert subprocess.run(
            ["git", "-C", str(library), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == before_head
        assert (library / ".git" / "index").read_bytes() == before_index
    else:
        assert not list((state_dir / "git").glob("*.git"))


def test_authorized_edit_rolls_back_ref_index_and_file_when_index_refresh_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "authorized"
    library.mkdir(parents=True)
    document = library / "note.md"
    original = "# Note\n\nOriginal authorized value.\n"
    document.write_text(original, encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main", str(library)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(library), "add", "note.md"], check=True)
    subprocess.run(
        [
            "git", "-C", str(library), "-c", "user.name=Owner", "-c",
            "user.email=owner@example.test", "commit", "-m", "baseline",
        ],
        check=True,
        capture_output=True,
    )

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library_id = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": True,
            },
        ).json()["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "note.md"},
        ).json()
        before_head = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        before_index = (library / ".git" / "index").read_bytes()
        original_run = GitRepository._run
        failed_refresh = False

        def fail_real_index_refresh(self, *arguments, **kwargs):
            nonlocal failed_refresh
            if arguments == ("read-tree", "HEAD") and not failed_refresh:
                failed_refresh = True
                raise GitHistoryError("injected real index refresh failure")
            return original_run(self, *arguments, **kwargs)

        monkeypatch.setattr(GitRepository, "_run", fail_real_index_refresh)
        failed = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "note.md",
                "content": "# Note\n\nUncommitted authorized value.\n",
                "expected_source_version": loaded["source_version"],
                "operation_id": "index-refresh-failure",
                "actor_type": "user",
                "source": "fault-injection",
            },
        )
        assert failed.status_code == 422
        assert document.read_text(encoding="utf-8") == original
        assert subprocess.run(
            ["git", "-C", str(library), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == before_head
        assert (library / ".git" / "index").read_bytes() == before_index
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_operations WHERE operation_id = ?",
            ("index-refresh-failure",),
        ).fetchone() == (0,)
        connection.close()


@pytest.mark.parametrize(
    ("failure_call", "mutation_kind"),
    [(2, "edit"), (3, "edit"), (4, "edit"), (5, "edit"), (5, "restore")],
)
def test_mutation_failure_at_each_post_commit_stage_is_fully_compensated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
    mutation_kind: str,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    document = library / "fact.md"
    original = "# Fact\n\nOriginal searchable value.\n"
    edited_content = "# Fact\n\nEdited searchable value.\n"
    document.write_text(original, encoding="utf-8")

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library_id = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library), "kind": "project"},
        ).json()["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "fact.md"},
        ).json()
        initial_commit = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()[0]["commit"]
        if mutation_kind == "restore":
            succeeded = client.put(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": "fact.md",
                    "content": edited_content,
                    "expected_source_version": loaded["source_version"],
                    "operation_id": "setup-edit",
                    "actor_type": "user",
                    "source": "test",
                },
            )
            assert succeeded.status_code == 200
            loaded = client.get(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                params={"path": "fact.md"},
            ).json()

        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        git_dir = Path(
            connection.execute(
                "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
                (library_id,),
            ).fetchone()[0]
        )

        def database_fingerprint() -> tuple[list[tuple], list[tuple], list[tuple], list[tuple]]:
            return (
                connection.execute(
                    "SELECT * FROM memory_documents WHERE library_id = ? ORDER BY id",
                    (library_id,),
                ).fetchall(),
                connection.execute(
                    "SELECT * FROM memory_chunks WHERE library_id = ? ORDER BY id",
                    (library_id,),
                ).fetchall(),
                connection.execute(
                    "SELECT * FROM memory_chunk_search ORDER BY chunk_id"
                ).fetchall(),
                connection.execute(
                    "SELECT * FROM memory_operations WHERE library_id = ? ORDER BY operation_id",
                    (library_id,),
                ).fetchall(),
            )

        before_database = database_fingerprint()
        before_content = document.read_bytes()
        before_head = subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        original_verify = PlatformState._verify_bound_document
        calls = 0

        def fail_at_stage(root, path, bound, identity) -> None:
            nonlocal calls
            calls += 1
            if calls == failure_call:
                raise MemoryMutationError(
                    f"injected failure at verification {failure_call}"
                )
            original_verify(root, path, bound, identity)

        monkeypatch.setattr(
            PlatformState, "_verify_bound_document", staticmethod(fail_at_stage)
        )
        if mutation_kind == "edit":
            failed = client.put(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": "fact.md",
                    "content": edited_content,
                    "expected_source_version": loaded["source_version"],
                    "operation_id": f"failed-{failure_call}",
                    "actor_type": "user",
                    "source": "fault-injection",
                },
            )
        else:
            failed = client.post(
                f"/api/v1/libraries/{library_id}/history/restore",
                headers=headers,
                json={
                    "path": "fact.md",
                    "commit": initial_commit,
                    "expected_source_version": loaded["source_version"],
                    "operation_id": "failed-restore",
                    "actor_type": "user",
                    "source": "fault-injection",
                },
            )
        assert failed.status_code == 422
        assert document.read_bytes() == before_content
        assert subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == before_head
        assert database_fingerprint() == before_database
        connection.close()


@pytest.mark.parametrize(
    ("external_change", "reuse_existing_git"),
    [
        ("atomic", False),
        ("atomic", True),
        ("inplace", False),
        ("inplace", True),
        ("delete", False),
        ("symlink", False),
        ("directory", False),
    ],
)
def test_failed_mutation_preserves_concurrent_external_file_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_change: str,
    reuse_existing_git: bool,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    document = library / "fact.md"
    original = "# Fact\n\nOriginal indexed value.\n"
    document.write_text(original, encoding="utf-8")
    if reuse_existing_git:
        subprocess.run(
            ["git", "init", "-b", "main", str(library)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(library), "add", "fact.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )
    real_connect = sqlite3.connect
    external_content = f"# Fact\n\nExternal {external_change} value.\n"
    external_target = library / "external-target.md"

    class ChangeFileAfterOperationCommit(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if self.injected:
                return
            table = self.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'memory_operations'"
            ).fetchone()
            if table is None or self.execute(
                "SELECT COUNT(*) FROM memory_operations"
            ).fetchone() == (0,):
                return
            type(self).injected = True
            if external_change == "atomic":
                replacement = library / "external-replacement.md"
                replacement.write_text(external_content, encoding="utf-8")
                replacement.replace(document)
            elif external_change == "inplace":
                document.write_text(external_content, encoding="utf-8")
            elif external_change == "delete":
                document.unlink()
            elif external_change == "symlink":
                external_target.write_text(external_content, encoding="utf-8")
                document.unlink()
                document.symlink_to(external_target.name)
            else:
                document.unlink()
                document.mkdir()

    def connect_with_race(*args, **kwargs):
        kwargs["factory"] = ChangeFileAfterOperationCommit
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect_with_race)
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library_id = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": reuse_existing_git,
            },
        ).json()["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "fact.md"},
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        before = real_connect(state_dir / "platform.sqlite3")
        before_database = (
            before.execute("SELECT * FROM memory_documents ORDER BY id").fetchall(),
            before.execute("SELECT * FROM memory_chunks ORDER BY id").fetchall(),
            before.execute(
                "SELECT * FROM memory_chunk_search ORDER BY chunk_id"
            ).fetchall(),
            before.execute("SELECT * FROM memory_operations ORDER BY operation_id").fetchall(),
        )
        git_dir = Path(
            before.execute(
                "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
                (library_id,),
            ).fetchone()[0]
        )
        before.close()
        before_head = subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        before_index = None
        if reuse_existing_git:
            before_index = (library / ".git" / "index").read_bytes()

        failed = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "fact.md",
                "content": "# Fact\n\nAttempted platform value.\n",
                "expected_source_version": loaded["source_version"],
                "operation_id": f"external-{external_change}-{reuse_existing_git}",
                "actor_type": "user",
                "source": "race-test",
            },
        )
        assert failed.status_code == 422
        assert "compensation conflict" in failed.json()["detail"]
        assert "external" in failed.json()["detail"]
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before

    assert ChangeFileAfterOperationCommit.injected
    if external_change in {"atomic", "inplace"}:
        assert document.read_text(encoding="utf-8") == external_content
    elif external_change == "delete":
        assert not document.exists()
    elif external_change == "symlink":
        assert document.is_symlink()
        assert document.read_text(encoding="utf-8") == external_content
    else:
        assert document.is_dir()
    after = real_connect(state_dir / "platform.sqlite3")
    assert (
        after.execute("SELECT * FROM memory_documents ORDER BY id").fetchall(),
        after.execute("SELECT * FROM memory_chunks ORDER BY id").fetchall(),
        after.execute("SELECT * FROM memory_chunk_search ORDER BY chunk_id").fetchall(),
        after.execute("SELECT * FROM memory_operations ORDER BY operation_id").fetchall(),
    ) == before_database
    after.close()
    assert subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == before_head
    if reuse_existing_git:
        assert (library / ".git" / "index").read_bytes() == before_index


@pytest.mark.parametrize("reuse_existing_git", [False, True])
@pytest.mark.parametrize(
    "failure_mode",
    [
        "first-reread-inplace",
        "exchange-unsupported",
        "temp-fsync-external",
        "temp-fsync-inplace",
        "temp-fsync-directory",
        "temp-fsync-symlink",
        "compensation-temp-fsync-external",
        "compensation-temp-fsync-inplace",
        "exchange-back-external",
        "recovery-replace-forward",
        "recovery-replace-compensation",
        "post-exchange-read",
        "post-exchange-read-external",
        "post-exchange-unlink",
        "post-exchange-unlink-external",
        "post-rename-fsync",
        "post-rename-fsync-external",
        "post-rename-stat",
        "post-rename-stat-external",
    ],
)
def test_early_replacement_failures_compensate_without_overwriting_external_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_existing_git: bool,
    failure_mode: str,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    library.mkdir(parents=True)
    document = library / "fact.md"
    original_content = "# Fact\n\nOriginal indexed value.\n"
    platform_content = "# Fact\n\nAttempted platform value.\n"
    external_content = "# Fact\n\nConcurrent external value.\n"
    later_external_content = "# Fact\n\nLater external value.\n"
    document.write_text(original_content, encoding="utf-8")
    if reuse_existing_git:
        subprocess.run(
            ["git", "init", "-b", "main", str(library)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(library), "add", "fact.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        library_id = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": reuse_existing_git,
            },
        ).json()["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "fact.md"},
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        git_dir = Path(
            connection.execute(
                "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
                (library_id,),
            ).fetchone()[0]
        )
        database_before = (
            connection.execute("SELECT * FROM memory_documents ORDER BY id").fetchall(),
            connection.execute("SELECT * FROM memory_chunks ORDER BY id").fetchall(),
            connection.execute(
                "SELECT * FROM memory_chunk_search ORDER BY chunk_id"
            ).fetchall(),
            connection.execute(
                "SELECT * FROM memory_operations ORDER BY operation_id"
            ).fetchall(),
        )
        head_before = subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        index_before = None
        if reuse_existing_git:
            index_before = (library / ".git" / "index").read_bytes()

        if failure_mode == "first-reread-inplace":
            original_read = PlatformState._read_bound_document
            read_calls = 0

            def change_before_first_reread(bound):
                nonlocal read_calls
                read_calls += 1
                if read_calls == 2:
                    document.write_text(external_content, encoding="utf-8")
                return original_read(bound)

            monkeypatch.setattr(
                PlatformState, "_read_bound_document", staticmethod(change_before_first_reread)
            )
        elif failure_mode == "exchange-unsupported":

            def fail_closed_exchange(*args, **kwargs) -> None:
                raise OSError(errno.ENOSYS, "renameat2 is unavailable")

            monkeypatch.setattr(state_module, "_rename_exchange", fail_closed_exchange)
        elif failure_mode in {
            "temp-fsync-external",
            "temp-fsync-inplace",
            "temp-fsync-directory",
            "temp-fsync-symlink",
            "compensation-temp-fsync-external",
            "compensation-temp-fsync-inplace",
            "exchange-back-external",
            "recovery-replace-forward",
            "recovery-replace-compensation",
        }:
            original_fsync = os.fsync
            regular_fsyncs = 0
            if failure_mode.startswith("compensation-") or failure_mode.endswith(
                "compensation"
            ):
                original_verify = PlatformState._verify_bound_document
                verify_calls = 0

                def fail_after_database_commit(root, path, bound, identity) -> None:
                    nonlocal verify_calls
                    verify_calls += 1
                    if verify_calls == 5:
                        raise MemoryMutationError("injected post-commit failure")
                    original_verify(root, path, bound, identity)

                monkeypatch.setattr(
                    PlatformState,
                    "_verify_bound_document",
                    staticmethod(fail_after_database_commit),
                )
            if failure_mode == "exchange-back-external":
                original_exchange = state_module._rename_exchange
                exchange_calls = 0

                def change_again_before_exchange_back(*args, **kwargs) -> None:
                    nonlocal exchange_calls
                    exchange_calls += 1
                    if exchange_calls == 2:
                        replacement = library / "later-external-replacement.md"
                        replacement.write_text(later_external_content, encoding="utf-8")
                        replacement.replace(document)
                    original_exchange(*args, **kwargs)

                monkeypatch.setattr(
                    state_module, "_rename_exchange", change_again_before_exchange_back
                )
            elif failure_mode.startswith("recovery-replace-"):
                original_exchange = state_module._rename_exchange
                exchange_calls = 0
                rollback_call = (
                    3 if failure_mode.endswith("compensation") else 2
                )

                def replace_recovery_after_exchange_back(*args, **kwargs) -> None:
                    nonlocal exchange_calls
                    exchange_calls += 1
                    original_exchange(*args, **kwargs)
                    if exchange_calls == rollback_call:
                        later = library / "later-external-replacement.md"
                        later.write_text(later_external_content, encoding="utf-8")
                        later.replace(library / args[1])

                monkeypatch.setattr(
                    state_module,
                    "_rename_exchange",
                    replace_recovery_after_exchange_back,
                )

            def replace_after_target_temp_fsync(descriptor: int) -> None:
                nonlocal regular_fsyncs
                original_fsync(descriptor)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    return
                regular_fsyncs += 1
                target_fsync = (
                    2
                    if failure_mode.startswith("compensation-")
                    or failure_mode.endswith("compensation")
                    else 1
                )
                if regular_fsyncs == target_fsync:
                    if failure_mode.endswith("inplace"):
                        document.write_text(external_content, encoding="utf-8")
                    elif failure_mode.endswith("directory"):
                        document.unlink()
                        document.mkdir()
                    elif failure_mode.endswith("symlink"):
                        target = library / "external-target.md"
                        target.write_text(external_content, encoding="utf-8")
                        document.unlink()
                        document.symlink_to(target.name)
                    else:
                        replacement = library / "external-replacement.md"
                        replacement.write_text(external_content, encoding="utf-8")
                        replacement.replace(document)

            monkeypatch.setattr(os, "fsync", replace_after_target_temp_fsync)
        elif "unlink" in failure_mode:
            original_unlink = os.unlink
            failed_unlink = False

            def fail_exchanged_target_unlink(path, *args, **kwargs) -> None:
                nonlocal failed_unlink
                if (
                    not failed_unlink
                    and isinstance(path, str)
                    and path.startswith(".fact.md.")
                    and path.endswith(".tmp")
                    and kwargs.get("dir_fd") is not None
                ):
                    failed_unlink = True
                    if failure_mode.endswith("external"):
                        replacement = library / "external-replacement.md"
                        replacement.write_text(external_content, encoding="utf-8")
                        replacement.replace(document)
                    raise OSError("injected exchanged target unlink failure")
                original_unlink(path, *args, **kwargs)

            monkeypatch.setattr(os, "unlink", fail_exchanged_target_unlink)
        elif "fsync" in failure_mode:
            original_fsync = os.fsync
            failed_fsync = False

            def fail_directory_fsync(descriptor: int) -> None:
                nonlocal failed_fsync
                if not failed_fsync and stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    failed_fsync = True
                    if failure_mode.endswith("external"):
                        replacement = library / "external-replacement.md"
                        replacement.write_text(external_content, encoding="utf-8")
                        replacement.replace(document)
                    raise OSError("injected post-rename directory fsync failure")
                original_fsync(descriptor)

            monkeypatch.setattr(os, "fsync", fail_directory_fsync)
        elif failure_mode.startswith("post-exchange-read"):
            original_open = os.open
            failed_open = False

            def fail_exchanged_target_read(path, flags, *args, **kwargs):
                nonlocal failed_open
                if (
                    not failed_open
                    and isinstance(path, str)
                    and path.startswith(".fact.md.")
                    and path.endswith(".tmp")
                    and kwargs.get("dir_fd") is not None
                    and flags & (os.O_WRONLY | os.O_RDWR) == 0
                ):
                    failed_open = True
                    if failure_mode.endswith("external"):
                        replacement = library / "external-replacement.md"
                        replacement.write_text(external_content, encoding="utf-8")
                        replacement.replace(document)
                    raise OSError("injected exchanged target read failure")
                return original_open(path, flags, *args, **kwargs)

            monkeypatch.setattr(os, "open", fail_exchanged_target_read)
        else:
            original_stat = os.stat
            document_stats = 0

            def fail_post_rename_stat(path, *args, **kwargs):
                nonlocal document_stats
                if (
                    path == "fact.md"
                    and kwargs.get("dir_fd") is not None
                    and kwargs.get("follow_symlinks") is False
                ):
                    document_stats += 1
                    if document_stats == 3:
                        if failure_mode.endswith("external"):
                            replacement = library / "external-replacement.md"
                            replacement.write_text(external_content, encoding="utf-8")
                            replacement.replace(document)
                        raise OSError("injected post-rename stat failure")
                return original_stat(path, *args, **kwargs)

            monkeypatch.setattr(os, "stat", fail_post_rename_stat)

        failed = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "fact.md",
                "content": platform_content,
                "expected_source_version": loaded["source_version"],
                "operation_id": f"early-failure-{failure_mode}-{reuse_existing_git}",
                "actor_type": "user",
                "source": "race-test",
            },
        )
        assert failed.status_code == 422
        preserves_external = (
            failure_mode == "first-reread-inplace"
            or "temp-fsync" in failure_mode
            or failure_mode.endswith("external")
        )
        recovery_files = list(library.glob(".fact.md.*.tmp"))
        if failure_mode.endswith("directory"):
            assert document.is_dir()
        elif failure_mode.endswith("symlink"):
            assert document.is_symlink()
            assert document.read_text(encoding="utf-8") == external_content
        elif failure_mode == "exchange-back-external":
            assert document.read_text(encoding="utf-8") == external_content
            assert any(
                path.read_text(encoding="utf-8") == later_external_content
                for path in recovery_files
            )
        elif failure_mode == "post-exchange-read-external":
            assert document.read_text(encoding="utf-8") == original_content
            assert any(
                path.read_text(encoding="utf-8") == external_content
                for path in recovery_files
            )
        elif failure_mode.startswith("recovery-replace-"):
            assert document.read_text(encoding="utf-8") == external_content
            assert any(
                path.read_text(encoding="utf-8") == later_external_content
                for path in recovery_files
            )
        else:
            assert document.read_text(encoding="utf-8") == (
                external_content if preserves_external else original_content
            )
        if preserves_external:
            assert "compensation conflict" in failed.json()["detail"]
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        assert subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == head_before
        if reuse_existing_git:
            assert (library / ".git" / "index").read_bytes() == index_before
        assert (
            connection.execute("SELECT * FROM memory_documents ORDER BY id").fetchall(),
            connection.execute("SELECT * FROM memory_chunks ORDER BY id").fetchall(),
            connection.execute(
                "SELECT * FROM memory_chunk_search ORDER BY chunk_id"
            ).fetchall(),
            connection.execute(
                "SELECT * FROM memory_operations ORDER BY operation_id"
            ).fetchall(),
        ) == database_before
        if failure_mode.startswith("recovery-replace-"):
            assert "compensation conflict" in failed.json()["detail"]
            assert ".fact.md." in failed.json()["detail"]
            import_external_changes(client, headers, library_id, library)
            assert [
                item["path"]
                for item in client.get(
                    f"/api/v1/libraries/{library_id}/documents", headers=headers
                ).json()
            ] == ["fact.md"]
            rescanned_document = client.get(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                params={"path": "fact.md"},
            ).json()
            retried = client.put(
                f"/api/v1/libraries/{library_id}/document",
                headers=headers,
                json={
                    "path": "fact.md",
                    "content": "# Fact\n\nFollow-up platform value.\n",
                    "expected_source_version": rescanned_document["source_version"],
                    "operation_id": f"retry-{failure_mode}-{reuse_existing_git}",
                    "actor_type": "user",
                    "source": "race-test",
                },
            )
            assert retried.status_code == 200
            committed_paths = subprocess.run(
                ["git", f"--git-dir={git_dir}", "ls-tree", "-r", "--name-only", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            assert all(not path.startswith(".fact.md.") for path in committed_paths)
            assert any(
                path.read_text(encoding="utf-8") == later_external_content
                for path in recovery_files
            )
        connection.close()


@pytest.mark.parametrize("existing_manifest", [False, True])
def test_authorized_initialization_failure_preserves_manifest_and_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_manifest: bool,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "authorized"
    library.mkdir(parents=True)
    (library / "note.md").write_text("# Note\n\nOriginal.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main", str(library)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(library), "add", "note.md"], check=True)
    subprocess.run(
        [
            "git", "-C", str(library), "-c", "user.name=Owner", "-c",
            "user.email=owner@example.test", "commit", "-m", "baseline",
        ],
        check=True,
        capture_output=True,
    )
    manifest = library / ".personal-agent-memory.json"
    if existing_manifest:
        manifest.write_text("existing user manifest\n", encoding="utf-8")
    before_manifest = manifest.read_bytes() if manifest.exists() else None
    before_head = subprocess.run(
        ["git", "-C", str(library), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    before_index = (library / ".git" / "index").read_bytes()

    def fail_commit(self, contents, message):
        raise GitHistoryError("injected initialization failure")

    monkeypatch.setattr(GitRepository, "commit", fail_commit)
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        response = client.post(
            "/api/v1/libraries",
            headers=auth_headers(state_dir),
            json={
                "path": str(library),
                "kind": "project",
                "reuse_existing_git": True,
            },
        )
        assert response.status_code == 422
    assert (manifest.read_bytes() if manifest.exists() else None) == before_manifest
    assert subprocess.run(
        ["git", "-C", str(library), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == before_head
    assert (library / ".git" / "index").read_bytes() == before_index


def test_history_initialization_hashes_dirfd_verified_bytes_during_parent_symlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    section = library / "section"
    section.mkdir(parents=True)
    (section / "note.md").write_text("SAFE INITIAL\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_text("OUTSIDE INITIAL\n", encoding="utf-8")
    held = library / "held-section"
    original_initialize = GitRepository.initialize

    def racing_initialize(
        repository: GitRepository,
        library_id: str,
        documents: dict[str, bytes],
        verify,
    ) -> str:
        section.rename(held)
        section.symlink_to(outside, target_is_directory=True)
        return original_initialize(repository, library_id, documents, verify)

    monkeypatch.setattr(GitRepository, "initialize", racing_initialize)
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        response = client.post(
            "/api/v1/libraries", headers=headers,
            json={"path": str(library), "kind": "project"},
        )
        assert response.status_code == 422
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        assert connection.execute("SELECT COUNT(*) FROM memory_libraries").fetchone() == (0,)
        connection.close()
        assert (outside / "note.md").read_text(encoding="utf-8") == "OUTSIDE INITIAL\n"
    section.unlink()
    held.rename(section)


@pytest.mark.parametrize("reuse_existing_git", [False, True])
def test_edit_hashes_frozen_bytes_during_parent_symlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reuse_existing_git: bool
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library = library_root / "project"
    section = library / "section"
    section.mkdir(parents=True)
    document = section / "note.md"
    document.write_text("# Fact\n\nSAFE BEFORE\n", encoding="utf-8")
    if reuse_existing_git:
        subprocess.run(["git", "init", "-b", "main", str(library)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(library), "add", "section/note.md"], check=True)
        subprocess.run(
            [
                "git", "-C", str(library), "-c", "user.name=Owner", "-c",
                "user.email=owner@example.test", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_text("# Fact\n\nOUTSIDE ATTACK\n", encoding="utf-8")
    held = library / "held-section"

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        headers = auth_headers(state_dir)
        registration = client.post(
            "/api/v1/libraries", headers=headers,
            json={
                "path": str(library), "kind": "project",
                "reuse_existing_git": reuse_existing_git,
            },
        )
        assert registration.status_code == 201
        library_id = registration.json()["id"]
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document", headers=headers,
            params={"path": "section/note.md"},
        ).json()
        original_commit = GitRepository.commit

        def racing_commit(
            repository: GitRepository,
            contents: dict[str, bytes | None],
            message: str,
        ) -> str:
            section.rename(held)
            section.symlink_to(outside, target_is_directory=True)
            return original_commit(repository, contents, message)

        monkeypatch.setattr(GitRepository, "commit", racing_commit)
        edited = client.put(
            f"/api/v1/libraries/{library_id}/document", headers=headers,
            json={
                "path": "section/note.md", "content": "# Fact\n\nSAFE EDITED\n",
                "expected_source_version": loaded["source_version"],
                "operation_id": f"race-{reuse_existing_git}",
                "actor_type": "user", "source": "race-test",
            },
        )
        assert edited.status_code == 422
        connection = sqlite3.connect(state_dir / "platform.sqlite3")
        git_dir = connection.execute(
            "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
            (library_id,),
        ).fetchone()[0]
        committed = subprocess.run(
            ["git", f"--git-dir={git_dir}", "show", "HEAD:section/note.md"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert committed == "# Fact\n\nSAFE BEFORE\n"
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_operations WHERE operation_id = ?",
            (f"race-{reuse_existing_git}",),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT source_version FROM memory_documents WHERE library_id = ? AND path = ?",
            (library_id, "section/note.md"),
        ).fetchone() == (loaded["source_version"],)
        connection.close()
        assert (held / "note.md").read_text(encoding="utf-8") == "# Fact\n\nSAFE BEFORE\n"
        assert (outside / "note.md").read_text(encoding="utf-8") == (
            "# Fact\n\nOUTSIDE ATTACK\n"
        )
        section.unlink()
        held.rename(section)


def test_web_history_selection_is_immutable_and_builds_a_bound_restore_request(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the executable Web history regression")
    script_path = (
        Path(__file__).parents[1]
        / "src"
        / "personal_agent_memory"
        / "static"
        / "editor_history.js"
    )
    program = """
const api = require(process.argv[1]);
const document = {path: 'first.md', source_version: 'version-one'};
const store = api.selectionStore();
const selection = store.select('library-one', document, 'abc123');
document.path = 'second.md';
document.source_version = 'version-two';
const request = api.request(selection, 'operation-one');
store.clear();
console.log(JSON.stringify({selection, request, afterClear: store.current()}));
"""
    result = subprocess.run(
        [node, "-e", program, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["selection"] == {
        "libraryId": "library-one",
        "path": "first.md",
        "commit": "abc123",
        "expectedSourceVersion": "version-one",
    }
    assert payload["request"] == {
        "url": "/api/v1/libraries/library-one/history/restore",
        "body": {
            "path": "first.md",
            "commit": "abc123",
            "expected_source_version": "version-one",
            "operation_id": "operation-one",
            "actor_type": "user",
            "source": "web-history",
        },
    }
    assert payload["afterClear"] is None


def test_web_context_store_rejects_stale_responses_and_freezes_save_payload(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the executable Web context regression")
    script_path = (
        Path(__file__).parents[1]
        / "src"
        / "personal_agent_memory"
        / "static"
        / "editor_history.js"
    )
    program = """
const api = require(process.argv[1]);
const store = api.contextStore();
store.selectLibrary('library-one');
const firstLoad = store.requestDocument('library-one', 'first.md');
const secondLoad = store.requestDocument('library-one', 'second.md');
const staleLoad = store.acceptDocument(firstLoad, {
  path: 'first.md', source_version: 'version-one', content: 'first content'
});
store.acceptDocument(secondLoad, {
  path: 'second.md', source_version: 'version-two', content: 'second content'
});
const edited = store.edit('frozen edit');
const preview = api.previewRequest(edited);
const save = api.saveRequest(edited, 'operation-one');
store.edit('newer edit');
const staleAfterEdit = store.isCurrent(edited);
store.selectLibrary('library-two');
const staleAfterLibraryChange = store.isCurrent(edited);
console.log(JSON.stringify({
  staleLoad, preview, save, staleAfterEdit, staleAfterLibraryChange,
  current: store.snapshot()
}));
"""
    result = subprocess.run(
        [node, "-e", program, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["staleLoad"] is None
    assert payload["staleAfterEdit"] is False
    assert payload["staleAfterLibraryChange"] is False
    assert payload["preview"]["body"] == {
        "path": "second.md",
        "content": "frozen edit",
        "expected_source_version": "version-two",
    }
    assert payload["save"] == {
        "url": "/api/v1/libraries/library-one/document",
        "body": {
            "path": "second.md",
            "content": "frozen edit",
            "expected_source_version": "version-two",
            "operation_id": "operation-one",
            "actor_type": "user",
            "source": "web",
        },
    }
    assert payload["current"]["libraryId"] == "library-two"


def test_management_ui_state_machines_reject_stale_async_results() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the executable management UI regression")
    html_path = (
        Path(__file__).parents[1]
        / "src"
        / "personal_agent_memory"
        / "static"
        / "index.html"
    )
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function section(startMarker, endMarker) {
  const start = html.indexOf(startMarker);
  const end = html.indexOf(endMarker, start);
  if (start < 0 || end < 0) throw new Error(`missing function section: ${startMarker}`);
  return html.slice(start, end);
}

async function candidateFailureState() {
  const candidateNodes = new Map();
  for (const id of [
    'candidate-title', 'candidate-sources', 'candidate-body', 'candidate-operator',
    'candidate-reason', 'candidate-resolution', 'candidate-effective',
    'candidate-condition', 'candidate-pin', 'candidate-save', 'candidate-approve',
    'candidate-resolve', 'candidate-reject', 'candidate-message',
    'candidate-governance', 'candidate-current', 'candidate-conflict-diff'
  ]) candidateNodes.set(`#${id}`, {
    textContent: 'stale', value: 'stale', hidden: false, disabled: false
  });
  const candidateDetail = {querySelector: (selector) => candidateNodes.get(selector)};
  const generations = {candidateGovernance: 0};
  let activeCandidate = null;
  const display = (value) => value;
  const setMessage = (node, text, kind) => { node.textContent = text; node.kind = kind; };
  const describeError = (_failure, context) => `${context}：网络连接失败，请稍后重试。`;
  const request = async () => ({ok: false, json: async () => ({detail: 'network'})});
  eval(section(
    'function clearCandidateGovernance()', '\n\n        async function candidateAction'
  ));
  await showCandidate({
    id: 'candidate-one', status: 'pending', suggested_type: 'decision',
    creator: 'test', created_at: 'now', source_references: [], body: 'body'
  });
  return Object.fromEntries([...candidateNodes].map(([key, value]) => [key, value]));
}

const bindingIntents = new Map();
const key = 'test-key';
const sent = [];
const deferred = [];
const request = async (_url, options) => {
  sent.push(JSON.parse(options.body).library_id);
  return await new Promise((resolve) => deferred.push(resolve));
};
const messages = [];
const bindingForm = {querySelector: () => ({})};
const setMessage = (_node, text) => messages.push(text);
const describeError = () => 'failed';
let refreshCount = 0;
const refreshBindings = async () => { refreshCount += 1; };
eval(section(
  'async function queueBindingUpdate(bindingId, select)',
  '\n\n        form.addEventListener'
));
async function run() {
  const candidateState = await candidateFailureState();
  const select = {value: 'library-b', disabled: false};
  const first = queueBindingUpdate('binding-one', select);
  select.value = 'library-c';
  const second = queueBindingUpdate('binding-one', select);
  await Promise.resolve();
  deferred.shift()({ok: true});
  await new Promise(setImmediate);
  deferred.shift()({ok: true});
  await Promise.all([first, second]);
  const reconciliationNodes = new Map();
  for (const id of [
    'reconciliation-title', 'reconciliation-status', 'reconciliation-base',
    'reconciliation-platform', 'reconciliation-external', 'reconciliation-final',
    'reconciliation-path', 'reconciliation-path-field', 'reconciliation-import',
    'reconciliation-restore', 'reconciliation-message'
  ]) reconciliationNodes.set(`#${id}`, {
    textContent: 'stale', value: 'stale', hidden: false, disabled: false,
    replaceChildren() { this.children = []; }
  });
  const reconciliationDetail = {querySelector: (selector) => reconciliationNodes.get(selector)};
  const reconciliationList = {children: ['stale'], replaceChildren() { this.children = []; }};
  const generations = {reconciliation: 4};
  const editorLibrary = {value: 'library-new'};
  let activeReconciliation = {id: 'old-change'};
  const setMessage = (node, text) => { node.textContent = text; };
  eval(section(
    'function clearReconciliation(invalidate = true)',
    '\n\n        function showReconciliation'
  ));
  const staleContext = {libraryId: 'library-old', generation: 4};
  clearReconciliation();
  console.log(JSON.stringify({
    sent, refreshCount, selectDisabled: select.disabled, messages,
    candidateState,
    reconciliation: {
      activeReconciliation,
      generation: generations.reconciliation,
      staleAccepted: reconciliationContextIsCurrent(staleContext),
      title: reconciliationNodes.get('#reconciliation-title').textContent,
      importDisabled: reconciliationNodes.get('#reconciliation-import').disabled,
      listChildren: reconciliationList.children,
    },
  }));
}
run();
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["sent"] == ["library-b", "library-c"]
    assert payload["refreshCount"] == 1
    assert payload["selectDisabled"] is False
    assert payload["candidateState"]["#candidate-governance"]["textContent"] == ""
    assert payload["candidateState"]["#candidate-current"]["hidden"] is True
    assert payload["candidateState"]["#candidate-conflict-diff"]["hidden"] is True
    assert payload["candidateState"]["#candidate-resolve"]["disabled"] is True
    assert payload["candidateState"]["#candidate-message"]["kind"] == "error"
    assert "候选治理信息加载失败" in payload["candidateState"]["#candidate-message"][
        "textContent"
    ]
    assert payload["reconciliation"] == {
        "activeReconciliation": None,
        "generation": 5,
        "staleAccepted": False,
        "title": "请选择带外变更",
        "importDisabled": True,
        "listChildren": [],
    }


def test_reconciliation_success_feedback_survives_refresh_but_not_library_switch() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the reconciliation feedback regression")
    html_path = (
        Path(__file__).parents[1]
        / "src"
        / "personal_agent_memory"
        / "static"
        / "index.html"
    )
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('async function resolveReconciliation(action)');
const end = html.indexOf('\n\n        async function refreshCandidates', start);
const implementation = html.slice(start, end);

async function scenario(switchLibrary) {
  const context = {libraryId: 'library-one', generation: 7};
  const editorLibrary = {value: 'library-one'};
  const generations = {reconciliation: 7};
  let activeReconciliation = {
    id: 'change-one', base_path: 'note.md', status: 'pending',
    external_withheld: false, uiContext: context
  };
  const nodes = {
    '#reconciliation-import': {disabled: false, dataset: {}},
    '#reconciliation-restore': {disabled: false, dataset: {}},
    '#reconciliation-final': {value: ''},
    '#reconciliation-path': {value: ''},
    '#reconciliation-message': {textContent: '', kind: ''},
  };
  const reconciliationDetail = {querySelector: (selector) => nodes[selector]};
  const reconciliationContextIsCurrent = (candidate) => (
    candidate.libraryId === editorLibrary.value
      && candidate.generation === generations.reconciliation
  );
  const confirmAction = async () => true;
  const performOnce = async (_name, _button, _label, action) => await action();
  const PamHistory = {reconciliationRequest: (libraryId, change) => ({
    url: `/libraries/${libraryId}/changes/${change.id}`,
    body: {},
  })};
  const key = 'test-key';
  const request = async () => ({ok: true});
  const describeError = () => 'failed';
  const setMessage = (node, text, kind) => {
    node.textContent = text;
    node.kind = kind;
  };
  const refreshDocuments = async () => {
    nodes['#reconciliation-message'].textContent = '';
    if (switchLibrary) editorLibrary.value = 'library-two';
  };
  eval(implementation);
  await resolveReconciliation('restore');
  return nodes['#reconciliation-message'];
}

Promise.all([scenario(false), scenario(true)]).then(([same, switched]) => {
  console.log(JSON.stringify({same, switched}));
});
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["same"] == {"textContent": "已恢复平台版本。", "kind": "success"}
    assert payload["switched"]["textContent"] == ""


def test_management_ui_renders_known_backend_states_as_chinese_badges() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the status badge regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('const labels = {');
const end = html.indexOf('const displaySensitiveSummary', start);
const document = {createElement: () => ({className: '', dataset: {}, textContent: ''})};
let result;
eval(html.slice(start, end) + `
  result = Object.fromEntries([
    'not_started', 'scanning', 'building', 'missing', 'replaced',
    'available', 'ready', 'error', 'reusable_experience', 'external_reference'
  ].map((value) => {
    const badge = createStatusBadge(value);
    return [value, {
      text: badge.textContent,
      className: badge.className,
      raw: badge.dataset.status
    }];
  }));
  result.degradation = Object.fromEntries([
    'vector_index_unavailable', 'embedding_unavailable', 'reranker_unavailable',
    'tokenizer_fallback'
  ].map((value) => [value, degradationLabels[value]]));
  result.searchStatus = formatSearchStatus({
    status: 'bound', results: [], graph_index_status: 'ready', degraded: true,
    degradation: [
      'tokenizer_fallback', 'vector_index_unavailable',
      'embedding_unavailable', 'reranker_unavailable'
    ]
  });
`);
console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert {
        key: value["text"]
        for key, value in payload.items()
        if key not in {"degradation", "searchStatus"}
    } == {
        "not_started": "尚未开始",
        "scanning": "扫描中",
        "building": "构建中",
        "missing": "路径缺失",
        "replaced": "路径已替换",
        "available": "可用",
        "ready": "就绪",
        "error": "错误",
        "reusable_experience": "可复用经验",
        "external_reference": "外部参考",
    }
    assert payload["degradation"] == {
        "vector_index_unavailable": "向量索引不可用",
        "embedding_unavailable": "嵌入服务不可用",
        "reranker_unavailable": "重排序服务不可用",
        "tokenizer_fallback": "分词器回退",
    }
    assert payload["searchStatus"] == {
        "text": (
            "没有找到匹配的记忆 · 图索引就绪 · 已降级：分词器回退、"
            "向量索引不可用、嵌入服务不可用、重排序服务不可用。"
        ),
        "kind": "error",
    }
    assert payload["available"]["className"] == "badge ok"
    assert payload["scanning"]["className"] == "badge warn"
    assert payload["missing"]["className"] == "badge bad"
    assert all(
        value["raw"] == key
        for key, value in payload.items()
        if key not in {"degradation", "searchStatus"}
    )


def test_management_ui_connection_status_tracks_transport_and_auth_only() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the connection status regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf("const connectionStatus = topbar.querySelector('#connection-status')");
const end = html.indexOf('async function performOnce', start);
const connection = {className: '', textContent: ''};
const topbar = {querySelector: () => connection};
const pending = [];
const window = {
  fetch: (name) => new Promise(
    (resolve, reject) => pending.push({name, resolve, reject})
  )
};
eval(html.slice(start, end) + '\nglobalThis.requestUnderTest = request;');

async function run() {
  const older = requestUnderTest('older-network');
  const newer = requestUnderTest('newer-success');
  pending.find((item) => item.name === 'newer-success').resolve({ok: true, status: 200});
  await newer;
  pending.find((item) => item.name === 'older-network').reject(new Error('offline'));
  await older;
  const afterStaleFailure = {...connection};

  const business = requestUnderTest('business');
  pending.find((item) => item.name === 'business').resolve({ok: false, status: 409});
  await business;
  const afterBusinessFailure = {...connection};

  const unauthorized = requestUnderTest('unauthorized');
  pending.find((item) => item.name === 'unauthorized').resolve({ok: false, status: 401});
  await unauthorized;
  const afterUnauthorized = {...connection};

  const recovered = requestUnderTest('recovered');
  pending.find((item) => item.name === 'recovered').resolve({ok: true, status: 200});
  await recovered;
  const afterRecovery = {...connection};

  const olderAfterBusiness = requestUnderTest('older-after-business');
  const newerBusiness = requestUnderTest('newer-business');
  pending.find((item) => item.name === 'newer-business').resolve({ok: false, status: 409});
  await newerBusiness;
  pending.find((item) => item.name === 'older-after-business').reject(new Error('offline'));
  await olderAfterBusiness;
  const afterBusinessSupersedesTransport = {...connection};

  const olderAfterValidation = requestUnderTest('older-after-validation');
  const newerValidation = requestUnderTest('newer-validation');
  pending.find((item) => item.name === 'newer-validation').resolve({ok: false, status: 422});
  await newerValidation;
  pending.find((item) => item.name === 'older-after-validation').reject(new Error('offline'));
  await olderAfterValidation;
  const afterValidationSupersedesTransport = {...connection};

  const failed = requestUnderTest('failed');
  pending.find((item) => item.name === 'failed').reject(new Error('offline'));
  const fallback = await failed;
  console.log(JSON.stringify({
    afterStaleFailure,
    afterBusinessFailure,
    afterUnauthorized,
    afterRecovery,
    afterBusinessSupersedesTransport,
    afterValidationSupersedesTransport,
    failed: {...connection},
    networkError: fallback.networkError
  }));
}
run();
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["afterStaleFailure"] == {
        "className": "connection ok",
        "textContent": "服务已连接",
    }
    assert payload["afterBusinessFailure"] == payload["afterStaleFailure"]
    assert payload["afterUnauthorized"] == {
        "className": "connection warn",
        "textContent": "认证已失效",
    }
    assert payload["afterRecovery"] == payload["afterStaleFailure"]
    assert payload["afterBusinessSupersedesTransport"] == payload["afterRecovery"]
    assert payload["afterValidationSupersedesTransport"] == payload["afterRecovery"]
    assert payload["failed"] == {
        "className": "connection bad",
        "textContent": "服务连接中断",
    }
    assert payload["networkError"] is True


def test_management_ui_success_feedback_survives_refresh_with_context_guards() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the success feedback regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function section(startMarker, endMarker) {
  const start = html.indexOf(startMarker);
  const end = html.indexOf(endMarker, start);
  if (start < 0 || end < 0) throw new Error(`missing section ${startMarker}`);
  return html.slice(start, end);
}
const setMessage = (node, text, kind) => { node.textContent = text; node.kind = kind; };
const performOnce = async (_name, _button, _label, action) => await action();
const describeError = () => 'failed';
const key = 'test-key';

async function candidateScenario(action, switchCandidate) {
  const message = {textContent: '', kind: ''};
  const nodes = new Map([
    ['#candidate-message', message], ['#candidate-save', {dataset: {}}],
    ['#candidate-approve', {dataset: {}}], ['#candidate-reject', {dataset: {}}],
    ['#candidate-operator', {value: 'operator'}], ['#candidate-reason', {value: 'reason'}],
    ['#candidate-body', {value: 'body'}],
  ]);
  const candidateDetail = {querySelector: (selector) => nodes.get(selector)};
  let candidateSelectionGeneration = 4;
  let activeCandidate = {id: 'candidate-one', status: 'pending'};
  const request = async () => ({
    ok: true,
    json: async () => ({
      id: 'candidate-one',
      status: action === 'edit' ? 'pending' : `${action}d`
    })
  });
  const refreshCandidates = async () => {
    message.textContent = '';
    if (switchCandidate) {
      ++candidateSelectionGeneration;
      activeCandidate = {id: 'candidate-two'};
    }
    else if (action !== 'edit') activeCandidate = null;
  };
  const refreshDocuments = async () => { message.textContent = ''; };
  eval(section(
    'async function candidateAction(action)',
    '\n\n        async function resolveCandidate'
  ));
  await candidateAction(action);
  return {...message};
}

async function pinScenario(switchCandidate) {
  const message = {textContent: '', kind: ''};
  const pin = {dataset: {}};
  const candidateDetail = {
    querySelector: (selector) => selector === '#candidate-pin' ? pin : message
  };
  let candidateSelectionGeneration = 2;
  let activeCandidate = {id: 'candidate-one'};
  const request = async () => ({ok: true});
  const refreshCandidates = async () => {
    message.textContent = '';
    if (switchCandidate) {
      ++candidateSelectionGeneration;
      activeCandidate = {id: 'candidate-two'};
    }
  };
  eval(section(
    'async function pinCandidate()',
    '\n\n        async function refreshSensitiveQuarantine'
  ));
  await pinCandidate();
  return {...message};
}

async function forgetScenario(switchLibrary) {
  const message = {textContent: '', kind: ''};
  const context = {token: 1, libraryId: 'library-one', path: 'note.md', sourceVersion: 'v1'};
  let current = context;
  const editorContext = {snapshot: () => current, isCurrent: (candidate) => candidate === current};
  const editorLibrary = {value: 'library-one'};
  const editor = {
    querySelector: (selector) => (
      selector === '#editor-message' ? message : {dataset: {}}
    )
  };
  const confirmAction = async () => true;
  const request = async () => ({ok: true});
  const refreshDocuments = async () => {
    message.textContent = '';
    if (switchLibrary) {
      editorLibrary.value = 'library-two';
      current = {libraryId: 'library-two', path: null};
    }
    else current = {libraryId: 'library-one', path: null};
  };
  const crypto = {randomUUID: () => 'operation'};
  let deletionPreviewVersion = 'v1';
  let deletionPreviewToken = 'preview';
  eval(section('async function forgetDocument()', '\n\n        async function refreshForgotten'));
  await forgetDocument();
  return {...message};
}

Promise.all([
  candidateScenario('edit', false), candidateScenario('approve', false),
  candidateScenario('reject', false), candidateScenario('edit', true),
  pinScenario(false), pinScenario(true), forgetScenario(false), forgetScenario(true)
]).then(([edit, approve, reject, staleCandidate, pin, stalePin, forget, staleForget]) => {
  console.log(JSON.stringify({
    edit, approve, reject, staleCandidate, pin, stalePin, forget, staleForget
  }));
});
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["edit"]["textContent"] == "候选记忆已保存。"
    assert payload["approve"]["textContent"] == "候选记忆已批准。"
    assert payload["reject"]["textContent"] == "候选记忆已拒绝。"
    assert payload["pin"]["textContent"] == "候选记忆已置顶。"
    assert payload["forget"]["textContent"] == "记忆已遗忘。"
    for key in ("edit", "approve", "reject", "pin", "forget"):
        assert payload[key]["kind"] == "success"
    assert payload["staleCandidate"]["textContent"] == ""
    assert payload["stalePin"]["textContent"] == ""
    assert payload["staleForget"]["textContent"] == ""


def test_management_ui_candidate_busy_state_stays_with_its_selection() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the candidate busy-state regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function section(startMarker, endMarker) {
  const start = html.indexOf(startMarker);
  const end = html.indexOf(endMarker, start);
  if (start < 0 || end < 0) throw new Error(`missing section ${startMarker}`);
  return html.slice(start, end);
}
const makeButton = (textContent) => ({
  dataset: {}, disabled: false, textContent,
  closest: () => null
});
const buttons = {
  save: makeButton('保存编辑'), approve: makeButton('批准'),
  reject: makeButton('拒绝'), resolve: makeButton('提交裁决'),
  pin: makeButton('置顶')
};
const fields = {
  '#candidate-save': buttons.save, '#candidate-approve': buttons.approve,
  '#candidate-reject': buttons.reject, '#candidate-resolve': buttons.resolve,
  '#candidate-pin': buttons.pin,
  '#candidate-operator': {value: 'operator'}, '#candidate-reason': {value: 'reason'},
  '#candidate-body': {value: 'body'}, '#candidate-resolution': {value: 'keep'},
  '#candidate-effective': {value: ''}, '#candidate-condition': {value: ''},
  '#candidate-message': {textContent: ''}
};
const candidateDetail = {querySelector: (selector) => fields[selector]};
const pending = new Set();
const buttonOperations = new WeakMap();
const setBusy = (button, busy, label) => {
  if (!button.dataset.label) button.dataset.label = button.textContent;
  button.disabled = busy;
  button.textContent = busy ? label : button.dataset.label;
};
const setMessage = () => {};
const describeError = () => 'failed';
const key = 'test-key';
const crypto = {randomUUID: () => 'operation'};
let candidateSelectionGeneration = 1;
let activeCandidate = {id: 'candidate-a', status: 'pending'};
const requests = [];
const request = (url) => new Promise((resolve) => requests.push({url, resolve}));
const refreshCandidates = async () => {};
const refreshDocuments = async () => {};
eval(section('async function performOnce(', '\n        function openView'));
eval(section('async function candidateAction(action)', '\n\n        async function pinCandidate'));

async function scenario(status) {
  activeCandidate = {id: 'candidate-a', status: 'pending'};
  for (const button of Object.values(buttons)) button.disabled = false;
  const requestStart = requests.length;
  const first = candidateAction('approve');
  await Promise.resolve();
  ++candidateSelectionGeneration;
  activeCandidate = {id: 'candidate-b', status};
  for (const button of Object.values(buttons)) button.disabled = true;
  requests[requestStart].resolve({
    ok: true,
    json: async () => ({id: 'candidate-a', status: 'approved'})
  });
  await first;
  await candidateAction('edit');
  await candidateAction('approve');
  await candidateAction('reject');
  await resolveCandidate();
  return {
    requestCount: requests.length - requestStart,
    disabled: Object.fromEntries(
      Object.entries(buttons).map(([name, button]) => [name, button.disabled])
    ),
    labels: Object.fromEntries(
      Object.entries(buttons).map(([name, button]) => [name, button.textContent])
    )
  };
}
async function run() {
  console.log(JSON.stringify({
    approved: await scenario('approved'),
    rejected: await scenario('rejected')
  }));
}
run();
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    for status in ("approved", "rejected"):
        assert payload[status]["requestCount"] == 1
        assert payload[status]["disabled"] == {
            "save": True,
            "approve": True,
            "reject": True,
            "resolve": True,
            "pin": True,
        }
        assert payload[status]["labels"]["approve"] == "批准"


def test_management_ui_failure_feedback_stays_with_original_candidate_and_document() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the stale failure regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function section(startMarker, endMarker) {
  const start = html.indexOf(startMarker);
  const end = html.indexOf(endMarker, start);
  return html.slice(start, end);
}
const setMessage = (node, text, kind) => { node.textContent = text; node.kind = kind; };
const performOnce = async (_name, _button, _label, action) => await action();
const describeError = (_failure, context) => `${context}：测试失败`;
const key = 'test-key';
const crypto = {randomUUID: () => 'operation'};

async function candidateFailure(switchCandidate) {
  const message = {textContent: '', kind: ''};
  const nodes = new Map([
    ['#candidate-message', message], ['#candidate-save', {dataset: {}}],
    ['#candidate-operator', {value: 'operator'}], ['#candidate-reason', {value: 'reason'}],
    ['#candidate-body', {value: 'body'}],
  ]);
  const candidateDetail = {querySelector: (selector) => nodes.get(selector)};
  let candidateSelectionGeneration = 1;
  let activeCandidate = {id: 'candidate-one', status: 'pending'};
  let finish;
  const request = async () => await new Promise((resolve) => { finish = resolve; });
  const refreshCandidates = async () => {};
  const refreshDocuments = async () => {};
  eval(section(
    'async function candidateAction(action)',
    '\n\n        async function resolveCandidate'
  ));
  const action = candidateAction('edit');
  await Promise.resolve();
  if (switchCandidate) {
    ++candidateSelectionGeneration;
    activeCandidate = {id: 'candidate-two', status: 'pending'};
  }
  finish({ok: false, json: async () => ({detail: 'failed'})});
  await action;
  return message;
}

async function pinFailure(switchCandidate) {
  const message = {textContent: '', kind: ''};
  const pin = {dataset: {}};
  const candidateDetail = {
    querySelector: (selector) => selector === '#candidate-pin' ? pin : message
  };
  let candidateSelectionGeneration = 1;
  let activeCandidate = {id: 'candidate-one'};
  let finish;
  const request = async () => await new Promise((resolve) => { finish = resolve; });
  const refreshCandidates = async () => {};
  eval(section(
    'async function pinCandidate()',
    '\n\n        async function refreshSensitiveQuarantine'
  ));
  const action = pinCandidate();
  await Promise.resolve();
  if (switchCandidate) {
    ++candidateSelectionGeneration;
    activeCandidate = {id: 'candidate-two'};
  }
  finish({ok: false});
  await action;
  return message;
}

async function forgetFailure(switchDocument) {
  const message = {textContent: '', kind: ''};
  const original = {token: 1, libraryId: 'library-one', path: 'note.md', sourceVersion: 'v1'};
  let current = original;
  const editorContext = {snapshot: () => current, isCurrent: (candidate) => candidate === current};
  const editorLibrary = {value: 'library-one'};
  const editor = {
    querySelector: (selector) => selector === '#editor-message' ? message : {dataset: {}}
  };
  const confirmAction = async () => true;
  let finish;
  const request = async () => await new Promise((resolve) => { finish = resolve; });
  const refreshDocuments = async () => {};
  let deletionPreviewVersion = 'v1';
  let deletionPreviewToken = 'preview';
  eval(section('async function forgetDocument()', '\n\n        async function refreshForgotten'));
  const action = forgetDocument();
  await Promise.resolve();
  await Promise.resolve();
  if (switchDocument) {
    current = {token: 2, libraryId: 'library-one', path: 'other.md', sourceVersion: 'v2'};
  }
  finish({ok: false, json: async () => ({detail: 'failed'})});
  await action;
  return message;
}

async function forgottenRestoreFailure(switchLibrary) {
  const message = {textContent: '', kind: ''};
  const tombstone = {
    value: 'tombstone-one', selectedIndex: 0,
    options: [{textContent: 'note.md · 遗忘于今天'}],
    dataset: {libraryId: 'library-one', generation: '4'},
  };
  const commit = {
    value: 'commit-one', dataset: {libraryId: 'library-one', generation: '4'}
  };
  const submit = {dataset: {}};
  let submitHandler;
  const forgottenForm = {
    addEventListener: (_type, handler) => { submitHandler = handler; },
    querySelector: (selector) => selector === '#forgotten-memory' ? tombstone
      : selector === '#forgotten-commit' ? commit
      : selector === '#forgotten-message' ? message : submit,
  };
  const editorLibrary = {value: 'library-one'};
  const generations = {forgotten: 4};
  const editorContext = {snapshot: () => ({libraryId: editorLibrary.value, path: null})};
  const confirmAction = async () => true;
  const refreshDocuments = async () => {};
  let finish;
  const request = async () => await new Promise((resolve) => { finish = resolve; });
  eval(section(
    "forgottenForm.addEventListener('submit'",
    "\n        editor.querySelector('#save-change')"
  ));
  const action = submitHandler({preventDefault() {}});
  await Promise.resolve();
  await Promise.resolve();
  if (switchLibrary) editorLibrary.value = 'library-two';
  finish({ok: false, json: async () => ({detail: 'failed'})});
  await action;
  return message;
}

Promise.all([
  candidateFailure(false), candidateFailure(true),
  pinFailure(false), pinFailure(true),
  forgetFailure(false), forgetFailure(true),
  forgottenRestoreFailure(false), forgottenRestoreFailure(true),
]).then(([
  candidate, staleCandidate, pin, stalePin, forget, staleForget, restore, staleRestore
]) => {
  console.log(JSON.stringify({
    candidate, staleCandidate, pin, stalePin, forget, staleForget, restore, staleRestore
  }));
});
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["candidate"] == {
        "textContent": "候选操作失败：测试失败",
        "kind": "error",
    }
    assert payload["forget"] == {
        "textContent": "记忆遗忘失败：测试失败",
        "kind": "error",
    }
    assert payload["pin"] == {
        "textContent": "候选记忆置顶失败。",
        "kind": "error",
    }
    assert payload["restore"] == {
        "textContent": "遗忘记忆恢复失败：测试失败",
        "kind": "error",
    }
    assert payload["staleCandidate"]["textContent"] == ""
    assert payload["stalePin"]["textContent"] == ""
    assert payload["staleForget"]["textContent"] == ""
    assert payload["staleRestore"]["textContent"] == ""


def test_management_ui_forgotten_restore_rejects_stale_library_context() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the forgotten restore regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function section(startMarker, endMarker) {
  const start = html.indexOf(startMarker);
  const end = html.indexOf(endMarker, start);
  return html.slice(start, end);
}
const refreshImplementation = section(
  'function invalidateForgottenSelection(',
  '\n\n        async function refreshBindings'
);
const submitImplementation = section(
  "forgottenForm.addEventListener('submit'",
  "\n        editor.querySelector('#save-change')"
);
const makeSelect = () => ({
  value: '', dataset: {}, options: [], selectedIndex: 0,
  append(option) { this.options.push(option); if (!this.value) this.value = option.value; },
  replaceChildren() { this.options = []; this.value = ''; },
});
const memory = makeSelect();
const commit = makeSelect();
const submit = {disabled: false, dataset: {}};
const message = {textContent: '', kind: ''};
let submitHandler;
const forgottenForm = {
  addEventListener: (_type, handler) => { submitHandler = handler; },
  querySelector: (selector) => selector === '#forgotten-memory' ? memory
    : selector === '#forgotten-commit' ? commit
    : selector === '#forgotten-message' ? message : submit,
};
let libraryChangeHandler;
const editorLibrary = {
  value: 'library-one',
  addEventListener: (_type, handler) => { libraryChangeHandler = handler; },
};
const historyLibrary = {value: ''};
const generations = {forgotten: 0};
const key = 'test-key';
const document = {createElement: () => ({value: '', textContent: ''})};
const setMessage = (node, value, kind = '') => { node.textContent = value; node.kind = kind; };
const describeError = (_failure, context) => context;
const crypto = {randomUUID: () => 'operation'};
const editorContext = {snapshot: () => ({libraryId: editorLibrary.value, path: null})};
const performOnce = async (_key, _button, _label, action) => await action();
const requests = [];
const request = (url) => new Promise((resolve) => requests.push({url, resolve}));
let confirmResolve;
const confirmAction = () => new Promise((resolve) => { confirmResolve = resolve; });
const refreshDocuments = async () => { invalidateForgottenSelection('正在加载遗忘记忆...'); };
eval(refreshImplementation);
eval(submitImplementation);
eval(html.match(/editorLibrary\.addEventListener\('change'.*$/m)[0]);

async function staleListScenario() {
  const oldLoad = refreshForgotten();
  editorLibrary.value = 'library-two';
  const switchAction = libraryChangeHandler();
  const immediate = {
    memoryCount: memory.options.length,
    commitCount: commit.options.length,
    disabled: submit.disabled,
    generation: generations.forgotten,
  };
  requests[0].resolve({
    ok: true,
    json: async () => [{id: 'old', path: 'old.md', deleted_at: 'today'}],
  });
  requests[1].resolve({
    ok: true,
    json: async () => [{commit: 'old-commit', subject: 'old'}],
  });
  await Promise.all([oldLoad, switchAction]);
  return {
    immediate,
    memoryCount: memory.options.length,
    commitCount: commit.options.length,
    disabled: submit.disabled,
  };
}

async function staleSubmitScenario() {
  editorLibrary.value = 'library-one';
  const generation = ++generations.forgotten;
  memory.value = 'tombstone-one';
  memory.options = [{textContent: 'note.md'}];
  memory.dataset = {libraryId: 'library-one', generation: String(generation)};
  commit.value = 'commit-one';
  commit.dataset = {libraryId: 'library-one', generation: String(generation)};
  const before = requests.length;
  const action = submitHandler({preventDefault() {}});
  await Promise.resolve();
  editorLibrary.value = 'library-two';
  invalidateForgottenSelection('正在加载遗忘记忆...');
  confirmResolve(true);
  await action;
  return {
    requestCount: requests.length - before,
    message: message.textContent,
    disabled: submit.disabled,
  };
}

(async () => {
  const staleList = await staleListScenario();
  const staleSubmit = await staleSubmitScenario();
  console.log(JSON.stringify({staleList, staleSubmit, generation: generations.forgotten}));
})();
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["staleList"] == {
        "immediate": {
            "memoryCount": 0,
            "commitCount": 0,
            "disabled": True,
            "generation": 2,
        },
        "memoryCount": 0,
        "commitCount": 0,
        "disabled": True,
    }
    assert payload["staleSubmit"] == {
        "requestCount": 0,
        "message": "正在加载遗忘记忆...",
        "disabled": True,
    }
    assert payload["generation"] == 4


def test_management_ui_sensitive_refresh_preserves_new_selection() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the sensitive refresh regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('async function refreshSensitiveQuarantine(');
const end = html.indexOf('\n\n        async function refreshDocuments', start);
const implementation = html.slice(start, end);
const makeNode = () => ({
  textContent: '', innerHTML: '', disabled: false, dataset: {}, children: [],
  append(child) { this.children.push(child); return child; },
  prepend(child) { this.children.unshift(child); },
  appendChild(child) { this.children.push(child); return child; },
  replaceChildren() { this.children = []; },
  addEventListener() {},
});
const nodes = {
  '#sensitive-title': makeNode(), '#sensitive-summary': makeNode(),
  '#sensitive-metadata': makeNode(), '#sensitive-message': makeNode(),
  '#sensitive-acknowledge': makeNode(), '#sensitive-discard': makeNode(),
};
const sensitiveDetail = {querySelector: (selector) => nodes[selector]};
const sensitiveList = makeNode();
const pageNodes = {
  '#sensitive-previous': makeNode(), '#sensitive-next': makeNode(),
  '#sensitive-page': makeNode(),
};
const sensitivePagination = {querySelector: (selector) => pageNodes[selector]};
const document = {createElement: () => makeNode()};
const display = (value) => value;
const displaySensitiveSummary = (value) => value;
const setMessage = () => {};
const describeError = () => '';
const confirmAction = async () => true;
const performOnce = async (_key, _button, _label, action) => await action();
const crypto = {randomUUID: () => 'operation'};
const key = 'test-key';
const generations = {sensitive: 0};
let sensitiveOffset = 0;
const sensitivePageSize = 25;
let sensitiveSelectionGeneration = 0;
let activeSensitiveRecord = null;
const first = {
  id: 'record-a', disposition: 'quarantined', summary: 'first',
  categories: ['secret'], created_at: 'today', resolved_at: null,
};
const second = {
  id: 'record-b', disposition: 'quarantined', summary: 'second',
  categories: ['secret'], created_at: 'today', resolved_at: null,
};
let finish;
const request = async () => await new Promise((resolve) => { finish = resolve; });
eval(implementation
  + '\nglobalThis.refreshUnderTest = refreshSensitiveQuarantine;'
  + '\nglobalThis.showUnderTest = showSensitiveRecord;');
showUnderTest(first);
const selectionContext = {recordId: first.id, selectionGeneration: sensitiveSelectionGeneration};
const refresh = refreshUnderTest(first.id, selectionContext);
showUnderTest(second);
finish({ok: true, json: async () => [first, second]});
refresh.then(() => console.log(JSON.stringify({
  activeId: activeSensitiveRecord.id,
  title: nodes['#sensitive-title'].textContent,
})));
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "activeId": "record-b",
        "title": "quarantined · unresolved",
    }


def test_management_ui_sensitive_resolution_stays_with_original_selection() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the sensitive resolution regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('function clearSensitiveRecord(');
const end = html.indexOf('\n\n        async function refreshDocuments', start);
const implementation = html.slice(start, end);
const setMessage = (node, text, kind = '') => {
  node.textContent = text;
  node.kind = kind;
};
const describeError = (_failure, context) => `${context}：测试失败`;
const performOnce = async (_name, _button, _label, action) => await action();
const confirmAction = async () => true;
const display = (value) => value;
const displaySensitiveSummary = (value) => value;
const key = 'test-key';

async function scenario(outcome, switchAt) {
  const message = {textContent: '', kind: ''};
  const buttons = {
    '#sensitive-acknowledge': {dataset: {}, disabled: false},
    '#sensitive-discard': {dataset: {}, disabled: false},
  };
  const nodes = {
    '#sensitive-title': {textContent: ''},
    '#sensitive-summary': {textContent: ''},
    '#sensitive-metadata': {textContent: ''},
    '#sensitive-message': message,
    ...buttons,
  };
  const sensitiveDetail = {querySelector: (selector) => nodes[selector]};
  let activeSensitiveRecord = null;
  let sensitiveSelectionGeneration = 0;
  const first = {
    id: 'record-a', disposition: 'quarantined', summary: 'first',
    categories: ['secret'], created_at: 'today', resolved_at: null
  };
  const second = {
    id: 'record-b', disposition: 'quarantined', summary: 'second',
    categories: ['secret'], created_at: 'today', resolved_at: null
  };
  let requestedUrl = null;
  let finish;
  const request = async (url) => {
    requestedUrl = url;
    return await new Promise((resolve) => { finish = resolve; });
  };
  const refreshSensitiveQuarantine = async () => {
    message.textContent = '';
    message.kind = '';
    if (switchAt === 'refresh') showSensitiveRecord(second);
  };
  eval(implementation + `
    \nglobalThis.showSensitiveRecordUnderTest = showSensitiveRecord;
    \nglobalThis.resolveSensitiveRecordUnderTest = resolveSensitiveRecord;
  `);
  showSensitiveRecordUnderTest(first);
  const action = resolveSensitiveRecordUnderTest('acknowledge');
  await Promise.resolve();
  if (switchAt === 'response') showSensitiveRecordUnderTest(second);
  finish(outcome === 'success'
    ? {ok: true, json: async () => ({...first, resolved_at: 'later', resolution: 'acknowledge'})}
    : {ok: false, json: async () => ({detail: 'failed'})});
  await action;
  return {message, requestedUrl, activeId: activeSensitiveRecord?.id};
}

Promise.all([
  scenario('success', 'none'),
  scenario('success', 'refresh'),
  scenario('failure', 'none'),
  scenario('failure', 'response'),
]).then(([success, staleSuccess, failure, staleFailure]) => {
  console.log(JSON.stringify({success, staleSuccess, failure, staleFailure}));
});
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["success"]["requestedUrl"].endswith(
        "/sensitive-quarantine/record-a/resolve"
    )
    assert payload["success"]["message"] == {
        "textContent": "已知悉此敏感记录。",
        "kind": "success",
    }
    assert payload["failure"]["message"] == {
        "textContent": "敏感记录处理失败：测试失败",
        "kind": "error",
    }
    assert payload["staleSuccess"]["activeId"] == "record-b"
    assert payload["staleSuccess"]["message"]["textContent"] == ""
    assert payload["staleFailure"]["activeId"] == "record-b"
    assert payload["staleFailure"]["message"]["textContent"] == ""


def test_management_ui_history_diff_selection_clears_stale_restore_and_handles_retry() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the history diff regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('async function loadHistoryDiff(');
const end = html.indexOf('\n\n        async function refreshHistory', start);
const implementation = html.slice(start, end);
const makeNode = () => ({
  textContent: '', children: [],
  addEventListener(type, listener) { this[type] = listener; },
  append(child) { this.children.push(child); },
  replaceChildren() { this.children = []; },
});
const historyDiff = makeNode();
const restoreAction = makeNode();
restoreAction.append({textContent: '旧恢复动作'});
historyDiff.textContent = '旧提交差异';
const historyMessage = {textContent: '', kind: ''};
const setMessage = (node, text, kind = '') => { node.textContent = text; node.kind = kind; };
const document = {createElement: () => makeNode()};
const key = 'test-key';
const generations = {historyDiff: 0};
const context = {libraryId: 'library-one', path: 'note.md', sourceVersion: 'v1', token: 9};
const editorContext = {isCurrent: (candidate) => candidate === context};
const historySelection = {
  clear() {},
  select(libraryId, selected, commit) {
    return {
      libraryId, path: selected.path, sourceVersion: selected.source_version,
      contextToken: selected.token, commit
    };
  },
};
const pending = new Set();
const requests = [];
const request = (url) => new Promise((resolve) => requests.push({url, resolve}));
eval(implementation + '\nglobalThis.loadHistoryDiffUnderTest = loadHistoryDiff;');

async function run() {
  const firstEntry = {commit: '111111111111aaaa'};
  const first = loadHistoryDiffUnderTest('library-one', firstEntry, context);
  const duplicate = await loadHistoryDiffUnderTest('library-one', firstEntry, context);
  const loading = {
    diff: historyDiff.textContent,
    actions: restoreAction.children.map((child) => child.textContent),
    message: historyMessage.textContent,
  };
  const secondEntry = {commit: '222222222222bbbb'};
  const second = loadHistoryDiffUnderTest('library-one', secondEntry, context);
  requests[1].resolve({ok: true, json: async () => ({diff: '第二个差异'})});
  await second;
  requests[0].resolve({ok: true, json: async () => ({diff: '迟到的第一个差异'})});
  await first;
  const afterRace = {
    diff: historyDiff.textContent,
    actions: restoreAction.children.map((child) => child.textContent),
  };

  const failedEntry = {commit: '333333333333cccc'};
  const failed = loadHistoryDiffUnderTest('library-one', failedEntry, context);
  requests[2].resolve({ok: false});
  await failed;
  const afterFailure = {
    diff: historyDiff.textContent,
    message: historyMessage.textContent,
    actions: restoreAction.children.map((child) => child.textContent),
  };
  const retry = restoreAction.children[0];
  const retried = retry.click();
  requests[3].resolve({ok: true, json: async () => ({diff: '重试后的差异'})});
  await retried;
  console.log(JSON.stringify({
    duplicate, requestCount: requests.length, loading, afterRace, afterFailure,
    afterRetry: {
      diff: historyDiff.textContent,
      message: historyMessage.textContent,
      actions: restoreAction.children.map((child) => child.textContent),
    },
  }));
}
run();
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["duplicate"] is False
    assert payload["requestCount"] == 4
    assert payload["loading"] == {
        "diff": "正在加载提交差异...",
        "actions": [],
        "message": "正在加载提交 111111111111 的差异...",
    }
    assert payload["afterRace"] == {
        "diff": "第二个差异",
        "actions": ["从 222222222222 恢复 note.md"],
    }
    assert payload["afterFailure"] == {
        "diff": "提交差异加载失败。",
        "message": "提交差异加载失败，请重试。",
        "actions": ["重试加载差异"],
    }
    assert payload["afterRetry"] == {
        "diff": "重试后的差异",
        "message": "已加载提交 333333333333 的差异。",
        "actions": ["从 333333333333 恢复 note.md"],
    }


def test_management_ui_load_document_returns_null_after_same_path_edit() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the document load regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    history_path = (
        Path(__file__).parents[1]
        / "src/personal_agent_memory/static/editor_history.js"
    )
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const PamHistory = require(process.argv[2]);
const start = html.indexOf('async function loadDocument(');
const end = html.indexOf('\n\n        function showEditorPanel', start);
const implementation = html.slice(start, end);
const editorContext = PamHistory.contextStore();
const editorLibrary = {value: 'library-one'};
const key = 'test-key';
const nodes = new Map();
const node = (selector) => {
  if (!nodes.has(selector)) nodes.set(selector, {
    textContent: '', value: '', disabled: true, hidden: false,
    setAttribute() {}, replaceChildren() {},
  });
  return nodes.get(selector);
};
const editor = {querySelector: node};
const historyDocument = {value: ''};
const restoreAction = {replaceChildren() {}};
const historySelection = {clear() {}};
const historyMessage = {textContent: '', kind: ''};
const setMessage = (target, text, kind = '') => {
  target.textContent = text;
  target.kind = kind;
};
const documentContextIsCurrent = (context) => {
  const current = editorContext.snapshot();
  return Boolean(context && current && editorContext.isCurrent(context)
    && editorLibrary.value === context.libraryId
    && current.libraryId === context.libraryId
    && current.path === context.path
    && current.token === context.token);
};
const request = async () => ({
  ok: true,
  json: async () => ({path: 'same.md', source_version: 'v2', content: 'loaded'}),
});
const showEditorPanel = () => {};
let activeDocument = null;
let deletionPreviewVersion = null;
let deletionPreviewToken = null;
const refreshHistory = () => {
  editorContext.edit('edited while history refresh waits');
  return Promise.resolve();
};
eval(implementation + '\nglobalThis.loadDocumentUnderTest = loadDocument;');

(async () => {
  const result = await loadDocumentUnderTest('same.md');
  const current = editorContext.snapshot();
  console.log(JSON.stringify({
    result: result === null ? null : result.token,
    path: current.path,
    content: current.content,
  }));
})();
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path), str(history_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "result": None,
        "path": "same.md",
        "content": "edited while history refresh waits",
    }


def test_management_ui_document_mutations_reject_same_path_in_another_library() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the document context regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function section(startMarker, endMarker) {
  const start = html.indexOf(startMarker);
  const end = html.indexOf(endMarker, start);
  return html.slice(start, end);
}
const restoreImplementation = section(
  'async function restoreDocument(selection)',
  '\n\n        async function previewForgetting'
);
const saveImplementation = section(
  'async function saveDocument()',
  '\n\n        async function refreshForgotten'
);
const setMessage = (node, text, kind = '') => { node.textContent = text; node.kind = kind; };
const describeError = (_failure, context) => `${context}：测试失败`;
const performOnce = async (_name, _button, _label, action) => await action();
const confirmAction = async () => true;
const crypto = {randomUUID: () => 'operation'};
const key = 'test-key';

async function restoreScenario(switchLibrary, editDuringHistory = false) {
  const historyMessage = {textContent: '', kind: ''};
  const restoreAction = {querySelector: () => ({dataset: {}}), replaceChildren() {}};
  const historySelection = {clear() {}};
  const editorLibrary = {value: 'library-one'};
  const original = {libraryId: 'library-one', path: 'same.md', sourceVersion: 'v1', token: 7};
  let current = original;
  const editorContext = {
    snapshot: () => current,
    isCurrent: (candidate) => candidate.token === current.token
  };
  const documentContextIsCurrent = (context) => Boolean(
    context && current && editorContext.isCurrent(context)
    && editorLibrary.value === context.libraryId
    && current.libraryId === context.libraryId
    && current.path === context.path
    && current.token === context.token
  );
  const selection = {
    libraryId: 'library-one', path: 'same.md', expectedSourceVersion: 'v1',
    contextToken: 7, commit: 'abcdef1234567890'
  };
  let finish;
  const request = async () => await new Promise((resolve) => { finish = resolve; });
  let loadCount = 0;
  const loadDocument = async () => {
    ++loadCount;
    current = {libraryId: editorLibrary.value, path: 'same.md', sourceVersion: 'v2', token: 8};
    if (editDuringHistory) {
      current = {...current, content: 'edited while history loads', token: 9};
      return null;
    }
    return current;
  };
  const PamHistory = {request: () => ({url: '/restore', body: {}})};
  eval(restoreImplementation + '\nglobalThis.restoreUnderTest = restoreDocument;');
  const action = restoreUnderTest(selection);
  await Promise.resolve();
  await Promise.resolve();
  if (switchLibrary) editorLibrary.value = 'library-two';
  finish({ok: true});
  await action;
  return {message: historyMessage, loadCount};
}

async function saveScenario(switchLibrary, editDuringHistory = false) {
  const message = {textContent: '', kind: ''};
  const editorLibrary = {value: 'library-one'};
  const original = {libraryId: 'library-one', path: 'same.md', sourceVersion: 'v1', token: 11};
  let current = original;
  const editorContext = {
    snapshot: () => current,
    isCurrent: (candidate) => candidate.token === current.token
  };
  const documentContextIsCurrent = (context) => Boolean(
    context && current && editorContext.isCurrent(context)
    && editorLibrary.value === context.libraryId
    && current.libraryId === context.libraryId
    && current.path === context.path
    && current.token === context.token
  );
  const editor = {
    querySelector: (selector) => selector === '#editor-message' ? message : {dataset: {}}
  };
  const previewChange = async () => true;
  let finish;
  const request = async () => await new Promise((resolve) => { finish = resolve; });
  let loadCount = 0;
  const loadDocument = async () => {
    ++loadCount;
    current = {libraryId: editorLibrary.value, path: 'same.md', sourceVersion: 'v2', token: 12};
    if (editDuringHistory) {
      current = {...current, content: 'edited while history loads', token: 13};
      return null;
    }
    return current;
  };
  const PamHistory = {saveRequest: () => ({url: '/save', body: {}})};
  eval(saveImplementation + '\nglobalThis.saveUnderTest = saveDocument;');
  const action = saveUnderTest();
  await Promise.resolve();
  await Promise.resolve();
  if (switchLibrary) editorLibrary.value = 'library-two';
  finish({ok: true});
  await action;
  return {message, loadCount};
}

Promise.all([
  restoreScenario(false), restoreScenario(true),
  restoreScenario(false, true), saveScenario(false), saveScenario(true),
  saveScenario(false, true)
]).then(([restore, staleRestore, editedRestore, save, staleSave, editedSave]) => {
  console.log(JSON.stringify({restore, staleRestore, editedRestore, save, staleSave, editedSave}));
});
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["restore"] == {
        "message": {
            "textContent": "已从 abcdef123456 恢复 same.md。",
            "kind": "success",
        },
        "loadCount": 1,
    }
    assert payload["save"] == {
        "message": {"textContent": "文档已保存。", "kind": "success"},
        "loadCount": 1,
    }
    assert payload["staleRestore"] == {
        "message": {"textContent": "", "kind": ""},
        "loadCount": 0,
    }
    assert payload["staleSave"] == {
        "message": {"textContent": "", "kind": ""},
        "loadCount": 0,
    }
    assert payload["editedRestore"] == {
        "message": {"textContent": "", "kind": ""},
        "loadCount": 1,
    }
    assert payload["editedSave"] == {
        "message": {"textContent": "", "kind": ""},
        "loadCount": 1,
    }


def test_management_ui_forgetting_preview_deduplicates_and_rejects_stale_results() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the forgetting preview regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('async function previewForgetting()');
const end = html.indexOf('\n\n        async function forgetDocument', start);
const implementation = html.slice(start, end);
const message = {textContent: '', kind: ''};
const diff = {textContent: ''};
const previewButton = {dataset: {}, disabled: false, textContent: '预览遗忘影响'};
const confirmButton = {disabled: true};
const editor = {querySelector: (selector) => selector === '#editor-message' ? message
  : selector === '#document-diff' ? diff
  : selector === '#preview-delete' ? previewButton : confirmButton};
const editorLibrary = {value: 'library-one'};
const contexts = {
  first: {libraryId: 'library-one', path: 'first.md', sourceVersion: 'v1', token: 1},
  second: {libraryId: 'library-one', path: 'second.md', sourceVersion: 'v2', token: 2},
};
let current = contexts.first;
const editorContext = {
  snapshot: () => current,
  isCurrent: (candidate) => candidate.token === current.token
};
const documentContextIsCurrent = (context) => Boolean(
  context && current && editorContext.isCurrent(context)
  && editorLibrary.value === context.libraryId
  && current.libraryId === context.libraryId
  && current.path === context.path
  && current.token === context.token
);
const pending = new Set();
const generations = {forgettingPreview: 0};
const setBusy = (button, busy) => { button.disabled = busy; };
const setMessage = (node, text, kind = '') => { node.textContent = text; node.kind = kind; };
const describeError = (_failure, context) => context;
const showEditorPanel = () => {};
const crypto = {randomUUID: () => 'operation'};
const key = 'test-key';
let deletionPreviewVersion = null;
let deletionPreviewToken = null;
const requests = [];
const request = (url) => new Promise((resolve) => requests.push({url, resolve}));
eval(implementation + '\nglobalThis.previewForgettingUnderTest = previewForgetting;');

async function run() {
  const first = previewForgettingUnderTest();
  const duplicate = await previewForgettingUnderTest();
  current = contexts.second;
  const second = previewForgettingUnderTest();
  requests[1].resolve({
    ok: true,
    json: async () => ({preview_token: 'second-preview', marker: 'second'})
  });
  await second;
  requests[0].resolve({
    ok: true,
    json: async () => ({preview_token: 'first-preview', marker: 'first'})
  });
  await first;
  console.log(JSON.stringify({
    duplicate, requestCount: requests.length, diff: diff.textContent,
    token: deletionPreviewToken, version: deletionPreviewVersion,
    confirmDisabled: confirmButton.disabled
  }));
}
run();
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "duplicate": False,
        "requestCount": 2,
        "diff": json.dumps(
            {"preview_token": "second-preview", "marker": "second"},
            ensure_ascii=False,
            indent=2,
        ),
        "token": "second-preview",
        "version": "v2",
        "confirmDisabled": False,
    }


def test_management_ui_retention_actions_confirm_and_report_after_refresh() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the retention action regression")
    html_path = Path(__file__).parents[1] / "src/personal_agent_memory/static/index.html"
    program = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function section(startMarker, endMarker) {
  const start = html.indexOf(startMarker);
  const end = html.indexOf(endMarker, start);
  return html.slice(start, end);
}
const setMessage = (node, text, kind) => { node.textContent = text; node.kind = kind; };
const performOnce = async (_name, _button, _label, action) => await action();
const key = 'test-key';

async function restoreScenario(switchLibrary) {
  const message = {textContent: '', kind: ''};
  const library = {value: 'library-one'};
  const retentionForm = {
    querySelector: (selector) => (
      selector === '#retention-library' ? library : message
    )
  };
  let retentionSelectionGeneration = 3;
  let confirmation;
  const confirmAction = async (title, description) => {
    confirmation = {title, description};
    return true;
  };
  const request = async () => ({ok: true});
  const refreshRetention = async () => {
    message.textContent = '';
    if (switchLibrary) { ++retentionSelectionGeneration; library.value = 'library-two'; }
  };
  const refreshCandidates = async () => {};
  eval(section(
    'async function restoreRecycledCandidate(item, button)',
    '\n\n        async function refreshRetention'
  ));
  await restoreRecycledCandidate({candidate: {id: 'candidate-one'}}, {dataset: {}});
  return {message, confirmation};
}

async function cleanupScenario(switchLibrary) {
  const message = {textContent: '', kind: ''};
  const library = {value: 'library-one'};
  const button = {dataset: {}};
  const retentionForm = {
    querySelector: (selector) => (
      selector === '#retention-library'
        ? library
        : selector === '#retention-run' ? button : message
    )
  };
  const generations = {retention: 8};
  let retentionSelectionGeneration = 5;
  let retentionPreviewContext = {
    libraryId: 'library-one', generation: 8, selectionGeneration: 5,
    affectedCount: 7, summary: '预计影响\n采集收件箱：2 条\n候选永久删除：5 条'
  };
  let confirmation;
  const confirmAction = async (title, description) => {
    confirmation = {title, description};
    return true;
  };
  const request = async () => ({ok: true});
  const refreshRetention = async () => {
    message.textContent = '';
    if (switchLibrary) { ++retentionSelectionGeneration; library.value = 'library-two'; }
  };
  const refreshCandidates = async () => {};
  eval(section(
    'async function runRetentionCleanup()',
    '\n\n        async function refreshReconciliation'
  ));
  await runRetentionCleanup();
  return {message, confirmation};
}

Promise.all([
  restoreScenario(false), restoreScenario(true),
  cleanupScenario(false), cleanupScenario(true)
]).then(([restore, staleRestore, cleanup, staleCleanup]) => {
  console.log(JSON.stringify({restore, staleRestore, cleanup, staleCleanup}));
});
"""
    result = subprocess.run(
        [node, "-e", program, str(html_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["restore"]["confirmation"] == {
        "title": "确认恢复候选记忆",
        "description": "将从记忆库 library-one 的回收站恢复候选 candidate-one。",
    }
    assert payload["restore"]["message"] == {
        "textContent": "候选记忆已恢复。",
        "kind": "success",
    }
    assert payload["staleRestore"]["message"]["textContent"] == ""
    cleanup_confirmation = payload["cleanup"]["confirmation"]
    assert cleanup_confirmation["title"] == "确认立即清理"
    assert "library-one" in cleanup_confirmation["description"]
    assert "预计影响 7 条" in cleanup_confirmation["description"]
    assert "采集收件箱：2 条" in cleanup_confirmation["description"]
    assert "候选永久删除：5 条" in cleanup_confirmation["description"]
    assert payload["cleanup"]["message"] == {
        "textContent": "保留数据清理完成。",
        "kind": "success",
    }
    assert payload["staleCleanup"]["message"]["textContent"] == ""
