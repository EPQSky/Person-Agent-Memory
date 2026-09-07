from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import personal_agent_memory.state as state_module
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.state import PlatformState

NOW = "2026-09-07T00:00:00Z"


def _client(
    tmp_path: Path, *, existing_markdown: str | None = None
) -> tuple[TestClient, dict[str, str], str]:
    state_dir = tmp_path / "state"
    memory_root = tmp_path / "memory"
    if existing_markdown is not None:
        memory_root.mkdir()
        (memory_root / "formal.md").write_text(existing_markdown)
    app = create_app(
        Settings(
            state_dir=state_dir,
            library_roots=(tmp_path,),
            retention_now=NOW,
        )
    )
    client = TestClient(app)
    client.__enter__()
    headers = {"Authorization": f"Bearer {(state_dir / 'api-key').read_text().strip()}"}
    response = client.post(
        "/api/v1/libraries",
        headers=headers,
        json={"path": str(memory_root), "kind": "project"},
    )
    assert response.status_code == 201
    return client, headers, str(response.json()["id"])


def _candidate(
    client: TestClient, headers: dict[str, str], library_id: str, key: str
) -> dict[str, object]:
    response = client.post(
        "/mcp/candidates",
        headers=headers,
        json={
            "library_id": library_id,
            "suggested_type": "decision",
            "body": f"# Decision\n\nRetain {key}.\n",
            "source_references": [f"codex-session:{key}#assistant-final"],
            "creator": "test",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 201
    return response.json()


def test_candidate_boundaries_pin_conflict_recycle_restore_and_delete(tmp_path: Path) -> None:
    client, headers, library_id = _client(tmp_path)
    try:
        expired = _candidate(client, headers, library_id, "expired")
        boundary = _candidate(client, headers, library_id, "boundary")
        pinned = _candidate(client, headers, library_id, "pinned")
        conflict = _candidate(client, headers, library_id, "conflict")
        database = tmp_path / "state" / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE candidate_memories SET created_at = '2026-06-08 23:59:59' WHERE id = ?",
                (expired["id"],),
            )
            connection.execute(
                "UPDATE candidate_memories SET created_at = '2026-06-09 00:00:00' "
                "WHERE id IN (?, ?, ?)",
                (boundary["id"], pinned["id"], conflict["id"]),
            )
            connection.execute(
                "UPDATE candidate_governance SET classification = 'conflict' "
                "WHERE candidate_id = ?",
                (conflict["id"],),
            )
        assert (
            client.put(
                f"/api/v1/candidates/{pinned['id']}/pin",
                headers=headers,
                json={"pinned": True},
            ).status_code
            == 200
        )

        preview = client.get(
            f"/api/v1/libraries/{library_id}/retention-cleanup/preview", headers=headers
        ).json()
        assert preview["counts"]["candidates_to_recycle"] == 2
        assert preview["counts"]["protected_candidates"] == 2
        first = client.post(f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers)
        assert first.status_code == 200
        recycled = client.get(
            f"/api/v1/libraries/{library_id}/candidate-recycle-bin", headers=headers
        ).json()
        assert {item["candidate"]["id"] for item in recycled} == {
            expired["id"],
            boundary["id"],
        }
        assert (
            client.post(f"/api/v1/candidates/{boundary['id']}/restore", headers=headers).status_code
            == 200
        )
        immediate = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        )
        assert immediate.status_code == 200
        assert immediate.json()["counts"]["candidates_to_recycle"] == 0
        assert client.get(
            f"/api/v1/candidates/{boundary['id']}", headers=headers
        ).status_code == 200

        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE candidate_retention SET recycled_at = '2026-08-08 00:00:00' "
                "WHERE candidate_id = ?",
                (expired["id"],),
            )
        second = client.post(f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers)
        assert second.status_code == 200
        assert client.get(f"/api/v1/candidates/{expired['id']}", headers=headers).status_code == 404
        assert client.get(f"/api/v1/candidates/{pinned['id']}", headers=headers).status_code == 200
        assert (
            client.get(f"/api/v1/candidates/{conflict['id']}", headers=headers).status_code == 200
        )
    finally:
        client.__exit__(None, None, None)


def test_inbox_early_cleanup_and_diagnostics_do_not_touch_authoritative_data(
    tmp_path: Path,
) -> None:
    expected = "# Formal\n\nKeep forever.\n"
    client, headers, library_id = _client(tmp_path, existing_markdown=expected)
    try:
        memory = tmp_path / "memory" / "formal.md"
        database = tmp_path / "state" / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute(
                """INSERT INTO capture_inbox
                   (event_id, request_hash, session_id, project_id, library_id, turn_id,
                    event_kind, content, occurred_at, received_at, consolidated_at)
                   VALUES ('old', 'hash-old', 'old-session', 'project', ?, 'turn',
                           'user', 'old private body', '2026-08-07T00:00:00Z',
                           '2026-08-07 00:00:00', NULL),
                          ('early', 'hash-early', 'early-session', 'project', ?, 'turn',
                           'user', 'early private body', '2026-09-06T00:00:00Z',
                           '2026-09-06 00:00:00', '2026-09-06 01:00:00')""",
                (library_id, library_id),
            )
        response = client.post(f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers)
        assert response.status_code == 200
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM capture_inbox").fetchone() == (0,)
            run = connection.execute(
                "SELECT counts_json, last_error FROM retention_cleanup_runs"
            ).fetchone()
            assert run is not None
            assert "private body" not in str(run)
            assert connection.execute("SELECT COUNT(*) FROM memory_documents").fetchone() == (1,)
        assert memory.read_text() == expected
        assert client.delete("/api/v1/diagnostics", headers=headers).json()["deleted"] == 1
    finally:
        client.__exit__(None, None, None)


def test_cleanup_state_survives_restart_and_repeated_runs_are_idempotent(
    tmp_path: Path,
) -> None:
    client, headers, library_id = _client(tmp_path)
    candidate = _candidate(client, headers, library_id, "restart")
    database = tmp_path / "state" / "platform.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE candidate_memories SET created_at = '2026-06-09 00:00:00' "
            "WHERE id = ?",
            (candidate["id"],),
        )
    assert client.post(
        f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
    ).status_code == 200
    client.__exit__(None, None, None)

    app = create_app(
        Settings(
            state_dir=tmp_path / "state",
            library_roots=(tmp_path,),
            retention_now=NOW,
        )
    )
    with TestClient(app) as restarted:
        restarted_headers = {
            "Authorization": (
                f"Bearer {(tmp_path / 'state' / 'api-key').read_text().strip()}"
            )
        }
        recycled = restarted.get(
            f"/api/v1/libraries/{library_id}/candidate-recycle-bin",
            headers=restarted_headers,
        ).json()
        assert [item["candidate"]["id"] for item in recycled] == [candidate["id"]]
        repeated = restarted.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup",
            headers=restarted_headers,
        )
        assert repeated.status_code == 200
        assert repeated.json()["counts"]["candidates_to_recycle"] == 0
        assert repeated.json()["counts"]["candidates_to_delete"] == 0


def test_manual_cleanup_restarts_the_background_cleanup_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [10.0]
    monkeypatch.setattr(state_module, "monotonic", lambda: clock[0])
    client, headers, library_id = _client(tmp_path)
    try:
        platform_state = cast(
            PlatformState, cast(FastAPI, client.app).state.platform_state
        )
        platform_state._last_retention_check = clock[0]
        clock[0] = 70.0

        response = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        )

        assert response.status_code == 200
        assert platform_state._last_retention_check == 70.0
        assert platform_state._retention_generation == 1
        clock[0] = 129.999
        assert state_module.monotonic() - platform_state._last_retention_check < 60.0
        clock[0] = 130.0
        assert state_module.monotonic() - platform_state._last_retention_check >= 60.0
    finally:
        client.__exit__(None, None, None)


def test_manual_cleanup_cancels_an_already_scheduled_background_batch(tmp_path: Path) -> None:
    client, headers, library_id = _client(tmp_path)
    try:
        platform_state = cast(
            PlatformState, cast(FastAPI, client.app).state.platform_state
        )
        scheduled_generation = platform_state._retention_generation

        response = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        )
        assert response.status_code == 200
        candidate = _candidate(client, headers, library_id, "scheduled-batch")
        assert client.put(
            f"/api/v1/libraries/{library_id}/retention-policy",
            headers=headers,
            json={
                "inbox_days": 30,
                "candidate_days": 0,
                "recycle_days": 0,
                "early_inbox_cleanup": True,
                "diagnostics_enabled": True,
                "diagnostic_days": 7,
            },
        ).status_code == 200

        assert (
            platform_state._run_scheduled_retention_cleanup(
                library_id, scheduled_generation
            )
            is None
        )
        preview = client.get(
            f"/api/v1/libraries/{library_id}/retention-cleanup/preview",
            headers=headers,
        )
        assert preview.status_code == 200
        assert preview.json()["counts"]["candidates_to_recycle"] == 1
        assert client.get(
            f"/api/v1/candidates/{candidate['id']}", headers=headers
        ).status_code == 200
    finally:
        client.__exit__(None, None, None)


def test_recycled_candidates_leave_governance_until_restored(tmp_path: Path) -> None:
    client, headers, library_id = _client(tmp_path)
    try:
        candidate = _candidate(client, headers, library_id, "governance-recycle")
        candidate_id = str(candidate["id"])
        database = tmp_path / "state" / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE candidate_memories SET created_at = '2026-06-08 23:59:59' "
                "WHERE id = ?",
                (candidate_id,),
            )

        assert client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).status_code == 200
        governance = client.get("/api/v1/candidate-governance", headers=headers)
        assert governance.status_code == 200
        assert all(item["candidate"]["id"] != candidate_id for item in governance.json())

        actions = (
            (
                "PUT",
                f"/api/v1/candidates/{candidate_id}",
                {
                    "body": "# Decision\n\nEdited while recycled.\n",
                    "operator": "retention-test",
                    "reason": "Must restore first",
                },
            ),
            (
                "POST",
                f"/api/v1/candidates/{candidate_id}/approve",
                {
                    "operator": "retention-test",
                    "reason": "Must restore first",
                    "operation_id": "recycled-approve",
                },
            ),
            (
                "POST",
                f"/api/v1/candidates/{candidate_id}/reject",
                {
                    "operator": "retention-test",
                    "reason": "Must restore first",
                    "operation_id": "recycled-reject",
                },
            ),
            (
                "POST",
                f"/api/v1/candidates/{candidate_id}/resolve",
                {
                    "action": "keep",
                    "operator": "retention-test",
                    "reason": "Must restore first",
                    "operation_id": "recycled-resolve",
                },
            ),
        )
        for method, url, payload in actions:
            response = client.request(method, url, headers=headers, json=payload)
            assert response.status_code == 409
            assert "restore it before governance actions" in str(response.json()["detail"])

        restored = client.post(f"/api/v1/candidates/{candidate_id}/restore", headers=headers)
        assert restored.status_code == 200
        assert any(
            item["candidate"]["id"] == candidate_id
            for item in client.get("/api/v1/candidate-governance", headers=headers).json()
        )
        edited = client.put(
            f"/api/v1/candidates/{candidate_id}",
            headers=headers,
            json={
                "body": "# Decision\n\nEdited after restore.\n",
                "operator": "retention-test",
                "reason": "Restored candidate is active again",
            },
        )
        assert edited.status_code == 200
    finally:
        client.__exit__(None, None, None)


def test_zero_day_policy_uses_one_precomputed_cleanup_plan(tmp_path: Path) -> None:
    client, headers, library_id = _client(tmp_path)
    try:
        candidate = _candidate(client, headers, library_id, "zero-day-plan")
        policy = {
            "inbox_days": 30,
            "candidate_days": 0,
            "recycle_days": 0,
            "early_inbox_cleanup": True,
            "diagnostics_enabled": True,
            "diagnostic_days": 7,
        }
        assert client.put(
            f"/api/v1/libraries/{library_id}/retention-policy",
            headers=headers,
            json=policy,
        ).status_code == 200

        first_preview = client.get(
            f"/api/v1/libraries/{library_id}/retention-cleanup/preview", headers=headers
        ).json()
        database = tmp_path / "state" / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT created_at FROM candidate_memories WHERE id = ?", (candidate["id"],)
            ).fetchone() == ("2026-09-07 00:00:00",)
        assert first_preview["counts"]["candidates_to_recycle"] == 1
        assert first_preview["counts"]["candidates_to_delete"] == 0
        first_run = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).json()
        assert first_run["counts"] == first_preview["counts"]
        assert client.get(
            f"/api/v1/candidates/{candidate['id']}", headers=headers
        ).status_code == 200

        second_preview = client.get(
            f"/api/v1/libraries/{library_id}/retention-cleanup/preview", headers=headers
        ).json()
        assert second_preview["counts"]["candidates_to_recycle"] == 0
        assert second_preview["counts"]["candidates_to_delete"] == 1
        second_run = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).json()
        assert second_run["counts"] == second_preview["counts"]
        assert client.get(
            f"/api/v1/candidates/{candidate['id']}", headers=headers
        ).status_code == 404
    finally:
        client.__exit__(None, None, None)


def test_recycled_candidate_keeps_capture_source_until_permanent_delete(
    tmp_path: Path,
) -> None:
    client, headers, library_id = _client(tmp_path)
    try:
        candidate = _candidate(client, headers, library_id, "capture-source")
        database = tmp_path / "state" / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute(
                """INSERT INTO capture_inbox
                   (event_id, request_hash, session_id, project_id, library_id, turn_id,
                    event_kind, content, occurred_at, received_at, consolidated_at)
                   VALUES ('source-event', 'source-hash', 'source-session', 'project', ?,
                           'source-turn', 'user', 'source evidence',
                           '2026-09-06T00:00:00Z', '2026-09-06 00:00:00',
                           '2026-09-06 00:01:00')""",
                (library_id,),
            )
            connection.execute(
                """INSERT INTO capture_rounds
                   (session_id, turn_id, library_id, status, candidate_id, updated_at)
                   VALUES ('source-session', 'source-turn', ?, 'done', ?, ?)""",
                (library_id, candidate["id"], "2026-09-06 00:01:00"),
            )
        policy = {
            "inbox_days": 30,
            "candidate_days": 0,
            "recycle_days": 0,
            "early_inbox_cleanup": True,
            "diagnostics_enabled": True,
            "diagnostic_days": 7,
        }
        assert client.put(
            f"/api/v1/libraries/{library_id}/retention-policy",
            headers=headers,
            json=policy,
        ).status_code == 200

        first = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).json()
        assert first["counts"]["candidates_to_recycle"] == 1
        assert first["counts"]["capture_inbox"] == 0
        second = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).json()
        assert second["counts"]["candidates_to_delete"] == 1
        assert second["counts"]["capture_inbox"] == 0
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT content FROM capture_inbox WHERE event_id = 'source-event'"
            ).fetchone() == ("source evidence",)

        third = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).json()
        assert third["counts"]["capture_inbox"] == 1
    finally:
        client.__exit__(None, None, None)


def test_busy_checkpoint_leaves_recoverable_physical_erasure_job(tmp_path: Path) -> None:
    plaintext = "Ticket16BusyCheckpointPlaintext-8a0448c1"
    client, headers, library_id = _client(tmp_path)
    try:
        candidate = _candidate(client, headers, library_id, plaintext)
        database = tmp_path / "state" / "platform.sqlite3"
        assert client.put(
            f"/api/v1/libraries/{library_id}/retention-policy",
            headers=headers,
            json={
                "inbox_days": 30,
                "candidate_days": 0,
                "recycle_days": 0,
                "early_inbox_cleanup": True,
                "diagnostics_enabled": True,
                "diagnostic_days": 7,
            },
        ).status_code == 200
        assert client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).status_code == 200

        reader = sqlite3.connect(database)
        try:
            reader.execute("BEGIN")
            assert reader.execute(
                "SELECT body FROM candidate_memories WHERE id = ?", (candidate["id"],)
            ).fetchone() is not None
            with pytest.raises(state_module.MemoryMutationError, match="checkpoint is busy"):
                client.post(
                    f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
                )
            with sqlite3.connect(database) as inspection:
                run = inspection.execute(
                    """SELECT status, last_error FROM retention_cleanup_runs
                       WHERE counts_json LIKE '%\"candidates_to_delete\": 1%'"""
                ).fetchone()
                assert run == ("error", "MemoryMutationError")
                job = inspection.execute(
                    "SELECT fingerprints_json, attempts FROM retention_erasure_jobs"
                ).fetchone()
                assert job is not None
                assert "hmac-sha256:" in str(job[0])
                assert plaintext not in str(job)
                assert int(job[1]) == 1
        finally:
            reader.rollback()
            reader.close()

        retry = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        )
        assert retry.status_code == 200
        assert retry.json()["counts"]["candidates_to_delete"] == 0
        with sqlite3.connect(database) as inspection:
            assert inspection.execute(
                "SELECT COUNT(*) FROM retention_erasure_jobs"
            ).fetchone() == (0,)
            assert inspection.execute(
                """SELECT status, last_error FROM retention_cleanup_runs
                   WHERE counts_json LIKE '%\"candidates_to_delete\": 1%'"""
            ).fetchone() == ("done", "")
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{database}{suffix}")
            if path.exists():
                assert plaintext.encode() not in path.read_bytes()
    finally:
        client.__exit__(None, None, None)


def test_startup_retries_pending_physical_erasure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, headers, library_id = _client(tmp_path)
    closed = False
    try:
        _candidate(client, headers, library_id, "startup-erasure")
        assert client.put(
            f"/api/v1/libraries/{library_id}/retention-policy",
            headers=headers,
            json={
                "inbox_days": 30,
                "candidate_days": 0,
                "recycle_days": 0,
                "early_inbox_cleanup": True,
                "diagnostics_enabled": True,
                "diagnostic_days": 7,
            },
        ).status_code == 200
        assert client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).status_code == 200
        platform_state = cast(
            PlatformState, cast(FastAPI, client.app).state.platform_state
        )

        def busy_checkpoint(_: tuple[bytes, ...]) -> None:
            raise state_module.MemoryMutationError("SQLite WAL checkpoint is busy")

        monkeypatch.setattr(platform_state, "_checkpoint_deleted_plaintext", busy_checkpoint)
        with pytest.raises(state_module.MemoryMutationError, match="checkpoint is busy"):
            client.post(
                f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
            )
        database = tmp_path / "state" / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM retention_erasure_jobs"
            ).fetchone() == (1,)
        client.__exit__(None, None, None)
        closed = True

        restarted_app = create_app(
            Settings(
                state_dir=tmp_path / "state",
                library_roots=(tmp_path,),
                retention_now=NOW,
            )
        )
        with TestClient(restarted_app), sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM retention_erasure_jobs"
            ).fetchone() == (0,)
            assert connection.execute(
                """SELECT status, last_error FROM retention_cleanup_runs
                   WHERE counts_json LIKE '%\"candidates_to_delete\": 1%'"""
            ).fetchone() == ("done", "")
    finally:
        if not closed:
            client.__exit__(None, None, None)


def test_permanent_cleanup_removes_plaintext_from_sqlite_files(tmp_path: Path) -> None:
    candidate_body = "Ticket16PhysicalCandidatePlaintext-6f51706d"
    inbox_body = "Ticket16PhysicalInboxPlaintext-a99dc31e"
    client, headers, library_id = _client(tmp_path)
    try:
        candidate = _candidate(client, headers, library_id, candidate_body)
        database = tmp_path / "state" / "platform.sqlite3"
        assert client.put(
            f"/api/v1/libraries/{library_id}/retention-policy",
            headers=headers,
            json={
                "inbox_days": 0,
                "candidate_days": 0,
                "recycle_days": 0,
                "early_inbox_cleanup": False,
                "diagnostics_enabled": True,
                "diagnostic_days": 7,
            },
        ).status_code == 200
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE candidate_memories SET body = ?, created_at = '2026-09-07 00:00:00' "
                "WHERE id = ?",
                (candidate_body, candidate["id"]),
            )
            connection.execute(
                "UPDATE candidate_audit SET body = ? WHERE candidate_id = ?",
                (candidate_body, candidate["id"]),
            )
            connection.execute(
                """INSERT INTO capture_inbox
                   (event_id, request_hash, session_id, project_id, library_id, turn_id,
                    event_kind, content, occurred_at, received_at)
                   VALUES ('physical-inbox', 'physical-hash', 'physical-session', 'project', ?,
                           'turn', 'user', ?, '2026-09-07T00:00:00Z',
                           '2026-09-07 00:00:00')""",
                (library_id, inbox_body),
            )

        first = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).json()
        assert first["counts"]["capture_inbox"] == 1
        assert first["counts"]["candidates_to_recycle"] == 1
        second = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        ).json()
        assert second["counts"]["candidates_to_delete"] == 1

        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{database}{suffix}")
            if not path.exists():
                continue
            persisted = path.read_bytes()
            assert candidate_body.encode() not in persisted
            assert inbox_body.encode() not in persisted
    finally:
        client.__exit__(None, None, None)


def test_cleanup_allows_plaintext_still_referenced_by_active_candidate(tmp_path: Path) -> None:
    shared_body = "Shared retained plaintext."
    client, headers, library_id = _client(tmp_path)
    try:
        candidate = _candidate(client, headers, library_id, "shared-retained")
        database = tmp_path / "state" / "platform.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE candidate_memories SET body = ? WHERE id = ?",
                (f"# Decision\n\nPreserve: {shared_body}\n", candidate["id"]),
            )
            connection.execute(
                """INSERT INTO capture_inbox
                   (event_id, request_hash, session_id, project_id, library_id, turn_id,
                    event_kind, content, occurred_at, received_at, consolidated_at)
                   VALUES ('shared-inbox', 'shared-hash', 'shared-session', 'project', ?,
                           'turn', 'user', ?, '2026-09-06T00:00:00Z',
                           '2026-09-06 00:00:00', '2026-09-06 00:01:00')""",
                (library_id, shared_body),
            )

        cleanup = client.post(
            f"/api/v1/libraries/{library_id}/retention-cleanup", headers=headers
        )
        assert cleanup.status_code == 200
        assert cleanup.json()["counts"]["capture_inbox"] == 1
        assert client.get(
            f"/api/v1/candidates/{candidate['id']}", headers=headers
        ).status_code == 200
    finally:
        client.__exit__(None, None, None)
