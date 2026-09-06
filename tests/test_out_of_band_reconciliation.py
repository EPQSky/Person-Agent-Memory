from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.state import PlatformState


def auth_headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def register(client: TestClient, state_dir: Path, root: Path) -> tuple[str, dict[str, str]]:
    headers = auth_headers(state_dir)
    response = client.post(
        "/api/v1/libraries",
        headers=headers,
        json={"path": str(root), "kind": "project"},
    )
    assert response.status_code == 201
    return response.json()["id"], headers


def pending(client: TestClient, headers: dict[str, str], library_id: str) -> list[dict]:
    response = client.get(
        f"/api/v1/libraries/{library_id}/out-of-band-changes", headers=headers
    )
    assert response.status_code == 200
    return response.json()


def test_external_edit_is_detected_without_index_or_git_propagation(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    note = root / "note.md"
    original = "# Fact\n\nBaselineNeedle.\n"
    external = "# Fact\n\nExternalNeedle.\n"
    note.write_text(original, encoding="utf-8")

    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        note.write_text(external, encoding="utf-8")
        detected = client.post(
            f"/api/v1/libraries/{library_id}/scan", headers=headers
        )
        assert detected.status_code == 200
        assert detected.json()["detected"] == 1
        change = pending(client, headers, library_id)[0]
        assert change["kind"] == "edit"
        assert change["base"] == original
        assert change["platform"] == original
        assert change["external"] == external
        assert "ExternalNeedle" in change["base_to_external_diff"]
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "ExternalNeedle"},
        ).json()
        assert search["results"] == []

        imported = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{change['id']}/resolve",
            headers=headers,
            json={"action": "import", "operation_id": "import-edit"},
        )
        assert imported.status_code == 200
        assert imported.json()["source_version"] == hashlib.sha256(external.encode()).hexdigest()
        assert pending(client, headers, library_id) == []
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "ExternalNeedle"},
        ).json()["results"]


def test_move_and_delete_can_be_restored_without_freezing_other_documents(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    moved = root / "move.md"
    deleted = root / "delete.md"
    unaffected = root / "unaffected.md"
    moved.write_text("# Move\n\nMoveNeedle.\n", encoding="utf-8")
    deleted.write_text("# Delete\n\nDeleteNeedle.\n", encoding="utf-8")
    unaffected.write_text("# Good\n\nGoodNeedle.\n", encoding="utf-8")

    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        moved.rename(root / "renamed.md")
        deleted.unlink()
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        changes = {change["base_path"]: change for change in pending(client, headers, library_id)}
        assert changes["move.md"]["kind"] == "move"
        assert changes["delete.md"]["kind"] == "delete"
        assert client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "unaffected.md"},
        ).status_code == 200
        unaffected_loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "unaffected.md"},
        ).json()
        unaffected_edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "unaffected.md",
                "content": "# Good\n\nGoodNeedle updated.\n",
                "expected_source_version": unaffected_loaded["source_version"],
                "operation_id": "edit-unaffected",
                "actor_type": "user",
                "source": "test",
            },
        )
        assert unaffected_edit.status_code == 200
        assert len(pending(client, headers, library_id)) == 2
        for path in ("move.md", "delete.md"):
            response = client.post(
                f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes[path]['id']}/resolve",
                headers=headers,
                json={"action": "restore", "operation_id": f"restore-{path}"},
            )
            assert response.status_code == 200
        assert (root / "move.md").is_file()
        assert not (root / "renamed.md").exists()
        assert (root / "delete.md").is_file()


def test_pending_change_survives_restart_and_requires_final_content_after_second_edit(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    note = root / "note.md"
    note.write_text("# Fact\n\nBase.\n", encoding="utf-8")
    settings = Settings(state_dir=state_dir, library_roots=(root.parent,))
    with TestClient(create_app(settings)) as client:
        library_id, headers = register(client, state_dir, root)
        note.write_text("# Fact\n\nExternal one.\n", encoding="utf-8")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
    note.write_text("# Fact\n\nExternal two.\n", encoding="utf-8")
    with TestClient(create_app(settings)) as client:
        headers = auth_headers(state_dir)
        deadline = time.monotonic() + 2
        changes: list[dict] = []
        while time.monotonic() < deadline:
            changes = pending(client, headers, library_id)
            if changes and changes[0]["status"] == "conflict":
                break
            time.sleep(0.05)
        assert changes[0]["status"] == "conflict"
        blocked = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes[0]['id']}/resolve",
            headers=headers,
            json={"action": "import", "operation_id": "missing-final"},
        )
        assert blocked.status_code == 409
        resolved = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes[0]['id']}/resolve",
            headers=headers,
            json={
                "action": "import",
                "operation_id": "explicit-final",
                "final_content": "# Fact\n\nChosen final.\n",
            },
        )
        assert resolved.status_code == 200
        assert note.read_text(encoding="utf-8") == "# Fact\n\nChosen final.\n"


def test_commit_that_persists_then_raises_returns_durable_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    note = root / "note.md"
    note.write_text("# Fact\n\nBase.\n", encoding="utf-8")
    real_connect = sqlite3.connect

    class RaiseAfterResolutionCommit(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if type(self).injected:
                return
            table = self.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'out_of_band_changes'"
            ).fetchone()
            if table is not None and self.execute(
                "SELECT COUNT(*) FROM memory_operations WHERE source LIKE 'out-of-band:%'"
            ).fetchone() != (0,):
                type(self).injected = True
                raise sqlite3.OperationalError("injected commit result ambiguity")

    def connect(*args, **kwargs):
        kwargs["factory"] = RaiseAfterResolutionCommit
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        note.write_text("# Fact\n\nImported.\n", encoding="utf-8")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        change = pending(client, headers, library_id)[0]
        response = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{change['id']}/resolve",
            headers=headers,
            json={"action": "import", "operation_id": "ambiguous-import"},
        )
        assert response.status_code == 200
        assert pending(client, headers, library_id) == []
        assert note.read_text(encoding="utf-8") == "# Fact\n\nImported.\n"
        git_dir = real_connect(state_dir / "platform.sqlite3").execute(
            "SELECT git_dir FROM library_git_repositories WHERE library_id = ?", (library_id,)
        ).fetchone()[0]
        assert subprocess.run(
            ["git", f"--git-dir={git_dir}", "show", "HEAD:note.md"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout == "# Fact\n\nImported.\n"


def test_resolving_one_change_keeps_other_pending_content_out_of_indexes(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    first = root / "first.md"
    second = root / "second.md"
    first.write_text("# First\n\nFirstBaseNeedle.\n", encoding="utf-8")
    second.write_text("# Second\n\nSecondBaseNeedle.\n", encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        first.write_text("# First\n\nFirstExternalNeedle.\n", encoding="utf-8")
        second.write_text("# Second\n\nSecondExternalNeedle.\n", encoding="utf-8")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        changes = {item["base_path"]: item for item in pending(client, headers, library_id)}
        resolved = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes['first.md']['id']}/resolve",
            headers=headers,
            json={"action": "import", "operation_id": "import-first-only"},
        )
        assert resolved.status_code == 200
        assert [item["base_path"] for item in pending(client, headers, library_id)] == ["second.md"]
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "FirstExternalNeedle"},
        ).json()["results"]
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "SecondExternalNeedle"},
        ).json()["results"] == []
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "SecondBaseNeedle"},
        ).json()["results"]


def test_second_external_edit_during_resolution_is_preserved(
    tmp_path: Path, monkeypatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    note = root / "note.md"
    note.write_text("# Fact\n\nBase.\n", encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        note.write_text("# Fact\n\nExternal one.\n", encoding="utf-8")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        change = pending(client, headers, library_id)[0]
        original_verify = type(client.app.state.platform_state)._verify_external_change

        def edit_after_verify(self, library_id_value, change_payload, documents, selected_path):
            original_verify(
                self, library_id_value, change_payload, documents, selected_path
            )
            note.write_text("# Fact\n\nExternal two.\n", encoding="utf-8")

        monkeypatch.setattr(
            type(client.app.state.platform_state),
            "_verify_external_change",
            edit_after_verify,
        )
        response = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{change['id']}/resolve",
            headers=headers,
            json={"action": "import", "operation_id": "raced-import"},
        )
        assert response.status_code == 409
        assert note.read_text(encoding="utf-8") == "# Fact\n\nExternal two.\n"
        assert pending(client, headers, library_id)


def test_ambiguous_identical_moves_are_visible_conflicts(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    content = "# Shared\n\nIdenticalMoveNeedle.\n"
    (root / "one.md").write_text(content, encoding="utf-8")
    (root / "two.md").write_text(content, encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        (root / "one.md").rename(root / "moved-a.md")
        (root / "two.md").rename(root / "moved-b.md")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        changes = pending(client, headers, library_id)
        assert {item["base_path"] for item in changes} == {"one.md", "two.md"}
        assert all(item["kind"] == "move" and item["status"] == "conflict" for item in changes)
        assert all(
            set(item["external_candidates"]) == {"moved-a.md", "moved-b.md"}
            for item in changes
        )
        assert all(item["base_to_external_diff"] is None for item in changes)
        blocked = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes[0]['id']}/resolve",
            headers=headers,
            json={"action": "import", "operation_id": "ambiguous-without-final"},
        )
        assert blocked.status_code == 409


def test_bulk_accept_is_rejected_without_propagating_external_content(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    note = root / "note.md"
    original = "# Fact\n\nOriginalBulkNeedle.\n"
    external = "# Fact\n\nExternalBulkNeedle.\n"
    note.write_text(original, encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        note.write_text(external, encoding="utf-8")

        rejected = client.post(
            f"/api/v1/libraries/{library_id}/scan?accept_external=true",
            headers=headers,
        )

        assert rejected.status_code == 422
        assert "bulk external acceptance is disabled" in rejected.json()["detail"]
        assert pending(client, headers, library_id) == []
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        assert note.read_text(encoding="utf-8") == external
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "ExternalBulkNeedle"},
        ).json()["results"] == []


def test_ambiguous_moves_require_and_preserve_one_to_one_candidate_selection(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    content = "# Shared\n\nIdenticalCandidateNeedle.\n"
    (root / "one.md").write_text(content, encoding="utf-8")
    (root / "two.md").write_text(content, encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        (root / "one.md").rename(root / "moved-a.md")
        (root / "two.md").rename(root / "moved-b.md")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        changes = {item["base_path"]: item for item in pending(client, headers, library_id)}

        missing = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes['one.md']['id']}/resolve",
            headers=headers,
            json={
                "action": "import",
                "operation_id": "missing-candidate",
                "final_content": "# One\n\nChosen one.\n",
            },
        )
        assert missing.status_code == 422
        invalid = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes['one.md']['id']}/resolve",
            headers=headers,
            json={
                "action": "import",
                "operation_id": "invalid-candidate",
                "external_path": "other.md",
                "final_content": "# One\n\nChosen one.\n",
            },
        )
        assert invalid.status_code == 422

        first = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes['one.md']['id']}/resolve",
            headers=headers,
            json={
                "action": "import",
                "operation_id": "choose-a",
                "external_path": "moved-a.md",
                "final_content": "# One\n\nChosen one.\n",
            },
        )
        assert first.status_code == 200
        duplicate = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes['two.md']['id']}/resolve",
            headers=headers,
            json={
                "action": "import",
                "operation_id": "choose-a-again",
                "external_path": "moved-a.md",
                "final_content": "# Two\n\nChosen two.\n",
            },
        )
        assert duplicate.status_code == 422
        second = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes['two.md']['id']}/resolve",
            headers=headers,
            json={
                "action": "import",
                "operation_id": "choose-b",
                "external_path": "moved-b.md",
                "final_content": "# Two\n\nChosen two.\n",
            },
        )
        assert second.status_code == 200
        assert pending(client, headers, library_id) == []
        assert not (root / "one.md").exists()
        assert not (root / "two.md").exists()
        assert (root / "moved-a.md").read_text(encoding="utf-8") == (
            "# One\n\nChosen one.\n"
        )
        assert (root / "moved-b.md").read_text(encoding="utf-8") == (
            "# Two\n\nChosen two.\n"
        )
        assert first.json()["path"] == "moved-a.md"
        assert second.json()["path"] == "moved-b.md"

        database = sqlite3.connect(state_dir / "platform.sqlite3")
        try:
            stored_paths = {
                str(row[0])
                for row in database.execute(
                    "SELECT path FROM memory_documents WHERE library_id = ?",
                    (library_id,),
                )
            }
            operation_paths = {
                str(row[0])
                for row in database.execute(
                    "SELECT document_path FROM memory_operations "
                    "WHERE operation_id IN ('choose-a', 'choose-b')",
                )
            }
            git_dir = database.execute(
                "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
                (library_id,),
            ).fetchone()[0]
        finally:
            database.close()
        assert stored_paths == {"moved-a.md", "moved-b.md"}
        assert operation_paths == {"moved-a.md", "moved-b.md"}
        tree_paths = set(
            subprocess.run(
                ["git", f"--git-dir={git_dir}", "ls-tree", "--name-only", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        assert {"moved-a.md", "moved-b.md"} <= tree_paths
        assert {"one.md", "two.md"}.isdisjoint(tree_paths)


def test_ambiguous_moves_restore_selected_candidates_one_to_one(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    content = "# Shared\n\nRestoreAmbiguousNeedle.\n"
    (root / "one.md").write_text(content, encoding="utf-8")
    (root / "two.md").write_text(content, encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        (root / "one.md").rename(root / "moved-a.md")
        (root / "two.md").rename(root / "moved-b.md")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        changes = {item["base_path"]: item for item in pending(client, headers, library_id)}

        missing = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes['one.md']['id']}/resolve",
            headers=headers,
            json={"action": "restore", "operation_id": "restore-missing-candidate"},
        )
        assert missing.status_code == 422
        first = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{changes['one.md']['id']}/resolve",
            headers=headers,
            json={
                "action": "restore",
                "operation_id": "restore-candidate-a",
                "external_path": "moved-a.md",
            },
        )
        assert first.status_code == 200
        assert first.json()["path"] == "one.md"
        remaining = pending(client, headers, library_id)
        assert len(remaining) == 1
        assert remaining[0]["base_path"] == "two.md"
        assert remaining[0]["external_candidates"] == ["moved-b.md"]
        assert remaining[0]["external_path"] == "moved-b.md"

        second = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{remaining[0]['id']}/resolve",
            headers=headers,
            json={
                "action": "restore",
                "operation_id": "restore-candidate-b",
                "external_path": "moved-b.md",
            },
        )
        assert second.status_code == 200
        assert second.json()["path"] == "two.md"
        assert pending(client, headers, library_id) == []
        assert (root / "one.md").read_text(encoding="utf-8") == content
        assert (root / "two.md").read_text(encoding="utf-8") == content
        assert not (root / "moved-a.md").exists()
        assert not (root / "moved-b.md").exists()


def test_single_candidate_move_rejects_recreated_base_for_import_and_restore(
    tmp_path: Path,
) -> None:
    for action in ("import", "restore"):
        state_dir = tmp_path / action / "state"
        root = tmp_path / action / "libraries" / "project"
        root.mkdir(parents=True)
        original = "# Move\n\nOriginalMoveNeedle.\n"
        concurrent = "# Move\n\nConcurrentReplacementNeedle.\n"
        (root / "note.md").write_text(original, encoding="utf-8")
        app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
        with TestClient(app) as client:
            library_id, headers = register(client, state_dir, root)
            history_before = client.get(
                f"/api/v1/libraries/{library_id}/history", headers=headers
            ).json()
            (root / "note.md").rename(root / "moved.md")
            client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
            change = pending(client, headers, library_id)[0]
            assert change["external_candidates"] == ["moved.md"]

            (root / "note.md").write_text(concurrent, encoding="utf-8")
            response = client.post(
                f"/api/v1/libraries/{library_id}/out-of-band-changes/"
                f"{change['id']}/resolve",
                headers=headers,
                json={"action": action, "operation_id": f"recreated-base-{action}"},
            )

            assert response.status_code == 409
            assert (root / "note.md").read_text(encoding="utf-8") == concurrent
            assert (root / "moved.md").read_text(encoding="utf-8") == original
            assert pending(client, headers, library_id) == [change]
            assert client.get(
                f"/api/v1/libraries/{library_id}/history", headers=headers
            ).json() == history_before


def test_degraded_ambiguous_move_rejects_recreated_base(tmp_path: Path) -> None:
    for action in ("import", "restore"):
        state_dir = tmp_path / action / "state"
        root = tmp_path / action / "libraries" / "project"
        root.mkdir(parents=True)
        original = "# Shared\n\nDegradedMoveNeedle.\n"
        concurrent = "# Shared\n\nConcurrentDegradedReplacement.\n"
        (root / "one.md").write_text(original, encoding="utf-8")
        (root / "two.md").write_text(original, encoding="utf-8")
        app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
        with TestClient(app) as client:
            library_id, headers = register(client, state_dir, root)
            (root / "one.md").rename(root / "moved-a.md")
            (root / "two.md").rename(root / "moved-b.md")
            client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
            changes = {
                item["base_path"]: item
                for item in pending(client, headers, library_id)
            }
            first = client.post(
                f"/api/v1/libraries/{library_id}/out-of-band-changes/"
                f"{changes['one.md']['id']}/resolve",
                headers=headers,
                json={
                    "action": "import",
                    "operation_id": f"degrade-ambiguous-candidates-{action}",
                    "external_path": "moved-a.md",
                    "final_content": "# One\n\nChosen first.\n",
                },
            )
            assert first.status_code == 200
            remaining = pending(client, headers, library_id)[0]
            assert remaining["external_candidates"] == ["moved-b.md"]
            assert remaining["external_path"] == "moved-b.md"
            (root / "two.md").write_text(concurrent, encoding="utf-8")

            blocked = client.post(
                f"/api/v1/libraries/{library_id}/out-of-band-changes/"
                f"{remaining['id']}/resolve",
                headers=headers,
                json={
                    "action": action,
                    "operation_id": f"degraded-recreated-base-{action}",
                },
            )

            assert blocked.status_code == 409
            assert (root / "two.md").read_text(encoding="utf-8") == concurrent
            assert (root / "moved-b.md").read_text(encoding="utf-8") == original
            assert pending(client, headers, library_id) == [remaining]


def test_failed_ambiguous_resolution_restores_all_pending_rows(
    tmp_path: Path, monkeypatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    original = "# Shared\n\nCompensatedAmbiguousNeedle.\n"
    (root / "one.md").write_text(original, encoding="utf-8")
    (root / "two.md").write_text(original, encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        (root / "one.md").rename(root / "moved-a.md")
        (root / "two.md").rename(root / "moved-b.md")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        changes_before = pending(client, headers, library_id)
        selected = next(item for item in changes_before if item["base_path"] == "one.md")

        monkeypatch.setattr(
            type(client.app.state.platform_state),
            "_persisted_reconciliation",
            lambda *args, **kwargs: False,
        )
        failed = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/"
            f"{selected['id']}/resolve",
            headers=headers,
            json={
                "action": "restore",
                "operation_id": "fail-after-sibling-consumption",
                "external_path": "moved-a.md",
            },
        )

        assert failed.status_code == 422
        assert pending(client, headers, library_id) == changes_before
        assert not (root / "one.md").exists()
        assert not (root / "two.md").exists()
        assert (root / "moved-a.md").read_text(encoding="utf-8") == original
        assert (root / "moved-b.md").read_text(encoding="utf-8") == original
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        database = sqlite3.connect(state_dir / "platform.sqlite3")
        try:
            indexed_paths = {
                str(row[0])
                for row in database.execute(
                    "SELECT path FROM memory_documents WHERE library_id = ?",
                    (library_id,),
                )
            }
            operation = database.execute(
                "SELECT 1 FROM memory_operations WHERE operation_id = ?",
                ("fail-after-sibling-consumption",),
            ).fetchone()
        finally:
            database.close()
        assert indexed_paths == {"one.md", "two.md"}
        assert operation is None


def test_sensitive_external_content_is_withheld_and_requires_safe_final_content(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    note = root / "note.md"
    note.write_text("# Fact\n\nSafe baseline.\n", encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        note.write_text("# Secret\n\nAKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        change = pending(client, headers, library_id)[0]
        assert change["external"] is None
        assert change["external_withheld"] is True
        assert change["base_to_external_diff"] is None
        blocked = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{change['id']}/resolve",
            headers=headers,
            json={"action": "import", "operation_id": "withheld-import"},
        )
        assert blocked.status_code == 409
        for operation_id, final_content in (
            ("withheld-empty", ""),
            ("withheld-whitespace", " \n\t "),
        ):
            blank = client.post(
                f"/api/v1/libraries/{library_id}/out-of-band-changes/{change['id']}/resolve",
                headers=headers,
                json={
                    "action": "import",
                    "operation_id": operation_id,
                    "final_content": final_content,
                },
            )
            assert blank.status_code == 409
            assert note.read_text(encoding="utf-8").endswith("AKIAIOSFODNN7EXAMPLE\n")
            assert pending(client, headers, library_id)[0]["id"] == change["id"]
        restored = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{change['id']}/resolve",
            headers=headers,
            json={"action": "restore", "operation_id": "withheld-restore"},
        )
        assert restored.status_code == 200
        assert note.read_text(encoding="utf-8") == "# Fact\n\nSafe baseline.\n"


def test_reconciliation_scan_and_index_snapshot_share_the_library_lock(
    tmp_path: Path, monkeypatch
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    note = root / "note.md"
    original = "# Fact\n\nPlatform baseline.\n"
    edited = "# Fact\n\nPlatform edit completed.\n"
    note.write_text(original, encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        loaded = client.get(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            params={"path": "note.md"},
        ).json()
        state = client.app.state.platform_state
        state._last_reconciliation_check = time.monotonic() + 60
        original_scan = __import__(
            "personal_agent_memory.state", fromlist=["scan_markdown"]
        ).scan_markdown
        scan_captured = threading.Event()
        edit_finished = threading.Event()
        edit_errors: list[BaseException] = []
        intercepted = False

        def delayed_scan(*args, **kwargs):
            nonlocal intercepted
            result = original_scan(*args, **kwargs)
            if not intercepted and Path(args[0]) == root:
                intercepted = True
                scan_captured.set()
                assert not edit_finished.wait(0.25)
            return result

        monkeypatch.setattr("personal_agent_memory.state.scan_markdown", delayed_scan)

        def edit_from_separate_connection() -> None:
            assert scan_captured.wait(5)
            other = PlatformState(state_dir / "platform.sqlite3", (root.parent,))
            other.connection = sqlite3.connect(other.database_path)
            other.connection.execute("PRAGMA foreign_keys=ON")
            other.sensitive_dedupe_key = state.sensitive_dedupe_key
            try:
                other.edit_document(
                    library_id,
                    "note.md",
                    edited,
                    loaded["source_version"],
                    "concurrent-platform-edit",
                    "user",
                    "test",
                )
            except BaseException as error:
                edit_errors.append(error)
            finally:
                other.connection.close()
                other.connection = None
                edit_finished.set()

        editor = threading.Thread(target=edit_from_separate_connection)
        editor.start()
        reconciled = client.post(
            f"/api/v1/libraries/{library_id}/scan", headers=headers
        )
        editor.join(5)
        assert not editor.is_alive()
        assert edit_errors == []
        assert reconciled.status_code == 200
        assert reconciled.json()["detected"] == 0
        assert pending(client, headers, library_id) == []
        assert note.read_text(encoding="utf-8") == edited


def test_restore_ignores_final_content_and_web_only_sends_it_for_import(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "libraries" / "project"
    root.mkdir(parents=True)
    note = root / "note.md"
    platform = "# Fact\n\nPlatform version.\n"
    note.write_text(platform, encoding="utf-8")
    app = create_app(Settings(state_dir=state_dir, library_roots=(root.parent,)))
    with TestClient(app) as client:
        library_id, headers = register(client, state_dir, root)
        note.write_text("# Fact\n\nExternal one.\n", encoding="utf-8")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        note.write_text("# Fact\n\nExternal two.\n", encoding="utf-8")
        client.post(f"/api/v1/libraries/{library_id}/scan", headers=headers)
        change = pending(client, headers, library_id)[0]
        assert change["status"] == "conflict"
        restored = client.post(
            f"/api/v1/libraries/{library_id}/out-of-band-changes/{change['id']}/resolve",
            headers=headers,
            json={
                "action": "restore",
                "operation_id": "restore-ignores-final",
                "final_content": "# Fact\n\nMust not win.\n",
            },
        )
        assert restored.status_code == 200
        assert restored.json()["resolution"] == "restore"
        assert note.read_text(encoding="utf-8") == platform

    node = shutil.which("node")
    if node is None:
        raise AssertionError("Node.js is required for the Web reconciliation regression")
    script_path = (
        Path(__file__).parents[1]
        / "src"
        / "personal_agent_memory"
        / "static"
        / "editor_history.js"
    )
    program = """
const api = require(process.argv[1]);
const change = {id: 'change-one', status: 'conflict', external_withheld: false};
const restore = api.reconciliationRequest(
  'library-one', change, 'restore', 'restore-one', 'external text'
);
const imported = api.reconciliationRequest(
  'library-one', change, 'import', 'import-one', 'chosen text', 'moved.md'
);
const ambiguousRestore = api.reconciliationRequest(
  'library-one', change, 'restore', 'restore-moved', 'ignored', 'moved.md'
);
console.log(JSON.stringify({restore, imported, ambiguousRestore}));
"""
    result = subprocess.run(
        [node, "-e", program, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    requests = json.loads(result.stdout)
    assert requests["restore"]["body"] == {
        "action": "restore",
        "operation_id": "restore-one",
    }
    assert requests["imported"]["body"] == {
        "action": "import",
        "operation_id": "import-one",
        "final_content": "chosen text",
        "external_path": "moved.md",
    }
    assert requests["ambiguousRestore"]["body"] == {
        "action": "restore",
        "operation_id": "restore-moved",
        "external_path": "moved.md",
    }
