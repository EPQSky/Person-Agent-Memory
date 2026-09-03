from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory import markdown_index
from personal_agent_memory import state as state_module
from personal_agent_memory.app import create_app
from personal_agent_memory.config import ConfigurationError, Settings
from personal_agent_memory.git_history import GitHistoryError, GitRepository
from personal_agent_memory.markdown_index import (
    MarkdownChunk,
    StoredChunk,
    chunk_markdown,
    match_chunk_ids,
)
from personal_agent_memory.state import MemoryMutationError, PlatformState


def auth_headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def require_relative_worktrees(repository: Path) -> None:
    help_result = subprocess.run(
        ["git", "-C", str(repository), "worktree", "add", "-h"],
        check=False,
        capture_output=True,
        text=True,
    )
    if "--[no-]relative-paths" not in help_result.stdout + help_result.stderr:
        pytest.skip("installed Git does not support worktree add --relative-paths")


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
        assert "URLSearchParams" not in response.text
        assert "Service ready" in response.text
        assert "Memory libraries" in response.text
        assert "Register library" in response.text
        assert "Project bindings" in response.text
        assert "Bind project" in response.text
        assert "Associate worktree" in response.text
        assert "Search project memory" in response.text
        assert "/api/v1/search" in response.text

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
    connection.commit()
    connection.close()

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(library_root,)))
    ) as client:
        source.write_text(original.replace("Old heading", "New heading"), encoding="utf-8")
        response = client.post(
            "/api/v1/libraries/legacy-library/scan", headers=auth_headers(state_dir)
        )
        assert response.status_code == 200
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
        assert unchanged["changed"] == 0
        assert unchanged["unchanged"] == 2
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
        failed = client.post(f"/api/v1/libraries/{library['id']}/scan", headers=headers)
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
        assert recovered.json()["unchanged"] == 1
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
        response = client.post(f"/api/v1/libraries/{library['id']}/scan", headers=headers)
        assert response.status_code == 200
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
        assert unbound == {"status": "unbound", "query": "ExactNeedle42", "results": []}

        original_chunk_id = hit["chunk_id"]
        source.write_text("# 新决策\n\nProject alpha uses ExactNeedle42.\n", encoding="utf-8")
        heading_rescan = client.post(
            f"/api/v1/libraries/{first['id']}/scan", headers=headers
        )
        assert heading_rescan.status_code == 200
        renamed = client.post(
            "/api/v1/search",
            headers=headers,
            json={"cwd": str(first_project), "query": "ExactNeedle42"},
        ).json()["results"][0]
        assert renamed["chunk_id"] == original_chunk_id
        assert renamed["heading"] == "新决策"

        source.write_text("# 决策\n\nProject alpha uses UpdatedNeedle77.\n", encoding="utf-8")
        rescanned = client.post(
            f"/api/v1/libraries/{first['id']}/scan", headers=headers
        ).json()
        assert rescanned["changed"] == 1
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
        removed = client.post(
            f"/api/v1/libraries/{first['id']}/scan", headers=headers
        ).json()
        assert removed["removed"] == 1
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
            rescanned = client.post(
                f"/api/v1/libraries/{library_id}/scan", headers=headers
            )
            assert rescanned.status_code == 200
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
