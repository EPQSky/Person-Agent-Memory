from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3
import stat
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import ConfigurationError, Settings
from personal_agent_memory.state import PlatformState


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
        assert created.json()["sync_status"] == "not_started"
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
            "sync_status": "not_started",
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
