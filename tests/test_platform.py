from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import ConfigurationError, Settings
from personal_agent_memory.state import PlatformState


def auth_headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def test_first_start_creates_protected_key_and_authenticated_surfaces(tmp_path: Path) -> None:
    app = create_app(Settings(state_dir=tmp_path))

    with TestClient(app) as client:
        key = (tmp_path / "api-key").read_text(encoding="utf-8").strip()
        assert key
        assert stat.S_IMODE((tmp_path / "api-key").stat().st_mode) == 0o600
        assert client.get("/health/live").status_code == 401
        assert client.get("/api/v1/status").status_code == 401
        assert (
            client.get("/mcp/health", headers={"Authorization": "Bearer wrong"}).status_code
            == 401
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
        assert client.get(
            "/api/v1/status", headers={"Authorization": "Bearer "}
        ).status_code == 401
        assert client.get(
            "/api/v1/status", headers={"Authorization": f"Bearer {generated_key}"}
        ).status_code == 200


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
        new_status = client.get(
            "/api/v1/status", headers={"Authorization": f"Bearer {new_key}"}
        )
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

        assert client.get("/api/v1/status").status_code == 401
        authenticated = client.get(
            "/api/v1/status", headers={"Authorization": f"Bearer {key}"}
        )
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
        listed = client.get(
            "/api/v1/libraries", headers=auth_headers(state_dir)
        ).json()[0]
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
        sync = client.get(
            f"/mcp/libraries/{registered['id']}/sync-status", headers=headers
        ).json()
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
            item["id"]: item
            for item in client.get("/api/v1/libraries", headers=headers).json()
        }
        sync = client.get(
            f"/mcp/libraries/{first['id']}/sync-status", headers=headers
        ).json()
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
            item["id"]: item
            for item in client.get("/api/v1/libraries", headers=headers).json()
        }
        sync = client.get(
            f"/mcp/libraries/{source['id']}/sync-status", headers=headers
        ).json()
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
