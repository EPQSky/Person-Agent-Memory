from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir))


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _create_region(
    client: TestClient,
    headers: dict[str, str],
    kind: str,
    name: str,
    *,
    region_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"kind": kind, "name": name}
    if region_id is not None:
        payload["region_id"] = region_id
    if session_id is not None:
        payload["session_id"] = session_id
    response = client.post("/api/v1/regions", headers=headers, json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _bind_project(
    client: TestClient,
    headers: dict[str, str],
    project_root: Path,
    region_id: str,
) -> dict[str, object]:
    response = client.post(
        "/api/v1/project-bindings",
        headers=headers,
        json={"project_root": str(project_root), "region_id": region_id},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _write_region_memory(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    content: str,
) -> str:
    response = client.post(
        "/api/v1/memories",
        headers=headers,
        json={
            "region_id": region_id,
            "content": content,
            "memory_type": "domain_fact",
            "source_references": ["test:regions"],
            "actor": "test",
            "idempotency_key": f"region-{region_id}-{content}",
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _search(
    client: TestClient,
    headers: dict[str, str],
    query: str,
    *,
    cwd: Path | None = None,
    region_ids: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"query": query, "limit": 20}
    if cwd is not None:
        payload["cwd"] = str(cwd)
    if region_ids is not None:
        payload["region_ids"] = region_ids
    response = client.post("/api/v1/search", headers=headers, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_regions_create_list_read_tree_and_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        user = _create_region(client, headers, "user", "我的记忆", region_id="user")
        topic = _create_region(client, headers, "topic", "Rust 学习", region_id="rust")
        project = _create_region(client, headers, "project", "平台项目", region_id="platform")
        session = _create_region(
            client,
            headers,
            "session",
            "当前会话",
            region_id="session-one",
            session_id="session-one",
        )

        assert user["space"] == "user"
        assert topic["space"] == "topic:rust"
        assert project["space"] == "project:platform"
        assert session["space"] == "chat"
        assert session["session_id"] == "session-one"

        listed = client.get("/api/v1/regions", headers=headers)
        assert listed.status_code == 200
        assert {item["region_id"] for item in listed.json()} == {
            "user",
            "rust",
            "platform",
            "session-one",
        }
        read = client.get("/api/v1/regions/rust", headers=headers)
        assert read.status_code == 200
        assert read.json()["name"] == "Rust 学习"

        tree = client.get(
            "/api/v1/regions/tree",
            headers=headers,
            params={"session_id": "temporary-session"},
        )
        assert tree.status_code == 200
        tree_payload = tree.json()
        assert tree_payload["root"]["region_id"] == "user"
        assert [item["region_id"] for item in tree_payload["topics"]] == ["rust"]
        assert [item["region_id"] for item in tree_payload["projects"]] == ["platform"]
        assert tree_payload["current_session"]["region_id"] == "session:temporary-session"
        assert tree_payload["current_session"]["temporary"] is True

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        persisted = client.get("/api/v1/regions", headers=headers).json()
        assert {item["region_id"] for item in persisted} == {
            "user",
            "rust",
            "platform",
            "session-one",
        }


def test_project_and_worktree_bindings_require_explicit_association_and_restart(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    main_root = tmp_path / "main"
    main_root.mkdir()
    subprocess.run(["git", "init", "-q", str(main_root)], check=True)
    subprocess.run(
        ["git", "-C", str(main_root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(main_root), "config", "user.name", "Test"], check=True)
    (main_root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(main_root), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(main_root), "commit", "-qm", "initial"], check=True)
    worktree_root = main_root / ".worktrees" / "feature"
    worktree_root.parent.mkdir()
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
            str(worktree_root),
        ],
        check=True,
    )
    (worktree_root / "src").mkdir()

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        project = _create_region(client, headers, "project", "主项目", region_id="main")
        binding = _bind_project(client, headers, main_root, "main")

        bound = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(main_root)},
        )
        assert bound.status_code == 200
        assert bound.json()["region_id"] == "main"
        assert bound.json()["space"] == "project:main"

        before = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(worktree_root)},
        )
        assert before.status_code == 200
        assert before.json()["status"] == "unbound"

        associated = client.post(
            "/api/v1/project-bindings/worktrees",
            headers=headers,
            json={
                "worktree_root": str(worktree_root),
                "main_project_binding_id": str(binding["binding_id"]),
            },
        )
        assert associated.status_code == 201, associated.text
        association = associated.json()
        assert association["kind"] == "worktree"
        assert association["region_id"] == project["region_id"]

        inherited = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(worktree_root / "src")},
        )
        assert inherited.status_code == 200
        assert inherited.json()["region_id"] == "main"
        assert inherited.json()["binding_kind"] == "worktree"

        _create_region(client, headers, "project", "替换项目", region_id="main-two")
        updated = client.put(
            f"/api/v1/project-bindings/{binding['binding_id']}",
            headers=headers,
            json={"region_id": "main-two"},
        )
        assert updated.status_code == 200, updated.text
        inherited_update = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(worktree_root / "src")},
        )
        assert inherited_update.json()["region_id"] == "main-two"

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        restarted = client.get(
            "/api/v1/project-bindings/resolve",
            headers=headers,
            params={"cwd": str(worktree_root / "src")},
        )
        assert restarted.status_code == 200
        assert restarted.json()["region_id"] == "main-two"


def test_project_search_defaults_to_current_region_and_explicit_recall_selects_only_requested(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _create_region(client, headers, "user", "用户", region_id="user")
        _create_region(client, headers, "topic", "专题", region_id="topic")
        _create_region(client, headers, "project", "项目 A", region_id="project-a")
        _create_region(client, headers, "project", "项目 B", region_id="project-b")
        _bind_project(client, headers, project_a, "project-a")
        _bind_project(client, headers, project_b, "project-b")

        _write_region_memory(client, headers, "user", "SHARED-MARKER user only")
        _write_region_memory(client, headers, "topic", "SHARED-MARKER topic only")
        _write_region_memory(client, headers, "project-a", "SHARED-MARKER project A only")
        _write_region_memory(client, headers, "project-b", "SHARED-MARKER project B only")

        default = _search(client, headers, "SHARED-MARKER", cwd=project_a)
        assert default["scope"]["kind"] == "project_cwd"
        assert default["scope"]["space"] == "project:project-a"
        assert {item["region_id"] for item in default["results"]} == {"project-a"}
        assert all("project B" not in item["content"] for item in default["results"])

        selected = _search(
            client,
            headers,
            "SHARED-MARKER",
            region_ids=["user", "topic"],
        )
        assert selected["scope"]["kind"] == "explicit_regions"
        assert selected["region_ids"] == ["user", "topic"]
        assert {item["region_id"] for item in selected["results"]} == {"user", "topic"}
        assert all("project A" not in item["content"] for item in selected["results"])
        assert all("project B" not in item["content"] for item in selected["results"])

        missing = client.get("/api/v1/regions/missing", headers=headers)
        assert missing.status_code == 404

        duplicate = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "project", "name": "重复", "region_id": "project-a"},
        )
        assert duplicate.status_code == 409


def test_management_ui_exposes_region_tree_and_explicit_bindings(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path / "state")) as client:
        response = client.get("/")
        assert response.status_code == 200
        body = response.text
        assert 'id="region-list"' in body
        assert "/api/v1/regions" in body
        assert "/api/v1/project-bindings" in body
        assert "/api/v1/promotions" in body
        assert "/api/v1/search" in body
        assert "/document/delete-preview" in body
        assert "/history" in body
        assert "/api/v1/candidates" not in body
        assert "candidate" not in body.lower()
        assert "editor_history.js" not in body


def test_region_write_idempotency_is_isolated_by_region(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _create_region(client, headers, "user", "用户", region_id="user")
        _create_region(client, headers, "topic", "专题", region_id="topic")
        payload = {
            "region_id": "user",
            "content": "IDEMPOTENT-REGION marker",
            "memory_type": "domain_fact",
            "source_references": ["test:regions"],
            "actor": "test",
            "idempotency_key": "shared-key",
        }
        first = client.post("/api/v1/memories", headers=headers, json=payload)
        second = client.post("/api/v1/memories", headers=headers, json=payload)
        assert first.status_code == 201, first.text
        assert second.status_code == 201, second.text
        assert first.json()["id"] == second.json()["id"]
        assert {first.json()["status"], second.json()["status"]} == {"created", "existing"}

        other_region = client.post(
            "/api/v1/memories",
            headers=headers,
            json={**payload, "region_id": "topic"},
        )
        assert other_region.status_code == 201, other_region.text
        assert other_region.json()["id"] != first.json()["id"]

        same_named_library = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                **payload,
                "region_id": None,
                "library_id": "user",
                "content": "IDEMPOTENT-LIBRARY marker",
            },
        )
        assert same_named_library.status_code == 201, same_named_library.text
        assert same_named_library.json()["id"] != first.json()["id"]


def test_project_binding_missing_region_is_not_found(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    bound_root = tmp_path / "bound-project"
    missing_root = tmp_path / "missing-project"
    bound_root.mkdir()
    missing_root.mkdir()

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _create_region(client, headers, "project", "已绑定项目", region_id="bound")
        binding = _bind_project(client, headers, bound_root, "bound")

        create_missing = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(missing_root), "region_id": "missing"},
        )
        assert create_missing.status_code in {404, 422}
        assert create_missing.status_code != 500

        update_missing = client.put(
            f"/api/v1/project-bindings/{binding['binding_id']}",
            headers=headers,
            json={"region_id": "missing"},
        )
        assert update_missing.status_code in {404, 422}
        assert update_missing.status_code != 500

        bindings = client.get("/api/v1/project-bindings", headers=headers).json()
        assert len(bindings) == 1
        assert bindings[0]["region_id"] == "bound"


def test_user_identifier_is_reserved_and_does_not_shadow_region_tree(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        project_conflict = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "project", "name": "错误占用", "region_id": "user"},
        )
        assert project_conflict.status_code == 422
        assert "reserved" in project_conflict.json()["detail"]

        user = _create_region(client, headers, "user", "用户记忆", region_id="user")
        assert user["kind"] == "user"
        assert user["space"] == "user"

        tree = client.get("/api/v1/regions/tree", headers=headers)
        assert tree.status_code == 200
        assert tree.json()["root"]["region_id"] == "user"
        assert tree.json()["root"]["kind"] == "user"
