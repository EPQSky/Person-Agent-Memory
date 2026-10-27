from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.mem2_kernel import Mem2MemoryKernel
from personal_agent_memory.memory_kernel import MemoryKernelPort


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def test_mem2_is_the_only_kernel_and_platform_state_is_not_assembled(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    app = create_app(Settings(state_dir=state_dir))

    assert isinstance(app.state.memory_kernel, MemoryKernelPort)
    assert app.state.memory_kernel.kind == "mem2"
    assert not hasattr(app.state, "platform_state")

    with TestClient(app) as client:
        status = client.get("/api/v1/status", headers=_headers(state_dir))
        assert status.status_code == 200
        payload = status.json()
        assert payload["memory_kernel"] == "mem2"
        assert payload["memory_kernel_status"] in {"ready", "degraded"}
        assert "database" not in payload
        assert "background_worker" not in payload

    assert isinstance(app.state.memory_kernel, Mem2MemoryKernel)


def test_removed_legacy_and_candidate_interfaces_are_not_registered(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(create_app(Settings(state_dir=state_dir))) as client:
        headers = _headers(state_dir)
        removed = (
            ("GET", "/api/v1/libraries", None),
            ("POST", "/api/v1/libraries", {"path": str(tmp_path), "kind": "project"}),
            ("POST", "/api/v1/libraries/library-one/scan", None),
            ("GET", "/api/v1/libraries/library-one/out-of-band-changes", None),
            ("GET", "/api/v1/jobs/1", None),
            ("GET", "/api/v1/libraries/library-one/ignore-rules", None),
            ("GET", "/api/v1/candidates", None),
            ("GET", "/api/v1/candidates/candidate-one", None),
            ("POST", "/mcp/candidates", {}),
            ("GET", "/mcp/candidates", None),
            ("GET", "/mcp/libraries", None),
            ("GET", "/mcp/libraries/library-one/sync-status", None),
            ("GET", "/api/v1/sensitive-quarantine", None),
            ("DELETE", "/api/v1/diagnostics", None),
        )
        for method, path, body in removed:
            response = client.request(method, path, headers=headers, json=body)
            assert response.status_code == 404, (method, path, response.text)
        assert not hasattr(client.app.state.memory_kernel, "clear_diagnostics")


def test_runtime_manifests_do_not_reference_removed_kernel_surfaces() -> None:
    root = Path(__file__).parents[1]
    active_files = (
        root / "compose.yaml",
        root / "Dockerfile",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "install.sh",
        root / "uninstall.sh",
        root / "README.md",
        root / "docs/user/release.md",
    )
    forbidden = (
        "personal_agent_memory.fake_model",
        "personal_agent_memory.state",
        "personal_agent_memory.graph_adapter",
        "personal_agent_memory.markdown_index",
        "personal_agent_memory.git_history",
        "tests/docker_acceptance.py",
        "JiuwenMemory==0.1.2",
        "FROM runtime AS acceptance",
        ".[acceptance]",
        'name = "selenium"',
    )
    for path in active_files:
        content = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in content, (path, value)


def test_mcp_exposes_only_the_mem2_memory_surface(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(create_app(Settings(state_dir=state_dir))) as client:
        response = client.post(
            "/mcp",
            headers=_headers(state_dir),
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )

    assert response.status_code == 200
    tools = response.json()["result"]["tools"]
    assert {tool["name"] for tool in tools} == {"memory_search", "memory_get"}
    assert not {"candidate_create", "candidate_list", "library_list", "sync_status"} & {
        tool["name"] for tool in tools
    }


def test_incompatible_state_makes_memory_routes_unavailable_without_fallback(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    mem2_dir = state_dir / "mem2"
    mem2_dir.mkdir(parents=True)
    (mem2_dir / "VERSION").write_text("jiuwen-memory-0.1.0\n", encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir))) as client:
        headers = _headers(state_dir)
        health = client.get("/mcp/health", headers=headers).json()
        assert health["memory_kernel"] == "mem2"
        assert health["memory_kernel_status"] == "unavailable"
        regions = client.get("/api/v1/regions", headers=headers)
        assert regions.status_code == 503
        assert "reset the state directory explicitly" in regions.json()["detail"]
        assert not hasattr(client.app.state, "platform_state")
