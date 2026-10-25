from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jiuwen_memory.common.type_def import Scope

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.context_package import TokenCounter
from personal_agent_memory.memory_kernel import SearchMemoryRequest
from personal_agent_memory.regions import MemoryRegion


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir))


def _long_region_id(index: int, *, namespace: str = "healthy") -> str:
    high_entropy = (
        hashlib.sha256(f"{namespace}-{index}-a".encode()).hexdigest()
        + hashlib.sha256(f"{namespace}-{index}-b".encode()).hexdigest()
    )[:117]
    return f"{index:02d}-{high_entropy}"


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _create_region(
    client: TestClient,
    headers: dict[str, str],
    kind: str,
    region_id: str,
    *,
    session_id: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "kind": kind,
        "name": region_id,
        "region_id": region_id,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json=payload,
    )
    assert response.status_code == 201, response.text


def _bind_project(
    client: TestClient,
    headers: dict[str, str],
    project_root: Path,
    region_id: str,
) -> None:
    response = client.post(
        "/api/v1/project-bindings",
        headers=headers,
        json={"project_root": str(project_root), "region_id": region_id},
    )
    assert response.status_code == 201, response.text


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
            "source_references": [f"source:{region_id}"],
            "actor": "cross-region-test",
            "idempotency_key": (
                "cross-region:"
                + hashlib.sha256(f"{region_id}\0{content}".encode()).hexdigest()
            ),
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _search(
    client: TestClient,
    headers: dict[str, str],
    payload: dict[str, object],
    *,
    endpoint: str = "/api/v1/search",
) -> dict[str, object]:
    response = client.post(endpoint, headers=headers, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for_search_result(
    client: TestClient,
    headers: dict[str, str],
    payload: dict[str, object],
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        package = _search(client, headers, payload)
        if package["results"] or time.monotonic() >= deadline:
            assert package["results"]
            return package
        time.sleep(0.1)


def _mcp_search(
    client: TestClient,
    headers: dict[str, str],
    arguments: dict[str, object],
) -> dict[str, object]:
    response = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": "cross-region-search",
            "method": "tools/call",
            "params": {"name": "memory_search", "arguments": arguments},
        },
    )
    assert response.status_code == 200, response.text
    rpc = response.json()
    assert rpc["result"]["isError"] is False, rpc
    structured = rpc["result"]["structuredContent"]
    assert isinstance(structured, dict)
    return structured


def _group(payload: dict[str, object], region_id: str) -> dict[str, object]:
    groups = payload["region_groups"]
    region_ids = payload["region_ids"]
    assert isinstance(groups, list)
    assert isinstance(region_ids, list)
    index = region_ids.index(region_id)
    return next(group for group in groups if group["region_index"] == index)


def _result_ids(payload: dict[str, object]) -> set[str]:
    results = payload["results"]
    assert isinstance(results, list)
    return {str(item["document_id"]) for item in results}


def test_cross_region_requires_explicit_selection_and_rejects_invalid_regions(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        for kind, region_id in (
            ("user", "user"),
            ("topic", "topic"),
            ("project", "project"),
            ("project", "unselected"),
        ):
            _create_region(client, headers, kind, region_id)
        _create_region(
            client,
            headers,
            "session",
            "session-region",
            session_id="cross-region-session",
        )
        _bind_project(client, headers, project_root, "project")
        for region_id in ("user", "topic", "project", "unselected"):
            _write_region_memory(
                client, headers, region_id, f"SELECTION-NEEDLE {region_id} memory"
            )

        invalid_payloads = (
            {"query": "SELECTION-NEEDLE"},
            {"query": "SELECTION-NEEDLE", "region_ids": []},
            {"query": "SELECTION-NEEDLE", "region_ids": ["user", "user"]},
            {"query": "SELECTION-NEEDLE", "region_ids": ["missing"]},
            {
                "query": "SELECTION-NEEDLE",
                "cwd": str(project_root),
                "region_ids": ["user", "topic"],
            },
            {"query": "SELECTION-NEEDLE", "region_ids": ["session-region"]},
            {"query": "SELECTION-NEEDLE", "region_ids": ["user", "session-region"]},
        )
        for payload in invalid_payloads:
            response = client.post("/api/v1/search", headers=headers, json=payload)
            assert response.status_code == 422, (payload, response.text)

        default = _search(
            client,
            headers,
            {
                "cwd": str(project_root),
                "query": "SELECTION-NEEDLE",
                "token_budget": 10_000,
            },
        )
        assert default["scope"]["kind"] == "project_cwd"
        assert {item["region_id"] for item in default["results"]} == {"project"}
        assert "region_groups" not in default
        assert "cross_region" not in default["budget"]

        selected = _search(
            client,
            headers,
            {
                "region_ids": ["user", "topic"],
                "query": "SELECTION-NEEDLE",
                "token_budget": 10_000,
            },
        )
        assert selected["scope"]["kind"] == "explicit_regions"
        assert selected["region_ids"] == ["user", "topic"]
        assert [
            selected["region_ids"][group["region_index"]]
            for group in selected["region_groups"]
        ] == ["user", "topic"]
        assert {item["region_id"] for item in selected["results"]} == {"user", "topic"}
        assert all("unselected memory" not in item["content"] for item in selected["results"])

        invalid_budget = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_ids": ["user", "topic"],
                "query": "SELECTION-NEEDLE",
                "cross_region_budget": {
                    "graph_fraction": 0.8,
                    "source_structure_fraction": 0.3,
                },
            },
        )
        assert invalid_budget.status_code == 422


def test_cross_region_default_budget_supports_maximum_region_selection(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        region_ids: list[str] = []
        for index in range(20):
            region_id = _long_region_id(index)
            assert len(region_id) == 120
            region_ids.append(region_id)
            _create_region(client, headers, "topic", region_id)

        package = _search(
            client,
            headers,
            {
                "region_ids": region_ids,
                "query": "NO-MATCH-MAXIMUM-REGIONS",
                "token_budget": 10_000,
            },
        )

        assert package["results"] == []
        groups = package["region_groups"]
        assert [
            package["region_ids"][group["region_index"]] for group in groups
        ] == region_ids
        cross = package["budget"]["cross_region"]
        assert (
            cross["source_structure_used_tokens"]
            <= cross["source_structure_limit_tokens"]
            == 1_000
        )
        assert package["budget"]["used_tokens"] <= 10_000


def test_cross_region_default_budget_supports_maximum_degraded_region_selection(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        region_ids: list[str] = []
        for index in range(20):
            region_id = _long_region_id(index, namespace="degraded")
            assert len(region_id) == 120
            region_ids.append(region_id)
            _create_region(client, headers, "topic", region_id)
            _write_region_memory(
                client,
                headers,
                region_id,
                f"DEGRADED-MAXIMUM {region_id} durable evidence",
            )

        package = _search(
            client,
            headers,
            {
                "region_ids": region_ids,
                "query": "NO-MATCH-DEGRADED-MAXIMUM",
                "token_budget": 10_000,
            },
        )

        groups = package["region_groups"]
        assert [
            package["region_ids"][group["region_index"]] for group in groups
        ] == region_ids
        assert all(group["status"] == "degraded" for group in groups)
        cross = package["budget"]["cross_region"]
        assert (
            cross["source_structure_used_tokens"]
            <= cross["source_structure_limit_tokens"]
            == 1_000
        )
        assert package["budget"]["used_tokens"] <= 10_000


def test_cross_region_groups_sources_and_conflicts_match_between_rest_and_mcp(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        for kind, region_id in (
            ("user", "user"),
            ("topic", "topic"),
            ("project", "project"),
        ):
            _create_region(client, headers, kind, region_id)
        _write_region_memory(
            client,
            headers,
            "user",
            "COLOR-CONFLICT preferred theme is dark mode",
        )
        _write_region_memory(
            client,
            headers,
            "topic",
            "COLOR-CONFLICT preferred theme is light mode",
        )
        _write_region_memory(
            client, headers, "project", "COLOR-CONFLICT project implementation note"
        )
        arguments: dict[str, object] = {
            "region_ids": ["user", "topic", "project"],
            "query": "COLOR-CONFLICT",
            "limit": 20,
            "token_budget": 10_000,
        }
        rest = _search(client, headers, arguments)
        mcp = _mcp_search(client, headers, arguments)

        assert rest["results"] == mcp["results"]
        assert rest["region_groups"] == mcp["region_groups"]
        assert rest["budget"] == mcp["budget"]
        assert rest["budget"]["cross_region"]["user_fraction"] == 0.30
        assert rest["budget"]["cross_region"]["selected_region_fraction"] == 0.20
        assert rest["budget"]["cross_region"]["graph_fraction"] == 0.20
        assert rest["budget"]["cross_region"]["source_structure_fraction"] == 0.10

        contents = {str(item["content"]) for item in rest["results"]}
        assert any("dark mode" in content for content in contents)
        assert any("light mode" in content for content in contents)
        assert len(rest["results"]) >= 2
        for result in rest["results"]:
            region_id = str(result["region_id"])
            assert result["source_references"] == [f"source:{region_id}"]
            group = _group(rest, region_id)
            assert result["document_id"] in group["result_ids"]
            assert group["source_references"] == [f"source:{region_id}"]
            assert group["result_count"] == 1

        tools_response = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert tools_response.status_code == 200
        tools = tools_response.json()["result"]["tools"]
        search_tool = next(tool for tool in tools if tool["name"] == "memory_search")
        assert "cross-region" in search_tool["description"]
        properties = search_tool["inputSchema"]["properties"]
        assert "region_ids" in properties
        assert "cross_region_budget" in properties


def test_spike_cross_space_retrieval_source_marking_and_budget_clipping(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _create_region(client, headers, "user", "user")
        for region_id in ("topic", "project", "unselected"):
            _create_region(client, headers, "topic", region_id)
        long_content = "SPIKE-BUDGET cross-space evidence " + ("durable preference " * 2_000)
        for region_id in ("user", "topic", "project", "unselected"):
            _write_region_memory(client, headers, region_id, long_content)
        _wait_for_search_result(
            client,
            headers,
            {
                "region_ids": ["user", "topic", "project"],
                "query": "SPIKE-BUDGET",
                "limit": 20,
                "token_budget": 10_000,
            },
        )

        budget = {
            "user_fraction": 0.25,
            "selected_region_fraction": 0.15,
            "graph_fraction": 0.20,
            "source_structure_fraction": 0.35,
        }
        package = _search(
            client,
            headers,
            {
                "region_ids": ["user", "topic", "project"],
                "query": "SPIKE-BUDGET",
                "limit": 20,
                "token_budget": 4_096,
                "cross_region_budget": budget,
            },
        )
        cross = package["budget"]["cross_region"]
        assert package["budget"]["effective_tokens"] == 4_096
        assert cross["user_limit_tokens"] == 1_024
        assert cross["selected_region_limit_tokens"] == 614
        assert cross["graph_limit_tokens"] == 819
        assert cross["source_structure_limit_tokens"] == 1_433
        assert cross["source_structure_used_tokens"] <= 1_433
        assert cross["graph_used_tokens"] <= 819
        assert cross["results_truncated"] is True
        assert {
            package["region_ids"][group["region_index"]]
            for group in package["region_groups"]
        } == {"user", "topic", "project"}
        included_groups = [
            group for group in package["region_groups"] if group["result_ids"]
        ]
        assert len(included_groups) >= 2
        for group in included_groups:
            region_id = package["region_ids"][group["region_index"]]
            assert group["content_tokens"] <= group["limit_tokens"]
            assert group["source_references"] == [f"source:{region_id}"]
        assert all(
            str(result["region_id"]) != "unselected"
            for result in package["results"]
        )
        counter = TokenCounter.for_model("gpt-4o-mini")
        encoded = json.dumps(package, ensure_ascii=False, separators=(",", ":"))
        assert counter.count(encoded) <= 4_096
        assert package["budget"]["telemetry_accounting"] == "exact_serialized_v1"
        assert package["budget"]["used_tokens"] == counter.count(encoded)


def test_cross_region_graph_expansion_and_graph_degradation_keep_region_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _create_region(client, headers, "user", "user")
        _create_region(client, headers, "topic", "graph-topic")
        _write_region_memory(client, headers, "user", "GRAPH-CROSS user alpha")
        first_id = _write_region_memory(
            client,
            headers,
            "graph-topic",
            "GRAPH-CROSS graph alpha beta seed",
        )
        second_id = _write_region_memory(
            client,
            headers,
            "graph-topic",
            "beta gamma related graph content",
        )
        for region_id in ("user", "graph-topic"):
            rebuilt = client.post(
                f"/api/v1/libraries/{region_id}/graph-index/rebuild",
                headers=headers,
            )
            assert rebuilt.status_code == 202, rebuilt.text
            assert rebuilt.json()["status"] == "ready", rebuilt.text

        arguments: dict[str, object] = {
            "region_ids": ["user", "graph-topic"],
            "query": "alpha",
            "limit": 20,
            "token_budget": 10_000,
            "cross_region_budget": {"source_structure_fraction": 0.25},
        }
        package = _search(client, headers, arguments)
        graph_results = [
            item
            for item in package["results"]
            if item["classification"] == "graph_expansion"
        ]
        assert any(item["document_id"] == second_id for item in graph_results)
        graph_result = next(
            item for item in graph_results if item["document_id"] == second_id
        )
        assert graph_result["region_id"] == "graph-topic"
        assert graph_result["source_references"] == ["source:graph-topic"]
        graph_group = _group(package, "graph-topic")
        assert second_id in graph_group["result_ids"]
        assert graph_group["source_references"] == ["source:graph-topic"]
        assert package["budget"]["cross_region"]["graph_used_tokens"] <= package["budget"][
            "cross_region"
        ]["graph_limit_tokens"]

        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None
        original_traverse_many = graph_store.traverse_many

        def fail_topic_traversal(
            scope: Scope, node_ids: list[str], *, depth: int, limit: int
        ):
            if scope.space == "topic:graph-topic":
                raise RuntimeError("graph-topic expansion unavailable")
            return original_traverse_many(scope, node_ids, depth=depth, limit=limit)

        monkeypatch.setattr(graph_store, "traverse_many", fail_topic_traversal)
        degraded = _search(client, headers, arguments)
        user_group = _group(degraded, "user")
        topic_group = _group(degraded, "graph-topic")
        assert user_group["status"] != "unavailable"
        assert topic_group["status"] == "degraded"
        assert "graph_unavailable" in topic_group["degradation"]
        assert degraded["graph_index_status"] == "degraded"
        assert "graph_unavailable" in degraded["degradation"]
        assert first_id in _result_ids(degraded)
        assert any(item["region_id"] == "user" for item in degraded["results"])


def test_cross_region_retrieval_failure_returns_other_regions_with_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _create_region(client, headers, "user", "user")
        _create_region(client, headers, "topic", "broken-topic")
        user_id = _write_region_memory(
            client, headers, "user", "REGION-FAILURE user memory remains"
        )
        _write_region_memory(
            client, headers, "broken-topic", "REGION-FAILURE topic memory"
        )
        kernel = client.app.state.memory_kernel
        original_search_scope = kernel._search_scope

        def fail_topic(
            bundle: Any,
            request: SearchMemoryRequest,
            scope: Scope,
            *,
            region: MemoryRegion | None = None,
        ) -> dict[str, object]:
            if region is not None and region.region_id == "broken-topic":
                raise RuntimeError("topic retrieval unavailable")
            return original_search_scope(bundle, request, scope, region=region)

        monkeypatch.setattr(kernel, "_search_scope", fail_topic)
        package = _search(
            client,
            headers,
            {
                "region_ids": ["user", "broken-topic"],
                "query": "REGION-FAILURE",
                "token_budget": 10_000,
            },
        )
        assert package["results"]
        assert user_id in _result_ids(package)
        assert _group(package, "user")["status"] != "unavailable"
        assert _group(package, "broken-topic")["status"] == "unavailable"
        assert "retrieval_unavailable" in package["degradation"]
        assert "region_unavailable" in package["degradation"]
        assert "region_unavailable:broken-topic" in package["degradation"]
