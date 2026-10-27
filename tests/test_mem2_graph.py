from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jiuwen_memory.api import Modality, legacy_request_context
from jiuwen_memory.common.errors import ConflictError, NotFoundError
from jiuwen_memory.common.type_def import Scope
from jiuwen_memory.storage.types import Edge, GraphQuery, Node

import personal_agent_memory.mem2_kernel as mem2_module
from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.memory_kernel import (
    DocumentUpdateRequest,
    EvolveMemoryRequest,
    RetentionCleanupRequest,
    SearchMemoryRequest,
)
from personal_agent_memory.regions import MemoryRegion
from personal_agent_memory.storage.milvus import MilvusLiteGraphStore


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir))


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _region(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json={"kind": "topic", "name": "图测试", "region_id": "graph-region"},
    )
    assert response.status_code == 201, response.text


def _write(
    client: TestClient,
    headers: dict[str, str],
    content: str,
    *,
    key: str,
    source: str,
) -> dict[str, object]:
    response = client.post(
        "/api/v1/memories",
        headers=headers,
        json={
            "region_id": "graph-region",
            "content": content,
            "memory_type": "domain_fact",
            "source_references": [source],
            "actor": "test",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_milvus_graph_store_contract_crud_bfs_scope_cascade_and_restart(
    tmp_path: Path,
) -> None:
    uri = tmp_path / "graph.db"
    first = Scope(org="local", space="topic:first", user="u", agent="a", session="")
    second = Scope(org="local", space="topic:second", user="u", agent="a", session="")
    store = MilvusLiteGraphStore(
        uri=uri,
        collection="graph_nodes_contract",
        edge_collection="graph_edges_contract",
    )
    try:
        store.insert(
            first,
            nodes=[
                Node(id="n1", label="memory", properties={"content": "alpha beta"}),
                Node(id="n2", label="memory", properties={"content": "beta gamma"}),
                Node(id="n3", label="memory", properties={"content": "gamma delta"}),
            ],
            edges=[
                Edge(id="e1", source="n1", target="n2", relation="related"),
                Edge(id="e2", source="n2", target="n3", relation="related"),
            ],
        )
        store.insert(
            second,
            nodes=[
                Node(id="n1", label="memory", properties={"content": "foreign"}),
                Node(id="n2", label="memory", properties={"content": "foreign two"}),
            ],
            edges=[Edge(id="e1", source="n1", target="n2", relation="related")],
        )

        assert {node.id for node in store.get(first, ["n1", "n2", "n3"])} == {
            "n1",
            "n2",
            "n3",
        }
        assert [node.id for node in store.get(second, ["n1"])] == ["n1"]
        assert store.seed_ids(first, {"alpha"}) == ["n1"]
        assert store.seed_ids(second, {"alpha"}) == []

        assert [(node.id, hop) for node, hop in store.traverse(first, "n1", depth=1)] == [("n2", 1)]
        assert [(node.id, hop) for node, hop in store.traverse(first, "n1", depth=2)] == [
            ("n2", 1),
            ("n3", 2),
        ]
        assert [
            node.id
            for node in store.search(
                first,
                GraphQuery(start_id="n1", relation="missing", depth=2),
            )
        ] == []

        # A semantic duplicate with a fresh edge id must not multiply the edge.
        store.insert(
            first,
            edges=[Edge(id="e1-duplicate", source="n1", target="n2", relation="related")],
        )
        assert store.counts(first) == {"nodes": 3, "edges": 2}

        store.update(
            first,
            edges=[Edge(id="e1", source="n1", target="n2", relation="updated")],
        )
        assert [
            node.id
            for node in store.search(
                first,
                GraphQuery(start_id="n1", relation="updated", depth=1),
            )
        ] == ["n2"]
        assert (
            store.search(
                first,
                GraphQuery(start_id="n1", relation="related", depth=1),
            )
            == []
        )
        assert [
            node.id
            for node in store.search(
                second,
                GraphQuery(start_id="n1", relation="related", depth=1),
            )
        ] == ["n2"]

        store.delete(first, edge_ids=["e1"])
        assert store.counts(first)["edges"] == 1
        assert store.counts(second)["edges"] == 1
        assert store.traverse(first, "n1", depth=1) == []
        assert [node.id for node, _ in store.traverse(second, "n1", depth=1)] == ["n2"]

        store.update(first, nodes=[Node(id="n2", label="updated", properties={"x": 1})])
        assert store.get(first, ["n2"])[0].label == "updated"
        with pytest.raises(ConflictError):
            store.insert(first, nodes=[Node(id="n1")])
        with pytest.raises(NotFoundError):
            store.update(first, nodes=[Node(id="missing")])

        store.delete(first, node_ids=["n2"])
        assert store.counts(first)["edges"] == 0
        assert store.traverse(first, "n1", depth=2) == []
        assert store.counts(second) == {"nodes": 2, "edges": 1}
        store.clear(first)
        assert store.counts(first) == {"nodes": 0, "edges": 0}
        assert store.counts(second) == {"nodes": 2, "edges": 1}
    finally:
        store.close()
        store.close()

    reopened = MilvusLiteGraphStore(
        uri=uri,
        collection="graph_nodes_contract",
        edge_collection="graph_edges_contract",
    )
    try:
        assert reopened.counts(first) == {"nodes": 0, "edges": 0}
        assert [node.id for node in reopened.get(second, ["n1", "n2"])] == ["n1", "n2"]
        assert [node.id for node, _ in reopened.traverse(second, "n1", depth=1)] == ["n2"]
    finally:
        reopened.close()


def test_graph_store_empty_session_scope_is_exact(tmp_path: Path) -> None:
    uri = tmp_path / "graph.db"
    base_scope = Scope(
        org="local",
        space="topic:session-isolation",
        user="u",
        agent="a",
        session="",
    )
    session_scope = Scope(
        org="local",
        space="topic:session-isolation",
        user="u",
        agent="a",
        session="session-a",
    )
    store = MilvusLiteGraphStore(
        uri=uri,
        collection="graph_nodes_session",
        edge_collection="graph_edges_session",
    )
    try:
        store.insert(
            session_scope,
            nodes=[
                Node(id="n1", label="memory", properties={}),
                Node(id="n2", label="memory", properties={}),
            ],
            edges=[Edge(id="e1", source="n1", target="n2", relation="related")],
        )
        assert store.node_ids(base_scope) == []
        assert store.counts(base_scope) == {"nodes": 0, "edges": 0}
        store.clear(base_scope)
        assert sorted(store.node_ids(session_scope)) == ["n1", "n2"]
        assert store.counts(session_scope) == {"nodes": 2, "edges": 1}
        long_session = Scope(
            org="local",
            space="topic:session-isolation",
            user="u",
            agent="a",
            session="s" * 256,
        )
        store.insert(
            long_session,
            nodes=[Node(id="long", label="memory", properties={})],
        )
        assert store.node_ids(long_session) == ["long"]
    finally:
        store.close()


def test_graph_projection_expands_search_reports_health_and_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first = _write(
            client,
            headers,
            "Graph nodes share alpha beta memory",
            key="graph-1",
            source="source:graph-1",
        )
        second = _write(
            client,
            headers,
            "Graph expansion shares beta gamma content",
            key="graph-2",
            source="source:graph-2",
        )
        first_id = str(first["id"])
        second_id = str(second["id"])

        status = client.get("/api/v1/status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["graph_index_status"] == "degraded"

        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready", rebuilt.text
        assert rebuilt.json()["job_id"]
        assert rebuilt.json()["job_ids"]

        graph_status = client.get(
            "/api/v1/libraries/graph-region/graph-status",
            headers=headers,
        )
        assert graph_status.status_code == 200, graph_status.text
        assert graph_status.json()["status"] == "ready"
        assert graph_status.json()["library_id"] == "graph-region"
        assert graph_status.json()["total_documents"] == 2
        assert graph_status.json()["projected_documents"] == 2
        assert graph_status.json()["last_error"] == ""
        assert graph_status.json()["updated_at"]
        assert graph_status.json()["nodes"] == 2
        assert graph_status.json()["edges"] == 1

        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "alpha",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        payload = search.json()
        assert payload["graph_index_status"] == "ready"
        results = {str(item["document_id"]): item for item in payload["results"]}
        assert results[first_id]["classification"] == "direct"
        assert results[second_id]["classification"] == "graph_expansion"
        assert results[second_id]["graph_hop"] == 1
        assert results[second_id]["region_id"] == "graph-region"
        assert results[second_id]["user_metadata"]["source_references"] == ["source:graph-2"]

        # Rebuild is deterministic: it clears the projection and recreates it.
        rebuilt_again = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt_again.status_code == 202, rebuilt_again.text
        graph_status = client.get(
            "/api/v1/libraries/graph-region/graph-status",
            headers=headers,
        ).json()
        assert graph_status["nodes"] == 2
        assert graph_status["edges"] == 1

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "alpha",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert any(
            item["document_id"] == second_id and item["classification"] == "graph_expansion"
            for item in search.json()["results"]
        )

        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None

        def fail_traverse(*_args: object, **_kwargs: object) -> list[object]:
            raise RuntimeError("deterministic graph outage")

        monkeypatch.setattr(graph_store, "traverse_many", fail_traverse)
        degraded = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "alpha",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert degraded.status_code == 200, degraded.text
        payload = degraded.json()
        assert payload["graph_index_status"] == "degraded"
        assert "graph_unavailable" in payload["degradation"]
        assert [item["document_id"] for item in payload["results"]] == [first_id]
        assert (
            client.get("/api/v1/libraries/graph-region/graph-status", headers=headers).json()[
                "status"
            ]
            == "degraded"
        )

        def fail_health(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("deterministic graph health outage")

        monkeypatch.setattr(graph_store, "health", fail_health)
        no_match = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "no-such-token",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert no_match.status_code == 200, no_match.text
        assert no_match.json()["results"] == []
        assert no_match.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in no_match.json()["degradation"]


def test_rebuild_reports_degraded_when_association_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "Graph rebuild failure alpha beta",
            key="graph-failure",
            source="source:graph-failure",
        )
        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None

        def fail_insert(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("deterministic graph insert outage")

        monkeypatch.setattr(graph_store, "insert", fail_insert)
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "degraded"
        assert "job_id" in rebuilt.json()
        assert (
            client.get("/api/v1/libraries/graph-region/graph-status", headers=headers).json()[
                "status"
            ]
            == "degraded"
        )


def test_graph_expansion_reserves_capacity_when_direct_results_fill_limit(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        seed = _write(
            client,
            headers,
            "alpha delta epsilon seed",
            key="graph-capacity-seed",
            source="source:graph-capacity-seed",
        )
        for index in range(9):
            _write(
                client,
                headers,
                f"alpha filler {index}",
                key=f"graph-capacity-{index}",
                source=f"source:graph-capacity-{index}",
            )
        target = _write(
            client,
            headers,
            "delta epsilon graph target",
            key="graph-capacity-target",
            source="source:graph-capacity-target",
        )
        assert seed["id"]
        target_id = str(target["id"])

        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"

        kernel = client.app.state.memory_kernel
        raw = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="alpha",
                    limit=10,
                )
            )
        )
        results = raw["results"]
        assert len(results) == 10
        target_result = next(item for item in results if item["document_id"] == target_id)
        assert target_result["classification"] == "graph_expansion"
        assert target_result["graph_hop"] == 1


def test_rebuild_reports_degraded_when_edge_association_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "alpha beta shared association",
            key="graph-edge-failure-1",
            source="source:graph-edge-failure-1",
        )
        _write(
            client,
            headers,
            "alpha beta related association",
            key="graph-edge-failure-2",
            source="source:graph-edge-failure-2",
        )
        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None
        original_insert = graph_store.insert

        def insert_nodes_then_fail_edges(
            scope: Scope,
            nodes: list[Node] | None = None,
            edges: list[Edge] | None = None,
        ) -> None:
            if edges:
                if nodes:
                    original_insert(scope, nodes=nodes)
                raise RuntimeError("deterministic edge insert outage")
            original_insert(scope, nodes=nodes)

        monkeypatch.setattr(graph_store, "insert", insert_nodes_then_fail_edges)
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "degraded"
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        assert status.json()["edges"] == 0
        assert status.json()["last_error"]

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "shared",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]


def test_evolve_edge_failure_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "alpha beta evolve failure one",
            key="graph-evolve-failure-1",
            source="source:graph-evolve-failure-1",
        )
        _write(
            client,
            headers,
            "alpha beta evolve failure two",
            key="graph-evolve-failure-2",
            source="source:graph-evolve-failure-2",
        )
        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None
        original_insert = graph_store.insert

        def insert_nodes_then_fail_edges(
            scope: Scope,
            nodes: list[Node] | None = None,
            edges: list[Edge] | None = None,
        ) -> None:
            if edges:
                if nodes:
                    original_insert(scope, nodes=nodes)
                raise RuntimeError("deterministic evolve edge outage")
            original_insert(scope, nodes=nodes)

        monkeypatch.setattr(graph_store, "insert", insert_nodes_then_fail_edges)
        asyncio.run(kernel.evolve(EvolveMemoryRequest()))
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "alpha",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]


def test_multi_scope_search_keeps_graph_expansion_after_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        kernel = client.app.state.memory_kernel
        first_scope = Scope(
            org="local",
            space="topic:graph-region",
            user="local",
            agent="personal-agent-memory",
            session="",
        )
        second_scope = Scope(
            org="local",
            space="topic:graph-region",
            user="local",
            agent="codex",
            session="",
        )
        region = MemoryRegion(
            region_id="graph-region",
            kind="topic",
            name="图测试",
            space="topic:graph-region",
        )

        def item(
            document_id: str,
            score: float,
            classification: str = "direct",
        ) -> dict[str, object]:
            return {
                "document_id": document_id,
                "score": score,
                "classification": classification,
            }

        first_results = [item(f"first-direct-{index}", 20.0 - index) for index in range(10)]
        second_results = [item(f"second-direct-{index}", 10.0 - index) for index in range(10)] + [
            item("graph-only", 0.0, "graph_expansion")
        ]

        def fake_search_scope(
            _bundle: object,
            _request: SearchMemoryRequest,
            scope: Scope,
            *,
            region: MemoryRegion | None = None,
        ) -> dict[str, object]:
            assert region is not None
            assert region.region_id == "graph-region"
            return {
                "results": (first_results if scope.agent == first_scope.agent else second_results),
                "degradation": [],
                "vector_index_status": "ready",
                "graph_index_status": "ready",
            }

        monkeypatch.setattr(kernel, "_search_scope", fake_search_scope)
        payload = kernel._search_across(
            kernel._require_bundle(),
            SearchMemoryRequest(
                region_id="graph-region",
                query="alpha",
                limit=10,
            ),
            [(first_scope, region), (second_scope, region)],
            scope_kind="explicit_regions",
            region_ids=["graph-region"],
            regions_by_id={"graph-region": region},
        )
        results = payload["results"]
        assert len(results) == 10
        assert [item["document_id"] for item in results].count("graph-only") == 1


def test_forgetting_memories_removes_graph_nodes_before_expansion_limit(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "uniquealpha alpha related seed anchor",
            key="graph-cleanup-seed",
            source="source:graph-cleanup-seed",
        )
        related = [
            _write(
                client,
                headers,
                f"alpha related {index}",
                key=f"graph-cleanup-{index}",
                source=f"source:graph-cleanup-{index}",
            )
            for index in range(11)
        ]
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"

        documents = client.get("/api/v1/libraries/graph-region/documents", headers=headers).json()
        revisions = {str(item["id"]): str(item["version"]) for item in documents}
        related_ids = sorted(str(item["id"]) for item in related)
        active_id = related_ids[-1]
        for index, unit_id in enumerate(related_ids[:-1]):
            revision = revisions[unit_id]
            preview = client.post(
                "/api/v1/libraries/graph-region/document/delete-preview",
                headers=headers,
                json={
                    "path": unit_id,
                    "expected_source_version": revision,
                    "operation_id": f"graph-cleanup-preview-{index}",
                },
            )
            assert preview.status_code == 200, preview.text
            deleted = client.request(
                "DELETE",
                "/api/v1/libraries/graph-region/document",
                headers=headers,
                json={
                    "path": unit_id,
                    "expected_source_version": revision,
                    "preview_token": preview.json()["preview_token"],
                    "operation_id": f"graph-cleanup-delete-{index}",
                    "actor_type": "user",
                    "source": "test-graph-cleanup",
                    "mode": "forget",
                },
            )
            assert deleted.status_code == 200, deleted.text

        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["nodes"] == 2

        kernel = client.app.state.memory_kernel
        raw = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="uniquealpha",
                    limit=10,
                )
            )
        )
        assert any(
            item["document_id"] == active_id and item["classification"] == "graph_expansion"
            for item in raw["results"]
        )

        rebuilt_again = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt_again.status_code == 202, rebuilt_again.text
        assert rebuilt_again.json()["status"] == "ready"
        status_after_rebuild = client.get(
            "/api/v1/libraries/graph-region/graph-status", headers=headers
        )
        assert status_after_rebuild.status_code == 200, status_after_rebuild.text
        assert status_after_rebuild.json()["nodes"] == 2
        raw_after_rebuild = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="uniquealpha",
                    limit=10,
                )
            )
        )
        assert any(
            item["document_id"] == active_id and item["classification"] == "graph_expansion"
            for item in raw_after_rebuild["results"]
        )


def test_downweight_keeps_active_memory_graph_node(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "uniquelead alpha beta seed",
            key="graph-downweight-seed",
            source="source:graph-downweight-seed",
        )
        target = _write(
            client,
            headers,
            "alpha beta retained target",
            key="graph-downweight-target",
            source="source:graph-downweight-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"

        documents = client.get("/api/v1/libraries/graph-region/documents", headers=headers).json()
        revision = next(str(item["version"]) for item in documents if str(item["id"]) == target_id)
        preview = client.post(
            "/api/v1/libraries/graph-region/document/delete-preview",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": revision,
                "operation_id": "graph-downweight-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        downweighted = client.request(
            "DELETE",
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": revision,
                "preview_token": preview.json()["preview_token"],
                "operation_id": "graph-downweight-delete",
                "actor_type": "user",
                "source": "test-graph-downweight",
                "mode": "downweight",
            },
        )
        assert downweighted.status_code == 200, downweighted.text
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["nodes"] == 2

        kernel = client.app.state.memory_kernel
        raw = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="uniquelead",
                    limit=10,
                )
            )
        )
        assert any(
            item["document_id"] == target_id and item["classification"] == "graph_expansion"
            for item in raw["results"]
        )


def test_retention_cleanup_removes_archived_capture_graph_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        session_region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={
                "kind": "session",
                "name": "图保留测试",
                "region_id": "graph-session",
                "session_id": "graph-session",
            },
        )
        assert session_region.status_code == 201, session_region.text
        _write(
            client,
            headers,
            "retentionseed alpha beta seed",
            key="graph-retention-seed",
            source="source:graph-retention-seed",
        )
        target = _write(
            client,
            headers,
            "alpha beta expired capture",
            key="graph-retention-target",
            source="source:graph-retention-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"

        kernel = client.app.state.memory_kernel
        region = kernel._governance_region("graph-region")
        plan = [
            (scope, unit)
            for scope, unit in kernel._region_unit_pairs(
                kernel._require_bundle(),
                region,
            )
            if unit.id == target_id
        ]
        assert len(plan) == 1
        monkeypatch.setattr(kernel, "_retention_plan", lambda _region: plan)
        cleaned = kernel._run_retention_cleanup(RetentionCleanupRequest(library_id="graph-session"))
        assert cleaned["counts"]["captures_archived"] == 1
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["nodes"] == 1


def test_supersede_reprojects_target_and_limit_one_prefers_direct(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        seed = _write(
            client,
            headers,
            "uniquelead alpha beta seed",
            key="graph-supersede-seed",
            source="source:graph-supersede-seed",
        )
        seed_id = str(seed["id"])
        target = _write(
            client,
            headers,
            "alpha beta original target",
            key="graph-supersede-target",
            source="source:graph-supersede-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        target_document = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert target_document.status_code == 200, target_document.text
        updated = client.put(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "content": "alpha beta updated target",
                "expected_source_version": target_document.json()["version"],
                "operation_id": "graph-supersede-update",
                "actor_type": "user",
                "source": "test-graph-supersede",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        updated_id = str(updated.json()["id"])
        assert updated_id != target_id
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        assert status.json()["nodes"] == 2
        assert status.json()["total_documents"] == 2
        assert status.json()["projected_documents"] == 2

        kernel = client.app.state.memory_kernel
        raw = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="uniquelead",
                    limit=10,
                )
            )
        )
        assert any(
            item["document_id"] == updated_id and item["classification"] == "graph_expansion"
            for item in raw["results"]
        )
        limited = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="uniquelead",
                    limit=1,
                )
            )
        )
        assert [item["document_id"] for item in limited["results"]] == [seed_id]
        assert limited["results"][0]["classification"] == "direct"


def test_overwrite_reprojects_edges_after_content_change(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "uniquelead alpha beta seed",
            key="graph-overwrite-seed",
            source="source:graph-overwrite-seed",
        )
        target = _write(
            client,
            headers,
            "alpha beta original target",
            key="graph-overwrite-target",
            source="source:graph-overwrite-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        target_document = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert target_document.status_code == 200, target_document.text
        updated = client.put(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "content": "zzz completely unrelated unicorn",
                "expected_source_version": target_document.json()["version"],
                "operation_id": "graph-overwrite-update",
                "actor_type": "user",
                "source": "test-graph-overwrite",
                "mode": "overwrite",
            },
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["id"] == target_id
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        assert status.json()["nodes"] == 2
        assert status.json()["edges"] == 0

        kernel = client.app.state.memory_kernel
        raw = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="uniquelead",
                    limit=10,
                )
            )
        )
        assert not any(
            item["document_id"] == target_id and item["classification"] == "graph_expansion"
            for item in raw["results"]
        )


def test_stale_graph_content_is_degraded_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    target_id = ""
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "stalelead alpha beta seed",
            key="graph-stale-seed",
            source="source:graph-stale-seed",
        )
        target = _write(
            client,
            headers,
            "alpha beta original target",
            key="graph-stale-target",
            source="source:graph-stale-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        target_document = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert target_document.status_code == 200, target_document.text
        kernel = client.app.state.memory_kernel
        monkeypatch.setattr(kernel, "_reproject_graph_scope", lambda *args, **kwargs: None)
        updated = client.put(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "content": "zzz completely unrelated replacement",
                "expected_source_version": target_document.json()["version"],
                "operation_id": "graph-stale-update",
                "actor_type": "user",
                "source": "test-graph-stale",
                "mode": "overwrite",
            },
        )
        assert updated.status_code == 200, updated.text

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        assert "stale_nodes=1" in status.json()["last_error"]
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "stalelead",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]
        assert not any(
            item["classification"] == "graph_expansion" for item in search.json()["results"]
        )
        recovered = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert recovered.status_code == 202, recovered.text
        assert recovered.json()["status"] == "ready"


def test_missing_or_stale_edges_are_degraded_after_restart(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "edgelead alpha beta seed",
            key="graph-edge-integrity-seed",
            source="source:graph-edge-integrity-seed",
        )
        _write(
            client,
            headers,
            "alpha beta related target",
            key="graph-edge-integrity-target",
            source="source:graph-edge-integrity-target",
        )
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        kernel = client.app.state.memory_kernel
        store = kernel._require_bundle().graph_store
        assert store is not None
        scope = Scope(
            org="local",
            space="topic:graph-region",
            user="local",
            agent="personal-agent-memory",
            session="",
        )
        edges = store.projection_edges(scope)
        assert len(edges) == 1
        store.delete(scope, edge_ids=[edges[0].id])

        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        assert "missing_edges=1" in status.json()["last_error"]
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "edgelead",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]
        assert not any(
            item["classification"] == "graph_expansion" for item in search.json()["results"]
        )

        recovered = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert recovered.status_code == 202, recovered.text
        assert recovered.json()["status"] == "ready"
        edge = store.projection_edges(scope)[0]
        store.update(
            scope,
            edges=[
                Edge(
                    id=edge.id,
                    source=edge.source,
                    target=edge.target,
                    relation="obsolete",
                    properties=edge.properties,
                )
            ],
        )
        stale_status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert stale_status.status_code == 200, stale_status.text
        assert stale_status.json()["status"] == "degraded"
        assert "stale_edges=1" in stale_status.json()["last_error"]
        stale_search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "edgelead",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert stale_search.status_code == 200, stale_search.text
        assert stale_search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in stale_search.json()["degradation"]

        recovered_again = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert recovered_again.status_code == 202, recovered_again.text
        assert recovered_again.json()["status"] == "ready"


@pytest.mark.parametrize("method_name", ["projection_nodes", "projection_edges"])
def test_graph_fingerprint_read_failure_degrades_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        seed = _write(
            client,
            headers,
            "recoverlead alpha beta seed",
            key="graph-fingerprint-seed",
            source="source:graph-fingerprint-seed",
        )
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        kernel = client.app.state.memory_kernel
        store = kernel._require_bundle().graph_store
        assert store is not None

        def fail_projection_read(*args: Any, **kwargs: Any) -> list[Any]:
            raise RuntimeError(f"{method_name} unavailable")

        monkeypatch.setattr(store, method_name, fail_projection_read)
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "recoverlead",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]
        direct = next(
            item for item in search.json()["results"] if item["document_id"] == seed["id"]
        )
        assert direct["classification"] == "direct"


def test_missing_projected_seed_reports_graph_unavailable(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "uniquelead alpha beta seed",
            key="graph-missing-seed",
            source="source:graph-missing-seed",
        )
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        _write(
            client,
            headers,
            "brandnew unprojected memory",
            key="graph-unprojected",
            source="source:graph-unprojected",
        )
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "brandnew",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        payload = search.json()
        assert payload["results"]
        assert payload["graph_index_status"] == "degraded"
        assert payload["degraded"] is True
        assert "graph_unavailable" in payload["degradation"]


def test_graph_projection_read_failure_degrades_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "projectionread alpha direct memory",
            key="graph-projection-read",
            source="source:graph-projection-read",
        )
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        graph_store = client.app.state.memory_kernel._require_bundle().graph_store
        assert graph_store is not None

        def fail_node_ids(*_args: object, **_kwargs: object) -> list[str]:
            raise RuntimeError("deterministic graph projection read outage")

        monkeypatch.setattr(graph_store, "node_ids", fail_node_ids)
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "projectionread",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        payload = search.json()
        assert payload["results"]
        assert payload["graph_index_status"] == "degraded"
        assert payload["degraded"] is True
        assert "graph_unavailable" in payload["degradation"]


def test_purge_cleans_all_cascade_deleted_graph_nodes(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "cascadelead alpha beta seed",
            key="graph-cascade-seed",
            source="source:graph-cascade-seed",
        )
        target = _write(
            client,
            headers,
            "alpha beta cascade target",
            key="graph-cascade-target",
            source="source:graph-cascade-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        target_document = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert target_document.status_code == 200, target_document.text
        preview = client.post(
            "/api/v1/libraries/graph-region/document/delete-preview",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": target_document.json()["version"],
                "operation_id": "graph-cascade-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        forgotten = client.request(
            "DELETE",
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": target_document.json()["version"],
                "preview_token": preview.json()["preview_token"],
                "operation_id": "graph-cascade-forget",
                "actor_type": "user",
                "source": "test-graph-cascade",
                "mode": "forget",
            },
        )
        assert forgotten.status_code == 200, forgotten.text
        restored = client.post(
            "/api/v1/libraries/graph-region/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": target_id,
                "commit": target_id,
                "operation_id": "graph-cascade-restore",
                "actor_type": "user",
                "source": "test-graph-cascade",
            },
        )
        assert restored.status_code == 200, restored.text
        tombstone = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert tombstone.status_code == 200, tombstone.text
        purge_preview = client.post(
            "/api/v1/libraries/graph-region/document/delete-preview",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": tombstone.json()["version"],
                "operation_id": "graph-cascade-purge-preview",
            },
        )
        assert purge_preview.status_code == 200, purge_preview.text
        purged = client.request(
            "DELETE",
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": tombstone.json()["version"],
                "preview_token": purge_preview.json()["preview_token"],
                "operation_id": "graph-cascade-purge",
                "actor_type": "user",
                "source": "test-graph-cascade",
                "mode": "purge",
            },
        )
        assert purged.status_code == 200, purged.text
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        assert status.json()["nodes"] == 1


def test_rebuild_clears_residual_nodes_after_kv_scope_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        target = _write(
            client,
            headers,
            "onlynode alpha beta memory",
            key="graph-empty-scope",
            source="source:graph-empty-scope",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        graph_store = client.app.state.memory_kernel._require_bundle().graph_store
        assert graph_store is not None
        original_delete = graph_store.delete

        def fail_delete(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("deterministic graph delete outage")

        monkeypatch.setattr(graph_store, "delete", fail_delete)
        target_document = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert target_document.status_code == 200, target_document.text
        preview = client.post(
            "/api/v1/libraries/graph-region/document/delete-preview",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": target_document.json()["version"],
                "operation_id": "graph-empty-scope-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        purged = client.request(
            "DELETE",
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": target_document.json()["version"],
                "preview_token": preview.json()["preview_token"],
                "operation_id": "graph-empty-scope-purge",
                "actor_type": "user",
                "source": "test-graph-empty-scope",
                "mode": "purge",
            },
        )
        assert purged.status_code == 200, purged.text
        monkeypatch.setattr(graph_store, "delete", original_delete)
        rebuilt_again = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt_again.status_code == 202, rebuilt_again.text
        assert rebuilt_again.json()["status"] == "ready"
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        assert status.json()["nodes"] == 0


def test_rebuild_clears_residual_session_graph_scope(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        graph_store = bundle.graph_store
        assert graph_store is not None
        stale_scope = Scope(
            org="local",
            space="topic:graph-region",
            user="local",
            agent="personal-agent-memory",
            session="stale-session",
        )
        graph_store.insert(
            stale_scope,
            nodes=[Node(id="stale", label="memory", properties={})],
        )
        before = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert before.status_code == 200, before.text
        assert before.json()["status"] == "degraded"
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        after = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert after.status_code == 200, after.text
        assert after.json()["status"] == "ready"
        assert after.json()["nodes"] == 0


def test_region_projection_error_does_not_leak_to_scope_only_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "leakcheck alpha beta one",
            key="graph-leak-one",
            source="source:graph-leak-one",
        )
        _write(
            client,
            headers,
            "leakcheck alpha beta two",
            key="graph-leak-two",
            source="source:graph-leak-two",
        )
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        graph_store = bundle.graph_store
        assert graph_store is not None
        original_insert = graph_store.insert

        def fail_edges(
            scope: Scope,
            nodes: list[Node] | None = None,
            edges: list[Edge] | None = None,
        ) -> None:
            if edges:
                raise RuntimeError("region projection outage")
            original_insert(scope, nodes=nodes)

        monkeypatch.setattr(graph_store, "insert", fail_edges)
        failed = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert failed.status_code == 202, failed.text
        assert failed.json()["status"] == "degraded"
        monkeypatch.setattr(graph_store, "insert", original_insert)

        standalone_scope = Scope(
            org="local",
            space="library:standalone",
            user="local",
            agent="personal-agent-memory",
            session="",
        )
        assert kernel._graph_projection_error() is None
        assert kernel._graph_ready_status(bundle, standalone_scope, None) == "ready"


def test_graph_errors_are_isolated_by_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        for region_id in ("region-a", "region-b"):
            created = client.post(
                "/api/v1/regions",
                headers=headers,
                json={
                    "kind": "topic",
                    "name": region_id,
                    "region_id": region_id,
                },
            )
            assert created.status_code == 201, created.text
            response = client.post(
                "/api/v1/memories",
                headers=headers,
                json={
                    "region_id": region_id,
                    "content": f"{region_id} alpha beta memory",
                    "memory_type": "domain_fact",
                    "source_references": [f"source:{region_id}"],
                    "actor": "test",
                    "idempotency_key": f"graph-isolation-{region_id}",
                },
            )
            assert response.status_code == 201, response.text
            rebuilt = client.post(
                f"/api/v1/libraries/{region_id}/graph-index/rebuild",
                headers=headers,
            )
            assert rebuilt.status_code == 202, rebuilt.text
            assert rebuilt.json()["status"] == "ready"

        graph_store = client.app.state.memory_kernel._require_bundle().graph_store
        assert graph_store is not None
        original_node_ids = graph_store.node_ids

        def fail_region_a(scope: Scope) -> list[str]:
            if scope.space == "topic:region-a":
                raise RuntimeError("region-a graph read outage")
            return original_node_ids(scope)

        monkeypatch.setattr(graph_store, "node_ids", fail_region_a)
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "region-a",
                "query": "region-a",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert "graph_unavailable" in search.json()["degradation"]
        status_b = client.get("/api/v1/libraries/region-b/graph-status", headers=headers)
        assert status_b.status_code == 200, status_b.text
        assert status_b.json()["status"] == "ready"
        assert "region-a graph read outage" not in status_b.json()["detail"]


def test_history_restore_reprojects_new_active_memory(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "historylead alpha beta seed",
            key="graph-history-seed",
            source="source:graph-history-seed",
        )
        target = _write(
            client,
            headers,
            "alpha beta history original",
            key="graph-history-target",
            source="source:graph-history-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        original = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert original.status_code == 200, original.text
        updated = client.put(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "content": "alpha beta history updated",
                "expected_source_version": original.json()["version"],
                "operation_id": "graph-history-update",
                "actor_type": "user",
                "source": "test-graph-history",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        current_id = str(updated.json()["id"])
        restored = client.post(
            "/api/v1/libraries/graph-region/history/restore",
            headers=headers,
            json={
                "path": current_id,
                "commit": target_id,
                "expected_source_version": updated.json()["version"],
                "operation_id": "graph-history-restore",
                "actor_type": "user",
                "source": "test-graph-history",
            },
        )
        assert restored.status_code == 200, restored.text
        restored_id = str(restored.json()["id"])
        assert restored_id != current_id
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        assert status.json()["nodes"] == 2
        kernel = client.app.state.memory_kernel
        raw = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="historylead",
                    limit=10,
                )
            )
        )
        assert any(
            item["document_id"] == restored_id and item["classification"] == "graph_expansion"
            for item in raw["results"]
        )


def test_restore_forgotten_reprojects_new_active_memory(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "restorelead alpha beta seed",
            key="graph-restore-seed",
            source="source:graph-restore-seed",
        )
        target = _write(
            client,
            headers,
            "alpha beta forgotten target",
            key="graph-restore-target",
            source="source:graph-restore-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        target_document = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert target_document.status_code == 200, target_document.text
        preview = client.post(
            "/api/v1/libraries/graph-region/document/delete-preview",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": target_document.json()["version"],
                "operation_id": "graph-restore-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        forgotten = client.request(
            "DELETE",
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            json={
                "path": target_id,
                "expected_source_version": target_document.json()["version"],
                "preview_token": preview.json()["preview_token"],
                "operation_id": "graph-restore-forget",
                "actor_type": "user",
                "source": "test-graph-restore",
                "mode": "forget",
            },
        )
        assert forgotten.status_code == 200, forgotten.text
        restored = client.post(
            "/api/v1/libraries/graph-region/forgotten/restore",
            headers=headers,
            json={
                "tombstone_id": target_id,
                "commit": target_id,
                "operation_id": "graph-restore-forgotten",
                "actor_type": "user",
                "source": "test-graph-restore",
            },
        )
        assert restored.status_code == 200, restored.text
        restored_id = str(restored.json()["id"])
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        assert status.json()["nodes"] == 2
        kernel = client.app.state.memory_kernel
        raw = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="graph-region",
                    query="restorelead",
                    limit=10,
                )
            )
        )
        assert any(
            item["document_id"] == restored_id and item["classification"] == "graph_expansion"
            for item in raw["results"]
        )


def test_user_region_evolve_projects_graph(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        user_region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "user", "name": "User", "region_id": "user"},
        )
        assert user_region.status_code == 201, user_region.text

        def write_user(content: str, key: str) -> dict[str, object]:
            response = client.post(
                "/api/v1/memories",
                headers=headers,
                json={
                    "region_id": "user",
                    "content": content,
                    "memory_type": "domain_fact",
                    "source_references": [f"source:{key}"],
                    "actor": "test",
                    "idempotency_key": key,
                },
            )
            assert response.status_code == 201, response.text
            return response.json()

        write_user("userlead alpha beta seed", "graph-user-seed")
        target = write_user("alpha beta user target", "graph-user-target")
        target_id = str(target["id"])
        kernel = client.app.state.memory_kernel
        asyncio.run(kernel.evolve(EvolveMemoryRequest()))
        status = client.get("/api/v1/libraries/user/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        assert status.json()["nodes"] == 2
        raw = asyncio.run(
            kernel.search(
                SearchMemoryRequest(
                    region_id="user",
                    query="userlead",
                    limit=10,
                )
            )
        )
        assert any(
            item["document_id"] == target_id and item["classification"] == "graph_expansion"
            for item in raw["results"]
        )


def test_graph_traversal_paginates_edges_and_scans_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = tmp_path / "graph.db"
    scope = Scope(org="local", space="topic:page", user="u", agent="a", session="")
    store = MilvusLiteGraphStore(
        uri=uri,
        collection="graph_nodes_page",
        edge_collection="graph_edges_page",
    )
    try:
        store.insert(
            scope,
            nodes=[Node(id=f"n{index}", label="memory", properties={}) for index in range(5)],
            edges=[
                Edge(
                    id=f"e{index}",
                    source=f"n{index}",
                    target=f"n{index + 1}",
                    relation="related",
                )
                for index in range(4)
            ],
        )
        monkeypatch.setattr(store, "_READ_LIMIT", 2)
        original_all_rows = store._all_rows
        edge_scans = 0

        def count_edge_scans(
            scan_scope: Scope,
            collection: str,
            fields: list[str],
            *,
            page_size: int | None = None,
        ) -> list[dict[str, Any]]:
            nonlocal edge_scans
            if collection == "graph_edges_page":
                edge_scans += 1
            return original_all_rows(
                scan_scope,
                collection,
                fields,
                page_size=page_size,
            )

        monkeypatch.setattr(store, "_all_rows", count_edge_scans)
        traversed = store.traverse_many(
            scope,
            ["n0", "n1"],
            depth=4,
            limit=10,
        )
        assert edge_scans == 1
        assert [node.id for node, _, _ in traversed] == ["n2", "n3", "n4"]
        assert store.counts(scope) == {"nodes": 5, "edges": 4}
    finally:
        store.close()


def test_new_project_capture_scope_is_projected_after_consolidation(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "project", "name": "图项目", "region_id": "graph-project"},
        )
        assert region.status_code == 201, region.text
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "region_id": "graph-project"},
        )
        assert binding.status_code == 201, binding.text
        captured = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "graph-project-event",
                "session_id": "graph-project-session",
                "turn_id": "graph-project-turn",
                "event_kind": "user",
                "content": "PROJECT-GRAPH-MARKER durable project fact",
                "occurred_at": "2026-10-25T00:00:00Z",
                "cwd": str(project_root),
            },
        )
        assert captured.status_code == 202, captured.text
        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "graph-project-session"},
        )
        assert consolidated.status_code == 202, consolidated.text

        status = client.get(
            "/api/v1/libraries/graph-project/graph-status",
            headers=headers,
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready", status.text

        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        graph_store = bundle.graph_store
        assert graph_store is not None
        project_region = kernel._regions.require_region("graph-project")
        region_scopes = kernel._region_scopes(bundle, project_region)
        active_by_scope = [
            (
                scope,
                {
                    unit.id
                    for unit in kernel._units_in_scope(bundle, scope)
                    if unit.lifecycle.value == "active"
                    and unit.system_metadata.get("middle") != "true"
                    and unit.user_metadata.get("restore_pending") is not True
                },
            )
            for scope in region_scopes
        ]
        active_ids = set().union(*(expected_ids for _, expected_ids in active_by_scope))
        assert active_ids
        for scope, expected_ids in active_by_scope:
            assert set(graph_store.node_ids(scope)) == expected_ids
        assert status.json()["total_documents"] == len(active_ids)
        assert status.json()["projected_documents"] == len(active_ids)


def test_consolidation_keeps_graph_error_from_capture_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "project", "name": "多 scope 项目", "region_id": "multi-scope"},
        )
        assert region.status_code == 201, region.text
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "region_id": "multi-scope"},
        )
        assert binding.status_code == 201, binding.text
        for event_id, event_kind, content in (
            (
                "multi-scope-user",
                "user",
                "MULTISCOPE alpha beta capture source",
            ),
            (
                "multi-scope-assistant",
                "assistant",
                "MULTISCOPE alpha beta capture target",
            ),
        ):
            captured = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json={
                    "event_id": event_id,
                    "session_id": "multi-scope-session",
                    "turn_id": "multi-scope-turn",
                    "event_kind": event_kind,
                    "content": content,
                    "occurred_at": "2026-10-25T00:00:00Z",
                    "cwd": str(project_root),
                },
            )
            assert captured.status_code == 202, captured.text

        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None
        original_insert = graph_store.insert

        def fail_capture_scope_edges(
            scope: Scope,
            nodes: list[Node] | None = None,
            edges: list[Edge] | None = None,
        ) -> None:
            if edges and scope.session == "multi-scope-session":
                if nodes:
                    original_insert(scope, nodes=nodes)
                raise RuntimeError("deterministic capture scope edge outage")
            original_insert(scope, nodes=nodes, edges=edges)

        monkeypatch.setattr(graph_store, "insert", fail_capture_scope_edges)
        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "multi-scope-session"},
        )
        assert consolidated.status_code == 202, consolidated.text
        status = client.get("/api/v1/libraries/multi-scope/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded", status.text

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get("/api/v1/libraries/multi-scope/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        assert status.json()["nodes"] == 3
        assert status.json()["edges"] == 0
        assert status.json()["last_error"]
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "cwd": str(project_root),
                "session_id": "multi-scope-session",
                "query": "MULTISCOPE",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        payload = search.json()
        assert payload["graph_index_status"] == "degraded"
        assert "graph_unavailable" in payload["degradation"]
        assert payload["results"]
        assert all(item["classification"] == "direct" for item in payload["results"])


def test_capture_updates_graph_and_keeps_search_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "project", "name": "capture graph", "region_id": "capture-graph"},
        )
        assert region.status_code == 201, region.text
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "region_id": "capture-graph"},
        )
        assert binding.status_code == 201, binding.text
        warmup = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "region_id": "capture-graph",
                "content": "CAPTUREGRAPH durable project baseline",
                "memory_type": "domain_fact",
                "source_references": ["source:capture-graph-baseline"],
                "actor": "test",
                "idempotency_key": "capture-graph-baseline",
            },
        )
        assert warmup.status_code == 201, warmup.text
        rebuilt = client.post(
            "/api/v1/libraries/capture-graph/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready", rebuilt.text

        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        update_started = threading.Event()
        release_update = threading.Event()
        original_associate = kernel._associate_graph_scope
        original_add = bundle.runtime.api.add
        statuses_during_add: list[dict[str, object]] = []

        def add_with_pending_check(*args: object, **kwargs: object) -> list[object]:
            units = original_add(*args, **kwargs)
            scope = args[1]
            if getattr(scope, "session", "") == "capture-graph-session":
                statuses_during_add.append(kernel.graph_status("capture-graph"))
            return units

        def controlled_associate(
            bundle: object,
            scope: Scope,
            *,
            region_id: str | None = None,
        ) -> str | None:
            if scope.session == "capture-graph-session":
                update_started.set()
                assert release_update.wait(timeout=5)
            return original_associate(bundle, scope, region_id=region_id)

        monkeypatch.setattr(bundle.runtime.api, "add", add_with_pending_check)
        monkeypatch.setattr(kernel, "_associate_graph_scope", controlled_associate)
        captured = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "capture-graph-user",
                "session_id": "capture-graph-session",
                "turn_id": "capture-graph-turn",
                "event_kind": "user",
                "content": "CAPTUREGRAPH alpha beta source",
                "occurred_at": "2026-10-25T00:00:00Z",
                "cwd": str(project_root),
            },
        )
        assert captured.status_code == 202, captured.text
        assert update_started.wait(timeout=5)
        assert len(statuses_during_add) == 1
        assert statuses_during_add[0]["status"] == "ready"
        assert statuses_during_add[0]["detail"] == "graph projection update pending"

        pending_status = client.get(
            "/api/v1/libraries/capture-graph/graph-status",
            headers=headers,
        )
        assert pending_status.status_code == 200, pending_status.text
        assert pending_status.json()["status"] == "ready", pending_status.text
        assert pending_status.json()["detail"] == "graph projection update pending"

        pending_search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "cwd": str(project_root),
                "session_id": "capture-graph-session",
                "query": "CAPTUREGRAPH",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert pending_search.status_code == 200, pending_search.text
        pending_payload = pending_search.json()
        assert pending_payload["graph_index_status"] == "ready"
        assert "graph_unavailable" not in pending_payload["degradation"]

        release_update.set()
        kernel._wait_for_scope_graph_updates()
        status = client.get(
            "/api/v1/libraries/capture-graph/graph-status",
            headers=headers,
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready", status.text
        assert status.json()["total_documents"] == 2
        assert status.json()["projected_documents"] == 2

        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "cwd": str(project_root),
                "session_id": "capture-graph-session",
                "query": "CAPTUREGRAPH",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        payload = search.json()
        assert payload["graph_index_status"] == "ready"
        assert "graph_unavailable" not in payload["degradation"]
        assert payload["results"]


def test_consolidation_deadline_queues_remaining_graph_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "project", "name": "deadline graph", "region_id": "deadline-graph"},
        )
        assert region.status_code == 201, region.text
        binding = client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project_root), "region_id": "deadline-graph"},
        )
        assert binding.status_code == 201, binding.text
        for event_id, event_kind, content in (
            ("deadline-user", "user", "DEADLINEGRAPH alpha beta source"),
            ("deadline-assistant", "assistant", "DEADLINEGRAPH alpha beta target"),
        ):
            captured = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json={
                    "event_id": event_id,
                    "session_id": "deadline-session",
                    "turn_id": "deadline-turn",
                    "event_kind": event_kind,
                    "content": content,
                    "occurred_at": "2026-10-25T00:00:00Z",
                    "cwd": str(project_root),
                },
            )
            assert captured.status_code == 202, captured.text

        kernel = client.app.state.memory_kernel
        kernel._wait_for_scope_graph_updates()
        clock = [1000.0]
        monkeypatch.setattr(mem2_module.time, "monotonic", lambda: clock[0])
        original_complete = kernel._complete_evolution

        def complete_then_expire(evolve_key: str, memory_id: str) -> None:
            original_complete(evolve_key, memory_id)
            clock[0] = 2000.0

        monkeypatch.setattr(kernel, "_complete_evolution", complete_then_expire)
        evolved = kernel._evolve(
            EvolveMemoryRequest(
                session_id="deadline-session",
                deadline=1500.0,
            )
        )
        assert evolved["queued"] == 1
        kernel._wait_for_scope_graph_updates()

        status = client.get(
            "/api/v1/libraries/deadline-graph/graph-status",
            headers=headers,
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready", status.text
        assert status.json()["total_documents"] == 3
        assert status.json()["projected_documents"] == 3


def test_rebuild_scope_enumeration_failure_is_degraded_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "enumeration alpha beta one",
            key="graph-enumeration-one",
            source="source:graph-enumeration-one",
        )
        graph_store = client.app.state.memory_kernel._require_bundle().graph_store
        assert graph_store is not None
        original_graph_scopes = graph_store.graph_scopes

        def fail_graph_scopes() -> list[Scope]:
            raise RuntimeError("deterministic graph scope enumeration outage")

        monkeypatch.setattr(graph_store, "graph_scopes", fail_graph_scopes)
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "degraded"
        monkeypatch.setattr(graph_store, "graph_scopes", original_graph_scopes)
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        assert "enumeration outage" in status.json()["last_error"]

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        assert "enumeration outage" in status.json()["last_error"]


def test_graph_projection_error_keys_are_unambiguous(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        kernel = client.app.state.memory_kernel
        first = Scope(
            org="local",
            space="topic:x",
            user="local",
            agent="codex",
            session="y:codex:z",
        )
        second = Scope(
            org="local",
            space="topic:x:codex:y",
            user="local",
            agent="codex",
            session="z",
        )
        assert kernel._graph_projection_key(None, first) != kernel._graph_projection_key(
            None, second
        )
        for region_id in (
            "plain",
            "v2:AAAA",
            "@region/v2:AAAA",
            "topic:中文",
        ):
            region_key = kernel._graph_projection_key(region_id, None)
            assert kernel._decode_graph_projection_key(region_key) == (region_id, None)
        kernel._set_graph_projection_error("first scope only", scope=first)
        assert kernel._graph_projection_error(scope=first) == "first scope only"
        assert kernel._graph_projection_error(scope=second) is None
        kernel._clear_graph_projection_error(scope=second)
        assert kernel._graph_projection_error(scope=first) == "first scope only"
        kernel._clear_graph_projection_error(scope=first)
        assert kernel._graph_projection_error(scope=first) is None


def test_region_rebuild_clears_stale_scope_projection_errors(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        kernel = client.app.state.memory_kernel
        stale_scope = Scope(
            org="local",
            space="topic:graph-region",
            user="local",
            agent="codex",
            session="stale-session",
        )
        kernel._set_graph_projection_error("stale scope failure", scope=stale_scope)
        kernel._set_graph_error("stale transient failure", scope=stale_scope)
        region_before = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert region_before.status_code == 200, region_before.text
        assert region_before.json()["status"] == "degraded"
        before = client.get("/api/v1/status", headers=headers)
        assert before.status_code == 200, before.text
        assert before.json()["graph_index_status"] == "degraded"
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        assert kernel._graph_projection_error(scope=stale_scope) is None
        assert kernel._graph_error_for(scope=stale_scope) is None
        after = client.get("/api/v1/status", headers=headers)
        assert after.status_code == 200, after.text
        assert after.json()["graph_index_status"] == "ready"


def test_v2_prefixed_legacy_region_error_migrates_and_recovers(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    region_id = "v2:AAAA"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        created = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "topic", "name": "Legacy v2", "region_id": region_id},
        )
        assert created.status_code == 201, created.text

    errors_path = state_dir / "mem2" / "graph_projection_errors.json"
    errors_path.write_text(
        json.dumps({f"region:{region_id}": "legacy region failure"}) + "\n",
        encoding="utf-8",
    )
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get(f"/api/v1/libraries/{region_id}/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        global_status = client.get("/api/v1/status", headers=headers)
        assert global_status.status_code == 200, global_status.text
        assert global_status.json()["graph_index_status"] == "degraded"
        rebuilt = client.post(
            f"/api/v1/libraries/{region_id}/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        status = client.get(f"/api/v1/libraries/{region_id}/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        global_status = client.get("/api/v1/status", headers=headers)
        assert global_status.json()["graph_index_status"] == "ready"


def test_legacy_library_region_error_is_visible_and_recoverable(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_id = "plain-library-v21"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        for index, content in enumerate(
            (
                "MIGRATEDLIBRARY alpha beta seed",
                "alpha beta related target",
            ),
            start=1,
        ):
            written = client.post(
                "/api/v1/memories",
                headers=headers,
                json={
                    "library_id": library_id,
                    "content": content,
                    "memory_type": "domain_fact",
                    "source_references": [f"source:migrated-{index}"],
                    "actor": "test",
                    "idempotency_key": f"migrated-library-{index}",
                },
            )
            assert written.status_code == 201, written.text
        rebuilt = client.post(
            f"/api/v1/libraries/{library_id}/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"

    errors_path = state_dir / "mem2" / "graph_projection_errors.json"
    errors_path.write_text(
        json.dumps({f"region:{library_id}": "legacy library graph failure"}) + "\n",
        encoding="utf-8",
    )
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get(f"/api/v1/libraries/{library_id}/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": library_id,
                "query": "migratedlibrary",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]
        assert all(item["classification"] == "direct" for item in search.json()["results"])
        recovered = client.post(
            f"/api/v1/libraries/{library_id}/graph-index/rebuild",
            headers=headers,
        )
        assert recovered.status_code == 202, recovered.text
        assert recovered.json()["status"] == "ready"
        status = client.get(f"/api/v1/libraries/{library_id}/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"


def test_session_region_rebuild_keeps_other_session_errors(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        for region_id, session_id in (
            ("session-one-region", "session-one"),
            ("session-two-region", "session-two"),
        ):
            created = client.post(
                "/api/v1/regions",
                headers=headers,
                json={
                    "kind": "session",
                    "name": region_id,
                    "region_id": region_id,
                    "session_id": session_id,
                },
            )
            assert created.status_code == 201, created.text
        kernel = client.app.state.memory_kernel
        first = Scope(
            org="local",
            space="chat",
            user="local",
            agent="codex",
            session="session-one",
        )
        second = Scope(
            org="local",
            space="chat",
            user="local",
            agent="codex",
            session="session-two",
        )
        kernel._set_graph_projection_error("session one failure", scope=first)
        kernel._set_graph_projection_error("session two failure", scope=second)
        kernel._set_graph_error("session one transient", scope=first)
        kernel._set_graph_error("session two transient", scope=second)
        for region_id in ("session-one-region", "session-two-region"):
            status = client.get(f"/api/v1/libraries/{region_id}/graph-status", headers=headers)
            assert status.status_code == 200, status.text
            assert status.json()["status"] == "degraded"
        rebuilt = client.post(
            "/api/v1/libraries/session-one-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        assert kernel._graph_projection_error(scope=first) is None
        assert kernel._graph_error_for(scope=first) is None
        assert kernel._graph_projection_error(scope=second) == "session two failure"
        assert kernel._graph_error_for(scope=second) == "session two transient"
        second_status = client.get(
            "/api/v1/libraries/session-two-region/graph-status", headers=headers
        )
        assert second_status.status_code == 200, second_status.text
        assert second_status.json()["status"] == "degraded"
        status = client.get("/api/v1/status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["graph_index_status"] == "degraded"


def test_global_graph_status_includes_graph_only_scopes(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "topic", "name": "Ghost", "region_id": "ghost"},
        )
        assert region.status_code == 201, region.text
        graph_store = client.app.state.memory_kernel._require_bundle().graph_store
        assert graph_store is not None
        ghost_scope = Scope(
            org="local",
            space="topic:ghost",
            user="local",
            agent="personal-agent-memory",
            session="",
        )
        graph_store.insert(
            ghost_scope,
            nodes=[Node(id="ghost-node", label="memory", properties={})],
        )
        region_status = client.get("/api/v1/libraries/ghost/graph-status", headers=headers)
        assert region_status.status_code == 200, region_status.text
        assert region_status.json()["status"] == "degraded"
        global_status = client.get("/api/v1/status", headers=headers)
        assert global_status.status_code == 200, global_status.text
        assert global_status.json()["graph_index_status"] == "degraded"


def test_graph_status_detects_dangling_edges_and_rebuild_recovers(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        region = client.post(
            "/api/v1/regions",
            headers=headers,
            json={"kind": "topic", "name": "Edge only", "region_id": "edge-only"},
        )
        assert region.status_code == 201, region.text
        graph_store = client.app.state.memory_kernel._require_bundle().graph_store
        assert graph_store is not None
        scope = Scope(
            org="local",
            space="topic:edge-only",
            user="local",
            agent="personal-agent-memory",
            session="",
        )
        graph_store.insert(
            scope,
            edges=[
                Edge(
                    id="dangling-edge",
                    source="missing-a",
                    target="missing-b",
                    relation="related",
                )
            ],
        )
        region_status = client.get("/api/v1/libraries/edge-only/graph-status", headers=headers)
        assert region_status.status_code == 200, region_status.text
        assert region_status.json()["status"] == "degraded"
        assert "dangling_edges=1" in region_status.json()["last_error"]
        global_status = client.get("/api/v1/status", headers=headers)
        assert global_status.status_code == 200, global_status.text
        assert global_status.json()["graph_index_status"] == "degraded"
        rebuilt = client.post(
            "/api/v1/libraries/edge-only/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        recovered = client.get("/api/v1/libraries/edge-only/graph-status", headers=headers)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["status"] == "ready"
        assert recovered.json()["edges"] == 0


def test_rebuild_failure_keeps_errors_for_unprocessed_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        kernel = client.app.state.memory_kernel
        failing_scope = Scope(
            org="local",
            space="topic:graph-region",
            user="local",
            agent="codex",
            session="aaa-stale",
        )
        unprocessed_scope = Scope(
            org="local",
            space="topic:graph-region",
            user="local",
            agent="codex",
            session="session-b",
        )
        bundle = kernel._require_bundle()
        added = bundle.runtime.api.add(
            "anything in unprocessed scope",
            unprocessed_scope,
            Modality.TEXT,
            security=legacy_request_context(unprocessed_scope),
        )
        assert added
        kernel._set_graph_projection_error(
            "pre-existing missing edges",
            scope=unprocessed_scope,
        )
        monkeypatch.setattr(
            kernel,
            "_region_scopes",
            lambda _bundle, _region: [failing_scope, unprocessed_scope],
        )

        def fail_first_scope(
            _bundle: object,
            scope: Scope,
            *,
            region_id: str | None = None,
        ) -> str | None:
            if scope.session == "aaa-stale":
                kernel._set_graph_projection_error("first scope failure", scope=scope)
                kernel._set_graph_error("first scope failure", scope=scope)
                return None
            return None

        monkeypatch.setattr(kernel, "_associate_graph_scope", fail_first_scope)
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "degraded"
        assert (
            kernel._graph_projection_error(scope=unprocessed_scope) == "pre-existing missing edges"
        )
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "session_id": "session-b",
                "query": "anything",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]


def test_rebuild_serializes_concurrent_graph_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "concurrent alpha beta one",
            key="concurrent-one",
            source="source:concurrent-one",
        )
        _write(
            client,
            headers,
            "concurrent alpha beta two",
            key="concurrent-two",
            source="source:concurrent-two",
        )
        kernel = client.app.state.memory_kernel
        original_associate = kernel._associate_graph_scope
        rebuild_paused = threading.Event()
        release_rebuild = threading.Event()
        evolve_finished = threading.Event()
        rebuild_thread_id: list[int] = []
        errors: list[BaseException] = []

        def controlled_associate(
            bundle: Any,
            scope: Scope,
            *,
            region_id: str | None = None,
        ) -> str | None:
            if threading.get_ident() == rebuild_thread_id[0] and not rebuild_paused.is_set():
                rebuild_paused.set()
                assert release_rebuild.wait(timeout=5)
                return original_associate(bundle, scope, region_id=region_id)
            kernel._set_graph_projection_error("concurrent graph failure", scope=scope)
            return None

        monkeypatch.setattr(kernel, "_associate_graph_scope", controlled_associate)

        def run_rebuild() -> None:
            rebuild_thread_id.append(threading.get_ident())
            try:
                kernel._rebuild_graph_serialized("graph-region")
            except BaseException as error:
                errors.append(error)

        def run_evolve() -> None:
            try:
                kernel._evolve(EvolveMemoryRequest())
            except BaseException as error:
                errors.append(error)
            finally:
                evolve_finished.set()

        rebuild_thread = threading.Thread(target=run_rebuild)
        rebuild_thread.start()
        assert rebuild_paused.wait(timeout=5)
        evolve_thread = threading.Thread(target=run_evolve)
        evolve_thread.start()
        assert not evolve_finished.wait(timeout=0.15)
        release_rebuild.set()
        rebuild_thread.join(timeout=10)
        evolve_thread.join(timeout=10)
        assert not errors
        assert evolve_finished.is_set()
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "alpha",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]


def test_graph_reads_wait_for_rebuild_between_nodes_and_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "readlock alpha beta seed",
            key="readlock-seed",
            source="source:readlock-seed",
        )
        _write(
            client,
            headers,
            "alpha beta readlock target",
            key="readlock-target",
            source="source:readlock-target",
        )
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        graph_store = bundle.graph_store
        assert graph_store is not None
        evolver = bundle.runtime.api._engine._evolver
        original_persist_graph = evolver._persist_graph
        nodes_written = threading.Event()
        release_edges = threading.Event()
        status_finished = threading.Event()
        search_finished = threading.Event()
        errors: list[BaseException] = []
        status_payloads: list[dict[str, object]] = []
        search_payloads: list[dict[str, object]] = []

        def controlled_persist_graph(units: list[Any], relations: list[Any]) -> None:
            if units and not nodes_written.is_set():
                scope = units[0].scope
                for unit in units:
                    graph_store.insert(
                        scope,
                        nodes=[
                            Node(
                                id=unit.id,
                                label=unit.tier.value,
                                properties={"content": unit.content},
                            )
                        ],
                    )
                nodes_written.set()
                assert release_edges.wait(timeout=5)
            original_persist_graph(units, relations)

        monkeypatch.setattr(evolver, "_persist_graph", controlled_persist_graph)

        def run_rebuild() -> None:
            try:
                kernel._rebuild_graph_serialized("graph-region")
            except BaseException as error:
                errors.append(error)

        def run_status() -> None:
            try:
                status_payloads.append(kernel.graph_status("graph-region"))
            except BaseException as error:
                errors.append(error)
            finally:
                status_finished.set()

        def run_search() -> None:
            try:
                search_payloads.append(
                    kernel._search(
                        SearchMemoryRequest(
                            region_id="graph-region",
                            query="readlock",
                            limit=10,
                        )
                    )
                )
            except BaseException as error:
                errors.append(error)
            finally:
                search_finished.set()

        rebuild_thread = threading.Thread(target=run_rebuild)
        rebuild_thread.start()
        assert nodes_written.wait(timeout=5)
        status_thread = threading.Thread(target=run_status)
        search_thread = threading.Thread(target=run_search)
        status_thread.start()
        search_thread.start()
        assert not status_finished.wait(timeout=0.15)
        assert not search_finished.wait(timeout=0.15)
        release_edges.set()
        for thread in (rebuild_thread, status_thread, search_thread):
            thread.join(timeout=10)
        assert not errors
        assert status_finished.is_set()
        assert search_finished.is_set()
        assert status_payloads[0]["status"] == "ready", status_payloads[0]
        assert int(status_payloads[0]["edges"]) == 1
        assert search_payloads[0]["graph_index_status"] == "ready"


def test_evolve_and_governance_graph_writes_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _write(
            client,
            headers,
            "uniquelead alpha beta seed",
            key="serialized-seed",
            source="source:serialized-seed",
        )
        target = _write(
            client,
            headers,
            "alpha beta original target",
            key="serialized-target",
            source="source:serialized-target",
        )
        target_id = str(target["id"])
        rebuilt = client.post(
            "/api/v1/libraries/graph-region/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready"
        target_document = client.get(
            "/api/v1/libraries/graph-region/document",
            headers=headers,
            params={"path": target_id},
        )
        assert target_document.status_code == 200, target_document.text
        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None
        original_insert = graph_store.insert
        original_clear = graph_store.clear
        evolve_paused = threading.Event()
        release_evolve = threading.Event()
        update_started = threading.Event()
        update_paused = threading.Event()
        release_update = threading.Event()
        evolve_finished = threading.Event()
        update_finished = threading.Event()
        errors: list[BaseException] = []

        def controlled_insert(
            scope: Scope,
            nodes: list[Node] | None = None,
            edges: list[Edge] | None = None,
        ) -> None:
            if not update_started.is_set() and not evolve_paused.is_set():
                evolve_paused.set()
                assert release_evolve.wait(timeout=5)
            original_insert(scope, nodes=nodes, edges=edges)

        def controlled_clear(scope: Scope) -> None:
            if update_started.is_set() and not update_paused.is_set():
                update_paused.set()
                assert release_update.wait(timeout=5)
            original_clear(scope)

        monkeypatch.setattr(graph_store, "insert", controlled_insert)
        monkeypatch.setattr(graph_store, "clear", controlled_clear)

        def run_evolve() -> None:
            try:
                kernel._evolve(EvolveMemoryRequest())
            except BaseException as error:
                errors.append(error)
            finally:
                evolve_finished.set()

        def run_update() -> None:
            update_started.set()
            try:
                asyncio.run(
                    kernel.update(
                        DocumentUpdateRequest(
                            library_id="graph-region",
                            path=target_id,
                            content="zzz unrelated replacement",
                            expected_revision=str(target_document.json()["version"]),
                            operation_id="serialized-update",
                            actor="test",
                            source="test-serialized-update",
                            mode="overwrite",
                        )
                    )
                )
            except BaseException as error:
                errors.append(error)
            finally:
                update_finished.set()

        evolve_thread = threading.Thread(target=run_evolve)
        evolve_thread.start()
        assert evolve_paused.wait(timeout=5)
        update_thread = threading.Thread(target=run_update)
        update_thread.start()
        assert not update_paused.wait(timeout=0.15)
        assert not evolve_finished.is_set()
        assert not update_finished.is_set()
        release_evolve.set()
        assert update_paused.wait(timeout=5)
        release_update.set()
        evolve_thread.join(timeout=10)
        update_thread.join(timeout=10)
        assert not errors
        assert evolve_finished.is_set()
        assert update_finished.is_set()
        bundle = kernel._require_bundle()
        target_scope = next(
            scope
            for scope in bundle.kv_store.scopes()
            if any(unit.id == target_id for unit in kernel._units_in_scope(bundle, scope))
        )
        assert graph_store.counts(target_scope)["edges"] == 0
        status = client.get("/api/v1/libraries/graph-region/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "ready"
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph-region",
                "query": "uniquelead",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert not any(
            item["document_id"] == target_id and item["classification"] == "graph_expansion"
            for item in search.json()["results"]
        )


def test_rebuild_blocks_concurrent_retention_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        created = client.post(
            "/api/v1/regions",
            headers=headers,
            json={
                "kind": "session",
                "name": "Retention session",
                "region_id": "retention-session",
                "session_id": "retention-session-id",
            },
        )
        assert created.status_code == 201, created.text
        kernel = client.app.state.memory_kernel
        original_associate = kernel._associate_graph_scope
        rebuild_paused = threading.Event()
        release_rebuild = threading.Event()
        retention_finished = threading.Event()
        rebuild_thread_id: list[int] = []
        errors: list[BaseException] = []

        def controlled_associate(
            bundle: Any,
            scope: Scope,
            *,
            region_id: str | None = None,
        ) -> str | None:
            if threading.get_ident() == rebuild_thread_id[0] and not rebuild_paused.is_set():
                rebuild_paused.set()
                assert release_rebuild.wait(timeout=5)
            return original_associate(bundle, scope, region_id=region_id)

        monkeypatch.setattr(kernel, "_associate_graph_scope", controlled_associate)

        def run_rebuild() -> None:
            rebuild_thread_id.append(threading.get_ident())
            try:
                kernel._rebuild_graph_serialized("retention-session")
            except BaseException as error:
                errors.append(error)

        def run_retention() -> None:
            try:
                kernel._run_retention_cleanup(
                    RetentionCleanupRequest(library_id="retention-session")
                )
            except BaseException as error:
                errors.append(error)
            finally:
                retention_finished.set()

        rebuild_thread = threading.Thread(target=run_rebuild)
        rebuild_thread.start()
        assert rebuild_paused.wait(timeout=5)
        retention_thread = threading.Thread(target=run_retention)
        retention_thread.start()
        assert not retention_finished.wait(timeout=0.15)
        release_rebuild.set()
        rebuild_thread.join(timeout=10)
        retention_thread.join(timeout=10)
        assert not errors
        assert retention_finished.is_set()


def test_library_scope_edge_failure_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    library_id = "legacy-library"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        for index, content in enumerate(
            (
                "UNIQUELEGACY alpha beta related seed",
                "alpha beta related target",
            ),
            start=1,
        ):
            written = client.post(
                "/api/v1/memories",
                headers=headers,
                json={
                    "library_id": library_id,
                    "content": content,
                    "memory_type": "domain_fact",
                    "source_references": [f"source:legacy-{index}"],
                    "actor": "test",
                    "idempotency_key": f"legacy-edge-failure-{index}",
                },
            )
            assert written.status_code == 201, written.text
        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None
        original_insert = graph_store.insert

        def insert_nodes_then_fail_edges(
            scope: Scope,
            nodes: list[Node] | None = None,
            edges: list[Edge] | None = None,
        ) -> None:
            if edges:
                if nodes:
                    original_insert(scope, nodes=nodes)
                raise RuntimeError("deterministic library edge outage")
            original_insert(scope, nodes=nodes)

        monkeypatch.setattr(graph_store, "insert", insert_nodes_then_fail_edges)
        asyncio.run(kernel.evolve(EvolveMemoryRequest()))
        assert (
            client.get("/api/v1/status", headers=headers).json()["graph_index_status"] == "degraded"
        )

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get("/api/v1/status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["graph_index_status"] == "degraded"
        library_status = client.get(f"/api/v1/libraries/{library_id}/graph-status", headers=headers)
        assert library_status.status_code == 200, library_status.text
        assert library_status.json()["status"] == "degraded"
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": library_id,
                "query": "uniquelegacy",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        payload = search.json()
        assert payload["graph_index_status"] == "degraded"
        assert "graph_unavailable" in payload["degradation"]
        assert payload["results"]
        assert all(item["classification"] == "direct" for item in payload["results"])
        rebuilt = client.post(
            f"/api/v1/libraries/{library_id}/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        assert rebuilt.json()["status"] == "ready", rebuilt.text
        recovered = client.get(f"/api/v1/libraries/{library_id}/graph-status", headers=headers)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["status"] == "ready"
        expanded = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": library_id,
                "query": "uniquelegacy",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert expanded.status_code == 200, expanded.text
        assert expanded.json()["graph_index_status"] == "ready"
        assert any(
            item["classification"] == "graph_expansion" for item in expanded.json()["results"]
        )
        recovered_store = client.app.state.memory_kernel._require_bundle().graph_store
        assert recovered_store is not None

        def fail_clear(_scope: Scope) -> None:
            raise RuntimeError("R19 restart clear outage")

        monkeypatch.setattr(recovered_store, "clear", fail_clear)
        failed_again = client.post(
            f"/api/v1/libraries/{library_id}/graph-index/rebuild",
            headers=headers,
        )
        assert failed_again.status_code == 202, failed_again.text
        assert failed_again.json()["status"] == "degraded"

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        status = client.get(f"/api/v1/libraries/{library_id}/graph-status", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": library_id,
                "query": "uniquelegacy",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        assert search.json()["graph_index_status"] == "degraded"
        assert "graph_unavailable" in search.json()["degradation"]
        assert all(item["classification"] == "direct" for item in search.json()["results"])
        recovered = client.post(
            f"/api/v1/libraries/{library_id}/graph-index/rebuild",
            headers=headers,
        )
        assert recovered.status_code == 202, recovered.text
        assert recovered.json()["status"] == "ready"
