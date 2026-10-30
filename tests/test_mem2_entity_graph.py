from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.entity_graph import (
    GRAPH_MAX_EDGE_LIMIT,
    GRAPH_MAX_NODE_LIMIT,
    GRAPH_SUPPORTING_EVIDENCE_LIMIT,
    GRAPH_SUPPORTING_MEMORY_LIMIT,
    EntityGraphStore,
    stable_entity_id,
    stable_relation_id,
)
from personal_agent_memory.mem2_kernel import Mem2MemoryKernel
from personal_agent_memory.memory_kernel import (
    MemoryKernelUnavailableError,
    UnavailableMemoryKernel,
)


def _extractor(context: dict[str, object]) -> dict[str, object]:
    content = str(context["memory"]["content"])
    if "Core" in content:
        return {
            "entities": [
                {"type": "project", "canonical_name": "Core", "aliases": ["core repo"]},
                {"type": "concept", "canonical_name": "Core"},
                {"type": "invalid", "canonical_name": "Rejected"},
            ],
            "relations": [],
        }
    return {
        "entities": [
            {"type": "project", "canonical_name": "Project Atlas", "aliases": ["atlas"]},
            {"type": "technology", "canonical_name": "Python"},
            {"type": "component", "canonical_name": "Missing Component"},
        ],
        "relations": [
            {
                "source": "project:Project Atlas",
                "target": "technology:Python",
                "type": "uses",
                "evidence": [content],
            },
            {
                "source": "project:Project Atlas",
                "target": "component:Missing Component",
                "type": "uses",
                "evidence": [],
            },
            {
                "source": "project:Project Atlas",
                "target": "technology:Unknown",
                "type": "depends_on",
                "evidence": [content],
            },
            {
                "source": "project:Project Atlas",
                "target": "technology:Python",
                "type": "free_form",
                "evidence": [content],
            },
        ],
    }


def _app(state_dir: Path, *, extractor: Any = _extractor):
    kernel = Mem2MemoryKernel(state_dir, graph_extractor=extractor)
    return create_app(Settings(state_dir=state_dir), memory_kernel=kernel)


def _headers(state_dir: Path) -> dict[str, str]:
    return {"Authorization": f"Bearer {(state_dir / 'api-key').read_text().strip()}"}


def _region(client: TestClient, headers: dict[str, str], region_id: str = "graph") -> None:
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json={"kind": "topic", "name": region_id, "region_id": region_id},
    )
    assert response.status_code == 201, response.text


def _memory(
    client: TestClient,
    headers: dict[str, str],
    content: str,
    key: str,
    *,
    memory_type: str = "domain_fact",
) -> str:
    response = client.post(
        "/api/v1/memories",
        headers=headers,
        json={
            "region_id": "graph",
            "content": content,
            "memory_type": memory_type,
            "source_references": [f"source:{key}"],
            "idempotency_key": key,
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def test_entity_graph_fixed_vocabulary_evidence_and_identity(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path / "state")) as client:
        state_dir = tmp_path / "state"
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(client, headers, "Project Atlas uses Python.", "one")

        graph = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert graph.status_code == 200, graph.text
        payload = graph.json()
        assert {node["type"] for node in payload["nodes"]} == {"project", "technology"}
        assert {edge["type"] for edge in payload["edges"]} == {"uses"}
        assert payload["edges"][0]["memory_ids"] == [memory_id]
        assert payload["edges"][0]["evidence"][0]["source_references"] == ["source:one"]
        assert all("Missing Component" not in node["canonical_name"] for node in payload["nodes"])

        _memory(client, headers, "Core is a project concept.", "two")
        graph = client.get("/api/v1/libraries/graph/entity-graph", headers=headers).json()
        core_nodes = [node for node in graph["nodes"] if node["canonical_name"] == "Core"]
        assert {node["type"] for node in core_nodes} == {"project", "concept"}


def test_entity_graph_rejects_sensitive_input_and_keeps_unmaterialized_memory(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def extractor(context: dict[str, object]) -> dict[str, object]:
        calls.append(context)
        return {
            "entities": [{"type": "technology", "canonical_name": "API key"}],
            "relations": [],
        }

    with TestClient(_app(tmp_path / "state", extractor=extractor)) as client:
        state_dir = tmp_path / "state"
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(
            client,
            headers,
            "api_key = sk-123456789012345678901234",
            "sensitive",
        )
        graph = client.get("/api/v1/libraries/graph/entity-graph", headers=headers).json()
        assert graph["nodes"] == []
        assert graph["edges"] == []
        listed = client.get("/api/v1/libraries/graph/memories", headers=headers).json()
        assert {item["id"] for item in listed["items"]} == {memory_id}
        assert calls == []


def test_entity_graph_is_idempotent_rebuild_and_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(client, headers, "Project Atlas uses Python.", "one")
        first = client.get("/api/v1/libraries/graph/entity-graph", headers=headers).json()
        rebuild = client.post("/api/v1/libraries/graph/entity-graph/rebuild", headers=headers)
        assert rebuild.status_code == 202, rebuild.text
        second = client.get("/api/v1/libraries/graph/entity-graph", headers=headers).json()
        assert first["nodes"] == second["nodes"]
        assert first["edges"] == second["edges"]
        assert first["edges"][0]["memory_ids"] == [memory_id]

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        restored = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert restored.status_code == 200, restored.text
        assert len(restored.json()["nodes"]) == 2
        assert len(restored.json()["edges"]) == 1


def test_memory_detail_exposes_read_only_entity_graph_and_supporting_sources(
    tmp_path: Path,
) -> None:
    with TestClient(_app(tmp_path / "state")) as client:
        state_dir = tmp_path / "state"
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(client, headers, "Project Atlas uses Python.", "one")
        detail = client.get(
            f"/api/v1/libraries/graph/memories/{memory_id}", headers=headers
        )
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        assert payload["associated_entities"]
        assert payload["associated_relations"][0]["type"] == "uses"
        assert payload["local_graph"]["read_only"] is True
        entity_id = payload["associated_entities"][0]["id"]
        entity = client.get(
            f"/api/v1/libraries/graph/entity-graph/entities/{entity_id}", headers=headers
        )
        assert entity.status_code == 200, entity.text
        assert memory_id in entity.json()["supporting_memory_ids"]
        assert entity.json()["supporting_memories"][0]["id"] == memory_id
        relation_id = payload["associated_relations"][0]["id"]
        relation = client.get(
            f"/api/v1/libraries/graph/entity-graph/relations/{relation_id}", headers=headers
        )
        assert relation.status_code == 200, relation.text
        assert relation.json()["supporting_memory_ids"] == [memory_id]
        assert relation.json()["supporting_memories"][0]["id"] == memory_id


def test_entity_graph_uses_formal_memories_only_and_rejects_sensitive_relations(
    tmp_path: Path,
) -> None:
    seen_contents: list[str] = []

    def extractor(context: dict[str, object]) -> dict[str, object]:
        content = str(context["memory"]["content"])
        seen_contents.append(content)
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": [content],
                },
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "references",
                    "evidence": ["api_key = sk-123456789012345678901234"],
                },
            ],
        }

    with TestClient(_app(tmp_path / "state", extractor=extractor)) as client:
        state_dir = tmp_path / "state"
        headers = _headers(state_dir)
        _region(client, headers)
        _memory(client, headers, "Project Atlas uses Python.", "formal")
        _memory(
            client,
            headers,
            "Project Atlas uses Python.",
            "evolved",
            memory_type="evolved",
        )
        graph = client.get("/api/v1/libraries/graph/entity-graph", headers=headers).json()
        assert len(graph["nodes"]) == 2
        assert {edge["type"] for edge in graph["edges"]} == {"uses"}
        assert seen_contents == ["Project Atlas uses Python."]


@pytest.mark.parametrize(
    "source_references",
    [
        pytest.param(None, id="missing"),
        pytest.param([], id="empty"),
        pytest.param([{}], id="mapping-item"),
        pytest.param([123], id="non-string-item"),
        pytest.param(["   "], id="blank-item"),
        pytest.param(["api_key = source-secret"], id="sensitive-item"),
    ],
)
def test_entity_graph_rejects_invalid_source_references(
    tmp_path: Path, source_references: object
) -> None:
    store = EntityGraphStore(tmp_path / "missing-sources.sqlite3")
    calls: list[dict[str, object]] = []
    memory = {
        "memory_id": "memory-without-sources",
        "content": "Project Atlas uses Python.",
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }
    if source_references is not None:
        memory["source_references"] = source_references

    def extractor(context: dict[str, object]) -> dict[str, object]:
        calls.append(context)
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": ["Project Atlas uses Python."],
                }
            ],
        }

    result = store.project("graph", memory, extractor)
    assert result["status"] == "skipped"
    assert result["rejected"] == 1
    assert calls == []
    graph = store.graph("graph")
    assert graph["nodes"] == []
    assert graph["edges"] == []


def test_mem2_kernel_preserves_invalid_source_reference_types_for_graph_validation(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def extractor(context: dict[str, object]) -> dict[str, object]:
        calls.append(context)
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": ["Project Atlas uses Python."],
                }
            ],
        }

    state_dir = tmp_path / "state"
    kernel = Mem2MemoryKernel(state_dir, graph_extractor=extractor)
    with TestClient(create_app(Settings(state_dir=state_dir), memory_kernel=kernel)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(client, headers, "Project Atlas uses Python.", "kernel-source")
        kernel._wait_for_entity_graph_updates()
        bundle = kernel._require_bundle()
        region = kernel._regions.require_region("graph")
        scope, unit = next(
            pair for pair in kernel._region_unit_pairs(bundle, region) if pair[1].id == memory_id
        )
        invalid_unit = replace(
            unit,
            id=f"{memory_id}-invalid-source",
            user_metadata={
                **dict(unit.user_metadata),
                "source_references": [{}],
            },
        )
        calls.clear()
        payload = kernel._entity_graph_memory_payload(bundle, region, scope, invalid_unit)
        assert payload["source_references"] == [{}]
        result = kernel._project_entity_graph_memory(bundle, region, scope, invalid_unit)
        assert result["status"] == "skipped"
        assert calls == []
        assert calls == []


@pytest.mark.parametrize("overflow_kind", ["entity", "relation"])
def test_entity_graph_rejects_overlimit_extractor_evidence_before_persistence(
    tmp_path: Path, overflow_kind: str
) -> None:
    store = EntityGraphStore(tmp_path / f"overlimit-{overflow_kind}.sqlite3")
    content = "Project Atlas uses Python."
    memory = {
        "memory_id": f"memory-overlimit-{overflow_kind}",
        "content": content,
        "source_references": [f"source:{overflow_kind}"],
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }
    calls: list[dict[str, object]] = []

    def extractor(context: dict[str, object]) -> dict[str, object]:
        calls.append(context)
        entities: list[dict[str, object]] = [
            {"type": "project", "canonical_name": "Project Atlas"},
            {"type": "technology", "canonical_name": "Python"},
        ]
        relations: list[dict[str, object]] = [
            {
                "source": "project:Project Atlas",
                "target": "technology:Python",
                "type": "uses",
                "evidence": [content],
            }
        ]
        if overflow_kind == "entity":
            entities[0]["evidence"] = [content] * (
                GRAPH_SUPPORTING_EVIDENCE_LIMIT + 1
            )
        else:
            relations[0]["evidence"] = [content] * (
                GRAPH_SUPPORTING_EVIDENCE_LIMIT + 1
            )
        return {"entities": entities, "relations": relations}

    with pytest.raises(ValueError, match="evidence exceeds 50 items"):
        store.project("graph", memory, extractor)
    assert len(calls) == 1
    assert store.graph("graph")["nodes"] == []
    with sqlite3.connect(tmp_path / f"overlimit-{overflow_kind}.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM entity_graph_entities WHERE region_id = ?",
            ("graph",),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM entity_graph_relations WHERE region_id = ?",
            ("graph",),
        ).fetchone() == (0,)


def test_entity_graph_accepts_evidence_at_hard_limit_and_rebuild_applies_same_guard(
    tmp_path: Path,
) -> None:
    store = EntityGraphStore(tmp_path / "evidence-boundary.sqlite3")
    content = "Project Atlas uses Python."
    memory = {
        "memory_id": "memory-boundary",
        "content": content,
        "source_references": ["source:boundary"],
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }

    def bounded_extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {
                    "type": "project",
                    "canonical_name": "Project Atlas",
                    "evidence": [content] * GRAPH_SUPPORTING_EVIDENCE_LIMIT,
                },
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": [content] * GRAPH_SUPPORTING_EVIDENCE_LIMIT,
                }
            ],
        }

    assert store.project("graph", memory, bounded_extractor)["status"] == "ready"
    before = store.graph("graph")
    assert len(before["nodes"]) == 2
    assert len(before["edges"]) == 1

    def overlimit_extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {
                    "type": "project",
                    "canonical_name": "Project Atlas",
                    "evidence": [content] * (GRAPH_SUPPORTING_EVIDENCE_LIMIT + 1),
                },
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [],
        }

    with pytest.raises(ValueError, match="evidence exceeds 50 items"):
        store.project("graph", memory, overlimit_extractor)
    assert store.graph("graph") == before

    rebuilt = store.rebuild("graph", [memory], overlimit_extractor)
    assert rebuilt["status"] == "degraded"
    assert store.graph("graph") == before


def test_entity_graph_default_projection_excludes_superseded_and_forgotten_memories(
    tmp_path: Path,
) -> None:
    def extractor(context: dict[str, object]) -> dict[str, object]:
        content = str(context["memory"]["content"])
        if content.startswith("OldThing"):
            source, target = "OldThing", "OldTool"
        else:
            source, target = "NewThing", "NewTool"
        return {
            "entities": [
                {"type": "concept", "canonical_name": source},
                {"type": "technology", "canonical_name": target},
            ],
            "relations": [
                {
                    "source": f"concept:{source}",
                    "target": f"technology:{target}",
                    "type": "uses",
                    "evidence": [content],
                }
            ],
        }

    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir, extractor=extractor)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        old_id = _memory(client, headers, "OldThing uses OldTool.", "old")

        old_detail = client.get(
            f"/api/v1/libraries/graph/memories/{old_id}", headers=headers
        )
        assert old_detail.status_code == 200, old_detail.text
        updated = client.put(
            "/api/v1/libraries/graph/document",
            headers=headers,
            json={
                "path": old_id,
                "content": "NewThing uses NewTool.",
                "expected_source_version": old_detail.json()["source_version"],
                "operation_id": "entity-graph-supersede",
                "actor_type": "user",
                "source": "entity-graph-test",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        new_id = str(updated.json()["id"])
        assert new_id != old_id

        graph = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert graph.status_code == 200, graph.text
        payload = graph.json()
        assert {node["canonical_name"] for node in payload["nodes"]} == {
            "NewThing",
            "NewTool",
        }
        projected_memory_ids = {
            memory_id
            for node in payload["nodes"]
            for memory_id in node["evidence_memory_ids"]
        }
        assert projected_memory_ids == {new_id}
        assert [edge["memory_ids"] for edge in payload["edges"]] == [[new_id]]
        assert old_id not in json.dumps(payload, ensure_ascii=False)

        new_detail = client.get(
            f"/api/v1/libraries/graph/memories/{new_id}", headers=headers
        )
        assert new_detail.status_code == 200, new_detail.text
        preview = client.post(
            "/api/v1/libraries/graph/document/delete-preview",
            headers=headers,
            json={
                "path": new_id,
                "expected_source_version": new_detail.json()["source_version"],
                "operation_id": "entity-graph-forget-preview",
            },
        )
        assert preview.status_code == 200, preview.text
        forgotten = client.request(
            "DELETE",
            "/api/v1/libraries/graph/document",
            headers=headers,
            json={
                "path": new_id,
                "expected_source_version": new_detail.json()["source_version"],
                "preview_token": preview.json()["preview_token"],
                "operation_id": "entity-graph-forget",
                "actor_type": "user",
                "source": "entity-graph-test",
                "mode": "forget",
            },
        )
        assert forgotten.status_code == 200, forgotten.text

        graph_after_forget = client.get(
            "/api/v1/libraries/graph/entity-graph", headers=headers
        )
        assert graph_after_forget.status_code == 200, graph_after_forget.text
        assert graph_after_forget.json()["nodes"] == []
        assert graph_after_forget.json()["edges"] == []


@pytest.mark.parametrize("lifecycle", ["supersede", "archive", "forget"])
def test_entity_graph_rebuild_failure_prunes_lifecycle_stale_projection(
    tmp_path: Path, lifecycle: str
) -> None:
    should_fail = False

    def extractor(context: dict[str, object]) -> dict[str, object]:
        if should_fail:
            raise ValueError("extractor unavailable")
        content = str(context["memory"]["content"])
        project = content.split(" uses ", 1)[0]
        return {
            "entities": [
                {"type": "project", "canonical_name": project},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": f"project:{project}",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": [content],
                }
            ],
        }

    state_dir = tmp_path / lifecycle
    with TestClient(_app(state_dir, extractor=extractor)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        stale_id = _memory(client, headers, "StaleThing uses Python.", "stale")
        anchor_id = _memory(client, headers, "AnchorThing uses Python.", "anchor")
        before = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert before.status_code == 200, before.text
        assert {node["canonical_name"] for node in before.json()["nodes"]} == {
            "AnchorThing",
            "Python",
            "StaleThing",
        }

        should_fail = True
        if lifecycle == "supersede":
            stale_detail = client.get(
                f"/api/v1/libraries/graph/memories/{stale_id}", headers=headers
            )
            assert stale_detail.status_code == 200, stale_detail.text
            updated = client.put(
                "/api/v1/libraries/graph/document",
                headers=headers,
                json={
                    "path": stale_id,
                    "content": "ReplacementThing uses Python.",
                    "expected_source_version": stale_detail.json()["source_version"],
                    "operation_id": "entity-graph-failure-supersede",
                    "actor_type": "user",
                    "source": "entity-graph-test",
                    "mode": "supersede",
                },
            )
            assert updated.status_code == 200, updated.text
        else:
            stale_detail = client.get(
                f"/api/v1/libraries/graph/memories/{stale_id}", headers=headers
            )
            assert stale_detail.status_code == 200, stale_detail.text
            preview = client.post(
                "/api/v1/libraries/graph/document/delete-preview",
                headers=headers,
                json={
                    "path": stale_id,
                    "expected_source_version": stale_detail.json()["source_version"],
                    "operation_id": f"entity-graph-failure-{lifecycle}-preview",
                },
            )
            assert preview.status_code == 200, preview.text
            deleted = client.request(
                "DELETE",
                "/api/v1/libraries/graph/document",
                headers=headers,
                json={
                    "path": stale_id,
                    "expected_source_version": stale_detail.json()["source_version"],
                    "preview_token": preview.json()["preview_token"],
                    "operation_id": f"entity-graph-failure-{lifecycle}",
                    "actor_type": "user",
                    "source": "entity-graph-test",
                    "mode": lifecycle,
                },
            )
            assert deleted.status_code == 200, deleted.text

        after = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert after.status_code == 200, after.text
        payload = after.json()
        assert {node["canonical_name"] for node in payload["nodes"]} == {
            "AnchorThing",
            "Python",
        }
        assert payload["edges"][0]["memory_ids"] == [anchor_id]
        assert stale_id not in json.dumps(payload, ensure_ascii=False)
        status = client.get(
            "/api/v1/libraries/graph/entity-graph/status", headers=headers
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "degraded"


def test_entity_graph_unconfigured_status_is_explicit_before_and_after_publish(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    kernel = Mem2MemoryKernel(state_dir)
    with TestClient(create_app(Settings(state_dir=state_dir), memory_kernel=kernel)) as client:
        headers = _headers(state_dir)
        _region(client, headers)

        before = client.get("/api/v1/libraries/graph/entity-graph/status", headers=headers)
        assert before.status_code == 200, before.text
        assert before.json()["status"] == "not_configured"
        assert before.json()["last_error"] == "entity graph extractor is not configured"

        memory_id = _memory(client, headers, "Project Atlas is documented.", "unconfigured")
        after = client.get("/api/v1/libraries/graph/entity-graph/status", headers=headers)
        assert after.status_code == 200, after.text
        assert after.json()["status"] == "not_configured"
        assert after.json()["projected_memories"] == 0
        assert memory_id

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        row = connection.execute(
            "SELECT status, last_error FROM entity_graph_status WHERE region_id = ?",
            ("graph",),
        ).fetchone()
    assert row == ("not_configured", "entity graph extractor is not configured")


def test_entity_graph_queue_rebuilds_after_blocked_refresh_and_is_idempotent(
    tmp_path: Path,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[str] = []

    def extractor(context: dict[str, object]) -> dict[str, object]:
        raw_memory = context["memory"]
        assert isinstance(raw_memory, dict)
        content = str(raw_memory["content"])
        calls.append(content)
        if content == "Project First uses Python.":
            first_started.set()
            assert release_first.wait(timeout=5)
        project = content.removesuffix(" uses Python.")
        return {
            "entities": [
                {"type": "project", "canonical_name": project},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": f"project:{project}",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": [content],
                }
            ],
        }

    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir, extractor=extractor)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        first_id = _memory(client, headers, "Project First uses Python.", "first")
        assert first_started.wait(timeout=5)
        second_id = _memory(client, headers, "Project Second uses Python.", "second")
        release_first.set()

        graph = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert graph.status_code == 200, graph.text
        payload = graph.json()
        memory_ids = {
            memory_id
            for node in payload["nodes"]
            for evidence in node["evidence"]
            if isinstance(evidence, dict)
            for memory_id in [evidence.get("memory_id")]
        }
        assert {first_id, second_id} <= memory_ids
        status = client.get("/api/v1/libraries/graph/entity-graph/status", headers=headers)
        assert status.json()["status"] == "ready"
        assert status.json()["projected_memories"] == 2
        assert calls.count("Project First uses Python.") >= 1
        assert calls.count("Project Second uses Python.") >= 1

        rebuilt = client.post(
            "/api/v1/libraries/graph/entity-graph/rebuild", headers=headers
        )
        assert rebuilt.status_code == 202, rebuilt.text
        repeated = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert repeated.json()["nodes"] == payload["nodes"]
        assert repeated.json()["edges"] == payload["edges"]


def test_entity_graph_relation_detail_queries_tail_beyond_default_window(
    tmp_path: Path,
) -> None:
    def extractor(context: dict[str, object]) -> dict[str, object]:
        content = str(context["memory"]["content"])
        names = [f"Entity-{index:03d}" for index in range(202)]
        return {
            "entities": [
                {"type": "concept", "canonical_name": name} for name in names
            ],
            "relations": [
                {
                    "source": "concept:Entity-200",
                    "target": "concept:Entity-201",
                    "type": "related_to",
                    "evidence": [content],
                }
            ],
        }

    content = " ".join(f"Entity-{index:03d}" for index in range(202))
    source_id = stable_entity_id("graph", "concept", "entity-200")
    target_id = stable_entity_id("graph", "concept", "entity-201")
    relation_id = stable_relation_id("graph", source_id, target_id, "related_to")
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir, extractor=extractor)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(client, headers, content, "tail")

        graph = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert graph.status_code == 200, graph.text
        assert len(graph.json()["nodes"]) == 200
        assert len(graph.json()["edges"]) == 1
        assert {node["canonical_name"] for node in graph.json()["nodes"]} >= {
            "Entity-200",
            "Entity-201",
        }

        detail = client.get(
            f"/api/v1/libraries/graph/entity-graph/relations/{relation_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        assert payload["relation"]["id"] == relation_id
        assert payload["source_entity"]["canonical_name"] == "Entity-200"
        assert payload["target_entity"]["canonical_name"] == "Entity-201"
        assert payload["supporting_memory_ids"] == [memory_id]
        assert payload["supporting_memories"][0]["id"] == memory_id
        assert [node["canonical_name"] for node in payload["graph"]["nodes"]] == [
            "Entity-200",
            "Entity-201",
        ]
        entity_detail = client.get(
            f"/api/v1/libraries/graph/entity-graph/entities/{target_id}",
            headers=headers,
        )
        assert entity_detail.status_code == 200, entity_detail.text
        assert entity_detail.json()["entity"]["canonical_name"] == "Entity-201"


def test_entity_graph_api_rebuild_failure_keeps_previous_projection(
    tmp_path: Path,
) -> None:
    should_fail = False

    def extractor(context: dict[str, object]) -> dict[str, object]:
        if should_fail:
            raise ValueError("extractor unavailable")
        content = str(context["memory"]["content"])
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": [content],
                }
            ],
        }

    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir, extractor=extractor)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _memory(client, headers, "Project Atlas uses Python.", "rebuild-old")
        before = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert before.status_code == 200
        should_fail = True
        rebuilt = client.post(
            "/api/v1/libraries/graph/entity-graph/rebuild", headers=headers
        )
        assert rebuilt.status_code == 202
        assert rebuilt.json()["status"] == "degraded"
        after = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert after.json()["nodes"] == before.json()["nodes"]
        assert after.json()["edges"] == before.json()["edges"]


def test_entity_graph_store_and_api_reject_forged_entity_evidence_without_persistence(
    tmp_path: Path,
) -> None:
    memory = {
        "memory_id": "memory-direct",
        "content": "Project Atlas uses Python.",
        "evidence": [],
        "evidence_snippets": [],
        "source_references": ["source:direct"],
    }
    store = EntityGraphStore(tmp_path / "direct.sqlite3")

    def forged_extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {
                    "type": "project",
                    "canonical_name": "Project Atlas",
                    "evidence": ["forged entity evidence"],
                },
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": ["forged relation evidence"],
                }
            ],
        }

    result = store.project("graph", memory, forged_extractor)
    assert result["rejected"] == 2
    with sqlite3.connect(tmp_path / "direct.sqlite3") as connection:
        rows = connection.execute(
            "SELECT evidence FROM entity_graph_entities UNION ALL "
            "SELECT evidence FROM entity_graph_relations"
        ).fetchall()
    assert len(rows) == 1
    assert "forged" not in (tmp_path / "direct.sqlite3").read_bytes().decode(
        "utf-8", errors="ignore"
    )

    def api_extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {
                    "type": "project",
                    "canonical_name": "Project Atlas",
                    "evidence": ["forged entity evidence"],
                },
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": ["Project Atlas uses Python."],
                }
            ],
        }

    state_dir = tmp_path / "api-state"
    with TestClient(_app(state_dir, extractor=api_extractor)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _memory(client, headers, "Project Atlas uses Python.", "forged-api")
        graph = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert graph.status_code == 200, graph.text
        assert [node["canonical_name"] for node in graph.json()["nodes"]] == ["Python"]
        assert graph.json()["edges"] == []
        assert "forged entity evidence" not in graph.text


def test_entity_graph_sensitive_region_and_references_never_reach_extractor_or_sqlite(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def extractor(context: dict[str, object]) -> dict[str, object]:
        calls.append(context)
        return {"entities": [], "relations": []}

    store = EntityGraphStore(tmp_path / "sensitive.sqlite3")
    for memory in (
        {
            "memory_id": "sensitive-region",
            "content": "Project Atlas uses Python.",
            "source_references": ["source:ok"],
            "region": {
                "region_id": "region-ok",
                "name": "api_key = region-secret",
                "kind": "topic",
            },
        },
        {
            "memory_id": "sensitive-reference",
            "content": "Project Atlas uses Python.",
            "source_references": ["source:ok", "api_key = reference-secret"],
            "region": {"region_id": "region-ok", "name": "safe", "kind": "topic"},
        },
    ):
        result = store.project("region-ok", memory, extractor)
        assert result["status"] == "skipped"
    assert calls == []
    raw = (tmp_path / "sensitive.sqlite3").read_bytes().decode("utf-8", errors="ignore")
    assert "region-secret" not in raw
    assert "reference-secret" not in raw


def test_entity_graph_rebuild_failure_preserves_old_projection_and_marks_degraded(
    tmp_path: Path,
) -> None:
    store = EntityGraphStore(tmp_path / "rebuild.sqlite3")
    memory = {
        "memory_id": "memory-old",
        "content": "Project Atlas uses Python.",
        "source_references": ["source:old"],
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }

    def working(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": ["Project Atlas uses Python."],
                }
            ],
        }

    assert store.project("graph", memory, working)["status"] == "ready"

    def failing(_: dict[str, object]) -> dict[str, object]:
        raise ValueError("model failure with secret api_key = hidden")

    result = store.rebuild("graph", [memory], failing)
    assert result["status"] == "degraded"
    assert result["error"] == "entity graph projection failed"
    graph = store.graph("graph")
    assert [node["canonical_name"] for node in graph["nodes"]] == ["Project Atlas", "Python"]
    assert graph["edges"][0]["memory_ids"] == ["memory-old"]
    assert store.status("graph")["last_error"] == "entity graph extractor returned invalid data"


def test_entity_graph_reprojection_rebuilds_entity_and_relation_evidence_per_memory(
    tmp_path: Path,
) -> None:
    store = EntityGraphStore(tmp_path / "evidence.sqlite3")

    def extractor(context: dict[str, object]) -> dict[str, object]:
        memory = context["memory"]
        assert isinstance(memory, dict)
        if memory["memory_id"] == "memory-removed":
            return {"entities": [], "relations": []}
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": ["Project Atlas uses Python."],
                }
            ],
        }

    def memory(memory_id: str) -> dict[str, object]:
        return {
            "memory_id": memory_id,
            "content": "Project Atlas uses Python.",
            "source_references": [f"source:{memory_id}"],
            "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
        }

    store.project("graph", memory("memory-removed"), extractor)
    store.project("graph", memory("memory-kept"), extractor)
    store.project("graph", memory("memory-removed"), extractor)
    graph = store.graph("graph")
    assert graph["nodes"][0]["evidence"][0]["memory_id"] == "memory-kept"
    assert graph["edges"][0]["memory_ids"] == ["memory-kept"]
    assert all(
        item.get("memory_id") == "memory-kept"
        for node in graph["nodes"]
        for item in node["evidence"]
    )


def test_entity_graph_new_canonical_name_matches_existing_explicit_alias(
    tmp_path: Path,
) -> None:
    store = EntityGraphStore(tmp_path / "aliases.sqlite3")

    def extractor(context: dict[str, object]) -> dict[str, object]:
        memory = context["memory"]
        assert isinstance(memory, dict)
        if memory["memory_id"] == "memory-one":
            name = "Atlas"
            aliases = ["Project Atlas"]
        else:
            name = "Project Atlas"
            aliases = ["Atlas"]
        return {
            "entities": [
                {"type": "project", "canonical_name": name, "aliases": aliases}
            ],
            "relations": [],
        }

    for memory_id, content in (
        ("memory-one", "Atlas is a project."),
        ("memory-two", "Project Atlas is documented."),
    ):
        assert store.project(
            "graph",
            {
                "memory_id": memory_id,
                "content": content,
                "source_references": [f"source:{memory_id}"],
                "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
            },
            extractor,
        )["status"] == "ready"
    graph = store.graph("graph")
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["evidence_memory_ids"] == ["memory-one", "memory-two"]
    assert {item["memory_id"] for item in graph["nodes"][0]["evidence"]} == {
        "memory-one",
        "memory-two",
    }


def test_entity_graph_reprojection_removes_stale_alias_before_new_canonical_resolution(
    tmp_path: Path,
) -> None:
    store = EntityGraphStore(tmp_path / "stale-alias.sqlite3")

    def extractor(context: dict[str, object]) -> dict[str, object]:
        memory_id = str(context["memory"]["memory_id"])
        content = str(context["memory"]["content"])
        if memory_id == "memory-one" and "Old Alias" in content:
            return {
                "entities": [
                    {
                        "type": "project",
                        "canonical_name": "Project Atlas",
                        "aliases": ["Old Alias"],
                    }
                ],
                "relations": [],
            }
        if memory_id == "memory-two":
            return {
                "entities": [
                    {"type": "project", "canonical_name": "Project Atlas"}
                ],
                "relations": [],
            }
        return {
            "entities": [{"type": "project", "canonical_name": "Old Alias"}],
            "relations": [],
        }

    def memory(memory_id: str, content: str) -> dict[str, object]:
        return {
            "memory_id": memory_id,
            "content": content,
            "source_references": [f"source:{memory_id}"],
            "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
        }

    assert store.project("graph", memory("memory-one", "Project Atlas Old Alias"), extractor)[
        "status"
    ] == "ready"
    assert store.project("graph", memory("memory-two", "Project Atlas"), extractor)[
        "status"
    ] == "ready"
    assert store.project("graph", memory("memory-one", "Project Atlas"), extractor)[
        "status"
    ] == "ready"

    with sqlite3.connect(tmp_path / "stale-alias.sqlite3") as connection:
        assert connection.execute(
            "SELECT 1 FROM entity_graph_aliases "
            "WHERE region_id = ? AND normalized_alias = ?",
            ("graph", "old alias"),
        ).fetchone() is None

    assert store.project("graph", memory("memory-three", "Old Alias"), extractor)[
        "status"
    ] == "ready"
    graph = store.graph("graph")
    assert len(graph["nodes"]) == 2
    old_alias_node = next(
        node for node in graph["nodes"] if node["canonical_name"] == "Old Alias"
    )
    assert old_alias_node["evidence_memory_ids"] == ["memory-three"]


def test_entity_graph_deep_sensitive_values_are_filtered_while_safe_evidence_survives(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    store = EntityGraphStore(tmp_path / "nested-sensitive.sqlite3")

    def extractor(context: dict[str, object]) -> dict[str, object]:
        calls.append(context)
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": ["Project Atlas uses Python."],
                }
            ],
        }

    memory = {
        "memory_id": "nested-sensitive",
        "content": "Project Atlas uses Python.",
        "evidence": [
            {
                "safe": {"snippet": "Project Atlas uses Python."},
                "nested": {"secret": "api_key = nested-evidence-secret"},
            }
        ],
        "evidence_snippets": [
            {"safe": "Project Atlas uses Python.", "secret": {"value": "sk-nested-secret"}}
        ],
        "source_references": ["source:nested"],
        "region": {
            "region_id": "graph",
            "name": "Graph",
            "kind": "topic",
            "metadata": {"secret": "token = nested-region-secret"},
        },
    }
    assert store.project("graph", memory, extractor)["status"] == "ready"
    assert calls
    context_text = json.dumps(calls[0], ensure_ascii=False, sort_keys=True)
    assert "nested-evidence-secret" not in context_text
    assert "nested-reference-secret" not in context_text
    assert "nested-region-secret" not in context_text
    assert "Project Atlas uses Python." in context_text
    graph_text = json.dumps(store.graph("graph"), ensure_ascii=False, sort_keys=True)
    assert "nested-evidence-secret" not in graph_text
    assert "nested-reference-secret" not in graph_text
    assert "nested-region-secret" not in graph_text
    raw = (tmp_path / "nested-sensitive.sqlite3").read_bytes().decode(
        "utf-8", errors="ignore"
    )
    assert "nested-evidence-secret" not in raw
    assert "nested-reference-secret" not in raw
    assert "nested-region-secret" not in raw


def test_entity_graph_routes_map_missing_region_and_detail_ids_to_404(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path / "state")) as client:
        state_dir = tmp_path / "state"
        headers = _headers(state_dir)
        for method, path in (
            ("get", "/api/v1/libraries/missing/entity-graph/status"),
            ("post", "/api/v1/libraries/missing/entity-graph/rebuild"),
            ("get", "/api/v1/libraries/missing/entity-graph"),
            ("get", "/api/v1/libraries/missing/entity-graph/entities/unknown"),
            ("get", "/api/v1/libraries/missing/entity-graph/relations/unknown"),
        ):
            response = getattr(client, method)(path, headers=headers)
            assert response.status_code == 404, response.text

        _region(client, headers)
        assert client.get(
            "/api/v1/libraries/graph/entity-graph/entities/unknown", headers=headers
        ).status_code == 404
        assert client.get(
            "/api/v1/libraries/graph/entity-graph/relations/unknown", headers=headers
        ).status_code == 404


def test_entity_graph_entity_detail_reads_tail_entity_without_default_graph_window(
    tmp_path: Path,
) -> None:
    store = EntityGraphStore(tmp_path / "tail.sqlite3")
    content = " ".join(f"Entity-{index:03d}" for index in range(202))

    def extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {"type": "concept", "canonical_name": f"Entity-{index:03d}"}
                for index in range(202)
            ],
            "relations": [
                {
                    "source": "concept:Entity-200",
                    "target": "concept:Entity-201",
                    "type": "related_to",
                    "evidence": [content],
                }
            ],
        }

    memory = {
        "memory_id": "memory-tail",
        "content": content,
        "source_references": ["source:tail"],
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }
    assert store.project("graph", memory, extractor)["status"] == "ready"
    entity_id = stable_entity_id("graph", "concept", "entity-201")
    detail = store.entity_detail("graph", entity_id)
    assert detail is not None
    assert detail["entity"]["canonical_name"] == "Entity-201"
    assert detail["supporting_memory_ids"] == ["memory-tail"]


def test_entity_graph_graph_edge_limit_is_bounded_at_spec_limit(tmp_path: Path) -> None:
    names = [f"Entity-{index:03d}" for index in range(200)]
    content = " ".join(names)
    pairs = [
        (source, target)
        for source_index, source in enumerate(names)
        for target_index, target in enumerate(names)
        if source_index != target_index
    ][: GRAPH_MAX_EDGE_LIMIT + 1]

    def extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {"type": "concept", "canonical_name": name} for name in names
            ],
            "relations": [
                {
                    "source": f"concept:{source}",
                    "target": f"concept:{target}",
                    "type": "related_to",
                    "evidence": [content],
                }
                for source, target in pairs
            ],
        }

    store = EntityGraphStore(tmp_path / "edge-limit.sqlite3")
    assert store.project(
        "graph",
        {
            "memory_id": "memory-edge-limit",
            "content": content,
            "source_references": ["source:edge-limit"],
            "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
        },
        extractor,
    )["status"] == "ready"
    graph = store.graph("graph", limit=200)
    assert len(graph["nodes"]) == 200
    assert len(graph["edges"]) == GRAPH_MAX_EDGE_LIMIT
    assert all(len(edge["memory_ids"]) <= GRAPH_SUPPORTING_MEMORY_LIMIT for edge in graph["edges"])


def test_memory_detail_does_not_wait_for_slow_entity_graph_update(tmp_path: Path) -> None:
    extractor_started = threading.Event()
    release_extractor = threading.Event()

    def extractor(context: dict[str, object]) -> dict[str, object]:
        extractor_started.set()
        assert release_extractor.wait(timeout=5)
        content = str(context["memory"]["content"])
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": [content],
                }
            ],
        }

    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir, extractor=extractor)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(client, headers, "Project Atlas uses Python.", "slow")
        assert extractor_started.wait(timeout=5)
        kernel = client.app.state.memory_kernel
        kernel.inspect_memory = lambda request: {  # type: ignore[method-assign]
            "id": str(request.path),
            "content": "Project Atlas uses Python.",
        }

        completed = threading.Event()
        result: dict[str, Any] = {}

        def read_detail() -> None:
            result["response"] = client.get(
                f"/api/v1/libraries/graph/memories/{memory_id}", headers=headers
            )
            completed.set()

        reader = threading.Thread(target=read_detail)
        reader.start()
        try:
            assert completed.wait(timeout=1), "memory detail waited for entity graph update"
            response = result["response"]
            assert response.status_code == 200, response.text
        finally:
            release_extractor.set()
            reader.join(timeout=10)
        assert not reader.is_alive()


def test_entity_graph_recovery_rebuilds_persisted_regions_after_restart(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"

    def extractor(context: dict[str, object]) -> dict[str, object]:
        content = str(context["memory"]["content"])
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": [content],
                }
            ],
        }

    first_kernel = Mem2MemoryKernel(state_dir, graph_extractor=extractor)
    with TestClient(
        create_app(Settings(state_dir=state_dir), memory_kernel=first_kernel)
    ) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        _memory(client, headers, "Project Atlas uses Python.", "restart")
        before = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert before.status_code == 200, before.text
        expected = before.json()
        assert len(expected["nodes"]) == 2
        assert len(expected["edges"]) == 1
        first_kernel._entity_graph_store.clear("graph")
        assert first_kernel._entity_graph_store.graph("graph")["nodes"] == []

    second_kernel = Mem2MemoryKernel(state_dir, graph_extractor=extractor)
    with TestClient(
        create_app(Settings(state_dir=state_dir), memory_kernel=second_kernel)
    ) as client:
        headers = _headers(state_dir)
        recovered: dict[str, object] | None = None
        for _ in range(100):
            candidate = second_kernel.entity_graph(
                "graph", wait_for_update=False
            )
            if candidate["nodes"] and candidate["edges"]:
                recovered = candidate
                break
            time.sleep(0.05)
        assert recovered is not None
        assert recovered["nodes"] == expected["nodes"]
        assert recovered["edges"] == expected["edges"]

        rebuilt = client.post(
            "/api/v1/libraries/graph/entity-graph/rebuild", headers=headers
        )
        assert rebuilt.status_code == 202, rebuilt.text
        after = client.get("/api/v1/libraries/graph/entity-graph", headers=headers)
        assert after.status_code == 200, after.text
        assert after.json()["nodes"] == expected["nodes"]
        assert after.json()["edges"] == expected["edges"]


def test_entity_graph_ignores_metadata_only_evidence_but_accepts_saved_text(
    tmp_path: Path,
) -> None:
    metadata_memory = {
        "memory_id": "metadata-only",
        "content": "The body does not name graph objects.",
        "evidence": [
            {
                "event_id": "event-not-evidence",
                "id": "event-not-evidence",
                "region_id": "region-not-evidence",
                "source_id": "source-not-evidence",
            }
        ],
        "source_references": ["source:metadata-only"],
        "region": {"region_id": "graph", "name": "Unrelated", "kind": "topic"},
    }

    def metadata_extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {"type": "concept", "canonical_name": "event-not-evidence"},
                {"type": "reference", "canonical_name": "source-not-evidence"},
            ],
            "relations": [
                {
                    "source": "concept:event-not-evidence",
                    "target": "reference:source-not-evidence",
                    "type": "related_to",
                    "evidence": ["event-not-evidence"],
                }
            ],
        }

    store = EntityGraphStore(tmp_path / "metadata-only.sqlite3")
    result = store.project("graph", metadata_memory, metadata_extractor)
    assert result["rejected"] == 3
    assert store.graph("graph")["nodes"] == []
    assert store.graph("graph")["edges"] == []

    saved_evidence_memory = {
        "memory_id": "saved-evidence",
        "content": "The body does not name graph objects.",
        "evidence": [
            {
                "event_id": "event-with-evidence",
                "content": "Project Atlas uses Python.",
                "region_id": "graph",
            }
        ],
        "source_references": ["source:saved-evidence"],
        "region": {"region_id": "graph", "name": "Unrelated", "kind": "topic"},
    }

    def saved_evidence_extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {"type": "project", "canonical_name": "Project Atlas"},
                {"type": "technology", "canonical_name": "Python"},
            ],
            "relations": [
                {
                    "source": "project:Project Atlas",
                    "target": "technology:Python",
                    "type": "uses",
                    "evidence": [{"snippet": "Project Atlas uses Python."}],
                }
            ],
        }

    assert store.project("graph", saved_evidence_memory, saved_evidence_extractor)[
        "status"
    ] == "ready"
    graph = store.graph("graph")
    assert {node["canonical_name"] for node in graph["nodes"]} == {
        "Project Atlas",
        "Python",
    }
    assert len(graph["edges"]) == 1


def test_entity_and_relation_detail_bound_supporting_memory_reads(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    kernel = Mem2MemoryKernel(
        state_dir,
        graph_extractor=lambda _: {"entities": [], "relations": []},
    )
    with TestClient(create_app(Settings(state_dir=state_dir), memory_kernel=kernel)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        store = kernel._entity_graph_store

        def extractor(context: dict[str, object]) -> dict[str, object]:
            content = str(context["memory"]["content"])
            return {
                "entities": [
                    {"type": "project", "canonical_name": "Project Atlas"},
                    {"type": "technology", "canonical_name": "Python"},
                ],
                "relations": [
                    {
                        "source": "project:Project Atlas",
                        "target": "technology:Python",
                        "type": "uses",
                        "evidence": [content],
                    }
                ],
            }

        for index in range(120):
            memory_id = f"memory-{index:03d}"
            assert store.project(
                "graph",
                {
                    "memory_id": memory_id,
                    "content": f"Project Atlas uses Python in memory {index:03d}.",
                    "source_references": [f"source:{index:03d}"],
                    "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
                },
                extractor,
            )["status"] == "ready"

        entity_id = stable_entity_id("graph", "project", "project atlas")
        source_id = stable_entity_id("graph", "project", "project atlas")
        target_id = stable_entity_id("graph", "technology", "python")
        relation_id = stable_relation_id("graph", source_id, target_id, "uses")
        inspect_calls: list[str] = []

        def fake_inspect(request: Any) -> dict[str, object]:
            inspect_calls.append(str(request.path))
            return {
                "id": str(request.path),
                "content": f"full body for {request.path}",
            }

        kernel.inspect_memory = fake_inspect  # type: ignore[method-assign]

        default_entity_response = client.get(
            f"/api/v1/libraries/graph/entity-graph/entities/{entity_id}",
            headers=headers,
        )
        assert default_entity_response.status_code == 200, default_entity_response.text
        default_entity_payload = default_entity_response.json()
        assert default_entity_payload["supporting_memory_limit"] == 50
        assert len(default_entity_payload["supporting_memory_ids"]) == 50
        assert len(default_entity_payload["supporting_memories"]) == 50
        assert len(inspect_calls) == 50
        inspect_calls.clear()

        entity_response = client.get(
            f"/api/v1/libraries/graph/entity-graph/entities/{entity_id}?limit=10",
            headers=headers,
        )
        assert entity_response.status_code == 200, entity_response.text
        entity_payload = entity_response.json()
        assert entity_payload["supporting_memory_count"] == 120
        assert len(entity_payload["supporting_memory_ids"]) == 10
        assert len(entity_payload["supporting_memories"]) == 10
        assert entity_payload["has_more"] is True
        assert entity_payload["next_cursor"] == "10"
        assert len(inspect_calls) == 10
        assert "full body for memory-119" not in entity_response.text

        cursor_entity_response = client.get(
            f"/api/v1/libraries/graph/entity-graph/entities/{entity_id}?limit=10&cursor=10",
            headers=headers,
        )
        assert cursor_entity_response.status_code == 200, cursor_entity_response.text
        cursor_entity_payload = cursor_entity_response.json()
        assert cursor_entity_payload["supporting_memory_offset"] == 10
        assert cursor_entity_payload["supporting_memory_ids"] == [
            f"memory-{index:03d}" for index in range(10, 20)
        ]
        assert cursor_entity_payload["next_cursor"] == "20"
        assert len(inspect_calls) == 20
        inspect_calls.clear()

        direct_entity_payload = kernel.entity_graph_entity(
            "graph", entity_id, limit=10, cursor="20"
        )
        assert direct_entity_payload["supporting_memory_offset"] == 20
        assert direct_entity_payload["supporting_memory_ids"] == [
            f"memory-{index:03d}" for index in range(20, 30)
        ]

        inspect_calls.clear()
        relation_response = client.get(
            f"/api/v1/libraries/graph/entity-graph/relations/{relation_id}?limit=10&offset=110",
            headers=headers,
        )
        assert relation_response.status_code == 200, relation_response.text
        relation_payload = relation_response.json()
        assert relation_payload["supporting_memory_count"] == 120
        assert relation_payload["supporting_memory_ids"] == [
            f"memory-{index:03d}" for index in range(110, 120)
        ]
        assert len(relation_payload["supporting_memories"]) == 10
        assert relation_payload["has_more"] is False
        assert relation_payload["next_cursor"] is None
        assert len(inspect_calls) == 10
        assert "full body for memory-000" not in relation_response.text

        cursor_relation_response = client.get(
            f"/api/v1/libraries/graph/entity-graph/relations/{relation_id}?limit=10&cursor=110",
            headers=headers,
        )
        assert cursor_relation_response.status_code == 200, cursor_relation_response.text
        cursor_relation_payload = cursor_relation_response.json()
        assert cursor_relation_payload["supporting_memory_offset"] == 110
        assert cursor_relation_payload["supporting_memory_ids"] == [
            f"memory-{index:03d}" for index in range(110, 120)
        ]
        direct_relation_payload = kernel.entity_graph_relation(
            "graph", relation_id, limit=10, cursor="110"
        )
        assert direct_relation_payload["supporting_memory_offset"] == 110
        assert direct_relation_payload["supporting_memory_ids"] == [
            f"memory-{index:03d}" for index in range(110, 120)
        ]

        assert (
            client.get(
                f"/api/v1/libraries/graph/entity-graph/entities/{entity_id}?limit=10&offset=10&cursor=10",
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            client.get(
                f"/api/v1/libraries/graph/entity-graph/relations/{relation_id}?limit=10&cursor=not-an-offset",
                headers=headers,
            ).status_code
            == 422
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            kernel.entity_graph_entity(
                "graph", entity_id, limit=10, offset=10, cursor="10"
            )

        full_graph = kernel.entity_graph("graph")
        assert all(
            len(node["evidence"]) <= GRAPH_SUPPORTING_EVIDENCE_LIMIT
            and len(node["evidence_memory_ids"]) <= GRAPH_SUPPORTING_MEMORY_LIMIT
            for node in full_graph["nodes"]
        )
        assert all(
            len(edge["evidence"]) <= GRAPH_SUPPORTING_EVIDENCE_LIMIT
            and len(edge["memory_ids"]) <= GRAPH_SUPPORTING_MEMORY_LIMIT
            for edge in full_graph["edges"]
        )
        local_graph = kernel.entity_graph("graph", memory_id="memory-000")
        assert all(
            len(node["evidence"]) <= GRAPH_SUPPORTING_EVIDENCE_LIMIT
            and len(node["evidence_memory_ids"]) <= GRAPH_SUPPORTING_MEMORY_LIMIT
            for node in local_graph["nodes"]
        )
        assert all(
            len(edge["evidence"]) <= GRAPH_SUPPORTING_EVIDENCE_LIMIT
            and len(edge["memory_ids"]) <= GRAPH_SUPPORTING_MEMORY_LIMIT
            for edge in local_graph["edges"]
        )
        memory_detail = store.memory_detail("graph", "memory-000")
        assert all(
            len(entity["evidence"]) <= GRAPH_SUPPORTING_EVIDENCE_LIMIT
            for entity in memory_detail["entities"]
        )
        assert all(
            len(relation["memory_ids"]) <= GRAPH_SUPPORTING_MEMORY_LIMIT
            and len(relation["evidence"]) <= GRAPH_SUPPORTING_EVIDENCE_LIMIT
            for relation in memory_detail["relations"]
        )

        assert (
            client.get(
                f"/api/v1/libraries/graph/entity-graph/entities/{entity_id}?limit=51",
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            len(store.entity_detail("graph", entity_id, limit=1_000)["supporting_memory_ids"])
            == 50
        )


def test_entity_graph_history_search_type_filter_and_degree_ordering(tmp_path: Path) -> None:
    def extractor(context: dict[str, object]) -> dict[str, object]:
        content = str(context["memory"]["content"])
        source, target = content.removesuffix(".").split(" uses ", 1)
        return {
            "entities": [
                {"type": "concept", "canonical_name": source},
                {"type": "technology", "canonical_name": target},
            ],
            "relations": [
                {
                    "source": f"concept:{source}",
                    "target": f"technology:{target}",
                    "type": "uses",
                    "evidence": [content],
                }
            ],
        }

    store = EntityGraphStore(tmp_path / "bounded-history.sqlite3")
    base = {
        "source_references": ["source:bounded"],
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }
    memories = (
        {
            **base,
            "memory_id": "hub-alpha",
            "content": "Hub uses Alpha.",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            **base,
            "memory_id": "hub-beta",
            "content": "Hub uses Beta.",
            "updated_at": "2026-01-02T00:00:00+00:00",
        },
        {
            **base,
            "memory_id": "hub-gamma",
            "content": "Hub uses Gamma.",
            "updated_at": "2025-01-01T00:00:00+00:00",
        },
        {
            **base,
            "memory_id": "fresh",
            "content": "Fresh uses NewTool.",
            "updated_at": "2026-02-01T00:00:00+00:00",
        },
        {
            **base,
            "memory_id": "legacy",
            "content": "Legacy uses OldTool.",
            "lifecycle": "superseded",
            "updated_at": "2025-12-01T00:00:00+00:00",
        },
        {
            **base,
            "memory_id": "forgotten",
            "content": "Forgotten uses Ghost.",
            "lifecycle": "forgotten",
            "updated_at": "2025-11-01T00:00:00+00:00",
        },
    )
    for memory in memories:
        result = store.project("graph", memory, extractor)
        assert result["status"] in {"ready", "skipped"}

    current = store.graph("graph", limit=GRAPH_MAX_NODE_LIMIT)
    current_names = {node["canonical_name"] for node in current["nodes"]}
    assert "Legacy" not in current_names
    assert "OldTool" not in current_names
    assert "Forgotten" not in current_names
    assert "Ghost" not in current_names
    assert current["nodes"][0]["canonical_name"] == "Hub"
    assert current["nodes"][0]["degree"] == 3
    current_order = [node["canonical_name"] for node in current["nodes"]]
    assert current_order.index("Hub") < current_order.index("Fresh")
    assert current["history"] is False

    for seed in (
        {"entity_id": stable_entity_id("graph", "concept", "No such entity")},
        {"memory_id": "no-such-memory"},
        {"q": "No such entity"},
    ):
        missing = store.graph("graph", hops=2, **seed)
        assert missing["nodes"] == []
        assert missing["edges"] == []
        assert missing["hops"] == 2
        assert missing["max_nodes"] == GRAPH_MAX_NODE_LIMIT
        assert missing["max_edges"] == GRAPH_MAX_EDGE_LIMIT
        assert missing["read_only"] is True

    searched = store.graph("graph", q="Beta", hops=1)
    assert [node["canonical_name"] for node in searched["nodes"]][:2] == [
        "Beta",
        "Hub",
    ]
    assert searched["nodes"][0]["graph_state"] == "direct"
    assert searched["nodes"][1]["graph_state"] == "background"

    filtered = store.graph("graph", q="Beta", entity_type="technology")
    assert [node["canonical_name"] for node in filtered["nodes"]] == ["Beta"]
    assert filtered["edges"] == []

    history = store.graph("graph", history=True, limit=GRAPH_MAX_NODE_LIMIT)
    history_names = {node["canonical_name"] for node in history["nodes"]}
    assert {"Legacy", "OldTool"}.issubset(history_names)
    assert "Forgotten" not in history_names
    historical_edges = [edge for edge in history["edges"] if edge["historical"]]
    assert historical_edges
    assert historical_edges[0]["display_state"] == "historical"
    assert "forgotten" not in json.dumps(history, ensure_ascii=False)


def test_entity_graph_search_escapes_like_wildcards_and_keeps_literal_queries(
    tmp_path: Path,
) -> None:
    def extractor(context: dict[str, object]) -> dict[str, object]:
        content = str(context["memory"]["content"])
        return {
            "entities": [
                {
                    "type": "concept",
                    "canonical_name": content.split(" | ", 1)[0],
                    "aliases": [content.split(" | ", 1)[1]],
                }
            ],
            "relations": [],
        }

    store = EntityGraphStore(tmp_path / "literal-search.sqlite3")
    base = {
        "source_references": ["source:literal-search"],
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }
    memories = (
        ("percent", "PercentEntity | literal%alias"),
        ("underscore", "UnderscoreEntity | literal_alias"),
        ("backslash", r"BackslashEntity | literal\alias"),
        ("normal", "NormalEntity | normal-alias"),
    )
    for memory_id, content in memories:
        memory = {
            **base,
            "memory_id": memory_id,
            "content": content,
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        assert store.project("graph", memory, extractor)["status"] == "ready"

    expected_by_query = {
        "%": "PercentEntity",
        "_": "UnderscoreEntity",
        "\\": "BackslashEntity",
        "normal": "NormalEntity",
    }
    for query, expected_name in expected_by_query.items():
        graph = store.graph("graph", q=query)
        assert [node["canonical_name"] for node in graph["nodes"]] == [expected_name]

    assert store.graph("graph", q="missing")["nodes"] == []
    assert store.graph("graph", q="literal%missing")["nodes"] == []
    assert store.graph("graph", q="   ")["nodes"] == []


def test_entity_graph_memory_seed_reads_a_bounded_entity_set(tmp_path: Path) -> None:
    names = [f"Entity-{index:03d}" for index in range(GRAPH_MAX_NODE_LIMIT + 5)]
    content = " ".join(names)

    def extractor(_: dict[str, object]) -> dict[str, object]:
        return {
            "entities": [
                {"type": "concept", "canonical_name": name} for name in names
            ],
            "relations": [],
        }

    store = EntityGraphStore(tmp_path / "bounded-memory-seed.sqlite3")
    result = store.project(
        "graph",
        {
            "memory_id": "memory-with-many-entities",
            "content": content,
            "source_references": ["source:bounded-memory-seed"],
            "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        extractor,
    )
    assert result["status"] == "ready"

    graph = store.graph(
        "graph",
        memory_id="memory-with-many-entities",
        limit=GRAPH_MAX_NODE_LIMIT,
    )
    assert len(graph["nodes"]) == GRAPH_MAX_NODE_LIMIT
    assert all(node["selected"] is True for node in graph["nodes"])
    assert len(graph["edges"]) <= GRAPH_MAX_EDGE_LIMIT


def test_unavailable_entity_graph_accepts_history_query_without_type_error(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    app = create_app(
        Settings(state_dir=state_dir),
        memory_kernel=UnavailableMemoryKernel(),
    )
    with TestClient(app) as client:
        headers = _headers(state_dir)
        response = client.get(
            "/api/v1/libraries/graph/entity-graph?history=true",
            headers=headers,
        )
    assert response.status_code == 503, response.text
    assert "memory kernel is unavailable" in response.json()["detail"]

    with pytest.raises(MemoryKernelUnavailableError):
        UnavailableMemoryKernel().entity_graph("graph", history=True)


def test_entity_graph_status_and_direct_search_degrade_independently(
    tmp_path: Path,
) -> None:
    def unavailable(_: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("deterministic entity graph outage")

    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir, extractor=unavailable)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(client, headers, "Direct retrieval remains available.", "degrade")

        status = client.get("/api/v1/libraries/graph/entity-graph/status", headers=headers)
        assert status.status_code == 200, status.text
        status_payload = status.json()
        assert status_payload["status"] == "degraded"
        assert status_payload["nodes"] == 0
        assert status_payload["edges"] == 0
        assert status_payload["pending"] == 0
        assert status_payload["error"]
        assert status_payload["updated_at"]

        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": "graph",
                "query": "Direct retrieval",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        payload = search.json()
        assert any(item["document_id"] == memory_id for item in payload["results"])
        assert payload["entity_graph_status"]["status"] == "degraded"
        assert "graph_unavailable" in payload["degradation"]


def test_library_search_falls_back_to_legacy_scope_after_region_registration(
    tmp_path: Path,
) -> None:
    def unavailable(_: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("deterministic entity graph outage")

    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir, extractor=unavailable)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        region_memory_id = _memory(
            client,
            headers,
            "Current region content is unrelated.",
            "region-only",
        )
        legacy_response = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "library_id": "graph",
                "content": "Legacy library marker remains searchable.",
                "memory_type": "domain_fact",
                "source_references": ["source:legacy-library"],
                "idempotency_key": "legacy-library",
            },
        )
        assert legacy_response.status_code == 201, legacy_response.text
        legacy_memory_id = str(legacy_response.json()["id"])

        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": "graph",
                "query": "Legacy library marker",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert search.status_code == 200, search.text
        payload = search.json()
        result_ids = {str(item["document_id"]) for item in payload["results"]}
        assert legacy_memory_id in result_ids
        assert region_memory_id not in result_ids
        assert payload["entity_graph_status"]["status"] == "degraded"
        assert "graph_unavailable" in payload["degradation"]

        explicit_region = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph",
                "query": "Legacy library marker",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert explicit_region.status_code == 200, explicit_region.text
        assert explicit_region.json()["results"] == []


def test_entity_graph_rejects_unknown_entity_type_and_accepts_fixed_type(
    tmp_path: Path,
) -> None:
    with TestClient(_app(tmp_path / "state")) as client:
        state_dir = tmp_path / "state"
        headers = _headers(state_dir)
        _region(client, headers)
        _memory(client, headers, "Project Atlas uses Python.", "type-filter")

        invalid = client.get(
            "/api/v1/libraries/graph/entity-graph",
            params={"entity_type": "invalid"},
            headers=headers,
        )
        assert invalid.status_code == 422, invalid.text
        assert "entity_type must be one of" in invalid.json()["detail"]

        valid = client.get(
            "/api/v1/libraries/graph/entity-graph",
            params={"entity_type": "technology"},
            headers=headers,
        )
        assert valid.status_code == 200, valid.text
        assert all(node["entity_type"] == "technology" for node in valid.json()["nodes"])


def test_entity_graph_two_hop_search_keeps_direct_seed_before_background_limit(
    tmp_path: Path,
) -> None:
    def extractor(context: dict[str, object]) -> dict[str, object]:
        source, target = str(context["memory"]["content"]).split(" connects ", 1)
        return {
            "entities": [
                {"type": "concept", "canonical_name": source},
                {"type": "concept", "canonical_name": target},
            ],
            "relations": [
                {
                    "source": f"concept:{source}",
                    "target": f"concept:{target}",
                    "type": "related_to",
                    "evidence": [str(context["memory"]["content"])],
                }
            ],
        }

    store = EntityGraphStore(tmp_path / "two-hop-seed.sqlite3")
    base = {
        "source_references": ["source:two-hop-seed"],
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }
    seed_memory = {
        **base,
        "memory_id": "seed-hub",
        "content": "Seed connects Hub",
        "updated_at": "2020-01-01T00:00:00+00:00",
    }
    assert store.project("graph", seed_memory, extractor)["status"] in {"ready", "skipped"}
    for index in range(250):
        memory = {
            **base,
            "memory_id": f"hub-neighbor-{index:03d}",
            "content": f"Hub connects Neighbor-{index:03d}",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        assert store.project("graph", memory, extractor)["status"] in {"ready", "skipped"}

    graph = store.graph("graph", q="Seed", hops=2, limit=GRAPH_MAX_NODE_LIMIT)
    seed = next(node for node in graph["nodes"] if node["canonical_name"] == "Seed")
    assert seed["selected"] is True
    assert seed["graph_state"] == "direct"
    assert "Hub" in {node["canonical_name"] for node in graph["nodes"]}
    assert len(graph["nodes"]) <= GRAPH_MAX_NODE_LIMIT
    assert len(graph["edges"]) <= GRAPH_MAX_EDGE_LIMIT


def test_entity_query_subgraph_marks_direct_and_expansion_entities_by_hop(
    tmp_path: Path,
) -> None:
    def extractor(context: dict[str, object]) -> dict[str, object]:
        source, target = str(context["memory"]["content"]).split(" connects ", 1)
        return {
            "entities": [
                {"type": "concept", "canonical_name": source},
                {"type": "concept", "canonical_name": target},
            ],
            "relations": [
                {
                    "source": f"concept:{source}",
                    "target": f"concept:{target}",
                    "type": "related_to",
                    "evidence": [str(context["memory"]["content"])],
                }
            ],
        }

    store = EntityGraphStore(tmp_path / "query-subgraph.sqlite3")
    base = {
        "source_references": ["source:query-subgraph"],
        "region": {"region_id": "graph", "name": "Graph", "kind": "topic"},
    }
    for memory_id, content in (
        ("seed-hub", "Seed connects Hub"),
        ("hub-mid", "Hub connects Mid"),
        ("mid-leaf", "Mid connects Leaf"),
    ):
        assert store.project(
            "graph",
            {
                **base,
                "memory_id": memory_id,
                "content": content,
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            extractor,
        )["status"] == "ready"

    one_hop = store.query_subgraph("graph", ["seed-hub"], hops=1)
    one_by_name = {str(node["canonical_name"]): node for node in one_hop["nodes"]}
    assert one_by_name["Seed"]["visual_state"] == "direct"
    assert one_by_name["Hub"]["visual_state"] == "direct"
    assert one_by_name["Mid"]["visual_state"] == "query_expansion"
    assert "Leaf" not in one_by_name
    assert one_hop["memory_entities"]["seed-hub"] == [
        one_by_name["Seed"]["id"],
        one_by_name["Hub"]["id"],
    ]

    two_hop = store.query_subgraph("graph", ["seed-hub"], hops=2)
    two_by_name = {str(node["canonical_name"]): node for node in two_hop["nodes"]}
    assert two_by_name["Leaf"]["visual_state"] == "query_expansion"
    assert len(two_hop["nodes"]) <= GRAPH_MAX_NODE_LIMIT
    assert len(two_hop["edges"]) <= GRAPH_MAX_EDGE_LIMIT


def test_search_returns_entity_query_subgraph_and_memory_mapping(
    tmp_path: Path,
) -> None:
    def extractor(context: dict[str, object]) -> dict[str, object]:
        source, target = str(context["memory"]["content"]).split(" connects ", 1)
        return {
            "entities": [
                {"type": "concept", "canonical_name": source},
                {"type": "concept", "canonical_name": target},
            ],
            "relations": [
                {
                    "source": f"concept:{source}",
                    "target": f"concept:{target}",
                    "type": "related_to",
                    "evidence": [str(context["memory"]["content"])],
                }
            ],
        }

    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir, extractor=extractor)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        seed_id = _memory(client, headers, "Seed connects Hub", "query-seed")
        _memory(client, headers, "Hub connects Mid", "query-mid")
        _memory(client, headers, "Mid connects Leaf", "query-leaf")

        one_hop = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph",
                "query": "Seed",
                "graph_hops": 1,
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert one_hop.status_code == 200, one_hop.text
        one_payload = one_hop.json()
        one_graph = one_payload["query_subgraph"]
        assert one_graph["status"] == "ready"
        assert one_graph["hops"] == 1
        assert one_graph["read_only"] is True
        assert len(one_graph["nodes"]) <= GRAPH_MAX_NODE_LIMIT
        assert len(one_graph["edges"]) <= GRAPH_MAX_EDGE_LIMIT
        one_names = {node["canonical_name"] for node in one_graph["nodes"]}
        assert {"Seed", "Hub", "Mid"} <= one_names
        assert "Leaf" not in one_names
        seed_result = next(
            item for item in one_payload["results"] if item["document_id"] == seed_id
        )
        assert seed_result["graph_mapped"] is True
        assert seed_result["graph_entity_ids"]

        two_hop = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph",
                "query": "Seed",
                "graph_hops": 2,
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert two_hop.status_code == 200, two_hop.text
        two_graph = two_hop.json()["query_subgraph"]
        assert two_graph["hops"] == 2
        assert "Leaf" in {node["canonical_name"] for node in two_graph["nodes"]}

        rejected = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph",
                "query": "Seed",
                "graph_hops": 3,
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert rejected.status_code == 422, rejected.text


def test_search_keeps_results_when_entity_mapping_is_missing_or_degraded(
    tmp_path: Path,
) -> None:
    def no_entities(_: dict[str, object]) -> dict[str, object]:
        return {"entities": [], "relations": []}

    state_dir = tmp_path / "no-entities"
    with TestClient(_app(state_dir, extractor=no_entities)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(
            client,
            headers,
            "A formal memory without graph entities",
            "no-entities",
        )
        response = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph",
                "query": "formal memory",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert any(item["document_id"] == memory_id for item in payload["results"])
        assert payload["query_subgraph"]["status"] == "no_mapped_entities"
        assert payload["query_subgraph"]["nodes"] == []

    def unavailable(_: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("deterministic query subgraph outage")

    state_dir = tmp_path / "degraded"
    with TestClient(_app(state_dir, extractor=unavailable)) as client:
        headers = _headers(state_dir)
        _region(client, headers)
        memory_id = _memory(client, headers, "A searchable degraded memory", "degraded")
        response = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "graph",
                "query": "searchable degraded",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert any(item["document_id"] == memory_id for item in payload["results"])
        assert payload["query_subgraph"]["status"] == "degraded"
        assert "graph_unavailable" in payload["degradation"]
