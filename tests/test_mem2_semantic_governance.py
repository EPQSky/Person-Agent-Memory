from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jiuwen_memory.storage.types import Node

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.mem2_kernel import Mem2MemoryKernel

SemanticExtractor = Callable[[dict[str, object]], list[dict[str, object]]]
PromotionClassifier = Callable[[str, Mapping[str, object]], Mapping[str, object]]


class ManualSemanticExecutor:
    def __init__(self) -> None:
        self._tasks: deque[Callable[[], None]] = deque()
        self._lock = threading.Lock()

    @property
    def worker_count(self) -> int:
        return 1

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._tasks)

    def submit(self, task_key: str, region_id: str, task: Callable[[], None]) -> bool:
        del task_key, region_id
        with self._lock:
            self._tasks.append(task)
        return True

    def drain(self) -> None:
        while self.pending_count:
            with self._lock:
                task = self._tasks.popleft()
            task()

    def close(self) -> None:
        self.drain()


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _semantic_app(
    state_dir: Path,
    extractor: SemanticExtractor,
    executor: ManualSemanticExecutor,
) -> FastAPI:
    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    return create_app(Settings(state_dir=state_dir), memory_kernel=kernel)


def _promotion_app(state_dir: Path, classifier: PromotionClassifier) -> FastAPI:
    kernel = Mem2MemoryKernel(state_dir, promotion_classifier=classifier)
    return create_app(Settings(state_dir=state_dir), memory_kernel=kernel)


def _region(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    *,
    kind: str = "project",
    name: str | None = None,
) -> None:
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json={"kind": kind, "name": name or region_id, "region_id": region_id},
    )
    assert response.status_code == 201, response.text


def _bind(client: TestClient, headers: dict[str, str], root: Path, region_id: str) -> None:
    response = client.post(
        "/api/v1/project-bindings",
        headers=headers,
        json={"project_root": str(root), "region_id": region_id},
    )
    assert response.status_code == 201, response.text


def _capture_turn(
    client: TestClient,
    headers: dict[str, str],
    *,
    session_id: str,
    turn_id: str,
    user_content: str,
    assistant_content: str,
    root: Path,
    second: int,
) -> None:
    for kind, content, offset in (
        ("user", user_content, 0),
        ("assistant", assistant_content, 1),
    ):
        response = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": f"{turn_id}-{kind}",
                "session_id": session_id,
                "turn_id": turn_id,
                "event_kind": kind,
                "content": content,
                "occurred_at": f"2026-10-28T02:00:{second + offset:02d}Z",
                "cwd": str(root),
            },
        )
        assert response.status_code == 202, response.text


def _round(
    client: TestClient,
    headers: dict[str, str],
    *,
    region_id: str,
    session_id: str,
    turn_id: str,
) -> dict[str, object]:
    response = client.get(
        "/api/v1/capture/rounds",
        headers=headers,
        params={"region_id": region_id, "session_id": session_id, "turn_id": turn_id},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload) == 1, payload
    return dict(payload[0])


def _memories(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
) -> list[dict[str, object]]:
    response = client.get(
        f"/api/v1/libraries/{region_id}/memories?limit=100",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return list(response.json()["items"])


def _detail(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    memory_id: str,
) -> dict[str, object]:
    response = client.get(
        f"/api/v1/libraries/{region_id}/memories/{memory_id}",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def _memory(
    *,
    content: str,
    memory_type: str,
    evidence: str,
    event_id: str,
    subject: str,
    value: str,
    region_hint: str,
) -> dict[str, object]:
    return {
        "content": content,
        "memory_type": memory_type,
        "confidence": 0.97,
        "region_hint": region_hint,
        "user_event_id": event_id,
        "evidence_snippet": evidence,
        "claim_subject": subject,
        "claim_value": value,
        "durability": "durable_statement",
    }


def _search(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    query: str,
) -> dict[str, object]:
    response = client.post(
        "/api/v1/search",
        headers=headers,
        json={"region_id": region_id, "query": query, "limit": 20, "token_budget": 10_000},
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def _wait_promotion(
    client: TestClient,
    headers: dict[str, str],
    job_id: str,
    *,
    timeout: float = 15.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/promotions/{job_id}", headers=headers)
        assert response.status_code == 200, response.text
        payload = dict(response.json())
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"promotion job {job_id} did not finish")


def test_semantic_routing_reuse_conflict_sensitive_and_retry_are_auditable(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()
    project_content = "The build system uses pnpm in this project."
    topic_content = "Rust ownership prevents data races."
    preference_content = "The user prefers dark mode."
    conflict_content = "The user prefers light mode."
    secret = "correct-horse-battery-staple"
    sensitive_content = f"The service credential is password='{secret}'."

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        )
        event_id = str(user["event_id"])
        if event_id == "routing-turn-user":
            return [
                _memory(
                    content=preference_content,
                    memory_type="preference",
                    evidence="I prefer dark mode",
                    event_id=event_id,
                    subject="user:preference:theme",
                    value="dark",
                    region_hint="user",
                ),
                _memory(
                    content=project_content,
                    memory_type="constraint",
                    evidence="The build system uses pnpm in this project",
                    event_id=event_id,
                    subject="project package manager",
                    value="pnpm",
                    region_hint="user",
                ),
                _memory(
                    content=topic_content,
                    memory_type="domain_fact",
                    evidence="Rust ownership prevents data races",
                    event_id=event_id,
                    subject="rust ownership",
                    value="prevents data races",
                    region_hint="rust-topic",
                ),
                _memory(
                    content="The user prefers the missing topic.",
                    memory_type="preference",
                    evidence="I prefer the missing topic",
                    event_id=event_id,
                    subject="user:preference:missing-topic",
                    value="missing",
                    region_hint="missing-topic",
                ),
            ]
        if event_id == "duplicate-turn-user":
            return [
                _memory(
                    content=preference_content,
                    memory_type="preference",
                    evidence="I prefer dark mode",
                    event_id=event_id,
                    subject="user:preference:theme",
                    value="dark",
                    region_hint="user",
                )
            ]
        if event_id == "conflict-turn-user":
            return [
                _memory(
                    content=conflict_content,
                    memory_type="preference",
                    evidence="I prefer light mode",
                    event_id=event_id,
                    subject="user:preference:theme",
                    value="light",
                    region_hint="user",
                )
            ]
        if event_id == "sensitive-turn-user":
            return [
                _memory(
                    content=sensitive_content,
                    memory_type="domain_fact",
                    evidence=f"password='{secret}'",
                    event_id=event_id,
                    subject="service credential",
                    value=secret,
                    region_hint="user",
                )
            ]
        raise AssertionError(f"unexpected semantic event: {event_id}")

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "semantic-project")
        _region(client, headers, "rust-topic", kind="topic")
        _bind(client, headers, root, "semantic-project")

        _capture_turn(
            client,
            headers,
            session_id="governance-session",
            turn_id="routing-turn",
            user_content=(
                "I prefer dark mode. The build system uses pnpm in this project. "
                "Rust ownership prevents data races. I prefer the missing topic."
            ),
            assistant_content="Noted.",
            root=root,
            second=0,
        )
        executor.drain()
        routed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="governance-session",
            turn_id="routing-turn",
        )
        assert routed["semantic_status"] == "completed", routed
        assert len(routed["semantic_memory_ids"]) == 4
        routing_diagnostics = " ".join(
            str(item) for item in routed["semantic_diagnostics"]
        )
        assert "project-scoped content was retained" in routing_diagnostics
        assert "missing-topic" in routing_diagnostics
        assert "retained in the source region" in routing_diagnostics
        assert {item["region_id"] for item in routed["semantic_memory_refs"]} == {
            "user",
            "rust-topic",
            "semantic-project",
        }

        regions = client.get("/api/v1/regions", headers=headers).json()
        assert {item["region_id"] for item in regions}.isdisjoint({"missing-topic"})
        project_items = _memories(client, headers, "semantic-project")
        user_items = _memories(client, headers, "user")
        topic_items = _memories(client, headers, "rust-topic")
        assert {item["summary"] for item in project_items} == {
            project_content,
            "The user prefers the missing topic.",
        }
        assert [item["summary"] for item in user_items] == [preference_content]
        assert [item["summary"] for item in topic_items] == [topic_content]

        user_id = str(user_items[0]["id"])
        user_detail = _detail(client, headers, "user", user_id)
        assert user_detail["region"]["kind"] == "user"
        assert user_detail["user_metadata"]["region_hint"] == "user"
        assert user_detail["user_metadata"]["target_region_id"] == "user"
        assert user_detail["provenance_summary"]["target_region"] == {
            "region_id": "user",
            "name": "用户记忆",
            "kind": "user",
        }
        assert user_detail["provenance_summary"]["source_regions"] == [
            {
                "region_id": "semantic-project",
                "name": "semantic-project",
                "kind": "project",
            }
        ]
        assert user_detail["provenance_summary"]["evidence_basis"]["snippets"] == [
            "I prefer dark mode"
        ]
        assert set(user_detail["provenance_summary"]["evidence_basis"]["event_ids"]) == {
            "routing-turn-user",
            "routing-turn-assistant",
        }
        assert len(user_detail["semantic_versions"]) == 1
        assert {
            item["event_id"] for item in user_detail["evidence"] if item.get("resolved")
        } == {"routing-turn-user", "routing-turn-assistant"}

        topic_detail = _detail(client, headers, "rust-topic", str(topic_items[0]["id"]))
        assert topic_detail["provenance_summary"]["target_region"]["kind"] == "topic"
        assert topic_detail["provenance_summary"]["source_regions"][0]["region_id"] == (
            "semantic-project"
        )

        project_memory_id = next(
            str(item["id"])
            for item in project_items
            if item["summary"] == project_content
        )
        project_detail = _detail(
            client,
            headers,
            "semantic-project",
            project_memory_id,
        )
        assert project_detail["provenance_summary"]["target_region"]["kind"] == "project"
        assert project_detail["provenance_summary"]["source_regions"][0]["region_id"] == (
            "semantic-project"
        )
        assert project_detail["provenance_summary"]["evidence_basis"]["snippets"] == [
            "The build system uses pnpm in this project"
        ]

        _capture_turn(
            client,
            headers,
            session_id="governance-session",
            turn_id="duplicate-turn",
            user_content="I prefer dark mode.",
            assistant_content="Noted again.",
            root=root,
            second=10,
        )
        executor.drain()
        duplicate = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="governance-session",
            turn_id="duplicate-turn",
        )
        assert duplicate["semantic_memory_ids"] == [user_id]
        user_items = _memories(client, headers, "user")
        assert len(user_items) == 1
        user_detail = _detail(client, headers, "user", user_id)
        assert len(user_detail["semantic_versions"]) == 2
        assert {
            item["event_id"] for item in user_detail["evidence"] if item.get("resolved")
        } >= {"routing-turn-user", "duplicate-turn-user"}

        _capture_turn(
            client,
            headers,
            session_id="governance-session",
            turn_id="conflict-turn",
            user_content="I prefer light mode.",
            assistant_content="Noted the alternative.",
            root=root,
            second=20,
        )
        executor.drain()
        conflict_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="governance-session",
            turn_id="conflict-turn",
        )
        assert conflict_round["semantic_status"] == "skipped"
        assert "conflicts" in " ".join(
            str(item) for item in conflict_round["semantic_diagnostics"]
        )
        assert [item["summary"] for item in _memories(client, headers, "user")] == [
            preference_content
        ]

        _capture_turn(
            client,
            headers,
            session_id="governance-session",
            turn_id="sensitive-turn",
            user_content=f"password='{secret}'",
            assistant_content="The sensitive value was not recorded.",
            root=root,
            second=30,
        )
        executor.drain()
        sensitive_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="governance-session",
            turn_id="sensitive-turn",
        )
        assert sensitive_round["semantic_status"] == "skipped"
        assert "sensitive-content" in " ".join(
            str(item) for item in sensitive_round["semantic_diagnostics"]
        )
        assert [item["summary"] for item in _memories(client, headers, "user")] == [
            preference_content
        ]
        assert _search(client, headers, "semantic-project", secret)["results"] == []
        assert _search(client, headers, "user", secret)["results"] == []

        rebuilt = client.post(
            "/api/v1/libraries/user/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        rebuilt_project = client.post(
            "/api/v1/libraries/sensitive-evidence-project/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt_project.status_code == 202, rebuilt_project.text
        graph_status = client.get(
            "/api/v1/libraries/user/graph-status",
            headers=headers,
        ).json()
        assert graph_status["total_documents"] == 1
        assert secret not in json.dumps(graph_status, ensure_ascii=False)
        kernel = client.app.state.memory_kernel
        graph_store = kernel._require_bundle().graph_store
        assert graph_store is not None
        graph_scopes = [
            scope
            for scope in graph_store.graph_scopes()
            if scope.space
            in {"user", "project:sensitive-evidence-project"}
        ]
        nodes = [
            node
            for scope in graph_scopes
            for node in graph_store.get(scope, graph_store.node_ids(scope))
        ]
        assert secret not in json.dumps(
            {node.id: node.properties for node in nodes},
            ensure_ascii=False,
        )
        assert str(user_items[0]["id"]) in {node.id for node in nodes}

        maintenance = state_dir / "mem2" / "maintenance.sqlite3"
        with sqlite3.connect(maintenance) as connection:
            audits = connection.execute(
                """SELECT action, operation_id, detail, before_content, after_content
                   FROM governance_events
                   WHERE actor = 'semantic-distillation'
                     AND action IN ('conflict', 'sensitive')
                   ORDER BY action"""
            ).fetchall()
        assert [row[0] for row in audits] == ["conflict", "sensitive"]
        assert len({(row[0], row[1]) for row in audits}) == 2
        assert secret not in "".join(
            str(value) for row in audits for value in row[2:]
        )
        conflict_detail = json.loads(str(audits[0][2]))
        assert set(conflict_detail["conflicting_values"]) == {"dark", "light"}
        operation_ids = {row[0]: row[1] for row in audits}

        with sqlite3.connect(maintenance) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET semantic_status = 'failed'
                   WHERE turn_id IN ('conflict-turn', 'sensitive-turn')"""
            )
        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "governance-session"},
        )
        assert consolidated.status_code == 202, consolidated.text
        executor.drain()

        with sqlite3.connect(maintenance) as connection:
            retried_audits = connection.execute(
                """SELECT action, operation_id
                   FROM governance_events
                   WHERE actor = 'semantic-distillation'
                     AND action IN ('conflict', 'sensitive')
                   ORDER BY action"""
            ).fetchall()
        assert [row[0] for row in retried_audits] == ["conflict", "sensitive"]
        assert {row[0]: row[1] for row in retried_audits} == operation_ids


def test_invalid_target_fallback_is_retained_when_semantic_claim_conflicts(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        )
        event_id = str(user["event_id"])
        if event_id == "fallback-conflict-base-user":
            return [
                _memory(
                    content="The user prefers dark mode.",
                    memory_type="preference",
                    evidence="I prefer dark mode",
                    event_id=event_id,
                    subject="user:preference:theme",
                    value="dark",
                    region_hint="source",
                )
            ]
        if event_id == "fallback-conflict-update-user":
            return [
                _memory(
                    content="The user prefers light mode.",
                    memory_type="preference",
                    evidence="I prefer light mode",
                    event_id=event_id,
                    subject="user:preference:theme",
                    value="light",
                    region_hint="missing-topic",
                )
            ]
        raise AssertionError(f"unexpected semantic event: {event_id}")

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "fallback-conflict-project")
        _bind(client, headers, root, "fallback-conflict-project")

        _capture_turn(
            client,
            headers,
            session_id="fallback-conflict-session",
            turn_id="fallback-conflict-base",
            user_content="I prefer dark mode.",
            assistant_content="Noted.",
            root=root,
            second=0,
        )
        executor.drain()
        base = _round(
            client,
            headers,
            region_id="fallback-conflict-project",
            session_id="fallback-conflict-session",
            turn_id="fallback-conflict-base",
        )
        assert base["semantic_status"] == "completed"

        _capture_turn(
            client,
            headers,
            session_id="fallback-conflict-session",
            turn_id="fallback-conflict-update",
            user_content="I prefer light mode.",
            assistant_content="Noted the alternative.",
            root=root,
            second=10,
        )
        executor.drain()
        conflict = _round(
            client,
            headers,
            region_id="fallback-conflict-project",
            session_id="fallback-conflict-session",
            turn_id="fallback-conflict-update",
        )

        assert conflict["semantic_status"] == "skipped"
        diagnostics = conflict["semantic_diagnostics"]
        assert isinstance(diagnostics, list) and len(diagnostics) == 1
        diagnostic = diagnostics[0]
        assert "conflicts" in str(diagnostic["reason"])
        assert "retained in the source region" in str(
            diagnostic["target_fallback_reason"]
        )
        assert [item["summary"] for item in _memories(
            client,
            headers,
            "fallback-conflict-project",
        )] == ["The user prefers dark mode."]


def test_promotion_targets_preserve_source_evidence_and_audit_once(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    preference = "I prefer dark mode in every editor."
    topic_fact = "Rust ownership prevents data races."
    project_fact = "This repository uses pnpm for builds."

    def classify(content: str, context: Mapping[str, object]) -> Mapping[str, object]:
        del context
        if content == preference:
            return {
                "category": "user",
                "confidence": 0.99,
                "explicit": True,
                "topic_id": None,
                "claim_subject": "user:preference:theme",
                "claim_value": "dark",
                "reason": "stable user preference",
            }
        if content == topic_fact:
            return {
                "category": "topic",
                "confidence": 0.99,
                "explicit": True,
                "topic_id": "rust-topic",
                "claim_subject": "rust ownership",
                "claim_value": "prevents data races",
                "reason": "generalizable domain knowledge",
            }
        return {
            "category": "project",
            "confidence": 0.99,
            "explicit": True,
            "topic_id": None,
            "claim_subject": "project package manager",
            "claim_value": "pnpm",
            "reason": "project-specific fact",
        }

    with TestClient(_promotion_app(state_dir, classify)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "promotion-project")
        _region(client, headers, "rust-topic", kind="topic")
        source_ids: dict[str, str] = {}
        for key, content, memory_type in (
            ("preference", preference, "preference"),
            ("topic", topic_fact, "domain_fact"),
            ("project", project_fact, "constraint"),
        ):
            response = client.post(
                "/api/v1/memories",
                headers=headers,
                json={
                    "region_id": "promotion-project",
                    "content": content,
                    "memory_type": memory_type,
                    "source_references": [f"source:{key}"],
                    "actor": "test",
                    "idempotency_key": f"promotion-{key}",
                },
            )
            assert response.status_code == 201, response.text
            source_ids[key] = str(response.json()["id"])

        started = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["promotion-project"]},
        )
        assert started.status_code == 202, started.text
        status = _wait_promotion(client, headers, str(started.json()["job_id"]))
        assert status["status"] == "completed", status
        assert status["summary"]["promoted"] == 2
        assert status["summary"]["skipped"] == 1

        user_items = _memories(client, headers, "user")
        topic_items = _memories(client, headers, "rust-topic")
        assert [item["summary"] for item in user_items] == [preference]
        assert [item["summary"] for item in topic_items] == [topic_fact]
        assert {
            str(item["summary"])
            for item in _memories(client, headers, "promotion-project")
        } == {preference, topic_fact, project_fact}

        user_detail = _detail(client, headers, "user", str(user_items[0]["id"]))
        summary = user_detail["provenance_summary"]
        assert summary["target_region"]["region_id"] == "user"
        assert summary["source_regions"] == [
            {
                "region_id": "promotion-project",
                "name": "promotion-project",
                "kind": "project",
            }
        ]
        assert summary["promotion"]["promoted"] is True
        assert summary["promotion"]["classification"] == "user"
        assert source_ids["preference"] in summary["evidence_basis"]["source_unit_ids"]
        assert source_ids["preference"] in user_detail["provenance"]
        assert user_detail["provenance_chain"][0]["id"] == source_ids["preference"]
        assert user_detail["provenance_chain"][0]["content"] == preference
        assert user_detail["versions"]

        topic_detail = _detail(client, headers, "rust-topic", str(topic_items[0]["id"]))
        assert topic_detail["provenance_summary"]["source_regions"][0]["region_id"] == (
            "promotion-project"
        )
        assert topic_detail["provenance_summary"]["promotion"]["classification"] == "topic"

        repeated = client.post(
            "/api/v1/promotions",
            headers=headers,
            json={"source_region_ids": ["promotion-project"]},
        )
        assert repeated.status_code == 202, repeated.text
        repeated_status = _wait_promotion(
            client,
            headers,
            str(repeated.json()["job_id"]),
        )
        assert repeated_status["summary"]["promoted"] == 0
        assert len(_memories(client, headers, "user")) == 1
        history = client.get("/api/v1/libraries/user/history", headers=headers).json()
        assert len([item for item in history if item["action"] == "promote"]) == 1


def test_sensitive_assistant_evidence_is_redacted_without_leaking_content(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()
    secret = "correct-horse-battery-staple"
    preference = "The user prefers dark mode."

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        )
        return [
            _memory(
                content=preference,
                memory_type="preference",
                evidence="I prefer dark mode",
                event_id=str(user["event_id"]),
                subject="user:preference:theme",
                value="dark",
                region_hint="user",
            )
        ]

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "sensitive-evidence-project")
        _bind(client, headers, root, "sensitive-evidence-project")
        _capture_turn(
            client,
            headers,
            session_id="sensitive-evidence-session",
            turn_id="sensitive-evidence-turn",
            user_content="I prefer dark mode.",
            assistant_content=f"api_key = '{secret}'",
            root=root,
            second=0,
        )
        executor.drain()
        completed = _round(
            client,
            headers,
            region_id="sensitive-evidence-project",
            session_id="sensitive-evidence-session",
            turn_id="sensitive-evidence-turn",
        )
        assert completed["semantic_status"] == "completed", completed
        user_items = _memories(client, headers, "user")
        assert [item["summary"] for item in user_items] == [preference]
        detail = _detail(client, headers, "user", str(user_items[0]["id"]))
        assert secret not in json.dumps(detail, ensure_ascii=False)
        assistant_evidence = next(
            item
            for item in detail["evidence"]
            if item["event_id"] == "sensitive-evidence-turn-assistant"
        )
        assert assistant_evidence["redacted"] is True
        assert assistant_evidence["redaction_reason"] == "sensitive"
        assert assistant_evidence["content"] == ""
        assert assistant_evidence["resolved"] is True
        assert assistant_evidence["region_id"] == "sensitive-evidence-project"
        assert assistant_evidence["session_id"] == "sensitive-evidence-session"
        assert assistant_evidence["turn_id"] == "sensitive-evidence-turn"
        user_evidence = next(
            item
            for item in detail["evidence"]
            if item["event_id"] == "sensitive-evidence-turn-user"
        )
        assert user_evidence["redacted"] is False
        assert user_evidence["content"] == "I prefer dark mode."
        assert _search(client, headers, "user", secret)["results"] == []
        assert _search(client, headers, "sensitive-evidence-project", secret)["results"] == []

        rebuilt = client.post(
            "/api/v1/libraries/user/graph-index/rebuild",
            headers=headers,
        )
        assert rebuilt.status_code == 202, rebuilt.text
        graph_status = client.get(
            "/api/v1/libraries/user/graph-status",
            headers=headers,
        ).json()
        assert graph_status["total_documents"] == 1
        assert secret not in json.dumps(graph_status, ensure_ascii=False)


@pytest.mark.parametrize(
    ("sensitive_field",),
    [("claim_subject",), ("claim_value",), ("region_hint",)],
)
def test_sensitive_semantic_model_fields_are_skipped_without_leaks(
    tmp_path: Path,
    sensitive_field: str,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()
    secret = "correct-horse-battery-staple"
    preference = "The user prefers dark mode."
    result = _memory(
        content=preference,
        memory_type="preference",
        evidence="I prefer dark mode",
        event_id="sensitive-field-turn-user",
        subject="user:preference:theme",
        value="dark",
        region_hint="user",
    )
    result[sensitive_field] = f"password='{secret}'"

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [result]

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "sensitive-field-project")
        _bind(client, headers, root, "sensitive-field-project")
        _capture_turn(
            client,
            headers,
            session_id="sensitive-field-session",
            turn_id="sensitive-field-turn",
            user_content="I prefer dark mode.",
            assistant_content="Noted.",
            root=root,
            second=0,
        )
        executor.drain()
        skipped = _round(
            client,
            headers,
            region_id="sensitive-field-project",
            session_id="sensitive-field-session",
            turn_id="sensitive-field-turn",
        )

        assert skipped["semantic_status"] == "skipped", skipped
        assert skipped["semantic_memory_ids"] == []
        assert secret not in json.dumps(skipped, ensure_ascii=False)
        diagnostics = json.dumps(skipped["semantic_diagnostics"], ensure_ascii=False)
        assert "sensitive-content" in diagnostics
        assert sensitive_field in diagnostics
        assert _memories(client, headers, "sensitive-field-project") == []
        assert _search(client, headers, "sensitive-field-project", secret)["results"] == []

        regions = client.get("/api/v1/regions", headers=headers)
        assert regions.status_code == 200, regions.text
        region_ids = {str(item["region_id"]) for item in regions.json()}
        if "user" in region_ids:
            assert _memories(client, headers, "user") == []
            assert _search(client, headers, "user", secret)["results"] == []

        for region_id in ("sensitive-field-project", "user"):
            if region_id not in region_ids:
                continue
            rebuilt = client.post(
                f"/api/v1/libraries/{region_id}/graph-index/rebuild",
                headers=headers,
            )
            assert rebuilt.status_code == 202, rebuilt.text
            graph_status = client.get(
                f"/api/v1/libraries/{region_id}/graph-status",
                headers=headers,
            )
            assert graph_status.status_code == 200, graph_status.text
            assert secret not in graph_status.text

        maintenance = state_dir / "mem2" / "maintenance.sqlite3"
        with sqlite3.connect(maintenance) as connection:
            audit_rows = connection.execute(
                """SELECT detail, before_content, after_content
                   FROM governance_events
                   WHERE actor = 'semantic-distillation'
                     AND action = 'sensitive'"""
            ).fetchall()
        assert len(audit_rows) == 1
        audit_text = json.dumps(audit_rows, ensure_ascii=False)
        assert secret not in audit_text
        audit_detail = json.loads(str(audit_rows[0][0]))
        assert audit_detail["sensitive_fields"] == [sensitive_field]
        if sensitive_field == "region_hint":
            assert audit_detail["region_hint"] == "[redacted]"


def test_cross_region_semantic_manual_supersede_preserves_provenance(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()
    calls = 0
    first_content = "The user prefers dark mode."
    corrected_content = "I prefer light mode."

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls > 1:
            return []
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        )
        return [
            _memory(
                content=first_content,
                memory_type="preference",
                evidence="I prefer dark mode",
                event_id=str(user["event_id"]),
                subject="user:preference:theme",
                value="dark",
                region_hint="user",
            )
        ]

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "cross-region-project")
        _bind(client, headers, root, "cross-region-project")
        _capture_turn(
            client,
            headers,
            session_id="cross-region-session",
            turn_id="cross-region-turn",
            user_content="I prefer dark mode.",
            assistant_content="Old conclusion.",
            root=root,
            second=0,
        )
        executor.drain()
        created = _round(
            client,
            headers,
            region_id="cross-region-project",
            session_id="cross-region-session",
            turn_id="cross-region-turn",
        )
        original_id = str(created["semantic_memory_ids"][0])
        original = _detail(client, headers, "user", original_id)
        assert original["provenance_summary"]["source_regions"][0]["region_id"] == (
            "cross-region-project"
        )
        assert len(original["semantic_versions"]) == 1

        updated = client.put(
            "/api/v1/libraries/user/document",
            headers=headers,
            json={
                "path": original_id,
                "content": corrected_content,
                "expected_source_version": original["source_version"],
                "operation_id": "cross-region-manual-supersede",
                "actor_type": "user",
                "source": "manual-cross-region-test",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        updated_id = str(updated.json()["id"])
        assert updated_id != original_id
        preserved = _detail(client, headers, "user", updated_id)
        metadata = preserved["user_metadata"]
        assert preserved["content"] == corrected_content
        assert metadata["semantic_manual_override"] is True
        assert "semantic_owned" not in metadata
        assert "claim_subject" not in metadata
        assert "claim_value" not in metadata
        assert metadata["source_region_id"] == "cross-region-project"
        assert "cross-region-project" in metadata["source_region_ids"]
        assert metadata["source_unit_ids"]
        assert metadata["source_event_ids"]
        assert metadata["provenance"] == metadata["source_unit_ids"]
        assert metadata["session_id"] == "cross-region-session"
        assert metadata["turn_id"] == "cross-region-turn"
        assert metadata["user_event_id"] == "cross-region-turn-user"
        assert metadata["evidence_snippet"] == "I prefer dark mode"
        assert len(preserved["semantic_versions"]) == 1
        assert preserved["provenance_chain"]
        assert preserved["provenance_summary"]["source_regions"][0]["region_id"] == (
            "cross-region-project"
        )

        final_assistant = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "cross-region-final-assistant",
                "session_id": "cross-region-session",
                "turn_id": "cross-region-turn",
                "event_kind": "assistant",
                "content": "Final conclusion.",
                "occurred_at": "2026-10-28T02:00:10Z",
                "cwd": str(root),
            },
        )
        assert final_assistant.status_code == 202, final_assistant.text
        executor.drain()
        final = _round(
            client,
            headers,
            region_id="cross-region-project",
            session_id="cross-region-session",
            turn_id="cross-region-turn",
        )
        assert calls == 2
        assert final["semantic_memory_ids"] == []
        assert all(item.get("status") != "reused" for item in final["semantic_diagnostics"])
        assert {
            item["summary"]
            for item in _memories(client, headers, "user")
            if item["lifecycle"] == "active"
        } == {corrected_content}
        final_detail = _detail(client, headers, "user", updated_id)
        assert final_detail["content"] == corrected_content
        assert len(final_detail["semantic_versions"]) == 1
        assert final_detail["provenance_summary"]["source_regions"][0]["region_id"] == (
            "cross-region-project"
        )


def test_project_phrase_inside_user_preference_stays_in_user_region(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()
    content = "I prefer this project to use pnpm."

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        )
        return [
            _memory(
                content=content,
                memory_type="preference",
                evidence="I prefer this project to use pnpm",
                event_id=str(user["event_id"]),
                subject="user:preference:package_manager",
                value="pnpm",
                region_hint="user",
            )
        ]

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "preference-project")
        _bind(client, headers, root, "preference-project")
        _capture_turn(
            client,
            headers,
            session_id="preference-project-session",
            turn_id="preference-project-turn",
            user_content="I prefer this project to use pnpm.",
            assistant_content="Noted.",
            root=root,
            second=0,
        )
        executor.drain()
        completed = _round(
            client,
            headers,
            region_id="preference-project",
            session_id="preference-project-session",
            turn_id="preference-project-turn",
        )
        assert completed["semantic_status"] == "completed"
        assert [item["summary"] for item in _memories(client, headers, "user")] == [content]
        assert _memories(client, headers, "preference-project") == []


def test_semantic_conflict_compatibilizes_prefer_and_preference_subjects(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers light mode.",
                memory_type="preference",
                evidence="I prefer light mode",
                event_id="compatible-conflict-turn-user",
                subject="user:preference:theme",
                value="light",
                region_hint="user",
            )
        ]

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "compatible-conflict-project")
        _region(client, headers, "user", kind="user", name="用户记忆")
        _bind(client, headers, root, "compatible-conflict-project")
        existing = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "region_id": "user",
                "content": "I prefer dark mode.",
                "memory_type": "preference",
                "source_references": ["test:ordinary-user-preference"],
                "actor": "test",
                "idempotency_key": "ordinary-user-preference",
            },
        )
        assert existing.status_code == 201, existing.text
        _capture_turn(
            client,
            headers,
            session_id="compatible-conflict-session",
            turn_id="compatible-conflict-turn",
            user_content="I prefer light mode.",
            assistant_content="Noted.",
            root=root,
            second=0,
        )
        executor.drain()
        conflict_round = _round(
            client,
            headers,
            region_id="compatible-conflict-project",
            session_id="compatible-conflict-session",
            turn_id="compatible-conflict-turn",
        )
        assert conflict_round["semantic_status"] == "skipped"
        assert "conflicts" in " ".join(
            str(item) for item in conflict_round["semantic_diagnostics"]
        )
        active = [
            item
            for item in _memories(client, headers, "user")
            if item["lifecycle"] == "active"
        ]
        assert [item["summary"] for item in active] == ["I prefer dark mode."]
        history = client.get("/api/v1/libraries/user/history", headers=headers).json()
        assert len([item for item in history if item["action"] == "conflict"]) == 1


def test_graph_association_filters_sensitive_units_and_keeps_safe_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(create_app(Settings(state_dir=state_dir))) as client:
        headers = _headers(state_dir)
        _region(client, headers, "graph-governance-project")
        safe = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "region_id": "graph-governance-project",
                "content": "The build uses pnpm.",
                "memory_type": "domain_fact",
                "source_references": ["test:safe"],
                "actor": "test",
                "idempotency_key": "graph-safe",
            },
        )
        assert safe.status_code == 201, safe.text
        secret = "password='probe-secret-123'"
        sensitive = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "region_id": "graph-governance-project",
                "content": f"The service credential is {secret}.",
                "memory_type": "domain_fact",
                "source_references": ["test:sensitive"],
                "actor": "test",
                "idempotency_key": "graph-sensitive",
            },
        )
        assert sensitive.status_code == 201, sensitive.text

        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        graph_store = bundle.graph_store
        assert graph_store is not None
        region = kernel._regions.require_region("graph-governance-project")
        scope = next(
            scope
            for scope in kernel._region_scopes(bundle, region)
            if kernel._units_in_scope(bundle, scope)
        )
        graph_store.insert(
            scope,
            nodes=[
                Node(
                    id=str(sensitive.json()["id"]),
                    label="long_term",
                    properties={"content": f"The service credential is {secret}."},
                )
            ],
        )

        associator = bundle.runtime.api._engine._evolver._associator
        original_associate = associator.associate
        observed: list[list[str]] = []

        def record_inputs(units: list[object]) -> object:
            observed.append([str(getattr(unit, "content", "")) for unit in units])
            return original_associate(units)

        monkeypatch.setattr(associator, "associate", record_inputs)
        assert kernel._associate_graph_scope(bundle, scope, region_id=region.region_id)

        assert observed
        assert all(secret not in content for inputs in observed for content in inputs)
        assert any("The build uses pnpm." in content for inputs in observed for content in inputs)
        node_ids = set(graph_store.node_ids(scope))
        assert str(safe.json()["id"]) in node_ids
        assert str(sensitive.json()["id"]) not in node_ids


def test_semantic_claim_value_equivalence_reuses_ordinary_and_manual_memories(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()
    calls = 0

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        event_id = str(
            next(
                item
                for item in context["messages"]
                if isinstance(item, dict) and item.get("role") == "user"
            )["event_id"]
        )
        value = "dark" if calls in {1, 3} else "dark mode"
        content = "The user prefers dark mode." if calls in {1, 3} else "The user prefers dark."
        return [
            _memory(
                content=content,
                memory_type="preference",
                evidence=f"I prefer {value}",
                event_id=event_id,
                subject="user:preference:theme",
                value=value,
                region_hint="user",
            )
        ]

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "equivalence-project")
        _bind(client, headers, root, "equivalence-project")
        for turn_id, second in (("ordinary-one", 0), ("ordinary-two", 10)):
            _capture_turn(
                client,
                headers,
                session_id="equivalence-session",
                turn_id=turn_id,
                user_content="I prefer dark mode.",
                assistant_content="Noted.",
                root=root,
                second=second,
            )
            executor.drain()
        first = _round(
            client,
            headers,
            region_id="equivalence-project",
            session_id="equivalence-session",
            turn_id="ordinary-one",
        )
        second = _round(
            client,
            headers,
            region_id="equivalence-project",
            session_id="equivalence-session",
            turn_id="ordinary-two",
        )
        memory_id = str(first["semantic_memory_ids"][0])
        assert second["semantic_memory_ids"] == [memory_id]
        assert second["semantic_status"] == "completed"
        detail = _detail(client, headers, "user", memory_id)
        assert len(detail["semantic_versions"]) == 2
        assert len(detail["user_metadata"]["source_unit_ids"]) == 4
        assert len(detail["user_metadata"]["evidence_snippets"]) == 2

        updated = client.put(
            "/api/v1/libraries/user/document",
            headers=headers,
            json={
                "path": memory_id,
                "content": "I prefer dark mode.",
                "expected_source_version": detail["source_version"],
                "operation_id": "equivalence-manual-override",
                "actor_type": "user",
                "source": "equivalence-test",
                "mode": "supersede",
            },
        )
        assert updated.status_code == 200, updated.text
        manual_id = str(updated.json()["id"])
        for turn_id, second in (("manual-one", 20),):
            _capture_turn(
                client,
                headers,
                session_id="equivalence-session",
                turn_id=turn_id,
                user_content="I prefer dark mode.",
                assistant_content="Noted.",
                root=root,
                second=second,
            )
            executor.drain()
        manual_round = _round(
            client,
            headers,
            region_id="equivalence-project",
            session_id="equivalence-session",
            turn_id="manual-one",
        )
        assert manual_round["semantic_memory_ids"] == [manual_id]
        assert manual_round["semantic_status"] == "completed"
        assert len(_detail(client, headers, "user", manual_id)["semantic_versions"]) == 3


def test_chinese_user_project_scope_stays_in_source_region(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    executor = ManualSemanticExecutor()
    content = "用户的项目使用 pnpm。"

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        user = next(
            item
            for item in context["messages"]
            if isinstance(item, dict) and item.get("role") == "user"
        )
        return [
            _memory(
                content=content,
                memory_type="constraint",
                evidence=content,
                event_id=str(user["event_id"]),
                subject="项目包管理器",
                value="pnpm",
                region_hint="user",
            )
        ]

    with TestClient(_semantic_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "chinese-project-scope")
        _bind(client, headers, root, "chinese-project-scope")
        _capture_turn(
            client,
            headers,
            session_id="chinese-project-session",
            turn_id="chinese-project-turn",
            user_content=content,
            assistant_content="已记录。",
            root=root,
            second=0,
        )
        executor.drain()
        completed = _round(
            client,
            headers,
            region_id="chinese-project-scope",
            session_id="chinese-project-session",
            turn_id="chinese-project-turn",
        )
        assert completed["semantic_status"] == "completed"
        assert [
            item["summary"] for item in _memories(client, headers, "chinese-project-scope")
        ] == [content]
        regions = client.get("/api/v1/regions", headers=headers).json()
        assert "user" not in {str(item["region_id"]) for item in regions}
