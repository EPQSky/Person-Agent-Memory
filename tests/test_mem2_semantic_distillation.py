from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.mem2_kernel import BoundedSemanticExecutor, Mem2MemoryKernel
from personal_agent_memory.memory_kernel import CaptureMemoryRequest

SemanticExtractor = Callable[[dict[str, object]], list[dict[str, object]]]


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

    def run_next(self) -> None:
        with self._lock:
            task = self._tasks.popleft()
        task()

    def drain(self) -> None:
        while self.pending_count:
            self.run_next()

    def close(self) -> None:
        self.drain()


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _app(
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


def _region(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    *,
    kind: str = "project",
) -> None:
    response = client.post(
        "/api/v1/regions",
        headers=headers,
        json={"kind": kind, "name": region_id, "region_id": region_id},
    )
    assert response.status_code == 201, response.text


def _bind(client: TestClient, headers: dict[str, str], root: Path, region_id: str) -> None:
    response = client.post(
        "/api/v1/project-bindings",
        headers=headers,
        json={"project_root": str(root), "region_id": region_id},
    )
    assert response.status_code == 201, response.text


def _capture(
    client: TestClient,
    headers: dict[str, str],
    *,
    event_id: str,
    session_id: str,
    turn_id: str,
    kind: str,
    content: str,
    cwd: Path,
    occurred_at: str,
) -> dict[str, object]:
    response = client.post(
        "/api/v1/capture/events",
        headers=headers,
        json={
            "event_id": event_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "event_kind": kind,
            "content": content,
            "occurred_at": occurred_at,
            "cwd": str(cwd),
        },
    )
    assert response.status_code == 202, response.text
    return dict(response.json())


def _capture_turn(
    client: TestClient,
    headers: dict[str, str],
    *,
    session_id: str,
    turn_id: str,
    user_content: str,
    assistant_content: str,
    cwd: Path,
    second: int = 0,
) -> None:
    _capture(
        client,
        headers,
        event_id=f"{turn_id}-user",
        session_id=session_id,
        turn_id=turn_id,
        kind="user",
        content=user_content,
        cwd=cwd,
        occurred_at=f"2026-10-28T01:00:{second:02d}Z",
    )
    _capture(
        client,
        headers,
        event_id=f"{turn_id}-assistant",
        session_id=session_id,
        turn_id=turn_id,
        kind="assistant",
        content=assistant_content,
        cwd=cwd,
        occurred_at=f"2026-10-28T01:00:{second + 1:02d}Z",
    )


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
        params={
            "region_id": region_id,
            "session_id": session_id,
            "turn_id": turn_id,
        },
    )
    assert response.status_code == 200, response.text
    rounds = response.json()
    assert len(rounds) == 1, rounds
    return dict(rounds[0])


def _formal_memories(
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


def _memory(
    *,
    content: str,
    memory_type: str,
    evidence: str,
    event_id: str,
    subject: str | None,
    value: str | None,
    confidence: float = 0.96,
    region_hint: str = "source",
    durability: str | None = "durable_statement",
) -> dict[str, object]:
    result: dict[str, object] = {
        "content": content,
        "memory_type": memory_type,
        "confidence": confidence,
        "region_hint": region_hint,
        "user_event_id": event_id,
        "evidence_snippet": evidence,
    }
    if subject is not None:
        result["claim_subject"] = subject
    if value is not None:
        result["claim_value"] = value
    if durability is not None:
        result["durability"] = durability
    return result


def _prepare_project(client: TestClient, headers: dict[str, str], root: Path) -> None:
    root.mkdir()
    _region(client, headers, "semantic-project")
    _bind(client, headers, root, "semantic-project")


def test_semantic_distillation_is_async_and_publishes_atomic_evidence_bound_memories(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    calls = 0

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        messages = context["messages"]
        assert isinstance(messages, list)
        user_event = next(
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "user"
        )
        assert user_event["content"] == (
            "I prefer TypeScript for browser automation and require weekly dependency audits."
        )
        return [
            _memory(
                content="The user prefers TypeScript for browser automation.",
                memory_type="preference",
                evidence="I prefer TypeScript for browser automation",
                event_id="semantic-turn-user",
                subject="user:preference:language",
                value="typescript",
            ),
            _memory(
                content="The user requires weekly dependency audits.",
                memory_type="constraint",
                evidence="require weekly dependency audits",
                event_id="semantic-turn-user",
                subject="dependency audit cadence",
                value="weekly",
            ),
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="semantic-session",
            turn_id="semantic-turn",
            user_content=(
                "I prefer TypeScript for browser automation and require weekly dependency audits."
            ),
            assistant_content="Confirmed. I will use TypeScript and schedule the audits.",
            cwd=root,
        )
        assert calls == 0
        assert executor.pending_count == 1
        executor.run_next()
        assert calls == 1
        assert executor.pending_count == 0

        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="semantic-session",
            turn_id="semantic-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        semantic_ids = round_payload["semantic_memory_ids"]
        assert isinstance(semantic_ids, list) and len(semantic_ids) == 2
        refs = round_payload["semantic_memory_refs"]
        assert isinstance(refs, list)
        assert {item["user_event_id"] for item in refs} == {"semantic-turn-user"}
        assert [item["evidence_snippet"] for item in refs] == [
            "I prefer TypeScript for browser automation",
            "require weekly dependency audits",
        ]
        assert calls == 1

        memories = _formal_memories(client, headers, "semantic-project")
        assert {item["id"] for item in memories} == set(semantic_ids)
        assert {item["memory_type"] for item in memories} == {"preference", "constraint"}
        detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{semantic_ids[0]}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        detail_payload = detail.json()
        assert detail_payload["user_metadata"]["evidence_snippet"] in {
            "I prefer TypeScript for browser automation",
            "require weekly dependency audits",
        }
        assert {
            item["event_id"] for item in detail_payload["evidence"] if item.get("resolved")
        } == {"semantic-turn-user", "semantic-turn-assistant"}

        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"region_id": "semantic-project", "query": "TypeScript", "limit": 10},
        )
        assert search.status_code == 200, search.text
        assert any("TypeScript" in item["content"] for item in search.json()["results"])
        mcp_search = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "memory_search",
                    "arguments": {"region_id": "semantic-project", "query": "TypeScript"},
                },
            },
        )
        assert mcp_search.status_code == 200, mcp_search.text
        mcp_payload = mcp_search.json()
        assert mcp_payload["result"]["isError"] is False, mcp_payload
        assert any(
            "TypeScript" in item["content"]
            for item in mcp_payload["result"]["structuredContent"]["results"]
        )
        codex_route = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "cwd": str(root),
                "session_id": "semantic-session",
                "query": "TypeScript",
                "limit": 10,
            },
        )
        assert codex_route.status_code == 200, codex_route.text
        assert any(
            "TypeScript" in item["content"] for item in codex_route.json()["results"]
        )


@pytest.mark.parametrize(
    ("extractor", "expected_status"),
    [
        (lambda context: [], "skipped"),
        (lambda context: (_ for _ in ()).throw(RuntimeError("model down")), "failed"),
    ],
)
def test_zero_results_and_model_failures_do_not_publish(
    tmp_path: Path,
    extractor: SemanticExtractor,
    expected_status: str,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="zero-session",
            turn_id="zero-turn",
            user_content="No durable statement is present here.",
            assistant_content="Acknowledged.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="zero-session",
            turn_id="zero-turn",
        )
        assert round_payload["semantic_status"] == expected_status
        assert round_payload["semantic_memory_ids"] == []
        assert _formal_memories(client, headers, "semantic-project") == []
        if expected_status == "failed":
            assert "RuntimeError: model down" in str(round_payload["semantic_error"])


def test_incomplete_turns_and_assistant_only_content_never_call_the_model(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    calls = 0

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return []

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture(
            client,
            headers,
            event_id="user-only",
            session_id="user-only-session",
            turn_id="user-only-turn",
            kind="user",
            content="I prefer durable answers.",
            cwd=root,
            occurred_at="2026-10-28T02:10:00Z",
        )
        _capture(
            client,
            headers,
            event_id="assistant-only",
            session_id="assistant-only-session",
            turn_id="assistant-only-turn",
            kind="assistant",
            content="I infer that the user prefers durable answers.",
            cwd=root,
            occurred_at="2026-10-28T02:10:01Z",
        )
        assert executor.pending_count == 0
        assert calls == 0
        assert _formal_memories(client, headers, "semantic-project") == []


def test_validation_rejects_invalid_semantic_results(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="How should the project be analyzed?",
                memory_type="domain_fact",
                evidence="help me",
                event_id="validation-turn-user",
                subject="project",
                value="analysis question",
            ),
            _memory(
                content="It prefers the setting mentioned above.",
                memory_type="preference",
                evidence="durable",
                event_id="validation-turn-user",
                subject="user:preference:setting",
                value="durable",
            ),
            _memory(
                content="The user prefers the durable setting.",
                memory_type="preference",
                evidence="durable",
                event_id="validation-turn-user",
                subject="user:preference:setting",
                value="durable",
                confidence=0.89,
            ),
            _memory(
                content="The user prefers short answers.",
                memory_type="preference",
                evidence="short answers",
                event_id="validation-turn-assistant",
                subject="user:preference:length",
                value="short",
            ),
            _memory(
                content="The user API key is api_key = actualsecretvalue1234567890.",
                memory_type="domain_fact",
                evidence="durable",
                event_id="validation-turn-user",
                subject="api key",
                value="actualsecretvalue1234567890",
            ),
            _memory(
                content="The project plan is current.",
                memory_type="decision",
                evidence="下一步稍后执行",
                event_id="validation-turn-user",
                subject="project plan",
                value="current",
            ),
            _memory(
                content="The command produced useful output.",
                memory_type="reusable_experience",
                evidence="stdout raw",
                event_id="validation-turn-user",
                subject="command output",
                value="useful",
            ),
            _memory(
                content="The user prefers short answers. The user requires weekly audits.",
                memory_type="preference",
                evidence="durable short answers",
                event_id="validation-turn-user",
                subject="user:preference:length",
                value="short and weekly",
            ),
            _memory(
                content="The project requires analysis.",
                memory_type="domain_fact",
                evidence="帮我分析",
                event_id="validation-turn-user",
                subject="project",
                value="analysis",
            ),
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="validation-session",
            turn_id="validation-turn",
            user_content=(
                "project analysis request; 帮我分析; durable short answers are useful; "
                "下一步稍后执行; stdout raw"
            ),
            assistant_content="The user prefers short answers.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="validation-session",
            turn_id="validation-turn",
        )
        assert round_payload["semantic_status"] == "skipped"
        assert round_payload["semantic_memory_ids"] == []
        diagnostics = round_payload["semantic_diagnostics"]
        assert isinstance(diagnostics, list) and len(diagnostics) == 9
        reasons = " ".join(str(item["reason"]) for item in diagnostics)
        assert "question" in reasons
        assert "request rather than a durable claim" in reasons
        assert "dialogue references" in reasons
        assert "confidence" in reasons
        assert "user_event_id" in reasons
        assert "sensitive-content" in reasons
        assert reasons.count("transient or non-semantic output") == 2
        assert "more than one atomic claim" in reasons
        assert _formal_memories(client, headers, "semantic-project") == []


def test_compound_and_unbound_claims_are_skipped(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers dark mode and requires weekly audits.",
                memory_type="preference",
                evidence="I prefer dark mode",
                event_id="compound-turn-user",
                subject=None,
                value=None,
            ),
            _memory(
                content="It requires weekly audits.",
                memory_type="constraint",
                evidence="require weekly audits",
                event_id="compound-turn-user",
                subject=None,
                value=None,
            ),
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="compound-session",
            turn_id="compound-turn",
            user_content="I prefer dark mode and require weekly audits.",
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="compound-session",
            turn_id="compound-turn",
        )
        assert round_payload["semantic_status"] == "skipped"
        reasons = " ".join(
            str(item.get("reason", "")) for item in round_payload["semantic_diagnostics"]
        )
        assert "more than one atomic claim" in reasons
        assert "unbound pronoun" in reasons
        assert _formal_memories(client, headers, "semantic-project") == []


def test_temporal_and_structured_plan_results_are_skipped(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="We will deploy the staging build tomorrow.",
                memory_type="decision",
                evidence="deploy the staging build tomorrow",
                event_id="plan-turn-user",
                subject=None,
                value=None,
            ),
            _memory(
                content="The staging rollout is scheduled for next sprint.",
                memory_type="decision",
                evidence="scheduled for next sprint",
                event_id="plan-turn-user",
                subject=None,
                value=None,
            ),
            _memory(
                content="The staging build becomes available after the gate clears.",
                memory_type="decision",
                evidence="becomes available after the gate clears",
                event_id="plan-turn-user",
                subject=None,
                value=None,
                durability="plan",
            ),
        ]

    user_content = (
        "We will deploy the staging build tomorrow. "
        "The staging rollout is scheduled for next sprint. "
        "The staging build becomes available after the gate clears."
    )
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="plan-session",
            turn_id="plan-turn",
            user_content=user_content,
            assistant_content="Plan recorded.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="plan-session",
            turn_id="plan-turn",
        )
        assert round_payload["semantic_status"] == "skipped"
        diagnostics = round_payload["semantic_diagnostics"]
        assert isinstance(diagnostics, list) and len(diagnostics) == 3
        reasons = " ".join(str(item["reason"]) for item in diagnostics)
        assert reasons.count("transient or non-semantic output") == 2
        assert "not marked durable" in reasons
        assert _formal_memories(client, headers, "semantic-project") == []


def test_conflict_skips_and_invalid_region_falls_back_to_source(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="Theme is light.",
                memory_type="preference",
                evidence="Theme should be light",
                event_id="conflict-turn-user",
                subject="theme",
                value="light",
                region_hint="missing-topic",
            ),
            _memory(
                content="The user prefers the missing topic.",
                memory_type="preference",
                evidence="prefer the missing topic",
                event_id="conflict-turn-user",
                subject="user:preference:topic",
                value="missing",
                region_hint="missing-topic",
            ),
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        existing = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "region_id": "semantic-project",
                "content": "Theme is dark.",
                "memory_type": "preference",
                "source_references": ["test:existing"],
                "actor": "test",
                "idempotency_key": "existing-theme",
            },
        )
        assert existing.status_code == 201, existing.text
        _capture_turn(
            client,
            headers,
            session_id="conflict-session",
            turn_id="conflict-turn",
            user_content="Theme should be light, and I prefer the missing topic.",
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="conflict-session",
            turn_id="conflict-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        reasons = " ".join(
            str(item) for item in round_payload["semantic_diagnostics"]
        )
        assert "conflicts" in reasons
        assert "missing-topic" in reasons
        assert "retained in the source region" in reasons
        conflict_diagnostics = [
            item
            for item in round_payload["semantic_diagnostics"]
            if item.get("status") == "skipped" and "conflicts" in str(item)
        ]
        assert len(conflict_diagnostics) == 1
        assert "retained in the source region" in str(
            conflict_diagnostics[0]["target_fallback_reason"]
        )
        memories = _formal_memories(client, headers, "semantic-project")
        assert {item["id"] for item in memories} == {
            existing.json()["id"],
            round_payload["semantic_memory_ids"][0],
        }
        assert {item["summary"] for item in memories} == {
            "Theme is dark.",
            "The user prefers the missing topic.",
        }


def test_region_hints_publish_to_user_and_existing_topic_regions(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers concise answers.",
                memory_type="preference",
                evidence="I prefer concise answers",
                event_id="routing-turn-user",
                subject="user:preference:length",
                value="concise",
                region_hint="user",
            ),
            _memory(
                content="Rust ownership prevents data races without a garbage collector.",
                memory_type="domain_fact",
                evidence="Rust ownership prevents data races",
                event_id="routing-turn-user",
                subject="rust ownership",
                value="prevents data races",
                region_hint="rust-topic",
            ),
            _memory(
                content="This project uses pnpm for builds.",
                memory_type="constraint",
                evidence="This project uses pnpm for builds",
                event_id="routing-turn-user",
                subject="project package manager",
                value="pnpm",
                region_hint="user",
            ),
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _region(client, headers, "rust-topic", kind="topic")
        _capture_turn(
            client,
            headers,
            session_id="routing-session",
            turn_id="routing-turn",
            user_content=(
                "I prefer concise answers. Rust ownership prevents data races in my project. "
                "This project uses pnpm for builds."
            ),
            assistant_content="Noted for the user, Rust topic, and project regions.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="routing-session",
            turn_id="routing-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        refs = round_payload["semantic_memory_refs"]
        assert {item["region_id"] for item in refs} == {
            "user",
            "rust-topic",
            "semantic-project",
        }
        assert "project-scoped content was retained" in " ".join(
            str(item) for item in round_payload["semantic_diagnostics"]
        )
        assert len(_formal_memories(client, headers, "user")) == 1
        assert len(_formal_memories(client, headers, "rust-topic")) == 1
        assert [item["summary"] for item in _formal_memories(
            client,
            headers,
            "semantic-project",
        )] == ["This project uses pnpm for builds."]


def test_memory_type_boundaries_override_cross_region_hints(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers dark mode.",
                memory_type="preference",
                evidence="I prefer dark mode",
                event_id="typed-routing-turn-user",
                subject="user:preference:theme",
                value="dark",
                region_hint="typed-routing-topic",
            ),
            _memory(
                content="Rust ownership prevents data races.",
                memory_type="domain_fact",
                evidence="Rust ownership prevents data races",
                event_id="typed-routing-turn-user",
                subject="rust ownership",
                value="prevents data races",
                region_hint="user",
            ),
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _region(client, headers, "typed-routing-topic", kind="topic")
        _capture_turn(
            client,
            headers,
            session_id="typed-routing-session",
            turn_id="typed-routing-turn",
            user_content="I prefer dark mode. Rust ownership prevents data races.",
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        completed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="typed-routing-session",
            turn_id="typed-routing-turn",
        )

        assert completed["semantic_status"] == "completed", completed
        refs_by_region = {
            item["region_id"]: item["memory_id"]
            for item in completed["semantic_memory_refs"]
        }
        assert set(refs_by_region) == {"user", "semantic-project"}
        assert set(refs_by_region.values()) == set(completed["semantic_memory_ids"])
        diagnostics = completed["semantic_diagnostics"]
        assert diagnostics[0]["target_fallback_reason"]
        assert "topic" in str(diagnostics[0]["target_fallback_reason"])
        assert diagnostics[1]["target_fallback_reason"]
        assert "user region" in str(diagnostics[1]["target_fallback_reason"])
        assert [item["summary"] for item in _formal_memories(client, headers, "user")] == [
            "The user prefers dark mode."
        ]
        assert [item["summary"] for item in _formal_memories(
            client,
            headers,
            "semantic-project",
        )] == ["Rust ownership prevents data races."]
        assert _formal_memories(client, headers, "typed-routing-topic") == []


def test_sentence_initial_project_scope_terms_stay_in_source_region(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    candidates = [
        ("Project uses pnpm for builds.", "project"),
        ("Repository uses pnpm for builds.", "repository"),
        ("Workspace uses pnpm for builds.", "workspace"),
        ("Codebase uses pnpm for builds.", "codebase"),
    ]

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content=content,
                memory_type="constraint",
                evidence=content.removesuffix("."),
                event_id="bare-project-scope-turn-user",
                subject=f"project:{term}:package-manager",
                value="pnpm",
                region_hint="user",
            )
            for content, term in candidates
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _region(client, headers, "builds-topic", kind="topic")
        _capture_turn(
            client,
            headers,
            session_id="bare-project-scope-session",
            turn_id="bare-project-scope-turn",
            user_content=" ".join(content for content, _ in candidates),
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="bare-project-scope-session",
            turn_id="bare-project-scope-turn",
        )

        assert round_payload["semantic_status"] == "completed"
        assert len(round_payload["semantic_memory_ids"]) == len(candidates)
        assert [item["region_id"] for item in round_payload["semantic_memory_refs"]] == [
            "semantic-project",
        ] * len(candidates)
        assert {
            item["summary"]
            for item in _formal_memories(client, headers, "semantic-project")
        } == {content for content, _ in candidates}
        regions = client.get("/api/v1/regions", headers=headers).json()
        region_ids = {str(item["region_id"]) for item in regions}
        assert "user" not in region_ids
        assert _formal_memories(client, headers, "builds-topic") == []


def test_possessive_project_scope_terms_stay_in_source_region(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    candidates = [
        ("My project uses pnpm for builds.", "my-project"),
        ("The user's repository uses npm for builds.", "user-repository"),
        ("我的项目使用 uv 构建。", "我的项目"),
    ]

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content=content,
                memory_type="constraint",
                evidence=content.removesuffix("。"),
                event_id="possessive-project-scope-turn-user",
                subject=f"project:{term}:package-manager",
                value="pnpm" if "pnpm" in content else ("npm" if "npm" in content else "uv"),
                region_hint="user",
            )
            for content, term in candidates
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="possessive-project-scope-session",
            turn_id="possessive-project-scope-turn",
            user_content=" ".join(content for content, _ in candidates),
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="possessive-project-scope-session",
            turn_id="possessive-project-scope-turn",
        )

        assert round_payload["semantic_status"] == "completed"
        assert {
            item["region_id"] for item in round_payload["semantic_memory_refs"]
        } == {"semantic-project"}
        assert {
            item["summary"]
            for item in _formal_memories(client, headers, "semantic-project")
        } == {content for content, _ in candidates}
        regions = client.get("/api/v1/regions", headers=headers).json()
        assert "user" not in {str(item["region_id"]) for item in regions}


@pytest.mark.parametrize(
    "content",
    ["I prefer this project to use pnpm.", "I prefer my project to use pnpm."],
)
def test_project_phrase_inside_user_preference_stays_in_user_region(
    tmp_path: Path,
    content: str,
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
        return [
            _memory(
                content=content,
                memory_type="preference",
                evidence=content.removesuffix("."),
                event_id=str(user["event_id"]),
                subject="user:preference:package_manager",
                value="pnpm",
                region_hint="user",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "preference-project")
        _bind(client, headers, root, "preference-project")
        _capture_turn(
            client,
            headers,
            session_id="preference-project-session",
            turn_id="preference-project-turn",
            user_content=content,
            assistant_content="Noted.",
            cwd=root,
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
        assert [item["summary"] for item in _formal_memories(client, headers, "user")] == [
            content
        ]
        assert _formal_memories(client, headers, "preference-project") == []


def test_missing_region_hint_falls_back_to_source(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    content = "The user prefers concise answers."

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content=content,
                memory_type="preference",
                evidence="I prefer concise answers",
                event_id="missing-hint-turn-user",
                subject="user:preference:length",
                value="concise",
                region_hint="",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="missing-hint-session",
            turn_id="missing-hint-turn",
            user_content="I prefer concise answers.",
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="missing-hint-session",
            turn_id="missing-hint-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        assert "region_hint was missing" in " ".join(
            str(item) for item in round_payload["semantic_diagnostics"]
        )
        assert [item["summary"] for item in _formal_memories(
            client,
            headers,
            "semantic-project",
        )] == [content]
        regions = client.get("/api/v1/regions", headers=headers).json()
        assert "user" not in {str(item["region_id"]) for item in regions}


def test_missing_claim_fields_do_not_collapse_distinct_results_and_same_body_is_reused(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "user"
        )
        if user["event_id"] == "claims-turn-user":
            return [
                _memory(
                    content="TypeScript is preferred for browser automation.",
                    memory_type="domain_fact",
                    evidence="TypeScript for browser automation",
                    event_id="claims-turn-user",
                    subject=None,
                    value=None,
                ),
                _memory(
                    content="Weekly audits are required for dependencies.",
                    memory_type="domain_fact",
                    evidence="weekly audits for dependencies",
                    event_id="claims-turn-user",
                    subject=None,
                    value=None,
                ),
            ]
        evidence = (
            "The deployment policy requires immutable images"
            if "The deployment policy" in str(user["content"])
            else "Deployment policy: immutable images are required"
        )
        return [
            _memory(
                content="The deployment policy requires immutable images.",
                memory_type="constraint",
                evidence=evidence,
                event_id=str(user["event_id"]),
                subject=None,
                value=None,
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="claims-session",
            turn_id="claims-turn",
            user_content=(
                "TypeScript for browser automation and weekly audits for dependencies."
            ),
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        first_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="claims-session",
            turn_id="claims-turn",
        )
        assert first_round["semantic_status"] == "completed"
        assert len(first_round["semantic_memory_ids"]) == 2

        same_body = "The deployment policy requires immutable images."
        for index, second in enumerate((10, 20), start=1):
            turn_id = f"same-body-{index}"
            _capture_turn(
                client,
                headers,
                session_id="claims-session",
                turn_id=turn_id,
                user_content=(
                    "The deployment policy requires immutable images."
                    if index == 1
                    else "Deployment policy: immutable images are required."
                ),
                assistant_content="Confirmed.",
                cwd=root,
                second=second,
            )
            executor.drain()
        second_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="claims-session",
            turn_id="same-body-1",
        )
        third_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="claims-session",
            turn_id="same-body-2",
        )
        assert second_round["semantic_memory_ids"] == third_round["semantic_memory_ids"]
        memories = _formal_memories(client, headers, "semantic-project")
        assert len(memories) == 3
        reused_id = second_round["semantic_memory_ids"][0]
        reused_summary = next(
            item for item in memories if item["id"] == reused_id
        )
        assert reused_summary["summary"] == same_body
        assert reused_summary["source_count"] == 7
        detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{reused_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        detail_payload = detail.json()
        metadata = detail_payload["user_metadata"]
        assert set(metadata["source_event_ids"]) == {
            "same-body-1-user",
            "same-body-1-assistant",
            "same-body-2-user",
            "same-body-2-assistant",
        }
        assert len(metadata["source_unit_ids"]) == 4
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            version_rows = connection.execute(
                """SELECT source_fingerprint FROM semantic_memory_versions
                   WHERE memory_id = ? ORDER BY session_id, turn_id""",
                (reused_id,),
            ).fetchall()
        assert len(version_rows) == 2
        assert {
            item["event_id"]
            for item in detail_payload["evidence"]
            if item.get("resolved")
        } == {
            "same-body-1-user",
            "same-body-1-assistant",
            "same-body-2-user",
            "same-body-2-assistant",
        }


def test_stable_declaration_followed_by_question_is_still_distilled(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers uv for this project.",
                memory_type="preference",
                evidence="I prefer uv for this project.",
                event_id="declaration-question-turn-user",
                subject=None,
                value=None,
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="declaration-question-session",
            turn_id="declaration-question-turn",
            user_content=(
                "I prefer uv for this project. Can you update the documentation?"
            ),
            assistant_content="I will update the documentation.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="declaration-question-session",
            turn_id="declaration-question-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        memories = _formal_memories(client, headers, "semantic-project")
        assert [item["summary"] for item in memories] == [
            "The user prefers uv for this project."
        ]
        detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{memories[0]['id']}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["user_metadata"]["evidence_snippet"] == (
            "I prefer uv for this project."
        )


def test_mid_sentence_pronouns_and_missing_durability_are_skipped(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user says it requires weekly dependency audits.",
                memory_type="constraint",
                evidence="Dependency audits",
                event_id="pronoun-plan-turn-user",
                subject=None,
                value=None,
            ),
            _memory(
                content="The team is migrating the staging cluster on Friday.",
                memory_type="decision",
                evidence="migrating the staging cluster on Friday",
                event_id="pronoun-plan-turn-user",
                subject=None,
                value=None,
                durability=None,
            ),
            _memory(
                content="The team is shifting workloads across the failover pool.",
                memory_type="decision",
                evidence="shifting workloads across the failover pool",
                event_id="pronoun-plan-turn-user",
                subject=None,
                value=None,
                durability=None,
            ),
        ]

    user_content = (
        "Dependency audits matter. The team is migrating the staging cluster on Friday. "
        "The team is shifting workloads across the failover pool."
    )
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="pronoun-plan-session",
            turn_id="pronoun-plan-turn",
            user_content=user_content,
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="pronoun-plan-session",
            turn_id="pronoun-plan-turn",
        )
        assert round_payload["semantic_status"] == "skipped"
        diagnostics = round_payload["semantic_diagnostics"]
        assert isinstance(diagnostics, list)
        reasons = " ".join(str(item["reason"]) for item in diagnostics)
        assert "unbound pronoun" in reasons
        assert reasons.count("explicit durable classification") == 2
        assert _formal_memories(client, headers, "semantic-project") == []


def test_missing_claims_are_derived_for_conflict_checks_and_uncheckable_claims_skip(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "user"
        )
        is_light = "light" in str(user["content"])
        return [
            _memory(
                content=f"The user prefers {'light' if is_light else 'dark'} mode.",
                memory_type="preference",
                evidence=f"I prefer {'light' if is_light else 'dark'} mode",
                event_id=str(user["event_id"]),
                subject=None,
                value=None,
            ),
            _memory(
                content="Opaque.",
                memory_type="domain_fact",
                evidence="Opaque",
                event_id=str(user["event_id"]),
                subject=None,
                value=None,
            ),
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="missing-claim-session",
            turn_id="dark-mode-turn",
            user_content="I prefer dark mode. Opaque.",
            assistant_content="Noted.",
            cwd=root,
            second=0,
        )
        executor.drain()
        dark_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="missing-claim-session",
            turn_id="dark-mode-turn",
        )
        assert dark_round["semantic_status"] == "completed"
        assert len(dark_round["semantic_memory_ids"]) == 1

        _capture_turn(
            client,
            headers,
            session_id="missing-claim-session",
            turn_id="light-mode-turn",
            user_content="I prefer light mode. Opaque.",
            assistant_content="Noted.",
            cwd=root,
            second=10,
        )
        executor.drain()
        light_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="missing-claim-session",
            turn_id="light-mode-turn",
        )
        assert light_round["semantic_status"] == "skipped"
        reasons = " ".join(
            str(item["reason"]) for item in light_round["semantic_diagnostics"]
        )
        assert "conflicts" in reasons
        assert "claim identity" in reasons
        memories = _formal_memories(client, headers, "semantic-project")
        assert [item["summary"] for item in memories] == ["The user prefers dark mode."]


def test_empty_user_metadata_keeps_the_legacy_write_hash(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    request_payload = {
        "region_id": "semantic-project",
        "content": "The user prefers legacy-compatible idempotency.",
        "memory_type": "preference",
        "source_references": ["test:legacy-hash"],
        "actor": "test",
        "idempotency_key": "legacy-empty-metadata",
    }
    legacy_hash = hashlib.sha256(
        json.dumps(
            {
                "library_id": "",
                "region_id": "semantic-project",
                "content": request_payload["content"],
                "memory_type": "preference",
                "source_references": ["test:legacy-hash"],
                "actor": "test",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with TestClient(_app(state_dir, lambda context: [], executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        created = client.post("/api/v1/memories", headers=headers, json=request_payload)
        assert created.status_code == 201, created.text
        with sqlite3.connect(state_dir / "mem2" / "idempotency.sqlite3") as connection:
            row = connection.execute(
                """SELECT request_hash, memory_id FROM memory_write_idempotency
                   WHERE idempotency_key = ?""",
                ("legacy-empty-metadata",),
            ).fetchone()
        assert row is not None
        assert row[0] == legacy_hash
        replay = client.post("/api/v1/memories", headers=headers, json=request_payload)
        assert replay.status_code == 201, replay.text
        assert replay.json()["id"] == created.json()["id"] == row[1]


def test_partial_multi_result_publication_is_cleaned_before_zero_result_retry(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    calls = 0
    semantic_writes = 0

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls
        del context
        calls += 1
        if calls > 1:
            return []
        return [
            _memory(
                content="The user prefers alpha mode.",
                memory_type="preference",
                evidence="I prefer alpha mode",
                event_id="atomic-partial-turn-user",
                subject="user:preference:alpha",
                value="alpha",
            ),
            _memory(
                content="The user prefers beta mode.",
                memory_type="preference",
                evidence="I prefer beta mode",
                event_id="atomic-partial-turn-user",
                subject="user:preference:beta",
                value="beta",
            ),
        ]

    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    original_write = kernel._write

    def fail_second_semantic_write(request: object) -> dict[str, object]:
        nonlocal semantic_writes
        if getattr(request, "actor", "") == "semantic-distillation":
            semantic_writes += 1
            if semantic_writes == 2:
                raise RuntimeError("simulated second semantic write failure")
        return original_write(request)  # type: ignore[arg-type]

    kernel._write = fail_second_semantic_write  # type: ignore[method-assign]
    app = create_app(Settings(state_dir=state_dir), memory_kernel=kernel)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="atomic-partial-session",
            turn_id="atomic-partial-turn",
            user_content="I prefer alpha mode. I prefer beta mode.",
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        failed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="atomic-partial-session",
            turn_id="atomic-partial-turn",
        )
        assert failed["semantic_status"] == "failed"
        assert "simulated second semantic write failure" in str(failed["semantic_error"])
        assert _formal_memories(client, headers, "semantic-project") == []

        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "atomic-partial-session"},
        )
        assert consolidated.status_code == 202, consolidated.text
        executor.drain()
        retried = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="atomic-partial-session",
            turn_id="atomic-partial-turn",
        )
        assert retried["semantic_status"] == "skipped"
        assert retried["semantic_memory_ids"] == []
        assert retried["semantic_memory_refs"] == []
        assert any(
            "model returned no semantic memories" in str(item.get("reason", ""))
            for item in retried["semantic_diagnostics"]
        )
        assert _formal_memories(client, headers, "semantic-project") == []


def test_restart_cleans_partial_publication_before_zero_result_retry(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    first_executor = ManualSemanticExecutor()
    semantic_writes = 0

    def first_extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers restart alpha mode.",
                memory_type="preference",
                evidence="I prefer restart alpha mode",
                event_id="restart-partial-turn-user",
                subject="user:preference:restart-alpha",
                value="restart alpha",
            ),
            _memory(
                content="The user prefers restart beta mode.",
                memory_type="preference",
                evidence="I prefer restart beta mode",
                event_id="restart-partial-turn-user",
                subject="user:preference:restart-beta",
                value="restart beta",
            ),
        ]

    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=first_extractor,
        semantic_executor=first_executor,
    )
    original_write = kernel._write

    def fail_second_semantic_write(request: object) -> dict[str, object]:
        nonlocal semantic_writes
        if getattr(request, "actor", "") == "semantic-distillation":
            semantic_writes += 1
            if semantic_writes == 2:
                raise RuntimeError("simulated interrupted semantic publication")
        return original_write(request)  # type: ignore[arg-type]

    kernel._write = fail_second_semantic_write  # type: ignore[method-assign]
    kernel._cleanup_semantic_version = lambda **kwargs: None  # type: ignore[method-assign]
    app = create_app(Settings(state_dir=state_dir), memory_kernel=kernel)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="restart-partial-session",
            turn_id="restart-partial-turn",
            user_content="I prefer restart alpha mode. I prefer restart beta mode.",
            assistant_content="Noted.",
            cwd=root,
        )
        first_executor.drain()
        failed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="restart-partial-session",
            turn_id="restart-partial-turn",
        )
        assert failed["semantic_status"] == "failed"
        assert [item["summary"] for item in _formal_memories(
            client, headers, "semantic-project"
        )] == ["The user prefers restart alpha mode."]

    recovering_executor = ManualSemanticExecutor()

    def zero_extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return []

    with TestClient(_app(state_dir, zero_extractor, recovering_executor)) as client:
        headers = _headers(state_dir)
        assert recovering_executor.pending_count == 1
        recovering_executor.drain()
        recovered = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="restart-partial-session",
            turn_id="restart-partial-turn",
        )
        assert recovered["semantic_status"] == "skipped"
        assert recovered["semantic_memory_ids"] == []
        assert _formal_memories(client, headers, "semantic-project") == []


def test_compound_preference_lists_are_skipped_without_rejecting_self_contained_facts(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers dark mode and concise answers.",
                memory_type="preference",
                evidence="I prefer dark mode and concise answers",
                event_id="boundary-turn-user",
                subject="user:preference:profile",
                value="dark mode and concise answers",
            ),
            _memory(
                content="用户偏好深色模式和简洁回答。",
                memory_type="preference",
                evidence="用户偏好深色模式和简洁回答",
                event_id="boundary-turn-user",
                subject="user:preference:profile",
                value="深色模式和简洁回答",
            ),
            _memory(
                content="The project supports Windows, Linux, and macOS.",
                memory_type="domain_fact",
                evidence="The project supports Windows, Linux, and macOS",
                event_id="boundary-turn-user",
                subject="supported platforms",
                value="Windows, Linux, macOS",
            ),
            _memory(
                content="The service retries the request when it fails.",
                memory_type="domain_fact",
                evidence="The service retries the request when it fails",
                event_id="boundary-turn-user",
                subject="the service",
                value="retries the request when it fails",
            ),
            _memory(
                content="The API uses PostgreSQL because it requires JSON operators.",
                memory_type="domain_fact",
                evidence="The API uses PostgreSQL because it requires JSON operators",
                event_id="boundary-turn-user",
                subject="the api database",
                value="postgresql",
            ),
            _memory(
                content="The deployment policy requires approval because this reduces risk.",
                memory_type="constraint",
                evidence="The deployment policy requires approval because this reduces risk",
                event_id="boundary-turn-user",
                subject="the deployment policy",
                value="requires approval",
            ),
            _memory(
                content="This project uses PostgreSQL 16.",
                memory_type="domain_fact",
                evidence="This project uses PostgreSQL 16",
                event_id="boundary-turn-user",
                subject="the project",
                value="uses postgresql 16",
            ),
            _memory(
                content="The API depends on it for caching.",
                memory_type="domain_fact",
                evidence="The API depends on it for caching",
                event_id="boundary-turn-user",
                subject="api caching dependency",
                value="unknown",
            ),
            _memory(
                content="The user says it requires weekly audits.",
                memory_type="constraint",
                evidence="The user says it requires weekly audits",
                event_id="boundary-turn-user",
                subject="audit cadence",
                value="weekly",
            ),
            _memory(
                content="It requires weekly audits.",
                memory_type="constraint",
                evidence="It requires weekly audits",
                event_id="boundary-turn-user",
                subject="audit cadence",
                value="weekly",
            ),
        ]

    user_content = (
        "I prefer dark mode and concise answers. 用户偏好深色模式和简洁回答。 "
        "The project supports Windows, Linux, and macOS. "
        "The service retries the request when it fails. "
        "The API uses PostgreSQL because it requires JSON operators. "
        "The deployment policy requires approval because this reduces risk. "
        "This project uses PostgreSQL 16. "
        "The API depends on it for caching. "
        "The user says it requires weekly audits. It requires weekly audits."
    )
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="boundary-session",
            turn_id="boundary-turn",
            user_content=user_content,
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="boundary-session",
            turn_id="boundary-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        reasons = " ".join(
            str(item.get("reason", "")) for item in round_payload["semantic_diagnostics"]
        )
        assert reasons.count("more than one atomic claim") == 2
        assert reasons.count("unbound pronoun") == 3
        assert {
            item["summary"] for item in _formal_memories(
                client, headers, "semantic-project"
            )
        } == {
            "The project supports Windows, Linux, and macOS.",
            "The service retries the request when it fails.",
            "The API uses PostgreSQL because it requires JSON operators.",
            "The deployment policy requires approval because this reduces risk.",
            "This project uses PostgreSQL 16.",
        }


def test_question_gate_allows_requests_embedded_questions_and_declarations(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="请求超时时间必须设置为 30 秒。",
                memory_type="constraint",
                evidence="请求超时时间必须设置为 30 秒",
                event_id="question-gate-turn-user",
                subject="请求超时时间",
                value="必须设置为 30 秒",
            ),
            _memory(
                content="请求头必须包含 trace_id。",
                memory_type="constraint",
                evidence="请求头必须包含 trace_id",
                event_id="question-gate-turn-user",
                subject="请求头",
                value="必须包含 trace_id",
            ),
            _memory(
                content="How we deploy is documented in the runbook.",
                memory_type="domain_fact",
                evidence="How we deploy is documented in the runbook",
                event_id="question-gate-turn-user",
                subject="deployment process",
                value="documented in the runbook",
            ),
            _memory(
                content="The user prefers dark mode for the editor.",
                memory_type="preference",
                evidence="I prefer dark mode for the editor",
                event_id="question-gate-turn-user",
                subject="user:preference:editor",
                value="dark mode",
            ),
            _memory(
                content="用户偏好深色模式。",
                memory_type="preference",
                evidence="我偏好深色模式",
                event_id="question-gate-turn-user",
                subject="user:preference:mode",
                value="dark",
            ),
        ]

    user_content = (
        "请求超时时间必须设置为 30 秒。请求头必须包含 trace_id。 "
        "How we deploy is documented in the runbook. "
        "I prefer dark mode for the editor. How do I enable it? "
        "我偏好深色模式，这样可以吗？"
    )
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="question-gate-session",
            turn_id="question-gate-turn",
            user_content=user_content,
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="question-gate-session",
            turn_id="question-gate-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        assert len(_formal_memories(client, headers, "semantic-project")) == 5
        assert "question" not in " ".join(
            str(item.get("reason", "")) for item in round_payload["semantic_diagnostics"]
        )


def _pending_rows(state_dir: Path) -> tuple[int, int]:
    with sqlite3.connect(state_dir / "mem2" / "idempotency.sqlite3") as connection:
        idempotency_pending = connection.execute(
            "SELECT COUNT(*) FROM memory_write_idempotency WHERE status = 'pending'"
        ).fetchone()[0]
    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        browser_pending = connection.execute(
            "SELECT COUNT(*) FROM memory_browser_index_pending"
        ).fetchone()[0]
    return int(idempotency_pending), int(browser_pending)


def test_pending_semantic_idempotency_recovers_after_same_process_retry(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    payload = _memory(
        content="The user prefers recovery-safe dark mode.",
        memory_type="preference",
        evidence="I prefer recovery-safe dark mode",
        event_id="pending-retry-turn-user",
        subject="user:preference:mode",
        value="recovery-safe dark mode",
    )

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [payload]

    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    original_complete = kernel._complete_idempotency
    calls = {"completion": 0}

    def fail_first_completion(
        library_id: str,
        idempotency_key: str,
        request_hash: str,
        memory_id: str,
    ) -> None:
        if idempotency_key.startswith("semantic:") and calls["completion"] == 0:
            calls["completion"] += 1
            raise RuntimeError("simulated crash before idempotency completion")
        original_complete(library_id, idempotency_key, request_hash, memory_id)

    kernel._complete_idempotency = fail_first_completion  # type: ignore[method-assign]
    app = create_app(Settings(state_dir=state_dir), memory_kernel=kernel)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="pending-retry-session",
            turn_id="pending-retry-turn",
            user_content="I prefer recovery-safe dark mode.",
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        failed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="pending-retry-session",
            turn_id="pending-retry-turn",
        )
        assert failed["semantic_status"] == "failed"
        assert _pending_rows(state_dir) == (0, 0)
        assert _formal_memories(client, headers, "semantic-project") == []

        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "pending-retry-session"},
        )
        assert consolidated.status_code == 202, consolidated.text
        executor.drain()
        completed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="pending-retry-session",
            turn_id="pending-retry-turn",
        )
        assert completed["semantic_status"] == "completed"
        assert _pending_rows(state_dir) == (0, 0)
        memories = _formal_memories(client, headers, "semantic-project")
        assert [item["summary"] for item in memories] == [
            "The user prefers recovery-safe dark mode."
        ]


def test_pending_semantic_idempotency_recovers_after_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    first_executor = ManualSemanticExecutor()
    payload = _memory(
        content="The user prefers restart-safe dark mode.",
        memory_type="preference",
        evidence="I prefer restart-safe dark mode",
        event_id="pending-restart-turn-user",
        subject="user:preference:mode",
        value="restart-safe dark mode",
    )

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [payload]

    first_kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=first_executor,
    )
    original_complete = first_kernel._complete_idempotency
    calls = {"completion": 0}

    def fail_first_completion(
        library_id: str,
        idempotency_key: str,
        request_hash: str,
        memory_id: str,
    ) -> None:
        if idempotency_key.startswith("semantic:") and calls["completion"] == 0:
            calls["completion"] += 1
            raise RuntimeError("simulated restart before idempotency completion")
        original_complete(library_id, idempotency_key, request_hash, memory_id)

    first_kernel._complete_idempotency = fail_first_completion  # type: ignore[method-assign]
    first_kernel._cleanup_semantic_version = lambda **kwargs: None  # type: ignore[method-assign]
    first_kernel._release_pending_idempotency = lambda *args: None  # type: ignore[method-assign]
    first_kernel._complete_browser_index_pending = lambda token: None  # type: ignore[method-assign]
    first_app = create_app(Settings(state_dir=state_dir), memory_kernel=first_kernel)
    with TestClient(first_app) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="pending-restart-session",
            turn_id="pending-restart-turn",
            user_content="I prefer restart-safe dark mode.",
            assistant_content="Noted.",
            cwd=root,
        )
        first_executor.drain()
        failed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="pending-restart-session",
            turn_id="pending-restart-turn",
        )
        assert failed["semantic_status"] == "failed"
        assert _pending_rows(state_dir) == (1, 1)

    recovering_executor = ManualSemanticExecutor()

    def recovering_extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [payload]

    with TestClient(
        _app(state_dir, recovering_extractor, recovering_executor)
    ) as client:
        headers = _headers(state_dir)
        assert recovering_executor.pending_count == 1
        recovering_executor.drain()
        recovered = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="pending-restart-session",
            turn_id="pending-restart-turn",
        )
        assert recovered["semantic_status"] == "completed"
        assert _pending_rows(state_dir) == (0, 0)
        memories = _formal_memories(client, headers, "semantic-project")
        assert [item["summary"] for item in memories] == [
            "The user prefers restart-safe dark mode."
        ]


def test_non_semantic_create_keeps_pending_browser_index_after_idempotency_failure(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    payload = {
        "region_id": "semantic-project",
        "content": "The project uses SQLite for local storage.",
        "memory_type": "domain_fact",
        "source_references": [],
        "actor": "test",
        "idempotency_key": "non-semantic-pending-1",
    }

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return []

    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    original_complete = kernel._complete_idempotency
    calls = {"completion": 0}

    def fail_first_completion(
        library_id: str,
        idempotency_key: str,
        request_hash: str,
        memory_id: str,
    ) -> None:
        if calls["completion"] == 0:
            calls["completion"] += 1
            raise RuntimeError("simulated crash before idempotency completion")
        original_complete(library_id, idempotency_key, request_hash, memory_id)

    kernel._complete_idempotency = fail_first_completion  # type: ignore[method-assign]
    with TestClient(create_app(Settings(state_dir=state_dir), memory_kernel=kernel)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        response = client.post("/api/v1/memories", headers=headers, json=payload)
        assert response.status_code == 503, response.text
        assert _pending_rows(state_dir) == (1, 1)
        listing = _formal_memories(client, headers, "semantic-project")
        assert [item["summary"] for item in listing] == [
            "The project uses SQLite for local storage."
        ]
        assert _pending_rows(state_dir) == (1, 0)
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"region_id": "semantic-project", "query": "SQLite", "limit": 10},
        )
        assert search.status_code == 200, search.text
        assert any(
            "SQLite" in item["content"] for item in search.json()["results"]
        )

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        listing = _formal_memories(client, headers, "semantic-project")
        assert [item["summary"] for item in listing] == [
            "The project uses SQLite for local storage."
        ]
        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"region_id": "semantic-project", "query": "SQLite", "limit": 10},
        )
        assert search.status_code == 200, search.text
        assert any(
            "SQLite" in item["content"] for item in search.json()["results"]
        )
        duplicate = client.post("/api/v1/memories", headers=headers, json=payload)
        assert duplicate.status_code == 409, duplicate.text
        assert [
            item["summary"]
            for item in _formal_memories(client, headers, "semantic-project")
        ] == ["The project uses SQLite for local storage."]


def test_compound_noun_lists_and_object_pronouns_are_skipped_without_false_positives(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers dark mode, weekly audits, and concise answers.",
                memory_type="preference",
                evidence="I prefer dark mode, weekly audits, and concise answers",
                event_id="parallel-turn-user",
                subject="user:preference:profile",
                value="dark mode, weekly audits, concise answers",
            ),
            _memory(
                content="The API depends on it for caching.",
                memory_type="domain_fact",
                evidence="The API depends on Redis for caching",
                event_id="parallel-turn-user",
                subject="api caching dependency",
                value="redis",
            ),
            _memory(
                content="The API depends on Redis for caching.",
                memory_type="domain_fact",
                evidence="The API depends on Redis for caching",
                event_id="parallel-turn-user",
                subject="api caching dependency",
                value="redis",
            ),
            _memory(
                content="The user prefers dark mode.",
                memory_type="preference",
                evidence="I prefer dark mode",
                event_id="parallel-turn-user",
                subject="user:preference:theme",
                value="dark",
            ),
            _memory(
                content="The project supports Windows, Linux, and macOS.",
                memory_type="domain_fact",
                evidence="The project supports Windows, Linux, and macOS",
                event_id="parallel-turn-user",
                subject="supported platforms",
                value="Windows, Linux, macOS",
            ),
        ]

    user_content = (
        "I prefer dark mode, weekly audits, and concise answers. "
        "The API depends on Redis for caching. I prefer dark mode. "
        "The project supports Windows, Linux, and macOS."
    )
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="parallel-session",
            turn_id="parallel-turn",
            user_content=user_content,
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="parallel-session",
            turn_id="parallel-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        reasons = " ".join(
            str(item.get("reason", "")) for item in round_payload["semantic_diagnostics"]
        )
        assert "more than one atomic claim" in reasons
        assert "unbound pronoun" in reasons
        assert {
            item["summary"] for item in _formal_memories(
                client, headers, "semantic-project"
            )
        } == {
            "The API depends on Redis for caching.",
            "The user prefers dark mode.",
            "The project supports Windows, Linux, and macOS.",
        }


def test_coordinated_predicates_and_object_pronouns_are_skipped(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The project uses PostgreSQL and stores vectors in Milvus.",
                memory_type="domain_fact",
                evidence="The project uses PostgreSQL and stores vectors in Milvus",
                event_id="coordinated-turn-user",
                subject="the project",
                value="postgresql and milvus",
            ),
            _memory(
                content="The service reads from PostgreSQL and writes to Redis.",
                memory_type="domain_fact",
                evidence="The service reads from PostgreSQL and writes to Redis",
                event_id="coordinated-turn-user",
                subject="the service",
                value="postgresql and redis",
            ),
            _memory(
                content="The project supports Windows and runs on Linux.",
                memory_type="domain_fact",
                evidence="The project supports Windows and runs on Linux",
                event_id="coordinated-turn-user",
                subject="the project",
                value="windows and linux",
            ),
            _memory(
                content="The API accepts JSON and returns XML.",
                memory_type="domain_fact",
                evidence="The API accepts JSON and returns XML",
                event_id="coordinated-turn-user",
                subject="the api",
                value="json and xml",
            ),
            _memory(
                content="The team adopted it.",
                memory_type="decision",
                evidence="The team adopted it",
                event_id="coordinated-turn-user",
                subject="adoption",
                value="adopted it",
            ),
            _memory(
                content="The company bought it.",
                memory_type="domain_fact",
                evidence="The company bought it",
                event_id="coordinated-turn-user",
                subject="purchase",
                value="bought it",
            ),
            _memory(
                content="The service handles it automatically.",
                memory_type="domain_fact",
                evidence="The service handles it automatically",
                event_id="coordinated-turn-user",
                subject="handling",
                value="handles it",
            ),
            _memory(
                content="The tool found it.",
                memory_type="domain_fact",
                evidence="The tool found it",
                event_id="coordinated-turn-user",
                subject="discovery",
                value="found it",
            ),
            _memory(
                content="The project supports Windows, Linux, and macOS.",
                memory_type="domain_fact",
                evidence="The project supports Windows, Linux, and macOS",
                event_id="coordinated-turn-user",
                subject="supported platforms",
                value="Windows, Linux, macOS",
            ),
            _memory(
                content="The project supports Windows and macOS.",
                memory_type="domain_fact",
                evidence="The project supports Windows and macOS",
                event_id="coordinated-turn-user",
                subject="supported desktop platforms",
                value="Windows and macOS",
            ),
            _memory(
                content="The API uses PostgreSQL because it requires JSON operators.",
                memory_type="domain_fact",
                evidence="The API uses PostgreSQL because it requires JSON operators",
                event_id="coordinated-turn-user",
                subject="the api database",
                value="postgresql",
            ),
            _memory(
                content="The service retries the request when it fails.",
                memory_type="domain_fact",
                evidence="The service retries the request when it fails",
                event_id="coordinated-turn-user",
                subject="retry policy",
                value="retries on failure",
            ),
            _memory(
                content="The deployment policy requires approval because this reduces risk.",
                memory_type="constraint",
                evidence="The deployment policy requires approval because this reduces risk",
                event_id="coordinated-turn-user",
                subject="the deployment policy",
                value="requires approval",
            ),
            _memory(
                content="This project uses PostgreSQL 16.",
                memory_type="domain_fact",
                evidence="This project uses PostgreSQL 16",
                event_id="coordinated-turn-user",
                subject="the project database",
                value="postgresql 16",
            ),
        ]

    user_content = (
        "The project uses PostgreSQL and stores vectors in Milvus. "
        "The service reads from PostgreSQL and writes to Redis. "
        "The project supports Windows and runs on Linux. "
        "The API accepts JSON and returns XML. "
        "The team adopted it. The company bought it. "
        "The service handles it automatically. The tool found it. "
        "The project supports Windows, Linux, and macOS. "
        "The project supports Windows and macOS. "
        "The API uses PostgreSQL because it requires JSON operators. "
        "The service retries the request when it fails. "
        "The deployment policy requires approval because this reduces risk. "
        "This project uses PostgreSQL 16."
    )
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="coordinated-session",
            turn_id="coordinated-turn",
            user_content=user_content,
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="coordinated-session",
            turn_id="coordinated-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        reasons = [
            str(item.get("reason", "")) for item in round_payload["semantic_diagnostics"]
        ]
        joined = " ".join(reasons)
        assert joined.count("more than one atomic claim") == 4
        assert joined.count("unbound pronoun") == 4
        assert {
            item["summary"] for item in _formal_memories(
                client, headers, "semantic-project"
            )
        } == {
            "The project supports Windows, Linux, and macOS.",
            "The project supports Windows and macOS.",
            "The API uses PostgreSQL because it requires JSON operators.",
            "The service retries the request when it fails.",
            "The deployment policy requires approval because this reduces risk.",
            "This project uses PostgreSQL 16.",
        }


def test_pure_question_event_cannot_supply_inferred_user_evidence(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The project database uses incremental migrations.",
                memory_type="domain_fact",
                evidence="the project database",
                event_id="pure-question-turn-user",
                subject="project database migration",
                value="incremental",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="pure-question-session",
            turn_id="pure-question-turn",
            user_content="How should the project database be migrated?",
            assistant_content="Use incremental migrations.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="pure-question-session",
            turn_id="pure-question-turn",
        )
        assert round_payload["semantic_status"] == "skipped"
        assert "user event is a question" in " ".join(
            str(item["reason"]) for item in round_payload["semantic_diagnostics"]
        )
        assert _formal_memories(client, headers, "semantic-project") == []


def test_wh_declarations_and_chinese_statement_forms_are_published(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The project uses PostgreSQL.",
                memory_type="domain_fact",
                evidence="What matters is that the project uses PostgreSQL",
                event_id="wh-statement-turn-user",
                subject="the project database",
                value="postgresql",
            ),
            _memory(
                content="The outage cause is documented in the postmortem.",
                memory_type="domain_fact",
                evidence="What caused the outage is documented in the postmortem",
                event_id="wh-statement-turn-user",
                subject="outage cause",
                value="documented in the postmortem",
            ),
            _memory(
                content="The database choice is undecided.",
                memory_type="decision",
                evidence="Which database to use is still undecided",
                event_id="wh-statement-turn-user",
                subject="database choice",
                value="undecided",
            ),
            _memory(
                content="The project cost is unclear.",
                memory_type="domain_fact",
                evidence="How much the project costs is unclear",
                event_id="wh-statement-turn-user",
                subject="project cost",
                value="unclear",
            ),
            _memory(
                content="缓存是否启用取决于部署环境。",
                memory_type="constraint",
                evidence="是否启用缓存取决于部署环境",
                event_id="wh-statement-turn-user",
                subject="缓存策略",
                value="取决于部署环境",
            ),
            _memory(
                content="部署流程已在运行手册中说明。",
                memory_type="domain_fact",
                evidence="如何部署已在运行手册中说明",
                event_id="wh-statement-turn-user",
                subject="部署流程",
                value="运行手册",
            ),
            _memory(
                content="构建失败原因已记录在日志中。",
                memory_type="domain_fact",
                evidence="为什么构建失败已记录在日志中",
                event_id="wh-statement-turn-user",
                subject="构建失败原因",
                value="日志",
            ),
        ]

    user_content = (
        "What matters is that the project uses PostgreSQL. "
        "What caused the outage is documented in the postmortem. "
        "Which database to use is still undecided. "
        "How much the project costs is unclear. "
        "是否启用缓存取决于部署环境。如何部署已在运行手册中说明。"
        "为什么构建失败已记录在日志中。"
    )
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="wh-statement-session",
            turn_id="wh-statement-turn",
            user_content=user_content,
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="wh-statement-session",
            turn_id="wh-statement-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        assert "question" not in " ".join(
            str(item.get("reason", "")) for item in round_payload["semantic_diagnostics"]
        )
        assert len(_formal_memories(client, headers, "semantic-project")) == 7


def test_pure_request_events_do_not_publish_semantic_memories(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The current project is a Python service.",
                memory_type="domain_fact",
                evidence="当前项目",
                event_id="pure-request-turn-user",
                subject="the project",
                value="a python service",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="pure-request-session",
            turn_id="pure-request-turn",
            user_content="帮我分析下当前项目",
            assistant_content="好的。",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="pure-request-session",
            turn_id="pure-request-turn",
        )
        assert round_payload["semantic_status"] == "skipped"
        assert "request rather than a durable statement" in " ".join(
            str(item["reason"]) for item in round_payload["semantic_diagnostics"]
        )
        assert _formal_memories(client, headers, "semantic-project") == []


def test_explicit_claim_values_conflict_only_when_mutually_exclusive(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "user"
        )
        event_id = str(user["event_id"])
        content_by_turn = {
            "claim-conflict-1-user": (
                "The deployment policy requires immutable images.",
                "requires immutable images",
                "immutable images",
            ),
            "claim-conflict-2-user": (
                "The deployment policy requires mutable images.",
                "requires mutable images",
                "mutable images",
            ),
            "claim-conflict-3-user": (
                "The deployment policy uses a blue-green rollout.",
                "uses a blue-green rollout",
                "blue-green rollout",
            ),
        }
        content, evidence, value = content_by_turn[event_id]
        return [
            _memory(
                content=content,
                memory_type="constraint",
                evidence=evidence,
                event_id=event_id,
                subject="deployment policy",
                value=value,
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        for index in (1, 2, 3):
            user_content = {
                1: "The deployment policy requires immutable images.",
                2: "The deployment policy requires mutable images.",
                3: "The deployment policy uses a blue-green rollout.",
            }[index]
            _capture_turn(
                client,
                headers,
                session_id="claim-conflict-session",
                turn_id=f"claim-conflict-{index}",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
                second=index * 10,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id="claim-conflict-session",
                turn_id=f"claim-conflict-{index}",
            )
            if index == 2:
                assert round_payload["semantic_status"] == "skipped"
                assert "conflicts" in " ".join(
                    str(item["reason"])
                    for item in round_payload["semantic_diagnostics"]
                )
            else:
                assert round_payload["semantic_status"] == "completed"
        assert {
            item["summary"] for item in _formal_memories(
                client, headers, "semantic-project"
            )
        } == {
            "The deployment policy requires immutable images.",
            "The deployment policy uses a blue-green rollout.",
        }


def test_frequency_claim_values_conflict_only_within_the_same_dimension(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        turn_id = str(context["turn_id"])
        payloads = {
            "cadence-1": _memory(
                content="Audits run weekly.",
                memory_type="constraint",
                evidence="Audits run weekly",
                event_id="cadence-1-user",
                subject="audit cadence",
                value="weekly",
            ),
            "cadence-2": _memory(
                content="Audits run monthly.",
                memory_type="constraint",
                evidence="Audits run monthly",
                event_id="cadence-2-user",
                subject="audit cadence",
                value="monthly",
            ),
        }
        return [payloads[turn_id]]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        for index, content in ((1, "Audits run weekly."), (2, "Audits run monthly.")):
            _capture_turn(
                client,
                headers,
                session_id="cadence-session",
                turn_id=f"cadence-{index}",
                user_content=content,
                assistant_content="Noted.",
                cwd=root,
                second=index * 10,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id="cadence-session",
                turn_id=f"cadence-{index}",
            )
            if index == 1:
                assert round_payload["semantic_status"] == "completed"
            else:
                assert round_payload["semantic_status"] == "skipped"
                assert "conflicts" in " ".join(
                    str(item["reason"])
                    for item in round_payload["semantic_diagnostics"]
                )
        assert [
            item["summary"] for item in _formal_memories(client, headers, "semantic-project")
        ] == ["Audits run weekly."]


def test_late_assistant_version_wins_and_old_thread_does_not_complete(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = BoundedSemanticExecutor(max_workers=1, max_region_workers=1)
    old_started = threading.Event()
    release_old = threading.Event()
    calls = 0

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        messages = context["messages"]
        assert isinstance(messages, list)
        assistant = [
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "assistant"
        ][-1]
        if "Old conclusion" in str(assistant["content"]):
            old_started.set()
            release_old.wait()
            return [
                _memory(
                    content="The user prefers the old deployment plan.",
                    memory_type="preference",
                    evidence="I prefer the old deployment plan",
                    event_id="version-user",
                    subject="user:preference:deployment",
                    value="old deployment plan",
                )
            ]
        return [
            _memory(
                content="The user prefers the final deployment plan.",
                memory_type="preference",
                evidence="I prefer the final deployment plan",
                event_id="version-user",
                subject="user:preference:deployment",
                value="final deployment plan",
            )
        ]

    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    app = create_app(Settings(state_dir=state_dir), memory_kernel=kernel)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture(
            client,
            headers,
            event_id="version-user",
            session_id="version-session",
            turn_id="version-turn",
            kind="user",
            content="I prefer the old deployment plan. I prefer the final deployment plan.",
            cwd=root,
            occurred_at="2026-10-28T03:00:00Z",
        )
        _capture(
            client,
            headers,
            event_id="version-old-assistant",
            session_id="version-session",
            turn_id="version-turn",
            kind="assistant",
            content="Old conclusion.",
            cwd=root,
            occurred_at="2026-10-28T03:00:01Z",
        )
        old_started.wait()
        _capture(
            client,
            headers,
            event_id="version-final-assistant",
            session_id="version-session",
            turn_id="version-turn",
            kind="assistant",
            content="Final conclusion.",
            cwd=root,
            occurred_at="2026-10-28T03:00:02Z",
        )
        release_old.set()
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="version-session",
            turn_id="version-turn",
        )
        assert round_payload["semantic_status"] == "completed"
        assert calls == 2
        memories = _formal_memories(client, headers, "semantic-project")
        assert [item["summary"] for item in memories] == [
            "The user prefers the final deployment plan."
        ]


def test_final_assistant_replaces_an_already_completed_old_version(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        assistant = [
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "assistant"
        ][-1]
        final = "Final conclusion" in str(assistant["content"])
        return [
            _memory(
                content=(
                    "The user prefers the final deployment plan."
                    if final
                    else "The user prefers the old deployment plan."
                ),
                memory_type="preference",
                evidence=(
                    "I prefer the final deployment plan"
                    if final
                    else "I prefer the old deployment plan"
                ),
                event_id="replacement-user",
                subject="user:preference:deployment",
                value="final deployment plan" if final else "old deployment plan",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture(
            client,
            headers,
            event_id="replacement-user",
            session_id="replacement-session",
            turn_id="replacement-turn",
            kind="user",
            content="I prefer the old deployment plan. I prefer the final deployment plan.",
            cwd=root,
            occurred_at="2026-10-28T04:00:00Z",
        )
        _capture(
            client,
            headers,
            event_id="replacement-old-assistant",
            session_id="replacement-session",
            turn_id="replacement-turn",
            kind="assistant",
            content="Old conclusion.",
            cwd=root,
            occurred_at="2026-10-28T04:00:01Z",
        )
        executor.drain()
        old_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="replacement-session",
            turn_id="replacement-turn",
        )
        assert old_round["semantic_status"] == "completed"
        old_id = old_round["semantic_memory_ids"][0]

        _capture(
            client,
            headers,
            event_id="replacement-final-assistant",
            session_id="replacement-session",
            turn_id="replacement-turn",
            kind="assistant",
            content="Final conclusion.",
            cwd=root,
            occurred_at="2026-10-28T04:00:02Z",
        )
        executor.drain()
        final_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="replacement-session",
            turn_id="replacement-turn",
        )
        assert final_round["semantic_status"] == "completed"
        final_id = final_round["semantic_memory_ids"][0]
        assert final_id != old_id
        memories = _formal_memories(client, headers, "semantic-project")
        assert [item["id"] for item in memories] == [final_id]
        assert [item["summary"] for item in memories] == [
            "The user prefers the final deployment plan."
        ]
        detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{final_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        metadata = detail.json()["user_metadata"]
        assert metadata["source_event_ids"] == [
            "replacement-user",
            "replacement-old-assistant",
            "replacement-final-assistant",
        ]
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            source_versions = connection.execute(
                """SELECT source_fingerprint FROM semantic_memory_versions
                   WHERE memory_id = ?""",
                (final_id,),
            ).fetchall()
        assert [row[0] for row in source_versions] == [
            final_round["semantic_memory_refs"][0]["source_fingerprint"]
        ]
        old_detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{old_id}",
            headers=headers,
        )
        assert old_detail.status_code == 404, old_detail.text


def test_final_assistant_replaces_a_partial_old_version(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    injected = False

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        assistant = [
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "assistant"
        ][-1]
        if "Final conclusion" in str(assistant["content"]):
            return [
                _memory(
                    content="The user prefers the final atomic deployment policy.",
                    memory_type="preference",
                    evidence="I prefer the final atomic deployment policy",
                    event_id="partial-user",
                    subject="user:preference:deployment",
                    value="final atomic deployment policy",
                )
            ]
        return [
            _memory(
                content="The user prefers the old first deployment policy.",
                memory_type="preference",
                evidence="I prefer the old first deployment policy",
                event_id="partial-user",
                subject="user:preference:deployment",
                value="old first deployment policy",
            ),
            _memory(
                content="The user requires the old second deployment gate.",
                memory_type="constraint",
                evidence="The deployment gate requires the old second deployment gate",
                event_id="partial-user",
                subject="deployment gate",
                value="old second deployment gate",
            ),
        ]

    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    original_write = kernel._write

    def write_and_inject(request: object) -> dict[str, object]:
        nonlocal injected
        response = original_write(request)  # type: ignore[arg-type]
        if (
            not injected
            and getattr(request, "actor", "") == "semantic-distillation"
        ):
            injected = True
            _capture(
                client,
                headers,
                event_id="partial-final-assistant",
                session_id="partial-session",
                turn_id="partial-turn",
                kind="assistant",
                content="Final conclusion.",
                cwd=root,
                occurred_at="2026-10-28T04:10:02Z",
            )
        return response

    kernel._write = write_and_inject  # type: ignore[method-assign]
    app = create_app(Settings(state_dir=state_dir), memory_kernel=kernel)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture(
            client,
            headers,
            event_id="partial-user",
            session_id="partial-session",
            turn_id="partial-turn",
            kind="user",
            content=(
                "I prefer the old first deployment policy. "
                "The deployment gate requires the old second deployment gate. "
                "I prefer the final atomic deployment policy."
            ),
            cwd=root,
            occurred_at="2026-10-28T04:10:00Z",
        )
        _capture(
            client,
            headers,
            event_id="partial-old-assistant",
            session_id="partial-session",
            turn_id="partial-turn",
            kind="assistant",
            content="Old conclusion.",
            cwd=root,
            occurred_at="2026-10-28T04:10:01Z",
        )
        executor.drain()
        assert injected is True
        final_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="partial-session",
            turn_id="partial-turn",
        )
        assert final_round["semantic_status"] == "completed"
        memories = _formal_memories(client, headers, "semantic-project")
        assert [item["summary"] for item in memories] == [
            "The user prefers the final atomic deployment policy."
        ]


def test_region_publication_is_atomic_under_concurrent_conflicts(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = BoundedSemanticExecutor(max_workers=2, max_region_workers=2)
    model_barrier = threading.Barrier(2)

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "user"
        )
        model_barrier.wait()
        value = "dark" if "dark" in str(user["content"]) else "light"
        return [
            _memory(
                content=f"The project theme is {value}.",
                memory_type="preference",
                evidence=f"theme is {value}",
                event_id=str(user["event_id"]),
                subject="theme",
                value=value,
            )
        ]

    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    app = create_app(Settings(state_dir=state_dir), memory_kernel=kernel)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="concurrent-session",
            turn_id="dark-turn",
            user_content="The project theme is dark.",
            assistant_content="Confirmed dark.",
            cwd=root,
            second=0,
        )
        _capture_turn(
            client,
            headers,
            session_id="concurrent-session",
            turn_id="light-turn",
            user_content="The project theme is light.",
            assistant_content="Confirmed light.",
            cwd=root,
            second=10,
        )
        executor.drain()
        dark_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="concurrent-session",
            turn_id="dark-turn",
        )
        light_round = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="concurrent-session",
            turn_id="light-turn",
        )
        assert {dark_round["semantic_status"], light_round["semantic_status"]} == {
            "completed",
            "skipped",
        }
        memories = _formal_memories(client, headers, "semantic-project")
        assert len(memories) == 1


def test_bounded_executor_keeps_worker_count_bounded_for_300_pending_tasks() -> None:
    executor = BoundedSemanticExecutor(
        max_workers=4,
        max_pending=64,
        max_region_workers=1,
    )
    gate = threading.Event()
    first_regions_started = threading.Event()
    started_lock = threading.Lock()
    started_by_region: dict[str, int] = {}

    def blocked_task(region_id: str) -> None:
        with started_lock:
            started_by_region[region_id] = started_by_region.get(region_id, 0) + 1
            if len(started_by_region) == 2:
                first_regions_started.set()
        gate.wait()

    accepted = sum(
        executor.submit(
            f"task-{index}",
            "region-a" if index % 2 == 0 else "region-b",
            lambda region_id="region-a" if index % 2 == 0 else "region-b": blocked_task(
                region_id
            ),
        )
        for index in range(300)
    )
    first_regions_started.wait()
    with started_lock:
        assert started_by_region == {"region-a": 1, "region-b": 1}
    worker_threads = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("semantic-distillation")
    ]
    assert accepted == 64
    assert executor.worker_count == 4
    assert len(worker_threads) <= 4
    gate.set()
    executor.drain()
    executor.close()


def test_replay_consolidate_retry_and_restart_are_idempotent(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    calls = 0
    fail_first = True

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls, fail_first
        calls += 1
        if fail_first:
            fail_first = False
            raise RuntimeError("transient semantic failure")
        del context
        return [
            _memory(
                content="The user prefers deterministic background tests.",
                memory_type="preference",
                evidence="prefer deterministic background tests",
                event_id="retry-turn-user",
                subject="user:preference:testing",
                value="deterministic background tests",
            )
        ]

    executor = ManualSemanticExecutor()
    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="retry-session",
            turn_id="retry-turn",
            user_content="I prefer deterministic background tests.",
            assistant_content="I will use deterministic background tests.",
            cwd=root,
        )
        executor.drain()
        failed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="retry-session",
            turn_id="retry-turn",
        )
        assert failed["semantic_status"] == "failed"
        assert "transient semantic failure" in str(failed["semantic_error"])

        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "retry-session"},
        )
        assert consolidated.status_code == 202, consolidated.text
        assert executor.pending_count >= 1
        executor.drain()
        completed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="retry-session",
            turn_id="retry-turn",
        )
        assert completed["semantic_status"] == "completed"
        memory_id = completed["semantic_memory_ids"][0]
        assert calls == 2

        assistant = {
            "event_id": "retry-turn-assistant",
            "session_id": "retry-session",
            "turn_id": "retry-turn",
            "event_kind": "assistant",
            "content": "I will use deterministic background tests.",
            "occurred_at": "2026-10-28T01:00:01Z",
            "cwd": str(root),
        }
        replay = client.post("/api/v1/capture/events", headers=headers, json=assistant)
        assert replay.status_code == 202, replay.text
        assert replay.json()["duplicate"] is True
        assert executor.pending_count == 0
        assert calls == 2
        assert len(_formal_memories(client, headers, "semantic-project")) == 1

    recovering_executor = ManualSemanticExecutor()

    def recovering_extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers deterministic background tests.",
                memory_type="preference",
                evidence="prefer deterministic background tests",
                event_id="retry-turn-user",
                subject="user:preference:testing",
                value="deterministic background tests",
            )
        ]

    with TestClient(
        _app(state_dir, recovering_extractor, recovering_executor)
    ) as client:
        headers = _headers(state_dir)
        assert recovering_executor.pending_count == 0
        recovered = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="retry-session",
            turn_id="retry-turn",
        )
        assert recovered["semantic_memory_ids"] == [memory_id]
        assert len(_formal_memories(client, headers, "semantic-project")) == 1


def test_restart_recovers_a_failed_record_without_duplicate_memory(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"

    def unavailable(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        raise RuntimeError("restart recovery failure")

    def recovered(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers restart-safe semantic distillation.",
                memory_type="preference",
                evidence="restart-safe semantic distillation",
                event_id="restart-turn-user",
                subject="user:preference:recovery",
                value="restart-safe semantic distillation",
            )
        ]

    failed_executor = ManualSemanticExecutor()
    with TestClient(_app(state_dir, unavailable, failed_executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="restart-session",
            turn_id="restart-turn",
            user_content="I prefer restart-safe semantic distillation.",
            assistant_content="I will preserve restart-safe semantic distillation.",
            cwd=root,
        )
        failed_executor.drain()
        failed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="restart-session",
            turn_id="restart-turn",
        )
        assert "restart recovery failure" in str(failed["semantic_error"])

    recovered_executor = ManualSemanticExecutor()
    with TestClient(_app(state_dir, recovered, recovered_executor)) as client:
        headers = _headers(state_dir)
        assert recovered_executor.pending_count == 1
        recovered_executor.drain()
        completed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="restart-session",
            turn_id="restart-turn",
        )
        assert completed["semantic_status"] == "completed"
        assert completed["semantic_memory_ids"]
        assert len(_formal_memories(client, headers, "semantic-project")) == 1


def test_semantic_rounds_expose_memory_links_and_idempotency_metadata(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers one source-backed semantic memory.",
                memory_type="preference",
                evidence="one source-backed semantic memory",
                event_id="metadata-turn-user",
                subject="user:preference:memory",
                value="source-backed semantic memory",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="metadata-session",
            turn_id="metadata-turn",
            user_content="I prefer one source-backed semantic memory.",
            assistant_content="The source-backed semantic memory is preserved.",
            cwd=root,
        )
        executor.drain()
        round_payload = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="metadata-session",
            turn_id="metadata-turn",
        )
        memory_id = round_payload["semantic_memory_ids"][0]
        detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{memory_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        metadata = detail.json()["user_metadata"]
        assert metadata["semantic_schema"] == "semantic-distillation-v1"
        assert metadata["session_id"] == "metadata-session"
        assert metadata["turn_id"] == "metadata-turn"
        assert metadata["user_event_id"] == "metadata-turn-user"
        assert metadata["evidence_snippet"] == "one source-backed semantic memory"
        assert metadata["source_event_ids"] == [
            "metadata-turn-user",
            "metadata-turn-assistant",
        ]
        assert metadata["source_region_id"] == "semantic-project"
        assert metadata["confidence"] == pytest.approx(0.96)
        assert metadata["provenance"]

        maintenance = state_dir / "mem2" / "maintenance.sqlite3"
        with sqlite3.connect(maintenance) as connection:
            row = connection.execute(
                """SELECT semantic_status, semantic_memory_ids, semantic_memory_refs
                   FROM capture_evolutions WHERE session_id = ? AND turn_id = ?""",
                ("metadata-session", "metadata-turn"),
            ).fetchone()
        assert row is not None
        assert row[0] == "completed"
        assert json.loads(row[1]) == [memory_id]
        refs = json.loads(row[2])
        assert refs[0]["evidence_snippet"] == "one source-backed semantic memory"


def test_final_validation_atomicity_and_transient_action_boundaries(tmp_path: Path) -> None:
    cases = [
        (
            "coordinated-predicate",
            "The project uses PostgreSQL and enables caching.",
            _memory(
                content="The project uses PostgreSQL and enables caching.",
                memory_type="domain_fact",
                evidence="The project uses PostgreSQL and enables caching",
                event_id="coordinated-predicate-turn-user",
                subject="the project",
                value="postgresql and caching",
            ),
            "skipped",
        ),
        (
            "coordinated-verb",
            "The service retries and times out.",
            _memory(
                content="The service retries and times out.",
                memory_type="domain_fact",
                evidence="The service retries and times out",
                event_id="coordinated-verb-turn-user",
                subject="the service",
                value="retries and times out",
            ),
            "skipped",
        ),
        (
            "coordinated-modifier",
            "I prefer black and white themes.",
            _memory(
                content="The user prefers black and white themes.",
                memory_type="preference",
                evidence="I prefer black and white themes",
                event_id="coordinated-modifier-turn-user",
                subject="user:preference:theme",
                value="black and white",
            ),
            "completed",
        ),
        (
            "in-progress-action",
            "We are deploying the staging build.",
            _memory(
                content="We are deploying the staging build.",
                memory_type="decision",
                evidence="We are deploying the staging build",
                event_id="in-progress-action-turn-user",
                subject="staging deployment",
                value="in progress",
            ),
            "skipped",
        ),
    ]
    for name, user_content, result, expected in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]
        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == expected, round_payload
            memories = _formal_memories(client, headers, "semantic-project")
            assert len(memories) == (1 if expected == "completed" else 0)


def test_final_validation_question_request_and_deictic_boundaries(tmp_path: Path) -> None:
    cases = [
        (
            "comma-question",
            "What database, if any, should we use?",
            "What database",
            "The project uses PostgreSQL.",
        ),
        (
            "comma-request",
            "Could you, please, use PostgreSQL?",
            "use PostgreSQL",
            "The project uses PostgreSQL.",
        ),
        (
            "zh-question-no-mark",
            "如何确保缓存在高并发下的一致性",
            "如何确保缓存在高并发下的一致性",
            "缓存一致性通过分布式锁保证。",
        ),
        (
            "zh-request-comma",
            "帮我分析当前项目，并告诉我用了什么数据库",
            "当前项目",
            "The project uses PostgreSQL.",
        ),
        (
            "as-discussed",
            "As discussed, the setting should be enabled.",
            "As discussed, the setting should be enabled",
            "As discussed, the setting should be enabled.",
        ),
        (
            "zh-as-above",
            "如上所述，该设置应启用。",
            "如上所述，该设置应启用",
            "如上所述，该设置应启用。",
        ),
    ]
    for name, user_content, evidence, content in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type="domain_fact",
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject="the project",
            value="postgresql",
            durability="durable_statement",
        )
        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]
        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "skipped", round_payload
            assert _formal_memories(client, headers, "semantic-project") == []


def test_final_validation_single_value_conflicts_and_demonstrative_nouns(
    tmp_path: Path,
) -> None:
    conflict_cases = [
        ("timezone", "user timezone", "My timezone is {}", "UTC", "Asia/Shanghai"),
        ("owner", "project owner", "The project owner is {}", "Alice", "Bob"),
    ]
    for name, subject, template, first_value, second_value in conflict_cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()

        def extractor(
            context: dict[str, object],
            *,
            template: str = template,
            subject: str = subject,
            first_value: str = first_value,
            second_value: str = second_value,
        ) -> list[dict[str, object]]:
            turn = str(context["turn_id"])
            value = first_value if turn.endswith("1") else second_value
            content = template.format(value)
            return [
                _memory(
                    content=content,
                    memory_type="domain_fact",
                    evidence=content.rstrip("."),
                    event_id=f"{turn}-user",
                    subject=subject,
                    value=value.casefold(),
                )
            ]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            for index, value in enumerate((first_value, second_value), start=1):
                turn_id = f"{name}-turn-{index}"
                content = template.format(value)
                _capture_turn(
                    client,
                    headers,
                    session_id=f"{name}-session",
                    turn_id=turn_id,
                    user_content=content,
                    assistant_content="Noted.",
                    cwd=root,
                    second=index * 10,
                )
                executor.drain()
                round_payload = _round(
                    client,
                    headers,
                    region_id="semantic-project",
                    session_id=f"{name}-session",
                    turn_id=turn_id,
                )
                assert round_payload["semantic_status"] == (
                    "completed" if index == 1 else "skipped"
                ), round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == 1

    demonstrative_cases = [
        ("series", "This series uses PostgreSQL.", "This series uses PostgreSQL"),
        ("services", "These services use PostgreSQL.", "These services use PostgreSQL"),
    ]
    for name, user_content, content in demonstrative_cases:
        base = tmp_path / f"demonstrative-{name}"
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type="domain_fact",
            evidence=user_content.rstrip("."),
            event_id=f"{name}-turn-user",
            subject="database",
            value="postgresql",
        )
        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]
        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "completed", round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == 1


def test_semantic_reuse_never_deletes_a_manual_memory(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()
    results = [
        [
            _memory(
                content="The user prefers dark mode.",
                memory_type="preference",
                evidence="I prefer dark mode",
                event_id="manual-reuse-turn-user",
                subject="user:preference:theme",
                value="dark",
            )
        ],
        [
            _memory(
                content="My timezone is UTC.",
                memory_type="domain_fact",
                evidence="My timezone is UTC",
                event_id="manual-reuse-turn-user",
                subject="user timezone",
                value="UTC",
            )
        ],
    ]

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return results.pop(0) if results else []

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        manual = client.post(
            "/api/v1/memories",
            headers=headers,
            json={
                "region_id": "semantic-project",
                "content": "The user prefers dark mode.",
                "memory_type": "preference",
                "idempotency_key": "manual-reuse-1",
                "actor": "user",
            },
        )
        assert manual.status_code == 201, manual.text
        manual_id = manual.json()["id"]
        _capture_turn(
            client,
            headers,
            session_id="manual-reuse-session",
            turn_id="manual-reuse-turn",
            user_content="I prefer dark mode. My timezone is UTC.",
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        assert (
            client.get(
                f"/api/v1/libraries/semantic-project/memories/{manual_id}",
                headers=headers,
            ).status_code
            == 200
        )
        _capture(
            client,
            headers,
            event_id="manual-reuse-late-assistant",
            session_id="manual-reuse-session",
            turn_id="manual-reuse-turn",
            kind="assistant",
            content="Correction: UTC.",
            cwd=root,
            occurred_at="2026-10-28T01:00:09Z",
        )
        executor.drain()
        detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{manual_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["content"] == "The user prefers dark mode."
        assert {
            item["summary"] for item in _formal_memories(client, headers, "semantic-project")
        } == {"The user prefers dark mode.", "My timezone is UTC."}


def test_final_validation_chinese_coordination_pronouns_and_no_mark_questions(
    tmp_path: Path,
) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _semantic_content_is_atomic,
        _semantic_has_unbound_pronoun,
    )

    assert _semantic_content_is_atomic("服务读取 PostgreSQL 并写入 Redis。") is False
    assert _semantic_content_is_atomic("项目使用 PostgreSQL，并提供 Redis 缓存。") is False
    assert _semantic_content_is_atomic("项目支持 Windows、Linux 和 macOS。") is True
    assert _semantic_has_unbound_pronoun("团队已经采纳了它。") is True
    assert _semantic_has_unbound_pronoun("该团队采用它。") is True
    assert _semantic_has_unbound_pronoun("他们迁移了它。") is True

    cases = [
        (
            "zh-no-mark-question",
            "项目用什么数据库",
            "项目用什么数据库",
            "项目使用 PostgreSQL。",
        ),
        (
            "zh-no-mark-implementation",
            "这个功能怎么实现",
            "这个功能怎么实现",
            "该功能通过缓存实现。",
        ),
        (
            "en-no-mark-question",
            "What database should we use",
            "What database",
            "The project uses PostgreSQL.",
        ),
    ]
    for name, user_content, evidence, content in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type="domain_fact",
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject="database",
            value="postgresql",
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "skipped", round_payload
            assert _formal_memories(client, headers, "semantic-project") == []


def test_final_validation_single_predicate_and_nominal_ing_are_published(
    tmp_path: Path,
) -> None:
    cases = [
        (
            "single-predicate-noun-list",
            "The project stores logs and metrics in S3.",
            "The project stores logs and metrics in S3",
            "domain_fact",
        ),
        (
            "nominal-ing",
            "The deploy target is staging.",
            "The deploy target is staging",
            "decision",
        ),
    ]
    for name, user_content, content, memory_type in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=user_content,
            event_id=f"{name}-turn-user",
            subject="deployment detail",
            value="staging",
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "completed", round_payload
            assert [item["summary"] for item in _formal_memories(
                client, headers, "semantic-project"
            )] == [content]


def test_final_validation_compound_subjects_and_chinese_measure_words(
    tmp_path: Path,
) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _semantic_content_is_atomic,
        _semantic_has_unbound_pronoun,
    )

    assert _semantic_content_is_atomic("Requests and responses must be logged.") is True
    assert _semantic_content_is_atomic("Logs and metrics are written to S3 daily.") is True
    assert _semantic_content_is_atomic("Users and groups are stored in PostgreSQL.") is True
    assert _semantic_has_unbound_pronoun("这项政策要求使用不可变镜像。") is False
    assert _semantic_has_unbound_pronoun("这条约束必须保留。") is False
    assert _semantic_has_unbound_pronoun("这个服务必须使用 HTTPS。") is False

    cases = [
        (
            "compound-subject-modal",
            "Requests and responses must be logged.",
            "Requests and responses must be logged.",
            "constraint",
        ),
        (
            "compound-subject-passive",
            "Logs and metrics are written to S3 daily.",
            "Logs and metrics are written to S3 daily.",
            "constraint",
        ),
        (
            "zh-measure-word",
            "这项政策要求使用不可变镜像。",
            "这项政策要求使用不可变镜像。",
            "constraint",
        ),
    ]
    for name, user_content, content, memory_type in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=user_content.rstrip("。"),
            event_id=f"{name}-turn-user",
            subject="policy constraint",
            value="logged",
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "completed", round_payload
            assert [item["summary"] for item in _formal_memories(
                client, headers, "semantic-project"
            )] == [content]


def test_final_validation_round_nine_edges(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _semantic_content_is_atomic,
        _semantic_evidence_supports_content,
    )

    assert _semantic_evidence_supports_content(
        "The project uses PostgreSQL 16.", "PostgreSQL"
    ) is False
    assert _semantic_evidence_supports_content(
        "The project uses PostgreSQL.", "I do not use PostgreSQL."
    ) is False
    assert _semantic_evidence_supports_content(
        "The user prefers dark mode.", "I prefer dark mode."
    ) is True
    assert _semantic_evidence_supports_content(
        "The user does not prefer dark mode.", "I do not prefer dark mode."
    ) is True
    assert _semantic_evidence_supports_content(
        "Rust ownership prevents data races without a garbage collector.",
        "Rust ownership prevents data races",
    ) is True
    assert _semantic_content_is_atomic("The API caches responses and validates tokens.") is False
    assert _semantic_content_is_atomic("The API is fast and cheap.") is False
    assert _semantic_content_is_atomic("The deployment is fast, safe, and repeatable.") is False
    assert _semantic_content_is_atomic("The project is compatible with Windows and Linux.") is True
    assert _semantic_content_is_atomic("The worker fetches jobs and processes them.") is False
    assert _semantic_content_is_atomic(
        "The gateway authenticates requests and forwards traffic."
    ) is False
    assert _semantic_content_is_atomic("Requests and responses must be logged.") is True

    cases = [
        (
            "future-shift",
            "The team will shift traffic to the new cluster at midnight.",
            "The team will shift traffic to the new cluster at midnight.",
            "The team will shift traffic to the new cluster at midnight",
            "skipped",
        ),
        (
            "imperative",
            "Deploy the build after tests pass.",
            "Deploy the build after tests pass.",
            "Deploy the build after tests pass",
            "skipped",
        ),
        (
            "wh-statement",
            "What matters should be documented.",
            "What matters should be documented.",
            "What matters should be documented",
            "completed",
        ),
        (
            "mixed-request",
            "Could you use PostgreSQL? The project standard is PostgreSQL 16.",
            "The project standard is PostgreSQL 16.",
            "The project standard is PostgreSQL 16",
            "completed",
        ),
        (
            "above-average",
            "The service has above-average throughput.",
            "The service has above-average throughput.",
            "The service has above-average throughput",
            "completed",
        ),
        (
            "unsupported-evidence",
            "We need to choose a database.",
            "The project uses PostgreSQL 16.",
            "We need to choose a database",
            "skipped",
        ),
        (
            "unsupported-version",
            "I am considering PostgreSQL.",
            "The project uses PostgreSQL 16.",
            "PostgreSQL",
            "skipped",
        ),
        (
            "undecided-option",
            "I am considering PostgreSQL.",
            "The project uses PostgreSQL.",
            "PostgreSQL",
            "skipped",
        ),
        (
            "imperative-evidence",
            "Use PostgreSQL.",
            "The project uses PostgreSQL.",
            "Use PostgreSQL",
            "skipped",
        ),
        (
            "use-of-constraint",
            "Use of PostgreSQL is required.",
            "Use of PostgreSQL is required.",
            "Use of PostgreSQL is required",
            "completed",
        ),
        (
            "negated-evidence",
            "I do not use PostgreSQL.",
            "The project uses PostgreSQL.",
            "I do not use PostgreSQL",
            "skipped",
        ),
        (
            "command-output",
            "The command printed 42.",
            "The command printed 42.",
            "The command printed 42",
            "skipped",
        ),
    ]
    for name, user_content, content, evidence, expected in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type="domain_fact",
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject="project behavior",
            value="documented",
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == expected, round_payload
            memories = _formal_memories(client, headers, "semantic-project")
            assert len(memories) == (1 if expected == "completed" else 0)


def test_final_validation_round_nine_single_value_subjects(tmp_path: Path) -> None:
    cases = [
        ("primary-datastore", "primary datastore", "postgresql", "mysql"),
        ("deployment-cloud", "deployment cloud", "aws", "gcp"),
    ]
    for name, subject, first_value, second_value in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()

        def extractor(
            context: dict[str, object],
            *,
            subject: str = subject,
            first_value: str = first_value,
            second_value: str = second_value,
        ) -> list[dict[str, object]]:
            turn = str(context["turn_id"])
            value = first_value if turn.endswith("1") else second_value
            content = f"The {subject} is {value.upper()}."
            return [
                _memory(
                    content=content,
                    memory_type="decision",
                    evidence=content.rstrip("."),
                    event_id=f"{turn}-user",
                    subject=subject,
                    value=value,
                )
            ]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            for index, value in enumerate((first_value, second_value), start=1):
                turn_id = f"{name}-turn-{index}"
                content = f"The {subject} is {value.upper()}."
                _capture_turn(
                    client,
                    headers,
                    session_id=f"{name}-session",
                    turn_id=turn_id,
                    user_content=content,
                    assistant_content="Noted.",
                    cwd=root,
                    second=index * 10,
                )
                executor.drain()
                round_payload = _round(
                    client,
                    headers,
                    region_id="semantic-project",
                    session_id=f"{name}-session",
                    turn_id=turn_id,
                )
                assert round_payload["semantic_status"] == (
                    "completed" if index == 1 else "skipped"
                ), round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == 1


def test_final_validation_round_ten_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _semantic_content_is_atomic,
        _semantic_evidence_supports_content,
        _semantic_user_event_is_pure_question,
    )

    assert _semantic_evidence_supports_content(
        "The user uses Python for production services.",
        "I use Python for prototyping",
        claim_value="production services",
    ) is False
    assert _semantic_evidence_supports_content(
        "The user uses Python for production services.",
        "I use Python for prototyping",
    ) is False
    assert _semantic_content_is_atomic(
        "The user prefers dark mode as well as weekly dependency audits."
    ) is False
    assert _semantic_content_is_atomic("The API is fast as well as reliable.") is False
    assert (
        _semantic_user_event_is_pure_question(
            "What should happen is documented in the runbook."
        )
        is False
    )

    cases = [
        (
            "evidence-inference",
            "I use Python for prototyping.",
            "The user uses Python for production services.",
            "I use Python for prototyping",
            "domain_fact",
            "user python usage",
            "production services",
            "skipped",
        ),
        (
            "atomic-as-well-as",
            "I prefer dark mode as well as weekly dependency audits.",
            "The user prefers dark mode as well as weekly dependency audits.",
            "I prefer dark mode as well as weekly dependency audits",
            "preference",
            "user:preference:profile",
            "dark mode as well as weekly dependency audits",
            "skipped",
        ),
        (
            "dependent-former",
            "The team discussed security reviews. The former option requires weekly audits.",
            "The former option requires weekly audits.",
            "The former option requires weekly audits",
            "constraint",
            "audit policy",
            "weekly",
            "skipped",
        ),
        (
            "plan-q1",
            "We plan the rollout in Q1.",
            "We plan the rollout in Q1.",
            "We plan the rollout in Q1",
            "decision",
            "rollout timing",
            "q1",
            "skipped",
        ),
        (
            "imperative-containerize",
            "Containerize the service with Go.",
            "The service is containerized with Go.",
            "Containerize the service with Go",
            "decision",
            "service implementation",
            "go",
            "skipped",
        ),
        (
            "tool-failure-output",
            "FAILED tests/test_foo.py - AssertionError: expected 1",
            "The test_foo test fails with an AssertionError.",
            "FAILED tests/test_foo.py - AssertionError",
            "reusable_experience",
            "test_foo result",
            "assertionerror",
            "skipped",
        ),
        (
            "wh-statement",
            "What should happen is documented in the runbook.",
            "The deployment process is documented in the runbook.",
            "What should happen is documented in the runbook",
            "domain_fact",
            "deployment documentation",
            "runbook",
            "completed",
        ),
    ]
    for (
        name,
        user_content,
        content,
        evidence,
        memory_type,
        subject,
        value,
        expected,
    ) in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == expected, round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == (
                1 if expected == "completed" else 0
            )

    state_dir = tmp_path / "conflict-unlisted" / "state"
    root = tmp_path / "conflict-unlisted" / "project"
    root.parent.mkdir()
    executor = ManualSemanticExecutor()

    def conflict_extractor(context: dict[str, object]) -> list[dict[str, object]]:
        turn = str(context["turn_id"])
        value = "node" if turn.endswith("1") else "python"
        content = f"The primary runtime is {value.title()}."
        return [
            _memory(
                content=content,
                memory_type="decision",
                evidence=content.rstrip("."),
                event_id=f"{turn}-user",
                subject="primary runtime",
                value=value,
            )
        ]

    with TestClient(_app(state_dir, conflict_extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        statuses: list[str] = []
        for index, value in ((1, "Node"), (2, "Python")):
            turn_id = f"unlisted-conflict-{index}"
            content = f"The primary runtime is {value}."
            _capture_turn(
                client,
                headers,
                session_id="unlisted-conflict-session",
                turn_id=turn_id,
                user_content=content,
                assistant_content="Noted.",
                cwd=root,
                second=index * 10,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id="unlisted-conflict-session",
                turn_id=turn_id,
            )
            statuses.append(str(round_payload["semantic_status"]))
        assert statuses == ["completed", "skipped"]
        assert [
            item["summary"]
            for item in _formal_memories(client, headers, "semantic-project")
        ] == ["The primary runtime is Node."]


def test_final_version_model_failure_retires_previous_memory(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        assistant = [
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "assistant"
        ][-1]
        if "Final conclusion" in str(assistant["content"]):
            raise RuntimeError("final model unavailable")
        return [
            _memory(
                content="The user prefers the old deployment plan.",
                memory_type="preference",
                evidence="I prefer the old deployment plan",
                event_id="version-failure-user",
                subject="user:preference:deployment",
                value="old deployment plan",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture(
            client,
            headers,
            event_id="version-failure-user",
            session_id="version-failure-session",
            turn_id="version-failure-turn",
            kind="user",
            content="I prefer the old deployment plan.",
            cwd=root,
            occurred_at="2026-10-28T05:00:00Z",
        )
        _capture(
            client,
            headers,
            event_id="version-failure-old",
            session_id="version-failure-session",
            turn_id="version-failure-turn",
            kind="assistant",
            content="Old conclusion.",
            cwd=root,
            occurred_at="2026-10-28T05:00:01Z",
        )
        executor.drain()
        first = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="version-failure-session",
            turn_id="version-failure-turn",
        )
        assert first["semantic_status"] == "completed"
        assert len(_formal_memories(client, headers, "semantic-project")) == 1

        _capture(
            client,
            headers,
            event_id="version-failure-final",
            session_id="version-failure-session",
            turn_id="version-failure-turn",
            kind="assistant",
            content="Final conclusion.",
            cwd=root,
            occurred_at="2026-10-28T05:00:02Z",
        )
        executor.drain()
        final = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="version-failure-session",
            turn_id="version-failure-turn",
        )
        assert final["semantic_status"] == "failed"
        assert "final model unavailable" in str(final["semantic_error"])
        assert _formal_memories(client, headers, "semantic-project") == []


def test_final_validation_round_eleven_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _promotion_values_conflict,
        _semantic_content_is_atomic,
        _semantic_evidence_supports_content,
        _semantic_user_event_is_pure_question,
    )

    assert _semantic_evidence_supports_content(
        "The user uses Python during production.",
        "I use Python for prototyping",
    ) is False
    assert _semantic_evidence_supports_content(
        "The project uses PostgreSQL 16.",
        "We use Postgres 16",
    ) is True
    assert _semantic_content_is_atomic("The API is fast but reliable.") is False
    assert _semantic_content_is_atomic("The service is secure yet slow.") is False
    assert (
        _semantic_user_event_is_pure_question("Which database is used in production")
        is True
    )
    assert _promotion_values_conflict(
        "deployment environment", ("staging", "production")
    ) is True

    cases = [
        (
            "evidence-during-production",
            "I use Python for prototyping.",
            "The user uses Python during production.",
            "I use Python for prototyping",
            "domain_fact",
            None,
            None,
            "skipped",
        ),
        (
            "previous-option",
            "The team discussed security reviews. The previous option requires weekly audits.",
            "The previous option requires weekly audits.",
            "The previous option requires weekly audits",
            "constraint",
            "audit policy",
            "weekly",
            "skipped",
        ),
        (
            "plan-january",
            "The rollout starts in January.",
            "The rollout starts in January.",
            "The rollout starts in January",
            "decision",
            "rollout timing",
            "january",
            "skipped",
        ),
        (
            "convert-imperative",
            "Convert the service to Go.",
            "The service is converted to Go.",
            "Convert the service to Go",
            "decision",
            "service implementation",
            "go",
            "skipped",
        ),
        (
            "passed-test-output",
            "PASSED tests/test_foo.py",
            "The test_foo test passed.",
            "PASSED tests/test_foo.py",
            "reusable_experience",
            "test_foo result",
            "passed",
            "skipped",
        ),
        (
            "wh-no-mark-question",
            "Which database is used in production",
            "Which database is used in production.",
            "Which database is used in production",
            "domain_fact",
            "production database",
            "unknown",
            "skipped",
        ),
        (
            "plan-noun",
            "The plan the team approved is documented in the runbook.",
            "The plan the team approved is documented in the runbook.",
            "The plan the team approved is documented in the runbook",
            "domain_fact",
            "team plan",
            "documented in the runbook",
            "completed",
        ),
        (
            "permission-may",
            "Admins may access the dashboard.",
            "Admins may access the dashboard.",
            "Admins may access the dashboard",
            "constraint",
            "dashboard access",
            "allowed",
            "completed",
        ),
        (
            "plan-noun-chinese",
            "项目计划已记录在文档中。",
            "项目计划已记录在文档中。",
            "项目计划已记录在文档中",
            "domain_fact",
            "项目计划",
            "已记录在文档中",
            "completed",
        ),
        (
            "postgres-alias",
            "We use Postgres 16.",
            "The project uses PostgreSQL 16.",
            "We use Postgres 16",
            "domain_fact",
            "database version",
            "16",
            "completed",
        ),
    ]
    for (
        name,
        user_content,
        content,
        evidence,
        memory_type,
        subject,
        value,
        expected,
    ) in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == expected, round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == (
                1 if expected == "completed" else 0
            )

    state_dir = tmp_path / "environment-conflict" / "state"
    root = tmp_path / "environment-conflict" / "project"
    root.parent.mkdir()
    executor = ManualSemanticExecutor()

    def conflict_extractor(context: dict[str, object]) -> list[dict[str, object]]:
        turn = str(context["turn_id"])
        value = "staging" if turn.endswith("1") else "production"
        content = f"The deployment environment is {value.title()}."
        return [
            _memory(
                content=content,
                memory_type="decision",
                evidence=content.rstrip("."),
                event_id=f"{turn}-user",
                subject="deployment environment",
                value=value,
            )
        ]

    with TestClient(_app(state_dir, conflict_extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        statuses: list[str] = []
        for index, value in ((1, "Staging"), (2, "Production")):
            turn_id = f"environment-conflict-{index}"
            content = f"The deployment environment is {value}."
            _capture_turn(
                client,
                headers,
                session_id="environment-conflict-session",
                turn_id=turn_id,
                user_content=content,
                assistant_content="Noted.",
                cwd=root,
                second=index * 10,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id="environment-conflict-session",
                turn_id=turn_id,
            )
            statuses.append(str(round_payload["semantic_status"]))
        assert statuses == ["completed", "skipped"]
        assert [
            item["summary"]
            for item in _formal_memories(client, headers, "semantic-project")
        ] == ["The deployment environment is Staging."]


def test_final_validation_round_twelve_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _SEMANTIC_NON_SEMANTIC_OUTPUT,
        _SEMANTIC_TRANSIENT,
        _promotion_values_conflict,
        _semantic_content_is_atomic,
        _semantic_evidence_supports_content,
        _semantic_user_event_is_pure_question,
    )

    for content in (
        "The API is fast whereas the database is slow.",
        "The API is fast while the database is slow.",
        "The API is fast although the database is old.",
    ):
        assert _semantic_content_is_atomic(content) is False
    for question in (
        "Which team owns the service",
        "Who maintains the billing service",
        "Which database stores the audit logs",
    ):
        assert _semantic_user_event_is_pure_question(question) is True
    assert _semantic_user_event_is_pure_question("May 2024 was the launch month.") is False
    assert _SEMANTIC_TRANSIENT.search("The migration will take place in January.")
    assert _SEMANTIC_TRANSIENT.search(
        "The plan to migrate to PostgreSQL is documented in the runbook."
    ) is None
    assert _semantic_evidence_supports_content(
        "The meeting is by noon.", "The meeting is at noon"
    ) is False
    assert _semantic_evidence_supports_content(
        "The user prefers JavaScript for browser automation.",
        "I prefer TypeScript for browser automation",
    ) is False
    assert _semantic_evidence_supports_content(
        "The project uses MongoDB.", "We use Mongo"
    ) is True
    assert _SEMANTIC_NON_SEMANTIC_OUTPUT.search("PASSED (0.12s) tests/test_foo.py")
    assert _SEMANTIC_NON_SEMANTIC_OUTPUT.search("PASSED: tests/test_foo.py")
    assert _promotion_values_conflict(
        "production API URL",
        ("https://one.example", "https://two.example"),
    ) is True

    cases = [
        (
            "whereas-atomic",
            "The API is fast whereas the database is slow.",
            "The API is fast whereas the database is slow.",
            "The API is fast whereas the database is slow",
            "domain_fact",
            "api comparison",
            "database slower",
            "skipped",
        ),
        (
            "previous-iteration",
            "The previous iteration used Vue.",
            "The previous iteration used Vue.",
            "The previous iteration used Vue",
            "domain_fact",
            "previous frontend",
            "vue",
            "skipped",
        ),
        (
            "active-wh-question",
            "Which team owns the service",
            "The team owns the service.",
            "Which team owns the service",
            "domain_fact",
            "service ownership",
            "team",
            "skipped",
        ),
        (
            "will-take-place",
            "The migration will take place in January.",
            "The migration will take place in January.",
            "The migration will take place in January",
            "decision",
            "migration timing",
            "january",
            "skipped",
        ),
        (
            "meeting-by-noon",
            "The meeting is at noon.",
            "The meeting is by noon.",
            "The meeting is at noon",
            "constraint",
            "meeting deadline",
            "noon",
            "skipped",
        ),
        (
            "javascript-substitution",
            "I prefer TypeScript for browser automation.",
            "The user prefers JavaScript for browser automation.",
            "I prefer TypeScript for browser automation",
            "preference",
            "user:preference:language",
            "browser automation",
            "skipped",
        ),
        (
            "passed-elapsed-output",
            "PASSED (0.12s) tests/test_foo.py",
            "The tests/test_foo.py verification succeeded.",
            "PASSED (0.12s) tests/test_foo.py",
            "reusable_experience",
            "test_foo result",
            "passed",
            "skipped",
        ),
        (
            "may-launch-month",
            "May 2024 was the launch month.",
            "May 2024 was the launch month.",
            "May 2024 was the launch month",
            "domain_fact",
            "launch month",
            "may 2024",
            "completed",
        ),
        (
            "plan-noun-to",
            "The plan to migrate to PostgreSQL is documented in the runbook.",
            "The plan to migrate to PostgreSQL is documented in the runbook.",
            "The plan to migrate to PostgreSQL is documented in the runbook",
            "domain_fact",
            "migration plan",
            "documented in the runbook",
            "completed",
        ),
        (
            "mongo-alias",
            "We use Mongo.",
            "The project uses MongoDB.",
            "We use Mongo",
            "domain_fact",
            "database",
            "mongodb",
            "completed",
        ),
    ]
    for (
        name,
        user_content,
        content,
        evidence,
        memory_type,
        subject,
        value,
        expected,
    ) in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == expected, round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == (
                1 if expected == "completed" else 0
            )

    state_dir = tmp_path / "api-url-conflict" / "state"
    root = tmp_path / "api-url-conflict" / "project"
    root.parent.mkdir()
    executor = ManualSemanticExecutor()

    def conflict_extractor(context: dict[str, object]) -> list[dict[str, object]]:
        turn = str(context["turn_id"])
        value = "https://one.example" if turn.endswith("1") else "https://two.example"
        content = f"The production API URL is {value}."
        return [
            _memory(
                content=content,
                memory_type="decision",
                evidence=content.rstrip("."),
                event_id=f"{turn}-user",
                subject="production API URL",
                value=value,
            )
        ]

    with TestClient(_app(state_dir, conflict_extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        statuses: list[str] = []
        for index, value in (
            (1, "https://one.example"),
            (2, "https://two.example"),
        ):
            turn_id = f"api-url-conflict-{index}"
            content = f"The production API URL is {value}."
            _capture_turn(
                client,
                headers,
                session_id="api-url-conflict-session",
                turn_id=turn_id,
                user_content=content,
                assistant_content="Noted.",
                cwd=root,
                second=index * 10,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id="api-url-conflict-session",
                turn_id=turn_id,
            )
            statuses.append(str(round_payload["semantic_status"]))
        assert statuses == ["completed", "skipped"]
        assert [
            item["summary"]
            for item in _formal_memories(client, headers, "semantic-project")
        ] == ["The production API URL is https://one.example."]


def test_manual_overwrite_survives_later_semantic_version(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        assistant = [
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "assistant"
        ][-1]
        if "Final conclusion" in str(assistant["content"]):
            return []
        return [
            _memory(
                content="The user prefers dark mode.",
                memory_type="preference",
                evidence="I prefer dark mode",
                event_id="manual-overwrite-user",
                subject="user:preference:theme",
                value="dark",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture(
            client,
            headers,
            event_id="manual-overwrite-user",
            session_id="manual-overwrite-session",
            turn_id="manual-overwrite-turn",
            kind="user",
            content="I prefer dark mode.",
            cwd=root,
            occurred_at="2026-10-28T06:00:00Z",
        )
        _capture(
            client,
            headers,
            event_id="manual-overwrite-old-assistant",
            session_id="manual-overwrite-session",
            turn_id="manual-overwrite-turn",
            kind="assistant",
            content="Old conclusion.",
            cwd=root,
            occurred_at="2026-10-28T06:00:01Z",
        )
        executor.drain()
        created = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="manual-overwrite-session",
            turn_id="manual-overwrite-turn",
        )
        memory_id = str(created["semantic_memory_ids"][0])
        detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{memory_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        updated = client.put(
            "/api/v1/libraries/semantic-project/document",
            headers=headers,
            json={
                "path": memory_id,
                "content": "The user prefers light mode.",
                "expected_source_version": detail.json()["source_version"],
                "operation_id": "manual-overwrite-update",
                "actor_type": "user",
                "source": "manual-test",
                "mode": "overwrite",
            },
        )
        assert updated.status_code == 200, updated.text
        updated_id = str(updated.json()["id"])

        _capture(
            client,
            headers,
            event_id="manual-overwrite-final-assistant",
            session_id="manual-overwrite-session",
            turn_id="manual-overwrite-turn",
            kind="assistant",
            content="Final conclusion.",
            cwd=root,
            occurred_at="2026-10-28T06:00:02Z",
        )
        executor.drain()
        final = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="manual-overwrite-session",
            turn_id="manual-overwrite-turn",
        )
        assert final["semantic_status"] == "skipped"
        preserved = client.get(
            f"/api/v1/libraries/semantic-project/memories/{updated_id}",
            headers=headers,
        )
        assert preserved.status_code == 200, preserved.text
        assert preserved.json()["content"] == "The user prefers light mode."


def test_manual_supersede_reuses_equivalent_later_semantic_claim(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user = next(
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "user"
        )
        if user["event_id"] == "manual-supersede-dark-user":
            content = "The user prefers dark mode."
            value = "dark"
        else:
            content = "The user prefers light mode."
            value = "light"
        return [
            _memory(
                content=content,
                memory_type="preference",
                evidence=str(user["content"]),
                event_id=str(user["event_id"]),
                subject="user:preference:theme",
                value=value,
                region_hint="user",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture_turn(
            client,
            headers,
            session_id="manual-supersede-session",
            turn_id="manual-supersede-dark",
            user_content="I prefer dark mode.",
            assistant_content="Noted.",
            cwd=root,
        )
        executor.drain()
        first = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="manual-supersede-session",
            turn_id="manual-supersede-dark",
        )
        original_id = str(first["semantic_memory_ids"][0])
        original = client.get(
            f"/api/v1/libraries/user/memories/{original_id}",
            headers=headers,
        )
        assert original.status_code == 200, original.text
        corrected = client.put(
            "/api/v1/libraries/user/document",
            headers=headers,
            json={
                "path": original_id,
                "content": "The user prefers light mode.",
                "expected_source_version": original.json()["source_version"],
                "operation_id": "manual-supersede-light",
                "actor_type": "user",
                "source": "manual-supersede-test",
                "mode": "supersede",
            },
        )
        assert corrected.status_code == 200, corrected.text
        corrected_id = str(corrected.json()["id"])
        assert corrected_id != original_id

        _capture_turn(
            client,
            headers,
            session_id="manual-supersede-session",
            turn_id="manual-supersede-light",
            user_content="I prefer light mode.",
            assistant_content="Noted again.",
            cwd=root,
            second=10,
        )
        executor.drain()
        second = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="manual-supersede-session",
            turn_id="manual-supersede-light",
        )

        assert second["semantic_status"] == "completed", second
        assert second["semantic_memory_ids"] == [corrected_id]
        active = [
            item
            for item in _formal_memories(client, headers, "user")
            if item["lifecycle"] == "active"
        ]
        assert [item["id"] for item in active] == [corrected_id]
        detail = client.get(
            f"/api/v1/libraries/user/memories/{corrected_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        assert payload["content"] == "The user prefers light mode."
        assert len(payload["user_metadata"]["source_unit_ids"]) >= 4
        assert set(payload["user_metadata"]["source_event_ids"]) >= {
            "manual-supersede-dark-user",
            "manual-supersede-light-user",
        }
        assert len(payload["semantic_versions"]) == 2
        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            conflict_count = connection.execute(
                """SELECT COUNT(*) FROM governance_events
                   WHERE actor = 'semantic-distillation' AND action = 'conflict'"""
            ).fetchone()[0]
        assert conflict_count == 0


def test_queue_saturation_retries_final_version_after_capacity_returns(
    tmp_path: Path,
) -> None:
    class CapacitySemanticExecutor:
        def __init__(self) -> None:
            self._tasks: deque[Callable[[], None]] = deque()
            self._lock = threading.Lock()
            self._slot_callback: Callable[[], None] | None = None
            self.full = False

        @property
        def worker_count(self) -> int:
            return 1

        @property
        def pending_count(self) -> int:
            with self._lock:
                return len(self._tasks)

        def set_slot_callback(self, callback: Callable[[], None]) -> None:
            self._slot_callback = callback

        def submit(
            self,
            task_key: str,
            region_id: str,
            task: Callable[[], None],
        ) -> bool:
            del task_key, region_id
            with self._lock:
                if self.full:
                    return False
                self._tasks.append(task)
            return True

        def drain(self) -> None:
            while self.pending_count:
                with self._lock:
                    task = self._tasks.popleft()
                task()

        def release_slot(self) -> None:
            self.full = False
            if self._slot_callback is not None:
                self._slot_callback()

        def close(self) -> None:
            self.drain()

    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = CapacitySemanticExecutor()
    calls = 0

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        messages = context["messages"]
        assert isinstance(messages, list)
        assistant = [
            item
            for item in messages
            if isinstance(item, dict) and item["role"] == "assistant"
        ][-1]
        if "Final conclusion" in str(assistant["content"]):
            return [
                _memory(
                    content="The user prefers the final deployment plan.",
                    memory_type="preference",
                    evidence="I prefer the final deployment plan",
                    event_id="queue-user",
                    subject="user:preference:deployment",
                    value="final deployment plan",
                )
            ]
        return [
            _memory(
                content="The user prefers the old deployment plan.",
                memory_type="preference",
                evidence="I prefer the old deployment plan",
                event_id="queue-user",
                subject="user:preference:deployment",
                value="old deployment plan",
            )
        ]

    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    app = create_app(Settings(state_dir=state_dir), memory_kernel=kernel)
    with TestClient(app) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture(
            client,
            headers,
            event_id="queue-user",
            session_id="queue-session",
            turn_id="queue-turn",
            kind="user",
            content=(
                "I prefer the old deployment plan. "
                "I prefer the final deployment plan."
            ),
            cwd=root,
            occurred_at="2026-10-28T06:10:00Z",
        )
        _capture(
            client,
            headers,
            event_id="queue-old-assistant",
            session_id="queue-session",
            turn_id="queue-turn",
            kind="assistant",
            content="Old conclusion.",
            cwd=root,
            occurred_at="2026-10-28T06:10:01Z",
        )
        executor.drain()
        old = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="queue-session",
            turn_id="queue-turn",
        )
        assert old["semantic_status"] == "completed"

        executor.full = True
        asyncio.run(
            kernel.write(
                CaptureMemoryRequest(
                    event_id="queue-final-assistant",
                    session_id="queue-session",
                    turn_id="queue-turn",
                    event_kind="assistant",
                    content="Final conclusion.",
                    occurred_at="2026-10-28T06:10:02Z",
                    cwd=str(root),
                )
            )
        )
        queued = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="queue-session",
            turn_id="queue-turn",
        )
        assert queued["semantic_status"] == "failed"
        assert "queue is full" in str(queued["semantic_error"])
        assert _formal_memories(client, headers, "semantic-project") == []

        executor.release_slot()
        executor.drain()
        completed = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="queue-session",
            turn_id="queue-turn",
        )
        assert completed["semantic_status"] == "completed"
        assert calls == 2
        assert [
            item["summary"]
            for item in _formal_memories(client, headers, "semantic-project")
        ] == ["The user prefers the final deployment plan."]


def test_final_validation_round_thirteen_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _SEMANTIC_TRANSIENT,
        _semantic_content_is_atomic,
        _semantic_evidence_supports_content,
        _semantic_text_is_imperative,
        _semantic_user_event_is_pure_question,
    )

    assert _semantic_evidence_supports_content(
        "The user prefers light mode for the editor.",
        "I prefer dark mode for the editor",
    ) is False
    assert _semantic_content_is_atomic(
        "The user prefers dark mode: the project uses PostgreSQL."
    ) is False
    assert _semantic_content_is_atomic(
        "The user prefers dark mode — the project uses PostgreSQL."
    ) is False
    assert _semantic_content_is_atomic(
        "The user prefers dark mode\nthe project uses PostgreSQL."
    ) is False
    assert _semantic_content_is_atomic("The user prefers while loops for polling.") is True
    assert _semantic_content_is_atomic(
        "The user works at Acme Inc. on API design."
    ) is True
    assert _semantic_user_event_is_pure_question("Why current architecture fails") is True
    assert _semantic_text_is_imperative("Review current architecture") is True
    assert _SEMANTIC_TRANSIENT.search("The migration will occur next year.")

    cases = [
        (
            "opposite-preference",
            "I prefer dark mode for the editor.",
            "The user prefers light mode for the editor.",
            "I prefer dark mode for the editor",
            "preference",
            "user:preference:editor",
            "light mode",
            "skipped",
        ),
        (
            "colon-multiple-claims",
            "I prefer dark mode: the project uses PostgreSQL.",
            "The user prefers dark mode: the project uses PostgreSQL.",
            "I prefer dark mode: the project uses PostgreSQL",
            "preference",
            "user:preference:profile",
            "dark mode",
            "skipped",
        ),
        (
            "adjective-wh-question",
            "Why current architecture fails",
            "The current architecture fails.",
            "Why current architecture fails",
            "domain_fact",
            "current architecture",
            "fails",
            "skipped",
        ),
        (
            "review-request",
            "Review current architecture",
            "The current architecture requires review.",
            "Review current architecture",
            "reusable_experience",
            "current architecture",
            "review",
            "skipped",
        ),
        (
            "as-described-before",
            "As described before, the API uses PostgreSQL.",
            "As described before, the API uses PostgreSQL.",
            "As described before, the API uses PostgreSQL",
            "domain_fact",
            "API database",
            "postgresql",
            "skipped",
        ),
        (
            "zh-before-description",
            "依照之前说明，该服务使用 PostgreSQL。",
            "依照之前说明，该服务使用 PostgreSQL。",
            "依照之前说明，该服务使用 PostgreSQL",
            "domain_fact",
            "服务数据库",
            "postgresql",
            "skipped",
        ),
        (
            "future-next-year",
            "The migration will occur next year.",
            "The migration will occur next year.",
            "The migration will occur next year",
            "decision",
            "migration timing",
            "next year",
            "skipped",
        ),
        (
            "while-loop",
            "I prefer while loops for polling.",
            "The user prefers while loops for polling.",
            "I prefer while loops for polling",
            "preference",
            "user:preference:polling",
            "while loops",
            "completed",
        ),
        (
            "acme-abbreviation",
            "I work at Acme Inc. on API design.",
            "The user works at Acme Inc. on API design.",
            "I work at Acme Inc. on API design",
            "domain_fact",
            "user employment",
            "API design",
            "completed",
        ),
    ]
    for (
        name,
        user_content,
        content,
        evidence,
        memory_type,
        subject,
        value,
        expected,
    ) in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == expected, round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == (
                1 if expected == "completed" else 0
            )


def test_final_validation_round_fourteen_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _SEMANTIC_DEICTIC_REFERENCE,
        _SEMANTIC_TRANSIENT,
        _semantic_content_is_atomic,
        _semantic_evidence_supports_content,
        _semantic_text_is_imperative,
        _semantic_user_event_is_pure_question,
    )

    assert _semantic_content_is_atomic(
        "The user prefers dark mode:the project uses PostgreSQL."
    ) is False
    assert _semantic_content_is_atomic(
        "The user prefers dark mode—the project uses PostgreSQL."
    ) is False
    assert _semantic_content_is_atomic(
        "The user prefers dark mode–the project uses PostgreSQL."
    ) is False
    assert _semantic_content_is_atomic(
        "The endpoint is https://example.com/api."
    ) is True
    assert _semantic_content_is_atomic("The meeting starts at 10:30.") is True
    assert _semantic_content_is_atomic(
        "The supported range is 10–20 requests per second."
    ) is True
    assert _semantic_evidence_supports_content(
        "The user prefers dark mode at night.",
        "I prefer dark mode",
    ) is False
    assert _semantic_evidence_supports_content(
        "The user prefers dark mode nightly.",
        "I prefer dark mode",
    ) is False
    assert _semantic_evidence_supports_content(
        "The user prefers javascript for backend services.",
        "I prefer TypeScript for backend services",
    ) is False
    assert _semantic_evidence_supports_content(
        "Rust ownership prevents data races without a garbage collector.",
        "Rust ownership prevents data races",
    ) is True
    assert _semantic_user_event_is_pure_question("Where logs are stored") is True
    assert _semantic_text_is_imperative("Explain caching") is True
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "As previously stated, the API uses PostgreSQL."
    )
    assert _SEMANTIC_TRANSIENT.search("The migration is planned soon.")

    cases = [
        (
            "colon-no-space",
            "I prefer dark mode:the project uses PostgreSQL.",
            "The user prefers dark mode:the project uses PostgreSQL.",
            "I prefer dark mode:the project uses PostgreSQL",
            "preference",
            "user:preference:theme",
            "dark mode",
        ),
        (
            "emdash-no-space",
            "I prefer dark mode—the project uses PostgreSQL.",
            "The user prefers dark mode—the project uses PostgreSQL.",
            "I prefer dark mode—the project uses PostgreSQL",
            "preference",
            "user:preference:theme",
            "dark mode",
        ),
        (
            "where-passive-question",
            "Where logs are stored",
            "The logs are stored locally.",
            "Where logs are stored",
            "domain_fact",
            "log storage",
            "locally",
        ),
        (
            "bare-explain-request",
            "Explain caching",
            "The caching policy is static.",
            "Explain caching",
            "domain_fact",
            "caching policy",
            "static",
        ),
        (
            "as-previously-stated",
            "As previously stated, the API uses PostgreSQL.",
            "As previously stated, the API uses PostgreSQL.",
            "As previously stated, the API uses PostgreSQL",
            "domain_fact",
            "API database",
            "postgresql",
        ),
        (
            "planned-soon",
            "The migration is planned soon.",
            "The migration is planned soon.",
            "The migration is planned soon",
            "decision",
            "migration timing",
            "soon",
        ),
        (
            "unsupported-temporal-qualifier",
            "I prefer dark mode.",
            "The user prefers dark mode at night.",
            "I prefer dark mode",
            "preference",
            "user:preference:theme",
            "dark mode at night",
        ),
        (
            "lowercase-entity-substitution",
            "I prefer TypeScript for backend services.",
            "The user prefers javascript for backend services.",
            "I prefer TypeScript for backend services",
            "preference",
            "user:preference:language",
            "javascript",
        ),
    ]
    for name, user_content, content, evidence, memory_type, subject, value in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "skipped", round_payload
            assert _formal_memories(client, headers, "semantic-project") == []


def test_manual_overwrite_clears_semantic_claim_metadata(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    executor = ManualSemanticExecutor()

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        del context
        return [
            _memory(
                content="The user prefers dark mode.",
                memory_type="preference",
                evidence="I prefer dark mode",
                event_id="ownership-user",
                subject="user:preference:theme",
                value="dark",
            )
        ]

    with TestClient(_app(state_dir, extractor, executor)) as client:
        headers = _headers(state_dir)
        _prepare_project(client, headers, root)
        _capture(
            client,
            headers,
            event_id="ownership-user",
            session_id="ownership-session",
            turn_id="ownership-turn",
            kind="user",
            content="I prefer dark mode.",
            cwd=root,
            occurred_at="2026-10-28T07:00:00Z",
        )
        _capture(
            client,
            headers,
            event_id="ownership-old-assistant",
            session_id="ownership-session",
            turn_id="ownership-turn",
            kind="assistant",
            content="Old conclusion.",
            cwd=root,
            occurred_at="2026-10-28T07:00:01Z",
        )
        executor.drain()
        created = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="ownership-session",
            turn_id="ownership-turn",
        )
        memory_id = str(created["semantic_memory_ids"][0])
        detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{memory_id}",
            headers=headers,
        )
        assert detail.status_code == 200, detail.text
        updated = client.put(
            "/api/v1/libraries/semantic-project/document",
            headers=headers,
            json={
                "path": memory_id,
                "content": "The user prefers light mode.",
                "expected_source_version": detail.json()["source_version"],
                "operation_id": "ownership-manual-light",
                "actor_type": "user",
                "source": "manual-ownership-test",
                "mode": "overwrite",
            },
        )
        assert updated.status_code == 200, updated.text
        updated_id = str(updated.json()["id"])
        updated_detail = client.get(
            f"/api/v1/libraries/semantic-project/memories/{updated_id}",
            headers=headers,
        )
        assert updated_detail.status_code == 200, updated_detail.text
        updated_metadata = updated_detail.json()["user_metadata"]
        assert updated_metadata["semantic_manual_override"] is True
        for key in (
            "semantic_owned",
            "semantic_key",
            "semantic_source_fingerprint",
            "claim_subject",
            "claim_value",
            "user_event_id",
            "evidence_snippet",
        ):
            assert key not in updated_metadata

        _capture(
            client,
            headers,
            event_id="ownership-final-assistant",
            session_id="ownership-session",
            turn_id="ownership-turn",
            kind="assistant",
            content="Final conclusion.",
            cwd=root,
            occurred_at="2026-10-28T07:00:02Z",
        )
        executor.drain()
        final = _round(
            client,
            headers,
            region_id="semantic-project",
            session_id="ownership-session",
            turn_id="ownership-turn",
        )
        assert final["semantic_status"] == "skipped", final
        assert final["semantic_memory_ids"] == []
        assert all(
            item["status"] != "reused" for item in final["semantic_diagnostics"]
        )
        memories = _formal_memories(client, headers, "semantic-project")
        assert [item["id"] for item in memories] == [updated_id]
        preserved = client.get(
            f"/api/v1/libraries/semantic-project/memories/{updated_id}",
            headers=headers,
        )
        assert preserved.status_code == 200, preserved.text
        payload = preserved.json()
        assert payload["content"] == "The user prefers light mode."
        assert payload["user_metadata"]["governance_actor"] == "user"
        assert payload["user_metadata"]["semantic_manual_override"] is True
        assert "claim_value" not in payload["user_metadata"]


def test_final_validation_round_fifteen_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _semantic_evidence_supports_content,
        _semantic_text_is_imperative,
        _semantic_user_event_is_pure_request,
    )

    assert _semantic_text_is_imperative(
        "Tell us how the API authenticates users."
    ) is True
    assert _semantic_user_event_is_pure_request(
        "Tell us how the API authenticates users."
    ) is True
    for content in (
        "The user always prefers dark mode.",
        "The user prefers only dark mode.",
        "The user prefers dark mode exclusively.",
        "The user still prefers dark mode.",
        "The user really prefers dark mode.",
        "The user just prefers dark mode.",
    ):
        assert _semantic_evidence_supports_content(
            content,
            "I prefer dark mode",
        ) is False
    assert _semantic_evidence_supports_content(
        "The user only prefers dark mode.",
        "I prefer only dark mode",
    ) is True

    cases = [
        (
            "tell-us-request",
            "Tell us how the API authenticates users.",
            "The API authenticates users.",
            "Tell us how the API authenticates users",
            "domain_fact",
            "api authentication",
            "users",
            "skipped",
        ),
        (
            "unsupported-always",
            "I prefer dark mode.",
            "The user always prefers dark mode.",
            "I prefer dark mode",
            "preference",
            "user:preference:theme",
            "dark mode",
            "skipped",
        ),
        (
            "unsupported-only",
            "I prefer dark mode.",
            "The user prefers only dark mode.",
            "I prefer dark mode",
            "preference",
            "user:preference:theme",
            "dark mode",
            "skipped",
        ),
        (
            "unsupported-exclusively",
            "I prefer dark mode.",
            "The user prefers dark mode exclusively.",
            "I prefer dark mode",
            "preference",
            "user:preference:theme",
            "dark mode",
            "skipped",
        ),
        (
            "supported-only",
            "I prefer only dark mode.",
            "The user only prefers dark mode.",
            "I prefer only dark mode",
            "preference",
            "user:preference:theme",
            "dark mode",
            "completed",
        ),
    ]
    for (
        name,
        user_content,
        content,
        evidence,
        memory_type,
        subject,
        value,
        expected,
    ) in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == expected, round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == (
                1 if expected == "completed" else 0
            )


def test_final_validation_round_sixteen_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _semantic_text_is_question,
        _semantic_user_event_is_pure_question,
    )

    assert _semantic_user_event_is_pure_question(
        "Where our application access logs are stored."
    ) is True
    assert _semantic_user_event_is_pure_question(
        "When our nightly backup verification jobs are completed."
    ) is True
    assert _semantic_user_event_is_pure_question(
        "Where our primary application access audit records are stored."
    ) is True
    assert _semantic_user_event_is_pure_question(
        "How we deploy is documented in the runbook."
    ) is False
    assert _semantic_user_event_is_pure_question(
        "How the team deploys is documented in the runbook."
    ) is False
    assert _semantic_user_event_is_pure_question(
        "How the engineering team deploys is documented in the runbook."
    ) is False
    assert _semantic_user_event_is_pure_question(
        "How our platform engineering team deploys is documented in the runbook."
    ) is False
    assert _semantic_user_event_is_pure_question(
        "What we decided is documented in the runbook."
    ) is False
    assert _semantic_user_event_is_pure_question(
        "What caused the outage is documented in the postmortem."
    ) is False
    assert _semantic_user_event_is_pure_question(
        "What happened is documented in the runbook."
    ) is False
    assert _semantic_text_is_question(
        "Where our application access logs are stored."
    ) is True

    cases = [
        (
            "where-long-subject",
            "Where our application access logs are stored.",
            "Application access logs are stored.",
            "Where our application access logs are stored",
            "domain_fact",
            "application access logs",
            "stored",
        ),
        (
            "when-long-subject",
            "When our nightly backup verification jobs are completed.",
            "The backup verification jobs complete nightly.",
            "When our nightly backup verification jobs are completed",
            "domain_fact",
            "backup verification schedule",
            "nightly",
        ),
    ]
    for name, user_content, content, evidence, memory_type, subject, value in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "skipped", round_payload
            assert _formal_memories(client, headers, "semantic-project") == []


def test_final_validation_round_seventeen_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _SEMANTIC_DEICTIC_REFERENCE,
        _SEMANTIC_TRANSIENT,
        _semantic_evidence_supports_content,
        _semantic_text_is_imperative,
        _semantic_user_event_is_pure_question,
    )

    assert _semantic_user_event_is_pure_question(
        "Where our application access logs are periodically archived."
    ) is True
    assert _semantic_user_event_is_pure_question(
        "Where our application access logs have been automatically archived."
    ) is True
    assert _semantic_user_event_is_pure_question(
        "Where our application access logs are not archived."
    ) is True
    assert _SEMANTIC_TRANSIENT.search("The migration is planned shortly.")
    assert _SEMANTIC_TRANSIENT.search("The migration is planned in a few days.")
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "As I previously said, the API uses PostgreSQL."
    )
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "Like I said earlier, the API uses PostgreSQL."
    )
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "As noted above, the API uses PostgreSQL."
    )
    assert _semantic_text_is_imperative(
        "How the engineering team deploys is documented in the runbook."
    ) is False
    assert _semantic_evidence_supports_content(
        "The user prefers high-contrast dark mode.",
        "I prefer dark mode",
        claim_value="dark",
        context="I prefer dark mode",
    ) is False

    cases = [
        (
            "where-adverb-passive",
            "Where our application access logs are periodically archived.",
            "Application access logs are archived.",
            "Where our application access logs are periodically archived",
            "domain_fact",
            "application access logs",
            "archived",
            "skipped",
        ),
        (
            "planned-shortly",
            "The migration is planned shortly.",
            "The migration is planned shortly.",
            "The migration is planned shortly",
            "decision",
            "migration timing",
            "shortly",
            "skipped",
        ),
        (
            "as-i-previously-said",
            "As I previously said, the API uses PostgreSQL.",
            "As I previously said, the API uses PostgreSQL.",
            "As I previously said, the API uses PostgreSQL",
            "domain_fact",
            "API database",
            "postgresql",
            "skipped",
        ),
        (
            "high-contrast-modifier",
            "I prefer dark mode.",
            "The user prefers high-contrast dark mode.",
            "I prefer dark mode",
            "preference",
            "user:preference:theme",
            "dark",
            "skipped",
        ),
        (
            "engineering-team-statement",
            "How the engineering team deploys is documented in the runbook.",
            "How the engineering team deploys is documented in the runbook.",
            "How the engineering team deploys is documented in the runbook",
            "domain_fact",
            "deployment process",
            "documented in the runbook",
            "completed",
        ),
    ]
    for (
        name,
        user_content,
        content,
        evidence,
        memory_type,
        subject,
        value,
        expected,
    ) in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == expected, round_payload
            assert len(_formal_memories(client, headers, "semantic-project")) == (
                1 if expected == "completed" else 0
            )


def test_final_validation_round_eighteen_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _SEMANTIC_TRANSIENT,
        _semantic_user_event_is_pure_question,
    )

    assert _semantic_user_event_is_pure_question(
        "Where our application access logs are not yet archived."
    ) is True
    assert _semantic_user_event_is_pure_question(
        "Where our application access logs have not yet been archived."
    ) is True
    assert _semantic_user_event_is_pure_question(
        "Where our application access logs haven't been archived."
    ) is True
    assert _semantic_user_event_is_pure_question(
        "Where our application access logs aren't archived."
    ) is True
    assert _SEMANTIC_TRANSIENT.search("The migration is planned for Friday.")
    assert _SEMANTIC_TRANSIENT.search("The migration is planned in two weeks.")
    assert _SEMANTIC_TRANSIENT.search("The migration is planned on Friday.")

    cases = [
        (
            "where-not-yet-archived",
            "Where our application access logs are not yet archived.",
            "Application access logs are not archived.",
            "Where our application access logs are not yet archived",
            "domain_fact",
            "application access logs",
            "not archived",
        ),
        (
            "planned-for-friday",
            "The migration is planned for Friday.",
            "The migration is planned for Friday.",
            "The migration is planned for Friday",
            "decision",
            "migration timing",
            "friday",
        ),
    ]
    for name, user_content, content, evidence, memory_type, subject, value in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "skipped", round_payload
            assert _formal_memories(client, headers, "semantic-project") == []


def test_final_validation_round_nineteen_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _SEMANTIC_DEICTIC_REFERENCE,
        _SEMANTIC_TRANSIENT,
        _semantic_user_event_is_pure_request,
    )

    assert _semantic_user_event_is_pure_request(
        "Walk us through how the API authenticates users."
    ) is True
    assert _semantic_user_event_is_pure_request(
        "Take me through the deployment."
    ) is True
    assert _semantic_user_event_is_pure_request(
        "I want you to explain caching."
    ) is True
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "As we discussed, the API uses PostgreSQL."
    )
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "Per our prior discussion, the API uses PostgreSQL."
    )
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "Based on our previous conversation, the API uses PostgreSQL."
    )
    assert _SEMANTIC_TRANSIENT.search("The migration is slated for Friday.")
    assert _SEMANTIC_TRANSIENT.search("The migration is set for Friday.")
    assert _SEMANTIC_TRANSIENT.search(
        "The migration is scheduled to launch soon."
    )

    cases = [
        (
            "walk-us-through",
            "Walk us through how the API authenticates users.",
            "The API authenticates users.",
            "Walk us through how the API authenticates users",
            "domain_fact",
            "api authentication",
            "users",
        ),
        (
            "as-we-discussed",
            "As we discussed, the API uses PostgreSQL.",
            "As we discussed, the API uses PostgreSQL.",
            "As we discussed, the API uses PostgreSQL",
            "domain_fact",
            "API database",
            "postgresql",
        ),
        (
            "per-prior-discussion",
            "Per our prior discussion, the API uses PostgreSQL.",
            "Per our prior discussion, the API uses PostgreSQL.",
            "Per our prior discussion, the API uses PostgreSQL",
            "domain_fact",
            "API database",
            "postgresql",
        ),
        (
            "slated-for-friday",
            "The migration is slated for Friday.",
            "The migration is slated for Friday.",
            "The migration is slated for Friday",
            "decision",
            "migration timing",
            "friday",
        ),
        (
            "set-for-friday",
            "The migration is set for Friday.",
            "The migration is set for Friday.",
            "The migration is set for Friday",
            "decision",
            "migration timing",
            "friday",
        ),
    ]
    for name, user_content, content, evidence, memory_type, subject, value in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "skipped", round_payload
            assert _formal_memories(client, headers, "semantic-project") == []


def test_final_validation_round_twenty_regressions(tmp_path: Path) -> None:
    from personal_agent_memory.mem2_kernel import (  # noqa: PLC0415
        _SEMANTIC_DEICTIC_REFERENCE,
        _SEMANTIC_TRANSIENT,
        _semantic_user_event_is_pure_question,
        _semantic_user_event_is_pure_request,
    )

    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "Like I said, the API uses PostgreSQL."
    )
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "As noted, the API uses PostgreSQL."
    )
    assert _SEMANTIC_DEICTIC_REFERENCE.search(
        "Per our last discussion, the API uses PostgreSQL."
    )
    assert _semantic_user_event_is_pure_question(
        "Where our application access logs are automatically and periodically archived."
    ) is True
    assert _SEMANTIC_TRANSIENT.search("The migration is targeted for Friday.")
    assert _semantic_user_event_is_pure_request(
        "I'd like you to explain deployment in the runbook."
    ) is True
    assert _semantic_user_event_is_pure_request(
        "I’d like you to explain deployment."
    ) is True

    cases = [
        (
            "like-i-said",
            "Like I said, the API uses PostgreSQL.",
            "Like I said, the API uses PostgreSQL.",
            "Like I said, the API uses PostgreSQL",
            "domain_fact",
            "API database",
            "postgresql",
        ),
        (
            "as-noted",
            "As noted, the API uses PostgreSQL.",
            "As noted, the API uses PostgreSQL.",
            "As noted, the API uses PostgreSQL",
            "domain_fact",
            "API database",
            "postgresql",
        ),
        (
            "per-last-discussion",
            "Per our last discussion, the API uses PostgreSQL.",
            "Per our last discussion, the API uses PostgreSQL.",
            "Per our last discussion, the API uses PostgreSQL",
            "domain_fact",
            "API database",
            "postgresql",
        ),
        (
            "parallel-ly-question",
            "Where our application access logs are automatically and periodically archived.",
            "Application access logs are archived.",
            "Where our application access logs are automatically and periodically archived",
            "domain_fact",
            "application access logs",
            "archived",
        ),
        (
            "targeted-for-friday",
            "The migration is targeted for Friday.",
            "The migration is targeted for Friday.",
            "The migration is targeted for Friday",
            "decision",
            "migration timing",
            "friday",
        ),
        (
            "contracted-request",
            "I'd like you to explain deployment in the runbook.",
            "The runbook explains deployment.",
            "I'd like you to explain deployment in the runbook",
            "domain_fact",
            "deployment documentation",
            "runbook",
        ),
    ]
    for name, user_content, content, evidence, memory_type, subject, value in cases:
        base = tmp_path / name
        state_dir = base / "state"
        root = base / "project"
        base.mkdir()
        executor = ManualSemanticExecutor()
        result = _memory(
            content=content,
            memory_type=memory_type,
            evidence=evidence,
            event_id=f"{name}-turn-user",
            subject=subject,
            value=value,
        )

        def extractor(
            context: dict[str, object],
            result: dict[str, object] = result,
        ) -> list[dict[str, object]]:
            del context
            return [result]

        with TestClient(_app(state_dir, extractor, executor)) as client:
            headers = _headers(state_dir)
            _prepare_project(client, headers, root)
            _capture_turn(
                client,
                headers,
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
                user_content=user_content,
                assistant_content="Noted.",
                cwd=root,
            )
            executor.drain()
            round_payload = _round(
                client,
                headers,
                region_id="semantic-project",
                session_id=f"{name}-session",
                turn_id=f"{name}-turn",
            )
            assert round_payload["semantic_status"] == "skipped", round_payload
            assert _formal_memories(client, headers, "semantic-project") == []
