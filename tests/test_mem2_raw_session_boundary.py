from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jiuwen_memory.api import Modality, legacy_request_context
from jiuwen_memory.common.type_def import Scope
from jiuwen_memory.control import BatchWriteItem

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings


def _app(state_dir: Path):
    return create_app(Settings(state_dir=state_dir))


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _region(
    client: TestClient,
    headers: dict[str, str],
    kind: str,
    region_id: str,
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


def _memory(
    client: TestClient,
    headers: dict[str, str],
    region_id: str,
    content: str,
    *,
    idempotency_key: str,
) -> str:
    response = client.post(
        "/api/v1/memories",
        headers=headers,
        json={
            "region_id": region_id,
            "content": content,
            "memory_type": "domain_fact",
            "source_references": ["test:raw-session-boundary"],
            "actor": "test",
            "idempotency_key": idempotency_key,
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


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
    return response.json()


def _direct_unit(
    client: TestClient,
    region_id: str,
    content: str,
    *,
    memory_type: str,
    session_id: str = "",
    extra_metadata: dict[str, object] | None = None,
) -> tuple[str, Scope]:
    kernel = client.app.state.memory_kernel
    bundle = kernel._require_bundle()
    region = kernel._regions.require_region(region_id)
    scope = Scope(
        org="local",
        space=region.space,
        user="local",
        agent="personal-agent-memory",
        session=session_id,
    )
    units = bundle.runtime.api.add(
        content,
        scope,
        Modality.TEXT,
        security=legacy_request_context(scope),
        user_metadata={
            "memory_type": memory_type,
            "region_id": region_id,
            **dict(extra_metadata or {}),
        },
    )
    assert units
    return units[0].id, scope


def _direct_batch_units(
    client: TestClient,
    region_id: str,
    items: list[tuple[str, dict[str, object]]],
) -> None:
    kernel = client.app.state.memory_kernel
    bundle = kernel._require_bundle()
    region = kernel._regions.require_region(region_id)
    scope = Scope(
        org="local",
        space=region.space,
        user="local",
        agent="personal-agent-memory",
        session="",
    )
    result = bundle.runtime.api.batch_add(
        [
            BatchWriteItem(
                content=content,
                user_metadata={"region_id": region_id, **metadata},
            )
            for content, metadata in items
        ],
        scope,
        security=legacy_request_context(scope),
    )
    errors = [outcome.error for outcome in result.outcomes if outcome.error]
    assert errors == []


def _search(
    client: TestClient,
    headers: dict[str, str],
    *,
    query: str,
    cwd: Path,
    session_id: str,
) -> dict[str, object]:
    response = client.post(
        "/api/v1/search",
        headers=headers,
        json={"query": query, "cwd": str(cwd), "session_id": session_id, "limit": 20},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _mcp_search(
    client: TestClient,
    headers: dict[str, str],
    *,
    query: str,
    cwd: Path | None = None,
    session_id: str | None = None,
    region_id: str | None = None,
    limit: int = 20,
) -> dict[str, object]:
    arguments: dict[str, object] = {"query": query, "limit": limit}
    if cwd is not None:
        arguments["cwd"] = str(cwd)
    if session_id is not None:
        arguments["session_id"] = session_id
    if region_id is not None:
        arguments["region_id"] = region_id
    response = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "memory_search",
                "arguments": arguments,
            },
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["result"]["isError"] is False, payload
    return payload["result"]["structuredContent"]


def test_default_rest_mcp_and_codex_search_exclude_raw_and_evolved_units(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()
    question = "帮我分析下当前项目"

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "boundary-project")
        _bind(client, headers, project_root, "boundary-project")
        formal_id = _memory(
            client,
            headers,
            "boundary-project",
            "项目使用 uv 管理 Python 依赖。",
            idempotency_key="boundary-formal",
        )
        _capture(
            client,
            headers,
            event_id="boundary-user",
            session_id="boundary-session",
            turn_id="boundary-turn",
            kind="user",
            content=question,
            cwd=project_root,
            occurred_at="2026-10-27T01:00:00Z",
        )
        _capture(
            client,
            headers,
            event_id="boundary-assistant",
            session_id="boundary-session",
            turn_id="boundary-turn",
            kind="assistant",
            content="已完成项目分析。",
            cwd=project_root,
            occurred_at="2026-10-27T01:00:01Z",
        )
        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "boundary-session"},
        )
        assert consolidated.status_code == 202, consolidated.text

        rest = _search(
            client,
            headers,
            query="项目",
            cwd=project_root,
            session_id="boundary-session",
        )
        mcp = _mcp_search(
            client,
            headers,
            query="项目",
            cwd=project_root,
            session_id="boundary-session",
        )
        for package in (rest, mcp):
            contents = [str(item["content"]) for item in package["results"]]
            assert any(formal_id in str(item["document_id"]) for item in package["results"])
            assert all(question not in content for content in contents)
            assert all(
                item.get("user_metadata", {}).get("memory_type") not in {"capture", "evolved"}
                for item in package["results"]
            )

        rounds = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={
                "region_id": "boundary-project",
                "session_id": "boundary-session",
                "turn_id": "boundary-turn",
            },
        )
        assert rounds.status_code == 200, rounds.text
        assert rounds.json()[0]["user_message"]["content"] == question

        events = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"region_id": "boundary-project", "session_id": "boundary-session"},
        )
        assert events.status_code == 200, events.text
        event_payload = events.json()
        capture_events = [
            item for item in event_payload if item["record_type"] == "capture_event"
        ]
        diagnostics = [
            item for item in event_payload if item["record_type"] == "evolved_diagnostic"
        ]
        assert [item["content"] for item in capture_events] == [question, "已完成项目分析。"]
        assert len(diagnostics) == 1
        assert diagnostics[0]["diagnostic_kind"] == "legacy_evolved"
        assert question in diagnostics[0]["content"]
        assert "已完成项目分析。" in diagnostics[0]["content"]


def test_identityless_capture_units_are_excluded_from_rest_mcp_and_graph_expansion(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "topic", "identityless-region")
        capture_id, scope = _direct_unit(
            client,
            "identityless-region",
            "captureonlymarker graphlink sharedalpha sharedbeta unrefined question",
            memory_type="capture",
        )
        seed_id, _ = _direct_unit(
            client,
            "identityless-region",
            "seedonlymarker graphlink sharedalpha sharedbeta durable fact",
            memory_type="domain_fact",
        )
        kernel = client.app.state.memory_kernel
        bundle = kernel._require_bundle()
        associated = kernel._associate_graph_scope(
            bundle,
            scope,
            region_id="identityless-region",
        )
        assert associated, "graph association should complete"
        graph_store = bundle.graph_store
        assert graph_store is not None
        graph_ids = set(graph_store.node_ids(scope))
        assert {capture_id, seed_id}.issubset(graph_ids)

        rest_capture = client.post(
            "/api/v1/search",
            headers=headers,
            json={"region_id": "identityless-region", "query": "captureonlymarker"},
        )
        assert rest_capture.status_code == 200, rest_capture.text
        assert rest_capture.json()["results"] == []
        mcp_capture = _mcp_search(
            client,
            headers,
            query="captureonlymarker",
            region_id="identityless-region",
        )
        assert mcp_capture["results"] == []

        expanded = client.post(
            "/api/v1/search",
            headers=headers,
            json={"region_id": "identityless-region", "query": "seedonlymarker"},
        )
        assert expanded.status_code == 200, expanded.text
        expanded_results = expanded.json()["results"]
        assert any(item["document_id"] == seed_id for item in expanded_results)
        assert all(item["document_id"] != capture_id for item in expanded_results)
        assert all(
            "captureonlymarker" not in str(item["content"]) for item in expanded_results
        )


def test_default_recall_backfills_formal_hits_after_filtering_raw_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    project_root = tmp_path / "project"
    project_root.mkdir()

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "candidate-pool-region")
        _bind(client, headers, project_root, "candidate-pool-region")
        formal_id = _memory(
            client,
            headers,
            "candidate-pool-region",
            "topkboundary formal durable fact",
            idempotency_key="candidate-pool-formal",
        )
        raw_units = [
            (
                f"{'topkboundary ' * 12} raw capture {index}",
                {"memory_type": "capture"},
            )
            for index in range(499)
        ]
        raw_units.append(
            (
                f"{'topkboundary ' * 12} identity-bearing formal-looking capture",
                {
                    "memory_type": "domain_fact",
                    "capture_event_id": "candidate-pool-poison",
                },
            )
        )
        _direct_batch_units(client, "candidate-pool-region", raw_units)
        monkeypatch.setattr(client.app.state.memory_kernel, "_vector_enabled", True)

        def assert_backfilled(package: dict[str, object]) -> None:
            results = package["results"]
            assert isinstance(results, list)
            assert [item["document_id"] for item in results] == [formal_id]
            assert "default_recall_candidate_truncated" not in package["degradation"]

        for _ in range(2):
            rest = client.post(
                "/api/v1/search",
                headers=headers,
                json={
                    "region_id": "candidate-pool-region",
                    "query": "topkboundary",
                    "limit": 5,
                },
            )
            assert rest.status_code == 200, rest.text
            assert_backfilled(rest.json())

            mcp = _mcp_search(
                client,
                headers,
                query="topkboundary",
                region_id="candidate-pool-region",
                limit=5,
            )
            assert_backfilled(mcp)

            codex = client.post(
                "/api/v1/search",
                headers=headers,
                json={
                    "cwd": str(project_root),
                    "session_id": "codex-candidate-session",
                    "query": "topkboundary",
                    "limit": 5,
                },
            )
            assert codex.status_code == 200, codex.text
            assert_backfilled(codex.json())

        monkeypatch.setattr(client.app.state.memory_kernel, "_vector_enabled", False)
        keyword = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "region_id": "candidate-pool-region",
                "query": "topkboundary",
                "limit": 5,
            },
        )
        assert keyword.status_code == 200, keyword.text
        assert_backfilled(keyword.json())


def test_truncated_default_recall_reports_explicit_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "topic", "truncated-region")
        _memory(
            client,
            headers,
            "truncated-region",
            "可检索正式记忆",
            idempotency_key="truncated-formal",
        )
        kernel = client.app.state.memory_kernel

        def truncated_search(
            *args: object, **kwargs: object
        ) -> tuple[list[object], list[object], bool]:
            del args, kwargs
            return [], [], True

        def truncated_fallback(
            *args: object, **kwargs: object
        ) -> tuple[list[object], bool, bool]:
            del args, kwargs
            return [], False, True

        monkeypatch.setattr(kernel, "_search_scope_candidates", truncated_search)
        monkeypatch.setattr(kernel, "_keyword_fallback", truncated_fallback)
        response = client.post(
            "/api/v1/search",
            headers=headers,
            json={"region_id": "truncated-region", "query": "正式", "limit": 5},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["results"] == []
        assert payload["degraded"] is True
        assert "default_recall_candidate_truncated" in payload["degradation"]


def test_legacy_capture_and_evolved_are_visible_only_in_raw_session_diagnostics(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "legacy-region")
        _region(client, headers, "project", "legacy-other-region")
        legacy_metadata = {
            "event_id": "legacy-user-event",
            "session_id": "legacy-session",
            "turn_id": "legacy-turn",
            "event_kind": "user",
            "occurred_at": "2026-10-27T07:00:00Z",
            "region_id": "legacy-region",
        }
        user_id, _ = _direct_unit(
            client,
            "legacy-region",
            "legacyrawmarker legacy user question",
            memory_type="capture",
            session_id="legacy-session",
            extra_metadata=legacy_metadata,
        )
        assistant_id, _ = _direct_unit(
            client,
            "legacy-region",
            "legacy assistant final answer",
            memory_type="capture",
            session_id="legacy-session",
            extra_metadata={
                **legacy_metadata,
                "event_id": "legacy-assistant-event",
                "event_kind": "assistant",
                "occurred_at": "2026-10-27T07:00:01Z",
            },
        )
        diagnostic_id, _ = _direct_unit(
            client,
            "legacy-region",
            "legacyevolvedmarker old raw concatenation",
            memory_type="evolved",
            extra_metadata={
                "session_id": "legacy-session",
                "turn_id": "legacy-turn",
                "region_id": "legacy-region",
                "source_unit_ids": [user_id, assistant_id],
            },
        )
        _direct_unit(
            client,
            "legacy-region",
            "other session legacy evolved",
            memory_type="evolved",
            extra_metadata={
                "session_id": "other-session",
                "turn_id": "legacy-turn",
                "region_id": "legacy-region",
            },
        )
        _direct_unit(
            client,
            "legacy-region",
            "other turn legacy evolved",
            memory_type="evolved",
            extra_metadata={
                "session_id": "legacy-session",
                "turn_id": "other-turn",
                "region_id": "legacy-region",
            },
        )
        _direct_unit(
            client,
            "legacy-other-region",
            "other region legacy evolved",
            memory_type="evolved",
            extra_metadata={
                "session_id": "legacy-session",
                "turn_id": "legacy-turn",
                "region_id": "legacy-other-region",
            },
        )

        filters = {
            "region_id": "legacy-region",
            "session_id": "legacy-session",
            "turn_id": "legacy-turn",
        }
        events_response = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params=filters,
        )
        assert events_response.status_code == 200, events_response.text
        events = events_response.json()
        assert [item["unit_id"] for item in events] == [user_id, assistant_id, diagnostic_id]
        assert [item["record_type"] for item in events] == [
            "capture_event",
            "capture_event",
            "evolved_diagnostic",
        ]
        assert [item["event_kind"] for item in events] == ["user", "assistant", ""]
        assert [item["legacy"] for item in events] == [True, True, True]
        assert events[2]["diagnostic_kind"] == "legacy_evolved"
        assert events[2]["content"] == "legacyevolvedmarker old raw concatenation"

        rounds_response = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params=filters,
        )
        assert rounds_response.status_code == 200, rounds_response.text
        assert len(rounds_response.json()) == 1
        round_payload = rounds_response.json()[0]
        assert round_payload["user_message"]["unit_id"] == user_id
        assert round_payload["assistant_final"]["unit_id"] == assistant_id
        assert [item["unit_id"] for item in round_payload["diagnostics"]] == [diagnostic_id]
        assert [item["record_type"] for item in round_payload["events"]] == [
            "capture_event",
            "capture_event",
            "evolved_diagnostic",
        ]

        rest_legacy_capture = client.post(
            "/api/v1/search",
            headers=headers,
            json={"region_id": "legacy-region", "query": "legacyrawmarker"},
        )
        assert rest_legacy_capture.status_code == 200, rest_legacy_capture.text
        assert rest_legacy_capture.json()["results"] == []
        mcp_legacy_evolved = _mcp_search(
            client,
            headers,
            query="legacyevolvedmarker",
            region_id="legacy-region",
        )
        assert mcp_legacy_evolved["results"] == []


def test_legacy_completed_evolution_fingerprint_backfills_then_reopens_on_new_event(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    session_id = "legacy-fingerprint-session"
    turn_id = "legacy-fingerprint-turn"
    region_id = "legacy-fingerprint-region"

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", region_id)
        _bind(client, headers, root, region_id)
        user = _capture(
            client,
            headers,
            event_id="legacy-fingerprint-user",
            session_id=session_id,
            turn_id=turn_id,
            kind="user",
            content="旧迁移用户消息",
            cwd=root,
            occurred_at="2026-10-27T07:00:00Z",
        )
        assistant = _capture(
            client,
            headers,
            event_id="legacy-fingerprint-assistant",
            session_id=session_id,
            turn_id=turn_id,
            kind="assistant",
            content="旧迁移助手回复",
            cwd=root,
            occurred_at="2026-10-27T07:00:01Z",
        )
        legacy_memory_id, _ = _direct_unit(
            client,
            region_id,
            "旧迁移完成诊断",
            memory_type="evolved",
            session_id=session_id,
            extra_metadata={
                "session_id": session_id,
                "turn_id": turn_id,
                "region_id": region_id,
                "source_unit_ids": [user["memory_id"], assistant["memory_id"]],
            },
        )

    evolve_key = hashlib.sha256(
        "\0".join(
            (
                "local",
                str(user["space"]),
                session_id,
                region_id,
                turn_id,
            )
        ).encode()
    ).hexdigest()
    maintenance_db = state_dir / "mem2" / "maintenance.sqlite3"
    with sqlite3.connect(maintenance_db) as connection:
        connection.execute("DROP TABLE capture_evolutions")
        connection.execute(
            """CREATE TABLE capture_evolutions (
                evolve_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                region_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.execute(
            """INSERT INTO capture_evolutions
               (evolve_key, session_id, turn_id, region_id, memory_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                evolve_key,
                session_id,
                turn_id,
                region_id,
                legacy_memory_id,
                "2026-10-27 07:00:02",
            ),
        )

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        params = {
            "region_id": region_id,
            "session_id": session_id,
            "turn_id": turn_id,
        }
        migrated = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params=params,
        )
        assert migrated.status_code == 200, migrated.text
        migrated_round = migrated.json()[0]
        assert migrated_round["distillation_status"] == "completed"
        assert migrated_round["status"] == "done"
        assert migrated_round["memory_id"] == legacy_memory_id

        with sqlite3.connect(maintenance_db) as connection:
            fingerprint = connection.execute(
                "SELECT source_fingerprint FROM capture_evolutions WHERE evolve_key = ?",
                (evolve_key,),
            ).fetchone()
        assert fingerprint is not None
        assert str(fingerprint[0])

        _capture(
            client,
            headers,
            event_id="legacy-fingerprint-late-user",
            session_id=session_id,
            turn_id=turn_id,
            kind="user",
            content="旧迁移后的新用户事件",
            cwd=root,
            occurred_at="2026-10-27T07:00:03Z",
        )
        reopened = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params=params,
        )
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()[0]["distillation_status"] == "pending_distillation"
        assert reopened.json()[0]["status"] == "pending"


def test_legacy_completed_fingerprint_does_not_swallow_event_before_first_read(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    session_id = "legacy-late-first-session"
    turn_id = "legacy-late-first-turn"
    region_id = "legacy-late-first-region"

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", region_id)
        _bind(client, headers, root, region_id)
        user = _capture(
            client,
            headers,
            event_id="legacy-late-first-user",
            session_id=session_id,
            turn_id=turn_id,
            kind="user",
            content="旧集合用户消息",
            cwd=root,
            occurred_at="2026-10-27T09:00:00Z",
        )
        assistant = _capture(
            client,
            headers,
            event_id="legacy-late-first-assistant",
            session_id=session_id,
            turn_id=turn_id,
            kind="assistant",
            content="旧集合助手回复",
            cwd=root,
            occurred_at="2026-10-27T09:00:01Z",
        )
        legacy_memory_id, _ = _direct_unit(
            client,
            region_id,
            "旧集合完成诊断",
            memory_type="evolved",
            session_id=session_id,
            extra_metadata={
                "session_id": session_id,
                "turn_id": turn_id,
                "region_id": region_id,
                "source_unit_ids": [user["memory_id"], assistant["memory_id"]],
            },
        )

    evolve_key = hashlib.sha256(
        "\0".join(
            ("local", str(user["space"]), session_id, region_id, turn_id)
        ).encode()
    ).hexdigest()
    maintenance_db = state_dir / "mem2" / "maintenance.sqlite3"
    with sqlite3.connect(maintenance_db) as connection:
        connection.execute("DROP TABLE capture_evolutions")
        connection.execute(
            """CREATE TABLE capture_evolutions (
                evolve_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                region_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.execute(
            """INSERT INTO capture_evolutions
               (evolve_key, session_id, turn_id, region_id, memory_id)
               VALUES (?, ?, ?, ?, ?)""",
            (evolve_key, session_id, turn_id, region_id, legacy_memory_id),
        )

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _capture(
            client,
            headers,
            event_id="legacy-late-first-extra",
            session_id=session_id,
            turn_id=turn_id,
            kind="user",
            content="首次读取前到达的晚到事件",
            cwd=root,
            occurred_at="2026-10-27T09:00:02Z",
        )
        params = {
            "region_id": region_id,
            "session_id": session_id,
            "turn_id": turn_id,
        }
        reopened = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params=params,
        )
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()[0]["distillation_status"] == "pending_distillation"
        with sqlite3.connect(maintenance_db) as connection:
            fingerprint = connection.execute(
                "SELECT source_fingerprint FROM capture_evolutions WHERE evolve_key = ?",
                (evolve_key,),
            ).fetchone()
        assert fingerprint == ("",)

        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": session_id},
        )
        assert consolidated.status_code == 202, consolidated.text
        assert consolidated.json() == {"queued": 1}
        completed = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params=params,
        ).json()[0]
        assert completed["distillation_status"] == "completed"
        assert "首次读取前到达的晚到事件" in "".join(
            str(item["content"]) for item in completed["diagnostics"]
        )


def test_raw_session_reads_are_region_session_turn_scoped_and_time_ordered(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root_a = tmp_path / "project-a"
    root_b = tmp_path / "project-b"
    root_a.mkdir()
    root_b.mkdir()

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "region-a")
        _region(client, headers, "project", "region-b")
        _bind(client, headers, root_a, "region-a")
        _bind(client, headers, root_b, "region-b")

        _capture(
            client,
            headers,
            event_id="a-assistant",
            session_id="shared-session",
            turn_id="shared-turn",
            kind="assistant",
            content="助手最终回复 A",
            cwd=root_a,
            occurred_at="2026-10-27T02:00:05Z",
        )
        user = _capture(
            client,
            headers,
            event_id="a-user",
            session_id="shared-session",
            turn_id="shared-turn",
            kind="user",
            content="用户消息 A",
            cwd=root_a,
            occurred_at="2026-10-27T02:00:01Z",
        )
        _capture(
            client,
            headers,
            event_id="a-other-session",
            session_id="other-session",
            turn_id="shared-turn",
            kind="user",
            content="其他会话",
            cwd=root_a,
            occurred_at="2026-10-27T02:00:02Z",
        )
        _capture(
            client,
            headers,
            event_id="b-user",
            session_id="shared-session",
            turn_id="shared-turn",
            kind="user",
            content="用户消息 B",
            cwd=root_b,
            occurred_at="2026-10-27T02:00:01Z",
        )

        events = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={
                "region_id": "region-a",
                "session_id": "shared-session",
                "turn_id": "shared-turn",
            },
        )
        assert events.status_code == 200, events.text
        payload = events.json()
        assert [item["event_id"] for item in payload] == ["a-user", "a-assistant"]
        assert {item["region_id"] for item in payload} == {"region-a"}
        assert {item["space"] for item in payload} == {str(user["space"])}
        assert all(item["received_at"] for item in payload)

        rounds = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={
                "region_id": "region-a",
                "session_id": "shared-session",
                "turn_id": "shared-turn",
            },
        )
        assert rounds.status_code == 200, rounds.text
        assert len(rounds.json()) == 1
        round_payload = rounds.json()[0]
        assert round_payload["user_message"]["event_id"] == "a-user"
        assert round_payload["assistant_final"]["event_id"] == "a-assistant"
        assert [item["event_id"] for item in round_payload["events"]] == [
            "a-user",
            "a-assistant",
        ]
        assert round_payload["received_at"] == max(item["received_at"] for item in payload)

        empty = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={
                "region_id": "region-a",
                "session_id": "shared-session",
                "turn_id": "missing-turn",
            },
        )
        assert empty.status_code == 200, empty.text
        assert empty.json() == []


def test_raw_rounds_sort_by_utc_time_with_stable_ties_and_invalid_timestamps(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "timezone-region")
        _bind(client, headers, root, "timezone-region")
        for event_id, turn_id, occurred_at in (
            ("timezone-z-event", "timezone-z", "2026-10-27T03:00:00Z"),
            ("timezone-plus-event", "timezone-plus", "2026-10-27T10:00:00+08:00"),
            ("timezone-tie-z-event", "timezone-tie-z", "2026-10-27T04:00:00Z"),
            (
                "timezone-tie-a-event",
                "timezone-tie-a",
                "2026-10-27T12:00:00+08:00",
            ),
        ):
            _capture(
                client,
                headers,
                event_id=event_id,
                session_id="timezone-session",
                turn_id=turn_id,
                kind="user",
                content=f"{turn_id} content",
                cwd=root,
                occurred_at=occurred_at,
            )
        _direct_unit(
            client,
            "timezone-region",
            "legacy invalid timestamp content",
            memory_type="capture",
            session_id="timezone-session",
            extra_metadata={
                "capture_event_id": "legacy-invalid-event",
                "event_id": "legacy-invalid-event",
                "session_id": "timezone-session",
                "turn_id": "legacy-invalid-turn",
                "event_kind": "user",
                "occurred_at": "not-a-timestamp",
                "region_id": "timezone-region",
            },
        )

        rounds = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={"region_id": "timezone-region", "session_id": "timezone-session"},
        )
        assert rounds.status_code == 200, rounds.text
        assert [item["turn_id"] for item in rounds.json()] == [
            "timezone-plus",
            "timezone-z",
            "timezone-tie-a",
            "timezone-tie-z",
            "legacy-invalid-turn",
        ]
        reversed_rounds = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={"region_id": "timezone-region", "session_id": "timezone-session"},
        )
        assert reversed_rounds.status_code == 200, reversed_rounds.text
        assert [item["turn_id"] for item in reversed_rounds.json()] == [
            item["turn_id"] for item in rounds.json()
        ]


def test_raw_rounds_report_completion_distillation_and_failure_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()
    app = _app(state_dir)

    with TestClient(app, raise_server_exceptions=False) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "status-region")
        _bind(client, headers, root, "status-region")
        _capture(
            client,
            headers,
            event_id="incomplete-user",
            session_id="status-session",
            turn_id="incomplete-turn",
            kind="user",
            content="尚未收到回复的提问",
            cwd=root,
            occurred_at="2026-10-27T03:00:00Z",
        )
        for turn_index, turn_id in enumerate(("ready-turn", "completed-turn")):
            _capture(
                client,
                headers,
                event_id=f"{turn_id}-user",
                session_id="status-session",
                turn_id=turn_id,
                kind="user",
                content=f"{turn_id} 用户消息",
                cwd=root,
                occurred_at=f"2026-10-27T03:00:0{turn_index + 2}Z",
            )
            _capture(
                client,
                headers,
                event_id=f"{turn_id}-assistant",
                session_id="status-session",
                turn_id=turn_id,
                kind="assistant",
                content=f"{turn_id} 助手回复",
                cwd=root,
                occurred_at=f"2026-10-27T03:00:1{turn_index + 2}Z",
            )
        consolidated = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "status-session"},
        )
        assert consolidated.status_code == 202, consolidated.text
        rounds = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={"region_id": "status-region", "session_id": "status-session"},
        )
        assert rounds.status_code == 200, rounds.text
        by_turn = {item["turn_id"]: item for item in rounds.json()}
        assert by_turn["incomplete-turn"]["distillation_status"] == "pending_completion"
        assert by_turn["incomplete-turn"]["status"] == "pending"
        assert by_turn["completed-turn"]["distillation_status"] == "completed"
        assert by_turn["completed-turn"]["status"] == "done"
        _capture(
            client,
            headers,
            event_id="completed-turn-late-user",
            session_id="status-session",
            turn_id="completed-turn",
            kind="user",
            content="完成回合的后续用户消息",
            cwd=root,
            occurred_at="2026-10-27T03:00:30Z",
        )
        stale_rounds = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={
                "region_id": "status-region",
                "session_id": "status-session",
                "turn_id": "completed-turn",
            },
        )
        assert stale_rounds.status_code == 200, stale_rounds.text
        assert stale_rounds.json()[0]["distillation_status"] == "pending_distillation"

        _capture(
            client,
            headers,
            event_id="failed-user",
            session_id="failed-session",
            turn_id="failed-turn",
            kind="user",
            content="失败回应用户消息",
            cwd=root,
            occurred_at="2026-10-27T04:00:00Z",
        )
        _capture(
            client,
            headers,
            event_id="failed-assistant",
            session_id="failed-session",
            turn_id="failed-turn",
            kind="assistant",
            content="失败回应助手消息",
            cwd=root,
            occurred_at="2026-10-27T04:00:01Z",
        )
        kernel = app.state.memory_kernel
        original_complete = kernel._complete_evolution

        def fail_completion(evolve_key: str, memory_id: str) -> None:
            del evolve_key, memory_id
            raise RuntimeError("simulated consolidation failure")

        monkeypatch.setattr(kernel, "_complete_evolution", fail_completion)
        failed = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "failed-session"},
        )
        assert failed.status_code == 500, failed.text
        failed_rounds = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={"region_id": "status-region", "session_id": "failed-session"},
        )
        assert failed_rounds.status_code == 200, failed_rounds.text
        failed_round = failed_rounds.json()[0]
        assert failed_round["distillation_status"] == "failed"
        assert failed_round["status"] == "failed"
        assert "simulated consolidation failure" in failed_round["failure_reason"]

        monkeypatch.setattr(kernel, "_complete_evolution", original_complete)
        retried = client.post(
            "/api/v1/capture/consolidate",
            headers=headers,
            json={"session_id": "failed-session"},
        )
        assert retried.status_code == 202, retried.text
        retried_round = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={"region_id": "status-region", "session_id": "failed-session"},
        ).json()[0]
        assert retried_round["distillation_status"] == "completed"
        assert retried_round["failure_reason"] == ""


def test_capture_read_parameters_remain_backward_compatible_and_validated(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "project"
    root.mkdir()

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        _region(client, headers, "project", "validation-region")
        _bind(client, headers, root, "validation-region")
        _capture(
            client,
            headers,
            event_id="validation-event",
            session_id="validation-session",
            turn_id="validation-turn",
            kind="user",
            content="参数校验",
            cwd=root,
            occurred_at="2026-10-27T05:00:00Z",
        )

        all_events = client.get("/api/v1/capture/events", headers=headers)
        assert all_events.status_code == 200, all_events.text
        assert [item["event_id"] for item in all_events.json()] == ["validation-event"]

        for params in (
            {"session_id": "   "},
            {"region_id": "   "},
            {"turn_id": "   "},
            {"session_id": "s" * 257},
            {"region_id": "r" * 121},
            {"turn_id": "t" * 201},
        ):
            invalid = client.get(
                "/api/v1/capture/events",
                headers=headers,
                params=params,
            )
            assert invalid.status_code == 422, (params, invalid.text)

        missing_region = client.get(
            "/api/v1/capture/events",
            headers=headers,
            params={"region_id": "missing-region"},
        )
        assert missing_region.status_code == 404, missing_region.text
