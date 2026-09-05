from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.model_client import ModelEndpoint
from personal_agent_memory.state import MemoryMutationError, PlatformState


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _client(tmp_path: Path, initial: str) -> tuple[TestClient, Path, Path]:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    (root / "fact.md").write_text(initial, encoding="utf-8")
    return (
        TestClient(create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))),
        state_dir,
        root,
    )


def _register(client: TestClient, headers: dict[str, str], root: Path) -> str:
    response = client.post(
        "/api/v1/libraries",
        headers=headers,
        json={"path": str(root), "kind": "project"},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _candidate(
    client: TestClient, headers: dict[str, str], library_id: str, key: str, body: str
) -> dict[str, object]:
    response = client.post(
        "/mcp/candidates",
        headers=headers,
        json={
            "library_id": library_id,
            "suggested_type": "decision",
            "body": body,
            "source_references": [f"codex-session:{key}#assistant-final"],
            "creator": "codex",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 201
    return response.json()


def _resolve(
    client: TestClient,
    headers: dict[str, str],
    candidate_id: object,
    action: str,
    key: str,
    **extra: object,
) -> dict[str, object]:
    response = client.post(
        f"/api/v1/candidates/{candidate_id}/resolve",
        headers=headers,
        json={
            "action": action,
            "operator": "epq",
            "reason": f"Resolve by {action}",
            "operation_id": key,
            **extra,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _transaction_state(
    state_dir: Path, library_id: str, candidate_id: object
) -> dict[str, object]:
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        return {
            "documents": connection.execute(
                "SELECT * FROM memory_documents WHERE library_id = ? ORDER BY id",
                (library_id,),
            ).fetchall(),
            "chunks": connection.execute(
                "SELECT * FROM memory_chunks WHERE library_id = ? ORDER BY id", (library_id,)
            ).fetchall(),
            "search": connection.execute(
                "SELECT search.* FROM memory_chunk_search AS search "
                "JOIN memory_chunks AS chunk ON chunk.id = search.chunk_id "
                "WHERE chunk.library_id = ? ORDER BY search.chunk_id",
                (library_id,),
            ).fetchall(),
            "versions": connection.execute(
                "SELECT * FROM memory_versions WHERE library_id = ? ORDER BY version_id",
                (library_id,),
            ).fetchall(),
            "candidate": connection.execute(
                "SELECT * FROM candidate_memories WHERE id = ?", (candidate_id,)
            ).fetchone(),
            "governance": connection.execute(
                "SELECT * FROM candidate_governance WHERE candidate_id = ?", (candidate_id,)
            ).fetchone(),
            "audit": connection.execute(
                "SELECT * FROM candidate_audit WHERE candidate_id = ? ORDER BY id",
                (candidate_id,),
            ).fetchall(),
            "graph_index": connection.execute(
                "SELECT * FROM memory_graph_indexes WHERE library_id = ?", (library_id,)
            ).fetchone(),
            "graph_documents": connection.execute(
                "SELECT * FROM memory_graph_documents WHERE library_id = ? ORDER BY document_id",
                (library_id,),
            ).fetchall(),
            "rebuild_jobs": connection.execute(
                "SELECT * FROM background_jobs "
                "WHERE kind IN ('vector_rebuild', 'graph_rebuild') "
                "AND json_extract(payload, '$.library_id') = ? ORDER BY id",
                (library_id,),
            ).fetchall(),
        }


def _git_state(state_dir: Path, library_id: str, root: Path) -> tuple[str, str]:
    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        git_dir = str(
            connection.execute(
                "SELECT git_dir FROM library_git_repositories WHERE library_id = ?",
                (library_id,),
            ).fetchone()[0]
        )
    head = subprocess.run(
        ["git", f"--git-dir={git_dir}", f"--work-tree={root}", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            f"--work-tree={root}",
            "status",
            "--porcelain=v2",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return head, status


def _assert_secret_not_persisted(root: Path, secret: str) -> None:
    encoded = secret.encode()
    for path in root.rglob("*"):
        if path.is_file():
            assert encoded not in path.read_bytes(), path


def test_exact_duplicate_augments_provenance_in_one_commit(tmp_path: Path) -> None:
    client, state_dir, root = _client(
        tmp_path, "# Database\n\nThe application database is PostgreSQL.\n"
    )
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        before = client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()
        candidate = _candidate(
            client,
            headers,
            library_id,
            "same-db",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        case = client.get(
            f"/api/v1/candidates/{candidate['id']}/governance", headers=headers
        ).json()
        assert case["classification"] == "exact_duplicate"
        decision = {
            "operator": "epq",
            "reason": "Equivalent confirmed",
            "operation_id": "augment-same-db",
        }
        first = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve", headers=headers, json=decision
        )
        retry = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve", headers=headers, json=decision
        )
        assert first.status_code == retry.status_code == 200
        assert first.json() == retry.json()
        assert list(root.glob("*.md")) == [root / "fact.md"]
        assert "codex-session:same-db#assistant-final" in (root / "fact.md").read_text()
        after = client.get(f"/api/v1/libraries/{library_id}/history", headers=headers).json()
        assert len(after) == len(before) + 1


def test_possible_duplicate_queues_without_overwrite(tmp_path: Path) -> None:
    client, state_dir, root = _client(
        tmp_path, "# Runtime\n\nUse Python 3.12 for the application runtime.\n"
    )
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = _candidate(
            client,
            headers,
            library_id,
            "similar-runtime",
            "# Runtime choice\n\nUse Python 3.12 for the main application runtime and tooling.\n",
        )
        queue = client.get(
            "/api/v1/candidate-governance?classification=possible_duplicate",
            headers=headers,
        ).json()
        assert [item["candidate"]["id"] for item in queue] == [candidate["id"]]
        assert (root / "fact.md").read_text().endswith("application runtime.\n")
        assert (
            client.post(
                f"/api/v1/candidates/{candidate['id']}/approve",
                headers=headers,
                json={
                    "operator": "epq",
                    "reason": "not enough",
                    "operation_id": "unsafe-auto-merge",
                },
            ).status_code
            == 422
        )


def test_direct_conflict_supports_keep_and_scoped_coexistence(tmp_path: Path) -> None:
    client, state_dir, root = _client(
        tmp_path, "# Database\n\nThe application database is Microsoft SQL Server.\n"
    )
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        keep = _candidate(
            client,
            headers,
            library_id,
            "keep-conflict",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        case = client.get(f"/api/v1/candidates/{keep['id']}/governance", headers=headers).json()
        assert case["classification"] == "conflict"
        assert "Microsoft SQL Server" in case["current"]["content"]
        assert "PostgreSQL" in case["candidate"]["body"]
        assert case["current"]["source_references"] == [
            f"markdown:fact.md@{case['current']['source_version']}"
        ]
        assert case["current"]["source_created_at"] is None
        assert case["current"]["recorded_at"]
        assert case["candidate"]["source_references"] == [
            "codex-session:keep-conflict#assistant-final"
        ]
        assert case["candidate"]["created_at"]
        assert case["diff"]
        kept = _resolve(client, headers, keep["id"], "keep", "keep-current")
        assert kept["candidate"]["status"] == "rejected"

        scoped = _candidate(
            client,
            headers,
            library_id,
            "scope-conflict",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        result = _resolve(
            client,
            headers,
            scoped["id"],
            "scope",
            "scope-db",
            condition="Use PostgreSQL only for analytics deployments.",
        )
        assert result["resolution"] == "scope"
        assert result["condition"] == "Use PostgreSQL only for analytics deployments."
        assert len(list(root.glob("*.md"))) == 2


def test_conflict_resolution_targets_the_document_with_the_conflicting_fact(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    root = tmp_path / "memory"
    root.mkdir()
    database = root / "database.md"
    migration = root / "migration.md"
    database.write_text(
        "# Database\n\nThe application database is Microsoft SQL Server. "
        "This authoritative decision also documents backups, failover, auditing, and support.\n",
        encoding="utf-8",
    )
    migration_content = (
        "# Migration\n\nThe application database migration evaluates PostgreSQL compatibility.\n"
    )
    migration.write_text(migration_content, encoding="utf-8")
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = _candidate(
            client,
            headers,
            library_id,
            "target-conflicting-document",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        governance = client.get(
            f"/api/v1/candidates/{candidate['id']}/governance", headers=headers
        ).json()
        assert governance["classification"] == "conflict"
        assert governance["target_path"] == "database.md"

        resolved = _resolve(
            client,
            headers,
            candidate["id"],
            "adopt",
            "target-conflicting-document-op",
            effective_at="2020-01-01T00:00:00Z",
        )
        assert resolved["candidate"]["published_path"] == "database.md"
        assert "PostgreSQL" in database.read_text(encoding="utf-8")
        assert migration.read_text(encoding="utf-8") == migration_content


def test_adopt_and_merge_form_restart_safe_multilevel_history(tmp_path: Path) -> None:
    client, state_dir, root = _client(
        tmp_path, "# Database\n\nThe application database is Microsoft SQL Server.\n"
    )
    library_id = ""
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        postgres = _candidate(
            client,
            headers,
            library_id,
            "adopt-postgres",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        adopted = _resolve(
            client,
            headers,
            postgres["id"],
            "adopt",
            "adopt-postgres-op",
            effective_at="2020-01-01T00:00:00+00:00",
        )
        assert adopted["effective_at"] == "2020-01-01T00:00:00+00:00"
        mysql = _candidate(
            client,
            headers,
            library_id,
            "merge-mysql",
            "# Database\n\nThe application database is MySQL.\n",
        )
        merged = _resolve(
            client,
            headers,
            mysql["id"],
            "merge",
            "merge-mysql-op",
            merged_body="# Database\n\nThe application database is MariaDB.\n",
            effective_at="2020-02-01T00:00:00+00:00",
        )
        assert merged["resolution"] == "merge"
        current = client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "MariaDB"},
        ).json()
        assert current["results"]
        assert (
            client.post(
                "/api/v1/search",
                headers=headers,
                json={"library_id": library_id, "query": "PostgreSQL"},
            ).json()["results"]
            == []
        )
        history = client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "PostgreSQL", "include_history": True},
        ).json()
        assert any(item["classification"] == "historical" for item in history["results"])

    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as restarted:
        headers = _headers(state_dir)
        history = restarted.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": library_id,
                "query": "Microsoft SQL Server",
                "include_history": True,
            },
        ).json()
        assert any(item["classification"] == "historical" for item in history["results"])


def test_future_effective_at_is_rejected_without_changing_current_memory(tmp_path: Path) -> None:
    client, state_dir, root = _client(
        tmp_path, "# Database\n\nThe application database is Microsoft SQL Server.\n"
    )
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = _candidate(
            client,
            headers,
            library_id,
            "future-effective",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        before_database = _transaction_state(state_dir, library_id, candidate["id"])
        before_git = _git_state(state_dir, library_id, root)
        before_file = (root / "fact.md").read_bytes()

        response = client.post(
            f"/api/v1/candidates/{candidate['id']}/resolve",
            headers=headers,
            json={
                "action": "adopt",
                "operator": "epq",
                "reason": "Future changes must not become current early",
                "operation_id": "future-effective-op",
                "effective_at": "2099-01-01T00:00:00Z",
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "effective_at cannot be in the future"
        assert (root / "fact.md").read_bytes() == before_file
        assert _git_state(state_dir, library_id, root) == before_git
        assert _transaction_state(state_dir, library_id, candidate["id"]) == before_database


@pytest.mark.parametrize(
    ("action", "sensitive_field"),
    [
        ("keep", "operator"),
        ("keep", "reason"),
        ("keep", "operation_id"),
        ("adopt", "operation_id"),
        ("adopt", "effective_at"),
        ("merge", "operation_id"),
        ("merge", "merged_body"),
        ("scope", "operation_id"),
        ("scope", "condition"),
    ],
)
def test_resolution_rejects_sensitive_metadata_before_persistence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, action: str, sensitive_field: str
) -> None:
    client, state_dir, root = _client(
        tmp_path, "# Database\n\nThe application database is Microsoft SQL Server.\n"
    )
    secret = "API_KEY=abc1234"
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = _candidate(
            client,
            headers,
            library_id,
            f"sensitive-{action}-{sensitive_field}",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        before_database = _transaction_state(state_dir, library_id, candidate["id"])
        before_git = _git_state(state_dir, library_id, root)
        before_files = {
            path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
        }
        extra: dict[str, str] = {}
        if action in {"adopt", "merge"}:
            extra["effective_at"] = "2020-01-01T00:00:00Z"
        if action == "merge":
            extra["merged_body"] = "# Database\n\nUse PostgreSQL with the existing policy.\n"
        if action == "scope":
            extra["condition"] = "Only for analytics workloads"

        payload = {
            "action": action,
            "operator": "epq",
            "reason": "Exercise metadata governance",
            "operation_id": f"safe-{action}-{sensitive_field}",
            **extra,
        }
        payload[sensitive_field] = secret
        response = client.post(
            f"/api/v1/candidates/{candidate['id']}/resolve",
            headers=headers,
            json=payload,
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "candidate content requires sensitive review"
        assert _git_state(state_dir, library_id, root) == before_git
        assert {
            path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
        } == before_files
        assert _transaction_state(state_dir, library_id, candidate["id"]) == before_database
        _assert_secret_not_persisted(tmp_path, secret)
        assert secret not in caplog.text


def test_history_query_returns_complete_supersession_chain(tmp_path: Path) -> None:
    long_context = " ".join(f"architecture-{index}" for index in range(3_000))
    client, state_dir, root = _client(
        tmp_path,
        f"# Database\n\nThe application database is SQLite.\n\n{long_context}\n",
    )
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        postgres = _candidate(
            client,
            headers,
            library_id,
            "chain-postgres",
            f"# Database\n\nThe application database is PostgreSQL.\n\n{long_context}\n",
        )
        _resolve(
            client,
            headers,
            postgres["id"],
            "adopt",
            "chain-postgres-op",
            effective_at="2020-01-01T00:00:00Z",
        )
        mysql = _candidate(
            client,
            headers,
            library_id,
            "chain-mysql",
            f"# Database\n\nThe application database is MySQL.\n\n{long_context}\n",
        )
        _resolve(
            client,
            headers,
            mysql["id"],
            "adopt",
            "chain-mysql-op",
            effective_at="2020-02-01T00:00:00Z",
        )

        response = client.post(
            "/api/v1/search",
            headers=headers,
            json={
                "library_id": library_id,
                "query": "database SQLite",
                "include_history": True,
                "token_budget": 10_000,
            },
        )
        assert response.status_code == 200
        chain = [
            item
            for item in response.json()["results"]
            if item["source_type"] == "markdown_history"
        ]
        assert [item["chain_depth"] for item in chain] == [0, 1, 2], chain
        assert [item["history_relation"] for item in chain] == [
            "matched",
            "successor",
            "successor",
        ]
        assert [item["memory_state"] for item in chain] == [
            "superseded",
            "superseded",
            "current",
        ]
        assert [
            next(
                database
                for database in ("SQLite", "PostgreSQL", "MySQL")
                if database in item["content"]
            )
            for item in chain
        ] == ["SQLite", "PostgreSQL", "MySQL"]
        assert chain[0]["superseded_by_version_ids"] == [chain[1]["version_id"]]
        assert chain[1]["supersedes_version_id"] == chain[0]["version_id"]
        assert chain[1]["superseded_by_version_ids"] == [chain[2]["version_id"]]
        assert chain[2]["supersedes_version_id"] == chain[1]["version_id"]
        assert chain[2]["classification"] == "history_current"

        for non_matching_query in ("base lite", "database SQL"):
            non_match = client.post(
                "/api/v1/search",
                headers=headers,
                json={
                    "library_id": library_id,
                    "query": non_matching_query,
                    "include_history": True,
                    "token_budget": 10_000,
                },
            )
            assert non_match.status_code == 200
            assert [
                item
                for item in non_match.json()["results"]
                if item["source_type"] == "markdown_history"
            ] == []


def test_resolution_callback_failure_rolls_back_every_transaction_participant(
    tmp_path: Path,
) -> None:
    client, state_dir, root = _client(
        tmp_path, "# Database\n\nThe application database is Microsoft SQL Server.\n"
    )
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = _candidate(
            client,
            headers,
            library_id,
            "callback-failure",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        client.app.state.platform_state.model_client.graph = ModelEndpoint(
            "http://127.0.0.1:9", "fault-injection-graph", retries=0
        )
        before_database = _transaction_state(state_dir, library_id, candidate["id"])
        before_git = _git_state(state_dir, library_id, root)
        before_file = (root / "fact.md").read_bytes()
        before_search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "Microsoft SQL Server"},
        ).json()
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute(
                """CREATE TRIGGER fail_resolution_callback
                   BEFORE INSERT ON candidate_audit
                   WHEN NEW.action = 'approved'
                   BEGIN SELECT RAISE(ABORT, 'forced resolution callback failure'); END"""
            )

        with pytest.raises(sqlite3.IntegrityError, match="forced resolution callback failure"):
            client.post(
                f"/api/v1/candidates/{candidate['id']}/resolve",
                headers=headers,
                json={
                    "action": "adopt",
                    "operator": "epq",
                    "reason": "Exercise callback rollback",
                    "operation_id": "callback-failure-op",
                    "effective_at": "2020-03-01T00:00:00+00:00",
                },
            )

        assert (root / "fact.md").read_bytes() == before_file
        assert _git_state(state_dir, library_id, root) == before_git
        assert _transaction_state(state_dir, library_id, candidate["id"]) == before_database
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "Microsoft SQL Server"},
        ).json() == before_search
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "PostgreSQL"},
        ).json()["results"] == []


def test_post_commit_verification_failure_compensates_resolution_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, state_dir, root = _client(
        tmp_path, "# Database\n\nThe application database is Microsoft SQL Server.\n"
    )
    with client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, root)
        candidate = _candidate(
            client,
            headers,
            library_id,
            "post-commit-failure",
            "# Database\n\nThe application database is PostgreSQL.\n",
        )
        client.app.state.platform_state.model_client.graph = ModelEndpoint(
            "http://127.0.0.1:9", "fault-injection-graph", retries=0
        )
        before_database = _transaction_state(state_dir, library_id, candidate["id"])
        before_git = _git_state(state_dir, library_id, root)
        before_file = (root / "fact.md").read_bytes()
        before_search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "Microsoft SQL Server"},
        ).json()
        original_verify = PlatformState._verify_bound_document
        verify_calls = 0

        def fail_final_verification(root_path, path, bound, identity) -> None:
            nonlocal verify_calls
            verify_calls += 1
            if verify_calls == 5:
                raise MemoryMutationError("injected final verification failure")
            original_verify(root_path, path, bound, identity)

        monkeypatch.setattr(
            PlatformState, "_verify_bound_document", staticmethod(fail_final_verification)
        )
        failed = client.post(
            f"/api/v1/candidates/{candidate['id']}/resolve",
            headers=headers,
            json={
                "action": "adopt",
                "operator": "epq",
                "reason": "Exercise durable compensation",
                "operation_id": "post-commit-failure-op",
                "effective_at": "2020-04-01T00:00:00+00:00",
            },
        )
        assert failed.status_code == 422
        assert "injected final verification failure" in failed.json()["detail"]
        assert verify_calls == 5

        assert (root / "fact.md").read_bytes() == before_file
        assert _git_state(state_dir, library_id, root) == before_git
        assert _transaction_state(state_dir, library_id, candidate["id"]) == before_database
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "Microsoft SQL Server"},
        ).json() == before_search
        assert client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "PostgreSQL"},
        ).json()["results"] == []
