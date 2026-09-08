from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _register(client: TestClient, headers: dict[str, str], root: Path) -> str:
    response = client.post(
        "/api/v1/libraries",
        headers=headers,
        json={"path": str(root), "kind": "project"},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _candidate_payload(library_id: str, key: str, body: str = "DurableCandidateFact") -> dict:
    return {
        "library_id": library_id,
        "suggested_type": "decision",
        "body": f"# Decision\n\n{body}\n",
        "source_references": ["codex-session:session-1#assistant-final"],
        "creator": "codex",
        "idempotency_key": key,
    }


def test_mcp_candidate_contract_is_idempotent_and_read_only(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        payload = _candidate_payload(library_id, "capture-1")

        created = client.post("/mcp/candidates", headers=headers, json=payload)
        duplicate = client.post("/mcp/candidates", headers=headers, json=payload)
        assert created.status_code == duplicate.status_code == 201
        candidate = created.json()
        assert duplicate.json() == candidate
        assert set(candidate) == {
            "id",
            "library_id",
            "suggested_type",
            "body",
            "source_references",
            "creator",
            "created_at",
            "status",
            "operator",
            "reason",
            "reviewed_at",
            "published_path",
            "commit",
        }
        assert candidate["status"] == "pending"
        assert client.get("/mcp/candidates", headers=headers).json() == [candidate]
        assert client.get(
            f"/mcp/candidates/{candidate['id']}", headers=headers
        ).json() == candidate

        conflict_payload = payload | {"body": "# Different"}
        conflict = client.post("/mcp/candidates", headers=headers, json=conflict_payload)
        assert conflict.status_code == 409
        assert "idempotency_key" in conflict.json()["detail"]
        assert client.post(
            f"/mcp/candidates/{candidate['id']}/approve", headers=headers, json={}
        ).status_code == 404
        assert client.put(
            f"/mcp/libraries/{library_id}/document", headers=headers, json={}
        ).status_code == 404

        search = client.post(
            "/mcp/search",
            headers=headers,
            json={"library_id": library_id, "query": "DurableCandidateFact"},
        )
        assert search.status_code == 200
        assert search.json()["results"] == []
        assert not list(library_root.glob("*.md"))


def test_web_edit_and_approval_publish_one_searchable_commit(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(library_id, "capture-approve"),
        ).json()
        initial_history = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()

        edited = client.put(
            f"/api/v1/candidates/{candidate['id']}",
            headers=headers,
            json={
                "body": "# Confirmed decision\n\nEditedPublishedFact\n",
                "operator": "epq",
                "reason": "Corrected wording before publication",
            },
        )
        assert edited.status_code == 200
        assert edited.json()["body"].endswith("EditedPublishedFact\n")

        decision = {
            "operator": "epq",
            "reason": "Verified against the project source",
            "operation_id": "approve-capture-1",
        }
        approved = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        )
        retry = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        )
        assert approved.status_code == retry.status_code == 200
        assert retry.json() == approved.json()
        result = approved.json()
        assert result["status"] == "approved"
        assert result["operator"] == "epq"
        assert result["reason"] == decision["reason"]
        assert result["published_path"] == f"memory-{candidate['id']}.md"
        assert result["commit"]

        published = (library_root / result["published_path"]).read_text(encoding="utf-8")
        assert "EditedPublishedFact" in published
        assert "codex-session:session-1#assistant-final" in published
        assert 'approved_by: "epq"' in published
        assert decision["reason"] in published

        search = client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": library_id, "query": "EditedPublishedFact"},
        ).json()
        assert [item["path"] for item in search["results"]] == [result["published_path"]]
        history = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        assert len(history) == len(initial_history) + 1
        assert history[0]["commit"] == result["commit"]
        assert history[0]["subject"] == f"Publish candidate memory {candidate['id']}"
        diff = client.get(
            f"/api/v1/libraries/{library_id}/history/{result['commit']}/diff",
            headers=headers,
        ).json()["diff"]
        assert f"b/{result['published_path']}" in diff
        assert "EditedPublishedFact" in diff


def test_rejection_keeps_audit_without_formal_projection(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(library_id, "capture-reject", "RejectedPrivateGuess"),
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        decision = {
            "operator": "epq",
            "reason": "Unverified assistant inference",
            "operation_id": "reject-capture-1",
        }
        rejected = client.post(
            f"/api/v1/candidates/{candidate['id']}/reject",
            headers=headers,
            json=decision,
        )
        retry = client.post(
            f"/api/v1/candidates/{candidate['id']}/reject",
            headers=headers,
            json=decision,
        )
        assert rejected.status_code == retry.status_code == 200
        assert retry.json() == rejected.json()
        assert rejected.json()["status"] == "rejected"
        assert rejected.json()["published_path"] is None
        assert rejected.json()["commit"] is None
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        assert not list(library_root.glob("*.md"))
        search = client.post(
            "/mcp/search",
            headers=headers,
            json={"library_id": library_id, "query": "RejectedPrivateGuess"},
        ).json()
        assert search["results"] == []

    with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
        audit = connection.execute(
            "SELECT action, operator, reason FROM candidate_audit WHERE candidate_id = ?",
            (candidate["id"],),
        ).fetchall()
    assert audit[-1] == ("rejected", "epq", "Unverified assistant inference")


def test_candidate_state_survives_daemon_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    settings = Settings(state_dir=state_dir, library_roots=(tmp_path,))
    with TestClient(create_app(settings)) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(library_id, "restart-candidate"),
        ).json()

    with TestClient(create_app(settings)) as restarted:
        restored = restarted.get(
            f"/mcp/candidates/{candidate['id']}", headers=headers
        )
        assert restored.status_code == 200
        assert restored.json() == candidate


def test_failed_approval_rolls_back_file_commit_index_and_candidate(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(library_id, "rollback-candidate", "RollbackCandidateFact"),
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute(
                """CREATE TRIGGER fail_candidate_approval
                   BEFORE INSERT ON candidate_audit
                   WHEN NEW.action = 'approved'
                   BEGIN SELECT RAISE(ABORT, 'forced approval failure'); END"""
            )

        decision = {
            "operator": "epq",
            "reason": "Exercise atomic compensation",
            "operation_id": "approval-that-rolls-back",
        }
        with pytest.raises(sqlite3.IntegrityError, match="forced approval failure"):
            client.post(
                f"/api/v1/candidates/{candidate['id']}/approve",
                headers=headers,
                json=decision,
            )
        assert client.get(
            f"/mcp/candidates/{candidate['id']}", headers=headers
        ).json()["status"] == "pending"
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_before
        assert not list(library_root.glob("*.md"))
        assert client.post(
            "/mcp/search",
            headers=headers,
            json={"library_id": library_id, "query": "RollbackCandidateFact"},
        ).json()["results"] == []

        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute("DROP TRIGGER fail_candidate_approval")
        approved = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "approved"


def test_approval_commit_then_raise_recovers_as_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = sqlite3.connect

    class RaiseAfterApprovalCommit(sqlite3.Connection):
        injected = False

        def commit(self) -> None:
            super().commit()
            if type(self).injected:
                return
            table = self.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type = 'table' AND name = 'candidate_memories'"""
            ).fetchone()
            if table is not None and self.execute(
                "SELECT COUNT(*) FROM candidate_memories WHERE status = 'approved'"
            ).fetchone() != (0,):
                type(self).injected = True
                raise sqlite3.OperationalError("injected failure after approval commit")

    def connect_with_failure(*args, **kwargs):
        kwargs["factory"] = RaiseAfterApprovalCommit
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect_with_failure)
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(
                library_id, "commit-then-raise", "CommitThenRaiseCandidate"
            ),
        ).json()
        history_before = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        decision = {
            "operator": "epq",
            "reason": "Verify durable commit outcome",
            "operation_id": "commit-then-raise-approval",
        }
        approved = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        )
        retry = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        )
        assert approved.status_code == retry.status_code == 200
        assert approved.json() == retry.json()
        result = approved.json()
        published = library_root / result["published_path"]
        assert result["status"] == "approved"
        assert published.read_text(encoding="utf-8").endswith(
            "# Decision\n\nCommitThenRaiseCandidate\n"
        )
        assert client.post(
            "/mcp/search",
            headers=headers,
            json={"library_id": library_id, "query": "CommitThenRaiseCandidate"},
        ).json()["results"][0]["path"] == result["published_path"]
        history = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        assert len(history) == len(history_before) + 1
        assert history[0]["commit"] == result["commit"]


def test_approved_candidate_retry_survives_later_approval(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        first_candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(library_id, "first-candidate", "FirstCandidateFact"),
        ).json()
        second_candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(library_id, "second-candidate", "SecondCandidateFact"),
        ).json()
        first_decision = {
            "operator": "epq",
            "reason": "Approve the first candidate",
            "operation_id": "approve-first-candidate",
        }
        second_decision = {
            "operator": "epq",
            "reason": "Approve the second candidate",
            "operation_id": "approve-second-candidate",
        }

        first_approval = client.post(
            f"/api/v1/candidates/{first_candidate['id']}/approve",
            headers=headers,
            json=first_decision,
        )
        second_approval = client.post(
            f"/api/v1/candidates/{second_candidate['id']}/approve",
            headers=headers,
            json=second_decision,
        )
        assert first_approval.status_code == second_approval.status_code == 200
        history_after_approvals = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()

        retry = client.post(
            f"/api/v1/candidates/{first_candidate['id']}/approve",
            headers=headers,
            json=first_decision,
        )

        assert retry.status_code == 200
        assert retry.json() == first_approval.json()
        assert client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json() == history_after_approvals
        assert history_after_approvals[0]["commit"] == second_approval.json()["commit"]
        assert history_after_approvals[1]["commit"] == first_approval.json()["commit"]


def test_approved_candidate_retry_rejects_missing_publication(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(library_id, "damaged-retry", "DamagedRetryFact"),
        ).json()
        decision = {
            "operator": "epq",
            "reason": "Initially valid approval",
            "operation_id": "damaged-retry-approval",
        }
        approved = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        ).json()
        (library_root / approved["published_path"]).unlink()

        retry = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        )
        assert retry.status_code == 422
        assert retry.json()["detail"] == "approved candidate publication is inconsistent"


@pytest.mark.parametrize("audit_damage", ["missing", "tampered", "duplicate"])
def test_approved_candidate_retry_rejects_inconsistent_audit(
    tmp_path: Path, audit_damage: str
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "memory"
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(tmp_path,)))
    ) as client:
        headers = _headers(state_dir)
        library_id = _register(client, headers, library_root)
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json=_candidate_payload(library_id, "damaged-audit", "AuditedApprovalFact"),
        ).json()
        decision = {
            "operator": "epq",
            "reason": "Approve with a durable audit trail",
            "operation_id": "damaged-audit-approval",
        }
        approved = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        )
        assert approved.status_code == 200

        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            if audit_damage == "missing":
                connection.execute(
                    "DELETE FROM candidate_audit WHERE candidate_id = ? AND action = 'approved'",
                    (candidate["id"],),
                )
            elif audit_damage == "tampered":
                connection.execute(
                    """UPDATE candidate_audit SET body = 'tampered'
                       WHERE candidate_id = ? AND action = 'approved'""",
                    (candidate["id"],),
                )
            else:
                connection.execute(
                    """INSERT INTO candidate_audit
                       (candidate_id, action, operator, reason, body)
                       SELECT candidate_id, action, operator, reason, body
                       FROM candidate_audit
                       WHERE candidate_id = ? AND action = 'approved'""",
                    (candidate["id"],),
                )

        retry = client.post(
            f"/api/v1/candidates/{candidate['id']}/approve",
            headers=headers,
            json=decision,
        )
        assert retry.status_code == 422
        assert retry.json()["detail"] == "approved candidate publication is inconsistent"


def test_management_interface_exposes_candidate_governance_controls(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(state_dir=tmp_path))) as client:
        page = client.get("/")
    assert page.status_code == 200
    assert "候选审核" in page.text
    assert 'id="candidate-approve"' in page.text
    assert 'id="candidate-reject"' in page.text
