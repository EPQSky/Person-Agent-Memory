from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.sensitive import inspect_sensitive_text
from personal_agent_memory.state import (
    SENSITIVE_QUARANTINE_MAX_BYTES,
    SENSITIVE_QUARANTINE_MAX_RECORDS,
    PlatformState,
)

ROOT = Path(__file__).parents[1]
CAPTURE_HOOK = ROOT / "plugins" / "personal-agent-memory" / "scripts" / "capture.mjs"


def _bound_client(tmp_path: Path) -> tuple[TestClient, dict[str, str], str, Path]:
    state_dir = tmp_path / "state"
    libraries = tmp_path / "libraries"
    project = tmp_path / "project"
    libraries.mkdir()
    project.mkdir()
    client = TestClient(create_app(Settings(state_dir=state_dir, library_roots=(libraries,))))
    client.__enter__()
    headers = {"Authorization": f"Bearer {(state_dir / 'api-key').read_text().strip()}"}
    library = client.post(
        "/api/v1/libraries",
        headers=headers,
        json={"path": str(libraries / "memory"), "kind": "project"},
    ).json()
    assert client.post(
        "/api/v1/project-bindings",
        headers=headers,
        json={"project_root": str(project), "library_id": library["id"]},
    ).status_code == 201
    return client, headers, str(library["id"]), project


def _event(project: Path, kind: str, content: str, suffix: str) -> dict[str, str]:
    return {
        "event_id": f"sensitive-{suffix}-{kind}",
        "session_id": "sensitive-session",
        "turn_id": suffix,
        "event_kind": kind,
        "content": content,
        "occurred_at": "2026-09-05T12:00:00Z",
        "cwd": str(project),
    }


def test_secret_patterns_include_segmented_tokens() -> None:
    assert inspect_sensitive_text("api_key=abcdefghijklmno12345") is not None
    assert inspect_sensitive_text("sk-abcde fghij klmno pqrst uvwxyz") is not None
    assert inspect_sensitive_text("-----BEGIN PRIVATE KEY-----") is not None
    assert inspect_sensitive_text("an example password placeholder") is None
    assert inspect_sensitive_text("# Secret management\n\nRotate deployment credentials.") is None
    assert inspect_sensitive_text("Passwords are configured by deployment automation.") is None
    assert inspect_sensitive_text("password: generated-at-deploy-time") is None
    assert inspect_sensitive_text("password=${DATABASE_PASSWORD}") is None
    assert inspect_sensitive_text("api_key: managed-by-vault") is None
    assert inspect_sensitive_text("password: correct-horse-battery-staple") is not None
    assert inspect_sensitive_text('DB_PASSWORD="correct horse battery staple"') is not None
    assert inspect_sensitive_text("password='s3cr3t'") is not None
    assert inspect_sensitive_text("password='generated at deploy time'") is None
    assert inspect_sensitive_text('"password": "correct horse battery staple"') is not None
    assert inspect_sensitive_text("'client_secret' = 'real client secret'") is not None
    assert inspect_sensitive_text('"password": "placeholder actual-secret"') is not None
    assert inspect_sensitive_text("'token' = 'sample-abcdefghijklmno'") is not None
    assert inspect_sensitive_text('"password": "placeholder"') is None
    assert inspect_sensitive_text(
        'DB_PASSWORD="generated at deploy time"; CLIENT_SECRET="real client secret"'
    ) is not None
    for variable in ("DB_PASSWORD", "CLIENT_SECRET", "API_KEY", "ACCESS_TOKEN"):
        assert inspect_sensitive_text(f"{variable}=correct-horse-battery-staple") is not None
    assert inspect_sensitive_text("API_KEY=abc1234") is not None
    assert inspect_sensitive_text("ACCESS_TOKEN=tok1234") is not None
    assert inspect_sensitive_text("DB_PASS=s3cr3t") is not None
    assert inspect_sensitive_text(f"AWS_SECRET_ACCESS_KEY={'a' * 40}") is not None
    assert inspect_sensitive_text("DB_PASS=${DATABASE_PASSWORD}") is None
    assert inspect_sensitive_text("AWS_SECRET_ACCESS_KEY=managed-by-vault") is None
    assert inspect_sensitive_text("DB_PASS=vault://secret/data/app#password") is None
    assert inspect_sensitive_text("DB_PASS=vault://secret/data/app?token=real#password") is not None
    assert inspect_sensitive_text(
        "DB_PASSWORD=${DATABASE_PASSWORD}\nCLIENT_SECRET=correct-horse-battery-staple"
    ) is not None
    assert inspect_sensitive_text("The deployment credential may need rotation.") is not None


def test_user_confirmation_binds_relation_value_and_scope() -> None:
    candidate = "# Decision\n\nUse PostgreSQL for the application database."

    assert PlatformState._user_confirms_fact(
        "Remember: we decided to use PostgreSQL for the application database.",
        candidate,
    )
    assert not PlatformState._user_confirms_fact(
        "Remember: PostgreSQL backups run weekly.",
        candidate,
    )
    assert not PlatformState._user_confirms_fact(
        "Remember: use PostgreSQL for the analytics database.",
        candidate,
    )


def test_local_policy_blocks_spelled_out_progress_proportions() -> None:
    assert not PlatformState._library_policy_allows(
        {
            "body": "# Progress\n\nThe migration reached seventy percent yesterday.",
        }
    )
    assert not PlatformState._library_policy_allows(
        {"body": "# Progress\n\nThe migration completed 7 of 10 steps."}
    )
    for body in (
        "The migration has 3 steps remaining.",
        "The rollout is three quarters complete.",
        "The migration is 3/4 complete.",
        "The deployment stands at 70 percent.",
        "There are 2 tasks before launch.",
        "The migration is 70 percent complete.",
        "7/10 migration tasks are complete.",
        "7 of 10 migration tasks are complete.",
        "Seven of ten migration tasks are complete.",
        "The migration is 7 out of 10 complete.",
        "We have completed seven of ten migration tasks.",
        "7 of 10 tasks are complete.",
        "Seven out of ten tasks are complete.",
        "Completed seven of ten tasks.",
        "Seven tasks out of ten are complete.",
        "7 steps out of 10 completed.",
        "Seven tasks out of ten have been completed.",
        "Tasks completed: 7 out of 10.",
        "Steps completed - seven out of ten.",
        "Twenty-one tasks out of one hundred are complete.",
        "One hundred tasks out of two hundred are complete.",
        "Tasks completed: twenty-one out of one hundred.",
        "Two hundred of three hundred tasks are complete.",
        "One hundred and twenty-one tasks out of two hundred are complete.",
        "One thousand tasks out of two thousand are complete.",
        "Tasks completed: one thousand one hundred out of two thousand.",
        "There remain 3 migration tasks.",
        "The migration has three steps remaining.",
    ):
        assert not PlatformState._library_policy_allows({"body": f"# Progress\n\n{body}"})
    assert PlatformState._library_policy_allows(
        {"body": "# Constraint\n\nAlways keep two deployment slots remaining."}
    )
    assert PlatformState._library_policy_allows(
        {
            "body": (
                "# Domain fact\n\n"
                "The service supports twenty-one workers across one hundred tenants."
            )
        }
    )


def test_unstructured_confirmation_requires_the_complete_proposition() -> None:
    assert PlatformState._user_confirms_fact(
        "Remember: I prefer concise release notes.",
        "# Preference\n\nPrefer concise release notes.",
    )
    assert not PlatformState._user_confirms_fact(
        "Remember: I prefer concise logs.",
        "# Preference\n\nPrefer concise release notes.",
    )
    assert PlatformState._user_confirms_fact(
        "Remember: we decided QualifiedDockerAutoPromotion.",
        (
            "# Captured decision\n\n"
            "user:\nRemember: we decided QualifiedDockerAutoPromotion.\n"
            "assistant:\nQualifiedDockerAutoPromotion is confirmed."
        ),
    )
    assert not PlatformState._user_confirms_fact(
        "Remember: use PostgreSQL for the application database.",
        (
            "# Decisions\n\nUse PostgreSQL for the application database, "
            "and prefer concise release notes."
        ),
    )
    assert PlatformState._user_confirms_fact(
        (
            "Remember: use PostgreSQL for the application database, "
            "and I prefer concise release notes."
        ),
        (
            "# Decisions\n\nUse PostgreSQL for the application database, "
            "and prefer concise release notes."
        ),
    )
    assert not PlatformState._user_confirms_fact(
        "Remember: use PostgreSQL for the application database.",
        (
            "# Decisions\n\nUse PostgreSQL for the application database.\n"
            "Prefer concise release notes."
        ),
    )
    assert PlatformState._user_confirms_fact(
        (
            "Remember: use PostgreSQL for the application database. "
            "I prefer concise release notes."
        ),
        (
            "# Decisions\n\nUse PostgreSQL for the application database.\n"
            "Prefer concise release notes."
        ),
    )


def test_capture_discards_secrets_and_quarantines_uncertain_content(tmp_path: Path) -> None:
    client, headers, _, project = _bound_client(tmp_path)
    try:
        secret = "ghp_abcdefghijklmnopqrstuvwxyz123456"
        discarded = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json=_event(project, "user", f"token={secret}", "secret"),
        )
        uncertain = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json=_event(project, "user", "The deployment credential may need rotation.", "maybe"),
        )
        assert discarded.json()["status"] == "discarded"
        assert uncertain.json()["status"] == "quarantined"
        assert client.get("/api/v1/capture/events", headers=headers).json() == []
        quarantine = client.get("/api/v1/sensitive-quarantine", headers=headers).json()
        assert {item["disposition"] for item in quarantine} == {"discarded", "quarantined"}
        serialized = json.dumps(quarantine)
        assert secret not in serialized
        resolved = client.post(
            f"/api/v1/sensitive-quarantine/{quarantine[0]['id']}/resolve",
            headers=headers,
            json={"resolution": "discard"},
        )
        assert resolved.status_code == 200
        assert resolved.json()["resolution"] == "discard"
    finally:
        client.__exit__(None, None, None)
    database = (tmp_path / "state" / "platform.sqlite3").read_bytes()
    assert secret.encode() not in database


def test_secret_in_capture_metadata_is_withheld_before_any_persistence(tmp_path: Path) -> None:
    client, headers, _, project = _bound_client(tmp_path)
    model_calls: list[str] = []
    client.app.state.platform_state.model_client.extract_candidate = model_calls.append
    secret = "ghp_capturemetadata123456789012"
    try:
        for field in ("event_id", "session_id", "turn_id", "occurred_at", "cwd"):
            event = _event(project, "user", "Safe capture content", f"metadata-{field}")
            event[field] = str(project / secret) if field == "cwd" else f"{field}-{secret}"
            response = client.post("/api/v1/capture/events", headers=headers, json=event)
            assert response.status_code == 202
            assert response.json() == {
                "status": "discarded",
                "event_id": "withheld",
                "duplicate": False,
            }
        assert client.get("/api/v1/capture/events", headers=headers).json() == []
        quarantine = client.get("/api/v1/sensitive-quarantine", headers=headers).json()
        assert len(quarantine) == 5
        assert secret not in json.dumps(quarantine)
        assert all(str(item["source_id"]).startswith("opaque:") for item in quarantine)
        assert model_calls == []
    finally:
        client.__exit__(None, None, None)
    assert secret.encode() not in (tmp_path / "state" / "platform.sqlite3").read_bytes()


def test_serialized_secret_assignments_are_withheld_before_inbox_and_model(
    tmp_path: Path,
) -> None:
    client, headers, _, project = _bound_client(tmp_path)
    model_calls: list[str] = []
    client.app.state.platform_state.model_client.extract_candidate = model_calls.append
    cases = (
        ("json", '"password": "json real secret"', "json real secret"),
        ("toml", "'client_secret' = 'toml real secret'", "toml real secret"),
        (
            "mixed-placeholder",
            '"password": "placeholder plus real material"',
            "placeholder plus real material",
        ),
        ("short-api-key", "API_KEY=abc1234", "abc1234"),
        ("short-access-token", "ACCESS_TOKEN=tok1234", "tok1234"),
        ("short-db-pass", "DB_PASS=s3cr3t", "s3cr3t"),
        (
            "aws-secret-access-key",
            f"AWS_SECRET_ACCESS_KEY={'a' * 40}",
            "a" * 40,
        ),
    )
    try:
        for suffix, content, _ in cases:
            response = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json=_event(project, "user", content, suffix),
            )
            assert response.status_code == 202
            assert response.json()["status"] == "discarded"
        assert client.get("/api/v1/capture/events", headers=headers).json() == []
        assert model_calls == []
    finally:
        client.__exit__(None, None, None)
    database = (tmp_path / "state" / "platform.sqlite3").read_bytes()
    for _, _, secret in cases:
        assert secret.encode() not in database


def test_managed_secret_reference_reaches_inbox_and_extraction_model(tmp_path: Path) -> None:
    client, headers, _, project = _bound_client(tmp_path)
    reference = "DB_PASS=vault://secret/data/app#password"
    model_calls: list[str] = []

    def extract(conversation: str) -> dict[str, object]:
        model_calls.append(conversation)
        return {"eligible": False}

    client.app.state.platform_state.model_client.extract_candidate = extract
    try:
        for kind, content in (("user", reference), ("assistant", "Reference confirmed.")):
            response = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json=_event(project, kind, content, "managed-vault-reference"),
            )
            assert response.status_code == 202
            assert response.json()["status"] == "accepted"
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not model_calls:
            time.sleep(0.05)
        captured = client.get("/api/v1/capture/events", headers=headers).json()
        by_kind = {str(event["event_kind"]): event for event in captured}
        assert set(by_kind) == {"user", "assistant"}
        assert by_kind["user"]["content"] == reference
        assert len(model_calls) == 1
        assert reference in model_calls[0]
        assert client.get("/api/v1/sensitive-quarantine", headers=headers).json() == []
    finally:
        client.__exit__(None, None, None)


def test_web_edit_and_markdown_import_reject_confirmed_secrets(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    libraries = tmp_path / "libraries"
    editable = libraries / "editable"
    imported = libraries / "imported"
    editable.mkdir(parents=True)
    imported.mkdir()
    original = "# Decision\n\nUse a safe value.\n"
    document = editable / "decision.md"
    document.write_text(original, encoding="utf-8")
    secret = "ghp_markdownentry123456789012345"
    (imported / "secret.md").write_text(f"# Secret\n\ntoken={secret}\n", encoding="utf-8")
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(libraries,)))
    ) as client:
        headers = {"Authorization": f"Bearer {(state_dir / 'api-key').read_text().strip()}"}
        library = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(editable), "kind": "project"},
        ).json()
        response = client.put(
            f"/api/v1/libraries/{library['id']}/document",
            headers=headers,
            json={
                "path": "decision.md",
                "content": f"# Decision\n\ntoken={secret}\n",
                "expected_source_version": hashlib.sha256(original.encode()).hexdigest(),
                "operation_id": "secret-web-edit",
                "actor_type": "user",
                "source": "web",
            },
        )
        assert response.status_code == 422
        assert document.read_text(encoding="utf-8") == original
        document.write_text(f"# Decision\n\ntoken={secret}\n", encoding="utf-8")
        rescan = client.post(
            f"/api/v1/libraries/{library['id']}/scan", headers=headers
        )
        assert rescan.status_code == 200
        assert secret not in rescan.text
        changes = client.get(
            f"/api/v1/libraries/{library['id']}/out-of-band-changes", headers=headers
        ).json()
        assert len(changes) == 1
        assert changes[0]["external"] is None
        assert changes[0]["external_withheld"] is True
        assert client.get(
            f"/api/v1/libraries/{library['id']}/documents", headers=headers
        ).json()[0]["source_version"] == hashlib.sha256(original.encode()).hexdigest()
        imported_response = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(imported), "kind": "project"},
        )
        assert imported_response.status_code == 422
        assert all(item["canonical_path"] != str(imported) for item in client.get(
            "/api/v1/libraries", headers=headers
        ).json())
    assert secret.encode() not in (state_dir / "platform.sqlite3").read_bytes()


def test_authoritative_markdown_can_discuss_secret_management_without_secret_values(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    libraries = tmp_path / "libraries"
    library_root = libraries / "security"
    library_root.mkdir(parents=True)
    original = "# Secret management\n\nRotate deployment credentials.\n"
    document = library_root / "operations.md"
    document.write_text(original, encoding="utf-8")
    with TestClient(
        create_app(Settings(state_dir=state_dir, library_roots=(libraries,)))
    ) as client:
        headers = {"Authorization": f"Bearer {(state_dir / 'api-key').read_text().strip()}"}
        response = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(library_root), "kind": "project"},
        )
        assert response.status_code == 201
        library_id = response.json()["id"]
        updated = "# Secret management\n\nRotate deployment credentials every quarter.\n"
        edit = client.put(
            f"/api/v1/libraries/{library_id}/document",
            headers=headers,
            json={
                "path": "operations.md",
                "content": updated,
                "expected_source_version": hashlib.sha256(original.encode()).hexdigest(),
                "operation_id": "update-credential-rotation-policy",
                "actor_type": "user",
                "source": "web",
            },
        )
        assert edit.status_code == 200
        assert document.read_text(encoding="utf-8") == updated


def test_candidate_secret_in_provenance_never_reaches_sqlite_or_markdown(tmp_path: Path) -> None:
    client, headers, library_id, _ = _bound_client(tmp_path)
    secret = "ghp_candidateprovenancesecret123456"
    try:
        response = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "# Decision\n\nSafe body",
                "source_references": [f"project:{secret}"],
                "creator": "codex",
                "idempotency_key": "candidate-sensitive-provenance",
            },
        )
        assert response.status_code == 422
        assert client.get("/api/v1/candidates", headers=headers).json() == []
        quarantine = client.get("/api/v1/sensitive-quarantine", headers=headers).json()
        assert secret not in json.dumps(quarantine)
    finally:
        client.__exit__(None, None, None)
    assert secret.encode() not in (tmp_path / "state" / "platform.sqlite3").read_bytes()
    assert not list((tmp_path / "libraries" / "memory").glob("*.md"))


def test_candidate_governance_metadata_cannot_persist_secrets(tmp_path: Path) -> None:
    client, headers, library_id, _ = _bound_client(tmp_path)
    secret = "ghp_governancemetadata1234567890"
    try:
        candidate = client.post(
            "/mcp/candidates",
            headers=headers,
            json={
                "library_id": library_id,
                "suggested_type": "decision",
                "body": "# Decision\n\nSafe governance body",
                "source_references": ["project:safe"],
                "creator": "codex",
                "idempotency_key": "candidate-governance-metadata",
            },
        ).json()
        for method, path, payload in (
            (
                client.put,
                f"/api/v1/candidates/{candidate['id']}",
                {"body": "# Decision\n\nEdited", "operator": "codex", "reason": secret},
            ),
            (
                client.post,
                f"/api/v1/candidates/{candidate['id']}/approve",
                {"operator": "codex", "reason": secret, "operation_id": "approve-safe"},
            ),
            (
                client.post,
                f"/api/v1/candidates/{candidate['id']}/reject",
                {"operator": "codex", "reason": secret, "operation_id": "reject-safe"},
            ),
        ):
            assert method(path, headers=headers, json=payload).status_code == 422
        assert client.get(
            f"/api/v1/candidates/{candidate['id']}", headers=headers
        ).json()["status"] == "pending"
    finally:
        client.__exit__(None, None, None)
    assert secret.encode() not in (tmp_path / "state" / "platform.sqlite3").read_bytes()
    assert not list((tmp_path / "libraries" / "memory").glob("*.md"))


def test_secret_never_reaches_extraction_model(tmp_path: Path) -> None:
    client, headers, _, project = _bound_client(tmp_path)
    calls: list[str] = []
    client.app.state.platform_state.model_client.extract_candidate = calls.append
    secret = "correct horse battery staple"
    try:
        for kind, content in (
            (
                "user",
                "Remember DB_PASSWORD=${DATABASE_PASSWORD}\n"
                f'CLIENT_SECRET="{secret}"',
            ),
            ("assistant", "Confirmed."),
        ):
            client.post(
                "/api/v1/capture/events",
                headers=headers,
                json=_event(project, kind, content, "secret-round"),
            )
        time.sleep(0.2)
        assert calls == []
        captured = client.get("/api/v1/capture/events", headers=headers).json()
        assert [event["event_kind"] for event in captured] == ["assistant"]
        assert secret not in json.dumps(captured)
    finally:
        client.__exit__(None, None, None)
    assert secret.encode() not in (tmp_path / "state" / "platform.sqlite3").read_bytes()
    assert not list((tmp_path / "libraries" / "memory").glob("*.md"))


def test_multi_proposition_candidate_stays_pending_without_complete_evidence(
    tmp_path: Path,
) -> None:
    client, headers, library_id, project = _bound_client(tmp_path)
    state = client.app.state.platform_state
    state.model_client.extract_candidate = lambda _: {
        "eligible": True,
        "suggested_type": "decision",
        "body": (
            "# Decisions\n\nUse PostgreSQL for the application database, "
            "and prefer concise release notes."
        ),
        "confidence": 1.0,
        "evidence_kind": "user_confirmed",
        "source_valid": True,
        "conflict": False,
        "policy_allowed": True,
    }
    try:
        for kind, content in (
            ("user", "Remember: use PostgreSQL for the application database."),
            ("assistant", "Use PostgreSQL and prefer concise release notes."),
        ):
            response = client.post(
                "/api/v1/capture/events",
                headers=headers,
                json=_event(project, kind, content, "multi-proposition"),
            )
            assert response.status_code == 202
        deadline = time.monotonic() + 3
        candidates: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            candidates = client.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
            if candidates:
                break
            time.sleep(0.05)
        assert len(candidates) == 1
        assert candidates[0]["status"] == "pending"
        assert not list((tmp_path / "libraries" / "memory").glob("*.md"))
    finally:
        client.__exit__(None, None, None)


def test_sensitive_quarantine_is_bounded_deduplicated_paginated_and_restart_safe(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    libraries = tmp_path / "libraries"
    project = tmp_path / "project"
    libraries.mkdir()
    project.mkdir()
    settings = Settings(state_dir=state_dir, library_roots=(libraries,))

    def open_client() -> tuple[TestClient, dict[str, str]]:
        client = TestClient(create_app(settings))
        client.__enter__()
        headers = {
            "Authorization": f"Bearer {(state_dir / 'api-key').read_text().strip()}"
        }
        return client, headers

    client, headers = open_client()
    try:
        duplicate = "password='same short secret'"
        for index in range(20):
            event = _event(project, "user", duplicate, f"duplicate-{index}")
            assert client.post(
                "/api/v1/capture/events", headers=headers, json=event
            ).status_code == 202
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM sensitive_quarantine"
            ).fetchone() == (1,)

        def submit(index: int) -> int:
            event = _event(
                project,
                "user",
                f'password="bounded secret value {index:04d}"',
                f"bounded-{index}",
            )
            return client.post(
                "/api/v1/capture/events", headers=headers, json=event
            ).status_code

        assert all(submit(index) == 202 for index in range(560))
        with ThreadPoolExecutor(max_workers=8) as executor:
            assert set(executor.map(submit, range(560, 640))) == {202}

        first_page = client.get(
            "/api/v1/sensitive-quarantine",
            headers=headers,
            params={"offset": 0, "limit": 37},
        ).json()
        second_page = client.get(
            "/api/v1/sensitive-quarantine",
            headers=headers,
            params={"offset": 37, "limit": 37},
        ).json()
        assert len(first_page) == len(second_page) == 37
        assert {item["id"] for item in first_page}.isdisjoint(
            {item["id"] for item in second_page}
        )
        assert client.get(
            "/api/v1/sensitive-quarantine",
            headers=headers,
            params={"limit": 101},
        ).status_code == 422

        resolved_id = str(first_page[0]["id"])
        unresolved_id = str(first_page[1]["id"])
        assert client.post(
            f"/api/v1/sensitive-quarantine/{resolved_id}/resolve",
            headers=headers,
            json={"resolution": "discard"},
        ).status_code == 200
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            connection.execute(
                "UPDATE sensitive_quarantine SET created_at = '2000-01-01 00:00:00' "
                "WHERE id IN (?, ?)",
                (resolved_id, unresolved_id),
            )
            connection.commit()
        assert submit(641) == 202
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            count, stored_bytes = connection.execute(
                "SELECT COUNT(*), coalesce(SUM(stored_bytes), 0) "
                "FROM sensitive_quarantine"
            ).fetchone()
            assert int(count) <= SENSITIVE_QUARANTINE_MAX_RECORDS
            assert int(stored_bytes) <= SENSITIVE_QUARANTINE_MAX_BYTES
            assert connection.execute(
                "SELECT COUNT(*) FROM sensitive_quarantine WHERE id IN (?, ?)",
                (resolved_id, unresolved_id),
            ).fetchone() == (0,)
    finally:
        client.__exit__(None, None, None)

    client, headers = open_client()
    try:
        assert len(client.get(
            "/api/v1/sensitive-quarantine", headers=headers
        ).json()) <= 100
        with sqlite3.connect(state_dir / "platform.sqlite3") as connection:
            count, stored_bytes = connection.execute(
                "SELECT COUNT(*), coalesce(SUM(stored_bytes), 0) "
                "FROM sensitive_quarantine"
            ).fetchone()
            assert int(count) <= SENSITIVE_QUARANTINE_MAX_RECORDS
            assert int(stored_bytes) <= SENSITIVE_QUARANTINE_MAX_BYTES
    finally:
        client.__exit__(None, None, None)


def test_auto_promotion_requires_every_hard_gate_and_commits_source(tmp_path: Path) -> None:
    client, headers, library_id, project = _bound_client(tmp_path)
    state = client.app.state.platform_state
    results = iter(
        (
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Decision\n\nAssistantOnlyClaim",
                "confidence": 1.0,
                "evidence_kind": "assistant_only",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Decision\n\nQualifiedAutoDecision",
                "confidence": 0.99,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
        )
    )
    state.model_client.extract_candidate = lambda _: next(results)
    try:
        for turn, user, assistant in (
            ("assistant-only", "What do you think?", "AssistantOnlyClaim is true."),
            ("qualified", "Remember: we decided to use QualifiedAutoDecision.", "Confirmed."),
        ):
            for kind, content in (("user", user), ("assistant", assistant)):
                response = client.post(
                    "/api/v1/capture/events",
                    headers=headers,
                    json=_event(project, kind, content, turn),
                )
                assert response.status_code == 202
        deadline = time.monotonic() + 4
        candidates: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            candidates = client.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
            if len(candidates) == 2 and any(item["status"] == "approved" for item in candidates):
                break
            time.sleep(0.05)
        by_body = {str(item["body"]): item for item in candidates}
        assert by_body["# Decision\n\nAssistantOnlyClaim"]["status"] == "pending"
        promoted = by_body["# Decision\n\nQualifiedAutoDecision"]
        assert promoted["status"] == "approved"
        history = client.get(
            f"/api/v1/libraries/{library_id}/history", headers=headers
        ).json()
        assert history[0]["subject"].startswith("Auto-promote candidate")
        assert "QualifiedAutoDecision" in (project.parent / "libraries" / "memory" / str(
            promoted["published_path"]
        )).read_text()
    finally:
        client.__exit__(None, None, None)


def test_model_cannot_turn_an_ordinary_question_into_user_confirmation(tmp_path: Path) -> None:
    client, headers, library_id, project = _bound_client(tmp_path)
    state = client.app.state.platform_state
    state.model_client.extract_candidate = lambda _: {
        "eligible": True,
        "suggested_type": "decision",
        "body": "# Decision\n\nUse PostgreSQL",
        "confidence": 1.0,
        "evidence_kind": "user_confirmed",
        "source_valid": True,
        "conflict": False,
        "policy_allowed": True,
    }
    try:
        for kind, content in (
            (
                "user",
                "Remember that we decided to deploy Redis. Please suggest a database.",
            ),
            ("assistant", "Use PostgreSQL."),
        ):
            client.post(
                "/api/v1/capture/events",
                headers=headers,
                json=_event(project, kind, content, "ordinary-question"),
            )
        deadline = time.monotonic() + 3
        candidates: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            candidates = client.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
            if candidates:
                break
            time.sleep(0.05)
        assert len(candidates) == 1
        assert candidates[0]["status"] == "pending"
        with sqlite3.connect(tmp_path / "state" / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_operations "
                "WHERE source LIKE 'background-auto-promotion:%'"
            ).fetchone() == (0,)
    finally:
        client.__exit__(None, None, None)


def test_model_cannot_expand_a_confirmed_preference_object(tmp_path: Path) -> None:
    client, headers, library_id, project = _bound_client(tmp_path)
    state = client.app.state.platform_state
    state.model_client.extract_candidate = lambda _: {
        "eligible": True,
        "suggested_type": "preference",
        "body": "# Preference\n\nPrefer concise release notes.",
        "confidence": 1.0,
        "evidence_kind": "user_confirmed",
        "source_valid": True,
        "conflict": False,
        "policy_allowed": True,
    }
    try:
        for kind, content in (
            ("user", "Remember: I prefer concise logs."),
            ("assistant", "Confirmed."),
        ):
            client.post(
                "/api/v1/capture/events",
                headers=headers,
                json=_event(project, kind, content, "preference-object"),
            )
        deadline = time.monotonic() + 3
        candidates: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            candidates = client.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
            if candidates:
                break
            time.sleep(0.05)
        assert len(candidates) == 1
        assert candidates[0]["status"] == "pending"
        assert not list((project.parent / "libraries" / "memory").glob("*.md"))
    finally:
        client.__exit__(None, None, None)


def test_local_policy_blocks_natural_progress_and_common_replacement_conflict(
    tmp_path: Path,
) -> None:
    client, headers, library_id, project = _bound_client(tmp_path)
    seed = client.post(
        "/mcp/candidates",
        headers=headers,
        json={
            "library_id": library_id,
            "suggested_type": "decision",
            "body": "# Decision\n\nThe application database is PostgreSQL.",
            "source_references": ["project:architecture"],
            "creator": "codex",
            "idempotency_key": "seed-postgresql-decision",
        },
    ).json()
    assert client.post(
        f"/api/v1/candidates/{seed['id']}/approve",
        headers=headers,
        json={"operator": "epq", "reason": "confirmed", "operation_id": "approve-seed"},
    ).status_code == 200
    state = client.app.state.platform_state
    results = iter(
        (
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Decision\n\nThe migration reached 70% yesterday.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Decision\n\nUse SQLite for test fixtures.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Decision\n\nThe application database is SQLite.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nThe migration completed 7 of 10 steps.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nThe migration has 3 steps remaining.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nSeven of ten migration tasks are complete.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nThe migration is 7 out of 10 complete.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nWe have completed seven of ten migration tasks.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\n7 of 10 tasks are complete.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nSeven out of ten tasks are complete.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nCompleted seven of ten tasks.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nSeven tasks out of ten are complete.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nTasks completed: 7 out of 10.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nTwenty-one tasks out of one hundred are complete.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nOne hundred tasks out of two hundred are complete.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nTasks completed: twenty-one out of one hundred.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "decision",
                "body": "# Progress\n\nTwo hundred of three hundred tasks are complete.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
            {
                "eligible": True,
                "suggested_type": "constraint",
                "body": "# Constraint\n\nAlways keep two deployment slots remaining.",
                "confidence": 1.0,
                "evidence_kind": "user_confirmed",
                "source_valid": True,
                "conflict": False,
                "policy_allowed": True,
            },
        )
    )
    state.model_client.extract_candidate = lambda _: next(results)
    try:
        for turn, user in (
            ("natural-progress", "Remember: the migration reached 70% yesterday."),
            ("scoped-test-database", "Remember: we decided to use SQLite for test fixtures."),
            (
                "replacement-conflict",
                "Remember: the application database is SQLite.",
            ),
            ("count-progress", "Remember: the migration completed 7 of 10 steps."),
            ("remaining-progress", "Remember: the migration has 3 steps remaining."),
            (
                "spelled-count-progress",
                "Remember: seven of ten migration tasks are complete.",
            ),
            (
                "out-of-progress",
                "Remember: the migration is 7 out of 10 complete.",
            ),
            (
                "completed-spelled-progress",
                "Remember: we have completed seven of ten migration tasks.",
            ),
            ("generic-count-progress", "Remember: 7 of 10 tasks are complete."),
            (
                "generic-out-of-progress",
                "Remember: seven out of ten tasks are complete.",
            ),
            (
                "generic-completed-progress",
                "Remember: completed seven of ten tasks.",
            ),
            (
                "inverted-out-of-progress",
                "Remember: seven tasks out of ten are complete.",
            ),
            (
                "label-first-progress",
                "Remember: tasks completed: 7 out of 10.",
            ),
            (
                "composite-out-of-progress",
                "Remember: twenty-one tasks out of one hundred are complete.",
            ),
            (
                "hundreds-out-of-progress",
                "Remember: one hundred tasks out of two hundred are complete.",
            ),
            (
                "composite-label-first-progress",
                "Remember: tasks completed: twenty-one out of one hundred.",
            ),
            (
                "hundreds-count-progress",
                "Remember: two hundred of three hundred tasks are complete.",
            ),
            (
                "durable-capacity-constraint",
                "Remember: always keep two deployment slots remaining.",
            ),
        ):
            for kind, content in (("user", user), ("assistant", "Confirmed.")):
                client.post(
                    "/api/v1/capture/events",
                    headers=headers,
                    json=_event(project, kind, content, turn),
                )
        deadline = time.monotonic() + 3
        candidates: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            candidates = client.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
            if len(candidates) == 19:
                break
            time.sleep(0.05)
        assert len(candidates) == 19
        by_body = {str(item["body"]): item for item in candidates}
        assert (
            by_body["# Decision\n\nThe migration reached 70% yesterday."]["status"]
            == "pending"
        )
        assert (
            by_body["# Decision\n\nUse SQLite for test fixtures."]["status"]
            == "approved"
        )
        assert (
            by_body["# Decision\n\nThe application database is SQLite."]["status"]
            == "pending"
        )
        assert (
            by_body["# Progress\n\nThe migration completed 7 of 10 steps."]["status"]
            == "pending"
        )
        assert (
            by_body["# Progress\n\nThe migration has 3 steps remaining."]["status"]
            == "pending"
        )
        assert (
            by_body["# Progress\n\nSeven of ten migration tasks are complete."]["status"]
            == "pending"
        )
        assert (
            by_body["# Progress\n\nThe migration is 7 out of 10 complete."]["status"]
            == "pending"
        )
        assert (
            by_body["# Progress\n\nWe have completed seven of ten migration tasks."]["status"]
            == "pending"
        )
        assert by_body["# Progress\n\n7 of 10 tasks are complete."]["status"] == "pending"
        assert (
            by_body["# Progress\n\nSeven out of ten tasks are complete."]["status"]
            == "pending"
        )
        assert (
            by_body["# Progress\n\nCompleted seven of ten tasks."]["status"]
            == "pending"
        )
        assert (
            by_body["# Progress\n\nSeven tasks out of ten are complete."]["status"]
            == "pending"
        )
        assert (
            by_body["# Progress\n\nTasks completed: 7 out of 10."]["status"]
            == "pending"
        )
        composite_progress = (
            "Twenty-one tasks out of one hundred are complete.",
            "One hundred tasks out of two hundred are complete.",
            "Tasks completed: twenty-one out of one hundred.",
            "Two hundred of three hundred tasks are complete.",
        )
        for body in composite_progress:
            assert by_body[f"# Progress\n\n{body}"]["status"] == "pending"
        assert (
            by_body["# Constraint\n\nAlways keep two deployment slots remaining."]["status"]
            == "approved"
        )
        published = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (tmp_path / "libraries" / "memory").glob("*.md")
        )
        assert "Seven tasks out of ten are complete." not in published
        assert "Tasks completed: 7 out of 10." not in published
        assert all(body not in published for body in composite_progress)
        with sqlite3.connect(tmp_path / "state" / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_operations "
                "WHERE source LIKE 'background-auto-promotion:%'"
            ).fetchone() == (2,)
    finally:
        client.__exit__(None, None, None)


def test_conflict_detection_matches_fact_relation_and_named_scope(tmp_path: Path) -> None:
    client, headers, library_id, _ = _bound_client(tmp_path)
    try:
        for key, body in (
            ("billing-database", "# Decision\n\nBilling service uses PostgreSQL."),
            ("application-broker", "# Decision\n\nUse Kafka for the application message broker."),
            (
                "application-database",
                "# Decision\n\nThe application database is Microsoft SQL Server.",
            ),
        ):
            seed = client.post(
                "/mcp/candidates",
                headers=headers,
                json={
                    "library_id": library_id,
                    "suggested_type": "decision",
                    "body": body,
                    "source_references": ["project:architecture"],
                    "creator": "codex",
                    "idempotency_key": key,
                },
            ).json()
            assert client.post(
                f"/api/v1/candidates/{seed['id']}/approve",
                headers=headers,
                json={
                    "operator": "epq",
                    "reason": "confirmed",
                    "operation_id": f"approve-{key}",
                },
            ).status_code == 200

        state = client.app.state.platform_state

        def conflicts(body: str) -> bool:
            assert client.portal is not None
            return client.portal.call(
                state._candidate_conflicts_with_library,
                {"library_id": library_id, "body": body},
            )

        assert not conflicts("# Decision\n\nAnalytics service uses SQLite.")
        assert conflicts("# Decision\n\nUse RabbitMQ for the application message broker.")
        assert conflicts("# Decision\n\nApplication message broker is RabbitMQ.")
        assert conflicts("# Decision\n\nThe application database is SQLite.")
        assert conflicts("# Decision\n\nThe application database is PostgreSQL.")
        assert not conflicts("# Preference\n\nPrefer concise release notes.")
    finally:
        client.__exit__(None, None, None)


def test_web_page_exposes_sensitive_quarantine_controls() -> None:
    html = (ROOT / "src" / "personal_agent_memory" / "static" / "index.html").read_text()
    for control_id in (
        "sensitive-quarantine-list",
        "sensitive-quarantine-detail",
        "sensitive-discard",
        "sensitive-acknowledge",
        "sensitive-message",
        "sensitive-previous",
        "sensitive-next",
        "sensitive-refresh",
    ):
        assert control_id in html
    assert "/api/v1/sensitive-quarantine" in html


def test_offline_hook_never_spools_secret_text(tmp_path: Path) -> None:
    secret = "correct horse battery staple"
    for variable, assignment, material in (
        ("DB_PASSWORD", f'DB_PASSWORD="{secret}"', secret),
        ("CLIENT_SECRET", f"CLIENT_SECRET='{secret}'", secret),
        ("API_KEY", "API_KEY=abc1234", "abc1234"),
        ("ACCESS_TOKEN", "ACCESS_TOKEN=tok1234", "tok1234"),
        ("SHORT_PASSWORD", "password='s3cr3t'", "s3cr3t"),
        ("DB_PASS", "DB_PASS=s3cr3t", "s3cr3t"),
        (
            "AWS_SECRET_ACCESS_KEY",
            f"AWS_SECRET_ACCESS_KEY={'a' * 40}",
            "a" * 40,
        ),
    ):
        plugin_data = tmp_path / f"plugin-{variable.casefold()}"
        event = {
            "session_id": "offline-sensitive",
            "turn_id": f"turn-{variable.casefold()}",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": "password: generated-at-deploy-time; " + assignment,
        }
        result = subprocess.run(
            ["node", str(CAPTURE_HOOK)],
            input=json.dumps(event),
            text=True,
            capture_output=True,
            timeout=4,
            env={
                **os.environ,
                "PERSONAL_AGENT_MEMORY_URL": "http://127.0.0.1:1",
                "PERSONAL_AGENT_MEMORY_API_KEY": "test-key",
                "PLUGIN_DATA": str(plugin_data),
                "PERSONAL_AGENT_MEMORY_TIMEOUT_MS": "100",
            },
            check=False,
        )
        assert result.returncode == 0
        assert not list((plugin_data / "capture" / "spool").glob("event-*.json"))
        persisted = b"".join(
            path.read_bytes()
            for path in (plugin_data / "capture").rglob("*")
            if path.is_file()
        )
        assert material.encode() not in persisted
        assert b"fingerprint" in persisted


def test_offline_hook_allows_managed_secret_references(tmp_path: Path) -> None:
    for suffix, reference in (
        ("generated", "password: generated-at-deploy-time"),
        ("vault", "DB_PASS=vault://secret/data/app#password"),
    ):
        plugin_data = tmp_path / f"plugin-{suffix}"
        result = subprocess.run(
            ["node", str(CAPTURE_HOOK)],
            input=json.dumps(
                {
                    "session_id": "offline-reference",
                    "turn_id": f"turn-reference-{suffix}",
                    "cwd": str(tmp_path),
                    "hook_event_name": "UserPromptSubmit",
                    "prompt": reference,
                }
            ),
            text=True,
            capture_output=True,
            timeout=4,
            env={
                **os.environ,
                "PERSONAL_AGENT_MEMORY_URL": "http://127.0.0.1:1",
                "PERSONAL_AGENT_MEMORY_API_KEY": "test-key",
                "PLUGIN_DATA": str(plugin_data),
                "PERSONAL_AGENT_MEMORY_TIMEOUT_MS": "100",
            },
            check=False,
        )
        assert result.returncode == 0
        spool = list((plugin_data / "capture" / "spool").glob("event-*.json"))
        assert len(spool) == 1
        assert reference in spool[0].read_text()


def test_offline_hook_never_persists_secret_capture_metadata(tmp_path: Path) -> None:
    secret = "ghp_hookmetadata1234567890123456"
    for field in ("session_id", "turn_id", "cwd", "occurred_at", "timestamp"):
        plugin_data = tmp_path / f"plugin-{field}"
        event = {
            "session_id": "offline-metadata",
            "turn_id": "turn-metadata",
            "cwd": str(tmp_path),
            "occurred_at": "2026-09-05T12:00:00Z",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Remember a safe decision.",
        }
        event[field] = str(tmp_path / secret) if field == "cwd" else f"{field}-{secret}"
        result = subprocess.run(
            ["node", str(CAPTURE_HOOK)],
            input=json.dumps(event),
            text=True,
            capture_output=True,
            timeout=4,
            env={
                **os.environ,
                "PERSONAL_AGENT_MEMORY_URL": "http://127.0.0.1:1",
                "PERSONAL_AGENT_MEMORY_API_KEY": "test-key",
                "PLUGIN_DATA": str(plugin_data),
                "PERSONAL_AGENT_MEMORY_TIMEOUT_MS": "100",
            },
            check=False,
        )
        assert result.returncode == 0
        capture_root = plugin_data / "capture"
        assert not list((capture_root / "spool").glob("event-*.json"))
        persisted = b"".join(
            path.read_bytes() for path in capture_root.rglob("*") if path.is_file()
        )
        assert secret.encode() not in persisted
        assert b"fingerprint" in persisted


def test_high_confidence_cannot_bypass_conflict_policy_or_source(tmp_path: Path) -> None:
    client, headers, library_id, project = _bound_client(tmp_path)
    state = client.app.state.platform_state
    result = {
        "eligible": True,
        "suggested_type": "decision",
        "body": "# Decision\n\nBlockedHighConfidence",
        "confidence": 1.0,
        "evidence_kind": "user_confirmed",
        "source_valid": True,
        "conflict": True,
        "policy_allowed": True,
    }
    state.model_client.extract_candidate = lambda _: result
    try:
        for kind, content in (
            ("user", "Remember: we decided BlockedHighConfidence."),
            ("assistant", "Confirmed."),
        ):
            client.post(
                "/api/v1/capture/events",
                headers=headers,
                json=_event(project, kind, content, "blocked"),
            )
        deadline = time.monotonic() + 3
        candidates: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            candidates = client.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
            if candidates:
                break
            time.sleep(0.05)
        assert candidates[0]["status"] == "pending"
        with sqlite3.connect(tmp_path / "state" / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM memory_operations "
                "WHERE source LIKE 'background-auto-promotion:%'"
            ).fetchone() == (0,)
    finally:
        client.__exit__(None, None, None)
