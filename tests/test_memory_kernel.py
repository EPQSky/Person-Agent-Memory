from __future__ import annotations

import asyncio
import fcntl
import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import ConfigurationError, Settings
from personal_agent_memory.memory_kernel import (
    AuditMemoryRequest,
    CandidateDecisionRequest,
    CandidateGovernanceListRequest,
    CandidateGovernanceRequest,
    CandidateInspectRequest,
    CandidatePinRequest,
    CandidateRecycleListRequest,
    CandidateRestoreRequest,
    CandidateUpdateRequest,
    CaptureEventsRequest,
    CaptureMemoryRequest,
    CaptureRoundsRequest,
    ClearDiagnosticsRequest,
    CreateMemoryRequest,
    DeleteMemoryRequest,
    DocumentInspectRequest,
    DocumentUpdateRequest,
    EvolveMemoryRequest,
    HistoryDiffRequest,
    KernelHealth,
    LegacyMemoryKernel,
    ListCandidatesRequest,
    ListDocumentsRequest,
    ListForgottenRequest,
    ListOutOfBandChangesRequest,
    MemoryInspectRequest,
    MemoryKernelPort,
    MemoryUpdateRequest,
    MemoryWriteRequest,
    OutOfBandChangeRequest,
    PreviewDocumentDeleteRequest,
    PreviewDocumentUpdateRequest,
    ResolveOutOfBandChangeRequest,
    ResolveRegionRequest,
    RestoreDocumentRequest,
    RestoreForgottenRequest,
    RetentionCleanupRequest,
    RetentionPolicyRequest,
    ScanLibraryRequest,
    SearchMemoryRequest,
    SensitiveQuarantineListRequest,
    SensitiveQuarantineResolutionRequest,
    UpdateRetentionPolicyRequest,
)
from personal_agent_memory.model_client import OpenAICompatibleClient
from personal_agent_memory.state import PlatformState


class ContractFakeKernel:
    kind = "mem2"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _called(self, name: str) -> None:
        self.calls.append(name)

    def health(self) -> KernelHealth:
        return KernelHealth(kind=self.kind, status="ready", detail="contract fake")

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]:
        self._called(f"write:{type(request).__name__}")
        return {"operation": "write", "request": type(request).__name__}

    async def search(self, request: SearchMemoryRequest) -> dict[str, object]:
        self._called("search")
        return {
            "status": "unbound",
            "scope": {"kind": "explicit_library", "library_id": request.library_id},
            "query": request.query,
            "results": [],
        }

    def inspect(self, request: MemoryInspectRequest) -> dict[str, object]:
        self._called(f"inspect:{type(request).__name__}")
        if isinstance(request, DocumentInspectRequest):
            return {
                "library_id": request.library_id,
                "path": request.path,
                "content": "# Document\n",
                "source_version": "revision-one",
            }
        return {"id": request.memory_id, "body": "# Candidate\n"}

    async def update(self, request: MemoryUpdateRequest) -> dict[str, object]:
        self._called(f"update:{type(request).__name__}")
        if isinstance(request, DocumentUpdateRequest):
            return {
                "library_id": request.library_id,
                "path": request.path,
                "content": request.content,
                "source_version": "revision-two",
            }
        return {"id": request.memory_id, "body": request.content}

    async def delete(self, request: DeleteMemoryRequest) -> dict[str, str]:
        self._called("delete")
        return {"status": "deleted", "path": request.path}

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]:
        self._called("audit")
        return [{"library_id": request.library_id, "action": "fake"}]

    async def evolve(self, request: EvolveMemoryRequest) -> dict[str, int]:
        self._called("evolve")
        return {"queued": 1}

    def resolve_region(self, request: ResolveRegionRequest) -> dict[str, str]:
        self._called("resolve_region")
        return {"status": "unbound", "cwd": request.cwd}

    async def scan_library(
        self, request: ScanLibraryRequest
    ) -> dict[str, int | str]:
        self._called("scan_library")
        return {"library_id": request.library_id, "documents": 0, "changed": 0}

    def list_out_of_band_changes(
        self, request: ListOutOfBandChangesRequest
    ) -> list[dict[str, object]]:
        self._called("list_out_of_band_changes")
        return []

    def out_of_band_change(
        self, request: OutOfBandChangeRequest
    ) -> dict[str, object]:
        self._called("out_of_band_change")
        return {"id": request.change_id}

    async def resolve_out_of_band_change(
        self, request: ResolveOutOfBandChangeRequest
    ) -> dict[str, str]:
        self._called("resolve_out_of_band_change")
        return {"status": "resolved", "change_id": request.change_id}

    def list_candidates(
        self, request: ListCandidatesRequest
    ) -> list[dict[str, object]]:
        self._called("list_candidates")
        return []

    def candidate_governance_queue(
        self, request: CandidateGovernanceListRequest
    ) -> list[dict[str, object]]:
        self._called("candidate_governance_queue")
        return []

    def candidate_governance(
        self, request: CandidateGovernanceRequest
    ) -> dict[str, object]:
        self._called("candidate_governance")
        return {"candidate": {"id": request.memory_id}}

    async def decide_candidate(
        self, request: CandidateDecisionRequest
    ) -> dict[str, object]:
        self._called("decide_candidate")
        return {"id": request.memory_id, "decision": request.decision}

    async def pin_candidate(
        self, request: CandidatePinRequest
    ) -> dict[str, object]:
        self._called("pin_candidate")
        return {"id": request.memory_id, "pinned": request.pinned}

    def list_recycled_candidates(
        self, request: CandidateRecycleListRequest
    ) -> list[dict[str, object]]:
        self._called("list_recycled_candidates")
        return []

    async def restore_recycled_candidate(
        self, request: CandidateRestoreRequest
    ) -> dict[str, object]:
        self._called("restore_recycled_candidate")
        return {"id": request.memory_id}

    def list_documents(
        self, request: ListDocumentsRequest
    ) -> list[dict[str, object]]:
        self._called("list_documents")
        return []

    def preview_document_update(
        self, request: PreviewDocumentUpdateRequest
    ) -> dict[str, object]:
        self._called("preview_document_update")
        return {"library_id": request.library_id, "path": request.path, "diff": ""}

    async def preview_document_delete(
        self, request: PreviewDocumentDeleteRequest
    ) -> dict[str, object]:
        self._called("preview_document_delete")
        return {"path": request.path, "preview_token": "fake-preview"}

    def list_forgotten(
        self, request: ListForgottenRequest
    ) -> list[dict[str, object]]:
        self._called("list_forgotten")
        return []

    async def restore_forgotten(
        self, request: RestoreForgottenRequest
    ) -> dict[str, str]:
        self._called("restore_forgotten")
        return {"status": "restored", "tombstone_id": request.tombstone_id}

    def history_diff(self, request: HistoryDiffRequest) -> dict[str, object]:
        self._called("history_diff")
        return {"library_id": request.library_id, "commit": request.commit, "diff": ""}

    async def restore_document(
        self, request: RestoreDocumentRequest
    ) -> dict[str, str]:
        self._called("restore_document")
        return {"status": "restored", "path": request.path}

    def get_retention_policy(
        self, request: RetentionPolicyRequest
    ) -> dict[str, object]:
        self._called("get_retention_policy")
        return {"library_id": request.library_id}

    async def update_retention_policy(
        self, request: UpdateRetentionPolicyRequest
    ) -> dict[str, object]:
        self._called("update_retention_policy")
        return {"library_id": request.library_id}

    async def preview_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        self._called("preview_retention_cleanup")
        return {"library_id": request.library_id, "counts": {}}

    async def run_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        self._called("run_retention_cleanup")
        return {"library_id": request.library_id, "counts": {}}

    def clear_diagnostics(self, request: ClearDiagnosticsRequest) -> int:
        self._called("clear_diagnostics")
        return 0

    def list_capture_events(
        self, request: CaptureEventsRequest
    ) -> list[dict[str, object]]:
        self._called("list_capture_events")
        return []

    def list_capture_rounds(
        self, request: CaptureRoundsRequest
    ) -> list[dict[str, object]]:
        self._called("list_capture_rounds")
        return []

    def list_sensitive_quarantine(
        self, request: SensitiveQuarantineListRequest
    ) -> list[dict[str, object]]:
        self._called("list_sensitive_quarantine")
        return []

    def resolve_sensitive_quarantine(
        self, request: SensitiveQuarantineResolutionRequest
    ) -> dict[str, object]:
        self._called("resolve_sensitive_quarantine")
        return {"id": request.record_id, "resolution": request.resolution}


def test_fake_kernel_satisfies_complete_memory_kernel_contract() -> None:
    asyncio.run(_fake_kernel_contract())


async def _fake_kernel_contract() -> None:
    kernel = ContractFakeKernel()
    assert isinstance(kernel, MemoryKernelPort)

    assert kernel.health().status == "ready"
    await kernel.write(
        CreateMemoryRequest(
            library_id="library-one",
            content="# Decision\n\nUse the port.\n",
            memory_type="decision",
            source_references=("codex-session:one#assistant-final",),
            actor="codex",
            idempotency_key="fake-write",
        )
    )
    await kernel.write(
        CaptureMemoryRequest(
            event_id="event-one",
            session_id="session-one",
            turn_id="turn-one",
            event_kind="user",
            content="Capture through the port.",
            occurred_at="2026-10-24T00:00:00Z",
            cwd="/projects/one",
        )
    )
    await kernel.search(SearchMemoryRequest(query="port", library_id="library-one"))
    kernel.inspect(DocumentInspectRequest(library_id="library-one", path="memory.md"))
    kernel.inspect(CandidateInspectRequest(memory_id="candidate-one"))
    await kernel.update(
        DocumentUpdateRequest(
            library_id="library-one",
            path="memory.md",
            content="# Updated\n",
            expected_revision="revision-one",
            operation_id="document-update",
            actor="user",
            source="web",
        )
    )
    await kernel.update(
        CandidateUpdateRequest(
            memory_id="candidate-one",
            content="# Updated candidate\n",
            actor="user",
            reason="Correct the candidate",
        )
    )
    await kernel.delete(
        DeleteMemoryRequest(
            library_id="library-one",
            path="memory.md",
            expected_revision="revision-one",
            operation_id="document-delete",
            actor="user",
            source="web",
            preview_token="preview-token",
        )
    )
    kernel.audit(AuditMemoryRequest(library_id="library-one"))
    await kernel.evolve(EvolveMemoryRequest(session_id="session-one"))
    kernel.resolve_region(ResolveRegionRequest(cwd="/projects/one"))

    assert kernel.calls == [
        "write:CreateMemoryRequest",
        "write:CaptureMemoryRequest",
        "search",
        "inspect:DocumentInspectRequest",
        "inspect:CandidateInspectRequest",
        "update:DocumentUpdateRequest",
        "update:CandidateUpdateRequest",
        "delete",
        "audit",
        "evolve",
        "resolve_region",
    ]


def test_app_uses_injected_kernel_contract_and_reports_health(tmp_path: Path) -> None:
    kernel = ContractFakeKernel()
    app = create_app(
        Settings(state_dir=tmp_path / "state", memory_kernel="mem2"),
        memory_kernel=kernel,
    )
    with TestClient(app) as client:
        key = (tmp_path / "state" / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}

        assert client.get("/health/live", headers=headers).json() == {"status": "ok"}
        status = client.get("/api/v1/status", headers=headers)
        assert status.status_code == 200
        assert status.json()["memory_kernel"] == "mem2"
        assert status.json()["memory_kernel_status"] == "ready"
        assert status.json()["memory_kernel_detail"] == "contract fake"


def test_feature_flag_defaults_to_legacy_and_starts_mem2_runtime(tmp_path: Path) -> None:
    legacy_app = create_app(Settings(state_dir=tmp_path / "legacy-state"))
    assert legacy_app.state.memory_kernel.kind == "legacy"
    with TestClient(legacy_app) as client:
        key = (tmp_path / "legacy-state" / "api-key").read_text(encoding="utf-8").strip()
        status = client.get(
            "/api/v1/status", headers={"Authorization": f"Bearer {key}"}
        ).json()
        assert status["status"] == "ready"
        assert status["memory_kernel"] == "legacy"
        assert status["memory_kernel_status"] == "ready"

    mem2_app = create_app(
        Settings(state_dir=tmp_path / "mem2-state", memory_kernel="mem2")
    )
    assert mem2_app.state.memory_kernel.kind == "mem2"
    with TestClient(mem2_app) as client:
        key = (tmp_path / "mem2-state" / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}
        status = client.get("/api/v1/status", headers=headers).json()
        assert status["status"] == "degraded"
        assert status["memory_kernel"] == "mem2"
        assert status["memory_kernel_status"] == "degraded"
        assert "embedding endpoint is not configured" in status["memory_kernel_detail"]
        searched = client.post(
            "/api/v1/search",
            headers=headers,
            json={"library_id": "library-one", "query": "kernel"},
        )
        assert searched.status_code == 200
        assert searched.json()["results"] == []


def test_feature_flag_rejects_unknown_kernel(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="memory kernel"):
        Settings(state_dir=tmp_path, memory_kernel="unknown")  # type: ignore[arg-type]


def test_legacy_kernel_contract_covers_port_operations(tmp_path: Path) -> None:
    async def scenario() -> None:
        library_path = tmp_path / "library"
        project_root = tmp_path / "project"
        library_path.mkdir()
        project_root.mkdir()
        (library_path / "memory.md").write_text(
            "# Project\n\nLegacyKernelFact is reachable through the port.\n",
            encoding="utf-8",
        )
        state = PlatformState(
            tmp_path / "platform.sqlite3",
            library_roots=(tmp_path,),
            model_client=OpenAICompatibleClient(None, None),
        )
        await state.start()
        try:
            library = state.register_library(str(library_path), "project")
            binding = state.bind_project(str(project_root), library.id)
            kernel = LegacyMemoryKernel(state)

            assert kernel.health().kind == "legacy"
            assert kernel.health().status == "ready"
            resolved = kernel.resolve_region(ResolveRegionRequest(cwd=str(project_root)))
            assert resolved["status"] == "bound"
            assert resolved["binding_id"] == binding.id

            search = await kernel.search(
                SearchMemoryRequest(cwd=str(project_root), query="LegacyKernelFact")
            )
            assert [item["path"] for item in search["results"]] == ["memory.md"]

            document = kernel.inspect(
                DocumentInspectRequest(library_id=library.id, path="memory.md")
            )
            updated = await kernel.update(
                DocumentUpdateRequest(
                    library_id=library.id,
                    path="memory.md",
                    content="# Project\n\nLegacyKernelFact updated through the port.\n",
                    expected_revision=document["source_version"],
                    operation_id="kernel-document-update",
                    actor="user",
                    source="kernel-contract",
                )
            )
            assert updated["path"] == "memory.md"
            history = kernel.audit(AuditMemoryRequest(library_id=library.id))
            assert history

            candidate = await kernel.write(
                CreateMemoryRequest(
                    library_id=library.id,
                    content="# Decision\n\nCandidate written through the port.\n",
                    memory_type="decision",
                    source_references=("codex-session:contract#assistant-final",),
                    actor="codex",
                    idempotency_key="kernel-candidate-write",
                )
            )
            candidate_id = str(candidate["id"])
            assert kernel.inspect(
                CandidateInspectRequest(memory_id=candidate_id)
            )["id"] == candidate_id
            edited = await kernel.update(
                CandidateUpdateRequest(
                    memory_id=candidate_id,
                    content="# Decision\n\nCandidate corrected through the port.\n",
                    actor="user",
                    reason="Contract correction",
                )
            )
            assert "corrected" in str(edited["body"])

            empty_evolution = await kernel.evolve(EvolveMemoryRequest())
            assert empty_evolution == {"queued": 0}
            captured = await kernel.write(
                CaptureMemoryRequest(
                    event_id="kernel-capture",
                    session_id="kernel-session",
                    turn_id="kernel-turn",
                    event_kind="user",
                    content="Capture through the legacy kernel port.",
                    occurred_at="2026-10-24T00:00:00Z",
                    cwd=str(project_root),
                )
            )
            assert captured["status"] == "accepted"

            preview = state.preview_document_deletion(
                library.id, "memory.md", str(updated["source_version"])
            )
            deleted = await kernel.delete(
                DeleteMemoryRequest(
                    library_id=library.id,
                    path="memory.md",
                    expected_revision=str(updated["source_version"]),
                    operation_id="kernel-document-delete",
                    actor="user",
                    source="kernel-contract",
                    preview_token=str(preview["preview_token"]),
                )
            )
            assert deleted["path"] == "memory.md"
        finally:
            await state.close()

    asyncio.run(scenario())


def test_all_migrated_rest_and_mcp_routes_call_injected_kernel(tmp_path: Path) -> None:
    kernel = ContractFakeKernel()
    app = create_app(
        Settings(state_dir=tmp_path / "state", memory_kernel="mem2"),
        memory_kernel=kernel,
    )
    with TestClient(app) as client:
        key = (tmp_path / "state" / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}

        def call(
            method: str,
            path: str,
            *,
            expected: int = 200,
            json: dict[str, object] | None = None,
            params: dict[str, str] | None = None,
        ) -> None:
            response = client.request(
                method,
                path,
                headers=headers,
                json=json,
                params=params,
            )
            assert response.status_code == expected, (method, path, response.text)

        call(
            "GET",
            "/api/v1/project-bindings/resolve",
            params={"cwd": "/projects/one"},
        )
        call(
            "POST",
            "/api/v1/search",
            json={"library_id": "library-one", "query": "kernel"},
        )
        call("POST", "/api/v1/libraries/library-one/scan")
        call("GET", "/api/v1/libraries/library-one/out-of-band-changes")
        call(
            "GET",
            "/api/v1/libraries/library-one/out-of-band-changes/change-one",
        )
        call(
            "POST",
            "/api/v1/libraries/library-one/out-of-band-changes/change-one/resolve",
            json={
                "action": "import",
                "operation_id": "resolve-out-of-band-one",
                "final_content": "# Resolved\n",
            },
        )
        call(
            "POST",
            "/api/v1/capture/events",
            expected=202,
            json={
                "event_id": "event-one",
                "session_id": "session-one",
                "turn_id": "turn-one",
                "event_kind": "user",
                "content": "Capture through the fake kernel.",
                "occurred_at": "2026-10-24T00:00:00Z",
                "cwd": "/projects/one",
            },
        )
        call(
            "POST",
            "/api/v1/capture/consolidate",
            expected=202,
            json={"session_id": "session-one"},
        )
        call("GET", "/api/v1/capture/events")
        call("GET", "/api/v1/capture/rounds")
        call("GET", "/api/v1/sensitive-quarantine")
        call(
            "POST",
            "/api/v1/sensitive-quarantine/record-one/resolve",
            json={"resolution": "discard"},
        )
        call("GET", "/api/v1/candidate-governance")
        call("GET", "/api/v1/candidates/candidate-one/governance")
        decision = {
            "operator": "epq",
            "reason": "Verified through the port",
            "operation_id": "decision-one",
        }
        call(
            "POST",
            "/api/v1/candidates/candidate-one/resolve",
            json={**decision, "action": "keep"},
        )
        call("GET", "/api/v1/candidates")
        call("GET", "/api/v1/candidates/candidate-one")
        call(
            "PUT",
            "/api/v1/candidates/candidate-one",
            json={
                "body": "# Candidate corrected\n",
                "operator": "epq",
                "reason": "Correction",
            },
        )
        call(
            "POST",
            "/api/v1/candidates/candidate-one/approve",
            json={**decision, "operation_id": "approve-one"},
        )
        call(
            "POST",
            "/api/v1/candidates/candidate-one/reject",
            json={**decision, "operation_id": "reject-one"},
        )
        call(
            "PUT",
            "/api/v1/candidates/candidate-one/pin",
            json={"pinned": True},
        )
        call("GET", "/api/v1/libraries/library-one/candidate-recycle-bin")
        call("POST", "/api/v1/candidates/candidate-one/restore")
        call("DELETE", "/api/v1/diagnostics")
        call("GET", "/api/v1/libraries/library-one/documents")
        call(
            "GET",
            "/api/v1/libraries/library-one/document",
            params={"path": "memory.md"},
        )
        call(
            "POST",
            "/api/v1/libraries/library-one/document/preview",
            json={
                "path": "memory.md",
                "content": "# Updated\n",
                "expected_source_version": "revision-one",
            },
        )
        call(
            "PUT",
            "/api/v1/libraries/library-one/document",
            json={
                "path": "memory.md",
                "content": "# Updated\n",
                "expected_source_version": "revision-one",
                "operation_id": "update-one",
            },
        )
        call(
            "POST",
            "/api/v1/libraries/library-one/document/delete-preview",
            json={
                "path": "memory.md",
                "expected_source_version": "revision-one",
                "operation_id": "delete-one",
            },
        )
        call(
            "DELETE",
            "/api/v1/libraries/library-one/document",
            json={
                "path": "memory.md",
                "expected_source_version": "revision-one",
                "operation_id": "delete-one",
                "preview_token": "preview-one",
            },
        )
        call("GET", "/api/v1/libraries/library-one/forgotten")
        call(
            "POST",
            "/api/v1/libraries/library-one/forgotten/restore",
            json={
                "tombstone_id": "tombstone-one",
                "commit": "commit-one",
                "operation_id": "restore-forgotten-one",
            },
        )
        call("GET", "/api/v1/libraries/library-one/history")
        call("GET", "/api/v1/libraries/library-one/history/commit-one/diff")
        call(
            "POST",
            "/api/v1/libraries/library-one/history/restore",
            json={
                "path": "memory.md",
                "commit": "commit-one",
                "expected_source_version": "revision-one",
                "operation_id": "restore-one",
            },
        )
        call("GET", "/api/v1/libraries/library-one/retention-policy")
        call(
            "PUT",
            "/api/v1/libraries/library-one/retention-policy",
            json={
                "inbox_days": 30,
                "candidate_days": 90,
                "recycle_days": 30,
                "early_inbox_cleanup": True,
                "diagnostics_enabled": True,
                "diagnostic_days": 7,
            },
        )
        call("GET", "/api/v1/libraries/library-one/retention-cleanup/preview")
        call("POST", "/api/v1/libraries/library-one/retention-cleanup")

        call(
            "POST",
            "/mcp/project-bindings/resolve",
            json={"cwd": "/projects/one"},
        )
        call(
            "POST",
            "/mcp/search",
            json={"library_id": "library-one", "query": "kernel"},
        )
        call(
            "POST",
            "/mcp/candidates",
            expected=201,
            json={
                "library_id": "library-one",
                "suggested_type": "decision",
                "body": "# Decision\n\nUse the port.\n",
                "source_references": ["codex-session:one#assistant-final"],
                "creator": "codex",
                "idempotency_key": "mcp-write-one",
            },
        )
        call("GET", "/mcp/candidates", params={"library_id": "library-one"})
        call("GET", "/mcp/candidates/candidate-one")
        for tool, arguments in (
            ("memory_get", {"library_id": "library-one", "path": "memory.md"}),
            ("candidate_list", {"library_id": "library-one"}),
        ):
            call(
                "POST",
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": tool,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": arguments},
                },
            )

    expected_calls = {
        "write:CreateMemoryRequest",
        "write:CaptureMemoryRequest",
        "search",
        "inspect:DocumentInspectRequest",
        "inspect:CandidateInspectRequest",
        "update:DocumentUpdateRequest",
        "update:CandidateUpdateRequest",
        "delete",
        "audit",
        "evolve",
        "resolve_region",
        "scan_library",
        "list_out_of_band_changes",
        "out_of_band_change",
        "resolve_out_of_band_change",
        "list_candidates",
        "candidate_governance_queue",
        "candidate_governance",
        "decide_candidate",
        "pin_candidate",
        "list_recycled_candidates",
        "restore_recycled_candidate",
        "list_documents",
        "preview_document_update",
        "preview_document_delete",
        "list_forgotten",
        "restore_forgotten",
        "history_diff",
        "restore_document",
        "get_retention_policy",
        "update_retention_policy",
        "preview_retention_cleanup",
        "run_retention_cleanup",
        "clear_diagnostics",
        "list_capture_events",
        "list_capture_rounds",
        "list_sensitive_quarantine",
        "resolve_sensitive_quarantine",
    }
    assert expected_calls <= set(kernel.calls)


def test_mem2_unavailable_matrix_does_not_fall_back_to_legacy_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    app = create_app(Settings(state_dir=state_dir, memory_kernel="mem2"))
    state = app.state.platform_state

    with TestClient(app) as client:
        key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}

        def fail_legacy_call(*args: object, **kwargs: object) -> object:
            raise AssertionError("mem2 request fell back to PlatformState")

        legacy_methods = (
            "approve_candidate",
            "candidate",
            "candidate_governance",
            "clear_diagnostics",
            "create_candidate",
            "delete_document",
            "edit_candidate",
            "edit_document",
            "history_diff",
            "ingest_capture_event",
            "library_history",
            "list_candidate_governance",
            "list_candidates",
            "list_capture_events",
            "list_capture_rounds",
            "list_documents",
            "list_forgotten_memories",
            "list_recycled_candidates",
            "list_sensitive_quarantine",
            "preview_document_deletion",
            "preview_document_edit",
            "preview_retention_cleanup",
            "reject_candidate",
            "resolve_candidate",
            "list_out_of_band_changes",
            "out_of_band_change",
            "reconcile_library",
            "resolve_out_of_band_change",
            "resolve_project_binding",
            "resolve_sensitive_quarantine",
            "restore_document",
            "restore_forgotten_memory",
            "restore_recycled_candidate",
            "retention_policy",
            "run_retention_cleanup",
            "search_library",
            "search_project",
            "set_candidate_pinned",
            "trigger_capture_consolidation",
            "update_retention_policy",
        )
        for name in legacy_methods:
            monkeypatch.setattr(state, name, fail_legacy_call)

        database = state_dir / "platform.sqlite3"
        assert not database.exists()
        assert state.connection is None

        def expect_unavailable(
            method: str,
            path: str,
            *,
            json: dict[str, object] | None = None,
            params: dict[str, str] | None = None,
        ) -> None:
            response = client.request(
                method,
                path,
                headers=headers,
                json=json,
                params=params,
            )
            assert response.status_code == 503, (method, path, response.text)
            assert "mem2 memory kernel is unavailable" in response.json()["detail"]

        status = client.get("/api/v1/status", headers=headers)
        assert status.status_code == 200
        assert status.json()["status"] == "degraded"
        assert status.json()["memory_kernel"] == "mem2"
        assert status.json()["memory_kernel_status"] == "degraded"

        decision = {
            "operator": "epq",
            "reason": "Unavailable kernel matrix",
            "operation_id": "decision-one",
        }
        document = {
            "path": "memory.md",
            "expected_source_version": "revision-one",
            "operation_id": "document-one",
        }
        unavailable_requests: list[
            tuple[str, str, dict[str, object] | None, dict[str, str] | None]
        ] = [
            ("GET", "/api/v1/candidates/candidate-one", None, None),
            (
                "GET",
                "/api/v1/libraries/library-one/document",
                None,
                {"path": "memory.md"},
            ),
            (
                "PUT",
                "/api/v1/candidates/candidate-one",
                {"body": "# Corrected\n", "operator": "epq", "reason": "Correction"},
                None,
            ),
            (
                "PUT",
                "/api/v1/libraries/library-one/document",
                {**document, "content": "# Updated\n"},
                None,
            ),
            (
                "DELETE",
                "/api/v1/libraries/library-one/document",
                {**document, "preview_token": "preview-one"},
                None,
            ),
            ("POST", "/api/v1/capture/consolidate", {"session_id": "session-one"}, None),
            ("POST", "/api/v1/libraries/library-one/scan", None, None),
            (
                "GET",
                "/api/v1/libraries/library-one/out-of-band-changes",
                None,
                None,
            ),
            (
                "GET",
                "/api/v1/libraries/library-one/out-of-band-changes/change-one",
                None,
                None,
            ),
            (
                "POST",
                "/api/v1/libraries/library-one/out-of-band-changes/change-one/resolve",
                {
                    "action": "import",
                    "operation_id": "resolve-out-of-band-one",
                    "final_content": "# Resolved\n",
                },
                None,
            ),
            ("GET", "/api/v1/capture/events", None, None),
            ("GET", "/api/v1/capture/rounds", None, None),
            ("GET", "/api/v1/sensitive-quarantine", None, None),
            (
                "POST",
                "/api/v1/sensitive-quarantine/record-one/resolve",
                {"resolution": "discard"},
                None,
            ),
            ("GET", "/api/v1/candidate-governance", None, None),
            ("GET", "/api/v1/candidates/candidate-one/governance", None, None),
            (
                "POST",
                "/api/v1/candidates/candidate-one/resolve",
                {**decision, "action": "keep"},
                None,
            ),
            ("GET", "/api/v1/candidates", None, None),
            ("GET", "/mcp/candidates", None, None),
            ("GET", "/mcp/candidates/candidate-one", None, None),
            (
                "POST",
                "/api/v1/candidates/candidate-one/approve",
                {**decision, "operation_id": "approve-one"},
                None,
            ),
            (
                "POST",
                "/api/v1/candidates/candidate-one/reject",
                {**decision, "operation_id": "reject-one"},
                None,
            ),
            (
                "PUT",
                "/api/v1/candidates/candidate-one/pin",
                {"pinned": True},
                None,
            ),
            ("GET", "/api/v1/libraries/library-one/candidate-recycle-bin", None, None),
            ("POST", "/api/v1/candidates/candidate-one/restore", None, None),
            ("DELETE", "/api/v1/diagnostics", None, None),
            ("GET", "/api/v1/libraries/library-one/documents", None, None),
            (
                "POST",
                "/api/v1/libraries/library-one/document/preview",
                {
                    "path": "memory.md",
                    "content": "# Updated\n",
                    "expected_source_version": "revision-one",
                },
                None,
            ),
            (
                "POST",
                "/api/v1/libraries/library-one/document/delete-preview",
                document,
                None,
            ),
            ("GET", "/api/v1/libraries/library-one/forgotten", None, None),
            (
                "POST",
                "/api/v1/libraries/library-one/forgotten/restore",
                {
                    "tombstone_id": "tombstone-one",
                    "commit": "commit-one",
                    "operation_id": "restore-forgotten-one",
                },
                None,
            ),
            ("GET", "/api/v1/libraries/library-one/history/commit-one/diff", None, None),
            (
                "POST",
                "/api/v1/libraries/library-one/history/restore",
                {
                    "path": "memory.md",
                    "commit": "commit-one",
                    "expected_source_version": "revision-one",
                    "operation_id": "restore-one",
                },
                None,
            ),
            ("GET", "/api/v1/libraries/library-one/retention-policy", None, None),
            (
                "PUT",
                "/api/v1/libraries/library-one/retention-policy",
                {
                    "inbox_days": 30,
                    "candidate_days": 90,
                    "recycle_days": 30,
                    "early_inbox_cleanup": True,
                    "diagnostics_enabled": True,
                    "diagnostic_days": 7,
                },
                None,
            ),
            (
                "GET",
                "/api/v1/libraries/library-one/retention-cleanup/preview",
                None,
                None,
            ),
            ("POST", "/api/v1/libraries/library-one/retention-cleanup", None, None),
        ]

        for tool, arguments in (
            ("memory_get", {"library_id": "library-one", "path": "memory.md"}),
            ("candidate_list", {"library_id": "library-one"}),
        ):
            unavailable_requests.append(
                (
                    "POST",
                    "/mcp",
                    {
                        "jsonrpc": "2.0",
                        "id": tool,
                        "method": "tools/call",
                        "params": {"name": tool, "arguments": arguments},
                    },
                    None,
                )
            )

        for method, path, payload, params in unavailable_requests:
            expect_unavailable(method, path, json=payload, params=params)

        assert not database.exists()
        assert state.connection is None


def test_mem2_start_does_not_start_workers_or_mutate_legacy_state(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_path = tmp_path / "library"
    library_path.mkdir()
    (library_path / "memory.md").write_text(
        "# Project\n\nOriginal legacy document.\n",
        encoding="utf-8",
    )

    async def seed_legacy_state() -> tuple[str, str]:
        state = PlatformState(
            state_dir / "platform.sqlite3",
            library_roots=(tmp_path,),
            model_client=OpenAICompatibleClient(None, None),
        )
        await state.start()
        try:
            library = state.register_library(str(library_path), "project")
            (library_path / "memory.md").write_text(
                "# Project\n\nPending out-of-band change.\n",
                encoding="utf-8",
            )
            state.reconcile_library(library.id)
            changes = state.list_out_of_band_changes(library.id)
            assert changes
            return library.id, str(changes[0]["id"])
        finally:
            await state.close()

    library_id, change_id = asyncio.run(seed_legacy_state())
    changed_document = "# Project\n\nChanged outside the platform after shutdown.\n"
    (library_path / "memory.md").write_text(changed_document, encoding="utf-8")
    document_digest = hashlib.sha256(changed_document.encode()).hexdigest()
    document_mtime = (library_path / "memory.md").stat().st_mtime_ns

    database = state_dir / "platform.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """INSERT INTO background_jobs
               (kind, payload, status, attempts, available_at)
               VALUES ('vector_rebuild', ?, 'pending', 0, 0)""",
            (f'{{"library_id":"{library_id}"}}',),
        )
        connection.commit()

    def read_legacy_state() -> dict[str, object]:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            return {
                "libraries": connection.execute(
                    "SELECT id, canonical_path, sync_status FROM memory_libraries "
                    "ORDER BY id"
                ).fetchall(),
                "documents": connection.execute(
                    "SELECT library_id, path, source_version FROM memory_documents "
                    "ORDER BY library_id, path"
                ).fetchall(),
                "jobs": connection.execute(
                    "SELECT kind, payload, status, attempts FROM background_jobs "
                    "ORDER BY id"
                ).fetchall(),
                "out_of_band_changes": connection.execute(
                    "SELECT id, library_id, status FROM out_of_band_changes ORDER BY id"
                ).fetchall(),
            }

    before = read_legacy_state()
    database_mtime = database.stat().st_mtime_ns

    app = create_app(Settings(state_dir=state_dir, memory_kernel="mem2"))
    with TestClient(app) as client:
        key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}
        status_response = client.get("/api/v1/status", headers=headers)
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["memory_kernel"] == "mem2"
        assert status["memory_kernel_status"] == "degraded"
        assert status["database"] == "not_started"
        assert status["background_worker"] == "not_started"
        assert app.state.platform_state.connection is None
        assert app.state.platform_state.worker_task is None
        assert app.state.platform_state.capture_worker_task is None
        assert client.get("/api/v1/libraries", headers=headers).status_code == 503
        for method, path, payload in (
            ("POST", "/api/v1/libraries/library-one/scan", None),
            (
                "GET",
                f"/api/v1/libraries/{library_id}/out-of-band-changes",
                None,
            ),
            (
                "GET",
                f"/api/v1/libraries/{library_id}/out-of-band-changes/{change_id}",
                None,
            ),
            (
                "POST",
                f"/api/v1/libraries/{library_id}/out-of-band-changes/{change_id}/resolve",
                {
                    "action": "import",
                    "operation_id": "mem2-must-not-resolve",
                    "final_content": "# Must not resolve\n",
                },
            ),
        ):
            response = client.request(
                method,
                path,
                headers=headers,
                json=payload,
            )
            assert response.status_code == 503, (method, path, response.text)
            assert "mem2 memory kernel is unavailable" in response.json()["detail"]
        time.sleep(1.2)

    assert read_legacy_state() == before
    assert database.stat().st_mtime_ns == database_mtime
    assert hashlib.sha256((library_path / "memory.md").read_bytes()).hexdigest() == document_digest
    assert (library_path / "memory.md").stat().st_mtime_ns == document_mtime


def test_legacy_default_start_still_starts_platform_workers(tmp_path: Path) -> None:
    app = create_app(Settings(state_dir=tmp_path / "state"))
    with TestClient(app) as client:
        state = app.state.platform_state
        assert state.connection is not None
        assert state.worker_task is not None
        assert state.capture_worker_task is not None
        assert state.startup_count == 1
        key = (tmp_path / "state" / "api-key").read_text(encoding="utf-8").strip()
        status = client.get(
            "/api/v1/status", headers={"Authorization": f"Bearer {key}"}
        ).json()
        assert status["database"] == "ok"
        assert status["background_worker"] == "running"
        assert status["memory_kernel"] == "legacy"

    assert state.connection is None


def _tree_snapshot(root: Path) -> dict[str, tuple[str, str]]:
    snapshot: dict[str, tuple[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_dir():
            snapshot[relative] = ("directory", "")
        else:
            snapshot[relative] = (
                "file",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return snapshot


def test_mem2_lock_routes_fail_before_creating_or_waiting_for_legacy_locks(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    app = create_app(Settings(state_dir=state_dir, memory_kernel="mem2"))
    with TestClient(app) as client:
        key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}
        lock_directory = state_dir / "locks"
        lock_directory.mkdir()
        lock_path = lock_directory / "library-one.lock"

        with lock_path.open("a+b") as held_lock:
            fcntl.flock(held_lock, fcntl.LOCK_EX)
            before = _tree_snapshot(state_dir)
            requests = (
                (
                    "POST",
                    "/api/v1/libraries/library-one/vector-index/rebuild",
                    None,
                ),
                (
                    "POST",
                    "/api/v1/libraries/library-one/graph-index/rebuild",
                    None,
                ),
                (
                    "PUT",
                    "/api/v1/libraries/library-one/ignore-rules",
                    {"patterns": []},
                ),
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                for method, path, payload in requests:
                    started = time.monotonic()
                    future = executor.submit(
                        client.request,
                        method,
                        path,
                        headers=headers,
                        json=payload,
                    )
                    try:
                        response = future.result(timeout=1)
                    except FutureTimeoutError:
                        fcntl.flock(held_lock, fcntl.LOCK_UN)
                        future.result(timeout=2)
                        pytest.fail("mem2 request waited for the external legacy lock")
                    elapsed = time.monotonic() - started
                    assert response.status_code == 503, (method, path, response.text)
                    assert elapsed < 0.5
                    assert "legacy platform state is not started" in response.json()["detail"]
                    assert _tree_snapshot(state_dir) == before
            fcntl.flock(held_lock, fcntl.LOCK_UN)


class BoundCaptureFakeKernel(ContractFakeKernel):
    def resolve_region(self, request: ResolveRegionRequest) -> dict[str, str]:
        self._called("resolve_region:bound")
        return {
            "status": "bound",
            "project_id": "project-one",
            "binding_id": "binding-one",
            "project_root": "/projects/one",
            "library_id": "legacy-library",
        }

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]:
        if isinstance(request, CaptureMemoryRequest):
            binding = self.resolve_region(ResolveRegionRequest(cwd=request.cwd))
            assert binding["status"] == "bound"
        return await super().write(request)


def test_bound_capture_with_injected_kernel_does_not_touch_legacy_state(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    kernel = BoundCaptureFakeKernel()
    app = create_app(
        Settings(state_dir=state_dir, memory_kernel="mem2"),
        memory_kernel=kernel,
    )
    with TestClient(app) as client:
        key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}
        before = _tree_snapshot(state_dir)
        response = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json={
                "event_id": "bound-fake-capture",
                "session_id": "bound-fake-session",
                "turn_id": "bound-fake-turn",
                "event_kind": "user",
                "content": "A bound fake kernel must not acquire the legacy lock.",
                "occurred_at": "2026-10-24T00:00:00Z",
                "cwd": "/projects/one",
            },
        )
        assert response.status_code == 202, response.text
        assert "write:CaptureMemoryRequest" in kernel.calls
        assert "resolve_region:bound" in kernel.calls
        assert _tree_snapshot(state_dir) == before
        assert not (state_dir / "platform.sqlite3").exists()
        assert not (state_dir / "locks").exists()
