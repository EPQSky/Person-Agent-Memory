from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NoReturn, Protocol, TypeAlias, cast, runtime_checkable

from personal_agent_memory.config import MemoryKernelKind, Settings
from personal_agent_memory.regions import RegionKind, RegionRegistry
from personal_agent_memory.state import (
    CandidateGovernanceError,
    PlatformState,
    ProjectBindingError,
)

KernelStatus = Literal["ready", "degraded", "unavailable"]
MAX_SCOPE_SPACE_LENGTH = 256
LIBRARY_SCOPE_PREFIX = "library:"
CWD_SCOPE_PREFIX = "cwd:"
MAX_LIBRARY_ID_LENGTH = MAX_SCOPE_SPACE_LENGTH - len(LIBRARY_SCOPE_PREFIX)


def validate_library_id(library_id: str) -> str:
    """Return the normalized library identifier or reject an invalid scope."""
    identifier = library_id.strip()
    if not identifier:
        raise ValueError("library_id must not be blank")
    if len(identifier) > MAX_LIBRARY_ID_LENGTH:
        raise ValueError(
            f"library_id exceeds the {MAX_LIBRARY_ID_LENGTH}-character scope limit"
        )
    return identifier


CAPTURE_BINDING_RETRY_LIMIT = 12
CandidateDecision = Literal["approve", "reject", "resolve"]
QuarantineResolution = Literal["discard", "acknowledge"]


@dataclass(frozen=True, slots=True)
class KernelHealth:
    kind: MemoryKernelKind
    status: KernelStatus
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CreateMemoryRequest:
    library_id: str
    content: str
    memory_type: str
    source_references: tuple[str, ...]
    actor: str
    idempotency_key: str
    region_id: str | None = None


@dataclass(frozen=True, slots=True)
class CaptureMemoryRequest:
    event_id: str
    session_id: str
    turn_id: str
    event_kind: Literal["user", "assistant"]
    content: str
    occurred_at: str
    cwd: str
    deadline: float | None = None
    topic_id: str | None = None


@dataclass(frozen=True, slots=True)
class SearchMemoryRequest:
    query: str
    cwd: str | None = None
    library_id: str | None = None
    region_id: str | None = None
    region_ids: tuple[str, ...] | None = None
    limit: int = 10
    max_graph_hops: int = 1
    include_history: bool = False
    session_id: str | None = None
    topic_id: str | None = None

    def __post_init__(self) -> None:
        selectors = (
            self.cwd is not None,
            self.library_id is not None,
            self.region_id is not None,
            self.region_ids is not None,
        )
        if sum(selectors) != 1:
            raise ValueError(
                "exactly one of cwd, library_id, region_id, or region_ids is required"
            )
        if self.region_ids is not None:
            if not self.region_ids:
                raise ValueError("at least one memory region is required")
            if len(set(self.region_ids)) != len(self.region_ids):
                raise ValueError("memory regions must not contain duplicates")
        if not self.query.strip():
            raise ValueError("search query must not be empty")


@dataclass(frozen=True, slots=True)
class DocumentInspectRequest:
    library_id: str
    path: str


@dataclass(frozen=True, slots=True)
class CandidateInspectRequest:
    memory_id: str


@dataclass(frozen=True, slots=True)
class DocumentUpdateRequest:
    library_id: str
    path: str
    content: str
    expected_revision: str
    operation_id: str
    actor: str
    source: str
    mode: Literal["supersede", "overwrite"] = "supersede"


@dataclass(frozen=True, slots=True)
class CandidateUpdateRequest:
    memory_id: str
    content: str
    actor: str
    reason: str


@dataclass(frozen=True, slots=True)
class DeleteMemoryRequest:
    library_id: str
    path: str
    expected_revision: str
    operation_id: str
    actor: str
    source: str
    preview_token: str
    mode: Literal["forget", "archive", "downweight", "purge"] = "forget"


@dataclass(frozen=True, slots=True)
class AuditMemoryRequest:
    library_id: str
    limit: int = 50


@dataclass(frozen=True, slots=True)
class EvolveMemoryRequest:
    session_id: str | None = None
    deadline: float | None = None


@dataclass(frozen=True, slots=True)
class ResolveRegionRequest:
    cwd: str


@dataclass(frozen=True, slots=True)
class CreateMemoryRegionRequest:
    kind: RegionKind
    name: str
    region_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class GetMemoryRegionRequest:
    region_id: str


@dataclass(frozen=True, slots=True)
class MemoryRegionTreeRequest:
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class BindProjectRegionRequest:
    project_root: str
    region_id: str


@dataclass(frozen=True, slots=True)
class AssociateWorktreeRegionRequest:
    worktree_root: str
    main_project_binding_id: str


@dataclass(frozen=True, slots=True)
class UpdateRegionBindingRequest:
    binding_id: str
    region_id: str


@dataclass(frozen=True, slots=True)
class DeleteRegionBindingRequest:
    binding_id: str


@dataclass(frozen=True, slots=True)
class ScanLibraryRequest:
    library_id: str


@dataclass(frozen=True, slots=True)
class ListOutOfBandChangesRequest:
    library_id: str


@dataclass(frozen=True, slots=True)
class OutOfBandChangeRequest:
    library_id: str
    change_id: str


@dataclass(frozen=True, slots=True)
class ResolveOutOfBandChangeRequest:
    library_id: str
    change_id: str
    action: Literal["import", "restore"]
    operation_id: str
    actor: str
    source: str
    final_content: str | None = None
    external_path: str | None = None


@dataclass(frozen=True, slots=True)
class ListCandidatesRequest:
    library_id: str | None = None
    status: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateGovernanceListRequest:
    classification: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateGovernanceRequest:
    memory_id: str


@dataclass(frozen=True, slots=True)
class CandidateDecisionRequest:
    memory_id: str
    decision: CandidateDecision
    actor: str
    reason: str
    operation_id: str
    resolution_action: str | None = None
    merged_body: str | None = None
    effective_at: str | None = None
    condition: str | None = None


@dataclass(frozen=True, slots=True)
class CandidatePinRequest:
    memory_id: str
    pinned: bool


@dataclass(frozen=True, slots=True)
class CandidateRecycleListRequest:
    library_id: str


@dataclass(frozen=True, slots=True)
class CandidateRestoreRequest:
    memory_id: str


@dataclass(frozen=True, slots=True)
class ListDocumentsRequest:
    library_id: str


@dataclass(frozen=True, slots=True)
class PreviewDocumentUpdateRequest:
    library_id: str
    path: str
    content: str
    expected_revision: str


@dataclass(frozen=True, slots=True)
class PreviewDocumentDeleteRequest:
    library_id: str
    path: str
    expected_revision: str


@dataclass(frozen=True, slots=True)
class ListForgottenRequest:
    library_id: str


@dataclass(frozen=True, slots=True)
class RestoreForgottenRequest:
    library_id: str
    tombstone_id: str
    commit: str
    operation_id: str
    actor: str
    source: str


@dataclass(frozen=True, slots=True)
class HistoryDiffRequest:
    library_id: str
    commit: str


@dataclass(frozen=True, slots=True)
class RestoreDocumentRequest:
    library_id: str
    path: str
    commit: str
    expected_revision: str
    operation_id: str
    actor: str
    source: str


@dataclass(frozen=True, slots=True)
class RetentionPolicyRequest:
    library_id: str


@dataclass(frozen=True, slots=True)
class UpdateRetentionPolicyRequest:
    library_id: str
    inbox_days: int
    candidate_days: int
    recycle_days: int
    early_inbox_cleanup: bool
    diagnostics_enabled: bool
    diagnostic_days: int


@dataclass(frozen=True, slots=True)
class RetentionCleanupRequest:
    library_id: str


@dataclass(frozen=True, slots=True)
class ClearDiagnosticsRequest:
    pass


@dataclass(frozen=True, slots=True)
class CaptureEventsRequest:
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class CaptureRoundsRequest:
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class SensitiveQuarantineListRequest:
    offset: int = 0
    limit: int = 100


@dataclass(frozen=True, slots=True)
class SensitiveQuarantineResolutionRequest:
    record_id: str
    resolution: QuarantineResolution


MemoryWriteRequest: TypeAlias = CreateMemoryRequest | CaptureMemoryRequest
MemoryInspectRequest: TypeAlias = DocumentInspectRequest | CandidateInspectRequest
MemoryUpdateRequest: TypeAlias = DocumentUpdateRequest | CandidateUpdateRequest


@runtime_checkable
class MemoryKernelPort(Protocol):
    kind: MemoryKernelKind

    def health(self) -> KernelHealth: ...

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]: ...

    async def search(self, request: SearchMemoryRequest) -> dict[str, object]: ...

    def inspect(self, request: MemoryInspectRequest) -> dict[str, object]: ...

    async def update(self, request: MemoryUpdateRequest) -> dict[str, object]: ...

    async def delete(self, request: DeleteMemoryRequest) -> dict[str, str]: ...

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]: ...

    async def evolve(self, request: EvolveMemoryRequest) -> dict[str, int]: ...

    def resolve_region(self, request: ResolveRegionRequest) -> dict[str, str]: ...

    def create_region(self, request: CreateMemoryRegionRequest) -> dict[str, object]: ...

    def list_regions(self) -> list[dict[str, object]]: ...

    def get_region(self, request: GetMemoryRegionRequest) -> dict[str, object]: ...

    def region_tree(self, request: MemoryRegionTreeRequest) -> dict[str, object]: ...

    def bind_project_region(self, request: BindProjectRegionRequest) -> dict[str, str]: ...

    def associate_worktree_region(
        self, request: AssociateWorktreeRegionRequest
    ) -> dict[str, str]: ...

    def list_region_bindings(self) -> list[dict[str, str]]: ...

    def update_region_binding(self, request: UpdateRegionBindingRequest) -> dict[str, str]: ...

    def delete_region_binding(self, request: DeleteRegionBindingRequest) -> None: ...

    async def scan_library(self, request: ScanLibraryRequest) -> dict[str, int | str]: ...

    def list_out_of_band_changes(
        self, request: ListOutOfBandChangesRequest
    ) -> list[dict[str, object]]: ...

    def out_of_band_change(
        self, request: OutOfBandChangeRequest
    ) -> dict[str, object]: ...

    async def resolve_out_of_band_change(
        self, request: ResolveOutOfBandChangeRequest
    ) -> dict[str, str]: ...

    def list_candidates(
        self, request: ListCandidatesRequest
    ) -> list[dict[str, object]]: ...

    def candidate_governance_queue(
        self, request: CandidateGovernanceListRequest
    ) -> list[dict[str, object]]: ...

    def candidate_governance(
        self, request: CandidateGovernanceRequest
    ) -> dict[str, object]: ...

    async def decide_candidate(
        self, request: CandidateDecisionRequest
    ) -> dict[str, object]: ...

    async def pin_candidate(
        self, request: CandidatePinRequest
    ) -> dict[str, object]: ...

    def list_recycled_candidates(
        self, request: CandidateRecycleListRequest
    ) -> list[dict[str, object]]: ...

    async def restore_recycled_candidate(
        self, request: CandidateRestoreRequest
    ) -> dict[str, object]: ...

    def list_documents(
        self, request: ListDocumentsRequest
    ) -> list[dict[str, object]]: ...

    def preview_document_update(
        self, request: PreviewDocumentUpdateRequest
    ) -> dict[str, object]: ...

    async def preview_document_delete(
        self, request: PreviewDocumentDeleteRequest
    ) -> dict[str, object]: ...

    def list_forgotten(self, request: ListForgottenRequest) -> list[dict[str, object]]: ...

    async def restore_forgotten(self, request: RestoreForgottenRequest) -> dict[str, str]: ...

    def history_diff(self, request: HistoryDiffRequest) -> dict[str, object]: ...

    async def restore_document(self, request: RestoreDocumentRequest) -> dict[str, str]: ...

    def get_retention_policy(
        self, request: RetentionPolicyRequest
    ) -> dict[str, object]: ...

    async def update_retention_policy(
        self, request: UpdateRetentionPolicyRequest
    ) -> dict[str, object]: ...

    async def preview_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]: ...

    async def run_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]: ...

    def clear_diagnostics(self, request: ClearDiagnosticsRequest) -> int: ...

    def list_capture_events(
        self, request: CaptureEventsRequest
    ) -> list[dict[str, object]]: ...

    def list_capture_rounds(
        self, request: CaptureRoundsRequest
    ) -> list[dict[str, object]]: ...

    def list_sensitive_quarantine(
        self, request: SensitiveQuarantineListRequest
    ) -> list[dict[str, object]]: ...

    def resolve_sensitive_quarantine(
        self, request: SensitiveQuarantineResolutionRequest
    ) -> dict[str, object]: ...


@runtime_checkable
class ManagedMemoryKernelPort(Protocol):
    """Kernel with an owning runtime that the daemon starts and closes."""

    async def start(self) -> None: ...

    async def close(self) -> None: ...


class MemoryKernelUnavailableError(RuntimeError):
    """Raised when the selected memory kernel cannot serve an operation."""


class MemoryIdempotencyConflictError(RuntimeError):
    """Raised when an idempotency key is reused for different content."""


class MemoryIdempotencyPendingError(MemoryIdempotencyConflictError):
    """Raised when an earlier write for the idempotency key is incomplete."""


class MemoryGovernanceError(ValueError):
    """Raised when an inspected or governed memory record is invalid."""


class MemoryRecordNotFoundError(MemoryGovernanceError):
    """Raised when a requested memory record does not exist."""


class MemoryRevisionConflictError(MemoryGovernanceError):
    """Raised when a caller acts on a stale memory revision."""


class MemoryLifecycleConflictError(MemoryGovernanceError):
    """Raised when a lifecycle state does not allow the requested governance action."""


class CaptureContractError(ValueError):
    """Raised when capture input violates a memory-region contract."""


class CaptureBindingUnstableError(RuntimeError):
    """Raised when a project binding keeps changing during capture."""


class LegacyMemoryKernel:
    """Compatibility adapter for the existing memory implementation."""

    kind: MemoryKernelKind = "legacy"

    def __init__(self, state: PlatformState, regions: RegionRegistry | None = None) -> None:
        self._state = state
        self._regions = regions or RegionRegistry(state.database_path.parent / "regions.sqlite3")

    def health(self) -> KernelHealth:
        if self._state.connection is None:
            return KernelHealth(
                kind=self.kind,
                status="unavailable",
                detail="legacy platform state is not started",
            )
        return KernelHealth(kind=self.kind, status="ready", detail="legacy memory path")

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]:
        if isinstance(request, CreateMemoryRequest):
            if request.region_id is not None:
                raise ValueError("memory region writes require the mem2 memory kernel")
            async with self._state.library_write_lock(request.library_id):
                return self._state.create_candidate(
                    request.library_id,
                    request.memory_type,
                    request.content,
                    request.source_references,
                    request.actor,
                    request.idempotency_key,
                )
        if isinstance(request, CaptureMemoryRequest):
            for _ in range(CAPTURE_BINDING_RETRY_LIMIT):
                try:
                    binding = self._state.resolve_project_binding(request.cwd)
                except ProjectBindingError:
                    return self._ingest_capture(request)
                if binding["status"] != "bound":
                    return self._ingest_capture(request)
                library_id = str(binding["library_id"])
                async with self._state.library_write_lock(library_id):
                    current = self._state.resolve_project_binding(request.cwd)
                    if current.get("library_id") != library_id:
                        continue
                    return self._ingest_capture(request)
            raise CaptureBindingUnstableError(
                "project binding remained unstable while capturing event"
            )
        raise TypeError("unsupported memory write request")

    async def search(self, request: SearchMemoryRequest) -> dict[str, object]:
        if request.region_id is not None or request.region_ids is not None:
            raise ValueError("memory region search requires the mem2 memory kernel")
        if request.cwd is not None:
            return await self._state.search_project(
                request.cwd,
                request.query,
                request.limit,
                request.max_graph_hops,
                request.include_history,
            )
        return await self._state.search_library(
            request.library_id or "",
            request.query,
            request.limit,
            request.max_graph_hops,
            request.include_history,
        )

    def inspect(self, request: MemoryInspectRequest) -> dict[str, object]:
        if isinstance(request, DocumentInspectRequest):
            return cast(
                dict[str, object],
                self._state.read_document(request.library_id, request.path),
            )
        if isinstance(request, CandidateInspectRequest):
            return self._state.candidate(request.memory_id)
        raise TypeError("unsupported memory inspect request")

    async def update(self, request: MemoryUpdateRequest) -> dict[str, object]:
        if isinstance(request, CandidateUpdateRequest):
            return await self._mutate_candidate(
                request.memory_id,
                lambda: self._state.edit_candidate(
                    request.memory_id,
                    request.content,
                    request.actor,
                    request.reason,
                ),
            )
        if isinstance(request, DocumentUpdateRequest):
            async with self._state.library_write_lock(request.library_id):
                return cast(
                    dict[str, object],
                    self._state.edit_document(
                        request.library_id,
                        request.path,
                        request.content,
                        request.expected_revision,
                        request.operation_id,
                        request.actor,
                        request.source,
                    ),
                )
        raise TypeError("unsupported memory update request")

    async def delete(self, request: DeleteMemoryRequest) -> dict[str, str]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.delete_document(
                request.library_id,
                request.path,
                request.expected_revision,
                request.operation_id,
                request.actor,
                request.source,
                request.preview_token,
            )

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]:
        return self._state.library_history(request.library_id, request.limit)

    async def evolve(self, request: EvolveMemoryRequest) -> dict[str, int]:
        queued = await asyncio.to_thread(
            self._state.trigger_capture_consolidation,
            request.session_id,
            request.deadline,
        )
        return {"queued": queued}

    def resolve_region(self, request: ResolveRegionRequest) -> dict[str, str]:
        return self._state.resolve_project_binding(request.cwd)

    def create_region(self, request: CreateMemoryRegionRequest) -> dict[str, object]:
        return self._regions.create_region(
            request.kind,
            request.name,
            region_id=request.region_id,
            session_id=request.session_id,
        ).payload()

    def list_regions(self) -> list[dict[str, object]]:
        return [region.payload() for region in self._regions.list_regions()]

    def get_region(self, request: GetMemoryRegionRequest) -> dict[str, object]:
        return self._regions.require_region(request.region_id).payload()

    def region_tree(self, request: MemoryRegionTreeRequest) -> dict[str, object]:
        return self._regions.region_tree(request.session_id)

    def bind_project_region(self, request: BindProjectRegionRequest) -> dict[str, str]:
        return self._state.bind_project(request.project_root, request.region_id).payload()

    def associate_worktree_region(
        self, request: AssociateWorktreeRegionRequest
    ) -> dict[str, str]:
        return self._state.associate_worktree(
            request.worktree_root, request.main_project_binding_id
        ).payload()

    def list_region_bindings(self) -> list[dict[str, str]]:
        return [binding.payload() for binding in self._state.list_project_bindings()]

    def update_region_binding(self, request: UpdateRegionBindingRequest) -> dict[str, str]:
        return self._state.update_project_binding(
            request.binding_id, request.region_id
        ).payload()

    def delete_region_binding(self, request: DeleteRegionBindingRequest) -> None:
        self._state.delete_project_binding(request.binding_id)

    async def scan_library(
        self, request: ScanLibraryRequest
    ) -> dict[str, int | str]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.reconcile_library(request.library_id)

    def list_out_of_band_changes(
        self, request: ListOutOfBandChangesRequest
    ) -> list[dict[str, object]]:
        return self._state.list_out_of_band_changes(request.library_id)

    def out_of_band_change(
        self, request: OutOfBandChangeRequest
    ) -> dict[str, object]:
        return self._state.out_of_band_change(request.library_id, request.change_id)

    async def resolve_out_of_band_change(
        self, request: ResolveOutOfBandChangeRequest
    ) -> dict[str, str]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.resolve_out_of_band_change(
                request.library_id,
                request.change_id,
                request.action,
                request.operation_id,
                request.actor,
                request.source,
                request.final_content,
                request.external_path,
            )

    def list_candidates(
        self, request: ListCandidatesRequest
    ) -> list[dict[str, object]]:
        return self._state.list_candidates(request.library_id, request.status)

    def candidate_governance_queue(
        self, request: CandidateGovernanceListRequest
    ) -> list[dict[str, object]]:
        return self._state.list_candidate_governance(request.classification)

    def candidate_governance(
        self, request: CandidateGovernanceRequest
    ) -> dict[str, object]:
        return self._state.candidate_governance(request.memory_id)

    async def decide_candidate(
        self, request: CandidateDecisionRequest
    ) -> dict[str, object]:
        if request.decision == "approve":
            return await self._mutate_candidate(
                request.memory_id,
                lambda: self._state.approve_candidate(
                    request.memory_id,
                    request.actor,
                    request.reason,
                    request.operation_id,
                ),
            )
        if request.decision == "reject":
            return await self._mutate_candidate(
                request.memory_id,
                lambda: self._state.reject_candidate(
                    request.memory_id,
                    request.actor,
                    request.reason,
                    request.operation_id,
                ),
            )
        if request.resolution_action is None:
            raise TypeError("candidate resolution action is required")
        return await self._mutate_candidate(
            request.memory_id,
            lambda: self._state.resolve_candidate(
                request.memory_id,
                request.resolution_action or "",
                request.actor,
                request.reason,
                request.operation_id,
                merged_body=request.merged_body,
                effective_at=request.effective_at,
                condition=request.condition,
            ),
        )

    async def pin_candidate(
        self, request: CandidatePinRequest
    ) -> dict[str, object]:
        return await self._mutate_candidate(
            request.memory_id,
            lambda: self._state.set_candidate_pinned(
                request.memory_id, request.pinned
            ),
        )

    def list_recycled_candidates(
        self, request: CandidateRecycleListRequest
    ) -> list[dict[str, object]]:
        return self._state.list_recycled_candidates(request.library_id)

    async def restore_recycled_candidate(
        self, request: CandidateRestoreRequest
    ) -> dict[str, object]:
        return await self._mutate_candidate(
            request.memory_id,
            lambda: self._state.restore_recycled_candidate(request.memory_id),
        )

    def list_documents(
        self, request: ListDocumentsRequest
    ) -> list[dict[str, object]]:
        return self._state.list_documents(request.library_id)

    def preview_document_update(
        self, request: PreviewDocumentUpdateRequest
    ) -> dict[str, object]:
        return self._state.preview_document_edit(
            request.library_id,
            request.path,
            request.content,
            request.expected_revision,
        )

    async def preview_document_delete(
        self, request: PreviewDocumentDeleteRequest
    ) -> dict[str, object]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.preview_document_deletion(
                request.library_id,
                request.path,
                request.expected_revision,
            )

    def list_forgotten(
        self, request: ListForgottenRequest
    ) -> list[dict[str, object]]:
        return self._state.list_forgotten_memories(request.library_id)

    async def restore_forgotten(
        self, request: RestoreForgottenRequest
    ) -> dict[str, str]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.restore_forgotten_memory(
                request.library_id,
                request.tombstone_id,
                request.commit,
                request.operation_id,
                request.actor,
                request.source,
            )

    def history_diff(self, request: HistoryDiffRequest) -> dict[str, object]:
        return self._state.history_diff(request.library_id, request.commit)

    async def restore_document(
        self, request: RestoreDocumentRequest
    ) -> dict[str, str]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.restore_document(
                request.library_id,
                request.path,
                request.commit,
                request.expected_revision,
                request.operation_id,
                request.actor,
                request.source,
            )

    def get_retention_policy(
        self, request: RetentionPolicyRequest
    ) -> dict[str, object]:
        return self._state.retention_policy(request.library_id)

    async def update_retention_policy(
        self, request: UpdateRetentionPolicyRequest
    ) -> dict[str, object]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.update_retention_policy(
                request.library_id,
                inbox_days=request.inbox_days,
                candidate_days=request.candidate_days,
                recycle_days=request.recycle_days,
                early_inbox_cleanup=request.early_inbox_cleanup,
                diagnostics_enabled=request.diagnostics_enabled,
                diagnostic_days=request.diagnostic_days,
            )

    async def preview_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.preview_retention_cleanup(request.library_id)

    async def run_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        async with self._state.library_write_lock(request.library_id):
            return self._state.run_retention_cleanup(request.library_id)

    def clear_diagnostics(self, request: ClearDiagnosticsRequest) -> int:
        return self._state.clear_diagnostics()

    def list_capture_events(
        self, request: CaptureEventsRequest
    ) -> list[dict[str, object]]:
        return self._state.list_capture_events(request.session_id)

    def list_capture_rounds(
        self, request: CaptureRoundsRequest
    ) -> list[dict[str, object]]:
        return self._state.list_capture_rounds(request.session_id)

    def list_sensitive_quarantine(
        self, request: SensitiveQuarantineListRequest
    ) -> list[dict[str, object]]:
        return self._state.list_sensitive_quarantine(request.offset, request.limit)

    def resolve_sensitive_quarantine(
        self, request: SensitiveQuarantineResolutionRequest
    ) -> dict[str, object]:
        return self._state.resolve_sensitive_quarantine(
            request.record_id, request.resolution
        )

    def _ingest_capture(self, request: CaptureMemoryRequest) -> dict[str, object]:
        return self._state.ingest_capture_event(
            request.event_id,
            request.session_id,
            request.turn_id,
            request.event_kind,
            request.content,
            request.occurred_at,
            request.cwd,
            request.deadline,
        )

    async def _mutate_candidate(
        self, candidate_id: str, operation: Callable[[], dict[str, object]]
    ) -> dict[str, object]:
        library_id = self._state.candidate_library_id(candidate_id)
        async with self._state.library_write_lock(library_id):
            if self._state.candidate_library_id(candidate_id) != library_id:
                raise CandidateGovernanceError("candidate memory changed libraries")
            return operation()


class UnavailableMemoryKernel:
    """Explicitly unavailable mem2 kernel used when runtime startup fails."""

    kind: MemoryKernelKind = "mem2"

    def health(self) -> KernelHealth:
        return KernelHealth(
            kind=self.kind,
            status="unavailable",
            detail="mem2 kernel runtime is unavailable",
        )

    def _raise(self) -> NoReturn:
        raise MemoryKernelUnavailableError(
            "mem2 memory kernel is unavailable for this unsupported operation"
        )

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]:
        self._raise()

    async def search(self, request: SearchMemoryRequest) -> dict[str, object]:
        self._raise()

    def inspect(self, request: MemoryInspectRequest) -> dict[str, object]:
        self._raise()

    async def update(self, request: MemoryUpdateRequest) -> dict[str, object]:
        self._raise()

    async def delete(self, request: DeleteMemoryRequest) -> dict[str, str]:
        self._raise()

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]:
        self._raise()

    async def evolve(self, request: EvolveMemoryRequest) -> dict[str, int]:
        self._raise()

    def resolve_region(self, request: ResolveRegionRequest) -> dict[str, str]:
        self._raise()

    def create_region(self, request: CreateMemoryRegionRequest) -> dict[str, object]:
        self._raise()

    def list_regions(self) -> list[dict[str, object]]:
        self._raise()

    def get_region(self, request: GetMemoryRegionRequest) -> dict[str, object]:
        self._raise()

    def region_tree(self, request: MemoryRegionTreeRequest) -> dict[str, object]:
        self._raise()

    def bind_project_region(self, request: BindProjectRegionRequest) -> dict[str, str]:
        self._raise()

    def associate_worktree_region(
        self, request: AssociateWorktreeRegionRequest
    ) -> dict[str, str]:
        self._raise()

    def list_region_bindings(self) -> list[dict[str, str]]:
        self._raise()

    def update_region_binding(self, request: UpdateRegionBindingRequest) -> dict[str, str]:
        self._raise()

    def delete_region_binding(self, request: DeleteRegionBindingRequest) -> None:
        self._raise()

    async def scan_library(
        self, request: ScanLibraryRequest
    ) -> dict[str, int | str]:
        self._raise()

    def list_out_of_band_changes(
        self, request: ListOutOfBandChangesRequest
    ) -> list[dict[str, object]]:
        self._raise()

    def out_of_band_change(
        self, request: OutOfBandChangeRequest
    ) -> dict[str, object]:
        self._raise()

    async def resolve_out_of_band_change(
        self, request: ResolveOutOfBandChangeRequest
    ) -> dict[str, str]:
        self._raise()

    def list_candidates(
        self, request: ListCandidatesRequest
    ) -> list[dict[str, object]]:
        self._raise()

    def candidate_governance_queue(
        self, request: CandidateGovernanceListRequest
    ) -> list[dict[str, object]]:
        self._raise()

    def candidate_governance(
        self, request: CandidateGovernanceRequest
    ) -> dict[str, object]:
        self._raise()

    async def decide_candidate(
        self, request: CandidateDecisionRequest
    ) -> dict[str, object]:
        self._raise()

    async def pin_candidate(
        self, request: CandidatePinRequest
    ) -> dict[str, object]:
        self._raise()

    def list_recycled_candidates(
        self, request: CandidateRecycleListRequest
    ) -> list[dict[str, object]]:
        self._raise()

    async def restore_recycled_candidate(
        self, request: CandidateRestoreRequest
    ) -> dict[str, object]:
        self._raise()

    def list_documents(
        self, request: ListDocumentsRequest
    ) -> list[dict[str, object]]:
        self._raise()

    def preview_document_update(
        self, request: PreviewDocumentUpdateRequest
    ) -> dict[str, object]:
        self._raise()

    async def preview_document_delete(
        self, request: PreviewDocumentDeleteRequest
    ) -> dict[str, object]:
        self._raise()

    def list_forgotten(
        self, request: ListForgottenRequest
    ) -> list[dict[str, object]]:
        self._raise()

    async def restore_forgotten(
        self, request: RestoreForgottenRequest
    ) -> dict[str, str]:
        self._raise()

    def history_diff(self, request: HistoryDiffRequest) -> dict[str, object]:
        self._raise()

    async def restore_document(
        self, request: RestoreDocumentRequest
    ) -> dict[str, str]:
        self._raise()

    def get_retention_policy(
        self, request: RetentionPolicyRequest
    ) -> dict[str, object]:
        self._raise()

    async def update_retention_policy(
        self, request: UpdateRetentionPolicyRequest
    ) -> dict[str, object]:
        self._raise()

    async def preview_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        self._raise()

    async def run_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        self._raise()

    def clear_diagnostics(self, request: ClearDiagnosticsRequest) -> int:
        self._raise()

    def list_capture_events(
        self, request: CaptureEventsRequest
    ) -> list[dict[str, object]]:
        self._raise()

    def list_capture_rounds(
        self, request: CaptureRoundsRequest
    ) -> list[dict[str, object]]:
        self._raise()

    def list_sensitive_quarantine(
        self, request: SensitiveQuarantineListRequest
    ) -> list[dict[str, object]]:
        self._raise()

    def resolve_sensitive_quarantine(
        self, request: SensitiveQuarantineResolutionRequest
    ) -> dict[str, object]:
        self._raise()


def select_memory_kernel(settings: Settings, state: PlatformState) -> MemoryKernelPort:
    regions = RegionRegistry(settings.state_dir / "regions.sqlite3")
    if settings.memory_kernel == "legacy":
        return LegacyMemoryKernel(state, regions)
    from personal_agent_memory.mem2_kernel import Mem2MemoryKernel

    return Mem2MemoryKernel(
        settings.state_dir,
        embedding=settings.embedding,
        embedding_dimension=settings.embedding_dimension,
        regions=regions,
    )
