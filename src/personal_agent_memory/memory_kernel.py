from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NoReturn, Protocol, TypeAlias, runtime_checkable

from personal_agent_memory.config import MemoryKernelKind, Settings
from personal_agent_memory.regions import RegionKind

KernelStatus = Literal["ready", "degraded", "unavailable"]
MAX_SCOPE_SPACE_LENGTH = 256
LIBRARY_SCOPE_PREFIX = "library:"
CWD_SCOPE_PREFIX = "cwd:"
MAX_LIBRARY_ID_LENGTH = MAX_SCOPE_SPACE_LENGTH - len(LIBRARY_SCOPE_PREFIX)
MEMORY_PAGE_SIZE = 50
FORMAL_MEMORY_TYPES = frozenset(
    {
        "decision",
        "constraint",
        "preference",
        "domain_fact",
        "reusable_experience",
        "external_reference",
    }
)
VISIBLE_MEMORY_LIFECYCLES = frozenset({"active", "superseded", "archived"})


def validate_library_id(library_id: str) -> str:
    """Return the normalized library identifier or reject an invalid scope."""
    identifier = library_id.strip()
    if not identifier:
        raise ValueError("library_id must not be blank")
    if len(identifier) > MAX_LIBRARY_ID_LENGTH:
        raise ValueError(f"library_id exceeds the {MAX_LIBRARY_ID_LENGTH}-character scope limit")
    return identifier


CAPTURE_BINDING_RETRY_LIMIT = 12


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
    user_metadata: Mapping[str, object] | None = None


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
            raise ValueError("exactly one of cwd, library_id, region_id, or region_ids is required")
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
    target_id: str | None = None

    def __post_init__(self) -> None:
        if self.target_id is not None:
            normalized = self.target_id.strip()
            if not normalized:
                raise ValueError("target_id must not be blank")
            object.__setattr__(self, "target_id", normalized)


@dataclass(frozen=True, slots=True)
class EvolveMemoryRequest:
    session_id: str | None = None
    deadline: float | None = None


@dataclass(frozen=True, slots=True)
class PromotionRunRequest:
    source_region_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.source_region_ids is None:
            return
        if not self.source_region_ids:
            raise ValueError("at least one source memory region is required")
        normalized = tuple(value.strip() for value in self.source_region_ids)
        if any(not value for value in normalized):
            raise ValueError("source memory region identifiers must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("source memory regions must not contain duplicates")
        object.__setattr__(self, "source_region_ids", normalized)


@dataclass(frozen=True, slots=True)
class PromotionStatusRequest:
    job_id: str

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("promotion job_id must not be blank")
        object.__setattr__(self, "job_id", self.job_id.strip())


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
class ListDocumentsRequest:
    library_id: str


@dataclass(frozen=True, slots=True)
class ListMemoriesRequest:
    library_id: str
    cursor: str | None = None
    q: str | None = None
    memory_type: str | None = None
    lifecycle: str | None = None
    limit: int = MEMORY_PAGE_SIZE

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        for name, value, maximum in (
            ("cursor", self.cursor, 2_048),
            ("q", self.q, 500),
            ("memory_type", self.memory_type, 100),
            ("lifecycle", self.lifecycle, 32),
        ):
            if value is None:
                continue
            normalized = value.strip()
            if len(normalized) > maximum:
                raise ValueError(f"{name} exceeds the {maximum}-character limit")
            object.__setattr__(self, name, normalized or None)
        if self.lifecycle is not None and self.lifecycle not in VISIBLE_MEMORY_LIFECYCLES:
            raise ValueError("lifecycle must be active, superseded, or archived")
        if self.memory_type is not None and self.memory_type not in FORMAL_MEMORY_TYPES:
            raise ValueError("memory_type must identify a formal semantic memory type")


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


@dataclass(frozen=True, slots=True)
class RetentionCleanupRequest:
    library_id: str




@dataclass(frozen=True, slots=True)
class CaptureEventsRequest:
    session_id: str | None = None
    region_id: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class CaptureRoundsRequest:
    session_id: str | None = None
    region_id: str | None = None
    turn_id: str | None = None


MemoryWriteRequest: TypeAlias = CreateMemoryRequest | CaptureMemoryRequest
MemoryInspectRequest: TypeAlias = DocumentInspectRequest
MemoryUpdateRequest: TypeAlias = DocumentUpdateRequest


@runtime_checkable
class MemoryKernelPort(Protocol):
    kind: MemoryKernelKind

    def health(self) -> KernelHealth: ...

    def graph_status(self, library_id: str | None = None) -> dict[str, object]: ...

    def entity_graph_status(self, library_id: str) -> dict[str, object]: ...

    async def rebuild_entity_graph(self, library_id: str) -> dict[str, object]: ...

    def entity_graph(
        self,
        library_id: str,
        *,
        entity_id: str | None = None,
        memory_id: str | None = None,
        entity_type: str | None = None,
        q: str | None = None,
        hops: int = 1,
        limit: int = 200,
    ) -> dict[str, object]: ...

    def entity_graph_memory(self, library_id: str, memory_id: str) -> dict[str, object]: ...

    def entity_graph_entity(
        self,
        library_id: str,
        entity_id: str,
        *,
        limit: int = 50,
        offset: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, object]: ...

    def entity_graph_relation(
        self,
        library_id: str,
        relation_id: str,
        *,
        limit: int = 50,
        offset: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, object]: ...

    async def rebuild_graph(self, library_id: str) -> dict[str, object]: ...

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]: ...

    async def search(self, request: SearchMemoryRequest) -> dict[str, object]: ...

    def inspect(self, request: MemoryInspectRequest) -> dict[str, object]: ...

    async def update(self, request: MemoryUpdateRequest) -> dict[str, object]: ...

    async def delete(self, request: DeleteMemoryRequest) -> dict[str, object]: ...

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]: ...

    async def evolve(self, request: EvolveMemoryRequest) -> dict[str, int]: ...

    async def start_promotion(self, request: PromotionRunRequest) -> dict[str, object]: ...

    def promotion_status(self, request: PromotionStatusRequest) -> dict[str, object]: ...

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












    def list_documents(self, request: ListDocumentsRequest) -> list[dict[str, object]]: ...

    def list_memories(self, request: ListMemoriesRequest) -> dict[str, object]: ...

    def inspect_memory(self, request: DocumentInspectRequest) -> dict[str, object]: ...

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

    def get_retention_policy(self, request: RetentionPolicyRequest) -> dict[str, object]: ...

    async def update_retention_policy(
        self, request: UpdateRetentionPolicyRequest
    ) -> dict[str, object]: ...

    async def preview_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]: ...

    async def run_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]: ...

    def list_capture_events(self, request: CaptureEventsRequest) -> list[dict[str, object]]: ...

    def list_capture_rounds(self, request: CaptureRoundsRequest) -> list[dict[str, object]]: ...




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




class UnavailableMemoryKernel:
    """Explicitly unavailable mem2 kernel used when runtime startup fails."""

    kind: MemoryKernelKind = "mem2"

    def health(self) -> KernelHealth:
        return KernelHealth(
            kind=self.kind,
            status="unavailable",
            detail="mem2 kernel runtime is unavailable",
        )

    def graph_status(self, library_id: str | None = None) -> dict[str, object]:
        return {"status": "unavailable", "backend": "mem2"}

    def entity_graph_status(self, library_id: str) -> dict[str, object]:
        self._raise()

    async def rebuild_entity_graph(self, library_id: str) -> dict[str, object]:
        self._raise()

    def entity_graph(
        self,
        library_id: str,
        *,
        entity_id: str | None = None,
        memory_id: str | None = None,
        entity_type: str | None = None,
        q: str | None = None,
        hops: int = 1,
        limit: int = 200,
    ) -> dict[str, object]:
        del entity_id, memory_id, entity_type, q, hops, limit
        self._raise()

    def entity_graph_memory(self, library_id: str, memory_id: str) -> dict[str, object]:
        self._raise()

    def entity_graph_entity(
        self,
        library_id: str,
        entity_id: str,
        *,
        limit: int = 50,
        offset: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, object]:
        del entity_id, limit, offset, cursor
        self._raise()

    def entity_graph_relation(
        self,
        library_id: str,
        relation_id: str,
        *,
        limit: int = 50,
        offset: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, object]:
        del relation_id, limit, offset, cursor
        self._raise()

    async def rebuild_graph(self, library_id: str) -> dict[str, object]:
        self._raise()

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

    async def delete(self, request: DeleteMemoryRequest) -> dict[str, object]:
        self._raise()

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]:
        self._raise()

    async def evolve(self, request: EvolveMemoryRequest) -> dict[str, int]:
        self._raise()

    async def start_promotion(self, request: PromotionRunRequest) -> dict[str, object]:
        self._raise()

    def promotion_status(self, request: PromotionStatusRequest) -> dict[str, object]:
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

    def associate_worktree_region(self, request: AssociateWorktreeRegionRequest) -> dict[str, str]:
        self._raise()

    def list_region_bindings(self) -> list[dict[str, str]]:
        self._raise()

    def update_region_binding(self, request: UpdateRegionBindingRequest) -> dict[str, str]:
        self._raise()

    def delete_region_binding(self, request: DeleteRegionBindingRequest) -> None:
        self._raise()












    def list_documents(self, request: ListDocumentsRequest) -> list[dict[str, object]]:
        self._raise()

    def list_memories(self, request: ListMemoriesRequest) -> dict[str, object]:
        self._raise()

    def inspect_memory(self, request: DocumentInspectRequest) -> dict[str, object]:
        self._raise()

    def preview_document_update(self, request: PreviewDocumentUpdateRequest) -> dict[str, object]:
        self._raise()

    async def preview_document_delete(
        self, request: PreviewDocumentDeleteRequest
    ) -> dict[str, object]:
        self._raise()

    def list_forgotten(self, request: ListForgottenRequest) -> list[dict[str, object]]:
        self._raise()

    async def restore_forgotten(self, request: RestoreForgottenRequest) -> dict[str, str]:
        self._raise()

    def history_diff(self, request: HistoryDiffRequest) -> dict[str, object]:
        self._raise()

    async def restore_document(self, request: RestoreDocumentRequest) -> dict[str, str]:
        self._raise()

    def get_retention_policy(self, request: RetentionPolicyRequest) -> dict[str, object]:
        self._raise()

    async def update_retention_policy(
        self, request: UpdateRetentionPolicyRequest
    ) -> dict[str, object]:
        self._raise()

    async def preview_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        self._raise()

    async def run_retention_cleanup(self, request: RetentionCleanupRequest) -> dict[str, object]:
        self._raise()

    def list_capture_events(self, request: CaptureEventsRequest) -> list[dict[str, object]]:
        self._raise()

    def list_capture_rounds(self, request: CaptureRoundsRequest) -> list[dict[str, object]]:
        self._raise()




def select_memory_kernel(settings: Settings) -> MemoryKernelPort:
    from personal_agent_memory.mem2_kernel import Mem2MemoryKernel

    return Mem2MemoryKernel(
        settings.state_dir,
        embedding=settings.embedding,
        embedding_dimension=settings.embedding_dimension,
        graph=settings.graph,
    )
