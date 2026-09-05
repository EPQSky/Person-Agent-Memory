from __future__ import annotations

import hmac
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse, Response
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field, ValidationError, model_validator

from personal_agent_memory.config import Settings
from personal_agent_memory.context_package import MIN_TOKEN_BUDGET, build_context_package
from personal_agent_memory.graph_adapter import JiuwenMilvusGraphAdapter
from personal_agent_memory.model_client import OpenAICompatibleClient
from personal_agent_memory.security import ApiKeyStore
from personal_agent_memory.state import (
    CandidateGovernanceError,
    DuplicateLibraryError,
    DuplicateProjectBindingError,
    LibraryKind,
    LibraryRegistrationError,
    MemoryMutationError,
    PlatformState,
    ProjectBindingError,
)


class LibraryRegistration(BaseModel):
    path: str
    kind: LibraryKind
    reuse_existing_git: bool = False


class ProjectBindingRegistration(BaseModel):
    project_root: str
    library_id: str


class ProjectBindingUpdate(BaseModel):
    library_id: str


class WorktreeAssociation(BaseModel):
    worktree_root: str
    main_project_binding_id: str


class CwdResolution(BaseModel):
    cwd: str


class McpRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class IgnoreRulesUpdate(BaseModel):
    patterns: list[str]


class SearchRequest(BaseModel):
    cwd: str | None = None
    library_id: str | None = None
    query: str
    limit: int = 10
    graph_hops: int = Field(default=1, ge=1, le=2)
    token_budget: int = Field(default=10_000, ge=MIN_TOKEN_BUDGET)
    target_model: str = Field(default="gpt-4o-mini", min_length=1, max_length=200)
    include_history: bool = False

    @model_validator(mode="after")
    def validate_scope(self) -> SearchRequest:
        if (self.cwd is None) == (self.library_id is None):
            raise ValueError("exactly one of cwd or library_id is required")
        return self


class DocumentPreview(BaseModel):
    path: str
    content: str
    expected_source_version: str


class DocumentEdit(DocumentPreview):
    operation_id: str
    actor_type: Literal["user", "platform"] = "user"
    source: str = "web"


class DocumentRestore(BaseModel):
    path: str
    commit: str
    expected_source_version: str
    operation_id: str
    actor_type: Literal["user", "platform"] = "user"
    source: str = "web"


CandidateType = Literal[
    "preference",
    "decision",
    "constraint",
    "domain_fact",
    "reusable_experience",
    "external_reference",
]


class CandidateCreate(BaseModel):
    library_id: str
    suggested_type: CandidateType
    body: str = Field(min_length=1, max_length=200_000)
    source_references: list[str] = Field(min_length=1, max_length=100)
    creator: str = Field(min_length=1, max_length=200)
    idempotency_key: str = Field(min_length=1, max_length=200)


class CandidateEdit(BaseModel):
    body: str = Field(min_length=1, max_length=200_000)
    operator: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)


class CandidateDecision(BaseModel):
    operator: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)
    operation_id: str = Field(min_length=1, max_length=200)


class CandidateResolution(CandidateDecision):
    action: Literal["keep", "adopt", "merge", "scope"]
    merged_body: str | None = Field(default=None, max_length=200_000)
    effective_at: str | None = Field(default=None, max_length=100)
    condition: str | None = Field(default=None, max_length=2_000)


class CaptureEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=1000)
    turn_id: str = Field(min_length=1, max_length=200)
    event_kind: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=65_536)
    occurred_at: str = Field(min_length=1, max_length=100)
    cwd: str = Field(min_length=1, max_length=4096)


class CaptureConsolidation(BaseModel):
    session_id: str | None = Field(default=None, min_length=1, max_length=1000)


class SensitiveResolution(BaseModel):
    resolution: Literal["discard", "acknowledge"]


def create_app(settings: Settings) -> FastAPI:
    key_store = ApiKeyStore(settings.state_dir / "api-key")
    model_client = OpenAICompatibleClient(settings.embedding, settings.reranker, settings.graph)
    platform_state = PlatformState(
        settings.state_dir / "platform.sqlite3",
        library_roots=settings.library_roots,
        model_client=model_client,
        graph_adapter=JiuwenMilvusGraphAdapter(settings.state_dir / "graphs", model_client),
    )
    markdown = MarkdownIt("commonmark", {"html": False})

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        key_store.ensure()
        await platform_state.start()
        try:
            yield
        finally:
            await platform_state.close()

    app = FastAPI(title="Personal Agent Memory", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.platform_state = platform_state

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        stored = key_store.read()
        if (
            scheme.lower() != "bearer"
            or not supplied
            or not stored
            or not hmac.compare_digest(supplied, stored)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def status_payload() -> dict[str, str | int | bool]:
        key = key_store.read()
        return {
            "status": "ready",
            "database": "ok",
            "background_worker": "running",
            "startup_count": platform_state.startup_count,
            "previous_shutdown_clean": platform_state.previous_shutdown_clean,
            "api_key_fingerprint": key_store.fingerprint(key),
        }

    @app.get("/health/live", dependencies=[Depends(authenticate)])
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/status", dependencies=[Depends(authenticate)])
    def api_status() -> dict[str, str | int | bool]:
        return status_payload()

    @app.get("/mcp/health", dependencies=[Depends(authenticate)])
    def mcp_status() -> dict[str, str | int | bool]:
        return status_payload()

    @app.post("/api/v1/auth/rotate", dependencies=[Depends(authenticate)])
    def rotate_key() -> dict[str, str]:
        new_key = key_store.rotate()
        return {"fingerprint": key_store.fingerprint(new_key)}

    def libraries_payload() -> list[dict[str, str]]:
        return [library.payload() for library in platform_state.list_libraries()]

    @app.post(
        "/api/v1/libraries",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_201_CREATED,
    )
    async def register_library(registration: LibraryRegistration) -> dict[str, str]:
        try:
            return platform_state.register_library(
                registration.path,
                registration.kind,
                registration.reuse_existing_git,
            ).payload()
        except DuplicateLibraryError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(error), "library_id": error.library_id},
            ) from error
        except LibraryRegistrationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.post("/api/v1/libraries/{library_id}/scan", dependencies=[Depends(authenticate)])
    async def scan_library(library_id: str) -> dict[str, int | str]:
        try:
            return platform_state.scan_library(library_id)
        except LibraryRegistrationError as error:
            status_code = (
                status.HTTP_404_NOT_FOUND
                if str(error) == "memory library not found"
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            )
            raise HTTPException(status_code=status_code, detail=str(error)) from error

    @app.post(
        "/api/v1/libraries/{library_id}/vector-index/rebuild",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def rebuild_vector_index(library_id: str) -> dict[str, object]:
        try:
            job_id = platform_state.enqueue_vector_rebuild(library_id)
        except LibraryRegistrationError as error:
            code = (
                status.HTTP_404_NOT_FOUND
                if str(error) == "memory library not found"
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            )
            raise HTTPException(status_code=code, detail=str(error)) from error
        return {"library_id": library_id, "job_id": job_id, "status": "pending"}

    @app.post(
        "/api/v1/libraries/{library_id}/graph-index/rebuild",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def rebuild_graph_index(library_id: str) -> dict[str, object]:
        try:
            job_id = platform_state.enqueue_graph_rebuild(library_id)
        except LibraryRegistrationError as error:
            code = (
                status.HTTP_404_NOT_FOUND
                if str(error) == "memory library not found"
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            )
            raise HTTPException(status_code=code, detail=str(error)) from error
        return {"library_id": library_id, "job_id": job_id, "status": "pending"}

    @app.get(
        "/api/v1/libraries/{library_id}/graph-status",
        dependencies=[Depends(authenticate)],
    )
    async def graph_status(library_id: str) -> dict[str, object]:
        try:
            return platform_state.graph_status(library_id)
        except LibraryRegistrationError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(authenticate)])
    async def background_job(job_id: int) -> dict[str, object]:
        job_state = platform_state.job_status(job_id)
        if job_state is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
        return {"job_id": job_id, "status": job_state}

    @app.get("/api/v1/libraries/{library_id}/ignore-rules", dependencies=[Depends(authenticate)])
    async def get_ignore_rules(library_id: str) -> dict[str, object]:
        try:
            patterns = platform_state.library_ignore_patterns(library_id)
        except LibraryRegistrationError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return {"library_id": library_id, "patterns": patterns}

    @app.put("/api/v1/libraries/{library_id}/ignore-rules", dependencies=[Depends(authenticate)])
    async def update_ignore_rules(library_id: str, update: IgnoreRulesUpdate) -> dict[str, object]:
        try:
            patterns = platform_state.update_library_ignore_patterns(
                library_id, tuple(update.patterns)
            )
        except LibraryRegistrationError as error:
            status_code = (
                status.HTTP_404_NOT_FOUND
                if str(error) == "memory library not found"
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            )
            raise HTTPException(status_code=status_code, detail=str(error)) from error
        return {"library_id": library_id, "patterns": patterns}

    @app.get("/api/v1/libraries", dependencies=[Depends(authenticate)])
    async def list_libraries() -> list[dict[str, str]]:
        return libraries_payload()

    @app.get("/mcp/libraries", dependencies=[Depends(authenticate)])
    async def mcp_list_libraries() -> list[dict[str, str]]:
        return libraries_payload()

    def project_bindings_payload() -> list[dict[str, str]]:
        return [binding.payload() for binding in platform_state.list_project_bindings()]

    @app.get("/api/v1/project-bindings/resolve", dependencies=[Depends(authenticate)])
    async def resolve_project_binding(cwd: str) -> dict[str, str]:
        try:
            return platform_state.resolve_project_binding(cwd)
        except ProjectBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.post("/mcp/project-bindings/resolve", dependencies=[Depends(authenticate)])
    async def mcp_resolve_project_binding(resolution: CwdResolution) -> dict[str, str]:
        try:
            return platform_state.resolve_project_binding(resolution.cwd)
        except ProjectBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.post(
        "/api/v1/project-bindings/worktrees",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_201_CREATED,
    )
    async def associate_worktree(association: WorktreeAssociation) -> dict[str, str]:
        try:
            return platform_state.associate_worktree(
                association.worktree_root, association.main_project_binding_id
            ).payload()
        except DuplicateProjectBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(error), "binding_id": error.binding_id},
            ) from error
        except ProjectBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.post(
        "/api/v1/project-bindings",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_201_CREATED,
    )
    async def bind_project(binding: ProjectBindingRegistration) -> dict[str, str]:
        try:
            return platform_state.bind_project(binding.project_root, binding.library_id).payload()
        except DuplicateProjectBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(error), "binding_id": error.binding_id},
            ) from error
        except ProjectBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.get("/api/v1/project-bindings", dependencies=[Depends(authenticate)])
    async def list_project_bindings() -> list[dict[str, str]]:
        return project_bindings_payload()

    @app.put("/api/v1/project-bindings/{binding_id}", dependencies=[Depends(authenticate)])
    async def update_project_binding(
        binding_id: str, update: ProjectBindingUpdate
    ) -> dict[str, str]:
        try:
            return platform_state.update_project_binding(binding_id, update.library_id).payload()
        except ProjectBindingError as error:
            status_code = (
                status.HTTP_404_NOT_FOUND
                if str(error) == "project binding not found"
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            )
            raise HTTPException(status_code=status_code, detail=str(error)) from error

    @app.delete(
        "/api/v1/project-bindings/{binding_id}",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_project_binding(binding_id: str) -> Response:
        try:
            platform_state.delete_project_binding(binding_id)
        except ProjectBindingError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/mcp/libraries/{library_id}/sync-status", dependencies=[Depends(authenticate)])
    async def mcp_sync_status(library_id: str) -> dict[str, str]:
        library = platform_state.library(library_id)
        if library is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library not found")
        return {
            "library_id": library.id,
            "availability": library.availability,
            "sync_status": library.sync_status,
        }

    async def search_payload(request: SearchRequest) -> dict[str, object]:
        try:
            raw = (
                await platform_state.search_project(
                    request.cwd,
                    request.query,
                    request.limit,
                    request.graph_hops,
                    request.include_history,
                )
                if request.cwd is not None
                else await platform_state.search_library(
                    request.library_id or "",
                    request.query,
                    request.limit,
                    request.graph_hops,
                    request.include_history,
                )
            )
            return build_context_package(
                raw,
                requested_budget=request.token_budget,
                target_model=request.target_model,
            )
        except ProjectBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        except LibraryRegistrationError as error:
            status_code = (
                status.HTTP_404_NOT_FOUND
                if str(error) == "memory library not found"
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            )
            raise HTTPException(status_code=status_code, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.post("/api/v1/search", dependencies=[Depends(authenticate)])
    async def search(request: SearchRequest) -> dict[str, object]:
        return await search_payload(request)

    @app.post("/mcp/search", dependencies=[Depends(authenticate)])
    async def mcp_search(request: SearchRequest) -> dict[str, object]:
        return await search_payload(request)

    def candidate_error(error: CandidateGovernanceError) -> HTTPException:
        detail = str(error)
        if detail in {"candidate memory not found", "memory library not found"}:
            code = status.HTTP_404_NOT_FOUND
        elif "already" in detail or "no longer pending" in detail:
            code = status.HTTP_409_CONFLICT
        else:
            code = status.HTTP_422_UNPROCESSABLE_CONTENT
        return HTTPException(status_code=code, detail=detail)

    def create_candidate_payload(candidate: CandidateCreate) -> dict[str, object]:
        try:
            return platform_state.create_candidate(
                candidate.library_id,
                candidate.suggested_type,
                candidate.body,
                tuple(candidate.source_references),
                candidate.creator,
                candidate.idempotency_key,
            )
        except (CandidateGovernanceError, MemoryMutationError) as error:
            if isinstance(error, MemoryMutationError):
                raise mutation_error(error) from error
            raise candidate_error(error) from error

    @app.post(
        "/api/v1/capture/events",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authenticate)],
    )
    async def capture_event(event: CaptureEvent) -> dict[str, object]:
        try:
            return platform_state.ingest_capture_event(
                event.event_id,
                event.session_id,
                event.turn_id,
                event.event_kind,
                event.content,
                event.occurred_at,
                event.cwd,
            )
        except (CandidateGovernanceError, ProjectBindingError) as error:
            raise candidate_error(CandidateGovernanceError(str(error))) from error

    @app.get("/api/v1/capture/events", dependencies=[Depends(authenticate)])
    async def capture_events(session_id: str | None = None) -> list[dict[str, object]]:
        return platform_state.list_capture_events(session_id)

    @app.get("/api/v1/sensitive-quarantine", dependencies=[Depends(authenticate)])
    async def sensitive_quarantine(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=100),
    ) -> list[dict[str, object]]:
        return platform_state.list_sensitive_quarantine(offset, limit)

    @app.post(
        "/api/v1/sensitive-quarantine/{record_id}/resolve",
        dependencies=[Depends(authenticate)],
    )
    async def resolve_sensitive_quarantine(
        record_id: str, decision: SensitiveResolution
    ) -> dict[str, object]:
        try:
            return platform_state.resolve_sensitive_quarantine(record_id, decision.resolution)
        except CandidateGovernanceError as error:
            raise candidate_error(error) from error

    @app.get("/api/v1/capture/rounds", dependencies=[Depends(authenticate)])
    async def capture_rounds(session_id: str | None = None) -> list[dict[str, object]]:
        return platform_state.list_capture_rounds(session_id)

    @app.get("/api/v1/candidate-governance", dependencies=[Depends(authenticate)])
    async def candidate_governance_queue(
        classification: str | None = None,
    ) -> list[dict[str, object]]:
        try:
            return platform_state.list_candidate_governance(classification)
        except CandidateGovernanceError as error:
            raise candidate_error(error) from error

    @app.get(
        "/api/v1/candidates/{candidate_id}/governance",
        dependencies=[Depends(authenticate)],
    )
    async def candidate_governance(candidate_id: str) -> dict[str, object]:
        try:
            return platform_state.candidate_governance(candidate_id)
        except CandidateGovernanceError as error:
            raise candidate_error(error) from error

    @app.post(
        "/api/v1/candidates/{candidate_id}/resolve",
        dependencies=[Depends(authenticate)],
    )
    async def resolve_candidate(
        candidate_id: str, resolution: CandidateResolution
    ) -> dict[str, object]:
        try:
            return platform_state.resolve_candidate(
                candidate_id,
                resolution.action,
                resolution.operator,
                resolution.reason,
                resolution.operation_id,
                merged_body=resolution.merged_body,
                effective_at=resolution.effective_at,
                condition=resolution.condition,
            )
        except (CandidateGovernanceError, MemoryMutationError) as error:
            if isinstance(error, MemoryMutationError):
                raise mutation_error(error) from error
            raise candidate_error(error) from error

    @app.post(
        "/api/v1/capture/consolidate",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authenticate)],
    )
    async def consolidate_capture(request: CaptureConsolidation) -> dict[str, int]:
        return {"queued": platform_state.trigger_capture_consolidation(request.session_id)}

    @app.post(
        "/mcp/candidates",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_201_CREATED,
    )
    async def mcp_create_candidate(candidate: CandidateCreate) -> dict[str, object]:
        return create_candidate_payload(candidate)

    @app.get("/mcp/candidates", dependencies=[Depends(authenticate)])
    async def mcp_list_candidates(
        library_id: str | None = None, candidate_status: str | None = None
    ) -> list[dict[str, object]]:
        try:
            return platform_state.list_candidates(library_id, candidate_status)
        except (CandidateGovernanceError, MemoryMutationError) as error:
            if isinstance(error, MemoryMutationError):
                raise mutation_error(error) from error
            raise candidate_error(error) from error

    @app.get("/mcp/candidates/{candidate_id}", dependencies=[Depends(authenticate)])
    async def mcp_get_candidate(candidate_id: str) -> dict[str, object]:
        try:
            return platform_state.candidate(candidate_id)
        except CandidateGovernanceError as error:
            raise candidate_error(error) from error

    @app.get("/api/v1/candidates", dependencies=[Depends(authenticate)])
    async def list_candidates(
        library_id: str | None = None, candidate_status: str | None = None
    ) -> list[dict[str, object]]:
        try:
            return platform_state.list_candidates(library_id, candidate_status)
        except (CandidateGovernanceError, MemoryMutationError) as error:
            if isinstance(error, MemoryMutationError):
                raise mutation_error(error) from error
            raise candidate_error(error) from error

    @app.get("/api/v1/candidates/{candidate_id}", dependencies=[Depends(authenticate)])
    async def get_candidate(candidate_id: str) -> dict[str, object]:
        try:
            return platform_state.candidate(candidate_id)
        except CandidateGovernanceError as error:
            raise candidate_error(error) from error

    @app.put("/api/v1/candidates/{candidate_id}", dependencies=[Depends(authenticate)])
    async def edit_candidate(candidate_id: str, edit: CandidateEdit) -> dict[str, object]:
        try:
            return platform_state.edit_candidate(
                candidate_id, edit.body, edit.operator, edit.reason
            )
        except CandidateGovernanceError as error:
            raise candidate_error(error) from error

    @app.post(
        "/api/v1/candidates/{candidate_id}/approve",
        dependencies=[Depends(authenticate)],
    )
    async def approve_candidate(
        candidate_id: str, decision: CandidateDecision
    ) -> dict[str, object]:
        try:
            return platform_state.approve_candidate(
                candidate_id, decision.operator, decision.reason, decision.operation_id
            )
        except CandidateGovernanceError as error:
            raise candidate_error(error) from error
        except MemoryMutationError as error:
            raise mutation_error(error) from error

    @app.post(
        "/api/v1/candidates/{candidate_id}/reject",
        dependencies=[Depends(authenticate)],
    )
    async def reject_candidate(candidate_id: str, decision: CandidateDecision) -> dict[str, object]:
        try:
            return platform_state.reject_candidate(
                candidate_id, decision.operator, decision.reason, decision.operation_id
            )
        except CandidateGovernanceError as error:
            raise candidate_error(error) from error

    def mutation_error(error: MemoryMutationError) -> HTTPException:
        detail = str(error)
        if detail in {"memory library not found", "memory document not found"}:
            code = status.HTTP_404_NOT_FOUND
        elif "version conflict" in detail or "changed outside" in detail:
            code = status.HTTP_409_CONFLICT
        else:
            code = status.HTTP_422_UNPROCESSABLE_CONTENT
        return HTTPException(status_code=code, detail=detail)

    @app.get("/api/v1/libraries/{library_id}/documents", dependencies=[Depends(authenticate)])
    async def list_documents(library_id: str) -> list[dict[str, object]]:
        try:
            return platform_state.list_documents(library_id)
        except MemoryMutationError as error:
            raise mutation_error(error) from error

    @app.get("/api/v1/libraries/{library_id}/document", dependencies=[Depends(authenticate)])
    async def read_document(library_id: str, path: str) -> dict[str, str]:
        try:
            return platform_state.read_document(library_id, path)
        except MemoryMutationError as error:
            raise mutation_error(error) from error

    @app.post(
        "/api/v1/libraries/{library_id}/document/preview",
        dependencies=[Depends(authenticate)],
    )
    async def preview_document(library_id: str, preview: DocumentPreview) -> dict[str, object]:
        try:
            result = platform_state.preview_document_edit(
                library_id, preview.path, preview.content, preview.expected_source_version
            )
        except MemoryMutationError as error:
            raise mutation_error(error) from error
        result["rendered_html"] = markdown.render(preview.content)
        return result

    @app.put("/api/v1/libraries/{library_id}/document", dependencies=[Depends(authenticate)])
    async def edit_document(library_id: str, edit: DocumentEdit) -> dict[str, str]:
        try:
            return platform_state.edit_document(
                library_id,
                edit.path,
                edit.content,
                edit.expected_source_version,
                edit.operation_id,
                edit.actor_type,
                edit.source,
            )
        except MemoryMutationError as error:
            raise mutation_error(error) from error

    @app.get("/api/v1/libraries/{library_id}/history", dependencies=[Depends(authenticate)])
    async def library_history(library_id: str, limit: int = 50) -> list[dict[str, str]]:
        try:
            return platform_state.library_history(library_id, limit)
        except MemoryMutationError as error:
            raise mutation_error(error) from error

    @app.get(
        "/api/v1/libraries/{library_id}/history/{commit}/diff",
        dependencies=[Depends(authenticate)],
    )
    async def history_diff(library_id: str, commit: str) -> dict[str, object]:
        try:
            return platform_state.history_diff(library_id, commit)
        except MemoryMutationError as error:
            raise mutation_error(error) from error

    @app.post(
        "/api/v1/libraries/{library_id}/history/restore",
        dependencies=[Depends(authenticate)],
    )
    async def restore_document(library_id: str, restore: DocumentRestore) -> dict[str, str]:
        try:
            return platform_state.restore_document(
                library_id,
                restore.path,
                restore.commit,
                restore.expected_source_version,
                restore.operation_id,
                restore.actor_type,
                restore.source,
            )
        except MemoryMutationError as error:
            raise mutation_error(error) from error

    mcp_tools: list[dict[str, object]] = [
        {
            "name": "memory_search",
            "description": "Search one project-bound or explicitly selected memory library.",
            "inputSchema": SearchRequest.model_json_schema(),
        },
        {
            "name": "memory_get",
            "description": "Read one authoritative Markdown document by library and relative path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "library_id": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["library_id", "path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "candidate_create",
            "description": "Create an unapproved candidate memory.",
            "inputSchema": CandidateCreate.model_json_schema(),
        },
        {
            "name": "candidate_list",
            "description": "List candidate memories, optionally filtered by library and status.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "library_id": {"type": "string"},
                    "candidate_status": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "library_list",
            "description": "List registered memory libraries.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "sync_status",
            "description": "Read availability and synchronization status for one memory library.",
            "inputSchema": {
                "type": "object",
                "properties": {"library_id": {"type": "string"}},
                "required": ["library_id"],
                "additionalProperties": False,
            },
        },
    ]

    def mcp_rpc_error(request_id: str | int | None, code: int, message: str) -> dict[str, object]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def mcp_tool_result(payload: object) -> dict[str, object]:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        result: dict[str, object] = {
            "content": [{"type": "text", "text": serialized}],
            "isError": False,
        }
        if isinstance(payload, dict):
            result["structuredContent"] = payload
        return result

    async def call_mcp_tool(name: str, arguments: dict[str, Any]) -> object:
        if name == "memory_search":
            return await search_payload(SearchRequest.model_validate(arguments))
        if name == "memory_get":
            library_id = str(arguments.get("library_id", ""))
            path = str(arguments.get("path", ""))
            return platform_state.read_document(library_id, path)
        if name == "candidate_create":
            return create_candidate_payload(CandidateCreate.model_validate(arguments))
        if name == "candidate_list":
            filter_library_id = arguments.get("library_id")
            candidate_status = arguments.get("candidate_status")
            return platform_state.list_candidates(
                None if filter_library_id is None else str(filter_library_id),
                None if candidate_status is None else str(candidate_status),
            )
        if name == "library_list":
            return libraries_payload()
        if name == "sync_status":
            library = platform_state.library(str(arguments.get("library_id", "")))
            if library is None:
                raise LibraryRegistrationError("memory library not found")
            return {
                "library_id": library.id,
                "availability": library.availability,
                "sync_status": library.sync_status,
            }
        raise ValueError("unknown MCP tool")

    @app.post("/mcp", dependencies=[Depends(authenticate)], response_model=None)
    async def mcp_streamable_http(payload: dict[str, Any]) -> Response | dict[str, object]:
        try:
            rpc = McpRpcRequest.model_validate(payload)
        except ValidationError:
            return mcp_rpc_error(None, -32600, "Invalid Request")
        if rpc.method == "notifications/initialized":
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        if rpc.method == "ping":
            return {"jsonrpc": "2.0", "id": rpc.id, "result": {}}
        if rpc.method == "initialize":
            requested = rpc.params.get("protocolVersion")
            protocol_version = requested if isinstance(requested, str) else "2025-06-18"
            return {
                "jsonrpc": "2.0",
                "id": rpc.id,
                "result": {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "personal-agent-memory", "version": "0.1.0"},
                },
            }
        if rpc.method == "tools/list":
            return {"jsonrpc": "2.0", "id": rpc.id, "result": {"tools": mcp_tools}}
        if rpc.method != "tools/call":
            return mcp_rpc_error(rpc.id, -32601, "Method not found")
        name = rpc.params.get("name")
        arguments = rpc.params.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return mcp_rpc_error(rpc.id, -32602, "Invalid params")
        try:
            result = await call_mcp_tool(name, arguments)
        except HTTPException as error:
            return {
                "jsonrpc": "2.0",
                "id": rpc.id,
                "result": {
                    "content": [{"type": "text", "text": str(error.detail)}],
                    "isError": True,
                },
            }
        except (
            CandidateGovernanceError,
            LibraryRegistrationError,
            MemoryMutationError,
            ProjectBindingError,
            ValidationError,
            ValueError,
        ) as error:
            return {
                "jsonrpc": "2.0",
                "id": rpc.id,
                "result": {
                    "content": [{"type": "text", "text": str(error)}],
                    "isError": True,
                },
            }
        return {"jsonrpc": "2.0", "id": rpc.id, "result": mcp_tool_result(result)}

    frontend = Path(__file__).parent / "static" / "index.html"
    editor_history = Path(__file__).parent / "static" / "editor_history.js"

    @app.get("/editor-history.js")
    def editor_history_script() -> FileResponse:
        return FileResponse(editor_history, media_type="text/javascript")

    @app.get("/", response_class=HTMLResponse)
    def management_interface() -> FileResponse:
        return FileResponse(frontend)

    return app
