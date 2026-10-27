from __future__ import annotations

import asyncio
import hmac
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from markdown_it import MarkdownIt
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from personal_agent_memory.config import Settings
from personal_agent_memory.context_package import MIN_TOKEN_BUDGET, build_context_package
from personal_agent_memory.memory_kernel import (
    MAX_LIBRARY_ID_LENGTH,
    AssociateWorktreeRegionRequest,
    AuditMemoryRequest,
    BindProjectRegionRequest,
    CaptureBindingUnstableError,
    CaptureContractError,
    CaptureEventsRequest,
    CaptureMemoryRequest,
    CaptureRoundsRequest,
    CreateMemoryRegionRequest,
    CreateMemoryRequest,
    DeleteMemoryRequest,
    DeleteRegionBindingRequest,
    DocumentInspectRequest,
    DocumentUpdateRequest,
    EvolveMemoryRequest,
    GetMemoryRegionRequest,
    HistoryDiffRequest,
    ListDocumentsRequest,
    ListForgottenRequest,
    ListMemoriesRequest,
    ManagedMemoryKernelPort,
    MemoryGovernanceError,
    MemoryIdempotencyConflictError,
    MemoryKernelPort,
    MemoryKernelUnavailableError,
    MemoryLifecycleConflictError,
    MemoryRecordNotFoundError,
    MemoryRegionTreeRequest,
    MemoryRevisionConflictError,
    PreviewDocumentDeleteRequest,
    PreviewDocumentUpdateRequest,
    PromotionRunRequest,
    PromotionStatusRequest,
    ResolveRegionRequest,
    RestoreDocumentRequest,
    RestoreForgottenRequest,
    RetentionCleanupRequest,
    RetentionPolicyRequest,
    SearchMemoryRequest,
    UpdateRegionBindingRequest,
    UpdateRetentionPolicyRequest,
    select_memory_kernel,
)
from personal_agent_memory.regions import (
    MAX_REGION_ID_LENGTH,
    MAX_SESSION_ID_LENGTH,
    DuplicateRegionBindingError,
    DuplicateRegionError,
    RegionBindingError,
    RegionBindingNotFoundError,
    RegionError,
    RegionNotFoundError,
)
from personal_agent_memory.security import ApiKeyStore, ManagementSessionStore

CAPTURE_BINDING_DEADLINE_SECONDS = 0.25
LibraryId = Annotated[
    str,
    BeforeValidator(lambda value: value.strip() if isinstance(value, str) else value),
]


def _stripped_identifier(value: object) -> object:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("identifier must not be blank")
    return value


RequiredIdentifier = Annotated[str, BeforeValidator(_stripped_identifier)]
OptionalIdentifier = Annotated[str | None, BeforeValidator(_stripped_identifier)]
CAPTURE_PERSISTENCE_COMMIT_MARGIN_SECONDS = 0.06
CAPTURE_PERSISTENCE_DEADLINE_HEADER = "X-Personal-Agent-Memory-Persistence-Deadline-Ms"


def capture_request_budget_seconds(
    client_deadline_ms: str | None,
    *,
    now_epoch_seconds: float | None = None,
    maximum_seconds: float = CAPTURE_BINDING_DEADLINE_SECONDS,
) -> float:
    if client_deadline_ms is None:
        return maximum_seconds
    try:
        deadline_ms = int(client_deadline_ms.strip())
    except ValueError:
        return 0.0
    now = time.time() if now_epoch_seconds is None else now_epoch_seconds
    remaining = deadline_ms / 1000 - now
    if remaining <= 0:
        return 0.0
    return min(maximum_seconds, remaining)




class MemoryRegionCreate(BaseModel):
    kind: Literal["user", "topic", "project", "session"]
    name: str = Field(min_length=1, max_length=200)
    region_id: LibraryId | None = Field(default=None, min_length=1, max_length=MAX_REGION_ID_LENGTH)
    session_id: OptionalIdentifier = Field(
        default=None, min_length=1, max_length=MAX_SESSION_ID_LENGTH
    )


class ProjectBindingRegistration(BaseModel):
    project_root: str
    library_id: LibraryId | None = Field(
        default=None, min_length=1, max_length=MAX_LIBRARY_ID_LENGTH
    )
    region_id: LibraryId | None = Field(default=None, min_length=1, max_length=MAX_REGION_ID_LENGTH)

    @model_validator(mode="after")
    def validate_region(self) -> ProjectBindingRegistration:
        if (self.library_id is None) == (self.region_id is None):
            raise ValueError("exactly one of library_id or region_id is required")
        return self

    @property
    def target_region_id(self) -> str:
        value = self.region_id or self.library_id
        assert value is not None
        return value


class ProjectBindingUpdate(BaseModel):
    library_id: LibraryId | None = Field(
        default=None, min_length=1, max_length=MAX_LIBRARY_ID_LENGTH
    )
    region_id: LibraryId | None = Field(default=None, min_length=1, max_length=MAX_REGION_ID_LENGTH)

    @model_validator(mode="after")
    def validate_region(self) -> ProjectBindingUpdate:
        if (self.library_id is None) == (self.region_id is None):
            raise ValueError("exactly one of library_id or region_id is required")
        return self

    @property
    def target_region_id(self) -> str:
        value = self.region_id or self.library_id
        assert value is not None
        return value


class WorktreeAssociation(BaseModel):
    worktree_root: str
    main_project_binding_id: str


class CrossRegionBudgetRequest(BaseModel):
    user_fraction: float = Field(default=0.30, gt=0, le=1)
    selected_region_fraction: float = Field(default=0.20, gt=0, le=1)
    graph_fraction: float = Field(default=0.20, gt=0, le=1)
    source_structure_fraction: float = Field(default=0.10, gt=0, le=1)

    @model_validator(mode="after")
    def validate_fractions(self) -> CrossRegionBudgetRequest:
        if self.graph_fraction + self.source_structure_fraction > 1:
            raise ValueError("graph and source/structure fractions must not exceed 1")
        return self


class CwdResolution(BaseModel):
    cwd: str


class McpRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)




class SearchRequest(BaseModel):
    cwd: str | None = None
    library_id: LibraryId | None = Field(
        default=None, min_length=1, max_length=MAX_LIBRARY_ID_LENGTH
    )
    region_id: LibraryId | None = Field(default=None, min_length=1, max_length=MAX_REGION_ID_LENGTH)
    region_ids: list[LibraryId] | None = Field(default=None, min_length=1, max_length=20)
    query: str
    limit: int = 10
    graph_hops: int = Field(default=1, ge=1, le=2)
    token_budget: int = Field(default=10_000, ge=MIN_TOKEN_BUDGET)
    target_model: str = Field(default="gpt-4o-mini", min_length=1, max_length=200)
    include_history: bool = False
    session_id: OptionalIdentifier = Field(
        default=None, min_length=1, max_length=MAX_SESSION_ID_LENGTH
    )
    topic_id: OptionalIdentifier = Field(
        default=None, min_length=1, max_length=MAX_REGION_ID_LENGTH
    )
    cross_region_budget: CrossRegionBudgetRequest | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> SearchRequest:
        selectors = (
            self.cwd is not None,
            self.library_id is not None,
            self.region_id is not None,
            self.region_ids is not None,
        )
        if sum(selectors) != 1:
            raise ValueError("exactly one of cwd, library_id, region_id, or region_ids is required")
        if self.region_ids is not None and len(set(self.region_ids)) != len(self.region_ids):
            raise ValueError("region_ids must not contain duplicates")
        if self.cross_region_budget is not None and self.region_ids is None:
            raise ValueError("cross_region_budget requires explicit region_ids")
        return self


class DocumentPreview(BaseModel):
    path: str
    content: str
    expected_source_version: str


class DocumentEdit(DocumentPreview):
    operation_id: str
    actor_type: Literal["user", "platform"] = "user"
    source: str = "web"
    mode: Literal["supersede", "overwrite"] = "supersede"


class DocumentRestore(BaseModel):
    path: str
    commit: str
    expected_source_version: str
    operation_id: str
    actor_type: Literal["user", "platform"] = "user"
    source: str = "web"




class DocumentDeletePreview(BaseModel):
    path: str
    expected_source_version: str
    operation_id: str


class DocumentDelete(DocumentDeletePreview):
    preview_token: str
    actor_type: Literal["user", "platform"] = "user"
    source: str = "web-delete"
    mode: Literal["forget", "archive", "downweight", "purge"] = "forget"


class ForgottenMemoryRestore(BaseModel):
    tombstone_id: str
    commit: str
    operation_id: str
    actor_type: Literal["user", "platform"] = "user"
    source: str = "web-forgotten-history"






class MemoryCreate(BaseModel):
    library_id: LibraryId | None = Field(
        default=None, min_length=1, max_length=MAX_LIBRARY_ID_LENGTH
    )
    region_id: LibraryId | None = Field(
        default=None, min_length=1, max_length=MAX_LIBRARY_ID_LENGTH
    )
    content: str = Field(min_length=1, max_length=65_535)
    memory_type: str = Field(default="domain_fact", min_length=1, max_length=100)
    source_references: list[str] = Field(default_factory=lambda: ["rest"])
    actor: str = Field(default="local", min_length=1, max_length=200)
    idempotency_key: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_region(self) -> MemoryCreate:
        if (self.library_id is None) == (self.region_id is None):
            raise ValueError("exactly one of library_id or region_id is required")
        return self








class CaptureEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    session_id: RequiredIdentifier = Field(min_length=1, max_length=MAX_SESSION_ID_LENGTH)
    turn_id: str = Field(min_length=1, max_length=200)
    event_kind: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=65_535)
    occurred_at: str = Field(min_length=1, max_length=100)
    cwd: str = Field(min_length=1, max_length=4096)
    topic_id: OptionalIdentifier = Field(
        default=None, min_length=1, max_length=MAX_REGION_ID_LENGTH
    )

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("occurred_at must include a valid timezone offset")
            parsed.astimezone(UTC)
        except (ValueError, OverflowError) as error:
            raise ValueError(
                "occurred_at must be a valid ISO-8601 timestamp with timezone"
            ) from error
        return value


class CaptureConsolidation(BaseModel):
    session_id: OptionalIdentifier = Field(
        default=None, min_length=1, max_length=MAX_SESSION_ID_LENGTH
    )


class PromotionRun(BaseModel):
    source_region_ids: list[RequiredIdentifier] | None = Field(
        default=None,
        max_length=100,
    )




class RetentionPolicyUpdate(BaseModel):
    inbox_days: int = Field(default=30, ge=0, le=3650)


class ManagementSessionLogin(BaseModel):
    api_key: str = Field(min_length=1, max_length=1000)




def create_app(settings: Settings, *, memory_kernel: MemoryKernelPort | None = None) -> FastAPI:
    key_store = ApiKeyStore(settings.state_dir / "api-key")
    selected_kernel = (
        memory_kernel if memory_kernel is not None else select_memory_kernel(settings)
    )
    markdown = MarkdownIt("commonmark", {"html": False})
    session_fixed_now: datetime | None = None
    if settings.session_now is not None:
        session_fixed_now = datetime.fromisoformat(settings.session_now.replace("Z", "+00:00"))
        if session_fixed_now.tzinfo is None:
            session_fixed_now = session_fixed_now.replace(tzinfo=UTC)
        session_fixed_now = session_fixed_now.astimezone(UTC)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        key_store.ensure()
        managed_kernel = (
            cast(ManagedMemoryKernelPort, selected_kernel)
            if isinstance(selected_kernel, ManagedMemoryKernelPort)
            else None
        )
        if managed_kernel is not None:
            await managed_kernel.start()
        try:
            yield
        finally:
            if managed_kernel is not None:
                await managed_kernel.close()

    app = FastAPI(title="Personal Agent Memory", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.memory_kernel = selected_kernel
    app.state.management_session_clock = (
        (lambda: session_fixed_now) if session_fixed_now is not None else lambda: datetime.now(UTC)
    )
    session_store = ManagementSessionStore(
        settings.state_dir / "management-sessions.json",
        now=lambda: app.state.management_session_clock(),
    )
    app.state.management_session_store = session_store

    def authenticate(
        authorization: str | None = Header(default=None),
        session_cookie: str | None = Cookie(default=None, alias="pam_management_session"),
    ) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        stored = key_store.read()
        if (
            scheme.lower() == "bearer"
            and supplied
            and stored
            and hmac.compare_digest(supplied, stored)
        ):
            return
        if session_store.verify(session_cookie or "", stored):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


    async def status_payload() -> dict[str, str]:
        key = key_store.read()
        kernel_health = await asyncio.to_thread(selected_kernel.health)
        graph_status = await asyncio.to_thread(selected_kernel.graph_status)
        return {
            "status": "ready" if kernel_health.status == "ready" else "degraded",
            "memory_kernel": kernel_health.kind,
            "memory_kernel_status": kernel_health.status,
            "memory_kernel_detail": kernel_health.detail,
            "graph_index_status": str(graph_status.get("status", "not_configured")),
            "api_key_fingerprint": key_store.fingerprint(key),
        }

    @app.exception_handler(MemoryIdempotencyConflictError)
    async def idempotency_conflict(
        _: Request, error: MemoryIdempotencyConflictError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(error)},
        )

    @app.exception_handler(MemoryKernelUnavailableError)
    async def kernel_unavailable(_: Request, error: MemoryKernelUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(error)},
        )


    @app.get("/health/live", dependencies=[Depends(authenticate)])
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/status", dependencies=[Depends(authenticate)])
    async def api_status() -> dict[str, str]:
        return await status_payload()

    @app.get("/mcp/health", dependencies=[Depends(authenticate)])
    async def mcp_status() -> dict[str, str]:
        return await status_payload()

    @app.post("/api/v1/auth/rotate", dependencies=[Depends(authenticate)])
    def rotate_key() -> dict[str, str]:
        old_key = key_store.read()
        new_key = key_store.rotate()
        session_store.invalidate_key(old_key)
        return {"fingerprint": key_store.fingerprint(new_key)}

    @app.post("/api/v1/auth/login")
    @app.post("/api/v1/auth/session")
    def create_management_session(
        login: ManagementSessionLogin, response: Response
    ) -> dict[str, object]:
        stored = key_store.read()
        if not stored or not hmac.compare_digest(login.api_key, stored):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token, expires_at = session_store.create(stored)
        response.set_cookie(
            "pam_management_session",
            token,
            max_age=int(ManagementSessionStore.lifetime.total_seconds()),
            expires=expires_at,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return {"expires_at": expires_at.isoformat(), "expires_in_seconds": 86_400}

    @app.post("/api/v1/auth/logout")
    def logout_management_session(
        response: Response,
        session_cookie: str | None = Cookie(default=None, alias="pam_management_session"),
    ) -> dict[str, str]:
        session_store.revoke(session_cookie or "")
        response.delete_cookie("pam_management_session", path="/")
        return {"status": "logged_out"}


    def region_error(error: RegionError) -> HTTPException:
        if isinstance(error, RegionNotFoundError):
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
        if isinstance(error, DuplicateRegionError):
            return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error))

    def binding_error(error: RegionError) -> HTTPException:
        detail = str(error)
        if detail in {"memory region not found", "project binding not found"}:
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)

    @app.post(
        "/api/v1/regions",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_201_CREATED,
    )
    async def create_region(region: MemoryRegionCreate) -> dict[str, object]:
        try:
            return selected_kernel.create_region(
                CreateMemoryRegionRequest(
                    kind=region.kind,
                    name=region.name,
                    region_id=region.region_id,
                    session_id=region.session_id,
                )
            )
        except RegionError as error:
            raise region_error(error) from error

    @app.get("/api/v1/regions", dependencies=[Depends(authenticate)])
    async def list_regions() -> list[dict[str, object]]:
        return selected_kernel.list_regions()

    @app.get("/api/v1/regions/tree", dependencies=[Depends(authenticate)])
    async def memory_region_tree(session_id: str | None = None) -> dict[str, object]:
        try:
            return selected_kernel.region_tree(MemoryRegionTreeRequest(session_id=session_id))
        except RegionError as error:
            raise region_error(error) from error

    @app.get("/api/v1/regions/{region_id}", dependencies=[Depends(authenticate)])
    async def get_region(region_id: str) -> dict[str, object]:
        try:
            return selected_kernel.get_region(GetMemoryRegionRequest(region_id=region_id))
        except RegionError as error:
            raise region_error(error) from error







    @app.post(
        "/api/v1/libraries/{library_id}/graph-index/rebuild",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def rebuild_graph_index(library_id: str) -> dict[str, object]:
        try:
            return await selected_kernel.rebuild_graph(library_id)
        except MemoryRecordNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

    @app.get(
        "/api/v1/libraries/{library_id}/graph-status",
        dependencies=[Depends(authenticate)],
    )
    async def graph_status(library_id: str) -> dict[str, object]:
        try:
            return await asyncio.to_thread(
                selected_kernel.graph_status,
                library_id,
            )
        except MemoryRecordNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error






    def project_bindings_payload() -> list[dict[str, str]]:
        return selected_kernel.list_region_bindings()

    @app.get("/api/v1/project-bindings/resolve", dependencies=[Depends(authenticate)])
    async def resolve_project_binding(cwd: str) -> dict[str, str]:
        try:
            return selected_kernel.resolve_region(ResolveRegionRequest(cwd=cwd))
        except RegionError as error:
            raise binding_error(error) from error

    @app.post("/mcp/project-bindings/resolve", dependencies=[Depends(authenticate)])
    async def mcp_resolve_project_binding(resolution: CwdResolution) -> dict[str, str]:
        try:
            return selected_kernel.resolve_region(ResolveRegionRequest(cwd=resolution.cwd))
        except RegionError as error:
            raise binding_error(error) from error

    @app.post(
        "/api/v1/project-bindings/worktrees",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_201_CREATED,
    )
    async def associate_worktree(association: WorktreeAssociation) -> dict[str, str]:
        try:
            return selected_kernel.associate_worktree_region(
                AssociateWorktreeRegionRequest(
                    worktree_root=association.worktree_root,
                    main_project_binding_id=association.main_project_binding_id,
                )
            )
        except DuplicateRegionBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(error), "binding_id": error.binding_id},
            ) from error
        except RegionError as error:
            raise binding_error(error) from error

    @app.post(
        "/api/v1/project-bindings",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_201_CREATED,
    )
    async def bind_project(binding: ProjectBindingRegistration) -> dict[str, str]:
        try:
            return selected_kernel.bind_project_region(
                BindProjectRegionRequest(
                    project_root=binding.project_root,
                    region_id=binding.target_region_id,
                )
            )
        except DuplicateRegionBindingError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(error), "binding_id": error.binding_id},
            ) from error
        except RegionError as error:
            raise binding_error(error) from error

    @app.get("/api/v1/project-bindings", dependencies=[Depends(authenticate)])
    async def list_project_bindings() -> list[dict[str, str]]:
        return project_bindings_payload()

    @app.put("/api/v1/project-bindings/{binding_id}", dependencies=[Depends(authenticate)])
    async def update_project_binding(
        binding_id: str, update: ProjectBindingUpdate
    ) -> dict[str, str]:
        try:
            return selected_kernel.update_region_binding(
                UpdateRegionBindingRequest(
                    binding_id=binding_id,
                    region_id=update.target_region_id,
                )
            )
        except RegionError as error:
            raise binding_error(error) from error

    @app.delete(
        "/api/v1/project-bindings/{binding_id}",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_project_binding(binding_id: str) -> Response:
        try:
            selected_kernel.delete_region_binding(DeleteRegionBindingRequest(binding_id=binding_id))
        except RegionBindingNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        except RegionBindingError as error:
            raise binding_error(error) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)


    async def search_payload(request: SearchRequest) -> dict[str, object]:
        try:
            raw = await selected_kernel.search(
                SearchMemoryRequest(
                    query=request.query,
                    cwd=request.cwd,
                    library_id=request.library_id,
                    region_id=request.region_id,
                    region_ids=(None if request.region_ids is None else tuple(request.region_ids)),
                    limit=request.limit,
                    max_graph_hops=request.graph_hops,
                    include_history=request.include_history,
                    session_id=request.session_id,
                    topic_id=request.topic_id,
                )
            )
            if request.region_ids is not None:
                raw["cross_region_budget"] = (
                    {}
                    if request.cross_region_budget is None
                    else request.cross_region_budget.model_dump()
                )
            return build_context_package(
                raw,
                requested_budget=request.token_budget,
                target_model=request.target_model,
            )
        except RegionError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.post(
        "/api/v1/memories",
        dependencies=[Depends(authenticate)],
        status_code=status.HTTP_201_CREATED,
    )
    async def create_memory(memory: MemoryCreate) -> dict[str, object]:
        try:
            return await selected_kernel.write(
                CreateMemoryRequest(
                    library_id=memory.library_id or "",
                    content=memory.content,
                    memory_type=memory.memory_type,
                    source_references=tuple(memory.source_references),
                    actor=memory.actor,
                    idempotency_key=memory.idempotency_key,
                    region_id=memory.region_id,
                )
            )
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



    @app.post(
        "/api/v1/capture/events",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authenticate)],
    )
    async def capture_event(
        event: CaptureEvent,
        client_deadline_ms: str | None = Header(
            default=None, alias=CAPTURE_PERSISTENCE_DEADLINE_HEADER
        ),
    ) -> dict[str, object]:
        loop = asyncio.get_running_loop()
        monotonic_before_wall_sample = loop.time()
        request_budget = capture_request_budget_seconds(
            client_deadline_ms,
            now_epoch_seconds=time.time(),
            maximum_seconds=(
                2.0
            ),
        )
        request_deadline = monotonic_before_wall_sample + request_budget
        persistence_deadline = request_deadline - CAPTURE_PERSISTENCE_COMMIT_MARGIN_SECONDS
        try:
            async with asyncio.timeout_at(request_deadline):
                return await selected_kernel.write(
                    CaptureMemoryRequest(
                        event_id=event.event_id,
                        session_id=event.session_id,
                        turn_id=event.turn_id,
                        event_kind=event.event_kind,
                        content=event.content,
                        occurred_at=event.occurred_at,
                        cwd=event.cwd,
                        deadline=persistence_deadline,
                        topic_id=event.topic_id,
                    )
                )
        except CaptureBindingUnstableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except TimeoutError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="capture timed out while waiting for a stable project binding",
            ) from error
        except CaptureContractError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        except RegionError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.get("/api/v1/capture/events", dependencies=[Depends(authenticate)])
    async def capture_events(
        session_id: Annotated[str | None, Query(max_length=MAX_SESSION_ID_LENGTH)] = None,
        region_id: Annotated[str | None, Query(max_length=MAX_REGION_ID_LENGTH)] = None,
        turn_id: Annotated[str | None, Query(max_length=200)] = None,
    ) -> list[dict[str, object]]:
        try:
            return selected_kernel.list_capture_events(
                CaptureEventsRequest(
                    session_id=session_id,
                    region_id=region_id,
                    turn_id=turn_id,
                )
            )
        except RegionError as error:
            raise region_error(error) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error



    @app.get("/api/v1/capture/rounds", dependencies=[Depends(authenticate)])
    async def capture_rounds(
        session_id: Annotated[str | None, Query(max_length=MAX_SESSION_ID_LENGTH)] = None,
        region_id: Annotated[str | None, Query(max_length=MAX_REGION_ID_LENGTH)] = None,
        turn_id: Annotated[str | None, Query(max_length=200)] = None,
    ) -> list[dict[str, object]]:
        try:
            return selected_kernel.list_capture_rounds(
                CaptureRoundsRequest(
                    session_id=session_id,
                    region_id=region_id,
                    turn_id=turn_id,
                )
            )
        except RegionError as error:
            raise region_error(error) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error




    @app.post(
        "/api/v1/capture/consolidate",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authenticate)],
    )
    async def consolidate_capture(
        request: CaptureConsolidation,
        client_deadline_ms: str | None = Header(
            default=None, alias=CAPTURE_PERSISTENCE_DEADLINE_HEADER
        ),
    ) -> dict[str, int]:
        loop = asyncio.get_running_loop()
        monotonic_before_wall_sample = loop.time()
        request_budget = capture_request_budget_seconds(
            client_deadline_ms,
            now_epoch_seconds=time.time(),
            maximum_seconds=(
                2.0
            ),
        )
        request_deadline = monotonic_before_wall_sample + request_budget
        persistence_deadline = request_deadline - CAPTURE_PERSISTENCE_COMMIT_MARGIN_SECONDS
        try:
            # The deadline is a pre-commit admission boundary. Once SQLite commit starts,
            # await its definitive result rather than returning 503 while it can still land.
            evolved = await selected_kernel.evolve(
                EvolveMemoryRequest(
                    session_id=request.session_id,
                    deadline=persistence_deadline,
                )
            )
            return {"queued": evolved["queued"]}
        except TimeoutError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="capture consolidation timed out",
            ) from error

    @app.post(
        "/api/v1/promotions",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authenticate)],
    )
    async def run_promotion(request: PromotionRun) -> dict[str, object]:
        try:
            return await selected_kernel.start_promotion(
                PromotionRunRequest(
                    source_region_ids=(
                        None
                        if request.source_region_ids is None
                        else tuple(request.source_region_ids)
                    )
                )
            )
        except MemoryKernelUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except MemoryRecordNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

    @app.get(
        "/api/v1/promotions/{job_id}",
        dependencies=[Depends(authenticate)],
    )
    async def promotion_status(job_id: str) -> dict[str, object]:
        try:
            return selected_kernel.promotion_status(PromotionStatusRequest(job_id=job_id))
        except MemoryRecordNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except MemoryKernelUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

    @app.get(
        "/api/v1/libraries/{library_id}/retention-policy",
        dependencies=[Depends(authenticate)],
    )
    async def retention_policy(library_id: str) -> dict[str, object]:
        try:
            return selected_kernel.get_retention_policy(
                RetentionPolicyRequest(library_id=library_id)
            )
        except RegionError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    @app.put(
        "/api/v1/libraries/{library_id}/retention-policy",
        dependencies=[Depends(authenticate)],
    )
    async def update_retention_policy(
        library_id: str, policy: RetentionPolicyUpdate
    ) -> dict[str, object]:
        try:
            return await selected_kernel.update_retention_policy(
                UpdateRetentionPolicyRequest(library_id=library_id, **policy.model_dump())
            )
        except RegionError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    @app.get(
        "/api/v1/libraries/{library_id}/retention-cleanup/preview",
        dependencies=[Depends(authenticate)],
    )
    async def preview_retention_cleanup(library_id: str) -> dict[str, object]:
        try:
            return await selected_kernel.preview_retention_cleanup(
                RetentionCleanupRequest(library_id=library_id)
            )
        except RegionError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    @app.post(
        "/api/v1/libraries/{library_id}/retention-cleanup",
        dependencies=[Depends(authenticate)],
    )
    async def run_retention_cleanup(library_id: str) -> dict[str, object]:
        try:
            return await selected_kernel.run_retention_cleanup(
                RetentionCleanupRequest(library_id=library_id)
            )
        except RegionError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error













    def mutation_error(error: Exception) -> HTTPException:
        detail = str(error)
        if isinstance(error, MemoryRecordNotFoundError):
            code = status.HTTP_404_NOT_FOUND
        elif isinstance(error, (MemoryRevisionConflictError, MemoryLifecycleConflictError)):
            code = status.HTTP_409_CONFLICT
        elif detail in {
            "memory library not found",
            "memory document not found",
            "forgotten memory not found",
        }:
            code = status.HTTP_404_NOT_FOUND
        elif (
            "version conflict" in detail
            or "changed outside" in detail
            or "out-of-band conflict" in detail
            or "version changed again" in detail
            or "withheld sensitive external content" in detail
            or "deletion preview" in detail
        ):
            code = status.HTTP_409_CONFLICT
        else:
            code = status.HTTP_422_UNPROCESSABLE_CONTENT
        return HTTPException(status_code=code, detail=detail)

    @app.get("/api/v1/libraries/{library_id}/documents", dependencies=[Depends(authenticate)])
    async def list_documents(library_id: str) -> list[dict[str, object]]:
        try:
            return selected_kernel.list_documents(ListDocumentsRequest(library_id=library_id))
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.get(
        "/api/v1/libraries/{library_id}/memories",
        dependencies=[Depends(authenticate)],
    )
    async def list_memories(
        request: Request,
        library_id: str,
        cursor: str | None = Query(default=None, max_length=2_048),
        q: str | None = Query(default=None, max_length=500),
        memory_type: str | None = Query(default=None, max_length=100),
        lifecycle: str | None = Query(default=None, max_length=32),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, object]:
        allowed = {"cursor", "q", "memory_type", "lifecycle", "limit"}
        unknown = sorted(set(request.query_params) - allowed)
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"unknown memory list parameter: {', '.join(unknown)}",
            )
        try:
            return await asyncio.to_thread(
                selected_kernel.list_memories,
                ListMemoriesRequest(
                    library_id=library_id,
                    cursor=cursor,
                    q=q,
                    memory_type=memory_type,
                    lifecycle=lifecycle,
                    limit=limit,
                ),
            )
        except (MemoryGovernanceError, ValueError) as error:
            raise mutation_error(error) from error

    @app.get(
        "/api/v1/libraries/{library_id}/memories/{memory_id}",
        dependencies=[Depends(authenticate)],
    )
    async def read_memory(library_id: str, memory_id: str) -> dict[str, object]:
        try:
            return await asyncio.to_thread(
                selected_kernel.inspect_memory,
                DocumentInspectRequest(library_id=library_id, path=memory_id),
            )
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.get(
        "/api/v1/libraries/{library_id}/export.md",
        dependencies=[Depends(authenticate)],
        response_class=PlainTextResponse,
    )
    async def export_library_markdown(library_id: str) -> PlainTextResponse:
        try:
            documents = selected_kernel.list_documents(ListDocumentsRequest(library_id=library_id))
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error
        lines = [f"# Memory export: {library_id}", ""]
        for document in documents:
            if document.get("lifecycle") not in {"active", "archived"}:
                continue
            lines.extend(
                [
                    f"## {document.get('id', 'unknown')}",
                    "",
                    f"- Lifecycle: {document.get('lifecycle', '')}",
                    f"- Version: {document.get('version', '')}",
                    f"- Source: {document.get('source', '')}",
                    "",
                    str(document.get("content", "")),
                    "",
                ]
            )
        return PlainTextResponse("\n".join(lines), media_type="text/markdown; charset=utf-8")

    @app.get("/api/v1/libraries/{library_id}/document", dependencies=[Depends(authenticate)])
    async def read_document(library_id: str, path: str) -> dict[str, object]:
        try:
            return selected_kernel.inspect(DocumentInspectRequest(library_id=library_id, path=path))
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.post(
        "/api/v1/libraries/{library_id}/document/preview",
        dependencies=[Depends(authenticate)],
    )
    async def preview_document(library_id: str, preview: DocumentPreview) -> dict[str, object]:
        try:
            result = selected_kernel.preview_document_update(
                PreviewDocumentUpdateRequest(
                    library_id=library_id,
                    path=preview.path,
                    content=preview.content,
                    expected_revision=preview.expected_source_version,
                )
            )
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error
        result["rendered_html"] = markdown.render(preview.content)
        return result

    @app.put("/api/v1/libraries/{library_id}/document", dependencies=[Depends(authenticate)])
    async def edit_document(library_id: str, edit: DocumentEdit) -> dict[str, str]:
        try:
            updated = await selected_kernel.update(
                DocumentUpdateRequest(
                    library_id=library_id,
                    path=edit.path,
                    content=edit.content,
                    expected_revision=edit.expected_source_version,
                    operation_id=edit.operation_id,
                    actor=edit.actor_type,
                    source=edit.source,
                    mode=edit.mode,
                )
            )
            return {key: str(value) for key, value in updated.items()}
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.post(
        "/api/v1/libraries/{library_id}/document/delete-preview",
        dependencies=[Depends(authenticate)],
    )
    async def preview_document_deletion(
        library_id: str, deletion: DocumentDeletePreview
    ) -> dict[str, object]:
        try:
            return await selected_kernel.preview_document_delete(
                PreviewDocumentDeleteRequest(
                    library_id=library_id,
                    path=deletion.path,
                    expected_revision=deletion.expected_source_version,
                )
            )
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.delete(
        "/api/v1/libraries/{library_id}/document",
        dependencies=[Depends(authenticate)],
    )
    async def delete_document(
        library_id: str,
        deletion: DocumentDelete,
    ) -> dict[str, object]:
        try:
            return await selected_kernel.delete(
                DeleteMemoryRequest(
                    library_id=library_id,
                    path=deletion.path,
                    expected_revision=deletion.expected_source_version,
                    operation_id=deletion.operation_id,
                    actor=deletion.actor_type,
                    source=deletion.source,
                    preview_token=deletion.preview_token,
                    mode=deletion.mode,
                )
            )
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.get(
        "/api/v1/libraries/{library_id}/forgotten",
        dependencies=[Depends(authenticate)],
    )
    async def list_forgotten_memories(library_id: str) -> list[dict[str, object]]:
        try:
            return selected_kernel.list_forgotten(ListForgottenRequest(library_id=library_id))
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.post(
        "/api/v1/libraries/{library_id}/forgotten/restore",
        dependencies=[Depends(authenticate)],
    )
    async def restore_forgotten_memory(
        library_id: str, restore: ForgottenMemoryRestore
    ) -> dict[str, str]:
        try:
            return await selected_kernel.restore_forgotten(
                RestoreForgottenRequest(
                    library_id=library_id,
                    tombstone_id=restore.tombstone_id,
                    commit=restore.commit,
                    operation_id=restore.operation_id,
                    actor=restore.actor_type,
                    source=restore.source,
                )
            )
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.get("/api/v1/libraries/{library_id}/history", dependencies=[Depends(authenticate)])
    async def library_history(
        library_id: str,
        limit: int = Query(default=50, ge=1, le=1000),
        target_id: str | None = Query(default=None, max_length=500),
    ) -> list[dict[str, str]]:
        try:
            return await asyncio.to_thread(
                selected_kernel.audit,
                AuditMemoryRequest(library_id=library_id, limit=limit, target_id=target_id),
            )
        except (MemoryGovernanceError, ValueError) as error:
            raise mutation_error(error) from error

    @app.get(
        "/api/v1/libraries/{library_id}/history/{commit}/diff",
        dependencies=[Depends(authenticate)],
    )
    async def history_diff(library_id: str, commit: str) -> dict[str, object]:
        try:
            return selected_kernel.history_diff(
                HistoryDiffRequest(library_id=library_id, commit=commit)
            )
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    @app.post(
        "/api/v1/libraries/{library_id}/history/restore",
        dependencies=[Depends(authenticate)],
    )
    async def restore_document(library_id: str, restore: DocumentRestore) -> dict[str, str]:
        try:
            return await selected_kernel.restore_document(
                RestoreDocumentRequest(
                    library_id=library_id,
                    path=restore.path,
                    commit=restore.commit,
                    expected_revision=restore.expected_source_version,
                    operation_id=restore.operation_id,
                    actor=restore.actor_type,
                    source=restore.source,
                )
            )
        except MemoryGovernanceError as error:
            raise mutation_error(error) from error

    mcp_tools: list[dict[str, object]] = [
        {
            "name": "memory_search",
            "description": (
                "Search one project-bound region, one explicit region, or multiple explicit "
                "regions through cross-region recall."
            ),
            "inputSchema": SearchRequest.model_json_schema(),
        },
        {
            "name": "memory_get",
            "description": "Read one memory unit by region and memory:// path.",
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
            return selected_kernel.inspect(DocumentInspectRequest(library_id=library_id, path=path))
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
        except (MemoryGovernanceError, ValidationError, ValueError) as error:
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


    @app.get("/")
    def management_interface() -> FileResponse:
        return FileResponse(frontend)

    return app
