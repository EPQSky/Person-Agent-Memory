from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from personal_agent_memory.config import Settings
from personal_agent_memory.security import ApiKeyStore
from personal_agent_memory.state import (
    DuplicateLibraryError,
    DuplicateProjectBindingError,
    LibraryKind,
    LibraryRegistrationError,
    PlatformState,
    ProjectBindingError,
)


class LibraryRegistration(BaseModel):
    path: str
    kind: LibraryKind


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


def create_app(settings: Settings) -> FastAPI:
    key_store = ApiKeyStore(settings.state_dir / "api-key")
    platform_state = PlatformState(
        settings.state_dir / "platform.sqlite3", library_roots=settings.library_roots
    )

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
            return platform_state.register_library(registration.path, registration.kind).payload()
        except DuplicateLibraryError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(error), "library_id": error.library_id},
            ) from error
        except LibraryRegistrationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

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

    frontend = Path(__file__).parent / "static" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def management_interface() -> FileResponse:
        return FileResponse(frontend)

    return app
