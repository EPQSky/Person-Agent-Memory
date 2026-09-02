from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse

from personal_agent_memory.config import Settings
from personal_agent_memory.security import ApiKeyStore
from personal_agent_memory.state import PlatformState


def create_app(settings: Settings) -> FastAPI:
    key_store = ApiKeyStore(settings.state_dir / "api-key")
    platform_state = PlatformState(settings.state_dir / "platform.sqlite3")

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

    frontend = Path(__file__).parent / "static" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def management_interface() -> FileResponse:
        return FileResponse(frontend)

    return app
