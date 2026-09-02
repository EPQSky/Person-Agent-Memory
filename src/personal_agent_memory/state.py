from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path


class PlatformState:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.connection: sqlite3.Connection | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self.stop_worker = asyncio.Event()
        self.startup_count = 0
        self.previous_shutdown_clean = True

    async def start(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.database_path)
        self.connection = connection
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity != ("ok",):
                raise sqlite3.DatabaseError(
                    f"state database integrity check failed: {integrity!r}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS platform_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS background_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            previous_count = self._metadata("startup_count")
            previous_clean = self._metadata("clean_shutdown")
            self.startup_count = int(previous_count or "0") + 1
            self.previous_shutdown_clean = previous_clean != "false"
            self._set_metadata("startup_count", str(self.startup_count))
            self._set_metadata("clean_shutdown", "false")
            connection.commit()
        except BaseException:
            connection.close()
            self.connection = None
            raise
        self.stop_worker.clear()
        self.worker_task = asyncio.create_task(self._worker(), name="memory-background-worker")

    async def close(self) -> None:
        if self.worker_task is not None:
            self.stop_worker.set()
            await self.worker_task
        if self.connection is not None:
            self._set_metadata("clean_shutdown", "true")
            self.connection.commit()
            self.connection.close()
            self.connection = None

    async def _worker(self) -> None:
        while not self.stop_worker.is_set():
            self._execute_pending_jobs()
            try:
                await asyncio.wait_for(self.stop_worker.wait(), timeout=0.1)
            except TimeoutError:
                continue

    def enqueue_job(self, kind: str, payload: str) -> int:
        assert self.connection is not None
        cursor = self.connection.execute(
            "INSERT INTO background_jobs (kind, payload) VALUES (?, ?)", (kind, payload)
        )
        self.connection.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def job_status(self, job_id: int) -> str | None:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT status FROM background_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def _execute_pending_jobs(self) -> None:
        assert self.connection is not None
        self.connection.execute(
            "UPDATE background_jobs SET status = 'done' WHERE status = 'pending'"
        )
        self.connection.commit()

    def _metadata(self, key: str) -> str | None:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT value FROM platform_metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def _set_metadata(self, key: str, value: str) -> None:
        assert self.connection is not None
        self.connection.execute(
            """
            INSERT INTO platform_metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
