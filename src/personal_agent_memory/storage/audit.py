"""Persistent audit storage for the local memory kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from jiuwen_memory.common.audit.audit_impl.sqlite_audit_logger import (
    SqliteAuditLogger as _JiuwenSqliteAuditLogger,
)


class SQLiteAuditLogger(_JiuwenSqliteAuditLogger):  # type: ignore[misc]
    """SQLite audit logger with explicit directory creation and shutdown."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        super().__init__(str(path))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
