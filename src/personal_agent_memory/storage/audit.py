"""Persistent audit storage for the local memory kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from jiuwen_memory.common.audit.audit_impl.sqlite_audit_logger import (
    SqliteAuditLogger as _JiuwenSqliteAuditLogger,
)
from jiuwen_memory.common.audit.audit_impl.sqlite_audit_logger import (
    _row_to_event,
)
from jiuwen_memory.common.type_def import AuditEvent


class SQLiteAuditLogger(_JiuwenSqliteAuditLogger):  # type: ignore[misc]
    """SQLite audit logger with explicit directory creation and shutdown."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        super().__init__(str(path))

    def query_latest(self, filters: dict[str, str], limit: int = 100) -> list[AuditEvent]:
        exact_fields = {
            "action": "action",
            "layer": "layer",
            "decision": "decision",
            "target_id": "target_id",
            "actor_org": "actor_org",
            "actor_space": "actor_space",
            "actor_user": "actor_user",
            "actor_agent": "actor_agent",
            "actor_session": "actor_session",
            "target_org": "target_org",
            "target_space": "target_space",
            "target_user": "target_user",
            "target_agent": "target_agent",
            "target_session": "target_session",
        }
        clauses: list[str] = []
        params: list[object] = []
        for field, column in exact_fields.items():
            if filters.get(field):
                clauses.append(f"{column} = ?")
                params.append(filters[field])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_events {where} ORDER BY seq DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_event(row) for row in rows]

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
