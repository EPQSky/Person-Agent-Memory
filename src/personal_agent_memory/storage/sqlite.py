"""SQLite-backed JiuwenMemory storage adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from jiuwen_memory.storage.kv_impl.sqlite_kv_store import (
    SQLiteKVStore as _JiuwenSQLiteKVStore,
)


class SQLiteKVStore(_JiuwenSQLiteKVStore):  # type: ignore[misc]
    """Persistent SQLite KV adapter for memory-unit source records."""

    def __init__(self, db_path: str | Path) -> None:
        value = str(db_path)
        if value != ":memory:":
            path = Path(value).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            value = str(path)
        super().__init__(value)

    def close(self) -> None:
        super().close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
