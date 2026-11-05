from __future__ import annotations

from pathlib import Path

from personal_agent_memory.storage import milvus as milvus_module
from personal_agent_memory.storage.milvus import (
    _MilvusLiteConnection,
    _quarantine_zero_byte_artifacts,
)


def _collection_file(root: Path, name: str, kind: str, payload: bytes) -> Path:
    path = (
        root / "collections" / "agent_memory_graph_nodes" / "partitions" / "_default" / kind / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_zero_byte_artifacts_are_quarantined_without_touching_real_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "milvus.db"
    truncated = _collection_file(root, "data_000181_000400.parquet", "data", b"")
    truncated_index = _collection_file(
        root, "data_000181_000400.embedding.flat.idx", "indexes", b""
    )
    healthy = _collection_file(root, "data_000015_000125.parquet", "data", b"PAR1data")
    lock = root / "LOCK"
    lock.write_bytes(b"")

    quarantined = _quarantine_zero_byte_artifacts(str(root))

    assert sorted(quarantined) == sorted(
        [
            str(truncated.relative_to(root)),
            str(truncated_index.relative_to(root)),
        ]
    )
    assert not truncated.exists()
    assert not truncated_index.exists()
    assert healthy.exists()
    assert lock.exists(), "the Milvus LOCK file is intentionally empty and must stay"
    stash = root.with_name("milvus.db.quarantine")
    assert list(stash.rglob("data_000181_000400.parquet"))


def test_zero_byte_scan_is_idempotent_and_tolerates_missing_store(tmp_path: Path) -> None:
    root = tmp_path / "milvus.db"
    _collection_file(root, "data_000001_000002.parquet", "data", b"")

    assert _quarantine_zero_byte_artifacts(str(root))
    assert _quarantine_zero_byte_artifacts(str(root)) == []
    assert _quarantine_zero_byte_artifacts(str(tmp_path / "absent")) == []


def test_connection_flushes_before_releasing_the_embedded_server(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Shutdown must seal segments, otherwise the exit leaves 0-byte parquet files."""
    calls: list[tuple[str, object]] = []

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def flush_all(self, timeout: float | None = None) -> None:
            calls.append(("flush_all", timeout))

        def close(self) -> None:
            calls.append(("close", None))

    released: list[str] = []
    monkeypatch.setattr(milvus_module, "MilvusClient", _FakeClient)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        milvus_module.server_manager_instance,
        "release_server",
        lambda uri: released.append(uri),
    )

    connection = _MilvusLiteConnection(str(tmp_path / "milvus.db"))
    assert connection.client is not None
    connection.close()

    assert [name for name, _ in calls] == ["flush_all", "close"]
    assert calls[0][1] == milvus_module._SHUTDOWN_FLUSH_TIMEOUT_SECONDS
    assert len(released) == 1
