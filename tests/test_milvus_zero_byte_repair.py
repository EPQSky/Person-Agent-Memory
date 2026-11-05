from __future__ import annotations

from pathlib import Path

from personal_agent_memory.storage import milvus as milvus_module
from personal_agent_memory.storage.milvus import (
    _MilvusLiteConnection,
    _prune_dangling_manifest_entries,
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

        def list_collections(self) -> list[str]:
            calls.append(("list_collections", None))
            return ["agent_memory_vectors", "agent_memory_fulltext"]

        def flush(self, name: str, timeout: float | None = None) -> None:
            calls.append((f"flush:{name}", timeout))

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

    assert [name for name, _ in calls] == [
        "list_collections",
        "flush:agent_memory_vectors",
        "flush:agent_memory_fulltext",
        "close",
    ]
    assert calls[1][1] == milvus_module._SHUTDOWN_FLUSH_TIMEOUT_SECONDS
    assert len(released) == 1


def test_dangling_manifest_entries_are_pruned_with_a_backup(tmp_path: Path) -> None:
    """A manifest that points at deleted segments makes every search return zero."""
    root = tmp_path / "milvus.db"
    collection = root / "collections" / "agent_memory_vectors"
    (collection / "partitions" / "_default" / "data").mkdir(parents=True)
    present = collection / "partitions" / "_default" / "data" / "data_000001_000010.parquet"
    present.write_bytes(b"PAR1")
    manifest = collection / "manifest.json"
    manifest.write_text(
        '{"partitions": {"_default": {"data_files": ['
        '"data/data_000001_000010.parquet", "data/data_000011_000020.parquet"], '
        '"delta_files": ["delta/delta_000021_000030.parquet"]}}, "version": 7}',
        encoding="utf-8",
    )

    pruned = _prune_dangling_manifest_entries(str(root))

    assert sorted(pruned) == sorted(
        [
            "agent_memory_vectors/data/data_000011_000020.parquet",
            "agent_memory_vectors/delta/delta_000021_000030.parquet",
        ]
    )
    repaired = manifest.read_text(encoding="utf-8")
    assert "data_000001_000010.parquet" in repaired
    assert "data_000011_000020.parquet" not in repaired
    assert "delta_000021_000030.parquet" not in repaired
    assert "\"version\": 7" in repaired
    stash = root.with_name("milvus.db.quarantine")
    assert list(stash.rglob("agent_memory_vectors/manifest.json"))
    assert _prune_dangling_manifest_entries(str(root)) == []
