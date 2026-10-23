from __future__ import annotations

import fcntl
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jiuwen_memory.common.errors import BackendError, ConflictError, NotFoundError
from jiuwen_memory.common.type_def import (
    AuditEvent,
    FilterClause,
    FilterOp,
    MemoryUnit,
    Scope,
    Segment,
)
from jiuwen_memory.common.type_def.memory import memory_key
from jiuwen_memory.common.type_def.memory_codec import dumps, loads
from jiuwen_memory.storage.base import StoreType
from jiuwen_memory.storage.types import Document, TextQuery, VectorQuery, VectorRecord
from pymilvus.exceptions import MilvusException

from personal_agent_memory.storage import (
    MilvusLiteFulltextStore,
    MilvusLiteVectorStore,
    SQLiteAuditLogger,
    SQLiteKVStore,
)


def _lock_is_available(data_dir: Path) -> bool:
    lock_path = data_dir / "LOCK"
    if not lock_path.exists():
        return True
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return True


def test_sqlite_kv_crud_scope_isolation_and_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    scope = Scope(org="local", space="project:alpha")
    other_scope = Scope(org="local", space="project:beta")
    unit = MemoryUnit(
        id="unit-1",
        scope=scope,
        segments=[Segment(content="first value")],
    )

    first = SQLiteKVStore(db_path)
    try:
        assert first.store_type() is StoreType.KV
        first.insert(scope, memory_key(unit.id), dumps(unit))
        assert loads(first.get(scope, memory_key(unit.id))) == unit
        assert not first.exists(other_scope, memory_key(unit.id))
        with pytest.raises(ConflictError):
            first.insert(scope, memory_key(unit.id), dumps(unit))

        updated = replace(unit, segments=[Segment(content="updated value")])
        first.update(scope, memory_key(unit.id), dumps(updated))
        assert loads(first.get(scope, memory_key(unit.id))) == updated
    finally:
        first.close()
        first.close()

    reopened = SQLiteKVStore(db_path)
    try:
        assert loads(reopened.get(scope, memory_key(unit.id))) == updated
        reopened.delete(scope, memory_key(unit.id))
        reopened.delete(scope, memory_key(unit.id))
        assert not reopened.exists(scope, memory_key(unit.id))
    finally:
        reopened.close()


def test_milvus_dense_vector_crud_filters_persistence_and_shutdown(tmp_path: Path) -> None:
    data_dir = tmp_path / "vectors.db"
    scope = Scope(org="local", space="project:alpha")
    other_scope = Scope(org="local", space="project:beta")
    store = MilvusLiteVectorStore(uri=data_dir, dim=3)
    try:
        assert store.store_type() is StoreType.VECTOR
        assert store.client._config.db_name == "default"
        store.insert(
            scope,
            [
                VectorRecord(
                    id="alpha",
                    vector=[1.0, 0.0, 0.0],
                    metadata={"user_metadata.kind": "one", "unit_id": "alpha"},
                ),
                VectorRecord(
                    id="beta",
                    vector=[0.0, 1.0, 0.0],
                    metadata={"user_metadata.kind": "two", "unit_id": "beta"},
                ),
            ],
        )
        store.insert(
            other_scope,
            [VectorRecord(id="foreign", vector=[1.0, 0.0, 0.0])],
        )
        with pytest.raises(ConflictError):
            store.insert(scope, [VectorRecord(id="alpha", vector=[1.0, 0.0, 0.0])])

        assert {record.id for record in store.get(scope, ["alpha", "beta"])} == {
            "alpha",
            "beta",
        }
        assert store.get(other_scope, ["alpha"]) == []
        assert store.search(scope, VectorQuery(vector=[1.0, 0.0, 0.0], top_k=1))[0].id == (
            "alpha"
        )
        assert store.search(
            scope,
            VectorQuery(
                vector=[1.0, 0.0, 0.0],
                top_k=10,
                filters=FilterClause("kind", FilterOp.IN, ["one"]),
            ),
        )[0].id == "alpha"
        assert [
            hit.id
            for hit in store.search(
                scope,
                VectorQuery(
                    vector=[1.0, 0.0, 0.0],
                    top_k=10,
                    filters=FilterClause("kind", FilterOp.NOT_IN, ["one"]),
                ),
            )
        ] == ["beta"]

        store.update(
            scope,
            [VectorRecord(id="alpha", vector=[0.0, 0.0, 1.0], metadata={})],
        )
        assert store.search(scope, VectorQuery(vector=[0.0, 0.0, 1.0], top_k=1))[0].id == (
            "alpha"
        )
        store.delete(scope, ["alpha"])
        assert store.get(scope, ["alpha"]) == []
    finally:
        store.close()
        store.close()

    assert _lock_is_available(data_dir)
    reopened = MilvusLiteVectorStore(uri=data_dir, dim=3)
    try:
        assert [record.id for record in reopened.get(scope, ["beta"])] == ["beta"]
    finally:
        reopened.close()


def test_milvus_bm25_crud_jieba_filters_scores_and_persistence(tmp_path: Path) -> None:
    data_dir = tmp_path / "fulltext.db"
    scope = Scope(org="local", space="project:alpha")
    other_scope = Scope(org="local", space="project:beta")
    store = MilvusLiteFulltextStore(uri=data_dir)
    try:
        assert store.store_type() is StoreType.FULLTEXT
        store.insert(
            scope,
            [
                Document(
                    id="sqlite",
                    text="Python 项目使用 SQLite 存储数据",
                    metadata={"user_metadata.kind": "one", "unit_id": "sqlite"},
                ),
                Document(
                    id="ml-project",
                    text="机器学习项目使用 Python",
                    metadata={"user_metadata.kind": "two", "unit_id": "ml-project"},
                ),
                Document(
                    id="ml-course",
                    text="机器学习课程笔记",
                    metadata={"user_metadata.kind": "two", "unit_id": "ml-course"},
                ),
            ],
        )
        store.insert(
            other_scope,
            [Document(id="foreign", text="Python 项目", metadata={})],
        )
        with pytest.raises(ConflictError):
            store.insert(scope, [Document(id="sqlite", text="duplicate")])

        chinese_hits = store.search(scope, TextQuery(text="机器学习", top_k=10))
        assert {hit.id for hit in chinese_hits} == {"ml-course", "ml-project"}
        assert all(hit.score > 0 for hit in chinese_hits)

        ranked = store.search(scope, TextQuery(text="项目 SQLite", top_k=10))
        assert ranked[0].id == "sqlite"
        assert ranked[0].score > ranked[1].score

        assert [
            hit.id
            for hit in store.search(
                scope,
                TextQuery(
                    text="机器学习",
                    top_k=10,
                    filters=FilterClause("kind", FilterOp.IN, ["two"]),
                ),
            )
        ] == ["ml-course", "ml-project"]
        assert [
            hit.id
            for hit in store.search(
                scope,
                TextQuery(
                    text="机器学习",
                    top_k=10,
                    filters=FilterClause("kind", FilterOp.NOT_IN, ["two"]),
                ),
            )
        ] == []

        assert store.get(other_scope, ["sqlite"]) == []
        assert store.get(scope, ["sqlite"])[0].text.startswith("Python")

        store.update(
            scope,
            [Document(id="sqlite", text="Rust 所有权规则", metadata={"kind": "three"})],
        )
        assert {hit.id for hit in store.search(scope, TextQuery(text="Python", top_k=10))} == {
            "ml-project"
        }
        store.delete(other_scope, ["sqlite"])
        assert store.get(scope, ["sqlite"])
        store.delete(scope, ["sqlite", "missing"])
        store.delete(scope, ["sqlite"])
        assert store.get(scope, ["sqlite"]) == []
        with pytest.raises(NotFoundError):
            store.update(scope, [Document(id="missing", text="missing")])
    finally:
        store.close()
        store.close()

    assert _lock_is_available(data_dir)
    reopened = MilvusLiteFulltextStore(uri=data_dir)
    try:
        assert {hit.id for hit in reopened.search(scope, TextQuery(text="学习", top_k=10))} == {
            "ml-course",
            "ml-project",
        }
        assert reopened.get(scope, ["ml-project"])[0].text == "机器学习项目使用 Python"
    finally:
        reopened.close()


def test_metadata_ne_and_not_in_include_missing_keys(tmp_path: Path) -> None:
    data_dir = tmp_path / "missing-metadata.db"
    scope = Scope(org="local", space="project:alpha")
    vectors = MilvusLiteVectorStore(uri=data_dir, dim=2)
    fulltext = MilvusLiteFulltextStore(uri=data_dir)
    try:
        vectors.insert(
            scope,
            [
                VectorRecord(id="missing", vector=[1.0, 0.0]),
                VectorRecord(
                    id="one",
                    vector=[1.0, 0.0],
                    metadata={"user_metadata.kind": "one"},
                ),
                VectorRecord(
                    id="two",
                    vector=[1.0, 0.0],
                    metadata={"user_metadata.kind": "two"},
                ),
            ],
        )
        fulltext.insert(
            scope,
            [
                Document(id="missing", text="match", metadata={}),
                Document(
                    id="one",
                    text="match",
                    metadata={"user_metadata.kind": "one"},
                ),
                Document(
                    id="two",
                    text="match",
                    metadata={"user_metadata.kind": "two"},
                ),
            ],
        )

        for op, value in ((FilterOp.NE, "one"), (FilterOp.NOT_IN, ["one"])):
            vector_ids = {
                hit.id
                for hit in vectors.search(
                    scope,
                    VectorQuery(
                        vector=[1.0, 0.0],
                        top_k=10,
                        filters=FilterClause("kind", op, value),
                    ),
                )
            }
            fulltext_ids = {
                hit.id
                for hit in fulltext.search(
                    scope,
                    TextQuery(
                        text="match",
                        top_k=10,
                        filters=FilterClause("kind", op, value),
                    ),
                )
            }
            assert vector_ids == {"missing", "two"}
            assert fulltext_ids == {"missing", "two"}
    finally:
        vectors.close()
        fulltext.close()


@pytest.mark.parametrize("operation", ["insert", "update"])
@pytest.mark.parametrize("failure_point", ["existing_ids", "client"])
def test_fulltext_insert_update_wrap_backend_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure_point: str,
) -> None:
    store = MilvusLiteFulltextStore(uri=tmp_path / "fulltext-failure.db")
    scope = Scope(org="local", space="project:alpha")

    def fail(*args: object, **kwargs: object) -> object:
        raise MilvusException(message="backend down")

    try:
        if failure_point == "existing_ids":
            monkeypatch.setattr(store, "_existing_ids", fail)
        else:
            monkeypatch.setattr(store, "_get_client", fail)
        with pytest.raises(BackendError, match=f"milvus fulltext {operation}"):
            getattr(store, operation)(scope, [Document(id="unit", text="value")])
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["insert", "update"])
@pytest.mark.parametrize("failure_point", ["existing_ids", "client"])
def test_vector_insert_update_wrap_backend_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure_point: str,
) -> None:
    store = MilvusLiteVectorStore(uri=tmp_path / "vector-failure.db", dim=2)
    scope = Scope(org="local", space="project:alpha")

    def fail(*args: object, **kwargs: object) -> object:
        raise MilvusException(message="backend down")

    class BrokenConnection:
        @property
        def client(self) -> object:
            raise MilvusException(message="backend down")

        def close(self) -> None:
            pass

    try:
        if failure_point == "existing_ids":
            monkeypatch.setattr(store, "_existing_ids", fail)
        else:
            monkeypatch.setattr(store, "_connection", BrokenConnection())
        expected_context = (
            "milvus vector connect" if failure_point == "client" else f"milvus vector {operation}"
        )
        with pytest.raises(BackendError, match=expected_context):
            getattr(store, operation)(
                scope,
                [VectorRecord(id="unit", vector=[1.0, 0.0])],
            )
    finally:
        store.close()


def test_scope_keys_are_unambiguous_for_vector_and_fulltext(tmp_path: Path) -> None:
    data_dir = tmp_path / "scope-encoding.db"
    first_scope = Scope(org="local", space="project:a:b")
    second_scope = Scope(org="local", space="project:a_b")
    vectors = MilvusLiteVectorStore(uri=data_dir, dim=2)
    fulltext = MilvusLiteFulltextStore(uri=data_dir)
    try:
        vectors.insert(
            first_scope,
            [VectorRecord(id="shared", vector=[1.0, 0.0], metadata={"owner": "first"})],
        )
        vectors.insert(
            second_scope,
            [VectorRecord(id="shared", vector=[0.0, 1.0], metadata={"owner": "second"})],
        )
        assert vectors.get(first_scope, ["shared"])[0].vector == [1.0, 0.0]
        assert vectors.get(second_scope, ["shared"])[0].vector == [0.0, 1.0]
        vectors.update(
            first_scope,
            [VectorRecord(id="shared", vector=[0.25, 0.75], metadata={"owner": "updated"})],
        )
        assert vectors.get(first_scope, ["shared"])[0].metadata == {"owner": "updated"}
        assert vectors.get(second_scope, ["shared"])[0].metadata == {"owner": "second"}
        vectors.delete(first_scope, ["shared"])
        assert vectors.get(first_scope, ["shared"]) == []
        assert vectors.get(second_scope, ["shared"])[0].metadata == {"owner": "second"}

        fulltext.insert(
            first_scope,
            [Document(id="shared", text="first scope document")],
        )
        fulltext.insert(
            second_scope,
            [Document(id="shared", text="second scope document")],
        )
        assert fulltext.get(first_scope, ["shared"])[0].text == "first scope document"
        assert fulltext.get(second_scope, ["shared"])[0].text == "second scope document"
        fulltext.update(
            first_scope,
            [Document(id="shared", text="first scope updated")],
        )
        assert fulltext.get(first_scope, ["shared"])[0].text == "first scope updated"
        assert fulltext.get(second_scope, ["shared"])[0].text == "second scope document"
        fulltext.delete(first_scope, ["shared"])
        assert fulltext.get(first_scope, ["shared"]) == []
        assert fulltext.get(second_scope, ["shared"])[0].text == "second scope document"
    finally:
        vectors.close()
        fulltext.close()


@pytest.mark.parametrize("operation", ["search", "recall"])
def test_vector_search_recall_wrap_first_connection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store = MilvusLiteVectorStore(uri=tmp_path / "vector-connect-failure.db", dim=2)
    scope = Scope(org="local", space="project:alpha")
    query = VectorQuery(vector=[1.0, 0.0], top_k=1)

    class BrokenConnection:
        @property
        def client(self) -> object:
            raise MilvusException(message="backend down")

        def close(self) -> None:
            pass

    try:
        monkeypatch.setattr(store, "_connection", BrokenConnection())
        with pytest.raises(BackendError, match="milvus vector connect"):
            if operation == "search":
                store.search(scope, query)
            else:
                store.recall(scope, query, output_fields=["metadata"])
    finally:
        store.close()


def test_vector_and_fulltext_share_local_service_until_last_close(tmp_path: Path) -> None:
    data_dir = tmp_path / "shared.db"
    scope = Scope(org="local", space="project:alpha")
    vectors = MilvusLiteVectorStore(uri=data_dir, dim=2)
    fulltext = MilvusLiteFulltextStore(uri=data_dir)
    try:
        vectors.insert(scope, [VectorRecord(id="v1", vector=[1.0, 0.0])])
        fulltext.insert(scope, [Document(id="d1", text="共享本地服务")])
        vectors.close()
        assert [hit.id for hit in fulltext.search(scope, TextQuery(text="共享"))] == ["d1"]
    finally:
        vectors.close()
        fulltext.close()

    assert _lock_is_available(data_dir)


def test_sqlite_audit_persists_across_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.sqlite3"
    scope = Scope(org="local", space="project:alpha")
    logger = SQLiteAuditLogger(db_path)
    try:
        logger.record(
            AuditEvent(
                id="event-1",
                actor=scope,
                action="write",
                target_id="unit-1",
                layer="api",
                decision="allow",
                occurred_at=datetime(2026, 10, 24, tzinfo=UTC),
                detail={"reason": "test"},
            )
        )
    finally:
        logger.close()
        logger.close()

    reopened = SQLiteAuditLogger(db_path)
    try:
        events = reopened.query({"action": "write", "target_id": "unit-1"})
        assert [event.id for event in events] == ["event-1"]
        assert events[0].detail == {"reason": "test"}
    finally:
        reopened.close()
