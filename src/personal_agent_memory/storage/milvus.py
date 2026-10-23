"""Milvus Lite storage adapters for the local JiuwenMemory kernel."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

from jiuwen_memory.common.errors import (
    BackendError,
    ConflictError,
    HealthCheckError,
    NotFoundError,
    ValidationError,
)
from jiuwen_memory.common.type_def import (
    FilterClause,
    FilterExpr,
    FilterGroup,
    FilterLogic,
    FilterOp,
    Scope,
    filter_field_metadata_key,
)
from jiuwen_memory.storage._support import scope_dims, wrap_backend
from jiuwen_memory.storage.base import StoreType
from jiuwen_memory.storage.fulltext import FulltextStore
from jiuwen_memory.storage.types import Document, ScoredID, TextQuery, VectorRecord
from jiuwen_memory.storage.vector_impl.milvus_vector import (
    MilvusVectorStore as _JiuwenMilvusVectorStore,
)
from milvus_lite.server_manager import server_manager_instance
from pymilvus import DataType, Function, FunctionType, MilvusClient

_SCOPE_FIELDS = (
    "scope_org",
    "scope_space",
    "scope_user",
    "scope_agent",
    "scope_session",
)
_CMP_OPS = {
    FilterOp.EQ: "==",
    FilterOp.NE: "!=",
    FilterOp.GT: ">",
    FilterOp.GTE: ">=",
    FilterOp.LT: "<",
    FilterOp.LTE: "<=",
}


def _normalized_uri(uri: str | Path) -> tuple[str, bool]:
    """Return the client URI and whether it names a local Milvus Lite directory."""
    value = str(uri)
    if value.startswith(("http://", "https://", "tcp://", "unix:")):
        return value, False
    return str(Path(value).expanduser().resolve()), True


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return str(value)
    raise ValidationError(f"unsupported filter literal: {value!r}")


def _compile_metadata_filter(expr: FilterExpr | None) -> str:
    """Compile JiuwenMemory filters, rewriting JSON IN/NOT IN for Milvus Lite."""
    if expr is None:
        return ""
    if isinstance(expr, FilterClause):
        field = f"metadata[{_literal(filter_field_metadata_key(expr.field))}]"
        if expr.op is FilterOp.NE:
            comparison = f"{field} != {_literal(expr.value)}"
            return f"(not exists({field}) || {comparison})"
        if expr.op in _CMP_OPS:
            return f"{field} {_CMP_OPS[expr.op]} {_literal(expr.value)}"
        if expr.op is FilterOp.IN:
            return "(" + " || ".join(
                f"{field} == {_literal(item)}" for item in expr.value
            ) + ")"
        if expr.op is FilterOp.NOT_IN:
            comparisons = " && ".join(
                f"{field} != {_literal(item)}" for item in expr.value
            )
            return f"(not exists({field}) || ({comparisons}))"
        if expr.op is FilterOp.CONTAINS:
            return f"json_contains({field}, {_literal(expr.value)})"
        raise ValidationError(f"unsupported filter op for Milvus: {expr.op}")
    if not isinstance(expr, FilterGroup):
        raise ValidationError(f"unsupported filter expression: {type(expr).__name__}")
    children = [_compile_metadata_filter(child) for child in expr.children]
    if expr.logic is FilterLogic.AND:
        return f"({' && '.join(children)})"
    if expr.logic is FilterLogic.OR:
        return f"({' || '.join(children)})"
    if expr.logic is FilterLogic.NOT:
        return f"(not ({children[0]}))"
    raise ValidationError(f"unsupported filter logic for Milvus: {expr.logic}")


def _scope_expression(scope: Scope) -> str:
    return " && ".join(
        f"scope_{dimension} == {_literal(value)}" for dimension, value in scope_dims(scope)
    )


def _combined_filter(scope: Scope, expr: FilterExpr | None) -> str:
    parts = [part for part in (_scope_expression(scope), _compile_metadata_filter(expr)) if part]
    return " && ".join(parts)


def _physical_id(scope: Scope, logical_id: str) -> str:
    scope_key = json.dumps(
        [scope.org, scope.space, scope.user, scope.agent, scope.session],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(scope_key.encode("utf-8")).hexdigest()
    return f"{digest}:{logical_id}"


def _logical_id(physical_id: str) -> str:
    return physical_id.split(":", 1)[1]


def _hit_id(hit: dict[str, Any]) -> str:
    entity = hit.get("entity") or {}
    logical_id = entity.get("logical_id") if isinstance(entity, dict) else None
    if logical_id:
        return str(logical_id)
    return _logical_id(str(hit["id"]))


@dataclass
class _SharedClient:
    uri: str
    options: dict[str, Any] = field(default_factory=dict)
    client: Any | None = None
    leases: int = 0


class _MilvusLiteConnection:
    """Reference-counted Milvus Lite client shared by vector and fulltext stores."""

    _registry: ClassVar[dict[str, _SharedClient]] = {}
    _registry_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, uri: str | Path, **options: Any) -> None:
        normalized, is_local = _normalized_uri(uri)
        self._uri = normalized
        self._is_local = is_local
        options.pop("db_name", None)
        with self._registry_lock:
            shared = self._registry.get(normalized)
            if shared is None:
                shared = _SharedClient(uri=normalized, options=options)
                self._registry[normalized] = shared
            shared.leases += 1
            self._shared = shared
        self._closed = False

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def client(self) -> Any:
        with self._registry_lock:
            if self._closed:
                raise BackendError("Milvus Lite connection is closed")
            if self._shared.client is None:
                self._shared.client = MilvusClient(
                    uri=self._uri,
                    db_name="default",
                    **self._shared.options,
                )
            return self._shared.client

    def close(self) -> None:
        with self._registry_lock:
            if self._closed:
                return
            self._closed = True
            self._shared.leases -= 1
            if self._shared.leases > 0:
                return
            self._registry.pop(self._uri, None)
            client = self._shared.client
            self._shared.client = None
            if client is not None:
                with suppress(Exception):
                    client.close()
            if self._is_local:
                server_manager_instance.release_server(self._uri)


class MilvusLiteVectorStore(_JiuwenMilvusVectorStore):  # type: ignore[misc]
    """JiuwenMemory dense-vector contract backed by a local Milvus Lite directory."""

    def __init__(
        self,
        *,
        uri: str | Path,
        dim: int,
        collection: str = "agent_memory_vectors",
        metric_type: str = "COSINE",
        consistency_level: str = "Strong",
        scope_field_max_length: int = 256,
        id_max_length: int = 512,
        load_timeout: float = 30.0,
    ) -> None:
        normalized, _ = _normalized_uri(uri)
        super().__init__(
            uri=normalized,
            collection=collection,
            dim=dim,
            metric_type=metric_type,
            consistency_level=consistency_level,
            scope_field_max_length=scope_field_max_length,
            id_max_length=id_max_length,
            load_timeout=load_timeout,
            db_name="default",
        )
        self._connection = _MilvusLiteConnection(normalized)
        self._state_lock = threading.RLock()
        self._collection_ready = False

    @property
    def client(self) -> Any:
        with self._state_lock:
            if not self._collection_ready:
                with wrap_backend("milvus vector connect"):
                    self._client = self._connection.client
                    self._client_uri = self._connection.uri
                    self._ensure_collection()
                    self._collection_ready = True
            return cast(Any, self._client)

    def _recover_connection(self, op: str) -> None:
        with self._state_lock:
            self._collection_ready = False
            self._client = None
        _ = self.client

    @staticmethod
    def _physical_id(scope: Scope, logical_id: str) -> str:
        return _physical_id(scope, logical_id)

    @staticmethod
    def _filter_clause(clause: FilterClause) -> str:
        return _compile_metadata_filter(clause)

    def insert(self, scope: Scope, records: list[VectorRecord]) -> None:
        with wrap_backend("milvus vector insert"):
            super().insert(scope, records)

    def update(self, scope: Scope, records: list[VectorRecord]) -> None:
        with wrap_backend("milvus vector update"):
            super().update(scope, records)

    def close(self) -> None:
        with self._state_lock:
            self._collection_ready = False
            self._client = None
            self._connection.close()

    def __enter__(self) -> MilvusLiteVectorStore:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


class MilvusLiteFulltextStore(FulltextStore):  # type: ignore[misc]
    """BM25 FulltextStore implemented with Milvus Lite schema functions and Jieba."""

    def __init__(
        self,
        *,
        uri: str | Path,
        collection: str = "agent_memory_fulltext",
        consistency_level: str = "Strong",
        scope_field_max_length: int = 256,
        id_max_length: int = 512,
    ) -> None:
        normalized, _ = _normalized_uri(uri)
        self._connection = _MilvusLiteConnection(normalized)
        self._collection = collection
        self._consistency = consistency_level
        self._scope_len = scope_field_max_length
        self._id_len = id_max_length
        self._physical_id_len = id_max_length + 5 * (scope_field_max_length + 1)
        self._lock = threading.RLock()
        self._collection_ready = False

    def _get_client(self) -> Any:
        with self._lock:
            client = self._connection.client
            if not self._collection_ready:
                self._ensure_collection(client)
                self._collection_ready = True
            return client

    def _ensure_collection(self, client: Any) -> None:
        if client.has_collection(self._collection):
            client.load_collection(self._collection)
            return
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            "id",
            DataType.VARCHAR,
            is_primary=True,
            max_length=self._physical_id_len,
        )
        schema.add_field("logical_id", DataType.VARCHAR, max_length=self._id_len)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={"tokenizer": {"type": "jieba", "mode": "search"}},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        for field_name in _SCOPE_FIELDS:
            schema.add_field(field_name, DataType.VARCHAR, max_length=self._scope_len)
        schema.add_field("metadata", DataType.JSON)
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
            consistency_level=self._consistency,
        )
        client.load_collection(self._collection)

    def store_type(self) -> StoreType:
        return StoreType.FULLTEXT

    def health(self) -> None:
        try:
            self._get_client().list_collections()
        except Exception as exc:
            raise HealthCheckError(f"Milvus Lite fulltext health failed: {exc}") from exc

    @classmethod
    def _row(cls, scope: Scope, document: Document) -> dict[str, Any]:
        return {
            "id": _physical_id(scope, document.id),
            "logical_id": document.id,
            "text": document.text,
            "scope_org": scope.org,
            "scope_space": scope.space,
            "scope_user": scope.user,
            "scope_agent": scope.agent,
            "scope_session": scope.session,
            "metadata": document.metadata,
        }

    @staticmethod
    def _document(row: dict[str, Any]) -> Document:
        return Document(
            id=str(row.get("logical_id") or _logical_id(str(row["id"]))),
            text=str(row.get("text") or ""),
            metadata=row.get("metadata") or {},
        )

    def _existing_ids(self, scope: Scope, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        physical_ids = ", ".join(_literal(_physical_id(scope, item)) for item in ids)
        filter_expression = f"id in [{physical_ids}]"
        scope_filter = _scope_expression(scope)
        if scope_filter:
            filter_expression = f"{filter_expression} && {scope_filter}"
        rows = self._get_client().query(
            self._collection,
            filter=filter_expression,
            output_fields=["logical_id"],
            consistency_level=self._consistency,
        )
        return {str(row["logical_id"]) for row in rows}

    def insert(self, scope: Scope, docs: list[Document]) -> None:
        if not docs:
            return
        with wrap_backend("milvus fulltext insert"):
            existing = self._existing_ids(scope, [doc.id for doc in docs])
            if existing:
                raise ConflictError(entity="document", key=next(iter(existing)))
            self._get_client().insert(
                self._collection,
                data=[self._row(scope, doc) for doc in docs],
            )

    def update(self, scope: Scope, docs: list[Document]) -> None:
        if not docs:
            return
        with wrap_backend("milvus fulltext update"):
            ids = [doc.id for doc in docs]
            missing = set(ids) - self._existing_ids(scope, ids)
            if missing:
                raise NotFoundError(entity="document", key=next(iter(missing)))
            self._get_client().upsert(
                self._collection,
                data=[self._row(scope, doc) for doc in docs],
            )

    def delete(self, scope: Scope, ids: list[str]) -> None:
        if not ids:
            return
        values = ", ".join(_literal(_physical_id(scope, item)) for item in ids)
        filter_expression = f"id in [{values}]"
        scope_filter = _scope_expression(scope)
        if scope_filter:
            filter_expression = f"{filter_expression} && {scope_filter}"
        try:
            self._get_client().delete(self._collection, filter=filter_expression)
        except Exception as exc:
            raise BackendError(f"milvus fulltext delete: {exc}") from exc

    def get(self, scope: Scope, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        values = ", ".join(_literal(_physical_id(scope, item)) for item in ids)
        filter_expression = f"id in [{values}]"
        scope_filter = _scope_expression(scope)
        if scope_filter:
            filter_expression = f"{filter_expression} && {scope_filter}"
        try:
            rows = self._get_client().query(
                self._collection,
                filter=filter_expression,
                output_fields=["id", "logical_id", "text", "metadata"],
                consistency_level=self._consistency,
            )
        except Exception as exc:
            raise BackendError(f"milvus fulltext get: {exc}") from exc
        return [self._document(row) for row in rows]

    def search(self, scope: Scope, query: TextQuery) -> list[ScoredID]:
        if not query.text.strip():
            return []
        filter_expression = _combined_filter(scope, query.filters)
        try:
            results = self._get_client().search(
                self._collection,
                data=[query.text],
                anns_field="sparse",
                limit=query.top_k,
                filter=filter_expression,
                output_fields=["logical_id"],
                search_params={"metric_type": "BM25"},
                consistency_level=self._consistency,
            )
        except Exception as exc:
            raise BackendError(f"milvus fulltext search: {exc}") from exc
        hits = results[0] if results else []
        return [
            ScoredID(id=_hit_id(hit), score=-float(hit["distance"]))
            for hit in hits
        ]

    def close(self) -> None:
        with self._lock:
            self._collection_ready = False
            self._connection.close()

    def __enter__(self) -> MilvusLiteFulltextStore:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
