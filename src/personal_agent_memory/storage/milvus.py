"""Milvus Lite storage adapters for the local JiuwenMemory kernel."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import suppress
from dataclasses import dataclass, field, replace
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
from jiuwen_memory.storage._support import wrap_backend
from jiuwen_memory.storage.base import StoreType
from jiuwen_memory.storage.fulltext import FulltextStore
from jiuwen_memory.storage.graph import GraphStore
from jiuwen_memory.storage.types import (
    Document,
    Edge,
    GraphQuery,
    Node,
    ScoredID,
    TextQuery,
    VectorRecord,
)
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
            return "(" + " || ".join(f"{field} == {_literal(item)}" for item in expr.value) + ")"
        if expr.op is FilterOp.NOT_IN:
            comparisons = " && ".join(f"{field} != {_literal(item)}" for item in expr.value)
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
        f"scope_{field_name.removeprefix('scope_')} == "
        f"{_literal(str(getattr(scope, field_name.removeprefix('scope_'))))}"
        for field_name in _SCOPE_FIELDS
    )


def _scope_key(scope: Scope) -> tuple[str, ...]:
    return tuple(
        str(getattr(scope, field_name.removeprefix("scope_"))) for field_name in _SCOPE_FIELDS
    )


def _graph_scope_value(value: str) -> str:
    return "e:" if value == "" else f"v:{value}"


def _decode_graph_scope_value(value: str) -> str:
    if value == "e:":
        return ""
    if value.startswith("v:"):
        return value[2:]
    return value


def _graph_scope_expression(scope: Scope) -> str:
    return " && ".join(
        f"scope_{field_name.removeprefix('scope_')} == "
        f"{_literal(_graph_scope_value(str(getattr(scope, field_name.removeprefix('scope_')))))}"
        for field_name in _SCOPE_FIELDS
    )


def _combined_filter(scope: Scope, expr: FilterExpr | None) -> str:
    parts = [part for part in (_scope_expression(scope), _compile_metadata_filter(expr)) if part]
    return " && ".join(parts)


def _recall_index_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    memory_type = str(normalized.get("user_metadata.memory_type") or "")
    if memory_type in {"capture", "evolved"} or bool(
        normalized.get("user_metadata.capture_event_id")
    ):
        normalized["user_metadata.default_recall_excluded"] = True
    return normalized


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

    def _scope_expr(self, scope: Scope) -> str:
        """Match every scope dimension exactly, including empty session values."""
        return _scope_expression(scope)

    def insert(self, scope: Scope, records: list[VectorRecord]) -> None:
        records = [
            replace(record, metadata=_recall_index_metadata(record.metadata))
            for record in records
        ]
        with wrap_backend("milvus vector insert"):
            super().insert(scope, records)

    def update(self, scope: Scope, records: list[VectorRecord]) -> None:
        records = [
            replace(record, metadata=_recall_index_metadata(record.metadata))
            for record in records
        ]
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
            "metadata": _recall_index_metadata(document.metadata),
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
        return [ScoredID(id=_hit_id(hit), score=-float(hit["distance"])) for hit in hits]

    def close(self) -> None:
        with self._lock:
            self._collection_ready = False
            self._connection.close()

    def __enter__(self) -> MilvusLiteFulltextStore:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


class MilvusLiteGraphStore(GraphStore):  # type: ignore[misc]
    """Persistent property graph backed by two Milvus Lite collections.

    Nodes and edges use the same scope fields as the vector/fulltext adapters.
    The adapter performs BFS in Python because Milvus Lite does not provide a
    graph traversal API. Collections are shared by the local runtime and are
    therefore safe to close and reopen.
    """

    _READ_LIMIT = 16_384

    def __init__(
        self,
        *,
        uri: str | Path,
        collection: str = "agent_memory_graph_nodes",
        edge_collection: str = "agent_memory_graph_edges",
        consistency_level: str = "Strong",
        scope_field_max_length: int = 256,
        id_max_length: int = 512,
    ) -> None:
        normalized, _ = _normalized_uri(uri)
        self._connection = _MilvusLiteConnection(normalized)
        self._collection = collection
        self._edge_collection = edge_collection
        self._consistency = consistency_level
        self._scope_len = scope_field_max_length + 2
        self._id_len = id_max_length
        self._physical_id_len = id_max_length + 5 * (scope_field_max_length + 1)
        self._lock = threading.RLock()
        self._nodes_ready = False
        self._edges_ready = False
        self._closed = False
        self._diagnostics: dict[tuple[str, ...], str] = {}

    def _get_client(self) -> Any:
        with self._lock:
            if self._closed:
                raise BackendError("Milvus Lite graph store is closed")
            client = self._connection.client
            if not self._nodes_ready:
                self._ensure_collection(client, self._collection, node=True)
                self._nodes_ready = True
            if not self._edges_ready:
                self._ensure_collection(client, self._edge_collection, node=False)
                self._edges_ready = True
            return client

    def _ensure_collection(self, client: Any, collection: str, *, node: bool) -> None:
        if client.has_collection(collection):
            client.load_collection(collection)
            return
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            "id",
            DataType.VARCHAR,
            is_primary=True,
            max_length=self._physical_id_len,
        )
        schema.add_field("logical_id", DataType.VARCHAR, max_length=self._id_len)
        for field_name in _SCOPE_FIELDS:
            schema.add_field(field_name, DataType.VARCHAR, max_length=self._scope_len)
        if node:
            schema.add_field("label", DataType.VARCHAR, max_length=256)
            schema.add_field("properties", DataType.JSON)
        else:
            schema.add_field("source", DataType.VARCHAR, max_length=self._id_len)
            schema.add_field("target", DataType.VARCHAR, max_length=self._id_len)
            schema.add_field("relation", DataType.VARCHAR, max_length=256)
            schema.add_field("properties", DataType.JSON)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=2)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE",
        )
        client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index_params,
            consistency_level=self._consistency,
        )
        client.load_collection(collection)

    def store_type(self) -> StoreType:
        return StoreType.GRAPH

    def health(self) -> None:
        try:
            client = self._get_client()
            client.get_collection_stats(self._collection)
            client.get_collection_stats(self._edge_collection)
        except Exception as exc:
            raise BackendError(f"milvus graph health: {exc}") from exc

    @staticmethod
    def _node_row(scope: Scope, node: Node) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": _physical_id(scope, node.id),
            "logical_id": node.id,
            "label": node.label[:256],
            "properties": node.properties,
            "embedding": [0.0, 0.0],
        }
        for field_name in _SCOPE_FIELDS:
            row[field_name] = _graph_scope_value(
                str(getattr(scope, field_name.removeprefix("scope_")))
            )
        return row

    @staticmethod
    def _edge_row(scope: Scope, edge: Edge) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": _physical_id(scope, edge.id),
            "logical_id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "relation": edge.relation[:256],
            "properties": edge.properties,
            "embedding": [0.0, 0.0],
        }
        for field_name in _SCOPE_FIELDS:
            row[field_name] = _graph_scope_value(
                str(getattr(scope, field_name.removeprefix("scope_")))
            )
        return row

    @staticmethod
    def _node_from_row(row: dict[str, Any]) -> Node:
        return Node(
            id=str(row.get("logical_id") or _logical_id(str(row["id"]))),
            label=str(row.get("label") or ""),
            properties=dict(row.get("properties") or {}),
        )

    @staticmethod
    def _edge_from_row(row: dict[str, Any]) -> Edge:
        return Edge(
            id=str(row.get("logical_id") or _logical_id(str(row["id"]))),
            source=str(row.get("source") or ""),
            target=str(row.get("target") or ""),
            relation=str(row.get("relation") or ""),
            properties=dict(row.get("properties") or {}),
        )

    def _existing_ids(self, scope: Scope, collection: str, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        values = ", ".join(_literal(_physical_id(scope, item)) for item in ids)
        filter_expression = f"id in [{values}] && {_graph_scope_expression(scope)}"
        rows = self._get_client().query(
            collection,
            filter=filter_expression,
            output_fields=["logical_id"],
            consistency_level=self._consistency,
        )
        return {str(row["logical_id"]) for row in rows}

    def clear_diagnostics(self, scope: Scope | None = None) -> None:
        with self._lock:
            if scope is None:
                self._diagnostics.clear()
            else:
                self._diagnostics.pop(_scope_key(scope), None)

    def last_error(self, scope: Scope | None = None) -> str | None:
        with self._lock:
            if scope is None:
                return next(iter(self._diagnostics.values()), None)
            return self._diagnostics.get(_scope_key(scope))

    def _set_diagnostic(self, scope: Scope, error: Exception) -> None:
        with self._lock:
            self._diagnostics[_scope_key(scope)] = f"{type(error).__name__}: {error}"

    def _query_rows(
        self,
        collection: str,
        filter_expression: str,
        fields: list[str],
        *,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        client = self._get_client()
        limit = max(1, min(page_size or self._READ_LIMIT, self._READ_LIMIT))
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = client.query(
                collection,
                filter=filter_expression,
                output_fields=fields,
                limit=limit,
                offset=offset,
                consistency_level=self._consistency,
            )
            page_rows = [dict(row) for row in page]
            rows.extend(page_rows)
            if len(page_rows) < limit:
                break
            offset += len(page_rows)
        return rows

    def _all_rows(
        self,
        scope: Scope,
        collection: str,
        fields: list[str],
        *,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._query_rows(
            collection,
            _graph_scope_expression(scope),
            fields,
            page_size=page_size,
        )

    def insert(
        self,
        scope: Scope,
        nodes: list[Node] | None = None,
        edges: list[Edge] | None = None,
    ) -> None:
        nodes = nodes or []
        edges = edges or []
        try:
            with wrap_backend("milvus graph insert"):
                existing_nodes = self._existing_ids(
                    scope, self._collection, [node.id for node in nodes]
                )
                if existing_nodes:
                    raise ConflictError(entity="node", key=next(iter(existing_nodes)))
                existing_edges = self._existing_ids(
                    scope, self._edge_collection, [edge.id for edge in edges]
                )
                if existing_edges:
                    raise ConflictError(entity="edge", key=next(iter(existing_edges)))
                client = self._get_client()
                if nodes:
                    client.insert(
                        self._collection,
                        data=[self._node_row(scope, node) for node in nodes],
                    )
                if edges:
                    signatures = {
                        (
                            edge.source,
                            edge.target,
                            edge.relation,
                            json.dumps(edge.properties, ensure_ascii=False, sort_keys=True),
                        )
                        for edge in (
                            self._edge_from_row(row)
                            for row in self._all_rows(
                                scope,
                                self._edge_collection,
                                [
                                    "id",
                                    "logical_id",
                                    "source",
                                    "target",
                                    "relation",
                                    "properties",
                                ],
                            )
                        )
                    }
                    fresh_edges: list[Edge] = []
                    for edge in edges:
                        signature = (
                            edge.source,
                            edge.target,
                            edge.relation,
                            json.dumps(
                                edge.properties,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        )
                        if signature in signatures:
                            continue
                        signatures.add(signature)
                        fresh_edges.append(edge)
                    if fresh_edges:
                        client.insert(
                            self._edge_collection,
                            data=[self._edge_row(scope, edge) for edge in fresh_edges],
                        )
        except (ConflictError, NotFoundError):
            raise
        except Exception as error:
            self._set_diagnostic(scope, error)
            raise
        else:
            self.clear_diagnostics(scope)

    def update(
        self,
        scope: Scope,
        nodes: list[Node] | None = None,
        edges: list[Edge] | None = None,
    ) -> None:
        nodes = nodes or []
        edges = edges or []
        try:
            with wrap_backend("milvus graph update"):
                missing_nodes = {node.id for node in nodes} - self._existing_ids(
                    scope, self._collection, [node.id for node in nodes]
                )
                if missing_nodes:
                    raise NotFoundError(entity="node", key=next(iter(missing_nodes)))
                missing_edges = {edge.id for edge in edges} - self._existing_ids(
                    scope, self._edge_collection, [edge.id for edge in edges]
                )
                if missing_edges:
                    raise NotFoundError(entity="edge", key=next(iter(missing_edges)))
                client = self._get_client()
                if nodes:
                    client.upsert(
                        self._collection,
                        data=[self._node_row(scope, node) for node in nodes],
                    )
                if edges:
                    client.upsert(
                        self._edge_collection,
                        data=[self._edge_row(scope, edge) for edge in edges],
                    )
        except (ConflictError, NotFoundError):
            raise
        except Exception as error:
            self._set_diagnostic(scope, error)
            raise
        else:
            self.clear_diagnostics(scope)

    def clear(self, scope: Scope) -> None:
        """Remove every node and edge in one scope without touching other scopes."""
        try:
            with wrap_backend("milvus graph clear"):
                client = self._get_client()
                client.delete(self._edge_collection, filter=_graph_scope_expression(scope))
                client.delete(self._collection, filter=_graph_scope_expression(scope))
        except Exception as error:
            self._set_diagnostic(scope, error)
            raise
        else:
            self.clear_diagnostics(scope)

    def delete(
        self,
        scope: Scope,
        node_ids: list[str] | None = None,
        edge_ids: list[str] | None = None,
    ) -> None:
        node_ids = node_ids or []
        edge_ids = list(edge_ids or [])
        try:
            with wrap_backend("milvus graph delete"):
                if node_ids:
                    node_set = set(node_ids)
                    for row in self._all_rows(
                        scope,
                        self._edge_collection,
                        ["logical_id", "source", "target"],
                    ):
                        if str(row.get("source")) in node_set or str(row.get("target")) in node_set:
                            edge_ids.append(str(row["logical_id"]))
                client = self._get_client()
                if edge_ids:
                    values = ", ".join(
                        _literal(_physical_id(scope, item)) for item in dict.fromkeys(edge_ids)
                    )
                    client.delete(
                        self._edge_collection,
                        filter=f"id in [{values}] && {_graph_scope_expression(scope)}",
                    )
                if node_ids:
                    values = ", ".join(
                        _literal(_physical_id(scope, item)) for item in dict.fromkeys(node_ids)
                    )
                    client.delete(
                        self._collection,
                        filter=f"id in [{values}] && {_graph_scope_expression(scope)}",
                    )
        except (ConflictError, NotFoundError):
            raise
        except Exception as error:
            self._set_diagnostic(scope, error)
            raise
        else:
            self.clear_diagnostics(scope)

    def get(self, scope: Scope, node_ids: list[str]) -> list[Node]:
        if not node_ids:
            return []
        values = ", ".join(_literal(_physical_id(scope, item)) for item in node_ids)
        rows = self._get_client().query(
            self._collection,
            filter=f"id in [{values}] && {_graph_scope_expression(scope)}",
            output_fields=["id", "logical_id", "label", "properties"],
            consistency_level=self._consistency,
        )
        return [self._node_from_row(dict(row)) for row in rows]

    def search(self, scope: Scope, query: GraphQuery) -> list[Node]:
        return [
            node
            for node, _ in self.traverse(
                scope,
                query.start_id,
                depth=query.depth,
                relation=query.relation,
                limit=query.limit,
            )
        ]

    def traverse(
        self,
        scope: Scope,
        start_id: str,
        *,
        depth: int,
        relation: str | None = None,
        limit: int = 100,
    ) -> list[tuple[Node, int]]:
        return [
            (node, hop)
            for node, hop, _ in self.traverse_many(
                scope,
                [start_id],
                depth=depth,
                relation=relation,
                limit=limit,
            )
        ]

    def traverse_many(
        self,
        scope: Scope,
        start_ids: list[str],
        *,
        depth: int,
        relation: str | None = None,
        limit: int = 100,
    ) -> list[tuple[Node, int, str]]:
        """Expand several seeds with one paginated edge scan and bounded BFS."""
        starts = list(dict.fromkeys(start_ids))
        max_results = max(limit, 0)
        max_depth = max(depth, 0)
        if not starts or max_results == 0 or max_depth == 0:
            return []
        edges = [
            self._edge_from_row(row)
            for row in self._all_rows(
                scope,
                self._edge_collection,
                ["id", "logical_id", "source", "target", "relation", "properties"],
            )
        ]
        adjacency: dict[str, list[str]] = {}
        for edge in edges:
            if relation is not None and edge.relation != relation:
                continue
            adjacency.setdefault(edge.source, []).append(edge.target)
            adjacency.setdefault(edge.target, []).append(edge.source)
        for neighbors in adjacency.values():
            neighbors.sort()
        visited = set(starts)
        frontier = [(start_id, start_id) for start_id in starts]
        reached: list[tuple[str, int, str]] = []
        for hop in range(1, max_depth + 1):
            next_frontier: list[tuple[str, str]] = []
            stop = False
            for node_id, origin_id in frontier:
                for neighbor in adjacency.get(node_id, []):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.append((neighbor, origin_id))
                    reached.append((neighbor, hop, origin_id))
                    if len(reached) >= max_results:
                        stop = True
                        break
                if stop:
                    break
            frontier = next_frontier
            if stop or not frontier:
                break
        nodes_by_id = {
            node.id: node for node in self.get(scope, [node_id for node_id, _, _ in reached])
        }
        return [
            (nodes_by_id[node_id], hop, origin_id)
            for node_id, hop, origin_id in reached
            if node_id in nodes_by_id
        ]

    def seed_ids(self, scope: Scope, tokens: set[str]) -> list[str]:
        if not tokens:
            return []
        rows = self._all_rows(
            scope,
            self._collection,
            ["logical_id", "label", "properties"],
        )
        result: list[str] = []
        for row in rows:
            node = self._node_from_row(row)
            content = str(node.properties.get("content", ""))
            if any(token and token in content for token in tokens):
                result.append(node.id)
        return result

    def node_ids(self, scope: Scope) -> list[str]:
        return [
            str(row["logical_id"])
            for row in self._all_rows(scope, self._collection, ["logical_id"])
        ]

    def projection_nodes(self, scope: Scope) -> list[Node]:
        return [
            self._node_from_row(row)
            for row in self._all_rows(
                scope,
                self._collection,
                ["id", "logical_id", "label", "properties"],
            )
        ]

    def projection_edges(self, scope: Scope) -> list[Edge]:
        return [
            self._edge_from_row(row)
            for row in self._all_rows(
                scope,
                self._edge_collection,
                ["id", "logical_id", "source", "target", "relation", "properties"],
            )
        ]

    def edge_endpoints(self, scope: Scope) -> list[tuple[str, str]]:
        return [
            (str(row["source"]), str(row["target"]))
            for row in self._all_rows(
                scope,
                self._edge_collection,
                ["source", "target"],
            )
        ]

    def graph_scopes(self) -> list[Scope]:
        fields = list(_SCOPE_FIELDS)
        rows = [
            *self._query_rows(self._collection, "", fields),
            *self._query_rows(self._edge_collection, "", fields),
        ]
        scopes: dict[tuple[str, ...], Scope] = {}
        for row in rows:
            values = tuple(str(row[field]) for field in _SCOPE_FIELDS)
            if values in scopes:
                continue
            decoded = tuple(_decode_graph_scope_value(value) for value in values)
            scopes[values] = Scope(
                org=decoded[0],
                space=decoded[1],
                user=decoded[2],
                agent=decoded[3],
                session=decoded[4],
            )
        return list(scopes.values())

    def counts(self, scope: Scope | None = None) -> dict[str, int]:
        filter_expression = "" if scope is None else _graph_scope_expression(scope)
        nodes = self._query_rows(self._collection, filter_expression, ["id"])
        edges = self._query_rows(self._edge_collection, filter_expression, ["id"])
        return {"nodes": len(nodes), "edges": len(edges)}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._nodes_ready = False
            self._edges_ready = False
            self._connection.close()

    def __enter__(self) -> MilvusLiteGraphStore:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
