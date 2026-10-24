"""JiuwenMemory 0.2.0 runtime adapter for the public memory kernel port."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from jiuwen_memory.api import (
    Context,
    DisclosureLevel,
    MemoryRuntime,
    MemoryUnit,
    Modality,
    RequestSecurityContext,
    Scope,
    assemble_runtime,
    legacy_request_context,
)
from jiuwen_memory.common.audit.base import AuditLogger, AuditProducer
from jiuwen_memory.common.embedder.base import Embedder, EmbedderProducer
from jiuwen_memory.common.embedder.embedder_impl.openai_embedder import OpenAIEmbedder
from jiuwen_memory.storage import FulltextStore, VectorStore
from jiuwen_memory.storage.fulltext import FulltextProducer
from jiuwen_memory.storage.kv import load_units
from jiuwen_memory.storage.types import ScoredID, TextQuery
from jiuwen_memory.storage.vector import VectorProducer

from personal_agent_memory.config import MemoryKernelKind
from personal_agent_memory.memory_kernel import (
    CWD_SCOPE_PREFIX,
    LIBRARY_SCOPE_PREFIX,
    AuditMemoryRequest,
    CaptureMemoryRequest,
    CreateMemoryRequest,
    KernelHealth,
    MemoryIdempotencyConflictError,
    MemoryIdempotencyPendingError,
    MemoryKernelUnavailableError,
    MemoryWriteRequest,
    ResolveRegionRequest,
    SearchMemoryRequest,
    UnavailableMemoryKernel,
    validate_library_id,
)
from personal_agent_memory.model_client import ModelEndpoint
from personal_agent_memory.storage import (
    MilvusLiteFulltextStore,
    MilvusLiteVectorStore,
    SQLiteAuditLogger,
    SQLiteKVStore,
)

_STORE_TARGET = "personal-agent-memory-local"
_AUDIT_TARGET = "personal-agent-memory-sqlite"
_EMBEDDER_TARGET = "personal-agent-memory-openai"
_LOCAL_ORG = "local"
_LOCAL_USER = "local"
_LOCAL_AGENT = "personal-agent-memory"
_DEFAULT_EMBEDDING_DIM = 1536
_VECTOR_MIN_SIMILARITY = 0.5
_STATE_VERSION = "jiuwen-memory-0.2.0"
_STORE_COLLECTOR: ContextVar[list[object] | None] = ContextVar(
    "personal_agent_memory_mem2_store_collector", default=None
)


def _remember_store(store: object) -> None:
    collector = _STORE_COLLECTOR.get()
    if collector is None:
        raise RuntimeError("mem2 storage assembly is not active")
    collector.append(store)


@VectorProducer.register(_STORE_TARGET)  # type: ignore[untyped-decorator]
def _build_vector_store(config: Any) -> VectorStore:
    store = MilvusLiteVectorStore(
        uri=str(config.get("uri")),
        dim=int(config.get("dim", _DEFAULT_EMBEDDING_DIM)),
        collection=str(config.get("collection")),
    )
    _remember_store(store)
    return store


@FulltextProducer.register(_STORE_TARGET)  # type: ignore[untyped-decorator]
def _build_fulltext_store(config: Any) -> FulltextStore:
    store = MilvusLiteFulltextStore(
        uri=str(config.get("uri")),
        collection=str(config.get("collection")),
    )
    _remember_store(store)
    return store


@AuditProducer.register(_AUDIT_TARGET)  # type: ignore[untyped-decorator]
def _build_audit_logger(config: Any) -> AuditLogger:
    logger = SQLiteAuditLogger(str(config.get("db_path")))
    _remember_store(logger)
    return logger


@EmbedderProducer.register(_EMBEDDER_TARGET)  # type: ignore[untyped-decorator]
def _build_embedder(config: Any) -> Embedder:
    embedder = OpenAIEmbedder(
        model_name=str(config.get("model_name")),
        base_url=str(config.get("base_url")),
        api_key=str(config.get("api_key", "")),
        dimension=int(config.get("dim", _DEFAULT_EMBEDDING_DIM)),
        timeout=float(config.get("timeout")),
        max_retries=int(config.get("max_retries", 0)),
    )
    _remember_store(embedder)
    return embedder


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (overrides or {}).items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class _RuntimeBundle:
    runtime: MemoryRuntime
    kv_store: SQLiteKVStore
    stores: tuple[object, ...]


class Mem2MemoryKernel(UnavailableMemoryKernel):
    """Persistent JiuwenMemory runtime exposed through ``MemoryKernelPort``."""

    kind: MemoryKernelKind = "mem2"

    def __init__(
        self,
        state_dir: Path,
        *,
        embedding: ModelEndpoint | None = None,
        embedding_dimension: int = _DEFAULT_EMBEDDING_DIM,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if embedding_dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self._state_dir = state_dir.expanduser().resolve()
        self._embedding = embedding
        self._embedding_dimension = embedding_dimension
        self._config_overrides = config_overrides
        self._bundle: _RuntimeBundle | None = None
        self._startup_error: str | None = None
        self._vector_enabled = False
        self._vector_degradation = "embedding_not_configured"
        self._embedding_detail = "embedding endpoint is not configured; vector recall is disabled"
        self._embedding_api_key_cache: str | None = None
        self._lock = threading.RLock()
        self._idempotency_locks: dict[tuple[str, str], tuple[threading.Lock, int]] = {}
        self._idempotency_locks_guard = threading.Lock()
        self._idempotency_schema_lock = threading.Lock()

    async def start(self) -> None:
        await asyncio.to_thread(self._start)

    def _start(self) -> None:
        with self._lock:
            if self._bundle is not None:
                return
            self._startup_error = None
            self._prepare_embedding()
            state_dir = self._state_dir / "mem2"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "VERSION").write_text(_STATE_VERSION + "\n", encoding="utf-8")
            stores: list[object] = []
            kv_store: SQLiteKVStore | None = None
            token = _STORE_COLLECTOR.set(stores)
            try:
                kv_store = SQLiteKVStore(state_dir / "memory.sqlite3")
                runtime = assemble_runtime(
                    kv=kv_store,
                    config=self._runtime_config(state_dir),
                )
            except Exception as error:
                _close_stores(stores)
                if kv_store is not None:
                    _close_quietly(kv_store)
                self._startup_error = f"mem2 runtime startup failed: {error}"
                return
            finally:
                _STORE_COLLECTOR.reset(token)
            self._bundle = _RuntimeBundle(runtime, kv_store, tuple(stores))

    async def close(self) -> None:
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        with self._lock:
            bundle = self._bundle
            self._bundle = None
            if bundle is None:
                return
            _close_quietly(bundle.runtime)
            _close_quietly(bundle.kv_store)
            _close_stores(list(bundle.stores))

    def health(self) -> KernelHealth:
        with self._lock:
            bundle = self._bundle
            startup_error = self._startup_error
            embedding_detail = self._embedding_detail
        if bundle is None:
            detail = startup_error or "mem2 runtime is not started"
            return KernelHealth(kind=self.kind, status="unavailable", detail=detail)
        failures: list[str] = []
        services: tuple[object, ...] = (bundle.kv_store, *bundle.stores)
        seen: set[int] = set()
        for service in services:
            if id(service) in seen:
                continue
            seen.add(id(service))
            health = getattr(service, "health", None)
            if not callable(health):
                continue
            try:
                health()
            except Exception as error:
                failures.append(f"{type(service).__name__}: {error}")
        if not self._vector_enabled:
            failures.insert(0, embedding_detail)
        if failures:
            return KernelHealth(
                kind=self.kind,
                status="degraded",
                detail="; ".join(failures)[:1000],
            )
        return KernelHealth(
            kind=self.kind,
            status="ready",
            detail="JiuwenMemory 0.2.0 runtime",
        )

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._write, request)

    def _write(self, request: MemoryWriteRequest) -> dict[str, object]:
        if isinstance(request, CreateMemoryRequest):
            normalized = replace(request, library_id=validate_library_id(request.library_id))
            key = (normalized.library_id, normalized.idempotency_key)
            with self._idempotency_lock(key):
                return self._write_locked(normalized)
        return self._write_locked(request)

    @contextmanager
    def _idempotency_lock(self, key: tuple[str, str]) -> Iterator[None]:
        """Serialize check-add-record for one library/key pair."""
        with self._idempotency_locks_guard:
            lock, users = self._idempotency_locks.get(key, (threading.Lock(), 0))
            self._idempotency_locks[key] = (lock, users + 1)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._idempotency_locks_guard:
                current_lock, users = self._idempotency_locks[key]
                if users == 1:
                    del self._idempotency_locks[key]
                else:
                    self._idempotency_locks[key] = (current_lock, users - 1)

    def _write_locked(self, request: MemoryWriteRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        if isinstance(request, CreateMemoryRequest):
            scope = _scope_for_normalized_library(request.library_id)
            request_hash = _write_request_hash(request)
            existing_id = self._reserve_idempotency(
                request.library_id, request.idempotency_key, request_hash
            )
            if existing_id is not None:
                return _write_response(request, existing_id, status="existing")
            user_metadata = {
                "memory_type": request.memory_type,
                "source_references": list(request.source_references),
                "request_actor": request.actor,
                "idempotency_key": request.idempotency_key,
            }
        elif isinstance(request, CaptureMemoryRequest):
            scope = _scope_for_cwd(request.cwd, request.session_id)
            user_metadata = {
                "event_id": request.event_id,
                "session_id": request.session_id,
                "turn_id": request.turn_id,
                "event_kind": request.event_kind,
                "occurred_at": request.occurred_at,
                "cwd": request.cwd,
            }
        else:
            raise TypeError("unsupported memory write request")
        security = _security_for(scope)
        try:
            units = bundle.runtime.api.add(
                request.content,
                scope,
                Modality.TEXT,
                security=security,
                user_metadata=user_metadata,
            )
        except Exception:
            if isinstance(request, CreateMemoryRequest):
                self._release_pending_idempotency(request.library_id, request.idempotency_key)
            raise
        if not units:
            if isinstance(request, CreateMemoryRequest):
                self._release_pending_idempotency(request.library_id, request.idempotency_key)
            raise MemoryKernelUnavailableError("mem2 runtime did not create a memory unit")
        unit = units[0]
        if isinstance(request, CreateMemoryRequest):
            try:
                self._complete_idempotency(
                    request.library_id,
                    request.idempotency_key,
                    request_hash,
                    unit.id,
                )
            except Exception as error:
                raise MemoryKernelUnavailableError(
                    "memory was created but its idempotency completion is pending"
                ) from error
        return _write_response(request, unit.id, status="created")

    async def search(self, request: SearchMemoryRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._search, request)

    def _search(self, request: SearchMemoryRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        if request.library_id is not None:
            library_id = validate_library_id(request.library_id)
            request = replace(request, library_id=library_id)
            scope = _scope_for_normalized_library(library_id)
        else:
            scope = _scope_for_cwd(request.cwd or "", None)
        security = _security_for(scope)
        degradation: list[str] = []
        embedding_configured = self._vector_enabled
        if not embedding_configured:
            degradation.append(self._vector_degradation)
            items, failed = self._keyword_fallback(bundle, scope, request)
            if failed:
                degradation.append("fulltext_unavailable")
            return _search_response(
                request,
                scope,
                items,
                [],
                degradation,
                vector_status="not_configured",
            )
        try:
            result = bundle.runtime.api.search(
                request.query,
                Context(scope=scope),
                security=security,
                top_k=max(1, min(request.limit, 100)),
                disclosure=DisclosureLevel.ADAPTIVE,
            )
        except Exception as retrieval_error:
            classification = _classify_failure(
                retrieval_error, embedding_configured=embedding_configured
            )
            if classification not in degradation:
                degradation.append(classification)
            units, failed = self._keyword_fallback(bundle, scope, request)
            if failed and "fulltext_unavailable" not in degradation:
                degradation.append("fulltext_unavailable")
            return _search_response(
                request,
                scope,
                units,
                [],
                degradation,
                vector_status="degraded",
            )

        reported_errors = list(result.errors)
        items = result.items
        if not items and result.errors:
            recovered, failed = self._keyword_fallback(bundle, scope, request)
            if recovered:
                items = recovered
                reported_errors = []
            elif failed and "fulltext_unavailable" not in degradation:
                degradation.append("fulltext_unavailable")
        for channel_error in reported_errors:
            channel = getattr(channel_error.channel, "value", str(channel_error.channel))
            failure = _channel_failure(channel, channel_error)
            if failure not in degradation:
                degradation.append(failure)
        vector_failed = any(
            getattr(channel_error.channel, "value", str(channel_error.channel)) == "vector"
            for channel_error in result.errors
        )
        vector_status = (
            "not_configured"
            if not embedding_configured
            else "degraded"
            if vector_failed
            else "ready"
        )
        return _search_response(
            request,
            scope,
            items,
            reported_errors,
            degradation,
            vector_status=vector_status,
        )

    def _keyword_fallback(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        request: SearchMemoryRequest,
    ) -> tuple[list[Any], bool]:
        fulltext = next(
            (
                store
                for store in bundle.stores
                if isinstance(store, MilvusLiteFulltextStore)
                and getattr(store, "_collection", "") == "agent_memory_fulltext"
            ),
            None,
        )
        if fulltext is None:
            return [], True
        try:
            hits: list[ScoredID] = fulltext.search(
                scope,
                TextQuery(text=request.query, top_k=max(1, min(request.limit, 100))),
            )
            if not hits:
                return [], False
            return list(load_units(bundle.kv_store, scope, [hit.id for hit in hits])), False
        except Exception:
            return [], True

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]:
        bundle = self._require_bundle()
        library_id = validate_library_id(request.library_id)
        scope = _scope_for_normalized_library(library_id)
        events = bundle.runtime.api.audit(
            {"target_space": scope.space},
            security=_security_for(scope),
            limit=max(1, min(request.limit, 1000)),
        )
        return [
            {
                "id": event.id,
                "action": event.action,
                "decision": event.decision,
                "actor": event.actor.user,
                "target": event.target.space,
                "occurred_at": (
                    event.occurred_at.isoformat() if event.occurred_at is not None else ""
                ),
            }
            for event in events
        ]

    def _idempotency_db(self) -> sqlite3.Connection:
        path = self._state_dir / "mem2" / "idempotency.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        with self._idempotency_schema_lock:
            self._ensure_idempotency_schema(connection)
        return connection

    @staticmethod
    def _create_idempotency_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS memory_write_idempotency (
                   library_id TEXT NOT NULL,
                   idempotency_key TEXT NOT NULL,
                   request_hash TEXT NOT NULL,
                   memory_id TEXT,
                   status TEXT NOT NULL
                       CHECK (status IN ('pending', 'completed')),
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                   completed_at TEXT,
                   PRIMARY KEY (library_id, idempotency_key)
               )"""
        )

    def _ensure_idempotency_schema(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(memory_write_idempotency)").fetchall()
        }
        if not columns:
            with connection:
                self._create_idempotency_table(connection)
            return
        memory_id_column = columns.get("memory_id")
        if "status" in columns and memory_id_column is not None and not memory_id_column[3]:
            return
        with connection:
            connection.execute(
                "ALTER TABLE memory_write_idempotency RENAME TO memory_write_idempotency_v1"
            )
            self._create_idempotency_table(connection)
            connection.execute(
                """INSERT OR REPLACE INTO memory_write_idempotency
                   (library_id, idempotency_key, request_hash, memory_id,
                    status, created_at, completed_at)
                   SELECT library_id, idempotency_key, request_hash, memory_id,
                          'completed', created_at, created_at
                   FROM memory_write_idempotency_v1"""
            )
            connection.execute("DROP TABLE memory_write_idempotency_v1")

    def _reserve_idempotency(
        self, library_id: str, idempotency_key: str, request_hash: str
    ) -> str | None:
        with closing(self._idempotency_db()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO memory_write_idempotency
                   (library_id, idempotency_key, request_hash, memory_id, status)
                   VALUES (?, ?, ?, NULL, 'pending')""",
                (library_id, idempotency_key, request_hash),
            )
            if cursor.rowcount == 1:
                return None
            row = connection.execute(
                """SELECT request_hash, memory_id, status
                   FROM memory_write_idempotency
                   WHERE library_id = ? AND idempotency_key = ?""",
                (library_id, idempotency_key),
            ).fetchone()
        if row is None:
            raise MemoryKernelUnavailableError("idempotency reservation was not persisted")
        if str(row[0]) != request_hash:
            raise MemoryIdempotencyConflictError(
                "idempotency_key was already used for another memory content"
            )
        if str(row[2]) == "completed" and row[1] is not None:
            return str(row[1])
        if str(row[2]) == "pending":
            raise MemoryIdempotencyPendingError(
                "idempotency_key has an incomplete earlier memory write"
            )
        raise MemoryKernelUnavailableError("completed idempotency record has no memory identifier")

    def _complete_idempotency(
        self,
        library_id: str,
        idempotency_key: str,
        request_hash: str,
        memory_id: str,
    ) -> None:
        with closing(self._idempotency_db()) as connection, connection:
            cursor = connection.execute(
                """UPDATE memory_write_idempotency
                   SET memory_id = ?, status = 'completed',
                       completed_at = CURRENT_TIMESTAMP
                   WHERE library_id = ? AND idempotency_key = ?
                     AND request_hash = ? AND status = 'pending'""",
                (memory_id, library_id, idempotency_key, request_hash),
            )
            if cursor.rowcount != 1:
                raise MemoryKernelUnavailableError(
                    "idempotency completion did not update the pending reservation"
                )

    def _release_pending_idempotency(self, library_id: str, idempotency_key: str) -> None:
        with closing(self._idempotency_db()) as connection, connection:
            connection.execute(
                """DELETE FROM memory_write_idempotency
                   WHERE library_id = ? AND idempotency_key = ?
                     AND status = 'pending'""",
                (library_id, idempotency_key),
            )

    def resolve_region(self, request: ResolveRegionRequest) -> dict[str, str]:
        scope = _scope_for_cwd(request.cwd, None)
        return {
            "status": "bound",
            "cwd": request.cwd,
            "project_id": scope.space,
            "library_id": scope.space,
            "space": scope.space,
        }

    def _require_bundle(self) -> _RuntimeBundle:
        with self._lock:
            bundle = self._bundle
            startup_error = self._startup_error
        if bundle is None:
            raise MemoryKernelUnavailableError(startup_error or "mem2 runtime is not started")
        return bundle

    def _runtime_config(self, state_dir: Path) -> dict[str, Any]:
        milvus_uri = str(state_dir / "milvus.db")

        def store(collection: str) -> dict[str, object]:
            return {
                "target": _STORE_TARGET,
                "params": {
                    "uri": milvus_uri,
                    "dim": self._embedding_dimension,
                    "collection": collection,
                },
            }

        vector_enabled = self._vector_enabled
        engine_params: dict[str, str] = {
            "index_builder": "default",
            "retriever": "default",
        }
        base: dict[str, Any] = {
            "globals": {
                "schema_enabled": False,
                "vector_enabled": vector_enabled,
                "graph_enabled": False,
                "rerank_enabled": False,
                "embedder_dim": self._embedding_dimension,
            },
            "engine": {
                "default": {
                    "target": "cloud",
                    "params": engine_params,
                }
            },
            "constructor": {
                "default": {
                    "target": "hybrid",
                    "params": {"embedder": "default"} if vector_enabled else {},
                }
            },
            "permission": {"default": {"target": "allow_all"}},
            "recaller": {
                "vector": {
                    "target": "vector",
                    "params": {"min_similarity": _VECTOR_MIN_SIMILARITY},
                }
            },
            "vector_store": {
                "default": store("agent_memory_vectors"),
                "layers_l0": store("agent_memory_vectors_l0"),
                "layers_l1": store("agent_memory_vectors_l1"),
            },
            "fulltext_store": {
                "default": store("agent_memory_fulltext"),
                "layers_l0": store("agent_memory_fulltext_l0"),
                "layers_l1": store("agent_memory_fulltext_l1"),
            },
            "audit": {
                "default": {
                    "target": _AUDIT_TARGET,
                    "params": {"db_path": str(state_dir / "audit.sqlite3")},
                }
            },
        }
        endpoint = self._embedding
        if vector_enabled and endpoint is not None:
            base["embedder"] = {
                "default": {
                    "target": _EMBEDDER_TARGET,
                    "params": {
                        "model_name": endpoint.model,
                        "base_url": endpoint.base_url,
                        "api_key": self._embedding_api_key(),
                        "dim": self._embedding_dimension,
                        "timeout": endpoint.timeout_seconds,
                        "max_retries": endpoint.retries,
                    },
                }
            }
            base["query_parser"] = {
                "default": {
                    "target": "simple",
                    "params": {"embedder": "default"},
                }
            }
        return _deep_merge(base, self._config_overrides)

    def _prepare_embedding(self) -> None:
        endpoint = self._embedding
        if endpoint is None:
            self._vector_enabled = False
            self._vector_degradation = "embedding_not_configured"
            self._embedding_detail = (
                "embedding endpoint is not configured; vector recall is disabled"
            )
            return
        try:
            embedder = self._new_openai_embedder(self._embedding_dimension)
            vectors = embedder.embed(["embedding dimension probe"])
        except Exception as error:
            self._vector_enabled = False
            self._vector_degradation = "embedding_unavailable"
            self._embedding_detail = (
                "embedding endpoint probe failed; vector recall is disabled: "
                f"{type(error).__name__}"
            )
            return
        actual_dimension = len(vectors[0]) if vectors and vectors[0] else 0
        if actual_dimension != self._embedding_dimension:
            self._vector_enabled = False
            self._vector_degradation = "embedding_dimension_mismatch"
            self._embedding_detail = (
                "embedding dimension mismatch: configured "
                f"{self._embedding_dimension}, actual {actual_dimension}; "
                "vector recall is disabled"
            )
            return
        self._vector_enabled = True
        self._vector_degradation = ""
        self._embedding_detail = "embedding endpoint validated"

    def _new_openai_embedder(self, dimension: int) -> OpenAIEmbedder:
        endpoint = self._embedding
        if endpoint is None:
            raise ValueError("embedding endpoint is not configured")
        return OpenAIEmbedder(
            model_name=endpoint.model,
            base_url=endpoint.base_url,
            api_key=self._embedding_api_key(),
            dimension=dimension,
            timeout=endpoint.timeout_seconds,
            max_retries=endpoint.retries,
        )

    def _embedding_api_key(self) -> str:
        endpoint = self._embedding
        if endpoint is None or endpoint.api_key_file is None:
            return ""
        if self._embedding_api_key_cache is None:
            self._embedding_api_key_cache = endpoint.api_key_file.read_text(
                encoding="utf-8"
            ).strip()
        return self._embedding_api_key_cache


def _write_request_hash(request: CreateMemoryRequest) -> str:
    payload = {
        "library_id": request.library_id,
        "content": request.content,
        "memory_type": request.memory_type,
        "source_references": list(request.source_references),
        "actor": request.actor,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_response(
    request: MemoryWriteRequest, memory_id: str, *, status: str
) -> dict[str, object]:
    return {
        "id": memory_id,
        "memory_id": memory_id,
        "status": status,
        "library_id": getattr(request, "library_id", ""),
        "content": request.content,
        "memory_type": getattr(request, "memory_type", "capture"),
    }


def _scope_for_normalized_library(library_id: str) -> Scope:
    return Scope(
        org=_LOCAL_ORG,
        space=f"{LIBRARY_SCOPE_PREFIX}{library_id}",
        user=_LOCAL_USER,
        agent=_LOCAL_AGENT,
        session="",
    )


def _scope_for_cwd(cwd: str, session_id: str | None) -> Scope:
    digest = hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:24]
    return Scope(
        org=_LOCAL_ORG,
        space=f"{CWD_SCOPE_PREFIX}{digest}",
        user=_LOCAL_USER,
        agent=_LOCAL_AGENT,
        session=session_id or "",
    )


def _security_for(scope: Scope) -> RequestSecurityContext:
    return legacy_request_context(
        Scope(
            org=scope.org,
            user=scope.user,
            agent=scope.agent,
            session=scope.session,
        )
    )


def _classify_failure(error: Exception, *, embedding_configured: bool) -> str:
    text = f"{type(error).__module__} {type(error).__name__} {error}".casefold()
    if embedding_configured or "embed" in text or "model" in text or "openai" in text:
        return "model_unavailable"
    if "fulltext" in text or "keyword" in text or "bm25" in text:
        return "fulltext_unavailable"
    if "vector" in text or "milvus" in text:
        return "vector_unavailable"
    return "retrieval_unavailable"


def _channel_failure(channel: str, error: Any) -> str:
    text = f"{getattr(error, 'error_type', '')} {getattr(error, 'message', '')}".casefold()
    if channel == "vector" and ("embed" in text or "model" in text):
        return "model_unavailable"
    if channel == "keyword":
        return "fulltext_unavailable"
    if channel == "vector":
        return "vector_unavailable"
    if channel == "graph":
        return "graph_unavailable"
    return f"{channel}_unavailable"


def _search_response(
    request: SearchMemoryRequest,
    scope: Scope,
    items: list[Any],
    errors: list[Any],
    degradation: list[str],
    *,
    vector_status: str,
) -> dict[str, object]:
    results = [_result_item(item, request) for item in items]
    response: dict[str, object] = {
        "status": "bound",
        "scope": {
            "kind": "explicit_library" if request.library_id else "project_cwd",
            "cwd": request.cwd,
            "library_id": request.library_id,
            "space": scope.space,
        },
        "query": request.query,
        "degraded": bool(degradation),
        "degradation": degradation,
        "vector_index_status": vector_status,
        "graph_index_status": "not_configured",
        "results": results[: max(1, min(request.limit, 100))],
    }
    if request.library_id is not None:
        response["library_id"] = request.library_id
    if errors:
        response["retrieval_errors"] = [
            {
                "channel": getattr(error.channel, "value", str(error.channel)),
                "source": error.source,
                "error_type": error.error_type,
                "message": error.message,
            }
            for error in errors
        ]
    return response


def _result_item(item: Any, request: SearchMemoryRequest) -> dict[str, object]:
    if isinstance(item, MemoryUnit):
        unit_id = item.id
        content = item.content
        score = 0.0
        system_metadata = dict(item.system_metadata)
        user_metadata = dict(item.user_metadata)
    else:
        unit_id = str(item.unit_id)
        content = str(item.content)
        score = float(item.score)
        system_metadata = dict(item.system_metadata)
        user_metadata = dict(item.user_metadata)
    return {
        "chunk_id": unit_id,
        "document_id": unit_id,
        "content": content,
        "library_id": request.library_id or "",
        "path": f"memory://{unit_id}",
        "heading": None,
        "start_line": 1,
        "end_line": max(1, content.count("\n") + 1),
        "source_version": unit_id,
        "source_type": "mem2",
        "classification": "direct",
        "score": score,
        "retrieval_sources": ["hybrid"],
        "system_metadata": system_metadata,
        "user_metadata": user_metadata,
    }


def _close_quietly(closeable: object) -> None:
    close = getattr(closeable, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


def _close_stores(stores: list[object]) -> None:
    seen: set[int] = set()
    for store in reversed(stores):
        if id(store) in seen:
            continue
        seen.add(id(store))
        _close_quietly(store)
    stores.clear()
