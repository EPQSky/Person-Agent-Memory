"""JiuwenMemory 0.2.0 runtime adapter for the public memory kernel port."""

from __future__ import annotations

import asyncio
import base64
import difflib
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
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
from jiuwen_memory.common.type_def import memory_key
from jiuwen_memory.common.type_def.memory_codec import dumps
from jiuwen_memory.construction import EvolveMode
from jiuwen_memory.control import (
    DeleteMode,
    DeleteSelector,
    JobStatus,
    MemoryPatch,
    UpdateMode,
)
from jiuwen_memory.storage import FulltextStore, GraphStore, VectorStore
from jiuwen_memory.storage.fulltext import FulltextProducer
from jiuwen_memory.storage.graph import GraphProducer
from jiuwen_memory.storage.kv import load_units
from jiuwen_memory.storage.types import ScoredID, TextQuery
from jiuwen_memory.storage.vector import VectorProducer

from personal_agent_memory.config import MemoryKernelKind
from personal_agent_memory.memory_kernel import (
    CWD_SCOPE_PREFIX,
    LIBRARY_SCOPE_PREFIX,
    AssociateWorktreeRegionRequest,
    AuditMemoryRequest,
    BindProjectRegionRequest,
    CaptureContractError,
    CaptureEventsRequest,
    CaptureMemoryRequest,
    CaptureRoundsRequest,
    CreateMemoryRegionRequest,
    CreateMemoryRequest,
    DeleteMemoryRequest,
    DeleteRegionBindingRequest,
    DocumentInspectRequest,
    DocumentUpdateRequest,
    EvolveMemoryRequest,
    GetMemoryRegionRequest,
    HistoryDiffRequest,
    KernelHealth,
    ListDocumentsRequest,
    ListForgottenRequest,
    MemoryGovernanceError,
    MemoryIdempotencyConflictError,
    MemoryIdempotencyPendingError,
    MemoryKernelUnavailableError,
    MemoryLifecycleConflictError,
    MemoryRecordNotFoundError,
    MemoryRegionTreeRequest,
    MemoryRevisionConflictError,
    MemoryUpdateRequest,
    MemoryWriteRequest,
    PreviewDocumentDeleteRequest,
    PreviewDocumentUpdateRequest,
    ResolveRegionRequest,
    RestoreDocumentRequest,
    RestoreForgottenRequest,
    RetentionCleanupRequest,
    RetentionPolicyRequest,
    SearchMemoryRequest,
    UnavailableMemoryKernel,
    UpdateRegionBindingRequest,
    UpdateRetentionPolicyRequest,
    validate_library_id,
)
from personal_agent_memory.model_client import ModelEndpoint
from personal_agent_memory.regions import (
    MAX_SESSION_ID_LENGTH,
    MemoryRegion,
    RegionBindingError,
    RegionError,
    RegionRegistry,
)
from personal_agent_memory.storage import (
    MilvusLiteFulltextStore,
    MilvusLiteGraphStore,
    MilvusLiteVectorStore,
    SQLiteAuditLogger,
    SQLiteKVStore,
)

_STORE_TARGET = "personal-agent-memory-local"
_AUDIT_TARGET = "personal-agent-memory-sqlite"
_EMBEDDER_TARGET = "personal-agent-memory-openai"
_GRAPH_PROJECTION_SCHEMA = 2
_GRAPH_PROJECTION_MANIFEST_SCHEMA = 1
_LOCAL_ORG = "local"
_LOCAL_USER = "local"
_LOCAL_AGENT = "personal-agent-memory"
_CODEX_AGENT = "codex"
_SEARCH_AGENTS = (_LOCAL_AGENT, _CODEX_AGENT)
_DEFAULT_EMBEDDING_DIM = 1536
_VECTOR_MIN_SIMILARITY = 0.5
_FULLTEXT_CONTENT_LIMIT = 65_535
_EVOLVE_CHUNK_LIMIT = 60_000
_STATE_VERSION = "jiuwen-memory-0.2.0"
_DEFAULT_SESSION_RETENTION_DAYS = 30
_SESSION_RETENTION_SCHEMA = """CREATE TABLE IF NOT EXISTS session_retention_policy (
    region_id TEXT PRIMARY KEY,
    retention_days INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)"""
_CAPTURE_EVOLUTIONS_SCHEMA = """CREATE TABLE IF NOT EXISTS capture_evolutions (
    evolve_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    region_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL DEFAULT '',
    memory_id TEXT,
    status TEXT NOT NULL DEFAULT 'completed'
        CHECK (status IN ('pending', 'completed')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
)"""
_GOVERNANCE_AUDIT_SCHEMA = """CREATE TABLE IF NOT EXISTS governance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    source TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    before_content TEXT NOT NULL DEFAULT '',
    after_content TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'committed',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)"""
_MAINTENANCE_SCHEMA = (
    _SESSION_RETENTION_SCHEMA
    + ";\n"
    + _CAPTURE_EVOLUTIONS_SCHEMA
    + ";\n"
    + _GOVERNANCE_AUDIT_SCHEMA
    + ";"
)
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


@GraphProducer.register(_STORE_TARGET)  # type: ignore[untyped-decorator]
def _build_graph_store(config: Any) -> GraphStore:
    store = MilvusLiteGraphStore(
        uri=str(config.get("uri")),
        collection=str(config.get("collection")),
        edge_collection=str(config.get("edge_collection")),
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
    graph_store: MilvusLiteGraphStore | None


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
        regions: RegionRegistry | None = None,
    ) -> None:
        if embedding_dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self._state_dir = state_dir.expanduser().resolve()
        self._embedding = embedding
        self._embedding_dimension = embedding_dimension
        self._config_overrides = config_overrides
        self._regions = regions or RegionRegistry(self._state_dir / "regions.sqlite3")
        self._bundle: _RuntimeBundle | None = None
        self._startup_error: str | None = None
        self._vector_enabled = False
        self._vector_degradation = "embedding_not_configured"
        self._embedding_detail = "embedding endpoint is not configured; vector recall is disabled"
        self._embedding_api_key_cache: str | None = None
        self._graph_errors: dict[str, str] = {}
        self._graph_projection_path = self._state_dir / "mem2" / "graph_projection_errors.json"
        self._graph_projection_manifest_path = (
            self._state_dir / "mem2" / "graph_projection_manifests.json"
        )
        self._graph_projection_lock = threading.RLock()
        self._graph_projection_errors = self._load_graph_projection_errors()
        self._graph_projection_manifests = self._load_graph_projection_manifests()
        self._lock = threading.RLock()
        self._maintenance_lock = threading.RLock()
        self._graph_write_lock = threading.RLock()
        self._governance_locks: dict[str, threading.RLock] = {}
        self._governance_locks_guard = threading.Lock()
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
            graph_store = next(
                (store for store in stores if isinstance(store, MilvusLiteGraphStore)),
                None,
            )
            self._bundle = _RuntimeBundle(runtime, kv_store, tuple(stores), graph_store)

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

    def graph_status(self, library_id: str | None = None) -> dict[str, object]:
        with self._lock:
            bundle = self._bundle
        if bundle is None or bundle.graph_store is None:
            return {
                "status": "not_configured",
                "backend": "milvus-lite",
                "library_id": library_id or "",
                "total_documents": 0,
                "projected_documents": 0,
                "last_error": "",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        identifier = validate_library_id(library_id) if library_id is not None else None
        region = self._graph_region_or_none(identifier) if identifier is not None else None
        region_id = region.region_id if region is not None else identifier or ""
        try:
            bundle.graph_store.health()
            counts = {"nodes": 0, "edges": 0}
            store_errors: list[str] = []
            projection_issues: list[str] = []
            projection_errors: list[str] = []
            transient_errors: list[str] = []
            active_documents = 0
            projected_documents = 0
            if identifier is None:
                scopes = self._graph_status_scopes(bundle)
            elif region is None:
                scopes = [_scope_for_normalized_library(identifier)]
            else:
                scopes = self._region_scopes(bundle, region)
            with self._graph_write_lock:
                for scope in scopes:
                    scope_counts = bundle.graph_store.counts(scope)
                    counts["nodes"] += int(scope_counts.get("nodes", 0))
                    counts["edges"] += int(scope_counts.get("edges", 0))
                    expected, actual, mismatch = self._graph_projection_snapshot(
                        bundle,
                        scope,
                    )
                    active_documents += expected
                    projected_documents += actual
                    if mismatch is not None:
                        projection_issues.append(mismatch)
                    store_error = self._graph_store_error(bundle, scope)
                    if store_error is not None:
                        store_errors.append(store_error)
                    projection_error = self._graph_projection_error(scope=scope)
                    if projection_error is not None:
                        projection_errors.append(projection_error)
                    transient_error = self._graph_error_for(scope=scope)
                    if transient_error is not None:
                        transient_errors.append(transient_error)
            if region is not None:
                for projection_error in self._graph_projection_errors_for_region(region):
                    if projection_error not in projection_errors:
                        projection_errors.append(projection_error)
                for transient_error in self._graph_errors_for_region(region):
                    if transient_error not in transient_errors:
                        transient_errors.append(transient_error)
        except Exception as error:
            detail = str(error)
            return {
                "status": "degraded",
                "backend": "milvus-lite",
                "library_id": region_id,
                "detail": detail,
                "total_documents": 0,
                "projected_documents": 0,
                "last_error": detail,
                "updated_at": datetime.now(UTC).isoformat(),
                "nodes": 0,
                "edges": 0,
            }
        if identifier is None:
            projection_error = self._any_graph_projection_error()
            transient_error = self._any_graph_error()
        else:
            projection_error = (
                projection_errors[0]
                if projection_errors
                else self._graph_projection_error(
                    region_id=region.region_id if region is not None else identifier
                )
            )
            transient_error = (
                transient_errors[0]
                if transient_errors
                else self._graph_error_for(
                    region_id=region.region_id if region is not None else identifier
                )
            )
        detail = (
            projection_error
            or (projection_issues[0] if projection_issues else None)
            or (store_errors[0] if store_errors else None)
            or transient_error
            or "one-hop graph expansion available"
        )
        degraded = bool(projection_error or projection_issues or store_errors or transient_error)
        return {
            "status": "degraded" if degraded else "ready",
            "backend": "milvus-lite",
            "library_id": region_id,
            "detail": detail,
            "total_documents": active_documents,
            "projected_documents": projected_documents,
            "last_error": "" if not degraded else detail,
            "updated_at": datetime.now(UTC).isoformat(),
            **counts,
        }

    async def rebuild_graph(self, library_id: str) -> dict[str, object]:
        return await asyncio.to_thread(self._rebuild_graph_serialized, library_id)

    def _rebuild_graph_serialized(self, library_id: str) -> dict[str, object]:
        identifier = validate_library_id(library_id)
        with (
            self._maintenance_lock,
            self._governance_region_lock(identifier),
            self._graph_write_lock,
        ):
            return self._rebuild_graph(identifier)

    def _rebuild_graph(self, library_id: str) -> dict[str, object]:
        bundle = self._require_bundle()
        identifier = validate_library_id(library_id)
        region = self._graph_region_or_none(identifier)
        region_id = region.region_id if region is not None else identifier
        if bundle.graph_store is None:
            return {"library_id": region_id, "status": "not_configured", "job_id": ""}
        job_ids: list[str] = []
        failure: str | None = None
        failure_scope: Scope | None = None
        scopes: list[Scope] = []
        try:
            bundle.graph_store.health()
            scopes = (
                self._region_scopes(bundle, region)
                if region is not None
                else [_scope_for_normalized_library(identifier)]
            )
            for scope in scopes:
                clear_diagnostics = getattr(bundle.graph_store, "clear_diagnostics", None)
                if callable(clear_diagnostics):
                    clear_diagnostics(scope)
                bundle.graph_store.clear(scope)
                job_id = self._associate_graph_scope(
                    bundle,
                    scope,
                    region_id=region.region_id if region is not None else None,
                )
                if job_id is None:
                    failure_scope = scope
                    failure = self._graph_error_for(scope=scope) or "graph association failed"
                    break
                job_ids.append(job_id)
                self._verify_graph_projection(bundle, scope)
        except Exception as error:
            failure = str(error)
        if failure is not None:
            if failure_scope is not None:
                self._set_graph_error(failure, scope=failure_scope)
                self._set_graph_projection_error(failure, scope=failure_scope)
            elif region is None:
                library_scope = _scope_for_normalized_library(identifier)
                self._set_graph_error(failure, scope=library_scope)
                self._set_graph_projection_error(failure, scope=library_scope)
            else:
                self._set_graph_error(failure, region_id=region_id)
                self._set_graph_projection_error(failure, region_id=region_id)
            return {
                "library_id": region_id,
                "status": "degraded",
                "detail": failure,
                "job_id": job_ids[0] if job_ids else "",
                "job_ids": job_ids,
            }
        self._clear_graph_error(region_id=region_id)
        if region is not None:
            self._clear_graph_errors_for_region(region)
            self._clear_graph_projection_errors_for_region(region)
        else:
            self._clear_graph_projection_error(scope=_scope_for_normalized_library(identifier))
            self._clear_graph_projection_error(region_id=identifier)
            self._clear_graph_error(region_id=identifier)
        return {
            "library_id": region_id,
            "status": "ready",
            "job_id": job_ids[0] if job_ids else "",
            "job_ids": job_ids,
            "graph": self.graph_status(region_id),
        }

    def _verify_graph_projection(self, bundle: _RuntimeBundle, scope: Scope) -> None:
        graph_store = bundle.graph_store
        if graph_store is None:
            return
        store_error = self._graph_store_error(bundle, scope)
        if store_error is not None:
            raise MemoryKernelUnavailableError(store_error)
        _, _, mismatch = self._graph_projection_snapshot(bundle, scope)
        if mismatch is not None:
            raise MemoryKernelUnavailableError(mismatch)

    def _active_graph_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> list[MemoryUnit]:
        return sorted(
            (
                unit
                for unit in self._units_in_scope(bundle, scope)
                if unit.system_metadata.get("middle") != "true"
                and unit.lifecycle.value == "active"
                and unit.user_metadata.get("restore_pending") is not True
            ),
            key=lambda unit: unit.id,
        )

    def _graph_projection_contents(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> tuple[int, int, str | None]:
        units_by_id = {unit.id: unit for unit in units}
        expected = set(units_by_id)
        graph_store = bundle.graph_store
        node_ids = getattr(graph_store, "node_ids", None)
        if graph_store is None or not callable(node_ids):
            return len(expected), len(expected), None
        try:
            actual = set(node_ids(scope))
        except Exception as error:
            return (
                len(expected),
                0,
                f"graph projection read failed: {error}",
            )
        edge_endpoints = getattr(graph_store, "edge_endpoints", None)
        dangling_edges = 0
        if callable(edge_endpoints):
            try:
                dangling_edges = sum(
                    1
                    for source, target in edge_endpoints(scope)
                    if source not in actual or target not in actual
                )
            except Exception as error:
                return (
                    len(expected),
                    len(actual),
                    f"graph edge projection read failed: {error}",
                )
        stale_nodes = 0
        get_nodes = getattr(graph_store, "get", None)
        if callable(get_nodes):
            try:
                projected_nodes = get_nodes(scope, sorted(expected & actual))
            except Exception as error:
                return (
                    len(expected),
                    len(actual),
                    f"graph node projection read failed: {error}",
                )
            for node in projected_nodes:
                unit = units_by_id.get(node.id)
                if unit is None:
                    continue
                if node.label != unit.tier.value or node.properties.get("content") != unit.content:
                    stale_nodes += 1
        missing = expected - actual
        extra = actual - expected
        if not missing and not extra and dangling_edges == 0 and stale_nodes == 0:
            return len(expected), len(actual), None
        return (
            len(expected),
            len(actual),
            "graph projection mismatch: "
            f"missing={len(missing)} extra={len(extra)} "
            f"dangling_edges={dangling_edges} stale_nodes={stale_nodes}",
        )

    def _graph_projection_snapshot(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> tuple[int, int, str | None]:
        units = self._active_graph_units(bundle, scope)
        expected_count, actual_count, mismatch = self._graph_projection_contents(
            bundle,
            scope,
            units,
        )
        if mismatch is not None:
            return expected_count, actual_count, mismatch
        if not units and actual_count == 0:
            return expected_count, actual_count, None
        manifest = self._graph_projection_manifest(scope)
        if manifest is None:
            return (
                expected_count,
                actual_count,
                "graph projection manifest missing; rebuild required",
            )
        try:
            kv_fingerprint = self._graph_kv_fingerprint(units)
            graph_fingerprint = self._graph_projection_fingerprint(bundle, scope)
        except Exception as error:
            return (
                expected_count,
                actual_count,
                f"graph projection fingerprint read failed: {error}",
            )
        if (
            manifest["kv_fingerprint"] == kv_fingerprint
            and manifest["graph_fingerprint"] == graph_fingerprint
        ):
            return expected_count, actual_count, None
        edge_mismatch = self._graph_edge_mismatch(bundle, scope, units)
        return (
            expected_count,
            actual_count,
            edge_mismatch or "graph projection changed after verification; rebuild required",
        )

    @staticmethod
    def _graph_properties_fingerprint(properties: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(properties),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def _graph_edge_signature(
        cls,
        source: str,
        target: str,
        relation: str,
        properties: Mapping[str, Any],
    ) -> tuple[str, str, str, str]:
        left, right = sorted((str(source), str(target)))
        return (
            left,
            right,
            str(relation),
            cls._graph_properties_fingerprint(properties),
        )

    @classmethod
    def _graph_kv_fingerprint(cls, units: list[MemoryUnit]) -> str:
        payload = [
            {
                "id": unit.id,
                "content": unit.content,
                "tier": unit.tier.value,
            }
            for unit in units
        ]
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _graph_projection_fingerprint(
        cls,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> str:
        graph_store = bundle.graph_store
        if graph_store is None:
            return ""
        projection_nodes = getattr(graph_store, "projection_nodes", None)
        projection_edges = getattr(graph_store, "projection_edges", None)
        if not callable(projection_nodes) or not callable(projection_edges):
            return "unavailable"
        nodes = sorted(
            (
                node.id,
                node.label,
                cls._graph_properties_fingerprint(node.properties),
            )
            for node in projection_nodes(scope)
        )
        edges = sorted(
            (
                edge.source,
                edge.target,
                edge.relation,
                cls._graph_properties_fingerprint(edge.properties),
            )
            for edge in projection_edges(scope)
        )
        return hashlib.sha256(
            json.dumps(
                {"nodes": nodes, "edges": edges},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _expected_graph_edge_signatures(
        cls,
        bundle: _RuntimeBundle,
        units: list[MemoryUnit],
    ) -> set[tuple[str, str, str, str]] | None:
        engine = getattr(bundle.runtime.api, "_engine", None)
        evolver = getattr(engine, "_evolver", None)
        associator = getattr(evolver, "_associator", None)
        associate = getattr(associator, "associate", None)
        if not callable(associate):
            return None
        relations = associate(list(units))
        return {
            cls._graph_edge_signature(
                relation.source_id,
                relation.target_id,
                relation.relation,
                getattr(relation, "metadata", {}) or {},
            )
            for relation in relations
        }

    @classmethod
    def _actual_graph_edge_signatures(
        cls,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> set[tuple[str, str, str, str]] | None:
        graph_store = bundle.graph_store
        projection_edges = getattr(graph_store, "projection_edges", None)
        if not callable(projection_edges):
            return None
        return {
            cls._graph_edge_signature(
                edge.source,
                edge.target,
                edge.relation,
                edge.properties,
            )
            for edge in projection_edges(scope)
        }

    def _graph_edge_mismatch(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> str | None:
        try:
            expected = self._expected_graph_edge_signatures(bundle, units)
            actual = self._actual_graph_edge_signatures(bundle, scope)
        except Exception as error:
            return f"graph edge projection verification failed: {error}"
        if expected is None or actual is None:
            return None
        missing = expected - actual
        stale = actual - expected
        if not missing and not stale:
            return None
        return (
            "graph edge projection mismatch: "
            f"expected_edges={len(expected)} actual_edges={len(actual)} "
            f"missing_edges={len(missing)} stale_edges={len(stale)}"
        )

    def _record_graph_projection(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> None:
        units = self._active_graph_units(bundle, scope)
        _, _, mismatch = self._graph_projection_contents(bundle, scope, units)
        if mismatch is not None:
            raise MemoryKernelUnavailableError(mismatch)
        expected = self._expected_graph_edge_signatures(bundle, units)
        actual = self._actual_graph_edge_signatures(bundle, scope)
        if expected is None or actual is None:
            raise MemoryKernelUnavailableError("graph edge projection verification is unavailable")
        if expected != actual:
            raise MemoryKernelUnavailableError(
                self._graph_edge_mismatch(bundle, scope, units) or "graph edge projection mismatch"
            )
        key = self._graph_projection_key(None, scope)
        manifest = {
            "kv_fingerprint": self._graph_kv_fingerprint(units),
            "graph_fingerprint": self._graph_projection_fingerprint(bundle, scope),
        }
        with self._graph_projection_lock:
            self._graph_projection_manifests[key] = manifest
            self._persist_graph_projection_manifests_locked()

    def _associate_graph_scope(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str | None = None,
    ) -> str | None:
        with self._graph_write_lock:
            return self._associate_graph_scope_locked(
                bundle,
                scope,
                region_id=region_id,
            )

    def _associate_graph_scope_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str | None = None,
    ) -> str | None:
        if bundle.graph_store is None:
            return None
        try:
            security = _security_for(scope)
            job_id = bundle.runtime.api.evolve(
                scope,
                EvolveMode.ASSOCIATE,
                security=security,
            )
            job_error = self._evolve_job_error(bundle, str(job_id))
            if job_error is not None:
                raise MemoryKernelUnavailableError(job_error)
            store_error = self._graph_store_error(bundle, scope)
            if store_error is not None:
                raise MemoryKernelUnavailableError(store_error)
            if region_id is not None:
                self._prune_inactive_graph_nodes(
                    bundle,
                    scope,
                    region_id=region_id,
                )
            self._record_graph_projection(bundle, scope)
            self._clear_graph_error(region_id=region_id, scope=scope)
            self._clear_graph_projection_error(scope=scope)
            return str(job_id)
        except Exception as error:
            failure = str(error)
            self._set_graph_error(
                failure,
                region_id=region_id,
                scope=scope,
            )
            self._set_graph_projection_error(failure, scope=scope)
            return None

    @staticmethod
    def _evolve_job_error(bundle: _RuntimeBundle, job_id: str) -> str | None:
        api = bundle.runtime.api
        engine = getattr(api, "_engine", None)
        scheduler = getattr(engine, "_scheduler", None)
        status = getattr(scheduler, "status", None)
        if not callable(status):
            return None
        try:
            job = status(job_id)
        except Exception:
            return None
        if getattr(job, "status", None) != JobStatus.FAILED:
            return None
        detail = getattr(job, "detail", {})
        error = detail.get("error") or detail.get("error_type")
        return f"graph association job failed: {error or 'unknown error'}"

    @staticmethod
    def _graph_store_error(bundle: _RuntimeBundle, scope: Scope) -> str | None:
        graph_store = bundle.graph_store
        if graph_store is None:
            return None
        last_error = getattr(graph_store, "last_error", None)
        if not callable(last_error):
            return None
        try:
            error = last_error(scope)
        except Exception as read_error:
            return f"graph diagnostics unavailable: {read_error}"
        return None if error is None else str(error)

    def _graph_projection_error(
        self,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> str | None:
        key = self._graph_projection_key(region_id, scope)
        if not key:
            return None
        with self._graph_projection_lock:
            return self._graph_projection_errors.get(key)

    def _any_graph_projection_error(self) -> str | None:
        with self._graph_projection_lock:
            return next(iter(self._graph_projection_errors.values()), None)

    @staticmethod
    def _graph_error_key(region_id: str | None, scope: Scope | None) -> str:
        if scope is not None:
            return Mem2MemoryKernel._graph_projection_key(None, scope)
        if region_id is not None:
            return Mem2MemoryKernel._graph_projection_key(region_id, None)
        return ""

    @staticmethod
    def _graph_projection_key(region_id: str | None, scope: Scope | None) -> str:
        if scope is not None:
            payload = json.dumps(
                [scope.org, scope.space, scope.user, scope.agent, scope.session],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
            return f"@scope/v2:{encoded}"
        if region_id is not None:
            encoded = (
                base64.urlsafe_b64encode(region_id.encode("utf-8")).decode("ascii").rstrip("=")
            )
            return f"@region/v2:{encoded}"
        return ""

    @staticmethod
    def _decode_graph_projection_key(
        key: str,
    ) -> tuple[str | None, Scope | None]:
        prefix = ""
        if key.startswith("@scope/v2:"):
            prefix = "@scope/v2:"
        elif key.startswith("@region/v2:"):
            prefix = "@region/v2:"
        else:
            return None, None
        encoded = key.removeprefix(prefix)
        try:
            padding = "=" * (-len(encoded) % 4)
            decoded = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
            if prefix == "@region/v2:":
                return decoded, None
            payload = json.loads(decoded)
        except (ValueError, UnicodeError, json.JSONDecodeError):
            return None, None
        if not isinstance(payload, list) or len(payload) != 5:
            return None, None
        return None, Scope(
            org=str(payload[0]),
            space=str(payload[1]),
            user=str(payload[2]),
            agent=str(payload[3]),
            session=str(payload[4]),
        )

    @staticmethod
    def _scope_matches_region(scope: Scope, region: MemoryRegion) -> bool:
        return scope.space == region.space and (
            region.kind != "session" or scope.session == (region.session_id or "")
        )

    def _set_graph_error(
        self,
        error: str,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        key = self._graph_error_key(region_id, scope)
        with self._graph_projection_lock:
            self._graph_errors[key] = error

    def _clear_graph_error(
        self,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        key = self._graph_error_key(region_id, scope)
        with self._graph_projection_lock:
            self._graph_errors.pop(key, None)

    def _graph_error_for(
        self,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> str | None:
        key = self._graph_error_key(region_id, scope)
        with self._graph_projection_lock:
            return self._graph_errors.get(key)

    def _any_graph_error(self) -> str | None:
        with self._graph_projection_lock:
            return next(iter(self._graph_errors.values()), None)

    def _load_graph_projection_errors(self) -> dict[str, str]:
        try:
            payload = json.loads(self._graph_projection_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        current_schema = payload.get("schema") == _GRAPH_PROJECTION_SCHEMA
        raw_errors = payload.get("errors") if current_schema else payload
        if not isinstance(raw_errors, dict):
            return {}
        errors: dict[str, str] = {}
        for raw_key, raw_error in raw_errors.items():
            if not str(raw_error):
                continue
            key = str(raw_key)
            if current_schema:
                if not key.startswith(("@scope/v2:", "@region/v2:")):
                    continue
            elif key.startswith("region:"):
                key = self._graph_projection_key(
                    region_id=key.removeprefix("region:"),
                    scope=None,
                )
            elif key.startswith("scope:"):
                continue
            else:
                continue
            errors[key] = str(raw_error)
        return errors

    def _persist_graph_projection_errors_locked(self) -> None:
        self._graph_projection_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._graph_projection_path.with_name(f"{self._graph_projection_path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": _GRAPH_PROJECTION_SCHEMA,
                    "errors": self._graph_projection_errors,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._graph_projection_path)

    def _load_graph_projection_manifests(self) -> dict[str, dict[str, str]]:
        try:
            payload = json.loads(self._graph_projection_manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _GRAPH_PROJECTION_MANIFEST_SCHEMA
            or not isinstance(payload.get("projections"), dict)
        ):
            return {}
        manifests: dict[str, dict[str, str]] = {}
        for raw_key, raw_manifest in payload["projections"].items():
            key = str(raw_key)
            if not key.startswith("@scope/v2:") or not isinstance(raw_manifest, dict):
                continue
            kv_fingerprint = raw_manifest.get("kv_fingerprint")
            graph_fingerprint = raw_manifest.get("graph_fingerprint")
            if not isinstance(kv_fingerprint, str) or not isinstance(
                graph_fingerprint,
                str,
            ):
                continue
            manifests[key] = {
                "kv_fingerprint": kv_fingerprint,
                "graph_fingerprint": graph_fingerprint,
            }
        return manifests

    def _persist_graph_projection_manifests_locked(self) -> None:
        self._graph_projection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._graph_projection_manifest_path.with_name(
            f"{self._graph_projection_manifest_path.name}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "schema": _GRAPH_PROJECTION_MANIFEST_SCHEMA,
                    "projections": self._graph_projection_manifests,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._graph_projection_manifest_path)

    def _graph_projection_manifest(
        self,
        scope: Scope,
    ) -> dict[str, str] | None:
        key = self._graph_projection_key(None, scope)
        with self._graph_projection_lock:
            manifest = self._graph_projection_manifests.get(key)
            return None if manifest is None else dict(manifest)

    def _set_graph_projection_error(
        self,
        error: str,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        key = self._graph_projection_key(region_id, scope)
        if not key:
            return
        with self._graph_projection_lock:
            self._graph_projection_errors[key] = error
            try:
                self._persist_graph_projection_errors_locked()
            except OSError as persist_error:
                self._set_graph_error(
                    f"graph projection status persist failed: {persist_error}",
                    region_id=region_id,
                    scope=scope,
                )

    def _clear_graph_projection_error(
        self,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        key = self._graph_projection_key(region_id, scope)
        if not key:
            return
        with self._graph_projection_lock:
            if self._graph_projection_errors.pop(key, None) is None:
                return
            try:
                self._persist_graph_projection_errors_locked()
            except OSError as persist_error:
                self._set_graph_error(
                    f"graph projection status persist failed: {persist_error}",
                    region_id=region_id,
                    scope=scope,
                )

    def _clear_graph_projection_errors_for_region(
        self,
        region: MemoryRegion,
    ) -> None:
        region_key = self._graph_projection_key(region.region_id, None)
        with self._graph_projection_lock:
            stale_keys = []
            for key in self._graph_projection_errors:
                region_id, scope = self._decode_graph_projection_key(key)
                if key == region_key or (
                    scope is not None and self._scope_matches_region(scope, region)
                ):
                    stale_keys.append(key)
            if not stale_keys:
                return
            for key in stale_keys:
                self._graph_projection_errors.pop(key, None)
            try:
                self._persist_graph_projection_errors_locked()
            except OSError as persist_error:
                self._set_graph_error(
                    f"graph projection status persist failed: {persist_error}",
                    region_id=region.region_id,
                )

    def _clear_graph_errors_for_region(self, region: MemoryRegion) -> None:
        region_key = self._graph_projection_key(region.region_id, None)
        with self._graph_projection_lock:
            stale_keys = []
            for key in self._graph_errors:
                region_id, scope = self._decode_graph_projection_key(key)
                if key == region_key or (
                    scope is not None and self._scope_matches_region(scope, region)
                ):
                    stale_keys.append(key)
            for key in stale_keys:
                self._graph_errors.pop(key, None)

    def _graph_projection_errors_for_region(
        self,
        region: MemoryRegion,
    ) -> list[str]:
        region_key = self._graph_projection_key(region.region_id, None)
        with self._graph_projection_lock:
            errors = []
            for key, error in self._graph_projection_errors.items():
                region_id, scope = self._decode_graph_projection_key(key)
                if key == region_key or (
                    scope is not None and self._scope_matches_region(scope, region)
                ):
                    errors.append(error)
            return errors

    def _graph_errors_for_region(self, region: MemoryRegion) -> list[str]:
        region_key = self._graph_projection_key(region.region_id, None)
        with self._graph_projection_lock:
            errors = []
            for key, error in self._graph_errors.items():
                region_id, scope = self._decode_graph_projection_key(key)
                if key == region_key or (
                    scope is not None and self._scope_matches_region(scope, region)
                ):
                    errors.append(error)
            return errors

    def _remove_graph_nodes(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        node_ids: list[str],
        region_id: str,
    ) -> None:
        with self._graph_write_lock:
            self._remove_graph_nodes_locked(bundle, scope, node_ids, region_id)

    def _remove_graph_nodes_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        node_ids: list[str],
        region_id: str,
    ) -> None:
        graph_store = bundle.graph_store
        if graph_store is None or not node_ids:
            return
        try:
            graph_store.delete(scope, node_ids=node_ids)
            self._record_graph_projection(bundle, scope)
        except Exception as error:
            failure = f"graph cleanup failed: {error}"
            self._set_graph_error(failure, region_id=region_id)
            self._set_graph_projection_error(failure, scope=scope)

    def _prune_inactive_graph_nodes(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str,
    ) -> None:
        graph_store = bundle.graph_store
        if graph_store is None:
            return
        node_ids = getattr(graph_store, "node_ids", None)
        if not callable(node_ids):
            return
        active_ids = {
            unit.id
            for unit in self._units_in_scope(bundle, scope)
            if unit.system_metadata.get("middle") != "true"
            and unit.lifecycle.value == "active"
            and unit.user_metadata.get("restore_pending") is not True
        }
        try:
            stale_ids = sorted(set(node_ids(scope)) - active_ids)
            if stale_ids:
                graph_store.delete(scope, node_ids=stale_ids)
        except Exception as error:
            failure = f"graph projection cleanup failed: {error}"
            self._set_graph_error(failure, region_id=region_id)
            self._set_graph_projection_error(failure, scope=scope)
            raise MemoryKernelUnavailableError(failure) from error

    def _reproject_graph_scope(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str,
    ) -> None:
        with self._graph_write_lock:
            self._reproject_graph_scope_locked(
                bundle,
                scope,
                region_id=region_id,
            )

    def _reproject_graph_scope_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str,
    ) -> None:
        graph_store = bundle.graph_store
        if graph_store is None:
            return
        try:
            clear_diagnostics = getattr(graph_store, "clear_diagnostics", None)
            if callable(clear_diagnostics):
                clear_diagnostics(scope)
            graph_store.clear(scope)
            job_id = self._associate_graph_scope(
                bundle,
                scope,
                region_id=region_id,
            )
            if job_id is None:
                return
            self._verify_graph_projection(bundle, scope)
        except Exception as error:
            failure = f"graph reprojection failed: {error}"
            self._set_graph_error(failure, region_id=region_id)
            self._set_graph_projection_error(failure, scope=scope)

    def _region_id_for_scope(self, scope: Scope) -> str | None:
        for region in self._regions.list_regions():
            if region.space != scope.space:
                continue
            if region.kind != "session":
                return region.region_id
            if scope.session and scope.session == region.session_id:
                return region.region_id
            if not scope.session and not region.session_id:
                return region.region_id
        return None

    def _graph_region_or_none(self, library_id: str) -> MemoryRegion | None:
        try:
            return self._regions.require_region(library_id)
        except RegionError:
            return None

    @staticmethod
    def _graph_status_scopes(bundle: _RuntimeBundle) -> list[Scope]:
        scopes = list(bundle.kv_store.scopes())
        graph_scopes = getattr(bundle.graph_store, "graph_scopes", None)
        if callable(graph_scopes):
            for scope in graph_scopes():
                if scope not in scopes:
                    scopes.append(scope)
        return scopes

    def _region_scopes(self, bundle: _RuntimeBundle, region: MemoryRegion) -> list[Scope]:
        scopes = []
        for scope in bundle.kv_store.scopes():
            if scope.space != region.space:
                continue
            if region.kind == "session" and scope.session != (region.session_id or ""):
                continue
            scopes.append(scope)
        graph_scopes = getattr(bundle.graph_store, "graph_scopes", None)
        if callable(graph_scopes):
            for scope in graph_scopes():
                if scope.space != region.space:
                    continue
                if region.kind == "session" and scope.session != (region.session_id or ""):
                    continue
                if scope not in scopes:
                    scopes.append(scope)
        base_session = region.session_id if region.kind == "session" else ""
        for agent in _SEARCH_AGENTS:
            expected = _scope_for_region(
                region,
                session_id=base_session,
                agent=agent,
            )
            if expected not in scopes:
                scopes.append(expected)
        return sorted(
            scopes,
            key=lambda scope: (
                scope.space,
                scope.agent,
                scope.session,
                scope.user,
                scope.org,
            ),
        )

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._write, request)

    def _write(self, request: MemoryWriteRequest) -> dict[str, object]:
        if isinstance(request, CreateMemoryRequest):
            if request.region_id is not None:
                region_id = request.region_id.strip()
                if not region_id:
                    raise ValueError("region_id must not be blank")
                self._regions.require_region(region_id)
                normalized = replace(request, region_id=region_id)
                key = (_write_identity(normalized), normalized.idempotency_key)
            else:
                normalized = replace(request, library_id=validate_library_id(request.library_id))
                key = (normalized.library_id, normalized.idempotency_key)
            with self._idempotency_lock(key):
                return self._write_locked(normalized)
        if isinstance(request, CaptureMemoryRequest):
            normalized_capture = replace(
                request,
                session_id=_normalize_session_id(request.session_id),
                topic_id=_normalize_optional_identifier(request.topic_id, "topic_id"),
                occurred_at=_validate_occurred_at(request.occurred_at),
            )
            with self._idempotency_lock(("capture", normalized_capture.event_id)):
                return self._write_locked(normalized_capture)
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
            if request.region_id is not None:
                region = self._regions.require_region(request.region_id)
                scope = _scope_for_region(region)
                identity = _write_identity(request)
            else:
                scope = _scope_for_normalized_library(request.library_id)
                identity = request.library_id
            request_hash = _write_request_hash(request)
            existing_id = self._reserve_idempotency(identity, request.idempotency_key, request_hash)
            if existing_id is not None:
                return _write_response(request, existing_id, status="existing")
            user_metadata = {
                "memory_type": request.memory_type,
                "source_references": list(request.source_references),
                "request_actor": request.actor,
                "idempotency_key": request.idempotency_key,
                "region_id": request.region_id,
            }
        elif isinstance(request, CaptureMemoryRequest):
            if request.deadline is not None and time.monotonic() >= request.deadline:
                raise MemoryKernelUnavailableError("capture persistence deadline expired")
            scope, region = self._capture_scope(request)
            identity = f"capture:{request.event_id}"
            request_hash = _capture_request_hash(request, scope)
            existing_id = self._reserve_capture_idempotency(
                bundle,
                scope,
                request,
                identity,
                request_hash,
            )
            if existing_id is not None:
                return _capture_response(
                    request,
                    existing_id,
                    region,
                    scope,
                    status="accepted",
                    duplicate=True,
                )
            user_metadata = {
                "event_id": request.event_id,
                "capture_event_id": request.event_id,
                "session_id": request.session_id,
                "turn_id": request.turn_id,
                "event_kind": request.event_kind,
                "occurred_at": request.occurred_at,
                "cwd": request.cwd,
                "region_id": region.region_id,
                "region_kind": region.kind,
                "topic_id": request.topic_id,
                "project_id": region.region_id if region.kind == "project" else None,
                "memory_type": "capture",
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
                self._release_pending_idempotency(_write_identity(request), request.idempotency_key)
            elif isinstance(request, CaptureMemoryRequest):
                try:
                    self._purge_capture_units(bundle, scope, request.event_id)
                except Exception as cleanup_error:
                    raise MemoryKernelUnavailableError(
                        "capture cleanup is pending after partial persistence"
                    ) from cleanup_error
                self._release_pending_idempotency(identity, request.event_id)
            raise
        if not units:
            if isinstance(request, CreateMemoryRequest):
                self._release_pending_idempotency(_write_identity(request), request.idempotency_key)
            elif isinstance(request, CaptureMemoryRequest):
                try:
                    self._purge_capture_units(bundle, scope, request.event_id)
                except Exception as cleanup_error:
                    raise MemoryKernelUnavailableError(
                        "capture cleanup is pending after partial persistence"
                    ) from cleanup_error
                self._release_pending_idempotency(identity, request.event_id)
            raise MemoryKernelUnavailableError("mem2 runtime did not create a memory unit")
        unit = units[0]
        if isinstance(request, CreateMemoryRequest):
            try:
                self._complete_idempotency(
                    _write_identity(request),
                    request.idempotency_key,
                    request_hash,
                    unit.id,
                )
            except Exception as error:
                raise MemoryKernelUnavailableError(
                    "memory was created but its idempotency completion is pending"
                ) from error
            return _write_response(request, unit.id, status="created")
        try:
            self._complete_idempotency(
                identity,
                request.event_id,
                request_hash,
                unit.id,
            )
        except Exception as error:
            raise MemoryKernelUnavailableError(
                "capture was stored but its idempotency completion is pending"
            ) from error
        return _capture_response(request, unit.id, region, scope, status="accepted")

    async def search(self, request: SearchMemoryRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._search, request)

    def _search(self, request: SearchMemoryRequest) -> dict[str, object]:
        if request.session_id is not None:
            request = replace(request, session_id=_normalize_session_id(request.session_id))
        if request.topic_id is not None:
            request = replace(
                request,
                topic_id=_normalize_optional_identifier(request.topic_id, "topic_id"),
            )
        bundle = self._require_bundle()
        if request.region_ids is not None:
            return self._search_regions(bundle, request)
        if request.region_id is not None:
            region = self._regions.require_region(request.region_id)
            return self._search_region_session(bundle, request, region, "explicit_region")
        if request.library_id is not None:
            library_id = validate_library_id(request.library_id)
            normalized = replace(request, library_id=library_id)
            return self._search_scope(
                bundle,
                normalized,
                _scope_for_normalized_library(library_id),
            )
        scope, resolved_region = self._scope_for_cwd_with_region(
            request.cwd or "", request.session_id
        )
        if resolved_region is not None:
            return self._search_region_session(bundle, request, resolved_region, "project_cwd")
        if request.topic_id is not None:
            topic_id = request.topic_id.strip()
            if not topic_id:
                raise ValueError("topic_id must not be blank")
            region = self._regions.require_region(topic_id)
            if region.kind != "topic":
                raise ValueError("topic_id must identify a topic memory region")
            return self._search_region_session(bundle, request, region, "topic_session")
        return self._search_non_project(bundle, request, scope)

    def _search_region_session(
        self,
        bundle: _RuntimeBundle,
        request: SearchMemoryRequest,
        region: MemoryRegion,
        scope_kind: str,
    ) -> dict[str, object]:
        targets: list[tuple[Scope, MemoryRegion]] = []
        base_session = region.session_id if region.kind == "session" else ""
        for agent in _SEARCH_AGENTS:
            targets.append(
                (
                    _scope_for_region(
                        region,
                        session_id=base_session,
                        agent=agent,
                    ),
                    region,
                )
            )
            if request.session_id and region.kind != "session":
                targets.append(
                    (
                        _scope_for_region(
                            region,
                            session_id=request.session_id,
                            agent=agent,
                        ),
                        region,
                    )
                )
        response = self._search_across(
            bundle,
            request,
            targets,
            scope_kind=scope_kind,
            region_ids=[region.region_id],
        )
        response["library_id"] = region.region_id
        response["region_id"] = region.region_id
        response["space"] = region.space
        scope = response.get("scope")
        if isinstance(scope, dict):
            scope["space"] = region.space
            scope["library_id"] = region.region_id
            scope["region_id"] = region.region_id
        return response

    def _search_non_project(
        self,
        bundle: _RuntimeBundle,
        request: SearchMemoryRequest,
        synthetic_scope: Scope,
    ) -> dict[str, object]:
        user_region = self._regions.ensure_user_region()
        if request.session_id:
            session_region = self._ensure_session_region(request.session_id)
        else:
            digest = hashlib.sha256(synthetic_scope.space.encode("utf-8")).hexdigest()[:24]
            session_region = MemoryRegion(
                region_id=f"session-{digest}",
                kind="session",
                name="当前会话",
                space="chat",
                session_id="",
                temporary=True,
            )
        targets: list[tuple[Scope, MemoryRegion]] = []
        for agent in _SEARCH_AGENTS:
            targets.append(
                (_scope_for_region(user_region, session_id="", agent=agent), user_region)
            )
            targets.append(
                (
                    _scope_for_region(
                        session_region,
                        session_id=session_region.session_id,
                        agent=agent,
                    ),
                    session_region,
                )
            )
        return self._search_across(
            bundle,
            request,
            targets,
            scope_kind="user_and_session",
            region_ids=[user_region.region_id, session_region.region_id],
        )

    def _search_regions(
        self, bundle: _RuntimeBundle, request: SearchMemoryRequest
    ) -> dict[str, object]:
        region_ids = request.region_ids or ()
        regions = [self._regions.require_region(region_id) for region_id in region_ids]
        targets: list[tuple[Scope, MemoryRegion]] = []
        for region in regions:
            base_session = region.session_id if region.kind == "session" else ""
            for agent in _SEARCH_AGENTS:
                targets.append(
                    (
                        _scope_for_region(
                            region,
                            session_id=base_session,
                            agent=agent,
                        ),
                        region,
                    )
                )
                if request.session_id and region.kind != "session":
                    targets.append(
                        (
                            _scope_for_region(
                                region,
                                session_id=request.session_id,
                                agent=agent,
                            ),
                            region,
                        )
                    )
        return self._search_across(
            bundle,
            request,
            targets,
            scope_kind="explicit_regions",
            region_ids=list(region_ids),
        )

    def _search_across(
        self,
        bundle: _RuntimeBundle,
        request: SearchMemoryRequest,
        targets: list[tuple[Scope, MemoryRegion]],
        *,
        scope_kind: str,
        region_ids: list[str],
    ) -> dict[str, object]:
        merged: dict[str, dict[str, object]] = {}
        degradation: list[str] = []
        vector_statuses: set[str] = set()
        graph_statuses: set[str] = set()
        for scope, region in targets:
            scoped_request = replace(
                request,
                cwd=None,
                library_id=None,
                region_id=region.region_id,
                region_ids=None,
                topic_id=None,
            )
            response = self._search_scope(bundle, scoped_request, scope, region=region)
            raw_degradation = response.get("degradation", [])
            if isinstance(raw_degradation, list):
                for reason in raw_degradation:
                    reason_text = str(reason)
                    if reason_text not in degradation:
                        degradation.append(reason_text)
            vector_statuses.add(str(response.get("vector_index_status", "degraded")))
            graph_statuses.add(str(response.get("graph_index_status", "degraded")))
            raw_results = response.get("results", [])
            if not isinstance(raw_results, list):
                continue
            for raw_result in raw_results:
                if not isinstance(raw_result, dict):
                    continue
                item = {str(key): value for key, value in raw_result.items()}
                item["region_id"] = region.region_id
                item["space"] = region.space
                item["library_id"] = region.region_id
                item_id = str(item.get("document_id", ""))
                previous = merged.get(item_id)
                item_score = float(str(item.get("score", 0.0)))
                previous_score = 0.0 if previous is None else float(str(previous.get("score", 0.0)))
                if previous is None or item_score > previous_score:
                    merged[item_id] = item
        ranked = sorted(
            merged.values(),
            key=lambda item: float(str(item.get("score", 0.0))),
            reverse=True,
        )
        results = _truncate_search_results(
            ranked,
            max(1, min(request.limit, 100)),
        )
        if vector_statuses == {"ready"}:
            vector_status = "ready"
        elif "degraded" in vector_statuses:
            vector_status = "degraded"
        else:
            vector_status = "not_configured"
        if graph_statuses == {"ready"}:
            graph_status = "ready"
        elif "degraded" in graph_statuses:
            graph_status = "degraded"
        else:
            graph_status = "not_configured"
        return {
            "status": "bound",
            "scope": {
                "kind": scope_kind,
                "cwd": request.cwd,
                "session_id": request.session_id,
                "topic_id": request.topic_id,
                "region_ids": region_ids,
                "spaces": list(dict.fromkeys(region.space for _, region in targets)),
            },
            "region_ids": region_ids,
            "query": request.query,
            "degraded": bool(degradation),
            "degradation": degradation,
            "vector_index_status": vector_status,
            "graph_index_status": graph_status,
            "results": results,
        }

    def _search_scope(
        self,
        bundle: _RuntimeBundle,
        request: SearchMemoryRequest,
        scope: Scope,
        *,
        region: MemoryRegion | None = None,
    ) -> dict[str, object]:
        security = _security_for(scope)
        degradation: list[str] = []
        embedding_configured = self._vector_enabled
        if not embedding_configured:
            degradation.append(self._vector_degradation)
            items, failed = self._keyword_fallback(bundle, scope, request)
            if failed:
                degradation.append("fulltext_unavailable")
            items = self._append_graph_expansions(
                bundle, scope, items, request, degradation, region
            )
            graph_status = self._graph_ready_status(bundle, scope, region)
            return _search_response(
                request,
                scope,
                items,
                [],
                degradation,
                vector_status="not_configured",
                graph_status=graph_status,
                region=region,
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
            units = self._append_graph_expansions(
                bundle, scope, units, request, degradation, region
            )
            graph_status = self._graph_ready_status(bundle, scope, region)
            return _search_response(
                request,
                scope,
                units,
                [],
                degradation,
                vector_status="degraded",
                graph_status=graph_status,
                region=region,
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
        items = self._append_graph_expansions(bundle, scope, items, request, degradation, region)
        graph_status = self._graph_ready_status(bundle, scope, region)
        return _search_response(
            request,
            scope,
            items,
            reported_errors,
            degradation,
            vector_status=vector_status,
            graph_status=graph_status,
            region=region,
        )

    def _graph_ready_status(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        region: MemoryRegion | None,
    ) -> str:
        if bundle.graph_store is None:
            return "not_configured"
        with self._graph_write_lock:
            return self._graph_ready_status_locked(bundle, scope, region)

    def _graph_ready_status_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        region: MemoryRegion | None,
    ) -> str:
        graph_store = bundle.graph_store
        assert graph_store is not None
        transient_error = self._graph_error_for(scope=scope)
        if transient_error is None and region is not None:
            transient_error = self._graph_error_for(region_id=region.region_id)
        if transient_error is not None:
            return "degraded"
        projection_error = self._graph_projection_error(scope=scope)
        if projection_error is None and region is not None:
            projection_error = self._graph_projection_error(region_id=region.region_id)
        if (
            projection_error is None
            and region is None
            and scope.space.startswith(LIBRARY_SCOPE_PREFIX)
        ):
            projection_error = self._graph_projection_error(
                region_id=scope.space.removeprefix(LIBRARY_SCOPE_PREFIX)
            )
        if projection_error is not None:
            return "degraded"
        if self._graph_store_error(bundle, scope):
            return "degraded"
        try:
            graph_store.health()
            _, _, mismatch = self._graph_projection_snapshot(bundle, scope)
            if mismatch is not None:
                return "degraded"
        except Exception:
            return "degraded"
        return "ready"

    def _append_graph_expansions(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        items: list[Any],
        request: SearchMemoryRequest,
        degradation: list[str],
        region: MemoryRegion | None,
    ) -> list[Any]:
        graph_store = bundle.graph_store
        if graph_store is None:
            return list(items)
        with self._graph_write_lock:
            return self._append_graph_expansions_locked(
                bundle,
                scope,
                items,
                request,
                degradation,
                region,
            )

    def _append_graph_expansions_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        items: list[Any],
        request: SearchMemoryRequest,
        degradation: list[str],
        region: MemoryRegion | None,
    ) -> list[Any]:
        graph_store = bundle.graph_store
        if graph_store is None:
            return list(items)
        projection_error = self._graph_projection_error(scope=scope)
        if projection_error is None and region is not None:
            projection_error = self._graph_projection_error(region_id=region.region_id)
        if (
            projection_error is None
            and region is None
            and scope.space.startswith(LIBRARY_SCOPE_PREFIX)
        ):
            projection_error = self._graph_projection_error(
                region_id=scope.space.removeprefix(LIBRARY_SCOPE_PREFIX)
            )
        if projection_error is not None:
            self._set_graph_error(
                projection_error,
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            if "graph_unavailable" not in degradation:
                degradation.append("graph_unavailable")
            return list(items)
        store_error = self._graph_store_error(bundle, scope)
        if store_error is not None:
            self._set_graph_error(
                store_error,
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            if "graph_unavailable" not in degradation:
                degradation.append("graph_unavailable")
            return list(items)
        _, _, projection_mismatch = self._graph_projection_snapshot(bundle, scope)
        if projection_mismatch is not None:
            self._set_graph_error(
                projection_mismatch,
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            if "graph_unavailable" not in degradation:
                degradation.append("graph_unavailable")
            return list(items)
        try:
            graph_store.health()
            if not items:
                return list(items)
            direct_ids = {
                str(item.id if isinstance(item, MemoryUnit) else item.unit_id)
                for item in items
                if isinstance(item, MemoryUnit) or hasattr(item, "unit_id")
            }
            if not direct_ids:
                return list(items)
            seeds = graph_store.get(scope, sorted(direct_ids))
            if not seeds:
                return list(items)
            depth = max(1, min(request.max_graph_hops, 2))
            limit = max(1, min(request.limit, 100))
            traversed = graph_store.traverse_many(
                scope,
                [seed.id for seed in seeds],
                depth=depth,
                limit=limit,
            )
            units_by_id = {
                unit.id: unit
                for unit in load_units(
                    bundle.kv_store,
                    scope,
                    [node.id for node, _, _ in traversed],
                )
            }
            expanded: list[dict[str, object]] = []
            for node, hop, seed_id in traversed:
                unit = units_by_id.get(node.id)
                if unit is None:
                    continue
                if unit.lifecycle.value != "active":
                    continue
                if unit.user_metadata.get("restore_pending") is True:
                    continue
                payload = _result_item(unit, request)
                payload["classification"] = "graph_expansion"
                payload["graph_hop"] = hop
                payload["graph_object_type"] = "node"
                payload["graph_source_id"] = seed_id
                payload["retrieval_sources"] = ["graph"]
                expanded.append(payload)
            self._clear_graph_error(
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            return [*items, *expanded]
        except Exception as error:
            self._set_graph_error(
                str(error),
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            if "graph_unavailable" not in degradation:
                degradation.append("graph_unavailable")
            return list(items)

    def _scope_for_cwd(self, cwd: str, session_id: str | None) -> Scope:
        scope, _ = self._scope_for_cwd_with_region(cwd, session_id)
        return scope

    def _scope_for_cwd_with_region(
        self, cwd: str, session_id: str | None
    ) -> tuple[Scope, MemoryRegion | None]:
        try:
            resolved = self._regions.resolve(cwd)
        except RegionBindingError:
            resolved = None
        if resolved is None:
            return _scope_for_synthetic_cwd(cwd, session_id), None
        return _scope_for_region(resolved.region, session_id=session_id), resolved.region

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

    def list_documents(self, request: ListDocumentsRequest) -> list[dict[str, object]]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        payloads = {
            unit.id: _unit_payload(unit, region)
            for _, unit in self._region_unit_pairs(bundle, region)
        }
        return [payloads[key] for key in sorted(payloads)]

    def inspect(self, request: object) -> dict[str, object]:
        if not isinstance(request, DocumentInspectRequest):
            self._raise()
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        scope, unit = self._find_region_unit(bundle, region, request.path)
        payload = _unit_payload(unit, region)
        pairs = self._region_unit_pairs(bundle, region)
        history = [_unit_payload(item, region) for _, item in _version_chain(unit, pairs)]
        payload["history"] = history
        payload["versions"] = history
        try:
            trace = bundle.runtime.api.trace(
                unit.id,
                scope,
                security=_security_for(scope),
            )
        except Exception:
            trace = []
        trace = [item for item in trace if item.id != unit.id]
        if not trace:
            references = unit.user_metadata.get("provenance")
            if not isinstance(references, list):
                references = unit.user_metadata.get("source_unit_ids")
            if isinstance(references, list):
                by_id = {item.id: item for _, item in pairs}
                trace = [
                    by_id[reference]
                    for reference in (str(value) for value in references)
                    if reference in by_id
                ]
        payload["provenance_chain"] = [_unit_payload(item, region) for item in trace]
        return payload

    async def update(self, request: MemoryUpdateRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._update_document_serialized, request)

    def _update_document_serialized(self, request: MemoryUpdateRequest) -> dict[str, object]:
        if not isinstance(request, DocumentUpdateRequest):
            return self._update_document(request)
        with self._governance_region_lock(request.library_id):
            return self._update_document(request)

    def _update_document(self, request: MemoryUpdateRequest) -> dict[str, object]:
        if not isinstance(request, DocumentUpdateRequest):
            self._raise()
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        scope, unit = self._find_region_unit(bundle, region, request.path)
        _require_revision(unit, request.expected_revision)
        _require_active(unit, request.mode)
        mode = UpdateMode.OVERWRITE if request.mode == "overwrite" else UpdateMode.SUPERSEDE
        event_id = self._begin_governance_event(
            region.region_id,
            unit.id,
            request.mode,
            request.actor,
            request.source,
            request.operation_id,
            before_content=unit.content,
            after_content=request.content,
            detail={"previous_unit_id": unit.id},
        )
        try:
            updated = bundle.runtime.api.update(
                unit.id,
                scope,
                MemoryPatch(
                    content=request.content,
                    mode=mode,
                    user_metadata={
                        "governance_actor": request.actor,
                        "governance_source": request.source,
                        "operation_id": request.operation_id,
                        "previous_revision": request.expected_revision,
                    },
                ),
                security=_security_for(scope),
            )
        except Exception:
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=unit.id,
                after_content="",
                detail={"previous_unit_id": unit.id},
            )
            raise
        if bundle.graph_store is not None:
            self._reproject_graph_scope(
                bundle,
                scope,
                region_id=region.region_id,
            )
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=updated.id,
            after_content=updated.content,
            detail={"previous_unit_id": unit.id, "result_unit_id": updated.id},
        )
        return _unit_payload(updated, region)

    def preview_document_update(self, request: PreviewDocumentUpdateRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        _, unit = self._find_region_unit(bundle, region, request.path)
        _require_revision(unit, request.expected_revision)
        _require_active(unit, "update preview")
        return {
            "path": unit.id,
            "id": unit.id,
            "current_content": unit.content,
            "proposed_content": request.content,
            "source_version": _unit_revision(unit),
            "preview_token": _unit_revision(unit),
            "lifecycle": unit.lifecycle.value,
            "provenance": list(unit.provenance),
            "diff": _content_diff(unit.content, request.content),
        }

    async def preview_document_delete(
        self, request: PreviewDocumentDeleteRequest
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._preview_document_delete, request)

    def _preview_document_delete(self, request: PreviewDocumentDeleteRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        _, unit = self._find_region_unit(bundle, region, request.path)
        _require_revision(unit, request.expected_revision)
        revision = _unit_revision(unit)
        return {
            "path": unit.id,
            "id": unit.id,
            "content": unit.content,
            "expected_source_version": revision,
            "preview_token": revision,
            "lifecycle": unit.lifecycle.value,
            "affected_units": 1,
            "provenance": list(unit.provenance),
        }

    async def delete(self, request: DeleteMemoryRequest) -> dict[str, str]:
        return await asyncio.to_thread(self._delete_document_serialized, request)

    def _delete_document_serialized(self, request: DeleteMemoryRequest) -> dict[str, str]:
        with self._governance_region_lock(request.library_id):
            return self._delete_document(request)

    def _delete_document(self, request: DeleteMemoryRequest) -> dict[str, str]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        scope, unit = self._find_region_unit(bundle, region, request.path)
        revision = _unit_revision(unit)
        _require_revision(unit, request.expected_revision)
        if request.preview_token != revision:
            raise MemoryRevisionConflictError("deletion preview is stale or invalid")
        _require_delete_lifecycle(unit, request.mode)
        event_id = self._begin_governance_event(
            region.region_id,
            unit.id,
            f"delete:{request.mode}",
            request.actor,
            request.source,
            request.operation_id,
            before_content=unit.content,
            after_content="",
            detail={"mode": request.mode},
        )
        try:
            deleted = bundle.runtime.api.delete(
                DeleteSelector(
                    unit_ids=[unit.id],
                    scope=scope,
                    mode=DeleteMode(request.mode),
                ),
                security=_security_for(scope),
            )
        except Exception:
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=unit.id,
                after_content="",
                detail={"mode": request.mode},
            )
            raise
        if deleted and request.mode != "downweight":
            deleted_ids = (
                [str(item) for item in deleted] if isinstance(deleted, list) else [unit.id]
            )
            self._remove_graph_nodes(
                bundle,
                scope,
                list(dict.fromkeys(deleted_ids)),
                region.region_id,
            )
        detail: dict[str, object] = {
            "deleted": bool(deleted),
            "mode": request.mode,
        }
        if request.mode == "forget" and deleted:
            try:
                refreshed = load_units(bundle.kv_store, scope, [unit.id])
                if refreshed:
                    forgotten = refreshed[0]
                    forgotten.temporal.t_invalid = datetime.now(UTC)
                    bundle.kv_store.update(scope, memory_key(forgotten.id), dumps(forgotten))
            except Exception:
                # The deletion already succeeded. Keep it successful and leave
                # a durable audit marker instead of returning a false 500.
                detail["deleted_at_persist_failed"] = True
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=unit.id,
            after_content="",
            detail=detail,
        )
        return {
            "id": unit.id,
            "path": unit.id,
            "status": request.mode,
            "deleted": "true" if deleted else "false",
        }

    def list_forgotten(self, request: ListForgottenRequest) -> list[dict[str, object]]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        pairs = self._region_unit_pairs(bundle, region)
        forgotten: list[dict[str, object]] = []
        for _, unit in pairs:
            if unit.lifecycle.value != "forgotten":
                continue
            if unit.user_metadata.get("restore_pending") is True:
                continue
            chain = _version_chain(unit, pairs)
            if any(item.lifecycle.value == "active" for _, item in chain):
                continue
            latest_forgotten = [item for _, item in chain if item.lifecycle.value == "forgotten"][
                -1
            ]
            if unit.id != latest_forgotten.id:
                continue
            payload = _unit_payload(unit, region)
            payload["history"] = [_unit_payload(item, region) for _, item in chain]
            forgotten.append(payload)
        return forgotten

    async def restore_forgotten(self, request: RestoreForgottenRequest) -> dict[str, str]:
        return await asyncio.to_thread(self._restore_forgotten_serialized, request)

    def _restore_forgotten_serialized(self, request: RestoreForgottenRequest) -> dict[str, str]:
        with self._governance_region_lock(request.library_id):
            return self._restore_forgotten(request)

    def _restore_forgotten(self, request: RestoreForgottenRequest) -> dict[str, str]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        pairs = self._region_unit_pairs(bundle, region)
        _, unit = self._find_region_unit(bundle, region, request.tombstone_id)
        if unit.lifecycle.value != "forgotten":
            raise MemoryGovernanceError("memory is not forgotten")
        chain = _version_chain(unit, pairs)
        chain_ids = {item.id for _, item in chain}
        forgotten_versions = [item for _, item in chain if item.lifecycle.value == "forgotten"]
        latest_forgotten = forgotten_versions[-1]
        if unit.id != latest_forgotten.id:
            raise MemoryLifecycleConflictError("restore must use the latest forgotten version")
        old_scope, old = self._find_region_unit(bundle, region, request.commit)
        if old.id not in chain_ids:
            raise MemoryRecordNotFoundError("memory version not found in document history")
        active_versions = [
            item
            for _, item in pairs
            if item.lifecycle.value == "active"
            and (
                item.id in chain_ids
                or item.supersedes in chain_ids
                or str(item.user_metadata.get("restored_from", "")) in chain_ids
                or any(str(reference) in chain_ids for reference in item.provenance)
            )
        ]
        if active_versions:
            raise MemoryLifecycleConflictError("forgotten memory already has an active version")
        event_id = self._begin_governance_event(
            region.region_id,
            latest_forgotten.id,
            "restore:forgotten",
            request.actor,
            request.source,
            request.operation_id,
            before_content=unit.content,
            after_content=old.content,
            detail={"restored_from": old.id, "previous_unit_id": latest_forgotten.id},
        )
        try:
            restored = bundle.runtime.api.add(
                old.content,
                old_scope,
                Modality.TEXT,
                security=_security_for(old_scope),
                user_metadata={
                    **dict(old.user_metadata),
                    "operation_id": request.operation_id,
                    "governance_actor": request.actor,
                    "governance_source": request.source,
                    "restored_from": old.id,
                    "provenance": [old.id],
                    "restore_pending": True,
                },
            )
        except Exception:
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=latest_forgotten.id,
                after_content="",
                detail={
                    "restored_from": old.id,
                    "previous_unit_id": latest_forgotten.id,
                },
            )
            raise
        if not restored:
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=latest_forgotten.id,
                after_content="",
                detail={
                    "restored_from": old.id,
                    "previous_unit_id": latest_forgotten.id,
                },
            )
            raise MemoryKernelUnavailableError("forgotten memory restore created no unit")
        restored_unit = restored[0]
        restored_unit.supersedes = latest_forgotten.id
        restored_unit.provenance = [old.id]
        restored_unit.user_metadata.pop("restore_pending", None)
        try:
            bundle.kv_store.update(old_scope, memory_key(restored_unit.id), dumps(restored_unit))
        except Exception:
            try:
                bundle.runtime.api.delete(
                    DeleteSelector(
                        unit_ids=[restored_unit.id],
                        scope=old_scope,
                        mode=DeleteMode.PURGE,
                    ),
                    security=_security_for(old_scope),
                )
            except Exception as purge_error:
                try:
                    bundle.runtime.api.delete(
                        DeleteSelector(
                            unit_ids=[restored_unit.id],
                            scope=old_scope,
                            mode=DeleteMode.FORGET,
                        ),
                        security=_security_for(old_scope),
                    )
                except Exception as forget_error:
                    self._finish_governance_event(
                        event_id,
                        status="pending",
                        unit_id=restored_unit.id,
                        after_content=restored_unit.content,
                        detail={
                            "restored_from": old.id,
                            "previous_unit_id": latest_forgotten.id,
                            "version_link_pending": True,
                            "rollback_error": f"purge={purge_error}; forget={forget_error}",
                        },
                    )
                    raise MemoryKernelUnavailableError(
                        "forgotten memory restore failed and rollback is pending"
                    ) from None
                self._finish_governance_event(
                    event_id,
                    status="failed",
                    unit_id=latest_forgotten.id,
                    after_content="",
                    detail={
                        "restored_from": old.id,
                        "previous_unit_id": latest_forgotten.id,
                        "rollback": "forgotten",
                    },
                )
                raise MemoryKernelUnavailableError(
                    "forgotten memory restore failed; orphan version was forgotten"
                ) from None
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=latest_forgotten.id,
                after_content="",
                detail={
                    "restored_from": old.id,
                    "previous_unit_id": latest_forgotten.id,
                    "rollback": "purged",
                },
            )
            raise MemoryKernelUnavailableError(
                "forgotten memory restore could not persist its version link"
            ) from None
        if bundle.graph_store is not None:
            self._reproject_graph_scope(
                bundle,
                old_scope,
                region_id=region.region_id,
            )
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=restored_unit.id,
            after_content=restored_unit.content,
            detail={
                "restored_from": old.id,
                "previous_unit_id": latest_forgotten.id,
            },
        )
        payload = _unit_payload(restored_unit, region)
        return {key: str(value) for key, value in payload.items()}

    def history_diff(self, request: HistoryDiffRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        _, unit = self._find_region_unit(bundle, region, request.commit)
        pairs = self._region_unit_pairs(bundle, region)
        chain = [item for _, item in _version_chain(unit, pairs)]
        active = [item for item in chain if item.lifecycle.value == "active"]
        current = active[-1] if active else chain[-1]
        return {
            "library_id": region.region_id,
            "commit": unit.id,
            "current": current.id,
            "before": unit.content,
            "after": current.content,
            "diff": _content_diff(unit.content, current.content),
        }

    async def restore_document(self, request: RestoreDocumentRequest) -> dict[str, str]:
        return await asyncio.to_thread(self._restore_document_serialized, request)

    def _restore_document_serialized(self, request: RestoreDocumentRequest) -> dict[str, str]:
        with self._governance_region_lock(request.library_id):
            return self._restore_document(request)

    def _governance_region_lock(self, library_id: str) -> threading.RLock:
        region_id = validate_library_id(library_id)
        with self._governance_locks_guard:
            return self._governance_locks.setdefault(region_id, threading.RLock())

    def _restore_document(self, request: RestoreDocumentRequest) -> dict[str, str]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        scope, current = self._find_region_unit(bundle, region, request.path)
        _, old = self._find_region_unit(bundle, region, request.commit)
        pairs = self._region_unit_pairs(bundle, region)
        chain_ids = {item.id for _, item in _version_chain(current, pairs)}
        if old.id not in chain_ids:
            raise MemoryRecordNotFoundError("memory version not found in document history")
        _require_revision(current, request.expected_revision)
        _require_active(current, "history restore")
        event_id = self._begin_governance_event(
            region.region_id,
            current.id,
            "restore",
            request.actor,
            request.source,
            request.operation_id,
            before_content=current.content,
            after_content=old.content,
            detail={"restored_from": old.id, "previous_unit_id": current.id},
        )
        try:
            restored = bundle.runtime.api.update(
                current.id,
                scope,
                MemoryPatch(
                    content=old.content,
                    mode=UpdateMode.SUPERSEDE,
                    user_metadata={
                        "operation_id": request.operation_id,
                        "governance_actor": request.actor,
                        "governance_source": request.source,
                        "restored_from": old.id,
                    },
                ),
                security=_security_for(scope),
            )
        except Exception:
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=current.id,
                after_content="",
                detail={"restored_from": old.id, "previous_unit_id": current.id},
            )
            raise
        if bundle.graph_store is not None:
            self._reproject_graph_scope(
                bundle,
                scope,
                region_id=region.region_id,
            )
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=restored.id,
            after_content=restored.content,
            detail={"restored_from": old.id, "previous_unit_id": current.id},
        )
        payload = _unit_payload(restored, region)
        return {key: str(value) for key, value in payload.items()}

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]:
        bundle = self._require_bundle()
        region: MemoryRegion | None
        try:
            region = self._governance_region(request.library_id)
        except MemoryRecordNotFoundError:
            region = None
        if region is None:
            library_id = validate_library_id(request.library_id)
            scope = _scope_for_normalized_library(library_id)
            target_space = scope.space
        else:
            scope = _scope_for_region(region)
            target_space = region.space
        events = bundle.runtime.api.audit(
            {"target_space": target_space},
            security=_security_for(scope),
            limit=max(1, min(request.limit, 1000)),
        )
        payloads = [
            {
                "id": event.id,
                "action": event.action,
                "decision": event.decision,
                "actor": event.actor.user,
                "target": event.target.space,
                "target_id": event.target_id,
                "commit": event.target_id or event.id,
                "subject": event.action,
                "authored_at": (
                    event.occurred_at.isoformat() if event.occurred_at is not None else ""
                ),
                "actor_type": event.actor.user,
                "source": "mem2",
                "occurred_at": (
                    event.occurred_at.isoformat() if event.occurred_at is not None else ""
                ),
            }
            for event in events
        ]
        if region is not None:
            payloads.extend(self._governance_audit_events(region.region_id, request.limit))
        if region is None and not payloads:
            raise MemoryRecordNotFoundError("memory region not found")
        return sorted(payloads, key=_audit_sort_key, reverse=True)[
            : max(1, min(request.limit, 1000))
        ]

    def _begin_governance_event(
        self,
        region_id: str,
        unit_id: str,
        action: str,
        actor: str,
        source: str,
        operation_id: str,
        *,
        before_content: str,
        after_content: str,
        detail: dict[str, object],
    ) -> int:
        with closing(self._maintenance_db()) as connection, connection:
            cursor = connection.execute(
                """INSERT INTO governance_events
                   (region_id, unit_id, action, actor, source, operation_id,
                    before_content, after_content, detail, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    region_id,
                    unit_id,
                    action,
                    actor,
                    source,
                    operation_id,
                    before_content,
                    after_content,
                    json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    datetime.now(UTC).isoformat(),
                ),
            )
            if cursor.lastrowid is None:
                raise MemoryKernelUnavailableError("governance audit event was not persisted")
            return cursor.lastrowid

    def _finish_governance_event(
        self,
        event_id: int,
        *,
        status: str,
        unit_id: str,
        after_content: str,
        detail: dict[str, object],
    ) -> None:
        try:
            with closing(self._maintenance_db()) as connection, connection:
                connection.execute(
                    """UPDATE governance_events
                       SET status = ?, unit_id = ?, after_content = ?, detail = ?
                       WHERE id = ?""",
                    (
                        status,
                        unit_id,
                        after_content,
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                        event_id,
                    ),
                )
        except (sqlite3.Error, OSError):
            # The pending row already preserves before_content. A completion
            # failure must not turn a committed mutation into a false 500.
            return

    def _governance_audit_events(self, region_id: str, limit: int) -> list[dict[str, str]]:
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                """SELECT id, unit_id, action, actor, source, operation_id,
                          before_content, after_content, detail, status, created_at
                   FROM governance_events
                   WHERE region_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (region_id, max(1, min(limit, 1000))),
            ).fetchall()
        payloads: list[dict[str, str]] = []
        for row in rows:
            occurred = _parse_timestamp(str(row[10]))
            occurred_at = occurred.isoformat() if occurred is not None else str(row[10])
            payloads.append(
                {
                    "id": f"governance-{row[0]}",
                    "action": str(row[2]),
                    "decision": "allow",
                    "actor": str(row[3]),
                    "target": region_id,
                    "target_id": str(row[1]),
                    "source": str(row[4]),
                    "operation_id": str(row[5]),
                    "before_content": str(row[6]),
                    "after_content": str(row[7]),
                    "detail": str(row[8]),
                    "status": str(row[9]),
                    "commit": str(row[1]),
                    "subject": str(row[2]),
                    "authored_at": occurred_at,
                    "actor_type": str(row[3]),
                    "occurred_at": occurred_at,
                }
            )
        return payloads

    def _governance_region(self, library_id: str) -> MemoryRegion:
        identifier = validate_library_id(library_id)
        try:
            return self._regions.require_region(identifier)
        except RegionError as error:
            raise MemoryRecordNotFoundError(str(error)) from error

    def _region_unit_pairs(
        self, bundle: _RuntimeBundle, region: MemoryRegion
    ) -> list[tuple[Scope, MemoryUnit]]:
        pairs: list[tuple[Scope, MemoryUnit]] = []
        for scope in bundle.kv_store.scopes():
            if scope.space != region.space:
                continue
            if region.kind == "session" and scope.session != (region.session_id or ""):
                continue
            pairs.extend((scope, unit) for unit in self._units_in_scope(bundle, scope))
        return sorted(pairs, key=_unit_sort_key)

    def _find_region_unit(
        self, bundle: _RuntimeBundle, region: MemoryRegion, unit_id: str
    ) -> tuple[Scope, MemoryUnit]:
        normalized = unit_id.strip()
        if not normalized:
            raise MemoryRecordNotFoundError("memory record not found")
        for scope, unit in self._region_unit_pairs(bundle, region):
            if unit.id == normalized:
                return scope, unit
        raise MemoryRecordNotFoundError("memory record not found")

    async def evolve(self, request: EvolveMemoryRequest) -> dict[str, int]:
        return await asyncio.to_thread(self._evolve, request)

    def _capture_scope(self, request: CaptureMemoryRequest) -> tuple[Scope, MemoryRegion]:
        topic_id = request.topic_id.strip() if request.topic_id is not None else ""
        if request.topic_id is not None and not topic_id:
            raise ValueError("topic_id must not be blank")
        try:
            resolved = self._regions.resolve(request.cwd)
        except RegionBindingError:
            resolved = None
        if resolved is not None:
            region = resolved.region
        elif topic_id:
            region = self._regions.require_region(topic_id)
            if region.kind != "topic":
                raise CaptureContractError("topic_id must identify a topic memory region")
        else:
            region = self._ensure_session_region(request.session_id)
        return (
            _scope_for_region(
                region,
                session_id=request.session_id,
                agent=_CODEX_AGENT,
            ),
            region,
        )

    def _ensure_session_region(self, session_id: str) -> MemoryRegion:
        normalized = _normalize_session_id(session_id)
        for region in self._regions.list_regions():
            if region.kind == "session" and region.session_id == normalized:
                return region
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
        try:
            return self._regions.create_region(
                "session",
                f"会话 {normalized[:80]}",
                region_id=f"session-{digest}",
                session_id=normalized,
            )
        except RegionError:
            for region in self._regions.list_regions():
                if region.kind == "session" and region.session_id == normalized:
                    return region
            raise

    def _evolve(self, request: EvolveMemoryRequest) -> dict[str, int]:
        with self._maintenance_lock:
            return self._evolve_locked(request)

    def _evolve_locked(self, request: EvolveMemoryRequest) -> dict[str, int]:
        bundle = self._require_bundle()
        session_id = (
            _normalize_session_id(request.session_id) if request.session_id is not None else None
        )
        created = 0
        all_scopes = [
            scope
            for scope in bundle.kv_store.scopes()
            if session_id is None or scope.session == session_id
        ]
        scopes = [scope for scope in all_scopes if scope.space not in {"user"}]
        target_scopes: list[Scope] = []
        for scope in scopes:
            units = self._units_in_scope(bundle, scope)
            captured = [
                unit
                for unit in units
                if unit.lifecycle.value == "active" and _is_capture_unit(unit)
            ]
            groups: dict[str, list[MemoryUnit]] = {}
            for unit in captured:
                turn_id = _metadata_text(unit, "turn_id")
                if turn_id:
                    groups.setdefault(turn_id, []).append(unit)
            for turn_id, source_units in groups.items():
                if request.deadline is not None and time.monotonic() >= request.deadline:
                    break
                ordered = sorted(
                    source_units,
                    key=lambda unit: (
                        _event_order(_metadata_text(unit, "event_kind")),
                        _metadata_text(unit, "occurred_at"),
                        unit.id,
                    ),
                )
                source_ids = [unit.id for unit in ordered]
                region_id = _metadata_text(ordered[0], "region_id")
                if not region_id:
                    continue
                try:
                    region = self._regions.require_region(region_id)
                except RegionError:
                    continue
                if region.space != scope.space:
                    continue
                evolve_key = _stable_evolve_key(scope, region.region_id, turn_id)
                source_fingerprint = _source_fingerprint(source_ids)
                target_scope = _scope_for_region(
                    region,
                    session_id=(region.session_id or "" if region.kind == "session" else ""),
                    agent=scope.agent,
                )
                if target_scope not in target_scopes:
                    target_scopes.append(target_scope)
                status, memory_id = self._reserve_evolution(
                    evolve_key,
                    scope.session,
                    turn_id,
                    region.region_id,
                    source_fingerprint,
                )
                if status == "completed" and memory_id is not None:
                    continue
                content = "\n\n".join(
                    f"{_event_label(_metadata_text(unit, 'event_kind'))}: {unit.content}"
                    for unit in ordered
                )
                chunks = _chunk_evolved_content(content)
                residual = self._evolution_units(bundle, target_scope, evolve_key)
                if status == "pending" and self._evolution_visible(
                    bundle,
                    target_scope,
                    residual,
                    len(chunks),
                    source_fingerprint,
                ):
                    self._complete_evolution(evolve_key, residual[0].id)
                    continue
                if residual:
                    self._purge_evolution_units(bundle, target_scope, residual)
                added_units: list[MemoryUnit] = []
                try:
                    for part, chunk in enumerate(chunks):
                        units_added = bundle.runtime.api.add(
                            chunk,
                            target_scope,
                            Modality.TEXT,
                            security=_security_for(target_scope),
                            user_metadata={
                                "memory_type": "evolved",
                                "evolve_key": evolve_key,
                                "source_fingerprint": source_fingerprint,
                                "evolve_part": part,
                                "evolve_parts": len(chunks),
                                "region_id": region.region_id,
                                "region_kind": region.kind,
                                "session_id": scope.session,
                                "turn_id": turn_id,
                                "source_unit_ids": source_ids,
                                "source_event_ids": [
                                    _metadata_text(unit, "event_id") for unit in ordered
                                ],
                                "provenance": source_ids,
                            },
                        )
                        if not units_added:
                            raise MemoryKernelUnavailableError(
                                "evolve did not create a memory unit"
                            )
                        added_units.append(units_added[0])
                    if not self._evolution_visible(
                        bundle,
                        target_scope,
                        added_units,
                        len(chunks),
                        source_fingerprint,
                    ):
                        self._purge_evolution_units(bundle, target_scope, added_units)
                        raise MemoryKernelUnavailableError(
                            "evolved memory is not visible in the fulltext index"
                        )
                except Exception:
                    residual = self._evolution_units(bundle, target_scope, evolve_key)
                    self._purge_evolution_units(bundle, target_scope, residual)
                    raise
                self._complete_evolution(evolve_key, added_units[0].id)
                created += 1
        if bundle.graph_store is not None:
            with self._graph_write_lock:
                graph_scopes = list(all_scopes)
                for scope in bundle.kv_store.scopes():
                    if session_id is not None and scope.session != session_id:
                        continue
                    if scope not in graph_scopes:
                        graph_scopes.append(scope)
                for scope in target_scopes:
                    if scope not in graph_scopes:
                        graph_scopes.append(scope)
                for scope in graph_scopes:
                    if request.deadline is not None and time.monotonic() >= request.deadline:
                        break
                    self._associate_graph_scope(
                        bundle,
                        scope,
                        region_id=self._region_id_for_scope(scope),
                    )
        return {"queued": created}

    def get_retention_policy(self, request: RetentionPolicyRequest) -> dict[str, object]:
        region = self._session_region(request.library_id)
        days = self._retention_days(region.region_id)
        return self._retention_payload(region, days)

    async def update_retention_policy(
        self, request: UpdateRetentionPolicyRequest
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._update_retention_policy, request)

    def _update_retention_policy(self, request: UpdateRetentionPolicyRequest) -> dict[str, object]:
        region = self._session_region(request.library_id)
        if request.inbox_days < 0:
            raise ValueError("retention days cannot be negative")
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """INSERT INTO session_retention_policy (region_id, retention_days)
                   VALUES (?, ?)
                   ON CONFLICT(region_id) DO UPDATE SET
                     retention_days = excluded.retention_days,
                     updated_at = CURRENT_TIMESTAMP""",
                (region.region_id, request.inbox_days),
            )
        return self._retention_payload(region, request.inbox_days)

    async def preview_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._preview_retention_cleanup, request)

    def _preview_retention_cleanup(self, request: RetentionCleanupRequest) -> dict[str, object]:
        region = self._session_region(request.library_id)
        plan = self._retention_plan(region)
        return {
            "library_id": region.region_id,
            "region_id": region.region_id,
            "as_of": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "policy": self._retention_payload(region, self._retention_days(region.region_id)),
            "counts": {
                "captures_to_archive": len(plan),
                "captures_archived": 0,
                "evolved_units_protected": self._evolved_count(region),
            },
        }

    async def run_retention_cleanup(self, request: RetentionCleanupRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._run_retention_cleanup, request)

    def _run_retention_cleanup(self, request: RetentionCleanupRequest) -> dict[str, object]:
        region = self._session_region(request.library_id)
        with (
            self._maintenance_lock,
            self._governance_region_lock(region.region_id),
            self._graph_write_lock,
        ):
            return self._run_retention_cleanup_locked(request)

    def _run_retention_cleanup_locked(self, request: RetentionCleanupRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._session_region(request.library_id)
        plan = self._retention_plan(region)
        for scope, units in _group_units_by_scope(plan):
            bundle.runtime.api.delete(
                DeleteSelector(
                    unit_ids=[unit.id for unit in units],
                    scope=scope,
                    mode=DeleteMode.ARCHIVE,
                ),
                security=_security_for(scope),
            )
            self._remove_graph_nodes(
                bundle,
                scope,
                [unit.id for unit in units],
                region.region_id,
            )
        return {
            "library_id": region.region_id,
            "region_id": region.region_id,
            "as_of": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "policy": self._retention_payload(region, self._retention_days(region.region_id)),
            "counts": {
                "captures_to_archive": len(plan),
                "captures_archived": len(plan),
                "evolved_units_protected": self._evolved_count(region),
            },
        }

    def list_capture_events(self, request: CaptureEventsRequest) -> list[dict[str, object]]:
        bundle = self._require_bundle()
        session_id = request.session_id.strip() if request.session_id is not None else None
        events: list[dict[str, object]] = []
        for scope in bundle.kv_store.scopes():
            if session_id is not None and scope.session != session_id:
                continue
            for unit in self._units_in_scope(bundle, scope):
                if unit.lifecycle.value != "active" or not _is_capture_unit(unit):
                    continue
                received = unit.temporal.t_ingest
                events.append(
                    {
                        "event_id": _metadata_text(unit, "event_id"),
                        "session_id": _metadata_text(unit, "session_id"),
                        "project_id": _metadata_text(unit, "project_id"),
                        "library_id": _metadata_text(unit, "region_id"),
                        "region_id": _metadata_text(unit, "region_id"),
                        "turn_id": _metadata_text(unit, "turn_id"),
                        "event_kind": _metadata_text(unit, "event_kind"),
                        "content": unit.content,
                        "occurred_at": _metadata_text(unit, "occurred_at"),
                        "received_at": (received.isoformat() if received is not None else ""),
                        "consolidated_at": None,
                    }
                )
        events.sort(key=lambda item: (str(item["received_at"]), str(item["event_id"])))
        return events

    def list_capture_rounds(self, request: CaptureRoundsRequest) -> list[dict[str, object]]:
        events = self.list_capture_events(CaptureEventsRequest(session_id=request.session_id))
        consolidated = {(session_id, turn_id) for session_id, turn_id in self._evolution_turns()}
        rounds: dict[tuple[str, str, str], dict[str, object]] = {}
        for event in events:
            key = (
                str(event["session_id"]),
                str(event["turn_id"]),
                str(event["region_id"]),
            )
            current = rounds.get(key)
            if current is None:
                rounds[key] = {
                    "session_id": event["session_id"],
                    "turn_id": event["turn_id"],
                    "library_id": event["region_id"],
                    "status": (
                        "done"
                        if (event["session_id"], event["turn_id"]) in consolidated
                        else "pending"
                    ),
                    "candidate_id": None,
                    "last_error": "",
                    "updated_at": event["received_at"],
                }
            else:
                current["updated_at"] = max(str(current["updated_at"]), str(event["received_at"]))
        return sorted(
            rounds.values(),
            key=lambda item: (str(item["updated_at"]), str(item["turn_id"])),
        )

    def _units_in_scope(self, bundle: _RuntimeBundle, scope: Scope) -> list[MemoryUnit]:
        rows = bundle.kv_store.scan(scope, "/memory/")
        unit_ids = [key.removeprefix("/memory/") for key, _ in rows]
        return list(load_units(bundle.kv_store, scope, unit_ids))

    def _session_region(self, region_id: str) -> MemoryRegion:
        region = self._regions.require_region(region_id)
        if region.kind != "session":
            raise RegionBindingError("retention applies to session memory regions")
        return region

    def _retention_days(self, region_id: str) -> int:
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                "SELECT retention_days FROM session_retention_policy WHERE region_id = ?",
                (region_id,),
            ).fetchone()
        return _DEFAULT_SESSION_RETENTION_DAYS if row is None else int(row[0])

    def _retention_payload(self, region: MemoryRegion, days: int) -> dict[str, object]:
        return {
            "library_id": region.region_id,
            "region_id": region.region_id,
            "inbox_days": days,
            "retention_days": days,
            "candidate_days": 0,
            "recycle_days": 0,
            "early_inbox_cleanup": True,
            "diagnostics_enabled": False,
            "diagnostic_days": 0,
        }

    def _retention_plan(self, region: MemoryRegion) -> list[tuple[Scope, MemoryUnit]]:
        bundle = self._require_bundle()
        cutoff = datetime.now(UTC) - timedelta(days=self._retention_days(region.region_id))
        expired: list[tuple[Scope, MemoryUnit]] = []
        for agent in _SEARCH_AGENTS:
            scope = _scope_for_region(region, session_id=region.session_id, agent=agent)
            for unit in self._units_in_scope(bundle, scope):
                if unit.lifecycle.value != "active" or not _is_capture_unit(unit):
                    continue
                occurred_at = _parse_timestamp(_metadata_text(unit, "occurred_at"))
                if occurred_at is not None and occurred_at < cutoff:
                    expired.append((scope, unit))
        return expired

    def _evolved_count(self, region: MemoryRegion) -> int:
        bundle = self._require_bundle()
        count = 0
        for agent in _SEARCH_AGENTS:
            scope = _scope_for_region(region, session_id=region.session_id, agent=agent)
            count += sum(
                1
                for unit in self._units_in_scope(bundle, scope)
                if unit.lifecycle.value == "active"
                and _metadata_text(unit, "memory_type") == "evolved"
            )
        return count

    def _maintenance_db(self) -> sqlite3.Connection:
        path = self._state_dir / "mem2" / "maintenance.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        with self._idempotency_schema_lock:
            self._ensure_maintenance_schema(connection)
        return connection

    @staticmethod
    def _ensure_maintenance_schema(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(capture_evolutions)").fetchall()
        }
        if not columns:
            connection.executescript(_MAINTENANCE_SCHEMA)
            Mem2MemoryKernel._ensure_governance_schema(connection)
            return
        memory_id_column = columns.get("memory_id")
        if (
            "status" in columns
            and "completed_at" in columns
            and memory_id_column is not None
            and not memory_id_column[3]
        ):
            connection.execute(_SESSION_RETENTION_SCHEMA)
            connection.execute(_GOVERNANCE_AUDIT_SCHEMA)
            Mem2MemoryKernel._ensure_governance_schema(connection)
            if "source_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE capture_evolutions "
                    "ADD COLUMN source_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            return
        with connection:
            connection.execute("ALTER TABLE capture_evolutions RENAME TO capture_evolutions_v1")
            connection.execute(_SESSION_RETENTION_SCHEMA)
            connection.execute(_CAPTURE_EVOLUTIONS_SCHEMA)
            connection.execute(_GOVERNANCE_AUDIT_SCHEMA)
            Mem2MemoryKernel._ensure_governance_schema(connection)
            connection.execute(
                """INSERT OR REPLACE INTO capture_evolutions
                   (evolve_key, session_id, turn_id, region_id, source_fingerprint, memory_id,
                    status, created_at, completed_at)
                   SELECT evolve_key, session_id, turn_id, region_id, '', memory_id,
                          'completed', created_at, created_at
                   FROM capture_evolutions_v1"""
            )
            connection.execute("DROP TABLE capture_evolutions_v1")

    @staticmethod
    def _ensure_governance_schema(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(governance_events)").fetchall()
        }
        if "status" not in columns:
            with connection:
                connection.execute(
                    "ALTER TABLE governance_events "
                    "ADD COLUMN status TEXT NOT NULL DEFAULT 'committed'"
                )

    def _reserve_evolution(
        self,
        evolve_key: str,
        session_id: str,
        turn_id: str,
        region_id: str,
        source_fingerprint: str,
    ) -> tuple[str, str | None]:
        with closing(self._maintenance_db()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO capture_evolutions
                   (evolve_key, session_id, turn_id, region_id, source_fingerprint,
                    memory_id, status)
                   VALUES (?, ?, ?, ?, ?, NULL, 'pending')""",
                (evolve_key, session_id, turn_id, region_id, source_fingerprint),
            )
            if cursor.rowcount == 1:
                return "pending", None
            row = connection.execute(
                """SELECT status, memory_id, source_fingerprint FROM capture_evolutions
                   WHERE evolve_key = ?""",
                (evolve_key,),
            ).fetchone()
        if row is None:
            raise MemoryKernelUnavailableError("evolve reservation was not persisted")
        if str(row[2]) != source_fingerprint:
            with closing(self._maintenance_db()) as connection, connection:
                connection.execute(
                    """UPDATE capture_evolutions
                       SET source_fingerprint = ?, memory_id = NULL,
                           status = 'pending', completed_at = NULL
                       WHERE evolve_key = ?""",
                    (source_fingerprint, evolve_key),
                )
            return "pending", None
        return str(row[0]), None if row[1] is None else str(row[1])

    def _complete_evolution(self, evolve_key: str, memory_id: str) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET memory_id = ?, status = 'completed',
                       completed_at = CURRENT_TIMESTAMP
                   WHERE evolve_key = ?""",
                (memory_id, evolve_key),
            )

    def _evolution_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        evolve_key: str,
    ) -> list[MemoryUnit]:
        return [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "evolve_key") == evolve_key
        ]

    def _purge_capture_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        event_id: str,
    ) -> None:
        units = [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "capture_event_id") == event_id
        ]
        if not units:
            return
        bundle.runtime.api.delete(
            DeleteSelector(
                unit_ids=[str(unit.id) for unit in units],
                scope=scope,
                mode=DeleteMode.PURGE,
            ),
            security=_security_for(scope),
        )
        if any(
            _metadata_text(unit, "capture_event_id") == event_id
            for unit in self._units_in_scope(bundle, scope)
        ):
            raise MemoryKernelUnavailableError("capture residue cleanup did not persist")
        if not self._fulltext_visible(
            bundle,
            scope,
            [str(unit.id) for unit in units],
            expected=False,
        ):
            raise MemoryKernelUnavailableError("capture index cleanup did not persist")

    def _capture_visible(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> bool:
        return len(units) == 1 and self._fulltext_visible(
            bundle,
            scope,
            [str(units[0].id)],
            expected=True,
        )

    def _fulltext_visible(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        unit_ids: list[str],
        *,
        expected: bool,
    ) -> bool:
        fulltext = _fulltext_store(bundle)
        if fulltext is None:
            return False
        try:
            documents = fulltext.get(scope, unit_ids)
        except Exception:
            return False
        visible_ids = {str(document.id) for document in documents}
        return visible_ids == set(unit_ids) if expected else not visible_ids.intersection(unit_ids)

    def _evolution_visible(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
        expected_parts: int,
        source_fingerprint: str,
    ) -> bool:
        if len(units) != expected_parts:
            return False
        parts = {_metadata_int(unit, "evolve_part") for unit in units}
        if parts != set(range(expected_parts)):
            return False
        if any(_metadata_text(unit, "source_fingerprint") != source_fingerprint for unit in units):
            return False
        return self._fulltext_visible(
            bundle,
            scope,
            [str(unit.id) for unit in units],
            expected=True,
        )

    def _purge_evolution_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> None:
        if not units:
            return
        bundle.runtime.api.delete(
            DeleteSelector(
                unit_ids=[str(unit.id) for unit in units],
                scope=scope,
                mode=DeleteMode.PURGE,
            ),
            security=_security_for(scope),
        )
        remaining = [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "evolve_key")
            in {_metadata_text(item, "evolve_key") for item in units}
        ]
        if remaining:
            raise MemoryKernelUnavailableError("evolved memory cleanup did not persist")

    def _evolution_turns(self) -> list[tuple[str, str]]:
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                """SELECT session_id, turn_id FROM capture_evolutions
                   WHERE status = 'completed'"""
            ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

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

    def _reserve_capture_idempotency(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        request: CaptureMemoryRequest,
        identity: str,
        request_hash: str,
    ) -> str | None:
        with closing(self._idempotency_db()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO memory_write_idempotency
                   (library_id, idempotency_key, request_hash, memory_id, status)
                   VALUES (?, ?, ?, NULL, 'pending')""",
                (identity, request.event_id, request_hash),
            )
            if cursor.rowcount == 1:
                return None
            row = connection.execute(
                """SELECT request_hash, memory_id, status
                   FROM memory_write_idempotency
                   WHERE library_id = ? AND idempotency_key = ?""",
                (identity, request.event_id),
            ).fetchone()
        if row is None:
            raise MemoryKernelUnavailableError("idempotency reservation was not persisted")
        if str(row[0]) != request_hash:
            raise MemoryIdempotencyConflictError(
                "idempotency_key was already used for another memory content"
            )
        if str(row[2]) == "completed" and row[1] is not None:
            return str(row[1])
        if str(row[2]) != "pending":
            raise MemoryKernelUnavailableError(
                "completed idempotency record has no memory identifier"
            )

        residual = [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "capture_event_id") == request.event_id
        ]
        if residual and self._capture_visible(bundle, scope, residual):
            self._complete_idempotency(
                identity,
                request.event_id,
                request_hash,
                residual[0].id,
            )
            return str(residual[0].id)
        if residual:
            self._purge_capture_units(bundle, scope, request.event_id)
        # Keep the reservation pending while this retry rebuilds the capture.
        # Releasing it here allows a timed-out background write and a new retry
        # to interleave and create duplicate units.
        return None

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
        resolved = self._regions.resolve(request.cwd)
        if resolved is None:
            return {"status": "unbound", "cwd": self._regions.canonical_cwd(request.cwd)}
        return {
            "status": "bound",
            "cwd": resolved.cwd,
            "project_id": resolved.region.region_id,
            "binding_id": resolved.binding.binding_id,
            "binding_kind": resolved.binding.kind,
            "project_root": resolved.binding.canonical_root,
            "region_id": resolved.region.region_id,
            "library_id": resolved.region.region_id,
            "space": resolved.region.space,
        }

    def create_region(self, request: CreateMemoryRegionRequest) -> dict[str, object]:
        return self._regions.create_region(
            request.kind,
            request.name,
            region_id=request.region_id,
            session_id=request.session_id,
        ).payload()

    def list_regions(self) -> list[dict[str, object]]:
        return [region.payload() for region in self._regions.list_regions()]

    def get_region(self, request: GetMemoryRegionRequest) -> dict[str, object]:
        return self._regions.require_region(request.region_id).payload()

    def region_tree(self, request: MemoryRegionTreeRequest) -> dict[str, object]:
        return self._regions.region_tree(request.session_id)

    def bind_project_region(self, request: BindProjectRegionRequest) -> dict[str, str]:
        return self._regions.bind_project(request.project_root, request.region_id).payload()

    def associate_worktree_region(self, request: AssociateWorktreeRegionRequest) -> dict[str, str]:
        return self._regions.associate_worktree(
            request.worktree_root, request.main_project_binding_id
        ).payload()

    def list_region_bindings(self) -> list[dict[str, str]]:
        return [binding.payload() for binding in self._regions.list_bindings()]

    def update_region_binding(self, request: UpdateRegionBindingRequest) -> dict[str, str]:
        return self._regions.update_binding(request.binding_id, request.region_id).payload()

    def delete_region_binding(self, request: DeleteRegionBindingRequest) -> None:
        self._regions.delete_binding(request.binding_id)

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
            "graph_store": {
                "default": {
                    "target": _STORE_TARGET,
                    "params": {
                        "uri": milvus_uri,
                        "collection": "agent_memory_graph_nodes",
                        "edge_collection": "agent_memory_graph_edges",
                    },
                }
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


def _write_identity(request: CreateMemoryRequest) -> str:
    if request.region_id is not None:
        return f"region:{request.region_id}"
    return request.library_id


def _normalize_session_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("session_id must not be blank")
    if len(normalized) > MAX_SESSION_ID_LENGTH:
        raise ValueError(f"session_id exceeds the {MAX_SESSION_ID_LENGTH}-character scope limit")
    return normalized


def _normalize_optional_identifier(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be blank")
    return normalized


def _validate_occurred_at(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("occurred_at must include a valid timezone offset")
        parsed.astimezone(UTC)
    except (ValueError, OverflowError) as error:
        raise ValueError("occurred_at must be a valid ISO-8601 timestamp with timezone") from error
    return value


def _capture_request_hash(request: CaptureMemoryRequest, scope: Scope) -> str:
    payload = {
        "event_id": request.event_id,
        "session_id": request.session_id,
        "turn_id": request.turn_id,
        "event_kind": request.event_kind,
        "content": request.content,
        "occurred_at": request.occurred_at,
        "cwd": request.cwd,
        "topic_id": request.topic_id,
        "space": scope.space,
        "agent": scope.agent,
        "session": scope.session,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _capture_response(
    request: CaptureMemoryRequest,
    memory_id: str,
    region: MemoryRegion,
    scope: Scope,
    *,
    status: str,
    duplicate: bool = False,
) -> dict[str, object]:
    response: dict[str, object] = {
        "id": memory_id,
        "memory_id": memory_id,
        "event_id": request.event_id,
        "status": status,
        "region_id": region.region_id,
        "library_id": region.region_id,
        "space": scope.space,
        "agent": scope.agent,
        "session_id": request.session_id,
    }
    if duplicate:
        response["duplicate"] = True
    return response


def _is_capture_unit(unit: MemoryUnit) -> bool:
    return bool(_metadata_text(unit, "capture_event_id"))


def _fulltext_store(bundle: _RuntimeBundle) -> MilvusLiteFulltextStore | None:
    return next(
        (
            store
            for store in bundle.stores
            if isinstance(store, MilvusLiteFulltextStore)
            and getattr(store, "_collection", "") == "agent_memory_fulltext"
        ),
        None,
    )


def _stable_evolve_key(scope: Scope, region_id: str, turn_id: str) -> str:
    return hashlib.sha256(
        "\0".join((scope.org, scope.space, scope.session, region_id, turn_id)).encode()
    ).hexdigest()


def _source_fingerprint(source_ids: list[str]) -> str:
    return hashlib.sha256("\0".join(source_ids).encode()).hexdigest()


def _metadata_text(unit: MemoryUnit, key: str) -> str:
    value = unit.user_metadata.get(key)
    return "" if value is None else str(value)


def _metadata_int(unit: MemoryUnit, key: str) -> int:
    value = unit.user_metadata.get(key)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return -1


def _chunk_evolved_content(content: str) -> list[str]:
    remaining = content
    chunks: list[str] = []
    while remaining:
        if (
            len(remaining) <= _FULLTEXT_CONTENT_LIMIT
            and len(remaining.encode("utf-8")) <= _EVOLVE_CHUNK_LIMIT
        ):
            chunks.append(remaining)
            break
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = remaining[:middle]
            if (
                len(candidate) <= _FULLTEXT_CONTENT_LIMIT
                and len(candidate.encode("utf-8")) <= _EVOLVE_CHUNK_LIMIT
            ):
                low = middle
            else:
                high = middle - 1
        boundary = max(
            remaining.rfind("\n\n", 0, low),
            remaining.rfind("\n", 0, low),
            remaining.rfind(" ", 0, low),
        )
        cut = boundary + 1 if boundary > low // 2 else low
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    return chunks or [""]


def _group_units_by_scope(
    items: list[tuple[Scope, MemoryUnit]],
) -> list[tuple[Scope, list[MemoryUnit]]]:
    grouped: dict[tuple[str, str, str, str, str], tuple[Scope, list[MemoryUnit]]] = {}
    for scope, unit in items:
        key = (scope.org, scope.space, scope.user, scope.agent, scope.session)
        grouped.setdefault(key, (scope, []))[1].append(unit)
    return list(grouped.values())


def _event_order(kind: str) -> int:
    return 0 if kind == "user" else 1


def _event_label(kind: str) -> str:
    return "用户" if kind == "user" else "助手"


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    except (ValueError, OverflowError):
        return None


def _audit_sort_key(item: dict[str, str]) -> datetime:
    parsed = _parse_timestamp(item.get("occurred_at", ""))
    return parsed if parsed is not None else datetime.min.replace(tzinfo=UTC)


def _write_request_hash(request: CreateMemoryRequest) -> str:
    payload = {
        "library_id": request.library_id,
        "region_id": request.region_id,
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
        "region_id": getattr(request, "region_id", None),
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


def _space_for_region(region: MemoryRegion) -> str:
    return region.space


def _scope_for_region(
    region: MemoryRegion,
    *,
    session_id: str | None = None,
    agent: str = _LOCAL_AGENT,
) -> Scope:
    return Scope(
        org=_LOCAL_ORG,
        space=region.space,
        user=_LOCAL_USER,
        agent=agent,
        session=(region.session_id or "") if session_id is None else session_id,
    )


def _scope_for_synthetic_cwd(cwd: str, session_id: str | None) -> Scope:
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
    graph_status: str = "not_configured",
    region: MemoryRegion | None = None,
) -> dict[str, object]:
    visible_items = [
        item
        for item in items
        if not (isinstance(item, MemoryUnit) and item.user_metadata.get("restore_pending") is True)
    ]
    results = [_result_payload(item, request) for item in visible_items]
    if region is not None:
        for result in results:
            result["region_id"] = region.region_id
            result["space"] = region.space
    scope_kind = (
        "explicit_region"
        if request.region_id is not None
        else "explicit_library"
        if request.library_id is not None
        else "project_cwd"
    )
    response: dict[str, object] = {
        "status": "bound",
        "scope": {
            "kind": scope_kind,
            "cwd": request.cwd,
            "library_id": request.library_id,
            "region_id": request.region_id,
            "space": scope.space,
        },
        "query": request.query,
        "degraded": bool(degradation),
        "degradation": degradation,
        "vector_index_status": vector_status,
        "graph_index_status": graph_status,
        "results": _truncate_search_results(results, max(1, min(request.limit, 100))),
    }
    if request.library_id is not None:
        response["library_id"] = request.library_id
    if request.region_id is not None:
        response["region_id"] = request.region_id
    if region is not None:
        response["region_id"] = region.region_id
        response["space"] = region.space
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


def _json_metadata(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    return str(value)


def _unit_revision(unit: MemoryUnit) -> str:
    payload = {
        "id": unit.id,
        "content": unit.content,
        "lifecycle": unit.lifecycle.value,
        "supersedes": unit.supersedes,
        "provenance": list(unit.provenance),
        "system_metadata": _json_metadata(dict(unit.system_metadata)),
        "user_metadata": _json_metadata(dict(unit.user_metadata)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unit_payload(unit: MemoryUnit, region: MemoryRegion) -> dict[str, object]:
    user_metadata = dict(unit.user_metadata)
    source_references = user_metadata.get("source_references")
    provenance = list(unit.provenance)
    if not provenance:
        metadata_provenance = user_metadata.get("provenance")
        if not isinstance(metadata_provenance, list):
            metadata_provenance = user_metadata.get("source_unit_ids")
        if isinstance(metadata_provenance, list):
            provenance = [str(value) for value in metadata_provenance]
    if not isinstance(source_references, list):
        source_references = provenance
    revision = _unit_revision(unit)
    ingested = unit.temporal.t_ingest or unit.temporal.t_valid or unit.temporal.t_message
    updated = unit.temporal.t_ingest
    if unit.lifecycle.value in {"superseded", "forgotten"}:
        updated = unit.temporal.t_invalid or updated
    deleted = unit.temporal.t_invalid if unit.lifecycle.value == "forgotten" else None
    return {
        "id": unit.id,
        "path": unit.id,
        "document_id": unit.id,
        "content": unit.content,
        "metadata": {
            "system": _json_metadata(dict(unit.system_metadata)),
            "user": _json_metadata(user_metadata),
        },
        "system_metadata": _json_metadata(dict(unit.system_metadata)),
        "user_metadata": _json_metadata(user_metadata),
        "lifecycle": unit.lifecycle.value,
        "tier": unit.tier.value,
        "version": revision,
        "revision": revision,
        "source_version": revision,
        "source": unit.source_ref or "mem2",
        "source_references": _json_metadata(source_references),
        "provenance": provenance,
        "supersedes": unit.supersedes,
        "region_id": region.region_id,
        "library_id": region.region_id,
        "space": region.space,
        "scope": {
            "org": unit.scope.org,
            "space": unit.scope.space,
            "user": unit.scope.user,
            "agent": unit.scope.agent,
            "session": unit.scope.session,
        },
        "created_at": ingested.isoformat() if ingested is not None else "",
        "updated_at": updated.isoformat() if updated else "",
        "deleted_at": deleted.isoformat() if deleted else "",
        "valid_from": unit.temporal.t_valid.isoformat() if unit.temporal.t_valid else "",
        "valid_until": unit.temporal.t_invalid.isoformat() if unit.temporal.t_invalid else "",
        "active": unit.lifecycle.value == "active",
    }


def _unit_sort_key(pair: tuple[Scope, MemoryUnit]) -> tuple[str, str]:
    _, unit = pair
    timestamp = (
        unit.temporal.t_ingest
        or unit.temporal.t_valid
        or unit.temporal.t_message
        or unit.temporal.t_event
    )
    return (timestamp.isoformat() if timestamp is not None else "", unit.id)


def _version_chain(
    unit: MemoryUnit, pairs: list[tuple[Scope, MemoryUnit]]
) -> list[tuple[Scope, MemoryUnit]]:
    ids = {unit.id}
    changed = True
    while changed:
        changed = False
        for _, candidate in pairs:
            additions = set()
            if candidate.id in ids and candidate.supersedes:
                additions.add(candidate.supersedes)
            if candidate.supersedes in ids:
                additions.add(candidate.id)
            new_ids = additions - ids
            if new_ids:
                ids.update(new_ids)
                changed = True
    return sorted((pair for pair in pairs if pair[1].id in ids), key=_unit_sort_key)


def _require_revision(unit: MemoryUnit, expected: str) -> None:
    if not expected or expected != _unit_revision(unit):
        raise MemoryRevisionConflictError("memory revision conflict")


def _require_active(unit: MemoryUnit, operation: str) -> None:
    if unit.lifecycle.value != "active":
        raise MemoryLifecycleConflictError(
            f"memory lifecycle {unit.lifecycle.value} does not allow {operation}"
        )


def _require_delete_lifecycle(unit: MemoryUnit, mode: str) -> None:
    allowed = {
        "forget": {"active", "archived", "superseded"},
        "archive": {"active", "archived"},
        "downweight": {"active"},
        "purge": {"active", "archived", "superseded", "forgotten"},
    }
    if unit.lifecycle.value not in allowed[mode]:
        raise MemoryLifecycleConflictError(
            f"memory lifecycle {unit.lifecycle.value} does not allow delete mode {mode}"
        )


def _content_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )


def _truncate_search_results(
    results: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    direct = [item for item in results if item.get("classification") != "graph_expansion"]
    graph = [item for item in results if item.get("classification") == "graph_expansion"]
    if limit <= 1:
        return (direct or graph)[: max(limit, 0)]
    if not graph:
        return direct[:limit]
    graph_slots = min(len(graph), max(1, limit // 5))
    if limit > 1:
        graph_slots = min(graph_slots, limit - 1)
    direct_slots = max(0, limit - graph_slots)
    selected = direct[:direct_slots] + graph[:graph_slots]
    remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(direct[direct_slots:][:remaining])
        remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(graph[graph_slots:][:remaining])
    return selected[:limit]


def _result_payload(item: Any, request: SearchMemoryRequest) -> dict[str, object]:
    if isinstance(item, dict):
        return {str(key): value for key, value in item.items()}
    return _result_item(item, request)


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
        "library_id": request.region_id or request.library_id or "",
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
