from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import tarfile
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Protocol

from personal_agent_memory.model_client import ModelServiceError, OpenAICompatibleClient


class GraphAdapterError(RuntimeError):
    """The optional graph projection is unavailable or returned invalid data."""


class GraphAdapterBusyError(GraphAdapterError):
    """Milvus Lite is busy mutating another local graph projection."""


@dataclass(frozen=True, slots=True)
class GraphSourceDocument:
    document_id: str
    path: str
    source_version: str
    content: str


@dataclass(frozen=True, slots=True)
class GraphExpansion:
    content: str
    document_id: str
    path: str
    source_version: str
    hop: int
    graph_object_type: str
    source_anchor: str = ""


class GraphPurge(Protocol):
    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class PreparedGraphRebuild(Protocol):
    def activate(self) -> GraphPurge: ...

    def discard(self) -> None: ...


class GraphAdapter(Protocol):
    def stage_purge(self, library_id: str, *, cleanup_id: str | None = None) -> GraphPurge: ...

    def stage_rebuild(
        self,
        library_id: str,
        documents: tuple[GraphSourceDocument, ...],
        *,
        cleanup_id: str | None = None,
    ) -> GraphPurge: ...

    def reconcile_staged_purge(
        self, library_id: str, cleanup_id: str, *, committed: bool
    ) -> None: ...

    async def rebuild(
        self, library_id: str, documents: tuple[GraphSourceDocument, ...]
    ) -> None: ...

    async def expand(
        self,
        library_id: str,
        seed_document_ids: tuple[str, ...],
        *,
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]: ...


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    path: str
    kind: str
    mode: int
    size: int
    digest: str


def _tree_manifest(root: Path) -> tuple[_TreeEntry, ...]:
    entries: list[_TreeEntry] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise GraphAdapterError("Milvus Lite graph projection contains a symlink")
        if path.is_dir():
            entries.append(_TreeEntry(relative, "directory", metadata.st_mode & 0o777, 0, ""))
            continue
        if not path.is_file():
            raise GraphAdapterError("Milvus Lite graph projection contains an unsupported entry")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(
            _TreeEntry(
                relative,
                "file",
                metadata.st_mode & 0o777,
                metadata.st_size,
                digest.hexdigest(),
            )
        )
    return tuple(entries)


@contextmanager
def _recoverable_tree_snapshot(root: Path) -> Iterator[tuple[BinaryIO, tuple[_TreeEntry, ...]]]:
    resources = ExitStack()
    try:
        manifest = _tree_manifest(root)
        archive = resources.enter_context(
            tempfile.TemporaryFile(prefix="personal-agent-memory-graph-")  # noqa: SIM115
        )
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            bundle.dereference = True
            bundle.add(root, arcname=".", recursive=True)
        archive.flush()
        archive.seek(0)
    except (OSError, tarfile.TarError) as error:
        resources.close()
        raise GraphAdapterError("Milvus Lite graph cleanup snapshot failed") from error
    try:
        yield archive, manifest
    finally:
        resources.close()


def _restore_tree_snapshot(root: Path, archive: BinaryIO, manifest: tuple[_TreeEntry, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive.seek(0)
    try:
        with tarfile.open(fileobj=archive, mode="r:") as bundle:
            members = bundle.getmembers()
            for member in members:
                relative = member.name.removeprefix("./")
                if relative in {"", "."}:
                    continue
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                    raise GraphAdapterError("Milvus Lite graph snapshot contains an unsafe entry")
                target = root / path
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=member.mode)
                    os.chmod(target, member.mode)
                    continue
                if not member.isfile():
                    raise GraphAdapterError(
                        "Milvus Lite graph snapshot contains an unsupported entry"
                    )
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = bundle.extractfile(member)
                if source is None:
                    raise GraphAdapterError("Milvus Lite graph snapshot cannot be read")
                temporary = target.with_name(f".{target.name}.restore-{uuid.uuid4().hex}")
                try:
                    with temporary.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
                        destination.flush()
                        os.fsync(destination.fileno())
                    os.chmod(temporary, member.mode)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
    except (OSError, tarfile.TarError) as error:
        raise GraphAdapterError("Milvus Lite graph cleanup snapshot restore failed") from error
    try:
        restored_manifest = _tree_manifest(root)
    except OSError as error:
        raise GraphAdapterError("Milvus Lite graph cleanup snapshot verification failed") from error
    if restored_manifest != manifest:
        raise GraphAdapterError("Milvus Lite graph cleanup snapshot restore was incomplete")


def _remove_tree_recoverably(path: Path) -> None:
    if not path.exists():
        return
    with _recoverable_tree_snapshot(path) as (archive, manifest):
        try:
            shutil.rmtree(path)
            if path.exists():
                raise OSError("graph projection cleanup left the directory present")
        except OSError as error:
            _restore_tree_snapshot(path, archive, manifest)
            raise GraphAdapterError("Milvus Lite graph projection cleanup failed") from error


@dataclass(slots=True)
class _FilesystemGraphPurge:
    original: Path
    quarantine: Path | None

    def commit(self) -> None:
        if self.quarantine is None:
            return
        quarantine = self.quarantine
        _remove_tree_recoverably(quarantine)
        self.quarantine = None

    def rollback(self) -> None:
        if self.quarantine is None:
            return
        if not self.quarantine.exists():
            raise GraphAdapterError("Milvus Lite graph purge rollback source is missing")
        if self.original.exists():
            raise GraphAdapterError("Milvus Lite graph purge rollback destination exists")
        try:
            os.replace(self.quarantine, self.original)
        except OSError as error:
            raise GraphAdapterError("Milvus Lite graph purge rollback failed") from error
        self.quarantine = None


@dataclass(slots=True)
class _CompositeGraphPurge:
    disposable: tuple[GraphPurge, ...]
    rollback_critical: tuple[GraphPurge, ...] = ()

    def commit(self) -> None:
        for purge in (*self.disposable, *self.rollback_critical):
            purge.commit()

    def rollback(self) -> None:
        errors: list[str] = []
        purges = (*reversed(self.rollback_critical), *reversed(self.disposable))
        for purge in purges:
            try:
                purge.rollback()
            except GraphAdapterError as error:
                errors.append(str(error))
        if errors:
            raise GraphAdapterError("; ".join(errors))


@dataclass(slots=True)
class _LockedGraphPurge:
    locks: tuple[Any, ...]
    purge: GraphPurge

    def commit(self) -> None:
        with ExitStack() as stack:
            for lock in self.locks:
                stack.enter_context(lock)
            self.purge.commit()

    def rollback(self) -> None:
        with ExitStack() as stack:
            for lock in self.locks:
                stack.enter_context(lock)
            self.purge.rollback()


@dataclass(slots=True)
class _FilesystemGraphReplacement:
    replacement: Path
    previous: _FilesystemGraphPurge
    detached_replacement: Path | None = None

    def commit(self) -> None:
        self.previous.commit()

    def rollback(self) -> None:
        if self.previous.quarantine is not None:
            if self.detached_replacement is not None:
                raise GraphAdapterError("Milvus Lite graph replacement rollback is inconsistent")
            if not self.replacement.exists():
                self.previous.rollback()
                return
            detached = self.replacement.parent / (
                f".{self.replacement.name}.rollback-{uuid.uuid4().hex}"
            )
            try:
                os.replace(self.replacement, detached)
            except OSError as error:
                raise GraphAdapterError(
                    "Milvus Lite graph replacement could not be detached"
                ) from error
            self.detached_replacement = detached
            try:
                self.previous.rollback()
            except BaseException:
                with suppress(OSError):
                    if not self.replacement.exists() and detached.exists():
                        os.replace(detached, self.replacement)
                        self.detached_replacement = None
                raise
        if self.detached_replacement is None:
            return
        detached = self.detached_replacement
        _remove_tree_recoverably(detached)
        self.detached_replacement = None


@dataclass(slots=True)
class _PreparedFilesystemGraphRebuild:
    adapter: JiuwenMilvusGraphAdapter
    library_id: str
    extracted: list[tuple[GraphSourceDocument, dict[str, object]]] | None

    def activate(self) -> GraphPurge:
        if self.extracted is None:
            raise GraphAdapterError("prepared graph rebuild is no longer available")
        extracted = self.extracted
        self.extracted = None
        return self.adapter._activate_prepared_rebuild(self.library_id, extracted)

    def discard(self) -> None:
        self.extracted = None


class JiuwenMilvusGraphAdapter:
    """Project-owned adapter over Jiuwen's graph store and local Milvus Lite.

    A separate Milvus Lite file is used for every memory library. The adapter never
    treats its contents as authoritative: every row carries a Markdown document ID
    and version which the platform validates again before returning it.
    """

    _PROJECTION_MARKER = "projection.ready"
    _MILVUS_COMPLETE_READ_LIMIT = 16_384

    def __init__(self, root: Path, model_client: OpenAICompatibleClient) -> None:
        self.root = root
        self.model_client = model_client
        self._operation_lock = threading.RLock()
        self._mutation_locks: dict[str, Any] = {}
        self._mutation_locks_guard = threading.Lock()

    def _mutation_lock(self, library_id: str) -> Any:
        with self._mutation_locks_guard:
            return self._mutation_locks.setdefault(library_id, threading.RLock())

    def stage_purge(self, library_id: str, *, cleanup_id: str | None = None) -> GraphPurge:
        lock = self._mutation_lock(library_id)
        with self._operation_lock, lock:
            return _LockedGraphPurge(
                (self._operation_lock, lock), self._stage_purge(library_id, cleanup_id=cleanup_id)
            )

    def _stage_purge(
        self, library_id: str, *excluded: Path, cleanup_id: str | None = None
    ) -> GraphPurge:
        excluded_paths = set(excluded)
        roots = sorted(self.root.glob(f".{library_id}.replacement-*"))
        roots.append(self.root / library_id)
        disposable: list[GraphPurge] = []
        rollback_critical: list[GraphPurge] = []
        try:
            for library_root in roots:
                if library_root in excluded_paths or not library_root.exists():
                    continue
                role = "official" if library_root == self.root / library_id else "disposable"
                quarantine = self._purge_path(library_id, cleanup_id, role)
                os.replace(library_root, quarantine)
                purge = _FilesystemGraphPurge(library_root, quarantine)
                if library_root == self.root / library_id:
                    rollback_critical.append(purge)
                else:
                    disposable.append(purge)
        except OSError as error:
            purges = (*reversed(rollback_critical), *reversed(disposable))
            for staged_purge in purges:
                staged_purge.rollback()
            raise GraphAdapterError("Milvus Lite graph purge could not be staged") from error
        return _CompositeGraphPurge(tuple(disposable), tuple(rollback_critical))

    def _stage_single_purge(
        self, library_root: Path, *, cleanup_id: str | None = None
    ) -> _FilesystemGraphPurge:
        if not library_root.exists():
            return _FilesystemGraphPurge(library_root, None)
        quarantine = self._purge_path(library_root.name, cleanup_id, "official")
        try:
            os.replace(library_root, quarantine)
        except OSError as error:
            raise GraphAdapterError("Milvus Lite graph purge could not be staged") from error
        return _FilesystemGraphPurge(library_root, quarantine)

    def stage_rebuild(
        self,
        library_id: str,
        documents: tuple[GraphSourceDocument, ...],
        *,
        cleanup_id: str | None = None,
    ) -> GraphPurge:
        extracted = self._extract_documents(documents)
        return self._activate_prepared_rebuild(library_id, extracted, cleanup_id=cleanup_id)

    def _activate_prepared_rebuild(
        self,
        library_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
        *,
        cleanup_id: str | None = None,
    ) -> GraphPurge:
        lock = self._mutation_lock(library_id)
        with self._operation_lock, lock:
            temporary_id = f".{library_id}.replacement-{uuid.uuid4().hex}"
            temporary = self.root / temporary_id
            try:
                self._replace_projection(temporary_id, extracted, library_id)
            except BaseException:
                if temporary.exists():
                    with suppress(OSError):
                        shutil.rmtree(temporary)
                raise
            stale_replacements = self._stage_purge(
                library_id,
                self.root / library_id,
                temporary,
                cleanup_id=cleanup_id,
            )
            previous = self._stage_single_purge(
                self.root / library_id, cleanup_id=cleanup_id
            )
            replacement = self.root / library_id
            try:
                if temporary.exists():
                    os.replace(temporary, replacement)
            except OSError as error:
                previous.rollback()
                stale_replacements.rollback()
                raise GraphAdapterError(
                    "Milvus Lite graph replacement could not be staged"
                ) from error
            return _LockedGraphPurge(
                (self._operation_lock, lock),
                _CompositeGraphPurge(
                    (stale_replacements,),
                    (_FilesystemGraphReplacement(replacement, previous),),
                ),
            )

    @staticmethod
    def _validate_cleanup_id(cleanup_id: str) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", cleanup_id) is None:
            raise GraphAdapterError("invalid graph cleanup identifier")

    def _purge_path(self, library_id: str, cleanup_id: str | None, role: str) -> Path:
        suffix = uuid.uuid4().hex
        if cleanup_id is None:
            return self.root / f".{library_id}.purge-{suffix}"
        self._validate_cleanup_id(cleanup_id)
        return self.root / f".{library_id}.purge-{cleanup_id}-{role}-{suffix}"

    def reconcile_staged_purge(
        self, library_id: str, cleanup_id: str, *, committed: bool
    ) -> None:
        self._validate_cleanup_id(cleanup_id)
        prefix = f".{library_id}.purge-{cleanup_id}-"
        lock = self._mutation_lock(library_id)
        with self._operation_lock, lock:
            try:
                candidates = tuple(self.root.iterdir()) if self.root.exists() else ()
            except OSError as error:
                raise GraphAdapterError("Milvus Lite graph cleanup scan failed") from error
            staged = tuple(
                candidate for candidate in candidates if candidate.name.startswith(prefix)
            )
            for candidate in staged:
                if candidate.is_symlink() or not candidate.is_dir():
                    raise GraphAdapterError("Milvus Lite graph cleanup target is unsafe")
            if committed:
                for candidate in staged:
                    _remove_tree_recoverably(candidate)
                return
            official = tuple(
                candidate for candidate in staged if candidate.name.startswith(prefix + "official-")
            )
            disposable = tuple(
                candidate
                for candidate in staged
                if candidate.name.startswith(prefix + "disposable-")
            )
            if len(official) > 1 or len(official) + len(disposable) != len(staged):
                raise GraphAdapterError("Milvus Lite graph cleanup state is inconsistent")
            live = self.root / library_id
            if official:
                if live.exists():
                    _remove_tree_recoverably(live)
                try:
                    os.replace(official[0], live)
                except OSError as error:
                    raise GraphAdapterError(
                        "Milvus Lite graph cleanup recovery failed"
                    ) from error
            for candidate in disposable:
                restored = self.root / f".{library_id}.replacement-recovered-{uuid.uuid4().hex}"
                try:
                    os.replace(candidate, restored)
                except OSError as error:
                    raise GraphAdapterError(
                        "Milvus Lite graph cleanup recovery failed"
                    ) from error

    async def prepare_rebuild(
        self, library_id: str, documents: tuple[GraphSourceDocument, ...]
    ) -> PreparedGraphRebuild:
        extracted = await asyncio.to_thread(self._extract_documents, documents)
        return _PreparedFilesystemGraphRebuild(self, library_id, extracted)

    async def rebuild(self, library_id: str, documents: tuple[GraphSourceDocument, ...]) -> None:
        extracted = await asyncio.to_thread(self._extract_documents, documents)
        try:
            await asyncio.to_thread(self._replace_projection, library_id, extracted)
        except GraphAdapterError:
            raise
        except Exception as error:
            raise GraphAdapterError("Milvus Lite graph rebuild failed") from error

    def _extract_documents(
        self, documents: tuple[GraphSourceDocument, ...]
    ) -> list[tuple[GraphSourceDocument, dict[str, object]]]:
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]] = []
        if documents and self.model_client.graph is None:
            raise GraphAdapterError("graph LLM is not configured")
        for document in documents:
            try:
                payload = self.model_client.extract_graph(document.content)
            except ModelServiceError as error:
                raise GraphAdapterError("graph LLM extraction failed") from error
            extracted.append((document, payload))
        return extracted

    async def expand(
        self,
        library_id: str,
        seed_document_ids: tuple[str, ...],
        *,
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        if not seed_document_ids or limit <= 0:
            return []
        bounded_hops = max(1, min(max_hops, 2))
        try:
            return await asyncio.to_thread(
                self._expand_projection,
                library_id,
                set(seed_document_ids),
                bounded_hops,
                limit,
            )
        except GraphAdapterBusyError:
            raise
        except Exception as error:
            raise GraphAdapterError("Milvus Lite graph query failed") from error

    def _replace_projection(
        self,
        library_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
        graph_library_id: str | None = None,
    ) -> None:
        with self._operation_lock, self._mutation_lock(graph_library_id or library_id):
            self._replace_projection_locked(library_id, extracted, graph_library_id)

    def _replace_projection_locked(
        self,
        library_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
        graph_library_id: str | None,
    ) -> None:
        library_root = self.root / library_id
        database = library_root / "graph.db"
        marker = library_root / self._PROJECTION_MARKER
        if not extracted:
            marker.unlink(missing_ok=True)
            database.unlink(missing_ok=True)
            return
        try:
            from jiuwen_memory.common.logging.log_config import (  # type: ignore[import-untyped]
                configure_log_config,
            )

            configure_log_config(
                {
                    "backend": "default",
                    "output": ["console"],
                    "interface_output": ["console"],
                    "performance_output": ["console"],
                }
            )
            from jiuwen_memory.foundation.store.graph import (  # type: ignore[import-untyped]
                Entity,
                Episode,
                Relation,
            )
        except ImportError as error:
            raise GraphAdapterError("JiuwenMemory or Milvus Lite is unavailable") from error

        library_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        marker.unlink(missing_ok=True)

        store = self._open_projection(database)
        try:
            store.rebuild()
            entities, relations, episodes = self._graph_objects(
                graph_library_id or library_id, extracted, Entity, Relation, Episode
            )
            self._write_projection_blocking(store, entities, relations, episodes)
        finally:
            store.close()
        temporary_marker = marker.with_suffix(".tmp")
        temporary_marker.write_text("ready\n", encoding="ascii")
        temporary_marker.replace(marker)

    def _write_projection_blocking(
        self,
        store: Any,
        entities: list[Any],
        relations: list[Any],
        episodes: list[Any],
    ) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._write_projection(store, entities, relations, episodes))
            return

        errors: list[BaseException] = []

        def write_projection() -> None:
            try:
                asyncio.run(self._write_projection(store, entities, relations, episodes))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=write_projection, name="graph-projection-write")
        thread.start()
        thread.join()
        if errors:
            raise errors[0]

    @classmethod
    def _graph_objects(
        cls,
        library_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
        entity_type: Any,
        relation_type: Any,
        episode_type: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        mentions: dict[str, list[tuple[GraphSourceDocument, dict[str, str]]]] = {}
        parsed: list[tuple[GraphSourceDocument, list[dict[str, str]], list[dict[str, str]]]] = []
        for document, payload in extracted:
            entities = cls._entities(payload)
            relations = cls._relations(payload)
            parsed.append((document, entities, relations))
            seen_keys: set[str] = set()
            for entity_data in entities:
                key = cls._entity_key(entity_data["name"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                mentions.setdefault(key, []).append((document, entity_data))

        graph_entities: dict[str, Any] = {}
        for key in sorted(mentions):
            source_mentions = sorted(
                mentions[key], key=lambda item: (item[0].path, item[0].document_id)
            )
            first = source_mentions[0][1]
            graph_entities[key] = entity_type(
                uuid=cls._object_id("entity", key),
                user_id=library_id,
                name=first["name"],
                content=first["content"],
                metadata={
                    "library_id": library_id,
                    "entity_key": key,
                    "sources": [
                        {
                            "document_id": document.document_id,
                            "path": document.path,
                            "source_version": document.source_version,
                            "content": entity_data["content"],
                        }
                        for document, entity_data in source_mentions
                    ],
                },
                language="en",
                content_embedding=cls._placeholder_vector(),
                name_embedding=cls._placeholder_vector(),
            )

        graph_relations: list[Any] = []
        graph_episodes: list[Any] = []
        for document, entities, relations in parsed:
            document_keys = {
                cls._entity_key(entity_data["name"])
                for entity_data in entities
                if cls._entity_key(entity_data["name"]) in graph_entities
            }
            episode = episode_type(
                uuid=cls._object_id("episode", document.document_id),
                user_id=library_id,
                content=document.content,
                entities=[graph_entities[key].uuid for key in sorted(document_keys)],
                metadata={
                    "library_id": library_id,
                    "document_id": document.document_id,
                    "path": document.path,
                    "source_version": document.source_version,
                },
                language="en",
                content_embedding=cls._placeholder_vector(),
            )
            graph_episodes.append(episode)
            for key in document_keys:
                graph_entities[key].episodes.append(episode.uuid)
            for relation_data in relations:
                lhs_key = cls._entity_key(relation_data["source"])
                rhs_key = cls._entity_key(relation_data["target"])
                if lhs_key not in document_keys or rhs_key not in document_keys:
                    continue
                relation = relation_type(
                    uuid=cls._object_id(
                        "relation",
                        document.document_id,
                        lhs_key,
                        rhs_key,
                        relation_data["content"],
                    ),
                    user_id=library_id,
                    name="related",
                    content=relation_data["content"],
                    lhs=graph_entities[lhs_key],
                    rhs=graph_entities[rhs_key],
                    metadata={
                        "library_id": library_id,
                        "document_id": document.document_id,
                        "path": document.path,
                        "source_version": document.source_version,
                    },
                    language="en",
                    content_embedding=cls._placeholder_vector(),
                ).update_connected_entities()
                graph_relations.append(relation)
        return list(graph_entities.values()), graph_relations, graph_episodes

    @staticmethod
    async def _write_projection(
        store: Any, entities: list[Any], relations: list[Any], episodes: list[Any]
    ) -> None:
        await store.add_entity(entities, flush=False, no_embed=True)
        await store.add_relation(relations, flush=False, no_embed=True)
        await store.add_episode(episodes, flush=False, no_embed=True)
        await store.refresh()

    def _expand_projection(
        self,
        library_id: str,
        seed_documents: set[str],
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        if not self._operation_lock.acquire(blocking=False):
            raise GraphAdapterBusyError("Milvus Lite graph projection is busy")
        try:
            with self._mutation_lock(library_id):
                return self._expand_projection_locked(
                    library_id, seed_documents, max_hops, limit
                )
        finally:
            self._operation_lock.release()

    def _expand_projection_locked(
        self,
        library_id: str,
        seed_documents: set[str],
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        library_root = self.root / library_id
        database = library_root / "graph.db"
        marker = library_root / self._PROJECTION_MARKER
        if not database.exists():
            if marker.exists():
                raise GraphAdapterError("graph projection database is missing")
            return []
        if not marker.exists():
            raise GraphAdapterError("graph projection is incomplete")
        store = self._open_projection(database)
        try:
            return asyncio.run(
                self._query_projection(store, library_id, seed_documents, max_hops, limit)
            )
        finally:
            store.close()

    @staticmethod
    def _open_projection(database: Path) -> Any:
        try:
            from jiuwen_memory.common.logging.log_config import (
                configure_log_config,
            )

            configure_log_config(
                {
                    "backend": "default",
                    "output": ["console"],
                    "interface_output": ["console"],
                    "performance_output": ["console"],
                }
            )
            from jiuwen_memory.foundation.store.graph import (
                Entity,
                Episode,
                GraphConfig,
                GraphStoreIndexConfig,
                GraphStoreStorageConfig,
                Relation,
            )
            from jiuwen_memory.foundation.store.graph.constants import (  # type: ignore[import-untyped]
                ENTITY_COLLECTION,
                EPISODE_COLLECTION,
                RELATION_COLLECTION,
            )
            from jiuwen_memory.foundation.store.graph.index_field import (  # type: ignore[import-untyped]
                MilvusFLAT,
            )
            from jiuwen_memory.foundation.store.graph.milvus import (  # type: ignore[import-untyped]
                MilvusGraphStore,
            )
            from pymilvus import MilvusClient  # type: ignore[import-untyped]
        except ImportError as error:
            raise GraphAdapterError("JiuwenMemory or Milvus Lite is unavailable") from error

        class MilvusLiteGraphStore(MilvusGraphStore):  # type: ignore[misc]
            """Jiuwen graph store with unsupported Lite database RPCs removed."""

            def __init__(self, config: Any) -> None:
                extras = config.extras.copy()
                self._config = config
                self._embedder = None
                self._local_uri = config.uri
                self.alias = extras.setdefault("alias", f"pam-graph-{id(self)}")
                self.client: Any = None
                try:
                    self.client = MilvusClient(
                        uri=config.uri,
                        token=config.token,
                        timeout=config.timeout,
                        **extras,
                    )
                    self.metric = (
                        config.db_embed_config.distance_metric.replace("dot", "ip")
                        .replace("euclidean", "l2")
                        .upper()
                    )
                    self.full_text_search_params = MappingProxyType({"metric_type": "BM25"})
                    self.dense_search_params = MappingProxyType({"metric_type": self.metric})
                    self.field_def = {
                        ENTITY_COLLECTION: [
                            key for key in Entity.model_fields if not key.endswith("_bm25")
                        ],
                        RELATION_COLLECTION: [
                            key for key in Relation.model_fields if not key.endswith("_bm25")
                        ],
                        EPISODE_COLLECTION: [
                            key for key in Episode.model_fields if not key.endswith("_bm25")
                        ],
                    }
                    self._build_lite_indices()
                except BaseException:
                    self._close_local_store()
                    raise

            def close(self) -> None:
                self._close_local_store()

            def _close_local_store(self) -> None:
                client = self.client
                self.client = None
                if client is not None:
                    with suppress(Exception):
                        client.close()
                from milvus_lite.server_manager import (  # type: ignore[import-untyped]
                    server_manager_instance,
                )

                with suppress(Exception):
                    server_manager_instance.release_server(self._local_uri)

            def _build_lite_indices(self) -> None:
                from jiuwen_memory.foundation.store.graph.milvus import (
                    generate_milvus_schema,
                )

                original = generate_milvus_schema.icu_analyzer
                original_stopwords = generate_milvus_schema.icu_analyzer_with_stopwords
                standard = {"tokenizer": "standard", "filter": ["lowercase"]}
                try:
                    generate_milvus_schema.icu_analyzer = standard
                    generate_milvus_schema.icu_analyzer_with_stopwords = standard
                    self._build_indices()
                finally:
                    generate_milvus_schema.icu_analyzer = original
                    generate_milvus_schema.icu_analyzer_with_stopwords = original_stopwords

            def rebuild(self) -> None:
                for collection in self.client.list_collections():
                    self.client.drop_collection(collection, timeout=self.config.timeout)
                self._build_lite_indices()

        config = GraphConfig(
            uri=str(database),
            name="default",
            embed_dim=32,
            db_storage_config=GraphStoreStorageConfig(uuid=64, user_id=64),
            db_embed_config=GraphStoreIndexConfig(
                index_type=MilvusFLAT(),
                distance_metric="cosine",
                bm25_analyzer_settings={"tokenizer": "standard", "filter": ["lowercase"]},
            ),
        )
        return MilvusLiteGraphStore(config)

    @classmethod
    async def _query_projection(
        cls,
        store: Any,
        library_id: str,
        seed_documents: set[str],
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        from jiuwen_memory.foundation.store.graph.constants import (
            ENTITY_COLLECTION,
            RELATION_COLLECTION,
        )
        from jiuwen_memory.foundation.store.graph.result_ranking import (  # type: ignore[import-untyped]
            WeightedRankConfig,
        )

        if store.is_empty(ENTITY_COLLECTION):
            return []
        entity_count = cls._complete_collection_size(store, ENTITY_COLLECTION)
        rows = await store.query(
            ENTITY_COLLECTION,
            limit=entity_count,
            output_fields=["uuid", "name", "metadata"],
        )
        cls._require_complete_rows(ENTITY_COLLECTION, rows, entity_count)
        entities = {str(row.get("uuid", "")): row for row in rows if row.get("uuid")}
        seed_entities = {
            entity_id
            for entity_id, row in entities.items()
            if any(
                str(source.get("document_id", "")) in seed_documents
                for source in cls._entity_sources(row, library_id)
            )
        }
        if not seed_entities:
            return []
        seed_query = " ".join(
            str(entities[entity_id].get("name", "")) for entity_id in sorted(seed_entities)
        )
        searched = await store.search(
            seed_query,
            entity_count,
            ENTITY_COLLECTION,
            WeightedRankConfig(name_dense=1, content_dense=0, content_sparse=0),
            bfs_depth=max_hops,
            bfs_k=entity_count,
            query_embedding=cls._placeholder_vector(),
            output_fields=["uuid", "name", "metadata"],
        )

        raw_candidates = searched.get(ENTITY_COLLECTION, [])
        if not raw_candidates:
            return []
        cls._require_complete_rows(ENTITY_COLLECTION, raw_candidates, entity_count)
        candidate_entities = {
            str(row.get("uuid", "")) for row in raw_candidates if row.get("uuid") in entities
        }
        if len(candidate_entities) != entity_count:
            raise GraphAdapterError("graph search returned an incomplete entity set")

        distance = {entity_id: 0 for entity_id in seed_entities}
        frontier = set(seed_entities)
        if store.is_empty(RELATION_COLLECTION):
            relations: list[dict[str, object]] = []
        else:
            relation_count = cls._complete_collection_size(store, RELATION_COLLECTION)
            relations = await store.query(
                RELATION_COLLECTION,
                limit=relation_count,
                output_fields=["lhs", "rhs"],
            )
            cls._require_complete_rows(RELATION_COLLECTION, relations, relation_count)
        adjacency: dict[str, set[str]] = {}
        for relation in relations:
            lhs, rhs = relation.get("lhs"), relation.get("rhs")
            if not isinstance(lhs, str) or not isinstance(rhs, str):
                continue
            if lhs not in candidate_entities or rhs not in candidate_entities:
                continue
            adjacency.setdefault(lhs, set()).add(rhs)
            adjacency.setdefault(rhs, set()).add(lhs)
        for depth in range(1, max_hops):
            frontier = {
                neighbor
                for entity_id in frontier
                for neighbor in adjacency.get(entity_id, ())
                if neighbor not in distance
            }
            for entity_id in frontier:
                distance[entity_id] = depth
            if not frontier:
                break

        seen_documents = set(seed_documents)
        results: list[GraphExpansion] = []
        for hop in range(1, max_hops + 1):
            candidates: list[tuple[str, dict[str, str]]] = []
            for entity_id in sorted(
                entity_id for entity_id, depth in distance.items() if depth + 1 == hop
            ):
                row = entities.get(entity_id, {})
                metadata = row.get("metadata")
                anchor = ""
                if isinstance(metadata, dict):
                    anchor = str(metadata.get("entity_key", ""))
                for source in cls._entity_sources(row, library_id):
                    candidates.append((anchor, source))
            for anchor, source in sorted(
                candidates,
                key=lambda item: (item[1]["path"], item[1]["document_id"], item[0]),
            ):
                document_id = source["document_id"]
                if document_id in seen_documents:
                    continue
                seen_documents.add(document_id)
                results.append(
                    GraphExpansion(
                        content=source["content"],
                        document_id=document_id,
                        path=source["path"],
                        source_version=source["source_version"],
                        hop=hop,
                        graph_object_type="entity",
                        source_anchor=anchor,
                    )
                )
                if len(results) >= limit:
                    return results
        return results

    @classmethod
    def _complete_collection_size(cls, store: Any, collection: str) -> int:
        try:
            raw_count = store.client.get_collection_stats(collection).get("row_count")
            count = int(raw_count)
        except (AttributeError, TypeError, ValueError) as error:
            raise GraphAdapterError("graph collection size is unavailable") from error
        if count <= 0:
            raise GraphAdapterError("graph collection size is invalid")
        if count > cls._MILVUS_COMPLETE_READ_LIMIT:
            raise GraphAdapterError(f"graph collection exceeds complete read limit: {collection}")
        return count

    @staticmethod
    def _require_complete_rows(
        collection: str, rows: list[dict[str, object]], expected: int
    ) -> None:
        if len(rows) != expected:
            raise GraphAdapterError(f"graph collection read was incomplete: {collection}")

    @staticmethod
    def _entity_sources(row: dict[str, object], library_id: str) -> list[dict[str, str]]:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("library_id") != library_id:
            return []
        raw_sources = metadata.get("sources")
        if not isinstance(raw_sources, list):
            return []
        sources: list[dict[str, str]] = []
        for raw in raw_sources:
            if not isinstance(raw, dict):
                continue
            document_id = raw.get("document_id")
            path = raw.get("path")
            source_version = raw.get("source_version")
            if not all(
                isinstance(value, str) and value for value in (document_id, path, source_version)
            ):
                continue
            sources.append(
                {
                    "document_id": str(document_id),
                    "path": str(path),
                    "source_version": str(source_version),
                    "content": str(raw.get("content", "")),
                }
            )
        return sources

    @staticmethod
    def _entities(payload: dict[str, object]) -> list[dict[str, str]]:
        raw = payload.get("entities", [])
        if not isinstance(raw, list):
            raise GraphAdapterError("graph LLM returned invalid entities")
        result: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise GraphAdapterError("graph LLM returned invalid entities")
            name, content = item.get("name"), item.get("content")
            if not isinstance(name, str) or not name.strip():
                raise GraphAdapterError("graph LLM returned an invalid entity name")
            result.append(
                {"name": name.strip()[:500], "content": str(content or name).strip()[:65535]}
            )
        return result

    @staticmethod
    def _relations(payload: dict[str, object]) -> list[dict[str, str]]:
        raw = payload.get("relations", [])
        if not isinstance(raw, list):
            raise GraphAdapterError("graph LLM returned invalid relations")
        result: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise GraphAdapterError("graph LLM returned invalid relations")
            source, target, content = item.get("source"), item.get("target"), item.get("content")
            if not all(isinstance(value, str) and value.strip() for value in (source, target)):
                raise GraphAdapterError("graph LLM returned an invalid relation")
            result.append(
                {
                    "source": str(source).strip()[:500],
                    "target": str(target).strip()[:500],
                    "content": str(content or "related").strip()[:65535],
                }
            )
        return result

    @staticmethod
    def _placeholder_vector() -> list[float]:
        return [1.0] + [0.0] * 31

    @staticmethod
    def _entity_key(value: str) -> str:
        normalized = re.sub(r"\s+", " ", value.strip().casefold())
        return normalized[:512]

    @staticmethod
    def _object_id(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()
