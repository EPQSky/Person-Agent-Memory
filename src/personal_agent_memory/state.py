from __future__ import annotations

import asyncio
import ctypes
import difflib
import errno
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from personal_agent_memory.git_history import (
    GitHistoryError,
    GitInitialization,
    GitRepository,
    allowed_markdown_path,
)
from personal_agent_memory.graph_adapter import (
    GraphAdapter,
    GraphAdapterError,
    GraphSourceDocument,
)
from personal_agent_memory.markdown_index import (
    MarkdownDocument,
    MarkdownScan,
    StoredChunk,
    chunk_markdown,
    fts_query,
    infer_legacy_chunk_kind,
    match_chunk_ids,
    scan_markdown,
    scan_markdown_fd,
    stable_document_id,
    validate_ignore_patterns,
)
from personal_agent_memory.model_client import (
    ModelServiceError,
    OpenAICompatibleClient,
    cosine_similarity,
)

LibraryKind = Literal["user", "project"]
BindingKind = Literal["project", "worktree"]
RENAME_EXCHANGE = 2


def _rename_exchange(
    source_dir_fd: int,
    source: str,
    destination_dir_fd: int,
    destination: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_dir_fd,
        os.fsencode(source),
        destination_dir_fd,
        os.fsencode(destination),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


class LibraryRegistrationError(ValueError):
    """Raised when a path cannot safely become a memory library."""


class DuplicateLibraryError(LibraryRegistrationError):
    def __init__(self, library_id: str) -> None:
        self.library_id = library_id
        super().__init__("canonical path is already registered")


class ProjectBindingError(ValueError):
    """Raised when a project binding cannot be changed safely."""


class DuplicateProjectBindingError(ProjectBindingError):
    def __init__(self, binding_id: str) -> None:
        self.binding_id = binding_id
        super().__init__("canonical project root is already bound")


class MemoryMutationError(ValueError):
    """Raised when an authoritative memory mutation cannot be completed safely."""


class CandidateGovernanceError(ValueError):
    """Raised when candidate memory governance cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class BoundDocument:
    root_identity: tuple[int, int]
    root_fd: int
    parent_fd: int
    name: str
    parent_identity: tuple[int, int]


@dataclass(slots=True)
class DocumentReplacement:
    identity: tuple[int, int] | None = None
    renamed: bool = False
    expected_target_exchanged: bool = False


@dataclass(frozen=True, slots=True)
class LibraryIndexSnapshot:
    sync_status: str
    documents: tuple[tuple[object, ...], ...]
    chunks: tuple[tuple[object, ...], ...]
    searches: tuple[tuple[object, ...], ...]
    vectors: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class MemoryLibrary:
    id: str
    kind: LibraryKind
    canonical_path: str
    availability: str
    sync_status: str
    graph_status: str
    graph_progress: str
    graph_last_error: str

    def payload(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectBinding:
    id: str
    project_id: str
    kind: BindingKind
    canonical_root: str
    library_id: str
    availability: str

    def payload(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateMemory:
    id: str
    library_id: str
    suggested_type: str
    body: str
    source_references: tuple[str, ...]
    creator: str
    created_at: str
    status: str
    operator: str | None
    reason: str | None
    reviewed_at: str | None
    published_path: str | None
    commit: str | None

    def payload(self) -> dict[str, object]:
        return asdict(self)


class PlatformState:
    def __init__(
        self,
        database_path: Path,
        library_roots: tuple[Path, ...] = (),
        model_client: OpenAICompatibleClient | None = None,
        graph_adapter: GraphAdapter | None = None,
    ) -> None:
        self.database_path = database_path
        self.library_roots = library_roots
        self.connection: sqlite3.Connection | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self.stop_worker = asyncio.Event()
        self.startup_count = 0
        self.previous_shutdown_clean = True
        self.model_client = model_client or OpenAICompatibleClient(None, None)
        self.graph_adapter = graph_adapter

    async def start(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.database_path)
        self.connection = connection
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity != ("ok",):
                raise sqlite3.DatabaseError(f"state database integrity check failed: {integrity!r}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS platform_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS background_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS memory_libraries (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK(kind IN ('user', 'project')),
                    canonical_path TEXT NOT NULL UNIQUE,
                    sync_status TEXT NOT NULL DEFAULT 'not_started',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS memory_documents (
                    id TEXT PRIMARY KEY,
                    library_id TEXT NOT NULL REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    UNIQUE(library_id, path)
                );
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES memory_documents(id) ON DELETE CASCADE,
                    library_id TEXT NOT NULL REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    heading TEXT,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    source_version TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunk_search USING fts5(
                    chunk_id UNINDEXED, content, heading, path, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS memory_chunk_vectors (
                    chunk_id TEXT PRIMARY KEY REFERENCES memory_chunks(id) ON DELETE CASCADE,
                    library_id TEXT NOT NULL REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    source_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS memory_vector_indexes (
                    library_id TEXT PRIMARY KEY
                        REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL CHECK(dimension >= 0),
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS memory_graph_indexes (
                    library_id TEXT PRIMARY KEY
                        REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'not_configured',
                    total_documents INTEGER NOT NULL DEFAULT 0,
                    projected_documents INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS memory_graph_documents (
                    library_id TEXT NOT NULL
                        REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    PRIMARY KEY(library_id, document_id)
                );
                CREATE TABLE IF NOT EXISTS project_bindings (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK(kind IN ('project', 'worktree')),
                    canonical_root TEXT NOT NULL UNIQUE,
                    root_device INTEGER NOT NULL,
                    root_inode INTEGER NOT NULL,
                    library_id TEXT REFERENCES memory_libraries(id),
                    main_project_binding_id TEXT REFERENCES project_bindings(id)
                        ON DELETE CASCADE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK(
                        (kind = 'project' AND library_id IS NOT NULL
                            AND main_project_binding_id IS NULL)
                        OR
                        (kind = 'worktree' AND library_id IS NULL
                            AND main_project_binding_id IS NOT NULL)
                    )
                );
                CREATE TABLE IF NOT EXISTS library_git_repositories (
                    library_id TEXT PRIMARY KEY REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    git_dir TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('sidecar', 'authorized_existing'))
                );
                CREATE TABLE IF NOT EXISTS memory_operations (
                    operation_id TEXT PRIMARY KEY,
                    library_id TEXT NOT NULL REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK(kind IN ('edit', 'restore')),
                    document_path TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    commit_id TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS candidate_memories (
                    id TEXT PRIMARY KEY,
                    library_id TEXT NOT NULL REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    suggested_type TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source_references_json TEXT NOT NULL,
                    creator TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'approved', 'rejected')),
                    operator TEXT,
                    reason TEXT,
                    reviewed_at TEXT,
                    published_path TEXT,
                    commit_id TEXT,
                    decision_operation_id TEXT,
                    decision_request_hash TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(library_id, idempotency_key),
                    UNIQUE(decision_operation_id)
                );
                CREATE TABLE IF NOT EXISTS candidate_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL REFERENCES candidate_memories(id) ON DELETE CASCADE,
                    action TEXT NOT NULL
                        CHECK(action IN ('created', 'edited', 'approved', 'rejected')),
                    operator TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS capture_inbox (
                    event_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    library_id TEXT NOT NULL REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    turn_id TEXT NOT NULL,
                    event_kind TEXT NOT NULL CHECK(event_kind IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    consolidated_at TEXT,
                    UNIQUE(session_id, turn_id, event_kind)
                );
                CREATE TABLE IF NOT EXISTS capture_rounds (
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    library_id TEXT NOT NULL REFERENCES memory_libraries(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'processing', 'done', 'error')),
                    candidate_id TEXT REFERENCES candidate_memories(id),
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(session_id, turn_id)
                );
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(memory_libraries)")
            }
            if "ignore_patterns" not in columns:
                connection.execute(
                    """ALTER TABLE memory_libraries
                       ADD COLUMN ignore_patterns TEXT NOT NULL DEFAULT '[]'"""
                )
            chunk_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(memory_chunks)")
            }
            if "kind" not in chunk_columns:
                connection.execute(
                    "ALTER TABLE memory_chunks ADD COLUMN kind TEXT NOT NULL DEFAULT 'paragraph'"
                )
                legacy_chunks = connection.execute(
                    "SELECT id, content FROM memory_chunks"
                ).fetchall()
                connection.executemany(
                    "UPDATE memory_chunks SET kind = ? WHERE id = ?",
                    [
                        (infer_legacy_chunk_kind(str(row[1])), str(row[0]))
                        for row in legacy_chunks
                    ],
                )
            operation_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(memory_operations)")
            }
            if "request_hash" not in operation_columns:
                connection.execute(
                    "ALTER TABLE memory_operations ADD COLUMN request_hash TEXT NOT NULL DEFAULT ''"
                )
            job_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(background_jobs)")
            }
            if "attempts" not in job_columns:
                connection.execute(
                    "ALTER TABLE background_jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
                )
            if "available_at" not in job_columns:
                connection.execute(
                    "ALTER TABLE background_jobs ADD COLUMN available_at REAL NOT NULL DEFAULT 0"
                )
            connection.execute(
                """UPDATE background_jobs SET status = 'pending'
                   WHERE kind = 'capture_consolidation' AND status = 'running'"""
            )
            connection.execute(
                """UPDATE capture_rounds SET status = 'pending', updated_at = CURRENT_TIMESTAMP
                   WHERE status = 'processing'"""
            )
            previous_count = self._metadata("startup_count")
            previous_clean = self._metadata("clean_shutdown")
            self.startup_count = int(previous_count or "0") + 1
            self.previous_shutdown_clean = previous_clean != "false"
            self._set_metadata("startup_count", str(self.startup_count))
            self._set_metadata("clean_shutdown", "false")
            connection.commit()
        except BaseException:
            connection.close()
            self.connection = None
            raise
        self.stop_worker.clear()
        self.worker_task = asyncio.create_task(self._worker(), name="memory-background-worker")

    async def close(self) -> None:
        if self.worker_task is not None:
            self.stop_worker.set()
            await self.worker_task
        if self.connection is not None:
            self._set_metadata("clean_shutdown", "true")
            self.connection.commit()
            self.connection.close()
            self.connection = None

    async def _worker(self) -> None:
        while not self.stop_worker.is_set():
            await self._execute_pending_jobs()
            try:
                await asyncio.wait_for(self.stop_worker.wait(), timeout=0.1)
            except TimeoutError:
                continue

    def enqueue_job(self, kind: str, payload: str) -> int:
        assert self.connection is not None
        cursor = self.connection.execute(
            "INSERT INTO background_jobs (kind, payload) VALUES (?, ?)", (kind, payload)
        )
        self.connection.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def enqueue_vector_rebuild(self, library_id: str) -> int:
        if self.library(library_id) is None:
            raise LibraryRegistrationError("memory library not found")
        if self.model_client.embedding is None:
            raise LibraryRegistrationError("embedding is not configured")
        return self._queue_vector_rebuild(library_id, False)

    def enqueue_graph_rebuild(self, library_id: str) -> int:
        if self.library(library_id) is None:
            raise LibraryRegistrationError("memory library not found")
        if self.graph_adapter is None or self.model_client.graph is None:
            raise LibraryRegistrationError("graph projection is not configured")
        return self._queue_graph_rebuild(library_id, False)

    def job_status(self, job_id: int) -> str | None:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT status FROM background_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def register_library(
        self, requested_path: str, kind: LibraryKind, reuse_existing_git: bool = False
    ) -> MemoryLibrary:
        assert self.connection is not None
        if not requested_path.strip():
            raise LibraryRegistrationError("memory library path must not be empty")
        requested = Path(requested_path).expanduser()
        if requested.is_symlink() and not requested.exists():
            raise LibraryRegistrationError(
                "memory library path cannot be resolved: dangling or cyclic symbolic link"
            )
        try:
            path = requested.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise LibraryRegistrationError(
                f"memory library path cannot be resolved: {error}"
            ) from error
        self._validate_library_path(path)
        canonical_path = os.path.normcase(str(path))
        duplicate = self.connection.execute(
            "SELECT id FROM memory_libraries WHERE canonical_path = ?", (canonical_path,)
        ).fetchone()
        if duplicate is not None:
            raise DuplicateLibraryError(str(duplicate[0]))

        if path.exists():
            if not path.is_dir():
                raise LibraryRegistrationError("memory library path must be a directory")
        else:
            try:
                path.mkdir(parents=True, mode=0o700)
            except OSError as error:
                raise LibraryRegistrationError(
                    f"memory library directory could not be created: {error.strerror or error}"
                ) from error
        self._require_accessible_directory(path)

        library_id = str(uuid.uuid4())
        repository: GitRepository | None = None
        initialization: GitInitialization | None = None
        root_fd = -1
        try:
            root_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            root_metadata = os.fstat(root_fd)
            root_identity = root_metadata.st_dev, root_metadata.st_ino
            registration_scan = scan_markdown_fd(
                root_fd, path, (), (self.database_path.parent,)
            )
            self._verify_registration_tree(
                path, root_fd, root_identity, registration_scan, None
            )
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "INSERT INTO memory_libraries (id, kind, canonical_path) VALUES (?, ?, ?)",
                (library_id, kind, canonical_path),
            )
            # An incomplete corpus remains registerable with an error sync status.
            with suppress(LibraryRegistrationError):
                self.scan_library(
                    library_id,
                    participate_in_transaction=True,
                    frozen_scan=registration_scan,
                )
            repository, initialization, mode = self._prepare_git_repository(
                library_id,
                reuse_existing_git,
                path,
                root_fd,
                root_identity,
                registration_scan,
            )
            manifest = (
                initialization.manifest_content
                if mode == "authorized_existing"
                else None
            )
            self._verify_registration_tree(
                path, root_fd, root_identity, registration_scan, manifest
            )
            self.connection.execute(
                """INSERT INTO library_git_repositories (library_id, git_dir, mode)
                   VALUES (?, ?, ?)""",
                (library_id, str(repository.git_dir), mode),
            )
            self._verify_registration_tree(
                path, root_fd, root_identity, registration_scan, manifest
            )
            self.connection.commit()
            self._verify_registration_tree(
                path, root_fd, root_identity, registration_scan, manifest
            )
        except BaseException as error:
            self.connection.rollback()
            cleanup_errors: list[str] = []
            try:
                persisted = self.connection.execute(
                    "SELECT 1 FROM memory_libraries WHERE id = ?", (library_id,)
                ).fetchone()
                if persisted is not None:
                    self.connection.execute("BEGIN IMMEDIATE")
                    chunk_ids = tuple(
                        str(row[0])
                        for row in self.connection.execute(
                            "SELECT id FROM memory_chunks WHERE library_id = ?",
                            (library_id,),
                        )
                    )
                    if chunk_ids:
                        placeholders = ",".join("?" for _ in chunk_ids)
                        self.connection.execute(
                            f"DELETE FROM memory_chunk_search "
                            f"WHERE chunk_id IN ({placeholders})",  # noqa: S608
                            chunk_ids,
                        )
                    self.connection.execute(
                        "DELETE FROM memory_libraries WHERE id = ?", (library_id,)
                    )
                    self.connection.commit()
            except BaseException as caught:
                self.connection.rollback()
                cleanup_errors.append(f"database cleanup: {caught}")
            if repository is not None and initialization is not None:
                try:
                    repository.rollback_initialization(initialization)
                except BaseException as caught:
                    cleanup_errors.append(f"history cleanup: {caught}")
            if isinstance(error, DuplicateLibraryError):
                raise
            detail = str(error)
            if cleanup_errors:
                detail += f"; rollback failed: {'; '.join(cleanup_errors)}"
            raise LibraryRegistrationError(
                f"memory library registration could not be completed: {detail}"
            ) from error
        finally:
            if root_fd >= 0:
                os.close(root_fd)
        library = self.library(library_id)
        assert library is not None
        return library

    def list_libraries(self) -> list[MemoryLibrary]:
        assert self.connection is not None
        rows = self.connection.execute(
            """SELECT library.id, library.kind, library.canonical_path,
                      library.sync_status, COALESCE(graph.status, 'not_configured'),
                      COALESCE(graph.projected_documents, 0),
                      COALESCE(graph.total_documents, 0), COALESCE(graph.last_error, '')
               FROM memory_libraries AS library
               LEFT JOIN memory_graph_indexes AS graph ON graph.library_id = library.id
               ORDER BY library.created_at, library.id"""
        ).fetchall()
        return [
            MemoryLibrary(
                id=str(row[0]),
                kind=row[1],
                canonical_path=str(row[2]),
                availability=self._availability(Path(str(row[2]))),
                sync_status=str(row[3]),
                graph_status=str(row[4]),
                graph_progress=f"{int(row[5])}/{int(row[6])}",
                graph_last_error=str(row[7]),
            )
            for row in rows
        ]

    def library(self, library_id: str) -> MemoryLibrary | None:
        return next((item for item in self.list_libraries() if item.id == library_id), None)

    def library_ignore_patterns(self, library_id: str) -> tuple[str, ...]:
        row = self.connection_or_raise.execute(
            "SELECT ignore_patterns FROM memory_libraries WHERE id = ?", (library_id,)
        ).fetchone()
        if row is None:
            raise LibraryRegistrationError("memory library not found")
        value = json.loads(str(row[0]))
        return tuple(str(item) for item in value)

    def update_library_ignore_patterns(
        self, library_id: str, patterns: tuple[str, ...]
    ) -> tuple[str, ...]:
        try:
            normalized = validate_ignore_patterns(patterns)
        except ValueError as error:
            raise LibraryRegistrationError(str(error)) from error
        cursor = self.connection_or_raise.execute(
            "UPDATE memory_libraries SET ignore_patterns = ? WHERE id = ?",
            (json.dumps(normalized), library_id),
        )
        if cursor.rowcount == 0:
            raise LibraryRegistrationError("memory library not found")
        self.connection_or_raise.commit()
        self.scan_library(library_id)
        return normalized

    def scan_library(
        self,
        library_id: str,
        *,
        participate_in_transaction: bool = False,
        frozen_scan: MarkdownScan | None = None,
    ) -> dict[str, int | str]:
        library = self.library(library_id)
        if library is None:
            raise LibraryRegistrationError("memory library not found")
        if library.availability != "available":
            raise LibraryRegistrationError("memory library is not available")
        root = Path(library.canonical_path)
        patterns = self.library_ignore_patterns(library_id)
        current = {
            str(row[0]): (str(row[1]), str(row[2]))
            for row in self.connection_or_raise.execute(
                "SELECT path, id, source_version FROM memory_documents WHERE library_id = ?",
                (library_id,),
            )
        }
        changed = 0
        unchanged = 0
        self.connection_or_raise.execute(
            "UPDATE memory_libraries SET sync_status = 'scanning' WHERE id = ?", (library_id,)
        )
        if not participate_in_transaction:
            self.connection_or_raise.commit()
        scan = frozen_scan or scan_markdown(
            root, patterns, (self.database_path.parent,)
        )
        if not scan.complete:
            self.connection_or_raise.execute(
                "UPDATE memory_libraries SET sync_status = 'error' WHERE id = ?", (library_id,)
            )
            if not participate_in_transaction:
                self.connection_or_raise.commit()
            details = "; ".join(scan.errors[:3])
            raise LibraryRegistrationError(f"Markdown scan incomplete: {details}")
        documents = scan.documents
        found = {document.path for document in documents}
        try:
            for document in documents:
                existing = current.get(document.path)
                if existing is not None and existing[1] == document.version:
                    unchanged += 1
                    continue
                document_id = (
                    existing[0]
                    if existing is not None
                    else stable_document_id(library_id, document.path)
                )
                if existing is not None:
                    old_chunks = self.connection_or_raise.execute(
                        """SELECT id, kind, content, heading, start_line, end_line
                           FROM memory_chunks
                           WHERE document_id = ? ORDER BY start_line, id""",
                        (document_id,),
                    ).fetchall()
                    self.connection_or_raise.executemany(
                        "DELETE FROM memory_chunk_search WHERE chunk_id = ?",
                        [(str(row[0]),) for row in old_chunks],
                    )
                    self.connection_or_raise.execute(
                        "DELETE FROM memory_chunks WHERE document_id = ?", (document_id,)
                    )
                else:
                    old_chunks = []
                self.connection_or_raise.execute(
                    """INSERT INTO memory_documents (id, library_id, path, source_version)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET source_version = excluded.source_version""",
                    (document_id, library_id, document.path, document.version),
                )
                chunks = chunk_markdown(document.content)
                chunk_ids = match_chunk_ids(
                    [
                        StoredChunk(
                            id=str(row[0]),
                            kind=str(row[1]),
                            content=str(row[2]),
                            heading=None if row[3] is None else str(row[3]),
                            start_line=int(row[4]),
                            end_line=int(row[5]),
                        )
                        for row in old_chunks
                    ],
                    chunks,
                )
                for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
                    self.connection_or_raise.execute(
                        """INSERT INTO memory_chunks
                           (id, document_id, library_id, path, kind, heading, start_line,
                            end_line, content, source_version)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk_id,
                            document_id,
                            library_id,
                            document.path,
                            chunk.kind,
                            chunk.heading,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.content,
                            document.version,
                        ),
                    )
                    self.connection_or_raise.execute(
                        """INSERT INTO memory_chunk_search (chunk_id, content, heading, path)
                           VALUES (?, ?, ?, ?)""",
                        (chunk_id, chunk.content, chunk.heading or "", document.path),
                    )
                changed += 1

            removed_paths = set(current) - found
            for path in removed_paths:
                document_id = current[path][0]
                chunk_ids = self.connection_or_raise.execute(
                    "SELECT id FROM memory_chunks WHERE document_id = ?", (document_id,)
                ).fetchall()
                self.connection_or_raise.executemany(
                    "DELETE FROM memory_chunk_search WHERE chunk_id = ?", chunk_ids
                )
                self.connection_or_raise.execute(
                    "DELETE FROM memory_documents WHERE id = ?", (document_id,)
                )
            self.connection_or_raise.execute(
                "UPDATE memory_libraries SET sync_status = 'ready' WHERE id = ?", (library_id,)
            )
            if not participate_in_transaction:
                self.connection_or_raise.commit()
            self._queue_vector_rebuild(library_id, participate_in_transaction)
            self._queue_graph_rebuild(library_id, participate_in_transaction)
        except BaseException:
            if not participate_in_transaction:
                self.connection_or_raise.rollback()
                self.connection_or_raise.execute(
                    "UPDATE memory_libraries SET sync_status = 'error' WHERE id = ?",
                    (library_id,),
                )
                self.connection_or_raise.commit()
            raise
        return {
            "library_id": library_id,
            "documents": len(documents),
            "changed": changed,
            "unchanged": unchanged,
            "removed": len(set(current) - found),
        }

    def list_documents(self, library_id: str) -> list[dict[str, object]]:
        library = self._require_available_library(library_id)
        rows = self.connection_or_raise.execute(
            """SELECT path, source_version FROM memory_documents
               WHERE library_id = ? ORDER BY path""",
            (library.id,),
        ).fetchall()
        return [
            {"library_id": library.id, "path": str(row[0]), "source_version": str(row[1])}
            for row in rows
        ]

    def create_candidate(
        self,
        library_id: str,
        suggested_type: str,
        body: str,
        source_references: tuple[str, ...],
        creator: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        self._require_available_library(library_id)
        normalized_type = self._validate_candidate_content(
            suggested_type, body, source_references, creator
        )
        self._validate_operation_identifier(idempotency_key, "idempotency_key")
        request_hash = self._candidate_request_hash(
            library_id, normalized_type, body, source_references, creator
        )
        existing = self.connection_or_raise.execute(
            """SELECT id, request_hash FROM candidate_memories
               WHERE library_id = ? AND idempotency_key = ?""",
            (library_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            if str(existing[1]) != request_hash:
                raise CandidateGovernanceError(
                    "idempotency_key was already used for another candidate"
                )
            return self.candidate(str(existing[0]))
        candidate_id = str(uuid.uuid4())
        try:
            self.connection_or_raise.execute("BEGIN IMMEDIATE")
            self.connection_or_raise.execute(
                """INSERT INTO candidate_memories
                   (id, library_id, suggested_type, body, source_references_json,
                    creator, idempotency_key, request_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate_id,
                    library_id,
                    normalized_type,
                    body,
                    json.dumps(source_references, ensure_ascii=False),
                    creator,
                    idempotency_key,
                    request_hash,
                ),
            )
            self.connection_or_raise.execute(
                """INSERT INTO candidate_audit
                   (candidate_id, action, operator, reason, body)
                   VALUES (?, 'created', ?, ?, ?)""",
                (candidate_id, creator, "candidate submitted", body),
            )
            self.connection_or_raise.commit()
        except sqlite3.IntegrityError:
            self.connection_or_raise.rollback()
            existing = self.connection_or_raise.execute(
                """SELECT id, request_hash FROM candidate_memories
                   WHERE library_id = ? AND idempotency_key = ?""",
                (library_id, idempotency_key),
            ).fetchone()
            if existing is None or str(existing[1]) != request_hash:
                raise CandidateGovernanceError(
                    "idempotency_key was already used for another candidate"
                ) from None
            return self.candidate(str(existing[0]))
        return self.candidate(candidate_id)

    def ingest_capture_event(
        self,
        event_id: str,
        session_id: str,
        turn_id: str,
        event_kind: str,
        content: str,
        occurred_at: str,
        cwd: str,
    ) -> dict[str, object]:
        for value, label, maximum in (
            (event_id, "event_id", 200),
            (session_id, "session_id", 1000),
            (turn_id, "turn_id", 200),
            (occurred_at, "occurred_at", 100),
        ):
            if not value.strip() or len(value) > maximum or "\x00" in value:
                raise CandidateGovernanceError(f"invalid {label}")
        if event_kind not in {"user", "assistant"}:
            raise CandidateGovernanceError("invalid capture event kind")
        if not content.strip() or len(content.encode()) > 64 * 1024 or "\x00" in content:
            raise CandidateGovernanceError("invalid capture content")
        binding = self.resolve_project_binding(cwd)
        if binding["status"] != "bound":
            return {"status": "unbound", "event_id": event_id}
        library_id = str(binding["library_id"])
        project_id = str(binding["project_id"])
        request_hash = hashlib.sha256(
            json.dumps(
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "event_kind": event_kind,
                    "content": content,
                    "project_id": project_id,
                    "library_id": library_id,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        existing = self.connection_or_raise.execute(
            "SELECT request_hash FROM capture_inbox WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != request_hash:
                raise CandidateGovernanceError("event_id was already used for another event")
            return {"status": "accepted", "event_id": event_id, "duplicate": True}
        try:
            self.connection_or_raise.execute("BEGIN IMMEDIATE")
            self.connection_or_raise.execute(
                """INSERT INTO capture_inbox
                   (event_id, request_hash, session_id, project_id, library_id, turn_id,
                    event_kind, content, occurred_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    request_hash,
                    session_id,
                    project_id,
                    library_id,
                    turn_id,
                    event_kind,
                    content,
                    occurred_at,
                ),
            )
            self.connection_or_raise.execute(
                """INSERT INTO capture_rounds (session_id, turn_id, library_id)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id, turn_id) DO UPDATE SET
                     status = CASE WHEN capture_rounds.status = 'done'
                                   THEN 'done' ELSE 'pending' END,
                     updated_at = CURRENT_TIMESTAMP""",
                (session_id, turn_id, library_id),
            )
            self._queue_capture_consolidation(session_id, turn_id, True)
            self.connection_or_raise.commit()
        except sqlite3.IntegrityError as error:
            self.connection_or_raise.rollback()
            raise CandidateGovernanceError(
                "capture event conflicts with an existing turn"
            ) from error
        return {"status": "accepted", "event_id": event_id, "duplicate": False}

    def trigger_capture_consolidation(self, session_id: str | None = None) -> int:
        rows = self.connection_or_raise.execute(
            """SELECT session_id, turn_id FROM capture_rounds
               WHERE status IN ('pending', 'error')
                 AND (? IS NULL OR session_id = ?)
               ORDER BY updated_at""",
            (session_id, session_id),
        ).fetchall()
        queued = 0
        for row in rows:
            if self._queue_capture_consolidation(str(row[0]), str(row[1]), True):
                queued += 1
        self.connection_or_raise.commit()
        return queued

    def list_capture_events(self, session_id: str | None = None) -> list[dict[str, object]]:
        rows = self.connection_or_raise.execute(
            """SELECT event_id, session_id, project_id, library_id, turn_id, event_kind,
                      content, occurred_at, received_at, consolidated_at
               FROM capture_inbox WHERE (? IS NULL OR session_id = ?)
               ORDER BY received_at, event_id""",
            (session_id, session_id),
        ).fetchall()
        keys = (
            "event_id", "session_id", "project_id", "library_id", "turn_id", "event_kind",
            "content", "occurred_at", "received_at", "consolidated_at",
        )
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def list_capture_rounds(self, session_id: str | None = None) -> list[dict[str, object]]:
        rows = self.connection_or_raise.execute(
            """SELECT session_id, turn_id, library_id, status, candidate_id, last_error,
                      updated_at FROM capture_rounds
               WHERE (? IS NULL OR session_id = ?) ORDER BY updated_at, turn_id""",
            (session_id, session_id),
        ).fetchall()
        keys = (
            "session_id", "turn_id", "library_id", "status", "candidate_id", "last_error",
            "updated_at",
        )
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def list_candidates(
        self, library_id: str | None = None, candidate_status: str | None = None
    ) -> list[dict[str, object]]:
        parameters: list[str] = []
        predicates: list[str] = []
        if library_id is not None:
            self._require_available_library(library_id)
            predicates.append("library_id = ?")
            parameters.append(library_id)
        if candidate_status is not None:
            if candidate_status not in {"pending", "approved", "rejected"}:
                raise CandidateGovernanceError("invalid candidate status")
            predicates.append("status = ?")
            parameters.append(candidate_status)
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        rows = self.connection_or_raise.execute(
            "SELECT id FROM candidate_memories"
            + where
            + " ORDER BY created_at DESC, id DESC",
            parameters,
        ).fetchall()
        return [self.candidate(str(row[0])) for row in rows]

    def candidate(self, candidate_id: str) -> dict[str, object]:
        row = self.connection_or_raise.execute(
            """SELECT id, library_id, suggested_type, body, source_references_json,
                      creator, created_at, status, operator, reason, reviewed_at,
                      published_path, commit_id
               FROM candidate_memories WHERE id = ?""",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise CandidateGovernanceError("candidate memory not found")
        return CandidateMemory(
            id=str(row[0]),
            library_id=str(row[1]),
            suggested_type=str(row[2]),
            body=str(row[3]),
            source_references=tuple(str(item) for item in json.loads(str(row[4]))),
            creator=str(row[5]),
            created_at=str(row[6]),
            status=str(row[7]),
            operator=None if row[8] is None else str(row[8]),
            reason=None if row[9] is None else str(row[9]),
            reviewed_at=None if row[10] is None else str(row[10]),
            published_path=None if row[11] is None else str(row[11]),
            commit=None if row[12] is None else str(row[12]),
        ).payload()

    def edit_candidate(
        self, candidate_id: str, body: str, operator: str, reason: str
    ) -> dict[str, object]:
        current = self.candidate(candidate_id)
        self._validate_governance_actor(operator, reason)
        if current["status"] != "pending":
            raise CandidateGovernanceError("only pending candidates can be edited")
        if not body.strip() or "\x00" in body:
            raise CandidateGovernanceError("candidate body must contain Markdown text")
        if body == current["body"]:
            return current
        try:
            self.connection_or_raise.execute("BEGIN IMMEDIATE")
            updated = self.connection_or_raise.execute(
                """UPDATE candidate_memories SET body = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND status = 'pending'""",
                (body, candidate_id),
            )
            if updated.rowcount != 1:
                raise CandidateGovernanceError("candidate is no longer pending")
            self.connection_or_raise.execute(
                """INSERT INTO candidate_audit
                   (candidate_id, action, operator, reason, body)
                   VALUES (?, 'edited', ?, ?, ?)""",
                (candidate_id, operator, reason, body),
            )
            self.connection_or_raise.commit()
        except BaseException:
            self.connection_or_raise.rollback()
            raise
        return self.candidate(candidate_id)

    def approve_candidate(
        self,
        candidate_id: str,
        operator: str,
        reason: str,
        operation_id: str,
    ) -> dict[str, object]:
        self._validate_governance_actor(operator, reason)
        self._validate_operation_identifier(operation_id, "operation_id")
        self._ensure_decision_operation_available(candidate_id, operation_id)
        current = self.candidate(candidate_id)
        decision_hash = self._decision_request_hash(
            candidate_id, "approved", str(current["body"]), operator, reason
        )
        if current["status"] != "pending":
            completed = self._completed_candidate_decision(
                current, operation_id, decision_hash
            )
            persisted_content = self._published_candidate_markdown(
                current, operator, reason
            ).encode()
            path = str(current["published_path"] or "")
            commit = str(current["commit"] or "")
            repository = self._git_repository(str(current["library_id"]))
            request_hash = self._publication_request_hash(
                str(current["library_id"]),
                path,
                persisted_content.decode(),
                "candidate-approval",
            )
            persisted = self._persisted_publication(
                str(current["library_id"]),
                path,
                persisted_content,
                operation_id,
                request_hash,
                commit,
                repository,
            )
            if persisted is None or not self._approved_candidate_decision_is_persisted(
                current,
                operator,
                reason,
                operation_id,
                decision_hash,
                {"path": path, "commit": commit},
            ):
                raise CandidateGovernanceError(
                    "approved candidate publication is inconsistent"
                )
            return completed
        path = f"memory-{candidate_id}.md"
        content = self._published_candidate_markdown(current, operator, reason)

        def record_approval(published: dict[str, str]) -> None:
            updated = self.connection_or_raise.execute(
                """UPDATE candidate_memories
                   SET status = 'approved', operator = ?, reason = ?,
                       reviewed_at = CURRENT_TIMESTAMP, published_path = ?, commit_id = ?,
                       decision_operation_id = ?, decision_request_hash = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND status = 'pending'""",
                (
                    operator,
                    reason,
                    published["path"],
                    published["commit"],
                    operation_id,
                    decision_hash,
                    candidate_id,
                ),
            )
            if updated.rowcount != 1:
                raise CandidateGovernanceError("candidate is no longer pending")
            self.connection_or_raise.execute(
                """INSERT INTO candidate_audit
                   (candidate_id, action, operator, reason, body)
                   VALUES (?, 'approved', ?, ?, ?)""",
                (candidate_id, operator, reason, current["body"]),
            )

        def approval_is_persisted(published: dict[str, str]) -> bool:
            return self._approved_candidate_decision_is_persisted(
                current,
                operator,
                reason,
                operation_id,
                decision_hash,
                published,
            )

        def compensate_approval() -> None:
            self.connection_or_raise.execute(
                """UPDATE candidate_memories
                   SET status = 'pending', operator = NULL, reason = NULL,
                       reviewed_at = NULL, published_path = NULL, commit_id = NULL,
                       decision_operation_id = NULL, decision_request_hash = NULL,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND decision_operation_id = ?""",
                (candidate_id, operation_id),
            )
            self.connection_or_raise.execute(
                """DELETE FROM candidate_audit
                   WHERE candidate_id = ? AND action = 'approved' AND operator = ?
                     AND reason = ? AND body = ?""",
                (candidate_id, operator, reason, current["body"]),
            )

        self._publish_document(
            str(current["library_id"]),
            path,
            content,
            operation_id,
            operator,
            "candidate-approval",
            f"Publish candidate memory {candidate_id}",
            record_approval,
            approval_is_persisted,
            compensate_approval,
        )
        return self.candidate(candidate_id)

    def reject_candidate(
        self,
        candidate_id: str,
        operator: str,
        reason: str,
        operation_id: str,
    ) -> dict[str, object]:
        self._validate_governance_actor(operator, reason)
        self._validate_operation_identifier(operation_id, "operation_id")
        self._ensure_decision_operation_available(candidate_id, operation_id)
        current = self.candidate(candidate_id)
        decision_hash = self._decision_request_hash(
            candidate_id, "rejected", str(current["body"]), operator, reason
        )
        if current["status"] != "pending":
            return self._completed_candidate_decision(current, operation_id, decision_hash)
        try:
            self.connection_or_raise.execute("BEGIN IMMEDIATE")
            updated = self.connection_or_raise.execute(
                """UPDATE candidate_memories
                   SET status = 'rejected', operator = ?, reason = ?,
                       reviewed_at = CURRENT_TIMESTAMP, decision_operation_id = ?,
                       decision_request_hash = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND status = 'pending'""",
                (operator, reason, operation_id, decision_hash, candidate_id),
            )
            if updated.rowcount != 1:
                raise CandidateGovernanceError("candidate is no longer pending")
            self.connection_or_raise.execute(
                """INSERT INTO candidate_audit
                   (candidate_id, action, operator, reason, body)
                   VALUES (?, 'rejected', ?, ?, ?)""",
                (candidate_id, operator, reason, current["body"]),
            )
            self.connection_or_raise.commit()
        except sqlite3.IntegrityError as error:
            self.connection_or_raise.rollback()
            raise CandidateGovernanceError(
                "operation_id was already used for another candidate decision"
            ) from error
        except BaseException:
            self.connection_or_raise.rollback()
            raise
        return self.candidate(candidate_id)

    @staticmethod
    def _validate_candidate_content(
        suggested_type: str,
        body: str,
        source_references: tuple[str, ...],
        creator: str,
    ) -> str:
        normalized_type = suggested_type.strip().lower().replace("-", "_")
        allowed = {
            "preference",
            "decision",
            "constraint",
            "domain_fact",
            "reusable_experience",
            "external_reference",
        }
        if normalized_type not in allowed:
            raise CandidateGovernanceError("unsupported candidate memory type")
        if not body.strip() or "\x00" in body:
            raise CandidateGovernanceError("candidate body must contain Markdown text")
        if not source_references or any(
            not reference.strip() or len(reference) > 2_000 for reference in source_references
        ):
            raise CandidateGovernanceError(
                "candidate must contain non-empty source references"
            )
        if not creator.strip() or len(creator) > 200:
            raise CandidateGovernanceError("creator must be present and at most 200 characters")
        return normalized_type

    @staticmethod
    def _validate_governance_actor(operator: str, reason: str) -> None:
        if not operator.strip() or len(operator) > 200:
            raise CandidateGovernanceError("operator must be present and at most 200 characters")
        if not reason.strip() or len(reason) > 2_000:
            raise CandidateGovernanceError("reason must be present and at most 2000 characters")

    @staticmethod
    def _validate_operation_identifier(value: str, field: str) -> None:
        if not value.strip() or len(value) > 200:
            raise CandidateGovernanceError(
                f"{field} must be present and at most 200 characters"
            )

    @staticmethod
    def _candidate_request_hash(
        library_id: str,
        suggested_type: str,
        body: str,
        source_references: tuple[str, ...],
        creator: str,
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "library_id": library_id,
                    "suggested_type": suggested_type,
                    "body": body,
                    "source_references": source_references,
                    "creator": creator,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()

    @staticmethod
    def _decision_request_hash(
        candidate_id: str, decision: str, body: str, operator: str, reason: str
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "decision": decision,
                    "body": body,
                    "operator": operator,
                    "reason": reason,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def _completed_candidate_decision(
        self, current: dict[str, object], operation_id: str, decision_hash: str
    ) -> dict[str, object]:
        row = self.connection_or_raise.execute(
            """SELECT decision_operation_id, decision_request_hash
               FROM candidate_memories WHERE id = ?""",
            (current["id"],),
        ).fetchone()
        if row is not None and str(row[0]) == operation_id and str(row[1]) == decision_hash:
            return current
        raise CandidateGovernanceError("candidate has already received a final decision")

    def _approved_candidate_decision_is_persisted(
        self,
        current: dict[str, object],
        operator: str,
        reason: str,
        operation_id: str,
        decision_hash: str,
        published: dict[str, str],
    ) -> bool:
        row = self.connection_or_raise.execute(
            """SELECT status, operator, reason, published_path, commit_id,
                      decision_operation_id, decision_request_hash
               FROM candidate_memories WHERE id = ?""",
            (current["id"],),
        ).fetchone()
        audits = self.connection_or_raise.execute(
            """SELECT operator, reason, body FROM candidate_audit
               WHERE candidate_id = ? AND action = 'approved' ORDER BY id""",
            (current["id"],),
        ).fetchall()
        return bool(
            row
            == (
                "approved",
                operator,
                reason,
                published["path"],
                published["commit"],
                operation_id,
                decision_hash,
            )
            and audits == [(operator, reason, current["body"])]
        )

    def _ensure_decision_operation_available(
        self, candidate_id: str, operation_id: str
    ) -> None:
        row = self.connection_or_raise.execute(
            "SELECT id FROM candidate_memories WHERE decision_operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is not None and str(row[0]) != candidate_id:
            raise CandidateGovernanceError(
                "operation_id was already used for another candidate decision"
            )

    @staticmethod
    def _published_candidate_markdown(
        candidate: dict[str, object], operator: str, reason: str
    ) -> str:
        references = cast(tuple[str, ...], candidate["source_references"])
        lines = [
            "---",
            f"memory_id: {json.dumps(candidate['id'])}",
            f"memory_type: {json.dumps(candidate['suggested_type'])}",
            f"candidate_id: {json.dumps(candidate['id'])}",
            f"created_by: {json.dumps(candidate['creator'], ensure_ascii=False)}",
            f"approved_by: {json.dumps(operator, ensure_ascii=False)}",
            f"approval_reason: {json.dumps(reason, ensure_ascii=False)}",
            "source_references:",
            *(f"  - {json.dumps(reference, ensure_ascii=False)}" for reference in references),
            "---",
            "",
            str(candidate["body"]).rstrip(),
            "",
        ]
        return "\n".join(lines)

    def read_document(self, library_id: str, path: str) -> dict[str, str]:
        library = self._require_available_library(library_id)
        normalized, source_version = self._document_record(library_id, path)
        content_bytes = self._read_document_bytes(Path(library.canonical_path), normalized)
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeError as error:
            raise MemoryMutationError(f"memory document cannot be read: {error}") from error
        actual_version = hashlib.sha256(content_bytes).hexdigest()
        if actual_version != source_version:
            raise MemoryMutationError(
                "document changed outside the platform; rescan before editing"
            )
        return {
            "library_id": library_id,
            "path": normalized,
            "content": content,
            "source_version": source_version,
        }

    def preview_document_edit(
        self, library_id: str, path: str, content: str, expected_source_version: str
    ) -> dict[str, object]:
        current = self.read_document(library_id, path)
        if current["source_version"] != expected_source_version:
            raise MemoryMutationError("document version conflict")
        diff = "".join(
            difflib.unified_diff(
                current["content"].splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{current['path']}",
                tofile=f"b/{current['path']}",
            )
        )
        return {"library_id": library_id, "path": current["path"], "diff": diff}

    def edit_document(
        self,
        library_id: str,
        path: str,
        content: str,
        expected_source_version: str,
        operation_id: str,
        actor_type: str,
        source: str,
    ) -> dict[str, str]:
        return self._mutate_document(
            library_id,
            path,
            content,
            expected_source_version,
            operation_id,
            actor_type,
            source,
            "edit",
            "Edit memory document",
        )

    def restore_document(
        self,
        library_id: str,
        path: str,
        commit: str,
        expected_source_version: str,
        operation_id: str,
        actor_type: str,
        source: str,
    ) -> dict[str, str]:
        repository = self._git_repository(library_id)
        try:
            content = repository.content_at(commit, path)
        except GitHistoryError as error:
            raise MemoryMutationError(str(error)) from error
        return self._mutate_document(
            library_id,
            path,
            content,
            expected_source_version,
            operation_id,
            actor_type,
            source,
            "restore",
            f"Restore memory document from {repository.resolve_commit(commit)[:12]}",
        )

    def library_history(self, library_id: str, limit: int = 50) -> list[dict[str, str]]:
        self._require_available_library(library_id)
        try:
            history = self._git_repository(library_id).history(limit)
        except GitHistoryError as error:
            raise MemoryMutationError(str(error)) from error
        operations = {
            str(row[0]): {"actor_type": str(row[1]), "source": str(row[2]), "kind": str(row[3])}
            for row in self.connection_or_raise.execute(
                """SELECT commit_id, actor_type, source, kind FROM memory_operations
                   WHERE library_id = ?""",
                (library_id,),
            )
        }
        operations.update(
            {
                str(row[0]): {
                    "actor_type": "user",
                    "source": "candidate-approval",
                    "kind": "approval",
                }
                for row in self.connection_or_raise.execute(
                    """SELECT commit_id FROM candidate_memories
                       WHERE library_id = ? AND status = 'approved'""",
                    (library_id,),
                )
                if row[0] is not None
            }
        )
        for entry in history:
            entry.update(operations.get(entry["commit"], {}))
        return history

    def history_diff(self, library_id: str, commit: str) -> dict[str, object]:
        self._require_available_library(library_id)
        try:
            diff = self._git_repository(library_id).diff(commit)
        except GitHistoryError as error:
            raise MemoryMutationError(str(error)) from error
        return {
            "library_id": library_id,
            "commit": commit,
            "diff": diff,
            "lines": _diff_lines(diff),
        }

    def _mutate_document(
        self,
        library_id: str,
        path: str,
        content: str,
        expected_source_version: str,
        operation_id: str,
        actor_type: str,
        source: str,
        kind: str,
        message: str,
    ) -> dict[str, str]:
        if not operation_id.strip() or len(operation_id) > 200:
            raise MemoryMutationError("operation_id must be present and at most 200 characters")
        if actor_type not in {"user", "platform"}:
            raise MemoryMutationError("actor_type must be user or platform")
        if not source.strip() or len(source) > 200:
            raise MemoryMutationError("source must be present and at most 200 characters")
        if "\x00" in content:
            raise MemoryMutationError("Markdown content cannot contain NUL bytes")
        request_hash = hashlib.sha256(
            json.dumps(
                {
                    "library_id": library_id,
                    "path": path,
                    "content": content,
                    "expected_source_version": expected_source_version,
                    "actor_type": actor_type,
                    "source": source,
                    "kind": kind,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        existing = self.connection_or_raise.execute(
            """SELECT response_json, request_hash FROM memory_operations
               WHERE operation_id = ?""",
            (operation_id,),
        ).fetchone()
        if existing is not None:
            if str(existing[1]) != request_hash:
                raise MemoryMutationError("operation_id was already used for another mutation")
            return {str(key): str(value) for key, value in json.loads(str(existing[0])).items()}
        library = self._require_available_library(library_id)
        with self._library_lock(library_id):
            existing = self.connection_or_raise.execute(
                """SELECT response_json, request_hash FROM memory_operations
                   WHERE operation_id = ?""",
                (operation_id,),
            ).fetchone()
            if existing is not None:
                if str(existing[1]) != request_hash:
                    raise MemoryMutationError(
                        "operation_id was already used for another mutation"
                    )
                return {str(key): str(value) for key, value in json.loads(str(existing[0])).items()}
            normalized, indexed_version = self._document_record(library_id, path)
            if indexed_version != expected_source_version:
                raise MemoryMutationError("document version conflict")
            root = Path(library.canonical_path)
            with self._bind_document(root, normalized) as bound:
                original_bytes, original_identity = self._read_bound_document(bound)
                try:
                    original = original_bytes.decode("utf-8")
                except UnicodeError as error:
                    raise MemoryMutationError(
                        f"memory document cannot be read: {error}"
                    ) from error
                actual_version = hashlib.sha256(original_bytes).hexdigest()
                if actual_version != indexed_version:
                    raise MemoryMutationError(
                        "document changed outside the platform; rescan before editing"
                    )
                if content == original:
                    raise MemoryMutationError("document content is unchanged")
                trusted_scan = self._trusted_library_scan(
                    root,
                    bound,
                    library_id,
                    normalized,
                    original_identity,
                    indexed_version,
                )
                repository = self._git_repository(library_id)
                try:
                    repository.ensure_index_clean()
                except GitHistoryError as error:
                    raise MemoryMutationError(str(error)) from error
                previous_head = repository.head()
                assert previous_head is not None
                index_snapshot = self._snapshot_library_index(library_id)
                commit: str | None = None
                database_committed = False
                replacement = DocumentReplacement()
                platform_content = content.encode()
                try:
                    self._replace_bound_document(
                        bound,
                        platform_content,
                        original_identity,
                        original_bytes,
                        replacement,
                    )
                    replacement_identity = replacement.identity
                    if replacement_identity is None:
                        raise MemoryMutationError(
                            "document replacement identity is unavailable"
                        )
                    expected_scan = self._updated_library_scan(
                        trusted_scan,
                        normalized,
                        content,
                        replacement_identity,
                    )
                    self._verify_bound_document(
                        root, normalized, bound, replacement_identity
                    )
                    observed_content, read_identity = self._read_bound_document(bound)
                    if (
                        read_identity != replacement_identity
                        or observed_content != platform_content
                    ):
                        raise MemoryMutationError(
                            "document changed while the platform was saving it"
                        )
                    commit = repository.commit({normalized: platform_content}, message)
                    self._verify_bound_document(
                        root, normalized, bound, replacement_identity
                    )
                    scan = self._scan_bound_library(root, bound, library_id)
                    self._require_matching_scan(scan, expected_scan)
                    self._verify_bound_library_scan(root, bound, library_id, scan)
                    self.connection_or_raise.execute("BEGIN IMMEDIATE")
                    self.scan_library(
                        library_id,
                        participate_in_transaction=True,
                        frozen_scan=scan,
                    )
                    self._verify_bound_library_scan(root, bound, library_id, scan)
                    self._verify_bound_document(
                        root, normalized, bound, replacement_identity
                    )
                    updated = self._document_record(library_id, normalized)[1]
                    expected_updated = hashlib.sha256(platform_content).hexdigest()
                    if updated != expected_updated:
                        raise MemoryMutationError(
                            "document index version does not match committed content"
                        )
                    response = {
                        "library_id": library_id,
                        "path": normalized,
                        "source_version": updated,
                        "commit": commit,
                        "operation_id": operation_id,
                    }
                    self.connection_or_raise.execute(
                        """INSERT INTO memory_operations
                           (operation_id, library_id, kind, document_path, actor_type, source,
                            request_hash, commit_id, response_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            operation_id,
                            library_id,
                            kind,
                            normalized,
                            actor_type,
                            source,
                            request_hash,
                            commit,
                            json.dumps(response, sort_keys=True),
                        ),
                    )
                    self._verify_bound_document(
                        root, normalized, bound, replacement_identity
                    )
                    self._verify_bound_library_scan(root, bound, library_id, scan)
                    self.connection_or_raise.commit()
                    database_committed = True
                    self._verify_bound_library_scan(root, bound, library_id, scan)
                    self._verify_bound_document(
                        root, normalized, bound, replacement_identity
                    )
                    return response
                except BaseException as error:
                    self.connection_or_raise.rollback()
                    compensation_errors: list[str] = []
                    compensation_conflicts: list[str] = []
                    if database_committed:
                        try:
                            self.connection_or_raise.execute("BEGIN IMMEDIATE")
                            self.connection_or_raise.execute(
                                "DELETE FROM memory_operations WHERE operation_id = ?",
                                (operation_id,),
                            )
                            self._restore_library_index(library_id, index_snapshot)
                            self.connection_or_raise.commit()
                        except BaseException as compensation_error:
                            self.connection_or_raise.rollback()
                            compensation_errors.append(f"database: {compensation_error}")
                    if commit is not None:
                        try:
                            repository.rollback_commit(commit, previous_head)
                        except BaseException as compensation_error:
                            compensation_errors.append(f"Git: {compensation_error}")
                    try:
                        current_bytes, current_identity = self._read_bound_document(bound)
                        if (
                            not replacement.renamed
                            and current_identity == original_identity
                            and current_bytes == original_bytes
                        ):
                            pass
                        elif (
                            replacement.renamed
                            and replacement.expected_target_exchanged
                            and replacement.identity is not None
                            and current_identity == replacement.identity
                            and current_bytes == platform_content
                        ):
                            self._replace_bound_document(
                                bound,
                                original_bytes,
                                replacement.identity,
                                platform_content,
                                DocumentReplacement(),
                            )
                        else:
                            compensation_conflicts.append(
                                "concurrent external file version preserved"
                            )
                    except (MemoryMutationError, OSError) as compensation_error:
                        compensation_conflicts.append(
                            "concurrent external file state preserved: "
                            f"{compensation_error}"
                        )
                    if compensation_errors or compensation_conflicts:
                        details = []
                        if compensation_errors:
                            details.append(
                                "compensation failed: " + "; ".join(compensation_errors)
                            )
                        if compensation_conflicts:
                            details.append(
                                "compensation conflict: "
                                + "; ".join(compensation_conflicts)
                            )
                        raise MemoryMutationError(f"{error}; {'; '.join(details)}") from error
                    if isinstance(
                        error,
                        (GitHistoryError, LibraryRegistrationError, MemoryMutationError),
                    ):
                        raise MemoryMutationError(str(error)) from error
                    raise

    @staticmethod
    def _publication_request_hash(
        library_id: str, path: str, content: str, source: str
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "library_id": library_id,
                    "path": path,
                    "content": content,
                    "actor_type": "user",
                    "source": source,
                    "kind": "edit",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def _publish_document(
        self,
        library_id: str,
        path: str,
        content: str,
        operation_id: str,
        operator: str,
        source: str,
        message: str,
        before_commit: Callable[[dict[str, str]], None] | None = None,
        committed_state_is_valid: Callable[[dict[str, str]], bool] | None = None,
        compensate_committed_state: Callable[[], None] | None = None,
    ) -> dict[str, str]:
        normalized = PurePosixPath(path).as_posix()
        if not allowed_markdown_path(normalized) or normalized != path.replace("\\", "/"):
            raise MemoryMutationError("invalid Markdown document path")
        if len(PurePosixPath(normalized).parts) != 1:
            raise MemoryMutationError("published memory path must be at the library root")
        if "\x00" in content:
            raise MemoryMutationError("Markdown content cannot contain NUL bytes")
        request_hash = self._publication_request_hash(
            library_id, normalized, content, source
        )
        existing = self.connection_or_raise.execute(
            "SELECT response_json, request_hash FROM memory_operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if existing is not None:
            if str(existing[1]) != request_hash:
                raise MemoryMutationError("operation_id was already used for another mutation")
            return {str(key): str(value) for key, value in json.loads(str(existing[0])).items()}
        library = self._require_available_library(library_id)
        root = Path(library.canonical_path)
        platform_content = content.encode()
        with self._library_lock(library_id):
            existing = self.connection_or_raise.execute(
                "SELECT response_json, request_hash FROM memory_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if existing is not None:
                if str(existing[1]) != request_hash:
                    raise MemoryMutationError(
                        "operation_id was already used for another mutation"
                    )
                return {
                    str(key): str(value)
                    for key, value in json.loads(str(existing[0])).items()
                }
            if self.connection_or_raise.execute(
                "SELECT 1 FROM memory_documents WHERE library_id = ? AND path = ?",
                (library_id, normalized),
            ).fetchone() is not None:
                raise MemoryMutationError("memory document already exists")
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            descriptor = -1
            created = False
            commit: str | None = None
            response: dict[str, str] | None = None
            repository = self._git_repository(library_id)
            previous_head = repository.head()
            assert previous_head is not None
            index_snapshot = self._snapshot_library_index(library_id)
            try:
                root_metadata = os.fstat(root_fd)
                trusted = scan_markdown_fd(
                    root_fd,
                    root,
                    self.library_ignore_patterns(library_id),
                    (self.database_path.parent,),
                )
                if not trusted.complete:
                    raise MemoryMutationError(
                        f"Markdown scan incomplete: {'; '.join(trusted.errors[:3])}"
                    )
                indexed = tuple(
                    (str(row[0]), str(row[1]))
                    for row in self.connection_or_raise.execute(
                        """SELECT path, source_version FROM memory_documents
                           WHERE library_id = ? ORDER BY path""",
                        (library_id,),
                    )
                )
                observed = tuple(
                    (document.path, document.version) for document in trusted.documents
                )
                if observed != indexed:
                    raise MemoryMutationError(
                        "memory library changed outside the platform; rescan before publishing"
                    )
                repository.ensure_index_clean()
                descriptor = os.open(
                    normalized,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=root_fd,
                )
                created = True
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = -1
                    stream.write(platform_content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.fsync(root_fd)
                current_root = os.fstat(root_fd)
                if (current_root.st_dev, current_root.st_ino) != (
                    root_metadata.st_dev,
                    root_metadata.st_ino,
                ):
                    raise MemoryMutationError("memory library changed while publishing")
                committed_bytes = self._read_relative_regular_file(root_fd, normalized)
                if committed_bytes != platform_content:
                    raise MemoryMutationError("published memory changed while saving")
                scan = scan_markdown_fd(
                    root_fd,
                    root,
                    self.library_ignore_patterns(library_id),
                    (self.database_path.parent,),
                )
                expected_documents = tuple(
                    sorted(
                        (
                            *trusted.documents,
                            MarkdownDocument(
                                normalized,
                                content,
                                hashlib.sha256(platform_content).hexdigest(),
                            ),
                        ),
                        key=lambda document: document.path,
                    )
                )
                if not scan.complete or scan.documents != expected_documents:
                    raise MemoryMutationError(
                        "memory library changed while the platform was publishing"
                    )
                commit = repository.commit({normalized: platform_content}, message)
                self.connection_or_raise.execute("BEGIN IMMEDIATE")
                self.scan_library(
                    library_id, participate_in_transaction=True, frozen_scan=scan
                )
                updated = self._document_record(library_id, normalized)[1]
                response = {
                    "library_id": library_id,
                    "path": normalized,
                    "source_version": updated,
                    "commit": commit,
                    "operation_id": operation_id,
                }
                self.connection_or_raise.execute(
                    """INSERT INTO memory_operations
                       (operation_id, library_id, kind, document_path, actor_type, source,
                        request_hash, commit_id, response_json)
                       VALUES (?, ?, 'edit', ?, 'user', ?, ?, ?, ?)""",
                    (
                        operation_id,
                        library_id,
                        normalized,
                        f"{source}:{operator}",
                        request_hash,
                        commit,
                        json.dumps(response, sort_keys=True),
                    ),
                )
                if before_commit is not None:
                    before_commit(response)
                self.connection_or_raise.commit()
                return response
            except BaseException as error:
                self.connection_or_raise.rollback()
                if commit is not None and response is not None:
                    persisted = self._persisted_publication(
                        library_id,
                        normalized,
                        platform_content,
                        operation_id,
                        request_hash,
                        commit,
                        repository,
                    )
                    if persisted is not None and (
                        committed_state_is_valid is None
                        or committed_state_is_valid(persisted)
                    ):
                        return persisted
                compensation_errors: list[str] = []
                if commit is not None:
                    try:
                        repository.rollback_commit(commit, previous_head)
                    except BaseException as compensation_error:
                        compensation_errors.append(f"Git: {compensation_error}")
                try:
                    current = self._read_relative_regular_file(root_fd, normalized)
                except (FileNotFoundError, MemoryMutationError):
                    current = None
                if created and current == platform_content:
                    try:
                        os.unlink(normalized, dir_fd=root_fd)
                        os.fsync(root_fd)
                    except OSError as compensation_error:
                        compensation_errors.append(f"file: {compensation_error}")
                if commit is not None:
                    try:
                        self.connection_or_raise.execute("BEGIN IMMEDIATE")
                        self.connection_or_raise.execute(
                            "DELETE FROM memory_operations WHERE operation_id = ?",
                            (operation_id,),
                        )
                        if compensate_committed_state is not None:
                            compensate_committed_state()
                        self._restore_library_index(library_id, index_snapshot)
                        self.connection_or_raise.commit()
                    except BaseException as compensation_error:
                        self.connection_or_raise.rollback()
                        compensation_errors.append(f"database: {compensation_error}")
                if compensation_errors:
                    raise MemoryMutationError(
                        f"{error}; compensation failed: {'; '.join(compensation_errors)}"
                    ) from error
                if isinstance(
                    error,
                    (GitHistoryError, LibraryRegistrationError, MemoryMutationError),
                ):
                    raise MemoryMutationError(str(error)) from error
                raise
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                os.close(root_fd)

    def _persisted_publication(
        self,
        library_id: str,
        path: str,
        content: bytes,
        operation_id: str,
        request_hash: str,
        commit: str,
        repository: GitRepository,
    ) -> dict[str, str] | None:
        row = self.connection_or_raise.execute(
            """SELECT response_json, request_hash, commit_id, document_path
               FROM memory_operations WHERE operation_id = ?""",
            (operation_id,),
        ).fetchone()
        if row is None or (str(row[1]), str(row[2]), str(row[3])) != (
            request_hash,
            commit,
            path,
        ):
            return None
        response = {str(key): str(value) for key, value in json.loads(str(row[0])).items()}
        expected_version = hashlib.sha256(content).hexdigest()
        if response != {
            "library_id": library_id,
            "path": path,
            "source_version": expected_version,
            "commit": commit,
            "operation_id": operation_id,
        }:
            return None
        try:
            stored_path, stored_version = self._document_record(library_id, path)
            file_content = self._read_document_bytes(repository.work_tree, path)
            committed_content = repository.content_at(commit, path).encode()
            document_id = stable_document_id(library_id, path)
            indexed_chunks = self.connection_or_raise.execute(
                """SELECT chunk.id, chunk.content, chunk.source_version,
                          search.chunk_id
                   FROM memory_chunks AS chunk
                   LEFT JOIN memory_chunk_search AS search ON search.chunk_id = chunk.id
                   WHERE chunk.document_id = ? ORDER BY chunk.start_line, chunk.id""",
                (document_id,),
            ).fetchall()
        except (GitHistoryError, MemoryMutationError, OSError):
            return None
        expected_chunks = chunk_markdown(content.decode())
        return (
            response
            if stored_path == path
            and stored_version == expected_version
            and file_content == content
            and committed_content == content
            and len(indexed_chunks) == len(expected_chunks)
            and all(
                str(row[1]) == expected.content
                and str(row[2]) == expected_version
                and str(row[3]) == str(row[0])
                for row, expected in zip(indexed_chunks, expected_chunks, strict=True)
            )
            else None
        )

    def _prepare_git_repository(
        self,
        library_id: str,
        reuse_existing_git: bool,
        root: Path,
        root_fd: int,
        root_identity: tuple[int, int],
        registration_scan: MarkdownScan,
    ) -> tuple[GitRepository, GitInitialization, str]:
        if reuse_existing_git:
            git_dir = root / ".git"
            if not git_dir.is_dir():
                raise GitHistoryError(
                    "explicit reuse requires a .git directory at the library root"
                )
            mode = "authorized_existing"
        else:
            git_dir = self.database_path.parent / "git" / f"{library_id}.git"
            mode = "sidecar"
        documents = (
            {document.path: document.content.encode() for document in registration_scan.documents}
            if registration_scan.complete
            else {}
        )

        def verify_frozen_documents() -> None:
            manifest = (
                f'{{"format":1,"library_id":"{library_id}"}}\n'.encode()
                if mode == "authorized_existing"
                else None
            )
            self._verify_registration_tree(
                root, root_fd, root_identity, registration_scan, manifest
            )

        repository = GitRepository(git_dir=git_dir, work_tree=root)
        initialization = repository.initialize(
            library_id, documents, verify_frozen_documents
        )
        return repository, initialization, mode

    def _verify_registration_tree(
        self,
        root: Path,
        root_fd: int,
        root_identity: tuple[int, int],
        expected: MarkdownScan,
        allowed_manifest: bytes | None,
    ) -> None:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) != root_identity:
                raise GitHistoryError("memory library changed during registration")
        finally:
            os.close(descriptor)
        current = scan_markdown_fd(
            root_fd, root, (), (self.database_path.parent,)
        )
        expected_identities = expected.identities
        current_identities = current.identities
        manifest_path = ".personal-agent-memory.json"
        expected_manifest = tuple(
            item for item in expected_identities if item[0] == manifest_path
        )
        current_manifest = tuple(
            item for item in current_identities if item[0] == manifest_path
        )
        if allowed_manifest is None or expected_manifest:
            manifest_matches = current_manifest == expected_manifest
        else:
            manifest_matches = len(current_manifest) == 1
            if manifest_matches:
                try:
                    manifest = self._read_relative_regular_file(root_fd, manifest_path)
                except MemoryMutationError as error:
                    raise GitHistoryError(str(error)) from error
                manifest_matches = manifest == allowed_manifest
            expected_identities = tuple(
                item for item in expected_identities if item[0] != manifest_path
            )
            current_identities = tuple(
                item for item in current_identities if item[0] != manifest_path
            )
        if not manifest_matches or (
            current.documents,
            current.errors,
            current_identities,
        ) != (
            expected.documents,
            expected.errors,
            expected_identities,
        ):
            raise GitHistoryError("memory library changed during registration")

    @staticmethod
    def _read_relative_regular_file(root_fd: int, path: str) -> bytes:
        parent_fd = os.dup(root_fd)
        try:
            parts = PurePosixPath(path).parts
            for part in parts[:-1]:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                os.close(parent_fd)
                parent_fd = child
            descriptor = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise MemoryMutationError("memory library entry is not a regular file")
                content = bytearray()
                while block := os.read(descriptor, 1024 * 1024):
                    content.extend(block)
                return bytes(content)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise MemoryMutationError(
                f"memory library entry cannot be read: {error}"
            ) from error
        finally:
            os.close(parent_fd)

    def _git_repository(self, library_id: str) -> GitRepository:
        row = self.connection_or_raise.execute(
            "SELECT git_dir FROM library_git_repositories WHERE library_id = ?", (library_id,)
        ).fetchone()
        library = self._require_available_library(library_id)
        if row is None:
            raise MemoryMutationError("memory Git history is not initialized")
        return GitRepository(Path(str(row[0])), Path(library.canonical_path))

    def _require_available_library(self, library_id: str) -> MemoryLibrary:
        library = self.library(library_id)
        if library is None:
            raise MemoryMutationError("memory library not found")
        if library.availability != "available":
            raise MemoryMutationError("memory library is not available")
        return library

    def _document_record(self, library_id: str, path: str) -> tuple[str, str]:
        normalized = PurePosixPath(path).as_posix()
        if not allowed_markdown_path(normalized) or normalized != path.replace("\\", "/"):
            raise MemoryMutationError("invalid Markdown document path")
        row = self.connection_or_raise.execute(
            """SELECT path, source_version FROM memory_documents
               WHERE library_id = ? AND path = ?""",
            (library_id, normalized),
        ).fetchone()
        if row is None:
            raise MemoryMutationError("memory document not found")
        return str(row[0]), str(row[1])

    @staticmethod
    def _read_document_bytes(root: Path, path: str) -> bytes:
        parent, name = PlatformState._open_document_parent(root, path)
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise MemoryMutationError("memory document is not a regular file")
                with os.fdopen(descriptor, "rb") as stream:
                    descriptor = -1
                    return stream.read()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        except OSError as error:
            raise MemoryMutationError(f"memory document cannot be read: {error}") from error
        finally:
            os.close(parent)

    @staticmethod
    @contextmanager
    def _bind_document(root: Path, path: str) -> Iterator[BoundDocument]:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_metadata = os.fstat(root_fd)
        parent_fd = root_fd
        try:
            for part in PurePosixPath(path).parts[:-1]:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                if parent_fd != root_fd:
                    os.close(parent_fd)
                parent_fd = child
            parent_metadata = os.fstat(parent_fd)
            yield BoundDocument(
                root_identity=(root_metadata.st_dev, root_metadata.st_ino),
                root_fd=root_fd,
                parent_fd=parent_fd,
                name=PurePosixPath(path).parts[-1],
                parent_identity=(parent_metadata.st_dev, parent_metadata.st_ino),
            )
        finally:
            if parent_fd != root_fd:
                os.close(parent_fd)
            os.close(root_fd)

    def _scan_bound_library(
        self, root: Path, bound: BoundDocument, library_id: str
    ) -> MarkdownScan:
        scan = scan_markdown_fd(
            bound.root_fd,
            root,
            self.library_ignore_patterns(library_id),
            (self.database_path.parent,),
        )
        if not scan.complete:
            details = "; ".join(scan.errors[:3])
            raise MemoryMutationError(f"Markdown scan incomplete: {details}")
        return scan

    def _trusted_library_scan(
        self,
        root: Path,
        bound: BoundDocument,
        library_id: str,
        path: str,
        document_identity: tuple[int, int],
        source_version: str,
    ) -> MarkdownScan:
        scan = self._scan_bound_library(root, bound, library_id)
        self._verify_bound_library_scan(root, bound, library_id, scan)
        self._verify_bound_document(root, path, bound, document_identity)
        indexed = tuple(
            (str(row[0]), str(row[1]))
            for row in self.connection_or_raise.execute(
                "SELECT path, source_version FROM memory_documents "
                "WHERE library_id = ? ORDER BY path",
                (library_id,),
            )
        )
        observed = tuple((document.path, document.version) for document in scan.documents)
        if observed != indexed:
            raise MemoryMutationError(
                "memory library changed outside the platform; rescan before editing"
            )
        expected_identity = (
            path,
            "file",
            document_identity[0],
            document_identity[1],
            source_version,
        )
        if expected_identity not in scan.identities:
            raise MemoryMutationError("document path changed while the platform was saving it")
        return scan

    @staticmethod
    def _updated_library_scan(
        trusted: MarkdownScan,
        path: str,
        content: str,
        identity: tuple[int, int],
    ) -> MarkdownScan:
        version = hashlib.sha256(content.encode()).hexdigest()
        documents = tuple(
            MarkdownDocument(path, content, version) if document.path == path else document
            for document in trusted.documents
        )
        if not any(document.path == path for document in trusted.documents):
            raise MemoryMutationError("memory document disappeared while saving")
        identities = tuple(
            (path, "file", identity[0], identity[1], version)
            if item[0] == path and item[1] == "file"
            else item
            for item in trusted.identities
        )
        if not any(item[:2] == (path, "file") for item in trusted.identities):
            raise MemoryMutationError("memory document disappeared while saving")
        return MarkdownScan(documents, (), tuple(sorted(identities)))

    @staticmethod
    def _require_matching_scan(actual: MarkdownScan, expected: MarkdownScan) -> None:
        if not actual.complete or (
            actual.documents,
            actual.identities,
        ) != (
            expected.documents,
            expected.identities,
        ):
            raise MemoryMutationError("memory library changed while the platform was saving it")

    def _verify_bound_library_scan(
        self,
        root: Path,
        bound: BoundDocument,
        library_id: str,
        expected: MarkdownScan,
    ) -> None:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) != bound.root_identity:
                raise MemoryMutationError("memory library changed while saving")
            current = scan_markdown_fd(
                descriptor,
                root,
                self.library_ignore_patterns(library_id),
                (self.database_path.parent,),
            )
        finally:
            os.close(descriptor)
        if not current.complete or (
            current.documents,
            current.identities,
        ) != (
            expected.documents,
            expected.identities,
        ):
            raise MemoryMutationError("memory library changed while the platform was saving it")

    @staticmethod
    def _read_bound_name(
        bound: BoundDocument, name: str
    ) -> tuple[bytes, tuple[int, int]]:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=bound.parent_fd
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise MemoryMutationError("memory document is not a regular file")
            content = b""
            while chunk := os.read(descriptor, 1024 * 1024):
                content += chunk
            return content, (metadata.st_dev, metadata.st_ino)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_bound_document(bound: BoundDocument) -> tuple[bytes, tuple[int, int]]:
        return PlatformState._read_bound_name(bound, bound.name)

    @staticmethod
    def _replace_bound_document(
        bound: BoundDocument,
        content: bytes,
        expected_identity: tuple[int, int],
        expected_content: bytes,
        replacement: DocumentReplacement,
    ) -> None:
        metadata = os.stat(bound.name, dir_fd=bound.parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise MemoryMutationError("memory document is not a regular file")
        if (metadata.st_dev, metadata.st_ino) != expected_identity:
            raise MemoryMutationError("document changed while the platform was saving it")
        temporary = f".{bound.name}.{uuid.uuid4().hex}.tmp"
        remove_temporary = True
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                stat.S_IMODE(metadata.st_mode),
                dir_fd=bound.parent_fd,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary_metadata = os.fstat(stream.fileno())
                temporary_identity = (
                    temporary_metadata.st_dev,
                    temporary_metadata.st_ino,
                )
            _rename_exchange(bound.parent_fd, temporary, bound.parent_fd, bound.name)
            replacement.identity = temporary_identity
            replacement.renamed = True

            def roll_back_exchange() -> None:
                nonlocal remove_temporary
                remove_temporary = False
                try:
                    _rename_exchange(
                        bound.parent_fd, temporary, bound.parent_fd, bound.name
                    )
                except OSError as error:
                    raise MemoryMutationError(
                        "compensation conflict: atomic replacement rollback failed; "
                        "external content is "
                        f"preserved at {temporary}: {error}"
                    ) from error
                replacement.identity = None
                replacement.renamed = False
                raise MemoryMutationError(
                    "compensation conflict: atomic replacement rollback preserved a "
                    f"recovery entry at {temporary}"
                )

            try:
                exchanged_content, exchanged_identity = PlatformState._read_bound_name(
                    bound, temporary
                )
            except (MemoryMutationError, OSError):
                roll_back_exchange()
            if (
                exchanged_identity != expected_identity
                or exchanged_content != expected_content
            ):
                roll_back_exchange()
            replacement.expected_target_exchanged = True
            os.unlink(temporary, dir_fd=bound.parent_fd)
            remove_temporary = False
            os.fsync(bound.parent_fd)
            replaced_metadata = os.stat(
                bound.name, dir_fd=bound.parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(replaced_metadata.st_mode)
                or (replaced_metadata.st_dev, replaced_metadata.st_ino)
                != temporary_identity
            ):
                raise MemoryMutationError(
                    "document changed while the platform was saving it"
                )
        except OSError as error:
            raise MemoryMutationError(
                f"memory document cannot be replaced: {error}"
            ) from error
        finally:
            if remove_temporary:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=bound.parent_fd)

    @staticmethod
    def _verify_bound_document(
        root: Path,
        path: str,
        bound: BoundDocument,
        expected_identity: tuple[int, int],
    ) -> None:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parent_fd = root_fd
        try:
            root_metadata = os.fstat(root_fd)
            if (root_metadata.st_dev, root_metadata.st_ino) != bound.root_identity:
                raise MemoryMutationError("memory library changed while saving")
            for part in PurePosixPath(path).parts[:-1]:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                if parent_fd != root_fd:
                    os.close(parent_fd)
                parent_fd = child
            parent_metadata = os.fstat(parent_fd)
            if (parent_metadata.st_dev, parent_metadata.st_ino) != bound.parent_identity:
                raise MemoryMutationError("document parent changed while saving")
            document_metadata = os.stat(
                bound.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(document_metadata.st_mode)
                or (document_metadata.st_dev, document_metadata.st_ino)
                != expected_identity
            ):
                raise MemoryMutationError("document changed while the platform was saving it")
        except OSError as error:
            raise MemoryMutationError(
                f"document path changed while the platform was saving it: {error}"
            ) from error
        finally:
            if parent_fd != root_fd:
                os.close(parent_fd)
            os.close(root_fd)

    def _snapshot_library_index(self, library_id: str) -> LibraryIndexSnapshot:
        connection = self.connection_or_raise
        sync_status = str(
            connection.execute(
                "SELECT sync_status FROM memory_libraries WHERE id = ?", (library_id,)
            ).fetchone()[0]
        )
        documents = tuple(
            connection.execute(
                "SELECT id, library_id, path, source_version FROM memory_documents "
                "WHERE library_id = ? ORDER BY id",
                (library_id,),
            )
        )
        chunks = tuple(
            connection.execute(
                "SELECT id, document_id, library_id, path, kind, heading, start_line, "
                "end_line, content, source_version FROM memory_chunks "
                "WHERE library_id = ? ORDER BY id",
                (library_id,),
            )
        )
        chunk_ids = tuple(str(row[0]) for row in chunks)
        searches: tuple[tuple[object, ...], ...] = ()
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            searches = tuple(
                connection.execute(
                    f"SELECT chunk_id, content, heading, path FROM memory_chunk_search "
                    f"WHERE chunk_id IN ({placeholders}) ORDER BY chunk_id",  # noqa: S608
                    chunk_ids,
                )
            )
        vectors = tuple(
            connection.execute(
                "SELECT chunk_id, library_id, source_version, model, vector_json, created_at "
                "FROM memory_chunk_vectors WHERE library_id = ? ORDER BY chunk_id",
                (library_id,),
            )
        )
        return LibraryIndexSnapshot(sync_status, documents, chunks, searches, vectors)

    def _restore_library_index(
        self, library_id: str, snapshot: LibraryIndexSnapshot
    ) -> None:
        connection = self.connection_or_raise
        chunk_ids = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT id FROM memory_chunks WHERE library_id = ?", (library_id,)
            )
        )
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            connection.execute(
                f"DELETE FROM memory_chunk_search WHERE chunk_id IN ({placeholders})",  # noqa: S608
                chunk_ids,
            )
        connection.execute("DELETE FROM memory_documents WHERE library_id = ?", (library_id,))
        connection.executemany(
            "INSERT INTO memory_documents (id, library_id, path, source_version) "
            "VALUES (?, ?, ?, ?)",
            snapshot.documents,
        )
        connection.executemany(
            "INSERT INTO memory_chunks (id, document_id, library_id, path, kind, heading, "
            "start_line, end_line, content, source_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            snapshot.chunks,
        )
        connection.executemany(
            "INSERT INTO memory_chunk_search (chunk_id, content, heading, path) "
            "VALUES (?, ?, ?, ?)",
            snapshot.searches,
        )
        connection.executemany(
            "INSERT INTO memory_chunk_vectors "
            "(chunk_id, library_id, source_version, model, vector_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            snapshot.vectors,
        )
        connection.execute(
            "UPDATE memory_libraries SET sync_status = ? WHERE id = ?",
            (snapshot.sync_status, library_id),
        )

    @staticmethod
    def _atomic_document_replace(root: Path, path: str, content: str) -> None:
        parent, name = PlatformState._open_document_parent(root, path)
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise MemoryMutationError("memory document is not a regular file")
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                stat.S_IMODE(metadata.st_mode),
                dir_fd=parent,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=parent)
            os.close(parent)

    @staticmethod
    def _open_document_parent(root: Path, path: str) -> tuple[int, str]:
        parts = PurePosixPath(path).parts
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            return descriptor, parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    @contextmanager
    def _library_lock(self, library_id: str) -> Iterator[None]:
        lock_directory = self.database_path.parent / "locks"
        lock_directory.mkdir(mode=0o700, exist_ok=True)
        with (lock_directory / f"{library_id}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    async def search_project(
        self,
        requested_cwd: str,
        query: str,
        limit: int = 10,
        max_graph_hops: int = 1,
    ) -> dict[str, object]:
        if not query.strip():
            raise ValueError("search query must not be empty")
        binding = self.resolve_project_binding(requested_cwd)
        if binding["status"] == "unbound":
            return {
                "status": "unbound",
                "scope": {"kind": "project_cwd", "cwd": requested_cwd},
                "query": query,
                "results": [],
            }
        return await self._search_library(
            str(binding["library_id"]),
            query,
            limit,
            max_graph_hops,
            scope={
                "kind": "project_cwd",
                "cwd": requested_cwd,
                "project_id": binding["project_id"],
                "library_id": binding["library_id"],
            },
        )

    async def search_library(
        self,
        library_id: str,
        query: str,
        limit: int = 10,
        max_graph_hops: int = 1,
    ) -> dict[str, object]:
        if not query.strip():
            raise ValueError("search query must not be empty")
        library = self.library(library_id)
        if library is None:
            raise LibraryRegistrationError("memory library not found")
        return await self._search_library(
            library.id,
            query,
            limit,
            max_graph_hops,
            scope={"kind": "explicit_library", "library_id": library.id},
        )

    async def _search_library(
        self,
        library_id: str,
        query: str,
        limit: int,
        max_graph_hops: int,
        *,
        scope: dict[str, object],
    ) -> dict[str, object]:
        if not query.strip():
            raise ValueError("search query must not be empty")
        expression = fts_query(query)
        bounded_limit = max(1, min(limit, 100))
        candidate_limit = min(500, max(20, bounded_limit * 5))
        rows = self.connection_or_raise.execute(
            """SELECT chunk.id, chunk.document_id, chunk.content, chunk.library_id,
                      chunk.path, chunk.heading,
                      chunk.start_line, chunk.end_line, chunk.source_version,
                      bm25(memory_chunk_search) AS rank
               FROM memory_chunk_search
               JOIN memory_chunks AS chunk ON chunk.id = memory_chunk_search.chunk_id
               WHERE memory_chunk_search MATCH ? AND chunk.library_id = ?
               ORDER BY rank, chunk.path, chunk.start_line
               LIMIT ?""",
            (expression, library_id, candidate_limit),
        ).fetchall()
        seen = {str(row[0]) for row in rows}
        keyword = query.strip().casefold()
        if len(rows) < candidate_limit:
            fallback_rows = self.connection_or_raise.execute(
                """SELECT id, document_id, content, library_id, path, heading,
                          start_line, end_line, source_version, 0.0 AS rank
                   FROM memory_chunks
                   WHERE library_id = ?
                   ORDER BY path, start_line""",
                (library_id,),
            ).fetchall()
            rows.extend(
                row
                for row in fallback_rows
                if str(row[0]) not in seen
                and keyword
                in "\n".join((str(row[2]), "" if row[5] is None else str(row[5]))).casefold()
            )
            rows = rows[:candidate_limit]
        candidates: dict[str, dict[str, object]] = {}
        for rank, row in enumerate(rows):
            candidates[str(row[0])] = {
                "chunk_id": str(row[0]),
                "document_id": str(row[1]),
                "content": str(row[2]),
                "library_id": str(row[3]),
                "path": str(row[4]),
                "heading": None if row[5] is None else str(row[5]),
                "start_line": int(row[6]),
                "end_line": int(row[7]),
                "source_version": str(row[8]),
                "source_type": "markdown",
                "classification": "direct",
                "score": 1.0 / (60.0 + rank),
                "retrieval_sources": ["full_text"],
            }
        degradation: list[str] = []
        embedding = self.model_client.embedding
        vector_status = "not_configured" if embedding is None else "ready"
        if embedding is not None:
            latest_job = self.connection_or_raise.execute(
                "SELECT status FROM background_jobs WHERE kind = 'vector_rebuild' "
                "AND json_extract(payload, '$.library_id') = ? ORDER BY id DESC LIMIT 1",
                (library_id,),
            ).fetchone()
            if latest_job is not None and str(latest_job[0]) in {"pending", "running"}:
                vector_status = "building"
            elif latest_job is not None and str(latest_job[0]) == "error":
                vector_status = "error"
                degradation.append("vector_index_unavailable")
            index_row = self.connection_or_raise.execute(
                "SELECT model, dimension FROM memory_vector_indexes WHERE library_id = ?",
                (library_id,),
            ).fetchone()
            if vector_status != "ready":
                pass
            elif index_row is None or str(index_row[0]) != embedding.model:
                if vector_status != "building":
                    vector_status = "error"
                    if "vector_index_unavailable" not in degradation:
                        degradation.append("vector_index_unavailable")
            elif int(index_row[1]) > 0:
                index_dimension = int(index_row[1])
                try:
                    query_vector = (
                        await asyncio.to_thread(self.model_client.embed, [query])
                    )[0]
                    if len(query_vector) != index_dimension:
                        raise ModelServiceError("embedding dimension changed")
                    vector_rows = self.connection_or_raise.execute(
                        """SELECT chunk.id, chunk.document_id, chunk.content, chunk.library_id,
                                  chunk.path, chunk.heading, chunk.start_line, chunk.end_line,
                                  chunk.source_version, vector.vector_json
                           FROM memory_chunk_vectors AS vector
                           JOIN memory_chunks AS chunk ON chunk.id = vector.chunk_id
                           WHERE vector.library_id = ?
                             AND vector.source_version = chunk.source_version
                             AND vector.model = ?""",
                        (library_id, embedding.model),
                    ).fetchall()
                    decoded_rows: list[tuple[list[float], tuple[object, ...]]] = []
                    for row in vector_rows:
                        raw_vector = json.loads(str(row[9]))
                        if (
                            not isinstance(raw_vector, list)
                            or len(raw_vector) != index_dimension
                            or any(
                                not isinstance(value, (int, float))
                                or not math.isfinite(float(value))
                                for value in raw_vector
                            )
                        ):
                            raise ModelServiceError("stored vector index is malformed")
                        decoded_rows.append(([float(value) for value in raw_vector], row))
                    scored = sorted(
                        (
                            (cosine_similarity(query_vector, stored_vector), row)
                            for stored_vector, row in decoded_rows
                        ),
                        key=lambda item: (-item[0], str(item[1][4]), int(str(item[1][6]))),
                    )[:candidate_limit]
                    for rank, (similarity, row) in enumerate(scored):
                        if similarity <= 0.05:
                            continue
                        chunk_id = str(row[0])
                        semantic_score = max(0.0, similarity) / (60.0 + rank)
                        existing = candidates.get(chunk_id)
                        if existing is not None:
                            existing["score"] = cast(float, existing["score"]) + semantic_score
                            sources = existing["retrieval_sources"]
                            assert isinstance(sources, list)
                            sources.append("vector")
                        else:
                            candidates[chunk_id] = {
                                "chunk_id": chunk_id,
                                "document_id": str(row[1]),
                                "content": str(row[2]),
                                "library_id": str(row[3]),
                                "path": str(row[4]),
                                "heading": None if row[5] is None else str(row[5]),
                                "start_line": int(str(row[6])),
                                "end_line": int(str(row[7])),
                                "source_version": str(row[8]),
                                "source_type": "markdown",
                                "classification": "direct",
                                "score": semantic_score,
                                "retrieval_sources": ["vector"],
                            }
                except (ModelServiceError, json.JSONDecodeError, TypeError, ValueError):
                    vector_status = "error"
                    degradation.append("embedding_unavailable")
        results = sorted(
            candidates.values(),
            key=lambda item: (
                -cast(float, item["score"]),
                str(item["path"]),
                cast(int, item["start_line"]),
            ),
        )
        if self.model_client.reranker is not None and results:
            original = results
            try:
                reranked = await asyncio.to_thread(
                    self.model_client.rerank,
                    query,
                    [str(item["content"]) for item in original],
                )
                results = []
                for index, score in reranked:
                    item = dict(original[index])
                    item["score"] = score
                    results.append(item)
            except ModelServiceError:
                results = original
                degradation.append("reranker_unavailable")
        graph_status = self.graph_status(library_id)
        ordered_direct = results
        graph_capacity = (
            min(5, max(1, bounded_limit * 3 // 10))
            if graph_status["status"] == "ready" and bounded_limit >= 2
            else 0
        )
        results = ordered_direct[: bounded_limit - graph_capacity]
        graph_limit = graph_capacity
        if (
            results
            and graph_limit
            and self.graph_adapter is not None
            and graph_status["status"] == "ready"
        ):
            seed_document_ids = tuple(
                dict.fromkeys(str(item["document_id"]) for item in results)
            )
            try:
                expanded = await self.graph_adapter.expand(
                    library_id,
                    seed_document_ids,
                    max_hops=max(1, min(max_graph_hops, 2)),
                    limit=graph_limit,
                )
                current_sources: dict[
                    str, list[tuple[str, str, str, str, str | None, int, int]]
                ] = {}
                for row in self.connection_or_raise.execute(
                    "SELECT document.id, document.path, document.source_version, "
                    "chunk.id, chunk.content, chunk.heading, chunk.start_line, "
                    "chunk.end_line FROM memory_documents AS document "
                    "JOIN memory_chunks AS chunk ON chunk.document_id = document.id "
                    "WHERE document.library_id = ? "
                    "ORDER BY document.path, chunk.start_line, chunk.id",
                    (library_id,),
                ):
                    current_sources.setdefault(str(row[0]), []).append(
                        (
                            str(row[1]),
                            str(row[2]),
                            str(row[3]),
                            str(row[4]),
                            None if row[5] is None else str(row[5]),
                            int(row[6]),
                            int(row[7]),
                        )
                    )
                for expansion in expanded:
                    document_sources = current_sources.get(expansion.document_id)
                    if (
                        not document_sources
                        or document_sources[0][0] != expansion.path
                        or document_sources[0][1] != expansion.source_version
                    ):
                        continue
                    anchor = expansion.source_anchor.casefold()
                    source = next(
                        (
                            item
                            for item in document_sources
                            if anchor
                            and anchor
                            in "\n".join((item[3], item[4] or "")).casefold()
                        ),
                        document_sources[0],
                    )
                    results.append(
                        {
                            "chunk_id": f"graph:{source[2]}:{expansion.hop}",
                            "document_id": expansion.document_id,
                            "content": source[3],
                            "library_id": library_id,
                            "path": source[0],
                            "heading": source[4],
                            "start_line": source[5],
                            "end_line": source[6],
                            "source_version": source[1],
                            "source_type": "markdown",
                            "classification": "graph_expansion",
                            "graph_object_type": expansion.graph_object_type,
                            "graph_hop": expansion.hop,
                            "score": 0.0,
                            "retrieval_sources": ["graph"],
                        }
                    )
            except GraphAdapterError as error:
                degradation.append("graph_unavailable")
                self.connection_or_raise.execute(
                    "UPDATE memory_graph_indexes SET status = 'error', last_error = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE library_id = ?",
                    (str(error)[:1000], library_id),
                )
                self.connection_or_raise.commit()
                graph_status = self.graph_status(library_id)
        if (
            results
            and self.model_client.graph is not None
            and graph_status["status"] == "error"
            and "graph_unavailable" not in degradation
        ):
            degradation.append("graph_unavailable")
        included_chunks = {str(item["chunk_id"]) for item in results}
        for item in ordered_direct:
            if len(results) >= bounded_limit:
                break
            if str(item["chunk_id"]) not in included_chunks:
                results.append(item)
                included_chunks.add(str(item["chunk_id"]))
        for item in results:
            item["degraded"] = bool(degradation)
            item["degradation"] = list(degradation)
        response: dict[str, object] = {
            "status": "bound",
            "scope": scope,
            "library_id": library_id,
            "query": query,
            "degraded": bool(degradation),
            "degradation": degradation,
            "vector_index_status": vector_status,
            "graph_index_status": graph_status["status"],
            "results": results,
        }
        if "project_id" in scope:
            response["project_id"] = scope["project_id"]
        return response

    def graph_status(self, library_id: str) -> dict[str, object]:
        if self.library(library_id) is None:
            raise LibraryRegistrationError("memory library not found")
        row = self.connection_or_raise.execute(
            "SELECT status, total_documents, projected_documents, last_error, updated_at "
            "FROM memory_graph_indexes WHERE library_id = ?",
            (library_id,),
        ).fetchone()
        if row is None:
            return {
                "library_id": library_id,
                "status": "not_configured",
                "total_documents": 0,
                "projected_documents": 0,
                "last_error": "",
                "updated_at": None,
            }
        return {
            "library_id": library_id,
            "status": str(row[0]),
            "total_documents": int(row[1]),
            "projected_documents": int(row[2]),
            "last_error": str(row[3]),
            "updated_at": str(row[4]),
        }

    def bind_project(self, requested_root: str, library_id: str) -> ProjectBinding:
        root, device, inode = self._canonical_project_root(requested_root)
        self._require_project_library(library_id)
        binding_id = str(uuid.uuid4())
        try:
            self.connection_or_raise.execute(
                """INSERT INTO project_bindings
                   (id, kind, canonical_root, root_device, root_inode, library_id)
                   VALUES (?, 'project', ?, ?, ?, ?)""",
                (binding_id, root, device, inode, library_id),
            )
            self.connection_or_raise.commit()
        except sqlite3.IntegrityError as error:
            duplicate = self.connection_or_raise.execute(
                "SELECT id FROM project_bindings WHERE canonical_root = ?", (root,)
            ).fetchone()
            if duplicate is not None:
                raise DuplicateProjectBindingError(str(duplicate[0])) from error
            raise
        return ProjectBinding(binding_id, binding_id, "project", root, library_id, "available")

    def associate_worktree(
        self, requested_root: str, main_project_binding_id: str
    ) -> ProjectBinding:
        root, device, inode = self._canonical_project_root(requested_root)
        main = self._direct_binding(main_project_binding_id)
        binding_id = str(uuid.uuid4())
        try:
            self.connection_or_raise.execute(
                """INSERT INTO project_bindings
                   (id, kind, canonical_root, root_device, root_inode,
                    main_project_binding_id)
                   VALUES (?, 'worktree', ?, ?, ?, ?)""",
                (binding_id, root, device, inode, main.id),
            )
            self.connection_or_raise.commit()
        except sqlite3.IntegrityError as error:
            duplicate = self.connection_or_raise.execute(
                "SELECT id FROM project_bindings WHERE canonical_root = ?", (root,)
            ).fetchone()
            if duplicate is not None:
                raise DuplicateProjectBindingError(str(duplicate[0])) from error
            raise
        return ProjectBinding(binding_id, main.id, "worktree", root, main.library_id, "available")

    def list_project_bindings(self) -> list[ProjectBinding]:
        rows = self.connection_or_raise.execute(
            """SELECT binding.id, binding.kind, binding.canonical_root,
                      binding.root_device, binding.root_inode,
                      COALESCE(binding.main_project_binding_id, binding.id),
                      COALESCE(binding.library_id, main.library_id)
               FROM project_bindings AS binding
               LEFT JOIN project_bindings AS main
                 ON main.id = binding.main_project_binding_id
               ORDER BY binding.created_at, binding.id"""
        ).fetchall()
        return [
            ProjectBinding(
                id=str(row[0]),
                kind=row[1],
                canonical_root=str(row[2]),
                availability=self._project_root_availability(
                    Path(str(row[2])), int(row[3]), int(row[4])
                ),
                project_id=str(row[5]),
                library_id=str(row[6]),
            )
            for row in rows
        ]

    def update_project_binding(self, binding_id: str, library_id: str) -> ProjectBinding:
        binding = self._direct_binding(binding_id)
        self._require_project_library(library_id)
        self.connection_or_raise.execute(
            "UPDATE project_bindings SET library_id = ? WHERE id = ?",
            (library_id, binding_id),
        )
        self.connection_or_raise.commit()
        return ProjectBinding(
            binding.id,
            binding.project_id,
            binding.kind,
            binding.canonical_root,
            library_id,
            binding.availability,
        )

    def delete_project_binding(self, binding_id: str) -> None:
        cursor = self.connection_or_raise.execute(
            "DELETE FROM project_bindings WHERE id = ?", (binding_id,)
        )
        if cursor.rowcount == 0:
            raise ProjectBindingError("project binding not found")
        self.connection_or_raise.commit()

    def resolve_project_binding(self, requested_cwd: str) -> dict[str, str]:
        if not requested_cwd.strip():
            raise ProjectBindingError("Codex working directory must not be empty")
        requested = Path(requested_cwd).expanduser()
        try:
            cwd = requested.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ProjectBindingError(
                f"Codex working directory cannot be resolved: {error}"
            ) from error
        if not cwd.is_dir():
            raise ProjectBindingError("Codex working directory must be a directory")

        matches = [
            binding
            for binding in self.list_project_bindings()
            if binding.availability == "available"
            and (cwd == Path(binding.canonical_root) or cwd.is_relative_to(binding.canonical_root))
        ]
        if not matches:
            return {"status": "unbound", "cwd": os.path.normcase(str(cwd))}
        binding = max(matches, key=lambda item: len(Path(item.canonical_root).parts))
        if self._has_unbound_worktree_boundary(cwd, Path(binding.canonical_root)):
            return {"status": "unbound", "cwd": os.path.normcase(str(cwd))}
        return {
            "status": "bound",
            "project_id": binding.project_id,
            "binding_id": binding.id,
            "binding_kind": binding.kind,
            "project_root": binding.canonical_root,
            "library_id": binding.library_id,
        }

    @staticmethod
    def _has_unbound_worktree_boundary(cwd: Path, binding_root: Path) -> bool:
        current = cwd
        while current != binding_root:
            if PlatformState._is_linked_git_worktree(current, binding_root):
                return True
            current = current.parent
        return False

    @staticmethod
    def _is_linked_git_worktree(candidate: Path, main_root: Path) -> bool:
        expected_common_dir = PlatformState._git_common_dir(main_root)
        if expected_common_dir is None:
            return False
        worktree = PlatformState._linked_worktree_metadata(candidate)
        if worktree is None:
            return False
        _, common_dir = worktree
        return common_dir == expected_common_dir

    @staticmethod
    def _git_common_dir(root: Path) -> Path | None:
        dot_git = root / ".git"
        try:
            metadata = dot_git.lstat()
        except OSError:
            return None
        if stat.S_ISDIR(metadata.st_mode):
            try:
                common_dir = dot_git.resolve(strict=True)
            except (OSError, RuntimeError):
                return None
            return common_dir if PlatformState._valid_common_git_dir(common_dir) else None
        if not stat.S_ISREG(metadata.st_mode):
            return None
        worktree = PlatformState._linked_worktree_metadata(root)
        return None if worktree is None else worktree[1]

    @staticmethod
    def _linked_worktree_metadata(candidate: Path) -> tuple[Path, Path] | None:
        marker = PlatformState._read_small_regular_file(candidate / ".git")
        if marker is None or not marker.startswith("gitdir: "):
            return None
        raw_git_dir = marker.removeprefix("gitdir: ")
        git_dir = PlatformState._resolve_git_path(raw_git_dir, candidate)
        if git_dir is None or not PlatformState._is_real_directory(git_dir):
            return None

        raw_common_dir = PlatformState._read_small_regular_file(git_dir / "commondir")
        raw_backlink = PlatformState._read_small_regular_file(git_dir / "gitdir")
        if raw_common_dir is None or raw_backlink is None:
            return None
        common_dir = PlatformState._resolve_git_path(raw_common_dir, git_dir)
        backlink = PlatformState._resolve_git_path(raw_backlink, git_dir)
        if common_dir is None or backlink is None:
            return None
        try:
            expected_backlink = (candidate / ".git").resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if backlink != expected_backlink:
            return None
        if git_dir.parent != common_dir / "worktrees":
            return None
        if not PlatformState._is_real_directory(common_dir / "worktrees"):
            return None
        if not PlatformState._valid_common_git_dir(common_dir):
            return None
        if PlatformState._read_small_regular_file(git_dir / "HEAD") is None:
            return None
        return git_dir, common_dir

    @staticmethod
    def _resolve_git_path(raw_path: str, base: Path) -> Path | None:
        if not raw_path or "\x00" in raw_path:
            return None
        path = Path(raw_path)
        unresolved = path if path.is_absolute() else base / path
        try:
            return unresolved.resolve(strict=True)
        except (OSError, RuntimeError):
            return None

    @staticmethod
    def _is_real_directory(path: Path) -> bool:
        try:
            metadata = path.lstat()
        except OSError:
            return False
        return stat.S_ISDIR(metadata.st_mode)

    @staticmethod
    def _valid_common_git_dir(common_dir: Path) -> bool:
        if not PlatformState._is_real_directory(common_dir):
            return False
        return PlatformState._read_small_regular_file(common_dir / "HEAD") is not None

    @staticmethod
    def _read_small_regular_file(path: Path, limit: int = 4096) -> str | None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError:
            return None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
                return None
            content = os.read(descriptor, limit + 1)
        except OSError:
            return None
        finally:
            os.close(descriptor)
        if len(content) > limit or len(content) != metadata.st_size:
            return None
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if "\x00" in text or "\r" in text:
            return None
        lines = text.splitlines()
        if len(lines) != 1:
            return None
        return lines[0]

    @property
    def connection_or_raise(self) -> sqlite3.Connection:
        assert self.connection is not None
        return self.connection

    def _direct_binding(self, binding_id: str) -> ProjectBinding:
        binding = next(
            (item for item in self.list_project_bindings() if item.id == binding_id), None
        )
        if binding is None:
            raise ProjectBindingError("project binding not found")
        if binding.kind != "project":
            raise ProjectBindingError("worktree associations cannot own a memory library")
        return binding

    def _require_project_library(self, library_id: str) -> MemoryLibrary:
        library = self.library(library_id)
        if library is None:
            raise ProjectBindingError("memory library not found")
        if library.kind != "project":
            raise ProjectBindingError("project bindings require a project memory library")
        return library

    @staticmethod
    def _canonical_project_root(requested_root: str) -> tuple[str, int, int]:
        if not requested_root.strip():
            raise ProjectBindingError("project root must not be empty")
        requested = Path(requested_root).expanduser()
        try:
            root = requested.resolve(strict=True)
            metadata = root.stat()
        except (OSError, RuntimeError) as error:
            raise ProjectBindingError(f"project root cannot be resolved: {error}") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProjectBindingError("project root must be a directory")
        if not os.access(root, os.R_OK | os.X_OK):
            raise ProjectBindingError("project root is not accessible to the daemon user")
        return os.path.normcase(str(root)), metadata.st_dev, metadata.st_ino

    @staticmethod
    def _project_root_availability(path: Path, device: int, inode: int) -> str:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return "missing"
        except OSError:
            return "unavailable"
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            return "replaced"
        if metadata.st_dev != device or metadata.st_ino != inode:
            return "replaced"
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return "unavailable"
        if os.path.normcase(str(resolved)) != os.path.normcase(str(path)):
            return "replaced"
        if not os.access(path, os.R_OK | os.X_OK):
            return "unavailable"
        return "available"

    def _validate_library_path(self, path: Path) -> None:
        if not self.library_roots:
            raise LibraryRegistrationError("no allowed memory library roots are configured")
        if not any(path == root or path.is_relative_to(root) for root in self.library_roots):
            roots = ", ".join(str(root) for root in self.library_roots)
            raise LibraryRegistrationError(f"path is outside allowed roots: {roots}")

    @staticmethod
    def _require_accessible_directory(path: Path) -> None:
        try:
            mode = path.stat().st_mode
        except OSError as error:
            raise LibraryRegistrationError(
                f"memory library path is not accessible: {error.strerror or error}"
            ) from error
        if not stat.S_ISDIR(mode):
            raise LibraryRegistrationError("memory library path must be a directory")
        if mode & 0o222 == 0:
            raise LibraryRegistrationError("memory library directory must be writable")
        if mode & 0o444 == 0 or mode & 0o111 == 0:
            raise LibraryRegistrationError(
                "memory library directory must be readable and searchable"
            )
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            raise LibraryRegistrationError(
                "memory library directory is not accessible to the daemon user"
            )

    def _availability(self, path: Path) -> str:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return "missing"
        except OSError:
            return "unavailable"
        if stat.S_ISLNK(metadata.st_mode):
            return "unavailable"

        try:
            current_path = path.resolve(strict=True)
            if os.path.normcase(str(current_path)) != os.path.normcase(str(path)):
                return "unavailable"
            self._validate_library_path(current_path)
            self._require_accessible_directory(current_path)
        except (LibraryRegistrationError, OSError, RuntimeError):
            return "unavailable"
        return "available"

    async def _execute_pending_jobs(self) -> None:
        assert self.connection is not None
        job = self.connection.execute(
            "SELECT id, kind, payload, attempts FROM background_jobs "
            "WHERE status = 'pending' "
            "AND available_at <= (julianday('now') - 2440587.5) * 86400.0 "
            "ORDER BY available_at, id LIMIT 1"
        ).fetchone()
        if job is None:
            return
        job_id, kind, payload, attempts = job
        claimed = self.connection.execute(
            "UPDATE background_jobs SET status = 'running' "
            "WHERE id = ? AND status = 'pending'",
            (job_id,),
        )
        self.connection.commit()
        if claimed.rowcount != 1:
            return
        try:
            if str(kind) == "vector_rebuild":
                parsed = json.loads(str(payload))
                await self._rebuild_vectors(str(parsed["library_id"]))
            elif str(kind) == "graph_rebuild":
                parsed = json.loads(str(payload))
                await self._rebuild_graph(str(parsed["library_id"]))
            elif str(kind) == "capture_consolidation":
                parsed = json.loads(str(payload))
                await self._consolidate_capture_round(
                    str(parsed["session_id"]), str(parsed["turn_id"])
                )
            self.connection.execute(
                "UPDATE background_jobs SET status = 'done' WHERE id = ?", (job_id,)
            )
        except (
            CandidateGovernanceError,
            GraphAdapterError,
            ModelServiceError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            self.connection.rollback()
            if str(kind) == "capture_consolidation":
                with suppress(KeyError, TypeError, ValueError, json.JSONDecodeError):
                    parsed_capture = json.loads(str(payload))
                    retry = int(attempts) < 5
                    self.connection.execute(
                        """UPDATE capture_rounds SET status = ?, last_error = ?,
                           updated_at = CURRENT_TIMESTAMP
                           WHERE session_id = ? AND turn_id = ?""",
                        (
                            "pending" if retry else "error",
                            str(error)[:1000],
                            str(parsed_capture["session_id"]),
                            str(parsed_capture["turn_id"]),
                        ),
                    )
            if str(kind) == "graph_rebuild":
                with suppress(KeyError, TypeError, ValueError, json.JSONDecodeError):
                    library_id = str(json.loads(str(payload))["library_id"])
                    self.connection.execute(
                        "INSERT INTO memory_graph_indexes "
                        "(library_id, status, last_error) VALUES (?, 'error', ?) "
                        "ON CONFLICT(library_id) DO UPDATE SET status = 'error', "
                        "last_error = excluded.last_error, updated_at = CURRENT_TIMESTAMP",
                        (library_id, str(error)[:1000]),
                    )
            if str(kind) == "capture_consolidation" and int(attempts) < 5:
                retry_delay = min(0.5 * (2 ** int(attempts)), 8.0)
                self.connection.execute(
                    """UPDATE background_jobs
                       SET status = 'pending', attempts = attempts + 1,
                           available_at = (julianday('now') - 2440587.5) * 86400.0 + ?
                       WHERE id = ?""",
                    (retry_delay, job_id),
                )
            else:
                self.connection.execute(
                    "UPDATE background_jobs SET status = 'error' WHERE id = ?", (job_id,)
                )
        self.connection.commit()

    def _queue_capture_consolidation(
        self, session_id: str, turn_id: str, participate_in_transaction: bool
    ) -> int:
        pending = self.connection_or_raise.execute(
            """SELECT id FROM background_jobs WHERE kind = 'capture_consolidation'
               AND status IN ('pending', 'running')
               AND json_extract(payload, '$.session_id') = ?
               AND json_extract(payload, '$.turn_id') = ? LIMIT 1""",
            (session_id, turn_id),
        ).fetchone()
        if pending is not None:
            return 0
        cursor = self.connection_or_raise.execute(
            "INSERT INTO background_jobs (kind, payload) VALUES ('capture_consolidation', ?)",
            (json.dumps({"session_id": session_id, "turn_id": turn_id}),),
        )
        if not participate_in_transaction:
            self.connection_or_raise.commit()
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    async def _consolidate_capture_round(self, session_id: str, turn_id: str) -> None:
        rows = self.connection_or_raise.execute(
            """SELECT event_id, library_id, event_kind, content, occurred_at
               FROM capture_inbox WHERE session_id = ? AND turn_id = ?
               ORDER BY CASE event_kind WHEN 'user' THEN 0 ELSE 1 END""",
            (session_id, turn_id),
        ).fetchall()
        by_kind = {str(row[2]): row for row in rows}
        if set(by_kind) != {"user", "assistant"}:
            return
        library_ids = {str(row[1]) for row in rows}
        if len(library_ids) != 1:
            raise CandidateGovernanceError("capture round crosses memory libraries")
        library_id = library_ids.pop()
        self.connection_or_raise.execute(
            """UPDATE capture_rounds SET status = 'processing', last_error = '',
               updated_at = CURRENT_TIMESTAMP WHERE session_id = ? AND turn_id = ?""",
            (session_id, turn_id),
        )
        self.connection_or_raise.commit()
        conversation = (
            f"User:\n{by_kind['user'][3]}\n\n"
            f"Assistant final reply:\n{by_kind['assistant'][3]}"
        )
        extracted = await asyncio.to_thread(self.model_client.extract_candidate, conversation)
        allowed_types = {
            "preference",
            "decision",
            "constraint",
            "domain_fact",
            "reusable_experience",
            "external_reference",
        }
        candidate_id: str | None = None
        if extracted.get("eligible") is True:
            suggested_type = extracted.get("suggested_type")
            body = extracted.get("body")
            if suggested_type not in allowed_types or not isinstance(body, str):
                raise ModelServiceError("candidate extraction returned malformed data")
            source_references = tuple(
                f"capture:{session_id}:{turn_id}:{str(row[0])}:{str(row[4])}" for row in rows
            )
            candidate = self.create_candidate(
                library_id,
                str(suggested_type),
                body,
                source_references,
                "session-capture",
                "capture-round:"
                + hashlib.sha256(f"{session_id}\0{turn_id}".encode()).hexdigest(),
            )
            candidate_id = str(candidate["id"])
        self.connection_or_raise.execute(
            """UPDATE capture_rounds SET status = 'done', candidate_id = ?, last_error = '',
               updated_at = CURRENT_TIMESTAMP WHERE session_id = ? AND turn_id = ?""",
            (candidate_id, session_id, turn_id),
        )
        self.connection_or_raise.execute(
            """UPDATE capture_inbox SET consolidated_at = CURRENT_TIMESTAMP
               WHERE session_id = ? AND turn_id = ?""",
            (session_id, turn_id),
        )
        self.connection_or_raise.commit()

    def _queue_vector_rebuild(self, library_id: str, participate_in_transaction: bool) -> int:
        if self.model_client.embedding is None:
            return 0
        pending = self.connection_or_raise.execute(
            "SELECT id FROM background_jobs WHERE kind = 'vector_rebuild' "
            "AND status = 'pending' AND json_extract(payload, '$.library_id') = ? "
            "ORDER BY id LIMIT 1",
            (library_id,),
        ).fetchone()
        if pending is not None:
            return int(pending[0])
        cursor = self.connection_or_raise.execute(
            "INSERT INTO background_jobs (kind, payload) VALUES ('vector_rebuild', ?)",
            (json.dumps({"library_id": library_id}),),
        )
        if not participate_in_transaction:
            self.connection_or_raise.commit()
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    def _queue_graph_rebuild(self, library_id: str, participate_in_transaction: bool) -> int:
        if self.graph_adapter is None or self.model_client.graph is None:
            self.connection_or_raise.execute(
                "INSERT INTO memory_graph_indexes (library_id, status) "
                "VALUES (?, 'not_configured') ON CONFLICT(library_id) DO NOTHING",
                (library_id,),
            )
            if not participate_in_transaction:
                self.connection_or_raise.commit()
            return 0
        pending = self.connection_or_raise.execute(
            "SELECT id FROM background_jobs WHERE kind = 'graph_rebuild' "
            "AND status = 'pending' AND json_extract(payload, '$.library_id') = ? "
            "ORDER BY id LIMIT 1",
            (library_id,),
        ).fetchone()
        if pending is not None:
            return int(pending[0])
        cursor = self.connection_or_raise.execute(
            "INSERT INTO background_jobs (kind, payload) VALUES ('graph_rebuild', ?)",
            (json.dumps({"library_id": library_id}),),
        )
        self.connection_or_raise.execute(
            "INSERT INTO memory_graph_indexes (library_id, status, last_error) "
            "VALUES (?, 'building', '') ON CONFLICT(library_id) DO UPDATE SET "
            "status = 'building', last_error = '', updated_at = CURRENT_TIMESTAMP",
            (library_id,),
        )
        if not participate_in_transaction:
            self.connection_or_raise.commit()
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    async def _rebuild_graph(self, library_id: str) -> None:
        adapter = self.graph_adapter
        if adapter is None:
            raise GraphAdapterError("graph projection is not configured")
        rows = self.connection_or_raise.execute(
            "SELECT document.id, document.path, document.source_version, "
            "group_concat(chunk.content, char(10) || char(10)) "
            "FROM memory_documents AS document "
            "LEFT JOIN memory_chunks AS chunk ON chunk.document_id = document.id "
            "WHERE document.library_id = ? GROUP BY document.id "
            "ORDER BY document.path",
            (library_id,),
        ).fetchall()
        documents = tuple(
            GraphSourceDocument(str(row[0]), str(row[1]), str(row[2]), str(row[3] or ""))
            for row in rows
        )
        self.connection_or_raise.execute(
            "UPDATE memory_graph_indexes SET total_documents = ?, "
            "projected_documents = 0, updated_at = CURRENT_TIMESTAMP WHERE library_id = ?",
            (len(documents), library_id),
        )
        self.connection_or_raise.commit()
        await adapter.rebuild(library_id, documents)
        current = {
            (str(row[0]), str(row[1]))
            for row in self.connection_or_raise.execute(
                "SELECT id, source_version FROM memory_documents WHERE library_id = ?",
                (library_id,),
            )
        }
        expected = {(document.document_id, document.source_version) for document in documents}
        if current != expected:
            self._queue_graph_rebuild(library_id, False)
            return
        self.connection_or_raise.execute("BEGIN IMMEDIATE")
        self.connection_or_raise.execute(
            "DELETE FROM memory_graph_documents WHERE library_id = ?", (library_id,)
        )
        self.connection_or_raise.executemany(
            "INSERT INTO memory_graph_documents "
            "(library_id, document_id, path, source_version) VALUES (?, ?, ?, ?)",
            [
                (library_id, item.document_id, item.path, item.source_version)
                for item in documents
            ],
        )
        self.connection_or_raise.execute(
            "UPDATE memory_graph_indexes SET status = 'ready', total_documents = ?, "
            "projected_documents = ?, last_error = '', updated_at = CURRENT_TIMESTAMP "
            "WHERE library_id = ?",
            (len(documents), len(documents), library_id),
        )
        self.connection_or_raise.commit()

    async def _rebuild_vectors(self, library_id: str) -> None:
        endpoint = self.model_client.embedding
        if endpoint is None:
            raise ModelServiceError("embedding is not configured")
        rows = self.connection_or_raise.execute(
            "SELECT id, content, source_version FROM memory_chunks "
            "WHERE library_id = ? ORDER BY id",
            (library_id,),
        ).fetchall()
        rebuilt: list[tuple[str, str, str, str, str]] = []
        dimension: int | None = None
        for offset in range(0, len(rows), 32):
            batch = rows[offset : offset + 32]
            vectors = await asyncio.to_thread(
                self.model_client.embed, [str(row[1]) for row in batch]
            )
            batch_dimension = len(vectors[0])
            if dimension is None:
                dimension = batch_dimension
            elif dimension != batch_dimension:
                raise ModelServiceError("embedding dimension changed during rebuild")
            rebuilt.extend(
                (
                    str(row[0]),
                    library_id,
                    str(row[2]),
                    endpoint.model,
                    json.dumps(vector, separators=(",", ":")),
                )
                for row, vector in zip(batch, vectors, strict=True)
            )
        self.connection_or_raise.execute("BEGIN IMMEDIATE")
        current = {
            (str(row[0]), str(row[1]))
            for row in self.connection_or_raise.execute(
                "SELECT id, source_version FROM memory_chunks WHERE library_id = ?",
                (library_id,),
            )
        }
        expected = {(row[0], row[2]) for row in rebuilt}
        if current != expected:
            self.connection_or_raise.rollback()
            self._queue_vector_rebuild(library_id, False)
            return
        self.connection_or_raise.execute(
            "DELETE FROM memory_chunk_vectors WHERE library_id = ?", (library_id,)
        )
        self.connection_or_raise.executemany(
            "INSERT INTO memory_chunk_vectors "
            "(chunk_id, library_id, source_version, model, vector_json) VALUES (?, ?, ?, ?, ?)",
            rebuilt,
        )
        self.connection_or_raise.execute(
            "INSERT INTO memory_vector_indexes (library_id, model, dimension) VALUES (?, ?, ?) "
            "ON CONFLICT(library_id) DO UPDATE SET model = excluded.model, "
            "dimension = excluded.dimension, updated_at = CURRENT_TIMESTAMP",
            (library_id, endpoint.model, 0 if dimension is None else dimension),
        )
        self.connection_or_raise.commit()

    def _metadata(self, key: str) -> str | None:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT value FROM platform_metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def _set_metadata(self, key: str, value: str) -> None:
        assert self.connection is not None
        self.connection.execute(
            """
            INSERT INTO platform_metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def _diff_lines(diff: str) -> list[dict[str, str]]:
    result = []
    for line in diff.splitlines():
        kind = "context"
        if line.startswith("+++") or line.startswith("---"):
            kind = "file"
        elif line.startswith("@@"):
            kind = "hunk"
        elif line.startswith("+"):
            kind = "addition"
        elif line.startswith("-"):
            kind = "deletion"
        result.append({"kind": kind, "text": line})
    return result
