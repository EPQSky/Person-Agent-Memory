from __future__ import annotations

import asyncio
import ctypes
import difflib
import errno
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from personal_agent_memory.git_history import (
    GitHistoryError,
    GitInitialization,
    GitRepository,
    allowed_markdown_path,
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


@dataclass(frozen=True, slots=True)
class MemoryLibrary:
    id: str
    kind: LibraryKind
    canonical_path: str
    availability: str
    sync_status: str

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


class PlatformState:
    def __init__(self, database_path: Path, library_roots: tuple[Path, ...] = ()) -> None:
        self.database_path = database_path
        self.library_roots = library_roots
        self.connection: sqlite3.Connection | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self.stop_worker = asyncio.Event()
        self.startup_count = 0
        self.previous_shutdown_clean = True

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
            self._execute_pending_jobs()
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
            """SELECT id, kind, canonical_path, sync_status
               FROM memory_libraries ORDER BY created_at, id"""
        ).fetchall()
        return [
            MemoryLibrary(
                id=str(row[0]),
                kind=row[1],
                canonical_path=str(row[2]),
                availability=self._availability(Path(str(row[2]))),
                sync_status=str(row[3]),
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
        return LibraryIndexSnapshot(sync_status, documents, chunks, searches)

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
    def search_project(self, requested_cwd: str, query: str, limit: int = 10) -> dict[str, object]:
        if not query.strip():
            raise ValueError("search query must not be empty")
        expression = fts_query(query)
        binding = self.resolve_project_binding(requested_cwd)
        if binding["status"] == "unbound":
            return {"status": "unbound", "query": query, "results": []}
        library_id = binding["library_id"]
        bounded_limit = max(1, min(limit, 100))
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
            (expression, library_id, bounded_limit),
        ).fetchall()
        seen = {str(row[0]) for row in rows}
        keyword = query.strip().casefold()
        if len(rows) < bounded_limit:
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
            rows = rows[:bounded_limit]
        results = [
            {
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
                "score": float(-row[9]),
            }
            for row in rows
        ]
        return {
            "status": "bound",
            "project_id": binding["project_id"],
            "library_id": library_id,
            "query": query,
            "results": results,
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

    def _execute_pending_jobs(self) -> None:
        assert self.connection is not None
        self.connection.execute(
            "UPDATE background_jobs SET status = 'done' WHERE status = 'pending'"
        )
        self.connection.commit()

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
