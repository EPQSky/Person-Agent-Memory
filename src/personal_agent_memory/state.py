from __future__ import annotations

import asyncio
import os
import sqlite3
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

LibraryKind = Literal["user", "project"]
BindingKind = Literal["project", "worktree"]


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
                """
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

    def register_library(self, requested_path: str, kind: LibraryKind) -> MemoryLibrary:
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
        self.connection.execute(
            "INSERT INTO memory_libraries (id, kind, canonical_path) VALUES (?, ?, ?)",
            (library_id, kind, canonical_path),
        )
        self.connection.commit()
        return MemoryLibrary(library_id, kind, canonical_path, "available", "not_started")

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
