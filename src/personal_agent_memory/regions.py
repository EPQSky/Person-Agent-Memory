from __future__ import annotations

import os
import sqlite3
import stat
import threading
import uuid
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

RegionKind = Literal["user", "topic", "project", "session"]
RegionBindingKind = Literal["project", "worktree"]
MAX_REGION_ID_LENGTH = 120
MAX_REGION_NAME_LENGTH = 200
MAX_SESSION_ID_LENGTH = 256


class RegionError(ValueError):
    """Raised when a memory region operation is invalid."""


class RegionNotFoundError(RegionError):
    """Raised when a requested memory region does not exist."""


class DuplicateRegionError(RegionError):
    def __init__(self, region_id: str) -> None:
        super().__init__(f"memory region already exists: {region_id}")
        self.region_id = region_id


class RegionBindingError(RegionError):
    """Raised when a project or worktree binding is invalid."""


class RegionBindingNotFoundError(RegionBindingError):
    """Raised when a requested project binding does not exist."""


class DuplicateRegionBindingError(RegionBindingError):
    def __init__(self, binding_id: str) -> None:
        super().__init__(f"project root is already bound: {binding_id}")
        self.binding_id = binding_id


@dataclass(frozen=True, slots=True)
class MemoryRegion:
    region_id: str
    kind: RegionKind
    name: str
    space: str
    session_id: str | None = None
    temporary: bool = False

    def payload(self) -> dict[str, object]:
        return {
            "id": self.region_id,
            "region_id": self.region_id,
            "kind": self.kind,
            "name": self.name,
            "space": self.space,
            "session_id": self.session_id,
            "temporary": self.temporary,
        }


@dataclass(frozen=True, slots=True)
class RegionBinding:
    binding_id: str
    kind: RegionBindingKind
    canonical_root: str
    region_id: str
    main_binding_id: str | None
    availability: str

    def payload(self) -> dict[str, str]:
        return {
            "id": self.binding_id,
            "binding_id": self.binding_id,
            "kind": self.kind,
            "canonical_root": self.canonical_root,
            "region_id": self.region_id,
            "library_id": self.region_id,
            "project_id": self.main_binding_id or self.binding_id,
            "main_binding_id": self.main_binding_id or "",
            "availability": self.availability,
        }


@dataclass(frozen=True, slots=True)
class ResolvedRegion:
    binding: RegionBinding
    region: MemoryRegion
    cwd: str


def _normalize_region_id(region_id: str | None, kind: RegionKind) -> str:
    if kind == "user":
        if region_id is not None and region_id.strip() not in {"", "user"}:
            raise RegionError("the user region identifier must be 'user'")
        return "user"
    value = (region_id or str(uuid.uuid4())).strip()
    if not value:
        raise RegionError("memory region identifier must not be blank")
    if value == "user":
        raise RegionError("memory region identifier 'user' is reserved for user regions")
    if len(value) > MAX_REGION_ID_LENGTH:
        raise RegionError(
            f"memory region identifier exceeds the {MAX_REGION_ID_LENGTH}-character limit"
        )
    if "/" in value or "\\" in value or any(character.isspace() for character in value):
        raise RegionError("memory region identifier must not contain whitespace or path separators")
    return value


def _space_for(kind: RegionKind, region_id: str) -> str:
    if kind == "user":
        return "user"
    if kind == "topic":
        return f"topic:{region_id}"
    if kind == "project":
        return f"project:{region_id}"
    return "chat"


class RegionRegistry:
    """Persistent central registry for memory regions and explicit project bindings."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialized = False
        self._initialization_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        if not self._initialized:
            with self._initialization_lock:
                if not self._initialized:
                    connection.execute("PRAGMA journal_mode=WAL")
                    self._initialize(connection)
                    self._initialized = True
        return connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_regions (
                region_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('user', 'topic', 'project', 'session')),
                name TEXT NOT NULL,
                space TEXT NOT NULL,
                session_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK(
                    (kind = 'session' AND space = 'chat' AND session_id IS NOT NULL)
                    OR
                    (kind != 'session' AND session_id IS NULL)
                )
            );
            CREATE UNIQUE INDEX IF NOT EXISTS memory_regions_one_user
                ON memory_regions(kind) WHERE kind = 'user';
            CREATE UNIQUE INDEX IF NOT EXISTS memory_regions_session
                ON memory_regions(session_id) WHERE kind = 'session';
            CREATE TABLE IF NOT EXISTS region_bindings (
                binding_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('project', 'worktree')),
                canonical_root TEXT NOT NULL UNIQUE,
                root_device INTEGER NOT NULL,
                root_inode INTEGER NOT NULL,
                region_id TEXT NOT NULL REFERENCES memory_regions(region_id),
                main_binding_id TEXT REFERENCES region_bindings(binding_id)
                    ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK(
                    (kind = 'project' AND main_binding_id IS NULL)
                    OR
                    (kind = 'worktree' AND main_binding_id IS NOT NULL)
                )
            );
            CREATE INDEX IF NOT EXISTS region_bindings_region
                ON region_bindings(region_id);
            """
        )
        connection.commit()

    def create_region(
        self,
        kind: RegionKind,
        name: str,
        *,
        region_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryRegion:
        normalized_name = name.strip()
        if not normalized_name:
            raise RegionError("memory region name must not be blank")
        if len(normalized_name) > MAX_REGION_NAME_LENGTH:
            raise RegionError(
                f"memory region name exceeds the {MAX_REGION_NAME_LENGTH}-character limit"
            )
        normalized_session = session_id.strip() if session_id is not None else None
        if kind == "session":
            if not normalized_session:
                raise RegionError("session memory regions require a session_id")
            if len(normalized_session) > MAX_SESSION_ID_LENGTH:
                raise RegionError(
                    f"session_id exceeds the {MAX_SESSION_ID_LENGTH}-character limit"
                )
        elif normalized_session is not None:
            raise RegionError("session_id is only valid for session memory regions")

        identifier = _normalize_region_id(region_id, kind)
        region = MemoryRegion(
            region_id=identifier,
            kind=kind,
            name=normalized_name,
            space=_space_for(kind, identifier),
            session_id=normalized_session,
        )
        with closing(self._connect()) as connection, connection:
            try:
                connection.execute(
                    """INSERT INTO memory_regions
                       (region_id, kind, name, space, session_id)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        region.region_id,
                        region.kind,
                        region.name,
                        region.space,
                        region.session_id,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as error:
                duplicate = connection.execute(
                    "SELECT region_id FROM memory_regions WHERE region_id = ?",
                    (region.region_id,),
                ).fetchone()
                if duplicate is not None:
                    raise DuplicateRegionError(str(duplicate[0])) from error
                duplicate_session = (
                    connection.execute(
                        "SELECT region_id FROM memory_regions WHERE session_id = ?",
                        (normalized_session,),
                    ).fetchone()
                    if normalized_session is not None
                    else None
                )
                if duplicate_session is not None:
                    raise DuplicateRegionError(str(duplicate_session[0])) from error
                raise
        return region

    def ensure_user_region(self) -> MemoryRegion:
        existing = self.get_region("user")
        if existing is not None:
            if existing.kind != "user":
                raise RegionError(
                    "memory region identifier 'user' is reserved for user regions; "
                    f"found an existing {existing.kind} region"
                )
            return existing
        try:
            return self.create_region("user", "用户记忆")
        except DuplicateRegionError:
            existing = self.get_region("user")
            if existing is None:
                raise
            return existing

    def list_regions(self) -> list[MemoryRegion]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """SELECT region_id, kind, name, space, session_id
                   FROM memory_regions
                   ORDER BY CASE kind
                       WHEN 'user' THEN 0 WHEN 'topic' THEN 1
                       WHEN 'project' THEN 2 ELSE 3 END,
                       created_at, region_id"""
            ).fetchall()
        return [self._region_from_row(row) for row in rows]

    def get_region(self, region_id: str) -> MemoryRegion | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """SELECT region_id, kind, name, space, session_id
                   FROM memory_regions WHERE region_id = ?""",
                (region_id,),
            ).fetchone()
        return None if row is None else self._region_from_row(row)

    def require_region(self, region_id: str) -> MemoryRegion:
        region = self.get_region(region_id)
        if region is None:
            raise RegionNotFoundError("memory region not found")
        return region

    def region_tree(self, session_id: str | None = None) -> dict[str, object]:
        root = self.ensure_user_region()
        regions = self.list_regions()
        current_session: MemoryRegion | None = None
        normalized_session = session_id.strip() if session_id is not None else ""
        if normalized_session:
            persisted = next(
                (
                    region
                    for region in regions
                    if region.kind == "session" and region.session_id == normalized_session
                ),
                None,
            )
            current_session = (
                replace(persisted, temporary=True)
                if persisted is not None
                else MemoryRegion(
                    region_id=f"session:{normalized_session}",
                    kind="session",
                    name=f"当前会话 {normalized_session}",
                    space="chat",
                    session_id=normalized_session,
                    temporary=True,
                )
            )
        return {
            "root": root.payload(),
            "topics": [region.payload() for region in regions if region.kind == "topic"],
            "projects": [region.payload() for region in regions if region.kind == "project"],
            "current_session": (
                None if current_session is None else current_session.payload()
            ),
        }

    def bind_project(self, project_root: str, region_id: str) -> RegionBinding:
        root, device, inode = self._canonical_root(project_root)
        region = self.require_region(region_id)
        if region.kind != "project":
            raise RegionBindingError("project memory region is required")
        binding_id = str(uuid.uuid4())
        with closing(self._connect()) as connection, connection:
            try:
                connection.execute(
                    """INSERT INTO region_bindings
                       (binding_id, kind, canonical_root, root_device, root_inode, region_id)
                       VALUES (?, 'project', ?, ?, ?, ?)""",
                    (binding_id, root, device, inode, region.region_id),
                )
                connection.commit()
            except sqlite3.IntegrityError as error:
                duplicate = connection.execute(
                    "SELECT binding_id FROM region_bindings WHERE canonical_root = ?",
                    (root,),
                ).fetchone()
                if duplicate is not None:
                    raise DuplicateRegionBindingError(str(duplicate[0])) from error
                raise
        return self._require_binding(binding_id)

    def associate_worktree(self, worktree_root: str, main_binding_id: str) -> RegionBinding:
        root, device, inode = self._canonical_root(worktree_root)
        main = self._require_binding(main_binding_id)
        if main.kind != "project":
            raise RegionBindingError("main project binding is required")
        binding_id = str(uuid.uuid4())
        with closing(self._connect()) as connection, connection:
            try:
                connection.execute(
                    """INSERT INTO region_bindings
                       (binding_id, kind, canonical_root, root_device, root_inode,
                        region_id, main_binding_id)
                       VALUES (?, 'worktree', ?, ?, ?, ?, ?)""",
                    (binding_id, root, device, inode, main.region_id, main.binding_id),
                )
                connection.commit()
            except sqlite3.IntegrityError as error:
                duplicate = connection.execute(
                    "SELECT binding_id FROM region_bindings WHERE canonical_root = ?",
                    (root,),
                ).fetchone()
                if duplicate is not None:
                    raise DuplicateRegionBindingError(str(duplicate[0])) from error
                raise
        return self._require_binding(binding_id)

    def list_bindings(self) -> list[RegionBinding]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """SELECT binding.binding_id, binding.kind, binding.canonical_root,
                          binding.root_device, binding.root_inode,
                          COALESCE(main.region_id, binding.region_id),
                          binding.main_binding_id
                   FROM region_bindings AS binding
                   LEFT JOIN region_bindings AS main
                     ON main.binding_id = binding.main_binding_id
                   ORDER BY binding.created_at, binding.binding_id"""
            ).fetchall()
        return [self._binding_from_row(row) for row in rows]

    def update_binding(self, binding_id: str, region_id: str) -> RegionBinding:
        binding = self._require_binding(binding_id)
        if binding.kind != "project":
            raise RegionBindingError("only a main project binding can be changed")
        region = self.require_region(region_id)
        if region.kind != "project":
            raise RegionBindingError("project memory region is required")
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE region_bindings SET region_id = ? WHERE binding_id = ?",
                (region.region_id, binding_id),
            )
            connection.commit()
        return self._require_binding(binding_id)

    def delete_binding(self, binding_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM region_bindings WHERE binding_id = ?", (binding_id,)
            )
            if cursor.rowcount == 0:
                raise RegionBindingNotFoundError("project binding not found")
            connection.commit()

    def resolve(self, requested_cwd: str) -> ResolvedRegion | None:
        cwd = self._canonical_cwd(requested_cwd)
        matches = [
            binding
            for binding in self.list_bindings()
            if binding.availability == "available"
            and (cwd == Path(binding.canonical_root) or cwd.is_relative_to(binding.canonical_root))
        ]
        if not matches:
            return None
        binding = max(matches, key=lambda item: len(Path(item.canonical_root).parts))
        if self._has_unbound_worktree_boundary(cwd, Path(binding.canonical_root)):
            return None
        region = self.require_region(binding.region_id)
        return ResolvedRegion(binding, region, os.path.normcase(str(cwd)))

    def canonical_cwd(self, requested_cwd: str) -> str:
        return os.path.normcase(str(self._canonical_cwd(requested_cwd)))

    def _require_binding(self, binding_id: str) -> RegionBinding:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """SELECT binding.binding_id, binding.kind, binding.canonical_root,
                          binding.root_device, binding.root_inode,
                          COALESCE(main.region_id, binding.region_id),
                          binding.main_binding_id
                   FROM region_bindings AS binding
                   LEFT JOIN region_bindings AS main
                     ON main.binding_id = binding.main_binding_id
                   WHERE binding.binding_id = ?""",
                (binding_id,),
            ).fetchone()
        if row is None:
            raise RegionBindingNotFoundError("project binding not found")
        return self._binding_from_row(row)

    @staticmethod
    def _region_from_row(row: tuple[object, ...]) -> MemoryRegion:
        return MemoryRegion(
            region_id=str(row[0]),
            kind=cast(RegionKind, str(row[1])),
            name=str(row[2]),
            space=str(row[3]),
            session_id=None if row[4] is None else str(row[4]),
        )

    def _binding_from_row(self, row: tuple[object, ...]) -> RegionBinding:
        canonical_root = str(row[2])
        return RegionBinding(
            binding_id=str(row[0]),
            kind=cast(RegionBindingKind, str(row[1])),
            canonical_root=canonical_root,
            region_id=str(row[5]),
            main_binding_id=None if row[6] is None else str(row[6]),
            availability=self._binding_availability(
                Path(canonical_root), cast(int, row[3]), cast(int, row[4])
            ),
        )

    @staticmethod
    def _canonical_root(requested_root: str) -> tuple[str, int, int]:
        if not requested_root.strip():
            raise RegionBindingError("project root must not be blank")
        try:
            root = Path(requested_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RegionBindingError(f"project root cannot be resolved: {error}") from error
        if not root.is_dir():
            raise RegionBindingError("project root must be a directory")
        metadata = root.stat()
        return str(root), metadata.st_dev, metadata.st_ino

    @staticmethod
    def _canonical_cwd(requested_cwd: str) -> Path:
        if not requested_cwd.strip():
            raise RegionBindingError("Codex working directory must not be empty")
        try:
            cwd = Path(requested_cwd).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RegionBindingError(
                f"Codex working directory cannot be resolved: {error}"
            ) from error
        if not cwd.is_dir():
            raise RegionBindingError("Codex working directory must be a directory")
        return cwd

    @staticmethod
    def _binding_availability(root: Path, device: int, inode: int) -> str:
        try:
            metadata = root.stat()
        except OSError:
            return "missing"
        if not stat.S_ISDIR(metadata.st_mode):
            return "missing"
        if metadata.st_dev != device or metadata.st_ino != inode:
            return "replaced"
        return "available"

    @staticmethod
    def _has_unbound_worktree_boundary(cwd: Path, binding_root: Path) -> bool:
        current = cwd
        while current != binding_root and current != current.parent:
            if RegionRegistry._is_linked_git_worktree(current, binding_root):
                return True
            current = current.parent
        return False

    @staticmethod
    def _is_linked_git_worktree(candidate: Path, main_root: Path) -> bool:
        expected_common_dir = RegionRegistry._git_common_dir(main_root)
        if expected_common_dir is None:
            return False
        worktree = RegionRegistry._linked_worktree_metadata(candidate)
        if worktree is None:
            return False
        return worktree[1] == expected_common_dir

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
            return common_dir if RegionRegistry._valid_common_git_dir(common_dir) else None
        if not stat.S_ISREG(metadata.st_mode):
            return None
        worktree = RegionRegistry._linked_worktree_metadata(root)
        return None if worktree is None else worktree[1]

    @staticmethod
    def _linked_worktree_metadata(candidate: Path) -> tuple[Path, Path] | None:
        marker = RegionRegistry._read_small_regular_file(candidate / ".git")
        if marker is None or not marker.startswith("gitdir: "):
            return None
        git_dir = RegionRegistry._resolve_git_path(marker.removeprefix("gitdir: "), candidate)
        if git_dir is None or not RegionRegistry._is_real_directory(git_dir):
            return None
        raw_common_dir = RegionRegistry._read_small_regular_file(git_dir / "commondir")
        raw_backlink = RegionRegistry._read_small_regular_file(git_dir / "gitdir")
        if raw_common_dir is None or raw_backlink is None:
            return None
        common_dir = RegionRegistry._resolve_git_path(raw_common_dir, git_dir)
        backlink = RegionRegistry._resolve_git_path(raw_backlink, git_dir)
        if common_dir is None or backlink is None:
            return None
        try:
            expected_backlink = (candidate / ".git").resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if backlink != expected_backlink or git_dir.parent != common_dir / "worktrees":
            return None
        if not RegionRegistry._valid_common_git_dir(common_dir):
            return None
        if RegionRegistry._read_small_regular_file(git_dir / "HEAD") is None:
            return None
        return git_dir, common_dir

    @staticmethod
    def _resolve_git_path(raw_path: str, base: Path) -> Path | None:
        if not raw_path or "\x00" in raw_path:
            return None
        path = Path(raw_path)
        try:
            return (path if path.is_absolute() else base / path).resolve(strict=True)
        except (OSError, RuntimeError):
            return None

    @staticmethod
    def _is_real_directory(path: Path) -> bool:
        try:
            return stat.S_ISDIR(path.lstat().st_mode)
        except OSError:
            return False

    @staticmethod
    def _valid_common_git_dir(common_dir: Path) -> bool:
        return RegionRegistry._is_real_directory(common_dir) and (
            RegionRegistry._read_small_regular_file(common_dir / "HEAD") is not None
        )

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
            return content.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
