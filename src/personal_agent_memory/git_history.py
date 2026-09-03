from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

PLATFORM_GIT_NAME = "Personal Agent Memory"
PLATFORM_GIT_EMAIL = "memory-platform@localhost"
MANIFEST_PATH = ".personal-agent-memory.json"
TOMBSTONE_DIRECTORY = ".personal-agent-memory-tombstones"


class GitHistoryError(RuntimeError):
    """Raised when isolated memory history cannot be maintained safely."""


@dataclass(frozen=True, slots=True)
class GitInitialization:
    commit: str
    previous_head: str | None
    manifest_created: bool
    repository_created: bool
    index_existed: bool
    index_content: bytes | None
    manifest_content: bytes


@dataclass(frozen=True, slots=True)
class GitRepository:
    git_dir: Path
    work_tree: Path

    def initialize(
        self,
        library_id: str,
        documents: dict[str, bytes],
        verify: Callable[[], None] | None = None,
    ) -> GitInitialization:
        expected = f'{{"format":1,"library_id":"{library_id}"}}\n'.encode()
        repository_created = not self.git_dir.exists()
        index_existed = False
        index_content: bytes | None = None
        manifest_created = False
        head: str | None = None
        snapshot_ready = False
        try:
            if repository_created:
                self.git_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._run(
                    "init",
                    "--bare",
                    "--initial-branch=main",
                    str(self.git_dir),
                    bare=False,
                )
            self._validate_repository()
            bare = self._is_bare()
            if not bare and self._run("status", "--porcelain").stdout:
                raise GitHistoryError("authorized dedicated memory repository must be clean")
            tracked = self._tracked_paths()
            invalid = sorted(path for path in tracked if not allowed_history_path(path))
            head = self.head()
            if head is not None:
                historical = set(
                    self._run("log", "--format=", "--name-only", "HEAD").stdout.splitlines()
                )
                invalid.extend(
                    sorted(path for path in historical if path and not allowed_history_path(path))
                )
            if invalid:
                raise GitHistoryError(
                    "dedicated memory repository tracks unsupported paths: "
                    + ", ".join(sorted(set(invalid))[:3])
                )
            index_existed, index_content = self._snapshot_real_index()
            snapshot_ready = True
            if not bare:
                manifest_created = _ensure_manifest(self.work_tree, expected)
            contents: dict[str, bytes | None] = dict(documents)
            contents[MANIFEST_PATH] = expected
            paths = tuple(sorted(contents))
            commit = head
            if head is None or any(path not in tracked for path in paths):
                commit = self.commit(contents, "Initialize memory library history")
            if verify is not None:
                verify()
            assert commit is not None
            return GitInitialization(
                commit,
                head,
                manifest_created,
                repository_created,
                index_existed,
                index_content,
                expected,
            )
        except BaseException as error:
            compensation_errors: list[str] = []
            if not repository_created and snapshot_ready and head != self.head():
                try:
                    self._restore_head(head)
                except BaseException as compensation_error:
                    compensation_errors.append(f"Git: {compensation_error}")
            if manifest_created:
                try:
                    _remove_manifest(self.work_tree, expected)
                except BaseException as compensation_error:
                    compensation_errors.append(f"manifest: {compensation_error}")
            if not repository_created and snapshot_ready:
                try:
                    self._restore_real_index(index_existed, index_content)
                except BaseException as compensation_error:
                    compensation_errors.append(f"index: {compensation_error}")
            if repository_created:
                try:
                    shutil.rmtree(self.git_dir)
                except BaseException as compensation_error:
                    compensation_errors.append(f"repository: {compensation_error}")
            if compensation_errors:
                raise GitHistoryError(
                    f"{error}; initialization compensation failed: "
                    f"{'; '.join(compensation_errors)}"
                ) from error
            raise

    def rollback_initialization(self, initialization: GitInitialization) -> None:
        errors: list[str] = []
        if initialization.repository_created:
            try:
                shutil.rmtree(self.git_dir)
            except BaseException as error:
                errors.append(f"repository: {error}")
            if errors:
                raise GitHistoryError(
                    "memory history initialization rollback failed: " + "; ".join(errors)
                )
            return
        if self.head() != initialization.previous_head:
            try:
                self._restore_head(initialization.previous_head)
            except BaseException as error:
                errors.append(f"Git: {error}")
        if initialization.manifest_created:
            try:
                _remove_manifest(self.work_tree, initialization.manifest_content)
            except BaseException as error:
                errors.append(f"manifest: {error}")
        try:
            self._restore_real_index(
                initialization.index_existed, initialization.index_content
            )
        except BaseException as error:
            errors.append(f"index: {error}")
        if errors:
            raise GitHistoryError(
                "memory history initialization rollback failed: " + "; ".join(errors)
            )

    def _restore_head(self, previous_head: str | None) -> None:
        reference = self._run("symbolic-ref", "HEAD").stdout.strip()
        current = self.head()
        if previous_head is None:
            if current is not None:
                self._run("update-ref", "-d", reference, current)
            if not self._is_bare():
                self._run("read-tree", "--empty")
            return
        if current is None:
            self._run("update-ref", reference, previous_head, "0" * 40)
        elif current != previous_head:
            self._run("update-ref", reference, previous_head, current)
        if not self._is_bare():
            self._run("read-tree", previous_head)

    def _snapshot_real_index(self) -> tuple[bool, bytes | None]:
        if self._is_bare():
            return False, None
        index = self.git_dir / "index"
        try:
            descriptor = os.open(index, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return False, None
        with os.fdopen(descriptor, "rb") as stream:
            return True, stream.read()

    def _restore_real_index(self, existed: bool, content: bytes | None) -> None:
        if self._is_bare():
            return
        index = self.git_dir / "index"
        if not existed:
            with suppress(FileNotFoundError):
                os.unlink(index)
            return
        assert content is not None
        descriptor, temporary = tempfile.mkstemp(
            prefix="pam-index-rollback-", dir=self.git_dir
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, index)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def head(self) -> str | None:
        result = self._run("rev-parse", "--verify", "HEAD", check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def commit(
        self,
        contents: dict[str, bytes | None],
        message: str,
    ) -> str:
        paths = tuple(sorted(contents))
        if not paths or any(not allowed_history_path(path) for path in paths):
            raise GitHistoryError("history commit contains an unsupported path")
        head = self.head()
        descriptor, index_path = tempfile.mkstemp(prefix="pam-index-")
        os.close(descriptor)
        os.unlink(index_path)
        try:
            env = {"GIT_INDEX_FILE": index_path}
            if head is not None:
                self._run("read-tree", head, extra_env=env)
            for path in paths:
                content = contents[path]
                if content is None:
                    self._run("update-index", "--remove", "--", path, extra_env=env)
                    continue
                blob = self._run(
                    "hash-object",
                    "--no-filters",
                    "-w",
                    "--stdin",
                    extra_env=env,
                    input_bytes=content,
                ).stdout.strip()
                self._run(
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    "100644",
                    blob,
                    path,
                    extra_env=env,
                )
            tree = self._run("write-tree", extra_env=env).stdout.strip()
            arguments = ["commit-tree", tree, "-m", message]
            if head is not None:
                arguments.extend(("-p", head))
            commit = self._run(*arguments, extra_env=env).stdout.strip()
        finally:
            with suppress(FileNotFoundError):
                os.unlink(index_path)
        reference = self._run("symbolic-ref", "HEAD").stdout.strip()
        update = ["update-ref", reference, commit]
        update.append(head or "0" * 40)
        self._run(*update)
        try:
            if not self._is_bare():
                self._run("read-tree", "HEAD")
        except BaseException:
            self.rollback_commit(commit, head)
            raise
        return commit

    def rollback_commit(self, commit: str, previous_head: str | None) -> None:
        reference = self._run("symbolic-ref", "HEAD").stdout.strip()
        if previous_head is None:
            self._run("update-ref", "-d", reference, commit)
        else:
            self._run("update-ref", reference, previous_head, commit)
        if not self._is_bare():
            if previous_head is None:
                self._run("read-tree", "--empty")
            else:
                self._run("read-tree", previous_head)

    def ensure_index_clean(self) -> None:
        if not self._is_bare():
            result = self._run("diff", "--cached", "--quiet", check=False)
            if result.returncode != 0:
                raise GitHistoryError(
                    "authorized dedicated memory repository has staged changes"
                )

    def history(self, limit: int = 50) -> list[dict[str, str]]:
        if self.head() is None:
            return []
        output = self._run(
            "log", f"--max-count={max(1, min(limit, 200))}",
            "--format=%H%x00%aI%x00%an%x00%ae%x00%s",
        ).stdout
        history = []
        for line in output.splitlines():
            commit, authored_at, author_name, author_email, subject = line.split("\0", 4)
            history.append(
                {
                    "commit": commit,
                    "authored_at": authored_at,
                    "author_name": author_name,
                    "author_email": author_email,
                    "subject": subject,
                }
            )
        return history

    def diff(self, commit: str) -> str:
        resolved = self.resolve_commit(commit)
        return self._run(
            "show", "--format=", "--no-ext-diff", "--unified=3", resolved,
        ).stdout

    def content_at(self, commit: str, path: str) -> str:
        if not allowed_markdown_path(path):
            raise GitHistoryError("only authoritative Markdown can be restored")
        resolved = self.resolve_commit(commit)
        result = self._run("show", f"{resolved}:{path}", check=False)
        if result.returncode != 0:
            raise GitHistoryError("document does not exist in the selected commit")
        return result.stdout

    def resolve_commit(self, commit: str) -> str:
        if not commit or any(character not in "0123456789abcdefABCDEF" for character in commit):
            raise GitHistoryError("invalid commit identifier")
        result = self._run("rev-parse", "--verify", f"{commit}^{{commit}}", check=False)
        if result.returncode != 0:
            raise GitHistoryError("history commit not found")
        resolved = result.stdout.strip()
        ancestor = self._run("merge-base", "--is-ancestor", resolved, "HEAD", check=False)
        if ancestor.returncode != 0:
            raise GitHistoryError("history commit is not reachable from this library")
        return resolved

    def _tracked_paths(self) -> set[str]:
        if self.head() is None:
            return set()
        return set(self._run("ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines())

    def _validate_repository(self) -> None:
        result = self._run("rev-parse", "--is-bare-repository", check=False)
        if result.returncode != 0:
            raise GitHistoryError("Git history path is not a repository")

    def _is_bare(self) -> bool:
        return self._run("config", "--bool", "core.bare").stdout.strip() == "true"

    def _run(
        self,
        *arguments: str,
        bare: bool = True,
        check: bool = True,
        extra_env: dict[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = ["git"]
        if bare:
            command.extend((f"--git-dir={self.git_dir}", f"--work-tree={self.work_tree}"))
        command.extend(arguments)
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("GIT_")
        }
        environment.update(
            {
                "GIT_AUTHOR_NAME": PLATFORM_GIT_NAME,
                "GIT_AUTHOR_EMAIL": PLATFORM_GIT_EMAIL,
                "GIT_COMMITTER_NAME": PLATFORM_GIT_NAME,
                "GIT_COMMITTER_EMAIL": PLATFORM_GIT_EMAIL,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        if extra_env:
            environment.update(extra_env)
        result = subprocess.run(
            command,
            cwd=self.work_tree,
            env=environment,
            check=False,
            capture_output=True,
            input=input_bytes,
            timeout=30,
        )
        stdout = result.stdout.decode(errors="replace")
        stderr = result.stderr.decode(errors="replace")
        if check and result.returncode != 0:
            detail = stderr.strip() or stdout.strip() or "Git command failed"
            raise GitHistoryError(detail)
        return subprocess.CompletedProcess(command, result.returncode, stdout, stderr)


def allowed_markdown_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        bool(path)
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and path.lower().endswith(".md")
    )


def allowed_history_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        allowed_markdown_path(path)
        or path == MANIFEST_PATH
        or (
            bool(path)
            and not candidate.is_absolute()
            and ".." not in candidate.parts
            and bool(candidate.parts)
            and candidate.parts[0] == TOMBSTONE_DIRECTORY
        )
    )


def _ensure_manifest(root: Path, content: bytes) -> bool:
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f".{MANIFEST_PATH}.{os.getpid()}.tmp"
    try:
        try:
            existing = os.open(MANIFEST_PATH, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        except FileNotFoundError:
            existing = -1
        if existing >= 0:
            try:
                with os.fdopen(existing, "rb") as stream:
                    if stream.read() != content:
                        raise GitHistoryError(
                            "portable library manifest belongs to another library"
                        )
            finally:
                existing = -1
            return False
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, MANIFEST_PATH, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
        return True
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory)
        os.close(directory)


def _remove_manifest(root: Path, expected: bytes) -> None:
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(MANIFEST_PATH, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as stream:
            if stream.read() != expected:
                raise GitHistoryError(
                    "portable library manifest changed while initialization was rolling back"
                )
        os.unlink(MANIFEST_PATH, dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)
