from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def request(
    url: str,
    key: str | None = None,
    payload: dict[str, object] | None = None,
    method: str | None = None,
) -> tuple[int, dict[str, object]]:
    headers = {} if key is None else {"Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    try:
        request_value = urllib.request.Request(url, headers=headers, data=data, method=method)
        with urllib.request.urlopen(request_value, timeout=2) as response:
            body = response.read()
            return response.status, {} if not body else json.loads(body)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def wait_for(url: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if request(url)[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"service did not become ready: {url}")


wait_for("http://127.0.0.1:18080/health")

key = Path(os.environ["API_KEY_FILE"]).read_text(encoding="utf-8").strip()
assert request("http://127.0.0.1:7331/health/live", key)[0] == 200

with urllib.request.urlopen("http://127.0.0.1:7331/", timeout=2) as response:
    assert response.status == 200
    login_shell = response.read()
    assert b"Personal Agent Memory" in login_shell
    assert key.encode() not in login_shell
    fingerprint = f"sha256:{hashlib.sha256(key.encode()).hexdigest()[:12]}".encode()
    assert fingerprint not in login_shell
    assert b"localStorage" not in login_shell
    assert b"URLSearchParams" not in login_shell

assert request("http://127.0.0.1:7331/api/v1/status")[0] == 401
status_code, status = request("http://127.0.0.1:7331/api/v1/status", key)
assert status_code == 200
assert status["database"] == "ok"
assert status["background_worker"] == "running"
assert request("http://127.0.0.1:7331/mcp/health", key)[0] == 200

memory_root = Path("/memory-libraries")
existing = memory_root / "existing"
existing.mkdir(parents=True, exist_ok=True)
existing_note = existing / "adopted.md"
existing_text = "# Adopted\n\nExisting Markdown stays byte-for-byte.\n"
existing_note.write_text(existing_text, encoding="utf-8")

created_code, created = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(memory_root / "created"), "kind": "user"},
)
assert created_code == 201
assert Path(str(created["canonical_path"])).is_dir()
adopted_code, adopted = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(existing), "kind": "project"},
)
assert adopted_code == 201
assert existing_note.read_text(encoding="utf-8") == existing_text
duplicate_code, duplicate = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(existing / ".." / "existing"), "kind": "user"},
)
assert duplicate_code == 409
assert duplicate["detail"]["library_id"] == adopted["id"]
outside_code, outside = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": "/tmp/outside-library", "kind": "project"},
)
assert outside_code == 422
assert "allowed roots" in outside["detail"]
unwritable = memory_root / "unwritable"
unwritable.mkdir(exist_ok=True)
unwritable.chmod(stat.S_IRUSR | stat.S_IXUSR)
try:
    denied_code, denied = request(
        "http://127.0.0.1:7331/api/v1/libraries",
        key,
        {"path": str(unwritable), "kind": "project"},
    )
    assert denied_code == 422
    assert "writable" in denied["detail"]
finally:
    unwritable.chmod(stat.S_IRWXU)

listed_code, listed = request("http://127.0.0.1:7331/mcp/libraries", key)
assert listed_code == 200
assert {item["id"] for item in listed} == {created["id"], adopted["id"]}
assert key not in json.dumps(listed)

outside_replacement = Path("/tmp/outside-replacement")
outside_replacement.mkdir(exist_ok=True)
created_path = Path(str(created["canonical_path"]))
created_path.rmdir()
created_path.symlink_to(existing, target_is_directory=True)
replacement_code, replacement_list = request("http://127.0.0.1:7331/api/v1/libraries", key)
assert replacement_code == 200
replacement = next(item for item in replacement_list if item["id"] == created["id"])
assert replacement["availability"] == "unavailable"
adopted_health = next(item for item in replacement_list if item["id"] == adopted["id"])
assert adopted_health["availability"] == "available"
replacement_sync_code, replacement_sync = request(
    f"http://127.0.0.1:7331/mcp/libraries/{created['id']}/sync-status", key
)
assert replacement_sync_code == 200
assert replacement_sync["availability"] == "unavailable"
created_path.unlink()
created_path.symlink_to(outside_replacement, target_is_directory=True)
outside_replacement_code, outside_replacement_list = request(
    "http://127.0.0.1:7331/api/v1/libraries", key
)
assert outside_replacement_code == 200
outside_health = next(item for item in outside_replacement_list if item["id"] == created["id"])
assert outside_health["availability"] == "unavailable"

source_parent = memory_root / "source-parent"
source_path = source_parent / "project"
target_parent = memory_root / "target-parent"
target_path = target_parent / "project"
ancestor_source_code, ancestor_source = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(source_path), "kind": "project"},
)
ancestor_target_code, ancestor_target = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(target_path), "kind": "project"},
)
assert ancestor_source_code == 201
assert ancestor_target_code == 201
source_path.rmdir()
source_parent.rmdir()
source_parent.symlink_to(target_parent, target_is_directory=True)
ancestor_code, ancestor_list = request("http://127.0.0.1:7331/api/v1/libraries", key)
assert ancestor_code == 200
ancestor_by_id = {item["id"]: item for item in ancestor_list}
assert ancestor_by_id[ancestor_source["id"]]["canonical_path"] == str(source_path)
assert ancestor_by_id[ancestor_source["id"]]["availability"] == "unavailable"
assert ancestor_by_id[ancestor_target["id"]]["canonical_path"] == str(target_path)
assert ancestor_by_id[ancestor_target["id"]]["availability"] == "available"
ancestor_sync_code, ancestor_sync = request(
    f"http://127.0.0.1:7331/mcp/libraries/{ancestor_source['id']}/sync-status", key
)
assert ancestor_sync_code == 200
assert ancestor_sync["availability"] == "unavailable"

# Project binding and Codex cwd resolution use explicit canonical roots only.
retrieval_library_path = memory_root / "binding-project"
retrieval_library_path.mkdir(parents=True, exist_ok=True)
retrieval_note = retrieval_library_path / "真实语料.md"
retrieval_note.write_text(
    """# Architecture 决策

English and 中文 share DockerNeedle42 in the current project memory.

## Repeated evidence

The same supporting paragraph appears twice.

The same supporting paragraph appears twice.

## Long paragraph

"""
    + "Long source context " * 80
    + """

## Complete code

```python
def docker_fixture() -> str:
    return "FenceNeedle77"
```
""",
    encoding="utf-8",
)
(retrieval_library_path / "ignored.md").write_text("IgnoredNeedle", encoding="utf-8")
(retrieval_library_path / "node_modules").mkdir()
(retrieval_library_path / "node_modules" / "hidden.md").write_text(
    "DependencyNeedle", encoding="utf-8"
)
(Path("/tmp") / "outside-retrieval.md").write_text("OutsideNeedle", encoding="utf-8")
(retrieval_library_path / "escape.md").symlink_to(Path("/tmp") / "outside-retrieval.md")
project_library_code, project_library = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(memory_root / "binding-project"), "kind": "project"},
)
replacement_library_code, replacement_library = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(memory_root / "binding-replacement"), "kind": "project"},
)
assert project_library_code == replacement_library_code == 201
docker_projects = Path("/project-roots")
shutil.rmtree(docker_projects, ignore_errors=True)
same_name_one = docker_projects / "one" / "service"
same_name_two = docker_projects / "two" / "service"
nested_project = same_name_one / "packages" / "nested"
unbound_same_name = docker_projects / "unbound" / "service"
worktree = docker_projects / "worktrees" / "feature"
embedded_worktree = same_name_one / ".worktrees" / "feature"
for path in (
    same_name_one / "src",
    same_name_two / "src",
    nested_project / "src",
    unbound_same_name / "src",
    worktree / "src",
    embedded_worktree / "src",
):
    path.mkdir(parents=True)
main_git = same_name_one / ".git"
worktree_git_dir = main_git / "worktrees" / "feature"
worktree_git_dir.mkdir(parents=True)
(main_git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
(embedded_worktree / ".git").write_text(
    f"gitdir: {os.path.relpath(worktree_git_dir, embedded_worktree)}\n",
    encoding="utf-8",
)
(worktree_git_dir / "commondir").write_text("../..\n", encoding="utf-8")
(worktree_git_dir / "gitdir").write_text(
    f"{os.path.relpath(embedded_worktree / '.git', worktree_git_dir)}\n",
    encoding="utf-8",
)
(worktree_git_dir / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")
fake_marker_root = same_name_one / "ordinary-marker"
(fake_marker_root / "src").mkdir(parents=True)
(fake_marker_root / ".git").write_text(
    "gitdir: /untrusted/content/is/not-read\n", encoding="utf-8"
)

user_binding_code, user_binding = request(
    "http://127.0.0.1:7331/api/v1/project-bindings",
    key,
    {"project_root": str(same_name_one), "library_id": created["id"]},
)
assert user_binding_code == 422
assert "project memory library" in user_binding["detail"]
empty_bindings_code, empty_bindings = request("http://127.0.0.1:7331/api/v1/project-bindings", key)
assert empty_bindings_code == 200
assert empty_bindings == []


def bind(root: Path, library_id: object) -> dict[str, object]:
    code, binding = request(
        "http://127.0.0.1:7331/api/v1/project-bindings",
        key,
        {"project_root": str(root), "library_id": library_id},
    )
    assert code == 201
    return binding


main_binding = bind(same_name_one, project_library["id"])
other_binding = bind(same_name_two, replacement_library["id"])
nested_binding = bind(nested_project, replacement_library["id"])

resolve_url = "http://127.0.0.1:7331/api/v1/project-bindings/resolve?cwd="
main_code, main_resolution = request(
    resolve_url + urllib.parse.quote(str(same_name_one / "src")), key
)
other_code, other_resolution = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(same_name_two / "src")},
)
nested_code, nested_resolution = request(
    resolve_url + urllib.parse.quote(str(nested_project / "src")), key
)
unbound_code, unbound_resolution = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(unbound_same_name / "src")},
)
worktree_before_code, worktree_before = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(worktree / "src")},
)
assert main_code == other_code == nested_code == unbound_code == worktree_before_code == 200
assert main_resolution["project_id"] == main_binding["id"]
assert main_resolution["library_id"] == project_library["id"]
assert other_resolution["project_id"] == other_binding["id"]
assert other_resolution["library_id"] == replacement_library["id"]
assert nested_resolution["project_id"] == nested_binding["id"]
assert nested_resolution["library_id"] == replacement_library["id"]
assert unbound_resolution["status"] == "unbound"
assert worktree_before["status"] == "unbound"
embedded_before_code, embedded_before = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(embedded_worktree / "src")},
)
assert embedded_before_code == 200
assert embedded_before["status"] == "unbound"
fake_marker_code, fake_marker_resolution = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(fake_marker_root / "src")},
)
assert fake_marker_code == 200
assert fake_marker_resolution["project_id"] == main_binding["id"]

worktree_code, worktree_binding = request(
    "http://127.0.0.1:7331/api/v1/project-bindings/worktrees",
    key,
    {
        "worktree_root": str(worktree),
        "main_project_binding_id": main_binding["id"],
    },
)
assert worktree_code == 201
worktree_after_code, worktree_after = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(worktree / "src")},
)
assert worktree_after_code == 200
assert worktree_after["project_id"] == main_binding["id"]
assert worktree_after["binding_id"] == worktree_binding["id"]
assert worktree_after["library_id"] == project_library["id"]

search_code, search = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "DockerNeedle42"},
)
assert search_code == 200
assert search["status"] == "bound"
assert search["library_id"] == project_library["id"]
assert len(search["results"]) == 1
search_hit = search["results"][0]
assert search_hit["path"] == "真实语料.md"
assert search_hit["heading"] == "Architecture 决策"
assert search_hit["start_line"] == 3
assert search_hit["source_type"] == "markdown"
assert search_hit["classification"] == "direct"
assert len(search_hit["document_id"]) == 36
assert len(search_hit["chunk_id"]) == 36
assert len(search_hit["source_version"]) == 64
chinese_code, chinese_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "中文"},
)
assert chinese_code == 200
assert chinese_search["results"][0]["path"] == "真实语料.md"

code_search_code, code_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "FenceNeedle77"},
)
assert code_search_code == 200
assert code_search["results"][0]["content"].count("```") == 2
ignore_code, ignore_result = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/ignore-rules",
    key,
    {"patterns": ["ignored.md"]},
    method="PUT",
)
assert ignore_code == 200
assert ignore_result["patterns"] == ["ignored.md"]
for excluded_query in ("IgnoredNeedle", "DependencyNeedle", "OutsideNeedle"):
    excluded_code, excluded = request(
        "http://127.0.0.1:7331/mcp/search",
        key,
        {"cwd": str(same_name_one / "src"), "query": excluded_query},
    )
    assert excluded_code == 200
    assert excluded["results"] == []

retrieval_note.write_text("# Updated\n\nIncrementalNeedle99\n", encoding="utf-8")
rescan_code, rescan = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/scan",
    key,
    method="POST",
)
assert rescan_code == 200
assert rescan["changed"] == 1
updated_code, updated_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "IncrementalNeedle99"},
)
assert updated_code == 200
assert len(updated_search["results"]) == 1
retrieval_note.unlink()
delete_rescan_code, delete_rescan = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/scan",
    key,
    method="POST",
)
assert delete_rescan_code == 200
assert delete_rescan["removed"] == 1
deleted_code, deleted_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "IncrementalNeedle99"},
)
assert deleted_code == 200
assert deleted_search["results"] == []

embedded_code, embedded_binding = request(
    "http://127.0.0.1:7331/api/v1/project-bindings/worktrees",
    key,
    {
        "worktree_root": str(embedded_worktree),
        "main_project_binding_id": main_binding["id"],
    },
)
assert embedded_code == 201
embedded_after_code, embedded_after = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(embedded_worktree / "src")},
)
assert embedded_after_code == 200
assert embedded_after["binding_id"] == embedded_binding["id"]
assert embedded_after["project_id"] == main_binding["id"]
assert embedded_after["library_id"] == project_library["id"]

update_code, updated_binding = request(
    f"http://127.0.0.1:7331/api/v1/project-bindings/{main_binding['id']}",
    key,
    {"library_id": replacement_library["id"]},
    method="PUT",
)
assert update_code == 200
assert updated_binding["library_id"] == replacement_library["id"]
inherited_update_code, inherited_update = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(worktree / "src")},
)
assert inherited_update_code == 200
assert inherited_update["library_id"] == replacement_library["id"]

delete_code, _ = request(
    f"http://127.0.0.1:7331/api/v1/project-bindings/{nested_binding['id']}",
    key,
    method="DELETE",
)
assert delete_code == 204
after_delete_code, after_delete = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(nested_project / "src")},
)
assert after_delete_code == 200
assert after_delete["project_id"] == main_binding["id"]

same_name_one.rename(docker_projects / "moved-service")
same_name_one.mkdir(parents=True)
diagnostic_code, diagnostics = request("http://127.0.0.1:7331/api/v1/project-bindings", key)
assert diagnostic_code == 200
diagnostic = next(item for item in diagnostics if item["id"] == main_binding["id"])
assert diagnostic["availability"] == "replaced"
replacement_resolution_code, replacement_resolution = request(
    "http://127.0.0.1:7331/mcp/project-bindings/resolve",
    key,
    {"cwd": str(same_name_one)},
)
assert replacement_resolution_code == 200
assert replacement_resolution["status"] == "unbound"

# Exercise persistence across two real daemon processes inside the isolated acceptance container.
restart_state = Path("/tmp/pam-restart-state")
shutil.rmtree(restart_state, ignore_errors=True)
restart_path = memory_root / "restart-check"
restart_command = [
    "personal-agent-memory",
    "serve",
    "--state-dir",
    str(restart_state),
    "--library-root",
    str(memory_root),
    "--port",
    "17331",
]


def start_restart_daemon() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        restart_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (restart_state / "api-key").exists():
            restart_key = (restart_state / "api-key").read_text(encoding="utf-8").strip()
            try:
                if request("http://127.0.0.1:17331/api/v1/status", restart_key)[0] == 200:
                    return process
            except OSError:
                pass
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError("restart acceptance daemon did not become ready")


restart_process = start_restart_daemon()
restart_key = (restart_state / "api-key").read_text(encoding="utf-8").strip()
try:
    restart_code, restart_library = request(
        "http://127.0.0.1:17331/api/v1/libraries",
        restart_key,
        {"path": str(restart_path), "kind": "project"},
    )
    assert restart_code == 201
finally:
    restart_process.send_signal(signal.SIGINT)
    assert restart_process.wait(timeout=10) == 0

restart_process = start_restart_daemon()
try:
    restored_code, restored = request("http://127.0.0.1:17331/api/v1/libraries", restart_key)
    assert restored_code == 200
    assert restored[0]["id"] == restart_library["id"]
    assert restored[0]["kind"] == "project"
finally:
    restart_process.send_signal(signal.SIGINT)
    assert restart_process.wait(timeout=10) == 0

# Run a separate daemon as nobody so ancestor permission failures are effective
# even though the Compose acceptance container itself starts as root.
permission_state = Path("/tmp/pam-permission-state")
permission_root = Path("/tmp/pam-permission-libraries")
permission_parent = permission_root / "protected"
permission_library = permission_parent / "project"
for path in (permission_state, permission_library):
    path.mkdir(parents=True, exist_ok=True)
for path in (permission_state, permission_root, permission_parent, permission_library):
    os.chown(path, 65534, 65534)
    path.chmod(0o700)


def drop_to_nobody() -> None:
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)


permission_process = subprocess.Popen(
    [
        "personal-agent-memory",
        "serve",
        "--state-dir",
        str(permission_state),
        "--library-root",
        str(permission_root),
        "--port",
        "17332",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    preexec_fn=drop_to_nobody,
)
try:
    deadline = time.monotonic() + 10
    permission_key = ""
    while time.monotonic() < deadline:
        if (permission_state / "api-key").exists():
            permission_key = (permission_state / "api-key").read_text(encoding="utf-8").strip()
            try:
                if request("http://127.0.0.1:17332/api/v1/status", permission_key)[0] == 200:
                    break
            except OSError:
                pass
        time.sleep(0.1)
    else:
        raise RuntimeError("non-root permission daemon did not become ready")
    permission_code, permission_registered = request(
        "http://127.0.0.1:17332/api/v1/libraries",
        permission_key,
        {"path": str(permission_library), "kind": "project"},
    )
    assert permission_code == 201
    permission_parent.chmod(0o000)
    inaccessible_code, inaccessible = request(
        "http://127.0.0.1:17332/api/v1/libraries", permission_key
    )
    assert inaccessible_code == 200
    assert inaccessible[0]["id"] == permission_registered["id"]
    assert inaccessible[0]["availability"] == "unavailable"
    permission_parent.chmod(0o700)
    recovered_code, recovered = request("http://127.0.0.1:17332/api/v1/libraries", permission_key)
    assert recovered_code == 200
    assert recovered[0]["id"] == permission_registered["id"]
    assert recovered[0]["availability"] == "available"
finally:
    permission_parent.chmod(0o700)
    permission_process.send_signal(signal.SIGINT)
    assert permission_process.wait(timeout=10) == 0

# Authoritative edits use isolated Git history even when a memory directory is
# physically inside a real project repository.
history_project = Path("/project-roots/history-project")
history_library = history_project / "memory"
history_library.mkdir(parents=True)
(history_project / ".gitignore").write_text("/memory/\n", encoding="utf-8")
history_document = history_library / "decisions.md"
history_original = "# Docker decision\n\nUse DockerHistoryOld.\n"
history_updated = (
    "# Docker decision\n\nUse DockerHistoryNew.\n\n```python\nprint('history')\n```\n\n"
    + "Long editable paragraph. " * 400
)
history_document.write_text(history_original, encoding="utf-8")
subprocess.run(["git", "init", "-b", "main", str(history_project)], check=True)
subprocess.run(["git", "-C", str(history_project), "add", ".gitignore"], check=True)
subprocess.run(
    [
        "git",
        "-C",
        str(history_project),
        "-c",
        "user.name=Docker Owner",
        "-c",
        "user.email=owner@docker.invalid",
        "commit",
        "-m",
        "project baseline",
    ],
    check=True,
)


def project_git_fingerprint() -> tuple[str, str, str, str]:
    def git_output(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(history_project), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    return (
        git_output("rev-parse", "HEAD"),
        git_output("branch", "--show-current"),
        hashlib.sha256((history_project / ".git" / "index").read_bytes()).hexdigest(),
        git_output("status", "--porcelain=v2"),
    )


project_before = project_git_fingerprint()
history_register_code, history_registered = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(history_library), "kind": "project"},
)
assert history_register_code == 201
history_id = str(history_registered["id"])
documents_code, documents = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/documents", key
)
assert documents_code == 200
assert documents[0]["path"] == "decisions.md"
document_url = (
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/document?"
    + urllib.parse.urlencode({"path": "decisions.md"})
)
loaded_code, loaded = request(document_url, key)
assert loaded_code == 200
preview_code, preview = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/document/preview",
    key,
    {
        "path": "decisions.md",
        "content": history_updated,
        "expected_source_version": loaded["source_version"],
    },
)
assert preview_code == 200
assert "+Use DockerHistoryNew." in preview["diff"]
assert "language-python" in preview["rendered_html"]
edit_code, edited = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/document",
    key,
    {
        "path": "decisions.md",
        "content": history_updated,
        "expected_source_version": loaded["source_version"],
        "operation_id": "docker-edit-1",
        "actor_type": "user",
        "source": "docker-web",
    },
    method="PUT",
)
assert edit_code == 200
assert history_document.read_text(encoding="utf-8") == history_updated
retry_code, retry = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/document",
    key,
    {
        "path": "decisions.md",
        "content": history_updated,
        "expected_source_version": loaded["source_version"],
        "operation_id": "docker-edit-1",
        "actor_type": "user",
        "source": "docker-web",
    },
    method="PUT",
)
assert retry_code == 200
assert retry == edited
history_code, commits = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history", key
)
assert history_code == 200
assert len(commits) == 2
assert commits[0]["author_name"] == "Personal Agent Memory"
assert commits[0]["author_email"] == "memory-platform@localhost"
assert commits[0]["actor_type"] == "user"
diff_code, commit_diff = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history/{edited['commit']}/diff",
    key,
)
assert diff_code == 200
assert {line["kind"] for line in commit_diff["lines"]} >= {"addition", "deletion"}
restore_code, restored = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history/restore",
    key,
    {
        "path": "decisions.md",
        "commit": commits[1]["commit"],
        "expected_source_version": edited["source_version"],
        "operation_id": "docker-restore-1",
        "actor_type": "user",
        "source": "docker-history",
    },
)
assert restore_code == 200
assert restored["commit"] != edited["commit"]
assert history_document.read_text(encoding="utf-8") == history_original
assert request(
    f"http://127.0.0.1:7331/mcp/libraries/{history_id}/document", key
)[0] == 404
assert project_git_fingerprint() == project_before

assert request("http://127.0.0.1:18080/v1/models")[1]["data"]
assert request(
    "http://127.0.0.1:18080/v1/chat/completions",
    payload={"messages": [{"role": "user", "content": "fixture"}]},
)[1]["choices"]
assert request("http://127.0.0.1:18080/v1/embeddings", payload={"input": ["fixture"]})[1]["data"]
assert request(
    "http://127.0.0.1:18080/v1/rerank",
    payload={"query": "fixture", "documents": ["a", "longer"]},
)[1]["results"]
