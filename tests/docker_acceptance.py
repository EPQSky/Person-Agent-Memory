from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


def request(
    url: str,
    key: str | None = None,
    payload: dict[str, object] | None = None,
    method: str | None = None,
    *,
    timeout: float = 10,
) -> tuple[int, dict[str, object]]:
    headers = {} if key is None else {"Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    try:
        request_value = urllib.request.Request(url, headers=headers, data=data, method=method)
        with urllib.request.urlopen(request_value, timeout=timeout) as response:
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

hook_script = Path("/app/plugins/personal-agent-memory/scripts/recall.mjs")
capture_script = Path("/app/plugins/personal-agent-memory/scripts/capture.mjs")
assert hook_script.is_file()
assert capture_script.is_file()
assert subprocess.run(["node", "--version"], capture_output=True, check=False).returncode == 0


def run_recall_hook(
    cwd: Path,
    prompt: str,
    *,
    api_key: str,
    url: str = "http://127.0.0.1:7331",
    timeout_ms: int = 2000,
    event_name: str = "UserPromptSubmit",
) -> subprocess.CompletedProcess[str]:
    event: dict[str, object] = {
        "session_id": "docker-ticket11",
        "transcript_path": "/private/docker-transcript.jsonl",
        "cwd": str(cwd),
        "hook_event_name": event_name,
    }
    if event_name == "UserPromptSubmit":
        event["prompt"] = prompt
    elif event_name == "SessionStart":
        event["source"] = "compact"
    else:
        event["trigger"] = "manual"
    return subprocess.run(
        ["node", str(hook_script)],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        timeout=4,
        env={
            **os.environ,
            "PERSONAL_AGENT_MEMORY_API_KEY": api_key,
            "PLUGIN_DATA": "/tmp/personal-agent-memory-plugin-data",
            "PERSONAL_AGENT_MEMORY_PRECOMPACT_QUERY": prompt,
            "PERSONAL_AGENT_MEMORY_URL": url,
            "PERSONAL_AGENT_MEMORY_TIMEOUT_MS": str(timeout_ms),
        },
        check=False,
    )


def run_capture_hook(
    cwd: Path,
    event_name: str,
    content: str,
    *,
    api_key: str,
    timestamp: str | None = None,
    url: str = "http://127.0.0.1:7331",
    session_id: str = "docker-ticket12",
    turn_id: str | None = None,
) -> subprocess.CompletedProcess[str]:
    event: dict[str, object] = {
        "session_id": session_id,
        "transcript_path": "/private/never-read.jsonl",
        "cwd": str(cwd),
        "hook_event_name": event_name,
        "hidden_reasoning": "never captured",
        "tool_output": "never captured",
        "memory_context": "never captured",
    }
    if timestamp is not None:
        event["timestamp"] = timestamp
    if turn_id is not None:
        event["turn_id"] = turn_id
    if event_name == "UserPromptSubmit":
        event["prompt"] = content
    else:
        event["last_assistant_message"] = content
    return subprocess.run(
        ["node", str(capture_script)],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        timeout=4,
        env={
            **os.environ,
            "PERSONAL_AGENT_MEMORY_API_KEY": api_key,
            "PLUGIN_DATA": "/tmp/personal-agent-memory-plugin-data-ticket12",
            "PERSONAL_AGENT_MEMORY_URL": url,
            "PERSONAL_AGENT_MEMORY_TIMEOUT_MS": "300",
        },
        check=False,
    )

key = Path(os.environ["API_KEY_FILE"]).read_text(encoding="utf-8").strip()
assert request("http://127.0.0.1:7331/health/live", key)[0] == 200

with urllib.request.urlopen("http://127.0.0.1:7331/", timeout=2) as response:
    assert response.status == 200
    login_shell = response.read()
    assert b"Personal Agent Memory" in login_shell
    assert b"Candidate memories" in login_shell
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
mcp_initialize_code, mcp_initialize = request(
    "http://127.0.0.1:7331/mcp",
    key,
    {
        "jsonrpc": "2.0",
        "id": "docker-initialize",
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
    },
)
assert mcp_initialize_code == 200
assert mcp_initialize["result"]["protocolVersion"] == "2025-06-18"
mcp_tools_code, mcp_tools = request(
    "http://127.0.0.1:7331/mcp",
    key,
    {"jsonrpc": "2.0", "id": "docker-tools", "method": "tools/list", "params": {}},
)
assert mcp_tools_code == 200
assert {tool["name"] for tool in mcp_tools["result"]["tools"]} == {
    "memory_search",
    "memory_get",
    "candidate_create",
    "candidate_list",
    "library_list",
    "sync_status",
}

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

Deploy each release to production with a reversible rollback plan.

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
(retrieval_library_path / "graph-seed.md").write_text(
    "# Graph seed\n\nGraphSeedNeedle anchors Alpha.\n\n"
    "Graph: Alpha -> Beta: Alpha points to Beta.\n",
    encoding="utf-8",
)
(retrieval_library_path / "graph-one.md").write_text(
    "# Graph one\n\nEntity: Beta: One-hop authoritative source.\n\n"
    "Graph: Beta -> Gamma: Beta points to Gamma.\n",
    encoding="utf-8",
)
(retrieval_library_path / "graph-two.md").write_text(
    "# Graph two\n\nEntity: Gamma: Two-hop authoritative source.\n",
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

# Build the derivative vector projection, then exercise deterministic hybrid retrieval.
rebuild_code, rebuild = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/vector-index/rebuild",
    key,
    {},
)
assert rebuild_code == 202
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    rebuild_status_code, rebuild_status = request(
        f"http://127.0.0.1:7331/api/v1/jobs/{rebuild['job_id']}", key
    )
    assert rebuild_status_code == 200
    if rebuild_status["status"] in {"done", "error"}:
        break
    time.sleep(0.05)
assert rebuild_status["status"] == "done"

graph_rebuild_code, graph_rebuild = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/graph-index/rebuild",
    key,
    {},
)
assert graph_rebuild_code == 202
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    graph_job_code, graph_job = request(
        f"http://127.0.0.1:7331/api/v1/jobs/{graph_rebuild['job_id']}", key
    )
    assert graph_job_code == 200
    if graph_job["status"] in {"done", "error"}:
        break
    time.sleep(0.05)
assert graph_job["status"] == "done"
graph_status_code, graph_status = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/graph-status",
    key,
)
assert graph_status_code == 200
assert graph_status["status"] == "ready"
assert graph_status["projected_documents"] == graph_status["total_documents"]

# The globally installable Codex plugin executes the real Node Hook over stdin/stdout.
replacement_path = Path(str(replacement_library["canonical_path"]))
(replacement_path / "same-name.md").write_text(
    "# Other service\n\nSameNameSecondNeedle belongs only to the second service.\n",
    encoding="utf-8",
)
assert request(
    f"http://127.0.0.1:7331/api/v1/libraries/{replacement_library['id']}/scan",
    key,
    {},
)[0] == 200


def hook_context(result: subprocess.CompletedProcess[str], event_name: str) -> str:
    assert result.returncode == 0
    assert result.stderr == ""
    output = json.loads(result.stdout)
    hook_output = output["hookSpecificOutput"]
    assert hook_output["hookEventName"] == event_name
    return str(hook_output["additionalContext"])


warm_hook_code, warm_hook_search = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "DockerNeedle42"},
)
assert warm_hook_code == 200
assert warm_hook_search["results"]
first_hook = run_recall_hook(same_name_one / "src", "DockerNeedle42", api_key=key)
first_context = hook_context(first_hook, "UserPromptSubmit")
assert "DockerNeedle42" in first_context
assert "SameNameSecondNeedle" not in first_context
assert str(same_name_one) not in first_context
assert key not in first_context
first_payload = json.loads(first_context.split("\n", 2)[2].rsplit("\n", 1)[0])
assert first_payload["library_id"] == project_library["id"]
assert first_payload["budget"]["effective_tokens"] == 10_000
assert first_payload["budget"]["used_tokens"] <= 10_000
assert len(first_hook.stdout.encode()) <= 96 * 1024 + 1

second_hook = run_recall_hook(same_name_two / "src", "SameNameSecondNeedle", api_key=key)
second_context = hook_context(second_hook, "UserPromptSubmit")
assert "SameNameSecondNeedle" in second_context
assert "DockerNeedle42" not in second_context
second_payload = json.loads(second_context.split("\n", 2)[2].rsplit("\n", 1)[0])
assert second_payload["library_id"] == replacement_library["id"]

unbound_hook = run_recall_hook(unbound_same_name / "src", "DockerNeedle42", api_key=key)
wrong_key_hook = run_recall_hook(same_name_one / "src", "DockerNeedle42", api_key="wrong")
stopped_hook = run_recall_hook(
    same_name_one / "src", "DockerNeedle42", api_key=key, url="http://127.0.0.1:1"
)
for silent_hook in (unbound_hook, wrong_key_hook, stopped_hook):
    assert silent_hook.returncode == 0
    assert silent_hook.stdout == ""
    assert silent_hook.stderr == ""


class SlowHookHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        time.sleep(1)
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


slow_server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHookHandler)
slow_thread = threading.Thread(target=slow_server.serve_forever, daemon=True)
slow_thread.start()
try:
    hook_started = time.monotonic()
    timeout_hook = run_recall_hook(
        same_name_one / "src",
        "DockerNeedle42",
        api_key=key,
        url=f"http://127.0.0.1:{slow_server.server_port}",
        timeout_ms=100,
    )
    assert time.monotonic() - hook_started < 1
    assert timeout_hook.returncode == 0
    assert timeout_hook.stdout == ""
    assert timeout_hook.stderr == ""
finally:
    slow_server.shutdown()
    slow_server.server_close()
    slow_thread.join()

compact_hook = run_recall_hook(
    same_name_one / "src",
    "DockerNeedle42",
    api_key=key,
    event_name="PreCompact",
)
assert compact_hook.returncode == 0
assert json.loads(compact_hook.stdout) == {}
restore_hook = run_recall_hook(
    same_name_one / "src",
    "DockerNeedle42",
    api_key=key,
    event_name="SessionStart",
)
compact_context = hook_context(restore_hook, "SessionStart")
assert "personal-agent-memory-context" in compact_context
assert "DockerNeedle42" in compact_context

# Session capture uses the actual Node adapter out of order with the official turn identifier.
capture_assistant = run_capture_hook(
    same_name_one / "src",
    "Stop",
    "DockerCaptureDecision42 is confirmed as the project decision.",
    api_key=key,
    turn_id="docker-out-of-order-turn",
)
capture_user = run_capture_hook(
    same_name_one / "src",
    "UserPromptSubmit",
    "Remember the durable DockerCaptureDecision42.",
    api_key=key,
    turn_id="docker-out-of-order-turn",
)
capture_duplicate = run_capture_hook(
    same_name_one / "src",
    "Stop",
    "DockerCaptureDecision42 is confirmed as the project decision.",
    api_key=key,
    turn_id="docker-out-of-order-turn",
)
for capture_result in (capture_assistant, capture_user, capture_duplicate):
    assert capture_result.returncode == 0
    assert capture_result.stderr == ""
assert capture_user.stdout == ""
assert json.loads(capture_assistant.stdout) == {}
assert json.loads(capture_duplicate.stdout) == {}

# An interrupted prompt is spooled while the daemon endpoint is unavailable and replayed later.
spooled_capture = run_capture_hook(
    same_name_one / "src",
    "UserPromptSubmit",
    "InterruptedDockerCapture43 remains in the Inbox.",
    api_key=key,
    timestamp="2026-09-05T10:01:00Z",
    url="http://127.0.0.1:1",
)
assert spooled_capture.returncode == 0
assert spooled_capture.stdout == spooled_capture.stderr == ""
capture_spool = Path("/tmp/personal-agent-memory-plugin-data-ticket12/capture/spool")
for malformed_index in range(270):
    (capture_spool / f"event-{malformed_index:064x}.json").write_text("{}", encoding="utf-8")
# Refresh the real record after the synthetic old backlog so the bounded policy retains it.
spooled_capture_retry = run_capture_hook(
    same_name_one / "src",
    "UserPromptSubmit",
    "InterruptedDockerCapture43 remains in the Inbox.",
    api_key=key,
    timestamp="2026-09-05T10:01:00Z",
    url="http://127.0.0.1:1",
)
assert spooled_capture_retry.returncode == 0
bounded_capture = run_capture_hook(
    same_name_one / "src",
    "UserPromptSubmit",
    "BoundedDockerCapture45.",
    api_key=key,
    timestamp="2026-09-05T10:01:30Z",
    url="http://127.0.0.1:1",
)
assert bounded_capture.returncode == 0
assert len(list(capture_spool.glob("event-*.json"))) <= 256
replay_capture = run_capture_hook(
    same_name_one / "src",
    "UserPromptSubmit",
    "ReplayTriggerDockerCapture44.",
    api_key=key,
    timestamp="2026-09-05T10:02:00Z",
)
assert replay_capture.returncode == 0
for _ in range(40):
    if not list(capture_spool.glob("event-*.json")):
        break
    replay_batch = run_capture_hook(
        same_name_one / "src",
        "PreCompact",
        "Continue bounded capture replay.",
        api_key=key,
    )
    assert replay_batch.returncode == 0
    assert json.loads(replay_batch.stdout) == {}
capture_quarantine = Path(
    "/tmp/personal-agent-memory-plugin-data-ticket12/capture/quarantine"
)
assert list(capture_quarantine.glob("event-*.json"))
assert len(list(capture_quarantine.glob("event-*.json"))) <= 256
assert not list(capture_spool.glob("event-*.json"))
capture_deadline = time.monotonic() + 5
captured_candidates: list[dict[str, object]] = []
while time.monotonic() < capture_deadline:
    capture_list_code, capture_list_payload = request(
        "http://127.0.0.1:7331/api/v1/candidates", key
    )
    assert capture_list_code == 200
    captured_candidates = cast(list[dict[str, object]], capture_list_payload)
    if any(
        candidate.get("creator") == "session-capture" for candidate in captured_candidates
    ):
        break
    time.sleep(0.1)
session_candidates = [
    candidate
    for candidate in captured_candidates
    if candidate.get("creator") == "session-capture"
]
assert len(session_candidates) == 1
assert len(session_candidates[0]["source_references"]) == 2
capture_events_code, capture_events = request(
    "http://127.0.0.1:7331/api/v1/capture/events?session_id=docker-ticket12", key
)
assert capture_events_code == 200
contents = {str(event["content"]) for event in capture_events}
assert "InterruptedDockerCapture43 remains in the Inbox." in contents
assert "BoundedDockerCapture45." in contents
serialized_contents = "\n".join(contents)
for excluded in ("never-read.jsonl", "never captured"):
    assert excluded not in serialized_contents
assert all(str(event["occurred_at"]).endswith("Z") for event in capture_events)

# A separate real daemon proves persisted extraction retry survives process restart.
retry_state = Path("/tmp/ticket12-retry-state")
retry_libraries = Path("/tmp/ticket12-retry-libraries")
retry_projects = Path("/tmp/ticket12-retry-projects")
retry_libraries.mkdir()
retry_projects.mkdir()
retry_project = retry_projects / "project"
retry_project.mkdir()
retry_command = [
    "personal-agent-memory",
    "serve",
    "--state-dir",
    str(retry_state),
    "--library-root",
    str(retry_libraries),
    "--library-root",
    str(retry_projects),
    "--host",
    "127.0.0.1",
    "--port",
    "17331",
    "--graph-url",
    "http://127.0.0.1:18080",
    "--graph-model",
    "deterministic-graph",
    "--model-timeout",
    "0.2",
    "--model-retries",
    "0",
]


def start_retry_daemon() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(retry_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("retry daemon exited during startup")
        if (retry_state / "api-key").is_file():
            retry_key = (retry_state / "api-key").read_text().strip()
            try:
                if request("http://127.0.0.1:17331/health/live", retry_key)[0] == 200:
                    return process
            except OSError:
                pass
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError("retry daemon did not become ready")


retry_daemon = start_retry_daemon()
retry_key = (retry_state / "api-key").read_text().strip()
try:
    retry_library_code, retry_library = request(
        "http://127.0.0.1:17331/api/v1/libraries",
        retry_key,
        {"path": str(retry_libraries / "memory"), "kind": "project"},
    )
    assert retry_library_code == 201
    assert request(
        "http://127.0.0.1:17331/api/v1/project-bindings",
        retry_key,
        {"project_root": str(retry_project), "library_id": retry_library["id"]},
    )[0] == 201
    assert request("http://127.0.0.1:18080/control/graph/error", payload={})[0] == 200
    retry_assistant = run_capture_hook(
        retry_project,
        "Stop",
        "RestartRecoveryDecision46 is confirmed.",
        api_key=retry_key,
        url="http://127.0.0.1:17331",
        session_id="docker-restart-recovery",
        turn_id="restart-turn",
    )
    retry_user = run_capture_hook(
        retry_project,
        "UserPromptSubmit",
        "Remember RestartRecoveryDecision46.",
        api_key=retry_key,
        url="http://127.0.0.1:17331",
        session_id="docker-restart-recovery",
        turn_id="restart-turn",
    )
    assert retry_assistant.returncode == retry_user.returncode == 0
    retry_deadline = time.monotonic() + 3
    retry_pending = False
    while time.monotonic() < retry_deadline:
        rounds_code, rounds = request(
            "http://127.0.0.1:17331/api/v1/capture/rounds"
            "?session_id=docker-restart-recovery",
            retry_key,
        )
        assert rounds_code == 200
        if rounds and rounds[0]["status"] == "pending" and rounds[0]["last_error"]:
            retry_pending = True
            break
        time.sleep(0.02)
    assert retry_pending
finally:
    retry_daemon.terminate()
    retry_daemon.wait(timeout=5)

assert request("http://127.0.0.1:18080/control/graph/ok", payload={})[0] == 200
retry_daemon = start_retry_daemon()
try:
    retry_deadline = time.monotonic() + 5
    retry_candidates: list[dict[str, object]] = []
    while time.monotonic() < retry_deadline:
        retry_candidates_code, retry_candidates_payload = request(
            "http://127.0.0.1:17331/api/v1/candidates", retry_key
        )
        assert retry_candidates_code == 200
        retry_candidates = cast(list[dict[str, object]], retry_candidates_payload)
        if retry_candidates:
            break
        time.sleep(0.05)
    assert len(retry_candidates) == 1
    time.sleep(0.7)
    assert len(
        cast(
            list[dict[str, object]],
            request("http://127.0.0.1:17331/api/v1/candidates", retry_key)[1],
        )
    ) == 1
finally:
    retry_daemon.terminate()
    retry_daemon.wait(timeout=5)

one_hop_code, one_hop = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 1},
)
assert one_hop_code == 200
assert [(item["path"], item["classification"]) for item in one_hop["results"]] == [
    ("graph-seed.md", "direct"),
    ("graph-one.md", "graph_expansion"),
]
assert one_hop["results"][1]["graph_hop"] == 1
assert one_hop["results"][1]["content"] == "Entity: Beta: One-hop authoritative source."
assert one_hop["results"][1]["heading"] == "Graph one"
assert one_hop["results"][1]["start_line"] == 3

two_hop_code, two_hop = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 2},
)
assert two_hop_code == 200
assert [item["path"] for item in two_hop["results"]] == [
    "graph-seed.md",
    "graph-one.md",
    "graph-two.md",
]
assert two_hop["results"][2]["graph_hop"] == 2
graph_allocation = two_hop["budget"]["allocation"]
assert any(item["classification"] == "graph_expansion" for item in two_hop["results"])
assert graph_allocation["direct_limit"] >= graph_allocation["available_tokens"] * 0.60 - 1
assert graph_allocation["graph_used"] <= graph_allocation["graph_limit"]
assert graph_allocation["metadata_used"] <= graph_allocation["metadata_limit"]

assert request("http://127.0.0.1:18080/control/graph/error", payload={})[0] == 200
failed_rebuild_code, failed_rebuild = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/graph-index/rebuild",
    key,
    {},
)
assert failed_rebuild_code == 202
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    failed_job_code, failed_job = request(
        f"http://127.0.0.1:7331/api/v1/jobs/{failed_rebuild['job_id']}", key
    )
    assert failed_job_code == 200
    if failed_job["status"] in {"done", "error"}:
        break
    time.sleep(0.05)
assert failed_job["status"] == "error"
graph_fallback_code, graph_fallback = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 2},
)
assert graph_fallback_code == 200
assert [item["path"] for item in graph_fallback["results"]] == ["graph-seed.md"]
assert "graph_unavailable" in graph_fallback["degradation"]
assert request("http://127.0.0.1:18080/control/graph/ok", payload={})[0] == 200
recovered_rebuild_code, recovered_rebuild = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/graph-index/rebuild",
    key,
    {},
)
assert recovered_rebuild_code == 202
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    recovered_job_code, recovered_job = request(
        f"http://127.0.0.1:7331/api/v1/jobs/{recovered_rebuild['job_id']}", key
    )
    assert recovered_job_code == 200
    if recovered_job["status"] in {"done", "error"}:
        break
    time.sleep(0.05)
assert recovered_job["status"] == "done"

# A platform edit invalidates the old source version synchronously. The old
# projection must not leak its graph hit while the replacement builds, and the
# background projection must converge to the edited authoritative Markdown.
graph_one_url = (
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/document?"
    + urllib.parse.urlencode({"path": "graph-one.md"})
)
graph_one_code, graph_one = request(graph_one_url, key)
assert graph_one_code == 200
graph_history_code, graph_history = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/history", key
)
assert graph_history_code == 200
graph_original_commit = str(graph_history[-1]["commit"])
edited_graph_content = (
    "# Graph one\n\nEntity: Beta: Edited one-hop authoritative source.\n\n"
    "Graph: Beta -> Gamma: Edited Beta points to Gamma.\n"
)
graph_edit_code, graph_edit = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/document",
    key,
    {
        "path": "graph-one.md",
        "content": edited_graph_content,
        "expected_source_version": graph_one["source_version"],
        "operation_id": "docker-graph-edit-1",
        "actor_type": "user",
        "source": "docker-graph-acceptance",
    },
    method="PUT",
)
assert graph_edit_code == 200
stale_graph_code, stale_graph = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 1},
)
assert stale_graph_code == 200
assert [item["path"] for item in stale_graph["results"]] == ["graph-seed.md"]
assert stale_graph["graph_index_status"] == "building"
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    _, edited_graph_status = request(
        f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/graph-status",
        key,
    )
    if edited_graph_status["status"] in {"ready", "error"}:
        break
    time.sleep(0.05)
assert edited_graph_status["status"] == "ready"
edited_graph_code, edited_graph = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 1},
)
assert edited_graph_code == 200
assert [item["path"] for item in edited_graph["results"]] == [
    "graph-seed.md",
    "graph-one.md",
]
assert edited_graph["results"][1]["content"] == (
    "Entity: Beta: Edited one-hop authoritative source."
)

# Restoring through the isolated history is another real source-version change.
# It receives the same immediate stale-result filter and eventual convergence.
graph_restore_code, graph_restore = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/history/restore",
    key,
    {
        "path": "graph-one.md",
        "commit": graph_original_commit,
        "expected_source_version": graph_edit["source_version"],
        "operation_id": "docker-graph-restore-1",
        "actor_type": "user",
        "source": "docker-graph-acceptance",
    },
)
assert graph_restore_code == 200
restoring_graph_code, restoring_graph = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 1},
)
assert restoring_graph_code == 200
assert [item["path"] for item in restoring_graph["results"]] == ["graph-seed.md"]
assert restoring_graph["graph_index_status"] == "building"
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    _, restored_graph_status = request(
        f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/graph-status",
        key,
    )
    if restored_graph_status["status"] in {"ready", "error"}:
        break
    time.sleep(0.05)
assert restored_graph_status["status"] == "ready"
restored_graph_code, restored_graph = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 1},
)
assert restored_graph_code == 200
assert [item["path"] for item in restored_graph["results"]] == [
    "graph-seed.md",
    "graph-one.md",
]
assert restored_graph["results"][1]["content"] == (
    "Entity: Beta: One-hop authoritative source."
)

# Removing a real source file and scanning it out immediately prevents the old
# graph projection from returning it, then clears it from the durable projection.
(retrieval_library_path / "graph-one.md").unlink()
graph_delete_scan_code, graph_delete_scan = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/scan",
    key,
    method="POST",
)
assert graph_delete_scan_code == 200
assert graph_delete_scan["removed"] == 1
missing_source_code, missing_source = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 2},
)
assert missing_source_code == 200
assert [item["path"] for item in missing_source["results"]] == ["graph-seed.md"]
assert missing_source["graph_index_status"] == "building"
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    _, deleted_graph_status = request(
        f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/graph-status",
        key,
    )
    if deleted_graph_status["status"] in {"ready", "error"}:
        break
    time.sleep(0.05)
assert deleted_graph_status["status"] == "ready"
deleted_graph_code, deleted_graph = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "GraphSeedNeedle", "graph_hops": 2},
)
assert deleted_graph_code == 200
assert [item["path"] for item in deleted_graph["results"]] == ["graph-seed.md"]

semantic_code, semantic = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "ship a production release"},
)
assert semantic_code == 200
assert semantic["degraded"] is False
assert semantic["vector_index_status"] == "ready"
assert "vector" in semantic["results"][0]["retrieval_sources"]
assert [item["score"] for item in semantic["results"]] == sorted(
    (item["score"] for item in semantic["results"]), reverse=True
)

duplicate_code, duplicate_hybrid = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "DockerNeedle42 production"},
)
assert duplicate_code == 200
assert len({item["chunk_id"] for item in duplicate_hybrid["results"]}) == len(
    duplicate_hybrid["results"]
)

assert request("http://127.0.0.1:18080/control/reranker/malformed", payload={})[0] == 200
rerank_degraded_code, rerank_degraded = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "DockerNeedle42"},
)
assert rerank_degraded_code == 200
assert rerank_degraded["degradation"] == ["reranker_unavailable"]
assert rerank_degraded["results"][0]["path"] == "真实语料.md"
assert request("http://127.0.0.1:18080/control/reranker/ok", payload={})[0] == 200

assert request("http://127.0.0.1:18080/control/embedding/error", payload={})[0] == 200
offline_code, offline = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "FenceNeedle77"},
)
assert offline_code == 200
assert "embedding_unavailable" in offline["degradation"]
assert offline["results"][0]["path"] == "真实语料.md"
assert request("http://127.0.0.1:18080/control/reranker/error", payload={})[0] == 200
fully_offline_code, fully_offline = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "DockerNeedle42"},
)
assert fully_offline_code == 200
assert set(fully_offline["degradation"]) == {
    "embedding_unavailable",
    "reranker_unavailable",
}
assert fully_offline["results"]
assert request("http://127.0.0.1:18080/control/reranker/ok", payload={})[0] == 200

assert request("http://127.0.0.1:18080/control/embedding/timeout", payload={})[0] == 200
timeout_started = time.monotonic()
timeout_code, timeout_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "DockerNeedle42"},
    timeout=2,
)
assert timeout_code == 200
assert time.monotonic() - timeout_started < 1.5
assert "embedding_unavailable" in timeout_search["degradation"]
assert timeout_search["results"]
assert request("http://127.0.0.1:18080/control/embedding/ok", payload={})[0] == 200

# The public REST/MCP contract budgets the complete encoded context package.
budget_note = retrieval_library_path / "budget.md"
budget_note.write_text(
    "# Budget\n\nBudgetNeedle " + ("多字节 memory content. " * 4_000) + "\n",
    encoding="utf-8",
)
budget_scan_code, budget_scan = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{project_library['id']}/scan",
    key,
    method="POST",
)
assert budget_scan_code == 200
assert budget_scan["changed"] == 1

low_code, low_budget = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {
        "cwd": str(same_name_one / "src"),
        "query": "BudgetNeedle",
        "token_budget": 4_000,
    },
)
assert low_code == 200
assert low_budget["budget"]["effective_tokens"] == 4_000
assert low_budget["budget"]["used_tokens"] <= 4_000
assert low_budget["results"][0]["truncated"] is True
assert low_budget["results"][0]["content"].count("```") % 2 == 0

exact_code, exact_budget = request(
    "http://127.0.0.1:7331/api/v1/search",
    key,
    {
        "library_id": project_library["id"],
        "query": "BudgetNeedle",
        "token_budget": 10_000,
    },
)
assert exact_code == 200
assert exact_budget["budget"]["requested_tokens"] == 10_000
assert exact_budget["budget"]["effective_tokens"] == 10_000
assert exact_budget["budget"]["used_tokens"] <= 10_000
assert exact_budget["results"][0]["truncated"] is True
assert "多字节" in exact_budget["results"][0]["content"]

hard_cap_code, hard_cap = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {
        "cwd": str(same_name_one / "src"),
        "query": "BudgetNeedle",
        "token_budget": 20_000,
    },
)
assert hard_cap_code == 200
assert hard_cap["budget"]["effective_tokens"] == 10_000
assert hard_cap["budget"]["used_tokens"] <= 10_000

fallback_code, fallback_budget = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {
        "cwd": str(same_name_one / "src"),
        "query": "BudgetNeedle",
        "target_model": "unsupported-tokenizer-model",
    },
)
assert fallback_code == 200
assert fallback_budget["budget"]["tokenizer"] == "conservative_utf8_bytes"
assert "tokenizer_fallback" in fallback_budget["degradation"]
fallback_json = json.dumps(fallback_budget, ensure_ascii=False, separators=(",", ":"))
assert len(fallback_json.encode("utf-8")) <= 10_000

explicit_user_root = memory_root / "explicit-user"
explicit_user_root.mkdir()
(explicit_user_root / "private.md").write_text(
    "# Private\n\nExplicitUserNeedle belongs only to the user library.\n",
    encoding="utf-8",
)
explicit_user_code, explicit_user = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(explicit_user_root), "kind": "user"},
)
assert explicit_user_code == 201
default_user_code, default_user = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(same_name_one / "src"), "query": "ExplicitUserNeedle"},
)
assert default_user_code == 200
assert default_user["results"] == []
explicit_user_search_code, explicit_user_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"library_id": explicit_user["id"], "query": "ExplicitUserNeedle"},
)
assert explicit_user_search_code == 200
assert explicit_user_search["results"][0]["path"] == "private.md"

unbound_search_code, unbound_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"cwd": str(unbound_same_name / "src"), "query": "DockerNeedle42"},
)
assert unbound_search_code == 200
assert unbound_search["status"] == "unbound"
assert unbound_search["results"] == []
assert unbound_search["scope"]["kind"] == "project_cwd"

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
    restart_candidate_code, restart_candidate = request(
        "http://127.0.0.1:17331/mcp/candidates",
        restart_key,
        {
            "library_id": restart_library["id"],
            "suggested_type": "constraint",
            "body": "# Restart candidate\n\nCandidateRestartPersistence\n",
            "source_references": ["docker-session:restart"],
            "creator": "docker-mcp",
            "idempotency_key": "docker-restart-candidate",
        },
    )
    assert restart_candidate_code == 201
finally:
    restart_process.send_signal(signal.SIGINT)
    assert restart_process.wait(timeout=10) == 0

restart_process = start_restart_daemon()
try:
    restored_code, restored = request("http://127.0.0.1:17331/api/v1/libraries", restart_key)
    assert restored_code == 200
    assert restored[0]["id"] == restart_library["id"]
    assert restored[0]["kind"] == "project"
    restored_candidate_code, restored_candidate = request(
        f"http://127.0.0.1:17331/mcp/candidates/{restart_candidate['id']}", restart_key
    )
    assert restored_candidate_code == 200
    assert restored_candidate == restart_candidate
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

# MCP can submit and inspect candidates, while only Web REST governance can
# edit, approve, reject, and create authoritative Markdown.
candidate_payload = {
    "library_id": history_id,
    "suggested_type": "decision",
    "body": "# Candidate\n\nDockerCandidateOriginal\n",
    "source_references": ["docker-session:governance#assistant-final"],
    "creator": "docker-mcp",
    "idempotency_key": "docker-candidate-create-1",
}
candidate_code, candidate = request(
    "http://127.0.0.1:7331/mcp/candidates", key, candidate_payload
)
duplicate_candidate_code, duplicate_candidate = request(
    "http://127.0.0.1:7331/mcp/candidates", key, candidate_payload
)
assert candidate_code == duplicate_candidate_code == 201
assert duplicate_candidate == candidate
assert candidate["status"] == "pending"
candidate_list_code, candidate_list = request(
    f"http://127.0.0.1:7331/mcp/candidates?library_id={history_id}", key
)
assert candidate_list_code == 200
assert candidate_list[0]["id"] == candidate["id"]
pending_search_code, pending_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"library_id": history_id, "query": "DockerCandidateOriginal"},
)
assert pending_search_code == 200
assert pending_search["results"] == []

candidate_edit_code, candidate_edited = request(
    f"http://127.0.0.1:7331/api/v1/candidates/{candidate['id']}",
    key,
    {
        "body": "# Approved candidate\n\nDockerCandidatePublished\n",
        "operator": "docker-user",
        "reason": "Corrected before approval",
    },
    method="PUT",
)
assert candidate_edit_code == 200
assert candidate_edited["body"].endswith("DockerCandidatePublished\n")
candidate_history_before = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history", key
)[1]
approval_payload = {
    "operator": "docker-user",
    "reason": "Verified in browser acceptance",
    "operation_id": "docker-candidate-approve-1",
}
approval_code, approved_candidate = request(
    f"http://127.0.0.1:7331/api/v1/candidates/{candidate['id']}/approve",
    key,
    approval_payload,
)
approval_retry_code, approval_retry = request(
    f"http://127.0.0.1:7331/api/v1/candidates/{candidate['id']}/approve",
    key,
    approval_payload,
)
assert approval_code == approval_retry_code == 200
assert approval_retry == approved_candidate
published_path = history_library / str(approved_candidate["published_path"])
published_content = published_path.read_text(encoding="utf-8")
assert "DockerCandidatePublished" in published_content
assert "docker-session:governance#assistant-final" in published_content
approved_search_code, approved_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"library_id": history_id, "query": "DockerCandidatePublished"},
)
assert approved_search_code == 200
assert approved_search["results"][0]["path"] == approved_candidate["published_path"]
candidate_history = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history", key
)[1]
assert len(candidate_history) == len(candidate_history_before) + 1
assert candidate_history[0]["commit"] == approved_candidate["commit"]
candidate_diff = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history/"
    f"{approved_candidate['commit']}/diff",
    key,
)[1]["diff"]
assert "DockerCandidatePublished" in candidate_diff

rejected_code, rejected_candidate = request(
    "http://127.0.0.1:7331/mcp/candidates",
    key,
    {
        **candidate_payload,
        "body": "# Rejected\n\nDockerCandidateRejected\n",
        "idempotency_key": "docker-candidate-reject-1",
    },
)
assert rejected_code == 201
rejection_history_before = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history", key
)[1]
rejection_payload = {
    "operator": "docker-user",
    "reason": "Unverified guess",
    "operation_id": "docker-candidate-rejection-1",
}
rejection_code, rejection = request(
    f"http://127.0.0.1:7331/api/v1/candidates/{rejected_candidate['id']}/reject",
    key,
    rejection_payload,
)
rejection_retry_code, rejection_retry = request(
    f"http://127.0.0.1:7331/api/v1/candidates/{rejected_candidate['id']}/reject",
    key,
    rejection_payload,
)
assert rejection_code == rejection_retry_code == 200
assert rejection_retry == rejection
assert rejection["published_path"] is None
assert request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history", key
)[1] == rejection_history_before
rejected_search = request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"library_id": history_id, "query": "DockerCandidateRejected"},
)[1]
assert rejected_search["results"] == []
assert project_git_fingerprint() == project_before

browser_approve_code, browser_approve_candidate = request(
    "http://127.0.0.1:7331/mcp/candidates",
    key,
    {
        **candidate_payload,
        "body": "# Browser approval\n\nBrowserCandidateOriginal\n",
        "idempotency_key": "docker-browser-approve",
    },
)
browser_reject_code, browser_reject_candidate = request(
    "http://127.0.0.1:7331/mcp/candidates",
    key,
    {
        **candidate_payload,
        "body": "# Browser rejection\n\nBrowserCandidateRejected\n",
        "idempotency_key": "docker-browser-reject",
    },
)
assert browser_approve_code == browser_reject_code == 201
browser_history_before = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history", key
)[1]

options = Options()
options.binary_location = "/usr/bin/chromium"
for argument in (
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--window-size=1280,1000",
):
    options.add_argument(argument)
browser = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=options)
wait = WebDriverWait(browser, 15)
try:
    browser.get("http://127.0.0.1:7331/")
    browser.find_element(By.ID, "key").send_keys(key)
    browser.find_element(By.ID, "connect").click()
    wait.until(lambda driver: driver.find_elements(By.ID, "candidate-title"))

    approve_id = str(browser_approve_candidate["id"])
    approve_button = wait.until(
        lambda driver: driver.find_element(
            By.XPATH,
            f"//nav[contains(@class, 'candidate-list')]/button[contains(., '{approve_id}')]",
        )
    )
    approve_button.click()
    body_input = browser.find_element(By.ID, "candidate-body")
    body_input.clear()
    body_input.send_keys("# Browser approved\n\nBrowserCandidatePublished\n")
    operator_input = browser.find_element(By.ID, "candidate-operator")
    reason_input = browser.find_element(By.ID, "candidate-reason")
    operator_input.send_keys("browser-user")
    reason_input.send_keys("Edited through the Web page")
    browser.find_element(By.ID, "candidate-save").click()
    wait.until(
        lambda _: request(
            f"http://127.0.0.1:7331/api/v1/candidates/{approve_id}", key
        )[1]["body"].endswith("BrowserCandidatePublished\n")
    )
    operator_input = browser.find_element(By.ID, "candidate-operator")
    reason_input = browser.find_element(By.ID, "candidate-reason")
    operator_input.send_keys("browser-user")
    reason_input.send_keys("Approved through the Web page")
    browser.find_element(By.ID, "candidate-approve").click()
    wait.until(
        lambda _: request(
            f"http://127.0.0.1:7331/api/v1/candidates/{approve_id}", key
        )[1]["status"]
        == "approved"
    )

    reject_id = str(browser_reject_candidate["id"])
    reject_button = wait.until(
        lambda driver: driver.find_element(
            By.XPATH,
            f"//nav[contains(@class, 'candidate-list')]/button[contains(., '{reject_id}')]",
        )
    )
    reject_button.click()
    browser.find_element(By.ID, "candidate-operator").send_keys("browser-user")
    browser.find_element(By.ID, "candidate-reason").send_keys(
        "Rejected through the Web page"
    )
    browser.find_element(By.ID, "candidate-reject").click()
    wait.until(
        lambda _: request(
            f"http://127.0.0.1:7331/api/v1/candidates/{reject_id}", key
        )[1]["status"]
        == "rejected"
    )
finally:
    browser.quit()

browser_approved = request(
    f"http://127.0.0.1:7331/api/v1/candidates/{browser_approve_candidate['id']}", key
)[1]
browser_rejected = request(
    f"http://127.0.0.1:7331/api/v1/candidates/{browser_reject_candidate['id']}", key
)[1]
assert browser_approved["operator"] == "browser-user"
assert browser_approved["reason"] == "Approved through the Web page"
assert browser_rejected["operator"] == "browser-user"
assert browser_rejected["reason"] == "Rejected through the Web page"
assert browser_rejected["published_path"] is None
browser_published = history_library / str(browser_approved["published_path"])
assert "BrowserCandidatePublished" in browser_published.read_text(encoding="utf-8")
browser_history = request(
    f"http://127.0.0.1:7331/api/v1/libraries/{history_id}/history", key
)[1]
assert len(browser_history) == len(browser_history_before) + 1
assert browser_history[0]["commit"] == browser_approved["commit"]
assert request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"library_id": history_id, "query": "BrowserCandidatePublished"},
)[1]["results"][0]["path"] == browser_approved["published_path"]
assert request(
    "http://127.0.0.1:7331/mcp/search",
    key,
    {"library_id": history_id, "query": "BrowserCandidateRejected"},
)[1]["results"] == []
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
