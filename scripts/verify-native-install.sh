#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_root="$(mktemp -d "${TMPDIR:-/tmp}/pam-native-acceptance.XXXXXX")"
daemon_pid=""
repo_status_before="$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)"
repo_remotes_before="$(git -C "$repo_root" remote -v)"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$daemon_pid" ]] && kill -0 "$daemon_pid" 2>/dev/null; then
    kill -TERM "$daemon_pid" 2>/dev/null || true
    wait "$daemon_pid" 2>/dev/null || true
  fi
  rm -rf "$work_root"
  exit "$status"
}
trap cleanup EXIT INT TERM

for command in uv git node codex; do
  command -v "$command" >/dev/null || {
    printf 'required command is unavailable: %s\n' "$command" >&2
    exit 2
  }
done
requested_python="${PAM_VERIFY_PYTHON:-python3}"
python_command="$(command -v "$requested_python" 2>/dev/null || true)"
[[ -n "$python_command" && -x "$python_command" ]] || {
  printf 'required Python interpreter is unavailable: %s\n' "$requested_python" >&2
  exit 2
}
"$python_command" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)' || {
  printf 'unsupported Python interpreter: %s\n' "$python_command" >&2
  exit 2
}

export HOME="$work_root/home"
export CODEX_HOME="$HOME/.codex"
export XDG_CACHE_HOME="$work_root/cache"
export XDG_CONFIG_HOME="$work_root/config"
export XDG_DATA_HOME="$work_root/data"
export XDG_BIN_HOME="$work_root/bin"
export UV_CACHE_DIR="$work_root/uv-cache"
mkdir -p "$HOME" "$CODEX_HOME" "$XDG_BIN_HOME"

uv build "$repo_root" --out-dir "$work_root/dist" >/dev/null
wheel="$(find "$work_root/dist" -maxdepth 1 -name '*.whl' -print -quit)"
[[ -n "$wheel" ]]
uv export --frozen --no-dev --no-hashes --no-emit-project \
  --project "$repo_root" --output-file "$work_root/constraints.txt" >/dev/null
uv tool install \
  --python "$python_command" \
  --from "$wheel" \
  --constraints "$work_root/constraints.txt" \
  --no-python-downloads \
  personal-agent-memory >/dev/null
tool_python="$(uv tool dir)/personal-agent-memory/bin/python"
"$tool_python" -c \
  'from importlib.metadata import version; assert version("JiuwenMemory") == "0.2.0"; assert version("milvus-lite") == "3.2.2"; assert version("pymilvus") == "3.0.2"'

codex plugin marketplace add "$repo_root" --json >/dev/null
codex plugin add personal-agent-memory@personal-agent-memory --json >"$work_root/plugin-add.json"
codex plugin list --json >"$work_root/plugins.json"
plugin_root="$($python_command - "$work_root/plugin-add.json" "$work_root/plugins.json" <<'PY'
import json
import sys
from pathlib import Path

installed = json.load(open(sys.argv[1], encoding="utf-8"))
plugins = json.load(open(sys.argv[2], encoding="utf-8"))
assert "personal-agent-memory" in json.dumps(plugins)
path = Path(installed["installedPath"])
assert path.is_dir()
print(path)
PY
)"

state_dir="$work_root/state"
library_root="$work_root/libraries"
mkdir -p "$library_root"
port="$($python_command - <<'PY'
import socket

with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    print(probe.getsockname()[1])
PY
)"
"$XDG_BIN_HOME/personal-agent-memory" serve \
  --state-dir "$state_dir" \
  --library-root "$library_root" \
  --port "$port" \
  --memory-kernel mem2 >"$work_root/daemon.log" 2>&1 &
daemon_pid=$!

project_root="$work_root/project"
mkdir -p "$project_root"
"$python_command" - "$state_dir/api-key" "$port" "$state_dir" "$project_root" <<'PY'
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

key_path = Path(sys.argv[1])
port = int(sys.argv[2])
state_dir = Path(sys.argv[3])
project_root = Path(sys.argv[4])
base_url = f"http://127.0.0.1:{port}"

def request(path, *, method="GET", payload=None):
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise OSError("API key is empty")
    data = None if payload is None else json.dumps(payload).encode()
    request_headers = {"Authorization": f"Bearer {key}"}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers=request_headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)

deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        assert request("/health/live") == {"status": "ok"}
        break
    except (OSError, urllib.error.URLError):
        time.sleep(0.1)
else:
    raise SystemExit("installed daemon did not become healthy")

health = request("/mcp/health")
assert health["memory_kernel"] == "mem2"
assert health["memory_kernel_status"] in {"ready", "degraded"}
assert request("/api/v1/regions/tree")["root"]["region_id"] == "user"
assert request("/api/v1/project-bindings") == []

region = request(
    "/api/v1/regions",
    method="POST",
    payload={"kind": "project", "name": "Native install", "region_id": "native-install"},
)
request(
    "/api/v1/project-bindings",
    method="POST",
    payload={"project_root": str(project_root), "region_id": region["region_id"]},
)
assert request(
    "/mcp/project-bindings/resolve",
    method="POST",
    payload={"cwd": str(project_root)},
)["status"] == "bound"
request(
    "/api/v1/memories",
    method="POST",
    payload={
        "region_id": region["region_id"],
        "content": "NATIVE-INSTALL-MEMORY is available.",
        "memory_type": "domain_fact",
        "source_references": ["native-install:acceptance"],
        "actor": "native-install",
        "idempotency_key": "native-install-memory",
    },
)
rpc = request(
    "/mcp",
    method="POST",
    payload={
        "jsonrpc": "2.0",
        "id": "native-install",
        "method": "tools/call",
        "params": {
            "name": "memory_search",
            "arguments": {
                "region_ids": [region["region_id"]],
                "query": "NATIVE-INSTALL-MEMORY",
            },
        },
    },
)
assert rpc["result"]["isError"] is False
assert rpc["result"]["structuredContent"]["results"]
project_search = request(
    "/api/v1/search",
    method="POST",
    payload={
        "cwd": str(project_root),
        "query": "NATIVE-INSTALL-MEMORY",
        "token_budget": 10_000,
    },
)
assert project_search["results"]
assert (state_dir / "mem2" / "VERSION").read_text(encoding="utf-8") == (
    "jiuwen-memory-0.2.0\n"
)
PY

export PERSONAL_AGENT_MEMORY_API_KEY_FILE="$state_dir/api-key"
export PERSONAL_AGENT_MEMORY_URL="http://127.0.0.1:$port"
export PLUGIN_DATA="$work_root/plugin-data"
recall_script="$plugin_root/scripts/recall.mjs"
[[ -n "$recall_script" ]]
[[ -f "$recall_script" ]]
"$python_command" - "$project_root" <<'PY' | node "$recall_script" >"$work_root/hook-output.json"
import json
import sys

print(json.dumps({
    "session_id": "native-install-check",
    "cwd": sys.argv[1],
    "hook_event_name": "UserPromptSubmit",
    "prompt": "NATIVE-INSTALL-MEMORY",
}))
PY

grep -q 'NATIVE-INSTALL-MEMORY' "$work_root/hook-output.json"
"$python_command" - "$project_root" <<'PY' | node "$plugin_root/scripts/capture.mjs"
import json
import sys

print(json.dumps({
    "session_id": "native-install-capture",
    "cwd": sys.argv[1],
    "hook_event_name": "UserPromptSubmit",
    "prompt": "remember the native installation acceptance",
}))
PY
[[ -z "$(find "$PLUGIN_DATA/spool" -maxdepth 1 -name 'event-*.json' -print -quit 2>/dev/null)" ]]
[[ "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" == "$repo_status_before" ]]
[[ "$(git -C "$repo_root" remote -v)" == "$repo_remotes_before" ]]
printf 'native installation acceptance passed\n'
