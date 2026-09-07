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

for command in uv git node codex python3; do
  command -v "$command" >/dev/null || {
    printf 'required command is unavailable: %s\n' "$command" >&2
    exit 2
  }
done

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
uv tool install \
  --python "$(command -v python3)" \
  --from "$wheel" \
  personal-agent-memory >/dev/null

codex plugin marketplace add "$repo_root" --json >/dev/null
codex plugin add personal-agent-memory@personal-agent-memory --json >"$work_root/plugin-add.json"
codex plugin list --json >"$work_root/plugins.json"
plugin_root="$(python3 - "$work_root/plugin-add.json" "$work_root/plugins.json" <<'PY'
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
port="$(python3 - <<'PY'
import socket

with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    print(probe.getsockname()[1])
PY
)"
"$XDG_BIN_HOME/personal-agent-memory" serve \
  --state-dir "$state_dir" \
  --library-root "$library_root" \
  --port "$port" >"$work_root/daemon.log" 2>&1 &
daemon_pid=$!

python3 - "$state_dir/api-key" "$port" <<'PY'
import pathlib
import sys
import time
import urllib.error
import urllib.request

key_path = pathlib.Path(sys.argv[1])
port = int(sys.argv[2])
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        key = key_path.read_text(encoding="utf-8").strip()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/health/live",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.status == 200
        break
    except (OSError, urllib.error.URLError):
        time.sleep(0.1)
else:
    raise SystemExit("installed daemon did not become healthy")
PY

export PERSONAL_AGENT_MEMORY_API_KEY_FILE="$state_dir/api-key"
export PERSONAL_AGENT_MEMORY_URL="http://127.0.0.1:$port"
export PLUGIN_DATA="$work_root/plugin-data"
recall_script="$plugin_root/scripts/recall.mjs"
[[ -n "$recall_script" ]]
[[ -f "$recall_script" ]]
python3 - "$library_root" <<'PY' | node "$recall_script" >"$work_root/hook-output.json"
import json
import sys

print(json.dumps({
    "session_id": "native-install-check",
    "cwd": sys.argv[1],
    "hook_event_name": "UserPromptSubmit",
    "prompt": "health check",
}))
PY

test ! -s "$work_root/hook-output.json"
[[ "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" == "$repo_status_before" ]]
[[ "$(git -C "$repo_root" remote -v)" == "$repo_remotes_before" ]]
printf 'native installation acceptance passed\n'
