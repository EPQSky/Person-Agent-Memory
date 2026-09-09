#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service_name="personal-agent-memory.service"
config_home="${XDG_CONFIG_HOME:-$HOME/.config}"
install_config_dir="$config_home/personal-agent-memory"
install_config_path="$install_config_dir/install.json"
unit_path="$config_home/systemd/user/$service_name"
codex_home="${CODEX_HOME:-$HOME/.codex}"
check_only=false
installer_arguments=()

if [[ "${PAM_INSTALL_ACCEPTANCE_CUSTOM_DIRS:-}" == "1" ]]; then
  state_dir="$HOME/installer acceptance/platform state"
  library_root="$HOME/installer acceptance/memory libraries"
  installer_arguments=(--state-dir "$state_dir" --library-root "$library_root")
else
  state_dir="$HOME/.local/share/personal-agent-memory"
  library_root="$HOME/memory-libraries"
fi

[[ "${1:-}" == "--check" ]] && check_only=true

skip() {
  printf 'SKIP: %s\n' "$1" >&2
  exit 77
}

[[ "${PAM_INSTALL_ACCEPTANCE_DEDICATED_USER:-}" == "1" ]] ||
  skip "set PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 for a disposable dedicated user"
[[ "$(uname -s)" == "Linux" ]] || skip "Linux is required"
[[ "$(uname -m)" == "x86_64" ]] || skip "x86_64 is required"
python_command=""
for candidate in python3 python3.13 python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
    'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)' \
    >/dev/null 2>&1; then
    python_command="$(command -v "$candidate")"
    break
  fi
done
[[ -n "$python_command" ]] || skip "Python 3.11, 3.12, or 3.13 is required"
. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" =~ ^(22\.04|24\.04)$ ]] ||
  skip "Ubuntu 22.04 or 24.04 is required for this release acceptance entry"
for command in uv git node codex systemctl; do
  command -v "$command" >/dev/null || skip "required command is unavailable: $command"
done
target_version="$("$python_command" - "$repo_root/pyproject.toml" <<'PY'
from pathlib import Path
import sys
import tomllib

with Path(sys.argv[1]).open("rb") as stream:
    print(tomllib.load(stream)["project"]["version"])
PY
)"
old_version="0.0.0"
legacy_commit="9383eba03a8ec8901ec51c29a467265cec86c433"
[[ "$target_version" != "$old_version" ]] || skip "release version must be newer than $old_version"
account_home="$(getent passwd "$(id -un)" | cut -d: -f6)"
[[ "$HOME" == "$account_home" ]] || skip "HOME must be the dedicated account home: $account_home"
systemctl --user show-environment >/dev/null 2>&1 || skip "a working systemd user bus is required"
[[ ! -e "$state_dir" ]] || skip "state directory already exists: $state_dir"
[[ ! -e "$library_root" ]] || skip "memory library root already exists: $library_root"
[[ ! -e "$install_config_path" ]] || skip "install metadata already exists: $install_config_path"
[[ ! -e "$unit_path" ]] || skip "user service already exists: $unit_path"
[[ ! -e "$codex_home" ]] || skip "Codex configuration already exists: $codex_home"
command -v personal-agent-memory >/dev/null 2>&1 &&
  skip "Personal Agent Memory command already exists for this user"

if $check_only; then
  printf 'real Linux installer acceptance prerequisites satisfied\n'
  exit 0
fi

work_root="$(mktemp -d "${TMPDIR:-/tmp}/pam-linux-installer.XXXXXX")"
repo_status_before="$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)"
repo_remotes_before="$(git -C "$repo_root" remote -v)"
old_release="$work_root/old-release"
mkdir -p "$old_release"
git -C "$repo_root" archive "$legacy_commit" | tar -C "$old_release" -xf -
"$python_command" - "$old_release" "$old_version" <<'PY'
import json
from pathlib import Path
import sys
import tomllib

root = Path(sys.argv[1])
old_version = sys.argv[2]
with (root / "pyproject.toml").open("rb") as stream:
    release_version = tomllib.load(stream)["project"]["version"]
replacements = {
    root / "pyproject.toml": (
        f'version = "{release_version}"',
        f'version = "{old_version}"',
    ),
    root / "src/personal_agent_memory/__init__.py": (
        f'__version__ = "{release_version}"',
        f'__version__ = "{old_version}"',
    ),
    root / "plugins/personal-agent-memory/.codex-plugin/plugin.json": (
        f'"version": "{release_version}"',
        f'"version": "{old_version}"',
    ),
}
for path, (current, old) in replacements.items():
    content = path.read_text(encoding="utf-8")
    if content.count(current) != 1:
        raise SystemExit(f"could not prepare old release version in {path}")
    path.write_text(content.replace(current, old), encoding="utf-8")
PY

cleanup() {
  status=$?
  trap - EXIT INT TERM
  systemctl --user disable --now "$service_name" >/dev/null 2>&1 || true
  rm -f "$unit_path"
  systemctl --user daemon-reload >/dev/null 2>&1 || true
  codex plugin remove personal-agent-memory@personal-agent-memory --json >/dev/null 2>&1 || true
  codex plugin marketplace remove personal-agent-memory --json >/dev/null 2>&1 || true
  uv tool uninstall personal-agent-memory >/dev/null 2>&1 || true
  rm -rf "$state_dir" "$library_root" "$install_config_dir" "$codex_home" "$work_root"
  rmdir "$HOME/installer acceptance" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

"$old_release/install.sh" "${installer_arguments[@]}" >"$work_root/install-old.out"
systemctl --user is-enabled "$service_name" | grep -qx enabled
systemctl --user is-active "$service_name" | grep -qx active
uv tool list | grep -Eq "^personal-agent-memory v${old_version//./\\.}$"
codex plugin list --json >"$work_root/plugins-old.json"
"$python_command" - "$work_root/plugins-old.json" "$old_version" <<'PY'
import json
from pathlib import Path
import sys

installed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["installed"]
matches = [
    item
    for item in installed
    if item.get("pluginId") == "personal-agent-memory@personal-agent-memory"
]
assert len(matches) == 1
assert matches[0]["version"] == sys.argv[2]
PY

key_file="$state_dir/api-key"
"$python_command" - "$install_config_path" "$unit_path" "$state_dir" "$library_root" "$key_file" \
  <<'PY'
import json
from pathlib import Path
import stat
import sys

config_path, unit_path, state_dir, library_root, key_file = map(Path, sys.argv[1:6])
metadata = json.loads(config_path.read_text(encoding="utf-8"))
assert metadata == {
    "schema_version": 1,
    "state_dir": str(state_dir),
    "library_root": str(library_root),
    "library_root_ownership": "user-content-never-delete",
}
assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
assert key_file.read_text(encoding="utf-8").strip() not in config_path.read_text(
    encoding="utf-8"
)
unit = unit_path.read_text(encoding="utf-8")
assert f'--state-dir "{state_dir}"' in unit
assert f'--library-root "{library_root}"' in unit
PY

systemctl --user restart "$service_name"
"$python_command" - "$key_file" <<'PY'
import pathlib
import sys
import time
import urllib.error
import urllib.request

key = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip()
request = urllib.request.Request(
    "http://127.0.0.1:7331/health/live",
    headers={"Authorization": f"Bearer {key}"},
)
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(request, timeout=1) as response:
            if response.status == 200:
                break
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(0.1)
else:
    raise SystemExit("daemon did not become healthy after service restart")
PY

project_root="$work_root/project"
memory_path="$library_root/installer-acceptance"
mkdir -p "$project_root" "$memory_path"
printf '# Installer acceptance\n\nInstaller acceptance fact.\n' >"$memory_path/memory.md"

"$python_command" - "$key_file" "$memory_path" "$project_root" <<'PY'
import json
import pathlib
import sys
import urllib.request

key = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip()
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def post(path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:7331{path}",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)

library = post("/api/v1/libraries", {"path": sys.argv[2], "kind": "project"})
post("/api/v1/project-bindings", {"project_root": sys.argv[3], "library_id": library["id"]})
post("/mcp/candidates", {
    "library_id": library["id"],
    "suggested_type": "decision",
    "body": "# Upgrade candidate\\n\\nPreserve this pending candidate.\\n",
    "source_references": ["codex-session:upgrade#assistant-final"],
    "creator": "codex",
    "idempotency_key": "linux-installer-upgrade-candidate",
})
post("/api/v1/capture/events", {
    "event_id": "linux-installer-upgrade-event",
    "session_id": "linux-installer-upgrade-session",
    "turn_id": "linux-installer-upgrade-turn",
    "event_kind": "user",
    "content": "Preserve this captured event during upgrade.",
    "occurred_at": "2026-09-08T00:00:00Z",
    "cwd": sys.argv[3],
})
PY

cp "$state_dir/api-key" "$work_root/api-key.before"
cp "$memory_path/memory.md" "$work_root/memory.before"
"$python_command" - "$state_dir/platform.sqlite3" "$work_root/state.before.json" <<'PY'
import json
import sqlite3
from pathlib import Path
import sys

database, output = map(Path, sys.argv[1:])
tables = (
    "memory_libraries",
    "project_bindings",
    "candidate_memories",
    "candidate_audit",
    "capture_inbox",
)
with sqlite3.connect(database) as connection:
    snapshot = {
        table: connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
        for table in tables
    }
for required in (
    "memory_libraries",
    "project_bindings",
    "candidate_memories",
    "candidate_audit",
    "capture_inbox",
):
    assert snapshot[required], f"upgrade fixture did not create {required} data"
output.write_text(json.dumps(snapshot, sort_keys=True), encoding="utf-8")
PY

"$repo_root/install.sh" --adopt-marketplace >"$work_root/install-upgrade.out"
"$repo_root/install.sh" >"$work_root/install-reinstall.out"
systemctl --user is-enabled "$service_name" | grep -qx enabled
systemctl --user is-active "$service_name" | grep -qx active
[[ "$(personal-agent-memory --version)" == "$target_version" ]]
cmp "$work_root/api-key.before" "$state_dir/api-key"
cmp "$work_root/memory.before" "$memory_path/memory.md"
"$python_command" - "$install_config_path" "$state_dir" "$library_root" "$target_version" \
  "$repo_root" "$old_release" <<'PY'
import json
from pathlib import Path
import sys

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert metadata == {
    "schema_version": 1,
    "state_dir": sys.argv[2],
    "library_root": sys.argv[3],
    "library_root_ownership": "user-content-never-delete",
    "installed_version": sys.argv[4],
    "marketplace_ownership": "preexisting",
    "marketplace_source": sys.argv[5],
    "marketplace_previous_source": sys.argv[6],
    "marketplace_update_pending": False,
}
PY

codex plugin list --json >"$work_root/plugins.json"
plugin_root="$("$python_command" - "$work_root/plugins.json" "$target_version" <<'PY'
import json
from pathlib import Path
import sys

plugins = json.load(open(sys.argv[1], encoding="utf-8"))["installed"]
matches = [item for item in plugins if item["pluginId"] == "personal-agent-memory@personal-agent-memory"]
assert len(matches) == 1
assert matches[0]["version"] == sys.argv[2]
source = matches[0].get("source")
assert isinstance(source, dict)
installed_path = source.get("path")
assert isinstance(installed_path, str)
assert Path(installed_path).is_dir()
print(installed_path)
PY
)"

"$python_command" - "$project_root" <<'PY' | node "$plugin_root/scripts/recall.mjs" >"$work_root/hook.out"
import json
import sys

print(json.dumps({
    "session_id": "real-installer-acceptance",
    "cwd": sys.argv[1],
    "hook_event_name": "UserPromptSubmit",
    "prompt": "installer acceptance fact",
}))
PY
grep -q 'Installer acceptance fact' "$work_root/hook.out"

"$python_command" - "$state_dir/platform.sqlite3" "$work_root/state.after.json" <<'PY'
import json
import sqlite3
from pathlib import Path
import sys

database, output = map(Path, sys.argv[1:])
tables = (
    "memory_libraries",
    "project_bindings",
    "candidate_memories",
    "candidate_audit",
    "capture_inbox",
)
with sqlite3.connect(database) as connection:
    snapshot = {
        table: connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
        for table in tables
    }
output.write_text(json.dumps(snapshot, sort_keys=True), encoding="utf-8")
PY
cmp "$work_root/state.before.json" "$work_root/state.after.json"

systemctl --user stop "$service_name"
"$python_command" - "$project_root" <<'PY' | node "$plugin_root/scripts/recall.mjs" >"$work_root/fail-open.out"
import json
import sys

print(json.dumps({
    "session_id": "real-installer-fail-open",
    "cwd": sys.argv[1],
    "hook_event_name": "UserPromptSubmit",
    "prompt": "installer acceptance fact",
}))
PY
test ! -s "$work_root/fail-open.out"

[[ "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" == "$repo_status_before" ]]
[[ "$(git -C "$repo_root" remote -v)" == "$repo_remotes_before" ]]
printf 'real Linux installer acceptance passed\n'
