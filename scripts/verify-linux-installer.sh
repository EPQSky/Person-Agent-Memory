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
. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" =~ ^(22\.04|24\.04)$ ]] ||
  skip "Ubuntu 22.04 or 24.04 is required for this release acceptance entry"
for command in uv git node codex python3 systemctl; do
  command -v "$command" >/dev/null || skip "required command is unavailable: $command"
done
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

"$repo_root/install.sh" "${installer_arguments[@]}" >"$work_root/install.out"
systemctl --user is-enabled "$service_name" | grep -qx enabled
systemctl --user is-active "$service_name" | grep -qx active

key_file="$state_dir/api-key"
python3 - "$install_config_path" "$unit_path" "$state_dir" "$library_root" "$key_file" <<'PY'
import json
from pathlib import Path
import stat
import sys

config_path, unit_path, state_dir, library_root, key_file = map(Path, sys.argv[1:])
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
python3 - "$key_file" <<'PY'
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

python3 - "$key_file" "$memory_path" "$project_root" <<'PY'
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
PY

codex plugin add personal-agent-memory@personal-agent-memory --json >"$work_root/plugin-add.json"
codex plugin list --json >"$work_root/plugins.json"
plugin_root="$(python3 - "$work_root/plugin-add.json" "$work_root/plugins.json" <<'PY'
import json
import sys

installed = json.load(open(sys.argv[1], encoding="utf-8"))
plugins = json.load(open(sys.argv[2], encoding="utf-8"))["installed"]
matches = [item for item in plugins if item["pluginId"] == "personal-agent-memory@personal-agent-memory"]
assert len(matches) == 1
print(installed["installedPath"])
PY
)"

python3 - "$project_root" <<'PY' | node "$plugin_root/scripts/recall.mjs" >"$work_root/hook.out"
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

systemctl --user stop "$service_name"
python3 - "$project_root" <<'PY' | node "$plugin_root/scripts/recall.mjs" >"$work_root/fail-open.out"
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
