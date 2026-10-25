#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service_name="personal-agent-memory.service"
config_home="${XDG_CONFIG_HOME:-$HOME/.config}"
install_config_dir="$config_home/personal-agent-memory"
install_config_path="$install_config_dir/install.json"
unit_path="$config_home/systemd/user/$service_name"
codex_home="${CODEX_HOME:-$HOME/.codex}"
web_port=7331
check_only=false
installer_arguments=()
install_arguments=()

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

# Keep the acceptance input auditable: scripts and package/plugin manifests must come from one
# release checkout. The GitHub Release source archive is expected to contain these same files.
for release_asset in install.sh uninstall.sh pyproject.toml uv.lock \
  plugins/personal-agent-memory/.codex-plugin/plugin.json; do
  [[ -f "$repo_root/$release_asset" ]] ||
    skip "release asset is missing from the checkout: $release_asset"
done
[[ -x "$repo_root/install.sh" && -x "$repo_root/uninstall.sh" ]] ||
  skip "release installer and uninstaller must be executable"
for command in git node codex systemctl; do
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
managed_bin_dir="${XDG_BIN_HOME:-$HOME/.local/bin}"
case ":$PATH:" in
  *":$managed_bin_dir:"*) ;;
  *) PATH="$managed_bin_dir:$PATH"; export PATH ;;
esac
uv_initially_available=false
command -v uv >/dev/null 2>&1 && uv_initially_available=true
if [[ "${PAM_INSTALL_ACCEPTANCE_MISSING_UV:-}" == "1" &&
  "${PAM_INSTALL_ACCEPTANCE_SCENARIO:-all}" == "all" &&
  "$uv_initially_available" == true ]]; then
  skip "PAM_INSTALL_ACCEPTANCE_MISSING_UV=1 requires uv to be absent before the acceptance run"
fi

if $check_only; then
  printf 'real Linux installer acceptance prerequisites satisfied\n'
  exit 0
fi

acceptance_scenario=${PAM_INSTALL_ACCEPTANCE_SCENARIO:-all}
case "$acceptance_scenario" in
  all)
    repo_status_before="$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)"
    repo_remotes_before="$(git -C "$repo_root" remote -v)"
    PAM_INSTALL_ACCEPTANCE_SCENARIO=normal "$0"
    PAM_INSTALL_ACCEPTANCE_SCENARIO=purge "$0"
    [[ "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" == \
      "$repo_status_before" ]]
    [[ "$(git -C "$repo_root" remote -v)" == "$repo_remotes_before" ]]
    printf 'real Linux installer acceptance passed\n'
    exit 0
    ;;
  normal|purge) ;;
  *)
    printf 'Unknown internal acceptance scenario: %s\n' "$acceptance_scenario" >&2
    exit 2
    ;;
esac

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

# The normal scenario starts with a Codex marketplace owned by the user. The installer must
# adopt it without taking ownership or deleting it during uninstall. The purge scenario starts
# from a clean account and verifies installer-managed marketplace cleanup.
if [[ "$acceptance_scenario" == "normal" ]]; then
  codex plugin marketplace add "$repo_root" --json >"$work_root/marketplace-preexisting.json"
  install_arguments=(--adopt-marketplace)
fi

"$repo_root/install.sh" "${install_arguments[@]}" "${installer_arguments[@]}" \
  >"$work_root/install.out"
if [[ "${PAM_INSTALL_ACCEPTANCE_MISSING_UV:-}" == "1" &&
  "$acceptance_scenario" == "normal" ]]; then
  [[ "$uv_initially_available" == false ]]
  command -v uv >/dev/null 2>&1
  printf 'missing uv bootstrap acceptance passed\n' >"$work_root/uv-bootstrap.out"
fi
systemctl --user is-enabled "$service_name" | grep -qx enabled
systemctl --user is-active "$service_name" | grep -qx active
uv tool list | grep -Eq "^personal-agent-memory v${target_version//./\\.}$"
codex plugin list --json >"$work_root/plugins-installed.json"
"$python_command" - "$work_root/plugins-installed.json" "$target_version" <<'PY'
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

daemon_command="$(command -v personal-agent-memory)"
"$daemon_command" serve --help >"$work_root/serve-help.out"
if grep -q -- '--memory-kernel' "$work_root/serve-help.out"; then
  printf 'installed CLI still exposes the removed memory kernel flag\n' >&2
  exit 1
fi
key_file="$state_dir/api-key"
[[ -s "$key_file" ]]
[[ "$(cat "$state_dir/mem2/VERSION")" == "jiuwen-memory-0.2.0" ]]
[[ ! -e "$state_dir/platform.sqlite3" ]]
unit_text="$(cat "$unit_path")"
[[ "$unit_text" != *"--memory-kernel"* ]]
[[ "$unit_text" == *"--state-dir \"$state_dir\""* ]]
[[ "$unit_text" == *"--library-root \"$library_root\""* ]]

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
user_content="$library_root/installer-acceptance"
mkdir -p "$project_root" "$user_content"
printf 'Installer acceptance user content.\n' >"$user_content/notes.txt"

"$python_command" - "$key_file" "$project_root" <<'PY'
import json
import pathlib
import sys
import urllib.request

key = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip()
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def request(path, *, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request_headers = dict(headers)
    if data is None:
        request_headers.pop("Content-Type")
    request = urllib.request.Request(
        f"http://127.0.0.1:7331{path}",
        data=data,
        headers=request_headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)

health = request("/mcp/health")
assert health["memory_kernel"] == "mem2"
assert health["memory_kernel_status"] in {"ready", "degraded"}
assert request("/api/v1/regions/tree")["root"]["region_id"] == "user"
region = request(
    "/api/v1/regions",
    method="POST",
    payload={
        "kind": "project",
        "name": "Installer acceptance",
        "region_id": "installer-acceptance",
    },
)
request(
    "/api/v1/project-bindings",
    method="POST",
    payload={"project_root": sys.argv[2], "region_id": region["region_id"]},
)
assert request(
    "/mcp/project-bindings/resolve",
    method="POST",
    payload={"cwd": sys.argv[2]},
)["status"] == "bound"
request(
    "/api/v1/memories",
    method="POST",
    payload={
        "region_id": region["region_id"],
        "content": "Installer acceptance fact.",
        "memory_type": "domain_fact",
        "source_references": ["linux-installer:acceptance"],
        "actor": "linux-installer",
        "idempotency_key": "linux-installer-acceptance-memory",
    },
)
search = request(
    "/mcp",
    method="POST",
    payload={
        "jsonrpc": "2.0",
        "id": "linux-installer",
        "method": "tools/call",
        "params": {
            "name": "memory_search",
            "arguments": {
                "region_ids": [region["region_id"]],
                "query": "Installer acceptance fact",
            },
        },
    },
)
assert search["result"]["isError"] is False
assert search["result"]["structuredContent"]["results"]
PY

cp "$key_file" "$work_root/api-key.before"
cp "$state_dir/.personal-agent-memory-owned.json" "$work_root/state-owner.before"
cp "$state_dir/mem2/VERSION" "$work_root/VERSION.before"
cp "$user_content/notes.txt" "$work_root/user-content.before"
stat -c '%d:%i' "$state_dir/mem2/regions.sqlite3" >"$work_root/regions-inode.before"
stat -c '%d:%i' "$state_dir/mem2/memory.sqlite3" >"$work_root/memory-inode.before"

"$repo_root/install.sh" "${install_arguments[@]}" "${installer_arguments[@]}" \
  >"$work_root/install-reinstall.out"
systemctl --user is-enabled "$service_name" | grep -qx enabled
systemctl --user is-active "$service_name" | grep -qx active
[[ "$(personal-agent-memory --version)" == "$target_version" ]]
cmp "$work_root/api-key.before" "$key_file"
cmp "$work_root/state-owner.before" "$state_dir/.personal-agent-memory-owned.json"
cmp "$work_root/VERSION.before" "$state_dir/mem2/VERSION"
cmp "$work_root/user-content.before" "$user_content/notes.txt"
cmp "$work_root/regions-inode.before" <(stat -c '%d:%i' "$state_dir/mem2/regions.sqlite3")
cmp "$work_root/memory-inode.before" <(stat -c '%d:%i' "$state_dir/mem2/memory.sqlite3")
[[ "$(cat "$state_dir/mem2/VERSION")" == "jiuwen-memory-0.2.0" ]]
[[ ! -e "$state_dir/platform.sqlite3" ]]

mapfile -t managed_paths < <("$python_command" - "$install_config_path" \
  "$state_dir" "$library_root" "$target_version" "$repo_root" \
  "$(command -v uv)" "$(command -v personal-agent-memory)" "$unit_path" \
  "$codex_home" "$acceptance_scenario" "$web_port" <<'PY'
import json
from pathlib import Path
import re
import stat
import sys

config_path = Path(sys.argv[1])
metadata = json.loads(config_path.read_text(encoding="utf-8"))
uninstall_sha256 = metadata.pop("uninstall_sha256")
uninstall_path = Path(metadata.pop("uninstall_path"))
unit_sha256 = metadata.pop("unit_sha256")
state_ownership_token = metadata.pop("state_ownership_token")
plugin_registration_source = Path(metadata.pop("plugin_registration_source"))
plugin_install_path = Path(metadata.pop("plugin_install_path"))
plugin_tree_sha256 = metadata.pop("plugin_tree_sha256")
scenario = sys.argv[10]
expected = {
    "schema_version": 1,
    "port": int(sys.argv[11]),
    "state_dir": sys.argv[2],
    "library_root": sys.argv[3],
    "library_root_ownership": "user-content-never-delete",
    "state_dir_ownership": "installer-created-exclusive",
    "installed_version": sys.argv[4],
    "marketplace_ownership": (
        "preexisting" if scenario == "normal" else "installer-managed"
    ),
    "marketplace_source": sys.argv[5],
    "marketplace_update_pending": False,
    "uninstall_ownership": "installer-managed",
    "uv_path": sys.argv[6],
    "daemon_path": sys.argv[7],
    "unit_path": sys.argv[8],
    "codex_home": sys.argv[9],
    "plugin_id": "personal-agent-memory@personal-agent-memory",
    "plugin_version": sys.argv[4],
}
assert metadata == expected
for digest in (uninstall_sha256, unit_sha256, plugin_tree_sha256, state_ownership_token):
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
assert uninstall_path.is_file()
assert uninstall_path.name == "personal-agent-memory-uninstall"
assert plugin_registration_source.is_dir()
assert plugin_install_path.is_dir()
assert str(config_path.read_text(encoding="utf-8")).find(
    Path(sys.argv[2], "api-key").read_text(encoding="utf-8").strip()
) == -1
print(uninstall_path)
print(metadata["uv_path"])
print(plugin_registration_source)
print(plugin_install_path)
PY
)
uninstall_command=${managed_paths[0]}
managed_uv_path=${managed_paths[1]}
managed_plugin_registration_source=${managed_paths[2]}
managed_plugin_install_path=${managed_paths[3]}

codex plugin list --json >"$work_root/plugins.json"
plugin_root="$("$python_command" - "$work_root/plugins.json" "$target_version" <<'PY'
import json
from pathlib import Path
import sys

plugins = json.load(open(sys.argv[1], encoding="utf-8"))["installed"]
matches = [
    item
    for item in plugins
    if item["pluginId"] == "personal-agent-memory@personal-agent-memory"
]
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
[[ "$(readlink -f "$plugin_root")" == \
  "$(readlink -f "$managed_plugin_registration_source")" ]]

"$python_command" - "$key_file" "$project_root" <<'PY'
import json
import pathlib
import sys
import urllib.request

key = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip()
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def request(path, *, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request_headers = dict(headers)
    if data is None:
        request_headers.pop("Content-Type")
    request = urllib.request.Request(
        f"http://127.0.0.1:7331{path}",
        data=data,
        headers=request_headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)

health = request("/mcp/health")
assert health["memory_kernel"] == "mem2"
assert health["memory_kernel_status"] in {"ready", "degraded"}
regions = request("/api/v1/regions")
assert any(region["region_id"] == "installer-acceptance" for region in regions)
assert request("/api/v1/regions/installer-acceptance")["region_id"] == "installer-acceptance"
assert request(
    "/mcp/project-bindings/resolve",
    method="POST",
    payload={"cwd": sys.argv[2]},
)["status"] == "bound"
search = request(
    "/mcp",
    method="POST",
    payload={
        "jsonrpc": "2.0",
        "id": "linux-installer-reinstall",
        "method": "tools/call",
        "params": {
            "name": "memory_search",
            "arguments": {
                "region_ids": ["installer-acceptance"],
                "query": "Installer acceptance fact",
            },
        },
    },
)
assert search["result"]["isError"] is False
assert search["result"]["structuredContent"]["results"]
project_search = request(
    "/api/v1/search",
    method="POST",
    payload={
        "cwd": sys.argv[2],
        "query": "Installer acceptance fact",
        "token_budget": 10_000,
    },
)
assert project_search["results"]
PY

export PLUGIN_DATA="$work_root/plugin-data"
"$python_command" - "$project_root" <<'PY' | \
  node "$managed_plugin_install_path/scripts/recall.mjs" >"$work_root/hook.out"
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
"$python_command" - "$project_root" <<'PY' | \
  node "$managed_plugin_install_path/scripts/capture.mjs"
import json
import sys

print(json.dumps({
    "session_id": "real-installer-capture",
    "cwd": sys.argv[1],
    "hook_event_name": "UserPromptSubmit",
    "prompt": "remember the installer acceptance",
}))
PY
[[ -z "$(find "$PLUGIN_DATA/spool" -maxdepth 1 -name 'event-*.json' -print -quit 2>/dev/null)" ]]

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

if [[ "$acceptance_scenario" == "normal" ]]; then
  "$uninstall_command" >"$work_root/uninstall.out"
else
  "$uninstall_command" --purge >"$work_root/purge.out"
fi
! systemctl --user is-enabled "$service_name" >/dev/null 2>&1
! systemctl --user is-active "$service_name" >/dev/null 2>&1
[[ ! -e "$unit_path" ]]
! command -v personal-agent-memory >/dev/null 2>&1
[[ ! -e "$uninstall_command" ]]
if [[ "$acceptance_scenario" == "normal" ]]; then
  [[ -e "$state_dir/api-key" ]]
  [[ -e "$state_dir/mem2/VERSION" ]]
  [[ -e "$state_dir/.personal-agent-memory-owned.json" ]]
  [[ -e "$install_config_path" ]]
  cmp "$work_root/api-key.before" "$state_dir/api-key"
  cmp "$work_root/state-owner.before" "$state_dir/.personal-agent-memory-owned.json"
  cmp "$work_root/VERSION.before" "$state_dir/mem2/VERSION"
  [[ ! -e "$state_dir/platform.sqlite3" ]]
else
  [[ ! -e "$state_dir" ]]
  [[ ! -e "$install_config_path" ]]
fi
[[ -e "$library_root" ]]
cmp "$work_root/user-content.before" "$user_content/notes.txt"
codex plugin list --json >"$work_root/plugins-after-uninstall.json"
codex plugin marketplace list --json >"$work_root/marketplaces-after-uninstall.json"
"$managed_uv_path" tool list >"$work_root/tools-after-uninstall.txt"
"$python_command" - "$acceptance_scenario" \
  "$work_root/plugins-after-uninstall.json" \
  "$work_root/marketplaces-after-uninstall.json" \
  "$work_root/tools-after-uninstall.txt" <<'PY'
# lifecycle-post-uninstall-contract
import json
from pathlib import Path
import sys

scenario = sys.argv[1]
installed = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["installed"]
assert not any(
    item.get("pluginId") == "personal-agent-memory@personal-agent-memory"
    for item in installed
)
marketplaces = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["marketplaces"]
marketplace_count = sum(
    item.get("name") == "personal-agent-memory" for item in marketplaces
)
assert marketplace_count == (1 if scenario == "normal" else 0)
tools = Path(sys.argv[4]).read_text(encoding="utf-8").splitlines()
assert not any(
    line.split(maxsplit=1)[0] == "personal-agent-memory"
    for line in tools
    if line.split()
)
PY

[[ "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" == "$repo_status_before" ]]
[[ "$(git -C "$repo_root" remote -v)" == "$repo_remotes_before" ]]
printf 'real Linux installer %s acceptance scenario passed\n' "$acceptance_scenario"
