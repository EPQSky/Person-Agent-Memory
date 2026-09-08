#!/usr/bin/env bash
set -euo pipefail

service_name="personal-agent-memory.service"
web_url="http://127.0.0.1:7331"
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
state_dir="${HOME}/.local/share/personal-agent-memory"
library_root="${HOME}/memory-libraries"
config_home="${XDG_CONFIG_HOME:-${HOME}/.config}"
codex_home="${CODEX_HOME:-${HOME}/.codex}"
unit_dir="${config_home}/systemd/user"
unit_path="${unit_dir}/${service_name}"
failed_stage="initialization"

on_error() {
  status=$?
  trap - ERR
  printf 'Installation failed during %s.\n' "$failed_stage" >&2
  printf 'Service status: systemctl --user status %s\n' "$service_name" >&2
  printf 'Service logs: journalctl --user -u %s --no-pager\n' "$service_name" >&2
  exit "$status"
}
trap on_error ERR

quote_unit_argument() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '"%s"' "$value"
}

dependency_error() {
  printf 'Dependency check failed: %s\n' "$1" >&2
  exit 1
}

failed_stage="dependency checks"

[[ "$(uname -s 2>/dev/null)" == "Linux" ]] || dependency_error \
  "Personal Agent Memory requires Linux. Use Ubuntu 22.04, Ubuntu 24.04, or another supported Linux distribution."

[[ "$(uname -m 2>/dev/null)" == "x86_64" ]] || dependency_error \
  "Personal Agent Memory requires x86_64. This installer does not support the detected CPU architecture."

if ! command -v systemctl >/dev/null 2>&1 || ! systemctl --user show-environment >/dev/null 2>&1; then
  dependency_error \
    "A working systemd user service manager is required. Install systemd and start a user session with systemd --user available."
fi

python_command=""
for candidate in python3 python3.13 python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
    'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)' \
    >/dev/null 2>&1; then
    python_command="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$python_command" ]]; then
  dependency_error \
    "Python 3.11, 3.12, or 3.13 is required. Install a supported Python version and rerun this installer."
fi

if ! command -v git >/dev/null 2>&1 || ! git --version >/dev/null 2>&1; then
  dependency_error "A working Git installation is required. Install Git and rerun this installer."
fi

if ! command -v node >/dev/null 2>&1 || ! node --version >/dev/null 2>&1 || ! node -e \
  'process.exit(typeof fetch === "function" && typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function" ? 0 : 1)' \
  >/dev/null 2>&1; then
  dependency_error \
    "A compatible Node.js installation with fetch and AbortSignal.timeout is required. Install a current Node.js release and rerun this installer."
fi

codex_plugin_help=""
codex_marketplace_help=""
if ! command -v codex >/dev/null 2>&1 || ! codex --version >/dev/null 2>&1 || \
  ! codex_plugin_help="$(codex plugin add --help 2>/dev/null)" || \
  ! codex_marketplace_help="$(codex plugin marketplace add --help 2>/dev/null)" || \
  [[ "$codex_plugin_help" != *"--json"* ]] || \
  [[ "$codex_marketplace_help" != *"--json"* ]]; then
  dependency_error \
    "A compatible Codex CLI with plugin and plugin marketplace commands is required. Update Codex CLI and rerun this installer."
fi

uv_bin_dir=""
uv_is_compatible() {
  local candidate=$1
  command -v "$candidate" >/dev/null 2>&1 || return 1
  "$candidate" --version >/dev/null 2>&1 || return 1
  "$candidate" tool install --python "$python_command" --force --from "$source_root" \
    personal-agent-memory --help >/dev/null 2>&1 || return 1
  uv_bin_dir="$("$candidate" tool dir --bin 2>/dev/null)" || return 1
  [[ -n "$uv_bin_dir" ]]
}

uv_command=""
if uv_is_compatible uv; then
  uv_command="$(command -v uv)"
else
  uv_install_dir="${XDG_BIN_HOME:-${HOME}/.local/bin}"
  if ! uv_installer="$(mktemp "${TMPDIR:-/tmp}/personal-agent-memory-uv.XXXXXX")"; then
    dependency_error \
      "Could not install uv in the user environment. Install uv manually from https://docs.astral.sh/uv/ and rerun this installer."
  fi
  cleanup_uv_installer() {
    rm -f "$uv_installer"
  }
  trap cleanup_uv_installer EXIT

  if ! command -v curl >/dev/null 2>&1 || ! curl --proto '=https' --tlsv1.2 -LsSf \
    https://astral.sh/uv/install.sh -o "$uv_installer"; then
    dependency_error \
      "Could not install uv in the user environment. Install uv manually from https://docs.astral.sh/uv/ and rerun this installer."
  fi
  if ! mkdir -p "$uv_install_dir" || ! UV_INSTALL_DIR="$uv_install_dir" UV_NO_MODIFY_PATH=1 \
    sh "$uv_installer" >/dev/null; then
    dependency_error \
      "Could not install uv in the user environment. Install uv manually from https://docs.astral.sh/uv/ and rerun this installer."
  fi
  uv_command="${uv_install_dir}/uv"
  if [[ ! -x "$uv_command" ]] || ! uv_is_compatible "$uv_command"; then
    dependency_error \
      "Could not install uv in the user environment. Install uv manually from https://docs.astral.sh/uv/ and rerun this installer."
  fi
  cleanup_uv_installer
  trap - EXIT
fi

failed_stage="directory preparation"
mkdir -p "$state_dir" "$library_root" "$unit_dir" "$codex_home"
chmod 700 "$state_dir"
chmod 700 "$codex_home"

failed_stage="Python tool installation"
"$uv_command" tool install --python "$python_command" --force --from "$source_root" \
  personal-agent-memory
daemon_command="${uv_bin_dir}/personal-agent-memory"
[[ -x "$daemon_command" ]]

failed_stage="Codex plugin installation"
codex plugin marketplace add "$source_root" --json >/dev/null
codex plugin add personal-agent-memory@personal-agent-memory --json >/dev/null

failed_stage="user service installation"
daemon_unit_command="$(quote_unit_argument "$daemon_command")"
state_unit_argument="$(quote_unit_argument "$state_dir")"
library_unit_argument="$(quote_unit_argument "$library_root")"
cat >"$unit_path" <<EOF
[Unit]
Description=Personal Agent Memory daemon
After=network.target

[Service]
Type=simple
ExecStart=${daemon_unit_command} serve --state-dir ${state_unit_argument} --library-root ${library_unit_argument} --host "127.0.0.1" --port "7331"
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$service_name"

failed_stage="authenticated health check"
"$python_command" - "$state_dir/api-key" "$web_url/health/live" <<'PY'
import pathlib
import sys
import time
import urllib.error
import urllib.request

key_path = pathlib.Path(sys.argv[1])
health_url = sys.argv[2]
deadline = time.monotonic() + 30
last_error = "API key was not created"
while time.monotonic() < deadline:
    try:
        key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise OSError("API key is empty")
        request = urllib.request.Request(
            health_url,
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            if response.status == 200:
                break
            last_error = f"health endpoint returned {response.status}"
    except (OSError, urllib.error.URLError) as error:
        last_error = str(error)
    time.sleep(0.1)
else:
    raise SystemExit(f"daemon did not become healthy: {last_error}")
PY

failed_stage="service status check"
service_status="$(systemctl --user is-active "$service_name")"
[[ "$service_status" == "active" ]]

trap - ERR
cat <<EOF
Personal Agent Memory installation complete.
Service status: ${service_status}
Web: ${web_url}
API key file: ${state_dir}/api-key

Start:   systemctl --user start ${service_name}
Stop:    systemctl --user stop ${service_name}
Restart: systemctl --user restart ${service_name}
Status:  systemctl --user status ${service_name}
Logs:    journalctl --user -u ${service_name} --no-pager
EOF
