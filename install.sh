#!/usr/bin/env bash
set -euo pipefail

service_name="personal-agent-memory.service"
web_url="http://127.0.0.1:7331"
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_home="${XDG_CONFIG_HOME:-${HOME}/.config}"
codex_home="${CODEX_HOME:-${HOME}/.codex}"
install_config_dir="${config_home}/personal-agent-memory"
install_config_path="${install_config_dir}/install.json"
unit_dir="${config_home}/systemd/user"
unit_path="${unit_dir}/${service_name}"
failed_stage="initialization"
state_dir_argument=""
library_root_argument=""

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
  value=${value//\$/\$\$}
  value=${value//\%/%%}
  printf '"%s"' "$value"
}

usage() {
  cat <<'EOF'
Usage: install.sh [--state-dir PATH] [--library-root PATH]

  --state-dir PATH     Directory for platform-owned state and credentials.
  --library-root PATH  Parent directory containing user-owned memory libraries.
  -h, --help           Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-dir|--library-root)
      [[ $# -ge 2 && -n "${2:-}" ]] || {
        printf 'Missing path for %s.\n' "$1" >&2
        usage >&2
        exit 2
      }
      if [[ "$1" == "--state-dir" ]]; then
        state_dir_argument=$2
      else
        library_root_argument=$2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown installer option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

dependency_error() {
  printf 'Dependency check failed: %s\n' "$1" >&2
  exit 1
}

failed_stage="dependency checks"

if [[ -n "${XDG_CONFIG_HOME:-}" && "${XDG_CONFIG_HOME}" != /* ]]; then
  dependency_error "XDG_CONFIG_HOME must be an absolute path when set."
fi

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

failed_stage="install directory validation"
directory_config="$($python_command - "$install_config_path" "$state_dir_argument" "$library_root_argument" <<'PY'
import json
import os
from pathlib import Path
import stat
import sys

MAX_INSTALL_METADATA_BYTES = 64 * 1024
config_path = Path(sys.argv[1]).expanduser().resolve()
state_argument = sys.argv[2]
library_argument = sys.argv[3]

def read_install_metadata(path: Path) -> object | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SystemExit(f"cannot open existing install metadata {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SystemExit(f"existing install metadata must be a regular file: {path}")
        if metadata.st_size > MAX_INSTALL_METADATA_BYTES:
            raise SystemExit(f"existing install metadata exceeds the 65536-byte limit: {path}")
        contents = bytearray()
        while len(contents) <= MAX_INSTALL_METADATA_BYTES:
            chunk = os.read(
                descriptor,
                min(8192, MAX_INSTALL_METADATA_BYTES + 1 - len(contents)),
            )
            if not chunk:
                break
            contents.extend(chunk)
        if len(contents) > MAX_INSTALL_METADATA_BYTES:
            raise SystemExit(f"existing install metadata exceeds the 65536-byte limit: {path}")
        return json.loads(contents.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read existing install metadata {path}: {error}") from error
    finally:
        os.close(descriptor)

saved = read_install_metadata(config_path)
if saved is not None:
    if not isinstance(saved, dict):
        raise SystemExit(f"existing install metadata is invalid: {config_path}")
    schema_version = saved.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise SystemExit(f"existing install metadata has an unsupported schema: {config_path}")
    if saved.get("library_root_ownership") != "user-content-never-delete":
        raise SystemExit(
            f"existing install metadata does not mark the memory library as user content: {config_path}"
        )

def selected(argument: str, key: str, default: str) -> Path:
    if argument:
        value = argument
    elif saved is None:
        value = default
    else:
        value = saved.get(key) if isinstance(saved, dict) else None
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise SystemExit(
                f"existing install metadata has an invalid {key}; provide --{key.replace('_', '-')} to replace it"
            )
    if not isinstance(value, str) or not value:
        raise SystemExit(f"existing install metadata has an invalid {key}")
    if any(character in value for character in "\r\n"):
        raise SystemExit(f"{key} must not contain line breaks")
    expanded = os.path.expanduser(value)
    if value.startswith("~") and expanded.startswith("~"):
        raise SystemExit(f"{key} contains a user home that could not be expanded: {value}")
    try:
        return Path(expanded).resolve()
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"{key} could not be resolved: {value}: {error}") from error

home = Path.home()
state_dir = selected(
    state_argument, "state_dir", str(home / ".local/share/personal-agent-memory")
)
library_root = selected(library_argument, "library_root", str(home / "memory-libraries"))

def contains(parent: Path, child: Path) -> bool:
    return child == parent or child.is_relative_to(parent)

if contains(state_dir, library_root) or contains(library_root, state_dir):
    raise SystemExit(
        "platform state directory and memory library root must be separate, non-overlapping paths"
    )
if contains(library_root, config_path):
    raise SystemExit(
        "memory library root conflicts with the install metadata location; choose a narrower memory library root"
    )

print(state_dir)
print(library_root)
print(config_path)
PY
)" || dependency_error "Invalid install directories. ${directory_config:-Review the selected paths and rerun the installer.}"
mapfile -t resolved_directories <<<"$directory_config"
state_dir=${resolved_directories[0]}
library_root=${resolved_directories[1]}
install_config_path=${resolved_directories[2]}
install_config_dir=${install_config_path%/*}

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
if ! "$python_command" - "$state_dir" "$library_root" "$install_config_dir" "$unit_dir" "$codex_home" <<'PY'
from pathlib import Path
import tempfile
import sys

for value in sys.argv[1:]:
    path = Path(value)
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise NotADirectoryError(path)
        with tempfile.NamedTemporaryFile(dir=path):
            pass
    except OSError as error:
        raise SystemExit(f"directory is not usable: {path}: {error}")
PY
then
  dependency_error "Could not create or write the selected install directories. Check their paths and permissions."
fi
chmod 700 "$state_dir"
chmod 700 "$codex_home"
chmod 700 "$install_config_dir"

failed_stage="Python tool installation"
"$uv_command" tool install --python "$python_command" --force --from "$source_root" \
  personal-agent-memory
daemon_command="${uv_bin_dir}/personal-agent-memory"
[[ -x "$daemon_command" ]]

failed_stage="Codex plugin installation"
codex plugin marketplace add "$source_root" --json >/dev/null
codex plugin add personal-agent-memory@personal-agent-memory --json >/dev/null

failed_stage="install metadata persistence"
"$python_command" - "$install_config_path" "$state_dir" "$library_root" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

config_path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "state_dir": sys.argv[2],
    "library_root": sys.argv[3],
    "library_root_ownership": "user-content-never-delete",
}
descriptor, temporary_name = tempfile.mkstemp(prefix=".install.", dir=config_path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(temporary_name, 0o600)
    os.replace(temporary_name, config_path)
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY

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
