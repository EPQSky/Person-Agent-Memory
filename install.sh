#!/usr/bin/env bash
set -euo pipefail

service_name="personal-agent-memory.service"
web_url="http://127.0.0.1:7331"
web_port=7331
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
port_argument=""
adopt_marketplace=false

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

persist_install_metadata() {
  local installed_version=$1
  local marketplace_ownership=$2
  local marketplace_source=$3
  local marketplace_previous_source=$4
  local marketplace_update_pending=$5
  local marketplace_update_from_source=$6
  local uninstall_path=$7
  local uninstall_sha256=$8
  local state_ownership_token=$9
  local state_dir_ownership=${10}
  local uv_path=${11}
  local daemon_path=${12}
  local managed_unit_path=${13}
  local managed_unit_sha256=${14}
  local managed_codex_home=${15}
  local plugin_id=${16}
  local plugin_version=${17}
  local plugin_registration_source=${18}
  local plugin_install_path=${19}
  local plugin_tree_sha256=${20}
  local port=${21}

  "$python_command" - "$install_config_path" "$state_dir" "$library_root" \
    "$installed_version" "$marketplace_ownership" "$marketplace_source" \
    "$marketplace_previous_source" "$marketplace_update_pending" \
    "$marketplace_update_from_source" "$uninstall_path" "$uninstall_sha256" \
    "$state_ownership_token" "$state_dir_ownership" "$uv_path" "$daemon_path" \
    "$managed_unit_path" "$managed_unit_sha256" "$managed_codex_home" \
    "$plugin_id" "$plugin_version" "$plugin_registration_source" \
    "$plugin_install_path" "$plugin_tree_sha256" "$port" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

config_path = Path(sys.argv[1])
pending_value = sys.argv[8]
if pending_value not in {"true", "false"}:
    raise SystemExit("invalid marketplace update state")
string_fields = {
    "state_dir": sys.argv[2],
    "library_root": sys.argv[3],
    "installed_version": sys.argv[4],
    "marketplace_ownership": sys.argv[5],
    "marketplace_source": sys.argv[6],
    "marketplace_previous_source": sys.argv[7],
    "marketplace_update_from_source": sys.argv[9],
    "uninstall_path": sys.argv[10],
    "uninstall_sha256": sys.argv[11],
    "state_dir_ownership": sys.argv[13],
    "uv_path": sys.argv[14],
    "daemon_path": sys.argv[15],
    "unit_path": sys.argv[16],
    "unit_sha256": sys.argv[17],
    "codex_home": sys.argv[18],
    "plugin_id": sys.argv[19],
    "plugin_version": sys.argv[20],
    "plugin_registration_source": sys.argv[21],
    "plugin_install_path": sys.argv[22],
    "plugin_tree_sha256": sys.argv[23],
}
port = sys.argv[24]
if not port.isdigit() or not 1 <= int(port) <= 65535:
    raise SystemExit("invalid listen port")
for name, value in string_fields.items():
    if any(character in value for character in "\r\n"):
        raise SystemExit(f"invalid line break in install metadata field: {name}")
if not Path(sys.argv[18]).is_absolute():
    raise SystemExit("invalid non-absolute install metadata field: codex_home")
payload = {
    "schema_version": 1,
    "state_dir": sys.argv[2],
    "library_root": sys.argv[3],
    "library_root_ownership": "user-content-never-delete",
    "state_dir_ownership": sys.argv[13],
    "installed_version": sys.argv[4],
    "marketplace_ownership": sys.argv[5],
    "marketplace_source": sys.argv[6],
    "marketplace_update_pending": pending_value == "true",
    "uninstall_ownership": "installer-managed",
    "uninstall_path": sys.argv[10],
    "uninstall_sha256": sys.argv[11],
    "uv_path": sys.argv[14],
    "daemon_path": sys.argv[15],
    "unit_path": sys.argv[16],
    "unit_sha256": sys.argv[17],
    "codex_home": sys.argv[18],
    "port": int(port),
}
plugin_identity = (sys.argv[19], sys.argv[20], sys.argv[21], sys.argv[22])
plugin_tree_sha256 = sys.argv[23]
if any(plugin_identity):
    if (
        plugin_identity[0] != "personal-agent-memory@personal-agent-memory"
        or not plugin_identity[1]
        or not Path(plugin_identity[2]).is_absolute()
        or not Path(plugin_identity[3]).is_absolute()
    ):
        raise SystemExit("invalid managed plugin identity")
    if plugin_tree_sha256 and (
        len(plugin_tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plugin_tree_sha256)
    ):
        raise SystemExit("invalid managed plugin tree digest")
    payload["plugin_id"] = plugin_identity[0]
    payload["plugin_version"] = plugin_identity[1]
    payload["plugin_registration_source"] = plugin_identity[2]
    payload["plugin_install_path"] = plugin_identity[3]
    if plugin_tree_sha256:
        payload["plugin_tree_sha256"] = plugin_tree_sha256
elif plugin_identity != ("", "", "", "") or plugin_tree_sha256:
    raise SystemExit("incomplete managed plugin identity")
if sys.argv[12]:
    payload["state_ownership_token"] = sys.argv[12]
if sys.argv[7]:
    payload["marketplace_previous_source"] = sys.argv[7]
if sys.argv[9]:
    payload["marketplace_update_from_source"] = sys.argv[9]
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
}

usage() {
  cat <<'EOF'
Usage: install.sh [--state-dir PATH] [--library-root PATH] [--port PORT] [--adopt-marketplace]

  --state-dir PATH     Directory for platform-owned state and credentials.
  --library-root PATH  Parent directory containing user-owned memory libraries.
  --port PORT          Loopback TCP port for the local web service (default: 7331).
  --adopt-marketplace  Replace an unowned legacy marketplace with this release.
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
    --port)
      [[ $# -ge 2 && -n "${2:-}" ]] || {
        printf 'Missing port for --port.\n' >&2
        usage >&2
        exit 2
      }
      port_argument=$2
      shift 2
      ;;
    --adopt-marketplace)
      adopt_marketplace=true
      shift
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

if [[ "$source_root" == *$'\n'* || "$source_root" == *$'\r'* ]]; then
  dependency_error "Installer release path must not contain line breaks."
fi

failed_stage="dependency checks"

if [[ -n "${XDG_CONFIG_HOME:-}" && "${XDG_CONFIG_HOME}" != /* ]]; then
  dependency_error "XDG_CONFIG_HOME must be an absolute path when set."
fi
if [[ "$codex_home" != /* || "$codex_home" == *$'\n'* || "$codex_home" == *$'\r'* ]]; then
  dependency_error "CODEX_HOME must be an absolute path without line breaks."
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

codex_home="$($python_command - "$codex_home" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)"
export CODEX_HOME="$codex_home"

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
codex_list_help=""
codex_marketplace_list_help=""
codex_marketplace_remove_help=""
if ! command -v codex >/dev/null 2>&1 || ! codex --version >/dev/null 2>&1 || \
  ! codex_plugin_help="$(codex plugin add --help 2>/dev/null)" || \
  ! codex_marketplace_help="$(codex plugin marketplace add --help 2>/dev/null)" || \
  ! codex_list_help="$(codex plugin list --help 2>/dev/null)" || \
  ! codex_marketplace_list_help="$(codex plugin marketplace list --help 2>/dev/null)" || \
  ! codex_marketplace_remove_help="$(codex plugin marketplace remove --help 2>/dev/null)" || \
  [[ "$codex_plugin_help" != *"--json"* ]] || \
  [[ "$codex_marketplace_help" != *"--json"* ]] || \
  [[ "$codex_list_help" != *"--json"* ]] || \
  [[ "$codex_marketplace_list_help" != *"--json"* ]] || \
  [[ "$codex_marketplace_remove_help" != *"--json"* ]]; then
  dependency_error \
    "A compatible Codex CLI with plugin and plugin marketplace commands is required. Update Codex CLI and rerun this installer."
fi

failed_stage="release validation"
target_version="$($python_command - "$source_root" <<'PY'
import json
from pathlib import Path
import sys
import tomllib

source_root = Path(sys.argv[1])
uninstaller = source_root / "uninstall.sh"
if not uninstaller.is_file() or uninstaller.is_symlink():
    raise SystemExit(f"release uninstaller is missing or unsafe: {uninstaller}")
with (source_root / "pyproject.toml").open("rb") as stream:
    package_version = tomllib.load(stream)["project"]["version"]
plugin = json.loads(
    (source_root / "plugins/personal-agent-memory/.codex-plugin/plugin.json").read_text(
        encoding="utf-8"
    )
)
plugin_version = plugin.get("version")
if not isinstance(package_version, str) or not package_version:
    raise SystemExit("Python package release version is missing")
if plugin_version != package_version:
    raise SystemExit(
        f"release version mismatch: Python package {package_version}, Codex plugin {plugin_version}"
    )
print(package_version)
PY
)"

failed_stage="install directory validation"
directory_config="$($python_command - "$install_config_path" "$state_dir_argument" \
  "$library_root_argument" "$codex_home" "$port_argument" <<'PY'
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

MAX_INSTALL_METADATA_BYTES = 64 * 1024
config_path = Path(os.path.abspath(os.path.normpath(str(Path(sys.argv[1]).expanduser()))))
state_argument = sys.argv[2]
library_argument = sys.argv[3]
current_codex_home = sys.argv[4]
port_argument = sys.argv[5]

def read_install_metadata(path: Path) -> object | None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_NONBLOCK
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        try:
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise SystemExit(f"existing install metadata must be a regular file: {path}")
        except FileNotFoundError:
            return None
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
    installed_version = saved.get("installed_version", "")
    if not isinstance(installed_version, str) or any(
        character in installed_version for character in "\r\n"
    ):
        raise SystemExit(f"existing install metadata has invalid installed_version: {config_path}")
    marketplace_ownership = saved.get("marketplace_ownership")
    marketplace_source = saved.get("marketplace_source")
    marketplace_previous_source = saved.get("marketplace_previous_source", "")
    marketplace_update_pending = saved.get("marketplace_update_pending", False)
    marketplace_update_from_source = saved.get("marketplace_update_from_source", "")
    state_ownership_token = saved.get("state_ownership_token", "")
    state_dir_ownership = saved.get("state_dir_ownership", "legacy-managed-contents")
    uv_path = saved.get("uv_path", "")
    daemon_path = saved.get("daemon_path", "")
    managed_unit_path = saved.get("unit_path", "")
    managed_unit_sha256 = saved.get("unit_sha256", "")
    managed_codex_home = saved.get("codex_home", current_codex_home)
    uninstall_ownership = saved.get("uninstall_ownership", "")
    uninstall_path = saved.get("uninstall_path", "")
    uninstall_sha256 = saved.get("uninstall_sha256", "")
    plugin_id = saved.get("plugin_id", "")
    plugin_version = saved.get("plugin_version", "")
    plugin_registration_source = saved.get("plugin_registration_source", "")
    plugin_install_path = saved.get("plugin_install_path", "")
    plugin_tree_sha256 = saved.get("plugin_tree_sha256", "")
    if marketplace_ownership is None and marketplace_source is None:
        if (
            marketplace_previous_source
            or marketplace_update_pending
            or marketplace_update_from_source
        ):
            raise SystemExit(
                f"existing install metadata has invalid marketplace recovery state: {config_path}"
            )
        marketplace_ownership = "legacy-unknown"
        marketplace_source = ""
    elif (
        marketplace_ownership not in {
            "installer-managed",
            "legacy-unowned",
            "preexisting",
        }
        or not isinstance(marketplace_source, str)
        or not marketplace_source
        or any(character in marketplace_source for character in "\r\n")
        or not Path(marketplace_source).is_absolute()
    ):
        raise SystemExit(f"existing install metadata has invalid marketplace ownership: {config_path}")
    if (
        not isinstance(marketplace_previous_source, str)
        or any(character in marketplace_previous_source for character in "\r\n")
        or (
            marketplace_previous_source
            and not Path(marketplace_previous_source).is_absolute()
        )
        or type(marketplace_update_pending) is not bool
        or not isinstance(marketplace_update_from_source, str)
        or any(character in marketplace_update_from_source for character in "\r\n")
        or (
            marketplace_update_from_source
            and not Path(marketplace_update_from_source).is_absolute()
        )
        or (not marketplace_update_pending and marketplace_update_from_source)
        or (
            marketplace_ownership == "legacy-unowned"
            and (
                marketplace_previous_source
                or marketplace_update_pending
                or marketplace_update_from_source
            )
        )
    ):
        raise SystemExit(f"existing install metadata has invalid marketplace recovery state: {config_path}")
    if state_ownership_token and (
        not isinstance(state_ownership_token, str)
        or len(state_ownership_token) != 64
        or any(character not in "0123456789abcdef" for character in state_ownership_token)
    ):
        raise SystemExit(f"existing install metadata has invalid state ownership: {config_path}")
    if state_dir_ownership not in {
        "installer-created-exclusive",
        "legacy-managed-contents",
        "preexisting-unproven",
    }:
        raise SystemExit(f"existing install metadata has invalid state directory ownership: {config_path}")
    if state_dir_ownership == "installer-created-exclusive" and not state_ownership_token:
        raise SystemExit(f"existing install metadata is missing state ownership proof: {config_path}")
    for name, value in {
        "uv_path": uv_path,
        "daemon_path": daemon_path,
        "unit_path": managed_unit_path,
    }.items():
        if value and (
            not isinstance(value, str)
            or any(character in value for character in "\r\n")
            or not Path(value).is_absolute()
        ):
            raise SystemExit(f"existing install metadata has invalid {name}: {config_path}")
    if (
        not isinstance(managed_codex_home, str)
        or not managed_codex_home
        or any(character in managed_codex_home for character in "\r\n")
        or not Path(managed_codex_home).is_absolute()
    ):
        raise SystemExit(f"existing install metadata has invalid codex_home: {config_path}")
    if managed_unit_sha256 and (
        not isinstance(managed_unit_sha256, str)
        or len(managed_unit_sha256) != 64
        or any(character not in "0123456789abcdef" for character in managed_unit_sha256)
        or not managed_unit_path
    ):
        raise SystemExit(f"existing install metadata has invalid unit ownership: {config_path}")
    uninstall_fields = (uninstall_ownership, uninstall_path, uninstall_sha256)
    if any(uninstall_fields):
        if (
            uninstall_ownership != "installer-managed"
            or not isinstance(uninstall_path, str)
            or not uninstall_path
            or any(character in uninstall_path for character in "\r\n")
            or not Path(uninstall_path).is_absolute()
            or not isinstance(uninstall_sha256, str)
            or len(uninstall_sha256) != 64
            or any(character not in "0123456789abcdef" for character in uninstall_sha256)
        ):
            raise SystemExit(f"existing install metadata has invalid uninstaller ownership: {config_path}")
    elif uninstall_fields != ("", "", ""):
        raise SystemExit(f"existing install metadata has invalid uninstaller ownership: {config_path}")
    plugin_identity = (
        plugin_id,
        plugin_version,
        plugin_registration_source,
        plugin_install_path,
    )
    if any(plugin_identity):
        if (
            plugin_id != "personal-agent-memory@personal-agent-memory"
            or not isinstance(plugin_version, str)
            or not plugin_version
            or any(character in plugin_version for character in "\r\n")
            or not isinstance(plugin_registration_source, str)
            or any(character in plugin_registration_source for character in "\r\n")
            or not Path(plugin_registration_source).is_absolute()
            or not isinstance(plugin_install_path, str)
            or any(character in plugin_install_path for character in "\r\n")
            or not Path(plugin_install_path).is_absolute()
        ):
            raise SystemExit(f"existing install metadata has invalid plugin identity: {config_path}")
        if plugin_tree_sha256 and (
            not isinstance(plugin_tree_sha256, str)
            or len(plugin_tree_sha256) != 64
            or any(character not in "0123456789abcdef" for character in plugin_tree_sha256)
        ):
            raise SystemExit(
                f"existing install metadata has invalid plugin tree digest: {config_path}"
            )
    elif plugin_identity != ("", "", "", "") or plugin_tree_sha256:
        raise SystemExit(f"existing install metadata has incomplete plugin identity: {config_path}")
else:
    installed_version = ""
    marketplace_ownership = "none"
    marketplace_source = ""
    marketplace_previous_source = ""
    marketplace_update_pending = False
    marketplace_update_from_source = ""
    state_ownership_token = ""
    state_dir_ownership = "preexisting-unproven"
    uv_path = ""
    daemon_path = ""
    managed_unit_path = ""
    managed_unit_sha256 = ""
    managed_codex_home = current_codex_home
    uninstall_ownership = ""
    uninstall_path = ""
    uninstall_sha256 = ""
    plugin_id = ""
    plugin_version = ""
    plugin_registration_source = ""
    plugin_install_path = ""
    plugin_tree_sha256 = ""

def selected_port(argument: str, saved_value: object) -> int:
    value: object = argument if argument else saved_value
    if not argument and saved is None:
        value = 7331
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SystemExit("port must be an integer between 1 and 65535")
    text = str(value)
    if not text.isascii() or not text.isdigit():
        raise SystemExit("port must be an integer between 1 and 65535")
    port = int(text)
    if not 1 <= port <= 65535:
        raise SystemExit("port must be an integer between 1 and 65535")
    return port

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
port = selected_port(port_argument, saved.get("port", 7331) if saved else 7331)

def contains(parent: Path, child: Path) -> bool:
    return child == parent or child.is_relative_to(parent)

if contains(state_dir, library_root) or contains(library_root, state_dir):
    raise SystemExit(
        "platform state directory and memory library root must be separate, non-overlapping paths"
    )
config_dir = config_path.parent
if contains(state_dir, config_dir) or contains(config_dir, state_dir):
    raise SystemExit(
        "platform state directory conflicts with the install metadata directory; "
        "choose separate, non-overlapping paths"
    )
if contains(library_root, config_path):
    raise SystemExit(
        "memory library root conflicts with the install metadata location; choose a narrower memory library root"
    )

def line_value(name: str, value: object) -> str:
    text = str(value)
    if any(character in text for character in "\r\n"):
        raise SystemExit(f"{name} must not contain line breaks")
    return text

print(line_value("state_dir", state_dir))
print(line_value("library_root", library_root))
print(line_value("install metadata path", config_path))
print(line_value("marketplace ownership", marketplace_ownership))
print(line_value("marketplace source", marketplace_source))
print(line_value("installed_version", installed_version))
print(line_value("marketplace previous source", marketplace_previous_source))
print("true" if marketplace_update_pending else "false")
print(line_value("marketplace update source", marketplace_update_from_source))
print(line_value("state ownership token", state_ownership_token))
print(line_value("state directory ownership", state_dir_ownership))
print(line_value("uv path", uv_path))
print(line_value("daemon path", daemon_path))
print(line_value("unit path", managed_unit_path))
print(line_value("unit sha256", managed_unit_sha256))
print(line_value("Codex home", Path(managed_codex_home).resolve()))
print(line_value("uninstaller ownership", uninstall_ownership))
print(line_value("uninstaller path", uninstall_path))
print(line_value("uninstaller sha256", uninstall_sha256))
print(line_value("plugin id", plugin_id))
print(line_value("plugin version", plugin_version))
print(line_value("plugin registration source", plugin_registration_source))
print(line_value("plugin install path", plugin_install_path))
print(line_value("plugin tree sha256", plugin_tree_sha256))
print(line_value("port", port))
PY
)" || dependency_error "Invalid install directories. ${directory_config:-Review the selected paths and rerun the installer.}"
mapfile -t resolved_directories <<<"$directory_config"
state_dir=${resolved_directories[0]}
library_root=${resolved_directories[1]}
install_config_path=${resolved_directories[2]}
saved_marketplace_ownership=${resolved_directories[3]}
saved_marketplace_source=${resolved_directories[4]:-}
saved_installed_version=${resolved_directories[5]:-}
saved_marketplace_previous_source=${resolved_directories[6]:-}
saved_marketplace_update_pending=${resolved_directories[7]:-false}
saved_marketplace_update_from_source=${resolved_directories[8]:-}
saved_state_ownership_token=${resolved_directories[9]:-}
saved_state_dir_ownership=${resolved_directories[10]:-preexisting-unproven}
saved_uv_path=${resolved_directories[11]:-}
saved_daemon_path=${resolved_directories[12]:-}
saved_unit_path=${resolved_directories[13]:-}
saved_unit_sha256=${resolved_directories[14]:-}
codex_home=${resolved_directories[15]}
saved_uninstall_ownership=${resolved_directories[16]:-}
saved_uninstall_path=${resolved_directories[17]:-}
saved_uninstall_sha256=${resolved_directories[18]:-}
saved_plugin_id=${resolved_directories[19]:-}
saved_plugin_version=${resolved_directories[20]:-}
saved_plugin_registration_source=${resolved_directories[21]:-}
saved_plugin_install_path=${resolved_directories[22]:-}
saved_plugin_tree_sha256=${resolved_directories[23]:-}
web_port=${resolved_directories[24]:-7331}
web_url="http://127.0.0.1:${web_port}"
export CODEX_HOME="$codex_home"
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
uv_needs_install=false
if uv_is_compatible uv; then
  uv_command="$(command -v uv)"
else
  uv_install_dir="${XDG_BIN_HOME:-${HOME}/.local/bin}"
  uv_bin_dir="$uv_install_dir"
  uv_command="${uv_install_dir}/uv"
  uv_needs_install=true
  export UV_TOOL_BIN_DIR="$uv_bin_dir"
fi

if [[ "$uv_bin_dir" != /* || "$uv_bin_dir" == *$'\n'* || "$uv_bin_dir" == *$'\r'* ]]; then
  dependency_error "uv reported an invalid user executable directory."
fi
uninstall_command="${uv_bin_dir}/personal-agent-memory-uninstall"
"$python_command" - "$uninstall_command" "$state_dir" "$library_root" <<'PY' || dependency_error \
  "The user executable directory overlaps platform state or memory libraries."
from pathlib import Path
import sys

uninstaller, state_dir, library_root = (Path(value).resolve() for value in sys.argv[1:])
if uninstaller.is_relative_to(state_dir) or uninstaller.is_relative_to(library_root):
    raise SystemExit("stable uninstaller would be stored inside a protected data directory")
PY

failed_stage="stable uninstaller ownership validation"
uninstall_destination_proof="$($python_command - "$uninstall_command" \
  "$saved_uninstall_ownership" "$saved_uninstall_path" "$saved_uninstall_sha256" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys

destination = Path(sys.argv[1])
ownership, recorded_path, recorded_sha256 = sys.argv[2:5]
recorded = bool(ownership or recorded_path or recorded_sha256)
if recorded and (
    ownership != "installer-managed"
    or recorded_path != str(destination)
    or len(recorded_sha256) != 64
    or any(character not in "0123456789abcdef" for character in recorded_sha256)
):
    raise SystemExit(
        f"existing install metadata does not prove ownership of stable uninstaller: {destination}"
    )

try:
    entry = os.lstat(destination)
except FileNotFoundError:
    print("absent")
    raise SystemExit(0)
if not recorded:
    raise SystemExit(f"stable uninstaller path already exists and is not installer-owned: {destination}")

flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(destination, flags)
except OSError as error:
    raise SystemExit(f"cannot verify stable uninstaller ownership {destination}: {error}") from error
try:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or (entry.st_dev, entry.st_ino) != (info.st_dev, info.st_ino):
        raise SystemExit(f"stable uninstaller is not the recorded regular file: {destination}")
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 8192):
        digest.update(chunk)
    if digest.hexdigest() != recorded_sha256:
        raise SystemExit(f"stable uninstaller contents do not match install metadata: {destination}")
    print(f"owned:{info.st_dev}:{info.st_ino}")
finally:
    os.close(descriptor)
PY
)"
IFS=: read -r uninstall_destination_state uninstall_destination_device \
  uninstall_destination_inode <<<"$uninstall_destination_proof"

failed_stage="Codex marketplace validation"
marketplace_list="$(codex plugin marketplace list --json)"
marketplace_plan="$($python_command - "$source_root" "$saved_marketplace_ownership" \
  "$saved_marketplace_source" "$saved_marketplace_previous_source" \
  "$saved_marketplace_update_pending" "$saved_marketplace_update_from_source" \
  "$adopt_marketplace" "$marketplace_list" <<'PY'
import json
from pathlib import Path
import sys

def line_safe(name: str, value: object) -> str:
    text = str(value)
    if any(character in text for character in "\r\n"):
        raise SystemExit(f"{name} must not contain line breaks")
    return text

def emit_plan(
    action: str,
    ownership: str,
    previous_source: Path | str | None,
    update_from_source: Path | str | None,
) -> None:
    print(line_safe("marketplace action", action))
    print(line_safe("marketplace ownership", ownership))
    print(line_safe("marketplace previous source", previous_source or ""))
    print(line_safe("marketplace update source", update_from_source or ""))

source_root = Path(line_safe("installer release path", sys.argv[1])).resolve()
saved_ownership = sys.argv[2]
saved_source = Path(sys.argv[3]).resolve() if sys.argv[3] else None
saved_previous = Path(sys.argv[4]).resolve() if sys.argv[4] else None
saved_pending = sys.argv[5] == "true"
saved_update_from = Path(sys.argv[6]).resolve() if sys.argv[6] else None
adopt_marketplace = sys.argv[7] == "true"
marketplaces = json.loads(sys.argv[8]).get("marketplaces", [])
matches = [item for item in marketplaces if item.get("name") == "personal-agent-memory"]
if len(matches) > 1:
    raise SystemExit("Codex reports duplicate personal-agent-memory marketplaces")
if not matches:
    emit_plan(
        "add",
        saved_ownership
        if saved_ownership in {"installer-managed", "preexisting"}
        else "installer-managed",
        saved_previous,
        None,
    )
elif isinstance(matches[0].get("root"), str):
    current_source = Path(
        line_safe("Codex marketplace root", matches[0]["root"])
    ).resolve()
    if current_source == source_root:
        if saved_ownership == "legacy-unknown":
            ownership = "legacy-unowned"
        elif saved_ownership in {
            "installer-managed",
            "legacy-unowned",
            "preexisting",
        }:
            ownership = saved_ownership
        else:
            ownership = "preexisting"
        emit_plan("keep", ownership, saved_previous, None)
    elif saved_pending and saved_update_from == current_source:
        emit_plan("replace", saved_ownership, saved_previous, current_source)
    elif (
        saved_ownership == "preexisting"
        and saved_source == current_source
        and adopt_marketplace
    ):
        emit_plan(
            "replace",
            "preexisting",
            saved_previous or current_source,
            current_source,
        )
    elif (
        saved_ownership == "preexisting"
        and saved_source == current_source
        and saved_previous is None
    ):
        raise SystemExit(
            "a preexisting personal-agent-memory marketplace requires explicit adoption; "
            "rerun with --adopt-marketplace only if this installer may replace it"
        )
    elif (
        saved_source == current_source
        and (
            saved_ownership == "installer-managed"
            or (saved_ownership == "preexisting" and saved_previous is not None)
        )
    ):
        emit_plan("replace", saved_ownership, saved_previous, current_source)
    elif saved_ownership in {"legacy-unknown", "legacy-unowned"} and adopt_marketplace:
        emit_plan("replace", "preexisting", current_source, current_source)
    elif saved_ownership in {"legacy-unknown", "legacy-unowned"}:
        raise SystemExit(
            "a legacy personal-agent-memory marketplace exists at a different source; "
            "rerun with --adopt-marketplace only if this installer may replace it"
        )
    else:
        raise SystemExit(
            "a personal-agent-memory marketplace already exists at a different source and is not owned by this installer"
        )
else:
    raise SystemExit(
        "a non-local personal-agent-memory marketplace already exists and will not be replaced"
    )
PY
)"
mapfile -t marketplace_plan_values <<<"$marketplace_plan"
marketplace_action=${marketplace_plan_values[0]}
marketplace_ownership=${marketplace_plan_values[1]}
marketplace_previous_source=${marketplace_plan_values[2]:-}
marketplace_update_from_source=${marketplace_plan_values[3]:-}

if $uv_needs_install; then
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
  planned_uv_bin_dir="$uv_bin_dir"
  if [[ ! -x "$uv_command" ]] || ! uv_is_compatible "$uv_command" || \
    [[ "$uv_bin_dir" != "$planned_uv_bin_dir" ]]; then
    dependency_error \
      "Could not install uv in the user environment. Install uv manually from https://docs.astral.sh/uv/ and rerun this installer."
  fi
  cleanup_uv_installer
  trap - EXIT
fi

failed_stage="directory preparation"
state_dir_preexisting=false
[[ -e "$state_dir" ]] && state_dir_preexisting=true
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
$state_dir_preexisting || chmod 700 "$state_dir"
chmod 700 "$codex_home"
chmod 700 "$install_config_dir"

failed_stage="platform state ownership validation"
state_ownership="$($python_command - "$state_dir" "$install_config_path" \
  "$saved_state_ownership_token" "$saved_state_dir_ownership" \
  "$state_dir_preexisting" <<'PY'
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile

state_dir = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
saved_token = sys.argv[3]
saved_ownership = sys.argv[4]
state_dir_preexisting = sys.argv[5] == "true"
marker_path = state_dir / ".personal-agent-memory-owned.json"
token = saved_token or secrets.token_hex(32)
ownership = saved_ownership
if marker_path.exists():
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        marker_descriptor = os.open(marker_path, flags)
        try:
            marker_info = os.fstat(marker_descriptor)
            if not stat.S_ISREG(marker_info.st_mode) or marker_info.st_size > 65536:
                raise OSError("marker must be a regular file no larger than 65536 bytes")
            marker_contents = os.read(marker_descriptor, 65537)
            if len(marker_contents) > 65536:
                raise OSError("marker exceeds the 65536-byte limit")
        finally:
            os.close(marker_descriptor)
        marker = json.loads(marker_contents.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read platform state ownership marker {marker_path}: {error}") from error
    expected_without_token = {
        "schema_version": 1,
        "ownership": "personal-agent-memory-platform-state",
        "state_dir": str(state_dir),
        "install_config_path": str(config_path),
    }
    marker_token = marker.get("token") if isinstance(marker, dict) else None
    if (
        not isinstance(marker_token, str)
        or len(marker_token) != 64
        or any(character not in "0123456789abcdef" for character in marker_token)
        or {key: marker.get(key) for key in expected_without_token} != expected_without_token
    ):
        raise SystemExit(f"platform state ownership marker is invalid: {marker_path}")
    if saved_token and marker_token != saved_token:
        raise SystemExit(f"platform state ownership marker does not match install metadata: {marker_path}")
    token = marker_token
    ownership = "installer-created-exclusive"
else:
    if not state_dir_preexisting:
        payload = {
            "schema_version": 1,
            "ownership": "personal-agent-memory-platform-state",
            "state_dir": str(state_dir),
            "install_config_path": str(config_path),
            "token": token,
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=".ownership.", dir=state_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, marker_path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        ownership = "installer-created-exclusive"
    else:
        token = ""
        if saved_token:
            ownership = "preexisting-unproven"
print(token)
print(ownership)
PY
)"
mapfile -t state_ownership_values <<<"$state_ownership"
state_ownership_token=${state_ownership_values[0]:-}
state_dir_ownership=${state_ownership_values[1]:-preexisting-unproven}

failed_stage="uninstall command installation"
uninstall_sha256="$($python_command - "$source_root/uninstall.sh" "$uninstall_command" \
  "$install_config_path" "$unit_path" "$uninstall_destination_state" \
  "${uninstall_destination_device:-}" "${uninstall_destination_inode:-}" \
  "$saved_uninstall_sha256" "$state_dir" "$codex_home" <<'PY'
import hashlib
import os
from pathlib import Path
import secrets
import shlex
import stat
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
config_path = Path(sys.argv[3])
unit_path = Path(sys.argv[4])
expected_state = sys.argv[5]
expected_identity = (
    (int(sys.argv[6]), int(sys.argv[7])) if expected_state == "owned" else None
)
expected_sha256 = sys.argv[8]
state_dir = Path(sys.argv[9])
codex_home = Path(sys.argv[10])
if not source.is_file():
    raise SystemExit(f"release is missing the uninstaller: {source}")
contents = source.read_text(encoding="utf-8")
first_line, remainder = contents.split("\n", 1)
contents = (
    f"{first_line}\n"
    f"PAM_INSTALL_CONFIG_PATH={shlex.quote(str(config_path))}\n"
    f"PAM_MANAGED_UNIT_PATH={shlex.quote(str(unit_path))}\n"
    f"PAM_MANAGED_UNINSTALL_PATH={shlex.quote(str(destination))}\n"
    f"PAM_MANAGED_STATE_DIR={shlex.quote(str(state_dir))}\n"
    f"PAM_MANAGED_CODEX_HOME={shlex.quote(str(codex_home))}\n"
    f"{remainder}"
)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.parent.resolve() != destination.parent:
    raise SystemExit(f"stable uninstaller parent is not a lexical directory: {destination.parent}")
directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
parent_descriptor = os.open(destination.parent, directory_flags)
entry_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)

def digest(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    value = hashlib.sha256()
    while chunk := os.read(descriptor, 8192):
        value.update(chunk)
    return value.hexdigest()

def verify_destination() -> None:
    try:
        current = os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        if expected_state == "absent":
            return
        raise SystemExit(f"recorded stable uninstaller is missing: {destination}")
    if expected_state != "owned":
        raise SystemExit(f"stable uninstaller path became occupied: {destination}")
    descriptor = os.open(destination.name, entry_flags, dir_fd=parent_descriptor)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or (current.st_dev, current.st_ino) != expected_identity
            or (info.st_dev, info.st_ino) != expected_identity
            or digest(descriptor) != expected_sha256
        ):
            raise SystemExit(f"stable uninstaller changed before update: {destination}")
    finally:
        os.close(descriptor)

temporary_name = f".personal-agent-memory-uninstall.{secrets.token_hex(12)}"
descriptor = os.open(
    temporary_name,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
    0o700,
    dir_fd=parent_descriptor,
)
try:
    payload = contents.encode("utf-8")
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary_name, 0o755, dir_fd=parent_descriptor, follow_symlinks=False)
    verify_destination()
    os.replace(
        temporary_name,
        destination.name,
        src_dir_fd=parent_descriptor,
        dst_dir_fd=parent_descriptor,
    )
finally:
    try:
        os.unlink(temporary_name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        pass
    os.close(parent_descriptor)
print(hashlib.sha256(payload).hexdigest())
PY
)"

stable_ledger_pending=false
[[ "$marketplace_action" == "keep" ]] || stable_ledger_pending=true
failed_stage="stable uninstaller ownership persistence"
persist_install_metadata "$saved_installed_version" "$marketplace_ownership" \
  "$source_root" "$marketplace_previous_source" "$stable_ledger_pending" \
  "$marketplace_update_from_source" "$uninstall_command" "$uninstall_sha256" \
  "$state_ownership_token" "$state_dir_ownership" "$uv_command" \
  "$saved_daemon_path" "$saved_unit_path" "$saved_unit_sha256" "$codex_home" \
  "$saved_plugin_id" "$saved_plugin_version" "$saved_plugin_registration_source" \
  "$saved_plugin_install_path" "$saved_plugin_tree_sha256" "$web_port"

failed_stage="user service shutdown"
if systemctl --user is-active --quiet "$service_name"; then
  systemctl --user stop "$service_name"
fi

failed_stage="Python tool installation"
"$uv_command" tool install --python "$python_command" --force --from "$source_root" \
  personal-agent-memory
daemon_command="${uv_bin_dir}/personal-agent-memory"
[[ -x "$daemon_command" ]]
installed_command_version="$($daemon_command --version)"
[[ "$installed_command_version" == "$target_version" ]]

failed_stage="Codex plugin installation"
if [[ "$marketplace_action" == "replace" ]]; then
  codex plugin marketplace remove personal-agent-memory --json >/dev/null
fi
if [[ "$marketplace_action" != "keep" ]]; then
  marketplace_result="$(codex plugin marketplace add "$source_root" --json)"
  "$python_command" - "$source_root" "$marketplace_result" <<'PY'
import json
from pathlib import Path
import sys

source_root = Path(sys.argv[1]).resolve()
result = json.loads(sys.argv[2])
if result.get("marketplaceName") != "personal-agent-memory":
    raise SystemExit("Codex registered an unexpected marketplace")
if Path(result.get("installedRoot", "")).resolve() != source_root:
    raise SystemExit("Codex marketplace did not converge to this release")
PY
fi
plugin_result="$(codex plugin add personal-agent-memory@personal-agent-memory --json)"
installed_plugin_path="$($python_command - "$target_version" "$plugin_result" <<'PY'
import json
from pathlib import Path
import sys

target_version = sys.argv[1]
result = json.loads(sys.argv[2])
if result.get("pluginId") != "personal-agent-memory@personal-agent-memory":
    raise SystemExit("Codex installed an unexpected plugin")
if result.get("version") != target_version:
    raise SystemExit("Codex plugin version does not match this release")
installed_path = result.get("installedPath")
if not isinstance(installed_path, str) or not Path(installed_path).is_dir():
    raise SystemExit("Codex did not report an installed plugin directory")
print(installed_path)
PY
)"
plugin_list="$(codex plugin list --json)"
managed_plugin_identity="$($python_command - "$target_version" "$installed_plugin_path" \
  "$source_root/plugins/personal-agent-memory" "$plugin_list" "$web_port" <<'PY'
import json
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile

MAX_ENTRIES = 4096
MAX_TOTAL_BYTES = 64 * 1024 * 1024


def plugin_tree_sha256(root: Path) -> str:
    try:
        root_info = os.lstat(root)
    except OSError as error:
        raise SystemExit(f"cannot inspect installed Codex plugin root: {error}") from error
    if not stat.S_ISDIR(root_info.st_mode):
        raise SystemExit("installed Codex plugin root is not a lexical directory")
    entries: list[tuple[bytes, Path, str]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as children:
                for child in children:
                    if len(entries) >= MAX_ENTRIES:
                        raise SystemExit("installed Codex plugin tree exceeds the entry limit")
                    path = Path(child.path)
                    relative = path.relative_to(root)
                    try:
                        relative_bytes = relative.as_posix().encode("utf-8")
                    except UnicodeEncodeError as error:
                        raise SystemExit(
                            "installed Codex plugin tree has a non-UTF-8 path"
                        ) from error
                    info = child.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise SystemExit(
                            f"installed Codex plugin tree contains a symlink: {relative}"
                        )
                    if stat.S_ISDIR(info.st_mode):
                        kind = "directory"
                        pending.append(path)
                    elif stat.S_ISREG(info.st_mode):
                        kind = "file"
                    else:
                        raise SystemExit(
                            f"installed Codex plugin tree contains a special entry: {relative}"
                        )
                    entries.append((relative_bytes, path, kind))
        except OSError as error:
            raise SystemExit(f"cannot scan installed Codex plugin tree: {error}") from error

    digest = hashlib.sha256(b"personal-agent-memory-plugin-tree-v1\0")
    total_bytes = 0
    for relative_bytes, path, kind in sorted(entries, key=lambda item: item[0]):
        digest.update(b"D" if kind == "directory" else b"F")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        if kind == "directory":
            continue
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as error:
            raise SystemExit(f"cannot read installed Codex plugin file {path}: {error}") from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise SystemExit(f"installed Codex plugin entry changed while hashing: {path}")
            total_bytes += info.st_size
            if total_bytes > MAX_TOTAL_BYTES:
                raise SystemExit("installed Codex plugin tree exceeds the byte limit")
            digest.update(info.st_size.to_bytes(8, "big"))
            remaining = info.st_size
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    raise SystemExit(f"installed Codex plugin file changed while hashing: {path}")
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise SystemExit(f"installed Codex plugin file changed while hashing: {path}")
        finally:
            os.close(descriptor)
    return digest.hexdigest()


target_version = sys.argv[1]
installed_path = Path(sys.argv[2]).resolve()
registration_source = Path(sys.argv[3]).resolve()
installed = json.loads(sys.argv[4]).get("installed", [])
port = sys.argv[5]
if not port.isdigit() or not 1 <= int(port) <= 65535:
    raise SystemExit("invalid listen port")
matches = [
    item
    for item in installed
    if item.get("pluginId") == "personal-agent-memory@personal-agent-memory"
]
if len(matches) != 1:
    raise SystemExit("Codex plugin registration did not converge to one installed record")
match = matches[0]
if match.get("version") != target_version or match.get("enabled") is not True:
    raise SystemExit("Codex plugin registration does not match this release")
source = match.get("source")
source_path = source.get("path") if isinstance(source, dict) else None
if (
    not isinstance(source_path, str)
    or Path(source_path).resolve() != registration_source
):
    raise SystemExit("Codex plugin list reported a different registration source")
manifest = json.loads(
    (installed_path / ".codex-plugin/plugin.json").read_text(encoding="utf-8")
)
if manifest.get("name") != "personal-agent-memory" or manifest.get("version") != target_version:
    raise SystemExit("installed Codex plugin manifest does not match this release")
mcp_path = installed_path / ".mcp.json"
mcp = json.loads(mcp_path.read_text(encoding="utf-8"))
server = mcp.get("mcpServers", {}).get("personal-agent-memory")
if not isinstance(server, dict):
    raise SystemExit("installed Codex plugin MCP configuration is invalid")
server["url"] = f"http://127.0.0.1:{int(port)}/mcp"
mcp_temporary_path: Path | None = None
try:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=mcp_path.parent, delete=False
    ) as stream:
        json.dump(mcp, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        mcp_temporary_path = Path(stream.name)
    os.replace(mcp_temporary_path, mcp_path)
    mcp_temporary_path = None
finally:
    if mcp_temporary_path is not None:
        mcp_temporary_path.unlink(missing_ok=True)
print(registration_source)
print(installed_path)
print(plugin_tree_sha256(installed_path))
PY
 )"
mapfile -t managed_plugin_identity_values <<<"$managed_plugin_identity"
managed_plugin_registration_source=${managed_plugin_identity_values[0]}
managed_plugin_install_path=${managed_plugin_identity_values[1]}
managed_plugin_tree_sha256=${managed_plugin_identity_values[2]}

failed_stage="managed plugin identity persistence"
persist_install_metadata "$target_version" "$marketplace_ownership" "$source_root" \
  "$marketplace_previous_source" false "" "$uninstall_command" "$uninstall_sha256" \
  "$state_ownership_token" "$state_dir_ownership" "$uv_command" "$daemon_command" \
  "$saved_unit_path" "$saved_unit_sha256" "$codex_home" \
  "personal-agent-memory@personal-agent-memory" "$target_version" \
  "$managed_plugin_registration_source" "$managed_plugin_install_path" \
  "$managed_plugin_tree_sha256" "$web_port"

failed_stage="user service installation"
daemon_unit_command="$(quote_unit_argument "$daemon_command")"
saved_daemon_unit_command="$(quote_unit_argument "${saved_daemon_path:-$daemon_command}")"
state_unit_argument="$(quote_unit_argument "$state_dir")"
library_unit_argument="$(quote_unit_argument "$library_root")"
unit_sha256="$($python_command - "$unit_path" "$daemon_unit_command" \
  "$state_unit_argument" "$library_unit_argument" "$saved_daemon_unit_command" \
  "$saved_unit_path" "$saved_unit_sha256" "$web_port" <<'PY'
import hashlib
import os
from pathlib import Path
import secrets
import stat
import sys

unit_path = Path(sys.argv[1])
legacy_daemon_command = sys.argv[5]
recorded_path = sys.argv[6]
recorded_sha256 = sys.argv[7]
port = sys.argv[8]
recorded = bool(recorded_path or recorded_sha256)
if recorded and (
    recorded_path != str(unit_path)
    or len(recorded_sha256) != 64
    or any(character not in "0123456789abcdef" for character in recorded_sha256)
):
    raise SystemExit(
        f"existing install metadata does not prove ownership of user service definition: {unit_path}"
    )
def unit_content(daemon_command: str) -> bytes:
    return f'''[Unit]
Description=Personal Agent Memory daemon
After=network.target

[Service]
Type=simple
ExecStart={daemon_command} serve --state-dir {sys.argv[3]} --library-root {sys.argv[4]} --host "127.0.0.1" --port "{port}"
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
'''.encode("utf-8")

payload = unit_content(sys.argv[2])
legacy_payload = unit_content(legacy_daemon_command)
if unit_path.parent.resolve() != unit_path.parent:
    raise SystemExit(f"user service directory is not a lexical directory: {unit_path.parent}")
directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
parent_descriptor = os.open(unit_path.parent, directory_flags)
entry_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)


def file_digest(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 8192):
        digest.update(chunk)
    return digest.hexdigest()


def current_state() -> tuple[str, tuple[int, int] | None]:
    try:
        entry = os.stat(unit_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return "absent", None
    try:
        descriptor = os.open(unit_path.name, entry_flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise SystemExit(f"cannot verify user service definition {unit_path}: {error}") from error
    try:
        info = os.fstat(descriptor)
        digest = file_digest(descriptor)
        if not stat.S_ISREG(info.st_mode) or (entry.st_dev, entry.st_ino) != (
            info.st_dev,
            info.st_ino,
        ):
            raise SystemExit(
                f"user service definition is not a regular installer-owned file: {unit_path}"
            )
        if recorded:
            if digest != recorded_sha256:
                raise SystemExit(
                    f"user service definition does not match install metadata: {unit_path}"
                )
        elif digest != hashlib.sha256(legacy_payload).hexdigest():
            raise SystemExit(
                f"user service definition already exists and is not installer-owned: {unit_path}"
            )
        return "owned", (info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)


expected_state, expected_identity = current_state()
temporary_name = f".personal-agent-memory.{secrets.token_hex(12)}"
descriptor = os.open(
    temporary_name,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
    0o600,
    dir_fd=parent_descriptor,
)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary_name, 0o644, dir_fd=parent_descriptor, follow_symlinks=False)
    current, identity = current_state()
    if current != expected_state or identity != expected_identity:
        raise SystemExit(f"user service definition changed before update: {unit_path}")
    os.replace(
        temporary_name,
        unit_path.name,
        src_dir_fd=parent_descriptor,
        dst_dir_fd=parent_descriptor,
    )
finally:
    try:
        os.unlink(temporary_name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        pass
    os.close(parent_descriptor)
print(hashlib.sha256(payload).hexdigest())
PY
)"

failed_stage="user service ownership persistence"
persist_install_metadata "$target_version" "$marketplace_ownership" "$source_root" \
  "$marketplace_previous_source" false "" "$uninstall_command" "$uninstall_sha256" \
  "$state_ownership_token" "$state_dir_ownership" "$uv_command" "$daemon_command" \
  "$unit_path" "$unit_sha256" "$codex_home" \
  "personal-agent-memory@personal-agent-memory" "$target_version" \
  "$managed_plugin_registration_source" "$managed_plugin_install_path" \
  "$managed_plugin_tree_sha256" "$web_port"

failed_stage="user service reload"
systemctl --user daemon-reload

failed_stage="user service startup"
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

failed_stage="install metadata persistence"
persist_install_metadata "$target_version" "$marketplace_ownership" "$source_root" \
  "$marketplace_previous_source" false "" "$uninstall_command" "$uninstall_sha256" \
  "$state_ownership_token" "$state_dir_ownership" "$uv_command" "$daemon_command" \
  "$unit_path" "$unit_sha256" "$codex_home" \
  "personal-agent-memory@personal-agent-memory" "$target_version" \
  "$managed_plugin_registration_source" "$managed_plugin_install_path" \
  "$managed_plugin_tree_sha256" "$web_port"

trap - ERR
cat <<EOF
Personal Agent Memory installation complete.
Installed version: ${target_version}
Service status: ${service_status}
Web: ${web_url}
API key file: ${state_dir}/api-key
Uninstall: ${uninstall_command}
Purge:     ${uninstall_command} --purge

Start:   systemctl --user start ${service_name}
Stop:    systemctl --user stop ${service_name}
Restart: systemctl --user restart ${service_name}
Status:  systemctl --user status ${service_name}
Logs:    journalctl --user -u ${service_name} --no-pager
EOF
