#!/usr/bin/env bash
set -euo pipefail

service_name="personal-agent-memory.service"
config_home="${XDG_CONFIG_HOME:-${HOME}/.config}"
install_config_dir="${config_home}/personal-agent-memory"
install_config_path="${PAM_INSTALL_CONFIG_PATH:-${install_config_dir}/install.json}"
install_config_dir="${install_config_path%/*}"
config_root="${install_config_dir%/*}"
unit_path="${config_root}/systemd/user/${service_name}"
purge=false
failed_stage="initialization"

usage() {
  cat <<'EOF'
Usage: personal-agent-memory-uninstall [--purge]

  --purge     Also remove verified platform state and installation metadata.
  -h, --help  Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge)
      purge=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown uninstaller option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

on_error() {
  status=$?
  trap - ERR
  printf 'Uninstallation failed during %s.\n' "$failed_stage" >&2
  printf 'Service status: systemctl --user status %s\n' "$service_name" >&2
  exit "$status"
}
trap on_error ERR

if [[ -n "${XDG_CONFIG_HOME:-}" && "${XDG_CONFIG_HOME}" != /* ]]; then
  printf 'XDG_CONFIG_HOME must be an absolute path when set.\n' >&2
  exit 1
fi
if [[ "$install_config_path" != /* || "$install_config_path" == *$'\n'* || "$install_config_path" == *$'\r'* ]]; then
  printf 'Install metadata path must be absolute and contain no line breaks.\n' >&2
  exit 1
fi

managed_unit_hint=${PAM_MANAGED_UNIT_PATH:-}
managed_uninstall_hint=${PAM_MANAGED_UNINSTALL_PATH:-}
managed_state_hint=${PAM_MANAGED_STATE_DIR:-}
managed_codex_home_hint=${PAM_MANAGED_CODEX_HOME:-}
for early_path in "$managed_unit_hint" "$managed_uninstall_hint" \
  "$managed_state_hint" "$managed_codex_home_hint"; do
  if [[ -n "$early_path" && ( "$early_path" != /* || "$early_path" == *$'\n'* || "$early_path" == *$'\r'* ) ]]; then
    printf 'Embedded managed paths must be absolute and contain no line breaks.\n' >&2
    exit 1
  fi
done

bind_early_entry() {
  local path=$1
  local grandparent_fd_name=$2
  local parent_fd_name=$3
  local entry_fd_name=$4
  local status_name=$5
  local parent=${path%/*}
  local grandparent=${parent%/*}
  local grandparent_fd
  local parent_fd
  local entry_fd
  local status=missing

  if [[ -z "$path" || "$parent" == "$path" || "$grandparent" == "$parent" ]]; then
    exec {grandparent_fd}</dev/null
    exec {parent_fd}</dev/null
    exec {entry_fd}</dev/null
    status=unavailable
  elif [[ ! -d "$grandparent" || -L "$grandparent" || ! -d "$parent" || -L "$parent" ]]; then
    exec {grandparent_fd}</dev/null
    exec {parent_fd}</dev/null
    exec {entry_fd}</dev/null
    status=unavailable
  else
    exec {grandparent_fd}<"$grandparent"
    exec {parent_fd}<"$parent"
    if [[ -f "$path" && ! -L "$path" ]]; then
      if exec {entry_fd}<>"$path"; then
        status=present
      else
        exec {entry_fd}</dev/null
        status=unavailable
      fi
    elif [[ -e "$path" || -L "$path" ]]; then
      exec {entry_fd}</dev/null
      status=unavailable
    else
      exec {entry_fd}</dev/null
    fi
  fi
  printf -v "$grandparent_fd_name" '%s' "$grandparent_fd"
  printf -v "$parent_fd_name" '%s' "$parent_fd"
  printf -v "$entry_fd_name" '%s' "$entry_fd"
  printf -v "$status_name" '%s' "$status"
}

bind_early_entry "$install_config_path" early_metadata_grandparent_fd \
  early_metadata_parent_fd early_metadata_fd early_metadata_status
bind_early_entry "$managed_unit_hint" early_unit_grandparent_fd early_unit_parent_fd \
  early_unit_fd early_unit_status
bind_early_entry "$managed_uninstall_hint" early_uninstall_grandparent_fd \
  early_uninstall_parent_fd early_uninstall_fd early_uninstall_status

bind_early_directory() {
  local path=$1
  local grandparent_fd_name=$2
  local parent_fd_name=$3
  local directory_fd_name=$4
  local status_name=$5
  local parent=${path%/*}
  local grandparent=${parent%/*}
  local grandparent_fd
  local parent_fd
  local directory_fd
  local status=missing

  [[ -n "$grandparent" ]] || grandparent=/
  if [[ -z "$path" || "$parent" == "$path" || ! -d "$grandparent" || \
    -L "$grandparent" || ! -d "$parent" || -L "$parent" ]]; then
    exec {grandparent_fd}</dev/null
    exec {parent_fd}</dev/null
    exec {directory_fd}</dev/null
    status=unavailable
  else
    exec {grandparent_fd}<"$grandparent"
    exec {parent_fd}<"$parent"
    if [[ -d "$path" && ! -L "$path" ]]; then
      if exec {directory_fd}<"$path"; then
        status=present
      else
        exec {directory_fd}</dev/null
        status=unavailable
      fi
    elif [[ -e "$path" || -L "$path" ]]; then
      exec {directory_fd}</dev/null
      status=unavailable
    else
      exec {directory_fd}</dev/null
    fi
  fi
  printf -v "$grandparent_fd_name" '%s' "$grandparent_fd"
  printf -v "$parent_fd_name" '%s' "$parent_fd"
  printf -v "$directory_fd_name" '%s' "$directory_fd"
  printf -v "$status_name" '%s' "$status"
}

bind_early_directory "$managed_state_hint" early_state_grandparent_fd \
  early_state_parent_fd early_state_dir_fd early_state_status
bind_early_directory "$managed_codex_home_hint" early_codex_grandparent_fd \
  early_codex_parent_fd early_codex_home_fd early_codex_home_status

if [[ "$early_metadata_status" == "missing" ]]; then
  trap - ERR
  printf 'No Personal Agent Memory installation metadata was found; nothing was removed.\n'
  exit 0
elif [[ "$early_metadata_status" != "present" ]]; then
  printf 'Install metadata could not be safely bound before dependency checks.\n' >&2
  exit 1
fi
if [[ -z "$managed_state_hint" || -z "$managed_codex_home_hint" ]]; then
  printf 'This stable uninstaller predates trusted state and Codex path hints; reinstall or upgrade Personal Agent Memory before uninstalling.\n' >&2
  exit 1
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
[[ -n "$python_command" ]] || {
  printf 'Python 3.11, 3.12, or 3.13 is required to uninstall Personal Agent Memory safely.\n' >&2
  exit 1
}

failed_stage="install metadata validation"
metadata="$($python_command - "$install_config_path" \
  3<&"$early_metadata_fd" 4<&"$early_metadata_parent_fd" \
  5<&"$early_metadata_grandparent_fd" <<'PY'
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

MAX_BYTES = 64 * 1024
path = Path(sys.argv[1])
descriptor = 3
parent_descriptor = 4
grandparent_descriptor = 5
try:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"install metadata must be a regular file: {path}")
    parent_info = os.fstat(parent_descriptor)
    grandparent_info = os.fstat(grandparent_descriptor)
    current_grandparent = os.stat(path.parent.parent, follow_symlinks=False)
    current_parent = os.stat(
        path.parent.name,
        dir_fd=grandparent_descriptor,
        follow_symlinks=False,
    )
    current_entry = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(grandparent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or (current_grandparent.st_dev, current_grandparent.st_ino)
        != (grandparent_info.st_dev, grandparent_info.st_ino)
        or not stat.S_ISDIR(current_parent.st_mode)
        or (current_parent.st_dev, current_parent.st_ino)
        != (parent_info.st_dev, parent_info.st_ino)
        or (current_entry.st_dev, current_entry.st_ino) != (info.st_dev, info.st_ino)
    ):
        raise SystemExit(f"install metadata changed during initialization: {path}")
    if info.st_size > MAX_BYTES:
        raise SystemExit(f"install metadata exceeds the 65536-byte limit: {path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    contents = bytearray()
    while len(contents) <= MAX_BYTES:
        chunk = os.read(descriptor, min(8192, MAX_BYTES + 1 - len(contents)))
        if not chunk:
            break
        contents.extend(chunk)
    if len(contents) > MAX_BYTES:
        raise SystemExit(f"install metadata exceeds the 65536-byte limit: {path}")
    data = json.loads(contents.decode("utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"cannot read install metadata {path}: {error}") from error

if (
    not isinstance(data, dict)
    or type(data.get("schema_version")) is not int
    or data.get("schema_version") != 1
):
    raise SystemExit(f"install metadata is invalid or unsupported: {path}")

def absolute_field(name: str, *, resolve: bool = True) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n"):
        raise SystemExit(f"install metadata has an invalid {name}: {path}")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise SystemExit(f"install metadata has a non-absolute {name}: {path}")
    return str(candidate.resolve()) if resolve else str(candidate)

state_dir = absolute_field("state_dir", resolve=False)
library_root = absolute_field("library_root")
if data.get("library_root_ownership") != "user-content-never-delete":
    raise SystemExit(f"install metadata does not protect the memory library: {path}")

ownership = data.get("marketplace_ownership", "legacy-unknown")
if ownership not in {"installer-managed", "preexisting", "legacy-unowned", "legacy-unknown"}:
    raise SystemExit(f"install metadata has invalid marketplace ownership: {path}")
source = data.get("marketplace_source", "")
if source:
    if not isinstance(source, str) or any(c in source for c in "\r\n") or not Path(source).is_absolute():
        raise SystemExit(f"install metadata has invalid marketplace source: {path}")
    source = str(Path(source).resolve())

uninstall_path = data.get("uninstall_path", "")
uninstall_sha256 = data.get("uninstall_sha256", "")
if uninstall_path:
    if (
        data.get("uninstall_ownership") != "installer-managed"
        or not isinstance(uninstall_path, str)
        or any(c in uninstall_path for c in "\r\n")
        or not Path(uninstall_path).is_absolute()
        or not isinstance(uninstall_sha256, str)
        or len(uninstall_sha256) != 64
        or any(c not in "0123456789abcdef" for c in uninstall_sha256)
    ):
        raise SystemExit(f"install metadata has invalid uninstaller ownership: {path}")
    uninstall_path = str(Path(uninstall_path))

state_ownership_token = data.get("state_ownership_token", "")
state_dir_ownership = data.get("state_dir_ownership", "legacy-managed-contents")
if state_dir_ownership not in {
    "installer-created-exclusive",
    "legacy-managed-contents",
    "preexisting-unproven",
}:
    raise SystemExit(f"install metadata has invalid state directory ownership: {path}")
if state_ownership_token and (
    not isinstance(state_ownership_token, str)
    or len(state_ownership_token) != 64
    or any(c not in "0123456789abcdef" for c in state_ownership_token)
):
    raise SystemExit(f"install metadata has invalid state ownership: {path}")
if state_dir_ownership == "installer-created-exclusive" and not state_ownership_token:
    raise SystemExit(f"install metadata is missing state ownership proof: {path}")

def optional_absolute_field(name: str, *, resolve: bool = True) -> str:
    value = data.get(name, "")
    if not value:
        return ""
    if not isinstance(value, str) or any(c in value for c in "\r\n") or not Path(value).is_absolute():
        raise SystemExit(f"install metadata has an invalid {name}: {path}")
    return str(Path(value).resolve()) if resolve else value

uv_path = optional_absolute_field("uv_path")
daemon_path = optional_absolute_field("daemon_path", resolve=False)
unit_path = optional_absolute_field("unit_path")
unit_sha256 = data.get("unit_sha256", "")
if unit_sha256 and (
    not isinstance(unit_sha256, str)
    or len(unit_sha256) != 64
    or any(c not in "0123456789abcdef" for c in unit_sha256)
    or not unit_path
):
    raise SystemExit(f"install metadata has invalid unit ownership: {path}")
codex_home = data.get("codex_home", os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
if (
    not isinstance(codex_home, str)
    or not codex_home
    or any(c in codex_home for c in "\r\n")
    or not Path(codex_home).is_absolute()
):
    raise SystemExit(f"install metadata has an invalid codex_home: {path}")
codex_home = str(Path(codex_home))
plugin_id = data.get("plugin_id", "")
plugin_version = data.get("plugin_version", "")
plugin_registration_source = data.get("plugin_registration_source", "")
plugin_install_path = data.get("plugin_install_path", "")
plugin_tree_sha256 = data.get("plugin_tree_sha256", "")
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
        or any(c in plugin_version for c in "\r\n")
        or not isinstance(plugin_registration_source, str)
        or any(c in plugin_registration_source for c in "\r\n")
        or not Path(plugin_registration_source).is_absolute()
        or not isinstance(plugin_install_path, str)
        or any(c in plugin_install_path for c in "\r\n")
        or not Path(plugin_install_path).is_absolute()
    ):
        raise SystemExit(f"install metadata has an invalid plugin identity: {path}")
    if plugin_tree_sha256 and (
        not isinstance(plugin_tree_sha256, str)
        or len(plugin_tree_sha256) != 64
        or any(c not in "0123456789abcdef" for c in plugin_tree_sha256)
    ):
        raise SystemExit(f"install metadata has an invalid plugin tree digest: {path}")
elif plugin_identity != ("", "", "", "") or plugin_tree_sha256:
    raise SystemExit(f"install metadata has an incomplete plugin identity: {path}")
print("present")
print(state_dir)
print(library_root)
print(ownership)
print(source)
print(state_dir_ownership)
print(uninstall_path)
print(uninstall_sha256)
print(state_ownership_token)
print(uv_path)
print(daemon_path)
print(unit_path)
print(unit_sha256)
print(codex_home)
print(plugin_id)
print(plugin_version)
print(plugin_registration_source)
print(plugin_install_path)
print(plugin_tree_sha256)
print(info.st_dev)
print(info.st_ino)
print(hashlib.sha256(contents).hexdigest())
PY
)"
mapfile -t metadata_values <<<"$metadata"
metadata_status=${metadata_values[0]}

if [[ "$metadata_status" == "missing" ]]; then
  trap - ERR
  printf 'No Personal Agent Memory installation metadata was found; nothing was removed.\n'
  exit 0
fi

state_dir="${HOME}/.local/share/personal-agent-memory"
library_root="${HOME}/memory-libraries"
marketplace_ownership="legacy-unknown"
marketplace_source=""
state_dir_ownership="preexisting-unproven"
owned_uninstall_path=""
owned_uninstall_sha256=""
state_ownership_token=""
managed_uv_path=""
managed_daemon_path=""
managed_unit_path=""
managed_unit_sha256=""
managed_codex_home="${CODEX_HOME:-${HOME}/.codex}"
if [[ "$metadata_status" == "present" ]]; then
  state_dir=${metadata_values[1]}
  library_root=${metadata_values[2]}
  marketplace_ownership=${metadata_values[3]}
  marketplace_source=${metadata_values[4]:-}
  state_dir_ownership=${metadata_values[5]:-preexisting-unproven}
  owned_uninstall_path=${metadata_values[6]:-}
  owned_uninstall_sha256=${metadata_values[7]:-}
  state_ownership_token=${metadata_values[8]:-}
  managed_uv_path=${metadata_values[9]:-}
  managed_daemon_path=${metadata_values[10]:-}
  managed_unit_path=${metadata_values[11]:-}
  managed_unit_sha256=${metadata_values[12]:-}
  managed_codex_home=${metadata_values[13]}
  managed_plugin_id=${metadata_values[14]:-}
  managed_plugin_version=${metadata_values[15]:-}
  managed_plugin_registration_source=${metadata_values[16]:-}
  managed_plugin_install_path=${metadata_values[17]:-}
  managed_plugin_tree_sha256=${metadata_values[18]:-}
  metadata_device=${metadata_values[19]}
  metadata_inode=${metadata_values[20]}
  metadata_sha256=${metadata_values[21]}
fi
if [[ "$state_dir" != "$managed_state_hint" || \
  "$managed_codex_home" != "$managed_codex_home_hint" ]]; then
  printf 'Install metadata paths do not match the trusted stable uninstaller hints.\n' >&2
  exit 1
fi

coproc OWNERSHIP_GUARD {
  "$python_command" - "$install_config_path" "$metadata_device" "$metadata_inode" \
    "$metadata_sha256" "$managed_unit_path" "$unit_path" "$managed_unit_sha256" \
    "$owned_uninstall_path" "$owned_uninstall_sha256" "$managed_unit_hint" \
    "$managed_uninstall_hint" "$early_unit_status" "$early_uninstall_status" \
    "$managed_codex_home" "$early_codex_home_status" "$state_dir" \
    "$early_state_status" "$state_dir_ownership" "$state_ownership_token" \
    "$library_root" "$purge" "$managed_plugin_id" "$managed_plugin_version" \
    "$managed_plugin_registration_source" "$managed_plugin_install_path" \
    "$managed_plugin_tree_sha256" \
    3<&0 4<&"$early_metadata_grandparent_fd" 5<&"$early_metadata_parent_fd" \
    6<&"$early_metadata_fd" 7<&"$early_unit_grandparent_fd" \
    8<&"$early_unit_parent_fd" 9<&"$early_unit_fd" \
    10<&"$early_uninstall_grandparent_fd" 11<&"$early_uninstall_parent_fd" \
    12<&"$early_uninstall_fd" 13<&"$early_codex_grandparent_fd" \
    14<&"$early_codex_parent_fd" 15<&"$early_codex_home_fd" \
    16<&"$early_state_grandparent_fd" 17<&"$early_state_parent_fd" \
    18<&"$early_state_dir_fd" <<'PY'
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys


def digest(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    value = hashlib.sha256()
    while chunk := os.read(descriptor, 8192):
        value.update(chunk)
    return value.hexdigest()


def current_identity(parent_descriptor: int, name: str) -> tuple[int, int] | None:
    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino


def grandparent_is_current(path: Path, grandparent_descriptor: int) -> bool:
    try:
        expected = os.fstat(grandparent_descriptor)
        current = os.stat(path.parent.parent, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(expected.st_mode)
        and stat.S_ISDIR(current.st_mode)
        and (expected.st_dev, expected.st_ino) == (current.st_dev, current.st_ino)
    )


def parent_is_current(
    path: Path,
    grandparent_descriptor: int,
    parent_descriptor: int,
) -> bool:
    try:
        grandparent_info = os.fstat(grandparent_descriptor)
        parent_info = os.fstat(parent_descriptor)
        current_parent = os.stat(
            path.parent.name,
            dir_fd=grandparent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return (
        stat.S_ISDIR(grandparent_info.st_mode)
        and stat.S_ISDIR(parent_info.st_mode)
        and grandparent_is_current(path, grandparent_descriptor)
        and stat.S_ISDIR(current_parent.st_mode)
        and (parent_info.st_dev, parent_info.st_ino)
        == (current_parent.st_dev, current_parent.st_ino)
    )


def entry_is_current(
    path: Path,
    grandparent_descriptor: int,
    parent_descriptor: int,
    expected_identity: tuple[int, int] | None,
    descriptor: int | None,
    expected_sha256: str,
) -> bool:
    if not parent_is_current(path, grandparent_descriptor, parent_descriptor):
        return False
    current = current_identity(parent_descriptor, path.name)
    if expected_identity is None:
        return current is None
    return (
        current == expected_identity
        and descriptor is not None
        and digest(descriptor) == expected_sha256
    )


def guarded_unlink(
    path: Path,
    grandparent_descriptor: int,
    parent_descriptor: int,
    expected_identity: tuple[int, int] | None,
    descriptor: int | None,
    expected_sha256: str,
) -> str:
    if not parent_is_current(path, grandparent_descriptor, parent_descriptor):
        return "replaced"
    current = current_identity(parent_descriptor, path.name)
    if current is None:
        return "missing"
    if expected_identity is None or current != expected_identity:
        return "replaced"
    if descriptor is None or digest(descriptor) != expected_sha256:
        return "replaced"
    os.unlink(path.name, dir_fd=parent_descriptor)
    return "removed"


metadata_path = Path(sys.argv[1])
metadata_expected = (int(sys.argv[2]), int(sys.argv[3]))
metadata_grandparent = 4
metadata_parent = 5
metadata_descriptor = 6
metadata_info = os.fstat(metadata_descriptor)
metadata_current = current_identity(metadata_parent, metadata_path.name)
if (
    not stat.S_ISREG(metadata_info.st_mode)
    or not parent_is_current(metadata_path, metadata_grandparent, metadata_parent)
    or (metadata_info.st_dev, metadata_info.st_ino) != metadata_expected
    or metadata_current != metadata_expected
):
    raise SystemExit("install metadata changed before ownership guard initialization")
if digest(metadata_descriptor) != sys.argv[4]:
    raise SystemExit("install metadata contents changed before ownership guard initialization")
metadata_deleted = False
metadata_parent_deleted = False

managed_unit_path = sys.argv[5]
expected_unit_path = sys.argv[6]
expected_unit_sha256 = sys.argv[7]
unit_hint = sys.argv[10]
early_unit_status = sys.argv[12]
unit_parent = None
unit_descriptor = None
unit_identity = None
unit_status = "unowned"
unit_path = Path(managed_unit_path) if managed_unit_path else None
if (
    managed_unit_path
    and managed_unit_path == expected_unit_path
    and managed_unit_path == unit_hint
):
    if early_unit_status == "missing":
        unit_grandparent = 7
        unit_parent = 8
        if parent_is_current(unit_path, unit_grandparent, unit_parent):
            unit_status = (
                "missing"
                if current_identity(unit_parent, unit_path.name) is None
                else "replaced"
            )
    elif early_unit_status == "present":
        unit_grandparent = 7
        unit_parent = 8
        unit_descriptor = 9
        unit_info = os.fstat(unit_descriptor)
        unit_identity = (unit_info.st_dev, unit_info.st_ino)
        unit_current = current_identity(unit_parent, unit_path.name)
        if unit_current != unit_identity:
            unit_status = "replaced"
        elif (
            stat.S_ISREG(unit_info.st_mode)
            and parent_is_current(unit_path, unit_grandparent, unit_parent)
            and digest(unit_descriptor) == expected_unit_sha256
        ):
            unit_status = "owned"

uninstall_path_value = sys.argv[8]
expected_uninstall_sha256 = sys.argv[9]
uninstall_hint = sys.argv[11]
early_uninstall_status = sys.argv[13]
uninstall_parent = None
uninstall_descriptor = None
uninstall_identity = None
uninstall_status = "unowned"
uninstall_path = Path(uninstall_path_value) if uninstall_path_value else None
if uninstall_path_value and uninstall_path_value == uninstall_hint:
    if early_uninstall_status == "missing":
        uninstall_grandparent = 10
        uninstall_parent = 11
        if parent_is_current(uninstall_path, uninstall_grandparent, uninstall_parent):
            uninstall_status = (
                "missing"
                if current_identity(uninstall_parent, uninstall_path.name) is None
                else "replaced"
            )
    elif early_uninstall_status == "present":
        uninstall_grandparent = 10
        uninstall_parent = 11
        uninstall_descriptor = 12
        uninstall_info = os.fstat(uninstall_descriptor)
        uninstall_identity = (uninstall_info.st_dev, uninstall_info.st_ino)
        uninstall_current = current_identity(uninstall_parent, uninstall_path.name)
        if uninstall_current != uninstall_identity:
            uninstall_status = "replaced"
        elif (
            stat.S_ISREG(uninstall_info.st_mode)
            and parent_is_current(
                uninstall_path,
                uninstall_grandparent,
                uninstall_parent,
            )
            and digest(uninstall_descriptor) == expected_uninstall_sha256
        ):
            uninstall_status = "owned"

codex_home_path = Path(sys.argv[14])
early_codex_home_status = sys.argv[15]
codex_home_grandparent = 13
codex_home_parent = 14
codex_home_descriptor = 15 if early_codex_home_status == "present" else None
codex_home_identity = None


def codex_home_is_current() -> bool:
    try:
        lexical = codex_home_path.resolve() == codex_home_path
    except OSError:
        return False
    if not lexical or not parent_is_current(
        codex_home_path,
        codex_home_grandparent,
        codex_home_parent,
    ):
        return False
    current = current_identity(codex_home_parent, codex_home_path.name)
    if early_codex_home_status == "missing":
        return current is None
    if codex_home_descriptor is None or codex_home_identity is None:
        return False
    try:
        directory_info = os.fstat(codex_home_descriptor)
    except OSError:
        return False
    return (
        stat.S_ISDIR(directory_info.st_mode)
        and current == codex_home_identity
        and (directory_info.st_dev, directory_info.st_ino) == codex_home_identity
    )


if early_codex_home_status not in {"present", "missing"}:
    raise SystemExit("recorded Codex home could not be safely bound")
if early_codex_home_status == "present":
    codex_home_info = os.fstat(codex_home_descriptor)
    codex_home_identity = (codex_home_info.st_dev, codex_home_info.st_ino)
if not codex_home_is_current():
    raise SystemExit("recorded Codex home changed before ownership guard initialization")

state_path = Path(sys.argv[16])
early_state_status = sys.argv[17]
state_ownership = sys.argv[18]
state_ownership_token = sys.argv[19]
library_root = Path(sys.argv[20]).resolve()
purge_requested = sys.argv[21] == "true"
state_grandparent = 16
state_parent = 17
state_descriptor = 18 if early_state_status == "present" else None
state_identity = None
state_status = early_state_status


def state_dir_is_current() -> bool:
    if not parent_is_current(state_path, state_grandparent, state_parent):
        return False
    current = current_identity(state_parent, state_path.name)
    if state_status == "missing":
        return current is None
    if state_status != "present" or state_descriptor is None or state_identity is None:
        return False
    try:
        state_info = os.fstat(state_descriptor)
        lexical = state_path.resolve() == state_path
    except OSError:
        return False
    return (
        lexical
        and stat.S_ISDIR(state_info.st_mode)
        and current == state_identity
        and (state_info.st_dev, state_info.st_ino) == state_identity
    )


if early_state_status not in {"present", "missing"}:
    raise SystemExit(
        f"recorded path is not a directory and could not be safely bound: {state_path}"
    )
if early_state_status == "present":
    state_info = os.fstat(state_descriptor)
    state_identity = (state_info.st_dev, state_info.st_ino)
if not state_dir_is_current():
    raise SystemExit("platform state directory changed before ownership guard initialization")

if purge_requested:
    home = Path.home().resolve()
    config_dir = metadata_path.parent.resolve()
    state_lexical = Path(os.path.normpath(str(state_path)))
    for protected in (home, Path("/").resolve(), library_root, config_dir):
        if (
            state_lexical == protected
            or protected.is_relative_to(state_lexical)
            or state_lexical.is_relative_to(library_root)
        ):
            raise SystemExit(
                f"refusing to purge unsafe or overlapping platform state path: {state_path}"
            )


def remove_state_entry(name: str) -> None:
    info = os.stat(name, dir_fd=state_descriptor, follow_symlinks=False)
    if stat.S_ISDIR(info.st_mode):
        shutil.rmtree(name, dir_fd=state_descriptor)
    else:
        os.unlink(name, dir_fd=state_descriptor)


def purge_state() -> str:
    global state_identity, state_status
    if state_status == "missing":
        return "removed"
    if not state_dir_is_current() or state_descriptor is None:
        return "replaced"
    try:
        if state_ownership == "installer-created-exclusive":
            marker_flags = (
                os.O_RDONLY
                | os.O_NONBLOCK
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0)
            )
            marker_descriptor = os.open(
                ".personal-agent-memory-owned.json",
                marker_flags,
                dir_fd=state_descriptor,
            )
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
            expected = {
                "schema_version": 1,
                "ownership": "personal-agent-memory-platform-state",
                "state_dir": str(state_path),
                "install_config_path": str(metadata_path.resolve()),
                "token": state_ownership_token,
            }
            if marker != expected:
                raise OSError("ownership marker does not match install metadata")
            for entry in list(os.scandir(state_descriptor)):
                remove_state_entry(entry.name)
        elif state_ownership == "legacy-managed-contents":
            expected_types = {
                "api-key": "file",
                "platform.sqlite3": "file",
                "platform.sqlite3-journal": "file",
                "platform.sqlite3-wal": "file",
                "platform.sqlite3-shm": "file",
                "sensitive-dedupe-key": "file",
                "tombstone-keys.json": "file",
                ".personal-agent-memory-owned.json": "file",
                "graphs": "directory",
                "git": "directory",
                "locks": "directory",
            }
            for name, expected_type in expected_types.items():
                try:
                    entry_info = os.stat(
                        name,
                        dir_fd=state_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if expected_type == "directory" and stat.S_ISDIR(entry_info.st_mode):
                    shutil.rmtree(name, dir_fd=state_descriptor)
                elif expected_type == "file" and stat.S_ISREG(entry_info.st_mode):
                    os.unlink(name, dir_fd=state_descriptor)
        else:
            raise OSError("installer ownership of platform state could not be proven")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot validate platform state ownership for {state_path}: {error}"
        ) from error

    remaining = sorted(entry.name for entry in os.scandir(state_descriptor))
    if remaining:
        return "preserved:" + ",".join(remaining)
    if not state_dir_is_current():
        return "replaced"
    try:
        os.rmdir(state_path.name, dir_fd=state_parent)
    except OSError as error:
        if error.errno in {errno.ENOTEMPTY, errno.EEXIST}:
            return "preserved:" + ",".join(
                sorted(entry.name for entry in os.scandir(state_descriptor))
            )
        raise
    state_status = "missing"
    state_identity = None
    return "removed"

def managed_entry_is_current(
    status: str,
    path: Path | None,
    grandparent_descriptor: int | None,
    parent_descriptor: int | None,
    identity: tuple[int, int] | None,
    descriptor: int | None,
    expected_sha256: str,
) -> bool:
    if status == "unowned":
        return True
    if path is None or grandparent_descriptor is None or parent_descriptor is None:
        return False
    return entry_is_current(
        path,
        grandparent_descriptor,
        parent_descriptor,
        identity,
        descriptor,
        expected_sha256,
    )


def check_all() -> str:
    if metadata_parent_deleted:
        if (
            not grandparent_is_current(metadata_path, metadata_grandparent)
            or current_identity(metadata_grandparent, metadata_path.parent.name) is not None
        ):
            return "metadata"
    elif metadata_deleted:
        if not entry_is_current(
            metadata_path,
            metadata_grandparent,
            metadata_parent,
            None,
            None,
            "",
        ):
            return "metadata"
    elif not entry_is_current(
        metadata_path,
        metadata_grandparent,
        metadata_parent,
        metadata_expected,
        metadata_descriptor,
        sys.argv[4],
    ):
        return "metadata"
    if not managed_entry_is_current(
        unit_status,
        unit_path,
        unit_grandparent if unit_parent is not None else None,
        unit_parent,
        unit_identity,
        unit_descriptor,
        expected_unit_sha256,
    ):
        return "unit"
    if not managed_entry_is_current(
        uninstall_status,
        uninstall_path,
        uninstall_grandparent if uninstall_parent is not None else None,
        uninstall_parent,
        uninstall_identity,
        uninstall_descriptor,
        expected_uninstall_sha256,
    ):
        return "uninstaller"
    if not codex_home_is_current():
        return "codex-home"
    if not state_dir_is_current():
        return "state-dir"
    return "same"


print(
    f"ready:{unit_status}:{uninstall_status}:{early_codex_home_status}:{state_status}",
    flush=True,
)
try:
    with os.fdopen(3, encoding="utf-8") as commands:
        for raw_command in commands:
            command = raw_command.rstrip("\n")
            if command == "check-all":
                print(f"all:{check_all()}", flush=True)
            elif command == "delete-unit":
                if unit_status == "unowned" or unit_parent is None:
                    result = "unowned"
                else:
                    result = guarded_unlink(
                        unit_path,
                        unit_grandparent,
                        unit_parent,
                        unit_identity,
                        unit_descriptor,
                        expected_unit_sha256,
                    )
                    if result in {"removed", "missing"}:
                        unit_status = "missing"
                        unit_identity = None
                print(f"unit:{result}", flush=True)
            elif command == "check-metadata":
                result = "same" if check_all() == "same" else "replaced"
                print(f"metadata:{result}", flush=True)
            elif command == "delete-metadata":
                result = guarded_unlink(
                    metadata_path,
                    metadata_grandparent,
                    metadata_parent,
                    metadata_expected,
                    metadata_descriptor,
                    sys.argv[4],
                )
                if result == "removed":
                    metadata_deleted = True
                print(f"metadata:{result}", flush=True)
            elif command == "delete-metadata-parent":
                if metadata_parent_deleted:
                    result = "missing"
                elif not metadata_deleted or not parent_is_current(
                    metadata_path,
                    metadata_grandparent,
                    metadata_parent,
                ):
                    result = "replaced"
                else:
                    try:
                        os.rmdir(
                            metadata_path.parent.name,
                            dir_fd=metadata_grandparent,
                        )
                    except OSError as error:
                        result = (
                            "not-empty"
                            if error.errno in {errno.ENOTEMPTY, errno.EEXIST}
                            else "replaced"
                        )
                    else:
                        metadata_parent_deleted = True
                        result = "removed"
                print(f"metadata-parent:{result}", flush=True)
            elif command == "delete-uninstaller":
                if uninstall_status == "unowned" or uninstall_parent is None:
                    result = "unowned"
                else:
                    result = guarded_unlink(
                        uninstall_path,
                        uninstall_grandparent,
                        uninstall_parent,
                        uninstall_identity,
                        uninstall_descriptor,
                        expected_uninstall_sha256,
                    )
                    if result in {"removed", "missing"}:
                        uninstall_status = "missing"
                        uninstall_identity = None
                print(f"uninstaller:{result}", flush=True)
            elif command == "purge-state":
                try:
                    result = purge_state()
                except (OSError, RuntimeError) as error:
                    print(f"state:error:{error}", flush=True)
                else:
                    print(f"state:{result}", flush=True)
            elif command == "close":
                print("closed", flush=True)
                break
finally:
    for descriptor in (
        unit_descriptor,
        unit_parent,
        unit_grandparent if unit_parent is not None else None,
        metadata_descriptor,
        metadata_parent,
        metadata_grandparent,
        uninstall_descriptor,
        uninstall_parent,
        uninstall_grandparent if uninstall_parent is not None else None,
        codex_home_descriptor,
        codex_home_parent,
        codex_home_grandparent,
        state_descriptor,
        state_parent,
        state_grandparent,
    ):
        if descriptor is not None:
            os.close(descriptor)
PY
}
ownership_guard_read=${OWNERSHIP_GUARD[0]}
ownership_guard_write=${OWNERSHIP_GUARD[1]}
ownership_guard_pid=$OWNERSHIP_GUARD_PID
exec {early_metadata_grandparent_fd}<&-
exec {early_metadata_parent_fd}<&-
exec {early_metadata_fd}>&-
exec {early_unit_grandparent_fd}<&-
exec {early_unit_parent_fd}<&-
exec {early_unit_fd}>&-
exec {early_uninstall_grandparent_fd}<&-
exec {early_uninstall_parent_fd}<&-
exec {early_uninstall_fd}>&-
exec {early_codex_grandparent_fd}<&-
exec {early_codex_parent_fd}<&-
exec {early_state_grandparent_fd}<&-
exec {early_state_parent_fd}<&-
exec {early_state_dir_fd}<&-
IFS= read -r ownership_guard_ready <&"$ownership_guard_read"
ownership_guard_statuses=${ownership_guard_ready#ready:}
unit_guard_status=${ownership_guard_statuses%%:*}
remaining_guard_statuses=${ownership_guard_statuses#*:}
uninstall_guard_status=${remaining_guard_statuses%%:*}
remaining_guard_statuses=${remaining_guard_statuses#*:}
codex_home_guard_status=${remaining_guard_statuses%%:*}
state_guard_status=${remaining_guard_statuses#*:}
if [[ "$unit_guard_status" == "replaced" ]]; then
  printf 'User service definition changed during uninstall initialization and was preserved: %s\n' \
    "$unit_path" >&2
  exit 1
fi
if [[ "$uninstall_guard_status" == "replaced" ]]; then
  printf 'Stable uninstaller changed during uninstall initialization and was preserved: %s\n' \
    "$owned_uninstall_path" >&2
  exit 1
fi
if [[ "$codex_home_guard_status" != "present" && "$codex_home_guard_status" != "missing" ]]; then
  printf 'Recorded Codex home could not be safely bound: %s\n' \
    "$managed_codex_home" >&2
  exit 1
fi
if [[ "$state_guard_status" != "present" && "$state_guard_status" != "missing" ]]; then
  printf 'Recorded platform state directory could not be safely bound: %s\n' \
    "$state_dir" >&2
  exit 1
fi

guard_check_all() {
  local guard_result
  printf 'check-all\n' >&"$ownership_guard_write"
  IFS= read -r guard_result <&"$ownership_guard_read"
  if [[ "$guard_result" != "all:same" ]]; then
    printf 'A managed uninstall path changed during uninstall and was preserved: %s\n' \
      "${guard_result#all:}" >&2
    exit 1
  fi
}

failed_stage="Python tool removal prerequisite validation"
if [[ -z "$managed_uv_path" || ! -f "$managed_uv_path" || -L "$managed_uv_path" || ! -x "$managed_uv_path" ]]; then
  printf 'The recorded uv command is required to safely remove the installed Personal Agent Memory tool.\n' >&2
  exit 1
fi

failed_stage="user service removal"
if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  unit_owned=false
  [[ "$unit_guard_status" == "owned" || "$unit_guard_status" == "missing" ]] && unit_owned=true
  if $unit_owned; then
    if systemctl --user is-active --quiet "$service_name"; then
      guard_check_all
      systemctl --user stop "$service_name" >/dev/null 2>&1
      guard_check_all
    fi
    if [[ -e "$unit_path" || -L "$unit_path" ]] || systemctl --user is-enabled --quiet "$service_name"; then
      guard_check_all
      systemctl --user disable "$service_name" >/dev/null 2>&1
      guard_check_all
    fi
    printf 'delete-unit\n' >&"$ownership_guard_write"
    IFS= read -r unit_removal <&"$ownership_guard_read"
    if [[ "$unit_removal" == "unit:replaced" || "$unit_removal" == "unit:unowned" ]]; then
      printf 'Preserved user service definition because it changed during uninstall: %s\n' \
        "$unit_path" >&2
      exit 1
    fi
  elif [[ -e "$unit_path" || -L "$unit_path" ]]; then
    printf 'Preserved user service definition because installer ownership could not be verified: %s\n' \
      "$unit_path"
  fi
  guard_check_all
  systemctl --user daemon-reload
  guard_check_all
  if systemctl --user is-active --quiet "$service_name"; then
    printf 'Personal Agent Memory service is still active.\n' >&2
    exit 1
  fi
  if systemctl --user is-enabled --quiet "$service_name"; then
    printf 'Personal Agent Memory service is still enabled.\n' >&2
    exit 1
  fi
else
  if [[ -n "$managed_unit_path" || -e "$unit_path" || -L "$unit_path" ]]; then
    printf 'A working systemd user manager is required to stop the installed service.\n' >&2
    exit 1
  fi
fi

failed_stage="Codex plugin removal"
if [[ "$codex_home_guard_status" == "present" ]] && command -v codex >/dev/null 2>&1; then
  codex_runtime_home=/proc/self/fd/19
  guard_check_all
  plugin_list="$(CODEX_HOME="$codex_runtime_home" codex plugin list --json \
    19<&"$early_codex_home_fd")"
  guard_check_all
  coproc PLUGIN_TREE_GUARD {
    "$python_command" /proc/self/fd/18 "$plugin_list" "$managed_plugin_id" \
    "$managed_plugin_version" "$managed_plugin_registration_source" \
    "$managed_plugin_install_path" "$managed_plugin_tree_sha256" \
    17<&"$early_codex_home_fd" 18<<'PY'
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys

MAX_ENTRIES = 4096
MAX_TOTAL_BYTES = 64 * 1024 * 1024


class UnverifiedTree(Exception):
    pass


def plugin_tree_sha256(root: Path, *, bound_root: bool = False) -> str:
    try:
        root_info = os.stat(root) if bound_root else os.lstat(root)
    except OSError as error:
        raise UnverifiedTree(f"cannot inspect plugin root: {error}") from error
    if not stat.S_ISDIR(root_info.st_mode):
        raise UnverifiedTree("plugin root is not a lexical directory")

    entries: list[tuple[bytes, Path, str]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as children:
                for child in children:
                    if len(entries) >= MAX_ENTRIES:
                        raise UnverifiedTree("plugin tree exceeds the entry limit")
                    path = Path(child.path)
                    relative = path.relative_to(root)
                    try:
                        relative_bytes = relative.as_posix().encode("utf-8")
                    except UnicodeEncodeError as error:
                        raise UnverifiedTree("plugin tree has a non-UTF-8 path") from error
                    try:
                        info = child.stat(follow_symlinks=False)
                    except OSError as error:
                        raise UnverifiedTree(
                            f"cannot inspect plugin entry: {relative}"
                        ) from error
                    if stat.S_ISLNK(info.st_mode):
                        raise UnverifiedTree(f"plugin tree contains a symlink: {relative}")
                    if stat.S_ISDIR(info.st_mode):
                        kind = "directory"
                        pending.append(path)
                    elif stat.S_ISREG(info.st_mode):
                        kind = "file"
                    else:
                        raise UnverifiedTree(
                            f"plugin tree contains a special entry: {relative}"
                        )
                    entries.append((relative_bytes, path, kind))
        except OSError as error:
            raise UnverifiedTree(f"cannot scan plugin tree: {error}") from error

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
            raise UnverifiedTree(f"cannot read plugin file: {path}") from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise UnverifiedTree(f"plugin entry changed while hashing: {path}")
            total_bytes += info.st_size
            if total_bytes > MAX_TOTAL_BYTES:
                raise UnverifiedTree("plugin tree exceeds the byte limit")
            digest.update(info.st_size.to_bytes(8, "big"))
            remaining = info.st_size
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    raise UnverifiedTree(f"plugin file changed while hashing: {path}")
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise UnverifiedTree(f"plugin file changed while hashing: {path}")
        finally:
            os.close(descriptor)
    return digest.hexdigest()


def emit(status: str) -> None:
    print(status, flush=True)


def rename_noreplace(
    source_parent_descriptor: int,
    source: str,
    destination_parent_descriptor: int,
    destination: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    result = renameat2(
        source_parent_descriptor,
        os.fsencode(source),
        destination_parent_descriptor,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def clear_directory(descriptor: int, removed: list[int]) -> None:
    with os.scandir(descriptor) as entries:
        for entry in entries:
            removed[0] += 1
            if removed[0] > MAX_ENTRIES:
                raise UnverifiedTree("quarantined plugin tree exceeds the entry limit")
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child_descriptor = os.open(
                    entry.name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_CLOEXEC
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    clear_directory(child_descriptor, removed)
                finally:
                    os.close(child_descriptor)
                os.rmdir(entry.name, dir_fd=descriptor)
            else:
                os.unlink(entry.name, dir_fd=descriptor)


data = json.loads(sys.argv[1])
if not isinstance(data, dict) or not isinstance(data.get("installed"), list):
    raise SystemExit("Codex plugin list returned an invalid payload")
matches = [
    item for item in data["installed"]
    if isinstance(item, dict)
    and item.get("pluginId") == "personal-agent-memory@personal-agent-memory"
]
if len(matches) > 1:
    raise SystemExit("Codex plugin list returned duplicate managed plugin records")
if not matches:
    emit("absent")
elif not all(sys.argv[2:7]):
    emit("unverified")
else:
    match = matches[0]
    source = match.get("source")
    source_path = source.get("path") if isinstance(source, dict) else None
    listed_path = match.get("installedPath")
    identity_matches = (
        sys.argv[2] == "personal-agent-memory@personal-agent-memory"
        and match.get("version") == sys.argv[3]
        and isinstance(source_path, str)
        and not any(c in source_path for c in "\r\n")
        and Path(source_path).resolve() == Path(sys.argv[4]).resolve()
        and (
            listed_path is None
            or (
                isinstance(listed_path, str)
                and not any(c in listed_path for c in "\r\n")
                and Path(listed_path).resolve() == Path(sys.argv[5]).resolve()
            )
        )
    )
    if not identity_matches:
        emit("replaced")
    else:
        root = Path(sys.argv[5])
        codex_home = Path("/proc/self/fd/17").resolve()
        cache_root = codex_home / "plugins/cache"
        try:
            cache_parts = root.resolve().relative_to(cache_root).parts
        except (OSError, ValueError):
            emit("unverified")
            raise SystemExit(0)
        if len(cache_parts) != 3:
            emit("unverified")
            raise SystemExit(0)
        cache_entry = root.parent
        cache_namespace = cache_entry.parent
        directory_flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
        )
        cache_namespace_descriptor = os.open(cache_namespace, directory_flags)
        cache_entry_descriptor = -1
        root_descriptor = -1
        quarantine_name = f".personal-agent-memory-uninstall-{secrets.token_hex(12)}"
        original_mode = stat.S_IMODE(os.fstat(cache_namespace_descriptor).st_mode)
        sealed = False
        quarantined = False
        try:
            cache_entry_descriptor = os.open(
                cache_entry.name,
                directory_flags,
                dir_fd=cache_namespace_descriptor,
            )
            cache_entry_info = os.fstat(cache_entry_descriptor)
            lexical_entry_info = os.stat(
                cache_entry.name,
                dir_fd=cache_namespace_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(cache_entry_info.st_mode)
                or (cache_entry_info.st_dev, cache_entry_info.st_ino)
                != (lexical_entry_info.st_dev, lexical_entry_info.st_ino)
            ):
                raise UnverifiedTree("plugin cache entry changed before ownership verification")
            with os.scandir(cache_entry_descriptor) as versions:
                first_entry = next(versions, None)
                second_entry = next(versions, None)
            if (
                first_entry is None
                or first_entry.name != root.name
                or second_entry is not None
            ):
                raise UnverifiedTree("plugin cache entry contains unowned versions")
            root_descriptor = os.open(
                root.name,
                directory_flags,
                dir_fd=cache_entry_descriptor,
            )
            root_info = os.fstat(root_descriptor)
            lexical_info = os.stat(
                root.name,
                dir_fd=cache_entry_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(root_info.st_mode)
                or (root_info.st_dev, root_info.st_ino)
                != (lexical_info.st_dev, lexical_info.st_ino)
            ):
                raise UnverifiedTree("plugin root changed before ownership verification")
            current_digest = plugin_tree_sha256(
                Path(f"/proc/self/fd/{root_descriptor}"), bound_root=True
            )
            lexical_info = os.stat(
                root.name,
                dir_fd=cache_entry_descriptor,
                follow_symlinks=False,
            )
            if (
                (root_info.st_dev, root_info.st_ino)
                != (lexical_info.st_dev, lexical_info.st_ino)
                or current_digest != sys.argv[6]
            ):
                raise UnverifiedTree("plugin tree no longer matches install metadata")
            rename_noreplace(
                cache_namespace_descriptor,
                cache_entry.name,
                17,
                quarantine_name,
            )
            quarantined = True
            quarantine_info = os.stat(
                quarantine_name, dir_fd=17, follow_symlinks=False
            )
            if (cache_entry_info.st_dev, cache_entry_info.st_ino) != (
                quarantine_info.st_dev,
                quarantine_info.st_ino,
            ):
                raise UnverifiedTree("plugin cache entry changed during quarantine")
            os.fchmod(cache_namespace_descriptor, original_mode & ~0o222)
            sealed = True
            emit("owned")
            action = sys.stdin.readline().strip()
            if action == "commit":
                lexical_info = os.stat(
                    root.name,
                    dir_fd=cache_entry_descriptor,
                    follow_symlinks=False,
                )
                if (
                    (root_info.st_dev, root_info.st_ino)
                    != (lexical_info.st_dev, lexical_info.st_ino)
                    or plugin_tree_sha256(
                        Path(f"/proc/self/fd/{root_descriptor}"), bound_root=True
                    )
                    != sys.argv[6]
                ):
                    raise UnverifiedTree(
                        "quarantined plugin contents changed after ownership verification"
                    )
                with os.scandir(cache_entry_descriptor) as versions:
                    first_entry = next(versions, None)
                    second_entry = next(versions, None)
                if (
                    first_entry is None
                    or first_entry.name != root.name
                    or second_entry is not None
                ):
                    raise UnverifiedTree(
                        "quarantined plugin cache entry changed after ownership verification"
                    )
                clear_directory(cache_entry_descriptor, [0])
                os.fchmod(cache_namespace_descriptor, original_mode)
                sealed = False
                quarantine_info = os.stat(
                    quarantine_name, dir_fd=17, follow_symlinks=False
                )
                if (cache_entry_info.st_dev, cache_entry_info.st_ino) != (
                    quarantine_info.st_dev,
                    quarantine_info.st_ino,
                ):
                    raise UnverifiedTree(
                        "quarantined plugin cache entry changed before deletion"
                    )
                os.rmdir(quarantine_name, dir_fd=17)
                quarantined = False
                emit("committed")
            else:
                os.fchmod(cache_namespace_descriptor, original_mode)
                sealed = False
                rename_noreplace(
                    17,
                    quarantine_name,
                    cache_namespace_descriptor,
                    cache_entry.name,
                )
                quarantined = False
                emit("rolled-back")
        except UnverifiedTree:
            if quarantined:
                if sealed:
                    os.fchmod(cache_namespace_descriptor, original_mode)
                    sealed = False
                try:
                    rename_noreplace(
                        17,
                        quarantine_name,
                        cache_namespace_descriptor,
                        cache_entry.name,
                    )
                    quarantined = False
                except OSError:
                    pass
            emit("content-replaced")
        except BaseException:
            if sealed:
                os.fchmod(cache_namespace_descriptor, original_mode)
            if quarantined:
                try:
                    rename_noreplace(
                        17,
                        quarantine_name,
                        cache_namespace_descriptor,
                        cache_entry.name,
                    )
                except OSError:
                    pass
            raise
        finally:
            if root_descriptor >= 0:
                os.close(root_descriptor)
            if cache_entry_descriptor >= 0:
                os.close(cache_entry_descriptor)
            os.close(cache_namespace_descriptor)
PY
  }
  plugin_guard_read=${PLUGIN_TREE_GUARD[0]}
  plugin_guard_write=${PLUGIN_TREE_GUARD[1]}
  plugin_guard_pid=$PLUGIN_TREE_GUARD_PID
  IFS= read -r plugin_status <&"$plugin_guard_read"
  if [[ "$plugin_status" == "owned" ]]; then
    plugin_removal_failed=false
    if ! CODEX_HOME="$codex_runtime_home" codex plugin remove \
      personal-agent-memory@personal-agent-memory --json \
      19<&"$early_codex_home_fd" >/dev/null; then
      plugin_removal_failed=true
    elif ! plugin_list="$(CODEX_HOME="$codex_runtime_home" codex plugin list --json \
      19<&"$early_codex_home_fd")"; then
      plugin_removal_failed=true
    elif ! "$python_command" - "$plugin_list" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
if not isinstance(data, dict) or not isinstance(data.get("installed"), list):
    raise SystemExit("Codex plugin list returned an invalid payload")
if any(
    isinstance(item, dict)
    and item.get("pluginId") == "personal-agent-memory@personal-agent-memory"
    for item in data["installed"]
):
    raise SystemExit("Codex plugin is still installed after removal")
PY
    then
      plugin_removal_failed=true
    fi
    if $plugin_removal_failed; then
      printf 'rollback\n' >&"$plugin_guard_write"
      IFS= read -r plugin_guard_result <&"$plugin_guard_read" || true
      wait "$plugin_guard_pid" || true
      exec {plugin_guard_write}>&-
      exec {plugin_guard_read}<&-
      guard_check_all
      printf 'Codex failed to remove the verified managed plugin.\n' >&2
      exit 1
    fi
    printf 'commit\n' >&"$plugin_guard_write"
    IFS= read -r plugin_guard_result <&"$plugin_guard_read"
    wait "$plugin_guard_pid"
    exec {plugin_guard_write}>&-
    exec {plugin_guard_read}<&-
    if [[ "$plugin_guard_result" != "committed" ]]; then
      printf 'Verified managed plugin cleanup did not commit safely.\n' >&2
      exit 1
    fi
    guard_check_all
  elif [[ "$plugin_status" == "content-replaced" ]]; then
    wait "$plugin_guard_pid"
    exec {plugin_guard_write}>&-
    exec {plugin_guard_read}<&-
    printf 'Preserved Codex plugin because its installed contents no longer match the installer record.\n'
  elif [[ "$plugin_status" == "replaced" || "$plugin_status" == "unverified" ]]; then
    wait "$plugin_guard_pid"
    exec {plugin_guard_write}>&-
    exec {plugin_guard_read}<&-
    printf 'Preserved Codex plugin because installer ownership could not be verified.\n'
  else
    wait "$plugin_guard_pid"
    exec {plugin_guard_write}>&-
    exec {plugin_guard_read}<&-
  fi

  if [[ "$marketplace_ownership" == "installer-managed" && -n "$marketplace_source" ]]; then
    guard_check_all
    marketplace_list="$(CODEX_HOME="$codex_runtime_home" codex plugin marketplace list --json \
      19<&"$early_codex_home_fd")"
    guard_check_all
    marketplace_owned="$($python_command - "$marketplace_source" "$marketplace_list" <<'PY'
import json
from pathlib import Path
import sys
source = Path(sys.argv[1]).resolve()
data = json.loads(sys.argv[2])
if not isinstance(data, dict) or not isinstance(data.get("marketplaces"), list):
    raise SystemExit("Codex marketplace list returned an invalid payload")
marketplaces = data["marketplaces"]
matches = [item for item in marketplaces if isinstance(item, dict) and item.get("name") == "personal-agent-memory"]
owned = len(matches) == 1 and isinstance(matches[0].get("root"), str)
if owned:
    root = matches[0]["root"]
    owned = not any(c in root for c in "\r\n") and Path(root).resolve() == source
print("true" if owned else "false")
PY
)"
    guard_check_all
    remaining_plugins="$(CODEX_HOME="$codex_runtime_home" codex plugin list --json \
      19<&"$early_codex_home_fd")"
    guard_check_all
    marketplace_shared="$($python_command - "$remaining_plugins" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
if not isinstance(data, dict) or not isinstance(data.get("installed"), list):
    raise SystemExit("Codex plugin list returned an invalid payload")
installed = data["installed"]
print("true" if any(
    isinstance(item, dict)
    and isinstance(item.get("pluginId"), str)
    and item["pluginId"].endswith("@personal-agent-memory")
    for item in installed
) else "false")
PY
)"
    if [[ "$marketplace_owned" == "true" && "$marketplace_shared" == "false" ]]; then
      guard_check_all
      CODEX_HOME="$codex_runtime_home" codex plugin marketplace remove \
        personal-agent-memory --json 19<&"$early_codex_home_fd" >/dev/null
      guard_check_all
      marketplace_list="$(CODEX_HOME="$codex_runtime_home" codex plugin marketplace list --json \
        19<&"$early_codex_home_fd")"
      guard_check_all
      "$python_command" - "$marketplace_list" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
if not isinstance(data, dict) or not isinstance(data.get("marketplaces"), list):
    raise SystemExit("Codex marketplace list returned an invalid payload")
if any(
    isinstance(item, dict) and item.get("name") == "personal-agent-memory"
    for item in data["marketplaces"]
):
    raise SystemExit("Codex marketplace is still registered after removal")
PY
    else
      printf 'Preserved Codex marketplace: ownership or current source could not be verified.\n'
    fi
  else
    printf 'Preserved Codex marketplace: it was not created exclusively by this installer.\n'
  fi
elif [[ "$codex_home_guard_status" == "present" ]]; then
  printf 'Codex CLI is required to verify and remove the installed plugin.\n' >&2
  exit 1
fi

failed_stage="Python tool removal"
managed_daemon_present=false
if [[ -n "$managed_daemon_path" && ( -e "$managed_daemon_path" || -L "$managed_daemon_path" ) ]]; then
  managed_daemon_present=true
fi
if [[ -n "$managed_uv_path" && -f "$managed_uv_path" && ! -L "$managed_uv_path" && -x "$managed_uv_path" ]]; then
  uv_uninstall_output=""
  guard_check_all
  if ! uv_uninstall_output="$("$managed_uv_path" tool uninstall personal-agent-memory 2>&1)"; then
    if [[ -e "$managed_daemon_path" || -L "$managed_daemon_path" ]] || \
      [[ "$uv_uninstall_output" != *"personal-agent-memory"* ]] || \
      [[ "$uv_uninstall_output" != *"not installed"* ]]; then
      printf 'uv failed to remove the installed Personal Agent Memory tool.\n' >&2
      [[ -z "$uv_uninstall_output" ]] || printf '%s\n' "$uv_uninstall_output" >&2
      exit 1
    fi
  fi
  guard_check_all
elif $managed_daemon_present; then
  printf 'uv is required to remove the installed Personal Agent Memory command.\n' >&2
  exit 1
fi
if [[ -n "$managed_daemon_path" && ( -e "$managed_daemon_path" || -L "$managed_daemon_path" ) ]]; then
  printf 'Personal Agent Memory command is still installed after uv removal.\n' >&2
  exit 1
fi

guard_check_all

if $purge; then
  failed_stage="platform state purge"
  if [[ "$metadata_status" != "present" || "$state_dir_ownership" == "preexisting-unproven" ]]; then
    printf 'Platform state was preserved because installer ownership could not be proven: %s\n' "$state_dir" >&2
    printf 'Memory libraries preserved: %s\n' "$library_root"
    exit 1
  fi
  guard_check_all
  printf 'purge-state\n' >&"$ownership_guard_write"
  IFS= read -r state_purge <&"$ownership_guard_read"
  if [[ "$state_purge" == "state:replaced" ]]; then
    printf 'Platform state directory changed during uninstall and was preserved: %s\n' \
      "$state_dir" >&2
    exit 1
  elif [[ "$state_purge" == state:error:* ]]; then
    printf '%s\n' "${state_purge#state:error:}" >&2
    exit 1
  fi
  purge_result=${state_purge#state:}
  guard_check_all
  printf 'delete-metadata\n' >&"$ownership_guard_write"
  IFS= read -r metadata_removal <&"$ownership_guard_read"
  if [[ "$metadata_removal" == "metadata:replaced" ]]; then
    printf 'Install metadata changed during uninstall and was preserved: %s\n' \
      "$install_config_path" >&2
    exit 1
  fi
  printf 'delete-metadata-parent\n' >&"$ownership_guard_write"
  IFS= read -r metadata_parent_removal <&"$ownership_guard_read"
  if [[ "$metadata_parent_removal" == "metadata-parent:replaced" ]]; then
    printf 'Install metadata directory changed during uninstall and was preserved: %s\n' \
      "$install_config_dir" >&2
    exit 1
  fi
  guard_check_all
  if [[ "$purge_result" == *$'\nremoved' || "$purge_result" == "removed" ]]; then
    printf 'Platform state removed: %s\n' "$state_dir"
  else
    printf 'Platform state known contents removed; unknown contents preserved under: %s\n' "$state_dir"
    [[ "$purge_result" != preserved:* ]] || printf 'Preserved unknown state entries: %s\n' "${purge_result#preserved:}"
  fi
else
  printf 'Platform state preserved: %s\n' "$state_dir"
  printf 'API key preserved: %s\n' "$state_dir/api-key"
  printf 'Install configuration preserved: %s\n' "$install_config_path"
fi

printf 'All memory libraries preserved under: %s\n' "$library_root"

if [[ -n "$owned_uninstall_path" ]]; then
  printf 'delete-uninstaller\n' >&"$ownership_guard_write"
  IFS= read -r uninstaller_removal <&"$ownership_guard_read"
  if [[ "$uninstaller_removal" == "uninstaller:replaced" ]]; then
    printf 'Stable uninstaller changed during uninstall and was preserved: %s\n' \
      "$owned_uninstall_path" >&2
    exit 1
  elif [[ "$uninstaller_removal" == "uninstaller:unowned" ]]; then
    printf 'Preserved uninstaller path because ownership could not be verified: %s\n' \
      "$owned_uninstall_path"
  fi
fi

guard_check_all
printf 'close\n' >&"$ownership_guard_write"
IFS= read -r ownership_guard_closed <&"$ownership_guard_read"
wait "$ownership_guard_pid"
exec {ownership_guard_write}>&-
exec {ownership_guard_read}<&-

trap - ERR
printf 'Personal Agent Memory uninstallation complete.\n'
