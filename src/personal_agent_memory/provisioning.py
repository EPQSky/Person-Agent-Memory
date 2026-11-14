from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import socket
import stat
import sys
import tempfile
from contextlib import suppress
from importlib import metadata as importlib_metadata
from pathlib import Path

from personal_agent_memory import __version__

SERVICE_NAME = "personal-agent-memory.service"
METADATA_SCHEMA_VERSION = 2
MAX_METADATA_BYTES = 64 * 1024
REQUIRED_COMMANDS = {
    "node": "Install Node.js and make the node command available on PATH.",
    "codex": "Install Codex CLI and make the codex command available on PATH.",
    "git": "Install Git and make the git command available on PATH.",
}


class ProvisioningError(RuntimeError):
    """A user-provisioning preflight or persistence error."""


def _absolute_path(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or any(char in str(path) for char in "\r\n"):
        raise ProvisioningError(f"{name} must be an absolute path without line breaks")
    return path.resolve()


def _read_metadata(path: Path) -> dict[str, object] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    try:
        if not stat.S_ISREG(info.st_mode):
            raise ProvisioningError(f"existing install metadata must be a regular file: {path}")
        if info.st_size > MAX_METADATA_BYTES:
            raise ProvisioningError(
                f"existing install metadata exceeds {MAX_METADATA_BYTES}-byte limit: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ProvisioningError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvisioningError(f"cannot read existing install metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProvisioningError(f"existing install metadata is invalid: {path}")
    return payload


def _plugin_manifest_path() -> Path:
    """Locate the plugin shipped beside the installed payload.

    A package installation places this module below
    ``<prefix>/venv/lib/pythonX.Y/site-packages`` while the plugin is at
    ``<prefix>/plugins``.  Walking ancestors avoids a hard-coded ``/opt``
    prefix and works for custom prefixes such as ``/srv/pam``.
    """
    candidates: list[Path] = []
    configured = os.environ.get("PERSONAL_AGENT_MEMORY_PLUGIN_MANIFEST")
    if configured:
        candidates.append(Path(configured).expanduser())

    configured_prefix = os.environ.get("PERSONAL_AGENT_MEMORY_INSTALL_PREFIX")
    if configured_prefix:
        candidates.append(
            Path(configured_prefix).expanduser()
            / "plugins/personal-agent-memory/.codex-plugin/plugin.json"
        )

    payload_manifest = os.environ.get("PERSONAL_AGENT_MEMORY_PAYLOAD_MANIFEST")
    if payload_manifest:
        payload_path = Path(payload_manifest).expanduser()
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            plugin_path = payload.get("plugin_manifest")
            if isinstance(plugin_path, str) and plugin_path:
                candidate = Path(plugin_path).expanduser()
                if not candidate.is_absolute():
                    candidate = payload_path.parent / candidate
                candidates.append(candidate)

    package_file = Path(__file__).resolve()
    plugin_relative = Path("plugins/personal-agent-memory/.codex-plugin/plugin.json")
    # For a wheel payload, one of these ancestors is <prefix>.  For a source
    # checkout, the repository root is also found by this walk.
    candidates.extend(ancestor / plugin_relative for ancestor in package_file.parents)
    source_root = package_file.parents[2]
    candidates.append(source_root / plugin_relative)
    candidates.extend(
        [
            Path(
                "/opt/personal-agent-memory/plugins/personal-agent-memory/.codex-plugin/plugin.json"
            ),
            Path("/usr/share/personal-agent-memory/plugin.json"),
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    raise ProvisioningError(
        "cannot find the installed Codex plugin manifest; "
        "set PERSONAL_AGENT_MEMORY_INSTALL_PREFIX or "
        "PERSONAL_AGENT_MEMORY_PLUGIN_MANIFEST"
    )


def _versions() -> tuple[str, str, str]:
    project_version = __version__
    try:
        package_version = importlib_metadata.version("personal-agent-memory")
    except importlib_metadata.PackageNotFoundError:
        package_version = project_version
    try:
        manifest = json.loads(_plugin_manifest_path().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvisioningError(
            f"cannot read the installed Codex plugin manifest: {error}"
        ) from error
    plugin_version = manifest.get("version") if isinstance(manifest, dict) else None
    if not isinstance(plugin_version, str) or not plugin_version:
        raise ProvisioningError("installed Codex plugin manifest has no valid version")
    if not package_version or len({project_version, package_version, plugin_version}) != 1:
        raise ProvisioningError(
            "package, plugin manifest, and project versions must match "
            f"(project={project_version}, package={package_version}, plugin={plugin_version})"
        )
    return project_version, package_version, plugin_version


def _check_dependencies() -> None:
    missing = [name for name in REQUIRED_COMMANDS if shutil.which(name) is None]
    if missing:
        details = "; ".join(f"{name}: {REQUIRED_COMMANDS[name]}" for name in missing)
        raise ProvisioningError(f"missing required dependencies: {', '.join(missing)}. {details}")


def _ensure_port_available(port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise ProvisioningError(f"port {port} on 127.0.0.1 is already in use") from error
    finally:
        probe.close()


def _check_command_shadowing() -> None:
    candidate = shutil.which("personal-agent-memory")
    if candidate is None:
        return
    try:
        current = Path(sys.argv[0]).resolve()
        found = Path(candidate).resolve()
    except OSError:
        return
    if found != current:
        raise ProvisioningError(
            "another personal-agent-memory command shadows this package installation on PATH: "
            f"{found}; remove it or invoke the package command by its absolute path"
        )


def _check_unit_conflict(unit_path: Path, metadata: dict[str, object] | None) -> None:
    try:
        info = unit_path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProvisioningError(
            f"user service definition must be a regular non-symlink file: {unit_path}"
        )
    # A package setup may be repeated. Existing metadata proves ownership only
    # when it points at this exact unit and records its current digest.
    if metadata and metadata.get("unit_path") == str(unit_path):
        recorded = metadata.get("unit_sha256")
        if isinstance(recorded, str) and len(recorded) == 64:
            try:
                digest = hashlib.sha256(unit_path.read_bytes()).hexdigest()
            except OSError as error:
                raise ProvisioningError(
                    f"cannot inspect user service definition: {error}"
                ) from error
            if digest == recorded:
                return
    raise ProvisioningError(
        f"user service definition already exists and is not owned by this package: {unit_path}; "
        "stop/remove the other installation before running setup"
    )


def _check_api_key_path(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        raise ProvisioningError(f"API key path must not be a symbolic link: {path}")
    if not stat.S_ISREG(mode):
        raise ProvisioningError(f"API key path is not a regular file: {path}")


def _write_secret(path: Path) -> None:
    _check_api_key_path(path)
    if path.exists():
        os.chmod(path, 0o600)
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".api-key.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(secrets.token_urlsafe(32))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".install.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def provision_package(
    *, state_dir: str | None, library_root: str | None, port: int | None
) -> dict[str, str | int]:
    """Run package-channel setup after a no-write preflight."""
    _check_dependencies()
    project_version, package_version, plugin_version = _versions()

    config_home = _absolute_path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")), "XDG_CONFIG_HOME"
    )
    metadata_path = config_home / "personal-agent-memory" / "install.json"
    metadata = _read_metadata(metadata_path)
    if metadata is not None:
        method = metadata.get("install_method")
        if method is None and "uninstall_ownership" in metadata:
            method = "script"
        if method == "script":
            raise ProvisioningError(
                "a script installation is already configured; uninstall it before "
                "using package setup"
            )
        schema_version = metadata.get("schema_version")
        if (
            method != "package"
            or type(schema_version) is not int
            or schema_version not in {1, METADATA_SCHEMA_VERSION}
        ):
            raise ProvisioningError(
                f"existing install metadata has an unsupported installation method: {metadata_path}"
            )

    chosen_state = (
        state_dir
        if state_dir is not None
        else str(metadata.get("state_dir"))
        if metadata
        else str(Path.home() / ".local/share/personal-agent-memory")
    )
    chosen_library = (
        library_root
        if library_root is not None
        else str(metadata.get("library_root"))
        if metadata
        else str(Path.home() / "memory-libraries")
    )
    metadata_port = metadata.get("port", 7331) if metadata else 7331
    if type(metadata_port) is not int:
        raise ProvisioningError(f"existing install metadata has an invalid port: {metadata_path}")
    chosen_port = port if port is not None else metadata_port
    if not 1 <= chosen_port <= 65535:
        raise ProvisioningError("port must be an integer between 1 and 65535")
    state_path = _absolute_path(chosen_state, "state directory")
    library_path = _absolute_path(chosen_library, "memory library root")
    config_dir = config_home / "personal-agent-memory"
    if _paths_overlap(state_path, library_path):
        raise ProvisioningError(
            "state directory and memory library root must not contain each other"
        )
    if _paths_overlap(state_path, config_dir) or _paths_overlap(library_path, config_dir):
        raise ProvisioningError(
            "state directory and memory library root must not overlap "
            "the install metadata directory"
        )
    api_key_path = state_path / "api-key"
    _check_api_key_path(api_key_path)

    _check_command_shadowing()
    _ensure_port_available(chosen_port)
    unit_path = config_home / "systemd/user" / SERVICE_NAME
    _check_unit_conflict(unit_path, metadata)

    # No persistent user state is written before this point.
    state_existed = state_path.exists()
    library_existed = library_path.exists()
    try:
        state_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        library_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not state_existed:
            os.chmod(state_path, 0o700)
        if not library_existed:
            os.chmod(library_path, 0o700)
        api_key_path = state_path / "api-key"
        _write_secret(api_key_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(metadata_path.parent, 0o700)
        payload: dict[str, object] = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "install_method": "package",
            "installed_version": package_version,
            "plugin_version": plugin_version,
            "state_dir": str(state_path),
            "state_dir_ownership": "package-managed",
            "library_root": str(library_path),
            "library_root_ownership": "user-content-never-delete",
            "api_key_path": str(api_key_path),
            "port": chosen_port,
        }
        # Preserve integration ownership records across the core phase.  The
        # service phase may then prove an existing unit and avoid needless
        # metadata replacement during an idempotent rerun.
        if metadata:
            for key in (
                "unit_path",
                "unit_sha256",
                "unit_ownership",
                "plugin_id",
                "plugin_ownership",
                "plugin_registration_source",
                "plugin_install_path",
            ):
                if key in metadata:
                    payload[key] = metadata[key]
        if metadata != payload:
            _write_metadata(metadata_path, payload)
    except OSError as error:
        raise ProvisioningError(f"package setup could not persist user state: {error}") from error

    return {
        "version": project_version,
        "state_dir": str(state_path),
        "library_root": str(library_path),
        "api_key_path": str(api_key_path),
        "port": chosen_port,
        "metadata_path": str(metadata_path),
    }
