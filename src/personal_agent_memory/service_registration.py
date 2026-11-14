"""Package-channel user service and Codex registration helpers.

The module is deliberately small and command-oriented. Package installation owns
only the executable payload; this module owns user-level integration after an
explicit ``setup`` invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from personal_agent_memory.provisioning import (
    SERVICE_NAME,
    ProvisioningError,
    _absolute_path,
    _plugin_manifest_path,
    _read_metadata,
    _write_metadata,
)


def _run(command: str, args: list[str], *, required: bool = True) -> str:
    path = shutil.which(command)
    if path is None:
        if required:
            raise ProvisioningError(f"required command is missing: {command}")
        return ""
    try:
        result = subprocess.run(
            [path, *args],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        if required:
            raise ProvisioningError(f"required command is unavailable: {command}") from None
        return ""
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ProvisioningError(f"{command} {' '.join(args)} failed{suffix}") from error
    return result.stdout.strip()


def _quote_unit_argument(value: str) -> str:
    """Quote a systemd unit argument using install.sh's escaping rules."""
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    value = value.replace("$", "$$")
    value = value.replace("%", "%%")
    return f'"{value}"'


def _unit_content(daemon: str, state: Path, library: Path, port: int) -> str:
    daemon_arg = _quote_unit_argument(daemon)
    state_arg = _quote_unit_argument(str(state))
    library_arg = _quote_unit_argument(str(library))
    return (
        "[Unit]\n"
        "Description=Personal Agent Memory daemon\n"
        "After=network.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={daemon_arg} serve --state-dir {state_arg} --library-root {library_arg} "
        f'--host "127.0.0.1" --port "{port}"\n'
        "Restart=on-failure\n"
        "RestartSec=2\n"
        "TimeoutStopSec=120\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _ensure_lexical_parent(path: Path) -> None:
    parent = path.parent
    while True:
        try:
            info = parent.lstat()
        except FileNotFoundError:
            info = None
        if info is not None and stat.S_ISLNK(info.st_mode):
            raise ProvisioningError(
                f"user service directory contains a symbolic-link parent: {parent}"
            )
        if parent == parent.parent:
            return
        parent = parent.parent


def _write_unit(path: Path, content: str) -> str:
    _ensure_lexical_parent(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        info = path.lstat()
    except FileNotFoundError:
        info = None
    if info is not None and (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise ProvisioningError(
            f"user service definition must be a regular non-symlink file: {path}"
        )
    if info is not None:
        try:
            if path.read_text(encoding="utf-8") == content:
                return hashlib.sha256(content.encode("utf-8")).hexdigest()
        except OSError as error:
            raise ProvisioningError(
                f"cannot inspect user service definition {path}: {error}"
            ) from error
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except OSError as error:
        raise ProvisioningError(f"cannot write user service definition {path}: {error}") from error
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _metadata_with_integration(
    metadata_path: Path, updates: dict[str, object]
) -> dict[str, object]:
    metadata = _read_metadata(metadata_path)
    if metadata is None:
        raise ProvisioningError(f"package install metadata is missing: {metadata_path}")
    updated = {**metadata, **updates}
    if updated != metadata:
        try:
            _write_metadata(metadata_path, updated)
        except OSError as error:
            raise ProvisioningError(
                f"cannot persist package integration metadata {metadata_path}: {error}"
            ) from error
    return updated


def _marketplace_source() -> Path:
    configured = os.environ.get("PERSONAL_AGENT_MEMORY_MARKETPLACE_SOURCE")
    if configured:
        return Path(configured).expanduser().resolve()
    manifest = _plugin_manifest_path()
    # Package payloads may place plugin.json directly under /usr/share or /opt.
    if manifest.parent.name == "personal-agent-memory":
        return manifest.parent
    # A release root contains plugins/personal-agent-memory/.codex-plugin/plugin.json.
    if len(manifest.parents) > 3 and manifest.parents[2].name == "plugins":
        return manifest.parents[3]
    return manifest.parent


def _parse_json_object(result: str, label: str) -> dict[str, object]:
    if not result:
        raise ProvisioningError(f"Codex {label} returned empty JSON")
    try:
        payload = json.loads(result)
    except json.JSONDecodeError as error:
        raise ProvisioningError(f"Codex {label} returned invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ProvisioningError(f"Codex {label} returned a JSON value that is not an object")
    return payload


def register_plugin(
    *,
    version: str,
    adopt_marketplace: bool = False,
    on_plugin_added: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Register and verify the package marketplace and plugin."""
    source = _marketplace_source()
    existing = _parse_json_object(
        _run("codex", ["plugin", "marketplace", "list", "--json"]),
        "marketplace list",
    )
    marketplaces = existing.get("marketplaces")
    matches = (
        [
            item
            for item in marketplaces
            if isinstance(item, dict) and item.get("name") == "personal-agent-memory"
        ]
        if isinstance(marketplaces, list)
        else []
    )
    if len(matches) > 1:
        raise ProvisioningError(
            "Codex marketplace list must contain exactly one personal-agent-memory "
            "entry when present"
        )
    should_add = not matches or adopt_marketplace
    if matches:
        root = matches[0].get("root")
        if not isinstance(root, str):
            raise ProvisioningError("Codex marketplace entry has no local source root")
        current_source = Path(root).expanduser().resolve()
        if current_source != source and not adopt_marketplace:
            raise ProvisioningError(
                "a personal-agent-memory marketplace exists at another source; "
                "rerun with --adopt-marketplace to replace it"
            )
        if adopt_marketplace:
            _run("codex", ["plugin", "marketplace", "remove", "personal-agent-memory", "--json"])
    if should_add:
        marketplace_result = _run("codex", ["plugin", "marketplace", "add", str(source), "--json"])
        marketplace = _parse_json_object(marketplace_result, "marketplace add")
        if marketplace.get("marketplaceName") != "personal-agent-memory":
            raise ProvisioningError("Codex registered an unexpected marketplace")
        installed_root = marketplace.get("installedRoot")
        if (
            not isinstance(installed_root, str)
            or Path(installed_root).expanduser().resolve() != source
        ):
            raise ProvisioningError("Codex marketplace did not converge to the package source")

    listing = _parse_json_object(
        _run("codex", ["plugin", "marketplace", "list", "--json"]),
        "marketplace list",
    )
    marketplaces = listing.get("marketplaces")
    matching_marketplaces = (
        [
            item
            for item in marketplaces
            if isinstance(item, dict) and item.get("name") == "personal-agent-memory"
        ]
        if isinstance(marketplaces, list)
        else []
    )
    if len(matching_marketplaces) != 1:
        raise ProvisioningError(
            "Codex marketplace list must contain exactly one personal-agent-memory entry"
        )
    marketplace_entry = matching_marketplaces[0]
    if (
        not isinstance(marketplace_entry.get("root"), str)
        or Path(marketplace_entry["root"]).expanduser().resolve() != source
    ):
        raise ProvisioningError("Codex marketplace list does not contain the package source")

    plugin_result = _parse_json_object(
        _run("codex", ["plugin", "add", "personal-agent-memory@personal-agent-memory", "--json"]),
        "plugin add",
    )
    plugin_id = plugin_result.get("pluginId")
    plugin_version = plugin_result.get("version")
    installed_path = plugin_result.get("installedPath")
    if plugin_id != "personal-agent-memory@personal-agent-memory":
        raise ProvisioningError(f"Codex registered an unexpected plugin: {plugin_id!r}")
    if plugin_version != version:
        raise ProvisioningError(
            f"Codex plugin version {plugin_version!r} does not match package {version!r}"
        )
    if not isinstance(installed_path, str) or not Path(installed_path).expanduser().is_dir():
        raise ProvisioningError("Codex did not report an installed plugin directory")
    if on_plugin_added is not None:
        on_plugin_added()

    plugins = _parse_json_object(_run("codex", ["plugin", "list", "--json"]), "plugin list")
    installed = plugins.get("installed")
    matching_plugins = (
        [item for item in installed if isinstance(item, dict) and item.get("pluginId") == plugin_id]
        if isinstance(installed, list)
        else []
    )
    if len(matching_plugins) != 1:
        raise ProvisioningError(
            "Codex plugin list must contain exactly one personal-agent-memory plugin"
        )
    plugin_entry = matching_plugins[0]
    if (
        plugin_entry.get("version") != version
        or plugin_entry.get("enabled") is not True
        or not isinstance(plugin_entry.get("installedPath"), str)
        or Path(plugin_entry["installedPath"]).expanduser().resolve()
        != Path(installed_path).expanduser().resolve()
        or not isinstance(plugin_entry.get("source"), dict)
        or not isinstance(plugin_entry["source"].get("path"), str)
        or Path(plugin_entry["source"]["path"]).expanduser().resolve() != source
    ):
        raise ProvisioningError(
            "Codex plugin list does not contain the installed package version and source"
        )
    return {
        "plugin_id": str(plugin_id),
        "plugin_ownership": "package-managed",
        "plugin_version": str(plugin_version),
        "plugin_install_path": str(Path(installed_path).expanduser().resolve()),
        "plugin_registration_source": str(source),
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        response: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def _health_check(api_key_path: Path, port: int, *, timeout: float = 30.0) -> None:
    """Wait for authenticated liveness and usable memory health."""
    deadline = time.monotonic() + timeout
    last_error = "API key was not created"
    while time.monotonic() < deadline:
        try:
            key = api_key_path.read_text(encoding="utf-8").strip()
            if not key:
                raise OSError("API key is empty")
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/health/live",
                headers={"Authorization": f"Bearer {key}"},
            )
            with _NO_REDIRECT_OPENER.open(request, timeout=5) as response:
                parsed = urllib.parse.urlparse(response.geturl())
                if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != port:
                    raise ProvisioningError(
                        "health endpoint redirected to an unexpected loopback address"
                    )
                payload = json.load(response)
            if payload == {"status": "ok"}:
                break
            last_error = f"unexpected live health response: {payload!r}"
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(0.1)
    else:
        raise ProvisioningError(
            f"daemon did not become healthy: {last_error}. "
            f"Inspect with systemctl --user status {SERVICE_NAME} and "
            f"journalctl --user -u {SERVICE_NAME} --no-pager"
        )

    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/mcp/health",
            headers={"Authorization": f"Bearer {key}"},
        )
        with _NO_REDIRECT_OPENER.open(request, timeout=60) as response:
            parsed = urllib.parse.urlparse(response.geturl())
            if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != port:
                raise ProvisioningError(
                    "memory health endpoint redirected to an unexpected loopback address"
                )
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise ProvisioningError(
            f"authenticated memory health check failed: {error}. "
            f"Inspect with systemctl --user status {SERVICE_NAME} and "
            f"journalctl --user -u {SERVICE_NAME} --no-pager"
        ) from error
    if not isinstance(payload, dict) or payload.get("memory_kernel") != "mem2":
        raise ProvisioningError(f"memory health selected the wrong kernel: {payload!r}")
    if payload.get("memory_kernel_status") not in {"ready", "degraded"}:
        raise ProvisioningError(f"memory kernel is not usable: {payload!r}")


def _service_tools_available() -> bool:
    return all(
        shutil.which(name) and Path(shutil.which(name) or "").exists()
        for name in ("codex", "systemctl")
    )


def _current_package_cli() -> str | None:
    configured = os.environ.get("PERSONAL_AGENT_MEMORY_DAEMON")
    if configured:
        return configured
    current = Path(sys.argv[0])
    if current.name == "personal-agent-memory" and current.is_file():
        return str(current.resolve())
    return shutil.which("personal-agent-memory")


def _package_command_available() -> bool:
    return _current_package_cli() is not None


def _require_integration_tools() -> None:
    missing = [name for name in ("codex", "systemctl") if not shutil.which(name)]
    if missing:
        raise ProvisioningError(
            "package setup requires missing integration tools: "
            + ", ".join(missing)
            + ". Install them and rerun setup."
        )


def _plugin_remove_idempotent() -> None:
    try:
        _run(
            "codex",
            ["plugin", "remove", "personal-agent-memory@personal-agent-memory", "--json"],
        )
    except ProvisioningError as error:
        detail = str(error).lower()
        if not any(marker in detail for marker in ("not found", "not installed", "no such")):
            raise


def _marketplace_previous_source() -> Path | None:
    listing = _parse_json_object(
        _run("codex", ["plugin", "marketplace", "list", "--json"]),
        "marketplace list",
    )
    marketplaces = listing.get("marketplaces")
    matches = (
        [
            item
            for item in marketplaces
            if isinstance(item, dict) and item.get("name") == "personal-agent-memory"
        ]
        if isinstance(marketplaces, list)
        else []
    )
    if len(matches) > 1:
        raise ProvisioningError(
            "Codex marketplace list must contain exactly one personal-agent-memory entry"
        )
    if not matches:
        return None
    root = matches[0].get("root")
    if not isinstance(root, str):
        raise ProvisioningError("Codex marketplace entry has no local source root")
    return Path(root).expanduser().resolve()


def _service_enabled_before() -> bool:
    try:
        result = _run("systemctl", ["--user", "is-enabled", SERVICE_NAME])
    except ProvisioningError as error:
        detail = str(error).lower()
        if any(
            marker in detail
            for marker in ("not found", "could not be found", "not loaded", "disabled", "no such")
        ):
            return False
        raise
    return result.strip() in {"enabled", "enabled-runtime"}


def _remove_created_marketplace() -> None:
    try:
        _run("codex", ["plugin", "marketplace", "remove", "personal-agent-memory", "--json"])
    except ProvisioningError as error:
        detail = str(error).lower()
        if not any(marker in detail for marker in ("not found", "no such", "not installed")):
            raise
    listing = _parse_json_object(
        _run("codex", ["plugin", "marketplace", "list", "--json"]),
        "marketplace rollback list",
    )
    marketplaces = listing.get("marketplaces")
    matches = (
        [
            item
            for item in marketplaces
            if isinstance(item, dict) and item.get("name") == "personal-agent-memory"
        ]
        if isinstance(marketplaces, list)
        else []
    )
    if matches:
        raise ProvisioningError("Codex marketplace rollback left a package marketplace behind")


def _restore_marketplace(source: Path) -> None:
    _run("codex", ["plugin", "marketplace", "remove", "personal-agent-memory", "--json"])
    result = _parse_json_object(
        _run("codex", ["plugin", "marketplace", "add", str(source), "--json"]),
        "marketplace restore",
    )
    installed_root = result.get("installedRoot")
    if (
        result.get("marketplaceName") != "personal-agent-memory"
        or not isinstance(installed_root, str)
        or Path(installed_root).expanduser().resolve() != source
    ):
        raise ProvisioningError("Codex marketplace rollback did not restore the previous source")
    listing = _parse_json_object(
        _run("codex", ["plugin", "marketplace", "list", "--json"]),
        "marketplace restore list",
    )
    marketplaces = listing.get("marketplaces")
    matches = (
        [
            item
            for item in marketplaces
            if isinstance(item, dict)
            and item.get("name") == "personal-agent-memory"
            and isinstance(item.get("root"), str)
            and Path(item["root"]).expanduser().resolve() == source
        ]
        if isinstance(marketplaces, list)
        else []
    )
    if len(matches) != 1:
        raise ProvisioningError("Codex marketplace rollback list verification failed")


def _plugin_present_before_setup() -> bool:
    listing = _parse_json_object(_run("codex", ["plugin", "list", "--json"]), "plugin list")
    installed = listing.get("installed")
    if not isinstance(installed, list):
        raise ProvisioningError("Codex plugin list has no installed list")
    matching = [
        item
        for item in installed
        if isinstance(item, dict)
        and item.get("pluginId") == "personal-agent-memory@personal-agent-memory"
    ]
    if len(matching) > 1:
        raise ProvisioningError(
            "Codex plugin list must contain at most one package plugin before setup"
        )
    return bool(matching)


def integrate_package_setup(
    summary: dict[str, str | int], *, adopt_marketplace: bool = False
) -> dict[str, str | int]:
    """Register plugin, install/start the user service, and check health."""
    if not _package_command_available():
        return {**summary, "service_status": "unavailable"}
    _require_integration_tools()

    metadata_path = Path(str(summary["metadata_path"]))
    state = Path(str(summary["state_dir"]))
    library = Path(str(summary["library_root"]))
    port = int(summary["port"])
    config_home = metadata_path.parent.parent
    unit_path = config_home / "systemd" / "user" / SERVICE_NAME
    daemon = _current_package_cli()
    if not daemon:
        raise ProvisioningError("cannot locate the package personal-agent-memory command")
    content = _unit_content(daemon, state, library, port)
    try:
        unit_info = unit_path.lstat()
    except FileNotFoundError:
        unit_info = None
    if unit_info is not None and stat.S_ISLNK(unit_info.st_mode):
        raise ProvisioningError(f"user service definition must not be a symbolic link: {unit_path}")
    unit_existed = unit_info is not None
    previous_content = unit_path.read_text(encoding="utf-8") if unit_existed else None
    metadata_before = _read_metadata(metadata_path)
    if metadata_before is None:
        raise ProvisioningError(f"package install metadata is missing: {metadata_path}")
    marketplace_before = _marketplace_previous_source()
    marketplace_created = marketplace_before is None
    marketplace_adopted = adopt_marketplace and marketplace_before is not None
    plugin_present_before = _plugin_present_before_setup()
    enabled_before = _service_enabled_before()
    plugin_added = False
    service_start_attempted = False
    digest = _write_unit(unit_path, content)
    try:
        version = str(summary["version"])

        def mark_plugin_added() -> None:
            nonlocal plugin_added
            plugin_added = True

        plugin = register_plugin(
            version=version,
            adopt_marketplace=adopt_marketplace,
            on_plugin_added=mark_plugin_added,
        )
        _metadata_with_integration(
            metadata_path,
            {
                "unit_path": str(unit_path),
                "unit_sha256": digest,
                "unit_ownership": "package-managed",
                **plugin,
                "plugin_registration_source": str(_marketplace_source()),
            },
        )
        _run("systemctl", ["--user", "daemon-reload"])
        service_start_attempted = True
        _run("systemctl", ["--user", "enable", "--now", SERVICE_NAME])
        _health_check(Path(str(summary["api_key_path"])), port)
        status = _run("systemctl", ["--user", "is-active", SERVICE_NAME]) or "active"
    except ProvisioningError as error:
        rollback_errors: list[str] = []
        if service_start_attempted:
            try:
                _run("systemctl", ["--user", "stop", SERVICE_NAME])
                if enabled_before:
                    _run("systemctl", ["--user", "enable", SERVICE_NAME])
                else:
                    _run("systemctl", ["--user", "disable", SERVICE_NAME])
            except ProvisioningError as rollback_error:
                rollback_errors.append(f"service enabled-state rollback failed: {rollback_error}")
        if plugin_added and not plugin_present_before:
            try:
                _plugin_remove_idempotent()
            except ProvisioningError as rollback_error:
                rollback_errors.append(f"plugin rollback failed: {rollback_error}")
        if marketplace_created:
            try:
                _remove_created_marketplace()
            except ProvisioningError as rollback_error:
                rollback_errors.append(f"marketplace rollback failed: {rollback_error}")
        elif marketplace_adopted and marketplace_before is not None:
            try:
                _restore_marketplace(marketplace_before)
            except ProvisioningError as rollback_error:
                rollback_errors.append(f"marketplace rollback failed: {rollback_error}")
        try:
            _write_metadata(metadata_path, metadata_before)
        except OSError as rollback_error:
            rollback_errors.append(f"metadata rollback failed: {rollback_error}")
        try:
            if not unit_existed:
                unit_path.unlink()
            elif previous_content is not None:
                unit_path.write_text(previous_content, encoding="utf-8")
        except OSError as rollback_error:
            rollback_errors.append(f"unit rollback failed: {rollback_error}")
        try:
            _run("systemctl", ["--user", "daemon-reload"])
        except ProvisioningError as rollback_error:
            rollback_errors.append(f"systemd daemon-reload rollback failed: {rollback_error}")
        rollback_note = " " + " ".join(rollback_errors) if rollback_errors else ""
        raise ProvisioningError(
            f"setup service or health check failed: {error}.{rollback_note} "
            f"Inspect with systemctl --user status {SERVICE_NAME} and "
            f"journalctl --user -u {SERVICE_NAME} --no-pager"
        ) from error
    return {**summary, "service_status": status, "unit_path": str(unit_path)}


def _remove_owned_plugin(metadata: dict[str, object]) -> str:
    plugin_id = metadata.get("plugin_id")
    ownership = metadata.get("plugin_ownership")
    source = metadata.get("plugin_registration_source")
    recorded_path = metadata.get("plugin_install_path")
    recorded_version = metadata.get("plugin_version")
    if (
        plugin_id != "personal-agent-memory@personal-agent-memory"
        or ownership != "package-managed"
        or not isinstance(source, str)
        or not Path(source).is_absolute()
        or not isinstance(recorded_path, str)
        or not Path(recorded_path).is_absolute()
        or not isinstance(recorded_version, str)
        or not recorded_version
    ):
        return "preserved (plugin ownership metadata is unavailable)"
    listing = _parse_json_object(_run("codex", ["plugin", "list", "--json"]), "plugin list")
    installed = listing.get("installed")
    if not isinstance(installed, list):
        raise ProvisioningError("refusing to remove plugin: Codex plugin list is invalid")
    matches = [
        item for item in installed if isinstance(item, dict) and item.get("pluginId") == plugin_id
    ]
    if not matches:
        return "not installed"
    if len(matches) != 1:
        raise ProvisioningError("refusing to remove duplicate package plugin entries")
    entry = matches[0]
    actual_source = entry.get("source")
    actual_source_path = actual_source.get("path") if isinstance(actual_source, dict) else None
    current_source = _marketplace_source().expanduser().resolve()
    if (
        Path(source).expanduser().resolve() != current_source
        or not isinstance(actual_source_path, str)
        or Path(actual_source_path).expanduser().resolve() != current_source
    ):
        return "preserved (plugin source does not match package marketplace)"
    actual_install_path = entry.get("installedPath")
    if (
        not isinstance(actual_install_path, str)
        or Path(actual_install_path).expanduser().resolve()
        != Path(recorded_path).expanduser().resolve()
    ):
        return "preserved (plugin installed path does not match metadata)"
    install_path = Path(actual_install_path).expanduser()
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    try:
        _ensure_lexical_parent(install_path)
        install_path.resolve(strict=False).relative_to(codex_home.resolve(strict=False))
    except (OSError, ValueError, ProvisioningError):
        return "preserved (plugin installed path is outside the controlled Codex layout)"
    if recorded_version is not None and entry.get("version") != recorded_version:
        return "preserved (plugin version does not match metadata)"
    _plugin_remove_idempotent()
    return "removed"


def _owned_unit_path(metadata: dict[str, object], expected_path: Path) -> tuple[Path | None, str]:
    unit_value = metadata.get("unit_path")
    recorded = metadata.get("unit_sha256")
    if not isinstance(unit_value, str) or not isinstance(recorded, str) or len(recorded) != 64:
        return None, "preserved (unit ownership metadata is unavailable)"
    if metadata.get("unit_ownership") != "package-managed":
        return None, "preserved (unit is not package-owned)"
    path = Path(unit_value)
    if not path.is_absolute():
        return None, "preserved (unit path is not absolute)"
    try:
        _ensure_lexical_parent(path)
        _ensure_lexical_parent(expected_path)
    except ProvisioningError:
        return None, "preserved (unit path contains a symbolic-link parent)"
    if path != expected_path or path.resolve(strict=False) != expected_path.resolve(strict=False):
        return None, "preserved (unit path is not the package service path)"
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None, "not present"
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return None, "preserved (unit is not a regular file)"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None, "preserved (unit cannot be inspected)"
    if digest != recorded:
        return None, "preserved (unit content does not match metadata)"
    return path, "owned"


def _ensure_no_symlink_path(path: Path) -> None:
    current = path
    chain: list[Path] = []
    while True:
        chain.append(current)
        if current == current.parent:
            break
        current = current.parent
    for entry in reversed(chain):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ProvisioningError(
                f"refusing to purge state through symbolic-link path component: {entry}"
            )


def uninstall_package(*, purge: bool = False) -> dict[str, str]:
    """Stop/remove package-owned user integration; never remove memory libraries."""
    config_home = _absolute_path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")), "XDG_CONFIG_HOME"
    )
    metadata_path = config_home / "personal-agent-memory" / "install.json"
    metadata = _read_metadata(metadata_path)
    if metadata is None:
        raise ProvisioningError(f"package install metadata is missing: {metadata_path}")
    if metadata.get("install_method") != "package":
        raise ProvisioningError("refusing to remove a non-package installation")
    expected_unit = config_home / "systemd" / "user" / SERVICE_NAME
    owned_unit, unit_status = _owned_unit_path(metadata, expected_unit)
    if owned_unit is not None and _service_tools_available():
        for args in (
            ["--user", "stop", SERVICE_NAME],
            ["--user", "disable", SERVICE_NAME],
            ["--user", "daemon-reload"],
        ):
            try:
                _run("systemctl", args)
            except ProvisioningError as error:
                # systemd reports an absent or inactive unit as a non-zero result;
                # those states are already the desired uninstall outcome.
                if args[-1] != SERVICE_NAME or not any(
                    marker in str(error).lower()
                    for marker in (
                        "not found",
                        "not loaded",
                        "inactive",
                        "no such",
                        "could not be found",
                    )
                ):
                    raise
    plugin_status = _remove_owned_plugin(metadata)
    if owned_unit is not None:
        owned_unit.unlink()
        if _service_tools_available():
            _run("systemctl", ["--user", "daemon-reload"])
    state = metadata.get("state_dir")
    library = metadata.get("library_root")
    if purge and isinstance(state, str):
        if metadata.get("state_dir_ownership") != "package-managed":
            raise ProvisioningError("refusing to purge state without package state ownership proof")
        state_path = Path(state).expanduser()
        if not state_path.is_absolute():
            raise ProvisioningError("package state directory must be an absolute path")
        _ensure_no_symlink_path(state_path)
        if isinstance(library, str):
            library_path = Path(library).expanduser()
            if not library_path.is_absolute():
                raise ProvisioningError("memory library directory must be an absolute path")
            _ensure_no_symlink_path(library_path)
            state_real = state_path.resolve(strict=False)
            library_real = library_path.resolve(strict=False)
            if (
                state_real == library_real
                or state_real.is_relative_to(library_real)
                or library_real.is_relative_to(state_real)
            ):
                raise ProvisioningError("refusing to purge state overlapping the memory library")
        try:
            info = state_path.lstat()
        except FileNotFoundError:
            info = None
        if info is not None and not stat.S_ISDIR(info.st_mode):
            raise ProvisioningError("refusing to purge package state that is not a directory")
        if info is not None:
            shutil.rmtree(state_path)
        metadata_path.unlink(missing_ok=True)
    elif purge:
        metadata_path.unlink(missing_ok=True)
    return {
        "metadata_path": str(metadata_path),
        "state_dir": str(state or ""),
        "library_root": str(library or ""),
        "plugin_status": plugin_status,
        "unit_status": unit_status,
        "purged": "true" if purge else "false",
    }


def setup_summary(summary: dict[str, str | int]) -> str:
    return "\n".join(
        [
            "Personal Agent Memory setup complete.",
            f"Service status: {summary.get('service_status', 'unknown')}",
            f"Web: http://127.0.0.1:{summary['port']}",
            f"API key file: {summary['api_key_path']}",
            f"Start: systemctl --user start {SERVICE_NAME}",
            f"Stop: systemctl --user stop {SERVICE_NAME}",
            f"Restart: systemctl --user restart {SERVICE_NAME}",
            f"Status: systemctl --user status {SERVICE_NAME}",
            f"Logs: journalctl --user -u {SERVICE_NAME} --no-pager",
        ]
    )


def uninstall_summary(summary: dict[str, str]) -> str:
    if summary.get("purged") == "true":
        state_line = f"Platform state deleted: {summary['state_dir']}"
    else:
        state_line = f"Platform state retained at: {summary['state_dir']}"
    return "\n".join(
        [
            "Personal Agent Memory package integration removed.",
            state_line,
            f"Memory libraries retained at: {summary['library_root']}",
            f"Unit: {summary.get('unit_status', 'unknown')}",
            f"Plugin: {summary.get('plugin_status', 'unknown')}",
        ]
    )
