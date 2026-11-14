#!/usr/bin/env python3
"""Validate one payload and emit package metadata shared by deb and rpm."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"package metadata: {message}")


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    payload = args.payload.resolve()
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    project_data = project.get("project")
    if not isinstance(project_data, dict) or not isinstance(project_data.get("version"), str):
        fail("project version is missing")
    version = project_data["version"]
    manifest = read_json(payload / "manifest.json")
    install_prefix = manifest.get("install_prefix")
    entrypoint = manifest.get("entrypoint")
    plugin_manifest_value = manifest.get("plugin_manifest")
    if not all(
        isinstance(item, str) and item
        for item in (install_prefix, entrypoint, plugin_manifest_value)
    ):
        fail("manifest install_prefix, entrypoint, and plugin_manifest are required")
    if install_prefix != "/opt/personal-agent-memory":
        fail("manifest install_prefix does not match the package layout")
    expected_entrypoint = install_prefix + "/bin/personal-agent-memory"
    expected_plugin_manifest = (
        install_prefix + "/plugins/personal-agent-memory/.codex-plugin/plugin.json"
    )
    if entrypoint != expected_entrypoint or plugin_manifest_value != expected_plugin_manifest:
        fail("manifest entrypoint or plugin_manifest does not match the package layout")
    entrypoint_path = payload / entrypoint.lstrip("/")
    plugin_manifest_path = payload / plugin_manifest_value.lstrip("/")
    if not entrypoint_path.is_file() or not entrypoint_path.stat().st_mode & 0o111:
        fail("payload entrypoint is missing or not executable")
    if not plugin_manifest_path.is_file():
        fail("payload plugin manifest is missing")
    plugin_manifest = read_json(plugin_manifest_path)
    versions = [
        version,
        manifest.get("project_version"),
        manifest.get("plugin_version"),
        plugin_manifest.get("version"),
    ]
    if any(not isinstance(item, str) or not item.strip() for item in versions):
        fail("project, payload, and plugin versions must be non-empty strings")
    if len(set(versions)) != 1:
        fail(f"version mismatch: {versions!r}")
    files = sorted(str(path.relative_to(payload)) for path in payload.rglob("*") if path.is_file())
    checksums = {item: hashlib.sha256((payload / item).read_bytes()).hexdigest() for item in files}
    metadata = {
        "schema_version": 1,
        "name": "personal-agent-memory",
        "version": version,
        "payload_prefix": install_prefix,
        "entrypoint": entrypoint,
        "plugin_manifest": plugin_manifest_value,
        "command": "/usr/bin/personal-agent-memory",
        "files": checksums,
        "package_installation": "files-only-no-user-operations",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
