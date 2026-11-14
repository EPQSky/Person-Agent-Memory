from __future__ import annotations

import json
from pathlib import Path

from personal_agent_memory import provisioning


def test_plugin_manifest_is_found_from_custom_payload_prefix(tmp_path: Path, monkeypatch) -> None:
    prefix = tmp_path / "srv" / "pam"
    package_file = prefix / (
        "venv/lib/python3.12/site-packages/personal_agent_memory/provisioning.py"
    )
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    manifest = prefix / "plugins/personal-agent-memory/.codex-plugin/plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"version": provisioning.__version__}), encoding="utf-8")

    monkeypatch.setattr(provisioning, "__file__", str(package_file))
    monkeypatch.delenv("PERSONAL_AGENT_MEMORY_PLUGIN_MANIFEST", raising=False)
    monkeypatch.delenv("PERSONAL_AGENT_MEMORY_INSTALL_PREFIX", raising=False)
    monkeypatch.delenv("PERSONAL_AGENT_MEMORY_PAYLOAD_MANIFEST", raising=False)

    assert provisioning._plugin_manifest_path() == manifest.resolve()


def test_package_setup_uses_custom_prefix_plugin_manifest(tmp_path: Path, monkeypatch) -> None:
    prefix = tmp_path / "srv" / "pam"
    package_file = prefix / (
        "venv/lib/python3.12/site-packages/personal_agent_memory/provisioning.py"
    )
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    manifest = prefix / "plugins/personal-agent-memory/.codex-plugin/plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"version": provisioning.__version__}), encoding="utf-8")

    home = tmp_path / "home"
    config = tmp_path / "config"
    monkeypatch.setattr(provisioning, "__file__", str(package_file))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("PERSONAL_AGENT_MEMORY_INSTALL_PREFIX", str(prefix))
    monkeypatch.setattr(
        provisioning.shutil,
        "which",
        lambda command: f"/usr/bin/{command}",
    )
    monkeypatch.setattr(provisioning, "_ensure_port_available", lambda _: None)
    monkeypatch.setattr(provisioning, "_check_command_shadowing", lambda: None)

    result = provisioning.provision_package(
        state_dir=str(tmp_path / "state"),
        library_root=str(tmp_path / "libraries"),
        port=17331,
    )

    assert result["version"] == provisioning.__version__
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["plugin_version"] == provisioning.__version__
    assert Path(result["api_key_path"]).is_file()
