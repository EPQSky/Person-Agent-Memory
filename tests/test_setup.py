from __future__ import annotations

import json
import socket
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from personal_agent_memory.cli import cli

runner = CliRunner()


def _dependencies(monkeypatch, *, shadow: str | None = None) -> None:
    commands = {"node": "/usr/bin/node", "codex": "/usr/bin/codex", "git": "/usr/bin/git"}
    if shadow is not None:
        commands["personal-agent-memory"] = shadow

    def which(command: str) -> str | None:
        return commands.get(command)

    monkeypatch.setattr("personal_agent_memory.provisioning.shutil.which", which)


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_setup_creates_package_state_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    state = tmp_path / "state"
    libraries = tmp_path / "libraries"
    config = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    port = _free_port()

    first = runner.invoke(
        cli,
        ["setup", "--state-dir", str(state), "--library-root", str(libraries), "--port", str(port)],
    )
    assert first.exit_code == 0, first.output
    key = state / "api-key"
    metadata_path = config / "personal-agent-memory/install.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["install_method"] == "package"
    assert metadata["state_dir"] == str(state)
    assert metadata["library_root"] == str(libraries)
    assert "uninstall_ownership" not in metadata
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert stat.S_IMODE(metadata_path.stat().st_mode) == 0o600
    original_key = key.read_text(encoding="utf-8")
    metadata_inode = metadata_path.stat().st_ino

    second = runner.invoke(cli, ["setup"])
    assert second.exit_code == 0, second.output
    assert key.read_text(encoding="utf-8") == original_key
    assert metadata_path.stat().st_ino == metadata_inode
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["port"] == port


def test_setup_missing_dependency_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "personal_agent_memory.provisioning.shutil.which",
        lambda command: None if command == "codex" else f"/usr/bin/{command}",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = runner.invoke(cli, ["setup", "--state-dir", str(tmp_path / "state")])
    assert result.exit_code == 2
    assert "codex" in result.output
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "config").exists()


def test_setup_rejects_script_installation_before_writes(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    config = tmp_path / "config"
    metadata_dir = config / "personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "install.json").write_text(
        json.dumps({"schema_version": 1, "install_method": "script"}), encoding="utf-8"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    state = tmp_path / "state"
    result = runner.invoke(cli, ["setup", "--state-dir", str(state)])
    assert result.exit_code == 2
    assert "script installation" in result.output
    assert not state.exists()


def test_setup_rejects_version_drift_before_writes(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    manifest = tmp_path / "plugin.json"
    manifest.write_text(json.dumps({"version": "9.9.9"}), encoding="utf-8")
    monkeypatch.setenv("PERSONAL_AGENT_MEMORY_PLUGIN_MANIFEST", str(manifest))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    state = tmp_path / "state"
    result = runner.invoke(cli, ["setup", "--state-dir", str(state)])
    assert result.exit_code == 2
    assert "versions must match" in result.output
    assert not state.exists()


def test_setup_rejects_port_conflict_before_writes(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    try:
        result = runner.invoke(
            cli,
            [
                "setup",
                "--state-dir",
                str(tmp_path / "state"),
                "--port",
                str(occupied.getsockname()[1]),
            ],
        )
    finally:
        occupied.close()
    assert result.exit_code == 2
    assert "already in use" in result.output
    assert not (tmp_path / "state").exists()


def test_setup_rejects_shadowed_command_before_writes(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch, shadow="/other/bin/personal-agent-memory")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = runner.invoke(cli, ["setup", "--state-dir", str(tmp_path / "state")])
    assert result.exit_code == 2
    assert "shadows" in result.output
    assert not (tmp_path / "state").exists()


def test_setup_rejects_api_key_symlink_before_writes(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    state = tmp_path / "state"
    state.mkdir()
    external = tmp_path / "external-secret"
    external.write_text("do-not-touch", encoding="utf-8")
    (state / "api-key").symlink_to(external)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = runner.invoke(cli, ["setup", "--state-dir", str(state), "--port", str(_free_port())])
    assert result.exit_code == 2
    assert "symbolic link" in result.output
    assert external.read_text(encoding="utf-8") == "do-not-touch"


def test_setup_rejects_unowned_user_service_before_writes(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    config = tmp_path / "config"
    unit = config / "systemd/user/personal-agent-memory.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Unit]\nDescription=other\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    result = runner.invoke(
        cli, ["setup", "--state-dir", str(tmp_path / "state"), "--port", str(_free_port())]
    )
    assert result.exit_code == 2
    assert "not owned" in result.output
    assert not (tmp_path / "state").exists()


def test_setup_rejects_service_added_after_first_setup(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    config = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    state = tmp_path / "state"
    libraries = tmp_path / "libraries"
    first = runner.invoke(
        cli,
        [
            "setup",
            "--state-dir",
            str(state),
            "--library-root",
            str(libraries),
            "--port",
            str(_free_port()),
        ],
    )
    assert first.exit_code == 0, first.output
    unit = config / "systemd/user/personal-agent-memory.service"
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("[Service]\nExecStart=/foreign\n", encoding="utf-8")
    second = runner.invoke(cli, ["setup", "--port", str(_free_port())])
    assert second.exit_code == 2
    assert "not owned" in second.output


def test_setup_rejects_boolean_metadata_values(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    config = tmp_path / "config"
    metadata_dir = config / "personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    metadata_path = metadata_dir / "install.json"
    metadata_path.write_text(
        json.dumps({"schema_version": True, "install_method": "package", "port": True}),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    result = runner.invoke(cli, ["setup", "--port", str(_free_port())])
    assert result.exit_code == 2
    assert "unsupported installation method" in result.output


def test_setup_rejects_metadata_symlink(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    config = tmp_path / "config"
    metadata_dir = config / "personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    external = tmp_path / "external-install.json"
    external.write_text(
        json.dumps({"schema_version": 2, "install_method": "package", "port": 7331}),
        encoding="utf-8",
    )
    (metadata_dir / "install.json").symlink_to(external)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    result = runner.invoke(cli, ["setup", "--port", str(_free_port())])
    assert result.exit_code == 2
    assert "regular file" in result.output
    assert external.exists()


@pytest.mark.parametrize("option", ["--state-dir", "--library-root"])
def test_setup_rejects_paths_overlapping_metadata_dir(
    tmp_path: Path, monkeypatch, option: str
) -> None:
    _dependencies(monkeypatch)
    config = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    result = runner.invoke(cli, ["setup", option, str(config / "personal-agent-memory")])
    assert result.exit_code == 2
    assert "overlap" in result.output


def test_setup_preserves_existing_directory_modes(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    state = tmp_path / "state"
    libraries = tmp_path / "libraries"
    state.mkdir(mode=0o755)
    libraries.mkdir(mode=0o755)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = runner.invoke(
        cli,
        [
            "setup",
            "--state-dir",
            str(state),
            "--library-root",
            str(libraries),
            "--port",
            str(_free_port()),
        ],
    )
    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(state.stat().st_mode) == 0o755
    assert stat.S_IMODE(libraries.stat().st_mode) == 0o755


def test_setup_rejects_empty_path_argument(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = runner.invoke(cli, ["setup", "--state-dir", ""])
    assert result.exit_code == 2
    assert "absolute path" in result.output
    assert not (tmp_path / "config").exists()
