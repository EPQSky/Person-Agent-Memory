from __future__ import annotations

import json
import socket
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from personal_agent_memory import service_registration
from personal_agent_memory.cli import cli
from personal_agent_memory.provisioning import ProvisioningError

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


def test_setup_registers_plugin_service_and_health_summary(tmp_path: Path, monkeypatch) -> None:
    _dependencies(monkeypatch)
    config = tmp_path / "config"
    state = tmp_path / "state"
    libraries = tmp_path / "libraries"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(service_registration, "_service_tools_available", lambda: True)
    monkeypatch.setattr("personal_agent_memory.provisioning._check_command_shadowing", lambda: None)
    monkeypatch.setattr(service_registration.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(service_registration, "_health_check", lambda *args, **kwargs: None)
    marketplace = tmp_path / "marketplace"
    plugin_path = tmp_path / "plugin"
    marketplace.mkdir()
    plugin_path.mkdir()
    monkeypatch.setattr(service_registration, "_marketplace_source", lambda: marketplace)

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        calls.append((command, args))
        if command == "codex" and args[:3] == ["plugin", "marketplace", "add"]:
            return json.dumps(
                {"marketplaceName": "personal-agent-memory", "installedRoot": str(marketplace)}
            )
        if command == "codex" and args[:3] == ["plugin", "marketplace", "list"]:
            return json.dumps(
                {"marketplaces": [{"name": "personal-agent-memory", "root": str(marketplace)}]}
            )
        if command == "codex" and args[:3] == [
            "plugin",
            "add",
            "personal-agent-memory@personal-agent-memory",
        ]:
            return json.dumps(
                {
                    "pluginId": "personal-agent-memory@personal-agent-memory",
                    "version": "0.2.1",
                    "installedPath": str(plugin_path),
                }
            )
        if command == "codex" and args[:2] == ["plugin", "list"]:
            return json.dumps(
                {
                    "installed": [
                        {
                            "pluginId": "personal-agent-memory@personal-agent-memory",
                            "version": "0.2.1",
                            "installedPath": str(plugin_path),
                            "source": {"path": str(marketplace)},
                            "enabled": True,
                        }
                    ]
                }
            )
        if command == "systemctl" and args[:2] == ["--user", "is-active"]:
            return "active"
        return ""

    monkeypatch.setattr(service_registration, "_run", fake_run)
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
    unit = config / "systemd/user/personal-agent-memory.service"
    content = unit.read_text(encoding="utf-8")
    assert f'--state-dir "{state}"' in content
    assert f'--library-root "{libraries}"' in content
    assert "Restart=on-failure" in content
    assert "RestartSec=2" in content
    assert "systemctl --user enable --now personal-agent-memory.service" in result.output or any(
        command == "systemctl" and args[1:3] == ["enable", "--now"] for command, args in calls
    )
    assert "Bearer" not in result.output
    assert not any(
        command == "codex" and args[:3] == ["plugin", "marketplace", "add"]
        for command, args in calls
    )
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_inode = metadata_path.stat().st_ino
    unit_inode = unit.stat().st_ino
    rerun = runner.invoke(cli, ["setup"])
    assert rerun.exit_code == 0, rerun.output
    assert metadata_path.stat().st_ino == metadata_inode
    assert unit.stat().st_ino == unit_inode


def test_setup_uninstall_removes_owned_service_and_plugin_but_keeps_data(
    tmp_path: Path, monkeypatch
) -> None:
    _dependencies(monkeypatch)
    config = tmp_path / "config"
    state = tmp_path / "state"
    libraries = tmp_path / "libraries"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(service_registration, "_service_tools_available", lambda: True)
    monkeypatch.setattr("personal_agent_memory.provisioning._check_command_shadowing", lambda: None)
    monkeypatch.setattr(service_registration.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(service_registration, "_health_check", lambda *args, **kwargs: None)
    marketplace = tmp_path / "marketplace"
    plugin_path = tmp_path / "codex/plugins/plugin"
    marketplace.mkdir()
    plugin_path.mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(service_registration, "_marketplace_source", lambda: marketplace)

    calls: list[tuple[str, list[str]]] = []

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        calls.append((command, args))
        if command == "codex" and args[:3] == ["plugin", "marketplace", "add"]:
            return json.dumps(
                {"marketplaceName": "personal-agent-memory", "installedRoot": str(marketplace)}
            )
        if command == "codex" and args[:3] == ["plugin", "marketplace", "list"]:
            return json.dumps(
                {"marketplaces": [{"name": "personal-agent-memory", "root": str(marketplace)}]}
            )
        if command == "codex" and args[:3] == [
            "plugin",
            "add",
            "personal-agent-memory@personal-agent-memory",
        ]:
            return json.dumps(
                {
                    "pluginId": "personal-agent-memory@personal-agent-memory",
                    "version": "0.2.1",
                    "installedPath": str(plugin_path),
                }
            )
        if command == "codex" and args[:2] == ["plugin", "list"]:
            return json.dumps(
                {
                    "installed": [
                        {
                            "pluginId": "personal-agent-memory@personal-agent-memory",
                            "version": "0.2.1",
                            "installedPath": str(plugin_path),
                            "source": {"path": str(marketplace)},
                            "enabled": True,
                        }
                    ]
                }
            )
        if command == "systemctl" and args[:2] == ["--user", "is-active"]:
            return "active"
        return ""

    monkeypatch.setattr(service_registration, "_run", fake_run)
    setup = runner.invoke(
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
    assert setup.exit_code == 0, setup.output
    libraries.joinpath("memory.md").write_text("keep", encoding="utf-8")
    removed = runner.invoke(cli, ["setup", "--uninstall"])
    assert removed.exit_code == 0, removed.output
    assert not (config / "systemd/user/personal-agent-memory.service").exists()
    assert state.exists()
    assert libraries.joinpath("memory.md").read_text(encoding="utf-8") == "keep"
    daemon_reloads = [
        index
        for index, (command, args) in enumerate(calls)
        if command == "systemctl" and args[1:] == ["daemon-reload"]
    ]
    assert len(daemon_reloads) >= 3


def test_service_unit_quotes_systemd_special_characters(tmp_path: Path) -> None:
    from personal_agent_memory.service_registration import _unit_content

    content = _unit_content(
        r"/opt/pam\\bin$%\"agent",
        tmp_path / 'state\\$%"dir',
        tmp_path / 'library%$"dir',
        7331,
    )
    assert 'ExecStart="' in content
    assert r'--state-dir "' in content
    assert r"\\" in content and r"\"" in content
    assert "$$" in content and "%%" in content


def test_service_unit_symlink_is_rejected(tmp_path: Path) -> None:
    from personal_agent_memory.service_registration import _write_unit

    target = tmp_path / "target"
    target.write_text("user-owned", encoding="utf-8")
    unit = tmp_path / "personal-agent-memory.service"
    unit.symlink_to(target)
    with pytest.raises(Exception, match="symbolic|regular"):
        _write_unit(unit, "[Service]\n")
    assert target.read_text(encoding="utf-8") == "user-owned"


def test_plugin_list_requires_exactly_one_marketplace_entry(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "marketplace"
    installed = tmp_path / "plugin"
    source.mkdir()
    installed.mkdir()
    monkeypatch.setattr(service_registration, "_marketplace_source", lambda: source)

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        if args[:3] == ["plugin", "marketplace", "add"]:
            return json.dumps(
                {"marketplaceName": "personal-agent-memory", "installedRoot": str(source)}
            )
        if args[:3] == ["plugin", "marketplace", "list"]:
            return json.dumps(
                {
                    "marketplaces": [
                        {"name": "personal-agent-memory", "root": str(source)},
                        {"name": "personal-agent-memory", "root": str(source)},
                    ]
                }
            )
        raise AssertionError(args)

    monkeypatch.setattr(service_registration, "_run", fake_run)
    with pytest.raises(ProvisioningError, match="exactly one"):
        service_registration.register_plugin(version="0.2.1")


def test_plugin_remove_not_found_is_idempotent_but_other_errors_propagate(monkeypatch) -> None:
    monkeypatch.setattr(
        service_registration,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProvisioningError("plugin not found")),
    )
    service_registration._plugin_remove_idempotent()
    monkeypatch.setattr(
        service_registration,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProvisioningError("permission denied")),
    )
    with pytest.raises(ProvisioningError, match="permission denied"):
        service_registration._plugin_remove_idempotent()


def test_service_failure_restores_metadata_unit_and_plugin(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    metadata_path = config / "personal-agent-memory/install.json"
    state = tmp_path / "state"
    libraries = tmp_path / "libraries"
    plugin = tmp_path / "plugin"
    source = tmp_path / "marketplace"
    state.mkdir()
    libraries.mkdir()
    plugin.mkdir()
    source.mkdir()
    (state / "api-key").write_text("secret\n", encoding="utf-8")
    metadata_path.parent.mkdir(parents=True)
    original = {
        "schema_version": 2,
        "install_method": "package",
        "installed_version": "0.2.1",
        "plugin_version": "0.2.1",
        "state_dir": str(state),
        "state_dir_ownership": "package-managed",
        "library_root": str(libraries),
        "library_root_ownership": "user-content-never-delete",
        "api_key_path": str(state / "api-key"),
        "port": 7331,
    }
    metadata_path.write_text(json.dumps(original), encoding="utf-8")
    summary = {
        "version": "0.2.1",
        "state_dir": str(state),
        "library_root": str(libraries),
        "api_key_path": str(state / "api-key"),
        "port": 7331,
        "metadata_path": str(metadata_path),
    }
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setenv("PERSONAL_AGENT_MEMORY_DAEMON", "/opt/pam/bin/personal-agent-memory")
    monkeypatch.setattr(service_registration, "_require_integration_tools", lambda: None)
    monkeypatch.setattr(service_registration, "_plugin_present_before_setup", lambda: False)
    monkeypatch.setattr(service_registration, "_marketplace_source", lambda: source)

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        calls.append((command, args))
        if args[:3] == ["plugin", "marketplace", "add"]:
            return json.dumps(
                {"marketplaceName": "personal-agent-memory", "installedRoot": str(source)}
            )
        if args[:3] == ["plugin", "marketplace", "list"]:
            return json.dumps(
                {"marketplaces": [{"name": "personal-agent-memory", "root": str(source)}]}
            )
        if args[:3] == ["plugin", "add", "personal-agent-memory@personal-agent-memory"]:
            return json.dumps(
                {
                    "pluginId": "personal-agent-memory@personal-agent-memory",
                    "version": "0.2.1",
                    "installedPath": str(plugin),
                }
            )
        if args[:2] == ["plugin", "list"]:
            return json.dumps(
                {
                    "installed": [
                        {
                            "pluginId": "personal-agent-memory@personal-agent-memory",
                            "version": "0.2.1",
                            "installedPath": str(plugin),
                            "source": {"path": str(source)},
                            "enabled": True,
                        }
                    ]
                }
            )
        if args[1:3] == ["is-enabled", "personal-agent-memory.service"]:
            return "enabled"
        if args[1:3] == ["enable", "--now"]:
            raise ProvisioningError("systemd start failed")
        return ""

    monkeypatch.setattr(service_registration, "_run", fake_run)
    with pytest.raises(ProvisioningError, match="systemd start failed"):
        service_registration.integrate_package_setup(summary)
    assert not (config / "systemd/user/personal-agent-memory.service").exists()
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == original
    assert any(args[1:3] == ["stop", "personal-agent-memory.service"] for _, args in calls)
    assert any(args[1:3] == ["enable", "personal-agent-memory.service"] for _, args in calls)
    assert any(
        args[:3] == ["plugin", "remove", "personal-agent-memory@personal-agent-memory"]
        for _, args in calls
    )


def test_setup_exposes_marketplace_adoption_flag() -> None:
    result = runner.invoke(cli, ["setup", "--help"])
    assert result.exit_code == 0
    assert "--adopt-marketplace" in result.output


def test_register_plugin_adopts_foreign_marketplace_only_when_requested(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "marketplace"
    foreign = tmp_path / "foreign"
    plugin = tmp_path / "plugin"
    source.mkdir()
    foreign.mkdir()
    plugin.mkdir()
    monkeypatch.setattr(service_registration, "_marketplace_source", lambda: source)
    marketplace_lists = iter(
        [
            {"marketplaces": [{"name": "personal-agent-memory", "root": str(foreign)}]},
            {"marketplaces": [{"name": "personal-agent-memory", "root": str(source)}]},
        ]
    )
    calls: list[tuple[str, list[str]]] = []

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        calls.append((command, args))
        if args[:3] == ["plugin", "marketplace", "list"]:
            return json.dumps(next(marketplace_lists))
        if args[:3] == ["plugin", "marketplace", "add"]:
            return json.dumps(
                {"marketplaceName": "personal-agent-memory", "installedRoot": str(source)}
            )
        if args[:3] == ["plugin", "marketplace", "remove"]:
            return "{}"
        if args[:3] == ["plugin", "add", "personal-agent-memory@personal-agent-memory"]:
            return json.dumps(
                {
                    "pluginId": "personal-agent-memory@personal-agent-memory",
                    "version": "0.2.1",
                    "installedPath": str(plugin),
                }
            )
        if args[:2] == ["plugin", "list"]:
            return json.dumps(
                {
                    "installed": [
                        {
                            "pluginId": "personal-agent-memory@personal-agent-memory",
                            "version": "0.2.1",
                            "installedPath": str(plugin),
                            "source": {"path": str(source)},
                            "enabled": True,
                        }
                    ]
                }
            )
        raise AssertionError(args)

    monkeypatch.setattr(service_registration, "_run", fake_run)
    with pytest.raises(ProvisioningError, match="adopt-marketplace"):
        service_registration.register_plugin(version="0.2.1")
    # Recreate responses for the explicit adoption attempt.
    marketplace_lists = iter(
        [
            {"marketplaces": [{"name": "personal-agent-memory", "root": str(foreign)}]},
            {"marketplaces": [{"name": "personal-agent-memory", "root": str(source)}]},
        ]
    )
    service_registration.register_plugin(version="0.2.1", adopt_marketplace=True)
    assert any(args[:3] == ["plugin", "marketplace", "remove"] for _, args in calls)


def test_service_unit_symlink_parent_is_rejected(tmp_path: Path) -> None:
    from personal_agent_memory.service_registration import _write_unit

    real_parent = tmp_path / "real-user"
    real_parent.mkdir()
    linked_parent = tmp_path / "user"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    unit = linked_parent / "personal-agent-memory.service"
    with pytest.raises(ProvisioningError, match="parent"):
        _write_unit(unit, "[Service]\n")
    assert not (real_parent / "personal-agent-memory.service").exists()


def test_purge_rejects_state_symlink_without_deleting_target(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    state_target = tmp_path / "state-target"
    state_target.mkdir()
    state_link = tmp_path / "state-link"
    state_link.symlink_to(state_target, target_is_directory=True)
    library = tmp_path / "libraries"
    library.mkdir()
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "install_method": "package",
                "state_dir": str(state_link),
                "state_dir_ownership": "package-managed",
                "library_root": str(library),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(service_registration, "_plugin_remove_idempotent", lambda: None)
    monkeypatch.setattr(service_registration, "_service_tools_available", lambda: False)
    result = runner.invoke(cli, ["setup", "--uninstall", "--purge"])
    assert result.exit_code == 2
    assert "symbolic-link" in result.output
    assert state_target.exists()
    assert metadata_path.exists()


def test_core_rejects_dangling_unit_symlink(tmp_path: Path) -> None:
    from personal_agent_memory.provisioning import _check_unit_conflict

    unit = tmp_path / "personal-agent-memory.service"
    unit.symlink_to(tmp_path / "missing-unit")
    with pytest.raises(ProvisioningError, match="regular non-symlink"):
        _check_unit_conflict(unit, None)


def test_marketplace_source_supports_release_layouts(tmp_path: Path, monkeypatch) -> None:
    layouts = [
        (
            tmp_path / "usr/share/personal-agent-memory/plugin.json",
            tmp_path / "usr/share/personal-agent-memory",
        ),
        (
            tmp_path
            / "opt/personal-agent-memory/plugins/personal-agent-memory/.codex-plugin/plugin.json",
            tmp_path / "opt/personal-agent-memory",
        ),
        (
            tmp_path / "release/plugins/personal-agent-memory/.codex-plugin/plugin.json",
            tmp_path / "release",
        ),
    ]
    for manifest, expected in layouts:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            service_registration, "_plugin_manifest_path", lambda manifest=manifest: manifest
        )
        assert service_registration._marketplace_source() == expected


def test_adoption_failure_restores_foreign_marketplace(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    state = tmp_path / "state"
    library = tmp_path / "library"
    source = tmp_path / "new-marketplace"
    foreign = tmp_path / "foreign-marketplace"
    state.mkdir()
    library.mkdir()
    source.mkdir()
    foreign.mkdir()
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    original = {
        "schema_version": 2,
        "install_method": "package",
        "state_dir": str(state),
        "state_dir_ownership": "package-managed",
        "library_root": str(library),
    }
    metadata_path.write_text(json.dumps(original), encoding="utf-8")
    summary = {
        "version": "0.2.1",
        "state_dir": str(state),
        "library_root": str(library),
        "api_key_path": str(state / "api-key"),
        "port": 7331,
        "metadata_path": str(metadata_path),
    }
    calls: list[tuple[str, list[str]]] = []
    marketplace_lists = iter(
        [
            {"marketplaces": [{"name": "personal-agent-memory", "root": str(foreign)}]},
            {"marketplaces": [{"name": "personal-agent-memory", "root": str(foreign)}]},
        ]
    )
    monkeypatch.setenv("PERSONAL_AGENT_MEMORY_DAEMON", "/opt/pam/bin/personal-agent-memory")
    monkeypatch.setattr(service_registration, "_require_integration_tools", lambda: None)
    monkeypatch.setattr(service_registration, "_plugin_present_before_setup", lambda: False)
    monkeypatch.setattr(service_registration, "_service_enabled_before", lambda: False)
    monkeypatch.setattr(service_registration, "_marketplace_source", lambda: source)

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        calls.append((command, args))
        if args[:3] == ["plugin", "marketplace", "list"]:
            return json.dumps(next(marketplace_lists))
        if args[:3] == ["plugin", "marketplace", "add"]:
            root = str(foreign) if str(foreign) in args else str(source)
            return json.dumps({"marketplaceName": "personal-agent-memory", "installedRoot": root})
        if args[:3] == ["plugin", "marketplace", "remove"]:
            return "{}"
        if args[:3] == ["plugin", "remove", "personal-agent-memory@personal-agent-memory"]:
            return "{}"
        return ""

    def fail_register(
        *,
        version: str,
        adopt_marketplace: bool = False,
        on_plugin_added=None,
    ) -> dict[str, str]:
        fake_run("codex", ["plugin", "marketplace", "remove", "personal-agent-memory", "--json"])
        fake_run("codex", ["plugin", "marketplace", "add", str(source), "--json"])
        raise ProvisioningError("plugin add failed")

    monkeypatch.setattr(service_registration, "_run", fake_run)
    monkeypatch.setattr(service_registration, "register_plugin", fail_register)
    with pytest.raises(ProvisioningError, match="plugin add failed"):
        service_registration.integrate_package_setup(summary, adopt_marketplace=True)
    restored_adds = [
        args
        for command, args in calls
        if command == "codex" and args[:3] == ["plugin", "marketplace", "add"]
    ]
    assert restored_adds[-1][3] == str(foreign)
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == original


def test_new_marketplace_is_removed_when_later_setup_fails(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    state = tmp_path / "state"
    library = tmp_path / "library"
    source = tmp_path / "marketplace"
    plugin = tmp_path / "plugin"
    state.mkdir()
    library.mkdir()
    source.mkdir()
    plugin.mkdir()
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    original = {
        "schema_version": 2,
        "install_method": "package",
        "state_dir": str(state),
        "state_dir_ownership": "package-managed",
        "library_root": str(library),
        "api_key_path": str(state / "api-key"),
    }
    metadata_path.write_text(json.dumps(original), encoding="utf-8")
    summary = {
        "version": "0.2.1",
        "state_dir": str(state),
        "library_root": str(library),
        "api_key_path": str(state / "api-key"),
        "port": 7331,
        "metadata_path": str(metadata_path),
    }
    marketplace_exists = False
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setenv("PERSONAL_AGENT_MEMORY_DAEMON", "/opt/pam/bin/personal-agent-memory")
    monkeypatch.setattr(service_registration, "_require_integration_tools", lambda: None)
    monkeypatch.setattr(service_registration, "_plugin_present_before_setup", lambda: False)
    monkeypatch.setattr(service_registration, "_service_enabled_before", lambda: False)
    monkeypatch.setattr(service_registration, "_marketplace_source", lambda: source)

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        nonlocal marketplace_exists
        calls.append((command, args))
        if args[:3] == ["plugin", "marketplace", "list"]:
            entries = (
                [{"name": "personal-agent-memory", "root": str(source)}]
                if marketplace_exists
                else []
            )
            return json.dumps({"marketplaces": entries})
        if args[:3] == ["plugin", "marketplace", "add"]:
            marketplace_exists = True
            return json.dumps(
                {"marketplaceName": "personal-agent-memory", "installedRoot": str(source)}
            )
        if args[:3] == ["plugin", "marketplace", "remove"]:
            marketplace_exists = False
            return "{}"
        if args[:3] == ["plugin", "add", "personal-agent-memory@personal-agent-memory"]:
            return json.dumps(
                {
                    "pluginId": "personal-agent-memory@personal-agent-memory",
                    "version": "0.2.1",
                    "installedPath": str(plugin),
                }
            )
        if args[:2] == ["plugin", "list"]:
            return json.dumps(
                {
                    "installed": [
                        {
                            "pluginId": "personal-agent-memory@personal-agent-memory",
                            "version": "0.2.1",
                            "installedPath": str(plugin),
                            "source": {"path": str(source)},
                            "enabled": True,
                        }
                    ]
                }
            )
        if args[1:3] == ["enable", "--now"]:
            raise ProvisioningError("systemd start failed")
        return ""

    monkeypatch.setattr(service_registration, "_run", fake_run)
    with pytest.raises(ProvisioningError, match="systemd start failed"):
        service_registration.integrate_package_setup(summary)
    assert not marketplace_exists
    assert any(args[:3] == ["plugin", "marketplace", "remove"] for _, args in calls)
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == original


def test_uninstall_old_core_metadata_preserves_same_name_plugin(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config"
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "install_method": "package",
                "state_dir": str(tmp_path / "state"),
                "library_root": str(tmp_path / "library"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(service_registration, "_service_tools_available", lambda: False)
    monkeypatch.setattr(
        service_registration,
        "_plugin_remove_idempotent",
        lambda: (_ for _ in ()).throw(AssertionError("must preserve plugin")),
    )
    result = runner.invoke(cli, ["setup", "--uninstall"])
    assert result.exit_code == 0, result.output
    assert "ownership metadata is unavailable" in result.output


def test_uninstall_preserves_external_same_name_plugin(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    source = tmp_path / "owned-source"
    foreign = tmp_path / "foreign-source"
    source.mkdir()
    foreign.mkdir()
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "install_method": "package",
                "state_dir": str(tmp_path / "state"),
                "library_root": str(tmp_path / "library"),
                "plugin_id": "personal-agent-memory@personal-agent-memory",
                "plugin_ownership": "package-managed",
                "plugin_registration_source": str(source),
                "plugin_install_path": str(source / "plugin"),
                "plugin_version": "0.2.1",
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(service_registration, "_service_tools_available", lambda: False)

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        calls.append(args)
        if args[:2] == ["plugin", "list"]:
            return json.dumps(
                {
                    "installed": [
                        {
                            "pluginId": "personal-agent-memory@personal-agent-memory",
                            "source": {"path": str(foreign)},
                            "installedPath": str(source / "plugin"),
                            "version": "0.2.1",
                        }
                    ]
                }
            )
        raise AssertionError(args)

    monkeypatch.setattr(service_registration, "_run", fake_run)
    result = runner.invoke(cli, ["setup", "--uninstall"])
    assert result.exit_code == 0, result.output
    assert "source does not match" in result.output
    assert not any(args[:2] == ["plugin", "remove"] for args in calls)


def test_purge_rejects_library_symlink(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    state = tmp_path / "state"
    target = tmp_path / "library-target"
    state.mkdir()
    target.mkdir()
    library = tmp_path / "library-link"
    library.symlink_to(target, target_is_directory=True)
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "install_method": "package",
                "state_dir": str(state),
                "state_dir_ownership": "package-managed",
                "library_root": str(library),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(service_registration, "_service_tools_available", lambda: False)
    monkeypatch.setattr(service_registration, "_plugin_remove_idempotent", lambda: None)
    result = runner.invoke(cli, ["setup", "--uninstall", "--purge"])
    assert result.exit_code == 2
    assert "symbolic-link" in result.output
    assert target.exists()
    assert state.exists()


def test_uninstall_summary_reports_purge_state() -> None:
    assert "Platform state deleted" in service_registration.uninstall_summary(
        {"state_dir": "/tmp/state", "library_root": "/tmp/library", "purged": "true"}
    )
    assert "Platform state retained" in service_registration.uninstall_summary(
        {"state_dir": "/tmp/state", "library_root": "/tmp/library", "purged": "false"}
    )


def test_health_check_rejects_unexpected_loopback_port(tmp_path: Path, monkeypatch) -> None:
    key = tmp_path / "api-key"
    key.write_text("secret\n", encoding="utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self) -> str:
            return "http://127.0.0.1:9999/health/live"

        def read(self, *args):
            return b'{"status":"ok"}'

    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") == "Bearer secret"
            return Response()

    monkeypatch.setattr(service_registration, "_NO_REDIRECT_OPENER", Opener())
    with pytest.raises(ProvisioningError, match="unexpected loopback address"):
        service_registration._health_check(key, 7331, timeout=0.1)


def test_health_redirect_handler_never_creates_redirect_request() -> None:
    handler = service_registration._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "http://evil.test/") is None


def test_mcp_health_uses_no_redirect_and_rejects_wrong_port(tmp_path: Path, monkeypatch) -> None:
    key = tmp_path / "api-key"
    key.write_text("secret\n", encoding="utf-8")

    class Response:
        def __init__(self, url: str, payload: dict[str, str]):
            self.url = url
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self) -> str:
            return self.url

        def read(self, *args):
            return json.dumps(self.payload).encode()

    responses = iter(
        [
            Response("http://127.0.0.1:7331/health/live", {"status": "ok"}),
            Response("http://127.0.0.1:9999/mcp/health", {"memory_kernel": "mem2"}),
        ]
    )
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return next(responses)

    monkeypatch.setattr(service_registration, "_NO_REDIRECT_OPENER", Opener())
    with pytest.raises(ProvisioningError, match="unexpected loopback address"):
        service_registration._health_check(key, 7331, timeout=0.2)
    assert len(requests) == 2
    assert all(request.get_header("Authorization") == "Bearer secret" for request in requests)


def test_uninstall_does_not_stop_replaced_external_unit(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config"
    unit = tmp_path / "external.service"
    unit.write_text("[Service]\nExecStart=/external\n", encoding="utf-8")
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "install_method": "package",
                "state_dir": str(tmp_path / "state"),
                "library_root": str(tmp_path / "library"),
                "unit_path": str(unit),
                "unit_ownership": "package-managed",
                "unit_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(service_registration, "_service_tools_available", lambda: True)
    monkeypatch.setattr(
        service_registration,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not stop or disable external unit")
        ),
    )
    result = runner.invoke(cli, ["setup", "--uninstall"])
    assert result.exit_code == 0, result.output
    assert unit.exists()
    assert "unit path is not the package service path" in result.output


def test_foreign_non_adoption_failure_does_not_remove_plugin_or_marketplace(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config"
    state = tmp_path / "state"
    library = tmp_path / "library"
    source = tmp_path / "source"
    foreign = tmp_path / "foreign"
    state.mkdir()
    library.mkdir()
    source.mkdir()
    foreign.mkdir()
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "install_method": "package",
                "state_dir": str(state),
                "library_root": str(library),
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "version": "0.2.1",
        "state_dir": str(state),
        "library_root": str(library),
        "api_key_path": str(state / "api-key"),
        "port": 7331,
        "metadata_path": str(metadata_path),
    }
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setenv("PERSONAL_AGENT_MEMORY_DAEMON", "/opt/pam/bin/personal-agent-memory")
    monkeypatch.setattr(service_registration, "_require_integration_tools", lambda: None)
    monkeypatch.setattr(service_registration, "_marketplace_source", lambda: source)
    monkeypatch.setattr(service_registration, "_marketplace_previous_source", lambda: foreign)
    monkeypatch.setattr(service_registration, "_plugin_present_before_setup", lambda: True)
    monkeypatch.setattr(service_registration, "_service_enabled_before", lambda: False)

    def fail_register(**kwargs):
        kwargs["on_plugin_added"]()
        raise ProvisioningError("foreign marketplace")

    monkeypatch.setattr(service_registration, "register_plugin", fail_register)

    def fake_run(command: str, args: list[str], **kwargs) -> str:
        calls.append((command, args))
        return ""

    monkeypatch.setattr(service_registration, "_run", fake_run)
    with pytest.raises(ProvisioningError, match="foreign marketplace"):
        service_registration.integrate_package_setup(summary, adopt_marketplace=False)
    assert not any(
        args[:3] == ["plugin", "remove", "personal-agent-memory@personal-agent-memory"]
        for _, args in calls
    )
    assert not any(args[:3] == ["plugin", "marketplace", "remove"] for _, args in calls)


def test_uninstall_preserves_plugin_when_recorded_version_missing(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config"
    source = tmp_path / "source"
    plugin = tmp_path / "plugin"
    source.mkdir()
    plugin.mkdir()
    metadata_path = config / "personal-agent-memory/install.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "install_method": "package",
                "state_dir": str(tmp_path / "state"),
                "library_root": str(tmp_path / "library"),
                "plugin_id": "personal-agent-memory@personal-agent-memory",
                "plugin_ownership": "package-managed",
                "plugin_registration_source": str(source),
                "plugin_install_path": str(plugin),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(service_registration, "_service_tools_available", lambda: False)
    monkeypatch.setattr(
        service_registration,
        "_plugin_remove_idempotent",
        lambda: (_ for _ in ()).throw(AssertionError("must preserve plugin")),
    )
    result = runner.invoke(cli, ["setup", "--uninstall"])
    assert result.exit_code == 0, result.output
    assert "ownership metadata is unavailable" in result.output
