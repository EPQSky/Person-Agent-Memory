from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from personal_agent_memory.state import PlatformState

ROOT = Path(__file__).parents[1]


def executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def plugin_tree_sha256(root: Path) -> str:
    entries: list[tuple[bytes, Path, str]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in os.scandir(directory):
            path = Path(child.path)
            relative = path.relative_to(root)
            relative_bytes = relative.as_posix().encode("utf-8")
            info = child.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                kind = "directory"
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                kind = "file"
            else:
                raise AssertionError(f"unexpected plugin fixture entry: {relative}")
            entries.append((relative_bytes, path, kind))
    digest = hashlib.sha256(b"personal-agent-memory-plugin-tree-v1\0")
    for relative_bytes, path, kind in sorted(entries, key=lambda item: item[0]):
        digest.update(b"D" if kind == "directory" else b"F")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        if kind == "file":
            contents = path.read_bytes()
            digest.update(len(contents).to_bytes(8, "big"))
            digest.update(contents)
    return digest.hexdigest()


def embedded_plugin_tree_digest(script_name: str) -> tuple[object, type[BaseException]]:
    script = (ROOT / script_name).read_text(encoding="utf-8")
    function_start = script.index("def plugin_tree_sha256")
    start = script.rindex("MAX_ENTRIES = 4096", 0, function_start)
    end_marker = "\n\ndef emit" if script_name == "uninstall.sh" else "\n\ntarget_version ="
    namespace: dict[str, object] = {
        "hashlib": hashlib,
        "os": os,
        "Path": Path,
        "stat": stat,
    }
    exec(script[start : script.index(end_marker, start)], namespace)
    error_type = namespace.get("UnverifiedTree", SystemExit)
    assert isinstance(error_type, type) and issubclass(error_type, BaseException)
    return namespace["plugin_tree_sha256"], error_type


def preflight_test_environment(
    tmp_path: Path, *, failure: str | None = None
) -> tuple[dict[str, str], Path, Path]:
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    command_log = tmp_path / "commands.log"
    home.mkdir()
    fake_bin.mkdir()

    commands = {
        "uname": """case "$1" in -s) echo Linux ;; -m) echo x86_64 ;; esac\n""",
        "systemctl": "exit 0\n",
        "python3": f'exec "{sys.executable}" "$@"\n',
        "python3.13": "exit 1\n",
        "python3.12": "exit 1\n",
        "python3.11": "exit 1\n",
        "git": "echo 'git version 2.43.0'\n",
        "node": "echo 'v20.0.0'\n",
        "codex": """case "$*" in
  '--version') echo 'codex-cli 1.0.00' ;;
  'plugin add --help'|'plugin marketplace add --help'|'plugin marketplace list --help'|\
  'plugin marketplace remove --help'|'plugin list --help') echo 'Usage: --json' ;;
  'plugin marketplace list --json') echo '{"marketplaces":[]}' ;;
esac
""",
        "uv": f"""case "$*" in
  '--version') echo 'uv 0.8.0' ;;
  'tool dir --bin') printf '%s\\n' "{fake_bin}" ;;
  *) exit 0 ;;
esac
""",
    }
    if failure == "operating-system":
        commands["uname"] = """case "$1" in -s) echo Darwin ;; -m) echo x86_64 ;; esac\n"""
    elif failure == "architecture":
        commands["uname"] = """case "$1" in -s) echo Linux ;; -m) echo aarch64 ;; esac\n"""
    elif failure == "node-capabilities":
        commands["node"] = """[ "$1" = '--version' ] && exit 0\nexit 1\n"""
    elif failure == "codex-plugin":
        commands["codex"] = """case "$*" in
  '--version'|'plugin --help'|'plugin marketplace --help') exit 0 ;;
  *) exit 1 ;;
esac
"""
    elif failure in commands:
        commands[failure] = "exit 1\n"

    for name, body in commands.items():
        executable(
            fake_bin / name,
            f'#!/bin/sh\nprintf \'{name} %s\\n\' "$*" >>"{command_log}"\n{body}',
        )
    for name in ("sudo", "apt", "apt-get", "dnf", "yum", "pacman", "zypper"):
        executable(
            fake_bin / name,
            f'#!/bin/sh\nprintf \'{name} %s\\n\' "$*" >>"{command_log}"\nexit 99\n',
        )

    return (
        {
            **os.environ,
            "HOME": str(home),
            "CODEX_HOME": str(home / ".codex"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        },
        home,
        command_log,
    )


def uninstall_test_environment(
    tmp_path: Path,
    *,
    marketplace_ownership: str = "installer-managed",
    shared_marketplace: bool = False,
    valid_state_marker: bool = True,
) -> tuple[dict[str, str], dict[str, Path], Path]:
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    config_dir = home / ".config/personal-agent-memory"
    config_path = config_dir / "install.json"
    state_dir = tmp_path / "platform-state"
    library_root = tmp_path / "memory-libraries"
    stable_uninstaller = fake_bin / "personal-agent-memory-uninstall"
    unit_path = home / ".config/systemd/user/personal-agent-memory.service"
    command_log = tmp_path / "uninstall-commands.log"
    marketplace_source = tmp_path / "release-marketplace"
    plugin_registration_source = marketplace_source / "plugins/personal-agent-memory"
    managed_codex_home = home / ".codex"
    managed_plugin_source = (
        managed_codex_home
        / "plugins/cache/personal-agent-memory/personal-agent-memory/0.1.0"
    )
    tool_environment = tmp_path / "tool-environment"
    token = "a" * 64
    for path in (
        home,
        fake_bin,
        config_dir,
        state_dir,
        library_root,
        marketplace_source,
        managed_codex_home,
    ):
        path.mkdir(parents=True)
    (state_dir / "api-key").write_text("preserved-secret\n", encoding="utf-8")
    (library_root / "memory.md").write_text("# Authoritative memory\n", encoding="utf-8")
    (home / ".config/systemd/user").mkdir(parents=True)
    unit_path.write_text("[Service]\n", encoding="utf-8")
    (tmp_path / "service-active").write_text("active\n", encoding="utf-8")
    (tmp_path / "service-enabled").write_text("enabled\n", encoding="utf-8")
    (tmp_path / "plugin-present").write_text("present\n", encoding="utf-8")
    (tmp_path / "marketplace-present").write_text(str(marketplace_source), encoding="utf-8")
    tool_environment.write_text("installed\n", encoding="utf-8")
    for plugin_root in (plugin_registration_source, managed_plugin_source):
        (plugin_root / ".codex-plugin").mkdir(parents=True)
        (plugin_root / ".codex-plugin/plugin.json").write_text(
            json.dumps({"name": "personal-agent-memory", "version": "0.1.0"}),
            encoding="utf-8",
        )
    if shared_marketplace:
        (tmp_path / "shared-plugin").write_text("present\n", encoding="utf-8")
    executable(
        fake_bin / "python3",
        f"""#!/bin/sh
if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-uninstaller-during-initialization' ] && \
  [ ! -e "{tmp_path}/uninstaller-initially-replaced" ]; then
  cp "{tmp_path}/replacement-uninstaller-source" "{stable_uninstaller}.replacement"
  chmod 755 "{stable_uninstaller}.replacement"
  mv "{stable_uninstaller}.replacement" "{stable_uninstaller}"
  : >"{tmp_path}/uninstaller-initially-replaced"
fi
if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-unit-during-initialization' ] && \
  [ ! -e "{tmp_path}/unit-initially-replaced" ]; then
  cp "{tmp_path}/replacement-unit-source" "{unit_path}.replacement"
  mv "{unit_path}.replacement" "{unit_path}"
  : >"{tmp_path}/unit-initially-replaced"
fi
if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-state-during-initialization' ] && \
  [ ! -e "{tmp_path}/state-initially-replaced" ]; then
  mv "{state_dir}" "{tmp_path}/original-platform-state"
  mkdir "{state_dir}"
  cp "{tmp_path}/original-platform-state/.personal-agent-memory-owned.json" \
    "{state_dir}/.personal-agent-memory-owned.json"
  printf '%s\n' 'replacement state sentinel' >"{state_dir}/user-sentinel.txt"
  : >"{tmp_path}/state-initially-replaced"
fi
if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-codex-home-during-initialization' ] && \
  [ ! -e "{tmp_path}/codex-home-initially-replaced" ]; then
  mv "{managed_codex_home}" "{tmp_path}/original-codex-home"
  mkdir "{managed_codex_home}"
  printf '%s\n' 'replacement codex sentinel' >"{managed_codex_home}/user-sentinel.txt"
  : >"{tmp_path}/codex-home-initially-replaced"
fi
exec "{sys.executable}" "$@"
""",
    )
    executable(fake_bin / "personal-agent-memory", "#!/bin/sh\nexit 0\n")

    release = tmp_path / "downloaded-release"
    release.mkdir()
    shutil.copy2(ROOT / "uninstall.sh", release / "uninstall.sh")
    source = (release / "uninstall.sh").read_text(encoding="utf-8")
    first_line, remainder = source.split("\n", 1)
    executable(
        stable_uninstaller,
        (
            f"{first_line}\n"
            f"PAM_INSTALL_CONFIG_PATH={config_path!s}\n"
            f"PAM_MANAGED_UNIT_PATH={unit_path!s}\n"
            f"PAM_MANAGED_UNINSTALL_PATH={stable_uninstaller!s}\n"
            f"PAM_MANAGED_STATE_DIR={state_dir!s}\n"
            f"PAM_MANAGED_CODEX_HOME={managed_codex_home!s}\n"
            f"{remainder}"
        ),
    )
    uninstall_sha256 = hashlib.sha256(stable_uninstaller.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "state_dir": str(state_dir),
        "state_dir_ownership": "installer-created-exclusive",
        "state_ownership_token": token,
        "library_root": str(library_root),
        "library_root_ownership": "user-content-never-delete",
        "installed_version": "0.1.0",
        "marketplace_ownership": marketplace_ownership,
        "marketplace_source": str(marketplace_source),
        "marketplace_update_pending": False,
        "uninstall_ownership": "installer-managed",
        "uninstall_path": str(stable_uninstaller),
        "uninstall_sha256": uninstall_sha256,
        "uv_path": str(fake_bin / "uv"),
        "daemon_path": str(fake_bin / "personal-agent-memory"),
        "unit_path": str(unit_path),
        "unit_sha256": hashlib.sha256(unit_path.read_bytes()).hexdigest(),
        "codex_home": str(managed_codex_home),
        "plugin_id": "personal-agent-memory@personal-agent-memory",
        "plugin_version": "0.1.0",
        "plugin_registration_source": str(plugin_registration_source),
        "plugin_install_path": str(managed_plugin_source),
        "plugin_tree_sha256": plugin_tree_sha256(managed_plugin_source),
    }
    config_path.write_text(json.dumps(metadata), encoding="utf-8")
    marker = {
        "schema_version": 1,
        "ownership": "personal-agent-memory-platform-state",
        "state_dir": str(state_dir),
        "install_config_path": str(config_path),
        "token": token if valid_state_marker else "b" * 64,
    }
    (state_dir / ".personal-agent-memory-owned.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )

    executable(
        fake_bin / "systemctl",
        f"""#!/bin/sh
printf 'systemctl %s\\n' "$*" >>"{command_log}"
case "$*" in
  '--user show-environment')
    [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" != 'manager-unavailable' ]
    ;;
  '--user stop personal-agent-memory.service')
    rm -f "{tmp_path}/service-active"
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-unit-during-uninstall' ]; then
      rm -f "{unit_path}"
      printf '%s\n' '[Service]' 'ExecStart=/bin/true' >"{unit_path}"
    elif [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-unit-parent-during-uninstall' ]; then
      mv "{unit_path.parent}" "{tmp_path}/original-unit-parent"
      mkdir "{unit_path.parent}"
      printf '%s\n' '[Service]' 'ExecStart=/bin/true' >"{unit_path}"
    fi
    ;;
  '--user disable personal-agent-memory.service')
    [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" != 'disable' ] || exit 44
    rm -f "{tmp_path}/service-enabled"
    ;;
  '--user daemon-reload') exit 0 ;;
  '--user is-active --quiet personal-agent-memory.service')
    [ -e "{tmp_path}/service-active" ]
    ;;
  '--user is-enabled --quiet personal-agent-memory.service')
    [ -e "{tmp_path}/service-enabled" ]
    ;;
esac
""",
    )
    executable(
        fake_bin / "uv",
        f"""#!/bin/sh
printf 'uv %s\\n' "$*" >>"{command_log}"
if [ "$*" = 'tool uninstall personal-agent-memory' ]; then
  [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" != 'uv-remove' ] || exit 45
  if [ -e "{tool_environment}" ]; then
    rm -f "{tool_environment}" "{fake_bin}/personal-agent-memory"
  else
    printf '%s\n' 'error: tool personal-agent-memory is not installed' >&2
    exit 2
  fi
fi
""",
    )
    executable(
        fake_bin / "codex",
        f"""#!/bin/sh
printf 'codex %s\\n' "$*" >>"{command_log}"
resolved_codex_home=$(readlink -f "${{CODEX_HOME:-}}")
case "$resolved_codex_home" in
  "{managed_codex_home}"|"{tmp_path}/original-codex-home") ;;
  *) exit 91 ;;
esac
case "$*" in
  'plugin list --json')
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-codex-home-during-list' ] && \
      [ ! -e "{tmp_path}/codex-home-replaced" ]; then
      mv "{managed_codex_home}" "{tmp_path}/original-codex-home"
      mkdir "{managed_codex_home}"
      printf '%s\n' 'replacement plugin state' >"{managed_codex_home}/plugin-sentinel"
      printf '%s\n' 'replacement marketplace state' >"{managed_codex_home}/marketplace-sentinel"
      : >"{tmp_path}/codex-home-replaced"
    fi
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-state-during-list' ] && \
      [ ! -e "{tmp_path}/state-dir-replaced" ]; then
      mv "{state_dir}" "{tmp_path}/original-platform-state"
      mkdir "{state_dir}"
      cp "{tmp_path}/original-platform-state/.personal-agent-memory-owned.json" \
        "{state_dir}/.personal-agent-memory-owned.json"
      printf '%s\n' 'replacement user state' >"{state_dir}/replacement-user-file.txt"
      : >"{tmp_path}/state-dir-replaced"
    fi
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-uninstaller-during-uninstall' ] && \
      [ ! -e "{tmp_path}/uninstaller-replaced" ]; then
      cp "{tmp_path}/replacement-uninstaller-source" \
        "{stable_uninstaller}.replacement"
      chmod 755 "{stable_uninstaller}.replacement"
      mv "{stable_uninstaller}.replacement" "{stable_uninstaller}"
      : >"{tmp_path}/uninstaller-replaced"
    fi
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-metadata-during-uninstall' ]; then
      rm -f "{config_path}"
      printf '%s\n' '{{"replacement":"user-owned"}}' >"{config_path}"
    fi
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'malformed-plugin-list' ]; then
      printf '%s\n' '{{not-json'
      exit 0
    fi
    if [ -e "{tmp_path}/plugin-present" ]; then
      if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replaced-plugin' ]; then
        printf '%s\\n' \
          '{{"installed":[{{"pluginId":"personal-agent-memory@personal-agent-memory",'\
'"version":"9.9.9","source":{{"path":"{tmp_path}/user-plugin"}}}}]}}'
      else
        printf '%s\\n' \
          '{{"installed":[{{"pluginId":"personal-agent-memory@personal-agent-memory",'\
'"version":"0.1.0","installedPath":"{managed_plugin_source}",'\
'"source":{{"path":"{plugin_registration_source}"}}}}]}}'
      fi
    elif [ -e "{tmp_path}/shared-plugin" ]; then
      printf '%s\\n' '{{"installed":[{{"pluginId":"shared-tool@personal-agent-memory"}}]}}'
    else
      printf '%s\\n' '{{"installed":[]}}'
    fi
    ;;
  'plugin remove personal-agent-memory@personal-agent-memory --json')
    if find "${{CODEX_HOME}}/plugins/cache" -type d \
      -name '.personal-agent-memory-uninstall-*' -print -quit 2>/dev/null | grep -q .; then
      printf '%s\n' 'quarantine was exposed as a cached plugin version' >&2
      exit 73
    fi
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = \
      'replace-quarantined-plugin-during-remove' ]; then
      quarantine=$(find "$resolved_codex_home" -maxdepth 1 -type d \
        -name '.personal-agent-memory-uninstall-*' -print -quit)
      mv "$quarantine/0.1.0" "$quarantine/0.1.0.installer-owned"
      mkdir "$quarantine/0.1.0"
      printf '%s\n' 'must survive' >"$quarantine/0.1.0/user-owned.txt"
    fi
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-codex-home-during-remove' ] && \
      [ ! -e "{tmp_path}/codex-home-replaced" ]; then
      mv "{managed_codex_home}" "{tmp_path}/original-codex-home"
      mkdir "{managed_codex_home}"
      printf '%s\n' 'replacement plugin state' >"{managed_codex_home}/plugin-sentinel"
      printf '%s\n' 'replacement marketplace state' >"{managed_codex_home}/marketplace-sentinel"
      : >"{tmp_path}/codex-home-replaced"
    fi
    rm -f "${{CODEX_HOME}}/plugin-sentinel"
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-plugin-during-remove' ]; then
      mv "{tmp_path}/racing-plugin-replacement" "{managed_plugin_source}" \
        2>/dev/null || true
    fi
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" != 'ineffective-plugin-remove' ]; then
      rm -f "{tmp_path}/plugin-present"
      rm -rf "{managed_plugin_source}"
    fi
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-uninstaller-parent-during-uninstall' ]; then
      mv "{fake_bin}" "{tmp_path}/original-uninstaller-parent"
      mkdir "{fake_bin}"
      cp "{tmp_path}/replacement-uninstaller-source" "{stable_uninstaller}"
      chmod 755 "{stable_uninstaller}"
    elif [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" = 'replace-metadata-parent-during-uninstall' ]; then
      mv "{config_dir}" "{tmp_path}/original-metadata-parent"
      mkdir "{config_dir}"
      printf '%s\n' '{{"replacement":"user-owned"}}' >"{config_path}"
      printf '%s\n' 'preserve me' >"{config_dir}/user-note.txt"
    fi
    printf '%s\\n' '{{"removed":true}}'
    ;;
  'plugin marketplace list --json')
    if [ -e "{tmp_path}/marketplace-present" ]; then
      printf '{{"marketplaces":[{{"name":"personal-agent-memory","root":"%s"}}]}}\\n' \
        "$(cat "{tmp_path}/marketplace-present")"
    else
      printf '%s\\n' '{{"marketplaces":[]}}'
    fi
    ;;
  'plugin marketplace remove personal-agent-memory --json')
    if [ "${{PAM_UNINSTALL_TEST_FAILURE:-}}" != 'ineffective-marketplace-remove' ]; then
      rm -f "{tmp_path}/marketplace-present"
    fi
    printf '%s\\n' '{{"removed":true}}'
    ;;
esac
""",
    )
    environment = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
    }
    paths = {
        "release": release,
        "state": state_dir,
        "library": library_root,
        "metadata": config_path,
        "unit": unit_path,
        "uninstaller": stable_uninstaller,
        "program": fake_bin / "personal-agent-memory",
        "uv": fake_bin / "uv",
        "marketplace": tmp_path / "marketplace-present",
        "plugin": tmp_path / "plugin-present",
        "plugin_source": managed_plugin_source,
        "plugin_registration_source": plugin_registration_source,
        "tool_environment": tool_environment,
        "codex_home": managed_codex_home,
    }
    return environment, paths, command_log


def test_stable_uninstaller_survives_release_removal_and_preserves_data(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    shutil.rmtree(paths["release"])

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not paths["unit"].exists()
    assert not paths["program"].exists()
    assert not paths["plugin"].exists()
    assert not paths["marketplace"].exists()
    assert not paths["uninstaller"].exists()
    assert paths["state"].is_dir()
    assert (paths["state"] / "api-key").read_text(encoding="utf-8") == "preserved-secret\n"
    assert paths["metadata"].is_file()
    assert (paths["library"] / "memory.md").read_text(encoding="utf-8") == (
        "# Authoritative memory\n"
    )
    assert f"Platform state preserved: {paths['state']}" in result.stdout
    assert f"API key preserved: {paths['state'] / 'api-key'}" in result.stdout
    assert f"Install configuration preserved: {paths['metadata']}" in result.stdout
    assert f"All memory libraries preserved under: {paths['library']}" in result.stdout
    log = command_log.read_text(encoding="utf-8")
    assert "systemctl --user stop personal-agent-memory.service" in log
    assert "codex plugin remove personal-agent-memory@personal-agent-memory --json" in log
    assert "codex plugin marketplace remove personal-agent-memory --json" in log
    assert "uv tool uninstall personal-agent-memory" in log


def test_uninstall_preserves_symlink_replacement_and_its_target(tmp_path: Path) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    replacement_target = tmp_path / "user-owned-uninstaller"
    replacement_target.write_bytes(paths["uninstaller"].read_bytes())
    replacement_target.chmod(0o755)
    paths["uninstaller"].unlink()
    paths["uninstaller"].symlink_to(replacement_target)

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert paths["uninstaller"].is_symlink()
    assert replacement_target.is_file()
    assert "ownership could not be verified" in result.stdout


def test_uninstall_preserves_entry_replaced_during_codex_removal(tmp_path: Path) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    replacement_source = tmp_path / "replacement-uninstaller-source"
    replacement_source.write_bytes(paths["uninstaller"].read_bytes())
    original_inode = paths["uninstaller"].stat().st_ino
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-uninstaller-during-uninstall"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["uninstaller"].is_file()
    assert paths["uninstaller"].stat().st_ino != original_inode
    assert paths["uninstaller"].read_bytes() == replacement_source.read_bytes()
    assert "changed during uninstall" in result.stderr
    assert "uninstallation complete" not in result.stdout


def test_uninstall_preserves_uninstaller_parent_replaced_during_codex_removal(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    replacement_source = tmp_path / "replacement-uninstaller-source"
    replacement_source.write_text("#!/bin/sh\nprintf 'user owned\\n'\n", encoding="utf-8")
    original_parent_inode = paths["uninstaller"].parent.stat().st_ino
    environment["PAM_UNINSTALL_TEST_FAILURE"] = (
        "replace-uninstaller-parent-during-uninstall"
    )

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["uninstaller"].parent.stat().st_ino != original_parent_inode
    assert paths["uninstaller"].read_bytes() == replacement_source.read_bytes()
    assert "changed during uninstall" in result.stderr
    assert "uninstallation complete" not in result.stdout


def test_uninstall_preserves_entry_replaced_during_python_probe(tmp_path: Path) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    replacement_source = tmp_path / "replacement-uninstaller-source"
    replacement_source.write_bytes(paths["uninstaller"].read_bytes())
    original_inode = paths["uninstaller"].stat().st_ino
    environment["PAM_UNINSTALL_TEST_FAILURE"] = (
        "replace-uninstaller-during-initialization"
    )

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["uninstaller"].is_file()
    assert paths["uninstaller"].stat().st_ino != original_inode
    assert paths["uninstaller"].read_bytes() == replacement_source.read_bytes()
    assert "initialization" in result.stderr
    assert "uninstallation complete" not in result.stdout
    assert not command_log.exists()


def test_purge_removes_only_verified_platform_state(tmp_path: Path) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not paths["state"].exists()
    assert not paths["metadata"].exists()
    assert not paths["uninstaller"].exists()
    assert (paths["library"] / "memory.md").read_text(encoding="utf-8") == (
        "# Authoritative memory\n"
    )
    assert f"Platform state removed: {paths['state']}" in result.stdout


def test_purge_preserves_state_when_ownership_marker_does_not_match(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path, valid_state_marker=False)

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["state"].is_dir()
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()
    assert (paths["library"] / "memory.md").is_file()
    assert "ownership marker does not match install metadata" in result.stderr


@pytest.mark.parametrize("marketplace_kind", ["preexisting", "shared"])
def test_uninstall_preserves_unowned_or_shared_marketplace(
    tmp_path: Path, marketplace_kind: str
) -> None:
    environment, paths, command_log = uninstall_test_environment(
        tmp_path,
        marketplace_ownership=(
            "preexisting" if marketplace_kind == "preexisting" else "installer-managed"
        ),
        shared_marketplace=marketplace_kind == "shared",
    )

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert paths["marketplace"].is_file()
    assert "marketplace remove personal-agent-memory" not in command_log.read_text(encoding="utf-8")


def test_uninstall_preserves_replaced_plugin_and_continues_cleanup(tmp_path: Path) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replaced-plugin"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert paths["plugin"].is_file()
    assert paths["plugin_source"].is_dir()
    assert paths["marketplace"].is_file()
    assert not paths["program"].exists()
    assert not paths["unit"].exists()
    log = command_log.read_text(encoding="utf-8")
    assert "codex plugin remove personal-agent-memory@personal-agent-memory --json" not in log
    assert "codex plugin marketplace remove personal-agent-memory --json" not in log
    assert "Preserved Codex plugin because installer ownership could not be verified." in (
        result.stdout
    )


def test_uninstall_preserves_plugin_when_legacy_metadata_has_no_identity(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    for field in (
        "plugin_id",
        "plugin_version",
        "plugin_registration_source",
        "plugin_install_path",
        "plugin_tree_sha256",
    ):
        metadata.pop(field)
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert paths["plugin"].is_file()
    assert paths["plugin_source"].is_dir()
    assert paths["marketplace"].is_file()
    assert not paths["program"].exists()
    assert not paths["unit"].exists()
    log = command_log.read_text(encoding="utf-8")
    assert "codex plugin remove personal-agent-memory@personal-agent-memory --json" not in log
    assert "codex plugin marketplace remove personal-agent-memory --json" not in log
    assert "Preserved Codex plugin because installer ownership could not be verified." in (
        result.stdout
    )


def test_uninstall_preserves_plugin_with_mutated_managed_contents(tmp_path: Path) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    mutated = paths["plugin_source"] / "scripts/replaced.mjs"
    mutated.parent.mkdir()
    mutated.write_text("export const userReplacement = true;\n", encoding="utf-8")

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert mutated.read_text(encoding="utf-8") == "export const userReplacement = true;\n"
    assert paths["plugin"].is_file()
    assert paths["marketplace"].is_file()
    assert not paths["program"].exists()
    assert not paths["unit"].exists()
    log = command_log.read_text(encoding="utf-8")
    assert "codex plugin remove personal-agent-memory@personal-agent-memory --json" not in log
    assert "codex plugin marketplace remove personal-agent-memory --json" not in log
    assert "installed contents no longer match the installer record" in result.stdout


def test_uninstall_plugin_remove_cannot_delete_a_racing_replacement(tmp_path: Path) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    cache_namespace = paths["plugin_source"].parent.parent
    plugin_parent_mode = cache_namespace.stat().st_mode & 0o777
    replacement = tmp_path / "racing-plugin-replacement"
    replacement.mkdir()
    replacement_sentinel = replacement / "user-owned.txt"
    replacement_sentinel.write_text("must survive\n", encoding="utf-8")
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-plugin-during-remove"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not paths["plugin"].exists()
    assert not paths["plugin_source"].exists()
    assert replacement_sentinel.read_text(encoding="utf-8") == "must survive\n"
    assert cache_namespace.stat().st_mode & 0o777 == plugin_parent_mode
    assert not list(paths["codex_home"].glob(".personal-agent-memory-uninstall-*"))
    assert "codex plugin remove personal-agent-memory@personal-agent-memory --json" in (
        command_log.read_text(encoding="utf-8")
    )


def test_uninstall_preserves_plugin_replaced_after_digest_verification(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = (
        "replace-quarantined-plugin-during-remove"
    )

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    replacement = paths["plugin_source"] / "user-owned.txt"
    assert result.returncode != 0
    assert replacement.read_text(encoding="utf-8") == "must survive\n"
    assert (paths["plugin_source"].parent / "0.1.0.installer-owned").is_dir()
    assert not list(paths["codex_home"].glob(".personal-agent-memory-uninstall-*"))
    assert "cleanup did not commit safely" in result.stderr
    assert "codex plugin remove personal-agent-memory@personal-agent-memory --json" in (
        command_log.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("script_name", ["install.sh", "uninstall.sh"])
def test_plugin_tree_digest_stops_streaming_at_the_entry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script_name: str
) -> None:
    digest, error_type = embedded_plugin_tree_digest(script_name)
    shared_file = tmp_path / "entry"
    shared_file.write_bytes(b"")

    class Entry:
        path = str(shared_file)

        @staticmethod
        def stat(*, follow_symlinks: bool) -> os.stat_result:
            assert follow_symlinks is False
            return shared_file.stat()

    class StreamingDirectory:
        consumed = 0

        def __enter__(self) -> StreamingDirectory:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self) -> StreamingDirectory:
            return self

        def __next__(self) -> Entry:
            self.consumed += 1
            if self.consumed > 4097:
                raise AssertionError("plugin digest read past its declared entry bound")
            return Entry()

    stream = StreamingDirectory()
    monkeypatch.setattr(os, "scandir", lambda _directory: stream)

    with pytest.raises(error_type, match="entry limit"):
        digest(tmp_path)
    assert stream.consumed == 4097


def test_real_codex_cli_distinguishes_registration_source_from_install_cache(
    tmp_path: Path,
) -> None:
    codex = shutil.which("codex")
    if codex is None:
        pytest.skip("Codex CLI is required for the real plugin JSON contract")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    environment = {**os.environ, "CODEX_HOME": str(codex_home)}

    subprocess.run(
        [codex, "plugin", "marketplace", "add", str(ROOT), "--json"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    added = subprocess.run(
        [codex, "plugin", "add", "personal-agent-memory@personal-agent-memory", "--json"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    listed = subprocess.run(
        [codex, "plugin", "list", "--json"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    add_payload = json.loads(added.stdout)
    installed_path = Path(add_payload["installedPath"]).resolve()
    records = json.loads(listed.stdout)["installed"]
    record = next(
        item
        for item in records
        if item.get("pluginId") == "personal-agent-memory@personal-agent-memory"
    )
    registration_source = Path(record["source"]["path"]).resolve()

    assert installed_path.is_dir()
    assert installed_path.is_relative_to((codex_home / "plugins/cache").resolve())
    assert registration_source == (ROOT / "plugins/personal-agent-memory").resolve()
    assert registration_source != installed_path

    removed = subprocess.run(
        [codex, "plugin", "remove", "personal-agent-memory@personal-agent-memory", "--json"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    listed_after_remove = subprocess.run(
        [codex, "plugin", "list", "--json"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(removed.stdout)["pluginId"] == (
        "personal-agent-memory@personal-agent-memory"
    )
    assert not installed_path.exists()
    assert json.loads(listed_after_remove.stdout)["installed"] == []


def test_real_codex_cli_add_to_managed_uninstall_contract(tmp_path: Path) -> None:
    codex = shutil.which("codex")
    if codex is None:
        pytest.skip("Codex CLI is required for the real managed uninstall contract")
    environment, paths, _ = uninstall_test_environment(tmp_path)
    shutil.rmtree(paths["codex_home"])
    paths["codex_home"].mkdir()
    setup_environment = {**environment, "CODEX_HOME": str(paths["codex_home"])}
    subprocess.run(
        [codex, "plugin", "marketplace", "add", str(ROOT), "--json"],
        env=setup_environment,
        text=True,
        capture_output=True,
        check=True,
    )
    added = subprocess.run(
        [codex, "plugin", "add", "personal-agent-memory@personal-agent-memory", "--json"],
        env=setup_environment,
        text=True,
        capture_output=True,
        check=True,
    )
    installed_path = Path(json.loads(added.stdout)["installedPath"]).resolve()
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata.update(
        {
            "marketplace_source": str(ROOT),
            "plugin_registration_source": str(
                ROOT / "plugins/personal-agent-memory"
            ),
            "plugin_install_path": str(installed_path),
            "plugin_tree_sha256": plugin_tree_sha256(installed_path),
        }
    )
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    executable(
        Path(environment["PATH"].split(":", 1)[0]) / "codex",
        f'#!/bin/sh\nexec "{codex}" "$@"\n',
    )

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not installed_path.exists()
    assert not any(
        path.name.startswith(".personal-agent-memory-uninstall-")
        for path in paths["codex_home"].iterdir()
    )
    listed = subprocess.run(
        [codex, "plugin", "list", "--json"],
        env=setup_environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(listed.stdout)["installed"] == []


def test_repeated_uninstall_with_missing_components_is_bounded(tmp_path: Path) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    paths["unit"].unlink()
    paths["program"].unlink()
    paths["plugin"].unlink()
    paths["marketplace"].unlink()
    (tmp_path / "service-active").unlink()

    first = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    second = subprocess.run(
        [str(ROOT / "uninstall.sh")],
        cwd=tmp_path,
        env={**environment, "PAM_INSTALL_CONFIG_PATH": str(paths["metadata"])},
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "predates trusted state and Codex path hints" in second.stderr
    assert paths["state"].is_dir()
    assert paths["metadata"].is_file()
    assert (paths["library"] / "memory.md").is_file()
    assert not any(
        line.startswith(("sudo ", "apt ", "apt-get ", "dnf ", "yum ", "pacman "))
        for line in command_log.read_text(encoding="utf-8").splitlines()
    )


def test_legacy_uninstaller_without_trusted_directory_hints_fails_safely(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    source = paths["uninstaller"].read_text(encoding="utf-8")
    source = "\n".join(
        line
        for line in source.splitlines()
        if not line.startswith(("PAM_MANAGED_STATE_DIR=", "PAM_MANAGED_CODEX_HOME="))
    )
    paths["uninstaller"].write_text(source + "\n", encoding="utf-8")

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "predates trusted state and Codex path hints" in result.stderr
    assert paths["state"].is_dir()
    assert paths["unit"].is_file()
    assert paths["plugin"].is_file()
    assert paths["marketplace"].is_file()
    assert paths["tool_environment"].is_file()
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()
    assert not command_log.exists()
    assert "uninstallation complete" not in result.stdout


@pytest.mark.parametrize(
    ("failure", "target", "sentinel", "expected_text"),
    [
        (
            "replace-state-during-initialization",
            "state",
            "user-sentinel.txt",
            "replacement state sentinel\n",
        ),
        (
            "replace-codex-home-during-initialization",
            "codex_home",
            "user-sentinel.txt",
            "replacement codex sentinel\n",
        ),
    ],
)
def test_uninstall_rejects_directory_replacement_during_first_python_probe(
    tmp_path: Path,
    failure: str,
    target: str,
    sentinel: str,
    expected_text: str,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = failure

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (paths[target] / sentinel).read_text(encoding="utf-8") == expected_text
    if target == "state":
        marker = json.loads(
            (paths["state"] / ".personal-agent-memory-owned.json").read_text(
                encoding="utf-8"
            )
        )
        assert marker["token"] == "a" * 64
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()
    assert paths["unit"].is_file()
    assert paths["plugin"].is_file()
    assert paths["marketplace"].is_file()
    assert paths["tool_environment"].is_file()
    log = command_log.read_text(encoding="utf-8") if command_log.exists() else ""
    assert "systemctl" not in log
    assert "codex" not in log
    assert "uv tool uninstall" not in log
    assert "uninstallation complete" not in result.stdout


def test_uninstall_removes_uv_tool_environment_when_command_entry_is_missing(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    paths["program"].unlink()

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not paths["tool_environment"].exists()
    assert "uv tool uninstall personal-agent-memory" in command_log.read_text(
        encoding="utf-8"
    )


def test_uninstall_fails_before_side_effects_when_uv_and_command_are_missing(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    paths["uv"].unlink()
    paths["program"].unlink()

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "recorded uv command is required" in result.stderr
    assert paths["tool_environment"].is_file()
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()
    assert paths["unit"].is_file()
    assert paths["plugin"].is_file()
    assert paths["marketplace"].is_file()
    log = command_log.read_text(encoding="utf-8") if command_log.exists() else ""
    assert "systemctl --user stop" not in log
    assert "codex plugin remove" not in log
    assert "codex plugin marketplace remove" not in log
    assert "uninstallation complete" not in result.stdout


def test_uninstall_uses_recorded_codex_home_when_callers_home_changes(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    wrong_codex_home = tmp_path / "wrong-codex-home"
    wrong_codex_home.mkdir()
    wrong_sentinel = wrong_codex_home / "must-remain"
    wrong_sentinel.write_text("untouched\n", encoding="utf-8")
    environment["CODEX_HOME"] = str(wrong_codex_home)

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not paths["plugin"].exists()
    assert not paths["marketplace"].exists()
    assert wrong_sentinel.read_text(encoding="utf-8") == "untouched\n"
    codex_commands = [
        line for line in command_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("codex ")
    ]
    assert codex_commands


def test_uninstall_rejects_codex_home_replaced_during_first_plugin_list(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-codex-home-during-list"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "codex-home" in result.stderr
    assert paths["codex_home"].is_dir()
    assert (paths["codex_home"] / "plugin-sentinel").read_text(encoding="utf-8") == (
        "replacement plugin state\n"
    )
    assert (paths["codex_home"] / "marketplace-sentinel").read_text(
        encoding="utf-8"
    ) == "replacement marketplace state\n"
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()
    log = command_log.read_text(encoding="utf-8")
    assert log.count("codex plugin list --json") == 1
    assert "codex plugin remove" not in log
    assert "codex plugin marketplace remove" not in log
    assert "uninstallation complete" not in result.stdout


def test_uninstall_codex_remove_uses_bound_home_when_lexical_home_is_replaced(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    original_plugin_state = paths["codex_home"] / "plugin-sentinel"
    original_plugin_state.write_text("original plugin state\n", encoding="utf-8")
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-codex-home-during-remove"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "codex-home" in result.stderr
    assert not (tmp_path / "original-codex-home/plugin-sentinel").exists()
    assert (paths["codex_home"] / "plugin-sentinel").read_text(
        encoding="utf-8"
    ) == "replacement plugin state\n"
    assert (paths["codex_home"] / "marketplace-sentinel").read_text(
        encoding="utf-8"
    ) == "replacement marketplace state\n"
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()
    log = command_log.read_text(encoding="utf-8")
    assert log.count("codex plugin remove") == 1
    assert "codex plugin marketplace remove" not in log
    assert "uninstallation complete" not in result.stdout


def test_purge_rejects_state_dir_replaced_during_codex_command(tmp_path: Path) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-state-during-list"

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "state-dir" in result.stderr
    assert (paths["state"] / "replacement-user-file.txt").read_text(
        encoding="utf-8"
    ) == "replacement user state\n"
    replacement_marker = json.loads(
        (paths["state"] / ".personal-agent-memory-owned.json").read_text(
            encoding="utf-8"
        )
    )
    assert replacement_marker["token"] == "a" * 64
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()
    assert "uninstallation complete" not in result.stdout


def test_uninstall_ignores_unrelated_command_with_same_name_on_path(tmp_path: Path) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    unrelated_bin = tmp_path / "unrelated-bin"
    unrelated_bin.mkdir()
    unrelated_command = unrelated_bin / "personal-agent-memory"
    executable(unrelated_command, "#!/bin/sh\nexit 0\n")
    environment["PATH"] = f"{unrelated_bin}:{environment['PATH']}"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert unrelated_command.is_file()
    assert not paths["program"].exists()


def test_uninstall_preserves_replaced_service_unit(tmp_path: Path) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    paths["unit"].write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")
    (tmp_path / "service-active").unlink()
    (tmp_path / "service-enabled").unlink()

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert paths["unit"].read_text(encoding="utf-8") == "[Service]\nExecStart=/bin/true\n"
    assert "ownership could not be verified" in result.stdout
    assert "systemctl --user disable personal-agent-memory.service" not in (
        command_log.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("failure", "expected_error", "remaining_path"),
    [
        ("malformed-plugin-list", "Expecting property name", "plugin"),
        ("ineffective-plugin-remove", "still installed after removal", "plugin"),
        (
            "ineffective-marketplace-remove",
            "still registered after removal",
            "marketplace",
        ),
    ],
)
def test_uninstall_rejects_unverified_codex_removal(
    tmp_path: Path,
    failure: str,
    expected_error: str,
    remaining_path: str,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = failure

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert paths[remaining_path].exists()
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()


def test_uninstall_propagates_service_disable_failure(tmp_path: Path) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "disable"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Uninstallation failed during user service removal" in result.stderr
    assert paths["unit"].is_file()
    assert paths["metadata"].is_file()


def test_uninstall_preserves_service_unit_replaced_during_systemctl(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-unit-during-uninstall"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["unit"].read_text(encoding="utf-8") == (
        "[Service]\nExecStart=/bin/true\n"
    )
    assert paths["metadata"].is_file()
    assert "changed during uninstall" in result.stderr
    assert "uninstallation complete" not in result.stdout


def test_uninstall_preserves_unit_parent_replaced_during_systemctl(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    original_parent_inode = paths["unit"].parent.stat().st_ino
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-unit-parent-during-uninstall"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["unit"].parent.stat().st_ino != original_parent_inode
    assert paths["unit"].read_text(encoding="utf-8") == (
        "[Service]\nExecStart=/bin/true\n"
    )
    assert paths["metadata"].is_file()
    assert "changed during uninstall" in result.stderr
    assert "uninstallation complete" not in result.stdout


def test_uninstall_preserves_unit_replaced_during_python_probe(tmp_path: Path) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    replacement_source = tmp_path / "replacement-unit-source"
    replacement_source.write_bytes(paths["unit"].read_bytes())
    original_inode = paths["unit"].stat().st_ino
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-unit-during-initialization"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["unit"].is_file()
    assert paths["unit"].stat().st_ino != original_inode
    assert paths["unit"].read_bytes() == replacement_source.read_bytes()
    assert "initialization" in result.stderr
    assert "uninstallation complete" not in result.stdout
    assert not command_log.exists()


def test_uninstall_fails_when_unowned_unit_service_remains_enabled(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    paths["unit"].write_text(
        "[Service]\nExecStart=/bin/true\n",
        encoding="utf-8",
    )
    (tmp_path / "service-active").unlink()

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["unit"].read_text(encoding="utf-8") == (
        "[Service]\nExecStart=/bin/true\n"
    )
    assert (tmp_path / "service-enabled").is_file()
    assert "service is still enabled" in result.stderr
    assert "uninstallation complete" not in result.stdout


def test_purge_preserves_metadata_replaced_during_codex_removal(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "replace-metadata-during-uninstall"

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert json.loads(paths["metadata"].read_text(encoding="utf-8")) == {
        "replacement": "user-owned"
    }
    assert paths["state"].is_dir()
    assert "changed during uninstall" in result.stderr
    assert "uninstallation complete" not in result.stdout


def test_purge_preserves_metadata_parent_replaced_during_codex_removal(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    original_parent_inode = paths["metadata"].parent.stat().st_ino
    environment["PAM_UNINSTALL_TEST_FAILURE"] = (
        "replace-metadata-parent-during-uninstall"
    )

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert paths["metadata"].parent.stat().st_ino != original_parent_inode
    assert json.loads(paths["metadata"].read_text(encoding="utf-8")) == {
        "replacement": "user-owned"
    }
    assert (paths["metadata"].parent / "user-note.txt").read_text(
        encoding="utf-8"
    ) == "preserve me\n"
    assert paths["state"].is_dir()
    assert "changed during uninstall" in result.stderr
    assert "uninstallation complete" not in result.stdout


def test_uninstall_rejects_uv_failure_with_dangling_managed_command(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    paths["program"].unlink()
    paths["program"].symlink_to(tmp_path / "missing-command-target")
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "uv-remove"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "uv failed to remove" in result.stderr
    assert paths["program"].is_symlink()
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()
    assert "uninstallation complete" not in result.stdout


def test_uninstall_requires_user_manager_when_managed_unit_is_missing(
    tmp_path: Path,
) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    paths["unit"].unlink()
    environment["PAM_UNINSTALL_TEST_FAILURE"] = "manager-unavailable"

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "working systemd user manager is required" in result.stderr
    assert paths["plugin"].is_file()
    assert paths["program"].is_file()
    assert paths["metadata"].is_file()
    assert "codex plugin list --json" not in command_log.read_text(encoding="utf-8")


def test_legacy_metadata_purges_only_known_platform_state(tmp_path: Path) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata.pop("state_dir_ownership")
    metadata.pop("state_ownership_token")
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    (paths["state"] / ".personal-agent-memory-owned.json").unlink()
    (paths["state"] / "platform.sqlite3").write_bytes(b"database")
    (paths["state"] / "platform.sqlite3-journal").write_bytes(b"journal")
    (paths["state"] / "platform.sqlite3-wal").write_bytes(b"wal")
    (paths["state"] / "platform.sqlite3-shm").write_bytes(b"shm")
    (paths["state"] / "sensitive-dedupe-key").write_bytes(b"dedupe-key")
    (paths["state"] / "tombstone-keys.json").write_text(
        '{"format":1,"active":"key","keys":{"key":"dGVzdA=="}}\n',
        encoding="utf-8",
    )
    (paths["state"] / "graphs").mkdir()
    (paths["state"] / "graphs/managed.json").write_text("{}\n", encoding="utf-8")
    (paths["state"] / "git/library.git").mkdir(parents=True)
    (paths["state"] / "git/library.git/HEAD").write_text("ref: refs/heads/main\n")
    (paths["state"] / "locks").mkdir()
    (paths["state"] / "locks/managed.lock").write_text("locked\n", encoding="utf-8")
    unknown = paths["state"] / "shared-user-file.txt"
    unknown.write_text("preserve me\n", encoding="utf-8")

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert unknown.read_text(encoding="utf-8") == "preserve me\n"
    assert sorted(path.name for path in paths["state"].iterdir()) == [unknown.name]
    assert not paths["metadata"].exists()
    assert f"unknown contents preserved under: {paths['state']}" in result.stdout
    assert f"Preserved unknown state entries: {unknown.name}" in result.stdout


def test_legacy_purge_preserves_file_only_entry_that_became_directory(
    tmp_path: Path,
) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata.pop("state_dir_ownership")
    metadata.pop("state_ownership_token")
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    (paths["state"] / ".personal-agent-memory-owned.json").unlink()
    (paths["state"] / "api-key").unlink()
    user_document = paths["state"] / "api-key/user-document.txt"
    user_document.parent.mkdir()
    user_document.write_text("authoritative user content\n", encoding="utf-8")

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert user_document.read_text(encoding="utf-8") == "authoritative user content\n"
    assert paths["state"].is_dir()
    assert "Preserved unknown state entries: api-key" in result.stdout
    assert not paths["metadata"].exists()


def test_legacy_purge_rejects_state_path_replaced_by_symlink(tmp_path: Path) -> None:
    environment, paths, _ = uninstall_test_environment(tmp_path)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata.pop("state_dir_ownership")
    metadata.pop("state_ownership_token")
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    shutil.rmtree(paths["state"])
    user_directory = tmp_path / "unrelated-user-directory"
    (user_directory / "graphs").mkdir(parents=True)
    (user_directory / "api-key").write_text("user secret\n", encoding="utf-8")
    user_document = user_directory / "graphs/user.md"
    user_document.write_text("authoritative user content\n", encoding="utf-8")
    paths["state"].symlink_to(user_directory, target_is_directory=True)

    result = subprocess.run(
        [str(paths["uninstaller"]), "--purge"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "recorded path is not a directory" in result.stderr
    assert paths["state"].is_symlink()
    assert (user_directory / "api-key").read_text(encoding="utf-8") == "user secret\n"
    assert user_document.read_text(encoding="utf-8") == "authoritative user content\n"
    assert paths["metadata"].is_file()
    assert paths["uninstaller"].is_file()


def test_uninstaller_selects_supported_versioned_python(tmp_path: Path) -> None:
    environment, paths, command_log = uninstall_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    executable(
        fake_bin / "python3",
        f"""#!/bin/sh
printf 'python3 %s\n' "$*" >>"{command_log}"
if [ "${{1:-}}" = '-c' ]; then
  exit 1
fi
exit 99
""",
    )
    executable(
        fake_bin / "python3.11",
        f"""#!/bin/sh
printf 'python3.11 %s\n' "$*" >>"{command_log}"
if [ "${{1:-}}" = '-c' ]; then
  exit 0
fi
exec "{sys.executable}" "$@"
""",
    )

    result = subprocess.run(
        [str(paths["uninstaller"])],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    version_check = (
        "-c import sys; raise SystemExit(0 if (3, 11) <= "
        "sys.version_info[:2] < (3, 14) else 1)"
    )
    log = command_log.read_text(encoding="utf-8").splitlines()
    assert f"python3 {version_check}" in log
    assert f"python3.11 {version_check}" in log
    assert any(line.startswith("python3.11 - ") for line in log)


@pytest.mark.parametrize(
    ("uv_mode", "python_fallback", "directory_mode"),
    [
        ("existing", False, "default"),
        ("missing", False, "default"),
        ("trimmed", False, "default"),
        ("existing", True, "default"),
        ("existing", False, "state-only"),
        ("existing", False, "library-only"),
        ("existing", False, "both"),
        ("existing", False, "literal-dollar"),
        ("existing", False, "empty-xdg-custom-state"),
        ("existing", False, "preexisting-state"),
    ],
    ids=[
        "existing-uv",
        "bootstrapped-uv",
        "trimmed-uv",
        "versioned-python",
        "custom-state",
        "custom-library",
        "custom-both",
        "literal-dollar",
        "empty-xdg-custom-state",
        "preexisting-state",
    ],
)
def test_supported_environment_installs_service_plugin_and_healthy_hook(
    tmp_path: Path, uv_mode: str, python_fallback: bool, directory_mode: str
) -> None:
    repo_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    repo_remotes = subprocess.run(
        ["git", "remote", "-v"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    plugin_install = (
        home
        / ".codex/plugins/cache/personal-agent-memory/personal-agent-memory/0.1.0"
    )
    command_log = tmp_path / "commands.log"
    request_log = tmp_path / "requests.jsonl"
    daemon_script = tmp_path / "daemon.py"
    home.mkdir()
    fake_bin.mkdir()
    for name in ("sudo", "apt", "apt-get", "dnf", "yum", "pacman", "zypper"):
        executable(
            fake_bin / name,
            f'#!/bin/sh\nprintf \'{name} %s\\n\' "$*" >>"{command_log}"\nexit 99\n',
        )
    if python_fallback:
        executable(fake_bin / "python3", "#!/bin/sh\nexit 1\n")
        executable(fake_bin / "python3.13", "#!/bin/sh\nexit 1\n")
        executable(fake_bin / "python3.12", "#!/bin/sh\nexit 1\n")
        executable(
            fake_bin / "python3.11",
            f"""#!/bin/sh
printf 'python3.11 %s\n' "$*" >>"{command_log}"
exec "{sys.executable}" "$@"
""",
        )

    daemon_script.write_text(
        """
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

state = Path(sys.argv[sys.argv.index('--state-dir') + 1])
state.mkdir(parents=True, exist_ok=True)
key_path = state / 'api-key'
if not key_path.exists():
    key_path.write_text('installer-test-secret\\n', encoding='utf-8')
    key_path.chmod(0o600)

class Handler(BaseHTTPRequestHandler):
    def reply(self, body):
        if self.headers.get('Authorization') != 'Bearer installer-test-secret':
            self.send_response(401)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({'status': 'ok'})

    def do_POST(self):
        self.rfile.read(int(self.headers.get('content-length', '0')))
        with open(r'{request_log}', 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({
                'path': self.path,
                'authorization': self.headers.get('Authorization'),
            }) + '\\n')
        self.reply({
            'schema_version': 'memory-context-package/v1',
            'status': 'unbound',
            'results': [],
        })

    def log_message(self, format, *args):
        pass

ThreadingHTTPServer(('127.0.0.1', 7331), Handler).serve_forever()
""".lstrip().replace("{request_log}", str(request_log)),
        encoding="utf-8",
    )
    uv_script = f"""#!/bin/sh
set -eu
printf 'uv %s\\n' "$*" >>"{command_log}"
if [ "$*" = '--version' ]; then
  printf 'uv 0.8.0\\n'
  exit 0
fi
if [ "${{1:-}} ${{2:-}} ${{3:-}} ${{5:-}} ${{6:-}} ${{8:-}} ${{9:-}}" = \
  'tool install --python --force --from personal-agent-memory --help' ]; then
  exit 0
fi
if [ "$*" = 'tool dir --bin' ]; then
  printf '%s\\n' "{fake_bin}"
  exit 0
fi
if [ "$*" = 'tool uninstall personal-agent-memory' ]; then
  rm -f "{fake_bin}/personal-agent-memory"
  exit 0
fi
if [ "${{PAM_INSTALL_TEST_FAILURE:-}}" = 'program-update' ]; then
  exit 41
fi
version=$(awk -F'"' '/^version = / {{print $2}}' "$7/pyproject.toml")
printf '%s\\n' "$version" >"{tmp_path}/tool-version-record"
cat >"{fake_bin}/personal-agent-memory" <<'EOF'
#!/bin/sh
if [ "${{1:-}}" = '--version' ]; then
  cat "{tmp_path}/tool-version-record"
  exit 0
fi
exec "{sys.executable}" "{daemon_script}" "$@"
EOF
chmod +x "{fake_bin}/personal-agent-memory"
"""
    if uv_mode == "existing":
        executable(fake_bin / "uv", uv_script)
    else:
        if uv_mode == "missing":
            executable(fake_bin / "uv", "#!/bin/sh\nexit 1\n")
        else:
            executable(
                fake_bin / "uv",
                f"""#!/bin/sh
printf 'trimmed-uv %s\n' "$*" >>"{command_log}"
case "$*" in
  '--version'|'tool install --help') exit 0 ;;
esac
if [ "${{1:-}} ${{2:-}} ${{3:-}} ${{5:-}} ${{7:-}}" = \
  'tool install --python --from --help' ]; then
  exit 0
fi
exit 1
""",
            )
        executable(
            fake_bin / "curl",
            f"""#!/bin/sh
set -eu
printf 'curl %s\\n' "$*" >>"{command_log}"
output=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = '-o' ]; then
    output=$2
    break
  fi
  shift
done
cat >"$output" <<'INSTALLER_EOF'
#!/bin/sh
set -eu
cat >"${{UV_INSTALL_DIR}}/uv" <<'UVEOF'
{uv_script}
UVEOF
chmod +x "${{UV_INSTALL_DIR}}/uv"
INSTALLER_EOF
""",
        )
    executable(
        fake_bin / "codex",
        f"""#!/bin/sh
set -eu
printf 'codex %s\\n' "$*" >>"{command_log}"
printf 'codex-home %s\\n' "$(readlink -f "${{CODEX_HOME:-}}")" >>"{command_log}"
if [ "$*" = '--version' ]; then
  printf 'codex-cli 1.0.0\\n'
  exit 0
fi
if [ "$*" = 'plugin add --help' ] || \
  [ "$*" = 'plugin marketplace add --help' ] || \
  [ "$*" = 'plugin marketplace list --help' ] || \
  [ "$*" = 'plugin marketplace remove --help' ] || \
  [ "$*" = 'plugin list --help' ]; then
  printf 'Usage: codex %s [--json]\\n' "$*"
  exit 0
fi
if [ "$*" = 'plugin marketplace list --json' ]; then
  if [ -s "{tmp_path}/marketplace-records" ]; then
    source=$(cat "{tmp_path}/marketplace-records")
    printf '{{"marketplaces":[{{"name":"personal-agent-memory","root":"%s"}}]}}\\n' \
      "$source"
  else
    printf '%s\\n' '{{"marketplaces":[]}}'
  fi
  exit 0
fi
if [ "$*" = 'plugin marketplace remove personal-agent-memory --json' ]; then
  [ "${{PAM_INSTALL_TEST_FAILURE:-}}" != 'marketplace-remove' ] || exit 44
  rm -f "{tmp_path}/marketplace-records"
  printf '%s\\n' '{{"removed":true}}'
  exit 0
fi
if [ "$1 $2 $3" = 'plugin marketplace add' ]; then
  mkdir -p "{plugin_install.parent}"
  printf '%s\\n' "$4" >"{tmp_path}/marketplace-records"
  printf '{{"marketplaceName":"personal-agent-memory","installedRoot":"%s"}}\\n' "$4"
  exit 0
fi
if [ "$1 $2" = 'plugin add' ]; then
  [ "${{PAM_INSTALL_TEST_FAILURE:-}}" != 'plugin-update' ] || exit 42
  source=$(cat "{tmp_path}/marketplace-records")
  version=$(awk -F'"' '/"version":/ {{print $4}}' \
    "$source/plugins/personal-agent-memory/.codex-plugin/plugin.json")
  rm -rf "{plugin_install}"
  cp -R "$source/plugins/personal-agent-memory" "{plugin_install}"
  printf '%s\\n' 'personal-agent-memory@personal-agent-memory' >"{tmp_path}/plugin-records"
  printf '%s\\n' "$version" >"{tmp_path}/plugin-version-record"
  printf '{{"pluginId":"personal-agent-memory@personal-agent-memory",'\
'"version":"%s","installedPath":"{plugin_install}"}}\\n' "$version"
  exit 0
fi
if [ "$*" = 'plugin remove personal-agent-memory@personal-agent-memory --json' ]; then
  rm -f "{tmp_path}/plugin-records" "{tmp_path}/plugin-version-record"
  rm -rf "{plugin_install}"
  printf '%s\\n' '{{"removed":true}}'
  exit 0
fi
if [ "$*" = 'plugin list --json' ]; then
  if [ ! -s "{tmp_path}/plugin-records" ]; then
    printf '%s\\n' '{{"installed":[]}}'
    exit 0
  fi
  version=$(cat "{tmp_path}/plugin-version-record")
  source=$(cat "{tmp_path}/marketplace-records")
  printf '{{"installed":[{{"pluginId":"personal-agent-memory@personal-agent-memory",'\
'"version":"%s","enabled":true,"source":{{"source":"local",'\
'"path":"%s/plugins/personal-agent-memory"}}}}]}}\\n' "$version" "$source"
fi
""",
    )
    executable(
        fake_bin / "systemctl",
        f"""#!/bin/sh
set -eu
printf 'systemctl %s\\n' "$*" >>"{command_log}"
case "$*" in
  '--user is-active --quiet personal-agent-memory.service')
    [ -s "{tmp_path}/daemon.pid" ] && kill -0 "$(cat "{tmp_path}/daemon.pid")" 2>/dev/null
    ;;
  '--user stop personal-agent-memory.service')
    if [ -s "{tmp_path}/daemon.pid" ]; then
      kill "$(cat "{tmp_path}/daemon.pid")" 2>/dev/null || true
      wait "$(cat "{tmp_path}/daemon.pid")" 2>/dev/null || true
      rm -f "{tmp_path}/daemon.pid"
    fi
    ;;
  '--user enable --now personal-agent-memory.service')
    [ "${{PAM_INSTALL_TEST_FAILURE:-}}" != 'service-startup' ] || exit 43
    config_home="${{XDG_CONFIG_HOME:-${{HOME}}/.config}}"
    unit="${{config_home}}/systemd/user/personal-agent-memory.service"
    command=$(sed -n 's/^ExecStart=//p' "$unit")
    command=$(printf '%s\n' "$command" | sed 's/[$][$]/$/g')
    sh -c "$command" >"{tmp_path}/daemon.log" 2>&1 &
    echo $! >"{tmp_path}/daemon.pid"
    printf 'enabled\n' >"{tmp_path}/service-enabled"
    ;;
  '--user disable personal-agent-memory.service')
    rm -f "{tmp_path}/service-enabled"
    ;;
  '--user is-enabled --quiet personal-agent-memory.service')
    [ -e "{tmp_path}/service-enabled" ]
    ;;
  '--user is-active personal-agent-memory.service')
    [ -s "{tmp_path}/daemon.pid" ] && kill -0 "$(cat "{tmp_path}/daemon.pid")" 2>/dev/null
    printf 'active\\n'
    ;;
esac
""",
    )

    environment = {
        **os.environ,
        "HOME": str(home),
        "CODEX_HOME": str(home / ".codex"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_BIN_HOME": str(fake_bin),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    state_dir = home / ".local/share/personal-agent-memory"
    library_root = home / "memory-libraries"
    installer_arguments: list[str] = []
    if directory_mode in {
        "state-only",
        "both",
        "empty-xdg-custom-state",
        "preexisting-state",
    }:
        state_dir = home / "custom platform state"
        installer_arguments.extend(["--state-dir", "~/custom platform state"])
        if directory_mode == "preexisting-state":
            state_dir.mkdir(parents=True)
            (state_dir / "shared-user-file.txt").write_text(
                "must survive purge\n", encoding="utf-8"
            )
    elif directory_mode == "literal-dollar":
        state_dir = home / "${PAM_LITERAL}" / "state"
        installer_arguments.extend(["--state-dir", "~/${PAM_LITERAL}/state"])
        environment["PAM_LITERAL"] = "${PAM_LITERAL}"
    if directory_mode == "empty-xdg-custom-state":
        environment["XDG_CONFIG_HOME"] = ""
    if directory_mode in {"library-only", "both"}:
        library_root = tmp_path / "custom memory libraries"
        installer_arguments.extend(["--library-root", str(library_root)])
    try:
        result = subprocess.run(
            [str(ROOT / "install.sh"), *installer_arguments],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        original_key = (state_dir / "api-key").read_bytes()
        state_sentinel = state_dir / "upgrade-state-sentinel.json"
        library_sentinel = library_root / "authoritative-upgrade-memory.md"
        install_metadata_path = home / ".config/personal-agent-memory/install.json"
        install_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
        old_marketplace_source = tmp_path / "old-release"
        legacy_upgrade = directory_mode == "default" and not python_fallback
        if legacy_upgrade:
            install_metadata.pop("installed_version")
            install_metadata.pop("marketplace_ownership")
            install_metadata.pop("marketplace_source")
            install_metadata.pop("marketplace_update_pending")
        else:
            install_metadata["marketplace_source"] = str(old_marketplace_source)
        install_metadata_path.write_text(json.dumps(install_metadata), encoding="utf-8")
        (tmp_path / "marketplace-records").write_text(
            f"{old_marketplace_source}\n", encoding="utf-8"
        )
        state_sentinel.write_text('{"preserved": true}\n', encoding="utf-8")
        library_sentinel.write_text("# Preserved authoritative memory\n", encoding="utf-8")
        if not legacy_upgrade:
            (home / ".config/systemd/user/personal-agent-memory.service").unlink()
        (fake_bin / "personal-agent-memory").unlink()
        shutil.rmtree(plugin_install)

        upgrade_arguments = ["--adopt-marketplace"] if legacy_upgrade else []
        if legacy_upgrade:
            remove_count = command_log.read_text(encoding="utf-8").count(
                "codex plugin marketplace remove personal-agent-memory --json"
            )
            refused = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert refused.returncode != 0
            assert "--adopt-marketplace" in refused.stderr
            assert (tmp_path / "marketplace-records").read_text().strip() == str(
                old_marketplace_source
            )
            assert (
                command_log.read_text(encoding="utf-8").count(
                    "codex plugin marketplace remove personal-agent-memory --json"
                )
                == remove_count
            )
        result = subprocess.run(
            [str(ROOT / "install.sh"), *upgrade_arguments],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert (state_dir / "api-key").read_bytes() == original_key
        assert state_sentinel.read_text(encoding="utf-8") == '{"preserved": true}\n'
        assert library_sentinel.read_text(encoding="utf-8") == "# Preserved authoritative memory\n"
        assert (tmp_path / "marketplace-records").read_text().splitlines() == [str(ROOT)]
        assert (tmp_path / "plugin-records").read_text().splitlines() == [
            "personal-agent-memory@personal-agent-memory"
        ]
        assert (home / ".codex").is_dir()
        unit = home / ".config/systemd/user/personal-agent-memory.service"
        assert state_dir.is_dir()
        assert library_root.is_dir()
        unit_text = unit.read_text(encoding="utf-8")
        expected_state_argument = str(state_dir).replace("$", "$$")
        assert f'--state-dir "{expected_state_argument}"' in unit_text
        assert f'--library-root "{library_root}"' in unit_text
        assert '--host "127.0.0.1" --port "7331"' in unit_text
        install_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "schema_version": 1,
            "state_dir": str(state_dir),
            "library_root": str(library_root),
            "library_root_ownership": "user-content-never-delete",
            "state_dir_ownership": (
                "preexisting-unproven"
                if directory_mode == "preexisting-state"
                else "installer-created-exclusive"
            ),
            "installed_version": "0.1.0",
            "marketplace_ownership": "preexisting" if legacy_upgrade else "installer-managed",
            "marketplace_source": str(ROOT),
            "marketplace_update_pending": False,
                "uninstall_ownership": "installer-managed",
                "uninstall_path": str(fake_bin / "personal-agent-memory-uninstall"),
                "codex_home": str(home / ".codex"),
                "uv_path": str(fake_bin / "uv"),
            "daemon_path": str(fake_bin / "personal-agent-memory"),
            "unit_path": str(unit),
            "plugin_id": "personal-agent-memory@personal-agent-memory",
            "plugin_version": "0.1.0",
            "plugin_registration_source": str(
                ROOT / "plugins/personal-agent-memory"
            ),
            "plugin_install_path": str(plugin_install),
        }
        if directory_mode == "preexisting-state":
            assert "state_ownership_token" not in install_metadata
            assert not (state_dir / ".personal-agent-memory-owned.json").exists()
        else:
            assert re.fullmatch(
                r"[0-9a-f]{64}", str(install_metadata["state_ownership_token"])
            )
            expected_metadata["state_ownership_token"] = install_metadata[
                "state_ownership_token"
            ]
        assert re.fullmatch(r"[0-9a-f]{64}", str(install_metadata["uninstall_sha256"]))
        assert re.fullmatch(r"[0-9a-f]{64}", str(install_metadata["unit_sha256"]))
        expected_metadata["uninstall_sha256"] = install_metadata["uninstall_sha256"]
        expected_metadata["unit_sha256"] = install_metadata["unit_sha256"]
        assert re.fullmatch(
            r"[0-9a-f]{64}", str(install_metadata["plugin_tree_sha256"])
        )
        expected_metadata["plugin_tree_sha256"] = install_metadata[
            "plugin_tree_sha256"
        ]
        if legacy_upgrade:
            expected_metadata["marketplace_previous_source"] = str(old_marketplace_source)
        assert install_metadata == expected_metadata
        assert install_metadata_path.stat().st_mode & 0o777 == 0o600
        assert not install_metadata_path.is_relative_to(library_root)
        assert "installer-test-secret" not in install_metadata_path.read_text(encoding="utf-8")
        log = command_log.read_text(encoding="utf-8")
        expected_uv_install_calls = 5 if legacy_upgrade else 4
        assert len(
            [line for line in log.splitlines() if line.startswith("uv tool install --python")]
        ) == expected_uv_install_calls
        assert log.count("uv tool dir --bin") == (3 if legacy_upgrade else 2)
        assert log.count("uv --version") >= 1
        assert ("curl " in log) is (uv_mode != "existing")
        if uv_mode == "trimmed":
            assert "trimmed-uv tool install --python" in log
        if python_fallback:
            assert f"uv tool install --python {fake_bin / 'python3.11'}" in log
            assert "python3.11 -c" in log
            assert f"python3.11 - {home / '.local/share/personal-agent-memory/api-key'}" in log
        assert log.count(f"codex plugin marketplace add {ROOT}") == 2
        assert log.count("codex plugin marketplace remove personal-agent-memory --json") == 1
        assert log.count("codex plugin add personal-agent-memory@personal-agent-memory") == 2
        assert log.count("codex plugin list --json") == 2
        log_lines = log.splitlines()
        stop_index = log_lines.index("systemctl --user stop personal-agent-memory.service")
        installs = [
            index
            for index, line in enumerate(log_lines)
            if line.startswith("uv tool install --python") and not line.endswith("--help")
        ]
        assert stop_index < installs[1]
        assert "systemctl --user daemon-reload" in log
        assert "systemctl --user enable --now personal-agent-memory.service" in log
        assert not any(
            line.startswith(("sudo ", "apt ", "apt-get ", "dnf ", "yum ", "pacman ", "zypper "))
            for line in log.splitlines()
        )
        assert "installer-test-secret" not in result.stdout
        assert "Installed version: 0.1.0" in result.stdout
        for expected in (
            "Service status: active",
            "Web: http://127.0.0.1:7331",
            f"API key file: {state_dir / 'api-key'}",
            "systemctl --user start personal-agent-memory.service",
            "systemctl --user stop personal-agent-memory.service",
            "systemctl --user restart personal-agent-memory.service",
            "systemctl --user status personal-agent-memory.service",
            "journalctl --user -u personal-agent-memory.service",
        ):
            assert expected in result.stdout

        hook = subprocess.run(
            ["node", str(plugin_install / "scripts/recall.mjs")],
            input=json.dumps(
                {
                    "session_id": "installer-check",
                    "cwd": str(library_root),
                    "hook_event_name": "UserPromptSubmit",
                    "prompt": "health check",
                }
            ),
            env=environment,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert hook.returncode == 0
        assert hook.stdout == ""
        assert hook.stderr == ""
        capture = subprocess.run(
            ["node", str(plugin_install / "scripts/capture.mjs")],
            input=json.dumps(
                {
                    "session_id": "installer-check",
                    "cwd": str(library_root),
                    "hook_event_name": "UserPromptSubmit",
                    "prompt": "remember this installation",
                }
            ),
            env=environment,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert capture.returncode == 0
        assert capture.stdout == ""
        assert capture.stderr == ""
        assert [json.loads(line) for line in request_log.read_text().splitlines()] == [
            {
                "path": "/api/v1/search",
                "authorization": "Bearer installer-test-secret",
            },
            {
                "path": "/api/v1/capture/events",
                "authorization": "Bearer installer-test-secret",
            },
        ]
        assert (
            subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            == repo_status
        )
        assert (
            subprocess.run(
                ["git", "remote", "-v"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            == repo_remotes
        )
        if uv_mode == "existing" and not python_fallback and directory_mode == "default":
            legacy_same_source_metadata = dict(install_metadata)
            for field in (
                "installed_version",
                "marketplace_ownership",
                "marketplace_source",
                    "marketplace_update_pending",
                    "marketplace_previous_source",
                    "marketplace_update_from_source",
                    "unit_path",
                    "unit_sha256",
                ):
                legacy_same_source_metadata.pop(field, None)
            install_metadata_path.write_text(
                json.dumps(legacy_same_source_metadata),
                encoding="utf-8",
            )
            legacy_unit_contents = unit.read_bytes()
            unit.write_bytes(legacy_unit_contents.replace(b"RestartSec=2", b"RestartSec=9"))
            legacy_replacement = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert legacy_replacement.returncode != 0
            assert "already exists and is not installer-owned" in legacy_replacement.stderr
            replacement_unit_contents = legacy_unit_contents.replace(
                b"RestartSec=2", b"RestartSec=9"
            )
            assert unit.read_bytes() == replacement_unit_contents
            unit.write_bytes(legacy_unit_contents)
            legacy_failure = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env={**environment, "PAM_INSTALL_TEST_FAILURE": "program-update"},
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert legacy_failure.returncode != 0
            legacy_retry_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            if "uninstall_sha256" in legacy_retry_metadata:
                assert legacy_retry_metadata["uninstall_sha256"] == hashlib.sha256(
                    (fake_bin / "personal-agent-memory-uninstall").read_bytes()
                ).hexdigest()
            same_source_repair = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert same_source_repair.returncode == 0, same_source_repair.stderr
            same_source_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            assert same_source_metadata["marketplace_ownership"] == "legacy-unowned"
            assert same_source_metadata["marketplace_source"] == str(ROOT)
            assert same_source_metadata["marketplace_update_pending"] is False
            assert "marketplace_previous_source" not in same_source_metadata

            adopted_release = tmp_path / "adopted-release"
            adopted_release.mkdir()
            shutil.copy2(ROOT / "install.sh", adopted_release / "install.sh")
            shutil.copy2(ROOT / "uninstall.sh", adopted_release / "uninstall.sh")
            shutil.copy2(ROOT / "pyproject.toml", adopted_release / "pyproject.toml")
            shutil.copytree(
                ROOT / "plugins" / "personal-agent-memory",
                adopted_release / "plugins" / "personal-agent-memory",
            )
            for path in (
                adopted_release / "pyproject.toml",
                adopted_release
                / "plugins"
                / "personal-agent-memory"
                / ".codex-plugin"
                / "plugin.json",
            ):
                content = path.read_text(encoding="utf-8")
                path.write_text(content.replace("0.1.0", "0.1.1"), encoding="utf-8")

            remove_count = command_log.read_text(encoding="utf-8").count(
                "codex plugin marketplace remove personal-agent-memory --json"
            )
            refused_adoption = subprocess.run(
                [str(adopted_release / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert refused_adoption.returncode != 0
            assert "--adopt-marketplace" in refused_adoption.stderr
            assert (
                command_log.read_text(encoding="utf-8").count(
                    "codex plugin marketplace remove personal-agent-memory --json"
                )
                == remove_count
            )

            explicit_adoption = subprocess.run(
                [str(adopted_release / "install.sh"), "--adopt-marketplace"],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert explicit_adoption.returncode == 0, explicit_adoption.stderr
            adopted_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            assert adopted_metadata["marketplace_ownership"] == "preexisting"
            assert adopted_metadata["marketplace_source"] == str(adopted_release)
            assert adopted_metadata["marketplace_previous_source"] == str(ROOT)

            returned = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert returned.returncode == 0, returned.stderr
            returned_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            returned_metadata.pop("marketplace_previous_source")
            install_metadata_path.write_text(json.dumps(returned_metadata), encoding="utf-8")

            drifted_marketplace_source = tmp_path / "drifted-marketplace"
            (tmp_path / "marketplace-records").write_text(
                f"{drifted_marketplace_source}\n", encoding="utf-8"
            )
            remove_count = command_log.read_text(encoding="utf-8").count(
                "codex plugin marketplace remove personal-agent-memory --json"
            )
            refused_drifted_adoption = subprocess.run(
                [str(adopted_release / "install.sh"), "--adopt-marketplace"],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert refused_drifted_adoption.returncode != 0
            assert (
                command_log.read_text(encoding="utf-8").count(
                    "codex plugin marketplace remove personal-agent-memory --json"
                )
                == remove_count
            )

            (tmp_path / "marketplace-records").write_text(f"{ROOT}\n", encoding="utf-8")
            refused_preexisting_adoption = subprocess.run(
                [str(adopted_release / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert refused_preexisting_adoption.returncode != 0
            assert "--adopt-marketplace" in refused_preexisting_adoption.stderr
            assert (
                command_log.read_text(encoding="utf-8").count(
                    "codex plugin marketplace remove personal-agent-memory --json"
                )
                == remove_count
            )

            adopted_preexisting = subprocess.run(
                [str(adopted_release / "install.sh"), "--adopt-marketplace"],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert adopted_preexisting.returncode == 0, adopted_preexisting.stderr
            adopted_preexisting_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            assert adopted_preexisting_metadata["marketplace_ownership"] == "preexisting"
            assert adopted_preexisting_metadata["marketplace_source"] == str(adopted_release)
            assert adopted_preexisting_metadata["marketplace_previous_source"] == str(ROOT)

            returned = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert returned.returncode == 0, returned.stderr
            returned_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            returned_metadata["marketplace_previous_source"] = str(old_marketplace_source)
            install_metadata_path.write_text(json.dumps(returned_metadata), encoding="utf-8")

            next_release = tmp_path / "next-release"
            next_release.mkdir()
            shutil.copy2(ROOT / "install.sh", next_release / "install.sh")
            shutil.copy2(ROOT / "uninstall.sh", next_release / "uninstall.sh")
            next_uninstaller = next_release / "uninstall.sh"
            next_uninstaller.write_text(
                next_uninstaller.read_text(encoding="utf-8")
                + "\n# next release uninstaller identity\n",
                encoding="utf-8",
            )
            shutil.copy2(ROOT / "pyproject.toml", next_release / "pyproject.toml")
            shutil.copytree(
                ROOT / "plugins" / "personal-agent-memory",
                next_release / "plugins" / "personal-agent-memory",
            )
            for path in (
                next_release / "pyproject.toml",
                next_release
                / "plugins"
                / "personal-agent-memory"
                / ".codex-plugin"
                / "plugin.json",
            ):
                content = path.read_text(encoding="utf-8")
                path.write_text(content.replace("0.1.0", "0.2.0"), encoding="utf-8")

            next_upgrade = subprocess.run(
                [str(next_release / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert next_upgrade.returncode == 0, next_upgrade.stderr
            next_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            assert next_metadata["installed_version"] == "0.2.0"
            assert next_metadata["marketplace_ownership"] == "preexisting"
            assert next_metadata["marketplace_source"] == str(next_release)
            assert next_metadata["marketplace_previous_source"] == str(old_marketplace_source)
            assert next_metadata["marketplace_update_pending"] is False
            assert "marketplace_update_from_source" not in next_metadata

            returned = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert returned.returncode == 0, returned.stderr
            returned_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            assert returned_metadata["marketplace_source"] == str(ROOT)
            assert returned_metadata["marketplace_previous_source"] == str(old_marketplace_source)

            remove_failure_environment = {
                **environment,
                "PAM_INSTALL_TEST_FAILURE": "marketplace-remove",
            }
            previous_uninstall_sha256 = returned_metadata["uninstall_sha256"]
            remove_failure = subprocess.run(
                [str(next_release / "install.sh")],
                cwd=tmp_path,
                env=remove_failure_environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert remove_failure.returncode != 0
            assert "Installation failed during Codex plugin installation." in (
                remove_failure.stderr
            )
            pending_replace = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            assert pending_replace["marketplace_source"] == str(next_release)
            assert pending_replace["marketplace_previous_source"] == str(old_marketplace_source)
            assert pending_replace["marketplace_update_pending"] is True
            assert pending_replace["marketplace_update_from_source"] == str(ROOT)
            assert pending_replace["uninstall_sha256"] != previous_uninstall_sha256
            assert pending_replace["uninstall_sha256"] == hashlib.sha256(
                (fake_bin / "personal-agent-memory-uninstall").read_bytes()
            ).hexdigest()
            assert (tmp_path / "marketplace-records").read_text().strip() == str(ROOT)

            remove_retry = subprocess.run(
                [str(next_release / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert remove_retry.returncode == 0, remove_retry.stderr
            retried_replace = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            assert retried_replace["marketplace_source"] == str(next_release)
            assert retried_replace["marketplace_previous_source"] == str(old_marketplace_source)
            assert retried_replace["marketplace_update_pending"] is False
            assert "marketplace_update_from_source" not in retried_replace

            returned = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert returned.returncode == 0, returned.stderr
            metadata_before_failures = install_metadata_path.read_bytes()
            for injected_failure, expected_stage in (
                ("program-update", "Python tool installation"),
                ("plugin-update", "Codex plugin installation"),
                ("service-startup", "user service startup"),
            ):
                failed_environment = {**environment, "PAM_INSTALL_TEST_FAILURE": injected_failure}
                failed = subprocess.run(
                    [str(ROOT / "install.sh")],
                    cwd=tmp_path,
                    env=failed_environment,
                    text=True,
                    capture_output=True,
                    timeout=20,
                    check=False,
                )

                assert failed.returncode != 0
                assert f"Installation failed during {expected_stage}." in failed.stderr
                assert "systemctl --user status personal-agent-memory.service" in failed.stderr
                assert "journalctl --user -u personal-agent-memory.service" in failed.stderr
                assert install_metadata_path.read_bytes() == metadata_before_failures
                assert (state_dir / "api-key").read_bytes() == original_key
                assert state_sentinel.read_text(encoding="utf-8") == '{"preserved": true}\n'
                assert (
                    library_sentinel.read_text(encoding="utf-8")
                    == "# Preserved authoritative memory\n"
                )

            restored = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert restored.returncode == 0, restored.stderr

            command_log.write_text("", encoding="utf-8")
            install_metadata_path.unlink()
            unit.unlink()
            refused_repair = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert refused_repair.returncode != 0
            assert "stable uninstaller path already exists" in refused_repair.stderr
            assert "codex plugin marketplace list --json" not in command_log.read_text(
                encoding="utf-8"
            )

            (fake_bin / "personal-agent-memory-uninstall").unlink()
            command_log.write_text("", encoding="utf-8")
            repaired = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert repaired.returncode == 0, repaired.stderr
            repair_log = command_log.read_text(encoding="utf-8").splitlines()
            stop_index = repair_log.index("systemctl --user stop personal-agent-memory.service")
            update_index = next(
                index
                for index, line in enumerate(repair_log)
                if line.startswith("uv tool install --python") and not line.endswith("--help")
            )
            assert stop_index < update_index

            subprocess.run(
                ["systemctl", "--user", "stop", "personal-agent-memory.service"],
                env=environment,
                check=True,
            )
            install_metadata_path.unlink()
            (fake_bin / "personal-agent-memory-uninstall").unlink()
            (tmp_path / "marketplace-records").unlink()
            unit.unlink()
            failed_environment = {**environment, "PAM_INSTALL_TEST_FAILURE": "plugin-update"}
            fresh_failure = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=failed_environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert fresh_failure.returncode != 0
            assert "Installation failed during Codex plugin installation." in fresh_failure.stderr
            pending_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            assert pending_metadata["marketplace_ownership"] == "installer-managed"
            assert pending_metadata["marketplace_source"] == str(ROOT)
            assert pending_metadata["marketplace_update_pending"] is True
            assert pending_metadata["uninstall_sha256"] == hashlib.sha256(
                (fake_bin / "personal-agent-memory-uninstall").read_bytes()
            ).hexdigest()

            retry = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert retry.returncode == 0, retry.stderr
            retry_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
            assert retry_metadata["marketplace_ownership"] == "installer-managed"
            assert retry_metadata["marketplace_update_pending"] is False

            fresh_cleanup = subprocess.run(
                [str(fake_bin / "personal-agent-memory-uninstall")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert fresh_cleanup.returncode == 0, fresh_cleanup.stderr
            install_metadata_path.unlink()

            service_failure_environment = {
                **environment,
                "PAM_INSTALL_TEST_FAILURE": "service-startup",
            }
            service_failure = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=service_failure_environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert service_failure.returncode != 0
            assert "Installation failed during user service startup." in service_failure.stderr
            failed_service_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            assert failed_service_metadata["unit_path"] == str(unit)
            assert failed_service_metadata["unit_sha256"] == hashlib.sha256(
                unit.read_bytes()
            ).hexdigest()
            service_failure_uninstall = subprocess.run(
                [str(fake_bin / "personal-agent-memory-uninstall")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert service_failure_uninstall.returncode == 0, service_failure_uninstall.stderr
            assert not unit.exists()
            assert not plugin_install.exists()
            assert not (fake_bin / "personal-agent-memory").exists()
            assert (state_dir / "api-key").read_bytes() == original_key
            assert library_sentinel.read_text(encoding="utf-8") == (
                "# Preserved authoritative memory\n"
            )
            install_metadata_path.unlink()

            retryable_service_failure = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=service_failure_environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert retryable_service_failure.returncode != 0
            service_retry = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert service_retry.returncode == 0, service_retry.stderr

            replacement_cleanup = subprocess.run(
                [str(fake_bin / "personal-agent-memory-uninstall")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert replacement_cleanup.returncode == 0, replacement_cleanup.stderr
            install_metadata_path.unlink()
            replacement_service_failure = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=service_failure_environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert replacement_service_failure.returncode != 0
            unit.write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")
            refused_replacement_retry = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert refused_replacement_retry.returncode != 0
            assert "does not match install metadata" in refused_replacement_retry.stderr
            assert unit.read_text(encoding="utf-8") == "[Service]\nExecStart=/bin/true\n"
            replacement_uninstall = subprocess.run(
                [str(fake_bin / "personal-agent-memory-uninstall")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert replacement_uninstall.returncode == 0, replacement_uninstall.stderr
            assert unit.read_text(encoding="utf-8") == "[Service]\nExecStart=/bin/true\n"
            assert "installer ownership could not be verified" in replacement_uninstall.stdout
            unit.unlink()
            install_metadata_path.unlink()
            restored_after_replacement = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert restored_after_replacement.returncode == 0, restored_after_replacement.stderr

        stable_uninstaller = fake_bin / "personal-agent-memory-uninstall"
        assert stable_uninstaller.is_file()
        assert os.access(stable_uninstaller, os.X_OK)
        metadata_before_uninstall = install_metadata_path.read_bytes()
        marketplace_before_uninstall = json.loads(
            install_metadata_path.read_text(encoding="utf-8")
        )["marketplace_ownership"]
        if directory_mode == "preexisting-state":
            refused_purge = subprocess.run(
                [str(stable_uninstaller), "--purge"],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert refused_purge.returncode != 0
            assert "ownership could not be proven" in refused_purge.stderr
            assert (state_dir / "shared-user-file.txt").read_text(encoding="utf-8") == (
                "must survive purge\n"
            )
            assert install_metadata_path.is_file()
            assert stable_uninstaller.is_file()
        wrong_codex_home = tmp_path / "wrong-codex-home"
        wrong_codex_home.mkdir()
        wrong_codex_sentinel = wrong_codex_home / "must-remain"
        wrong_codex_sentinel.write_text("untouched\n", encoding="utf-8")
        uninstall = subprocess.run(
            [str(stable_uninstaller)],
            cwd=tmp_path,
            env={**environment, "CODEX_HOME": str(wrong_codex_home)},
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert uninstall.returncode == 0, uninstall.stderr
        assert not stable_uninstaller.exists()
        assert not unit.exists()
        assert not (fake_bin / "personal-agent-memory").exists()
        assert not plugin_install.exists()
        assert not (tmp_path / "plugin-records").exists()
        assert wrong_codex_sentinel.read_text(encoding="utf-8") == "untouched\n"
        assert state_dir.is_dir()
        assert (state_dir / "api-key").read_bytes() == original_key
        assert library_sentinel.read_text(encoding="utf-8") == (
            "# Preserved authoritative memory\n"
        )
        assert install_metadata_path.read_bytes() == metadata_before_uninstall
        assert f"Platform state preserved: {state_dir}" in uninstall.stdout
        assert f"API key preserved: {state_dir / 'api-key'}" in uninstall.stdout
        assert f"Install configuration preserved: {install_metadata_path}" in uninstall.stdout
        assert f"All memory libraries preserved under: {library_root}" in uninstall.stdout
        if marketplace_before_uninstall == "installer-managed":
            assert not (tmp_path / "marketplace-records").exists()
        else:
            assert (tmp_path / "marketplace-records").exists()

        uninstall_log = command_log.read_text(encoding="utf-8").splitlines()
        assert {
            line.removeprefix("codex-home ")
            for line in uninstall_log
            if line.startswith("codex-home ")
        } == {str(home / ".codex")}
        assert "systemctl --user stop personal-agent-memory.service" in uninstall_log
        assert "systemctl --user disable personal-agent-memory.service" in uninstall_log
        assert "codex plugin remove personal-agent-memory@personal-agent-memory --json" in (
            uninstall_log
        )
        assert "uv tool uninstall personal-agent-memory" in uninstall_log
        assert not any(
            line.startswith(("sudo ", "apt ", "apt-get ", "dnf ", "yum ", "pacman ", "zypper "))
            for line in uninstall_log
        )
        if uv_mode == "existing" and not python_fallback and directory_mode == "default":
            shutil.rmtree(state_dir)
            state_dir.mkdir(parents=True)
            replacement_file = state_dir / "replacement-user-file.txt"
            replacement_file.write_text("must survive\n", encoding="utf-8")

            reinstall = subprocess.run(
                [str(ROOT / "install.sh")],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert reinstall.returncode == 0, reinstall.stderr
            reinstalled_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            assert reinstalled_metadata["state_dir_ownership"] == "preexisting-unproven"
            assert "state_ownership_token" not in reinstalled_metadata
            assert not (state_dir / ".personal-agent-memory-owned.json").exists()

            refused_replacement_purge = subprocess.run(
                [str(fake_bin / "personal-agent-memory-uninstall"), "--purge"],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert refused_replacement_purge.returncode != 0
            assert "ownership could not be proven" in refused_replacement_purge.stderr
            assert replacement_file.read_text(encoding="utf-8") == "must survive\n"
            assert install_metadata_path.is_file()
    finally:
        pid_file = tmp_path / "daemon.pid"
        if pid_file.exists():
            subprocess.run(["kill", pid_file.read_text().strip()], check=False)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--state-dir", "~/same", "--library-root", "~/same"], "non-overlapping"),
        (
            ["--state-dir", "~/state", "--library-root", "~/state/libraries"],
            "non-overlapping",
        ),
        (
            ["--state-dir", "/opt/personal-agent-memory", "--library-root", "~"],
            "install metadata location",
        ),
        (
            ["--state-dir", "~/.config/personal-agent-memory"],
            "install metadata directory",
        ),
        (["--state-dir", "~/.config"], "install metadata directory"),
        (
            ["--state-dir", "~/.config/personal-agent-memory/state"],
            "install metadata directory",
        ),
        (["--state-dir", "~pam-no-such-user/state"], "could not be expanded"),
    ],
)
def test_invalid_custom_directories_fail_before_installation(
    tmp_path: Path, arguments: list[str], message: str
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    executable(
        fake_bin / "python3",
        f'#!/bin/sh\nexec {sys.executable!s} "$@"\n',
    )

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh"), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not (home / ".local/share/personal-agent-memory").exists()
    assert not (home / ".codex").exists()
    assert not (home / ".config/personal-agent-memory/install.json").exists()
    log = command_log.read_text(encoding="utf-8")
    assert "codex plugin marketplace list --json" not in log
    assert "codex plugin marketplace remove personal-agent-memory --json" not in log
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in log.splitlines()
    )
    assert "systemctl --user stop personal-agent-memory.service" not in log


def test_unusable_custom_directory_does_not_report_partial_success(tmp_path: Path) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    executable(fake_bin / "python3", f'#!/bin/sh\nexec {sys.executable!s} "$@"\n')
    unusable_state = tmp_path / "not-a-directory"
    unusable_state.write_text("occupied", encoding="utf-8")

    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "install.sh"),
            "--state-dir",
            str(unusable_state),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert str(unusable_state) in result.stderr
    assert "Could not create or write the selected install directories" in result.stderr
    assert "installation complete" not in result.stdout
    assert not (home / ".codex").exists()
    assert not (home / ".config/systemd/user/personal-agent-memory.service").exists()
    assert not (home / ".config/personal-agent-memory/install.json").exists()
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in command_log.read_text(encoding="utf-8").splitlines()
    )


def test_saved_directories_are_reused_when_arguments_are_omitted(tmp_path: Path) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    executable(fake_bin / "python3", f'#!/bin/sh\nexec {sys.executable!s} "$@"\n')
    custom_state = tmp_path / "saved-state-file"
    custom_state.write_text("occupied", encoding="utf-8")
    custom_library_root = tmp_path / "saved-memory-root"
    metadata_dir = home / ".config/personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "install.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_dir": str(custom_state),
                "library_root": str(custom_library_root),
                "library_root_ownership": "user-content-never-delete",
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert str(custom_state) in result.stderr
    assert "Could not create or write the selected install directories" in result.stderr
    assert not (home / ".local/share/personal-agent-memory").exists()
    assert not (home / "memory-libraries").exists()
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in command_log.read_text(encoding="utf-8").splitlines()
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"schema_version": 99},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": "1"},
        {"schema_version": None},
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "platform-state",
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_ownership": "installer-managed",
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_ownership": "installer-managed",
            "marketplace_source": "relative/source",
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_ownership": "shared",
            "marketplace_source": "/tmp/marketplace",
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_ownership": "preexisting",
            "marketplace_source": "/tmp/marketplace",
            "marketplace_previous_source": "relative/source",
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_ownership": "installer-managed",
            "marketplace_source": "/tmp/marketplace",
            "marketplace_update_pending": 1,
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_update_pending": True,
            "marketplace_update_from_source": "/tmp/marketplace",
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_ownership": "installer-managed",
            "marketplace_source": "/tmp/marketplace",
            "marketplace_update_pending": True,
            "marketplace_update_from_source": "relative/source",
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_ownership": "installer-managed",
            "marketplace_source": "/tmp/marketplace",
            "marketplace_update_pending": False,
            "marketplace_update_from_source": "/tmp/old-marketplace",
        },
        {
            "schema_version": 1,
            "state_dir": "/tmp/state",
            "library_root": "/tmp/memory",
            "library_root_ownership": "user-content-never-delete",
            "marketplace_ownership": "legacy-unowned",
            "marketplace_source": "/tmp/marketplace",
            "marketplace_previous_source": "/tmp/original-marketplace",
        },
    ],
)
def test_invalid_install_metadata_is_rejected_before_installation(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    metadata_dir = home / ".config/personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "install.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "existing install metadata" in result.stderr
    assert "uv tool install --python" not in command_log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "conflict_kind",
    ["first-install", "content-drift", "path-drift", "symlink"],
)
def test_installer_refuses_unowned_or_changed_stable_uninstaller_before_side_effects(
    tmp_path: Path,
    conflict_kind: str,
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    destination = fake_bin / "personal-agent-memory-uninstall"
    user_contents = b"#!/bin/sh\nprintf 'user owned\\n'\n"
    metadata_path = home / ".config/personal-agent-memory/install.json"
    metadata_before: bytes | None = None

    if conflict_kind == "symlink":
        target = tmp_path / "user-uninstaller-target"
        target.write_bytes(user_contents)
        destination.symlink_to(target)
    else:
        destination.write_bytes(user_contents)
        destination.chmod(0o755)

    if conflict_kind in {"content-drift", "path-drift"}:
        managed_contents = b"#!/bin/sh\nprintf 'managed\\n'\n"
        metadata_path.parent.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "state_dir": str(tmp_path / "state"),
            "library_root": str(tmp_path / "memory"),
            "library_root_ownership": "user-content-never-delete",
            "uninstall_ownership": "installer-managed",
            "uninstall_path": str(
                destination
                if conflict_kind == "content-drift"
                else fake_bin / "different-uninstaller"
            ),
            "uninstall_sha256": hashlib.sha256(managed_contents).hexdigest(),
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        metadata_before = metadata_path.read_bytes()

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert destination.is_symlink() if conflict_kind == "symlink" else destination.is_file()
    if conflict_kind != "symlink":
        assert destination.read_bytes() == user_contents
    if metadata_before is not None:
        assert metadata_path.read_bytes() == metadata_before
    log = command_log.read_text(encoding="utf-8").splitlines()
    assert "codex plugin marketplace list --json" not in log
    assert "systemctl --user stop personal-agent-memory.service" not in log
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in log
    )
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "memory").exists()


@pytest.mark.parametrize("line_break", ["\n", "\r"], ids=["lf", "cr"])
def test_installed_version_line_break_is_rejected_before_installer_side_effects(
    tmp_path: Path, line_break: str
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    marketplace_source = tmp_path / "preexisting-release"
    metadata_dir = home / ".config/personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "install.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_dir": str(tmp_path / "state"),
                "library_root": str(tmp_path / "memory"),
                "library_root_ownership": "user-content-never-delete",
                "installed_version": f"0.1.0{line_break}/tmp/forged-previous-source",
                "marketplace_ownership": "preexisting",
                "marketplace_source": str(marketplace_source),
                "marketplace_update_pending": False,
            }
        ),
        encoding="utf-8",
    )
    executable(
        fake_bin / "codex",
        f"""#!/bin/sh
printf 'codex %s\\n' "$*" >>"{command_log}"
case "$*" in
  '--version') printf 'codex-cli 1.0.0\\n' ;;
  'plugin add --help'|'plugin marketplace add --help'|'plugin marketplace list --help'|\
  'plugin marketplace remove --help'|'plugin list --help') printf 'Usage: --json\\n' ;;
  'plugin marketplace list --json')
    printf '{{"marketplaces":[{{"name":"personal-agent-memory","root":"%s"}}]}}\\n' \
      "{marketplace_source}"
    ;;
  'plugin marketplace remove personal-agent-memory --json') exit 99 ;;
esac
""",
    )

    result = subprocess.run(
        [str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid installed_version" in result.stderr
    log = command_log.read_text(encoding="utf-8").splitlines()
    assert "codex plugin marketplace list --json" not in log
    assert "codex plugin marketplace remove personal-agent-memory --json" not in log
    assert "systemctl --user stop personal-agent-memory.service" not in log
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help") for line in log
    )


@pytest.mark.parametrize("line_break", ["\n", "\r"], ids=["lf", "cr"])
def test_release_path_line_break_is_rejected_before_installer_side_effects(
    tmp_path: Path, line_break: str
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    release = tmp_path / f"release{line_break}forged"
    release.mkdir()
    shutil.copy2(ROOT / "install.sh", release / "install.sh")

    result = subprocess.run(
        [str(release / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Installer release path must not contain line breaks" in result.stderr
    log = command_log.read_text(encoding="utf-8") if command_log.exists() else ""
    assert "marketplace remove" not in log
    assert "marketplace add" not in log
    assert "systemctl --user stop" not in log
    assert "uv tool install" not in log
    assert not (home / ".config/personal-agent-memory/install.json").exists()


@pytest.mark.parametrize("line_break", ["\n", "\r"], ids=["lf", "cr"])
@pytest.mark.parametrize("ownership", ["preexisting", "legacy-adoption"])
def test_marketplace_root_line_break_is_rejected_before_installer_side_effects(
    tmp_path: Path, line_break: str, ownership: str
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    metadata_dir = home / ".config/personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "state_dir": str(tmp_path / "state"),
        "library_root": str(tmp_path / "memory"),
        "library_root_ownership": "user-content-never-delete",
    }
    if ownership == "preexisting":
        metadata.update(
            {
                "installed_version": "0.1.0",
                "marketplace_ownership": "preexisting",
                "marketplace_source": str(tmp_path / "original-marketplace"),
                "marketplace_update_pending": False,
            }
        )
    metadata_path = metadata_dir / "install.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_before = metadata_path.read_bytes()
    unsafe_marketplace_root = f"{tmp_path / 'marketplace'}{line_break}forged"
    marketplace_payload = json.dumps(
        {
            "marketplaces": [
                {
                    "name": "personal-agent-memory",
                    "root": unsafe_marketplace_root,
                }
            ]
        }
    )
    executable(
        fake_bin / "codex",
        f"""#!/bin/sh
printf 'codex %s\\n' "$*" >>"{command_log}"
case "$*" in
  '--version') printf 'codex-cli 1.0.0\\n' ;;
  'plugin add --help'|'plugin marketplace add --help'|'plugin marketplace list --help'|\
  'plugin marketplace remove --help'|'plugin list --help') printf 'Usage: --json\\n' ;;
  'plugin marketplace list --json') printf '%s\\n' '{marketplace_payload}' ;;
  'plugin marketplace remove personal-agent-memory --json') exit 99 ;;
esac
""",
    )

    arguments = ["--adopt-marketplace"] if ownership == "legacy-adoption" else []
    result = subprocess.run(
        [str(ROOT / "install.sh"), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Codex marketplace root must not contain line breaks" in result.stderr
    assert metadata_path.read_bytes() == metadata_before
    log = command_log.read_text(encoding="utf-8").splitlines()
    assert "codex plugin marketplace remove personal-agent-memory --json" not in log
    assert not any(
        line.startswith("codex plugin marketplace add ") and not line.endswith("--help")
        for line in log
    )
    assert "systemctl --user stop personal-agent-memory.service" not in log
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help") for line in log
    )
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "memory").exists()


def test_legacy_metadata_does_not_authorize_replacing_an_existing_marketplace(
    tmp_path: Path,
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    shared_source = tmp_path / "shared-marketplace"
    metadata_dir = home / ".config/personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "install.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_dir": str(tmp_path / "legacy-state"),
                "library_root": str(tmp_path / "legacy-memory"),
                "library_root_ownership": "user-content-never-delete",
            }
        ),
        encoding="utf-8",
    )
    executable(
        fake_bin / "codex",
        f"""#!/bin/sh
printf 'codex %s\\n' "$*" >>"{command_log}"
case "$*" in
  '--version') printf 'codex-cli 1.0.0\\n' ;;
  'plugin add --help'|'plugin marketplace add --help'|'plugin marketplace list --help'|\
  'plugin marketplace remove --help'|'plugin list --help') printf 'Usage: --json\\n' ;;
  'plugin marketplace list --json')
    printf '%s\\n' \
      '{{"marketplaces":[{{"name":"personal-agent-memory",'\
'"root":"{shared_source}"}}]}}'
    ;;
  'plugin marketplace remove personal-agent-memory --json') exit 99 ;;
esac
""",
    )

    result = subprocess.run(
        [str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--adopt-marketplace" in result.stderr
    log = command_log.read_text(encoding="utf-8")
    assert "codex plugin marketplace remove personal-agent-memory --json" not in log
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in log.splitlines()
    )
    assert not (home / ".local/share/personal-agent-memory").exists()
    assert not (home / "memory-libraries").exists()


@pytest.mark.parametrize("field", ["state_dir", "library_root"])
@pytest.mark.parametrize("invalid_kind", ["missing", "null", "empty", "relative"])
def test_invalid_persisted_directory_is_rejected_before_installation(
    tmp_path: Path, field: str, invalid_kind: str
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "state_dir": str(tmp_path / "state"),
        "library_root": str(tmp_path / "memory"),
        "library_root_ownership": "user-content-never-delete",
    }
    if invalid_kind == "missing":
        del metadata[field]
    elif invalid_kind == "null":
        metadata[field] = None
    elif invalid_kind == "empty":
        metadata[field] = ""
    else:
        metadata[field] = "relative/path"
    metadata_dir = home / ".config/personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "install.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"existing install metadata has an invalid {field}" in result.stderr
    assert "uv tool install --python" not in command_log.read_text(encoding="utf-8")


@pytest.mark.parametrize("field", ["state_dir", "library_root"])
def test_explicit_option_replaces_only_its_invalid_persisted_directory(
    tmp_path: Path, field: str
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    blocker = tmp_path / "other-persisted-directory"
    blocker.write_text("occupied", encoding="utf-8")
    metadata: dict[str, object] = {
        "schema_version": 1,
        "state_dir": str(tmp_path / "state"),
        "library_root": str(tmp_path / "memory"),
        "library_root_ownership": "user-content-never-delete",
    }
    metadata[field] = "relative/invalid"
    if field == "state_dir":
        metadata["library_root"] = str(blocker)
        arguments = ["--state-dir", str(tmp_path / "replacement-state")]
    else:
        metadata["state_dir"] = str(blocker)
        arguments = ["--library-root", str(tmp_path / "replacement-memory")]
    metadata_dir = home / ".config/personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "install.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh"), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Could not create or write the selected install directories" in result.stderr
    assert f"invalid {field}" not in result.stderr
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in command_log.read_text(encoding="utf-8").splitlines()
    )


@pytest.mark.parametrize(
    ("metadata_kind", "message"),
    [
        ("fifo", "must be a regular file"),
        ("device", "must be a regular file"),
        ("oversized", "exceeds the 65536-byte limit"),
    ],
)
def test_unsafe_install_metadata_is_rejected_without_blocking(
    tmp_path: Path, metadata_kind: str, message: str
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    metadata_dir = home / ".config/personal-agent-memory"
    metadata_dir.mkdir(parents=True)
    metadata_path = metadata_dir / "install.json"
    if metadata_kind == "fifo":
        os.mkfifo(metadata_path)
    elif metadata_kind == "device":
        metadata_path.symlink_to("/dev/null")
    else:
        metadata_path.write_bytes(b" " * (64 * 1024 + 1))

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=2,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in command_log.read_text(encoding="utf-8").splitlines()
    )


def test_directory_option_requires_a_path(tmp_path: Path) -> None:
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh"), "--state-dir"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Missing path for --state-dir" in result.stderr
    assert "unbound variable" not in result.stderr


def test_relative_xdg_config_home_is_rejected_before_installation(tmp_path: Path) -> None:
    environment, home, _ = preflight_test_environment(tmp_path)
    environment["XDG_CONFIG_HOME"] = "relative-config"

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh"), "--state-dir", str(tmp_path / "state")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "XDG_CONFIG_HOME must be an absolute path" in result.stderr
    assert not (tmp_path / "relative-config").exists()
    assert not (home / ".codex").exists()


@pytest.mark.parametrize("hook_name", ["recall.mjs", "capture.mjs"])
@pytest.mark.parametrize(
    "unsafe_input",
    ["metadata-fifo", "metadata-oversized", "key-fifo", "key-oversized", "relative-xdg"],
)
def test_installed_hooks_fail_open_for_unsafe_config_files(
    tmp_path: Path, hook_name: str, unsafe_input: str
) -> None:
    home = tmp_path / "home"
    config_home = home / ".config"
    metadata_dir = config_home / "personal-agent-memory"
    state_dir = tmp_path / "custom-state"
    home.mkdir()
    metadata_dir.mkdir(parents=True)
    state_dir.mkdir()
    metadata_path = metadata_dir / "install.json"
    metadata = json.dumps(
        {
            "schema_version": 1,
            "state_dir": str(state_dir),
            "library_root": str(tmp_path / "memory"),
            "library_root_ownership": "user-content-never-delete",
        }
    )
    if unsafe_input == "metadata-fifo":
        os.mkfifo(metadata_path)
    elif unsafe_input == "metadata-oversized":
        metadata_path.write_text(" " * (64 * 1024 + 1), encoding="utf-8")
    else:
        metadata_path.write_text(metadata, encoding="utf-8")
        key_path = state_dir / "api-key"
        if unsafe_input == "key-fifo":
            os.mkfifo(key_path)
        elif unsafe_input == "key-oversized":
            key_path.write_text("k" * 4097, encoding="utf-8")

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PERSONAL_AGENT_MEMORY_")
    }
    environment["HOME"] = str(home)
    environment["XDG_CONFIG_HOME"] = (
        "relative-config" if unsafe_input == "relative-xdg" else str(config_home)
    )
    event = {
        "session_id": "unsafe-config",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "bounded hook check",
    }

    result = subprocess.run(
        ["node", str(ROOT / "plugins/personal-agent-memory/scripts" / hook_name)],
        input=json.dumps(event),
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=2,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_installer_never_contains_privileged_or_system_service_operations() -> None:
    script = (ROOT / "install.sh").read_text(encoding="utf-8")

    assert "sudo" not in script
    assert "/etc/systemd" not in script
    assert "systemctl --user" in script


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("operating-system", "requires Linux"),
        ("architecture", "requires x86_64"),
        ("systemctl", "working systemd user service manager"),
        ("python3", "Python 3.11, 3.12, or 3.13"),
        ("git", "working Git"),
        ("node", "compatible Node.js"),
        ("node-capabilities", "fetch and AbortSignal.timeout"),
        ("codex", "compatible Codex CLI"),
        ("codex-plugin", "plugin and plugin marketplace commands"),
    ],
)
def test_dependency_failure_precedes_all_persistent_installation(
    tmp_path: Path, failure: str, message: str
) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path, failure=failure)

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not (home / ".local/share/personal-agent-memory").exists()
    assert not (home / "memory-libraries").exists()
    assert not (home / ".codex").exists()
    assert not (home / ".config/systemd/user/personal-agent-memory.service").exists()
    log = command_log.read_text(encoding="utf-8")
    if failure == "codex-plugin":
        assert "codex plugin add --help" in log
    assert "uv tool install --python" not in log
    assert "codex plugin marketplace add" not in log
    assert "codex plugin add personal-agent-memory" not in log
    assert not any(
        line.startswith(("sudo ", "apt ", "apt-get ", "dnf ", "yum ", "pacman ", "zypper "))
        for line in log.splitlines()
    )


def test_uv_bootstrap_failure_leaves_no_partial_installation(tmp_path: Path) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    executable(
        fake_bin / "uv",
        f"""#!/bin/sh
echo "trimmed-uv $*" >>"{command_log}"
case "$*" in
  '--version'|'tool install --help') exit 0 ;;
esac
if [ "${{1:-}} ${{2:-}} ${{3:-}} ${{5:-}} ${{7:-}}" = \
  'tool install --python --from --help' ]; then
  exit 0
fi
exit 1
""",
    )
    executable(fake_bin / "curl", f"#!/bin/sh\necho 'curl $*' >>\"{command_log}\"\nexit 23\n")

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Could not install uv in the user environment" in result.stderr
    assert not (home / ".local/share/personal-agent-memory").exists()
    assert not (home / "memory-libraries").exists()
    assert not (home / ".codex").exists()
    log = command_log.read_text(encoding="utf-8")
    assert "trimmed-uv tool install --python" in log
    assert "curl " in log
    assert not any(
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in log.splitlines()
    )
    assert not any(
        line.startswith(("sudo ", "apt ", "apt-get ", "dnf ", "yum ", "pacman ", "zypper "))
        for line in log.splitlines()
    )


def test_uv_bootstrap_temp_file_failure_is_actionable(tmp_path: Path) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    executable(fake_bin / "uv", f"#!/bin/sh\necho 'uv $*' >>\"{command_log}\"\nexit 1\n")
    executable(fake_bin / "mktemp", "#!/bin/sh\nexit 1\n")

    result = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Could not install uv in the user environment" in result.stderr
    assert not (home / ".local/share/personal-agent-memory").exists()
    assert not (home / "memory-libraries").exists()
    assert not (home / ".codex").exists()
    assert "curl " not in command_log.read_text(encoding="utf-8")


def test_real_lifecycle_verifier_reports_reproducible_skip_gate() -> None:
    environment = {**os.environ}
    environment.pop("PAM_INSTALL_ACCEPTANCE_DEDICATED_USER", None)

    result = subprocess.run(
        [str(ROOT / "scripts/verify-linux-installer.sh"), "--check"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 77
    assert "PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1" in result.stderr

    script = (ROOT / "scripts/verify-linux-installer.sh").read_text(encoding="utf-8")
    assert '"${VERSION_ID:-}" =~ ^(22\\.04|24\\.04)$' in script
    assert 'python_command="$(command -v "$candidate")"' in script
    assert "Python 3.11, 3.12, or 3.13 is required" in script
    assert "python3 - " not in script
    assert 'managed_bin_dir="${XDG_BIN_HOME:-$HOME/.local/bin}"' in script
    assert 'PATH="$managed_bin_dir:$PATH"; export PATH' in script
    assert '"$unit_path" "$codex_home" "$acceptance_scenario" <<\x27PY\x27' in script
    assert '"$unit_path" "$CODEX_HOME" "$acceptance_scenario"' not in script
    assert script.index(
        'skip "Personal Agent Memory command already exists for this user"'
    ) < script.index('managed_bin_dir="${XDG_BIN_HOME:-$HOME/.local/bin}"')
    assert script.count('"$python_command" - ') == 14
    assert '"$repo_root/install.sh"' in script
    assert "PAM_INSTALL_ACCEPTANCE_CUSTOM_DIRS" in script
    assert "systemctl --user is-enabled" in script
    assert "systemctl --user is-active" in script
    assert 'systemctl --user restart "$service_name"' in script
    assert '"library_root_ownership": "user-content-never-delete"' in script
    assert 'legacy_commit="9383eba03a8ec8901ec51c29a467265cec86c433"' in script
    assert 'git -C "$repo_root" archive "$legacy_commit"' in script
    assert '"preexisting" if scenario == "normal" else "installer-managed"' in script
    assert '"marketplace_update_pending": False' in script
    assert '"$old_release/install.sh"' in script
    assert script.count('"$repo_root/install.sh"') >= 2
    assert '[[ "$(personal-agent-memory --version)" == "$target_version" ]]' in script
    assert 'cmp "$work_root/state.before.json" "$work_root/state.after.json"' in script
    assert "codex plugin list --json" in script
    after_install = script.split(
        '"$repo_root/install.sh" >"$work_root/install-reinstall.out"', maxsplit=1
    )[1]
    assert "codex plugin add" not in after_install
    assert 'installed_path = source.get("path")' in after_install
    assert 'node "$managed_plugin_install_path/scripts/recall.mjs"' in script
    assert "grep -q 'Installer acceptance fact'" in script
    assert script.index("grep -q 'Installer acceptance fact'") < script.index(
        '"$work_root/state.after.json"'
    )
    assert 'PAM_INSTALL_ACCEPTANCE_SCENARIO=normal "$0"' in script
    assert 'PAM_INSTALL_ACCEPTANCE_SCENARIO=purge "$0"' in script
    assert '"$uninstall_command" >"$work_root/uninstall.out"' in script
    assert '"$uninstall_command" --purge >"$work_root/purge.out"' in script
    assert '"$repo_root/uninstall.sh" --purge' not in script
    assert script.count('[[ ! -e "$state_dir" ]]') >= 2
    assert script.count('[[ ! -e "$install_config_path" ]]') >= 2
    assert 'codex plugin marketplace list --json' in script
    assert '"$managed_uv_path" tool list >"$work_root/tools-after-uninstall.txt"' in script
    assert 'cmp "$work_root/state.before.json" "$work_root/state.after-uninstall.json"' in script
    upgrade_branch = script[script.index('if [[ "$acceptance_scenario" == "normal" ]]'):]
    assert upgrade_branch.index('codex plugin marketplace remove') < upgrade_branch.index(
        '"$repo_root/install.sh" >"$work_root/install-upgrade.out"'
    )
    assert '"installer-managed"' in upgrade_branch
    assert "sudo" not in script


@pytest.mark.parametrize(
    ("scenario", "marketplaces"),
    [
        (
            "normal",
            [{"name": "personal-agent-memory", "root": "/tmp/preexisting"}],
        ),
        ("purge", []),
    ],
)
def test_real_lifecycle_post_uninstall_contract_is_behavioral(
    tmp_path: Path,
    scenario: str,
    marketplaces: list[dict[str, str]],
) -> None:
    script = (ROOT / "scripts/verify-linux-installer.sh").read_text(encoding="utf-8")
    marker = "# lifecycle-post-uninstall-contract\n"
    contract = script.split(marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]
    plugin_payload = tmp_path / "plugins.json"
    marketplace_payload = tmp_path / "marketplaces.json"
    tool_list = tmp_path / "tools.txt"
    plugin_payload.write_text('{"installed": []}\n', encoding="utf-8")
    marketplace_payload.write_text(
        json.dumps({"marketplaces": marketplaces}), encoding="utf-8"
    )
    tool_list.write_text("another-tool v1.0.0\n", encoding="utf-8")

    contract_argv = [
        "contract",
        scenario,
        str(plugin_payload),
        str(marketplace_payload),
        str(tool_list),
    ]
    original_argv = sys.argv
    try:
        sys.argv = contract_argv
        exec(compile(contract, "lifecycle-post-uninstall-contract", "exec"), {})

        wrong_marketplaces = [] if scenario == "normal" else marketplaces + [
            {"name": "personal-agent-memory", "root": "/tmp/unexpected"}
        ]
        marketplace_payload.write_text(
            json.dumps({"marketplaces": wrong_marketplaces}), encoding="utf-8"
        )
        with pytest.raises(AssertionError):
            exec(compile(contract, "lifecycle-post-uninstall-contract", "exec"), {})

        marketplace_payload.write_text(
            json.dumps({"marketplaces": marketplaces}), encoding="utf-8"
        )
        tool_list.write_text("personal-agent-memory v0.1.0\n", encoding="utf-8")
        with pytest.raises(AssertionError):
            exec(compile(contract, "lifecycle-post-uninstall-contract", "exec"), {})
    finally:
        sys.argv = original_argv


def test_real_lifecycle_snapshot_tables_match_product_database_schema(
    tmp_path: Path,
) -> None:
    async def create_database() -> None:
        state = PlatformState(tmp_path / "platform.sqlite3")
        await state.start()
        await state.close()

    asyncio.run(create_database())
    script = (ROOT / "scripts/verify-linux-installer.sh").read_text(encoding="utf-8")
    table_blocks = re.findall(
        r"tables = \(\n((?:    \"[a-z_]+\",\n)+)\)",
        script,
    )
    snapshot_tables = [ast.literal_eval(f"({block})") for block in table_blocks]

    assert len(snapshot_tables) == 3
    assert snapshot_tables[0] == snapshot_tables[1] == snapshot_tables[2]
    assert "capture_inbox" in snapshot_tables[0]
    assert script.count('    "capture_inbox",') == 4
    assert '    "capture_events",' not in script
    with sqlite3.connect(tmp_path / "platform.sqlite3") as connection:
        for tables in snapshot_tables:
            for table in tables:
                connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()


def test_real_lifecycle_verifier_selects_supported_versioned_python(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    command_log = tmp_path / "commands.log"
    fake_bin.mkdir()
    executable(
        fake_bin / "uname",
        '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo x86_64 ;; esac\n',
    )
    executable(
        fake_bin / "python3",
        f'#!/bin/sh\nprintf \'python3 %s\\n\' "$*" >>"{command_log}"\nexit 1\n',
    )
    executable(
        fake_bin / "python3.11",
        f'#!/bin/sh\nprintf \'python3.11 %s\\n\' "$*" >>"{command_log}"\n[ "${{1:-}}" = \'-c\' ]\n',
    )
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "PAM_INSTALL_ACCEPTANCE_DEDICATED_USER": "1",
    }

    result = subprocess.run(
        [str(ROOT / "scripts/verify-linux-installer.sh"), "--check"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 77
    assert "Ubuntu 22.04 or 24.04 is required" in result.stderr
    version_check = (
        "-c import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)"
    )
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        f"python3 {version_check}",
        f"python3.11 {version_check}",
    ]
