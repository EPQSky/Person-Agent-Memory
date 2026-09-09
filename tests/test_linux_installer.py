from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from personal_agent_memory.state import PlatformState

ROOT = Path(__file__).parents[1]


def executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


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
        "uv": "echo 'uv 0.8.0'\n",
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
    plugin_install = tmp_path / "installed-plugin"
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
if [ "$*" = 'plugin list --json' ]; then
  version=$(cat "{tmp_path}/plugin-version-record")
  printf '{{"installed":[{{"pluginId":"personal-agent-memory@personal-agent-memory",'\
'"version":"%s","enabled":true,"source":{{"source":"local",'\
'"path":"{plugin_install}"}}}}]}}\\n' "$version"
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
    if directory_mode in {"state-only", "both", "empty-xdg-custom-state"}:
        state_dir = home / "custom platform state"
        installer_arguments.extend(["--state-dir", "~/custom platform state"])
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
            assert command_log.read_text(encoding="utf-8").count(
                "codex plugin marketplace remove personal-agent-memory --json"
            ) == remove_count
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
            "installed_version": "0.1.0",
            "marketplace_ownership": "preexisting" if legacy_upgrade else "installer-managed",
            "marketplace_source": str(ROOT),
            "marketplace_update_pending": False,
        }
        if legacy_upgrade:
            expected_metadata["marketplace_previous_source"] = str(old_marketplace_source)
        assert install_metadata == expected_metadata
        assert install_metadata_path.stat().st_mode & 0o777 == 0o600
        assert not install_metadata_path.is_relative_to(library_root)
        assert "installer-test-secret" not in install_metadata_path.read_text(encoding="utf-8")
        log = command_log.read_text(encoding="utf-8")
        assert (
            len([line for line in log.splitlines() if line.startswith("uv tool install --python")])
            == 4
        )
        assert log.count("uv tool dir --bin") == 2
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
            install_metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state_dir": str(state_dir),
                        "library_root": str(library_root),
                        "library_root_ownership": "user-content-never-delete",
                    }
                ),
                encoding="utf-8",
            )
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
            same_source_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            assert same_source_metadata["marketplace_ownership"] == "legacy-unowned"
            assert same_source_metadata["marketplace_source"] == str(ROOT)
            assert same_source_metadata["marketplace_update_pending"] is False
            assert "marketplace_previous_source" not in same_source_metadata

            adopted_release = tmp_path / "adopted-release"
            adopted_release.mkdir()
            shutil.copy2(ROOT / "install.sh", adopted_release / "install.sh")
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
            assert command_log.read_text(encoding="utf-8").count(
                "codex plugin marketplace remove personal-agent-memory --json"
            ) == remove_count

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
            adopted_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
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
            returned_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            returned_metadata.pop("marketplace_previous_source")
            install_metadata_path.write_text(
                json.dumps(returned_metadata), encoding="utf-8"
            )

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
            assert command_log.read_text(encoding="utf-8").count(
                "codex plugin marketplace remove personal-agent-memory --json"
            ) == remove_count

            (tmp_path / "marketplace-records").write_text(
                f"{ROOT}\n", encoding="utf-8"
            )
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
            assert command_log.read_text(encoding="utf-8").count(
                "codex plugin marketplace remove personal-agent-memory --json"
            ) == remove_count

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
            assert adopted_preexisting_metadata["marketplace_source"] == str(
                adopted_release
            )
            assert adopted_preexisting_metadata["marketplace_previous_source"] == str(
                ROOT
            )

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
            returned_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            returned_metadata["marketplace_previous_source"] = str(
                old_marketplace_source
            )
            install_metadata_path.write_text(
                json.dumps(returned_metadata), encoding="utf-8"
            )

            next_release = tmp_path / "next-release"
            next_release.mkdir()
            shutil.copy2(ROOT / "install.sh", next_release / "install.sh")
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
            assert next_metadata["marketplace_previous_source"] == str(
                old_marketplace_source
            )
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
            returned_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            assert returned_metadata["marketplace_source"] == str(ROOT)
            assert returned_metadata["marketplace_previous_source"] == str(
                old_marketplace_source
            )

            remove_failure_environment = {
                **environment,
                "PAM_INSTALL_TEST_FAILURE": "marketplace-remove",
            }
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
            pending_replace = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            assert pending_replace["marketplace_source"] == str(next_release)
            assert pending_replace["marketplace_previous_source"] == str(
                old_marketplace_source
            )
            assert pending_replace["marketplace_update_pending"] is True
            assert pending_replace["marketplace_update_from_source"] == str(ROOT)
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
            retried_replace = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            assert retried_replace["marketplace_source"] == str(next_release)
            assert retried_replace["marketplace_previous_source"] == str(
                old_marketplace_source
            )
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
            stop_index = repair_log.index(
                "systemctl --user stop personal-agent-memory.service"
            )
            update_index = next(
                index
                for index, line in enumerate(repair_log)
                if line.startswith("uv tool install --python")
                and not line.endswith("--help")
            )
            assert stop_index < update_index

            subprocess.run(
                ["systemctl", "--user", "stop", "personal-agent-memory.service"],
                env=environment,
                check=True,
            )
            install_metadata_path.unlink()
            (tmp_path / "marketplace-records").unlink()
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
            pending_metadata = json.loads(
                install_metadata_path.read_text(encoding="utf-8")
            )
            assert pending_metadata["marketplace_ownership"] == "installer-managed"
            assert pending_metadata["marketplace_source"] == str(ROOT)
            assert pending_metadata["marketplace_update_pending"] is True

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
        f"#!/bin/sh\nexec {sys.executable!s} \"$@\"\n",
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
    assert "uv tool install --python" not in command_log.read_text(encoding="utf-8")


def test_unusable_custom_directory_does_not_report_partial_success(tmp_path: Path) -> None:
    environment, home, command_log = preflight_test_environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    executable(fake_bin / "python3", f"#!/bin/sh\nexec {sys.executable!s} \"$@\"\n")
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
    executable(fake_bin / "python3", f"#!/bin/sh\nexec {sys.executable!s} \"$@\"\n")
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
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in log
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
        line.startswith("uv tool install --python") and not line.endswith("--help")
        for line in log
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
    assert script.count('"$python_command" - ') == 12
    assert '"$repo_root/install.sh"' in script
    assert "PAM_INSTALL_ACCEPTANCE_CUSTOM_DIRS" in script
    assert "systemctl --user is-enabled" in script
    assert "systemctl --user is-active" in script
    assert 'systemctl --user restart "$service_name"' in script
    assert '"library_root_ownership": "user-content-never-delete"' in script
    assert 'legacy_commit="9383eba03a8ec8901ec51c29a467265cec86c433"' in script
    assert 'git -C "$repo_root" archive "$legacy_commit"' in script
    assert '"marketplace_ownership": "preexisting"' in script
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
    assert 'node "$plugin_root/scripts/recall.mjs"' in script
    assert "grep -q 'Installer acceptance fact'" in script
    assert script.index("grep -q 'Installer acceptance fact'") < script.index(
        '"$work_root/state.after.json"'
    )
    assert "sudo" not in script


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

    assert len(snapshot_tables) == 2
    assert snapshot_tables[0] == snapshot_tables[1]
    assert "capture_inbox" in snapshot_tables[0]
    assert script.count('    "capture_inbox",') == 3
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
        "#!/bin/sh\ncase \"$1\" in -s) echo Linux ;; -m) echo x86_64 ;; esac\n",
    )
    executable(
        fake_bin / "python3",
        f'#!/bin/sh\nprintf \'python3 %s\\n\' "$*" >>"{command_log}"\nexit 1\n',
    )
    executable(
        fake_bin / "python3.11",
        f'#!/bin/sh\nprintf \'python3.11 %s\\n\' "$*" >>"{command_log}"\n'
        "[ \"${1:-}\" = '-c' ]\n",
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
        "-c import sys; raise SystemExit(0 if (3, 11) <= "
        "sys.version_info[:2] < (3, 14) else 1)"
    )
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        f"python3 {version_check}",
        f"python3.11 {version_check}",
    ]
