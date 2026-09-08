from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
        "python3": """[ "$1" = '-c' ] && exit 0\nexit 97\n""",
        "python3.13": "exit 1\n",
        "python3.12": "exit 1\n",
        "python3.11": "exit 1\n",
        "git": "echo 'git version 2.43.0'\n",
        "node": "echo 'v20.0.0'\n",
        "codex": """case "$*" in
  '--version') echo 'codex-cli 1.0.00' ;;
  'plugin add --help'|'plugin marketplace add --help') echo 'Usage: --json' ;;
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
    ("uv_mode", "python_fallback"),
    [("existing", False), ("missing", False), ("trimmed", False), ("existing", True)],
    ids=["existing-uv", "bootstrapped-uv", "trimmed-uv", "versioned-python"],
)
def test_supported_environment_installs_service_plugin_and_healthy_hook(
    tmp_path: Path, uv_mode: str, python_fallback: bool
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
(state / 'api-key').write_text('installer-test-secret\\n', encoding='utf-8')
(state / 'api-key').chmod(0o600)

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
cat >"{fake_bin}/personal-agent-memory" <<'EOF'
#!/bin/sh
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
if [ "$*" = 'plugin add --help' ] || [ "$*" = 'plugin marketplace add --help' ]; then
  printf 'Usage: codex %s [--json]\\n' "$*"
  exit 0
fi
if [ "$1 $2 $3" = 'plugin marketplace add' ]; then
  mkdir -p "{plugin_install.parent}"
fi
if [ "$1 $2" = 'plugin add' ]; then
  rm -rf "{plugin_install}"
  cp -R "{ROOT / "plugins" / "personal-agent-memory"}" "{plugin_install}"
fi
""",
    )
    executable(
        fake_bin / "systemctl",
        f"""#!/bin/sh
set -eu
printf 'systemctl %s\\n' "$*" >>"{command_log}"
case "$*" in
  '--user enable --now personal-agent-memory.service')
    unit="${{XDG_CONFIG_HOME}}/systemd/user/personal-agent-memory.service"
    command=$(sed -n 's/^ExecStart=//p' "$unit")
    sh -c "$command" >"{tmp_path}/daemon.log" 2>&1 &
    echo $! >"{tmp_path}/daemon.pid"
    ;;
  '--user is-active personal-agent-memory.service') printf 'active\\n' ;;
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
    try:
        result = subprocess.run(
            [str(ROOT / "install.sh")],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        state_dir = home / ".local/share/personal-agent-memory"
        library_root = home / "memory-libraries"
        assert (home / ".codex").is_dir()
        unit = home / ".config/systemd/user/personal-agent-memory.service"
        assert state_dir.is_dir()
        assert library_root.is_dir()
        unit_text = unit.read_text(encoding="utf-8")
        assert f'--state-dir "{state_dir}"' in unit_text
        assert f'--library-root "{library_root}"' in unit_text
        assert '--host "127.0.0.1" --port "7331"' in unit_text
        log = command_log.read_text(encoding="utf-8")
        assert (
            len([line for line in log.splitlines() if line.startswith("uv tool install --python")])
            == 2
        )
        assert log.count("uv tool dir --bin") == 1
        assert log.count("uv --version") >= 1
        assert ("curl " in log) is (uv_mode != "existing")
        if uv_mode == "trimmed":
            assert "trimmed-uv tool install --python" in log
        if python_fallback:
            assert f"uv tool install --python {fake_bin / 'python3.11'}" in log
            assert "python3.11 -c" in log
            assert f"python3.11 - {home / '.local/share/personal-agent-memory/api-key'}" in log
        assert log.count(f"codex plugin marketplace add {ROOT}") == 1
        assert log.count("codex plugin add personal-agent-memory@personal-agent-memory") == 1
        assert "systemctl --user daemon-reload" in log
        assert "systemctl --user enable --now personal-agent-memory.service" in log
        assert not any(
            line.startswith(("sudo ", "apt ", "apt-get ", "dnf ", "yum ", "pacman ", "zypper "))
            for line in log.splitlines()
        )
        assert "installer-test-secret" not in result.stdout
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
            env={
                **environment,
                "PERSONAL_AGENT_MEMORY_API_KEY_FILE": str(state_dir / "api-key"),
            },
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert hook.returncode == 0
        assert hook.stdout == ""
        assert hook.stderr == ""
        assert [json.loads(line) for line in request_log.read_text().splitlines()] == [
            {
                "path": "/api/v1/search",
                "authorization": "Bearer installer-test-secret",
            }
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
    finally:
        pid_file = tmp_path / "daemon.pid"
        if pid_file.exists():
            subprocess.run(["kill", pid_file.read_text().strip()], check=False)


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
    assert '"$repo_root/install.sh"' in script
    assert "systemctl --user is-enabled" in script
    assert "systemctl --user is-active" in script
    assert "codex plugin list --json" in script
    assert 'node "$plugin_root/scripts/recall.mjs"' in script
    assert "grep -q 'Installer acceptance fact'" in script
    assert "sudo" not in script
