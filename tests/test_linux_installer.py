from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_supported_environment_installs_service_plugin_and_healthy_hook(tmp_path: Path) -> None:
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
    executable(
        fake_bin / "uv",
        f"""#!/bin/sh
set -eu
printf 'uv %s\\n' "$*" >>"{command_log}"
if [ "$*" = 'tool dir --bin' ]; then
  printf '%s\\n' "{fake_bin}"
  exit 0
fi
cat >"{fake_bin}/personal-agent-memory" <<'EOF'
#!/bin/sh
exec "{os.environ.get('PYTHON', 'python3')}" "{daemon_script}" "$@"
EOF
chmod +x "{fake_bin}/personal-agent-memory"
""",
    )
    executable(
        fake_bin / "codex",
        f"""#!/bin/sh
set -eu
printf 'codex %s\\n' "$*" >>"{command_log}"
if [ "$1 $2 $3" = 'plugin marketplace add' ]; then
  mkdir -p "{plugin_install.parent}"
fi
if [ "$1 $2" = 'plugin add' ]; then
  rm -rf "{plugin_install}"
  cp -R "{ROOT / 'plugins' / 'personal-agent-memory'}" "{plugin_install}"
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
        assert log.count("uv tool install") == 1
        assert log.count("uv tool dir --bin") == 1
        assert log.count(f"codex plugin marketplace add {ROOT}") == 1
        assert log.count("codex plugin add personal-agent-memory@personal-agent-memory") == 1
        assert "systemctl --user daemon-reload" in log
        assert "systemctl --user enable --now personal-agent-memory.service" in log
        assert "sudo" not in log
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
