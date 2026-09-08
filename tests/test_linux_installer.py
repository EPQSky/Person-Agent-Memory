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
        "python3": f'exec "{sys.executable}" "$@"\n',
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
    config_home="${{XDG_CONFIG_HOME:-${{HOME}}/.config}}"
    unit="${{config_home}}/systemd/user/personal-agent-memory.service"
    command=$(sed -n 's/^ExecStart=//p' "$unit")
    command=$(printf '%s\n' "$command" | sed 's/[$][$]/$/g')
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
        assert (home / ".codex").is_dir()
        unit = home / ".config/systemd/user/personal-agent-memory.service"
        assert state_dir.is_dir()
        assert library_root.is_dir()
        unit_text = unit.read_text(encoding="utf-8")
        expected_state_argument = str(state_dir).replace("$", "$$")
        assert f'--state-dir "{expected_state_argument}"' in unit_text
        assert f'--library-root "{library_root}"' in unit_text
        assert '--host "127.0.0.1" --port "7331"' in unit_text
        install_metadata_path = home / ".config/personal-agent-memory/install.json"
        install_metadata = json.loads(install_metadata_path.read_text(encoding="utf-8"))
        assert install_metadata == {
            "schema_version": 1,
            "state_dir": str(state_dir),
            "library_root": str(library_root),
            "library_root_ownership": "user-content-never-delete",
        }
        assert install_metadata_path.stat().st_mode & 0o777 == 0o600
        assert not install_metadata_path.is_relative_to(library_root)
        assert "installer-test-secret" not in install_metadata_path.read_text(encoding="utf-8")
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
    assert '"$repo_root/install.sh"' in script
    assert "PAM_INSTALL_ACCEPTANCE_CUSTOM_DIRS" in script
    assert "systemctl --user is-enabled" in script
    assert "systemctl --user is-active" in script
    assert 'systemctl --user restart "$service_name"' in script
    assert '"library_root_ownership": "user-content-never-delete"' in script
    assert "codex plugin list --json" in script
    assert 'node "$plugin_root/scripts/recall.mjs"' in script
    assert "grep -q 'Installer acceptance fact'" in script
    assert "sudo" not in script
