from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest


def fake_docker_environment(tmp_path: Path, *, mode: str = "success") -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
        'case "$*" in *" up "*)\n'
        '  if [ "${DOCKER_MODE:-success}" = "block" ]; then\n'
        "    trap 'exit 0' INT TERM\n"
        "    while :; do sleep 1; done\n"
        "  fi\n"
        '  if [ "${DOCKER_MODE:-success}" = "ignore-signal" ]; then\n'
        "    trap '' INT TERM\n"
        "    while :; do sleep 1; done\n"
        "  fi\n"
        '  exit "${DOCKER_UP_EXIT:-0}"\n'
        ";; esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "DOCKER_LOG": str(tmp_path / "docker.log"),
            "DOCKER_MODE": mode,
            "DOCKER_UP_EXIT": "7" if mode == "fail" else "0",
        }
    )
    return environment


def run_verifier(
    repo_root: Path, tmp_path: Path, *, mode: str = "success"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo_root / "scripts" / "verify-docker.sh")],
        cwd=repo_root,
        env=fake_docker_environment(tmp_path, mode=mode),
        check=False,
        capture_output=True,
        text=True,
    )


def test_docker_verifier_is_isolated_repeatable_and_always_cleans_up(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]

    first = run_verifier(repo_root, tmp_path)
    second = run_verifier(repo_root, tmp_path)

    assert first.returncode == second.returncode == 0
    calls = (tmp_path / "docker.log").read_text(encoding="utf-8").splitlines()
    project_names = [call.split()[2] for call in calls if " up " in f" {call} "]
    assert len(project_names) == 2
    assert project_names[0] != project_names[1]
    for project_name in project_names:
        project_calls = [call for call in calls if f"-p {project_name} " in call]
        expected_down = "down --volumes --remove-orphans --rmi local"
        assert project_calls[0].endswith(expected_down)
        expected_up = "up --build --abort-on-container-exit --exit-code-from acceptance"
        assert expected_up in project_calls[1]
        assert project_calls[2].endswith(expected_down)


def test_docker_verifier_preserves_failure_status_and_cleans_up(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]

    result = run_verifier(repo_root, tmp_path, mode="fail")

    assert result.returncode == 7
    calls = (tmp_path / "docker.log").read_text(encoding="utf-8").splitlines()
    assert calls[-1].endswith("down --volumes --remove-orphans --rmi local")


@pytest.mark.parametrize(
    ("sent_signal", "expected_status"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_docker_verifier_forwards_signal_and_cleans_up_with_bounded_exit(
    tmp_path: Path, sent_signal: signal.Signals, expected_status: int
) -> None:
    repo_root = Path(__file__).parents[1]
    process = subprocess.Popen(
        [str(repo_root / "scripts" / "verify-docker.sh")],
        cwd=repo_root,
        env=fake_docker_environment(tmp_path, mode="block"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path = tmp_path / "docker.log"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if log_path.exists() and " up " in f" {log_path.read_text(encoding='utf-8')} ":
            break
        time.sleep(0.02)
    else:
        process.kill()
        raise AssertionError("verifier did not start its Compose child")

    process.send_signal(sent_signal)
    process.communicate(timeout=5)

    assert process.returncode == expected_status
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert calls[-1].endswith("down --volumes --remove-orphans --rmi local")


def test_docker_verifier_force_stops_an_unresponsive_child(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    process = subprocess.Popen(
        [str(repo_root / "scripts" / "verify-docker.sh")],
        cwd=repo_root,
        env=fake_docker_environment(tmp_path, mode="ignore-signal"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path = tmp_path / "docker.log"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if log_path.exists() and " up " in f" {log_path.read_text(encoding='utf-8')} ":
            break
        time.sleep(0.02)
    else:
        process.kill()
        raise AssertionError("verifier did not start its Compose child")

    started = time.monotonic()
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=4)

    assert process.returncode == 143
    assert time.monotonic() - started < 4
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert calls[-1].endswith("down --volumes --remove-orphans --rmi local")
