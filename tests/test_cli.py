from __future__ import annotations

import socket

from typer.testing import CliRunner

from personal_agent_memory.cli import cli


def test_serve_requires_state_directory() -> None:
    result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code != 0
    assert "--state-dir" in result.output


def test_serve_rejects_non_loopback_before_starting(tmp_path) -> None:
    result = CliRunner().invoke(
        cli,
        ["serve", "--state-dir", str(tmp_path), "--host", "0.0.0.0"],
    )
    assert result.exit_code == 2
    assert "loopback" in result.output + result.stderr


def test_serve_reports_port_occupation(tmp_path, monkeypatch) -> None:
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    port = occupied.getsockname()[1]
    monkeypatch.setattr("personal_agent_memory.cli.uvicorn.run", lambda *args, **kwargs: None)

    try:
        result = CliRunner().invoke(
            cli,
            ["serve", "--state-dir", str(tmp_path), "--port", str(port)],
        )
    finally:
        occupied.close()

    assert result.exit_code == 2
    assert "already in use" in result.output + result.stderr
