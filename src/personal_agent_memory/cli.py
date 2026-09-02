from __future__ import annotations

import socket
from pathlib import Path

import typer
import uvicorn

from personal_agent_memory.app import create_app
from personal_agent_memory.config import ConfigurationError, Settings

cli = typer.Typer(no_args_is_help=True, add_completion=False)


@cli.callback()
def root() -> None:
    """Manage the local Personal Agent Memory daemon."""


def ensure_port_available(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as error:
        raise ConfigurationError(f"port {port} on {host} is already in use") from error
    finally:
        probe.close()


@cli.command()
def serve(
    state_dir: str = typer.Option(..., help="Directory for protected platform state."),
    library_root: list[str] | None = typer.Option(  # noqa: B008
        None, "--library-root", help="Allowed parent directory for memory libraries."
    ),
    host: str = typer.Option("127.0.0.1", help="Loopback listen address."),
    port: int = typer.Option(7331, min=1, max=65535),
) -> None:
    """Run the single local memory daemon."""
    try:
        settings = Settings(
            state_dir=Path(state_dir),
            host=host,
            port=port,
            library_roots=tuple(Path(root) for root in library_root or ()),
        )
        ensure_port_available(settings.host, settings.port)
    except ConfigurationError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)


def main() -> None:
    cli()
