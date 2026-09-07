from __future__ import annotations

import socket
from pathlib import Path

import typer
import uvicorn

from personal_agent_memory.app import create_app
from personal_agent_memory.config import ConfigurationError, Settings
from personal_agent_memory.model_client import ModelEndpoint

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
    embedding_url: str | None = typer.Option(None, help="OpenAI-compatible embedding base URL."),
    embedding_model: str = typer.Option("text-embedding", help="Embedding model name."),
    embedding_api_key_file: str | None = typer.Option(
        None, help="File containing the embedding service API key."
    ),
    reranker_url: str | None = typer.Option(None, help="OpenAI-compatible reranker base URL."),
    reranker_model: str = typer.Option("reranker", help="Reranker model name."),
    reranker_api_key_file: str | None = typer.Option(
        None, help="File containing the reranker service API key."
    ),
    graph_url: str | None = typer.Option(None, help="OpenAI-compatible graph LLM base URL."),
    graph_model: str = typer.Option("graph-extraction", help="Graph LLM model name."),
    graph_api_key_file: str | None = typer.Option(
        None, help="File containing the graph LLM API key."
    ),
    model_timeout: float = typer.Option(2.0, min=0.05, max=30.0),
    model_concurrency: int = typer.Option(2, min=1, max=32),
    model_retries: int = typer.Option(1, min=0, max=3),
    retention_now: str | None = typer.Option(
        None,
        envvar="PERSONAL_AGENT_MEMORY_RETENTION_NOW",
        help="Fixed UTC clock for retention acceptance tests.",
    ),
) -> None:
    """Run the single local memory daemon."""
    try:
        settings = Settings(
            state_dir=Path(state_dir),
            host=host,
            port=port,
            library_roots=tuple(Path(root) for root in library_root or ()),
            embedding=(
                ModelEndpoint(
                    embedding_url,
                    embedding_model,
                    None if embedding_api_key_file is None else Path(embedding_api_key_file),
                    model_timeout,
                    model_concurrency,
                    model_retries,
                )
                if embedding_url is not None
                else None
            ),
            reranker=(
                ModelEndpoint(
                    reranker_url,
                    reranker_model,
                    None if reranker_api_key_file is None else Path(reranker_api_key_file),
                    model_timeout,
                    model_concurrency,
                    model_retries,
                )
                if reranker_url is not None
                else None
            ),
            graph=(
                ModelEndpoint(
                    graph_url,
                    graph_model,
                    None if graph_api_key_file is None else Path(graph_api_key_file),
                    model_timeout,
                    model_concurrency,
                    model_retries,
                )
                if graph_url is not None
                else None
            ),
            retention_now=retention_now,
        )
        ensure_port_available(settings.host, settings.port)
    except (ConfigurationError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)


def main() -> None:
    cli()
