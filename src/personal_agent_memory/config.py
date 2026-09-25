from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from personal_agent_memory.model_client import ModelEndpoint


class ConfigurationError(ValueError):
    """Raised when startup configuration would violate the local security boundary."""


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class Settings:
    state_dir: Path
    host: str = "127.0.0.1"
    port: int = 7331
    library_roots: tuple[Path, ...] = ()
    embedding: ModelEndpoint | None = None
    reranker: ModelEndpoint | None = None
    graph: ModelEndpoint | None = None
    retention_now: str | None = None
    session_now: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_dir", self.state_dir.expanduser().resolve())
        roots = self.library_roots or (self.state_dir.parent / "memory-libraries",)
        object.__setattr__(
            self,
            "library_roots",
            tuple(root.expanduser().resolve() for root in roots),
        )
        for endpoint in (self.embedding, self.reranker, self.graph):
            if endpoint is None or endpoint.api_key_file is None:
                continue
            if any(
                endpoint.api_key_file == root or endpoint.api_key_file.is_relative_to(root)
                for root in self.library_roots
            ):
                raise ConfigurationError(
                    "model API key files must be outside memory library roots"
                )
        if not _is_loopback(self.host):
            raise ConfigurationError("host must be an explicit loopback address")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        if self.retention_now is not None:
            try:
                parsed = datetime.fromisoformat(self.retention_now.replace("Z", "+00:00"))
            except ValueError as error:
                raise ConfigurationError("retention clock must be an ISO-8601 timestamp") from error
            if parsed.tzinfo is None:
                raise ConfigurationError("retention clock must include a UTC offset")
        if self.session_now is not None:
            try:
                parsed = datetime.fromisoformat(self.session_now.replace("Z", "+00:00"))
            except ValueError as error:
                raise ConfigurationError("session clock must be an ISO-8601 timestamp") from error
            if parsed.tzinfo is None:
                raise ConfigurationError("session clock must include a UTC offset")
