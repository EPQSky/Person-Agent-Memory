from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_dir", self.state_dir.expanduser().resolve())
        if not _is_loopback(self.host):
            raise ConfigurationError("host must be an explicit loopback address")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
