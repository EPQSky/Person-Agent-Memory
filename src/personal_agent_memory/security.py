from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path


class ApiKeyStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def ensure(self) -> str:
        if self.path.exists():
            os.chmod(self.path, 0o600)
            value = self.path.read_text(encoding="utf-8").strip()
            if value:
                return value
        return self.rotate()

    def read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def rotate(self) -> str:
        value = secrets.token_urlsafe(32)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        temporary = self.path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(f"{value}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)
        return value

    @staticmethod
    def fingerprint(value: str) -> str:
        return f"sha256:{hashlib.sha256(value.encode()).hexdigest()[:12]}"
