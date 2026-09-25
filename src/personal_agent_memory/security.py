from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
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


class ManagementSessionStore:
    lifetime = timedelta(hours=24)
    max_records = 64
    cleanup_limit = 64

    def __init__(self, path: Path, now: Callable[[], datetime]) -> None:
        self.path = path
        self.now = now
        self._lock = threading.RLock()

    def create(self, api_key: str) -> tuple[str, datetime]:
        with self._lock:
            now = self._utc_now()
            expires_at = now + self.lifetime
            records = self._load()
            records = self._clean(records, now, self._digest(api_key))
            token = secrets.token_urlsafe(32)
            records.append(
                {
                    "token_hash": self._digest(token),
                    "key_digest": self._digest(api_key),
                    "created_at": now.isoformat(),
                    "expires_at": expires_at.isoformat(),
                }
            )
            self._write(records[-self.max_records :])
            return token, expires_at

    def verify(self, token: str, api_key: str) -> bool:
        if not token or not api_key:
            return False
        with self._lock:
            now = self._utc_now()
            key_digest = self._digest(api_key)
            records = self._load()
            retained = self._clean(records, now, key_digest)
            token_hash = self._digest(token)
            valid = False
            for record in retained:
                if not secrets.compare_digest(str(record.get("token_hash", "")), token_hash):
                    continue
                valid = (
                    secrets.compare_digest(str(record.get("key_digest", "")), key_digest)
                    and self._parse(record.get("expires_at")) > now
                )
                break
            if retained != records or not valid:
                self._write(retained)
            return valid

    def revoke(self, token: str) -> None:
        if not token:
            return
        with self._lock:
            records = self._load()
            token_hash = self._digest(token)
            retained = [
                record
                for record in records
                if not secrets.compare_digest(str(record.get("token_hash", "")), token_hash)
            ]
            if retained != records:
                self._write(retained)

    def invalidate_key(self, api_key: str) -> None:
        with self._lock:
            key_digest = self._digest(api_key)
            records = self._load()
            retained = [
                record
                for record in records
                if not secrets.compare_digest(str(record.get("key_digest", "")), key_digest)
            ]
            if retained != records:
                self._write(retained)

    def _load(self) -> list[dict[str, object]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        if not isinstance(raw, list):
            return []
        return [record for record in raw if isinstance(record, dict)]

    def _write(self, records: list[dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(records, stream, ensure_ascii=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

    def _clean(
        self,
        records: list[dict[str, object]],
        now: datetime,
        current_key_digest: str,
    ) -> list[dict[str, object]]:
        retained: list[dict[str, object]] = []
        for record in records[-self.cleanup_limit :]:
            expires_at = self._parse(record.get("expires_at"))
            key_digest = str(record.get("key_digest", ""))
            if expires_at > now and secrets.compare_digest(key_digest, current_key_digest):
                retained.append(record)
        return retained[-self.max_records :]

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse(value: object) -> datetime:
        if not isinstance(value, str):
            return datetime.min.replace(tzinfo=UTC)
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _utc_now(self) -> datetime:
        current = self.now()
        if current.tzinfo is None:
            return current.replace(tzinfo=UTC)
        return current.astimezone(UTC)
