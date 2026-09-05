from __future__ import annotations

import re
import secrets
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SensitiveFinding:
    disposition: str
    categories: tuple[str, ...]
    fingerprint: str


_CONFIRMED_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----", re.I)),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("api-token", re.compile(r"\bsk[-\s_]+[A-Za-z0-9_-]{20,}\b", re.I)),
    ("bearer-token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.I)),
)
_ASSIGNED_SECRET_KEY_PATTERN = re.compile(
    r"(?<![a-z0-9_-])[\"']?(?P<key>(?:[a-z0-9]+[_-])*(?:api[\s_-]*key|access[\s_-]*token|"
    r"client[\s_-]*secret|secret[\s_-]*access[\s_-]*key|password|passwd|pass|secret|token))\b"
    r"[\"']?\s*(?::|=)\s*",
    re.I,
)
_UNCERTAIN_VALUE_PATTERN = re.compile(
    r"\b(?:credential|api[\s_-]*key|access[\s_-]*token|password|passwd|"
    r"private[\s_-]*key|secret|token)\b"
    r"(?:\s+\w+){0,3}\s+(?:may|might|could|possibly)\b"
    r"(?:\s+\w+){0,3}",
    re.I,
)
_PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:example|sample|placeholder|redacted|masked|dummy|fake|not-a-secret|\*{4,})\b",
    re.I,
)
_ASSIGNED_PLACEHOLDER_PATTERN = re.compile(
    r"^(?:example|sample|placeholder|redacted|masked|dummy|fake|not-a-secret|\*{4,})$",
    re.I,
)
_BENIGN_ASSIGNED_VALUE_PATTERN = re.compile(
    r"^(?:"
    r"\$\{?[A-Z][A-Z0-9_]*\}?|"
    r"(?:env|environment)(?::|[._-])?[A-Z][A-Z0-9_]*|"
    r"vault://[a-z0-9._-]+(?:/[a-z0-9._-]+)+#[a-z0-9._-]+|"
    r"(?:generated|managed|configured|injected|provided|resolved|loaded|fetched)"
    r"(?:[-_](?:at|by|from|during|via)[-_][a-z0-9_-]+)+"
    r")$",
    re.I,
)


def _benign_assigned_value(value: str) -> bool:
    candidate = value.strip().rstrip(",.;)")
    normalized = re.sub(r"[\s_]+", "-", candidate)
    return bool(
        _ASSIGNED_PLACEHOLDER_PATTERN.fullmatch(candidate)
        or _BENIGN_ASSIGNED_VALUE_PATTERN.fullmatch(candidate)
        or _BENIGN_ASSIGNED_VALUE_PATTERN.fullmatch(normalized)
    )


def _assigned_secret_values(text: str) -> tuple[tuple[str, str], ...]:
    assignments: list[tuple[str, str]] = []
    for match in _ASSIGNED_SECRET_KEY_PATTERN.finditer(text):
        start = match.end()
        if start >= len(text):
            assignments.append((match.group("key"), ""))
            continue
        quote = text[start] if text[start] in {"\"", "'"} else None
        value_start = start + 1 if quote else start
        if quote:
            end = value_start
            escaped = False
            while end < len(text):
                character = text[end]
                if character in "\r\n" and not escaped:
                    break
                if character == quote and not escaped:
                    break
                escaped = character == "\\" and not escaped
                if character != "\\":
                    escaped = False
                end += 1
        else:
            boundaries = tuple(
                boundary
                for boundary in (text.find(";", value_start), text.find("\n", value_start))
                if boundary >= 0
            )
            end = min(boundaries, default=len(text))
        assignments.append((match.group("key"), text[value_start:end].strip()))
    return tuple(assignments)


def _material_assigned_secret(key: str, value: str) -> bool:
    candidate = value.strip().rstrip(",.;)")
    return bool(candidate) and not _benign_assigned_value(candidate)


def inspect_sensitive_text(text: str) -> SensitiveFinding | None:
    categories = tuple(
        name for name, pattern in _CONFIRMED_PATTERNS if pattern.search(text)
    )
    if any(_material_assigned_secret(key, value) for key, value in _assigned_secret_values(text)):
        categories = (*categories, "assigned-secret")
    compact = re.sub(r"[\s\\]+", "", text)
    split_categories = tuple(
        name
        for name, pattern in (
            ("api-token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b", re.I)),
            ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
            ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
        )
        if pattern.search(compact)
    )
    categories = tuple(dict.fromkeys((*categories, *split_categories)))
    fingerprint = "opaque:" + secrets.token_hex(8)
    if categories:
        return SensitiveFinding("discard", categories, fingerprint)
    uncertain = _UNCERTAIN_VALUE_PATTERN.search(text)
    if uncertain and not _PLACEHOLDER_PATTERN.search(uncertain.group(0)):
        return SensitiveFinding("quarantine", ("sensitive-language",), fingerprint)
    return None


def controlled_sensitive_summary(finding: SensitiveFinding, text: str) -> str:
    categories = ", ".join(finding.categories)
    return (
        f"Sensitive content withheld ({categories}); {len(text.encode())} bytes; "
        f"fingerprint {finding.fingerprint}."
    )
