"""Deterministic, evidence-bound entity graph projection for formal memories."""

# SQL statements remain readable as schema/query blocks.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from personal_agent_memory.sensitive import inspect_sensitive_text

ENTITY_TYPES = frozenset(
    {
        "project",
        "technology",
        "component",
        "concept",
        "person",
        "organization",
        "constraint",
        "reference",
    }
)
RELATION_TYPES = frozenset(
    {
        "part_of",
        "uses",
        "depends_on",
        "implements",
        "has_constraint",
        "prefers",
        "references",
        "supersedes",
        "related_to",
    }
)
EntityGraphExtractor = Callable[
    [dict[str, object]], Mapping[str, object] | list[dict[str, object]]
]
DETAIL_DEFAULT_LIMIT = 50
DETAIL_MAX_LIMIT = 50
DETAIL_MAX_OFFSET = 1_000_000
GRAPH_MAX_NODE_LIMIT = 200
GRAPH_MAX_EDGE_LIMIT = 400
GRAPH_SUPPORTING_MEMORY_LIMIT = 50
GRAPH_SUPPORTING_EVIDENCE_LIMIT = 50
GRAPH_VISIBLE_LIFECYCLES = frozenset({"active", "superseded", "archived"})
_EVIDENCE_TEXT_KEYS = frozenset(
    {
        "content",
        "text",
        "snippet",
        "evidence",
        "evidence_text",
        "evidence_snippet",
        "evidence_snippets",
        "excerpt",
        "quote",
    }
)
ENTITY_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_graph_entities (
    entity_id TEXT PRIMARY KEY,
    region_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
    evidence TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (region_id, entity_type, normalized_name)
);
CREATE INDEX IF NOT EXISTS entity_graph_entities_region ON entity_graph_entities (region_id, entity_type, normalized_name);
CREATE TABLE IF NOT EXISTS entity_graph_aliases (
    region_id TEXT NOT NULL, entity_type TEXT NOT NULL, normalized_alias TEXT NOT NULL, entity_id TEXT NOT NULL,
    PRIMARY KEY (region_id, entity_type, normalized_alias)
);
CREATE TABLE IF NOT EXISTS entity_graph_relations (
    relation_id TEXT PRIMARY KEY, region_id TEXT NOT NULL, source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL, relation_type TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE (region_id, source_entity_id, target_entity_id, relation_type)
);
CREATE INDEX IF NOT EXISTS entity_graph_relations_region ON entity_graph_relations (region_id, relation_type);
CREATE TABLE IF NOT EXISTS entity_graph_memory_entities (
    region_id TEXT NOT NULL, memory_id TEXT NOT NULL, entity_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL DEFAULT '', aliases TEXT NOT NULL DEFAULT '[]',
    evidence TEXT NOT NULL DEFAULT '[]', lifecycle TEXT NOT NULL DEFAULT 'active',
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (region_id, memory_id, entity_id)
);
CREATE TABLE IF NOT EXISTS entity_graph_memory_relations (
    region_id TEXT NOT NULL, memory_id TEXT NOT NULL, relation_id TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '[]', lifecycle TEXT NOT NULL DEFAULT 'active',
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (region_id, memory_id, relation_id)
);
CREATE INDEX IF NOT EXISTS entity_graph_memory_relations_relation
    ON entity_graph_memory_relations (region_id, relation_id, memory_id);
CREATE TABLE IF NOT EXISTS entity_graph_status (
    region_id TEXT PRIMARY KEY, status TEXT NOT NULL, projected_memories INTEGER NOT NULL DEFAULT 0,
    rejected_items INTEGER NOT NULL DEFAULT 0, last_error TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL
)
"""


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return " ".join(text.split())


def _escape_like_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(value: object, default: object) -> object:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default
    return parsed


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _safe_text(value: object) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()
    return ""


_OMIT_SENSITIVE = object()


def _sanitize_nested(value: object) -> tuple[object, bool]:
    """Remove sensitive leaf strings while retaining safe nested evidence."""
    if isinstance(value, str):
        if inspect_sensitive_text(value) is not None:
            return _OMIT_SENSITIVE, True
        return value, False
    if isinstance(value, Mapping):
        sanitized: dict[object, object] = {}
        found_sensitive = False
        for raw_key, raw_value in value.items():
            clean_key, key_sensitive = _sanitize_nested(raw_key)
            clean_value, value_sensitive = _sanitize_nested(raw_value)
            found_sensitive = found_sensitive or key_sensitive or value_sensitive
            if clean_key is _OMIT_SENSITIVE or clean_value is _OMIT_SENSITIVE:
                continue
            sanitized[clean_key] = clean_value
        return sanitized, found_sensitive
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        sanitized_items: list[object] = []
        found_sensitive = False
        for item in value:
            clean_item, item_sensitive = _sanitize_nested(item)
            found_sensitive = found_sensitive or item_sensitive
            if clean_item is not _OMIT_SENSITIVE:
                sanitized_items.append(clean_item)
        return sanitized_items, found_sensitive
    return value, False


def _explicit_texts(value: object, *, allow_direct: bool = False) -> list[str]:
    """Collect only values from fields that explicitly carry evidence text.

    Evidence records also contain identifiers and provenance metadata. Those
    values are useful for tracing, but they are not text evidence and must not
    satisfy entity or relation support checks.
    """
    if isinstance(value, str):
        return [value] if allow_direct and value.strip() else []
    if isinstance(value, Mapping):
        values: list[str] = []
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip().casefold()
            values.extend(
                _explicit_texts(
                    raw_value,
                    allow_direct=key in _EVIDENCE_TEXT_KEYS,
                )
            )
        return values
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        sequence_values: list[str] = []
        for item in value:
            sequence_values.extend(_explicit_texts(item, allow_direct=allow_direct))
        return sequence_values
    return []


def _safe_explicit_texts(value: object, *, allow_direct: bool = False) -> list[str]:
    return list(
        dict.fromkeys(
            text
            for text in _explicit_texts(value, allow_direct=allow_direct)
            if inspect_sensitive_text(text) is None
        )
    )


def _has_nonempty_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_nonempty_value(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(_has_nonempty_value(item) for item in value)
    return True


def _validated_source_references(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    references: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        reference = item.strip()
        if not reference or inspect_sensitive_text(reference) is not None:
            return None
        references.append(reference)
    return list(dict.fromkeys(references)) or None


def _evidence_size(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, Mapping):
        explicit_values = [
            raw_value
            for raw_key, raw_value in value.items()
            if str(raw_key).strip().casefold() in _EVIDENCE_TEXT_KEYS
        ]
        if explicit_values:
            return sum(_evidence_size(item) for item in explicit_values)
        return 1 if value else 0
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return len(value)
    return 1


def _validate_extractor_evidence(result: Mapping[str, object]) -> None:
    for collection_name in ("entities", "relations"):
        collection = result.get(collection_name, [])
        if not isinstance(collection, Sequence) or isinstance(
            collection, (str, bytes, bytearray)
        ):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, Mapping):
                continue
            evidence_size = sum(
                _evidence_size(value)
                for raw_key, value in item.items()
                if str(raw_key).strip().casefold() in _EVIDENCE_TEXT_KEYS
            )
            if evidence_size > GRAPH_SUPPORTING_EVIDENCE_LIMIT:
                raise ValueError(
                    f"entity graph {collection_name}[{index}] evidence exceeds "
                    f"{GRAPH_SUPPORTING_EVIDENCE_LIMIT} items"
                )


def stable_entity_id(region_id: str, entity_type: str, normalized_name: str) -> str:
    digest = hashlib.sha256(
        _json([region_id, entity_type, normalized_name]).encode("utf-8")
    ).hexdigest()
    return f"entity:{digest}"


def stable_relation_id(
    region_id: str, source_entity_id: str, target_entity_id: str, relation_type: str
) -> str:
    digest = hashlib.sha256(
        _json([region_id, source_entity_id, target_entity_id, relation_type]).encode("utf-8")
    ).hexdigest()
    return f"relation:{digest}"


def _evidence_texts(memory: Mapping[str, object]) -> list[str]:
    values = _safe_explicit_texts(memory.get("content"), allow_direct=True)
    values.extend(_safe_explicit_texts(memory.get("evidence"), allow_direct=True))
    values.extend(
        _safe_explicit_texts(memory.get("evidence_snippets"), allow_direct=True)
    )
    return list(dict.fromkeys(value for value in values if value))


def _region_texts(memory: Mapping[str, object]) -> list[str]:
    raw_region = memory.get("region")
    if not isinstance(raw_region, Mapping):
        return []
    values: list[str] = []
    for key in ("name", "label", "description", "text"):
        values.extend(_safe_explicit_texts(raw_region.get(key), allow_direct=True))
    return list(dict.fromkeys(values))


def _has_exact_support(value: str, texts: Sequence[str]) -> bool:
    return bool(value.strip() and any(value.strip() in text for text in texts))


def _entity_evidence(
    raw: Mapping[str, object],
    memory: Mapping[str, object],
    texts: Sequence[str],
    entity_texts: Sequence[str],
    names: Sequence[str],
) -> list[dict[str, object]] | None:
    result: list[dict[str, object]] = []
    snippets: list[str] = []
    has_raw_evidence = False
    for key in ("evidence", "evidence_snippet", "evidence_snippets"):
        if key not in raw:
            continue
        raw_evidence = raw.get(key)
        has_raw_evidence = has_raw_evidence or _has_nonempty_value(raw_evidence)
        snippets.extend(_safe_explicit_texts(raw_evidence, allow_direct=True))
    snippets = list(dict.fromkeys(snippets))
    if len(snippets) > GRAPH_SUPPORTING_EVIDENCE_LIMIT:
        return None
    if has_raw_evidence and not snippets:
        return None
    if snippets:
        for snippet in snippets:
            if not snippet or not _has_exact_support(snippet, texts):
                return None
            if not any(
                _has_exact_support(name, [snippet]) for name in names if name
            ):
                return None
            result.append(
                {"memory_id": str(memory.get("memory_id", "")), "evidence": snippet}
            )
    if not result:
        fallback = next(
            (
                text
                for text in entity_texts
                if text
                and any(_has_exact_support(name, [text]) for name in names if name)
            ),
            "",
        )
        if not fallback:
            return None
        result.append(
            {
                "memory_id": str(memory.get("memory_id", "")),
                "evidence": fallback,
            }
        )
    return result


def _relation_evidence(
    raw: Mapping[str, object], references: Sequence[str], memory_id: str
) -> list[dict[str, object]]:
    snippets = _safe_explicit_texts(raw.get("evidence"), allow_direct=True)
    if not snippets:
        snippets = _safe_explicit_texts(raw.get("evidence_snippet"), allow_direct=True)
    if not snippets:
        snippets = _safe_explicit_texts(
            raw.get("evidence_snippets"), allow_direct=True
        )
    if len(snippets) > GRAPH_SUPPORTING_EVIDENCE_LIMIT:
        return []
    return [
        {
            "memory_id": memory_id,
            "evidence": snippet,
            "source_references": references,
        }
        for snippet in snippets
    ]


def _cursor_offset(cursor: str) -> int:
    if not cursor or not cursor.isascii() or not cursor.isdigit():
        raise ValueError("detail cursor must be a non-negative decimal offset")
    offset = int(cursor)
    if offset > DETAIL_MAX_OFFSET:
        raise ValueError("detail cursor exceeds the maximum offset")
    return offset


def _detail_window(
    limit: int, offset: int | None = None, cursor: str | None = None
) -> tuple[int, int]:
    if limit < 1:
        raise ValueError("detail limit must be positive")
    if offset is not None and cursor is not None:
        raise ValueError("detail cursor and offset are mutually exclusive")
    resolved_offset = _cursor_offset(cursor) if cursor is not None else (offset or 0)
    if resolved_offset < 0:
        raise ValueError("detail offset must not be negative")
    if resolved_offset > DETAIL_MAX_OFFSET:
        raise ValueError("detail offset exceeds the maximum offset")
    return min(limit, DETAIL_MAX_LIMIT), resolved_offset


def _detail_page(
    total: int, memory_ids: Sequence[str], limit: int, offset: int
) -> dict[str, object]:
    has_more = total > offset + len(memory_ids)
    return {
        "supporting_memory_count": total,
        "supporting_memory_limit": limit,
        "supporting_memory_offset": offset,
        "has_more": has_more,
        "next_cursor": str(offset + len(memory_ids)) if has_more else None,
    }


class EntityGraphStore:
    """SQLite projection store. Formal memory remains the source of truth."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.executescript(ENTITY_GRAPH_SCHEMA)
        for table in ("entity_graph_memory_entities", "entity_graph_memory_relations"):
            columns = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "evidence" not in columns:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN evidence TEXT NOT NULL DEFAULT '[]'"
                )
            if "lifecycle" not in columns:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'active'"
                )
            if "updated_at" not in columns:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"
                )
            if table == "entity_graph_memory_entities":
                if "canonical_name" not in columns:
                    connection.execute(
                        "ALTER TABLE entity_graph_memory_entities "
                        "ADD COLUMN canonical_name TEXT NOT NULL DEFAULT ''"
                    )
                if "aliases" not in columns:
                    connection.execute(
                        "ALTER TABLE entity_graph_memory_entities "
                        "ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'"
                    )
                connection.execute(
                    """UPDATE entity_graph_memory_entities
                       SET canonical_name = COALESCE(
                           (SELECT canonical_name FROM entity_graph_entities
                            WHERE entity_graph_entities.entity_id = entity_graph_memory_entities.entity_id),
                           '')
                       WHERE canonical_name = ''"""
                )
        connection.commit()
        return connection

    def project(
        self,
        region_id: str,
        memory: Mapping[str, object],
        extractor: EntityGraphExtractor,
    ) -> dict[str, object]:
        memory_id = _safe_text(memory.get("memory_id"))
        if not memory_id:
            return {"status": "rejected", "rejected": 1, "reason": "memory_id is missing"}
        if inspect_sensitive_text(memory_id) is not None:
            self.remove_memory(region_id, memory_id)
            return {
                "status": "skipped",
                "rejected": 1,
                "reason": "sensitive memory identifier",
            }
        lifecycle = _safe_text(memory.get("lifecycle")) or "active"
        if lifecycle == "forgotten":
            self.remove_memory(region_id, memory_id)
            return {"status": "skipped", "rejected": 0, "reason": "forgotten memory"}
        if lifecycle not in GRAPH_VISIBLE_LIFECYCLES:
            self.remove_memory(region_id, memory_id)
            return {
                "status": "skipped",
                "rejected": 1,
                "reason": "unsupported memory lifecycle",
            }
        updated_at = _safe_text(memory.get("updated_at"))
        raw_content = _safe_text(memory.get("content"))
        raw_region = memory.get("region")
        direct_region_values = (
            [_safe_text(raw_region.get(key)) for key in ("name", "region_id", "kind")]
            if isinstance(raw_region, Mapping)
            else []
        )
        raw_references = memory.get("source_references")
        references = _validated_source_references(raw_references)
        if (
            inspect_sensitive_text(raw_content) is not None
            or any(inspect_sensitive_text(value) is not None for value in direct_region_values)
            or references is None
        ):
            self.remove_memory(region_id, memory_id)
            return {
                "status": "skipped",
                "rejected": 1,
                "reason": "sensitive or missing graph input",
            }
        sanitized_memory, _ = _sanitize_nested(memory)
        if not isinstance(sanitized_memory, Mapping):
            self.remove_memory(region_id, memory_id)
            return {
                "status": "skipped",
                "rejected": 1,
                "reason": "invalid graph input",
            }
        safe_memory = dict(sanitized_memory)
        texts = _evidence_texts(safe_memory)
        if not texts or references is None:
            self.remove_memory(region_id, memory_id)
            return {
                "status": "skipped",
                "rejected": 1,
                "reason": "sensitive or missing graph input",
            }
        raw_region = safe_memory.get("region")
        region_texts = _region_texts(safe_memory)
        entity_texts = [*texts, *(value for value in region_texts if value)]
        region = {
            "region_id": _safe_text(raw_region.get("region_id")) if isinstance(raw_region, Mapping) else region_id,
            "kind": _safe_text(raw_region.get("kind")) if isinstance(raw_region, Mapping) else "",
            "name": _safe_text(raw_region.get("name")) if isinstance(raw_region, Mapping) else "",
        }
        context: dict[str, object] = {
            "schema": "entity-graph-v1",
            "region": region,
            "memory": {
                "memory_id": memory_id,
                "content": _safe_text(safe_memory.get("content")),
                "memory_type": _safe_text(safe_memory.get("memory_type")),
                "evidence": safe_memory.get("evidence") or [],
                "evidence_snippets": safe_memory.get("evidence_snippets") or [],
                "source_references": references,
            },
        }
        try:
            raw = extractor(context)
        except Exception as error:
            del error
            raise ValueError("entity graph extractor failed") from None
        if isinstance(raw, Mapping):
            result: Mapping[str, object] = raw
        elif isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], Mapping):
            result = cast(Mapping[str, object], raw[0])
        else:
            raise ValueError("entity graph extractor must return one object")
        _validate_extractor_evidence(result)
        sanitized_result, _ = _sanitize_nested(result)
        if not isinstance(sanitized_result, Mapping):
            raise ValueError("entity graph extractor must return one object")
        result = sanitized_result
        entities_raw = result.get("entities", [])
        relations_raw = result.get("relations", [])
        if not isinstance(entities_raw, Sequence) or isinstance(
            entities_raw, (str, bytes, bytearray)
        ):
            raise ValueError("entity graph entities must be a list")
        if not isinstance(relations_raw, Sequence) or isinstance(
            relations_raw, (str, bytes, bytearray)
        ):
            raise ValueError("entity graph relations must be a list")
        now = datetime.now(UTC).isoformat()
        updated_at = updated_at or now
        rejected = 0
        with self._connection() as connection:
            self._remove_memory_locked(connection, region_id, memory_id)
            self._cleanup(connection, region_id)
            self._rebuild_aggregates(connection, region_id)
            entity_refs: dict[str, str] = {}
            accepted_count = 0
            for raw_entity in entities_raw:
                if not isinstance(raw_entity, Mapping):
                    rejected += 1
                    continue
                entity_type = _safe_text(raw_entity.get("type") or raw_entity.get("entity_type"))
                canonical = _safe_text(raw_entity.get("canonical_name") or raw_entity.get("name"))
                aliases = _string_list(raw_entity.get("aliases"))
                raw_entity_id = _safe_text(raw_entity.get("id"))
                normalized = normalize_name(canonical)
                names = [canonical, *aliases]
                supported = any(
                    _has_exact_support(name, entity_texts) for name in names if name
                )
                if (
                    entity_type not in ENTITY_TYPES
                    or not canonical
                    or not normalized
                    or not supported
                    or inspect_sensitive_text(canonical) is not None
                    or any(inspect_sensitive_text(alias) is not None for alias in aliases)
                    or inspect_sensitive_text(raw_entity_id) is not None
                ):
                    rejected += 1
                    continue
                entity_id = self._resolve_entity(
                    connection, region_id, entity_type, normalized, aliases
                )
                if entity_id is None:
                    rejected += 1
                    continue
                evidence = _entity_evidence(
                    raw_entity, memory, texts, entity_texts, names
                )
                if evidence is None:
                    rejected += 1
                    continue
                if any(
                    inspect_sensitive_text(str(item.get("evidence", ""))) is not None
                    for item in evidence
                ):
                    rejected += 1
                    continue
                accepted_count += 1
                for reference in [canonical, *aliases, _safe_text(raw_entity.get("id"))]:
                    normalized_reference = normalize_name(reference)
                    if normalized_reference:
                        entity_refs[f"{entity_type}:{normalized_reference}"] = entity_id
                connection.execute(
                    """INSERT OR REPLACE INTO entity_graph_memory_entities
                       (region_id, memory_id, entity_id, canonical_name, aliases, evidence,
                        lifecycle, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        region_id,
                        memory_id,
                        entity_id,
                        canonical,
                        _json(aliases),
                        _json(evidence),
                        lifecycle,
                        updated_at,
                    ),
                )
                self._merge_entity(
                    connection,
                    entity_id,
                    region_id,
                    entity_type,
                    canonical,
                    normalized,
                    aliases,
                    evidence,
                    memory_id,
                    now,
                )
            for raw_relation in relations_raw:
                if not isinstance(raw_relation, Mapping):
                    rejected += 1
                    continue
                relation_type = _safe_text(
                    raw_relation.get("type") or raw_relation.get("relation_type")
                )
                source_type, source_name = self._relation_ref(
                    raw_relation.get("source"), raw_relation.get("source_type")
                )
                target_type, target_name = self._relation_ref(
                    raw_relation.get("target"), raw_relation.get("target_type")
                )
                source_id = entity_refs.get(f"{source_type}:{normalize_name(source_name)}")
                target_id = entity_refs.get(f"{target_type}:{normalize_name(target_name)}")
                evidence = _relation_evidence(raw_relation, references, memory_id)
                if (
                    relation_type not in RELATION_TYPES
                    or not source_id
                    or not target_id
                    or source_id == target_id
                    or not evidence
                    or any(
                        not _has_exact_support(str(item["evidence"]), texts) for item in evidence
                    )
                    or any(
                        inspect_sensitive_text(str(item["evidence"])) is not None
                        for item in evidence
                    )
                ):
                    rejected += 1
                    continue
                relation_id = stable_relation_id(region_id, source_id, target_id, relation_type)
                connection.execute(
                    """INSERT INTO entity_graph_relations
                   (relation_id, region_id, source_entity_id, target_entity_id, relation_type, evidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(relation_id) DO UPDATE SET evidence = ?, updated_at = ?""",
                    (
                        relation_id,
                        region_id,
                        source_id,
                        target_id,
                        relation_type,
                        _json([]),
                        now,
                        now,
                        _json([]),
                        now,
                    ),
                )
                connection.execute(
                    """INSERT OR REPLACE INTO entity_graph_memory_relations
                       (region_id, memory_id, relation_id, evidence, lifecycle, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (region_id, memory_id, relation_id, _json(evidence), lifecycle, updated_at),
                )
            self._cleanup(connection, region_id)
            self._rebuild_aggregates(connection, region_id)
            connection.execute(
                """INSERT INTO entity_graph_status (region_id, status, projected_memories, rejected_items, last_error, updated_at)
                VALUES (?, 'ready', (SELECT COUNT(DISTINCT memory_id) FROM entity_graph_memory_entities WHERE region_id = ?), ?, '', ?)
                ON CONFLICT(region_id) DO UPDATE SET status = excluded.status, projected_memories = excluded.projected_memories,
                rejected_items = excluded.rejected_items, last_error = '', updated_at = excluded.updated_at""",
                (region_id, region_id, rejected, now),
            )
        return {"status": "ready", "rejected": rejected, "entity_count": accepted_count}

    def remove_memory(self, region_id: str, memory_id: str) -> None:
        with self._connection() as connection:
            self._remove_memory_locked(connection, region_id, memory_id)
            self._cleanup(connection, region_id)
            self._rebuild_aggregates(connection, region_id)

    def rebuild(
        self,
        region_id: str,
        memories: Sequence[Mapping[str, object]],
        extractor: EntityGraphExtractor,
    ) -> dict[str, object]:
        active_memory_ids = {
            memory_id
            for memory in memories
            if (memory_id := _safe_text(memory.get("memory_id")))
        }
        memory_lifecycles = {
            memory_id: (
                _safe_text(memory.get("lifecycle")) or "active",
                _safe_text(memory.get("updated_at")),
            )
            for memory in memories
            if (memory_id := _safe_text(memory.get("memory_id")))
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix="entity-graph-rebuild-", suffix=".sqlite3", dir=self._path.parent
        )
        os.close(file_descriptor)
        projected = 0
        rejected = 0
        try:
            temporary_store = EntityGraphStore(Path(temporary_name))
            for memory in memories:
                result = temporary_store.project(region_id, memory, extractor)
                if result.get("status") == "ready":
                    projected += 1
                rejected_value = result.get("rejected", 0)
                if isinstance(rejected_value, (int, float)):
                    rejected += int(rejected_value)
            tables = (
                "entity_graph_entities",
                "entity_graph_aliases",
                "entity_graph_relations",
                "entity_graph_memory_entities",
                "entity_graph_memory_relations",
            )
            with temporary_store._connection() as source, self._connection() as target:
                target.execute("BEGIN IMMEDIATE")
                for table in (
                    "entity_graph_memory_relations",
                    "entity_graph_memory_entities",
                    "entity_graph_relations",
                    "entity_graph_aliases",
                    "entity_graph_entities",
                ):
                    target.execute(f"DELETE FROM {table} WHERE region_id = ?", (region_id,))
                for table in tables:
                    columns = [
                        str(row[1])
                        for row in source.execute(f"PRAGMA table_info({table})").fetchall()
                    ]
                    rows = source.execute(
                        f"SELECT {', '.join(columns)} FROM {table} WHERE region_id = ?",
                        (region_id,),
                    ).fetchall()
                    if rows:
                        placeholders = ", ".join("?" for _ in columns)
                        target.executemany(
                            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                            [tuple(row) for row in rows],
                        )
                now = datetime.now(UTC).isoformat()
                target.execute(
                    """INSERT INTO entity_graph_status
                       (region_id, status, projected_memories, rejected_items, last_error, updated_at)
                       VALUES (?, 'ready', ?, ?, '', ?)
                       ON CONFLICT(region_id) DO UPDATE SET status = 'ready',
                       projected_memories = excluded.projected_memories,
                       rejected_items = excluded.rejected_items, last_error = '', updated_at = excluded.updated_at""",
                    (region_id, projected, rejected, now),
                )
            return self.status(region_id)
        except Exception:
            # A failed rebuild may follow a lifecycle change. Keep the last
            # usable projection for memories that are still active, but never
            # expose supports for memories absent from the current active set.
            self._synchronize_memory_lifecycles(region_id, memory_lifecycles)
            self._prune_inactive_memories(region_id, active_memory_ids)
            self.fail(region_id, "entity graph extractor returned invalid data")
            return {
                **self.status(region_id),
                "error": "entity graph projection failed",
            }
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)

    def _synchronize_memory_lifecycles(
        self,
        region_id: str,
        memory_lifecycles: Mapping[str, tuple[str, str]],
    ) -> None:
        if not memory_lifecycles:
            return
        with self._connection() as connection:
            for memory_id, (lifecycle, updated_at) in memory_lifecycles.items():
                if lifecycle not in GRAPH_VISIBLE_LIFECYCLES:
                    self._remove_memory_locked(connection, region_id, memory_id)
                    continue
                connection.execute(
                    """UPDATE entity_graph_memory_entities
                       SET lifecycle = ?, updated_at = COALESCE(NULLIF(?, ''), updated_at)
                       WHERE region_id = ? AND memory_id = ?""",
                    (lifecycle, updated_at, region_id, memory_id),
                )
                connection.execute(
                    """UPDATE entity_graph_memory_relations
                       SET lifecycle = ?, updated_at = COALESCE(NULLIF(?, ''), updated_at)
                       WHERE region_id = ? AND memory_id = ?""",
                    (lifecycle, updated_at, region_id, memory_id),
                )
            self._cleanup(connection, region_id)
            self._rebuild_aggregates(connection, region_id)

    @staticmethod
    def _remove_memory_locked(
        connection: sqlite3.Connection, region_id: str, memory_id: str
    ) -> None:
        connection.execute(
            "DELETE FROM entity_graph_memory_relations WHERE region_id = ? AND memory_id = ?",
            (region_id, memory_id),
        )
        connection.execute(
            "DELETE FROM entity_graph_memory_entities WHERE region_id = ? AND memory_id = ?",
            (region_id, memory_id),
        )

    def _prune_inactive_memories(
        self, region_id: str, active_memory_ids: set[str]
    ) -> None:
        with self._connection() as connection:
            projected_memory_rows = connection.execute(
                """SELECT DISTINCT memory_id FROM entity_graph_memory_entities
                   WHERE region_id = ?
                   UNION
                   SELECT DISTINCT memory_id FROM entity_graph_memory_relations
                   WHERE region_id = ?""",
                (region_id, region_id),
            ).fetchall()
            stale_memory_ids = [
                str(row[0])
                for row in projected_memory_rows
                if str(row[0]) not in active_memory_ids
            ]
            for memory_id in stale_memory_ids:
                self._remove_memory_locked(connection, region_id, memory_id)
            if stale_memory_ids:
                self._cleanup(connection, region_id)
                self._rebuild_aggregates(connection, region_id)

    @staticmethod
    def _rebuild_aggregates(connection: sqlite3.Connection, region_id: str) -> None:
        entity_rows = connection.execute(
            "SELECT entity_id FROM entity_graph_entities WHERE region_id = ?", (region_id,)
        ).fetchall()
        aggregate_aliases: dict[str, list[str]] = {}
        for row in entity_rows:
            entity_id = str(row[0])
            supports = connection.execute(
                """SELECT memory_id, canonical_name, aliases, evidence
                   FROM entity_graph_memory_entities
                   WHERE region_id = ? AND entity_id = ? ORDER BY memory_id LIMIT ?""",
                (region_id, entity_id, GRAPH_SUPPORTING_MEMORY_LIMIT),
            ).fetchall()
            memory_ids: list[str] = []
            evidence: list[object] = []
            aliases: list[str] = []
            for support in supports:
                memory_ids.append(str(support[0]))
                aliases.extend(_string_list(_loads(support[2], [])))
                parsed = _loads(support[3], [])
                if isinstance(parsed, list):
                    evidence.extend(parsed)
            deduped = list({_json(item): item for item in evidence}.values())[
                :GRAPH_SUPPORTING_EVIDENCE_LIMIT
            ]
            deduped_aliases = list(dict.fromkeys(alias for alias in aliases if alias))
            aggregate_aliases[entity_id] = deduped_aliases
            connection.execute(
                """UPDATE entity_graph_entities
                   SET aliases = ?, evidence_memory_ids = ?, evidence = ?
                   WHERE region_id = ? AND entity_id = ?""",
                (
                    _json(deduped_aliases),
                    _json(list(dict.fromkeys(memory_ids))[:GRAPH_SUPPORTING_MEMORY_LIMIT]),
                    _json(deduped),
                    region_id,
                    entity_id,
                ),
            )
        connection.execute("DELETE FROM entity_graph_aliases WHERE region_id = ?", (region_id,))
        for row in entity_rows:
            entity_id = str(row[0])
            entity = connection.execute(
                """SELECT entity_type, canonical_name FROM entity_graph_entities
                   WHERE region_id = ? AND entity_id = ?""",
                (region_id, entity_id),
            ).fetchone()
            if entity is None:
                continue
            entity_type = str(entity[0])
            references = [str(entity[1]), *aggregate_aliases.get(entity_id, [])]
            for reference in references:
                normalized = normalize_name(reference)
                if normalized:
                    connection.execute(
                        """INSERT OR IGNORE INTO entity_graph_aliases
                           (region_id, entity_type, normalized_alias, entity_id)
                           VALUES (?, ?, ?, ?)""",
                        (region_id, entity_type, normalized, entity_id),
                    )
        relation_rows = connection.execute(
            "SELECT relation_id FROM entity_graph_relations WHERE region_id = ?", (region_id,)
        ).fetchall()
        for row in relation_rows:
            relation_id = str(row[0])
            supports = connection.execute(
                """SELECT evidence FROM entity_graph_memory_relations
                   WHERE region_id = ? AND relation_id = ? ORDER BY memory_id LIMIT ?""",
                (region_id, relation_id, GRAPH_SUPPORTING_MEMORY_LIMIT),
            ).fetchall()
            evidence = []
            for support in supports:
                parsed = _loads(support[0], [])
                if isinstance(parsed, list):
                    evidence.extend(parsed)
            deduped = list({_json(item): item for item in evidence}.values())[
                :GRAPH_SUPPORTING_EVIDENCE_LIMIT
            ]
            connection.execute(
                """UPDATE entity_graph_relations SET evidence = ?
                   WHERE region_id = ? AND relation_id = ?""",
                (_json(deduped), region_id, relation_id),
            )

    @staticmethod
    def _relation_ref(value: object, type_hint: object) -> tuple[str, str]:
        if isinstance(value, Mapping):
            return _safe_text(value.get("type") or type_hint), _safe_text(
                value.get("canonical_name") or value.get("name") or value.get("id")
            )
        text = _safe_text(value)
        if ":" in text:
            entity_type, name = text.split(":", 1)
            return entity_type.strip(), name.strip()
        return _safe_text(type_hint), text

    @staticmethod
    def _resolve_entity(
        connection: sqlite3.Connection,
        region_id: str,
        entity_type: str,
        normalized: str,
        aliases: Sequence[str],
    ) -> str | None:
        row = connection.execute(
            "SELECT entity_id FROM entity_graph_entities WHERE region_id = ? AND entity_type = ? AND normalized_name = ?",
            (region_id, entity_type, normalized),
        ).fetchone()
        if row is not None:
            return str(row[0])
        candidates: set[str] = set()
        for alias_norm in dict.fromkeys([normalized, *(normalize_name(alias) for alias in aliases)]):
            if not alias_norm:
                continue
            rows = connection.execute(
                "SELECT entity_id FROM entity_graph_aliases WHERE region_id = ? AND entity_type = ? AND normalized_alias = ?",
                (region_id, entity_type, alias_norm),
            ).fetchall()
            candidates.update(str(item[0]) for item in rows)
        return (
            None
            if len(candidates) > 1
            else next(iter(candidates), stable_entity_id(region_id, entity_type, normalized))
        )

    @staticmethod
    def _merge_entity(
        connection: sqlite3.Connection,
        entity_id: str,
        region_id: str,
        entity_type: str,
        canonical: str,
        normalized: str,
        aliases: Sequence[str],
        evidence: list[dict[str, object]],
        memory_id: str,
        now: str,
    ) -> None:
        row = connection.execute(
            """SELECT entity_type, canonical_name, normalized_name, aliases,
                      evidence_memory_ids, evidence
               FROM entity_graph_entities WHERE entity_id = ?""",
            (entity_id,),
        ).fetchone()
        if row is not None:
            entity_type = str(row[0])
            canonical = str(row[1])
            normalized = str(row[2])
        old_aliases = _string_list(_loads(row[3], []) if row else [])
        old_memory_ids = _string_list(_loads(row[4], []) if row else [])[
            :GRAPH_SUPPORTING_MEMORY_LIMIT
        ]
        old_evidence = _loads(row[5], []) if row else []
        evidence_items = list(old_evidence) if isinstance(old_evidence, list) else []
        evidence_items.extend(evidence)
        merged_evidence = list({_json(item): item for item in evidence_items}.values())[
            :GRAPH_SUPPORTING_EVIDENCE_LIMIT
        ]
        merged_aliases = list(dict.fromkeys([*old_aliases, *aliases]))
        merged_memory_ids = list(dict.fromkeys([*old_memory_ids, memory_id]))[
            :GRAPH_SUPPORTING_MEMORY_LIMIT
        ]
        connection.execute(
            """INSERT INTO entity_graph_entities
            (entity_id, region_id, entity_type, canonical_name, normalized_name, aliases, evidence_memory_ids, evidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET aliases = ?, evidence_memory_ids = ?, evidence = ?, updated_at = ?""",
            (
                entity_id,
                region_id,
                entity_type,
                canonical,
                normalized,
                _json(merged_aliases),
                _json(merged_memory_ids),
                _json(merged_evidence),
                now,
                now,
                _json(merged_aliases),
                _json(merged_memory_ids),
                _json(merged_evidence),
                now,
            ),
        )

    @staticmethod
    def _cleanup(connection: sqlite3.Connection, region_id: str) -> None:
        connection.execute(
            "DELETE FROM entity_graph_entities WHERE region_id = ? AND entity_id NOT IN (SELECT entity_id FROM entity_graph_memory_entities WHERE region_id = ?)",
            (region_id, region_id),
        )
        connection.execute(
            "DELETE FROM entity_graph_aliases WHERE region_id = ? AND entity_id NOT IN (SELECT entity_id FROM entity_graph_entities WHERE region_id = ?)",
            (region_id, region_id),
        )
        connection.execute(
            "DELETE FROM entity_graph_relations WHERE region_id = ? AND (source_entity_id NOT IN (SELECT entity_id FROM entity_graph_entities WHERE region_id = ?) OR target_entity_id NOT IN (SELECT entity_id FROM entity_graph_entities WHERE region_id = ?))",
            (region_id, region_id, region_id),
        )
        connection.execute(
            "DELETE FROM entity_graph_memory_relations WHERE region_id = ? AND relation_id NOT IN (SELECT relation_id FROM entity_graph_relations WHERE region_id = ?)",
            (region_id, region_id),
        )

    def clear(self, region_id: str) -> None:
        with self._connection() as connection:
            for table in (
                "entity_graph_memory_relations",
                "entity_graph_memory_entities",
                "entity_graph_relations",
                "entity_graph_aliases",
                "entity_graph_entities",
            ):
                connection.execute(f"DELETE FROM {table} WHERE region_id = ?", (region_id,))

    def status(
        self,
        region_id: str,
        *,
        history: bool = False,
        pending: int = 0,
    ) -> dict[str, object]:
        lifecycles = ("active", "superseded", "archived") if history else ("active",)
        lifecycle_placeholders = ", ".join("?" for _ in lifecycles)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT status, projected_memories, rejected_items, last_error, updated_at FROM entity_graph_status WHERE region_id = ?",
                (region_id,),
            ).fetchone()
            counts = connection.execute(
                f"""SELECT COUNT(DISTINCT entity_id)
                    FROM entity_graph_memory_entities
                    WHERE region_id = ? AND lifecycle IN ({lifecycle_placeholders})""",
                (region_id, *lifecycles),
            ).fetchone()
            relations = connection.execute(
                f"""SELECT COUNT(DISTINCT relation_id)
                    FROM entity_graph_memory_relations
                    WHERE region_id = ? AND lifecycle IN ({lifecycle_placeholders})""",
                (region_id, *lifecycles),
            ).fetchone()
            projected = connection.execute(
                f"""SELECT COUNT(DISTINCT memory_id)
                    FROM (
                        SELECT memory_id FROM entity_graph_memory_entities
                        WHERE region_id = ? AND lifecycle IN ({lifecycle_placeholders})
                        UNION
                        SELECT memory_id FROM entity_graph_memory_relations
                        WHERE region_id = ? AND lifecycle IN ({lifecycle_placeholders})
                    )""",
                (region_id, *lifecycles, region_id, *lifecycles),
            ).fetchone()
        last_error = str(row[3]) if row else ""
        node_count = int(counts[0]) if counts else 0
        edge_count = int(relations[0]) if relations else 0
        projected_count = int(projected[0]) if projected else 0
        return {
            "status": str(row[0]) if row else "ready",
            "region_id": region_id,
            "history": history,
            "entities": node_count,
            "relations": edge_count,
            "nodes": node_count,
            "edges": edge_count,
            "projected_memories": projected_count,
            "pending": max(0, pending),
            "pending_count": max(0, pending),
            "rejected_items": int(row[2]) if row else 0,
            "last_error": last_error,
            "error": last_error,
            "updated_at": str(row[4]) if row else datetime.now(UTC).isoformat(),
        }

    def not_configured(self, region_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO entity_graph_status
                (region_id, status, projected_memories, rejected_items, last_error, updated_at)
                VALUES (?, 'not_configured', 0, 0, ?, ?)
                ON CONFLICT(region_id) DO UPDATE SET
                    status = 'not_configured',
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at""",
                (region_id, "entity graph extractor is not configured", now),
            )

    def relation_detail(
        self,
        region_id: str,
        relation_id: str,
        *,
        limit: int = DETAIL_DEFAULT_LIMIT,
        offset: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, object] | None:
        limit, offset = _detail_window(limit, offset, cursor)
        with self._connection() as connection:
            relation_row = connection.execute(
                """SELECT relation_id, source_entity_id, target_entity_id, relation_type, evidence
                   FROM entity_graph_relations
                   WHERE region_id = ? AND relation_id = ?""",
                (region_id, relation_id),
            ).fetchone()
            if relation_row is None:
                return None
            source_row = connection.execute(
                """SELECT entity_id, entity_type, canonical_name, normalized_name, aliases, evidence
                   FROM entity_graph_entities
                   WHERE region_id = ? AND entity_id = ?""",
                (region_id, relation_row[1]),
            ).fetchone()
            target_row = connection.execute(
                """SELECT entity_id, entity_type, canonical_name, normalized_name, aliases, evidence
                   FROM entity_graph_entities
                   WHERE region_id = ? AND entity_id = ?""",
                (region_id, relation_row[2]),
            ).fetchone()
            supporting_count_row = connection.execute(
                """SELECT COUNT(*) FROM entity_graph_memory_relations
                   WHERE region_id = ? AND relation_id = ?""",
                (region_id, relation_id),
            ).fetchone()
            memory_rows = connection.execute(
                """SELECT memory_id FROM entity_graph_memory_relations
                   WHERE region_id = ? AND relation_id = ?
                   ORDER BY memory_id LIMIT ? OFFSET ?""",
                (region_id, relation_id, limit, offset),
            ).fetchall()
        if source_row is None or target_row is None:
            return None
        memory_ids = list(dict.fromkeys(str(row[0]) for row in memory_rows))
        supporting_count = int(supporting_count_row[0]) if supporting_count_row else 0
        relation = self._relation_payload(relation_row, evidence_limit=limit)
        relation["memory_ids"] = memory_ids
        source_entity = self._entity_payload(source_row, evidence_limit=limit)
        target_entity = self._entity_payload(target_row, evidence_limit=limit)
        page = _detail_page(supporting_count, memory_ids, limit, offset)
        return {
            "region_id": region_id,
            "relation": relation,
            "source_entity": source_entity,
            "target_entity": target_entity,
            "supporting_memory_ids": memory_ids,
            **page,
            "graph": {
                "region_id": region_id,
                "nodes": [source_entity, target_entity],
                "edges": [relation],
                "entities": [source_entity, target_entity],
                "relations": [relation],
                "hops": 1,
                "read_only": True,
            },
        }

    def entity_detail(
        self,
        region_id: str,
        entity_id: str,
        *,
        limit: int = DETAIL_DEFAULT_LIMIT,
        offset: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, object] | None:
        limit, offset = _detail_window(limit, offset, cursor)
        with self._connection() as connection:
            entity_row = connection.execute(
                """SELECT entity_id, entity_type, canonical_name, normalized_name, aliases, evidence
                   FROM entity_graph_entities WHERE region_id = ? AND entity_id = ?""",
                (region_id, entity_id),
            ).fetchone()
            if entity_row is None:
                return None
            relation_rows = connection.execute(
                """SELECT relation_id, source_entity_id, target_entity_id, relation_type, evidence
                   FROM entity_graph_relations
                   WHERE region_id = ? AND (source_entity_id = ? OR target_entity_id = ?)
                   ORDER BY relation_type, relation_id LIMIT 200""",
                (region_id, entity_id, entity_id),
            ).fetchall()
            relation_ids = [str(row[0]) for row in relation_rows]
            supporting_count_row = connection.execute(
                """WITH supporting(memory_id) AS (
                       SELECT memory_id
                       FROM entity_graph_memory_entities
                       WHERE region_id = ? AND entity_id = ?
                       UNION
                       SELECT m.memory_id
                       FROM entity_graph_memory_relations m
                       JOIN entity_graph_relations r
                         ON r.region_id = m.region_id AND r.relation_id = m.relation_id
                       WHERE m.region_id = ?
                         AND (r.source_entity_id = ? OR r.target_entity_id = ?)
                   )
                   SELECT COUNT(*) FROM supporting""",
                (region_id, entity_id, region_id, entity_id, entity_id),
            ).fetchone()
            memory_rows = connection.execute(
                """WITH supporting(memory_id) AS (
                       SELECT memory_id
                       FROM entity_graph_memory_entities
                       WHERE region_id = ? AND entity_id = ?
                       UNION
                       SELECT m.memory_id
                       FROM entity_graph_memory_relations m
                       JOIN entity_graph_relations r
                         ON r.region_id = m.region_id AND r.relation_id = m.relation_id
                       WHERE m.region_id = ?
                         AND (r.source_entity_id = ? OR r.target_entity_id = ?)
                   )
                   SELECT memory_id FROM supporting
                   ORDER BY memory_id LIMIT ? OFFSET ?""",
                (region_id, entity_id, region_id, entity_id, entity_id, limit, offset),
            ).fetchall()
            relation_memory_rows = []
            memory_ids = list(dict.fromkeys(str(row[0]) for row in memory_rows))
            if relation_ids and memory_ids:
                relation_placeholders = ", ".join("?" for _ in relation_ids)
                memory_placeholders = ", ".join("?" for _ in memory_ids)
                relation_memory_rows = connection.execute(
                    f"""SELECT relation_id, memory_id FROM entity_graph_memory_relations
                        WHERE region_id = ? AND relation_id IN ({relation_placeholders})
                          AND memory_id IN ({memory_placeholders})
                        ORDER BY relation_id, memory_id""",
                    (region_id, *relation_ids, *memory_ids),
                ).fetchall()
        entity = self._entity_payload(entity_row, evidence_limit=limit)
        relations = [
            self._relation_payload(row, evidence_limit=limit) for row in relation_rows
        ]
        relation_memory_ids: dict[str, list[str]] = {}
        for row in relation_memory_rows:
            relation_memory_ids.setdefault(str(row[0]), []).append(str(row[1]))
        for relation in relations:
            relation["memory_ids"] = list(
                dict.fromkeys(relation_memory_ids.get(str(relation["id"]), []))
            )
        supporting_count = int(supporting_count_row[0]) if supporting_count_row else 0
        page = _detail_page(supporting_count, memory_ids, limit, offset)
        return {
            "region_id": region_id,
            "entity": entity,
            "relations": relations,
            "supporting_memory_ids": memory_ids,
            **page,
            "graph": {
                "region_id": region_id,
                "nodes": [entity],
                "edges": relations,
                "entities": [entity],
                "relations": relations,
                "hops": 1,
                "read_only": True,
            },
        }

    def fail(self, region_id: str, reason: str) -> None:
        safe_reason = "entity graph projection failed"
        if reason in {
            "entity graph extractor is not configured",
            "entity graph extractor returned invalid data",
        }:
            safe_reason = reason
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            projected = connection.execute(
                "SELECT COUNT(DISTINCT memory_id) FROM entity_graph_memory_entities WHERE region_id = ?",
                (region_id,),
            ).fetchone()
            rejected = connection.execute(
                "SELECT rejected_items FROM entity_graph_status WHERE region_id = ?",
                (region_id,),
            ).fetchone()
            connection.execute(
                """INSERT INTO entity_graph_status
                (region_id, status, projected_memories, rejected_items, last_error, updated_at)
                VALUES (?, 'degraded', ?, ?, ?, ?)
                ON CONFLICT(region_id) DO UPDATE SET status = 'degraded', projected_memories = excluded.projected_memories,
                    rejected_items = excluded.rejected_items, last_error = excluded.last_error, updated_at = excluded.updated_at""",
                (region_id, int(projected[0]) if projected else 0, int(rejected[0]) if rejected else 0, safe_reason, now),
            )

    def memory_detail(self, region_id: str, memory_id: str) -> dict[str, object]:
        with self._connection() as connection:
            entities = connection.execute(
                """SELECT e.entity_id, e.entity_type, e.canonical_name, e.normalized_name, e.aliases, e.evidence
                FROM entity_graph_entities e JOIN entity_graph_memory_entities m ON m.entity_id = e.entity_id AND m.region_id = e.region_id
                WHERE m.region_id = ? AND m.memory_id = ?
                ORDER BY e.entity_type, e.normalized_name LIMIT ?""",
                (region_id, memory_id, GRAPH_MAX_NODE_LIMIT),
            ).fetchall()
            relations = connection.execute(
                """SELECT r.relation_id, r.source_entity_id, r.target_entity_id, r.relation_type, r.evidence
                FROM entity_graph_relations r JOIN entity_graph_memory_relations m ON m.relation_id = r.relation_id AND m.region_id = r.region_id
                WHERE m.region_id = ? AND m.memory_id = ?
                ORDER BY r.relation_type, r.relation_id LIMIT ?""",
                (region_id, memory_id, GRAPH_MAX_EDGE_LIMIT),
            ).fetchall()
            relation_ids = [str(row[0]) for row in relations]
            relation_memory_rows = self._bounded_relation_memory_rows(
                connection, region_id, relation_ids
            )
        relation_memory_ids: dict[str, list[str]] = {}
        for row in relation_memory_rows:
            relation_memory_ids.setdefault(str(row[0]), []).append(str(row[1]))
        relation_payloads = [
            self._relation_payload(row, evidence_limit=GRAPH_SUPPORTING_EVIDENCE_LIMIT)
            for row in relations
        ]
        for relation in relation_payloads:
            relation["memory_ids"] = list(
                dict.fromkeys(relation_memory_ids.get(str(relation["id"]), []))
            )
        return {
            "entities": [
                self._entity_payload(
                    row, evidence_limit=GRAPH_SUPPORTING_EVIDENCE_LIMIT
                )
                for row in entities
            ],
            "relations": relation_payloads,
        }

    def graph(
        self,
        region_id: str,
        *,
        entity_id: str | None = None,
        memory_id: str | None = None,
        entity_type: str | None = None,
        q: str | None = None,
        hops: int = 1,
        limit: int = 200,
        history: bool = False,
    ) -> dict[str, object]:
        if hops not in {1, 2}:
            raise ValueError("hops must be 1 or 2")
        if entity_type is not None and entity_type not in ENTITY_TYPES:
            allowed = ", ".join(sorted(ENTITY_TYPES))
            raise ValueError(f"entity_type must be one of: {allowed}")
        limit = max(1, min(limit, GRAPH_MAX_NODE_LIMIT))
        lifecycles = (
            ("active", "superseded", "archived") if history else ("active",)
        )
        edge_budget = min(GRAPH_MAX_EDGE_LIMIT, max(1, limit * 4))
        relation_limit = min(GRAPH_MAX_EDGE_LIMIT + 1, edge_budget + 1)
        candidate_limit = min(GRAPH_MAX_NODE_LIMIT + 1, limit + 1)
        normalized_q = normalize_name(q) if q is not None else ""
        has_explicit_seed = (
            entity_id is not None or memory_id is not None or q is not None
        )
        with self._connection() as connection:
            seed_ids: list[str] = []
            seed_rows: list[sqlite3.Row] = []
            if entity_id is not None:
                if entity_id:
                    seed_rows = self._candidate_entity_rows(
                        connection,
                        region_id,
                        history=history,
                        entity_type=entity_type,
                        entity_ids=[entity_id],
                        limit=1,
                    )
                seed_ids = [str(row[0]) for row in seed_rows]
            elif memory_id is not None:
                if memory_id:
                    seed_rows = connection.execute(
                        f"""SELECT DISTINCT m.entity_id
                            FROM entity_graph_memory_entities m
                            JOIN entity_graph_entities e
                              ON e.region_id = m.region_id AND e.entity_id = m.entity_id
                            WHERE m.region_id = ? AND m.memory_id = ?
                              AND m.lifecycle IN ({', '.join('?' for _ in lifecycles)})
                              {"AND e.entity_type = ?" if entity_type else ""}
                            ORDER BY e.entity_type, e.normalized_name, e.entity_id
                            LIMIT ?""",
                        (
                            region_id,
                            memory_id,
                            *lifecycles,
                            *((entity_type,) if entity_type else ()),
                            limit,
                        ),
                    ).fetchall()
                seed_ids = [str(row[0]) for row in seed_rows]
            elif q is not None and normalized_q:
                seed_rows = self._candidate_entity_rows(
                    connection,
                    region_id,
                    history=history,
                    entity_type=entity_type,
                    q=normalized_q,
                    # Reserve the direct matches before adding any neighborhood
                    # candidates.  The graph output is bounded by ``limit``;
                    # direct matches must not be lost to the expansion query.
                    limit=limit,
                )
                seed_ids = [str(row[0]) for row in seed_rows]

            relation_rows: list[sqlite3.Row] = []
            selected_ids = set(seed_ids[:limit])
            frontier = set(selected_ids)
            for _ in range(hops if selected_ids else 0):
                if not frontier:
                    break
                rows = self._relation_rows_for_entities(
                    connection,
                    region_id,
                    sorted(frontier),
                    relation_limit,
                    history=history,
                    outward=True,
                )
                known_relations = {str(row[0]) for row in relation_rows}
                relation_rows.extend(
                    row for row in rows if str(row[0]) not in known_relations
                )
                next_ids = {
                    str(row[1]) if str(row[2]) in frontier else str(row[2])
                    for row in rows
                }
                next_ids -= selected_ids
                if not next_ids:
                    break
                selected_ids.update(next_ids)
                if len(selected_ids) > GRAPH_MAX_NODE_LIMIT * 2:
                    selected_ids = set(
                        [*seed_ids[:limit], *sorted(selected_ids - set(seed_ids))][
                            : GRAPH_MAX_NODE_LIMIT * 2
                        ]
                    )
                frontier = next_ids

            if selected_ids:
                entity_rows = self._candidate_entity_rows(
                    connection,
                    region_id,
                    history=history,
                    entity_type=entity_type,
                    entity_ids=sorted(selected_ids),
                    # ``selected_ids`` is capped at twice the graph node
                    # budget below.  Fetch that bounded working set in full so
                    # final ordering can keep every direct seed before filling
                    # the remaining node budget with background entities.
                    limit=GRAPH_MAX_NODE_LIMIT * 2,
                )
            elif has_explicit_seed:
                # An explicit seed with no match is an empty graph query, not
                # a request for the default whole-region graph.
                # Keep the normal bounded/read-only response contract below.
                entity_rows = []
            else:
                entity_rows = self._candidate_entity_rows(
                    connection,
                    region_id,
                    history=history,
                    entity_type=entity_type,
                    limit=candidate_limit,
                )

            row_by_id = {str(row[0]): row for row in entity_rows}
            seed_set = {entity for entity in seed_ids if entity in row_by_id}
            ordered_rows = [row_by_id[entity] for entity in seed_ids if entity in row_by_id]
            remaining_rows = [row for row in entity_rows if str(row[0]) not in seed_set]
            remaining_rows.sort(key=lambda row: str(row[0]))
            remaining_rows.sort(key=lambda row: str(row[3]))
            remaining_rows.sort(key=lambda row: str(row[1]))
            remaining_rows.sort(key=lambda row: str(row[7]), reverse=True)
            remaining_rows.sort(key=lambda row: int(row[6]), reverse=True)
            ordered_rows.extend(remaining_rows)
            nodes_truncated = len(ordered_rows) > limit
            entity_rows = ordered_rows[:limit]
            entity_ids = [str(row[0]) for row in entity_rows]
            relation_rows = self._relation_rows_for_entities(
                connection,
                region_id,
                entity_ids,
                relation_limit,
                history=history,
            )
            edges_truncated = len(relation_rows) > edge_budget
            relation_ids = [str(row[0]) for row in relation_rows]
            entity_supports = self._entity_support_rows(
                connection, region_id, entity_ids, lifecycles
            )
            relation_supports = self._relation_support_rows(
                connection, region_id, relation_ids, lifecycles
            )

        entities_list = []
        for row in entity_rows:
            entity = self._entity_payload(
                row,
                evidence_limit=GRAPH_SUPPORTING_EVIDENCE_LIMIT,
                supports=entity_supports.get(str(row[0]), []),
            )
            entity["selected"] = str(row[0]) in seed_set
            entity["graph_state"] = "direct" if entity["selected"] else "background"
            entity["degree"] = int(row[6])
            entity["updated_at"] = str(row[7])
            entities_list.append(entity)
        entity_id_set = {str(item["id"]) for item in entities_list}
        relations = [
            self._relation_payload(
                row,
                evidence_limit=GRAPH_SUPPORTING_EVIDENCE_LIMIT,
                supports=relation_supports.get(str(row[0]), []),
            )
            for row in relation_rows
            if str(row[1]) in entity_id_set and str(row[2]) in entity_id_set
        ][:edge_budget]
        for edge in relations:
            edge["memory_ids"] = [
                str(row[1]) for row in relation_supports.get(str(edge["id"]), [])
            ]
        return {
            "region_id": region_id,
            "nodes": entities_list,
            "edges": relations,
            "entities": entities_list,
            "relations": relations,
            "hops": hops,
            "history": history,
            "mode": "history" if history else "active",
            "max_nodes": GRAPH_MAX_NODE_LIMIT,
            "max_edges": GRAPH_MAX_EDGE_LIMIT,
            "truncated": nodes_truncated or edges_truncated,
            "read_only": True,
        }

    @staticmethod
    def _candidate_entity_rows(
        connection: sqlite3.Connection,
        region_id: str,
        *,
        history: bool,
        entity_type: str | None = None,
        q: str | None = None,
        entity_ids: Sequence[str] | None = None,
        limit: int = GRAPH_MAX_NODE_LIMIT,
    ) -> list[sqlite3.Row]:
        lifecycles = (
            ("active", "superseded", "archived") if history else ("active",)
        )
        lifecycle_placeholders = ", ".join("?" for _ in lifecycles)
        where = [
            "e.region_id = ?",
            f"em.lifecycle IN ({lifecycle_placeholders})",
        ]
        values: list[object] = [region_id, *lifecycles]
        if entity_type:
            where.append("e.entity_type = ?")
            values.append(entity_type)
        if q:
            query = f"%{_escape_like_literal(normalize_name(q))}%"
            where.append(
                "(e.normalized_name LIKE ? ESCAPE '\\' "
                "OR LOWER(e.aliases) LIKE LOWER(?) ESCAPE '\\')"
            )
            values.extend([query, query])
        if entity_ids is not None:
            if not entity_ids:
                return []
            placeholders = ", ".join("?" for _ in entity_ids)
            where.append(f"e.entity_id IN ({placeholders})")
            values.extend(entity_ids)
        return connection.execute(
            f"""SELECT e.entity_id, e.entity_type, e.canonical_name, e.normalized_name,
                       e.aliases, e.evidence,
                       COUNT(DISTINCT CASE WHEN rm.relation_id IS NOT NULL THEN r.relation_id END)
                           AS degree,
                       MAX(em.updated_at) AS recent_updated_at
                FROM entity_graph_entities e
                JOIN entity_graph_memory_entities em
                  ON em.region_id = e.region_id AND em.entity_id = e.entity_id
                LEFT JOIN entity_graph_relations r
                  ON r.region_id = e.region_id
                 AND (r.source_entity_id = e.entity_id OR r.target_entity_id = e.entity_id)
                LEFT JOIN entity_graph_memory_relations rm
                  ON rm.region_id = r.region_id AND rm.relation_id = r.relation_id
                 AND rm.lifecycle IN ({lifecycle_placeholders})
                WHERE {' AND '.join(where)}
                GROUP BY e.entity_id, e.entity_type, e.canonical_name,
                         e.normalized_name, e.aliases, e.evidence
                ORDER BY degree DESC, recent_updated_at DESC, e.entity_type,
                         e.normalized_name, e.entity_id
                LIMIT ?""",
            (*lifecycles, *values, max(1, limit)),
        ).fetchall()

    @staticmethod
    def _entity_support_rows(
        connection: sqlite3.Connection,
        region_id: str,
        entity_ids: Sequence[str],
        lifecycles: Sequence[str],
    ) -> dict[str, list[sqlite3.Row]]:
        result: dict[str, list[sqlite3.Row]] = {}
        placeholders = ", ".join("?" for _ in lifecycles)
        for entity_id in entity_ids:
            result[entity_id] = connection.execute(
                f"""SELECT region_id, memory_id, entity_id, canonical_name, aliases,
                           evidence, lifecycle, updated_at
                    FROM entity_graph_memory_entities
                    WHERE region_id = ? AND entity_id = ?
                      AND lifecycle IN ({placeholders})
                    ORDER BY updated_at DESC, memory_id LIMIT ?""",
                (region_id, entity_id, *lifecycles, GRAPH_SUPPORTING_MEMORY_LIMIT),
            ).fetchall()
        return result

    @staticmethod
    def _relation_support_rows(
        connection: sqlite3.Connection,
        region_id: str,
        relation_ids: Sequence[str],
        lifecycles: Sequence[str],
    ) -> dict[str, list[sqlite3.Row]]:
        result: dict[str, list[sqlite3.Row]] = {}
        placeholders = ", ".join("?" for _ in lifecycles)
        for relation_id in relation_ids:
            result[relation_id] = connection.execute(
                f"""SELECT region_id, memory_id, relation_id, evidence, lifecycle, updated_at
                    FROM entity_graph_memory_relations
                    WHERE region_id = ? AND relation_id = ?
                      AND lifecycle IN ({placeholders})
                    ORDER BY updated_at DESC, memory_id LIMIT ?""",
                (region_id, relation_id, *lifecycles, GRAPH_SUPPORTING_MEMORY_LIMIT),
            ).fetchall()
        return result

    @staticmethod
    def _relation_rows_for_entities(
        connection: sqlite3.Connection,
        region_id: str,
        entity_ids: Sequence[str],
        limit: int,
        *,
        history: bool = False,
        outward: bool = False,
    ) -> list[sqlite3.Row]:
        if not entity_ids:
            return []
        placeholders = ", ".join("?" for _ in entity_ids)
        lifecycles = (
            ("active", "superseded", "archived") if history else ("active",)
        )
        lifecycle_placeholders = ", ".join("?" for _ in lifecycles)
        endpoint_filter = (
            f"(r.source_entity_id IN ({placeholders}) OR "
            f"r.target_entity_id IN ({placeholders}))"
            if outward
            else f"r.source_entity_id IN ({placeholders}) AND "
            f"r.target_entity_id IN ({placeholders})"
        )
        return connection.execute(
            f"""SELECT r.relation_id, r.source_entity_id, r.target_entity_id,
                       r.relation_type, r.evidence
                FROM entity_graph_relations r
                JOIN entity_graph_memory_relations m
                  ON m.region_id = r.region_id AND m.relation_id = r.relation_id
                 AND m.lifecycle IN ({lifecycle_placeholders})
                WHERE r.region_id = ? AND {endpoint_filter}
                GROUP BY r.relation_id, r.source_entity_id, r.target_entity_id,
                         r.relation_type, r.evidence
                ORDER BY MAX(m.updated_at) DESC, r.relation_type, r.relation_id LIMIT ?""",
            (*lifecycles, region_id, *entity_ids, *entity_ids, max(1, limit)),
        ).fetchall()

    @staticmethod
    def _bounded_relation_memory_rows(
        connection: sqlite3.Connection,
        region_id: str,
        relation_ids: Sequence[str],
        *,
        limit: int = GRAPH_SUPPORTING_MEMORY_LIMIT,
    ) -> list[sqlite3.Row]:
        bounded_limit = min(max(limit, 1), GRAPH_SUPPORTING_MEMORY_LIMIT)
        rows: list[sqlite3.Row] = []
        for relation_id in dict.fromkeys(str(value) for value in relation_ids):
            rows.extend(
                connection.execute(
                    """SELECT relation_id, memory_id
                       FROM entity_graph_memory_relations
                       WHERE region_id = ? AND relation_id = ?
                       ORDER BY memory_id LIMIT ?""",
                    (region_id, relation_id, bounded_limit),
                ).fetchall()
            )
        return rows

    @staticmethod
    def _entity_payload(
        row: sqlite3.Row,
        *,
        evidence_limit: int | None = None,
        supports: Sequence[sqlite3.Row] | None = None,
    ) -> dict[str, object]:
        support_rows = list(supports or [])
        aliases: list[str] = _string_list(_loads(row[4], []))
        evidence: list[object] = []
        raw_evidence = _loads(row[5], [])
        if isinstance(raw_evidence, list):
            evidence.extend(raw_evidence)
        lifecycle = "active"
        updated_at = ""
        if support_rows:
            ordered_support_rows = sorted(support_rows, key=lambda item: str(item[1]))
            aliases = []
            evidence = []
            lifecycle_values = [str(item[6]) for item in support_rows]
            lifecycle = (
                "active"
                if "active" in lifecycle_values
                else lifecycle_values[0]
            )
            updated_at = max(str(item[7]) for item in support_rows)
            for support in ordered_support_rows:
                aliases.extend(_string_list(_loads(support[4], [])))
                parsed = _loads(support[5], [])
                if isinstance(parsed, list):
                    evidence.extend(parsed)
            aliases = list(dict.fromkeys(aliases))
        evidence_items = evidence
        effective_evidence_limit = (
            GRAPH_SUPPORTING_EVIDENCE_LIMIT
            if evidence_limit is None
            else min(evidence_limit, GRAPH_SUPPORTING_EVIDENCE_LIMIT)
        )
        evidence_items = evidence_items[:effective_evidence_limit]
        evidence_memory_ids = list(
            dict.fromkeys(
                str(item["memory_id"])
                for item in evidence_items
                if isinstance(item, Mapping) and item.get("memory_id")
            )
        )[:GRAPH_SUPPORTING_MEMORY_LIMIT]
        payload: dict[str, object] = {
            "id": str(row[0]),
            "type": str(row[1]),
            "entity_type": str(row[1]),
            "canonical_name": str(row[2]),
            "normalized_name": str(row[3]),
            "aliases": aliases if isinstance(aliases, list) else [],
            "evidence_memory_ids": evidence_memory_ids,
            "evidence": evidence_items,
        }
        if support_rows:
            payload["lifecycle"] = lifecycle
            payload["display_state"] = "current" if lifecycle == "active" else "historical"
            payload["historical"] = lifecycle != "active"
            payload["updated_at"] = updated_at
        return payload

    @staticmethod
    def _relation_payload(
        row: sqlite3.Row,
        *,
        evidence_limit: int | None = None,
        supports: Sequence[sqlite3.Row] | None = None,
    ) -> dict[str, object]:
        support_rows = list(supports or [])
        evidence: list[object] = []
        raw_evidence = _loads(row[4], [])
        if isinstance(raw_evidence, list):
            evidence.extend(raw_evidence)
        lifecycle = "active"
        updated_at = ""
        memory_ids: list[str] = []
        if support_rows:
            ordered_support_rows = sorted(support_rows, key=lambda item: str(item[1]))
            evidence = []
            lifecycle_values = [str(item[4]) for item in support_rows]
            lifecycle = (
                "active"
                if "active" in lifecycle_values
                else lifecycle_values[0]
            )
            updated_at = max(str(item[5]) for item in support_rows)
            for support in ordered_support_rows:
                memory_ids.append(str(support[1]))
                parsed = _loads(support[3], [])
                if isinstance(parsed, list):
                    evidence.extend(parsed)
        evidence_items = evidence
        effective_evidence_limit = (
            GRAPH_SUPPORTING_EVIDENCE_LIMIT
            if evidence_limit is None
            else min(evidence_limit, GRAPH_SUPPORTING_EVIDENCE_LIMIT)
        )
        evidence_items = evidence_items[:effective_evidence_limit]
        payload: dict[str, object] = {
            "id": str(row[0]),
            "source": str(row[1]),
            "target": str(row[2]),
            "type": str(row[3]),
            "relation_type": str(row[3]),
            "evidence": evidence_items,
        }
        if support_rows:
            payload["memory_ids"] = list(dict.fromkeys(memory_ids))
            payload["lifecycle"] = lifecycle
            payload["display_state"] = "current" if lifecycle == "active" else "historical"
            payload["historical"] = lifecycle != "active"
            payload["updated_at"] = updated_at
        return payload
