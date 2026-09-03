from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import stat
import uuid
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from markdown_it import MarkdownIt

DEFAULT_IGNORES = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    ".personal-agent-memory",
)

_MARKDOWN = MarkdownIt("commonmark")
_MAX_EXACT_ALIGNMENT_CELLS = 2_000_000


@dataclass(frozen=True, slots=True)
class MarkdownDocument:
    path: str
    content: str
    version: str


@dataclass(frozen=True, slots=True)
class MarkdownChunk:
    kind: str
    heading: str | None
    start_line: int
    end_line: int
    content: str


@dataclass(frozen=True, slots=True)
class StoredChunk:
    id: str
    kind: str
    content: str
    heading: str | None
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class MarkdownScan:
    documents: tuple[MarkdownDocument, ...]
    errors: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.errors


def validate_ignore_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in patterns:
        pattern = raw.strip().replace("\\", "/")
        if not pattern:
            continue
        candidate = PurePosixPath(pattern.removesuffix("/"))
        if candidate.is_absolute() or ".." in candidate.parts or "\x00" in pattern:
            raise ValueError("ignore patterns must be relative and cannot contain '..'")
        normalized.append(pattern)
    return tuple(dict.fromkeys(normalized))


def scan_markdown(
    root: Path, custom_ignores: tuple[str, ...], excluded_roots: tuple[Path, ...] = ()
) -> MarkdownScan:
    documents: list[MarkdownDocument] = []
    errors: list[str] = []

    def walk(directory: Path, relative: PurePosixPath) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            errors.append(f"cannot enumerate directory: {relative.as_posix()}")
            return
        for entry in entries:
            child_relative = relative / entry.name
            relative_text = child_relative.as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    continue
                if stat.S_ISDIR(metadata.st_mode):
                    child_path = Path(entry.path)
                    if _excluded(child_path, excluded_roots) or _ignored(
                        relative_text, entry.name, True, custom_ignores
                    ):
                        continue
                    walk(child_path, child_relative)
                elif (
                    stat.S_ISREG(metadata.st_mode)
                    and entry.name.lower().endswith(".md")
                    and not _ignored(relative_text, entry.name, False, custom_ignores)
                ):
                    content, error = _read_regular_utf8(Path(entry.path), root)
                    if error is not None:
                        errors.append(f"{relative_text}: {error}")
                    elif content is not None:
                        documents.append(
                            MarkdownDocument(
                                path=relative_text,
                                content=content,
                                version=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                            )
                        )
            except OSError as error:
                errors.append(f"cannot inspect {relative_text}: {error.strerror or error}")
                continue

    walk(root, PurePosixPath())
    return MarkdownScan(tuple(documents), tuple(errors))


def chunk_markdown(content: str) -> list[MarkdownChunk]:
    lines = content.splitlines(keepends=True)
    headings: list[str] = []
    chunks: list[MarkdownChunk] = []
    pending_heading: tuple[int, int, str | None] | None = None

    def title() -> str | None:
        return " > ".join(headings) if headings else None

    def append(
        start: int, end: int, kind: str, heading: str | None = None
    ) -> None:
        nonlocal pending_heading
        text = "".join(lines[start:end]).strip("\n")
        if not text.strip():
            return
        chunks.append(
            MarkdownChunk(
                kind=kind,
                heading=title() if heading is None else heading,
                start_line=start + 1,
                end_line=end,
                content=text,
            )
        )
        if kind != "heading":
            pending_heading = None

    def flush_pending_heading() -> None:
        nonlocal pending_heading
        if pending_heading is None:
            return
        start, end, heading = pending_heading
        append(start, end, "heading", heading)
        pending_heading = None

    environment: dict[str, Any] = {}
    tokens = _MARKDOWN.parse(content, environment)
    for index, token in enumerate(tokens):
        source_map = token.map
        if token.type == "heading_open" and token.level == 0:
            flush_pending_heading()
            level = int(token.tag.removeprefix("h"))
            heading_text = " ".join(tokens[index + 1].content.split())
            headings[level - 1 :] = [heading_text]
            if source_map is not None:
                pending_heading = (source_map[0], source_map[1], title())
        elif source_map is not None and token.type == "heading_open":
            append(source_map[0], source_map[1], "heading")
        elif source_map is not None and token.type == "paragraph_open":
            append(source_map[0], source_map[1], "paragraph")
        elif source_map is not None and token.type in {"fence", "code_block"}:
            append(source_map[0], source_map[1], "code")
        elif source_map is not None and token.type == "hr":
            append(source_map[0], source_map[1], "thematic_break")
        elif source_map is not None and token.type == "html_block":
            append(source_map[0], source_map[1], "html")

    flush_pending_heading()

    references = environment.get("references", {})
    duplicate_references = environment.get("duplicate_refs", [])
    reference_entries: list[Any] = []
    if isinstance(references, dict):
        reference_entries.extend(references.values())
    if isinstance(duplicate_references, list):
        reference_entries.extend(duplicate_references)

    seen_reference_ranges: set[tuple[int, int]] = set()
    for reference in reference_entries:
        if not isinstance(reference, dict):
            continue
        source_map = reference.get("map")
        if (
            not isinstance(source_map, list)
            or len(source_map) != 2
            or not all(isinstance(value, int) for value in source_map)
        ):
            continue
        start, end = source_map
        if (start, end) in seen_reference_ranges:
            continue
        seen_reference_ranges.add((start, end))
        append(start, end, "reference_definition", _heading_at_line(chunks, start + 1))

    return sorted(chunks, key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.kind))


def _heading_at_line(chunks: list[MarkdownChunk], line: int) -> str | None:
    preceding = [chunk for chunk in chunks if chunk.start_line <= line and chunk.heading]
    return preceding[-1].heading if preceding else None


def infer_legacy_chunk_kind(content: str) -> str:
    chunks = chunk_markdown(content)
    if len(chunks) == 1 and chunks[0].content == content.strip("\n"):
        return chunks[0].kind
    return "paragraph"


def stable_document_id(library_id: str, path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pam:{library_id}:{path}"))


def match_chunk_ids(old_chunks: list[StoredChunk], new_chunks: list[MarkdownChunk]) -> list[str]:
    """Carry sidecar identities without allowing an ID to change block type or content."""
    old_groups: dict[tuple[str, str], list[StoredChunk]] = {}
    new_groups: dict[tuple[str, str], list[int]] = {}
    for stored in old_chunks:
        old_groups.setdefault((stored.kind, stored.content), []).append(stored)
    for index, current in enumerate(new_chunks):
        new_groups.setdefault((current.kind, current.content), []).append(index)

    anchors: list[tuple[int, int]] = []
    heading_renames: set[tuple[str | None, str | None]] = set()
    for signature, old_candidates in old_groups.items():
        new_candidates = new_groups.get(signature, [])
        if len(old_candidates) == len(new_candidates) == 1:
            old_chunk = old_candidates[0]
            new_chunk = new_chunks[new_candidates[0]]
            anchors.append(
                (
                    old_chunk.start_line,
                    new_chunk.start_line - old_chunk.start_line,
                )
            )
            heading_renames.add((old_chunk.heading, new_chunk.heading))
    anchors.sort()

    matched: list[str | None] = [None] * len(new_chunks)
    for signature, new_indexes in new_groups.items():
        available_old = sorted(
            old_groups.get(signature, []),
            key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.id),
        )
        remaining_old = set(range(len(available_old)))
        remaining_new = set(new_indexes)

        exact_heading_pairs = {
            (chunk.heading, chunk.heading) for chunk in available_old
        }
        rename_pairs = heading_renames - exact_heading_pairs
        heading_pairs = [
            *sorted(exact_heading_pairs, key=lambda pair: str(pair[0])),
            *sorted(rename_pairs, key=lambda pair: (str(pair[0]), str(pair[1]))),
        ]
        for old_heading, new_heading in heading_pairs:
            old_subset = [
                index
                for index in sorted(remaining_old)
                if available_old[index].heading == old_heading
            ]
            new_subset = [
                index
                for index in sorted(remaining_new)
                if new_chunks[index].heading == new_heading
            ]
            for old_offset, new_index in _minimum_displacement_pairs(
                [available_old[index] for index in old_subset],
                new_subset,
                new_chunks,
                anchors,
                heading_renames,
            ):
                old_index = old_subset[old_offset]
                matched[new_index] = available_old[old_index].id
                remaining_old.remove(old_index)
                remaining_new.remove(new_index)

        # A duplicate that moved under a different heading is distinguishable from
        # the remaining old duplicates. Reuse across headings only when a unique
        # content anchor established that exact heading rename above.
    return [chunk_id if chunk_id is not None else str(uuid.uuid4()) for chunk_id in matched]


def _minimum_displacement_pairs(
    old: list[StoredChunk],
    new_indexes: list[int],
    new: list[MarkdownChunk],
    anchors: list[tuple[int, int]],
    heading_renames: set[tuple[str | None, str | None]],
) -> tuple[tuple[int, int], ...]:
    def cost(old_index: int, new_offset: int) -> int:
        current = new[new_indexes[new_offset]]
        offset = _nearest_edit_offset(old[old_index].start_line, anchors)
        if old[old_index].heading == current.heading:
            heading_cost = 0
        elif (old[old_index].heading, current.heading) in heading_renames:
            heading_cost = 10**6
        else:
            heading_cost = 10**9
        return heading_cost + abs(old[old_index].start_line + offset - current.start_line) + abs(
            old[old_index].end_line + offset - current.end_line
        )

    if not old or not new_indexes:
        return ()
    if len(old) == len(new_indexes):
        return tuple((index, new_indexes[index]) for index in range(len(old)))
    alignment_cells = min(len(old), len(new_indexes)) * (
        abs(len(old) - len(new_indexes)) + 1
    )
    if alignment_cells > _MAX_EXACT_ALIGNMENT_CELLS:
        return _bounded_displacement_pairs(old, new_indexes, new, anchors, cost)
    if len(old) < len(new_indexes):
        return _align_with_skips(len(old), len(new_indexes), cost, new_indexes, skip_new=True)
    return _align_with_skips(len(new_indexes), len(old), cost, new_indexes, skip_new=False)


def _bounded_displacement_pairs(
    old: list[StoredChunk],
    new_indexes: list[int],
    new: list[MarkdownChunk],
    anchors: list[tuple[int, int]],
    cost: Any,
) -> tuple[tuple[int, int], ...]:
    """Choose an ordered alignment without allocating a match-by-delta matrix."""
    pairs: list[tuple[int, int]] = []
    if len(old) < len(new_indexes):
        candidate_lines = [new[index].start_line for index in new_indexes]
        previous = -1
        for old_index, stored in enumerate(old):
            lower = previous + 1
            upper = len(new_indexes) - (len(old) - old_index)
            target = stored.start_line + _nearest_edit_offset(stored.start_line, anchors)
            insertion = bisect_left(candidate_lines, target, lower, upper + 1)
            candidates = {lower, upper, min(insertion, upper), max(insertion - 1, lower)}
            new_offset = min(candidates, key=lambda offset: (cost(old_index, offset), offset))
            pairs.append((old_index, new_indexes[new_offset]))
            previous = new_offset
        return tuple(pairs)

    candidate_lines = [stored.start_line for stored in old]
    previous = -1
    for new_offset, new_index in enumerate(new_indexes):
        lower = previous + 1
        upper = len(old) - (len(new_indexes) - new_offset)
        target = new[new_index].start_line
        insertion = bisect_left(candidate_lines, target, lower, upper + 1)
        candidates = {lower, upper, min(insertion, upper), max(insertion - 1, lower)}
        old_index = min(candidates, key=lambda index: (cost(index, new_offset), index))
        pairs.append((old_index, new_index))
        previous = old_index
    return tuple(pairs)


def _align_with_skips(
    matched_count: int,
    candidate_count: int,
    cost: Any,
    new_indexes: list[int],
    *,
    skip_new: bool,
) -> tuple[tuple[int, int], ...]:
    extra = candidate_count - matched_count
    # With no matches chosen yet, every legal number of leading candidates can
    # be skipped at zero cost. This is what lets a new duplicate be inserted at
    # the front instead of forcing the first old ID onto it.
    previous = [0] * (extra + 1)
    decisions = [bytearray(extra + 1) for _ in range(matched_count + 1)]

    for matched in range(1, matched_count + 1):
        current = [10**18] * (extra + 1)
        for skipped in range(extra + 1):
            old_index = matched - 1 + (0 if skip_new else skipped)
            new_offset = matched - 1 + (skipped if skip_new else 0)
            paired = previous[skipped] + cost(old_index, new_offset)
            if skipped == 0:
                current[skipped] = paired
                decisions[matched][skipped] = 1
                continue
            skipped_cost = current[skipped - 1]
            if paired <= skipped_cost:
                current[skipped] = paired
                decisions[matched][skipped] = 1
            else:
                current[skipped] = skipped_cost
        previous = current

    pairs: list[tuple[int, int]] = []
    matched = matched_count
    skipped = extra
    while matched:
        if decisions[matched][skipped]:
            old_index = matched - 1 + (0 if skip_new else skipped)
            new_offset = matched - 1 + (skipped if skip_new else 0)
            pairs.append((old_index, new_indexes[new_offset]))
            matched -= 1
        else:
            skipped -= 1
    pairs.reverse()
    return tuple(pairs)


def _nearest_edit_offset(position: int, anchors: list[tuple[int, int]]) -> int:
    if not anchors:
        return 0
    insertion = bisect_left(anchors, (position, -1))
    candidates = anchors[max(0, insertion - 1) : min(len(anchors), insertion + 1)]
    _, offset = min(candidates, key=lambda anchor: (abs(anchor[0] - position), anchor[0]))
    return offset


def _ignored(path: str, name: str, is_directory: bool, patterns: tuple[str, ...]) -> bool:
    if any(part in DEFAULT_IGNORES for part in PurePosixPath(path).parts):
        return True
    for pattern in patterns:
        candidate = pattern.removesuffix("/")
        if pattern.endswith("/") and not is_directory:
            continue
        if "/" in candidate:
            if PurePosixPath(path).match(candidate) or fnmatch.fnmatchcase(path, candidate):
                return True
        elif fnmatch.fnmatchcase(name, candidate):
            return True
    return False


def _excluded(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def fts_query(query: str) -> str:
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    if not terms:
        raise ValueError("search query must contain a keyword")
    return " AND ".join(f'"{term}"' for term in terms)


def _read_regular_utf8(path: Path, root: Path) -> tuple[str | None, str | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        return None, f"cannot open Markdown file: {error.strerror or error}"
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None, "source changed while scanning"
        try:
            opened_path = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            if not opened_path.is_relative_to(root):
                return None, "source escaped the memory library while scanning"
        except (OSError, RuntimeError) as error:
            return None, f"cannot verify opened Markdown file: {error}"
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
    except OSError as error:
        return None, f"cannot read Markdown file: {error.strerror or error}"
    finally:
        os.close(descriptor)
    try:
        return bytes(raw).decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "Markdown file is not valid UTF-8"
