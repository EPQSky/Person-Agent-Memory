#!/usr/bin/env python3
"""Dry-run-first initializer for the VibeForge Lite project template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

TEMPLATE_VERSION = "0.3.5"
CURRENT_NAMESPACE = "vibeforge-lite"
LEGACY_NAMESPACE_SHA256 = "53e14fc40e22cdf808c881164f0c78c478b558db6574bf6e834e2011417f6ccd"
MARKER_NAMESPACE = r"(?P<namespace>vibeforge-lite|[a-z0-9-]+-vibecoding)"
START_RE = re.compile(rf"<!-- {MARKER_NAMESPACE}:start(?: version=[^ ]+)? -->")
END_RE = re.compile(rf"<!-- {MARKER_NAMESPACE}:end -->")
STATE_PATH = Path(".vibecoding/state.json")
VENDORED_SKILLS_PATH = Path(".agents/skills")
VENDORED_METADATA_PATH = Path(".agents/vibeforge-lite")
SKILL_MANIFEST_NAME = "skill-manifest.json"
IGNORED_VENDOR_DIRS = {"__pycache__"}
IGNORED_VENDOR_SUFFIXES = {".pyc", ".pyo"}


@dataclass
class Action:
    kind: str
    path: str
    reason: str
    content: str | None = None

    def public(self) -> dict[str, str]:
        return {"action": self.kind, "path": self.path, "reason": self.reason}


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def supported_marker_namespace(namespace: str) -> bool:
    # Keep one retired namespace migratable without restoring it as product branding.
    return namespace == CURRENT_NAMESPACE or sha256(namespace) == LEGACY_NAMESPACE_SHA256


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def distribution_layout() -> tuple[Path, Path]:
    skills_root = Path(__file__).resolve().parents[2]
    distribution_root = skills_root.parent
    if (distribution_root / ".codex-plugin/plugin.json").is_file():
        return skills_root, distribution_root
    metadata_root = distribution_root / "vibeforge-lite"
    if (metadata_root / "plugin.json").is_file():
        return skills_root, metadata_root
    raise RuntimeError("cannot locate VibeForge Lite distribution metadata")


def source_text(path: Path) -> str:
    if path.is_symlink():
        raise RuntimeError(f"distribution contains symbolic link: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"distribution file is not UTF-8 text: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"distribution file cannot be read: {path}") from exc


def locked_skill_names(metadata_root: Path) -> set[str]:
    lock_path = metadata_root / "UPSTREAM.lock"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise RuntimeError("distribution metadata is missing: UPSTREAM.lock")
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError("distribution UPSTREAM.lock is invalid") from exc
    skills = lock.get("skills", {})
    if not isinstance(skills, dict) or not skills:
        raise RuntimeError("distribution UPSTREAM.lock contains no skills")
    names = set(skills)
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) for name in names):
        raise RuntimeError("distribution UPSTREAM.lock contains an unsafe skill name")
    return names


def source_skill_paths(skills_root: Path, metadata_root: Path, skill_names: set[str]) -> list[Path]:
    manifest_path = metadata_root / SKILL_MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(source_text(manifest_path))
        except json.JSONDecodeError as exc:
            raise RuntimeError("distribution skill manifest is invalid JSON") from exc
        entries = manifest.get("files", {}) if isinstance(manifest, dict) else {}
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("template_version") != TEMPLATE_VERSION
            or not isinstance(entries, dict)
            or not entries
        ):
            raise RuntimeError("distribution skill manifest contains no files")
        paths: list[Path] = []
        for raw, expected_hash in entries.items():
            if (
                not isinstance(raw, str)
                or not isinstance(expected_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            ):
                raise RuntimeError("distribution skill manifest contains an unsafe entry")
            relative = Path(raw)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] not in skill_names
            ):
                raise RuntimeError("distribution skill manifest contains an unsafe entry")
            source = skills_root / relative
            content = source_text(source)
            if sha256(content) != expected_hash:
                raise RuntimeError(f"distribution skill differs from manifest: {relative.as_posix()}")
            paths.append(source)
        return sorted(paths)

    distribution_root = skills_root.parent
    if metadata_root == distribution_root and (distribution_root / ".git").exists():
        tracked = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "ls-files", "-z", "--", "skills"],
            cwd=distribution_root,
            check=False,
            capture_output=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError("cannot read tracked distribution skills")
        paths = []
        for raw in tracked.stdout.split(b"\0"):
            if not raw:
                continue
            relative = Path(raw.decode("utf-8"))
            if len(relative.parts) >= 2 and relative.parts[0] == "skills" and relative.parts[1] in skill_names:
                paths.append(distribution_root / relative)
        return sorted(paths)

    paths = []
    for name in sorted(skill_names):
        skill_dir = skills_root / name
        if skill_dir.is_symlink() or not (skill_dir / "SKILL.md").is_file():
            raise RuntimeError(f"distribution skill is missing or unsafe: {name}")
        for source in sorted(skill_dir.rglob("*")):
            relative = source.relative_to(skills_root)
            if any(part in IGNORED_VENDOR_DIRS for part in relative.parts):
                continue
            if source.is_symlink():
                raise RuntimeError(f"distribution contains symbolic link: {relative}")
            if source.is_file() and source.suffix not in IGNORED_VENDOR_SUFFIXES:
                paths.append(source)
    return paths


def vendored_files(skills_root: Path, metadata_root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    skill_names = locked_skill_names(metadata_root)
    source_paths = source_skill_paths(skills_root, metadata_root, skill_names)
    relative_paths = {source.relative_to(skills_root).as_posix() for source in source_paths}
    missing_entrypoints = sorted(f"{name}/SKILL.md" for name in skill_names if f"{name}/SKILL.md" not in relative_paths)
    if missing_entrypoints:
        raise RuntimeError(f"distribution skills are incomplete: {', '.join(missing_entrypoints)}")
    skill_hashes: dict[str, str] = {}
    for source in source_paths:
        relative = source.relative_to(skills_root)
        content = source_text(source)
        destination = VENDORED_SKILLS_PATH / relative
        files[destination.as_posix()] = content
        skill_hashes[relative.as_posix()] = sha256(content)
    installed_manifest = {
        "schema_version": 1,
        "template_version": TEMPLATE_VERSION,
        "files": dict(sorted(skill_hashes.items())),
    }
    files[(VENDORED_METADATA_PATH / SKILL_MANIFEST_NAME).as_posix()] = (
        json.dumps(installed_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )

    plugin_manifest = metadata_root / ".codex-plugin/plugin.json"
    if not plugin_manifest.is_file():
        plugin_manifest = metadata_root / "plugin.json"
    metadata_files = {
        VENDORED_METADATA_PATH / "plugin.json": plugin_manifest,
        VENDORED_METADATA_PATH / "UPSTREAM.lock": metadata_root / "UPSTREAM.lock",
        VENDORED_METADATA_PATH / "THIRD_PARTY_NOTICES.md": metadata_root / "THIRD_PARTY_NOTICES.md",
        VENDORED_METADATA_PATH / "LICENSE": metadata_root / "LICENSE",
    }
    for destination, source in metadata_files.items():
        if not source.is_file():
            raise RuntimeError(f"distribution metadata is missing: {source.name}")
        files[destination.as_posix()] = source_text(source)

    licenses_root = metadata_root / "licenses"
    if not licenses_root.is_dir():
        raise RuntimeError("distribution metadata is missing: licenses")
    for source in sorted(licenses_root.rglob("*")):
        relative = source.relative_to(licenses_root)
        if source.is_symlink():
            raise RuntimeError(f"distribution contains symbolic link: licenses/{relative}")
        if source.is_file():
            destination = VENDORED_METADATA_PATH / "licenses" / relative
            files[destination.as_posix()] = source_text(source)
    return files


def managed_path(root: Path, rel: str | Path) -> Path:
    relative = Path(rel)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"managed path escapes target repository: {relative}")
    return root / relative


def path_conflict(root: Path, rel: str | Path) -> str | None:
    path = managed_path(root, rel)
    current = root
    parts = path.relative_to(root).parts
    for index, part in enumerate(parts):
        current /= part
        if current.is_symlink():
            return f"managed path contains symbolic link: {current.relative_to(root)}"
        if current.exists() and index < len(parts) - 1 and not current.is_dir():
            return f"managed path parent is not a directory: {current.relative_to(root)}"
    if path.exists() and not path.is_file():
        return f"managed path is not a regular file: {path.relative_to(root)}"
    return None


def atomic_write(root: Path, rel: str | Path, content: str) -> None:
    path = managed_path(root, rel)
    conflict = path_conflict(root, rel)
    if conflict:
        raise RuntimeError(conflict)
    path.parent.mkdir(parents=True, exist_ok=True)
    conflict = path_conflict(root, rel)
    if conflict:
        raise RuntimeError(conflict)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_state(root: Path) -> dict:
    raw = read_text(root / STATE_PATH)
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"_error": f"invalid JSON: {exc}"}
    if not isinstance(value, dict):
        return {"_error": "state root is not an object"}
    return value


def managed_template(asset_root: Path) -> str:
    full = (asset_root / "AGENTS.md").read_text(encoding="utf-8")
    start = START_RE.search(full)
    end = END_RE.search(full, start.end() if start else 0)
    if (
        start is None
        or end is None
        or start.group("namespace") != CURRENT_NAMESPACE
        or end.group("namespace") != CURRENT_NAMESPACE
    ):
        raise RuntimeError("template AGENTS.md has invalid managed markers")
    return full[start.start() : end.end()]


def merge_agents(existing: str | None, block: str) -> tuple[str | None, str | None]:
    if existing is None:
        return f"# Project Agent Guide\n\n{block}\n", None
    starts = list(START_RE.finditer(existing))
    ends = list(END_RE.finditer(existing))
    if not starts and not ends:
        separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        return f"{existing}{separator}{block}\n", None
    if len(starts) != 1 or len(ends) != 1 or ends[0].start() < starts[0].end():
        return None, "managed markers are missing, duplicated, or out of order"
    start_namespace = starts[0].group("namespace")
    end_namespace = ends[0].group("namespace")
    if start_namespace != end_namespace:
        return None, "managed marker namespaces do not match"
    if not supported_marker_namespace(start_namespace):
        return None, "managed marker namespace is not recognized"
    end = ends[0].end()
    return f"{existing[:starts[0].start()]}{block}{existing[end:]}", None


def plan_managed_file(
    actions: list[Action],
    root: Path,
    rel: str,
    desired: str,
    prior_hashes: dict[str, str],
    reason: str,
) -> None:
    conflict = path_conflict(root, rel)
    if conflict:
        actions.append(Action("conflict", rel, conflict))
        return
    current = read_text(root / rel)
    if current is None:
        actions.append(Action("create", rel, reason, desired))
    elif current == desired:
        actions.append(Action("leave unchanged", rel, "content is current"))
    elif prior_hashes.get(rel) == sha256(current):
        actions.append(Action("migrate", rel, f"update an unmodified {reason}", desired))
    else:
        actions.append(Action("conflict", rel, "existing content is user-authored or changed since initialization"))


def plan(
    root: Path,
    asset_root: Path,
    tracker: str,
    skills_mode: str,
    skills_root: Path,
    metadata_root: Path,
) -> tuple[list[Action], dict]:
    actions: list[Action] = []
    state_error = path_conflict(root, STATE_PATH)
    state = {"_error": state_error} if state_error else load_state(root)
    if "_error" in state:
        actions.append(Action("conflict", str(STATE_PATH), state["_error"]))
        prior_hashes: dict[str, str] = {}
    else:
        prior_hashes = state.get("managed_files", {}) if isinstance(state.get("managed_files", {}), dict) else {}

    block = managed_template(asset_root)
    agents_path = root / "AGENTS.md"
    agents_error = path_conflict(root, "AGENTS.md")
    existing_agents = None if agents_error else read_text(agents_path)
    merged, error = merge_agents(existing_agents, block) if not agents_error else (None, agents_error)
    if error:
        actions.append(Action("conflict", "AGENTS.md", error))
    elif merged == existing_agents:
        actions.append(Action("leave unchanged", "AGENTS.md", "managed block is current"))
    else:
        kind = "create" if existing_agents is None else "update managed block"
        actions.append(Action(kind, "AGENTS.md", "install the versioned workflow block", merged))

    doc_assets = {
        "docs/agents/domain.md": "docs/agents/domain.md",
        "docs/agents/issue-tracker.md": f"docs/agents/issue-tracker-{tracker}.md" if tracker != "local" else "docs/agents/issue-tracker.md",
        "docs/agents/triage-labels.md": "docs/agents/triage-labels.md",
    }
    for rel, asset_rel in doc_assets.items():
        asset = asset_root / asset_rel
        if not asset.exists():
            actions.append(Action("conflict", rel, f"tracker template is unavailable: {tracker}"))
            continue
        desired = asset.read_text(encoding="utf-8")
        plan_managed_file(actions, root, rel, desired, prior_hashes, "project workflow document")

    if skills_mode == "project":
        try:
            desired_vendor_files = vendored_files(skills_root, metadata_root)
        except RuntimeError as exc:
            actions.append(Action("conflict", VENDORED_SKILLS_PATH.as_posix(), str(exc)))
        else:
            for rel, desired in desired_vendor_files.items():
                plan_managed_file(actions, root, rel, desired, prior_hashes, "vendored VibeForge Lite file")

    directories = ["docs/adr/.gitkeep"]
    if tracker == "local":
        directories.insert(0, ".scratch/.gitkeep")
    for rel in directories:
        conflict = path_conflict(root, rel)
        if conflict:
            actions.append(Action("conflict", rel, conflict))
            continue
        if (root / rel).exists():
            actions.append(Action("leave unchanged", rel, "path already exists"))
        else:
            actions.append(Action("create", rel, "materialize the workflow directory", ""))

    scans = {
        ".codex/skills": "legacy project skill path; migrate deliberately to .agents/skills",
        "CLAUDE.md": "another agent's instructions detected; left untouched",
    }
    if skills_mode == "plugin" and (root / VENDORED_SKILLS_PATH).exists():
        scans[VENDORED_SKILLS_PATH.as_posix()] = "project-local skills detected; plugin mode leaves them untouched"
    for rel, reason in scans.items():
        if (root / rel).exists():
            actions.append(Action("leave unchanged", rel, reason))

    next_hashes = dict(prior_hashes)
    for action in actions:
        is_managed = action.path.startswith(("docs/agents/", ".agents/skills/", ".agents/vibeforge-lite/"))
        if action.content is not None and is_managed:
            next_hashes[action.path] = sha256(action.content)
        elif action.kind == "leave unchanged" and is_managed:
            current = read_text(root / action.path)
            if current is not None:
                next_hashes[action.path] = sha256(current)

    next_state = {
        "schema_version": 2,
        "template_version": TEMPLATE_VERSION,
        "tracker": tracker,
        "skills_mode": skills_mode,
        "managed_files": dict(sorted(next_hashes.items())),
    }
    desired_state = json.dumps(next_state, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    current_state = None if path_conflict(root, STATE_PATH) else read_text(root / STATE_PATH)
    if any(action.kind == "conflict" for action in actions):
        if not any(action.path == str(STATE_PATH) and action.kind == "conflict" for action in actions):
            actions.append(Action("conflict", str(STATE_PATH), "state is not updated while unresolved conflicts remain"))
    elif current_state == desired_state:
        actions.append(Action("leave unchanged", str(STATE_PATH), "template state is current"))
    else:
        kind = "create" if current_state is None else "migrate"
        actions.append(Action(kind, str(STATE_PATH), "record applied template version and managed hashes", desired_state))
    return actions, next_state


def apply_actions(root: Path, actions: list[Action]) -> int:
    conflicts = [action for action in actions if action.kind == "conflict"]
    if conflicts:
        return 2
    for action in actions:
        if action.content is not None and action.kind in {"create", "migrate", "update managed block"}:
            atomic_write(root, action.path, action.content)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path.cwd(), help="target repository (default: current directory)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="apply a conflict-free plan")
    mode.add_argument("--dry-run", action="store_true", help="show the plan without writing (default)")
    parser.add_argument("--tracker", choices=("local", "github", "gitlab"), default="local")
    parser.add_argument(
        "--skills",
        choices=("project", "plugin"),
        default="project",
        help="install project-local skills (default) or rely on an installed Plugin",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.target.expanduser().resolve()
    if not root.is_dir():
        print(f"target is not a directory: {root}", file=sys.stderr)
        return 2
    asset_root = Path(__file__).resolve().parents[1] / "assets" / "project"
    try:
        skills_root, metadata_root = distribution_layout()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    actions, state = plan(root, asset_root, args.tracker, args.skills, skills_root, metadata_root)
    exit_code = apply_actions(root, actions) if args.apply else (2 if any(a.kind == "conflict" for a in actions) else 0)
    payload = {
        "mode": "apply" if args.apply else "dry-run",
        "target": str(root),
        "template_version": TEMPLATE_VERSION,
        "skills_mode": args.skills,
        "actions": [action.public() for action in actions],
        "state": state,
        "exit_code": exit_code,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"VibeForge Lite {TEMPLATE_VERSION} {payload['mode']}: {root}")
        for action in actions:
            print(f"- {action.kind:20} {action.path}: {action.reason}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
