#!/usr/bin/env python3
"""校验顺序实施 Ticket 的 Tracker、提交前门禁和提交后门禁。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

STATUS_RE = re.compile(r"^\*\*(?:Status|状态)[:：]\*\*\s*(\S+)\s*$")
CHECK_RE = re.compile(r"^- \[([ xX])\]\s+(.+?)\s*$")
ALLOWED_STATUSES = {"ready-for-agent", "in-progress", "done"}
MAX_REPAIR_ROUNDS = 9
BLOCKING_IMPACTS = {"incorrect-result", "resource-exhaustion", "acceptance-failure"}


class GateError(RuntimeError):
    """表示门禁条件不满足。"""


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GateError(f"Git 命令失败: git {' '.join(args)}: {detail}")
    return result


def git_bytes(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GateError(f"Git 命令失败: git {' '.join(args)}: {detail}")
    return result.stdout


def repo_root(start: Path) -> Path:
    result = git(start, "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


def relative_path(repo: Path, path: Path) -> str:
    candidate = path if path.is_absolute() else repo / path
    try:
        return candidate.resolve().relative_to(repo).as_posix()
    except ValueError as exc:
        raise GateError(f"路径不在仓库内: {path}") from exc


def parse_ticket(content: str, source: str) -> tuple[str, list[tuple[bool, str]]]:
    statuses = [match.group(1) for line in content.splitlines() if (match := STATUS_RE.match(line))]
    if len(statuses) != 1:
        raise GateError(f"{source} 必须且只能包含一个 **Status:** 或 **状态：**")
    status = statuses[0]
    if status not in ALLOWED_STATUSES:
        raise GateError(f"{source} 使用了非法状态: {status}")

    criteria = [
        (match.group(1).lower() == "x", match.group(2))
        for line in content.splitlines()
        if (match := CHECK_RE.match(line))
    ]
    if not criteria:
        raise GateError(f"{source} 没有验收清单")
    texts = [text for _, text in criteria]
    if len(texts) != len(set(texts)):
        raise GateError(f"{source} 包含重复验收项，无法建立唯一证据映射")
    return status, criteria


def require_tracker_complete(content: str, source: str) -> list[str]:
    status, criteria = parse_ticket(content, source)
    if status != "done":
        raise GateError(f"{source} 状态必须为 done，当前为 {status}")
    unchecked = [text for checked, text in criteria if not checked]
    if unchecked:
        raise GateError(f"{source} 仍有未勾选验收项: {unchecked[0]}")
    return [text for _, text in criteria]


def load_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"无法读取状态文件 {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise GateError("状态文件根节点必须是对象")
    return state


def require_passed_record(gate: dict[str, Any], name: str) -> None:
    record = gate.get(name)
    if not isinstance(record, dict):
        raise GateError(f"ticket_gate.{name} 必须是对象")
    if record.get("status") != "passed" or not str(record.get("evidence", "")).strip():
        raise GateError(f"ticket_gate.{name} 必须为 passed 且包含 evidence")


def validate_review_and_repairs(
    gate: dict[str, Any],
    repair_round: int,
    implementer: str,
    final_result: str,
) -> None:
    review_history = gate.get("review_history")
    if not isinstance(review_history, list) or len(review_history) != repair_round + 1:
        raise GateError("review_history 数量必须等于 repair_round + 1")
    final_review = review_history[-1]
    if isinstance(final_review, dict) and final_result == "passed" and final_review.get("result") == "blocked":
        raise GateError("最后一次独立评审仍为 blocked，必须继续修复并重新评审，禁止进入提交前门禁")
    reviewers: set[str] = set()
    for index, review in enumerate(review_history):
        if not isinstance(review, dict):
            raise GateError("review_history 每项必须是对象")
        reviewer = str(review.get("reviewer", "")).strip()
        result = review.get("result")
        if not reviewer or reviewer == implementer:
            raise GateError("每轮评审者必须存在且不同于实现者")
        if reviewer in reviewers:
            raise GateError("每轮独立复审必须使用新的 Reviewer")
        reviewers.add(reviewer)
        expected = final_result if index == len(review_history) - 1 else "blocked"
        if result != expected or not str(review.get("evidence", "")).strip():
            raise GateError(f"第 {index + 1} 次评审必须为 {expected} 且包含证据")

    repair_history = gate.get("repair_history")
    if not isinstance(repair_history, list) or len(repair_history) != repair_round:
        raise GateError("repair_history 数量必须等于 repair_round")
    for index, repair in enumerate(repair_history, start=1):
        if not isinstance(repair, dict) or repair.get("round") != index:
            raise GateError("repair_history 轮次必须从 1 连续编号")
        if not str(repair.get("agent", "")).strip() or not str(repair.get("evidence", "")).strip():
            raise GateError(f"第 {index} 轮修复必须包含 agent 和 evidence")
    repair_agents = {str(repair["agent"]).strip() for repair in repair_history}
    if reviewers & repair_agents:
        raise GateError("Reviewer 不得同时充当任一轮修复 Agent")


def validate_evidence(
    repo: Path,
    state: dict[str, Any],
    ticket: str,
    criteria: list[str],
) -> dict[str, Any]:
    required = {
        "spec",
        "run_id",
        "original_head",
        "current_ticket",
        "ticket_review_base",
        "phase",
        "repair_round",
        "snapshot_dir",
        "preexisting_staged_patch_sha256",
        "ticket_gate",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise GateError(f"状态文件缺少字段: {', '.join(missing)}")
    if not str(state["run_id"]).strip():
        raise GateError("run_id 不能为空")
    spec = repo / relative_path(repo, Path(str(state["spec"])))
    if not spec.is_file():
        raise GateError("状态文件引用的 Spec 不存在")
    git(repo, "rev-parse", "--verify", f"{state['original_head']}^{{commit}}")
    if relative_path(repo, Path(str(state["current_ticket"]))) != ticket:
        raise GateError("状态文件 current_ticket 与门禁 Ticket 不一致")
    if not Path(str(state["snapshot_dir"])).is_dir():
        raise GateError("状态文件 snapshot_dir 不存在，无法证明工作区归属")
    staged_patch_hash = str(state["preexisting_staged_patch_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", staged_patch_hash):
        raise GateError("preexisting_staged_patch_sha256 必须是 SHA-256")

    repair_round = state["repair_round"]
    if (
        not isinstance(repair_round, int)
        or isinstance(repair_round, bool)
        or not 0 <= repair_round <= MAX_REPAIR_ROUNDS
    ):
        raise GateError(f"repair_round 必须是 0 到 {MAX_REPAIR_ROUNDS} 的整数")

    gate = state["ticket_gate"]
    if not isinstance(gate, dict):
        raise GateError("ticket_gate 必须是对象")
    implementer = str(gate.get("implementer", "")).strip()
    if not implementer:
        raise GateError("ticket_gate.implementer 不能为空")
    for name in ("implementation_audit", "ownership_check", "diff_inspection"):
        require_passed_record(gate, name)

    acceptance = gate.get("acceptance")
    if not isinstance(acceptance, list):
        raise GateError("ticket_gate.acceptance 必须是数组")
    acceptance_map: dict[str, dict[str, Any]] = {}
    for item in acceptance:
        if not isinstance(item, dict) or not str(item.get("criterion", "")).strip():
            raise GateError("每条 acceptance 必须包含 criterion")
        criterion = str(item["criterion"]).strip()
        if criterion in acceptance_map:
            raise GateError(f"acceptance 重复记录验收项: {criterion}")
        acceptance_map[criterion] = item
    if set(acceptance_map) != set(criteria):
        raise GateError("acceptance 必须与 Ticket 验收清单逐条精确对应")
    for criterion in criteria:
        item = acceptance_map[criterion]
        if item.get("status") != "passed" or not str(item.get("evidence", "")).strip():
            raise GateError(f"验收项缺少 passed 证据: {criterion}")

    verification = gate.get("verification")
    if not isinstance(verification, list) or not verification:
        raise GateError("ticket_gate.verification 至少需要一条验证记录")
    for item in verification:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            raise GateError("每条 verification 必须包含 name")
        required_check = item.get("required")
        if not isinstance(required_check, bool):
            raise GateError("verification.required 必须是布尔值")
        status = item.get("status")
        evidence = str(item.get("evidence", "")).strip()
        if required_check and (status != "passed" or not evidence):
            raise GateError(f"必需验证未通过或缺少证据: {item['name']}")
        if not required_check and status not in {"passed", "skipped"}:
            raise GateError(f"非必需验证状态非法: {item['name']}")
        if not evidence:
            raise GateError(f"验证记录缺少 evidence: {item['name']}")

    validate_review_and_repairs(gate, repair_round, implementer, "passed")

    path_fields = (("preexisting_paths", True), ("owned_paths", False), ("staged_paths", False))
    for name, allow_empty in path_fields:
        paths = gate.get(name)
        if not isinstance(paths, list) or (not allow_empty and not paths):
            raise GateError(f"ticket_gate.{name} 必须是路径数组")
        normalized: list[str] = []
        for raw_path in paths:
            path = Path(str(raw_path))
            if path.is_absolute() or ".." in path.parts:
                raise GateError(f"{name} 只能包含仓库相对路径")
            normalized.append(path.as_posix())
        if len(normalized) != len(set(normalized)):
            raise GateError(f"{name} 不得包含重复路径")
        gate[name] = sorted(normalized)
    if ticket not in gate["owned_paths"] or ticket not in gate["staged_paths"]:
        raise GateError("owned_paths 和 staged_paths 都必须包含 Ticket 文件")
    if gate["owned_paths"] != gate["staged_paths"]:
        raise GateError("owned_paths 必须与 staged_paths 完全一致，禁止遗漏 Ticket 改动")
    return gate


def staged_paths(repo: Path) -> list[str]:
    output = git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMRD").stdout
    return sorted(line for line in output.splitlines() if line)


def commit_paths(repo: Path, commit: str) -> list[str]:
    output = git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).stdout
    return sorted(line for line in output.splitlines() if line)


def effective_changed_paths(repo: Path, review_base: str) -> set[str]:
    tracked = git(repo, "diff", "--name-only", review_base, "--").stdout.splitlines()
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").stdout.splitlines()
    return {path for path in [*tracked, *untracked] if path}


def current_staged_patch_sha256(repo: Path) -> str:
    patch = git_bytes(repo, "diff", "--cached", "--binary", "--full-index")
    return hashlib.sha256(patch).hexdigest()


def ensure_state_untracked(repo: Path, state_path: str) -> None:
    tracked = git(repo, "ls-files", "--error-unmatch", "--", state_path, check=False)
    if tracked.returncode == 0:
        raise GateError("执行状态文件不得被 Git 跟踪")
    if state_path in staged_paths(repo):
        raise GateError("执行状态文件不得进入暂存区")


def ticket_number(path: str) -> int:
    match = re.match(r"^(\d+)-", Path(path).name)
    if not match:
        raise GateError(f"Ticket 文件名缺少数字前缀: {path}")
    return int(match.group(1))


def validate_skip(repo: Path, ticket: str, state_path: str, state: dict[str, Any]) -> None:
    if (
        state.get("phase") != "repair-exhausted"
        or state.get("repair_round") != MAX_REPAIR_ROUNDS
    ):
        raise GateError(
            "skip 门禁要求 phase=repair-exhausted 且 "
            f"repair_round={MAX_REPAIR_ROUNDS}"
        )
    if relative_path(repo, Path(str(state.get("current_ticket", "")))) != ticket:
        raise GateError("状态文件 current_ticket 与跳过 Ticket 不一致")
    if not Path(str(state.get("snapshot_dir", ""))).is_dir():
        raise GateError("snapshot_dir 不存在，禁止跳过")
    ensure_state_untracked(repo, state_path)

    status, _ = parse_ticket((repo / ticket).read_text(encoding="utf-8"), ticket)
    if status != "in-progress":
        raise GateError("修复耗尽 Ticket 必须保持 in-progress")
    gate = state.get("ticket_gate")
    if not isinstance(gate, dict):
        raise GateError("ticket_gate 必须是对象")
    implementer = str(gate.get("implementer", "")).strip()
    if not implementer:
        raise GateError("ticket_gate.implementer 不能为空")
    for name in ("implementation_audit", "ownership_check", "diff_inspection"):
        require_passed_record(gate, name)
    validate_review_and_repairs(gate, MAX_REPAIR_ROUNDS, implementer, "blocked")

    findings = state.get("blocking_findings")
    if not isinstance(findings, list) or not findings:
        raise GateError("修复耗尽必须记录 blocking_findings")
    for finding in findings:
        if not isinstance(finding, dict):
            raise GateError("blocking_findings 每项必须是对象")
        required = ("severity", "issue", "impact", "affected_capabilities", "evidence")
        if any(not finding.get(field) for field in required):
            raise GateError("每条 blocking finding 必须包含严重度、问题、影响分类、影响能力和证据")
        if finding["impact"] not in BLOCKING_IMPACTS:
            raise GateError("blocking finding 的 impact 必须是错误结果、资源耗尽或验收失败")

    archive = state.get("repair_exhausted_archive")
    if not isinstance(archive, dict):
        raise GateError("修复耗尽必须记录 repair_exhausted_archive")
    for field, expected_type in (
        ("patch", "file"),
        ("untracked_backup", "dir"),
        ("recovery_instructions", "file"),
    ):
        path = Path(str(archive.get(field, "")))
        valid = path.is_file() if expected_type == "file" else path.is_dir()
        if not valid:
            raise GateError(f"修复耗尽封存缺少 {field}")
    isolation = state.get("workspace_isolation")
    if not isinstance(isolation, dict) or isolation.get("status") != "passed":
        raise GateError("workspace_isolation 必须为 passed")
    if not str(isolation.get("evidence", "")).strip():
        raise GateError("workspace_isolation 必须包含 evidence")

    preexisting = gate.get("preexisting_paths")
    if not isinstance(preexisting, list):
        raise GateError("ticket_gate.preexisting_paths 必须是数组")
    review_base = git(repo, "rev-parse", str(state.get("ticket_review_base", ""))).stdout.strip()
    actual_new = effective_changed_paths(repo, review_base) - set(preexisting) - {state_path}
    if actual_new != {ticket}:
        raise GateError(f"失败改动未完全隔离，当前额外变化: {sorted(actual_new)}")

    current_number = ticket_number(ticket)
    later_tickets: list[str] = []
    for path in sorted((repo / ticket).parent.glob("*.md")):
        relative = path.relative_to(repo).as_posix()
        if ticket_number(relative) <= current_number:
            continue
        later_status, _ = parse_ticket(path.read_text(encoding="utf-8"), relative)
        if later_status != "done":
            later_tickets.append(relative)
    assessments = state.get("candidate_assessments")
    if not isinstance(assessments, list):
        raise GateError("candidate_assessments 必须是数组")
    assessed_tickets = [item.get("ticket") for item in assessments if isinstance(item, dict)]
    if assessed_tickets != later_tickets:
        raise GateError("candidate_assessments 必须按顺序覆盖全部后续未完成 Ticket")
    eligible: list[str] = []
    for item in assessments:
        if not isinstance(item, dict) or not str(item.get("evidence", "")).strip():
            raise GateError("每个候选 Ticket 必须包含影响判断证据")
        dependency = item.get("depends_on_exhausted")
        impact = item.get("impact")
        declared_eligible = item.get("eligible")
        if not isinstance(dependency, bool) or impact not in {"affected", "unaffected", "unknown"}:
            raise GateError("候选 Ticket 的依赖或影响字段非法")
        expected_eligible = not dependency and impact == "unaffected"
        if declared_eligible is not expected_eligible:
            raise GateError("候选 Ticket 的 eligible 与依赖/影响判断不一致")
        if expected_eligible:
            eligible.append(str(item["ticket"]))
    expected_next = eligible[0] if eligible else None
    if state.get("next_ticket") != expected_next:
        raise GateError("next_ticket 必须是编号最小的可安全继续 Ticket，或为 null")


def validate_pre_commit(repo: Path, ticket: str, state_path: str, state: dict[str, Any]) -> None:
    if state.get("phase") != "ready-to-commit":
        raise GateError("提交前状态 phase 必须是 ready-to-commit")
    index_content = git(repo, "show", f":{ticket}").stdout
    criteria = require_tracker_complete(index_content, f"暂存区中的 {ticket}")
    gate = validate_evidence(repo, state, ticket, criteria)
    ensure_state_untracked(repo, state_path)

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    review_base = git(repo, "rev-parse", str(state["ticket_review_base"])).stdout.strip()
    if head != review_base:
        raise GateError("提交前 HEAD 必须等于固定 ticket-review-base，检测到未授权 Commit")
    actual_staged = staged_paths(repo)
    if actual_staged != gate["staged_paths"]:
        raise GateError(f"暂存路径与 staged_paths 不一致: {actual_staged}")
    if not actual_staged:
        raise GateError("禁止创建空 Ticket Commit")
    changed = effective_changed_paths(repo, review_base)
    preexisting = set(gate["preexisting_paths"])
    expected_new = set(gate["owned_paths"]) - preexisting
    actual_new = changed - preexisting - {state_path}
    if actual_new != expected_new:
        raise GateError(
            "Ticket 新增变化、owned_paths 与暂存范围不一致: "
            f"actual={sorted(actual_new)}, expected={sorted(expected_new)}"
        )
    for path in actual_staged:
        if git(repo, "diff", "--quiet", "--", path, check=False).returncode != 0:
            raise GateError(f"暂存后又发生未暂存修改，必须重新验收: {path}")


def validate_post_commit(
    repo: Path,
    ticket: str,
    state_path: str,
    state: dict[str, Any],
    commit_arg: str | None,
) -> None:
    if state.get("phase") != "committed":
        raise GateError("提交后状态 phase 必须是 committed")
    commit = git(repo, "rev-parse", commit_arg or "HEAD").stdout.strip()
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if head != commit:
        raise GateError("提交后门禁只允许校验当前 HEAD")
    ticket_content = git(repo, "show", f"{commit}:{ticket}").stdout
    criteria = require_tracker_complete(ticket_content, f"Commit {commit} 中的 {ticket}")
    gate = validate_evidence(repo, state, ticket, criteria)
    ensure_state_untracked(repo, state_path)

    parent = git(repo, "rev-parse", f"{commit}^").stdout.strip()
    review_base = git(repo, "rev-parse", str(state["ticket_review_base"])).stdout.strip()
    if parent != review_base:
        raise GateError("Ticket Commit 的直接父提交必须等于固定 ticket-review-base")
    actual_paths = commit_paths(repo, commit)
    if actual_paths != gate["staged_paths"]:
        raise GateError(f"Commit 路径与提交前 staged_paths 不一致: {actual_paths}")
    if gate.get("commit") != commit:
        raise GateError("ticket_gate.commit 未记录当前 Commit")
    if current_staged_patch_sha256(repo) != state["preexisting_staged_patch_sha256"]:
        raise GateError("提交后用户原有暂存补丁未被原样恢复")

    completed = state.get("completed_commits")
    if not isinstance(completed, list):
        raise GateError("completed_commits 必须是数组")
    matching = [
        item
        for item in completed
        if isinstance(item, dict) and item.get("ticket") == Path(ticket).stem.split("-", 1)[0]
    ]
    if len(matching) != 1:
        raise GateError("completed_commits 必须且只能记录一次当前 Ticket")
    record = matching[0]
    if (
        record.get("commit") != commit
        or record.get("repair_rounds") != state["repair_round"]
        or record.get("review") != "passed"
    ):
        raise GateError("completed_commits 的 Commit、修复轮数或评审结论不一致")

    results = state.get("ticket_results")
    if not isinstance(results, list):
        raise GateError("ticket_results 必须是数组")
    result_matches = [
        item
        for item in results
        if isinstance(item, dict) and item.get("ticket") == Path(ticket).stem.split("-", 1)[0]
    ]
    if len(result_matches) != 1:
        raise GateError("ticket_results 必须且只能冻结一次当前 Ticket")
    result_record = result_matches[0]
    if (
        result_record.get("commit") != commit
        or result_record.get("repair_rounds") != state["repair_round"]
        or result_record.get("review") != "passed"
        or result_record.get("ticket_gate") != gate
    ):
        raise GateError("ticket_results 未完整冻结当前 Ticket 的门禁证据")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("tracker", "pre-commit", "post-commit", "skip"),
        required=True,
    )
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--commit")
    args = parser.parse_args()

    try:
        repo = repo_root(Path.cwd())
        ticket = relative_path(repo, args.ticket)
        ticket_path = repo / ticket
        if args.phase == "tracker":
            require_tracker_complete(ticket_path.read_text(encoding="utf-8"), ticket)
        else:
            if args.state is None:
                raise GateError("pre-commit 和 post-commit 必须提供 --state")
            state_path = relative_path(repo, args.state)
            state = load_state(repo / state_path)
            if args.phase == "pre-commit":
                validate_pre_commit(repo, ticket, state_path, state)
            elif args.phase == "skip":
                validate_skip(repo, ticket, state_path, state)
            else:
                validate_post_commit(repo, ticket, state_path, state, args.commit)
    except (GateError, OSError) as exc:
        print(f"门禁失败: {exc}", file=sys.stderr)
        return 1

    print(f"门禁通过: {args.phase} {ticket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
