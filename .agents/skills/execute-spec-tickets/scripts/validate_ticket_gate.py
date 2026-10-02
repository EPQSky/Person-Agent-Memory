#!/usr/bin/env python3
"""校验并行实施 Ticket 的 Tracker、提交、合并和跳过门禁。"""

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
MIN_STALLED_REPAIR_ROUNDS = 3
BLOCKING_IMPACTS = {"incorrect-result", "resource-exhaustion", "acceptance-failure"}
MAX_PARALLEL_TICKETS = 3


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


def ticket_state(state: dict[str, Any], ticket: str) -> dict[str, Any]:
    """返回 Ticket 独立状态，兼容旧版单 Ticket 状态文件。"""
    runs = state.get("ticket_runs")
    if runs is None:
        return state
    if not isinstance(runs, dict):
        raise GateError("ticket_runs 必须是对象")
    record = runs.get(ticket)
    if not isinstance(record, dict):
        raise GateError(f"ticket_runs 缺少当前 Ticket: {ticket}")
    return record


def require_spec_runtime_layout(
    repo: Path,
    state: dict[str, Any],
    ticket: str,
    gate: dict[str, Any],
    *,
    expect_ticket_worktree: bool = True,
) -> None:
    """确保恢复材料和 worktree 持久留在当前 Spec 目录内。"""
    required = {
        "execution_root",
        "snapshot_dir",
        "worktrees_root",
        "integration_worktree",
        "integration_branch",
        "max_parallel_tickets",
        "active_tickets",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise GateError(f"状态文件缺少并行执行字段: {', '.join(missing)}")
    integration_branch = str(state["integration_branch"]).strip()
    if not integration_branch or integration_branch.startswith("-"):
        raise GateError("integration_branch 必须是执行开始时的有效分支名")
    if git(repo, "check-ref-format", "--branch", integration_branch, check=False).returncode != 0:
        raise GateError(f"integration_branch 不是有效分支名: {integration_branch}")
    if state["max_parallel_tickets"] != MAX_PARALLEL_TICKETS:
        raise GateError("max_parallel_tickets 必须为 3")
    active = state["active_tickets"]
    if not isinstance(active, list) or len(active) > MAX_PARALLEL_TICKETS:
        raise GateError("active_tickets 必须是最多 3 张 Ticket 的数组")
    if len(active) != len(set(active)):
        raise GateError("active_tickets 不得重复")

    integration_worktree = Path(str(state["integration_worktree"])).resolve()
    if not integration_worktree.is_dir():
        raise GateError("integration_worktree 不存在")
    current_integration_branch = git(integration_worktree, "branch", "--show-current").stdout.strip()
    if current_integration_branch != integration_branch:
        raise GateError("integration_worktree 当前分支必须等于 integration_branch")
    if git(integration_worktree, "show-ref", "--verify", f"refs/heads/{integration_branch}", check=False).returncode != 0:
        raise GateError(f"integration_branch 不存在: {integration_branch}")
    spec_path = Path(str(state["spec"]))
    spec = (integration_worktree / spec_path if not spec_path.is_absolute() else spec_path).resolve()
    if not spec.is_file():
        raise GateError("状态文件引用的 Spec 不存在")
    spec_dir = spec.parent.resolve()

    def inside_spec(raw: Any, field: str) -> Path:
        path = Path(str(raw))
        resolved = (integration_worktree / path if not path.is_absolute() else path).resolve()
        try:
            resolved.relative_to(spec_dir)
        except ValueError as exc:
            raise GateError(f"{field} 必须位于 Spec 目录内: {raw}") from exc
        if path.is_absolute() and ("/tmp/" in f"{resolved}/" or resolved == Path("/tmp")):
            raise GateError(f"{field} 不得位于 /tmp 或系统临时目录: {raw}")
        return resolved

    execution_root = inside_spec(state["execution_root"], "execution_root")
    snapshot_dir = inside_spec(state["snapshot_dir"], "snapshot_dir")
    worktrees_root = inside_spec(state["worktrees_root"], "worktrees_root")
    if not execution_root.is_dir():
        raise GateError("execution_root 不存在")
    if not snapshot_dir.is_dir():
        raise GateError("snapshot_dir 不存在，无法证明工作区归属")
    if not worktrees_root.is_dir():
        raise GateError("worktrees_root 不存在，无法证明 Ticket 隔离")

    worktree = gate.get("worktree")
    if not isinstance(worktree, dict):
        raise GateError("ticket_gate.worktree 必须是对象")
    worktree_path = inside_spec(worktree.get("path"), "ticket worktree")
    try:
        worktree_path.relative_to(worktrees_root)
    except ValueError as exc:
        raise GateError("Ticket worktree 必须位于 worktrees_root 下") from exc
    if not worktree_path.is_dir():
        raise GateError("Ticket worktree 不存在")
    if expect_ticket_worktree and state.get("ticket_runs") is not None and repo.resolve() != worktree_path:
        raise GateError("当前 Git 工作区必须是当前 Ticket 的独占 worktree")
    branch = str(worktree.get("branch", "")).strip()
    if not branch:
        raise GateError("Ticket worktree.branch 不能为空")
    base_branch = str(worktree.get("base_branch", "")).strip()
    if base_branch != integration_branch:
        raise GateError("Ticket worktree.base_branch 必须等于创建 worktree 时的 integration_branch")
    base_commit = str(worktree.get("base_commit", "")).strip()
    if not base_commit:
        raise GateError("Ticket worktree.base_commit 不能为空")
    git(repo, "rev-parse", "--verify", f"{base_commit}^{{commit}}")
    branch_ref = git(repo, "show-ref", "--verify", f"refs/heads/{branch}", check=False)
    if branch_ref.returncode != 0:
        raise GateError(f"Ticket worktree 分支不存在: {branch}")
    if ticket not in active and state.get("ticket_runs") is not None:
        raise GateError("活动 Ticket 必须出现在 active_tickets")


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
    finding_definitions: dict[str, tuple[str, str]] = {}
    review_finding_batches: list[list[str]] = []
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
        findings = review.get("blocking_findings")
        if not isinstance(findings, list):
            raise GateError(f"第 {index + 1} 次评审必须包含 blocking_findings 数组")
        if result == "blocked" and not findings:
            raise GateError(f"第 {index + 1} 次 blocked 评审必须记录完整阻断问题批次")
        if result == "passed" and findings:
            raise GateError(f"第 {index + 1} 次 passed 评审不得包含 blocking_findings")
        batch_ids: list[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                raise GateError("review_history.blocking_findings 每项必须是对象")
            finding_id = str(finding.get("id", "")).strip()
            required = ("severity", "issue", "impact", "evidence")
            if not finding_id or any(not str(finding.get(field, "")).strip() for field in required):
                raise GateError("每条 Review Finding 必须包含 ID、严重度、问题、影响分类和证据")
            if finding["impact"] not in BLOCKING_IMPACTS:
                raise GateError("Review Finding 的 impact 必须是错误结果、资源耗尽或验收失败")
            definition = (str(finding["issue"]).strip(), str(finding["impact"]).strip())
            if finding_id in batch_ids:
                raise GateError(f"Review Finding ID 在同一批次内不得重复: {finding_id}")
            previous_definition = finding_definitions.get(finding_id)
            if previous_definition is not None and previous_definition != definition:
                raise GateError(f"跨轮复用 Review Finding ID 时不得改变问题定义: {finding_id}")
            finding_definitions[finding_id] = definition
            batch_ids.append(finding_id)
        review_finding_batches.append(batch_ids)

    repair_history = gate.get("repair_history")
    if not isinstance(repair_history, list) or len(repair_history) != repair_round:
        raise GateError("repair_history 数量必须等于 repair_round")
    for index, repair in enumerate(repair_history, start=1):
        if not isinstance(repair, dict) or repair.get("round") != index:
            raise GateError("repair_history 轮次必须从 1 连续编号")
        if not str(repair.get("agent", "")).strip() or not str(repair.get("evidence", "")).strip():
            raise GateError(f"第 {index} 轮修复必须包含 agent 和 evidence")
        finding_ids = repair.get("finding_ids")
        if not isinstance(finding_ids, list) or any(not str(item).strip() for item in finding_ids):
            raise GateError(f"第 {index} 轮修复必须包含 finding_ids")
        normalized_ids = [str(item).strip() for item in finding_ids]
        if len(normalized_ids) != len(set(normalized_ids)):
            raise GateError(f"第 {index} 轮修复的 finding_ids 不得重复")
        expected_ids = review_finding_batches[index - 1]
        if set(normalized_ids) != set(expected_ids):
            raise GateError(f"第 {index} 轮修复必须精确覆盖上一轮 Review 的全部 Findings")
        resolutions = repair.get("resolutions")
        if not isinstance(resolutions, list):
            raise GateError(f"第 {index} 轮修复必须包含 resolutions 数组")
        resolution_ids: list[str] = []
        for resolution in resolutions:
            if not isinstance(resolution, dict):
                raise GateError("repair_history.resolutions 每项必须是对象")
            finding_id = str(resolution.get("finding_id", "")).strip()
            if (
                not finding_id
                or resolution.get("status") != "fixed"
                or not str(resolution.get("evidence", "")).strip()
            ):
                raise GateError("每条 Finding resolution 必须标记 fixed 并包含 evidence")
            resolution_ids.append(finding_id)
        if len(resolution_ids) != len(set(resolution_ids)) or set(resolution_ids) != set(expected_ids):
            raise GateError(f"第 {index} 轮 resolutions 必须逐项覆盖上一轮全部 Findings")
    repair_agents = {str(repair["agent"]).strip() for repair in repair_history}
    if reviewers & repair_agents:
        raise GateError("Reviewer 不得同时充当任一轮修复 Agent")


def validate_evidence(
    repo: Path,
    state: dict[str, Any],
    ticket: str,
    criteria: list[str],
    *,
    expect_ticket_worktree: bool = True,
) -> dict[str, Any]:
    ticket_record = ticket_state(state, ticket)
    required = {
        "spec",
        "run_id",
        "original_head",
        "integration_worktree",
        "preexisting_staged_patch_sha256",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise GateError(f"状态文件缺少字段: {', '.join(missing)}")
    if not str(state["run_id"]).strip():
        raise GateError("run_id 不能为空")
    integration_worktree = Path(str(state["integration_worktree"])).resolve()
    spec_path = Path(str(state["spec"]))
    spec = (integration_worktree / spec_path if not spec_path.is_absolute() else spec_path).resolve()
    if not spec.is_file():
        raise GateError("状态文件引用的 Spec 不存在")
    git(repo, "rev-parse", "--verify", f"{state['original_head']}^{{commit}}")
    if state.get("ticket_runs") is None and relative_path(
        repo, Path(str(state.get("current_ticket", "")))
    ) != ticket:
        raise GateError("状态文件 current_ticket 与门禁 Ticket 不一致")
    gate = ticket_record.get("ticket_gate")
    if not isinstance(gate, dict):
        raise GateError("ticket_gate 必须是对象")
    require_spec_runtime_layout(
        repo,
        state,
        ticket,
        gate,
        expect_ticket_worktree=expect_ticket_worktree,
    )
    staged_patch_hash = str(state["preexisting_staged_patch_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", staged_patch_hash):
        raise GateError("preexisting_staged_patch_sha256 必须是 SHA-256")

    repair_round = ticket_record.get("repair_round")
    if not isinstance(repair_round, int) or isinstance(repair_round, bool) or repair_round < 0:
        raise GateError("repair_round 必须是非负整数")

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


def state_path_for_repo(repo: Path, state_path: str) -> str | None:
    try:
        return relative_path(repo, Path(state_path))
    except GateError:
        return None


def exclude_execution_runtime(
    repo: Path,
    state: dict[str, Any],
    paths: set[str],
) -> set[str]:
    try:
        runtime = relative_path(repo, Path(str(state["execution_root"]))).rstrip("/") + "/"
    except GateError:
        return paths
    return {
        path
        for path in paths
        if path != runtime.rstrip("/") and not path.startswith(runtime)
    }


def current_staged_patch_sha256(repo: Path) -> str:
    patch = git_bytes(repo, "diff", "--cached", "--binary", "--full-index")
    return hashlib.sha256(patch).hexdigest()


def ensure_state_untracked(
    repo: Path,
    state_path: str,
    integration_worktree: Path | None = None,
) -> None:
    check_repo = (integration_worktree or repo).resolve()
    candidate = Path(state_path)
    if candidate.is_absolute():
        try:
            state_rel = candidate.resolve().relative_to(check_repo).as_posix()
        except ValueError:
            state_rel = state_path
    else:
        state_rel = candidate.as_posix()
    tracked = git(check_repo, "ls-files", "--error-unmatch", "--", state_rel, check=False)
    if tracked.returncode == 0:
        raise GateError("执行状态文件不得被 Git 跟踪")
    if state_rel in staged_paths(check_repo):
        raise GateError("执行状态文件不得进入暂存区")


def ticket_number(path: str) -> int:
    match = re.match(r"^(\d+)-", Path(path).name)
    if not match:
        raise GateError(f"Ticket 文件名缺少数字前缀: {path}")
    return int(match.group(1))


def validate_skip(repo: Path, ticket: str, state_path: str, state: dict[str, Any]) -> None:
    ticket_record = ticket_state(state, ticket)
    repair_round = ticket_record.get("repair_round")
    if (
        ticket_record.get("phase") != "repair-stalled"
        or not isinstance(repair_round, int)
        or isinstance(repair_round, bool)
        or repair_round < MIN_STALLED_REPAIR_ROUNDS
    ):
        raise GateError("skip 门禁要求 phase=repair-stalled 且至少已有三轮修复")
    if state.get("ticket_runs") is None and relative_path(
        repo, Path(str(state.get("current_ticket", "")))
    ) != ticket:
        raise GateError("状态文件 current_ticket 与跳过 Ticket 不一致")
    ensure_state_untracked(repo, state_path, Path(str(state["integration_worktree"])))

    status, _ = parse_ticket((repo / ticket).read_text(encoding="utf-8"), ticket)
    if status != "in-progress":
        raise GateError("修复停滞 Ticket 必须保持 in-progress")
    gate = ticket_record.get("ticket_gate")
    if not isinstance(gate, dict):
        raise GateError("ticket_gate 必须是对象")
    require_spec_runtime_layout(repo, state, ticket, gate)
    implementer = str(gate.get("implementer", "")).strip()
    if not implementer:
        raise GateError("ticket_gate.implementer 不能为空")
    for name in ("implementation_audit", "ownership_check", "diff_inspection"):
        require_passed_record(gate, name)
    validate_review_and_repairs(gate, repair_round, implementer, "blocked")

    recent_reviews = gate["review_history"][-(MIN_STALLED_REPAIR_ROUNDS + 1) :]
    recent_batches = [
        {str(finding["id"]).strip() for finding in review["blocking_findings"]}
        for review in recent_reviews
    ]
    if len(recent_batches) != MIN_STALLED_REPAIR_ROUNDS + 1 or any(
        batch != recent_batches[0] for batch in recent_batches[1:]
    ):
        raise GateError("repair-stalled 要求连续三轮修复前后的阻断 Finding 集合完全相同")
    stall_assessments = ticket_record.get("stall_assessments", state.get("stall_assessments"))
    if not isinstance(stall_assessments, list) or len(stall_assessments) != MIN_STALLED_REPAIR_ROUNDS:
        raise GateError("repair-stalled 必须记录最后三轮 stall_assessments")
    expected_rounds = list(range(repair_round - MIN_STALLED_REPAIR_ROUNDS + 1, repair_round + 1))
    for item, expected_round in zip(stall_assessments, expected_rounds, strict=True):
        if not isinstance(item, dict) or item.get("round") != expected_round:
            raise GateError("stall_assessments 必须对应最后三轮连续修复")
        finding_ids = item.get("finding_ids")
        if not isinstance(finding_ids, list) or {str(value).strip() for value in finding_ids} != recent_batches[0]:
            raise GateError("stall_assessments 必须引用持续未关闭的完整 Finding 集合")
        if item.get("status") != "no-progress" or not str(item.get("evidence", "")).strip():
            raise GateError("每轮 stall assessment 必须标记 no-progress 并包含证据")

    findings = ticket_record.get("blocking_findings", state.get("blocking_findings"))
    if not isinstance(findings, list) or not findings:
        raise GateError("修复停滞必须记录 blocking_findings")
    final_review_findings = gate["review_history"][-1]["blocking_findings"]
    final_review_ids = {str(item["id"]).strip() for item in final_review_findings}
    exhausted_ids: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            raise GateError("blocking_findings 每项必须是对象")
        required = ("id", "severity", "issue", "impact", "affected_capabilities", "evidence")
        if any(not finding.get(field) for field in required):
            raise GateError("每条 blocking finding 必须包含 ID、严重度、问题、影响分类、影响能力和证据")
        if finding["impact"] not in BLOCKING_IMPACTS:
            raise GateError("blocking finding 的 impact 必须是错误结果、资源耗尽或验收失败")
        finding_id = str(finding["id"]).strip()
        if finding_id in exhausted_ids:
            raise GateError(f"修复停滞 blocking_findings 不得重复 ID: {finding_id}")
        exhausted_ids.add(finding_id)
    if exhausted_ids != final_review_ids:
        raise GateError("修复停滞 blocking_findings 必须与最后一次 blocked Review 完整一致")

    archive = ticket_record.get("repair_stalled_archive", state.get("repair_stalled_archive"))
    if not isinstance(archive, dict):
        raise GateError("修复停滞必须记录 repair_stalled_archive")
    for field, expected_type in (
        ("patch", "file"),
        ("untracked_backup", "dir"),
        ("recovery_instructions", "file"),
    ):
        path = Path(str(archive.get(field, "")))
        valid = path.is_file() if expected_type == "file" else path.is_dir()
        if not valid:
            raise GateError(f"修复停滞封存缺少 {field}")
    isolation = ticket_record.get("workspace_isolation", state.get("workspace_isolation"))
    if not isinstance(isolation, dict) or isolation.get("status") != "passed":
        raise GateError("workspace_isolation 必须为 passed")
    if not str(isolation.get("evidence", "")).strip():
        raise GateError("workspace_isolation 必须包含 evidence")

    preexisting = gate.get("preexisting_paths")
    if not isinstance(preexisting, list):
        raise GateError("ticket_gate.preexisting_paths 必须是数组")
    review_base = git(repo, "rev-parse", str(ticket_record.get("ticket_review_base", ""))).stdout.strip()
    state_relative = state_path_for_repo(repo, state_path)
    excluded_paths = {state_relative} if state_relative else set()
    actual_new = exclude_execution_runtime(
        repo,
        state,
        effective_changed_paths(repo, review_base),
    ) - set(preexisting) - excluded_paths
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
    assessments = ticket_record.get("candidate_assessments", state.get("candidate_assessments"))
    if not isinstance(assessments, list):
        raise GateError("candidate_assessments 必须是数组")
    assessed_tickets = [item.get("ticket") for item in assessments if isinstance(item, dict)]
    if assessed_tickets != later_tickets:
        raise GateError("candidate_assessments 必须按顺序覆盖全部后续未完成 Ticket")
    eligible: list[str] = []
    for item in assessments:
        if not isinstance(item, dict) or not str(item.get("evidence", "")).strip():
            raise GateError("每个候选 Ticket 必须包含影响判断证据")
        dependency = item.get("depends_on_stalled")
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
    if ticket_record.get("next_ticket", state.get("next_ticket")) != expected_next:
        raise GateError("next_ticket 必须是编号最小的可安全继续 Ticket，或为 null")


def validate_pre_commit(repo: Path, ticket: str, state_path: str, state: dict[str, Any]) -> None:
    ticket_record = ticket_state(state, ticket)
    if ticket_record.get("phase") != "ready-to-commit":
        raise GateError("提交前状态 phase 必须是 ready-to-commit")
    index_content = git(repo, "show", f":{ticket}").stdout
    criteria = require_tracker_complete(index_content, f"暂存区中的 {ticket}")
    gate = validate_evidence(repo, state, ticket, criteria)
    ensure_state_untracked(repo, state_path, Path(str(state["integration_worktree"])))

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    review_base = git(repo, "rev-parse", str(ticket_record["ticket_review_base"])).stdout.strip()
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
    state_relative = state_path_for_repo(repo, state_path)
    excluded_paths = {state_relative} if state_relative else set()
    actual_new = changed - preexisting - excluded_paths
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
    ticket_record = ticket_state(state, ticket)
    if ticket_record.get("phase") not in {"committed", "ready-to-merge"}:
        raise GateError("提交后状态 phase 必须是 committed 或 ready-to-merge")
    commit = git(repo, "rev-parse", commit_arg or "HEAD").stdout.strip()
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if head != commit:
        raise GateError("提交后门禁只允许校验当前 HEAD")
    ticket_content = git(repo, "show", f"{commit}:{ticket}").stdout
    criteria = require_tracker_complete(ticket_content, f"Commit {commit} 中的 {ticket}")
    gate = validate_evidence(repo, state, ticket, criteria)
    ensure_state_untracked(repo, state_path, Path(str(state["integration_worktree"])))

    parent = git(repo, "rev-parse", f"{commit}^").stdout.strip()
    review_base = git(repo, "rev-parse", str(ticket_record["ticket_review_base"])).stdout.strip()
    if parent != review_base:
        raise GateError("Ticket Commit 的直接父提交必须等于固定 ticket-review-base")
    actual_paths = commit_paths(repo, commit)
    if actual_paths != gate["staged_paths"]:
        raise GateError(f"Commit 路径与提交前 staged_paths 不一致: {actual_paths}")
    if gate.get("commit") != commit:
        raise GateError("ticket_gate.commit 未记录当前 Commit")
    integration_worktree = Path(str(state["integration_worktree"])).resolve()
    if current_staged_patch_sha256(integration_worktree) != state["preexisting_staged_patch_sha256"]:
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
        or record.get("repair_rounds") != ticket_record["repair_round"]
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
        or result_record.get("repair_rounds") != ticket_record["repair_round"]
        or result_record.get("review") != "passed"
        or result_record.get("ticket_gate") != gate
    ):
        raise GateError("ticket_results 未完整冻结当前 Ticket 的门禁证据")


def validate_merge(
    repo: Path,
    ticket: str,
    state_path: str,
    state: dict[str, Any],
    merge_arg: str | None,
) -> None:
    ticket_record = ticket_state(state, ticket)
    if ticket_record.get("phase") != "merged":
        raise GateError("合并门禁要求 phase=merged")
    gate = ticket_record.get("ticket_gate")
    if not isinstance(gate, dict):
        raise GateError("ticket_gate 必须是对象")
    require_spec_runtime_layout(
        repo,
        state,
        ticket,
        gate,
        expect_ticket_worktree=False,
    )
    ensure_state_untracked(repo, state_path, Path(str(state["integration_worktree"])))

    integration_branch = str(state.get("integration_branch", "")).strip()
    if not integration_branch:
        raise GateError("合并门禁要求记录 integration_branch")
    branch = git(repo, "branch", "--show-current").stdout.strip()
    if branch != integration_branch:
        raise GateError("合并门禁必须在记录的 integration_branch 上执行")

    merge = ticket_record.get("merge") or gate.get("merge")
    if not isinstance(merge, dict):
        raise GateError("合并完成必须记录 merge 对象")
    required = (
        "integration_branch",
        "source_branch",
        "before_head",
        "merge_commit",
        "after_head",
        "verification",
    )
    if any(not str(merge.get(field, "")).strip() for field in required):
        raise GateError("merge 必须包含分支、前后 HEAD、合并 Commit 和验证证据")
    if merge["integration_branch"] != integration_branch:
        raise GateError("merge.integration_branch 必须等于状态中的 integration_branch")

    merge_commit = git(repo, "rev-parse", merge_arg or "HEAD").stdout.strip()
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if head != merge_commit or merge.get("merge_commit") != merge_commit:
        raise GateError("合并门禁只允许校验当前 integration_branch HEAD")
    if merge.get("after_head") != merge_commit:
        raise GateError("merge.after_head 必须等于当前 integration_branch HEAD")

    before_head = git(repo, "rev-parse", str(merge["before_head"])).stdout.strip()
    source_branch = str(merge["source_branch"]).strip()
    source_commit = git(repo, "rev-parse", f"{source_branch}^{{commit}}").stdout.strip()
    if source_commit != str(gate.get("commit", "")).strip():
        raise GateError("merge.source_branch 必须指向当前 Ticket 分支 Commit")
    if git(repo, "merge-base", "--is-ancestor", before_head, merge_commit, check=False).returncode != 0:
        raise GateError("merge.before_head 必须是合并 Commit 的祖先")
    if git(repo, "merge-base", "--is-ancestor", source_commit, merge_commit, check=False).returncode != 0:
        raise GateError("Ticket 分支 Commit 必须已经合入 integration_branch")
    parents = git(repo, "rev-list", "--parents", "-n", "1", merge_commit).stdout.split()
    if len(parents) < 3:
        raise GateError("integration_branch 上的 Ticket 合并必须产生带两个父提交的 merge commit")

    criteria = require_tracker_complete((repo / ticket).read_text(encoding="utf-8"), ticket)
    validate_evidence(
        repo,
        state,
        ticket,
        criteria,
        expect_ticket_worktree=False,
    )
    completed = state.get("completed_commits")
    if not isinstance(completed, list):
        raise GateError("completed_commits 必须是数组")
    matches = [
        item for item in completed
        if isinstance(item, dict) and item.get("ticket") == Path(ticket).stem.split("-", 1)[0]
    ]
    if len(matches) != 1 or matches[0].get("merge_commit") != merge_commit:
        raise GateError("completed_commits 必须记录当前 Ticket 的 merge_commit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("tracker", "pre-commit", "post-commit", "merge", "skip"),
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
            state_path = str(
                (args.state if args.state.is_absolute() else Path.cwd() / args.state).resolve()
            )
            state = load_state(Path(state_path))
            integration_worktree = Path(str(state.get("integration_worktree", ""))).resolve()
            try:
                Path(state_path).relative_to(integration_worktree)
            except ValueError as exc:
                raise GateError("--state 必须位于状态文件记录的 integration_worktree 内") from exc
            if args.phase == "pre-commit":
                validate_pre_commit(repo, ticket, state_path, state)
            elif args.phase == "skip":
                validate_skip(repo, ticket, state_path, state)
            elif args.phase == "merge":
                validate_merge(repo, ticket, state_path, state, args.commit)
            else:
                validate_post_commit(repo, ticket, state_path, state, args.commit)
    except (GateError, OSError) as exc:
        print(f"门禁失败: {exc}", file=sys.stderr)
        return 1

    print(f"门禁通过: {args.phase} {ticket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
