# 执行状态与门禁契约

仅在创建、恢复、提交、合并或跳过 Ticket 时读取本文件。状态文件位于 Spec 目录下的 `.execute-spec-tickets-state.json`，不得进入 Git。快照、封存材料和 Ticket worktree 必须位于同一 Spec 目录下的 `.execute-spec-tickets/`，不得使用 `/tmp` 或其他系统临时目录。

## 阶段

每张 Ticket 使用以下阶段，阶段切换前先原子更新状态文件：

1. `implementing`：实现或补齐中。
2. `reviewing`：等待或执行独立评审。
3. `repairing`：持续整批修复；轮次从 1 连续编号且没有固定上限。
4. `ready-to-commit`：最后一次独立 Review 已为 `passed`，全部验收和证据齐备，已形成精确暂存树。
5. `committing`：提交前门禁通过，正在创建 Ticket 分支 Commit。
6. `ready-to-merge`：Ticket 分支 Commit 已创建并通过提交后门禁，等待合入记录的 base branch。
7. `merging`：已取得 integration branch 合并锁，正在预演或执行合并。
8. `merged`：Ticket 已合入记录的 base branch，合并后验证和门禁已通过。
9. `repair-stalled`：连续三轮没有可验证进展且阻断 Finding 集合不变，等待封存和影响判断。

旧状态使用其他阶段名时，恢复 Agent 必须先根据 Git、Tracker、评审记录和快照迁移；无法唯一判断时停止。旧 `repair-exhausted` 不能仅凭九轮计数迁移为停滞，默认回到 `repairing`；只有最后连续三轮的 Review 与修复证据满足当前停滞契约时，才能迁移为 `repair-stalled`。

## Ticket Gate

`ticket_gate` 是防漏台账。以下字段必须由主 Agent 根据实际证据填写，不能让实现 Agent 自报即通过：

状态根节点还必须记录 `preexisting_staged_patch_sha256`：执行开始时对 `git diff --cached --binary --full-index` 的完整输出计算 SHA-256。没有原始暂存修改时记录空字节的 SHA-256，而不是省略字段。

```json
{
  "integration_branch": "<执行开始时的当前分支>",
  "max_parallel_tickets": 3,
  "execution_root": ".scratch/demo/.execute-spec-tickets",
  "snapshot_dir": ".scratch/demo/.execute-spec-tickets/runs/run-123/snapshot",
  "worktrees_root": ".scratch/demo/.execute-spec-tickets/worktrees",
  "integration_worktree": "<absolute integration worktree path>",
  "active_tickets": [".scratch/demo/issues/01-demo.md"],
  "ticket_runs": {
    ".scratch/demo/issues/01-demo.md": {
      "worktree": {
        "path": ".scratch/demo/.execute-spec-tickets/worktrees/01-demo",
        "branch": "codex/demo-ticket-01",
        "base_branch": "<integration_branch at worktree creation>",
        "base_commit": "<integration_branch HEAD at worktree creation>"
      },
      "phase": "implementing",
      "repair_round": 0,
      "ticket_gate": {
        "implementer": "实现 Agent ID",
        "implementation_audit": {"status": "passed", "evidence": "已实现/部分实现审计摘要"},
        "ownership_check": {"status": "passed", "evidence": "Diff 与执行前快照核对结果"},
        "diff_inspection": {"status": "passed", "evidence": "主 Agent 实际检查 Diff 的结果"},
        "acceptance": [
          {"criterion": "与 Ticket 清单完全相同的文字", "status": "passed", "evidence": "代码、测试或运行证据"}
        ],
        "verification": [
          {"name": "完整命令", "required": true, "status": "passed", "evidence": "退出码与关键结果"}
        ],
        "review_history": [
          {
            "reviewer": "独立 Reviewer ID",
            "result": "blocked",
            "evidence": "穷尽式评审结论或报告位置",
            "blocking_findings": [
              {"id": "R1-F1", "severity": "P1", "issue": "问题摘要", "impact": "incorrect-result", "evidence": "复现、代码路径或验收条款"}
            ]
          },
          {"reviewer": "新的独立 Reviewer ID", "result": "passed", "evidence": "完整复审无阻断问题", "blocking_findings": []}
        ],
        "repair_history": [
          {
            "round": 1,
            "agent": "修复 Agent ID",
            "finding_ids": ["R1-F1"],
            "resolutions": [
              {"finding_id": "R1-F1", "status": "fixed", "evidence": "修复 Diff 与验证证据"}
            ],
            "evidence": "整批修复验证摘要"
          }
        ],
        "preexisting_paths": ["建立本票 Review Base 时已经存在的修改或未跟踪文件"],
        "owned_paths": ["当前 Ticket 实际产生或明确接管的全部路径"],
        "staged_paths": ["当前 Ticket 的全部暂存路径"],
        "merge": {
          "integration_branch": "<integration_branch>",
          "source_branch": "codex/demo-ticket-01",
          "before_head": "<integration_branch HEAD before merge>",
          "merge_commit": "<integration_branch merge commit>",
          "after_head": "<integration_branch HEAD after merge>",
          "verification": "合并后整体验证证据"
        }
      }
    }
  }
}
```

约束：

- `integration_branch` 必须是执行开始时记录的有效分支名；`max_parallel_tickets` 必须是 `3`；`active_tickets` 不得超过 3 张且不得重复。
- `integration_worktree` 是记录的 base branch 所在工作区；Ticket 门禁可以在独立 worktree 中运行，但必须用它解析 Spec、快照、封存和 worktree 根目录。传给门禁的 `--state` 必须是 integration worktree 内状态文件的绝对路径。`execution_root`、`snapshot_dir`、`worktrees_root` 和每个 Ticket 的 `worktree.path` 必须解析到 Spec 目录内；`snapshot_dir` 不得包含 `/tmp`、系统临时目录或仓库外路径。
- 每个活动 Ticket 必须有独占 worktree、独占分支和不可变 `base_commit`；两个活动 Ticket 不得复用 worktree 或分支。创建 worktree 失败时不得退回主 worktree。
- `ticket_runs` 是并行执行的权威状态；每个 Ticket 的 `phase`、`repair_round`、`ticket_gate`、`worktree` 和 `merge` 独立记录。顶层 `current_ticket` 只能作为恢复提示，不能代替每票状态。
- `acceptance` 与 Markdown 验收清单逐条精确对应，每条都必须有独立证据。
- `review_history` 数量等于 `repair_round + 1`；前面的结果为 `blocked`，最后一次为 `passed`。每次 Review 都必须完成当前 Diff 的穷尽式检查，不能遇到首个问题即返回。
- 每个 `blocked` Review 的 `blocking_findings` 必须是非空数组，Finding ID 在同一批次内唯一；同一问题跨轮仍存在时复用原 ID 和问题定义，新问题使用新 ID。每项包含严重度、问题、合法影响分类和证据。`passed` Review 的 `blocking_findings` 必须为空。
- 最后一次 Review 为 `blocked` 时只能进入 `repairing`，或在满足连续三轮无进展证据后进入 `repair-stalled`；不得标记 Ticket 为 `done`、暂存完成状态、进入 `ready-to-commit` 或调用 `pre-commit`。提交前门禁只复核已通过 Review 的候选，不承担 Review 分流。
- 每轮 Reviewer 必须不同于实现者和所有修复 Agent，并使用新的 Reviewer ID。
- `repair_history` 数量等于 `repair_round`，轮次从 1 连续编号。第 N 轮的 `finding_ids` 与第 N 次 blocked Review 的全部 Finding ID 必须精确一致，`resolutions` 必须逐项标记 `fixed` 并提供证据；一轮只修复部分 Findings 不得进入下一次 Review。
- 必需验证只能是 `passed`；非必需验证可以是 `skipped`，但必须说明依据和原因。
- `preexisting_paths` 来自本票修改前的快照，必须展开到文件级，不能只记录目录名。
- `owned_paths` 是主 Agent 对照 Review Base、工作区和未跟踪文件后确认的本票全部变化；`staged_paths` 必须与其完全一致并包含 Ticket 文件。
- 校验脚本会计算相对 Review Base 的实际变化。任何未进入 `owned_paths` 的新增、修改或未跟踪文件都会阻断提交，防止漏暂存；执行前原有修改仍按快照和补丁级归属检查保护。
- `ready-to-merge` 只能在 Ticket 分支 Commit 通过 `post-commit` 后进入；`merged` 只能在记录的 integration branch 上记录 `merge` 证据并通过合并后验证后进入。未合并的 Ticket 即使分支 Review 通过，也不得把 Tracker 标为 `done`。

## 强制命令

Tracker 审计：

校验器兼容项目现存的 `**Status:**` 与旧版 `**状态：**` 字段；状态值仍必须使用 `ready-for-agent`、`in-progress` 或 `done`。

```bash
python3 <skill-dir>/scripts/validate_ticket_gate.py \
  --phase tracker --ticket <ticket-path>
```

最后一次独立 Review 为 `passed` 且验收、验证全部通过后，才能把阶段设为 `ready-to-commit` 并形成精确暂存树。Review 为 `blocked` 时禁止运行此命令：

```bash
python3 <skill-dir>/scripts/validate_ticket_gate.py \
  --phase pre-commit --ticket <ticket-path> --state <state-path>
```

门禁通过后把阶段设为 `committing`，创建 Commit；随后记录 Commit、`completed_commits`，把阶段设为 `committed`，再运行：

```bash
python3 <skill-dir>/scripts/validate_ticket_gate.py \
  --phase post-commit --ticket <ticket-path> --state <state-path> --commit HEAD
```

Ticket 分支提交后，主 Agent 在 integration worktree 的 base branch 中完成无冲突合并并更新该票阶段为 `merged`，然后运行：

```bash
python3 <skill-dir>/scripts/validate_ticket_gate.py \
  --phase merge --ticket <ticket-path> --state <state-path> --commit HEAD
```

合并门禁要求当前分支等于状态中的 `integration_branch`，`merge.source_branch` 指向通过提交后门禁的 Ticket 分支，`merge.integration_branch` 与状态一致，`merge.before_head` 和 `merge.after_head` 可由 Git 历史证明，且合并后 Tracker、验收和整体验证仍然成立。合并门禁失败时不得删除 Ticket worktree、分支或恢复材料。

任一门禁失败都不得提交、合并、清理恢复材料或进入下一票。修复门禁数据或实现后必须完整重跑门禁，禁止人工声明“等价通过”。

Ticket 分支提交成功后，在覆盖该票 `ticket_gate` 处理下一阶段前，必须把它完整深拷贝到 `ticket_results`；合并成功后再补记 `merge_commit` 和合并后验证。每条结果同时保存 Ticket 编号、Ticket 分支 Commit、合并 Commit、修复轮数和最终评审结论。提交后与合并门禁会校验 `ticket_results` 与该票 `ticket_gate` 完全一致，防止最终报告时只剩 Commit Hash 而丢失逐条验收证据。

## 原有暂存修改

若执行前已有用户暂存内容，先把暂存补丁、SHA-256 和索引树标识写入 Spec 内快照。每次 Ticket 分支提交前，必须让该 worktree 的索引只包含 `staged_paths`；提交后再按快照恢复主 worktree 用户暂存状态。提交后门禁会重新计算暂存补丁 SHA-256，无法无损分离或恢复时停止，不得夹带提交或擅自取消暂存。

## 修复停滞记录

修复轮数没有固定上限，轮数本身不能触发封存或跳票。只有最后三轮修复前后的四次独立 Review 返回完全相同的阻断 Finding ID 集合，而且三轮都没有新增验收证据、验证进展或可观察行为改善时，Ticket 才能进入 `repair-stalled`。

每个修复停滞 Ticket 至少记录：Ticket、全部 Review 与 Repair 历史、最后三轮 `stall_assessments`、最后一次 blocked Review 的完整剩余 Findings 批次、失败验证、影响的接口或契约、所属改动路径、封存补丁和未跟踪副本位置、候选后续票的直接/传递依赖判断与代码影响证据。每个 `stall_assessment` 包含连续轮次、完整 Finding ID 集合、`no-progress` 状态和具体证据。每个剩余 Finding 必须保留 Review 中的 ID，其 `impact` 必须是 `incorrect-result`、`resource-exhaustion` 或 `acceptance-failure`；风格、理论边角和低影响建议不得进入修复停滞记录。

封存并移出失败代码后，Ticket 文件自身保留 `in-progress`，然后运行：

```bash
python3 <skill-dir>/scripts/validate_ticket_gate.py \
  --phase skip --ticket <ticket-path> --state <state-path>
```

`skip` 门禁要求：

- 至少三轮 Repair，且最后三轮修复前后的四次独立 Review 具有完全相同的阻断 Finding ID 集合。
- `stall_assessments` 精确覆盖最后三轮，每轮标记 `no-progress` 并包含没有新增验收、验证或行为改善的证据。
- `blocking_findings` 每项包含严重度、问题、影响分类、受影响能力和证据；影响分类只能是错误结果、资源耗尽或验收失败。
- `repair_stalled_archive` 指向存在的补丁、未跟踪文件副本目录和恢复说明。
- 失败代码已经移出工作区，除执行状态外只允许保留失败 Ticket 的 `in-progress` Tracker 修改。
- `candidate_assessments` 按编号覆盖全部后续未完成票，每票记录传递依赖、代码影响、证据和计算出的 `eligible`。
- `next_ticket` 必须是编号最小的安全候选；没有候选时为 `null` 并停止整组。
