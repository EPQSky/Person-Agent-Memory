# 执行状态与门禁契约

仅在创建、恢复、提交或跳过 Ticket 时读取本文件。状态文件位于 Spec 目录下的 `.execute-spec-tickets-state.json`，不得进入 Git。

## 阶段

每张 Ticket 使用以下阶段，阶段切换前先原子更新状态文件：

1. `implementing`：实现或补齐中。
2. `reviewing`：等待或执行独立评审。
3. `repairing`：第 1 至第 9 轮修复。
4. `ready-to-commit`：最后一次独立 Review 已为 `passed`，全部验收和证据齐备，已形成精确暂存树。
5. `committing`：提交前门禁通过，正在创建 Commit。
6. `committed`：Commit 已创建并写回状态，等待提交后门禁。
7. `repair-exhausted`：九轮后仍有阻断问题，等待封存和影响判断。

旧状态使用其他阶段名时，恢复 Agent 必须先根据 Git、Tracker、评审记录和快照迁移；无法唯一判断时停止。

## Ticket Gate

`ticket_gate` 是防漏台账。以下字段必须由主 Agent 根据实际证据填写，不能让实现 Agent 自报即通过：

状态根节点还必须记录 `preexisting_staged_patch_sha256`：执行开始时对 `git diff --cached --binary --full-index` 的完整输出计算 SHA-256。没有原始暂存修改时记录空字节的 SHA-256，而不是省略字段。

```json
{
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
    {"reviewer": "独立 Reviewer ID", "result": "passed", "evidence": "评审结论或报告位置"}
  ],
  "repair_history": [],
  "preexisting_paths": ["建立本票 Review Base 时已经存在的修改或未跟踪文件"],
  "owned_paths": ["当前 Ticket 实际产生或明确接管的全部路径"],
  "staged_paths": ["当前 Ticket 的全部暂存路径"]
}
```

约束：

- `acceptance` 与 Markdown 验收清单逐条精确对应，每条都必须有独立证据。
- `review_history` 数量等于 `repair_round + 1`；前面的结果为 `blocked`，最后一次为 `passed`。
- 最后一次 Review 为 `blocked` 时只能进入 `repairing` 或 `repair-exhausted`，不得标记 Ticket 为 `done`、暂存完成状态、进入 `ready-to-commit` 或调用 `pre-commit`。提交前门禁只复核已通过 Review 的候选，不承担 Review 分流。
- 每轮 Reviewer 必须不同于实现者和所有修复 Agent，并使用新的 Reviewer ID。
- `repair_history` 数量等于 `repair_round`，轮次从 1 连续编号，每轮包含执行 Agent 和验证证据。
- 必需验证只能是 `passed`；非必需验证可以是 `skipped`，但必须说明依据和原因。
- `preexisting_paths` 来自本票修改前的快照，必须展开到文件级，不能只记录目录名。
- `owned_paths` 是主 Agent 对照 Review Base、工作区和未跟踪文件后确认的本票全部变化；`staged_paths` 必须与其完全一致并包含 Ticket 文件。
- 校验脚本会计算相对 Review Base 的实际变化。任何未进入 `owned_paths` 的新增、修改或未跟踪文件都会阻断提交，防止漏暂存；执行前原有修改仍按快照和补丁级归属检查保护。

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

任一门禁失败都不得提交、不得清理恢复材料、不得进入下一票。修复门禁数据或实现后必须完整重跑门禁，禁止人工声明“等价通过”。

提交成功后，在覆盖 `ticket_gate` 处理下一票前，必须把它完整深拷贝到 `ticket_results`。每条结果同时保存 Ticket 编号、Commit、修复轮数和最终评审结论。提交后门禁会校验 `ticket_results` 与当前 `ticket_gate` 完全一致，防止最终报告时只剩 Commit Hash 而丢失逐条验收证据。

## 原有暂存修改

若执行前已有用户暂存内容，先把暂存补丁、SHA-256 和索引树标识写入快照。每次 Ticket 提交前，必须让索引只包含 `staged_paths`；提交后再按快照恢复用户暂存状态。提交后门禁会重新计算暂存补丁 SHA-256，无法无损分离或恢复时停止，不得夹带提交或擅自取消暂存。

## 修复耗尽记录

每个修复耗尽 Ticket 至少记录：Ticket、九轮 Review 与 Repair 历史、剩余 Findings、失败验证、影响的接口或契约、所属改动路径、封存补丁和未跟踪副本位置、候选后续票的直接/传递依赖判断与代码影响证据。每个剩余 Finding 的 `impact` 必须是 `incorrect-result`、`resource-exhaustion` 或 `acceptance-failure`；风格、理论边角和低影响建议不得进入修复耗尽记录。

封存并移出失败代码后，Ticket 文件自身保留 `in-progress`，然后运行：

```bash
python3 <skill-dir>/scripts/validate_ticket_gate.py \
  --phase skip --ticket <ticket-path> --state <state-path>
```

`skip` 门禁要求：

- 正好九轮 Repair、十次结果为 `blocked` 的独立 Review。
- `blocking_findings` 每项包含严重度、问题、影响分类、受影响能力和证据；影响分类只能是错误结果、资源耗尽或验收失败。
- `repair_exhausted_archive` 指向存在的补丁、未跟踪文件副本目录和恢复说明。
- 失败代码已经移出工作区，除执行状态外只允许保留失败 Ticket 的 `in-progress` Tracker 修改。
- `candidate_assessments` 按编号覆盖全部后续未完成票，每票记录传递依赖、代码影响、证据和计算出的 `eligible`。
- `next_ticket` 必须是编号最小的安全候选；没有候选时为 `null` 并停止整组。
