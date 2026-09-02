---
name: code-review
description: Review changes since a fixed point along Standards and Spec axes, reporting actionable bugs, regressions, security risks, scope drift, and missing tests. Use for branches, pull requests, or work-in-progress diffs.
---

# Code Review

Review the complete effective change since a fixed point along two independent axes. This includes committed, staged, unstaged, and untracked work.

- **Standards:** repository rules, correctness, security, maintainability, and relevant code smells.
- **Spec:** missing requirements, incorrect behavior, scope creep, and verification gaps against the originating request.

## 1. Pin The Review

- Resolve the user-provided commit, branch, tag, or merge-base.
- If no fixed point is provided, infer the branch merge-base when unambiguous; otherwise ask.
- Resolve `<review-base>` with `git merge-base <fixed-point> HEAD`, then capture `git diff <review-base>` so tracked working-tree changes are included.
- Capture staged and unstaged summaries separately when that distinction affects ownership or review scope.
- Capture untracked paths with `git ls-files --others --exclude-standard`; read relevant untracked files as part of the review.
- Capture `git log <fixed-point>..HEAD --oneline` once.
- Stop early only when the committed diff, tracked working-tree diff, and relevant untracked-file set are all empty.

## 2. Gather Sources

- Read applicable `AGENTS.md`, repository standards, ADRs, and verification commands.
- Find the originating issue or spec from the user argument, commit references, branch name, or `.scratch/`.
- If no spec exists, state that the Spec axis is limited to the user's stated request.

## 3. Execute Both Axes

When multi-agent delegation is available and explicitly requested or permitted by applicable guidance, Standards and Spec may run in parallel. Otherwise perform both reviews inline. The quality bar and report shape are identical in either mode; lack of sub-agents is never a reason to skip an axis.

Prioritize concrete defects over stylistic preferences. For each finding include severity, file and line, the failure mode, and the smallest credible fix. Treat undocumented style opinions and smell heuristics as judgment calls, not hard violations.

## 4. Report

Lead with findings ordered by severity across both axes, while labeling every item `Standards`, `Spec`, or both. Then include:

- open questions or assumptions;
- a short verification and residual-risk note;
- a one-line count by axis.

If no actionable findings exist, say so clearly and name any test gaps or residual risks. Do not modify code during review unless the user asks for fixes.
