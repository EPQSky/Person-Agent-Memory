---
name: implement
description: Implement one clear piece of work from a spec or ready ticket, verify it, and review the resulting diff. Use when the user asks to build an approved slice.
---

# Implement

Implement one approved, user-visible slice from the referenced spec or ticket.

1. Read applicable `AGENTS.md`, domain docs, ADRs, the spec, and the ticket.
2. Confirm the ticket is unblocked. Before editing, capture `review-base` as the current `HEAD`; store the existing staged and unstaged patches plus untracked-file hashes in the OS temporary directory so unrelated user work remains distinguishable. For a local tracker, then move `ready-for-agent` to `in-progress`.
3. Identify the smallest public behavior seam. Use `$tdd` when the seam is testable and the user or repository has agreed to test-first work.
4. Implement the complete vertical slice. Keep unrelated refactors out of scope.
5. Run focused tests and type/lint checks during the work, then the repository's required final verification.
6. Run `$code-review` against the captured `review-base`, explicitly including committed, staged, unstaged, and untracked changes. Identify unchanged pre-existing user work separately instead of silently excluding or reviewing it as part of the slice. Address blocking findings and rerun affected verification.
7. Mark the local ticket `done` only after implementation, verification, and review complete.

Do not commit by default. Commit only when the user, ticket, or applicable repository guidance explicitly requires it. When committing, include only the intended slice and preserve unrelated user changes.
