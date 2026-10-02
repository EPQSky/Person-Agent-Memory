# Local issue tracker

The default tracker is local Markdown:

- Specs: `.scratch/<feature-slug>/spec.md`
- Tickets: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`
- Ticket order: blockers first, numbered from `01`

Each ticket states what to build, blockers, status, and verifiable acceptance criteria. `$to-tickets` creates tickets in `ready-for-agent`; `$implement` moves the selected ticket to `in-progress` and only marks `done` after verification.

Allowed implementation states are `ready-for-agent`, `in-progress`, and `done`. Do not use `claimed` or `resolved` for new local tickets.

When `$execute-spec-tickets` runs a Spec, it records the current integration branch, then creates the execution state, snapshots, archives, and one Git worktree per active Ticket under that Spec's `.execute-spec-tickets/` directory. Up to three unblocked Tickets may run in parallel when code-impact evidence proves they are independent; each Ticket must pass review and verification before its branch is merged back into the base branch from which its worktree was created.
