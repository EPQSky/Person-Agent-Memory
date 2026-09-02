# Local issue tracker

The default tracker is local Markdown:

- Specs: `.scratch/<feature-slug>/spec.md`
- Tickets: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`
- Ticket order: blockers first, numbered from `01`

Each ticket states what to build, blockers, status, and verifiable acceptance criteria. `$to-tickets` creates tickets in `ready-for-agent`; `$implement` moves the selected ticket to `in-progress` and only marks `done` after verification.

Allowed implementation states are `ready-for-agent`, `in-progress`, and `done`. Do not use `claimed` or `resolved` for new local tickets.
