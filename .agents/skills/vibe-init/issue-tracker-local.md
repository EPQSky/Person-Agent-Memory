# Issue Tracker: Local Markdown

Issues, specs, and implementation tickets live as Markdown under `.scratch/`.

## Conventions

- One effort per directory: `.scratch/<feature-slug>/`.
- The specification is `.scratch/<feature-slug>/spec.md`.
- Tickets are individual files at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01` in blocker-first order.
- Each ticket records `Blocked by:` and one `**Status:** <state>` line near the top.
- Comments and execution notes append under `## Comments`.

## Execution States

- `ready-for-agent`: clear, approved, and unblocked work.
- `in-progress`: an Agent or human has started the ticket.
- `done`: implementation, required verification, and review are complete.

Do not use `claimed` or `resolved` for implementation tickets. During migration, map `claimed` to `in-progress` and `resolved` to `done` only after checking that the implementation and verification really exist.

## Skill Operations

- When a skill says "publish to the issue tracker", create the appropriate spec or one-file-per-ticket artifacts under `.scratch/<feature-slug>/`.
- When a skill says "fetch the relevant ticket", read the referenced file and any blockers.
- A ticket is actionable only when every blocker is `done`.
- `$implement` moves the selected ticket to `in-progress`, and to `done` only after its completion gates pass.
