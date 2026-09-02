# Project Agent Guide

<!-- vibeforge-lite:start version=0.3.5 -->
## Vibecoding workflow

- Prefer the project-local workflow in `.agents/skills/`; rely on an installed Plugin only when the project explicitly selects Plugin mode.
- Use `$vibe-guide` when the right workflow is unclear.
- Clarify unresolved product, domain, and architecture decisions with `$grill-with-docs`; use `$batch-grill-with-docs` when a batch round is preferred.
- Publish stable decisions through the tracker configured in `docs/agents/issue-tracker.md` with `$to-spec`, then split multi-session work into vertical tickets with `$to-tickets`.
- Implement one approved slice at a time with `$implement`; use `$tdd` when a stable behavior seam exists. Do not commit unless the user or repository policy explicitly asks.
- Use `$execute-spec-tickets` only when the user explicitly wants an approved ticket set executed serially with independent reviews and one scoped commit per completed ticket.
- Before completion, review from the captured base with `$code-review <review-base>` along both Standards and Spec axes, including committed and working-tree changes.
- Use `$handoff` when another task must continue the work.

Project artifacts use `CONTEXT.md` only for stable domain language, `docs/adr/` for durable decisions, `docs/agents/` for workflow conventions, and the configured tracker for specs and tickets.

Local implementation ticket states are `ready-for-agent`, `in-progress`, and `done`.
<!-- vibeforge-lite:end -->
