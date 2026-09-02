# Domain documentation

Read domain documentation before changing behavior:

- `CONTEXT.md` is the glossary of stable domain terms and boundaries.
- `CONTEXT-MAP.md`, when present, maps multiple bounded contexts.
- `docs/adr/` records durable decisions with real trade-offs.

Create these files lazily. Do not put workflow instructions, implementation plans, tickets, transient notes, or generic engineering terms into `CONTEXT.md`.

When a term conflicts with code or existing documentation, surface the conflict before editing. Prefer no ADR over an ADR that merely restates an implementation detail.
