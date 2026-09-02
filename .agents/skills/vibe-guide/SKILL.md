---
name: vibe-guide
description: Route a software task through the installed VibeForge Lite workflow. Use when the user asks which workflow or skill fits their situation.
---

# Vibe Guide

Route only to skills that ship in this project. Do not suggest an unavailable skill.

## Main Flow

1. Use `$grill-with-docs` when a repository-backed idea still has unresolved product, domain, or architecture decisions.
2. Use `$batch-grill-with-docs` when the user wants every currently unblocked question in one round while preserving stable decisions in project docs.
3. Use `$to-spec` once the discussion is stable enough to become a product-facing specification.
4. Use `$to-tickets` when a multi-session implementation needs independently verifiable vertical slices.
5. Use `$implement` for one approved ticket or one small, already clear change. Use `$tdd` where a public behavior seam is agreed.
6. Use `$execute-spec-tickets` when the user explicitly wants an approved set of tickets executed serially with independent reviews and one scoped commit per completed ticket.
7. Use `$code-review <fixed-point>` before declaring an ordinary implementation complete; `$execute-spec-tickets` includes this review gate per ticket.
8. Use `$handoff` when another task or session must continue with the current context.

Keep adjacent phases in the current task when the next phase needs the full reasoning that produced the current result. At a genuine phase boundary, use [PHASE-BOUNDARIES.md](PHASE-BOUNDARIES.md) to choose between continuing, opening a clean task, writing a portable `$handoff`, delegating bounded work, or compacting context.

## Entry Points

- Use `$vibe-init` once to configure a repository, or again to preview/apply a template migration.
- Use `$grill-me` for a plan that does not belong to a repository and needs no durable project documentation.
- Use `$triage` for raw incoming issues or external pull requests. Tickets created by `$to-tickets` are already `ready-for-agent` and do not need triage.
- Use `$domain-modeling` directly when the problem is unclear terminology or an ADR-worthy decision rather than a full product interview.
- Use `$batch-grill-me` when batch questioning is wanted without writing project domain docs.

## Selection Rules

- Prefer the smallest workflow that produces a complete user-visible result.
- Do not create a spec and tickets for a trivial, well-understood edit.
- Do not route directly from an unresolved idea to implementation.
- Keep `CONTEXT.md` limited to domain language; keep workflow instructions in `AGENTS.md` and `docs/agents/`.
- Make context transitions only at phase boundaries. Do not compact in the middle of an unresolved interview, implementation slice, or verification loop.
