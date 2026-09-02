---
name: batch-grill-with-docs
description: Batch interview a plan, feature, architecture decision, or bug investigation while capturing stable domain terms and ADR-worthy decisions. Use when the user wants grill-with-docs behavior but asks for batch questions, frontier questions, fewer back-and-forth turns, or a batched version of grilling with documentation.
---

# Batch Grill With Docs

Run a `$batch-grill-me` session while applying `$domain-modeling` whenever the discussion settles domain language or architecture decisions.

## Workflow

1. Read the repo's domain docs before asking:
   - `CONTEXT.md`, if present.
   - Relevant ADRs under `docs/adr/`, if present.
   - `docs/agents/domain.md`, if present.
2. Explore facts yourself. Use filesystem and tools for facts that can be discovered locally; ask the user only for decisions.
3. Model the topic as a design tree. A question is on the frontier only when all of its prerequisites are already settled.
4. Ask the full current frontier in one round:
   - Number every question.
   - Give a recommended answer for each question.
   - Format each as `Q<number> - <title>: <question>`, followed by `Recommended: <answer and brief reasoning>`.
   - Mention dependencies when a question is intentionally deferred.
   - Use 简体中文 by default for user-facing questions and summaries in this repo.
5. Wait for the user's answers before asking the next round.
6. After each answer round, immediately capture stable outcomes:
   - Update `CONTEXT.md` for settled domain terms and meanings.
   - Create or update an ADR only when the decision is hard to reverse, surprising without context, and the result of a real trade-off.
   - Do not write speculative, unresolved, or implementation-only notes into `CONTEXT.md`.
7. Recompute the frontier and repeat until no unanswered decision remains.
8. End with a concise shared-understanding summary and ask the user to confirm. Do not implement until they confirm.

## Documentation Rules

Keep `CONTEXT.md` as a glossary, not a spec or scratchpad. Use the format from `$domain-modeling` when editing it.

Use ADRs sparingly. Prefer no ADR over a weak ADR.

If a user's answer contradicts existing glossary terms or ADRs, surface the conflict before writing files.

## Difference From Related Skills

`$grill-with-docs` uses `$grilling`, so it asks one question at a time and updates docs as decisions settle.

`$batch-grill-me` asks the current frontier in batches, but does not automatically maintain domain docs.

`$batch-grill-with-docs` combines the batch interview rhythm with domain-document maintenance.
