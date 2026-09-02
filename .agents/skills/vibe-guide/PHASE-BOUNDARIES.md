# Phase Boundaries

A phase is a coherent chunk of work inside a task, such as grilling, implementation, review, or verification. Decide how to manage context only when one phase has actually ended.

Work through these options in order. The first suitable option wins.

1. **Continue in the current task.** Prefer this when the next phase needs the current phase as a primary source, or when enough context remains. Grilling followed by immediate implementation is the common case.
2. **Open a clean task.** Use a new task when the current context is irrelevant to what comes next and no rationale needs to travel with it.
3. **Use `$handoff`.** Write a portable handoff when work must move to another harness, repository, directory, colleague, or independently resumed task.
4. **Delegate bounded work.** Use a sub-agent only when delegation is available and permitted, the subtask is independently scoped, and the parent task can continue without losing ownership of the main decision path.
5. **Compact context.** Use compaction when the context remains relevant, the work stays in the same task and repository, and continuing with the complete history is no longer practical.

Every option except continuing turns primary conversational history into a secondary summary or artifact. That trade can be useful, but it is lossy. Do not compact or hand off in the middle of an unresolved phase merely to reduce context size.
