# Personal Agent Memory Platform

## Problem Statement

个人使用 Codex 处理多个本地或远程代码项目时，项目知识、长期约束、已确认决定、个人偏好和可复用经验散落在 Markdown 文档及历史会话中。Codex 的单次上下文有限，跨会话后无法稳定延续这些信息；直接把完整会话或整个文档目录塞入上下文既超出预算，也会引入噪声、过期内容、敏感信息和提示注入风险。

现有 JiuwenMemory 已提供长期记忆、Markdown 文件索引、Graph Memory、REST、MCP 和 Codex Hooks 的基础能力，但这些能力尚未形成符合个人本地记忆需求的完整产品：任意 Markdown 目录还不是受治理的记忆库，Graph Memory 尚未与 Markdown 首轮检索形成受来源约束的二阶段链路，现有 Codex 插件不能完整沉淀用户消息与助手最终回复，也缺少候选审批、冲突裁决、真正遗忘、本地 Git 审计和 10K tokens 上下文硬预算。

用户需要一个本地优先、可审计、不会干扰 Codex 主流程的个人记忆平台。它必须以 Markdown 为唯一权威事实来源，以项目为默认检索边界，通过 Graph Memory 做有限关联扩展，并允许用户从平台中治理所有正式记忆。

## Solution

构建一个单用户本地记忆平台。用户可以注册任意目录作为用户记忆库或项目记忆库；目录不存在时平台自动创建，目录内已有 Markdown 可以导入并建立文本、向量和图派生索引。每个项目通过中央注册表显式绑定一个项目记忆库，Codex 会话默认且仅检索当前项目记忆库，用户记忆库只能被显式查询。

平台以 Markdown 作为唯一权威记忆。向量索引、全文索引和 Graph Memory 都是可删除、可重建的派生视图。检索先通过全文和向量混合召回 Markdown 片段，再以直接命中为种子执行有来源约束的 Graph Memory 扩展。服务端对最终记忆上下文包执行 10K tokens 硬限制，直接命中至少获得 60% 预算，图扩展最多使用 30%，来源与结构信息最多使用 10%。

一个本地守护进程统一提供 Web、REST、MCP、目录变更检查、会话采集和后台任务。全局 Codex 插件使用 Hooks 逐事件采集用户消息和助手最终回复，并在生成前注入相关记忆；MCP 提供显式检索、创建候选和状态查询能力。记忆服务不可用时 Hook 写入本地 spool 或跳过召回，绝不阻塞 Codex。

所有会话内容先进入采集收件箱，再异步生成候选记忆。只有有长期价值、有明确证据、有来源、无冲突且不敏感的允许类型才可自动晋升。用户通过本地 Web 平台检查、编辑、确认、删除和恢复正式记忆。每次完成的权威记忆操作形成隔离的本地 Git 原子提交；平台不修改项目仓库、不自动配置远端，也不执行 push 或 pull。

## User Stories

1. As a user, I want to register a local directory as a memory library, so that its Markdown documents can become available to my coding agents.
2. As a user, I want the platform to create a registered directory when it does not exist, so that I do not need to prepare the filesystem manually.
3. As a user, I want to register a directory containing existing Markdown, so that I can adopt my current notes without rewriting them.
4. As a user, I want every memory library to have a stable identifier independent of its directory name, so that similarly named projects do not share memory accidentally.
5. As a user, I want to distinguish user memory libraries from project memory libraries, so that personal information and project facts have separate boundaries.
6. As a user, I want to bind a project root explicitly to one project memory library, so that retrieval does not rely on ambiguous directory-name inference.
7. As a user, I want a worktree to inherit a main project's binding only when configured, so that related worktrees can share memory without accidental coupling.
8. As a Codex user, I want a project session to search only its current project memory library by default, so that unrelated personal or project information is not injected.
9. As a Codex user, I want an unbound project to report that no binding exists, so that the platform does not silently fall back to the wrong library.
10. As a user, I want to query my user memory library explicitly, so that cross-project personal memory remains available without becoming an automatic dependency.
11. As a user, I want Markdown to remain the authoritative memory representation, so that I can inspect the actual facts without opening a proprietary database.
12. As a user, I want derived indexes and graph data to be rebuildable from Markdown, so that corruption or upgrades cannot create an independent source of truth.
13. As a user, I want to edit authoritative memory through the platform, so that every change updates provenance, indexes, graph state and history consistently.
14. As a user, I want a Markdown source editor and rendered preview, so that I can safely author and inspect memory documents.
15. As a user, I want to see a diff before saving an edit, so that I understand exactly what will change.
16. As a user, I want the platform to detect direct external edits, moves or deletions, so that unsupported changes do not silently corrupt platform state.
17. As a user, I want to review a three-way comparison for an out-of-band change, so that I can import it or restore the platform version deliberately.
18. As a user, I want existing documents to keep their original formatting when possible, so that adopting the platform does not cause unnecessary Markdown churn.
19. As a user, I want metadata for existing documents to live outside their content by default, so that identifiers and index state do not pollute my notes.
20. As a user, I want system-generated formal memories to include readable provenance metadata, so that I can understand when and why they were created.
21. As a user, I want every automatic memory to reference its source session, document, project and time, so that I can verify its basis.
22. As a user, I want graph-expanded results to reference authoritative Markdown, so that associations never appear as unsupported facts.
23. As a Codex user, I want retrieval to start with relevant Markdown fragments, so that direct evidence is prioritized over graph associations.
24. As a Codex user, I want keyword and semantic retrieval to work together, so that exact identifiers and conceptual matches can both be found.
25. As a Codex user, I want Markdown chunking to respect headings, paragraphs and complete code blocks, so that recalled fragments preserve useful context.
26. As a Codex user, I want Graph Memory to expand from direct hits by one hop by default, so that related facts are discovered without excessive drift.
27. As a Codex user, I want graph traversal capped at two hops, so that association remains bounded and predictable.
28. As a Codex user, I want every injected result to include its library, document location, source type, score and source version, so that recalled context is auditable.
29. As a Codex user, I want the memory context package limited to 10K tokens by the server, so that memory cannot consume the entire model context.
30. As a Codex user, I want to request a smaller retrieval budget, so that lightweight tasks can avoid unnecessary context.
31. As a Codex user, I want stale graph results rejected when their Markdown source version has changed, so that eventual consistency never returns superseded graph facts.
32. As a Codex user, I want keyword retrieval to remain available when model services fail, so that core memory is still usable offline or during outages.
33. As a Codex user, I want reranking failure to fall back to the original retrieval order, so that an optional enhancement cannot break recall.
34. As a Codex user, I want Graph or LLM failure to return direct Markdown hits with a degradation indicator, so that I know the result is incomplete.
35. As a Codex user, I want memory content treated as untrusted data, so that a recalled document cannot override agent rules or authorize tools.
36. As a user, I want secrets filtered before capture is persisted or sent to a model, so that API keys, passwords, tokens and private keys do not become memory.
37. As a user, I want suspicious sensitive content quarantined outside retrieval, so that I can decide whether it should be retained.
38. As a user, I want raw retrieval queries excluded from permanent history by default, so that diagnostic data does not become another personal archive.
39. As a user, I want a global Codex plugin to capture sessions across bound projects, so that I do not need to install configuration into every repository.
40. As a user, I want Codex Hooks to capture each user prompt, so that a session interrupted before completion is not entirely lost.
41. As a user, I want Codex Hooks to capture the assistant's final reply, so that confirmed outcomes and explanations can be considered for memory extraction.
42. As a user, I want hidden reasoning, raw tool output and complete file contents excluded from default capture, so that memory remains focused and safer.
43. As a user, I want capture events written with idempotency identifiers, so that retries do not create duplicate session history.
44. As a Codex user, I want Hook failures to fail open, so that memory infrastructure never blocks my coding work.
45. As a user, I want locally spooled Hook events processed after the daemon recovers, so that temporary outages do not unnecessarily lose capture data.
46. As a user, I want captured events consolidated asynchronously, so that model extraction does not add latency to each Codex turn.
47. As a user, I want raw capture events retained for 30 days by default, so that extraction can be audited without keeping transcripts indefinitely.
48. As a user, I want successfully processed capture data removable earlier, so that I can minimize retained conversation data.
49. As a user, I want extracted content to enter a candidate state before becoming formal memory, so that uncertain conclusions can be governed.
50. As a user, I want stable preferences, confirmed decisions, project constraints, verified domain facts, reusable experience and external references to be eligible memory types, so that formal memory has durable value.
51. As a user, I want task progress, temporary plans, one-off output and unverified guesses excluded from formal memory, so that the library does not accumulate operational noise.
52. As a user, I want assistant statements to require user confirmation or independent project evidence before becoming facts, so that confident model language is not mistaken for truth.
53. As a user, I want automatic promotion to require policy, evidence, provenance, conflict and sensitivity checks, so that model confidence alone cannot publish memory.
54. As a user, I want high-confidence eligible candidates to be promoted automatically, so that routine memory growth does not require constant approval.
55. As a user, I want sensitive, conflicting or high-impact candidates to require manual approval, so that consequential memory stays under my control.
56. As a user, I want unreviewed candidates retained for 90 days and then moved through a recoverable grace period, so that the candidate queue does not grow forever.
57. As a user, I want to pin a candidate or conflict so that it does not expire while I am still considering it.
58. As a user, I want semantically equivalent candidates to add provenance to an existing memory instead of duplicating its body, so that recall remains concise.
59. As a user, I want uncertain duplicates placed in a merge queue, so that the system does not silently combine distinct facts.
60. As a user, I want conflicting candidates to preserve the current formal memory until I decide, so that unreviewed updates do not change agent behavior.
61. As a user, I want conflict resolution options to keep, replace, merge or scope facts by condition, so that real-world nuance is representable.
62. As a user, I want a changed fact to supersede rather than overwrite its predecessor, so that historical truth remains available.
63. As a user, I want current retrieval to prefer the effective memory version, so that Codex receives the latest confirmed state.
64. As a user, I want historical queries to retrieve superseded versions, so that I can reconstruct past decisions.
65. As a user, I want to delete a formal memory through the platform, so that its text, summary, embedding and graph data are removed consistently.
66. As a user, I want deletion to retain only a keyed fingerprint and source range, so that old sessions cannot recreate forgotten content without retaining the content itself.
67. As a user, I want to restore a deleted or earlier memory version through a new recorded operation, so that history remains append-only and auditable.
68. As a user, I want each completed memory mutation to create one atomic local Git commit, so that changes are easy to inspect and reverse.
69. As a user, I want batch operations to produce one clearly identified batch commit, so that a deliberate group of changes remains coherent.
70. As a user, I want automatic promotions to create commits that identify their source task and session, so that background changes are attributable.
71. As a user, I want memory Git history isolated from my project repository, so that the platform cannot commit unrelated source-code changes.
72. As a user, I want explicit authorization before the platform reuses an existing dedicated memory repository, so that repository discovery cannot grant write authority.
73. As a user, I want the platform to avoid automatic remote, push and pull operations, so that local memory is never transmitted implicitly.
74. As a user, I want restores implemented as new commits rather than rewritten history, so that the complete audit chain is preserved.
75. As a user, I want platform commits to use a fixed local service identity while separately recording the operation actor, so that Git history is stable and attribution remains meaningful.
76. As a user, I want a Web view of Git history and diffs, so that I can audit memory without using command-line Git.
77. As a user, I want one local daemon to manage all libraries, so that service operation and upgrades remain simple.
78. As a user, I want each library to have independent locks, queues, indexes and health status, so that one busy or damaged library does not corrupt another.
79. As a user, I want the Web, REST and MCP interfaces bound to loopback by default, so that the MVP is not exposed to the network.
80. As a user, I want the platform to generate a random API key on first startup, so that local clients require explicit credentials.
81. As a user, I want to rotate the API key and see only its fingerprint in logs, so that credential exposure can be contained.
82. As a Codex user, I want MCP tools for search, candidate creation, memory lookup, candidate listing, library listing and sync status, so that the agent can use memory without unrestricted mutation rights.
83. As a user, I want formal edit, deletion, restoration and approval restricted to the Web platform in the MVP, so that agent tools cannot silently rewrite authoritative memory.
84. As a user, I want initial text indexing to make a newly registered library searchable before graph construction finishes, so that large imports remain usable.
85. As a user, I want Graph construction progress and degraded state visible in the platform, so that I can distinguish incomplete associations from missing documents.
86. As a user, I want model endpoints for LLM, Embedding and Reranker configured independently, so that I can combine local and remote OpenAI-compatible services.
87. As a user, I want model credentials kept in protected platform configuration rather than memory directories, so that memory export does not leak secrets.
88. As a maintainer, I want JiuwenMemory pinned to a verified release or commit behind an adapter, so that upstream changes do not silently alter behavior.
89. As a maintainer, I want upstream upgrades tested against import, retrieval, graph and Codex integration behavior, so that compatibility is demonstrated before adoption.
90. As a user, I want the platform installed directly on Linux through a Python package workflow, so that it can access host directories, Git and Codex Hooks naturally.
91. As a maintainer, I want a reproducible Docker acceptance environment, so that complete behavior can be verified independently of a developer workstation.
92. As a maintainer, I want the project distributed under Apache-2.0, so that it remains compatible with JiuwenMemory and can accept external contributions.

## Implementation Decisions

- The system is a single-user, local-first platform composed of one long-running daemon, one browser management interface and one globally installed Codex plugin.
- The daemon owns the public Web, REST and MCP surfaces, capture ingestion, background extraction, directory reconciliation, retrieval orchestration, model adapters, indexing and per-library write coordination.
- The backend uses Python 3.11 or newer and FastAPI. The management interface uses React and TypeScript. Codex Hook adapters use small Node.js programs that communicate with the daemon.
- JiuwenMemory is consumed as a pinned external dependency through a project-owned adapter layer. The project does not copy its source or initially maintain a fork.
- Upstream upgrades require an explicit compatibility run covering existing Markdown import, mixed retrieval, Graph Memory, persistence migrations, MCP contracts and Codex Hook behavior.
- Every memory library has a stable library identifier, a canonical directory path, a kind of user or project, indexing state, graph state, retention policy and independent work queue.
- A central user-level registry stores library metadata and explicit project bindings. It does not store authoritative memory content.
- Project lookup uses a canonical project root and stable project identity. Repository basename matching is not authoritative. Worktree inheritance is explicit.
- An unbound Codex working directory produces an unbound result and no memory injection. It never falls back to the user library or another project library.
- The default query scope is exactly the current project memory library. The user memory library and any additional project libraries require an explicit library identifier.
- If a registered directory does not exist, the platform creates it. Registration of an existing directory starts text indexing immediately and schedules graph construction asynchronously.
- Markdown is the only authoritative memory. Full-text indexes, vector indexes and Graph Memory are derivative and must be rebuildable without losing formal memory.
- The managed area logically separates published memories from capture Inbox events, candidate memories and platform state. Only existing user Markdown and published-memory Markdown participate in normal retrieval.
- Existing Markdown is not forced to accept platform frontmatter. Stable identifiers, document versions and indexing metadata are kept in sidecar state. System-generated published memories include readable provenance metadata.
- Formal memory mutations are accepted only through the platform. Direct external file edits, moves and deletions are detected as out-of-band changes and suspended from indexing until the user imports or rejects them.
- Out-of-band conflict handling uses a base version, the platform version and the external version. No automatic merge is performed when both sides changed.
- A single write coordinator serializes mutations for each library using file locks, atomic file replacement and idempotent operation identifiers.
- The platform uses a sidecar Git repository for each memory library, with the memory directory as its worktree. It does not create a nested Git directory and does not stage or commit through a containing project repository.
- Reuse of an existing dedicated memory repository requires explicit user authorization. Detection of Git metadata alone never grants commit authority.
- Each completed edit, approval, promotion, deletion, restoration or accepted external change creates one atomic commit. Deliberate batch operations create one labeled batch commit.
- Automatic promotions create separate commits carrying task and source identifiers. Git commits use a stable service identity; actor attribution is preserved in platform operation metadata.
- Git tracks authoritative Markdown, tombstone material and a portable library manifest. Runtime databases, derived indexes, graph files, Inbox data, candidate state, logs, caches and credentials are ignored.
- The MVP does not configure a Git remote or execute push or pull. Restoration creates a new commit and never rewrites existing history.
- SQLite stores the central registry, library sidecar metadata, capture events, candidate workflow, operation audit, idempotency records, full-text search data and background task state.
- JiuwenMemory's file-index capability is adapted for Markdown vector search. Graph Memory uses its currently supported local Milvus Lite storage.
- Only published authoritative memories are projected into Graph Memory. Inbox events and candidates are not graph inputs.
- Every derived chunk and graph object carries an authoritative source identifier and source version. Search discards graph results whose source version no longer matches Markdown state.
- Initial retrieval performs structured Markdown chunking. Headings and paragraph boundaries are honored, and fenced code blocks remain intact.
- Direct retrieval combines full-text or keyword scoring with vector semantic scoring. Results are deduplicated and optionally reranked.
- Graph expansion starts only from direct Markdown hits. The default is one hop; a second hop is allowed only when direct and first-hop results are insufficient and budget remains.
- Graph results without a valid Markdown provenance reference are never included in a memory context package.
- Context allocation is dynamic but constrained: direct Markdown results receive at least 60% of the available budget, graph expansion may consume at most 30%, and provenance plus structural metadata may consume at most 10%. Unused graph or metadata budget can be filled by additional direct hits.
- The server uses the selected target model's tokenizer when available and a conservative fallback tokenizer otherwise. The server enforces an absolute maximum of 10K tokens; callers may request less but never more.
- Search results include content, library identifier, document path, heading or line location, provenance type, direct-hit or graph-expansion classification, relevance score, source version and degradation state.
- Retrieval failures degrade progressively. Reranking failure keeps the pre-rerank order; Graph or LLM failure returns direct Markdown; embedding failure retains full-text search. Degradation is visible in the response.
- Memory content is inserted into model context as explicitly delimited untrusted data. It cannot alter system policy, authorize tools or directly initiate commands.
- Capture performs secret detection before durable persistence and before any model call. Confirmed secrets are discarded. Suspected sensitive events are quarantined and excluded from search until the user resolves them.
- The global Codex plugin resolves the current project through the Hook working directory and the central registry. It does not write plugin configuration into each project.
- Codex Hooks capture allowed user prompts and assistant final replies as separate idempotent events. Hidden reasoning, raw tool output, subagent internals and full file contents are excluded by default.
- Hook ingestion is asynchronous and fail-open. When the daemon is unavailable, the plugin writes a bounded local spool; when neither daemon nor spool is available, it skips capture and retrieval without blocking Codex.
- Pre-generation retrieval uses a short timeout. A timeout results in no memory injection for that turn.
- Pre-compaction and session-lifecycle events trigger idempotent consolidation but are not the sole source of capture correctness.
- Raw capture events are retained for 30 days by default and may be removed earlier after successful extraction when no unresolved candidate depends on them.
- Extraction produces candidate memories rather than directly publishing arbitrary summaries. Allowed formal types are decisions, constraints, preferences, domain facts, reusable experience and external references.
- Task progress, temporary plans, one-off command output, transient paths and unverified guesses are ineligible for formal memory.
- Assistant output alone is insufficient evidence for a user or project fact. User confirmation or independently verifiable project evidence is required.
- Automatic promotion requires an eligible type, explicit evidence, valid provenance, no unresolved conflict, no sensitive content and policy permission. Model confidence is only one input.
- Sensitive, conflicting and high-impact candidates require user approval. Unreviewed candidates expire after 90 days into a 30-day recoverable state unless pinned or involved in an unresolved conflict.
- Equivalent candidate content augments the provenance of an existing memory. Uncertain similarity creates a merge candidate instead of automatically combining content.
- New facts that replace old facts create a supersession relation with an effective time. Current retrieval prefers the effective version while historical retrieval can include superseded versions.
- Conflict resolution supports retaining the existing memory, adopting the candidate, merging into a new version, or declaring conditional coexistence. Every outcome is a memory mutation transaction.
- Deleting a published memory removes its body, summary, vector entries and graph projection. A tombstone retains only a keyed HMAC fingerprint, source range and deletion time to prevent regeneration from old sources.
- Tombstones never retain plaintext, summaries or embeddings. Restoring a forgotten memory is an explicit platform operation with a new Git commit.
- The local Web platform provides library registration and binding, document navigation, Markdown source and preview, save diffs, provenance views, candidate approval, conflict comparison, deletion and restoration, Git history, import progress, index health and graph synchronization state.
- MCP exposes search, candidate creation, memory lookup, candidate listing, library listing and synchronization status. The MVP does not expose unrestricted formal edit, approval, deletion or restoration through MCP.
- The daemon binds Web, REST and MCP to loopback by default. A random API key is generated on first startup, stored with restrictive filesystem permissions, and can be rotated. Logs record only its fingerprint.
- Original search queries are not retained permanently by default. Diagnostic retention is short-lived, configurable and user-clearable.
- LLM, Embedding and Reranker providers are configured independently through OpenAI-compatible interfaces. Credentials live in protected platform configuration and never inside a memory library.
- One daemon manages all libraries, while each library has independent locks, queue ordering, indexes, Graph storage namespace, Git repository and health state.
- The primary installation path is a native Linux package workflow using `uv` or Python packaging. Docker is not the primary runtime because host filesystem, Git and Codex Hook access are core behaviors.
- Docker is used for reproducible acceptance testing. The acceptance environment mounts temporary memory directories and runs the real daemon, SQLite, Git operations, JiuwenMemory/Milvus Lite integration and deterministic fake model services.
- The project is licensed under Apache-2.0.

## Testing Decisions

- Tests assert externally visible behavior and durable artifacts rather than internal call order or private implementation structure.
- The primary testing seam is a Docker-orchestrated system boundary. A test starts the real daemon and its real persistence components, mounts isolated temporary project and memory directories, and interacts through the same Web/REST/MCP and Hook contracts used by clients.
- The Docker acceptance environment uses a deterministic fake OpenAI-compatible service for LLM, Embedding and Reranker behavior. Fixtures control extraction, conflicts, graph entities, ranking, failures, timeouts and malformed responses without calling an external paid service.
- Acceptance tests use real SQLite databases, real filesystem watching, real sidecar Git repositories and real Git commands. They verify file content, commit boundaries, ignored runtime state, restore commits and non-interference with a containing project repository.
- Graph acceptance tests use the actual JiuwenMemory Graph Memory adapter and local Milvus Lite storage. They verify source-version checks, one-hop and conditional two-hop expansion, provenance requirements and Markdown-only degradation.
- REST and MCP are tested as public contracts. Tests cover authentication, library registration, project binding, explicit user-library queries, unbound-project responses, candidate creation, retrieval metadata, synchronization state and token-budget enforcement.
- Codex Hook tests execute the actual Node.js adapters with representative JSON on stdin and inspect stdout, exit status, daemon requests and local spool files. Daemon outage, timeout, retry and duplicate-event cases must demonstrate fail-open behavior.
- Web acceptance tests use a browser against the running Docker environment. They cover library creation, existing-document adoption, platform editing, save diff, candidate approval, conflict resolution, Git history, deletion, restoration and out-of-band change reconciliation.
- Security acceptance tests place instruction-like content and secret fixtures in Markdown and Hook events. They verify that recalled content remains delimited data, secrets never enter durable capture or model requests, quarantined content is not searchable, and credentials are absent from memory Git history and logs.
- Retrieval acceptance tests use deterministic corpora containing exact identifiers, semantic paraphrases, code blocks, duplicate facts, superseded facts and graph relationships. They verify hybrid recall, structural chunking, reranking, source citations, current-versus-history behavior and dynamic budget allocation.
- Token-budget tests use the target tokenizer adapter and conservative fallback. They verify that encoded responses never exceed the requested budget or the absolute 10K ceiling, including metadata and graph results.
- Lifecycle tests cover Inbox retention, early cleanup, candidate expiry, pinned candidates, conflict retention, tombstone creation, blocked regeneration and explicit restoration.
- Concurrency tests run multiple simulated Codex sessions against one library and multiple libraries. They verify per-library serialization, cross-library independence, idempotency and atomic file replacement.
- Upgrade tests run against the pinned JiuwenMemory baseline and each proposed upgraded version. They verify existing Markdown import, sidecar metadata, vector rebuilding, Graph rebuilding, API contracts and Hook compatibility before the pin changes.
- A small set of focused unit tests is allowed for pure boundary algorithms whose exhaustive cases are clearer below the system seam, including token allocation, HMAC tombstone matching, secret redaction, path containment, ignore-rule evaluation and source-version validation.
- Unit tests must not duplicate acceptance assertions or couple to implementation-specific class structure.
- MVP completion requires the complete Docker acceptance flow: register an existing Markdown directory, bind a project, build direct and graph indexes, retrieve a source-attributed context package below 10K tokens from a new Codex session, capture a completed turn into a candidate, approve it through Web, observe one Git commit, then edit, delete and restore it through additional atomic commits.
- MVP completion also requires demonstrated degradation: stop model or graph dependencies and verify Markdown retrieval remains usable; stop the daemon and verify Codex Hook execution remains successful and bounded.

## Out of Scope

- Multi-user accounts, organizations, teams, shared approval workflows and tenant isolation beyond separate local libraries.
- Codex Cloud capture or any Hook execution that occurs on an OpenAI-managed remote worker.
- Automatic synchronization between machines.
- LAN or public-network service exposure.
- OAuth, OIDC, role-based access control or multiple API-key permission tiers.
- Automatic Git remote creation, pull, push, merge, rebase or history rewriting.
- Cloud-hosted database, vector or graph infrastructure as an MVP requirement.
- Windows and macOS support guarantees for the MVP.
- Mobile applications or native desktop applications.
- Automatic ingestion of non-Markdown file formats.
- Full repository source-code indexing as memory.
- Capture of hidden model reasoning, raw tool output, complete file bodies or unrestricted subagent traces.
- Treating Graph Memory, vector storage, SQLite or captured transcripts as an independent authoritative fact store.
- Silent adoption of direct external Markdown changes.
- Unrestricted authoritative memory mutation through MCP or autonomous agents.
- Docker as the primary end-user installation and runtime model.
- Automatic remote backup of memory content or Git history.

## Further Notes

- JiuwenMemory's current Graph Memory is independent from its main LongTermMemory message pipeline. The platform adapter must explicitly orchestrate authoritative Markdown projection and graph retrieval rather than assuming upstream integration.
- JiuwenMemory's existing Codex integration provides useful Hook and MCP prior art, but the platform requires broader capture semantics, local spooling, candidate governance, project binding and source-bounded graph expansion.
- The platform must preserve the distinction between readable files and supported mutation paths: Markdown is inspectable and portable, while edits are platform-managed to maintain provenance and derived consistency.
- The 10K limit applies only to the memory context package returned by this service, not to Codex's complete model context.
- The local Git history is an audit and recovery mechanism, not a synchronization mechanism. Runtime audit data remains necessary for actor identity, idempotency and background processing even though document revisions are represented by Git commits.
- Docker acceptance tests may use bind mounts or named temporary volumes, but every test run must start from isolated state and must not write into the developer's real memory libraries, project repositories or Codex configuration.
