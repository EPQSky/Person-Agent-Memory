# 自动记忆晋升与管理台会话

## Problem Statement

当前会话记忆流程虽然已经具备模型判断和自动晋升能力，但产品文档容易让用户误以为每条记忆都必须进入候选审核。实际需要的是“自动晋升优先、人工审核兜底”：明确、稳定、可追溯且低风险的长期内容应由模型判断后直接进入正式记忆，只有低置信度、冲突、敏感、重复、重大影响或模型异常的内容才需要人工介入。

同时，模型调用没有显式的生成参数。候选判断和图谱抽取请求未设置 `temperature`、`top_p`、`top_k` 或有界输出上限，行为依赖 OpenAI 兼容服务的默认值，不利于结构化抽取的稳定性和问题复现。

管理台目前只在页面内存中保存用户输入的 API key。刷新页面后凭据丢失，用户必须重新输入。对于单用户本地平台，应在认证成功后提供有限时长的本地会话，同时避免把长期 API key 保存到浏览器脚本可访问的存储中。

## Solution

建立统一的“模型判断、规则校验、自动晋升、人工兜底”流程。模型先判断当前会话是否包含可长期保存的内容；不符合条件的会话不创建候选。符合条件的内容生成候选记录，并在通过类型、用户证据、来源、置信度、冲突、重复、敏感信息和策略检查后自动发布为正式记忆。只有无法安全自动发布的候选保留在人工审核队列中。

为候选判断和图谱抽取增加任务级模型配置档案，默认采用确定性优先的生成策略：`temperature=0`、`top_p=1`、不强制发送供应商特有的 `top_k`，并设置有界输出上限。嵌入和重排序不使用生成采样参数。

管理台在 API key 认证成功后签发固定有效期为 24 小时的本地会话。会话通过 HttpOnly、同源限制的 Cookie 传递，服务端在用户级状态目录中保存可验证的会话状态。刷新页面、服务重启或普通页面导航不会要求重新输入 API key；显式退出、API key 轮换或会话过期会要求重新认证。

## User Stories

1. As a user, I want the model to decide whether a completed Codex interaction contains durable information, so that ordinary task chatter does not become memory.
2. As a user, I want ineligible conversations to produce no candidate memory, so that the candidate queue remains focused.
3. As a user, I want clear decisions, constraints, preferences, domain facts, reusable experiences and external references to be promoted automatically, so that routine memory growth does not require constant approval.
4. As a user, I want automatic promotion to retain the original session and project provenance, so that convenient automation remains auditable.
5. As a user, I want assistant-only claims to be blocked from automatic publication, so that model confidence cannot turn an unsupported answer into a user fact.
6. As a user, I want natural user statements to count as evidence without requiring fixed words such as “remember” or “confirmed”, so that ordinary conversation is not penalized by keyword matching.
7. As a user, I want questions, speculation and ambiguous proposals to remain outside automatic publication, so that the system does not mistake exploration for a decision.
8. As a user, I want low-confidence extraction results to enter manual review, so that uncertain content can be corrected before publication.
9. As a user, I want conflicting candidates to remain pending while the existing formal memory stays effective, so that unresolved changes do not alter Codex behavior.
10. As a user, I want possible duplicates to be held for provenance augmentation or governance, so that equivalent facts do not create redundant Markdown.
11. As a user, I want sensitive candidates to be withheld from automatic publication and ordinary retrieval, so that credentials and private data do not become memory.
12. As a user, I want high-impact or policy-sensitive candidates to require manual approval, so that consequential memory changes remain under my control.
13. As a user, I want model extraction failures to fail closed for publication but fail open for Codex, so that memory errors cannot create false facts or block my coding session.
14. As a user, I want the management interface to show only exceptional pending candidates as requiring attention, so that an empty or small queue reflects normal operation.
15. As a user, I want automatically published memories to remain visible in history and audit records, so that automatic behavior is inspectable.
16. As a user, I want candidate status to distinguish pending, approved and rejected outcomes, so that I can understand whether a memory was automated or manually governed.
17. As a maintainer, I want candidate extraction to use deterministic sampling parameters, so that structured JSON behavior is repeatable across equivalent requests.
18. As a maintainer, I want graph extraction and candidate extraction to have separate task profiles, so that their output budgets can match their different workloads.
19. As a maintainer, I want unsupported provider-specific parameters omitted from requests, so that the service remains compatible with OpenAI-compatible endpoints.
20. As a user, I want embedding and reranking requests to remain free of generation-only parameters, so that their contracts stay compatible with their respective APIs.
21. As a user, I want to enter the platform API key once in the management interface, so that I can work without repeated credential prompts.
22. As a user, I want a successful login to create a 24-hour local session, so that refreshing the management interface does not log me out.
23. As a user, I want the browser session credential to be inaccessible to page JavaScript, so that the platform does not store the long-lived API key in local or session storage.
24. As a user, I want the session to survive a memory daemon restart during its valid period, so that routine service restarts do not interrupt management work.
25. As a user, I want an explicit logout action to invalidate the current session, so that I can end access immediately.
26. As a user, I want API key rotation to invalidate existing management sessions, so that compromised credentials can be contained.
27. As a user, I want an expired session to return a clear authentication state, so that I know when re-authentication is required.
28. As a user, I want the session to remain limited to the local management origin, so that the MVP does not become a remote multi-user authentication system.
29. As a maintainer, I want session records to be bounded and cleanable, so that expired browser sessions do not accumulate indefinitely.
30. As a user, I want the Chinese management documentation to explain automatic promotion and 24-hour login accurately, so that operating instructions match actual behavior.

## Implementation Decisions

- The capture pipeline remains asynchronous and fail-open for Codex. User prompts and assistant final replies are captured as separate events and consolidated into one model extraction round.
- The extraction model returns a structured eligibility result. An ineligible result ends the round without creating a candidate memory.
- An eligible result creates a candidate record first, preserving the governance and audit boundary even when the candidate is subsequently auto-promoted.
- Automatic promotion is the default publication path for eligible, low-risk content. It requires:
  - an allowed formal memory type;
  - semantically clear user evidence;
  - valid capture provenance for both user and assistant events;
  - model-reported source validity;
  - no unresolved conflict or duplicate relationship;
  - policy permission;
  - no sensitive content;
  - model confidence of at least `0.90`.
- User evidence is semantic rather than keyword-only. A user message must make a clear assertion or choice and must not be only a question. Assistant output alone cannot satisfy the evidence requirement, and extracted content must not expand beyond the user-supported object or scope.
- Pending manual-review candidates are the exception path for low confidence, incomplete evidence or provenance, conflicts, possible duplicates, sensitive content, high-impact policy cases and model or persistence anomalies.
- Automatically approved candidates remain queryable as approved records and produce the existing atomic memory mutation, provenance, operation audit and Git history.
- Candidate extraction and graph extraction use task-level model profiles. The default deterministic profile sends `temperature=0` and `top_p=1`, omits provider-specific `top_k`, and applies a bounded output limit appropriate to the task.
- Embedding and reranking payloads do not receive generation sampling parameters. Provider-specific options are optional and are omitted when unsupported.
- Existing model endpoint configuration remains independently configurable for embedding, reranking and graph/candidate extraction. The graph model remains the model used for graph extraction and candidate extraction unless a later decision adds a separate candidate model.
- The current model timeout, concurrency and retry controls remain separate transport controls and are not treated as sampling parameters.
- The daemon adds a management-session authentication path alongside the existing API key authentication path. API key authentication remains available for Codex Hooks, MCP, REST clients and initial browser login.
- Successful browser login creates a server-verifiable session with a fixed absolute lifetime of 24 hours. The session is persisted in the user-level platform state so a daemon restart does not invalidate a still-valid session.
- The browser receives the session through an HttpOnly, same-origin-restricted cookie. The API key is never written to `localStorage`, `sessionStorage`, page markup, URLs or normal application state after login.
- Cookie-authenticated requests are accepted by the Web and REST surfaces that are used by the management interface. Bearer API key authentication remains supported for non-browser clients.
- Logout removes or invalidates the current session. API key rotation invalidates all sessions associated with the previous key generation. Expired sessions are removed during authenticated requests or bounded cleanup.
- The management interface shows authentication expiry and connection state without exposing session material or the API key.
- The Chinese README and management interface copy describe the normal automatic-promotion path and identify manual review as an exception workflow.

## Testing Decisions

- Tests assert public behavior and durable artifacts, not private helper call order or exact implementation structure.
- Backend tests use the existing `TestClient` seam with isolated SQLite state, temporary memory libraries, project bindings and deterministic model responses. This seam verifies model payload behavior, extraction eligibility, auto-promotion, pending exceptions, provenance, publication, audit and Git effects.
- Model-client contract tests use a local OpenAI-compatible fake service to inspect outgoing JSON. They verify deterministic sampling parameters for graph and candidate extraction, bounded output configuration, omission of `top_k` when not configured, and absence of generation parameters from embedding and reranking requests.
- Session-capture tests extend the existing capture governance coverage to verify:
  - ineligible extraction creates no candidate;
  - a valid high-confidence user-supported result is automatically approved;
  - low confidence, assistant-only evidence, conflict, duplicate, sensitive and policy-blocked results remain pending;
  - automatic approval preserves provenance and creates the expected memory mutation history.
- Browser acceptance tests use the existing local web-server browser seam. They verify first login, absence of API key in page storage and URLs, cookie-backed access after refresh, access after a daemon restart, expiry after 24 hours, explicit logout, and invalidation after API key rotation.
- Authentication tests verify that an invalid or expired session receives the existing unauthenticated response and that bearer API key clients continue to work.
- Time-dependent tests use the project's controllable clock or an equivalent deterministic session-expiry boundary rather than sleeping for a real day.
- Documentation checks verify that the README describes automatic promotion as the default and manual review as an exception, and that the 24-hour browser session behavior is documented.
- Regression tests must preserve existing requirements for secret quarantine, source-bounded memory, true forgetting, atomic Git history, fail-open Hooks and local-only service binding.

## Out of Scope

- Multi-user accounts, organizations, roles, OAuth, OIDC, remote login or network-facing session management.
- Storing the API key in browser `localStorage`, `sessionStorage`, URL parameters or page-visible state.
- Changing the single-user loopback security boundary.
- Allowing assistant-only claims, sensitive content or unresolved conflicts to bypass governance.
- Automatically publishing task progress, temporary plans, one-off output or unverified guesses.
- Adding a user-facing control panel for arbitrary provider-specific sampling parameters in this change.
- Introducing a separate candidate-extraction model endpoint; the existing graph model endpoint remains the extraction model.
- Sliding session expiry, “remember me” sessions longer than 24 hours, or cross-device session synchronization.
- Changing the authoritative Markdown, sidecar Git, Graph Memory provenance or 10K-token retrieval architecture.

## Further Notes

- The existing implementation already contains an automatic-promotion path, but the user-facing documentation must stop implying that every eligible memory requires manual approval.
- The current browser implementation keeps the API key only in page memory and therefore intentionally loses access on refresh. The new session path must improve usability without making the API key persistent in browser-readable storage.
- A candidate can be created and immediately become approved; the candidate record remains useful for provenance, audit and governance history even when it is not shown in the pending queue.
- The first implementation should keep the 0.90 confidence threshold and hard safety gates explicit. Threshold tuning should be based on observed acceptance data rather than silently weakening evidence or conflict checks.
