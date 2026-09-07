# Personal Agent Memory

Local-first memory platform for a single user and local Codex sessions.

## Run

Python 3.11 or newer, Git, and `uv` are required.

```bash
uv sync --all-extras
uv run personal-agent-memory serve \
  --state-dir ~/.local/share/personal-agent-memory \
  --library-root ~/memory-libraries
```

Open `http://127.0.0.1:7331` to load the public login shell, then enter the API key
to retrieve protected service status. The login shell contains no service status or
key material. The first startup writes the generated API key to
`~/.local/share/personal-agent-memory/api-key` with mode `0600`; the key is kept only
in the page's current input value and request header. The daemon rejects non-loopback
listen addresses. Repeat `--library-root` to allow additional local directory trees;
registration resolves symlinks and rejects paths outside those explicit boundaries.
Existing Markdown is indexed at registration without rewriting the files. The scanner skips
version-control, dependency, build, cache, virtual-environment and platform-state directories.
Symbolic links and non-regular files are intentionally excluded. A Markdown file that cannot be
read completely as UTF-8, or any transient filesystem inspection failure, makes the scan fail
without removing prior index entries. Use the authenticated ignore-rule and scan endpoints to apply
library-specific exclusions and incrementally reconcile additions, changes and deletions:
`GET`/`PUT /api/v1/libraries/{library_id}/ignore-rules` and
`POST /api/v1/libraries/{library_id}/scan`.

Release-style native installation, global Codex plugin setup, model configuration, backup
responsibilities, and MVP limitations are documented in `docs/release.md`.

Direct keyword search is available through `POST /api/v1/search`, `POST /mcp/search`, and the Web
interface. Both APIs accept `cwd`, `query`, and an optional `limit`; the daemon resolves `cwd`
through explicit project bindings and searches only that project's memory library. Unbound
working directories return an explicit `unbound` response with no results.

The Web interface can browse indexed Markdown, edit source, render a sandboxed preview, inspect a
save diff, and audit or restore local history. Equivalent authenticated REST endpoints live under
`/api/v1/libraries/{library_id}/documents`, `/document`, and `/history`. Every edit or restore needs
an expected source version and an idempotent operation identifier, replaces one document atomically,
reindexes it, and creates one local commit with the fixed `Personal Agent Memory` service identity.
Actor type and operation source are recorded separately in platform state.

By default each library uses a bare sidecar repository below the platform state directory, with the
memory directory only as its work tree; no nested `.git`, remote, pull, or push is created. An existing
dedicated repository is reused only when registration explicitly sends `reuse_existing_git: true`.
History commits are built with an isolated temporary Git index and track only Markdown, the portable
library manifest, and reserved tombstone paths, so a containing project repository's index, branch,
history, and unrelated working tree remain untouched. MCP exposes search and status but no direct
document edit or history-restore operation.

MCP clients can create candidate memories with `POST /mcp/candidates` and inspect them with
`GET /mcp/candidates` or `GET /mcp/candidates/{candidate_id}`. A candidate carries a stable ID,
allowed memory type, Markdown body, provenance references, creator, timestamp, and governance
status, but it is excluded from text, vector, graph, and Codex retrieval. The Web interface is the
only governance surface: it records an operator and reason when editing, approving, or rejecting.
Approval publishes provenance-bearing Markdown through the same coordinator used for authoritative
edits, refreshes direct search immediately, and creates one idempotent Git commit. Rejection keeps
the SQLite audit record without creating any authoritative or derived projection.

Embedding and reranking are independently optional OpenAI-compatible services configured with
`--embedding-url` and `--reranker-url`. API keys are read only from the corresponding
`--embedding-api-key-file` and `--reranker-api-key-file`; they are not stored in a library, Git, or
response payload. Vector data is a rebuildable SQLite projection. A scan invalidates vectors for
changed chunks and schedules a bounded background rebuild. Search merges and deduplicates full-text
and semantic candidates, while embedding failures retain full-text results and reranker failures
retain the pre-rerank order with an explicit degradation marker. Model calls have configurable
timeouts, concurrency limits, and at most three retries.

Graph projection uses the pinned `JiuwenMemory==0.1.2` graph object contract through a
project-owned adapter and stores each library in its own local Milvus Lite file under platform
state. Configure its independent OpenAI-compatible extraction model with `--graph-url`,
`--graph-model`, and optionally `--graph-api-key-file`. Published Markdown is projected
asynchronously; every graph row carries the authoritative document identifier and source version.
Graph expansion begins only after direct Markdown hits, uses one hop by default, and accepts at
most two hops. Search validates every expanded result against the current SQLite Markdown index,
so edits and deletions invalidate stale graph data immediately even while rebuilding. Graph or LLM
failure leaves direct Markdown retrieval available and reports `graph_unavailable`.

Search returns a `memory-context-package/v1` response from both `POST /api/v1/search` and
`POST /mcp/search`. Callers select exactly one scope: a project `cwd`, or an explicit
`library_id` (required for user libraries). The server counts the complete JSON package with the
requested `target_model` tokenizer, falls back to conservative UTF-8 byte counting when that
tokenizer is unavailable, and enforces both `token_budget` and the absolute 10,000-token ceiling.
The `budget.telemetry_accounting` field identifies the counting convention. Under
`conservative_fixed_width_v1`, telemetry integers are normalized to fixed-width upper-bound
placeholders before counting, so `used_tokens` is a stable conservative upper bound for the final
serialized package rather than a self-referential exact count; the server separately verifies that
the actual final JSON also fits the effective budget.
Direct Markdown receives at least 60% of available result capacity, graph expansion receives at
most 30%, and source/structure metadata receives at most 10%; unused graph capacity is returned to
direct hits. Every result carries source and degradation metadata, and the package marks recalled
content as untrusted data with no policy, tool-authorization, or command semantics.

## Codex plugin

The globally installable plugin lives at `plugins/personal-agent-memory`. Install it once for the
user and set `PERSONAL_AGENT_MEMORY_API_KEY` to the daemon key (or use the default protected key
file). Codex then discovers lifecycle Hooks and the authenticated loopback MCP connection without
writing files into individual projects. `UserPromptSubmit` recalls only the project memory library
bound to the Hook event's `cwd`; unbound projects and daemon failures are silent and fail open.
The same global plugin captures only the explicit user prompt and assistant final-message fields;
it never reads transcript files, hidden reasoning, raw tool output, full file bodies, subagent traces,
or recalled memory context. Events enter an idempotent SQLite Inbox and are consolidated by the
daemon's background worker into governed candidates. During daemon outages the Hook uses a private,
bounded seven-day spool below `PLUGIN_DATA`, then replays valid records and quarantines malformed
records on a later Hook invocation without blocking Codex.

## Verify

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

Docker acceptance uses a deterministic local OpenAI-compatible fake model and no paid service:

```bash
./scripts/verify-docker.sh
```

The verifier uses a unique Compose project for every run and removes its containers,
networks, and named volumes on success, failure, or interruption.

Native package and global plugin installation use a separate isolated acceptance check. It builds
and installs the wheel, installs the Codex plugin, starts the daemon, and executes a real Hook
without changing the current user's home directory or Codex configuration:

```bash
./scripts/verify-native-install.sh
```
