# Personal Agent Memory MVP release

## Supported scope

The MVP supports one local Linux user, one loopback-only daemon, multiple independently managed
Markdown memory libraries, explicit project bindings, source-attributed retrieval, bounded graph
expansion, Codex Hooks and MCP, governed candidate publication, local sidecar Git history, true
forgetting, restoration, and retention cleanup. Python 3.11-3.13, Git, Node.js, `uv`, and Codex are
required for the native installation.

## Native installation

Install the daemon from a release checkout or source archive, then install the repository's local
Codex marketplace and plugin:

```bash
uv tool install .
codex plugin marketplace add "$PWD"
codex plugin add personal-agent-memory@personal-agent-memory
```

Start the daemon with explicit host directories:

```bash
personal-agent-memory serve \
  --state-dir ~/.local/share/personal-agent-memory \
  --library-root ~/memory-libraries
```

The first start creates `~/.local/share/personal-agent-memory/api-key` with mode `0600`. Export the
key before starting Codex so the plugin's authenticated MCP connection can use it:

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(cat ~/.local/share/personal-agent-memory/api-key)"
```

Open `http://127.0.0.1:7331`, enter the same key, and confirm the authenticated status is ready.
Run `./scripts/verify-native-install.sh` from a checkout to exercise package build, isolated `uv`
tool installation, global plugin installation, first daemon start, authenticated health, and a real
Node.js Hook without changing the caller's home or Codex configuration.

## Model configuration

Embedding, reranking, and graph extraction are independent OpenAI-compatible endpoints. Configure
each with its own URL, model name, protected API-key file, timeout, concurrency, and retry limit.
No paid provider is required: direct Markdown retrieval remains available without model endpoints,
and the acceptance suite uses only the deterministic local fake model.

## Backup responsibility

The user is responsible for backing up both every authoritative Markdown library and the daemon
state directory. Markdown and each library's sidecar Git history contain the auditable content
history; daemon state contains bindings, candidates, capture state, tombstones, derived-index
metadata, and protected credentials. A Git remote is never configured or synchronized by the MVP.
Stop the daemon or use a filesystem snapshot mechanism that captures these locations consistently.

## Known limitations

- Linux is the supported operating system; Windows and macOS are not release targets.
- The daemon is intentionally loopback-only and uses one local API key rather than user accounts.
- Vector and graph indexes are eventually consistent, rebuildable projections and may temporarily
  degrade to direct Markdown retrieval.
- External Markdown edits require explicit reconciliation and are not silently accepted.
- Backups, daemon supervision, upgrades, and credential export to the Codex process remain explicit
  user or operating-system responsibilities.

## Explicit exclusions

Codex Cloud and other remote workers, remote synchronization, Git push or pull automation,
multi-user or team collaboration, LAN/public serving, OAuth/OIDC, mobile clients, and unrestricted
agent mutation of authoritative memory are outside the MVP.
