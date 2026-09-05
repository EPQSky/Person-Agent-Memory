# Personal Agent Memory Codex plugin

Install this plugin once for the user. It never writes configuration into a project. It connects
Codex to the loopback-only Personal Agent Memory daemon through lifecycle Hooks and MCP.

Set `PERSONAL_AGENT_MEMORY_API_KEY` to the daemon API key, or leave it unset to read the default
`~/.local/share/personal-agent-memory/api-key` file. Optional global settings are:

- `PERSONAL_AGENT_MEMORY_URL` (default `http://127.0.0.1:7331`)
- `PERSONAL_AGENT_MEMORY_TIMEOUT_MS` (default `2000`, clamped to `100`-`2000`)
- `PERSONAL_AGENT_MEMORY_TOKEN_BUDGET` (default `10000`, clamped to `512`-`10000`)
- `PERSONAL_AGENT_MEMORY_TARGET_MODEL` (default `gpt-4o-mini`)

The URL must use HTTP and a loopback host. Recall failures produce no context and always leave the
Codex lifecycle event unblocked.
