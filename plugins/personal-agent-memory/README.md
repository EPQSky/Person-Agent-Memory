# Personal Agent Memory Codex plugin

Install this plugin once for the user. It never writes configuration into a project. It connects
Codex to the loopback-only Personal Agent Memory daemon through lifecycle Hooks and MCP.

Set `PERSONAL_AGENT_MEMORY_API_KEY` to the daemon API key, or leave it unset to read the default
`~/.local/share/personal-agent-memory/api-key` file. Installations with a custom state directory
discover the key location from `~/.config/personal-agent-memory/install.json` (or the matching
`XDG_CONFIG_HOME` path). `PERSONAL_AGENT_MEMORY_API_KEY_FILE` explicitly overrides that discovery.
Optional global settings are:

- `PERSONAL_AGENT_MEMORY_URL` (default `http://127.0.0.1:7331`)
- `PERSONAL_AGENT_MEMORY_TIMEOUT_MS` (default `2000`, clamped to `100`-`2000`)
- `PERSONAL_AGENT_MEMORY_TOKEN_BUDGET` (default `10000`, clamped to `512`-`10000`)
- `PERSONAL_AGENT_MEMORY_TARGET_MODEL` (default `gpt-4o-mini`)

The URL must use HTTP and a loopback host. Recall failures produce no context and always leave the
Codex lifecycle event unblocked.

Capture stores only the explicit user prompt and assistant final-message fields. It never reads the
transcript path or captures hidden reasoning, raw tool output, complete files, subagent traces, or
injected memory. Daemon outages use a private, bounded seven-day spool below `PLUGIN_DATA`; later
Hook runs replay valid records and quarantine malformed records without blocking Codex.
