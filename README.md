# Personal Agent Memory

Local-first memory platform for a single user and local Codex sessions.

## Run

Python 3.11 or newer and `uv` are required.

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

## Verify

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

Docker acceptance uses a deterministic local OpenAI-compatible fake model and no paid service:

```bash
docker compose up --build --abort-on-container-exit --exit-code-from acceptance
docker compose down --volumes
```
