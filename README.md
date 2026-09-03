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
Existing Markdown is indexed at registration without rewriting the files. The scanner skips
version-control, dependency, build, cache, virtual-environment and platform-state directories.
Symbolic links and non-regular files are intentionally excluded. A Markdown file that cannot be
read completely as UTF-8, or any transient filesystem inspection failure, makes the scan fail
without removing prior index entries. Use the authenticated ignore-rule and scan endpoints to apply
library-specific exclusions and incrementally reconcile additions, changes and deletions:
`GET`/`PUT /api/v1/libraries/{library_id}/ignore-rules` and
`POST /api/v1/libraries/{library_id}/scan`.

Direct keyword search is available through `POST /api/v1/search`, `POST /mcp/search`, and the Web
interface. Both APIs accept `cwd`, `query`, and an optional `limit`; the daemon resolves `cwd`
through explicit project bindings and searches only that project's memory library. Unbound
working directories return an explicit `unbound` response with no results.

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
