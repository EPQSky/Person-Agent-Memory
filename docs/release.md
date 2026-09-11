# Personal Agent Memory Linux Release

This document describes the rootless Linux release lifecycle. The release is deliberately
downloadable and inspectable: the recommended flow is to download `install.sh`, `uninstall.sh`,
and the matching source archive from the same GitHub Release, review them, and then execute the
local files. Do not use `curl | sh` for a production installation.

## Support Matrix

Ubuntu 22.04 and Ubuntu 24.04 on x86_64 are the tested release targets. The same scripts are
intended to work on other mainstream x86_64 Linux distributions when a compatible user-level
`systemd`, Python 3.11-3.13, Git, Node.js, Codex CLI, and `uv` are available, but those
distributions are capability-compatibility targets rather than tested release claims. Alpine,
non-systemd environments, ARM, Windows, and macOS are outside this release.

The daemon is loopback-only and runs as the current user through `systemd --user`. Installation,
upgrade, and removal never require `sudo`, never invoke a system package manager, and never write
system-level service files. The service starts when the user logs in; the installer does not enable
systemd linger and does not promise operation while the user is logged out.

## Release Assets

Each release must publish these matching, independently reviewable assets:

- `install.sh` - rootless installer and in-place upgrade entry point.
- `uninstall.sh` - stable user-level uninstall entry point copied into the user's bin directory.
- A source archive for the same tag, including `pyproject.toml`, `uv.lock`, the plugin tree, and
  both scripts.

Review the downloaded scripts and confirm that the version in `pyproject.toml` matches the plugin
manifest before running the installer. Keep the source archive for auditing; the installed
uninstaller remains usable after the archive is removed.

## Automatic Installation

From a directory containing the reviewed release assets:

```bash
chmod +x install.sh uninstall.sh
./install.sh
```

The installer checks Linux, x86_64, Python, Git, Node.js, Codex, and a working user systemd bus.
If `uv` is missing, it bootstraps `uv` into the user's executable directory without using a
package manager or elevated privileges. A missing required dependency other than `uv` stops before
partial installation and prints the manual remediation. The installer creates the user service,
installs the release Python tool and global Codex plugin, starts the service, and waits for an
authenticated health check. It never opens a browser or prints the API key contents.

The default locations are:

- Platform state: `~/.local/share/personal-agent-memory`
- Memory libraries: `~/memory-libraries`
- Install metadata: `~/.config/personal-agent-memory/install.json`
- User service: `~/.config/systemd/user/personal-agent-memory.service`

Use explicit paths when required:

```bash
./install.sh \
  --state-dir "$HOME/data/personal-agent-memory" \
  --library-root "$HOME/documents/memory"
```

The paths are persisted in install metadata and are reused by upgrades and the stable uninstaller.

## Manual `uv` Installation

Advanced users can keep the existing package-first workflow:

```bash
uv tool install .
codex plugin marketplace add "$PWD"
codex plugin add personal-agent-memory@personal-agent-memory
```

This path is useful for development or a deliberately hand-managed service. It does not create
the release-managed user service or stable uninstaller; use the release installer when you want
the complete lifecycle contract.

## Supported scope

The MVP supports one local Linux user, one loopback-only daemon, multiple independently managed
Markdown memory libraries, explicit project bindings, source-attributed retrieval, bounded graph
expansion, Codex Hooks and MCP, governed candidate publication, local sidecar Git history, true
forgetting, restoration, and retention cleanup. Python 3.11-3.13, Git, Node.js, `uv`, and Codex are
required for the native installation.

## Service Management

After installation, manage the user service with:

```bash
systemctl --user status personal-agent-memory.service
systemctl --user start personal-agent-memory.service
systemctl --user stop personal-agent-memory.service
systemctl --user restart personal-agent-memory.service
journalctl --user -u personal-agent-memory.service --no-pager
```

The first start creates `api-key` below the platform state directory with mode `0600`. The plugin
uses that protected file by default. To inspect authenticated status manually, export the key for
the current shell only:

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(cat ~/.local/share/personal-agent-memory/api-key)"
```

Open `http://127.0.0.1:7331` yourself, enter the same key, and confirm the authenticated status is
ready. Installation never opens the browser automatically.

## Upgrade and Reinstall

Run the newer release's reviewed installer in place:

```bash
./install.sh
```

The installer stops the user service before replacing the Python tool, plugin, and service unit,
then starts the service again and performs an authenticated health check. Running the same release
installer again is an idempotent repair/reinstall. Both paths preserve the API key, platform
database, project bindings, candidates, audit records, capture state, and every authoritative
Markdown library. A failed upgrade leaves those data locations intact and reports the service
status and journal command for diagnosis.

## Uninstall

The installer copies a verified uninstaller to the user's executable directory. It can be run
without the original release directory:

```bash
personal-agent-memory-uninstall
```

Normal uninstall stops and removes the user service, Python tool, Codex plugin, and only the
installer-owned marketplace registration. It preserves platform state, the API key, install
metadata, and all memory libraries. To remove verified platform state as well:

```bash
personal-agent-memory-uninstall --purge
```

`--purge` removes the recorded platform state and install metadata, but **never** removes the
default or custom memory-library root or any registered library. If ownership cannot be proven,
the uninstaller preserves the path and reports why. This is intentional data protection.

## Release Verification

The single release acceptance entry point is:

```bash
PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 \
  ./scripts/verify-linux-installer.sh
```

Run it as a disposable ordinary user with a working user systemd bus. The same entry point covers
default and custom directories, missing-`uv` bootstrap behavior, same-version reinstall, a
cross-version upgrade, real Codex marketplace/plugin registration, a real Node.js Hook, authenticated
health, normal uninstall, `--purge`, data preservation, fail-open behavior, and unchanged Git
working-tree/remote state. The release gate is run on both Ubuntu 22.04 and Ubuntu 24.04 x86_64.

To make the bootstrap evidence explicit, run the entry point in a disposable user whose `uv` is
not on `PATH` and set:

```bash
PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 \
PAM_INSTALL_ACCEPTANCE_MISSING_UV=1 \
  ./scripts/verify-linux-installer.sh
```

The verifier records `missing uv bootstrap acceptance passed` after the old-release install has
bootstrapped a usable user-local `uv`; the normal and purge lifecycle checks then continue with the
same installation.

For a fast environment-only check:

```bash
PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 \
  ./scripts/verify-linux-installer.sh --check
```

The repository also retains the package-level verifier for the manual `uv` path:

```bash
./scripts/verify-native-install.sh
```

## Model configuration

Embedding, reranking, and graph extraction are independent OpenAI-compatible endpoints. Configure
each with its own URL, model name, protected API-key file, timeout, concurrency, and retry limit.
No paid provider is required: direct Markdown retrieval remains available without model endpoints,
and the acceptance suite uses only the deterministic local fake model.

## Backup Responsibility

The user is responsible for backing up both every authoritative Markdown library and the daemon
state directory. Markdown and each library's sidecar Git history contain the auditable content
history; daemon state contains bindings, candidates, capture state, tombstones, derived-index
metadata, and protected credentials. A Git remote is never configured or synchronized by the MVP.
Stop the daemon or use a filesystem snapshot mechanism that captures these locations consistently.

## Known Limitations

- Linux is the supported operating system; Windows and macOS are not release targets.
- The daemon is intentionally loopback-only and uses one local API key rather than user accounts.
- Vector and graph indexes are eventually consistent, rebuildable projections and may temporarily
  degrade to direct Markdown retrieval.
- External Markdown edits require explicit reconciliation and are not silently accepted.
- Backups, daemon supervision, upgrades, and credential export to the Codex process remain explicit
  user or operating-system responsibilities.

## Explicit Exclusions

Codex Cloud and other remote workers, remote synchronization, Git push or pull automation,
multi-user or team collaboration, LAN/public serving, OAuth/OIDC, mobile clients, and unrestricted
agent mutation of authoritative memory are outside the MVP. `.deb`, RPM, AppImage, Snap, Flatpak,
system-level services, and automatic background updates are also not part of this release.
