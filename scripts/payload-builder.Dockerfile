# The base image is pinned by the release build invocation.  It is an ARG so
# the same builder can target a later supported distribution after review.
ARG BASE_IMAGE=ubuntu@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc
FROM ${BASE_IMAGE} AS builder

ARG RUNTIME_URL
ARG RUNTIME_SHA256
ARG RUNTIME_VERSION
ARG INSTALL_PREFIX=/opt/personal-agent-memory
ARG SERVICE_REGISTRATION=user-systemd
ARG UV_VERSION=0.8.0
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    VIRTUAL_ENV=/build/venv \
    UV_PROJECT_ENVIRONMENT=/build/venv \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl tar xz-utils unzip \
    && rm -rf /var/lib/apt/lists/*

# The runtime is downloaded only while building.  The digest is mandatory; a
# missing or mismatched digest aborts the build before any payload is emitted.
RUN test -n "$RUNTIME_URL" \
    && test -n "$RUNTIME_SHA256" \
    && curl --fail --location --retry 3 --proto '=https' --tlsv1.2 "$RUNTIME_URL" -o /tmp/cpython-runtime \
    && echo "$RUNTIME_SHA256  /tmp/cpython-runtime" | sha256sum --check --status \
    && mkdir -p /build/runtime \
    && tar -xf /tmp/cpython-runtime -C /build/runtime --strip-components=1 \
    && test -x /build/runtime/bin/python3 \
    && /build/runtime/bin/python3 --version \
    && rm -f /tmp/cpython-runtime

# A minor-only declaration (3.12) checks the supported minor, while a
# three-component declaration (3.12.8) checks the complete runtime version.
RUN /build/runtime/bin/python3 - "$RUNTIME_VERSION" <<'PY'
import sys
expected = sys.argv[1]
minor = f"{sys.version_info[0]}.{sys.version_info[1]}"
full = f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
actual = minor if expected.count(".") == 1 else full
if actual != expected:
    raise SystemExit(f"runtime version {full} does not match requested {expected}")
PY

# uv is a build-time resolver only.  It is never copied into the payload.
RUN curl --fail --location --retry 3 --proto '=https' --tlsv1.2 \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
      -o /tmp/uv.tar.gz \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu

WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY plugins ./plugins

# uv --frozen consumes uv.lock and verifies the lockfile hashes.  No dev
# dependencies enter the release environment.
RUN /build/runtime/bin/python3 - <<'PY'
import json
import pathlib
import tomllib

project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
manifest = json.loads(pathlib.Path("plugins/personal-agent-memory/.codex-plugin/plugin.json").read_text())
project_version = project.get("project", {}).get("version")
plugin_version = manifest.get("version")
if not isinstance(project_version, str) or not project_version:
    raise SystemExit("project version is missing")
if plugin_version != project_version:
    raise SystemExit(f"plugin version {plugin_version!r} does not match project version {project_version!r}")
if project.get("project", {}).get("requires-python") != ">=3.11,<3.14":
    raise SystemExit("release requires Python 3.11, 3.12, and 3.13")
PY
RUN uv sync --frozen --no-dev --no-editable --python /build/runtime/bin/python3

# Assemble an ordinary directory payload.  Nothing below this path is copied
# into the final runtime image except the self-contained prefix and metadata.
COPY scripts/payload-entrypoint.sh scripts/payload-service.template /tmp/
RUN set -eux; \
    case "$SERVICE_REGISTRATION" in none|user-systemd) ;; *) echo "invalid service registration" >&2; exit 1 ;; esac; \
    case "$INSTALL_PREFIX" in /*) ;; *) echo "install prefix must be absolute" >&2; exit 1 ;; esac; \
    case "$INSTALL_PREFIX" in *[!A-Za-z0-9._/-]*) echo "install prefix contains characters unsafe for payload templates" >&2; exit 1 ;; esac; \
    mkdir -p "/payload${INSTALL_PREFIX}/runtime" "/payload${INSTALL_PREFIX}/venv" "/payload${INSTALL_PREFIX}/bin" "/payload${INSTALL_PREFIX}/plugins" "/payload${INSTALL_PREFIX}/share"; \
    cp -a /build/runtime/. "/payload${INSTALL_PREFIX}/runtime/"; \
    cp -a /build/venv/. "/payload${INSTALL_PREFIX}/venv/"; \
    cp -a plugins/personal-agent-memory "/payload${INSTALL_PREFIX}/plugins/"; \
    sed -e "s|@INSTALL_PREFIX@|${INSTALL_PREFIX}|g" /tmp/payload-entrypoint.sh > "/payload${INSTALL_PREFIX}/bin/personal-agent-memory"; \
    chmod 0755 "/payload${INSTALL_PREFIX}/bin/personal-agent-memory"; \
    if [ "$SERVICE_REGISTRATION" = user-systemd ]; then \
      mkdir -p "/payload${INSTALL_PREFIX}/share/systemd/user"; \
      sed "s|@INSTALL_PREFIX@|${INSTALL_PREFIX}|g" /tmp/payload-service.template > "/payload${INSTALL_PREFIX}/share/systemd/user/personal-agent-memory.service"; \
    fi

COPY scripts/create-payload-manifest.py /tmp/create-payload-manifest.py
RUN /build/runtime/bin/python3 /tmp/create-payload-manifest.py "$INSTALL_PREFIX" "$RUNTIME_VERSION" "$SERVICE_REGISTRATION"

# The runtime image has a shell and file utilities for emission, but no
# system Python and no uv. The payload itself supplies its Python runtime.
ARG BASE_IMAGE=ubuntu@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc
FROM ${BASE_IMAGE} AS final
COPY --from=builder /payload/ /
COPY scripts/verify-payload-health.py /usr/local/bin/verify-payload-health.py
RUN printf '%s\n' \
  '#!/bin/sh' \
  'set -eu' \
  'mode=${1:?payload mode is required}' \
  'case "$mode" in' \
  '  --emit)' \
  '    out=${2:?output directory is required}' \
  '    requested_prefix=${3:?payload prefix is required}' \
  '    ;;' \
  '  --verify)' \
  '    requested_prefix=${2:?payload prefix is required}' \
  '    ;;' \
  '  *) echo "unknown payload mode: $mode" >&2; exit 2 ;;' \
  'esac' \
  'case "$requested_prefix" in /*) ;; *) echo "payload prefix must be absolute" >&2; exit 2 ;; esac' \
  'prefix=${requested_prefix#/}' \
  'case "$prefix" in ""|.*|*../*|*/../*|*/..|..*) echo "invalid payload prefix" >&2; exit 2 ;; esac' \
  'source_prefix="/$prefix"' \
  'test -x "$source_prefix/runtime/bin/python3"' \
  'test ! -e /usr/bin/python' \
  'test ! -e /usr/bin/python3' \
  'test ! -e /usr/local/bin/uv' \
  'if [ "$mode" = --emit ]; then' \
  '  target_prefix="$out/$prefix"' \
  '  mkdir -p "$target_prefix"' \
  '  cp -a "$source_prefix/." "$target_prefix/"' \
  '  cp /manifest.json "$out/manifest.json"' \
  'else' \
  '  cleanup() { kill "$daemon_pid" 2>/dev/null || true; wait "$daemon_pid" 2>/dev/null || true; }' \
  '  trap cleanup EXIT INT TERM' \
  '  "$source_prefix/bin/personal-agent-memory" serve --state-dir /state --library-root /memory-libraries --host 127.0.0.1 --port 7331 >/tmp/personal-agent-memory.log 2>&1 &' \
  '  daemon_pid=$!' \
  '  "$source_prefix/runtime/bin/python3" /usr/local/bin/verify-payload-health.py' \
  'fi' \
  > /usr/local/bin/emit-payload \
  && chmod 0755 /usr/local/bin/emit-payload
ENTRYPOINT ["/usr/local/bin/emit-payload"]
