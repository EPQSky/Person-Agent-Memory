#!/usr/bin/env bash
set -Eeuo pipefail

# Build the release payload in a fixed, isolated container. The checkout is
# mounted read-only and all generated files are written below --output.
BASE_IMAGE="${PAM_BUILD_BASE_IMAGE:-ubuntu@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc}"
RUNTIME_URL="${PAM_RUNTIME_URL:-}"
RUNTIME_SHA256="${PAM_RUNTIME_SHA256:-}"
RUNTIME_VERSION="${PAM_RUNTIME_VERSION:-3.12}"
INSTALL_PREFIX="${PAM_INSTALL_PREFIX:-/opt/personal-agent-memory}"
SERVICE_REGISTRATION="${PAM_SERVICE_REGISTRATION:-user-systemd}"
CONTAINER_ENGINE="${PAM_CONTAINER_ENGINE:-docker}"
OUTPUT=""
IMAGE="personal-agent-memory-payload-builder"
KEEP_IMAGE=false

usage() {
  cat <<'USAGE'
Usage: scripts/build-payload.sh --output DIR [options]

Build a self-contained Personal Agent Memory payload in a fixed container.
The source checkout is mounted read-only. Only DIR is created or changed.

Required:
  --output DIR             Output directory (outside the source tree is recommended).

Runtime options (also accepted through PAM_* environment variables):
  --runtime-url URL        CPython standalone archive URL (required).
  --runtime-sha256 HASH    SHA-256 for the runtime archive (required).
  --runtime-version VER    Runtime version (default: 3.12).
  --install-prefix PATH    Payload prefix (default: /opt/personal-agent-memory).
  --service-registration M none or user-systemd (default: user-systemd).
  --base-image IMAGE       Digest-pinned builder image (default: Ubuntu 22.04 digest).
  --engine COMMAND         docker or podman (default: docker).
  --image NAME             Temporary builder image name.
  --keep-image             Keep the temporary builder image.
  -h, --help               Show this help.
USAGE
}

fail() { printf 'payload build failed: %s\n' "$*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) [[ $# -ge 2 ]] || fail "missing value for --output"; OUTPUT=$2; shift 2 ;;
    --runtime-url) [[ $# -ge 2 ]] || fail "missing value for --runtime-url"; RUNTIME_URL=$2; shift 2 ;;
    --runtime-sha256) [[ $# -ge 2 ]] || fail "missing value for --runtime-sha256"; RUNTIME_SHA256=$2; shift 2 ;;
    --runtime-version) [[ $# -ge 2 ]] || fail "missing value for --runtime-version"; RUNTIME_VERSION=$2; shift 2 ;;
    --install-prefix) [[ $# -ge 2 ]] || fail "missing value for --install-prefix"; INSTALL_PREFIX=$2; shift 2 ;;
    --service-registration) [[ $# -ge 2 ]] || fail "missing value for --service-registration"; SERVICE_REGISTRATION=$2; shift 2 ;;
    --base-image) [[ $# -ge 2 ]] || fail "missing value for --base-image"; BASE_IMAGE=$2; shift 2 ;;
    --engine) [[ $# -ge 2 ]] || fail "missing value for --engine"; CONTAINER_ENGINE=$2; shift 2 ;;
    --image) [[ $# -ge 2 ]] || fail "missing value for --image"; IMAGE=$2; shift 2 ;;
    --keep-image) KEEP_IMAGE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -n "$OUTPUT" ]] || fail "--output is required"
[[ -n "$RUNTIME_URL" ]] || fail "--runtime-url (or PAM_RUNTIME_URL) is required"
[[ "$RUNTIME_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || fail "--runtime-sha256 must be a 64-character hexadecimal digest"
[[ "$RUNTIME_VERSION" =~ ^3\.(11|12|13)(\.[0-9]+)?$ ]] || fail "runtime version must be Python 3.11, 3.12, or 3.13"
[[ "$INSTALL_PREFIX" == /* && "$INSTALL_PREFIX" != *$'\n'* && "$INSTALL_PREFIX" != *$'\r'* ]] || fail "install prefix must be an absolute path without line breaks"
case "$SERVICE_REGISTRATION" in none|user-systemd) ;; *) fail "service registration must be none or user-systemd" ;; esac
[[ "$BASE_IMAGE" =~ ^[A-Za-z0-9./:_-]+@sha256:[0-9a-fA-F]{64}$ ]] || fail "base image must include an immutable @sha256: digest (mutable tags are rejected)"
[[ "$IMAGE" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid builder image name"

if [[ "$OUTPUT" != /* ]]; then OUTPUT="$PWD/$OUTPUT"; fi
[[ "$OUTPUT" != *,* ]] || fail "output directory must not contain commas (container mount syntax)"
INSTALL_PREFIX="${INSTALL_PREFIX%/}"
[[ "$INSTALL_PREFIX" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "install prefix may contain only letters, digits, '.', '_', '-', and '/'"
case "$INSTALL_PREFIX" in */../*|*/..|../*|..|*/./*|*/.) fail "install prefix must not contain dot path components" ;; esac
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -f "$SOURCE_ROOT/pyproject.toml" && -f "$SOURCE_ROOT/uv.lock" ]] || fail "pyproject.toml and uv.lock are required"
command -v "$CONTAINER_ENGINE" >/dev/null 2>&1 || fail "container engine not found: $CONTAINER_ENGINE"

assert_no_symlink_components() {
  local path=$1
  while [[ "$path" != / ]]; do
    [[ ! -L "$path" ]] || fail "output path must not contain symlink components: $path"
    path=$(dirname -- "$path")
  done
}
assert_no_symlink_components "$OUTPUT"
output_parent=$(dirname -- "$OUTPUT")
output_name=$(basename -- "$OUTPUT")
[[ -d "$output_parent" ]] || fail "output parent directory does not exist: $output_parent"
output_parent=$(cd "$output_parent" && pwd -P)
if [[ "$output_parent" == / && "$output_name" == / ]]; then
  OUTPUT_TARGET=/
else
  OUTPUT_TARGET="$output_parent/$output_name"
fi
case "$OUTPUT_TARGET" in
  /|//|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/media|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var)
    fail "refusing to use a dangerous system directory as output"
    ;;
esac
[[ "$OUTPUT_TARGET" != "$SOURCE_ROOT" && "$OUTPUT_TARGET" != "$SOURCE_ROOT"/* ]] || fail "output must not be inside the source checkout"
[[ ! -L "$OUTPUT_TARGET" ]] || fail "output must not be a symbolic link"
if [[ -e "$OUTPUT_TARGET" ]]; then
  [[ -d "$OUTPUT_TARGET" ]] || fail "existing output must be a directory"
  [[ -z "$(find "$OUTPUT_TARGET" -mindepth 1 -print -quit 2>/dev/null)" ]] || fail "existing output directory must be empty"
fi
IMAGE_TAG="$IMAGE"
IMAGE_CREATED=false
if "$CONTAINER_ENGINE" image inspect "$IMAGE" >/dev/null 2>&1; then
  IMAGE_TAG="${IMAGE}-payload-build-${BASHPID:-$$}-${RANDOM}"
  if "$CONTAINER_ENGINE" image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    fail "temporary builder image tag already exists: $IMAGE_TAG"
  fi
fi
IMAGE_CREATED=true
STAGING=$(mktemp -d "$output_parent/.personal-agent-memory-payload.XXXXXX")
VERIFY_HOME=""
cleanup_image() {
  if [[ "$KEEP_IMAGE" != true && "$IMAGE_CREATED" == true ]]; then
    "$CONTAINER_ENGINE" image rm "$IMAGE_TAG" >/dev/null 2>&1 || true
  fi
}
cleanup_all() {
  [[ -z "$VERIFY_HOME" ]] || rm -rf "$VERIFY_HOME"
  [[ -z "$STAGING" ]] || rm -rf "$STAGING"
  cleanup_image
}
trap cleanup_all EXIT

"$CONTAINER_ENGINE" build \
  --file "$SOURCE_ROOT/scripts/payload-builder.Dockerfile" \
  --tag "$IMAGE_TAG" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "RUNTIME_URL=$RUNTIME_URL" \
  --build-arg "RUNTIME_SHA256=$RUNTIME_SHA256" \
  --build-arg "RUNTIME_VERSION=$RUNTIME_VERSION" \
  --build-arg "INSTALL_PREFIX=$INSTALL_PREFIX" \
  --build-arg "SERVICE_REGISTRATION=$SERVICE_REGISTRATION" \
  "$SOURCE_ROOT"

"$CONTAINER_ENGINE" run --rm \
  --network none \
  --read-only \
  --mount "type=bind,src=$STAGING,dst=/output" \
  "$IMAGE_TAG" --emit /output "$INSTALL_PREFIX"

"$CONTAINER_ENGINE" run --rm \
  --network none \
  --read-only \
  --tmpfs /state \
  --tmpfs /memory-libraries \
  --tmpfs /tmp \
  "$IMAGE_TAG" --verify "$INSTALL_PREFIX"

[[ -f "$STAGING/manifest.json" ]] || fail "builder did not produce manifest.json"
[[ -d "$STAGING$INSTALL_PREFIX" ]] || fail "builder did not produce the configured payload prefix"
[[ -x "$STAGING$INSTALL_PREFIX/bin/personal-agent-memory" ]] || fail "payload entrypoint is missing"
PLUGIN_MANIFEST="$STAGING$INSTALL_PREFIX/plugins/personal-agent-memory/.codex-plugin/plugin.json"
[[ -f "$PLUGIN_MANIFEST" ]] || fail "payload Codex plugin is missing"
"$STAGING$INSTALL_PREFIX/runtime/bin/python3" - "$STAGING/manifest.json" "$PLUGIN_MANIFEST" <<'PY'
import json
import pathlib
import sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
plugin = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
versions = [manifest.get("project_version"), manifest.get("plugin_version"), plugin.get("version")]
if not all(isinstance(version, str) and version.strip() for version in versions):
    raise SystemExit("payload manifest and plugin version fields must be non-empty strings")
if len(set(versions)) != 1:
    raise SystemExit(f"payload project/plugin versions do not match: {versions!r}")
PY
VERIFY_HOME=$(mktemp -d)
if ! env -i HOME="$VERIFY_HOME" PATH="" "$STAGING$INSTALL_PREFIX/bin/personal-agent-memory" --version >/dev/null; then
  fail "payload did not start with an empty PATH (system Python and uv unavailable)"
fi
rm -rf "$VERIFY_HOME"
VERIFY_HOME=""
if [[ -e "$OUTPUT_TARGET" ]]; then
  rmdir "$OUTPUT_TARGET" || fail "output directory changed while payload was being built"
fi
mv "$STAGING" "$OUTPUT_TARGET"
STAGING=""
OUTPUT="$OUTPUT_TARGET"
printf 'Payload written to %s\n' "$OUTPUT"
