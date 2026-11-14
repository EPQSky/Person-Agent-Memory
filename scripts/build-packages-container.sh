#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
ENGINE=${PAM_CONTAINER_ENGINE:-docker}
IMAGE=${PAM_PACKAGE_BUILDER_IMAGE:-personal-agent-memory-package-builder:0.3.1}
PAYLOAD=""
OUTPUT=""
SOURCE="$ROOT"
usage() {
  printf '%s\n' 'Usage: scripts/build-packages-container.sh --payload DIR --output DIR [--source DIR]'
}
fail() { printf 'container package build failed: %s\n' "$*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --payload) PAYLOAD=${2:?missing payload}; shift 2 ;;
    --output) OUTPUT=${2:?missing output}; shift 2 ;;
    --source) SOURCE=${2:?missing source}; shift 2 ;;
    --engine) ENGINE=${2:?missing engine}; shift 2 ;;
    --image) IMAGE=${2:?missing image}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done
[[ -n "$PAYLOAD" && -n "$OUTPUT" ]] || fail '--payload and --output are required'
command -v "$ENGINE" >/dev/null 2>&1 || fail "container engine unavailable: $ENGINE"
PAYLOAD=$(cd "$PAYLOAD" && pwd -P)
OUTPUT=$(mkdir -p "$OUTPUT" && cd "$OUTPUT" && pwd -P)
SOURCE=$(cd "$SOURCE" && pwd -P)
case "$OUTPUT" in
  "$PAYLOAD"|"$PAYLOAD"/*|"$SOURCE"|"$SOURCE"/*) fail 'output must not overlap payload or source' ;;
esac
if ! "$ENGINE" image inspect "$IMAGE" >/dev/null 2>&1; then
  "$ENGINE" build --tag "$IMAGE" --file "$ROOT/scripts/package-builder.Dockerfile" "$ROOT"
fi
"$ENGINE" run --rm --network none \
  --volume "$SOURCE:/src:ro" \
  --volume "$PAYLOAD:/payload:ro" \
  --volume "$OUTPUT:/output" \
  "$IMAGE" \
  /src/scripts/build-packages.sh --source /src --payload /payload --output /output
