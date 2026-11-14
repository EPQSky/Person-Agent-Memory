#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
REPOSITORY=""
IMAGE=""
ENGINE="${CONTAINER_ENGINE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository) REPOSITORY=${2:?missing repository}; shift 2 ;;
    --image) IMAGE=${2:?missing image}; shift 2 ;;
    --engine) ENGINE=${2:?missing engine}; shift 2 ;;
    -h|--help) printf '%s\n' 'Usage: scripts/verify-offline-repository.sh --repository DIR --image IMAGE'; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$REPOSITORY" && -n "$IMAGE" ]] || { echo '--repository and --image are required' >&2; exit 2; }
[[ -f "$REPOSITORY/SHA256SUMS" && -f "$REPOSITORY/ASSET-MANIFEST.json" ]] || { echo 'repository manifest is incomplete' >&2; exit 2; }
(cd "$REPOSITORY" && sha256sum -c SHA256SUMS)
if [[ -z "$ENGINE" ]]; then
  if command -v docker >/dev/null 2>&1; then ENGINE=docker
  elif command -v podman >/dev/null 2>&1; then ENGINE=podman
  else echo 'SKIP: Docker or Podman is unavailable; offline install E2E was not run.' >&2; exit 77; fi
fi
command -v "$ENGINE" >/dev/null 2>&1 || { echo "container engine unavailable: $ENGINE" >&2; exit 77; }
REPOSITORY=$(cd "$REPOSITORY" && pwd -P)
"$ENGINE" run --rm --network none -v "$REPOSITORY:/opt/offline-repository:ro" "$IMAGE" sh -ceu '
  test -f /opt/offline-repository/ASSET-MANIFEST.json
  test -f /opt/offline-repository/apt/dists/stable/main/binary-amd64/Packages.gz
  test -d /opt/offline-repository/rpm/repodata
  printf "offline repository assets are available without network\n"
'
