#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PACKAGE_DIR=""
OUTPUT=""
usage() { printf '%s\n' 'Usage: scripts/build-offline-repository.sh --packages DIR --output DIR'; }
fail() { printf 'offline repository build failed: %s\n' "$*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --packages) PACKAGE_DIR=${2:?missing package directory}; shift 2 ;;
    --output) OUTPUT=${2:?missing output directory}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done
[[ -n "$PACKAGE_DIR" && -n "$OUTPUT" ]] || fail '--packages and --output are required'
PACKAGE_DIR=$(cd "$PACKAGE_DIR" && pwd -P)
OUTPUT=$(mkdir -p "$OUTPUT" && cd "$OUTPUT" && pwd -P)
PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null 2>&1 || fail 'python is required'
[[ "$OUTPUT" != "$PACKAGE_DIR" && "$OUTPUT" != "$PACKAGE_DIR"/* ]] || fail 'output must not be inside packages'
mapfile -t DEBS < <(find "$PACKAGE_DIR" -maxdepth 1 -type f -name '*.deb' -printf '%f\n' | LC_ALL=C sort)
mapfile -t RPMS < <(find "$PACKAGE_DIR" -maxdepth 1 -type f -name '*.rpm' -printf '%f\n' | LC_ALL=C sort)
((${#DEBS[@]} > 0)) || fail 'no deb packages found'
((${#RPMS[@]} > 0)) || fail 'no rpm packages found'
command -v dpkg-scanpackages >/dev/null 2>&1 || fail 'dpkg-scanpackages is required'
command -v createrepo_c >/dev/null 2>&1 || fail 'createrepo_c is required'
WORK=$(mktemp -d "$OUTPUT/.offline-repository.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/apt/pool/main" "$WORK/rpm"
for name in "${DEBS[@]}"; do cp -- "$PACKAGE_DIR/$name" "$WORK/apt/pool/main/$name"; done
for name in "${RPMS[@]}"; do cp -- "$PACKAGE_DIR/$name" "$WORK/rpm/$name"; done
mkdir -p "$WORK/apt/dists/stable/main/binary-amd64"
(
  cd "$WORK/apt"
  dpkg-scanpackages --multiversion pool/main /dev/null \
    | LC_ALL=C sort -k1,1 > dists/stable/main/binary-amd64/Packages
)
gzip -n -9 -c "$WORK/apt/dists/stable/main/binary-amd64/Packages" \
  > "$WORK/apt/dists/stable/main/binary-amd64/Packages.gz"
createrepo_c --database --unique-md-filenames "$WORK/rpm" >/dev/null
"$PYTHON" "$ROOT/scripts/create-offline-manifest.py" --root "$WORK" --output "$WORK/ASSET-MANIFEST.json"
mkdir -p "$OUTPUT"
rm -rf "$OUTPUT/apt" "$OUTPUT/rpm" "$OUTPUT/ASSET-MANIFEST.json" "$OUTPUT/SHA256SUMS"
cp -a "$WORK/apt" "$OUTPUT/apt"
cp -a "$WORK/rpm" "$OUTPUT/rpm"
cp "$WORK/ASSET-MANIFEST.json" "$OUTPUT/ASSET-MANIFEST.json"
(
  cd "$OUTPUT"
  find apt rpm -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
printf 'Offline repository written to %s\n' "$OUTPUT"
