#!/usr/bin/env bash
set -Eeuo pipefail

# Build deb and rpm from exactly one payload tree. Package lifecycle scripts only
# install/remove files; user setup is always a separate `setup` command.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PAYLOAD=""
OUTPUT=""
SOURCE_ROOT="$ROOT"

usage() {
  cat <<'USAGE'
Usage: scripts/build-packages.sh --payload DIR --output DIR [--source DIR]

Build .deb and .rpm from one self-contained payload directory.
The package manager scripts never access HOME, user systemd, Codex, or network.
USAGE
}
fail() { printf 'package build failed: %s\n' "$*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --payload) [[ $# -gt 1 ]] || fail "missing --payload value"; PAYLOAD=$2; shift 2 ;;
    --output) [[ $# -gt 1 ]] || fail "missing --output value"; OUTPUT=$2; shift 2 ;;
    --source) [[ $# -gt 1 ]] || fail "missing --source value"; SOURCE_ROOT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done
[[ -n "$PAYLOAD" && -n "$OUTPUT" ]] || fail "--payload and --output are required"
PAYLOAD=$(cd "$PAYLOAD" && pwd -P)
OUTPUT=$(mkdir -p "$OUTPUT" && cd "$OUTPUT" && pwd -P)
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd -P)
paths_overlap() {
  "$PYTHON" - "$1" "$2" <<'PY'
import os
import sys
first, second = (os.path.realpath(value) for value in sys.argv[1:])
try:
    print("true" if os.path.commonpath((first, second)) in {first, second} else "false")
except ValueError:
    print(False)
PY
}
PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null 2>&1 || fail "python is required"
if [[ "$(paths_overlap "$OUTPUT" "$PAYLOAD")" == true || "$(paths_overlap "$OUTPUT" "$SOURCE_ROOT")" == true ]]; then
  fail "output directory must not overlap payload or source"
fi
[[ -f "$PAYLOAD/manifest.json" ]] || fail "payload manifest.json is missing"
[[ -x "$PAYLOAD/opt/personal-agent-memory/bin/personal-agent-memory" ]] || fail "payload command is missing"
command -v dpkg-deb >/dev/null 2>&1 || fail "dpkg-deb is required"
command -v rpmbuild >/dev/null 2>&1 || fail "rpmbuild is required"
PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null 2>&1 || fail "python is required"
WORK=$(mktemp -d "$OUTPUT/.package-build.XXXXXX")
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
METADATA="$WORK/package-metadata.json"
"$PYTHON" "$ROOT/scripts/package-metadata.py" --source "$SOURCE_ROOT" --payload "$PAYLOAD" --output "$METADATA"
VERSION=$("$PYTHON" - "$METADATA" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["version"])
PY
)

make_root() {
  local root=$1
  mkdir -p "$root/opt" "$root/usr/bin" "$root/usr/share/doc/personal-agent-memory"
  cp -a "$PAYLOAD/opt/personal-agent-memory" "$root/opt/"
  install -m 0755 "$ROOT/packaging/personal-agent-memory-wrapper" "$root/usr/bin/personal-agent-memory"
  install -m 0644 "$METADATA" "$root/usr/share/doc/personal-agent-memory/package-metadata.json"
  install -m 0644 "$SOURCE_ROOT/README.md" "$root/usr/share/doc/personal-agent-memory/README.md"
  install -m 0644 "$SOURCE_ROOT/LICENSE" "$root/usr/share/doc/personal-agent-memory/LICENSE"
}

DEB_ROOT="$WORK/deb-root"
make_root "$DEB_ROOT"
mkdir -p "$DEB_ROOT/DEBIAN"
sed "s/@VERSION@/$VERSION/g" "$ROOT/packaging/deb/control.in" >"$DEB_ROOT/DEBIAN/control"
for script in postinst prerm postrm; do
  install -m 0755 "$ROOT/packaging/deb/$script" "$DEB_ROOT/DEBIAN/$script"
done
dpkg-deb --build --root-owner-group "$DEB_ROOT" "$OUTPUT/personal-agent-memory_${VERSION}_amd64.deb" >/dev/null

RPM_ROOT="$WORK/rpm-root"
make_root "$RPM_ROOT"
RPM_TOP="$WORK/rpmbuild"
mkdir -p "$RPM_TOP"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
tar -C "$RPM_ROOT" -czf "$RPM_TOP/SOURCES/root.tar.gz" .
sed "s/@VERSION@/$VERSION/g" "$ROOT/packaging/rpm/personal-agent-memory.spec.in" >"$RPM_TOP/SPECS/personal-agent-memory.spec"
rpmbuild --define "_topdir $RPM_TOP" -bb "$RPM_TOP/SPECS/personal-agent-memory.spec" >/dev/null
cp "$RPM_TOP/RPMS/x86_64/"*.rpm "$OUTPUT/"
printf 'Built packages in %s\n' "$OUTPUT"
