#!/usr/bin/env bash
set -Eeuo pipefail

DEB=""
RPM=""
ENGINE="${CONTAINER_ENGINE:-}"
IMAGES=("ubuntu:22.04" "ubuntu:24.04" "rockylinux:9" "fedora:latest")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deb) DEB=${2:?missing deb}; shift 2 ;;
    --rpm) RPM=${2:?missing rpm}; shift 2 ;;
    --engine) ENGINE=${2:?missing engine}; shift 2 ;;
    --image) IMAGES+=("${2:?missing image}"); shift 2 ;;
    -h|--help) printf '%s\n' 'Usage: scripts/verify-release-matrix.sh --deb FILE --rpm FILE [--engine docker|podman]'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$DEB" && -n "$RPM" ]] || { echo '--deb and --rpm are required' >&2; exit 2; }
[[ -f "$DEB" && -f "$RPM" ]] || { echo 'package artifact not found' >&2; exit 2; }
if [[ -z "$ENGINE" ]]; then
  if command -v docker >/dev/null 2>&1; then ENGINE=docker
  elif command -v podman >/dev/null 2>&1; then ENGINE=podman
  else echo 'SKIP: Docker or Podman is unavailable; release matrix was not run.' >&2; exit 77; fi
fi
command -v "$ENGINE" >/dev/null 2>&1 || { echo "container engine unavailable: $ENGINE" >&2; exit 77; }
DEB=$(cd "$(dirname "$DEB")" && pwd -P)/$(basename "$DEB")
RPM=$(cd "$(dirname "$RPM")" && pwd -P)/$(basename "$RPM")
for image in "${IMAGES[@]}"; do
  case "$image" in *ubuntu*) INSTALL='apt-get install -y /tmp/package' ;; *) INSTALL='dnf install -y /tmp/package.rpm' ;; esac
  printf 'release matrix: %s\n' "$image"
  "$ENGINE" run --rm --network none \
    -e HOME=/tmp/home -e XDG_CONFIG_HOME=/tmp/xdg/config \
    -e XDG_STATE_HOME=/tmp/xdg/state -e XDG_DATA_HOME=/tmp/xdg/data \
    -v "$DEB:/tmp/package:ro" -v "$RPM:/tmp/package.rpm:ro" \
    "$image" sh -ceu "
      mkdir -p /tmp/home /tmp/xdg/config /tmp/xdg/state /tmp/xdg/data
      if command -v apt-get >/dev/null 2>&1; then apt-get update -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false >/dev/null 2>&1 || true; $INSTALL; else $INSTALL; fi
      command -v personal-agent-memory
      test -x /opt/personal-agent-memory/bin/personal-agent-memory
      state=/tmp/xdg/state/personal-agent-memory
      library=/tmp/xdg/data/personal-agent-memory-libraries
      personal-agent-memory setup --state-dir \"\$state\" --library-root \"\$library\" --port 7331
      key=\$(cat \"\$state/api-key\")
      curl -fsS -H \"Authorization: Bearer \$key\" http://127.0.0.1:7331/health/live
      test ! -x /usr/local/bin/personal-agent-memory-legacy || ! personal-agent-memory setup --state-dir \"\$state\" --library-root \"\$library\" --port 7331
      personal-agent-memory setup --uninstall
      test -f \"\$state/api-key\"
      test -d \"\$library\"
    "
done
printf 'release matrix passed\n'
