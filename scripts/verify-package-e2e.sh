#!/usr/bin/env bash
set -Eeuo pipefail

# Run package acceptance in one clean container. The host must provide Docker
# or Podman and an image with the package manager, curl, and project runtime.
IMAGE=""
PACKAGE=""
ENGINE="${CONTAINER_ENGINE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE=${2:?missing image}; shift 2 ;;
    --package) PACKAGE=${2:?missing package}; shift 2 ;;
    --engine) ENGINE=${2:?missing engine}; shift 2 ;;
    -h|--help)
      printf '%s\n' 'Usage: scripts/verify-package-e2e.sh --image IMAGE --package FILE [--engine docker|podman]'
      exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$IMAGE" && -n "$PACKAGE" ]] || { echo '--image and --package are required' >&2; exit 2; }
[[ -f "$PACKAGE" ]] || { echo "package not found: $PACKAGE" >&2; exit 2; }
if [[ -z "$ENGINE" ]]; then
  if command -v docker >/dev/null 2>&1; then ENGINE=docker
  elif command -v podman >/dev/null 2>&1; then ENGINE=podman
  else
    echo 'SKIP: Docker or Podman is unavailable; clean-container E2E was not run.' >&2
    exit 77
  fi
fi
command -v "$ENGINE" >/dev/null 2>&1 || { echo "container engine unavailable: $ENGINE" >&2; exit 77; }
PACKAGE=$(cd "$(dirname "$PACKAGE")" && pwd -P)/$(basename "$PACKAGE")
case "$PACKAGE" in
  *.deb) INSTALL='dpkg -i /tmp/package' ;;
  *.rpm) INSTALL='rpm -U /tmp/package' ;;
  *) echo 'package must be .deb or .rpm' >&2; exit 2 ;;
esac
"$ENGINE" run --rm --network none -v "$PACKAGE:/tmp/package:ro" "$IMAGE" sh -ceu "
  $INSTALL
  command -v personal-agent-memory
  test -x /opt/personal-agent-memory/bin/personal-agent-memory
  state=/tmp/pam-state
  libraries=/tmp/pam-libraries
  port=7331
  personal-agent-memory setup --state-dir \"\$state\" --library-root \"\$libraries\" --port \"\$port\"
  key=\$(cat \"\$state/api-key\")
  base=http://127.0.0.1:\$port
  curl -fsS \"\$base/health/live\" | grep -q 'ok'
  curl -fsS -H \"Authorization: Bearer \$key\" \"\$base/mcp/health\"
  curl -fsS -X POST -H \"Authorization: Bearer \$key\" -H 'Content-Type: application/json' \"\$base/api/v1/regions\" -d '{\"kind\":\"project\",\"name\":\"Package E2E\",\"region_id\":\"package-e2e\"}'
  curl -fsS -X POST -H \"Authorization: Bearer \$key\" -H 'Content-Type: application/json' \"\$base/api/v1/memories\" -d '{\"region_id\":\"package-e2e\",\"content\":\"package e2e acceptance\",\"memory_type\":\"domain_fact\",\"source_references\":[\"package-e2e\"],\"actor\":\"package-e2e\",\"idempotency_key\":\"package-e2e-memory\"}'
  curl -fsS -X POST -H \"Authorization: Bearer \$key\" -H 'Content-Type: application/json' \"\$base/api/v1/search\" -d '{\"query\":\"package e2e acceptance\",\"cwd\":\"/tmp\",\"token_budget\":1000}' | grep -q 'package e2e acceptance'
  personal-agent-memory setup --uninstall
  test -f \"\$state/api-key\"
  test -d \"\$libraries\"
"
printf 'package container E2E passed\n'
