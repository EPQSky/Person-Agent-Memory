#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_name="pam-acceptance-$(date +%s)-$$-$RANDOM"
child_pid=""

stop_child() {
  signal=$1
  if [[ -z "$child_pid" ]] || ! kill -0 "$child_pid" 2>/dev/null; then
    return
  fi
  kill -"$signal" "$child_pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$child_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$child_pid" 2>/dev/null; then
    kill -KILL "$child_pid" 2>/dev/null || true
  fi
  wait "$child_pid" 2>/dev/null || true
  child_pid=""
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  stop_child TERM
  docker compose -p "$project_name" down \
    --volumes \
    --remove-orphans \
    --rmi local >/dev/null 2>&1 || true
  exit "$status"
}

forward_signal() {
  signal=$1
  status=$2
  child_signal=$signal
  if [[ "$signal" == "INT" ]]; then
    child_signal=TERM
  fi
  trap - INT TERM
  stop_child "$child_signal"
  exit "$status"
}

trap cleanup EXIT
trap 'forward_signal INT 130' INT
trap 'forward_signal TERM 143' TERM

cd "$repo_root"
docker compose -p "$project_name" down \
  --volumes \
  --remove-orphans \
  --rmi local >/dev/null 2>&1 || true
docker compose -p "$project_name" up \
  --build \
  --abort-on-container-exit \
  --exit-code-from acceptance &
child_pid=$!
wait "$child_pid"
status=$?
child_pid=""
exit "$status"
