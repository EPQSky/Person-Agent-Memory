#!/usr/bin/env bash
set -euo pipefail

service_name="personal-agent-memory.service"
web_url="http://127.0.0.1:7331"
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
state_dir="${HOME}/.local/share/personal-agent-memory"
library_root="${HOME}/memory-libraries"
config_home="${XDG_CONFIG_HOME:-${HOME}/.config}"
codex_home="${CODEX_HOME:-${HOME}/.codex}"
unit_dir="${config_home}/systemd/user"
unit_path="${unit_dir}/${service_name}"
failed_stage="initialization"

on_error() {
  status=$?
  trap - ERR
  printf 'Installation failed during %s.\n' "$failed_stage" >&2
  printf 'Service status: systemctl --user status %s\n' "$service_name" >&2
  printf 'Service logs: journalctl --user -u %s --no-pager\n' "$service_name" >&2
  exit "$status"
}
trap on_error ERR

quote_unit_argument() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '"%s"' "$value"
}

failed_stage="directory preparation"
mkdir -p "$state_dir" "$library_root" "$unit_dir" "$codex_home"
chmod 700 "$state_dir"
chmod 700 "$codex_home"

failed_stage="Python tool installation"
uv tool install --force --from "$source_root" personal-agent-memory
daemon_command="$(uv tool dir --bin)/personal-agent-memory"
[[ -x "$daemon_command" ]]

failed_stage="Codex plugin installation"
codex plugin marketplace add "$source_root" --json >/dev/null
codex plugin add personal-agent-memory@personal-agent-memory --json >/dev/null

failed_stage="user service installation"
daemon_unit_command="$(quote_unit_argument "$daemon_command")"
state_unit_argument="$(quote_unit_argument "$state_dir")"
library_unit_argument="$(quote_unit_argument "$library_root")"
cat >"$unit_path" <<EOF
[Unit]
Description=Personal Agent Memory daemon
After=network.target

[Service]
Type=simple
ExecStart=${daemon_unit_command} serve --state-dir ${state_unit_argument} --library-root ${library_unit_argument} --host "127.0.0.1" --port "7331"
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$service_name"

failed_stage="authenticated health check"
python3 - "$state_dir/api-key" "$web_url/health/live" <<'PY'
import pathlib
import sys
import time
import urllib.error
import urllib.request

key_path = pathlib.Path(sys.argv[1])
health_url = sys.argv[2]
deadline = time.monotonic() + 30
last_error = "API key was not created"
while time.monotonic() < deadline:
    try:
        key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise OSError("API key is empty")
        request = urllib.request.Request(
            health_url,
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            if response.status == 200:
                break
            last_error = f"health endpoint returned {response.status}"
    except (OSError, urllib.error.URLError) as error:
        last_error = str(error)
    time.sleep(0.1)
else:
    raise SystemExit(f"daemon did not become healthy: {last_error}")
PY

failed_stage="service status check"
service_status="$(systemctl --user is-active "$service_name")"
[[ "$service_status" == "active" ]]

trap - ERR
cat <<EOF
Personal Agent Memory installation complete.
Service status: ${service_status}
Web: ${web_url}
API key file: ${state_dir}/api-key

Start:   systemctl --user start ${service_name}
Stop:    systemctl --user stop ${service_name}
Restart: systemctl --user restart ${service_name}
Status:  systemctl --user status ${service_name}
Logs:    journalctl --user -u ${service_name} --no-pager
EOF
