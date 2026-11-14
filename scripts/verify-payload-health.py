from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

KEY_PATH = Path("/state/api-key")
BASE_URL = "http://127.0.0.1:7331"
DEADLINE_SECONDS = 60.0


def request(path: str) -> object:
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("payload daemon created an empty API key")
    request = urllib.request.Request(
        BASE_URL + path,
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


deadline = time.monotonic() + DEADLINE_SECONDS
last_error = "daemon did not create an API key"
while time.monotonic() < deadline:
    try:
        if request("/health/live") != {"status": "ok"}:
            raise RuntimeError("unexpected /health/live response")
        break
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        last_error = str(error)
        time.sleep(0.2)
else:
    raise SystemExit(f"payload daemon failed authenticated /health/live: {last_error}")

try:
    health = request("/mcp/health")
except (OSError, ValueError, urllib.error.URLError) as error:
    raise SystemExit(f"payload daemon failed authenticated /mcp/health: {error}") from error
if not isinstance(health, dict):
    raise SystemExit("payload daemon returned a non-object /mcp/health response")
if health.get("memory_kernel") != "mem2":
    kernel = health.get("memory_kernel")
    raise SystemExit(f"payload daemon selected unexpected memory kernel: {kernel!r}")
if health.get("memory_kernel_status") not in {"ready", "degraded"}:
    status = health.get("memory_kernel_status")
    raise SystemExit(f"payload daemon returned unusable memory kernel status: {status!r}")
