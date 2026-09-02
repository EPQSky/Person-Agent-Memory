from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(
    url: str, key: str | None = None, payload: dict[str, object] | None = None
) -> tuple[int, dict[str, object]]:
    headers = {} if key is None else {"Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    try:
        request_value = urllib.request.Request(url, headers=headers, data=data)
        with urllib.request.urlopen(request_value, timeout=2) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def wait_for(url: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if request(url)[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"service did not become ready: {url}")


wait_for("http://127.0.0.1:18080/health")

key = Path(os.environ["API_KEY_FILE"]).read_text(encoding="utf-8").strip()
assert request("http://127.0.0.1:7331/health/live", key)[0] == 200

with urllib.request.urlopen("http://127.0.0.1:7331/", timeout=2) as response:
    assert response.status == 200
    login_shell = response.read()
    assert b"Personal Agent Memory" in login_shell
    assert key.encode() not in login_shell
    fingerprint = f"sha256:{hashlib.sha256(key.encode()).hexdigest()[:12]}".encode()
    assert fingerprint not in login_shell
    assert b"localStorage" not in login_shell
    assert b"URLSearchParams" not in login_shell

assert request("http://127.0.0.1:7331/api/v1/status")[0] == 401
status_code, status = request("http://127.0.0.1:7331/api/v1/status", key)
assert status_code == 200
assert status["database"] == "ok"
assert status["background_worker"] == "running"
assert request("http://127.0.0.1:7331/mcp/health", key)[0] == 200
assert request("http://127.0.0.1:18080/v1/models")[1]["data"]
assert request(
    "http://127.0.0.1:18080/v1/chat/completions",
    payload={"messages": [{"role": "user", "content": "fixture"}]},
)[1]["choices"]
assert request("http://127.0.0.1:18080/v1/embeddings", payload={"input": ["fixture"]})[1][
    "data"
]
assert request(
    "http://127.0.0.1:18080/v1/rerank",
    payload={"query": "fixture", "documents": ["a", "longer"]},
)[1]["results"]
