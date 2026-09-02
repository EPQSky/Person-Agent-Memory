from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
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

memory_root = Path("/memory-libraries")
existing = memory_root / "existing"
existing.mkdir(parents=True, exist_ok=True)
existing_note = existing / "adopted.md"
existing_text = "# Adopted\n\nExisting Markdown stays byte-for-byte.\n"
existing_note.write_text(existing_text, encoding="utf-8")

created_code, created = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(memory_root / "created"), "kind": "user"},
)
assert created_code == 201
assert Path(str(created["canonical_path"])).is_dir()
adopted_code, adopted = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(existing), "kind": "project"},
)
assert adopted_code == 201
assert existing_note.read_text(encoding="utf-8") == existing_text
duplicate_code, duplicate = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(existing / ".." / "existing"), "kind": "user"},
)
assert duplicate_code == 409
assert duplicate["detail"]["library_id"] == adopted["id"]
outside_code, outside = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": "/tmp/outside-library", "kind": "project"},
)
assert outside_code == 422
assert "allowed roots" in outside["detail"]
unwritable = memory_root / "unwritable"
unwritable.mkdir(exist_ok=True)
unwritable.chmod(stat.S_IRUSR | stat.S_IXUSR)
try:
    denied_code, denied = request(
        "http://127.0.0.1:7331/api/v1/libraries",
        key,
        {"path": str(unwritable), "kind": "project"},
    )
    assert denied_code == 422
    assert "writable" in denied["detail"]
finally:
    unwritable.chmod(stat.S_IRWXU)

listed_code, listed = request("http://127.0.0.1:7331/mcp/libraries", key)
assert listed_code == 200
assert {item["id"] for item in listed} == {created["id"], adopted["id"]}
assert key not in json.dumps(listed)

outside_replacement = Path("/tmp/outside-replacement")
outside_replacement.mkdir(exist_ok=True)
created_path = Path(str(created["canonical_path"]))
created_path.rmdir()
created_path.symlink_to(existing, target_is_directory=True)
replacement_code, replacement_list = request(
    "http://127.0.0.1:7331/api/v1/libraries", key
)
assert replacement_code == 200
replacement = next(item for item in replacement_list if item["id"] == created["id"])
assert replacement["availability"] == "unavailable"
adopted_health = next(item for item in replacement_list if item["id"] == adopted["id"])
assert adopted_health["availability"] == "available"
replacement_sync_code, replacement_sync = request(
    f"http://127.0.0.1:7331/mcp/libraries/{created['id']}/sync-status", key
)
assert replacement_sync_code == 200
assert replacement_sync["availability"] == "unavailable"
created_path.unlink()
created_path.symlink_to(outside_replacement, target_is_directory=True)
outside_replacement_code, outside_replacement_list = request(
    "http://127.0.0.1:7331/api/v1/libraries", key
)
assert outside_replacement_code == 200
outside_health = next(
    item for item in outside_replacement_list if item["id"] == created["id"]
)
assert outside_health["availability"] == "unavailable"

source_parent = memory_root / "source-parent"
source_path = source_parent / "project"
target_parent = memory_root / "target-parent"
target_path = target_parent / "project"
ancestor_source_code, ancestor_source = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(source_path), "kind": "project"},
)
ancestor_target_code, ancestor_target = request(
    "http://127.0.0.1:7331/api/v1/libraries",
    key,
    {"path": str(target_path), "kind": "project"},
)
assert ancestor_source_code == 201
assert ancestor_target_code == 201
source_path.rmdir()
source_parent.rmdir()
source_parent.symlink_to(target_parent, target_is_directory=True)
ancestor_code, ancestor_list = request(
    "http://127.0.0.1:7331/api/v1/libraries", key
)
assert ancestor_code == 200
ancestor_by_id = {item["id"]: item for item in ancestor_list}
assert ancestor_by_id[ancestor_source["id"]]["canonical_path"] == str(source_path)
assert ancestor_by_id[ancestor_source["id"]]["availability"] == "unavailable"
assert ancestor_by_id[ancestor_target["id"]]["canonical_path"] == str(target_path)
assert ancestor_by_id[ancestor_target["id"]]["availability"] == "available"
ancestor_sync_code, ancestor_sync = request(
    f"http://127.0.0.1:7331/mcp/libraries/{ancestor_source['id']}/sync-status", key
)
assert ancestor_sync_code == 200
assert ancestor_sync["availability"] == "unavailable"

# Exercise persistence across two real daemon processes inside the isolated acceptance container.
restart_state = Path("/tmp/pam-restart-state")
shutil.rmtree(restart_state, ignore_errors=True)
restart_path = memory_root / "restart-check"
restart_command = [
    "personal-agent-memory",
    "serve",
    "--state-dir",
    str(restart_state),
    "--library-root",
    str(memory_root),
    "--port",
    "17331",
]


def start_restart_daemon() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        restart_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (restart_state / "api-key").exists():
            restart_key = (restart_state / "api-key").read_text(encoding="utf-8").strip()
            try:
                if request("http://127.0.0.1:17331/api/v1/status", restart_key)[0] == 200:
                    return process
            except OSError:
                pass
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError("restart acceptance daemon did not become ready")


restart_process = start_restart_daemon()
restart_key = (restart_state / "api-key").read_text(encoding="utf-8").strip()
try:
    restart_code, restart_library = request(
        "http://127.0.0.1:17331/api/v1/libraries",
        restart_key,
        {"path": str(restart_path), "kind": "project"},
    )
    assert restart_code == 201
finally:
    restart_process.send_signal(signal.SIGINT)
    assert restart_process.wait(timeout=10) == 0

restart_process = start_restart_daemon()
try:
    restored_code, restored = request("http://127.0.0.1:17331/api/v1/libraries", restart_key)
    assert restored_code == 200
    assert restored[0]["id"] == restart_library["id"]
    assert restored[0]["kind"] == "project"
finally:
    restart_process.send_signal(signal.SIGINT)
    assert restart_process.wait(timeout=10) == 0

# Run a separate daemon as nobody so ancestor permission failures are effective
# even though the Compose acceptance container itself starts as root.
permission_state = Path("/tmp/pam-permission-state")
permission_root = Path("/tmp/pam-permission-libraries")
permission_parent = permission_root / "protected"
permission_library = permission_parent / "project"
for path in (permission_state, permission_library):
    path.mkdir(parents=True, exist_ok=True)
for path in (permission_state, permission_root, permission_parent, permission_library):
    os.chown(path, 65534, 65534)
    path.chmod(0o700)


def drop_to_nobody() -> None:
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)


permission_process = subprocess.Popen(
    [
        "personal-agent-memory",
        "serve",
        "--state-dir",
        str(permission_state),
        "--library-root",
        str(permission_root),
        "--port",
        "17332",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    preexec_fn=drop_to_nobody,
)
try:
    deadline = time.monotonic() + 10
    permission_key = ""
    while time.monotonic() < deadline:
        if (permission_state / "api-key").exists():
            permission_key = (permission_state / "api-key").read_text(
                encoding="utf-8"
            ).strip()
            try:
                if request(
                    "http://127.0.0.1:17332/api/v1/status", permission_key
                )[0] == 200:
                    break
            except OSError:
                pass
        time.sleep(0.1)
    else:
        raise RuntimeError("non-root permission daemon did not become ready")
    permission_code, permission_registered = request(
        "http://127.0.0.1:17332/api/v1/libraries",
        permission_key,
        {"path": str(permission_library), "kind": "project"},
    )
    assert permission_code == 201
    permission_parent.chmod(0o000)
    inaccessible_code, inaccessible = request(
        "http://127.0.0.1:17332/api/v1/libraries", permission_key
    )
    assert inaccessible_code == 200
    assert inaccessible[0]["id"] == permission_registered["id"]
    assert inaccessible[0]["availability"] == "unavailable"
    permission_parent.chmod(0o700)
    recovered_code, recovered = request(
        "http://127.0.0.1:17332/api/v1/libraries", permission_key
    )
    assert recovered_code == 200
    assert recovered[0]["id"] == permission_registered["id"]
    assert recovered[0]["availability"] == "available"
finally:
    permission_parent.chmod(0o700)
    permission_process.send_signal(signal.SIGINT)
    assert permission_process.wait(timeout=10) == 0

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
