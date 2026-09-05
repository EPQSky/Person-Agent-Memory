from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.model_client import ModelServiceError

ROOT = Path(__file__).parents[1]
CAPTURE_HOOK = ROOT / "plugins" / "personal-agent-memory" / "scripts" / "capture.mjs"


def _bound_client(tmp_path: Path) -> tuple[TestClient, dict[str, str], str, Path]:
    state_dir = tmp_path / "state"
    libraries = tmp_path / "libraries"
    project = tmp_path / "project"
    libraries.mkdir()
    project.mkdir()
    client = TestClient(create_app(Settings(state_dir=state_dir, library_roots=(libraries,))))
    client.__enter__()
    key = (state_dir / "api-key").read_text().strip()
    headers = {"Authorization": f"Bearer {key}"}
    library = client.post(
        "/api/v1/libraries",
        headers=headers,
        json={"path": str(libraries / "project-memory"), "kind": "project"},
    ).json()
    response = client.post(
        "/api/v1/project-bindings",
        headers=headers,
        json={"project_root": str(project), "library_id": library["id"]},
    )
    assert response.status_code == 201
    return client, headers, str(library["id"]), project


def _event(project: Path, kind: str, content: str, event_id: str) -> dict[str, str]:
    return {
        "event_id": event_id,
        "session_id": "session-capture-1",
        "turn_id": "turn-1",
        "event_kind": kind,
        "content": content,
        "occurred_at": "2026-09-05T10:00:00Z",
        "cwd": str(project),
    }


def test_capture_inbox_is_idempotent_out_of_order_and_asynchronous(tmp_path: Path) -> None:
    client, headers, library_id, project = _bound_client(tmp_path)
    try:
        state = client.app.state.platform_state
        state.model_client.extract_candidate = lambda conversation: {
            "eligible": True,
            "suggested_type": "decision",
            "body": "# Decision\n\nUse the captured architecture.",
        }
        assistant = _event(project, "assistant", "The architecture is confirmed.", "event-a")
        user = _event(project, "user", "Remember this architecture decision.", "event-u")
        first = client.post("/api/v1/capture/events", headers=headers, json=assistant)
        duplicate = client.post("/api/v1/capture/events", headers=headers, json=assistant)
        second = client.post("/api/v1/capture/events", headers=headers, json=user)
        assert first.status_code == duplicate.status_code == second.status_code == 202
        assert first.json()["duplicate"] is False
        assert duplicate.json()["duplicate"] is True

        deadline = time.monotonic() + 3
        candidates: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            candidates = client.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
            if candidates:
                break
            time.sleep(0.05)
        assert len(candidates) == 1
        assert candidates[0]["creator"] == "session-capture"
        references = candidates[0]["source_references"]
        assert isinstance(references, list)
        assert len(references) == 2
        assert all("session-capture-1:turn-1" in str(value) for value in references)
        with sqlite3.connect(tmp_path / "state" / "platform.sqlite3") as connection:
            assert connection.execute("SELECT COUNT(*) FROM capture_inbox").fetchone() == (2,)
            assert connection.execute("SELECT status FROM capture_rounds").fetchone() == ("done",)
    finally:
        client.__exit__(None, None, None)


def test_interrupted_turn_stays_durable_without_lifecycle_event(tmp_path: Path) -> None:
    client, headers, _, project = _bound_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/capture/events",
            headers=headers,
            json=_event(project, "user", "Unfinished but retained prompt", "interrupted-u"),
        )
        assert response.status_code == 202
        time.sleep(0.2)
        with sqlite3.connect(tmp_path / "state" / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT content, consolidated_at FROM capture_inbox"
            ).fetchone() == ("Unfinished but retained prompt", None)
            assert connection.execute("SELECT status FROM capture_rounds").fetchone() == (
                "pending",
            )
    finally:
        client.__exit__(None, None, None)


def test_transient_extraction_retries_after_restart_without_duplicate_candidate(
    tmp_path: Path,
) -> None:
    client, headers, library_id, project = _bound_client(tmp_path)
    calls = 0

    def one_shot_failure(conversation: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelServiceError("one-shot extraction failure")
        return {
            "eligible": True,
            "suggested_type": "decision",
            "body": "# Recovered decision\n\nRetry completed after restart.",
        }

    client.app.state.platform_state.model_client.extract_candidate = one_shot_failure
    try:
        for event in (
            _event(project, "user", "Retry this durable round.", "restart-u"),
            _event(project, "assistant", "The retry decision is confirmed.", "restart-a"),
        ):
            response = client.post("/api/v1/capture/events", headers=headers, json=event)
            assert response.status_code == 202
        database = tmp_path / "state" / "platform.sqlite3"
        deadline = time.monotonic() + 2
        attempts = 0
        while time.monotonic() < deadline:
            with sqlite3.connect(database) as connection:
                row = connection.execute(
                    """SELECT attempts, status,
                              available_at - (julianday('now') - 2440587.5) * 86400.0
                       FROM background_jobs
                       WHERE kind = 'capture_consolidation' ORDER BY id DESC LIMIT 1"""
                ).fetchone()
            if (
                row is not None
                and int(row[0]) == 1
                and row[1] == "pending"
                and float(row[2]) > 0.1
            ):
                attempts = 1
                break
            time.sleep(0.02)
        assert attempts == 1
    finally:
        client.__exit__(None, None, None)

    settings = Settings(
        state_dir=tmp_path / "state", library_roots=(tmp_path / "libraries",)
    )
    restarted_app = create_app(settings)
    restarted_app.state.platform_state.model_client.extract_candidate = one_shot_failure
    with TestClient(restarted_app) as restarted:
        deadline = time.monotonic() + 3
        candidates: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            candidates = restarted.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
            if candidates:
                break
            time.sleep(0.05)
        assert len(candidates) == 1
        time.sleep(0.7)
        assert len(
            restarted.get(
                "/api/v1/candidates", headers=headers, params={"library_id": library_id}
            ).json()
        ) == 1
        assert calls == 2
        with sqlite3.connect(tmp_path / "state" / "platform.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM background_jobs WHERE kind = 'capture_consolidation'"
            ).fetchone() == (1,)


class CaptureHandler(BaseHTTPRequestHandler):
    events: list[dict[str, object]] = []
    response_delay = 0.0
    permanent_contents: set[str] = set()

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(size))
        if self.path.endswith("/events"):
            type(self).events.append(payload)
        time.sleep(type(self).response_delay)
        status = 422 if payload.get("content") in type(self).permanent_contents else 202
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def capture_server(
    response_delay: float = 0.0, permanent_contents: set[str] | None = None
) -> Iterator[str]:
    CaptureHandler.events = []
    CaptureHandler.response_delay = response_delay
    CaptureHandler.permanent_contents = permanent_contents or set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        CaptureHandler.response_delay = 0.0
        CaptureHandler.permanent_contents = set()


def _run_hook(
    event: dict[str, object], url: str, plugin_data: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(CAPTURE_HOOK)],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        timeout=4,
        env={
            **os.environ,
            "PERSONAL_AGENT_MEMORY_URL": url,
            "PERSONAL_AGENT_MEMORY_API_KEY": "capture-test-key",
            "PERSONAL_AGENT_MEMORY_TIMEOUT_MS": "100",
            "PLUGIN_DATA": str(plugin_data),
        },
        check=False,
    )


def test_real_hook_spools_replays_and_quarantines_without_transcript_capture(
    tmp_path: Path,
) -> None:
    plugin_data = tmp_path / "plugin-data"
    event = {
        "session_id": "hook-session",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Allowed user prompt",
        "transcript_path": "/private/full-transcript.jsonl",
        "hidden_reasoning": "must not persist",
        "tool_output": "must not persist",
        "memory_context": "must not persist",
        "timestamp": "2026-09-05T10:00:00Z",
    }
    stopped = _run_hook(event, "http://127.0.0.1:1", plugin_data)
    assert stopped.returncode == 0
    assert stopped.stdout == stopped.stderr == ""
    spool = plugin_data / "capture" / "spool"
    files = list(spool.glob("event-*.json"))
    assert len(files) == 1
    assert files[0].stat().st_mode & 0o077 == 0
    stored = files[0].read_text()
    assert "Allowed user prompt" in stored
    assert "full-transcript" not in stored
    assert "hidden_reasoning" not in stored
    (spool / "event-bad.json").write_text("not json")

    with capture_server() as url:
        replay = _run_hook(
            {**event, "prompt": "Next allowed prompt", "timestamp": "2026-09-05T10:01:00Z"},
            url,
            plugin_data,
        )
    assert replay.returncode == 0
    assert len(CaptureHandler.events) == 2
    assert not list(spool.glob("event-*.json"))
    assert (plugin_data / "capture" / "quarantine" / "event-bad.json").is_file()


def test_real_hook_spools_behind_fresh_maintenance_lock_and_replays_after_takeover(
    tmp_path: Path,
) -> None:
    plugin_data = tmp_path / "plugin-data"
    spool = plugin_data / "capture" / "spool"
    lock = spool / ".maintenance-lock"
    lock.mkdir(parents=True)
    event = {
        "session_id": "locked-spool-session",
        "turn_id": "locked-spool-turn",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Persist even while maintenance is locked.",
        "timestamp": "2026-09-05T10:01:30Z",
    }

    stopped = _run_hook(event, "http://127.0.0.1:1", plugin_data)

    assert stopped.returncode == 0
    assert stopped.stdout == stopped.stderr == ""
    files = list(spool.glob("event-*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["content"] == event["prompt"]

    expired = time.time() - 6
    os.utime(lock, (expired, expired))
    with capture_server() as url:
        replay = _run_hook(
            {**event, "prompt": "Replay after stale-lock takeover."},
            url,
            plugin_data,
        )

    assert replay.returncode == 0
    assert {captured["content"] for captured in CaptureHandler.events} == {
        "Persist even while maintenance is locked.",
        "Replay after stale-lock takeover.",
    }
    assert not list(spool.glob("event-*.json"))


def test_assistant_stop_returns_valid_advisory_json(tmp_path: Path) -> None:
    with capture_server() as url:
        result = _run_hook(
            {
                "session_id": "assistant-stop",
                "turn_id": "turn-stop",
                "cwd": str(tmp_path),
                "hook_event_name": "Stop",
                "last_assistant_message": "Final reply only",
                "timestamp": "2026-09-05T10:02:30Z",
            },
            url,
            tmp_path / "plugin-data",
        )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {}
    assert result.stderr == ""


def test_official_hook_payload_without_timestamp_uses_iso_event_time(tmp_path: Path) -> None:
    with capture_server() as url:
        result = _run_hook(
            {
                "session_id": "official-payload",
                "transcript_path": "/private/transcript.jsonl",
                "cwd": str(tmp_path),
                "hook_event_name": "UserPromptSubmit",
                "prompt": "No timestamp field is present.",
            },
            url,
            tmp_path / "plugin-data",
        )
    assert result.returncode == 0
    assert len(CaptureHandler.events) == 1
    occurred_at = CaptureHandler.events[0]["occurred_at"]
    assert isinstance(occurred_at, str)
    assert occurred_at.endswith("Z")
    assert "hook-time-unavailable" not in occurred_at


def test_current_event_precedes_bounded_slow_replay(tmp_path: Path) -> None:
    plugin_data = tmp_path / "plugin-data"
    spool = plugin_data / "capture" / "spool"
    spool.mkdir(parents=True)
    for index in range(20):
        event = _event(tmp_path, "user", f"Backlog {index}", f"backlog-{index}")
        (spool / f"event-{index:064x}.json").write_text(json.dumps(event))

    started = time.monotonic()
    with capture_server(response_delay=0.07) as url:
        result = _run_hook(
            {
                "session_id": "priority-session",
                "cwd": str(tmp_path),
                "hook_event_name": "UserPromptSubmit",
                "prompt": "Current event must be delivered first.",
            },
            url,
            plugin_data,
        )
        elapsed = time.monotonic() - started

    assert result.returncode == 0
    assert CaptureHandler.events[0]["content"] == "Current event must be delivered first."
    assert elapsed < 0.8
    assert list(spool.glob("event-*.json"))


def test_identical_official_prompts_get_distinct_turns_and_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    plugin_data = tmp_path / "plugin-data"
    prompt = {
        "session_id": "distinct-turn-session",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "The same legitimate prompt.",
    }
    with capture_server() as url:
        first = _run_hook(prompt, url, plugin_data)
        second = _run_hook(prompt, url, plugin_data)
        stop = _run_hook(
            {
                "session_id": prompt["session_id"],
                "cwd": str(tmp_path),
                "hook_event_name": "Stop",
                "last_assistant_message": "Assistant reply for the active turn.",
            },
            url,
            plugin_data,
        )

    assert first.returncode == second.returncode == stop.returncode == 0
    first_event, second_event, stop_event = CaptureHandler.events
    assert first_event["turn_id"] != second_event["turn_id"]
    assert first_event["event_id"] != second_event["event_id"]
    assert stop_event["turn_id"] == second_event["turn_id"]

    spooled = _run_hook(
        {**prompt, "prompt": "Replay this transport exactly once."},
        "http://127.0.0.1:1",
        plugin_data,
    )
    assert spooled.returncode == 0
    spool = plugin_data / "capture" / "spool"
    stored = json.loads(next(spool.glob("event-*.json")).read_text())
    with capture_server() as url:
        for _ in range(2):
            replay = _run_hook(
                {
                    "session_id": "distinct-turn-session",
                    "cwd": str(tmp_path),
                    "hook_event_name": "PreCompact",
                },
                url,
                plugin_data,
            )
            assert replay.returncode == 0
    replayed = [event for event in CaptureHandler.events if event["event_id"] == stored["event_id"]]
    assert len(replayed) == 1
    assert replayed[0]["turn_id"] == stored["turn_id"]


def test_stop_retry_after_new_prompt_reuses_completed_identity(tmp_path: Path) -> None:
    plugin_data = tmp_path / "plugin-data"
    common = {"session_id": "stop-retry-session", "cwd": str(tmp_path)}
    with capture_server() as url:
        prompt1 = _run_hook(
            {**common, "hook_event_name": "UserPromptSubmit", "prompt": "Prompt one"},
            url,
            plugin_data,
        )
        stop1 = _run_hook(
            {**common, "hook_event_name": "Stop", "last_assistant_message": "Reply one"},
            url,
            plugin_data,
        )
        prompt2 = _run_hook(
            {**common, "hook_event_name": "UserPromptSubmit", "prompt": "Prompt two"},
            url,
            plugin_data,
        )
        retry1 = _run_hook(
            {**common, "hook_event_name": "Stop", "last_assistant_message": "Reply one"},
            url,
            plugin_data,
        )
        stop2 = _run_hook(
            {**common, "hook_event_name": "Stop", "last_assistant_message": "Reply two"},
            url,
            plugin_data,
        )

    assert all(result.returncode == 0 for result in (prompt1, stop1, prompt2, retry1, stop2))
    first_prompt, first_stop, second_prompt, retried_stop, second_stop = CaptureHandler.events
    assert first_stop["turn_id"] == first_prompt["turn_id"]
    assert retried_stop["turn_id"] == first_stop["turn_id"]
    assert retried_stop["event_id"] == first_stop["event_id"]
    assert second_stop["turn_id"] == second_prompt["turn_id"]
    assert second_stop["event_id"] != first_stop["event_id"]


def test_permanent_replay_rejection_is_quarantined_and_next_record_continues(
    tmp_path: Path,
) -> None:
    plugin_data = tmp_path / "plugin-data"
    spool = plugin_data / "capture" / "spool"
    spool.mkdir(parents=True)
    rejected = _event(tmp_path, "user", "Reject permanently", "reject-422")
    accepted = _event(tmp_path, "user", "Accept after rejection", "accept-next")
    rejected_path = spool / f"event-{'0' * 64}.json"
    accepted_path = spool / f"event-{'1' * 64}.json"
    rejected_path.write_text(json.dumps(rejected))
    accepted_path.write_text(json.dumps(accepted))

    with capture_server(permanent_contents={"Reject permanently"}) as url:
        replay = _run_hook(
            {
                "session_id": "validation-replay",
                "cwd": str(tmp_path),
                "hook_event_name": "PreCompact",
            },
            url,
            plugin_data,
        )

    assert replay.returncode == 0
    assert [event["content"] for event in CaptureHandler.events[:2]] == [
        "Reject permanently",
        "Accept after rejection",
    ]
    assert not accepted_path.exists()
    assert (plugin_data / "capture" / "quarantine" / rejected_path.name).is_file()


def test_large_spool_directory_uses_bounded_incremental_maintenance(tmp_path: Path) -> None:
    plugin_data = tmp_path / "plugin-data"
    spool = plugin_data / "capture" / "spool"
    spool.mkdir(parents=True)
    for index in range(20_000):
        (spool / f"event-{index:064x}.json").write_text("not json")

    counts = [20_000]
    with capture_server() as url:
        for _ in range(2):
            started = time.monotonic()
            result = _run_hook(
                {
                    "session_id": "large-directory",
                    "cwd": str(tmp_path),
                    "hook_event_name": "PreCompact",
                },
                url,
                plugin_data,
            )
            assert result.returncode == 0
            assert time.monotonic() - started < 1.8
            counts.append(len(list(spool.glob("event-*.json"))))
    assert counts[0] > counts[1] > counts[2]


def test_real_hook_enforces_spool_file_limit(tmp_path: Path) -> None:
    plugin_data = tmp_path / "plugin-data"
    spool = plugin_data / "capture" / "spool"
    spool.mkdir(parents=True)
    for index in range(270):
        (spool / f"event-{index:064x}.json").write_text("{}")
    result = _run_hook(
        {
            "session_id": "bounded-spool",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Bounded spool event",
            "timestamp": "2026-09-05T10:03:00Z",
        },
        "http://127.0.0.1:1",
        plugin_data,
    )
    assert result.returncode == 0
    assert len(list(spool.glob("event-*.json"))) <= 256


def test_concurrent_fail_open_prunes_records_and_stale_temps_without_active_writer_loss(
    tmp_path: Path,
) -> None:
    plugin_data = tmp_path / "plugin-data"
    spool = plugin_data / "capture" / "spool"
    lock = spool / ".maintenance-lock"
    lock.mkdir(parents=True)
    stale = time.time() - 60
    for index in range(40):
        path = spool / f"event-{index:064x}.json.{index:032x}.tmp"
        path.write_bytes(b"x" * 20_000)
        os.utime(path, (stale, stale))
    active_temp = spool / f"event-{'f' * 64}.json.{'f' * 32}.tmp"
    active_temp.write_bytes(b"active writer")
    writers = plugin_data / "capture" / "writers"
    fresh_writer = writers / "00"
    stale_writer = writers / "01"
    fresh_writer.mkdir(parents=True)
    stale_writer.mkdir()
    (fresh_writer / "record.tmp").write_bytes(b"active writer")
    (stale_writer / "record.tmp").write_bytes(b"crashed writer")
    os.utime(stale_writer, (stale, stale))

    def invoke(index: int) -> subprocess.CompletedProcess[str]:
        return _run_hook(
            {
                "session_id": f"concurrent-spool-session-{index}",
                "cwd": str(tmp_path),
                "hook_event_name": "UserPromptSubmit",
                "prompt": f"{index}:" + "x" * 20_000,
            },
            "http://127.0.0.1:1",
            plugin_data,
        )

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(invoke, range(320)))

    assert all(result.returncode == 0 for result in results)
    assert active_temp.exists()
    assert fresh_writer.exists()
    assert not stale_writer.exists()
    maintained = [
        path
        for path in spool.iterdir()
        if path.is_file() and (path.suffix == ".json" or path.name.endswith(".tmp"))
        and path != active_temp
    ]
    assert len(maintained) <= 256
    assert sum(path.stat().st_size for path in maintained) <= 4 * 1024 * 1024
    assert not any(path.name.endswith(".tmp") for path in maintained)
    turns = list((plugin_data / "capture" / "turns").glob("*.json"))
    assert len(turns) <= 256
    assert sum(path.stat().st_size for path in turns) <= 1024 * 1024


def test_identity_state_is_ttl_count_and_byte_bounded_without_session_end(tmp_path: Path) -> None:
    plugin_data = tmp_path / "plugin-data"
    identities = plugin_data / "capture" / "identities"
    identities.mkdir(parents=True)
    now = int(time.time() * 1000)
    for index in range(300):
        (identities / f"{index:064x}.json").write_text(
            json.dumps(
                {
                    "turn_id": f"turn-{index}",
                    "event_id": f"event-{index}",
                    "updated_at": now,
                }
            )
        )
    expired = identities / f"{'f' * 64}.json"
    expired.write_text(json.dumps({"turn_id": "old", "event_id": "old", "updated_at": 0}))

    result = _run_hook(
        {
            "session_id": "state-maintenance",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Ordinary capture also maintains bounded state.",
        },
        "http://127.0.0.1:1",
        plugin_data,
    )

    assert result.returncode == 0
    files = list(identities.glob("*.json"))
    assert not expired.exists()
    assert len(files) <= 256
    assert sum(path.stat().st_size for path in files) <= 1024 * 1024


def test_concurrent_replay_keeps_quarantine_within_file_and_byte_caps(tmp_path: Path) -> None:
    plugin_data = tmp_path / "plugin-data"
    spool = plugin_data / "capture" / "spool"
    quarantine = plugin_data / "capture" / "quarantine"
    spool.mkdir(parents=True)
    quarantine.mkdir(parents=True)
    for index in range(257):
        (spool / f"event-{index:064x}.json").write_bytes(b"x" * 20_000)
    expired_spool = spool / (
        "event-0000000000000000000000000000000000000000000000000000000000000000.json"
    )
    expired_quarantine = quarantine / "event-expired.json"
    expired_quarantine.write_text("expired")
    expired = time.time() - 8 * 24 * 60 * 60
    os.utime(expired_spool, (expired, expired))
    os.utime(expired_quarantine, (expired, expired))
    event = {
        "session_id": "concurrent-maintenance",
        "cwd": str(tmp_path),
        "hook_event_name": "PreCompact",
    }
    with capture_server() as url, ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: _run_hook(event, url, plugin_data),
                range(2),
            )
        )
    assert all(result.returncode == 0 for result in results)
    files = list(quarantine.glob("event-*.json"))
    assert not expired_spool.exists()
    assert not expired_quarantine.exists()
    assert len(files) <= 256
    assert sum(path.stat().st_size for path in files) <= 4 * 1024 * 1024
