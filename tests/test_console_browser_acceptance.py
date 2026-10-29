from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import uvicorn

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.mem2_kernel import Mem2MemoryKernel

ROOT = Path(__file__).parents[1]
CHROME = Path("/usr/bin/google-chrome")


class _ManualSemanticExecutor:
    def __init__(self) -> None:
        self._tasks: deque[Callable[[], None]] = deque()
        self._lock = threading.Lock()

    @property
    def worker_count(self) -> int:
        return 1

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._tasks)

    def submit(self, task_key: str, region_id: str, task: Callable[[], None]) -> bool:
        del task_key, region_id
        with self._lock:
            self._tasks.append(task)
        return True

    def drain(self) -> None:
        while self.pending_count:
            with self._lock:
                task = self._tasks.popleft()
            task()

    def close(self) -> None:
        self.drain()


class _ReadyServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.ready = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        self.ready.set()


@contextmanager
def _manual_semantic_live_app(
    tmp_path: Path,
    extractor: Callable[[dict[str, object]], list[dict[str, object]]],
) -> Iterator[tuple[str, str, Path, _ManualSemanticExecutor]]:
    state_dir = tmp_path / "manual-state"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    executor = _ManualSemanticExecutor()
    kernel = Mem2MemoryKernel(
        state_dir,
        semantic_extractor=extractor,
        semantic_executor=executor,
    )
    app = create_app(Settings(state_dir=state_dir), memory_kernel=kernel)
    server = _ReadyServer(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_config=None,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    server.ready.wait()
    api_key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    try:
        yield f"http://127.0.0.1:{port}", api_key, state_dir, executor
    finally:
        server.should_exit = True
        thread.join()


def _wait_for_app(
    process: subprocess.Popen[str],
    state_dir: Path,
    base_url: str,
) -> str:
    key_file = state_dir / "api-key"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(f"local app exited before startup:\n{output}")
        if key_file.exists():
            candidate = key_file.read_text(encoding="utf-8").strip()
            if candidate:
                try:
                    response = httpx.get(
                        f"{base_url}/api/v1/status",
                        headers={"Authorization": f"Bearer {candidate}"},
                        timeout=1,
                        trust_env=False,
                    )
                except httpx.HTTPError:
                    response = None
                if response is not None and response.status_code == 200:
                    return candidate
        time.sleep(0.1)
    raise AssertionError("local app did not become ready")


@pytest.fixture()
def live_app(tmp_path: Path) -> Iterator[tuple[str, str, Path]]:
    if not CHROME.exists():
        pytest.skip("system Google Chrome is unavailable")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    state_dir = tmp_path / "state"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from personal_agent_memory.cli import main; main()",
            "serve",
            "--state-dir",
            str(state_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        api_key = _wait_for_app(process, state_dir, base_url)
        yield base_url, api_key, state_dir
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_console_login_region_memory_detail_and_governance(
    live_app: tuple[str, str, Path],
) -> None:
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    expect = playwright_api.expect
    base_url, api_key, state_dir = live_app
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=10,
        trust_env=False,
    ) as client:
        region = client.post(
            "/api/v1/regions",
            json={
                "kind": "topic",
                "name": "浏览器验收区域",
                "region_id": "browser-acceptance",
            },
        )
        assert region.status_code == 201, region.text
        memory = client.post(
            "/api/v1/memories",
            json={
                "region_id": "browser-acceptance",
                "content": "浏览器验收：项目优先使用 uv 管理依赖。",
                "memory_type": "constraint",
                "source_references": ["session:browser-acceptance"],
                "actor": "test",
                "idempotency_key": "browser-acceptance-memory",
            },
        )
        assert memory.status_code == 201, memory.text
        second_memory = client.post(
            "/api/v1/memories",
            json={
                "region_id": "browser-acceptance",
                "content": "浏览器验收第二条记忆。",
                "memory_type": "domain_fact",
                "source_references": ["session:browser-acceptance-second"],
                "actor": "test",
                "idempotency_key": "browser-acceptance-memory-second",
            },
        )
        assert second_memory.status_code == 201, second_memory.text

    with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
        for index in range(51):
            content = f"分页验收记忆 {index:02d}"
            connection.execute(
                f"""INSERT OR REPLACE INTO memory_browser_index
                    (region_id, unit_id, region_name, region_kind,
                     scope_org, scope_space, scope_user, scope_agent, scope_session,
                     content, summary, memory_type, memory_type_resolution, lifecycle,
                     updated_at, version, supersedes, source_count, source_references,
                     provenance, user_metadata)
                    VALUES ('browser-acceptance', 'bulk-page-{index:02d}',
                            '浏览器验收区域', 'topic', 'local',
                            'topic:browser-acceptance', 'local',
                            'personal-agent-memory', '', '{content}', '{content}',
                            'domain_fact', 'native', 'active',
                            '2020-01-01T00:00:{index:02d}+00:00',
                            'bulk-version-{index:02d}', '', 1,
                            '["source:bulk"]', '[]', '{{}}')"""
            )
        connection.commit()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(CHROME),
            headless=True,
            args=["--no-sandbox", "--no-proxy-server", "--proxy-bypass-list=*"],
        )
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.goto(base_url, wait_until="domcontentloaded")
            page.get_by_label("API key").fill(api_key)
            page.get_by_role("button", name="登录").click()
            page.locator(".eyebrow").get_by_text("区域工作区").wait_for()
            page.get_by_role(
                "button", name="浏览器验收区域 · topic · browser-acceptance"
            ).click()
            memory_button = page.get_by_role(
                "button", name="查看记忆：浏览器验收：项目优先使用 uv 管理依赖。"
            )
            memory_button.wait_for()
            next_page = page.get_by_role("button", name="加载下一页")
            expect(next_page).to_be_enabled()
            page.get_by_label("关键词").fill("未应用筛选")
            expect(next_page).to_be_disabled()
            page.get_by_label("关键词").fill("")
            memory_button.click()

            detail = page.get_by_role("heading", name="记忆详情")
            detail.wait_for()
            expect(page.locator("#detail-region")).to_have_text("浏览器验收区域 · topic")
            expect(page.locator("#detail-type")).to_have_text("约束")
            expect(page.locator("#source-list")).to_contain_text("session:browser-acceptance")
            assert (
                page.get_by_label("完整正文").input_value()
                == "浏览器验收：项目优先使用 uv 管理依赖。"
            )

            detail_route = "**/api/v1/libraries/*/memories/*"
            page.route(
                detail_route,
                lambda route: route.fulfill(
                    status=500,
                    content_type="application/json",
                    body='{"detail":"forced detail failure"}',
                ),
            )
            memory_button.click()
            page.get_by_text("加载记忆详情失败，详情已清空").wait_for()
            assert page.get_by_label("完整正文").input_value() == ""
            expect(page.locator("#detail-version")).to_have_text("—")
            assert page.get_by_role("button", name="保存更正").is_disabled()
            assert page.get_by_role("button", name="遗忘记忆").is_disabled()
            assert page.get_by_role("button", name="查看审计").is_disabled()
            page.unroute(detail_route)
            memory_button.click()
            expect(page.locator("#detail-region")).to_have_text("浏览器验收区域 · topic")
            assert (
                page.get_by_label("完整正文").input_value()
                == "浏览器验收：项目优先使用 uv 管理依赖。"
            )
            first_memory_id = page.locator(".memory-card.active").get_attribute(
                "data-memory-id"
            )
            assert first_memory_id
            page.evaluate(
                """targetId => {
                    const originalFetch = window.fetch;
                    window.fetch = async (input, init) => {
                        const url = String(input);
                        if (url.includes('/history?') && url.includes(`target_id=${targetId}`)) {
                            await new Promise(resolve => setTimeout(resolve, 900));
                        }
                        return originalFetch(input, init);
                    };
                }""",
                first_memory_id,
            )
            page.get_by_role("button", name="查看审计").click()
            second_memory_button = page.get_by_role(
                "button", name="查看记忆：浏览器验收第二条记忆。"
            )
            second_memory_button.click()
            expect(page.get_by_label("完整正文")).to_have_value("浏览器验收第二条记忆。")
            expect(page.locator("#history-list")).not_to_contain_text(first_memory_id)
            page.wait_for_timeout(1100)
            expect(page.locator("#history-list")).not_to_contain_text(first_memory_id)
            memory_button.click()
            expect(page.get_by_label("完整正文")).to_have_value(
                "浏览器验收：项目优先使用 uv 管理依赖。"
            )

            page.get_by_label("完整正文").fill(
                "浏览器验收更正后：项目使用 uv sync 管理依赖。"
            )
            page.get_by_role("button", name="保存更正").click()
            page.get_by_text("更正已保存并通过当前列表同步").wait_for()
            assert (
                page.get_by_label("完整正文").input_value()
                == "浏览器验收更正后：项目使用 uv sync 管理依赖。"
            )
            assert page.get_by_role(
                "button", name="查看记忆：浏览器验收更正后：项目使用 uv sync 管理依赖。"
            ).is_visible()

            superseded_button = page.get_by_role(
                "button", name="查看记忆：浏览器验收：项目优先使用 uv 管理依赖。"
            )
            superseded_button.click()
            expect(page.get_by_label("完整正文")).to_have_value(
                "浏览器验收：项目优先使用 uv 管理依赖。"
            )
            expect(page.locator("#detail-lifecycle")).to_have_text("已被替代")
            expect(page.get_by_label("完整正文")).to_have_attribute("readonly", "")
            assert page.get_by_role("button", name="保存更正").is_disabled()
            assert page.get_by_role("button", name="遗忘记忆").is_disabled()
            assert page.get_by_role("button", name="查看审计").is_enabled()
            page.get_by_role(
                "button", name="查看记忆：浏览器验收更正后：项目使用 uv sync 管理依赖。"
            ).click()

            page.get_by_role("button", name="查看审计").click()
            page.get_by_text("supersede", exact=False).wait_for()

            active_memory_id = page.locator(".memory-card.active").get_attribute(
                "data-memory-id"
            )
            assert active_memory_id
            with httpx.Client(
                base_url=base_url,
                headers=headers,
                timeout=10,
                trust_env=False,
            ) as client:
                current = client.get(
                    f"/api/v1/libraries/browser-acceptance/memories/{active_memory_id}"
                )
                assert current.status_code == 200, current.text
                external = client.put(
                    "/api/v1/libraries/browser-acceptance/document",
                    json={
                        "path": active_memory_id,
                        "content": "外部客户端更新后的记忆正文。",
                        "expected_source_version": current.json()["source_version"],
                        "operation_id": "browser-acceptance-external-update",
                        "actor_type": "user",
                        "source": "external-acceptance-client",
                    },
                )
                assert external.status_code == 200, external.text

            page.get_by_label("完整正文").fill("页面使用旧版本提交的更正")
            page.get_by_role("button", name="保存更正").click()
            page.get_by_text("版本冲突：该记忆已被其他操作更新").wait_for()
            assert page.get_by_label("完整正文").input_value() == ""
            expect(page.locator("#detail-version")).to_have_text("—")
            assert page.get_by_role("button", name="保存更正").is_disabled()
            assert page.get_by_role("button", name="遗忘记忆").is_disabled()
            assert page.get_by_role("button", name="查看审计").is_disabled()
            external_memory_button = page.get_by_role(
                "button", name="查看记忆：外部客户端更新后的记忆正文。"
            )
            external_memory_button.wait_for()
            external_memory_button.click()
            expect(page.locator("#detail-version")).not_to_have_text("—")
            external_memory_id = page.locator(".memory-card.active").get_attribute(
                "data-memory-id"
            )
            assert external_memory_id
            assert page.get_by_role("button", name="保存更正").is_enabled()
            assert (
                page.get_by_label("完整正文").input_value()
                == "外部客户端更新后的记忆正文。"
            )
            forget_target_id = page.locator(".memory-card.active").get_attribute(
                "data-memory-id"
            )
            assert forget_target_id
            with httpx.Client(
                base_url=base_url,
                headers=headers,
                timeout=10,
                trust_env=False,
            ) as client:
                current = client.get(
                    f"/api/v1/libraries/browser-acceptance/memories/{forget_target_id}"
                )
                assert current.status_code == 200, current.text
                external = client.put(
                    "/api/v1/libraries/browser-acceptance/document",
                    json={
                        "path": forget_target_id,
                        "content": "遗忘冲突后的最新正文。",
                        "expected_source_version": current.json()["source_version"],
                        "operation_id": "browser-acceptance-forget-conflict-update",
                        "actor_type": "user",
                        "source": "external-acceptance-client",
                    },
                )
                assert external.status_code == 200, external.text

            page.once("dialog", lambda dialog: dialog.accept())
            page.get_by_role("button", name="遗忘记忆").click()
            page.get_by_text("版本冲突：记忆状态已变化，无法执行遗忘").wait_for()
            assert page.get_by_label("完整正文").input_value() == ""
            expect(page.locator("#detail-version")).to_have_text("—")
            assert page.get_by_role("button", name="保存更正").is_disabled()
            assert page.get_by_role("button", name="遗忘记忆").is_disabled()
            assert page.get_by_role("button", name="查看审计").is_disabled()
            latest_memory_button = page.get_by_role(
                "button", name="查看记忆：遗忘冲突后的最新正文。"
            )
            latest_memory_button.wait_for()
            latest_memory_button.click()
            expect(page.locator("#detail-version")).not_to_have_text("—")
            assert page.get_by_label("完整正文").input_value() == "遗忘冲突后的最新正文。"
            assert page.get_by_role("button", name="保存更正").is_enabled()
            assert page.get_by_role("button", name="遗忘记忆").is_enabled()
            assert page.get_by_role("button", name="查看审计").is_enabled()

            document_route = "**/api/v1/libraries/*/document"
            page.route(
                document_route,
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "id": forget_target_id,
                            "browser_index_sync": "pending",
                            "warning": "authoritative update committed",
                        }
                    ),
                )
                if route.request.method == "PUT"
                else route.continue_(),
            )
            page.get_by_label("完整正文").fill("索引待同步更正后的正文")
            page.get_by_role("button", name="保存更正").click()
            page.get_by_text("更正已提交，索引待同步").wait_for()
            assert page.get_by_label("完整正文").input_value() == ""
            expect(page.locator("#detail-version")).to_have_text("—")
            assert page.get_by_role("button", name="保存更正").is_disabled()
            assert page.get_by_role("button", name="遗忘记忆").is_disabled()
            assert page.get_by_role("button", name="查看审计").is_disabled()
            expect(
                page.get_by_role(
                    "button", name="查看记忆：遗忘冲突后的最新正文。"
                )
            ).to_have_count(0)
            page.unroute(document_route)
            page.get_by_role("button", name="应用筛选").click()

            restored_after_update = page.get_by_role(
                "button", name="查看记忆：遗忘冲突后的最新正文。"
            )
            restored_after_update.wait_for()
            restored_after_update.click()
            expect(page.locator("#detail-version")).not_to_have_text("—")
            delete_selected_id = page.locator(".memory-card.active").get_attribute(
                "data-memory-id"
            )
            assert delete_selected_id
            chain_affected_ids = list(
                dict.fromkeys(
                    [
                        first_memory_id,
                        active_memory_id,
                        external_memory_id,
                        delete_selected_id,
                    ]
                )
            )
            page.route(
                document_route,
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "id": delete_selected_id,
                            "status": "forget",
                            "deleted": "true",
                            "browser_index_sync": "pending",
                            "affected_units": str(len(chain_affected_ids)),
                            "affected_unit_ids": chain_affected_ids,
                            "warning": "authoritative delete committed",
                        }
                    ),
                )
                if route.request.method == "DELETE"
                else route.continue_(),
            )
            page.once("dialog", lambda dialog: dialog.accept())
            page.get_by_role("button", name="遗忘记忆").click()
            page.get_by_text("遗忘已提交，索引待同步").wait_for()
            assert page.get_by_label("完整正文").input_value() == ""
            expect(page.locator("#detail-version")).to_have_text("—")
            assert page.get_by_role("button", name="保存更正").is_disabled()
            assert page.get_by_role("button", name="遗忘记忆").is_disabled()
            assert page.get_by_role("button", name="查看审计").is_disabled()
            expect(
                page.get_by_role(
                    "button", name="查看记忆：遗忘冲突后的最新正文。"
                )
            ).to_have_count(0)
            expect(
                page.get_by_role(
                    "button", name="查看记忆：浏览器验收：项目优先使用 uv 管理依赖。"
                )
            ).to_have_count(0)
            expect(
                page.get_by_role(
                    "button", name="查看记忆：浏览器验收更正后：项目使用 uv sync 管理依赖。"
                )
            ).to_have_count(0)
            expect(
                page.get_by_role(
                    "button", name="查看记忆：外部客户端更新后的记忆正文。"
                )
            ).to_have_count(0)
            page.unroute(document_route)
            with httpx.Client(
                base_url=base_url,
                headers=headers,
                timeout=10,
                trust_env=False,
            ) as client:
                current = client.get(
                    f"/api/v1/libraries/browser-acceptance/memories/{delete_selected_id}"
                )
                assert current.status_code == 200, current.text
                preview = client.post(
                    "/api/v1/libraries/browser-acceptance/document/delete-preview",
                    json={
                        "path": delete_selected_id,
                        "expected_source_version": current.json()["source_version"],
                        "operation_id": "browser-acceptance-chain-delete-preview",
                    },
                )
                assert preview.status_code == 200, preview.text
                deleted = client.request(
                    "DELETE",
                    "/api/v1/libraries/browser-acceptance/document",
                    json={
                        "path": delete_selected_id,
                        "expected_source_version": current.json()["source_version"],
                        "preview_token": preview.json()["preview_token"],
                        "operation_id": "browser-acceptance-chain-delete",
                        "actor_type": "user",
                        "source": "browser-acceptance",
                        "mode": "forget",
                    },
                )
                assert deleted.status_code == 200, deleted.text
                assert set(deleted.json()["affected_unit_ids"]) == set(
                    chain_affected_ids
                )
            page.get_by_role("button", name="应用筛选").click()
            expect(
                page.get_by_role(
                    "button", name="查看记忆：遗忘冲突后的最新正文。"
                )
            ).to_have_count(0)
            expect(
                page.get_by_role(
                    "button", name="查看记忆：浏览器验收：项目优先使用 uv 管理依赖。"
                )
            ).to_have_count(0)
            expect(
                page.get_by_role(
                    "button", name="查看记忆：浏览器验收更正后：项目使用 uv sync 管理依赖。"
                )
            ).to_have_count(0)
            expect(
                page.get_by_role(
                    "button", name="查看记忆：外部客户端更新后的记忆正文。"
                )
            ).to_have_count(0)
            page.get_by_role(
                "button", name="查看记忆：浏览器验收第二条记忆。"
            ).click()
            expect(page.locator("#detail-version")).not_to_have_text("—")

            list_route = "**/api/v1/libraries/*/memories?*"
            page.route(
                list_route,
                lambda route: route.fulfill(
                    status=500,
                    content_type="application/json",
                    body='{"detail":"forced list failure"}',
                ),
            )
            page.get_by_label("完整正文").fill("列表失败时的第二次更正")
            page.get_by_role("button", name="保存更正").click()
            page.get_by_text("更正已保存，但列表刷新失败").wait_for()
            assert page.get_by_label("完整正文").input_value() == ""
            expect(page.locator("#detail-version")).to_have_text("—")
            assert page.get_by_role("button", name="保存更正").is_disabled()
            assert page.get_by_role("button", name="遗忘记忆").is_disabled()
            page.unroute(list_route)
        finally:
            browser.close()


def test_console_raw_session_keeps_questions_out_of_formal_memory(
    live_app: tuple[str, str, Path],
) -> None:
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    expect = playwright_api.expect
    base_url, api_key, state_dir = live_app
    headers = {"Authorization": f"Bearer {api_key}"}
    project_root = state_dir.parent / "raw-session-project"
    project_root.mkdir()

    with httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=10,
        trust_env=False,
    ) as client:
        region = client.post(
            "/api/v1/regions",
            json={
                "kind": "project",
                "name": "原始会话验收区域",
                "region_id": "raw-session-acceptance",
            },
        )
        assert region.status_code == 201, region.text
        binding = client.post(
            "/api/v1/project-bindings",
            json={
                "project_root": str(project_root),
                "region_id": "raw-session-acceptance",
            },
        )
        assert binding.status_code == 201, binding.text
        memory = client.post(
            "/api/v1/memories",
            json={
                "region_id": "raw-session-acceptance",
                "content": "浏览器验收：项目使用 uv sync 管理依赖。",
                "memory_type": "constraint",
                "source_references": ["session:raw-session-acceptance"],
                "actor": "browser-acceptance",
                "idempotency_key": "raw-session-formal-memory",
            },
        )
        assert memory.status_code == 201, memory.text
        question = "帮我分析下当前项目"
        for event in (
            {
                "event_id": "raw-question",
                "session_id": "raw-session",
                "turn_id": "completed-turn",
                "event_kind": "user",
                "content": question,
                "occurred_at": "2026-10-27T06:00:00Z",
            },
            {
                "event_id": "raw-answer",
                "session_id": "raw-session",
                "turn_id": "completed-turn",
                "event_kind": "assistant",
                "content": "当前项目分析已完成。",
                "occurred_at": "2026-10-27T06:00:01Z",
            },
            {
                "event_id": "raw-incomplete",
                "session_id": "raw-session",
                "turn_id": "incomplete-turn",
                "event_kind": "user",
                "content": "尚未完成的后续问题",
                "occurred_at": "2026-10-27T06:00:02Z",
            },
        ):
            captured = client.post(
                "/api/v1/capture/events",
                json={**event, "cwd": str(project_root)},
            )
            assert captured.status_code == 202, captured.text
        assistant_only = client.post(
            "/api/v1/capture/events",
            json={
                "event_id": "assistant-only",
                "session_id": "assistant-only-session",
                "turn_id": "assistant-only-turn",
                "event_kind": "assistant",
                "content": "只有助手回复的正文",
                "occurred_at": "2026-10-27T06:00:03Z",
                "cwd": str(project_root),
            },
        )
        assert assistant_only.status_code == 202, assistant_only.text
        consolidated = client.post(
            "/api/v1/capture/consolidate",
            json={"session_id": "raw-session"},
        )
        assert consolidated.status_code == 202, consolidated.text
        for event in (
            {
                "event_id": "ready-question",
                "session_id": "ready-session",
                "turn_id": "ready-turn",
                "event_kind": "user",
                "content": "等待提炼的完整回合",
                "occurred_at": "2026-10-27T06:00:10Z",
            },
            {
                "event_id": "ready-answer",
                "session_id": "ready-session",
                "turn_id": "ready-turn",
                "event_kind": "assistant",
                "content": "等待提炼的助手最终回复",
                "occurred_at": "2026-10-27T06:00:11Z",
            },
        ):
            captured = client.post(
                "/api/v1/capture/events",
                json={**event, "cwd": str(project_root)},
            )
            assert captured.status_code == 202, captured.text
        empty_region = client.post(
            "/api/v1/regions",
            json={
                "kind": "project",
                "name": "空原始会话区域",
                "region_id": "raw-empty",
            },
        )
        assert empty_region.status_code == 201, empty_region.text

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(CHROME),
            headless=True,
            args=["--no-sandbox", "--no-proxy-server", "--proxy-bypass-list=*"],
        )
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.goto(base_url, wait_until="domcontentloaded")
            page.get_by_label("API key").fill(api_key)
            page.get_by_role("button", name="登录").click()
            page.get_by_role(
                "button", name="原始会话验收区域 · project · raw-session-acceptance"
            ).click()

            memory_tab = page.get_by_role("tab", name="记忆")
            raw_tab = page.get_by_role("tab", name="原始会话")
            expect(memory_tab).to_have_attribute("aria-selected", "true")
            formal_card = page.get_by_role(
                "button", name="查看记忆：浏览器验收：项目使用 uv sync 管理依赖。"
            )
            formal_card.wait_for()
            expect(page.locator("#memory-workspace").get_by_text(question)).to_have_count(0)

            raw_tab.click()
            expect(raw_tab).to_have_attribute("aria-selected", "true")
            expect(memory_tab).to_have_attribute("aria-selected", "false")
            expect(page.locator("#memory-workspace")).to_be_hidden()
            expect(page.locator("#raw-session-workspace")).to_be_visible()
            page.get_by_role(
                "button", name="查看原始会话：raw-session"
            ).click()
            expect(page.get_by_text(question, exact=True)).to_be_visible()
            expect(page.get_by_text("完成", exact=True)).to_be_visible()
            expect(page.get_by_text("待完成", exact=True)).to_be_visible()
            expect(
                page.get_by_text("缺少助手最终回复，回合尚未完成。", exact=True)
            ).to_be_visible()
            expect(
                page.get_by_text("缺少用户消息，回合尚未完成。", exact=True)
            ).to_have_count(0)
            page.get_by_role(
                "button", name="查看原始会话：assistant-only-session"
            ).click()
            expect(
                page.get_by_text("缺少用户消息，回合尚未完成。", exact=True)
            ).to_be_visible()
            expect(page.get_by_text("只有助手回复的正文", exact=True)).to_be_visible()
            expect(
                page.get_by_text("缺少助手最终回复，回合尚未完成。", exact=True)
            ).to_have_count(0)
            page.get_by_role(
                "button", name="查看原始会话：ready-session"
            ).click()
            expect(page.get_by_text("待提炼", exact=True)).to_be_visible()
            expect(page.locator("#raw-session-workspace input")).to_have_count(0)
            expect(page.get_by_role("button", name="重新提炼")).to_have_count(0)

            memory_tab.click()
            formal_card.wait_for()
            expect(page.locator("#memory-workspace").get_by_text(question)).to_have_count(0)

            raw_tab.click()
            page.get_by_role(
                "button", name="空原始会话区域 · project · raw-empty"
            ).click()
            expect(
                page.get_by_text("当前区域没有原始会话记录。", exact=True)
            ).to_be_visible()
            memory_tab.click()
            expect(
                page.get_by_text("当前区域或筛选条件下没有正式语义记忆。", exact=True)
            ).to_be_visible()

            raw_tab.click()
            rounds_route = "**/api/v1/capture/rounds?*"
            page.route(
                rounds_route,
                lambda route: route.fulfill(
                    status=500,
                    content_type="application/json",
                    body='{"detail":"forced raw session failure"}',
                ),
            )
            page.get_by_role(
                "button", name="原始会话验收区域 · project · raw-session-acceptance"
            ).click()
            expect(page.get_by_text("加载原始会话失败", exact=False)).to_be_visible()
            page.unroute(rounds_route)

            legacy_event_time = "2026-10-27T08:00:00+00:00"
            legacy_user = {
                "record_type": "capture_event",
                "event_id": "legacy-user",
                "session_id": "legacy-session",
                "turn_id": "legacy-turn",
                "region_id": "raw-session-acceptance",
                "event_kind": "user",
                "content": "旧捕获问题",
                "unit_id": "legacy-user",
                "occurred_at": legacy_event_time,
                "received_at": legacy_event_time,
                "legacy": True,
            }
            legacy_assistant = {
                **legacy_user,
                "event_id": "legacy-assistant",
                "event_kind": "assistant",
                "content": "旧捕获助手回复",
                "unit_id": "legacy-assistant",
            }
            legacy_diagnostic = {
                "record_type": "evolved_diagnostic",
                "diagnostic_kind": "legacy_evolved",
                "event_id": "",
                "session_id": "legacy-session",
                "turn_id": "legacy-turn",
                "region_id": "raw-session-acceptance",
                "event_kind": "",
                "content": "旧提炼单元正文",
                "unit_id": "legacy-diagnostic",
                "occurred_at": "",
                "received_at": legacy_event_time,
                "legacy": True,
            }
            archived_completed_diagnostic = {
                **legacy_diagnostic,
                "turn_id": "archived-completed-turn",
                "content": "归档完成诊断正文",
                "unit_id": "archived-completed-diagnostic",
                "received_at": "2026-10-27T08:01:00+00:00",
            }
            archived_failed_diagnostic = {
                **legacy_diagnostic,
                "turn_id": "archived-failed-turn",
                "content": "归档失败诊断正文",
                "unit_id": "archived-failed-diagnostic",
                "received_at": "2026-10-27T08:02:00+00:00",
            }
            page.route(
                rounds_route,
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "session_id": "legacy-session",
                                "turn_id": "legacy-turn",
                                "region_id": "raw-session-acceptance",
                                "status": "pending",
                                "distillation_status": "pending_distillation",
                                "failure_reason": "",
                                "started_at": legacy_event_time,
                                "updated_at": legacy_event_time,
                                "received_at": legacy_event_time,
                                "events": [legacy_user, legacy_assistant, legacy_diagnostic],
                                "user_message": legacy_user,
                                "assistant_final": legacy_assistant,
                                "diagnostics": [legacy_diagnostic],
                            },
                            {
                                "session_id": "legacy-session",
                                "turn_id": "archived-completed-turn",
                                "region_id": "raw-session-acceptance",
                                "status": "done",
                                "distillation_status": "completed",
                                "memory_id": "archived-completed-memory",
                                "failure_reason": "",
                                "started_at": "2026-10-27T08:01:00+00:00",
                                "updated_at": "2026-10-27T08:01:00+00:00",
                                "received_at": "2026-10-27T08:01:00+00:00",
                                "events": [archived_completed_diagnostic],
                                "user_message": None,
                                "assistant_final": None,
                                "diagnostics": [archived_completed_diagnostic],
                            },
                            {
                                "session_id": "legacy-session",
                                "turn_id": "archived-failed-turn",
                                "region_id": "raw-session-acceptance",
                                "status": "failed",
                                "distillation_status": "failed",
                                "memory_id": None,
                                "failure_reason": "归档后失败原因",
                                "started_at": "2026-10-27T08:02:00+00:00",
                                "updated_at": "2026-10-27T08:02:00+00:00",
                                "received_at": "2026-10-27T08:02:00+00:00",
                                "events": [archived_failed_diagnostic],
                                "user_message": None,
                                "assistant_final": None,
                                "diagnostics": [archived_failed_diagnostic],
                            },
                            {
                                "session_id": "empty-round-session",
                                "turn_id": "empty-round-turn",
                                "region_id": "raw-session-acceptance",
                                "status": "pending",
                                "distillation_status": "pending_completion",
                                "failure_reason": "",
                                "started_at": "2026-10-27T08:03:00+00:00",
                                "updated_at": "2026-10-27T08:03:00+00:00",
                                "received_at": "2026-10-27T08:03:00+00:00",
                                "events": [],
                                "user_message": None,
                                "assistant_final": None,
                                "diagnostics": [],
                            },
                        ],
                        ensure_ascii=False,
                    ),
                ),
            )
            page.get_by_role(
                "button", name="原始会话验收区域 · project · raw-session-acceptance"
            ).click()
            page.get_by_role(
                "button", name="查看原始会话：legacy-session"
            ).click()
            expect(page.get_by_text("旧捕获问题", exact=True)).to_be_visible()
            expect(page.get_by_text("旧捕获助手回复", exact=True)).to_be_visible()
            expect(
                page.get_by_text("只读诊断 / 旧提炼单元", exact=True)
            ).to_have_count(3)
            expect(page.get_by_text("旧提炼单元正文", exact=True)).to_be_visible()
            expect(page.get_by_text("归档完成诊断正文", exact=True)).to_be_visible()
            expect(page.get_by_text("归档失败诊断正文", exact=True)).to_be_visible()
            round_list = page.locator("#raw-round-list")
            expect(round_list.get_by_text("完成", exact=True)).to_be_visible()
            expect(round_list.get_by_text("失败", exact=True)).to_be_visible()
            expect(round_list.get_by_text("回合已完成提炼状态。", exact=True)).to_be_visible()
            expect(
                round_list.get_by_text("提炼失败：归档后失败原因", exact=True)
            ).to_be_visible()
            expect(
                round_list.get_by_text("缺少助手最终回复，回合尚未完成。", exact=True)
            ).to_have_count(0)
            page.get_by_role(
                "button", name="查看原始会话：empty-round-session"
            ).click()
            expect(
                round_list.get_by_text(
                    "回合消息不完整，缺少用户消息和助手最终回复。",
                    exact=True,
                )
            ).to_be_visible()
            expect(
                round_list.get_by_text("缺少用户消息，回合尚未完成。", exact=True)
            ).to_have_count(0)
            expect(
                round_list.get_by_text("缺少助手最终回复，回合尚未完成。", exact=True)
            ).to_have_count(0)
            page.unroute(rounds_route)
        finally:
            browser.close()


def test_console_archived_real_retention_rounds_keep_stable_timeline(
    live_app: tuple[str, str, Path],
) -> None:
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    expect = playwright_api.expect
    base_url, api_key, state_dir = live_app
    headers = {"Authorization": f"Bearer {api_key}"}
    project_root = state_dir.parent / "retention-browser-project"
    project_root.mkdir()
    session_id = "retention-browser-session"
    region_id = ""

    with httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=10,
        trust_env=False,
    ) as client:
        for turn_id, marker, base_second in (
            ("turn-a-completed", "REALRETENTION-COMPLETED", 0),
            ("turn-b-failed", "REALRETENTION-FAILED", 10),
        ):
            for event_id, kind, content, second in (
                (f"{turn_id}-user", "user", f"{marker} 提问", "00"),
                (f"{turn_id}-assistant", "assistant", f"{marker} 最终回复", "01"),
            ):
                captured = client.post(
                    "/api/v1/capture/events",
                    json={
                        "event_id": event_id,
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "event_kind": kind,
                        "content": content,
                        "occurred_at": (
                            f"2020-01-01T00:00:{base_second + int(second):02d}Z"
                        ),
                        "cwd": str(project_root),
                    },
                )
                assert captured.status_code == 202, captured.text
                captured_region_id = str(captured.json()["region_id"])
                if region_id:
                    assert captured_region_id == region_id
                else:
                    region_id = captured_region_id
        regions = client.get("/api/v1/regions", headers=headers)
        assert regions.status_code == 200, regions.text
        region_payload = next(
            item for item in regions.json() if item["region_id"] == region_id
        )
        consolidated = client.post(
            "/api/v1/capture/consolidate",
            json={"session_id": session_id},
        )
        assert consolidated.status_code == 202, consolidated.text
        assert consolidated.json()["queued"] == 2

        with sqlite3.connect(state_dir / "mem2" / "maintenance.sqlite3") as connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET status = 'pending', memory_id = NULL, completed_at = NULL,
                       last_error = ?
                   WHERE session_id = ? AND turn_id = ?""",
                ("真实归档失败原因", session_id, "turn-b-failed"),
            )

        before = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={"region_id": region_id, "session_id": session_id},
        )
        assert before.status_code == 200, before.text
        before_by_turn = {item["turn_id"]: item for item in before.json()}
        assert before_by_turn["turn-b-failed"]["distillation_status"] == "failed"
        assert before_by_turn["turn-b-failed"]["failure_reason"] == "真实归档失败原因"
        completed_times = (
            before_by_turn["turn-a-completed"]["started_at"],
            before_by_turn["turn-a-completed"]["received_at"],
            before_by_turn["turn-a-completed"]["updated_at"],
        )
        failed_times = (
            before_by_turn["turn-b-failed"]["started_at"],
            before_by_turn["turn-b-failed"]["received_at"],
            before_by_turn["turn-b-failed"]["updated_at"],
        )

        policy = client.put(
            f"/api/v1/libraries/{region_id}/retention-policy",
            headers=headers,
            json={"inbox_days": 0},
        )
        assert policy.status_code == 200, policy.text
        cleaned = client.post(
            f"/api/v1/libraries/{region_id}/retention-cleanup",
            headers=headers,
        )
        assert cleaned.status_code == 200, cleaned.text
        assert cleaned.json()["counts"]["captures_archived"] == 4

        after = client.get(
            "/api/v1/capture/rounds",
            headers=headers,
            params={"region_id": region_id, "session_id": session_id},
        )
        assert after.status_code == 200, after.text
        archived = after.json()
        assert [item["turn_id"] for item in archived] == [
            "turn-a-completed",
            "turn-b-failed",
        ]
        by_turn = {item["turn_id"]: item for item in archived}
        completed_round = by_turn["turn-a-completed"]
        failed_round = by_turn["turn-b-failed"]
        assert completed_round["distillation_status"] == "completed"
        assert completed_round["memory_id"]
        assert completed_round["created_at"]
        assert (
            completed_round["started_at"],
            completed_round["received_at"],
            completed_round["updated_at"],
        ) == completed_times
        assert completed_round["completed_at"]
        assert failed_round["distillation_status"] == "failed"
        assert failed_round["failure_reason"] == "真实归档失败原因"
        assert failed_round["completed_at"] is None
        assert failed_round["created_at"]
        assert (
            failed_round["started_at"],
            failed_round["received_at"],
            failed_round["updated_at"],
        ) == failed_times
        assert completed_round["diagnostics"]
        assert failed_round["diagnostics"]
        starts = [
            datetime.fromisoformat(str(item["started_at"]).replace("Z", "+00:00")).astimezone(
                UTC
            )
            for item in archived
        ]
        assert starts == sorted(starts)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(CHROME),
            headless=True,
            args=["--no-sandbox", "--no-proxy-server", "--proxy-bypass-list=*"],
        )
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.goto(base_url, wait_until="domcontentloaded")
            page.get_by_label("API key").fill(api_key)
            page.get_by_role("button", name="登录").click()
            page.get_by_role(
                "button",
                name=(
                    f"{region_payload['name']} · {region_payload['kind']} · {region_id}"
                ),
            ).click()
            page.get_by_role("tab", name="原始会话").click()
            page.get_by_role(
                "button", name="查看原始会话：retention-browser-session"
            ).click()
            round_list = page.locator("#raw-round-list")
            expect(round_list.get_by_text("完成", exact=True)).to_be_visible()
            expect(round_list.get_by_text("失败", exact=True)).to_be_visible()
            expect(
                round_list.get_by_text("回合已完成提炼状态。", exact=True)
            ).to_be_visible()
            expect(
                round_list.get_by_text("提炼失败：真实归档失败原因", exact=True)
            ).to_be_visible()
            expect(
                round_list.get_by_text("REALRETENTION-COMPLETED 提问", exact=False)
            ).to_be_visible()
            expect(
                round_list.get_by_text("REALRETENTION-FAILED 最终回复", exact=False)
            ).to_be_visible()
            expect(page.locator("#raw-round-list h3")).to_have_text(
                ["回合 turn-a-completed", "回合 turn-b-failed"]
            )
            round_times = page.locator("#raw-round-list .raw-round-time")
            expect(round_times).to_have_count(2)
            assert all("—" not in text for text in round_times.all_text_contents())
        finally:
            browser.close()


def test_console_semantic_distillation_routes_project_and_user_memories_with_evidence(
    tmp_path: Path,
) -> None:
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    expect = playwright_api.expect
    project_root = tmp_path / "semantic-browser-project"
    project_root.mkdir()
    region_id = "semantic-browser-region"
    session_id = "semantic-browser-session"
    question = "帮我分析下当前项目"
    project_fact_question = "This project uses pnpm for builds."
    project_fact_memory = "This project uses pnpm for builds."
    preference_question = "我偏好项目统一使用 uv 管理依赖。"
    preference_memory = "用户偏好项目统一使用 uv 管理依赖。"
    sensitive_assistant = "api_key = 'correct-horse-battery-staple'"

    def extractor(context: dict[str, object]) -> list[dict[str, object]]:
        messages = context["messages"]
        assert isinstance(messages, list)
        user_messages = [
            item
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        memories: list[dict[str, object]] = []
        for user_message in user_messages:
            content = str(user_message.get("content", ""))
            if "This project uses pnpm" in content:
                memories.append(
                    {
                        "content": project_fact_memory,
                        "memory_type": "constraint",
                        "confidence": 0.97,
                        "region_hint": "source",
                        "user_event_id": user_message["event_id"],
                        "evidence_snippet": "This project uses pnpm for builds",
                        "claim_subject": "project:dependency_manager",
                        "claim_value": "pnpm",
                        "durability": "durable_statement",
                    }
                )
            elif "我偏好" in content:
                memories.append(
                    {
                        "content": preference_memory,
                        "memory_type": "preference",
                        "confidence": 0.97,
                        "region_hint": "user",
                        "user_event_id": user_message["event_id"],
                        "evidence_snippet": "我偏好项目统一使用 uv 管理依赖",
                        "claim_subject": "user:preference:dependency_manager",
                        "claim_value": "uv",
                        "durability": "durable_statement",
                    }
                )
        return memories

    with _manual_semantic_live_app(tmp_path, extractor) as (
        base_url,
        api_key,
        state_dir,
        executor,
    ):
        headers = {"Authorization": f"Bearer {api_key}"}
        with httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=15,
            trust_env=False,
        ) as client:
            region = client.post(
                "/api/v1/regions",
                json={
                    "kind": "project",
                    "name": "语义提炼验收区域",
                    "region_id": region_id,
                },
            )
            assert region.status_code == 201, region.text
            binding = client.post(
                "/api/v1/project-bindings",
                json={"project_root": str(project_root), "region_id": region_id},
            )
            assert binding.status_code == 201, binding.text
            for turn_id, marker, user_content, assistant_content, second in (
                ("question-turn", "QUESTION", question, "当前项目分析已完成。", 0),
                (
                    "project-turn",
                    "PROJECT",
                    project_fact_question,
                    "已确认项目依赖管理器。",
                    10,
                ),
                (
                    "preference-turn",
                    "PREFERENCE",
                    preference_question,
                    sensitive_assistant,
                    20,
                ),
            ):
                for kind, content, offset in (
                    ("user", user_content, 0),
                    ("assistant", assistant_content, 1),
                ):
                    captured = client.post(
                        "/api/v1/capture/events",
                        json={
                            "event_id": f"{marker}-{kind}",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "event_kind": kind,
                            "content": content,
                            "occurred_at": f"2026-10-28T04:00:{second + offset:02d}Z",
                            "cwd": str(project_root),
                        },
                    )
                    assert captured.status_code == 202, captured.text

            assert executor.pending_count == 3
            executor.drain()
            assert executor.pending_count == 0
            rounds = client.get(
                "/api/v1/capture/rounds",
                params={"region_id": region_id, "session_id": session_id},
            )
            assert rounds.status_code == 200, rounds.text
            rounds_by_turn = {
                str(item["turn_id"]): item for item in rounds.json()
            }
            assert rounds_by_turn["question-turn"]["semantic_status"] == "skipped"
            assert rounds_by_turn["question-turn"]["semantic_memory_ids"] == []
            assert rounds_by_turn["project-turn"]["semantic_status"] == "completed"
            project_memory_ids = rounds_by_turn["project-turn"]["semantic_memory_ids"]
            assert isinstance(project_memory_ids, list) and len(project_memory_ids) == 1
            assert rounds_by_turn["preference-turn"]["semantic_status"] == "completed"
            preference_memory_ids = rounds_by_turn["preference-turn"]["semantic_memory_ids"]
            assert isinstance(preference_memory_ids, list) and len(preference_memory_ids) == 1

            formal = client.get(f"/api/v1/libraries/{region_id}/memories?limit=50")
            assert formal.status_code == 200, formal.text
            formal_items = formal.json()["items"]
            assert [item["summary"] for item in formal_items] == [project_fact_memory]
            assert question not in json.dumps(formal_items, ensure_ascii=False)
            user_formal = client.get("/api/v1/libraries/user/memories?limit=50")
            assert user_formal.status_code == 200, user_formal.text
            assert [item["summary"] for item in user_formal.json()["items"]] == [
                preference_memory
            ]
            question_search = client.post(
                "/api/v1/search",
                json={"region_id": region_id, "query": question, "limit": 10},
            )
            assert question_search.status_code == 200, question_search.text
            assert question not in json.dumps(
                question_search.json()["results"], ensure_ascii=False
            )
            preference_search = client.post(
                "/api/v1/search",
                json={"region_id": "user", "query": "uv", "limit": 10},
            )
            assert preference_search.status_code == 200, preference_search.text
            assert any(
                item["document_id"] == preference_memory_ids[0]
                for item in preference_search.json()["results"]
            )
            preference_detail = client.get(
                f"/api/v1/libraries/user/memories/{preference_memory_ids[0]}"
            )
            assert preference_detail.status_code == 200, preference_detail.text
            assert "correct-horse-battery-staple" not in preference_detail.text
            sensitive_evidence = next(
                item
                for item in preference_detail.json()["evidence"]
                if item["event_id"] == "PREFERENCE-assistant"
            )
            assert sensitive_evidence["redacted"] is True
            assert sensitive_evidence["redaction_reason"] == "sensitive"
            project_search = client.post(
                "/api/v1/search",
                json={"region_id": region_id, "query": "pnpm", "limit": 10},
            )
            assert project_search.status_code == 200, project_search.text
            assert any(
                item["document_id"] == project_memory_ids[0]
                for item in project_search.json()["results"]
            )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=str(CHROME),
                headless=True,
                args=["--no-sandbox", "--no-proxy-server", "--proxy-bypass-list=*"],
            )
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1100})
                page.goto(base_url, wait_until="domcontentloaded")
                page.get_by_label("API key").fill(api_key)
                page.get_by_role("button", name="登录").click()
                page.get_by_role(
                    "button", name="语义提炼验收区域 · project · semantic-browser-region"
                ).click()

                project_memory_button = page.get_by_role(
                    "button", name=f"查看记忆：{project_fact_memory}"
                )
                project_memory_button.wait_for()
                expect(page.locator("#memory-workspace").get_by_text(question)).to_have_count(0)
                expect(
                    page.locator("#memory-workspace").get_by_text(preference_memory)
                ).to_have_count(0)
                project_memory_button.click()
                expect(page.locator("#detail-target-region")).to_contain_text(
                    "语义提炼验收区域 · project · semantic-browser-region"
                )
                expect(page.locator("#detail-source-regions")).to_contain_text(
                    "semantic-browser-region"
                )
                expect(page.locator("#evidence-list")).to_contain_text(project_fact_question)
                expect(page.locator("#source-list")).to_contain_text("turn:project-turn")

                page.get_by_role("button", name="用户记忆 · user · user").click()
                memory_button = page.get_by_role(
                    "button", name=f"查看记忆：{preference_memory}"
                )
                memory_button.wait_for()
                expect(
                    page.locator("#memory-workspace").get_by_text(project_fact_memory)
                ).to_have_count(0)
                memory_button.click()
                expect(page.locator("#detail-target-region")).to_contain_text(
                    "用户记忆 · user · user"
                )
                expect(page.locator("#detail-source-regions")).to_contain_text(
                    "semantic-browser-region"
                )
                expect(page.locator("#detail-evidence-basis")).to_contain_text(
                    "我偏好项目统一使用 uv 管理依赖"
                )
                expect(page.locator("#evidence-list")).to_contain_text(preference_question)
                expect(page.locator("#evidence-list")).to_contain_text(
                    "正文因敏感内容已脱敏"
                )
                expect(page.locator("#evidence-list")).not_to_contain_text(
                    "correct-horse-battery-staple"
                )
                expect(page.locator("#source-list")).to_contain_text("turn:preference-turn")

                page.get_by_role(
                    "button", name="语义提炼验收区域 · project · semantic-browser-region"
                ).click()
                page.get_by_role("tab", name="原始会话").click()
                page.get_by_role(
                    "button", name=f"查看原始会话：{session_id}"
                ).click()
                round_list = page.locator("#raw-round-list")
                expect(round_list.get_by_text(question, exact=True)).to_be_visible()
                expect(
                    round_list.get_by_text("语义提炼：已跳过", exact=True)
                ).to_be_visible()
                expect(
                    round_list.get_by_text(
                        "语义诊断：model returned no semantic memories", exact=True
                    )
                ).to_be_visible()
                expect(
                    round_list.get_by_text(preference_question, exact=True)
                ).to_be_visible()
                expect(
                    round_list.get_by_text(project_fact_question, exact=True)
                ).to_be_visible()
                expect(
                    round_list.get_by_text("语义提炼：语义完成", exact=True)
                ).to_have_count(2)
                expect(round_list).to_contain_text(preference_memory_ids[0])
                expect(round_list).to_contain_text(project_memory_ids[0])
                expect(
                    round_list.get_by_text(
                        "用户证据：我偏好项目统一使用 uv 管理依赖", exact=True
                    )
                ).to_be_visible()
                expect(page.locator("#raw-session-workspace input")).to_have_count(0)
                expect(page.get_by_role("button", name="重新提炼")).to_have_count(0)
            finally:
                browser.close()
