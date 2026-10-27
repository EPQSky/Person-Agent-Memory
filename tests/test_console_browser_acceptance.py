from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).parents[1]
CHROME = Path("/usr/bin/google-chrome")


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
