from __future__ import annotations

import json
import math
from pathlib import Path

from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.context_package import (
    HARD_TOKEN_LIMIT,
    TokenCounter,
    _normalized_package_tokens,
    build_context_package,
    is_safe_markdown_fragment,
)


def result(
    content: str,
    *,
    classification: str = "direct",
    path: str = "memory.md",
) -> dict[str, object]:
    return {
        "chunk_id": f"chunk-{classification}-{path}",
        "document_id": f"document-{path}",
        "content": content,
        "library_id": "library-one",
        "path": path,
        "heading": "Decision",
        "start_line": 3,
        "end_line": 3,
        "source_version": "a" * 64,
        "source_type": "markdown",
        "classification": classification,
        "score": 0.75,
        "retrieval_sources": ["full_text" if classification == "direct" else "graph"],
        "degraded": False,
        "degradation": [],
    }


def raw(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "bound",
        "scope": {"kind": "explicit_library", "library_id": "library-one"},
        "query": "needle",
        "degradation": [],
        "vector_index_status": "ready",
        "graph_index_status": "ready",
        "results": results,
    }


def encoded_tokens(payload: dict[str, object]) -> int:
    counter = TokenCounter.for_model("gpt-4o-mini")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return counter.count(encoded)


def assert_conservative_telemetry(
    package: dict[str, object], counter: TokenCounter
) -> None:
    budget = package["budget"]
    allocation = budget["allocation"]
    results = package["results"]
    empty_content = [{**item, "content": ""} for item in results]
    actual = counter.count(json.dumps(package, ensure_ascii=False, separators=(",", ":")))
    normalized_used = _normalized_package_tokens(package, counter, results)
    normalized_metadata = max(
        0,
        _normalized_package_tokens(package, counter, empty_content)
        - allocation["fixed_envelope_tokens"],
    )

    assert budget["telemetry_accounting"] == "conservative_fixed_width_v1"
    assert budget["used_tokens"] == normalized_used
    assert budget["used_tokens"] >= actual
    assert budget["used_tokens"] - actual <= 64
    assert budget["used_tokens"] <= budget["effective_tokens"]
    assert allocation["metadata_used"] == normalized_metadata
    assert allocation["metadata_used"] <= allocation["metadata_limit"]


def test_context_package_enforces_requested_and_absolute_budget() -> None:
    package = build_context_package(
        raw([result("token " * 30_000)]),
        requested_budget=HARD_TOKEN_LIMIT + 5_000,
        target_model="gpt-4o-mini",
    )

    budget = package["budget"]
    assert isinstance(budget, dict)
    assert budget["effective_tokens"] == HARD_TOKEN_LIMIT
    assert_conservative_telemetry(package, TokenCounter.for_model("gpt-4o-mini"))
    assert encoded_tokens(package) <= HARD_TOKEN_LIMIT
    assert package["results"]

def test_context_package_allocates_direct_first_and_caps_graph() -> None:
    package = build_context_package(
        raw(
            [
                result("direct " * 8_000),
                result("graph " * 8_000, classification="graph_expansion", path="graph.md"),
            ]
        ),
        requested_budget=10_000,
        target_model="gpt-4o-mini",
    )

    budget = package["budget"]
    assert isinstance(budget, dict)
    allocation = budget["allocation"]
    assert isinstance(allocation, dict)
    assert allocation["direct_limit"] >= allocation["available_tokens"] * 0.60 - 1
    assert allocation["graph_used"] <= allocation["graph_limit"]
    assert allocation["metadata_used"] <= allocation["metadata_limit"]
    assert allocation["direct_used"] >= allocation["direct_limit"]


def test_context_package_backfills_unused_graph_budget_with_direct_content() -> None:
    package = build_context_package(
        raw([result("direct " * 8_000)]),
        requested_budget=4_000,
        target_model="gpt-4o-mini",
    )

    budget = package["budget"]
    assert isinstance(budget, dict)
    allocation = budget["allocation"]
    assert isinstance(allocation, dict)
    assert allocation["direct_used"] > allocation["direct_limit"]
    assert allocation["direct_used"] > (
        allocation["direct_limit"] + allocation["graph_limit"]
    )
    assert allocation["graph_used"] == 0


def test_fallback_tokenizer_is_conservative_and_preserves_safe_boundaries(
    monkeypatch,
) -> None:
    import personal_agent_memory.context_package as module

    original_import = module.importlib.import_module

    def unavailable(name: str):
        if name == "tiktoken":
            raise ImportError
        return original_import(name)

    monkeypatch.setattr(module.importlib, "import_module", unavailable)
    package = build_context_package(
        raw([result("多字节内容 " * 2_000), result("```python\n" + "x = 1\n" * 2_000 + "```")]),
        requested_budget=3_000,
        target_model="unavailable-model",
    )

    assert "tokenizer_fallback" in package["degradation"]
    budget = package["budget"]
    assert isinstance(budget, dict)
    assert budget["tokenizer"] == "conservative_utf8_bytes"
    encoded = json.dumps(package, ensure_ascii=False, separators=(",", ":"))
    assert len(encoded.encode("utf-8")) <= 3_000
    assert all(is_safe_markdown_fragment(item["content"]) for item in package["results"])
    assert all("tokenizer_fallback" in item["degradation"] for item in package["results"])


def test_truncation_preserves_tilde_fence_type_and_actual_length() -> None:
    counter = TokenCounter.for_model("gpt-4o-mini")
    complete = "Before.\n\n~~~~python\nprint('safe')\n~~~~\n\nAfter."
    assert counter.truncate(complete, counter.count(complete)) == (complete, False)
    assert is_safe_markdown_fragment(complete)

    inside_fence_limit = counter.count("Before.\n\n~~~~python\nprint")
    truncated, was_truncated = counter.truncate(complete, inside_fence_limit)
    assert was_truncated is True
    assert truncated == "Before."
    assert is_safe_markdown_fragment(truncated)

    short_closer = "Before.\n\n~~~~python\nprint('unsafe')\n~~~"
    cleaned, _ = counter.truncate(short_closer + (" tail" * 100), inside_fence_limit)
    assert cleaned == "Before."
    assert not is_safe_markdown_fragment(short_closer)

    info_is_not_a_closer = "Before.\n\n~~~~python\n~~~~nested\n~~~~"
    assert is_safe_markdown_fragment(info_is_not_a_closer)


def test_truncation_handles_fences_inside_blockquote_and_list_containers() -> None:
    counter = TokenCounter.for_model("gpt-4o-mini")
    complete = (
        "> Quoted.\n>\n> ~~~~python\n> print('quoted')\n> ~~~~\n\n"
        "- Listed\n\n  ~~~text\n  listed content\n  ~~~\n"
    )
    assert is_safe_markdown_fragment(complete)

    quote_limit = counter.count("> Quoted.\n>\n> ~~~~python\n> print")
    quote_truncated, _ = counter.truncate(complete, quote_limit)
    assert quote_truncated == "> Quoted.\n>"
    assert is_safe_markdown_fragment(quote_truncated)

    unclosed_list = "- Listed\n\n  ~~~~text\n  content\n  ~~~"
    assert not is_safe_markdown_fragment(unclosed_list)
    cleaned, _ = counter.truncate(unclosed_list + (" tail" * 100), counter.count(unclosed_list))
    assert cleaned == "- Listed"
    assert is_safe_markdown_fragment(cleaned)


def test_fence_closer_indentation_is_relative_to_commonmark_container_baseline() -> None:
    top_valid = "   ~~~text\ncontent\n   ~~~"
    top_invalid = "   ~~~text\ncontent\n    ~~~"
    quote_valid = ">   ````text\n> content\n> ````"
    quote_invalid = ">   ````text\n> content\n>     ````"
    list_valid = "- item\n\n     ~~~text\n     content\n     ~~~"
    list_invalid = "- item\n\n     ~~~text\n     content\n      ~~~"

    assert is_safe_markdown_fragment(top_valid)
    assert not is_safe_markdown_fragment(top_invalid)
    assert is_safe_markdown_fragment(quote_valid)
    assert not is_safe_markdown_fragment(quote_invalid)
    assert is_safe_markdown_fragment(list_valid)
    assert not is_safe_markdown_fragment(list_invalid)


def test_fence_closer_must_match_the_opening_commonmark_containers() -> None:
    top_level = "````text\n> ````\nstill fenced\n````"
    quote_list = "> - item\n>\n>   ~~~text\n>   quoted list content\n>   ~~~"
    nested_list_quote = (
        "- outer\n\n  > - inner\n  >\n  >   ```text\n"
        "  >   nested content\n  >   ```"
    )
    wrong_container = "> ```text\n> content\n```"

    assert is_safe_markdown_fragment(top_level)
    assert is_safe_markdown_fragment(quote_list)
    assert is_safe_markdown_fragment(nested_list_quote)
    assert not is_safe_markdown_fragment(wrong_container)


def test_fence_closer_supports_commonmark_tab_expansion_in_lists() -> None:
    counter = TokenCounter.for_model("gpt-4o-mini")
    samples = [
        "-\t```text\n\tx\n\t```",
        "- \t```text\n    x\n    ```",
        "-\t```text\n  \tx\n  \t```",
        "> -\t```text\n> \tx\n> \t```",
        "1.\t```text\n\tx\n\t```",
        "1. \t```text\n    x\n    ```",
    ]

    for sample in samples:
        assert is_safe_markdown_fragment(sample)
        suffix = "\n\n" + ("tail content " * 200)
        truncated, was_truncated = counter.truncate(
            sample + suffix, counter.count(sample) + 1
        )
        assert was_truncated is True
        assert truncated == sample
        assert is_safe_markdown_fragment(truncated)


def test_allocation_uses_actual_retained_content_tokens_after_safe_truncation() -> None:
    counter = TokenCounter.for_model("gpt-4o-mini")
    package = build_context_package(
        raw(
            [
                result("Intro.\n\n```python\n" + ("x = 1\n" * 8_000) + "```"),
                result("second direct " * 8_000, path="second.md"),
                result("graph evidence " * 8_000, classification="graph_expansion", path="g.md"),
            ]
        ),
        requested_budget=10_000,
        target_model="gpt-4o-mini",
    )
    allocation = package["budget"]["allocation"]
    direct_tokens = sum(
        counter.count(str(item["content"]))
        for item in package["results"]
        if item["classification"] == "direct"
    )
    graph_tokens = sum(
        counter.count(str(item["content"]))
        for item in package["results"]
        if item["classification"] == "graph_expansion"
    )
    assert allocation["direct_used"] == direct_tokens
    assert allocation["graph_used"] == graph_tokens
    assert any(item["path"] == "second.md" for item in package["results"])


def test_graph_is_omitted_when_direct_provenance_cannot_fit() -> None:
    oversized_direct = result("direct evidence", path=("very-long-path/" * 300) + "memory.md")
    graph = result("graph evidence", classification="graph_expansion", path="graph.md")
    package = build_context_package(
        raw([oversized_direct, graph]),
        requested_budget=4_000,
        target_model="gpt-4o-mini",
    )

    assert not any(item["classification"] == "graph_expansion" for item in package["results"])


def test_full_package_search_handles_json_escaping_and_recalculates_telemetry() -> None:
    counter = TokenCounter.for_model("gpt-4o-mini")
    package = build_context_package(
        raw([result(('"\\多字节 ' * 20_000) + "safe tail")]),
        requested_budget=4_000,
        target_model="gpt-4o-mini",
    )

    assert package["results"]
    assert package["results"][0]["truncated"] is True
    assert is_safe_markdown_fragment(package["results"][0]["content"])
    assert encoded_tokens(package) <= 4_000
    assert_conservative_telemetry(package, counter)
    assert package["budget"]["allocation"]["direct_used"] == sum(
        counter.count(str(item["content"]))
        for item in package["results"]
        if item["classification"] == "direct"
    )


def test_final_telemetry_obeys_strict_metadata_share() -> None:
    package = build_context_package(
        raw(
            [
                result("direct evidence " * 2_000),
                result(
                    "graph evidence " * 2_000,
                    classification="graph_expansion",
                    path="graph.md",
                ),
            ]
        ),
        requested_budget=1_959,
        target_model="gpt-4o-mini",
    )

    budget = package["budget"]
    allocation = budget["allocation"]
    assert allocation["metadata_limit"] == math.floor(
        allocation["available_tokens"] * 0.10
    )
    assert allocation["metadata_used"] <= allocation["metadata_limit"]
    assert_conservative_telemetry(package, TokenCounter.for_model("gpt-4o-mini"))
    assert not package["results"] or any(
        item["classification"] != "graph_expansion" for item in package["results"]
    )


def test_final_gpt_telemetry_uses_a_conservative_normalized_count() -> None:
    package = build_context_package(
        raw([result("direct evidence " * 2_000)]),
        requested_budget=1_080,
        target_model="gpt-4o-mini",
    )
    counter = TokenCounter.for_model("gpt-4o-mini")
    assert_conservative_telemetry(package, counter)


def test_final_fallback_telemetry_uses_a_conservative_normalized_count(
    monkeypatch,
) -> None:
    import personal_agent_memory.context_package as module

    original_import = module.importlib.import_module

    def unavailable(name: str):
        if name == "tiktoken":
            raise ImportError
        return original_import(name)

    monkeypatch.setattr(module.importlib, "import_module", unavailable)
    package = build_context_package(
        raw([result("多字节 direct evidence " * 2_000)]),
        requested_budget=4_583,
        target_model="unavailable-model",
    )
    assert_conservative_telemetry(package, TokenCounter.for_model("unavailable-model"))


def test_rest_and_mcp_budget_success_is_monotonic(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    roots = tmp_path / "libraries"
    memory = roots / "project"
    memory.mkdir(parents=True)
    (memory / "memory.md").write_text(
        "# Memory\n\nContinuousBudgetNeedle evidence.", encoding="utf-8"
    )

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(roots,)))) as client:
        key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}
        registered = client.post(
            "/api/v1/libraries",
            headers=headers,
            json={"path": str(memory), "kind": "project"},
        )
        assert registered.status_code == 201
        library_id = registered.json()["id"]

        for endpoint in ("/api/v1/search", "/mcp/search"):
            for model, final_budget in (
                ("gpt-4o-mini", 3_000),
                ("unsupported-tokenizer-model", 5_000),
            ):
                seen_success = False
                counter = TokenCounter.for_model(model)
                for token_budget in range(512, final_budget + 1):
                    response = client.post(
                        endpoint,
                        headers=headers,
                        json={
                            "library_id": library_id,
                            "query": "ContinuousBudgetNeedle",
                            "token_budget": token_budget,
                            "target_model": model,
                        },
                    )
                    assert response.status_code in (200, 422)
                    if response.status_code == 422:
                        assert not seen_success, (endpoint, model, token_budget)
                        continue
                    seen_success = True
                    package = response.json()
                    assert_conservative_telemetry(package, counter)
                    allocation = package["budget"]["allocation"]
                    assert allocation["graph_used"] <= allocation["graph_limit"]
                    assert not package["results"] or any(
                        item["classification"] != "graph_expansion"
                        for item in package["results"]
                    )
                assert seen_success


def test_search_contract_requires_explicit_scope_and_user_library(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    roots = tmp_path / "libraries"
    project_memory = roots / "project"
    user_memory = roots / "user"
    project = tmp_path / "project-root"
    project_memory.mkdir(parents=True)
    user_memory.mkdir()
    project.mkdir()
    (project_memory / "project.md").write_text(
        "# Project\n\nSharedNeedle project.", encoding="utf-8"
    )
    (user_memory / "user.md").write_text("# User\n\nSharedNeedle user.", encoding="utf-8")

    with TestClient(create_app(Settings(state_dir=state_dir, library_roots=(roots,)))) as client:
        key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {key}"}

        def register(path: Path, kind: str) -> dict[str, object]:
            response = client.post(
                "/api/v1/libraries", headers=headers, json={"path": str(path), "kind": kind}
            )
            assert response.status_code == 201
            return response.json()

        project_library = register(project_memory, "project")
        user_library = register(user_memory, "user")
        assert client.post(
            "/api/v1/project-bindings",
            headers=headers,
            json={"project_root": str(project), "library_id": project_library["id"]},
        ).status_code == 201

        missing = client.post("/mcp/search", headers=headers, json={"query": "SharedNeedle"})
        ambiguous = client.post(
            "/mcp/search",
            headers=headers,
            json={
                "cwd": str(project),
                "library_id": user_library["id"],
                "query": "SharedNeedle",
            },
        )
        assert missing.status_code == ambiguous.status_code == 422
        too_small = client.post(
            "/mcp/search",
            headers=headers,
            json={"cwd": str(project), "query": "SharedNeedle", "token_budget": 511},
        )
        assert too_small.status_code == 422
        fallback_too_small = client.post(
            "/mcp/search",
            headers=headers,
            json={
                "cwd": str(project),
                "query": "SharedNeedle",
                "token_budget": 512,
                "target_model": "unsupported-tokenizer-model",
            },
        )
        assert fallback_too_small.status_code == 422
        assert "complete memory-context-package/v1 envelope" in fallback_too_small.json()[
            "detail"
        ]

        unbound = tmp_path / "unbound"
        unbound.mkdir()
        blank_unbound = client.post(
            "/mcp/search",
            headers=headers,
            json={"cwd": str(unbound), "query": "   "},
        )
        assert blank_unbound.status_code == 422
        assert blank_unbound.json()["detail"] == "search query must not be empty"

        project_result = client.post(
            "/mcp/search",
            headers=headers,
            json={"cwd": str(project), "query": "SharedNeedle", "token_budget": 4_000},
        ).json()
        assert [item["path"] for item in project_result["results"]] == ["project.md"]
        assert project_result["scope"]["kind"] == "project_cwd"
        assert project_result["trust"]["classification"] == "untrusted_data"

        user_result = client.post(
            "/mcp/search",
            headers=headers,
            json={"library_id": user_library["id"], "query": "SharedNeedle"},
        ).json()
        assert [item["path"] for item in user_result["results"]] == ["user.md"]
        assert user_result["scope"] == {
            "kind": "explicit_library",
            "library_id": user_library["id"],
        }
        hit = user_result["results"][0]
        assert set(
            (
                "content",
                "library_id",
                "path",
                "heading",
                "start_line",
                "end_line",
                "source_type",
                "classification",
                "score",
                "source_version",
                "degraded",
                "degradation",
            )
        ).issubset(hit)
