from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from personal_agent_memory.model_client import (
    ModelEndpoint,
    ModelTaskProfile,
    OpenAICompatibleClient,
)


class _ModelContractHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict[str, Any]]] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        with type(self).lock:
            type(self).requests.append((self.path, payload))
        if self.path == "/v1/embeddings":
            response: dict[str, object] = {
                "data": [{"index": index, "embedding": [1.0, 0.0]} for index, _ in enumerate(
                    payload["input"]
                )]
            }
        elif self.path == "/v1/rerank":
            response = {
                "results": [
                    {"index": index, "relevance_score": 1.0}
                    for index, _ in enumerate(payload["documents"])
                ]
            }
        else:
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"entities": [], "relations": []})
                        }
                    }
                ]
            }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def _model_contract_server() -> Iterator[tuple[str, type[_ModelContractHandler]]]:
    handler = _ModelContractHandler
    handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_extraction_tasks_send_deterministic_profiles_with_bounded_output() -> None:
    with _model_contract_server() as (base_url, handler):
        client = OpenAICompatibleClient(
            ModelEndpoint(base_url, "embedding-model"),
            ModelEndpoint(base_url, "reranker-model"),
            ModelEndpoint(base_url, "graph-model"),
        )

        client.extract_graph("Entity: alpha")
        client.extract_candidate("User: choose alpha")

        chat_requests = [
            payload for path, payload in handler.requests if path == "/v1/chat/completions"
        ]
        graph = chat_requests[0]
        assert graph["temperature"] == 0
        assert graph["top_p"] == 1
        assert graph["max_tokens"] == 4096
        assert "top_k" not in graph

        candidate = chat_requests[1]
        assert candidate["temperature"] == 0
        assert candidate["top_p"] == 1
        assert candidate["max_tokens"] == 2048
        assert "top_k" not in candidate


def test_embedding_and_reranking_payloads_have_no_generation_parameters() -> None:
    with _model_contract_server() as (base_url, handler):
        client = OpenAICompatibleClient(
            ModelEndpoint(base_url, "embedding-model"),
            ModelEndpoint(base_url, "reranker-model"),
        )

        assert client.embed(["alpha"]) == [[1.0, 0.0]]
        assert client.rerank("alpha", ["alpha"]) == [(0, 1.0)]

        for path, payload in handler.requests:
            assert path in {"/v1/embeddings", "/v1/rerank"}
            assert not {"temperature", "top_p", "top_k", "max_tokens"} & payload.keys()


def test_task_profiles_are_independent_and_optional_top_k_is_not_forced() -> None:
    with _model_contract_server() as (base_url, handler):
        client = OpenAICompatibleClient(
            None,
            None,
            ModelEndpoint(base_url, "graph-model"),
            graph_profile=ModelTaskProfile(max_tokens=111, top_k=7),
            candidate_profile=ModelTaskProfile(max_tokens=222),
        )

        client.extract_graph("Entity: alpha")
        client.extract_candidate("User: choose alpha")

        chat_requests = [
            payload for path, payload in handler.requests if path == "/v1/chat/completions"
        ]
        assert chat_requests[0]["max_tokens"] == 111
        assert chat_requests[0]["top_k"] == 7
        assert chat_requests[1]["max_tokens"] == 222
        assert "top_k" not in chat_requests[1]
