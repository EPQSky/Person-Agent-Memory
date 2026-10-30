from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jiuwen_memory.common.errors import BackendError
from jiuwen_memory.storage.base import StoreType
from jiuwen_memory.storage.fulltext import FulltextProducer, FulltextStore
from jiuwen_memory.storage.types import (
    Document,
    ScoredID,
    TextQuery,
    VectorQuery,
    VectorRecord,
)
from jiuwen_memory.storage.vector import VectorProducer, VectorStore

from personal_agent_memory.app import create_app
from personal_agent_memory.config import Settings
from personal_agent_memory.mem2_kernel import Mem2MemoryKernel
from personal_agent_memory.model_client import ModelEndpoint

LIBRARY_ID = "default"
MEMORY_PATH = "/api/v1/memories"
SEARCH_PATH = "/api/v1/search"
MCP_SEARCH_PATH = "/mcp/search"


def _semantic_vector(value: str, dimension: int = 1_536) -> list[float]:
    text = value.casefold()
    vector = [0.0] * dimension
    if any(word in text for word in ("deploy", "release", "ship", "production")):
        vector[0] = 1.0
    elif any(word in text for word in ("lunch", "recipe", "cook", "food")):
        vector[1] = 1.0
    else:
        vector[2] = 1.0
    return vector


class _EmbeddingHandler(BaseHTTPRequestHandler):
    fail = False
    dimension = 1_536
    requests: list[tuple[str, list[str], str]] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        raw_input = payload.get("input", [])
        inputs = [raw_input] if isinstance(raw_input, str) else [str(item) for item in raw_input]
        with type(self).lock:
            type(self).requests.append(
                (str(payload.get("model", "")), inputs, self.headers.get("Authorization", ""))
            )
        if type(self).fail:
            body = json.dumps({"error": {"message": "embedding unavailable"}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(
            {
                "data": [
                    {
                        "index": index,
                        "embedding": _semantic_vector(item, type(self).dimension),
                    }
                    for index, item in enumerate(inputs)
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def _embedding_server() -> Iterator[tuple[str, type[_EmbeddingHandler]]]:
    handler = _EmbeddingHandler
    handler.fail = False
    handler.dimension = 1_536
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


class _SearchFailingFulltext(FulltextStore):
    def __init__(self) -> None:
        self._documents: dict[tuple[tuple[str, str, str, str, str], str], Document] = {}

    @staticmethod
    def _key(scope: object, document_id: str) -> tuple[tuple[str, str, str, str, str], str]:
        values = tuple(
            str(getattr(scope, field)) for field in ("org", "space", "user", "agent", "session")
        )
        return values, document_id  # type: ignore[return-value]

    def store_type(self) -> StoreType:
        return StoreType.FULLTEXT

    def health(self) -> None:
        return None

    def insert(self, scope: object, docs: list[Document]) -> None:
        for document in docs:
            self._documents[self._key(scope, document.id)] = document

    def update(self, scope: object, docs: list[Document]) -> None:
        self.insert(scope, docs)

    def delete(self, scope: object, ids: list[str]) -> None:
        for document_id in ids:
            self._documents.pop(self._key(scope, document_id), None)

    def get(self, scope: object, ids: list[str]) -> list[Document]:
        return [
            self._documents[key]
            for document_id in ids
            if (key := self._key(scope, document_id)) in self._documents
        ]

    def search(self, scope: object, query: TextQuery) -> list[ScoredID]:
        raise BackendError("fulltext index is unavailable")


class _SearchFailingVector(VectorStore):
    def store_type(self) -> StoreType:
        return StoreType.VECTOR

    def health(self) -> None:
        return None

    def insert(self, scope: object, records: list[VectorRecord]) -> None:
        return None

    def update(self, scope: object, records: list[VectorRecord]) -> None:
        return None

    def delete(self, scope: object, ids: list[str]) -> None:
        return None

    def get(self, scope: object, ids: list[str]) -> list[VectorRecord]:
        return []

    def search(self, scope: object, query: VectorQuery) -> list[ScoredID]:
        raise BackendError("vector index is unavailable")


_registered = False


@pytest.fixture(autouse=True)
def _bypass_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")


def _register_test_backends() -> None:
    global _registered
    if _registered:
        return

    @FulltextProducer.register("test-search-failing-fulltext")  # type: ignore[untyped-decorator]
    def _build_fulltext(config: Any) -> FulltextStore:
        return _SearchFailingFulltext()

    @VectorProducer.register("test-search-failing-vector")  # type: ignore[untyped-decorator]
    def _build_vector(config: Any) -> VectorStore:
        return _SearchFailingVector()

    _registered = True


def _headers(state_dir: Path) -> dict[str, str]:
    key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {key}"}


def _endpoint(tmp_path: Path, base_url: str) -> ModelEndpoint:
    key_file = tmp_path / "embedding-api-key"
    key_file.write_text("test-secret\n", encoding="utf-8")
    return ModelEndpoint(
        base_url,
        "test-embedding-model",
        key_file,
        timeout_seconds=1.0,
        max_concurrency=1,
        retries=0,
    )


def _write(
    client: TestClient,
    headers: dict[str, str],
    content: str,
    key: str,
    *,
    library_id: str = LIBRARY_ID,
) -> str:
    response = client.post(
        MEMORY_PATH,
        headers=headers,
        json={
            "library_id": library_id,
            "content": content,
            "memory_type": "domain_fact",
            "source_references": ["test:mem2"],
            "actor": "test",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 201, response.text
    memory_id = response.json()["id"]
    assert isinstance(memory_id, str) and memory_id
    return memory_id


def _search(
    client: TestClient,
    headers: dict[str, str],
    query: str,
    *,
    path: str = SEARCH_PATH,
    token_budget: int = 10_000,
    library_id: str = LIBRARY_ID,
) -> dict[str, object]:
    response = client.post(
        path,
        headers=headers,
        json={
            "library_id": library_id,
            "query": query,
            "limit": 10,
            "token_budget": token_budget,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _app(
    state_dir: Path,
    *,
    embedding: ModelEndpoint | None = None,
    embedding_dimension: int = 1_536,
    kernel: Mem2MemoryKernel | None = None,
):
    settings = Settings(
        state_dir=state_dir,
        embedding=embedding,
        embedding_dimension=embedding_dimension,
    )
    return create_app(settings, memory_kernel=kernel)


def test_mem2_rest_and_mcp_write_search_health_without_project_binding(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with (
        _embedding_server() as (base_url, handler),
        TestClient(_app(state_dir, embedding=_endpoint(tmp_path, base_url))) as client,
    ):
        headers = _headers(state_dir)
        memory_id = _write(client, headers, "ORACLE-42 deployment rollback", "rest-write")

        health_response = client.get("/mcp/health", headers=headers)
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ready"
        assert health_response.json()["memory_kernel"] == "mem2"

        rest = _search(client, headers, "ORACLE-42")
        assert rest["status"] == "bound"
        assert rest["results"][0]["document_id"] == memory_id
        assert rest["vector_index_status"] == "ready"

        mcp = _search(client, headers, "rollback deployment", path=MCP_SEARCH_PATH)
        assert mcp["results"][0]["document_id"] == memory_id

    assert handler.requests
    assert {model for model, _, _ in handler.requests} == {"test-embedding-model"}
    assert {authorization for _, _, authorization in handler.requests} == {"Bearer test-secret"}


def test_mem2_vector_recall_uses_synonyms_and_filters_unrelated_memories(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with (
        _embedding_server() as (base_url, _),
        TestClient(_app(state_dir, embedding=_endpoint(tmp_path, base_url))) as client,
    ):
        headers = _headers(state_dir)
        deployment_id = _write(client, headers, "deploy release cycle", "deploy")
        lunch_id = _write(client, headers, "lunch recipe", "lunch")

        result = _search(client, headers, "ship to production")
        result_ids = {item["document_id"] for item in result["results"]}
        assert deployment_id in result_ids
        assert lunch_id not in result_ids
        assert lunch_id not in " ".join(item["content"] for item in result["results"])


def test_mem2_without_embedding_is_explicitly_degraded_without_vector(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        memory_id = _write(client, headers, "KEYWORD-NO-MODEL", "keyword-only")
        health = client.get("/mcp/health", headers=headers).json()
        assert health["status"] == "degraded"
        assert health["memory_kernel_status"] == "degraded"
        assert "embedding endpoint is not configured" in health["memory_kernel_detail"]

        result = _search(client, headers, "KEYWORD-NO-MODEL")
        assert result["results"][0]["document_id"] == memory_id
        assert result["vector_index_status"] == "not_configured"
        assert result["degraded"] is True
        assert result["degradation"] == [
            "embedding_not_configured",
            "graph_unavailable",
        ]


def test_mem2_restart_preserves_memory_and_search(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with _embedding_server() as (base_url, _):
        embedding = _endpoint(tmp_path, base_url)
        with TestClient(_app(state_dir, embedding=embedding)) as client:
            headers = _headers(state_dir)
            memory_id = _write(client, headers, "PERSIST-77 survives restarts", "restart-write")

        with TestClient(_app(state_dir, embedding=embedding)) as client:
            headers = _headers(state_dir)
            result = _search(client, headers, "PERSIST-77")
            assert result["results"][0]["document_id"] == memory_id


def test_mem2_fresh_state_writes_version_marker_without_touching_legacy_state(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    legacy_database = state_dir / "platform.sqlite3"
    legacy_database.write_bytes(b"legacy-state-must-not-be-read")

    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        assert client.get("/mcp/health", headers=headers).json()["memory_kernel"] == "mem2"
        regions = client.get("/api/v1/regions/tree", headers=headers)
        assert regions.status_code == 200
        assert regions.json()["root"]["region_id"] == "user"

    assert (state_dir / "mem2" / "VERSION").read_text(encoding="utf-8") == (
        "jiuwen-memory-0.2.0\n"
    )
    assert (state_dir / "mem2" / "regions.sqlite3").is_file()
    assert not (state_dir / "regions.sqlite3").exists()
    assert legacy_database.read_bytes() == b"legacy-state-must-not-be-read"


@pytest.mark.parametrize("state_kind", ["unversioned", "unsupported"])
def test_mem2_rejects_old_state_without_reading_or_rewriting_it(
    tmp_path: Path, state_kind: str
) -> None:
    state_dir = tmp_path / "state"
    mem2_dir = state_dir / "mem2"
    mem2_dir.mkdir(parents=True)
    sentinel = mem2_dir / "memory.sqlite3"
    sentinel.write_bytes(b"old-mem2-state-must-survive")
    version_path = mem2_dir / "VERSION"
    if state_kind == "unsupported":
        version_path.write_text("jiuwen-memory-0.1.0\n", encoding="utf-8")

    def files() -> dict[str, bytes]:
        return {
            path.relative_to(mem2_dir).as_posix(): path.read_bytes()
            for path in mem2_dir.rglob("*")
            if path.is_file()
        }

    before = files()
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        health = client.get("/mcp/health", headers=headers).json()
        assert health["memory_kernel"] == "mem2"
        assert health["memory_kernel_status"] == "unavailable"
        assert "reset the state directory explicitly" in health["memory_kernel_detail"]
        blocked_responses = [
            client.get("/api/v1/regions", headers=headers),
            client.post(
                MEMORY_PATH,
                headers=headers,
                json={
                    "region_id": "user",
                    "content": "must not touch rejected state",
                    "memory_type": "domain_fact",
                    "source_references": ["test:rejected-state"],
                    "actor": "test",
                    "idempotency_key": "rejected-state-write",
                },
            ),
            client.post(
                MEMORY_PATH,
                headers=headers,
                json={
                    "library_id": LIBRARY_ID,
                    "content": "must not create rejected idempotency state",
                    "memory_type": "domain_fact",
                    "source_references": ["test:rejected-state"],
                    "actor": "test",
                    "idempotency_key": "rejected-state-library-write",
                },
            ),
            client.get("/api/v1/promotions/does-not-exist", headers=headers),
            client.get("/api/v1/libraries/user/retention-policy", headers=headers),
            client.get(
                "/api/v1/libraries/user/retention-cleanup/preview", headers=headers
            ),
            client.post("/api/v1/libraries/user/retention-cleanup", headers=headers),
        ]
        for response in blocked_responses:
            assert response.status_code == 503, response.text
            assert "reset the state directory explicitly" in response.json()["detail"]

    assert files() == before
    for forbidden in ("regions.sqlite3", "maintenance.sqlite3", "idempotency.sqlite3"):
        assert not (mem2_dir / forbidden).exists()
    assert sentinel.read_bytes() == b"old-mem2-state-must-survive"


def test_mem2_search_respects_hard_token_budget(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with (
        _embedding_server() as (base_url, _),
        TestClient(_app(state_dir, embedding=_endpoint(tmp_path, base_url))) as client,
    ):
        headers = _headers(state_dir)
        _write(
            client,
            headers,
            "BUDGET-MARKER " + "z " * 20_000,
            "budget-write",
        )

        result = _search(client, headers, "BUDGET-MARKER")
        budget = result["budget"]
        assert budget["hard_limit_tokens"] == 10_000
        assert budget["effective_tokens"] <= 10_000
        assert budget["used_tokens"] <= budget["effective_tokens"]
        assert result["results"]
        assert result["results"][0]["truncated"] is True


def test_mem2_fulltext_failure_returns_vector_result_and_degradation(
    tmp_path: Path,
) -> None:
    _register_test_backends()
    state_dir = tmp_path / "state"
    with _embedding_server() as (base_url, _):
        embedding = _endpoint(tmp_path, base_url)
        kernel = Mem2MemoryKernel(
            state_dir,
            embedding=embedding,
            config_overrides={
                "fulltext_store": {
                    "default": {
                        "target": "test-search-failing-fulltext",
                        "params": {},
                    }
                }
            },
        )
        with TestClient(_app(state_dir, embedding=embedding, kernel=kernel)) as client:
            headers = _headers(state_dir)
            memory_id = _write(client, headers, "VECTOR-ONLY fallback token", "fulltext-failure")
            result = _search(client, headers, "VECTOR-ONLY fallback")
            assert result["results"][0]["document_id"] == memory_id
            assert result["degraded"] is True
            assert "fulltext_unavailable" in result["degradation"]


def test_mem2_vector_failure_returns_keyword_result_and_degradation(
    tmp_path: Path,
) -> None:
    _register_test_backends()
    state_dir = tmp_path / "state"
    with _embedding_server() as (base_url, _):
        embedding = _endpoint(tmp_path, base_url)
        kernel = Mem2MemoryKernel(
            state_dir,
            embedding=embedding,
            config_overrides={
                "vector_store": {
                    "default": {
                        "target": "test-search-failing-vector",
                        "params": {},
                    }
                }
            },
        )
        with TestClient(_app(state_dir, embedding=embedding, kernel=kernel)) as client:
            headers = _headers(state_dir)
            memory_id = _write(client, headers, "KEYWORD-ONLY fallback token", "vector-failure")
            result = _search(client, headers, "KEYWORD-ONLY")
            assert result["results"][0]["document_id"] == memory_id
            assert result["degraded"] is True
            assert "vector_unavailable" in result["degradation"]
            assert result["vector_index_status"] == "degraded"
            assert any(
                item["channel"] == "vector"
                for item in result["retrieval_errors"]
            )


def test_mem2_accepts_configured_1024_dimension(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with _embedding_server() as (base_url, handler):  # noqa: SIM117
        handler.dimension = 1_024
        with TestClient(
            _app(
                state_dir,
                embedding=_endpoint(tmp_path, base_url),
                embedding_dimension=1_024,
            )
        ) as client:
            headers = _headers(state_dir)
            memory_id = _write(client, headers, "DIMENSION-1024 release", "dimension-1024")
            health = client.get("/mcp/health", headers=headers).json()
            assert health["status"] == "ready"
            result = _search(client, headers, "DIMENSION-1024")
            assert result["results"][0]["document_id"] == memory_id
            assert result["vector_index_status"] == "ready"


def test_mem2_dimension_mismatch_is_degraded_and_never_reports_vector_ready(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with _embedding_server() as (base_url, handler):  # noqa: SIM117
        handler.dimension = 1_024
        with TestClient(
            _app(
                state_dir,
                embedding=_endpoint(tmp_path, base_url),
                embedding_dimension=1_536,
            )
        ) as client:
            headers = _headers(state_dir)
            memory_id = _write(client, headers, "DIMENSION-MISMATCH keyword", "mismatch")
            health = client.get("/mcp/health", headers=headers).json()
            assert health["status"] == "degraded"
            assert "embedding dimension mismatch" in health["memory_kernel_detail"]
            result = _search(client, headers, "DIMENSION-MISMATCH")
            assert result["results"][0]["document_id"] == memory_id
            assert result["degraded"] is True
            assert result["degradation"] == [
                "embedding_dimension_mismatch",
                "graph_unavailable",
            ]
            assert result["vector_index_status"] == "not_configured"


def test_mem2_enforces_fulltext_content_limit_at_rest_boundary(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        payload = {
            "library_id": LIBRARY_ID,
            "content": "x" * 65_535,
            "memory_type": "domain_fact",
            "source_references": ["test:mem2"],
            "actor": "test",
            "idempotency_key": "at-limit",
        }
        accepted = client.post(MEMORY_PATH, headers=headers, json=payload)
        assert accepted.status_code == 201, accepted.text

        rejected = client.post(
            MEMORY_PATH,
            headers=headers,
            json={**payload, "content": "x" * 70_000, "idempotency_key": "too-long"},
        )
        assert rejected.status_code == 422



def test_mem2_idempotency_persists_and_conflicts_on_changed_content(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with _embedding_server() as (base_url, _):
        embedding = _endpoint(tmp_path, base_url)
        with TestClient(_app(state_dir, embedding=embedding)) as client:
            headers = _headers(state_dir)
            first_id = _write(client, headers, "IDEMPOTENT original", "same-key")

        with TestClient(_app(state_dir, embedding=embedding)) as client:
            headers = _headers(state_dir)
            second_id = _write(client, headers, "IDEMPOTENT original", "same-key")
            assert second_id == first_id
            other_library_id = _write(
                client,
                headers,
                "IDEMPOTENT original",
                "same-key",
                library_id="other-library",
            )
            assert other_library_id != first_id

            conflict = client.post(
                MEMORY_PATH,
                headers=headers,
                json={
                    "library_id": LIBRARY_ID,
                    "content": "IDEMPOTENT changed",
                    "memory_type": "domain_fact",
                    "source_references": ["test:mem2"],
                    "actor": "test",
                    "idempotency_key": "same-key",
                },
            )
            assert conflict.status_code == 409


def test_mem2_pending_idempotency_completion_never_duplicates_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    kernel = Mem2MemoryKernel(state_dir)
    completion_calls = 0

    def fail_completion(
        library_id: str,
        idempotency_key: str,
        request_hash: str,
        memory_id: str,
    ) -> None:
        del library_id, idempotency_key, request_hash, memory_id
        nonlocal completion_calls
        completion_calls += 1
        raise RuntimeError("injected idempotency completion failure")

    monkeypatch.setattr(kernel, "_complete_idempotency", fail_completion)
    request = {
        "library_id": LIBRARY_ID,
        "content": "PENDING idempotency marker",
        "memory_type": "domain_fact",
        "source_references": ["test:mem2"],
        "actor": "test",
        "idempotency_key": "pending-key",
    }
    with TestClient(_app(state_dir, kernel=kernel)) as client:
        headers = _headers(state_dir)
        first = client.post(MEMORY_PATH, headers=headers, json=request)
        assert first.status_code == 503
        assert "idempotency completion is pending" in first.json()["detail"]

    replacement_kernel = Mem2MemoryKernel(state_dir)
    with TestClient(_app(state_dir, kernel=replacement_kernel)) as client:
        headers = _headers(state_dir)
        retry = client.post(MEMORY_PATH, headers=headers, json=request)
        assert retry.status_code == 409
        assert "incomplete earlier memory write" in retry.json()["detail"]

        result = _search(client, headers, "PENDING")
        assert [item["document_id"] for item in result["results"]] != []
        assert len(result["results"]) == 1

    assert completion_calls == 1


def test_mem2_cwd_and_explicit_library_scopes_do_not_collide(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    cwd = "/tmp/cwd-library-collision"
    colliding_library_id = hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:24]
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        memory_id = _write(
            client,
            headers,
            "EXPLICIT-LIBRARY-ONLY marker",
            "namespace-collision",
            library_id=colliding_library_id,
        )

        explicit = _search(
            client, headers, "EXPLICIT-LIBRARY-ONLY", library_id=colliding_library_id
        )
        assert explicit["results"][0]["document_id"] == memory_id

        cwd_search = client.post(
            SEARCH_PATH,
            headers=headers,
            json={
                "cwd": cwd,
                "query": "EXPLICIT-LIBRARY-ONLY",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert cwd_search.status_code == 200, cwd_search.text
        assert cwd_search.json()["results"] == []


def test_mem2_normalizes_library_id_once_for_write_and_search(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    request = {
        "library_id": " default ",
        "content": "NORMALIZED library marker",
        "memory_type": "domain_fact",
        "source_references": ["test:mem2"],
        "actor": "test",
        "idempotency_key": "normalized-library",
    }
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        first = client.post(MEMORY_PATH, headers=headers, json=request)
        assert first.status_code == 201, first.text
        assert first.json()["library_id"] == "default"

        retry = client.post(MEMORY_PATH, headers=headers, json={**request, "library_id": "default"})
        assert retry.status_code == 201, retry.text
        assert retry.json()["id"] == first.json()["id"]
        assert retry.json()["library_id"] == "default"

        result = _search(client, headers, "NORMALIZED", library_id=" default ")
        assert result["library_id"] == "default"
        assert result["scope"]["library_id"] == "default"
        assert result["results"][0]["document_id"] == first.json()["id"]
        assert result["entity_graph_status"]["status"] == "not_configured"

        max_library_id = "x" * 248
        padded = client.post(
            MEMORY_PATH,
            headers=headers,
            json={
                **request,
                "library_id": f"  {max_library_id}  ",
                "content": "PADDED-LIBRARY marker",
                "idempotency_key": "padded-library",
            },
        )
        assert padded.status_code == 201, padded.text
        assert padded.json()["library_id"] == max_library_id

        blank_add = client.post(
            MEMORY_PATH,
            headers=headers,
            json={**request, "library_id": "   ", "idempotency_key": "blank-library"},
        )
        assert blank_add.status_code == 422

        blank_search = client.post(
            SEARCH_PATH,
            headers=headers,
            json={
                "library_id": "   ",
                "query": "NORMALIZED",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert blank_search.status_code == 422


def test_mem2_concurrent_same_idempotency_key_creates_one_memory(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        barrier = threading.Barrier(2)

        def write_same_request() -> object:
            barrier.wait(timeout=10)
            return client.post(
                MEMORY_PATH,
                headers=headers,
                json={
                    "library_id": LIBRARY_ID,
                    "content": "CONCURRENT idempotency marker",
                    "memory_type": "domain_fact",
                    "source_references": ["test:mem2"],
                    "actor": "test",
                    "idempotency_key": "concurrent-key",
                },
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = [
                future.result(timeout=30)
                for future in (pool.submit(write_same_request), pool.submit(write_same_request))
            ]

        assert [response.status_code for response in responses] == [201, 201]
        ids = {response.json()["id"] for response in responses}
        assert len(ids) == 1
        assert {response.json()["status"] for response in responses} == {
            "created",
            "existing",
        }

        result = _search(client, headers, "CONCURRENT")
        assert [item["document_id"] for item in result["results"]] == [ids.pop()]


def test_mem2_library_scope_limit_is_consistent_for_add_and_search(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    max_library_id = "x" * 248
    oversized_library_id = "x" * 249
    with TestClient(_app(state_dir)) as client:
        headers = _headers(state_dir)
        memory_id = _write(
            client,
            headers,
            "MAX-LIBRARY marker",
            "max-library",
            library_id=max_library_id,
        )
        result = _search(
            client,
            headers,
            "MAX-LIBRARY",
            library_id=max_library_id,
        )
        assert result["results"][0]["document_id"] == memory_id

        payload = {
            "library_id": oversized_library_id,
            "content": "oversized library",
            "memory_type": "domain_fact",
            "source_references": ["test:mem2"],
            "actor": "test",
            "idempotency_key": "oversized-library",
        }
        rejected_add = client.post(MEMORY_PATH, headers=headers, json=payload)
        assert rejected_add.status_code == 422

        rejected_search = client.post(
            SEARCH_PATH,
            headers=headers,
            json={
                "library_id": oversized_library_id,
                "query": "oversized",
                "limit": 10,
                "token_budget": 10_000,
            },
        )
        assert rejected_search.status_code == 422


def test_mem2_audit_history_is_isolated_by_library_and_survives_restart(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with _embedding_server() as (base_url, _):
        embedding = _endpoint(tmp_path, base_url)
        with TestClient(_app(state_dir, embedding=embedding)) as client:
            headers = _headers(state_dir)
            _write(client, headers, "LIB-A audit marker", "lib-a", library_id="lib-a")
            _write(client, headers, "LIB-B audit marker", "lib-b", library_id="lib-b")

        with TestClient(_app(state_dir, embedding=embedding)) as client:
            headers = _headers(state_dir)
            history_a = client.get("/api/v1/libraries/lib-a/history", headers=headers).json()
            history_b = client.get("/api/v1/libraries/lib-b/history", headers=headers).json()

    assert history_a
    assert history_b
    assert {event["target"] for event in history_a} == {"library:lib-a"}
    assert {event["target"] for event in history_b} == {"library:lib-b"}
    assert {event["id"] for event in history_a}.isdisjoint({event["id"] for event in history_b})


def test_mem2_model_failure_uses_public_rest_and_mcp_degraded_fallback(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    with (
        _embedding_server() as (base_url, handler),
        TestClient(_app(state_dir, embedding=_endpoint(tmp_path, base_url))) as client,
    ):
        headers = _headers(state_dir)
        memory_id = _write(
            client,
            headers,
            "MODEL-FALLBACK deterministic keyword",
            "model-failure",
        )
        handler.fail = True

        rest = _search(client, headers, "MODEL-FALLBACK")
        assert rest["results"][0]["document_id"] == memory_id
        assert rest["degraded"] is True
        assert "model_unavailable" in rest["degradation"]

        mcp = _search(client, headers, "MODEL-FALLBACK", path=MCP_SEARCH_PATH)
        assert mcp["results"][0]["document_id"] == memory_id
        assert mcp["degraded"] is True
        assert "model_unavailable" in mcp["degradation"]
