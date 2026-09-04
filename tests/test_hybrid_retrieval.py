from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from personal_agent_memory.app import create_app
from personal_agent_memory.config import ConfigurationError, Settings
from personal_agent_memory.model_client import ModelEndpoint, OpenAICompatibleClient
from personal_agent_memory.state import PlatformState


class DeterministicModelHandler(BaseHTTPRequestHandler):
    modes = {"embedding": "ok", "reranker": "ok"}
    authorizations: list[str] = []
    embedding_requests = 0
    embedding_started = threading.Event()
    active_embeddings = 0
    max_active_embeddings = 0
    counter_lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        component = "embedding" if self.path == "/v1/embeddings" else "reranker"
        self.authorizations.append(self.headers.get("Authorization", ""))
        mode = self.modes[component]
        if component == "embedding":
            type(self).embedding_started.set()
            with type(self).counter_lock:
                type(self).active_embeddings += 1
                type(self).max_active_embeddings = max(
                    type(self).max_active_embeddings, type(self).active_embeddings
                )
        if mode in {"delay", "timeout", "slow"}:
            time.sleep({"delay": 0.1, "timeout": 0.2, "slow": 2.0}[mode])
        if mode == "error":
            if component == "embedding":
                with type(self).counter_lock:
                    type(self).active_embeddings -= 1
            self.send_response(503)
            self.end_headers()
            return
        if component == "embedding":
            type(self).embedding_requests += 1
            if mode == "malformed":
                response: dict[str, Any] = {"data": [{"index": 0, "embedding": "bad"}]}
            else:
                inputs = payload["input"]
                request_number = type(self).embedding_requests
                response = {
                    "data": [
                        {
                            "index": index,
                            "embedding": self.vector_for_mode(
                                str(value), mode, request_number
                            ),
                        }
                        for index, value in enumerate(inputs)
                    ]
                }
        elif mode == "malformed":
            response = {"results": [{"index": -1, "relevance_score": "bad"}]}
        else:
            documents = payload["documents"]
            indices = (
                list(reversed(range(len(documents))))
                if mode == "reverse"
                else list(range(len(documents)))
            )
            response = {
                "results": [
                    {"index": index, "relevance_score": float(len(documents) - rank)}
                    for rank, index in enumerate(indices)
                ]
            }
        if component == "embedding":
            with type(self).counter_lock:
                type(self).active_embeddings -= 1
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def vector(value: str) -> list[float]:
        text = value.casefold()
        return [
            float(sum(word in text for word in ("deploy", "release", "production", "ship"))),
            float(sum(word in text for word in ("rollback", "reverse", "undo", "revert"))),
            float(sum(word in text for word in ("database", "storage", "sqlite"))),
        ]

    @classmethod
    def vector_for_mode(cls, value: str, mode: str, request_number: int) -> list[float]:
        vector = cls.vector(value)
        if mode == "dimension_two":
            return vector[:2]
        if mode == "cross_batch_dimension" and request_number % 2 == 0:
            return vector[:2]
        if mode == "nan":
            return [float("nan"), *vector[1:]]
        return vector

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def deterministic_model() -> Iterator[tuple[str, type[DeterministicModelHandler]]]:
    handler = DeterministicModelHandler
    handler.modes = {"embedding": "ok", "reranker": "ok"}
    handler.authorizations = []
    handler.embedding_requests = 0
    handler.embedding_started = threading.Event()
    handler.active_embeddings = 0
    handler.max_active_embeddings = 0
    handler.counter_lock = threading.Lock()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def wait_for_job(client: TestClient, headers: dict[str, str], job_id: int) -> str:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()["status"]
        if status in {"done", "error"}:
            return str(status)
        time.sleep(0.02)
    raise AssertionError("background vector job did not finish")


def test_hybrid_retrieval_rebuild_versioning_and_degradation(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project"
    project = tmp_path / "project"
    library_path.mkdir(parents=True)
    project.mkdir()
    source = library_path / "knowledge.md"
    source.write_text(
        """# Operations

Deploy releases to production with a rollback plan.

Ship a release after approval.

Use command `pam-sync --force` when ExactErrorE42 appears.

# Storage

SQLite keeps local durable state.
""",
        encoding="utf-8",
    )
    secret = "embedding-secret-never-persist"
    embedding_key = tmp_path / "embedding.key"
    reranker_key = tmp_path / "reranker.key"
    embedding_key.write_text(secret, encoding="utf-8")
    reranker_key.write_text("reranker-secret", encoding="utf-8")

    with deterministic_model() as (base_url, handler):
        embedding = ModelEndpoint(base_url, "embed-v1", embedding_key, 0.05, 1, 1)
        reranker = ModelEndpoint(base_url, "rerank-v1", reranker_key, 0.05, 1, 1)
        settings = Settings(
            state_dir=state_dir,
            library_roots=(library_root,),
            embedding=embedding,
            reranker=reranker,
        )
        with TestClient(create_app(settings)) as client:
            key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
            headers = {"Authorization": f"Bearer {key}"}
            registered = client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(library_path), "kind": "project"},
            ).json()
            client.post(
                "/api/v1/project-bindings",
                headers=headers,
                json={"project_root": str(project), "library_id": registered["id"]},
            )
            rebuilt = client.post(
                f"/api/v1/libraries/{registered['id']}/vector-index/rebuild",
                headers=headers,
            )
            assert rebuilt.status_code == 202
            assert wait_for_job(client, headers, rebuilt.json()["job_id"]) == "done"

            semantic = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "reverse a production release"},
            ).json()
            assert semantic["results"]
            assert "rollback plan" in semantic["results"][0]["content"]
            assert "vector" in semantic["results"][0]["retrieval_sources"]
            assert semantic["degraded"] is False
            pre_rerank_ids = [item["chunk_id"] for item in semantic["results"]]

            handler.modes["reranker"] = "reverse"
            reranked = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "reverse a production release"},
            ).json()
            assert [item["chunk_id"] for item in reranked["results"]] == list(
                reversed(pre_rerank_ids)
            )

            exact = client.post(
                "/mcp/search",
                headers=headers,
                json={"cwd": str(project), "query": "pam-sync --force ExactErrorE42"},
            ).json()
            assert any("full_text" in item["retrieval_sources"] for item in exact["results"])
            assert len({item["chunk_id"] for item in exact["results"]}) == len(exact["results"])

            handler.modes["reranker"] = "malformed"
            rerank_fallback = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "reverse a production release"},
            ).json()
            assert rerank_fallback["degradation"] == ["reranker_unavailable"]
            assert [item["chunk_id"] for item in rerank_fallback["results"]] == pre_rerank_ids

            handler.modes["embedding"] = "error"
            fts_fallback = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "ExactErrorE42"},
            ).json()
            assert "embedding_unavailable" in fts_fallback["degradation"]
            assert fts_fallback["results"][0]["path"] == "knowledge.md"

            handler.modes = {"embedding": "ok", "reranker": "ok"}
            old_versions = {
                row[0]
                for row in sqlite3.connect(state_dir / "platform.sqlite3").execute(
                    "SELECT source_version FROM memory_chunk_vectors"
                )
            }
            source.write_text("# Current\n\nUse FreshExact99 for storage.\n", encoding="utf-8")
            client.post(f"/api/v1/libraries/{registered['id']}/scan", headers=headers)
            current = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "FreshExact99"},
            ).json()
            assert current["results"][0]["content"] == "Use FreshExact99 for storage."
            assert current["results"][0]["source_version"] not in old_versions
            placeholders = ",".join("?" for _ in old_versions)
            assert not sqlite3.connect(state_dir / "platform.sqlite3").execute(
                f"SELECT 1 FROM memory_chunk_vectors "
                f"WHERE source_version IN ({placeholders})",
                tuple(old_versions),
            ).fetchall()

            refresh = client.post(
                f"/api/v1/libraries/{registered['id']}/vector-index/rebuild",
                headers=headers,
            ).json()
            assert wait_for_job(client, headers, refresh["job_id"]) == "done"
            handler.modes["embedding"] = "timeout"
            started = time.monotonic()
            timeout_fallback = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "FreshExact99"},
            ).json()
            assert time.monotonic() - started < 0.5
            assert timeout_fallback["results"]
            assert "embedding_unavailable" in timeout_fallback["degradation"]

        assert "Bearer " + secret in handler.authorizations
        assert secret.encode() not in (state_dir / "platform.sqlite3").read_bytes()
        assert secret.encode() not in source.read_bytes()


def test_model_credentials_cannot_be_placed_in_a_memory_root(tmp_path: Path) -> None:
    library_root = tmp_path / "libraries"
    credential = library_root / "secret.md"
    endpoint = ModelEndpoint("http://127.0.0.1:9999", "model", credential)

    try:
        Settings(
            state_dir=tmp_path / "state",
            library_roots=(library_root,),
            embedding=endpoint,
        )
    except ConfigurationError as error:
        assert "outside memory library roots" in str(error)
    else:
        raise AssertionError("credential path inside a library root was accepted")


def test_model_client_enforces_embedding_concurrency_limit() -> None:
    with deterministic_model() as (base_url, handler):
        handler.modes["embedding"] = "delay"
        client = OpenAICompatibleClient(
            ModelEndpoint(base_url, "embed-v1", timeout_seconds=1, max_concurrency=2),
            None,
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            vectors = list(executor.map(lambda index: client.embed([str(index)]), range(8)))

        assert len(vectors) == 8
        assert handler.max_active_embeddings == 2


def test_slow_models_do_not_block_state_queries_or_fts(tmp_path: Path) -> None:
    async def scenario(base_url: str, handler: type[DeterministicModelHandler]) -> None:
        library_root = tmp_path / "libraries"
        vector_library = library_root / "vector"
        fts_library = library_root / "fts"
        vector_library.mkdir(parents=True)
        fts_library.mkdir()
        (vector_library / "vector.md").write_text(
            "# Release\n\nDeploy to production with rollback.\n", encoding="utf-8"
        )
        (fts_library / "exact.md").write_text(
            "# Exact\n\nWorkerResponsiveNeedle is current.\n", encoding="utf-8"
        )
        vector_project = tmp_path / "vector-project"
        fts_project = tmp_path / "fts-project"
        vector_project.mkdir()
        fts_project.mkdir()
        state = PlatformState(
            tmp_path / "platform.sqlite3",
            (library_root,),
            OpenAICompatibleClient(
                ModelEndpoint(base_url, "embed-v1", timeout_seconds=0.4, max_concurrency=1),
                None,
            ),
        )
        await state.start()
        try:
            vector = state.register_library(str(vector_library), "project")
            fts = state.register_library(str(fts_library), "project")
            state.bind_project(str(vector_project), vector.id)
            state.bind_project(str(fts_project), fts.id)
            vector_job = state.enqueue_vector_rebuild(vector.id)
            fts_job = state.enqueue_vector_rebuild(fts.id)
            while state.job_status(vector_job) not in {"done", "error"}:
                await asyncio.sleep(0.02)
            while state.job_status(fts_job) not in {"done", "error"}:
                await asyncio.sleep(0.02)
            state.connection_or_raise.execute(
                "DELETE FROM memory_vector_indexes WHERE library_id = ?", (fts.id,)
            )
            state.connection_or_raise.commit()

            handler.modes["embedding"] = "timeout"
            handler.embedding_started.clear()
            slow_search = asyncio.create_task(
                state.search_project(str(vector_project), "production rollback")
            )
            deadline = time.monotonic() + 1
            while not handler.embedding_started.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert handler.embedding_started.is_set()

            worker_job = state.enqueue_job("foundation-check", "{}")
            started = time.monotonic()
            fts_result = await state.search_project(
                str(fts_project), "WorkerResponsiveNeedle"
            )
            assert time.monotonic() - started < 0.15
            assert fts_result["results"][0]["path"] == "exact.md"  # type: ignore[index]
            deadline = time.monotonic() + 0.15
            while state.job_status(worker_job) != "done" and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert state.job_status(worker_job) == "done"
            await slow_search
        finally:
            await state.close()

    with deterministic_model() as (base_url, handler):
        asyncio.run(scenario(base_url, handler))


def test_vector_dimension_contract_is_atomic_and_persists_across_restart(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project"
    project = tmp_path / "project"
    library_path.mkdir(parents=True)
    project.mkdir()
    source = library_path / "many.md"
    source.write_text(
        "\n".join(
            f"# Item {index}\n\nDeploy release {index} with rollback."
            for index in range(40)
        ),
        encoding="utf-8",
    )

    with deterministic_model() as (base_url, handler):
        endpoint = ModelEndpoint(base_url, "embed-v1", timeout_seconds=0.2)
        settings = Settings(
            state_dir=state_dir,
            library_roots=(library_root,),
            embedding=endpoint,
        )
        with TestClient(create_app(settings)) as client:
            key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
            headers = {"Authorization": f"Bearer {key}"}
            library = client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(library_path), "kind": "project"},
            ).json()
            client.post(
                "/api/v1/project-bindings",
                headers=headers,
                json={"project_root": str(project), "library_id": library["id"]},
            )
            first = client.post(
                f"/api/v1/libraries/{library['id']}/vector-index/rebuild", headers=headers
            ).json()
            assert wait_for_job(client, headers, first["job_id"]) == "done"
            database = sqlite3.connect(state_dir / "platform.sqlite3")
            original_vectors = database.execute(
                "SELECT chunk_id, vector_json FROM memory_chunk_vectors ORDER BY chunk_id"
            ).fetchall()
            original_metadata = database.execute(
                "SELECT model, dimension FROM memory_vector_indexes WHERE library_id = ?",
                (library["id"],),
            ).fetchone()
            database.close()
            assert original_metadata == ("embed-v1", 3)

            handler.embedding_requests = 0
            handler.modes["embedding"] = "cross_batch_dimension"
            failed = client.post(
                f"/api/v1/libraries/{library['id']}/vector-index/rebuild", headers=headers
            ).json()
            assert wait_for_job(client, headers, failed["job_id"]) == "error"
            database = sqlite3.connect(state_dir / "platform.sqlite3")
            assert database.execute(
                "SELECT chunk_id, vector_json FROM memory_chunk_vectors ORDER BY chunk_id"
            ).fetchall() == original_vectors
            assert database.execute(
                "SELECT model, dimension FROM memory_vector_indexes WHERE library_id = ?",
                (library["id"],),
            ).fetchone() == original_metadata
            database.close()
            after_failure = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "Deploy release 1"},
            ).json()
            assert "vector_index_unavailable" in after_failure["degradation"]
            assert all(
                "vector" not in item["retrieval_sources"] for item in after_failure["results"]
            )

            handler.modes["embedding"] = "dimension_two"
            rebuilt = client.post(
                f"/api/v1/libraries/{library['id']}/vector-index/rebuild", headers=headers
            ).json()
            assert wait_for_job(client, headers, rebuilt["job_id"]) == "done"
            handler.modes["embedding"] = "ok"
            drifted = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "Deploy release 1"},
            ).json()
            assert "embedding_unavailable" in drifted["degradation"]
            assert all("vector" not in item["retrieval_sources"] for item in drifted["results"])

            handler.modes["embedding"] = "nan"
            non_finite = client.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "Deploy release 1"},
            ).json()
            assert "embedding_unavailable" in non_finite["degradation"]
            assert all(
                "vector" not in item["retrieval_sources"] for item in non_finite["results"]
            )

        handler.modes["embedding"] = "dimension_two"
        with TestClient(create_app(settings)) as restarted:
            persisted = restarted.post(
                "/api/v1/search",
                headers=headers,
                json={"cwd": str(project), "query": "ship production release"},
            ).json()
            assert "vector" in persisted["results"][0]["retrieval_sources"]


def test_vector_rebuild_requests_and_scans_coalesce_per_library(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    library_root = tmp_path / "libraries"
    library_path = library_root / "project"
    library_path.mkdir(parents=True)
    source = library_path / "knowledge.md"
    source.write_text("# Current\n\nQueueNeedle 0.\n", encoding="utf-8")

    with deterministic_model() as (base_url, handler):
        settings = Settings(
            state_dir=state_dir,
            library_roots=(library_root,),
            embedding=ModelEndpoint(base_url, "embed-v1", timeout_seconds=3),
        )
        with TestClient(create_app(settings)) as client:
            key = (state_dir / "api-key").read_text(encoding="utf-8").strip()
            headers = {"Authorization": f"Bearer {key}"}
            library = client.post(
                "/api/v1/libraries",
                headers=headers,
                json={"path": str(library_path), "kind": "project"},
            ).json()
            initial = client.post(
                f"/api/v1/libraries/{library['id']}/vector-index/rebuild", headers=headers
            ).json()
            assert wait_for_job(client, headers, initial["job_id"]) == "done"

            handler.modes["embedding"] = "slow"
            handler.embedding_started.clear()
            running = client.post(
                f"/api/v1/libraries/{library['id']}/vector-index/rebuild", headers=headers
            ).json()
            assert handler.embedding_started.wait(1)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if client.get(
                    f"/api/v1/jobs/{running['job_id']}", headers=headers
                ).json()["status"] == "running":
                    break
                time.sleep(0.01)

            with ThreadPoolExecutor(max_workers=8) as executor:
                responses = list(
                    executor.map(
                        lambda _: client.post(
                            f"/api/v1/libraries/{library['id']}/vector-index/rebuild",
                            headers=headers,
                        ).json(),
                        range(60),
                    )
                )
            assert len({response["job_id"] for response in responses}) == 1
            for version in range(1, 11):
                source.write_text(
                    f"# Current\n\nQueueNeedle {version}.\n", encoding="utf-8"
                )
                assert client.post(
                    f"/api/v1/libraries/{library['id']}/scan", headers=headers
                ).status_code == 200

            database = sqlite3.connect(state_dir / "platform.sqlite3")
            active = database.execute(
                "SELECT status FROM background_jobs WHERE kind = 'vector_rebuild' "
                "AND json_extract(payload, '$.library_id') = ? "
                "AND status IN ('running', 'pending')",
                (library["id"],),
            ).fetchall()
            database.close()
            assert sorted(row[0] for row in active) == ["pending", "running"]
