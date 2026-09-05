from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelServiceError(RuntimeError):
    """A bounded model request failed without exposing request credentials."""


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    base_url: str
    model: str
    api_key_file: Path | None = None
    timeout_seconds: float = 2.0
    max_concurrency: int = 2
    retries: int = 1

    def __post_init__(self) -> None:
        base_url = self.base_url.rstrip("/")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("model base URL must use http or https")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("model credentials must use an API key file")
        if parsed.query or parsed.fragment:
            raise ValueError("model base URL must not contain a query or fragment")
        if not self.model.strip():
            raise ValueError("model name must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("model timeout must be positive")
        if self.max_concurrency < 1:
            raise ValueError("model concurrency must be at least one")
        if not 0 <= self.retries <= 3:
            raise ValueError("model retries must be between zero and three")
        object.__setattr__(self, "base_url", base_url)
        if self.api_key_file is not None:
            object.__setattr__(self, "api_key_file", self.api_key_file.expanduser().resolve())


class OpenAICompatibleClient:
    def __init__(
        self,
        embedding: ModelEndpoint | None,
        reranker: ModelEndpoint | None,
        graph: ModelEndpoint | None = None,
    ) -> None:
        self.embedding = embedding
        self.reranker = reranker
        self.graph = graph
        self._limits = {
            "embedding": threading.BoundedSemaphore(
                embedding.max_concurrency if embedding is not None else 1
            ),
            "reranker": threading.BoundedSemaphore(
                reranker.max_concurrency if reranker is not None else 1
            ),
            "graph": threading.BoundedSemaphore(
                graph.max_concurrency if graph is not None else 1
            ),
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        endpoint = self.embedding
        if endpoint is None:
            raise ModelServiceError("embedding is not configured")
        response = self._request(
            "embedding", endpoint, "/v1/embeddings", {"model": endpoint.model, "input": texts}
        )
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ModelServiceError("embedding returned malformed data")
        by_index: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                raise ModelServiceError("embedding returned malformed data")
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ModelServiceError("embedding returned malformed data")
            values: list[float] = []
            for value in vector:
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ModelServiceError("embedding returned malformed data")
                values.append(float(value))
            by_index[int(item["index"])] = values
        if set(by_index) != set(range(len(texts))):
            raise ModelServiceError("embedding returned malformed data")
        dimensions = {len(vector) for vector in by_index.values()}
        if len(dimensions) != 1:
            raise ModelServiceError("embedding returned inconsistent dimensions")
        return [by_index[index] for index in range(len(texts))]

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        endpoint = self.reranker
        if endpoint is None:
            raise ModelServiceError("reranker is not configured")
        response = self._request(
            "reranker",
            endpoint,
            "/v1/rerank",
            {"model": endpoint.model, "query": query, "documents": documents},
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise ModelServiceError("reranker returned malformed data")
        ordered: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise ModelServiceError("reranker returned malformed data")
            index = item.get("index")
            score = item.get("relevance_score")
            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(documents)
                or index in seen
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ModelServiceError("reranker returned malformed data")
            seen.add(index)
            ordered.append((index, float(score)))
        return ordered

    def extract_graph(self, document: str) -> dict[str, object]:
        endpoint = self.graph
        if endpoint is None:
            raise ModelServiceError("graph LLM is not configured")
        response = self._request(
            "graph",
            endpoint,
            "/v1/chat/completions",
            {
                "model": endpoint.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "graph-extraction-v1: return JSON with entities and relations; "
                            "each relation source and target must name an entity"
                        ),
                    },
                    {"role": "user", "content": document},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ModelServiceError("graph LLM returned malformed data")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ModelServiceError("graph LLM returned malformed data")
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise ModelServiceError("graph LLM returned malformed data")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as error:
            raise ModelServiceError("graph LLM returned malformed JSON") from error
        if not isinstance(decoded, dict):
            raise ModelServiceError("graph LLM returned malformed JSON")
        return decoded

    def extract_candidate(self, conversation: str) -> dict[str, object]:
        endpoint = self.graph
        if endpoint is None:
            raise ModelServiceError("candidate extraction LLM is not configured")
        response = self._request(
            "graph",
            endpoint,
            "/v1/chat/completions",
            {
                "model": endpoint.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "candidate-extraction-v1: return one JSON object with eligible, "
                            "suggested_type, body, confidence, evidence_kind, source_valid, "
                            "conflict, and policy_allowed. Only durable decisions, constraints, "
                            "preferences, domain facts, reusable experience, or external "
                            "references are eligible. Never include hidden reasoning, tool output, "
                            "complete files, injected memory, or transient task progress. "
                            "evidence_kind is user_confirmed only when the user message itself "
                            "confirms the fact; an assistant reply alone is never evidence."
                        ),
                    },
                    {"role": "user", "content": conversation},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ModelServiceError("candidate extraction returned malformed data")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ModelServiceError("candidate extraction returned malformed data")
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise ModelServiceError("candidate extraction returned malformed data")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as error:
            raise ModelServiceError("candidate extraction returned malformed JSON") from error
        if not isinstance(decoded, dict):
            raise ModelServiceError("candidate extraction returned malformed JSON")
        return decoded

    def _request(
        self,
        component: str,
        endpoint: ModelEndpoint,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, Any]:
        limit = self._limits[component]
        if not limit.acquire(timeout=endpoint.timeout_seconds):
            raise ModelServiceError(f"{component} concurrency limit timed out")
        try:
            headers = {"Content-Type": "application/json"}
            if endpoint.api_key_file is not None:
                try:
                    key = endpoint.api_key_file.read_text(encoding="utf-8").strip()
                except OSError as error:
                    raise ModelServiceError(f"{component} credential is unavailable") from error
                if not key:
                    raise ModelServiceError(f"{component} credential is empty")
                headers["Authorization"] = f"Bearer {key}"
            request_body = json.dumps(payload, separators=(",", ":")).encode()
            for attempt in range(endpoint.retries + 1):
                request = urllib.request.Request(
                    endpoint.base_url + path,
                    data=request_body,
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=endpoint.timeout_seconds
                    ) as response:
                        raw = response.read(4 * 1024 * 1024 + 1)
                    if len(raw) > 4 * 1024 * 1024:
                        raise ModelServiceError(f"{component} response is too large")
                    decoded = json.loads(raw)
                    if not isinstance(decoded, dict):
                        raise ModelServiceError(f"{component} returned malformed JSON")
                    return decoded
                except (
                    OSError,
                    TimeoutError,
                    urllib.error.HTTPError,
                    json.JSONDecodeError,
                ) as error:
                    if attempt >= endpoint.retries:
                        raise ModelServiceError(f"{component} request failed") from error
                    time.sleep(min(0.05 * (attempt + 1), endpoint.timeout_seconds))
            raise AssertionError("model retry loop did not return")
        finally:
            limit.release()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
