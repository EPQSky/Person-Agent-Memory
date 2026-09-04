from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Deterministic OpenAI-compatible fake model")
MODES = {"embedding": "ok", "reranker": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": "deterministic-test-model", "object": "model"}]}


@app.post("/control/{component}/{mode}")
def control(component: str, mode: str) -> dict[str, str]:
    if component not in MODES or mode not in {"ok", "error", "malformed", "timeout"}:
        raise HTTPException(status_code=422, detail="unsupported deterministic control")
    MODES[component] = mode
    return {"component": component, "mode": mode}


def _apply_mode(component: str) -> str:
    mode = MODES[component]
    if mode == "timeout":
        time.sleep(1.0)
    if mode == "error":
        raise HTTPException(status_code=503, detail="deterministic failure")
    return mode


def _vector(value: str) -> list[float]:
    text = value.casefold()
    groups = (
        ("deploy", "release", "ship", "production"),
        ("rollback", "revert", "undo", "reverse"),
        ("database", "sqlite", "storage", "persistence"),
        ("timeout", "latency", "slow", "deadline"),
    )
    semantic = [float(sum(token in text for token in group)) for group in groups]
    return semantic


@app.post("/v1/chat/completions")
def chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    content = "deterministic-response"
    if payload.get("messages"):
        content = str(payload["messages"][-1].get("content", content))
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }


@app.post("/v1/embeddings")
def embeddings(payload: dict[str, Any]) -> dict[str, Any]:
    if _apply_mode("embedding") == "malformed":
        return {"data": [{"index": 0, "embedding": "invalid"}]}
    inputs = payload.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": _vector(str(value))}
            for index, value in enumerate(inputs)
        ],
    }


@app.post("/v1/rerank")
def rerank(payload: dict[str, Any]) -> dict[str, Any]:
    if _apply_mode("reranker") == "malformed":
        return {"results": [{"index": -1, "relevance_score": "invalid"}]}
    documents = payload.get("documents", [])
    return {
        "results": [
            {"index": index, "relevance_score": float(len(str(document)))}
            for index, document in sorted(
                enumerate(documents), key=lambda item: len(str(item[1])), reverse=True
            )
        ]
    }
