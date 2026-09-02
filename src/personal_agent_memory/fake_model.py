from __future__ import annotations

from typing import Any

from fastapi import FastAPI

app = FastAPI(title="Deterministic OpenAI-compatible fake model")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": "deterministic-test-model", "object": "model"}]}


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
    inputs = payload.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": [float(len(str(value))), 1.0]}
            for index, value in enumerate(inputs)
        ],
    }


@app.post("/v1/rerank")
def rerank(payload: dict[str, Any]) -> dict[str, Any]:
    documents = payload.get("documents", [])
    return {
        "results": [
            {"index": index, "relevance_score": float(len(str(document)))}
            for index, document in sorted(
                enumerate(documents), key=lambda item: len(str(item[1])), reverse=True
            )
        ]
    }
