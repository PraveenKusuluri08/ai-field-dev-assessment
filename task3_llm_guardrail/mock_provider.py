"""Mock OpenAI-compatible provider that streams the last user message back.

Text is streamed in small, irregular chunks (so PII gets split mid-token), with
an optional per-chunk delay to make latency observable.
Request extras: ``mock_chunk_size`` (int), ``mock_delay_ms`` (int).
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def create_app() -> FastAPI:
    app = FastAPI(title="Mock LLM provider")

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        text = next((m["content"] for m in reversed(body["messages"]) if m.get("role") == "user"), "")
        size = max(1, int(body.get("mock_chunk_size", 3)))
        delay = int(body.get("mock_delay_ms", 0)) / 1000
        cid, created, model = "chatcmpl-mock", int(time.time()), body.get("model", "mock-1")

        if not body.get("stream"):
            return JSONResponse(
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                }
            )

        def chunk(delta: dict, finish=None) -> str:
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        async def gen():
            yield chunk({"role": "assistant", "content": ""})
            for i in range(0, len(text), size):
                if delay:
                    await asyncio.sleep(delay)
                yield chunk({"content": text[i : i + size]})
            yield chunk({}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9101")))
