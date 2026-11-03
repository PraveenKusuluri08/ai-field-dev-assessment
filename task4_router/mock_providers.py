"""Mock OpenAI-compatible providers for demoing failover by hand.

    python -m task4_router.mock_providers primary 9201
    python -m task4_router.mock_providers secondary 9202

Switch behaviour at runtime:
    curl -X POST localhost:9201/_mode/429      # ok | 429 | slow | 500
"""

from __future__ import annotations

import asyncio
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODES = {"ok", "429", "slow", "500"}


def create_app(name: str, mode: str = "ok", slow_seconds: float = 5.0) -> FastAPI:
    app = FastAPI(title=f"mock provider {name}")
    app.state.mode = mode

    @app.post("/_mode/{mode}")
    async def set_mode(mode: str):
        if mode not in MODES:
            return JSONResponse({"error": f"mode must be one of {sorted(MODES)}"}, status_code=400)
        app.state.mode = mode
        return {"mode": mode}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        mode = app.state.mode
        if mode == "429":
            return JSONResponse({"error": {"message": "quota exhausted"}}, status_code=429, headers={"retry-after": "7"})
        if mode == "500":
            return JSONResponse({"error": "Traceback (most recent call last): internal provider details"}, status_code=500)
        if mode == "slow":
            await asyncio.sleep(slow_seconds)
        prompt = sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4 + 1
        completion = 12
        return {
            "id": f"cmpl-{name}-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": f"Hello from {name}."}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
        }

    return app


if __name__ == "__main__":
    import uvicorn

    provider_name = sys.argv[1] if len(sys.argv) > 1 else "primary"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9201
    uvicorn.run(create_app(provider_name), host="127.0.0.1", port=port)
