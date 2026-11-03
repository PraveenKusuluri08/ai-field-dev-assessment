"""Task 3 - LLM Gateway with a streaming PII-redaction guardrail.

``POST /v1/chat/completions`` (OpenAI-compatible) is proxied to an upstream
provider. Streamed responses pass through ``StreamRedactor`` delta by delta:

    upstream SSE ──▶ parse event ──▶ per-choice StreamRedactor ──▶ re-emit SSE
                    (one at a time)  (holds back a few chars)      (immediately)

* Nothing is accumulated. Each upstream event is handled and forwarded before
  the next is read. Per-stream state is one small hold-back buffer per choice.
* Back-pressure is end-to-end: an async generator over ``httpx`` ``aiter_lines``
  means a slow client slows the upstream read, with no unbounded queue in between.
* If the client disconnects, the generator is cancelled and the
  ``async with client.stream(...)`` closes the upstream connection, so tokens
  are not paid for when nobody is reading them.
* Upstream errors are answered with a generic gateway error. The provider's
  body is logged, not returned.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from task3_llm_guardrail.redactor import StreamRedactor, redact_text

logger = logging.getLogger("llm-guardrail")


def sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def error_body(message: str, type_: str = "upstream_error") -> dict:
    return {"error": {"type": type_, "message": message}}


class RedactingStream:
    """Turns upstream OpenAI-style SSE lines into redacted SSE events."""

    def __init__(self) -> None:
        self.redactors: dict[int, StreamRedactor] = {}
        self.last_chunk: dict[str, Any] | None = None  # template for the final flush event

    def _redactor(self, idx: int) -> StreamRedactor:
        if idx not in self.redactors:
            self.redactors[idx] = StreamRedactor()
        return self.redactors[idx]

    def process(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Redact one parsed chunk in place. Returns None if nothing is left to send."""
        self.last_chunk = chunk
        keep = bool(chunk.get("usage"))
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            red = self._redactor(choice.get("index", 0))
            text = red.feed(delta["content"]) if isinstance(delta.get("content"), str) else ""
            if choice.get("finish_reason") is not None:
                text += red.flush()  # choice is done: release its held-back tail
            if "content" in delta or text:
                delta["content"] = text
                choice["delta"] = delta
            if text or choice.get("finish_reason") is not None or set(delta) - {"content"}:
                keep = True  # role / tool_calls / finish_reason are worth forwarding even with no text
        return chunk if keep else None

    def flush_all(self) -> dict[str, Any] | None:
        """Emit whatever is still held (upstream ended without finish_reason)."""
        choices = []
        for idx, red in self.redactors.items():
            if red.pending:
                choices.append({"index": idx, "delta": {"content": red.flush()}, "finish_reason": None})
        if not choices:
            return None
        base = dict(self.last_chunk or {"object": "chat.completion.chunk"})
        base["choices"] = choices
        base.pop("usage", None)
        return base


def create_app(
    upstream_base_url: str | None = None,
    upstream_api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    upstream_base_url = (upstream_base_url or os.environ.get("UPSTREAM_BASE_URL", "http://127.0.0.1:9101")).rstrip("/")
    upstream_api_key = upstream_api_key if upstream_api_key is not None else os.environ.get("UPSTREAM_API_KEY", "")
    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="LLM Gateway - PII guardrail", docs_url=None, redoc_url=None, lifespan=lifespan)

    def upstream_headers() -> dict[str, str]:
        h = {"content-type": "application/json", "accept": "text/event-stream"}
        if upstream_api_key:
            h["authorization"] = f"Bearer {upstream_api_key}"
        return h

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(error_body("Request body must be JSON", "invalid_request"), status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse(error_body("'messages' is required", "invalid_request"), status_code=400)

        url = f"{upstream_base_url}/v1/chat/completions"

        if not body.get("stream"):
            return await non_streaming(url, body)

        # Open the upstream stream *before* returning, so upstream HTTP errors
        # become a proper status code instead of a broken 200 stream.
        req = client.build_request("POST", url, headers=upstream_headers(), json=body)
        try:
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            logger.error("upstream connect failed: %r", exc)
            return JSONResponse(error_body("Upstream provider unavailable"), status_code=502)
        if upstream.status_code != 200:
            detail = (await upstream.aread())[:500]
            await upstream.aclose()
            logger.error("upstream status=%s body=%r", upstream.status_code, detail)
            status = upstream.status_code if upstream.status_code in (400, 401, 403, 404, 429) else 502
            return JSONResponse(error_body(f"Upstream provider returned {upstream.status_code}"), status_code=status)

        return StreamingResponse(
            relay(upstream),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    async def relay(upstream: httpx.Response) -> AsyncIterator[bytes]:
        state = RedactingStream()
        try:
            async for line in upstream.aiter_lines():
                if not line.startswith("data:"):
                    continue  # blank separators, comments, `event:` lines
                data = line[5:].strip()
                if data == "[DONE]":
                    tail = state.flush_all()
                    if tail:
                        yield sse(json.dumps(tail))
                    yield sse("[DONE]")
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("dropping unparseable upstream event")
                    continue
                out = state.process(chunk)
                if out is not None:
                    yield sse(json.dumps(out, separators=(",", ":")))
            # upstream closed without [DONE]: still release held text
            tail = state.flush_all()
            if tail:
                yield sse(json.dumps(tail))
        except httpx.HTTPError as exc:
            logger.error("upstream stream broke: %r", exc)
            tail = state.flush_all()
            if tail:
                yield sse(json.dumps(tail))
            yield sse(json.dumps(error_body("Upstream stream interrupted")))
        finally:
            await upstream.aclose()

    async def non_streaming(url: str, body: dict) -> JSONResponse:
        try:
            resp = await client.post(url, headers=upstream_headers(), json=body)
        except httpx.HTTPError as exc:
            logger.error("upstream failed: %r", exc)
            return JSONResponse(error_body("Upstream provider unavailable"), status_code=502)
        if resp.status_code != 200:
            logger.error("upstream status=%s body=%r", resp.status_code, resp.text[:500])
            return JSONResponse(error_body(f"Upstream provider returned {resp.status_code}"), status_code=502)
        data = resp.json()
        for choice in data.get("choices", []):
            msg = choice.get("message") or {}
            if isinstance(msg.get("content"), str):
                msg["content"] = redact_text(msg["content"])
        return JSONResponse(data)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "9100")))
