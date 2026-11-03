"""Task 4 - LLM Gateway: per-tenant token rate limiting and model fallback.

POST /v1/chat/completions
  1. authenticate the tenant API key            -> 401 authentication_error
  2. validate the body                          -> 400 invalid_request_error
  3. reserve estimated tokens (sliding window)  -> 429 rate_limit_exceeded + Retry-After
  4. route: primary, then secondary on 429/timeout/5xx
  5. success: reconcile reservation to real usage; failure: release it

Config (env)
  TENANT_KEYS          "sk-key-a:tenant-a,sk-key-b:tenant-b"
  RATE_LIMIT_TOKENS    default 50000 (per tenant per minute)
  RATE_LIMIT_DB        default ./data/gateway.sqlite3
  PRIMARY_BASE_URL / PRIMARY_MODEL / PRIMARY_API_KEY
  SECONDARY_BASE_URL / SECONDARY_MODEL / SECONDARY_API_KEY
  PRIMARY_TIMEOUT_MS   default 3000
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from task4_router import errors
from task4_router.errors import GatewayError, error_response
from task4_router.rate_limiter import RateLimitExceeded, RequestTooLarge, SlidingWindowTokenLimiter, estimate_tokens
from task4_router.router import ModelRouter, Provider

logger = logging.getLogger("llm-gateway")


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(pattern="^(system|user|assistant|tool|developer)$")
    content: Any = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # pass through temperature, top_p, ...
    model: str | None = None
    messages: list[Message] = Field(min_length=1, max_length=1000)
    max_tokens: int | None = Field(default=None, ge=1, le=32_000)
    stream: bool = False


def parse_tenant_keys(spec: str) -> dict[str, str]:
    """'key:tenant,...' -> {sha256(key): tenant}. Raw keys are never kept in memory or on disk."""
    out = {}
    for pair in filter(None, (p.strip() for p in spec.split(","))):
        key, _, tenant = pair.partition(":")
        if key and tenant:
            out[hashlib.sha256(key.encode()).hexdigest()] = tenant
    return out


def create_app(
    providers: list[Provider] | None = None,
    tenant_keys: dict[str, str] | None = None,
    limiter: SlidingWindowTokenLimiter | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float | None = None,
) -> FastAPI:
    providers = providers or [
        Provider(
            "primary",
            os.environ.get("PRIMARY_BASE_URL", "http://127.0.0.1:9201"),
            os.environ.get("PRIMARY_MODEL", "primary-model"),
            os.environ.get("PRIMARY_API_KEY", ""),
        ),
        Provider(
            "secondary",
            os.environ.get("SECONDARY_BASE_URL", "http://127.0.0.1:9202"),
            os.environ.get("SECONDARY_MODEL", "secondary-model"),
            os.environ.get("SECONDARY_API_KEY", ""),
        ),
    ]
    tenant_keys = tenant_keys if tenant_keys is not None else parse_tenant_keys(os.environ.get("TENANT_KEYS", ""))
    limiter = limiter or SlidingWindowTokenLimiter(
        os.environ.get("RATE_LIMIT_DB", "./data/gateway.sqlite3"),
        limit_tokens=int(os.environ.get("RATE_LIMIT_TOKENS", "50000")),
    )
    timeout_s = timeout_s if timeout_s is not None else int(os.environ.get("PRIMARY_TIMEOUT_MS", "3000")) / 1000

    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(timeout_s + 1),  # asyncio.timeout in the router is the real deadline
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    router = ModelRouter(providers, client, timeout_s=timeout_s)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()
        limiter.close()

    app = FastAPI(title="LLM Gateway - rate limit & fallback", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.limiter = limiter

    def request_id_of(request: Request) -> str:
        if not hasattr(request.state, "request_id"):
            request.state.request_id = f"req_{uuid.uuid4().hex[:20]}"
        return request.state.request_id

    # ---- every error path goes through the same sanitized envelope ----

    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError):
        return error_response(exc, request_id_of(request))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return error_response(errors.invalid_request("Malformed request body."), request_id_of(request))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        type_ = "not_found" if exc.status_code == 404 else "invalid_request_error"
        return error_response(GatewayError(exc.status_code, type_, "Request could not be served."), request_id_of(request))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("req=%s unhandled error", request_id_of(request))  # full trace stays in logs
        return error_response(errors.internal_error(), request_id_of(request))

    def authenticate(request: Request) -> str:
        scheme, _, key = (request.headers.get("authorization") or "").partition(" ")
        if scheme.lower() != "bearer" or not key:
            raise errors.unauthorized()
        digest = hashlib.sha256(key.strip().encode()).hexdigest()
        for known, tenant in tenant_keys.items():
            if hmac.compare_digest(known, digest):
                return tenant
        raise errors.unauthorized()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        rid = request_id_of(request)
        tenant = authenticate(request)

        try:
            body = await request.json()
            chat = ChatRequest.model_validate(body)
        except (ValueError, ValidationError):
            raise errors.invalid_request("Malformed request body.") from None
        if chat.stream:
            raise errors.invalid_request("Streaming is not supported on this endpoint.")

        estimate = estimate_tokens([m.model_dump() for m in chat.messages], chat.max_tokens)
        try:
            reservation = await limiter.reserve(tenant, estimate)
        except RequestTooLarge:
            raise errors.invalid_request("Request exceeds the per-minute token budget.") from None
        except RateLimitExceeded as exc:
            logger.info("req=%s tenant=%s limited used=%s req=%s", rid, tenant, exc.used, exc.requested)
            return error_response(
                errors.rate_limited(exc.retry_after),
                rid,
                {"x-ratelimit-limit-tokens": str(exc.limit), "x-ratelimit-remaining-tokens": str(max(0, exc.limit - exc.used))},
            )

        try:
            result = await router.complete(chat.model_dump(exclude_none=True), rid)
        except BaseException:
            await limiter.release(reservation)  # no completion -> don't bill the tenant's window
            raise

        usage = result.body.get("usage") or {}
        actual = usage.get("total_tokens")
        if isinstance(actual, int) and actual >= 0:
            await limiter.reconcile(reservation, actual)
            remaining = reservation.remaining + reservation.tokens - actual
        else:
            remaining = reservation.remaining

        return JSONResponse(
            result.body,
            headers={
                "x-request-id": rid,
                "x-gateway-provider": result.provider,
                "x-ratelimit-limit-tokens": str(limiter.limit),
                "x-ratelimit-remaining-tokens": str(max(0, remaining)),
            },
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "9200")))
