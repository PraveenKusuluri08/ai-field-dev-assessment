"""Task 2 - MCP Security Gateway (HTTP JSON-RPC reverse proxy).

    AI agent ──HTTP + Bearer JWT──▶ gateway ──HTTP──▶ downstream MCP server

Request pipeline
  1. Authenticate: ``Authorization: Bearer <jwt>`` (HS256) -> ``Principal(sub, role)``.
     A missing, expired or malformed token, or an unknown role, gets HTTP 401
     and never reaches the downstream server.
  2. Parse the JSON-RPC wire format: a single message or a batch array.
     Malformed JSON gives -32700; a structurally invalid message gives -32600.
  3. Authorize each message:
       tools/list  -> forwarded transparently
       tools/call  -> if params.name starts with "admin_" the role must be admin,
                      otherwise answered locally with -32001 "Unauthorized Tool Call"
       (other methods are forwarded; policy lives in one function, ``authorize``)
  4. Forward the allowed messages. A single request's response is streamed back
     unchanged (JSON or SSE). For a batch, denied entries are answered locally
     and merged with the downstream responses for the allowed entries.

The client's Authorization header is *not* forwarded. The downstream server
receives the verified identity as ``X-Gateway-Subject`` / ``X-Gateway-Role``.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("mcp-gateway")

ROLES = {"admin", "viewer"}
ADMIN_TOOL_PREFIX = "admin_"
MAX_BODY_BYTES = 1 * 1024 * 1024

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED_TOOL_CALL = -32001
UNAUTHENTICATED = -32000

# headers passed through in each direction (everything else is dropped)
FORWARD_REQUEST_HEADERS = {"content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id"}
FORWARD_RESPONSE_HEADERS = {"content-type", "mcp-session-id", "cache-control"}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str


class AuthError(Exception):
    pass


def jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def authenticate(header: str | None, secret: str) -> Principal:
    if not header:
        raise AuthError("missing Authorization header")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'")
    try:
        claims = jwt.decode(
            token.strip(),
            secret,
            algorithms=["HS256"],  # pinned: rejects alg=none and algorithm-confusion tokens
            options={"require": ["sub", "role", "exp"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from None
    role = claims.get("role")
    if role not in ROLES:
        raise AuthError("token carries an unknown role")
    return Principal(subject=str(claims["sub"]), role=role)


# ---------------------------------------------------------------------------
# Authorization policy
# ---------------------------------------------------------------------------


def is_admin_tool(name: str) -> bool:
    # case/whitespace-insensitive so "Admin_reset_key" or " admin_x" cannot slip past
    return name.strip().lower().startswith(ADMIN_TOOL_PREFIX)


def authorize(msg: dict, principal: Principal) -> dict | None:
    """Return a JSON-RPC error response if ``msg`` must be blocked, else None (forward)."""
    method = msg.get("method")
    if method != "tools/call":
        return None  # tools/list and everything else are forwarded transparently

    params = msg.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not name.strip():
        # fail closed: don't let the downstream server guess at a malformed call
        return jsonrpc_error(msg.get("id"), INVALID_PARAMS, "Invalid params: tools/call requires a string 'name'")

    if is_admin_tool(name) and principal.role != "admin":
        logger.warning("DENY tools/call name=%s sub=%s role=%s", name, principal.subject, principal.role)
        return jsonrpc_error(
            msg.get("id"),
            UNAUTHORIZED_TOOL_CALL,
            "Unauthorized Tool Call",
            {"tool": name, "required_role": "admin"},
        )
    logger.info("ALLOW tools/call name=%s sub=%s role=%s", name, principal.subject, principal.role)
    return None


def validate_message(msg: Any) -> dict | None:
    """Minimal JSON-RPC 2.0 structural check. Returns an error response or None."""
    if not isinstance(msg, dict):
        return jsonrpc_error(None, INVALID_REQUEST, "Invalid Request")
    req_id = msg.get("id")
    if msg.get("jsonrpc") != "2.0":
        return jsonrpc_error(req_id, INVALID_REQUEST, "Invalid Request: jsonrpc must be '2.0'")
    if "method" in msg:
        if not isinstance(msg["method"], str):
            return jsonrpc_error(req_id, INVALID_REQUEST, "Invalid Request: method must be a string")
        if "params" in msg and not isinstance(msg["params"], (dict, list)):
            return jsonrpc_error(req_id, INVALID_REQUEST, "Invalid Request: params must be object or array")
    elif "result" not in msg and "error" not in msg:  # client->server responses (e.g. elicitation) are fine
        return jsonrpc_error(req_id, INVALID_REQUEST, "Invalid Request")
    return None


def expects_response(msg: dict) -> bool:
    return "method" in msg and "id" in msg  # notifications and responses get no reply


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(
    downstream_url: str | None = None,
    jwt_secret: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    downstream_url = downstream_url or os.environ.get("DOWNSTREAM_MCP_URL", "http://127.0.0.1:9001/mcp")
    jwt_secret = jwt_secret or os.environ.get("GATEWAY_JWT_SECRET", "")
    if len(jwt_secret) < 32:
        raise RuntimeError("GATEWAY_JWT_SECRET must be set to at least 32 characters")

    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="MCP Security Gateway", docs_url=None, redoc_url=None, lifespan=lifespan)

    def upstream_headers(request: Request, principal: Principal) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQUEST_HEADERS}
        headers["x-gateway-subject"] = principal.subject
        headers["x-gateway-role"] = principal.role
        if request.client:
            headers["x-forwarded-for"] = request.client.host
        return headers

    def downstream_failure(req_id: Any) -> JSONResponse:
        return JSONResponse(jsonrpc_error(req_id, INTERNAL_ERROR, "Downstream MCP server unavailable"), status_code=502)

    async def stream_downstream(method: str, headers: dict[str, str], body: bytes | None, req_id: Any) -> Response:
        """Forward and stream the downstream response back unchanged (works for JSON and SSE)."""
        req = client.build_request(method, downstream_url, headers=headers, content=body)
        try:
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            logger.error("downstream error: %r", exc)
            return downstream_failure(req_id)

        async def body_iter():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        out_headers = {k: v for k, v in upstream.headers.items() if k.lower() in FORWARD_RESPONSE_HEADERS}
        return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=out_headers)

    def unauthenticated(reason: str) -> JSONResponse:
        logger.info("401: %s", reason)
        return JSONResponse(
            jsonrpc_error(None, UNAUTHENTICATED, "Unauthorized: valid Bearer token required"),
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'},
        )

    @app.post("/mcp")
    async def mcp_post(request: Request) -> Response:
        try:
            principal = authenticate(request.headers.get("authorization"), jwt_secret)
        except AuthError as exc:
            return unauthenticated(str(exc))

        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse(jsonrpc_error(None, INVALID_REQUEST, "Request too large"), status_code=413)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(jsonrpc_error(None, PARSE_ERROR, "Parse error"), status_code=400)

        headers = upstream_headers(request, principal)

        # ---- single message: authorize, then stream through untouched ----
        if not isinstance(payload, list):
            denial = validate_message(payload) or authorize(payload, principal)
            if denial is not None:
                return JSONResponse(denial)  # intercepted: downstream is never called
            return await stream_downstream("POST", headers, body, payload.get("id"))

        # ---- batch: split into locally-answered and forwarded entries ----
        if not payload:
            return JSONResponse(jsonrpc_error(None, INVALID_REQUEST, "Invalid Request: empty batch"), status_code=400)
        local: list[dict] = []
        forward: list[dict] = []
        for msg in payload:
            denial = validate_message(msg) or authorize(msg, principal)
            if denial is None:
                forward.append(msg)
            elif not isinstance(msg, dict) or expects_response(msg) or "method" not in msg:
                local.append(denial)

        remote: list[dict] = []
        if forward:
            try:
                resp = await client.post(downstream_url, headers=headers, content=json.dumps(forward))
            except httpx.HTTPError as exc:
                logger.error("downstream error: %r", exc)
                return downstream_failure(None)
            if resp.content:
                try:
                    parsed = resp.json()
                except ValueError:
                    logger.error("downstream returned non-JSON for batch (status %s)", resp.status_code)
                    return downstream_failure(None)
                remote = parsed if isinstance(parsed, list) else [parsed]

        merged = remote + local
        if not merged:
            return Response(status_code=202)  # batch of notifications only
        return JSONResponse(merged)

    @app.api_route("/mcp", methods=["GET", "DELETE"])
    async def mcp_stream_or_terminate(request: Request) -> Response:
        """GET opens the server->client SSE stream; DELETE ends a session. Auth required, no body."""
        try:
            principal = authenticate(request.headers.get("authorization"), jwt_secret)
        except AuthError as exc:
            return unauthenticated(str(exc))
        return await stream_downstream(request.method, upstream_headers(request, principal), None, None)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


def mint_token(subject: str, role: str, secret: str, ttl_seconds: int = 3600) -> str:
    """Helper for local testing / demos."""
    import time

    now = int(time.time())
    return jwt.encode({"sub": subject, "role": role, "iat": now, "exp": now + ttl_seconds}, secret, algorithm="HS256")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "9000")))
