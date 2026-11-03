"""Mock downstream MCP server (HTTP JSON-RPC) for exercising the gateway.

It records every message it receives in ``app.state.received`` (with the
headers it saw), so tests can prove that a denied call never arrived.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

TOOLS = [
    {"name": "get_customer_record", "description": "Read a customer", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Rotate a tenant API key", "inputSchema": {"type": "object"}},
]


def handle(msg: dict) -> dict | None:
    if "id" not in msg:
        return None  # notification
    method = msg.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mock-downstream", "version": "0.1"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = msg["params"]["name"]
        result = {"content": [{"type": "text", "text": f"executed {name}"}], "isError": False}
    else:
        return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "Method not found"}}
    return {"jsonrpc": "2.0", "id": msg["id"], "result": result}


def create_app() -> FastAPI:
    app = FastAPI(title="Mock downstream MCP server")
    app.state.received = []

    @app.post("/mcp")
    async def mcp(request: Request) -> Response:
        payload = await request.json()
        messages = payload if isinstance(payload, list) else [payload]
        for m in messages:
            app.state.received.append({"message": m, "headers": dict(request.headers)})
        replies = [r for r in (handle(m) for m in messages) if r is not None]
        if not replies:
            return Response(status_code=202)
        return JSONResponse(replies if isinstance(payload, list) else replies[0], headers={"mcp-session-id": "sess-1"})

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9001")))
