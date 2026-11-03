"""Task 2 tests: gateway in front of the mock downstream, wired in-process via ASGI transports."""

import json
import time

import httpx
import jwt
import pytest

from task2_mcp_gateway.gateway import create_app, mint_token
from task2_mcp_gateway.mock_downstream import create_app as create_downstream

SECRET = "test-secret-that-is-at-least-32-chars!!"


@pytest.fixture
def downstream():
    return create_downstream()


@pytest.fixture
async def gw(downstream):
    app = create_app(
        downstream_url="http://downstream/mcp",
        jwt_secret=SECRET,
        transport=httpx.ASGITransport(app=downstream),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c


def auth(role: str, sub: str = "user-1") -> dict:
    return {"Authorization": f"Bearer {mint_token(sub, role, SECRET)}"}


def call(name, req_id=1):
    return {"jsonrpc": "2.0", "id": req_id, "method": "tools/call", "params": {"name": name, "arguments": {}}}


LIST = {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}}


# --- authentication ----------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        None,
        "Basic abc",
        "Bearer",
        "Bearer not-a-jwt",
        "Bearer " + jwt.encode({"sub": "x", "role": "admin", "exp": int(time.time()) - 10}, SECRET, algorithm="HS256"),
        "Bearer " + jwt.encode({"sub": "x", "role": "admin", "exp": int(time.time()) + 60}, "wrong-secret" * 4, algorithm="HS256"),
        "Bearer " + jwt.encode({"sub": "x", "role": "root", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256"),
        "Bearer " + jwt.encode({"sub": "x", "role": "admin", "exp": int(time.time()) + 60}, None, algorithm="none"),
    ],
    ids=["missing", "basic", "empty", "garbage", "expired", "bad-signature", "unknown-role", "alg-none"],
)
async def test_rejects_bad_tokens_before_downstream(gw, downstream, header):
    headers = {"Authorization": header} if header else {}
    r = await gw.post("/mcp", json=LIST, headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert downstream.state.received == []


# --- tools/list ---------------------------------------------------------------


@pytest.mark.parametrize("role", ["admin", "viewer"])
async def test_tools_list_forwarded_transparently(gw, downstream, role):
    r = await gw.post("/mcp", json=LIST, headers=auth(role))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 7
    assert {t["name"] for t in body["result"]["tools"]} == {"get_customer_record", "admin_reset_key"}
    assert r.headers["mcp-session-id"] == "sess-1"  # response headers proxied
    assert downstream.state.received[0]["message"] == LIST  # payload forwarded byte-for-byte


# --- tools/call authorization ------------------------------------------------


async def test_viewer_blocked_from_admin_tool_without_downstream_call(gw, downstream):
    r = await gw.post("/mcp", json=call("admin_reset_key", 42), headers=auth("viewer"))
    assert r.status_code == 200
    assert r.json() == {
        "jsonrpc": "2.0",
        "id": 42,
        "error": {
            "code": -32001,
            "message": "Unauthorized Tool Call",
            "data": {"tool": "admin_reset_key", "required_role": "admin"},
        },
    }
    assert downstream.state.received == []


@pytest.mark.parametrize("name", ["Admin_reset_key", "ADMIN_RESET_KEY", " admin_reset_key"])
async def test_admin_prefix_check_is_not_bypassable(gw, downstream, name):
    r = await gw.post("/mcp", json=call(name), headers=auth("viewer"))
    assert r.json()["error"]["code"] == -32001
    assert downstream.state.received == []


async def test_admin_can_call_admin_tool(gw, downstream):
    r = await gw.post("/mcp", json=call("admin_reset_key"), headers=auth("admin"))
    assert r.json()["result"]["content"][0]["text"] == "executed admin_reset_key"
    assert len(downstream.state.received) == 1


async def test_viewer_can_call_regular_tool(gw, downstream):
    r = await gw.post("/mcp", json=call("get_customer_record"), headers=auth("viewer"))
    assert r.json()["result"]["isError"] is False


async def test_identity_forwarded_but_bearer_token_is_not(gw, downstream):
    await gw.post("/mcp", json=call("get_customer_record"), headers=auth("viewer", sub="alice"))
    seen = downstream.state.received[0]["headers"]
    assert "authorization" not in seen
    assert seen["x-gateway-subject"] == "alice" and seen["x-gateway-role"] == "viewer"


@pytest.mark.parametrize("params", [{}, {"name": 123}, {"name": ""}, None])
async def test_malformed_tools_call_fails_closed(gw, downstream, params):
    msg = {"jsonrpc": "2.0", "id": 3, "method": "tools/call"}
    if params is not None:
        msg["params"] = params
    r = await gw.post("/mcp", json=msg, headers=auth("admin"))
    assert r.json()["error"]["code"] == -32602
    assert downstream.state.received == []


# --- wire format -------------------------------------------------------------


async def test_parse_error(gw):
    r = await gw.post("/mcp", content=b"{not json", headers={**auth("admin"), "content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == -32700


@pytest.mark.parametrize("msg", [{"id": 1, "method": "tools/list"}, {"jsonrpc": "2.0", "id": 1, "method": 5}, "str"])
async def test_invalid_request(gw, downstream, msg):
    r = await gw.post("/mcp", json=msg, headers=auth("admin"))
    assert r.json()["error"]["code"] == -32600
    assert downstream.state.received == []


async def test_batch_mixed_allow_and_deny(gw, downstream):
    batch = [
        LIST,
        call("admin_reset_key", 2),
        call("get_customer_record", 3),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    r = await gw.post("/mcp", json=batch, headers=auth("viewer"))
    by_id = {m["id"]: m for m in r.json()}
    assert set(by_id) == {7, 2, 3}
    assert by_id[2]["error"]["code"] == -32001
    assert "result" in by_id[3] and "result" in by_id[7]
    forwarded = [m["message"].get("params", {}).get("name") for m in downstream.state.received]
    assert "admin_reset_key" not in forwarded
    assert len(downstream.state.received) == 3  # list, regular call, notification


async def test_batch_fully_denied_never_hits_downstream(gw, downstream):
    r = await gw.post("/mcp", json=[call("admin_a", 1), call("admin_b", 2)], headers=auth("viewer"))
    assert [m["error"]["code"] for m in r.json()] == [-32001, -32001]
    assert downstream.state.received == []


async def test_downstream_unreachable_returns_jsonrpc_error():
    app = create_app(downstream_url="http://127.0.0.1:1/mcp", jwt_secret=SECRET)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/mcp", json=LIST, headers=auth("admin"))
    assert r.status_code == 502
    assert r.json()["error"]["code"] == -32603 and r.json()["id"] == 7


def test_weak_secret_refused():
    with pytest.raises(RuntimeError):
        create_app(downstream_url="http://x/mcp", jwt_secret="short")
