"""Task 1 tests: drive the real server process over stdio.

Two angles:
  * a raw JSON-RPC client that asserts *every byte* on stdout is a JSON-RPC message
  * the official MCP Python client, to prove interoperability
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

SERVER = Path(__file__).resolve().parents[1] / "task1_mcp_server" / "server.py"


class RawClient:
    """Line-delimited JSON-RPC over the server's stdin/stdout, one request at a time."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "LOG_LEVEL": "DEBUG"},
        )
        self.stdout_lines: list[bytes] = []
        self._id = 0

    def send(self, msg: dict) -> None:
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self.send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        line = self.proc.stdout.readline()
        self.stdout_lines.append(line)
        return json.loads(line)

    def call(self, name: str, arguments) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> tuple[bytes, bytes]:
        out, err = self.proc.communicate(timeout=10)
        self.stdout_lines.extend(out.splitlines(keepends=True))
        return out, err


@pytest.fixture
def raw():
    c = RawClient()
    init = c.request(
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
    )
    assert init["result"]["serverInfo"]["name"] == "customer-support-mcp"
    c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    yield c
    if c.proc.poll() is None:
        c.proc.kill()


def assert_invalid_params(resp: dict, field: str | None = None):
    assert "result" not in resp, resp
    assert resp["error"]["code"] == -32602, resp
    if field:
        assert any(d["field"] == field for d in resp["error"]["data"]), resp


# --- STDIO isolation --------------------------------------------------------


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr(raw):
    raw.request("tools/list")
    raw.call("get_customer_record", {"customer_id": "CUST-00001"})
    raw.call("get_customer_record", {"customer_id": "bad"})
    raw.call("trigger_refund", {"customer_id": "CUST-00001", "amount": 5, "reason": "Late delivery refund"})
    _, stderr = raw.close()

    assert raw.stdout_lines, "no protocol output captured"
    for line in raw.stdout_lines:
        msg = json.loads(line)  # raises if anything non-JSON leaked onto stdout
        assert msg["jsonrpc"] == "2.0"
    assert b"ready on stdio" in stderr and b"refund issued" in stderr


def test_stray_print_is_redirected_to_stderr():
    """Even a print() executed inside the server process cannot reach stdout."""
    code = (
        "import sys, runpy; sys.argv=['server'];"
        "mod = runpy.run_path(%r);"
        "mod['isolate_stdout'](); print('OOPS-DEBUG'); import os; os.write(1, b'RAW-FD-WRITE\\n')" % str(SERVER)
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=10)
    assert p.stdout == b""
    assert b"OOPS-DEBUG" in p.stderr and b"RAW-FD-WRITE" in p.stderr


# --- protocol + happy paths --------------------------------------------------


def test_tools_list_exposes_strict_schemas(raw):
    tools = {t["name"]: t for t in raw.request("tools/list")["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    refund = tools["trigger_refund"]["inputSchema"]
    assert refund["additionalProperties"] is False
    assert set(refund["required"]) == {"customer_id", "amount", "reason"}
    assert refund["properties"]["customer_id"]["pattern"] == r"^CUST-\d{5}$"
    assert refund["properties"]["reason"]["minLength"] == 10


def test_get_customer_record_ok(raw):
    res = raw.call("get_customer_record", {"customer_id": "CUST-00042"})["result"]
    assert res["isError"] is False
    assert res["structuredContent"]["name"] == "Daniel Kim"


def test_trigger_refund_ok(raw):
    res = raw.call(
        "trigger_refund", {"customer_id": "CUST-00042", "amount": 19.99, "reason": "Item arrived damaged"}
    )["result"]
    assert res["isError"] is False
    assert res["structuredContent"]["status"] == "processed"
    assert res["structuredContent"]["refund_id"].startswith("RF-")


def test_unknown_customer_is_tool_error_not_protocol_error(raw):
    res = raw.call("get_customer_record", {"customer_id": "CUST-99999"})["result"]
    assert res["isError"] is True and "not found" in res["content"][0]["text"]


def test_refund_over_balance_is_tool_error(raw):
    res = raw.call("trigger_refund", {"customer_id": "CUST-12345", "amount": 500.0, "reason": "Customer goodwill"})
    assert res["result"]["isError"] is True


# --- validation edge cases -> -32602 ----------------------------------------


@pytest.mark.parametrize(
    "customer_id",
    ["CUST-1234", "CUST-123456", "cust-12345", "CUST-1234A", " CUST-12345", "CUST-12345\n", "CUST12345", "", 12345, None],
)
def test_bad_customer_id(raw, customer_id):
    assert_invalid_params(raw.call("get_customer_record", {"customer_id": customer_id}), "customer_id")


@pytest.mark.parametrize("amount", [0, -1, -0.01, "10.00", True, None, 10.001, 1e12, [], {}])
def test_bad_amount(raw, amount):
    resp = raw.call("trigger_refund", {"customer_id": "CUST-00001", "amount": amount, "reason": "A valid reason here"})
    assert_invalid_params(resp, "amount")


@pytest.mark.parametrize("reason", ["too short", "", "          ", "   short   ", 1234567890123, None, "x" * 501])
def test_bad_reason(raw, reason):
    resp = raw.call("trigger_refund", {"customer_id": "CUST-00001", "amount": 1.0, "reason": reason})
    assert_invalid_params(resp, "reason")


def test_missing_and_extra_fields(raw):
    assert_invalid_params(raw.call("trigger_refund", {"customer_id": "CUST-00001"}), "amount")
    assert_invalid_params(
        raw.call("get_customer_record", {"customer_id": "CUST-00001", "is_admin": True}), "is_admin"
    )


def test_unknown_tool(raw):
    resp = raw.call("drop_database", {})
    assert resp["error"]["code"] == -32602 and "Unknown tool" in resp["error"]["message"]


def test_error_does_not_echo_input_values(raw):
    resp = raw.call("get_customer_record", {"customer_id": "SSN-123-45-6789"})
    assert "123-45-6789" not in json.dumps(resp)


def test_unknown_method_is_method_not_found(raw):
    assert raw.request("resources/frobnicate")["error"]["code"] in (-32601, -32602)


# --- interoperability with the official client ------------------------------


async def test_official_mcp_client_roundtrip():
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = {t.name for t in (await session.list_tools()).tools}
        assert names == {"get_customer_record", "trigger_refund"}

        ok = await session.call_tool("get_customer_record", {"customer_id": "CUST-00001"})
        assert not ok.isError and ok.structuredContent["tier"] == "gold"

        with pytest.raises(McpError) as exc:
            await session.call_tool("trigger_refund", {"customer_id": "CUST-00001", "amount": -5, "reason": "x"})
        assert exc.value.error.code == -32602
