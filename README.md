# AI Field Developer / Solutions Engineer Assessment

Python 3.12 solutions for all four tasks. Each one runs on its own and ships with mock upstreams and tests.

| Task | What | Entry point | Tests |
|---|---|---|---|
| 1 | MCP server, strict validation, stdio | `task1_mcp_server/server.py` | `tests/test_task1_mcp_server.py` |
| 2 | MCP security gateway (JWT role auth, tool filtering) | `task2_mcp_gateway/gateway.py` | `tests/test_task2_mcp_gateway.py` |
| 3 | LLM gateway, streaming PII redaction | `task3_llm_guardrail/gateway.py` | `tests/test_task3_guardrail.py` |
| 4 | Token rate limiter (SQLite) + model failover router | `task4_router/app.py` | `tests/test_task4_router.py` |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest            # full suite
```

---

## Task 1: MCP server (`task1_mcp_server/`)

Tools: `get_customer_record(customer_id)` and `trigger_refund(customer_id, amount, reason)`. Built on the official `mcp` SDK (low-level `Server`), with Pydantic v2 models.

```bash
.venv/bin/python task1_mcp_server/server.py          # speaks JSON-RPC on stdio
```

Claude Desktop / any MCP client config:

```json
{ "mcpServers": { "customer-support": {
    "command": "/ABSOLUTE/PATH/.venv/bin/python",
    "args": ["/ABSOLUTE/PATH/task1_mcp_server/server.py"] } } }
```

**STDIO isolation.** At start-up the real stdout fd is `dup`'d and handed only to the MCP transport. Then fd 1 is `dup2`'d onto stderr and `sys.stdout = sys.stderr`. So a stray `print()`, a third-party library or a C extension writing to fd 1 all land on stderr. All logging goes to stderr. Tests check that every stdout line parses as JSON-RPC 2.0, and that a deliberate `print()` and a raw `os.write(1, …)` never reach stdout.

**Validation → JSON-RPC errors.** The SDK's `@server.call_tool()` decorator turns *every* exception into `isError: true` results. So the `tools/call` handler is registered directly, and validation failures raise `McpError(-32602 Invalid params)`, as the MCP spec says for invalid arguments or an unknown tool. The `data` lists `{field, error}` pairs and never echoes the input values, because they may be sensitive.

| Rule | Details |
|---|---|
| `customer_id` | strict str, `^CUST-\d{5}$` (case-sensitive, no whitespace, no ints) |
| `amount` | strict float (no `"10"`, no `true`), `> 0`, `≤ 1,000,000`, finite, ≤ 2 decimals |
| `reason` | strict str, 10–500 chars *after trimming* (10 spaces don't count) |
| all | `extra="forbid"`: unknown keys such as `is_admin` are rejected |

Business failures (unknown customer, refund over the refundable balance) are *tool execution* errors (`isError: true`), not protocol errors, per the spec.

*Assumption:* `CUST-XXXXX` means five digits.

---

## Task 2: MCP security gateway (`task2_mcp_gateway/`)

```bash
.venv/bin/python -m task2_mcp_gateway.mock_downstream                     # :9001
GATEWAY_JWT_SECRET=$(openssl rand -hex 32) DOWNSTREAM_MCP_URL=http://127.0.0.1:9001/mcp \
  .venv/bin/python -m task2_mcp_gateway.gateway                           # :9000
# mint a token:
.venv/bin/python -c "from task2_mcp_gateway.gateway import mint_token; print(mint_token('alice','viewer','<secret>'))"
```

Pipeline: **authenticate → parse → authorize → forward**.

- **Auth:** `Authorization: Bearer <JWT>`, HS256 pinned (rejects `alg=none` and algorithm confusion), with `sub`, `role` and `exp` required. Role must be `admin` or `viewer`. Failures get HTTP 401 + `WWW-Authenticate` and never touch the downstream server.
- **`tools/list`** is forwarded byte-for-byte.
- **`tools/call`**: if `params.name` starts with `admin_` and role ≠ admin, the gateway answers `{"code": -32001, "message": "Unauthorized Tool Call"}` itself. The downstream server is never called (tests assert it received nothing). The prefix check is case- and whitespace-insensitive, so `Admin_x` and `" admin_x"` don't slip through. A `tools/call` with a missing or non-string name fails closed (`-32602`).
- **Wire format:** `-32700` parse error, `-32600` invalid request, and full **batch** support. Denied entries are answered locally, allowed entries are forwarded as one batch, and the results are merged. Notifications get no reply.
- **Proxying:** single requests are streamed back unchanged (JSON or SSE, so streamable HTTP works). `Mcp-Session-Id` is passed through. GET (SSE) and DELETE (session end) are proxied after auth. The client's bearer token is **not** forwarded: the downstream server gets `X-Gateway-Subject` / `X-Gateway-Role` instead. Downstream outage → 502 with JSON-RPC `-32603`. 1 MB body cap.

JSON-RPC level denials return HTTP 200 (the transport worked, the call was refused). Auth failures return HTTP 401.

---

## Task 3: Streaming PII guardrail (`task3_llm_guardrail/`)

```bash
.venv/bin/python -m task3_llm_guardrail.mock_provider                     # :9101 (echoes the prompt in tiny chunks)
UPSTREAM_BASE_URL=http://127.0.0.1:9101 .venv/bin/python -m task3_llm_guardrail.gateway   # :9100
curl -N localhost:9100/v1/chat/completions -H 'content-type: application/json' \
  -d '{"stream":true,"mock_chunk_size":2,"messages":[{"role":"user","content":"mail a.b@c.io ssn 123-45-6789"}]}'
```

Point `UPSTREAM_BASE_URL` / `UPSTREAM_API_KEY` at any OpenAI-compatible provider.

**Algorithm (`redactor.py`).** Every redacted pattern is a contiguous run from a small alphabet: email chars, or digits with single space/dash separators. So only the **trailing run that could still grow into a match** has to be held back. Everything before it is redacted and emitted at once.

- Ordinary prose is held back by one partial word, a few characters, so TTFT stays essentially the upstream's.
- If a match straddles the emit boundary, the whole match is held, so half a card number is never emitted.
- Hold-back is capped at `MAX_HOLD` (the longest possible match), so memory per stream is **O(MAX_HOLD + chunk)** no matter how long the response.
- Each step scans only the held tail plus the new chunk, so work per delta is O(chunk).
- Patterns: email; SSN (consistent `-` or space separator); 13–19 digit card **with Luhn check**, to avoid redacting order numbers.

**Correctness invariant (property-tested):** for *any* chunking, `concat(stream outputs) == redact_text(full_text)`. Tests cover every 2-way split of a PII-dense sample, fixed chunk sizes 1–64, and 5,000 random fuzz cases. The fuzzer caught one real boundary bug during development (an SSN glued to `-` lost its left context), now fixed.

**Proxy (`gateway.py`).** It parses SSE line by line, keeps one redactor per `choices[i].index`, rewrites `delta.content` and re-emits each event immediately. It flushes held text on `finish_reason`, `[DONE]` or an upstream EOF. Async generator end to end, so back-pressure propagates. A client disconnect closes the upstream connection. Upstream HTTP errors are mapped to generic errors and never echoed. Non-streaming responses are redacted too. A real-socket test checks that the first token reaches the client before the upstream finishes.

---

## Task 4: Rate limiter + fallback router (`task4_router/`)

```bash
.venv/bin/python -m task4_router.mock_providers primary 9201 &
.venv/bin/python -m task4_router.mock_providers secondary 9202 &
TENANT_KEYS="sk-demo-key:demo-tenant" .venv/bin/python -m task4_router.app      # :9200, DB at ./data/gateway.sqlite3
curl localhost:9200/v1/chat/completions -H 'Authorization: Bearer sk-demo-key' \
  -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hi"}]}'
curl -X POST localhost:9201/_mode/slow     # or 429 / 500 / ok: watch x-gateway-provider flip to secondary
```

**Rate limiter (`rate_limiter.py`)**: a sliding log in **on-disk SQLite**, 50,000 tokens/min per tenant by default.
- `reserve(estimate)` → `reconcile(actual usage)` on success, or `release()` if every provider failed. Concurrent in-flight requests see each other's reservations, so a burst can't overshoot.
- Admission is one `BEGIN IMMEDIATE` transaction (evict expired → sum window → insert), so it is **atomic across threads and processes**. A test races 64 threads over 4 separate connections and admits exactly 25 × 2,000 tokens. WAL mode, `busy_timeout`, indexes on `(tenant, ts)` and `ts`.
- Eviction runs in the same transaction, so the table holds at most one window of rows.
- `Retry-After` is exact: the time until enough of the oldest usage slides out.
- Tenant API keys are kept only as SHA-256 hashes and compared in constant time.

**Router (`router.py`)**: primary → secondary.
- Fails over on **429**, **timeout (3000 ms default)**, 5xx and connection errors. Other 4xx stop, because the request itself is bad and a fallback would just bill twice.
- Each attempt runs under `asyncio.timeout`, which *cancels* the in-flight request. A late primary response can never race the fallback; a test asserts the cancelled request really is torn down. The deadline covers the full response body, not just connect.
- The `model` field is rewritten per provider. If every provider rate-limits, the gateway returns 429 with the smallest upstream `Retry-After`. Otherwise it returns 503.

**Error sanitization (`errors.py`)**: every non-2xx response is `{"error": {"type", "message", "request_id"}}` with a gateway-authored message. Handlers cover `GatewayError`, request validation, 404/405 and *any* unhandled exception. Upstream bodies, stack traces, URLs and provider keys go to logs only, keyed by `request_id`. Tests assert none of them leak.

*Notes / trade-offs:* the token estimate is chars/4 plus `max_tokens` (swap in `tiktoken` for exact prompt counts; `reconcile` corrects it either way). The window uses wall-clock time so that several processes share one clock. For multi-node deployments the same design maps onto Redis sorted sets or a Lua script.
