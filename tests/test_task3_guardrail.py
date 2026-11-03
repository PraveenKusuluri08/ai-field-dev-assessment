"""Task 3 tests: redactor correctness across arbitrary chunk splits, plus the gateway end to end."""

import json
import random
import time

import httpx
import pytest

from task3_llm_guardrail.gateway import create_app
from task3_llm_guardrail.mock_provider import create_app as create_provider
from task3_llm_guardrail.redactor import MAX_HOLD, StreamRedactor, redact_text

SAMPLE = (
    "Sure! Contact john.doe+billing@example.co.uk or jane_d@mail.example.com today. "
    "Her SSN is 123-45-6789 (or 987 65 4321). Card: 4111 1111 1111 1111, amex 3782-822463-10005, "
    "raw 5555555555554444. Not PII: order 1234 5678 9012 3456, phone 555-123-4567, v1.2.3, a@b, 12-34-5678."
)


# --- redactor unit tests ------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail me at a.b@c.io", "mail me at [REDACTED]"),
        ("ssn 078-05-1120.", "ssn [REDACTED]."),
        ("visa 4111-1111-1111-1111!", "visa [REDACTED]!"),
        ("luhn-invalid 4111 1111 1111 1112", "luhn-invalid 4111 1111 1111 1112"),
        ("mixed sep 123-45 6789", "mixed sep 123-45 6789"),
        ("phone 555-123-4567", "phone 555-123-4567"),
        ("no pii here", "no pii here"),
    ],
)
def test_redact_text(text, expected):
    assert redact_text(text) == expected


def test_sample_redacts_all_pii_and_nothing_else():
    out = redact_text(SAMPLE)
    for pii in ["john.doe", "jane_d", "123-45-6789", "987 65 4321", "4111 1111", "3782-822463", "5555555555554444"]:
        assert pii not in out
    for ok in ["1234 5678 9012 3456", "555-123-4567", "v1.2.3", "a@b", "12-34-5678"]:
        assert ok in out


def stream(text: str, sizes) -> str:
    r = StreamRedactor()
    out, i = [], 0
    for n in sizes:
        out.append(r.feed(text[i : i + n]))
        i += n
    out.append(r.feed(text[i:]))
    return "".join(out) + r.flush()


@pytest.mark.parametrize("split", range(1, len(SAMPLE)))
def test_every_two_way_split_matches_whole_text(split):
    assert stream(SAMPLE, [split]) == redact_text(SAMPLE)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 16, 64])
def test_fixed_chunk_sizes(size):
    assert stream(SAMPLE, [size] * (len(SAMPLE) // size + 1)) == redact_text(SAMPLE)


def test_random_chunking_fuzz():
    rng = random.Random(1234)
    alphabet = "ab @.-_ 0123456789\n"
    pieces = ["x.y@z.com", "123-45-6789", "4111111111111111", "4111 1111 1111 1111", "5555-5555-5555-4444", "word"]
    for _ in range(5000):
        text = "".join(rng.choice(pieces) if rng.random() < 0.3 else rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
        sizes = [rng.randint(1, 8) for _ in range(len(text))]
        assert stream(text, sizes) == redact_text(text), text


def test_no_partial_pii_ever_emitted():
    r = StreamRedactor()
    emitted = "".join(r.feed(c) for c in "card 4111 1111 1111 1111 end")
    assert "4111" not in emitted + r.flush()


def test_low_latency_prose_streams_immediately():
    r = StreamRedactor()
    assert r.feed("Hello there, ") == "Hello there, "
    assert r.feed("how are") == "how "  # only the partial word "are" is held
    assert r.pending == 3


def test_memory_is_bounded_for_long_unbroken_tokens():
    r = StreamRedactor()
    peak = 0
    for _ in range(10_000):
        r.feed("A" * 7)
        peak = max(peak, r.pending)
    assert peak <= MAX_HOLD + 7


def test_redactor_throughput():
    text = (SAMPLE + " ") * 200  # ~70 KB
    start = time.perf_counter()
    stream(text, [4] * (len(text) // 4 + 1))
    assert time.perf_counter() - start < 2.0


# --- gateway end to end ---------------------------------------------------------


@pytest.fixture
async def gw():
    app = create_app(
        upstream_base_url="http://provider",
        upstream_api_key="sk-test",
        transport=httpx.ASGITransport(app=create_provider()),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c


def req(text, **extra):
    return {"model": "mock-1", "stream": True, "messages": [{"role": "user", "content": text}], **extra}


async def collect(resp) -> tuple[str, list[dict], bool]:
    text, events, done = "", [], False
    async for line in resp.aiter_lines():
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            done = True
            continue
        ev = json.loads(line[6:])
        events.append(ev)
        for ch in ev.get("choices", []):
            text += ch.get("delta", {}).get("content") or ""
    return text, events, done


@pytest.mark.parametrize("chunk", [1, 3, 11])
async def test_streaming_redaction_end_to_end(gw, chunk):
    async with gw.stream("POST", "/v1/chat/completions", json=req(SAMPLE, mock_chunk_size=chunk)) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text, events, done = await collect(r)
    assert text == redact_text(SAMPLE)
    assert done
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


async def test_first_token_arrives_before_stream_finishes():
    """TTFT: the first redacted text reaches the client long before upstream completes."""
    # ASGITransport buffers whole responses, so this test runs both apps on real sockets
    import socket
    import threading

    import uvicorn

    def free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    servers = []
    p_port, g_port = free_port(), free_port()
    for a, port in [(create_provider(), p_port), (create_app(upstream_base_url=f"http://127.0.0.1:{p_port}"), g_port)]:
        srv = uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=srv.run, daemon=True).start()
        servers.append(srv)
    try:
        for _ in range(100):
            if all(s.started for s in servers):
                break
            time.sleep(0.05)
        text = "word " * 40  # 40 chunks x 25ms = ~1s total
        async with httpx.AsyncClient(timeout=10) as c:
            t0 = time.perf_counter()
            first = None
            async with c.stream(
                "POST", f"http://127.0.0.1:{g_port}/v1/chat/completions", json=req(text, mock_chunk_size=5, mock_delay_ms=25)
            ) as r:
                got = ""
                async for line in r.aiter_lines():
                    if line.startswith("data: {"):
                        content = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        if content and first is None:
                            first = time.perf_counter() - t0
                        got += content or ""
            total = time.perf_counter() - t0
        assert got == text
        assert first is not None and first < 0.3 < total, (first, total)
    finally:
        for s in servers:
            s.should_exit = True


async def test_non_streaming_is_redacted(gw):
    r = await gw.post("/v1/chat/completions", json={**req("my ssn 123-45-6789"), "stream": False})
    assert r.json()["choices"][0]["message"]["content"] == "my ssn [REDACTED]"


async def test_upstream_error_is_sanitized():
    def boom(request):
        return httpx.Response(500, json={"error": "Traceback (most recent call last): secret internals"})

    app = create_app(upstream_base_url="http://provider", transport=httpx.MockTransport(boom))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json=req("hi"))
    assert r.status_code == 502
    assert "Traceback" not in r.text and r.json()["error"]["type"] == "upstream_error"


async def test_bad_request_rejected(gw):
    r = await gw.post("/v1/chat/completions", json={"model": "x"})
    assert r.status_code == 400
