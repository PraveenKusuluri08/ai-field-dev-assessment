"""Task 4 tests: SQLite sliding-window limiter, failover router, and the gateway app."""

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from task4_router.app import create_app, parse_tenant_keys
from task4_router.mock_providers import create_app as create_mock
from task4_router.rate_limiter import RateLimitExceeded, RequestTooLarge, SlidingWindowTokenLimiter, estimate_tokens
from task4_router.router import ModelRouter, Provider
from task4_router.errors import GatewayError


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "limits.sqlite3")


# --- rate limiter ----------------------------------------------------------------


def test_admits_until_limit_then_rejects(db):
    lim = SlidingWindowTokenLimiter(db, limit_tokens=50_000, clock=FakeClock())
    for _ in range(5):
        lim.reserve_sync("t1", 10_000)
    with pytest.raises(RateLimitExceeded) as exc:
        lim.reserve_sync("t1", 1)
    assert exc.value.used == 50_000 and exc.value.limit == 50_000


def test_window_slides_and_retry_after_is_exact(db):
    clock = FakeClock()
    lim = SlidingWindowTokenLimiter(db, limit_tokens=50_000, window_seconds=60, clock=clock)
    lim.reserve_sync("t1", 30_000)  # t=0
    clock.t += 20
    lim.reserve_sync("t1", 20_000)  # t=20
    clock.t += 10
    with pytest.raises(RateLimitExceeded) as exc:
        lim.reserve_sync("t1", 5_000)  # t=30: must wait for the t=0 entry to expire at t=60
    assert exc.value.retry_after == pytest.approx(30.0)

    clock.t += 29.9  # t=59.9: still inside the window
    with pytest.raises(RateLimitExceeded):
        lim.reserve_sync("t1", 5_000)
    clock.t += 0.2  # t=60.1: first entry slid out
    lim.reserve_sync("t1", 5_000)
    assert lim.usage_sync("t1") == 25_000


def test_expired_rows_are_evicted_from_disk(db):
    clock = FakeClock()
    lim = SlidingWindowTokenLimiter(db, limit_tokens=50_000, clock=clock)
    for _ in range(100):
        lim.reserve_sync("t1", 10)
    assert lim.row_count_sync() == 100
    clock.t += 61
    lim.reserve_sync("t2", 10)
    assert lim.row_count_sync() == 1


def test_tenants_are_isolated(db):
    lim = SlidingWindowTokenLimiter(db, limit_tokens=1_000, clock=FakeClock())
    lim.reserve_sync("a", 1_000)
    lim.reserve_sync("b", 1_000)  # unaffected by tenant a
    with pytest.raises(RateLimitExceeded):
        lim.reserve_sync("a", 1)


def test_reconcile_and_release(db):
    lim = SlidingWindowTokenLimiter(db, limit_tokens=10_000, clock=FakeClock())
    r = lim.reserve_sync("t", 4_000)
    lim.reconcile_sync(r, 1_500)
    assert lim.usage_sync("t") == 1_500
    lim.release_sync(r)
    assert lim.usage_sync("t") == 0


def test_request_larger_than_budget(db):
    lim = SlidingWindowTokenLimiter(db, limit_tokens=100)
    with pytest.raises(RequestTooLarge):
        lim.reserve_sync("t", 101)


def test_state_persists_on_disk_across_instances(db):
    clock = FakeClock()
    SlidingWindowTokenLimiter(db, limit_tokens=1_000, clock=clock).reserve_sync("t", 900)
    fresh = SlidingWindowTokenLimiter(db, limit_tokens=1_000, clock=clock)  # e.g. after a restart
    with pytest.raises(RateLimitExceeded):
        fresh.reserve_sync("t", 200)
    assert sqlite3.connect(db).execute("SELECT SUM(tokens) FROM token_usage").fetchone()[0] == 900


def test_concurrent_reservations_never_overshoot(db):
    """64 threads across 4 independent connections (like 4 worker processes) racing on one tenant."""
    limiters = [SlidingWindowTokenLimiter(db, limit_tokens=50_000) for _ in range(4)]
    start = threading.Barrier(64)

    def worker(i):
        start.wait()
        try:
            limiters[i % 4].reserve_sync("hot-tenant", 2_000)
            return True
        except RateLimitExceeded:
            return False

    with ThreadPoolExecutor(64) as pool:
        results = list(pool.map(worker, range(64)))
    assert sum(results) == 25  # exactly 50_000 / 2_000 admitted
    assert limiters[0].usage_sync("hot-tenant") == 50_000


async def test_async_api_does_not_block_loop(db):
    lim = SlidingWindowTokenLimiter(db, limit_tokens=50_000)
    rs = await asyncio.gather(*(lim.reserve("t", 1_000) for _ in range(50)))
    assert len({r.id for r in rs}) == 50
    with pytest.raises(RateLimitExceeded):
        await lim.reserve("t", 1)


def test_estimate_tokens():
    assert estimate_tokens([{"role": "user", "content": "x" * 400}], max_tokens=100) == 3 + 4 + 100 + 100
    assert estimate_tokens([{"role": "user", "content": "hi"}], None) > 512


# --- router ------------------------------------------------------------------------

PRIMARY = Provider("primary", "http://primary", "gpt-primary", api_key="sk-primary-secret")
SECONDARY = Provider("secondary", "http://secondary", "gpt-backup")
BODY = {"messages": [{"role": "user", "content": "hi"}]}


def mock_transport(behaviours: dict[str, str], calls: list | None = None, slow_s: float = 5.0):
    async def handler(request: httpx.Request):
        host = request.url.host
        if calls is not None:
            calls.append((host, json.loads(request.content)["model"]))
        mode = behaviours[host]
        if mode == "429":
            return httpx.Response(429, json={"error": "quota"}, headers={"retry-after": "7"})
        if mode == "500":
            return httpx.Response(500, text="Traceback (most recent call last): db password=hunter2")
        if mode == "400":
            return httpx.Response(400, json={"error": "context length exceeded"})
        if mode == "down":
            raise httpx.ConnectError("connection refused", request=request)
        if mode == "slow":
            await asyncio.sleep(slow_s)
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"from {host}"}}],
                "usage": {"total_tokens": 42},
            },
        )

    return httpx.MockTransport(handler)


async def route(behaviours, timeout_s=0.3, calls=None):
    async with httpx.AsyncClient(transport=mock_transport(behaviours, calls)) as client:
        return await ModelRouter([PRIMARY, SECONDARY], client, timeout_s=timeout_s).complete(BODY, "req_test")


async def test_primary_success():
    calls = []
    r = await route({"primary": "ok", "secondary": "ok"}, calls=calls)
    assert r.provider == "primary" and calls == [("primary", "gpt-primary")]


@pytest.mark.parametrize("mode", ["429", "slow", "500", "down"])
async def test_failover_to_secondary(mode):
    calls = []
    t0 = time.perf_counter()
    r = await route({"primary": mode, "secondary": "ok"}, calls=calls)
    elapsed = time.perf_counter() - t0
    assert r.provider == "secondary"
    assert r.body["choices"][0]["message"]["content"] == "from secondary"
    assert calls[-1] == ("secondary", "gpt-backup")  # model rewritten per provider
    assert elapsed < 1.0  # slow primary is abandoned at the 0.3s deadline, not after 5s


async def test_default_timeout_is_3000ms():
    async with httpx.AsyncClient() as client:
        assert ModelRouter([PRIMARY], client).timeout_s == 3.0


async def test_timeout_cancels_primary_cleanly():
    """The abandoned primary request is cancelled, not left running in the background."""
    started, finished = asyncio.Event(), asyncio.Event()

    async def handler(request):
        if request.url.host == "primary":
            started.set()
            try:
                await asyncio.sleep(5)
            finally:
                finished.set()  # runs on cancellation
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        r = await ModelRouter([PRIMARY, SECONDARY], client, timeout_s=0.1).complete(BODY, "req")
    assert r.provider == "secondary" and started.is_set() and finished.is_set()


async def test_client_error_does_not_fail_over():
    calls = []
    with pytest.raises(GatewayError) as exc:
        await route({"primary": "400", "secondary": "ok"}, calls=calls)
    assert exc.value.status == 400 and len(calls) == 1


async def test_all_providers_down_is_503():
    with pytest.raises(GatewayError) as exc:
        await route({"primary": "slow", "secondary": "500"})
    assert exc.value.status == 503 and exc.value.type == "upstream_unavailable"


async def test_all_providers_rate_limited_is_429_with_hint():
    with pytest.raises(GatewayError) as exc:
        await route({"primary": "429", "secondary": "429"})
    assert exc.value.status == 429 and exc.value.retry_after == 7


# --- gateway app (end to end) ----------------------------------------------------

KEY_A, KEY_B = "sk-tenant-a-key", "sk-tenant-b-key"


@pytest.fixture
async def make_gw(db):
    clients = []

    async def _make(behaviours, limit=50_000, timeout_s=0.3):
        app = create_app(
            providers=[PRIMARY, SECONDARY],
            tenant_keys=parse_tenant_keys(f"{KEY_A}:tenant-a,{KEY_B}:tenant-b"),
            limiter=SlidingWindowTokenLimiter(db, limit_tokens=limit),
            transport=mock_transport(behaviours),
            timeout_s=timeout_s,
        )
        c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw")
        clients.append(c)
        return app, c

    yield _make
    for c in clients:
        await c.aclose()


def assert_error_envelope(r, status, type_):
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error"} and set(body["error"]) == {"type", "message", "request_id"}
    assert body["error"]["type"] == type_
    assert body["error"]["request_id"] == r.headers["x-request-id"]
    for leak in ["Traceback", "hunter2", "sk-primary-secret", "http://primary", "context length"]:
        assert leak not in r.text


def chat(max_tokens=100):
    return {"model": "any", "messages": [{"role": "user", "content": "hello"}], "max_tokens": max_tokens}


async def test_gateway_happy_path_and_reconcile(make_gw):
    app, c = await make_gw({"primary": "ok", "secondary": "ok"})
    r = await c.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {KEY_A}"})
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "primary"
    assert await app.state.limiter.usage("tenant-a") == 42  # estimate replaced by real usage
    assert r.headers["x-ratelimit-remaining-tokens"] == str(50_000 - 42)


async def test_gateway_failover(make_gw):
    _, c = await make_gw({"primary": "429", "secondary": "ok"})
    r = await c.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {KEY_A}"})
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "secondary"


@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic abc"])
async def test_gateway_auth(make_gw, header):
    _, c = await make_gw({"primary": "ok", "secondary": "ok"})
    r = await c.post("/v1/chat/completions", json=chat(), headers={"Authorization": header} if header else {})
    assert_error_envelope(r, 401, "authentication_error")


async def test_gateway_rate_limit_429(make_gw):
    _, c = await make_gw({"primary": "ok", "secondary": "ok"}, limit=1_000)
    h = {"Authorization": f"Bearer {KEY_A}"}
    big = chat(max_tokens=900)
    # estimates are reconciled to 42 real tokens, so the budget is consumed by real usage
    for _ in range(23):
        assert (await c.post("/v1/chat/completions", json=chat(max_tokens=10), headers=h)).status_code == 200
    r = await c.post("/v1/chat/completions", json=big, headers=h)
    assert_error_envelope(r, 429, "rate_limit_exceeded")
    assert int(r.headers["retry-after"]) >= 1
    # another tenant is unaffected
    r2 = await c.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {KEY_B}"})
    assert r2.status_code == 200


async def test_gateway_failed_upstream_releases_reservation(make_gw):
    app, c = await make_gw({"primary": "500", "secondary": "down"})
    r = await c.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {KEY_A}"})
    assert_error_envelope(r, 503, "upstream_unavailable")
    assert await app.state.limiter.usage("tenant-a") == 0


async def test_gateway_upstream_400_sanitized(make_gw):
    _, c = await make_gw({"primary": "400", "secondary": "ok"})
    r = await c.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {KEY_A}"})
    assert_error_envelope(r, 400, "invalid_request_error")


@pytest.mark.parametrize(
    "body", [{"messages": []}, {"messages": "hi"}, {"messages": [{"role": "hacker", "content": "x"}]}, {"stream": True, **chat()}]
)
async def test_gateway_bad_body(make_gw, body):
    _, c = await make_gw({"primary": "ok", "secondary": "ok"})
    r = await c.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {KEY_A}"})
    assert_error_envelope(r, 400, "invalid_request_error")


async def test_gateway_unhandled_exception_is_sanitized(make_gw, monkeypatch):
    app, c = await make_gw({"primary": "ok", "secondary": "ok"})

    async def explode(*a, **k):
        raise RuntimeError("sqlite at /secret/path/db exploded")

    monkeypatch.setattr(app.state.limiter, "reserve", explode)
    r = await c.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {KEY_A}"})
    assert_error_envelope(r, 500, "internal_error")
    assert "/secret/path" not in r.text


async def test_gateway_unknown_route_uses_envelope(make_gw):
    _, c = await make_gw({"primary": "ok", "secondary": "ok"})
    assert_error_envelope(await c.get("/admin"), 404, "not_found")


def test_tenant_keys_are_stored_hashed():
    keys = parse_tenant_keys("sk-abc:t1")
    assert "sk-abc" not in keys and keys[hashlib.sha256(b"sk-abc").hexdigest()] == "t1"


async def test_mock_provider_modes():
    app = create_mock("primary", mode="429")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p") as c:
        assert (await c.post("/v1/chat/completions", json=BODY)).status_code == 429
        await c.post("/_mode/ok")
        assert (await c.post("/v1/chat/completions", json=BODY)).json()["usage"]["total_tokens"] > 0
