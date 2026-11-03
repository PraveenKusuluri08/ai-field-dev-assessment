"""Token-aware sliding-window rate limiter backed by on-disk SQLite.

Model
-----
Every admitted request inserts a row ``(tenant, ts, tokens)``. A tenant's usage
is the sum of ``tokens`` over rows with ``ts`` inside the last ``window``
seconds: a true sliding window (a sliding log), not fixed buckets, so there is
no burst at a bucket boundary.

Lifecycle of one request
  1. ``reserve(tenant, estimate)`` charges an *estimate* upfront (prompt tokens
     plus ``max_tokens``). Concurrent requests see each other's reservations, so
     a burst cannot overshoot the budget while the first requests are in flight.
  2. ``reconcile(id, actual)`` replaces the estimate with the provider-reported usage.
  3. ``release(id)`` refunds the reservation if no provider produced a completion.

Concurrency
-----------
Check-and-insert runs in one ``BEGIN IMMEDIATE`` transaction, which takes
SQLite's write lock up front. That makes admission atomic across threads *and*
across processes sharing the DB file (for example, several uvicorn workers).
WAL mode keeps readers unblocked. In-process calls are also serialized with a
lock, because a ``sqlite3.Connection`` must not be used by two threads at once.

Eviction
--------
Rows that leave the window are deleted inside the same transaction (indexed on
``ts``), so the table holds at most one window of data.
"""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant  TEXT    NOT NULL,
    ts      REAL    NOT NULL,
    tokens  INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS ix_usage_tenant_ts ON token_usage (tenant, ts);
CREATE INDEX IF NOT EXISTS ix_usage_ts ON token_usage (ts);
"""


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float, limit: int, used: int, requested: int):
        super().__init__("rate limit exceeded")
        self.retry_after = retry_after
        self.limit = limit
        self.used = used
        self.requested = requested


class RequestTooLarge(Exception):
    """The request alone exceeds the per-window budget and can never be admitted."""


@dataclass(frozen=True)
class Reservation:
    id: int
    tenant: str
    tokens: int
    remaining: int  # budget left in the window after this reservation


class SlidingWindowTokenLimiter:
    def __init__(
        self,
        db_path: str,
        limit_tokens: int = 50_000,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ):
        if limit_tokens <= 0 or window_seconds <= 0:
            raise ValueError("limit and window must be positive")
        self.limit = limit_tokens
        self.window = window_seconds
        self.clock = clock
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        # isolation_level=None: we issue BEGIN/COMMIT ourselves
        self._conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, timeout=5.0)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)

    # ---- sync core (runs in a worker thread when called from async code) ----

    def reserve_sync(self, tenant: str, tokens: int) -> Reservation:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self.limit:
            raise RequestTooLarge(f"request needs {tokens} tokens; window budget is {self.limit}")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                now = self.clock()
                horizon = now - self.window
                cur.execute("DELETE FROM token_usage WHERE ts <= ?", (horizon,))  # evict expired rows
                (used,) = cur.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?",
                    (tenant, horizon),
                ).fetchone()

                if used + tokens > self.limit:
                    retry_after = self._retry_after(cur, tenant, horizon, now, used, tokens)
                    cur.execute("COMMIT")  # keep the eviction
                    raise RateLimitExceeded(retry_after, self.limit, used, tokens)

                cur.execute("INSERT INTO token_usage (tenant, ts, tokens) VALUES (?, ?, ?)", (tenant, now, tokens))
                rid = cur.lastrowid
                cur.execute("COMMIT")
                return Reservation(id=rid, tenant=tenant, tokens=tokens, remaining=self.limit - used - tokens)
            except RateLimitExceeded:
                raise
            except BaseException:
                cur.execute("ROLLBACK")
                raise

    def _retry_after(self, cur, tenant: str, horizon: float, now: float, used: int, tokens: int) -> float:
        """Seconds until enough of the oldest usage slides out of the window."""
        need_to_free = used + tokens - self.limit
        freed = 0
        for ts, row_tokens in cur.execute(
            "SELECT ts, tokens FROM token_usage WHERE tenant = ? AND ts > ? ORDER BY ts ASC", (tenant, horizon)
        ):
            freed += row_tokens
            if freed >= need_to_free:
                return max(0.0, ts + self.window - now)
        return self.window  # unreachable while tokens <= limit

    def reconcile_sync(self, reservation: Reservation, actual_tokens: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, int(actual_tokens)), reservation.id)
            )

    def release_sync(self, reservation: Reservation) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM token_usage WHERE id = ?", (reservation.id,))

    def usage_sync(self, tenant: str) -> int:
        with self._lock:
            (used,) = self._conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?",
                (tenant, self.clock() - self.window),
            ).fetchone()
            return used

    def row_count_sync(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]

    # ---- async API: keep blocking SQLite I/O off the event loop ----

    async def reserve(self, tenant: str, tokens: int) -> Reservation:
        return await asyncio.to_thread(self.reserve_sync, tenant, tokens)

    async def reconcile(self, reservation: Reservation, actual_tokens: int) -> None:
        await asyncio.to_thread(self.reconcile_sync, reservation, actual_tokens)

    async def release(self, reservation: Reservation) -> None:
        await asyncio.to_thread(self.release_sync, reservation)

    async def usage(self, tenant: str) -> int:
        return await asyncio.to_thread(self.usage_sync, tenant)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def estimate_tokens(messages: list[dict], max_tokens: int | None, default_completion: int = 512) -> int:
    """Cheap upper-ish estimate: ~4 chars/token plus per-message overhead plus completion budget.

    Swap in ``tiktoken`` (or the provider's tokenizer) for exact prompt counts; the
    limiter only needs a number, and reconcile() corrects it with real usage.
    """
    prompt = 3
    for m in messages:
        content = m.get("content")
        text = content if isinstance(content, str) else str(content or "")
        prompt += 4 + math.ceil(len(text) / 4)
    return prompt + (max_tokens if max_tokens is not None else default_completion)
