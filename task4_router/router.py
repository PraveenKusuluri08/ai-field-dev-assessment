"""Model router with timeout and 429 failover.

Policy for each attempt, in provider order (primary, then secondary):

    outcome                         action
    ------------------------------  -----------------------------------------
    2xx                             return it
    429 Too Many Requests           fail over
    no response within timeout_s    cancel the attempt and fail over
    5xx / connection error          fail over (same "provider is unhealthy" signal)
    other 4xx                       stop: the request itself is bad, and the
                                    next provider would reject it too (and bill for it)

Timeout race: each attempt runs under ``asyncio.timeout``, which *cancels* the
in-flight httpx request when the deadline fires. So a slow primary answer
arriving after we moved on cannot be returned, double-counted or leak a
connection: an attempt yields either its response or ``TimeoutError``, never
both. The deadline covers the whole response (headers + body), not just connect.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx

from task4_router import errors

logger = logging.getLogger("model-router")


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)  # never printed in logs/reprs


@dataclass
class RouteResult:
    provider: str
    body: dict
    attempts: list[str]


class _Retryable(Exception):
    def __init__(self, reason: str, retry_after: float | None = None):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


class ModelRouter:
    def __init__(self, providers: list[Provider], client: httpx.AsyncClient, timeout_s: float = 3.0):
        if not providers:
            raise ValueError("at least one provider required")
        self.providers = providers
        self.client = client
        self.timeout_s = timeout_s

    async def complete(self, body: dict, request_id: str) -> RouteResult:
        attempts: list[str] = []
        all_rate_limited = True
        retry_hint: float | None = None

        for provider in self.providers:
            try:
                result = await self._attempt(provider, body, request_id)
                attempts.append(f"{provider.name}:ok")
                return RouteResult(provider=provider.name, body=result, attempts=attempts)
            except _Retryable as exc:
                attempts.append(f"{provider.name}:{exc.reason}")
                logger.warning("req=%s provider=%s failed (%s), failing over", request_id, provider.name, exc.reason)
                if exc.reason != "rate_limited":
                    all_rate_limited = False
                elif exc.retry_after is not None:
                    retry_hint = exc.retry_after if retry_hint is None else min(retry_hint, exc.retry_after)

        logger.error("req=%s all providers failed: %s", request_id, attempts)
        if all_rate_limited:
            err = errors.GatewayError(
                429, "upstream_rate_limited", "All model providers are rate limiting. Please retry later.", retry_hint or 1.0
            )
            raise err
        raise errors.upstream_unavailable()

    async def _attempt(self, provider: Provider, body: dict, request_id: str) -> dict:
        payload = {**body, "model": provider.model, "stream": False}
        headers = {"content-type": "application/json", "x-request-id": request_id}
        if provider.api_key:
            headers["authorization"] = f"Bearer {provider.api_key}"
        url = f"{provider.base_url.rstrip('/')}/v1/chat/completions"

        try:
            async with asyncio.timeout(self.timeout_s):
                resp = await self.client.post(url, json=payload, headers=headers)
                raw = resp.content  # body read is inside the deadline too
        except TimeoutError:
            raise _Retryable("timeout") from None
        except httpx.TimeoutException:
            raise _Retryable("timeout") from None
        except httpx.HTTPError as exc:
            logger.warning("req=%s provider=%s transport error: %r", request_id, provider.name, exc)
            raise _Retryable("connection_error") from None

        if resp.status_code == 429:
            raise _Retryable("rate_limited", _parse_retry_after(resp.headers.get("retry-after")))
        if resp.status_code >= 500:
            logger.warning("req=%s provider=%s %s body=%r", request_id, provider.name, resp.status_code, raw[:300])
            raise _Retryable(f"http_{resp.status_code}")
        if resp.status_code >= 400:
            logger.warning("req=%s provider=%s rejected %s body=%r", request_id, provider.name, resp.status_code, raw[:300])
            raise errors.upstream_rejected()
        try:
            data = resp.json()
        except ValueError:
            logger.warning("req=%s provider=%s returned non-JSON", request_id, provider.name)
            raise _Retryable("bad_response") from None
        if not isinstance(data, dict) or "choices" not in data:
            raise _Retryable("bad_response")
        return data


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None
