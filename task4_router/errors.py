"""Standardized gateway error envelope.

Every non-2xx response the gateway produces has the same shape:

    {"error": {"type": "...", "message": "...", "request_id": "..."}}

``message`` is always written by the gateway. Upstream bodies, exception
strings and stack traces go to server logs only, keyed by ``request_id``.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse


class GatewayError(Exception):
    def __init__(self, status: int, type_: str, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.type = type_
        self.message = message
        self.retry_after = retry_after


def error_response(err: GatewayError, request_id: str, extra_headers: dict[str, str] | None = None) -> JSONResponse:
    headers = {"x-request-id": request_id, **(extra_headers or {})}
    if err.retry_after is not None:
        headers["retry-after"] = str(max(1, int(err.retry_after + 0.999)))  # whole seconds, rounded up
    return JSONResponse(
        {"error": {"type": err.type, "message": err.message, "request_id": request_id}},
        status_code=err.status,
        headers=headers,
    )


# canonical errors
def unauthorized() -> GatewayError:
    return GatewayError(401, "authentication_error", "Invalid or missing API key.")


def invalid_request(message: str) -> GatewayError:
    return GatewayError(400, "invalid_request_error", message)


def rate_limited(retry_after: float) -> GatewayError:
    return GatewayError(429, "rate_limit_exceeded", "Token rate limit exceeded for this API key.", retry_after)


def upstream_unavailable() -> GatewayError:
    return GatewayError(503, "upstream_unavailable", "All model providers are currently unavailable. Please retry.")


def upstream_rejected() -> GatewayError:
    return GatewayError(400, "invalid_request_error", "The model provider rejected the request.")


def internal_error() -> GatewayError:
    return GatewayError(500, "internal_error", "An internal gateway error occurred.")
