"""Task 1 - Customer support MCP server (stdio transport).

Tools:
  * get_customer_record(customer_id)
  * trigger_refund(customer_id, amount, reason)

Design notes
------------
* stdout is the JSON-RPC channel and nothing else. At start-up we duplicate the
  real stdout file descriptor for the protocol, then point fd 1 *and*
  ``sys.stdout`` at stderr. A stray ``print()`` - or a C extension writing to
  fd 1 - lands on stderr instead of corrupting the protocol stream.
* All logging goes to stderr.
* Arguments are validated with strict Pydantic models. A schema violation or an
  unknown tool is a *protocol* error and returns JSON-RPC ``-32602 Invalid
  params``. A business failure (for example, an unknown customer) is a *tool
  execution* error and returns ``CallToolResult(isError=True)``, as the MCP
  spec separates the two.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import sys
import uuid
from decimal import Decimal
from typing import Annotated, Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

logger = logging.getLogger("customer-mcp")

SERVER_NAME = "customer-support-mcp"
SERVER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

CustomerId = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^CUST-\d{5}$"),
    Field(description="Customer identifier in the form CUST-XXXXX (5 digits), e.g. CUST-00042"),
]


class StrictArgs(BaseModel):
    # forbid unknown keys, no type coercion ("10" is not a number), no NaN/inf
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class GetCustomerRecordArgs(StrictArgs):
    customer_id: CustomerId


class TriggerRefundArgs(StrictArgs):
    customer_id: CustomerId
    amount: float = Field(gt=0, le=1_000_000, description="Refund amount, positive, max 2 decimal places")
    reason: str = Field(min_length=10, max_length=500, description="Why the refund is issued (>= 10 chars)")

    @field_validator("amount")
    @classmethod
    def _two_decimals(cls, v: float) -> float:
        if not math.isfinite(v):  # belt and braces on top of allow_inf_nan=False
            raise ValueError("amount must be a finite number")
        if Decimal(str(v)).as_tuple().exponent < -2:
            raise ValueError("amount must have at most 2 decimal places")
        return v

    @field_validator("reason")
    @classmethod
    def _meaningful_reason(cls, v: str) -> str:
        # "          " (10 spaces) must not satisfy min_length
        if len(v.strip()) < 10:
            raise ValueError("reason must contain at least 10 non-whitespace-padded characters")
        return v.strip()


TOOL_MODELS: dict[str, type[StrictArgs]] = {
    "get_customer_record": GetCustomerRecordArgs,
    "trigger_refund": TriggerRefundArgs,
}

TOOL_DESCRIPTIONS = {
    "get_customer_record": "Fetch a customer's profile and account balance by customer ID.",
    "trigger_refund": "Issue a refund to a customer's original payment method.",
}

# ---------------------------------------------------------------------------
# Mock data store
# ---------------------------------------------------------------------------

CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Asha Rao", "tier": "gold", "lifetime_spend": 1240.50, "refunded": 0.0},
    "CUST-00042": {"name": "Daniel Kim", "tier": "silver", "lifetime_spend": 310.00, "refunded": 0.0},
    "CUST-12345": {"name": "Maria Lopez", "tier": "bronze", "lifetime_spend": 89.99, "refunded": 0.0},
}


def get_customer_record(args: GetCustomerRecordArgs) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return tool_error(f"Customer {args.customer_id} not found")
    return tool_ok({"customer_id": args.customer_id, **record})


def trigger_refund(args: TriggerRefundArgs) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return tool_error(f"Customer {args.customer_id} not found")
    refundable = round(record["lifetime_spend"] - record["refunded"], 2)
    if args.amount > refundable:
        return tool_error(f"Refund of {args.amount:.2f} exceeds refundable balance of {refundable:.2f}")
    record["refunded"] = round(record["refunded"] + args.amount, 2)
    refund_id = f"RF-{uuid.uuid4().hex[:10].upper()}"
    logger.info("refund issued id=%s customer=%s amount=%.2f", refund_id, args.customer_id, args.amount)
    return tool_ok(
        {
            "refund_id": refund_id,
            "customer_id": args.customer_id,
            "amount": args.amount,
            "reason": args.reason,
            "status": "processed",
        }
    )


HANDLERS = {"get_customer_record": get_customer_record, "trigger_refund": trigger_refund}

# ---------------------------------------------------------------------------
# MCP wiring
# ---------------------------------------------------------------------------


def tool_ok(payload: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


def tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], isError=True)


def invalid_params(message: str, data: Any = None) -> McpError:
    return McpError(types.ErrorData(code=types.INVALID_PARAMS, message=message, data=data))


def validate_arguments(tool_name: str, arguments: dict[str, Any] | None) -> StrictArgs:
    model = TOOL_MODELS.get(tool_name)
    if model is None:
        raise invalid_params(f"Unknown tool: {tool_name}")
    try:
        return model.model_validate(arguments or {})
    except ValidationError as exc:
        # include_input=False: never echo raw (possibly sensitive) values back
        details = [
            {"field": ".".join(str(p) for p in err["loc"]) or "<root>", "error": err["msg"]}
            for err in exc.errors(include_input=False, include_url=False)
        ]
        raise invalid_params(f"Invalid arguments for tool '{tool_name}'", data=details) from None


def build_server() -> Server:
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=name, description=TOOL_DESCRIPTIONS[name], inputSchema=model.model_json_schema())
            for name, model in TOOL_MODELS.items()
        ]

    # The SDK's @server.call_tool() decorator turns *every* exception into an
    # isError tool result. Validation failures must instead be JSON-RPC errors,
    # so the handler is registered directly. McpError raised here becomes a
    # JSON-RPC error response.
    async def call_tool(req: types.CallToolRequest) -> types.ServerResult:
        name = req.params.name
        args = validate_arguments(name, req.params.arguments)
        logger.debug("tools/call name=%s", name)
        try:
            result = HANDLERS[name](args)
        except Exception:  # unexpected bug: log details to stderr, keep them off the wire
            logger.exception("tool %s crashed", name)
            raise McpError(types.ErrorData(code=types.INTERNAL_ERROR, message="Internal error")) from None
        return types.ServerResult(result)

    server.request_handlers[types.CallToolRequest] = call_tool
    return server


def isolate_stdout() -> io.TextIOWrapper:
    """Reserve the real stdout for JSON-RPC and send everything else to stderr."""
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # fd 1 -> stderr (covers C-level writes)
    sys.stdout = sys.stderr  # print() -> stderr
    return io.TextIOWrapper(os.fdopen(protocol_fd, "wb", buffering=0), encoding="utf-8", write_through=True)


async def run() -> None:
    protocol_out = isolate_stdout()
    server = build_server()
    stdout = anyio.wrap_file(protocol_out)
    async with stdio_server(stdout=stdout) as (read_stream, write_stream):
        logger.info("%s %s ready on stdio", SERVER_NAME, SERVER_VERSION)
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(NotificationOptions(), {}),
            ),
        )


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    anyio.run(run)


if __name__ == "__main__":
    main()
