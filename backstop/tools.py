"""The three tools the agent can call, and a seeded shop to call them against.

Arguments are validated before the tool body runs. A model that sends
amount_gbp as the string "forty" gets a typed error back, not a crash and not a
silent coercion that refunds the wrong amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- seeded data ------------------------------------------------------------

ORDERS: dict[str, dict] = {
    "A-1001": {"customer_id": "cust-1", "item": "Running shoes", "total_gbp": 42.00,
               "status": "delivered", "delivered_days_ago": 3},
    "A-1002": {"customer_id": "cust-1", "item": "Winter coat", "total_gbp": 129.99,
               "status": "delivered", "delivered_days_ago": 40},
    "A-1003": {"customer_id": "cust-1", "item": "Wool socks", "total_gbp": 12.50,
               "status": "in_transit", "delivered_days_ago": None},
    "A-1004": {"customer_id": "cust-1", "item": "Desk lamp", "total_gbp": 75.00,
               "status": "delivered", "delivered_days_ago": 10},
    # Two more under-cap orders. Individually they pass the cap without being
    # asked about, which is the point: the interesting question is what happens
    # when somebody asks for both.
    "A-1005": {"customer_id": "cust-1", "item": "Office chair", "total_gbp": 45.00,
               "status": "delivered", "delivered_days_ago": 6},
    "A-1006": {"customer_id": "cust-1", "item": "Mug", "total_gbp": 5.00,
               "status": "delivered", "delivered_days_ago": 2},
    "A-9001": {"customer_id": "cust-2", "item": "Headphones", "total_gbp": 89.00,
               "status": "delivered", "delivered_days_ago": 5},
}

RETURN_WINDOW_DAYS = 30


class ToolError(Exception):
    """Something the model can read and recover from, not a crash."""


@dataclass
class ShopState:
    """One conversation's worth of state. The guardrails read this, so it is the
    session's view of the world and not the model's."""

    customer_id: str = "cust-1"
    orders: dict = field(default_factory=lambda: ORDERS)
    refunded_orders: set = field(default_factory=set)
    refunded_total_gbp: float = 0.0
    refund_confirmed_for: str | None = None
    pending_confirm_order: str | None = None
    off_topic_turns: int = 0
    consecutive_tool_failures: int = 0
    escalated: bool = False
    escalation_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "orders": self.orders,
            "refunded_orders": self.refunded_orders,
            "refunded_total_gbp": self.refunded_total_gbp,
            "refund_confirmed_for": self.refund_confirmed_for,
            "pending_confirm_order": self.pending_confirm_order,
            "off_topic_turns": self.off_topic_turns,
            "consecutive_tool_failures": self.consecutive_tool_failures,
        }


# --- argument validation ----------------------------------------------------

def _require_str(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{key} must be a non-empty string, got {value!r}")
    return value.strip()


def _require_money(args: dict, key: str) -> float:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{key} must be a number in pounds, got {value!r}")
    if value <= 0:
        raise ToolError(f"{key} must be greater than zero, got {value}")
    return round(float(value), 2)


# --- the tools --------------------------------------------------------------

def lookup_order(args: dict, state: ShopState) -> dict:
    order_id = _require_str(args, "order_id")
    order = state.orders.get(order_id)
    if order is None:
        raise ToolError(f"no order with id {order_id}")
    return {"order_id": order_id, **{k: v for k, v in order.items() if k != "customer_id"}}


def issue_refund(args: dict, state: ShopState) -> dict:
    order_id = _require_str(args, "order_id")
    amount = _require_money(args, "amount_gbp")
    order = state.orders.get(order_id)
    if order is None:
        raise ToolError(f"no order with id {order_id}")
    if order["status"] != "delivered":
        raise ToolError(f"order {order_id} is {order['status']} and cannot be refunded yet")
    days = order.get("delivered_days_ago")
    if days is not None and days > RETURN_WINDOW_DAYS:
        raise ToolError(
            f"order {order_id} was delivered {days} days ago, outside the "
            f"{RETURN_WINDOW_DAYS} day return window"
        )
    if amount > order["total_gbp"]:
        raise ToolError(
            f"refund of GBP {amount:.2f} is more than the order total of "
            f"GBP {order['total_gbp']:.2f}"
        )
    state.refunded_orders.add(order_id)
    state.refunded_total_gbp = round(state.refunded_total_gbp + amount, 2)
    return {"order_id": order_id, "refunded_gbp": amount, "status": "refunded"}


def escalate_to_human(args: dict, state: ShopState) -> dict:
    reason = _require_str(args, "reason")
    state.escalated = True
    state.escalation_reason = reason
    return {"status": "escalated", "reason": reason}


REGISTRY: dict[str, Any] = {
    "lookup_order": lookup_order,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

SCHEMA = [
    {"name": "lookup_order",
     "description": "Look up one order belonging to the signed in customer.",
     "parameters": {"type": "object", "required": ["order_id"],
                    "properties": {"order_id": {"type": "string"}}}},
    {"name": "issue_refund",
     "description": "Refund an order. Refunds over the limit need confirmation first.",
     "parameters": {"type": "object", "required": ["order_id", "amount_gbp"],
                    "properties": {"order_id": {"type": "string"},
                                   "amount_gbp": {"type": "number"}}}},
    {"name": "escalate_to_human",
     "description": "Hand the conversation to a human agent and stop.",
     "parameters": {"type": "object", "required": ["reason"],
                    "properties": {"reason": {"type": "string"}}}},
]
