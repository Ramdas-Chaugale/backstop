"""A deterministic stand-in for the model, used to record the cassettes that
ship with this repository.

Why it exists: a fresh clone and a CI runner have no API key, and the point of
this project is the guardrail and evaluation layer rather than the model. So the
committed cassettes were recorded from this policy, which means `python -m
evals.run_evals` works the moment you clone.

It is not pretending to be a language model. It is a small rule based policy
that produces the same shape of answer a tool calling model produces, so the
rest of the system can be exercised end to end. To record against a real model
instead, pass a live client to RecordProvider and re-record with
BACKSTOP_MODE=record. Nothing else changes.
"""

from __future__ import annotations

import re

ORDER_RE = re.compile(r"\bA-\d{4}\b", re.IGNORECASE)
# The customer does not have to say the word "refund". Caught by the evaluation
# suite: "I want my money back" looked up the order and then stopped.
REFUND_WORDS = ("refund", "money back", "return it", "send it back", "my money")

YES = {"yes", "yeah", "yep", "ok", "okay", "confirm", "confirmed", "go", "ahead",
       "please", "do", "it", "sure", "correct"}


def _order_in(text: str) -> str | None:
    m = ORDER_RE.search(text or "")
    return m.group(0).upper() if m else None


def _text(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": []}


def _call(name: str, **arguments) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"name": name, "arguments": arguments}]}


def respond(messages: list[dict], tools: list[dict] | None = None) -> dict:
    last = messages[-1]

    # --- reacting to a tool result -----------------------------------------
    if last.get("role") == "tool":
        body = last.get("content", "")

        if body.startswith("refused by order_ownership"):
            return _text("That order is not on your account, so I cannot refund it. "
                         "If you think that is wrong I can pass you to a colleague.")
        if body.startswith("refused by unknown_order"):
            return _text("I cannot find an order with that number. Could you check it "
                         "and read it back to me?")
        if body.startswith("refused by refund_once"):
            return _text("That order has already been refunded, so there is nothing "
                         "further to pay back.")
        if body.startswith("refused by refund_cap"):
            amount = re.search(r"GBP ([\d.]+)", body)
            shown = amount.group(1) if amount else "that amount"
            return _text(f"That refund comes to GBP {shown}, which is over the amount "
                         f"I can return without checking. Shall I go ahead?")
        if body.startswith("error:"):
            return _text(f"I could not do that. {body[len('error:'):].strip()}")
        if "'status': 'refunded'" in body:
            amount = re.search(r"'refunded_gbp': ([\d.]+)", body)
            shown = amount.group(1) if amount else ""
            return _text(f"Done, I have refunded GBP {shown}. It should be back with "
                         f"you in three to five working days.")
        if "'status': 'escalated'" in body:
            return _text("I am passing you to a colleague now.")

        # a successful lookup: refund the full order total
        order = _order_in(body)
        total = re.search(r"'total_gbp': ([\d.]+)", body)
        asked = " ".join(m.get("content", "") for m in messages
                         if m.get("role") == "user").lower()
        wanted_refund = any(p in asked for p in REFUND_WORDS)
        if order and total and wanted_refund:
            return _call("issue_refund", order_id=order, amount_gbp=float(total.group(1)))
        if order and total:
            status = re.search(r"'status': '(\w+)'", body)
            return _text(f"Order {order} is {status.group(1) if status else 'on file'} "
                         f"and came to GBP {total.group(1)}.")
        return _text("I have looked that up for you.")

    # --- reacting to the customer ------------------------------------------
    text = (last.get("content") or "").lower()
    order = _order_in(text)
    words = {w.strip(".,!?'\"") for w in text.split()}

    # a plain yes, carrying on from a confirmation question
    if words & YES and not order and len(words) <= 6:
        for m in reversed(messages):
            if m.get("role") == "tool" and m.get("content", "").startswith("refused by refund_cap"):
                prior = _order_in(m["content"])
                amount = re.search(r"GBP ([\d.]+)", m["content"])
                if prior and amount:
                    return _call("issue_refund", order_id=prior,
                                 amount_gbp=float(amount.group(1)))
        return _text("Thanks. Is there anything else about an order I can help with?")

    if order and any(p in text for p in REFUND_WORDS):
        return _call("lookup_order", order_id=order)
    if order:
        return _call("lookup_order", order_id=order)
    if any(p in text for p in REFUND_WORDS):
        return _text("Of course. What is the order number?")
    if any(w in text for w in ("human", "agent", "person", "manager", "complain")):
        return _call("escalate_to_human", reason="customer asked for a person")

    return _text("I can help with orders and refunds. What is the order number?")
