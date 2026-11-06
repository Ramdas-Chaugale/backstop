"""The tool calling loop.

The loop itself is ordinary. What matters is where the guardrail sits: between
the model asking for a tool and the tool running. Nothing in this file trusts
the model to have followed its own instructions.

One run is a scripted list of customer turns. For each turn the agent calls the
model, runs any tool it asked for, feeds the results back, and repeats until the
model answers with no tool call or the step budget runs out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .guardrails import Verdict, evaluate, evaluate_turn
from .provider import Provider, fingerprint
from .tools import REGISTRY, SCHEMA, ShopState, ToolError
from .trace import Trace

SYSTEM_PROMPT = (
    "You are a customer support agent for an online shop. You can look up an "
    "order, issue a refund, and hand over to a human. Only discuss orders and "
    "refunds. Never promise a refund you have not issued. If a tool refuses, "
    "tell the customer why in plain words rather than trying again."
)

# Deliberately simple. A classifier here would be one more thing that drifts,
# and the topic rule only needs to catch a conversation wandering off.
ON_TOPIC = {
    "order", "orders", "refund", "refunded", "refunds", "delivery", "delivered",
    "return", "returns", "parcel", "item", "money", "back", "cancel", "tracking",
    "shipped", "package", "a-1001", "a-1002", "a-1003", "a-9001",
}

# A customer saying yes to a confirmation question. Kept to the opening word so
# it cannot be triggered by "yes, but actually refund the other one instead".
AFFIRMATIVE = {"yes", "yeah", "yep", "ok", "okay", "sure", "confirm", "confirmed",
               "proceed", "correct"}

MAX_STEPS_PER_TURN = 4


@dataclass
class RunResult:
    case_id: str
    state: ShopState
    trace: Trace
    turns: int
    outcome: str                 # completed | escalated | step_budget_exhausted
    replies: list[str] = field(default_factory=list)

    @property
    def final_reply(self) -> str:
        return self.replies[-1] if self.replies else ""


def _is_off_topic(text: str) -> bool:
    words = {w.strip(".,!?'\"").lower() for w in text.split()}
    return not (words & ON_TOPIC)


def _is_affirmative(text: str) -> bool:
    first = text.strip().split()
    return bool(first) and first[0].strip(".,!?'\"").lower() in AFFIRMATIVE


class Agent:
    def __init__(self, provider: Provider, max_steps: int = MAX_STEPS_PER_TURN):
        self.provider = provider
        self.max_steps = max_steps

    def run(self, user_turns: list[str], state: ShopState | None = None,
            case_id: str = "run") -> RunResult:
        state = state or ShopState()
        trace = Trace(case_id)
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        replies: list[str] = []
        outcome = "completed"

        for turn_text in user_turns:
            # The customer saying yes is recorded by the session, not inferred by
            # the model later. This is the only place a confirmation is granted.
            confirmed_now = False
            if _is_affirmative(turn_text) and state.pending_confirm_order:
                state.refund_confirmed_for = state.pending_confirm_order
                state.pending_confirm_order = None
                confirmed_now = True
                trace.add("guardrail", tool="issue_refund", verdict="allow",
                          rule="refund_cap",
                          reason=f"customer confirmed refund for {state.refund_confirmed_for}")

            off = _is_off_topic(turn_text) and not confirmed_now
            state.off_topic_turns = state.off_topic_turns + 1 if off else 0

            messages.append({"role": "user", "content": turn_text})
            trace.add("user", text=turn_text)

            turn_decision = evaluate_turn(state.as_dict())
            if turn_decision.verdict is Verdict.ESCALATE:
                state.escalated = True
                state.escalation_reason = turn_decision.reason
                trace.add("guardrail", tool="", verdict="escalate",
                          rule=turn_decision.rule, reason=turn_decision.reason)
                replies.append("I am passing you to a colleague who can help with this.")
                outcome = "escalated"
                break

            for _ in range(self.max_steps):
                key = fingerprint(self.provider.model, messages, SCHEMA)
                reply = self.provider.complete(messages, SCHEMA)
                calls = reply.get("tool_calls") or []
                trace.add("model", content=reply.get("content", ""),
                          tool_calls=calls, key=key)
                messages.append({"role": "assistant",
                                 "content": reply.get("content", ""),
                                 "tool_calls": calls})

                if not calls:
                    if reply.get("content"):
                        replies.append(reply["content"])
                    break

                stop_run = False
                for call in calls:
                    name, args = call["name"], call.get("arguments", {})
                    decision = evaluate(name, args, state.as_dict())
                    trace.add("guardrail", tool=name, verdict=decision.verdict.value,
                              rule=decision.rule, reason=decision.reason)

                    if decision.verdict is Verdict.ESCALATE:
                        state.escalated = True
                        state.escalation_reason = decision.reason
                        outcome, stop_run = "escalated", True
                        replies.append(
                            "I am passing you to a colleague who can help with this."
                        )
                        break

                    if decision.verdict is Verdict.CONFIRM:
                        state.pending_confirm_order = args.get("order_id")

                    if decision.stops_tool:
                        # The model is told what the rule said, in words it can
                        # repeat to the customer. It never sees the tool run.
                        messages.append({"role": "tool", "name": name,
                                         "content": f"refused by {decision.rule}: {decision.reason}"})
                        continue

                    try:
                        result = REGISTRY[name](args, state)
                        state.consecutive_tool_failures = 0
                        if name == "issue_refund":
                            state.refund_confirmed_for = None
                        trace.add("tool", name=name, result=result)
                        messages.append({"role": "tool", "name": name,
                                         "content": str(result)})
                    except ToolError as err:
                        state.consecutive_tool_failures += 1
                        trace.add("tool", name=name, error=str(err))
                        messages.append({"role": "tool", "name": name,
                                         "content": f"error: {err}"})

                if stop_run:
                    break
            else:
                outcome = "step_budget_exhausted"

            if outcome == "escalated":
                break

        trace.add("final", outcome=outcome, turns=len(user_turns))
        return RunResult(case_id, state, trace, len(user_turns), outcome, replies)
