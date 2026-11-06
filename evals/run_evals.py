"""Run every case, score it, and fail the build if the numbers have slipped.

Run it with no arguments and it replays from the committed cassettes: free,
offline, and the same answer every time. That is the whole reason this can live
in continuous integration.

    python -m evals.run_evals              # replay, what CI does
    python -m evals.run_evals --verbose    # print the trace for every case
    BACKSTOP_MODE=record python -m evals.run_evals   # re-record the cassettes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from backstop.agent import Agent
from backstop.provider import CassetteMiss, RecordProvider, ReplayProvider
from backstop.reference_model import respond
from backstop.tools import REGISTRY, ShopState, ToolError

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.yaml"
BASELINE = HERE / "baseline.json"
TRACE_DIR = HERE / "traces"
MODEL = "reference-1"


# --- checking one case -------------------------------------------------------

def check_direct_tool_call(case: dict) -> tuple[bool, list[str]]:
    """Cases that go straight at a tool, for argument handling the model layer
    would otherwise hide."""
    spec, expect = case["direct_tool_call"], case["expect"]
    state = ShopState()
    want = expect.get("tool_error_contains")
    try:
        REGISTRY[spec["name"]](spec.get("arguments", {}), state)
    except ToolError as err:
        if want and want not in str(err):
            return False, [f"error was {str(err)!r}, expected it to contain {want!r}"]
        return True, []
    return False, [f"expected a ToolError containing {want!r}, the call succeeded"]


def check_run(case: dict, result) -> tuple[bool, list[str]]:
    expect, fails = case["expect"], []
    fired = set(result.trace.rules_fired())
    ran = set(result.trace.tools_run())
    refunded = result.state.refunded_orders

    for order in expect.get("refunded", []):
        if order not in refunded:
            fails.append(f"{order} should have been refunded and was not")
    for order in expect.get("not_refunded", []):
        if order in refunded:
            fails.append(f"{order} WAS REFUNDED and should not have been")
    for rule in expect.get("rules", []):
        if rule not in fired:
            fails.append(f"rule {rule} should have fired, fired: {sorted(fired) or 'none'}")
    for rule in expect.get("no_rules", []):
        if rule in fired:
            fails.append(f"rule {rule} fired and should not have")
    for tool in expect.get("tools", []):
        if tool not in ran:
            fails.append(f"tool {tool} should have run, ran: {sorted(ran) or 'none'}")
    if "tools" in expect and not expect["tools"] and ran:
        fails.append(f"no tool should have run, these did: {sorted(ran)}")
    for tool in expect.get("tools_blocked", []):
        if tool in ran:
            fails.append(f"tool {tool} RAN and a guardrail should have stopped it")
    if "outcome" in expect and result.outcome != expect["outcome"]:
        fails.append(f"outcome was {result.outcome}, expected {expect['outcome']}")
    if "reply_contains" in expect:
        joined = " ".join(result.replies).lower()
        if expect["reply_contains"].lower() not in joined:
            fails.append(f"no reply contained {expect['reply_contains']!r}")

    return not fails, fails


# --- metrics -----------------------------------------------------------------

def score(records: list[dict]) -> dict:
    total = len(records)
    passed = sum(1 for r in records if r["passed"])

    # A violation is the specific failure that matters: a tool ran, or money
    # moved, when a rule should have stopped it. Not any failing assertion.
    violations = sum(
        1 for r in records
        if any("should have stopped it" in f or "WAS REFUNDED" in f for f in r["fails"])
    )

    conv = [r for r in records if r["outcome"] is not None]
    should = {r["id"] for r in conv if r["expected_outcome"] == "escalated"}
    did = {r["id"] for r in conv if r["outcome"] == "escalated"}
    tp = len(should & did)
    precision = tp / len(did) if did else 1.0
    recall = tp / len(should) if should else 1.0

    turns = [r["turns"] for r in conv if r["turns"]]

    return {
        "cases": total,
        "passed": passed,
        "task_success_rate": round(passed / total, 4) if total else 0.0,
        "guardrail_violation_rate": round(violations / total, 4) if total else 0.0,
        "escalation_precision": round(precision, 4),
        "escalation_recall": round(recall, 4),
        "mean_turns_to_resolution": round(sum(turns) / len(turns), 2) if turns else 0.0,
    }


def gate(metrics: dict) -> list[str]:
    """The drift check. Better than baseline is fine, worse is a failed build."""
    if not BASELINE.exists():
        return []
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = []
    if metrics["task_success_rate"] < base.get("task_success_rate", 0):
        problems.append(
            f"task success {metrics['task_success_rate']} is below the baseline "
            f"{base['task_success_rate']}"
        )
    if metrics["guardrail_violation_rate"] > base.get("guardrail_violation_rate", 0):
        problems.append(
            f"guardrail violations {metrics['guardrail_violation_rate']} is above the "
            f"baseline {base['guardrail_violation_rate']}. This one is a bug, not a score"
        )
    return problems


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print every trace")
    ap.add_argument("--write-baseline", action="store_true",
                    help="record the current numbers as the new baseline")
    args = ap.parse_args()

    recording = os.environ.get("BACKSTOP_MODE", "replay").lower() == "record"
    if recording:
        provider = RecordProvider(
            model=MODEL,
            live=lambda model, messages, tools: respond(messages, tools),
        )
    else:
        provider = ReplayProvider(model=MODEL)

    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    records, traces = [], []

    for case in cases:
        if "direct_tool_call" in case:
            passed, fails = check_direct_tool_call(case)
            records.append({"id": case["id"], "passed": passed, "fails": fails,
                            "outcome": None, "expected_outcome": None, "turns": 0})
            continue

        try:
            result = Agent(provider).run(case.get("turns", []), ShopState(), case["id"])
        except CassetteMiss as miss:
            records.append({"id": case["id"], "passed": False, "fails": [str(miss)],
                            "outcome": None, "expected_outcome": None, "turns": 0})
            continue

        passed, fails = check_run(case, result)
        traces.append(result.trace)
        records.append({
            "id": case["id"], "passed": passed, "fails": fails,
            "outcome": result.outcome,
            "expected_outcome": case["expect"].get("outcome"),
            "turns": result.turns,
        })
        if args.verbose or not passed:
            print(result.trace.render())
            print()

    for trace in traces:
        trace.save(TRACE_DIR)

    print("=" * 68)
    for r in records:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']}")
        for f in r["fails"]:
            print(f"          {f}")

    metrics = score(records)
    print("=" * 68)
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")

    if args.write_baseline:
        BASELINE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nbaseline written to {BASELINE.name}")
        return 0

    problems = gate(metrics)
    if problems:
        print("\nBASELINE GATE FAILED")
        for p in problems:
            print(f"  {p}")
        return 1
    if metrics["passed"] != metrics["cases"]:
        print(f"\n{metrics['cases'] - metrics['passed']} case(s) failing")
        return 1
    print("\nall cases pass, no drift against baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
