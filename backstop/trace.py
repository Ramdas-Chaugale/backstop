"""A readable record of one run.

Adverts for this job ask you to debug failures from execution traces. That only
works if the trace says why something happened, not just what. So every entry
carries the thing that caused it: the guardrail name, the tool error, the
cassette key the model answer came from.

A failing eval prints its trace. You read the timeline and you know which rule
fired, or which tool argument was wrong, without re-running anything.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Event:
    step: int
    kind: str          # user | model | guardrail | tool | final
    detail: dict

    def line(self) -> str:
        d = self.detail
        if self.kind == "user":
            return f"  user      {d.get('text','')!r}"
        if self.kind == "model":
            calls = d.get("tool_calls") or []
            if calls:
                shown = ", ".join(f"{c['name']}({c['arguments']})" for c in calls)
                return f"  model     wants {shown}   [cassette {d.get('key','?')}]"
            return f"  model     says {d.get('content','')!r}   [cassette {d.get('key','?')}]"
        if self.kind == "guardrail":
            v = d.get("verdict")
            if v == "allow":
                return f"  guardrail allow    {d.get('tool')}"
            return f"  guardrail {v:<8} {d.get('tool')}  <- {d.get('rule')}: {d.get('reason')}"
        if self.kind == "tool":
            if d.get("error"):
                return f"  tool      {d.get('name')} FAILED: {d.get('error')}"
            return f"  tool      {d.get('name')} -> {d.get('result')}"
        if self.kind == "final":
            return f"  final     {d.get('outcome')}  after {d.get('turns')} turns"
        return f"  {self.kind} {d}"


@dataclass
class Trace:
    case_id: str
    started: float = field(default_factory=time.time)
    events: list[Event] = field(default_factory=list)

    def add(self, kind: str, **detail) -> None:
        self.events.append(Event(len(self.events), kind, detail))

    # --- questions a failing case asks of its own trace ---------------------

    def rules_fired(self) -> list[str]:
        return [e.detail["rule"] for e in self.events
                if e.kind == "guardrail" and e.detail.get("verdict") != "allow"]

    def tools_run(self) -> list[str]:
        return [e.detail["name"] for e in self.events
                if e.kind == "tool" and not e.detail.get("error")]

    def tool_errors(self) -> list[str]:
        return [e.detail["error"] for e in self.events
                if e.kind == "tool" and e.detail.get("error")]

    # --- output -------------------------------------------------------------

    def render(self) -> str:
        head = f"trace {self.case_id}"
        return "\n".join([head] + [e.line() for e in self.events])

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "started": self.started,
                "events": [{"step": e.step, "kind": e.kind, "detail": e.detail}
                           for e in self.events]}

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.case_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path
