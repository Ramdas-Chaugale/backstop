"""Model layer with record and replay.

The problem this solves: an agent evaluation suite that calls a live model cannot
run in continuous integration. It costs money on every push, and the same input
gives a different answer each time, so a failing build tells you nothing about
whether your change caused it.

So the model layer records every call once against a real provider, keyed by a
hash of what was sent, and replays from disk afterwards. CI runs in replay mode:
free, offline and deterministic. Recording a new cassette is a deliberate act.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CASSETTE_DIR = Path(__file__).resolve().parent.parent / "evals" / "cassettes"


def fingerprint(model: str, messages: list[dict], tools: list[dict] | None) -> str:
    """A stable key for one model call.

    sort_keys matters. Without it a dict that happens to serialise in a different
    order produces a different hash and silently misses the cassette.
    """
    payload = json.dumps(
        {"model": model, "messages": messages, "tools": tools or []},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class CassetteMiss(RuntimeError):
    """Replay was asked for a call that was never recorded."""


@dataclass
class Provider:
    """Base class. Subclasses implement complete()."""

    model: str
    calls: int = field(default=0, init=False)

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        raise NotImplementedError


@dataclass
class ReplayProvider(Provider):
    """Reads recorded responses from disk. Never touches the network."""

    cassette_dir: Path = CASSETTE_DIR
    strict: bool = True

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls += 1
        key = fingerprint(self.model, messages, tools)
        path = self.cassette_dir / f"{key}.json"
        if not path.exists():
            if self.strict:
                raise CassetteMiss(
                    f"no cassette for {key}. Record it with BACKSTOP_MODE=record, "
                    f"then commit evals/cassettes/{key}.json"
                )
            return {"role": "assistant", "content": "", "tool_calls": []}
        return json.loads(path.read_text(encoding="utf-8"))["response"]


@dataclass
class RecordProvider(Provider):
    """Calls the real provider once, writes the answer to a cassette, returns it.

    Kept deliberately thin. The point of this file is the record and replay
    mechanism, not provider specific plumbing, so the live call is injected.
    """

    live: Any = None
    cassette_dir: Path = CASSETTE_DIR

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls += 1
        key = fingerprint(self.model, messages, tools)
        path = self.cassette_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.live is None:
            raise RuntimeError("record mode needs a live client passed as live=")
        response = self.live(model=self.model, messages=messages, tools=tools)
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"model": self.model, "messages": messages, "tools": tools,
                 "response": response},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return response


def build(model: str = "stub-1", live: Any = None) -> Provider:
    """Pick a provider from BACKSTOP_MODE. Replay is the default, so a fresh
    clone and a CI runner behave the same way without configuration."""
    mode = os.environ.get("BACKSTOP_MODE", "replay").lower()
    if mode == "record":
        return RecordProvider(model=model, live=live)
    return ReplayProvider(model=model)
