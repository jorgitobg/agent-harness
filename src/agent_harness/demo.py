"""Offline demo: ``python -m agent_harness.demo``.

Runs the real loop, registry, guardrails and budgets with a scripted model and two
in-memory tools. No API keys, no network. The model asks for a read (allowed by
default) and then a write (asks the operator first). Pass ``--router`` to put a cheap tier
that is down in front of the scripted model (the router falls back). Pass ``--deny`` to answer the
confirmation with "no" and watch the guard refuse the write without running it.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from agent_harness.adapters.confirmation import ScriptedConfirmation
from agent_harness.adapters.scripted import ScriptedModel
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.guardrails import GuardedRegistry
from agent_harness.application.hooks import HookedDispatcher
from agent_harness.application.registry import ToolRegistry
from agent_harness.application.router import ModelRouter
from agent_harness.domain.loop import Budget, ModelTurn, Usage
from agent_harness.domain.policy import PermissionPolicy
from agent_harness.domain.tools import PermissionLevel, ToolCall, ToolResult, ToolSpec

NOTES: dict[str, str] = {
    "welcome": "Agent harness demo note.",
    "todo": "Write the next user story.",
    "deploy": "Staging login uses token sk-DEMO-0123456789 (fake, for the redaction demo).",
}
SECRET = re.compile(r"sk-[A-Za-z0-9-]{8,}")

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1}},
    "required": ["query"],
    "additionalProperties": False,
}

READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string", "minLength": 1}},
    "required": ["title"],
    "additionalProperties": False,
}

WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "body": {"type": "string"},
    },
    "required": ["title", "body"],
    "additionalProperties": False,
}


def search_notes(args: Mapping[str, Any]) -> list[str]:
    query = str(args["query"]).lower()
    return [title for title, body in NOTES.items() if query in body.lower() or query in title]


def read_note(args: Mapping[str, Any]) -> str:
    return NOTES.get(str(args["title"]), "")


class RedactSecrets:
    """Post-hook: mask anything that looks like an API token before the model sees it."""

    def after(self, call: ToolCall, result: ToolResult) -> ToolResult:
        if result.ok and isinstance(result.output, str):
            return replace(result, output=SECRET.sub("[REDACTED]", result.output))
        return result


def write_note(args: Mapping[str, Any]) -> str:
    title = str(args["title"])
    NOTES[title] = str(args["body"])
    return f"saved note {title!r}"


def build_registry(*, with_read_note: bool = False) -> ToolRegistry:
    """Two tools by default (the recorded live run pins that set); the demo adds read_note."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "search_notes",
            "Search notes by a substring of the title or body.",
            SEARCH_SCHEMA,
            PermissionLevel.READ,
        ),
        search_notes,
    )
    if with_read_note:
        registry.register(
            ToolSpec(
                "read_note",
                "Return the body of one note by title.",
                READ_SCHEMA,
                PermissionLevel.READ,
            ),
            read_note,
        )
    registry.register(
        ToolSpec(
            "write_note",
            "Create or overwrite a note.",
            WRITE_SCHEMA,
            PermissionLevel.WRITE,
        ),
        write_note,
    )
    return registry


def scripted_turns() -> list[ModelTurn]:
    usage = Usage(input_tokens=120, output_tokens=30, cost_usd=0.0)
    return [
        ModelTurn(
            text="Let me look for notes about the next step.",
            tool_calls=(ToolCall("c1", "search_notes", {"query": "next"}),),
            usage=usage,
        ),
        ModelTurn(
            text="Checking the deploy note.",
            tool_calls=(ToolCall("c1b", "read_note", {"title": "deploy"}),),
            usage=usage,
        ),
        ModelTurn(
            text="Recording the plan as a note.",
            tool_calls=(
                ToolCall(
                    "c2",
                    "write_note",
                    {"title": "plan", "body": "Ship the offline demo."},
                ),
            ),
            usage=usage,
        ),
        ModelTurn(text="Done: the plan is saved as note 'plan'.", usage=usage),
    ]


class _Offline:
    """A model tier that is down, to show the router's fallback."""

    def complete(self, messages: object, tools: object) -> ModelTurn:
        raise ConnectionError("tier unavailable")


def run(approve: bool, router: bool = False) -> int:
    registry = build_registry(with_read_note=True)
    confirmation = ScriptedConfirmation(answers=[approve])
    guarded = GuardedRegistry(registry, PermissionPolicy(), confirmation)
    scripted = ScriptedModel(scripted_turns())
    routed = ModelRouter(_Offline(), scripted) if router else None
    loop = AgentLoop(
        routed if routed is not None else scripted,
        registry,
        Budget(max_steps=5),
        dispatcher=HookedDispatcher(guarded, post=[RedactSecrets()]),
    )

    print(f"operator answer to confirmations: {'yes' if approve else 'no'}")
    result = loop.run("Find the next step in my notes and save the plan.")

    for message in result.transcript:
        if message.tool_result is not None:
            r = message.tool_result
            status = "ok" if r.ok else f"refused/failed ({r.error_kind})"
            print(f"  tool {r.name}: {status} -> {r.output if r.ok else r.error_message}")
    print(f"confirmations asked: {confirmation.asked}")
    print(f"stop_reason={result.stop_reason.value} steps={result.steps}")
    print(f"final_answer={result.final_answer!r}")
    print(f"notes now: {sorted(NOTES)}")
    if routed is not None:
        served = [f"{r.wanted}->{r.served_by}" for r in routed.routes]
        print(f"model routes: {served}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    return run(approve="--deny" not in args, router="--router" in args)


if __name__ == "__main__":
    raise SystemExit(main())
