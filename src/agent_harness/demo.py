"""Offline demo: ``python -m agent_harness.demo``.

Runs the real loop, registry, guardrails and budgets with a scripted model and two
in-memory tools. No API keys, no network. The model asks for a read (allowed by
default) and then a write (asks the operator first). Pass ``--deny`` to answer the
confirmation with "no" and watch the guard refuse the write without running it.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from agent_harness.adapters.confirmation import ScriptedConfirmation
from agent_harness.adapters.scripted import ScriptedModel
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.guardrails import GuardedRegistry
from agent_harness.application.registry import ToolRegistry
from agent_harness.domain.loop import Budget, ModelTurn, Usage
from agent_harness.domain.policy import PermissionPolicy
from agent_harness.domain.tools import PermissionLevel, ToolCall, ToolSpec

NOTES: dict[str, str] = {
    "welcome": "Agent harness demo note.",
    "todo": "Write the next user story.",
}

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1}},
    "required": ["query"],
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


def write_note(args: Mapping[str, Any]) -> str:
    title = str(args["title"])
    NOTES[title] = str(args["body"])
    return f"saved note {title!r}"


def build_registry() -> ToolRegistry:
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


def run(approve: bool) -> int:
    registry = build_registry()
    confirmation = ScriptedConfirmation(answers=[approve])
    guarded = GuardedRegistry(registry, PermissionPolicy(), confirmation)
    loop = AgentLoop(
        ScriptedModel(scripted_turns()),
        registry,
        Budget(max_steps=5),
        dispatcher=guarded,
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
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    return run(approve="--deny" not in args)


if __name__ == "__main__":
    raise SystemExit(main())
