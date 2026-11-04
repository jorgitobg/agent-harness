"""Confirmation adapters: answer the ``ConfirmationPort`` without a human at the keyboard.

``ScriptedConfirmation`` replays a fixed list of answers and records every question it
was asked, so tests and the offline demo can assert on what the operator was shown.
A real CLI prompt would implement the same port.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_harness.domain.tools import ToolCall, ToolSpec


class ConfirmationScriptExhausted(RuntimeError):
    """More questions were asked than answers were scripted."""


class ScriptedConfirmation:
    def __init__(self, answers: Sequence[bool]) -> None:
        self._answers = list(answers)
        self.asked: list[tuple[str, str]] = []

    def confirm(self, call: ToolCall, spec: ToolSpec) -> bool:
        self.asked.append((call.name, spec.level.value))
        if not self._answers:
            raise ConfirmationScriptExhausted(f"no scripted answer for {call.name!r}")
        return self._answers.pop(0)
