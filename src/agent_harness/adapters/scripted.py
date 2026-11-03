"""Scripted model client: replays a fixed list of turns. For tests and the offline demo.

No network, no keys. Running out of script raises, which the loop reports as MODEL_ERROR.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_harness.domain.loop import Message, ModelTurn
from agent_harness.domain.tools import ToolSpec


class ScriptExhausted(RuntimeError):
    """The script has no turn left for the model to play."""


class ScriptedModel:
    def __init__(self, turns: Sequence[ModelTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[tuple[Message, ...]] = []

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        self.calls.append(tuple(messages))
        if not self._turns:
            raise ScriptExhausted(f"script exhausted after {len(self.calls)} calls")
        return self._turns.pop(0)
