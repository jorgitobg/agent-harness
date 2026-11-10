"""Terminal confirmation: ask the operator before a risky tool call runs.

Fails closed: only an explicit ``y`` or ``yes`` approves. Anything else (empty line,
``n``, noise), end of input (no terminal, closed pipe) or Ctrl-C refuses the call.
``read`` and ``write`` are injectable so tests drive it without a terminal.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from agent_harness.domain.tools import ToolCall, ToolSpec

MAX_ARGS_CHARS = 200
APPROVE = frozenset({"y", "yes"})


def _shown_args(call: ToolCall) -> str:
    text = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= MAX_ARGS_CHARS else text[: MAX_ARGS_CHARS - 1] + "…"


class CliConfirmation:
    def __init__(
        self,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        self._read = read
        self._write = write

    def confirm(self, call: ToolCall, spec: ToolSpec) -> bool:
        self._write(f"[{spec.level.value}] {call.name} {_shown_args(call)}")
        try:
            answer = self._read("Allow this call? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            self._write("no answer: refused")
            return False
        return answer.strip().lower() in APPROVE
