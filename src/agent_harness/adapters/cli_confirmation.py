"""Terminal confirmation: ask the operator before a risky tool call runs.

Fails closed: only an explicit ``y`` or ``yes`` approves. Anything else (empty line,
``n``, noise), end of input (no terminal, closed pipe) or Ctrl-C refuses the call.

The operator approves exactly what is shown: if the tool name or the arguments are too
long to display in full (``MAX_NAME_CHARS`` / ``MAX_ARGS_CHARS``), the call is refused
without prompting, so nothing can hide behind a cut-off. Control characters, newlines,
escape sequences and bidi marks in the name or values are shown as ``\\x..``/``\\u....``
escapes, never sent raw to the terminal. Raw arguments are not written anywhere else.
``read`` and ``write`` are injectable so tests drive it without a terminal.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from agent_harness.domain.tools import ToolCall, ToolSpec

MAX_ARGS_CHARS = 200
MAX_NAME_CHARS = 64
APPROVE = frozenset({"y", "yes"})


def _inert(text: str) -> str:
    """Replace every non-printable character (controls, DEL, C1, bidi, separators) by an escape."""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch.isprintable():
            out.append(ch)
        elif code <= 0xFF:
            out.append(f"\\x{code:02x}")
        elif code <= 0xFFFF:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    return "".join(out)


def _shown_args(call: ToolCall) -> str | None:
    """Escaped argument text, or ``None`` when it is too long to review in full."""
    text = _inert(json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, default=str))
    return text if len(text) <= MAX_ARGS_CHARS else None


class CliConfirmation:
    def __init__(
        self,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        self._read = read
        self._write = write

    def confirm(self, call: ToolCall, spec: ToolSpec) -> bool:
        name = _inert(call.name)
        shown = _shown_args(call)
        if len(name) > MAX_NAME_CHARS or shown is None:
            self._write(
                f"[{spec.level.value}] {name[:MAX_NAME_CHARS]}: too long to review: refused"
            )
            return False
        self._write(f"[{spec.level.value}] {name} {shown}")
        try:
            answer = self._read("Allow this call? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            self._write("no answer: refused")
            return False
        return answer.strip().lower() in APPROVE
