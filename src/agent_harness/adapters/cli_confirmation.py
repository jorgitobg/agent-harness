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
from typing import cast

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


def _fits_json(value: object, remaining: int, active: set[int]) -> int:
    """Subtract a lower bound on JSON size, stopping before large values are visited.

    Only plain JSON containers/primitives are accepted: no custom serialization or
    traversal hooks run while deciding whether an operator can review the call.
    """
    kind = type(value)
    if kind is str:
        remaining -= len(cast(str, value)) + 2  # escaping can only increase the length
    elif kind is int:
        number = cast(int, value)
        if number.bit_length() > MAX_ARGS_CHARS * 4:
            return -1
        remaining -= len(json.dumps(number))
    elif kind in (float, bool) or value is None:
        remaining -= len(json.dumps(value, ensure_ascii=False))
    elif kind in (dict, list, tuple):
        if id(value) in active:
            return -1
        size = len(cast(dict[str, object] | list[object] | tuple[object, ...], value))
        # Even empty strings/scalars require characters; reject before iteration.
        minimum = 2 + max(0, size - 1) * 2 + size * (5 if kind is dict else 1)
        remaining -= minimum
        if remaining < 0:
            return -1
        active.add(id(value))
        try:
            if kind is dict:
                for key, item in cast(dict[object, object], value).items():
                    if type(key) is not str:
                        return -1
                    # Minimum already includes two quotes for each key and one
                    # character for each value; charge their additional length.
                    remaining -= len(key)
                    if remaining < 0:
                        return -1
                    remaining = _fits_json(item, remaining + 1, active)
                    if remaining < 0:
                        return -1
            else:
                for item in cast(list[object] | tuple[object, ...], value):
                    remaining = _fits_json(item, remaining + 1, active)
                    if remaining < 0:
                        return -1
        finally:
            active.remove(id(value))
    else:
        return -1
    return remaining


def _shown_args(call: ToolCall) -> str | None:
    """Escaped argument text, or ``None`` when it is too long to review in full."""
    if _fits_json(call.arguments, MAX_ARGS_CHARS, set()) < 0:
        return None
    try:
        text = _inert(json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError, OverflowError):
        return None
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
        name = _inert(call.name[: MAX_NAME_CHARS + 1])
        if len(call.name) > MAX_NAME_CHARS or len(name) > MAX_NAME_CHARS:
            shown = None
        else:
            shown = _shown_args(call)
        if shown is None:
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
