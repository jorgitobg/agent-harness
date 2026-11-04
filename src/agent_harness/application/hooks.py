"""Pre/post tool hooks wrapped around any dispatcher.

Flow for one call, outermost layer:
1. pre-hooks run in order. Each returns ``Allow``, ``Block(reason)`` or ``Rewrite(arguments)``.
   The first ``Block`` stops the call: the inner dispatcher never runs and the model gets a
   ``blocked_by_hook`` result. A ``Rewrite`` replaces the arguments seen by later hooks and by
   the inner dispatcher, which still validates them against the schema and applies policy.
2. the inner dispatcher runs (typically ``GuardedRegistry``).
3. post-hooks run in order and may return a replacement result (e.g. with redacted output).

Fail closed: a pre-hook that raises blocks the call; a post-hook that raises replaces the
result with a ``blocked_by_hook`` failure so unredacted output is never passed on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from agent_harness.domain.tools import ToolCall, ToolErrorKind, ToolResult


@dataclass(frozen=True, slots=True)
class Allow:
    """Let the call through unchanged."""


@dataclass(frozen=True, slots=True)
class Block:
    reason: str


@dataclass(frozen=True, slots=True)
class Rewrite:
    arguments: Mapping[str, Any]


PreDecision = Allow | Block | Rewrite


class PreToolHook(Protocol):
    def before(self, call: ToolCall) -> PreDecision: ...


class PostToolHook(Protocol):
    def after(self, call: ToolCall, result: ToolResult) -> ToolResult: ...


class _Dispatcher(Protocol):
    def dispatch(self, call: ToolCall) -> ToolResult: ...


class HookedDispatcher:
    def __init__(
        self,
        inner: _Dispatcher,
        pre: Sequence[PreToolHook] = (),
        post: Sequence[PostToolHook] = (),
    ) -> None:
        self._inner = inner
        self._pre = tuple(pre)
        self._post = tuple(post)

    def dispatch(self, call: ToolCall) -> ToolResult:
        current = call
        for hook in self._pre:
            try:
                decision = hook.before(current)
            except Exception:  # noqa: BLE001 - fail closed
                return ToolResult.failure(
                    call, ToolErrorKind.BLOCKED_BY_HOOK, "a pre-tool hook failed; call not run"
                )
            if isinstance(decision, Block):
                return ToolResult.failure(call, ToolErrorKind.BLOCKED_BY_HOOK, decision.reason)
            if isinstance(decision, Rewrite):
                current = replace(current, arguments=dict(decision.arguments))

        result = self._inner.dispatch(current)

        for post in self._post:
            try:
                result = post.after(current, result)
            except Exception:  # noqa: BLE001 - fail closed: never forward unredacted output
                return ToolResult.failure(
                    current,
                    ToolErrorKind.BLOCKED_BY_HOOK,
                    "a post-tool hook failed; output withheld",
                )
        return result
