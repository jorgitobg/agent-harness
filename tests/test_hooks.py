from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_harness.application.guardrails import GuardedRegistry
from agent_harness.application.hooks import (
    Allow,
    Block,
    HookedDispatcher,
    PreDecision,
    Rewrite,
)
from agent_harness.application.registry import ToolRegistry
from agent_harness.domain.policy import PermissionPolicy
from agent_harness.domain.tools import (
    PermissionLevel,
    ToolCall,
    ToolErrorKind,
    ToolResult,
    ToolSpec,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


class Confirm:
    def confirm(self, call: ToolCall, spec: ToolSpec) -> bool:
        return True


def make() -> tuple[GuardedRegistry, list[Mapping[str, Any]]]:
    reg = ToolRegistry()
    seen: list[Mapping[str, Any]] = []

    def handler(args: Mapping[str, Any]) -> Any:
        seen.append(args)
        return f"echo:{args['text']} token=sk-SECRET123"

    reg.register(ToolSpec("echo", "Echo text", SCHEMA, PermissionLevel.READ), handler)
    return GuardedRegistry(reg, PermissionPolicy(), Confirm()), seen


def call(text: str = "hi") -> ToolCall:
    return ToolCall("c1", "echo", {"text": text})


class Blocker:
    def before(self, call: ToolCall) -> PreDecision:
        return Block("no secrets") if "password" in str(call.arguments) else Allow()


class Upper:
    def before(self, call: ToolCall) -> PreDecision:
        return Rewrite({"text": str(call.arguments["text"]).upper()})


class BadRewrite:
    def before(self, call: ToolCall) -> PreDecision:
        return Rewrite({"nope": 1})


class Redact:
    def after(self, call: ToolCall, result: ToolResult) -> ToolResult:
        if not result.ok:
            return result
        out = str(result.output).replace("sk-SECRET123", "[REDACTED]")
        return ToolResult.success(call, out)


class Boom:
    def before(self, call: ToolCall) -> PreDecision:
        raise RuntimeError("hook bug")

    def after(self, call: ToolCall, result: ToolResult) -> ToolResult:
        raise RuntimeError("hook bug")


class Observer:
    def __init__(self) -> None:
        self.log: list[tuple[str, bool]] = []

    def after(self, call: ToolCall, result: ToolResult) -> ToolResult:
        self.log.append((call.name, result.ok))
        return result


def test_allow_passes_through_unchanged() -> None:
    inner, seen = make()
    res = HookedDispatcher(inner, pre=[Blocker()]).dispatch(call())
    assert res.ok and seen == [{"text": "hi"}]


def test_block_skips_handler_and_reports_reason() -> None:
    inner, seen = make()
    res = HookedDispatcher(inner, pre=[Blocker()]).dispatch(call("my password"))
    assert not res.ok and res.error_kind is ToolErrorKind.BLOCKED_BY_HOOK
    assert res.error_message == "no secrets" and seen == []


def test_rewrite_changes_arguments_seen_by_handler() -> None:
    inner, seen = make()
    res = HookedDispatcher(inner, pre=[Upper()]).dispatch(call("hi"))
    assert res.ok and seen == [{"text": "HI"}]


def test_rewrite_is_revalidated_by_inner_schema() -> None:
    inner, seen = make()
    res = HookedDispatcher(inner, pre=[BadRewrite()]).dispatch(call())
    assert res.error_kind is ToolErrorKind.INVALID_ARGUMENTS and seen == []


def test_later_pre_hook_sees_rewritten_arguments() -> None:
    inner, seen = make()
    res = HookedDispatcher(inner, pre=[Upper(), Blocker()]).dispatch(call("x"))
    assert res.ok and seen == [{"text": "X"}]


def test_post_hook_redacts_output() -> None:
    inner, _ = make()
    res = HookedDispatcher(inner, post=[Redact()]).dispatch(call())
    assert res.ok and "sk-SECRET123" not in str(res.output) and "[REDACTED]" in str(res.output)


def test_post_hooks_observe_failures_too_and_chain_in_order() -> None:
    inner, _ = make()
    obs = Observer()
    HookedDispatcher(inner, post=[Redact(), obs]).dispatch(ToolCall("c2", "missing", {}))
    assert obs.log == [("missing", False)]


def test_pre_hook_exception_fails_closed() -> None:
    inner, seen = make()
    res = HookedDispatcher(inner, pre=[Boom()]).dispatch(call())
    assert res.error_kind is ToolErrorKind.BLOCKED_BY_HOOK and seen == []


def test_post_hook_exception_withholds_output() -> None:
    inner, _ = make()
    res = HookedDispatcher(inner, post=[Boom()]).dispatch(call())
    assert not res.ok and res.error_kind is ToolErrorKind.BLOCKED_BY_HOOK
    assert "sk-SECRET123" not in str(res.for_model())
