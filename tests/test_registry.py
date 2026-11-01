from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agent_harness.application.registry import ToolRegistrationError, ToolRegistry
from agent_harness.domain.tools import (
    PermissionLevel,
    ToolCall,
    ToolErrorKind,
    ToolSpec,
)

ADD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def add(args: Mapping[str, Any]) -> int:
    return int(args["a"]) + int(args["b"])


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec("add", "Add two integers.", ADD_SCHEMA), add)
    return reg


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(call_id="c1", name=name, arguments=arguments)


def test_valid_call_runs_handler(registry: ToolRegistry) -> None:
    result = registry.dispatch(call("add", a=2, b=3))
    assert result.ok
    assert result.output == 5
    assert result.for_model() == {"ok": True, "output": 5}


def test_invalid_arguments_do_not_run_handler() -> None:
    ran: list[bool] = []
    reg = ToolRegistry()
    reg.register(
        ToolSpec("add", "Add two integers.", ADD_SCHEMA),
        lambda args: ran.append(True),
    )
    result = reg.dispatch(call("add", a="two", b=3))
    assert not result.ok
    assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
    assert ran == []
    assert result.issues[0].path == "$.a"


def test_missing_and_extra_arguments_are_reported(registry: ToolRegistry) -> None:
    result = registry.dispatch(call("add", a=1, c=9))
    messages = " ".join(i.message for i in result.issues)
    assert "'b' is a required property" in messages
    assert "'c' was unexpected" in messages
    payload = result.for_model()
    assert payload["error"] == "invalid_arguments"
    assert len(payload["issues"]) == 2


def test_unknown_tool_lists_available_tools(registry: ToolRegistry) -> None:
    result = registry.dispatch(call("multiply", a=1, b=2))
    assert result.error_kind is ToolErrorKind.UNKNOWN_TOOL
    assert result.error_message is not None and "add" in result.error_message


def test_handler_exception_is_returned_as_data() -> None:
    reg = ToolRegistry()

    def boom(_: Mapping[str, Any]) -> None:
        raise RuntimeError("disk full")

    reg.register(ToolSpec("boom", "Always fails.", {"type": "object"}), boom)
    result = reg.dispatch(call("boom"))
    assert result.error_kind is ToolErrorKind.EXECUTION_FAILED
    assert result.error_message == "RuntimeError: disk full"


def test_duplicate_registration_fails(registry: ToolRegistry) -> None:
    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.register(ToolSpec("add", "Again.", ADD_SCHEMA), add)


def test_broken_schema_fails_at_registration() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolRegistrationError, match="invalid input schema"):
        reg.register(ToolSpec("bad", "Bad schema.", {"type": "objekt"}), add)


def test_schema_must_describe_an_object() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolRegistrationError, match="must describe an object"):
        reg.register(ToolSpec("s", "String input.", {"type": "string"}), add)


def test_specs_keep_registration_order_and_level(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec("rm", "Remove a file.", {"type": "object"}, PermissionLevel.DESTRUCTIVE),
        lambda _: None,
    )
    assert [s.name for s in registry.specs()] == ["add", "rm"]
    assert registry.specs()[1].level is PermissionLevel.DESTRUCTIVE
    assert "rm" in registry and len(registry) == 2


@pytest.mark.parametrize("name", ["", "has space", "semi;colon"])
def test_invalid_tool_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="invalid tool name"):
        ToolSpec(name, "x", {"type": "object"})


def test_empty_description_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a description"):
        ToolSpec("ok", "  ", {"type": "object"})


def test_nested_paths_are_rendered_as_json_path() -> None:
    reg = ToolRegistry()
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
    }
    reg.register(ToolSpec("sum", "Sum items.", schema), lambda a: sum(a["items"]))
    result = reg.dispatch(call("sum", items=[1, "x"]))
    assert result.issues[0].path == "$.items[1]"
