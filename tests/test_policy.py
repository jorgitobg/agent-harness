from __future__ import annotations

import pytest

from agent_harness.domain.policy import Decision, PermissionPolicy
from agent_harness.domain.tools import PermissionLevel, ToolSpec


def spec(name: str, level: PermissionLevel) -> ToolSpec:
    return ToolSpec(name, f"Tool {name}.", {"type": "object"}, level)


def test_default_policy_is_read_allow_write_ask_destructive_ask() -> None:
    policy = PermissionPolicy()
    assert policy.decide(spec("r", PermissionLevel.READ)) is Decision.ALLOW
    assert policy.decide(spec("w", PermissionLevel.WRITE)) is Decision.ASK
    assert policy.decide(spec("d", PermissionLevel.DESTRUCTIVE)) is Decision.ASK


def test_level_decisions_can_be_overridden() -> None:
    policy = PermissionPolicy(
        level_decisions={
            PermissionLevel.READ: Decision.ALLOW,
            PermissionLevel.WRITE: Decision.DENY,
            PermissionLevel.DESTRUCTIVE: Decision.DENY,
        }
    )
    assert policy.decide(spec("w", PermissionLevel.WRITE)) is Decision.DENY


def test_deny_list_overrides_level_allow() -> None:
    policy = PermissionPolicy(deny=frozenset({"read_secret"}))
    assert policy.decide(spec("read_secret", PermissionLevel.READ)) is Decision.DENY


def test_allow_list_overrides_level_ask() -> None:
    policy = PermissionPolicy(allow=frozenset({"write_note"}))
    assert policy.decide(spec("write_note", PermissionLevel.WRITE)) is Decision.ALLOW


def test_name_in_both_lists_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="both allow and deny: rm"):
        PermissionPolicy(allow=frozenset({"rm"}), deny=frozenset({"rm"}))


def test_policy_must_cover_every_level() -> None:
    with pytest.raises(ValueError, match="no decision for levels"):
        PermissionPolicy(level_decisions={PermissionLevel.READ: Decision.ALLOW})
