from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_harness.adapters.openai_compat import (
    ModelCallError,
    OpenAICompatConfig,
    OpenAICompatModel,
    UrllibPost,
    parse_response,
)
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.demo import build_registry
from agent_harness.domain.loop import Message, Role, StopReason

FIXTURE: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "openai_compat_tool_roundtrip.json").read_text()
)
PROMPT = "Search my notes for the word 'story' and tell me the title of the matching note."


class Replay:
    """Serves the recorded responses in order and checks each request matches the recording."""

    def __init__(self, recorded: list[dict[str, Any]]) -> None:
        self.recorded = list(recorded)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[Mapping[str, str]] = []

    def post(self, url: str, headers: Mapping[str, str], body: bytes) -> bytes:
        self.requests.append(json.loads(body))
        self.headers.append(headers)
        return json.dumps(self.recorded.pop(0)["response"]).encode()


def model(http: Any, **kw: Any) -> OpenAICompatModel:
    config = OpenAICompatConfig(
        "https://api.example.test/v1", "any-model", "SECRET-KEY", system_prompt="Use the tools "
        "when needed. Be brief.", **kw,
    )  # fmt: skip
    return OpenAICompatModel(config, http)


def test_recorded_tool_conversation_replays_through_the_real_loop() -> None:
    http = Replay(FIXTURE)
    run = AgentLoop(model(http), build_registry()).run(PROMPT)
    assert run.stop_reason is StopReason.FINAL_ANSWER and run.steps == 2
    assert run.final_answer is not None and "todo" in run.final_answer
    assert run.usage.input_tokens == 888 and run.usage.output_tokens == 154


def test_requests_match_what_the_real_service_accepted() -> None:
    http = Replay(FIXTURE)
    AgentLoop(model(http), build_registry()).run(PROMPT)
    for sent, recorded in zip(http.requests, FIXTURE, strict=True):
        assert sent["messages"] == recorded["request"]["messages"]
        assert [t["function"]["name"] for t in sent["tools"]] == recorded["request"]["tool_names"]
        assert sent["stream"] is False and sent["max_tokens"] == 2048


def test_auth_header_only_when_a_key_is_set() -> None:
    http = Replay(FIXTURE)
    model(http).complete([Message(Role.USER, content="hi")], [])
    assert http.headers[0]["authorization"] == "Bearer SECRET-KEY"
    keyless = Replay(FIXTURE)
    OpenAICompatModel(OpenAICompatConfig("http://localhost:1/v1", "m"), keyless).complete(
        [Message(Role.USER, content="hi")], []
    )
    assert "authorization" not in keyless.headers[0]
    assert "SECRET-KEY" not in repr(OpenAICompatConfig("u", "m", "SECRET-KEY"))


def choice(message: dict[str, Any], finish: str = "stop") -> bytes:
    return json.dumps({"choices": [{"finish_reason": finish, "message": message}]}).encode()


def test_tool_call_without_id_gets_a_stable_one_and_empty_arguments_are_allowed() -> None:
    turn = parse_response(
        choice({"content": None, "tool_calls": [{"function": {"name": "search_notes"}}]})
    )
    assert turn.tool_calls[0].call_id == "call_0" and dict(turn.tool_calls[0].arguments) == {}


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        b'{"choices": []}',
        b'{"choices": [{}]}',
        choice({"content": None, "tool_calls": [{"function": {"arguments": "{}"}}]}),
        choice({"content": None, "tool_calls": [{"function": {"name": "t", "arguments": "{"}}]}),
        choice({"content": None, "tool_calls": [{"function": {"name": "t", "arguments": "[1]"}}]}),
        choice({"content": None}),
        choice({"content": "  "}),
    ],
)
def test_malformed_responses_are_model_errors(raw: bytes) -> None:
    with pytest.raises(ModelCallError):
        parse_response(raw)


def test_exhausted_token_budget_is_named() -> None:
    with pytest.raises(ModelCallError, match="token budget exhausted"):
        parse_response(choice({"content": None, "reasoning": "thinking"}, finish="length"))


def test_odd_usage_values_are_ignored_not_trusted() -> None:
    raw = json.dumps(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": -5, "completion_tokens": True, "cost": "free"},
        }
    ).encode()
    turn = parse_response(raw)
    assert turn.usage.total_tokens == 0 and turn.usage.cost_usd is None


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf"), float("-inf"), 10**400])
def test_malformed_provider_cost_is_unknown_without_losing_model_output(cost: object) -> None:
    raw = json.dumps(
        {"choices": [{"message": {"content": "ok"}}], "usage": {"cost": cost}}
    ).encode()

    turn = parse_response(raw)

    assert turn.text == "ok"
    assert turn.usage.cost_usd is None


def test_explicit_zero_provider_cost_is_preserved() -> None:
    raw = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {"cost": 0}}).encode()

    assert parse_response(raw).usage.cost_usd == 0.0


def test_a_model_error_stops_the_loop_with_a_typed_reason() -> None:
    class Down:
        def post(self, url: str, headers: Mapping[str, str], body: bytes) -> bytes:
            raise ModelCallError("http status 503")

    run = AgentLoop(model(Down()), build_registry()).run("hi")
    assert run.stop_reason is StopReason.MODEL_ERROR


def test_tool_message_without_result_is_rejected() -> None:
    with pytest.raises(ModelCallError):
        model(Replay(FIXTURE)).complete([Message(Role.TOOL)], [])


class Resp:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, n: int) -> bytes:
        return self.data[:n]


def patch_urlopen(monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    def fake(request: urllib.request.Request, timeout: float) -> Resp:
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, bytes)
        return Resp(result)

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_transport_success_and_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_urlopen(monkeypatch, b"x" * 11)
    assert UrllibPost(max_bytes=11).post("https://x.test/", {}, b"{}") == b"x" * 11
    with pytest.raises(ModelCallError, match="too large"):
        UrllibPost(max_bytes=10).post("https://x.test/", {}, b"{}")


def test_transport_allows_plain_http_only_for_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_urlopen(monkeypatch, b"{}")
    assert UrllibPost().post("http://localhost:11434/v1/chat/completions", {}, b"{}") == b"{}"
    for url in ("http://example.test/", "file:///etc/passwd", "ftp://x.test/"):
        with pytest.raises(ModelCallError, match="https"):
            UrllibPost().post(url, {}, b"{}")


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("https://x.test/", 429, "SECRET", None, None),  # type: ignore[arg-type]
        urllib.error.URLError("SECRET-DETAIL"),
        TimeoutError("SECRET-DETAIL"),
    ],
)
def test_transport_errors_never_carry_bodies(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    patch_urlopen(monkeypatch, error)
    with pytest.raises(ModelCallError) as info:
        UrllibPost().post("https://x.test/", {}, b"{}")
    assert "SECRET" not in str(info.value)
