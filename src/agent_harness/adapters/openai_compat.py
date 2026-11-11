"""Model client for any OpenAI-compatible chat-completions endpoint with tool calling.

Works with hosted services and local servers (llama.cpp, Ollama, OpenRouter). Standard
library only. Errors carry a status code or a short reason, never request or response bodies
(they may contain prompts, tool output or credentials). The HTTP layer is a port so tests
replay recorded responses without a network.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from agent_harness.domain.loop import Message, ModelTurn, Role, Usage
from agent_harness.domain.tools import ToolCall, ToolSpec

MAX_RESPONSE_BYTES = 4_000_000


class ModelCallError(RuntimeError):
    """The model call failed. The loop reports it as a MODEL_ERROR stop."""


class HttpPost(Protocol):
    def post(self, url: str, headers: Mapping[str, str], body: bytes) -> bytes: ...


class UrllibPost:
    """POST with a timeout and a response-size cap. https required unless the host is local."""

    def __init__(self, timeout_s: float = 60.0, max_bytes: int = MAX_RESPONSE_BYTES) -> None:
        self._timeout_s = timeout_s
        self._max_bytes = max_bytes

    def post(self, url: str, headers: Mapping[str, str], body: bytes) -> bytes:
        parsed = urlparse(url)
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ModelCallError("endpoint must be https (http only for localhost)")
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                data: bytes = response.read(self._max_bytes + 1)
        except urllib.error.HTTPError as error:
            raise ModelCallError(f"http status {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ModelCallError(f"network error ({type(error).__name__})") from error
        if len(data) > self._max_bytes:
            raise ModelCallError("response too large")
        return data


@dataclass(frozen=True, slots=True)
class OpenAICompatConfig:
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    max_tokens: int = 2048  # reasoning models spend this on hidden thinking; keep it generous
    system_prompt: str | None = None


def _tool_payload(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _message_payload(message: Message) -> dict[str, Any]:
    if message.role is Role.USER:
        return {"role": "user", "content": message.content}
    if message.role is Role.ASSISTANT:
        payload: dict[str, Any] = {"role": "assistant", "content": message.content or None}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                }
                for call in message.tool_calls
            ]
        return payload
    result = message.tool_result
    if result is None:
        raise ModelCallError("tool message without a result")
    return {
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": json.dumps(result.for_model(), default=str),
    }


def _as_dict(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _as_cost(value: object) -> float | None:
    """Provider cost, or ``None`` when absent/malformed.

    Malformed usage metadata must not cost us the model's real output: an unusable cost
    degrades to "unknown" instead of raising.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        cost = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(cost) or cost < 0:
        return None
    return cost


def _parse_arguments(arguments: object) -> dict[str, Any]:
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed: object = json.loads(arguments)
    except ValueError as error:
        raise ModelCallError("tool call arguments are not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ModelCallError("tool call arguments are not a JSON object")
    return cast(dict[str, Any], parsed)


def parse_response(raw: bytes) -> ModelTurn:
    """Turn one chat-completions response into a ModelTurn, or raise ModelCallError."""
    try:
        data = _as_dict(json.loads(raw))
        choice = _as_dict(cast(list[Any], data["choices"])[0])
        message = _as_dict(choice["message"])
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ModelCallError("response is not a chat completion") from error
    usage_data = _as_dict(data.get("usage"))
    usage = Usage(
        input_tokens=_as_int(usage_data.get("prompt_tokens")),
        output_tokens=_as_int(usage_data.get("completion_tokens")),
        cost_usd=_as_cost(usage_data.get("cost")),
    )
    raw_calls = message.get("tool_calls")
    calls: list[ToolCall] = []
    for index, item in enumerate(cast(list[Any], raw_calls) if isinstance(raw_calls, list) else []):
        entry = _as_dict(item)
        function = _as_dict(entry.get("function"))
        name = function.get("name")
        arguments = function.get("arguments", "{}")
        if not isinstance(name, str) or not name:
            raise ModelCallError("tool call without a name")
        parsed = _parse_arguments(arguments)
        call_id = entry.get("id")
        calls.append(
            ToolCall(
                call_id=call_id if isinstance(call_id, str) and call_id else f"call_{index}",
                name=name,
                arguments=parsed,
            )
        )
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    if not calls and not text.strip():
        reason = "token budget exhausted" if choice.get("finish_reason") == "length" else "empty"
        raise ModelCallError(f"model returned no text and no tool call ({reason})")
    return ModelTurn(text=text, tool_calls=tuple(calls), usage=usage)


class OpenAICompatModel:
    """Implements the ``ModelClient`` port."""

    def __init__(self, config: OpenAICompatConfig, http: HttpPost | None = None) -> None:
        self._config = config
        self._http: HttpPost = http if http is not None else UrllibPost()
        self._url = config.base_url.rstrip("/") + "/chat/completions"

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        payload_messages: list[dict[str, Any]] = []
        if self._config.system_prompt:
            payload_messages.append({"role": "system", "content": self._config.system_prompt})
        payload_messages += [_message_payload(m) for m in messages]
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": payload_messages,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if tools:
            body["tools"] = [_tool_payload(t) for t in tools]
        headers = {"content-type": "application/json"}
        if self._config.api_key:
            headers["authorization"] = f"Bearer {self._config.api_key}"
        return parse_response(self._http.post(self._url, headers, json.dumps(body).encode()))
