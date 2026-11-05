from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_harness import live
from tests.test_openai_compat import FIXTURE, PROMPT, Replay


def test_live_cli_runs_the_loop_against_a_replayed_endpoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    http = Replay(FIXTURE)
    original = live.OpenAICompatModel

    def build(config: Any) -> Any:
        return original(config, http)

    monkeypatch.setattr(live, "OpenAICompatModel", build)
    monkeypatch.setenv("AGENT_HARNESS_API_KEY", "SECRET-KEY")
    code = live.main(["--base-url", "https://api.example.test/v1", "--model", "m", PROMPT])
    out = capsys.readouterr().out
    assert code == 0 and "stop_reason=final_answer" in out and "todo" in out
    assert "SECRET-KEY" not in out
    assert http.headers[0]["authorization"] == "Bearer SECRET-KEY"


def test_live_cli_exits_nonzero_when_the_model_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.delenv("AGENT_HARNESS_API_KEY", raising=False)
    code = live.main(["--base-url", "ftp://nope.test", "--model", "m", "hi"])
    assert code == 1 and "stop_reason=model_error" in capsys.readouterr().out
    assert json.dumps(FIXTURE)  # fixture stays valid JSON
