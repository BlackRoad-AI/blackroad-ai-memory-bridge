"""Tests for src/ollama_router.py — Ollama routing layer."""
import json
import sys
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import ollama_router
from ollama_router import (
    OLLAMA_AGENTS,
    OllamaError,
    OllamaUnavailableError,
    chat,
    detect_agent,
    route,
    routes_to_ollama,
)


# ── detect_agent ──────────────────────────────────────────────────────────────

def test_detect_agent_ollama():
    assert detect_agent("@ollama explain this") == "ollama"


def test_detect_agent_copilot():
    assert detect_agent("hey @copilot, help me") == "copilot"


def test_detect_agent_lucidia():
    assert detect_agent("@lucidia what do you think?") == "lucidia"


def test_detect_agent_blackboxprogramming():
    assert detect_agent("@blackboxprogramming fix the bug") == "blackboxprogramming"


def test_detect_agent_case_insensitive():
    assert detect_agent("@Copilot review this") == "copilot"
    assert detect_agent("@LUCIDIA help") == "lucidia"
    assert detect_agent("@BlackBoxProgramming go") == "blackboxprogramming"


def test_detect_agent_trailing_dot_stripped():
    # User typed "@copilot." or "@blackboxprogramming."
    assert detect_agent("@copilot. fix it") == "copilot"
    assert detect_agent("@blackboxprogramming. do it") == "blackboxprogramming"
    assert detect_agent("@lucidia. thanks") == "lucidia"


def test_detect_agent_returns_first_match():
    result = detect_agent("@lucidia and @copilot, both help")
    assert result == "lucidia"


def test_detect_agent_no_mention_returns_none():
    assert detect_agent("just a plain message") is None


def test_detect_agent_unknown_mention_returns_none():
    assert detect_agent("@claude help me") is None
    assert detect_agent("@chatgpt explain") is None


def test_detect_agent_empty_string_returns_none():
    assert detect_agent("") is None


# ── routes_to_ollama ──────────────────────────────────────────────────────────

def test_routes_to_ollama_true_for_all_agents():
    for agent in OLLAMA_AGENTS:
        assert routes_to_ollama(f"@{agent} do something") is True


def test_routes_to_ollama_false_without_mention():
    assert routes_to_ollama("no agent here") is False


def test_routes_to_ollama_false_for_external_providers():
    assert routes_to_ollama("@claude help") is False
    assert routes_to_ollama("@chatgpt answer this") is False
    assert routes_to_ollama("@openai generate") is False


# ── OLLAMA_AGENTS set ─────────────────────────────────────────────────────────

def test_ollama_agents_contains_required_aliases():
    assert "ollama"              in OLLAMA_AGENTS
    assert "copilot"             in OLLAMA_AGENTS
    assert "lucidia"             in OLLAMA_AGENTS
    assert "blackboxprogramming" in OLLAMA_AGENTS


def test_ollama_agents_excludes_external_providers():
    for external in ("claude", "chatgpt", "openai", "gemini"):
        assert external not in OLLAMA_AGENTS


# ── chat (HTTP mocked) ────────────────────────────────────────────────────────

def _mock_response(content: str) -> MagicMock:
    body = json.dumps({"message": {"content": content}}).encode()
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_chat_returns_response_text():
    with patch("urllib.request.urlopen", return_value=_mock_response("Hello!")):
        result = chat("@ollama say hello")
    assert result == "Hello!"


def test_chat_sends_system_prompt():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        return _mock_response("ok")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        chat("hi", system="You are a helpful assistant.")

    messages = captured["body"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"


def test_chat_no_system_prompt_skips_system_message():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        return _mock_response("ok")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        chat("hi", system="")

    messages = captured["body"]["messages"]
    assert all(m["role"] != "system" for m in messages)


def test_chat_raises_unavailable_on_url_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(OllamaUnavailableError):
            chat("@ollama test")


def test_chat_raises_ollama_error_on_bad_payload():
    mock = MagicMock()
    mock.read.return_value = b'{"unexpected": "format"}'
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock):
        with pytest.raises(OllamaError):
            chat("@ollama test")


# ── route ─────────────────────────────────────────────────────────────────────

def test_route_returns_true_and_response_when_agent_found():
    with patch("ollama_router.chat", return_value="Sure!") as mock_chat:
        routed, response = route("@ollama help me")
    assert routed is True
    assert response == "Sure!"
    mock_chat.assert_called_once()


def test_route_returns_false_and_empty_when_no_agent():
    with patch("ollama_router.chat") as mock_chat:
        routed, response = route("just a plain message")
    assert routed is False
    assert response == ""
    mock_chat.assert_not_called()


def test_route_passes_model_to_chat():
    captured = {}

    def fake_chat(prompt, model, system):
        captured["model"] = model
        captured["system"] = system
        return "done"

    with patch("ollama_router.chat", side_effect=fake_chat):
        route("@lucidia go", model="mistral", system="Be helpful.")

    assert captured["model"] == "mistral"
    assert captured["system"] == "Be helpful."
