"""Tests for Ollama routing in src/server.py."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient

import server
from server import OLLAMA_AGENTS, _detect_agent, app

client = TestClient(app)


# ── _detect_agent ─────────────────────────────────────────────────────────────

def test_detect_agent_ollama():
    assert _detect_agent("@ollama what is 2+2") == "@ollama"


def test_detect_agent_lucidia():
    assert _detect_agent("Hey @lucidia, can you help me?") == "@lucidia"


def test_detect_agent_copilot():
    assert _detect_agent("@copilot explain this code") == "@copilot"


def test_detect_agent_blackboxprogramming():
    assert _detect_agent("@blackboxprogramming help") == "@blackboxprogramming"


def test_detect_agent_returns_none_for_plain_message():
    assert _detect_agent("just a plain message with no mention") is None


def test_detect_agent_case_insensitive():
    assert _detect_agent("@OLLAMA hello") == "@ollama"
    assert _detect_agent("@Lucidia hello") == "@lucidia"


def test_detect_agent_strips_trailing_punctuation():
    assert _detect_agent("Hello @ollama, what time is it?") == "@ollama"


def test_detect_agent_returns_first_match():
    result = _detect_agent("@ollama and @copilot both here")
    assert result in OLLAMA_AGENTS


# ── OLLAMA_AGENTS set ─────────────────────────────────────────────────────────

def test_ollama_agents_contains_required_mentions():
    assert "@ollama" in OLLAMA_AGENTS
    assert "@copilot" in OLLAMA_AGENTS
    assert "@lucidia" in OLLAMA_AGENTS
    assert "@blackboxprogramming" in OLLAMA_AGENTS


def test_ollama_agents_no_external_providers():
    external = {"@openai", "@anthropic", "@chatgpt", "@claude", "@gpt4"}
    assert not OLLAMA_AGENTS & external


# ── /chat endpoint ────────────────────────────────────────────────────────────

def _mock_ollama_client(response_text: str = "Hello from Ollama!",
                         model: str = "llama3"):
    """Return a context-manager mock that simulates a successful Ollama call."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": response_text, "model": model}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


def test_chat_routes_ollama_mention_to_ollama():
    with patch("httpx.AsyncClient", return_value=_mock_ollama_client()):
        resp = client.post("/chat", json={"message": "@ollama what is 2+2"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "ollama"
    assert data["response"] == "Hello from Ollama!"


def test_chat_routes_copilot_to_ollama():
    with patch("httpx.AsyncClient", return_value=_mock_ollama_client()):
        resp = client.post("/chat", json={"message": "@copilot explain recursion"})
    assert resp.status_code == 200
    assert resp.json()["provider"] == "ollama"


def test_chat_routes_lucidia_to_ollama():
    with patch("httpx.AsyncClient", return_value=_mock_ollama_client()):
        resp = client.post("/chat", json={"message": "@lucidia what do you know about me?"})
    assert resp.status_code == 200
    assert resp.json()["provider"] == "ollama"


def test_chat_routes_blackboxprogramming_to_ollama():
    with patch("httpx.AsyncClient", return_value=_mock_ollama_client()):
        resp = client.post("/chat", json={"message": "@blackboxprogramming help debug this"})
    assert resp.status_code == 200
    assert resp.json()["provider"] == "ollama"


def test_chat_uses_agent_param_without_message_mention():
    with patch("httpx.AsyncClient", return_value=_mock_ollama_client()):
        resp = client.post("/chat", json={"message": "plain message", "agent": "ollama"})
    assert resp.status_code == 200
    assert resp.json()["provider"] == "ollama"


def test_chat_no_mention_returns_400():
    resp = client.post("/chat", json={"message": "what is the weather today?"})
    assert resp.status_code == 400
    assert "No recognised @agent mention" in resp.json()["detail"]


def test_chat_ollama_unreachable_returns_503():
    import httpx as _httpx

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=_httpx.ConnectError("refused"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        resp = client.post("/chat", json={"message": "@ollama are you there?"})
    assert resp.status_code == 503
    assert "Ollama" in resp.json()["detail"]


def test_chat_returns_model_name():
    with patch("httpx.AsyncClient", return_value=_mock_ollama_client(model="mistral")):
        resp = client.post(
            "/chat",
            json={"message": "@ollama hello", "model": "mistral"},
        )
    assert resp.status_code == 200
    assert resp.json()["model"] == "mistral"
