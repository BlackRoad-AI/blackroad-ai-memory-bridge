"""
BlackRoad AI — Ollama Router

Routes @-mention agent requests to a local Ollama instance instead of any
external AI provider (ChatGPT, Copilot, Claude, etc.).

Supported agent aliases (all route to Ollama):
    @ollama               — direct Ollama mention
    @copilot              — GitHub Copilot alias
    @lucidia              — Lucidia AI alias
    @blackboxprogramming  — BlackBox Programming alias

Configuration (environment variables):
    OLLAMA_BASE_URL   — Ollama server base URL (default: http://localhost:11434)
    OLLAMA_MODEL      — Default model name     (default: llama3)
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional, Tuple

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL:   str = os.environ.get("OLLAMA_MODEL",    "llama3")

# Every @-mention alias that must be routed to Ollama — no external providers.
OLLAMA_AGENTS = frozenset({
    "ollama",
    "copilot",
    "lucidia",
    "blackboxprogramming",
})

_MENTION_RE = re.compile(r"@([\w.]+)", re.IGNORECASE)


class OllamaError(RuntimeError):
    """Raised when Ollama returns an unexpected response."""


class OllamaUnavailableError(OllamaError):
    """Raised when the Ollama server cannot be reached."""


def detect_agent(text: str) -> Optional[str]:
    """
    Return the first @-mention alias that maps to Ollama, or None.

    Matching is case-insensitive; trailing dots are stripped so that
    ``@copilot.`` is treated the same as ``@copilot``.

    Examples::

        detect_agent("@ollama, explain this")          -> "ollama"
        detect_agent("hey @Lucidia help me")           -> "lucidia"
        detect_agent("@blackboxprogramming. do it")    -> "blackboxprogramming"
        detect_agent("no agent mention here")          -> None
    """
    for match in _MENTION_RE.finditer(text):
        alias = match.group(1).lower().rstrip(".")
        if alias in OLLAMA_AGENTS:
            return alias
    return None


def routes_to_ollama(text: str) -> bool:
    """Return True if *text* contains an @-mention that maps to Ollama."""
    return detect_agent(text) is not None


def chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str = "",
    timeout: int = 120,
) -> str:
    """
    Send a chat request to the local Ollama instance and return the reply.

    Args:
        prompt:  The user message.
        model:   Ollama model name (default: OLLAMA_MODEL env var or 'llama3').
        system:  Optional system prompt prepended to the conversation.
        timeout: HTTP request timeout in seconds (default: 120).

    Returns:
        The model's response text.

    Raises:
        OllamaUnavailableError: The Ollama server could not be reached.
        OllamaError:            Ollama returned an unexpected payload.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            return body["message"]["content"]
    except urllib.error.URLError as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}: {exc}"
        ) from exc
    except (KeyError, json.JSONDecodeError) as exc:
        raise OllamaError(f"Unexpected Ollama response format: {exc}") from exc


def route(
    text: str,
    model: str = DEFAULT_MODEL,
    system: str = "",
) -> Tuple[bool, str]:
    """
    Route *text* to Ollama if it contains a known @-mention.

    Returns:
        ``(True, response_text)``  — routed to Ollama.
        ``(False, "")``            — no Ollama agent mention found.
    """
    if routes_to_ollama(text):
        return True, chat(text, model=model, system=system)
    return False, ""
