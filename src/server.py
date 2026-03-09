"""
BlackRoad AI Memory Bridge — FastAPI server
Provides HTTP API for cross-agent memory access and direct Ollama routing.

When a message contains @copilot, @lucidia, @blackboxprogramming, or @ollama
the request is routed directly to a local Ollama instance — no external
provider is involved.
"""
import os
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memory_bridge import MemoryBridge, MemoryEntry

# ── Ollama routing ─────────────────────────────────────────────────────────────

# All @ mentions that are routed directly to local Ollama.
OLLAMA_AGENTS = {"@copilot", "@lucidia", "@blackboxprogramming", "@ollama"}

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3")


def _detect_agent(message: str) -> Optional[str]:
    """Return the first OLLAMA_AGENTS mention found in *message*, or None."""
    for token in message.split():
        normalised = token.lower().rstrip(".,!?:;\"'")
        if normalised in OLLAMA_AGENTS:
            return normalised
    return None


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BlackRoad AI Memory Bridge",
    description="Persistent memory + direct Ollama routing for BlackRoad AI agents",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bridge = MemoryBridge()


# ── Request / Response models ─────────────────────────────────────────────────

class MemoryWrite(BaseModel):
    agent_id: str = "unknown"
    content: str
    memory_type: str = "episodic"
    importance: float = 0.5
    tags: list = []
    expires_at: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    agent: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "ollama_agents": sorted(OLLAMA_AGENTS)}


@app.post("/memory")
def write_memory(req: MemoryWrite):
    entry = MemoryEntry(
        agent_id=req.agent_id,
        content=req.content,
        memory_type=req.memory_type,
        importance=req.importance,
        tags=req.tags,
        expires_at=req.expires_at,
    )
    result = _bridge.store_memory(entry)
    return {"memory_id": result.memory_id, "agent_id": req.agent_id}


@app.get("/memory")
def list_memories(agent_id: Optional[str] = None,
                  memory_type: Optional[str] = None,
                  limit: int = 50):
    memories = _bridge.list_memories(agent_id=agent_id, mtype=memory_type)
    return [
        {
            "memory_id": m.memory_id,
            "agent_id": m.agent_id,
            "content": m.content,
            "memory_type": m.memory_type,
            "importance": m.importance,
            "access_count": m.access_count,
            "tags": m.tags,
            "expires_at": m.expires_at,
        }
        for m in memories[:limit]
    ]


@app.get("/search")
def search_memory(q: str, agent_id: Optional[str] = None,
                  top_k: int = 10, min_similarity: float = 0.0):
    results = _bridge.search_semantic(
        q, agent_id=agent_id, top_k=top_k, min_similarity=min_similarity
    )
    return [
        {
            "memory_id": r.memory_id,
            "content": r.content,
            "similarity": r.similarity,
            "importance": r.importance,
            "memory_type": r.memory_type,
            "agent_id": r.agent_id,
        }
        for r in results
    ]


@app.get("/context")
def get_context(agent_id: str, query: str = "", max_tokens: int = 512):
    return {"context": _bridge.get_context(agent_id, query, max_tokens)}


@app.post("/consolidate")
def consolidate(agent_id: str):
    report = _bridge.consolidate_memories(agent_id)
    return {
        "agent_id": report.agent_id,
        "memories_scanned": report.memories_scanned,
        "memories_merged": report.memories_merged,
        "memories_pruned": report.memories_pruned,
        "memories_promoted": report.memories_promoted,
        "duration_ms": report.duration_ms,
    }


@app.post("/clear-expired")
def clear_expired():
    n = _bridge.clear_expired()
    return {"deleted": n}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Route a chat message directly to local Ollama.

    Triggers on @copilot, @lucidia, @blackboxprogramming, or @ollama mentions
    in the message body, or when `agent` is one of those values.
    No external AI provider is contacted.
    """
    detected = _detect_agent(req.message)
    agent = req.agent.lower() if req.agent else None
    if agent and not agent.startswith("@"):
        agent = f"@{agent}"

    if not (detected or (agent and agent in OLLAMA_AGENTS)):
        raise HTTPException(
            status_code=400,
            detail=(
                "No recognised @agent mention in message. "
                "Use @ollama, @copilot, @lucidia, or @blackboxprogramming "
                "to route the request to Ollama."
            ),
        )

    model = req.model or OLLAMA_DEFAULT_MODEL
    payload = {"model": model, "prompt": req.message, "stream": False}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama not reachable at {OLLAMA_BASE_URL}. "
                       "Ensure Ollama is running locally.",
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Ollama error: {exc.response.text}",
            )

    data = resp.json()
    return {
        "response": data.get("response", ""),
        "model": data.get("model", model),
        "provider": "ollama",
        "agent": detected or agent,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

