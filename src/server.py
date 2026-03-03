"""
BlackRoad AI Memory Bridge — FastAPI server
Provides HTTP API for cross-model memory access.

Authentication: set BLACKROAD_API_KEY env var and pass the value
in the X-API-Key header on every request.
"""
import os
import secrets
import sys
from pathlib import Path
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
from memory_bridge import MemoryBridge, MemoryEntry

# ── Auth ──────────────────────────────────────────────────────────────────────
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_CONFIGURED_KEY: str = os.environ.get("BLACKROAD_API_KEY", "").strip()


async def _require_api_key(api_key: str = Security(_API_KEY_HEADER)) -> None:
    """Reject requests that don't supply the correct API key.

    When BLACKROAD_API_KEY is not set the server runs unauthenticated
    (development mode only — always set the key in production).
    Comparison uses secrets.compare_digest to prevent timing attacks.
    """
    if not _CONFIGURED_KEY:
        return  # dev mode — no key configured
    if not api_key or not secrets.compare_digest(api_key, _CONFIGURED_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


_auth = [Depends(_require_api_key)]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BlackRoad AI Memory Bridge",
    description="PS-SHA∞ persistent memory for AI model collaboration",
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
    key: str
    value: Any
    agent_id: str = "default"
    memory_type: str = "episodic"
    importance: float = 0.5
    ttl_hours: int = 0


class SearchRequest(BaseModel):
    query: str
    agent_id: Optional[str] = None
    top_k: int = 10
    min_similarity: float = 0.0


class ContextRequest(BaseModel):
    agent_id: str
    query: str = ""
    max_tokens: int = 512


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/memory", dependencies=_auth)
def write_memory(req: MemoryWrite):
    from datetime import datetime, timedelta
    expires_at = None
    if req.ttl_hours:
        expires_at = (datetime.utcnow() + timedelta(hours=req.ttl_hours)).isoformat()
    entry = MemoryEntry(
        agent_id=req.agent_id,
        content=str(req.value),
        memory_type=req.memory_type,
        importance=req.importance,
        expires_at=expires_at,
    )
    result = _bridge.store_memory(entry)
    return {"memory_id": result.memory_id, "key": req.key}


@app.get("/memory/{agent_id}", dependencies=_auth)
def read_memory(agent_id: str, mtype: Optional[str] = None, limit: int = 10):
    entries = _bridge.list_memories(agent_id=agent_id, mtype=mtype)[:limit]
    if not entries:
        raise HTTPException(404, "No memories found for agent")
    return [
        {
            "memory_id": e.memory_id,
            "content": e.content,
            "memory_type": e.memory_type,
            "importance": e.importance,
            "access_count": e.access_count,
        }
        for e in entries
    ]


@app.get("/search", dependencies=_auth)
def search_memory(q: str, agent_id: Optional[str] = None, limit: int = 10):
    results = _bridge.search_semantic(
        q, agent_id=agent_id, top_k=limit, min_similarity=0.0
    )
    return [
        {
            "memory_id": r.memory_id,
            "content": r.content,
            "similarity": r.similarity,
            "memory_type": r.memory_type,
            "agent_id": r.agent_id,
        }
        for r in results
    ]


@app.get("/context", dependencies=_auth)
def get_prompt_context(agent_id: str, query: str = "", max_tokens: int = 512):
    ctx = _bridge.get_context(agent_id, query, max_tokens=max_tokens)
    return {"context": ctx}


@app.post("/consolidate", dependencies=_auth)
def consolidate(agent_id: str, threshold: float = 0.9):
    report = _bridge.consolidate_memories(agent_id, similarity_threshold=threshold)
    return {
        "agent_id": report.agent_id,
        "memories_scanned": report.memories_scanned,
        "memories_merged": report.memories_merged,
        "memories_pruned": report.memories_pruned,
        "memories_promoted": report.memories_promoted,
        "duration_ms": report.duration_ms,
    }


@app.delete("/expired", dependencies=_auth)
def clear_expired():
    n = _bridge.clear_expired()
    return {"cleared": n}


@app.get("/verify")
def verify():
    return {"valid": True, "message": "BlackRoad Memory Bridge operational"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
