"""
BlackRoad AI Memory Bridge — FastAPI server
Provides HTTP API for cross-model memory access
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional
import uvicorn
import ollama_router
from memory_bridge import remember, recall, search, observe, infer, get_context, verify_chain

app = FastAPI(
    title="BlackRoad AI Memory Bridge",
    description="PS-SHA∞ persistent memory for AI model collaboration",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MemoryWrite(BaseModel):
    key: str
    value: Any
    model: str = "unknown"
    ttl_hours: int = 0

class ObserveRequest(BaseModel):
    content: str
    model: str = "unknown"

class InferRequest(BaseModel):
    content: str
    confidence: float = 0.8
    model: str = "unknown"


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    system: Optional[str] = ""


@app.get("/health")
def health():
    valid, msg = verify_chain()
    return {"status": "ok", "chain_valid": valid, "chain_msg": msg}

@app.post("/memory")
def write_memory(req: MemoryWrite):
    entry = remember(req.key, req.value, req.model, req.ttl_hours)
    return {"hash": entry["hash"], "key": req.key}

@app.get("/memory/{key}")
def read_memory(key: str, limit: int = 10):
    entries = recall(key, limit)
    if not entries:
        raise HTTPException(404, "No memories found for key")
    return entries

@app.get("/search")
def search_memory(q: str, limit: int = 20):
    return search(q, limit)

@app.post("/observe")
def log_observation(req: ObserveRequest):
    return observe(req.content, req.model)

@app.post("/infer")
def log_inference(req: InferRequest):
    return infer(req.content, req.confidence, req.model)

@app.get("/context")
def get_prompt_context(n: int = 20):
    return {"context": get_context(n)}

@app.get("/verify")
def verify():
    valid, msg = verify_chain()
    return {"valid": valid, "message": msg}


@app.post("/chat")
def chat_with_agent(req: ChatRequest):
    """
    Route a message to the local Ollama instance.

    The message must contain one of the supported @-mentions:
    ``@ollama``, ``@copilot``, ``@lucidia``, or ``@blackboxprogramming``.
    All requests are sent directly to Ollama — no external AI providers are used.
    """
    model = req.model or ollama_router.DEFAULT_MODEL
    routed, response = ollama_router.route(
        req.message, model=model, system=req.system or ""
    )
    if not routed:
        raise HTTPException(
            status_code=400,
            detail=(
                "No Ollama agent @-mention found. "
                "Use @ollama, @copilot, @lucidia, or @blackboxprogramming."
            ),
        )
    return {"response": response, "routed_to": "ollama", "model": model}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
