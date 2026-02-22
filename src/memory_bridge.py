"""
BlackRoad AI Memory Bridge
Persistent cross-model memory using PS-SHA∞ hash-chain journals
"""
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, List
from datetime import datetime


MEMORY_DIR = Path.home() / ".blackroad" / "memory"
JOURNAL_FILE = MEMORY_DIR / "journals" / "master-journal.jsonl"
LEDGER_FILE = MEMORY_DIR / "ledger" / "memory-ledger.jsonl"
VECTOR_INDEX_FILE = MEMORY_DIR / "vector-index.jsonl"


def _ensure_dirs():
    for d in [MEMORY_DIR / "journals", MEMORY_DIR / "ledger",
              MEMORY_DIR / "sessions", MEMORY_DIR / "context"]:
        d.mkdir(parents=True, exist_ok=True)


def _ps_sha(content: str, prev_hash: str = "") -> str:
    """PS-SHA∞: Persistent State SHA — hash-chain entry."""
    payload = f"{prev_hash}:{content}:{time.time_ns()}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _last_hash() -> str:
    """Get the last hash in the journal for chain continuity."""
    if not JOURNAL_FILE.exists():
        return "GENESIS"
    with open(JOURNAL_FILE) as f:
        lines = f.readlines()
    if not lines:
        return "GENESIS"
    try:
        return json.loads(lines[-1])["hash"]
    except Exception:
        return "GENESIS"


def remember(content: str, memory_type: str = "fact",
             truth_state: int = 1, metadata: dict = None) -> dict:
    """Store a memory in the PS-SHA∞ hash-chain journal."""
    _ensure_dirs()
    prev = _last_hash()
    h = _ps_sha(content, prev)
    entry = {
        "hash": h,
        "prev": prev,
        "content": content,
        "type": memory_type,
        "truth_state": truth_state,  # 1=true, 0=unknown, -1=false
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat(),
        "timestamp_ns": time.time_ns(),
    }
    with open(JOURNAL_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def recall(query: str, limit: int = 10, memory_type: str = None) -> List[dict]:
    """Recall memories matching query (keyword search)."""
    if not JOURNAL_FILE.exists():
        return []
    results = []
    query_lower = query.lower()
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry["truth_state"] == -1:
                    continue  # Skip negated memories
                if memory_type and entry.get("type") != memory_type:
                    continue
                if query_lower in entry["content"].lower():
                    results.append(entry)
            except Exception:
                continue
    return results[-limit:]


def recall_recent(limit: int = 20) -> List[dict]:
    """Recall the most recent memories."""
    if not JOURNAL_FILE.exists():
        return []
    results = []
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry["truth_state"] != -1:
                    results.append(entry)
            except Exception:
                continue
    return results[-limit:]


def negate(hash_id: str) -> bool:
    """Mark a memory as false (truth_state = -1)."""
    if not JOURNAL_FILE.exists():
        return False
    lines = []
    found = False
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry["hash"] == hash_id:
                    entry["truth_state"] = -1
                    found = True
                lines.append(json.dumps(entry))
            except Exception:
                lines.append(line.strip())
    if found:
        with open(JOURNAL_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")
    return found


def verify_chain() -> bool:
    """Verify the integrity of the PS-SHA∞ hash chain."""
    if not JOURNAL_FILE.exists():
        return True
    prev = "GENESIS"
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry["prev"] != prev:
                    return False
                prev = entry["hash"]
            except Exception:
                return False
    return True


def export_context(session_id: str = "default", limit: int = 50) -> str:
    """Export recent memories as a context string for AI prompts."""
    entries = recall_recent(limit)
    if not entries:
        return "No memories stored yet."
    lines = [f"## BlackRoad Memory Context (Session: {session_id})", ""]
    for e in entries:
        type_icon = {"fact": "📌", "observation": "👁", "inference": "💭",
                     "commitment": "🤝"}.get(e["type"], "•")
        ts = e.get("timestamp", "")[:10]
        lines.append(f"{type_icon} [{ts}] {e['content']}")
    return "\n".join(lines)


def sync_to_remote(gateway_url: str = "http://127.0.0.1:8787") -> dict:
    """Sync local memory journal to the BlackRoad gateway."""
    import urllib.request
    entries = recall_recent(100)
    payload = json.dumps({"memories": entries}).encode()
    req = urllib.request.Request(
        f"{gateway_url}/memory/sync",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Demo
    entry = remember("BlackRoad gateway is running at 127.0.0.1:8787", "fact")
    remember("User prefers dark mode", "observation", metadata={"source": "ui"})
    remember("Pi fleet has 4 devices", "fact")
    print("Stored 3 memories")
    print("Chain valid:", verify_chain())
    results = recall("gateway")
    print(f"Recall 'gateway': {len(results)} results")
    print("\nContext preview:")
    print(export_context()[:500])
