"""
BlackRoad AI Memory Bridge
Persistent cross-model memory using PS-SHA∞ hash-chain journals
"""
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime


MEMORY_DIR = Path.home() / ".blackroad" / "memory"
JOURNAL_FILE = MEMORY_DIR / "journals" / "master-journal.jsonl"
LEDGER_FILE = MEMORY_DIR / "ledger" / "memory-ledger.jsonl"


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


def remember(key: str, value: Any, model: str = "unknown", ttl_hours: int = 0) -> dict:
    """Store a memory entry with hash-chain integrity."""
    _ensure_dirs()
    prev = _last_hash()
    entry = {
        "hash": _ps_sha(f"{key}:{json.dumps(value)}", prev),
        "prev": prev,
        "key": key,
        "value": value,
        "model": model,
        "type": "fact",
        "truth_state": 1,
        "ttl_hours": ttl_hours,
        "ts": datetime.utcnow().isoformat() + "Z"
    }
    with open(JOURNAL_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def recall(key: str, limit: int = 10) -> list[dict]:
    """Retrieve memory entries by key."""
    if not JOURNAL_FILE.exists():
        return []
    results = []
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("key") == key:
                    results.append(entry)
            except json.JSONDecodeError:
                continue
    return results[-limit:]


def search(query: str, limit: int = 20) -> list[dict]:
    """Full-text search across all memories."""
    if not JOURNAL_FILE.exists():
        return []
    results = []
    q = query.lower()
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line)
                val = json.dumps(entry.get("value", "")).lower()
                if q in entry.get("key", "").lower() or q in val:
                    results.append(entry)
            except json.JSONDecodeError:
                continue
    return results[-limit:]


def observe(content: str, model: str = "unknown") -> dict:
    """Log an observation (perceived fact)."""
    return remember(f"obs:{int(time.time())}", content, model=model)


def infer(content: str, confidence: float = 0.8, model: str = "unknown") -> dict:
    """Log an inference with confidence score."""
    _ensure_dirs()
    prev = _last_hash()
    entry = {
        "hash": _ps_sha(f"infer:{content}", prev),
        "prev": prev,
        "key": f"inf:{int(time.time())}",
        "value": content,
        "model": model,
        "type": "inference",
        "confidence": confidence,
        "truth_state": 1 if confidence >= 0.7 else 0,
        "ts": datetime.utcnow().isoformat() + "Z"
    }
    with open(JOURNAL_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def get_context(n: int = 20) -> str:
    """Get recent memory context as a formatted string for AI prompts."""
    if not JOURNAL_FILE.exists():
        return "No prior context."
    entries = []
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    recent = entries[-n:]
    lines = ["## Recent Memory Context\n"]
    for e in recent:
        ts = e.get("ts", "")[:10]
        t = e.get("type", "fact")
        v = e.get("value", "")
        if isinstance(v, dict):
            v = json.dumps(v)
        lines.append(f"[{ts}] [{t}] {v}")
    return "\n".join(lines)


def verify_chain() -> tuple[bool, str]:
    """Verify the integrity of the hash chain."""
    if not JOURNAL_FILE.exists():
        return True, "Empty journal — chain valid."
    entries = []
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    for i, entry in enumerate(entries[1:], 1):
        if entry["prev"] != entries[i-1]["hash"]:
            return False, f"Chain broken at entry {i}: expected {entries[i-1]['hash'][:8]}, got {entry['prev'][:8]}"
    return True, f"Chain valid — {len(entries)} entries verified."
