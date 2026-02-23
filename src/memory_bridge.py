"""
BlackRoad AI Memory Bridge — Vector store interface, semantic search,
and memory consolidation for AI agents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── ANSI colours ─────────────────────────────────────────────────────────────
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; M = "\033[0;35m"; NC = "\033[0m"
BOLD = "\033[1m"

DB_PATH = Path.home() / ".blackroad" / "memory_bridge.db"
EMBED_DIM = 64   # Simulated embedding dimension


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class MemoryEntry:
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    content: str = ""
    memory_type: str = "episodic"   # episodic, semantic, working, procedural
    importance: float = 0.5
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed: Optional[str] = None


@dataclass
class VectorEmbedding:
    embed_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    memory_id: str = ""
    vector: List[float] = field(default_factory=list)
    model: str = "simulated-minilm"
    dim: int = EMBED_DIM
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SearchResult:
    memory_id: str
    content: str
    agent_id: str
    memory_type: str
    similarity: float
    importance: float
    created_at: str


@dataclass
class ConsolidationReport:
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    memories_scanned: int = 0
    memories_merged: int = 0
    memories_pruned: int = 0
    memories_promoted: int = 0
    duration_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── helpers ───────────────────────────────────────────────────────────────────
def _embed(text: str) -> List[float]:
    """Deterministic pseudo-embedding via hashing (no torch required at runtime)."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(EMBED_DIM)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return round(max(-1.0, min(1.0, dot)), 6)


# ── Core class ────────────────────────────────────────────────────────────────
class MemoryBridge:
    """Memory bridge: store, search, and consolidate agent memories."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_db()
        # In-memory vector index: memory_id → vector
        self._vectors: Dict[str, List[float]] = {}
        self._load_vectors()

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                agent_id TEXT, content TEXT,
                memory_type TEXT, importance REAL,
                access_count INTEGER DEFAULT 0,
                tags_json TEXT,
                expires_at TEXT,
                created_at TEXT,
                last_accessed TEXT
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                embed_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                vector_json TEXT,
                model TEXT,
                dim INTEGER,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS consolidation_reports (
                report_id TEXT PRIMARY KEY,
                agent_id TEXT,
                memories_scanned INTEGER,
                memories_merged INTEGER,
                memories_pruned INTEGER,
                memories_promoted INTEGER,
                duration_ms REAL,
                created_at TEXT
            );
        """)
        self._conn.commit()

    def _load_vectors(self) -> None:
        rows = self._conn.execute(
            "SELECT memory_id, vector_json FROM embeddings"
        ).fetchall()
        for mid, vjson in rows:
            self._vectors[mid] = json.loads(vjson or "[]")

    def store_memory(self, entry: MemoryEntry) -> MemoryEntry:
        """Persist a memory and compute its embedding."""
        vec = _embed(entry.content)
        self._conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?)",
            (entry.memory_id, entry.agent_id, entry.content,
             entry.memory_type, entry.importance, entry.access_count,
             json.dumps(entry.tags), entry.expires_at,
             entry.created_at, entry.last_accessed)
        )
        embed = VectorEmbedding(memory_id=entry.memory_id, vector=vec, dim=EMBED_DIM)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
            (embed.embed_id, embed.memory_id, json.dumps(embed.vector),
             embed.model, embed.dim, embed.created_at)
        )
        self._conn.commit()
        self._vectors[entry.memory_id] = vec
        type_color = {
            "episodic": C, "semantic": G, "working": Y, "procedural": M
        }.get(entry.memory_type, NC)
        print(f"{G}✓{NC} Memory stored [{type_color}{entry.memory_type}{NC}] "
              f"{C}{entry.memory_id}{NC} importance={entry.importance:.2f}")
        return entry

    def search_semantic(
        self, query: str, agent_id: Optional[str] = None,
        top_k: int = 5, min_similarity: float = 0.1
    ) -> List[SearchResult]:
        """Find memories semantically similar to the query."""
        qvec = _embed(query)
        rows = self._conn.execute(
            "SELECT memory_id, content, agent_id, memory_type, importance, created_at "
            "FROM memories" + (" WHERE agent_id=?" if agent_id else ""),
            (agent_id,) if agent_id else ()
        ).fetchall()
        results = []
        for mid, content, aid, mtype, imp, cat in rows:
            vec = self._vectors.get(mid, [])
            if not vec:
                continue
            sim = _cosine(qvec, vec)
            if sim >= min_similarity:
                results.append(SearchResult(
                    memory_id=mid, content=content, agent_id=aid,
                    memory_type=mtype, similarity=sim,
                    importance=imp, created_at=cat
                ))
        results.sort(key=lambda x: x.similarity * x.importance, reverse=True)
        results = results[:top_k]
        # Update access counts
        for r in results:
            self._conn.execute(
                "UPDATE memories SET access_count=access_count+1, last_accessed=? "
                "WHERE memory_id=?",
                (datetime.utcnow().isoformat(), r.memory_id)
            )
        self._conn.commit()
        return results

    def consolidate_memories(self, agent_id: str, similarity_threshold: float = 0.9) -> ConsolidationReport:
        """Merge highly similar memories, prune expired, promote important ones."""
        t0 = time.perf_counter()
        report = ConsolidationReport(agent_id=agent_id)
        now = datetime.utcnow()

        # Prune expired
        expired = self._conn.execute(
            "SELECT memory_id FROM memories WHERE agent_id=? AND expires_at IS NOT NULL "
            "AND expires_at < ?",
            (agent_id, now.isoformat())
        ).fetchall()
        for (mid,) in expired:
            self._conn.execute("DELETE FROM memories WHERE memory_id=?", (mid,))
            self._conn.execute("DELETE FROM embeddings WHERE memory_id=?", (mid,))
            self._vectors.pop(mid, None)
            report.memories_pruned += 1

        # Promote high-access memories to semantic
        promoted = self._conn.execute(
            "SELECT memory_id FROM memories WHERE agent_id=? AND memory_type='episodic' "
            "AND access_count >= 5",
            (agent_id,)
        ).fetchall()
        for (mid,) in promoted:
            self._conn.execute(
                "UPDATE memories SET memory_type='semantic', importance=MIN(importance+0.1,1.0) "
                "WHERE memory_id=?", (mid,)
            )
            report.memories_promoted += 1

        # Merge near-duplicates
        mids = [r[0] for r in self._conn.execute(
            "SELECT memory_id FROM memories WHERE agent_id=?", (agent_id,)
        ).fetchall()]
        report.memories_scanned = len(mids)
        merged_set = set()
        for i, m1 in enumerate(mids):
            if m1 in merged_set:
                continue
            v1 = self._vectors.get(m1, [])
            for m2 in mids[i+1:]:
                if m2 in merged_set:
                    continue
                v2 = self._vectors.get(m2, [])
                if v1 and v2 and _cosine(v1, v2) >= similarity_threshold:
                    self._conn.execute("DELETE FROM memories WHERE memory_id=?", (m2,))
                    self._conn.execute("DELETE FROM embeddings WHERE memory_id=?", (m2,))
                    self._vectors.pop(m2, None)
                    merged_set.add(m2)
                    report.memories_merged += 1

        self._conn.commit()
        report.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._conn.execute(
            "INSERT INTO consolidation_reports VALUES (?,?,?,?,?,?,?,?)",
            (report.report_id, report.agent_id, report.memories_scanned,
             report.memories_merged, report.memories_pruned,
             report.memories_promoted, report.duration_ms, report.created_at)
        )
        self._conn.commit()
        print(f"{G}✓{NC} Consolidation [{C}{agent_id}{NC}]: "
              f"scanned={report.memories_scanned} "
              f"merged={Y}{report.memories_merged}{NC} "
              f"pruned={R}{report.memories_pruned}{NC} "
              f"promoted={G}{report.memories_promoted}{NC} "
              f"({report.duration_ms}ms)")
        return report

    def get_context(self, agent_id: str, query: str, max_tokens: int = 512) -> str:
        """Build a context string from the most relevant memories."""
        results = self.search_semantic(query, agent_id=agent_id, top_k=10)
        context_parts = []
        total = 0
        for r in results:
            tokens = len(r.content.split())
            if total + tokens > max_tokens:
                break
            context_parts.append(f"[{r.memory_type}|{r.similarity:.2f}] {r.content}")
            total += tokens
        return "\n".join(context_parts)

    def clear_expired(self) -> int:
        """Remove all expired memories across all agents."""
        now = datetime.utcnow().isoformat()
        expired = self._conn.execute(
            "SELECT memory_id FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,)
        ).fetchall()
        for (mid,) in expired:
            self._conn.execute("DELETE FROM memories WHERE memory_id=?", (mid,))
            self._conn.execute("DELETE FROM embeddings WHERE memory_id=?", (mid,))
            self._vectors.pop(mid, None)
        self._conn.commit()
        return len(expired)

    def list_memories(self, agent_id: Optional[str] = None, mtype: Optional[str] = None) -> List[MemoryEntry]:
        q = "SELECT * FROM memories WHERE 1=1"
        p = []
        if agent_id:
            q += " AND agent_id=?"
            p.append(agent_id)
        if mtype:
            q += " AND memory_type=?"
            p.append(mtype)
        q += " ORDER BY importance DESC, created_at DESC LIMIT 50"
        rows = self._conn.execute(q, p).fetchall()
        return [MemoryEntry(memory_id=r[0], agent_id=r[1], content=r[2],
                            memory_type=r[3], importance=r[4],
                            access_count=r[5], tags=json.loads(r[6] or "[]"),
                            expires_at=r[7], created_at=r[8], last_accessed=r[9])
                for r in rows]

    def close(self) -> None:
        self._conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="memory-bridge",
        description="BlackRoad AI Memory Bridge — semantic memory for agents"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("store", help="Store a memory")
    st.add_argument("--agent-id", required=True)
    st.add_argument("--content", required=True)
    st.add_argument("--type", choices=["episodic", "semantic", "working", "procedural"],
                    default="episodic")
    st.add_argument("--importance", type=float, default=0.5)
    st.add_argument("--tags", nargs="*", default=[])
    st.add_argument("--ttl", type=int, default=0, help="Expiry in seconds (0=never)")

    srch = sub.add_parser("search", help="Semantic search")
    srch.add_argument("--query", required=True)
    srch.add_argument("--agent-id", default=None)
    srch.add_argument("--top-k", type=int, default=5)
    srch.add_argument("--min-sim", type=float, default=0.05)

    ctx = sub.add_parser("context", help="Build context string")
    ctx.add_argument("--agent-id", required=True)
    ctx.add_argument("--query", required=True)
    ctx.add_argument("--max-tokens", type=int, default=512)

    con = sub.add_parser("consolidate", help="Consolidate memories")
    con.add_argument("--agent-id", required=True)
    con.add_argument("--threshold", type=float, default=0.9)

    lst = sub.add_parser("list", help="List memories")
    lst.add_argument("--agent-id", default=None)
    lst.add_argument("--type", default=None)

    sub.add_parser("clear-expired", help="Remove expired memories")

    args = parser.parse_args()
    bridge = MemoryBridge()

    try:
        if args.cmd == "store":
            expires_at = None
            if args.ttl > 0:
                expires_at = (datetime.utcnow() + timedelta(seconds=args.ttl)).isoformat()
            entry = MemoryEntry(
                agent_id=args.agent_id, content=args.content,
                memory_type=args.type, importance=args.importance,
                tags=args.tags, expires_at=expires_at,
            )
            bridge.store_memory(entry)

        elif args.cmd == "search":
            results = bridge.search_semantic(
                args.query, agent_id=args.agent_id,
                top_k=args.top_k, min_similarity=args.min_sim
            )
            if not results:
                print(f"{Y}No results for: {args.query}{NC}")
            else:
                print(f"\n{BOLD}{B}── Search: {args.query[:50]} ──────────{NC}")
                for i, r in enumerate(results, 1):
                    bar = "█" * int(r.similarity * 10) + "░" * (10 - int(r.similarity * 10))
                    print(f"  {i}. {C}{r.memory_id}{NC} sim={Y}{r.similarity:.3f}{NC} "
                          f"{C}{bar}{NC}")
                    print(f"     {r.content[:100]}")

        elif args.cmd == "context":
            ctx_str = bridge.get_context(args.agent_id, args.query, args.max_tokens)
            print(f"\n{BOLD}{B}── Context for {args.agent_id} ──────────{NC}")
            print(ctx_str or f"{Y}(empty){NC}")

        elif args.cmd == "consolidate":
            bridge.consolidate_memories(args.agent_id, args.threshold)

        elif args.cmd == "list":
            memories = bridge.list_memories(agent_id=args.agent_id, mtype=args.type)
            if not memories:
                print(f"{Y}No memories found.{NC}")
                return
            for m in memories:
                type_color = {
                    "episodic": C, "semantic": G, "working": Y, "procedural": M
                }.get(m.memory_type, NC)
                print(f"  {C}{m.memory_id}{NC} [{type_color}{m.memory_type}{NC}] "
                      f"imp={m.importance:.2f} acc={m.access_count} "
                      f"— {m.content[:60]}")

        elif args.cmd == "clear-expired":
            n = bridge.clear_expired()
            print(f"{G}✓{NC} Cleared {n} expired memories.")

    finally:
        bridge.close()


if __name__ == "__main__":
    main()
