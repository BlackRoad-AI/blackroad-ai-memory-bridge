"""
BlackRoad AI Memory Bridge — vector store, semantic search, and memory
consolidation for AI agents (episodic, semantic, working, procedural).

Usage:
    python memory_bridge.py store --agent-id lucidia --type semantic \
        --content "User prefers dark mode" --importance 0.8
    python memory_bridge.py search --query "user preferences" --agent-id lucidia
    python memory_bridge.py context --agent-id lucidia --query "preferences"
    python memory_bridge.py consolidate --agent-id lucidia
    python memory_bridge.py list --agent-id lucidia --type semantic
    python memory_bridge.py clear-expired
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── ANSI colours ─────────────────────────────────────────────────────────────
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; M = "\033[0;35m"; NC = "\033[0m"
BOLD = "\033[1m"

EMBED_DIM = 64
MEMORY_TYPES = ("episodic", "semantic", "working", "procedural")
DB_PATH = Path(os.environ.get(
    "BLACKROAD_MEMORY_DB",
    str(Path.home() / ".blackroad" / "memory_bridge.db"),
))

# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed(text: str) -> List[float]:
    """Deterministic 64-dim unit vector derived from SHA-256 hash of text."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(EMBED_DIM)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(v1: List[float], v2: List[float]) -> float:
    """Cosine similarity between two vectors (pre-normalised → plain dot product)."""
    return round(sum(a * b for a, b in zip(v1, v2)), 8)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory unit stored in the bridge."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    content: str = ""
    memory_type: str = "episodic"   # episodic | semantic | working | procedural
    importance: float = 0.5         # 0.0 – 1.0
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[str] = None  # ISO-8601 datetime or None (permanent)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(
                f"memory_type must be one of {MEMORY_TYPES}, got {self.memory_type!r}"
            )
        self.importance = max(0.0, min(1.0, self.importance))


@dataclass
class VectorEmbedding:
    """Stored vector embedding for a memory entry."""
    embed_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    memory_id: str = ""
    vector: List[float] = field(default_factory=list)
    model: str = "sha256-gauss"
    dim: int = EMBED_DIM
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SearchResult:
    """Result of a semantic memory search."""
    memory_id: str
    content: str
    similarity: float
    importance: float
    memory_type: str
    agent_id: str


@dataclass
class ConsolidationReport:
    """Report produced by MemoryBridge.consolidate_memories()."""
    agent_id: str
    memories_scanned: int = 0
    memories_merged: int = 0
    memories_pruned: int = 0
    memories_promoted: int = 0
    duration_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Core class ────────────────────────────────────────────────────────────────

class MemoryBridge:
    """
    BlackRoad AI Memory Bridge.

    Persistent, agent-scoped memory with SHA-256 vector embeddings,
    cosine-similarity search, TTL expiry, and consolidation pipeline.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._vectors: Dict[str, List[float]] = {}
        self._init_schema()
        self._load_vectors()

    # ── Schema ────────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id    TEXT PRIMARY KEY,
                agent_id     TEXT NOT NULL DEFAULT '',
                content      TEXT NOT NULL DEFAULT '',
                memory_type  TEXT NOT NULL DEFAULT 'episodic',
                importance   REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                tags_json    TEXT NOT NULL DEFAULT '[]',
                expires_at   TEXT,
                created_at   TEXT NOT NULL,
                last_accessed TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                embed_id    TEXT PRIMARY KEY,
                memory_id   TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                model       TEXT NOT NULL DEFAULT 'sha256-gauss',
                dim         INTEGER NOT NULL DEFAULT 64,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS consolidation_reports (
                report_id          TEXT PRIMARY KEY,
                agent_id           TEXT,
                memories_scanned   INTEGER,
                memories_merged    INTEGER,
                memories_pruned    INTEGER,
                memories_promoted  INTEGER,
                duration_ms        REAL,
                created_at         TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_mem_type  ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_emb_mid   ON embeddings(memory_id);
        """)
        self._conn.commit()

    def _load_vectors(self) -> None:
        rows = self._conn.execute(
            "SELECT memory_id, vector_json FROM embeddings"
        ).fetchall()
        for r in rows:
            self._vectors[r["memory_id"]] = json.loads(r["vector_json"])

    def _row_to_entry(self, row) -> MemoryEntry:
        return MemoryEntry(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            access_count=row["access_count"],
            tags=json.loads(row["tags_json"] or "[]"),
            expires_at=row["expires_at"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
        )

    # ── Public API ────────────────────────────────────────────────────────────
    def store_memory(self, entry: MemoryEntry) -> MemoryEntry:
        """Store a memory and compute its embedding."""
        vec = _embed(entry.content)
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?)",
            (entry.memory_id, entry.agent_id, entry.content, entry.memory_type,
             entry.importance, entry.access_count,
             json.dumps(entry.tags), entry.expires_at,
             entry.created_at or now, entry.last_accessed or now)
        )
        embed_id = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
            (embed_id, entry.memory_id, json.dumps(vec),
             "sha256-gauss", EMBED_DIM, now)
        )
        self._conn.commit()
        self._vectors[entry.memory_id] = vec
        return entry

    def list_memories(self, agent_id: Optional[str] = None,
                      mtype: Optional[str] = None) -> List[MemoryEntry]:
        """List memories with optional filters, sorted by importance descending."""
        q = "SELECT * FROM memories"
        params: List = []
        conditions = []
        if agent_id is not None:
            conditions.append("agent_id=?")
            params.append(agent_id)
        if mtype is not None:
            conditions.append("memory_type=?")
            params.append(mtype)
        if conditions:
            q += " WHERE " + " AND ".join(conditions)
        q += " ORDER BY importance DESC"
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def search_semantic(self, query: str, agent_id: Optional[str] = None,
                        top_k: int = 5,
                        min_similarity: float = 0.0) -> List[SearchResult]:
        """Return top-k memories ranked by cosine similarity to query."""
        query_vec = _embed(query)
        memories = self.list_memories(agent_id=agent_id)
        results = []
        for entry in memories:
            vec = self._vectors.get(entry.memory_id)
            if vec is None:
                continue
            sim = _cosine(query_vec, vec)
            if sim >= min_similarity:
                results.append(SearchResult(
                    memory_id=entry.memory_id,
                    content=entry.content,
                    similarity=sim,
                    importance=entry.importance,
                    memory_type=entry.memory_type,
                    agent_id=entry.agent_id,
                ))
        results.sort(key=lambda r: r.similarity, reverse=True)
        results = results[:top_k]
        # Increment access_count for returned results
        now = datetime.utcnow().isoformat()
        for r in results:
            self._conn.execute(
                "UPDATE memories SET access_count=access_count+1, last_accessed=? "
                "WHERE memory_id=?",
                (now, r.memory_id)
            )
        self._conn.commit()
        return results

    def get_context(self, agent_id: str, query: str,
                    max_tokens: int = 512) -> str:
        """Build a token-limited context string from the most relevant memories."""
        results = self.search_semantic(query, agent_id=agent_id,
                                       top_k=20, min_similarity=0.0)
        if not results:
            return ""
        lines = []
        total = 0
        for r in results:
            line = f"[{r.memory_type}|{r.similarity:.3f}] {r.content}"
            tokens = len(line.split())
            if total + tokens > max_tokens:
                break
            lines.append(line)
            total += tokens
        return "\n".join(lines)

    def clear_expired(self) -> int:
        """Delete memories whose expires_at is in the past. Returns count deleted."""
        now = datetime.utcnow().isoformat()
        rows = self._conn.execute(
            "SELECT memory_id FROM memories "
            "WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,)
        ).fetchall()
        ids = [r["memory_id"] for r in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM memories WHERE memory_id IN ({placeholders})", ids
            )
            for mid in ids:
                self._vectors.pop(mid, None)
            self._conn.commit()
        return len(ids)

    def consolidate_memories(self, agent_id: str,
                              similarity_threshold: float = 0.9) -> ConsolidationReport:
        """Prune expired memories, merge near-duplicates, promote high-access episodic."""
        start = time.time()
        report = ConsolidationReport(agent_id=agent_id)
        report.memories_scanned = len(self.list_memories(agent_id=agent_id))

        # 1. Prune expired
        report.memories_pruned = self.clear_expired()

        # 2. Merge near-duplicates
        memories = self.list_memories(agent_id=agent_id)
        merged_ids: set = set()
        vecs = [(m, self._vectors.get(m.memory_id)) for m in memories]
        for i, (ma, va) in enumerate(vecs):
            if ma.memory_id in merged_ids or va is None:
                continue
            for j, (mb, vb) in enumerate(vecs):
                if i >= j or mb.memory_id in merged_ids or vb is None:
                    continue
                if _cosine(va, vb) >= similarity_threshold:
                    to_remove = mb if ma.importance >= mb.importance else ma
                    self._conn.execute(
                        "DELETE FROM memories WHERE memory_id=?",
                        (to_remove.memory_id,)
                    )
                    self._vectors.pop(to_remove.memory_id, None)
                    merged_ids.add(to_remove.memory_id)
                    report.memories_merged += 1
        if merged_ids:
            self._conn.commit()

        # 3. Promote high-access episodic → semantic (threshold: 3 accesses)
        for m in self.list_memories(agent_id=agent_id):
            row = self._conn.execute(
                "SELECT access_count FROM memories WHERE memory_id=?",
                (m.memory_id,)
            ).fetchone()
            if row and row["access_count"] >= 3 and m.memory_type == "episodic":
                self._conn.execute(
                    "UPDATE memories SET memory_type='semantic' WHERE memory_id=?",
                    (m.memory_id,)
                )
                report.memories_promoted += 1
        if report.memories_promoted:
            self._conn.commit()

        report.duration_ms = round((time.time() - start) * 1000, 2)
        self._conn.execute(
            "INSERT INTO consolidation_reports VALUES (?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4())[:8], agent_id, report.memories_scanned,
             report.memories_merged, report.memories_pruned,
             report.memories_promoted, report.duration_ms, report.created_at)
        )
        self._conn.commit()
        return report

    def close(self) -> None:
        self._conn.close()


# ── Pretty printing ───────────────────────────────────────────────────────────
_TYPE_COLOR = {t: c for t, c in zip(MEMORY_TYPES, [C, G, Y, M])}


def _print_entry(e: MemoryEntry, score: Optional[float] = None) -> None:
    tc = _TYPE_COLOR.get(e.memory_type, NC)
    score_str = f"  sim={Y}{score:.4f}{NC}" if score is not None else ""
    tags_str = f"  {B}[{', '.join(e.tags)}]{NC}" if e.tags else ""
    print(f"  {C}{e.memory_id}{NC} [{tc}{e.memory_type}{NC}] "
          f"imp={e.importance:.2f}{score_str}{tags_str}")
    print(f"     {e.content[:120]}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory-bridge",
        description="BlackRoad AI Memory Bridge — SHA-256 vector semantic memory store",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # store
    st = sub.add_parser("store", help="Store a new memory")
    st.add_argument("--agent-id", required=True)
    st.add_argument("--content", required=True)
    st.add_argument("--type", choices=list(MEMORY_TYPES), default="episodic",
                    dest="memory_type")
    st.add_argument("--importance", type=float, default=0.5)
    st.add_argument("--ttl", type=int, default=0, metavar="SECONDS",
                    help="TTL in seconds; 0 = permanent")
    st.add_argument("--tags", nargs="*", default=[])

    # search
    sr = sub.add_parser("search", help="Semantic search")
    sr.add_argument("--query", required=True)
    sr.add_argument("--agent-id", default=None)
    sr.add_argument("--top-k", type=int, default=5)
    sr.add_argument("--min-sim", type=float, default=0.0)

    # context
    cx = sub.add_parser("context", help="Build LLM context string")
    cx.add_argument("--agent-id", required=True)
    cx.add_argument("--query", required=True)
    cx.add_argument("--max-tokens", type=int, default=512)

    # consolidate
    co = sub.add_parser("consolidate", help="Consolidate memories")
    co.add_argument("--agent-id", required=True)
    co.add_argument("--threshold", type=float, default=0.9)

    # list
    ls = sub.add_parser("list", help="List memories")
    ls.add_argument("--agent-id", default=None)
    ls.add_argument("--type", default=None, dest="memory_type")

    # clear-expired
    sub.add_parser("clear-expired", help="Delete all expired memories")

    return p


def main() -> None:
    args = _build_parser().parse_args()
    bridge = MemoryBridge()

    try:
        if args.cmd == "store":
            expires_at = None
            if args.ttl:
                expires_at = (
                    datetime.utcnow() + timedelta(seconds=args.ttl)
                ).isoformat()
            entry = MemoryEntry(
                agent_id=args.agent_id,
                content=args.content,
                memory_type=args.memory_type,
                importance=args.importance,
                tags=args.tags,
                expires_at=expires_at,
            )
            result = bridge.store_memory(entry)
            tc = _TYPE_COLOR.get(result.memory_type, NC)
            print(f"{G}✓{NC} Memory stored [{tc}{result.memory_type}{NC}] "
                  f"{C}{result.memory_id}{NC} importance={result.importance:.2f}")

        elif args.cmd == "search":
            results = bridge.search_semantic(
                args.query, agent_id=args.agent_id,
                top_k=args.top_k, min_similarity=args.min_sim,
            )
            if not results:
                print(f"{Y}No results for: {args.query!r}{NC}")
            else:
                print(f"\n{BOLD}{B}── Search: {args.query[:60]} ─────{NC}")
                for r in results:
                    e = MemoryEntry(
                        memory_id=r.memory_id, agent_id=r.agent_id,
                        content=r.content, memory_type=r.memory_type,
                        importance=r.importance,
                    )
                    _print_entry(e, score=r.similarity)

        elif args.cmd == "context":
            ctx = bridge.get_context(args.agent_id, args.query,
                                     max_tokens=args.max_tokens)
            print(f"\n{BOLD}{B}── Context for {args.agent_id} ──────────{NC}")
            print(ctx or f"{Y}(no memories){NC}")

        elif args.cmd == "consolidate":
            report = bridge.consolidate_memories(args.agent_id, args.threshold)
            print(f"{G}✓{NC} Consolidation [{C}{report.agent_id}{NC}]: "
                  f"scanned={report.memories_scanned} "
                  f"merged={report.memories_merged} "
                  f"pruned={report.memories_pruned} "
                  f"promoted={report.memories_promoted} "
                  f"({report.duration_ms:.1f}ms)")

        elif args.cmd == "list":
            memories = bridge.list_memories(agent_id=args.agent_id,
                                            mtype=args.memory_type)
            scope = args.agent_id or "all agents"
            print(f"\n{BOLD}{B}── Memories [{scope}] — {len(memories)} entries ─────{NC}")
            for e in memories:
                _print_entry(e)

        elif args.cmd == "clear-expired":
            n = bridge.clear_expired()
            print(f"{G}✓{NC} Cleared {R}{n}{NC} expired memories.")

    finally:
        bridge.close()


if __name__ == "__main__":
    main()
