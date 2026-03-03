"""
BlackRoad AI Memory Bridge — Vector store interface, semantic search,
and memory consolidation for AI agents.  Persistent episodic, semantic,
working, and procedural memory with deterministic embeddings.

Usage:
    python memory_bridge.py store   --agent-id lucidia --content "..." --type semantic
    python memory_bridge.py search  --query "user preferences" --agent-id lucidia
    python memory_bridge.py context --agent-id lucidia --query "deployment"
    python memory_bridge.py consolidate --agent-id lucidia
    python memory_bridge.py list    --agent-id lucidia --type episodic
    python memory_bridge.py clear-expired
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

EMBED_DIM = 64
DB_PATH = Path.home() / ".blackroad" / "memory_bridge.db"

MEMORY_TYPES = ("episodic", "semantic", "working", "procedural")
_PROMOTE_THRESHOLD = 5  # access_count >= this promotes episodic -> semantic

# ANSI colours
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; M = "\033[0;35m"; NC = "\033[0m"
BOLD = "\033[1m"


# ── Embedding helpers ──────────────────────────────────────────────────────────
def _embed(text: str) -> List[float]:
    """Deterministic 64-dim unit-sphere embedding via SHA-256 seeded RNG."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(EMBED_DIM)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(v1: List[float], v2: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    m1 = math.sqrt(sum(v * v for v in v1)) or 1.0
    m2 = math.sqrt(sum(v * v for v in v2)) or 1.0
    return dot / (m1 * m2)


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class MemoryEntry:
    """A single memory unit stored in the bridge."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    content: str = ""
    memory_type: str = "episodic"
    importance: float = 0.5
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed: Optional[str] = None

    def __post_init__(self) -> None:
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(
                f"memory_type must be one of {MEMORY_TYPES}, got {self.memory_type!r}"
            )
        self.importance = max(0.0, min(1.0, self.importance))


@dataclass
class VectorEmbedding:
    """Stored embedding vector for a memory entry."""
    embed_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    memory_id: str = ""
    vector: List[float] = field(default_factory=list)
    model: str = "sha256-gauss-64"
    dim: int = EMBED_DIM
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SearchResult:
    """Result of a semantic memory search."""
    memory_id: str = ""
    content: str = ""
    similarity: float = 0.0
    importance: float = 0.5
    memory_type: str = "episodic"
    agent_id: str = ""


@dataclass
class ConsolidationReport:
    """Report from a memory consolidation run."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
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

    Stores agent memories in SQLite with vector embeddings,
    cosine-similarity search, TTL expiry, and consolidation.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
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
                memory_id     TEXT PRIMARY KEY,
                agent_id      TEXT NOT NULL DEFAULT '',
                content       TEXT NOT NULL DEFAULT '',
                memory_type   TEXT NOT NULL DEFAULT 'episodic',
                importance    REAL NOT NULL DEFAULT 0.5,
                access_count  INTEGER NOT NULL DEFAULT 0,
                tags_json     TEXT NOT NULL DEFAULT '[]',
                expires_at    TEXT,
                created_at    TEXT NOT NULL,
                last_accessed TEXT
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                embed_id    TEXT PRIMARY KEY,
                memory_id   TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                model       TEXT NOT NULL DEFAULT 'sha256-gauss-64',
                dim         INTEGER NOT NULL DEFAULT 64,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consolidation_reports (
                report_id         TEXT PRIMARY KEY,
                agent_id          TEXT NOT NULL DEFAULT '',
                memories_scanned  INTEGER NOT NULL DEFAULT 0,
                memories_merged   INTEGER NOT NULL DEFAULT 0,
                memories_pruned   INTEGER NOT NULL DEFAULT 0,
                memories_promoted INTEGER NOT NULL DEFAULT 0,
                duration_ms       REAL NOT NULL DEFAULT 0.0,
                created_at        TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_mem_type  ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_mem_imp   ON memories(importance DESC);
        """)
        self._conn.commit()

    def _load_vectors(self) -> None:
        """Load all stored embeddings into the in-memory cache."""
        rows = self._conn.execute(
            "SELECT memory_id, vector_json FROM embeddings"
        ).fetchall()
        for row in rows:
            self._vectors[row["memory_id"]] = json.loads(row["vector_json"])

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
        """Store a memory entry and compute its vector embedding."""
        entry.__post_init__()
        vec = _embed(entry.content)
        self._conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                entry.memory_id, entry.agent_id, entry.content,
                entry.memory_type, entry.importance, entry.access_count,
                json.dumps(entry.tags), entry.expires_at,
                entry.created_at, entry.last_accessed,
            ),
        )
        embed_id = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
            (
                embed_id, entry.memory_id, json.dumps(vec),
                "sha256-gauss-64", EMBED_DIM, datetime.utcnow().isoformat(),
            ),
        )
        self._conn.commit()
        self._vectors[entry.memory_id] = vec
        return entry

    def search_semantic(
        self,
        query: str,
        agent_id: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """Cosine-similarity semantic search over stored memories."""
        q_vec = _embed(query)
        sql = "SELECT * FROM memories"
        params: list = []
        if agent_id is not None:
            sql += " WHERE agent_id=?"
            params.append(agent_id)
        rows = self._conn.execute(sql, params).fetchall()

        results: List[SearchResult] = []
        for row in rows:
            mid = row["memory_id"]
            vec = self._vectors.get(mid)
            if vec is None:
                continue
            sim = _cosine(q_vec, vec)
            if sim >= min_similarity:
                results.append(
                    SearchResult(
                        memory_id=mid,
                        content=row["content"],
                        similarity=round(sim, 6),
                        importance=row["importance"],
                        memory_type=row["memory_type"],
                        agent_id=row["agent_id"],
                    )
                )

        results.sort(key=lambda r: r.similarity, reverse=True)
        results = results[:top_k]

        if results:
            now = datetime.utcnow().isoformat()
            for r in results:
                self._conn.execute(
                    "UPDATE memories SET access_count=access_count+1, last_accessed=?"
                    " WHERE memory_id=?",
                    (now, r.memory_id),
                )
            self._conn.commit()

        return results

    def list_memories(
        self,
        agent_id: Optional[str] = None,
        mtype: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """List memories, optionally filtered by agent and/or type."""
        sql = "SELECT * FROM memories"
        params: list = []
        conditions: List[str] = []
        if agent_id is not None:
            conditions.append("agent_id=?")
            params.append(agent_id)
        if mtype is not None:
            conditions.append("memory_type=?")
            params.append(mtype)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY importance DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_context(
        self,
        agent_id: str,
        query: str,
        max_tokens: int = 512,
    ) -> str:
        """Build a token-limited context string for LLM prompts."""
        results = self.search_semantic(
            query, agent_id=agent_id, top_k=20, min_similarity=0.0
        )
        if not results:
            return ""
        lines: List[str] = []
        token_count = 0
        for r in results:
            line = f"[{r.memory_type}|{r.similarity:.3f}] {r.content}"
            tokens = len(line.split())
            if token_count + tokens > max_tokens:
                break
            lines.append(line)
            token_count += tokens
        return "\n".join(lines)

    def clear_expired(self) -> int:
        """Delete all memories past their expires_at time. Returns count deleted."""
        now = datetime.utcnow().isoformat()
        rows = self._conn.execute(
            "SELECT memory_id FROM memories"
            " WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        ).fetchall()
        ids = [r["memory_id"] for r in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM memories WHERE memory_id IN ({placeholders})", ids
            )
            self._conn.execute(
                f"DELETE FROM embeddings WHERE memory_id IN ({placeholders})", ids
            )
            self._conn.commit()
            for mid in ids:
                self._vectors.pop(mid, None)
        return len(ids)

    def consolidate_memories(
        self,
        agent_id: str,
        similarity_threshold: float = 0.9,
    ) -> ConsolidationReport:
        """Merge near-duplicates, prune expired, promote high-access episodic."""
        t_start = time.monotonic()
        report = ConsolidationReport(agent_id=agent_id)

        # ── Prune expired ───────────────────────────────────────────────────
        now = datetime.utcnow().isoformat()
        expired = self._conn.execute(
            "SELECT memory_id FROM memories"
            " WHERE agent_id=? AND expires_at IS NOT NULL AND expires_at <= ?",
            (agent_id, now),
        ).fetchall()
        expired_ids = [r["memory_id"] for r in expired]
        if expired_ids:
            ph = ",".join("?" * len(expired_ids))
            self._conn.execute(
                f"DELETE FROM memories WHERE memory_id IN ({ph})", expired_ids
            )
            self._conn.execute(
                f"DELETE FROM embeddings WHERE memory_id IN ({ph})", expired_ids
            )
            for mid in expired_ids:
                self._vectors.pop(mid, None)
            report.memories_pruned = len(expired_ids)

        # ── Load remaining agent memories ───────────────────────────────────
        entries = self.list_memories(agent_id=agent_id)
        report.memories_scanned = len(entries)

        # ── Merge near-duplicates ───────────────────────────────────────────
        merged_ids: set = set()
        for i, e1 in enumerate(entries):
            if e1.memory_id in merged_ids:
                continue
            v1 = self._vectors.get(e1.memory_id)
            if v1 is None:
                continue
            for e2 in entries[i + 1:]:
                if e2.memory_id in merged_ids:
                    continue
                v2 = self._vectors.get(e2.memory_id)
                if v2 is None:
                    continue
                if _cosine(v1, v2) >= similarity_threshold:
                    drop = (
                        e2.memory_id
                        if e1.importance >= e2.importance
                        else e1.memory_id
                    )
                    merged_ids.add(drop)

        if merged_ids:
            ids_list = list(merged_ids)
            ph = ",".join("?" * len(ids_list))
            self._conn.execute(
                f"DELETE FROM memories WHERE memory_id IN ({ph})", ids_list
            )
            self._conn.execute(
                f"DELETE FROM embeddings WHERE memory_id IN ({ph})", ids_list
            )
            for mid in merged_ids:
                self._vectors.pop(mid, None)
            report.memories_merged = len(merged_ids)

        # ── Promote high-access episodic -> semantic ────────────────────────
        promote = self._conn.execute(
            "SELECT memory_id FROM memories"
            " WHERE agent_id=? AND memory_type='episodic'"
            f" AND access_count>={_PROMOTE_THRESHOLD}",
            (agent_id,),
        ).fetchall()
        promote_ids = [r["memory_id"] for r in promote]
        if promote_ids:
            ph = ",".join("?" * len(promote_ids))
            self._conn.execute(
                f"UPDATE memories SET memory_type='semantic'"
                f" WHERE memory_id IN ({ph})",
                promote_ids,
            )
            report.memories_promoted = len(promote_ids)

        self._conn.commit()
        report.duration_ms = round((time.monotonic() - t_start) * 1000, 2)

        # Persist the report
        self._conn.execute(
            "INSERT INTO consolidation_reports VALUES (?,?,?,?,?,?,?,?)",
            (
                report.report_id, report.agent_id, report.memories_scanned,
                report.memories_merged, report.memories_pruned,
                report.memories_promoted, report.duration_ms, report.created_at,
            ),
        )
        self._conn.commit()
        return report

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory-bridge",
        description="BlackRoad AI Memory Bridge — vector memory store",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("store", help="Store a new memory")
    st.add_argument("--agent-id", required=True)
    st.add_argument("--content", required=True)
    st.add_argument("--type", choices=list(MEMORY_TYPES), default="episodic",
                    dest="memory_type")
    st.add_argument("--importance", type=float, default=0.5)
    st.add_argument("--tags", nargs="*", default=[])
    st.add_argument("--ttl", type=int, default=0, help="TTL in seconds (0 = no expiry)")

    sr = sub.add_parser("search", help="Semantic search")
    sr.add_argument("--query", required=True)
    sr.add_argument("--agent-id", default=None)
    sr.add_argument("--top-k", type=int, default=5)
    sr.add_argument("--min-sim", type=float, default=0.1)

    ctx = sub.add_parser("context", help="Build LLM context string")
    ctx.add_argument("--agent-id", required=True)
    ctx.add_argument("--query", required=True)
    ctx.add_argument("--max-tokens", type=int, default=512)

    cons = sub.add_parser("consolidate", help="Consolidate memories")
    cons.add_argument("--agent-id", required=True)
    cons.add_argument("--threshold", type=float, default=0.9)

    ls = sub.add_parser("list", help="List memories")
    ls.add_argument("--agent-id", default=None)
    ls.add_argument("--type", default=None, dest="memory_type")

    sub.add_parser("clear-expired", help="Remove expired memories")

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
            print(
                f"{G}✓{NC} Memory stored [{result.memory_type}]"
                f" {C}{result.memory_id}{NC} importance={result.importance:.2f}"
            )

        elif args.cmd == "search":
            results = bridge.search_semantic(
                args.query, agent_id=args.agent_id,
                top_k=args.top_k, min_similarity=args.min_sim,
            )
            if not results:
                print(f"{Y}No results for: {args.query!r}{NC}")
            else:
                print(f"\n{BOLD}{B}── Search: {args.query[:60]} ──────{NC}")
                for i, r in enumerate(results, 1):
                    bar = "█" * int(r.similarity * 10) + "░" * (10 - int(r.similarity * 10))
                    print(f"  {i}. {C}{r.memory_id}{NC} sim={r.similarity:.3f} {bar}")
                    print(f"     {r.content[:120]}")

        elif args.cmd == "context":
            ctx = bridge.get_context(
                args.agent_id, args.query, max_tokens=args.max_tokens
            )
            if ctx:
                print(f"\n{BOLD}{B}── Context for {args.agent_id} ──────{NC}")
                print(ctx)
            else:
                print(f"{Y}No memories found for agent {args.agent_id!r}{NC}")

        elif args.cmd == "consolidate":
            report = bridge.consolidate_memories(
                args.agent_id, similarity_threshold=args.threshold
            )
            print(
                f"{G}✓{NC} Consolidation [{C}{report.agent_id}{NC}]:"
                f" scanned={report.memories_scanned}"
                f" merged={report.memories_merged}"
                f" pruned={report.memories_pruned}"
                f" promoted={report.memories_promoted}"
                f" ({report.duration_ms:.1f}ms)"
            )

        elif args.cmd == "list":
            entries = bridge.list_memories(
                agent_id=args.agent_id, mtype=args.memory_type
            )
            if not entries:
                print(f"{Y}No memories found.{NC}")
            else:
                for e in entries:
                    print(
                        f"  {C}{e.memory_id}{NC} [{e.memory_type}]"
                        f" imp={e.importance:.2f} acc={e.access_count}"
                        f" — {e.content[:80]}"
                    )

        elif args.cmd == "clear-expired":
            n = bridge.clear_expired()
            print(f"{G}✓{NC} Cleared {n} expired memories.")

    finally:
        bridge.close()


if __name__ == "__main__":
    main()
