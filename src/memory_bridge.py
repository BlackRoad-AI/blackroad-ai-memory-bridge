"""
BlackRoad AI Memory Bridge — TF-IDF semantic search, SQLite FTS5 indexing,
and agent-scoped memory management with session summarisation.

Usage:
    python memory_bridge.py store --agent lucidia --type fact \
        --content "User prefers dark mode" --importance 0.8
    python memory_bridge.py search --query "user preferences" --agent lucidia
    python memory_bridge.py recent --agent lucidia --hours 48
    python memory_bridge.py prune --agent lucidia --threshold 0.3
    python memory_bridge.py summarize --agent lucidia
    python memory_bridge.py export --agent lucidia --out lucidia_memory.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── ANSI colours ─────────────────────────────────────────────────────────────
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; M = "\033[0;35m"; NC = "\033[0m"
BOLD = "\033[1m"

DB_PATH = Path.home() / ".blackroad" / "memory_bridge.db"

# Memory types accepted by the bridge
MEMORY_TYPES = ("fact", "conversation", "task", "code")

# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class MemoryEntry:
    """
    A single memory unit stored in the bridge.

    Fields:
        id            — UUID (first 8 chars) uniquely identifying this memory.
        content       — Raw text content of the memory.
        embedding_str — Space-joined TF-IDF feature string (top-N weighted terms).
        type          — One of: fact | conversation | task | code.
        source_agent  — Name of the agent that stored this memory.
        timestamp     — ISO-8601 creation time (UTC).
        importance    — Float in [0, 1]; higher = more important.
        tags          — Arbitrary string labels for filtering.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    embedding_str: str = ""             # TF-IDF derived feature string
    type: str = "fact"                  # fact | conversation | task | code
    source_agent: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    importance: float = 0.5            # 0.0 – 1.0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in MEMORY_TYPES:
            raise ValueError(f"type must be one of {MEMORY_TYPES}, got {self.type!r}")
        self.importance = max(0.0, min(1.0, self.importance))

    def age_hours(self) -> float:
        """Hours since this memory was created."""
        try:
            created = datetime.fromisoformat(self.timestamp)
            return (datetime.utcnow() - created).total_seconds() / 3600
        except Exception:
            return 0.0

    def decay_score(self, half_life_hours: float = 72.0) -> float:
        """Importance decayed by exponential time decay."""
        age = self.age_hours()
        decay = math.exp(-math.log(2) * age / max(half_life_hours, 1.0))
        return round(self.importance * decay, 6)


@dataclass
class SearchResult:
    """Result of a memory search query."""
    entry: MemoryEntry
    score: float           # TF-IDF similarity score (higher = more relevant)
    rank: int = 0

    def __repr__(self) -> str:
        return (f"SearchResult(id={self.entry.id!r}, "
                f"score={self.score:.4f}, type={self.entry.type!r})")


@dataclass
class SessionSummary:
    """Summary produced by MemoryBridge.summarize_session()."""
    agent: str
    total_memories: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    avg_importance: float = 0.0
    top_terms: List[str] = field(default_factory=list)
    most_recent: Optional[str] = None
    oldest: Optional[str] = None
    high_importance_count: int = 0     # importance ≥ 0.7
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── TF-IDF helpers ────────────────────────────────────────────────────────────
_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of",
    "and", "or", "for", "with", "was", "are", "be", "this", "that",
    "i", "you", "we", "they", "he", "she", "has", "had", "have",
    "from", "but", "not", "by", "as", "so", "if", "do", "did",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = []
    word = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            word.append(ch)
        else:
            if word:
                w = "".join(word)
                if w not in _STOPWORDS and len(w) > 1:
                    tokens.append(w)
                word = []
    if word:
        w = "".join(word)
        if w not in _STOPWORDS and len(w) > 1:
            tokens.append(w)
    return tokens


def _tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency (log-normalised): 1 + log(count)."""
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return {
        term: 1.0 + math.log(count / total + 1.0)
        for term, count in counts.items()
    }


def _idf(term: str, corpus_tokens: List[List[str]], n_docs: int) -> float:
    """Inverse document frequency: log((N+1) / (df+1))."""
    df = sum(1 for doc in corpus_tokens if term in doc)
    return math.log((n_docs + 1.0) / (df + 1.0))


def _tfidf_vector(tokens: List[str],
                  corpus_tokens: List[List[str]]) -> Dict[str, float]:
    """Full TF-IDF vector for a single document against a corpus."""
    n = max(len(corpus_tokens), 1)
    tf_scores = _tf(tokens)
    return {
        term: round(tf_val * _idf(term, corpus_tokens, n), 6)
        for term, tf_val in tf_scores.items()
    }


def _embedding_str(content: str, corpus_tokens: Optional[List[List[str]]] = None,
                   top_n: int = 20) -> str:
    """
    Produce the embedding_str: space-joined top-N TF-IDF terms.

    When corpus_tokens is None (single document), falls back to
    pure TF log-normalisation.
    """
    tokens = _tokenize(content)
    if not tokens:
        return ""
    if corpus_tokens:
        scores = _tfidf_vector(tokens, corpus_tokens)
    else:
        scores = _tf(tokens)
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return " ".join(term for term, _ in top)


def _cosine_sim(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return round(dot / (mag_a * mag_b), 6)


def _embedding_str_to_vec(emb_str: str) -> Dict[str, float]:
    """Reconstruct a TF vector from an embedding_str (positional weight)."""
    terms = emb_str.split()
    n = max(len(terms), 1)
    return {term: 1.0 + math.log((n - i) / n + 1.0)
            for i, term in enumerate(terms)}


# ── Core class ────────────────────────────────────────────────────────────────
class MemoryBridge:
    """
    BlackRoad AI Memory Bridge.

    Stores agent memories in SQLite with FTS5 full-text search,
    TF-IDF vector similarity, importance decay, and session export.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                content       TEXT NOT NULL,
                embedding_str TEXT DEFAULT '',
                type          TEXT NOT NULL DEFAULT 'fact',
                source_agent  TEXT NOT NULL DEFAULT '',
                timestamp     TEXT NOT NULL,
                importance    REAL NOT NULL DEFAULT 0.5,
                tags_json     TEXT DEFAULT '[]'
            );

            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(
                id UNINDEXED,
                content,
                embedding_str,
                source_agent UNINDEXED,
                content_rowid='rowid'
            );

            CREATE TABLE IF NOT EXISTS search_log (
                query_id     TEXT PRIMARY KEY,
                query        TEXT,
                agent        TEXT,
                results_n    INTEGER,
                top_score    REAL,
                queried_at   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(source_agent);
            CREATE INDEX IF NOT EXISTS idx_mem_type  ON memories(type);
            CREATE INDEX IF NOT EXISTS idx_mem_ts    ON memories(timestamp);
            CREATE INDEX IF NOT EXISTS idx_mem_imp   ON memories(importance);
        """)
        self._conn.commit()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _row_to_entry(self, row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            embedding_str=row["embedding_str"],
            type=row["type"],
            source_agent=row["source_agent"],
            timestamp=row["timestamp"],
            importance=row["importance"],
            tags=json.loads(row["tags_json"] or "[]"),
        )

    def _all_corpus_tokens(self, agent: Optional[str] = None) -> List[List[str]]:
        """Return tokenised content for all (or agent-scoped) memories."""
        q = "SELECT content FROM memories"
        p: List = []
        if agent:
            q += " WHERE source_agent=?"
            p.append(agent)
        rows = self._conn.execute(q, p).fetchall()
        return [_tokenize(r["content"]) for r in rows]

    # ── Public API ────────────────────────────────────────────────────────────
    def store(self, content: str, type: str, agent: str,
              importance: float = 0.5,
              tags: Optional[List[str]] = None) -> MemoryEntry:
        """
        Store a new memory, computing its TF-IDF embedding_str.

        Args:
            content:    Raw text to remember.
            type:       One of fact | conversation | task | code.
            agent:      Name of the agent storing this memory.
            importance: Salience score 0–1 (default 0.5).
            tags:       Optional labels for filtering.

        Returns:
            The persisted MemoryEntry.
        """
        corpus = self._all_corpus_tokens(agent)
        tokens = _tokenize(content)
        corpus.append(tokens)  # include self in IDF calculation
        emb = _embedding_str(content, corpus_tokens=corpus)

        entry = MemoryEntry(
            content=content,
            embedding_str=emb,
            type=type,
            source_agent=agent,
            importance=importance,
            tags=tags or [],
        )
        entry.__post_init__()  # validate type and clamp importance

        self._conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?)",
            (entry.id, entry.content, entry.embedding_str,
             entry.type, entry.source_agent, entry.timestamp,
             entry.importance, json.dumps(entry.tags))
        )
        # Keep FTS5 in sync
        self._conn.execute(
            "INSERT INTO memories_fts(id, content, embedding_str, source_agent) "
            "VALUES (?,?,?,?)",
            (entry.id, entry.content, entry.embedding_str, entry.source_agent)
        )
        self._conn.commit()
        type_color = {
            "fact": G, "conversation": C, "task": Y, "code": M
        }.get(entry.type, NC)
        print(f"{G}✓{NC} Stored [{type_color}{entry.type}{NC}] "
              f"{C}{entry.id}{NC}  imp={entry.importance:.2f}"
              + (f"  tags={entry.tags}" if entry.tags else ""))
        return entry

    def search(self, query: str, top_k: int = 5,
               agent: Optional[str] = None,
               memory_type: Optional[str] = None,
               min_score: float = 0.0) -> List[SearchResult]:
        """
        Retrieve the top-k memories most relevant to `query`.

        Algorithm:
            1. FTS5 BM25 pre-filter to narrow the candidate set.
            2. Re-rank candidates by cosine similarity of TF-IDF vectors.
            3. Blend with importance: final_score = 0.7·sim + 0.3·importance.

        Args:
            query:       Free-text search query.
            top_k:       Maximum results to return (default 5).
            agent:       Restrict to a specific agent's memories.
            memory_type: Restrict to a specific memory type.
            min_score:   Minimum blended score threshold.

        Returns:
            List of SearchResult ordered by descending score.
        """
        # Step 1: FTS5 candidate retrieval
        fts_query = " OR ".join(f'"{t}"' for t in _tokenize(query)) or query
        fts_rows = self._conn.execute(
            "SELECT id FROM memories_fts WHERE memories_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (fts_query, max(top_k * 5, 50))
        ).fetchall()
        candidate_ids = {r["id"] for r in fts_rows}

        # Step 2: Build full candidate list (FTS + fallback for small corpora)
        q = "SELECT * FROM memories"
        params: List = []
        conditions = []
        if candidate_ids:
            placeholders = ",".join("?" * len(candidate_ids))
            conditions.append(f"id IN ({placeholders})")
            params.extend(candidate_ids)
        if agent:
            conditions.append("source_agent=?")
            params.append(agent)
        if memory_type:
            conditions.append("type=?")
            params.append(memory_type)
        if conditions:
            q += " WHERE " + " AND ".join(conditions)
        q += " LIMIT 200"
        rows = self._conn.execute(q, params).fetchall()

        if not rows and not candidate_ids:
            # No FTS matches — scan entire (agent-scoped) corpus
            q2 = "SELECT * FROM memories"
            p2: List = []
            if agent:
                q2 += " WHERE source_agent=?"
                p2.append(agent)
            if memory_type:
                q2 += (" AND " if agent else " WHERE ") + "type=?"
                p2.append(memory_type)
            rows = self._conn.execute(q2, p2).fetchall()

        # Step 3: TF-IDF cosine re-rank
        corpus_tokens = [_tokenize(r["content"]) for r in rows]
        query_tokens = _tokenize(query)
        query_vec = _tfidf_vector(query_tokens, corpus_tokens) if corpus_tokens \
            else _tf(query_tokens)

        results: List[SearchResult] = []
        for row in rows:
            entry = self._row_to_entry(row)
            doc_vec = _embedding_str_to_vec(entry.embedding_str)
            sim = _cosine_sim(query_vec, doc_vec)
            score = round(0.7 * sim + 0.3 * entry.importance, 6)
            if score >= min_score:
                results.append(SearchResult(entry=entry, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        for i, r in enumerate(results):
            r.rank = i + 1

        # Log search
        self._conn.execute(
            "INSERT INTO search_log VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4())[:8], query, agent or "",
             len(results),
             results[0].score if results else 0.0,
             datetime.utcnow().isoformat())
        )
        self._conn.commit()
        return results

    def get_recent(self, hours: float = 24.0,
                   agent: Optional[str] = None) -> List[MemoryEntry]:
        """
        Return all memories created within the last `hours` hours.

        Args:
            hours: Lookback window (default 24).
            agent: Restrict to a specific agent.

        Returns:
            List of MemoryEntry sorted newest-first.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        q = "SELECT * FROM memories WHERE timestamp >= ?"
        params: List = [cutoff]
        if agent:
            q += " AND source_agent=?"
            params.append(agent)
        q += " ORDER BY timestamp DESC"
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def prune_old(self, keep_importance_above: float = 0.3,
                  older_than_hours: float = 168.0,
                  agent: Optional[str] = None) -> int:
        """
        Delete memories that are both old and low-importance.

        A memory is pruned when:
            age > `older_than_hours`  AND  importance <= `keep_importance_above`

        Args:
            keep_importance_above: Memories with importance > this are kept.
            older_than_hours:      Age threshold in hours (default 168 = 7 days).
            agent:                 Restrict pruning to a specific agent.

        Returns:
            Number of memories deleted.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=older_than_hours)).isoformat()
        q = ("SELECT id FROM memories "
             "WHERE timestamp < ? AND importance <= ?")
        params: List = [cutoff, keep_importance_above]
        if agent:
            q += " AND source_agent=?"
            params.append(agent)

        rows = self._conn.execute(q, params).fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", ids
            )
            # FTS5 doesn't auto-cascade; rebuild via delete+insert is
            # expensive for large corpora so we use a content-table approach:
            self._conn.execute(
                f"DELETE FROM memories_fts WHERE id IN ({placeholders})", ids
            )
            self._conn.commit()
        scope = f"agent={agent}" if agent else "all agents"
        print(f"{G}✓{NC} Pruned {R}{len(ids)}{NC} memories "
              f"(imp≤{keep_importance_above}, age>{older_than_hours:.0f}h) "
              f"[{scope}]")
        return len(ids)

    def summarize_session(self, agent: Optional[str] = None,
                          hours: float = 24.0) -> SessionSummary:
        """
        Produce a structured summary of recent memory activity.

        Covers memories stored within the last `hours` window.
        Extracts top recurring terms across the session corpus.

        Args:
            agent: Restrict summary to a specific agent.
            hours: Session window in hours (default 24).

        Returns:
            SessionSummary dataclass.
        """
        entries = self.get_recent(hours=hours, agent=agent)
        summary = SessionSummary(agent=agent or "all")
        if not entries:
            return summary

        summary.total_memories = len(entries)
        by_type: Dict[str, int] = {}
        importances: List[float] = []
        timestamps: List[str] = []
        term_counter: Counter = Counter()

        for e in entries:
            by_type[e.type] = by_type.get(e.type, 0) + 1
            importances.append(e.importance)
            timestamps.append(e.timestamp)
            if e.importance >= 0.7:
                summary.high_importance_count += 1
            term_counter.update(_tokenize(e.content))

        summary.by_type = by_type
        summary.avg_importance = round(
            sum(importances) / max(len(importances), 1), 4
        )
        summary.top_terms = [t for t, _ in term_counter.most_common(10)]
        summary.most_recent = max(timestamps)
        summary.oldest = min(timestamps)
        return summary

    def export_for_agent(self, agent_name: str,
                         output_path: Optional[Path] = None,
                         include_low_importance: bool = False) -> Dict:
        """
        Export all memories for a given agent as a JSON-serialisable dict.

        Suitable for injecting into an agent's system prompt or for
        cross-provider identity transfer.

        Args:
            agent_name:             Target agent to export.
            output_path:            Write JSON to this path (optional).
            include_low_importance: Include memories with importance < 0.3.

        Returns:
            The export dict.
        """
        q = "SELECT * FROM memories WHERE source_agent=?"
        params: List = [agent_name]
        if not include_low_importance:
            q += " AND importance >= 0.3"
        q += " ORDER BY importance DESC, timestamp DESC"

        rows = self._conn.execute(q, params).fetchall()
        entries = [self._row_to_entry(r) for r in rows]

        export = {
            "schema": "blackroad-memory-bridge-v1",
            "agent": agent_name,
            "exported_at": datetime.utcnow().isoformat(),
            "total": len(entries),
            "by_type": {
                t: sum(1 for e in entries if e.type == t)
                for t in MEMORY_TYPES
            },
            "memories": [
                {
                    "id": e.id,
                    "content": e.content,
                    "type": e.type,
                    "importance": e.importance,
                    "tags": e.tags,
                    "timestamp": e.timestamp,
                    "embedding_str": e.embedding_str,
                }
                for e in entries
            ],
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(export, indent=2))
            print(f"{G}✓{NC} Exported {len(entries)} memories for "
                  f"{C}{agent_name}{NC} → {output_path}")
        return export

    # ── Utilities ─────────────────────────────────────────────────────────────
    def update_importance(self, memory_id: str, importance: float) -> bool:
        """Adjust the importance score of an existing memory."""
        importance = max(0.0, min(1.0, importance))
        result = self._conn.execute(
            "UPDATE memories SET importance=? WHERE id=?",
            (importance, memory_id)
        )
        self._conn.commit()
        if result.rowcount:
            print(f"{G}✓{NC} Memory {C}{memory_id}{NC} "
                  f"importance → {importance:.2f}")
        return result.rowcount > 0

    def tag_memory(self, memory_id: str, *new_tags: str) -> bool:
        """Append tags to a memory without replacing existing ones."""
        row = self._conn.execute(
            "SELECT tags_json FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if not row:
            return False
        existing = json.loads(row["tags_json"] or "[]")
        merged = list(dict.fromkeys(existing + list(new_tags)))
        self._conn.execute(
            "UPDATE memories SET tags_json=? WHERE id=?",
            (json.dumps(merged), memory_id)
        )
        self._conn.commit()
        return True

    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def list_agents(self) -> List[Tuple[str, int]]:
        """Return (agent_name, memory_count) tuples sorted by count desc."""
        rows = self._conn.execute(
            "SELECT source_agent, COUNT(*) as n FROM memories "
            "GROUP BY source_agent ORDER BY n DESC"
        ).fetchall()
        return [(r["source_agent"], r["n"]) for r in rows]

    def delete(self, memory_id: str) -> bool:
        """Delete a single memory by ID."""
        result = self._conn.execute(
            "DELETE FROM memories WHERE id=?", (memory_id,)
        )
        if result.rowcount:
            self._conn.execute(
                "DELETE FROM memories_fts WHERE id=?", (memory_id,)
            )
            self._conn.commit()
            print(f"{G}✓{NC} Deleted memory {C}{memory_id}{NC}")
        return result.rowcount > 0

    def close(self) -> None:
        self._conn.close()


# ── Pretty printing ───────────────────────────────────────────────────────────
_TYPE_COLOR = {"fact": G, "conversation": C, "task": Y, "code": M}


def _print_entry(e: MemoryEntry, score: Optional[float] = None) -> None:
    tc = _TYPE_COLOR.get(e.type, NC)
    score_str = f"  score={Y}{score:.4f}{NC}" if score is not None else ""
    tags_str = f"  {B}[{', '.join(e.tags)}]{NC}" if e.tags else ""
    print(f"  {C}{e.id}{NC} [{tc}{e.type}{NC}] "
          f"imp={e.importance:.2f}{score_str}{tags_str}")
    print(f"     {e.content[:120]}")


def _print_summary(s: SessionSummary) -> None:
    print(f"\n{BOLD}{B}╔══ Session Summary: {s.agent} ══════════════╗{NC}")
    print(f"  {C}Window{NC}      last 24h  |  {s.total_memories} memories")
    for t in MEMORY_TYPES:
        cnt = s.by_type.get(t, 0)
        tc = _TYPE_COLOR.get(t, NC)
        bar = "█" * cnt + "░" * max(0, 10 - cnt)
        print(f"  {tc}{t:<14}{NC} {bar}  {cnt}")
    print(f"  {C}Avg importance{NC}  {s.avg_importance:.2f}  "
          f"High (≥0.7): {G}{s.high_importance_count}{NC}")
    print(f"  {C}Top terms{NC}       {', '.join(s.top_terms[:8])}")
    print(f"  {C}Range{NC}           {s.oldest[:19] if s.oldest else '—'} → "
          f"{s.most_recent[:19] if s.most_recent else '—'}")
    print(f"{BOLD}{B}╚════════════════════════════════════════════╝{NC}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory-bridge",
        description="BlackRoad AI Memory Bridge — TF-IDF semantic memory store",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # store
    st = sub.add_parser("store", help="Store a new memory")
    st.add_argument("--agent", required=True)
    st.add_argument("--content", required=True)
    st.add_argument("--type", choices=list(MEMORY_TYPES), default="fact")
    st.add_argument("--importance", type=float, default=0.5)
    st.add_argument("--tags", nargs="*", default=[])

    # search
    sr = sub.add_parser("search", help="Semantic search")
    sr.add_argument("--query", required=True)
    sr.add_argument("--agent", default=None)
    sr.add_argument("--type", default=None)
    sr.add_argument("--top-k", type=int, default=5)
    sr.add_argument("--min-score", type=float, default=0.0)

    # recent
    rc = sub.add_parser("recent", help="List recent memories")
    rc.add_argument("--agent", default=None)
    rc.add_argument("--hours", type=float, default=24.0)

    # prune
    pr = sub.add_parser("prune", help="Prune old/low-importance memories")
    pr.add_argument("--agent", default=None)
    pr.add_argument("--threshold", type=float, default=0.3)
    pr.add_argument("--older-than", type=float, default=168.0,
                    metavar="HOURS", dest="older_than")

    # summarize
    sm = sub.add_parser("summarize", help="Summarize session activity")
    sm.add_argument("--agent", default=None)
    sm.add_argument("--hours", type=float, default=24.0)

    # export
    ex = sub.add_parser("export", help="Export agent memory to JSON")
    ex.add_argument("--agent", required=True)
    ex.add_argument("--out", default=None)
    ex.add_argument("--all", action="store_true",
                    help="Include low-importance memories")

    # tag
    tg = sub.add_parser("tag", help="Add tags to a memory")
    tg.add_argument("memory_id")
    tg.add_argument("tags", nargs="+")

    # delete
    dl = sub.add_parser("delete", help="Delete a memory by ID")
    dl.add_argument("memory_id")

    # agents
    sub.add_parser("agents", help="List agents and memory counts")

    return p


def main() -> None:
    args = _build_parser().parse_args()
    bridge = MemoryBridge()

    try:
        if args.cmd == "store":
            bridge.store(
                content=args.content, type=args.type,
                agent=args.agent, importance=args.importance,
                tags=args.tags,
            )

        elif args.cmd == "search":
            results = bridge.search(
                query=args.query, top_k=args.top_k,
                agent=args.agent, memory_type=args.type,
                min_score=args.min_score,
            )
            if not results:
                print(f"{Y}No results for: {args.query!r}{NC}")
            else:
                print(f"\n{BOLD}{B}── Search: {args.query[:60]} ─────{NC}")
                for r in results:
                    _print_entry(r.entry, score=r.score)

        elif args.cmd == "recent":
            entries = bridge.get_recent(hours=args.hours, agent=args.agent)
            scope = args.agent or "all agents"
            print(f"\n{BOLD}{B}── Recent ({args.hours:.0f}h) [{scope}] "
                  f"— {len(entries)} entries ─────{NC}")
            for e in entries:
                _print_entry(e)

        elif args.cmd == "prune":
            bridge.prune_old(
                keep_importance_above=args.threshold,
                older_than_hours=args.older_than,
                agent=args.agent,
            )

        elif args.cmd == "summarize":
            s = bridge.summarize_session(agent=args.agent, hours=args.hours)
            _print_summary(s)

        elif args.cmd == "export":
            out = Path(args.out) if args.out else None
            data = bridge.export_for_agent(
                args.agent, output_path=out,
                include_low_importance=args.all,
            )
            if not out:
                print(json.dumps(data, indent=2))

        elif args.cmd == "tag":
            bridge.tag_memory(args.memory_id, *args.tags)

        elif args.cmd == "delete":
            bridge.delete(args.memory_id)

        elif args.cmd == "agents":
            agents = bridge.list_agents()
            if not agents:
                print(f"{Y}No agents found.{NC}")
                return
            print(f"\n{BOLD}{B}── Agents ─────────────────{NC}")
            for name, count in agents:
                bar = "█" * min(count, 30) + "░" * max(0, 30 - count)
                print(f"  {C}{name:<20}{NC} {G}{bar}{NC} {count}")

    finally:
        bridge.close()


if __name__ == "__main__":
    main()
