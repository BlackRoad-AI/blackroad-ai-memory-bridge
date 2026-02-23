"""Tests for src/memory_bridge.py — BlackRoad AI Memory Bridge."""
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from memory_bridge import (
    MemoryEntry, VectorEmbedding, SearchResult,
    ConsolidationReport, MemoryBridge, _embed, _cosine,
)

EMBED_DIM = 64


@pytest.fixture
def bridge(tmp_path):
    b = MemoryBridge(db_path=tmp_path / "test_memory.db")
    yield b
    b.close()


# ── embedding helpers ─────────────────────────────────────────────────────────
def test_embed_returns_correct_dimension():
    vec = _embed("hello world")
    assert len(vec) == EMBED_DIM


def test_embed_returns_unit_vector():
    vec = _embed("unit vector test")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_embed_is_deterministic():
    assert _embed("same text") == _embed("same text")


def test_embed_different_inputs_differ():
    assert _embed("cats and dogs") != _embed("quantum computing")


def test_cosine_identical_vectors():
    v = _embed("test sentence")
    assert abs(_cosine(v, v) - 1.0) < 1e-5


def test_cosine_bounded():
    v1, v2 = _embed("alpha"), _embed("omega")
    sim = _cosine(v1, v2)
    assert -1.0 <= sim <= 1.0


# ── memory storage ────────────────────────────────────────────────────────────
def test_store_memory_adds_to_index(bridge):
    entry = MemoryEntry(agent_id="cece", content="The user prefers dark mode.", importance=0.8)
    result = bridge.store_memory(entry)
    assert result.memory_id in bridge._vectors
    assert len(bridge._vectors[result.memory_id]) == EMBED_DIM


def test_store_memory_persists_to_db(bridge):
    entry = MemoryEntry(agent_id="alice", content="Deploy at midnight.")
    bridge.store_memory(entry)
    row = bridge._conn.execute(
        "SELECT content FROM memories WHERE memory_id=?", (entry.memory_id,)
    ).fetchone()
    assert row is not None
    assert row[0] == "Deploy at midnight."


def test_store_memory_types(bridge):
    for mtype in ["episodic", "semantic", "working", "procedural"]:
        e = MemoryEntry(agent_id="bot", content=f"{mtype} memory", memory_type=mtype)
        bridge.store_memory(e)
    memories = bridge.list_memories(agent_id="bot")
    types = {m.memory_type for m in memories}
    assert types == {"episodic", "semantic", "working", "procedural"}


# ── listing ───────────────────────────────────────────────────────────────────
def test_list_memories_empty(bridge):
    assert bridge.list_memories(agent_id="nobody") == []


def test_list_memories_sorted_by_importance(bridge):
    bridge.store_memory(MemoryEntry(agent_id="a1", content="Low", importance=0.2))
    bridge.store_memory(MemoryEntry(agent_id="a1", content="High", importance=0.9))
    bridge.store_memory(MemoryEntry(agent_id="a1", content="Mid", importance=0.5))
    memories = bridge.list_memories(agent_id="a1")
    assert memories[0].importance >= memories[1].importance >= memories[2].importance


def test_list_memories_filter_by_type(bridge):
    bridge.store_memory(MemoryEntry(agent_id="a1", content="Ep", memory_type="episodic"))
    bridge.store_memory(MemoryEntry(agent_id="a1", content="Sem", memory_type="semantic"))
    episodic = bridge.list_memories(agent_id="a1", mtype="episodic")
    assert len(episodic) == 1
    assert episodic[0].memory_type == "episodic"


# ── semantic search ───────────────────────────────────────────────────────────
def test_search_returns_list(bridge):
    bridge.store_memory(MemoryEntry(agent_id="bot", content="Python programming language"))
    results = bridge.search_semantic("code", agent_id="bot", top_k=5, min_similarity=0.0)
    assert isinstance(results, list)


def test_search_increments_access_count(bridge):
    entry = MemoryEntry(agent_id="bob", content="access counter test data")
    bridge.store_memory(entry)
    bridge.search_semantic("access counter", agent_id="bob", top_k=5, min_similarity=0.0)
    bridge.search_semantic("access counter", agent_id="bob", top_k=5, min_similarity=0.0)
    memories = bridge.list_memories(agent_id="bob")
    assert any(m.access_count > 0 for m in memories)


def test_search_respects_top_k(bridge):
    for i in range(10):
        bridge.store_memory(MemoryEntry(agent_id="topk", content=f"memory item {i}"))
    results = bridge.search_semantic("memory", agent_id="topk", top_k=3, min_similarity=0.0)
    assert len(results) <= 3


# ── context building ──────────────────────────────────────────────────────────
def test_get_context_returns_string(bridge):
    bridge.store_memory(MemoryEntry(agent_id="ctx", content="Important context info about deployment"))
    ctx = bridge.get_context("ctx", "deployment", max_tokens=512)
    assert isinstance(ctx, str)


def test_get_context_empty_agent(bridge):
    ctx = bridge.get_context("no-memories-agent", "query")
    assert ctx == ""


# ── expiry & consolidation ────────────────────────────────────────────────────
def test_clear_expired_removes_past(bridge):
    past = (datetime.utcnow() - timedelta(seconds=10)).isoformat()
    entry = MemoryEntry(agent_id="expire-agent", content="Expired", expires_at=past)
    bridge.store_memory(entry)
    n = bridge.clear_expired()
    assert n == 1
    assert bridge.list_memories(agent_id="expire-agent") == []


def test_clear_expired_keeps_future(bridge):
    future = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    entry = MemoryEntry(agent_id="keep-agent", content="Still valid", expires_at=future)
    bridge.store_memory(entry)
    n = bridge.clear_expired()
    assert n == 0
    assert len(bridge.list_memories(agent_id="keep-agent")) == 1


def test_consolidate_prunes_expired(bridge):
    past = (datetime.utcnow() - timedelta(seconds=10)).isoformat()
    bridge.store_memory(MemoryEntry(agent_id="cons", content="Old memory", expires_at=past))
    bridge.store_memory(MemoryEntry(agent_id="cons", content="Fresh memory"))
    report = bridge.consolidate_memories("cons")
    assert report.memories_pruned == 1


def test_consolidate_promotes_high_access(bridge):
    entry = MemoryEntry(agent_id="promo", content="Accessed many times", memory_type="episodic")
    bridge.store_memory(entry)
    bridge._conn.execute(
        "UPDATE memories SET access_count=5 WHERE memory_id=?", (entry.memory_id,)
    )
    bridge._conn.commit()
    report = bridge.consolidate_memories("promo")
    assert report.memories_promoted >= 1
    mem = bridge.list_memories(agent_id="promo")
    assert mem[0].memory_type == "semantic"


def test_consolidation_report_fields(bridge):
    bridge.store_memory(MemoryEntry(agent_id="rep", content="Memory for report"))
    report = bridge.consolidate_memories("rep")
    assert report.memories_scanned >= 1
    assert report.duration_ms >= 0
    assert report.agent_id == "rep"
