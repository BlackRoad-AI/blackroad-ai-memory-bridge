"""Tests for memory bridge"""
import pytest
import sys
sys.path.insert(0, "../src")
from memory_bridge import remember, recall, search, observe, infer, verify_chain

def test_remember_and_recall(tmp_path, monkeypatch):
    import memory_bridge
    monkeypatch.setattr(memory_bridge, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory_bridge, "JOURNAL_FILE", tmp_path / "journal.jsonl")
    monkeypatch.setattr(memory_bridge, "LEDGER_FILE", tmp_path / "ledger.jsonl")
    
    entry = remember("test_key", "test_value", model="qwen")
    assert entry["key"] == "test_key"
    assert entry["value"] == "test_value"
    assert "hash" in entry
    
    results = recall("test_key")
    assert len(results) == 1
    assert results[0]["value"] == "test_value"

def test_chain_integrity(tmp_path, monkeypatch):
    import memory_bridge
    monkeypatch.setattr(memory_bridge, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory_bridge, "JOURNAL_FILE", tmp_path / "journal.jsonl")
    monkeypatch.setattr(memory_bridge, "LEDGER_FILE", tmp_path / "ledger.jsonl")
    
    remember("k1", "v1")
    remember("k2", "v2")
    remember("k3", "v3")
    valid, msg = verify_chain()
    assert valid, f"Chain should be valid: {msg}"
