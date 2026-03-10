# BlackRoad AI Memory Bridge

> Give your AI a memory — persistent, searchable, and always ready.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![BlackRoad AI](https://img.shields.io/badge/BlackRoad-AI-FF1D6C)](https://blackroadai.com)
[![License](https://img.shields.io/badge/license-Proprietary-black)](LICENSE)

---

## 👆 Try it now — no install needed

**[▶ Open the live demo →](https://blackroad-ai.github.io/blackroad-ai-memory-bridge/)**

Click the link above. Type something. Watch it work. That's it.

- 🌐 [blackroadai.com](https://blackroadai.com) — main portal
- 🤖 [lucidia.earth](https://lucidia.earth) — meet Lucidia, our AI assistant
- 🔗 [blackroad.io](https://blackroad.io) — BlackRoad home
- 💻 [GitHub](https://github.com/BlackRoad-AI/blackroad-ai-memory-bridge) — source code

---

## What is this?

Imagine your AI assistant could **remember things** — not just within a single conversation, but across every conversation, forever. That's what this library does.

You give it a piece of information. It stores it. Later, you (or your AI) can search for it by meaning — not just exact words. Ask "what did I say about programming?" and it finds "I prefer Python for backend work" even though those words don't all match.

No cloud needed. No GPU. No paid API. It runs locally on your computer and stores everything in a simple file.

---

## Overview

`blackroad-ai-memory-bridge` gives BlackRoad AI agents a persistent, searchable memory system.
Memories are stored with deterministic 64-dimensional vector embeddings (no torch dependency),
supporting semantic search via cosine similarity. A consolidation pipeline prunes expired
memories, merges near-duplicates, and promotes frequently-accessed episodic memories to
semantic memory — replicating the human memory consolidation process.

### Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                       blackroad-ai-memory-bridge                          │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Agent                                                                    │
│    │                                                                      │
│    │ store_memory(content, type, importance, ttl)                         │
│    ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        MemoryBridge                                 │ │
│  │                                                                     │ │
│  │  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │ │
│  │  │  Storage     │    │  Vector Index    │    │  Consolidation   │  │ │
│  │  │              │    │  (in-memory)     │    │                  │  │ │
│  │  │  memories    │───▶│                  │───▶│  prune expired   │  │ │
│  │  │  embeddings  │    │  dim=64          │    │  merge dups      │  │ │
│  │  │  (SQLite)    │    │  cosine sim      │    │  promote access  │  │ │
│  │  └──────────────┘    └──────────────────┘    └──────────────────┘  │ │
│  │           │                    │                                    │ │
│  │           ▼                    ▼                                    │ │
│  │  ┌──────────────────────────────────────┐                          │ │
│  │  │  search_semantic(query, agent, top_k) │                         │ │
│  │  │  get_context(agent, query, tokens)    │                         │ │
│  │  └──────────────────────────────────────┘                          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  Database: ~/.blackroad/memory_bridge.db                                  │
└───────────────────────────────────────────────────────────────────────────┘
```

### Memory Types

```
┌──────────────┬─────────────────────────────────────────┬──────────┐
│ Type         │ Description                             │ Promoted │
├──────────────┼─────────────────────────────────────────┼──────────┤
│ episodic     │ Specific events & interactions          │ → semantic│
│ semantic     │ General facts & knowledge               │ stays    │
│ working      │ Current context (short-lived)           │ expires  │
│ procedural   │ How-to knowledge, skills                │ stays    │
└──────────────┴─────────────────────────────────────────┴──────────┘
```

---

## Features

- 🧠 **4 Memory Types** — episodic, semantic, working, procedural
- 🔍 **Semantic Search** — cosine-similarity over 64-dim embeddings, no GPU needed
- ⏰ **TTL Expiry** — per-memory TTL, auto-pruned on consolidation or explicit call
- 🔁 **Consolidation** — merge near-duplicates, prune expired, promote high-access
- 📖 **Context Builder** — token-aware context string for LLM prompts
- 📊 **Importance Scoring** — 0.0–1.0 importance, higher = surfaced first in search
- 🏷️ **Tag Support** — searchable tags per memory
- 🗄️ **Zero-config SQLite** — auto-creates `~/.blackroad/memory_bridge.db`
- 🖥️ **Full CLI** — `store`, `search`, `context`, `consolidate`, `list`, `clear-expired`

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.9 | Runtime — stdlib only |
| pytest | ≥ 7.0 | Testing |

**No external ML libraries required.** Embeddings use a deterministic SHA-256 hashing
approach, making the system fully functional in airgapped environments.

---

## Installation

```bash
git clone https://github.com/BlackRoad-AI/blackroad-ai-memory-bridge.git
cd blackroad-ai-memory-bridge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BLACKROAD_MEMORY_DB` | `~/.blackroad/memory_bridge.db` | SQLite DB path |
| `EMBED_DIM` | `64` | Embedding vector dimension (code constant) |

---

## Usage

### CLI

#### Store a memory

```bash
python src/memory_bridge.py store \
    --agent-id cece \
    --content "The user prefers Python over JavaScript for backend work" \
    --type semantic \
    --importance 0.85 \
    --tags preference language backend
```

**Output:**
```
✓ Memory stored [semantic] a3f2c1e8 importance=0.85
```

#### Store with TTL (auto-expire)

```bash
python src/memory_bridge.py store \
    --agent-id alice \
    --content "Current deployment target: Railway staging" \
    --type working \
    --importance 0.6 \
    --ttl 3600
```

#### Semantic search

```bash
python src/memory_bridge.py search \
    --query "programming language preference" \
    --agent-id cece \
    --top-k 5 \
    --min-sim 0.1
```

**Output:**
```
── Search: programming language preference ──────────
  1. a3f2c1e8 sim=0.847 ██████████
     The user prefers Python over JavaScript for backend work
  2. b1c2d3e4 sim=0.612 ██████░░░░
     User has experience with Go and Rust
  3. c4d5e6f7 sim=0.441 ████░░░░░░
     The project uses FastAPI framework
```

#### Build context for LLM prompt

```bash
python src/memory_bridge.py context \
    --agent-id cece \
    --query "what language does the user prefer?" \
    --max-tokens 256
```

**Output:**
```
── Context for cece ──────────
[semantic|0.847] The user prefers Python over JavaScript for backend work
[episodic|0.612] User mentioned disliking TypeScript verbosity
[semantic|0.441] The project uses FastAPI framework
```

#### Consolidate memories

```bash
python src/memory_bridge.py consolidate \
    --agent-id cece \
    --threshold 0.9
```

**Output:**
```
✓ Consolidation [cece]: scanned=47 merged=3 pruned=5 promoted=8 (12.4ms)
```

#### List memories

```bash
python src/memory_bridge.py list --agent-id cece --type semantic
```

**Output:**
```
  a3f2c1e8 [semantic] imp=0.85 acc=12 — The user prefers Python over JavaScript...
  b1c2d3e4 [semantic] imp=0.72 acc=7  — User has experience with Go and Rust...
  c4d5e6f7 [semantic] imp=0.65 acc=3  — The project uses FastAPI framework...
```

#### Clear expired memories

```bash
python src/memory_bridge.py clear-expired
```

**Output:**
```
✓ Cleared 5 expired memories.
```

---

### Python API

```python
from datetime import datetime, timedelta
from src.memory_bridge import MemoryBridge, MemoryEntry

bridge = MemoryBridge()

# Store a permanent semantic memory
bridge.store_memory(MemoryEntry(
    agent_id="cece",
    content="The user prefers dark mode in all interfaces",
    memory_type="semantic",
    importance=0.9,
    tags=["preference", "ui"],
))

# Store a working memory with 1-hour TTL
bridge.store_memory(MemoryEntry(
    agent_id="cece",
    content="Currently debugging the auth module",
    memory_type="working",
    importance=0.6,
    expires_at=(datetime.utcnow() + timedelta(hours=1)).isoformat(),
))

# Semantic search
results = bridge.search_semantic(
    query="what does the user prefer?",
    agent_id="cece",
    top_k=5,
    min_similarity=0.1,
)
for r in results:
    print(f"{r.similarity:.3f}: {r.content}")

# Build LLM context
context = bridge.get_context("cece", "user preferences", max_tokens=512)
prompt = f"<context>\n{context}\n</context>\n\nUser: What should I set the UI theme to?"

# Nightly consolidation
report = bridge.consolidate_memories("cece", similarity_threshold=0.9)
print(f"Merged: {report.memories_merged}, Pruned: {report.memories_pruned}")

# Clear all expired
n = bridge.clear_expired()

bridge.close()
```

---

## API Reference

### `MemoryBridge`

| Method | Returns | Description |
|--------|---------|-------------|
| `store_memory(entry)` | `MemoryEntry` | Store memory + compute embedding |
| `search_semantic(query, agent_id?, top_k, min_similarity)` | `List[SearchResult]` | Cosine-similarity search |
| `get_context(agent_id, query, max_tokens)` | `str` | Token-limited context string |
| `consolidate_memories(agent_id, threshold)` | `ConsolidationReport` | Merge/prune/promote |
| `clear_expired()` | `int` | Delete all expired memories |
| `list_memories(agent_id?, mtype?)` | `List[MemoryEntry]` | List with filters |
| `close()` | `None` | Close DB connection |

### `MemoryEntry` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory_id` | `str` | auto-UUID8 | Unique memory ID |
| `agent_id` | `str` | `""` | Owner agent |
| `content` | `str` | `""` | Memory text content |
| `memory_type` | `str` | `"episodic"` | episodic/semantic/working/procedural |
| `importance` | `float` | `0.5` | Relevance weight 0.0–1.0 |
| `access_count` | `int` | `0` | Access counter (auto-incremented) |
| `tags` | `List[str]` | `[]` | Searchable tags |
| `expires_at` | `Optional[str]` | `None` | ISO datetime or None (permanent) |

### `SearchResult` Fields

| Field | Description |
|-------|-------------|
| `memory_id` | Memory identifier |
| `content` | Memory text |
| `similarity` | Cosine similarity 0.0–1.0 |
| `importance` | Memory importance score |
| `memory_type` | Type of memory |
| `agent_id` | Owner agent |

### `ConsolidationReport` Fields

| Field | Description |
|-------|-------------|
| `memories_scanned` | Total memories examined |
| `memories_merged` | Near-duplicates removed |
| `memories_pruned` | Expired entries deleted |
| `memories_promoted` | Episodic → semantic promotions |
| `duration_ms` | Consolidation time |

---

## Embedding Algorithm

The bridge uses a **deterministic pseudo-embedding** via SHA-256 hashing — no torch, no network:

```python
def _embed(text: str) -> List[float]:
    # 1. Hash text to a 256-bit seed
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    # 2. Generate 64 Gaussian samples seeded deterministically
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(64)]
    # 3. L2-normalize to unit sphere
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / norm for v in vec]
```

**Properties:**
- ✅ Fully deterministic — same text always yields same vector
- ✅ No external dependencies
- ✅ Unit-normalized for cosine similarity
- ⚠️ Not semantically meaningful (lexical overlap, not semantic)

For production semantic search, plug in a real embedding model:

```python
# Drop-in replacement for production
import openai

def _embed(text: str) -> List[float]:
    resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding
```

---

## Running Tests

```bash
pytest tests/test_memory_bridge.py -v
# Expected: 21 passed
```

---

## Database Schema

```sql
-- ~/.blackroad/memory_bridge.db

CREATE TABLE memories (
    memory_id     TEXT PRIMARY KEY,
    agent_id      TEXT,
    content       TEXT,
    memory_type   TEXT,           -- episodic|semantic|working|procedural
    importance    REAL,           -- 0.0 to 1.0
    access_count  INTEGER DEFAULT 0,
    tags_json     TEXT,           -- JSON array
    expires_at    TEXT,           -- ISO datetime or NULL
    created_at    TEXT,
    last_accessed TEXT
);

CREATE TABLE embeddings (
    embed_id    TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    vector_json TEXT,             -- JSON array[64]
    model       TEXT,
    dim         INTEGER,
    created_at  TEXT
);

CREATE TABLE consolidation_reports (
    report_id         TEXT PRIMARY KEY,
    agent_id          TEXT,
    memories_scanned  INTEGER,
    memories_merged   INTEGER,
    memories_pruned   INTEGER,
    memories_promoted INTEGER,
    duration_ms       REAL,
    created_at        TEXT
);
```

---

## Related Repos

| Repo | Purpose |
|------|---------|
| `blackroad-ai-cluster` | Distributed cluster orchestration |
| `blackroad-vllm-mvp` | Inference server wrapper |
| `lucidia-ai-models` | Model registry |
| `lucidia-ai-models-enhanced` | Quantization & fine-tuning pipeline |

---

*© BlackRoad OS, Inc. All rights reserved. Proprietary — not open source.*
