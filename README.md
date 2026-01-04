# Agent Memory System POV

A HINDSIGHT-inspired memory system for AI agents that enables persistent, evolving memory with confidence tracking.

## Overview

This proof-of-value demonstrates that structured memory architectures enable more consistent AI agents compared to standard RAG approaches. The system implements two core components from the HINDSIGHT paper:

1. **Opinion Memory with Confidence Tracking** - Opinions are stored with confidence scores that evolve based on new evidence
2. **Hybrid-Recall with RRF** - Combines semantic search, BM25, and recency scoring via Reciprocal Rank Fusion

## Features

- **Memory Types**: Facts, opinions, experiences, and observations with distinct handling
- **Confidence Evolution**: Non-linear updates based on evidence count and type (reinforce/weaken/contradict)
- **Hybrid Retrieval**: Query-type-adaptive weighting for optimal recall
- **Token Budget Management**: Respects context limits when recalling memories
- **Automatic Memory Extraction**: LLM-powered extraction from conversations

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# - ANTHROPIC_API_KEY
# - OPENAI_API_KEY

# Start services
docker compose up -d

# Run the server
uv run fastapi dev src/main.py

# Run tests
uv run pytest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Chat with memory integration |
| `/memories/{user_id}` | GET | List all memories for a user |
| `/memories/{user_id}/opinions` | GET | List opinions with confidence |
| `/memories/{user_id}/sync` | POST | Extract memories from conversation |
| `/memories/{user_id}/recall` | POST | Query memories with hybrid search |

### Example: Chat with Memory

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "message": "I love Python for data science!"}'
```

Response includes:
- Agent response informed by memories
- List of memories used for context
- Any confidence updates from the conversation

## Architecture

```
src/
├── main.py              # FastAPI entrypoint
├── config.py            # Settings (pydantic-settings)
├── exceptions.py        # Exception hierarchy
├── api/
│   ├── routes.py        # API endpoints
│   └── schemas.py       # Request/response models
├── db/
│   ├── postgres.py      # PostgreSQL + pgvector
│   └── qdrant.py        # Qdrant vector store
├── llm/
│   ├── client.py        # Claude + OpenAI clients
│   └── prompts.py       # System prompts
└── memory/
    ├── models.py        # Domain models
    ├── opinion.py       # Opinion memory manager
    ├── recall.py        # Hybrid-recall engine
    └── retention.py     # Memory extraction
```

## Core Concepts

### Confidence Update Formula

Opinions become more stable over time (harder to change with more evidence):

```python
stability_factor = 1 / (1 + log(evidence_count + 1))
type_multiplier = {"reinforce": 1.0, "weaken": 1.3, "contradict": 2.0}
effective_change = impact * stability_factor * type_multiplier[evidence_type]

if evidence_type == "reinforce":
    new_confidence = confidence + (1 - confidence) * effective_change
else:
    new_confidence = confidence * (1 - effective_change)
```

### Reciprocal Rank Fusion

Combines multiple retrieval methods with query-type-adaptive weights:

```python
WEIGHT_PROFILES = {
    "temporal": {"semantic": 0.2, "bm25": 0.2, "recency": 0.6},
    "factual":  {"semantic": 0.5, "bm25": 0.4, "recency": 0.1},
    "opinion":  {"semantic": 0.6, "bm25": 0.2, "recency": 0.2},
    "general":  {"semantic": 0.4, "bm25": 0.35, "recency": 0.25},
}
```

## Tech Stack

- **Backend**: Python 3.11+ / FastAPI
- **Database**: PostgreSQL 16 with pgvector
- **Vector Store**: Qdrant
- **LLM**: Claude API (Anthropic)
- **Embeddings**: text-embedding-3-small (OpenAI)

## Development

```bash
# Linting
uv run ruff check src/

# Formatting
uv run ruff format src/

# Type checking
uv run mypy src/

# Tests with coverage
uv run pytest --cov=src
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `ANTHROPIC_API_KEY` | Claude API key | - |
| `OPENAI_API_KEY` | OpenAI API key (embeddings) | - |
| `DEFAULT_TOKEN_BUDGET` | Max tokens for recall | `2000` |
| `SIMILARITY_THRESHOLD` | Opinion deduplication threshold | `0.85` |
| `CONFIDENCE_DECAY_RATE` | Daily decay for unused opinions | `0.01` |

## Limitations (POV Scope)

- No graph traversal for entity relationships
- No personality profiles in reflect phase
- Single-user focus (no multi-tenancy optimization)
- German language configured for full-text search

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [HINDSIGHT Paper](https://arxiv.org/abs/2412.06256) - Original research
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL vector extension
- [Qdrant](https://qdrant.tech/) - Vector database
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - RRF paper
