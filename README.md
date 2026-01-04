# Agent Memory System POV

HINDSIGHT-inspired Memory System for AI Agents.

## Features

- **Opinion Memory with Confidence Tracking** - Opinions stored with evolutionary confidence values
- **Hybrid-Recall** - Semantic search + BM25 via Reciprocal Rank Fusion

## Quick Start

```bash
uv sync
cp .env.example .env
docker compose up -d
uv run fastapi dev src/main.py
```

## Tech Stack

- Python 3.11+ / FastAPI
- PostgreSQL 16 + pgvector
- Qdrant
- Claude API + OpenAI Embeddings
