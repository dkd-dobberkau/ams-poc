# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projektübersicht

HINDSIGHT-inspiriertes Memory-System für KI-Agenten mit zwei Kernkomponenten:

1. **Opinion Memory mit Konfidenz-Tracking** - Meinungen mit evolutionär angepassten Konfidenzwerten
2. **Hybrid-Recall** - Kombination aus semantischer Suche und BM25 via Reciprocal Rank Fusion

POV (Proof of Value) zum Nachweis, dass strukturierte Memory-Architekturen konsistentere Agenten ermöglichen als Standard-RAG.

## Build & Development Commands

```bash
# Setup
uv sync                              # Dependencies installieren
docker compose up -d                 # Services starten (PostgreSQL, Qdrant)
uv run alembic upgrade head          # DB-Migrationen ausführen

# Development
uv run fastapi dev src/main.py       # Server mit Hot-Reload auf localhost:8000

# Tests
uv run pytest                        # Alle Tests ausführen
uv run pytest tests/test_opinion.py  # Einzelne Testdatei
uv run pytest -k "test_confidence"   # Tests nach Pattern filtern
uv run pytest -x                     # Bei erstem Fehler stoppen

# Linting & Formatting
uv run ruff check src/               # Linting
uv run ruff check src/ --fix         # Auto-Fix
uv run ruff format src/              # Formatting

# Debugging & Utilities
docker compose logs -f postgres      # PostgreSQL-Logs
docker compose logs -f qdrant        # Qdrant-Logs
uv run python -m src.scripts.seed_data  # Test-Daten generieren
```

## Tech-Stack

- **Backend:** Python 3.11+ mit FastAPI
- **Datenbank:** PostgreSQL 16 mit pgvector Extension
- **Vector Store:** Qdrant (self-hosted via Docker)
- **LLM:** Claude API (Anthropic)
- **Embeddings:** text-embedding-3-small (OpenAI) oder voyage-3 (Voyage AI)
- **Orchestration:** Eigene Implementierung (kein LangChain)

## Architecture

### Projektstruktur

```
src/
├── main.py              # FastAPI Entrypoint
├── config.py            # Settings via pydantic-settings
├── exceptions.py        # Exception-Hierarchie
├── memory/
│   ├── models.py        # Pydantic Models & DB Schemas
│   ├── opinion.py       # Opinion Memory Manager
│   ├── retention.py     # Memory-Extraktion aus Conversations
│   └── recall.py        # Hybrid-Recall Engine
├── db/
│   ├── postgres.py      # PostgreSQL Connection & Queries
│   └── qdrant.py        # Qdrant Vector Store Client
├── llm/
│   ├── client.py        # Claude API Wrapper
│   └── prompts.py       # System Prompts & Templates
└── api/
    ├── routes.py        # API Endpoints
    └── schemas.py       # Request/Response Models
```

### Kernkonzepte

**Opinion Memory Konfidenz-Formel:**
```python
stability_factor = 1 / (1 + log(evidence_count + 1))
type_multiplier = {"reinforce": 1.0, "weaken": 1.3, "contradict": 2.0}
effective_change = impact * stability_factor * type_multiplier[evidence_type]

if evidence_type == "reinforce":
    new_confidence = confidence + (1 - confidence) * effective_change
else:
    new_confidence = confidence * (1 - effective_change)
```

**Reciprocal Rank Fusion:**
```python
rrf_score = sum(weight * (1 / (k + rank + 1)) for each retrieval method)
# k = 60 (Standard)
```

**Query-Typ-abhängige Hybrid-Recall Gewichtung:**
```python
WEIGHT_PROFILES = {
    "temporal": {"semantic": 0.2, "bm25": 0.2, "recency": 0.6},
    "factual":  {"semantic": 0.5, "bm25": 0.4, "recency": 0.1},
    "opinion":  {"semantic": 0.6, "bm25": 0.2, "recency": 0.2},
    "general":  {"semantic": 0.4, "bm25": 0.35, "recency": 0.25}
}
```

### Memory-Typen

| Typ | Beschreibung | Beispiel |
|-----|--------------|----------|
| `fact` | Verifizierbare Weltfakten | "Alice arbeitet bei Google" |
| `opinion` | Subjektive Meinung mit Konfidenz | "Python ist gut für Data Science" (0.85) |
| `experience` | Agenten-Handlungen | "Ich empfahl Alice Yosemite" |
| `observation` | Neutrale Zusammenfassungen | "Alice ist ML-Spezialistin" |

## Coding-Konventionen

- **Formatter:** Ruff (Format + Lint)
- **Type Hints:** Überall, strikt
- **Docstrings:** Google Style für öffentliche Funktionen
- **Async:** Konsequent async/await für I/O-Operationen
- **Logging:** Strukturiert mit `structlog`

```python
async def process_opinion(
    user_id: str,
    statement: str,
    context: str | None = None
) -> OpinionUpdateResult:
    """
    Verarbeitet eine neue Meinungsäußerung.

    Args:
        user_id: Eindeutige Nutzer-ID
        statement: Die zu verarbeitende Aussage
        context: Optionaler Konversationskontext

    Returns:
        OpinionUpdateResult mit Action und neuer Konfidenz

    Raises:
        DatabaseError: Bei Verbindungsproblemen
    """
```

### Namenskonventionen

- **Dateien:** snake_case (`opinion_memory.py`)
- **Klassen:** PascalCase (`OpinionMemoryManager`)
- **Funktionen/Variablen:** snake_case (`update_confidence`)
- **Konstanten:** UPPER_SNAKE_CASE (`DEFAULT_CONFIDENCE`)
- **Pydantic Models:** PascalCase mit Suffix (`OpinionCreate`, `MemoryResponse`)

### Exception-Hierarchie

```python
class MemoryError(Exception):
    """Basis für Memory-System Fehler."""

class OpinionNotFoundError(MemoryError):
    """Opinion existiert nicht."""

class ConfidenceUpdateError(MemoryError):
    """Fehler beim Konfidenz-Update."""
```

## Datenbank-Schema

```sql
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    memory_type VARCHAR(50) NOT NULL CHECK (memory_type IN ('fact', 'opinion', 'experience', 'observation')),
    content TEXT NOT NULL,
    content_embedding VECTOR(1536),
    confidence DECIMAL(3,2) DEFAULT 0.70,
    evidence_count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX idx_memories_user_type ON memories(user_id, memory_type);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (content_embedding vector_cosine_ops);
CREATE INDEX idx_memories_fulltext ON memories USING gin(to_tsvector('german', content));
```

## API Endpoints

```
POST /chat                         # Haupt-Chat-Endpoint mit Memory
GET  /memories/{user_id}           # Alle Memories eines Users
GET  /memories/{user_id}/opinions  # Nur Opinions mit Konfidenz
POST /memories/{user_id}/sync      # Manueller Memory-Sync
```

Response-Format:
```json
{
  "response": "Agent-Antwort...",
  "memories_used": [{"id": "...", "content": "...", "relevance_score": 0.87}],
  "confidence_updates": [{"opinion": "...", "old": 0.70, "new": 0.78, "action": "reinforced"}]
}
```

## Wichtige Implementierungsdetails

### Opinion-Ähnlichkeit erkennen

Bevor eine neue Opinion gespeichert wird, prüfen ob semantisch ähnliche existiert:
- Threshold: 0.85 Cosine Similarity
- Bei Match: Konfidenz updaten statt neu anlegen
- LLM klassifiziert Beziehung (reinforce/weaken/contradict)

### Token-Budget Management

- Default: 2000 Tokens
- Priorisierung nach RRF-Score
- Truncation bei Überschreitung

## Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/memory_db
QDRANT_URL=http://localhost:6333
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...  # Für Embeddings

# Optional
LOG_LEVEL=INFO
DEFAULT_TOKEN_BUDGET=2000
CONFIDENCE_DECAY_RATE=0.01  # Pro Tag ohne Reinforcement
```

## Bekannte Einschränkungen (POV-Scope)

1. **Keine Graph-Traversierung** - Entity-Relationen vorbereitet aber nicht implementiert
2. **Keine Persönlichkeitsprofile** - Reflect-Phase ohne konfigurierbare Disposition
3. **Single-User-Fokus** - Keine Multi-Tenancy-Optimierung
4. **Deutsche Sprache** - Full-Text-Search auf Deutsch konfiguriert
