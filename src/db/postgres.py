"""PostgreSQL connection and queries using asyncpg."""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from src.config import get_settings
from src.exceptions import ConnectionError, QueryError

logger = structlog.get_logger()


class PostgresClient:
    """Async PostgreSQL client with pgvector support."""

    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        self._database_url = database_url or settings.database_url
        # Convert asyncpg URL format
        self._database_url = self._database_url.replace("postgresql+asyncpg://", "postgresql://")
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=2,
                max_size=10,
                init=self._init_connection,
            )
            logger.info("PostgreSQL connection pool created")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize each connection with pgvector."""
        await register_vector(conn)

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self._pool:
            raise ConnectionError("Connection pool not initialized")
        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query without returning results."""
        try:
            async with self.acquire() as conn:
                return await conn.execute(query, *args)
        except Exception as e:
            raise QueryError(f"Query execution failed: {e}") from e

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Execute a query and return all results."""
        try:
            async with self.acquire() as conn:
                return await conn.fetch(query, *args)
        except Exception as e:
            raise QueryError(f"Query failed: {e}") from e

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Execute a query and return a single row."""
        try:
            async with self.acquire() as conn:
                return await conn.fetchrow(query, *args)
        except Exception as e:
            raise QueryError(f"Query failed: {e}") from e

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute a query and return a single value."""
        try:
            async with self.acquire() as conn:
                return await conn.fetchval(query, *args)
        except Exception as e:
            raise QueryError(f"Query failed: {e}") from e

    # Memory-specific queries
    async def insert_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        embedding: list[float],
        confidence: float = 0.70,
    ) -> uuid.UUID:
        """Insert a new memory entry."""
        query = """
            INSERT INTO memories (user_id, memory_type, content, content_embedding, confidence)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """
        memory_id = await self.fetchval(query, user_id, memory_type, content, embedding, confidence)
        logger.info("Memory inserted", memory_id=str(memory_id), memory_type=memory_type)
        return memory_id

    async def get_memory(self, memory_id: uuid.UUID) -> asyncpg.Record | None:
        """Get a memory by ID."""
        query = "SELECT * FROM memories WHERE id = $1"
        return await self.fetchrow(query, memory_id)

    async def get_user_memories(
        self,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[asyncpg.Record]:
        """Get all memories for a user, optionally filtered by type."""
        if memory_type:
            query = """
                SELECT * FROM memories
                WHERE user_id = $1 AND memory_type = $2
                ORDER BY updated_at DESC
                LIMIT $3
            """
            return await self.fetch(query, user_id, memory_type, limit)
        else:
            query = """
                SELECT * FROM memories
                WHERE user_id = $1
                ORDER BY updated_at DESC
                LIMIT $2
            """
            return await self.fetch(query, user_id, limit)

    async def update_confidence(
        self,
        memory_id: uuid.UUID,
        new_confidence: float,
        increment_evidence: bool = True,
    ) -> asyncpg.Record | None:
        """Update confidence score for a memory."""
        if increment_evidence:
            query = """
                UPDATE memories
                SET confidence = $2,
                    evidence_count = evidence_count + 1,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
            """
        else:
            query = """
                UPDATE memories
                SET confidence = $2, updated_at = NOW()
                WHERE id = $1
                RETURNING *
            """
        result = await self.fetchrow(query, memory_id, new_confidence)
        if result:
            logger.info(
                "Confidence updated",
                memory_id=str(memory_id),
                new_confidence=new_confidence,
            )
        return result

    async def semantic_search(
        self,
        user_id: str,
        embedding: list[float],
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[asyncpg.Record]:
        """Search memories by vector similarity."""
        if memory_type:
            query = """
                SELECT *, 1 - (content_embedding <=> $2) AS similarity
                FROM memories
                WHERE user_id = $1 AND memory_type = $3
                ORDER BY content_embedding <=> $2
                LIMIT $4
            """
            return await self.fetch(query, user_id, embedding, memory_type, limit)
        else:
            query = """
                SELECT *, 1 - (content_embedding <=> $2) AS similarity
                FROM memories
                WHERE user_id = $1
                ORDER BY content_embedding <=> $2
                LIMIT $3
            """
            return await self.fetch(query, user_id, embedding, limit)

    async def fulltext_search(
        self,
        user_id: str,
        search_query: str,
        limit: int = 10,
    ) -> list[asyncpg.Record]:
        """Search memories using full-text search (BM25-like via ts_rank)."""
        query = """
            SELECT *,
                   ts_rank_cd(to_tsvector('german', content), plainto_tsquery('german', $2)) AS rank
            FROM memories
            WHERE user_id = $1
              AND to_tsvector('german', content) @@ plainto_tsquery('german', $2)
            ORDER BY rank DESC
            LIMIT $3
        """
        return await self.fetch(query, user_id, search_query, limit)

    async def find_similar_opinion(
        self,
        user_id: str,
        embedding: list[float],
        threshold: float = 0.85,
    ) -> asyncpg.Record | None:
        """Find an existing opinion that's semantically similar."""
        query = """
            SELECT *, 1 - (content_embedding <=> $2) AS similarity
            FROM memories
            WHERE user_id = $1
              AND memory_type = 'opinion'
              AND 1 - (content_embedding <=> $2) >= $3
            ORDER BY content_embedding <=> $2
            LIMIT 1
        """
        return await self.fetchrow(query, user_id, embedding, threshold)

    async def delete_memory(self, memory_id: uuid.UUID) -> bool:
        """Delete a memory by ID."""
        query = "DELETE FROM memories WHERE id = $1"
        result = await self.execute(query, memory_id)
        return result == "DELETE 1"


# Global instance
_postgres_client: PostgresClient | None = None


async def get_postgres() -> PostgresClient:
    """Get or create the global PostgreSQL client."""
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresClient()
        await _postgres_client.connect()
    return _postgres_client


async def close_postgres() -> None:
    """Close the global PostgreSQL client."""
    global _postgres_client
    if _postgres_client:
        await _postgres_client.disconnect()
        _postgres_client = None
