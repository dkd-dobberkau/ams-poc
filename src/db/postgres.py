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

    async def find_similar_memory(
        self,
        user_id: str,
        embedding: list[float],
        memory_type: str,
        threshold: float = 0.90,
    ) -> asyncpg.Record | None:
        """Find an existing memory of any type that's semantically similar."""
        query = """
            SELECT *, 1 - (content_embedding <=> $2) AS similarity
            FROM memories
            WHERE user_id = $1
              AND memory_type = $3
              AND 1 - (content_embedding <=> $2) >= $4
            ORDER BY content_embedding <=> $2
            LIMIT 1
        """
        return await self.fetchrow(query, user_id, embedding, memory_type, threshold)

    async def delete_memory(self, memory_id: uuid.UUID) -> bool:
        """Delete a memory by ID."""
        query = "DELETE FROM memories WHERE id = $1"
        result = await self.execute(query, memory_id)
        return result == "DELETE 1"

    # Conversation methods
    async def create_conversation(
        self,
        user_id: str,
        title: str | None = None,
    ) -> uuid.UUID:
        """Create a new conversation."""
        query = """
            INSERT INTO conversations (user_id, title)
            VALUES ($1, $2)
            RETURNING id
        """
        return await self.fetchval(query, user_id, title)

    async def get_conversations(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[asyncpg.Record]:
        """Get all conversations for a user, ordered by most recent."""
        query = """
            SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at,
                   (SELECT content FROM chat_messages
                    WHERE conversation_id = c.id
                    ORDER BY created_at ASC LIMIT 1) as first_message
            FROM conversations c
            WHERE c.user_id = $1
            ORDER BY c.updated_at DESC
            LIMIT $2
        """
        return await self.fetch(query, user_id, limit)

    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
    ) -> asyncpg.Record | None:
        """Get a conversation by ID."""
        query = "SELECT * FROM conversations WHERE id = $1"
        return await self.fetchrow(query, conversation_id)

    async def update_conversation_title(
        self,
        conversation_id: uuid.UUID,
        title: str,
    ) -> None:
        """Update conversation title."""
        query = "UPDATE conversations SET title = $2 WHERE id = $1"
        await self.execute(query, conversation_id, title)

    async def delete_conversation(
        self,
        conversation_id: uuid.UUID,
    ) -> bool:
        """Delete a conversation and all its messages."""
        query = "DELETE FROM conversations WHERE id = $1"
        result = await self.execute(query, conversation_id)
        return result == "DELETE 1"

    # Chat message methods
    async def insert_chat_message(
        self,
        conversation_id: uuid.UUID,
        user_id: str,
        role: str,
        content: str,
        memories_used: list[dict] | None = None,
    ) -> uuid.UUID:
        """Insert a chat message."""
        import json
        query = """
            INSERT INTO chat_messages (conversation_id, user_id, role, content, memories_used)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """
        memories_json = json.dumps(memories_used or [])
        result = await self.fetchval(
            query, conversation_id, user_id, role, content, memories_json
        )
        # Update conversation's updated_at
        await self.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
            conversation_id
        )
        return result

    async def get_conversation_messages(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> list[asyncpg.Record]:
        """Get messages for a specific conversation."""
        query = """
            SELECT id, conversation_id, user_id, role, content, memories_used, created_at
            FROM chat_messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            LIMIT $2
        """
        return await self.fetch(query, conversation_id, limit)

    async def get_chat_history(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[asyncpg.Record]:
        """Get chat history for a user (legacy, returns all messages)."""
        query = """
            SELECT id, user_id, role, content, memories_used, created_at
            FROM chat_messages
            WHERE user_id = $1
            ORDER BY created_at ASC
            LIMIT $2
        """
        return await self.fetch(query, user_id, limit)

    async def clear_chat_history(self, user_id: str) -> int:
        """Clear all chat history for a user (deletes conversations too)."""
        # Delete conversations (cascade will delete messages)
        query = "DELETE FROM conversations WHERE user_id = $1"
        result = await self.execute(query, user_id)
        return int(result.split()[1]) if result else 0


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
