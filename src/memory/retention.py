"""Memory retention - extraction from conversations."""

import json
from uuid import UUID

import structlog

from src.config import get_settings
from src.db.postgres import PostgresClient, get_postgres
from src.db.qdrant import QdrantClient, get_qdrant
from src.exceptions import ClassificationError
from src.llm.client import ClaudeClient, EmbeddingClient, get_claude, get_embedding_client
from src.llm.prompts import MEMORY_EXTRACTION_SYSTEM, MEMORY_EXTRACTION_USER
from src.memory.models import (
    ConfidenceUpdate,
    Evidence,
    EvidenceType,
    ExtractedMemory,
    Memory,
    MemoryType,
    RetentionResult,
)
from src.memory.opinion import OpinionMemoryManager, get_opinion_manager

logger = structlog.get_logger()


class RetentionEngine:
    """
    Memory retention engine.

    Extracts and stores memories from conversations:
    - Facts: Stored directly with confidence 1.0
    - Opinions: Processed through OpinionMemoryManager
    - Experiences: Stored directly
    - Observations: Stored directly
    """

    def __init__(
        self,
        postgres: PostgresClient | None = None,
        qdrant: QdrantClient | None = None,
        claude: ClaudeClient | None = None,
        embedding: EmbeddingClient | None = None,
        opinion_manager: OpinionMemoryManager | None = None,
    ) -> None:
        self._postgres = postgres
        self._qdrant = qdrant
        self._claude = claude
        self._embedding = embedding
        self._opinion_manager = opinion_manager
        self._settings = get_settings()

    async def _get_postgres(self) -> PostgresClient:
        if self._postgres is None:
            self._postgres = await get_postgres()
        return self._postgres

    async def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = await get_qdrant()
        return self._qdrant

    def _get_claude(self) -> ClaudeClient:
        if self._claude is None:
            self._claude = get_claude()
        return self._claude

    def _get_embedding(self) -> EmbeddingClient:
        if self._embedding is None:
            self._embedding = get_embedding_client()
        return self._embedding

    async def _get_opinion_manager(self) -> OpinionMemoryManager:
        if self._opinion_manager is None:
            self._opinion_manager = await get_opinion_manager()
        return self._opinion_manager

    async def extract_memories(
        self,
        conversation: str,
    ) -> list[ExtractedMemory]:
        """
        Extract memories from a conversation using LLM.

        Returns list of extracted memories with types and confidence.
        """
        claude = self._get_claude()

        prompt = MEMORY_EXTRACTION_USER.format(conversation=conversation)

        try:
            response = await claude.complete_json(
                system=MEMORY_EXTRACTION_SYSTEM,
                user_message=prompt,
                max_tokens=2048,
            )

            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            data = json.loads(response)

            memories = []
            for item in data.get("memories", []):
                memories.append(
                    ExtractedMemory(
                        content=item["content"],
                        memory_type=MemoryType(item["memory_type"]),
                        confidence=float(item.get("confidence", 0.7)),
                    )
                )

            logger.info("Memories extracted", count=len(memories))
            return memories

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to extract memories", error=str(e))
            raise ClassificationError(f"Failed to extract memories: {e}") from e

    async def store_memory(
        self,
        user_id: str,
        extracted: ExtractedMemory,
    ) -> Memory | ConfidenceUpdate:
        """
        Store an extracted memory.

        Opinions are processed through OpinionMemoryManager.
        Other types are stored directly.
        """
        if extracted.memory_type == MemoryType.OPINION:
            # Use opinion manager for confidence tracking
            opinion_manager = await self._get_opinion_manager()
            result = await opinion_manager.process_statement(
                user_id=user_id,
                statement=extracted.content,
            )
            return result

        # Store non-opinion memories directly
        postgres = await self._get_postgres()
        qdrant = await self._get_qdrant()
        embedding_client = self._get_embedding()

        embedding = await embedding_client.embed(extracted.content)

        # Facts always have confidence 1.0
        confidence = 1.0 if extracted.memory_type == MemoryType.FACT else extracted.confidence

        memory_id = await postgres.insert_memory(
            user_id=user_id,
            memory_type=extracted.memory_type.value,
            content=extracted.content,
            embedding=embedding,
            confidence=confidence,
        )

        await qdrant.upsert(
            memory_id=memory_id,
            user_id=user_id,
            memory_type=extracted.memory_type.value,
            content=extracted.content,
            embedding=embedding,
            confidence=confidence,
        )

        record = await postgres.get_memory(memory_id)

        return Memory(
            id=record["id"],
            user_id=record["user_id"],
            content=record["content"],
            memory_type=record["memory_type"],
            confidence=float(record["confidence"]),
            evidence_count=record["evidence_count"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )

    async def process_conversation(
        self,
        user_id: str,
        conversation: str,
    ) -> RetentionResult:
        """
        Process a conversation and store all extracted memories.

        Returns extraction and update results.
        """
        # Extract memories from conversation
        extracted = await self.extract_memories(conversation)

        stored_memories: list[ExtractedMemory] = []
        confidence_updates: list[ConfidenceUpdate] = []

        for memory in extracted:
            result = await self.store_memory(user_id, memory)

            if isinstance(result, ConfidenceUpdate):
                confidence_updates.append(result)
            else:
                stored_memories.append(memory)

        logger.info(
            "Conversation processed",
            user_id=user_id,
            new_memories=len(stored_memories),
            confidence_updates=len(confidence_updates),
        )

        return RetentionResult(
            extracted_memories=stored_memories,
            updated_opinions=confidence_updates,
        )


# Global instance
_retention_engine: RetentionEngine | None = None


async def get_retention_engine() -> RetentionEngine:
    """Get or create the global retention engine."""
    global _retention_engine
    if _retention_engine is None:
        _retention_engine = RetentionEngine()
    return _retention_engine
