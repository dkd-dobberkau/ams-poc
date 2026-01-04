"""Qdrant vector store client."""

import uuid

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from src.config import get_settings
from src.exceptions import SimilaritySearchError, VectorStoreError

logger = structlog.get_logger()


class QdrantClient:
    """Async Qdrant client for vector operations."""

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._url = url or settings.qdrant_url
        self._collection_name = collection_name or settings.qdrant_collection
        self._dimensions = settings.embedding_dimensions
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        """Initialize the Qdrant client."""
        try:
            self._client = AsyncQdrantClient(url=self._url)
            await self._ensure_collection()
            logger.info("Qdrant client connected", collection=self._collection_name)
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Qdrant: {e}") from e

    async def _ensure_collection(self) -> None:
        """Ensure the collection exists, create if not."""
        if not self._client:
            raise VectorStoreError("Client not initialized")

        collections = await self._client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self._collection_name not in collection_names:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self._dimensions,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            # Create payload index for user_id filtering
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="user_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="memory_type",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            logger.info("Qdrant collection created", collection=self._collection_name)

    async def disconnect(self) -> None:
        """Close the Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client disconnected")

    async def upsert(
        self,
        memory_id: uuid.UUID,
        user_id: str,
        memory_type: str,
        content: str,
        embedding: list[float],
        confidence: float,
    ) -> None:
        """Insert or update a vector in Qdrant."""
        if not self._client:
            raise VectorStoreError("Client not initialized")

        try:
            await self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=str(memory_id),
                        vector=embedding,
                        payload={
                            "user_id": user_id,
                            "memory_type": memory_type,
                            "content": content,
                            "confidence": confidence,
                        },
                    )
                ],
            )
            logger.debug("Vector upserted", memory_id=str(memory_id))
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert vector: {e}") from e

    async def search(
        self,
        user_id: str,
        embedding: list[float],
        limit: int = 10,
        memory_type: str | None = None,
        min_score: float = 0.0,
    ) -> list[qdrant_models.ScoredPoint]:
        """Search for similar vectors."""
        if not self._client:
            raise VectorStoreError("Client not initialized")

        try:
            # Build filter conditions
            must_conditions = [
                qdrant_models.FieldCondition(
                    key="user_id",
                    match=qdrant_models.MatchValue(value=user_id),
                )
            ]

            if memory_type:
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key="memory_type",
                        match=qdrant_models.MatchValue(value=memory_type),
                    )
                )

            results = await self._client.search(
                collection_name=self._collection_name,
                query_vector=embedding,
                query_filter=qdrant_models.Filter(must=must_conditions),
                limit=limit,
                score_threshold=min_score,
            )

            logger.debug("Vector search completed", results_count=len(results))
            return results

        except Exception as e:
            raise SimilaritySearchError(f"Similarity search failed: {e}") from e

    async def delete(self, memory_id: uuid.UUID) -> None:
        """Delete a vector by ID."""
        if not self._client:
            raise VectorStoreError("Client not initialized")

        try:
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[str(memory_id)],
                ),
            )
            logger.debug("Vector deleted", memory_id=str(memory_id))
        except Exception as e:
            raise VectorStoreError(f"Failed to delete vector: {e}") from e

    async def get_collection_info(self) -> qdrant_models.CollectionInfo:
        """Get collection information."""
        if not self._client:
            raise VectorStoreError("Client not initialized")
        return await self._client.get_collection(self._collection_name)


# Global instance
_qdrant_client: QdrantClient | None = None


async def get_qdrant() -> QdrantClient:
    """Get or create the global Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient()
        await _qdrant_client.connect()
    return _qdrant_client


async def close_qdrant() -> None:
    """Close the global Qdrant client."""
    global _qdrant_client
    if _qdrant_client:
        await _qdrant_client.disconnect()
        _qdrant_client = None
