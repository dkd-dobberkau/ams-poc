"""Hybrid-Recall Engine using Reciprocal Rank Fusion."""

from datetime import datetime, timezone
from uuid import UUID

import structlog
import tiktoken

from src.config import get_settings
from src.db.postgres import PostgresClient, get_postgres
from src.exceptions import RecallError
from src.llm.client import ClaudeClient, EmbeddingClient, get_claude, get_embedding_client
from src.llm.prompts import QUERY_CLASSIFICATION_SYSTEM
from src.memory.models import (
    MemoryWithScore,
    QueryType,
    RecallQuery,
    RecallResult,
    WEIGHT_PROFILES,
)

logger = structlog.get_logger()


class HybridRecallEngine:
    """
    Hybrid memory recall using Reciprocal Rank Fusion (RRF).

    Combines:
    - Semantic search (vector similarity)
    - BM25 keyword search (full-text)
    - Recency scoring (temporal decay)

    RRF formula: score = sum(weight * (1 / (k + rank + 1)))
    """

    def __init__(
        self,
        postgres: PostgresClient | None = None,
        claude: ClaudeClient | None = None,
        embedding: EmbeddingClient | None = None,
    ) -> None:
        self._postgres = postgres
        self._claude = claude
        self._embedding = embedding
        self._settings = get_settings()
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    async def _get_postgres(self) -> PostgresClient:
        if self._postgres is None:
            self._postgres = await get_postgres()
        return self._postgres

    def _get_claude(self) -> ClaudeClient:
        if self._claude is None:
            self._claude = get_claude()
        return self._claude

    def _get_embedding(self) -> EmbeddingClient:
        if self._embedding is None:
            self._embedding = get_embedding_client()
        return self._embedding

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))

    async def classify_query_type(self, query: str) -> QueryType:
        """
        Classify query type for optimal retrieval weighting.

        Uses LLM to determine if query is temporal, factual, opinion, or general.
        """
        claude = self._get_claude()

        try:
            response = await claude.complete(
                system=QUERY_CLASSIFICATION_SYSTEM,
                user_message=query,
                max_tokens=10,
                temperature=0.0,
            )

            query_type_str = response.strip().lower()

            # Map to QueryType enum
            type_mapping = {
                "temporal": QueryType.TEMPORAL,
                "factual": QueryType.FACTUAL,
                "opinion": QueryType.OPINION,
                "general": QueryType.GENERAL,
            }

            return type_mapping.get(query_type_str, QueryType.GENERAL)

        except Exception as e:
            logger.warning("Query classification failed, using general", error=str(e))
            return QueryType.GENERAL

    def calculate_recency_score(
        self,
        updated_at: datetime,
        max_age_days: int = 365,
    ) -> float:
        """
        Calculate recency score with temporal decay.

        Returns 1.0 for very recent, approaching 0.0 for old memories.
        """
        now = datetime.now(timezone.utc)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        age_seconds = (now - updated_at).total_seconds()
        age_days = age_seconds / 86400

        # Exponential decay
        if age_days <= 0:
            return 1.0
        if age_days >= max_age_days:
            return 0.0

        # Half-life of 30 days
        half_life = 30
        return 0.5 ** (age_days / half_life)

    def reciprocal_rank_fusion(
        self,
        rankings: dict[str, list[tuple[UUID, float]]],
        weights: dict[str, float],
        k: int = 60,
    ) -> list[tuple[UUID, float]]:
        """
        Combine multiple rankings using Reciprocal Rank Fusion.

        Args:
            rankings: Dict mapping method name to list of (memory_id, score) tuples
            weights: Dict mapping method name to weight
            k: RRF parameter (default 60)

        Returns:
            List of (memory_id, rrf_score) sorted by score descending
        """
        rrf_scores: dict[UUID, float] = {}

        for method, ranking in rankings.items():
            weight = weights.get(method, 1.0)

            for rank, (memory_id, _) in enumerate(ranking):
                rrf_contribution = weight * (1 / (k + rank + 1))
                rrf_scores[memory_id] = rrf_scores.get(memory_id, 0) + rrf_contribution

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    async def recall(self, query: RecallQuery) -> RecallResult:
        """
        Perform hybrid memory recall.

        Combines semantic search, BM25, and recency scoring using RRF.
        Respects token budget.
        """
        postgres = await self._get_postgres()
        embedding_client = self._get_embedding()

        # Get weight profile for query type
        weights = WEIGHT_PROFILES[query.query_type]

        # Generate query embedding
        query_embedding = await embedding_client.embed(query.query)

        # Perform parallel searches
        semantic_results = await postgres.semantic_search(
            user_id=query.user_id,
            embedding=query_embedding,
            limit=query.limit * 2,  # Get extra for fusion
        )

        bm25_results = await postgres.fulltext_search(
            user_id=query.user_id,
            search_query=query.query,
            limit=query.limit * 2,
        )

        # Build rankings for RRF
        rankings: dict[str, list[tuple[UUID, float]]] = {
            "semantic": [
                (r["id"], float(r["similarity"]))
                for r in semantic_results
            ],
            "bm25": [
                (r["id"], float(r["rank"]))
                for r in bm25_results
            ],
        }

        # Add recency ranking if weight > 0
        if weights.recency > 0:
            # Combine all unique memories and rank by recency
            all_memories = {r["id"]: r for r in semantic_results}
            all_memories.update({r["id"]: r for r in bm25_results})

            recency_ranking = [
                (mid, self.calculate_recency_score(m["updated_at"]))
                for mid, m in all_memories.items()
            ]
            recency_ranking.sort(key=lambda x: x[1], reverse=True)
            rankings["recency"] = recency_ranking

        # Apply RRF
        weight_dict = {
            "semantic": weights.semantic,
            "bm25": weights.bm25,
            "recency": weights.recency,
        }

        fused_results = self.reciprocal_rank_fusion(
            rankings=rankings,
            weights=weight_dict,
            k=self._settings.rrf_k,
        )

        # Fetch full memory data and apply token budget
        memories: list[MemoryWithScore] = []
        total_tokens = 0
        all_memories_dict = {r["id"]: r for r in semantic_results}
        all_memories_dict.update({r["id"]: r for r in bm25_results})

        for memory_id, rrf_score in fused_results[:query.limit]:
            if memory_id not in all_memories_dict:
                # Fetch from database if not in cache
                record = await postgres.get_memory(memory_id)
                if not record:
                    continue
            else:
                record = all_memories_dict[memory_id]

            content_tokens = self.count_tokens(record["content"])

            # Check token budget
            if total_tokens + content_tokens > query.token_budget:
                logger.info(
                    "Token budget reached",
                    total_tokens=total_tokens,
                    budget=query.token_budget,
                )
                break

            total_tokens += content_tokens

            # Normalize RRF score to 0-1 range (approximate)
            normalized_score = min(1.0, rrf_score * 10)

            memories.append(
                MemoryWithScore(
                    id=record["id"],
                    user_id=record["user_id"],
                    content=record["content"],
                    memory_type=record["memory_type"],
                    confidence=float(record["confidence"]),
                    evidence_count=record["evidence_count"],
                    created_at=record["created_at"],
                    updated_at=record["updated_at"],
                    relevance_score=normalized_score,
                )
            )

        logger.info(
            "Recall completed",
            query_type=query.query_type.value,
            memories_count=len(memories),
            total_tokens=total_tokens,
        )

        return RecallResult(
            memories=memories,
            total_tokens=total_tokens,
            query_type=query.query_type,
        )

    async def recall_with_auto_classification(
        self,
        user_id: str,
        query: str,
        token_budget: int | None = None,
        limit: int = 10,
    ) -> RecallResult:
        """
        Convenience method that auto-classifies query type.
        """
        query_type = await self.classify_query_type(query)

        return await self.recall(
            RecallQuery(
                user_id=user_id,
                query=query,
                query_type=query_type,
                token_budget=token_budget or self._settings.default_token_budget,
                limit=limit,
            )
        )


# Global instance
_recall_engine: HybridRecallEngine | None = None


async def get_recall_engine() -> HybridRecallEngine:
    """Get or create the global recall engine."""
    global _recall_engine
    if _recall_engine is None:
        _recall_engine = HybridRecallEngine()
    return _recall_engine
