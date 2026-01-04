"""Tests for Hybrid-Recall Engine."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from src.memory.models import QueryType, RecallQuery, WEIGHT_PROFILES
from src.memory.recall import HybridRecallEngine


class TestRecencyScoring:
    """Tests for recency score calculation."""

    def test_recent_memory_high_score(self, recall_engine: HybridRecallEngine):
        """Very recent memories should have high recency score."""
        now = datetime.now(timezone.utc)
        score = recall_engine.calculate_recency_score(now)
        assert score > 0.95

    def test_old_memory_low_score(self, recall_engine: HybridRecallEngine):
        """Old memories should have low recency score."""
        old_date = datetime.now(timezone.utc) - timedelta(days=180)
        score = recall_engine.calculate_recency_score(old_date)
        assert score < 0.1

    def test_half_life_decay(self, recall_engine: HybridRecallEngine):
        """Score should roughly halve every 30 days (half-life)."""
        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        score_now = recall_engine.calculate_recency_score(now)
        score_30d = recall_engine.calculate_recency_score(thirty_days_ago)

        # Should be approximately half
        ratio = score_30d / score_now
        assert 0.45 < ratio < 0.55


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def test_rrf_combines_rankings(self, recall_engine: HybridRecallEngine):
        """RRF should combine multiple rankings."""
        id1, id2, id3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        rankings = {
            "semantic": [(id1, 0.9), (id2, 0.8), (id3, 0.7)],
            "bm25": [(id2, 0.95), (id1, 0.85), (id3, 0.75)],
        }
        weights = {"semantic": 0.5, "bm25": 0.5}

        result = recall_engine.reciprocal_rank_fusion(rankings, weights, k=60)

        # Result should contain all IDs
        result_ids = [r[0] for r in result]
        assert id1 in result_ids
        assert id2 in result_ids
        assert id3 in result_ids

    def test_rrf_respects_weights(self, recall_engine: HybridRecallEngine):
        """Higher weighted rankings should have more influence."""
        id1, id2 = uuid.uuid4(), uuid.uuid4()

        # id1 is #1 in semantic, id2 is #1 in bm25
        rankings = {
            "semantic": [(id1, 0.9), (id2, 0.7)],
            "bm25": [(id2, 0.9), (id1, 0.7)],
        }

        # Heavy weight on semantic
        weights_semantic = {"semantic": 0.9, "bm25": 0.1}
        result_semantic = recall_engine.reciprocal_rank_fusion(
            rankings, weights_semantic, k=60
        )

        # id1 should be first when semantic is weighted higher
        assert result_semantic[0][0] == id1

        # Heavy weight on bm25
        weights_bm25 = {"semantic": 0.1, "bm25": 0.9}
        result_bm25 = recall_engine.reciprocal_rank_fusion(rankings, weights_bm25, k=60)

        # id2 should be first when bm25 is weighted higher
        assert result_bm25[0][0] == id2

    def test_rrf_k_parameter(self, recall_engine: HybridRecallEngine):
        """Higher k should smooth out ranking differences."""
        id1, id2 = uuid.uuid4(), uuid.uuid4()

        rankings = {
            "method": [(id1, 0.9), (id2, 0.8)],
        }
        weights = {"method": 1.0}

        # Low k: more emphasis on top ranks
        result_low_k = recall_engine.reciprocal_rank_fusion(rankings, weights, k=1)
        score_diff_low = result_low_k[0][1] - result_low_k[1][1]

        # High k: more even distribution
        result_high_k = recall_engine.reciprocal_rank_fusion(rankings, weights, k=100)
        score_diff_high = result_high_k[0][1] - result_high_k[1][1]

        # Score difference should be smaller with higher k
        assert score_diff_high < score_diff_low


class TestQueryTypeClassification:
    """Tests for query type classification."""

    @pytest.mark.asyncio
    async def test_classify_temporal_query(
        self, recall_engine: HybridRecallEngine, mock_claude
    ):
        """Should classify temporal queries correctly."""
        mock_claude.complete.return_value = "temporal"

        result = await recall_engine.classify_query_type("Was habe ich gestern gesagt?")
        assert result == QueryType.TEMPORAL

    @pytest.mark.asyncio
    async def test_classify_factual_query(
        self, recall_engine: HybridRecallEngine, mock_claude
    ):
        """Should classify factual queries correctly."""
        mock_claude.complete.return_value = "factual"

        result = await recall_engine.classify_query_type("Wo arbeitet Alice?")
        assert result == QueryType.FACTUAL

    @pytest.mark.asyncio
    async def test_fallback_to_general(
        self, recall_engine: HybridRecallEngine, mock_claude
    ):
        """Should fallback to general on classification failure."""
        mock_claude.complete.side_effect = Exception("API Error")

        result = await recall_engine.classify_query_type("Random query")
        assert result == QueryType.GENERAL


class TestWeightProfiles:
    """Tests for weight profiles."""

    def test_all_query_types_have_profiles(self):
        """All query types should have weight profiles."""
        for query_type in QueryType:
            assert query_type in WEIGHT_PROFILES

    def test_weights_sum_to_one(self):
        """All weight profiles should sum to approximately 1."""
        for query_type, profile in WEIGHT_PROFILES.items():
            total = profile.semantic + profile.bm25 + profile.recency
            assert 0.99 <= total <= 1.01, f"{query_type} weights sum to {total}"

    def test_temporal_emphasizes_recency(self):
        """Temporal queries should emphasize recency."""
        profile = WEIGHT_PROFILES[QueryType.TEMPORAL]
        assert profile.recency > profile.semantic
        assert profile.recency > profile.bm25

    def test_factual_emphasizes_semantic_and_bm25(self):
        """Factual queries should emphasize semantic and BM25."""
        profile = WEIGHT_PROFILES[QueryType.FACTUAL]
        assert profile.semantic > profile.recency
        assert profile.bm25 > profile.recency

    def test_opinion_emphasizes_semantic(self):
        """Opinion queries should emphasize semantic search."""
        profile = WEIGHT_PROFILES[QueryType.OPINION]
        assert profile.semantic > profile.bm25
        assert profile.semantic > profile.recency


class TestTokenBudget:
    """Tests for token budget management."""

    def test_count_tokens(self, recall_engine: HybridRecallEngine):
        """Should count tokens correctly."""
        text = "Dies ist ein Test."
        tokens = recall_engine.count_tokens(text)
        assert tokens > 0
        assert tokens < 100  # Short text

    def test_count_tokens_longer_text(self, recall_engine: HybridRecallEngine):
        """Longer text should have more tokens."""
        short = "Kurz."
        long = "Dies ist ein viel längerer Text mit vielen Wörtern und Sätzen."

        short_tokens = recall_engine.count_tokens(short)
        long_tokens = recall_engine.count_tokens(long)

        assert long_tokens > short_tokens


class TestRecall:
    """Integration tests for the recall method."""

    @pytest.mark.asyncio
    async def test_recall_returns_results(
        self,
        recall_engine: HybridRecallEngine,
        mock_postgres: AsyncMock,
        mock_embedding: AsyncMock,
        sample_memory_record: dict,
    ):
        """Recall should return results from combined searches."""
        # Setup mock returns
        mock_postgres.semantic_search.return_value = [sample_memory_record]
        mock_postgres.fulltext_search.return_value = [sample_memory_record]

        query = RecallQuery(
            user_id="test-user",
            query="Python data science",
            query_type=QueryType.GENERAL,
            token_budget=2000,
            limit=10,
        )

        result = await recall_engine.recall(query)

        assert len(result.memories) > 0
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_recall_respects_token_budget(
        self,
        recall_engine: HybridRecallEngine,
        mock_postgres: AsyncMock,
        mock_embedding: AsyncMock,
        sample_memory_record: dict,
    ):
        """Recall should stop when token budget is exceeded."""
        # Create multiple memories that would exceed budget
        memories = []
        for i in range(10):
            mem = {**sample_memory_record}
            mem["id"] = uuid.uuid4()
            mem["content"] = "A" * 500  # Long content
            memories.append(mem)

        mock_postgres.semantic_search.return_value = memories
        mock_postgres.fulltext_search.return_value = []

        query = RecallQuery(
            user_id="test-user",
            query="Test",
            query_type=QueryType.GENERAL,
            token_budget=200,  # Small budget
            limit=10,
        )

        result = await recall_engine.recall(query)

        # Should have stopped before getting all memories
        assert len(result.memories) < 10
        assert result.total_tokens <= 200
