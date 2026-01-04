"""Tests for Opinion Memory Manager."""

import uuid
from datetime import datetime, timezone
from math import log
from unittest.mock import AsyncMock

import pytest

from src.exceptions import InvalidConfidenceError, OpinionNotFoundError
from src.memory.models import Evidence, EvidenceType, MemoryType
from src.memory.opinion import OpinionMemoryManager


class TestConfidenceCalculation:
    """Tests for the confidence update formula."""

    def test_reinforce_increases_confidence(self, opinion_manager: OpinionMemoryManager):
        """Reinforcing evidence should increase confidence."""
        result = opinion_manager.calculate_new_confidence(
            current_confidence=0.70,
            evidence_count=1,
            evidence_type=EvidenceType.REINFORCE,
            impact=0.3,
        )
        assert result > 0.70

    def test_weaken_decreases_confidence(self, opinion_manager: OpinionMemoryManager):
        """Weakening evidence should decrease confidence."""
        result = opinion_manager.calculate_new_confidence(
            current_confidence=0.70,
            evidence_count=1,
            evidence_type=EvidenceType.WEAKEN,
            impact=0.3,
        )
        assert result < 0.70

    def test_contradict_decreases_confidence_more(self, opinion_manager: OpinionMemoryManager):
        """Contradicting evidence should decrease confidence more than weakening."""
        weaken_result = opinion_manager.calculate_new_confidence(
            current_confidence=0.70,
            evidence_count=1,
            evidence_type=EvidenceType.WEAKEN,
            impact=0.3,
        )
        contradict_result = opinion_manager.calculate_new_confidence(
            current_confidence=0.70,
            evidence_count=1,
            evidence_type=EvidenceType.CONTRADICT,
            impact=0.3,
        )
        assert contradict_result < weaken_result

    def test_stability_factor_reduces_with_evidence(
        self, opinion_manager: OpinionMemoryManager
    ):
        """More evidence should make the opinion more stable (smaller changes)."""
        # Same impact, different evidence counts
        change_early = opinion_manager.calculate_new_confidence(
            current_confidence=0.70,
            evidence_count=1,
            evidence_type=EvidenceType.REINFORCE,
            impact=0.3,
        )
        change_late = opinion_manager.calculate_new_confidence(
            current_confidence=0.70,
            evidence_count=10,
            evidence_type=EvidenceType.REINFORCE,
            impact=0.3,
        )

        # Early change should be larger
        assert abs(change_early - 0.70) > abs(change_late - 0.70)

    def test_confidence_bounded_zero_one(self, opinion_manager: OpinionMemoryManager):
        """Confidence should always stay within [0, 1]."""
        # Try to push above 1
        high_result = opinion_manager.calculate_new_confidence(
            current_confidence=0.99,
            evidence_count=1,
            evidence_type=EvidenceType.REINFORCE,
            impact=1.0,
        )
        assert high_result <= 1.0

        # Try to push below 0
        low_result = opinion_manager.calculate_new_confidence(
            current_confidence=0.01,
            evidence_count=1,
            evidence_type=EvidenceType.CONTRADICT,
            impact=1.0,
        )
        assert low_result >= 0.0

    def test_invalid_confidence_raises_error(self, opinion_manager: OpinionMemoryManager):
        """Invalid confidence values should raise an error."""
        with pytest.raises(InvalidConfidenceError):
            opinion_manager.calculate_new_confidence(
                current_confidence=1.5,  # Invalid
                evidence_count=1,
                evidence_type=EvidenceType.REINFORCE,
                impact=0.3,
            )

    def test_invalid_impact_raises_error(self, opinion_manager: OpinionMemoryManager):
        """Invalid impact values should raise an error."""
        with pytest.raises(InvalidConfidenceError):
            opinion_manager.calculate_new_confidence(
                current_confidence=0.70,
                evidence_count=1,
                evidence_type=EvidenceType.REINFORCE,
                impact=1.5,  # Invalid
            )

    def test_parametrized_confidence_updates(
        self, opinion_manager: OpinionMemoryManager, confidence_test_case
    ):
        """Test various confidence update scenarios."""
        current, count, etype, impact, expected_dir = confidence_test_case

        result = opinion_manager.calculate_new_confidence(
            current_confidence=current,
            evidence_count=count,
            evidence_type=etype,
            impact=impact,
        )

        if expected_dir == "increase":
            assert result > current
        else:
            assert result < current


class TestOpinionCreation:
    """Tests for creating new opinions."""

    @pytest.mark.asyncio
    async def test_create_new_opinion(
        self,
        opinion_manager: OpinionMemoryManager,
        mock_postgres: AsyncMock,
        mock_embedding: AsyncMock,
        sample_opinion_record: dict,
    ):
        """Creating a new opinion should store it in both databases."""
        # Setup: No similar opinion exists
        mock_postgres.find_similar_opinion.return_value = None
        mock_postgres.get_memory.return_value = sample_opinion_record

        opinion = await opinion_manager.create_opinion(
            user_id="test-user",
            content="Python is great for data science",
            confidence=0.70,
        )

        assert opinion.content == "Python is great for data science"
        assert opinion.confidence == 0.70
        mock_postgres.insert_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_updates_similar_opinion(
        self,
        opinion_manager: OpinionMemoryManager,
        mock_postgres: AsyncMock,
        sample_opinion_record: dict,
    ):
        """Creating an opinion similar to existing should update, not create new."""
        # Setup: Similar opinion exists
        mock_postgres.find_similar_opinion.return_value = sample_opinion_record
        mock_postgres.get_memory.return_value = {
            **sample_opinion_record,
            "confidence": 0.75,  # Updated
            "evidence_count": 2,
        }

        opinion = await opinion_manager.create_opinion(
            user_id="test-user",
            content="Python is excellent for data science",  # Similar
            confidence=0.70,
        )

        # Should update existing, not insert new
        mock_postgres.insert_memory.assert_not_called()
        mock_postgres.update_confidence.assert_called_once()


class TestConfidenceUpdates:
    """Tests for updating opinion confidence."""

    @pytest.mark.asyncio
    async def test_update_confidence_reinforcement(
        self,
        opinion_manager: OpinionMemoryManager,
        mock_postgres: AsyncMock,
        sample_opinion_record: dict,
    ):
        """Updating with reinforcing evidence should increase confidence."""
        mock_postgres.get_memory.return_value = sample_opinion_record
        mock_postgres.update_confidence.return_value = {
            **sample_opinion_record,
            "confidence": 0.80,
            "evidence_count": 2,
        }

        evidence = Evidence(
            statement="I love using Python for ML",
            evidence_type=EvidenceType.REINFORCE,
            impact=0.3,
        )

        result = await opinion_manager.update_confidence(
            opinion_id=sample_opinion_record["id"],
            evidence=evidence,
        )

        assert result.new_confidence > result.old_confidence
        assert result.evidence_type == EvidenceType.REINFORCE

    @pytest.mark.asyncio
    async def test_update_nonexistent_opinion_raises(
        self,
        opinion_manager: OpinionMemoryManager,
        mock_postgres: AsyncMock,
    ):
        """Updating a non-existent opinion should raise an error."""
        mock_postgres.get_memory.return_value = None

        evidence = Evidence(
            statement="Test",
            evidence_type=EvidenceType.REINFORCE,
            impact=0.3,
        )

        with pytest.raises(OpinionNotFoundError):
            await opinion_manager.update_confidence(
                opinion_id=uuid.uuid4(),
                evidence=evidence,
            )


class TestEvidenceClassification:
    """Tests for LLM-based evidence classification."""

    @pytest.mark.asyncio
    async def test_classify_reinforcing_evidence(
        self,
        opinion_manager: OpinionMemoryManager,
        mock_claude,
    ):
        """Should correctly classify reinforcing evidence."""
        mock_claude.complete_json.return_value = (
            '{"evidence_type": "reinforce", "impact": 0.4, "reasoning": "Supports opinion"}'
        )

        result = await opinion_manager.classify_evidence(
            opinion_content="Python is great",
            statement="Python is excellent for beginners",
        )

        assert result.evidence_type == EvidenceType.REINFORCE
        assert result.impact == 0.4

    @pytest.mark.asyncio
    async def test_classify_contradicting_evidence(
        self,
        opinion_manager: OpinionMemoryManager,
        mock_claude,
    ):
        """Should correctly classify contradicting evidence."""
        mock_claude.complete_json.return_value = (
            '{"evidence_type": "contradict", "impact": 0.8, "reasoning": "Direct opposition"}'
        )

        result = await opinion_manager.classify_evidence(
            opinion_content="Python is great",
            statement="Python is terrible and slow",
        )

        assert result.evidence_type == EvidenceType.CONTRADICT
        assert result.impact == 0.8


class TestStabilityFormula:
    """Tests for the stability factor formula."""

    def test_stability_formula_values(self):
        """Verify stability factor formula produces expected values."""
        # stability_factor = 1 / (1 + log(evidence_count + 1))

        # With 1 evidence: 1 / (1 + log(2)) ≈ 0.59
        sf_1 = 1 / (1 + log(1 + 1))
        assert 0.58 < sf_1 < 0.60

        # With 10 evidence: 1 / (1 + log(11)) ≈ 0.29
        sf_10 = 1 / (1 + log(10 + 1))
        assert 0.28 < sf_10 < 0.31

        # With 100 evidence: 1 / (1 + log(101)) ≈ 0.18
        sf_100 = 1 / (1 + log(100 + 1))
        assert 0.17 < sf_100 < 0.19

    def test_type_multipliers(self, opinion_manager: OpinionMemoryManager):
        """Verify type multipliers are correct."""
        assert opinion_manager.TYPE_MULTIPLIERS[EvidenceType.REINFORCE] == 1.0
        assert opinion_manager.TYPE_MULTIPLIERS[EvidenceType.WEAKEN] == 1.3
        assert opinion_manager.TYPE_MULTIPLIERS[EvidenceType.CONTRADICT] == 2.0
