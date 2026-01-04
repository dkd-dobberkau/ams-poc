"""Opinion Memory Manager with confidence tracking."""

import json
import uuid
from math import log

import structlog

from src.config import get_settings
from src.db.postgres import PostgresClient, get_postgres
from src.db.qdrant import QdrantClient, get_qdrant
from src.exceptions import (
    ClassificationError,
    ConfidenceUpdateError,
    InvalidConfidenceError,
    OpinionNotFoundError,
)
from src.llm.client import ClaudeClient, EmbeddingClient, get_claude, get_embedding_client
from src.llm.prompts import EVIDENCE_CLASSIFICATION_SYSTEM, EVIDENCE_CLASSIFICATION_USER
from src.memory.models import (
    ConfidenceUpdate,
    Evidence,
    EvidenceClassification,
    EvidenceType,
    MemoryType,
    Opinion,
)

logger = structlog.get_logger()


class OpinionMemoryManager:
    """
    Manages opinion memories with confidence tracking.

    Implements the HINDSIGHT-inspired confidence update formula:
    - stability_factor = 1 / (1 + log(evidence_count + 1))
    - type_multiplier = {reinforce: 1.0, weaken: 1.3, contradict: 2.0}
    - effective_change = impact * stability_factor * type_multiplier

    For reinforcement:
        new_confidence = confidence + (1 - confidence) * effective_change
    For weakening/contradiction:
        new_confidence = confidence * (1 - effective_change)
    """

    TYPE_MULTIPLIERS = {
        EvidenceType.REINFORCE: 1.0,
        EvidenceType.WEAKEN: 1.3,
        EvidenceType.CONTRADICT: 2.0,
    }

    def __init__(
        self,
        postgres: PostgresClient | None = None,
        qdrant: QdrantClient | None = None,
        claude: ClaudeClient | None = None,
        embedding: EmbeddingClient | None = None,
    ) -> None:
        self._postgres = postgres
        self._qdrant = qdrant
        self._claude = claude
        self._embedding = embedding
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

    def calculate_new_confidence(
        self,
        current_confidence: float,
        evidence_count: int,
        evidence_type: EvidenceType,
        impact: float,
    ) -> float:
        """
        Calculate new confidence using the non-linear update formula.

        Args:
            current_confidence: Current confidence score (0-1)
            evidence_count: Number of evidence pieces seen so far
            evidence_type: Type of evidence (reinforce, weaken, contradict)
            impact: Impact factor of the evidence (0-1)

        Returns:
            New confidence score (0-1)
        """
        if not 0 <= current_confidence <= 1:
            raise InvalidConfidenceError(current_confidence)
        if not 0 <= impact <= 1:
            raise InvalidConfidenceError(impact)

        # Stability factor decreases with more evidence (opinions become more stable)
        stability_factor = 1 / (1 + log(evidence_count + 1))

        # Type multiplier - contradictions have higher impact
        type_multiplier = self.TYPE_MULTIPLIERS[evidence_type]

        # Effective change
        effective_change = impact * stability_factor * type_multiplier

        # Calculate new confidence
        if evidence_type == EvidenceType.REINFORCE:
            # Move towards 1.0
            new_confidence = current_confidence + (1 - current_confidence) * effective_change
        else:
            # Move towards 0.0
            new_confidence = current_confidence * (1 - effective_change)

        # Clamp to valid range
        return max(0.0, min(1.0, new_confidence))

    async def create_opinion(
        self,
        user_id: str,
        content: str,
        confidence: float = 0.70,
    ) -> Opinion:
        """
        Create a new opinion or update existing similar opinion.

        If a semantically similar opinion exists (>= similarity threshold),
        the existing opinion's confidence is updated instead of creating new.
        """
        postgres = await self._get_postgres()
        qdrant = await self._get_qdrant()
        embedding_client = self._get_embedding()

        # Generate embedding for the opinion
        embedding = await embedding_client.embed(content)

        # Check for existing similar opinion
        existing = await postgres.find_similar_opinion(
            user_id=user_id,
            embedding=embedding,
            threshold=self._settings.similarity_threshold,
        )

        if existing:
            # Update existing opinion's confidence
            logger.info(
                "Found similar opinion, updating confidence",
                existing_id=str(existing["id"]),
                similarity=existing["similarity"],
            )
            # Classify as reinforcement since it's similar
            update_result = await self.update_confidence(
                opinion_id=existing["id"],
                evidence=Evidence(
                    statement=content,
                    evidence_type=EvidenceType.REINFORCE,
                    impact=0.2,  # Low impact for similar statements
                ),
            )
            updated = await postgres.get_memory(existing["id"])
            return self._record_to_opinion(updated)

        # Create new opinion
        memory_id = await postgres.insert_memory(
            user_id=user_id,
            memory_type=MemoryType.OPINION.value,
            content=content,
            embedding=embedding,
            confidence=confidence,
        )

        # Also store in Qdrant for fast similarity search
        await qdrant.upsert(
            memory_id=memory_id,
            user_id=user_id,
            memory_type=MemoryType.OPINION.value,
            content=content,
            embedding=embedding,
            confidence=confidence,
        )

        logger.info("Opinion created", opinion_id=str(memory_id))

        record = await postgres.get_memory(memory_id)
        return self._record_to_opinion(record)

    async def get_opinion(self, opinion_id: uuid.UUID) -> Opinion:
        """Get an opinion by ID."""
        postgres = await self._get_postgres()
        record = await postgres.get_memory(opinion_id)
        if not record or record["memory_type"] != MemoryType.OPINION.value:
            raise OpinionNotFoundError(f"Opinion {opinion_id} not found")
        return self._record_to_opinion(record)

    async def get_user_opinions(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[Opinion]:
        """Get all opinions for a user."""
        postgres = await self._get_postgres()
        records = await postgres.get_user_memories(
            user_id=user_id,
            memory_type=MemoryType.OPINION.value,
            limit=limit,
        )
        return [self._record_to_opinion(r) for r in records]

    async def update_confidence(
        self,
        opinion_id: uuid.UUID,
        evidence: Evidence,
    ) -> ConfidenceUpdate:
        """
        Update opinion confidence based on new evidence.

        Uses the non-linear confidence update formula.
        """
        postgres = await self._get_postgres()
        qdrant = await self._get_qdrant()

        # Get current opinion
        record = await postgres.get_memory(opinion_id)
        if not record or record["memory_type"] != MemoryType.OPINION.value:
            raise OpinionNotFoundError(f"Opinion {opinion_id} not found")

        old_confidence = float(record["confidence"])
        evidence_count = record["evidence_count"]

        # Calculate new confidence
        new_confidence = self.calculate_new_confidence(
            current_confidence=old_confidence,
            evidence_count=evidence_count,
            evidence_type=evidence.evidence_type,
            impact=evidence.impact,
        )

        # Update in PostgreSQL
        updated = await postgres.update_confidence(
            memory_id=opinion_id,
            new_confidence=new_confidence,
            increment_evidence=True,
        )

        if not updated:
            raise ConfidenceUpdateError(f"Failed to update opinion {opinion_id}")

        # Update in Qdrant
        embedding_client = self._get_embedding()
        embedding = await embedding_client.embed(record["content"])
        await qdrant.upsert(
            memory_id=opinion_id,
            user_id=record["user_id"],
            memory_type=MemoryType.OPINION.value,
            content=record["content"],
            embedding=embedding,
            confidence=new_confidence,
        )

        logger.info(
            "Confidence updated",
            opinion_id=str(opinion_id),
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            evidence_type=evidence.evidence_type.value,
        )

        return ConfidenceUpdate(
            opinion_id=opinion_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            evidence_type=evidence.evidence_type,
            evidence_count=evidence_count + 1,
        )

    async def classify_evidence(
        self,
        opinion_content: str,
        statement: str,
    ) -> EvidenceClassification:
        """
        Use LLM to classify how a statement affects an opinion.

        Returns the evidence type (reinforce/weaken/contradict) and impact.
        """
        claude = self._get_claude()

        prompt = EVIDENCE_CLASSIFICATION_USER.format(
            opinion=opinion_content,
            statement=statement,
        )

        try:
            response = await claude.complete_json(
                system=EVIDENCE_CLASSIFICATION_SYSTEM,
                user_message=prompt,
            )

            # Parse JSON response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            data = json.loads(response)

            return EvidenceClassification(
                evidence_type=EvidenceType(data["evidence_type"]),
                impact=float(data["impact"]),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse evidence classification", error=str(e))
            raise ClassificationError(f"Failed to classify evidence: {e}") from e

    async def process_statement(
        self,
        user_id: str,
        statement: str,
    ) -> ConfidenceUpdate | Opinion:
        """
        Process a new statement that may affect existing opinions.

        If the statement is similar to an existing opinion, classifies the
        relationship and updates confidence. Otherwise, creates a new opinion.
        """
        postgres = await self._get_postgres()
        embedding_client = self._get_embedding()

        # Generate embedding
        embedding = await embedding_client.embed(statement)

        # Find similar existing opinion
        similar = await postgres.find_similar_opinion(
            user_id=user_id,
            embedding=embedding,
            threshold=self._settings.similarity_threshold,
        )

        if similar:
            # Classify the relationship
            classification = await self.classify_evidence(
                opinion_content=similar["content"],
                statement=statement,
            )

            # Update confidence
            evidence = Evidence(
                statement=statement,
                evidence_type=classification.evidence_type,
                impact=classification.impact,
            )

            return await self.update_confidence(
                opinion_id=similar["id"],
                evidence=evidence,
            )
        else:
            # Create new opinion
            return await self.create_opinion(
                user_id=user_id,
                content=statement,
                confidence=self._settings.default_confidence,
            )

    def _record_to_opinion(self, record) -> Opinion:
        """Convert database record to Opinion model."""
        return Opinion(
            id=record["id"],
            user_id=record["user_id"],
            content=record["content"],
            confidence=float(record["confidence"]),
            evidence_count=record["evidence_count"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )


# Global instance
_opinion_manager: OpinionMemoryManager | None = None


async def get_opinion_manager() -> OpinionMemoryManager:
    """Get or create the global opinion manager."""
    global _opinion_manager
    if _opinion_manager is None:
        _opinion_manager = OpinionMemoryManager()
    return _opinion_manager
