"""Exception hierarchy for the Agent Memory System."""


class MemorySystemError(Exception):
    """Base exception for all memory system errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Memory Errors
class MemoryError(MemorySystemError):
    """Base for memory-related errors."""


class MemoryNotFoundError(MemoryError):
    """Memory entry does not exist."""


class MemoryCreationError(MemoryError):
    """Failed to create memory entry."""


class MemoryUpdateError(MemoryError):
    """Failed to update memory entry."""


# Opinion Errors
class OpinionError(MemoryError):
    """Base for opinion-related errors."""


class OpinionNotFoundError(OpinionError):
    """Opinion does not exist."""


class ConfidenceUpdateError(OpinionError):
    """Failed to update confidence score."""


class InvalidConfidenceError(OpinionError):
    """Confidence value is out of valid range [0, 1]."""

    def __init__(self, confidence: float) -> None:
        super().__init__(
            f"Confidence must be between 0 and 1, got {confidence}",
            details={"confidence": confidence},
        )


# Database Errors
class DatabaseError(MemorySystemError):
    """Base for database-related errors."""


class ConnectionError(DatabaseError):
    """Failed to connect to database."""


class QueryError(DatabaseError):
    """Database query failed."""


# Vector Store Errors
class VectorStoreError(MemorySystemError):
    """Base for vector store errors."""


class EmbeddingError(VectorStoreError):
    """Failed to create or retrieve embeddings."""


class SimilaritySearchError(VectorStoreError):
    """Similarity search failed."""


# LLM Errors
class LLMError(MemorySystemError):
    """Base for LLM-related errors."""


class LLMAPIError(LLMError):
    """LLM API call failed."""


class ClassificationError(LLMError):
    """Failed to classify content."""


# Recall Errors
class RecallError(MemorySystemError):
    """Base for recall-related errors."""


class TokenBudgetExceededError(RecallError):
    """Token budget would be exceeded."""

    def __init__(self, required: int, budget: int) -> None:
        super().__init__(
            f"Required {required} tokens but budget is {budget}",
            details={"required": required, "budget": budget},
        )
