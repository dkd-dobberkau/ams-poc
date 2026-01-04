"""Pydantic models for memory system."""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryType(str, Enum):
    """Types of memories that can be stored."""

    FACT = "fact"
    OPINION = "opinion"
    EXPERIENCE = "experience"
    OBSERVATION = "observation"


class EvidenceType(str, Enum):
    """Types of evidence that affect opinion confidence."""

    REINFORCE = "reinforce"
    WEAKEN = "weaken"
    CONTRADICT = "contradict"


# Base Models
class MemoryBase(BaseModel):
    """Base model for memory entries."""

    user_id: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    memory_type: MemoryType


class MemoryCreate(MemoryBase):
    """Model for creating a new memory."""

    confidence: float = Field(default=0.70, ge=0.0, le=1.0)


class Memory(MemoryBase):
    """Full memory model with all fields."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class MemoryWithScore(Memory):
    """Memory with relevance/similarity score."""

    relevance_score: float = Field(ge=0.0, le=1.0)


# Opinion Models
class OpinionCreate(BaseModel):
    """Model for creating a new opinion."""

    user_id: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    confidence: float = Field(default=0.70, ge=0.0, le=1.0)


class Opinion(BaseModel):
    """Opinion with confidence tracking."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int
    created_at: datetime
    updated_at: datetime


class ConfidenceUpdate(BaseModel):
    """Result of a confidence update operation."""

    opinion_id: UUID
    old_confidence: float
    new_confidence: float
    evidence_type: EvidenceType
    evidence_count: int


# Evidence Models
class Evidence(BaseModel):
    """Evidence that affects opinion confidence."""

    statement: str = Field(..., min_length=1)
    evidence_type: EvidenceType
    impact: float = Field(default=0.1, ge=0.0, le=1.0)


class EvidenceClassification(BaseModel):
    """LLM classification result for evidence."""

    evidence_type: EvidenceType
    impact: float = Field(ge=0.0, le=1.0)
    reasoning: str


# Recall Models
class QueryType(str, Enum):
    """Types of queries for recall optimization."""

    TEMPORAL = "temporal"
    FACTUAL = "factual"
    OPINION = "opinion"
    GENERAL = "general"


class RecallQuery(BaseModel):
    """Query for memory recall."""

    user_id: str
    query: str
    query_type: QueryType = QueryType.GENERAL
    token_budget: int = Field(default=2000, ge=100)
    limit: int = Field(default=10, ge=1, le=100)


class RecallResult(BaseModel):
    """Result of a memory recall operation."""

    memories: list[MemoryWithScore]
    total_tokens: int
    query_type: QueryType


# Retention Models
class ExtractedMemory(BaseModel):
    """Memory extracted from conversation."""

    content: str
    memory_type: MemoryType
    confidence: float = Field(default=0.70, ge=0.0, le=1.0)


class RetentionResult(BaseModel):
    """Result of memory retention/extraction."""

    extracted_memories: list[ExtractedMemory]
    updated_opinions: list[ConfidenceUpdate]


# Weight Profiles for Hybrid Recall
class WeightProfile(BaseModel):
    """Weight profile for hybrid recall fusion."""

    semantic: float = Field(ge=0.0, le=1.0)
    bm25: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)

    @field_validator("recency")
    @classmethod
    def weights_sum_to_one(cls, v: float, info) -> float:
        """Validate that weights approximately sum to 1."""
        if info.data:
            total = info.data.get("semantic", 0) + info.data.get("bm25", 0) + v
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Weights must sum to 1, got {total}")
        return v


# Predefined weight profiles
WEIGHT_PROFILES: dict[QueryType, WeightProfile] = {
    QueryType.TEMPORAL: WeightProfile(semantic=0.2, bm25=0.2, recency=0.6),
    QueryType.FACTUAL: WeightProfile(semantic=0.5, bm25=0.4, recency=0.1),
    QueryType.OPINION: WeightProfile(semantic=0.6, bm25=0.2, recency=0.2),
    QueryType.GENERAL: WeightProfile(semantic=0.4, bm25=0.35, recency=0.25),
}
