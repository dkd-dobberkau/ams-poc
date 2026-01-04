"""API request and response schemas."""

from uuid import UUID

from pydantic import BaseModel, Field

from src.memory.models import ConfidenceUpdate, MemoryType, MemoryWithScore, QueryType


# Request Models
class ChatRequest(BaseModel):
    """Request for chat endpoint."""

    user_id: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1)
    include_memories: bool = Field(default=True)
    store_memories: bool = Field(default=True)


class MemorySyncRequest(BaseModel):
    """Request for manual memory sync."""

    conversation: str = Field(..., min_length=1)


class OpinionCreateRequest(BaseModel):
    """Request for creating an opinion."""

    content: str = Field(..., min_length=1)
    confidence: float = Field(default=0.70, ge=0.0, le=1.0)


class RecallRequest(BaseModel):
    """Request for memory recall."""

    query: str = Field(..., min_length=1)
    query_type: QueryType | None = None
    token_budget: int = Field(default=2000, ge=100)
    limit: int = Field(default=10, ge=1, le=100)


# Response Models
class MemoryResponse(BaseModel):
    """Response with memory data."""

    id: UUID
    user_id: str
    content: str
    memory_type: MemoryType
    confidence: float
    evidence_count: int
    relevance_score: float | None = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    response: str
    memories_used: list[MemoryResponse]
    confidence_updates: list[ConfidenceUpdate]


class MemoriesListResponse(BaseModel):
    """Response with list of memories."""

    memories: list[MemoryResponse]
    total: int


class OpinionsListResponse(BaseModel):
    """Response with list of opinions."""

    opinions: list[MemoryResponse]
    total: int


class SyncResponse(BaseModel):
    """Response from memory sync."""

    new_memories: int
    updated_opinions: int
    confidence_updates: list[ConfidenceUpdate]


class RecallResponse(BaseModel):
    """Response from memory recall."""

    memories: list[MemoryResponse]
    total_tokens: int
    query_type: QueryType


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    postgres: bool
    qdrant: bool


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
