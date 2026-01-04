"""FastAPI routes for the Agent Memory System."""

from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConfidenceUpdate,
    ErrorResponse,
    HealthResponse,
    MemoriesListResponse,
    MemoryResponse,
    MemorySyncRequest,
    OpinionCreateRequest,
    OpinionsListResponse,
    RecallRequest,
    RecallResponse,
    SyncResponse,
)
from src.db.postgres import get_postgres
from src.db.qdrant import get_qdrant
from src.exceptions import MemoryNotFoundError, MemorySystemError, OpinionNotFoundError
from src.llm.client import get_claude, get_embedding_client
from src.llm.prompts import CHAT_SYSTEM_TEMPLATE
from src.memory.models import MemoryType
from src.memory.opinion import get_opinion_manager
from src.memory.recall import get_recall_engine
from src.memory.retention import get_retention_engine

logger = structlog.get_logger()

router = APIRouter()


# Health Check
@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Check system health."""
    postgres_ok = False
    qdrant_ok = False

    try:
        postgres = await get_postgres()
        await postgres.fetchval("SELECT 1")
        postgres_ok = True
    except Exception as e:
        logger.error("PostgreSQL health check failed", error=str(e))

    try:
        qdrant = await get_qdrant()
        await qdrant.get_collection_info()
        qdrant_ok = True
    except Exception as e:
        logger.error("Qdrant health check failed", error=str(e))

    return HealthResponse(
        status="healthy" if (postgres_ok and qdrant_ok) else "degraded",
        postgres=postgres_ok,
        qdrant=qdrant_ok,
    )


# Chat Endpoint
@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint with memory integration.

    - Recalls relevant memories for context
    - Generates response using Claude
    - Optionally stores new memories from the conversation
    """
    try:
        recall_engine = await get_recall_engine()
        retention_engine = await get_retention_engine()
        claude = get_claude()

        memories_used: list[MemoryResponse] = []
        confidence_updates: list[ConfidenceUpdate] = []

        # Recall relevant memories
        memory_context = ""
        if request.include_memories:
            recall_result = await recall_engine.recall_with_auto_classification(
                user_id=request.user_id,
                query=request.message,
            )

            for mem in recall_result.memories:
                memories_used.append(
                    MemoryResponse(
                        id=mem.id,
                        user_id=mem.user_id,
                        content=mem.content,
                        memory_type=MemoryType(mem.memory_type),
                        confidence=mem.confidence,
                        evidence_count=mem.evidence_count,
                        relevance_score=mem.relevance_score,
                    )
                )
                # Build context string
                memory_context += f"- [{mem.memory_type}] {mem.content}"
                if mem.memory_type == "opinion":
                    memory_context += f" (Konfidenz: {mem.confidence:.0%})"
                memory_context += "\n"

        # Generate response
        system_prompt = CHAT_SYSTEM_TEMPLATE.format(
            memories=memory_context if memory_context else "Keine relevanten Erinnerungen."
        )

        response_text = await claude.complete(
            system=system_prompt,
            user_message=request.message,
            max_tokens=2048,
        )

        # Store new memories from conversation
        if request.store_memories:
            conversation = f"User: {request.message}\nAssistant: {response_text}"
            retention_result = await retention_engine.process_conversation(
                user_id=request.user_id,
                conversation=conversation,
            )
            confidence_updates = retention_result.updated_opinions

        return ChatResponse(
            response=response_text,
            memories_used=memories_used,
            confidence_updates=confidence_updates,
        )

    except MemorySystemError as e:
        logger.error("Chat error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# Memory Endpoints
@router.get(
    "/memories/{user_id}",
    response_model=MemoriesListResponse,
    tags=["memories"],
)
async def get_user_memories(
    user_id: str,
    memory_type: MemoryType | None = None,
    limit: int = 100,
) -> MemoriesListResponse:
    """Get all memories for a user."""
    try:
        postgres = await get_postgres()

        records = await postgres.get_user_memories(
            user_id=user_id,
            memory_type=memory_type.value if memory_type else None,
            limit=limit,
        )

        memories = [
            MemoryResponse(
                id=r["id"],
                user_id=r["user_id"],
                content=r["content"],
                memory_type=MemoryType(r["memory_type"]),
                confidence=float(r["confidence"]),
                evidence_count=r["evidence_count"],
            )
            for r in records
        ]

        return MemoriesListResponse(memories=memories, total=len(memories))

    except MemorySystemError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/memories/{user_id}/opinions",
    response_model=OpinionsListResponse,
    tags=["memories"],
)
async def get_user_opinions(
    user_id: str,
    limit: int = 100,
) -> OpinionsListResponse:
    """Get all opinions for a user with confidence scores."""
    try:
        opinion_manager = await get_opinion_manager()
        opinions = await opinion_manager.get_user_opinions(user_id, limit)

        opinion_responses = [
            MemoryResponse(
                id=o.id,
                user_id=o.user_id,
                content=o.content,
                memory_type=MemoryType.OPINION,
                confidence=o.confidence,
                evidence_count=o.evidence_count,
            )
            for o in opinions
        ]

        return OpinionsListResponse(opinions=opinion_responses, total=len(opinion_responses))

    except MemorySystemError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/memories/{user_id}/opinions",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["memories"],
)
async def create_opinion(
    user_id: str,
    request: OpinionCreateRequest,
) -> MemoryResponse:
    """Create a new opinion for a user."""
    try:
        opinion_manager = await get_opinion_manager()
        opinion = await opinion_manager.create_opinion(
            user_id=user_id,
            content=request.content,
            confidence=request.confidence,
        )

        return MemoryResponse(
            id=opinion.id,
            user_id=opinion.user_id,
            content=opinion.content,
            memory_type=MemoryType.OPINION,
            confidence=opinion.confidence,
            evidence_count=opinion.evidence_count,
        )

    except MemorySystemError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/memories/{user_id}/sync",
    response_model=SyncResponse,
    tags=["memories"],
)
async def sync_memories(
    user_id: str,
    request: MemorySyncRequest,
) -> SyncResponse:
    """Manually sync/extract memories from a conversation."""
    try:
        retention_engine = await get_retention_engine()
        result = await retention_engine.process_conversation(
            user_id=user_id,
            conversation=request.conversation,
        )

        return SyncResponse(
            new_memories=len(result.extracted_memories),
            updated_opinions=len(result.updated_opinions),
            confidence_updates=result.updated_opinions,
        )

    except MemorySystemError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/memories/{user_id}/recall",
    response_model=RecallResponse,
    tags=["memories"],
)
async def recall_memories(
    user_id: str,
    request: RecallRequest,
) -> RecallResponse:
    """Recall relevant memories for a query."""
    try:
        recall_engine = await get_recall_engine()

        if request.query_type:
            from src.memory.models import RecallQuery

            result = await recall_engine.recall(
                RecallQuery(
                    user_id=user_id,
                    query=request.query,
                    query_type=request.query_type,
                    token_budget=request.token_budget,
                    limit=request.limit,
                )
            )
        else:
            result = await recall_engine.recall_with_auto_classification(
                user_id=user_id,
                query=request.query,
                token_budget=request.token_budget,
                limit=request.limit,
            )

        memories = [
            MemoryResponse(
                id=m.id,
                user_id=m.user_id,
                content=m.content,
                memory_type=MemoryType(m.memory_type),
                confidence=m.confidence,
                evidence_count=m.evidence_count,
                relevance_score=m.relevance_score,
            )
            for m in result.memories
        ]

        return RecallResponse(
            memories=memories,
            total_tokens=result.total_tokens,
            query_type=result.query_type,
        )

    except MemorySystemError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/memories/{user_id}/{memory_id}",
    response_model=MemoryResponse,
    tags=["memories"],
)
async def get_memory(
    user_id: str,
    memory_id: UUID,
) -> MemoryResponse:
    """Get a specific memory by ID."""
    try:
        postgres = await get_postgres()
        record = await postgres.get_memory(memory_id)

        if not record or record["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )

        return MemoryResponse(
            id=record["id"],
            user_id=record["user_id"],
            content=record["content"],
            memory_type=MemoryType(record["memory_type"]),
            confidence=float(record["confidence"]),
            evidence_count=record["evidence_count"],
        )

    except HTTPException:
        raise
    except MemorySystemError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete(
    "/memories/{user_id}/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["memories"],
)
async def delete_memory(
    user_id: str,
    memory_id: UUID,
) -> None:
    """Delete a specific memory."""
    try:
        postgres = await get_postgres()
        qdrant = await get_qdrant()

        # Verify ownership
        record = await postgres.get_memory(memory_id)
        if not record or record["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )

        # Delete from both stores
        await postgres.delete_memory(memory_id)
        await qdrant.delete(memory_id)

    except HTTPException:
        raise
    except MemorySystemError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
