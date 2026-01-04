"""Pytest fixtures for Agent Memory System tests."""

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.config import Settings
from src.db.postgres import PostgresClient
from src.db.qdrant import QdrantClient
from src.llm.client import ClaudeClient, EmbeddingClient
from src.memory.models import EvidenceType, MemoryType
from src.memory.opinion import OpinionMemoryManager
from src.memory.recall import HybridRecallEngine
from src.memory.retention import RetentionEngine


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        database_url="postgresql+asyncpg://test:test@localhost:5432/test_db",
        qdrant_url="http://localhost:6333",
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        default_confidence=0.70,
        similarity_threshold=0.85,
    )


@pytest.fixture
def mock_postgres() -> AsyncMock:
    """Create mock PostgreSQL client."""
    mock = AsyncMock(spec=PostgresClient)

    # Default return values
    mock.fetchval.return_value = uuid.uuid4()
    mock.fetch.return_value = []
    mock.fetchrow.return_value = None

    return mock


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    """Create mock Qdrant client."""
    mock = AsyncMock(spec=QdrantClient)
    mock.search.return_value = []
    return mock


@pytest.fixture
def mock_claude() -> MagicMock:
    """Create mock Claude client."""
    mock = MagicMock(spec=ClaudeClient)
    mock.complete = AsyncMock(return_value="Test response")
    mock.complete_json = AsyncMock(
        return_value='{"evidence_type": "reinforce", "impact": 0.3, "reasoning": "Test"}'
    )
    return mock


@pytest.fixture
def mock_embedding() -> MagicMock:
    """Create mock embedding client."""
    mock = MagicMock(spec=EmbeddingClient)
    # Return a 1536-dimensional embedding
    mock.embed = AsyncMock(return_value=[0.1] * 1536)
    mock.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    return mock


@pytest.fixture
def sample_memory_record() -> dict:
    """Create a sample memory record as returned by PostgreSQL."""
    return {
        "id": uuid.uuid4(),
        "user_id": "test-user",
        "memory_type": MemoryType.OPINION.value,
        "content": "Python is great for data science",
        "confidence": 0.70,
        "evidence_count": 1,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "similarity": 0.95,
        "rank": 1.0,
    }


@pytest.fixture
def sample_opinion_record(sample_memory_record) -> dict:
    """Create a sample opinion record."""
    return sample_memory_record


@pytest_asyncio.fixture
async def opinion_manager(
    mock_postgres: AsyncMock,
    mock_qdrant: AsyncMock,
    mock_claude: MagicMock,
    mock_embedding: MagicMock,
) -> OpinionMemoryManager:
    """Create OpinionMemoryManager with mocked dependencies."""
    return OpinionMemoryManager(
        postgres=mock_postgres,
        qdrant=mock_qdrant,
        claude=mock_claude,
        embedding=mock_embedding,
    )


@pytest_asyncio.fixture
async def recall_engine(
    mock_postgres: AsyncMock,
    mock_claude: MagicMock,
    mock_embedding: MagicMock,
) -> HybridRecallEngine:
    """Create HybridRecallEngine with mocked dependencies."""
    return HybridRecallEngine(
        postgres=mock_postgres,
        claude=mock_claude,
        embedding=mock_embedding,
    )


@pytest_asyncio.fixture
async def retention_engine(
    mock_postgres: AsyncMock,
    mock_qdrant: AsyncMock,
    mock_claude: MagicMock,
    mock_embedding: MagicMock,
    opinion_manager: OpinionMemoryManager,
) -> RetentionEngine:
    """Create RetentionEngine with mocked dependencies."""
    return RetentionEngine(
        postgres=mock_postgres,
        qdrant=mock_qdrant,
        claude=mock_claude,
        embedding=mock_embedding,
        opinion_manager=opinion_manager,
    )


# Parametrized test data
CONFIDENCE_UPDATE_TEST_CASES = [
    # (current_confidence, evidence_count, evidence_type, impact, expected_direction)
    (0.70, 1, EvidenceType.REINFORCE, 0.3, "increase"),
    (0.70, 1, EvidenceType.WEAKEN, 0.3, "decrease"),
    (0.70, 1, EvidenceType.CONTRADICT, 0.5, "decrease"),
    (0.90, 5, EvidenceType.REINFORCE, 0.3, "increase"),  # Stable opinion
    (0.30, 1, EvidenceType.CONTRADICT, 0.8, "decrease"),  # Strong contradiction
]


@pytest.fixture(params=CONFIDENCE_UPDATE_TEST_CASES)
def confidence_test_case(request):
    """Parametrized confidence update test cases."""
    return request.param
