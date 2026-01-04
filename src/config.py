"""Configuration settings via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://ams_user:ams_password@localhost:5432/memory_db"

    # Vector Store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "memories"

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # LLM Settings
    claude_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Memory Settings
    default_token_budget: int = 2000
    confidence_decay_rate: float = 0.01
    similarity_threshold: float = 0.85
    default_confidence: float = 0.70
    rrf_k: int = 60

    # Logging
    log_level: str = "INFO"

    @property
    def sync_database_url(self) -> str:
        """Get synchronous database URL for Alembic."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
