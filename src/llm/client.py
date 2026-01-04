"""Claude API wrapper and embedding client."""

import anthropic
import openai
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.exceptions import EmbeddingError, LLMAPIError

logger = structlog.get_logger()


class ClaudeClient:
    """Async Claude API client."""

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key
        self._model = settings.claude_model
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from Claude."""
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                temperature=temperature,
            )
            content = response.content[0]
            if content.type == "text":
                return content.text
            raise LLMAPIError("Unexpected response type from Claude")
        except anthropic.APIError as e:
            logger.error("Claude API error", error=str(e))
            raise LLMAPIError(f"Claude API error: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete_json(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a JSON completion from Claude."""
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.0,  # Deterministic for JSON
            )
            content = response.content[0]
            if content.type == "text":
                return content.text
            raise LLMAPIError("Unexpected response type from Claude")
        except anthropic.APIError as e:
            logger.error("Claude API error", error=str(e))
            raise LLMAPIError(f"Claude API error: {e}") from e


class EmbeddingClient:
    """OpenAI embedding client."""

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._client = openai.AsyncOpenAI(api_key=self._api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
                dimensions=self._dimensions,
            )
            return response.data[0].embedding
        except openai.APIError as e:
            logger.error("OpenAI embedding error", error=str(e))
            raise EmbeddingError(f"Embedding error: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
            )
            return [item.embedding for item in response.data]
        except openai.APIError as e:
            logger.error("OpenAI embedding error", error=str(e))
            raise EmbeddingError(f"Embedding error: {e}") from e


# Global instances
_claude_client: ClaudeClient | None = None
_embedding_client: EmbeddingClient | None = None


def get_claude() -> ClaudeClient:
    """Get or create the global Claude client."""
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client


def get_embedding_client() -> EmbeddingClient:
    """Get or create the global embedding client."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
