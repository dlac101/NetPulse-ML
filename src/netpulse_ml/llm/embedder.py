"""Text embedding using sentence-transformers (all-MiniLM-L6-v2, 384-dim, CPU).

All encode() calls are synchronous/CPU-bound. Callers in async context must
use run_in_executor() to avoid blocking the event loop.
"""

import asyncio
from functools import partial

import structlog
from sentence_transformers import SentenceTransformer

from netpulse_ml.config import settings

log = structlog.get_logger()


class Embedder:
    """Wraps sentence-transformers for text embedding."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        """Load the embedding model into memory. Call once at startup."""
        log.info("Loading embedding model", model=self._model_name)
        self._model = SentenceTransformer(self._model_name)
        log.info("Embedding model loaded", dimension=self._model.get_sentence_embedding_dimension())

    async def load_async(self) -> None:
        """Load the model in a thread to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.load)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string (sync, CPU-bound). Returns a 384-dim vector."""
        if self._model is None:
            raise RuntimeError("Embedder not loaded. Call load() first.")
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts (sync, CPU-bound). Returns list of 384-dim vectors."""
        if self._model is None:
            raise RuntimeError("Embedder not loaded. Call load() first.")
        embeddings = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return embeddings.tolist()

    async def aembed_text(self, text: str) -> list[float]:
        """Async wrapper: offloads embedding to thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.embed_text, text))

    async def aembed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Async wrapper: offloads batch embedding to thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.embed_batch, texts, batch_size))

    @property
    def dimension(self) -> int:
        return settings.embedding_dimension
