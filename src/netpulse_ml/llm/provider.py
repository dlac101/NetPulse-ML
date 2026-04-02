"""Ollama LLM provider: async HTTP client for local model inference."""

from collections.abc import AsyncGenerator

import httpx
import orjson
import structlog

from netpulse_ml.config import settings

log = structlog.get_logger()


class OllamaProvider:
    """Async client for Ollama's HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model
        self._client = httpx.AsyncClient(timeout=120.0)

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        temp = temperature if temperature is not None else settings.ollama_temperature

        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temp},
        }

        response = await self._client.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def generate_stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response, yielding tokens as they arrive."""
        temp = temperature if temperature is not None else settings.ollama_temperature

        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {"temperature": temp},
        }

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = orjson.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
                except orjson.JSONDecodeError:
                    continue

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            if response.status_code != 200:
                return False
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check if our model (or a partial match) is available
            return any(self._model.split(":")[0] in m for m in models)
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
