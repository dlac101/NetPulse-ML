"""FastAPI dependency injection providers."""

import asyncio
from functools import partial
from typing import Annotated, Any, AsyncGenerator, Callable

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from netpulse_ml.config import settings
from netpulse_ml.db.engine import async_session_factory
from netpulse_ml.features.store import FeatureStore, feature_store
from netpulse_ml.serving.predictor import Predictor


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""
    async with async_session_factory() as session:
        yield session


async def get_predictor(request: Request) -> Predictor:
    """Get the model predictor singleton from app state."""
    return request.app.state.predictor


async def get_feature_store() -> FeatureStore:
    """Get the feature store singleton."""
    return feature_store


async def verify_api_key(x_api_key: str = Header(default="")) -> str:
    """Verify API key from X-Api-Key header. Skipped if api_key is empty (dev mode)."""
    if not settings.api_key:
        return ""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


async def run_in_executor(fn: Callable, *args: Any) -> Any:
    """Offload a synchronous CPU-bound function to the thread pool executor.

    Prevents blocking the asyncio event loop during scikit-learn inference/training.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args) if args else fn)


DB = Annotated[AsyncSession, Depends(get_db)]
PredictorDep = Annotated[Predictor, Depends(get_predictor)]
FeatureStoreDep = Annotated[FeatureStore, Depends(get_feature_store)]
ApiKeyDep = Annotated[str, Depends(verify_api_key)]
