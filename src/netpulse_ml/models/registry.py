"""Model registry: version, store, and retrieve trained model artifacts."""

from datetime import datetime, timezone
from pathlib import Path

import structlog
from sqlalchemy import select, update

from netpulse_ml.db.engine import async_session_factory
from netpulse_ml.db.models import ModelRegistry
from netpulse_ml.models.base import ModelWrapper

log = structlog.get_logger()


async def register_model(
    model: ModelWrapper,
    artifact_path: Path,
    metrics: dict[str, float],
    activate: bool = True,
) -> None:
    """Register a trained model version in the database.

    Uses an atomic transaction with row locking to prevent race conditions
    when concurrent retrain jobs complete simultaneously.
    """
    async with async_session_factory() as session:
        async with session.begin():
            if activate:
                # Lock existing rows for this model to prevent concurrent activation
                await session.execute(
                    select(ModelRegistry)
                    .where(ModelRegistry.name == model.name)
                    .with_for_update()
                )
                await session.execute(
                    update(ModelRegistry)
                    .where(ModelRegistry.name == model.name, ModelRegistry.is_active == True)
                    .values(is_active=False)
                )

            entry = ModelRegistry(
                name=model.name,
                version=model.version,
                algorithm=type(model).__name__,
                artifact_path=str(artifact_path),
                metrics=metrics,
                feature_names=model.feature_names,
                trained_at=datetime.now(timezone.utc),
                is_active=activate,
            )
            session.add(entry)
            # commit happens at end of `async with session.begin()`

    log.info(
        "Model registered",
        name=model.name,
        version=model.version,
        active=activate,
    )


async def get_active_model_path(name: str) -> Path | None:
    """Get the artifact path for the currently active model version."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(ModelRegistry)
            .where(ModelRegistry.name == name, ModelRegistry.is_active == True)
            .order_by(ModelRegistry.trained_at.desc())
            .limit(1)
        )
        entry = result.scalar_one_or_none()

    if entry is None:
        return None
    return Path(entry.artifact_path)


async def list_models() -> list[dict]:
    """List registered models (active versions only, plus latest inactive)."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(ModelRegistry)
            .order_by(ModelRegistry.name, ModelRegistry.trained_at.desc())
            .limit(100)
        )
        entries = result.scalars().all()

    models = []
    for entry in entries:
        models.append({
            "name": entry.name,
            "version": entry.version,
            "algorithm": entry.algorithm,
            "isActive": entry.is_active,
            "trainedAt": entry.trained_at.isoformat() if entry.trained_at else None,
            "metrics": entry.metrics,
            "featureCount": len(entry.feature_names) if entry.feature_names else 0,
        })
    return models
