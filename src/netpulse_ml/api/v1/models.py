"""Model registry and retraining API endpoints."""

import asyncio

from fastapi import APIRouter, HTTPException

from netpulse_ml.api.schemas import ModelInfo, ModelListResponse
from netpulse_ml.dependencies import PredictorDep, run_in_executor
from netpulse_ml.models.registry import list_models
from netpulse_ml.training.pipeline import (
    train_all,
    train_anomaly_detector,
    train_churn_predictor,
    train_fleet_clusterer,
)

router = APIRouter()

# Lock to prevent concurrent training jobs
_training_lock = asyncio.Lock()


@router.get("/models", response_model=ModelListResponse)
async def get_models(predictor: PredictorDep) -> ModelListResponse:
    """List all registered model versions."""
    models = await list_models()

    by_name: dict[str, ModelInfo] = {}
    for m in models:
        name = m["name"]
        if name not in by_name or m.get("isActive"):
            by_name[name] = ModelInfo(
                name=name,
                activeVersion=m["version"] if m.get("isActive") else None,
                algorithm=m["algorithm"],
                trainedAt=m.get("trainedAt"),
                metrics=m.get("metrics"),
                featureCount=m.get("featureCount", 0),
            )

    for model_name in predictor.loaded_model_names:
        if model_name not in by_name:
            model = predictor.get_model(model_name)
            by_name[model_name] = ModelInfo(
                name=model_name,
                activeVersion=model.version if model and model.is_fitted else None,
                algorithm=type(model).__name__ if model else "unknown",
                featureCount=len(model.feature_names) if model else 0,
            )

    return ModelListResponse(models=list(by_name.values()))


@router.post("/models/{model_name}/retrain")
async def retrain_model(
    predictor: PredictorDep,
    model_name: str,
) -> dict:
    """Trigger retraining for a specific model. Only one training job at a time."""
    trainers = {
        "anomaly_detector": train_anomaly_detector,
        "churn_predictor": train_churn_predictor,
        "fleet_clusterer": train_fleet_clusterer,
    }

    trainer = trainers.get(model_name)
    if trainer is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    if _training_lock.locked():
        raise HTTPException(status_code=409, detail="Training already in progress")

    async with _training_lock:
        try:
            metrics = await trainer()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {e}")

        predictor.reload_model(model_name)

    return {"model": model_name, "status": "completed", "metrics": metrics}


@router.post("/models/retrain-all")
async def retrain_all_models(predictor: PredictorDep) -> dict:
    """Trigger retraining for all models."""
    if _training_lock.locked():
        raise HTTPException(status_code=409, detail="Training already in progress")

    async with _training_lock:
        try:
            results = await train_all()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {e}")

        for name in results:
            predictor.reload_model(name)

    return {"status": "completed", "results": results}
