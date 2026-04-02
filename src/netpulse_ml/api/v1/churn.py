"""Churn prediction API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from netpulse_ml.api.schemas import ChurnListResponse, ChurnPredictionResponse, PaginationMeta
from netpulse_ml.config import settings
from netpulse_ml.dependencies import FeatureStoreDep, PredictorDep, run_in_executor
from netpulse_ml.models.churn_predictor import score_to_risk_level
from netpulse_ml.serving.cache import get_cached, make_cache_key, set_cached

router = APIRouter()


@router.get("/churn/predictions", response_model=ChurnListResponse)
async def list_churn_predictions(
    predictor: PredictorDep,
    store: FeatureStoreDep,
    risk_level: str = Query("", description="Comma-separated risk levels: low,medium,high,critical"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ChurnListResponse:
    """List churn predictions, optionally filtered by risk level."""
    model = predictor.churn_predictor

    if not model.is_fitted:
        raise HTTPException(status_code=503, detail="Churn predictor not yet trained")

    cache_key = make_cache_key("churn", risk_level=risk_level, limit=limit, offset=offset)
    cached = await get_cached(cache_key)
    if cached is not None:
        return ChurnListResponse(**cached)

    fleet_df = await store.get_fleet_features()
    if fleet_df.empty:
        return ChurnListResponse(
            data=[], pagination=PaginationMeta(total=0, limit=limit, offset=offset)
        )

    # Offload CPU-bound inference to thread pool
    scores = await run_in_executor(model.predict, fleet_df)
    now = datetime.now(timezone.utc)

    filter_levels = set()
    if risk_level:
        filter_levels = {lvl.strip().lower() for lvl in risk_level.split(",")}

    items = []
    for i, (device_id, row) in enumerate(fleet_df.iterrows()):
        score = float(scores[i])
        level = score_to_risk_level(score)

        if filter_levels and level not in filter_levels:
            continue

        features_dict = row.to_dict()
        top_factors = model.get_top_factors(features_dict, n_top=5)

        items.append(ChurnPredictionResponse(
            deviceId=str(device_id),
            subscriberId=str(features_dict.get("subscriber_id", device_id)),
            riskScore=round(score, 1),
            riskLevel=level,
            topFactors=top_factors,
            predictedAt=now,
            modelVersion=model.version,
        ))

    items.sort(key=lambda x: x.riskScore, reverse=True)
    total = len(items)
    items = items[offset : offset + limit]

    response = ChurnListResponse(
        data=items,
        pagination=PaginationMeta(total=total, limit=limit, offset=offset),
    )
    await set_cached(cache_key, response.model_dump(mode="json"), settings.cache_ttl_churn)
    return response


@router.get("/subscribers/{subscriber_id}/churn", response_model=ChurnPredictionResponse)
async def get_subscriber_churn(
    predictor: PredictorDep,
    store: FeatureStoreDep,
    subscriber_id: str,
) -> ChurnPredictionResponse:
    """Get churn prediction for a specific subscriber."""
    model = predictor.churn_predictor
    now = datetime.now(timezone.utc)

    features = await store.get_latest_features(subscriber_id)

    if not model.is_fitted:
        return ChurnPredictionResponse(
            deviceId=subscriber_id,
            subscriberId=subscriber_id,
            riskScore=0.0,
            riskLevel="low",
            modelReady=False,
            topFactors=[],
            predictedAt=now,
            modelVersion=model.version,
        )

    if not features:
        return ChurnPredictionResponse(
            deviceId=subscriber_id,
            subscriberId=subscriber_id,
            riskScore=0.0,
            riskLevel="low",
            topFactors=[],
            predictedAt=now,
            modelVersion=model.version,
        )

    result = await run_in_executor(model.predict_single, features)

    return ChurnPredictionResponse(
        deviceId=subscriber_id,
        subscriberId=subscriber_id,
        riskScore=result["riskScore"],
        riskLevel=result["riskLevel"],
        topFactors=result["topFactors"],
        predictedAt=now,
        modelVersion=model.version,
    )
