"""Anomaly detection API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from netpulse_ml.api.schemas import (
    AnomalyListItem,
    AnomalyListResponse,
    AnomalyScoreResponse,
    FeatureContribution,
    PaginationMeta,
)
from netpulse_ml.dependencies import FeatureStoreDep, PredictorDep, run_in_executor

router = APIRouter()


@router.get("/anomalies", response_model=AnomalyListResponse)
async def list_anomalies(
    predictor: PredictorDep,
    store: FeatureStoreDep,
    threshold: float = Query(0.7, ge=0, le=1, description="Min anomaly score"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> AnomalyListResponse:
    """List devices with anomaly scores above threshold."""
    detector = predictor.anomaly_detector

    if not detector.is_fitted:
        raise HTTPException(status_code=503, detail="Anomaly detector not yet trained")

    fleet_df = await store.get_fleet_features()
    if fleet_df.empty:
        return AnomalyListResponse(
            data=[], pagination=PaginationMeta(total=0, limit=limit, offset=offset)
        )

    # Offload CPU-bound inference to thread pool
    scores = await run_in_executor(detector.predict, fleet_df)
    now = datetime.now(timezone.utc)

    items = []
    for i, (device_id, row) in enumerate(fleet_df.iterrows()):
        score = float(scores[i])
        if score >= threshold:
            features_dict = row.to_dict()
            top = detector.get_top_features(features_dict, n_top=3)
            items.append(AnomalyListItem(
                deviceId=str(device_id),
                anomalyScore=round(score, 4),
                isAnomaly=True,
                topFeatures=[
                    FeatureContribution(name=f["name"], value=f["value"], zscore=f["zscore"])
                    for f in top
                ],
                scoredAt=now,
                modelVersion=detector.version,
            ))

    items.sort(key=lambda x: x.anomalyScore, reverse=True)
    total = len(items)
    items = items[offset : offset + limit]

    return AnomalyListResponse(
        data=items,
        pagination=PaginationMeta(total=total, limit=limit, offset=offset),
    )


@router.get("/devices/{device_id}/anomaly-score", response_model=AnomalyScoreResponse)
async def get_device_anomaly(
    predictor: PredictorDep,
    store: FeatureStoreDep,
    device_id: str,
) -> AnomalyScoreResponse:
    """Get anomaly score and contributing features for a specific device."""
    detector = predictor.anomaly_detector
    features = await store.get_latest_features(device_id)
    now = datetime.now(timezone.utc)

    if not detector.is_fitted:
        return AnomalyScoreResponse(
            deviceId=device_id,
            anomalyScore=0.0,
            isAnomaly=False,
            modelReady=False,
            topFeatures=[],
            scoredAt=now,
            modelVersion=detector.version,
        )

    if not features:
        return AnomalyScoreResponse(
            deviceId=device_id,
            anomalyScore=0.0,
            isAnomaly=False,
            modelReady=True,
            topFeatures=[],
            scoredAt=now,
            modelVersion=detector.version,
        )

    score = await run_in_executor(detector.predict_single, features)
    top = detector.get_top_features(features, n_top=5)

    return AnomalyScoreResponse(
        deviceId=device_id,
        anomalyScore=round(score, 4),
        isAnomaly=score >= 0.7,
        modelReady=True,
        topFeatures=[
            FeatureContribution(name=f["name"], value=f["value"], zscore=f["zscore"])
            for f in top
        ],
        scoredAt=now,
        modelVersion=detector.version,
    )
