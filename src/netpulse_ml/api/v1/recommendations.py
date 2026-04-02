"""ML recommendation API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, update

from netpulse_ml.api.schemas import (
    ApproveRequest,
    DeviceRecommendationsResponse,
    MLRecommendationResponse,
)
from netpulse_ml.db.models import Recommendation
from netpulse_ml.dependencies import DB

router = APIRouter()


@router.get("/devices/{device_id}/recommendations", response_model=DeviceRecommendationsResponse)
async def get_device_recommendations(
    db: DB,
    device_id: str,
) -> DeviceRecommendationsResponse:
    """Get ML recommendations for a specific device."""
    result = await db.execute(
        select(Recommendation)
        .where(
            Recommendation.device_id == device_id,
            Recommendation.status == "pending",
        )
        .order_by(Recommendation.confidence.desc())
    )
    recs = result.scalars().all()

    items = [
        MLRecommendationResponse(
            id=rec.id,
            deviceId=rec.device_id,
            type=rec.type,
            title=rec.title,
            description=rec.description,
            confidence=rec.confidence,
            impact=rec.impact,
            autoExecutable=rec.auto_executable,
            createdAt=rec.created_at,
            status=rec.status,
        )
        for rec in recs
    ]

    return DeviceRecommendationsResponse(deviceId=device_id, recommendations=items)


@router.post("/recommendations/{rec_id}/approve")
async def approve_recommendation(
    db: DB,
    rec_id: str,
    body: ApproveRequest,
) -> dict:
    """Approve a recommendation for execution."""
    result = await db.execute(
        update(Recommendation)
        .where(Recommendation.id == rec_id)
        .values(
            status="approved",
            executed_at=datetime.now(timezone.utc),
            executed_by=body.executedBy,
        )
    )
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    return {"id": rec_id, "status": "approved"}


@router.post("/recommendations/{rec_id}/dismiss")
async def dismiss_recommendation(
    db: DB,
    rec_id: str,
) -> dict:
    """Dismiss a recommendation."""
    result = await db.execute(
        update(Recommendation)
        .where(Recommendation.id == rec_id)
        .values(status="dismissed")
    )
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    return {"id": rec_id, "status": "dismissed"}
