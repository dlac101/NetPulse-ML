"""Churn label management API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from netpulse_ml.db.models import ChurnLabel
from netpulse_ml.dependencies import DB

router = APIRouter()


class ChurnLabelRequest(BaseModel):
    subscriberId: str
    deviceId: str | None = None
    churned: bool
    reason: str | None = Field(None, max_length=256)
    labeledBy: str = Field(default="manual", max_length=64)


class ChurnLabelResponse(BaseModel):
    id: str
    subscriberId: str
    deviceId: str | None
    churned: bool
    reason: str | None
    labeledBy: str
    labeledAt: datetime


@router.post("/labels/churn", response_model=ChurnLabelResponse)
async def add_churn_label(db: DB, body: ChurnLabelRequest) -> ChurnLabelResponse:
    """Add a manual churn label for a subscriber."""
    label = ChurnLabel(
        subscriber_id=body.subscriberId,
        device_id=body.deviceId,
        churned=body.churned,
        reason=body.reason,
        labeled_by=body.labeledBy,
    )
    db.add(label)
    await db.commit()
    await db.refresh(label)

    return ChurnLabelResponse(
        id=label.id,
        subscriberId=label.subscriber_id,
        deviceId=label.device_id,
        churned=label.churned,
        reason=label.reason,
        labeledBy=label.labeled_by,
        labeledAt=label.labeled_at,
    )


@router.get("/labels/churn")
async def list_churn_labels(
    db: DB,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """List all churn labels."""
    count_result = await db.execute(select(func.count()).select_from(ChurnLabel))
    total = count_result.scalar() or 0

    result = await db.execute(
        select(ChurnLabel).order_by(ChurnLabel.labeled_at.desc()).offset(offset).limit(limit)
    )
    labels = result.scalars().all()

    return {
        "data": [
            {
                "id": l.id,
                "subscriberId": l.subscriber_id,
                "deviceId": l.device_id,
                "churned": l.churned,
                "reason": l.reason,
                "labeledBy": l.labeled_by,
                "labeledAt": l.labeled_at.isoformat() if l.labeled_at else None,
            }
            for l in labels
        ],
        "pagination": {"total": total, "limit": limit, "offset": offset},
    }


@router.get("/labels/churn/stats")
async def churn_label_stats(db: DB) -> dict:
    """Get churn label statistics."""
    total = (await db.execute(select(func.count()).select_from(ChurnLabel))).scalar() or 0
    churned = (
        await db.execute(
            select(func.count()).select_from(ChurnLabel).where(ChurnLabel.churned)
        )
    ).scalar() or 0

    return {
        "totalLabels": total,
        "churnedCount": churned,
        "retainedCount": total - churned,
        "churnRate": churned / max(total, 1),
        "sufficientForTraining": total >= 20 and churned >= 5,
    }
