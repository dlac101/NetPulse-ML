"""Router fleet management API endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from netpulse_ml.db.models import RouterRegistry
from netpulse_ml.dependencies import DB

router = APIRouter()


class RouterCreateRequest(BaseModel):
    name: str = Field(max_length=128)
    host: str = Field(max_length=256)
    username: str = Field(default="admin", max_length=64)
    password: str = Field(default="", max_length=256)


class RouterResponse(BaseModel):
    id: str
    name: str
    host: str
    username: str
    model: str | None = None
    firmware: str | None = None
    mac: str | None = None
    enabled: bool = True
    lastSeen: str | None = None
    createdAt: str


@router.post("/routers", response_model=RouterResponse)
async def add_router(db: DB, body: RouterCreateRequest) -> RouterResponse:
    """Register a new SmartOS router for fleet management."""
    entry = RouterRegistry(
        name=body.name,
        host=body.host,
        username=body.username,
        password=body.password,
        enabled=True,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)

    return RouterResponse(
        id=entry.id,
        name=entry.name,
        host=entry.host,
        username=entry.username,
        model=entry.model,
        firmware=entry.firmware,
        mac=entry.mac,
        enabled=entry.enabled,
        lastSeen=entry.last_seen.isoformat() if entry.last_seen else None,
        createdAt=entry.created_at.isoformat() if entry.created_at else "",
    )


@router.get("/routers")
async def list_routers(
    db: DB,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """List all registered routers."""
    total = (await db.execute(select(func.count()).select_from(RouterRegistry))).scalar() or 0
    result = await db.execute(
        select(RouterRegistry).order_by(RouterRegistry.name).offset(offset).limit(limit)
    )
    entries = result.scalars().all()

    return {
        "data": [
            {
                "id": r.id,
                "name": r.name,
                "host": r.host,
                "username": r.username,
                "model": r.model,
                "firmware": r.firmware,
                "mac": r.mac,
                "enabled": r.enabled,
                "lastSeen": r.last_seen.isoformat() if r.last_seen else None,
            }
            for r in entries
        ],
        "pagination": {"total": total, "limit": limit, "offset": offset},
    }


@router.delete("/routers/{router_id}")
async def remove_router(db: DB, router_id: str) -> dict:
    """Remove a router from the fleet registry."""
    result = await db.execute(
        select(RouterRegistry).where(RouterRegistry.id == router_id)
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise HTTPException(status_code=404, detail="Router not found")

    await db.delete(entry)
    await db.commit()
    return {"id": router_id, "status": "deleted"}
