"""Agent orchestrator API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import select, func as sqlfunc

from netpulse_ml.api.schemas import (
    AgentConfigRequest,
    AgentExecutionItem,
    AgentHistoryResponse,
    AgentStatusResponse,
    AgentTriggerResponse,
    PaginationMeta,
)
from netpulse_ml.db.models import AgentExecution
from netpulse_ml.dependencies import DB

router = APIRouter()


def _get_orchestrator(request: Request):
    """Get agent orchestrator from app state."""
    orch = getattr(request.app.state, "agent_orchestrator", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    return orch


@router.get("/agents/status", response_model=AgentStatusResponse)
async def get_agent_status(request: Request) -> AgentStatusResponse:
    """Get current agent orchestrator status."""
    orch = _get_orchestrator(request)
    return AgentStatusResponse(**orch.status)


@router.get("/agents/history", response_model=AgentHistoryResponse)
async def get_agent_history(
    db: DB,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> AgentHistoryResponse:
    """Get recent agent execution history."""
    # Count total
    count_result = await db.execute(select(sqlfunc.count()).select_from(AgentExecution))
    total = count_result.scalar() or 0

    # Fetch page
    result = await db.execute(
        select(AgentExecution)
        .order_by(AgentExecution.started_at.desc())
        .offset(offset)
        .limit(limit)
    )
    executions = result.scalars().all()

    items = [
        AgentExecutionItem(
            id=ex.id,
            deviceId=ex.device_id,
            startedAt=ex.started_at,
            completedAt=ex.completed_at,
            status=ex.status,
            diagnosis=ex.diagnosis or "",
            recommendedAction=ex.recommended_action,
            autoExecuted=ex.auto_executed,
            verified=ex.verified,
            recommendationId=ex.recommendation_id,
        )
        for ex in executions
    ]

    return AgentHistoryResponse(
        data=items,
        pagination=PaginationMeta(total=total, limit=limit, offset=offset),
    )


@router.post("/agents/{device_id}/trigger", response_model=AgentTriggerResponse)
async def trigger_agent(
    request: Request,
    device_id: str,
) -> AgentTriggerResponse:
    """Manually trigger the remediation agent for a specific device."""
    orch = _get_orchestrator(request)

    final_state = await orch.run_for_device(device_id)

    return AgentTriggerResponse(
        executionId=final_state.get("recommendation_id", "") or device_id,
        deviceId=device_id,
        status=final_state.get("status", "unknown"),
        diagnosis=final_state.get("diagnosis"),
        recommendedAction=final_state.get("recommended_action"),
    )


@router.get("/agents/config")
async def get_agent_config(request: Request) -> dict:
    """Get current agent configuration."""
    orch = _get_orchestrator(request)
    s = orch._settings
    return {
        "anomalyThreshold": s.agent_anomaly_threshold,
        "scanIntervalMinutes": s.agent_scan_interval_minutes,
        "cooldownHours": s.agent_cooldown_hours,
        "maxConcurrent": s.agent_max_concurrent,
        "autoExecuteEnabled": s.agent_enable_auto_execute,
        "verifyDelayMinutes": s.agent_verify_delay_minutes,
    }


@router.put("/agents/config")
async def update_agent_config(
    request: Request,
    body: AgentConfigRequest,
) -> dict:
    """Update agent configuration (runtime only, does not persist to .env)."""
    orch = _get_orchestrator(request)
    s = orch._settings

    if body.anomalyThreshold is not None:
        s.agent_anomaly_threshold = body.anomalyThreshold
    if body.scanIntervalMinutes is not None:
        s.agent_scan_interval_minutes = body.scanIntervalMinutes
    if body.cooldownHours is not None:
        s.agent_cooldown_hours = body.cooldownHours
    if body.autoExecuteEnabled is not None:
        s.agent_enable_auto_execute = body.autoExecuteEnabled

    return {"status": "updated", "config": await get_agent_config(request)}
