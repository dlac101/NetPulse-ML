"""WebSocket endpoint for real-time dashboard updates.

Connected clients receive push notifications when new telemetry arrives
or when model predictions change. Uses a simple broadcast pattern.
"""

import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = structlog.get_logger()

router = APIRouter()

# Connected WebSocket clients
_clients: set[WebSocket] = set()


async def broadcast(event_type: str, data: dict) -> None:
    """Broadcast an event to all connected dashboard clients."""
    if not _clients:  # noqa: F823
        return

    message = json.dumps({"type": event_type, "data": data})
    disconnected = set()

    for ws in _clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)

    _clients -= disconnected


@router.websocket("/updates/ws")
async def dashboard_updates(websocket: WebSocket) -> None:
    """WebSocket for real-time dashboard updates.

    Clients connect and receive push notifications:
    - {"type": "telemetry", "data": {"device_id": "...", "features_count": N}}
    - {"type": "anomaly", "data": {"device_id": "...", "score": 0.85}}
    - {"type": "agent_run", "data": {"device_id": "...", "status": "escalated"}}
    - {"type": "model_trained", "data": {"model": "anomaly_detector", "metrics": {...}}}
    """
    await websocket.accept()
    _clients.add(websocket)
    log.info("Dashboard client connected", total_clients=len(_clients))

    try:
        while True:
            # Keep connection alive; clients don't send data, they just receive
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)
        log.info("Dashboard client disconnected", total_clients=len(_clients))
