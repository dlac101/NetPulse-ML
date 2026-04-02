"""API router: mounts all v1 endpoint groups with API key auth."""

from fastapi import APIRouter, Depends

from netpulse_ml.api.v1 import (
    agents,
    labels,
    routers,
    websocket_updates,
    anomalies,
    chat,
    churn,
    clusters,
    forecasts,
    health,
    models,
    recommendations,
)
from netpulse_ml.dependencies import verify_api_key

api_router = APIRouter(dependencies=[Depends(verify_api_key)])

api_router.include_router(health.router, tags=["health"])
api_router.include_router(anomalies.router, tags=["anomalies"])
api_router.include_router(churn.router, tags=["churn"])
api_router.include_router(forecasts.router, tags=["forecasts"])
api_router.include_router(clusters.router, tags=["clusters"])
api_router.include_router(recommendations.router, tags=["recommendations"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(agents.router, tags=["agents"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(labels.router, tags=["labels"])
api_router.include_router(routers.router, tags=["routers"])
api_router.include_router(websocket_updates.router, tags=["updates"])
