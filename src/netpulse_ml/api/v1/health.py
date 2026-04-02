"""Health check endpoint."""

from fastapi import APIRouter, Request
from sqlalchemy import text

from netpulse_ml.api.schemas import HealthResponse
from netpulse_ml.db.engine import async_session_factory

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Check service health: models loaded, DB connected, MQTT running."""
    predictor = request.app.state.predictor
    mqtt = getattr(request.app.state, "mqtt_consumer", None)

    # Check model status
    model_status = {}
    for name in ["anomaly_detector", "churn_predictor", "qoe_forecaster", "fleet_clusterer"]:
        model = predictor.get_model(name)
        if model and model.is_fitted:
            model_status[name] = "loaded"
        elif model:
            model_status[name] = "initialized_not_trained"
        else:
            model_status[name] = "not_loaded"

    # Check DB
    db_ok = False
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    # Check MQTT
    mqtt_ok = mqtt.is_running if mqtt else False

    # Overall status
    all_models_loaded = all(v == "loaded" for v in model_status.values())
    if db_ok and mqtt_ok and all_models_loaded:
        status = "healthy"
    elif db_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        models=model_status,
        mqttConnected=mqtt_ok,
        dbConnected=db_ok,
    )
