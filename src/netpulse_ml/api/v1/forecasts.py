"""QoE forecast API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Query

from netpulse_ml.api.schemas import QoEForecastResponse
from netpulse_ml.dependencies import FeatureStoreDep, PredictorDep, run_in_executor

router = APIRouter()

MAX_FORECAST_STEPS = 96 * 7  # Cap at 7 days


@router.get("/devices/{device_id}/qoe-forecast", response_model=QoEForecastResponse)
async def get_qoe_forecast(
    predictor: PredictorDep,
    store: FeatureStoreDep,
    device_id: str,
    horizon: str = Query("24h", description="Forecast horizon: 24h or 7d"),
) -> QoEForecastResponse:
    """Get QoE forecast with confidence intervals for a device."""
    forecaster = predictor.qoe_forecaster
    now = datetime.now(timezone.utc)

    history = await store.read_features(device_id)

    if history.empty or "qoe_composite_latest" not in history.columns:
        return QoEForecastResponse(
            deviceId=device_id,
            currentQoE=0.0,
            forecast=[],
            trendDirection="stable",
            forecastedAt=now,
            modelVersion=forecaster.version,
            warning="No QoE history available",
        )

    qoe_series = history["qoe_composite_latest"].dropna()

    steps = 96 if horizon == "24h" else min(96 * 7, MAX_FORECAST_STEPS)

    # Offload CPU-bound SARIMAX fitting to thread pool
    result = await run_in_executor(forecaster.forecast_device, qoe_series, steps)

    return QoEForecastResponse(
        deviceId=device_id,
        currentQoE=result.get("currentQoE", 0.0),
        forecast=result.get("forecast", []),
        trendDirection=result.get("trendDirection", "stable"),
        modelFit=result.get("modelFit", {}),
        forecastedAt=now,
        modelVersion=forecaster.version,
        warning=result.get("warning"),
    )
