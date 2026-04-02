"""QoE forecasting using SARIMAX time series models.

Produces 24-hour QoE composite score forecasts with confidence intervals.
"""

import numpy as np
import pandas as pd
import structlog
from statsmodels.tsa.statespace.sarimax import SARIMAX

from netpulse_ml.models.base import ModelWrapper

log = structlog.get_logger()

# 96 periods = 24 hours at 15-minute intervals
SEASONAL_PERIOD = 96
MIN_HISTORY_POINTS = 96 * 7  # 7 days minimum


class QoEForecaster(ModelWrapper):
    """Per-device SARIMAX forecaster for QoE composite scores."""

    name = "qoe_forecaster"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._feature_names = ["qoe_composite_latest"]
        # Store per-device fitted models
        self._device_models: dict[str, object] = {}

    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, float]:
        """Train is a no-op for SARIMAX; models are fit per-device on demand."""
        self._is_fitted = True
        return {"status": 1.0, "note_fit_on_predict": 1.0}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Not used for time series. Use forecast_device() instead."""
        raise NotImplementedError("Use forecast_device() for time series forecasting")

    def forecast_device(
        self,
        qoe_series: pd.Series,
        horizon_steps: int = 96,
        confidence: float = 0.05,
    ) -> dict:
        """Forecast QoE for a single device.

        Args:
            qoe_series: Time-indexed Series of QoE composite scores at 15-min intervals.
            horizon_steps: Number of 15-min steps to forecast (96 = 24h).
            confidence: Confidence level for intervals (0.05 = 95% CI).

        Returns:
            Dict with forecast points, confidence intervals, and trend direction.
        """
        if len(qoe_series) < 48:
            return self._insufficient_data_response(qoe_series, horizon_steps)

        # Determine model order based on available history
        if len(qoe_series) >= MIN_HISTORY_POINTS:
            seasonal_order = (1, 1, 1, SEASONAL_PERIOD)
        elif len(qoe_series) >= 96:
            # Shorter history: simpler seasonal
            seasonal_order = (1, 0, 1, 96)
        else:
            seasonal_order = (0, 0, 0, 0)

        try:
            model = SARIMAX(
                endog=qoe_series.values,
                order=(1, 1, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False, maxiter=100)

            forecast_obj = results.get_forecast(steps=horizon_steps)
            predicted = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=confidence)

            # Clip to valid QoE range
            predicted = np.clip(predicted, 0, 100)
            conf_int = np.clip(conf_int, 0, 100)

            # Generate timestamps
            last_ts = qoe_series.index[-1]
            freq = pd.Timedelta(minutes=15)
            timestamps = [last_ts + freq * (i + 1) for i in range(horizon_steps)]

            # Trend direction from slope
            if len(predicted) >= 2:
                slope_per_step = (predicted[-1] - predicted[0]) / len(predicted)
                slope_per_hour = slope_per_step * 4  # 4 steps per hour
                if slope_per_hour > 0.5:
                    trend = "improving"
                elif slope_per_hour < -0.5:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            forecast_points = []
            for i in range(horizon_steps):
                forecast_points.append({
                    "timestamp": timestamps[i].isoformat(),
                    "predicted": round(float(predicted[i]), 2),
                    "lower95": round(float(conf_int[i, 0]), 2),
                    "upper95": round(float(conf_int[i, 1]), 2),
                })

            return {
                "currentQoE": round(float(qoe_series.iloc[-1]), 2),
                "forecast": forecast_points,
                "trendDirection": trend,
                "modelFit": {
                    "aic": round(float(results.aic), 2),
                    "bic": round(float(results.bic), 2),
                },
            }

        except Exception as e:
            log.warning("SARIMAX fitting failed, falling back to naive forecast", error=str(e))
            return self._naive_forecast(qoe_series, horizon_steps)

    def _insufficient_data_response(
        self, series: pd.Series, horizon: int
    ) -> dict:
        """Return a flat forecast when insufficient history exists."""
        current = float(series.iloc[-1]) if len(series) > 0 else 50.0
        return {
            "currentQoE": round(current, 2),
            "forecast": [],
            "trendDirection": "stable",
            "modelFit": {"aic": 0.0, "bic": 0.0},
            "warning": "Insufficient history for forecasting (need 48+ data points)",
        }

    def _naive_forecast(self, series: pd.Series, horizon: int) -> dict:
        """Simple moving-average fallback forecast."""
        window = min(len(series), 96)
        mean_val = float(series.iloc[-window:].mean())
        std_val = float(series.iloc[-window:].std()) if window > 1 else 5.0

        last_ts = series.index[-1]
        freq = pd.Timedelta(minutes=15)

        forecast_points = []
        for i in range(horizon):
            forecast_points.append({
                "timestamp": (last_ts + freq * (i + 1)).isoformat(),
                "predicted": round(mean_val, 2),
                "lower95": round(max(mean_val - 2 * std_val, 0), 2),
                "upper95": round(min(mean_val + 2 * std_val, 100), 2),
            })

        return {
            "currentQoE": round(float(series.iloc[-1]), 2),
            "forecast": forecast_points,
            "trendDirection": "stable",
            "modelFit": {"aic": 0.0, "bic": 0.0},
            "warning": "Using naive forecast (SARIMAX failed)",
        }
