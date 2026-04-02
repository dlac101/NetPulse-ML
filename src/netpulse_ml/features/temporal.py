"""Temporal feature computation: rolling windows, trends, and rate-of-change."""

import numpy as np
import pandas as pd


def rolling_stats(
    series: pd.Series, window: str = "24h"
) -> dict[str, float]:
    """Compute rolling mean, std, min, max for a time-indexed series."""
    if series.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}

    rolled = series.last(window)
    return {
        "mean": float(rolled.mean()) if len(rolled) > 0 else 0.0,
        "std": float(rolled.std()) if len(rolled) > 1 else 0.0,
        "min": float(rolled.min()) if len(rolled) > 0 else 0.0,
        "max": float(rolled.max()) if len(rolled) > 0 else 0.0,
        "count": float(len(rolled)),
    }


def linear_trend(series: pd.Series) -> float:
    """Compute linear regression slope over a time series.

    Returns slope in units-per-day.
    """
    if len(series) < 2:
        return 0.0

    # Convert timestamps to days since first observation
    t = (series.index - series.index[0]).total_seconds() / 86400.0  # type: ignore[union-attr]
    t_arr = np.array(t, dtype=np.float64)
    y_arr = np.array(series.values, dtype=np.float64)

    # Remove NaN
    mask = ~(np.isnan(t_arr) | np.isnan(y_arr))
    t_arr = t_arr[mask]
    y_arr = y_arr[mask]

    if len(t_arr) < 2:
        return 0.0

    # Simple linear regression
    slope, _ = np.polyfit(t_arr, y_arr, 1)
    return float(slope)


def fraction_below_threshold(series: pd.Series, threshold: float) -> float:
    """Fraction of values in the series below the threshold."""
    if series.empty:
        return 0.0
    return float((series < threshold).sum() / len(series))


def count_drops(series: pd.Series, drop_size: float = 10.0) -> int:
    """Count instances where the value drops by more than drop_size between consecutive readings."""
    if len(series) < 2:
        return 0
    diffs = series.diff().dropna()
    return int((diffs < -drop_size).sum())


def compute_temporal_features(
    feature_name: str,
    history: pd.DataFrame,
    windows: list[str] | None = None,
) -> dict[str, float]:
    """Compute a full set of temporal features for a named feature.

    Args:
        feature_name: The base feature name (e.g., "qoe_composite_latest")
        history: DataFrame with 'timestamp' index and feature columns
        windows: Time windows to compute over (default: ["24h", "7d"])
    """
    if windows is None:
        windows = ["24h", "7d"]

    if feature_name not in history.columns:
        return {}

    series = history[feature_name].dropna()
    if series.empty:
        return {}

    result: dict[str, float] = {}

    for window in windows:
        suffix = window.replace("h", "h").replace("d", "d")
        stats = rolling_stats(series, window)
        result[f"{feature_name}_mean_{suffix}"] = stats["mean"]
        result[f"{feature_name}_std_{suffix}"] = stats["std"]
        result[f"{feature_name}_min_{suffix}"] = stats["min"]

    # 7-day trend
    if len(series) >= 2:
        result[f"{feature_name}_trend_7d"] = linear_trend(series.last("7d"))

    return result
