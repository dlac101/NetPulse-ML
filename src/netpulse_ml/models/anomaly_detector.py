"""Anomaly detection using Isolation Forest.

Detects devices with unusual telemetry patterns across speed, QoE, WiFi,
and traffic features. Outputs a 0-1 anomaly score (higher = more anomalous).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from netpulse_ml.models.base import ModelWrapper

# Features used for anomaly detection (21 total)
ANOMALY_FEATURES = [
    "dl_mbps_latest",
    "ul_mbps_latest",
    "dl_capacity_pct",
    "dl_bloat_pct",
    "latency_idle_ms",
    "jitter_idle_ms",
    "qoe_composite_latest",
    "qoe_wan_latest",
    "qoe_wifi_latest",
    "qoe_mesh_latest",
    "qoe_system_latest",
    "wifi_avg_rssi",
    "wifi_avg_retransmit_rate",
    "wifi_airtime_util_max",
    "mesh_backhaul_min_dl_mbps",
    "traffic_max_risk_score",
    "traffic_total_bytes_1h",
    "events_critical_24h",
    "events_warning_24h",
    "wifi_client_count",
    "mesh_satellite_count",
]


class AnomalyDetector(ModelWrapper):
    """IsolationForest-based anomaly detector for device telemetry."""

    name = "anomaly_detector"
    version = "1.0.0"

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200) -> None:
        super().__init__()
        self._feature_names = ANOMALY_FEATURES.copy()
        self._contamination = contamination
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                max_samples="auto",
                max_features=1.0,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        # Store min/max of decision_function for score normalization
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, float]:
        """Train the anomaly detector on fleet feature data.

        Args:
            X: DataFrame with device features (one row per device snapshot).
            y: Not used (unsupervised). Ignored.

        Returns:
            Dict of metrics: n_samples, n_features, contamination, n_anomalies.
        """
        X_clean = self._validate_features(X.copy()).fillna(0)
        self._pipeline.fit(X_clean)

        # Compute normalization bounds from training data
        raw_scores = self._pipeline.decision_function(X_clean)
        self._score_min = float(np.min(raw_scores))
        self._score_max = float(np.max(raw_scores))

        # Count anomalies in training set
        labels = self._pipeline.predict(X_clean)
        n_anomalies = int((labels == -1).sum())

        self._is_fitted = True

        return {
            "n_samples": len(X_clean),
            "n_features": len(self._feature_names),
            "contamination": self._contamination,
            "n_anomalies": n_anomalies,
            "anomaly_rate": n_anomalies / max(len(X_clean), 1),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Score devices for anomaly. Returns array of scores 0.0-1.0.

        Higher score = more anomalous.
        Uses IsolationForest decision_function, normalized to [0, 1].
        """
        X_clean = self._validate_features(X.copy()).fillna(0)
        raw_scores = self._pipeline.decision_function(X_clean)

        # Normalize: IsolationForest returns negative for anomalies
        # Invert so higher = more anomalous, then scale to [0, 1]
        score_range = self._score_max - self._score_min
        if score_range == 0:
            return np.zeros(len(X_clean))

        normalized = 1.0 - (raw_scores - self._score_min) / score_range
        return np.clip(normalized, 0.0, 1.0)

    def predict_single(self, features: dict[str, float]) -> float:
        """Score a single device. Returns anomaly score 0.0-1.0."""
        df = pd.DataFrame([features])
        scores = self.predict(df)
        return float(scores[0])

    def get_top_features(
        self, features: dict[str, float], n_top: int = 5
    ) -> list[dict[str, float | str]]:
        """Identify the top contributing features to the anomaly score.

        Uses z-score deviation from the scaler's learned distribution.
        """
        scaler: StandardScaler = self._pipeline.named_steps["scaler"]
        means = scaler.mean_
        stds = scaler.scale_

        contributions = []
        for i, feat_name in enumerate(self._feature_names):
            val = features.get(feat_name, 0.0)
            if stds[i] > 0:
                zscore = abs(val - means[i]) / stds[i]
            else:
                zscore = 0.0
            contributions.append({
                "name": feat_name,
                "value": val,
                "zscore": round(zscore, 3),
            })

        contributions.sort(key=lambda c: c["zscore"], reverse=True)
        return contributions[:n_top]
