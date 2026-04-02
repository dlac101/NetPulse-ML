"""Churn prediction using HistGradientBoostingClassifier.

Predicts subscriber churn risk from device and subscriber features.
Outputs risk score 0-100, risk level, and top contributing factors via SHAP.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from netpulse_ml.models.base import ModelWrapper

# Categorical features that need ordinal encoding
CATEGORICAL_FEATURES = [
    "model_encoded",
    "firmware_encoded",
    "qoe_worst_category",
]

# All features for churn prediction
CHURN_FEATURES = [
    # Device speed
    "dl_mbps_latest",
    "ul_mbps_latest",
    "dl_capacity_pct",
    "dl_bloat_pct",
    "latency_idle_ms",
    "jitter_idle_ms",
    # QoE
    "qoe_composite_latest",
    "qoe_wan_latest",
    "qoe_wifi_latest",
    "qoe_mesh_latest",
    "qoe_composite_mean_24h",
    "qoe_composite_std_24h",
    "qoe_composite_min_24h",
    "qoe_composite_trend_7d",
    "qoe_drops_24h",
    "qoe_below_70_pct_7d",
    "qoe_worst_category",
    # WiFi
    "wifi_client_count",
    "wifi_avg_rssi",
    "wifi_avg_retransmit_rate",
    "wifi_airtime_util_max",
    "mesh_satellite_count",
    "mesh_backhaul_min_dl_mbps",
    # Traffic
    "traffic_total_bytes_1h",
    "traffic_max_risk_score",
    "traffic_streaming_pct",
    # Events
    "events_critical_24h",
    "events_warning_24h",
    "events_total_7d",
    "events_connection_lost_7d",
    # Device metadata
    "model_encoded",
    "firmware_encoded",
    "firmware_age_days",
    "device_age_days",
    "provisioned_dl_mbps",
    # Subscriber
    "subscriber_device_count",
    "subscriber_avg_qoe_7d",
    "subscriber_min_qoe_7d",
    "subscriber_total_events_30d",
    "subscriber_speed_test_frequency",
    "subscriber_tenure_days",
    "subscriber_dl_utilization_ratio",
]

RISK_LEVELS = {
    (0, 25): "low",
    (25, 50): "medium",
    (50, 75): "high",
    (75, 101): "critical",
}


def score_to_risk_level(score: float) -> str:
    """Convert 0-100 risk score to risk level string."""
    for (lo, hi), level in RISK_LEVELS.items():
        if lo <= score < hi:
            return level
    return "critical"


class ChurnPredictor(ModelWrapper):
    """HistGradientBoosting-based churn prediction with SHAP explanations."""

    name = "churn_predictor"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._feature_names = CHURN_FEATURES.copy()

        numerical_features = [f for f in CHURN_FEATURES if f not in CATEGORICAL_FEATURES]

        preprocessor = ColumnTransformer(
            [
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    CATEGORICAL_FEATURES,
                ),
                ("num", "passthrough", numerical_features),
            ],
            remainder="drop",
        )

        self._pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingClassifier(
                max_iter=500,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=1.0,
                max_bins=255,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                scoring="roc_auc",
                random_state=42,
            )),
        ])

    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, float]:
        """Train churn predictor.

        Args:
            X: Feature matrix with CHURN_FEATURES columns.
            y: Binary target (1 = churned, 0 = retained).
        """
        if y is None:
            raise ValueError("Churn predictor requires target labels (y)")

        X_clean = self._validate_features(X.copy()).fillna(0)

        # Time-based split for evaluation
        tscv = TimeSeriesSplit(n_splits=3)
        auc_scores = []

        for train_idx, val_idx in tscv.split(X_clean):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            self._pipeline.fit(X_tr, y_tr)
            proba = self._pipeline.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, proba))

        # Final fit on all data
        self._pipeline.fit(X_clean, y)
        self._is_fitted = True

        # Evaluation on full dataset
        self._pipeline.predict(X_clean)
        y_proba = self._pipeline.predict_proba(X_clean)[:, 1]

        return {
            "roc_auc_cv_mean": float(np.mean(auc_scores)),
            "roc_auc_cv_std": float(np.std(auc_scores)),
            "roc_auc_full": float(roc_auc_score(y, y_proba)),
            "n_samples": len(X_clean),
            "n_churned": int(y.sum()),
            "churn_rate": float(y.mean()),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn risk scores (0-100) for each row."""
        X_clean = self._validate_features(X.copy()).fillna(0)
        proba = self._pipeline.predict_proba(X_clean)[:, 1]
        return proba * 100.0

    def predict_single(self, features: dict[str, float]) -> dict:
        """Predict churn for a single subscriber.

        Returns dict matching ChurnPrediction from src/types/ml.ts.
        """
        df = pd.DataFrame([features])
        scores = self.predict(df)
        score = float(scores[0])

        top_factors = self.get_top_factors(features)

        return {
            "riskScore": round(score, 1),
            "riskLevel": score_to_risk_level(score),
            "topFactors": top_factors,
        }

    def get_top_factors(
        self, features: dict[str, float], n_top: int = 5
    ) -> list[str]:
        """Get top contributing features using model feature importances.

        Falls back to feature importance when SHAP is unavailable.
        """
        try:
            model: HistGradientBoostingClassifier = self._pipeline.named_steps["model"]
            importances = model.feature_importances_
        except AttributeError:
            # Fallback: equal importance if model hasn't computed importances
            n_feats = len(CATEGORICAL_FEATURES) + len(self._feature_names) - len(CATEGORICAL_FEATURES)
            importances = [1.0 / max(n_feats, 1)] * n_feats

        # Map importance back to original feature names
        # After ColumnTransformer, order is: categorical first, then numerical
        numerical_features = [f for f in self._feature_names if f not in CATEGORICAL_FEATURES]
        all_features = CATEGORICAL_FEATURES + numerical_features

        scored = sorted(
            zip(all_features, importances, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top N feature names where the device value is notable
        top = []
        for feat_name, importance in scored[:n_top]:
            val = features.get(feat_name, 0.0)
            top.append(f"{feat_name}={val:.1f} (importance={importance:.3f})")
        return top
