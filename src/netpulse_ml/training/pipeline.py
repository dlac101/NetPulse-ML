"""Training pipeline: orchestrates train -> evaluate -> register for all models."""

from datetime import datetime, timezone
from pathlib import Path

import structlog

from netpulse_ml.config import settings
from netpulse_ml.features.store import feature_store
from netpulse_ml.models.anomaly_detector import AnomalyDetector
from netpulse_ml.models.churn_predictor import ChurnPredictor
from netpulse_ml.models.fleet_clusterer import FleetClusterer
from netpulse_ml.models.registry import register_model

log = structlog.get_logger()


async def train_anomaly_detector() -> dict[str, float]:
    """Train the anomaly detection model on latest fleet features."""
    log.info("Training anomaly detector")

    # Get latest features for all devices
    fleet_df = await feature_store.get_fleet_features()
    if fleet_df.empty:
        log.warning("No feature data available for anomaly training")
        return {"error": 1.0, "message_no_data": 1.0}

    model = AnomalyDetector()
    metrics = model.train(fleet_df)

    # Save artifact
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model.version = version
    artifact_path = settings.model_dir / f"anomaly_detector.joblib"
    model.save(artifact_path)

    # Register in DB
    await register_model(model, artifact_path, metrics, activate=True)

    log.info("Anomaly detector trained", **metrics)
    return metrics


async def train_churn_predictor() -> dict[str, float]:
    """Train the churn prediction model.

    Note: Requires labeled data. In cold-start, uses proxy labels.
    """
    log.info("Training churn predictor")

    fleet_df = await feature_store.get_fleet_features()
    if fleet_df.empty:
        log.warning("No feature data available for churn training")
        return {"error": 1.0}

    # Cold-start proxy labels:
    # Devices with events_connection_lost_7d > 5 AND qoe below 50 -> proxy churned
    # This is a placeholder; real labels come from CRM integration
    import pandas as pd

    y = pd.Series(0, index=fleet_df.index)
    if "events_connection_lost_7d" in fleet_df.columns:
        high_disconnect = fleet_df.get("events_connection_lost_7d", 0) > 5
        low_qoe = fleet_df.get("qoe_composite_latest", 100) < 50
        y[high_disconnect & low_qoe] = 1

    if y.sum() < 5:
        # Not enough positive labels for training
        log.warning("Insufficient churn labels", n_positive=int(y.sum()))
        return {"error": 1.0, "insufficient_labels": 1.0}

    model = ChurnPredictor()
    metrics = model.train(fleet_df, y)

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model.version = version
    artifact_path = settings.model_dir / "churn_predictor.joblib"
    model.save(artifact_path)

    await register_model(model, artifact_path, metrics, activate=True)

    log.info("Churn predictor trained", **metrics)
    return metrics


async def train_fleet_clusterer() -> dict[str, float]:
    """Train fleet segmentation (nightly batch)."""
    log.info("Training fleet clusterer")

    fleet_df = await feature_store.get_fleet_features()
    if fleet_df.empty or len(fleet_df) < 20:
        log.warning("Not enough devices for clustering", n_devices=len(fleet_df))
        return {"error": 1.0}

    model = FleetClusterer()
    metrics = model.train(fleet_df)

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model.version = version
    artifact_path = settings.model_dir / "fleet_clusterer.joblib"
    model.save(artifact_path)

    await register_model(model, artifact_path, metrics, activate=True)

    log.info("Fleet clusterer trained", **metrics)
    return metrics


async def train_all() -> dict[str, dict]:
    """Train all models in sequence."""
    results = {}
    results["anomaly_detector"] = await train_anomaly_detector()
    results["churn_predictor"] = await train_churn_predictor()
    results["fleet_clusterer"] = await train_fleet_clusterer()
    return results
