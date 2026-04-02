"""Tests for the anomaly detection model."""

import numpy as np
import pandas as pd
import pytest

from netpulse_ml.models.anomaly_detector import ANOMALY_FEATURES, AnomalyDetector


@pytest.fixture
def sample_fleet_data() -> pd.DataFrame:
    """Generate a small fleet dataset for testing."""
    np.random.seed(42)
    n = 100
    data = {}
    for feat in ANOMALY_FEATURES:
        data[feat] = np.random.randn(n) * 10 + 50
    return pd.DataFrame(data)


@pytest.fixture
def trained_detector(sample_fleet_data: pd.DataFrame) -> AnomalyDetector:
    """Return a trained anomaly detector."""
    model = AnomalyDetector(contamination=0.05, n_estimators=50)
    model.train(sample_fleet_data)
    return model


class TestAnomalyDetector:
    def test_init(self) -> None:
        model = AnomalyDetector()
        assert model.name == "anomaly_detector"
        assert not model.is_fitted
        assert len(model.feature_names) == 21

    def test_train_returns_metrics(self, sample_fleet_data: pd.DataFrame) -> None:
        model = AnomalyDetector(contamination=0.05, n_estimators=50)
        metrics = model.train(sample_fleet_data)

        assert model.is_fitted
        assert metrics["n_samples"] == 100
        assert metrics["n_features"] == 21
        assert 0 <= metrics["anomaly_rate"] <= 1

    def test_predict_returns_scores(
        self, trained_detector: AnomalyDetector, sample_fleet_data: pd.DataFrame
    ) -> None:
        scores = trained_detector.predict(sample_fleet_data)

        assert len(scores) == 100
        assert all(0 <= s <= 1 for s in scores)

    def test_predict_single(self, trained_detector: AnomalyDetector) -> None:
        features = {feat: 50.0 for feat in ANOMALY_FEATURES}
        score = trained_detector.predict_single(features)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_get_top_features(self, trained_detector: AnomalyDetector) -> None:
        features = {feat: 50.0 for feat in ANOMALY_FEATURES}
        features["latency_idle_ms"] = 500.0  # Extreme value

        top = trained_detector.get_top_features(features, n_top=3)

        assert len(top) == 3
        assert all("name" in f and "zscore" in f for f in top)

    def test_handles_missing_features(self, trained_detector: AnomalyDetector) -> None:
        # Only provide a subset of features
        features = {"dl_mbps_latest": 100.0, "qoe_composite_latest": 80.0}
        score = trained_detector.predict_single(features)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_save_and_load(self, trained_detector: AnomalyDetector, tmp_path) -> None:
        path = tmp_path / "test_model.joblib"
        trained_detector.save(path)

        loaded = AnomalyDetector()
        loaded.load(path)

        assert loaded.is_fitted
        assert loaded.version == trained_detector.version

        # Predictions should be identical
        features = {feat: 50.0 for feat in ANOMALY_FEATURES}
        assert trained_detector.predict_single(features) == loaded.predict_single(features)
