"""Tests for fleet clusterer model."""

import numpy as np
import pandas as pd
import pytest

from netpulse_ml.models.fleet_clusterer import CLUSTER_FEATURES, FleetClusterer


@pytest.fixture
def sample_fleet_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 50
    data = {}
    for feat in CLUSTER_FEATURES:
        data[feat] = np.random.randn(n) * 10 + 50
    return pd.DataFrame(data)


class TestFleetClusterer:
    def test_init(self) -> None:
        model = FleetClusterer()
        assert model.name == "fleet_clusterer"
        assert not model.is_fitted
        assert len(model.feature_names) == 13

    def test_train(self, sample_fleet_data: pd.DataFrame) -> None:
        model = FleetClusterer(n_clusters=3)
        metrics = model.train(sample_fleet_data)

        assert model.is_fitted
        assert "n_clusters" in metrics
        assert "n_outliers" in metrics
        assert "silhouette_score" in metrics
        assert metrics["n_devices"] == 50

    def test_predict(self, sample_fleet_data: pd.DataFrame) -> None:
        model = FleetClusterer(n_clusters=3)
        model.train(sample_fleet_data)

        labels = model.predict(sample_fleet_data)
        assert len(labels) == 50
        # Should have some non-outlier labels (>= 0)
        assert any(l >= 0 for l in labels)

    def test_cluster_summary(self, sample_fleet_data: pd.DataFrame) -> None:
        model = FleetClusterer(n_clusters=3)
        model.train(sample_fleet_data)

        summaries = model.get_cluster_summary(sample_fleet_data)
        assert len(summaries) > 0
        for s in summaries:
            assert "clusterId" in s
            assert "label" in s
            assert "deviceCount" in s

    def test_save_and_load(self, sample_fleet_data: pd.DataFrame, tmp_path) -> None:
        model = FleetClusterer(n_clusters=3)
        model.train(sample_fleet_data)

        path = tmp_path / "clusterer.joblib"
        model.save(path)

        loaded = FleetClusterer()
        loaded.load(path)

        assert loaded.is_fitted
        labels_orig = model.predict(sample_fleet_data)
        labels_loaded = loaded.predict(sample_fleet_data)
        np.testing.assert_array_equal(labels_orig, labels_loaded)
