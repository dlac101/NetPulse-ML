"""Tests for the churn prediction model."""

import numpy as np
import pandas as pd
import pytest

from netpulse_ml.models.churn_predictor import CHURN_FEATURES, ChurnPredictor, score_to_risk_level


@pytest.fixture
def sample_churn_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate labeled churn dataset."""
    np.random.seed(42)
    n = 200
    data = {}
    for feat in CHURN_FEATURES:
        data[feat] = np.random.randn(n) * 10 + 50
    X = pd.DataFrame(data)
    # 10% churn rate
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.9, 0.1]))
    return X, y


class TestChurnPredictor:
    def test_init(self) -> None:
        model = ChurnPredictor()
        assert model.name == "churn_predictor"
        assert not model.is_fitted

    def test_train_requires_labels(self) -> None:
        model = ChurnPredictor()
        X = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="requires target"):
            model.train(X)

    def test_train_returns_metrics(
        self, sample_churn_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_churn_data
        model = ChurnPredictor()
        metrics = model.train(X, y)

        assert model.is_fitted
        assert "roc_auc_full" in metrics
        assert 0 <= metrics["roc_auc_full"] <= 1
        assert metrics["n_samples"] == 200

    def test_predict_returns_0_100(
        self, sample_churn_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_churn_data
        model = ChurnPredictor()
        model.train(X, y)

        scores = model.predict(X)
        assert len(scores) == 200
        assert all(0 <= s <= 100 for s in scores)

    def test_predict_single(
        self, sample_churn_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_churn_data
        model = ChurnPredictor()
        model.train(X, y)

        features = {feat: 50.0 for feat in CHURN_FEATURES}
        result = model.predict_single(features)

        assert "riskScore" in result
        assert "riskLevel" in result
        assert "topFactors" in result
        assert 0 <= result["riskScore"] <= 100

    def test_score_to_risk_level(self) -> None:
        assert score_to_risk_level(10) == "low"
        assert score_to_risk_level(30) == "medium"
        assert score_to_risk_level(60) == "high"
        assert score_to_risk_level(90) == "critical"
