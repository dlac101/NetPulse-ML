"""Abstract base class for all ML model wrappers."""

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()


class ModelWrapper(ABC):
    """Abstract interface for ML models in the NetPulse pipeline."""

    name: str = "base"
    version: str = "0.0.0"

    def __init__(self) -> None:
        self._pipeline: object | None = None
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, float]:
        """Train the model. Returns a dict of evaluation metrics."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input features."""

    def save(self, path: Path) -> None:
        """Save model artifact to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": self._pipeline,
            "feature_names": self._feature_names,
            "name": self.name,
            "version": self.version,
        }
        joblib.dump(artifact, path)
        log.info("Model saved", name=self.name, version=self.version, path=str(path))

    def load(self, path: Path) -> None:
        """Load model artifact from disk."""
        artifact = joblib.load(path)
        self._pipeline = artifact["pipeline"]
        self._feature_names = artifact["feature_names"]
        self.version = artifact.get("version", self.version)
        self._is_fitted = True
        log.info("Model loaded", name=self.name, version=self.version, path=str(path))

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure input has expected features, filling missing with 0.

        Always operates on a copy to avoid mutating the caller's DataFrame.
        """
        X = X.copy()
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            for col in missing:
                X[col] = 0.0
        return X[self._feature_names]
