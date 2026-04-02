"""Model serving: loads models into memory for real-time and batch inference."""

from pathlib import Path

import structlog

from netpulse_ml.models.anomaly_detector import AnomalyDetector
from netpulse_ml.models.base import ModelWrapper
from netpulse_ml.models.churn_predictor import ChurnPredictor
from netpulse_ml.models.fleet_clusterer import FleetClusterer
from netpulse_ml.models.qoe_forecaster import QoEForecaster

log = structlog.get_logger()

# Model class mapping
MODEL_CLASSES: dict[str, type[ModelWrapper]] = {
    "anomaly_detector": AnomalyDetector,
    "churn_predictor": ChurnPredictor,
    "qoe_forecaster": QoEForecaster,
    "fleet_clusterer": FleetClusterer,
}


class Predictor:
    """Singleton model server that holds all active models in memory."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._models: dict[str, ModelWrapper] = {}

    @property
    def loaded_model_names(self) -> list[str]:
        return list(self._models.keys())

    def load_all(self) -> None:
        """Load all available model artifacts from the model directory."""
        self._model_dir.mkdir(parents=True, exist_ok=True)

        for name, cls in MODEL_CLASSES.items():
            artifact_path = self._model_dir / f"{name}.joblib"
            model = cls()

            if artifact_path.exists():
                try:
                    model.load(artifact_path)
                    log.info("Model loaded from disk", name=name, path=str(artifact_path))
                except Exception as e:
                    log.warning("Failed to load model, using untrained", name=name, error=str(e))

            self._models[name] = model

    def get_model(self, name: str) -> ModelWrapper | None:
        """Get a loaded model by name."""
        return self._models.get(name)

    @property
    def anomaly_detector(self) -> AnomalyDetector:
        model = self._models.get("anomaly_detector")
        if model is None:
            model = AnomalyDetector()
            self._models["anomaly_detector"] = model
        return model  # type: ignore[return-value]

    @property
    def churn_predictor(self) -> ChurnPredictor:
        model = self._models.get("churn_predictor")
        if model is None:
            model = ChurnPredictor()
            self._models["churn_predictor"] = model
        return model  # type: ignore[return-value]

    @property
    def qoe_forecaster(self) -> QoEForecaster:
        model = self._models.get("qoe_forecaster")
        if model is None:
            model = QoEForecaster()
            self._models["qoe_forecaster"] = model
        return model  # type: ignore[return-value]

    @property
    def fleet_clusterer(self) -> FleetClusterer:
        model = self._models.get("fleet_clusterer")
        if model is None:
            model = FleetClusterer()
            self._models["fleet_clusterer"] = model
        return model  # type: ignore[return-value]

    def reload_model(self, name: str) -> bool:
        """Reload a specific model from disk."""
        cls = MODEL_CLASSES.get(name)
        if cls is None:
            return False

        artifact_path = self._model_dir / f"{name}.joblib"
        if not artifact_path.exists():
            return False

        model = cls()
        model.load(artifact_path)
        self._models[name] = model
        log.info("Model reloaded", name=name)
        return True
