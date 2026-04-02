"""Fleet segmentation using DBSCAN outlier detection + KMeans clustering.

Two-stage pipeline:
  1. DBSCAN identifies outlier devices (label=-1)
  2. KMeans segments non-outlier devices into clusters
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from netpulse_ml.models.base import ModelWrapper

log = structlog.get_logger()

CLUSTER_FEATURES = [
    "dl_mbps_latest",
    "ul_mbps_latest",
    "qoe_composite_latest",
    "wifi_client_count",
    "wifi_avg_rssi",
    "traffic_total_bytes_1h",
    "traffic_streaming_pct",
    "mesh_satellite_count",
    "dl_capacity_pct",
    "events_total_7d",
    "device_age_days",
    "wifi_airtime_util_max",
    "latency_idle_ms",
]


class FleetClusterer(ModelWrapper):
    """Two-stage fleet segmentation: DBSCAN outliers + KMeans segments."""

    name = "fleet_clusterer"
    version = "1.0.0"

    def __init__(self, n_clusters: int = 8) -> None:
        super().__init__()
        self._feature_names = CLUSTER_FEATURES.copy()
        self._n_clusters = n_clusters
        self._scaler = StandardScaler()
        self._dbscan: DBSCAN | None = None
        self._kmeans: KMeans | None = None
        self._centroids: np.ndarray | None = None
        # Stored from training for predict-time outlier detection
        self._dbscan_eps: float = 0.5
        self._dbscan_core_samples: np.ndarray | None = None

    def save(self, path: Path) -> None:
        """Save all clusterer state (scaler, DBSCAN, KMeans, core samples)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "feature_names": self._feature_names,
            "name": self.name,
            "version": self.version,
            "scaler": self._scaler,
            "dbscan": self._dbscan,
            "kmeans": self._kmeans,
            "centroids": self._centroids,
            "dbscan_eps": self._dbscan_eps,
            "dbscan_core_samples": self._dbscan_core_samples,
            "n_clusters": self._n_clusters,
        }
        joblib.dump(artifact, path)
        log.info("Clusterer saved", name=self.name, path=str(path))

    def load(self, path: Path) -> None:
        """Load all clusterer state."""
        artifact = joblib.load(path)
        self._feature_names = artifact["feature_names"]
        self.version = artifact.get("version", self.version)
        self._scaler = artifact["scaler"]
        self._dbscan = artifact["dbscan"]
        self._kmeans = artifact["kmeans"]
        self._centroids = artifact["centroids"]
        self._dbscan_eps = artifact["dbscan_eps"]
        self._dbscan_core_samples = artifact["dbscan_core_samples"]
        self._n_clusters = artifact["n_clusters"]
        self._is_fitted = True
        log.info("Clusterer loaded", name=self.name, path=str(path))

    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, float]:
        """Train the two-stage clustering pipeline.

        Args:
            X: Fleet feature matrix (one row per device).
        """
        X_clean = self._validate_features(X.copy()).fillna(0)
        X_scaled = self._scaler.fit_transform(X_clean)

        # Stage 1: DBSCAN for outlier detection
        eps = self._find_optimal_eps(X_scaled)
        self._dbscan = DBSCAN(
            eps=eps,
            min_samples=max(10, len(X_clean) // 200),
            metric="euclidean",
            n_jobs=-1,
        )
        dbscan_labels = self._dbscan.fit_predict(X_scaled)
        self._dbscan_eps = eps
        # Store core samples for predict-time outlier detection (avoids re-fitting)
        core_mask = np.zeros(len(X_scaled), dtype=bool)
        if hasattr(self._dbscan, "core_sample_indices_") and len(self._dbscan.core_sample_indices_) > 0:
            core_mask[self._dbscan.core_sample_indices_] = True
        self._dbscan_core_samples = X_scaled[core_mask] if core_mask.any() else X_scaled

        outlier_mask = dbscan_labels == -1
        n_outliers = int(outlier_mask.sum())

        # Stage 2: KMeans on non-outliers
        X_non_outlier = X_scaled[~outlier_mask]

        if len(X_non_outlier) < self._n_clusters:
            self._n_clusters = max(2, len(X_non_outlier) // 2)

        # Auto-tune n_clusters via silhouette score
        best_k = self._n_clusters
        best_score = -1.0
        if len(X_non_outlier) > 20:
            for k in range(max(2, self._n_clusters - 3), self._n_clusters + 4):
                if k >= len(X_non_outlier):
                    break
                km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
                labels = km.fit_predict(X_non_outlier)
                if len(set(labels)) > 1:
                    # Sample for silhouette to avoid O(n^2) on large fleets
                    if len(X_non_outlier) > 10000:
                        sample_idx = np.random.choice(len(X_non_outlier), 10000, replace=False)
                        score = silhouette_score(X_non_outlier[sample_idx], labels[sample_idx])
                    else:
                        score = silhouette_score(X_non_outlier, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

        self._n_clusters = best_k
        self._kmeans = KMeans(
            n_clusters=best_k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        kmeans_labels = self._kmeans.fit_predict(X_non_outlier)
        self._centroids = self._kmeans.cluster_centers_

        self._is_fitted = True

        sil_score = float(best_score) if best_score > -1 else 0.0

        return {
            "n_devices": len(X_clean),
            "n_outliers": n_outliers,
            "outlier_rate": n_outliers / max(len(X_clean), 1),
            "n_clusters": best_k,
            "silhouette_score": sil_score,
            "dbscan_eps": eps,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign devices to clusters. Outliers get label -1.

        Uses stored core samples from training for outlier detection (NearestNeighbors)
        instead of re-fitting DBSCAN, ensuring deterministic assignments.
        """
        X_clean = self._validate_features(X.copy()).fillna(0)
        X_scaled = self._scaler.transform(X_clean)

        result = np.full(len(X_scaled), -1, dtype=int)

        # Outlier detection: distance to nearest training core sample > eps = outlier
        if self._dbscan_core_samples is not None and len(self._dbscan_core_samples) > 0:
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self._dbscan_core_samples)
            distances, _ = nn.kneighbors(X_scaled)
            non_outlier_mask = distances.ravel() <= self._dbscan_eps
        else:
            non_outlier_mask = np.ones(len(X_scaled), dtype=bool)

        # KMeans for non-outliers
        if non_outlier_mask.any() and self._kmeans is not None:
            kmeans_labels = self._kmeans.predict(X_scaled[non_outlier_mask])
            result[non_outlier_mask] = kmeans_labels

        return result

    def get_cluster_summary(self, X: pd.DataFrame) -> list[dict]:
        """Generate a summary of each cluster's characteristics."""
        X_clean = self._validate_features(X.copy()).fillna(0)
        labels = self.predict(X_clean)

        summaries = []
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            mask = labels == label
            cluster_data = X_clean[mask]

            summary = {
                "clusterId": int(label),
                "label": self._generate_cluster_label(cluster_data),
                "isOutlier": label == -1,
                "deviceCount": int(mask.sum()),
            }

            # Add centroid stats
            for feat in self._feature_names:
                if feat in cluster_data.columns:
                    summary[f"avg_{feat}"] = round(float(cluster_data[feat].mean()), 2)

            summaries.append(summary)

        return summaries

    def _generate_cluster_label(self, cluster_data: pd.DataFrame) -> str:
        """Generate a human-readable label from cluster centroid characteristics."""
        if cluster_data.empty:
            return "Unknown"

        avg_qoe = cluster_data.get("qoe_composite_latest", pd.Series([50])).mean()
        avg_dl = cluster_data.get("dl_mbps_latest", pd.Series([100])).mean()
        avg_clients = cluster_data.get("wifi_client_count", pd.Series([0])).mean()
        avg_streaming = cluster_data.get("traffic_streaming_pct", pd.Series([0])).mean()

        parts = []
        if avg_qoe >= 85:
            parts.append("High-QoE")
        elif avg_qoe < 60:
            parts.append("Low-QoE")

        if avg_dl >= 500:
            parts.append("High-Speed")
        elif avg_dl < 100:
            parts.append("Low-Speed")

        if avg_clients >= 15:
            parts.append("Heavy-Use")
        elif avg_clients <= 3:
            parts.append("Light-Use")

        if avg_streaming > 0.5:
            parts.append("Streamers")

        return " ".join(parts) if parts else "Standard"

    def _find_optimal_eps(self, X_scaled: np.ndarray) -> float:
        """Find optimal DBSCAN eps using k-nearest neighbor distance knee method."""
        k = max(10, len(X_scaled) // 200)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])

        # Simple knee detection: find the point of maximum curvature
        if len(k_distances) < 3:
            return 0.5

        diffs = np.diff(k_distances)
        diffs2 = np.diff(diffs)
        if len(diffs2) == 0:
            return float(np.median(k_distances))

        knee_idx = np.argmax(diffs2) + 1
        return float(k_distances[knee_idx])
