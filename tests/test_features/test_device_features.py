"""Tests for device feature extraction functions."""

import pytest

from netpulse_ml.features.device_features import (
    extract_bbst_features,
    extract_qoe_features,
    extract_wifi_features,
)
from netpulse_ml.ingestion.validators import (
    BbstPayload,
    LatencyMetrics,
    QoECategories,
    QoECategory,
    QoEPayload,
    SpeedMetrics,
    WifiAirtime,
    WifiClient,
    WifiPayload,
)


class TestBbstFeatures:
    def test_basic_extraction(self) -> None:
        payload = BbstPayload(
            id="test-1",
            deviceId="dev-1",
            timestamp="2026-04-01T12:00:00Z",
            download=SpeedMetrics(mbps=500, capacityPercent=85, bloatPercent=30, bloatGrade="B", provisionedMbps=600),
            upload=SpeedMetrics(mbps=100, capacityPercent=90, bloatPercent=10, bloatGrade="A", provisionedMbps=110),
            latency=LatencyMetrics(idleMs=5.0, downloadMs=15.0, uploadMs=8.0, idleJitterMs=1.0, downloadJitterMs=3.0, uploadJitterMs=1.5),
        )
        features = extract_bbst_features(payload)

        assert features["dl_mbps_latest"] == 500
        assert features["ul_mbps_latest"] == 100
        assert features["latency_idle_ms"] == 5.0
        assert features["dl_bloat_pct"] == pytest.approx(200.0)  # (15-5)/5*100
        assert features["provisioned_dl_mbps"] == 600

    def test_zero_idle_latency(self) -> None:
        """Should not divide by zero when idle latency is 0."""
        payload = BbstPayload(
            id="test-2",
            deviceId="dev-2",
            timestamp="2026-04-01T12:00:00Z",
            download=SpeedMetrics(mbps=100),
            upload=SpeedMetrics(mbps=50),
            latency=LatencyMetrics(idleMs=0.0, downloadMs=10.0, uploadMs=5.0),
        )
        features = extract_bbst_features(payload)
        assert features["dl_bloat_pct"] >= 0  # Should not raise


class TestQoEFeatures:
    def _make_cat(self, name: str, score: float) -> QoECategory:
        return QoECategory(name=name, score=score, grade="B", weight=0.2, impactFactors=[])

    def test_basic_extraction(self) -> None:
        payload = QoEPayload(
            id="qoe-1",
            deviceId="dev-1",
            timestamp="2026-04-01T12:00:00Z",
            compositeScore=78,
            compositeGrade="B",
            categories=QoECategories(
                wan=self._make_cat("wan", 85),
                wifi=self._make_cat("wifi", 70),
                mesh=self._make_cat("mesh", 90),
                system=self._make_cat("system", 75),
                lan=self._make_cat("lan", 80),
            ),
        )
        features = extract_qoe_features(payload)

        assert features["qoe_composite_latest"] == 78
        assert features["qoe_wan_latest"] == 85
        assert features["qoe_wifi_latest"] == 70
        assert features["qoe_worst_category"] == 1  # wifi=1 (lowest score)


class TestWifiFeatures:
    def test_empty_clients(self) -> None:
        payload = WifiPayload(clients=[], satellites=[], airtime=[])
        features = extract_wifi_features(payload)
        assert features["wifi_client_count"] == 0.0
        assert features["wifi_avg_rssi"] == 0.0

    def test_band_distribution(self) -> None:
        clients = [
            WifiClient(mac="aa:bb:cc:dd:ee:01", band="2.4GHz", channel=6),
            WifiClient(mac="aa:bb:cc:dd:ee:02", band="2.4GHz", channel=6),
            WifiClient(mac="aa:bb:cc:dd:ee:03", band="5GHz", channel=36),
        ]
        payload = WifiPayload(clients=clients, satellites=[], airtime=[])
        features = extract_wifi_features(payload)

        assert features["wifi_client_count"] == 3.0
        assert features["wifi_pct_2_4ghz"] == pytest.approx(2 / 3)
        assert features["wifi_pct_5ghz"] == pytest.approx(1 / 3)
