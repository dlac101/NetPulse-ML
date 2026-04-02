"""Tests for agent node implementations."""

import pytest

from netpulse_ml.agents.nodes import diagnose_node, plan_node


class TestDiagnoseNode:
    @pytest.mark.asyncio
    async def test_bufferbloat_diagnosis(self) -> None:
        state = {
            "device_id": "dev-001",
            "top_anomaly_features": [
                {"name": "dl_bloat_pct", "value": 350.0, "zscore": 3.5},
                {"name": "latency_idle_ms", "value": 8.0, "zscore": 1.0},
            ],
            "features": {"wifi_pct_2_4ghz": 0.3},
        }
        result = await diagnose_node(state)
        assert result["diagnosis"] == "bufferbloat"
        assert result["status"] == "diagnosed"

    @pytest.mark.asyncio
    async def test_band_congestion_diagnosis(self) -> None:
        state = {
            "device_id": "dev-002",
            "top_anomaly_features": [
                {"name": "wifi_client_count", "value": 15.0, "zscore": 1.0},
            ],
            "features": {"wifi_pct_2_4ghz": 0.75},  # > 0.6 threshold
        }
        result = await diagnose_node(state)
        assert result["diagnosis"] == "band_congestion"

    @pytest.mark.asyncio
    async def test_no_diagnosis(self) -> None:
        state = {
            "device_id": "dev-003",
            "top_anomaly_features": [
                {"name": "dl_mbps_latest", "value": 500.0, "zscore": 0.5},
            ],
            "features": {"wifi_pct_2_4ghz": 0.2},
        }
        result = await diagnose_node(state)
        assert result["diagnosis"] == "none"

    @pytest.mark.asyncio
    async def test_weak_wifi_diagnosis(self) -> None:
        state = {
            "device_id": "dev-004",
            "top_anomaly_features": [
                {"name": "wifi_avg_rssi", "value": -78.0, "zscore": 2.5},
            ],
            "features": {"wifi_pct_2_4ghz": 0.3},
        }
        result = await diagnose_node(state)
        assert result["diagnosis"] == "weak_wifi_signal"


class TestPlanNode:
    @pytest.mark.asyncio
    async def test_bufferbloat_plan(self) -> None:
        state = {"diagnosis": "bufferbloat", "anomaly_score": 0.85}
        result = await plan_node(state)
        assert result["recommended_action"] == "enable_sqm"
        assert result["auto_executable"] is True
        assert result["action_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_weak_wifi_plan(self) -> None:
        state = {"diagnosis": "weak_wifi_signal", "anomaly_score": 0.7}
        result = await plan_node(state)
        assert result["recommended_action"] == "mesh_ap_add"
        assert result["auto_executable"] is False

    @pytest.mark.asyncio
    async def test_band_congestion_plan(self) -> None:
        state = {"diagnosis": "band_congestion", "anomaly_score": 0.6}
        result = await plan_node(state)
        assert result["recommended_action"] == "band_steering"
        assert result["auto_executable"] is True

    @pytest.mark.asyncio
    async def test_no_diagnosis_plan(self) -> None:
        state = {"diagnosis": "none", "anomaly_score": 0.3}
        result = await plan_node(state)
        assert result["recommended_action"] is None
