"""Tests for SmartOS agent tools."""

import pytest

from netpulse_ml.agents.tools import (
    TOOL_REGISTRY,
    configure_band_steering,
    enable_sqm,
    reboot_device,
    schedule_firmware_upgrade,
)


class TestTools:
    @pytest.mark.asyncio
    async def test_enable_sqm(self) -> None:
        result = await enable_sqm("dev-001")
        assert result["success"] is True
        assert result["device_id"] == "dev-001"
        assert result["action"] == "sqm_enabled"

    @pytest.mark.asyncio
    async def test_band_steering(self) -> None:
        result = await configure_band_steering("dev-002")
        assert result["success"] is True
        assert result["action"] == "band_steering_enabled"

    @pytest.mark.asyncio
    async def test_firmware_upgrade(self) -> None:
        result = await schedule_firmware_upgrade("dev-003", "23.11.1.2")
        assert result["success"] is True
        assert result["target_version"] == "23.11.1.2"

    @pytest.mark.asyncio
    async def test_reboot(self) -> None:
        result = await reboot_device("dev-004")
        assert result["success"] is True

    def test_tool_registry(self) -> None:
        assert "enable_sqm" in TOOL_REGISTRY
        assert "band_steering" in TOOL_REGISTRY
        assert "firmware_upgrade" in TOOL_REGISTRY
        assert TOOL_REGISTRY["mesh_ap_add"] is None  # No auto tool
        assert TOOL_REGISTRY["service_tier_change"] is None
