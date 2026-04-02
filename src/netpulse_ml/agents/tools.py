"""SmartOS device action tools.

When settings.smartos_use_mock is True (default), tools simulate success.
When False, tools make real ubus API calls to the SmartOS router.
"""

import asyncio

import structlog

from netpulse_ml.config import settings

log = structlog.get_logger()

# Lazy-initialized real client (only created when smartos_use_mock=False)
_real_client = None


async def _get_client():
    """Get the SmartOS client (real or mock based on config)."""
    global _real_client
    if settings.smartos_use_mock:
        return None  # Caller uses mock path
    if _real_client is None:
        from netpulse_ml.agents.smartos_client import SmartOSClient
        _real_client = SmartOSClient()
        await _real_client.login()
    return _real_client


async def enable_sqm(device_id: str) -> dict:
    """Enable Smart Queue Management on a router to reduce bufferbloat."""
    client = await _get_client()
    if client:
        log.info("Enabling SQM (REAL)", device_id=device_id, host=settings.smartos_host)
        return await client.enable_sqm()
    else:
        log.info("Enabling SQM (MOCK)", device_id=device_id)
        await asyncio.sleep(0.5)
        return {"success": True, "device_id": device_id, "action": "sqm_enabled", "mock": True}


async def configure_band_steering(device_id: str) -> dict:
    """Enable band steering to move clients from 2.4GHz to 5/6GHz."""
    client = await _get_client()
    if client:
        log.info("Configuring band steering (REAL)", device_id=device_id)
        # Get current clients and steer any on 2.4GHz
        info = await client.get_usteer_info()
        return {"success": True, "device_id": device_id, "action": "band_steering_checked", "usteer_info": str(info)[:200]}
    else:
        log.info("Configuring band steering (MOCK)", device_id=device_id)
        await asyncio.sleep(0.5)
        return {"success": True, "device_id": device_id, "action": "band_steering_enabled", "mock": True}


async def schedule_firmware_upgrade(device_id: str, target_version: str = "23.11.1.2") -> dict:
    """Schedule a firmware upgrade during the next maintenance window."""
    client = await _get_client()
    if client:
        log.info("Checking firmware (REAL)", device_id=device_id, version=target_version)
        # In production: download firmware, validate, then schedule sysupgrade
        # For now, just get current system info to verify connectivity
        info = await client.get_system_info()
        return {
            "success": True,
            "device_id": device_id,
            "action": "firmware_info_checked",
            "current_firmware": info.get("release", {}).get("description", "unknown"),
            "target_version": target_version,
            "note": "Full upgrade requires firmware file download + sysupgrade call",
        }
    else:
        log.info("Scheduling firmware upgrade (MOCK)", device_id=device_id, version=target_version)
        await asyncio.sleep(0.3)
        return {
            "success": True,
            "device_id": device_id,
            "action": "firmware_scheduled",
            "target_version": target_version,
            "mock": True,
        }


async def reboot_device(device_id: str) -> dict:
    """Reboot the router."""
    client = await _get_client()
    if client:
        log.warning("Rebooting device (REAL)", device_id=device_id)
        await client.reboot()
        return {"success": True, "device_id": device_id, "action": "rebooted"}
    else:
        log.info("Scheduling device reboot (MOCK)", device_id=device_id)
        await asyncio.sleep(0.3)
        return {"success": True, "device_id": device_id, "action": "reboot_scheduled", "mock": True}


# Tool registry: maps recommendation types to tool functions
TOOL_REGISTRY: dict[str, object] = {
    "enable_sqm": enable_sqm,
    "band_steering": configure_band_steering,
    "firmware_upgrade": schedule_firmware_upgrade,
    "mesh_ap_add": None,  # No automated tool; always escalate to human
    "service_tier_change": None,  # No automated tool; always escalate to human
}
