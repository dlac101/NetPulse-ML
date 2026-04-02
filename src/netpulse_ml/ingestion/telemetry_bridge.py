"""SmartOS Telemetry Bridge: polls real router data and publishes to MQTT.

Connects to a SmartOS router via JUCI WebSocket, polls telemetry at intervals,
and publishes it to the local Mosquitto broker in the format our MQTT consumer
expects. This bridges the gap until routers push telemetry directly.

Usage:
    PYTHONPATH=src python -m netpulse_ml.ingestion.telemetry_bridge
"""

import asyncio
import json
from datetime import UTC, datetime

import structlog
from gmqtt import Client as MQTTClient
from gmqtt.mqtt.constants import MQTTv311

from netpulse_ml.agents.smartos_client import SmartOSClient
from netpulse_ml.config import settings

log = structlog.get_logger()

POLL_INTERVAL = 60  # seconds between polls


class TelemetryBridge:
    """Polls SmartOS router and publishes telemetry to MQTT."""

    def __init__(self) -> None:
        self._router: SmartOSClient | None = None
        self._mqtt: MQTTClient | None = None
        self._device_mac: str = ""
        self._running = True

    async def start(self) -> None:
        """Connect to router and MQTT broker, start polling."""
        # Connect to router
        self._router = SmartOSClient()
        await self._router.connect()
        await self._router.login()

        # Get router's own MAC for device_id
        board = await self._router.call("system", "board")
        self._device_mac = board.get("system", "smartos-router")
        log.info("Router connected", model=board.get("model"), mac=self._device_mac)

        # Connect to MQTT broker
        self._mqtt = MQTTClient(client_id="telemetry-bridge")
        await self._mqtt.connect(settings.mqtt_broker, port=settings.mqtt_port, version=MQTTv311)
        log.info("MQTT broker connected", broker=settings.mqtt_broker)

        # Start polling loop
        while self._running:
            try:
                await self._poll_and_publish()
            except Exception as e:
                log.warning("Poll cycle failed", error=str(e))
            await asyncio.sleep(POLL_INTERVAL)

    async def _poll_and_publish(self) -> None:
        """Poll all telemetry sources and publish to MQTT."""
        device_id = self._device_mac
        now = datetime.now(UTC).isoformat()

        # 1. FlowStatd device stats
        await self._publish_flowstats(device_id, now)

        # 2. Wireless status -> WiFi payload
        await self._publish_wifi(device_id, now)

        # 3. System info -> meta heartbeat
        await self._publish_meta(device_id, now)

        log.info("Telemetry published", device_id=device_id)

    async def _publish_flowstats(self, device_id: str, now: str) -> None:
        """Poll FlowStatd and publish device traffic stats."""
        try:
            result = await self._router._request(
                "call", [self._router._sid, "/juci/flowstats", "devices", {}]
            )
            if isinstance(result, list) and len(result) > 1:
                data = result[1]
            else:
                data = result
            if isinstance(data, dict) and "parse_this_json" in data:
                data = json.loads(data["parse_this_json"])

            devices = data.get("devices", [])

            # Publish per-device flowstats as our expected format
            for dev in devices[:50]:  # Cap at 50 devices
                mac = dev.get("mac", "")
                if not mac:
                    continue

                # Format as TrafficCategory-like payload
                payload = {
                    "device_mac": mac,
                    "categories": [{
                        "masterProtocol": "IP",
                        "appProtocol": dev.get("name", "unknown"),
                        "category": "General",
                        "rxBytes": dev.get("rx_bytes", 0),
                        "txBytes": dev.get("tx_bytes", 0),
                        "totalBytes": dev.get("rx_bytes", 0) + dev.get("tx_bytes", 0),
                        "activeTimeSec": 0,
                        "flowCount": 1,
                        "rxRate": dev.get("rx_bps", 0),
                        "txRate": dev.get("tx_bps", 0),
                        "maxRiskScore": 0,
                    }],
                }

                topic = f"smartos/{mac.replace(':', '')}/flowstatd/stats"
                self._mqtt.publish(topic, json.dumps(payload).encode(), qos=0)

        except Exception as e:
            log.debug("FlowStatd poll failed", error=str(e))

    async def _publish_wifi(self, device_id: str, now: str) -> None:
        """Poll wireless status and publish WiFi payload."""
        try:
            wifi = await self._router.get_wireless_status()

            clients = []
            airtime = []

            for _radio_name, radio_data in wifi.items():
                if not isinstance(radio_data, dict) or "config" not in radio_data:
                    continue

                config = radio_data["config"]
                band_map = {"2g": "2.4GHz", "5g": "5GHz", "6g": "6GHz"}
                band = band_map.get(config.get("band", ""), "5GHz")
                channel = config.get("channel", 0)

                # Build airtime entry
                airtime.append({
                    "band": band,
                    "channel": int(channel) if str(channel).isdigit() else 0,
                    "txPercent": 0,
                    "rxPercent": 0,
                    "wifiInterferencePercent": 0,
                    "nonWifiInterferencePercent": 0,
                    "totalUtilizationPercent": 0,
                    "clientCount": 0,
                })

                # Extract clients from interfaces
                for iface in radio_data.get("interfaces", []):
                    for assoc in iface.get("assoclist", []):
                        clients.append({
                            "mac": assoc.get("mac", ""),
                            "band": band,
                            "channel": int(channel) if str(channel).isdigit() else 0,
                            "channelWidth": 80,
                            "phyRateMbps": assoc.get("rx", {}).get("rate", 0) / 1000,
                            "rssi": assoc.get("signal", 0),
                            "retransmissionRate": 0,
                            "mcs": assoc.get("rx", {}).get("mcs", 0),
                            "nss": assoc.get("rx", {}).get("nss", 1),
                            "rxBytes": assoc.get("rx", {}).get("bytes", 0),
                            "txBytes": assoc.get("tx", {}).get("bytes", 0),
                        })

            payload = {
                "clients": clients,
                "satellites": [],
                "airtime": airtime,
            }

            topic = f"smartos/{device_id}/wifi"
            self._mqtt.publish(topic, json.dumps(payload).encode(), qos=0)

        except Exception as e:
            log.debug("WiFi poll failed", error=str(e))

    async def _publish_meta(self, device_id: str, now: str) -> None:
        """Publish device metadata heartbeat."""
        try:
            info = await self._router.get_system_info()

            payload = {
                "mac": device_id,
                "model": info.get("model", "unknown"),
                "firmware": info.get("release", {}).get("description", "unknown"),
                "status": "online",
                "timestamp": now,
            }

            topic = f"smartos/{device_id}/meta"
            self._mqtt.publish(topic, json.dumps(payload).encode(), qos=0)

        except Exception as e:
            log.debug("Meta poll failed", error=str(e))

    def stop(self) -> None:
        self._running = False


async def main() -> None:
    bridge = TelemetryBridge()
    try:
        await bridge.start()
    except KeyboardInterrupt:
        bridge.stop()
    except Exception as e:
        log.error("Bridge failed", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
