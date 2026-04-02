"""Async MQTT consumer using gmqtt (pure async, works on Windows + Linux).

Subscribes to SmartOS telemetry topics and routes messages to the feature engine.
"""

import asyncio

import structlog
from gmqtt import Client as MQTTClient
from gmqtt.mqtt.constants import MQTTv311

from netpulse_ml.config import Settings
from netpulse_ml.ingestion.parsers import parse_payload

log = structlog.get_logger()


class MQTTConsumer:
    """Consumes MQTT messages from SmartOS routers and routes to feature engine."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._running = True
        self._message_count = 0
        self._error_count = 0
        self._connected = asyncio.Event()
        self._client: MQTTClient | None = None

    def stop(self) -> None:
        self._running = False
        if self._client:
            asyncio.ensure_future(self._client.disconnect())

    @property
    def message_count(self) -> int:
        return self._message_count

    @property
    def is_running(self) -> bool:
        return self._running and self._connected.is_set()

    async def run(self) -> None:
        """Main consumer loop with automatic reconnection."""
        while self._running:
            try:
                await self._consume()
            except asyncio.CancelledError:
                log.info("MQTT consumer cancelled")
                break
            except Exception as e:
                log.warning("MQTT connection error, reconnecting in 5s", error=str(e))
                self._connected.clear()
                await asyncio.sleep(5)

    async def _consume(self) -> None:
        """Connect, subscribe, and process messages."""
        client = MQTTClient(client_id="netpulse-ml")
        self._client = client

        if self._settings.mqtt_username:
            client.set_auth_credentials(
                self._settings.mqtt_username,
                self._settings.mqtt_password or None,
            )

        # Set up callbacks
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect

        await client.connect(
            self._settings.mqtt_broker,
            port=self._settings.mqtt_port,
            version=MQTTv311,
        )

        self._connected.set()
        log.info("MQTT connected", broker=self._settings.mqtt_broker)

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    def _on_connect(self, client, flags, rc, properties) -> None:
        """Called when MQTT connection is established."""
        for topic in self._settings.mqtt_topic_list:
            client.subscribe(topic, qos=1)
            log.info("Subscribed to MQTT topic", topic=topic)

    def _on_disconnect(self, client, packet, exc=None) -> None:
        """Called when MQTT connection is lost."""
        self._connected.clear()
        if self._running:
            log.warning("MQTT disconnected")

    def _on_message(self, client, topic: str, payload: bytes, qos, properties) -> None:
        """Called for each received message. Schedules async processing."""
        self._message_count += 1

        try:
            device_id, msg_type, parsed = parse_payload(topic, payload)
            if parsed is not None:
                # Schedule async processing on the event loop
                asyncio.ensure_future(self._route_message(device_id, msg_type, parsed))
        except Exception as e:
            self._error_count += 1
            log.warning("Failed to process MQTT message", topic=topic, error=str(e))

        # gmqtt requires returning 0 to acknowledge
        return 0

    async def _route_message(self, device_id: str, msg_type: str, payload: object) -> None:
        """Route a parsed message to the feature computation pipeline."""
        from netpulse_ml.features.store import feature_store
        await feature_store.process_message(device_id, msg_type, payload)
