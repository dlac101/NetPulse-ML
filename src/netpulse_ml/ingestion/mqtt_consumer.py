"""Async MQTT consumer that subscribes to SmartOS telemetry topics."""

import asyncio

import structlog
from aiomqtt import Client, MqttError

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

    def stop(self) -> None:
        self._running = False

    @property
    def message_count(self) -> int:
        return self._message_count

    @property
    def is_running(self) -> bool:
        return self._running

    async def run(self) -> None:
        """Main consumer loop with automatic reconnection."""
        while self._running:
            try:
                await self._consume()
            except MqttError as e:
                log.warning("MQTT connection lost, reconnecting in 5s", error=str(e))
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                log.info("MQTT consumer cancelled")
                break
            except Exception as e:
                log.error("Unexpected MQTT error", error=str(e))
                await asyncio.sleep(5)

    async def _consume(self) -> None:
        """Connect and consume messages."""
        connect_kwargs: dict = {
            "hostname": self._settings.mqtt_broker,
            "port": self._settings.mqtt_port,
        }
        if self._settings.mqtt_username:
            connect_kwargs["username"] = self._settings.mqtt_username
            connect_kwargs["password"] = self._settings.mqtt_password

        async with Client(**connect_kwargs) as client:
            for topic in self._settings.mqtt_topic_list:
                await client.subscribe(topic, qos=1)
                log.info("Subscribed to MQTT topic", topic=topic)

            async for message in client.messages:
                if not self._running:
                    break

                topic_str = str(message.topic)
                payload_bytes = message.payload
                if isinstance(payload_bytes, str):
                    payload_bytes = payload_bytes.encode()

                self._message_count += 1

                try:
                    device_id, msg_type, parsed = parse_payload(topic_str, payload_bytes)
                    if parsed is not None:
                        await self._route_message(device_id, msg_type, parsed)
                except Exception as e:
                    self._error_count += 1
                    log.warning(
                        "Failed to process MQTT message",
                        topic=topic_str,
                        error=str(e),
                    )

    async def _route_message(self, device_id: str, msg_type: str, payload: object) -> None:
        """Route a parsed message to the feature computation pipeline.

        In Phase 1, this computes features and writes to the feature store.
        The feature computation is imported lazily to avoid circular imports.
        """
        from netpulse_ml.features.store import feature_store

        await feature_store.process_message(device_id, msg_type, payload)
