"""Parse raw MQTT JSON payloads into validated Pydantic models."""

import orjson
import structlog

from netpulse_ml.ingestion.validators import (
    BbstPayload,
    CategoryHoursPayload,
    ClassifiPayload,
    EventPayload,
    FlowStatsPayload,
    MetaPayload,
    QoEPayload,
    WifiPayload,
)

log = structlog.get_logger()

# Topic suffix -> parser mapping
_PARSERS: dict[str, type] = {
    "bbst": BbstPayload,
    "qoe": QoEPayload,
    "classifi": ClassifiPayload,
    "wifi": WifiPayload,
    "events": EventPayload,
    "meta": MetaPayload,
}


def parse_topic(topic: str) -> tuple[str, str, str]:
    """Extract device_id and message type from MQTT topic.

    Topic format: smartos/{device_id}/{type}[/subtype]
    Returns: (device_id, message_type, subtopic)
    """
    parts = topic.split("/")
    if len(parts) < 3 or parts[0] != "smartos":
        raise ValueError(f"Invalid topic format: {topic}")

    device_id = parts[1]
    msg_type = parts[2]
    subtopic = "/".join(parts[3:]) if len(parts) > 3 else ""
    return device_id, msg_type, subtopic


def parse_payload(
    topic: str, raw: bytes
) -> tuple[str, str, BbstPayload | QoEPayload | ClassifiPayload | WifiPayload | EventPayload | MetaPayload | FlowStatsPayload | CategoryHoursPayload | None]:
    """Parse an MQTT message into a typed payload.

    Returns: (device_id, message_type, parsed_payload_or_None)
    """
    device_id, msg_type, subtopic = parse_topic(topic)

    try:
        data = orjson.loads(raw)
    except orjson.JSONDecodeError:
        log.warning("Invalid JSON payload", topic=topic, size=len(raw))
        return device_id, msg_type, None

    try:
        if msg_type == "flowstatd":
            if subtopic == "stats":
                return device_id, "flowstatd_stats", FlowStatsPayload.model_validate(data)
            elif subtopic == "category_hours":
                return device_id, "flowstatd_hours", CategoryHoursPayload.model_validate(data)
            else:
                log.debug("Unknown flowstatd subtopic", subtopic=subtopic)
                return device_id, msg_type, None

        parser_cls = _PARSERS.get(msg_type)
        if parser_cls is None:
            log.debug("No parser for message type", msg_type=msg_type)
            return device_id, msg_type, None

        parsed = parser_cls.model_validate(data)
        return device_id, msg_type, parsed

    except Exception as e:
        log.warning(
            "Payload validation failed",
            topic=topic,
            msg_type=msg_type,
            error=str(e),
        )
        return device_id, msg_type, None
