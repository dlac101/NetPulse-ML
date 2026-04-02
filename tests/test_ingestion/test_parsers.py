"""Tests for MQTT payload parsing."""

import orjson
import pytest

from netpulse_ml.ingestion.parsers import parse_payload, parse_topic
from netpulse_ml.ingestion.validators import QoEPayload, BbstPayload


class TestParseTopic:
    def test_valid_topic(self) -> None:
        device_id, msg_type, sub = parse_topic("smartos/dev-123/qoe")
        assert device_id == "dev-123"
        assert msg_type == "qoe"
        assert sub == ""

    def test_flowstatd_subtopic(self) -> None:
        device_id, msg_type, sub = parse_topic("smartos/dev-456/flowstatd/stats")
        assert device_id == "dev-456"
        assert msg_type == "flowstatd"
        assert sub == "stats"

    def test_invalid_topic(self) -> None:
        with pytest.raises(ValueError, match="Invalid topic"):
            parse_topic("invalid/topic")


class TestParsePayload:
    def test_qoe_payload(self) -> None:
        raw = orjson.dumps({
            "id": "qoe-1",
            "deviceId": "dev-1",
            "timestamp": "2026-04-01T12:00:00Z",
            "compositeScore": 78,
            "compositeGrade": "B",
            "categories": {
                "wan": {"name": "wan", "score": 80, "grade": "B", "weight": 0.25, "impactFactors": []},
                "wifi": {"name": "wifi", "score": 70, "grade": "C", "weight": 0.25, "impactFactors": []},
                "mesh": {"name": "mesh", "score": 85, "grade": "B", "weight": 0.25, "impactFactors": []},
                "system": {"name": "system", "score": 75, "grade": "B", "weight": 0.15, "impactFactors": []},
                "lan": {"name": "lan", "score": 90, "grade": "A", "weight": 0.10, "impactFactors": []},
            },
        })

        device_id, msg_type, parsed = parse_payload("smartos/dev-1/qoe", raw)

        assert device_id == "dev-1"
        assert msg_type == "qoe"
        assert isinstance(parsed, QoEPayload)
        assert parsed.compositeScore == 78

    def test_invalid_json(self) -> None:
        device_id, msg_type, parsed = parse_payload("smartos/dev-1/qoe", b"not json")
        assert parsed is None

    def test_unknown_type(self) -> None:
        raw = orjson.dumps({"key": "value"})
        device_id, msg_type, parsed = parse_payload("smartos/dev-1/unknown", raw)
        assert parsed is None
