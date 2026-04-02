"""Threshold-based alerting rules engine.

Evaluates device features against configurable thresholds and dispatches
notifications when conditions are met. Runs alongside the agent orchestrator.
"""

import structlog

from netpulse_ml.notifications.dispatcher import notify_escalation

log = structlog.get_logger()

# Default alert rules matching the frontend's DEFAULT_ALERT_THRESHOLDS
ALERT_RULES = [
    {
        "id": "qoe_critical",
        "label": "QoE Score Critical",
        "feature": "qoe_composite_latest",
        "operator": "lt",
        "threshold": 50.0,
        "severity": "critical",
        "impact": "high",
    },
    {
        "id": "qoe_warning",
        "label": "QoE Score Warning",
        "feature": "qoe_composite_latest",
        "operator": "lt",
        "threshold": 70.0,
        "severity": "warning",
        "impact": "medium",
    },
    {
        "id": "bloat_critical",
        "label": "Download Bufferbloat Critical",
        "feature": "dl_bloat_pct",
        "operator": "gt",
        "threshold": 500.0,
        "severity": "critical",
        "impact": "high",
    },
    {
        "id": "bloat_warning",
        "label": "Download Bufferbloat Warning",
        "feature": "dl_bloat_pct",
        "operator": "gt",
        "threshold": 200.0,
        "severity": "warning",
        "impact": "medium",
    },
    {
        "id": "latency_critical",
        "label": "Idle Latency Critical",
        "feature": "latency_idle_ms",
        "operator": "gt",
        "threshold": 50.0,
        "severity": "critical",
        "impact": "high",
    },
    {
        "id": "wifi_retransmit",
        "label": "WiFi Retransmission High",
        "feature": "wifi_avg_retransmit_rate",
        "operator": "gt",
        "threshold": 15.0,
        "severity": "warning",
        "impact": "medium",
    },
    {
        "id": "traffic_risk",
        "label": "Security Risk Detected",
        "feature": "traffic_max_risk_score",
        "operator": "gt",
        "threshold": 100.0,
        "severity": "critical",
        "impact": "high",
    },
]


def evaluate_rule(rule: dict, features: dict[str, float]) -> bool:
    """Check if a single rule fires for the given features."""
    feature_name = rule["feature"]
    value = features.get(feature_name)
    if value is None:
        return False

    threshold = rule["threshold"]
    op = rule["operator"]

    if op == "gt":
        return value > threshold
    elif op == "lt":
        return value < threshold
    elif op == "gte":
        return value >= threshold
    elif op == "lte":
        return value <= threshold
    elif op == "eq":
        return value == threshold
    return False


async def evaluate_device_alerts(
    device_id: str,
    features: dict[str, float],
    rules: list[dict] | None = None,
    notify: bool = True,
) -> list[dict]:
    """Evaluate all alert rules for a device's features.

    Returns list of fired alerts. Optionally dispatches notifications.
    """
    if rules is None:
        rules = ALERT_RULES

    fired = []
    for rule in rules:
        if evaluate_rule(rule, features):
            alert = {
                "device_id": device_id,
                "rule_id": rule["id"],
                "label": rule["label"],
                "severity": rule["severity"],
                "feature": rule["feature"],
                "value": features.get(rule["feature"], 0),
                "threshold": rule["threshold"],
                "operator": rule["operator"],
            }
            fired.append(alert)

            if notify:
                await notify_escalation(
                    device_id=device_id,
                    title=f"Alert: {rule['label']}",
                    description=f"{rule['feature']}={features.get(rule['feature'], 0):.1f} {rule['operator']} {rule['threshold']}",
                    diagnosis=rule["id"],
                    confidence=1.0,
                    impact=rule["impact"],
                    recommendation_id="",
                )

    if fired:
        log.info("Alerts fired", device_id=device_id, count=len(fired),
                 rules=[a["rule_id"] for a in fired])

    return fired
