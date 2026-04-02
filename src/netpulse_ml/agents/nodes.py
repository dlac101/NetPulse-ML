"""LangGraph node implementations for the remediation agent.

Each node is a pure function: state in -> partial state update out.
Nodes access ML models and feature store via module-level imports.
"""

import uuid

import structlog

from netpulse_ml.agents.state import RemediationState
from netpulse_ml.agents.tools import TOOL_REGISTRY
from netpulse_ml.db.engine import async_session_factory
from netpulse_ml.db.models import Recommendation
from netpulse_ml.features.store import feature_store
from netpulse_ml.serving.predictor import Predictor

log = structlog.get_logger()

# Diagnosis rules: map top anomaly feature names to diagnosis categories
_DIAGNOSIS_RULES: list[tuple[str, str, float]] = [
    # (feature_name, diagnosis, min_zscore)
    ("dl_bloat_pct", "bufferbloat", 2.0),
    ("wifi_avg_rssi", "weak_wifi_signal", 2.0),
    ("wifi_avg_retransmit_rate", "wifi_interference", 2.0),
    ("wifi_airtime_util_max", "wifi_interference", 2.0),
    ("latency_idle_ms", "high_latency", 2.0),
    ("traffic_max_risk_score", "security_risk", 2.0),
]

# Map diagnosis to recommendation
_PLAN_RULES: dict[str, dict] = {
    "bufferbloat": {
        "type": "enable_sqm",
        "title": "Enable SQM to reduce bufferbloat",
        "description": "Device shows elevated bufferbloat. Enabling Smart Queue Management will reduce latency under load.",
        "auto_executable": True,
        "impact": "high",
    },
    "weak_wifi_signal": {
        "type": "mesh_ap_add",
        "title": "Add mesh satellite for better coverage",
        "description": "Weak WiFi signal detected. Adding a mesh satellite near affected clients will improve coverage.",
        "auto_executable": False,
        "impact": "high",
    },
    "wifi_interference": {
        "type": "band_steering",
        "title": "Enable band steering to reduce interference",
        "description": "High WiFi interference or retransmission rate. Band steering will move clients to less congested bands.",
        "auto_executable": True,
        "impact": "medium",
    },
    "band_congestion": {
        "type": "band_steering",
        "title": "Enable band steering for 5/6GHz migration",
        "description": "Over 60% of clients on 2.4GHz. Band steering will move capable clients to faster 5/6GHz bands.",
        "auto_executable": True,
        "impact": "medium",
    },
    "high_latency": {
        "type": "firmware_upgrade",
        "title": "Upgrade firmware to improve latency",
        "description": "Elevated idle latency detected. A firmware upgrade may resolve known latency issues.",
        "auto_executable": False,
        "impact": "medium",
    },
    "security_risk": {
        "type": "firmware_upgrade",
        "title": "Upgrade firmware to address security risks",
        "description": "High nDPI risk scores detected in traffic flows. Firmware upgrade includes security patches.",
        "auto_executable": False,
        "impact": "high",
    },
    "degraded_general": {
        "type": "service_tier_change",
        "title": "Review service tier alignment",
        "description": "General performance degradation detected across multiple metrics. Service tier may need adjustment.",
        "auto_executable": False,
        "impact": "low",
    },
}


async def analyze_node(state: RemediationState, predictor: Predictor) -> dict:
    """ANALYZE: Gather device features and run ML predictions."""
    device_id = state["device_id"]

    features = await feature_store.get_latest_features(device_id)
    if not features:
        return {"status": "failed", "error": "No features available for device"}

    anomaly_score = 0.0
    top_features: list[dict] = []
    churn_risk = 0.0

    if predictor.anomaly_detector.is_fitted:
        anomaly_score = predictor.anomaly_detector.predict_single(features)
        top_features = predictor.anomaly_detector.get_top_features(features, n_top=5)

    if predictor.churn_predictor.is_fitted:
        churn_result = predictor.churn_predictor.predict_single(features)
        churn_risk = churn_result.get("riskScore", 0.0)

    log.info(
        "Agent analyzed device",
        device_id=device_id,
        anomaly_score=round(anomaly_score, 3),
        churn_risk=round(churn_risk, 1),
    )

    return {
        "features": features,
        "anomaly_score": anomaly_score,
        "churn_risk": churn_risk,
        "top_anomaly_features": top_features,
        "status": "analyzed",
    }


async def diagnose_node(state: RemediationState) -> dict:
    """DIAGNOSE: Determine root cause from top anomaly features."""
    top_features = state.get("top_anomaly_features", [])
    features = state.get("features", {})

    # Check band congestion first (special rule based on raw feature, not z-score)
    if features.get("wifi_pct_2_4ghz", 0) > 0.6:
        return {"diagnosis": "band_congestion", "status": "diagnosed"}

    # Check z-score based rules
    for feat in top_features:
        feat_name = feat.get("name", "")
        zscore = feat.get("zscore", 0.0)
        for rule_feat, diagnosis, min_z in _DIAGNOSIS_RULES:
            if feat_name == rule_feat and zscore >= min_z:
                log.info("Agent diagnosed issue", diagnosis=diagnosis, feature=feat_name, zscore=zscore)
                return {"diagnosis": diagnosis, "status": "diagnosed"}

    # Multiple moderate factors
    high_z_count = sum(1 for f in top_features if f.get("zscore", 0) >= 1.5)
    if high_z_count >= 3:
        return {"diagnosis": "degraded_general", "status": "diagnosed"}

    return {"diagnosis": "none", "status": "diagnosed"}


async def plan_node(state: RemediationState) -> dict:
    """PLAN: Map diagnosis to a concrete recommendation."""
    diagnosis = state.get("diagnosis", "none")
    anomaly_score = state.get("anomaly_score", 0.0)

    plan = _PLAN_RULES.get(diagnosis)
    if plan is None:
        return {"recommended_action": None, "status": "planned"}

    return {
        "recommended_action": plan["type"],
        "action_confidence": anomaly_score,
        "action_title": plan["title"],
        "action_description": plan["description"],
        "auto_executable": plan["auto_executable"],
        "status": "planned",
    }


async def execute_node(state: RemediationState) -> dict:
    """EXECUTE: Run the SmartOS tool for auto-executable actions."""
    action = state.get("recommended_action")
    device_id = state["device_id"]

    tool_fn = TOOL_REGISTRY.get(action or "")
    if tool_fn is None or not callable(tool_fn):
        return {"execution_result": "no_tool_available", "status": "failed"}

    try:
        result = await tool_fn(device_id)
        log.info("Agent executed action", device_id=device_id, action=action, result=result)
        return {"execution_result": str(result), "status": "executed"}
    except Exception as e:
        log.error("Agent execution failed", device_id=device_id, action=action, error=str(e))
        return {"execution_result": f"error: {e}", "status": "failed"}


async def escalate_node(state: RemediationState) -> dict:
    """ESCALATE: Create a pending Recommendation for human review."""
    device_id = state["device_id"]
    rec_id = str(uuid.uuid4())

    async with async_session_factory() as session:
        rec = Recommendation(
            id=rec_id,
            device_id=device_id,
            type=state.get("recommended_action", "service_tier_change"),
            title=state.get("action_title", "Review device"),
            description=state.get("action_description", "Agent-generated recommendation"),
            confidence=state.get("action_confidence", 0.0),
            impact=_PLAN_RULES.get(state.get("diagnosis", ""), {}).get("impact", "low"),
            auto_executable=False,
            status="pending",
            model_name="agent_remediation",
            model_version="2.0.0",
        )
        session.add(rec)
        await session.commit()

    log.info("Agent escalated to human", device_id=device_id, recommendation_id=rec_id)

    return {"recommendation_id": rec_id, "status": "escalated"}


async def verify_node(state: RemediationState, predictor: Predictor) -> dict:
    """VERIFY: Re-check device metrics after execution to confirm improvement.

    Note: In production this would wait agent_verify_delay_minutes before re-checking.
    In Phase 2, we do an immediate re-check (metrics won't change with mocked tools,
    but the structure is in place for real verification).
    """
    device_id = state["device_id"]

    features = await feature_store.get_latest_features(device_id)
    if not features:
        return {"verified": False, "status": "verified"}

    # Re-score anomaly
    new_score = 0.0
    if predictor.anomaly_detector.is_fitted:
        new_score = predictor.anomaly_detector.predict_single(features)

    old_score = state.get("anomaly_score", 0.0)
    improved = new_score < old_score

    log.info(
        "Agent verified execution",
        device_id=device_id,
        old_score=round(old_score, 3),
        new_score=round(new_score, 3),
        improved=improved,
    )

    return {"verified": improved, "status": "verified"}
