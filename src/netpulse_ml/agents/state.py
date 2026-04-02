"""LangGraph state definition for the remediation agent."""

from typing import TypedDict


class RemediationState(TypedDict, total=False):
    """State passed through the LangGraph remediation workflow.

    Each node reads and updates specific fields. LangGraph manages
    state persistence between nodes.
    """

    # Input (set by orchestrator before graph runs)
    device_id: str

    # Populated by ANALYZE node
    features: dict[str, float]
    anomaly_score: float
    churn_risk: float
    top_anomaly_features: list[dict]

    # Populated by DIAGNOSE node
    diagnosis: str  # bufferbloat | weak_wifi_signal | wifi_interference | band_congestion | high_latency | security_risk | degraded_general | none

    # Populated by PLAN node
    recommended_action: str | None  # firmware_upgrade | enable_sqm | band_steering | mesh_ap_add | service_tier_change
    action_confidence: float
    action_title: str
    action_description: str
    auto_executable: bool
    recommendation_id: str | None

    # Populated by EXECUTE node
    execution_result: str | None

    # Populated by VERIFY node
    verified: bool

    # Status tracking
    status: str  # analyzing | diagnosed | planned | executed | escalated | verified | failed
    error: str | None
