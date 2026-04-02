"""Per-device feature extraction from parsed telemetry messages.

Each function takes a parsed payload and returns a dict of feature name -> value.
Features are accumulated in the feature store across message types.
"""

from netpulse_ml.ingestion.validators import (
    BbstPayload,
    ClassifiPayload,
    EventPayload,
    FlowStatsPayload,
    QoEPayload,
    WifiPayload,
)

# Grade encoding for ordinal features
_GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
_QOE_CAT_MAP = {"wan": 0, "wifi": 1, "mesh": 2, "system": 3, "lan": 4}


def extract_bbst_features(payload: BbstPayload) -> dict[str, float]:
    """Extract speed test features from a BBST result."""
    dl = payload.download
    ul = payload.upload
    lat = payload.latency

    # Bloat percentage: how much latency increases under load
    idle = max(lat.idleMs, 0.1)
    dl_bloat = ((lat.downloadMs - idle) / idle) * 100.0
    ul_bloat = ((lat.uploadMs - idle) / idle) * 100.0

    return {
        "dl_mbps_latest": dl.mbps,
        "ul_mbps_latest": ul.mbps,
        "dl_capacity_pct": dl.capacityPercent,
        "ul_capacity_pct": ul.capacityPercent,
        "dl_bloat_pct": max(dl_bloat, 0.0),
        "ul_bloat_pct": max(ul_bloat, 0.0),
        "latency_idle_ms": lat.idleMs,
        "latency_dl_ms": lat.downloadMs,
        "latency_ul_ms": lat.uploadMs,
        "jitter_idle_ms": lat.idleJitterMs,
        "jitter_dl_ms": lat.downloadJitterMs,
        "bloat_grade_dl": _GRADE_MAP.get(dl.bloatGrade, 2),
        "bloat_grade_ul": _GRADE_MAP.get(ul.bloatGrade, 2),
        "provisioned_dl_mbps": dl.provisionedMbps,
        "provisioned_ul_mbps": ul.provisionedMbps,
        "test_duration_sec": payload.durationSec,
    }


def extract_qoe_features(payload: QoEPayload) -> dict[str, float]:
    """Extract QoE features from a QoE score snapshot."""
    cats = payload.categories

    # Find worst category
    cat_scores = {
        "wan": cats.wan.score,
        "wifi": cats.wifi.score,
        "mesh": cats.mesh.score,
        "system": cats.system.score,
        "lan": cats.lan.score,
    }
    worst_cat = min(cat_scores, key=cat_scores.get)  # type: ignore[arg-type]

    # Count impact factors
    total_impacts = sum(
        len(getattr(cats, name).impactFactors)
        for name in ("wan", "wifi", "mesh", "system", "lan")
    )
    total_penalty = sum(
        sum(f.penalty for f in getattr(cats, name).impactFactors)
        for name in ("wan", "wifi", "mesh", "system", "lan")
    )

    return {
        "qoe_composite_latest": payload.compositeScore,
        "qoe_wan_latest": cats.wan.score,
        "qoe_wifi_latest": cats.wifi.score,
        "qoe_mesh_latest": cats.mesh.score,
        "qoe_system_latest": cats.system.score,
        "qoe_lan_latest": cats.lan.score,
        "qoe_worst_category": _QOE_CAT_MAP[worst_cat],
        "qoe_impact_count": total_impacts,
        "qoe_total_penalty": total_penalty,
        "qoe_satellite_count": payload.satelliteCount,
    }


def extract_wifi_features(payload: WifiPayload) -> dict[str, float]:
    """Extract WiFi metrics features."""
    clients = payload.clients
    satellites = payload.satellites
    airtime = payload.airtime

    n_clients = len(clients)

    # Client aggregates
    if n_clients > 0:
        avg_rssi = sum(c.rssi for c in clients) / n_clients
        min_rssi = min(c.rssi for c in clients)
        avg_retransmit = sum(c.retransmissionRate for c in clients) / n_clients
        max_retransmit = max(c.retransmissionRate for c in clients)

        band_counts = {"2.4GHz": 0, "5GHz": 0, "6GHz": 0}
        for c in clients:
            band_counts[c.band] = band_counts.get(c.band, 0) + 1
        pct_2_4 = band_counts["2.4GHz"] / n_clients
        pct_5 = band_counts["5GHz"] / n_clients
        pct_6 = band_counts["6GHz"] / n_clients
    else:
        avg_rssi = min_rssi = avg_retransmit = max_retransmit = 0.0
        pct_2_4 = pct_5 = pct_6 = 0.0

    # Airtime
    max_airtime = max((a.totalUtilizationPercent for a in airtime), default=0.0)
    max_interference = max(
        (a.wifiInterferencePercent + a.nonWifiInterferencePercent for a in airtime),
        default=0.0,
    )

    # Mesh
    n_satellites = len(satellites)
    offline_satellites = sum(1 for s in satellites if s.status == "offline")
    min_backhaul_dl = min((s.backhaulDlMbps for s in satellites), default=0.0)
    max_hops = max((s.hops for s in satellites), default=0)

    return {
        "wifi_client_count": float(n_clients),
        "wifi_avg_rssi": avg_rssi,
        "wifi_min_rssi": min_rssi,
        "wifi_avg_retransmit_rate": avg_retransmit,
        "wifi_max_retransmit_rate": max_retransmit,
        "wifi_pct_2_4ghz": pct_2_4,
        "wifi_pct_5ghz": pct_5,
        "wifi_pct_6ghz": pct_6,
        "wifi_airtime_util_max": max_airtime,
        "wifi_interference_max": max_interference,
        "mesh_satellite_count": float(n_satellites),
        "mesh_satellite_offline_count": float(offline_satellites),
        "mesh_backhaul_min_dl_mbps": min_backhaul_dl,
        "mesh_max_hops": float(max_hops),
    }


def extract_traffic_features(payload: FlowStatsPayload) -> dict[str, float]:
    """Extract DPI traffic features from flowstatd stats."""
    categories = payload.categories

    total_bytes = sum(c.totalBytes for c in categories)
    total_flows = sum(c.flowCount for c in categories)
    max_risk = max((c.maxRiskScore for c in categories), default=0)

    # Category percentages
    streaming_bytes = sum(c.totalBytes for c in categories if c.category == "Streaming")
    gaming_bytes = sum(c.totalBytes for c in categories if c.category == "Gaming")
    voip_bytes = sum(c.totalBytes for c in categories if c.category == "VoIP")

    denom = max(total_bytes, 1)

    return {
        "traffic_total_bytes_1h": float(total_bytes),
        "traffic_flow_count_1h": float(total_flows),
        "traffic_max_risk_score": float(max_risk),
        "traffic_streaming_pct": streaming_bytes / denom,
        "traffic_gaming_pct": gaming_bytes / denom,
        "traffic_voip_pct": voip_bytes / denom,
    }


def extract_classifi_features(payload: ClassifiPayload) -> dict[str, float]:
    """Extract DPI classification features from classifi flows."""
    flows = payload.flows
    high_risk = sum(1 for f in flows if f.riskScore > 50)
    max_risk = max((f.riskScore for f in flows), default=0)

    return {
        "traffic_high_risk_flow_count": float(high_risk),
        "traffic_max_risk_score": float(max_risk),
    }


def extract_event_features(payload: EventPayload) -> dict[str, float]:
    """Extract features from a single event.

    Note: Event features are accumulated over time windows in the feature store.
    This returns increment values for the relevant counters.
    """
    features: dict[str, float] = {}

    # Severity counter increments
    if payload.severity == "critical":
        features["_event_critical_inc"] = 1.0
    elif payload.severity == "warning":
        features["_event_warning_inc"] = 1.0

    # Type-specific counter increments
    type_feature_map = {
        "connection_lost": "_event_connection_lost_inc",
        "bloat_exceeds_threshold": "_event_bloat_threshold_inc",
        "security_risk_detected": "_event_security_risk_inc",
        "high_packet_loss": "_event_packet_loss_inc",
        "qoe_score_drop": "_event_qoe_drop_inc",
    }
    feat_key = type_feature_map.get(payload.type)
    if feat_key:
        features[feat_key] = 1.0

    return features
