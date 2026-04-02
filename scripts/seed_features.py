"""Seed the feature store with synthetic device data for testing.

Generates realistic feature snapshots for ~100 devices (matching the frontend
mock data) and inserts them into TimescaleDB. Run after docker compose up.

Usage:
    PYTHONPATH=src python scripts/seed_features.py
"""

import asyncio
import random
from datetime import datetime, timedelta, timezone

from netpulse_ml.db.engine import init_db, async_session_factory
from netpulse_ml.db.models import FeatureSnapshot

# Seed for reproducibility
random.seed(42)

# Adtran SmartOS models (matching frontend faker/devices.ts)
MODELS = [
    "SDG-8733", "SDG-8733v", "SDG-8734", "SDG-8734v",
    "SDG-8732", "SDG-8732v", "SDG-8712", "SDG-8712v",
    "SDG-8622", "SDG-8632", "SDG-8614", "SDG-8612",
]
FIRMWARES = ["23.11.1.2", "23.11.1.1", "23.08.3.0", "23.05.2.1", "22.11.4.0"]
STATUSES = ["online"] * 19 + ["degraded"]  # 95% online


def generate_device_features(device_id: str, status: str) -> dict[str, float]:
    """Generate a realistic feature snapshot for one device."""
    is_degraded = status == "degraded"

    # Base QoE: degraded devices get lower scores
    base_qoe = random.uniform(30, 60) if is_degraded else random.uniform(65, 98)

    # Speed: based on a provisioned plan of 500-1000 Mbps
    provisioned = random.choice([500, 750, 1000])
    dl_ratio = random.uniform(0.4, 0.7) if is_degraded else random.uniform(0.75, 0.98)
    ul_ratio = random.uniform(0.3, 0.6) if is_degraded else random.uniform(0.7, 0.95)

    dl_mbps = provisioned * dl_ratio
    ul_mbps = (provisioned * 0.5) * ul_ratio  # Upload typically half of download plan

    # Latency
    idle_latency = random.uniform(15, 45) if is_degraded else random.uniform(1.5, 8)
    dl_latency = idle_latency * random.uniform(1.5, 4.0) if is_degraded else idle_latency * random.uniform(1.1, 2.0)

    # Bloat
    bloat_pct = ((dl_latency - idle_latency) / max(idle_latency, 0.1)) * 100

    # WiFi
    n_clients = random.randint(1, 24)
    pct_2_4 = random.uniform(0.5, 0.8) if is_degraded else random.uniform(0.1, 0.4)

    # Traffic
    traffic_bytes = random.uniform(1e8, 5e10)  # 100MB to 50GB

    # Events
    critical_events = random.randint(2, 8) if is_degraded else random.randint(0, 1)
    warning_events = random.randint(3, 12) if is_degraded else random.randint(0, 3)

    return {
        "dl_mbps_latest": round(dl_mbps, 1),
        "ul_mbps_latest": round(ul_mbps, 1),
        "dl_capacity_pct": round(dl_ratio * 100, 1),
        "ul_capacity_pct": round(ul_ratio * 100, 1),
        "dl_bloat_pct": round(max(bloat_pct, 0), 1),
        "ul_bloat_pct": round(random.uniform(10, 100), 1),
        "latency_idle_ms": round(idle_latency, 2),
        "latency_dl_ms": round(dl_latency, 2),
        "latency_ul_ms": round(idle_latency * random.uniform(1.0, 1.5), 2),
        "jitter_idle_ms": round(random.uniform(0.2, 5.0), 2),
        "jitter_dl_ms": round(random.uniform(0.5, 10.0), 2),
        "bloat_grade_dl": random.choice([0, 1, 2, 3]) if is_degraded else random.choice([0, 0, 1]),
        "bloat_grade_ul": random.choice([0, 1, 2, 3]),
        "provisioned_dl_mbps": float(provisioned),
        "provisioned_ul_mbps": float(provisioned // 2),
        "test_duration_sec": round(random.uniform(10, 30), 1),
        "qoe_composite_latest": round(base_qoe, 1),
        "qoe_wan_latest": round(base_qoe + random.uniform(-10, 10), 1),
        "qoe_wifi_latest": round(base_qoe + random.uniform(-15, 5), 1),
        "qoe_mesh_latest": round(base_qoe + random.uniform(-5, 10), 1),
        "qoe_system_latest": round(base_qoe + random.uniform(-5, 5), 1),
        "qoe_lan_latest": round(base_qoe + random.uniform(-3, 8), 1),
        "qoe_worst_category": random.randint(0, 4),
        "qoe_impact_count": random.randint(3, 12) if is_degraded else random.randint(0, 3),
        "qoe_total_penalty": round(random.uniform(10, 40) if is_degraded else random.uniform(0, 10), 1),
        "qoe_satellite_count": random.randint(0, 3),
        "wifi_client_count": float(n_clients),
        "wifi_avg_rssi": round(random.uniform(-75, -55) if is_degraded else random.uniform(-55, -30), 1),
        "wifi_min_rssi": round(random.uniform(-85, -65) if is_degraded else random.uniform(-65, -40), 1),
        "wifi_avg_retransmit_rate": round(random.uniform(8, 25) if is_degraded else random.uniform(0.5, 6), 1),
        "wifi_max_retransmit_rate": round(random.uniform(15, 40) if is_degraded else random.uniform(2, 12), 1),
        "wifi_pct_2_4ghz": round(pct_2_4, 2),
        "wifi_pct_5ghz": round(1 - pct_2_4 - random.uniform(0, 0.1), 2),
        "wifi_pct_6ghz": round(random.uniform(0, 0.1), 2),
        "wifi_airtime_util_max": round(random.uniform(40, 85) if is_degraded else random.uniform(5, 35), 1),
        "wifi_interference_max": round(random.uniform(15, 40) if is_degraded else random.uniform(1, 10), 1),
        "mesh_satellite_count": float(random.randint(0, 3)),
        "mesh_satellite_offline_count": float(random.randint(0, 1) if is_degraded else 0),
        "mesh_backhaul_min_dl_mbps": round(random.uniform(100, 400) if random.random() > 0.3 else 0, 1),
        "mesh_max_hops": float(random.randint(1, 3)),
        "traffic_total_bytes_1h": round(traffic_bytes, 0),
        "traffic_flow_count_1h": float(random.randint(50, 500)),
        "traffic_max_risk_score": float(random.randint(0, 30) if not is_degraded else random.randint(20, 120)),
        "traffic_high_risk_flow_count": float(random.randint(0, 2) if not is_degraded else random.randint(1, 8)),
        "traffic_streaming_pct": round(random.uniform(0.2, 0.6), 2),
        "traffic_gaming_pct": round(random.uniform(0.05, 0.25), 2),
        "traffic_voip_pct": round(random.uniform(0.02, 0.15), 2),
        "events_critical_24h": float(critical_events),
        "events_warning_24h": float(warning_events),
        "events_total_7d": float((critical_events + warning_events) * 7),
        "events_connection_lost_7d": float(random.randint(2, 10) if is_degraded else 0),
        "events_bloat_threshold_7d": float(random.randint(1, 5) if is_degraded else 0),
        "events_security_risk_7d": float(random.randint(0, 3)),
        "model_encoded": float(MODELS.index(random.choice(MODELS))),
        "firmware_encoded": float(FIRMWARES.index(random.choice(FIRMWARES))),
        "firmware_age_days": float(random.randint(10, 365)),
        "device_age_days": float(random.randint(30, 1000)),
        "subscriber_device_count": float(random.randint(1, 3)),
        "subscriber_avg_qoe_7d": round(base_qoe + random.uniform(-5, 5), 1),
        "subscriber_min_qoe_7d": round(base_qoe - random.uniform(5, 20), 1),
        "subscriber_total_events_30d": float((critical_events + warning_events) * 30),
        "subscriber_speed_test_frequency": round(random.uniform(1, 12), 1),
        "subscriber_tenure_days": float(random.randint(30, 1500)),
        "subscriber_dl_utilization_ratio": round(dl_ratio, 2),
    }


async def seed(n_devices: int = 100, days_back: int = 7) -> None:
    """Generate and insert feature snapshots."""
    await init_db()

    now = datetime.now(timezone.utc)
    total_snapshots = 0

    async with async_session_factory() as session:
        for i in range(n_devices):
            device_id = f"dev-{i + 1:04d}"
            status = random.choice(STATUSES)

            # Generate multiple snapshots over the past N days (one per hour)
            for hour in range(0, days_back * 24, 4):  # Every 4 hours
                ts = now - timedelta(hours=hour)
                features = generate_device_features(device_id, status)

                # Add some time-series variation
                noise = random.uniform(-5, 5)
                features["qoe_composite_latest"] = max(0, min(100,
                    features["qoe_composite_latest"] + noise))
                features["dl_mbps_latest"] = max(0,
                    features["dl_mbps_latest"] + random.uniform(-30, 30))

                snapshot = FeatureSnapshot(
                    device_id=device_id,
                    timestamp=ts,
                    features=features,
                )
                session.add(snapshot)
                total_snapshots += 1

            # Flush every 10 devices to avoid memory buildup
            if (i + 1) % 10 == 0:
                await session.flush()
                print(f"  Seeded {i + 1}/{n_devices} devices ({total_snapshots} snapshots)")

        await session.commit()

    print(f"\nDone: {n_devices} devices, {total_snapshots} feature snapshots inserted.")


if __name__ == "__main__":
    asyncio.run(seed())
