"""End-to-end API test: hit every endpoint and verify responses.

Assumes: FastAPI backend is running on localhost:8000 with seeded data.

Usage:
    python scripts/e2e_test.py
"""

import httpx
import sys


BASE = "http://localhost:8000/v1"
PASS = 0
FAIL = 0


def check(name: str, response: httpx.Response, expected_status: int = 200) -> bool:
    global PASS, FAIL
    ok = response.status_code == expected_status
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {name}: {response.status_code} ({len(response.content)} bytes)")
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"         Body: {response.text[:200]}")
    return ok


def main() -> None:
    global PASS, FAIL
    client = httpx.Client(timeout=30.0)

    print("=" * 60)
    print("NetPulse ML Backend - End-to-End Test")
    print("=" * 60)

    # Health
    print("\n--- Health ---")
    r = client.get(f"{BASE}/health")
    check("GET /health", r)
    if r.status_code == 200:
        data = r.json()
        print(f"         Status: {data.get('status')}")
        print(f"         Models: {data.get('models')}")

    # Anomalies
    print("\n--- Anomaly Detection ---")
    r = client.get(f"{BASE}/anomalies?threshold=0.3&limit=5")
    check("GET /anomalies", r)
    if r.status_code == 200:
        data = r.json()
        print(f"         Found: {data['pagination']['total']} anomalies")

    r = client.get(f"{BASE}/devices/dev-0001/anomaly-score")
    check("GET /devices/{id}/anomaly-score", r)

    # Churn
    print("\n--- Churn Prediction ---")
    r = client.get(f"{BASE}/churn/predictions?limit=5")
    check("GET /churn/predictions", r)

    r = client.get(f"{BASE}/subscribers/dev-0001/churn")
    check("GET /subscribers/{id}/churn", r)

    # QoE Forecast
    print("\n--- QoE Forecast ---")
    r = client.get(f"{BASE}/devices/dev-0001/qoe-forecast?horizon=24h")
    check("GET /devices/{id}/qoe-forecast", r)

    # Fleet Clusters
    print("\n--- Fleet Clusters ---")
    r = client.get(f"{BASE}/fleet/clusters")
    check("GET /fleet/clusters", r)

    # Recommendations
    print("\n--- Recommendations ---")
    r = client.get(f"{BASE}/devices/dev-0001/recommendations")
    check("GET /devices/{id}/recommendations", r)

    # Models
    print("\n--- Model Registry ---")
    r = client.get(f"{BASE}/models")
    check("GET /models", r)

    # Retrain (this is the big one)
    print("\n--- Model Training ---")
    r = client.post(f"{BASE}/models/anomaly_detector/retrain")
    check("POST /models/anomaly_detector/retrain", r)
    if r.status_code == 200:
        data = r.json()
        print(f"         Metrics: {data.get('metrics')}")

    # Re-check anomaly after training
    print("\n--- Post-Training Anomaly Check ---")
    r = client.get(f"{BASE}/anomalies?threshold=0.3&limit=5")
    check("GET /anomalies (post-train)", r)
    if r.status_code == 200:
        data = r.json()
        print(f"         Found: {data['pagination']['total']} anomalies")
        for item in data["data"][:3]:
            print(f"         {item['deviceId']}: score={item['anomalyScore']:.3f}")

    # Agents
    print("\n--- Agent Orchestrator ---")
    r = client.get(f"{BASE}/agents/status")
    check("GET /agents/status", r)

    r = client.get(f"{BASE}/agents/history")
    check("GET /agents/history", r)

    r = client.get(f"{BASE}/agents/config")
    check("GET /agents/config", r)

    # LLM Insights (requires Ollama)
    print("\n--- LLM Insights ---")
    r = client.get(f"{BASE}/insights/fleet-summary")
    check("GET /insights/fleet-summary", r)
    if r.status_code == 200:
        data = r.json()
        print(f"         Model: {data.get('model')}")
        print(f"         Content: {data.get('content', '')[:150]}...")

    r = client.get(f"{BASE}/insights/device/dev-0001")
    check("GET /insights/device/{id}", r)

    # Chat
    print("\n--- Chat ---")
    r = client.post(f"{BASE}/chat", json={"message": "Why is device dev-0001 underperforming?"})
    check("POST /chat", r)
    if r.status_code == 200:
        data = r.json()
        print(f"         Response: {data.get('response', '')[:150]}...")

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
