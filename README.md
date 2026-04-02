# NetPulse ML Backend

AI/ML pipeline for the NetPulse Admin Console. Provides anomaly detection, churn prediction, QoE forecasting, fleet segmentation, autonomous remediation agents, notification dispatch, and RAG-powered natural language insights for SmartOS router fleets.

## Architecture

```
SmartOS Routers --> MQTT Broker --> netpulse-ml (FastAPI)
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
              Feature Store       ML Models          LLM / RAG
             (TimescaleDB)     (scikit-learn)     (Ollama + pgvector)
                    |                   |                   |
                    +--- Redis Cache ---+--- Notifications -+
                    |                   |                   |
                    +-------------------+-------------------+
                                        |
                                   REST API (/v1/*)
                                        |
                                   netpulse-admin (Next.js)
```

## Components

### Phase 1: ML Foundation
- **MQTT Ingestion**: 8 telemetry topics (speed tests, QoE, DPI traffic, WiFi, events)
- **Feature Engineering**: 60+ features across speed/QoE/WiFi/traffic/events
- **Anomaly Detection**: IsolationForest (21 features, 0-1 score)
- **Churn Prediction**: HistGradientBoosting + SHAP explanations (40+ features)
- **QoE Forecasting**: SARIMAX with 95% confidence intervals
- **Fleet Segmentation**: DBSCAN outliers + KMeans clusters
- **Redis Caching**: TTL-based cache on fleet predictions (5m anomalies, 1h churn), auto-invalidated on retrain
- **17 REST API endpoints** with thread-offloaded inference

### Phase 2: Autonomous Agents
- **LangGraph StateGraph**: ANALYZE > DIAGNOSE > PLAN > EXECUTE/ESCALATE > VERIFY
- **Rule-based diagnosis**: bufferbloat, weak WiFi, band congestion, high latency, security risk
- **4 SmartOS tools**: SQM, band steering, firmware upgrade, reboot (mocked)
- **Human-in-the-loop**: pending recommendations for manual approval
- **Fleet scanner**: runs every 15 min via APScheduler
- **Notification dispatch**: email (SMTP), Slack (webhook), generic webhook on escalation

### Phase 3: LLM Layer
- **Ollama provider**: async HTTP client with streaming support
- **Embeddings**: sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU)
- **Vector store**: pgvector on TimescaleDB
- **RAG pipeline**: embed > retrieve > generate
- **Chat API**: REST + WebSocket streaming
- **Insights**: fleet health summary, per-device diagnosis

## Quick Start

### Prerequisites
- Python 3.12+
- Docker (for TimescaleDB, Redis, Mosquitto)
- Ollama with llama3.1:8b model

### Setup

```bash
# Start infrastructure
docker compose up -d

# Install dependencies
pip install -e ".[dev]"

# Pull Ollama model
ollama pull llama3.1:8b

# Seed test data (100 devices, 4,200 feature snapshots)
PYTHONPATH=src python scripts/seed_features.py

# Start the backend
PYTHONPATH=src uvicorn netpulse_ml.main:app --host 0.0.0.0 --port 8000

# Train models
curl -X POST http://localhost:8000/v1/models/retrain-all

# Run end-to-end test (17 endpoints)
python scripts/e2e_test.py
```

### API Docs
Once running, visit `http://localhost:8000/docs` for the OpenAPI/Swagger documentation.

### Key Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/health` | GET | Service health + model status |
| `/v1/anomalies` | GET | Fleet anomaly list (cached) |
| `/v1/devices/{id}/anomaly-score` | GET | Device anomaly detail |
| `/v1/churn/predictions` | GET | Churn risk list (cached) |
| `/v1/subscribers/{id}/churn` | GET | Subscriber churn prediction |
| `/v1/devices/{id}/qoe-forecast` | GET | QoE forecast with confidence bands |
| `/v1/fleet/clusters` | GET | Fleet segments (cached) |
| `/v1/devices/{id}/recommendations` | GET | ML recommendations |
| `/v1/recommendations/{id}/approve` | POST | Approve recommendation |
| `/v1/models` | GET | Model registry |
| `/v1/models/{name}/retrain` | POST | Trigger model retraining |
| `/v1/agents/status` | GET | Agent orchestrator status |
| `/v1/agents/history` | GET | Agent execution history |
| `/v1/agents/{device_id}/trigger` | POST | Manually trigger agent |
| `/v1/agents/config` | GET/PUT | Agent configuration |
| `/v1/insights/fleet-summary` | GET | AI fleet health summary |
| `/v1/insights/device/{id}` | GET | AI device diagnosis |
| `/v1/chat` | POST | RAG-powered Q&A |
| `/v1/chat/ws` | WebSocket | Streaming chat |

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

42 tests covering:
- Anomaly detector (8): init, train, predict, save/load, missing features
- Churn predictor (6): init, train, predict, risk levels
- Fleet clusterer (5): init, train, predict, summary, save/load
- Feature extraction (5): BBST, QoE, WiFi
- MQTT parsers (5): topic parsing, payload validation
- Agent nodes (8): diagnose (4 diagnoses), plan (4 actions)
- Agent tools (5): SQM, band steering, firmware, reboot, registry

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):
1. **Lint**: ruff check on src/ and tests/
2. **Test**: pytest with TimescaleDB + Redis Docker services
3. **Docker**: build and push to ghcr.io (master only)

## Tech Stack

- **Framework**: FastAPI (async)
- **ML**: scikit-learn, statsmodels, SHAP
- **Agents**: LangGraph (deterministic StateGraph)
- **LLM**: Ollama (llama3.1:8b)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Database**: TimescaleDB (PostgreSQL) + pgvector
- **Cache**: Redis (async, TTL-based)
- **MQTT**: aiomqtt
- **Notifications**: SMTP email, Slack webhook, generic webhook
- **Scheduling**: APScheduler
- **CI/CD**: GitHub Actions

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | TimescaleDB connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for prediction caching |
| `MQTT_BROKER` | `localhost` | MQTT broker host |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model name |
| `API_KEY` | (empty) | API key (empty = no auth, dev mode) |
| `AGENT_ENABLE_AUTO_EXECUTE` | `false` | Allow agents to auto-execute actions |
| `AGENT_ANOMALY_THRESHOLD` | `0.5` | Min anomaly score to trigger agent |
| `AGENT_SCAN_INTERVAL_MINUTES` | `15` | Fleet scan frequency |
| `NOTIFICATIONS_ENABLED` | `true` | Global notification toggle |
| `SMTP_HOST` | (empty) | SMTP server for email notifications |
| `SLACK_WEBHOOK_URL` | (empty) | Slack incoming webhook URL |
| `NOTIFY_WEBHOOK_URL` | (empty) | Generic webhook (PagerDuty, etc.) |
