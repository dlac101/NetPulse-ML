# NetPulse ML Backend

AI/ML pipeline for the NetPulse Admin Console. Provides anomaly detection, churn prediction, QoE forecasting, fleet segmentation, autonomous remediation agents, and RAG-powered natural language insights for SmartOS router fleets.

## Architecture

```
SmartOS Routers --> MQTT Broker --> netpulse-ml (FastAPI)
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
              Feature Store       ML Models          LLM / RAG
             (TimescaleDB)     (scikit-learn)     (Ollama + pgvector)
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
- **13 REST API endpoints** with thread-offloaded inference

### Phase 2: Autonomous Agents
- **LangGraph StateGraph**: ANALYZE > DIAGNOSE > PLAN > EXECUTE/ESCALATE > VERIFY
- **Rule-based diagnosis**: bufferbloat, weak WiFi, band congestion, high latency, security risk
- **4 SmartOS tools**: SQM, band steering, firmware upgrade, reboot (mocked)
- **Human-in-the-loop**: pending recommendations for manual approval
- **Fleet scanner**: runs every 15 min via APScheduler

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
- Docker (for TimescaleDB, Redis, MLflow, Mosquitto)
- Ollama with llama3.1:8b model

### Setup

```bash
# Start infrastructure
docker compose up -d

# Install dependencies
pip install -e ".[dev]"

# Pull Ollama model
ollama pull llama3.1:8b

# Run the backend
uvicorn netpulse_ml.main:app --reload
```

### API Docs
Once running, visit `http://localhost:8000/docs` for the OpenAPI/Swagger documentation.

### Key Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/health` | GET | Service health + model status |
| `/v1/anomalies` | GET | Fleet anomaly list |
| `/v1/devices/{id}/anomaly-score` | GET | Device anomaly detail |
| `/v1/churn/predictions` | GET | Churn risk list |
| `/v1/devices/{id}/qoe-forecast` | GET | QoE forecast with confidence bands |
| `/v1/fleet/clusters` | GET | Fleet segments |
| `/v1/devices/{id}/recommendations` | GET | ML recommendations |
| `/v1/agents/status` | GET | Agent orchestrator status |
| `/v1/agents/{device_id}/trigger` | POST | Manually trigger agent |
| `/v1/insights/fleet-summary` | GET | AI fleet health summary |
| `/v1/insights/device/{id}` | GET | AI device diagnosis |
| `/v1/chat` | POST | RAG-powered Q&A |
| `/v1/chat/ws` | WebSocket | Streaming chat |

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

24 tests covering: anomaly detector, churn predictor, feature extraction, MQTT parsers.

## Tech Stack

- **Framework**: FastAPI (async)
- **ML**: scikit-learn, statsmodels, SHAP
- **Agents**: LangGraph (deterministic StateGraph)
- **LLM**: Ollama (llama3.1:8b)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Database**: TimescaleDB (PostgreSQL) + pgvector
- **Cache**: Redis
- **MQTT**: aiomqtt
- **Scheduling**: APScheduler

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | TimescaleDB connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MQTT_BROKER` | `localhost` | MQTT broker host |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model name |
| `API_KEY` | (empty) | API key (empty = no auth, dev mode) |
| `AGENT_ENABLE_AUTO_EXECUTE` | `false` | Allow agents to auto-execute actions |
