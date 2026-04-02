# NetPulse ML - Developer Notes

## Critical: MQTT Client Library

**Use `gmqtt`, NOT `aiomqtt` or `paho-mqtt`.**

aiomqtt wraps paho-mqtt, which uses `add_reader`/`add_writer` on the asyncio event loop. These are only available on `SelectorEventLoop` (Linux default) but NOT on Windows' `ProactorEventLoop`. This causes `NotImplementedError` crashes on Windows.

`gmqtt` is a pure async MQTT client that works on all platforms without event loop workarounds. No `SelectorEventLoopPolicy` hack needed.

## Critical: SmartOS Router API

SmartOS routers use **JUCI WebSocket API**, NOT standard ubus HTTP.

- **Endpoint**: `ws://<router_ip>/websocket/` (not `/cgi-bin/ubus`)
- **Auth**: Challenge-response (md5crypt + MD5 hash), not HTTP basic auth
- **Object names**: Use `/`-prefixed paths (`/system`, `/uci`, `/network.wireless`), not bare names
- **Client**: `src/netpulse_ml/agents/smartos_client.py`
- **JUCI source**: https://github.com/Adtran-SOS/juci-openwrt-feed (Adtran SSO required)

## Critical: Database Timestamps

All SQLAlchemy datetime columns MUST use `DateTime(timezone=True)` (maps to TIMESTAMPTZ in PostgreSQL). Using bare `datetime` maps to `TIMESTAMP WITHOUT TIME ZONE` which causes asyncpg errors when Python passes timezone-aware datetimes.

## Critical: pgvector Query Syntax

When using pgvector with SQLAlchemy `text()` queries, use `cast(:param AS vector)` instead of `:param::vector`. The `::` cast syntax conflicts with SQLAlchemy's `:name` parameter binding.

## ML Inference in Async Endpoints

All scikit-learn `.predict()`, `.fit()`, and sentence-transformers `.encode()` calls are CPU-bound and synchronous. Always wrap with `await run_in_executor(fn, args)` (from `dependencies.py`) when calling from async code. Never call these directly in `async def` endpoint handlers.

## Runtime

- **Bun**, not Node.js, is the frontend JS runtime (see parent CLAUDE.md)
- **Python 3.12+** for the backend (3.14 tested on Windows)
- **Docker Compose** provides TimescaleDB, Redis, and Mosquitto
- **Ollama** required for LLM features (pull `llama3.1:8b`)
