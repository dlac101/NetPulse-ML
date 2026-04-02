"""FastAPI application factory with lifespan events."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from netpulse_ml.api.router import api_router
from netpulse_ml.config import settings
from netpulse_ml.db.engine import engine, init_db
from netpulse_ml.dependencies import verify_api_key
from netpulse_ml.ingestion.mqtt_consumer import MQTTConsumer
from netpulse_ml.agents.orchestrator import AgentOrchestrator
from netpulse_ml.llm.embedder import Embedder
from netpulse_ml.llm.indexer import Indexer
from netpulse_ml.llm.provider import OllamaProvider
from netpulse_ml.llm.rag import RAGPipeline
from netpulse_ml.llm.vector_store import VectorStore
from netpulse_ml.serving.predictor import Predictor
from netpulse_ml.training.scheduler import create_training_scheduler

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle for the application."""
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
    )

    log.info("Starting NetPulse ML backend", version="0.1.0")

    # Initialize database tables
    await init_db()
    log.info("Database initialized")

    # Load ML models into memory
    predictor = Predictor(settings.model_dir)
    predictor.load_all()
    app.state.predictor = predictor
    log.info("Model predictor initialized", models_loaded=predictor.loaded_model_names)

    # Start MQTT consumer as background task
    # Skip on Windows: aiomqtt uses paho-mqtt which requires SelectorEventLoop,
    # but uvicorn on Windows uses ProactorEventLoop (no add_reader/add_writer support)
    import sys
    mqtt_task = None
    if sys.platform != "win32":
        mqtt = MQTTConsumer(settings)
        mqtt_task = asyncio.create_task(mqtt.run())
        app.state.mqtt_consumer = mqtt
        log.info("MQTT consumer started", broker=settings.mqtt_broker)
    else:
        log.warning("MQTT consumer skipped on Windows (ProactorEventLoop incompatible with paho-mqtt)")

    # Start training scheduler
    scheduler = create_training_scheduler()
    scheduler.start()
    app.state.scheduler = scheduler
    log.info("Training scheduler started")

    # Start agent orchestrator
    agent_orch = AgentOrchestrator(settings, predictor)
    agent_orch.start_scheduler()
    app.state.agent_orchestrator = agent_orch
    log.info("Agent orchestrator started")

    # Initialize LLM / RAG pipeline (non-fatal: API works without LLM)
    ollama = OllamaProvider()
    try:
        embedder = Embedder()
        await embedder.load_async()
        vector_store = VectorStore()
        await vector_store.initialize()
        indexer = Indexer(embedder, vector_store)
        await indexer.index_project_docs()
        rag = RAGPipeline(embedder, vector_store, ollama, predictor)

        app.state.ollama_provider = ollama
        app.state.embedder = embedder
        app.state.vector_store = vector_store
        app.state.indexer = indexer
        app.state.rag_pipeline = rag

        ollama_ok = await ollama.is_available()
        log.info("LLM/RAG pipeline initialized", ollama_available=ollama_ok)
    except Exception as e:
        log.warning("LLM/RAG pipeline failed to initialize (non-fatal)", error=str(e))
        app.state.ollama_provider = ollama

    yield

    # Shutdown
    log.info("Shutting down NetPulse ML backend")
    await ollama.close()
    agent_orch.stop_scheduler()
    scheduler.shutdown(wait=False)
    if mqtt_task and hasattr(app.state, "mqtt_consumer"):
        app.state.mqtt_consumer.stop()
        mqtt_task.cancel()
        try:
            await mqtt_task
        except asyncio.CancelledError:
            pass
    await engine.dispose()
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NetPulse ML API",
        description="AI/ML backend for SmartOS router fleet analytics",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/v1")

    return app


app = create_app()
