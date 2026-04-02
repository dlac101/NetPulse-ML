"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database (TimescaleDB) - no default forces explicit config in production
    database_url: str = "postgresql+asyncpg://netpulse:netpulse@localhost:5432/netpulse_ml"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MQTT
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_topics: str = (
        "smartos/+/bbst,smartos/+/qoe,smartos/+/classifi,"
        "smartos/+/flowstatd/#,smartos/+/wifi,smartos/+/events,smartos/+/meta"
    )

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Model artifacts directory
    model_dir: Path = Path("./models")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1  # Single worker: models loaded per-process, MQTT consumer per-process
    log_level: str = "info"

    # Security
    api_key: str = ""  # Set in production; empty = no auth (dev mode)
    cors_origins: str = "http://localhost:3000"  # Comma-separated allowed origins

    # Cache TTLs (seconds)
    cache_ttl_anomaly: int = 300
    cache_ttl_churn: int = 3600
    cache_ttl_forecast: int = 900

    # Fleet query safety
    fleet_query_limit: int = 50000  # Max devices in a single fleet query

    # Agent orchestrator
    agent_scan_interval_minutes: int = 15
    agent_anomaly_threshold: float = 0.5
    agent_cooldown_hours: int = 4
    agent_max_concurrent: int = 5
    agent_enable_auto_execute: bool = False  # Safety: disabled by default
    agent_verify_delay_minutes: int = 15

    # LLM / Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_temperature: float = 0.3

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # RAG
    rag_top_k: int = 5
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50

    @property
    def mqtt_topic_list(self) -> list[str]:
        return [t.strip() for t in self.mqtt_topics.split(",") if t.strip()]

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
