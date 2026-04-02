"""SQLAlchemy ORM models for the ML pipeline."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Timezone-aware timestamp type for all datetime columns
TZDateTime = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class FeatureSnapshot(Base):
    """Per-device feature snapshots stored as JSONB in a TimescaleDB hypertable."""

    __tablename__ = "feature_snapshots"

    device_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(TZDateTime, primary_key=True, server_default=func.now())
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)


class Prediction(Base):
    """ML model predictions stored in a TimescaleDB hypertable."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subscriber_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now(), index=True)
    prediction: Mapped[dict] = mapped_column(JSONB, nullable=False)
    features_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ModelRegistry(Base):
    """Registered ML model versions with metadata and metrics."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    feature_names: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    trained_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
    )


class Recommendation(Base):
    """ML-generated recommendations matching the frontend MLRecommendation type."""

    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    impact: Mapped[str] = mapped_column(String(16), nullable=False)
    auto_executable: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())
    executed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    executed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_recommendations_device_status", "device_id", "status"),
    )


class AgentExecution(Base):
    """Agent execution history for audit trail and debugging."""

    __tablename__ = "agent_executions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    diagnosis: Mapped[str] = mapped_column(String(64), default="")
    recommended_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    auto_executed: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    recommendation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


class ClusterAssignment(Base):
    """Fleet cluster assignments refreshed nightly."""

    __tablename__ = "cluster_assignments"

    device_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    cluster_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_outlier: Mapped[bool] = mapped_column(Boolean, default=False)
    distance_to_centroid: Mapped[float | None] = mapped_column(Float, nullable=True)
    assigned_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())
