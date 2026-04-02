"""Pydantic response schemas matching the TypeScript types in src/types/ml.ts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int


class OutlierInfo(BaseModel):
    deviceCount: int = 0


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------
class FeatureContribution(BaseModel):
    name: str
    value: float
    zscore: float


class AnomalyScoreResponse(BaseModel):
    deviceId: str
    anomalyScore: float = Field(ge=0, le=1)
    isAnomaly: bool
    modelReady: bool = True
    topFeatures: list[FeatureContribution] = []
    scoredAt: datetime
    modelVersion: str


class AnomalyListItem(BaseModel):
    deviceId: str
    anomalyScore: float
    isAnomaly: bool = True
    topFeatures: list[FeatureContribution] = []
    scoredAt: datetime
    modelVersion: str


class AnomalyListResponse(BaseModel):
    data: list[AnomalyListItem]
    pagination: PaginationMeta


# ---------------------------------------------------------------------------
# Churn Prediction (mirrors ChurnPrediction from src/types/ml.ts)
# ---------------------------------------------------------------------------
class ChurnPredictionResponse(BaseModel):
    deviceId: str
    subscriberId: str
    riskScore: float = Field(ge=0, le=100)
    riskLevel: Literal["low", "medium", "high", "critical"]
    topFactors: list[str] = []
    modelReady: bool = True
    predictedAt: datetime
    modelVersion: str


class ChurnListResponse(BaseModel):
    data: list[ChurnPredictionResponse]
    pagination: PaginationMeta


# ---------------------------------------------------------------------------
# QoE Forecasting
# ---------------------------------------------------------------------------
class ForecastPoint(BaseModel):
    timestamp: str
    predicted: float
    lower95: float
    upper95: float


class ModelFitStats(BaseModel):
    aic: float = 0
    bic: float = 0


class QoEForecastResponse(BaseModel):
    deviceId: str
    currentQoE: float
    forecast: list[ForecastPoint] = []
    trendDirection: Literal["improving", "stable", "declining"]
    modelFit: ModelFitStats = ModelFitStats()
    forecastedAt: datetime
    modelVersion: str
    warning: str | None = None


# ---------------------------------------------------------------------------
# Fleet Clustering
# ---------------------------------------------------------------------------
class ClusterInfo(BaseModel):
    clusterId: int
    label: str
    deviceCount: int
    isOutlier: bool = False
    avgQoE: float = 0
    avgDlMbps: float = 0


class FleetClustersResponse(BaseModel):
    clusters: list[ClusterInfo]
    outliers: OutlierInfo = OutlierInfo()
    silhouetteScore: float = 0
    clusterCount: int
    clusteredAt: datetime
    modelVersion: str


# ---------------------------------------------------------------------------
# Recommendations (mirrors MLRecommendation from src/types/ml.ts)
# ---------------------------------------------------------------------------
RecommendationType = Literal[
    "firmware_upgrade", "enable_sqm", "band_steering", "mesh_ap_add", "service_tier_change"
]


class MLRecommendationResponse(BaseModel):
    id: str
    deviceId: str
    type: RecommendationType
    title: str
    description: str
    confidence: float
    impact: Literal["high", "medium", "low"]
    autoExecutable: bool
    createdAt: datetime
    status: str = "pending"


class DeviceRecommendationsResponse(BaseModel):
    deviceId: str
    recommendations: list[MLRecommendationResponse] = []


class ApproveRequest(BaseModel):
    executedBy: str = Field(max_length=64, pattern=r"^[a-zA-Z0-9_.\-@]+$")


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
class ModelInfo(BaseModel):
    name: str
    activeVersion: str | None = None
    algorithm: str
    trainedAt: datetime | None = None
    metrics: dict | None = None
    featureCount: int = 0


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    models: dict[str, str] = {}
    mqttConnected: bool = False
    dbConnected: bool = False
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Agent Orchestrator
# ---------------------------------------------------------------------------
class AgentStatusResponse(BaseModel):
    isRunning: bool
    lastScanAt: datetime | None = None
    lastScanFlagged: int = 0
    totalRuns: int = 0
    cooldownDevices: int = 0
    scanIntervalMinutes: int = 15
    anomalyThreshold: float = 0.5
    autoExecuteEnabled: bool = False


class AgentExecutionItem(BaseModel):
    id: str
    deviceId: str
    startedAt: datetime
    completedAt: datetime | None = None
    status: str
    diagnosis: str = ""
    recommendedAction: str | None = None
    autoExecuted: bool = False
    verified: bool = False
    recommendationId: str | None = None


class AgentHistoryResponse(BaseModel):
    data: list[AgentExecutionItem]
    pagination: PaginationMeta


class AgentTriggerResponse(BaseModel):
    executionId: str
    deviceId: str
    status: str
    diagnosis: str | None = None
    recommendedAction: str | None = None


class AgentConfigRequest(BaseModel):
    anomalyThreshold: float | None = Field(None, ge=0, le=1)
    scanIntervalMinutes: int | None = Field(None, ge=1, le=1440)
    cooldownHours: int | None = Field(None, ge=0, le=168)
    autoExecuteEnabled: bool | None = None


# ---------------------------------------------------------------------------
# LLM Chat & Insights
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    message: str
    response: str
    generatedAt: datetime
    model: str


class InsightResponse(BaseModel):
    content: str
    generatedAt: datetime
    model: str
