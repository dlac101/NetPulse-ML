"""Pydantic models for validating raw MQTT payloads.

These mirror the TypeScript types in netpulse-admin/src/types/ exactly.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# QoE (src/types/qoe.ts)
# ---------------------------------------------------------------------------
class QoEImpactFactor(BaseModel):
    name: str
    subject: str
    detail: str
    penalty: float
    severity: Literal["critical", "major", "minor"]
    timestamp: datetime


class QoECategory(BaseModel):
    name: Literal["wan", "wifi", "mesh", "system", "lan"]
    score: float = Field(ge=0, le=100)
    grade: Literal["A", "B", "C", "D", "F"]
    weight: float = Field(ge=0, le=1)
    impactFactors: list[QoEImpactFactor] = []


class QoECategories(BaseModel):
    wan: QoECategory
    wifi: QoECategory
    mesh: QoECategory
    system: QoECategory
    lan: QoECategory


class QoEPayload(BaseModel):
    id: str
    deviceId: str
    timestamp: datetime
    compositeScore: float = Field(ge=0, le=100)
    compositeGrade: Literal["A", "B", "C", "D", "F"]
    categories: QoECategories
    satelliteCount: int = 0
    durationSec: float = 0
    anomalyScore: float | None = None


# ---------------------------------------------------------------------------
# Speed Test / BBST (src/types/speed-test.ts + bbst-record-schema.md)
# ---------------------------------------------------------------------------
class SpeedMetrics(BaseModel):
    mbps: float
    capacityPercent: float = 0
    bloatPercent: float = 0
    bloatGrade: Literal["A", "B", "C", "D"] = "A"
    provisionedMbps: float = 0


class LatencyMetrics(BaseModel):
    idleMs: float
    downloadMs: float
    uploadMs: float
    idleJitterMs: float = 0
    downloadJitterMs: float = 0
    uploadJitterMs: float = 0


class ServerInfo(BaseModel):
    host: str = ""
    city: str = ""
    country: str = ""
    distanceKm: float = 0


class ClientInfo(BaseModel):
    ip: str = ""
    isp: str = ""
    city: str = ""
    country: str = ""


class BbstPayload(BaseModel):
    id: str
    deviceId: str
    timestamp: datetime
    download: SpeedMetrics
    upload: SpeedMetrics
    latency: LatencyMetrics
    server: ServerInfo = ServerInfo()
    client: ClientInfo = ClientInfo()
    durationSec: float = 0
    testType: Literal["scheduled", "on_demand", "technician"] = "scheduled"
    anomalyScore: float | None = None


# ---------------------------------------------------------------------------
# WiFi (src/types/wifi.ts)
# ---------------------------------------------------------------------------
class WifiClient(BaseModel):
    mac: str
    hostname: str | None = None
    band: Literal["2.4GHz", "5GHz", "6GHz"]
    channel: int
    channelWidth: int = 20
    phyRateMbps: float = 0
    rssi: float = 0
    retransmissionRate: float = 0
    mcs: int = 0
    nss: int = 1
    connectedSatellite: str | None = None
    associatedSince: datetime | None = None
    rxBytes: int = 0
    txBytes: int = 0


class MeshSatellite(BaseModel):
    id: str
    hostname: str = ""
    mac: str
    backhaulDlMbps: float = 0
    backhaulUlMbps: float = 0
    backhaulBand: Literal["2.4GHz", "5GHz", "6GHz"] = "5GHz"
    connectedClients: int = 0
    hops: int = 1
    status: Literal["online", "offline", "degraded"] = "online"


class WifiAirtime(BaseModel):
    band: Literal["2.4GHz", "5GHz", "6GHz"]
    channel: int
    txPercent: float = 0
    rxPercent: float = 0
    wifiInterferencePercent: float = 0
    nonWifiInterferencePercent: float = 0
    totalUtilizationPercent: float = 0
    clientCount: int = 0


class WifiPayload(BaseModel):
    clients: list[WifiClient] = []
    satellites: list[MeshSatellite] = []
    airtime: list[WifiAirtime] = []


# ---------------------------------------------------------------------------
# Traffic / DPI (src/types/traffic.ts)
# ---------------------------------------------------------------------------
class TrafficCategory(BaseModel):
    masterProtocol: str
    appProtocol: str
    category: str
    rxBytes: int = 0
    txBytes: int = 0
    totalBytes: int = 0
    activeTimeSec: float = 0
    flowCount: int = 0
    rxRate: float = 0
    txRate: float = 0
    maxRiskScore: int = 0


class ActiveFlow(BaseModel):
    id: str
    deviceId: str
    srcIp: str
    srcPort: int
    dstIp: str
    dstPort: int
    protocol: Literal["TCP", "UDP", "OTHER"]
    masterProtocol: str = ""
    appProtocol: str = ""
    category: str = ""
    hostname: str | None = None
    riskScore: int = 0
    riskTypes: list[str] = []
    ja4Client: str | None = None
    tcpFingerprint: str | None = None
    osHint: str | None = None
    protocolStack: list[str] = []
    rxBytes: int = 0
    txBytes: int = 0
    startedAt: datetime | None = None
    lastSeenAt: datetime | None = None


class FlowStatsPayload(BaseModel):
    device_mac: str
    categories: list[TrafficCategory] = []


class ClassifiPayload(BaseModel):
    flows: list[ActiveFlow] = []
    timestamp: datetime | None = None


class CategoryHourlyEntry(BaseModel):
    deviceId: str
    mac: str
    masterProtocol: str
    appProtocol: str
    category: str
    hour: str
    rxBytes: int = 0
    txBytes: int = 0
    activeTimeSec: float = 0


class CategoryHoursPayload(BaseModel):
    device_mac: str
    hours: list[CategoryHourlyEntry] = []


# ---------------------------------------------------------------------------
# Events (src/types/event.ts)
# ---------------------------------------------------------------------------
EVENT_TYPES = [
    "high_packet_loss",
    "download_capacity_limited",
    "upload_capacity_limited",
    "bloat_exceeds_threshold",
    "qoe_score_drop",
    "firmware_version_changed",
    "connection_lost",
    "connection_restored",
    "low_link_utilization",
    "retransmission_rate_high",
    "security_risk_detected",
    "mesh_satellite_offline",
    "wifi_interference_high",
    "speed_test_completed",
]


class EventPayload(BaseModel):
    id: str
    deviceId: str
    type: str
    severity: Literal["critical", "warning", "info"]
    message: str = ""
    detail: str = ""
    timestamp: datetime
    acknowledged: bool = False


# ---------------------------------------------------------------------------
# Device Metadata (heartbeat)
# ---------------------------------------------------------------------------
class DeviceGeo(BaseModel):
    lat: float
    lng: float
    address: str | None = None
    city: str | None = None
    state: str | None = None
    zip: str | None = None
    country: str | None = None
    serviceArea: str | None = None


class MetaPayload(BaseModel):
    mac: str
    model: str
    firmware: str
    status: Literal["online", "offline", "degraded"]
    geo: DeviceGeo | None = None
    subscriberId: str | None = None
    timestamp: datetime
