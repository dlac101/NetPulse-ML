"""Feature store: read/write device features to TimescaleDB."""

import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from netpulse_ml.config import settings
from netpulse_ml.db.engine import async_session_factory
from netpulse_ml.db.models import FeatureSnapshot
from netpulse_ml.features.device_features import (
    extract_bbst_features,
    extract_classifi_features,
    extract_event_features,
    extract_qoe_features,
    extract_traffic_features,
    extract_wifi_features,
)
from netpulse_ml.ingestion.validators import (
    BbstPayload,
    ClassifiPayload,
    EventPayload,
    FlowStatsPayload,
    QoEPayload,
    WifiPayload,
)

log = structlog.get_logger()

# Union of all parseable payload types
TelemetryPayload = (
    BbstPayload | QoEPayload | WifiPayload | FlowStatsPayload | ClassifiPayload | EventPayload
)


class FeatureStore:
    """Manages feature snapshot lifecycle: compute, write, read."""

    async def process_message(
        self, device_id: str, msg_type: str, payload: TelemetryPayload
    ) -> None:
        """Compute features from a parsed telemetry message and write to store."""
        features: dict[str, float] = {}

        if isinstance(payload, BbstPayload):
            features = extract_bbst_features(payload)
        elif isinstance(payload, QoEPayload):
            features = extract_qoe_features(payload)
        elif isinstance(payload, WifiPayload):
            features = extract_wifi_features(payload)
        elif isinstance(payload, FlowStatsPayload):
            features = extract_traffic_features(payload)
        elif isinstance(payload, ClassifiPayload):
            features = extract_classifi_features(payload)
        elif isinstance(payload, EventPayload):
            features = extract_event_features(payload)
        else:
            return

        if not features:
            return

        await self.write_features(device_id, features)

    async def write_features(
        self,
        device_id: str,
        features: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        """Write a feature snapshot to TimescaleDB with retry on transient failure."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        for attempt in range(3):
            try:
                async with async_session_factory() as session:
                    snapshot = FeatureSnapshot(
                        device_id=device_id,
                        timestamp=timestamp,
                        features=features,
                    )
                    session.add(snapshot)
                    await session.commit()
                    return
            except OperationalError as e:
                log.warning("DB write failed, retrying", attempt=attempt, error=str(e))
                await asyncio.sleep(1)
            except Exception as e:
                log.error("Feature write failed", device_id=device_id, error=str(e))
                return

        log.error("Feature write permanently failed after retries", device_id=device_id)

    async def read_features(
        self,
        device_id: str,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
    ) -> pd.DataFrame:
        """Read feature snapshots for a device within a time range.

        Returns a DataFrame with timestamp index and feature columns (from JSONB).
        """
        if to_ts is None:
            to_ts = datetime.now(UTC)
        if from_ts is None:
            from_ts = to_ts - timedelta(days=7)

        async with async_session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT timestamp, features
                    FROM feature_snapshots
                    WHERE device_id = :device_id
                      AND timestamp >= :from_ts
                      AND timestamp <= :to_ts
                    ORDER BY timestamp ASC
                    """
                ),
                {"device_id": device_id, "from_ts": from_ts, "to_ts": to_ts},
            )
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            entry = {"timestamp": row[0]}
            entry.update(row[1])
            records.append(entry)

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    async def get_latest_features(self, device_id: str) -> dict[str, float]:
        """Get the most recent feature snapshot for a device."""
        async with async_session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT features FROM feature_snapshots
                    WHERE device_id = :device_id
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                ),
                {"device_id": device_id},
            )
            row = result.fetchone()

        if row is None:
            return {}
        return dict(row[0])

    async def get_fleet_features(
        self,
        device_ids: list[str] | None = None,
        feature_names: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Get latest features for multiple devices (fleet-level).

        Returns DataFrame with device_id index and feature columns.
        Hard-capped at settings.fleet_query_limit to prevent OOM at scale.
        """
        max_limit = limit or settings.fleet_query_limit

        async with async_session_factory() as session:
            if device_ids:
                query = text(
                    """
                    SELECT DISTINCT ON (device_id) device_id, features
                    FROM feature_snapshots
                    WHERE device_id = ANY(:device_ids)
                    ORDER BY device_id, timestamp DESC
                    LIMIT :max_limit
                    """
                )
                result = await session.execute(
                    query, {"device_ids": device_ids, "max_limit": max_limit}
                )
            else:
                query = text(
                    """
                    SELECT DISTINCT ON (device_id) device_id, features
                    FROM feature_snapshots
                    ORDER BY device_id, timestamp DESC
                    LIMIT :max_limit
                    """
                )
                result = await session.execute(query, {"max_limit": max_limit})

            rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            entry = {"device_id": row[0]}
            feats = row[1]
            if feature_names:
                feats = {k: v for k, v in feats.items() if k in feature_names}
            entry.update(feats)
            records.append(entry)

        df = pd.DataFrame(records)
        df.set_index("device_id", inplace=True)
        return df


# Singleton instance (stateless - safe for concurrent async access)
feature_store = FeatureStore()
