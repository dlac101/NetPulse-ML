"""Import historical BBST speed test records into the feature store.

Reads real BBST JSON files (bbst_results.json, bbst_web_full_*.json)
from the project root and inserts them as feature snapshots.

Usage:
    PYTHONPATH=src python scripts/import_bbst.py [path_to_json_file]
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from netpulse_ml.db.engine import init_db
from netpulse_ml.features.device_features import extract_bbst_features
from netpulse_ml.features.store import feature_store
from netpulse_ml.ingestion.validators import BbstPayload, LatencyMetrics, SpeedMetrics


def parse_bbst_record(data: dict) -> BbstPayload | None:
    """Parse a raw BBST JSON record into our BbstPayload format."""
    try:
        dl = data.get("download", {})
        ul = data.get("upload", {})
        lat = data.get("latency", {})
        client = data.get("client", {})
        server = data.get("server", {})

        return BbstPayload(
            id=data.get("test_id", f"bbst-{data.get('starttime', 0)}"),
            deviceId=f"bbst-{client.get('ip', 'unknown')}",
            timestamp=datetime.fromtimestamp(
                data.get("starttime", 0), tz=timezone.utc
            ).isoformat(),
            download=SpeedMetrics(
                mbps=dl.get("mbps", 0),
                capacityPercent=dl.get("pct_utilization", 0),
                provisionedMbps=data.get("test_options", {}).get(
                    "downstream_service_rate", 0
                ),
            ),
            upload=SpeedMetrics(
                mbps=ul.get("mbps", 0),
                capacityPercent=ul.get("pct_utilization", 0),
                provisionedMbps=data.get("test_options", {}).get(
                    "upstream_service_rate", 0
                ),
            ),
            latency=LatencyMetrics(
                idleMs=lat.get("idle_avg", 0),
                downloadMs=lat.get("download_avg", 0),
                uploadMs=lat.get("upload_avg", 0),
                idleJitterMs=lat.get("idle_jitter", 0),
                downloadJitterMs=lat.get("download_jitter", 0),
                uploadJitterMs=lat.get("upload_jitter", 0),
            ),
            durationSec=data.get("runtime", 0),
            testType="scheduled",
        )
    except Exception as e:
        print(f"  Failed to parse record: {e}")
        return None


async def import_file(file_path: Path) -> int:
    """Import a single BBST JSON file."""
    print(f"Importing {file_path.name}...")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    payload = parse_bbst_record(data)
    if payload is None:
        return 0

    features = extract_bbst_features(payload)
    ts = datetime.fromtimestamp(data.get("starttime", 0), tz=timezone.utc)

    await feature_store.write_features(
        device_id=payload.deviceId,
        features=features,
        timestamp=ts,
    )

    print(f"  Imported: {payload.deviceId} at {ts}, DL={payload.download.mbps:.0f} Mbps")
    return 1


async def main() -> None:
    await init_db()

    # Find BBST files
    project_root = Path(__file__).parent.parent.parent
    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
    else:
        files = list(project_root.glob("bbst*.json"))

    if not files:
        print("No BBST JSON files found. Pass a file path as argument.")
        return

    total = 0
    for f in files:
        total += await import_file(f)

    print(f"\nDone: {total} records imported from {len(files)} files.")


if __name__ == "__main__":
    asyncio.run(main())
