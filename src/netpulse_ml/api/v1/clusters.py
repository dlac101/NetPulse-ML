"""Fleet clustering/segmentation API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from netpulse_ml.api.schemas import ClusterInfo, FleetClustersResponse, OutlierInfo, PaginationMeta
from netpulse_ml.config import settings
from netpulse_ml.dependencies import FeatureStoreDep, PredictorDep, run_in_executor
from netpulse_ml.serving.cache import get_cached, make_cache_key, set_cached

router = APIRouter()


@router.get("/fleet/clusters", response_model=FleetClustersResponse)
async def get_fleet_clusters(
    predictor: PredictorDep,
    store: FeatureStoreDep,
) -> FleetClustersResponse:
    """Get fleet cluster assignments and characteristics."""
    clusterer = predictor.fleet_clusterer
    now = datetime.now(timezone.utc)

    if not clusterer.is_fitted:
        raise HTTPException(status_code=503, detail="Fleet clusterer not yet trained")

    cache_key = make_cache_key("clusters")
    cached = await get_cached(cache_key)
    if cached is not None:
        return FleetClustersResponse(**cached)

    fleet_df = await store.get_fleet_features()
    if fleet_df.empty:
        return FleetClustersResponse(
            clusters=[],
            clusterCount=0,
            clusteredAt=now,
            modelVersion=clusterer.version,
        )

    # Offload CPU-bound clustering to thread pool
    summaries = await run_in_executor(clusterer.get_cluster_summary, fleet_df)

    clusters = []
    outlier_info = OutlierInfo()
    for s in summaries:
        if s["isOutlier"]:
            outlier_info = OutlierInfo(deviceCount=s["deviceCount"])
        else:
            clusters.append(ClusterInfo(
                clusterId=s["clusterId"],
                label=s["label"],
                deviceCount=s["deviceCount"],
                avgQoE=s.get("avg_qoe_composite_latest", 0.0),
                avgDlMbps=s.get("avg_dl_mbps_latest", 0.0),
            ))

    response = FleetClustersResponse(
        clusters=clusters,
        outliers=outlier_info,
        clusterCount=len(clusters),
        clusteredAt=now,
        modelVersion=clusterer.version,
    )
    await set_cached(cache_key, response.model_dump(mode="json"), 300)  # 5 min cache
    return response


@router.get("/fleet/segments/{cluster_id}/devices")
async def get_cluster_devices(
    predictor: PredictorDep,
    store: FeatureStoreDep,
    cluster_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """List devices in a specific cluster."""
    clusterer = predictor.fleet_clusterer

    if not clusterer.is_fitted:
        raise HTTPException(status_code=503, detail="Fleet clusterer not yet trained")

    fleet_df = await store.get_fleet_features()
    if fleet_df.empty:
        return {"clusterId": cluster_id, "devices": [], "pagination": {"total": 0}}

    labels = await run_in_executor(clusterer.predict, fleet_df)

    mask = labels == cluster_id
    device_ids = fleet_df.index[mask].tolist()

    total = len(device_ids)
    paged = device_ids[offset : offset + limit]

    return {
        "clusterId": cluster_id,
        "devices": [{"deviceId": did} for did in paged],
        "pagination": PaginationMeta(total=total, limit=limit, offset=offset).model_dump(),
    }
