"""Scheduled model retraining using APScheduler."""

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from netpulse_ml.training.pipeline import (
    train_anomaly_detector,
    train_churn_predictor,
    train_fleet_clusterer,
)

log = structlog.get_logger()


def create_training_scheduler() -> AsyncIOScheduler:
    """Create and configure the training scheduler.

    Schedule:
      - Anomaly detector: nightly at 02:00 UTC
      - Churn predictor: weekly Sunday at 03:00 UTC
      - Fleet clusterer: nightly at 04:00 UTC
    """
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        train_anomaly_detector,
        trigger="cron",
        hour=2,
        minute=0,
        id="train_anomaly_detector",
        name="Nightly anomaly detector training",
        replace_existing=True,
    )

    scheduler.add_job(
        train_churn_predictor,
        trigger="cron",
        day_of_week="sun",
        hour=3,
        minute=0,
        id="train_churn_predictor",
        name="Weekly churn predictor training",
        replace_existing=True,
    )

    scheduler.add_job(
        train_fleet_clusterer,
        trigger="cron",
        hour=4,
        minute=0,
        id="train_fleet_clusterer",
        name="Nightly fleet clustering",
        replace_existing=True,
    )

    log.info("Training scheduler configured", jobs=3)
    return scheduler
