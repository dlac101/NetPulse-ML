"""Agent orchestrator: fleet scanner + agent lifecycle manager.

Runs on APScheduler, scanning the fleet for anomalous devices and
running the LangGraph remediation workflow for each flagged device.
"""

import uuid
from datetime import datetime, timedelta, timezone

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from netpulse_ml.agents.graph import build_remediation_graph
from netpulse_ml.agents.state import RemediationState
from netpulse_ml.config import Settings
from netpulse_ml.db.engine import async_session_factory
from netpulse_ml.db.models import AgentExecution
from netpulse_ml.dependencies import run_in_executor
from netpulse_ml.features.store import feature_store
from netpulse_ml.serving.predictor import Predictor

log = structlog.get_logger()


class AgentOrchestrator:
    """Manages the remediation agent lifecycle: scanning, running, and logging."""

    def __init__(self, settings: Settings, predictor: Predictor) -> None:
        self._settings = settings
        self._predictor = predictor
        self._graph = build_remediation_graph(predictor)
        self._scheduler: AsyncIOScheduler | None = None
        self._cooldowns: dict[str, datetime] = {}  # device_id -> last_run_time
        self._last_scan_at: datetime | None = None
        self._last_scan_flagged: int = 0
        self._total_runs: int = 0
        self._scan_lock = asyncio.Lock()  # Prevents concurrent fleet scans

    @property
    def status(self) -> dict:
        return {
            "isRunning": self._scan_lock.locked(),
            "lastScanAt": self._last_scan_at.isoformat() if self._last_scan_at else None,
            "lastScanFlagged": self._last_scan_flagged,
            "totalRuns": self._total_runs,
            "cooldownDevices": len(self._cooldowns),
            "scanIntervalMinutes": self._settings.agent_scan_interval_minutes,
            "anomalyThreshold": self._settings.agent_anomaly_threshold,
            "autoExecuteEnabled": self._settings.agent_enable_auto_execute,
        }

    def start_scheduler(self) -> None:
        """Start the periodic fleet scan scheduler."""
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self.scan_fleet,
            trigger="interval",
            minutes=self._settings.agent_scan_interval_minutes,
            id="agent_fleet_scan",
            name="Agent fleet scan",
            replace_existing=True,
        )
        self._scheduler.start()
        log.info("Agent scheduler started", interval_min=self._settings.agent_scan_interval_minutes)

    def stop_scheduler(self) -> None:
        """Stop the scheduler gracefully."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            log.info("Agent scheduler stopped")

    async def scan_fleet(self) -> int:
        """Scan fleet for anomalous devices and run remediation agent for each.

        Returns the number of devices processed.
        """
        if self._scan_lock.locked():
            log.warning("Fleet scan already in progress, skipping")
            return 0

        async with self._scan_lock:
            return await self._do_scan()

    async def _do_scan(self) -> int:
        """Internal scan implementation (called under lock)."""
        self._last_scan_at = datetime.now(timezone.utc)
        processed = 0

        # Prune expired cooldowns to prevent memory leak
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._settings.agent_cooldown_hours)
        self._cooldowns = {k: v for k, v in self._cooldowns.items() if v > cutoff}

        try:
            # Get fleet features
            fleet_df = await feature_store.get_fleet_features(limit=1000)
            if fleet_df.empty:
                log.info("Fleet scan: no feature data available")
                return 0

            # Batch anomaly scoring (offload to thread pool)
            detector = self._predictor.anomaly_detector
            if not detector.is_fitted:
                log.warning("Fleet scan: anomaly detector not trained, skipping")
                return 0

            scores = await run_in_executor(detector.predict, fleet_df)

            # Filter devices above threshold
            threshold = self._settings.agent_anomaly_threshold
            flagged_devices = []
            for i, (device_id, _) in enumerate(fleet_df.iterrows()):
                if float(scores[i]) >= threshold:
                    flagged_devices.append((str(device_id), float(scores[i])))

            self._last_scan_flagged = len(flagged_devices)
            log.info("Fleet scan complete", total=len(fleet_df), flagged=len(flagged_devices))

            # Run agent for each flagged device (sequential to limit load)
            for device_id, score in flagged_devices:
                if self._is_on_cooldown(device_id):
                    continue
                await self.run_for_device(device_id)
                processed += 1

        except Exception as e:
            log.error("Fleet scan failed", error=str(e))

        return processed

    async def run_for_device(self, device_id: str) -> dict:
        """Run the remediation agent for a single device.

        Returns the final agent state.
        """
        execution_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        log.info("Agent run starting", device_id=device_id, execution_id=execution_id)

        initial_state: RemediationState = {
            "device_id": device_id,
            "status": "analyzing",
        }

        try:
            # Run the LangGraph with timeout (5 min max)
            final_state = await asyncio.wait_for(
                self._graph.ainvoke(initial_state), timeout=300
            )
            status = final_state.get("status", "unknown")
        except asyncio.TimeoutError:
            log.error("Agent run timed out", device_id=device_id)
            final_state = {**initial_state, "status": "failed", "error": "timeout"}
            status = "failed"
        except Exception as e:
            log.error("Agent run failed", device_id=device_id, error=str(e))
            final_state = {**initial_state, "status": "failed", "error": str(e)}
            status = "failed"

        completed_at = datetime.now(timezone.utc)

        # Record execution
        await self._log_execution(
            execution_id=execution_id,
            device_id=device_id,
            started_at=started_at,
            completed_at=completed_at,
            state=final_state,
        )

        # Set cooldown
        self._cooldowns[device_id] = completed_at
        self._total_runs += 1

        log.info(
            "Agent run complete",
            device_id=device_id,
            status=status,
            diagnosis=final_state.get("diagnosis"),
            action=final_state.get("recommended_action"),
        )

        return final_state

    def _is_on_cooldown(self, device_id: str) -> bool:
        """Check if a device is within the cooldown window."""
        last_run = self._cooldowns.get(device_id)
        if last_run is None:
            return False
        cooldown = timedelta(hours=self._settings.agent_cooldown_hours)
        return datetime.now(timezone.utc) - last_run < cooldown

    async def _log_execution(
        self,
        execution_id: str,
        device_id: str,
        started_at: datetime,
        completed_at: datetime,
        state: dict,
    ) -> None:
        """Persist agent execution record to the database."""
        try:
            async with async_session_factory() as session:
                execution = AgentExecution(
                    id=execution_id,
                    device_id=device_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    status=state.get("status", "unknown"),
                    diagnosis=state.get("diagnosis", ""),
                    recommended_action=state.get("recommended_action"),
                    auto_executed=state.get("status") == "executed",
                    execution_result=state.get("execution_result"),
                    verified=state.get("verified", False),
                    recommendation_id=state.get("recommendation_id"),
                )
                session.add(execution)
                await session.commit()
        except Exception as e:
            log.error("Failed to log agent execution", error=str(e))
