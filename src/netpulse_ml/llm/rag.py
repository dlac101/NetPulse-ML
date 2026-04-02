"""RAG pipeline: embed query -> retrieve context -> generate response."""

from collections.abc import AsyncGenerator

import structlog

from netpulse_ml.config import settings
from netpulse_ml.features.store import feature_store
from netpulse_ml.llm.embedder import Embedder
from netpulse_ml.llm.prompts import (
    format_device_diagnosis,
    format_fleet_insight,
    format_technician_qa,
)
from netpulse_ml.llm.provider import OllamaProvider
from netpulse_ml.llm.vector_store import VectorStore
from netpulse_ml.serving.predictor import Predictor

log = structlog.get_logger()


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for fleet/device insights."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        provider: OllamaProvider,
        predictor: Predictor,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._provider = provider
        self._predictor = predictor

    async def _retrieve_context(self, query: str, top_k: int | None = None) -> str:
        """Embed query and retrieve relevant documents."""
        k = top_k or settings.rag_top_k
        query_embedding = await self._embedder.aembed_text(query)
        docs = await self._vector_store.search(query_embedding, top_k=k)

        if not docs:
            return "No relevant documentation found."

        context_parts = []
        for doc in docs:
            source = doc["metadata"].get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{doc['content']}")

        return "\n\n---\n\n".join(context_parts)

    async def fleet_insight(self) -> str:
        """Generate a fleet health summary using RAG."""
        # Get fleet-level context
        fleet_df = await feature_store.get_fleet_features(limit=100)

        if fleet_df.empty:
            return "No fleet data available for analysis."

        # Build summary text from fleet features
        online_count = len(fleet_df)
        avg_qoe = fleet_df.get("qoe_composite_latest", []).mean() if "qoe_composite_latest" in fleet_df.columns else 0
        avg_dl = fleet_df.get("dl_mbps_latest", []).mean() if "dl_mbps_latest" in fleet_df.columns else 0

        fleet_text = (
            f"Fleet size: {online_count} devices. "
            f"Average QoE: {avg_qoe:.1f}/100. "
            f"Average download: {avg_dl:.0f} Mbps."
        )

        # Retrieve relevant docs
        context = await self._retrieve_context("fleet health summary QoE performance")
        full_context = f"{fleet_text}\n\n{context}"

        system, prompt = format_fleet_insight(full_context)
        return await self._provider.generate(prompt, system=system)

    async def device_diagnosis(self, device_id: str) -> str:
        """Generate a diagnosis for a specific device using RAG."""
        features = await feature_store.get_latest_features(device_id)
        if not features:
            return f"No telemetry data available for device {device_id}."

        # Get ML predictions
        anomaly_score = 0.0
        churn_risk = 0.0
        top_features_text = "No anomaly data available."

        if self._predictor.anomaly_detector.is_fitted:
            anomaly_score = self._predictor.anomaly_detector.predict_single(features)
            top_feats = self._predictor.anomaly_detector.get_top_features(features, n_top=5)
            top_features_text = "\n".join(
                f"- {f['name']}: value={f['value']:.2f}, z-score={f['zscore']:.2f}"
                for f in top_feats
            )

        if self._predictor.churn_predictor.is_fitted:
            churn_result = self._predictor.churn_predictor.predict_single(features)
            churn_risk = churn_result.get("riskScore", 0.0)

        # Retrieve relevant documentation
        context = await self._retrieve_context(
            f"device {device_id} QoE anomaly troubleshooting WiFi bufferbloat"
        )

        system, prompt = format_device_diagnosis(
            device_id=device_id,
            anomaly_score=anomaly_score,
            churn_risk=churn_risk,
            qoe_score=features.get("qoe_composite_latest", 0),
            dl_mbps=features.get("dl_mbps_latest", 0),
            ul_mbps=features.get("ul_mbps_latest", 0),
            latency_ms=features.get("latency_idle_ms", 0),
            top_features=top_features_text,
            context=context,
        )

        return await self._provider.generate(prompt, system=system)

    async def chat(self, message: str) -> str:
        """Answer a technician question using RAG (non-streaming)."""
        context = await self._retrieve_context(message)
        system, prompt = format_technician_qa(message, context)
        return await self._provider.generate(prompt, system=system, temperature=0.5)

    async def chat_stream(self, message: str) -> AsyncGenerator[str, None]:
        """Answer a technician question using RAG (streaming tokens)."""
        context = await self._retrieve_context(message)
        system, prompt = format_technician_qa(message, context)
        async for token in self._provider.generate_stream(prompt, system=system, temperature=0.5):
            yield token
