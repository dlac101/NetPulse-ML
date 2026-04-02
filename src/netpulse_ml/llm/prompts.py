"""Prompt templates for fleet insights, device diagnosis, and technician Q&A."""

SYSTEM_PROMPT = (
    "You are an expert ISP network technician assistant for SmartOS router fleets. "
    "You analyze telemetry data (QoE scores, speed tests, WiFi metrics, DPI traffic, events) "
    "to diagnose issues and explain them clearly. Be concise and actionable. "
    "When referencing metrics, include the actual values."
)

FLEET_INSIGHT_PROMPT = """Summarize the fleet health for the past 24 hours based on the following telemetry data.
Include: overall health assessment, top issues, devices needing attention, and recommended actions.
Keep it under 200 words.

Fleet Data:
{context}

Summary:"""

DEVICE_DIAGNOSIS_PROMPT = """Explain why device {device_id} is showing performance issues.

Device Metrics:
- Anomaly Score: {anomaly_score}/1.0 (higher = more anomalous)
- Churn Risk: {churn_risk}/100
- QoE Score: {qoe_score}/100
- Download: {dl_mbps} Mbps
- Upload: {ul_mbps} Mbps
- Latency: {latency_ms} ms

Top Contributing Anomaly Features:
{top_features}

Relevant Context:
{context}

Provide a clear diagnosis in 2-3 sentences, then list 1-3 specific recommended actions:"""

TECHNICIAN_QA_PROMPT = """Answer the technician's question using the provided context.
If the context doesn't contain enough information, say so.
Be specific and reference actual device IDs, metrics, and values when available.

Context:
{context}

Question: {question}

Answer:"""


def format_fleet_insight(context: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for fleet insight generation."""
    return SYSTEM_PROMPT, FLEET_INSIGHT_PROMPT.format(context=context)


def format_device_diagnosis(
    device_id: str,
    anomaly_score: float,
    churn_risk: float,
    qoe_score: float,
    dl_mbps: float,
    ul_mbps: float,
    latency_ms: float,
    top_features: str,
    context: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for device diagnosis."""
    prompt = DEVICE_DIAGNOSIS_PROMPT.format(
        device_id=device_id,
        anomaly_score=f"{anomaly_score:.2f}",
        churn_risk=f"{churn_risk:.0f}",
        qoe_score=f"{qoe_score:.0f}",
        dl_mbps=f"{dl_mbps:.0f}",
        ul_mbps=f"{ul_mbps:.0f}",
        latency_ms=f"{latency_ms:.1f}",
        top_features=top_features,
        context=context,
    )
    return SYSTEM_PROMPT, prompt


def format_technician_qa(question: str, context: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for technician Q&A."""
    prompt = TECHNICIAN_QA_PROMPT.format(context=context, question=question)
    return SYSTEM_PROMPT, prompt
