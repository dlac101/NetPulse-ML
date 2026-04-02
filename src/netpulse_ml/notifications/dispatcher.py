"""Notification dispatcher: routes events to configured channels based on severity."""

import structlog

from netpulse_ml.config import settings
from netpulse_ml.notifications.channels import send_email, send_slack, send_webhook

log = structlog.get_logger()

# Severity -> notification behavior
# critical: email + slack + webhook
# warning:  email + slack
# info:     slack only
SEVERITY_CHANNELS = {
    "critical": ["email", "slack", "webhook"],
    "high":     ["email", "slack"],
    "medium":   ["slack"],
    "low":      [],
}


def _format_email_html(
    device_id: str,
    title: str,
    description: str,
    diagnosis: str,
    confidence: float,
    impact: str,
) -> str:
    """Format an HTML email body for an agent escalation."""
    impact_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}.get(impact, "#8899b0")

    return f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto; background: #0b0f19; color: #e8ecf4; padding: 24px; border-radius: 8px;">
        <div style="border-left: 3px solid #00C8E6; padding-left: 12px; margin-bottom: 16px;">
            <h2 style="margin: 0; font-size: 16px; color: #00C8E6;">NetPulse Agent Alert</h2>
        </div>
        <p style="font-size: 14px; color: #8899b0;">The remediation agent has escalated an issue requiring human review.</p>
        <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
            <tr><td style="padding: 8px 0; color: #5a6a80; font-size: 12px;">Device</td><td style="padding: 8px 0; font-family: monospace; color: #e8ecf4;">{device_id}</td></tr>
            <tr><td style="padding: 8px 0; color: #5a6a80; font-size: 12px;">Diagnosis</td><td style="padding: 8px 0; color: #e8ecf4;">{diagnosis}</td></tr>
            <tr><td style="padding: 8px 0; color: #5a6a80; font-size: 12px;">Recommendation</td><td style="padding: 8px 0; color: #e8ecf4; font-weight: 600;">{title}</td></tr>
            <tr><td style="padding: 8px 0; color: #5a6a80; font-size: 12px;">Confidence</td><td style="padding: 8px 0; font-family: monospace; color: #00C8E6;">{confidence:.0%}</td></tr>
            <tr><td style="padding: 8px 0; color: #5a6a80; font-size: 12px;">Impact</td><td style="padding: 8px 0;"><span style="background: {impact_color}22; color: {impact_color}; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase;">{impact}</span></td></tr>
        </table>
        <p style="font-size: 13px; color: #8899b0;">{description}</p>
        <p style="font-size: 11px; color: #5a6a80; margin-top: 16px;">Review and approve this recommendation in the NetPulse Admin Console.</p>
    </div>
    """


def _format_slack_message(
    device_id: str,
    title: str,
    diagnosis: str,
    confidence: float,
    impact: str,
) -> str:
    """Format a Slack message for an agent escalation."""
    emoji = {"high": ":red_circle:", "medium": ":large_orange_circle:", "low": ":white_circle:"}.get(impact, ":white_circle:")
    return (
        f"{emoji} *NetPulse Agent Alert*\n"
        f"Device `{device_id}` needs attention.\n"
        f"*Diagnosis:* {diagnosis}\n"
        f"*Recommendation:* {title}\n"
        f"*Confidence:* {confidence:.0%} | *Impact:* {impact.upper()}\n"
        f"Review in the Admin Console."
    )


async def notify_escalation(
    device_id: str,
    title: str,
    description: str,
    diagnosis: str,
    confidence: float,
    impact: str,
    recommendation_id: str,
) -> dict[str, bool]:
    """Dispatch notifications for an agent escalation event.

    Sends to email, Slack, and/or webhook based on impact severity.
    Returns dict of channel -> success/failure.
    """
    if not settings.notifications_enabled:
        log.debug("Notifications disabled globally")
        return {}

    channels = SEVERITY_CHANNELS.get(impact, [])
    results: dict[str, bool] = {}

    if "email" in channels:
        html = _format_email_html(device_id, title, description, diagnosis, confidence, impact)
        results["email"] = await send_email(
            subject=f"[NetPulse] Agent Alert: {title} ({device_id})",
            body_html=html,
        )

    if "slack" in channels:
        msg = _format_slack_message(device_id, title, diagnosis, confidence, impact)
        results["slack"] = await send_slack(msg)

    if "webhook" in channels:
        results["webhook"] = await send_webhook({
            "event": "agent_escalation",
            "device_id": device_id,
            "recommendation_id": recommendation_id,
            "title": title,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "impact": impact,
        })

    log.info("Escalation notifications dispatched", device_id=device_id, results=results)
    return results
