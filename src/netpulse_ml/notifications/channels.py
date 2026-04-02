"""Notification channel implementations: email (SMTP) and Slack (webhook)."""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx
import structlog

from netpulse_ml.config import settings

log = structlog.get_logger()


async def send_email(
    subject: str,
    body_html: str,
    to: str | None = None,
) -> bool:
    """Send an email notification via SMTP.

    Returns True on success, False on failure (non-fatal).
    """
    if not settings.smtp_host:
        log.debug("Email notification skipped (SMTP not configured)")
        return False

    recipient = to or settings.notify_email_to
    if not recipient:
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.smtp_from
        msg["To"] = recipient
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
            if settings.smtp_tls:
                server.starttls()
            if settings.smtp_username:
                server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(msg)

        log.info("Email notification sent", to=recipient, subject=subject)
        return True
    except Exception as e:
        log.warning("Email notification failed", error=str(e), to=recipient)
        return False


async def send_slack(
    message: str,
    webhook_url: str | None = None,
) -> bool:
    """Send a Slack notification via incoming webhook.

    Returns True on success, False on failure (non-fatal).
    """
    url = webhook_url or settings.slack_webhook_url
    if not url:
        log.debug("Slack notification skipped (webhook not configured)")
        return False

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json={"text": message})
            response.raise_for_status()

        log.info("Slack notification sent")
        return True
    except Exception as e:
        log.warning("Slack notification failed", error=str(e))
        return False


async def send_webhook(
    payload: dict,
    webhook_url: str | None = None,
) -> bool:
    """Send a generic webhook notification (e.g., PagerDuty).

    Returns True on success, False on failure (non-fatal).
    """
    url = webhook_url or settings.notify_webhook_url
    if not url:
        log.debug("Webhook notification skipped (URL not configured)")
        return False

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        log.info("Webhook notification sent", url=url)
        return True
    except Exception as e:
        log.warning("Webhook notification failed", error=str(e), url=url)
        return False
