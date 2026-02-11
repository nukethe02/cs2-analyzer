"""Paddle webhook endpoint for subscription management."""
import hashlib
import hmac
import json
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from sqlalchemy import select

from opensight.auth.tiers import Tier
from opensight.infra.database import SessionLocal, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/paddle", tags=["paddle"])

# Set this via environment variable PADDLE_WEBHOOK_SECRET
PADDLE_WEBHOOK_SECRET: Optional[str] = None

try:
    import os
    PADDLE_WEBHOOK_SECRET = os.environ.get("PADDLE_WEBHOOK_SECRET")
except Exception:
    pass


def verify_paddle_signature(raw_body: bytes, signature: str) -> bool:
    """Verify Paddle webhook signature using HMAC-SHA256."""
    if not PADDLE_WEBHOOK_SECRET:
        logger.warning("PADDLE_WEBHOOK_SECRET not set -- skipping verification")
        return True

    try:
        # Paddle v2 signature format: ts=TIMESTAMP;h1=HASH
        parts = dict(p.split("=", 1) for p in signature.split(";"))
        ts = parts.get("ts", "")
        h1 = parts.get("h1", "")

        # Reconstruct signed payload
        signed_payload = f"{ts}:{raw_body.decode('utf-8')}"
        expected = hmac.new(
            PADDLE_WEBHOOK_SECRET.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(h1, expected)
    except Exception as e:
        logger.error("Signature verification failed: %s", e)
        return False


PLAN_TO_TIER = {
    "pri_PLACEHOLDER_PLUS": Tier.PLUS,
    "pri_PLACEHOLDER_PRO": Tier.PRO,
    # Replace with real Paddle price IDs when ready
}


def _resolve_tier(price_id: str) -> Tier:
    """Map a Paddle price ID to an OpenSight tier."""
    return PLAN_TO_TIER.get(price_id, Tier.FREE)


@router.post("/webhook")
async def paddle_webhook(
    request: Request,
    paddle_signature: Optional[str] = Header(None, alias="Paddle-Signature"),
):
    """
    Handle Paddle subscription lifecycle events.

    Events handled:
      - subscription.created  -> upgrade user tier
      - subscription.updated  -> change tier if plan changed
      - subscription.canceled -> downgrade to FREE
      - subscription.paused   -> downgrade to FREE
      - subscription.resumed  -> restore tier
    """
    raw_body = await request.body()

    # Verify signature in production
    if PADDLE_WEBHOOK_SECRET and paddle_signature:
        if not verify_paddle_signature(raw_body, paddle_signature):
            raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = payload.get("event_type", "")
    data = payload.get("data", {})

    logger.info("Paddle event: %s", event_type)

    # Extract custom_data.steam_id (set during checkout)
    custom_data = data.get("custom_data", {}) or {}
    steam_id = custom_data.get("steam_id")

    if not steam_id:
        # Try passthrough field (Paddle v1 compat)
        passthrough = data.get("passthrough")
        if passthrough:
            try:
                pt = json.loads(passthrough)
                steam_id = pt.get("steam_id")
            except (json.JSONDecodeError, AttributeError):
                pass

    if not steam_id:
        logger.warning("No steam_id in webhook payload -- ignoring")
        return {"status": "ok", "detail": "no steam_id"}

    # Determine new tier based on event
    new_tier: Optional[Tier] = None

    if event_type in ("subscription.created", "subscription.updated", "subscription.resumed"):
        items = data.get("items", [])
        if items:
            price_id = items[0].get("price", {}).get("id", "")
            new_tier = _resolve_tier(price_id)
        else:
            new_tier = Tier.PLUS  # Default upgrade

    elif event_type in ("subscription.canceled", "subscription.paused"):
        new_tier = Tier.FREE

    else:
        logger.info("Unhandled event type: %s", event_type)
        return {"status": "ok", "detail": f"ignored {event_type}"}

    # Update user tier in database
    try:
        with SessionLocal() as session:
            stmt = select(User).where(User.steam_id == steam_id)
            user = session.execute(stmt).scalar_one_or_none()

            if user is None:
                logger.warning("User not found for steam_id=%s", steam_id)
                return {"status": "ok", "detail": "user not found"}

            old_tier = user.tier
            user.tier = new_tier.value
            session.commit()
            logger.info(
                "Updated user %s tier: %s -> %s",
                steam_id, old_tier, new_tier.value,
            )
    except Exception as e:
        logger.error("Database error updating tier: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")

    return {"status": "ok", "event": event_type, "tier": new_tier.value}
