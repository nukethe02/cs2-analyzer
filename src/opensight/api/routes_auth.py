"""
Authentication route handlers (placeholder — backed by in-memory storage until Wave 5).

Endpoints:
- POST /auth/register — register a new user, returns JWT
- POST /auth/login — login with email/password, returns JWT
- GET /auth/me — return current user from token
"""

import logging
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

# ---------------------------------------------------------------------------
# In-memory user store (replaced by DB User model in Wave 5)
# ---------------------------------------------------------------------------
_users_lock = threading.Lock()
_users: dict[str, dict[str, Any]] = {}  # email -> {id, email, password_hash, tier}
_next_id = 1


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/auth/register")
async def register(body: RegisterRequest) -> dict[str, Any]:
    """Register a new user and return a JWT."""
    global _next_id

    try:
        from opensight.auth.jwt import create_access_token
        from opensight.auth.passwords import hash_password
    except ImportError as e:
        logger.warning("Auth module not available: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Auth not available. Required dependencies not installed.",
        ) from e

    email = body.email.lower().strip()

    with _users_lock:
        if email in _users:
            raise HTTPException(status_code=409, detail="Email already registered")

        user_id = _next_id
        _next_id += 1
        _users[email] = {
            "id": user_id,
            "email": email,
            "password_hash": hash_password(body.password),
            "tier": "free",
        }

    try:
        token = create_access_token(user_id, email)
    except RuntimeError as e:
        logger.warning("Auth not configured: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Auth not configured. Check server configuration.",
        ) from e
    logger.info(f"User registered: {email} (id={user_id})")
    return {"token": token, "user_id": user_id, "email": email, "tier": "free"}


@router.post("/auth/login")
async def login(body: LoginRequest) -> dict[str, Any]:
    """Authenticate with email/password and return a JWT."""
    try:
        from opensight.auth.jwt import create_access_token
        from opensight.auth.passwords import verify_password
    except ImportError as e:
        logger.warning("Auth module not available: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Auth not available. Required dependencies not installed.",
        ) from e

    email = body.email.lower().strip()

    with _users_lock:
        user = _users.get(email)

    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    try:
        token = create_access_token(user["id"], email)
    except RuntimeError as e:
        logger.warning("Auth not configured: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Auth not configured. Check server configuration.",
        ) from e
    return {"token": token, "user_id": user["id"], "email": email, "tier": user["tier"]}


@router.get("/auth/me")
async def get_me(request: Request) -> dict[str, Any]:
    """Return the current user's profile from their JWT."""
    try:
        from opensight.auth.middleware import get_current_user
    except ImportError as e:
        logger.warning("Auth module not available: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Auth not available. Required dependencies not installed.",
        ) from e

    user = await get_current_user(request)

    with _users_lock:
        stored = _users.get(user["email"])

    tier = stored["tier"] if stored else "free"
    return {"user_id": user["user_id"], "email": user["email"], "tier": tier}
