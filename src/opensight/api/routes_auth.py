"""
Authentication route handlers — backed by SQLAlchemy User model.

Endpoints:
- POST /auth/register — register a new user, returns JWT
- POST /auth/login — login with email/password, returns JWT
- GET  /auth/me — return current user from token
- POST /auth/verify-token — verify JWT validity
- GET  /auth/health — auth service health check
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from opensight.auth.jwt import create_access_token, decode_token
from opensight.auth.passwords import hash_password, verify_password
from opensight.infra.database import (
    User,
    get_session,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
    create_user,
    update_user_last_login,
    user_exists,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str = Field(..., description="Email address")
    username: str = Field(..., min_length=3, max_length=100, description="Username")
    password: str = Field(..., min_length=8, description="Password (min 8 chars)")


class LoginRequest(BaseModel):
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    tier: str
    steam_id: str | None = None
    created_at: str | None = None
    last_login: str | None = None


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_current_user(
    request: Request,
    db: Session = Depends(get_session),
) -> User:
    """Extract and validate current user from JWT in Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_str = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Register a new user account."""
    if user_exists(db, email=body.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    if user_exists(db, username=body.username):
        raise HTTPException(status_code=409, detail="Username already taken")

    pw_hash = hash_password(body.password)
    user = create_user(
        db=db,
        email=body.email,
        username=body.username,
        password_hash=pw_hash,
        tier="free",
    )
    if not user:
        raise HTTPException(status_code=500, detail="Error creating user account")

    try:
        token = create_access_token(user.id, user.email)
    except RuntimeError as e:
        logger.warning("Auth not configured: %s", e)
        raise HTTPException(status_code=503, detail="Auth not configured") from e

    logger.info("User registered: %s (id=%d)", user.email, user.id)
    return {
        "token": token,
        "user": UserResponse(**user.to_dict()).model_dump(),
    }


@router.post("/auth/login")
async def login(
    body: LoginRequest,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Authenticate with email/password and return a JWT."""
    user = get_user_by_email(db, body.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    update_user_last_login(db, user.id)

    try:
        token = create_access_token(user.id, user.email)
    except RuntimeError as e:
        logger.warning("Auth not configured: %s", e)
        raise HTTPException(status_code=503, detail="Auth not configured") from e

    logger.info("User logged in: %s (id=%d)", user.email, user.id)
    return {
        "token": token,
        "user": UserResponse(**user.to_dict()).model_dump(),
    }


@router.get("/auth/me")
async def me(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    """Return current authenticated user info."""
    return {"user": UserResponse(**current_user.to_dict()).model_dump()}


@router.post("/auth/verify-token")
async def verify_token(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Verify that the provided JWT token is valid."""
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
    }


@router.get("/auth/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for auth service."""
    return {"status": "ok", "service": "auth"}
