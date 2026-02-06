"""JWT token creation and verification using python-jose."""

import os
from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt

SECRET = os.getenv("JWT_SECRET", "")
ALGORITHM = "HS256"
EXPIRY_HOURS = 24


def create_access_token(user_id: int, email: str) -> str:
    """Create a signed JWT access token."""
    if not SECRET:
        raise RuntimeError("JWT_SECRET environment variable not set")
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(UTC) + timedelta(hours=EXPIRY_HOURS),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT token. Raises ValueError on failure."""
    if not SECRET:
        raise RuntimeError("JWT_SECRET environment variable not set")
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}") from e
