"""FastAPI authentication middleware — extract and verify JWT from requests."""

from fastapi import HTTPException, Request


async def get_current_user_optional(request: Request) -> dict | None:
    """Extract user from Bearer token if present, return None otherwise."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        from opensight.auth.jwt import decode_token

        return decode_token(auth[7:])
    except (ValueError, RuntimeError):
        return None


async def get_current_user(request: Request) -> dict:
    """Extract user from Bearer token. Raises 401 if missing or invalid."""
    user = await get_current_user_optional(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user
