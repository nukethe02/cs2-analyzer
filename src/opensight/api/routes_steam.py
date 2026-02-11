"""
Steam OpenID authentication routes for OpenSight CS2 Analyzer.

Endpoints:
    GET /auth/steam/login    - Redirect user to Steam login page
    GET /auth/steam/callback - Handle Steam's redirect, create/update user, return JWT
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from opensight.auth.steam import build_auth_url, validate_response, get_player_summary
from opensight.auth.jwt import create_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/steam", tags=["auth-steam"])


def _get_callback_url(request: Request) -> str:
    """Build the callback URL for Steam to redirect back to.

    Uses X-Forwarded-Proto and X-Forwarded-Host headers if behind a proxy
    (e.g., HuggingFace Spaces, nginx, Cloudflare).
    """
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)

    # Fallback for HuggingFace Spaces
    if not host or host == "":
        host = request.headers.get("host", "localhost:7860")

    return f"{proto}://{host}/auth/steam/callback"


@router.get("/login")
async def steam_login(request: Request):
    """Redirect user to Steam's OpenID login page.

    The user will be taken to Steam to authenticate, then redirected
    back to /auth/steam/callback with their Steam identity.
    """
    callback_url = _get_callback_url(request)
    auth_url = build_auth_url(callback_url)
    logger.info(f"Redirecting to Steam login, callback: {callback_url}")
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/callback")
async def steam_callback(request: Request):
    """Handle Steam's OpenID callback after user authenticates.

    Flow:
    1. Validate the OpenID response from Steam
    2. Extract Steam64 ID
    3. Create or update user in database
    4. Fetch Steam profile (avatar, name) if API key is set
    5. Issue JWT token
    6. Redirect to frontend with token
    """
    # Get all query params from Steam's redirect
    params = dict(request.query_params)

    # Validate the OpenID response
    steam_id = await validate_response(params)

    if not steam_id:
        logger.warning("Steam OpenID validation failed")
        # Redirect to frontend with error
        return RedirectResponse(
            url="/?auth_error=steam_validation_failed",
            status_code=status.HTTP_302_FOUND,
        )

    logger.info(f"Steam login successful for Steam64 ID: {steam_id}")

    # Try to fetch Steam profile for display name and avatar
    profile = await get_player_summary(steam_id)
    display_name = profile.get("display_name", f"Player_{steam_id[-4:]}") if profile else f"Player_{steam_id[-4:]}"
    avatar_url = profile.get("avatar_url", "") if profile else ""

    # Create or update user in database
    try:
        from opensight.infra.database import get_session_context, get_user_by_steam_id, create_user, User
        from sqlalchemy import update

        with get_session_context() as db:
            user = get_user_by_steam_id(db, steam_id)

            if user is None:
                # New user — create account linked to Steam
                user = create_user(
                    db=db,
                    email=f"{steam_id}@steam.opensight.local",  # Placeholder email for Steam users
                    username=display_name,
                    password_hash="STEAM_OPENID_NO_PASSWORD",  # No password for Steam users
                    tier="free",
                    steam_id=steam_id,
                )
                if user is None:
                    # Username might be taken, try with Steam ID suffix
                    user = create_user(
                        db=db,
                        email=f"{steam_id}@steam.opensight.local",
                        username=f"{display_name}_{steam_id[-4:]}",
                        password_hash="STEAM_OPENID_NO_PASSWORD",
                        tier="free",
                        steam_id=steam_id,
                    )

                if user is None:
                    logger.error(f"Failed to create user for Steam ID {steam_id}")
                    return RedirectResponse(
                        url="/?auth_error=user_creation_failed",
                        status_code=status.HTTP_302_FOUND,
                    )
                logger.info(f"Created new user {user.id} for Steam ID {steam_id}")
            else:
                # Existing user — update last login
                from opensight.infra.database import update_user_last_login
                update_user_last_login(db, user.id)
                logger.info(f"Existing user {user.id} logged in via Steam")

            # Generate JWT token
            token = create_access_token({
                "user_id": user.id,
                "email": user.email,
                "steam_id": steam_id,
            })

    except Exception as e:
        logger.error(f"Database error during Steam auth: {e}")
        return RedirectResponse(
            url="/?auth_error=database_error",
            status_code=status.HTTP_302_FOUND,
        )

    # Redirect to frontend with JWT token
    # The frontend JavaScript will extract the token from the URL fragment
    # and store it in localStorage
    redirect_params = urlencode({
        "token": token,
        "steam_id": steam_id,
        "username": display_name,
        "avatar": avatar_url,
    })

    return RedirectResponse(
        url=f"/?auth_success=1&{redirect_params}",
        status_code=status.HTTP_302_FOUND,
    )


@router.get("/status")
async def steam_auth_status():
    """Check if Steam authentication is configured and available."""
    has_api_key = bool(os.environ.get("STEAM_API_KEY", ""))
    return {
        "steam_auth_enabled": True,
        "steam_profile_enrichment": has_api_key,
        "login_url": "/auth/steam/login",
    }
