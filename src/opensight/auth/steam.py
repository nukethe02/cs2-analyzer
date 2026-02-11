"""
Steam OpenID 2.0 Authentication for OpenSight CS2 Analyzer.

Steam uses OpenID 2.0 (NOT OAuth2). The flow is:
1. User clicks "Login with Steam"
2. Redirect to steamcommunity.com/openid/login
3. Steam redirects back with user's Steam64 ID in URL
4. We create/update user in SQLite DB
5. Issue JWT token

No API key required for basic auth.
Optional STEAM_API_KEY for player profile data (avatar, display name).
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlencode, parse_qs, urlparse

import httpx

logger = logging.getLogger(__name__)

# Steam OpenID 2.0 constants
STEAM_OPENID_URL = "https://steamcommunity.com/openid/login"
STEAM_OPENID_NS = "http://specs.openid.net/auth/2.0"
STEAM_CLAIMED_ID_PATTERN = re.compile(
    r"^https://steamcommunity\.com/openid/id/(\d{17})$"
)

# Steam Web API (optional, for profile data)
STEAM_API_BASE = "https://api.steampowered.com"


def build_auth_url(return_url: str) -> str:
    """Build the Steam OpenID login redirect URL.

    Args:
        return_url: The callback URL Steam will redirect to after auth.
                    Must be HTTPS in production.

    Returns:
        Full Steam OpenID URL to redirect the user to.
    """
    params = {
        "openid.ns": STEAM_OPENID_NS,
        "openid.mode": "checkid_setup",
        "openid.return_to": return_url,
        "openid.realm": _get_realm(return_url),
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
    }
    return f"{STEAM_OPENID_URL}?{urlencode(params)}"


async def validate_response(params: dict) -> str | None:
    """Validate Steam's OpenID response and extract Steam64 ID.

    This performs the check_authentication step to verify the response
    is genuinely from Steam (prevents replay/forgery attacks).

    Args:
        params: The query parameters from Steam's callback redirect.

    Returns:
        Steam64 ID (17-digit string starting with 7656...) or None if invalid.
    """
    # Check mode is id_res (positive assertion)
    if params.get("openid.mode") != "id_res":
        logger.warning("Steam OpenID: mode is not id_res")
        return None

    # Extract claimed_id and validate format
    claimed_id = params.get("openid.claimed_id", "")
    match = STEAM_CLAIMED_ID_PATTERN.match(claimed_id)
    if not match:
        logger.warning(f"Steam OpenID: invalid claimed_id format: {claimed_id}")
        return None

    steam_id = match.group(1)

    # Build verification request (check_authentication)
    verify_params = dict(params)
    verify_params["openid.mode"] = "check_authentication"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(STEAM_OPENID_URL, data=verify_params)
            resp.raise_for_status()

            # Steam returns key-value pairs separated by newlines
            body = resp.text
            if "is_valid:true" in body:
                logger.info(f"Steam OpenID: validated Steam64 ID {steam_id}")
                return steam_id
            else:
                logger.warning(f"Steam OpenID: validation failed. Response: {body}")
                return None

    except httpx.HTTPError as e:
        logger.error(f"Steam OpenID: HTTP error during validation: {e}")
        return None
    except Exception as e:
        logger.error(f"Steam OpenID: unexpected error during validation: {e}")
        return None


async def get_player_summary(steam_id: str) -> dict | None:
    """Fetch player profile data from Steam Web API.

    Requires STEAM_API_KEY environment variable.
    Returns avatar URL, display name, profile URL, etc.

    Args:
        steam_id: Steam64 ID (17-digit string)

    Returns:
        Player summary dict or None if API key not set or request fails.
    """
    api_key = os.environ.get("STEAM_API_KEY", "")
    if not api_key:
        logger.debug("Steam API key not set, skipping player summary")
        return None

    url = f"{STEAM_API_BASE}/ISteamUser/GetPlayerSummaries/v0002/"
    params = {"key": api_key, "steamids": steam_id}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            players = data.get("response", {}).get("players", [])
            if players:
                player = players[0]
                return {
                    "steam_id": steam_id,
                    "display_name": player.get("personaname", ""),
                    "avatar_url": player.get("avatarfull", ""),
                    "avatar_medium": player.get("avatarmedium", ""),
                    "profile_url": player.get("profileurl", ""),
                    "country_code": player.get("loccountrycode", ""),
                    "last_logoff": player.get("lastlogoff"),
                }
            return None

    except httpx.HTTPError as e:
        logger.warning(f"Steam API error fetching player summary: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching player summary: {e}")
        return None


def extract_steam_id_from_url(url: str) -> str | None:
    """Extract Steam64 ID from a Steam profile URL.

    Supports formats:
    - https://steamcommunity.com/id/vanityname
    - https://steamcommunity.com/profiles/76561198012345678

    Args:
        url: Steam profile URL

    Returns:
        Steam64 ID string or None if not a valid profile URL
    """
    parsed = urlparse(url)
    if "steamcommunity.com" not in parsed.netloc:
        return None

    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) >= 2 and path_parts[0] == "profiles":
        steam_id = path_parts[1]
        if re.match(r"^\d{17}$", steam_id):
            return steam_id
    return None


def _get_realm(return_url: str) -> str:
    """Extract the realm (scheme + host) from a URL.

    Args:
        return_url: Full callback URL

    Returns:
        Realm string (e.g., "https://opensight.gg")
    """
    parsed = urlparse(return_url)
    return f"{parsed.scheme}://{parsed.netloc}"
