"""User tier definitions and access control."""

from enum import IntEnum


class Tier(IntEnum):
    FREE = 0
    PRO = 1
    TEAM = 2
    ADMIN = 3


TIER_LIMITS: dict[str, dict] = {
    "free": {
        "demos_per_day": 3,
        "ai_enabled": False,
        "scouting_enabled": False,
        "export_enabled": False,
    },
    "pro": {
        "demos_per_day": 20,
        "ai_enabled": True,
        "scouting_enabled": True,
        "export_enabled": True,
    },
    "team": {
        "demos_per_day": -1,
        "ai_enabled": True,
        "scouting_enabled": True,
        "export_enabled": True,
    },
    "admin": {
        "demos_per_day": -1,
        "ai_enabled": True,
        "scouting_enabled": True,
        "export_enabled": True,
    },
}


def check_tier(user_tier: str, minimum: str) -> bool:
    """Check if user_tier meets or exceeds the minimum tier requirement."""
    tier_map = {"free": 0, "pro": 1, "team": 2, "admin": 3}
    return tier_map.get(user_tier, 0) >= tier_map[minimum]
