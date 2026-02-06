"""
Heatmap Generation Module for CS2 Demo Visualization.

Generates kill and grenade heatmap data from plain dict inputs,
suitable for frontend rendering on radar images.

Uses coordinate transforms from radar.py (CoordinateTransformer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HeatmapPoint:
    """A single point on a heatmap overlay."""

    x: float
    y: float
    point_type: str  # "kill", "death", "grenade"
    player_steam_id: str
    weapon: str
    round_number: int


def _get_transformer(map_name: str):
    """Lazy-import and build a CoordinateTransformer for *map_name*."""
    from opensight.visualization.radar import CoordinateTransformer

    return CoordinateTransformer(map_name)


def _passes_filters(kill: dict, filters: dict) -> bool:
    """Return True if *kill* passes all active filters."""
    steam_id = filters.get("steam_id")
    if steam_id is not None:
        att_id = str(kill.get("attacker_steamid", ""))
        vic_id = str(kill.get("victim_steamid", ""))
        if str(steam_id) not in (att_id, vic_id):
            return False

    side = filters.get("side")
    if side is not None:
        att_side = str(kill.get("attacker_side", "")).upper()
        if side.upper() != att_side:
            return False

    weapon = filters.get("weapon")
    if weapon is not None:
        if str(kill.get("weapon", "")).lower() != weapon.lower():
            return False

    round_range = filters.get("round_range")
    if round_range is not None:
        rnd = kill.get("round", kill.get("round_num", 0))
        lo, hi = round_range
        if not (lo <= rnd <= hi):
            return False

    return True


def generate_kill_heatmap(
    kills_data: list[dict],
    map_name: str,
    filters: dict | None = None,
) -> dict:
    """Generate kill heatmap data.

    Args:
        kills_data: List of kill dicts. Expected keys include
            ``attacker_x``, ``attacker_y``, ``victim_x``, ``victim_y``,
            ``attacker_steamid``, ``victim_steamid``, ``weapon``,
            ``round`` (or ``round_num``), ``is_headshot``.
        map_name: CS2 map name for coordinate transform (e.g. ``"de_dust2"``).
        filters: Optional dict with any of ``steam_id``, ``side``,
            ``weapon``, ``round_range`` (tuple of ``(lo, hi)``).

    Returns:
        ``{"map_name": str, "points": list[dict], "total": int}``
    """
    transformer = _get_transformer(map_name)
    active_filters = filters or {}
    points: list[dict] = []

    for kill in kills_data:
        if not _passes_filters(kill, active_filters):
            continue

        round_num = kill.get("round", kill.get("round_num", 0))
        weapon = str(kill.get("weapon", ""))

        # Attacker (kill) position
        att_x = kill.get("attacker_x")
        att_y = kill.get("attacker_y")
        if att_x is not None and att_y is not None:
            try:
                pos = transformer.game_to_radar(float(att_x), float(att_y))
                if pos.is_valid:
                    points.append(
                        {
                            "x": round(pos.x, 1),
                            "y": round(pos.y, 1),
                            "type": "kill",
                            "steam_id": str(kill.get("attacker_steamid", "")),
                            "weapon": weapon,
                            "round": round_num,
                            "headshot": bool(kill.get("is_headshot", False)),
                        }
                    )
            except Exception:
                pass

        # Victim (death) position
        vic_x = kill.get("victim_x")
        vic_y = kill.get("victim_y")
        if vic_x is not None and vic_y is not None:
            try:
                pos = transformer.game_to_radar(float(vic_x), float(vic_y))
                if pos.is_valid:
                    points.append(
                        {
                            "x": round(pos.x, 1),
                            "y": round(pos.y, 1),
                            "type": "death",
                            "steam_id": str(kill.get("victim_steamid", "")),
                            "weapon": weapon,
                            "round": round_num,
                            "headshot": bool(kill.get("is_headshot", False)),
                        }
                    )
            except Exception:
                pass

    return {
        "map_name": map_name,
        "points": points,
        "total": len(points),
    }


def generate_grenade_heatmap(
    grenades_data: list[dict],
    map_name: str,
    grenade_type: str | None = None,
) -> dict:
    """Generate grenade landing-position heatmap.

    Args:
        grenades_data: List of grenade dicts with ``x``, ``y``,
            ``grenade_type``, ``player_steamid``, ``round`` (or ``round_num``).
        map_name: CS2 map name for coordinate transform.
        grenade_type: Optional filter — e.g. ``"flashbang"``, ``"smoke"``,
            ``"hegrenade"``, ``"molotov"``.

    Returns:
        ``{"map_name": str, "points": list[dict], "total": int}``
    """
    transformer = _get_transformer(map_name)
    points: list[dict] = []

    for g in grenades_data:
        g_type = str(g.get("grenade_type", "")).lower()
        if grenade_type is not None and g_type != grenade_type.lower():
            continue

        gx = g.get("x")
        gy = g.get("y")
        if gx is None or gy is None:
            continue

        try:
            pos = transformer.game_to_radar(float(gx), float(gy))
            if pos.is_valid:
                points.append(
                    {
                        "x": round(pos.x, 1),
                        "y": round(pos.y, 1),
                        "type": g_type,
                        "steam_id": str(g.get("player_steamid", "")),
                        "round": g.get("round", g.get("round_num", 0)),
                    }
                )
        except Exception:
            pass

    return {
        "map_name": map_name,
        "points": points,
        "total": len(points),
    }
