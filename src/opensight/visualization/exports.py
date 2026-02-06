"""
Export Module for CS2 Demo Analysis Results.

Provides JSON and CSV serialization for match data, player stats,
and round-by-round breakdowns.  Uses only the standard library.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any


def export_match_json(match_data: dict) -> str:
    """Serialize *match_data* to a pretty-printed JSON string.

    ``datetime`` and other non-serializable objects are coerced via ``str``.
    """
    return json.dumps(match_data, indent=2, default=str)


_PLAYER_CSV_COLUMNS = [
    "name",
    "steam_id",
    "team",
    "kills",
    "deaths",
    "assists",
    "adr",
    "rating",
    "kast_pct",
    "hs_pct",
]


def export_player_stats_csv(players: list[dict]) -> str:
    """Export player stats to CSV.

    Each row contains: name, steam_id, team, kills, deaths, assists,
    adr, rating, kast_pct, hs_pct.

    Args:
        players: List of player stat dicts.  Keys are matched
            case-insensitively and common aliases are handled
            (e.g. ``headshot_pct`` → ``hs_pct``).

    Returns:
        CSV string including a header row.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_PLAYER_CSV_COLUMNS)

    for p in players:
        writer.writerow(
            [
                p.get("name", p.get("player_name", "")),
                p.get("steam_id", p.get("steamid", "")),
                p.get("team", ""),
                p.get("kills", 0),
                p.get("deaths", 0),
                p.get("assists", 0),
                p.get("adr", 0.0),
                p.get("rating", p.get("hltv_rating", 0.0)),
                p.get("kast_pct", p.get("kast", 0.0)),
                p.get("hs_pct", p.get("headshot_pct", p.get("headshot_percentage", 0.0))),
            ]
        )

    return output.getvalue()


_ROUND_CSV_COLUMNS = [
    "round",
    "winner",
    "win_reason",
    "t_score",
    "ct_score",
    "t_equipment",
    "ct_equipment",
    "t_buy_type",
    "ct_buy_type",
]


def export_rounds_csv(rounds: list[dict]) -> str:
    """Export round-by-round data to CSV.

    Each row contains: round, winner, win_reason, t_score, ct_score,
    t_equipment, ct_equipment, t_buy_type, ct_buy_type.

    Args:
        rounds: List of round dicts.

    Returns:
        CSV string including a header row.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_ROUND_CSV_COLUMNS)

    for r in rounds:
        writer.writerow(
            [
                r.get("round", r.get("round_num", "")),
                r.get("winner", ""),
                r.get("win_reason", r.get("reason", "")),
                r.get("t_score", 0),
                r.get("ct_score", 0),
                r.get("t_equipment", r.get("team_t_val", 0)),
                r.get("ct_equipment", r.get("team_ct_val", 0)),
                r.get("t_buy_type", r.get("t_buy", "")),
                r.get("ct_buy_type", r.get("ct_buy", "")),
            ]
        )

    return output.getvalue()


def export_highlights_json(highlights: list[Any]) -> str:
    """Serialize a list of Highlight dataclasses to JSON.

    Accepts either dicts or objects with a ``__dict__`` attribute.
    """
    items = []
    for h in highlights:
        if isinstance(h, dict):
            items.append(h)
        elif hasattr(h, "__dict__"):
            items.append(h.__dict__)
        else:
            items.append(str(h))
    return json.dumps(items, indent=2, default=str)
