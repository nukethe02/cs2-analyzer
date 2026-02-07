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
        stats = p.get("stats", {})
        rating_data = p.get("rating", {})
        writer.writerow(
            [
                p.get("name", p.get("player_name", "")),
                p.get("steam_id", p.get("steamid", "")),
                p.get("team", ""),
                stats.get("kills", p.get("kills", 0)),
                stats.get("deaths", p.get("deaths", 0)),
                stats.get("assists", p.get("assists", 0)),
                stats.get("adr", p.get("adr", 0.0)),
                rating_data.get("hltv_rating", p.get("hltv_rating", 0.0))
                if isinstance(rating_data, dict)
                else p.get("rating", p.get("hltv_rating", 0.0)),
                rating_data.get("kast_percentage", p.get("kast_pct", p.get("kast", 0.0)))
                if isinstance(rating_data, dict)
                else p.get("kast_pct", p.get("kast", 0.0)),
                stats.get("headshot_pct", p.get("hs_pct", p.get("headshot_percentage", 0.0))),
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

    Scores are computed cumulatively from the ``winner`` field of each
    round since the orchestrator round_timeline does not include running
    score totals.

    Args:
        rounds: List of round dicts (from ``round_timeline``).

    Returns:
        CSV string including a header row.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_ROUND_CSV_COLUMNS)

    # Orchestrator round_timeline does NOT include running t_score / ct_score.
    # Compute cumulatively from the winner field of each round.
    ct_score = 0
    t_score = 0

    for r in rounds:
        # Accumulate scores from round winner
        winner = r.get("winner", "")
        if winner == "CT":
            ct_score += 1
        elif winner == "T":
            t_score += 1

        econ = r.get("economy") or {}
        t_econ = econ.get("t") or {}
        ct_econ = econ.get("ct") or {}
        writer.writerow(
            [
                r.get("round_num", r.get("round", "")),
                winner,
                r.get("win_reason", r.get("reason", "")),
                t_score,
                ct_score,
                t_econ.get("equipment", r.get("t_equipment", 0)),
                ct_econ.get("equipment", r.get("ct_equipment", 0)),
                t_econ.get("buy_type", r.get("t_buy_type", "")),
                ct_econ.get("buy_type", r.get("ct_buy_type", "")),
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
