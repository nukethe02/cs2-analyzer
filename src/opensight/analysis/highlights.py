"""
Highlight Detection Module for CS2 Demo Analysis.

Detects notable plays from kill/round data:
  - Aces (5K in a round)
  - 4K / 3K rounds
  - Clutch wins (1vN)
  - Noscope kills
  - Wallbang kills
  - Through-smoke kills

Returns highlights sorted by impact score (descending).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Weapons that qualify as "noscope" when they have scope
SNIPER_WEAPONS = {"awp", "ssg08", "scar20", "g3sg1"}

# Impact score ranges by highlight type
IMPACT_SCORES = {
    "ace": 90,
    "4k": 60,
    "3k": 40,
    "clutch_1v5": 95,
    "clutch_1v4": 85,
    "clutch_1v3": 70,
    "clutch_1v2": 55,
    "clutch_1v1": 45,
    "noscope": 30,
    "wallbang": 25,
    "through_smoke": 20,
}


@dataclass
class Highlight:
    """A single highlight moment from the match."""

    round_number: int
    player_steam_id: str
    player_name: str
    highlight_type: str  # ace, 4k, 3k, clutch_1vN, noscope, wallbang, through_smoke
    start_tick: int
    end_tick: int
    impact_score: float  # 0-100
    description: str
    weapons_used: list[str] = field(default_factory=list)


def _detect_multi_kills(
    kills_by_round_player: dict[int, dict[str, list[dict]]],
) -> list[Highlight]:
    """Detect aces, 4Ks, and 3Ks."""
    highlights: list[Highlight] = []

    for round_num, players in kills_by_round_player.items():
        for steam_id, kills in players.items():
            count = len(kills)
            if count < 3:
                continue

            name = kills[0].get("attacker_name", "Unknown")
            weapons = list({str(k.get("weapon", "")) for k in kills})
            ticks = [k.get("tick", 0) for k in kills]
            start_tick = min(ticks) if ticks else 0
            end_tick = max(ticks) if ticks else 0

            if count >= 5:
                h_type = "ace"
                score = IMPACT_SCORES["ace"]
                desc = f"{name} got an ACE in round {round_num} ({count} kills)"
            elif count == 4:
                h_type = "4k"
                score = IMPACT_SCORES["4k"]
                desc = f"{name} got a 4K in round {round_num}"
            else:
                h_type = "3k"
                score = IMPACT_SCORES["3k"]
                desc = f"{name} got a 3K in round {round_num}"

            # Bonus for headshot-heavy multi-kills
            hs_count = sum(1 for k in kills if k.get("is_headshot", False))
            if hs_count >= count - 1:
                score = min(100, score + 5)
                desc += " (all headshots)" if hs_count == count else " (mostly headshots)"

            highlights.append(
                Highlight(
                    round_number=round_num,
                    player_steam_id=str(steam_id),
                    player_name=name,
                    highlight_type=h_type,
                    start_tick=start_tick,
                    end_tick=end_tick,
                    impact_score=score,
                    description=desc,
                    weapons_used=weapons,
                )
            )

    return highlights


def _detect_clutches(
    kills_by_round_player: dict[int, dict[str, list[dict]]],
    rounds_data: list[dict] | None,
) -> list[Highlight]:
    """Detect clutch wins (1v2+) from round data if available."""
    if not rounds_data:
        return []

    highlights: list[Highlight] = []

    for rnd in rounds_data:
        round_num = rnd.get("round", rnd.get("round_num", 0))
        clutch_player = rnd.get("clutch_player_steamid") or rnd.get("clutch_player")
        clutch_enemies = rnd.get("clutch_enemies", 0)
        clutch_won = rnd.get("clutch_won", False)

        if not clutch_player or clutch_enemies < 2 or not clutch_won:
            continue

        steam_id = str(clutch_player)
        player_kills = kills_by_round_player.get(round_num, {}).get(steam_id, [])
        name = player_kills[0].get("attacker_name", "Unknown") if player_kills else "Unknown"
        weapons = list({str(k.get("weapon", "")) for k in player_kills})
        ticks = [k.get("tick", 0) for k in player_kills]
        start_tick = min(ticks) if ticks else 0
        end_tick = max(ticks) if ticks else 0

        key = f"clutch_1v{min(clutch_enemies, 5)}"
        score = IMPACT_SCORES.get(key, 55)
        desc = f"{name} won a 1v{clutch_enemies} clutch in round {round_num}"

        highlights.append(
            Highlight(
                round_number=round_num,
                player_steam_id=steam_id,
                player_name=name,
                highlight_type=f"clutch_1v{clutch_enemies}",
                start_tick=start_tick,
                end_tick=end_tick,
                impact_score=score,
                description=desc,
                weapons_used=weapons,
            )
        )

    return highlights


def _detect_special_kills(
    all_kills: list[dict],
) -> list[Highlight]:
    """Detect noscope, wallbang, and through-smoke kills."""
    highlights: list[Highlight] = []

    for kill in all_kills:
        weapon = str(kill.get("weapon", "")).lower()
        round_num = kill.get("round", kill.get("round_num", 0))
        tick = kill.get("tick", 0)
        steam_id = str(kill.get("attacker_steamid", ""))
        name = kill.get("attacker_name", "Unknown")

        # Noscope detection: sniper weapon + noscope flag
        if kill.get("is_noscope", False) and weapon in SNIPER_WEAPONS:
            highlights.append(
                Highlight(
                    round_number=round_num,
                    player_steam_id=steam_id,
                    player_name=name,
                    highlight_type="noscope",
                    start_tick=tick,
                    end_tick=tick,
                    impact_score=IMPACT_SCORES["noscope"],
                    description=f"{name} noscope kill with {weapon} in round {round_num}",
                    weapons_used=[weapon],
                )
            )

        # Wallbang detection
        if kill.get("is_wallbang", False) or kill.get("penetrated_objects", 0) > 0:
            highlights.append(
                Highlight(
                    round_number=round_num,
                    player_steam_id=steam_id,
                    player_name=name,
                    highlight_type="wallbang",
                    start_tick=tick,
                    end_tick=tick,
                    impact_score=IMPACT_SCORES["wallbang"],
                    description=f"{name} wallbang kill with {weapon} in round {round_num}",
                    weapons_used=[weapon],
                )
            )

        # Through-smoke detection
        if kill.get("is_through_smoke", False) or kill.get("thru_smoke", False):
            highlights.append(
                Highlight(
                    round_number=round_num,
                    player_steam_id=steam_id,
                    player_name=name,
                    highlight_type="through_smoke",
                    start_tick=tick,
                    end_tick=tick,
                    impact_score=IMPACT_SCORES["through_smoke"],
                    description=f"{name} through-smoke kill with {weapon} in round {round_num}",
                    weapons_used=[weapon],
                )
            )

    return highlights


def detect_highlights(
    kills_data: list[dict],
    rounds_data: list[dict] | None = None,
) -> list[Highlight]:
    """Detect highlights from kill data.

    Detects:
      - Aces (5+ kills in a round) — impact 80-100
      - 4K rounds — impact 60
      - 3K rounds — impact 40
      - Clutch wins (1vN from rounds_data) — impact 45-95
      - Noscope kills — impact 30
      - Wallbang kills — impact 25
      - Through-smoke kills — impact 20

    Args:
        kills_data: List of kill dicts. Expected keys:
            ``attacker_steamid``, ``attacker_name``, ``weapon``,
            ``round`` (or ``round_num``), ``tick``, ``is_headshot``,
            and optionally ``is_noscope``, ``is_wallbang``,
            ``penetrated_objects``, ``is_through_smoke``, ``thru_smoke``.
        rounds_data: Optional list of round dicts with clutch info:
            ``round``, ``clutch_player_steamid``, ``clutch_enemies``,
            ``clutch_won``.

    Returns:
        List of :class:`Highlight` sorted by ``impact_score`` descending.
    """
    # Group kills by (round, attacker) for multi-kill detection
    kills_by_round_player: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for kill in kills_data:
        round_num = kill.get("round", kill.get("round_num", 0))
        steam_id = str(kill.get("attacker_steamid", ""))
        kills_by_round_player[round_num][steam_id].append(kill)

    highlights: list[Highlight] = []

    # Multi-kills (ace / 4K / 3K)
    highlights.extend(_detect_multi_kills(kills_by_round_player))

    # Clutches (from rounds data)
    highlights.extend(_detect_clutches(kills_by_round_player, rounds_data))

    # Special kills (noscope, wallbang, through-smoke)
    highlights.extend(_detect_special_kills(kills_data))

    # Sort by impact score descending, then round number
    highlights.sort(key=lambda h: (-h.impact_score, h.round_number))

    logger.info(f"Detected {len(highlights)} highlights")
    return highlights
