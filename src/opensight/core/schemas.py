"""
OpenSight Data Contracts

EVERY data structure that crosses a module boundary is defined here.
If you need a field that doesn't exist here, ADD IT HERE FIRST,
then update the producer and consumer.

Producers: parser.py, analytics.py, cache.py
Consumers: strat_engine.py, scouting.py, self_review.py, static/index.html
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

# ============================================================
# ROUND TIMELINE — the central data structure
# ============================================================
# Produced by: cache.py _build_round_timeline()
# Consumed by: strat_engine.py, self_review.py, scouting.py, index.html


class TimelineKillEvent(TypedDict):
    """A kill event within a round timeline entry."""

    tick: int
    time_seconds: float
    type: str  # always "kill"
    killer: str  # attacker display name
    killer_team: str  # "CT" or "T"
    killer_steamid: NotRequired[int]
    victim: str  # victim display name
    victim_team: str  # "CT" or "T"
    victim_steamid: NotRequired[int]
    weapon: str
    headshot: bool
    is_first_kill: bool
    # Position data (for entry frag and strat detection)
    killer_x: NotRequired[float]
    killer_y: NotRequired[float]
    victim_x: NotRequired[float]
    victim_y: NotRequired[float]
    # Zone/callout (for strat engine)
    killer_zone: NotRequired[str]  # e.g., "A Site", "Mid", "B Apartments"
    victim_zone: NotRequired[str]
    # Win probability at this moment
    ct_prob: NotRequired[float]
    t_prob: NotRequired[float]


class TimelineBombEvent(TypedDict):
    """A bomb event within a round timeline entry."""

    tick: int
    time_seconds: float
    type: str  # "bomb_plant", "bomb_defuse", "bomb_explode"
    player: NotRequired[str]
    site: NotRequired[str]  # "A" or "B"


class TimelineUtilityEvent(TypedDict):
    """A utility event within a round timeline entry."""

    tick: int
    time_seconds: float
    type: str  # "flash", "smoke", "he", "molotov", "incendiary"
    player: str
    player_team: str  # "CT" or "T"
    player_steamid: NotRequired[int]
    x: NotRequired[float]
    y: NotRequired[float]
    zone: NotRequired[str]  # target zone/callout
    # Flash-specific
    enemies_flashed: NotRequired[int]
    teammates_flashed: NotRequired[int]
    total_blind_duration: NotRequired[float]


class PlayerPositionSnapshot(TypedDict):
    """A player's position at a point in time (e.g., 30 seconds into round)."""

    player_name: str
    player_steamid: int
    side: str  # "CT" or "T"
    x: float
    y: float
    z: NotRequired[float]
    zone: str  # callout name, e.g., "A Site", "Mid"
    is_alive: bool


class ClutchInfo(TypedDict):
    """Clutch situation detected in a round."""

    player: str
    player_steamid: NotRequired[int]
    player_team: str  # "CT" or "T"
    scenario: str  # "1v1", "1v2", "1v3", etc.
    won: bool
    kills_in_clutch: NotRequired[int]


class EconomyIQData(TypedDict, total=False):
    """Economy analysis data for one team in one round."""

    loss_bonus: int
    consecutive_losses: int
    equipment: int
    buy_type: str  # "full_buy", "eco", "force", "pistol"
    decision_flag: str  # "ok", "bad_force", "bad_eco"
    decision_grade: str  # "A" through "F"
    loss_bonus_next: int
    is_bad_force: bool
    is_good_force: bool
    prediction: dict | None  # Economy prediction data


class RoundEconomy(TypedDict, total=False):
    """Economy data for both teams in a round."""

    ct: EconomyIQData
    t: EconomyIQData


class RoundTimelineEntry(TypedDict):
    """
    One round in the match timeline.
    THIS IS THE CONTRACT. If strat_engine needs a field, it goes here.
    """

    round_num: int
    round_type: str  # "pistol", "eco", "force", "full_buy"
    winner: str  # "CT" or "T"
    win_reason: str  # "Elimination", "BombDefused", "TargetBombed", "Time"
    first_kill: str | None  # player name
    first_death: str | None  # player name
    ct_kills: int
    t_kills: int

    # Events in this round (kills + bombs + utility, sorted by tick)
    events: list[TimelineKillEvent | TimelineBombEvent | TimelineUtilityEvent]

    # Structured access to events by type (so consumers don't have to filter)
    kills: NotRequired[list[TimelineKillEvent]]
    utility: NotRequired[list[TimelineUtilityEvent]]
    blinds: NotRequired[list[TimelineUtilityEvent]]  # flash events with blind data

    # Player positions at round start (~30 seconds in, after setup)
    player_positions: NotRequired[list[PlayerPositionSnapshot]]
    alive_ct: NotRequired[int]
    alive_t: NotRequired[int]

    # Clutch situations
    clutches: NotRequired[list[ClutchInfo]]

    # Win probability / momentum
    momentum: NotRequired[dict]

    # Economy
    economy: NotRequired[RoundEconomy]


# ============================================================
# HEATMAP DATA
# ============================================================
# Produced by: cache.py _build_heatmap_data()
# Consumed by: index.html renderHeatmap()


class HeatmapPosition(TypedDict):
    """A single position point on the heatmap."""

    x: float
    y: float
    z: NotRequired[float]
    zone: str
    side: str  # "CT" or "T"
    phase: str  # "pre_plant" or "post_plant"
    round_type: str
    round_num: int
    player_name: str
    player_steamid: int
    weapon: NotRequired[str]  # kills only
    headshot: NotRequired[bool]  # kills only


class HeatmapData(TypedDict):
    """Complete heatmap data for the match."""

    map_name: str
    kill_positions: list[HeatmapPosition]
    death_positions: list[HeatmapPosition]
    grenade_positions: NotRequired[list[dict]]  # NEW: grenade landing spots
    zone_stats: dict[str, dict]
    zone_definitions: dict
    dry_peek_data: NotRequired[list[dict]]


# ============================================================
# MATCH RESULT — the top-level response from cache.py
# ============================================================
# Produced by: cache.py CachedAnalyzer.analyze()
# Consumed by: api.py (returned to frontend), all AI modules


class MatchInfo(TypedDict):
    """Top-level match metadata."""

    map: str
    rounds: int
    duration_minutes: int | float
    score: str  # "13 - 9"
    total_kills: int
    team1_name: str
    team2_name: str
    score_ct: NotRequired[int]
    score_t: NotRequired[int]


class MatchResult(TypedDict):
    """
    The complete analysis result returned by CachedAnalyzer.analyze().
    This is what api.py stores in job_store and returns to the frontend.
    """

    demo_info: MatchInfo
    players: list[dict]  # Sorted by rating
    mvp: dict
    round_timeline: list[RoundTimelineEntry]
    kill_matrix: list[dict]
    heatmap_data: HeatmapData
    coaching: list[dict]
    tactical: dict
    synergy: dict
    timeline_graph: dict
    analyzed_at: str  # ISO format datetime


# ============================================================
# ERROR REPORTING — distinguish "0 results" from "couldn't analyze"
# ============================================================


class AnalysisWarning(TypedDict):
    """A non-fatal issue detected during analysis."""

    module: str  # which module generated this warning
    code: str  # machine-readable code, e.g. "ROUND_COL_MISSING"
    message: str  # human-readable explanation
    impact: str  # what feature is degraded
