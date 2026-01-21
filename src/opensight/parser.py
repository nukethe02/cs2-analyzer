"""
Demo Parser for CS2 Replay Files - awpy Edition

Extracts data from CS2 demos using awpy library (primary parser).
awpy handles edge cases like warmup, overtime, and malformed demos
that custom parsing code often crashes on.

Data extracted:
- Kills, Deaths, Assists with position/angle data
- All damage events with hitgroups
- Round events with economy data
- Grenade events (flash, HE, smoke, molotov)
- Bomb events (plant, defuse, explode)
- Player positions and velocities (tick data)

The standardized MatchData structure provides a robust, well-typed
interface for analysis code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Parser availability flag - checked lazily on first use
_AWPY_AVAILABLE: bool | None = None


def _check_awpy_available() -> bool:
    """Lazily check if awpy is available."""
    global _AWPY_AVAILABLE
    if _AWPY_AVAILABLE is None:
        try:
            from awpy import Demo as _  # noqa: F401
            _AWPY_AVAILABLE = True
        except ImportError:
            _AWPY_AVAILABLE = False
    return _AWPY_AVAILABLE


# =============================================================================
# Safe Type Conversion Helpers
# =============================================================================

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    """Safely convert a value to string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert a value to bool."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return bool(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# Data Classes - Standardized Schema Following awpy's Output
# =============================================================================

@dataclass
class KillEvent:
    """A kill event with timing, position, and angle data.

    Schema aligned with awpy's kills DataFrame.
    """
    tick: int
    round_num: int
    attacker_steamid: int
    attacker_name: str
    attacker_side: str  # "CT" or "T"
    victim_steamid: int
    victim_name: str
    victim_side: str
    weapon: str
    headshot: bool
    penetrated: bool = False  # Wallbang
    noscope: bool = False
    thrusmoke: bool = False
    attackerblind: bool = False  # Attacker was flashed
    assister_steamid: int | None = None
    assister_name: str | None = None
    assister_side: str | None = None
    flash_assist: bool = False
    # Attacker position and view angles
    attacker_x: float | None = None
    attacker_y: float | None = None
    attacker_z: float | None = None
    attacker_pitch: float | None = None
    attacker_yaw: float | None = None
    # Victim position
    victim_x: float | None = None
    victim_y: float | None = None
    victim_z: float | None = None
    # Distance
    distance: float | None = None


@dataclass
class DamageEvent:
    """A damage event with full details.

    Schema aligned with awpy's damages DataFrame.
    """
    tick: int
    round_num: int
    attacker_steamid: int
    attacker_name: str
    attacker_side: str
    victim_steamid: int
    victim_name: str
    victim_side: str
    damage: int
    damage_armor: int
    health_remaining: int
    armor_remaining: int
    weapon: str
    hitgroup: str  # 'head', 'chest', 'stomach', 'left_arm', 'right_arm', 'left_leg', 'right_leg'


@dataclass
class RoundInfo:
    """Comprehensive round information.

    Schema aligned with awpy's rounds DataFrame.
    """
    round_num: int
    start_tick: int
    end_tick: int
    freeze_end_tick: int
    winner: str  # "CT" or "T"
    reason: str  # 'bomb_defused', 'target_bombed', 'elimination', 'time_expired', etc.
    ct_score: int = 0
    t_score: int = 0
    bomb_plant_tick: int | None = None
    bomb_site: str = ""  # 'A', 'B', or empty if not planted
    # Economy data (optional)
    ct_team_money: int = 0
    t_team_money: int = 0
    ct_equipment_value: int = 0
    t_equipment_value: int = 0
    round_type: str = ""  # 'pistol', 'eco', 'force', 'full_buy'


@dataclass
class GrenadeEvent:
    """A grenade event (thrown or detonated).

    Schema aligned with awpy's grenades DataFrame.
    """
    tick: int
    round_num: int
    player_steamid: int
    player_name: str
    player_side: str
    grenade_type: str  # 'flashbang', 'hegrenade', 'smokegrenade', 'molotov', 'incgrenade', 'decoy'
    x: float | None = None
    y: float | None = None
    z: float | None = None
    entity_id: int | None = None


@dataclass
class BombEvent:
    """A bomb-related event.

    Schema aligned with awpy's bomb DataFrame.
    """
    tick: int
    round_num: int
    player_steamid: int
    player_name: str
    event_type: str  # 'dropped', 'carried', 'planted', 'defused', 'detonated'
    site: str = ""  # 'A' or 'B'
    x: float | None = None
    y: float | None = None
    z: float | None = None


@dataclass
class SmokeEvent:
    """A smoke grenade effect.

    Schema aligned with awpy's smokes DataFrame.
    """
    start_tick: int
    end_tick: int
    round_num: int
    thrower_steamid: int
    thrower_name: str
    thrower_side: str
    x: float
    y: float
    z: float
    entity_id: int | None = None


@dataclass
class InfernoEvent:
    """A molotov/incendiary fire effect.

    Schema aligned with awpy's infernos DataFrame.
    """
    start_tick: int
    end_tick: int
    round_num: int
    thrower_steamid: int
    thrower_name: str
    thrower_side: str
    x: float
    y: float
    z: float
    entity_id: int | None = None


@dataclass
class WeaponFireEvent:
    """A weapon fire event for accuracy tracking.

    Schema aligned with awpy's shots DataFrame.
    """
    tick: int
    round_num: int
    player_steamid: int
    player_name: str
    player_side: str
    weapon: str
    player_x: float | None = None
    player_y: float | None = None
    player_z: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    is_silenced: bool = False


@dataclass
class BlindEvent:
    """A player blind event from flashbang."""
    tick: int
    round_num: int
    attacker_steamid: int
    attacker_name: str
    attacker_side: str
    victim_steamid: int
    victim_name: str
    victim_side: str
    blind_duration: float
    is_teammate: bool = False


# =============================================================================
# MatchData - The Primary Data Structure
# =============================================================================

@dataclass
class MatchData:
    """
    Complete parsed match data - robust structure using awpy's standardized schema.

    This dataclass holds all parsed demo data in a well-typed, consistent format.
    It handles edge cases like warmup rounds, overtime, and malformed demos that
    custom parsing code often crashes on.

    Attributes:
        file_path: Path to the original demo file
        map_name: Map name (e.g., 'de_dust2', 'de_inferno')
        duration_seconds: Total demo duration in seconds
        tick_rate: Server tick rate (usually 64 or 128)
        num_rounds: Total number of rounds played

        game_rounds: List of round objects with start/end ticks, winner, reason
        kills: List of kill events using awpy's standardized schema
        damages: List of damage events
        bomb_events: List of plant/defuse/explode events
        grenades: List of grenade throw events
        smokes: List of smoke effect events (start/end)
        infernos: List of molotov/incendiary fire events
        weapon_fires: List of weapon fire events (shots)
        blinds: List of player blind events

        player_stats: Dictionary of basic player statistics
        player_names: Mapping of steamid -> player name
        player_teams: Mapping of steamid -> team (CT/T)

        kills_df: Raw kills DataFrame from awpy
        damages_df: Raw damages DataFrame from awpy
        rounds_df: Raw rounds DataFrame from awpy
    """
    file_path: Path
    map_name: str
    duration_seconds: float
    tick_rate: int
    num_rounds: int
    server_name: str = ""
    game_mode: str = ""  # 'competitive', 'premier', 'casual'

    # Core Game Data - Lists of typed events
    game_rounds: list[RoundInfo] = field(default_factory=list)
    kills: list[KillEvent] = field(default_factory=list)
    damages: list[DamageEvent] = field(default_factory=list)
    bomb_events: list[BombEvent] = field(default_factory=list)

    # Extended Events
    grenades: list[GrenadeEvent] = field(default_factory=list)
    smokes: list[SmokeEvent] = field(default_factory=list)
    infernos: list[InfernoEvent] = field(default_factory=list)
    weapon_fires: list[WeaponFireEvent] = field(default_factory=list)
    blinds: list[BlindEvent] = field(default_factory=list)

    # Player Info
    player_stats: dict[int, dict] = field(default_factory=dict)
    player_names: dict[int, str] = field(default_factory=dict)
    player_teams: dict[int, str] = field(default_factory=dict)

    # Raw DataFrames for detailed analysis (from awpy)
    kills_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    damages_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    rounds_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    grenades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bomb_events_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    weapon_fires_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    blinds_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ticks_df: pd.DataFrame | None = None

    # Match Summary
    final_score_ct: int = 0
    final_score_t: int = 0

    # Backward compatibility alias
    @property
    def rounds(self) -> list[RoundInfo]:
        """Alias for game_rounds for backward compatibility."""
        return self.game_rounds


# Backward compatibility alias
DemoData = MatchData


# =============================================================================
# Demo Parser using awpy
# =============================================================================

class DemoParser:
    """
    CS2 demo parser using awpy library.

    awpy handles edge cases (warmup, overtime, malformed demos) robustly,
    providing a standardized output format based on demoparser2.

    Usage:
        parser = DemoParser("demo.dem")
        data = parser.parse()

        # Access typed events
        for kill in data.kills:
            print(f"{kill.attacker_name} killed {kill.victim_name}")

        # Access raw DataFrames for vectorized analysis
        kills_df = data.kills_df
    """

    # Player properties to extract for tick data
    PLAYER_PROPS = [
        "X", "Y", "Z",
        "pitch", "yaw",
        "velocity_X", "velocity_Y", "velocity_Z",
        "health", "armor_value",
        "is_alive", "is_scoped",
        "balance", "current_equip_value",
        "last_place_name",
        "in_crouch", "is_walking",
        "has_helmet", "has_defuser",
    ]

    def __init__(self, demo_path: str | Path):
        """
        Initialize the demo parser.

        Args:
            demo_path: Path to the .dem file
        """
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")

        self._data: MatchData | None = None
        self._demo: Any | None = None  # awpy Demo object

    def parse(self, include_ticks: bool = False) -> MatchData:
        """
        Parse the demo file using awpy and extract all relevant data.

        Args:
            include_ticks: If True, parse tick-level position data (slower, more memory)

        Returns:
            MatchData with all parsed events and DataFrames
        """
        if self._data is not None:
            return self._data

        if not _check_awpy_available():
            raise ImportError(
                "awpy is required for demo parsing. Install with: pip install awpy\n"
                "Note: awpy requires Python >= 3.11"
            )

        from awpy import Demo

        logger.info(f"Parsing demo with awpy: {self.demo_path}")
        parse_start = time.time()

        # Initialize awpy Demo
        demo = Demo(str(self.demo_path), verbose=False)
        self._demo = demo

        # Parse with player props if we want tick data
        if include_ticks:
            demo.parse(player_props=self.PLAYER_PROPS)
        else:
            demo.parse()

        logger.info(f"awpy parsing took {time.time() - parse_start:.2f}s")

        # Extract header metadata
        header = demo.header or {}
        map_name = header.get("map_name", "unknown")
        server_name = header.get("server_name", "")

        # Convert Polars DataFrames to Pandas
        kills_df = demo.kills.to_pandas() if demo.kills is not None else pd.DataFrame()
        damages_df = demo.damages.to_pandas() if demo.damages is not None else pd.DataFrame()
        rounds_df = demo.rounds.to_pandas() if demo.rounds is not None else pd.DataFrame()
        grenades_df = demo.grenades.to_pandas() if demo.grenades is not None else pd.DataFrame()
        bomb_df = demo.bomb.to_pandas() if demo.bomb is not None else pd.DataFrame()
        shots_df = demo.shots.to_pandas() if demo.shots is not None else pd.DataFrame()
        ticks_df = demo.ticks.to_pandas() if include_ticks and demo.ticks is not None else None

        # Log parsed data sizes
        logger.info(f"Parsed: {len(kills_df)} kills, {len(damages_df)} damages, "
                   f"{len(rounds_df)} rounds, {len(grenades_df)} grenades, "
                   f"{len(bomb_df)} bomb events, {len(shots_df)} shots")

        if not kills_df.empty:
            logger.debug(f"Kill columns: {list(kills_df.columns)}")

        # Calculate duration and tick rate
        tick_rate = 64  # CS2 default
        max_tick = 0
        if not kills_df.empty and "tick" in kills_df.columns:
            max_tick = int(kills_df["tick"].max())
        if not rounds_df.empty and "end" in rounds_df.columns:
            max_tick = max(max_tick, int(rounds_df["end"].max()))
        duration_seconds = max_tick / tick_rate

        num_rounds = len(rounds_df) if not rounds_df.empty else 1

        # Build typed event lists
        kills = self._build_kills(kills_df)
        damages = self._build_damages(damages_df)
        game_rounds = self._build_rounds(rounds_df)
        grenades = self._build_grenades(grenades_df)
        bomb_events = self._build_bomb_events(bomb_df)
        weapon_fires = self._build_weapon_fires(shots_df)

        # Build smokes and infernos from awpy's dedicated DataFrames
        smokes = self._build_smokes(demo)
        infernos = self._build_infernos(demo)

        # Extract player info
        player_names, player_teams = self._extract_players(kills_df, damages_df)
        player_stats = self._calculate_stats(
            kills_df, damages_df, player_names, player_teams, num_rounds
        )

        # Calculate final scores
        final_ct = sum(1 for r in game_rounds if r.winner == "CT")
        final_t = sum(1 for r in game_rounds if r.winner == "T")

        self._data = MatchData(
            file_path=self.demo_path,
            map_name=map_name,
            duration_seconds=duration_seconds,
            tick_rate=tick_rate,
            num_rounds=num_rounds,
            server_name=server_name,
            # Core events
            game_rounds=game_rounds,
            kills=kills,
            damages=damages,
            bomb_events=bomb_events,
            # Extended events
            grenades=grenades,
            smokes=smokes,
            infernos=infernos,
            weapon_fires=weapon_fires,
            # Player info
            player_stats=player_stats,
            player_names=player_names,
            player_teams=player_teams,
            # Raw DataFrames
            kills_df=kills_df,
            damages_df=damages_df,
            rounds_df=rounds_df,
            grenades_df=grenades_df,
            bomb_events_df=bomb_df,
            weapon_fires_df=shots_df,
            ticks_df=ticks_df,
            # Scores
            final_score_ct=final_ct,
            final_score_t=final_t,
        )

        logger.info(f"Parsing complete: {len(player_stats)} players, {num_rounds} rounds, "
                   f"{len(kills)} kills, {final_ct}-{final_t}")
        return self._data

    def _normalize_side(self, value: Any) -> str:
        """Normalize team/side values to 'CT' or 'T'."""
        if value is None:
            return "Unknown"
        if isinstance(value, str):
            upper = value.upper()
            if "CT" in upper or "COUNTER" in upper:
                return "CT"
            elif "T" in upper and "CT" not in upper:
                return "T"
            return value
        elif isinstance(value, (int, float)):
            # CS2 team numbers: 2 = T, 3 = CT
            val = int(value)
            if val == 3:
                return "CT"
            elif val == 2:
                return "T"
        return "Unknown"

    def _build_kills(self, df: pd.DataFrame) -> list[KillEvent]:
        """Build KillEvent list from awpy kills DataFrame."""
        kills = []
        if df.empty:
            return kills

        for _, row in df.iterrows():
            try:
                # Extract attacker/victim sides with fallback column names
                att_side = row.get("attacker_side", row.get("attacker_team"))
                vic_side = row.get("victim_side", row.get("victim_team"))
                ass_side = row.get("assister_side")

                # Extract assister info if present
                ass_steamid = row.get("assister_steamid")
                ass_name = row.get("assister_name")

                kill = KillEvent(
                    tick=safe_int(row.get("tick")),
                    round_num=safe_int(row.get("round_num", row.get("round", 0))),
                    attacker_steamid=safe_int(row.get("attacker_steamid")),
                    attacker_name=safe_str(
                        row.get("attacker_name", row.get("attacker", ""))
                    ),
                    attacker_side=self._normalize_side(att_side),
                    victim_steamid=safe_int(
                        row.get("victim_steamid", row.get("user_steamid"))
                    ),
                    victim_name=safe_str(
                        row.get("victim_name", row.get("user_name", ""))
                    ),
                    victim_side=self._normalize_side(vic_side),
                    weapon=safe_str(row.get("weapon", "")),
                    headshot=safe_bool(row.get("headshot")),
                    penetrated=safe_bool(row.get("penetrated")),
                    noscope=safe_bool(row.get("noscope")),
                    thrusmoke=safe_bool(row.get("thrusmoke")),
                    attackerblind=safe_bool(row.get("attackerblind")),
                    assister_steamid=safe_int(ass_steamid) if ass_steamid else None,
                    assister_name=safe_str(ass_name) if ass_name else None,
                    assister_side=self._normalize_side(ass_side) if ass_side else None,
                    flash_assist=safe_bool(row.get("flash_assist")),
                    # Position data (may not always be present)
                    attacker_x=safe_float(row.get("attacker_X")) if "attacker_X" in row else None,
                    attacker_y=safe_float(row.get("attacker_Y")) if "attacker_Y" in row else None,
                    attacker_z=safe_float(row.get("attacker_Z")) if "attacker_Z" in row else None,
                    victim_x=safe_float(row.get("victim_X")) if "victim_X" in row else None,
                    victim_y=safe_float(row.get("victim_Y")) if "victim_Y" in row else None,
                    victim_z=safe_float(row.get("victim_Z")) if "victim_Z" in row else None,
                )
                kills.append(kill)
            except Exception as e:
                logger.debug(f"Error parsing kill event: {e}")
                continue

        return kills

    def _build_damages(self, df: pd.DataFrame) -> list[DamageEvent]:
        """Build DamageEvent list from awpy damages DataFrame."""
        damages = []
        if df.empty:
            return damages

        for _, row in df.iterrows():
            try:
                # Extract attacker/victim sides with fallback column names
                att_side = row.get("attacker_side", row.get("attacker_team"))
                vic_side = row.get("victim_side", row.get("victim_team"))

                # Extract damage with multiple fallback column names
                dmg = row.get("damage", row.get("health_damage", row.get("dmg_health", 0)))

                damage = DamageEvent(
                    tick=safe_int(row.get("tick")),
                    round_num=safe_int(row.get("round_num", row.get("round", 0))),
                    attacker_steamid=safe_int(row.get("attacker_steamid")),
                    attacker_name=safe_str(
                        row.get("attacker_name", row.get("attacker", ""))
                    ),
                    attacker_side=self._normalize_side(att_side),
                    victim_steamid=safe_int(
                        row.get("victim_steamid", row.get("user_steamid"))
                    ),
                    victim_name=safe_str(
                        row.get("victim_name", row.get("user_name", ""))
                    ),
                    victim_side=self._normalize_side(vic_side),
                    damage=safe_int(dmg),
                    damage_armor=safe_int(row.get("armor_damage", row.get("dmg_armor", 0))),
                    health_remaining=safe_int(row.get("health", 0)),
                    armor_remaining=safe_int(row.get("armor", 0)),
                    weapon=safe_str(row.get("weapon", "")),
                    hitgroup=safe_str(row.get("hitgroup", "generic")),
                )
                damages.append(damage)
            except Exception as e:
                logger.debug(f"Error parsing damage event: {e}")
                continue

        return damages

    def _build_rounds(self, df: pd.DataFrame) -> list[RoundInfo]:
        """Build RoundInfo list from awpy rounds DataFrame."""
        rounds = []
        if df.empty:
            return rounds

        for idx, row in df.iterrows():
            try:
                # Determine winner
                winner = self._normalize_side(row.get("winner"))
                reason = safe_str(row.get("reason", ""))

                # Infer winner from reason if not explicitly set
                if winner == "Unknown" and reason:
                    reason_lower = reason.lower()
                    ct_reasons = ["bomb_defused", "ct_win", "target_saved", "ct_killed"]
                    t_reasons = ["target_bombed", "t_win", "bomb_exploded", "terrorists_win"]
                    if any(r in reason_lower for r in ct_reasons):
                        winner = "CT"
                    elif any(r in reason_lower for r in t_reasons):
                        winner = "T"

                # Handle bomb plant info
                bomb_plant_tick = safe_int(row.get("bomb_plant")) if row.get("bomb_plant") else None
                bomb_site_raw = safe_str(row.get("bomb_site", ""))
                bomb_site = ""
                if bomb_site_raw:
                    if "a" in bomb_site_raw.lower():
                        bomb_site = "A"
                    elif "b" in bomb_site_raw.lower():
                        bomb_site = "B"

                round_info = RoundInfo(
                    round_num=safe_int(row.get("round_num", idx + 1)),
                    start_tick=safe_int(row.get("start", 0)),
                    end_tick=safe_int(row.get("end", row.get("official_end", 0))),
                    freeze_end_tick=safe_int(row.get("freeze_end", 0)),
                    winner=winner,
                    reason=reason,
                    bomb_plant_tick=bomb_plant_tick,
                    bomb_site=bomb_site,
                )
                rounds.append(round_info)
            except Exception as e:
                logger.debug(f"Error parsing round info: {e}")
                continue

        # Calculate cumulative scores
        ct_score = 0
        t_score = 0
        for r in rounds:
            if r.winner == "CT":
                ct_score += 1
            elif r.winner == "T":
                t_score += 1
            r.ct_score = ct_score
            r.t_score = t_score

        return rounds

    def _build_grenades(self, df: pd.DataFrame) -> list[GrenadeEvent]:
        """Build GrenadeEvent list from awpy grenades DataFrame."""
        grenades = []
        if df.empty:
            return grenades

        for _, row in df.iterrows():
            try:
                grenade = GrenadeEvent(
                    tick=safe_int(row.get("tick")),
                    round_num=safe_int(row.get("round_num", 0)),
                    player_steamid=safe_int(row.get("thrower_steamid", row.get("steamid"))),
                    player_name=safe_str(row.get("thrower", row.get("thrower_name", ""))),
                    player_side=self._normalize_side(row.get("thrower_side")),
                    grenade_type=safe_str(row.get("grenade_type", "")),
                    x=safe_float(row.get("X")) if "X" in row else None,
                    y=safe_float(row.get("Y")) if "Y" in row else None,
                    z=safe_float(row.get("Z")) if "Z" in row else None,
                    entity_id=safe_int(row.get("entity_id")) if row.get("entity_id") else None,
                )
                grenades.append(grenade)
            except Exception as e:
                logger.debug(f"Error parsing grenade event: {e}")
                continue

        return grenades

    def _build_bomb_events(self, df: pd.DataFrame) -> list[BombEvent]:
        """Build BombEvent list from awpy bomb DataFrame."""
        bomb_events = []
        if df.empty:
            return bomb_events

        for _, row in df.iterrows():
            try:
                # Normalize status to event_type
                status = safe_str(row.get("status", row.get("event_type", "")))

                # Handle bombsite
                bombsite = safe_str(row.get("bombsite", row.get("site", "")))
                site = ""
                if bombsite:
                    if "a" in bombsite.lower():
                        site = "A"
                    elif "b" in bombsite.lower():
                        site = "B"

                bomb_event = BombEvent(
                    tick=safe_int(row.get("tick")),
                    round_num=safe_int(row.get("round_num", 0)),
                    player_steamid=safe_int(row.get("steamid", row.get("player_steamid"))),
                    player_name=safe_str(row.get("name", row.get("player_name", ""))),
                    event_type=status,
                    site=site,
                    x=safe_float(row.get("X")) if "X" in row else None,
                    y=safe_float(row.get("Y")) if "Y" in row else None,
                    z=safe_float(row.get("Z")) if "Z" in row else None,
                )
                bomb_events.append(bomb_event)
            except Exception as e:
                logger.debug(f"Error parsing bomb event: {e}")
                continue

        return bomb_events

    def _build_weapon_fires(self, df: pd.DataFrame) -> list[WeaponFireEvent]:
        """Build WeaponFireEvent list from awpy shots DataFrame."""
        fires = []
        if df.empty:
            return fires

        for _, row in df.iterrows():
            try:
                fire = WeaponFireEvent(
                    tick=safe_int(row.get("tick")),
                    round_num=safe_int(row.get("round_num", 0)),
                    player_steamid=safe_int(row.get("steamid", row.get("player_steamid"))),
                    player_name=safe_str(row.get("name", row.get("player_name", ""))),
                    player_side=self._normalize_side(row.get("side", row.get("team"))),
                    weapon=safe_str(row.get("weapon", "")),
                    player_x=safe_float(row.get("X")) if "X" in row else None,
                    player_y=safe_float(row.get("Y")) if "Y" in row else None,
                    player_z=safe_float(row.get("Z")) if "Z" in row else None,
                    is_silenced=safe_bool(row.get("silenced", row.get("is_silenced"))),
                )
                fires.append(fire)
            except Exception as e:
                logger.debug(f"Error parsing weapon fire event: {e}")
                continue

        return fires

    def _build_smokes(self, demo: Any) -> list[SmokeEvent]:
        """Build SmokeEvent list from awpy smokes DataFrame."""
        smokes = []
        if demo.smokes is None:
            return smokes

        try:
            df = demo.smokes.to_pandas()
            for _, row in df.iterrows():
                try:
                    smoke = SmokeEvent(
                        start_tick=safe_int(row.get("start_tick")),
                        end_tick=safe_int(row.get("end_tick")),
                        round_num=safe_int(row.get("round_num", 0)),
                        thrower_steamid=safe_int(row.get("thrower_steamid")),
                        thrower_name=safe_str(row.get("thrower_name", "")),
                        thrower_side=self._normalize_side(row.get("thrower_side")),
                        x=safe_float(row.get("X")),
                        y=safe_float(row.get("Y")),
                        z=safe_float(row.get("Z")),
                        entity_id=safe_int(row.get("entity_id")) if row.get("entity_id") else None,
                    )
                    smokes.append(smoke)
                except Exception as e:
                    logger.debug(f"Error parsing smoke event: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Could not parse smokes: {e}")

        return smokes

    def _build_infernos(self, demo: Any) -> list[InfernoEvent]:
        """Build InfernoEvent list from awpy infernos DataFrame."""
        infernos = []
        if demo.infernos is None:
            return infernos

        try:
            df = demo.infernos.to_pandas()
            for _, row in df.iterrows():
                try:
                    inferno = InfernoEvent(
                        start_tick=safe_int(row.get("start_tick")),
                        end_tick=safe_int(row.get("end_tick")),
                        round_num=safe_int(row.get("round_num", 0)),
                        thrower_steamid=safe_int(row.get("thrower_steamid")),
                        thrower_name=safe_str(row.get("thrower_name", "")),
                        thrower_side=self._normalize_side(row.get("thrower_side")),
                        x=safe_float(row.get("X")),
                        y=safe_float(row.get("Y")),
                        z=safe_float(row.get("Z")),
                        entity_id=safe_int(row.get("entity_id")) if row.get("entity_id") else None,
                    )
                    infernos.append(inferno)
                except Exception as e:
                    logger.debug(f"Error parsing inferno event: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Could not parse infernos: {e}")

        return infernos

    def _extract_players(
        self, kills_df: pd.DataFrame, damages_df: pd.DataFrame
    ) -> tuple[dict[int, str], dict[int, str]]:
        """Extract player names and teams from DataFrames."""
        names: dict[int, str] = {}
        teams: dict[int, str] = {}

        def extract_from_df(df: pd.DataFrame, id_col: str, name_col: str, side_col: str):
            if df.empty or id_col not in df.columns:
                return
            for _, row in df.drop_duplicates(subset=[id_col]).iterrows():
                sid = safe_int(row.get(id_col))
                if sid and sid not in names:
                    if name_col in df.columns:
                        names[sid] = safe_str(row.get(name_col))
                    if side_col in df.columns:
                        teams[sid] = self._normalize_side(row.get(side_col))

        # Extract from kills (attackers and victims)
        for prefix in ["attacker", "victim"]:
            id_col = f"{prefix}_steamid"
            name_col = f"{prefix}_name"
            side_col = f"{prefix}_side"
            extract_from_df(kills_df, id_col, name_col, side_col)

        # Also check alternative column names
        extract_from_df(kills_df, "user_steamid", "user_name", "user_side")
        extract_from_df(damages_df, "attacker_steamid", "attacker_name", "attacker_side")
        extract_from_df(damages_df, "victim_steamid", "victim_name", "victim_side")

        return names, teams

    def _calculate_stats(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        player_names: dict[int, str],
        player_teams: dict[int, str],
        num_rounds: int
    ) -> dict[int, dict]:
        """Calculate basic player statistics."""
        stats: dict[int, dict] = {}
        num_rounds = max(num_rounds, 1)

        for steam_id, name in player_names.items():
            kills = 0
            deaths = 0
            assists = 0
            headshots = 0
            total_damage = 0
            weapon_kills: dict[str, int] = {}

            # Count kills
            if not kills_df.empty and "attacker_steamid" in kills_df.columns:
                player_kills = kills_df[kills_df["attacker_steamid"] == steam_id]
                kills = len(player_kills)
                if "headshot" in kills_df.columns and kills > 0:
                    headshots = int(player_kills["headshot"].sum())
                if "weapon" in kills_df.columns and kills > 0:
                    weapon_kills = player_kills["weapon"].value_counts().to_dict()

            # Count deaths - check for victim_steamid or user_steamid columns
            has_victim = "victim_steamid" in kills_df.columns
            victim_col = "victim_steamid" if has_victim else "user_steamid"
            if not kills_df.empty and victim_col in kills_df.columns:
                deaths = len(kills_df[kills_df[victim_col] == steam_id])

            # Count assists
            if not kills_df.empty and "assister_steamid" in kills_df.columns:
                assists = len(kills_df[kills_df["assister_steamid"] == steam_id])

            # Calculate total damage
            dmg_col = None
            for col in ["damage", "health_damage", "dmg_health"]:
                if col in damages_df.columns:
                    dmg_col = col
                    break

            if not damages_df.empty and dmg_col and "attacker_steamid" in damages_df.columns:
                player_dmg = damages_df[damages_df["attacker_steamid"] == steam_id]
                total_damage = int(player_dmg[dmg_col].sum())

            stats[steam_id] = {
                "name": name,
                "team": player_teams.get(steam_id, "Unknown"),
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "headshots": headshots,
                "hs_percent": round((headshots / kills * 100), 1) if kills > 0 else 0.0,
                "total_damage": total_damage,
                "adr": round(total_damage / num_rounds, 1),
                "kd_ratio": round(kills / deaths, 2) if deaths > 0 else float(kills),
                "weapon_kills": weapon_kills,
            }

        return stats


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_demo(
    demo_path: str | Path,
    include_ticks: bool = False,
) -> MatchData:
    """
    Convenience function to parse a demo file.

    Args:
        demo_path: Path to the .dem file
        include_ticks: If True, parse tick-level position data (slower, more memory)

    Returns:
        MatchData with all parsed events and DataFrames

    Example:
        data = parse_demo("demo.dem")

        # Iterate over kills
        for kill in data.kills:
            print(f"{kill.attacker_name} -> {kill.victim_name}")

        # Use DataFrames for analysis
        adr = data.damages_df.groupby("attacker_steamid")["damage"].sum() / data.num_rounds
    """
    parser = DemoParser(demo_path)
    return parser.parse(include_ticks=include_ticks)


# =============================================================================
# Backward Compatibility - Legacy Enums and Functions
# =============================================================================

class ParserBackend(Enum):
    """Available parser backends (legacy - awpy is now the only backend)."""
    AWPY = "awpy"
    AUTO = "auto"


class ParseMode(Enum):
    """Parsing mode (legacy - awpy always parses comprehensively)."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


# Legacy dataframe optimization (no longer needed with awpy)
def optimize_dataframe_dtypes(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """Legacy function - returns DataFrame unchanged (awpy handles optimization)."""
    return df


# Legacy aliases
PlayerState = None  # Was PlayerRoundSnapshot
PlayerRoundSnapshot = None
WeaponFireEvent_legacy = WeaponFireEvent
BlindEvent_legacy = BlindEvent
