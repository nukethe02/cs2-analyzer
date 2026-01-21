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
    tick_rate: float
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
    smokes_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    infernos_df: pd.DataFrame = field(default_factory=pd.DataFrame)
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
        "X",
        "Y",
        "Z",
        "pitch",
        "yaw",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
        "health",
        "armor_value",
        "is_alive",
        "is_scoped",
        "balance",
        "current_equip_value",
        "last_place_name",
        "in_crouch",
        "is_walking",
        "has_helmet",
        "has_defuser",
    ]

    # Memory-efficient dtypes for DataFrame columns
    OPTIMIZED_DTYPES = {
        "tick": "int32",
        "round": "int16",
        "round_num": "int16",
        "total_rounds_played": "int16",
        "dmg_health": "int16",
        "dmg_armor": "int16",
        "damage": "int16",
        "health": "int16",
        "armor": "int16",
        "health_remaining": "int16",
        "armor_remaining": "int16",
        "headshot": "bool",
        "is_alive": "bool",
        "is_scoped": "bool",
        "flash_assist": "bool",
    }

    # Chunk size for processing large tick data
    TICK_CHUNK_SIZE = 500000

    def __init__(self, demo_path: str | Path):
        """
        Initialize the demo parser.

        Args:
            demo_path: Path to the .dem file
        """
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        self._data: Optional[DemoData] = None
        self._parser: Optional[Demoparser2] = None
        # Cache for column lookups
        self._column_cache: dict[str, Optional[str]] = {}

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

        # SAFETY: awpy grenades can contain tick-level trajectory data (millions of rows)
        # We only need one row per grenade throw, so deduplicate by entity_id
        if not grenades_df.empty and len(grenades_df) > 1000:
            original_len = len(grenades_df)

            # Try to deduplicate by entity_id (get first occurrence = throw event)
            if "entity_id" in grenades_df.columns:
                grenades_df = grenades_df.drop_duplicates(subset=["entity_id"], keep="first")
                logger.info(
                    f"Deduplicated grenades by entity_id: {original_len} -> {len(grenades_df)} rows"
                )

            # If still too large, apply hard cap to prevent memory exhaustion
            if len(grenades_df) > 10000:
                logger.warning(
                    f"Grenade dataset still large ({len(grenades_df)} rows). Limiting to 10,000 to prevent crash."
                )
                grenades_df = grenades_df.head(10000)

        bomb_df = demo.bomb.to_pandas() if demo.bomb is not None else pd.DataFrame()
        shots_df = demo.shots.to_pandas() if demo.shots is not None else pd.DataFrame()
        ticks_df = demo.ticks.to_pandas() if include_ticks and demo.ticks is not None else None

        # Log parsed data sizes
        logger.info(
            f"Parsed: {len(kills_df)} kills, {len(damages_df)} damages, "
            f"{len(rounds_df)} rounds, {len(grenades_df)} grenades, "
            f"{len(bomb_df)} bomb events, {len(shots_df)} shots"
        )
        if kills_df.empty:
            # Fallback without player props
            kills_df = self._parse_event_safe(parser, "player_death")
        logger.info(f"Parsed {len(kills_df)} kills. Columns: {list(kills_df.columns)[:15]}...")

        # Parse damages with hitgroup data
        damages_df = self._parse_event_safe(parser, "player_hurt")
        logger.info(f"Parsed {len(damages_df)} damage events")

        # Parse round events
        round_end_df = self._parse_event_safe(parser, "round_end")
        round_start_df = self._parse_event_safe(parser, "round_start")
        round_freeze_df = self._parse_event_safe(parser, "round_freeze_end")
        logger.info(f"Parsed {len(round_end_df)} round_end, {len(round_start_df)} round_start events")

        # ===========================================
        # EXTENDED EVENTS - For comprehensive analysis
        # ===========================================
        weapon_fires_df = pd.DataFrame()
        blinds_df = pd.DataFrame()
        grenades_thrown_df = pd.DataFrame()
        flash_det_df = pd.DataFrame()
        he_det_df = pd.DataFrame()
        smoke_det_df = pd.DataFrame()
        molly_det_df = pd.DataFrame()
        inferno_start_df = pd.DataFrame()
        inferno_end_df = pd.DataFrame()
        bomb_planted_df = pd.DataFrame()
        bomb_defused_df = pd.DataFrame()
        bomb_exploded_df = pd.DataFrame()

        if comprehensive:
            # Weapon fire events (for accuracy tracking)
            weapon_fires_df = self._parse_event_safe(
                parser, "weapon_fire",
                player_props=["X", "Y", "Z", "pitch", "yaw", "velocity_X", "velocity_Y", "velocity_Z", "is_scoped"]
            )
            if weapon_fires_df.empty:
                weapon_fires_df = self._parse_event_safe(parser, "weapon_fire")
            logger.info(f"Parsed {len(weapon_fires_df)} weapon_fire events (for accuracy)")

            # Player blind events (flash effectiveness)
            blinds_df = self._parse_event_safe(parser, "player_blind")
            logger.info(f"Parsed {len(blinds_df)} player_blind events")

            # Grenade events
            grenades_thrown_df = self._parse_event_safe(parser, "grenade_thrown")
            flash_det_df = self._parse_event_safe(parser, "flashbang_detonate")
            he_det_df = self._parse_event_safe(parser, "hegrenade_detonate")
            smoke_det_df = self._parse_event_safe(parser, "smokegrenade_detonate")
            molly_det_df = self._parse_event_safe(parser, "molotov_detonate")
            inferno_start_df = self._parse_event_safe(parser, "inferno_startburn")
            inferno_end_df = self._parse_event_safe(parser, "inferno_expire")
            logger.info(f"Parsed grenades: {len(grenades_thrown_df)} thrown, {len(flash_det_df)} flash, {len(he_det_df)} HE, {len(smoke_det_df)} smoke, {len(molly_det_df)} molly")

            # Bomb events
            bomb_planted_df = self._parse_event_safe(parser, "bomb_planted")
            bomb_defused_df = self._parse_event_safe(parser, "bomb_defused")
            bomb_exploded_df = self._parse_event_safe(parser, "bomb_exploded")
            logger.info(f"Parsed bomb events: {len(bomb_planted_df)} plants, {len(bomb_defused_df)} defuses, {len(bomb_exploded_df)} explosions")

        # ===========================================
        # TICK DATA - Optional detailed tracking
        # ===========================================
        ticks_df = None
        if include_ticks:
            ticks_df = self._process_tick_data_chunked(parser, self.PLAYER_PROPS)
            if ticks_df is not None:
                logger.info(f"Parsed {len(ticks_df)} tick entries (memory optimized)")

        # Optimize dtypes for memory efficiency
        kills_df = self._optimize_dtypes(kills_df)
        damages_df = self._optimize_dtypes(damages_df)
        weapon_fires_df = self._optimize_dtypes(weapon_fires_df) if comprehensive else weapon_fires_df
        blinds_df = self._optimize_dtypes(blinds_df) if comprehensive else blinds_df

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
        rounds = self._build_rounds(round_end_df, round_start_df, round_freeze_df)

        # Build structured events - EXTENDED
        weapon_fires = self._build_weapon_fires(weapon_fires_df) if comprehensive else []
        blinds = self._build_blinds(blinds_df, player_teams) if comprehensive else []
        grenades = self._build_grenades(grenades_thrown_df, flash_det_df, he_det_df, smoke_det_df, molly_det_df) if comprehensive else []
        bomb_events = self._build_bomb_events(bomb_planted_df, bomb_defused_df, bomb_exploded_df) if comprehensive else []

        # Merge grenade DataFrames efficiently (only include non-empty ones)
        grenades_df = pd.DataFrame()
        if comprehensive:
            grenade_frames = []
            if not grenades_thrown_df.empty:
                grenade_frames.append(grenades_thrown_df.assign(event_type='thrown'))
            if not flash_det_df.empty:
                grenade_frames.append(flash_det_df.assign(event_type='detonate', grenade_type='flashbang'))
            if not he_det_df.empty:
                grenade_frames.append(he_det_df.assign(event_type='detonate', grenade_type='hegrenade'))
            if not smoke_det_df.empty:
                grenade_frames.append(smoke_det_df.assign(event_type='detonate', grenade_type='smokegrenade'))
            if not molly_det_df.empty:
                grenade_frames.append(molly_det_df.assign(event_type='detonate', grenade_type='molotov'))
            if grenade_frames:
                grenades_df = pd.concat(grenade_frames, ignore_index=True)

        # Merge bomb DataFrames efficiently
        bomb_events_df = pd.DataFrame()
        if comprehensive:
            bomb_frames = []
            if not bomb_planted_df.empty:
                bomb_frames.append(bomb_planted_df.assign(event_type='planted'))
            if not bomb_defused_df.empty:
                bomb_frames.append(bomb_defused_df.assign(event_type='defused'))
            if not bomb_exploded_df.empty:
                bomb_frames.append(bomb_exploded_df.assign(event_type='exploded'))
            if bomb_frames:
                bomb_events_df = pd.concat(bomb_frames, ignore_index=True)

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
            smokes_df=smokes_df,
            infernos_df=infernos_df,
            weapon_fires_df=shots_df,
            ticks_df=ticks_df,
            # Scores
            final_score_ct=final_ct,
            final_score_t=final_t,
        )

        logger.info(
            f"Parsing complete: {len(player_stats)} players, {num_rounds} rounds, "
            f"{len(kills)} kills, {final_ct}-{final_t}"
        )
        return self._data

    def _find_column(self, df: pd.DataFrame, options: list[str], cache_key: str = None) -> Optional[str]:
        """Find first matching column from options with caching."""
        # Use cache if available
        if cache_key and cache_key in self._column_cache:
            return self._column_cache[cache_key]

        for col in options:
            if col in df.columns:
                if cache_key:
                    self._column_cache[cache_key] = col
                return col

        if cache_key:
            self._column_cache[cache_key] = None
        return None

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting dtypes."""
        if df.empty:
            return df

        for col, dtype in self.OPTIMIZED_DTYPES.items():
            if col in df.columns:
                try:
                    if dtype == "bool":
                        df[col] = df[col].astype(bool)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                except (ValueError, TypeError):
                    pass  # Keep original dtype if conversion fails

        return df

    def _process_tick_data_chunked(self, parser: 'Demoparser2', props: list[str]) -> Optional[pd.DataFrame]:
        """Process tick data in chunks for memory efficiency."""
        try:
            ticks_df = parser.parse_ticks(props)
            if ticks_df is None or ticks_df.empty:
                return None

            # If data is large, process in chunks and optimize dtypes
            if len(ticks_df) > self.TICK_CHUNK_SIZE:
                logger.info(f"Processing {len(ticks_df)} ticks in chunks for memory efficiency")
                chunks = []
                for start in range(0, len(ticks_df), self.TICK_CHUNK_SIZE):
                    chunk = ticks_df.iloc[start:start + self.TICK_CHUNK_SIZE].copy()
                    chunk = self._optimize_dtypes(chunk)
                    chunks.append(chunk)
                ticks_df = pd.concat(chunks, ignore_index=True)
            else:
                ticks_df = self._optimize_dtypes(ticks_df)

            return ticks_df
        except Exception as e:
            logger.warning(f"Failed to parse ticks: {e}")
            return None

    def _extract_players(self, kills_df: pd.DataFrame, damages_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
        """Extract player names and teams from DataFrames using vectorized operations."""
        names: dict[int, str] = {}
        teams: dict[int, str] = {}

        # Column name variations
        att_id_cols = ["attacker_steamid", "attacker_steam_id"]
        att_name_cols = ["attacker_name"]
        att_team_cols = ["attacker_team_name", "attacker_side", "attacker_team"]
        vic_id_cols = ["user_steamid", "victim_steamid", "victim_steam_id"]
        vic_name_cols = ["user_name", "victim_name"]
        vic_team_cols = ["user_team_name", "victim_side", "victim_team"]

        def parse_team(team_val) -> str:
            """Parse team value to CT/T/Unknown."""
            if pd.isna(team_val):
                return "Unknown"
            if isinstance(team_val, str):
                team_upper = team_val.upper()
                if "CT" in team_upper:
                    return "CT"
                elif "T" in team_upper:
                    return "T"
                return team_val
            elif isinstance(team_val, (int, float)):
                return "CT" if int(team_val) == 3 else "T" if int(team_val) == 2 else "Unknown"
            return "Unknown"

        def extract_from_df_vectorized(df, id_cols, name_cols, team_cols):
            """Extract player info using vectorized operations."""
            if df is None or df.empty:
                return

            id_col = self._find_column(df, id_cols)
            name_col = self._find_column(df, name_cols)
            team_col = self._find_column(df, team_cols)

            if not id_col or not name_col:
                return

            # Get unique player records efficiently
            unique_df = df[[id_col, name_col] + ([team_col] if team_col else [])].drop_duplicates(subset=[id_col])

            # Filter out invalid steamids
            unique_df = unique_df[unique_df[id_col].notna() & (unique_df[id_col] != 0)]

            # Vectorized extraction
            for _, row in unique_df.iterrows():
                sid = safe_int(row[id_col])
                if sid and sid not in names:
                    names[sid] = safe_str(row[name_col])
                    if team_col:
                        teams[sid] = parse_team(row[team_col])

        # Extract from kills (attackers and victims)
        extract_from_df_vectorized(kills_df, att_id_cols, att_name_cols, att_team_cols)
        extract_from_df_vectorized(kills_df, vic_id_cols, vic_name_cols, vic_team_cols)
        # Extract from damages
        extract_from_df_vectorized(damages_df, att_id_cols, att_name_cols, att_team_cols)
        extract_from_df_vectorized(damages_df, vic_id_cols, vic_name_cols, vic_team_cols)

        return names, teams

    def _calculate_stats(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        player_names: dict[int, str],
        player_teams: dict[int, str],
        num_rounds: int,
    ) -> dict[int, dict]:
        """Calculate player statistics using vectorized operations."""
        stats: dict[int, dict] = {}
        num_rounds = max(num_rounds, 1)

        # Early return if no kills data
        if kills_df.empty and (damages_df is None or damages_df.empty):
            for steam_id, name in player_names.items():
                stats[steam_id] = {
                    "name": name,
                    "team": player_teams.get(steam_id, "Unknown"),
                    "kills": 0, "deaths": 0, "assists": 0, "headshots": 0,
                    "hs_percent": 0.0, "total_damage": 0, "adr": 0.0,
                    "kd_ratio": 0.0, "weapon_kills": {},
                }
            return stats

        # Find columns with caching
        att_col = self._find_column(kills_df, ["attacker_steamid", "attacker_steam_id"], "kills_att_id")
        vic_col = self._find_column(kills_df, ["user_steamid", "victim_steamid", "victim_steam_id"], "kills_vic_id")
        hs_col = self._find_column(kills_df, ["headshot"], "kills_hs")
        weapon_col = self._find_column(kills_df, ["weapon"], "kills_weapon")
        assist_col = self._find_column(kills_df, ["assister_steamid", "assister_steam_id"], "kills_assist")
        dmg_att_col = self._find_column(damages_df, ["attacker_steamid", "attacker_steam_id"], "dmg_att_id") if not damages_df.empty else None
        dmg_col = self._find_column(damages_df, ["dmg_health", "damage", "dmg"], "dmg_val") if not damages_df.empty else None

        # Pre-compute aggregated stats using groupby for efficiency
        kills_by_player = {}
        deaths_by_player = {}
        assists_by_player = {}
        headshots_by_player = {}
        damage_by_player = {}
        weapon_kills_by_player = {}

        if not kills_df.empty:
            if att_col:
                # Convert to numeric for groupby
                kills_df_numeric = kills_df.copy()
                kills_df_numeric[att_col] = pd.to_numeric(kills_df_numeric[att_col], errors='coerce')

                # Count kills per player
                kills_by_player = kills_df_numeric.groupby(att_col).size().to_dict()

                # Count headshots per player
                if hs_col:
                    headshots_by_player = kills_df_numeric.groupby(att_col)[hs_col].sum().to_dict()

                # Weapon kills per player
                if weapon_col:
                    for steam_id in player_names.keys():
                        player_kills_df = kills_df_numeric[kills_df_numeric[att_col] == float(steam_id)]
                        if not player_kills_df.empty:
                            weapon_kills_by_player[steam_id] = player_kills_df[weapon_col].value_counts().to_dict()

            if vic_col:
                kills_df_numeric = kills_df.copy()
                kills_df_numeric[vic_col] = pd.to_numeric(kills_df_numeric[vic_col], errors='coerce')
                deaths_by_player = kills_df_numeric.groupby(vic_col).size().to_dict()

            if assist_col and assist_col in kills_df.columns:
                kills_df_numeric = kills_df.copy()
                kills_df_numeric[assist_col] = pd.to_numeric(kills_df_numeric[assist_col], errors='coerce')
                assists_by_player = kills_df_numeric.groupby(assist_col).size().to_dict()

        if not damages_df.empty and dmg_att_col and dmg_col:
            damages_numeric = damages_df.copy()
            damages_numeric[dmg_att_col] = pd.to_numeric(damages_numeric[dmg_att_col], errors='coerce')
            damage_by_player = damages_numeric.groupby(dmg_att_col)[dmg_col].sum().to_dict()

        # Build stats dict from pre-computed values
        for steam_id, name in player_names.items():
            steam_id_float = float(steam_id)
            kills = int(kills_by_player.get(steam_id_float, 0))
            deaths = int(deaths_by_player.get(steam_id_float, 0))
            assists = int(assists_by_player.get(steam_id_float, 0))
            headshots = int(headshots_by_player.get(steam_id_float, 0))
            total_damage = int(damage_by_player.get(steam_id_float, 0))
            weapon_kills = weapon_kills_by_player.get(steam_id, {})

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
