"""
Demo Parser for CS2 Replay Files - COMPREHENSIVE EDITION

Extracts ALL available data from CS2 demos using demoparser2:
- Kills, Deaths, Assists with position/angle data
- All damage events with hitgroups
- Weapon fire events (for accuracy tracking)
- Player blind events (flash effectiveness)
- All grenade events (flash, HE, smoke, molotov)
- Bomb events (plant, defuse, explode)
- Round events with economy data
- Player positions and velocities (for movement analysis)

This parser aims to extract the same level of detail you would get
from watching the entire demo and taking comprehensive notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
import logging
import time

import pandas as pd
import numpy as np

from opensight.profiling import stage_timer, get_timing_collector

# Type checking imports
if TYPE_CHECKING:
    from demoparser2 import DemoParser as Demoparser2

# Try demoparser2 first (more control over what data we extract)
try:
    from demoparser2 import DemoParser as Demoparser2
    DEMOPARSER2_AVAILABLE = True
except ImportError:
    DEMOPARSER2_AVAILABLE = False

# awpy as fallback
try:
    from awpy import Demo
    AWPY_AVAILABLE = True
except ImportError:
    AWPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# Safe type conversion helpers
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


@dataclass
class KillEvent:
    """A kill event with timing, position, and angle data."""
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
    assister_steamid: Optional[int] = None
    assister_name: Optional[str] = None
    flash_assist: bool = False
    # Attacker position and view angles
    attacker_x: Optional[float] = None
    attacker_y: Optional[float] = None
    attacker_z: Optional[float] = None
    attacker_pitch: Optional[float] = None
    attacker_yaw: Optional[float] = None
    # Victim position
    victim_x: Optional[float] = None
    victim_y: Optional[float] = None
    victim_z: Optional[float] = None
    # Distance
    distance: Optional[float] = None


@dataclass
class DamageEvent:
    """A damage event with full details."""
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
class WeaponFireEvent:
    """A weapon fire event for accuracy tracking."""
    tick: int
    round_num: int
    player_steamid: int
    player_name: str
    player_side: str
    weapon: str
    # Position and aim data
    player_x: Optional[float] = None
    player_y: Optional[float] = None
    player_z: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    # Movement state (for counter-strafing analysis)
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    velocity_z: Optional[float] = None
    is_scoped: bool = False


@dataclass
class BlindEvent:
    """A player blind event from flashbang."""
    tick: int
    round_num: int
    attacker_steamid: int  # Who threw the flash
    attacker_name: str
    attacker_side: str
    victim_steamid: int  # Who got flashed
    victim_name: str
    victim_side: str
    blind_duration: float  # Duration in seconds
    is_teammate: bool = False


@dataclass
class GrenadeEvent:
    """A grenade event (thrown or detonated)."""
    tick: int
    round_num: int
    player_steamid: int
    player_name: str
    player_side: str
    grenade_type: str  # 'flashbang', 'hegrenade', 'smokegrenade', 'molotov', 'incgrenade', 'decoy'
    event_type: str  # 'thrown', 'detonate', 'expire'
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    # For HE grenades
    damage_dealt: int = 0
    enemies_hit: int = 0
    teammates_hit: int = 0


@dataclass
class BombEvent:
    """A bomb-related event."""
    tick: int
    round_num: int
    player_steamid: int
    player_name: str
    event_type: str  # 'beginplant', 'planted', 'begindefuse', 'defused', 'exploded', 'dropped', 'pickup'
    site: str = ""  # 'A' or 'B'
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


@dataclass
class RoundInfo:
    """Comprehensive round information."""
    round_num: int
    start_tick: int
    end_tick: int
    freeze_end_tick: int
    winner: str  # "CT" or "T"
    reason: str  # 'bomb_defused', 'target_bombed', 'elimination', 'time_expired'
    ct_score: int = 0
    t_score: int = 0
    # Economy data
    ct_team_money: int = 0
    t_team_money: int = 0
    ct_equipment_value: int = 0
    t_equipment_value: int = 0
    # Round type classification
    round_type: str = ""  # 'pistol', 'eco', 'force', 'full_buy'


@dataclass
class PlayerRoundSnapshot:
    """Player state snapshot at a point in time."""
    tick: int
    round_num: int
    steamid: int
    name: str
    side: str
    # Position
    x: float
    y: float
    z: float
    # View angles
    pitch: float
    yaw: float
    # Movement
    velocity_x: float
    velocity_y: float
    velocity_z: float
    # State
    health: int
    armor: int
    is_alive: bool
    is_scoped: bool
    is_walking: bool
    is_crouching: bool
    # Economy
    money: int
    equipment_value: int
    # Location
    place_name: str  # e.g., "BombsiteA", "CTSpawn", "LongA"


@dataclass
class DemoData:
    """Complete parsed demo data - comprehensive edition."""
    file_path: Path
    map_name: str
    duration_seconds: float
    tick_rate: int
    num_rounds: int
    server_name: str = ""
    game_mode: str = ""  # 'competitive', 'premier', 'casual'

    # Player info
    player_stats: dict[int, dict] = field(default_factory=dict)
    player_names: dict[int, str] = field(default_factory=dict)
    player_teams: dict[int, str] = field(default_factory=dict)

    # Core Events
    kills: list[KillEvent] = field(default_factory=list)
    damages: list[DamageEvent] = field(default_factory=list)
    rounds: list[RoundInfo] = field(default_factory=list)

    # Extended Events (NEW - for comprehensive analysis)
    weapon_fires: list[WeaponFireEvent] = field(default_factory=list)
    blinds: list[BlindEvent] = field(default_factory=list)
    grenades: list[GrenadeEvent] = field(default_factory=list)
    bomb_events: list[BombEvent] = field(default_factory=list)

    # DataFrames for detailed analysis
    kills_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    damages_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    rounds_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    weapon_fires_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    blinds_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    grenades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bomb_events_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ticks_df: Optional[pd.DataFrame] = None

    # Match summary
    final_score_ct: int = 0
    final_score_t: int = 0


class DemoParser:
    """
    Comprehensive CS2 demo parser using demoparser2.

    Extracts ALL available data for complete match analysis:
    - Kills with position, angles, wallbangs, noscopes, thrusmoke
    - Damage events with hitgroups
    - Weapon fire events (for accuracy tracking)
    - Player blind events (flash effectiveness)
    - Grenade events (all types)
    - Bomb events (plant, defuse, etc.)
    - Round data with economy
    - Player positions/velocities (tick data)
    """

    # Comprehensive player properties to extract
    PLAYER_PROPS = [
        "X", "Y", "Z",                    # Position
        "pitch", "yaw",                    # View angles
        "velocity_X", "velocity_Y", "velocity_Z",  # Movement
        "health", "armor_value",           # Health/armor
        "is_alive", "is_scoped",           # State
        "balance", "current_equip_value",  # Economy
        "last_place_name",                 # Location name
        "in_crouch", "is_walking",         # Movement state
    ]

    # Events to parse
    EVENTS_TO_PARSE = [
        "player_death",      # Kills
        "player_hurt",       # Damage
        "weapon_fire",       # Shots fired (for accuracy)
        "player_blind",      # Flash effectiveness
        "grenade_thrown",    # All grenade throws
        "flashbang_detonate",
        "hegrenade_detonate",
        "smokegrenade_detonate",
        "molotov_detonate",
        "inferno_startburn",
        "inferno_expire",
        "bomb_planted",
        "bomb_defused",
        "bomb_exploded",
        "bomb_dropped",
        "bomb_pickup",
        "bomb_beginplant",
        "bomb_begindefuse",
        "round_start",
        "round_end",
        "round_freeze_end",
    ]

    def __init__(self, demo_path: str | Path):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        self._data: Optional[DemoData] = None
        self._parser: Optional[Demoparser2] = None

    def parse(self, include_ticks: bool = False, comprehensive: bool = True) -> DemoData:
        """
        Parse the demo file and extract all relevant data.

        Args:
            include_ticks: If True, parse tick-level position data (slower but more detailed)
            comprehensive: If True, parse all events including weapon_fire, grenades, blinds
        """
        if self._data is not None:
            return self._data

        if DEMOPARSER2_AVAILABLE:
            logger.info("Using demoparser2 for comprehensive parsing")
            return self._parse_with_demoparser2(include_ticks, comprehensive)
        elif AWPY_AVAILABLE:
            logger.info("Using awpy parser (fallback - limited data)")
            return self._parse_with_awpy(include_ticks)
        else:
            raise ImportError("No parser available. Install demoparser2: pip install demoparser2")

    def _parse_event_safe(self, parser: Demoparser2, event_name: str,
                          player_props: list[str] = None, other_props: list[str] = None) -> pd.DataFrame:
        """Safely parse an event, returning empty DataFrame on failure."""
        try:
            kwargs = {}
            if player_props:
                kwargs["player"] = player_props
            if other_props:
                kwargs["other"] = other_props

            df = parser.parse_event(event_name, **kwargs) if kwargs else parser.parse_event(event_name)

            if df is not None and not df.empty:
                logger.debug(f"Parsed {len(df)} {event_name} events")
                return df
        except Exception as e:
            logger.debug(f"Could not parse {event_name}: {e}")
        return pd.DataFrame()

    def _parse_with_demoparser2(self, include_ticks: bool = False, comprehensive: bool = True) -> DemoData:
        """Parse using demoparser2 with comprehensive event extraction."""
        logger.info(f"Parsing demo: {self.demo_path}")

        # Record file info for timing collector
        collector = get_timing_collector()
        if collector:
            try:
                file_size = self.demo_path.stat().st_size
                collector.set_file_info(path=str(self.demo_path), size_bytes=file_size)
            except Exception:
                pass

        with stage_timer("parser_init"):
            parser = Demoparser2(str(self.demo_path))
            self._parser = parser

        # Parse header
        map_name = "unknown"
        server_name = ""
        with stage_timer("parse_header"):
            try:
                header = parser.parse_header()
                if isinstance(header, dict):
                    map_name = header.get("map_name", "unknown")
                    server_name = header.get("server_name", "")
                    logger.info(f"Map: {map_name}, Server: {server_name}")
            except Exception as e:
                logger.warning(f"Failed to parse header: {e}")

        # ===========================================
        # CORE EVENTS - Always parse these
        # ===========================================

        with stage_timer("parse_core_events"):
            # Parse kills WITH comprehensive data
            kills_df = self._parse_event_safe(
                parser, "player_death",
                player_props=["X", "Y", "Z", "pitch", "yaw", "velocity_X", "velocity_Y", "velocity_Z"],
                other_props=["total_rounds_played"]
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
            with stage_timer("parse_extended_events"):
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
            with stage_timer("parse_tick_data"):
                try:
                    ticks_df = parser.parse_ticks(self.PLAYER_PROPS)
                    if ticks_df is not None and not ticks_df.empty:
                        logger.info(f"Parsed {len(ticks_df)} tick entries")
                        # Record tick count for timing collector
                        if collector:
                            collector.set_file_info(tick_count=len(ticks_df))
                except Exception as e:
                    logger.warning(f"Failed to parse ticks: {e}")

        # Calculate duration
        tick_rate = 64
        max_tick = 0
        if not kills_df.empty and "tick" in kills_df.columns:
            max_tick = max(max_tick, int(kills_df["tick"].max()))
        duration_seconds = max_tick / tick_rate

        # Determine round count
        num_rounds = 1
        if not round_end_df.empty:
            num_rounds = len(round_end_df)
        elif not kills_df.empty:
            round_col = self._find_column(kills_df, ["total_rounds_played", "round", "round_num"])
            if round_col:
                num_rounds = int(kills_df[round_col].max())

        # Extract player info and calculate stats
        with stage_timer("extract_players"):
            player_names, player_teams = self._extract_players(kills_df, damages_df)
            player_stats = self._calculate_stats(kills_df, damages_df, player_names, player_teams, num_rounds)

        # Build structured events - CORE
        with stage_timer("build_structured_events"):
            kills = self._build_kills(kills_df)
            damages = self._build_damages(damages_df)
            rounds = self._build_rounds(round_end_df, round_start_df, round_freeze_df)

            # Build structured events - EXTENDED
            weapon_fires = self._build_weapon_fires(weapon_fires_df) if comprehensive else []
            blinds = self._build_blinds(blinds_df, player_teams) if comprehensive else []
            grenades = self._build_grenades(grenades_thrown_df, flash_det_df, he_det_df, smoke_det_df, molly_det_df) if comprehensive else []
            bomb_events = self._build_bomb_events(bomb_planted_df, bomb_defused_df, bomb_exploded_df) if comprehensive else []

        # Merge grenade DataFrames for easier analysis
        grenades_df = pd.concat([
            grenades_thrown_df.assign(event_type='thrown') if not grenades_thrown_df.empty else pd.DataFrame(),
            flash_det_df.assign(event_type='detonate', grenade_type='flashbang') if not flash_det_df.empty else pd.DataFrame(),
            he_det_df.assign(event_type='detonate', grenade_type='hegrenade') if not he_det_df.empty else pd.DataFrame(),
            smoke_det_df.assign(event_type='detonate', grenade_type='smokegrenade') if not smoke_det_df.empty else pd.DataFrame(),
            molly_det_df.assign(event_type='detonate', grenade_type='molotov') if not molly_det_df.empty else pd.DataFrame(),
        ], ignore_index=True) if comprehensive else pd.DataFrame()

        # Merge bomb DataFrames
        bomb_events_df = pd.concat([
            bomb_planted_df.assign(event_type='planted') if not bomb_planted_df.empty else pd.DataFrame(),
            bomb_defused_df.assign(event_type='defused') if not bomb_defused_df.empty else pd.DataFrame(),
            bomb_exploded_df.assign(event_type='exploded') if not bomb_exploded_df.empty else pd.DataFrame(),
        ], ignore_index=True) if comprehensive else pd.DataFrame()

        # Calculate final scores
        final_ct = 0
        final_t = 0
        if rounds:
            for r in rounds:
                if r.winner == "CT":
                    final_ct += 1
                elif r.winner == "T":
                    final_t += 1

        self._data = DemoData(
            file_path=self.demo_path,
            map_name=map_name,
            duration_seconds=duration_seconds,
            tick_rate=tick_rate,
            num_rounds=num_rounds,
            server_name=server_name,
            player_stats=player_stats,
            player_names=player_names,
            player_teams=player_teams,
            # Core events
            kills=kills,
            damages=damages,
            rounds=rounds,
            # Extended events
            weapon_fires=weapon_fires,
            blinds=blinds,
            grenades=grenades,
            bomb_events=bomb_events,
            # DataFrames
            kills_df=kills_df,
            damages_df=damages_df,
            rounds_df=round_end_df,
            weapon_fires_df=weapon_fires_df,
            blinds_df=blinds_df,
            grenades_df=grenades_df,
            bomb_events_df=bomb_events_df,
            ticks_df=ticks_df,
            # Scores
            final_score_ct=final_ct,
            final_score_t=final_t,
        )

        logger.info(f"Parsing complete: {len(player_stats)} players, {num_rounds} rounds, {len(kills)} kills")
        if comprehensive:
            logger.info(f"Extended data: {len(weapon_fires)} shots, {len(blinds)} blinds, {len(grenades)} grenades, {len(bomb_events)} bomb events")
        return self._data

    def _find_column(self, df: pd.DataFrame, options: list[str]) -> Optional[str]:
        """Find first matching column from options."""
        for col in options:
            if col in df.columns:
                return col
        return None

    def _extract_players(self, kills_df: pd.DataFrame, damages_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
        """Extract player names and teams from DataFrames."""
        names: dict[int, str] = {}
        teams: dict[int, str] = {}

        # Column name variations
        att_id_cols = ["attacker_steamid", "attacker_steam_id"]
        att_name_cols = ["attacker_name"]
        att_team_cols = ["attacker_team_name", "attacker_side", "attacker_team"]
        vic_id_cols = ["user_steamid", "victim_steamid", "victim_steam_id"]
        vic_name_cols = ["user_name", "victim_name"]
        vic_team_cols = ["user_team_name", "victim_side", "victim_team"]

        def extract_from_df(df, id_cols, name_cols, team_cols):
            if df.empty:
                return
            id_col = self._find_column(df, id_cols)
            name_col = self._find_column(df, name_cols)
            team_col = self._find_column(df, team_cols)

            if id_col and name_col:
                for _, row in df.drop_duplicates(subset=[id_col]).iterrows():
                    sid = safe_int(row.get(id_col))
                    if sid and sid not in names:
                        names[sid] = safe_str(row.get(name_col))
                        if team_col:
                            team_val = row.get(team_col)
                            # Handle various team formats
                            if isinstance(team_val, str):
                                if "CT" in team_val.upper():
                                    teams[sid] = "CT"
                                elif "T" in team_val.upper() and "CT" not in team_val.upper():
                                    teams[sid] = "T"
                                else:
                                    teams[sid] = team_val
                            elif isinstance(team_val, (int, float)):
                                teams[sid] = "CT" if int(team_val) == 3 else "T" if int(team_val) == 2 else "Unknown"
                            else:
                                teams[sid] = "Unknown"

        # Extract from kills (attackers)
        extract_from_df(kills_df, att_id_cols, att_name_cols, att_team_cols)
        # Extract from kills (victims)
        extract_from_df(kills_df, vic_id_cols, vic_name_cols, vic_team_cols)
        # Extract from damages
        extract_from_df(damages_df, att_id_cols, att_name_cols, att_team_cols)
        extract_from_df(damages_df, vic_id_cols, vic_name_cols, vic_team_cols)

        return names, teams

    def _calculate_stats(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        player_names: dict[int, str],
        player_teams: dict[int, str],
        num_rounds: int
    ) -> dict[int, dict]:
        """Calculate player statistics."""
        stats: dict[int, dict] = {}
        num_rounds = max(num_rounds, 1)

        att_col = self._find_column(kills_df, ["attacker_steamid", "attacker_steam_id"])
        vic_col = self._find_column(kills_df, ["user_steamid", "victim_steamid", "victim_steam_id"])
        hs_col = self._find_column(kills_df, ["headshot"])
        weapon_col = self._find_column(kills_df, ["weapon"])
        assist_col = self._find_column(kills_df, ["assister_steamid", "assister_steam_id"])
        dmg_att_col = self._find_column(damages_df, ["attacker_steamid", "attacker_steam_id"])
        dmg_col = self._find_column(damages_df, ["dmg_health", "damage", "dmg"])

        for steam_id, name in player_names.items():
            kills = 0
            deaths = 0
            assists = 0
            headshots = 0
            total_damage = 0
            weapon_kills: dict[str, int] = {}

            # Use float comparison to handle int/float mismatch in steamids
            if not kills_df.empty and att_col:
                player_kills_df = kills_df[kills_df[att_col].astype(float) == float(steam_id)]
                kills = len(player_kills_df)
                if hs_col and kills > 0:
                    headshots = int(player_kills_df[hs_col].sum())
                if weapon_col and kills > 0:
                    weapon_kills = player_kills_df[weapon_col].value_counts().to_dict()

            if not kills_df.empty and vic_col:
                deaths = len(kills_df[kills_df[vic_col].astype(float) == float(steam_id)])

            if not kills_df.empty and assist_col:
                assists = len(kills_df[kills_df[assist_col].astype(float) == float(steam_id)])

            if not damages_df.empty and dmg_att_col and dmg_col:
                player_dmg = damages_df[damages_df[dmg_att_col].astype(float) == float(steam_id)]
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

    def _build_kills(self, kills_df: pd.DataFrame) -> list[KillEvent]:
        """Build KillEvent list from kills DataFrame."""
        kills = []
        if kills_df.empty:
            return kills

        # Find columns
        att_id = self._find_column(kills_df, ["attacker_steamid", "attacker_steam_id"])
        att_name = self._find_column(kills_df, ["attacker_name"])
        att_team = self._find_column(kills_df, ["attacker_team_name", "attacker_side"])
        vic_id = self._find_column(kills_df, ["user_steamid", "victim_steamid"])
        vic_name = self._find_column(kills_df, ["user_name", "victim_name"])
        vic_team = self._find_column(kills_df, ["user_team_name", "victim_side"])
        round_col = self._find_column(kills_df, ["total_rounds_played", "round", "round_num"])

        # Position columns (from player props)
        att_x = self._find_column(kills_df, ["attacker_X", "attacker_x"])
        att_y = self._find_column(kills_df, ["attacker_Y", "attacker_y"])
        att_z = self._find_column(kills_df, ["attacker_Z", "attacker_z"])
        att_pitch = self._find_column(kills_df, ["attacker_pitch"])
        att_yaw = self._find_column(kills_df, ["attacker_yaw"])
        vic_x = self._find_column(kills_df, ["user_X", "victim_X", "user_x", "victim_x"])
        vic_y = self._find_column(kills_df, ["user_Y", "victim_Y", "user_y", "victim_y"])
        vic_z = self._find_column(kills_df, ["user_Z", "victim_Z", "user_z", "victim_z"])

        logger.info(f"Position columns found - attacker: X={att_x}, Y={att_y}, Z={att_z}, pitch={att_pitch}, yaw={att_yaw}")
        logger.info(f"Position columns found - victim: X={vic_x}, Y={vic_y}, Z={vic_z}")

        for _, row in kills_df.iterrows():
            # Get team values
            att_side = "Unknown"
            if att_team:
                att_side_val = row.get(att_team)
                if isinstance(att_side_val, str):
                    att_side = "CT" if "CT" in att_side_val.upper() else "T" if "T" in att_side_val.upper() else att_side_val
                elif isinstance(att_side_val, (int, float)):
                    att_side = "CT" if int(att_side_val) == 3 else "T" if int(att_side_val) == 2 else "Unknown"

            vic_side = "Unknown"
            if vic_team:
                vic_side_val = row.get(vic_team)
                if isinstance(vic_side_val, str):
                    vic_side = "CT" if "CT" in vic_side_val.upper() else "T" if "T" in vic_side_val.upper() else vic_side_val
                elif isinstance(vic_side_val, (int, float)):
                    vic_side = "CT" if int(vic_side_val) == 3 else "T" if int(vic_side_val) == 2 else "Unknown"

            kill = KillEvent(
                tick=safe_int(row.get("tick")),
                round_num=safe_int(row.get(round_col)) if round_col else 0,
                attacker_steamid=safe_int(row.get(att_id)) if att_id else 0,
                attacker_name=safe_str(row.get(att_name)) if att_name else "",
                attacker_side=att_side,
                victim_steamid=safe_int(row.get(vic_id)) if vic_id else 0,
                victim_name=safe_str(row.get(vic_name)) if vic_name else "",
                victim_side=vic_side,
                weapon=safe_str(row.get("weapon", "")),
                headshot=safe_bool(row.get("headshot")),
                assister_steamid=safe_int(row.get("assister_steamid")) if row.get("assister_steamid") else None,
                assister_name=safe_str(row.get("assister_name")) if row.get("assister_name") else None,
                flash_assist=safe_bool(row.get("flash_assist")),
                # Position data
                attacker_x=safe_float(row.get(att_x)) if att_x and row.get(att_x) is not None else None,
                attacker_y=safe_float(row.get(att_y)) if att_y and row.get(att_y) is not None else None,
                attacker_z=safe_float(row.get(att_z)) if att_z and row.get(att_z) is not None else None,
                attacker_pitch=safe_float(row.get(att_pitch)) if att_pitch and row.get(att_pitch) is not None else None,
                attacker_yaw=safe_float(row.get(att_yaw)) if att_yaw and row.get(att_yaw) is not None else None,
                victim_x=safe_float(row.get(vic_x)) if vic_x and row.get(vic_x) is not None else None,
                victim_y=safe_float(row.get(vic_y)) if vic_y and row.get(vic_y) is not None else None,
                victim_z=safe_float(row.get(vic_z)) if vic_z and row.get(vic_z) is not None else None,
            )
            kills.append(kill)

        # Log position data availability
        kills_with_pos = sum(1 for k in kills if k.attacker_x is not None)
        logger.info(f"Built {len(kills)} kill events, {kills_with_pos} have position data")

        return kills

    def _build_damages(self, damages_df: pd.DataFrame) -> list[DamageEvent]:
        """Build DamageEvent list from damages DataFrame."""
        damages = []
        if damages_df.empty:
            return damages

        att_id = self._find_column(damages_df, ["attacker_steamid", "attacker_steam_id"])
        att_name = self._find_column(damages_df, ["attacker_name"])
        att_team = self._find_column(damages_df, ["attacker_team_name", "attacker_side"])
        vic_id = self._find_column(damages_df, ["user_steamid", "victim_steamid"])
        vic_name = self._find_column(damages_df, ["user_name", "victim_name"])
        vic_team = self._find_column(damages_df, ["user_team_name", "victim_side"])
        dmg_col = self._find_column(damages_df, ["dmg_health", "damage", "dmg"])
        round_col = self._find_column(damages_df, ["total_rounds_played", "round", "round_num"])

        for _, row in damages_df.iterrows():
            att_side = "Unknown"
            if att_team:
                att_side_val = row.get(att_team)
                if isinstance(att_side_val, str):
                    att_side = "CT" if "CT" in att_side_val.upper() else "T" if "T" in att_side_val.upper() else att_side_val

            vic_side = "Unknown"
            if vic_team:
                vic_side_val = row.get(vic_team)
                if isinstance(vic_side_val, str):
                    vic_side = "CT" if "CT" in vic_side_val.upper() else "T" if "T" in vic_side_val.upper() else vic_side_val

            dmg = DamageEvent(
                tick=safe_int(row.get("tick")),
                round_num=safe_int(row.get(round_col)) if round_col else 0,
                attacker_steamid=safe_int(row.get(att_id)) if att_id else 0,
                attacker_name=safe_str(row.get(att_name)) if att_name else "",
                attacker_side=att_side,
                victim_steamid=safe_int(row.get(vic_id)) if vic_id else 0,
                victim_name=safe_str(row.get(vic_name)) if vic_name else "",
                victim_side=vic_side,
                damage=safe_int(row.get(dmg_col)) if dmg_col else 0,
                damage_armor=safe_int(row.get("dmg_armor", 0)),
                health_remaining=safe_int(row.get("health", row.get("health_remaining", 0))),
                armor_remaining=safe_int(row.get("armor", row.get("armor_remaining", 0))),
                weapon=safe_str(row.get("weapon", "")),
                hitgroup=safe_str(row.get("hitgroup", "generic")),
            )
            damages.append(dmg)

        return damages

    def _build_rounds(self, rounds_df: pd.DataFrame,
                      round_start_df: pd.DataFrame = None,
                      round_freeze_df: pd.DataFrame = None) -> list[RoundInfo]:
        """Build RoundInfo list from rounds DataFrame."""
        rounds = []
        if rounds_df.empty:
            return rounds

        for idx, row in rounds_df.iterrows():
            # Determine winner from reason or winner column
            winner = "Unknown"
            reason = safe_str(row.get("reason", ""))
            winner_col = row.get("winner")

            if winner_col is not None:
                if isinstance(winner_col, str):
                    winner = "CT" if "CT" in winner_col.upper() else "T" if "T" in winner_col.upper() else winner_col
                elif isinstance(winner_col, (int, float)):
                    winner = "CT" if int(winner_col) == 3 else "T" if int(winner_col) == 2 else "Unknown"
            elif reason:
                # Infer from reason
                ct_reasons = ["bomb_defused", "ct_win", "target_saved"]
                t_reasons = ["target_bombed", "terrorist_win", "t_win"]
                reason_lower = reason.lower()
                if any(r in reason_lower for r in ct_reasons):
                    winner = "CT"
                elif any(r in reason_lower for r in t_reasons):
                    winner = "T"

            # Get start tick from round_start_df if available
            start_tick = safe_int(row.get("start_tick", 0))
            freeze_end_tick = 0
            if round_start_df is not None and not round_start_df.empty:
                matching_start = round_start_df[round_start_df.index == idx]
                if not matching_start.empty:
                    start_tick = safe_int(matching_start.iloc[0].get("tick", start_tick))
            if round_freeze_df is not None and not round_freeze_df.empty:
                matching_freeze = round_freeze_df[round_freeze_df.index == idx]
                if not matching_freeze.empty:
                    freeze_end_tick = safe_int(matching_freeze.iloc[0].get("tick", 0))

            round_info = RoundInfo(
                round_num=idx + 1,
                start_tick=start_tick,
                end_tick=safe_int(row.get("tick", row.get("end_tick", 0))),
                freeze_end_tick=freeze_end_tick,
                winner=winner,
                reason=reason,
                ct_score=safe_int(row.get("ct_score", 0)),
                t_score=safe_int(row.get("t_score", 0)),
            )
            rounds.append(round_info)

        return rounds

    def _build_weapon_fires(self, weapon_fires_df: pd.DataFrame) -> list[WeaponFireEvent]:
        """Build WeaponFireEvent list for accuracy tracking."""
        fires = []
        if weapon_fires_df.empty:
            return fires

        # Find columns
        player_id = self._find_column(weapon_fires_df, ["user_steamid", "player_steamid", "steamid"])
        player_name_col = self._find_column(weapon_fires_df, ["user_name", "player_name", "name"])
        player_team = self._find_column(weapon_fires_df, ["user_team_name", "player_team"])
        round_col = self._find_column(weapon_fires_df, ["total_rounds_played", "round", "round_num"])
        weapon_col = self._find_column(weapon_fires_df, ["weapon"])

        # Position columns
        x_col = self._find_column(weapon_fires_df, ["user_X", "player_X", "X"])
        y_col = self._find_column(weapon_fires_df, ["user_Y", "player_Y", "Y"])
        z_col = self._find_column(weapon_fires_df, ["user_Z", "player_Z", "Z"])
        pitch_col = self._find_column(weapon_fires_df, ["user_pitch", "player_pitch", "pitch"])
        yaw_col = self._find_column(weapon_fires_df, ["user_yaw", "player_yaw", "yaw"])
        vel_x = self._find_column(weapon_fires_df, ["user_velocity_X", "velocity_X"])
        vel_y = self._find_column(weapon_fires_df, ["user_velocity_Y", "velocity_Y"])
        vel_z = self._find_column(weapon_fires_df, ["user_velocity_Z", "velocity_Z"])
        scoped_col = self._find_column(weapon_fires_df, ["user_is_scoped", "is_scoped"])

        for _, row in weapon_fires_df.iterrows():
            side = "Unknown"
            if player_team:
                team_val = row.get(player_team)
                if isinstance(team_val, str):
                    side = "CT" if "CT" in team_val.upper() else "T" if "T" in team_val.upper() else team_val

            fire = WeaponFireEvent(
                tick=safe_int(row.get("tick")),
                round_num=safe_int(row.get(round_col)) if round_col else 0,
                player_steamid=safe_int(row.get(player_id)) if player_id else 0,
                player_name=safe_str(row.get(player_name_col)) if player_name_col else "",
                player_side=side,
                weapon=safe_str(row.get(weapon_col)) if weapon_col else "",
                player_x=safe_float(row.get(x_col)) if x_col else None,
                player_y=safe_float(row.get(y_col)) if y_col else None,
                player_z=safe_float(row.get(z_col)) if z_col else None,
                pitch=safe_float(row.get(pitch_col)) if pitch_col else None,
                yaw=safe_float(row.get(yaw_col)) if yaw_col else None,
                velocity_x=safe_float(row.get(vel_x)) if vel_x else None,
                velocity_y=safe_float(row.get(vel_y)) if vel_y else None,
                velocity_z=safe_float(row.get(vel_z)) if vel_z else None,
                is_scoped=safe_bool(row.get(scoped_col)) if scoped_col else False,
            )
            fires.append(fire)

        logger.info(f"Built {len(fires)} weapon fire events")
        return fires

    def _build_blinds(self, blinds_df: pd.DataFrame, player_teams: dict[int, str]) -> list[BlindEvent]:
        """Build BlindEvent list for flash effectiveness tracking."""
        blinds = []
        if blinds_df.empty:
            return blinds

        # Find columns
        att_id = self._find_column(blinds_df, ["attacker_steamid", "attacker_steam_id"])
        att_name = self._find_column(blinds_df, ["attacker_name"])
        vic_id = self._find_column(blinds_df, ["user_steamid", "userid", "victim_steamid"])
        vic_name = self._find_column(blinds_df, ["user_name", "victim_name"])
        duration_col = self._find_column(blinds_df, ["blind_duration", "duration"])
        round_col = self._find_column(blinds_df, ["total_rounds_played", "round", "round_num"])

        for _, row in blinds_df.iterrows():
            attacker_sid = safe_int(row.get(att_id)) if att_id else 0
            victim_sid = safe_int(row.get(vic_id)) if vic_id else 0

            att_side = player_teams.get(attacker_sid, "Unknown")
            vic_side = player_teams.get(victim_sid, "Unknown")
            is_teammate = att_side == vic_side and att_side != "Unknown"

            blind = BlindEvent(
                tick=safe_int(row.get("tick")),
                round_num=safe_int(row.get(round_col)) if round_col else 0,
                attacker_steamid=attacker_sid,
                attacker_name=safe_str(row.get(att_name)) if att_name else "",
                attacker_side=att_side,
                victim_steamid=victim_sid,
                victim_name=safe_str(row.get(vic_name)) if vic_name else "",
                victim_side=vic_side,
                blind_duration=safe_float(row.get(duration_col)) if duration_col else 0.0,
                is_teammate=is_teammate,
            )
            blinds.append(blind)

        logger.info(f"Built {len(blinds)} blind events")
        return blinds

    def _build_grenades(self, thrown_df: pd.DataFrame, flash_df: pd.DataFrame,
                        he_df: pd.DataFrame, smoke_df: pd.DataFrame, molly_df: pd.DataFrame) -> list[GrenadeEvent]:
        """Build GrenadeEvent list from various grenade DataFrames."""
        grenades = []

        def process_thrown(df: pd.DataFrame):
            if df.empty:
                return
            player_id = self._find_column(df, ["user_steamid", "player_steamid", "steamid"])
            player_name_col = self._find_column(df, ["user_name", "player_name"])
            player_team = self._find_column(df, ["user_team_name", "player_team"])
            grenade_type_col = self._find_column(df, ["weapon", "grenade_type", "grenade"])
            round_col = self._find_column(df, ["total_rounds_played", "round"])
            x_col = self._find_column(df, ["X", "x"])
            y_col = self._find_column(df, ["Y", "y"])
            z_col = self._find_column(df, ["Z", "z"])

            for _, row in df.iterrows():
                side = "Unknown"
                if player_team:
                    team_val = row.get(player_team)
                    if isinstance(team_val, str):
                        side = "CT" if "CT" in team_val.upper() else "T"

                grenades.append(GrenadeEvent(
                    tick=safe_int(row.get("tick")),
                    round_num=safe_int(row.get(round_col)) if round_col else 0,
                    player_steamid=safe_int(row.get(player_id)) if player_id else 0,
                    player_name=safe_str(row.get(player_name_col)) if player_name_col else "",
                    player_side=side,
                    grenade_type=safe_str(row.get(grenade_type_col)) if grenade_type_col else "unknown",
                    event_type="thrown",
                    x=safe_float(row.get(x_col)) if x_col else None,
                    y=safe_float(row.get(y_col)) if y_col else None,
                    z=safe_float(row.get(z_col)) if z_col else None,
                ))

        def process_detonate(df: pd.DataFrame, grenade_type: str):
            if df.empty:
                return
            player_id = self._find_column(df, ["user_steamid", "player_steamid", "steamid", "thrower_steamid"])
            player_name_col = self._find_column(df, ["user_name", "player_name", "thrower_name"])
            round_col = self._find_column(df, ["total_rounds_played", "round"])
            x_col = self._find_column(df, ["X", "x"])
            y_col = self._find_column(df, ["Y", "y"])
            z_col = self._find_column(df, ["Z", "z"])

            for _, row in df.iterrows():
                grenades.append(GrenadeEvent(
                    tick=safe_int(row.get("tick")),
                    round_num=safe_int(row.get(round_col)) if round_col else 0,
                    player_steamid=safe_int(row.get(player_id)) if player_id else 0,
                    player_name=safe_str(row.get(player_name_col)) if player_name_col else "",
                    player_side="Unknown",
                    grenade_type=grenade_type,
                    event_type="detonate",
                    x=safe_float(row.get(x_col)) if x_col else None,
                    y=safe_float(row.get(y_col)) if y_col else None,
                    z=safe_float(row.get(z_col)) if z_col else None,
                ))

        process_thrown(thrown_df)
        process_detonate(flash_df, "flashbang")
        process_detonate(he_df, "hegrenade")
        process_detonate(smoke_df, "smokegrenade")
        process_detonate(molly_df, "molotov")

        logger.info(f"Built {len(grenades)} grenade events")
        return grenades

    def _build_bomb_events(self, planted_df: pd.DataFrame, defused_df: pd.DataFrame,
                           exploded_df: pd.DataFrame) -> list[BombEvent]:
        """Build BombEvent list from bomb DataFrames."""
        bomb_events = []

        def process_bomb_df(df: pd.DataFrame, event_type: str):
            if df.empty:
                return
            player_id = self._find_column(df, ["user_steamid", "player_steamid", "steamid"])
            player_name_col = self._find_column(df, ["user_name", "player_name"])
            round_col = self._find_column(df, ["total_rounds_played", "round"])
            site_col = self._find_column(df, ["site", "bombsite"])
            x_col = self._find_column(df, ["X", "x"])
            y_col = self._find_column(df, ["Y", "y"])
            z_col = self._find_column(df, ["Z", "z"])

            for _, row in df.iterrows():
                bomb_events.append(BombEvent(
                    tick=safe_int(row.get("tick")),
                    round_num=safe_int(row.get(round_col)) if round_col else 0,
                    player_steamid=safe_int(row.get(player_id)) if player_id else 0,
                    player_name=safe_str(row.get(player_name_col)) if player_name_col else "",
                    event_type=event_type,
                    site=safe_str(row.get(site_col)) if site_col else "",
                    x=safe_float(row.get(x_col)) if x_col else None,
                    y=safe_float(row.get(y_col)) if y_col else None,
                    z=safe_float(row.get(z_col)) if z_col else None,
                ))

        process_bomb_df(planted_df, "planted")
        process_bomb_df(defused_df, "defused")
        process_bomb_df(exploded_df, "exploded")

        logger.info(f"Built {len(bomb_events)} bomb events")
        return bomb_events

    def _parse_with_awpy(self, include_ticks: bool = False) -> DemoData:
        """Fallback parser using awpy."""
        logger.info(f"Parsing demo with awpy: {self.demo_path}")

        demo = Demo(str(self.demo_path), verbose=False)

        # Parse with player props if we want tick data
        if include_ticks:
            demo.parse(player_props=self.PLAYER_PROPS)
        else:
            demo.parse()

        header = demo.header or {}
        map_name = header.get("map_name", "unknown")

        # Convert Polars to Pandas
        kills_df = demo.kills.to_pandas() if demo.kills is not None else pd.DataFrame()
        damages_df = demo.damages.to_pandas() if demo.damages is not None else pd.DataFrame()
        rounds_df = demo.rounds.to_pandas() if demo.rounds is not None else pd.DataFrame()
        ticks_df = demo.ticks.to_pandas() if include_ticks and demo.ticks is not None else None

        logger.info(f"awpy parsed: {len(kills_df)} kills, {len(damages_df)} damages, {len(rounds_df)} rounds")
        if not kills_df.empty:
            logger.info(f"Kill columns: {list(kills_df.columns)}")

        # Calculate stats
        tick_rate = 64
        max_tick = 0
        if not kills_df.empty and "tick" in kills_df.columns:
            max_tick = int(kills_df["tick"].max())
        duration_seconds = max_tick / tick_rate

        num_rounds = len(rounds_df) if not rounds_df.empty else 1

        player_names, player_teams = self._extract_players(kills_df, damages_df)
        player_stats = self._calculate_stats(kills_df, damages_df, player_names, player_teams, num_rounds)
        kills = self._build_kills(kills_df)
        damages = self._build_damages(damages_df)
        rounds = self._build_rounds(rounds_df)

        # Calculate scores
        final_ct = sum(1 for r in rounds if r.winner == "CT")
        final_t = sum(1 for r in rounds if r.winner == "T")

        self._data = DemoData(
            file_path=self.demo_path,
            map_name=map_name,
            duration_seconds=duration_seconds,
            tick_rate=tick_rate,
            num_rounds=num_rounds,
            player_stats=player_stats,
            player_names=player_names,
            player_teams=player_teams,
            kills=kills,
            damages=damages,
            rounds=rounds,
            kills_df=kills_df,
            damages_df=damages_df,
            rounds_df=rounds_df,
            ticks_df=ticks_df,
            final_score_ct=final_ct,
            final_score_t=final_t,
        )

        logger.info("awpy parsing complete (limited data - no weapon_fire/blinds/grenades)")
        return self._data


def parse_demo(demo_path: str | Path, include_ticks: bool = False, comprehensive: bool = True) -> DemoData:
    """
    Convenience function to parse a demo file.

    Args:
        demo_path: Path to the .dem file
        include_ticks: If True, parse tick-level position data (slower)
        comprehensive: If True, parse all events (weapon_fire, grenades, blinds, bombs)

    Returns:
        DemoData with all parsed events and DataFrames
    """
    parser = DemoParser(demo_path)
    return parser.parse(include_ticks=include_ticks, comprehensive=comprehensive)


# Alias for backward compatibility
PlayerState = PlayerRoundSnapshot
