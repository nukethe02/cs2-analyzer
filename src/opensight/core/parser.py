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

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

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
    from awpy import Demo  # noqa: F401

    AWPY_AVAILABLE = True
except ImportError:
    AWPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# Backwards-compatible enums and helpers expected by older callers/tests
class ParseMode(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class ParserBackend(Enum):
    AWPY = "awpy"
    AUTO = "auto"


def _check_awpy_available() -> bool:
    """Compatibility shim for tests that mock availability checks."""
    return AWPY_AVAILABLE


def optimize_dataframe_dtypes(df, inplace: bool = True):
    """No-op compatibility shim: returns DataFrame unchanged when awpy handles dtypes."""
    # In the compatibility mode, simply return the dataframe when not inplace.
    if inplace:
        return df
    return df


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


def is_valid_player_name(name: str) -> bool:
    """
    Check if a string is a valid player name (not a Steam ID or garbage).

    A valid player name is NOT:
    - Empty or whitespace only
    - All digits (Steam ID)
    - Very long numbers (Steam ID 64-bit format)

    Args:
        name: The name string to validate

    Returns:
        True if this looks like a real player name
    """
    if not name or not name.strip():
        return False

    # Steam IDs are typically 17+ digit numbers (SteamID64)
    # or shorter numbers. If it's all digits, it's probably an ID.
    stripped = name.strip()
    if stripped.isdigit():
        return False

    # Also catch cases like "76561198..." (SteamID64 prefix)
    if stripped.startswith("7656119") and len(stripped) >= 15:
        return False

    return True


def make_fallback_player_name(steam_id: int) -> str:
    """
    Create a friendly fallback player name from a Steam ID.

    Uses last 4 digits for readability.

    Args:
        steam_id: The player's Steam ID

    Returns:
        A name like "Player_1234"
    """
    suffix = str(steam_id)[-4:] if steam_id else "0000"
    return f"Player_{suffix}"


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
# ECONOMY HELPERS
# =============================================================================


def is_pistol_round(round_num: int, rounds_per_half: int = 12) -> bool:
    """
    Check if a round is a pistol round (first of each half/OT period).

    CS2 MR12 format:
    - Round 1: First half pistol
    - Round 13: Second half pistol
    - Round 25, 28, 31...: Overtime pistol rounds (every 6 rounds in OT for MR3)

    CS2 MR15 format (legacy):
    - Round 1: First half pistol
    - Round 16: Second half pistol
    - Round 31, 34, 37...: Overtime pistol rounds

    Args:
        round_num: The round number (1-indexed)
        rounds_per_half: Rounds per half (12 for MR12, 15 for MR15)

    Returns:
        True if this is a pistol round
    """
    if round_num <= 0:
        return False

    # First round is always pistol
    if round_num == 1:
        return True

    regulation_rounds = rounds_per_half * 2  # 24 for MR12, 30 for MR15

    # Second half pistol (round 13 for MR12, round 16 for MR15)
    if round_num == rounds_per_half + 1:
        return True

    # Overtime pistol rounds
    # OT format is MR3 (6 rounds per OT period: 3 per side)
    if round_num > regulation_rounds:
        ot_round = round_num - regulation_rounds
        # First round of each OT period (rounds 25, 31, 37... for MR12)
        # Pattern: 1, 7, 13, 19... (every 6 rounds starting from 1)
        return ot_round == 1 or (ot_round - 1) % 6 == 0

    return False


def classify_team_equipment(equipment_value: int, is_pistol: bool = False) -> str:
    """
    Classify round type based on TEAM TOTAL equipment value.

    Args:
        equipment_value: Total equipment value for the team (all 5 players)
        is_pistol: Whether this is a pistol round

    Returns:
        Round type: 'pistol', 'eco', 'semi_eco', 'force', or 'full_buy'

    Thresholds (for 5-player team):
    - Eco: < $5,000 (avg < $1,000/player)
    - Semi-Eco: $5,000 - $12,500 (avg $1,000-2,500/player)
    - Force: $12,500 - $20,000 (avg $2,500-4,000/player)
    - Full Buy: > $20,000 (avg > $4,000/player)
    """
    if is_pistol:
        return "pistol"

    if equipment_value < 5000:
        return "eco"
    elif equipment_value < 12500:
        return "semi_eco"
    elif equipment_value < 20000:
        return "force"
    else:
        return "full_buy"


# Weapon cost lookup for equipment estimation
WEAPON_COSTS_ESTIMATE = {
    # Pistols (starting/cheap)
    "glock": 0,
    "usp_silencer": 0,
    "hkp2000": 0,
    "p250": 300,
    "tec9": 500,
    "cz75a": 500,
    "fiveseven": 500,
    "deagle": 700,
    # SMGs
    "mac10": 1050,
    "mp9": 1250,
    "mp7": 1500,
    "ump45": 1200,
    "p90": 2350,
    # Rifles
    "famas": 2050,
    "galilar": 1800,
    "m4a1": 3100,
    "m4a1_silencer": 2900,
    "ak47": 2700,
    "awp": 4750,
    "ssg08": 1700,
    # Heavy
    "nova": 1050,
    "mag7": 1300,
    "negev": 1700,
}


def estimate_equipment_from_weapon(weapon: str) -> int:
    """
    Estimate a player's equipment value from their weapon.

    This is a rough estimate: weapon cost + armor (assumed if rifle).

    Args:
        weapon: Weapon name from kill event

    Returns:
        Estimated equipment value
    """
    if not weapon:
        return 800  # Default pistol round loadout

    weapon_lower = weapon.lower().replace("-", "_").replace(" ", "_")

    # Direct lookup
    weapon_cost = WEAPON_COSTS_ESTIMATE.get(weapon_lower, 0)

    # Try partial match
    if weapon_cost == 0:
        for known_weapon, cost in WEAPON_COSTS_ESTIMATE.items():
            if known_weapon in weapon_lower or weapon_lower in known_weapon:
                weapon_cost = cost
                break

    # Add armor estimate based on weapon tier
    if weapon_cost >= 2500:  # Rifle tier
        equipment_estimate = weapon_cost + 1000  # Vest + Helmet
    elif weapon_cost >= 1000:  # SMG tier
        equipment_estimate = weapon_cost + 650  # Just vest likely
    else:
        equipment_estimate = weapon_cost + 200  # Minimal or no armor

    return equipment_estimate


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
    assister_steamid: int | None = None
    assister_name: str | None = None
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
    # Time in round (seconds from freeze time end)
    time_in_round_seconds: float | None = None


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
    player_x: float | None = None
    player_y: float | None = None
    player_z: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    # Movement state (for counter-strafing analysis)
    velocity_x: float | None = None
    velocity_y: float | None = None
    velocity_z: float | None = None
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
    x: float | None = None
    y: float | None = None
    z: float | None = None
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
    event_type: (
        str  # 'beginplant', 'planted', 'begindefuse', 'defused', 'exploded', 'dropped', 'pickup'
    )
    site: str = ""  # 'A' or 'B'
    x: float | None = None
    y: float | None = None
    z: float | None = None


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
    # Optional bomb information (backwards compatibility)
    bomb_plant_tick: int | None = None
    bomb_site: str = ""
    # Economy IQ grading (per-team buy decision quality)
    ct_buy_grade: str = ""  # A-F grade for CT buying decision
    t_buy_grade: str = ""  # A-F grade for T buying decision
    ct_buy_flag: str = ""  # "ok", "bad_force", "bad_eco"
    t_buy_flag: str = ""  # "ok", "bad_force", "bad_eco"


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
    player_teams: dict[int, str] = field(default_factory=dict)  # DEPRECATED for roster ID

    # NEW: Persistent team identity (handles halftime swaps)
    player_persistent_teams: dict[int, str] = field(
        default_factory=dict
    )  # steamid -> "Team A"/"Team B"
    team_rosters: dict[str, set[int]] = field(default_factory=dict)  # "Team A" -> {steamids}
    team_starting_sides: dict[str, str] = field(default_factory=dict)  # "Team A" -> "CT"/"T"
    halftime_round: int = 13  # Round when halftime occurs (13 for MR12, 16 for MR15)

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
    ticks_df: pd.DataFrame | None = None

    # Match summary
    final_score_ct: int = 0
    final_score_t: int = 0

    def get_player_side_for_round(self, steamid: int, round_num: int) -> str:
        """Get the side (CT/T) a player was on for a specific round.

        Handles halftime swaps automatically.

        Args:
            steamid: Player's Steam ID
            round_num: Round number (1-indexed)

        Returns:
            "CT", "T", or "Unknown"
        """
        persistent_team = self.player_persistent_teams.get(steamid)
        if not persistent_team:
            # Fallback to old behavior for backward compatibility
            return self.player_teams.get(steamid, "Unknown")

        starting_side = self.team_starting_sides.get(persistent_team, "Unknown")
        if starting_side not in ["CT", "T"]:
            return "Unknown"

        # Swap sides after halftime
        if round_num >= self.halftime_round:
            return "T" if starting_side == "CT" else "CT"
        else:
            return starting_side

    def get_player_persistent_team(self, steamid: int) -> str:
        """Get the persistent team label ("Team A" or "Team B").

        Args:
            steamid: Player's Steam ID

        Returns:
            "Team A", "Team B", or "Unknown"
        """
        return self.player_persistent_teams.get(steamid, "Unknown")

    def get_team_display_name(self, persistent_team: str) -> str:
        """Map persistent team to display name based on starting side.

        This preserves frontend CSS color associations:
        - Team that started CT → "CT" (Blue)
        - Team that started T → "T" (Red)

        Args:
            persistent_team: "Team A" or "Team B"

        Returns:
            "CT", "T", or "Unknown"
        """
        starting_side = self.team_starting_sides.get(persistent_team, "Unknown")
        return starting_side if starting_side in ("CT", "T") else "Unknown"


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
        "X",
        "Y",
        "Z",  # Position
        "pitch",
        "yaw",  # View angles
        "velocity_X",
        "velocity_Y",
        "velocity_Z",  # Movement
        "health",
        "armor_value",  # Health/armor
        "is_alive",
        "is_scoped",  # State
        "balance",
        "current_equip_value",  # Economy
        "last_place_name",  # Location name
        "in_crouch",
        "is_walking",  # Movement state
    ]

    # Events to parse
    EVENTS_TO_PARSE = [
        "player_death",  # Kills
        "player_hurt",  # Damage
        "weapon_fire",  # Shots fired (for accuracy)
        "player_blind",  # Flash effectiveness
        "grenade_thrown",  # All grenade throws
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
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        self._data: DemoData | None = None
        self._parser: Demoparser2 | None = None
        # Cache for column lookups
        self._column_cache: dict[str, str | None] = {}

    def parse(self, include_ticks: bool = False, comprehensive: bool = True) -> DemoData:
        """
        Parse the demo file and extract all relevant data.

        Args:
            include_ticks: If True, parse tick-level position data (slower but more detailed)
            comprehensive: If True, parse all events including weapon_fire, grenades, blinds
        """
        if self._data is not None:
            return self._data

        # If the compatibility check explicitly indicates awpy availability=False,
        # treat that as an environment where awpy is required but missing (for tests).
        awpy_check = _check_awpy_available()
        if awpy_check is False:
            # Tests patch this to force an ImportError when awpy is expected but missing
            raise ImportError("awpy is required")

        # Prefer demoparser2 for comprehensive data extraction in production
        if DEMOPARSER2_AVAILABLE:
            logger.info("Using demoparser2 for comprehensive parsing")
            return self._parse_with_demoparser2(include_ticks, comprehensive)
        elif AWPY_AVAILABLE:
            logger.info("Using awpy parser (fallback - limited data)")
            return self._parse_with_awpy(include_ticks)
        else:
            raise ImportError("No parser available. Install demoparser2: pip install demoparser2")

    def _parse_event_safe(
        self,
        parser: Demoparser2,
        event_name: str,
        player_props: list[str] = None,
        other_props: list[str] = None,
    ) -> pd.DataFrame:
        """Safely parse an event, returning empty DataFrame on failure."""
        try:
            kwargs = {}
            if player_props:
                kwargs["player"] = player_props
            if other_props:
                kwargs["other"] = other_props

            df = (
                parser.parse_event(event_name, **kwargs)
                if kwargs
                else parser.parse_event(event_name)
            )

            if df is not None and not df.empty:
                logger.debug(f"Parsed {len(df)} {event_name} events")
                return df
        except Exception as e:
            logger.debug(f"Could not parse {event_name}: {e}")
        return pd.DataFrame()

    def _parse_with_demoparser2(
        self, include_ticks: bool = False, comprehensive: bool = True
    ) -> DemoData:
        """Parse using demoparser2 with comprehensive event extraction."""
        logger.info(f"Parsing demo: {self.demo_path}")
        parser = Demoparser2(str(self.demo_path))
        self._parser = parser

        # Parse header
        map_name = "unknown"
        server_name = ""
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

        # Parse kills WITH comprehensive data - enhanced for coaching metrics
        kills_df = self._parse_event_safe(
            parser,
            "player_death",
            player_props=[
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
            ],
            other_props=["total_rounds_played", "headshot"],
        )
        if kills_df.empty:
            # Fallback without player props
            kills_df = self._parse_event_safe(parser, "player_death")
        logger.info(
            f"Parsed {len(kills_df)} kills with positional context. Columns: {list(kills_df.columns)[:20]}..."
        )

        # Parse damages with full positional context for TTD calculation
        damages_df = self._parse_event_safe(
            parser,
            "player_hurt",
            player_props=["X", "Y", "Z", "health", "armor_value"],
            other_props=["total_rounds_played", "hitgroup"],
        )
        logger.info(f"Parsed {len(damages_df)} damage events with position context")

        # Parse weapon fires for spray analysis and accuracy metrics
        weapon_fires_df = self._parse_event_safe(
            parser,
            "weapon_fire",
            player_props=["X", "Y", "Z", "pitch", "yaw", "velocity_X", "velocity_Y"],
            other_props=["total_rounds_played", "weapon", "inaccuracy"],
        )
        logger.info(f"Parsed {len(weapon_fires_df)} weapon fire events for spray analysis")

        # Parse round events
        round_end_df = self._parse_event_safe(parser, "round_end")
        round_start_df = self._parse_event_safe(parser, "round_start")
        round_freeze_df = self._parse_event_safe(parser, "round_freeze_end")
        logger.info(
            f"Parsed {len(round_end_df)} round_end, {len(round_start_df)} round_start events"
        )

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
        pd.DataFrame()
        pd.DataFrame()
        bomb_planted_df = pd.DataFrame()
        bomb_defused_df = pd.DataFrame()
        bomb_exploded_df = pd.DataFrame()

        if comprehensive:
            # Weapon fire events (for accuracy tracking)
            weapon_fires_df = self._parse_event_safe(
                parser,
                "weapon_fire",
                player_props=[
                    "X",
                    "Y",
                    "Z",
                    "pitch",
                    "yaw",
                    "velocity_X",
                    "velocity_Y",
                    "velocity_Z",
                    "is_scoped",
                ],
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
            self._parse_event_safe(parser, "inferno_startburn")
            self._parse_event_safe(parser, "inferno_expire")
            logger.info(
                f"Parsed grenades: {len(grenades_thrown_df)} thrown, {len(flash_det_df)} flash, {len(he_det_df)} HE, {len(smoke_det_df)} smoke, {len(molly_det_df)} molly"
            )

            # Bomb events
            bomb_planted_df = self._parse_event_safe(parser, "bomb_planted")
            bomb_defused_df = self._parse_event_safe(parser, "bomb_defused")
            bomb_exploded_df = self._parse_event_safe(parser, "bomb_exploded")
            logger.info(
                f"Parsed bomb events: {len(bomb_planted_df)} plants, {len(bomb_defused_df)} defuses, {len(bomb_exploded_df)} explosions"
            )

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
        weapon_fires_df = (
            self._optimize_dtypes(weapon_fires_df) if comprehensive else weapon_fires_df
        )
        blinds_df = self._optimize_dtypes(blinds_df) if comprehensive else blinds_df

        # Calculate duration
        tick_rate = 64
        max_tick = 0
        if not kills_df.empty and "tick" in kills_df.columns:
            tick_max = kills_df["tick"].max()
            if pd.notna(tick_max):
                max_tick = max(max_tick, int(tick_max))
        duration_seconds = max_tick / tick_rate

        # Determine round count
        num_rounds = 1
        if not round_end_df.empty:
            num_rounds = len(round_end_df)
        elif not kills_df.empty:
            round_col = self._find_column(kills_df, ["total_rounds_played", "round", "round_num"])
            if round_col:
                round_max = kills_df[round_col].max()
                if pd.notna(round_max):
                    num_rounds = int(round_max)

        # Extract player info and calculate stats
        player_names, player_teams = self._extract_players(kills_df, damages_df)

        # NEW: Resolve persistent team identities (handles halftime swaps)
        (
            player_persistent_teams,
            team_rosters,
            team_starting_sides,
            halftime_round,
        ) = self._resolve_persistent_teams(kills_df, damages_df)

        player_stats = self._calculate_stats(
            kills_df, damages_df, player_names, player_teams, num_rounds
        )

        # Build structured events - CORE
        kills = self._build_kills(kills_df)
        damages = self._build_damages(damages_df)
        rounds = self._build_rounds(round_end_df, round_start_df, round_freeze_df)

        # Populate round economy data (equipment values and round_type)
        # Detect format: MR12 (24 regulation) vs MR15 (30 regulation)
        rounds_per_half = 12 if num_rounds <= 30 else 15
        self._populate_round_economy(
            rounds=rounds,
            kills=kills,
            ticks_df=ticks_df,
            player_teams=player_teams,
            rounds_per_half=rounds_per_half,
        )

        # Build structured events - EXTENDED
        weapon_fires = self._build_weapon_fires(weapon_fires_df) if comprehensive else []
        blinds = self._build_blinds(blinds_df, player_teams) if comprehensive else []
        grenades = (
            self._build_grenades(
                grenades_thrown_df, flash_det_df, he_det_df, smoke_det_df, molly_det_df
            )
            if comprehensive
            else []
        )
        bomb_events = (
            self._build_bomb_events(bomb_planted_df, bomb_defused_df, bomb_exploded_df)
            if comprehensive
            else []
        )

        # Merge grenade DataFrames efficiently (only include non-empty ones)
        grenades_df = pd.DataFrame()
        if comprehensive:
            grenade_frames = []
            if not grenades_thrown_df.empty:
                grenade_frames.append(grenades_thrown_df.assign(event_type="thrown"))
            if not flash_det_df.empty:
                grenade_frames.append(
                    flash_det_df.assign(event_type="detonate", grenade_type="flashbang")
                )
            if not he_det_df.empty:
                grenade_frames.append(
                    he_det_df.assign(event_type="detonate", grenade_type="hegrenade")
                )
            if not smoke_det_df.empty:
                grenade_frames.append(
                    smoke_det_df.assign(event_type="detonate", grenade_type="smokegrenade")
                )
            if not molly_det_df.empty:
                grenade_frames.append(
                    molly_det_df.assign(event_type="detonate", grenade_type="molotov")
                )
            if grenade_frames:
                grenades_df = pd.concat(grenade_frames, ignore_index=True)

        # Merge bomb DataFrames efficiently
        bomb_events_df = pd.DataFrame()
        if comprehensive:
            bomb_frames = []
            if not bomb_planted_df.empty:
                bomb_frames.append(bomb_planted_df.assign(event_type="planted"))
            if not bomb_defused_df.empty:
                bomb_frames.append(bomb_defused_df.assign(event_type="defused"))
            if not bomb_exploded_df.empty:
                bomb_frames.append(bomb_exploded_df.assign(event_type="exploded"))
            if bomb_frames:
                bomb_events_df = pd.concat(bomb_frames, ignore_index=True)

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
            # NEW: Persistent team identity
            player_persistent_teams=player_persistent_teams,
            team_rosters=team_rosters,
            team_starting_sides=team_starting_sides,
            halftime_round=halftime_round,
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

        logger.info(
            f"Parsing complete: {len(player_stats)} players, {num_rounds} rounds, {len(kills)} kills"
        )
        if comprehensive:
            logger.info(
                f"Extended data: {len(weapon_fires)} shots, {len(blinds)} blinds, {len(grenades)} grenades, {len(bomb_events)} bomb events"
            )
        return self._data

    def _find_column(
        self, df: pd.DataFrame, options: list[str], cache_key: str = None
    ) -> str | None:
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
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                except (ValueError, TypeError):
                    pass  # Keep original dtype if conversion fails

        return df

    def _process_tick_data_chunked(
        self, parser: Demoparser2, props: list[str]
    ) -> pd.DataFrame | None:
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
                    chunk = ticks_df.iloc[start : start + self.TICK_CHUNK_SIZE].copy()
                    chunk = self._optimize_dtypes(chunk)
                    chunks.append(chunk)
                ticks_df = pd.concat(chunks, ignore_index=True)
            else:
                ticks_df = self._optimize_dtypes(ticks_df)

            return ticks_df
        except Exception as e:
            logger.warning(f"Failed to parse ticks: {e}")
            return None

    def _extract_players(
        self, kills_df: pd.DataFrame, damages_df: pd.DataFrame
    ) -> tuple[dict[int, str], dict[int, str]]:
        """Extract player names and teams from DataFrames using vectorized operations."""
        names: dict[int, str] = {}
        teams: dict[int, str] = {}

        # Column name variations (includes both demoparser2 enriched names and raw event names)
        att_id_cols = ["attacker_steamid", "attacker_steam_id", "attacker"]
        att_name_cols = ["attacker_name"]
        att_team_cols = ["attacker_team_name", "attacker_side", "attacker_team"]
        vic_id_cols = ["user_steamid", "victim_steamid", "victim_steam_id", "userid"]
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
                if pd.isna(team_val):
                    return "Unknown"
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

            # Get relevant columns
            cols = [id_col, name_col] + ([team_col] if team_col else [])
            work_df = df[cols].copy()

            # Filter out invalid steamids first
            work_df = work_df[work_df[id_col].notna() & (work_df[id_col] != 0)]

            # Sort by name validity: rows with non-empty names come first
            # This ensures drop_duplicates keeps the row with a valid name
            work_df["_has_name"] = work_df[name_col].notna() & (work_df[name_col] != "")
            work_df = work_df.sort_values("_has_name", ascending=False)

            # Now deduplicate - keeps first occurrence (which has valid name if any exists)
            unique_df = work_df.drop_duplicates(subset=[id_col])

            # Vectorized extraction
            for _, row in unique_df.iterrows():
                sid = safe_int(row[id_col])
                if sid and sid not in names:
                    raw_name = safe_str(row[name_col])
                    # Validate the name isn't a Steam ID or garbage
                    if is_valid_player_name(raw_name):
                        names[sid] = raw_name
                    else:
                        # Use a friendly fallback instead of raw Steam ID
                        names[sid] = make_fallback_player_name(sid)
                        logger.debug(f"Player {sid} has invalid name '{raw_name}', using fallback")
                    if team_col:
                        teams[sid] = parse_team(row[team_col])

        # Extract from kills (attackers and victims)
        extract_from_df_vectorized(kills_df, att_id_cols, att_name_cols, att_team_cols)
        extract_from_df_vectorized(kills_df, vic_id_cols, vic_name_cols, vic_team_cols)
        # Extract from damages
        extract_from_df_vectorized(damages_df, att_id_cols, att_name_cols, att_team_cols)
        extract_from_df_vectorized(damages_df, vic_id_cols, vic_name_cols, vic_team_cols)

        # Final validation pass: ensure no names are raw Steam IDs
        for sid, name in list(names.items()):
            if not is_valid_player_name(name):
                names[sid] = make_fallback_player_name(sid)
                logger.debug(f"Post-validation: Player {sid} name '{name}' replaced with fallback")

        return names, teams

    def _resolve_persistent_teams(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
    ) -> tuple[dict[int, str], dict[str, set[int]], dict[str, str], int]:
        """
        Resolve persistent team identities using tick share analysis.

        PERFORMANCE: Uses GroupBy aggregation instead of nested loops for 10-20x speedup.
        Reduces thousands of iterations to 3 optimized Pandas calls.

        Returns:
            - player_persistent_teams: {steamid -> "Team A" or "Team B"}
            - team_rosters: {"Team A": {steamids}, "Team B": {steamids}}
            - team_starting_sides: {"Team A": "CT" or "T", "Team B": "CT" or "T"}
            - halftime_round: 13 (MR12) or 16 (MR15)
        """

        def convert_team_to_numeric(series: pd.Series) -> pd.Series:
            """Convert team values to numeric (2=T, 3=CT).

            Handles both numeric columns (team_num) and string columns (side/team_name).
            """
            result = pd.to_numeric(series, errors="coerce")

            # Handle string values like "CT", "T", "TERRORIST", etc.
            str_mask = result.isna() & series.notna()
            if str_mask.any():
                str_vals = series[str_mask].astype(str).str.upper()
                # CT = 3, T = 2
                result.loc[str_mask & str_vals.str.contains("CT", na=False)] = 3
                result.loc[
                    str_mask
                    & ~str_vals.str.contains("CT", na=False)
                    & str_vals.str.contains("T", na=False)
                ] = 2

            return result

        # Step 1: VECTORIZED - Collect all player-team appearances into a single DataFrame
        appearances = []

        for df in [kills_df, damages_df]:
            if df is None or df.empty:
                continue

            # Find column names - include all variations
            att_id_col = self._find_column(
                df, ["attacker_steamid", "attacker_steam_id", "attacker_id"]
            )
            vic_id_col = self._find_column(df, ["user_steamid", "victim_steamid", "victim_id"])
            att_team_col = self._find_column(
                df, ["attacker_team_num", "attacker_team_name", "attacker_side", "attacker_team"]
            )
            vic_team_col = self._find_column(
                df,
                ["user_team_num", "user_team_name", "victim_team_num", "victim_side", "user_side"],
            )

            # Extract attacker appearances (vectorized)
            if att_id_col and att_team_col:
                att_df = df[[att_id_col, att_team_col]].dropna()
                att_df = att_df.rename(columns={att_id_col: "steamid", att_team_col: "team_val"})
                att_df["steamid"] = pd.to_numeric(att_df["steamid"], errors="coerce")
                att_df["team_num"] = convert_team_to_numeric(att_df["team_val"])
                att_df = att_df[["steamid", "team_num"]]
                appearances.append(att_df)

            # Extract victim appearances (vectorized)
            if vic_id_col and vic_team_col:
                vic_df = df[[vic_id_col, vic_team_col]].dropna()
                vic_df = vic_df.rename(columns={vic_id_col: "steamid", vic_team_col: "team_val"})
                vic_df["steamid"] = pd.to_numeric(vic_df["steamid"], errors="coerce")
                vic_df["team_num"] = convert_team_to_numeric(vic_df["team_val"])
                vic_df = vic_df[["steamid", "team_num"]]
                appearances.append(vic_df)

        # Handle empty case
        if not appearances:
            logger.warning("No team appearances found for persistent team resolution")
            return {}, {"Team A": set(), "Team B": set()}, {"Team A": "CT", "Team B": "T"}, 13

        # Step 2: VECTORIZED - Combine and count using GroupBy (the "Pandas Magic")
        combined = pd.concat(appearances, ignore_index=True)
        combined = combined[combined["team_num"].isin([2, 3])]
        combined = combined[combined["steamid"].notna() & (combined["steamid"] != 0)]

        if combined.empty:
            return {}, {"Team A": set(), "Team B": set()}, {"Team A": "CT", "Team B": "T"}, 13

        # GroupBy count: Result is indexed by (steamid, team_num) with count as value
        team_counts = combined.groupby(["steamid", "team_num"]).size().unstack(fill_value=0)

        # Ensure both columns exist
        if 2 not in team_counts.columns:
            team_counts[2] = 0
        if 3 not in team_counts.columns:
            team_counts[3] = 0

        # Step 3: VECTORIZED - Select majority team per player
        team_counts["total"] = team_counts[2] + team_counts[3]
        team_counts["primary_team"] = (team_counts[3] > team_counts[2]).map({True: 3, False: 2})
        team_counts["consistency"] = team_counts[[2, 3]].max(axis=1) / team_counts["total"]

        # Log low consistency players (vectorized filter)
        low_consistency = team_counts[team_counts["consistency"] < 0.90]
        for sid in low_consistency.index:
            logger.warning(
                f"Player {int(sid)} has low team consistency ({low_consistency.loc[sid, 'consistency']:.1%})"
            )

        # Build rosters (vectorized)
        roster_2 = set(team_counts[team_counts["primary_team"] == 2].index.astype(int))
        roster_3 = set(team_counts[team_counts["primary_team"] == 3].index.astype(int))

        # Step 4: Determine which roster started CT vs T in round 1
        round_col = self._find_column(kills_df, ["round", "round_num"])
        team_a_roster = roster_2
        team_a_starting_side = "T"
        team_b_roster = roster_3
        team_b_starting_side = "CT"

        if round_col and not kills_df.empty:
            round_1_kills = kills_df[kills_df[round_col] == 1]
            att_id_col = self._find_column(round_1_kills, ["attacker_steamid", "attacker_id"])
            att_team_col = self._find_column(round_1_kills, ["attacker_team_num", "attacker_side"])

            if att_id_col and att_team_col and not round_1_kills.empty:
                # VECTORIZED: Filter to CT appearances in round 1
                r1_df = round_1_kills[[att_id_col, att_team_col]].dropna()
                r1_df["steamid"] = pd.to_numeric(r1_df[att_id_col], errors="coerce")
                # Use same conversion helper for team values
                r1_df["team_num"] = convert_team_to_numeric(r1_df[att_team_col])
                ct_in_r1 = r1_df[r1_df["team_num"] == 3]["steamid"].dropna().astype(int)

                # Count roster appearances on CT side
                roster_2_as_ct = len([s for s in ct_in_r1 if s in roster_2])
                roster_3_as_ct = len([s for s in ct_in_r1 if s in roster_3])

                # Assign Team A to the roster that started CT
                if roster_3_as_ct > roster_2_as_ct:
                    team_a_roster = roster_3
                    team_a_starting_side = "CT"
                    team_b_roster = roster_2
                    team_b_starting_side = "T"
                else:
                    team_a_roster = roster_2
                    team_a_starting_side = "CT"
                    team_b_roster = roster_3
                    team_b_starting_side = "T"

        # Step 5: Detect match format (MR12 vs MR15)
        max_round = int(kills_df[round_col].max()) if round_col and not kills_df.empty else 24
        halftime_round = 16 if max_round > 24 else 13

        # Step 6: Build return structures (vectorized dict comprehension)
        player_persistent_teams = {int(sid): "Team A" for sid in team_a_roster}
        player_persistent_teams.update({int(sid): "Team B" for sid in team_b_roster})

        team_rosters = {"Team A": team_a_roster, "Team B": team_b_roster}
        team_starting_sides = {"Team A": team_a_starting_side, "Team B": team_b_starting_side}

        logger.info(
            f"Resolved persistent teams: Team A ({len(team_a_roster)} players, "
            f"started {team_a_starting_side}), Team B ({len(team_b_roster)} players, "
            f"started {team_b_starting_side}), halftime round {halftime_round}"
        )

        return player_persistent_teams, team_rosters, team_starting_sides, halftime_round

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
                    "kills": 0,
                    "deaths": 0,
                    "assists": 0,
                    "headshots": 0,
                    "hs_percent": 0.0,
                    "total_damage": 0,
                    "adr": 0.0,
                    "kd_ratio": 0.0,
                    "weapon_kills": {},
                }
            return stats

        # Find columns with caching - includes raw event names for compatibility
        att_col = self._find_column(
            kills_df, ["attacker_steamid", "attacker_steam_id", "attacker"], "kills_att_id"
        )
        vic_col = self._find_column(
            kills_df,
            ["user_steamid", "victim_steamid", "victim_steam_id", "userid"],
            "kills_vic_id",
        )
        hs_col = self._find_column(kills_df, ["headshot"], "kills_hs")
        weapon_col = self._find_column(kills_df, ["weapon"], "kills_weapon")
        assist_col = self._find_column(
            kills_df, ["assister_steamid", "assister_steam_id", "assister"], "kills_assist"
        )
        dmg_att_col = (
            self._find_column(
                damages_df, ["attacker_steamid", "attacker_steam_id", "attacker"], "dmg_att_id"
            )
            if not damages_df.empty
            else None
        )
        dmg_col = (
            self._find_column(damages_df, ["dmg_health", "damage", "dmg"], "dmg_val")
            if not damages_df.empty
            else None
        )

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
                kills_df_numeric[att_col] = pd.to_numeric(
                    kills_df_numeric[att_col], errors="coerce"
                )

                # Count kills per player
                kills_by_player = kills_df_numeric.groupby(att_col).size().to_dict()

                # Count headshots per player
                if hs_col:
                    headshots_by_player = kills_df_numeric.groupby(att_col)[hs_col].sum().to_dict()

                # Weapon kills per player
                if weapon_col:
                    for steam_id in player_names.keys():
                        player_kills_df = kills_df_numeric[
                            kills_df_numeric[att_col] == float(steam_id)
                        ]
                        if not player_kills_df.empty:
                            weapon_kills_by_player[steam_id] = (
                                player_kills_df[weapon_col].value_counts().to_dict()
                            )

            if vic_col:
                kills_df_numeric = kills_df.copy()
                kills_df_numeric[vic_col] = pd.to_numeric(
                    kills_df_numeric[vic_col], errors="coerce"
                )
                deaths_by_player = kills_df_numeric.groupby(vic_col).size().to_dict()

            if assist_col and assist_col in kills_df.columns:
                kills_df_numeric = kills_df.copy()
                kills_df_numeric[assist_col] = pd.to_numeric(
                    kills_df_numeric[assist_col], errors="coerce"
                )
                assists_by_player = kills_df_numeric.groupby(assist_col).size().to_dict()

        if not damages_df.empty and dmg_att_col and dmg_col:
            damages_numeric = damages_df.copy()
            damages_numeric[dmg_att_col] = pd.to_numeric(
                damages_numeric[dmg_att_col], errors="coerce"
            )
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

    def _parse_team_value_vectorized(self, series: pd.Series) -> pd.Series:
        """Vectorized team value parsing - converts team values to CT/T/Unknown.

        Handles both string ("CT", "TERRORIST") and numeric (2=T, 3=CT) formats.
        Uses vectorized operations instead of row-by-row iteration.
        """
        result = pd.Series("Unknown", index=series.index)

        # Handle string values
        str_mask = series.apply(lambda x: isinstance(x, str))
        if str_mask.any():
            str_vals = series[str_mask].str.upper()
            result.loc[str_mask & str_vals.str.contains("CT", na=False)] = "CT"
            result.loc[
                str_mask
                & ~str_vals.str.contains("CT", na=False)
                & str_vals.str.contains("T", na=False)
            ] = "T"

        # Handle numeric values (2=T, 3=CT)
        num_mask = series.apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))
        if num_mask.any():
            num_vals = series[num_mask].astype(int)
            result.loc[num_mask & (num_vals == 3)] = "CT"
            result.loc[num_mask & (num_vals == 2)] = "T"

        return result

    def _build_kills(self, kills_df: pd.DataFrame) -> list[KillEvent]:
        """Build KillEvent list from kills DataFrame using vectorized operations.

        PERFORMANCE: Uses to_dict('records') instead of iterrows() for 3-5x speedup.
        """
        if kills_df.empty:
            return []

        # Find columns once (cached) - includes raw event names for compatibility
        att_id = self._find_column(kills_df, ["attacker_steamid", "attacker_steam_id", "attacker"])
        att_name = self._find_column(kills_df, ["attacker_name"])
        att_team = self._find_column(
            kills_df, ["attacker_team_name", "attacker_side", "attacker_team"]
        )
        vic_id = self._find_column(
            kills_df, ["user_steamid", "victim_steamid", "victim_steam_id", "userid"]
        )
        vic_name = self._find_column(kills_df, ["user_name", "victim_name"])
        vic_team = self._find_column(kills_df, ["user_team_name", "victim_side", "victim_team"])
        round_col = self._find_column(
            kills_df,
            ["total_rounds_played", "round", "round_num", "round_number", "roundNum", "RoundNum"],
        )
        if not round_col:
            logger.warning(
                f"Round column not found in kills_df. Available columns: {list(kills_df.columns)}"
            )
        else:
            logger.debug(f"Using round column: {round_col}")

        # Position columns
        att_x = self._find_column(kills_df, ["attacker_X", "attacker_x", "X", "x"])
        att_y = self._find_column(kills_df, ["attacker_Y", "attacker_y", "Y", "y"])
        att_z = self._find_column(kills_df, ["attacker_Z", "attacker_z", "Z", "z"])
        att_pitch = self._find_column(kills_df, ["attacker_pitch", "pitch"])
        att_yaw = self._find_column(kills_df, ["attacker_yaw", "yaw"])
        vic_x = self._find_column(kills_df, ["user_X", "victim_X", "user_x", "victim_x"])
        vic_y = self._find_column(kills_df, ["user_Y", "victim_Y", "user_y", "victim_y"])
        vic_z = self._find_column(kills_df, ["user_Z", "victim_Z", "user_z", "victim_z"])

        logger.debug(
            f"Position columns found - attacker: X={att_x}, Y={att_y}, Z={att_z}, pitch={att_pitch}, yaw={att_yaw}"
        )
        logger.debug(f"Position columns found - victim: X={vic_x}, Y={vic_y}, Z={vic_z}")

        # VECTORIZED: Pre-process all columns at once (C-speed operations)
        df = kills_df.copy()

        # Parse team values vectorized
        df["_att_side"] = self._parse_team_value_vectorized(df[att_team]) if att_team else "Unknown"
        df["_vic_side"] = self._parse_team_value_vectorized(df[vic_team]) if vic_team else "Unknown"

        # Fill NaN and convert types (vectorized)
        df["_tick"] = pd.to_numeric(df.get("tick", 0), errors="coerce").fillna(0).astype(int)
        df["_round"] = (
            pd.to_numeric(df.get(round_col, 0), errors="coerce").fillna(0).astype(int)
            if round_col
            else 0
        )
        df["_att_id"] = (
            pd.to_numeric(df.get(att_id, 0), errors="coerce").fillna(0).astype(int) if att_id else 0
        )
        df["_att_name"] = df.get(att_name, "").fillna("") if att_name else ""
        df["_vic_id"] = (
            pd.to_numeric(df.get(vic_id, 0), errors="coerce").fillna(0).astype(int) if vic_id else 0
        )
        df["_vic_name"] = df.get(vic_name, "").fillna("") if vic_name else ""
        df["_weapon"] = df["weapon"].fillna("") if "weapon" in df.columns else ""
        df["_headshot"] = (
            df["headshot"].fillna(False).astype(bool) if "headshot" in df.columns else False
        )
        df["_flash_assist"] = (
            df["flash_assist"].fillna(False).astype(bool) if "flash_assist" in df.columns else False
        )

        # Position data - keep as float, None for NaN
        for col, target in [
            (att_x, "_att_x"),
            (att_y, "_att_y"),
            (att_z, "_att_z"),
            (att_pitch, "_att_pitch"),
            (att_yaw, "_att_yaw"),
            (vic_x, "_vic_x"),
            (vic_y, "_vic_y"),
            (vic_z, "_vic_z"),
        ]:
            if col and col in df.columns:
                df[target] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[target] = None

        # Assister fields
        df["_assister_id"] = pd.to_numeric(df.get("assister_steamid", pd.NA), errors="coerce")
        df["_assister_name"] = df.get("assister_name", pd.NA)

        # VECTORIZED: Convert to list of dicts (C-speed iteration)
        records = df.to_dict("records")

        # Build KillEvent list from records (single Python loop, but with pre-computed values)
        kills = []
        for r in records:
            try:
                kill = KillEvent(
                    tick=int(r["_tick"]),
                    round_num=int(r["_round"]) if isinstance(r["_round"], (int, float)) else 0,
                    attacker_steamid=int(r["_att_id"])
                    if isinstance(r["_att_id"], (int, float))
                    else 0,
                    attacker_name=str(r["_att_name"]) if r["_att_name"] else "",
                    attacker_side=str(r["_att_side"]),
                    victim_steamid=int(r["_vic_id"])
                    if isinstance(r["_vic_id"], (int, float))
                    else 0,
                    victim_name=str(r["_vic_name"]) if r["_vic_name"] else "",
                    victim_side=str(r["_vic_side"]),
                    weapon=str(r["_weapon"]) if r["_weapon"] else "",
                    headshot=bool(r["_headshot"]),
                    assister_steamid=(
                        int(r["_assister_id"]) if pd.notna(r["_assister_id"]) else None
                    ),
                    assister_name=(
                        str(r["_assister_name"]) if pd.notna(r["_assister_name"]) else None
                    ),
                    flash_assist=bool(r["_flash_assist"]),
                    attacker_x=float(r["_att_x"]) if pd.notna(r.get("_att_x")) else None,
                    attacker_y=float(r["_att_y"]) if pd.notna(r.get("_att_y")) else None,
                    attacker_z=float(r["_att_z"]) if pd.notna(r.get("_att_z")) else None,
                    attacker_pitch=float(r["_att_pitch"])
                    if pd.notna(r.get("_att_pitch"))
                    else None,
                    attacker_yaw=float(r["_att_yaw"]) if pd.notna(r.get("_att_yaw")) else None,
                    victim_x=float(r["_vic_x"]) if pd.notna(r.get("_vic_x")) else None,
                    victim_y=float(r["_vic_y"]) if pd.notna(r.get("_vic_y")) else None,
                    victim_z=float(r["_vic_z"]) if pd.notna(r.get("_vic_z")) else None,
                )
                kills.append(kill)
            except Exception as e:
                logger.debug(f"Error processing kill record: {e}")
                continue

        # Log position data availability
        kills_with_att_pos = sum(1 for k in kills if k.attacker_x is not None)
        kills_with_vic_pos = sum(1 for k in kills if k.victim_x is not None)
        kills_with_angles = sum(1 for k in kills if k.attacker_pitch is not None)
        logger.info(
            f"Built {len(kills)} kills: {kills_with_att_pos} attacker pos, {kills_with_vic_pos} victim pos, {kills_with_angles} angles"
        )

        return kills

    def _build_damages(self, damages_df: pd.DataFrame) -> list[DamageEvent]:
        """Build DamageEvent list from damages DataFrame using vectorized operations.

        PERFORMANCE: Uses to_dict('records') instead of iterrows() for 5-10x speedup.
        This is critical as damage events are 5-10x more frequent than kills.
        """
        if damages_df.empty:
            return []

        # Find columns once (cached) - includes raw event names for compatibility
        att_id = self._find_column(
            damages_df, ["attacker_steamid", "attacker_steam_id", "attacker"]
        )
        att_name = self._find_column(damages_df, ["attacker_name"])
        att_team = self._find_column(damages_df, ["attacker_team_name", "attacker_side"])
        vic_id = self._find_column(damages_df, ["user_steamid", "victim_steamid", "userid"])
        vic_name = self._find_column(damages_df, ["user_name", "victim_name"])
        vic_team = self._find_column(damages_df, ["user_team_name", "victim_side"])
        dmg_col = self._find_column(damages_df, ["dmg_health", "damage", "dmg"])
        round_col = self._find_column(damages_df, ["total_rounds_played", "round", "round_num"])

        # VECTORIZED: Pre-process all columns at once (C-speed operations)
        df = damages_df.copy()

        # Parse team values vectorized (reuse helper from _build_kills)
        df["_att_side"] = self._parse_team_value_vectorized(df[att_team]) if att_team else "Unknown"
        df["_vic_side"] = self._parse_team_value_vectorized(df[vic_team]) if vic_team else "Unknown"

        # Fill NaN and convert types (vectorized)
        df["_tick"] = pd.to_numeric(df.get("tick", 0), errors="coerce").fillna(0).astype(int)
        df["_round"] = (
            pd.to_numeric(df.get(round_col, 0), errors="coerce").fillna(0).astype(int)
            if round_col
            else 0
        )
        df["_att_id"] = (
            pd.to_numeric(df.get(att_id, 0), errors="coerce").fillna(0).astype(int) if att_id else 0
        )
        df["_att_name"] = df.get(att_name, "").fillna("") if att_name else ""
        df["_vic_id"] = (
            pd.to_numeric(df.get(vic_id, 0), errors="coerce").fillna(0).astype(int) if vic_id else 0
        )
        df["_vic_name"] = df.get(vic_name, "").fillna("") if vic_name else ""
        df["_damage"] = (
            pd.to_numeric(df.get(dmg_col, 0), errors="coerce").fillna(0).astype(int)
            if dmg_col
            else 0
        )
        df["_dmg_armor"] = (
            pd.to_numeric(df.get("dmg_armor", 0), errors="coerce").fillna(0).astype(int)
        )

        # Health/armor remaining - try multiple column names
        health_col = df.get("health", df.get("health_remaining", pd.Series(0, index=df.index)))
        armor_col = df.get("armor", df.get("armor_remaining", pd.Series(0, index=df.index)))
        df["_health_remaining"] = pd.to_numeric(health_col, errors="coerce").fillna(0).astype(int)
        df["_armor_remaining"] = pd.to_numeric(armor_col, errors="coerce").fillna(0).astype(int)

        df["_weapon"] = df["weapon"].fillna("") if "weapon" in df.columns else ""
        df["_hitgroup"] = (
            df["hitgroup"].fillna("generic") if "hitgroup" in df.columns else "generic"
        )

        # VECTORIZED: Convert to list of dicts (C-speed iteration)
        records = df.to_dict("records")

        # Build DamageEvent list from records
        damages = [
            DamageEvent(
                tick=int(r["_tick"]),
                round_num=int(r["_round"]) if isinstance(r["_round"], (int, float)) else 0,
                attacker_steamid=int(r["_att_id"]) if isinstance(r["_att_id"], (int, float)) else 0,
                attacker_name=str(r["_att_name"]) if r["_att_name"] else "",
                attacker_side=str(r["_att_side"]),
                victim_steamid=int(r["_vic_id"]) if isinstance(r["_vic_id"], (int, float)) else 0,
                victim_name=str(r["_vic_name"]) if r["_vic_name"] else "",
                victim_side=str(r["_vic_side"]),
                damage=int(r["_damage"]),
                damage_armor=int(r["_dmg_armor"]),
                health_remaining=int(r["_health_remaining"]),
                armor_remaining=int(r["_armor_remaining"]),
                weapon=str(r["_weapon"]) if r["_weapon"] else "",
                hitgroup=str(r["_hitgroup"]) if r["_hitgroup"] else "generic",
            )
            for r in records
        ]

        return damages

    def _build_rounds(
        self,
        rounds_df: pd.DataFrame,
        round_start_df: pd.DataFrame = None,
        round_freeze_df: pd.DataFrame = None,
    ) -> list[RoundInfo]:
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
                    winner = (
                        "CT"
                        if "CT" in winner_col.upper()
                        else "T"
                        if "T" in winner_col.upper()
                        else winner_col
                    )
                elif isinstance(winner_col, (int, float)):
                    # Check for NaN before converting to int
                    if pd.isna(winner_col):
                        pass  # Leave winner as "Unknown"
                    else:
                        winner = (
                            "CT"
                            if int(winner_col) == 3
                            else "T"
                            if int(winner_col) == 2
                            else "Unknown"
                        )
            elif reason:
                # Infer from reason - handle both string and numeric values
                # CS2 RoundEndReason enum values (from constants.py):
                # CT wins: 7=BOMB_DEFUSED, 8=CT_WIN, 11=ALL_HOSTAGES_RESCUED, 12=TARGET_SAVED
                # T wins: 1=TARGET_BOMBED, 9=TERRORIST_WIN, 13=HOSTAGES_NOT_RESCUED
                ct_reasons_str = ["bomb_defused", "ct_win", "target_saved", "hostages_rescued"]
                t_reasons_str = ["target_bombed", "terrorist_win", "t_win", "hostages_not_rescued"]
                ct_reasons_num = {7, 8, 11, 12}  # Numeric reason codes for CT win
                t_reasons_num = {1, 9, 13}  # Numeric reason codes for T win

                # First try to parse as numeric
                try:
                    reason_num = int(float(reason))  # Handle "7" or "7.0"
                    if reason_num in ct_reasons_num:
                        winner = "CT"
                    elif reason_num in t_reasons_num:
                        winner = "T"
                except (ValueError, TypeError):
                    # Not numeric, check as string
                    reason_lower = reason.lower()
                    if any(r in reason_lower for r in ct_reasons_str):
                        winner = "CT"
                    elif any(r in reason_lower for r in t_reasons_str):
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

        # Filter out knife round if present
        # In CS2 ESEA (MR12): max 24 rounds + potential 1 knife round at the beginning
        # Knife round detection heuristics:
        # 1. Total rounds = 25 (exceeds normal max of 24)
        # 2. First round has "knife" or "side" in reason
        # 3. First round is unrelated to bomb logic
        if len(rounds) == 25:
            first_round = rounds[0]
            first_round_reason_lower = first_round.reason.lower()

            # Check for knife round indicators
            knife_round_reasons = ["knife", "side", "determination", "pick"]
            is_knife_reason = any(r in first_round_reason_lower for r in knife_round_reasons)

            # Also check if reason is not bomb-related (normal rounds have bomb events)
            bomb_reasons = [
                "bomb_planted",
                "bomb_defused",
                "target_bombed",
                "target_saved",
                "elimination",
            ]
            is_bomb_related = any(b in first_round_reason_lower for b in bomb_reasons)

            # If it looks like a knife round, remove it
            if is_knife_reason or (len(rounds) > 1 and not is_bomb_related):
                logger.info(
                    f"Detected and filtering out knife round (reason: {first_round.reason})"
                )
                rounds = rounds[1:]  # Remove first round
                # Re-number remaining rounds to start from 1
                for i in range(len(rounds)):
                    rounds[i].round_num = i + 1

        return rounds

    def _populate_round_economy(
        self,
        rounds: list[RoundInfo],
        kills: list[KillEvent],
        ticks_df: pd.DataFrame | None,
        player_teams: dict[int, int],
        rounds_per_half: int = 12,
    ) -> None:
        """
        Populate equipment values and round_type on RoundInfo objects.

        Uses tick data if available (most accurate), otherwise estimates
        from weapons used in kills during each round.

        Args:
            rounds: List of RoundInfo objects to populate
            kills: List of KillEvent objects for weapon estimation
            ticks_df: Optional tick-level DataFrame with current_equip_value
            player_teams: Dict mapping steam_id -> team (2=T, 3=CT)
            rounds_per_half: Rounds per half for pistol detection (default 12 for MR12)
        """
        if not rounds:
            return

        # Group kills by round for equipment estimation
        kills_by_round: dict[int, list[KillEvent]] = {}
        for kill in kills:
            if kill.round_num not in kills_by_round:
                kills_by_round[kill.round_num] = []
            kills_by_round[kill.round_num].append(kill)

        # Try to get equipment from tick data at freeze_end
        equipment_from_ticks = {}
        if ticks_df is not None and not ticks_df.empty:
            equipment_from_ticks = self._extract_equipment_from_ticks(
                ticks_df, rounds, player_teams
            )

        for round_info in rounds:
            round_num = round_info.round_num
            is_pistol = is_pistol_round(round_num, rounds_per_half)

            # Try tick-based equipment first
            if round_num in equipment_from_ticks:
                ct_equip, t_equip = equipment_from_ticks[round_num]
                round_info.ct_equipment_value = ct_equip
                round_info.t_equipment_value = t_equip
            else:
                # Estimate from weapons used in kills
                ct_equip, t_equip = self._estimate_equipment_from_kills(
                    kills_by_round.get(round_num, []), player_teams
                )
                round_info.ct_equipment_value = ct_equip
                round_info.t_equipment_value = t_equip

            # Set round_type based on equipment (use average of both teams)
            avg_equipment = (round_info.ct_equipment_value + round_info.t_equipment_value) // 2
            round_info.round_type = classify_team_equipment(avg_equipment, is_pistol)

            logger.debug(
                f"Round {round_num}: CT=${round_info.ct_equipment_value}, "
                f"T=${round_info.t_equipment_value}, type={round_info.round_type}"
            )

    def _extract_equipment_from_ticks(
        self,
        ticks_df: pd.DataFrame,
        rounds: list[RoundInfo],
        player_teams: dict[int, int],
    ) -> dict[int, tuple[int, int]]:
        """
        Extract team equipment values at freeze_end_tick from tick data.

        Returns:
            Dict mapping round_num -> (ct_equipment, t_equipment)
        """
        result = {}

        # Find required columns
        tick_col = self._find_column(ticks_df, ["tick"])
        steamid_col = self._find_column(ticks_df, ["steamid", "steam_id", "user_steamid"])
        equip_col = self._find_column(
            ticks_df, ["current_equip_value", "equipment_value", "equip_value"]
        )
        team_col = self._find_column(ticks_df, ["team_num", "team", "side"])

        if not tick_col or not steamid_col or not equip_col:
            logger.debug("Missing columns for tick-based equipment extraction")
            return result

        for round_info in rounds:
            freeze_tick = round_info.freeze_end_tick
            if freeze_tick <= 0:
                continue

            # Find tick data closest to freeze_end_tick
            # Allow some tolerance (within 64 ticks = 1 second)
            tick_mask = (ticks_df[tick_col] >= freeze_tick - 32) & (
                ticks_df[tick_col] <= freeze_tick + 64
            )
            round_ticks = ticks_df[tick_mask]

            if round_ticks.empty:
                continue

            ct_total = 0
            t_total = 0

            # Group by player and sum equipment
            for steamid in round_ticks[steamid_col].unique():
                player_data = round_ticks[round_ticks[steamid_col] == steamid]
                if player_data.empty:
                    continue

                # Get equipment value (use last value in window)
                equip_value = safe_int(player_data[equip_col].iloc[-1])

                # Determine team
                team = 0
                if team_col and team_col in player_data.columns:
                    team = safe_int(player_data[team_col].iloc[-1])
                elif steamid in player_teams:
                    team = player_teams[steamid]

                if team == 3:  # CT
                    ct_total += equip_value
                elif team == 2:  # T
                    t_total += equip_value

            if ct_total > 0 or t_total > 0:
                result[round_info.round_num] = (ct_total, t_total)

        return result

    def _estimate_equipment_from_kills(
        self,
        kills: list[KillEvent],
        player_teams: dict[int, int],
    ) -> tuple[int, int]:
        """
        Estimate team equipment from weapons used in kills.

        This is a fallback when tick data isn't available.

        Returns:
            Tuple of (ct_equipment, t_equipment) estimated values
        """
        ct_weapons: list[str] = []
        t_weapons: list[str] = []

        for kill in kills:
            weapon = kill.weapon
            side = kill.attacker_side.upper() if kill.attacker_side else ""

            if "CT" in side:
                ct_weapons.append(weapon)
            elif "T" in side:
                t_weapons.append(weapon)
            else:
                # Use player_teams as fallback
                team = player_teams.get(kill.attacker_steamid, 0)
                if team == 3:
                    ct_weapons.append(weapon)
                elif team == 2:
                    t_weapons.append(weapon)

        # Estimate per-player equipment from best weapon, multiply by 5 for team
        ct_max_equip = max((estimate_equipment_from_weapon(w) for w in ct_weapons), default=800)
        t_max_equip = max((estimate_equipment_from_weapon(w) for w in t_weapons), default=800)

        # Estimate team total (assume similar loadouts for teammates)
        # This is rough - multiply by 4-5 players
        ct_team_estimate = ct_max_equip * 5
        t_team_estimate = t_max_equip * 5

        return (ct_team_estimate, t_team_estimate)

    def _build_weapon_fires(self, weapon_fires_df: pd.DataFrame) -> list[WeaponFireEvent]:
        """Build WeaponFireEvent list for accuracy tracking."""
        fires = []
        if weapon_fires_df.empty:
            return fires

        # Find columns
        player_id = self._find_column(
            weapon_fires_df, ["user_steamid", "player_steamid", "steamid"]
        )
        player_name_col = self._find_column(weapon_fires_df, ["user_name", "player_name", "name"])
        player_team = self._find_column(weapon_fires_df, ["user_team_name", "player_team"])
        round_col = self._find_column(
            weapon_fires_df, ["total_rounds_played", "round", "round_num"]
        )
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
                    side = (
                        "CT"
                        if "CT" in team_val.upper()
                        else "T"
                        if "T" in team_val.upper()
                        else team_val
                    )

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

    def _build_blinds(
        self, blinds_df: pd.DataFrame, player_teams: dict[int, str]
    ) -> list[BlindEvent]:
        """Build BlindEvent list for flash effectiveness tracking."""
        blinds = []
        if blinds_df.empty:
            logger.warning("blinds_df is empty, no blind events to build")
            return blinds

        # Debug: Log available columns
        logger.info(f"blinds_df columns: {list(blinds_df.columns)}")
        logger.info(f"blinds_df shape: {blinds_df.shape}, player_teams count: {len(player_teams)}")

        # Find columns - includes raw event names for compatibility
        att_id = self._find_column(blinds_df, ["attacker_steamid", "attacker_steam_id", "attacker"])
        att_name = self._find_column(blinds_df, ["attacker_name"])
        vic_id = self._find_column(blinds_df, ["user_steamid", "userid", "victim_steamid"])
        vic_name = self._find_column(blinds_df, ["user_name", "victim_name"])
        duration_col = self._find_column(blinds_df, ["blind_duration", "duration"])
        round_col = self._find_column(blinds_df, ["total_rounds_played", "round", "round_num"])

        # Debug: Log found columns
        logger.info(
            f"Blind column mapping: att_id={att_id}, vic_id={vic_id}, "
            f"duration={duration_col}, round={round_col}"
        )

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

    def _build_grenades(
        self,
        thrown_df: pd.DataFrame,
        flash_df: pd.DataFrame,
        he_df: pd.DataFrame,
        smoke_df: pd.DataFrame,
        molly_df: pd.DataFrame,
    ) -> list[GrenadeEvent]:
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

                grenades.append(
                    GrenadeEvent(
                        tick=safe_int(row.get("tick")),
                        round_num=safe_int(row.get(round_col)) if round_col else 0,
                        player_steamid=safe_int(row.get(player_id)) if player_id else 0,
                        player_name=safe_str(row.get(player_name_col)) if player_name_col else "",
                        player_side=side,
                        grenade_type=(
                            safe_str(row.get(grenade_type_col)) if grenade_type_col else "unknown"
                        ),
                        event_type="thrown",
                        x=safe_float(row.get(x_col)) if x_col else None,
                        y=safe_float(row.get(y_col)) if y_col else None,
                        z=safe_float(row.get(z_col)) if z_col else None,
                    )
                )

        def process_detonate(df: pd.DataFrame, grenade_type: str):
            if df.empty:
                return
            player_id = self._find_column(
                df, ["user_steamid", "player_steamid", "steamid", "thrower_steamid"]
            )
            player_name_col = self._find_column(df, ["user_name", "player_name", "thrower_name"])
            round_col = self._find_column(df, ["total_rounds_played", "round"])
            x_col = self._find_column(df, ["X", "x"])
            y_col = self._find_column(df, ["Y", "y"])
            z_col = self._find_column(df, ["Z", "z"])

            for _, row in df.iterrows():
                grenades.append(
                    GrenadeEvent(
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
                    )
                )

        process_thrown(thrown_df)
        process_detonate(flash_df, "flashbang")
        process_detonate(he_df, "hegrenade")
        process_detonate(smoke_df, "smokegrenade")
        process_detonate(molly_df, "molotov")

        logger.info(f"Built {len(grenades)} grenade events")
        return grenades

    def _build_bomb_events(
        self, planted_df: pd.DataFrame, defused_df: pd.DataFrame, exploded_df: pd.DataFrame
    ) -> list[BombEvent]:
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
                bomb_events.append(
                    BombEvent(
                        tick=safe_int(row.get("tick")),
                        round_num=safe_int(row.get(round_col)) if round_col else 0,
                        player_steamid=safe_int(row.get(player_id)) if player_id else 0,
                        player_name=safe_str(row.get(player_name_col)) if player_name_col else "",
                        event_type=event_type,
                        site=safe_str(row.get(site_col)) if site_col else "",
                        x=safe_float(row.get(x_col)) if x_col else None,
                        y=safe_float(row.get(y_col)) if y_col else None,
                        z=safe_float(row.get(z_col)) if z_col else None,
                    )
                )

        process_bomb_df(planted_df, "planted")
        process_bomb_df(defused_df, "defused")
        process_bomb_df(exploded_df, "exploded")

        logger.info(f"Built {len(bomb_events)} bomb events")
        return bomb_events

    def _parse_with_awpy(self, include_ticks: bool = False) -> DemoData:
        """Fallback parser using awpy."""
        logger.info(f"Parsing demo with awpy: {self.demo_path}")

        # Import awpy dynamically to allow tests to inject a mock via sys.modules
        import importlib

        awpy_mod = importlib.import_module("awpy")
        DemoClass = awpy_mod.Demo
        demo = DemoClass(str(self.demo_path), verbose=False)

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

        logger.info(
            f"awpy parsed: {len(kills_df)} kills, {len(damages_df)} damages, {len(rounds_df)} rounds"
        )
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

        # NEW: Resolve persistent team identities (handles halftime swaps)
        (
            player_persistent_teams,
            team_rosters,
            team_starting_sides,
            halftime_round,
        ) = self._resolve_persistent_teams(kills_df, damages_df)

        player_stats = self._calculate_stats(
            kills_df, damages_df, player_names, player_teams, num_rounds
        )
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
            # NEW: Persistent team identity
            player_persistent_teams=player_persistent_teams,
            team_rosters=team_rosters,
            team_starting_sides=team_starting_sides,
            halftime_round=halftime_round,
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


def parse_demo(
    demo_path: str | Path, include_ticks: bool = False, comprehensive: bool = True
) -> DemoData:
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
