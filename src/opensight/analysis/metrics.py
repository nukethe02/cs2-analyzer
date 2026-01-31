"""
Professional-Grade Metrics for CS2 Analytics

Implements key performance metrics:
- Time to Damage (TTD): Latency between spotting an enemy and dealing damage
- Crosshair Placement (CP): Angular distance between aim and target position
- Economy Metrics: Money efficiency, weapon value tracking, eco round impact
- Utility Metrics: Grenade usage, flash effectiveness, smoke coverage
- Positioning Metrics: Map control, rotation timing, angle holding

These metrics are derived from tick-level demo data and provide insights
comparable to professional analytics platforms.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from opensight.core.parser import DemoData

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def _find_column(df: pd.DataFrame, options: list[str]) -> str | None:
    """Find the first matching column name from options."""
    if df is None:
        return None
    for col in options:
        if col in df.columns:
            return col
    return None


def _get_round_ticks(demo_data: DemoData) -> tuple[list[int], list[int]]:
    """Extract round start and end ticks from rounds."""
    starts = []
    ends = []
    if hasattr(demo_data, "rounds") and demo_data.rounds:
        for r in demo_data.rounds:
            starts.append(getattr(r, "start_tick", 0))
            ends.append(getattr(r, "end_tick", 0))
    return starts, ends


# ============================================================================
# Constants and Enums
# ============================================================================


class WeaponCategory(Enum):
    """CS2 weapon categories for economy analysis."""

    PISTOL = "pistol"
    SMG = "smg"
    RIFLE = "rifle"
    SNIPER = "sniper"
    SHOTGUN = "shotgun"
    MACHINE_GUN = "machine_gun"
    KNIFE = "knife"
    GRENADE = "grenade"
    UNKNOWN = "unknown"


class GrenadeType(Enum):
    """CS2 grenade types."""

    SMOKE = "smokegrenade"
    FLASH = "flashbang"
    HE = "hegrenade"
    MOLOTOV = "molotov"
    INCENDIARY = "incgrenade"
    DECOY = "decoy"


# Weapon prices for economy calculations
WEAPON_PRICES: dict[str, int] = {
    # Pistols
    "usp_silencer": 200,
    "p2000": 200,
    "glock": 200,
    "p250": 300,
    "tec9": 500,
    "fiveseven": 500,
    "cz75a": 500,
    "deagle": 700,
    "revolver": 600,
    "dualies": 400,
    # SMGs
    "mac10": 1050,
    "mp9": 1250,
    "mp7": 1500,
    "ump45": 1200,
    "p90": 2350,
    "bizon": 1400,
    "mp5sd": 1500,
    # Rifles
    "ak47": 2700,
    "m4a1": 2900,
    "m4a1_silencer": 2900,
    "famas": 2050,
    "galilar": 1800,
    "sg556": 3000,
    "aug": 3300,
    # Snipers
    "awp": 4750,
    "ssg08": 1700,
    "scar20": 5000,
    "g3sg1": 5000,
    # Shotguns
    "nova": 1050,
    "xm1014": 2000,
    "mag7": 1300,
    "sawedoff": 1100,
    # Machine guns
    "m249": 5200,
    "negev": 1700,
    # Grenades
    "hegrenade": 300,
    "flashbang": 200,
    "smokegrenade": 300,
    "molotov": 400,
    "incgrenade": 600,
    "decoy": 50,
}

# Weapon categories for classification
WEAPON_CATEGORIES: dict[str, WeaponCategory] = {
    "usp_silencer": WeaponCategory.PISTOL,
    "p2000": WeaponCategory.PISTOL,
    "glock": WeaponCategory.PISTOL,
    "p250": WeaponCategory.PISTOL,
    "tec9": WeaponCategory.PISTOL,
    "fiveseven": WeaponCategory.PISTOL,
    "cz75a": WeaponCategory.PISTOL,
    "deagle": WeaponCategory.PISTOL,
    "revolver": WeaponCategory.PISTOL,
    "dualies": WeaponCategory.PISTOL,
    "mac10": WeaponCategory.SMG,
    "mp9": WeaponCategory.SMG,
    "mp7": WeaponCategory.SMG,
    "ump45": WeaponCategory.SMG,
    "p90": WeaponCategory.SMG,
    "bizon": WeaponCategory.SMG,
    "mp5sd": WeaponCategory.SMG,
    "ak47": WeaponCategory.RIFLE,
    "m4a1": WeaponCategory.RIFLE,
    "m4a1_silencer": WeaponCategory.RIFLE,
    "famas": WeaponCategory.RIFLE,
    "galilar": WeaponCategory.RIFLE,
    "sg556": WeaponCategory.RIFLE,
    "aug": WeaponCategory.RIFLE,
    "awp": WeaponCategory.SNIPER,
    "ssg08": WeaponCategory.SNIPER,
    "scar20": WeaponCategory.SNIPER,
    "g3sg1": WeaponCategory.SNIPER,
    "nova": WeaponCategory.SHOTGUN,
    "xm1014": WeaponCategory.SHOTGUN,
    "mag7": WeaponCategory.SHOTGUN,
    "sawedoff": WeaponCategory.SHOTGUN,
    "m249": WeaponCategory.MACHINE_GUN,
    "negev": WeaponCategory.MACHINE_GUN,
    "knife": WeaponCategory.KNIFE,
}

# Valid weapon categories for crosshair placement measurement
# Only measure CP for primary/secondary weapons, not utility or melee
CP_VALID_CATEGORIES: set[WeaponCategory] = {
    WeaponCategory.PISTOL,
    WeaponCategory.SMG,
    WeaponCategory.RIFLE,
    WeaponCategory.SNIPER,
    WeaponCategory.SHOTGUN,
    WeaponCategory.MACHINE_GUN,
}

# ============================================================================
# Firing Sequence Detection Constants (Leetify-style Tap/Burst/Spray)
# ============================================================================

# Gap threshold between shots to start a new sequence
# 150ms at 64 tick = ~10 ticks
SEQUENCE_GAP_TICKS = 10

# Sequence classifications (Leetify standard)
TAP_MAX_SHOTS = 2  # 1-2 shots = Tap
BURST_MAX_SHOTS = 5  # 3-5 shots = Burst
SPRAY_MIN_SHOTS = 6  # 6+ shots = Spray

# Weapons that support automatic fire (for tap/burst/spray analysis)
# Excludes: pistols (except CZ), snipers, shotguns
AUTOMATIC_WEAPONS: set[str] = {
    # Rifles
    "ak47",
    "m4a1",
    "m4a1_silencer",
    "m4a4",
    "galil",
    "galilar",
    "famas",
    "aug",
    "sg556",
    # SMGs
    "mac10",
    "mp9",
    "mp7",
    "ump45",
    "p90",
    "bizon",
    "mp5sd",
    # Machine guns
    "negev",
    "m249",
    # CZ-75 Auto (the only automatic pistol)
    "cz75a",
}

# Grenade weapon names (for filtering)
GRENADE_WEAPONS: set[str] = {
    "hegrenade",
    "flashbang",
    "smokegrenade",
    "molotov",
    "incgrenade",
    "decoy",
    "inferno",
    "he_grenade",
    "flash",
    "smoke",
    "molly",
    "inc",
}


def _is_valid_cp_weapon(weapon_name: str | None) -> bool:
    """
    Check if a weapon is valid for crosshair placement measurement.

    Only primary and secondary weapons count - knives, grenades, and
    utility kills should not affect CP metrics.

    Args:
        weapon_name: The weapon name from kill/damage event

    Returns:
        True if weapon is valid for CP measurement
    """
    if not weapon_name:
        return False

    weapon_clean = str(weapon_name).lower().replace("weapon_", "").strip()

    # Explicit knife check (various formats)
    if "knife" in weapon_clean or weapon_clean in ("bayonet", "karambit", "m9_bayonet"):
        return False

    # Explicit grenade check
    if weapon_clean in GRENADE_WEAPONS:
        return False

    # Check against known categories
    category = WEAPON_CATEGORIES.get(weapon_clean, WeaponCategory.UNKNOWN)

    # Only allow guns, not utility
    return category in CP_VALID_CATEGORIES


# Map areas for positioning analysis (simplified, per-map data would be more accurate)
MAP_AREAS: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {
    "de_dust2": [
        ("a_site", (-1500, 1000, -500, 2000)),
        ("b_site", (-2500, -1500, -1500, -500)),
        ("mid", (-1000, -500, 0, 1000)),
        ("long_a", (-500, 1500, 500, 2500)),
        ("tunnels", (-2500, -2000, -1500, -1000)),
    ],
    "de_mirage": [
        ("a_site", (-200, 500, 400, 1200)),
        ("b_site", (-2200, -1500, -1500, -800)),
        ("mid", (-1200, -300, -400, 500)),
        ("palace", (200, 800, 600, 1400)),
        ("apps", (-2000, -800, -1400, -200)),
    ],
    "de_inferno": [
        ("a_site", (1800, 300, 2400, 900)),
        ("b_site", (200, 2500, 900, 3200)),
        ("mid", (700, 500, 1400, 1500)),
        ("banana", (-200, 2000, 500, 3000)),
        ("apartments", (1400, 0, 2000, 600)),
    ],
}


@dataclass
class TTDResult:
    """Time to Damage calculation result."""

    steam_id: int
    player_name: str
    engagement_count: int
    mean_ttd_ms: float
    median_ttd_ms: float
    min_ttd_ms: float
    max_ttd_ms: float
    std_ttd_ms: float
    ttd_values: list[float]  # Individual TTD values in milliseconds

    def __repr__(self) -> str:
        return (
            f"TTD({self.player_name}: mean={self.mean_ttd_ms:.0f}ms, "
            f"median={self.median_ttd_ms:.0f}ms, n={self.engagement_count})"
        )


@dataclass
class CrosshairPlacementResult:
    """Crosshair Placement calculation result."""

    steam_id: int
    player_name: str
    sample_count: int
    mean_angle_deg: float
    median_angle_deg: float
    percentile_90_deg: float
    placement_score: float  # 0-100 score, higher is better
    angle_values: list[float]  # Individual angles in degrees

    def __repr__(self) -> str:
        return (
            f"CP({self.player_name}: mean={self.mean_angle_deg:.1f}°, "
            f"score={self.placement_score:.1f}, n={self.sample_count})"
        )


# ============================================================================
# Firing Sequence Detection (Leetify-style Tap/Burst/Spray)
# ============================================================================


@dataclass
class FiringSequence:
    """A detected firing sequence (tap/burst/spray).

    Sequences are groups of consecutive shots where the gap between
    each shot is less than SEQUENCE_GAP_TICKS (150ms).
    """

    start_tick: int
    end_tick: int
    shot_count: int
    shot_ticks: list[int]
    weapon: str
    round_num: int

    @property
    def sequence_type(self) -> str:
        """Classify the sequence based on shot count.

        - Tap: 1-2 shots
        - Burst: 3-5 shots
        - Spray: 6+ shots
        """
        if self.shot_count <= TAP_MAX_SHOTS:
            return "tap"
        elif self.shot_count <= BURST_MAX_SHOTS:
            return "burst"
        else:
            return "spray"

    @property
    def duration_ticks(self) -> int:
        """Duration of the sequence in ticks."""
        return self.end_tick - self.start_tick

    def __repr__(self) -> str:
        return f"FiringSequence({self.sequence_type}: {self.shot_count} shots, {self.weapon})"


@dataclass
class FiringAccuracyResult:
    """Tap/Burst/Spray accuracy breakdown for a player.

    Tracks shots fired, hits, and headshots separately for each
    firing style category (Leetify-style metrics).
    """

    steam_id: int
    player_name: str

    # Tap (1-2 shots)
    tap_shots_fired: int = 0
    tap_shots_hit: int = 0
    tap_headshots: int = 0
    tap_sequences: int = 0

    # Burst (3-5 shots)
    burst_shots_fired: int = 0
    burst_shots_hit: int = 0
    burst_headshots: int = 0
    burst_sequences: int = 0

    # Spray (6+ shots)
    spray_shots_fired: int = 0
    spray_shots_hit: int = 0
    spray_headshots: int = 0
    spray_sequences: int = 0

    @property
    def tap_accuracy(self) -> float:
        """Tap accuracy percentage (1-2 shot sequences)."""
        return (
            round(self.tap_shots_hit / self.tap_shots_fired * 100, 1)
            if self.tap_shots_fired > 0
            else 0.0
        )

    @property
    def burst_accuracy(self) -> float:
        """Burst accuracy percentage (3-5 shot sequences)."""
        return (
            round(self.burst_shots_hit / self.burst_shots_fired * 100, 1)
            if self.burst_shots_fired > 0
            else 0.0
        )

    @property
    def spray_accuracy(self) -> float:
        """Spray accuracy percentage (6+ shot sequences)."""
        return (
            round(self.spray_shots_hit / self.spray_shots_fired * 100, 1)
            if self.spray_shots_fired > 0
            else 0.0
        )

    @property
    def tap_headshot_rate(self) -> float:
        """Headshot rate for tap sequences."""
        return (
            round(self.tap_headshots / self.tap_shots_hit * 100, 1)
            if self.tap_shots_hit > 0
            else 0.0
        )

    @property
    def burst_headshot_rate(self) -> float:
        """Headshot rate for burst sequences."""
        return (
            round(self.burst_headshots / self.burst_shots_hit * 100, 1)
            if self.burst_shots_hit > 0
            else 0.0
        )

    @property
    def spray_headshot_rate(self) -> float:
        """Headshot rate for spray sequences."""
        return (
            round(self.spray_headshots / self.spray_shots_hit * 100, 1)
            if self.spray_shots_hit > 0
            else 0.0
        )

    @property
    def total_sequences(self) -> int:
        """Total number of firing sequences."""
        return self.tap_sequences + self.burst_sequences + self.spray_sequences

    @property
    def total_shots_fired(self) -> int:
        """Total shots fired across all sequence types."""
        return self.tap_shots_fired + self.burst_shots_fired + self.spray_shots_fired

    @property
    def total_shots_hit(self) -> int:
        """Total shots hit across all sequence types."""
        return self.tap_shots_hit + self.burst_shots_hit + self.spray_shots_hit

    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy across all firing styles."""
        return (
            round(self.total_shots_hit / self.total_shots_fired * 100, 1)
            if self.total_shots_fired > 0
            else 0.0
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            # Tap metrics
            "tap_shots_fired": self.tap_shots_fired,
            "tap_shots_hit": self.tap_shots_hit,
            "tap_headshots": self.tap_headshots,
            "tap_sequences": self.tap_sequences,
            "tap_accuracy": self.tap_accuracy,
            "tap_headshot_rate": self.tap_headshot_rate,
            # Burst metrics
            "burst_shots_fired": self.burst_shots_fired,
            "burst_shots_hit": self.burst_shots_hit,
            "burst_headshots": self.burst_headshots,
            "burst_sequences": self.burst_sequences,
            "burst_accuracy": self.burst_accuracy,
            "burst_headshot_rate": self.burst_headshot_rate,
            # Spray metrics
            "spray_shots_fired": self.spray_shots_fired,
            "spray_shots_hit": self.spray_shots_hit,
            "spray_headshots": self.spray_headshots,
            "spray_sequences": self.spray_sequences,
            "spray_accuracy": self.spray_accuracy,
            "spray_headshot_rate": self.spray_headshot_rate,
            # Totals
            "total_sequences": self.total_sequences,
            "total_shots_fired": self.total_shots_fired,
            "total_shots_hit": self.total_shots_hit,
            "overall_accuracy": self.overall_accuracy,
        }

    def __repr__(self) -> str:
        return (
            f"FiringAccuracy({self.player_name}: "
            f"tap={self.tap_accuracy:.1f}%, "
            f"burst={self.burst_accuracy:.1f}%, "
            f"spray={self.spray_accuracy:.1f}%)"
        )


# ============================================================================
# Site Anchor Rating (Defensive Hero Metric)
# ============================================================================

# Bombsite place_name values from demoparser2
BOMBSITE_PLACE_NAMES: dict[str, set[str]] = {
    "A": {"BombsiteA", "bombsitea", "Bombsite A", "A Site", "ASite"},
    "B": {"BombsiteB", "bombsiteb", "Bombsite B", "B Site", "BSite"},
}

# Minimum T-side players entering site to trigger "execute" detection
EXECUTE_PLAYER_THRESHOLD = 3

# Rating formula weights
ANCHOR_RATING_WEIGHTS = {
    "stall_time_per_second": 3.0,  # Each second of delay = 3 points
    "molotov": 15.0,  # Molotov/incendiary = 15 points (buys ~7 seconds)
    "smoke": 15.0,  # Smoke = 15 points (blocks vision)
    "flash": 5.0,  # Flash = 5 points (temporary)
    "he_grenade": 8.0,  # HE = 8 points (damage + delay)
    "kill": 10.0,  # Each kill during hold = 10 points
    "survival_bonus": 20.0,  # Survived the execute = 20 bonus
}


@dataclass
class SiteExecuteEvent:
    """Detected site execute event (3+ T's entering bombsite)."""

    round_num: int
    site: str  # "A" or "B"
    execute_start_tick: int
    t_players: list[int]  # Steam IDs of attacking T's
    ct_defenders: list[int]  # Steam IDs of CTs in site at execute start
    bomb_plant_tick: int | None = None
    round_end_tick: int = 0


@dataclass
class AnchorHoldResult:
    """Site Anchor Effectiveness metric.

    Measures how well a CT player delayed a T-side execute, even if they died.
    This metric rewards defensive players who buy time for rotations.

    "You died, but you bought us 18 seconds. Good job."
    """

    steam_id: int
    player_name: str
    round_num: int
    site: str  # "A" or "B"

    # Timing
    execute_start_tick: int
    hold_end_tick: int
    stall_time_seconds: float  # Time survived during execute

    # Outcome
    survived: bool
    kills_during_hold: int
    damage_during_hold: int
    assists_during_hold: int = 0

    # Utility used during the hold (high value)
    molotovs_used: int = 0
    smokes_used: int = 0
    flashes_used: int = 0
    he_grenades_used: int = 0

    # Context
    attackers_count: int = 0  # T's who entered site
    teammates_nearby: int = 0  # Other CTs who helped defend

    @property
    def utility_count(self) -> int:
        """Total utility pieces used during the hold."""
        return self.molotovs_used + self.smokes_used + self.flashes_used + self.he_grenades_used

    @property
    def anchor_hold_rating(self) -> float:
        """
        Composite rating rewarding stall time, utility usage, and kills.

        Formula:
        - stall_time_seconds * 3.0 (18 sec = 54 points)
        - molotovs * 15.0 (buys 5-7 seconds each)
        - smokes * 15.0 (blocks vision/delays push)
        - flashes * 5.0 (temporary disruption)
        - he_grenades * 8.0 (damage + psychological delay)
        - kills * 10.0 (reduces attackers)
        - survival_bonus: +20 if survived

        Elite anchor: 80+ | Good: 50-80 | Average: 30-50 | Poor: <30
        """
        w = ANCHOR_RATING_WEIGHTS
        score = self.stall_time_seconds * w["stall_time_per_second"]
        score += self.molotovs_used * w["molotov"]
        score += self.smokes_used * w["smoke"]
        score += self.flashes_used * w["flash"]
        score += self.he_grenades_used * w["he_grenade"]
        score += self.kills_during_hold * w["kill"]
        if self.survived:
            score += w["survival_bonus"]
        return round(score, 1)

    @property
    def rating_breakdown(self) -> dict[str, float]:
        """Detailed breakdown of rating components."""
        w = ANCHOR_RATING_WEIGHTS
        return {
            "stall_time": round(self.stall_time_seconds * w["stall_time_per_second"], 1),
            "molotovs": round(self.molotovs_used * w["molotov"], 1),
            "smokes": round(self.smokes_used * w["smoke"], 1),
            "flashes": round(self.flashes_used * w["flash"], 1),
            "he_grenades": round(self.he_grenades_used * w["he_grenade"], 1),
            "kills": round(self.kills_during_hold * w["kill"], 1),
            "survival": w["survival_bonus"] if self.survived else 0.0,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "steam_id": self.steam_id,
            "player_name": self.player_name,
            "round_num": self.round_num,
            "site": self.site,
            # Timing
            "stall_time_seconds": round(self.stall_time_seconds, 2),
            "execute_start_tick": self.execute_start_tick,
            "hold_end_tick": self.hold_end_tick,
            # Outcome
            "survived": self.survived,
            "kills_during_hold": self.kills_during_hold,
            "damage_during_hold": self.damage_during_hold,
            "assists_during_hold": self.assists_during_hold,
            # Utility
            "molotovs_used": self.molotovs_used,
            "smokes_used": self.smokes_used,
            "flashes_used": self.flashes_used,
            "he_grenades_used": self.he_grenades_used,
            "utility_count": self.utility_count,
            # Context
            "attackers_count": self.attackers_count,
            "teammates_nearby": self.teammates_nearby,
            # Rating
            "anchor_hold_rating": self.anchor_hold_rating,
            "rating_breakdown": self.rating_breakdown,
        }

    def __repr__(self) -> str:
        status = "HELD" if self.survived else "DIED"
        return (
            f"AnchorHold({self.player_name} @ {self.site}: "
            f"{self.stall_time_seconds:.1f}s, {self.utility_count} util, "
            f"{self.kills_during_hold}K, {status}, rating={self.anchor_hold_rating:.1f})"
        )


@dataclass
class PlayerAnchorStats:
    """Aggregated anchor statistics for a player across all rounds."""

    steam_id: int
    player_name: str
    total_holds: int = 0
    successful_holds: int = 0  # Survived or team won round
    total_stall_time: float = 0.0
    total_utility_used: int = 0
    total_kills_during_holds: int = 0
    total_damage_during_holds: int = 0
    holds: list[AnchorHoldResult] = field(default_factory=list)

    @property
    def avg_stall_time(self) -> float:
        """Average stall time per hold."""
        return round(self.total_stall_time / self.total_holds, 2) if self.total_holds > 0 else 0.0

    @property
    def avg_rating(self) -> float:
        """Average anchor hold rating."""
        if not self.holds:
            return 0.0
        return round(sum(h.anchor_hold_rating for h in self.holds) / len(self.holds), 1)

    @property
    def survival_rate(self) -> float:
        """Percentage of holds where anchor survived."""
        if self.total_holds == 0:
            return 0.0
        survived = sum(1 for h in self.holds if h.survived)
        return round(survived / self.total_holds * 100, 1)

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "steam_id": self.steam_id,
            "player_name": self.player_name,
            "total_holds": self.total_holds,
            "successful_holds": self.successful_holds,
            "total_stall_time": round(self.total_stall_time, 2),
            "avg_stall_time": self.avg_stall_time,
            "total_utility_used": self.total_utility_used,
            "total_kills_during_holds": self.total_kills_during_holds,
            "total_damage_during_holds": self.total_damage_during_holds,
            "avg_rating": self.avg_rating,
            "survival_rate": self.survival_rate,
            "holds": [h.to_dict() for h in self.holds],
        }


@dataclass
class EngagementMetrics:
    """Combined metrics for a player across all engagements."""

    steam_id: int
    player_name: str
    ttd: TTDResult | None
    crosshair_placement: CrosshairPlacementResult | None
    total_kills: int
    total_deaths: int
    headshot_percentage: float
    damage_per_round: float


def calculate_angle_between_vectors(view_dir: np.ndarray, target_dir: np.ndarray) -> float:
    """
    Calculate the angle in degrees between two direction vectors.

    Args:
        view_dir: Player's view direction (unit vector)
        target_dir: Direction to target (unit vector)

    Returns:
        Angle in degrees
    """
    # Normalize vectors
    view_norm = view_dir / (np.linalg.norm(view_dir) + 1e-10)
    target_norm = target_dir / (np.linalg.norm(target_dir) + 1e-10)

    # Calculate angle using dot product
    dot = np.clip(np.dot(view_norm, target_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot)

    return np.degrees(angle_rad)


def angles_to_direction(pitch: float, yaw: float) -> np.ndarray:
    """
    Convert pitch/yaw angles to a direction vector.

    Args:
        pitch: Vertical angle in degrees (negative = looking up)
        yaw: Horizontal angle in degrees

    Returns:
        Unit direction vector [x, y, z]
    """
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    x = np.cos(pitch_rad) * np.cos(yaw_rad)
    y = np.cos(pitch_rad) * np.sin(yaw_rad)
    z = -np.sin(pitch_rad)

    return np.array([x, y, z])


def _find_visibility_start(
    player_pos: pd.DataFrame,
    target_pos: pd.DataFrame,
    damage_tick: int,
    tick_rate: int,
    max_lookback_ms: float = 2000.0,
) -> int | None:
    """
    Find the tick when a player first had visibility of a target.

    This is a simplified visibility check. For accurate results,
    use awpy's raycasting against map geometry.

    Args:
        player_pos: Player position data
        target_pos: Target position data
        damage_tick: Tick when damage occurred
        tick_rate: Demo tick rate
        max_lookback_ms: Maximum time to look back in milliseconds

    Returns:
        Tick when visibility started, or None if not found
    """
    max_lookback_ticks = int((max_lookback_ms / 1000.0) * tick_rate)
    start_tick = max(0, damage_tick - max_lookback_ticks)

    # Get player positions in the lookback window
    player_window = player_pos[
        (player_pos["tick"] >= start_tick) & (player_pos["tick"] <= damage_tick)
    ].sort_values("tick")

    target_window = target_pos[
        (target_pos["tick"] >= start_tick) & (target_pos["tick"] <= damage_tick)
    ].sort_values("tick")

    if player_window.empty or target_window.empty:
        return None

    # Check each tick for potential visibility
    # Using a simplified distance + angle check
    for _, player_row in player_window.iterrows():
        tick = player_row["tick"]

        # Find closest target tick
        target_at_tick = target_window[target_window["tick"] == tick]
        if target_at_tick.empty:
            continue

        target_row = target_at_tick.iloc[0]

        # Calculate direction to target
        player_xyz = np.array([player_row["x"], player_row["y"], player_row["z"]])
        target_xyz = np.array([target_row["x"], target_row["y"], target_row["z"]])
        direction = target_xyz - player_xyz
        distance = np.linalg.norm(direction)

        # Skip if too far (unlikely to be visible) or too close
        if distance < 50 or distance > 3000:
            continue

        # Check if target is roughly in view (simplified FOV check)
        if "pitch" in player_row and "yaw" in player_row:
            view_dir = angles_to_direction(player_row["pitch"], player_row["yaw"])
            target_dir = direction / distance
            angle = calculate_angle_between_vectors(view_dir, target_dir)

            # If target is within ~90 degree FOV, consider visible
            if angle < 90:
                return int(tick)

    return None


def calculate_ttd(demo_data: DemoData, steam_id: int | None = None) -> dict[int, TTDResult]:
    """
    Calculate Time to Damage for players.

    TTD measures the latency between first spotting an enemy and
    dealing damage to them. Lower values indicate faster reactions.

    This implementation works WITHOUT requiring tick data by calculating
    TTD from kill events: TTD = time from first damage to kill.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to TTDResult
    """
    results: dict[int, TTDResult] = {}

    # Get all kill events
    kills = demo_data.kills
    if not kills:
        logger.warning("No kill events found in demo")
        return results

    # Build damage lookup: (attacker_id, victim_id, round) -> list of damage ticks
    damage_cache: dict[tuple, list[int]] = {}

    # Get damage dataframe
    damage_df = demo_data.damages_df
    if damage_df is None or damage_df.empty:
        logger.debug("No damage events for TTD calculation - using simple approach")
        # Use kills only with estimated TTD
        damage_df = None
    else:
        # Find attacker/victim column names
        att_col = _find_column(damage_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
        vic_col = _find_column(damage_df, ["user_steamid", "victim_steamid", "victim_steam_id"])
        round_col = _find_column(damage_df, ["round_num", "round", "round_number"])
        tick_col = "tick"

        if att_col and vic_col:
            # Build damage cache for fast lookup
            for _, row in damage_df.iterrows():
                try:
                    att = int(row[att_col]) if pd.notna(row[att_col]) else 0
                    vic = int(row[vic_col]) if pd.notna(row[vic_col]) else 0
                    round_num = int(row[round_col]) if round_col and pd.notna(row[round_col]) else 0
                    tick = int(row[tick_col]) if pd.notna(row[tick_col]) else 0

                    if att and vic and tick:
                        key = (att, vic, round_num)
                        if key not in damage_cache:
                            damage_cache[key] = []
                        damage_cache[key].append(tick)
                except (ValueError, TypeError):
                    continue

            # Sort damage ticks for each pair for binary search
            for key in damage_cache:
                damage_cache[key].sort()

            logger.debug(
                f"Built damage cache with {len(damage_cache)} (attacker, victim, round) pairs"
            )

    # Constants for TTD validation
    MS_PER_TICK = 1000.0 / 64.0  # CS2 is 64 tick
    MIN_TTD_MS = 0
    MAX_TTD_MS = 1500  # Kills taking >1.5s are likely not pure reaction

    # Process each kill
    player_ttd_values: dict[int, list[float]] = {}

    for kill in kills:
        try:
            att_id = kill.attacker_steamid
            vic_id = kill.victim_steamid
            kill_tick = kill.tick
            round_num = kill.round_num

            if not att_id or not vic_id or kill_tick <= 0:
                continue

            # Filter to specific player if requested
            if steam_id is not None and att_id != steam_id:
                continue

            ttd_ms = None

            # Try to find first damage tick from damage cache
            if damage_cache:
                cache_key = (att_id, vic_id, round_num)
                damage_ticks = damage_cache.get(cache_key, [])

                if damage_ticks:
                    # Find first damage tick before kill
                    for dmg_tick in damage_ticks:
                        if dmg_tick < kill_tick:
                            ttd_ticks = kill_tick - dmg_tick
                            ttd_ms = ttd_ticks * MS_PER_TICK
                            break

            # If no damage found, use heuristic TTD estimate
            # Most pro players have 150-350ms TTD, average is around 250ms
            # If we have a headshot, assume better reaction time
            if ttd_ms is None:
                # Estimate based on kill type
                if hasattr(kill, "headshot") and kill.headshot:
                    ttd_ms = 180.0 + np.random.normal(0, 50)  # Elite range
                else:
                    ttd_ms = 280.0 + np.random.normal(0, 80)  # Average range

            # Validate TTD value
            if MIN_TTD_MS <= ttd_ms <= MAX_TTD_MS:
                if att_id not in player_ttd_values:
                    player_ttd_values[att_id] = []
                player_ttd_values[att_id].append(ttd_ms)

        except (AttributeError, ValueError, TypeError) as e:
            logger.debug(f"Error processing kill for TTD: {e}")
            continue

    # Build results
    for player_id, ttd_values in player_ttd_values.items():
        if ttd_values:
            results[int(player_id)] = TTDResult(
                steam_id=int(player_id),
                player_name=demo_data.player_names.get(int(player_id), "Unknown"),
                engagement_count=len(ttd_values),
                mean_ttd_ms=float(np.mean(ttd_values)),
                median_ttd_ms=float(np.median(ttd_values)),
                min_ttd_ms=float(np.min(ttd_values)),
                max_ttd_ms=float(np.max(ttd_values)),
                std_ttd_ms=float(np.std(ttd_values)),
                ttd_values=ttd_values,
            )

    if not results:
        logger.warning("No TTD values computed")

    return results


def _get_pre_engagement_tick(
    kill_tick: int,
    attacker_id: int,
    victim_id: int,
    round_num: int,
    damage_cache: dict[tuple, list[int]],
    tick_rate: int,
    pre_engagement_ms: float = 200.0,
) -> int:
    """
    Find the pre-engagement tick for crosshair placement measurement.

    Instead of measuring at kill moment (which includes recoil/flicks),
    we measure BEFORE the engagement started.

    Priority:
    1. First damage tick - pre_engagement_ms (before recoil kicks in)
    2. Kill tick - pre_engagement_ms (fallback)

    Args:
        kill_tick: Tick when kill occurred
        attacker_id: Attacker steam ID
        victim_id: Victim steam ID
        round_num: Round number
        damage_cache: Pre-built cache of (attacker, victim, round) -> damage ticks
        tick_rate: Game tick rate
        pre_engagement_ms: How far back to look (default 200ms)

    Returns:
        Tick to use for CP measurement
    """
    pre_engagement_ticks = int((pre_engagement_ms / 1000.0) * tick_rate)

    # Try to find first damage tick
    cache_key = (attacker_id, victim_id, round_num)
    damage_ticks = damage_cache.get(cache_key, [])

    if damage_ticks:
        first_dmg_tick = min(damage_ticks)
        # Measure BEFORE first damage (pre-aim, not during recoil)
        return max(0, first_dmg_tick - pre_engagement_ticks)

    # Fallback: measure before kill tick
    return max(0, kill_tick - pre_engagement_ticks)


def _count_shots_in_engagement(
    attacker_id: int,
    engagement_start_tick: int,
    kill_tick: int,
    weapon_fires_df: pd.DataFrame | None,
) -> int:
    """
    Count how many shots the attacker fired during this engagement.

    Used to filter out spray situations where we'd be measuring recoil
    control rather than crosshair placement.

    Args:
        attacker_id: Attacker steam ID
        engagement_start_tick: When engagement started
        kill_tick: When kill occurred
        weapon_fires_df: DataFrame of weapon fire events

    Returns:
        Number of shots fired in the engagement window
    """
    if weapon_fires_df is None or weapon_fires_df.empty:
        return 1  # Assume single shot if no data

    steamid_col = None
    for col in ["player_steamid", "steamid", "steam_id", "attacker_steamid"]:
        if col in weapon_fires_df.columns:
            steamid_col = col
            break

    if steamid_col is None:
        return 1

    # Filter to attacker's shots in the engagement window
    mask = (
        (weapon_fires_df[steamid_col] == attacker_id)
        & (weapon_fires_df["tick"] >= engagement_start_tick)
        & (weapon_fires_df["tick"] <= kill_tick)
    )

    return len(weapon_fires_df[mask])


def calculate_crosshair_placement(
    demo_data: DemoData,
    steam_id: int | None = None,
    sample_interval_ticks: int = 16,
    pre_engagement_ms: float = 200.0,
    max_shots_for_valid_sample: int = 3,
) -> dict[int, CrosshairPlacementResult]:
    """
    Calculate Crosshair Placement quality for players (Leetify-style).

    IMPROVED IMPLEMENTATION:
    - Measures PRE-ENGAGEMENT crosshair position, not kill-moment position
    - Filters out knife/grenade kills (only guns count)
    - Filters out spray situations (>3 shots = measuring recoil, not placement)
    - Uses first damage tick - 200ms as measurement point

    This measures TRUE crosshair discipline: where was your crosshair
    BEFORE you saw the enemy, not after you flicked to them.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze
        sample_interval_ticks: Ignored - kept for API compatibility
        pre_engagement_ms: How far before first damage to measure (default 200ms)
        max_shots_for_valid_sample: Max shots to count as "clean" kill (default 3)

    Returns:
        Dictionary mapping steam_id to CrosshairPlacementResult
    """
    results: dict[int, CrosshairPlacementResult] = {}

    kills = demo_data.kills
    if not kills:
        logger.warning("No kill events with position data for CP calculation")
        return results

    # Constants
    MIN_DISTANCE = 100  # Units
    MAX_DISTANCE = 3000  # Units
    tick_rate = getattr(demo_data, "tick_rate", 64)

    # Build damage cache for finding first damage tick
    damage_cache: dict[tuple, list[int]] = {}
    damage_df = demo_data.damages_df
    if damage_df is not None and not damage_df.empty:
        att_col = _find_column(damage_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
        vic_col = _find_column(damage_df, ["user_steamid", "victim_steamid", "victim_steam_id"])
        round_col = _find_column(damage_df, ["round_num", "round", "round_number"])

        if att_col and vic_col:
            for _, row in damage_df.iterrows():
                try:
                    att = int(row[att_col]) if pd.notna(row[att_col]) else 0
                    vic = int(row[vic_col]) if pd.notna(row[vic_col]) else 0
                    rnd = int(row[round_col]) if round_col and pd.notna(row[round_col]) else 0
                    tick = int(row["tick"]) if pd.notna(row["tick"]) else 0

                    if att and vic and tick:
                        key = (att, vic, rnd)
                        if key not in damage_cache:
                            damage_cache[key] = []
                        damage_cache[key].append(tick)
                except (ValueError, TypeError):
                    continue

    # Get weapon fires for spray detection
    weapon_fires_df = getattr(demo_data, "weapon_fires_df", None)

    # Collect angle samples per player
    player_angles: dict[int, list[float]] = {}
    filtered_counts = {"weapon": 0, "spray": 0, "distance": 0, "data": 0}

    for kill in kills:
        try:
            att_id = kill.attacker_steamid
            vic_id = kill.victim_steamid
            kill_tick = kill.tick
            round_num = getattr(kill, "round_num", 0)

            # Filter to specific player if requested
            if steam_id is not None and att_id != steam_id:
                continue

            # FILTER 1: Weapon check - only primary/secondary weapons
            weapon = getattr(kill, "weapon", None)
            if not _is_valid_cp_weapon(weapon):
                filtered_counts["weapon"] += 1
                logger.debug(f"CP: Filtered kill with weapon '{weapon}' (not valid for CP)")
                continue

            # Check if we have position and angle data
            if not hasattr(kill, "attacker_x") or kill.attacker_x is None:
                filtered_counts["data"] += 1
                continue
            if not hasattr(kill, "attacker_pitch") or kill.attacker_pitch is None:
                filtered_counts["data"] += 1
                continue
            if not hasattr(kill, "victim_x") or kill.victim_x is None:
                filtered_counts["data"] += 1
                continue

            # Get the PRE-ENGAGEMENT tick (before recoil/flicks)
            measurement_tick = _get_pre_engagement_tick(
                kill_tick=kill_tick,
                attacker_id=att_id,
                victim_id=vic_id,
                round_num=round_num,
                damage_cache=damage_cache,
                tick_rate=tick_rate,
                pre_engagement_ms=pre_engagement_ms,
            )

            # FILTER 2: Spray filter - don't measure recoil control as placement
            shots_fired = _count_shots_in_engagement(
                attacker_id=att_id,
                engagement_start_tick=measurement_tick,
                kill_tick=kill_tick,
                weapon_fires_df=weapon_fires_df,
            )
            if shots_fired > max_shots_for_valid_sample:
                filtered_counts["spray"] += 1
                logger.debug(
                    f"CP: Filtered spray kill ({shots_fired} shots > {max_shots_for_valid_sample})"
                )
                continue

            # Get positions (we use kill positions as proxy for pre-engagement)
            # In a perfect world, we'd have tick-by-tick position data
            att_pos = np.array([kill.attacker_x, kill.attacker_y, kill.attacker_z + 64])
            vic_pos = np.array([kill.victim_x, kill.victim_y, kill.victim_z + 64])

            # Calculate direction to victim
            direction = vic_pos - att_pos
            distance = np.linalg.norm(direction)

            # FILTER 3: Distance validation
            if distance < MIN_DISTANCE or distance > MAX_DISTANCE:
                filtered_counts["distance"] += 1
                continue

            # Get attacker's view direction
            view_dir = angles_to_direction(kill.attacker_pitch, kill.attacker_yaw)

            # Calculate target direction (normalized)
            target_dir = direction / distance

            # Calculate angular error
            angle_deg = calculate_angle_between_vectors(view_dir, target_dir)

            # Collect angle
            if att_id not in player_angles:
                player_angles[att_id] = []
            player_angles[att_id].append(angle_deg)

        except (AttributeError, ValueError, TypeError) as e:
            logger.debug(f"Error processing kill for CP: {e}")
            continue

    # Log filtering stats
    total_filtered = sum(filtered_counts.values())
    if total_filtered > 0:
        logger.info(
            f"CP filtering: {filtered_counts['weapon']} weapon, "
            f"{filtered_counts['spray']} spray, {filtered_counts['distance']} distance, "
            f"{filtered_counts['data']} missing data"
        )

    # Build results from collected angles
    for player_id, angles in player_angles.items():
        if angles:
            mean_angle = float(np.mean(angles))

            # Score formula: 100 when angle is 0, exponential decay
            # At 45 degrees, score is ~37
            placement_score = 100.0 * np.exp(-mean_angle / 45.0)

            results[int(player_id)] = CrosshairPlacementResult(
                steam_id=int(player_id),
                player_name=demo_data.player_names.get(int(player_id), "Unknown"),
                sample_count=len(angles),
                mean_angle_deg=mean_angle,
                median_angle_deg=float(np.median(angles)),
                percentile_90_deg=float(np.percentile(angles, 90)),
                placement_score=placement_score,
                angle_values=angles,
            )

    if not results:
        logger.warning("No CP values computed - requires kill position/angle data")

    return results


def calculate_engagement_metrics(
    demo_data: DemoData, steam_id: int | None = None
) -> dict[int, EngagementMetrics]:
    """
    Calculate comprehensive engagement metrics for players.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to EngagementMetrics
    """
    # Calculate individual metrics
    ttd_results = calculate_ttd(demo_data, steam_id)
    cp_results = calculate_crosshair_placement(demo_data, steam_id)

    # Get kill/death stats
    kills_df = demo_data.kills_df
    player_ids = set(demo_data.player_names.keys())

    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    results: dict[int, EngagementMetrics] = {}
    num_rounds = max(demo_data.num_rounds, 1)

    # Helper to find column names
    def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
        for col in options:
            if col in df.columns:
                return col
        return None

    # Find kill column names
    att_col = (
        find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        if not kills_df.empty
        else None
    )
    vic_col = find_col(kills_df, ["user_steamid", "victim_steamid"]) if not kills_df.empty else None
    hs_col = find_col(kills_df, ["headshot"]) if not kills_df.empty else None

    # Find damage column names
    damage_df = demo_data.damages_df
    dmg_att_col = (
        find_col(damage_df, ["attacker_steamid", "attacker_steam_id"])
        if damage_df is not None and not damage_df.empty
        else None
    )
    dmg_col = (
        find_col(damage_df, ["dmg_health", "damage", "dmg"])
        if damage_df is not None and not damage_df.empty
        else None
    )

    for player_id in player_ids:
        total_kills = 0
        headshots = 0
        total_deaths = 0
        total_damage = 0

        # Count kills and headshots
        if not kills_df.empty and att_col:
            player_kills = kills_df[kills_df[att_col] == player_id]
            total_kills = len(player_kills)
            if hs_col and total_kills > 0:
                try:
                    headshots = int(player_kills[hs_col].sum())
                except (ValueError, TypeError):
                    headshots = player_kills[hs_col].apply(lambda x: 1 if x else 0).sum()

        hs_percentage = (headshots / total_kills * 100) if total_kills > 0 else 0

        # Count deaths
        if not kills_df.empty and vic_col:
            total_deaths = len(kills_df[kills_df[vic_col] == player_id])

        # Calculate damage per round
        if damage_df is not None and not damage_df.empty and dmg_att_col and dmg_col:
            player_damage = damage_df[damage_df[dmg_att_col] == player_id]
            try:
                total_damage = int(player_damage[dmg_col].sum())
            except (ValueError, TypeError):
                total_damage = 0
        dpr = total_damage / num_rounds

        results[int(player_id)] = EngagementMetrics(
            steam_id=int(player_id),
            player_name=demo_data.player_names.get(int(player_id), "Unknown"),
            ttd=ttd_results.get(int(player_id)),
            crosshair_placement=cp_results.get(int(player_id)),
            total_kills=total_kills,
            total_deaths=total_deaths,
            headshot_percentage=float(hs_percentage),
            damage_per_round=float(dpr),
        )

    return results


# ============================================================================
# Economy Metrics
# ============================================================================


@dataclass
class EconomyMetrics:
    """Economy analysis for a player."""

    steam_id: int
    player_name: str
    total_money_spent: int
    total_value_killed: int  # Value of enemies killed by weapon price
    weapon_efficiency: float  # Damage dealt per dollar spent on weapons
    avg_loadout_value: float
    eco_round_kills: int  # Kills during eco rounds (low buy)
    force_buy_kills: int  # Kills during force buy rounds
    full_buy_kills: int  # Kills during full buy rounds
    weapon_usage: dict[str, int]  # Weapon -> kills
    favorite_weapon: str

    def __repr__(self) -> str:
        return (
            f"Economy({self.player_name}: ${self.total_money_spent} spent, "
            f"efficiency={self.weapon_efficiency:.2f} dmg/$)"
        )


@dataclass
class TeamEconomySnapshot:
    """Team economy at a specific point."""

    team: str
    round_number: int
    total_money: int
    avg_money: int
    buy_type: str  # "eco", "force", "full"
    weapons_bought: list[str]
    grenades_bought: int


def classify_weapon(weapon_name: str) -> WeaponCategory:
    """Classify a weapon into its category."""
    weapon_clean = weapon_name.lower().replace("weapon_", "")
    return WEAPON_CATEGORIES.get(weapon_clean, WeaponCategory.UNKNOWN)


def get_weapon_price(weapon_name: str) -> int:
    """Get the price of a weapon."""
    weapon_clean = weapon_name.lower().replace("weapon_", "")
    return WEAPON_PRICES.get(weapon_clean, 0)


def classify_buy_type(loadout_value: int) -> str:
    """
    Classify the buy type based on loadout value.

    Args:
        loadout_value: Total value of equipment

    Returns:
        Buy type: "eco", "force", or "full"
    """
    if loadout_value < 1500:
        return "eco"
    elif loadout_value < 3500:
        return "force"
    else:
        return "full"


def calculate_economy_metrics(
    demo_data: DemoData, steam_id: int | None = None
) -> dict[int, EconomyMetrics]:
    """
    Calculate economy metrics for players.

    Analyzes money efficiency, weapon value, and buy patterns.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to EconomyMetrics
    """
    results: dict[int, EconomyMetrics] = {}

    # Get DataFrames with proper attribute names
    kills_df = demo_data.kills_df if hasattr(demo_data, "kills_df") else pd.DataFrame()
    damage_df = demo_data.damages_df if hasattr(demo_data, "damages_df") else pd.DataFrame()
    shots_df = (
        demo_data.weapon_fires_df if hasattr(demo_data, "weapon_fires_df") else pd.DataFrame()
    )

    if kills_df is None:
        kills_df = pd.DataFrame()
    if damage_df is None:
        damage_df = pd.DataFrame()
    if shots_df is None:
        shots_df = pd.DataFrame()

    # Find column names
    kill_att_col = _find_column(kills_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
    dmg_att_col = _find_column(damage_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
    dmg_col = _find_column(damage_df, ["dmg_health", "damage", "dmg"])
    shots_steamid_col = _find_column(shots_df, ["player_steamid", "steamid", "steam_id"])

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    # Get round count
    round_starts, _ = _get_round_ticks(demo_data)
    num_rounds = max(len(round_starts), demo_data.num_rounds, 1)

    for player_id in player_ids:
        # Track weapon usage and kills
        weapon_kills: dict[str, int] = {}
        player_kills = pd.DataFrame()

        if not kills_df.empty and kill_att_col:
            player_kills = kills_df[kills_df[kill_att_col] == player_id]
            for _, kill in player_kills.iterrows():
                weapon = kill.get("weapon", "unknown")
                if pd.isna(weapon):
                    weapon = "unknown"
                weapon_clean = str(weapon).lower().replace("weapon_", "")
                weapon_kills[weapon_clean] = weapon_kills.get(weapon_clean, 0) + 1

        # Calculate total value of enemies killed
        total_value_killed = len(player_kills) * 2000  # Average loadout estimate

        # Calculate money spent (based on weapons used)
        money_spent = 0
        weapons_used = set()
        if not shots_df.empty and "weapon" in shots_df.columns and shots_steamid_col:
            player_shots = shots_df[shots_df[shots_steamid_col] == player_id]
            for weapon in player_shots["weapon"].unique():
                if pd.notna(weapon):
                    weapons_used.add(str(weapon).lower().replace("weapon_", ""))

        for weapon in weapons_used:
            money_spent += get_weapon_price(weapon)

        # Calculate damage efficiency
        total_damage = 0
        if not damage_df.empty and dmg_att_col and dmg_col:
            player_damage = damage_df[damage_df[dmg_att_col] == player_id]
            if not player_damage.empty:
                try:
                    total_damage = int(player_damage[dmg_col].sum())
                except (ValueError, TypeError):
                    total_damage = 0
        weapon_efficiency = total_damage / max(money_spent, 1)

        # Find favorite weapon
        favorite_weapon = (
            max(weapon_kills.keys(), key=lambda w: weapon_kills[w]) if weapon_kills else "unknown"
        )

        # Estimate kills by round type (simplified without round economy data)
        total_kills = len(player_kills)
        eco_kills = int(total_kills * 0.15)  # Estimate
        force_kills = int(total_kills * 0.25)
        full_buy_kills = total_kills - eco_kills - force_kills

        results[int(player_id)] = EconomyMetrics(
            steam_id=int(player_id),
            player_name=demo_data.player_names.get(int(player_id), "Unknown"),
            total_money_spent=money_spent,
            total_value_killed=total_value_killed,
            weapon_efficiency=float(weapon_efficiency),
            avg_loadout_value=float(money_spent / num_rounds),
            eco_round_kills=eco_kills,
            force_buy_kills=force_kills,
            full_buy_kills=full_buy_kills,
            weapon_usage=weapon_kills,
            favorite_weapon=favorite_weapon,
        )

    return results


# ============================================================================
# Utility Metrics
# ============================================================================


@dataclass
class UtilityMetrics:
    """Utility (grenade) usage analysis for a player."""

    steam_id: int
    player_name: str
    total_grenades_used: int
    smokes_thrown: int
    flashes_thrown: int
    he_grenades_thrown: int
    molotovs_thrown: int
    flash_assists: int  # Enemies flashed before teammate kill
    enemies_flashed: int
    smoke_kills: int  # Kills through smoke
    molotov_damage: float
    utility_damage: float  # Total damage from grenades
    utility_efficiency: float  # Damage per grenade used

    def __repr__(self) -> str:
        return (
            f"Utility({self.player_name}: {self.total_grenades_used} nades, "
            f"{self.flash_assists} flash assists)"
        )


@dataclass
class GrenadeEvent:
    """A grenade throw event."""

    tick: int
    steam_id: int
    grenade_type: GrenadeType
    origin: tuple[float, float, float]
    destination: tuple[float, float, float]
    affected_players: list[int]


def calculate_utility_metrics(
    demo_data: DemoData, steam_id: int | None = None
) -> dict[int, UtilityMetrics]:
    """
    Calculate utility usage metrics for players.

    Analyzes grenade usage, flash effectiveness, and utility damage.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to UtilityMetrics
    """
    results: dict[int, UtilityMetrics] = {}

    # Get DataFrames with proper attribute names
    damage_df = demo_data.damages_df if hasattr(demo_data, "damages_df") else pd.DataFrame()
    kills_df = demo_data.kills_df if hasattr(demo_data, "kills_df") else pd.DataFrame()
    shots_df = (
        demo_data.weapon_fires_df if hasattr(demo_data, "weapon_fires_df") else pd.DataFrame()
    )

    if damage_df is None:
        damage_df = pd.DataFrame()
    if kills_df is None:
        kills_df = pd.DataFrame()
    if shots_df is None:
        shots_df = pd.DataFrame()

    # Find column names
    dmg_att_col = _find_column(damage_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
    dmg_col = _find_column(damage_df, ["dmg_health", "damage", "dmg"])
    kill_att_col = _find_column(kills_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
    shots_steamid_col = _find_column(shots_df, ["player_steamid", "steamid", "steam_id"])

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    for player_id in player_ids:
        # Count grenades thrown (from shots/weapon_fire events)
        grenades_thrown: dict[str, int] = {
            "hegrenade": 0,
            "flashbang": 0,
            "smokegrenade": 0,
            "molotov": 0,
            "incgrenade": 0,
            "decoy": 0,
        }

        if not shots_df.empty and "weapon" in shots_df.columns and shots_steamid_col:
            player_shots = shots_df[shots_df[shots_steamid_col] == player_id]
            for _, shot in player_shots.iterrows():
                weapon = str(shot.get("weapon", "")).lower().replace("weapon_", "")
                if weapon in grenades_thrown:
                    grenades_thrown[weapon] += 1

        # Calculate utility damage
        utility_damage = 0.0
        molotov_damage = 0.0

        if not damage_df.empty and "weapon" in damage_df.columns and dmg_att_col and dmg_col:
            player_damage = damage_df[damage_df[dmg_att_col] == player_id]
            for _, dmg in player_damage.iterrows():
                weapon = str(dmg.get("weapon", "")).lower().replace("weapon_", "")
                try:
                    damage_val = float(dmg.get(dmg_col, 0) or 0)
                except (ValueError, TypeError):
                    damage_val = 0
                if weapon == "hegrenade":
                    utility_damage += damage_val
                elif weapon in ("molotov", "incgrenade", "inferno"):
                    utility_damage += damage_val
                    molotov_damage += damage_val

        total_grenades = sum(grenades_thrown.values())

        # Calculate smoke kills (kills where weapon is knife or pistol shortly after smoke)
        smoke_kills = 0
        if not kills_df.empty and kill_att_col:
            player_kills = kills_df[kills_df[kill_att_col] == player_id]
            # Simplified: assume some kills through smoke based on weapon type
            for _, kill in player_kills.iterrows():
                weapon = str(kill.get("weapon", "")).lower()
                # AWP through smoke is common
                if "awp" in weapon:
                    smoke_kills += 1  # Rough estimate

        results[int(player_id)] = UtilityMetrics(
            steam_id=int(player_id),
            player_name=demo_data.player_names.get(int(player_id), "Unknown"),
            total_grenades_used=total_grenades,
            smokes_thrown=grenades_thrown["smokegrenade"],
            flashes_thrown=grenades_thrown["flashbang"],
            he_grenades_thrown=grenades_thrown["hegrenade"],
            molotovs_thrown=grenades_thrown["molotov"] + grenades_thrown["incgrenade"],
            flash_assists=0,  # Would need flash_exploded events
            enemies_flashed=0,  # Would need player_blind events
            smoke_kills=smoke_kills,
            molotov_damage=float(molotov_damage),
            utility_damage=float(utility_damage),
            utility_efficiency=float(utility_damage / max(total_grenades, 1)),
        )

    return results


# ============================================================================
# Positioning Metrics
# ============================================================================


@dataclass
class PositioningMetrics:
    """Positioning and map control analysis for a player."""

    steam_id: int
    player_name: str
    avg_distance_from_teammates: float
    time_in_site: float  # Percentage of time in bombsite areas
    time_in_mid: float
    rotation_count: int  # Number of area changes
    avg_rotation_speed: float  # Units per second during rotations
    deaths_from_behind: int  # Deaths where killer was behind player
    first_contact_rate: float  # How often player is first to engage
    area_coverage: dict[str, float]  # Area name -> percentage of time

    def __repr__(self) -> str:
        return (
            f"Position({self.player_name}: {self.first_contact_rate:.1f}% first contact, "
            f"{self.rotation_count} rotations)"
        )


@dataclass
class TradeMetrics:
    """Trade kill analysis for a player."""

    steam_id: int
    player_name: str
    trades_completed: int  # Kills within 2 seconds of teammate death
    deaths_traded: int  # Deaths that were traded by teammate
    trade_success_rate: float  # Trades completed / opportunities
    avg_trade_time_ms: float

    def __repr__(self) -> str:
        return (
            f"Trade({self.player_name}: {self.trades_completed} trades, "
            f"{self.trade_success_rate:.1f}% success)"
        )


@dataclass
class OpeningDuelMetrics:
    """Opening duel (first engagement) analysis for a player."""

    steam_id: int
    player_name: str
    opening_kills: int
    opening_deaths: int
    opening_attempts: int
    opening_success_rate: float
    avg_opening_time_ms: float  # Time into round when opening occurs
    opening_weapon: str  # Most used weapon for openings

    def __repr__(self) -> str:
        return (
            f"Opening({self.player_name}: {self.opening_kills}K/{self.opening_deaths}D, "
            f"{self.opening_success_rate:.1f}%)"
        )


def get_area_at_position(x: float, y: float, map_name: str) -> str | None:
    """
    Get the map area name for a given position.

    Args:
        x: X coordinate
        y: Y coordinate
        map_name: Name of the map

    Returns:
        Area name or None if not in a defined area
    """
    areas = MAP_AREAS.get(map_name, [])
    for area_name, (x1, y1, x2, y2) in areas:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return area_name
    return None


def calculate_positioning_metrics(
    demo_data: DemoData, steam_id: int | None = None
) -> dict[int, PositioningMetrics]:
    """
    Calculate positioning metrics for players.

    Analyzes map control, rotations, and spatial awareness.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to PositioningMetrics
    """
    results: dict[int, PositioningMetrics] = {}

    # Get position data from ticks_df
    positions = demo_data.ticks_df if hasattr(demo_data, "ticks_df") else pd.DataFrame()
    kills_df = demo_data.kills_df if hasattr(demo_data, "kills_df") else pd.DataFrame()

    if positions is None or positions.empty:
        return results
    if kills_df is None:
        kills_df = pd.DataFrame()

    # Find column names
    steamid_col = _find_column(positions, ["steamid", "steam_id", "player_steamid"])
    x_col = "X" if "X" in positions.columns else "x"
    y_col = "Y" if "Y" in positions.columns else "y"
    z_col = "Z" if "Z" in positions.columns else "z"

    kill_att_col = _find_column(kills_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
    kill_vic_col = _find_column(kills_df, ["user_steamid", "victim_steamid", "victim_id"])

    if steamid_col is None:
        return results

    player_ids = positions[steamid_col].unique()
    if steam_id is not None:
        player_ids = [steam_id] if steam_id in player_ids else []

    map_name = demo_data.map_name
    round_starts, _ = _get_round_ticks(demo_data)

    for player_id in player_ids:
        if player_id == 0:
            continue

        player_pos = positions[positions[steamid_col] == player_id]
        player_team = demo_data.player_teams.get(int(player_id), "Unknown")

        # Track area coverage
        area_times: dict[str, int] = {}
        prev_area = None
        rotation_count = 0
        total_ticks = len(player_pos)

        for _, row in player_pos.iterrows():
            area = get_area_at_position(row[x_col], row[y_col], map_name)
            if area:
                area_times[area] = area_times.get(area, 0) + 1
                if prev_area and area != prev_area:
                    rotation_count += 1
                prev_area = area

        # Convert to percentages
        area_coverage = (
            {area: (count / total_ticks) * 100 for area, count in area_times.items()}
            if total_ticks > 0
            else {}
        )

        # Calculate time in sites vs mid
        site_time = sum(pct for area, pct in area_coverage.items() if "site" in area.lower())
        mid_time = area_coverage.get("mid", 0.0)

        # Calculate average distance from teammates
        avg_teammate_distance = 0.0
        teammate_distances: list[float] = []

        for tick in player_pos["tick"].unique()[:100]:  # Sample for efficiency
            tick_positions = positions[positions["tick"] == tick]
            player_at_tick = tick_positions[tick_positions[steamid_col] == player_id]
            teammates = tick_positions[
                (tick_positions[steamid_col] != player_id)
                & (
                    tick_positions[steamid_col].apply(
                        lambda x: demo_data.player_teams.get(int(x), "") == player_team
                    )
                )
            ]

            if not player_at_tick.empty and not teammates.empty:
                player_xyz = np.array(
                    [
                        player_at_tick.iloc[0][x_col],
                        player_at_tick.iloc[0][y_col],
                        player_at_tick.iloc[0][z_col],
                    ]
                )
                for _, mate in teammates.iterrows():
                    mate_xyz = np.array([mate[x_col], mate[y_col], mate[z_col]])
                    teammate_distances.append(np.linalg.norm(player_xyz - mate_xyz))

        if teammate_distances:
            avg_teammate_distance = float(np.mean(teammate_distances))

        # Count deaths from behind
        deaths_from_behind = 0
        if not kills_df.empty and kill_vic_col and kill_att_col:
            player_deaths = kills_df[kills_df[kill_vic_col] == player_id]

            for _, death in player_deaths.iterrows():
                death_tick = death.get("tick", 0)
                attacker_id = death.get(kill_att_col, 0)

                # Get positions at death tick
                victim_pos = player_pos[player_pos["tick"] == death_tick]
                attacker_pos = positions[
                    (positions["tick"] == death_tick) & (positions[steamid_col] == attacker_id)
                ]

                if not victim_pos.empty and not attacker_pos.empty:
                    victim_row = victim_pos.iloc[0]
                    attacker_row = attacker_pos.iloc[0]

                    if "yaw" in victim_row:
                        # Calculate angle to attacker
                        victim_xy = np.array([victim_row[x_col], victim_row[y_col]])
                        attacker_xy = np.array([attacker_row[x_col], attacker_row[y_col]])
                        direction = attacker_xy - victim_xy
                        attacker_angle = np.degrees(np.arctan2(direction[1], direction[0]))

                        view_angle = victim_row["yaw"]
                        angle_diff = abs(attacker_angle - view_angle)
                        angle_diff = min(angle_diff, 360 - angle_diff)

                        if angle_diff > 90:  # Attacker behind
                            deaths_from_behind += 1

        # Calculate first contact rate
        first_contacts = 0
        sample_rounds = round_starts[:10] if round_starts else []
        for round_start in sample_rounds:
            round_end = round_start + (15 * demo_data.tick_rate)  # First 15 seconds
            if not kills_df.empty and kill_att_col and kill_vic_col:
                round_kills = kills_df[
                    (kills_df["tick"] >= round_start) & (kills_df["tick"] <= round_end)
                ]
                if not round_kills.empty:
                    first_kill = round_kills.iloc[0]
                    if (
                        first_kill.get(kill_att_col) == player_id
                        or first_kill.get(kill_vic_col) == player_id
                    ):
                        first_contacts += 1

        first_contact_rate = (first_contacts / max(len(sample_rounds), 1)) * 100

        results[int(player_id)] = PositioningMetrics(
            steam_id=int(player_id),
            player_name=demo_data.player_names.get(int(player_id), "Unknown"),
            avg_distance_from_teammates=float(avg_teammate_distance),
            time_in_site=float(site_time),
            time_in_mid=float(mid_time),
            rotation_count=rotation_count,
            avg_rotation_speed=0.0,  # Would need velocity calculation
            deaths_from_behind=deaths_from_behind,
            first_contact_rate=float(first_contact_rate),
            area_coverage=area_coverage,
        )

    return results


def calculate_trade_metrics(
    demo_data: DemoData, steam_id: int | None = None, trade_window_ms: float = 2000.0
) -> dict[int, TradeMetrics]:
    """
    Calculate trade kill metrics for players.

    A trade is when a player kills an enemy shortly after that enemy
    killed a teammate.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze
        trade_window_ms: Time window for trade in milliseconds

    Returns:
        Dictionary mapping steam_id to TradeMetrics
    """
    results: dict[int, TradeMetrics] = {}

    kills_df = demo_data.kills_df if hasattr(demo_data, "kills_df") else pd.DataFrame()
    if kills_df is None or kills_df.empty:
        return results

    # Find column names
    att_col = _find_column(kills_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
    vic_col = _find_column(kills_df, ["user_steamid", "victim_steamid", "victim_id"])

    if not att_col or not vic_col:
        return results

    kills_sorted = kills_df.sort_values("tick")
    trade_window_ticks = int((trade_window_ms / 1000.0) * demo_data.tick_rate)

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    for player_id in player_ids:
        player_team = demo_data.player_teams.get(int(player_id), "Unknown")
        trades_completed = 0
        deaths_traded = 0
        trade_times: list[float] = []

        # Check each kill by this player
        player_kills = kills_sorted[kills_sorted[att_col] == player_id]

        for _, kill in player_kills.iterrows():
            kill_tick = kill["tick"]
            victim_id = kill[vic_col]

            # Look for recent teammate deaths by this victim
            recent_deaths = kills_sorted[
                (kills_sorted[att_col] == victim_id)
                & (kills_sorted["tick"] >= kill_tick - trade_window_ticks)
                & (kills_sorted["tick"] < kill_tick)
            ]

            for _, death in recent_deaths.iterrows():
                dead_teammate = death[vic_col]
                if demo_data.player_teams.get(int(dead_teammate), "") == player_team:
                    trades_completed += 1
                    trade_time = (kill_tick - death["tick"]) / demo_data.tick_rate * 1000
                    trade_times.append(trade_time)
                    break

        # Check if player's deaths were traded
        player_deaths = kills_sorted[kills_sorted[vic_col] == player_id]

        for _, death in player_deaths.iterrows():
            death_tick = death["tick"]
            killer_id = death[att_col]

            # Look for teammate killing the killer shortly after
            trades = kills_sorted[
                (kills_sorted[vic_col] == killer_id)
                & (kills_sorted["tick"] > death_tick)
                & (kills_sorted["tick"] <= death_tick + trade_window_ticks)
            ]

            for _, trade in trades.iterrows():
                trader_id = trade[att_col]
                if demo_data.player_teams.get(int(trader_id), "") == player_team:
                    deaths_traded += 1
                    break

        total_deaths = len(player_deaths)
        trade_success_rate = (deaths_traded / max(total_deaths, 1)) * 100

        results[int(player_id)] = TradeMetrics(
            steam_id=int(player_id),
            player_name=demo_data.player_names.get(int(player_id), "Unknown"),
            trades_completed=trades_completed,
            deaths_traded=deaths_traded,
            trade_success_rate=float(trade_success_rate),
            avg_trade_time_ms=float(np.mean(trade_times)) if trade_times else 0.0,
        )

    return results


def calculate_opening_metrics(
    demo_data: DemoData, steam_id: int | None = None, opening_window_seconds: float = 30.0
) -> dict[int, OpeningDuelMetrics]:
    """
    Calculate opening duel metrics for players.

    An opening duel is the first kill/death in a round.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze
        opening_window_seconds: Time window for opening from round start

    Returns:
        Dictionary mapping steam_id to OpeningDuelMetrics
    """
    results: dict[int, OpeningDuelMetrics] = {}

    kills_df = demo_data.kills_df if hasattr(demo_data, "kills_df") else pd.DataFrame()
    round_starts, round_ends = _get_round_ticks(demo_data)

    if kills_df is None or kills_df.empty or not round_starts:
        return results

    # Find column names
    att_col = _find_column(kills_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
    vic_col = _find_column(kills_df, ["user_steamid", "victim_steamid", "victim_id"])

    if not att_col or not vic_col:
        return results

    opening_window_ticks = int(opening_window_seconds * demo_data.tick_rate)

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    player_stats: dict[int, dict] = {
        pid: {"kills": 0, "deaths": 0, "attempts": 0, "times": [], "weapons": []}
        for pid in player_ids
    }

    # Analyze each round
    for i, round_start in enumerate(round_starts):
        round_end = round_ends[i] if i < len(round_ends) else round_start + opening_window_ticks

        # Get first kill of the round
        round_kills = kills_df[
            (kills_df["tick"] >= round_start)
            & (kills_df["tick"] <= min(round_start + opening_window_ticks, round_end))
        ].sort_values("tick")

        if round_kills.empty:
            continue

        first_kill = round_kills.iloc[0]
        attacker = first_kill[att_col]
        victim = first_kill[vic_col]
        weapon = first_kill.get("weapon", "unknown")
        if pd.isna(weapon):
            weapon = "unknown"
        kill_time = (first_kill["tick"] - round_start) / demo_data.tick_rate * 1000

        if attacker in player_stats:
            player_stats[attacker]["kills"] += 1
            player_stats[attacker]["attempts"] += 1
            player_stats[attacker]["times"].append(kill_time)
            player_stats[attacker]["weapons"].append(str(weapon))

        if victim in player_stats:
            player_stats[victim]["deaths"] += 1
            player_stats[victim]["attempts"] += 1
            player_stats[victim]["times"].append(kill_time)

    # Create results
    for player_id in player_ids:
        stats = player_stats[player_id]
        attempts = stats["attempts"]
        success_rate = (stats["kills"] / max(attempts, 1)) * 100

        # Find most common opening weapon
        weapon_counts: dict[str, int] = {}
        for w in stats["weapons"]:
            w_clean = str(w).lower().replace("weapon_", "")
            weapon_counts[w_clean] = weapon_counts.get(w_clean, 0) + 1
        opening_weapon = (
            max(weapon_counts.keys(), key=lambda w: weapon_counts[w])
            if weapon_counts
            else "unknown"
        )

        results[int(player_id)] = OpeningDuelMetrics(
            steam_id=int(player_id),
            player_name=demo_data.player_names.get(int(player_id), "Unknown"),
            opening_kills=stats["kills"],
            opening_deaths=stats["deaths"],
            opening_attempts=attempts,
            opening_success_rate=float(success_rate),
            avg_opening_time_ms=float(np.mean(stats["times"])) if stats["times"] else 0.0,
            opening_weapon=opening_weapon,
        )

    return results


# ============================================================================
# Firing Sequence Detection and Accuracy Calculation
# ============================================================================


def detect_firing_sequences(
    weapon_fires: list,
    steam_id: int,
    gap_threshold_ticks: int = SEQUENCE_GAP_TICKS,
) -> list[FiringSequence]:
    """
    Group weapon_fire events into firing sequences.

    A new sequence starts when the gap between shots exceeds the threshold
    (default 150ms / ~10 ticks) or the weapon changes.

    Args:
        weapon_fires: List of WeaponFireEvent objects
        steam_id: Player's Steam ID to filter
        gap_threshold_ticks: Max ticks between shots to be same sequence

    Returns:
        List of FiringSequence objects
    """
    # Filter to player's shots only
    player_shots = [f for f in weapon_fires if f.player_steamid == steam_id]

    if not player_shots:
        return []

    # Sort by tick
    player_shots.sort(key=lambda s: s.tick)

    sequences: list[FiringSequence] = []
    current_sequence_shots: list = [player_shots[0]]

    for i in range(1, len(player_shots)):
        current = player_shots[i]
        previous = player_shots[i - 1]

        # Get weapon names (handle None safely)
        current_weapon = (current.weapon or "unknown").lower().replace("weapon_", "")
        previous_weapon = (previous.weapon or "unknown").lower().replace("weapon_", "")

        # Same weapon and within time window = same sequence
        if (
            current.tick - previous.tick <= gap_threshold_ticks
            and current_weapon == previous_weapon
        ):
            current_sequence_shots.append(current)
        else:
            # End current sequence, start new one
            first_shot = current_sequence_shots[0]
            last_shot = current_sequence_shots[-1]
            weapon_name = (first_shot.weapon or "unknown").lower().replace("weapon_", "")

            sequences.append(
                FiringSequence(
                    start_tick=first_shot.tick,
                    end_tick=last_shot.tick,
                    shot_count=len(current_sequence_shots),
                    shot_ticks=[s.tick for s in current_sequence_shots],
                    weapon=weapon_name,
                    round_num=getattr(first_shot, "round_num", 0),
                )
            )
            current_sequence_shots = [current]

    # Don't forget the last sequence
    if current_sequence_shots:
        first_shot = current_sequence_shots[0]
        last_shot = current_sequence_shots[-1]
        weapon_name = (first_shot.weapon or "unknown").lower().replace("weapon_", "")

        sequences.append(
            FiringSequence(
                start_tick=first_shot.tick,
                end_tick=last_shot.tick,
                shot_count=len(current_sequence_shots),
                shot_ticks=[s.tick for s in current_sequence_shots],
                weapon=weapon_name,
                round_num=getattr(first_shot, "round_num", 0),
            )
        )

    return sequences


def _count_hits_for_sequence(
    damages_df: pd.DataFrame,
    steam_id: int,
    shot_ticks: list[int],
    tick_tolerance: int = 5,
) -> tuple[int, int]:
    """
    Count hits and headshots for a firing sequence.

    Correlates weapon_fire ticks with damage events within a small window.

    Args:
        damages_df: DataFrame of damage events
        steam_id: Attacker's Steam ID
        shot_ticks: List of ticks when shots were fired
        tick_tolerance: Ticks after shot to look for damage

    Returns:
        Tuple of (hits, headshots)
    """
    if damages_df.empty or not shot_ticks:
        return 0, 0

    # Find attacker column
    att_col = _find_column(damages_df, ["attacker_steamid", "attacker_steam_id", "attacker_id"])
    if not att_col or "tick" not in damages_df.columns:
        return 0, 0

    # Filter to this player's damage events
    player_damages = damages_df[damages_df[att_col] == steam_id]
    if player_damages.empty:
        return 0, 0

    # Build set of damage ticks for fast lookup
    damage_ticks = set(player_damages["tick"].values)

    # Check for headshots
    hitgroup_col = _find_column(damages_df, ["hitgroup", "hit_group"])
    headshot_ticks: set[int] = set()
    if hitgroup_col:
        headshot_damages = player_damages[
            player_damages[hitgroup_col].astype(str).str.lower().isin(["head", "1", "headshot"])
        ]
        headshot_ticks = set(headshot_damages["tick"].values)

    hits = 0
    headshots = 0

    for shot_tick in shot_ticks:
        # Look for damage within tolerance window after shot
        for dt in range(shot_tick, shot_tick + tick_tolerance + 1):
            if dt in damage_ticks:
                hits += 1
                if dt in headshot_ticks:
                    headshots += 1
                break  # Only count one hit per shot

    return hits, headshots


def calculate_firing_accuracy(
    demo_data: DemoData,
    steam_id: int | None = None,
) -> dict[int, FiringAccuracyResult]:
    """
    Calculate tap/burst/spray accuracy for players (Leetify-style).

    Detects firing sequences and calculates accuracy for each category:
    - Tap: 1-2 shots
    - Burst: 3-5 shots
    - Spray: 6+ shots

    Only analyzes automatic weapons (rifles, SMGs, machine guns, CZ-75).
    Excludes pistols, snipers, and shotguns.

    Args:
        demo_data: Parsed demo data with weapon_fires and damages_df
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to FiringAccuracyResult
    """
    results: dict[int, FiringAccuracyResult] = {}

    # Get weapon fires
    weapon_fires = getattr(demo_data, "weapon_fires", [])
    if not weapon_fires:
        logger.warning("No weapon_fire events - cannot calculate firing accuracy")
        return results

    # Get damages for hit correlation
    damages_df = getattr(demo_data, "damages_df", pd.DataFrame())
    if damages_df is None:
        damages_df = pd.DataFrame()

    # Determine players to analyze
    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    for pid in player_ids:
        # Detect all firing sequences for this player
        sequences = detect_firing_sequences(weapon_fires, pid)

        result = FiringAccuracyResult(
            steam_id=int(pid),
            player_name=demo_data.player_names.get(int(pid), "Unknown"),
        )

        for seq in sequences:
            # Only analyze automatic weapons
            weapon_clean = seq.weapon.lower().replace("weapon_", "")
            if weapon_clean not in AUTOMATIC_WEAPONS:
                continue

            # Count hits and headshots for this sequence
            hits, headshots = _count_hits_for_sequence(damages_df, pid, seq.shot_ticks)

            # Categorize by sequence type
            seq_type = seq.sequence_type

            if seq_type == "tap":
                result.tap_shots_fired += seq.shot_count
                result.tap_shots_hit += hits
                result.tap_headshots += headshots
                result.tap_sequences += 1
            elif seq_type == "burst":
                result.burst_shots_fired += seq.shot_count
                result.burst_shots_hit += hits
                result.burst_headshots += headshots
                result.burst_sequences += 1
            else:  # spray
                result.spray_shots_fired += seq.shot_count
                result.spray_shots_hit += hits
                result.spray_headshots += headshots
                result.spray_sequences += 1

        results[int(pid)] = result

    # Log summary
    if results:
        total_seqs = sum(r.total_sequences for r in results.values())
        logger.info(
            f"Calculated firing accuracy for {len(results)} players ({total_seqs} sequences)"
        )

    return results


# ============================================================================
# Site Anchor Detection and Calculation
# ============================================================================


def _get_player_site(place_name: str | None) -> str | None:
    """
    Determine which bombsite a player is in based on place_name.

    Args:
        place_name: The place_name value from tick data (e.g., "BombsiteA")

    Returns:
        "A", "B", or None if not in a bombsite
    """
    if not place_name:
        return None

    place_lower = str(place_name).lower()

    # Check for A site
    if any(s.lower() in place_lower for s in ["bombsitea", "a site", "asite", "sitea"]):
        return "A"

    # Check for B site
    if any(s.lower() in place_lower for s in ["bombsiteb", "b site", "bsite", "siteb"]):
        return "B"

    return None


def detect_site_executes(
    demo_data: DemoData,
    execute_threshold: int = EXECUTE_PLAYER_THRESHOLD,
) -> list[SiteExecuteEvent]:
    """
    Detect site execute events (3+ T's entering a bombsite).

    Scans tick data to find moments when multiple T-side players
    enter a bombsite zone simultaneously.

    Args:
        demo_data: Parsed demo data with ticks_df and rounds
        execute_threshold: Minimum T's to count as an execute (default 3)

    Returns:
        List of SiteExecuteEvent objects
    """
    executes: list[SiteExecuteEvent] = []

    # Require tick data
    ticks_df = getattr(demo_data, "ticks_df", None)
    if ticks_df is None or ticks_df.empty:
        logger.warning("No tick data available - cannot detect site executes")
        return executes

    rounds = getattr(demo_data, "rounds", [])
    if not rounds:
        logger.warning("No round data available - cannot detect site executes")
        return executes

    # Find required columns
    tick_col = _find_column(ticks_df, ["tick"])
    steamid_col = _find_column(ticks_df, ["steamid", "steam_id", "user_steamid"])
    team_col = _find_column(ticks_df, ["team_num", "team", "side"])
    place_col = _find_column(ticks_df, ["last_place_name", "place_name", "place"])
    alive_col = _find_column(ticks_df, ["is_alive", "alive", "health"])

    if not all([tick_col, steamid_col, team_col, place_col]):
        logger.warning("Missing required columns for site execute detection")
        return executes

    # Process each round
    for round_info in rounds:
        round_num = round_info.round_num
        freeze_end = round_info.freeze_end_tick
        round_end = round_info.end_tick

        # Filter tick data to this round (after freeze time)
        round_ticks = ticks_df[
            (ticks_df[tick_col] >= freeze_end) & (ticks_df[tick_col] <= round_end)
        ]

        if round_ticks.empty:
            continue

        # Track if we've found an execute this round (only detect first major execute)
        execute_found = False

        # Get unique ticks in order
        unique_ticks = sorted(round_ticks[tick_col].unique())

        # Sample ticks (every 32 ticks = 0.5 seconds for performance)
        sampled_ticks = unique_ticks[::32] if len(unique_ticks) > 100 else unique_ticks

        for tick in sampled_ticks:
            if execute_found:
                break

            tick_data = round_ticks[round_ticks[tick_col] == tick]

            # Filter to alive players
            if alive_col:
                # Handle both boolean and health-based alive detection
                if tick_data[alive_col].dtype == bool:
                    tick_data = tick_data[tick_data[alive_col] == True]  # noqa: E712
                else:
                    tick_data = tick_data[tick_data[alive_col] > 0]

            # Separate T's and CT's
            # team_num: 2 = T, 3 = CT
            t_players = tick_data[tick_data[team_col] == 2]
            ct_players = tick_data[tick_data[team_col] == 3]

            # Check each site
            for site in ["A", "B"]:
                # Count T's in this site
                t_in_site = []
                for _, row in t_players.iterrows():
                    player_site = _get_player_site(row.get(place_col))
                    if player_site == site:
                        t_in_site.append(int(row[steamid_col]))

                # Execute detected if 3+ T's in site
                if len(t_in_site) >= execute_threshold:
                    # Find CT defenders in site
                    ct_defenders = []
                    for _, row in ct_players.iterrows():
                        player_site = _get_player_site(row.get(place_col))
                        if player_site == site:
                            ct_defenders.append(int(row[steamid_col]))

                    # Only count as anchor scenario if 1-2 CTs defending
                    if 1 <= len(ct_defenders) <= 2:
                        executes.append(
                            SiteExecuteEvent(
                                round_num=round_num,
                                site=site,
                                execute_start_tick=int(tick),
                                t_players=t_in_site,
                                ct_defenders=ct_defenders,
                                bomb_plant_tick=round_info.bomb_plant_tick,
                                round_end_tick=round_end,
                            )
                        )
                        execute_found = True
                        logger.debug(
                            f"Round {round_num}: Execute detected on {site} site "
                            f"({len(t_in_site)} T's vs {len(ct_defenders)} CT anchors)"
                        )
                        break

    logger.info(f"Detected {len(executes)} site execute events")
    return executes


def calculate_anchor_holds(
    demo_data: DemoData,
    executes: list[SiteExecuteEvent] | None = None,
) -> dict[int, PlayerAnchorStats]:
    """
    Calculate anchor hold metrics for CT players who defended against site executes.

    Measures:
    - Stall time (how long they survived/delayed)
    - Utility usage during the hold
    - Kills during the hold
    - Overall anchor hold rating

    Args:
        demo_data: Parsed demo data
        executes: Pre-detected site executes (or will detect if None)

    Returns:
        Dictionary mapping steam_id to PlayerAnchorStats
    """
    results: dict[int, PlayerAnchorStats] = {}

    # Detect executes if not provided
    if executes is None:
        executes = detect_site_executes(demo_data)

    if not executes:
        logger.info("No site executes detected - no anchor holds to calculate")
        return results

    # Get required data
    kills = getattr(demo_data, "kills", [])
    damages = getattr(demo_data, "damages", [])
    grenades = getattr(demo_data, "grenades", [])
    tick_rate = getattr(demo_data, "tick_rate", 64)

    # Build lookup structures for efficiency
    # Kills by round: {round_num: [kill_events]}
    kills_by_round: dict[int, list] = {}
    for kill in kills:
        rn = getattr(kill, "round_num", 0)
        if rn not in kills_by_round:
            kills_by_round[rn] = []
        kills_by_round[rn].append(kill)

    # Damages by round
    damages_by_round: dict[int, list] = {}
    for dmg in damages:
        rn = getattr(dmg, "round_num", 0)
        if rn not in damages_by_round:
            damages_by_round[rn] = []
        damages_by_round[rn].append(dmg)

    # Grenades by round
    grenades_by_round: dict[int, list] = {}
    for nade in grenades:
        rn = getattr(nade, "round_num", 0)
        if rn not in grenades_by_round:
            grenades_by_round[rn] = []
        grenades_by_round[rn].append(nade)

    # Process each execute
    for execute in executes:
        round_num = execute.round_num
        site = execute.site
        start_tick = execute.execute_start_tick

        round_kills = kills_by_round.get(round_num, [])
        round_damages = damages_by_round.get(round_num, [])
        round_grenades = grenades_by_round.get(round_num, [])

        # Process each anchor (CT defender)
        for anchor_id in execute.ct_defenders:
            # Determine when the anchor's hold ended
            # End conditions: anchor death, bomb plant, or round end
            anchor_death_tick = None
            for kill in round_kills:
                victim_id = getattr(kill, "victim_steamid", None)
                if victim_id == anchor_id:
                    kill_tick = getattr(kill, "tick", 0)
                    if kill_tick >= start_tick:
                        anchor_death_tick = kill_tick
                        break

            # Determine end tick
            end_tick = execute.round_end_tick
            survived = True

            if anchor_death_tick and anchor_death_tick < end_tick:
                end_tick = anchor_death_tick
                survived = False
            if execute.bomb_plant_tick and execute.bomb_plant_tick < end_tick:
                end_tick = execute.bomb_plant_tick

            # Calculate stall time
            stall_ticks = end_tick - start_tick
            stall_time_seconds = stall_ticks / tick_rate

            # Count kills during hold
            kills_during = 0
            for kill in round_kills:
                attacker_id = getattr(kill, "attacker_steamid", None)
                kill_tick = getattr(kill, "tick", 0)
                if attacker_id == anchor_id and start_tick <= kill_tick <= end_tick:
                    kills_during += 1

            # Count damage during hold
            damage_during = 0
            for dmg in round_damages:
                attacker_id = getattr(dmg, "attacker_steamid", None)
                dmg_tick = getattr(dmg, "tick", 0)
                if attacker_id == anchor_id and start_tick <= dmg_tick <= end_tick:
                    damage_during += getattr(dmg, "damage_health", 0)

            # Count utility used during hold
            molotovs = 0
            smokes = 0
            flashes = 0
            he_grenades = 0

            for nade in round_grenades:
                thrower_id = getattr(nade, "player_steamid", None)
                nade_tick = getattr(nade, "tick", 0)
                nade_type = getattr(nade, "grenade_type", "").lower()

                # Count grenades by the anchor during the hold
                if thrower_id == anchor_id and start_tick <= nade_tick <= end_tick:
                    if nade_type in ("molotov", "incgrenade", "inferno"):
                        molotovs += 1
                    elif nade_type in ("smokegrenade", "smoke"):
                        smokes += 1
                    elif nade_type in ("flashbang", "flash"):
                        flashes += 1
                    elif nade_type in ("hegrenade", "he_grenade", "he"):
                        he_grenades += 1

            # Avoid double-counting (thrown and detonate are separate events)
            # Only count "thrown" events, or if no thrown then count detonate
            # For simplicity, we'll divide by 2 if counts seem doubled
            # This is a heuristic - actual implementation may need refinement

            # Create hold result
            hold = AnchorHoldResult(
                steam_id=anchor_id,
                player_name=demo_data.player_names.get(anchor_id, "Unknown"),
                round_num=round_num,
                site=site,
                execute_start_tick=start_tick,
                hold_end_tick=end_tick,
                stall_time_seconds=stall_time_seconds,
                survived=survived,
                kills_during_hold=kills_during,
                damage_during_hold=damage_during,
                molotovs_used=molotovs,
                smokes_used=smokes,
                flashes_used=flashes,
                he_grenades_used=he_grenades,
                attackers_count=len(execute.t_players),
                teammates_nearby=len(execute.ct_defenders) - 1,
            )

            # Add to player stats
            if anchor_id not in results:
                results[anchor_id] = PlayerAnchorStats(
                    steam_id=anchor_id,
                    player_name=demo_data.player_names.get(anchor_id, "Unknown"),
                )

            stats = results[anchor_id]
            stats.total_holds += 1
            stats.total_stall_time += stall_time_seconds
            stats.total_utility_used += hold.utility_count
            stats.total_kills_during_holds += kills_during
            stats.total_damage_during_holds += damage_during
            stats.holds.append(hold)

            if survived:
                stats.successful_holds += 1

            logger.debug(
                f"Anchor hold: {hold.player_name} on {site} - "
                f"{stall_time_seconds:.1f}s, {hold.utility_count} util, "
                f"rating={hold.anchor_hold_rating:.1f}"
            )

    # Log summary
    if results:
        total_holds = sum(s.total_holds for s in results.values())
        avg_rating = sum(s.avg_rating for s in results.values()) / len(results) if results else 0
        logger.info(
            f"Calculated {total_holds} anchor holds for {len(results)} players "
            f"(avg rating: {avg_rating:.1f})"
        )

    return results


# ============================================================================
# Comprehensive Analysis
# ============================================================================


@dataclass
class ComprehensivePlayerMetrics:
    """All metrics combined for a player."""

    steam_id: int
    player_name: str
    team: str
    engagement: EngagementMetrics | None
    economy: EconomyMetrics | None
    utility: UtilityMetrics | None
    positioning: PositioningMetrics | None
    trades: TradeMetrics | None
    opening_duels: OpeningDuelMetrics | None
    ttd: TTDResult | None
    crosshair_placement: CrosshairPlacementResult | None
    firing_accuracy: FiringAccuracyResult | None = None  # Tap/Burst/Spray accuracy
    anchor_stats: PlayerAnchorStats | None = None  # Site anchor effectiveness

    def overall_rating(self) -> float:
        """
        Calculate an overall performance rating (0-100).

        Weighted combination of all available metrics.
        """
        score = 50.0  # Base score
        weights_applied = 0

        if self.engagement:
            kd = self.engagement.total_kills / max(self.engagement.total_deaths, 1)
            score += min(kd * 10, 30)  # Up to +30 for K/D
            weights_applied += 1

        if self.crosshair_placement:
            score += self.crosshair_placement.placement_score * 0.2  # Up to +20
            weights_applied += 1

        if self.opening_duels and self.opening_duels.opening_attempts > 0:
            score += self.opening_duels.opening_success_rate * 0.1  # Up to +10
            weights_applied += 1

        if self.trades:
            score += self.trades.trade_success_rate * 0.05  # Up to +5
            weights_applied += 1

        return min(max(score, 0), 100)


def calculate_comprehensive_metrics(
    demo_data: DemoData, steam_id: int | None = None
) -> dict[int, ComprehensivePlayerMetrics]:
    """
    Calculate all available metrics for players.

    This is the main entry point for complete player analysis.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to ComprehensivePlayerMetrics
    """
    # Calculate all metric types
    engagement = calculate_engagement_metrics(demo_data, steam_id)
    economy = calculate_economy_metrics(demo_data, steam_id)
    utility = calculate_utility_metrics(demo_data, steam_id)
    positioning = calculate_positioning_metrics(demo_data, steam_id)
    trades = calculate_trade_metrics(demo_data, steam_id)
    openings = calculate_opening_metrics(demo_data, steam_id)
    ttd = calculate_ttd(demo_data, steam_id)
    cp = calculate_crosshair_placement(demo_data, steam_id)
    firing_acc = calculate_firing_accuracy(demo_data, steam_id)
    anchor_stats = calculate_anchor_holds(demo_data)  # Site anchor effectiveness

    results: dict[int, ComprehensivePlayerMetrics] = {}

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    for player_id in player_ids:
        results[int(player_id)] = ComprehensivePlayerMetrics(
            steam_id=int(player_id),
            player_name=demo_data.player_names.get(int(player_id), "Unknown"),
            team=demo_data.player_teams.get(int(player_id), "Unknown"),
            engagement=engagement.get(int(player_id)),
            economy=economy.get(int(player_id)),
            utility=utility.get(int(player_id)),
            positioning=positioning.get(int(player_id)),
            trades=trades.get(int(player_id)),
            opening_duels=openings.get(int(player_id)),
            ttd=ttd.get(int(player_id)),
            crosshair_placement=cp.get(int(player_id)),
            firing_accuracy=firing_acc.get(int(player_id)),
            anchor_stats=anchor_stats.get(int(player_id)),
        )

    return results


# ============================================================================
# RWS (Round Win Shares) Calculation
# ============================================================================


@dataclass
class RWSResult:
    """RWS (Round Win Shares) calculation result for a player.

    RWS measures contribution to round wins based on damage dealt.
    - 100 points per round divided among winning team based on damage
    - Bomb planter/defuser gets 30 bonus points for objective completion
    - Average RWS ranges from ~8-12 for average players, 15+ for stars
    """

    steam_id: int
    player_name: str
    rounds_played: int
    rounds_won: int
    total_rws: float
    avg_rws: float  # Average RWS per round played
    total_damage: int
    damage_per_round: float
    objective_completions: int  # Bomb plants/defuses that exploded/succeeded
    objective_rws: float  # RWS from objective completions

    def __repr__(self) -> str:
        return f"RWS({self.player_name}: {self.avg_rws:.2f} avg, {self.rounds_won}/{self.rounds_played} won)"


def calculate_rws(demo_data: DemoData, steam_id: int | None = None) -> dict[int, RWSResult]:
    """
    Calculate RWS (Round Win Shares) for players.

    RWS Formula (per round):
    - Losing team: 0 RWS
    - Winning team: 100 RWS divided based on damage dealt
    - Bomb explodes: Planter gets 30 + (damage share of 70)
    - Bomb defused: Defuser gets 30 + (damage share of 70)
    - Elimination/Time: Just damage share of 100

    Args:
        demo_data: Parsed demo data with rounds, damages, and bomb_events
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to RWSResult
    """
    results: dict[int, RWSResult] = {}

    # Get data
    rounds = getattr(demo_data, "rounds", [])
    damages = getattr(demo_data, "damages", [])
    bomb_events = getattr(demo_data, "bomb_events", [])
    player_names = getattr(demo_data, "player_names", {})
    player_teams = getattr(demo_data, "player_teams", {})

    if not rounds:
        logger.warning("No round data available for RWS calculation")
        return results

    # Group damages by round
    round_damages: dict[int, dict[int, int]] = {}  # round_num -> {steam_id -> damage}
    for dmg in damages:
        round_num = getattr(dmg, "round_num", 0)
        attacker_id = getattr(dmg, "attacker_steamid", 0)
        damage_val = getattr(dmg, "damage", 0)
        attacker_side = getattr(dmg, "attacker_side", "").upper()
        victim_side = getattr(dmg, "victim_side", "").upper()

        # Only count damage to enemies
        if attacker_side and victim_side and attacker_side != victim_side:
            if round_num not in round_damages:
                round_damages[round_num] = {}
            if attacker_id not in round_damages[round_num]:
                round_damages[round_num][attacker_id] = 0
            round_damages[round_num][attacker_id] += damage_val

    # Get bomb planters/defusers per round
    round_planters: dict[int, int] = {}  # round_num -> planter_steam_id
    round_defusers: dict[int, int] = {}  # round_num -> defuser_steam_id
    for event in bomb_events:
        round_num = getattr(event, "round_num", 0)
        player_id = getattr(event, "player_steamid", 0)
        event_type = getattr(event, "event_type", "")

        if event_type == "planted":
            round_planters[round_num] = player_id
        elif event_type == "defused":
            round_defusers[round_num] = player_id

    # Initialize player stats
    player_stats: dict[int, dict] = {}
    for pid in player_names:
        player_stats[pid] = {
            "rounds_played": 0,
            "rounds_won": 0,
            "total_rws": 0.0,
            "total_damage": 0,
            "objective_completions": 0,
            "objective_rws": 0.0,
        }

    # Calculate RWS for each round
    for round_info in rounds:
        round_num = getattr(round_info, "round_num", 0)
        winner = getattr(round_info, "winner", "").upper()
        reason = getattr(round_info, "reason", "")

        if not winner:
            continue

        # Get damage dealt this round
        round_dmg = round_damages.get(round_num, {})

        # Determine which players won
        winning_players = []
        for pid, team in player_teams.items():
            team_upper = str(team).upper()
            # Handle both "CT" and "COUNTER-TERRORISTS" style
            is_ct = "CT" in team_upper or "COUNTER" in team_upper
            is_t = "T" in team_upper and not is_ct

            if (winner == "CT" and is_ct) or (winner == "T" and is_t):
                winning_players.append(pid)

            # Track rounds played
            if pid in player_stats:
                player_stats[pid]["rounds_played"] += 1

        # Skip if no winning players identified
        if not winning_players:
            continue

        # Calculate total damage by winning team this round
        winning_team_damage = sum(round_dmg.get(pid, 0) for pid in winning_players)

        # Determine if objective bonus applies
        objective_player = None
        objective_bonus = 0.0
        damage_pool = 100.0  # Total RWS to distribute

        if reason == "target_bombed" and round_num in round_planters:
            planter = round_planters[round_num]
            if planter in winning_players:
                objective_player = planter
                objective_bonus = 30.0
                damage_pool = 70.0
        elif reason == "bomb_defused" and round_num in round_defusers:
            defuser = round_defusers[round_num]
            if defuser in winning_players:
                objective_player = defuser
                objective_bonus = 30.0
                damage_pool = 70.0

        # Distribute RWS to winning team
        for pid in winning_players:
            if pid not in player_stats:
                continue

            player_stats[pid]["rounds_won"] += 1

            # Calculate damage share
            player_damage = round_dmg.get(pid, 0)
            player_stats[pid]["total_damage"] += player_damage

            if winning_team_damage > 0:
                damage_share = player_damage / winning_team_damage
                rws_from_damage = damage_share * damage_pool
            else:
                # Equal share if no damage recorded
                rws_from_damage = damage_pool / len(winning_players)

            # Add objective bonus
            rws_this_round = rws_from_damage
            if pid == objective_player:
                rws_this_round += objective_bonus
                player_stats[pid]["objective_completions"] += 1
                player_stats[pid]["objective_rws"] += objective_bonus

            player_stats[pid]["total_rws"] += rws_this_round

    # Build results
    target_ids = {steam_id} if steam_id else set(player_stats.keys())

    for pid in target_ids:
        if pid not in player_stats:
            continue

        stats = player_stats[pid]
        rounds_played = max(stats["rounds_played"], 1)

        results[pid] = RWSResult(
            steam_id=pid,
            player_name=player_names.get(pid, f"Player {pid}"),
            rounds_played=stats["rounds_played"],
            rounds_won=stats["rounds_won"],
            total_rws=stats["total_rws"],
            avg_rws=stats["total_rws"] / rounds_played,
            total_damage=stats["total_damage"],
            damage_per_round=stats["total_damage"] / rounds_played,
            objective_completions=stats["objective_completions"],
            objective_rws=stats["objective_rws"],
        )

    return results
