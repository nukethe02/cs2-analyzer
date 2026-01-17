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

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import logging

import numpy as np
import pandas as pd

from opensight.parser import DemoData

logger = logging.getLogger(__name__)


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
    "usp_silencer": 200, "p2000": 200, "glock": 200, "p250": 300,
    "tec9": 500, "fiveseven": 500, "cz75a": 500, "deagle": 700,
    "revolver": 600, "dualies": 400,
    # SMGs
    "mac10": 1050, "mp9": 1250, "mp7": 1500, "ump45": 1200,
    "p90": 2350, "bizon": 1400, "mp5sd": 1500,
    # Rifles
    "ak47": 2700, "m4a1": 2900, "m4a1_silencer": 2900, "famas": 2050,
    "galilar": 1800, "sg556": 3000, "aug": 3300,
    # Snipers
    "awp": 4750, "ssg08": 1700, "scar20": 5000, "g3sg1": 5000,
    # Shotguns
    "nova": 1050, "xm1014": 2000, "mag7": 1300, "sawedoff": 1100,
    # Machine guns
    "m249": 5200, "negev": 1700,
    # Grenades
    "hegrenade": 300, "flashbang": 200, "smokegrenade": 300,
    "molotov": 400, "incgrenade": 600, "decoy": 50,
}

# Weapon categories for classification
WEAPON_CATEGORIES: dict[str, WeaponCategory] = {
    "usp_silencer": WeaponCategory.PISTOL, "p2000": WeaponCategory.PISTOL,
    "glock": WeaponCategory.PISTOL, "p250": WeaponCategory.PISTOL,
    "tec9": WeaponCategory.PISTOL, "fiveseven": WeaponCategory.PISTOL,
    "cz75a": WeaponCategory.PISTOL, "deagle": WeaponCategory.PISTOL,
    "revolver": WeaponCategory.PISTOL, "dualies": WeaponCategory.PISTOL,
    "mac10": WeaponCategory.SMG, "mp9": WeaponCategory.SMG,
    "mp7": WeaponCategory.SMG, "ump45": WeaponCategory.SMG,
    "p90": WeaponCategory.SMG, "bizon": WeaponCategory.SMG,
    "mp5sd": WeaponCategory.SMG,
    "ak47": WeaponCategory.RIFLE, "m4a1": WeaponCategory.RIFLE,
    "m4a1_silencer": WeaponCategory.RIFLE, "famas": WeaponCategory.RIFLE,
    "galilar": WeaponCategory.RIFLE, "sg556": WeaponCategory.RIFLE,
    "aug": WeaponCategory.RIFLE,
    "awp": WeaponCategory.SNIPER, "ssg08": WeaponCategory.SNIPER,
    "scar20": WeaponCategory.SNIPER, "g3sg1": WeaponCategory.SNIPER,
    "nova": WeaponCategory.SHOTGUN, "xm1014": WeaponCategory.SHOTGUN,
    "mag7": WeaponCategory.SHOTGUN, "sawedoff": WeaponCategory.SHOTGUN,
    "m249": WeaponCategory.MACHINE_GUN, "negev": WeaponCategory.MACHINE_GUN,
    "knife": WeaponCategory.KNIFE,
}

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


@dataclass
class EngagementMetrics:
    """Combined metrics for a player across all engagements."""

    steam_id: int
    player_name: str
    ttd: Optional[TTDResult]
    crosshair_placement: Optional[CrosshairPlacementResult]
    total_kills: int
    total_deaths: int
    headshot_percentage: float
    damage_per_round: float


def calculate_angle_between_vectors(
    view_dir: np.ndarray,
    target_dir: np.ndarray
) -> float:
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
    max_lookback_ms: float = 2000.0
) -> Optional[int]:
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
        (player_pos["tick"] >= start_tick) &
        (player_pos["tick"] <= damage_tick)
    ].sort_values("tick")

    target_window = target_pos[
        (target_pos["tick"] >= start_tick) &
        (target_pos["tick"] <= damage_tick)
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


def calculate_ttd(
    demo_data: DemoData,
    steam_id: Optional[int] = None
) -> dict[int, TTDResult]:
    """
    Calculate Time to Damage for players.

    TTD measures the latency between first spotting an enemy and
    dealing damage to them. Lower values indicate faster reactions.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze

    Returns:
        Dictionary mapping steam_id to TTDResult
    """
    results: dict[int, TTDResult] = {}

    # Get all damage events
    damage_df = demo_data.damage_events
    if damage_df.empty:
        logger.warning("No damage events found in demo")
        return results

    # Filter to specific player if requested
    if steam_id is not None:
        damage_df = damage_df[damage_df["attacker_id"] == steam_id]

    # Group by attacker
    for attacker_id, attacker_damage in damage_df.groupby("attacker_id"):
        if attacker_id == 0:
            continue  # Skip world damage

        ttd_values: list[float] = []

        # Analyze each damage event
        for _, event in attacker_damage.iterrows():
            damage_tick = event["tick"]
            victim_id = event["victim_id"]

            # Get position data for attacker and victim
            attacker_pos = demo_data.player_positions[
                demo_data.player_positions["steam_id"] == attacker_id
            ]
            victim_pos = demo_data.player_positions[
                demo_data.player_positions["steam_id"] == victim_id
            ]

            # Find when attacker first saw victim
            visibility_tick = _find_visibility_start(
                attacker_pos,
                victim_pos,
                damage_tick,
                demo_data.tick_rate
            )

            if visibility_tick is not None and visibility_tick < damage_tick:
                ttd_ticks = damage_tick - visibility_tick
                ttd_ms = (ttd_ticks / demo_data.tick_rate) * 1000
                ttd_values.append(ttd_ms)

        if ttd_values:
            results[int(attacker_id)] = TTDResult(
                steam_id=int(attacker_id),
                player_name=demo_data.player_names.get(int(attacker_id), "Unknown"),
                engagement_count=len(ttd_values),
                mean_ttd_ms=float(np.mean(ttd_values)),
                median_ttd_ms=float(np.median(ttd_values)),
                min_ttd_ms=float(np.min(ttd_values)),
                max_ttd_ms=float(np.max(ttd_values)),
                std_ttd_ms=float(np.std(ttd_values)),
                ttd_values=ttd_values,
            )

    return results


def calculate_crosshair_placement(
    demo_data: DemoData,
    steam_id: Optional[int] = None,
    sample_interval_ticks: int = 16
) -> dict[int, CrosshairPlacementResult]:
    """
    Calculate Crosshair Placement quality for players.

    CP measures the angular distance between a player's aim and
    enemy positions. Lower angles indicate better placement.

    Args:
        demo_data: Parsed demo data
        steam_id: Optional specific player to analyze
        sample_interval_ticks: How often to sample (default every 16 ticks)

    Returns:
        Dictionary mapping steam_id to CrosshairPlacementResult
    """
    results: dict[int, CrosshairPlacementResult] = {}

    positions = demo_data.player_positions
    if positions.empty:
        logger.warning("No position data found in demo")
        return results

    if "pitch" not in positions.columns or "yaw" not in positions.columns:
        logger.warning("View angle data not available")
        return results

    # Get unique players
    player_ids = positions["steam_id"].unique()
    if steam_id is not None:
        player_ids = [steam_id] if steam_id in player_ids else []

    for player_id in player_ids:
        if player_id == 0:
            continue

        player_team = demo_data.teams.get(int(player_id), "Unknown")
        angle_values: list[float] = []

        # Sample positions at intervals
        player_pos = positions[positions["steam_id"] == player_id]
        sample_ticks = player_pos["tick"].unique()[::sample_interval_ticks]

        for tick in sample_ticks:
            player_at_tick = player_pos[player_pos["tick"] == tick]
            if player_at_tick.empty:
                continue

            player_row = player_at_tick.iloc[0]
            player_xyz = np.array([
                player_row["x"],
                player_row["y"],
                player_row["z"] + 64  # Eye height approximation
            ])

            # Get view direction
            view_dir = angles_to_direction(
                player_row["pitch"],
                player_row["yaw"]
            )

            # Find enemies at this tick
            enemies_at_tick = positions[
                (positions["tick"] == tick) &
                (positions["steam_id"] != player_id)
            ]

            # Filter to enemy team only
            enemies_at_tick = enemies_at_tick[
                enemies_at_tick["steam_id"].apply(
                    lambda x: demo_data.teams.get(int(x), "") != player_team
                )
            ]

            if enemies_at_tick.empty:
                continue

            # Calculate angle to closest enemy
            min_angle = float("inf")
            for _, enemy_row in enemies_at_tick.iterrows():
                enemy_xyz = np.array([
                    enemy_row["x"],
                    enemy_row["y"],
                    enemy_row["z"] + 64  # Eye height
                ])

                direction = enemy_xyz - player_xyz
                distance = np.linalg.norm(direction)

                if distance < 100 or distance > 2000:
                    continue

                target_dir = direction / distance
                angle = calculate_angle_between_vectors(view_dir, target_dir)
                min_angle = min(min_angle, angle)

            if min_angle < float("inf"):
                angle_values.append(min_angle)

        if angle_values:
            # Calculate placement score (inverse of angle, normalized to 0-100)
            mean_angle = np.mean(angle_values)
            # Score formula: 100 when angle is 0, drops as angle increases
            # Using exponential decay with 45 degrees as the midpoint
            placement_score = 100 * np.exp(-mean_angle / 45)

            results[int(player_id)] = CrosshairPlacementResult(
                steam_id=int(player_id),
                player_name=demo_data.player_names.get(int(player_id), "Unknown"),
                sample_count=len(angle_values),
                mean_angle_deg=float(mean_angle),
                median_angle_deg=float(np.median(angle_values)),
                percentile_90_deg=float(np.percentile(angle_values, 90)),
                placement_score=float(placement_score),
                angle_values=angle_values,
            )

    return results


def calculate_engagement_metrics(
    demo_data: DemoData,
    steam_id: Optional[int] = None
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
    kills_df = demo_data.kill_events
    player_ids = set(demo_data.player_names.keys())

    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    results: dict[int, EngagementMetrics] = {}
    num_rounds = max(len(demo_data.round_starts), 1)

    for player_id in player_ids:
        # Count kills and headshots
        player_kills = kills_df[kills_df["attacker_id"] == player_id]
        total_kills = len(player_kills)
        headshots = player_kills["headshot"].sum() if not player_kills.empty else 0
        hs_percentage = (headshots / total_kills * 100) if total_kills > 0 else 0

        # Count deaths
        total_deaths = len(kills_df[kills_df["victim_id"] == player_id])

        # Calculate damage per round
        damage_df = demo_data.damage_events
        player_damage = damage_df[damage_df["attacker_id"] == player_id]
        total_damage = player_damage["damage"].sum() if not player_damage.empty else 0
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
    demo_data: DemoData,
    steam_id: Optional[int] = None
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

    kills_df = demo_data.kill_events
    damage_df = demo_data.damage_events
    shots_df = demo_data.shots_fired

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    for player_id in player_ids:
        # Track weapon usage and kills
        player_kills = kills_df[kills_df["attacker_id"] == player_id]
        weapon_kills: dict[str, int] = {}

        for _, kill in player_kills.iterrows():
            weapon = kill.get("weapon", "unknown")
            weapon_clean = weapon.lower().replace("weapon_", "")
            weapon_kills[weapon_clean] = weapon_kills.get(weapon_clean, 0) + 1

        # Calculate total value of enemies killed
        total_value_killed = 0
        for _, kill in player_kills.iterrows():
            victim_id = kill.get("victim_id", 0)
            # Estimate victim loadout value (simplified)
            total_value_killed += 2000  # Average loadout estimate

        # Calculate money spent (based on weapons used)
        money_spent = 0
        weapons_used = set()
        if not shots_df.empty and "weapon" in shots_df.columns:
            player_shots = shots_df[shots_df["steam_id"] == player_id]
            for weapon in player_shots["weapon"].unique():
                weapons_used.add(weapon.lower().replace("weapon_", ""))

        for weapon in weapons_used:
            money_spent += get_weapon_price(weapon)

        # Calculate damage efficiency
        player_damage = damage_df[damage_df["attacker_id"] == player_id]
        total_damage = player_damage["damage"].sum() if not player_damage.empty else 0
        weapon_efficiency = total_damage / max(money_spent, 1)

        # Find favorite weapon
        favorite_weapon = max(weapon_kills.keys(), key=lambda w: weapon_kills[w]) if weapon_kills else "unknown"

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
            avg_loadout_value=float(money_spent / max(len(demo_data.round_starts), 1)),
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
    demo_data: DemoData,
    steam_id: Optional[int] = None
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

    damage_df = demo_data.damage_events
    kills_df = demo_data.kill_events
    shots_df = demo_data.shots_fired

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    for player_id in player_ids:
        # Count grenades thrown (from shots/weapon_fire events)
        grenades_thrown: dict[str, int] = {
            "hegrenade": 0, "flashbang": 0, "smokegrenade": 0,
            "molotov": 0, "incgrenade": 0, "decoy": 0
        }

        if not shots_df.empty and "weapon" in shots_df.columns:
            player_shots = shots_df[shots_df["steam_id"] == player_id]
            for _, shot in player_shots.iterrows():
                weapon = shot["weapon"].lower().replace("weapon_", "")
                if weapon in grenades_thrown:
                    grenades_thrown[weapon] += 1

        # Calculate utility damage
        utility_damage = 0.0
        molotov_damage = 0.0

        if not damage_df.empty and "weapon" in damage_df.columns:
            player_damage = damage_df[damage_df["attacker_id"] == player_id]
            for _, dmg in player_damage.iterrows():
                weapon = dmg.get("weapon", "").lower().replace("weapon_", "")
                damage = dmg.get("damage", 0)
                if weapon == "hegrenade":
                    utility_damage += damage
                elif weapon in ("molotov", "incgrenade", "inferno"):
                    utility_damage += damage
                    molotov_damage += damage

        total_grenades = sum(grenades_thrown.values())

        # Calculate smoke kills (kills where weapon is knife or pistol shortly after smoke)
        smoke_kills = 0
        player_kills = kills_df[kills_df["attacker_id"] == player_id]
        # Simplified: assume some kills through smoke based on weapon type
        for _, kill in player_kills.iterrows():
            weapon = kill.get("weapon", "").lower()
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


def get_area_at_position(
    x: float,
    y: float,
    map_name: str
) -> Optional[str]:
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
    demo_data: DemoData,
    steam_id: Optional[int] = None
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

    positions = demo_data.player_positions
    kills_df = demo_data.kill_events

    if positions.empty:
        return results

    player_ids = positions["steam_id"].unique()
    if steam_id is not None:
        player_ids = [steam_id] if steam_id in player_ids else []

    map_name = demo_data.map_name

    for player_id in player_ids:
        if player_id == 0:
            continue

        player_pos = positions[positions["steam_id"] == player_id]
        player_team = demo_data.teams.get(int(player_id), "Unknown")

        # Track area coverage
        area_times: dict[str, int] = {}
        prev_area = None
        rotation_count = 0
        total_ticks = len(player_pos)

        for _, row in player_pos.iterrows():
            area = get_area_at_position(row["x"], row["y"], map_name)
            if area:
                area_times[area] = area_times.get(area, 0) + 1
                if prev_area and area != prev_area:
                    rotation_count += 1
                prev_area = area

        # Convert to percentages
        area_coverage = {
            area: (count / total_ticks) * 100
            for area, count in area_times.items()
        } if total_ticks > 0 else {}

        # Calculate time in sites vs mid
        site_time = sum(
            pct for area, pct in area_coverage.items()
            if "site" in area.lower()
        )
        mid_time = area_coverage.get("mid", 0.0)

        # Calculate average distance from teammates
        avg_teammate_distance = 0.0
        teammate_distances: list[float] = []

        for tick in player_pos["tick"].unique()[:100]:  # Sample for efficiency
            tick_positions = positions[positions["tick"] == tick]
            player_at_tick = tick_positions[tick_positions["steam_id"] == player_id]
            teammates = tick_positions[
                (tick_positions["steam_id"] != player_id) &
                (tick_positions["steam_id"].apply(
                    lambda x: demo_data.teams.get(int(x), "") == player_team
                ))
            ]

            if not player_at_tick.empty and not teammates.empty:
                player_xyz = np.array([
                    player_at_tick.iloc[0]["x"],
                    player_at_tick.iloc[0]["y"],
                    player_at_tick.iloc[0]["z"]
                ])
                for _, mate in teammates.iterrows():
                    mate_xyz = np.array([mate["x"], mate["y"], mate["z"]])
                    teammate_distances.append(np.linalg.norm(player_xyz - mate_xyz))

        if teammate_distances:
            avg_teammate_distance = np.mean(teammate_distances)

        # Count deaths from behind
        deaths_from_behind = 0
        player_deaths = kills_df[kills_df["victim_id"] == player_id]

        for _, death in player_deaths.iterrows():
            death_tick = death["tick"]
            attacker_id = death["attacker_id"]

            # Get positions at death tick
            victim_pos = player_pos[player_pos["tick"] == death_tick]
            attacker_pos = positions[
                (positions["tick"] == death_tick) &
                (positions["steam_id"] == attacker_id)
            ]

            if not victim_pos.empty and not attacker_pos.empty:
                victim_row = victim_pos.iloc[0]
                attacker_row = attacker_pos.iloc[0]

                if "yaw" in victim_row:
                    # Calculate angle to attacker
                    victim_xyz = np.array([victim_row["x"], victim_row["y"]])
                    attacker_xyz = np.array([attacker_row["x"], attacker_row["y"]])
                    direction = attacker_xyz - victim_xyz
                    attacker_angle = np.degrees(np.arctan2(direction[1], direction[0]))

                    view_angle = victim_row["yaw"]
                    angle_diff = abs(attacker_angle - view_angle)
                    angle_diff = min(angle_diff, 360 - angle_diff)

                    if angle_diff > 90:  # Attacker behind
                        deaths_from_behind += 1

        # Calculate first contact rate
        first_contacts = 0
        for round_start in demo_data.round_starts[:10]:  # Sample rounds
            round_end = round_start + (15 * demo_data.tick_rate)  # First 15 seconds
            round_kills = kills_df[
                (kills_df["tick"] >= round_start) &
                (kills_df["tick"] <= round_end)
            ]
            if not round_kills.empty:
                first_kill = round_kills.iloc[0]
                if first_kill["attacker_id"] == player_id or first_kill["victim_id"] == player_id:
                    first_contacts += 1

        first_contact_rate = (first_contacts / max(len(demo_data.round_starts[:10]), 1)) * 100

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
    demo_data: DemoData,
    steam_id: Optional[int] = None,
    trade_window_ms: float = 2000.0
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

    kills_df = demo_data.kill_events
    if kills_df.empty:
        return results

    kills_sorted = kills_df.sort_values("tick")
    trade_window_ticks = int((trade_window_ms / 1000.0) * demo_data.tick_rate)

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    for player_id in player_ids:
        player_team = demo_data.teams.get(int(player_id), "Unknown")
        trades_completed = 0
        deaths_traded = 0
        trade_times: list[float] = []

        # Check each kill by this player
        player_kills = kills_sorted[kills_sorted["attacker_id"] == player_id]

        for _, kill in player_kills.iterrows():
            kill_tick = kill["tick"]
            victim_id = kill["victim_id"]

            # Look for recent teammate deaths by this victim
            recent_deaths = kills_sorted[
                (kills_sorted["attacker_id"] == victim_id) &
                (kills_sorted["tick"] >= kill_tick - trade_window_ticks) &
                (kills_sorted["tick"] < kill_tick)
            ]

            for _, death in recent_deaths.iterrows():
                dead_teammate = death["victim_id"]
                if demo_data.teams.get(int(dead_teammate), "") == player_team:
                    trades_completed += 1
                    trade_time = (kill_tick - death["tick"]) / demo_data.tick_rate * 1000
                    trade_times.append(trade_time)
                    break

        # Check if player's deaths were traded
        player_deaths = kills_sorted[kills_sorted["victim_id"] == player_id]

        for _, death in player_deaths.iterrows():
            death_tick = death["tick"]
            killer_id = death["attacker_id"]

            # Look for teammate killing the killer shortly after
            trades = kills_sorted[
                (kills_sorted["victim_id"] == killer_id) &
                (kills_sorted["tick"] > death_tick) &
                (kills_sorted["tick"] <= death_tick + trade_window_ticks)
            ]

            for _, trade in trades.iterrows():
                trader_id = trade["attacker_id"]
                if demo_data.teams.get(int(trader_id), "") == player_team:
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
    demo_data: DemoData,
    steam_id: Optional[int] = None,
    opening_window_seconds: float = 30.0
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

    kills_df = demo_data.kill_events
    if kills_df.empty or not demo_data.round_starts:
        return results

    opening_window_ticks = int(opening_window_seconds * demo_data.tick_rate)

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    player_stats: dict[int, dict] = {
        pid: {
            "kills": 0, "deaths": 0, "attempts": 0,
            "times": [], "weapons": []
        }
        for pid in player_ids
    }

    # Analyze each round
    for i, round_start in enumerate(demo_data.round_starts):
        round_end = demo_data.round_ends[i] if i < len(demo_data.round_ends) else round_start + opening_window_ticks

        # Get first kill of the round
        round_kills = kills_df[
            (kills_df["tick"] >= round_start) &
            (kills_df["tick"] <= min(round_start + opening_window_ticks, round_end))
        ].sort_values("tick")

        if round_kills.empty:
            continue

        first_kill = round_kills.iloc[0]
        attacker = first_kill["attacker_id"]
        victim = first_kill["victim_id"]
        weapon = first_kill.get("weapon", "unknown")
        kill_time = (first_kill["tick"] - round_start) / demo_data.tick_rate * 1000

        if attacker in player_stats:
            player_stats[attacker]["kills"] += 1
            player_stats[attacker]["attempts"] += 1
            player_stats[attacker]["times"].append(kill_time)
            player_stats[attacker]["weapons"].append(weapon)

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
            w_clean = w.lower().replace("weapon_", "")
            weapon_counts[w_clean] = weapon_counts.get(w_clean, 0) + 1
        opening_weapon = max(weapon_counts.keys(), key=lambda w: weapon_counts[w]) if weapon_counts else "unknown"

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
# Comprehensive Analysis
# ============================================================================

@dataclass
class ComprehensivePlayerMetrics:
    """All metrics combined for a player."""

    steam_id: int
    player_name: str
    team: str
    engagement: Optional[EngagementMetrics]
    economy: Optional[EconomyMetrics]
    utility: Optional[UtilityMetrics]
    positioning: Optional[PositioningMetrics]
    trades: Optional[TradeMetrics]
    opening_duels: Optional[OpeningDuelMetrics]
    ttd: Optional[TTDResult]
    crosshair_placement: Optional[CrosshairPlacementResult]

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
    demo_data: DemoData,
    steam_id: Optional[int] = None
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

    results: dict[int, ComprehensivePlayerMetrics] = {}

    player_ids = set(demo_data.player_names.keys())
    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    for player_id in player_ids:
        results[int(player_id)] = ComprehensivePlayerMetrics(
            steam_id=int(player_id),
            player_name=demo_data.player_names.get(int(player_id), "Unknown"),
            team=demo_data.teams.get(int(player_id), "Unknown"),
            engagement=engagement.get(int(player_id)),
            economy=economy.get(int(player_id)),
            utility=utility.get(int(player_id)),
            positioning=positioning.get(int(player_id)),
            trades=trades.get(int(player_id)),
            opening_duels=openings.get(int(player_id)),
            ttd=ttd.get(int(player_id)),
            crosshair_placement=cp.get(int(player_id)),
        )

    return results
