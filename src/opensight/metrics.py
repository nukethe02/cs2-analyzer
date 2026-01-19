"""
Professional-Grade Metrics for CS2 Analytics

Implements key performance metrics:
- Time to Damage (TTD): Latency between spotting an enemy and dealing damage
- Crosshair Placement (CP): Angular distance between aim and target position

These metrics are derived from tick-level demo data and provide insights
comparable to professional analytics platforms.
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd

from opensight.parser import DemoData

logger = logging.getLogger(__name__)


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
    damage_df = demo_data.damages_df
    if damage_df is None or damage_df.empty:
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

        # Check if tick data is available for position tracking
        ticks_df = demo_data.ticks_df
        if ticks_df is None or ticks_df.empty:
            logger.debug("No tick data available for detailed TTD analysis")
            continue

        # Find the steamid column in ticks_df
        steamid_col = None
        for col in ["steamid", "steam_id", "player_steamid"]:
            if col in ticks_df.columns:
                steamid_col = col
                break
        if steamid_col is None:
            logger.debug("No steamid column found in ticks_df")
            continue

        # Analyze each damage event
        for _, event in attacker_damage.iterrows():
            damage_tick = event.get("tick", 0)
            victim_id = event.get("victim_id", event.get("user_steamid", 0))

            # Get position data for attacker and victim from ticks_df
            attacker_pos = ticks_df[ticks_df[steamid_col] == attacker_id]
            victim_pos = ticks_df[ticks_df[steamid_col] == victim_id]

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

    # Use ticks_df for position data
    positions = demo_data.ticks_df
    if positions is None or positions.empty:
        logger.warning("No position data found in demo (ticks_df required)")
        return results

    if "pitch" not in positions.columns or "yaw" not in positions.columns:
        logger.warning("View angle data not available")
        return results

    # Find the steamid column
    steamid_col = None
    for col in ["steamid", "steam_id", "player_steamid"]:
        if col in positions.columns:
            steamid_col = col
            break
    if steamid_col is None:
        logger.warning("No steamid column found in position data")
        return results

    # Find position columns (may be uppercase or lowercase)
    x_col = "X" if "X" in positions.columns else "x"
    y_col = "Y" if "Y" in positions.columns else "y"
    z_col = "Z" if "Z" in positions.columns else "z"

    # Get unique players
    player_ids = positions[steamid_col].unique()
    if steam_id is not None:
        player_ids = [steam_id] if steam_id in player_ids else []

    for player_id in player_ids:
        if player_id == 0:
            continue

        player_team = demo_data.player_teams.get(int(player_id), 0)
        angle_values: list[float] = []

        # Sample positions at intervals
        player_pos = positions[positions[steamid_col] == player_id]
        sample_ticks = player_pos["tick"].unique()[::sample_interval_ticks]

        for tick in sample_ticks:
            player_at_tick = player_pos[player_pos["tick"] == tick]
            if player_at_tick.empty:
                continue

            player_row = player_at_tick.iloc[0]
            player_xyz = np.array([
                player_row[x_col],
                player_row[y_col],
                player_row[z_col] + 64  # Eye height approximation
            ])

            # Get view direction
            view_dir = angles_to_direction(
                player_row["pitch"],
                player_row["yaw"]
            )

            # Find enemies at this tick
            enemies_at_tick = positions[
                (positions["tick"] == tick) &
                (positions[steamid_col] != player_id)
            ]

            # Filter to enemy team only
            enemies_at_tick = enemies_at_tick[
                enemies_at_tick[steamid_col].apply(
                    lambda x: demo_data.player_teams.get(int(x), 0) != player_team
                )
            ]

            if enemies_at_tick.empty:
                continue

            # Calculate angle to closest enemy
            min_angle = float("inf")
            for _, enemy_row in enemies_at_tick.iterrows():
                enemy_xyz = np.array([
                    enemy_row[x_col],
                    enemy_row[y_col],
                    enemy_row[z_col] + 64  # Eye height
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
    kills_df = demo_data.kills_df
    player_ids = set(demo_data.player_names.keys())

    if steam_id is not None:
        player_ids = {steam_id} if steam_id in player_ids else set()

    results: dict[int, EngagementMetrics] = {}
    num_rounds = max(demo_data.num_rounds, 1)

    # Helper to find column names
    def find_col(df: pd.DataFrame, options: list[str]) -> Optional[str]:
        for col in options:
            if col in df.columns:
                return col
        return None

    # Find kill column names
    att_col = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"]) if not kills_df.empty else None
    vic_col = find_col(kills_df, ["user_steamid", "victim_steamid"]) if not kills_df.empty else None
    hs_col = find_col(kills_df, ["headshot"]) if not kills_df.empty else None

    # Find damage column names
    damage_df = demo_data.damages_df
    dmg_att_col = find_col(damage_df, ["attacker_steamid", "attacker_steam_id"]) if damage_df is not None and not damage_df.empty else None
    dmg_col = find_col(damage_df, ["dmg_health", "damage", "dmg"]) if damage_df is not None and not damage_df.empty else None

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
