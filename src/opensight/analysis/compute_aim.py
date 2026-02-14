"""
Aim computation methods extracted from analytics.py.

This module contains:
- TTD (Time to Damage / Engagement Duration) computation
- True TTD (Reaction time from visibility to first damage)
- Crosshair Placement (CP) computation
- Accuracy statistics calculation

All functions accept a DemoAnalyzer instance as the first parameter
to access analyzer state and data.
"""

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from opensight.core.parser import DemoData, safe_float, safe_int

if TYPE_CHECKING:
    from opensight.analysis.analytics import DemoAnalyzer

logger = logging.getLogger(__name__)

# Import optimized metrics if available
try:
    from opensight.analysis.metrics_optimized import MetricType

    HAS_OPTIMIZED_METRICS = True
except ImportError:
    HAS_OPTIMIZED_METRICS = False
    MetricType = None


# ============================================================================
# Helper Functions
# ============================================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.

    Args:
        value: Any value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int.

    Args:
        value: Any value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def _compute_view_direction(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Convert pitch and yaw angles (in degrees) to a normalized 3D direction vector.

    Uses Source Engine conventions:
    - Yaw: rotation around Z axis (0 = +X, 90 = +Y)
    - Pitch: rotation around horizontal (positive = looking down in Source)

    Args:
        pitch_deg: Pitch angle in degrees
        yaw_deg: Yaw angle in degrees

    Returns:
        Normalized 3D direction vector as numpy array [x, y, z]
    """
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)

    # View direction from Euler angles (Source Engine convention)
    x = np.cos(yaw_rad) * np.cos(pitch_rad)
    y = np.sin(yaw_rad) * np.cos(pitch_rad)
    z = -np.sin(pitch_rad)  # Negative because positive pitch = looking down

    return np.array([x, y, z])


def _compute_angular_error(
    attacker_pos: np.ndarray,
    attacker_pitch: float,
    attacker_yaw: float,
    victim_pos: np.ndarray,
) -> float | None:
    """
    Compute the angular error between where the attacker was looking
    and where they needed to look to hit the victim.

    Args:
        attacker_pos: Attacker position [x, y, z]
        attacker_pitch: Attacker pitch angle in degrees
        attacker_yaw: Attacker yaw angle in degrees
        victim_pos: Victim position [x, y, z]

    Returns:
        Angular error in degrees, or None if computation fails
    """
    try:
        # Direction the attacker was actually looking
        view_vec = _compute_view_direction(attacker_pitch, attacker_yaw)

        # Ideal direction from attacker to victim
        ideal_vec = victim_pos - attacker_pos
        distance = np.linalg.norm(ideal_vec)

        if distance < 0.001:
            return 0.0  # Extremely close, no error

        ideal_vec = ideal_vec / distance  # Normalize

        # Compute angle between vectors using dot product
        dot = np.clip(np.dot(view_vec, ideal_vec), -1.0, 1.0)
        angular_error_rad = np.arccos(dot)
        angular_error_deg = np.degrees(angular_error_rad)

        return float(angular_error_deg)

    except Exception as e:
        logger.debug(f"Angular error computation failed: {e}")
        return None


def calculate_player_metrics(match_data: DemoData) -> dict[str, Any]:
    """
    Calculate tier 1 player metrics from awpy MatchData.

    Computes core metrics for each player in the match:
    - Basic stats (K/D/A, damage, ADR)
    - Time to Damage (TTD) - reaction time metric
    - Crosshair Placement (CP) - aim accuracy metric

    Args:
        match_data: Parsed match data from awpy DemoParser

    Returns:
        Dictionary mapping player_name to PlayerMetrics

    TTD Calculation:
        For each kill, find the first damage event from the same attacker
        to the same victim in the same round, before the kill tick.
        TTD = (kill_tick - first_damage_tick) * (1000.0 / tick_rate)

    Crosshair Placement Calculation:
        For each kill with position/angle data:
        1. Get attacker's position and view angles
        2. Convert view angles to a 3D direction vector
        3. Compute direction from attacker to victim
        4. Calculate angle between the two vectors using arccos(dot product)

    Missing Data Handling:
        - Uses .get() on dicts and .fillna() on DataFrames
        - Skips kills missing required position/angle data for CP
        - Skips kills with no prior damage events for TTD
        - Returns None for TTD/CP if no valid measurements
    """
    # Initialize player data structures
    player_data: dict[str, dict[str, Any]] = {}

    # Get player names from match_data
    for steam_id, name in match_data.player_names.items():
        player_data[name] = {
            "steam_id": steam_id,
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "headshot_kills": 0,
            "damage_total": 0,
            "ttd_values": [],
            "cp_values": [],
        }

    # Note: This is a simplified version - full implementation would
    # require more extensive processing. This is kept here for compatibility
    # but the main processing happens in DemoAnalyzer.

    return player_data


# ============================================================================
# TTD Computation
# ============================================================================


def compute_ttd(analyzer: "DemoAnalyzer") -> None:
    """Compute Time to Damage for each kill with optimized indexing."""
    from opensight.analysis.analytics import TTDResult

    # Check for damage data
    if analyzer.data.damages_df.empty:
        logger.warning("No damage data for TTD computation")
        return

    # Check for kill data - either kills list or kills_df
    has_kills = bool(analyzer.data.kills) or (
        hasattr(analyzer.data, "kills_df") and not analyzer.data.kills_df.empty
    )
    if not has_kills:
        logger.warning("No kill data for TTD computation")
        return

    # Use optimized vectorized implementation if available
    if analyzer._use_optimized and analyzer._metrics_computer is not None:
        logger.info("Using vectorized TTD computation")
        analyzer._metrics_computer.compute(MetricType.TTD)

        # Transfer results to player stats (engagement duration from vectorized computation)
        for steam_id, player in analyzer._players.items():
            player.engagement_duration_values = analyzer._metrics_computer.get_ttd_values(steam_id)
            # ttd_median_ms and ttd_mean_ms are read-only @property on PlayerMatchStats
            # that auto-compute from engagement_duration_values - no explicit set needed
            # Note: prefire_count will be updated by _compute_true_ttd if tick data available

        ttd_metrics = analyzer._metrics_computer.ttd_metrics
        if ttd_metrics:
            logger.info(
                f"Computed TTD (vectorized) for {ttd_metrics.total_engagements} engagements"
            )
        return

    # Fallback: Original per-kill loop implementation
    logger.info("Using per-kill TTD computation (fallback)")
    damages_df = analyzer.data.damages_df
    logger.info(
        f"TTD computation: {len(damages_df)} damage events, {len(analyzer.data.kills)} kills"
    )
    logger.debug(f"Damage columns available: {list(damages_df.columns)}")

    # Find the right column names
    def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
        for col in options:
            if col in df.columns:
                return col
        return None

    dmg_att_col = find_col(damages_df, ["attacker_steamid", "attacker_steam_id", "attacker"])
    dmg_vic_col = find_col(
        damages_df, ["user_steamid", "victim_steamid", "victim_steam_id", "userid"]
    )
    logger.info(f"TTD columns: attacker={dmg_att_col}, victim={dmg_vic_col}")

    if not dmg_att_col or not dmg_vic_col:
        logger.warning(f"Missing columns for TTD. Have: {list(damages_df.columns)}")
        return

    # Pre-build damage lookup index for (attacker, victim) pairs
    # This avoids repeated DataFrame filtering for each kill
    damage_cache: dict[tuple[int, int], list[int]] = {}

    # Convert columns to numeric for reliable comparison
    damages_df = damages_df.copy()
    damages_df[dmg_att_col] = pd.to_numeric(damages_df[dmg_att_col], errors="coerce")
    damages_df[dmg_vic_col] = pd.to_numeric(damages_df[dmg_vic_col], errors="coerce")

    # Group damages by (attacker, victim) and store ticks sorted
    for (att, vic), group in damages_df.groupby([dmg_att_col, dmg_vic_col]):
        if pd.notna(att) and pd.notna(vic):
            damage_cache[(int(att), int(vic))] = sorted(group["tick"].dropna().astype(int).tolist())

    logger.info(f"Built damage cache with {len(damage_cache)} (attacker, victim) pairs")

    # Maximum engagement window: only look for damage within this many ticks
    # before the kill. Prevents picking damage from a previous round.
    max_engagement_ticks = int(5000 / analyzer.MS_PER_TICK) + 1  # ~5 seconds

    # Process kills using cached damage lookups
    raw_ttd_values: dict[int, list[float]] = {}  # For outlier removal later

    # Use kills list if available, otherwise fall back to kills_df
    kills_source = analyzer.data.kills
    use_df_fallback = (
        not kills_source and hasattr(analyzer.data, "kills_df") and not analyzer.data.kills_df.empty
    )

    if use_df_fallback:
        logger.info("Using kills_df for TTD computation (kills list empty)")
        kills_df = analyzer.data.kills_df
        kill_att_col = analyzer._find_col(kills_df, analyzer.ATT_ID_COLS)
        kill_vic_col = analyzer._find_col(kills_df, analyzer.VIC_ID_COLS)

        if kill_att_col and kill_vic_col and "tick" in kills_df.columns:
            for _, row in kills_df.iterrows():
                try:
                    att_id = safe_int(row.get(kill_att_col))
                    vic_id = safe_int(row.get(kill_vic_col))
                    kill_tick = safe_int(row.get("tick"))
                    round_num = safe_int(row.get("round_num", row.get("round", 0)))

                    if not att_id or not vic_id or kill_tick <= 0:
                        continue

                    cache_key = (att_id, vic_id)
                    damage_ticks = damage_cache.get(cache_key, [])

                    if not damage_ticks:
                        continue

                    # Find first damage tick within current engagement window
                    # (not earliest ever — that would span previous rounds)
                    engagement_start = kill_tick - max_engagement_ticks
                    first_dmg_tick = None
                    for tick in damage_ticks:
                        if tick > kill_tick:
                            break  # Past the kill, stop
                        if tick >= engagement_start:
                            first_dmg_tick = tick
                            break  # First damage in window = engagement start

                    if first_dmg_tick is None:
                        continue

                    ttd_ticks = kill_tick - first_dmg_tick
                    ttd_ms = ttd_ticks * analyzer.MS_PER_TICK

                    if ttd_ms < 0 or ttd_ms > 5000:
                        continue

                    is_prefire = ttd_ms <= analyzer.TTD_MIN_MS

                    if att_id not in raw_ttd_values:
                        raw_ttd_values[att_id] = []

                    if not is_prefire and ttd_ms <= analyzer.TTD_MAX_MS:
                        raw_ttd_values[att_id].append(ttd_ms)
                    elif is_prefire and att_id in analyzer._players:
                        analyzer._players[att_id].prefire_count += 1

                    analyzer._ttd_results.append(
                        TTDResult(
                            tick_first_damage=first_dmg_tick,
                            tick_kill=kill_tick,
                            duration_ticks=ttd_ticks,
                            duration_ms=ttd_ms,
                            attacker_steamid=att_id,
                            victim_steamid=vic_id,
                            weapon=str(row.get("weapon", "unknown")),
                            headshot=bool(row.get("headshot", False)),
                            round_num=round_num,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error processing kill row for TTD: {e}")
                    continue
    else:
        # Original path using kills list
        for kill in analyzer.data.kills:
            try:
                att_id = kill.attacker_steamid
                vic_id = kill.victim_steamid
                kill_tick = kill.tick
                round_num = kill.round_num

                if not att_id or not vic_id or kill_tick <= 0:
                    continue

                # Use cached damage lookup
                cache_key = (att_id, vic_id)
                damage_ticks = damage_cache.get(cache_key, [])

                if not damage_ticks:
                    continue

                # Find first damage tick within current engagement window
                # (not earliest ever — that would span previous rounds)
                engagement_start = kill_tick - max_engagement_ticks
                first_dmg_tick = None
                for tick in damage_ticks:
                    if tick > kill_tick:
                        break  # Past the kill, stop
                    if tick >= engagement_start:
                        first_dmg_tick = tick
                        break  # First damage in window = engagement start

                if first_dmg_tick is None:
                    continue

                ttd_ticks = kill_tick - first_dmg_tick
                ttd_ms = ttd_ticks * analyzer.MS_PER_TICK

                # Validate TTD value (filter out invalid/negative values)
                if ttd_ms < 0 or ttd_ms > 5000:  # Max 5 seconds is reasonable
                    continue

                is_prefire = ttd_ms <= analyzer.TTD_MIN_MS

                # Account for wallbangs/through-smoke kills (may have higher TTD)
                getattr(kill, "penetrated", False)
                getattr(kill, "thrusmoke", False)

                # Store raw values for later outlier removal
                if att_id not in raw_ttd_values:
                    raw_ttd_values[att_id] = []

                if not is_prefire and ttd_ms <= analyzer.TTD_MAX_MS:
                    raw_ttd_values[att_id].append(ttd_ms)
                elif is_prefire and att_id in analyzer._players:
                    analyzer._players[att_id].prefire_count += 1

                analyzer._ttd_results.append(
                    TTDResult(
                        tick_first_damage=first_dmg_tick,
                        tick_kill=kill_tick,
                        duration_ticks=ttd_ticks,
                        duration_ms=ttd_ms,
                        attacker_steamid=att_id,
                        victim_steamid=vic_id,
                        weapon=kill.weapon,
                        headshot=kill.headshot,
                        round_num=round_num,
                    )
                )

            except Exception as e:
                logger.debug(f"Error processing kill for TTD: {e}")
                continue

    # Transfer engagement duration values to player stats
    for att_id, values in raw_ttd_values.items():
        if att_id in analyzer._players and values:
            analyzer._players[att_id].engagement_duration_values = values
            analyzer._players[att_id].ttd_median_ms = float(np.median(values))
            analyzer._players[att_id].ttd_mean_ms = float(np.mean(values))

    # Summary of engagement results per player
    players_with_data = sum(1 for p in analyzer._players.values() if p.engagement_duration_values)
    total_values = sum(len(p.engagement_duration_values) for p in analyzer._players.values())
    logger.info(
        f"Computed engagement duration for {len(analyzer._ttd_results)} engagements, "
        f"{players_with_data} players have data ({total_values} total samples)"
    )

    # Calculate true TTD (reaction time) if tick data is available
    compute_true_ttd(analyzer)


def compute_true_ttd(analyzer: "DemoAnalyzer") -> None:
    """Compute true Time to Damage (reaction time: visibility to first damage).

    This uses tick-level position data to determine when the attacker
    first had visibility of the victim, then calculates the time until
    first damage was dealt.

    Requires: analyzer.data.ticks_df with X, Y, Z, pitch, yaw columns
    """
    # Check if tick data is available
    if analyzer.data.ticks_df is None or analyzer.data.ticks_df.empty:
        logger.info("No tick data available for true TTD calculation - skipping")
        return

    ticks_df = analyzer.data.ticks_df
    damages_df = analyzer.data.damages_df

    # Check for required columns
    required_cols = ["tick", "X", "Y", "Z"]
    view_cols = ["pitch", "yaw"]

    has_position = all(col in ticks_df.columns for col in required_cols)
    has_view = all(col in ticks_df.columns for col in view_cols)

    if not has_position:
        logger.info("Tick data missing position columns for true TTD - skipping")
        return

    if not has_view:
        logger.info("Tick data missing view angle columns - using position-only visibility")

    # Find steamid column in tick data
    steamid_col = None
    for col in ["steamid", "steam_id", "user_steamid"]:
        if col in ticks_df.columns:
            steamid_col = col
            break

    if not steamid_col:
        logger.warning("No steamid column in tick data for true TTD")
        return

    logger.info(f"Computing true TTD with {len(ticks_df)} tick entries")

    # For each damage event, try to find visibility start
    dmg_att_col = analyzer._find_col(damages_df, analyzer.ATT_ID_COLS)
    dmg_vic_col = analyzer._find_col(damages_df, analyzer.VIC_ID_COLS)

    if not dmg_att_col or not dmg_vic_col:
        logger.warning("Missing columns in damage data for true TTD")
        return

    true_ttd_values: dict[int, list[float]] = {}
    prefire_counts: dict[int, int] = {}
    processed = 0
    visibility_found = 0

    # Process unique (attacker, victim) pairs from damage events
    # Group damages to get first damage tick per engagement
    for (att_id, vic_id), group in damages_df.groupby([dmg_att_col, dmg_vic_col]):
        try:
            att_id = int(att_id)
            vic_id = int(vic_id)
            first_dmg_tick = int(group["tick"].min())

            # Get position data for attacker and victim
            attacker_ticks = ticks_df[ticks_df[steamid_col] == att_id]
            victim_ticks = ticks_df[ticks_df[steamid_col] == vic_id]

            if attacker_ticks.empty or victim_ticks.empty:
                continue

            # Find visibility start using simplified check
            visibility_tick = find_visibility_start_simple(
                analyzer, attacker_ticks, victim_ticks, first_dmg_tick, has_view
            )

            if visibility_tick is None:
                continue

            visibility_found += 1

            # Calculate true TTD
            reaction_ticks = first_dmg_tick - visibility_tick
            reaction_ms = reaction_ticks * analyzer.MS_PER_TICK

            # Sanity filter
            if reaction_ms < 0:
                continue  # Invalid - damage before visibility

            processed += 1

            # Classify as prefire or valid reaction
            if reaction_ms < analyzer.REACTION_TIME_MIN_MS:
                # Prefire - player was pre-aiming
                prefire_counts[att_id] = prefire_counts.get(att_id, 0) + 1
            elif reaction_ms <= analyzer.REACTION_TIME_MAX_MS:
                # Valid reaction time sample
                if att_id not in true_ttd_values:
                    true_ttd_values[att_id] = []
                true_ttd_values[att_id].append(reaction_ms)
            # > REACTION_TIME_MAX_MS: visibility logic likely failed, skip

        except Exception as e:
            logger.debug(f"Error processing true TTD for engagement: {e}")
            continue

    # Transfer to player stats
    for att_id, values in true_ttd_values.items():
        if att_id in analyzer._players and values:
            analyzer._players[att_id].true_ttd_values = values

    # Update prefire counts (true prefires based on reaction time)
    for att_id, count in prefire_counts.items():
        if att_id in analyzer._players:
            analyzer._players[att_id].prefire_count = count

    players_with_ttd = sum(1 for p in analyzer._players.values() if p.true_ttd_values)
    total_prefires = sum(prefire_counts.values())

    logger.info(
        f"True TTD computed: {processed} engagements analyzed, "
        f"{visibility_found} with visibility data, "
        f"{players_with_ttd} players have reaction time data, "
        f"{total_prefires} prefires detected"
    )


def find_visibility_start_simple(
    analyzer: "DemoAnalyzer",
    attacker_ticks: pd.DataFrame,
    victim_ticks: pd.DataFrame,
    damage_tick: int,
    use_view_angles: bool = True,
) -> int | None:
    """Find when attacker first had visibility of victim (simplified).

    Uses a distance + FOV check without ray-casting against map geometry.
    This is an approximation that works reasonably well for open areas.

    Args:
        analyzer: DemoAnalyzer instance
        attacker_ticks: Attacker position data (needs X, Y, Z, optionally pitch/yaw)
        victim_ticks: Victim position data (needs X, Y, Z)
        damage_tick: Tick when damage occurred
        use_view_angles: Whether to check if victim is in FOV

    Returns:
        Tick when visibility started, or None if not determinable
    """
    MAX_LOOKBACK_TICKS = int(2000 / analyzer.MS_PER_TICK)  # 2 seconds max lookback
    MIN_DISTANCE = 50  # Units - too close is likely already in combat
    MAX_DISTANCE = 3000  # Units - beyond this visibility is questionable
    FOV_THRESHOLD = 90  # Degrees - consider visible if within this FOV

    start_tick = max(0, damage_tick - MAX_LOOKBACK_TICKS)

    # Get data in lookback window
    att_window = attacker_ticks[
        (attacker_ticks["tick"] >= start_tick) & (attacker_ticks["tick"] <= damage_tick)
    ].sort_values("tick")

    vic_window = victim_ticks[
        (victim_ticks["tick"] >= start_tick) & (victim_ticks["tick"] <= damage_tick)
    ].sort_values("tick")

    if att_window.empty or vic_window.empty:
        return None

    # Check each tick from earliest to find first visibility
    for _, att_row in att_window.iterrows():
        tick = int(att_row["tick"])

        # Find victim position at same tick (or closest)
        vic_at_tick = vic_window[vic_window["tick"] == tick]
        if vic_at_tick.empty:
            # Try closest tick
            vic_at_tick = vic_window.iloc[(vic_window["tick"] - tick).abs().argsort()[:1]]
            if vic_at_tick.empty:
                continue

        vic_row = vic_at_tick.iloc[0]

        # Calculate distance
        try:
            att_pos = np.array([att_row["X"], att_row["Y"], att_row["Z"]])
            vic_pos = np.array([vic_row["X"], vic_row["Y"], vic_row["Z"]])
            direction = vic_pos - att_pos
            distance = np.linalg.norm(direction)

            if distance < MIN_DISTANCE or distance > MAX_DISTANCE:
                continue

            # If we have view angles, check FOV
            if use_view_angles and "pitch" in att_row and "yaw" in att_row:
                pitch = float(att_row["pitch"])
                yaw = float(att_row["yaw"])

                # Convert view angles to direction vector
                pitch_rad = np.radians(pitch)
                yaw_rad = np.radians(yaw)
                view_dir = np.array(
                    [
                        np.cos(yaw_rad) * np.cos(pitch_rad),
                        np.sin(yaw_rad) * np.cos(pitch_rad),
                        -np.sin(pitch_rad),
                    ]
                )

                # Calculate angle to target
                target_dir = direction / distance
                dot = np.clip(np.dot(view_dir, target_dir), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))

                if angle <= FOV_THRESHOLD:
                    return tick
            else:
                # No view angles - assume visible if in range
                return tick

        except Exception as e:
            logger.debug(f"Error checking visibility at tick: {e}")
            continue

    return None


# ============================================================================
# Crosshair Placement Computation
# ============================================================================


def compute_crosshair_placement(analyzer: "DemoAnalyzer") -> None:
    """Compute crosshair placement error for each kill.

    Uses vectorized numpy implementation when available for ~5-20x speedup.
    Falls back to per-kill loop for compatibility.
    """
    # Use optimized vectorized implementation if available
    if analyzer._use_optimized and analyzer._metrics_computer is not None:
        logger.info("Using vectorized CP computation")
        analyzer._metrics_computer.compute(MetricType.CP)

        # Transfer results to player stats
        for steam_id, player in analyzer._players.items():
            player.cp_values = analyzer._metrics_computer.get_cp_values(steam_id)

        cp_metrics = analyzer._metrics_computer.cp_metrics
        if cp_metrics:
            logger.info(f"Computed CP (vectorized) for {cp_metrics.total_kills_analyzed} kills")
        return

    # Fallback: Original implementation
    logger.info("Using per-kill CP computation (fallback)")

    # First try using KillEvent objects directly (preferred - they have embedded position data)
    # Count what data is available
    kills_with_x = sum(1 for k in analyzer.data.kills if k.attacker_x is not None)
    kills_with_pitch = sum(1 for k in analyzer.data.kills if k.attacker_pitch is not None)
    kills_with_victim_x = sum(1 for k in analyzer.data.kills if k.victim_x is not None)
    logger.info(
        f"CP data availability: {len(analyzer.data.kills)} kills, "
        f"{kills_with_x} with attacker_x, {kills_with_pitch} with pitch, "
        f"{kills_with_victim_x} with victim_x"
    )

    kills_with_pos = [
        k
        for k in analyzer.data.kills
        if k.attacker_x is not None and k.attacker_pitch is not None and k.victim_x is not None
    ]

    if kills_with_pos:
        logger.info(f"Computing CP from {len(kills_with_pos)} KillEvent objects with position data")
        compute_cp_from_kill_events(analyzer, kills_with_pos)
        return

    # Fallback: check DataFrame for position columns
    if not analyzer.data.kills_df.empty:
        kills_df = analyzer.data.kills_df

        # Check various column name patterns
        pos_patterns = [
            [
                "attacker_X",
                "attacker_Y",
                "attacker_Z",
                "victim_X",
                "victim_Y",
                "victim_Z",
            ],
            [
                "attacker_x",
                "attacker_y",
                "attacker_z",
                "victim_x",
                "victim_y",
                "victim_z",
            ],
            [
                "attacker_X",
                "attacker_Y",
                "attacker_Z",
                "user_X",
                "user_Y",
                "user_Z",
            ],
        ]
        angle_patterns = [
            ["attacker_pitch", "attacker_yaw"],
        ]

        has_positions = any(
            all(col in kills_df.columns for col in pattern) for pattern in pos_patterns
        )
        has_angles = any(
            all(col in kills_df.columns for col in pattern) for pattern in angle_patterns
        )

        if has_positions and has_angles:
            logger.info(f"Computing CP from DataFrame. Columns: {list(kills_df.columns)}")
            compute_cp_from_events(analyzer)
            return

    # Final fallback: tick data
    if analyzer.data.ticks_df is not None and not analyzer.data.ticks_df.empty:
        compute_cp_from_ticks(analyzer)
    else:
        logger.warning(
            "No position/angle data available for CP computation. Position data requires parsing with player props."
        )


def compute_cp_from_kill_events(analyzer: "DemoAnalyzer", kills: list) -> None:
    """Compute CP from KillEvent objects with optimized vectorized calculations."""
    from opensight.analysis.analytics import CrosshairPlacementResult

    # Constants
    MAX_DISTANCE = 2000.0  # Skip kills beyond this distance (unlikely to be meaningful CP data)
    EYE_HEIGHT = 64.0

    # Pre-allocate arrays for batch processing
    valid_kills = []
    att_positions = []
    vic_positions = []
    att_pitches = []
    att_yaws = []

    # First pass: validate and collect data
    for kill in kills:
        att_id = kill.attacker_steamid
        vic_id = kill.victim_steamid

        if not att_id or not vic_id:
            continue

        # Validate position data (not zero or NaN)
        att_x = kill.attacker_x
        att_y = kill.attacker_y
        att_z = kill.attacker_z
        vic_x = kill.victim_x
        vic_y = kill.victim_y
        vic_z = kill.victim_z

        if any(
            v is None or (isinstance(v, float) and np.isnan(v))
            for v in [att_x, att_y, vic_x, vic_y]
        ):
            continue

        # Skip if positions are clearly invalid (all zeros)
        if abs(att_x) < 0.001 and abs(att_y) < 0.001:
            continue
        if abs(vic_x) < 0.001 and abs(vic_y) < 0.001:
            continue

        # Calculate distance for filtering
        dx = vic_x - att_x
        dy = vic_y - att_y
        dz = (vic_z or 0) - (att_z or 0)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Skip very long distance kills
        if distance > MAX_DISTANCE:
            continue

        valid_kills.append(kill)
        att_positions.append([att_x, att_y, (att_z or 0) + EYE_HEIGHT])
        vic_positions.append([vic_x, vic_y, (vic_z or 0) + EYE_HEIGHT])
        att_pitches.append(kill.attacker_pitch or 0.0)
        att_yaws.append(kill.attacker_yaw or 0.0)

    if not valid_kills:
        logger.info("No valid kills with position data for CP calculation")
        return

    # Convert to numpy arrays for vectorized operations
    att_pos_arr = np.array(att_positions)
    vic_pos_arr = np.array(vic_positions)
    pitch_arr = np.array(att_pitches)
    yaw_arr = np.array(att_yaws)

    # Vectorized angular error calculation
    pitch_rad = np.radians(pitch_arr)
    yaw_rad = np.radians(yaw_arr)

    # View vectors from Euler angles (vectorized)
    view_x = np.cos(yaw_rad) * np.cos(pitch_rad)
    view_y = np.sin(yaw_rad) * np.cos(pitch_rad)
    view_z = -np.sin(pitch_rad)
    view_vecs = np.column_stack([view_x, view_y, view_z])

    # Ideal vectors (normalized direction to victim)
    ideal_vecs = vic_pos_arr - att_pos_arr
    distances = np.linalg.norm(ideal_vecs, axis=1, keepdims=True)
    distances = np.maximum(distances, 0.001)  # Avoid division by zero
    ideal_vecs = ideal_vecs / distances

    # Dot products and angular errors (vectorized)
    dots = np.sum(view_vecs * ideal_vecs, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angular_errors = np.degrees(np.arccos(dots))

    # Separate pitch/yaw errors
    ideal_pitches = np.degrees(np.arcsin(-ideal_vecs[:, 2]))
    ideal_yaws = np.degrees(np.arctan2(ideal_vecs[:, 1], ideal_vecs[:, 0]))
    pitch_errors = pitch_arr - ideal_pitches
    yaw_errors = yaw_arr - ideal_yaws

    # Normalize yaw errors to [-180, 180]
    yaw_errors = np.mod(yaw_errors + 180, 360) - 180

    # Store results
    for i, kill in enumerate(valid_kills):
        att_id = kill.attacker_steamid
        angular_error = float(angular_errors[i])
        pitch_error = float(pitch_errors[i])
        yaw_error = float(yaw_errors[i])

        if att_id in analyzer._players:
            analyzer._players[att_id].cp_values.append(angular_error)

        analyzer._cp_results.append(
            CrosshairPlacementResult(
                tick=kill.tick,
                attacker_steamid=att_id,
                victim_steamid=kill.victim_steamid,
                angular_error_deg=angular_error,
                pitch_error_deg=pitch_error,
                yaw_error_deg=yaw_error,
                round_num=kill.round_num,
            )
        )

    logger.info(
        f"Computed CP for {len(analyzer._cp_results)} kills from KillEvent objects (vectorized)"
    )


def compute_cp_from_events(analyzer: "DemoAnalyzer") -> None:
    """Compute CP from position data embedded in kill events DataFrame (optimized)."""
    from opensight.analysis.analytics import CrosshairPlacementResult

    kills_df = analyzer.data.kills_df
    logger.info("Computing CP from DataFrame event-embedded positions")

    # Constants
    MAX_DISTANCE = 2000.0
    EYE_HEIGHT = 64.0

    # Find position columns
    att_x_col = "attacker_X" if "attacker_X" in kills_df.columns else "attacker_x"
    att_y_col = "attacker_Y" if "attacker_Y" in kills_df.columns else "attacker_y"
    att_z_col = "attacker_Z" if "attacker_Z" in kills_df.columns else "attacker_z"
    vic_x_col = (
        "victim_X"
        if "victim_X" in kills_df.columns
        else ("user_X" if "user_X" in kills_df.columns else "victim_x")
    )
    vic_y_col = (
        "victim_Y"
        if "victim_Y" in kills_df.columns
        else ("user_Y" if "user_Y" in kills_df.columns else "victim_y")
    )
    vic_z_col = (
        "victim_Z"
        if "victim_Z" in kills_df.columns
        else ("user_Z" if "user_Z" in kills_df.columns else "victim_z")
    )

    # Filter valid rows with position data
    required_cols = [
        att_x_col,
        att_y_col,
        vic_x_col,
        vic_y_col,
        "attacker_pitch",
        "attacker_yaw",
    ]
    if not all(col in kills_df.columns for col in required_cols):
        logger.warning(f"Missing position columns for CP. Have: {list(kills_df.columns)}")
        return

    # Create working copy with validated data
    df = kills_df.copy()
    df["_att_x"] = pd.to_numeric(df[att_x_col], errors="coerce")
    df["_att_y"] = pd.to_numeric(df[att_y_col], errors="coerce")
    if att_z_col and att_z_col in df.columns:
        df["_att_z"] = pd.to_numeric(df[att_z_col], errors="coerce").fillna(0) + EYE_HEIGHT
    else:
        df["_att_z"] = float(EYE_HEIGHT)
    df["_vic_x"] = pd.to_numeric(df[vic_x_col], errors="coerce")
    df["_vic_y"] = pd.to_numeric(df[vic_y_col], errors="coerce")
    if vic_z_col and vic_z_col in df.columns:
        df["_vic_z"] = pd.to_numeric(df[vic_z_col], errors="coerce").fillna(0) + EYE_HEIGHT
    else:
        df["_vic_z"] = float(EYE_HEIGHT)
    df["_pitch"] = pd.to_numeric(df["attacker_pitch"], errors="coerce").fillna(0)
    df["_yaw"] = pd.to_numeric(df["attacker_yaw"], errors="coerce").fillna(0)

    # Filter out invalid positions - require all notna and at least some non-zero positions
    valid_mask = (
        df["_att_x"].notna()
        & df["_att_y"].notna()
        & df["_vic_x"].notna()
        & df["_vic_y"].notna()
        & ((df["_att_x"].abs() > 0.001) | (df["_att_y"].abs() > 0.001))
        & ((df["_vic_x"].abs() > 0.001) | (df["_vic_y"].abs() > 0.001))
    )
    df = df[valid_mask]

    if df.empty:
        logger.warning("No valid positions for CP calculation")
        return

    # Calculate distances
    df["_dist"] = np.sqrt(
        (df["_vic_x"] - df["_att_x"]) ** 2
        + (df["_vic_y"] - df["_att_y"]) ** 2
        + (df["_vic_z"] - df["_att_z"]) ** 2
    )

    # Filter by distance
    df = df[df["_dist"] <= MAX_DISTANCE]

    if df.empty:
        return

    # Vectorized calculations
    pitch_rad = np.radians(df["_pitch"].values)
    yaw_rad = np.radians(df["_yaw"].values)

    # View vectors
    view_x = np.cos(yaw_rad) * np.cos(pitch_rad)
    view_y = np.sin(yaw_rad) * np.cos(pitch_rad)
    view_z = -np.sin(pitch_rad)

    # Ideal vectors
    ideal_x = (df["_vic_x"].values - df["_att_x"].values) / df["_dist"].values
    ideal_y = (df["_vic_y"].values - df["_att_y"].values) / df["_dist"].values
    ideal_z = (df["_vic_z"].values - df["_att_z"].values) / df["_dist"].values

    # Angular errors
    dots = view_x * ideal_x + view_y * ideal_y + view_z * ideal_z
    dots = np.clip(dots, -1.0, 1.0)
    angular_errors = np.degrees(np.arccos(dots))

    # Component errors
    ideal_pitches = np.degrees(np.arcsin(-ideal_z))
    ideal_yaws = np.degrees(np.arctan2(ideal_y, ideal_x))
    pitch_errors = df["_pitch"].values - ideal_pitches
    yaw_errors = np.mod(df["_yaw"].values - ideal_yaws + 180, 360) - 180

    # Store results
    att_id_col = analyzer._find_col(df, analyzer.ATT_ID_COLS)
    vic_id_col = analyzer._find_col(df, analyzer.VIC_ID_COLS)

    for i, (_idx, row) in enumerate(df.iterrows()):
        att_id = safe_int(row.get(att_id_col)) if att_id_col else 0
        vic_id = safe_int(row.get(vic_id_col)) if vic_id_col else 0
        tick = safe_int(row.get("tick"))
        round_num = safe_int(row.get("round_num", 0))

        if att_id in analyzer._players:
            analyzer._players[att_id].cp_values.append(angular_errors[i])

        analyzer._cp_results.append(
            CrosshairPlacementResult(
                tick=tick,
                attacker_steamid=att_id,
                victim_steamid=vic_id,
                angular_error_deg=float(angular_errors[i]),
                pitch_error_deg=float(pitch_errors[i]),
                yaw_error_deg=float(yaw_errors[i]),
                round_num=round_num,
            )
        )

    logger.info(f"Computed CP for {len(analyzer._cp_results)} kills (vectorized)")


def compute_cp_from_ticks(analyzer: "DemoAnalyzer") -> None:
    """Compute CP from tick-level data (fallback)."""
    from opensight.analysis.analytics import CrosshairPlacementResult

    ticks_df = analyzer.data.ticks_df
    kills_df = analyzer.data.kills_df
    logger.info("Computing CP from tick data (slower)")

    required_cols = ["steamid", "X", "Y", "Z", "pitch", "yaw", "tick"]
    if not all(col in ticks_df.columns for col in required_cols):
        logger.warning("Missing columns for tick-based CP")
        return

    for _, kill_row in kills_df.iterrows():
        try:
            att_id = safe_int(kill_row.get("attacker_steamid"))
            vic_id = safe_int(kill_row.get("victim_steamid"))
            kill_tick = safe_int(kill_row.get("tick"))
            round_num = safe_int(kill_row.get("round_num", 0))

            if not att_id or not vic_id:
                continue

            att_ticks = ticks_df[(ticks_df["steamid"] == att_id) & (ticks_df["tick"] <= kill_tick)]
            vic_ticks = ticks_df[(ticks_df["steamid"] == vic_id) & (ticks_df["tick"] <= kill_tick)]

            if att_ticks.empty or vic_ticks.empty:
                continue

            att_state = att_ticks.iloc[-1]
            vic_state = vic_ticks.iloc[-1]

            att_pos = np.array(
                [
                    safe_float(att_state["X"]),
                    safe_float(att_state["Y"]),
                    safe_float(att_state["Z"]) + 64,
                ]
            )
            att_pitch = safe_float(att_state["pitch"])
            att_yaw = safe_float(att_state["yaw"])

            vic_pos = np.array(
                [
                    safe_float(vic_state["X"]),
                    safe_float(vic_state["Y"]),
                    safe_float(vic_state["Z"]) + 64,
                ]
            )

            angular_error, pitch_error, yaw_error = calculate_angular_error(
                att_pos, att_pitch, att_yaw, vic_pos
            )

            if att_id in analyzer._players:
                analyzer._players[att_id].cp_values.append(angular_error)

            analyzer._cp_results.append(
                CrosshairPlacementResult(
                    tick=kill_tick,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    angular_error_deg=angular_error,
                    pitch_error_deg=pitch_error,
                    yaw_error_deg=yaw_error,
                    round_num=round_num,
                )
            )

        except Exception as e:
            logger.debug(f"Error in tick-based CP: {e}")
            continue

    logger.info(f"Computed CP for {len(analyzer._cp_results)} kills (tick-based)")


def calculate_angular_error(
    attacker_pos: np.ndarray,
    pitch_deg: float,
    yaw_deg: float,
    victim_pos: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate angular error between view direction and target."""
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)

    # View vector from Euler angles
    view_x = math.cos(yaw_rad) * math.cos(pitch_rad)
    view_y = math.sin(yaw_rad) * math.cos(pitch_rad)
    view_z = -math.sin(pitch_rad)
    view_vec = np.array([view_x, view_y, view_z])

    # Ideal vector
    ideal_vec = victim_pos - attacker_pos
    distance = np.linalg.norm(ideal_vec)
    if distance < 0.001:
        return 0.0, 0.0, 0.0

    ideal_vec = ideal_vec / distance

    # Total angular error
    dot = np.clip(np.dot(view_vec, ideal_vec), -1.0, 1.0)
    angular_error = math.degrees(math.acos(dot))

    # Separate pitch/yaw errors
    ideal_pitch = math.degrees(math.asin(-ideal_vec[2]))
    ideal_yaw = math.degrees(math.atan2(ideal_vec[1], ideal_vec[0]))

    pitch_error = pitch_deg - ideal_pitch
    yaw_error = yaw_deg - ideal_yaw

    while yaw_error > 180:
        yaw_error -= 360
    while yaw_error < -180:
        yaw_error += 360

    return angular_error, pitch_error, yaw_error


# ============================================================================
# Accuracy Computation
# ============================================================================


def calculate_accuracy_stats(analyzer: "DemoAnalyzer") -> None:
    """Calculate accuracy statistics from weapon_fire events."""
    if not hasattr(analyzer.data, "weapon_fires") or not analyzer.data.weapon_fires:
        logger.info("No weapon_fire data available for accuracy stats")
        return

    damages_df = analyzer.data.damages_df

    # Pre-build weapon_fires lookup by steamid for O(n) instead of O(n*m)
    fires_by_player: dict[int, list] = {}
    for f in analyzer.data.weapon_fires:
        fires_by_player.setdefault(f.player_steamid, []).append(f)

    # Pre-compute damage lookup columns
    att_col = analyzer._find_col(damages_df, analyzer.ATT_ID_COLS) if not damages_df.empty else None
    hitgroup_col = analyzer._find_col(damages_df, ["hitgroup"]) if not damages_df.empty else None

    for steam_id, player in analyzer._players.items():
        # Count shots fired (use pre-built lookup)
        player_shots = fires_by_player.get(steam_id, [])
        if not player_shots:
            # Try float comparison as fallback (handles int/float steamid mismatch)
            player_shots = fires_by_player.get(int(float(steam_id)), [])
        player.shots_fired = len(player_shots)

        # Count shots that hit (from damage events)
        # Deduplicate by tick to avoid overcounting (one bullet = one hit even if
        # it causes multiple damage rows, e.g. armor + health)
        if not damages_df.empty and att_col:
            player_hits = analyzer._match_steamid(damages_df, att_col, steam_id)
            if "tick" in player_hits.columns:
                # Count unique ticks = unique hits
                player.shots_hit = player_hits["tick"].nunique()
            else:
                player.shots_hit = len(player_hits)

            # Count headshot hits — hitgroup is numeric in demoparser2 (1 = head)
            if hitgroup_col:
                hitgroup_vals = player_hits[hitgroup_col]
                # Handle both numeric (demoparser2: 1=head) and string formats
                try:
                    head_hits = player_hits[hitgroup_vals.astype(float).eq(1.0)]
                except (ValueError, TypeError):
                    head_hits = player_hits[
                        hitgroup_vals.astype(str).str.lower().str.contains("head", na=False)
                    ]
                if "tick" in head_hits.columns:
                    player.headshot_hits = head_hits["tick"].nunique()
                else:
                    player.headshot_hits = len(head_hits)

        # Calculate spray accuracy
        calculate_spray_accuracy_for_player(analyzer, player, player_shots, damages_df)

        # Calculate counter-strafing
        calculate_counter_strafing_for_player(analyzer, player, steam_id)

    logger.info("Calculated accuracy stats (including spray and counter-strafing)")


def calculate_spray_accuracy_for_player(
    analyzer: "DemoAnalyzer",
    player: Any,
    player_shots: list,
    damages_df: pd.DataFrame,
) -> None:
    """Calculate spray accuracy for a single player.

    Spray accuracy = hits after 4th bullet / shots after 4th bullet in a burst.
    """
    # Define weapons that support spray (exclude pistols, snipers, shotguns)
    spray_weapons = {
        "ak47",
        "m4a1",
        "m4a1_silencer",
        "m4a4",
        "galil",
        "famas",
        "aug",
        "sg556",
        "mp9",
        "mac10",
        "mp7",
        "ump45",
        "p90",
        "bizon",
        "mp5sd",
        "negev",
        "m249",
    }
    burst_tick_window = 20  # ~312ms at 64 tick

    if not player_shots:
        return

    # Filter to spray weapons only
    spray_shots = [
        s
        for s in player_shots
        if s.weapon and s.weapon.lower().replace("weapon_", "") in spray_weapons
    ]

    if not spray_shots:
        return

    # Sort by tick
    spray_shots.sort(key=lambda s: s.tick)

    # Detect bursts of 4+ consecutive shots, then include ALL shots in each burst.
    # Leetify counts every shot in a qualifying spray, not just shots 4+.
    spray_shot_ticks = []
    burst_start = 0
    burst_shot_count = 1

    for i in range(1, len(spray_shots)):
        current = spray_shots[i]
        previous = spray_shots[i - 1]

        if current.tick - previous.tick <= burst_tick_window:
            burst_shot_count += 1
        else:
            # End of burst — if it was 4+ shots, include ALL shots from burst_start
            if burst_shot_count >= 4:
                for j in range(burst_start, i):
                    spray_shot_ticks.append(spray_shots[j].tick)
            burst_start = i
            burst_shot_count = 1

    # Don't forget the last burst
    if burst_shot_count >= 4:
        for j in range(burst_start, len(spray_shots)):
            spray_shot_ticks.append(spray_shots[j].tick)

    player.spray_shots_fired = len(spray_shot_ticks)

    # Count spray hits
    if not damages_df.empty and spray_shot_ticks:
        att_col = analyzer._find_col(damages_df, analyzer.ATT_ID_COLS)
        if att_col and "tick" in damages_df.columns:
            player_damages = analyzer._match_steamid(damages_df, att_col, player.steam_id)
            if not player_damages.empty:
                damage_ticks = set(player_damages["tick"].values)
                spray_hits = 0
                for shot_tick in spray_shot_ticks:
                    for dt in range(shot_tick, shot_tick + 5):
                        if dt in damage_ticks:
                            spray_hits += 1
                            break
                player.spray_shots_hit = spray_hits


def calculate_counter_strafing_for_player(
    analyzer: "DemoAnalyzer", player: Any, steam_id: int
) -> None:
    """Calculate counter-strafing percentage for a player.

    Leetify-style metric: measures movement discipline across ALL shots fired,
    not just kill shots. A shot is "stationary" if velocity < 34 units/s.

    Excludes non-gun weapons (knife, grenades, C4) since movement doesn't
    affect their accuracy.
    """
    # Velocity threshold - standard CS2 "stopped" speed for accurate shooting
    cs_velocity_threshold = 34.0

    # Weapons to EXCLUDE from counter-strafe tracking (movement doesn't matter)
    excluded_weapons = {
        "knife",
        "knife_t",
        "bayonet",
        "hegrenade",
        "flashbang",
        "smokegrenade",
        "molotov",
        "incgrenade",
        "decoy",
        "c4",
        "taser",  # Zeus
    }

    if not hasattr(analyzer.data, "weapon_fires") or not analyzer.data.weapon_fires:
        return

    # Count stationary vs moving shots across ALL weapon_fire events
    shots_stationary = 0
    shots_with_velocity = 0

    # Also track kill-based stats for backward compatibility
    counter_strafe_kills = 0
    tracked_kills = 0

    # Build velocity lookup for kill-based tracking (backward compat)
    velocity_by_tick: dict[int, float] = {}

    for fire in analyzer.data.weapon_fires:
        if fire.player_steamid != steam_id:
            continue

        # Get weapon name, normalize it
        weapon = (fire.weapon or "").lower().replace("weapon_", "")

        # Skip non-gun weapons
        if weapon in excluded_weapons:
            continue

        # Check if velocity data is available
        if fire.velocity_x is None or fire.velocity_y is None:
            continue

        # Calculate 2D velocity (horizontal movement)
        vel_x = fire.velocity_x or 0.0
        vel_y = fire.velocity_y or 0.0
        velocity_2d = math.sqrt(vel_x**2 + vel_y**2)

        # Track for shot-based metric (NEW - Leetify parity)
        shots_with_velocity += 1
        if velocity_2d < cs_velocity_threshold:
            shots_stationary += 1

        # Store for kill-based lookup (backward compat)
        velocity_by_tick[fire.tick] = velocity_2d

    # Set shot-based stats (NEW - Leetify parity)
    player.shots_stationary = shots_stationary
    player.shots_with_velocity = shots_with_velocity

    # Calculate kill-based stats for backward compatibility
    if velocity_by_tick:
        kills_by_player = [k for k in analyzer.data.kills if k.attacker_steamid == steam_id]

        for kill in kills_by_player:
            kill_tick = kill.tick
            closest_velocity = None
            min_tick_diff = float("inf")

            for fire_tick, velocity in velocity_by_tick.items():
                tick_diff = abs(fire_tick - kill_tick)
                if tick_diff < min_tick_diff and fire_tick <= kill_tick and tick_diff <= 10:
                    min_tick_diff = tick_diff
                    closest_velocity = velocity

            if closest_velocity is not None:
                tracked_kills += 1
                if closest_velocity < cs_velocity_threshold:
                    counter_strafe_kills += 1

    # Set kill-based stats (DEPRECATED but kept for backward compatibility)
    player.counter_strafe_kills = counter_strafe_kills
    player.total_kills_with_velocity = tracked_kills
