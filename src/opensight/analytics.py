"""
Advanced Analytics for CS2 Demo Analysis

Implements Time to Damage (TTD) and Crosshair Placement (CP) metrics.
Based on professional esports analytics methodology.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import math

import numpy as np
import pandas as pd

from opensight.parser import DemoData, KillEvent, DamageEvent, safe_int, safe_str, safe_bool

logger = logging.getLogger(__name__)


def safe_float(value, default: float = 0.0) -> float:
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
class TTDResult:
    """Time to Damage result for a single engagement."""
    tick_spotted: int
    tick_damage: int
    ttd_ticks: int
    ttd_ms: float
    attacker_steamid: int
    victim_steamid: int
    weapon: str
    headshot: bool
    is_prefire: bool  # TTD <= 0 indicates prediction/prefire


@dataclass
class CrosshairPlacementResult:
    """Crosshair placement analysis for a kill."""
    tick: int
    attacker_steamid: int
    victim_steamid: int
    angular_error_deg: float  # Degrees off-target at moment of engagement
    pitch_error_deg: float  # Vertical error (negative = aiming too low)
    yaw_error_deg: float  # Horizontal error


@dataclass
class PlayerAnalytics:
    """Complete analytics for a player."""
    steam_id: int
    name: str
    team: str

    # Basic stats
    kills: int
    deaths: int
    assists: int
    adr: float
    hs_percent: float

    # TTD Stats
    ttd_median_ms: Optional[float]
    ttd_mean_ms: Optional[float]
    ttd_min_ms: Optional[float]
    ttd_max_ms: Optional[float]
    ttd_std_ms: Optional[float]
    ttd_count: int
    prefire_count: int

    # Crosshair Placement Stats
    cp_median_error_deg: Optional[float]
    cp_mean_error_deg: Optional[float]
    cp_pitch_bias_deg: Optional[float]  # Negative = tends to aim low

    # Weapon breakdown
    weapon_kills: dict[str, int]

    # Raw data for histograms
    ttd_values: list[float]
    cp_values: list[float]


class DemoAnalyzer:
    """Analyzer for computing advanced metrics from parsed demo data."""

    # Constants
    TICK_RATE = 64  # CS2 tick rate
    MS_PER_TICK = 1000 / TICK_RATE  # ~15.625ms

    # TTD filtering thresholds
    TTD_MIN_MS = 0  # Below this is prefire
    TTD_MAX_MS = 1500  # Above this is likely not a reaction shot

    def __init__(self, demo_data: DemoData):
        self.data = demo_data
        self._ttd_results: list[TTDResult] = []
        self._cp_results: list[CrosshairPlacementResult] = []

    def analyze(self) -> dict[int, PlayerAnalytics]:
        """Run full analysis and return per-player analytics."""
        logger.info("Starting advanced analysis...")

        # Compute TTD for all kills
        self._compute_ttd()

        # Compute crosshair placement (if tick data available)
        if self.data.ticks_df is not None:
            self._compute_crosshair_placement()

        # Build per-player analytics
        player_analytics = self._build_player_analytics()

        logger.info(f"Analysis complete. {len(player_analytics)} players analyzed.")
        return player_analytics

    def _compute_ttd(self) -> None:
        """
        Compute Time to Damage for each kill.

        TTD = Time from first visibility to first damage.
        Since we don't have full visibility raycasting, we approximate
        by using the time from first damage to kill.
        """
        if self.data.damages_df.empty or self.data.kills_df.empty:
            logger.warning("No damage or kill data for TTD computation")
            return

        kills_df = self.data.kills_df
        damages_df = self.data.damages_df

        # Find column names
        def find_col(df: pd.DataFrame, options: list[str]) -> Optional[str]:
            for col in options:
                if col in df.columns:
                    return col
            return None

        kill_att = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        kill_vic = find_col(kills_df, ["user_steamid", "victim_steamid"])
        kill_tick = find_col(kills_df, ["tick"])
        kill_weapon = find_col(kills_df, ["weapon"])
        kill_hs = find_col(kills_df, ["headshot"])

        dmg_att = find_col(damages_df, ["attacker_steamid", "attacker_steam_id"])
        dmg_vic = find_col(damages_df, ["user_steamid", "victim_steamid"])
        dmg_tick = find_col(damages_df, ["tick"])

        if not all([kill_att, kill_vic, kill_tick, dmg_att, dmg_vic, dmg_tick]):
            logger.warning("Missing columns for TTD computation")
            return

        # For each kill, find the first damage event from that attacker to that victim
        for _, kill_row in kills_df.iterrows():
            try:
                att_id = safe_int(kill_row[kill_att])
                vic_id = safe_int(kill_row[kill_vic])
                kill_tick_val = safe_int(kill_row[kill_tick])

                if not att_id or not vic_id:
                    continue

                # Find damage events from this attacker to this victim before this kill
                mask = (
                    (damages_df[dmg_att] == att_id) &
                    (damages_df[dmg_vic] == vic_id) &
                    (damages_df[dmg_tick] <= kill_tick_val)
                )
                engagement_damages = damages_df[mask].sort_values(dmg_tick)

                if engagement_damages.empty:
                    continue

                # First damage tick in this engagement
                first_dmg_tick = safe_int(engagement_damages.iloc[0][dmg_tick])

                # Calculate TTD (using first damage as proxy for "spotted")
                # This is a simplified approximation
                ttd_ticks = kill_tick_val - first_dmg_tick
                ttd_ms = ttd_ticks * self.MS_PER_TICK

                result = TTDResult(
                    tick_spotted=first_dmg_tick,
                    tick_damage=kill_tick_val,
                    ttd_ticks=ttd_ticks,
                    ttd_ms=ttd_ms,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    weapon=safe_str(kill_row.get(kill_weapon)) if kill_weapon else "",
                    headshot=safe_bool(kill_row.get(kill_hs)) if kill_hs else False,
                    is_prefire=ttd_ms <= self.TTD_MIN_MS,
                )
                self._ttd_results.append(result)
            except Exception as e:
                logger.debug(f"Error processing kill for TTD: {e}")
                continue

        logger.info(f"Computed TTD for {len(self._ttd_results)} engagements")

    def _compute_crosshair_placement(self) -> None:
        """
        Compute crosshair placement error for each kill.

        Angular error = angle between player's view vector and ideal vector to target.
        """
        if self.data.ticks_df is None or self.data.ticks_df.empty:
            logger.warning("No tick data for crosshair placement computation")
            return

        if self.data.kills_df.empty:
            return

        ticks_df = self.data.ticks_df
        kills_df = self.data.kills_df

        # Find required columns
        def find_col(df: pd.DataFrame, options: list[str]) -> Optional[str]:
            for col in options:
                if col in df.columns:
                    return col
            return None

        # Kill columns
        kill_att = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        kill_vic = find_col(kills_df, ["user_steamid", "victim_steamid"])
        kill_tick = find_col(kills_df, ["tick"])

        # Tick columns
        tick_steamid = find_col(ticks_df, ["steamid", "steam_id"])
        tick_x = find_col(ticks_df, ["X", "x"])
        tick_y = find_col(ticks_df, ["Y", "y"])
        tick_z = find_col(ticks_df, ["Z", "z"])
        tick_pitch = find_col(ticks_df, ["pitch"])
        tick_yaw = find_col(ticks_df, ["yaw"])
        tick_col = find_col(ticks_df, ["tick"])

        if not all([kill_att, kill_vic, kill_tick, tick_steamid, tick_x, tick_y, tick_z, tick_pitch, tick_yaw, tick_col]):
            logger.warning("Missing columns for crosshair placement computation")
            return

        for _, kill_row in kills_df.iterrows():
            try:
                att_id = safe_int(kill_row[kill_att])
                vic_id = safe_int(kill_row[kill_vic])
                kill_tick_val = safe_int(kill_row[kill_tick])

                if not att_id or not vic_id:
                    continue

                # Get attacker and victim positions at kill tick (or closest tick before)
                att_ticks = ticks_df[(ticks_df[tick_steamid] == att_id) & (ticks_df[tick_col] <= kill_tick_val)]
                vic_ticks = ticks_df[(ticks_df[tick_steamid] == vic_id) & (ticks_df[tick_col] <= kill_tick_val)]

                if att_ticks.empty or vic_ticks.empty:
                    continue

                # Get closest tick data
                att_state = att_ticks.iloc[-1]
                vic_state = vic_ticks.iloc[-1]

                # Attacker position and angles
                att_pos = np.array([
                    safe_float(att_state[tick_x]),
                    safe_float(att_state[tick_y]),
                    safe_float(att_state[tick_z]) + 64  # Eye height offset
                ])
                att_pitch = safe_float(att_state[tick_pitch])
                att_yaw = safe_float(att_state[tick_yaw])

                # Victim position (head level)
                vic_pos = np.array([
                    safe_float(vic_state[tick_x]),
                    safe_float(vic_state[tick_y]),
                    safe_float(vic_state[tick_z]) + 64  # Head height
                ])

                # Calculate angular error
                angular_error, pitch_error, yaw_error = self._calculate_angular_error(
                    att_pos, att_pitch, att_yaw, vic_pos
                )

                result = CrosshairPlacementResult(
                    tick=kill_tick_val,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    angular_error_deg=angular_error,
                    pitch_error_deg=pitch_error,
                    yaw_error_deg=yaw_error,
                )
                self._cp_results.append(result)
            except Exception as e:
                logger.debug(f"Error processing kill for CP: {e}")
                continue

        logger.info(f"Computed crosshair placement for {len(self._cp_results)} kills")

    def _calculate_angular_error(
        self,
        attacker_pos: np.ndarray,
        pitch_deg: float,
        yaw_deg: float,
        victim_pos: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Calculate angular error between view direction and target.

        Returns: (total_error_deg, pitch_error_deg, yaw_error_deg)
        """
        # Convert angles to radians
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)

        # View vector from Euler angles (Source engine convention)
        # X = forward, Y = right, Z = up
        view_x = math.cos(yaw_rad) * math.cos(pitch_rad)
        view_y = math.sin(yaw_rad) * math.cos(pitch_rad)
        view_z = -math.sin(pitch_rad)  # Negative because positive pitch is down in Source
        view_vec = np.array([view_x, view_y, view_z])

        # Ideal vector from attacker to victim
        ideal_vec = victim_pos - attacker_pos
        distance = np.linalg.norm(ideal_vec)
        if distance < 0.001:
            return 0.0, 0.0, 0.0

        ideal_vec = ideal_vec / distance  # Normalize

        # Total angular error using dot product
        dot = np.clip(np.dot(view_vec, ideal_vec), -1.0, 1.0)
        angular_error = math.degrees(math.acos(dot))

        # Calculate pitch and yaw errors separately
        ideal_pitch = math.degrees(math.asin(-ideal_vec[2]))
        ideal_yaw = math.degrees(math.atan2(ideal_vec[1], ideal_vec[0]))

        pitch_error = pitch_deg - ideal_pitch
        yaw_error = yaw_deg - ideal_yaw

        # Normalize yaw error to [-180, 180]
        while yaw_error > 180:
            yaw_error -= 360
        while yaw_error < -180:
            yaw_error += 360

        return angular_error, pitch_error, yaw_error

    def _build_player_analytics(self) -> dict[int, PlayerAnalytics]:
        """Build comprehensive analytics for each player."""
        analytics: dict[int, PlayerAnalytics] = {}

        for steam_id, stats in self.data.player_stats.items():
            # Gather TTD values for this player
            player_ttd = [
                r.ttd_ms for r in self._ttd_results
                if r.attacker_steamid == steam_id
                and not r.is_prefire
                and r.ttd_ms <= self.TTD_MAX_MS
            ]
            prefire_count = sum(
                1 for r in self._ttd_results
                if r.attacker_steamid == steam_id and r.is_prefire
            )

            # Gather CP values for this player
            player_cp = [
                r.angular_error_deg for r in self._cp_results
                if r.attacker_steamid == steam_id
            ]
            player_pitch_errors = [
                r.pitch_error_deg for r in self._cp_results
                if r.attacker_steamid == steam_id
            ]

            # Calculate TTD stats
            ttd_median = float(np.median(player_ttd)) if player_ttd else None
            ttd_mean = float(np.mean(player_ttd)) if player_ttd else None
            ttd_min = float(np.min(player_ttd)) if player_ttd else None
            ttd_max = float(np.max(player_ttd)) if player_ttd else None
            ttd_std = float(np.std(player_ttd)) if len(player_ttd) > 1 else None

            # Calculate CP stats
            cp_median = float(np.median(player_cp)) if player_cp else None
            cp_mean = float(np.mean(player_cp)) if player_cp else None
            cp_pitch_bias = float(np.mean(player_pitch_errors)) if player_pitch_errors else None

            analytics[steam_id] = PlayerAnalytics(
                steam_id=steam_id,
                name=stats["name"],
                team=stats["team"],
                kills=stats["kills"],
                deaths=stats["deaths"],
                assists=stats["assists"],
                adr=stats["adr"],
                hs_percent=stats["hs_percent"],
                ttd_median_ms=ttd_median,
                ttd_mean_ms=ttd_mean,
                ttd_min_ms=ttd_min,
                ttd_max_ms=ttd_max,
                ttd_std_ms=ttd_std,
                ttd_count=len(player_ttd),
                prefire_count=prefire_count,
                cp_median_error_deg=cp_median,
                cp_mean_error_deg=cp_mean,
                cp_pitch_bias_deg=cp_pitch_bias,
                weapon_kills=stats.get("weapon_kills", {}),
                ttd_values=player_ttd,
                cp_values=player_cp,
            )

        return analytics


def analyze_demo(demo_data: DemoData) -> dict[int, PlayerAnalytics]:
    """Convenience function to analyze a parsed demo."""
    analyzer = DemoAnalyzer(demo_data)
    return analyzer.analyze()
