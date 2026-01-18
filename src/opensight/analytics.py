"""
Advanced Analytics for CS2 Demo Analysis

Implements Time to Damage (TTD) and Crosshair Placement (CP) metrics.
Uses awpy's clean data structures for reliable analysis.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import math

import numpy as np
import pandas as pd

from opensight.parser import DemoData, KillEvent, DamageEvent, safe_int, safe_str, safe_float

logger = logging.getLogger(__name__)


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
    round_num: int = 0


@dataclass
class CrosshairPlacementResult:
    """Crosshair placement analysis for a kill."""
    tick: int
    attacker_steamid: int
    victim_steamid: int
    angular_error_deg: float
    pitch_error_deg: float
    yaw_error_deg: float
    round_num: int = 0


@dataclass
class PlayerAnalytics:
    """Complete analytics for a player."""
    steam_id: int
    name: str
    team: str  # "CT", "T", or "Unknown"

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
    cp_pitch_bias_deg: Optional[float]

    # Weapon breakdown
    weapon_kills: dict[str, int]

    # Raw data for histograms
    ttd_values: list[float]
    cp_values: list[float]


class DemoAnalyzer:
    """Analyzer for computing advanced metrics from parsed demo data."""

    TICK_RATE = 64
    MS_PER_TICK = 1000 / TICK_RATE  # ~15.625ms

    # TTD filtering thresholds
    TTD_MIN_MS = 0
    TTD_MAX_MS = 1500

    def __init__(self, demo_data: DemoData):
        self.data = demo_data
        self._ttd_results: list[TTDResult] = []
        self._cp_results: list[CrosshairPlacementResult] = []

    def analyze(self) -> dict[int, PlayerAnalytics]:
        """Run full analysis and return per-player analytics."""
        logger.info("Starting advanced analysis...")

        # Compute TTD for all kills
        self._compute_ttd()

        # Compute crosshair placement (uses position data from events or ticks)
        self._compute_crosshair_placement()

        # Build per-player analytics
        player_analytics = self._build_player_analytics()

        logger.info(f"Analysis complete. {len(player_analytics)} players analyzed.")
        return player_analytics

    def _compute_ttd(self) -> None:
        """Compute Time to Damage for each kill."""
        if self.data.damages_df.empty or self.data.kills_df.empty:
            logger.warning("No damage or kill data for TTD computation")
            return

        kills_df = self.data.kills_df
        damages_df = self.data.damages_df

        # awpy uses consistent column names
        kill_att = "attacker_steamid" if "attacker_steamid" in kills_df.columns else None
        kill_vic = "victim_steamid" if "victim_steamid" in kills_df.columns else None
        kill_tick = "tick" if "tick" in kills_df.columns else None
        kill_weapon = "weapon" if "weapon" in kills_df.columns else None
        kill_hs = "headshot" if "headshot" in kills_df.columns else None
        kill_round = "round_num" if "round_num" in kills_df.columns else None

        dmg_att = "attacker_steamid" if "attacker_steamid" in damages_df.columns else None
        dmg_vic = "victim_steamid" if "victim_steamid" in damages_df.columns else None
        dmg_tick = "tick" if "tick" in damages_df.columns else None

        if not all([kill_att, kill_vic, kill_tick, dmg_att, dmg_vic, dmg_tick]):
            logger.warning("Missing columns for TTD computation")
            return

        for _, kill_row in kills_df.iterrows():
            try:
                att_id = safe_int(kill_row[kill_att])
                vic_id = safe_int(kill_row[kill_vic])
                kill_tick_val = safe_int(kill_row[kill_tick])
                round_num = safe_int(kill_row.get(kill_round, 0)) if kill_round else 0

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

                first_dmg_tick = safe_int(engagement_damages.iloc[0][dmg_tick])
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
                    headshot=bool(kill_row.get(kill_hs, False)) if kill_hs else False,
                    is_prefire=ttd_ms <= self.TTD_MIN_MS,
                    round_num=round_num,
                )
                self._ttd_results.append(result)
            except Exception as e:
                logger.debug(f"Error processing kill for TTD: {e}")
                continue

        logger.info(f"Computed TTD for {len(self._ttd_results)} engagements")

    def _compute_crosshair_placement(self) -> None:
        """Compute crosshair placement error for each kill."""
        if self.data.kills_df.empty:
            return

        kills_df = self.data.kills_df

        # Check for position data in kills_df (awpy may include it)
        has_positions = all(col in kills_df.columns for col in [
            "attacker_X", "attacker_Y", "attacker_Z",
            "victim_X", "victim_Y", "victim_Z"
        ])

        # Check for angle data
        has_angles = all(col in kills_df.columns for col in ["attacker_pitch", "attacker_yaw"])

        if has_positions and has_angles:
            self._compute_cp_from_events()
        elif self.data.ticks_df is not None and not self.data.ticks_df.empty:
            self._compute_cp_from_ticks()
        else:
            logger.info("No position/angle data available for CP computation")

    def _compute_cp_from_events(self) -> None:
        """Compute CP from position data embedded in kill events."""
        kills_df = self.data.kills_df
        logger.info("Computing CP from event-embedded positions")

        for _, row in kills_df.iterrows():
            try:
                att_id = safe_int(row.get("attacker_steamid"))
                vic_id = safe_int(row.get("victim_steamid"))
                tick = safe_int(row.get("tick"))
                round_num = safe_int(row.get("round_num", 0))

                if not att_id or not vic_id:
                    continue

                att_pos = np.array([
                    safe_float(row.get("attacker_X")),
                    safe_float(row.get("attacker_Y")),
                    safe_float(row.get("attacker_Z")) + 64
                ])
                att_pitch = safe_float(row.get("attacker_pitch"))
                att_yaw = safe_float(row.get("attacker_yaw"))

                vic_pos = np.array([
                    safe_float(row.get("victim_X")),
                    safe_float(row.get("victim_Y")),
                    safe_float(row.get("victim_Z")) + 64
                ])

                if np.allclose(att_pos[:2], 0) or np.allclose(vic_pos[:2], 0):
                    continue

                angular_error, pitch_error, yaw_error = self._calculate_angular_error(
                    att_pos, att_pitch, att_yaw, vic_pos
                )

                result = CrosshairPlacementResult(
                    tick=tick,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    angular_error_deg=angular_error,
                    pitch_error_deg=pitch_error,
                    yaw_error_deg=yaw_error,
                    round_num=round_num,
                )
                self._cp_results.append(result)
            except Exception as e:
                logger.debug(f"Error processing kill for CP: {e}")
                continue

        logger.info(f"Computed CP for {len(self._cp_results)} kills")

    def _compute_cp_from_ticks(self) -> None:
        """Compute CP from tick-level data (fallback)."""
        ticks_df = self.data.ticks_df
        kills_df = self.data.kills_df
        logger.info("Computing CP from tick data (slower)")

        # Find columns
        steamid_col = "steamid" if "steamid" in ticks_df.columns else None
        x_col = "X" if "X" in ticks_df.columns else None
        y_col = "Y" if "Y" in ticks_df.columns else None
        z_col = "Z" if "Z" in ticks_df.columns else None
        pitch_col = "pitch" if "pitch" in ticks_df.columns else None
        yaw_col = "yaw" if "yaw" in ticks_df.columns else None
        tick_col = "tick" if "tick" in ticks_df.columns else None

        if not all([steamid_col, x_col, y_col, z_col, pitch_col, yaw_col, tick_col]):
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

                att_ticks = ticks_df[(ticks_df[steamid_col] == att_id) & (ticks_df[tick_col] <= kill_tick)]
                vic_ticks = ticks_df[(ticks_df[steamid_col] == vic_id) & (ticks_df[tick_col] <= kill_tick)]

                if att_ticks.empty or vic_ticks.empty:
                    continue

                att_state = att_ticks.iloc[-1]
                vic_state = vic_ticks.iloc[-1]

                att_pos = np.array([
                    safe_float(att_state[x_col]),
                    safe_float(att_state[y_col]),
                    safe_float(att_state[z_col]) + 64
                ])
                att_pitch = safe_float(att_state[pitch_col])
                att_yaw = safe_float(att_state[yaw_col])

                vic_pos = np.array([
                    safe_float(vic_state[x_col]),
                    safe_float(vic_state[y_col]),
                    safe_float(vic_state[z_col]) + 64
                ])

                angular_error, pitch_error, yaw_error = self._calculate_angular_error(
                    att_pos, att_pitch, att_yaw, vic_pos
                )

                result = CrosshairPlacementResult(
                    tick=kill_tick,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    angular_error_deg=angular_error,
                    pitch_error_deg=pitch_error,
                    yaw_error_deg=yaw_error,
                    round_num=round_num,
                )
                self._cp_results.append(result)
            except Exception as e:
                logger.debug(f"Error in tick-based CP: {e}")
                continue

        logger.info(f"Computed CP for {len(self._cp_results)} kills (tick-based)")

    def _calculate_angular_error(
        self,
        attacker_pos: np.ndarray,
        pitch_deg: float,
        yaw_deg: float,
        victim_pos: np.ndarray
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

    def _build_player_analytics(self) -> dict[int, PlayerAnalytics]:
        """Build comprehensive analytics for each player."""
        analytics: dict[int, PlayerAnalytics] = {}

        for steam_id, stats in self.data.player_stats.items():
            # TTD values for this player
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

            # CP values
            player_cp = [r.angular_error_deg for r in self._cp_results if r.attacker_steamid == steam_id]
            player_pitch_errors = [r.pitch_error_deg for r in self._cp_results if r.attacker_steamid == steam_id]

            # TTD stats
            ttd_median = float(np.median(player_ttd)) if player_ttd else None
            ttd_mean = float(np.mean(player_ttd)) if player_ttd else None
            ttd_min = float(np.min(player_ttd)) if player_ttd else None
            ttd_max = float(np.max(player_ttd)) if player_ttd else None
            ttd_std = float(np.std(player_ttd)) if len(player_ttd) > 1 else None

            # CP stats
            cp_median = float(np.median(player_cp)) if player_cp else None
            cp_mean = float(np.mean(player_cp)) if player_cp else None
            cp_pitch_bias = float(np.mean(player_pitch_errors)) if player_pitch_errors else None

            analytics[steam_id] = PlayerAnalytics(
                steam_id=steam_id,
                name=stats["name"],
                team=stats["team"],
                kills=stats["kills"],
                deaths=stats["deaths"],
                assists=stats.get("assists", 0),
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
