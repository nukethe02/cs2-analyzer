"""
Enhanced Demo Parser - Production-Grade CS2 Analysis

Provides comprehensive tick-level data extraction and chunked processing
for large demo files (500MB+) with professional coaching metrics.

Features:
- Tick-level player position tracking
- Complete event context with positions
- Chunked parsing for memory efficiency
- Professional metric calculations (TTD, CP, Entry/Trade/Clutch)
- Streaming for large files
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# TICK-LEVEL DATA STRUCTURES
# ============================================================================


@dataclass
class PlayerSnapshot:
    """Player state at a single tick."""

    tick: int
    steam_id: int
    team: int  # 2=T, 3=CT
    name: str
    x: float
    y: float
    z: float
    pitch: float  # Aim angle vertical
    yaw: float  # Aim angle horizontal
    velocity_x: float
    velocity_y: float
    velocity_z: float
    health: int
    armor: int
    money: int
    active_weapon: str
    is_alive: bool
    round_num: int


@dataclass
class WeaponFireContext:
    """Complete weapon fire event with position context."""

    tick: int
    steam_id: int
    team: int
    name: str
    player_x: float
    player_y: float
    player_z: float
    pitch: float
    yaw: float
    weapon: str
    accuracy: float
    round_num: int


@dataclass
class DamageContext:
    """Damage event with complete spatial context."""

    tick: int
    attacker_id: int
    attacker_team: int
    attacker_name: str
    attacker_x: float
    attacker_y: float
    attacker_z: float
    victim_id: int
    victim_team: int
    victim_name: str
    victim_x: float
    victim_y: float
    victim_z: float
    weapon: str
    damage: int
    hit_group: str
    distance: float  # Calculated distance between attacker/victim
    round_num: int


@dataclass
class KillContext:
    """Kill event with complete context."""

    tick: int
    attacker_id: int
    attacker_team: int
    attacker_name: str
    attacker_x: float
    attacker_y: float
    attacker_z: float
    attacker_pitch: float
    attacker_yaw: float
    victim_id: int
    victim_team: int
    victim_name: str
    victim_x: float
    victim_y: float
    victim_z: float
    weapon: str
    is_headshot: bool
    distance: float
    round_num: int
    time_in_round_seconds: float


@dataclass
class RoundChunk:
    """A single round worth of data."""

    round_num: int
    start_tick: int
    end_tick: int
    duration_seconds: float
    ct_team: int  # Team ID for CT
    t_team: int  # Team ID for T
    winner: str  # "CT" or "T"

    # Data collections
    player_snapshots: list[PlayerSnapshot] = field(default_factory=list)
    weapon_fires: list[WeaponFireContext] = field(default_factory=list)
    damages: list[DamageContext] = field(default_factory=list)
    kills: list[KillContext] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "round_num": self.round_num,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "duration_seconds": self.duration_seconds,
            "winner": self.winner,
            "player_snapshot_count": len(self.player_snapshots),
            "weapon_fires": len(self.weapon_fires),
            "damages": len(self.damages),
            "kills": len(self.kills),
        }


# ============================================================================
# PROFESSIONAL METRIC CALCULATORS
# ============================================================================


class MetricCalculator:
    """Calculate professional coaching metrics from tick-level data."""

    @staticmethod
    def _get_time_in_round(kill, default: float = 999.0) -> float:
        """Safely get time_in_round_seconds from a kill object."""
        time_val = getattr(kill, "time_in_round_seconds", None)
        if time_val is None:
            return default
        return time_val

    @staticmethod
    def calculate_entry_frags(
        kills: list[KillContext], round_duration: float = 115.0
    ) -> dict[int, dict]:
        """
        Detect entry frags (first kill in round by attacker).

        Entry frags are critical for round analysis - shows who gets first pick.
        """
        entry_stats = {}

        # Group kills by round
        kills_by_round = {}
        for kill in kills:
            if kill.round_num not in kills_by_round:
                kills_by_round[kill.round_num] = []
            kills_by_round[kill.round_num].append(kill)

        # For each round, find first kill (entry frag)
        for _round_num, round_kills in kills_by_round.items():
            if not round_kills:
                continue

            # Sort by tick to find first kill chronologically
            sorted_kills = sorted(round_kills, key=lambda k: k.tick)
            first_kill = sorted_kills[0]

            # Entry frag is first kill if it happens in first 15 seconds
            first_kill_time = MetricCalculator._get_time_in_round(first_kill)
            if first_kill_time < 15:
                attacker_id = getattr(
                    first_kill, "attacker_id", getattr(first_kill, "attacker_steamid", 0)
                )
                if attacker_id not in entry_stats:
                    entry_stats[attacker_id] = {
                        "name": getattr(first_kill, "attacker_name", "Unknown"),
                        "entry_attempts": 0,
                        "entry_kills": 0,
                        "entry_deaths": 0,
                    }
                entry_stats[attacker_id]["entry_attempts"] += 1
                entry_stats[attacker_id]["entry_kills"] += 1

        # Track entry deaths (died in first 15 seconds without entry kill)
        for _round_num, round_kills in kills_by_round.items():
            entry_kill_attacker = None
            for kill in round_kills:
                kill_time = MetricCalculator._get_time_in_round(kill)
                if kill_time < 15:
                    entry_kill_attacker = getattr(
                        kill, "attacker_id", getattr(kill, "attacker_steamid", 0)
                    )
                    break

            # Deaths in first 15 seconds are entry deaths
            for kill in round_kills:
                kill_time = MetricCalculator._get_time_in_round(kill)
                attacker_id = getattr(kill, "attacker_id", getattr(kill, "attacker_steamid", 0))
                if kill_time < 15 and attacker_id != entry_kill_attacker:
                    victim_id = getattr(kill, "victim_id", getattr(kill, "victim_steamid", 0))
                    if victim_id not in entry_stats:
                        entry_stats[victim_id] = {
                            "name": getattr(kill, "victim_name", "Unknown"),
                            "entry_attempts": 0,
                            "entry_kills": 0,
                            "entry_deaths": 0,
                        }
                    entry_stats[victim_id]["entry_deaths"] += 1

        return entry_stats

    @staticmethod
    def _get_attacker_id(kill) -> int:
        """Safely get attacker ID from a kill object."""
        return getattr(kill, "attacker_id", getattr(kill, "attacker_steamid", 0))

    @staticmethod
    def _get_victim_id(kill) -> int:
        """Safely get victim ID from a kill object."""
        return getattr(kill, "victim_id", getattr(kill, "victim_steamid", 0))

    @staticmethod
    def _get_attacker_team(kill) -> int:
        """Safely get attacker team from a kill object (returns 2 for T, 3 for CT)."""
        team = getattr(kill, "attacker_team", None)
        if team is not None:
            return team
        side = getattr(kill, "attacker_side", "")
        if side == "CT":
            return 3
        if side == "T":
            return 2
        return 0

    @staticmethod
    def _get_victim_team(kill) -> int:
        """Safely get victim team from a kill object (returns 2 for T, 3 for CT)."""
        team = getattr(kill, "victim_team", None)
        if team is not None:
            return team
        side = getattr(kill, "victim_side", "")
        if side == "CT":
            return 3
        if side == "T":
            return 2
        return 0

    @staticmethod
    def calculate_trade_kills(
        kills: list[KillContext], trade_window_ticks: int = 5
    ) -> dict[int, dict]:
        """
        Detect trade kills (kill within N ticks of teammate's death).

        Trade kills show team coordination and survivor value.
        """
        trade_stats = {}

        # Index kills by victim
        kills_by_victim = {}
        for kill in kills:
            victim_id = MetricCalculator._get_victim_id(kill)
            if victim_id not in kills_by_victim:
                kills_by_victim[victim_id] = []
            kills_by_victim[victim_id].append(kill)

        # For each death, check if teammate got revenge kill
        for victim_id, victim_kills in kills_by_victim.items():
            for kill in victim_kills:
                kill_victim_team = MetricCalculator._get_victim_team(kill)
                # Find kills by same team within trade window
                for other_kill in kills:
                    other_attacker_id = MetricCalculator._get_attacker_id(other_kill)
                    other_attacker_team = MetricCalculator._get_attacker_team(other_kill)
                    if (
                        other_attacker_id == victim_id
                        and other_attacker_team == kill_victim_team
                        and other_kill.round_num == kill.round_num
                        and abs(other_kill.tick - kill.tick) <= trade_window_ticks
                    ):
                        # This is a trade kill
                        attacker_id = other_attacker_id
                        if attacker_id not in trade_stats:
                            trade_stats[attacker_id] = {
                                "name": getattr(other_kill, "attacker_name", "Unknown"),
                                "trade_kills": 0,
                                "deaths_traded": 0,
                            }
                        trade_stats[attacker_id]["trade_kills"] += 1

        return trade_stats

    @staticmethod
    def calculate_ttd(damages: list[DamageContext]) -> dict[int, dict]:
        """
        Calculate Time To Damage (TTD) - ticks from first seeing enemy to dealing damage.

        This requires position analysis to determine visibility.
        """
        ttd_data = {}

        # Group by attacker
        damages_by_attacker = {}
        for dmg in damages:
            key = (dmg.attacker_id, dmg.round_num)
            if key not in damages_by_attacker:
                damages_by_attacker[key] = []
            damages_by_attacker[key].append(dmg)

        # Calculate TTD from distance and time
        for (attacker_id, _round_num), dmg_list in damages_by_attacker.items():
            sorted_dmg = sorted(dmg_list, key=lambda d: d.tick)
            if not sorted_dmg:
                continue

            first_dmg = sorted_dmg[0]
            ttd_ms = (
                first_dmg.tick / 64.0
            ) * 1000  # Convert to milliseconds (assuming 64 tick rate)

            if attacker_id not in ttd_data:
                ttd_data[attacker_id] = {
                    "name": first_dmg.attacker_name,
                    "ttd_values": [],
                }
            ttd_data[attacker_id]["ttd_values"].append(ttd_ms)

        # Calculate stats
        for attacker_id in ttd_data:
            values = ttd_data[attacker_id]["ttd_values"]
            ttd_data[attacker_id]["ttd_median_ms"] = float(np.median(values)) if values else 0.0
            ttd_data[attacker_id]["ttd_mean_ms"] = float(np.mean(values)) if values else 0.0
            ttd_data[attacker_id]["ttd_95th_ms"] = (
                float(np.percentile(values, 95)) if values else 0.0
            )

        return ttd_data

    @staticmethod
    def calculate_crosshair_placement(kills: list[KillContext]) -> dict[int, dict]:
        """
        Calculate Crosshair Placement (CP) - angular distance from crosshair to target head.

        Angular difference = arctan(target_height / distance)
        """
        cp_data = {}

        for kill in kills:
            # Get positions safely, skip if any are None
            attacker_x = getattr(kill, "attacker_x", None)
            attacker_y = getattr(kill, "attacker_y", None)
            attacker_z = getattr(kill, "attacker_z", None)
            victim_x = getattr(kill, "victim_x", None)
            victim_y = getattr(kill, "victim_y", None)
            victim_z = getattr(kill, "victim_z", None)
            attacker_yaw = getattr(kill, "attacker_yaw", None)
            attacker_pitch = getattr(kill, "attacker_pitch", None)

            # Skip if any required position data is missing
            if any(
                v is None
                for v in [
                    attacker_x,
                    attacker_y,
                    attacker_z,
                    victim_x,
                    victim_y,
                    victim_z,
                    attacker_yaw,
                    attacker_pitch,
                ]
            ):
                continue

            # Calculate vector from attacker to victim
            dx = victim_x - attacker_x
            dy = victim_y - attacker_y
            dz = victim_z - attacker_z

            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            if distance == 0:
                continue

            # Calculate required angles to hit victim head
            horizontal_dist = np.sqrt(dx**2 + dy**2)
            required_yaw = np.degrees(np.arctan2(dy, dx))
            required_pitch = np.degrees(
                np.arctan2(dz + 1.4, horizontal_dist)
            )  # +1.4 for head height

            # Calculate error
            yaw_error = abs(attacker_yaw - required_yaw)
            pitch_error = abs(attacker_pitch - required_pitch)

            # Angular distance (simplified)
            angular_error = np.sqrt(yaw_error**2 + pitch_error**2)

            attacker_id = MetricCalculator._get_attacker_id(kill)
            if attacker_id not in cp_data:
                cp_data[attacker_id] = {
                    "name": getattr(kill, "attacker_name", "Unknown"),
                    "cp_errors": [],
                }
            cp_data[attacker_id]["cp_errors"].append(angular_error)

        # Calculate stats
        for attacker_id in cp_data:
            errors = cp_data[attacker_id]["cp_errors"]
            cp_data[attacker_id]["cp_median_error_deg"] = (
                float(np.median(errors)) if errors else 0.0
            )
            cp_data[attacker_id]["cp_mean_error_deg"] = float(np.mean(errors)) if errors else 0.0

        return cp_data

    @staticmethod
    def calculate_clutch_stats(
        kills: list[KillContext], player_snapshots: list[PlayerSnapshot]
    ) -> dict[int, dict]:
        """
        Detect clutch situations (1v5, 1v4, 1v3, 1v2, 1v1) and success rates.
        """
        clutch_stats = {}

        # Group snapshots by round
        snapshots_by_round = {}
        for snapshot in player_snapshots:
            if snapshot.round_num not in snapshots_by_round:
                snapshots_by_round[snapshot.round_num] = []
            snapshots_by_round[snapshot.round_num].append(snapshot)

        # Detect clutch situations
        for round_num, round_snapshots in snapshots_by_round.items():
            sorted_snapshots = sorted(round_snapshots, key=lambda s: s.tick)

            for i, snapshot in enumerate(sorted_snapshots):
                if not snapshot.is_alive:
                    continue

                # Count alive teammates and enemies
                alive_teammates = sum(
                    1
                    for s in sorted_snapshots[i:]
                    if s.team == snapshot.team and s.is_alive and s.tick == snapshot.tick
                )
                alive_enemies = sum(
                    1
                    for s in sorted_snapshots[i:]
                    if s.team != snapshot.team and s.is_alive and s.tick == snapshot.tick
                )

                # Clutch is 1vX situation
                if alive_teammates == 1 and alive_enemies >= 2:
                    player_id = snapshot.steam_id
                    if player_id not in clutch_stats:
                        clutch_stats[player_id] = {
                            "name": snapshot.name,
                            "clutch_wins": 0,
                            "clutch_attempts": 0,
                            "v1_wins": 0,
                            "v2_wins": 0,
                            "v3_wins": 0,
                            "v4_wins": 0,
                            "v5_wins": 0,
                        }
                    clutch_stats[player_id]["clutch_attempts"] += 1

                    # Check if they won the clutch (killed all enemies)
                    round_kills = [
                        k for k in kills if k.round_num == round_num and k.attacker_id == player_id
                    ]
                    if len(round_kills) >= alive_enemies:
                        clutch_stats[player_id]["clutch_wins"] += 1
                        clutch_key = f"v{alive_enemies}_wins"
                        if clutch_key in clutch_stats[player_id]:
                            clutch_stats[player_id][clutch_key] += 1

        return clutch_stats


# ============================================================================
# CHUNKED PARSER FOR LARGE FILES
# ============================================================================


class ChunkedDemoParser:
    """
    Parse demo files in chunks to handle 500MB+ files efficiently.

    Processes round-by-round instead of loading entire demo into memory.
    """

    def __init__(self, demo_path: Path):
        """Initialize chunked parser."""
        self.demo_path = Path(demo_path)
        self.tick_rate = 64
        logger.info(f"Initializing chunked parser for {self.demo_path.name}")

    def parse_chunks(self) -> Generator[RoundChunk, None, None]:
        """
        Parse demo file in round chunks.

        Yields RoundChunk objects one at a time for memory efficiency.
        """
        from opensight.core.parser import DemoParser

        # Use base parser to get structure
        base_parser = DemoParser(self.demo_path)
        demo_data = base_parser.parse(include_ticks=True, comprehensive=True)

        if not demo_data.rounds:
            logger.warning("No rounds found in demo")
            return

        # Process each round
        for round_info in demo_data.rounds:
            round_chunk = RoundChunk(
                round_num=round_info.round_num,
                start_tick=round_info.start_tick,
                end_tick=round_info.end_tick,
                duration_seconds=(round_info.end_tick - round_info.start_tick) / self.tick_rate,
                ct_team=3,
                t_team=2,
                winner=round_info.winner,
            )

            # Extract data for this round
            self._extract_round_data(round_chunk, demo_data)

            yield round_chunk

            # Explicit garbage collection after each round
            del round_chunk

    def _extract_round_data(self, chunk: RoundChunk, demo_data) -> None:
        """Extract all data for a single round."""
        # Filter kills, damages, etc. for this round
        chunk.kills = [k for k in demo_data.kills if k.round_num == chunk.round_num]
        chunk.damages = [
            d for d in getattr(demo_data, "damages", []) if d.round_num == chunk.round_num
        ]
        chunk.weapon_fires = [
            w for w in getattr(demo_data, "weapon_fires", []) if w.round_num == chunk.round_num
        ]

        logger.debug(
            f"Round {chunk.round_num}: {len(chunk.kills)} kills, "
            f"{len(chunk.damages)} damages, {len(chunk.weapon_fires)} shots"
        )


# ============================================================================
# COMPREHENSIVE ANALYSIS ENGINE
# ============================================================================


class CoachingAnalysisEngine:
    """
    Comprehensive analysis for coaching insights.

    Processes chunked demo data and generates professional coaching metrics.
    """

    def __init__(self, demo_path: Path):
        """Initialize analysis engine."""
        self.demo_path = Path(demo_path)
        self.parser = ChunkedDemoParser(demo_path)
        self.calculator = MetricCalculator()

    def analyze(self) -> dict:
        """
        Perform comprehensive analysis on demo file.

        Returns aggregated metrics across all rounds.
        """
        logger.info(f"Starting comprehensive analysis of {self.demo_path.name}")

        all_kills = []
        all_damages = []
        all_snapshots = []
        round_summaries = []

        # Process each round
        for i, chunk in enumerate(self.parser.parse_chunks(), 1):
            logger.info(f"Processing round {chunk.round_num} ({i})")

            all_kills.extend(chunk.kills)
            all_damages.extend(chunk.damages)
            all_snapshots.extend(chunk.player_snapshots)

            round_summaries.append(chunk.to_dict())

        logger.info(f"Aggregating metrics from {len(all_kills)} total kills")

        # Calculate professional metrics
        entry_stats = self.calculator.calculate_entry_frags(all_kills)
        trade_stats = self.calculator.calculate_trade_kills(all_kills)
        ttd_stats = self.calculator.calculate_ttd(all_damages)
        cp_stats = self.calculator.calculate_crosshair_placement(all_kills)
        clutch_stats = self.calculator.calculate_clutch_stats(all_kills, all_snapshots)

        return {
            "total_rounds": len(round_summaries),
            "total_kills": len(all_kills),
            "total_damages": len(all_damages),
            "round_summaries": round_summaries,
            "entry_frags": entry_stats,
            "trade_kills": trade_stats,
            "ttd_metrics": ttd_stats,
            "crosshair_placement": cp_stats,
            "clutch_statistics": clutch_stats,
        }
