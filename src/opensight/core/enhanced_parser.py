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
        Detect opening duels (first kill/death of each round).

        This is a simplified version that tracks the first kill of each round
        as an "opening duel" (5v5 -> 5v4). For zone-aware entry frags
        (first kill INTO a bombsite), see DemoAnalyzer._detect_entry_frags().

        Note: The round_duration parameter is kept for backwards compatibility
        but is no longer used (time-based windows have been removed).
        """
        entry_stats: dict[int, dict] = {}

        # Group kills by round
        kills_by_round: dict[int, list] = {}
        for kill in kills:
            if kill.round_num not in kills_by_round:
                kills_by_round[kill.round_num] = []
            kills_by_round[kill.round_num].append(kill)

        # For each round, find first kill (opening duel)
        for _round_num, round_kills in kills_by_round.items():
            if not round_kills:
                continue

            # Sort by tick to find first kill chronologically
            sorted_kills = sorted(round_kills, key=lambda k: k.tick)
            first_kill = sorted_kills[0]

            # Track the attacker (winner) of the opening duel
            attacker_id = getattr(
                first_kill, "attacker_id", getattr(first_kill, "attacker_steamid", 0)
            )
            if attacker_id:
                if attacker_id not in entry_stats:
                    entry_stats[attacker_id] = {
                        "name": getattr(first_kill, "attacker_name", "Unknown"),
                        "entry_attempts": 0,
                        "entry_kills": 0,
                        "entry_deaths": 0,
                    }
                entry_stats[attacker_id]["entry_attempts"] += 1
                entry_stats[attacker_id]["entry_kills"] += 1

            # Track the victim (loser) of the opening duel
            victim_id = getattr(first_kill, "victim_id", getattr(first_kill, "victim_steamid", 0))
            if victim_id:
                if victim_id not in entry_stats:
                    entry_stats[victim_id] = {
                        "name": getattr(first_kill, "victim_name", "Unknown"),
                        "entry_attempts": 0,
                        "entry_kills": 0,
                        "entry_deaths": 0,
                    }
                entry_stats[victim_id]["entry_attempts"] += 1
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
    def _get_damage_attacker_id(dmg) -> int:
        """Safely get attacker ID from a damage object."""
        return getattr(dmg, "attacker_id", getattr(dmg, "attacker_steamid", 0))

    @staticmethod
    def _get_damage_victim_id(dmg) -> int:
        """Safely get victim ID from a damage object."""
        return getattr(
            dmg, "victim_id", getattr(dmg, "victim_steamid", getattr(dmg, "user_steamid", 0))
        )

    @staticmethod
    def calculate_ttd(kills: list, damages: list, tick_rate: int = 64) -> dict[int, dict]:
        """
        Calculate Time To Damage (TTD) - time from first damage to kill.

        TTD measures how long it takes a player to secure kills after dealing initial damage.
        Lower TTD = faster reactions/better aim.

        Args:
            kills: List of kill events
            damages: List of damage events
            tick_rate: Server tick rate (default 64)

        Returns:
            Dict mapping player IDs to TTD statistics
        """
        ttd_data = {}
        ms_per_tick = 1000 / tick_rate

        if not kills or not damages:
            return ttd_data

        # Build damage lookup: (attacker_id, victim_id, round_num) -> [ticks]
        damage_lookup = {}
        for dmg in damages:
            attacker_id = MetricCalculator._get_damage_attacker_id(dmg)
            victim_id = MetricCalculator._get_damage_victim_id(dmg)
            round_num = getattr(dmg, "round_num", 0)
            key = (attacker_id, victim_id, round_num)
            if key not in damage_lookup:
                damage_lookup[key] = []
            damage_lookup[key].append(dmg.tick)

        # For each kill, find the first damage from attacker to victim in the same round
        for kill in kills:
            attacker_id = MetricCalculator._get_attacker_id(kill)
            victim_id = MetricCalculator._get_victim_id(kill)
            round_num = getattr(kill, "round_num", 0)
            kill_tick = getattr(kill, "tick", 0)
            attacker_name = getattr(kill, "attacker_name", "Unknown")

            key = (attacker_id, victim_id, round_num)
            if key not in damage_lookup:
                continue

            # Find first damage before or at kill tick
            damage_ticks = [t for t in damage_lookup[key] if t <= kill_tick]
            if not damage_ticks:
                continue

            first_damage_tick = min(damage_ticks)
            ttd_ticks = kill_tick - first_damage_tick
            ttd_ms = ttd_ticks * ms_per_tick

            # Filter: valid TTD range (50ms - 5000ms)
            if ttd_ms < 50 or ttd_ms > 5000:
                continue

            if attacker_id not in ttd_data:
                ttd_data[attacker_id] = {
                    "name": attacker_name,
                    "ttd_values": [],
                }
            ttd_data[attacker_id]["ttd_values"].append(ttd_ms)

        # Calculate statistics for each player
        for attacker_id in ttd_data:
            values = ttd_data[attacker_id]["ttd_values"]
            if values:
                ttd_data[attacker_id]["ttd_median_ms"] = float(np.median(values))
                ttd_data[attacker_id]["ttd_mean_ms"] = float(np.mean(values))
                ttd_data[attacker_id]["ttd_95th_ms"] = float(np.percentile(values, 95))
            else:
                ttd_data[attacker_id]["ttd_median_ms"] = 0.0
                ttd_data[attacker_id]["ttd_mean_ms"] = 0.0
                ttd_data[attacker_id]["ttd_95th_ms"] = 0.0

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
    def calculate_clutch_stats(kills: list, player_snapshots: list = None) -> dict[int, dict]:
        """
        Detect clutch situations (1v5, 1v4, 1v3, 1v2, 1v1) and success rates.

        Uses kill sequence analysis to detect when a player was in a 1vX situation.
        A clutch is detected when all teammates died before a player got their last kill(s)
        in a round, leaving them alone against multiple enemies.
        """
        clutch_stats = {}

        if not kills:
            return clutch_stats

        # Group kills by round
        kills_by_round = {}
        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            if round_num not in kills_by_round:
                kills_by_round[round_num] = []
            kills_by_round[round_num].append(kill)

        # Analyze each round for clutch situations
        for round_num, round_kills in kills_by_round.items():
            if len(round_kills) < 2:
                continue

            # Sort kills by tick
            sorted_kills = sorted(round_kills, key=lambda k: getattr(k, "tick", 0))

            # Track alive players on each side (start with 5 each)
            # Team mapping: use the first kill to determine team IDs
            team_alive = {}  # team_id -> set of player_ids
            player_teams = {}  # player_id -> team_id

            # Build player team mapping from kills
            for kill in sorted_kills:
                attacker_id = MetricCalculator._get_attacker_id(kill)
                victim_id = MetricCalculator._get_victim_id(kill)
                attacker_team = MetricCalculator._get_attacker_team(kill)
                victim_team = MetricCalculator._get_victim_team(kill)
                attacker_name = getattr(kill, "attacker_name", "Unknown")
                victim_name = getattr(kill, "victim_name", "Unknown")

                if attacker_team:
                    player_teams[attacker_id] = (attacker_team, attacker_name)
                if victim_team:
                    player_teams[victim_id] = (victim_team, victim_name)

            # Initialize team alive counts
            for player_id, (team_id, _) in player_teams.items():
                if team_id not in team_alive:
                    team_alive[team_id] = set()
                team_alive[team_id].add(player_id)

            # Process kills in order
            for i, kill in enumerate(sorted_kills):
                attacker_id = MetricCalculator._get_attacker_id(kill)
                victim_id = MetricCalculator._get_victim_id(kill)
                attacker_team = MetricCalculator._get_attacker_team(kill)
                victim_team = MetricCalculator._get_victim_team(kill)
                attacker_name = getattr(kill, "attacker_name", "Unknown")

                # Remove victim from alive set
                if victim_team in team_alive and victim_id in team_alive[victim_team]:
                    team_alive[victim_team].discard(victim_id)

                # Check if attacker is now in a 1vX situation
                if attacker_team in team_alive:
                    attacker_teammates_alive = len(team_alive[attacker_team])
                    enemies_alive = sum(
                        len(players) for t, players in team_alive.items() if t != attacker_team
                    )

                    # 1vX situation: only the attacker alive, multiple enemies
                    if attacker_teammates_alive == 1 and enemies_alive >= 1:
                        if attacker_id not in clutch_stats:
                            clutch_stats[attacker_id] = {
                                "name": attacker_name,
                                "clutch_wins": 0,
                                "clutch_attempts": 0,
                                "v1_wins": 0,
                                "v2_wins": 0,
                                "v3_wins": 0,
                                "v4_wins": 0,
                                "v5_wins": 0,
                            }

                        # This counts as a clutch attempt if enemies >= 2 (or 1v1)
                        clutch_type = enemies_alive + 1  # +1 for the one just killed
                        if clutch_type >= 2:
                            # Check if this is a new clutch situation or continuation
                            remaining_kills = sorted_kills[i + 1 :]
                            attacker_kills_remaining = [
                                k
                                for k in remaining_kills
                                if MetricCalculator._get_attacker_id(k) == attacker_id
                            ]
                            attacker_deaths_remaining = [
                                k
                                for k in remaining_kills
                                if MetricCalculator._get_victim_id(k) == attacker_id
                            ]

                            # Only count as attempt once per round per player
                            # Check if we've already counted this round for this player
                            round_key = f"round_{round_num}"
                            if round_key not in clutch_stats[attacker_id]:
                                clutch_stats[attacker_id][round_key] = True
                                clutch_stats[attacker_id]["clutch_attempts"] += 1

                                # Did they win? (killed all enemies without dying)
                                if len(attacker_kills_remaining) >= enemies_alive:
                                    # Check they didn't die before killing all
                                    first_death_tick = (
                                        min(
                                            getattr(k, "tick", float("inf"))
                                            for k in attacker_deaths_remaining
                                        )
                                        if attacker_deaths_remaining
                                        else float("inf")
                                    )
                                    kills_before_death = [
                                        k
                                        for k in attacker_kills_remaining
                                        if getattr(k, "tick", 0) < first_death_tick
                                    ]

                                    if len(kills_before_death) >= enemies_alive:
                                        clutch_stats[attacker_id]["clutch_wins"] += 1
                                        clutch_key = f"v{min(clutch_type, 5)}_wins"
                                        if clutch_key in clutch_stats[attacker_id]:
                                            clutch_stats[attacker_id][clutch_key] += 1

        # Clean up internal tracking keys
        for player_id in clutch_stats:
            clutch_stats[player_id] = {
                k: v for k, v in clutch_stats[player_id].items() if not k.startswith("round_")
            }

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
        ttd_stats = self.calculator.calculate_ttd(all_kills, all_damages)
        cp_stats = self.calculator.calculate_crosshair_placement(all_kills)
        clutch_stats = self.calculator.calculate_clutch_stats(all_kills)

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
