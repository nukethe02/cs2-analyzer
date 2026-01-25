"""
Player Performance Metrics Module

Provides detailed player performance analysis including:
- KDA breakdown with context
- Kill distance analysis (average, close/mid/long range distribution)
- Nemesis tracking (deaths by specific enemy players)
- Time alive and survival analysis
- Engagement pattern analysis

These metrics complement the core analytics module with deeper insights.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from opensight.parser import MatchData

# =============================================================================
# Distance Classifications (CS2 units, approximately)
# =============================================================================

CLOSE_RANGE_MAX = 500  # ~5 meters - shotgun/SMG range
MID_RANGE_MAX = 1500  # ~15 meters - rifle effective range
# Beyond MID_RANGE_MAX is long range (AWP territory)


@dataclass
class KillDistanceStats:
    """Statistics about kill distances for a player."""

    total_kills: int = 0
    total_distance: float = 0.0
    close_range_kills: int = 0  # < 500 units
    mid_range_kills: int = 0  # 500-1500 units
    long_range_kills: int = 0  # > 1500 units
    distances: list[float] = field(default_factory=list)

    @property
    def average_kill_distance(self) -> float:
        """Average distance of all kills in game units."""
        if self.total_kills == 0:
            return 0.0
        return round(self.total_distance / self.total_kills, 1)

    @property
    def average_kill_distance_meters(self) -> float:
        """Average distance in approximate real-world meters."""
        # CS2 units to meters: roughly 1 unit = 0.01905 meters (1 inch)
        return round(self.average_kill_distance * 0.01905, 1)

    @property
    def close_range_percentage(self) -> float:
        """Percentage of kills at close range."""
        if self.total_kills == 0:
            return 0.0
        return round(self.close_range_kills / self.total_kills * 100, 1)

    @property
    def mid_range_percentage(self) -> float:
        """Percentage of kills at mid range."""
        if self.total_kills == 0:
            return 0.0
        return round(self.mid_range_kills / self.total_kills * 100, 1)

    @property
    def long_range_percentage(self) -> float:
        """Percentage of kills at long range."""
        if self.total_kills == 0:
            return 0.0
        return round(self.long_range_kills / self.total_kills * 100, 1)

    @property
    def median_distance(self) -> float:
        """Median kill distance."""
        if not self.distances:
            return 0.0
        sorted_distances = sorted(self.distances)
        mid = len(sorted_distances) // 2
        if len(sorted_distances) % 2 == 0:
            return (sorted_distances[mid - 1] + sorted_distances[mid]) / 2
        return sorted_distances[mid]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_kills": self.total_kills,
            "average_distance_units": self.average_kill_distance,
            "average_distance_meters": self.average_kill_distance_meters,
            "median_distance_units": round(self.median_distance, 1),
            "close_range_kills": self.close_range_kills,
            "mid_range_kills": self.mid_range_kills,
            "long_range_kills": self.long_range_kills,
            "close_range_pct": self.close_range_percentage,
            "mid_range_pct": self.mid_range_percentage,
            "long_range_pct": self.long_range_percentage,
        }


@dataclass
class NemesisStats:
    """Track which enemies killed a player most often."""

    deaths_by_enemy: dict[str, int] = field(default_factory=dict)
    deaths_by_enemy_steamid: dict[int, int] = field(default_factory=dict)
    kills_against_enemy: dict[str, int] = field(default_factory=dict)

    @property
    def nemesis(self) -> tuple[str, int] | None:
        """The enemy who killed this player most often (name, count)."""
        if not self.deaths_by_enemy:
            return None
        nemesis_name = max(self.deaths_by_enemy, key=self.deaths_by_enemy.get)
        return (nemesis_name, self.deaths_by_enemy[nemesis_name])

    @property
    def victim(self) -> tuple[str, int] | None:
        """The enemy this player killed most often (name, count)."""
        if not self.kills_against_enemy:
            return None
        victim_name = max(self.kills_against_enemy, key=self.kills_against_enemy.get)
        return (victim_name, self.kills_against_enemy[victim_name])

    def get_matchup_vs(self, enemy_name: str) -> dict:
        """Get head-to-head stats against a specific enemy."""
        kills = self.kills_against_enemy.get(enemy_name, 0)
        deaths = self.deaths_by_enemy.get(enemy_name, 0)
        return {
            "enemy": enemy_name,
            "kills": kills,
            "deaths": deaths,
            "kd_ratio": round(kills / deaths, 2) if deaths > 0 else float(kills),
            "net": kills - deaths,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        nemesis = self.nemesis
        victim = self.victim
        return {
            "nemesis": {"name": nemesis[0], "deaths": nemesis[1]} if nemesis else None,
            "favorite_victim": {"name": victim[0], "kills": victim[1]} if victim else None,
            "deaths_by_enemy": self.deaths_by_enemy,
            "kills_against_enemy": self.kills_against_enemy,
        }


@dataclass
class TimeAliveStats:
    """Track time alive and survival patterns."""

    total_rounds: int = 0
    rounds_survived: int = 0
    total_time_alive_ticks: int = 0
    death_ticks: list[int] = field(default_factory=list)  # Tick within round when died
    tick_rate: float = 64.0

    @property
    def survival_rate(self) -> float:
        """Percentage of rounds survived."""
        if self.total_rounds == 0:
            return 0.0
        return round(self.rounds_survived / self.total_rounds * 100, 1)

    @property
    def average_time_alive_seconds(self) -> float:
        """Average time alive per round in seconds."""
        if self.total_rounds == 0:
            return 0.0
        avg_ticks = self.total_time_alive_ticks / self.total_rounds
        return round(avg_ticks / self.tick_rate, 1)

    @property
    def average_death_timing_seconds(self) -> float | None:
        """Average time into round when player dies (for rounds where they died)."""
        if not self.death_ticks:
            return None
        avg_ticks = sum(self.death_ticks) / len(self.death_ticks)
        return round(avg_ticks / self.tick_rate, 1)

    @property
    def early_death_rate(self) -> float:
        """Percentage of deaths that happen in first 30 seconds of round."""
        if not self.death_ticks:
            return 0.0
        early_threshold = 30 * self.tick_rate  # 30 seconds in ticks
        early_deaths = sum(1 for t in self.death_ticks if t < early_threshold)
        return round(early_deaths / len(self.death_ticks) * 100, 1)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_rounds": self.total_rounds,
            "rounds_survived": self.rounds_survived,
            "survival_rate": self.survival_rate,
            "average_time_alive_seconds": self.average_time_alive_seconds,
            "average_death_timing_seconds": self.average_death_timing_seconds,
            "early_death_rate": self.early_death_rate,
        }


@dataclass
class KDABreakdown:
    """Detailed KDA statistics with context."""

    kills: int = 0
    deaths: int = 0
    assists: int = 0
    headshot_kills: int = 0
    wallbang_kills: int = 0
    noscope_kills: int = 0
    smoke_kills: int = 0  # Kills through smoke
    blind_kills: int = 0  # Kills while flashed
    first_kills: int = 0  # Opening kills
    first_deaths: int = 0  # Opening deaths
    clutch_kills: int = 0
    trade_kills: int = 0
    flash_assists: int = 0

    @property
    def kda_ratio(self) -> float:
        """(Kills + Assists) / Deaths ratio."""
        if self.deaths == 0:
            return float(self.kills + self.assists)
        return round((self.kills + self.assists) / self.deaths, 2)

    @property
    def kd_ratio(self) -> float:
        """Kill/Death ratio."""
        if self.deaths == 0:
            return float(self.kills)
        return round(self.kills / self.deaths, 2)

    @property
    def kd_diff(self) -> int:
        """Kill - Death difference."""
        return self.kills - self.deaths

    @property
    def headshot_percentage(self) -> float:
        """Percentage of kills that were headshots."""
        if self.kills == 0:
            return 0.0
        return round(self.headshot_kills / self.kills * 100, 1)

    @property
    def opening_duel_win_rate(self) -> float:
        """Win rate in opening duels."""
        total = self.first_kills + self.first_deaths
        if total == 0:
            return 0.0
        return round(self.first_kills / total * 100, 1)

    @property
    def special_kills_count(self) -> int:
        """Total of wallbangs + noscopes + smoke kills + blind kills."""
        return self.wallbang_kills + self.noscope_kills + self.smoke_kills + self.blind_kills

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "kda_ratio": self.kda_ratio,
            "kd_ratio": self.kd_ratio,
            "kd_diff": self.kd_diff,
            "headshot_kills": self.headshot_kills,
            "headshot_percentage": self.headshot_percentage,
            "wallbang_kills": self.wallbang_kills,
            "noscope_kills": self.noscope_kills,
            "smoke_kills": self.smoke_kills,
            "blind_kills": self.blind_kills,
            "first_kills": self.first_kills,
            "first_deaths": self.first_deaths,
            "opening_duel_win_rate": self.opening_duel_win_rate,
            "clutch_kills": self.clutch_kills,
            "trade_kills": self.trade_kills,
            "flash_assists": self.flash_assists,
            "special_kills": self.special_kills_count,
        }


@dataclass
class PlayerPerformanceMetrics:
    """Complete performance metrics for a player."""

    steam_id: int
    name: str
    team: str

    kda: KDABreakdown = field(default_factory=KDABreakdown)
    distance_stats: KillDistanceStats = field(default_factory=KillDistanceStats)
    nemesis_stats: NemesisStats = field(default_factory=NemesisStats)
    time_alive: TimeAliveStats = field(default_factory=TimeAliveStats)

    # Additional context
    rounds_played: int = 0
    total_damage: int = 0

    @property
    def adr(self) -> float:
        """Average Damage per Round."""
        if self.rounds_played == 0:
            return 0.0
        return round(self.total_damage / self.rounds_played, 1)

    @property
    def kills_per_round(self) -> float:
        """Average kills per round."""
        if self.rounds_played == 0:
            return 0.0
        return round(self.kda.kills / self.rounds_played, 2)

    @property
    def deaths_per_round(self) -> float:
        """Average deaths per round."""
        if self.rounds_played == 0:
            return 0.0
        return round(self.kda.deaths / self.rounds_played, 2)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "steam_id": str(self.steam_id),
            "name": self.name,
            "team": self.team,
            "rounds_played": self.rounds_played,
            "total_damage": self.total_damage,
            "adr": self.adr,
            "kills_per_round": self.kills_per_round,
            "deaths_per_round": self.deaths_per_round,
            "kda": self.kda.to_dict(),
            "distance_stats": self.distance_stats.to_dict(),
            "nemesis_stats": self.nemesis_stats.to_dict(),
            "time_alive": self.time_alive.to_dict(),
        }


# =============================================================================
# Calculation Functions
# =============================================================================


def calculate_kill_distance(
    attacker_x: float | None,
    attacker_y: float | None,
    attacker_z: float | None,
    victim_x: float | None,
    victim_y: float | None,
    victim_z: float | None,
) -> float | None:
    """Calculate 3D distance between attacker and victim positions."""
    if any(v is None for v in [attacker_x, attacker_y, attacker_z, victim_x, victim_y, victim_z]):
        return None

    dx = attacker_x - victim_x
    dy = attacker_y - victim_y
    dz = attacker_z - victim_z

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_player_performance_metrics(
    match_data: MatchData,
) -> dict[int, PlayerPerformanceMetrics]:
    """
    Compute detailed performance metrics for all players in a match.

    Args:
        match_data: Parsed match data from DemoParser

    Returns:
        Dictionary mapping steam_id to PlayerPerformanceMetrics
    """
    metrics: dict[int, PlayerPerformanceMetrics] = {}

    # Initialize metrics for all known players
    for steam_id, name in match_data.player_names.items():
        team = match_data.player_teams.get(steam_id, "Unknown")
        metrics[steam_id] = PlayerPerformanceMetrics(
            steam_id=steam_id,
            name=name,
            team=team,
            rounds_played=match_data.num_rounds,
        )
        metrics[steam_id].time_alive.total_rounds = match_data.num_rounds
        metrics[steam_id].time_alive.tick_rate = match_data.tick_rate

    # Build round info for time alive calculations
    round_start_ticks: dict[int, int] = {}
    round_end_ticks: dict[int, int] = {}
    for round_info in match_data.game_rounds:
        round_start_ticks[round_info.round_num] = (
            round_info.freeze_end_tick or round_info.start_tick
        )
        round_end_ticks[round_info.round_num] = round_info.end_tick

    # Track deaths per round for survival calculation
    deaths_in_round: dict[int, dict[int, int]] = defaultdict(dict)  # round -> {steamid: tick}

    # Process kills
    for kill in match_data.kills:
        attacker_id = kill.attacker_steamid
        victim_id = kill.victim_steamid

        # Skip invalid entries
        if attacker_id == 0 or victim_id == 0:
            continue

        # Ensure both players have metrics
        if attacker_id not in metrics:
            metrics[attacker_id] = PlayerPerformanceMetrics(
                steam_id=attacker_id,
                name=kill.attacker_name,
                team=kill.attacker_side,
                rounds_played=match_data.num_rounds,
            )
        if victim_id not in metrics:
            metrics[victim_id] = PlayerPerformanceMetrics(
                steam_id=victim_id,
                name=kill.victim_name,
                team=kill.victim_side,
                rounds_played=match_data.num_rounds,
            )

        attacker = metrics[attacker_id]
        victim = metrics[victim_id]

        # KDA
        attacker.kda.kills += 1
        victim.kda.deaths += 1

        if kill.headshot:
            attacker.kda.headshot_kills += 1
        if kill.penetrated:
            attacker.kda.wallbang_kills += 1
        if kill.noscope:
            attacker.kda.noscope_kills += 1
        if kill.thrusmoke:
            attacker.kda.smoke_kills += 1
        if kill.attackerblind:
            attacker.kda.blind_kills += 1

        # Kill distance
        distance = calculate_kill_distance(
            kill.attacker_x,
            kill.attacker_y,
            kill.attacker_z,
            kill.victim_x,
            kill.victim_y,
            kill.victim_z,
        )
        if distance is not None:
            attacker.distance_stats.total_kills += 1
            attacker.distance_stats.total_distance += distance
            attacker.distance_stats.distances.append(distance)

            if distance < CLOSE_RANGE_MAX:
                attacker.distance_stats.close_range_kills += 1
            elif distance < MID_RANGE_MAX:
                attacker.distance_stats.mid_range_kills += 1
            else:
                attacker.distance_stats.long_range_kills += 1

        # Nemesis tracking
        attacker_name = kill.attacker_name
        victim_name = kill.victim_name

        attacker.nemesis_stats.kills_against_enemy[victim_name] = (
            attacker.nemesis_stats.kills_against_enemy.get(victim_name, 0) + 1
        )
        victim.nemesis_stats.deaths_by_enemy[attacker_name] = (
            victim.nemesis_stats.deaths_by_enemy.get(attacker_name, 0) + 1
        )
        victim.nemesis_stats.deaths_by_enemy_steamid[attacker_id] = (
            victim.nemesis_stats.deaths_by_enemy_steamid.get(attacker_id, 0) + 1
        )

        # Track death timing for time alive calculation
        round_num = kill.round_num
        if round_num > 0:
            round_start = round_start_ticks.get(round_num, 0)
            tick_into_round = kill.tick - round_start
            if tick_into_round > 0:
                deaths_in_round[round_num][victim_id] = tick_into_round
                victim.time_alive.death_ticks.append(tick_into_round)

        # Assists
        if kill.assister_steamid and kill.assister_steamid in metrics:
            metrics[kill.assister_steamid].kda.assists += 1
            if kill.flash_assist:
                metrics[kill.assister_steamid].kda.flash_assists += 1

    # Calculate survival stats
    for steam_id, player in metrics.items():
        deaths_count = 0
        for round_num in range(1, match_data.num_rounds + 1):
            if steam_id not in deaths_in_round.get(round_num, {}):
                player.time_alive.rounds_survived += 1
                # Full round survived - add round duration
                round_duration = round_end_ticks.get(round_num, 0) - round_start_ticks.get(
                    round_num, 0
                )
                player.time_alive.total_time_alive_ticks += max(0, round_duration)
            else:
                # Died in round - add time until death
                player.time_alive.total_time_alive_ticks += deaths_in_round[round_num][steam_id]
                deaths_count += 1

    # Calculate total damage from damages list
    for damage in match_data.damages:
        attacker_id = damage.attacker_steamid
        if attacker_id in metrics:
            metrics[attacker_id].total_damage += damage.damage

    # Detect opening duels (first kill of each round)
    kills_by_round: dict[int, list] = defaultdict(list)
    for kill in match_data.kills:
        kills_by_round[kill.round_num].append(kill)

    for _round_num, round_kills in kills_by_round.items():
        if round_kills:
            # Sort by tick to find first kill
            round_kills.sort(key=lambda k: k.tick)
            first_kill = round_kills[0]

            if first_kill.attacker_steamid in metrics:
                metrics[first_kill.attacker_steamid].kda.first_kills += 1
            if first_kill.victim_steamid in metrics:
                metrics[first_kill.victim_steamid].kda.first_deaths += 1

    return metrics


def get_performance_summary(metrics: PlayerPerformanceMetrics) -> dict:
    """
    Get a summary of performance with ratings and insights.

    Args:
        metrics: Player performance metrics

    Returns:
        Dictionary with performance summary and ratings
    """
    summary = {
        "player": metrics.name,
        "team": metrics.team,
        "overview": {
            "kd_ratio": metrics.kda.kd_ratio,
            "kda_ratio": metrics.kda.kda_ratio,
            "adr": metrics.adr,
            "headshot_pct": metrics.kda.headshot_percentage,
            "survival_rate": metrics.time_alive.survival_rate,
        },
        "strengths": [],
        "weaknesses": [],
        "notable_stats": [],
    }

    # Identify strengths
    if metrics.kda.headshot_percentage >= 50:
        summary["strengths"].append("Excellent aim precision (high headshot %)")
    if metrics.kda.opening_duel_win_rate >= 60:
        summary["strengths"].append("Strong opening duel player")
    if metrics.time_alive.survival_rate >= 50:
        summary["strengths"].append("Good survival instincts")
    if metrics.distance_stats.long_range_percentage >= 30:
        summary["strengths"].append("Effective at long range")
    if metrics.kda.kd_ratio >= 1.5:
        summary["strengths"].append("High impact fragger")

    # Identify weaknesses
    if metrics.kda.headshot_percentage < 30:
        summary["weaknesses"].append("Low headshot percentage - work on crosshair placement")
    if (
        metrics.kda.opening_duel_win_rate < 40
        and (metrics.kda.first_kills + metrics.kda.first_deaths) >= 3
    ):
        summary["weaknesses"].append("Losing opening duels - consider adjusting positioning")
    if metrics.time_alive.early_death_rate > 50:
        summary["weaknesses"].append("Dying too early in rounds - slow down entries")
    if metrics.kda.kd_ratio < 0.8:
        summary["weaknesses"].append("Negative K/D impact - focus on staying alive")

    # Notable stats
    if metrics.kda.wallbang_kills >= 2:
        summary["notable_stats"].append(f"{metrics.kda.wallbang_kills} wallbang kills")
    if metrics.kda.noscope_kills >= 1:
        summary["notable_stats"].append(f"{metrics.kda.noscope_kills} noscope kills")
    if metrics.kda.blind_kills >= 1:
        summary["notable_stats"].append(f"{metrics.kda.blind_kills} kills while flashed")

    nemesis = metrics.nemesis_stats.nemesis
    if nemesis and nemesis[1] >= 3:
        summary["notable_stats"].append(f"Nemesis: {nemesis[0]} ({nemesis[1]} deaths)")

    victim = metrics.nemesis_stats.victim
    if victim and victim[1] >= 3:
        summary["notable_stats"].append(f"Favorite victim: {victim[0]} ({victim[1]} kills)")

    return summary
