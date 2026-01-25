"""
Team Performance Metrics Module

Calculates comprehensive team-level performance metrics from CS2 demo data.
All calculations are performed locally - no external services required (FREE).

Metrics include:
- Team KDA ratio (Kills + Assists / Deaths)
- Average kill distance
- Deaths by enemy team
- Time spent alive/in-game
- Round win rates
- Economy efficiency
- Trade success rates
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Safe Type Conversion Helpers
# =============================================================================


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TeamMetrics:
    """Comprehensive team performance metrics."""

    team_name: str
    team_side: str  # "CT" or "T" (or "Both" for overall)

    # Core Stats
    total_kills: int = 0
    total_deaths: int = 0
    total_assists: int = 0
    kda_ratio: float = 0.0  # (K + A) / D

    # Kill Analysis
    avg_kill_distance: float = 0.0  # Average distance of kills in units
    headshot_kills: int = 0
    headshot_percentage: float = 0.0
    wallbang_kills: int = 0
    noscope_kills: int = 0
    smoke_kills: int = 0  # Through smoke

    # Deaths Analysis
    deaths_by_enemy: int = 0  # Deaths caused by enemy team
    deaths_by_bomb: int = 0  # Deaths from bomb explosion
    team_kills: int = 0  # Friendly fire deaths

    # Time Analysis
    total_time_alive_seconds: float = 0.0
    avg_time_alive_per_round: float = 0.0
    total_rounds_played: int = 0

    # Round Performance
    rounds_won: int = 0
    rounds_lost: int = 0
    round_win_rate: float = 0.0

    # Economy
    total_money_spent: int = 0
    avg_money_per_round: float = 0.0
    eco_rounds_won: int = 0  # Rounds won with < $10k team buy
    force_buy_rounds_won: int = 0

    # Trades
    successful_trades: int = 0
    deaths_traded: int = 0
    trade_success_rate: float = 0.0

    # Utility
    total_flash_assists: int = 0
    total_damage_from_utility: float = 0.0

    # Objective
    bombs_planted: int = 0
    bombs_defused: int = 0

    # Player breakdown
    player_stats: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "team_name": self.team_name,
            "team_side": self.team_side,
            "core_stats": {
                "total_kills": self.total_kills,
                "total_deaths": self.total_deaths,
                "total_assists": self.total_assists,
                "kda_ratio": round(self.kda_ratio, 2),
            },
            "kill_analysis": {
                "avg_kill_distance": round(self.avg_kill_distance, 1),
                "headshot_kills": self.headshot_kills,
                "headshot_percentage": round(self.headshot_percentage, 1),
                "wallbang_kills": self.wallbang_kills,
                "noscope_kills": self.noscope_kills,
                "smoke_kills": self.smoke_kills,
            },
            "deaths_analysis": {
                "deaths_by_enemy": self.deaths_by_enemy,
                "deaths_by_bomb": self.deaths_by_bomb,
                "team_kills": self.team_kills,
            },
            "time_analysis": {
                "total_time_alive_seconds": round(self.total_time_alive_seconds, 1),
                "avg_time_alive_per_round": round(self.avg_time_alive_per_round, 1),
                "total_rounds_played": self.total_rounds_played,
            },
            "round_performance": {
                "rounds_won": self.rounds_won,
                "rounds_lost": self.rounds_lost,
                "round_win_rate": round(self.round_win_rate, 1),
            },
            "economy": {
                "total_money_spent": self.total_money_spent,
                "avg_money_per_round": round(self.avg_money_per_round, 0),
                "eco_rounds_won": self.eco_rounds_won,
                "force_buy_rounds_won": self.force_buy_rounds_won,
            },
            "trades": {
                "successful_trades": self.successful_trades,
                "deaths_traded": self.deaths_traded,
                "trade_success_rate": round(self.trade_success_rate, 1),
            },
            "utility": {
                "total_flash_assists": self.total_flash_assists,
                "total_damage_from_utility": round(self.total_damage_from_utility, 1),
            },
            "objectives": {
                "bombs_planted": self.bombs_planted,
                "bombs_defused": self.bombs_defused,
            },
            "player_stats": self.player_stats,
        }


@dataclass
class MatchTeamAnalysis:
    """Team analysis for an entire match."""

    map_name: str
    total_rounds: int
    final_score_ct: int
    final_score_t: int

    ct_metrics: TeamMetrics | None = None
    t_metrics: TeamMetrics | None = None

    # Per-round breakdown
    round_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "map_name": self.map_name,
            "total_rounds": self.total_rounds,
            "final_score": {
                "ct": self.final_score_ct,
                "t": self.final_score_t,
            },
            "ct_team": self.ct_metrics.to_dict() if self.ct_metrics else None,
            "t_team": self.t_metrics.to_dict() if self.t_metrics else None,
            "round_results": self.round_results,
        }


# =============================================================================
# Team Performance Calculator
# =============================================================================


class TeamPerformanceCalculator:
    """
    Calculates team-level performance metrics from demo data.

    All processing is done locally - no external API calls (FREE).
    """

    # Trade window in ticks (5 seconds at 64 tick = 320 ticks)
    TRADE_WINDOW_TICKS = 320

    # Economy thresholds
    ECO_BUY_THRESHOLD = 10000  # Team total < $10k
    FORCE_BUY_THRESHOLD = 20000  # Team total $10k-$20k

    def __init__(self, match_data: Any):
        """
        Initialize calculator with parsed match data.

        Args:
            match_data: MatchData/DemoData object from parser
        """
        self.data = match_data
        self._ct_players: set[int] = set()
        self._t_players: set[int] = set()
        self._identify_teams()

    def _identify_teams(self) -> None:
        """Identify which players belong to which team."""
        # Use player_teams mapping if available
        if hasattr(self.data, "player_teams") and self.data.player_teams:
            for steam_id, team in self.data.player_teams.items():
                if team in ("CT", "ct", 3):
                    self._ct_players.add(steam_id)
                elif team in ("T", "t", "TERRORIST", 2):
                    self._t_players.add(steam_id)
            return

        # Fall back to analyzing kills
        if hasattr(self.data, "kills"):
            for kill in self.data.kills:
                if hasattr(kill, "attacker_side"):
                    if kill.attacker_side in ("CT", "ct"):
                        self._ct_players.add(kill.attacker_steamid)
                    elif kill.attacker_side in ("T", "t", "TERRORIST"):
                        self._t_players.add(kill.attacker_steamid)
                if hasattr(kill, "victim_side"):
                    if kill.victim_side in ("CT", "ct"):
                        self._ct_players.add(kill.victim_steamid)
                    elif kill.victim_side in ("T", "t", "TERRORIST"):
                        self._t_players.add(kill.victim_steamid)

    def calculate_team_metrics(self, side: str = "CT") -> TeamMetrics:
        """
        Calculate comprehensive metrics for one team.

        Args:
            side: "CT" or "T"

        Returns:
            TeamMetrics object with all calculated metrics
        """
        players = self._ct_players if side == "CT" else self._t_players
        enemy_players = self._t_players if side == "CT" else self._ct_players

        metrics = TeamMetrics(
            team_name=f"Team {side}",
            team_side=side,
        )

        # Calculate kill-based metrics
        self._calculate_kill_metrics(metrics, players, enemy_players)

        # Calculate death metrics
        self._calculate_death_metrics(metrics, players, enemy_players)

        # Calculate time metrics
        self._calculate_time_metrics(metrics, players, side)

        # Calculate round performance
        self._calculate_round_metrics(metrics, side)

        # Calculate economy metrics
        self._calculate_economy_metrics(metrics, players, side)

        # Calculate trade metrics
        self._calculate_trade_metrics(metrics, players)

        # Calculate utility metrics
        self._calculate_utility_metrics(metrics, players)

        # Calculate objective metrics
        self._calculate_objective_metrics(metrics, side)

        # Calculate KDA ratio
        if metrics.total_deaths > 0:
            metrics.kda_ratio = (metrics.total_kills + metrics.total_assists) / metrics.total_deaths
        else:
            metrics.kda_ratio = float(metrics.total_kills + metrics.total_assists)

        # Calculate per-player stats
        self._calculate_player_breakdown(metrics, players)

        return metrics

    def _calculate_kill_metrics(
        self, metrics: TeamMetrics, team_players: set[int], enemy_players: set[int]
    ) -> None:
        """Calculate kill-related metrics."""
        if not hasattr(self.data, "kills"):
            return

        kill_distances: list[float] = []

        for kill in self.data.kills:
            attacker_id = getattr(kill, "attacker_steamid", None)
            if attacker_id not in team_players:
                continue

            metrics.total_kills += 1

            # Headshot
            if getattr(kill, "headshot", False):
                metrics.headshot_kills += 1

            # Special kills
            if getattr(kill, "penetrated", False):
                metrics.wallbang_kills += 1
            if getattr(kill, "noscope", False):
                metrics.noscope_kills += 1
            if getattr(kill, "thrusmoke", False):
                metrics.smoke_kills += 1

            # Flash assist
            if getattr(kill, "flash_assist", False):
                metrics.total_flash_assists += 1

            # Calculate kill distance
            distance = self._calculate_kill_distance(kill)
            if distance is not None and distance > 0:
                kill_distances.append(distance)

            # Count assists
            assister_id = getattr(kill, "assister_steamid", None)
            if assister_id and assister_id in team_players:
                metrics.total_assists += 1

        # Calculate averages
        if kill_distances:
            metrics.avg_kill_distance = sum(kill_distances) / len(kill_distances)

        if metrics.total_kills > 0:
            metrics.headshot_percentage = (metrics.headshot_kills / metrics.total_kills) * 100

    def _calculate_kill_distance(self, kill: Any) -> float | None:
        """Calculate the distance of a kill in game units."""
        try:
            ax = safe_float(getattr(kill, "attacker_x", None))
            ay = safe_float(getattr(kill, "attacker_y", None))
            az = safe_float(getattr(kill, "attacker_z", None))
            vx = safe_float(getattr(kill, "victim_x", None))
            vy = safe_float(getattr(kill, "victim_y", None))
            vz = safe_float(getattr(kill, "victim_z", None))

            if all(v != 0.0 for v in [ax, ay, vx, vy]):
                dx = ax - vx
                dy = ay - vy
                dz = az - vz
                return math.sqrt(dx * dx + dy * dy + dz * dz)
        except (TypeError, ValueError):
            pass
        return None

    def _calculate_death_metrics(
        self, metrics: TeamMetrics, team_players: set[int], enemy_players: set[int]
    ) -> None:
        """Calculate death-related metrics."""
        if not hasattr(self.data, "kills"):
            return

        for kill in self.data.kills:
            victim_id = getattr(kill, "victim_steamid", None)
            if victim_id not in team_players:
                continue

            metrics.total_deaths += 1

            attacker_id = getattr(kill, "attacker_steamid", None)
            if attacker_id in enemy_players:
                metrics.deaths_by_enemy += 1
            elif attacker_id in team_players and attacker_id != victim_id:
                metrics.team_kills += 1

        # Check bomb deaths
        if hasattr(self.data, "bomb_events"):
            for event in self.data.bomb_events:
                event_type = getattr(event, "event_type", "")
                if event_type == "explode":
                    # Approximate: T players near bomb when it explodes
                    # This is a simplification - real implementation would check positions
                    pass

    def _calculate_time_metrics(
        self, metrics: TeamMetrics, team_players: set[int], side: str
    ) -> None:
        """Calculate time-related metrics."""
        metrics.total_rounds_played = getattr(self.data, "num_rounds", 0)

        # If we have tick data, calculate actual time alive
        tick_rate = getattr(self.data, "tick_rate", 64)
        if tick_rate <= 0:
            tick_rate = 64

        # Estimate time alive based on when players died in rounds
        if hasattr(self.data, "kills") and hasattr(self.data, "game_rounds"):
            rounds_info = self.data.game_rounds or []
            total_ticks_alive = 0

            for round_info in rounds_info:
                round_num = getattr(round_info, "round_num", 0)
                round_start = getattr(round_info, "start_tick", 0)
                round_end = getattr(
                    round_info, "end_tick", round_start + 64 * 115
                )  # ~115 sec default

                # Find when each team player died this round
                for player_id in team_players:
                    death_tick = None
                    for kill in self.data.kills:
                        if (
                            getattr(kill, "victim_steamid", None) == player_id
                            and getattr(kill, "round_num", -1) == round_num
                        ):
                            death_tick = getattr(kill, "tick", None)
                            break

                    if death_tick:
                        total_ticks_alive += death_tick - round_start
                    else:
                        # Survived the round
                        total_ticks_alive += round_end - round_start

            metrics.total_time_alive_seconds = total_ticks_alive / tick_rate

            if metrics.total_rounds_played > 0:
                metrics.avg_time_alive_per_round = (
                    metrics.total_time_alive_seconds / metrics.total_rounds_played
                )

    def _calculate_round_metrics(self, metrics: TeamMetrics, side: str) -> None:
        """Calculate round win/loss metrics."""
        if not hasattr(self.data, "game_rounds"):
            # Use final score as fallback
            if side == "CT":
                metrics.rounds_won = getattr(self.data, "final_score_ct", 0)
                metrics.rounds_lost = getattr(self.data, "final_score_t", 0)
            else:
                metrics.rounds_won = getattr(self.data, "final_score_t", 0)
                metrics.rounds_lost = getattr(self.data, "final_score_ct", 0)
        else:
            for round_info in self.data.game_rounds:
                winner = getattr(round_info, "winner", "")
                if winner.upper() == side.upper():
                    metrics.rounds_won += 1
                elif winner:
                    metrics.rounds_lost += 1

        total_rounds = metrics.rounds_won + metrics.rounds_lost
        if total_rounds > 0:
            metrics.round_win_rate = (metrics.rounds_won / total_rounds) * 100

    def _calculate_economy_metrics(
        self, metrics: TeamMetrics, team_players: set[int], side: str
    ) -> None:
        """Calculate economy metrics."""
        # This would require round-by-round economy data
        # For now, estimate from player stats if available
        if hasattr(self.data, "player_stats"):
            for player_id in team_players:
                stats = self.data.player_stats.get(player_id, {})
                # Sum up equipment values if available
                equip_value = stats.get("total_equipment_value", 0)
                metrics.total_money_spent += safe_int(equip_value)

        if metrics.total_rounds_played > 0:
            metrics.avg_money_per_round = metrics.total_money_spent / metrics.total_rounds_played

    def _calculate_trade_metrics(self, metrics: TeamMetrics, team_players: set[int]) -> None:
        """Calculate trade kill metrics."""
        if not hasattr(self.data, "kills"):
            return

        kills_list = list(self.data.kills)
        kills_list.sort(key=lambda k: getattr(k, "tick", 0))

        for i, kill in enumerate(kills_list):
            victim_id = getattr(kill, "victim_steamid", None)
            if victim_id not in team_players:
                continue

            kill_tick = getattr(kill, "tick", 0)
            attacker_id = getattr(kill, "attacker_steamid", None)

            # Look for a trade within the window
            for j in range(i + 1, len(kills_list)):
                next_kill = kills_list[j]
                next_tick = getattr(next_kill, "tick", 0)

                if next_tick - kill_tick > self.TRADE_WINDOW_TICKS:
                    break

                # Check if the original attacker was killed by our team
                next_victim = getattr(next_kill, "victim_steamid", None)
                next_attacker = getattr(next_kill, "attacker_steamid", None)

                if next_victim == attacker_id and next_attacker in team_players:
                    metrics.successful_trades += 1
                    metrics.deaths_traded += 1
                    break

        if metrics.total_deaths > 0:
            metrics.trade_success_rate = (metrics.deaths_traded / metrics.total_deaths) * 100

    def _calculate_utility_metrics(self, metrics: TeamMetrics, team_players: set[int]) -> None:
        """Calculate utility usage metrics."""
        if hasattr(self.data, "damages"):
            for damage in self.data.damages:
                attacker_id = getattr(damage, "attacker_steamid", None)
                if attacker_id not in team_players:
                    continue

                weapon = getattr(damage, "weapon", "")
                if weapon in ("hegrenade", "molotov", "incgrenade", "inferno"):
                    dmg = safe_float(getattr(damage, "damage", 0))
                    metrics.total_damage_from_utility += dmg

    def _calculate_objective_metrics(self, metrics: TeamMetrics, side: str) -> None:
        """Calculate bomb plant/defuse metrics."""
        if not hasattr(self.data, "bomb_events"):
            return

        for event in self.data.bomb_events:
            event_type = getattr(event, "event_type", "")
            if event_type == "plant" and side == "T":
                metrics.bombs_planted += 1
            elif event_type == "defuse" and side == "CT":
                metrics.bombs_defused += 1

    def _calculate_player_breakdown(self, metrics: TeamMetrics, team_players: set[int]) -> None:
        """Calculate per-player stats breakdown."""
        player_kills: dict[int, int] = dict.fromkeys(team_players, 0)
        player_deaths: dict[int, int] = dict.fromkeys(team_players, 0)
        player_assists: dict[int, int] = dict.fromkeys(team_players, 0)
        player_headshots: dict[int, int] = dict.fromkeys(team_players, 0)

        if hasattr(self.data, "kills"):
            for kill in self.data.kills:
                attacker_id = getattr(kill, "attacker_steamid", None)
                victim_id = getattr(kill, "victim_steamid", None)
                assister_id = getattr(kill, "assister_steamid", None)

                if attacker_id in player_kills:
                    player_kills[attacker_id] += 1
                    if getattr(kill, "headshot", False):
                        player_headshots[attacker_id] += 1

                if victim_id in player_deaths:
                    player_deaths[victim_id] += 1

                if assister_id and assister_id in player_assists:
                    player_assists[assister_id] += 1

        # Get player names
        player_names = getattr(self.data, "player_names", {})

        for player_id in team_players:
            name = player_names.get(player_id, f"Player_{player_id}")
            k = player_kills.get(player_id, 0)
            d = player_deaths.get(player_id, 0)
            a = player_assists.get(player_id, 0)
            hs = player_headshots.get(player_id, 0)

            metrics.player_stats[str(player_id)] = {
                "name": name,
                "kills": k,
                "deaths": d,
                "assists": a,
                "headshots": hs,
                "kda": round((k + a) / d, 2) if d > 0 else k + a,
                "hs_percentage": round((hs / k) * 100, 1) if k > 0 else 0.0,
            }

    def analyze_match(self) -> MatchTeamAnalysis:
        """
        Perform complete team analysis for the match.

        Returns:
            MatchTeamAnalysis with metrics for both teams
        """
        ct_metrics = self.calculate_team_metrics("CT")
        t_metrics = self.calculate_team_metrics("T")

        # Build round results
        round_results = []
        if hasattr(self.data, "game_rounds"):
            for round_info in self.data.game_rounds:
                round_results.append(
                    {
                        "round_num": getattr(round_info, "round_num", 0),
                        "winner": getattr(round_info, "winner", ""),
                        "win_reason": getattr(round_info, "reason", ""),
                        "ct_score": getattr(round_info, "ct_score", 0),
                        "t_score": getattr(round_info, "t_score", 0),
                    }
                )

        return MatchTeamAnalysis(
            map_name=getattr(self.data, "map_name", "Unknown"),
            total_rounds=getattr(self.data, "num_rounds", 0),
            final_score_ct=getattr(self.data, "final_score_ct", 0),
            final_score_t=getattr(self.data, "final_score_t", 0),
            ct_metrics=ct_metrics,
            t_metrics=t_metrics,
            round_results=round_results,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def calculate_team_metrics(match_data: Any) -> MatchTeamAnalysis:
    """
    Calculate team performance metrics from parsed demo data.

    Args:
        match_data: MatchData/DemoData object from parser

    Returns:
        MatchTeamAnalysis with comprehensive team metrics
    """
    calculator = TeamPerformanceCalculator(match_data)
    return calculator.analyze_match()


def get_team_summary(match_data: Any) -> dict[str, Any]:
    """
    Get a summary of team performance as a dictionary.

    Args:
        match_data: MatchData/DemoData object from parser

    Returns:
        Dictionary with team performance summary
    """
    analysis = calculate_team_metrics(match_data)
    return analysis.to_dict()
