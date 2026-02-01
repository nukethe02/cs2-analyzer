"""
Data models for opponent scouting.

Defines dataclasses for player profiles, team reports, and aggregated statistics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PlayStyle(str, Enum):
    """Player playstyle classification based on aggression metrics."""

    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    MIXED = "mixed"


class EconomyTendency(str, Enum):
    """Team economy management tendency."""

    AGGRESSIVE = "aggressive"  # Force buys often, eco rarely
    CONSERVATIVE = "conservative"  # Proper eco rounds, full saves
    BALANCED = "balanced"  # Mix of both approaches


@dataclass
class PositionTendency:
    """Player's tendency to hold/take specific positions."""

    zone_name: str
    frequency: float  # 0-1, how often they play this position
    success_rate: float  # 0-1, K/D ratio in this position
    avg_time_held: float  # seconds


@dataclass
class WeaponPreference:
    """Player's weapon usage statistics."""

    weapon_name: str
    usage_rate: float  # 0-1, how often they use this weapon
    kills_with: int
    accuracy: float  # 0-1


@dataclass
class PlayerScoutProfile:
    """Comprehensive scouting profile for an individual player."""

    steamid: str
    name: str

    # Core metrics
    play_style: PlayStyle
    aggression_score: float  # 0-100, higher = more aggressive
    consistency_score: float  # 0-100, how consistent across demos

    # Combat statistics (aggregated)
    avg_kills_per_round: float
    avg_deaths_per_round: float
    avg_adr: float
    avg_kast: float
    headshot_rate: float  # 0-1

    # Entry/opening statistics
    entry_attempt_rate: float  # 0-1, how often they attempt entry
    entry_success_rate: float  # 0-1, success when attempting entry
    opening_duel_win_rate: float  # 0-1

    # AWP usage
    awp_usage_rate: float  # 0-1, rounds with AWP
    awp_kills_per_awp_round: float

    # Timing patterns
    avg_first_kill_time_seconds: float  # Time into round for first kill
    avg_rotation_time_seconds: float  # Time to rotate on CT

    # Position preferences (by map)
    favorite_positions: dict[str, list[PositionTendency]] = field(default_factory=dict)

    # Weapon preferences
    weapon_preferences: list[WeaponPreference] = field(default_factory=list)

    # Clutch statistics
    clutch_attempts: int = 0
    clutch_wins: int = 0

    # Sample size
    demos_analyzed: int = 0
    rounds_analyzed: int = 0

    @property
    def clutch_win_rate(self) -> float:
        """Calculate clutch win rate."""
        if self.clutch_attempts == 0:
            return 0.0
        return self.clutch_wins / self.clutch_attempts

    @property
    def kd_ratio(self) -> float:
        """Calculate K/D ratio from per-round averages."""
        if self.avg_deaths_per_round == 0:
            return self.avg_kills_per_round
        return self.avg_kills_per_round / self.avg_deaths_per_round

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "steamid": self.steamid,
            "name": self.name,
            "play_style": self.play_style.value,
            "aggression_score": round(self.aggression_score, 1),
            "consistency_score": round(self.consistency_score, 1),
            "avg_kills_per_round": round(self.avg_kills_per_round, 2),
            "avg_deaths_per_round": round(self.avg_deaths_per_round, 2),
            "kd_ratio": round(self.kd_ratio, 2),
            "avg_adr": round(self.avg_adr, 1),
            "avg_kast": round(self.avg_kast, 1),
            "headshot_rate": round(self.headshot_rate * 100, 1),
            "entry_attempt_rate": round(self.entry_attempt_rate * 100, 1),
            "entry_success_rate": round(self.entry_success_rate * 100, 1),
            "opening_duel_win_rate": round(self.opening_duel_win_rate * 100, 1),
            "awp_usage_rate": round(self.awp_usage_rate * 100, 1),
            "awp_kills_per_awp_round": round(self.awp_kills_per_awp_round, 2),
            "avg_first_kill_time_seconds": round(self.avg_first_kill_time_seconds, 1),
            "avg_rotation_time_seconds": round(self.avg_rotation_time_seconds, 1),
            "favorite_positions": {
                map_name: [
                    {
                        "zone": pos.zone_name,
                        "frequency": round(pos.frequency * 100, 1),
                        "success_rate": round(pos.success_rate * 100, 1),
                    }
                    for pos in positions[:3]  # Top 3 positions
                ]
                for map_name, positions in self.favorite_positions.items()
            },
            "weapon_preferences": [
                {
                    "weapon": wp.weapon_name,
                    "usage_rate": round(wp.usage_rate * 100, 1),
                    "kills": wp.kills_with,
                }
                for wp in self.weapon_preferences[:5]  # Top 5 weapons
            ],
            "clutch_attempts": self.clutch_attempts,
            "clutch_wins": self.clutch_wins,
            "clutch_win_rate": round(self.clutch_win_rate * 100, 1),
            "demos_analyzed": self.demos_analyzed,
            "rounds_analyzed": self.rounds_analyzed,
        }


@dataclass
class MapTendency:
    """Team tendencies on a specific map."""

    map_name: str
    demos_analyzed: int
    rounds_analyzed: int

    # T-side tendencies
    t_side_default_setup: str  # Description of common default
    t_side_common_executes: list[str]  # List of common execute sites/styles
    t_side_aggression: float  # 0-100

    # CT-side tendencies
    ct_side_default_setup: str  # Description of default positions
    ct_side_rotation_speed: str  # "fast", "medium", "slow"
    ct_side_aggression: float  # 0-100

    # Round timing
    avg_execute_time_seconds: float  # When T executes typically happen
    avg_first_contact_seconds: float  # When first contact typically happens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "map_name": self.map_name,
            "demos_analyzed": self.demos_analyzed,
            "rounds_analyzed": self.rounds_analyzed,
            "t_side": {
                "default_setup": self.t_side_default_setup,
                "common_executes": self.t_side_common_executes,
                "aggression": round(self.t_side_aggression, 1),
            },
            "ct_side": {
                "default_setup": self.ct_side_default_setup,
                "rotation_speed": self.ct_side_rotation_speed,
                "aggression": round(self.ct_side_aggression, 1),
            },
            "timing": {
                "avg_execute_time": round(self.avg_execute_time_seconds, 1),
                "avg_first_contact": round(self.avg_first_contact_seconds, 1),
            },
        }


@dataclass
class TeamScoutReport:
    """Comprehensive scouting report for an opponent team."""

    team_name: str
    demos_analyzed: int
    total_rounds: int

    # Player profiles
    players: list[PlayerScoutProfile] = field(default_factory=list)

    # Map tendencies
    map_tendencies: list[MapTendency] = field(default_factory=list)

    # Overall team tendencies
    economy_tendency: EconomyTendency = EconomyTendency.BALANCED
    force_buy_rate: float = 0.0  # 0-1, how often they force
    eco_round_rate: float = 0.0  # 0-1, how often they eco

    # Anti-strat recommendations
    anti_strats: list[str] = field(default_factory=list)

    # Confidence indicator
    confidence_level: str = "low"  # "low" (1 demo), "medium" (2-3), "high" (4+)

    # Maps analyzed
    maps_analyzed: list[str] = field(default_factory=list)

    @property
    def t_side_win_rate(self) -> float:
        """Calculate overall T-side win rate."""
        # This would be calculated during report generation
        return 0.0

    @property
    def ct_side_win_rate(self) -> float:
        """Calculate overall CT-side win rate."""
        # This would be calculated during report generation
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "team_name": self.team_name,
            "demos_analyzed": self.demos_analyzed,
            "total_rounds": self.total_rounds,
            "confidence_level": self.confidence_level,
            "maps_analyzed": self.maps_analyzed,
            "players": [p.to_dict() for p in self.players],
            "map_tendencies": [mt.to_dict() for mt in self.map_tendencies],
            "economy": {
                "tendency": self.economy_tendency.value,
                "force_buy_rate": round(self.force_buy_rate * 100, 1),
                "eco_round_rate": round(self.eco_round_rate * 100, 1),
            },
            "anti_strats": self.anti_strats,
        }


@dataclass
class ScoutingSession:
    """Represents an active scouting session with multiple demos."""

    session_id: str
    created_at: float  # Unix timestamp
    demo_ids: list[str] = field(default_factory=list)
    opponent_steamids: set[str] = field(default_factory=set)
    team_name: str = "Unknown Team"

    # Cached data for quick access
    player_names: dict[str, str] = field(default_factory=dict)  # steamid -> name
    maps_included: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "demo_count": len(self.demo_ids),
            "opponent_count": len(self.opponent_steamids),
            "team_name": self.team_name,
            "player_names": self.player_names,
            "maps_included": list(self.maps_included),
        }
