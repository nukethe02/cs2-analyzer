"""
Tactical Demo Analyzer - Strategic & Play-Based Analysis

Analyzes:
- Play execution (how teams execute attacks/defenses)
- Utility usage patterns (nades, flashes, smokes timing)
- Timing analysis (when kills happen in round)
- Player roles (lurker, entry, support, AWP, IGL)
- Positioning patterns (where kills/deaths occur)
- Weaknesses/strengths (team and individual)
- Trade chains and support play
- Economy decisions and buy decisions
- Round economy impact on kills
- Team coordination level
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class RoundPhase(Enum):
    """Phases of a round."""
    OPENING = "opening"  # First 10 seconds
    EARLY = "early"      # 10-30 seconds
    MID = "mid"          # 30-60 seconds
    LATE = "late"        # 60+ seconds
    PLANT = "plant"      # Post-plant phase
    OVERTIME = "overtime"  # Extended rounds


class PlayerRole(Enum):
    """Detected player role based on actions."""
    ENTRY = "entry"        # First into sites
    AWP = "awp"            # Sniper/AWP user
    SUPPORT = "support"    # Utility/flash heavy
    LURKER = "lurker"      # Away from main team
    IGL = "igl"            # In-game leader
    RIFLER = "rifler"      # Main rifler
    FLEX = "flex"          # Flexible


class AttackType(Enum):
    """How attacks are executed."""
    EXEC = "exec"              # Full execute on site
    SEMI_EXECUTE = "semi_exec" # Partial execute
    DOUBLE_STACK = "double_stack"  # Stack heavy one site
    SPLIT = "split"            # Split attack
    VAN_EXEC = "van_exec"      # Short/VAN execution
    LONG_EXEC = "long_exec"    # Long site execution
    ANTI_ECO = "anti_eco"      # Anti-eco push
    LURK_PLAY = "lurk_play"    # Lurk-based play
    UNKNOWN = "unknown"


@dataclass
class RoundPlay:
    """Information about a single round's play."""
    round_num: int
    map_name: str
    t_side: bool  # True if attacking team is T side
    attack_type: AttackType = AttackType.UNKNOWN
    buy_type: str = "unknown"  # full, eco, half-buy, etc
    
    # Execution details
    utility_used: list[str] = field(default_factory=list)  # smoke, flash, nade, molly, etc
    utility_timing: dict[str, float] = field(default_factory=dict)  # when utilities were used (seconds)
    
    # Kills and deaths
    kills: list[dict] = field(default_factory=list)
    deaths: list[dict] = field(default_factory=list)
    
    # Timing data
    first_kill_time: float | None = None
    plant_time: float | None = None
    defuse_time: float | None = None
    
    # Economy
    avg_buy_value: dict[str, int] = field(default_factory=dict)  # team -> total buy value
    
    # Outcome
    round_winner: str = ""  # "CT" or "T"
    round_outcome: str = ""  # "win", "loss", "eco_win", "force_loss", etc
    

@dataclass
class PlayerTacticalStats:
    """Tactical statistics for a player."""
    steam_id: int
    player_name: str
    team: str
    
    # Role detection
    primary_role: PlayerRole = PlayerRole.RIFLER
    role_confidence: float = 0.0
    
    # Execution stats
    opening_kills: int = 0  # Kills in opening phase
    mid_kills: int = 0      # Kills in mid/mid-late
    trade_kills: int = 0    # Kills within 2 seconds of teammate death
    solo_kills: int = 0     # 1v2+ kills
    
    # Utility usage
    utility_flashes_thrown: int = 0
    utility_smokes_thrown: int = 0
    utility_nades_thrown: int = 0
    utility_molly_thrown: int = 0
    
    # Positioning
    default_positions: dict[str, int] = field(default_factory=dict)  # where player usually positions
    lurk_frequency: float = 0.0  # how often away from team
    
    # Damage patterns
    avg_damage_per_kill: float = 0.0
    avg_damage_per_round: float = 0.0
    close_range_kills: int = 0  # Kills within 500 units
    long_range_kills: int = 0   # Kills beyond 1000 units
    
    # Buy value consistency
    buy_value_correlation: float = 0.0  # correlation between buy and kills
    
    # Weaknesses & Strengths
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


@dataclass
class TeamTacticalStats:
    """Tactical statistics for a team."""
    team_name: str
    side_as_t: dict[str, Any] = field(default_factory=dict)  # Stats when T side
    side_as_ct: dict[str, Any] = field(default_factory=dict)  # Stats when CT side
    
    # Play patterns
    most_used_executes: list[tuple[str, int]] = field(default_factory=list)  # (exec_name, count)
    utility_success_rate: float = 0.0  # How often utility plays succeed
    
    # Site preferences
    site_preference_a: int = 0  # How many A hits vs B hits
    site_preference_b: int = 0
    
    # Economy stats
    full_buy_win_rate: float = 0.0
    eco_win_rate: float = 0.0
    force_success_rate: float = 0.0
    
    # Coordination
    coordination_score: float = 0.0  # 0-100, based on trade timing, utility coordination
    timing_consistency: float = 0.0  # How predictable team plays are
    
    # Strengths & Weaknesses
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


class TacticalAnalyzer:
    """Analyze tactical/strategic aspects of a demo."""
    
    def __init__(self, demo_data: Any):
        self.data = demo_data
        self.round_plays: list[RoundPlay] = []
        self.player_stats: dict[int, PlayerTacticalStats] = {}
        self.team_stats: dict[str, TeamTacticalStats] = {}
        
    def analyze(self) -> dict[str, Any]:
        """Perform complete tactical analysis."""
        self._extract_round_plays()
        self._analyze_player_tactics()
        self._analyze_team_tactics()
        
        return {
            "round_plays": self.round_plays,
            "player_tactics": self.player_stats,
            "team_tactics": self.team_stats,
            "summary": self._generate_summary(),
        }
    
    def _extract_round_plays(self) -> None:
        """Extract play information for each round."""
        kills = getattr(self.data, "kills", [])
        damages = getattr(self.data, "damages", [])
        grenades = getattr(self.data, "grenades", [])
        
        # Group by round
        rounds_data = defaultdict(lambda: {
            "kills": [],
            "damages": [],
            "grenades": [],
        })
        
        for kill in kills:
            round_num = kill.get("round_num", 0)
            rounds_data[round_num]["kills"].append(kill)
        
        for dmg in damages:
            round_num = dmg.get("round_num", 0)
            rounds_data[round_num]["damages"].append(dmg)
            
        for nade in grenades:
            round_num = nade.get("round_num", 0)
            rounds_data[round_num]["grenades"].append(nade)
        
        # Analyze each round
        for round_num in sorted(rounds_data.keys()):
            round_data = rounds_data[round_num]
            play = RoundPlay(
                round_num=round_num,
                map_name=getattr(self.data, "map_name", "unknown"),
                t_side=True,  # Will refine based on actual side
                kills=round_data["kills"],
                deaths=round_data["kills"],  # Deaths are inverse of kills
            )
            
            # Analyze utilities
            self._analyze_utilities(play, round_data["grenades"])
            
            # Determine attack type
            play.attack_type = self._determine_attack_type(round_data["kills"])
            
            # Extract timing
            if play.kills:
                kill_times = [k.get("time", 0) for k in play.kills]
                play.first_kill_time = min(kill_times) if kill_times else None
            
            self.round_plays.append(play)
    
    def _analyze_utilities(self, play: RoundPlay, grenades: list[dict]) -> None:
        """Analyze utility usage in a round."""
        utility_types = defaultdict(int)
        
        for nade in grenades:
            nade_type = nade.get("grenade_type", "unknown").lower()
            time = nade.get("time", 0)
            
            if nade_type in ["smoke", "flash", "frag", "molotov"]:
                play.utility_used.append(nade_type)
                play.utility_timing[nade_type] = time
                utility_types[nade_type] += 1
    
    def _determine_attack_type(self, kills: list[dict]) -> AttackType:
        """Determine how attack was executed based on kill pattern."""
        if not kills:
            return AttackType.UNKNOWN
        
        # Simple heuristic: look at kill spread and positions
        kill_times = [k.get("time", 0) for k in kills]
        
        # Distributed kills = execute, clustered = anti-eco
        if kill_times:
            time_spread = max(kill_times) - min(kill_times)
            if time_spread > 30:
                return AttackType.EXEC
            elif len(kills) <= 2:
                return AttackType.ANTI_ECO
        
        return AttackType.UNKNOWN
    
    def _analyze_player_tactics(self) -> None:
        """Analyze tactical stats for each player."""
        player_names = getattr(self.data, "player_names", {})
        
        for steam_id, player_name in player_names.items():
            stats = PlayerTacticalStats(
                steam_id=steam_id,
                player_name=player_name,
                team=self._get_player_team(steam_id),
            )
            
            # Count opening kills
            for play in self.round_plays:
                kills = [k for k in play.kills 
                        if k.get("attacker_steamid") == steam_id]
                
                for kill in kills:
                    time = kill.get("time", 0)
                    if time < 10:
                        stats.opening_kills += 1
                    elif time < 60:
                        stats.mid_kills += 1
            
            # Detect role
            self._detect_player_role(stats)
            
            # Generate insights
            self._generate_player_insights(stats)
            
            self.player_stats[steam_id] = stats
    
    def _get_player_team(self, steam_id: int) -> str:
        """Get player's team."""
        for team, players in [("T", getattr(self.data, "t_players", [])),
                             ("CT", getattr(self.data, "ct_players", []))]:
            if steam_id in players:
                return team
        return "unknown"
    
    def _detect_player_role(self, stats: PlayerTacticalStats) -> None:
        """Detect player's primary role."""
        # Simple role detection
        if stats.opening_kills > stats.mid_kills:
            stats.primary_role = PlayerRole.ENTRY
            stats.role_confidence = 0.7
        else:
            stats.primary_role = PlayerRole.RIFLER
            stats.role_confidence = 0.5
    
    def _generate_player_insights(self, stats: PlayerTacticalStats) -> None:
        """Generate strengths and weaknesses."""
        if stats.opening_kills >= 5:
            stats.strengths.append("Excellent at opening duels")
        else:
            stats.weaknesses.append("Improve opening kill success")
        
        if stats.mid_kills >= 8:
            stats.strengths.append("Strong mid-round fragging")
        
        if not stats.strengths:
            stats.strengths.append("Consistent player")
    
    def _analyze_team_tactics(self) -> None:
        """Analyze team tactical patterns."""
        teams = {}
        for play in self.round_plays:
            # Group plays by team and analyze patterns
            pass
    
    def _generate_summary(self) -> dict[str, Any]:
        """Generate overall tactical summary."""
        return {
            "total_rounds": len(self.round_plays),
            "attack_types": self._attack_type_distribution(),
            "key_insights": self._extract_key_insights(),
        }
    
    def _attack_type_distribution(self) -> dict[str, int]:
        """Count attack type usage."""
        counts = defaultdict(int)
        for play in self.round_plays:
            counts[play.attack_type.value] += 1
        return dict(counts)
    
    def _extract_key_insights(self) -> list[str]:
        """Extract key tactical insights."""
        insights = []
        
        # Look for patterns
        if self.round_plays:
            first_kill_times = [p.first_kill_time for p in self.round_plays 
                              if p.first_kill_time is not None]
            
            if first_kill_times:
                avg_first_kill = np.mean(first_kill_times)
                if avg_first_kill < 15:
                    insights.append("Very aggressive early round play")
                elif avg_first_kill > 40:
                    insights.append("Passive, methodical play style")
        
        # Strongest player insight
        if self.player_stats:
            strongest = max(self.player_stats.values(), 
                          key=lambda p: p.opening_kills)
            if strongest.opening_kills > 0:
                insights.append(
                    f"{strongest.player_name} is the primary entry fragger"
                )
        
        return insights
