"""
Professional Analytics Engine for CS2 Demo Analysis

Implements industry-standard metrics:
- HLTV 2.0 Rating
- KAST% (Kill/Assist/Survived/Traded)
- ADR (Average Damage per Round)
- Trade kill detection
- Clutch detection
- Opening duel analysis
- Multi-kill tracking
- TTD (Time to Damage)
- Crosshair Placement
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import logging
import math

import numpy as np
import pandas as pd

from opensight.parser import DemoData, KillEvent, DamageEvent, safe_int, safe_str, safe_float
from opensight.constants import (
    CS2_TICK_RATE,
    TRADE_WINDOW_SECONDS,
    OPENING_DUEL_WINDOW,
    HLTV_RATING_COEFFICIENTS,
    IMPACT_COEFFICIENTS,
)

logger = logging.getLogger(__name__)

# Import optimized metrics computation
try:
    from opensight.metrics_optimized import (
        MetricType,
        OptimizedMetricsComputer,
        compute_ttd_vectorized,
        compute_cp_vectorized,
        compute_cp_from_dataframe_vectorized,
        set_cache_directory,
    )
    HAS_OPTIMIZED_METRICS = True
except ImportError:
    HAS_OPTIMIZED_METRICS = False
    MetricType = None  # Placeholder
    OptimizedMetricsComputer = None  # Placeholder
    logger.debug("Optimized metrics module not available")

# Import economy and combat modules for integration
try:
    from opensight.economy import EconomyAnalyzer, EconomyStats, PlayerEconomyProfile, BuyType
    HAS_ECONOMY = True
except ImportError:
    HAS_ECONOMY = False
    logger.debug("Economy module not available")

try:
    from opensight.combat import CombatAnalyzer, CombatAnalysisResult, PlayerCombatStats
    HAS_COMBAT = True
except ImportError:
    HAS_COMBAT = False
    logger.debug("Combat module not available")


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure.

    Args:
        value: The value to convert.
        default: The default value to return if conversion fails.

    Returns:
        The converted float value, or the default if conversion fails.
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def compute_kill_positions(match_data) -> list[dict]:
    """
    Compute kill positions for Kill Map visualization.

    Extracts kill data from match_data.kills for plotting on map radar images.
    Uses victim position as the kill location (where the player died).

    Args:
        match_data: MatchData or DemoData object with kills list and player_names dict.

    Returns:
        List of dicts with kill information:
        - attacker_name: Name of the player who got the kill
        - victim_name: Name of the player who died
        - attacker_team: Team of attacker ("CT" or "T")
        - victim_team: Team of victim ("CT" or "T")
        - weapon: Weapon used for the kill
        - is_headshot: Whether it was a headshot
        - x: X coordinate (victim position)
        - y: Y coordinate (victim position)
        - z: Z coordinate (victim position)
        - round_num: Round number when the kill occurred
    """
    kill_positions = []

    # Handle both DemoData (with kills list) and MatchData
    kills = getattr(match_data, 'kills', [])
    player_names = getattr(match_data, 'player_names', {})

    for kill in kills:
        try:
            # Get player names
            attacker_name = player_names.get(kill.attacker_steamid, kill.attacker_name or "Unknown")
            victim_name = player_names.get(kill.victim_steamid, kill.victim_name or "Unknown")

            # Use victim position as the kill location (where the death occurred)
            x = kill.victim_x if kill.victim_x is not None else kill.attacker_x
            y = kill.victim_y if kill.victim_y is not None else kill.attacker_y
            z = kill.victim_z if kill.victim_z is not None else (kill.attacker_z or 0)

            if x is None or y is None:
                continue

            # Determine attacker team from side info
            attacker_team = "Unknown"
            if hasattr(kill, 'attacker_side') and kill.attacker_side:
                side = str(kill.attacker_side).upper()
                if "CT" in side:
                    attacker_team = "CT"
                elif "T" in side and "CT" not in side:
                    attacker_team = "T"

            victim_team = "Unknown"
            if hasattr(kill, 'victim_side') and kill.victim_side:
                side = str(kill.victim_side).upper()
                if "CT" in side:
                    victim_team = "CT"
                elif "T" in side and "CT" not in side:
                    victim_team = "T"

            kill_positions.append({
                "attacker_name": attacker_name,
                "victim_name": victim_name,
                "attacker_team": attacker_team,
                "victim_team": victim_team,
                "weapon": kill.weapon or "Unknown",
                "is_headshot": bool(kill.headshot),
                "x": float(x),
                "y": float(y),
                "z": float(z) if z else 0.0,
                "round_num": kill.round_num or 0,
            })
        except Exception as e:
            logger.debug(f"Error extracting kill position: {e}")
            continue

    logger.info(f"Computed {len(kill_positions)} kill positions for Kill Map")
    return kill_positions


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
    is_prefire: bool
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
class OpeningDuelStats:
    """Opening duel statistics for entry fragging analysis.

    Entry duels are the first kills of each round, filtered by context:
    - T-side: Attacker entering/pushing towards a bombsite
    - CT-side: Defender holding against T push

    Metrics:
    - win_rate: Entry Success % - How often you win opening duels
    - entry_kills_total: Raw count of entry kills
    - entry_ttd_values: TTD specifically for entry kills (reaction time on first contact)
    """
    wins: int = 0
    losses: int = 0
    attempts: int = 0
    # Entry-specific tracking
    entry_ttd_values: list = None  # TTD values for entry kills specifically
    t_side_entries: int = 0  # Entry kills while on T side (aggressive)
    ct_side_entries: int = 0  # Entry kills while on CT side (defensive)

    def __post_init__(self):
        if self.entry_ttd_values is None:
            self.entry_ttd_values = []

    @property
    def win_rate(self) -> float:
        """Entry Success % - percentage of opening duels won."""
        return round(self.wins / self.attempts * 100, 1) if self.attempts > 0 else 0.0

    @property
    def entry_ttd_median_ms(self) -> Optional[float]:
        """Median TTD for entry kills - measures reaction time on first contact."""
        if self.entry_ttd_values:
            return float(np.median(self.entry_ttd_values))
        return None

    @property
    def entry_ttd_mean_ms(self) -> Optional[float]:
        """Mean TTD for entry kills."""
        if self.entry_ttd_values:
            return float(np.mean(self.entry_ttd_values))
        return None


@dataclass
class TradeStats:
    """Trade kill statistics - measures how well you avenge teammates.

    Trade Kill: When a teammate is killed, check if you killed the enemy
    who killed them within 5 seconds.

    Metrics:
    - trade_rate: Trade Kill % - How often you avenge teammates when opportunity arises
    - deaths_traded_rate: How often your deaths were avenged by teammates
    """
    kills_traded: int = 0  # Kills that avenged a teammate (you killed the enemy who killed teammate)
    deaths_traded: int = 0  # Your deaths that were avenged by teammates
    trade_attempts: int = 0  # Opportunities to trade (teammate deaths where you could trade)
    failed_trades: int = 0  # Trade opportunities where you didn't get the trade

    @property
    def trade_rate(self) -> float:
        """Trade Kill % - How often you avenge your teammates when given the opportunity."""
        return round(self.kills_traded / self.trade_attempts * 100, 1) if self.trade_attempts > 0 else 0.0

    @property
    def deaths_traded_rate(self) -> float:
        """Percentage of your deaths that were avenged by teammates."""
        total_deaths = self.deaths_traded + self.failed_trades
        return round(self.deaths_traded / total_deaths * 100, 1) if total_deaths > 0 else 0.0


@dataclass
class ClutchStats:
    """Clutch situation statistics - 1vX scenarios where player is last alive.

    Clutch: Rounds where you were the last player alive on your team
    facing one or more enemies. Wins are determined by round outcome.

    Metrics:
    - win_rate: Clutch Won % - Overall clutch success rate
    - v1_win_rate: Clutch 1v1 % - Success rate in 1v1 clutches
    - v2_win_rate: Clutch 1v2 % - Success rate in 1v2 clutches
    """
    total_situations: int = 0
    total_wins: int = 0
    v1_attempts: int = 0
    v1_wins: int = 0
    v2_attempts: int = 0
    v2_wins: int = 0
    v3_attempts: int = 0
    v3_wins: int = 0
    v4_attempts: int = 0
    v4_wins: int = 0
    v5_attempts: int = 0
    v5_wins: int = 0

    @property
    def win_rate(self) -> float:
        """Clutch Won % - Overall clutch success rate across all 1vX situations."""
        return round(self.total_wins / self.total_situations * 100, 1) if self.total_situations > 0 else 0.0

    @property
    def v1_win_rate(self) -> float:
        """Clutch 1v1 % - Success rate in 1v1 clutch situations."""
        return round(self.v1_wins / self.v1_attempts * 100, 1) if self.v1_attempts > 0 else 0.0

    @property
    def v2_win_rate(self) -> float:
        """Clutch 1v2 % - Success rate in 1v2 clutch situations."""
        return round(self.v2_wins / self.v2_attempts * 100, 1) if self.v2_attempts > 0 else 0.0

    @property
    def v3_win_rate(self) -> float:
        """Clutch 1v3 % - Success rate in 1v3 clutch situations."""
        return round(self.v3_wins / self.v3_attempts * 100, 1) if self.v3_attempts > 0 else 0.0

    @property
    def v4_win_rate(self) -> float:
        """Clutch 1v4 % - Success rate in 1v4 clutch situations."""
        return round(self.v4_wins / self.v4_attempts * 100, 1) if self.v4_attempts > 0 else 0.0

    @property
    def v5_win_rate(self) -> float:
        """Clutch 1v5 % - Success rate in 1v5 clutch situations."""
        return round(self.v5_wins / self.v5_attempts * 100, 1) if self.v5_attempts > 0 else 0.0

    # Alias properties for API compatibility
    @property
    def situations_1v1(self) -> int:
        return self.v1_attempts

    @situations_1v1.setter
    def situations_1v1(self, value: int) -> None:
        self.v1_attempts = value

    @property
    def wins_1v1(self) -> int:
        return self.v1_wins

    @wins_1v1.setter
    def wins_1v1(self, value: int) -> None:
        self.v1_wins = value

    @property
    def situations_1v2(self) -> int:
        return self.v2_attempts

    @situations_1v2.setter
    def situations_1v2(self, value: int) -> None:
        self.v2_attempts = value

    @property
    def wins_1v2(self) -> int:
        return self.v2_wins

    @wins_1v2.setter
    def wins_1v2(self, value: int) -> None:
        self.v2_wins = value

    @property
    def situations_1v3(self) -> int:
        return self.v3_attempts

    @situations_1v3.setter
    def situations_1v3(self, value: int) -> None:
        self.v3_attempts = value

    @property
    def wins_1v3(self) -> int:
        return self.v3_wins

    @wins_1v3.setter
    def wins_1v3(self, value: int) -> None:
        self.v3_wins = value

    @property
    def situations_1v4(self) -> int:
        return self.v4_attempts

    @situations_1v4.setter
    def situations_1v4(self, value: int) -> None:
        self.v4_attempts = value

    @property
    def wins_1v4(self) -> int:
        return self.v4_wins

    @wins_1v4.setter
    def wins_1v4(self, value: int) -> None:
        self.v4_wins = value

    @property
    def situations_1v5(self) -> int:
        return self.v5_attempts

    @situations_1v5.setter
    def situations_1v5(self, value: int) -> None:
        self.v5_attempts = value

    @property
    def wins_1v5(self) -> int:
        return self.v5_wins

    @wins_1v5.setter
    def wins_1v5(self, value: int) -> None:
        self.v5_wins = value


@dataclass
class MultiKillStats:
    """Multi-kill round statistics."""
    rounds_with_1k: int = 0
    rounds_with_2k: int = 0
    rounds_with_3k: int = 0
    rounds_with_4k: int = 0
    rounds_with_5k: int = 0

    @property
    def total_multi_kill_rounds(self) -> int:
        return self.rounds_with_2k + self.rounds_with_3k + self.rounds_with_4k + self.rounds_with_5k


@dataclass
class WeaponStats:
    """Per-weapon statistics for a player."""
    weapon: str
    kills: int = 0
    headshots: int = 0
    damage: int = 0
    shots_fired: int = 0
    shots_hit: int = 0

    @property
    def headshot_percentage(self) -> float:
        """Percentage of kills that were headshots."""
        if self.kills <= 0:
            return 0.0
        return round(self.headshots / self.kills * 100, 1)

    @property
    def accuracy(self) -> float:
        """Shot accuracy (hits / shots fired)."""
        if self.shots_fired <= 0:
            return 0.0
        return round(self.shots_hit / self.shots_fired * 100, 1)

    @property
    def damage_per_shot(self) -> float:
        """Average damage per shot fired."""
        if self.shots_fired <= 0:
            return 0.0
        return round(self.damage / self.shots_fired, 1)


@dataclass
class UtilityStats:
    """Utility usage statistics."""
    flashbangs_thrown: int = 0
    smokes_thrown: int = 0
    he_thrown: int = 0
    molotovs_thrown: int = 0
    enemies_flashed: int = 0
    teammates_flashed: int = 0
    flash_assists: int = 0
    he_damage: int = 0
    he_team_damage: int = 0
    molotov_damage: int = 0
    molotov_team_damage: int = 0
    total_blind_time: float = 0.0  # Total seconds enemies were blinded

    @property
    def total_utility(self) -> int:
        return self.flashbangs_thrown + self.smokes_thrown + self.he_thrown + self.molotovs_thrown

    @property
    def enemies_flashed_per_flash(self) -> float:
        """Average enemies flashed per flashbang thrown."""
        if self.flashbangs_thrown <= 0:
            return 0.0
        return self.enemies_flashed / self.flashbangs_thrown

    @property
    def teammates_flashed_per_flash(self) -> float:
        """Average teammates flashed per flashbang thrown (bad)."""
        if self.flashbangs_thrown <= 0:
            return 0.0
        return self.teammates_flashed / self.flashbangs_thrown

    @property
    def avg_blind_time(self) -> float:
        """Average blind time per enemy flashed."""
        if self.enemies_flashed <= 0:
            return 0.0
        return self.total_blind_time / self.enemies_flashed

    @property
    def he_damage_per_nade(self) -> float:
        """Average HE damage per grenade."""
        if self.he_thrown <= 0:
            return 0.0
        return self.he_damage / self.he_thrown

    # Backwards compatibility alias
    @property
    def molotov_thrown(self) -> int:
        return self.molotovs_thrown

    @molotov_thrown.setter
    def molotov_thrown(self, value: int) -> None:
        self.molotovs_thrown = value


@dataclass
class UtilityMetrics:
    """
    Per-player utility usage metrics (Scope.gg style).

    This dataclass provides a comprehensive view of each player's utility usage,
    including grenade counts, utility damage, and flash effectiveness.
    """
    player_name: str
    player_steamid: int
    team: str = ""

    # Grenade counts
    smokes_thrown: int = 0
    flashes_thrown: int = 0
    he_thrown: int = 0
    molotovs_thrown: int = 0

    # Utility effectiveness
    total_utility_damage: float = 0.0  # HE + Molotov damage to enemies
    flashes_enemies_total: int = 0     # Number of enemy players flashed (>1.1s blind)
    flashes_teammates_total: int = 0   # Number of teammates flashed (mistake tracking)
    flash_assists: int = 0             # Kills on enemies player flashed
    total_blind_time: float = 0.0      # Total seconds enemies were blinded

    # Per-grenade efficiency metrics
    he_damage: int = 0
    molotov_damage: int = 0

    @property
    def total_utility_thrown(self) -> int:
        """Total grenades thrown."""
        return self.smokes_thrown + self.flashes_thrown + self.he_thrown + self.molotovs_thrown

    @property
    def enemies_flashed_per_flash(self) -> float:
        """Average enemies flashed per flashbang."""
        if self.flashes_thrown <= 0:
            return 0.0
        return self.flashes_enemies_total / self.flashes_thrown

    @property
    def he_damage_per_nade(self) -> float:
        """Average HE damage per grenade."""
        if self.he_thrown <= 0:
            return 0.0
        return self.he_damage / self.he_thrown

    @property
    def molotov_damage_per_nade(self) -> float:
        """Average molotov damage per grenade."""
        if self.molotovs_thrown <= 0:
            return 0.0
        return self.molotov_damage / self.molotovs_thrown

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "player_name": self.player_name,
            "player_steamid": str(self.player_steamid),
            "team": self.team,
            "smokes_thrown": self.smokes_thrown,
            "flashes_thrown": self.flashes_thrown,
            "he_thrown": self.he_thrown,
            "molotovs_thrown": self.molotovs_thrown,
            "total_utility_thrown": self.total_utility_thrown,
            "total_utility_damage": round(self.total_utility_damage, 1),
            "flashes_enemies_total": self.flashes_enemies_total,
            "flashes_teammates_total": self.flashes_teammates_total,
            "flash_assists": self.flash_assists,
            "total_blind_time": round(self.total_blind_time, 2),
            "enemies_flashed_per_flash": round(self.enemies_flashed_per_flash, 2),
            "he_damage": self.he_damage,
            "he_damage_per_nade": round(self.he_damage_per_nade, 1),
            "molotov_damage": self.molotov_damage,
            "molotov_damage_per_nade": round(self.molotov_damage_per_nade, 1),
        }


@dataclass
class SideStats:
    """Per-side (CT/T) statistics."""
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    damage: int = 0
    rounds_played: int = 0

    @property
    def kd_ratio(self) -> float:
        return round(self.kills / self.deaths, 2) if self.deaths > 0 else float(self.kills)

    @property
    def adr(self) -> float:
        return round(self.damage / self.rounds_played, 1) if self.rounds_played > 0 else 0.0


@dataclass
class MistakesStats:
    """Mistake tracking (Scope.gg style)."""
    team_kills: int = 0
    team_damage: int = 0
    teammates_flashed: int = 0
    suicides: int = 0

    @property
    def total_mistakes(self) -> int:
        """Total mistake score (weighted)."""
        # Team kills are worst, team damage next, flashes last
        return (self.team_kills * 10) + (self.team_damage // 10) + self.teammates_flashed + (self.suicides * 5)


@dataclass
class AdvancedMetrics:
    """
    Advanced coaching metrics inspired by Scope.gg.

    These metrics provide coach-level insights into:
    - Opening duels: Who gets the first kill of each round
    - Clutches: 1vX situations and win rates
    - Trades: How quickly teammates avenge deaths

    Useful for identifying:
    - Entry fraggers (high opening_kills)
    - Clutch players (high clutch_win_rate)
    - Team players (high trade_success_rate)
    """
    player_name: str
    steam_id: int = 0

    # Opening duel metrics
    opening_kills: int = 0  # First kill of a round
    opening_deaths: int = 0  # First death of a round
    opening_success_rate: float = 0.0  # opening_kills / (opening_kills + opening_deaths)

    # Clutch metrics (1vX situations)
    clutches_1vx_attempted: int = 0  # Total clutch situations faced
    clutches_1vx_won: int = 0  # Total clutch situations won
    clutch_win_rate: float = 0.0  # clutches_won / clutches_attempted

    # Trade metrics
    trade_kills: int = 0  # Kills that avenged a teammate
    trade_attempts: int = 0  # Opportunities to trade (teammate died, enemy visible)
    trade_success_rate: float = 0.0  # trade_kills / trade_attempts

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "player_name": self.player_name,
            "steam_id": str(self.steam_id),
            "opening_kills": self.opening_kills,
            "opening_deaths": self.opening_deaths,
            "opening_success_rate": round(self.opening_success_rate, 1),
            "clutches_1vx_attempted": self.clutches_1vx_attempted,
            "clutches_1vx_won": self.clutches_1vx_won,
            "clutch_win_rate": round(self.clutch_win_rate, 1),
            "trade_kills": self.trade_kills,
            "trade_attempts": self.trade_attempts,
            "trade_success_rate": round(self.trade_success_rate, 1),
        }


@dataclass
class LurkStats:
    """Lurk statistics from State Machine."""
    kills: int = 0
    deaths: int = 0
    rounds_lurking: int = 0


@dataclass
class PlayerMatchStats:
    """Complete match statistics for a player."""
    # Identity
    steam_id: int
    name: str
    team: str  # "CT", "T", or side they played more

    # Basic stats
    kills: int
    deaths: int
    assists: int
    headshots: int
    total_damage: int
    rounds_played: int

    # Component stats
    opening_duels: OpeningDuelStats = field(default_factory=OpeningDuelStats)
    trades: TradeStats = field(default_factory=TradeStats)
    clutches: ClutchStats = field(default_factory=ClutchStats)
    multi_kills: MultiKillStats = field(default_factory=MultiKillStats)
    utility: UtilityStats = field(default_factory=UtilityStats)

    # Side-based stats (Leetify style)
    ct_stats: SideStats = field(default_factory=SideStats)
    t_stats: SideStats = field(default_factory=SideStats)

    # Mistakes tracking (Scope.gg style)
    mistakes: MistakesStats = field(default_factory=MistakesStats)

    # Lurk stats (State Machine)
    lurk: LurkStats = field(default_factory=LurkStats)

    # KAST tracking
    kast_rounds: int = 0  # Rounds with Kill/Assist/Survived/Traded
    rounds_survived: int = 0

    # State Machine enhanced stats
    effective_flashes: int = 0      # Flashes > 2.0 seconds blind duration
    ineffective_flashes: int = 0    # Flashes < 2.0 seconds
    utility_adr: float = 0.0        # HE + Molotov damage per round

    # Weapon breakdown
    weapon_kills: dict = field(default_factory=dict)

    # TTD stats
    ttd_values: list = field(default_factory=list)
    prefire_count: int = 0

    # CP stats
    cp_values: list = field(default_factory=list)

    # Accuracy stats (Leetify style)
    shots_fired: int = 0
    shots_hit: int = 0
    headshot_hits: int = 0  # Hits to head (not just kills)

    # Economy stats (integrated from economy module)
    avg_equipment_value: float = 0.0
    eco_rounds: int = 0
    force_rounds: int = 0
    full_buy_rounds: int = 0
    damage_per_dollar: float = 0.0
    kills_per_dollar: float = 0.0

    # Combat stats (integrated from combat module)
    trade_kill_time_avg_ms: float = 0.0
    untraded_deaths: int = 0

    # Derived properties
    @property
    def kd_ratio(self) -> float:
        return round(self.kills / self.deaths, 2) if self.deaths > 0 else float(self.kills)

    @property
    def kd_diff(self) -> int:
        return self.kills - self.deaths

    @property
    def adr(self) -> float:
        return round(self.total_damage / self.rounds_played, 1) if self.rounds_played > 0 else 0.0

    @property
    def headshot_percentage(self) -> float:
        return round(self.headshots / self.kills * 100, 1) if self.kills > 0 else 0.0

    @property
    def kast_percentage(self) -> float:
        return round(self.kast_rounds / self.rounds_played * 100, 1) if self.rounds_played > 0 else 0.0

    @property
    def survival_rate(self) -> float:
        return round(self.rounds_survived / self.rounds_played * 100, 1) if self.rounds_played > 0 else 0.0

    @property
    def kills_per_round(self) -> float:
        return round(self.kills / self.rounds_played, 2) if self.rounds_played > 0 else 0.0

    @property
    def deaths_per_round(self) -> float:
        return round(self.deaths / self.rounds_played, 2) if self.rounds_played > 0 else 0.0

    @property
    def assists_per_round(self) -> float:
        return round(self.assists / self.rounds_played, 2) if self.rounds_played > 0 else 0.0

    @property
    def multi_kill_round_rate(self) -> float:
        """Percentage of rounds with 2+ kills."""
        if self.rounds_played == 0:
            return 0.0
        return round(self.multi_kills.total_multi_kill_rounds / self.rounds_played * 100, 1)

    @property
    def impact_rating(self) -> float:
        """
        Impact rating component for HLTV 2.0.
        Impact = 2.13*KPR + 0.42*APR - 0.41 + clutch/multikill bonuses
        """
        base = (
            IMPACT_COEFFICIENTS["kpr"] * self.kills_per_round +
            IMPACT_COEFFICIENTS["apr"] * self.assists_per_round +
            IMPACT_COEFFICIENTS["base"]
        )
        # Add clutch bonus
        clutch_bonus = self.clutches.total_wins * 0.1
        # Add multi-kill bonus
        mk_bonus = (
            self.multi_kills.rounds_with_3k * 0.1 +
            self.multi_kills.rounds_with_4k * 0.2 +
            self.multi_kills.rounds_with_5k * 0.3
        )
        return round(base + clutch_bonus + mk_bonus, 3)

    @property
    def hltv_rating(self) -> float:
        """
        HLTV 2.0 Rating (approximated).

        Rating = 0.0073*KAST + 0.3591*KPR + (-0.5329)*DPR +
                 0.2372*Impact + 0.0032*ADR + 0.1587*RMK
        """
        kast = self.kast_percentage
        kpr = self.kills_per_round
        dpr = self.deaths_per_round
        impact = self.impact_rating
        adr = self.adr
        rmk = self.multi_kill_round_rate

        rating = (
            HLTV_RATING_COEFFICIENTS["kast"] * kast +
            HLTV_RATING_COEFFICIENTS["kpr"] * kpr +
            HLTV_RATING_COEFFICIENTS["dpr"] * dpr +
            HLTV_RATING_COEFFICIENTS["impact"] * impact +
            HLTV_RATING_COEFFICIENTS["adr"] * adr +
            HLTV_RATING_COEFFICIENTS["rmk"] * rmk
        )

        return round(max(0.0, rating), 2)

    # TTD properties
    @property
    def ttd_median_ms(self) -> Optional[float]:
        return float(np.median(self.ttd_values)) if self.ttd_values else None

    @property
    def ttd_mean_ms(self) -> Optional[float]:
        return float(np.mean(self.ttd_values)) if self.ttd_values else None

    # CP properties
    @property
    def cp_median_error_deg(self) -> Optional[float]:
        return float(np.median(self.cp_values)) if self.cp_values else None

    @property
    def cp_mean_error_deg(self) -> Optional[float]:
        return float(np.mean(self.cp_values)) if self.cp_values else None

    # Accuracy properties (Leetify style)
    @property
    def accuracy(self) -> float:
        """Overall accuracy - shots hit / shots fired."""
        return round(self.shots_hit / self.shots_fired * 100, 1) if self.shots_fired > 0 else 0.0

    @property
    def head_hit_rate(self) -> float:
        """% of hits that were headshots (different from HS kill %)."""
        return round(self.headshot_hits / self.shots_hit * 100, 1) if self.shots_hit > 0 else 0.0

    # Utility Rating (Leetify style composite)
    @property
    def utility_quantity_rating(self) -> float:
        """
        Leetify-style Utility Quantity Rating.
        Based on utility thrown vs expected (3 per round).
        Uses x^(2/3) scaling, max 100.
        """
        total_utility = (
            self.utility.flashbangs_thrown +
            self.utility.he_thrown +
            self.utility.molotovs_thrown +
            self.utility.smokes_thrown
        )
        expected = 3.0 * self.rounds_played
        if expected <= 0:
            return 0.0
        ratio = min(total_utility / expected, 1.0)  # Cap at 1.0
        # Apply x^(2/3) scaling
        scaled = ratio ** (2/3)
        return round(scaled * 100, 1)

    @property
    def utility_quality_rating(self) -> float:
        """
        Leetify-style Utility Quality Rating.
        Based on flash effectiveness, HE damage, etc.
        """
        score = 50.0  # Start at average

        # Flash quality (0-30 points)
        if self.utility.flashbangs_thrown > 0:
            # Enemies flashed per flash (0.5 = average, 1.0+ = good)
            flash_score = min(self.utility.enemies_flashed_per_flash / 0.5, 2.0) * 15
            # Penalize teammate flashes
            flash_score -= self.utility.teammates_flashed_per_flash * 10
            score += max(flash_score - 15, -15)  # -15 to +15

        # HE quality (0-20 points)
        if self.utility.he_thrown > 0:
            # 30 damage per HE is good
            he_score = min(self.utility.he_damage_per_nade / 30, 2.0) * 10
            # Penalize team damage
            if self.utility.he_team_damage > 0:
                he_score -= min(self.utility.he_team_damage / 20, 10)
            score += max(he_score - 10, -10)  # -10 to +10

        return round(max(0, min(100, score)), 1)

    @property
    def utility_rating(self) -> float:
        """
        Leetify-style overall Utility Rating.
        Geometric mean of quantity and quality ratings.
        """
        quantity = self.utility_quantity_rating
        quality = self.utility_quality_rating
        if quantity <= 0 or quality <= 0:
            return 0.0
        # Geometric mean
        return round(math.sqrt(quantity * quality), 1)

    # Aim Rating (Leetify style composite)
    @property
    def aim_rating(self) -> float:
        """
        Leetify-style Aim Rating composite.
        Based on: TTD, Crosshair Placement, HS%, accuracy.
        Score is 0-100 where 50 is average.
        """
        score = 50.0  # Start at average

        # TTD component (faster = better, 400ms is average)
        if self.ttd_median_ms:
            # Lower is better: 200ms = +20, 400ms = 0, 600ms = -20
            ttd_delta = (400 - self.ttd_median_ms) / 10
            score += max(-20, min(20, ttd_delta))

        # Crosshair Placement component (lower error = better, 10 deg is average)
        if self.cp_median_error_deg:
            # Lower is better: 5 deg = +15, 10 deg = 0, 15 deg = -15
            cp_delta = (10 - self.cp_median_error_deg) * 3
            score += max(-15, min(15, cp_delta))

        # Headshot % component (25% is average)
        hs_delta = (self.headshot_percentage - 25) / 2
        score += max(-10, min(10, hs_delta))

        # Prefire bonus (shows game sense)
        if self.kills > 0:
            prefire_rate = self.prefire_count / self.kills * 100
            score += min(prefire_rate * 2, 5)  # Up to +5 for prefires

        return round(max(0, min(100, score)), 1)

    @property
    def entry_success_rate(self) -> float:
        """Entry Success % - percentage of opening duels won."""
        return self.opening_duels.win_rate

    @property
    def entry_kills_per_round(self) -> float:
        """Entry Kills per Round - average number of opening kills per round."""
        return round(self.opening_duels.wins / self.rounds_played, 2) if self.rounds_played > 0 else 0.0

    @property
    def entry_ttd(self) -> Optional[float]:
        """Entry TTD - median time to damage on opening kills (ms)."""
        return self.opening_duels.entry_ttd_median_ms

    @property
    def trade_kill_rate(self) -> float:
        """Trade Kill % - how often you avenge your teammates."""
        return self.trades.trade_rate

    @property
    def clutch_win_rate(self) -> float:
        """Clutch Won % - overall success rate in 1vX situations."""
        return self.clutches.win_rate

    @property
    def clutch_1v1_rate(self) -> float:
        """Clutch 1v1 % - success rate in 1v1 clutches."""
        return self.clutches.v1_win_rate

    @property
    def clutch_1v2_rate(self) -> float:
        """Clutch 1v2 % - success rate in 1v2 clutches."""
        return self.clutches.v2_win_rate


@dataclass
class RoundTimeline:
    """Timeline entry for a single round."""
    round_num: int
    winner: str  # "CT" or "T"
    win_reason: str  # "elimination", "bomb_exploded", "bomb_defused", "time"
    ct_score: int
    t_score: int
    first_kill_player: str = ""
    first_death_player: str = ""
    mvp_player: str = ""
    buy_type_ct: str = ""
    buy_type_t: str = ""
    clutch_player: str = ""
    clutch_scenario: str = ""


@dataclass
class KillMatrixEntry:
    """Entry in kill matrix showing who killed whom."""
    attacker_name: str
    victim_name: str
    count: int
    weapons: list[str] = field(default_factory=list)


@dataclass
class MatchAnalysis:
    """Complete analysis of a match."""
    players: dict[int, PlayerMatchStats]
    team1_score: int
    team2_score: int
    total_rounds: int
    map_name: str

    # Enhanced data (integrated from other modules)
    round_timeline: list[RoundTimeline] = field(default_factory=list)
    kill_matrix: list[KillMatrixEntry] = field(default_factory=list)
    team_trade_rates: dict[str, float] = field(default_factory=dict)
    team_opening_rates: dict[str, float] = field(default_factory=dict)

    # Position data for heatmaps (list of {x, y, player, event_type})
    kill_positions: list[dict] = field(default_factory=list)
    death_positions: list[dict] = field(default_factory=list)

    # Grenade trajectory data for utility visualization
    grenade_positions: list[dict] = field(default_factory=list)
    grenade_team_stats: dict[str, dict] = field(default_factory=dict)

    # AI Coaching insights
    coaching_insights: list[dict] = field(default_factory=list)

    # Weapon-specific statistics per player
    # Key: player name, Value: list of WeaponStats
    weapon_stats: dict[str, list[WeaponStats]] = field(default_factory=dict)

    def get_leaderboard(self, sort_by: str = "hltv_rating") -> list[PlayerMatchStats]:
        """Get players sorted by specified metric (descending)."""
        players_list = list(self.players.values())
        if sort_by == "hltv_rating":
            return sorted(players_list, key=lambda p: p.hltv_rating, reverse=True)
        elif sort_by == "kills":
            return sorted(players_list, key=lambda p: p.kills, reverse=True)
        elif sort_by == "adr":
            return sorted(players_list, key=lambda p: p.adr, reverse=True)
        elif sort_by == "kast":
            return sorted(players_list, key=lambda p: p.kast_percentage, reverse=True)
        return players_list

    def get_mvp(self) -> Optional[PlayerMatchStats]:
        """Get match MVP (highest rated player)."""
        leaderboard = self.get_leaderboard()
        return leaderboard[0] if leaderboard else None

    def get_kill_matrix_for_player(self, player_name: str) -> dict[str, int]:
        """Get kills this player got against each opponent."""
        return {e.victim_name: e.count for e in self.kill_matrix if e.attacker_name == player_name}

    def get_death_matrix_for_player(self, player_name: str) -> dict[str, int]:
        """Get deaths this player suffered from each opponent."""
        return {e.attacker_name: e.count for e in self.kill_matrix if e.victim_name == player_name}


class DemoAnalyzer:
    """Analyzer for computing professional-grade metrics from parsed demo data.

    Supports configurable metrics computation for performance optimization.
    When only specific metrics are needed (e.g., just KD ratio), you can skip
    expensive computations like TTD and CP.

    Usage:
        # Full analysis (default)
        analyzer = DemoAnalyzer(demo_data)
        result = analyzer.analyze()

        # Only basic metrics (faster)
        analyzer = DemoAnalyzer(demo_data, metrics="basic")
        result = analyzer.analyze()

        # Specific metrics
        analyzer = DemoAnalyzer(demo_data, metrics=["ttd", "cp", "kd"])
        result = analyzer.analyze()
    """

    TICK_RATE = CS2_TICK_RATE
    MS_PER_TICK = 1000 / TICK_RATE

    # TTD filtering thresholds
    TTD_MIN_MS = 0
    TTD_MAX_MS = 1500

    # Column name variations
    ROUND_COLS = ["round_num", "total_rounds_played", "round"]
    ATT_ID_COLS = ["attacker_steamid", "attacker_steam_id"]
    VIC_ID_COLS = ["victim_steamid", "user_steamid", "victim_steam_id"]
    ATT_SIDE_COLS = ["attacker_side", "attacker_team_name", "attacker_team"]
    VIC_SIDE_COLS = ["victim_side", "user_team_name", "victim_team"]

    # Available metric categories
    METRIC_CATEGORIES = {
        "basic": ["kd", "adr", "headshots"],
        "kast": ["kast", "survival"],
        "ttd": ["ttd", "prefire"],
        "cp": ["crosshair_placement"],
        "trades": ["trade_kills", "trade_deaths"],
        "opening": ["opening_duels"],
        "multi_kills": ["multi_kill_rounds"],
        "clutches": ["clutch_situations"],
        "utility": ["utility_usage", "flash_effectiveness"],
        "accuracy": ["shots_fired", "shots_hit", "accuracy_percent"],
        "economy": ["equipment_value", "damage_per_dollar"],
        "sides": ["ct_stats", "t_stats"],
        "mistakes": ["team_damage", "team_kills"],
    }

    def __init__(
        self,
        demo_data: DemoData,
        metrics: Optional[str | list[str]] = None,
        use_cache: bool = True,
        use_optimized: bool = True
    ):
        """
        Initialize analyzer.

        Args:
            demo_data: Parsed demo data from DemoParser
            metrics: Which metrics to compute. Options:
                - None or "full": Compute all metrics (default)
                - "basic": Only basic stats (KD, ADR, HS%)
                - "advanced": Basic + TTD, CP, trades, opening duels
                - List of specific categories: ["ttd", "cp", "trades"]
            use_cache: Whether to use metrics caching (default True)
            use_optimized: Whether to use vectorized implementations (default True)
        """
        self.data = demo_data
        self._ttd_results: list[TTDResult] = []
        self._cp_results: list[CrosshairPlacementResult] = []
        self._players: dict[int, PlayerMatchStats] = {}
        # Cache column lookups
        self._round_col: Optional[str] = None
        self._att_id_col: Optional[str] = None
        self._vic_id_col: Optional[str] = None
        self._att_side_col: Optional[str] = None
        self._vic_side_col: Optional[str] = None

        # Metrics configuration
        self._use_cache = use_cache
        self._use_optimized = use_optimized and HAS_OPTIMIZED_METRICS
        self._metrics_computer: Optional[OptimizedMetricsComputer] = None
        self._requested_metrics = self._parse_metrics_config(metrics)

    def _parse_metrics_config(self, metrics: Optional[str | list[str]]) -> set[str]:
        """Parse metrics configuration into a set of metric categories."""
        if metrics is None or metrics == "full":
            return set(self.METRIC_CATEGORIES.keys())

        if metrics == "basic":
            return {"basic", "kast", "multi_kills"}

        if metrics == "advanced":
            return {"basic", "kast", "ttd", "cp", "trades", "opening", "multi_kills", "utility"}

        if isinstance(metrics, str):
            return {metrics}

        return set(metrics)

    def _find_col(self, df: pd.DataFrame, options: list[str]) -> Optional[str]:
        """Find first matching column from options."""
        for col in options:
            if col in df.columns:
                return col
        return None

    def _match_steamid(self, df: pd.DataFrame, col: str, steam_id: int) -> pd.DataFrame:
        """Match steamid handling type differences (int vs float)."""
        try:
            # Convert both to float for comparison to handle int/float mismatch
            return df[df[col].astype(float) == float(steam_id)]
        except (ValueError, TypeError):
            # Fallback to direct comparison
            return df[df[col] == steam_id]

    def _init_column_cache(self) -> None:
        """Initialize column name cache for kills DataFrame."""
        kills_df = self.data.kills_df
        if kills_df.empty:
            return
        self._round_col = self._find_col(kills_df, self.ROUND_COLS)
        self._att_id_col = self._find_col(kills_df, self.ATT_ID_COLS)
        self._vic_id_col = self._find_col(kills_df, self.VIC_ID_COLS)
        self._att_side_col = self._find_col(kills_df, self.ATT_SIDE_COLS)
        self._vic_side_col = self._find_col(kills_df, self.VIC_SIDE_COLS)
        logger.info(f"Column cache: round={self._round_col}, att_id={self._att_id_col}, vic_id={self._vic_id_col}")

    def analyze(self) -> MatchAnalysis:
        """Run full analysis and return match analysis.

        Respects the metrics configuration set in __init__.
        Only computes requested metrics for better performance.
        """
        logger.info(f"Starting professional analysis (metrics: {self._requested_metrics})...")

        # Initialize column name cache
        self._init_column_cache()

        # Initialize player stats (always needed)
        self._init_player_stats()

        # Calculate basic stats (always needed)
        self._calculate_basic_stats()

        # Initialize optimized metrics computer if using optimized implementations
        if self._use_optimized:
            self._metrics_computer = OptimizedMetricsComputer(
                self.data,
                use_cache=self._use_cache
            )

        # Calculate multi-kill rounds
        if "multi_kills" in self._requested_metrics or "basic" in self._requested_metrics:
            self._calculate_multi_kills()

        # Detect opening duels
        if "opening" in self._requested_metrics:
            self._detect_opening_duels()

        # Detect trade kills
        if "trades" in self._requested_metrics:
            self._detect_trades()

        # Detect clutches
        if "clutches" in self._requested_metrics:
            self._detect_clutches()

        # Calculate KAST
        if "kast" in self._requested_metrics:
            self._calculate_kast()

        # Compute TTD (using optimized vectorized implementation)
        if "ttd" in self._requested_metrics:
            self._compute_ttd()

        # Compute crosshair placement (using optimized vectorized implementation)
        if "cp" in self._requested_metrics:
            self._compute_crosshair_placement()

        # Calculate side-based stats (CT vs T)
        if "sides" in self._requested_metrics:
            self._calculate_side_stats()

        # Calculate utility stats
        if "utility" in self._requested_metrics:
            self._calculate_utility_stats()

        # Calculate accuracy stats (from weapon_fire events)
        if "accuracy" in self._requested_metrics:
            self._calculate_accuracy_stats()

        # Calculate mistakes
        if "mistakes" in self._requested_metrics:
            self._calculate_mistakes()

        # Run State Machine for pro-level analytics (Entry/Trade/Lurk)
        if any(m in self._requested_metrics for m in ["trades", "opening", "utility"]):
            self._run_state_machine()

        # Integrate Economy Module
        economy_stats = {}
        if "economy" in self._requested_metrics:
            economy_stats = self._integrate_economy()

        # Integrate Combat Module
        combat_stats = {}
        if "trades" in self._requested_metrics:
            combat_stats = self._integrate_combat()

        # Build kill matrix (always useful)
        kill_matrix = self._build_kill_matrix()

        # Build round timeline
        round_timeline = self._build_round_timeline()

        # Extract position data for heatmaps
        kill_positions, death_positions = self._extract_position_data()

        # Extract grenade trajectory data for utility visualization
        grenade_positions = []
        grenade_team_stats = {}
        if "utility" in self._requested_metrics:
            grenade_positions, grenade_team_stats = self._extract_grenade_trajectories()

        # Generate AI coaching insights
        coaching_insights = self._generate_coaching_insights()

        # Calculate weapon-specific statistics (stub - returns empty dict)
        weapon_stats: dict[str, list] = {}

        # Build result
        team_scores = self._calculate_team_scores()
        analysis = MatchAnalysis(
            players=self._players,
            team1_score=team_scores[0],
            team2_score=team_scores[1],
            total_rounds=self.data.num_rounds,
            map_name=self.data.map_name,
            round_timeline=round_timeline,
            kill_matrix=kill_matrix,
            team_trade_rates=combat_stats.get("trade_rates", {}),
            team_opening_rates=combat_stats.get("opening_rates", {}),
            kill_positions=kill_positions,
            death_positions=death_positions,
            grenade_positions=grenade_positions,
            grenade_team_stats=grenade_team_stats,
            coaching_insights=coaching_insights,
            weapon_stats=weapon_stats,
        )

        logger.info(f"Analysis complete. {len(self._players)} players analyzed.")
        return analysis

    def _init_player_stats(self) -> None:
        """Initialize PlayerMatchStats for each player."""
        logger.info(f"Initializing stats for {len(self.data.player_names)} players")
        for steam_id, name in self.data.player_names.items():
            team = self.data.player_teams.get(steam_id, "Unknown")
            self._players[steam_id] = PlayerMatchStats(
                steam_id=steam_id,
                name=name,
                team=team,
                kills=0,
                deaths=0,
                assists=0,
                headshots=0,
                total_damage=0,
                rounds_played=self.data.num_rounds,
            )
            logger.debug(f"Initialized player: {name} (steamid={steam_id}, team={team})")

    def _calculate_basic_stats(self) -> None:
        """Calculate basic K/D/A and damage stats."""
        kills_df = self.data.kills_df
        damages_df = self.data.damages_df

        # Use cached column names for kills
        att_id_col = self._att_id_col or "attacker_steamid"
        vic_id_col = self._vic_id_col or "victim_steamid"

        # Find damage columns
        dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS) if not damages_df.empty else None
        dmg_col = self._find_col(damages_df, ["dmg_health", "damage", "dmg"]) if not damages_df.empty else None

        # Log DataFrame info for debugging
        if not kills_df.empty and att_id_col in kills_df.columns:
            unique_attackers = kills_df[att_id_col].dropna().unique()
            logger.info(f"DataFrame has {len(unique_attackers)} unique attackers in column '{att_id_col}'")
            logger.info(f"Player steamids: {list(self._players.keys())[:5]}...")
            logger.info(f"DataFrame attacker steamids (sample): {list(unique_attackers[:5])}")
            logger.info(f"Attacker column dtype: {kills_df[att_id_col].dtype}")

        for steam_id, player in self._players.items():
            # Kills - use cached column
            if not kills_df.empty and att_id_col in kills_df.columns:
                # Convert to same type for comparison (handle float vs int issue)
                player_kills = kills_df[kills_df[att_id_col].astype(float) == float(steam_id)]
                player.kills = len(player_kills)

                if "headshot" in kills_df.columns:
                    player.headshots = int(player_kills["headshot"].sum())

                if "weapon" in kills_df.columns:
                    player.weapon_kills = player_kills["weapon"].value_counts().to_dict()

            # Deaths - use cached column (handles user_steamid vs victim_steamid)
            if not kills_df.empty and vic_id_col in kills_df.columns:
                player.deaths = len(kills_df[kills_df[vic_id_col].astype(float) == float(steam_id)])

            # Assists
            if not kills_df.empty and "assister_steamid" in kills_df.columns:
                player.assists = len(kills_df[kills_df["assister_steamid"].astype(float) == float(steam_id)])

            # Damage - use dynamic column finding
            if dmg_att_col and dmg_col:
                player_dmg = damages_df[damages_df[dmg_att_col].astype(float) == float(steam_id)]
                player.total_damage = int(player_dmg[dmg_col].sum())

            # Flash assists
            if not kills_df.empty and "flash_assist" in kills_df.columns and "assister_steamid" in kills_df.columns:
                flash_assists = kills_df[
                    (kills_df["assister_steamid"].astype(float) == float(steam_id)) &
                    (kills_df["flash_assist"] == True)
                ]
                player.utility.flash_assists = len(flash_assists)

        # Log results
        total_kills = sum(p.kills for p in self._players.values())
        total_deaths = sum(p.deaths for p in self._players.values())
        logger.info(f"Basic stats calculated: {total_kills} total kills, {total_deaths} total deaths across {len(self._players)} players")

    def _calculate_multi_kills(self) -> None:
        """Calculate multi-kill rounds for each player."""
        kills_df = self.data.kills_df
        if kills_df.empty or not self._round_col or not self._att_id_col:
            logger.info("Skipping multi-kill calculation - missing columns")
            return

        for steam_id, player in self._players.items():
            player_kills = kills_df[kills_df[self._att_id_col].astype(float) == float(steam_id)]
            if player_kills.empty:
                continue
            kills_per_round = player_kills.groupby(self._round_col).size()

            player.multi_kills.rounds_with_1k = int((kills_per_round == 1).sum())
            player.multi_kills.rounds_with_2k = int((kills_per_round == 2).sum())
            player.multi_kills.rounds_with_3k = int((kills_per_round == 3).sum())
            player.multi_kills.rounds_with_4k = int((kills_per_round == 4).sum())
            player.multi_kills.rounds_with_5k = int((kills_per_round >= 5).sum())

    def _detect_opening_duels(self) -> None:
        """Detect opening duels (first kill of each round) with Entry TTD tracking.

        Entry duels are the first kills of each round. This method:
        1. Identifies the first kill of each round
        2. Calculates Entry TTD (time from first damage to kill for entry frags)
        3. Tracks T-side vs CT-side entries for context
        """
        kills_df = self.data.kills_df
        damages_df = self.data.damages_df
        if kills_df.empty or not self._round_col or not self._att_id_col or not self._vic_id_col:
            logger.info("Skipping opening duels - missing columns")
            return

        # Find damage columns for Entry TTD calculation
        dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS) if not damages_df.empty else None
        dmg_vic_col = self._find_col(damages_df, self.VIC_ID_COLS) if not damages_df.empty else None

        entry_kills_count = 0

        # Get first kill of each round
        for round_num in kills_df[self._round_col].unique():
            round_kills = kills_df[kills_df[self._round_col] == round_num].sort_values("tick")
            if round_kills.empty:
                continue

            first_kill = round_kills.iloc[0]
            attacker_id = safe_int(first_kill.get(self._att_id_col))
            victim_id = safe_int(first_kill.get(self._vic_id_col))
            kill_tick = safe_int(first_kill.get("tick"))

            # Get attacker side for T/CT classification
            attacker_side = ""
            if self._att_side_col and self._att_side_col in kills_df.columns:
                side_val = first_kill.get(self._att_side_col)
                if isinstance(side_val, str):
                    attacker_side = "CT" if "CT" in side_val.upper() else "T" if "T" in side_val.upper() else ""

            if attacker_id in self._players:
                self._players[attacker_id].opening_duels.attempts += 1
                self._players[attacker_id].opening_duels.wins += 1
                entry_kills_count += 1

                # Track T-side vs CT-side entries
                if attacker_side == "T":
                    self._players[attacker_id].opening_duels.t_side_entries += 1
                elif attacker_side == "CT":
                    self._players[attacker_id].opening_duels.ct_side_entries += 1

                # Calculate Entry TTD (time from first damage to kill)
                if dmg_att_col and dmg_vic_col and not damages_df.empty:
                    entry_damages = damages_df[
                        (damages_df[dmg_att_col].astype(float) == float(attacker_id)) &
                        (damages_df[dmg_vic_col].astype(float) == float(victim_id)) &
                        (damages_df["tick"] <= kill_tick)
                    ].sort_values("tick")

                    if not entry_damages.empty:
                        first_dmg_tick = safe_int(entry_damages.iloc[0]["tick"])
                        entry_ttd_ticks = kill_tick - first_dmg_tick
                        entry_ttd_ms = entry_ttd_ticks * self.MS_PER_TICK

                        # Only record reasonable TTD values (0-1500ms)
                        if 0 < entry_ttd_ms <= self.TTD_MAX_MS:
                            self._players[attacker_id].opening_duels.entry_ttd_values.append(entry_ttd_ms)

            if victim_id in self._players:
                self._players[victim_id].opening_duels.attempts += 1
                self._players[victim_id].opening_duels.losses += 1

        logger.info(f"Detected {entry_kills_count} entry kills across {len(kills_df[self._round_col].unique())} rounds")

    def _detect_trades(self) -> None:
        """Detect trade kills and track trade opportunities.

        Trade Kill: When a teammate is killed, check if you killed the enemy
        who killed them within 5 seconds.

        This method tracks:
        - kills_traded: How many times you avenged a teammate
        - deaths_traded: How many of your deaths were avenged
        - trade_attempts: Opportunities to trade (teammate deaths you could respond to)
        - failed_trades: Trade opportunities you didn't take
        """
        kills_df = self.data.kills_df
        if kills_df.empty or not self._round_col:
            logger.info("Skipping trade detection - missing round column")
            return

        if not self._vic_id_col or not self._att_id_col:
            logger.info("Skipping trade detection - missing id columns")
            return

        trade_window_ticks = int(TRADE_WINDOW_SECONDS * self.TICK_RATE)
        total_trades = 0
        total_opportunities = 0

        for round_num in kills_df[self._round_col].unique():
            round_kills = kills_df[kills_df[self._round_col] == round_num].sort_values("tick")

            for idx, kill in round_kills.iterrows():
                victim_id = safe_int(kill.get(self._vic_id_col))
                victim_team = safe_str(kill.get(self._vic_side_col)) if self._vic_side_col else ""
                killer_id = safe_int(kill.get(self._att_id_col))
                kill_tick = safe_int(kill.get("tick"))

                if not victim_id or not killer_id:
                    continue

                # Need attacker_side column for trade detection
                if not self._att_side_col or self._att_side_col not in round_kills.columns:
                    continue

                # Track trade opportunity: When someone dies, their teammates have
                # an opportunity to trade (kill the enemy who killed them)
                # Find alive teammates who could potentially trade
                teammates = [
                    sid for sid, player in self._players.items()
                    if player.team == victim_team and sid != victim_id
                ]

                # Each alive teammate gets a trade opportunity
                for teammate_id in teammates:
                    if teammate_id in self._players:
                        self._players[teammate_id].trades.trade_attempts += 1
                        total_opportunities += 1

                # Look for trade (teammate kills the killer within window)
                potential_trades = round_kills[
                    (round_kills["tick"] > kill_tick) &
                    (round_kills["tick"] <= kill_tick + trade_window_ticks) &
                    (round_kills[self._vic_id_col].astype(float) == float(killer_id)) &
                    (round_kills[self._att_side_col] == victim_team)
                ]

                if not potential_trades.empty:
                    # Trade occurred
                    trader_id = safe_int(potential_trades.iloc[0].get(self._att_id_col))
                    total_trades += 1

                    if victim_id in self._players:
                        self._players[victim_id].trades.deaths_traded += 1

                    if trader_id in self._players:
                        self._players[trader_id].trades.kills_traded += 1
                else:
                    # Trade opportunity missed - mark the victim's death as not traded
                    if victim_id in self._players:
                        self._players[victim_id].trades.failed_trades += 1

        logger.info(f"Detected {total_trades} trades from {total_opportunities} opportunities")

    def _detect_clutches(self) -> None:
        """Detect clutch situations (1vX where player is last alive) with win tracking.

        Clutch: A round where you were the last player alive on your team
        facing one or more enemies. Uses round winner data to determine success.

        This method tracks:
        - total_situations: Total 1vX clutch attempts
        - total_wins: Clutches won (determined by round outcome)
        - Per-scenario tracking (1v1, 1v2, etc.) with wins and attempts
        """
        kills_df = self.data.kills_df
        if kills_df.empty or not self._round_col or not self._vic_id_col or not self._vic_side_col:
            logger.info("Skipping clutch detection - missing columns")
            return

        # Build round winner lookup from rounds data
        round_winners = {}
        for round_info in self.data.rounds:
            round_winners[round_info.round_num] = round_info.winner

        total_clutch_situations = 0
        total_clutch_wins = 0

        for round_num in kills_df[self._round_col].unique():
            round_kills = kills_df[kills_df[self._round_col] == round_num].sort_values("tick")
            if len(round_kills) < 4:  # Need at least 4 deaths for a 1vX situation
                continue

            # Get round winner (CT or T)
            round_winner = round_winners.get(int(round_num), "Unknown")

            # Track deaths in order for each team
            ct_deaths = []
            t_deaths = []

            for _, kill in round_kills.iterrows():
                victim_id = safe_int(kill.get(self._vic_id_col))
                victim_side = safe_str(kill.get(self._vic_side_col))

                if "CT" in victim_side.upper():
                    ct_deaths.append(victim_id)
                elif "T" in victim_side.upper() and "CT" not in victim_side.upper():
                    t_deaths.append(victim_id)

            # Detect clutch situations for each side
            for side, deaths, enemy_deaths in [("CT", ct_deaths, t_deaths), ("T", t_deaths, ct_deaths)]:
                if len(deaths) < 4:  # Need 4+ teammates dead for 1vX (5-player team)
                    continue

                # Find players on this team
                team_players = [
                    sid for sid, p in self._players.items()
                    if side in p.team.upper()
                ]

                if len(team_players) != 5:
                    continue  # Skip if team size is unexpected

                # When 4th teammate dies, the 5th is in a clutch
                # Count how many enemies were alive at that moment
                enemies_alive_at_clutch = 5 - len([d for d in enemy_deaths if d in enemy_deaths[:len(deaths)]])

                # More accurate: count enemy deaths that happened BEFORE 4th teammate death
                # We look at kill order in the DataFrame
                fourth_death_idx = -1
                current_ct_deaths = 0
                current_t_deaths = 0

                for i, (_, kill) in enumerate(round_kills.iterrows()):
                    victim_side = safe_str(kill.get(self._vic_side_col))
                    if "CT" in victim_side.upper():
                        current_ct_deaths += 1
                        if side == "CT" and current_ct_deaths == 4:
                            fourth_death_idx = i
                            break
                    elif "T" in victim_side.upper():
                        current_t_deaths += 1
                        if side == "T" and current_t_deaths == 4:
                            fourth_death_idx = i
                            break

                if fourth_death_idx < 0:
                    continue

                # Count enemies alive when 4th teammate died
                if side == "CT":
                    enemies_alive = 5 - current_t_deaths
                else:
                    enemies_alive = 5 - current_ct_deaths

                if enemies_alive < 1 or enemies_alive > 5:
                    continue

                # Find the survivor (player on this team not in first 4 deaths)
                first_four_dead = deaths[:4]
                survivor_id = None
                for player_id in team_players:
                    if player_id not in first_four_dead:
                        survivor_id = player_id
                        break

                if survivor_id is None or survivor_id not in self._players:
                    continue

                player = self._players[survivor_id]
                clutch_won = (round_winner == side)

                # Update totals
                player.clutches.total_situations += 1
                total_clutch_situations += 1
                if clutch_won:
                    player.clutches.total_wins += 1
                    total_clutch_wins += 1

                # Update per-scenario stats
                if enemies_alive == 1:
                    player.clutches.v1_attempts += 1
                    if clutch_won:
                        player.clutches.v1_wins += 1
                elif enemies_alive == 2:
                    player.clutches.v2_attempts += 1
                    if clutch_won:
                        player.clutches.v2_wins += 1
                elif enemies_alive == 3:
                    player.clutches.v3_attempts += 1
                    if clutch_won:
                        player.clutches.v3_wins += 1
                elif enemies_alive == 4:
                    player.clutches.v4_attempts += 1
                    if clutch_won:
                        player.clutches.v4_wins += 1
                elif enemies_alive == 5:
                    player.clutches.v5_attempts += 1
                    if clutch_won:
                        player.clutches.v5_wins += 1

        logger.info(f"Detected {total_clutch_situations} clutch situations, {total_clutch_wins} won")

    def _calculate_kast(self) -> None:
        """Calculate KAST (Kill/Assist/Survived/Traded) for each player."""
        kills_df = self.data.kills_df
        if kills_df.empty or not self._round_col or not self._att_id_col or not self._vic_id_col:
            logger.info("Skipping KAST calculation - missing columns")
            return

        trade_window_ticks = int(TRADE_WINDOW_SECONDS * self.TICK_RATE)

        # Get unique round numbers from the data
        round_nums = sorted(kills_df[self._round_col].unique())

        for round_num in round_nums:
            round_kills = kills_df[kills_df[self._round_col] == round_num].sort_values("tick")

            for steam_id, player in self._players.items():
                kast_this_round = False

                # K - Got a kill
                if len(round_kills[round_kills[self._att_id_col] == steam_id]) > 0:
                    kast_this_round = True

                # A - Got an assist
                if not kast_this_round and "assister_steamid" in round_kills.columns:
                    if len(round_kills[round_kills["assister_steamid"] == steam_id]) > 0:
                        kast_this_round = True

                # S - Survived
                player_deaths = round_kills[round_kills[self._vic_id_col] == steam_id]
                if player_deaths.empty:
                    kast_this_round = True
                    player.rounds_survived += 1

                # T - Was traded (check if death was traded)
                if not kast_this_round and not player_deaths.empty:
                    death = player_deaths.iloc[0]
                    death_tick = safe_int(death.get("tick"))
                    killer_id = safe_int(death.get(self._att_id_col))
                    player_team = player.team

                    # Check if teammate killed the killer within trade window
                    if self._att_side_col and self._att_side_col in round_kills.columns:
                        trades = round_kills[
                            (round_kills["tick"] > death_tick) &
                            (round_kills["tick"] <= death_tick + trade_window_ticks) &
                            (round_kills[self._vic_id_col] == killer_id) &
                            (round_kills[self._att_side_col] == player_team)
                        ]
                        if not trades.empty:
                            kast_this_round = True

                if kast_this_round:
                    player.kast_rounds += 1

    def _compute_ttd(self) -> None:
        """Compute Time to Damage for each kill.

        Uses vectorized implementation when available for ~10-50x speedup.
        Falls back to per-kill loop for compatibility.
        """
        if self.data.damages_df.empty or not self.data.kills:
            logger.warning("No damage or kill data for TTD computation")
            return

        # Use optimized vectorized implementation if available
        if self._use_optimized and self._metrics_computer is not None:
            logger.info("Using vectorized TTD computation")
            self._metrics_computer.compute(MetricType.TTD)

            # Transfer results to player stats
            for steam_id, player in self._players.items():
                player.ttd_values = self._metrics_computer.get_ttd_values(steam_id)
                player.prefire_count = self._metrics_computer.get_prefire_count(steam_id)

            ttd_metrics = self._metrics_computer.ttd_metrics
            if ttd_metrics:
                logger.info(f"Computed TTD (vectorized) for {ttd_metrics.total_engagements} engagements")
            return

        # Fallback: Original per-kill loop implementation
        logger.info("Using per-kill TTD computation (fallback)")
        damages_df = self.data.damages_df

        # Find the right column names
        def find_col(df, options):
            for col in options:
                if col in df.columns:
                    return col
            return None

        dmg_att_col = find_col(damages_df, ["attacker_steamid", "attacker_steam_id"])
        dmg_vic_col = find_col(damages_df, ["user_steamid", "victim_steamid", "victim_steam_id"])

        if not dmg_att_col or not dmg_vic_col:
            logger.warning(f"Missing columns for TTD. Have: {list(damages_df.columns)}")
            return

        # Use KillEvent objects directly
        for kill in self.data.kills:
            try:
                att_id = kill.attacker_steamid
                vic_id = kill.victim_steamid
                kill_tick = kill.tick
                round_num = kill.round_num

                if not att_id or not vic_id:
                    continue

                # Find first damage from attacker to victim before kill
                mask = (
                    (damages_df[dmg_att_col] == att_id) &
                    (damages_df[dmg_vic_col] == vic_id) &
                    (damages_df["tick"] <= kill_tick)
                )
                engagement_damages = damages_df[mask].sort_values("tick")

                if engagement_damages.empty:
                    continue

                first_dmg_tick = safe_int(engagement_damages.iloc[0]["tick"])
                ttd_ticks = kill_tick - first_dmg_tick
                ttd_ms = ttd_ticks * self.MS_PER_TICK

                is_prefire = ttd_ms <= self.TTD_MIN_MS

                # Add to player stats
                if att_id in self._players and not is_prefire and ttd_ms <= self.TTD_MAX_MS:
                    self._players[att_id].ttd_values.append(ttd_ms)
                elif att_id in self._players and is_prefire:
                    self._players[att_id].prefire_count += 1

                self._ttd_results.append(TTDResult(
                    tick_spotted=first_dmg_tick,
                    tick_damage=kill_tick,
                    ttd_ticks=ttd_ticks,
                    ttd_ms=ttd_ms,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    weapon=kill.weapon,
                    headshot=kill.headshot,
                    is_prefire=is_prefire,
                    round_num=round_num,
                ))

            except Exception as e:
                logger.debug(f"Error processing kill for TTD: {e}")
                continue

        logger.info(f"Computed TTD for {len(self._ttd_results)} engagements")

    def _compute_crosshair_placement(self) -> None:
        """Compute crosshair placement error for each kill.

        Uses vectorized numpy implementation when available for ~5-20x speedup.
        Falls back to per-kill loop for compatibility.
        """
        # Use optimized vectorized implementation if available
        if self._use_optimized and self._metrics_computer is not None:
            logger.info("Using vectorized CP computation")
            self._metrics_computer.compute(MetricType.CP)

            # Transfer results to player stats
            for steam_id, player in self._players.items():
                player.cp_values = self._metrics_computer.get_cp_values(steam_id)

            cp_metrics = self._metrics_computer.cp_metrics
            if cp_metrics:
                logger.info(f"Computed CP (vectorized) for {cp_metrics.total_kills_analyzed} kills")
            return

        # Fallback: Original implementation
        logger.info("Using per-kill CP computation (fallback)")

        # First try using KillEvent objects directly (preferred - they have embedded position data)
        kills_with_pos = [k for k in self.data.kills if k.attacker_x is not None and k.attacker_pitch is not None and k.victim_x is not None]

        if kills_with_pos:
            logger.info(f"Computing CP from {len(kills_with_pos)} KillEvent objects with position data")
            self._compute_cp_from_kill_events(kills_with_pos)
            return

        # Fallback: check DataFrame for position columns
        if not self.data.kills_df.empty:
            kills_df = self.data.kills_df

            # Check various column name patterns
            pos_patterns = [
                ["attacker_X", "attacker_Y", "attacker_Z", "victim_X", "victim_Y", "victim_Z"],
                ["attacker_x", "attacker_y", "attacker_z", "victim_x", "victim_y", "victim_z"],
                ["attacker_X", "attacker_Y", "attacker_Z", "user_X", "user_Y", "user_Z"],
            ]
            angle_patterns = [
                ["attacker_pitch", "attacker_yaw"],
            ]

            has_positions = any(all(col in kills_df.columns for col in pattern) for pattern in pos_patterns)
            has_angles = any(all(col in kills_df.columns for col in pattern) for pattern in angle_patterns)

            if has_positions and has_angles:
                logger.info(f"Computing CP from DataFrame. Columns: {list(kills_df.columns)}")
                self._compute_cp_from_events()
                return

        # Final fallback: tick data
        if self.data.ticks_df is not None and not self.data.ticks_df.empty:
            self._compute_cp_from_ticks()
        else:
            logger.warning("No position/angle data available for CP computation. Position data requires parsing with player props.")

    def _compute_cp_from_kill_events(self, kills: list) -> None:
        """Compute CP from KillEvent objects with embedded position data."""
        for kill in kills:
            try:
                att_id = kill.attacker_steamid
                vic_id = kill.victim_steamid

                if not att_id or not vic_id:
                    continue

                # Build position arrays (add 64 units for eye height)
                att_pos = np.array([
                    kill.attacker_x,
                    kill.attacker_y,
                    kill.attacker_z + 64 if kill.attacker_z else 0
                ])
                vic_pos = np.array([
                    kill.victim_x,
                    kill.victim_y,
                    kill.victim_z + 64 if kill.victim_z else 0
                ])

                att_pitch = kill.attacker_pitch or 0.0
                att_yaw = kill.attacker_yaw or 0.0

                # Skip if positions are zero
                if np.allclose(att_pos[:2], 0) or np.allclose(vic_pos[:2], 0):
                    continue

                angular_error, pitch_error, yaw_error = self._calculate_angular_error(
                    att_pos, att_pitch, att_yaw, vic_pos
                )

                if att_id in self._players:
                    self._players[att_id].cp_values.append(angular_error)

                self._cp_results.append(CrosshairPlacementResult(
                    tick=kill.tick,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    angular_error_deg=angular_error,
                    pitch_error_deg=pitch_error,
                    yaw_error_deg=yaw_error,
                    round_num=kill.round_num,
                ))

            except Exception as e:
                logger.debug(f"Error processing kill for CP: {e}")
                continue

        logger.info(f"Computed CP for {len(self._cp_results)} kills from KillEvent objects")

    def _compute_cp_from_events(self) -> None:
        """Compute CP from position data embedded in kill events DataFrame."""
        kills_df = self.data.kills_df
        logger.info("Computing CP from DataFrame event-embedded positions")

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

                if att_id in self._players:
                    self._players[att_id].cp_values.append(angular_error)

                self._cp_results.append(CrosshairPlacementResult(
                    tick=tick,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    angular_error_deg=angular_error,
                    pitch_error_deg=pitch_error,
                    yaw_error_deg=yaw_error,
                    round_num=round_num,
                ))

            except Exception as e:
                logger.debug(f"Error processing kill for CP: {e}")
                continue

        logger.info(f"Computed CP for {len(self._cp_results)} kills")

    def _compute_cp_from_ticks(self) -> None:
        """Compute CP from tick-level data (fallback)."""
        ticks_df = self.data.ticks_df
        kills_df = self.data.kills_df
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

                att_pos = np.array([
                    safe_float(att_state["X"]),
                    safe_float(att_state["Y"]),
                    safe_float(att_state["Z"]) + 64
                ])
                att_pitch = safe_float(att_state["pitch"])
                att_yaw = safe_float(att_state["yaw"])

                vic_pos = np.array([
                    safe_float(vic_state["X"]),
                    safe_float(vic_state["Y"]),
                    safe_float(vic_state["Z"]) + 64
                ])

                angular_error, pitch_error, yaw_error = self._calculate_angular_error(
                    att_pos, att_pitch, att_yaw, vic_pos
                )

                if att_id in self._players:
                    self._players[att_id].cp_values.append(angular_error)

                self._cp_results.append(CrosshairPlacementResult(
                    tick=kill_tick,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    angular_error_deg=angular_error,
                    pitch_error_deg=pitch_error,
                    yaw_error_deg=yaw_error,
                    round_num=round_num,
                ))

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

    def _calculate_side_stats(self) -> None:
        """Calculate CT-side vs T-side performance breakdown."""
        kills_df = self.data.kills_df
        damages_df = self.data.damages_df

        if kills_df.empty or not self._att_id_col or not self._att_side_col:
            logger.info("Skipping side stats - missing columns")
            return

        # Determine which rounds were CT/T side for each player
        # In CS2, sides swap at halftime (typically round 13)
        # For simplicity, we use the attacker's side at kill time

        for steam_id, player in self._players.items():
            # Count CT-side kills
            if self._att_side_col:
                ct_kills_df = kills_df[
                    (kills_df[self._att_id_col] == steam_id) &
                    (kills_df[self._att_side_col].astype(str).str.upper().str.contains("CT", na=False))
                ]
                player.ct_stats.kills = len(ct_kills_df)

                t_kills_df = kills_df[
                    (kills_df[self._att_id_col] == steam_id) &
                    (~kills_df[self._att_side_col].astype(str).str.upper().str.contains("CT", na=False)) &
                    (kills_df[self._att_side_col].astype(str).str.upper().str.contains("T", na=False))
                ]
                player.t_stats.kills = len(t_kills_df)

            # Count CT-side deaths
            if self._vic_id_col and self._vic_side_col:
                ct_deaths_df = kills_df[
                    (kills_df[self._vic_id_col] == steam_id) &
                    (kills_df[self._vic_side_col].astype(str).str.upper().str.contains("CT", na=False))
                ]
                player.ct_stats.deaths = len(ct_deaths_df)

                t_deaths_df = kills_df[
                    (kills_df[self._vic_id_col] == steam_id) &
                    (~kills_df[self._vic_side_col].astype(str).str.upper().str.contains("CT", na=False)) &
                    (kills_df[self._vic_side_col].astype(str).str.upper().str.contains("T", na=False))
                ]
                player.t_stats.deaths = len(t_deaths_df)

            # Estimate rounds per side (typically half each)
            total_rounds = max(self.data.num_rounds, 1)
            half_rounds = total_rounds // 2
            player.ct_stats.rounds_played = half_rounds
            player.t_stats.rounds_played = total_rounds - half_rounds

            # Calculate side-specific damage
            if not damages_df.empty:
                dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS)
                dmg_att_side = self._find_col(damages_df, self.ATT_SIDE_COLS)
                dmg_col = self._find_col(damages_df, ["dmg_health", "damage", "dmg"])

                if dmg_att_col and dmg_att_side and dmg_col:
                    ct_dmg = damages_df[
                        (damages_df[dmg_att_col] == steam_id) &
                        (damages_df[dmg_att_side].astype(str).str.upper().str.contains("CT", na=False))
                    ]
                    player.ct_stats.total_damage = int(ct_dmg[dmg_col].sum())

                    t_dmg = damages_df[
                        (damages_df[dmg_att_col] == steam_id) &
                        (~damages_df[dmg_att_side].astype(str).str.upper().str.contains("CT", na=False)) &
                        (damages_df[dmg_att_side].astype(str).str.upper().str.contains("T", na=False))
                    ]
                    player.t_stats.total_damage = int(t_dmg[dmg_col].sum())

        logger.info("Calculated side-based stats")

    def _calculate_utility_stats(self) -> None:
        """Calculate comprehensive utility statistics (Leetify-style) using all available data."""

        # ===========================================
        # Use BLINDS data for accurate flash stats
        # ===========================================
        if hasattr(self.data, 'blinds') and self.data.blinds:
            logger.info(f"Using {len(self.data.blinds)} blind events for flash stats")
            for steam_id, player in self._players.items():
                player_blinds = [b for b in self.data.blinds if b.attacker_steamid == steam_id]

                # Separate enemy vs teammate blinds
                enemy_blinds = [b for b in player_blinds if not b.is_teammate]
                team_blinds = [b for b in player_blinds if b.is_teammate]

                # Only count blinds > 1.1 seconds as "real" blinds (Leetify threshold)
                significant_enemy_blinds = [b for b in enemy_blinds if b.blind_duration >= 1.1]

                player.utility.enemies_flashed = len(significant_enemy_blinds)
                player.utility.teammates_flashed = len(team_blinds)
                player.utility.total_blind_time = sum(b.blind_duration for b in enemy_blinds)

                # Count unique flashbangs (group blinds by tick proximity)
                if player_blinds:
                    blind_ticks = sorted(set(b.tick for b in player_blinds))
                    # Group ticks within 10 ticks as same flash
                    flash_count = 1
                    for i in range(1, len(blind_ticks)):
                        if blind_ticks[i] - blind_ticks[i-1] > 10:
                            flash_count += 1
                    player.utility.flashbangs_thrown = flash_count

        # ===========================================
        # Use GRENADES data for accurate grenade counts
        # ===========================================
        if hasattr(self.data, 'grenades') and self.data.grenades:
            logger.info(f"Using {len(self.data.grenades)} grenade events")
            for steam_id, player in self._players.items():
                player_grenades = [g for g in self.data.grenades if g.player_steamid == steam_id]

                # Count by type (grenade_type is sufficient - no event_type on GrenadeEvent)
                player.utility.smokes_thrown = len([g for g in player_grenades
                    if 'smoke' in g.grenade_type.lower()])
                he_thrown = len([g for g in player_grenades
                    if 'hegrenade' in g.grenade_type.lower() or 'he_grenade' in g.grenade_type.lower()])
                molly_thrown = len([g for g in player_grenades
                    if 'molotov' in g.grenade_type.lower() or 'incendiary' in g.grenade_type.lower()])
                flash_thrown = len([g for g in player_grenades
                    if 'flash' in g.grenade_type.lower()])

                # Only update if we got data (don't overwrite with 0 if no grenades events)
                if he_thrown > 0:
                    player.utility.he_thrown = he_thrown
                if molly_thrown > 0:
                    player.utility.molotovs_thrown = molly_thrown
                if flash_thrown > 0 and player.utility.flashbangs_thrown == 0:
                    player.utility.flashbangs_thrown = flash_thrown

        # ===========================================
        # Use DAMAGES data for HE/Molly damage (fallback and supplement)
        # ===========================================
        damages_df = self.data.damages_df
        if not damages_df.empty:
            att_col = self._find_col(damages_df, self.ATT_ID_COLS)
            att_side = self._find_col(damages_df, self.ATT_SIDE_COLS)
            vic_side = self._find_col(damages_df, self.VIC_SIDE_COLS)
            weapon_col = self._find_col(damages_df, ["weapon"])
            dmg_col = self._find_col(damages_df, ["dmg_health", "damage", "dmg"])

            if att_col and weapon_col and dmg_col:
                he_weapons = ["hegrenade", "he_grenade", "grenade_he", "hegrenade_projectile"]
                molly_weapons = ["molotov", "incgrenade", "inferno", "molotov_projectile", "incendiary"]

                for steam_id, player in self._players.items():
                    player_dmg = damages_df[damages_df[att_col] == steam_id]

                    # HE damage
                    he_dmg = player_dmg[player_dmg[weapon_col].str.lower().isin(he_weapons)]
                    if not he_dmg.empty:
                        if att_side and vic_side:
                            enemy_he = he_dmg[he_dmg[att_side] != he_dmg[vic_side]]
                            team_he = he_dmg[he_dmg[att_side] == he_dmg[vic_side]]
                            player.utility.he_damage = int(enemy_he[dmg_col].sum())
                            player.utility.he_team_damage = int(team_he[dmg_col].sum())
                        else:
                            player.utility.he_damage = int(he_dmg[dmg_col].sum())
                        if player.utility.he_thrown == 0:
                            player.utility.he_thrown = max(1, len(he_dmg[dmg_col].unique()))

                    # Molotov damage
                    molly_dmg = player_dmg[player_dmg[weapon_col].str.lower().isin(molly_weapons)]
                    if not molly_dmg.empty:
                        if att_side and vic_side:
                            enemy_molly = molly_dmg[molly_dmg[att_side] != molly_dmg[vic_side]]
                            team_molly = molly_dmg[molly_dmg[att_side] == molly_dmg[vic_side]]
                            player.utility.molotov_damage = int(enemy_molly[dmg_col].sum())
                            player.utility.molotov_team_damage = int(team_molly[dmg_col].sum())
                        else:
                            player.utility.molotov_damage = int(molly_dmg[dmg_col].sum())
                        if player.utility.molotovs_thrown == 0:
                            player.utility.molotovs_thrown = max(1, len(set(molly_dmg['tick'])) if 'tick' in molly_dmg.columns else 1)

        # ===========================================
        # Flash assists from kills (supplement)
        # ===========================================
        kills_df = self.data.kills_df
        if not kills_df.empty and "assister_steamid" in kills_df.columns and "flash_assist" in kills_df.columns:
            for steam_id, player in self._players.items():
                flash_assists = kills_df[
                    (kills_df["assister_steamid"] == steam_id) &
                    (kills_df["flash_assist"] == True)
                ]
                player.utility.flash_assists = len(flash_assists)

        logger.info("Calculated utility stats")

    def _calculate_accuracy_stats(self) -> None:
        """Calculate accuracy statistics from weapon_fire events."""
        if not hasattr(self.data, 'weapon_fires') or not self.data.weapon_fires:
            logger.info("No weapon_fire data available for accuracy stats")
            return

        damages_df = self.data.damages_df

        for steam_id, player in self._players.items():
            # Count shots fired
            player_shots = [f for f in self.data.weapon_fires if f.player_steamid == steam_id]
            player.shots_fired = len(player_shots)

            # Count shots that hit (from damage events)
            if not damages_df.empty:
                att_col = self._find_col(damages_df, self.ATT_ID_COLS)
                if att_col:
                    player_hits = damages_df[damages_df[att_col] == steam_id]
                    player.shots_hit = len(player_hits)

                    # Count headshot hits
                    hitgroup_col = self._find_col(damages_df, ["hitgroup"])
                    if hitgroup_col:
                        head_hits = player_hits[player_hits[hitgroup_col].str.lower().str.contains("head", na=False)]
                        player.headshot_hits = len(head_hits)

        logger.info("Calculated accuracy stats")

    def _calculate_mistakes(self) -> None:
        """Calculate mistakes (Scope.gg style)."""
        kills_df = self.data.kills_df
        damages_df = self.data.damages_df

        # Team kills (friendly fire deaths)
        if not kills_df.empty and self._att_id_col and self._vic_id_col and self._att_side_col and self._vic_side_col:
            for steam_id, player in self._players.items():
                # Check for team kills (attacker and victim same team)
                team_kills = kills_df[
                    (kills_df[self._att_id_col] == steam_id) &
                    (kills_df[self._att_side_col] == kills_df[self._vic_side_col])
                ]
                player.mistakes.team_kills = len(team_kills)

        # Team damage
        if not damages_df.empty:
            att_col = self._find_col(damages_df, self.ATT_ID_COLS)
            att_side = self._find_col(damages_df, self.ATT_SIDE_COLS)
            vic_side = self._find_col(damages_df, self.VIC_SIDE_COLS)
            dmg_col = self._find_col(damages_df, ["dmg_health", "damage", "dmg"])

            if att_col and att_side and vic_side and dmg_col:
                for steam_id, player in self._players.items():
                    team_dmg = damages_df[
                        (damages_df[att_col] == steam_id) &
                        (damages_df[att_side] == damages_df[vic_side])
                    ]
                    player.mistakes.team_damage = int(team_dmg[dmg_col].sum())

                    # Teammates flashed (from utility stats)
                    player.mistakes.teammates_flashed = player.utility.teammates_flashed

        logger.info("Calculated mistakes")

    def _run_state_machine(self) -> None:
        """
        Run State Machine analysis for pro-level metrics.

        Enhances PlayerMatchStats with:
        - Entry Kill (from State Machine - more accurate than opening duels)
        - Trade Kill (with 4-second window and proper killer tracking)
        - Lurk Kill (distance from team center of mass)
        - Flash Effectiveness (>2.0s blind duration)
        - Utility ADR (HE + Molotov damage per round)
        """
        try:
            from opensight.state_machine import StateMachine
        except ImportError as e:
            logger.warning(f"State Machine not available: {e}")
            return

        try:
            machine = StateMachine(self.data)
            result = machine.analyze()

            # Merge state machine results into PlayerMatchStats
            for steam_id, context_stats in result.players.items():
                if steam_id not in self._players:
                    continue

                player = self._players[steam_id]

                # Update entry stats (State Machine is more accurate)
                player.opening_duels.wins = context_stats.entry_kills
                player.opening_duels.losses = context_stats.entry_deaths
                player.opening_duels.attempts = context_stats.entry_attempts

                # Update trade stats (State Machine uses tighter 4-second window)
                player.trades.kills_traded = context_stats.trade_kills
                player.trades.deaths_traded = context_stats.deaths_traded
                player.trades.trade_attempts = context_stats.trade_opportunities

                # Add lurk stats
                player.lurk.kills = context_stats.lurk_kills
                player.lurk.deaths = context_stats.lurk_deaths

                # Flash effectiveness (>2.0 seconds = effective)
                player.effective_flashes = context_stats.effective_flashes
                player.ineffective_flashes = context_stats.ineffective_flashes

                # Utility ADR
                util_damage = context_stats.he_damage + context_stats.molotov_damage
                player.utility_adr = round(util_damage / player.rounds_played, 1) if player.rounds_played > 0 else 0.0

            logger.info(f"State Machine complete: {result.total_entry_kills} entries, "
                       f"{result.total_trade_kills} trades, {result.total_lurk_kills} lurks")

        except Exception as e:
            logger.warning(f"State Machine analysis failed: {e}")

    def _integrate_economy(self) -> dict:
        """Integrate economy module data into player stats."""
        if not HAS_ECONOMY:
            logger.debug("Economy module not available, skipping integration")
            return {}

        try:
            economy_analyzer = EconomyAnalyzer(self.data)
            economy_stats = economy_analyzer.analyze()

            # Merge economy data into player stats
            for steam_id, player in self._players.items():
                profile = economy_analyzer.get_player_profile(steam_id)
                if profile:
                    player.avg_equipment_value = profile.avg_equipment_value
                    player.eco_rounds = profile.eco_rounds
                    player.force_rounds = profile.force_rounds
                    player.full_buy_rounds = profile.full_buy_rounds
                    player.damage_per_dollar = profile.damage_per_dollar
                    player.kills_per_dollar = profile.kills_per_dollar

            logger.info(f"Economy integration complete: {economy_stats.rounds_analyzed} rounds")
            return {
                "avg_equipment": economy_stats.avg_equipment_value,
                "damage_per_dollar": economy_stats.damage_per_dollar,
            }
        except Exception as e:
            logger.warning(f"Economy integration failed: {e}")
            return {}

    def _integrate_combat(self) -> dict:
        """Integrate combat module data into player stats."""
        if not HAS_COMBAT:
            logger.debug("Combat module not available, skipping integration")
            return {}

        try:
            combat_analyzer = CombatAnalyzer(self.data)
            combat_stats = combat_analyzer.analyze()

            # Merge combat data into player stats
            for steam_id, player in self._players.items():
                if steam_id in combat_stats.player_stats:
                    cs = combat_stats.player_stats[steam_id]
                    player.trade_kill_time_avg_ms = cs.trade_kill_time_avg_ms
                    player.untraded_deaths = cs.untraded_deaths

            # Build team-level stats
            trade_rates = {
                "CT": combat_stats.team_trade_rate.get(3, 0.0),
                "T": combat_stats.team_trade_rate.get(2, 0.0),
            }
            opening_rates = {
                "CT": combat_stats.team_opening_win_rate.get(3, 0.0),
                "T": combat_stats.team_opening_win_rate.get(2, 0.0),
            }

            logger.info(f"Combat integration complete: {len(combat_stats.trade_kills)} trades, "
                       f"{len(combat_stats.opening_duels)} opening duels")
            return {
                "trade_rates": trade_rates,
                "opening_rates": opening_rates,
            }
        except Exception as e:
            logger.warning(f"Combat integration failed: {e}")
            return {}

    def _build_kill_matrix(self) -> list:
        """Build kill matrix showing who killed whom and how often."""
        kills_df = self.data.kills_df
        if kills_df.empty or not self._att_id_col or not self._vic_id_col:
            return []

        matrix_entries = []
        weapon_col = self._find_col(kills_df, ["weapon"])

        # Group by attacker-victim pairs
        try:
            grouped = kills_df.groupby([self._att_id_col, self._vic_id_col])

            for (att_id, vic_id), group in grouped:
                att_id = safe_int(att_id)
                vic_id = safe_int(vic_id)

                if att_id == 0 or vic_id == 0:
                    continue

                att_name = self.data.player_names.get(att_id, "Unknown")
                vic_name = self.data.player_names.get(vic_id, "Unknown")

                weapons = []
                if weapon_col:
                    weapons = group[weapon_col].value_counts().head(3).index.tolist()

                matrix_entries.append(KillMatrixEntry(
                    attacker_name=att_name,
                    victim_name=vic_name,
                    count=len(group),
                    weapons=weapons,
                ))

            logger.info(f"Built kill matrix with {len(matrix_entries)} entries")
        except Exception as e:
            logger.warning(f"Kill matrix building failed: {e}")

        return matrix_entries

    def _build_round_timeline(self) -> list:
        """Build round-by-round timeline with key events."""
        timeline = []

        if not self.data.rounds:
            return timeline

        kills_df = self.data.kills_df

        for round_info in self.data.rounds:
            try:
                round_num = round_info.round_num
                winner = round_info.winner or "Unknown"
                win_reason = round_info.win_reason or "unknown"

                # Get first kill of round
                first_kill_player = ""
                first_death_player = ""
                if not kills_df.empty and self._round_col and self._att_id_col and self._vic_id_col:
                    round_kills = kills_df[kills_df[self._round_col] == round_num]
                    if not round_kills.empty:
                        round_kills = round_kills.sort_values("tick")
                        first = round_kills.iloc[0]
                        att_id = safe_int(first.get(self._att_id_col))
                        vic_id = safe_int(first.get(self._vic_id_col))
                        first_kill_player = self.data.player_names.get(att_id, "")
                        first_death_player = self.data.player_names.get(vic_id, "")

                timeline.append(RoundTimeline(
                    round_num=round_num,
                    winner=winner,
                    win_reason=win_reason,
                    ct_score=round_info.ct_score,
                    t_score=round_info.t_score,
                    first_kill_player=first_kill_player,
                    first_death_player=first_death_player,
                ))
            except Exception as e:
                logger.debug(f"Error building timeline for round: {e}")
                continue

        logger.info(f"Built round timeline with {len(timeline)} rounds")
        return timeline

    def _extract_position_data(self) -> tuple[list, list]:
        """Extract position data for heatmap and kill map visualization."""
        kill_positions = []
        death_positions = []

        # Extract from KillEvent objects (they have position data)
        for kill in self.data.kills:
            try:
                att_name = self.data.player_names.get(kill.attacker_steamid, "Unknown")
                vic_name = self.data.player_names.get(kill.victim_steamid, "Unknown")

                # Attacker position (kill location)
                if kill.attacker_x is not None and kill.attacker_y is not None:
                    kill_positions.append({
                        "x": kill.attacker_x,
                        "y": kill.attacker_y,
                        "z": kill.attacker_z or 0,
                        "player": att_name,
                        "attacker": att_name,
                        "victim": vic_name,
                        "attacker_team": kill.attacker_side,
                        "victim_team": kill.victim_side,
                        "weapon": kill.weapon,
                        "round": kill.round_num,
                        "headshot": kill.headshot,
                    })

                # Victim position (death location)
                if kill.victim_x is not None and kill.victim_y is not None:
                    death_positions.append({
                        "x": kill.victim_x,
                        "y": kill.victim_y,
                        "z": kill.victim_z or 0,
                        "player": vic_name,
                        "attacker": att_name,
                        "victim_team": kill.victim_side,
                        "attacker_team": kill.attacker_side,
                        "round": kill.round_num,
                    })
            except Exception as e:
                logger.debug(f"Error extracting position: {e}")
                continue

        logger.info(f"Extracted {len(kill_positions)} kill positions, {len(death_positions)} death positions")
        return kill_positions, death_positions

    def _extract_grenade_trajectories(self) -> tuple[list, dict]:
        """
        Extract grenade trajectory data for utility visualization.

        Returns:
            Tuple of (grenade_positions, team_stats) where:
            - grenade_positions: List of dicts with position and metadata
            - team_stats: Dict with team-level utility statistics
        """
        from opensight.trajectory import GRENADE_COLORS, GRENADE_CSS_CLASSES

        grenade_positions = []
        team_stats = {
            "CT": {
                "total_utility": 0,
                "flashbangs": 0,
                "smokes": 0,
                "molotovs": 0,
                "he_grenades": 0,
                "enemies_flashed": 0,
            },
            "T": {
                "total_utility": 0,
                "flashbangs": 0,
                "smokes": 0,
                "molotovs": 0,
                "he_grenades": 0,
                "enemies_flashed": 0,
            },
        }

        # Process grenade events to get positions (detonation points)
        if hasattr(self.data, 'grenades') and self.data.grenades:
            for grenade in self.data.grenades:
                # Only include grenades with valid positions
                if grenade.x is not None and grenade.y is not None:
                    grenade_type = grenade.grenade_type.lower()
                    thrower_team = self.data.player_teams.get(grenade.player_steamid, "Unknown")

                    position = {
                        "x": grenade.x,
                        "y": grenade.y,
                        "z": grenade.z or 0,
                        "grenade_type": grenade_type,
                        "thrower_steamid": str(grenade.player_steamid),
                        "thrower_name": self.data.player_names.get(grenade.player_steamid, "Unknown"),
                        "thrower_team": thrower_team,
                        "round_num": grenade.round_num,
                        "tick": grenade.tick,
                        "color": GRENADE_COLORS.get(grenade_type, "#ffffff"),
                        "css_class": GRENADE_CSS_CLASSES.get(grenade_type, "grenade-unknown"),
                    }
                    grenade_positions.append(position)

                # Count grenades for team stats (count each grenade once)
                thrower_team = self.data.player_teams.get(grenade.player_steamid, "Unknown")
                if thrower_team in team_stats:
                    grenade_type = grenade.grenade_type.lower()
                    team_stats[thrower_team]["total_utility"] += 1

                    if "flash" in grenade_type:
                        team_stats[thrower_team]["flashbangs"] += 1
                    elif "smoke" in grenade_type:
                        team_stats[thrower_team]["smokes"] += 1
                    elif "molotov" in grenade_type or "inc" in grenade_type:
                        team_stats[thrower_team]["molotovs"] += 1
                    elif "hegrenade" in grenade_type:
                        team_stats[thrower_team]["he_grenades"] += 1

        # Count enemies flashed from blinds data
        if hasattr(self.data, 'blinds') and self.data.blinds:
            for blind in self.data.blinds:
                if not blind.is_teammate and blind.blind_duration >= 1.1:
                    attacker_team = self.data.player_teams.get(blind.attacker_steamid, "Unknown")
                    if attacker_team in team_stats:
                        team_stats[attacker_team]["enemies_flashed"] += 1

        logger.info(f"Extracted {len(grenade_positions)} grenade positions")
        return grenade_positions, team_stats

    def _generate_coaching_insights(self) -> list:
        """Generate AI-powered coaching insights based on player performance."""
        insights = []

        for steam_id, player in self._players.items():
            player_insights = []

            # HLTV Rating insights
            if player.hltv_rating < 0.8:
                player_insights.append({
                    "type": "warning",
                    "category": "overall",
                    "message": f"Low overall rating ({player.hltv_rating:.2f}). Focus on fundamentals.",
                    "priority": "high",
                })
            elif player.hltv_rating > 1.3:
                player_insights.append({
                    "type": "positive",
                    "category": "overall",
                    "message": f"Excellent performance ({player.hltv_rating:.2f}). Keep it up!",
                    "priority": "low",
                })

            # TTD insights (reaction time)
            if player.ttd_median_ms:
                if player.ttd_median_ms > 500:
                    player_insights.append({
                        "type": "warning",
                        "category": "aim",
                        "message": f"Slow time-to-damage ({player.ttd_median_ms:.0f}ms). Consider aim training.",
                        "priority": "medium",
                    })
                elif player.ttd_median_ms < 200:
                    player_insights.append({
                        "type": "positive",
                        "category": "aim",
                        "message": f"Fast reactions ({player.ttd_median_ms:.0f}ms TTD)!",
                        "priority": "low",
                    })

            # Crosshair placement insights
            if player.cp_median_error_deg:
                if player.cp_median_error_deg > 15:
                    player_insights.append({
                        "type": "warning",
                        "category": "aim",
                        "message": f"Poor crosshair placement ({player.cp_median_error_deg:.1f}° error). Keep crosshair at head level.",
                        "priority": "high",
                    })
                elif player.cp_median_error_deg < 5:
                    player_insights.append({
                        "type": "positive",
                        "category": "aim",
                        "message": f"Excellent crosshair placement ({player.cp_median_error_deg:.1f}°)!",
                        "priority": "low",
                    })

            # Trade insights
            if player.untraded_deaths > player.deaths * 0.6 and player.deaths > 3:
                player_insights.append({
                    "type": "warning",
                    "category": "positioning",
                    "message": f"Too many untraded deaths ({player.untraded_deaths}/{player.deaths}). Stay closer to teammates.",
                    "priority": "medium",
                })

            # Utility insights
            if player.utility.total_utility < player.rounds_played * 1.5 and player.rounds_played >= 10:
                player_insights.append({
                    "type": "warning",
                    "category": "utility",
                    "message": f"Low utility usage ({player.utility.total_utility} thrown in {player.rounds_played} rounds). Buy and use more grenades.",
                    "priority": "medium",
                })

            if player.utility.teammates_flashed > 5:
                player_insights.append({
                    "type": "mistake",
                    "category": "utility",
                    "message": f"Flashed teammates {player.utility.teammates_flashed} times. Communicate flash usage.",
                    "priority": "high",
                })

            # Opening duel insights
            if player.opening_duels.attempts >= 5:
                if player.opening_duels.win_rate < 30:
                    player_insights.append({
                        "type": "warning",
                        "category": "duels",
                        "message": f"Low opening duel win rate ({player.opening_duels.win_rate:.0f}%). Reconsider early aggression.",
                        "priority": "medium",
                    })
                elif player.opening_duels.win_rate > 70:
                    player_insights.append({
                        "type": "positive",
                        "category": "duels",
                        "message": f"Dominant entry fragging ({player.opening_duels.win_rate:.0f}% win rate)!",
                        "priority": "low",
                    })

            # KAST insights
            if player.kast_percentage < 60 and player.rounds_played >= 10:
                player_insights.append({
                    "type": "warning",
                    "category": "impact",
                    "message": f"Low KAST ({player.kast_percentage:.0f}%). Try to have more round impact.",
                    "priority": "medium",
                })

            # Team damage insights
            if player.mistakes.team_damage > 100:
                player_insights.append({
                    "type": "mistake",
                    "category": "mistakes",
                    "message": f"High team damage ({player.mistakes.team_damage}). Be more careful with grenades and fire.",
                    "priority": "high",
                })

            if player_insights:
                insights.append({
                    "steam_id": str(steam_id),
                    "player_name": player.name,
                    "insights": player_insights,
                })

        logger.info(f"Generated coaching insights for {len(insights)} players")
        return insights

    def _calculate_team_scores(self) -> tuple[int, int]:
        """Calculate team scores from round data."""
        if not self.data.rounds:
            return (0, 0)

        ct_wins = sum(1 for r in self.data.rounds if r.winner == "CT")
        t_wins = sum(1 for r in self.data.rounds if r.winner == "T")
        return (ct_wins, t_wins)


def compute_utility_metrics(match_data: DemoData) -> dict[str, UtilityMetrics]:
    """
    Compute utility usage metrics for all players from match data.

    This function provides a standalone way to extract utility statistics
    similar to Scope.gg's nade stats, using awpy's grenade, smoke, inferno,
    and blind data.

    Args:
        match_data: Parsed demo data (MatchData/DemoData) from DemoParser

    Returns:
        Dictionary mapping steam_id (as string) to UtilityMetrics for each player

    Example:
        >>> from opensight.parser import parse_demo
        >>> from opensight.analytics import compute_utility_metrics
        >>> data = parse_demo("match.dem")
        >>> utility_stats = compute_utility_metrics(data)
        >>> for steam_id, metrics in utility_stats.items():
        ...     print(f"{metrics.player_name}: {metrics.total_utility_thrown} grenades")
    """
    result: dict[str, UtilityMetrics] = {}

    # Initialize metrics for all known players
    for steam_id, name in match_data.player_names.items():
        team = match_data.player_teams.get(steam_id, "Unknown")
        result[str(steam_id)] = UtilityMetrics(
            player_name=name,
            player_steamid=steam_id,
            team=team,
        )

    # ===========================================
    # Count grenades from grenades list
    # ===========================================
    if hasattr(match_data, 'grenades') and match_data.grenades:
        for grenade in match_data.grenades:
            steam_id = str(grenade.player_steamid)
            if steam_id not in result:
                # Player not in player_names, add them
                result[steam_id] = UtilityMetrics(
                    player_name=grenade.player_name,
                    player_steamid=grenade.player_steamid,
                    team=grenade.player_side,
                )

            grenade_type = grenade.grenade_type.lower()

            # Count by grenade type (awpy uses grenade_type field)
            if 'smoke' in grenade_type:
                result[steam_id].smokes_thrown += 1
            elif 'flash' in grenade_type:
                result[steam_id].flashes_thrown += 1
            elif 'hegrenade' in grenade_type or 'he_grenade' in grenade_type:
                result[steam_id].he_thrown += 1
            elif 'molotov' in grenade_type or 'incgrenade' in grenade_type or 'incendiary' in grenade_type:
                result[steam_id].molotovs_thrown += 1

    # ===========================================
    # Count smokes from smokes list (more accurate count)
    # ===========================================
    if hasattr(match_data, 'smokes') and match_data.smokes:
        # Reset smoke counts and use smoke events for more accurate tracking
        for steam_id in result:
            result[steam_id].smokes_thrown = 0

        for smoke in match_data.smokes:
            steam_id = str(smoke.thrower_steamid)
            if steam_id in result:
                result[steam_id].smokes_thrown += 1

    # ===========================================
    # Count molotovs from infernos list (more accurate count)
    # ===========================================
    if hasattr(match_data, 'infernos') and match_data.infernos:
        # Reset molotov counts and use inferno events for more accurate tracking
        for steam_id in result:
            result[steam_id].molotovs_thrown = 0

        for inferno in match_data.infernos:
            steam_id = str(inferno.thrower_steamid)
            if steam_id in result:
                result[steam_id].molotovs_thrown += 1

    # ===========================================
    # Process blind events for flash effectiveness
    # ===========================================
    if hasattr(match_data, 'blinds') and match_data.blinds:
        for blind in match_data.blinds:
            steam_id = str(blind.attacker_steamid)
            if steam_id not in result:
                continue

            # Only count significant blinds (>1.1 seconds per Leetify standard)
            if blind.blind_duration >= 1.1:
                if blind.is_teammate:
                    result[steam_id].flashes_teammates_total += 1
                else:
                    result[steam_id].flashes_enemies_total += 1

            # Accumulate total blind time for enemies only
            if not blind.is_teammate:
                result[steam_id].total_blind_time += blind.blind_duration

    # ===========================================
    # Calculate utility damage from damages DataFrame
    # ===========================================
    damages_df = match_data.damages_df
    if not damages_df.empty:
        # Find column names (different parsers use different names)
        att_col = None
        for col in ["attacker_steamid", "attacker", "att_steamid"]:
            if col in damages_df.columns:
                att_col = col
                break

        weapon_col = "weapon" if "weapon" in damages_df.columns else None
        dmg_col = None
        for col in ["dmg_health", "damage", "dmg"]:
            if col in damages_df.columns:
                dmg_col = col
                break

        att_side_col = None
        for col in ["attacker_side", "attacker_team"]:
            if col in damages_df.columns:
                att_side_col = col
                break

        vic_side_col = None
        for col in ["victim_side", "victim_team"]:
            if col in damages_df.columns:
                vic_side_col = col
                break

        if att_col and weapon_col and dmg_col:
            he_weapons = ["hegrenade", "he_grenade", "grenade_he", "hegrenade_projectile"]
            molly_weapons = ["molotov", "incgrenade", "inferno", "molotov_projectile", "incendiary"]

            for steam_id, metrics in result.items():
                steam_id_int = int(steam_id)
                player_dmg = damages_df[damages_df[att_col] == steam_id_int]

                # HE damage
                he_dmg = player_dmg[player_dmg[weapon_col].str.lower().isin(he_weapons)]
                if not he_dmg.empty:
                    if att_side_col and vic_side_col:
                        # Only count enemy damage
                        enemy_he = he_dmg[he_dmg[att_side_col] != he_dmg[vic_side_col]]
                        metrics.he_damage = int(enemy_he[dmg_col].sum())
                    else:
                        metrics.he_damage = int(he_dmg[dmg_col].sum())

                # Molotov damage
                molly_dmg = player_dmg[player_dmg[weapon_col].str.lower().isin(molly_weapons)]
                if not molly_dmg.empty:
                    if att_side_col and vic_side_col:
                        # Only count enemy damage
                        enemy_molly = molly_dmg[molly_dmg[att_side_col] != molly_dmg[vic_side_col]]
                        metrics.molotov_damage = int(enemy_molly[dmg_col].sum())
                    else:
                        metrics.molotov_damage = int(molly_dmg[dmg_col].sum())

                # Total utility damage
                metrics.total_utility_damage = float(metrics.he_damage + metrics.molotov_damage)

    # ===========================================
    # Count flash assists from kills DataFrame
    # ===========================================
    kills_df = match_data.kills_df
    if not kills_df.empty and "assister_steamid" in kills_df.columns and "flash_assist" in kills_df.columns:
        for steam_id, metrics in result.items():
            steam_id_int = int(steam_id)
            flash_assists = kills_df[
                (kills_df["assister_steamid"] == steam_id_int) &
                (kills_df["flash_assist"] == True)
            ]
            metrics.flash_assists = len(flash_assists)

    logger.info(f"Computed utility metrics for {len(result)} players")
    return result


def analyze_demo(
    demo_data: DemoData,
    metrics: Optional[str | list[str]] = None,
    use_cache: bool = True,
    use_optimized: bool = True
) -> MatchAnalysis:
    """Convenience function to analyze a parsed demo.

    Args:
        demo_data: Parsed demo data from DemoParser
        metrics: Which metrics to compute. Options:
            - None or "full": Compute all metrics (default)
            - "basic": Only basic stats (KD, ADR, HS%)
            - "advanced": Basic + TTD, CP, trades, opening duels
            - List of specific categories: ["ttd", "cp", "trades"]
        use_cache: Whether to use metrics caching (default True)
        use_optimized: Whether to use vectorized implementations (default True)

    Returns:
        MatchAnalysis with computed metrics
    """
    analyzer = DemoAnalyzer(
        demo_data,
        metrics=metrics,
        use_cache=use_cache,
        use_optimized=use_optimized
    )
    return analyzer.analyze()


def get_player_comparison_stats(
    player_a: PlayerMatchStats,
    player_b: PlayerMatchStats,
    normalize: bool = True
) -> dict:
    """
    Generate comparison statistics for two players for radar chart visualization.

    Returns normalized stats (0-100 scale) suitable for radar charts when normalize=True.
    The axes are:
    - HLTV Rating (higher is better)
    - Impact Score (higher is better)
    - TTD (inverted - lower TTD means faster reactions, so we invert for display)
    - Headshot % (higher is better)
    - Utility Damage (HE + Molotov damage, higher is better)

    Args:
        player_a: First player's match stats
        player_b: Second player's match stats
        normalize: If True, normalize values to 0-100 scale

    Returns:
        Dict with comparison data for both players
    """
    def get_ttd_score(player: PlayerMatchStats) -> float:
        """Calculate TTD score. Lower TTD is better, so we invert it."""
        ttd = player.ttd_median_ms
        if ttd is None:
            return 50.0 if normalize else 0.0  # Default to average if no data

        if normalize:
            # TTD typically ranges from 100ms (elite) to 600ms (slow)
            # Invert so that lower (faster) TTD gives higher score
            # 100ms -> 100 score, 600ms -> 0 score
            clamped = max(100, min(600, ttd))
            return round(100 - ((clamped - 100) / 500 * 100), 1)
        return round(ttd, 1)

    def get_hltv_score(player: PlayerMatchStats) -> float:
        """Normalize HLTV rating to 0-100 scale."""
        rating = player.hltv_rating
        if normalize:
            # HLTV rating typically ranges from 0.5 to 2.0
            # Map 0.5 -> 0, 1.0 -> 50, 1.5 -> 100
            clamped = max(0.5, min(2.0, rating))
            return round((clamped - 0.5) / 1.5 * 100, 1)
        return round(rating, 2)

    def get_impact_score(player: PlayerMatchStats) -> float:
        """Normalize impact rating to 0-100 scale."""
        impact = player.impact_rating
        if normalize:
            # Impact typically ranges from 0.0 to 2.0
            clamped = max(0.0, min(2.0, impact))
            return round(clamped / 2.0 * 100, 1)
        return round(impact, 2)

    def get_hs_score(player: PlayerMatchStats) -> float:
        """Get headshot percentage (already 0-100)."""
        return round(player.headshot_percentage, 1)

    def get_utility_damage_score(player: PlayerMatchStats) -> float:
        """Calculate utility damage score (HE + Molotov damage)."""
        he_dmg = player.utility.he_damage
        molotov_dmg = player.utility.molotov_damage
        total_dmg = he_dmg + molotov_dmg

        if normalize:
            # Utility damage can vary widely, let's use 0-500 as a typical range
            # Map 0 -> 0, 250 -> 50, 500+ -> 100
            clamped = max(0, min(500, total_dmg))
            return round(clamped / 500 * 100, 1)
        return total_dmg

    # Calculate stats for both players
    player_a_stats = {
        "name": player_a.name,
        "steam_id": str(player_a.steam_id),
        "team": player_a.team,
        "metrics": {
            "hltv_rating": get_hltv_score(player_a),
            "impact_score": get_impact_score(player_a),
            "ttd_score": get_ttd_score(player_a),
            "headshot_pct": get_hs_score(player_a),
            "utility_damage": get_utility_damage_score(player_a),
        },
        "raw_values": {
            "hltv_rating": round(player_a.hltv_rating, 2),
            "impact_rating": round(player_a.impact_rating, 2),
            "ttd_median_ms": round(player_a.ttd_median_ms, 1) if player_a.ttd_median_ms else None,
            "headshot_pct": round(player_a.headshot_percentage, 1),
            "utility_damage": player_a.utility.he_damage + player_a.utility.molotov_damage,
        }
    }

    player_b_stats = {
        "name": player_b.name,
        "steam_id": str(player_b.steam_id),
        "team": player_b.team,
        "metrics": {
            "hltv_rating": get_hltv_score(player_b),
            "impact_score": get_impact_score(player_b),
            "ttd_score": get_ttd_score(player_b),
            "headshot_pct": get_hs_score(player_b),
            "utility_damage": get_utility_damage_score(player_b),
        },
        "raw_values": {
            "hltv_rating": round(player_b.hltv_rating, 2),
            "impact_rating": round(player_b.impact_rating, 2),
            "ttd_median_ms": round(player_b.ttd_median_ms, 1) if player_b.ttd_median_ms else None,
            "headshot_pct": round(player_b.headshot_percentage, 1),
            "utility_damage": player_b.utility.he_damage + player_b.utility.molotov_damage,
        }
    }

    return {
        "labels": ["HLTV Rating", "Impact Score", "TTD (Reaction)", "Headshot %", "Utility Damage"],
        "player_a": player_a_stats,
        "player_b": player_b_stats,
        "normalized": normalize,
    }


def compare_players(
    match_analysis: MatchAnalysis,
    player_a_name: str,
    player_b_name: str
) -> dict:
    """
    Compare two players using Scope.gg-style radar chart axes.

    Axes (all normalized to 0-100 score for visualization):
    - ADR: Average Damage per Round (scaled 0-150, where 100 ADR = 66.7 score)
    - Opening Success %: Percentage of opening duels won (0-100%)
    - Clutch Win %: Percentage of clutch situations won (0-100%)
    - Trade Success %: Percentage of trade opportunities converted (0-100%)
    - Utility Usage: Total grenades thrown per round (scaled 0-4 per round)

    Args:
        match_analysis: The MatchAnalysis object containing player data
        player_a_name: Name of the first player to compare
        player_b_name: Name of the second player to compare

    Returns:
        Dict with:
            - player_a_name: First player's name
            - player_b_name: Second player's name
            - axes: List of axis names
            - scores_a: List of normalized scores (0-100) for player A
            - scores_b: List of normalized scores (0-100) for player B
            - raw_values_a: Dict of raw metric values for player A
            - raw_values_b: Dict of raw metric values for player B
    """
    # Find players by name
    player_a = None
    player_b = None

    for player in match_analysis.players.values():
        if player.name.lower() == player_a_name.lower():
            player_a = player
        elif player.name.lower() == player_b_name.lower():
            player_b = player

    if player_a is None:
        raise ValueError(f"Player '{player_a_name}' not found in match")
    if player_b is None:
        raise ValueError(f"Player '{player_b_name}' not found in match")

    def normalize_adr(adr: float) -> float:
        """
        Normalize ADR to 0-100 scale.
        Scale: 0 ADR -> 0, 150 ADR -> 100 (linear)
        Most pros average 70-90 ADR, exceptional is 100+
        """
        return round(min(100, max(0, (adr / 150) * 100)), 1)

    def normalize_percentage(pct: float) -> float:
        """
        Normalize a percentage (0-100) to 0-100 score.
        Already in the right range, just ensure bounds.
        """
        return round(min(100, max(0, pct)), 1)

    def normalize_utility_usage(player: PlayerMatchStats) -> float:
        """
        Normalize utility usage to 0-100 scale.
        Based on grenades thrown per round.
        Scale: 0 per round -> 0, 4+ per round -> 100
        Average player uses ~2-3 utility per round.
        """
        if player.rounds_played == 0:
            return 0.0
        total_utility = (
            player.utility.flashbangs_thrown +
            player.utility.he_thrown +
            player.utility.molotovs_thrown +
            player.utility.smokes_thrown
        )
        utility_per_round = total_utility / player.rounds_played
        # Scale: 4 grenades per round = 100
        return round(min(100, max(0, (utility_per_round / 4) * 100)), 1)

    def get_utility_per_round(player: PlayerMatchStats) -> float:
        """Get raw utility usage per round."""
        if player.rounds_played == 0:
            return 0.0
        total_utility = (
            player.utility.flashbangs_thrown +
            player.utility.he_thrown +
            player.utility.molotovs_thrown +
            player.utility.smokes_thrown
        )
        return round(total_utility / player.rounds_played, 2)

    # Define axes
    axes = [
        "ADR",
        "Opening Success %",
        "Clutch Win %",
        "Trade Success %",
        "Utility Usage"
    ]

    # Calculate normalized scores for player A
    scores_a = [
        normalize_adr(player_a.adr),
        normalize_percentage(player_a.opening_duels.win_rate),
        normalize_percentage(player_a.clutches.win_rate),
        normalize_percentage(player_a.trades.trade_rate),
        normalize_utility_usage(player_a)
    ]

    # Calculate normalized scores for player B
    scores_b = [
        normalize_adr(player_b.adr),
        normalize_percentage(player_b.opening_duels.win_rate),
        normalize_percentage(player_b.clutches.win_rate),
        normalize_percentage(player_b.trades.trade_rate),
        normalize_utility_usage(player_b)
    ]

    # Raw values for display
    raw_values_a = {
        "adr": round(player_a.adr, 1),
        "opening_success_pct": round(player_a.opening_duels.win_rate, 1),
        "opening_attempts": player_a.opening_duels.attempts,
        "opening_wins": player_a.opening_duels.wins,
        "clutch_win_pct": round(player_a.clutches.win_rate, 1),
        "clutch_situations": player_a.clutches.total_situations,
        "clutch_wins": player_a.clutches.total_wins,
        "trade_success_pct": round(player_a.trades.trade_rate, 1),
        "trade_kills": player_a.trades.kills_traded,
        "trade_attempts": player_a.trades.trade_attempts,
        "utility_per_round": get_utility_per_round(player_a),
        "total_utility": (
            player_a.utility.flashbangs_thrown +
            player_a.utility.he_thrown +
            player_a.utility.molotovs_thrown +
            player_a.utility.smokes_thrown
        ),
        # Additional stats for the comparison table
        "kills": player_a.kills,
        "deaths": player_a.deaths,
        "kd_ratio": player_a.kd_ratio,
        "headshot_pct": round(player_a.headshot_percentage, 1),
        "hltv_rating": round(player_a.hltv_rating, 2),
        "kast_pct": round(player_a.kast_percentage, 1),
        "ttd_median_ms": round(player_a.ttd_median_ms, 1) if player_a.ttd_median_ms else None,
    }

    raw_values_b = {
        "adr": round(player_b.adr, 1),
        "opening_success_pct": round(player_b.opening_duels.win_rate, 1),
        "opening_attempts": player_b.opening_duels.attempts,
        "opening_wins": player_b.opening_duels.wins,
        "clutch_win_pct": round(player_b.clutches.win_rate, 1),
        "clutch_situations": player_b.clutches.total_situations,
        "clutch_wins": player_b.clutches.total_wins,
        "trade_success_pct": round(player_b.trades.trade_rate, 1),
        "trade_kills": player_b.trades.kills_traded,
        "trade_attempts": player_b.trades.trade_attempts,
        "utility_per_round": get_utility_per_round(player_b),
        "total_utility": (
            player_b.utility.flashbangs_thrown +
            player_b.utility.he_thrown +
            player_b.utility.molotovs_thrown +
            player_b.utility.smokes_thrown
        ),
        # Additional stats for the comparison table
        "kills": player_b.kills,
        "deaths": player_b.deaths,
        "kd_ratio": player_b.kd_ratio,
        "headshot_pct": round(player_b.headshot_percentage, 1),
        "hltv_rating": round(player_b.hltv_rating, 2),
        "kast_pct": round(player_b.kast_percentage, 1),
        "ttd_median_ms": round(player_b.ttd_median_ms, 1) if player_b.ttd_median_ms else None,
    }

    return {
        "player_a_name": player_a.name,
        "player_b_name": player_b.name,
        "player_a_team": player_a.team,
        "player_b_team": player_b.team,
        "player_a_steam_id": str(player_a.steam_id),
        "player_b_steam_id": str(player_b.steam_id),
        "axes": axes,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "raw_values_a": raw_values_a,
        "raw_values_b": raw_values_b,
    }


# Alias for backward compatibility
PlayerAnalytics = PlayerMatchStats


# =============================================================================
# Tier 1 Player Metrics - Clean MatchData Interface
# =============================================================================

@dataclass
class PlayerMetrics:
    """
    Tier 1 player metrics computed from awpy MatchData.

    This is a simpler, focused dataclass for core player statistics
    that consumes the awpy-based MatchData structure directly.

    Fields:
        player_name: Display name of the player
        kills: Total kill count
        deaths: Total death count
        assists: Total assist count
        headshot_kills: Number of kills that were headshots
        damage_total: Total damage dealt to enemies
        adr: Average Damage per Round
        mean_ttd_ms: Mean Time to Damage in milliseconds (None if no data)
        median_crosshair_error_deg: Median crosshair placement error in degrees (None if no data)
    """
    player_name: str
    kills: int
    deaths: int
    assists: int
    headshot_kills: int
    damage_total: int
    adr: float
    mean_ttd_ms: Optional[float] = None
    median_crosshair_error_deg: Optional[float] = None

    # Optional extended metrics
    headshot_percentage: float = 0.0
    median_ttd_ms: Optional[float] = None
    mean_crosshair_error_deg: Optional[float] = None

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.kills > 0:
            self.headshot_percentage = round(self.headshot_kills / self.kills * 100, 1)


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
    victim_pos: np.ndarray
) -> Optional[float]:
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

    except Exception:
        return None


def calculate_player_metrics(match_data: DemoData) -> dict[str, PlayerMetrics]:
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
    kills_df = match_data.kills_df
    damages_df = match_data.damages_df
    num_rounds = max(match_data.num_rounds, 1)
    tick_rate = match_data.tick_rate if match_data.tick_rate > 0 else CS2_TICK_RATE

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

    # ===========================================
    # Calculate basic stats from DataFrames
    # ===========================================

    # Find column names (handle various naming conventions)
    def find_column(df: pd.DataFrame, options: list[str]) -> Optional[str]:
        for col in options:
            if col in df.columns:
                return col
        return None

    att_steamid_col = find_column(kills_df, ["attacker_steamid", "attacker_steam_id"])
    vic_steamid_col = find_column(kills_df, ["victim_steamid", "user_steamid", "victim_steam_id"])
    round_col = find_column(kills_df, ["round_num", "round", "total_rounds_played"])

    dmg_att_col = find_column(damages_df, ["attacker_steamid", "attacker_steam_id"])
    dmg_vic_col = find_column(damages_df, ["victim_steamid", "user_steamid", "victim_steam_id"])
    dmg_round_col = find_column(damages_df, ["round_num", "round", "total_rounds_played"])
    dmg_amount_col = find_column(damages_df, ["damage", "dmg_health", "health_damage", "dmg"])

    # Count kills, deaths, assists, headshots per player
    for steam_id, name in match_data.player_names.items():
        if name not in player_data:
            continue

        # Kills and headshots
        if not kills_df.empty and att_steamid_col:
            player_kills = kills_df[kills_df[att_steamid_col].fillna(0).astype(float) == float(steam_id)]
            player_data[name]["kills"] = len(player_kills)

            if "headshot" in kills_df.columns:
                player_data[name]["headshot_kills"] = int(player_kills["headshot"].fillna(False).sum())

        # Deaths
        if not kills_df.empty and vic_steamid_col:
            player_deaths = kills_df[kills_df[vic_steamid_col].fillna(0).astype(float) == float(steam_id)]
            player_data[name]["deaths"] = len(player_deaths)

        # Assists
        if not kills_df.empty and "assister_steamid" in kills_df.columns:
            player_assists = kills_df[kills_df["assister_steamid"].fillna(0).astype(float) == float(steam_id)]
            player_data[name]["assists"] = len(player_assists)

        # Total damage
        if not damages_df.empty and dmg_att_col and dmg_amount_col:
            player_dmg = damages_df[damages_df[dmg_att_col].fillna(0).astype(float) == float(steam_id)]
            player_data[name]["damage_total"] = int(player_dmg[dmg_amount_col].fillna(0).sum())

    # ===========================================
    # Calculate TTD (Time to Damage)
    # ===========================================
    #
    # For each kill:
    # 1. Find the first damage event in the same round
    # 2. From the same attacker to the same victim
    # 3. Before the kill tick
    # 4. Compute TTD = (kill_tick - first_damage_tick) * (1000 / tick_rate)

    ms_per_tick = 1000.0 / tick_rate
    ttd_min_ms = 0.0
    ttd_max_ms = 1500.0  # Filter out unreasonable values

    if not kills_df.empty and not damages_df.empty and att_steamid_col and vic_steamid_col:
        if dmg_att_col and dmg_vic_col and "tick" in kills_df.columns and "tick" in damages_df.columns:

            for _, kill_row in kills_df.iterrows():
                try:
                    att_id = kill_row.get(att_steamid_col)
                    vic_id = kill_row.get(vic_steamid_col)
                    kill_tick = _safe_int(kill_row.get("tick"))
                    kill_round = kill_row.get(round_col) if round_col else None

                    if pd.isna(att_id) or pd.isna(vic_id) or kill_tick <= 0:
                        continue

                    att_id = float(att_id)
                    vic_id = float(vic_id)

                    # Find first damage from this attacker to this victim before the kill
                    # Optionally filter by round if available
                    dmg_mask = (
                        (damages_df[dmg_att_col].fillna(0).astype(float) == att_id) &
                        (damages_df[dmg_vic_col].fillna(0).astype(float) == vic_id) &
                        (damages_df["tick"].fillna(0) <= kill_tick)
                    )

                    if kill_round is not None and dmg_round_col and not pd.isna(kill_round):
                        dmg_mask = dmg_mask & (damages_df[dmg_round_col].fillna(-1) == kill_round)

                    relevant_damages = damages_df[dmg_mask].sort_values("tick")

                    if relevant_damages.empty:
                        continue  # No damage found before kill

                    first_dmg_tick = _safe_int(relevant_damages.iloc[0]["tick"])
                    if first_dmg_tick <= 0:
                        continue

                    ttd_ticks = kill_tick - first_dmg_tick
                    ttd_ms = ttd_ticks * ms_per_tick

                    # Filter reasonable TTD values
                    if ttd_min_ms < ttd_ms <= ttd_max_ms:
                        # Find player name for this attacker
                        attacker_name = match_data.player_names.get(int(att_id))
                        if attacker_name and attacker_name in player_data:
                            player_data[attacker_name]["ttd_values"].append(ttd_ms)

                except Exception:
                    continue  # Skip problematic kills

    # ===========================================
    # Calculate Crosshair Placement (CP)
    # ===========================================
    #
    # For each kill with position/angle data:
    # 1. Get attacker position (X, Y, Z) and view angles (pitch, yaw)
    # 2. Get victim position (X, Y, Z)
    # 3. Convert view angles to a normalized 3D direction vector
    # 4. Compute direction from attacker to victim, normalized
    # 5. Compute angle between vectors: arccos(dot(view_vec, ideal_vec))

    # Eye height offset (player eye level above origin)
    EYE_HEIGHT = 64.0

    # Try to compute CP from KillEvent objects first (have embedded position data)
    kills_with_pos = [
        k for k in match_data.kills
        if k.attacker_x is not None and k.attacker_pitch is not None and k.victim_x is not None
    ]

    if kills_with_pos:
        for kill in kills_with_pos:
            try:
                att_name = match_data.player_names.get(kill.attacker_steamid)
                if not att_name or att_name not in player_data:
                    continue

                # Attacker position (add eye height)
                att_x = _safe_float(kill.attacker_x)
                att_y = _safe_float(kill.attacker_y)
                att_z = _safe_float(kill.attacker_z) + EYE_HEIGHT

                # Victim position (add eye height for head-level)
                vic_x = _safe_float(kill.victim_x)
                vic_y = _safe_float(kill.victim_y)
                vic_z = _safe_float(kill.victim_z) + EYE_HEIGHT

                # Skip if positions are at origin (bad data)
                if abs(att_x) < 0.001 and abs(att_y) < 0.001:
                    continue
                if abs(vic_x) < 0.001 and abs(vic_y) < 0.001:
                    continue

                att_pos = np.array([att_x, att_y, att_z])
                vic_pos = np.array([vic_x, vic_y, vic_z])

                att_pitch = _safe_float(kill.attacker_pitch)
                att_yaw = _safe_float(kill.attacker_yaw)

                angular_error = _compute_angular_error(att_pos, att_pitch, att_yaw, vic_pos)

                if angular_error is not None and 0 <= angular_error <= 180:
                    player_data[att_name]["cp_values"].append(angular_error)

            except Exception:
                continue

    else:
        # Fallback: try to compute from DataFrame columns
        pos_cols = ["attacker_X", "attacker_Y", "attacker_Z", "victim_X", "victim_Y", "victim_Z"]
        angle_cols = ["attacker_pitch", "attacker_yaw"]

        has_pos = all(col in kills_df.columns for col in pos_cols)
        has_angles = all(col in kills_df.columns for col in angle_cols)

        if not kills_df.empty and has_pos and has_angles and att_steamid_col:
            for _, row in kills_df.iterrows():
                try:
                    att_id = row.get(att_steamid_col)
                    if pd.isna(att_id):
                        continue

                    att_name = match_data.player_names.get(int(float(att_id)))
                    if not att_name or att_name not in player_data:
                        continue

                    # Attacker position
                    att_x = _safe_float(row.get("attacker_X"))
                    att_y = _safe_float(row.get("attacker_Y"))
                    att_z = _safe_float(row.get("attacker_Z")) + EYE_HEIGHT

                    # Victim position
                    vic_x = _safe_float(row.get("victim_X"))
                    vic_y = _safe_float(row.get("victim_Y"))
                    vic_z = _safe_float(row.get("victim_Z")) + EYE_HEIGHT

                    # Skip origin positions
                    if abs(att_x) < 0.001 and abs(att_y) < 0.001:
                        continue
                    if abs(vic_x) < 0.001 and abs(vic_y) < 0.001:
                        continue

                    att_pos = np.array([att_x, att_y, att_z])
                    vic_pos = np.array([vic_x, vic_y, vic_z])

                    att_pitch = _safe_float(row.get("attacker_pitch"))
                    att_yaw = _safe_float(row.get("attacker_yaw"))

                    angular_error = _compute_angular_error(att_pos, att_pitch, att_yaw, vic_pos)

                    if angular_error is not None and 0 <= angular_error <= 180:
                        player_data[att_name]["cp_values"].append(angular_error)

                except Exception:
                    continue

    # ===========================================
    # Build final PlayerMetrics objects
    # ===========================================

    result: dict[str, PlayerMetrics] = {}

    for name, data in player_data.items():
        kills = data["kills"]
        damage_total = data["damage_total"]
        ttd_values = data["ttd_values"]
        cp_values = data["cp_values"]

        # Compute ADR
        adr = round(damage_total / num_rounds, 1) if num_rounds > 0 else 0.0

        # Compute TTD statistics
        mean_ttd = None
        median_ttd = None
        if ttd_values:
            mean_ttd = float(np.mean(ttd_values))
            median_ttd = float(np.median(ttd_values))

        # Compute CP statistics
        median_cp = None
        mean_cp = None
        if cp_values:
            median_cp = float(np.median(cp_values))
            mean_cp = float(np.mean(cp_values))

        result[name] = PlayerMetrics(
            player_name=name,
            kills=kills,
            deaths=data["deaths"],
            assists=data["assists"],
            headshot_kills=data["headshot_kills"],
            damage_total=damage_total,
            adr=adr,
            mean_ttd_ms=mean_ttd,
            median_crosshair_error_deg=median_cp,
            median_ttd_ms=median_ttd,
            mean_crosshair_error_deg=mean_cp,
        )

    return result


def calculate_economy_history(match_data: DemoData) -> list[dict]:
    """
    Calculate round-by-round economy history for both teams.

    Uses equipment values from kill/damage events to estimate team wealth
    per round. This enables economy flow visualization for coaching.

    Args:
        match_data: Parsed demo data (DemoData) from DemoParser

    Returns:
        List of dicts with round economy data:
        [
            {"round": 1, "team_t_val": 3500, "team_ct_val": 4500, "t_buy": "pistol", "ct_buy": "pistol"},
            {"round": 2, "team_t_val": 8000, "team_ct_val": 12000, "t_buy": "eco", "ct_buy": "full"},
            ...
        ]

    Example:
        >>> from opensight.parser import DemoParser
        >>> from opensight.analytics import calculate_economy_history
        >>> parser = DemoParser("match.dem")
        >>> data = parser.parse()
        >>> economy = calculate_economy_history(data)
        >>> for round_data in economy:
        ...     print(f"Round {round_data['round']}: T=${round_data['team_t_val']}, CT=${round_data['team_ct_val']}")
    """
    try:
        from opensight.economy import EconomyAnalyzer
    except ImportError:
        logger.warning("Economy module not available, returning empty history")
        return []

    try:
        analyzer = EconomyAnalyzer(match_data)
        stats = analyzer.analyze()
    except Exception as e:
        logger.warning(f"Economy analysis failed: {e}")
        return []

    # Build round-by-round history from team economies
    # Team 2 = T, Team 3 = CT (CS2 convention)
    t_rounds = {tr.round_num: tr for tr in stats.team_economies.get(2, [])}
    ct_rounds = {tr.round_num: tr for tr in stats.team_economies.get(3, [])}

    # Get all round numbers
    all_rounds = sorted(set(t_rounds.keys()) | set(ct_rounds.keys()))

    if not all_rounds:
        # Fallback: Generate basic data from round count if no economy data
        logger.info("No detailed economy data, generating estimates from round count")
        history = []
        for round_num in range(1, match_data.num_rounds + 1):
            is_pistol = round_num in [1, 13, 16, 28]  # Pistol rounds (MR12/MR15)
            is_second = round_num in [2, 14, 17, 29]  # Often eco after pistol loss

            if is_pistol:
                t_val, ct_val = 800, 800
                t_buy, ct_buy = "pistol", "pistol"
            elif is_second:
                t_val, ct_val = 2000, 2000
                t_buy, ct_buy = "eco", "eco"
            else:
                # Estimate mid-game average
                t_val, ct_val = 20000, 22000
                t_buy, ct_buy = "full", "full"

            history.append({
                "round": round_num,
                "team_t_val": t_val,
                "team_ct_val": ct_val,
                "t_buy": t_buy,
                "ct_buy": ct_buy,
            })
        return history

    history = []
    for round_num in all_rounds:
        t_round = t_rounds.get(round_num)
        ct_round = ct_rounds.get(round_num)

        t_val = t_round.total_equipment if t_round else 0
        ct_val = ct_round.total_equipment if ct_round else 0

        # Get buy type string
        t_buy = t_round.buy_type.value if t_round else "unknown"
        ct_buy = ct_round.buy_type.value if ct_round else "unknown"

        history.append({
            "round": round_num,
            "team_t_val": t_val,
            "team_ct_val": ct_val,
            "t_buy": t_buy,
            "ct_buy": ct_buy,
        })

    return history
