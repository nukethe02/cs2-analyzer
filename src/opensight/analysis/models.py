"""
Data Models for CS2 Demo Analysis

This module contains all dataclass definitions and standalone utility functions
used throughout the analytics engine. Extracted from analytics.py for better
code organization and maintainability.

Contains:
- 28 dataclass definitions for various statistics and metrics
- 2 standalone utility functions (calculate_win_probability, calculate_role_scores)
"""

import math
from dataclasses import dataclass, field

import numpy as np

from opensight.core.constants import IMPACT_COEFFICIENTS

# =============================================================================
# Core Engagement and Timing Results
# =============================================================================


@dataclass
class EngagementResult:
    """Duel/engagement duration result (time from first damage to kill).

    Note: This is NOT reaction time. This measures how long it took to
    finish a kill after first dealing damage (spray control, tracking).
    """

    tick_first_damage: int
    tick_kill: int
    duration_ticks: int
    duration_ms: float
    attacker_steamid: int
    victim_steamid: int
    weapon: str
    headshot: bool
    round_num: int = 0


@dataclass
class TrueTTDResult:
    """True Time to Damage - reaction time from visibility to first damage.

    This measures actual human reaction time: how long from when the player
    could first see the enemy until they dealt damage.
    """

    tick_visibility_start: int
    tick_first_damage: int
    reaction_ticks: int
    reaction_ms: float
    attacker_steamid: int
    victim_steamid: int
    weapon: str
    is_prefire: bool  # < 100ms indicates pre-aim/game sense
    round_num: int = 0


# Legacy alias for backwards compatibility
TTDResult = EngagementResult


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


# =============================================================================
# Opening Duels and Entry Stats
# =============================================================================


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
    - map_control_kills: Kills in non-bombsite zones (mid, connectors, etc.)
    - site_kills: Kills in bombsite zones
    """

    wins: int = 0
    losses: int = 0
    attempts: int = 0
    # Entry-specific tracking
    entry_ttd_values: list[float] = field(default_factory=list)
    t_side_entries: int = 0  # Entry kills while on T side (aggressive)
    ct_side_entries: int = 0  # Entry kills while on CT side (defensive)

    # Zone-based classification (Task 2)
    map_control_kills: int = 0  # Opening kills in non-bombsite zones
    site_kills: int = 0  # Opening kills in bombsite zones
    kill_zones: dict[str, int] = field(default_factory=dict)  # zone_name -> count

    # Dry peek tracking - entries without utility support
    supported_entries: int = 0  # Entry kills with teammate flash/smoke support
    unsupported_entries: int = 0  # Entry kills without utility (dry peek wins)
    supported_deaths: int = 0  # Entry deaths with utility support (unlucky)
    unsupported_deaths: int = 0  # Entry deaths without utility (dry peek punished)

    @property
    def dry_peek_rate(self) -> float:
        """Dry Peek % - percentage of opening duels taken without utility support.

        Only counts rifler entries (excludes AWP/Scout which legitimately hold angles).
        High dry peek rate indicates player is taking unnecessary risks without team support.

        Benchmarks:
        - <30%: Good discipline, using utility
        - 30-50%: Average, room for improvement
        - >50%: Too aggressive, need to coordinate with team
        """
        total_entries = self.supported_entries + self.unsupported_entries
        total_deaths = self.supported_deaths + self.unsupported_deaths
        total = total_entries + total_deaths
        if total <= 0:
            return 0.0
        unsupported = self.unsupported_entries + self.unsupported_deaths
        return round(unsupported / total * 100, 1)

    @property
    def dry_peek_death_rate(self) -> float:
        """% of dry peeks that resulted in death (punishment rate for bad plays)."""
        total_dry = self.unsupported_entries + self.unsupported_deaths
        if total_dry <= 0:
            return 0.0
        return round(self.unsupported_deaths / total_dry * 100, 1)

    @property
    def win_rate(self) -> float:
        """Entry Success % - percentage of opening duels won."""
        return round(self.wins / self.attempts * 100, 1) if self.attempts > 0 else 0.0

    @property
    def map_control_rate(self) -> float:
        """Percentage of opening kills that were map control (not site)."""
        if self.wins <= 0:
            return 0.0
        return round(self.map_control_kills / self.wins * 100, 1)

    @property
    def entry_ttd_median_ms(self) -> float | None:
        """Median TTD for entry kills - measures reaction time on first contact."""
        if self.entry_ttd_values:
            return float(np.median(self.entry_ttd_values))
        return None

    @property
    def entry_ttd_mean_ms(self) -> float | None:
        """Mean TTD for entry kills."""
        if self.entry_ttd_values:
            return float(np.mean(self.entry_ttd_values))
        return None

    @property
    def kills(self) -> int:
        """Alias for wins - opening kills count."""
        return self.wins

    @property
    def deaths(self) -> int:
        """Alias for losses - opening deaths count."""
        return self.losses


@dataclass
class OpeningEngagementStats:
    """Opening engagement stats - who FOUGHT first, not just who DIED first.

    An engagement participant = any player who dealt OR received damage
    before the first kill of the round. This captures the true "first contact"
    scenario more accurately than kill-based tracking alone.

    Metrics:
    - engagement_attempts: Rounds where player was involved in pre-kill damage
    - first_damage_dealt: Rounds where player dealt the very first damage
    - opening_damage_total: Total damage dealt during opening phases
    """

    # Core engagement tracking
    engagement_attempts: int = 0  # Rounds involved in pre-kill damage
    engagement_wins: int = 0  # Player's team got first kill
    engagement_losses: int = 0  # Enemy got first kill

    # First damage tracking
    first_damage_dealt: int = 0  # Rounds where this player dealt FIRST damage
    first_damage_taken: int = 0  # Rounds where this player TOOK first damage

    # Damage accumulation during opening phase
    opening_damage_total: int = 0  # Total damage dealt before first kill
    opening_damage_values: list[int] = field(default_factory=list)  # Per-round values

    @property
    def engagement_win_rate(self) -> float:
        """Win rate when involved in opening engagement."""
        if self.engagement_attempts <= 0:
            return 0.0
        return round(self.engagement_wins / self.engagement_attempts * 100, 1)

    @property
    def first_damage_rate(self) -> float:
        """How often this player deals the first damage of a round."""
        if self.engagement_attempts <= 0:
            return 0.0
        return round(self.first_damage_dealt / self.engagement_attempts * 100, 1)

    @property
    def opening_damage_avg(self) -> float:
        """Average damage dealt during opening phases."""
        if not self.opening_damage_values:
            return 0.0
        return round(sum(self.opening_damage_values) / len(self.opening_damage_values), 1)


@dataclass
class EntryFragStats:
    """Zone-aware entry frag stats - first kill INTO a bombsite.

    Entry Frag = first kill in a specific bombsite for a round.
    Distinguished from Opening Duels (map control kills in mid/connectors).

    Metrics:
    - a_site_entries: Entry kills at A bombsite
    - b_site_entries: Entry kills at B bombsite
    - entry_frag_rate: Success rate (kills vs deaths on site entries)
    """

    # Site-specific entry frags
    a_site_entries: int = 0
    a_site_entry_deaths: int = 0
    b_site_entries: int = 0
    b_site_entry_deaths: int = 0

    # Overall entry frag stats
    total_entry_frags: int = 0
    total_entry_deaths: int = 0

    # Round outcomes after entry
    entry_rounds_won: int = 0
    entry_rounds_lost: int = 0

    @property
    def entry_frag_rate(self) -> float:
        """Entry frag success rate (kills vs deaths on site)."""
        total = self.total_entry_frags + self.total_entry_deaths
        if total <= 0:
            return 0.0
        return round(self.total_entry_frags / total * 100, 1)

    @property
    def a_site_success_rate(self) -> float:
        """A site entry success rate."""
        total = self.a_site_entries + self.a_site_entry_deaths
        if total <= 0:
            return 0.0
        return round(self.a_site_entries / total * 100, 1)

    @property
    def b_site_success_rate(self) -> float:
        """B site entry success rate."""
        total = self.b_site_entries + self.b_site_entry_deaths
        if total <= 0:
            return 0.0
        return round(self.b_site_entries / total * 100, 1)

    @property
    def entry_round_win_rate(self) -> float:
        """Win rate in rounds with entry frag."""
        total = self.entry_rounds_won + self.entry_rounds_lost
        if total <= 0:
            return 0.0
        return round(self.entry_rounds_won / total * 100, 1)


# =============================================================================
# Trading and Clutch Stats
# =============================================================================


@dataclass
class TradeStats:
    """Trade kill statistics - measures trading performance (Leetify-style).

    Trade Kill: When a teammate is killed, you kill their killer within 5 seconds.

    Leetify-style metrics:
    - Trade Kill Opportunities: Teammate deaths where you were alive
    - Trade Kill Attempts: Did you engage (damage/fire at) the killer?
    - Trade Kill Success: Did you kill the killer?
    - Traded Death Opportunities: Your deaths where teammates were alive
    - Traded Death Attempts: Did teammates engage your killer?
    - Traded Death Success: Did teammates kill your killer?
    """

    # === Trade Kill Stats (you trading for teammates) ===
    trade_kill_opportunities: int = 0  # Teammate died, you were alive
    trade_kill_attempts: int = 0  # You damaged/shot at the killer
    trade_kill_success: int = 0  # You killed the killer (= kills_traded)

    # === Traded Death Stats (teammates trading for you) ===
    traded_death_opportunities: int = 0  # You died, teammates were alive
    traded_death_attempts: int = 0  # Teammates damaged/shot at your killer
    traded_death_success: int = 0  # Teammates killed your killer (= deaths_traded)

    # === Legacy fields for backwards compatibility ===
    kills_traded: int = 0  # Alias for trade_kill_success
    deaths_traded: int = 0  # Alias for traded_death_success
    trade_attempts: int = 0  # Legacy: opportunities where you could trade
    failed_trades: int = 0  # Legacy: opportunities where you didn't trade

    # === Entry-specific trade stats ===
    traded_entry_kills: int = 0  # Trade kills where original was entry frag
    traded_entry_deaths: int = 0  # Entry deaths traded by teammates

    # === Time to Trade analysis ===
    time_to_trade_ticks: list[int] = field(default_factory=list)

    @property
    def trade_kill_attempts_pct(self) -> float:
        """Trade Kill Attempts % - Did you TRY to trade when given opportunity?"""
        if self.trade_kill_opportunities <= 0:
            return 0.0
        return round(self.trade_kill_attempts / self.trade_kill_opportunities * 100, 1)

    @property
    def trade_kill_success_pct(self) -> float:
        """Trade Kill Success % - Of attempts, how many did you convert?"""
        if self.trade_kill_attempts <= 0:
            return 0.0
        return round(self.trade_kill_success / self.trade_kill_attempts * 100, 1)

    @property
    def traded_death_attempts_pct(self) -> float:
        """Traded Death Attempts % - Did teammates TRY to trade your death?"""
        if self.traded_death_opportunities <= 0:
            return 0.0
        return round(self.traded_death_attempts / self.traded_death_opportunities * 100, 1)

    @property
    def traded_death_success_pct(self) -> float:
        """Traded Death Success % - Of teammate attempts, how many converted?"""
        if self.traded_death_attempts <= 0:
            return 0.0
        return round(self.traded_death_success / self.traded_death_attempts * 100, 1)

    @property
    def trade_rate(self) -> float:
        """Trade Kill % - How often you avenge teammates (legacy, = success/opportunities)."""
        if self.trade_kill_opportunities <= 0:
            return 0.0
        return round(self.trade_kill_success / self.trade_kill_opportunities * 100, 1)

    @property
    def deaths_traded_rate(self) -> float:
        """Percentage of your deaths that were avenged by teammates."""
        if self.traded_death_opportunities <= 0:
            return 0.0
        return round(self.traded_death_success / self.traded_death_opportunities * 100, 1)

    @property
    def avg_time_to_trade_ms(self) -> float | None:
        """Average time to complete a trade in milliseconds."""
        if not self.time_to_trade_ticks:
            return None
        ms_per_tick = 1000.0 / 64.0
        return round(sum(self.time_to_trade_ticks) / len(self.time_to_trade_ticks) * ms_per_tick, 1)

    @property
    def median_time_to_trade_ms(self) -> float | None:
        """Median time to complete a trade in milliseconds."""
        if not self.time_to_trade_ticks:
            return None
        ms_per_tick = 1000.0 / 64.0
        return round(float(np.median(self.time_to_trade_ticks)) * ms_per_tick, 1)


@dataclass
class ClutchEvent:
    """Individual clutch event details."""

    round_number: int
    type: str  # "1v1", "1v2", "1v3", "1v4", "1v5"
    outcome: str  # "WON", "LOST", "SAVED"
    enemies_killed: int = 0
    # Tick information for replay bookmarks
    tick_start: int = 0  # Tick when clutch situation began (last teammate died)
    clutcher_steamid: int = 0  # Steam ID of clutcher
    clutcher_team: str = ""  # "CT" or "T"
    enemies_at_start: int = 0  # Number of enemies when clutch began


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
    clutches: list["ClutchEvent"] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Clutch Won % - Overall clutch success rate across all 1vX situations."""
        return (
            round(self.total_wins / self.total_situations * 100, 1)
            if self.total_situations > 0
            else 0.0
        )

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


# =============================================================================
# Weapon and Multi-Kill Stats
# =============================================================================


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
class SprayTransferStats:
    """Spray transfer kill statistics - 2+ kills in a single spray."""

    double_sprays: int = 0  # 2-kill sprays
    triple_sprays: int = 0  # 3-kill sprays
    quad_sprays: int = 0  # 4-kill sprays
    ace_sprays: int = 0  # 5-kill sprays (full ace in one spray)
    total_spray_kills: int = 0  # Total kills from spray transfers
    _spray_times_ms: list = field(default_factory=list)  # Track times for average

    @property
    def total_sprays(self) -> int:
        """Total spray transfer events."""
        return self.double_sprays + self.triple_sprays + self.quad_sprays + self.ace_sprays

    @property
    def avg_spray_time_ms(self) -> float:
        """Average time span of spray transfers."""
        if not self._spray_times_ms:
            return 0.0
        return round(sum(self._spray_times_ms) / len(self._spray_times_ms), 1)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "double_sprays": self.double_sprays,
            "triple_sprays": self.triple_sprays,
            "quad_sprays": self.quad_sprays,
            "ace_sprays": self.ace_sprays,
            "total_sprays": self.total_sprays,
            "total_spray_kills": self.total_spray_kills,
            "avg_spray_time_ms": self.avg_spray_time_ms,
        }


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


# =============================================================================
# Utility Stats
# =============================================================================


@dataclass
class UtilityStats:
    """Comprehensive utility usage statistics (Leetify-style)."""

    # Grenade counts
    flashbangs_thrown: int = 0
    smokes_thrown: int = 0
    he_thrown: int = 0
    molotovs_thrown: int = 0

    # Flash effectiveness (attacker-side)
    enemies_flashed: int = 0  # Enemies blinded > 1.5s (significant)
    teammates_flashed: int = 0  # Teammates blinded > 1.5s (mistake)
    flash_assists: int = 0  # Kills within 3s of blinding enemy
    total_blind_time: float = 0.0  # Total seconds enemies were blinded
    effective_flashes: int = 0  # Unique flashbangs with >= 1 significant enemy blind

    # Flash received (victim-side) - Leetify "Avg Blind Time"
    times_blinded: int = 0  # How many times player was blinded by enemies
    total_time_blinded: float = 0.0  # Total seconds player was blinded

    # Damage stats
    he_damage: int = 0  # Damage to enemies from HE grenades
    he_team_damage: int = 0  # Damage to teammates from HE
    molotov_damage: int = 0  # Damage to enemies from molotov/incendiary
    molotov_team_damage: int = 0  # Damage to teammates from fire

    # Economy - unused utility at death
    unused_utility_value: int = 0  # Average $ of utility not used when dying

    # Round context (set by analyzer)
    _rounds_played: int = 0

    @property
    def total_utility(self) -> int:
        """Total grenades thrown."""
        return self.flashbangs_thrown + self.smokes_thrown + self.he_thrown + self.molotovs_thrown

    @property
    def total_utility_damage(self) -> int:
        """Total damage dealt with utility (HE + Molotov)."""
        return self.he_damage + self.molotov_damage

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
    def avg_enemies_per_flash(self) -> float:
        """Average enemies significantly blinded per effective flash (Flash Efficiency)."""
        if self.effective_flashes <= 0:
            return 0.0
        return round(self.enemies_flashed / self.effective_flashes, 2)

    @property
    def flash_effectiveness_pct(self) -> float:
        """Percentage of flashes that were effective (had >= 1 significant blind)."""
        if self.flashbangs_thrown <= 0:
            return 0.0
        return round(self.effective_flashes / self.flashbangs_thrown * 100, 1)

    @property
    def enemies_flashed_per_round(self) -> float:
        """Average enemies flashed per round (Leetify column)."""
        if self._rounds_played <= 0:
            return 0.0
        return round(self.enemies_flashed / self._rounds_played, 2)

    @property
    def friends_flashed_per_round(self) -> float:
        """Average teammates flashed per round (Leetify column)."""
        if self._rounds_played <= 0:
            return 0.0
        return round(self.teammates_flashed / self._rounds_played, 2)

    @property
    def avg_blind_time(self) -> float:
        """Average blind time per enemy flashed (seconds)."""
        if self.enemies_flashed <= 0:
            return 0.0
        return round(self.total_blind_time / self.enemies_flashed, 2)

    @property
    def avg_time_blinded(self) -> float:
        """Average blind time received per flash (victim-side, Leetify 'Avg Blind Time')."""
        if self.times_blinded <= 0:
            return 0.0
        return round(self.total_time_blinded / self.times_blinded, 2)

    @property
    def avg_he_damage(self) -> float:
        """Average HE damage per round."""
        if self._rounds_played <= 0:
            return 0.0
        return round(self.he_damage / self._rounds_played, 1)

    @property
    def avg_he_team_damage(self) -> float:
        """Average HE team damage per round."""
        if self._rounds_played <= 0:
            return 0.0
        return round(self.he_team_damage / self._rounds_played, 1)

    @property
    def he_damage_per_nade(self) -> float:
        """Average HE damage per grenade."""
        if self.he_thrown <= 0:
            return 0.0
        return round(self.he_damage / self.he_thrown, 1)

    @property
    def molotov_damage_per_nade(self) -> float:
        """Average molotov damage per grenade."""
        if self.molotovs_thrown <= 0:
            return 0.0
        return round(self.molotov_damage / self.molotovs_thrown, 1)

    @property
    def flash_assist_pct(self) -> float:
        """Percentage of flashes that resulted in assists (Leetify column)."""
        if self.flashbangs_thrown <= 0:
            return 0.0
        return round(self.flash_assists / self.flashbangs_thrown * 100, 1)

    @property
    def utility_quality_rating(self) -> float:
        """
        Utility Quality Rating (0-100) - how effective utility is when used.

        Based on:
        - Flash effectiveness (enemy blind ratio vs teammate blind)
        - Average blind duration
        - HE damage per grenade
        - Flash assists

        Higher = better utility usage
        """
        if self.total_utility <= 0:
            return 0.0

        score = 0.0

        # Flash quality (0-40 points)
        if self.flashbangs_thrown > 0:
            # Enemies flashed vs teammates (ratio bonus)
            enemy_ratio = self.enemies_flashed / self.flashbangs_thrown
            team_ratio = self.teammates_flashed / self.flashbangs_thrown
            flash_score = min(20, enemy_ratio * 10)  # Up to 20 for good enemy flash rate
            flash_score -= min(10, team_ratio * 5)  # Penalty for team flashes
            flash_score = max(0, flash_score)

            # Blind duration bonus (up to 10 points)
            if self.avg_blind_time >= 2.5:
                flash_score += 10
            elif self.avg_blind_time >= 1.5:
                flash_score += 5

            # Flash assist bonus (up to 10 points)
            assist_rate = self.flash_assists / self.flashbangs_thrown
            flash_score += min(10, assist_rate * 20)

            score += flash_score

        # HE quality (0-30 points)
        if self.he_thrown > 0:
            dmg_per_nade = self.he_damage_per_nade
            if dmg_per_nade >= 50:
                score += 30
            elif dmg_per_nade >= 30:
                score += 20
            elif dmg_per_nade >= 15:
                score += 10
            # Penalty for team damage
            if self.he_team_damage > 0:
                team_dmg_ratio = self.he_team_damage / max(1, self.he_damage + self.he_team_damage)
                score -= min(10, team_dmg_ratio * 20)

        # Molotov quality (0-20 points)
        if self.molotovs_thrown > 0:
            dmg_per_molly = self.molotov_damage_per_nade
            if dmg_per_molly >= 40:
                score += 20
            elif dmg_per_molly >= 20:
                score += 12
            elif dmg_per_molly >= 10:
                score += 6

        # Smoke quality (0-10 points) - assume smokes are always useful if thrown
        if self.smokes_thrown > 0:
            score += 10

        return round(min(100, max(0, score)), 1)

    @property
    def utility_quantity_rating(self) -> float:
        """
        Utility Quantity Rating (0-100) - how much utility is used per round.

        Based on:
        - Grenades thrown per round vs expected (4 per round max)
        - Actually using utility (not dying with nades)

        Higher = using more utility appropriately
        """
        if self._rounds_played <= 0:
            return 0.0

        # Expected max utility per round is about 4 grenades
        expected_per_round = 4.0
        utility_per_round = self.total_utility / self._rounds_played

        # Base score from utility usage rate
        usage_rate = min(1.0, utility_per_round / expected_per_round)
        score = usage_rate * 80  # Up to 80 points for using utility

        # Penalty for dying with unused utility (if tracked)
        if self.unused_utility_value > 0:
            avg_unused = self.unused_utility_value / self._rounds_played
            # Each $300 unused = 5 point penalty (max 20)
            penalty = min(20, avg_unused / 300 * 5)
            score -= penalty

        # Bonus for variety (using all types)
        types_used = sum(
            [
                1 if self.flashbangs_thrown > 0 else 0,
                1 if self.smokes_thrown > 0 else 0,
                1 if self.he_thrown > 0 else 0,
                1 if self.molotovs_thrown > 0 else 0,
            ]
        )
        score += types_used * 5  # Up to 20 points for variety

        return round(min(100, max(0, score)), 1)

    # Backwards compatibility alias
    @property
    def molotov_thrown(self) -> int:
        return self.molotovs_thrown

    @molotov_thrown.setter
    def molotov_thrown(self, value: int) -> None:
        self.molotovs_thrown = value

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "flashbangs_thrown": self.flashbangs_thrown,
            "smokes_thrown": self.smokes_thrown,
            "he_thrown": self.he_thrown,
            "molotovs_thrown": self.molotovs_thrown,
            "total_utility": self.total_utility,
            "enemies_flashed": self.enemies_flashed,
            "teammates_flashed": self.teammates_flashed,
            "effective_flashes": self.effective_flashes,
            "avg_enemies_per_flash": self.avg_enemies_per_flash,
            "flash_effectiveness_pct": self.flash_effectiveness_pct,
            "flash_assists": self.flash_assists,
            "flash_assist_pct": self.flash_assist_pct,
            "enemies_flashed_per_round": self.enemies_flashed_per_round,
            "friends_flashed_per_round": self.friends_flashed_per_round,
            "avg_blind_time": self.avg_blind_time,
            "total_blind_time": round(self.total_blind_time, 2),
            # Victim-side blind metrics (Leetify "Avg Blind Time")
            "times_blinded": self.times_blinded,
            "total_time_blinded": round(self.total_time_blinded, 2),
            "avg_time_blinded": self.avg_time_blinded,
            "he_damage": self.he_damage,
            "he_team_damage": self.he_team_damage,
            "avg_he_damage": self.avg_he_damage,
            "avg_he_team_damage": self.avg_he_team_damage,
            "he_damage_per_nade": self.he_damage_per_nade,
            "molotov_damage": self.molotov_damage,
            "molotov_team_damage": self.molotov_team_damage,
            "molotov_damage_per_nade": self.molotov_damage_per_nade,
            "total_utility_damage": self.total_utility_damage,
            "unused_utility_value": self.unused_utility_value,
            "utility_quality_rating": self.utility_quality_rating,
            "utility_quantity_rating": self.utility_quantity_rating,
        }


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
    flashes_enemies_total: int = 0  # Number of enemy players flashed (>1.1s blind)
    flashes_teammates_total: int = 0  # Number of teammates flashed (mistake tracking)
    flash_assists: int = 0  # Kills on enemies player flashed
    total_blind_time: float = 0.0  # Total seconds enemies were blinded

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


# =============================================================================
# Side Stats and Mistakes
# =============================================================================


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

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "damage": self.damage,
            "rounds_played": self.rounds_played,
            "kd_ratio": self.kd_ratio,
            "adr": self.adr,
        }


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
        return (
            (self.team_kills * 10)
            + (self.team_damage // 10)
            + self.teammates_flashed
            + (self.suicides * 5)
        )


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
class AimStats:
    """Comprehensive aim statistics (Leetify style).

    Metrics tracked:
    - spotted_accuracy: % of hits when enemy visible (requires visibility data)
    - time_to_damage_ms: Median reaction time from seeing enemy to dealing damage
    - crosshair_placement_deg: Median angular error at moment of kill
    - head_accuracy: % of hits that were headshots
    - hs_kill_pct: % of kills that were headshots
    - spray_accuracy: % of hits after 4th bullet in a burst
    - counter_strafe_pct: % of kills while near-stationary
    - accuracy_all: Overall accuracy (hits / shots fired)
    """

    # Core accuracy metrics
    shots_fired: int = 0
    shots_hit: int = 0
    headshot_hits: int = 0  # Hits to head (not just kills)

    # Spray tracking (hits/shots after 4th bullet in burst)
    spray_shots_fired: int = 0
    spray_shots_hit: int = 0

    # Counter-strafing (shots while near-stationary)
    # NEW: Shot-based tracking (Leetify parity)
    shots_stationary: int = 0  # Shots fired with velocity < 34 u/s
    shots_with_velocity: int = 0  # Total shots where velocity was measurable
    # DEPRECATED: Kill-based tracking (kept for backward compatibility)
    counter_strafe_kills: int = 0
    total_kills_for_cs: int = 0  # Total kills where velocity was tracked

    # TTD and CP (from existing implementation)
    ttd_median_ms: float | None = None
    ttd_mean_ms: float | None = None
    cp_median_deg: float | None = None
    cp_mean_deg: float | None = None

    # Headshot stats
    total_kills: int = 0
    headshot_kills: int = 0

    @property
    def accuracy_all(self) -> float:
        """Overall accuracy - shots hit / shots fired."""
        return round(self.shots_hit / self.shots_fired * 100, 1) if self.shots_fired > 0 else 0.0

    @property
    def head_accuracy(self) -> float:
        """Head Accuracy - % of hits that were headshots."""
        return round(self.headshot_hits / self.shots_hit * 100, 1) if self.shots_hit > 0 else 0.0

    @property
    def hs_kill_pct(self) -> float:
        """HS Kill % - % of kills that were headshots."""
        return (
            round(self.headshot_kills / self.total_kills * 100, 1) if self.total_kills > 0 else 0.0
        )

    @property
    def spray_accuracy(self) -> float:
        """Spray Accuracy - % of hits after 4th bullet in a burst."""
        return (
            round(self.spray_shots_hit / self.spray_shots_fired * 100, 1)
            if self.spray_shots_fired > 0
            else 0.0
        )

    @property
    def counter_strafe_pct(self) -> float:
        """Counter-Strafing % - % of shots fired while near-stationary (< 34 velocity).

        This is the Leetify-style metric measuring movement discipline on ALL shots,
        not just kill shots. Higher is better - indicates proper counter-strafing.
        """
        return (
            round(self.shots_stationary / self.shots_with_velocity * 100, 1)
            if self.shots_with_velocity > 0
            else 0.0
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            # Raw counts
            "shots_fired": self.shots_fired,
            "shots_hit": self.shots_hit,
            "headshot_hits": self.headshot_hits,
            "spray_shots_fired": self.spray_shots_fired,
            "spray_shots_hit": self.spray_shots_hit,
            # NEW: Shot-based counter-strafe tracking
            "shots_stationary": self.shots_stationary,
            "shots_with_velocity": self.shots_with_velocity,
            # DEPRECATED: Kill-based tracking (kept for backward compatibility)
            "counter_strafe_kills": self.counter_strafe_kills,
            "total_kills_for_cs": self.total_kills_for_cs,
            # Computed percentages (Leetify format)
            "accuracy_all": self.accuracy_all,
            "head_accuracy": self.head_accuracy,
            "hs_kill_pct": self.hs_kill_pct,
            "spray_accuracy": self.spray_accuracy,
            "counter_strafe_pct": self.counter_strafe_pct,
            # TTD and CP
            "time_to_damage_ms": round(self.ttd_median_ms, 1) if self.ttd_median_ms else None,
            "crosshair_placement_deg": (
                round(self.cp_median_deg, 1) if self.cp_median_deg else None
            ),
            # Benchmark indicators
            "ttd_rating": self._get_ttd_rating(),
            "cp_rating": self._get_cp_rating(),
        }

    def _get_ttd_rating(self) -> str:
        """Get TTD rating category."""
        if self.ttd_median_ms is None:
            return "unknown"
        if self.ttd_median_ms < 200:
            return "elite"
        elif self.ttd_median_ms < 350:
            return "good"
        elif self.ttd_median_ms < 500:
            return "average"
        return "slow"

    def _get_cp_rating(self) -> str:
        """Get Crosshair Placement rating category."""
        if self.cp_median_deg is None:
            return "unknown"
        if self.cp_median_deg < 5:
            return "elite"
        elif self.cp_median_deg < 15:
            return "good"
        elif self.cp_median_deg < 25:
            return "average"
        return "needs_work"


# =============================================================================
# Player Match Stats (Main Stats Container)
# =============================================================================


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
    opening_engagements: OpeningEngagementStats = field(default_factory=OpeningEngagementStats)
    entry_frags: EntryFragStats = field(default_factory=EntryFragStats)
    trades: TradeStats = field(default_factory=TradeStats)
    clutches: ClutchStats = field(default_factory=ClutchStats)
    multi_kills: MultiKillStats = field(default_factory=MultiKillStats)
    spray_transfers: SprayTransferStats = field(default_factory=SprayTransferStats)
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
    effective_flashes: int = 0  # Flashes > 2.0 seconds blind duration
    ineffective_flashes: int = 0  # Flashes < 2.0 seconds
    utility_adr: float = 0.0  # HE + Molotov damage per round

    # Weapon breakdown
    weapon_kills: dict = field(default_factory=dict)

    # Engagement Duration stats (time from first damage to kill - measures spray/tracking)
    engagement_duration_values: list = field(default_factory=list)

    # True TTD stats (reaction time: visibility to first damage)
    true_ttd_values: list = field(default_factory=list)
    prefire_count: int = 0  # Kills with reaction time < 100ms (pre-aim/game sense)

    # Legacy alias for backwards compatibility
    @property
    def ttd_values(self) -> list:
        """Legacy alias - returns engagement duration values."""
        return self.engagement_duration_values

    # CP stats
    cp_values: list = field(default_factory=list)

    # Accuracy stats (Leetify style)
    shots_fired: int = 0
    shots_hit: int = 0
    headshot_hits: int = 0  # Hits to head (not just kills)

    # Spray accuracy (hits after 4th bullet in burst)
    spray_shots_fired: int = 0
    spray_shots_hit: int = 0

    # Counter-strafing (shots while near-stationary) - Leetify parity
    # NEW: Shot-based tracking (measures technique across ALL shots)
    shots_stationary: int = 0  # Shots fired with velocity < 34 u/s
    shots_with_velocity: int = 0  # Total shots where velocity was measurable
    # DEPRECATED: Kill-based tracking (kept for backward compatibility)
    counter_strafe_kills: int = 0
    total_kills_with_velocity: int = 0  # Kills where velocity was trackable

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

    # Discipline stats (greedy play detection)
    greedy_repeeks: int = 0  # Deaths from re-peeking same angle after a kill
    discipline_rating: float = 100.0  # (safe_kills / total_kills) * 100

    # RWS (Round Win Shares) - ESEA style
    rws: float = 0.0  # Average RWS across rounds won
    damage_in_won_rounds: int = 0
    rounds_won: int = 0

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
        return (
            round(self.kast_rounds / self.rounds_played * 100, 1) if self.rounds_played > 0 else 0.0
        )

    @property
    def survival_rate(self) -> float:
        return (
            round(self.rounds_survived / self.rounds_played * 100, 1)
            if self.rounds_played > 0
            else 0.0
        )

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
            IMPACT_COEFFICIENTS["kpr"] * self.kills_per_round
            + IMPACT_COEFFICIENTS["apr"] * self.assists_per_round
            + IMPACT_COEFFICIENTS["base"]
        )
        # Add clutch bonus
        clutch_bonus = self.clutches.total_wins * 0.1
        # Add multi-kill bonus
        mk_bonus = (
            self.multi_kills.rounds_with_3k * 0.1
            + self.multi_kills.rounds_with_4k * 0.2
            + self.multi_kills.rounds_with_5k * 0.3
        )
        return round(base + clutch_bonus + mk_bonus, 3)

    @property
    def hltv_rating(self) -> float:
        """
        HLTV 2.0 Rating using the verified engine from hltv_rating.py.

        Formula: 0.0073*KAST + 0.3591*KPR - 0.5329*DPR +
                 0.2372*Impact + 0.0032*ADR + 0.1587*RMK

        Uses calculate_rating_from_stats for proper handling of:
        - KAST normalization (percentage vs decimal)
        - RMK as decimal (not percentage)
        - Safe accessors with defaults
        - Output clamping to [0.0, 3.0]
        """
        # Import here to avoid circular dependency
        from opensight.analysis.hltv_rating import calculate_rating_from_stats

        # Build stats dict for the rating engine
        stats = {
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "adr": self.adr,
            "kast": self.kast_percentage,  # Already percentage (0-100)
            "2k": self.multi_kills.rounds_with_2k if self.multi_kills else 0,
            "3k": self.multi_kills.rounds_with_3k if self.multi_kills else 0,
            "4k": self.multi_kills.rounds_with_4k if self.multi_kills else 0,
            "5k": self.multi_kills.rounds_with_5k if self.multi_kills else 0,
            "clutch_wins": self.clutches.total_wins if self.clutches else 0,
        }

        return calculate_rating_from_stats(stats, self.rounds_played)

    @property
    def impact_plus_minus(self) -> float:
        """
        Leetify-style Impact +/- Rating centered on 0.00.

        Positive weights (reward high-impact plays):
        - Opening Kill: +0.50 (putting team at 5v4)
        - Clutch Win: +1.00 (round-winning play)
        - Trade Kill: +0.30 (maintaining man advantage)
        - Multi-Kill 2K: +0.25, 3K: +0.75, 4K: +1.25, 5K: +2.00
        - Flash Assist: +0.25 (enabling teammates)

        Negative weights (penalize liability plays):
        - Opening Death: -0.50 (putting team at 4v5)
        - Untraded Death: -0.25 (dying for free)
        - Team Flash: -0.15 (hindering teammates)

        Formula: (Total Impact Points / Rounds) * 10
        Scale factor of 10 makes "carry" performances reach ~+5.0

        Benchmarks:
        - +5.00 or higher: Hard carry
        - +2.00 to +5.00: Strong performance
        - +0.50 to +2.00: Above average
        - -0.50 to +0.50: Average
        - -2.00 to -0.50: Below average
        - -5.00 or lower: Liability
        """
        if self.rounds_played == 0:
            return 0.0

        # === POSITIVE IMPACT ===
        # Opening kills - high value, puts team at advantage
        opening_kills_points = self.opening_duels.kills * 0.50

        # Clutch wins - round-winning plays
        clutch_wins_points = self.clutches.total_wins * 1.00

        # Trade kills - maintaining numbers advantage
        trade_kills_points = self.trades.trade_kill_success * 0.30

        # Multi-kills - scaled bonuses (extra value beyond first kill)
        # 2K = +0.25 (1 extra kill), 3K = +0.75 (2 extra), 4K = +1.25 (3 extra), 5K = +2.00 (4 extra)
        multi_kill_points = (
            self.multi_kills.rounds_with_2k * 0.25
            + self.multi_kills.rounds_with_3k * 0.75
            + self.multi_kills.rounds_with_4k * 1.25
            + self.multi_kills.rounds_with_5k * 2.00
        )

        # Flash assists - enabling teammate kills
        flash_assist_points = self.utility.flash_assists * 0.25

        # === NEGATIVE IMPACT ===
        # Opening deaths - putting team at disadvantage
        opening_deaths_points = self.opening_duels.deaths * -0.50

        # Untraded deaths - dying without value
        untraded_deaths_points = self.untraded_deaths * -0.25

        # Team flashes - hindering teammates
        team_flash_points = self.utility.teammates_flashed * -0.15

        # === TOTAL ===
        total_points = (
            opening_kills_points
            + clutch_wins_points
            + trade_kills_points
            + multi_kill_points
            + flash_assist_points
            + opening_deaths_points
            + untraded_deaths_points
            + team_flash_points
        )

        # Normalize by rounds and scale by 10 to reach ~+/-5.0 range for carry/liability
        impact_per_round = total_points / self.rounds_played
        scaled_impact = impact_per_round * 10

        return round(scaled_impact, 2)

    # Engagement Duration properties (time from first damage to kill)
    @property
    def engagement_duration_median_ms(self) -> float | None:
        """Median time from first damage to kill (spray control/tracking)."""
        return (
            float(np.median(self.engagement_duration_values))
            if self.engagement_duration_values
            else None
        )

    @property
    def engagement_duration_mean_ms(self) -> float | None:
        """Mean time from first damage to kill."""
        return (
            float(np.mean(self.engagement_duration_values))
            if self.engagement_duration_values
            else None
        )

    # Legacy TTD aliases (these now return engagement duration, not reaction time)
    @property
    def ttd_median_ms(self) -> float | None:
        """Legacy alias - returns engagement_duration_median_ms."""
        return self.engagement_duration_median_ms

    @property
    def ttd_mean_ms(self) -> float | None:
        """Legacy alias - returns engagement_duration_mean_ms."""
        return self.engagement_duration_mean_ms

    # True TTD properties (reaction time: visibility to first damage)
    @property
    def reaction_time_median_ms(self) -> float | None:
        """Median reaction time (visibility to first damage) - TRUE TTD."""
        return float(np.median(self.true_ttd_values)) if self.true_ttd_values else None

    @property
    def reaction_time_mean_ms(self) -> float | None:
        """Mean reaction time (visibility to first damage)."""
        return float(np.mean(self.true_ttd_values)) if self.true_ttd_values else None

    @property
    def prefire_percentage(self) -> float:
        """Percentage of engagements that were prefires (reaction time < 100ms).

        Prefires indicate high game-sense - anticipating enemy positions.
        """
        if not self.true_ttd_values:
            # Fallback to engagement duration count if no true TTD data
            if not self.engagement_duration_values:
                return 0.0
            return round(self.prefire_count / len(self.engagement_duration_values) * 100, 1)
        return round(self.prefire_count / len(self.true_ttd_values) * 100, 1)

    # CP properties
    @property
    def cp_median_error_deg(self) -> float | None:
        return float(np.median(self.cp_values)) if self.cp_values else None

    @property
    def cp_mean_error_deg(self) -> float | None:
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

    @property
    def spray_accuracy(self) -> float:
        """Spray Accuracy - % of hits after 4th bullet in a burst."""
        return (
            round(self.spray_shots_hit / self.spray_shots_fired * 100, 1)
            if self.spray_shots_fired > 0
            else 0.0
        )

    @property
    def counter_strafe_pct(self) -> float:
        """Counter-Strafing % - % of shots fired while near-stationary (< 34 velocity).

        This is the Leetify-style metric measuring movement discipline on ALL shots,
        not just kill shots. Higher is better - indicates proper counter-strafing.
        """
        return (
            round(self.shots_stationary / self.shots_with_velocity * 100, 1)
            if self.shots_with_velocity > 0
            else 0.0
        )

    @property
    def aim_stats(self) -> AimStats:
        """Get comprehensive aim stats as AimStats dataclass."""
        return AimStats(
            shots_fired=self.shots_fired,
            shots_hit=self.shots_hit,
            headshot_hits=self.headshot_hits,
            spray_shots_fired=self.spray_shots_fired,
            spray_shots_hit=self.spray_shots_hit,
            # NEW: Shot-based counter-strafe tracking
            shots_stationary=self.shots_stationary,
            shots_with_velocity=self.shots_with_velocity,
            # DEPRECATED: Kill-based tracking (kept for backward compatibility)
            counter_strafe_kills=self.counter_strafe_kills,
            total_kills_for_cs=self.total_kills_with_velocity,
            ttd_median_ms=self.ttd_median_ms,
            ttd_mean_ms=self.ttd_mean_ms,
            cp_median_deg=self.cp_median_error_deg,
            cp_mean_deg=self.cp_mean_error_deg,
            total_kills=self.kills,
            headshot_kills=self.headshots,
        )

    # Utility Rating (Leetify style composite)
    @property
    def utility_quantity_rating(self) -> float:
        """
        Leetify-style Utility Quantity Rating.
        Based on utility thrown vs expected (3 per round).
        Uses x^(2/3) scaling, max 100.
        """
        total_utility = (
            self.utility.flashbangs_thrown
            + self.utility.he_thrown
            + self.utility.molotovs_thrown
            + self.utility.smokes_thrown
        )
        expected = 3.0 * self.rounds_played
        if expected <= 0:
            return 0.0
        ratio = min(total_utility / expected, 1.0)  # Cap at 1.0
        # Apply x^(2/3) scaling
        scaled = ratio ** (2 / 3)
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

    # Aim Rating (penalty-based formula with fallback)
    @property
    def aim_rating(self) -> float:
        """
        Aim Rating using penalty-based formula.

        Primary Formula (requires tick data):
            Aim = 100 - (TTD_Penalty + CP_Penalty + Recoil_Penalty)
            - TTD_Penalty: Based on reaction time (150ms elite → 600ms slow)
            - CP_Penalty: Based on crosshair placement error (3° elite → 30° poor)
            - Recoil_Penalty: Based on spray accuracy (80% elite → 0% poor)

        Fallback Formula (when tick data unavailable):
            Aim = HS_Score * 0.35 + TTK_Score * 0.25 + CP_Score * 0.25 + Acc_Score * 0.15
            Uses engagement duration, headshot %, CP, and overall accuracy

        Score is 0-100 where higher is better.
        """
        # Try primary formula first (requires tick-level data)
        reaction_time = self.reaction_time_median_ms
        cp_error = self.cp_median_error_deg

        if reaction_time is not None and reaction_time > 0 and cp_error is not None:
            return self._aim_rating_primary(reaction_time, cp_error)

        # Fallback: Use available metrics when tick data isn't available
        return self._aim_rating_fallback()

    def _aim_rating_primary(self, reaction_time: float, cp_error: float) -> float:
        """Primary aim rating using reaction time + CP + spray accuracy."""
        # Import logger here to avoid circular dependency
        import logging

        logger = logging.getLogger(__name__)

        ttd_penalty = 0.0
        error_penalty = 0.0
        recoil_penalty = 0.0

        # UNIT VERIFICATION: Check if TTD is in milliseconds (should be 200ms not 0.2s)
        if reaction_time < 10:
            logger.warning(
                f"Player {self.name}: reaction_time={reaction_time} is too low! "
                f"Likely in seconds. Converting: {reaction_time}s → {reaction_time * 1000}ms"
            )
            reaction_time = reaction_time * 1000

        # Scale: 150ms (elite) = 0, 250ms (good) = 10, 400ms (avg) = 25, 600ms (slow) = 40
        if reaction_time <= 150:
            ttd_penalty = 0
        elif reaction_time <= 250:
            ttd_penalty = (reaction_time - 150) / 10  # 0-10
        elif reaction_time <= 400:
            ttd_penalty = 10 + (reaction_time - 250) / 10  # 10-25
        else:
            ttd_penalty = 25 + (reaction_time - 400) / 13.33  # 25-40 at 600ms
        ttd_penalty = min(40, ttd_penalty)  # Cap at 40

        # Scale: 3° (elite) = 0, 5° (good) = 5, 10° (avg) = 15, 20° (poor) = 30, 30° = 40
        if cp_error <= 3:
            error_penalty = 0
        elif cp_error <= 5:
            error_penalty = (cp_error - 3) * 2.5  # 0-5
        elif cp_error <= 10:
            error_penalty = 5 + (cp_error - 5) * 2  # 5-15
        elif cp_error <= 20:
            error_penalty = 15 + (cp_error - 10) * 1.5  # 15-30
        else:
            error_penalty = 30 + (cp_error - 20) * 1  # 30-40 at 40°
        error_penalty = min(40, error_penalty)  # Cap at 40

        # Recoil Control Penalty (spray accuracy)
        spray_acc = self.spray_accuracy
        if spray_acc >= 80:
            recoil_penalty = 0
        elif spray_acc >= 60:
            recoil_penalty = (80 - spray_acc) / 2  # 0-10
        elif spray_acc >= 40:
            recoil_penalty = 10 + (60 - spray_acc) / 2  # 10-20
        elif spray_acc >= 20:
            recoil_penalty = 20 + (40 - spray_acc) / 2  # 20-30
        else:
            recoil_penalty = 30 + (20 - spray_acc) / 4  # 30-35
        recoil_penalty = min(35, recoil_penalty)  # Cap at 35

        # Calculate final score
        total_penalty = ttd_penalty + error_penalty + recoil_penalty
        aim_score = 100 - total_penalty

        logger.debug(
            f"Aim (primary) for {self.name}: "
            f"TTD_penalty={ttd_penalty:.1f} (reaction={reaction_time}ms), "
            f"CP_penalty={error_penalty:.1f} (cp={cp_error}°), "
            f"Recoil_penalty={recoil_penalty:.1f} (spray={spray_acc}%), "
            f"Final={aim_score:.1f}"
        )

        return round(max(0, min(100, aim_score)), 1)

    def _aim_rating_fallback(self) -> float:
        """Fallback aim rating using HS%, engagement duration, CP (if available), and accuracy.

        Used when tick-level data isn't available for true reaction time calculation.
        """
        # Import logger here to avoid circular dependency
        import logging

        logger = logging.getLogger(__name__)

        # Component weights (must sum to 1.0)
        hs_weight = 0.35  # Headshot % is a strong aim indicator
        ttk_weight = 0.25  # Time to kill (engagement duration)
        cp_weight = 0.25  # Crosshair placement if available
        acc_weight = 0.15  # Overall accuracy

        # 1. Headshot Score (0-100)
        # Elite: 60%+ HS = 100, Average: 35% = 50, Poor: 10% = 0
        hs_pct = self.headshot_percentage
        if hs_pct >= 60:
            hs_score = 100
        elif hs_pct >= 35:
            hs_score = 50 + (hs_pct - 35) * 2  # 50-100
        elif hs_pct >= 10:
            hs_score = (hs_pct - 10) * 2  # 0-50
        else:
            hs_score = 0

        # 2. Time to Kill Score (engagement duration - first damage to kill)
        # Elite: <200ms = 100, Good: 400ms = 70, Average: 700ms = 40, Slow: >1000ms = 0
        ttk_ms = self.ttd_median_ms  # This is engagement duration
        if ttk_ms is None or ttk_ms <= 0:
            ttk_score = 50  # Neutral if no data
        elif ttk_ms <= 200:
            ttk_score = 100
        elif ttk_ms <= 400:
            ttk_score = 70 + (400 - ttk_ms) / 200 * 30  # 70-100
        elif ttk_ms <= 700:
            ttk_score = 40 + (700 - ttk_ms) / 300 * 30  # 40-70
        elif ttk_ms <= 1000:
            ttk_score = (1000 - ttk_ms) / 300 * 40  # 0-40
        else:
            ttk_score = 0

        # 3. Crosshair Placement Score (if available)
        cp_error = self.cp_median_error_deg
        if cp_error is not None and cp_error >= 0:
            # Elite: <3° = 100, Good: 5° = 80, Average: 10° = 50, Poor: 20° = 20
            if cp_error <= 3:
                cp_score = 100
            elif cp_error <= 5:
                cp_score = 80 + (5 - cp_error) * 10  # 80-100
            elif cp_error <= 10:
                cp_score = 50 + (10 - cp_error) * 6  # 50-80
            elif cp_error <= 20:
                cp_score = 20 + (20 - cp_error) * 3  # 20-50
            else:
                cp_score = max(0, 20 - (cp_error - 20))  # 0-20
        else:
            # If no CP data, redistribute weight to other components
            cp_score = 50  # Neutral

        # 4. Accuracy Score (overall shots hit / shots fired)
        accuracy = self.accuracy
        if accuracy > 0:
            # Elite: 40%+ = 100, Good: 30% = 70, Average: 20% = 40, Poor: 10% = 10
            if accuracy >= 40:
                acc_score = 100
            elif accuracy >= 30:
                acc_score = 70 + (accuracy - 30) * 3  # 70-100
            elif accuracy >= 20:
                acc_score = 40 + (accuracy - 20) * 3  # 40-70
            elif accuracy >= 10:
                acc_score = 10 + (accuracy - 10) * 3  # 10-40
            else:
                acc_score = accuracy  # 0-10
        else:
            acc_score = 50  # Neutral if no data

        # Calculate weighted score
        aim_score = (
            hs_score * hs_weight
            + ttk_score * ttk_weight
            + cp_score * cp_weight
            + acc_score * acc_weight
        )

        logger.debug(
            f"Aim (fallback) for {self.name}: "
            f"HS={hs_score:.0f} (hs%={hs_pct:.1f}), "
            f"TTK={ttk_score:.0f} (ttk={ttk_ms}ms), "
            f"CP={cp_score:.0f} (cp={cp_error}°), "
            f"Acc={acc_score:.0f} (acc={accuracy:.1f}%), "
            f"Final={aim_score:.1f}"
        )

        return round(max(0, min(100, aim_score)), 1)

    @property
    def entry_success_rate(self) -> float:
        """Entry Success % - percentage of opening duels won."""
        return self.opening_duels.win_rate

    @property
    def entry_kills_per_round(self) -> float:
        """Entry Kills per Round - average number of opening kills per round."""
        return (
            round(self.opening_duels.wins / self.rounds_played, 2)
            if self.rounds_played > 0
            else 0.0
        )

    @property
    def entry_ttd(self) -> float | None:
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

    # AWP/Sniper properties for role detection
    @property
    def awp_kills(self) -> int:
        """Total kills with AWP/SSG08 (sniper weapons)."""
        awp_count = self.weapon_kills.get("awp", 0) + self.weapon_kills.get("AWP", 0)
        ssg_count = self.weapon_kills.get("ssg08", 0) + self.weapon_kills.get("SSG08", 0)
        return awp_count + ssg_count

    @property
    def awp_kill_percentage(self) -> float:
        """Percentage of kills that were with sniper weapons (AWP/SSG08)."""
        if self.kills == 0:
            return 0.0
        return round(self.awp_kills / self.kills * 100, 1)

    @property
    def is_primary_awper(self) -> bool:
        """
        Determine if player is a primary AWPer based on weapon usage.

        Threshold: 35% of kills with sniper rifles indicates AWPer role.
        This overrides other role detection as AWPing is a distinct playstyle.
        """
        return self.awp_kill_percentage >= 35.0


# =============================================================================
# Role Scoring and Win Probability
# =============================================================================


@dataclass
class RoleScores:
    """
    Role scores for player identity detection.

    Each role is scored 0-100 based on behavioral metrics, not just K/D.
    The highest score determines the player's primary role.
    """

    entry: float = 0.0
    support: float = 0.0
    lurker: float = 0.0
    awper: float = 0.0
    rifler: float = 0.0

    @property
    def primary_role(self) -> str:
        """Get the highest-scoring role."""
        scores = {
            "entry": self.entry,
            "support": self.support,
            "lurker": self.lurker,
            "awper": self.awper,
            "rifler": self.rifler,
        }
        best_role = max(scores, key=lambda k: scores[k])
        best_score = scores[best_role]

        # If best score is very low, default to flex
        if best_score < 20:
            return "flex"

        # Check for tie (within 10% of each other)
        second_best = sorted(scores.values(), reverse=True)[1]
        if best_score > 0 and (best_score - second_best) / best_score < 0.10:
            return "flex"  # Too close to call

        return best_role

    @property
    def confidence(self) -> str:
        """Get confidence level based on score spread."""
        scores = [self.entry, self.support, self.lurker, self.awper, self.rifler]
        best_score = max(scores)
        if best_score >= 70:
            return "high"
        elif best_score >= 40:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "entry": round(self.entry, 1),
            "support": round(self.support, 1),
            "lurker": round(self.lurker, 1),
            "awper": round(self.awper, 1),
            "rifler": round(self.rifler, 1),
            "primary_role": self.primary_role,
            "confidence": self.confidence,
        }


@dataclass
class WinProbEvent:
    """A single win probability data point at a state change."""

    tick: int
    time_seconds: float  # Time since round start
    event_type: str  # "round_start", "kill", "bomb_plant", "bomb_defuse", "bomb_explode"
    ct_alive: int
    t_alive: int
    bomb_planted: bool
    ct_win_prob: float  # CT's win probability [0.0, 1.0]
    t_win_prob: float  # T's win probability [0.0, 1.0]
    description: str = ""  # e.g., "Player1 killed Player2"


@dataclass
class RoundMomentum:
    """
    Win probability tracking for a single round.

    Used to identify "throw" rounds (had advantage, lost) and
    "heroic" rounds (had disadvantage, won).
    """

    round_num: int
    winner: str  # "CT" or "T"
    win_prob_timeline: list[WinProbEvent] = field(default_factory=list)

    # Computed from timeline
    ct_peak_prob: float = 0.5  # Max CT win probability during round
    ct_min_prob: float = 0.5  # Min CT win probability during round
    t_peak_prob: float = 0.5  # Max T win probability during round
    t_min_prob: float = 0.5  # Min T win probability during round

    @property
    def ct_is_throw(self) -> bool:
        """CT had >=80% win prob but lost."""
        return self.ct_peak_prob >= 0.80 and self.winner == "T"

    @property
    def ct_is_heroic(self) -> bool:
        """CT had <=20% win prob but won."""
        return self.ct_min_prob <= 0.20 and self.winner == "CT"

    @property
    def t_is_throw(self) -> bool:
        """T had >=80% win prob but lost."""
        return self.t_peak_prob >= 0.80 and self.winner == "CT"

    @property
    def t_is_heroic(self) -> bool:
        """T had <=20% win prob but won."""
        return self.t_min_prob <= 0.20 and self.winner == "T"

    @property
    def round_tag(self) -> str:
        """Get the most significant tag for this round."""
        if self.ct_is_throw:
            return "CT_THROW"
        if self.t_is_throw:
            return "T_THROW"
        if self.ct_is_heroic:
            return "CT_HEROIC"
        if self.t_is_heroic:
            return "T_HEROIC"
        return ""

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "round_num": self.round_num,
            "winner": self.winner,
            "ct_peak_prob": round(self.ct_peak_prob, 2),
            "ct_min_prob": round(self.ct_min_prob, 2),
            "t_peak_prob": round(self.t_peak_prob, 2),
            "t_min_prob": round(self.t_min_prob, 2),
            "ct_is_throw": self.ct_is_throw,
            "ct_is_heroic": self.ct_is_heroic,
            "t_is_throw": self.t_is_throw,
            "t_is_heroic": self.t_is_heroic,
            "round_tag": self.round_tag,
            "timeline": [
                {
                    "tick": e.tick,
                    "time": round(e.time_seconds, 1),
                    "event": e.event_type,
                    "ct_alive": e.ct_alive,
                    "t_alive": e.t_alive,
                    "bomb_planted": e.bomb_planted,
                    "ct_prob": round(e.ct_win_prob, 2),
                    "t_prob": round(e.t_win_prob, 2),
                    "desc": e.description,
                }
                for e in self.win_prob_timeline
            ],
        }


def calculate_win_probability(
    perspective: str,
    ct_alive: int,
    t_alive: int,
    bomb_planted: bool,
    ct_has_defuser: bool = True,
    time_remaining: float = 60.0,
) -> float:
    """
    Calculate live win probability using CS2 standard heuristics.

    This implements a simple but effective heuristic based on:
    - Man advantage (+/-0.10 per player difference)
    - Bomb state (+/-0.20 for bomb plant)
    - Defuse kit advantage (+0.15 for CT when bomb not planted)

    Args:
        perspective: "CT" or "T" - whose win probability to calculate
        ct_alive: Number of CTs alive
        t_alive: Number of Ts alive
        bomb_planted: Whether the bomb is currently planted
        ct_has_defuser: Whether at least one CT has a defuse kit
        time_remaining: Seconds remaining in round (for future refinement)

    Returns:
        Win probability as float between 0.0 and 1.0

    Examples:
        >>> calculate_win_probability("CT", 5, 4, False)  # 5v4, no bomb
        0.60
        >>> calculate_win_probability("T", 3, 3, True)  # 3v3, bomb planted
        0.70
        >>> calculate_win_probability("CT", 1, 1, True)  # 1v1, bomb planted
        0.30
    """
    # Base probability
    base = 0.50

    # Man advantage: +0.10 per player difference
    if perspective == "CT":
        man_diff = ct_alive - t_alive
    else:
        man_diff = t_alive - ct_alive

    prob = base + (man_diff * 0.10)

    # Bomb state modifiers
    if bomb_planted:
        # Bomb planted heavily favors T
        if perspective == "T":
            prob += 0.20
        else:
            prob -= 0.20
    else:
        # No bomb planted - CT has map control advantage
        if perspective == "CT" and ct_has_defuser:
            prob += 0.15
        elif perspective == "T":
            prob -= 0.15

    # Handle elimination edge cases
    if ct_alive == 0:
        prob = 0.0 if perspective == "CT" else 1.0
    elif t_alive == 0:
        prob = 1.0 if perspective == "CT" else 0.0

    # Clamp to valid probability range
    return max(0.0, min(1.0, prob))


# =============================================================================
# Match Data Models
# =============================================================================


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
    # Win probability tracking
    momentum: RoundMomentum | None = None


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

    # Team names (extracted from demo or inferred from player clan tags)
    team1_name: str = "Team 1"
    team2_name: str = "Team 2"

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

    def get_mvp(self) -> PlayerMatchStats | None:
        """Get match MVP (highest rated player)."""
        leaderboard = self.get_leaderboard()
        return leaderboard[0] if leaderboard else None

    def get_kill_matrix_for_player(self, player_name: str) -> dict[str, int]:
        """Get kills this player got against each opponent."""
        return {e.victim_name: e.count for e in self.kill_matrix if e.attacker_name == player_name}

    def get_death_matrix_for_player(self, player_name: str) -> dict[str, int]:
        """Get deaths this player suffered from each opponent."""
        return {e.attacker_name: e.count for e in self.kill_matrix if e.victim_name == player_name}


def calculate_role_scores(
    player: PlayerMatchStats,
    first_contact_rate: float | None = None,
    avg_distance_from_teammates: float | None = None,
) -> RoleScores:
    """
    Calculate behavioral role scores for player identity detection.

    This implements the unified Role Scoring Engine that uses BEHAVIORAL metrics
    (positioning, utility timing, aggression patterns) rather than just K/D.

    Args:
        player: PlayerMatchStats object with all computed metrics
        first_contact_rate: % of rounds where player was first to engage (from PositioningMetrics)
        avg_distance_from_teammates: Average distance from teammates (from PositioningMetrics)

    Returns:
        RoleScores with scores for each role (0-100 scale)

    Role Detection Priority:
        1. AWPer check (35%+ kills with sniper) - overrides everything
        2. Entry score (high opening attempts + first contact)
        3. Support score (high utility + low first contact)
        4. Lurker score (high isolation + impact)
        5. Rifler (default high-frag player)
    """
    scores = RoleScores()

    # =========================================================================
    # STEP 1: AWPer Detection (Highest Priority)
    # If 35%+ of kills are with AWP/SSG08, this defines the player's role
    # =========================================================================
    if player.is_primary_awper:
        scores.awper = 85.0 + min(player.awp_kill_percentage - 35, 15)  # 85-100 based on %
        # AWPers can still have secondary role tendencies, but AWPer dominates
        # Continue scoring other roles but they'll be lower

    # =========================================================================
    # STEP 2: Entry Score
    # Based on: opening duel attempts (NOT just kills), first contact rate
    # =========================================================================
    entry_score = 0.0

    # Opening duel ATTEMPTS (shows aggression, not just success)
    attempts_per_round = (
        player.opening_duels.attempts / player.rounds_played if player.rounds_played > 0 else 0
    )
    # 0.3+ attempts/round = very aggressive entry
    entry_score += min(attempts_per_round / 0.3, 1.0) * 40  # Up to 40 points

    # Entry success rate (reward winning, not just attempting)
    if player.opening_duels.attempts >= 3:
        entry_score += min(player.opening_duels.win_rate / 50, 1.0) * 25  # Up to 25 points

    # First contact rate (behavioral - who takes fights first)
    if first_contact_rate is not None:
        # 50%+ first contact = dedicated entry
        entry_score += min(first_contact_rate / 50, 1.0) * 25  # Up to 25 points
    else:
        # Fallback: use opening kills as proxy for first contact
        if player.rounds_played > 0:
            opening_kill_rate = player.opening_duels.wins / player.rounds_played * 100
            entry_score += min(opening_kill_rate / 25, 1.0) * 15  # Up to 15 points

    # Bonus: Self-flash before entry (shows intentional entry, not random aggression)
    # We don't have this exact metric, but effective flashes can be a proxy
    if player.effective_flashes > 3:
        entry_score += 5  # Bonus for utility-supported entries

    scores.entry = min(entry_score, 100)

    # =========================================================================
    # STEP 3: Support Score
    # Based on: utility effectiveness, flash assists, LOW first contact
    # =========================================================================
    support_score = 0.0

    # Effective flashes (shows intentional team support)
    support_score += min(player.effective_flashes / 8, 1.0) * 25  # Up to 25 points

    # Flash assists (direct team contribution)
    support_score += min(player.utility.flash_assists / 4, 1.0) * 25  # Up to 25 points

    # Total utility usage quantity
    total_utility = (
        player.utility.flashbangs_thrown
        + player.utility.he_thrown
        + player.utility.molotovs_thrown
        + player.utility.smokes_thrown
    )
    if player.rounds_played > 0:
        utility_per_round = total_utility / player.rounds_played
        support_score += min(utility_per_round / 3, 1.0) * 20  # Up to 20 points

    # LOW first contact rate (supports play passive)
    if first_contact_rate is not None:
        # < 20% first contact = supportive positioning
        if first_contact_rate < 30:
            support_score += (30 - first_contact_rate) / 30 * 15  # Up to 15 points
    else:
        # Fallback: low opening attempts = passive player
        if attempts_per_round < 0.15:
            support_score += 10

    # Trade success (being in position to trade teammates)
    if player.trades.trade_kill_opportunities > 0:
        trade_attempt_rate = (
            player.trades.trade_kill_attempts / player.trades.trade_kill_opportunities
        )
        support_score += trade_attempt_rate * 15  # Up to 15 points

    scores.support = min(support_score, 100)

    # =========================================================================
    # STEP 4: Lurker Score
    # Based on: high untraded deaths (isolation), but WITH IMPACT
    # A player who dies alone without impact is NOT a lurker - they're feeding
    # =========================================================================
    lurker_score = 0.0

    # Isolation indicator: untraded deaths
    if player.deaths > 3:
        isolation_rate = player.untraded_deaths / player.deaths
        # > 50% untraded deaths = operating alone
        if isolation_rate > 0.5:
            lurker_score += (isolation_rate - 0.5) * 2 * 30  # Up to 30 points

    # Distance from teammates (if available)
    if avg_distance_from_teammates is not None:
        # > 1000 units = playing far from team
        if avg_distance_from_teammates > 800:
            lurker_score += min((avg_distance_from_teammates - 800) / 400, 1.0) * 25  # Up to 25

    # LOW first contact (lurkers let team take initial contact)
    if first_contact_rate is not None and first_contact_rate < 25:
        lurker_score += (25 - first_contact_rate) / 25 * 15  # Up to 15 points

    # CRITICAL: Must have IMPACT to be a lurker, not a feeder
    # Check multiple impact indicators
    has_impact = False
    kpr = player.kills_per_round
    if kpr >= 0.5:  # Decent kill rate
        has_impact = True
        lurker_score += 20
    if player.hltv_rating >= 0.9:  # Positive impact
        has_impact = True
        lurker_score += 15

    # Multi-kills (catching rotations often leads to multi-kills)
    if player.multi_kills.rounds_with_2k >= 3:
        lurker_score += 10

    # If no impact, heavily penalize lurker score
    if not has_impact and lurker_score > 30:
        lurker_score *= 0.3  # Reduce to 30% if no impact

    scores.lurker = min(lurker_score, 100)

    # =========================================================================
    # STEP 5: Rifler Score (Default High-Frag Role)
    # Based on: high kills, high HS%, consistent damage
    # =========================================================================
    rifler_score = 0.0

    # Kill count (raw fragging)
    if player.rounds_played > 0:
        kpr = player.kills_per_round
        rifler_score += min(kpr / 0.8, 1.0) * 30  # Up to 30 points for 0.8+ KPR

    # ADR (consistent damage)
    rifler_score += min(player.adr / 85, 1.0) * 25  # Up to 25 points for 85+ ADR

    # Headshot percentage (rifle discipline)
    rifler_score += min(player.headshot_percentage / 50, 1.0) * 20  # Up to 20 points

    # Multi-kills (impact fragging)
    multi_kill_rounds = (
        player.multi_kills.rounds_with_2k
        + player.multi_kills.rounds_with_3k * 2
        + player.multi_kills.rounds_with_4k * 3
        + player.multi_kills.rounds_with_5k * 4
    )
    rifler_score += min(multi_kill_rounds / 8, 1.0) * 15  # Up to 15 points

    # HLTV rating bonus
    if player.hltv_rating >= 1.15:
        rifler_score += 10

    scores.rifler = min(rifler_score, 100)

    return scores


# =============================================================================
# Validation
# =============================================================================


@dataclass
class ValidationResult:
    """Result of validating analysis data."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


# =============================================================================
# Tier 1 Player Metrics (awpy-based)
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
    mean_ttd_ms: float | None = None
    median_crosshair_error_deg: float | None = None

    # Optional extended metrics
    headshot_percentage: float = 0.0
    median_ttd_ms: float | None = None
    mean_crosshair_error_deg: float | None = None

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.kills > 0:
            self.headshot_percentage = round(self.headshot_kills / self.kills * 100, 1)
