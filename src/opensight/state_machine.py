"""
State Reconstruction Engine for Pro-Level CS2 Analytics

This module implements a State Machine architecture that "understands" the context
of each kill - not just "Player A killed Player B", but WHY (Trade, Entry, Lurk).

Architecture:
- Round-by-round processing (memory efficient)
- Time-window lookback buffers (for trade detection)
- Vector math for position-based analysis (lurk detection)
- Uses Polars for 10x faster coordinate math

Implements Leetify-grade metrics:
- Entry Kill: First kill of the round
- Trade Kill: Kill within 4 seconds of teammate death by same enemy
- Lurk Kill: Killer > 1500 units from team center of mass
- Flash Effectiveness: Blinds > 2.0 seconds (not just any flash)
- Utility ADR: Damage per round from HE/Molotov
- Crosshair Placement: Vertical adjustment error analysis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

# Use Polars for fast vector math (10x faster than Pandas on coordinate calculations)
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from opensight.constants import CS2_TICK_RATE
from opensight.parser import BlindEvent, DemoData, KillEvent

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS - The "Rules" of Pro-Level Analytics
# ============================================================================

# Trade Kill: Must kill the enemy who killed teammate within this window
TRADE_WINDOW_SECONDS = 4.0  # Leetify uses ~4 seconds (tighter than the 5s in basic analytics)
TRADE_WINDOW_TICKS = int(TRADE_WINDOW_SECONDS * CS2_TICK_RATE)

# Lurk Kill: Player must be > this distance from team's center of mass
LURK_DISTANCE_UNITS = 1500.0  # CS2 units (roughly 38 meters)

# Flash Effectiveness: Only count flashes that blind for > this duration
EFFECTIVE_FLASH_DURATION = 2.0  # seconds

# Crosshair Placement: Height adjustment for eye level
EYE_HEIGHT_UNITS = 64.0  # CS2 eye height offset


# ============================================================================
# DATA STRUCTURES - State Tracking
# ============================================================================


@dataclass
class KillContext:
    """Extended kill information with combat context."""

    # Original kill data
    kill: KillEvent

    # Combat Context (The "Why")
    is_entry_kill: bool = False  # First kill of the round
    is_trade_kill: bool = False  # Killed the enemy who killed a teammate
    is_lurk_kill: bool = False  # Killer far from team
    traded_teammate_id: int | None = None  # Who was traded (if trade kill)

    # Position Context (The "Where")
    distance_to_team: float | None = None  # Distance from team center
    team_center: tuple[float, float, float] | None = None

    # Engagement Context
    engagement_distance: float | None = None  # Distance to victim


@dataclass
class FlashContext:
    """Extended flash information with effectiveness analysis."""

    blind: BlindEvent
    is_effective: bool = False  # Duration > 2.0 seconds
    is_assist_worthy: bool = False  # Led to a kill within window


@dataclass
class RoundState:
    """State tracking for a single round - used for lookback buffers."""

    round_num: int
    start_tick: int
    end_tick: int

    # Lookback buffers (for trade detection)
    recent_deaths: list[tuple[int, int, int]] = field(default_factory=list)
    # Format: (death_tick, victim_steamid, killer_steamid)

    # Team positions at key moments (for lurk detection)
    ct_positions: list[tuple[int, float, float, float]] = field(default_factory=list)
    t_positions: list[tuple[int, float, float, float]] = field(default_factory=list)
    # Format: (steamid, x, y, z)

    # Round outcomes
    kills_contextualized: list[KillContext] = field(default_factory=list)
    entry_kill_id: int | None = None  # SteamID of entry killer


@dataclass
class UtilityRoundStats:
    """Utility statistics for a single round."""

    round_num: int

    # Flash stats (per player: steamid -> count)
    effective_flashes: dict[int, int] = field(default_factory=dict)
    ineffective_flashes: dict[int, int] = field(default_factory=dict)
    total_blind_time: dict[int, float] = field(default_factory=dict)

    # Utility damage (per player: steamid -> damage)
    he_damage: dict[int, int] = field(default_factory=dict)
    molotov_damage: dict[int, int] = field(default_factory=dict)


@dataclass
class PlayerContextStats:
    """Aggregated contextual statistics for a player."""

    steam_id: int
    name: str

    # Entry stats
    entry_kills: int = 0
    entry_deaths: int = 0
    entry_attempts: int = 0  # Rounds where involved in first duel

    # Trade stats (enhanced)
    trade_kills: int = 0  # Kills that avenged a teammate
    deaths_traded: int = 0  # Deaths that were avenged
    trade_opportunities: int = 0  # Times a trade was possible
    failed_trades: int = 0  # Opportunities not taken

    # Lurk stats
    lurk_kills: int = 0
    lurk_deaths: int = 0

    # Flash effectiveness
    effective_flashes: int = 0
    ineffective_flashes: int = 0
    total_blind_time: float = 0.0

    # Utility damage
    he_damage: int = 0
    molotov_damage: int = 0

    # Derived properties
    @property
    def entry_success_rate(self) -> float:
        """Percentage of entry duels won."""
        return (self.entry_kills / self.entry_attempts * 100) if self.entry_attempts > 0 else 0.0

    @property
    def trade_rate(self) -> float:
        """Percentage of trade opportunities taken."""
        opportunities = self.trade_opportunities
        return (self.trade_kills / opportunities * 100) if opportunities > 0 else 0.0

    @property
    def deaths_traded_rate(self) -> float:
        """Percentage of deaths that were traded."""
        return (
            (self.deaths_traded / (self.entry_deaths + self.deaths_traded + self.lurk_deaths) * 100)
            if (self.entry_deaths + self.deaths_traded + self.lurk_deaths) > 0
            else 0.0
        )

    @property
    def flash_effectiveness(self) -> float:
        """Percentage of flashes that were effective (>2s blind)."""
        total = self.effective_flashes + self.ineffective_flashes
        return (self.effective_flashes / total * 100) if total > 0 else 0.0

    @property
    def utility_adr(self) -> float:
        """Utility damage per round (requires rounds_played to be set externally)."""
        return 0.0  # Calculated externally with rounds_played


@dataclass
class StateAnalysisResult:
    """Complete state analysis results."""

    players: dict[int, PlayerContextStats]
    kills_contextualized: list[KillContext]
    rounds_analyzed: int

    # Summary stats
    total_entry_kills: int = 0
    total_trade_kills: int = 0
    total_lurk_kills: int = 0


# ============================================================================
# STATE MACHINE - The Core Engine
# ============================================================================


class StateMachine:
    """
    State Reconstruction Engine for pro-level CS2 analytics.

    Processes the match round-by-round, maintaining state buffers
    for context-aware kill classification.

    Usage:
        machine = StateMachine(demo_data)
        result = machine.analyze()
    """

    def __init__(self, demo_data: DemoData):
        self.data = demo_data
        self._players: dict[int, PlayerContextStats] = {}
        self._round_states: list[RoundState] = []
        self._kills_contextualized: list[KillContext] = []

        # Build Polars DataFrames for fast vector math
        if POLARS_AVAILABLE:
            self._kills_pl = self._build_polars_kills()
            self._damages_pl = self._build_polars_damages()
            self._blinds_pl = self._build_polars_blinds()
        else:
            logger.warning("Polars not available, falling back to NumPy-based calculations")
            self._kills_pl = None
            self._damages_pl = None
            self._blinds_pl = None

    def _build_polars_kills(self) -> pl.DataFrame | None:
        """Convert kills to Polars DataFrame for fast operations."""
        if not POLARS_AVAILABLE or not self.data.kills:
            return None

        records = []
        for k in self.data.kills:
            records.append(
                {
                    "tick": k.tick,
                    "round_num": k.round_num,
                    "attacker_steamid": k.attacker_steamid,
                    "attacker_side": k.attacker_side,
                    "victim_steamid": k.victim_steamid,
                    "victim_side": k.victim_side,
                    "attacker_x": k.attacker_x or 0.0,
                    "attacker_y": k.attacker_y or 0.0,
                    "attacker_z": k.attacker_z or 0.0,
                    "attacker_pitch": k.attacker_pitch or 0.0,
                    "attacker_yaw": k.attacker_yaw or 0.0,
                    "victim_x": k.victim_x or 0.0,
                    "victim_y": k.victim_y or 0.0,
                    "victim_z": k.victim_z or 0.0,
                    "headshot": k.headshot,
                    "weapon": k.weapon,
                }
            )

        return pl.DataFrame(records) if records else None

    def _build_polars_damages(self) -> pl.DataFrame | None:
        """Convert damages to Polars DataFrame."""
        if not POLARS_AVAILABLE or not self.data.damages:
            return None

        records = []
        for d in self.data.damages:
            records.append(
                {
                    "tick": d.tick,
                    "round_num": d.round_num,
                    "attacker_steamid": d.attacker_steamid,
                    "attacker_side": d.attacker_side,
                    "victim_steamid": d.victim_steamid,
                    "victim_side": d.victim_side,
                    "damage": d.damage,
                    "weapon": d.weapon,
                }
            )

        return pl.DataFrame(records) if records else None

    def _build_polars_blinds(self) -> pl.DataFrame | None:
        """Convert blinds to Polars DataFrame."""
        if not POLARS_AVAILABLE or not self.data.blinds:
            return None

        records = []
        for b in self.data.blinds:
            records.append(
                {
                    "tick": b.tick,
                    "round_num": b.round_num,
                    "attacker_steamid": b.attacker_steamid,
                    "attacker_side": b.attacker_side,
                    "victim_steamid": b.victim_steamid,
                    "victim_side": b.victim_side,
                    "blind_duration": b.blind_duration,
                    "is_teammate": b.is_teammate,
                }
            )

        return pl.DataFrame(records) if records else None

    def analyze(self) -> StateAnalysisResult:
        """
        Run complete state analysis.

        Process flow:
        1. Initialize player stats
        2. For each round:
           a. Reset round state
           b. Process kills in tick order
           c. Classify each kill (entry, trade, lurk)
           d. Track utility effectiveness
           e. Clear round data (memory management)
        3. Aggregate results
        """
        logger.info("Starting State Machine analysis...")

        # Initialize players
        self._init_players()

        # Get round boundaries
        round_boundaries = self._get_round_boundaries()

        # Process each round
        for round_num, (start_tick, end_tick) in round_boundaries.items():
            self._analyze_round(round_num, start_tick, end_tick)

        # Aggregate utility stats
        self._aggregate_utility_stats()

        # Build result
        result = StateAnalysisResult(
            players=self._players,
            kills_contextualized=self._kills_contextualized,
            rounds_analyzed=len(round_boundaries),
            total_entry_kills=sum(p.entry_kills for p in self._players.values()),
            total_trade_kills=sum(p.trade_kills for p in self._players.values()),
            total_lurk_kills=sum(p.lurk_kills for p in self._players.values()),
        )

        logger.info(
            f"State Machine analysis complete: "
            f"{result.total_entry_kills} entries, "
            f"{result.total_trade_kills} trades, "
            f"{result.total_lurk_kills} lurks"
        )

        return result

    def _init_players(self) -> None:
        """Initialize PlayerContextStats for each player."""
        for steam_id, name in self.data.player_names.items():
            self._players[steam_id] = PlayerContextStats(
                steam_id=steam_id,
                name=name,
            )

    def _get_round_boundaries(self) -> dict[int, tuple[int, int]]:
        """Get start/end ticks for each round."""
        boundaries = {}

        if self.data.rounds:
            for r in self.data.rounds:
                boundaries[r.round_num] = (r.start_tick, r.end_tick)
        else:
            # Fallback: infer from kills
            if POLARS_AVAILABLE and self._kills_pl is not None:
                rounds = self._kills_pl.select("round_num").unique().sort("round_num")
                for row in rounds.iter_rows():
                    round_num = row[0]
                    round_kills = self._kills_pl.filter(pl.col("round_num") == round_num)
                    boundaries[round_num] = (
                        int(round_kills.select("tick").min().item()),
                        int(round_kills.select("tick").max().item()),
                    )
            else:
                # NumPy fallback
                round_nums = {k.round_num for k in self.data.kills}
                for round_num in sorted(round_nums):
                    round_kills = [k for k in self.data.kills if k.round_num == round_num]
                    if round_kills:
                        boundaries[round_num] = (
                            min(k.tick for k in round_kills),
                            max(k.tick for k in round_kills),
                        )

        return boundaries

    def _analyze_round(self, round_num: int, start_tick: int, end_tick: int) -> None:
        """
        Analyze a single round using State Machine logic.

        This is the core of the State Machine - we process kills in order,
        maintaining a "recent_deaths" buffer for trade detection.
        """
        # Initialize round state
        round_state = RoundState(
            round_num=round_num,
            start_tick=start_tick,
            end_tick=end_tick,
        )

        # Get round kills sorted by tick
        if POLARS_AVAILABLE and self._kills_pl is not None:
            round_kills_pl = self._kills_pl.filter(pl.col("round_num") == round_num).sort("tick")
            round_kills = self._polars_to_kill_events(round_kills_pl)
        else:
            round_kills = sorted(
                [k for k in self.data.kills if k.round_num == round_num], key=lambda k: k.tick
            )

        if not round_kills:
            return

        # Build team position snapshots at round start (for lurk detection)
        self._build_team_positions(round_state, round_kills)

        # Process each kill
        for idx, kill in enumerate(round_kills):
            context = self._classify_kill(kill, round_state, idx == 0)
            round_state.kills_contextualized.append(context)
            self._kills_contextualized.append(context)

            # Update recent deaths buffer
            round_state.recent_deaths.append(
                (kill.tick, kill.victim_steamid, kill.attacker_steamid)
            )

            # Update player stats
            self._update_player_stats(context)

        # Store round state
        self._round_states.append(round_state)

        # Memory cleanup (important for large demos)
        round_state.recent_deaths.clear()
        round_state.ct_positions.clear()
        round_state.t_positions.clear()

    def _polars_to_kill_events(self, df: pl.DataFrame) -> list[KillEvent]:
        """Convert Polars DataFrame back to KillEvent list."""
        kills = []
        for row in df.iter_rows(named=True):
            kill = KillEvent(
                tick=row["tick"],
                round_num=row["round_num"],
                attacker_steamid=row["attacker_steamid"],
                attacker_name=self.data.player_names.get(row["attacker_steamid"], ""),
                attacker_side=row["attacker_side"],
                victim_steamid=row["victim_steamid"],
                victim_name=self.data.player_names.get(row["victim_steamid"], ""),
                victim_side=row["victim_side"],
                weapon=row["weapon"],
                headshot=row["headshot"],
                attacker_x=row["attacker_x"] if row["attacker_x"] != 0.0 else None,
                attacker_y=row["attacker_y"] if row["attacker_y"] != 0.0 else None,
                attacker_z=row["attacker_z"] if row["attacker_z"] != 0.0 else None,
                attacker_pitch=row["attacker_pitch"] if row["attacker_pitch"] != 0.0 else None,
                attacker_yaw=row["attacker_yaw"] if row["attacker_yaw"] != 0.0 else None,
                victim_x=row["victim_x"] if row["victim_x"] != 0.0 else None,
                victim_y=row["victim_y"] if row["victim_y"] != 0.0 else None,
                victim_z=row["victim_z"] if row["victim_z"] != 0.0 else None,
            )
            kills.append(kill)
        return kills

    def _build_team_positions(self, round_state: RoundState, round_kills: list[KillEvent]) -> None:
        """Build team position snapshots from first kill of round (approximation)."""
        # We use the first kill's positions as a proxy for team positions
        # In a full implementation, we'd use tick data
        first_kill = round_kills[0] if round_kills else None
        if not first_kill:
            return

        # Track which players are on which team
        for steam_id, team in self.data.player_teams.items():
            # Try to find this player's position from kills in this round
            player_pos = None
            for kill in round_kills[:5]:  # Check first few kills
                if kill.attacker_steamid == steam_id and kill.attacker_x is not None:
                    player_pos = (
                        steam_id,
                        kill.attacker_x,
                        kill.attacker_y,
                        kill.attacker_z or 0.0,
                    )
                    break
                if kill.victim_steamid == steam_id and kill.victim_x is not None:
                    player_pos = (steam_id, kill.victim_x, kill.victim_y, kill.victim_z or 0.0)
                    break

            if player_pos:
                if team == "CT":
                    round_state.ct_positions.append(player_pos)
                elif team == "T":
                    round_state.t_positions.append(player_pos)

    def _classify_kill(
        self, kill: KillEvent, round_state: RoundState, is_first: bool
    ) -> KillContext:
        """
        Classify a kill with full context.

        Logic gates:
        1. Entry Kill: First kill of the round
        2. Trade Kill: Killed the enemy who killed teammate < 4 seconds ago
        3. Lurk Kill: Killer > 1500 units from team center of mass
        """
        context = KillContext(kill=kill)

        # 1. ENTRY KILL - First kill of the round
        if is_first:
            context.is_entry_kill = True
            round_state.entry_kill_id = kill.attacker_steamid

        # 2. TRADE KILL - Killed the enemy who killed a teammate within window
        traded_teammate = self._check_trade(kill, round_state)
        if traded_teammate is not None:
            context.is_trade_kill = True
            context.traded_teammate_id = traded_teammate

        # 3. LURK KILL - Far from team center of mass
        lurk_info = self._check_lurk(kill, round_state)
        if lurk_info:
            context.is_lurk_kill = True
            context.distance_to_team = lurk_info[0]
            context.team_center = lurk_info[1]

        # Calculate engagement distance
        if kill.attacker_x is not None and kill.victim_x is not None:
            context.engagement_distance = self._calculate_distance(
                (kill.attacker_x, kill.attacker_y, kill.attacker_z or 0.0),
                (kill.victim_x, kill.victim_y, kill.victim_z or 0.0),
            )

        return context

    def _check_trade(self, kill: KillEvent, round_state: RoundState) -> int | None:
        """
        Check if this kill is a trade.

        Trade Logic:
        - A teammate died < 4 seconds ago
        - The killer of this kill is the one who killed that teammate

        Returns the traded teammate's steamid, or None if not a trade.
        """
        kill_tick = kill.tick
        victim_id = kill.victim_steamid  # The enemy we just killed
        killer_team = kill.attacker_side

        # Look through recent deaths
        for death_tick, dead_teammate_id, teammate_killer_id in round_state.recent_deaths:
            # Check if:
            # 1. Death was within trade window
            # 2. The person we just killed is the one who killed our teammate
            # 3. The dead person was on our team
            time_diff = kill_tick - death_tick

            if time_diff <= TRADE_WINDOW_TICKS and time_diff > 0:
                # Check if victim we just killed is the one who killed our teammate
                if victim_id == teammate_killer_id:
                    # Verify dead person was on our team
                    dead_team = self.data.player_teams.get(dead_teammate_id, "Unknown")
                    if dead_team == killer_team:
                        return dead_teammate_id

        return None

    def _check_lurk(
        self, kill: KillEvent, round_state: RoundState
    ) -> tuple[float, tuple[float, float, float]] | None:
        """
        Check if this kill was a lurk (far from team).

        Lurk Logic:
        - Calculate center of mass of the 4 teammates
        - If killer is > 1500 units away, it's a lurk

        Returns (distance, team_center) or None if not a lurk.
        """
        if kill.attacker_x is None:
            return None

        killer_pos = (kill.attacker_x, kill.attacker_y, kill.attacker_z or 0.0)
        killer_team = kill.attacker_side
        killer_id = kill.attacker_steamid

        # Get team positions (excluding killer)
        if killer_team == "CT":
            team_positions = [p for p in round_state.ct_positions if p[0] != killer_id]
        elif killer_team == "T":
            team_positions = [p for p in round_state.t_positions if p[0] != killer_id]
        else:
            return None

        if len(team_positions) < 2:  # Need at least 2 teammates for meaningful center
            return None

        # Calculate center of mass
        center = self._calculate_center_of_mass([(p[1], p[2], p[3]) for p in team_positions])

        # Calculate distance to center
        distance = self._calculate_distance(killer_pos, center)

        if distance > LURK_DISTANCE_UNITS:
            return (distance, center)

        return None

    def _calculate_distance(
        self, pos1: tuple[float, float, float], pos2: tuple[float, float, float]
    ) -> float:
        """Calculate 3D Euclidean distance between two positions."""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2
        )

    def _calculate_center_of_mass(
        self, positions: list[tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        """Calculate center of mass for a list of positions."""
        if not positions:
            return (0.0, 0.0, 0.0)

        if POLARS_AVAILABLE:
            df = pl.DataFrame(
                {
                    "x": [p[0] for p in positions],
                    "y": [p[1] for p in positions],
                    "z": [p[2] for p in positions],
                }
            )
            return (
                df.select("x").mean().item(),
                df.select("y").mean().item(),
                df.select("z").mean().item(),
            )
        else:
            # NumPy fallback
            arr = np.array(positions)
            return tuple(np.mean(arr, axis=0))

    def _update_player_stats(self, context: KillContext) -> None:
        """Update player statistics based on kill context."""
        kill = context.kill
        attacker_id = kill.attacker_steamid
        victim_id = kill.victim_steamid

        # Entry kill stats
        if context.is_entry_kill:
            if attacker_id in self._players:
                self._players[attacker_id].entry_kills += 1
                self._players[attacker_id].entry_attempts += 1
            if victim_id in self._players:
                self._players[victim_id].entry_deaths += 1
                self._players[victim_id].entry_attempts += 1

        # Trade kill stats
        if context.is_trade_kill:
            if attacker_id in self._players:
                self._players[attacker_id].trade_kills += 1
            # Mark the traded teammate
            if context.traded_teammate_id and context.traded_teammate_id in self._players:
                self._players[context.traded_teammate_id].deaths_traded += 1

        # Lurk kill stats
        if context.is_lurk_kill:
            if attacker_id in self._players:
                self._players[attacker_id].lurk_kills += 1

    def _aggregate_utility_stats(self) -> None:
        """Aggregate utility statistics from blinds and damages."""
        # Process flash effectiveness
        if POLARS_AVAILABLE and self._blinds_pl is not None:
            # Use Polars for fast aggregation
            effective = (
                self._blinds_pl.filter(
                    (pl.col("blind_duration") >= EFFECTIVE_FLASH_DURATION)
                    & ~pl.col("is_teammate")
                )
                .group_by("attacker_steamid")
                .agg(
                    pl.count().alias("effective_count"),
                    pl.sum("blind_duration").alias("total_blind_time"),
                )
            )

            ineffective = (
                self._blinds_pl.filter(
                    (pl.col("blind_duration") < EFFECTIVE_FLASH_DURATION)
                    & ~pl.col("is_teammate")
                )
                .group_by("attacker_steamid")
                .agg(pl.count().alias("ineffective_count"))
            )

            for row in effective.iter_rows(named=True):
                steam_id = row["attacker_steamid"]
                if steam_id in self._players:
                    self._players[steam_id].effective_flashes = row["effective_count"]
                    self._players[steam_id].total_blind_time = row["total_blind_time"]

            for row in ineffective.iter_rows(named=True):
                steam_id = row["attacker_steamid"]
                if steam_id in self._players:
                    self._players[steam_id].ineffective_flashes = row["ineffective_count"]
        else:
            # Fallback to iteration
            for blind in self.data.blinds:
                if blind.is_teammate:
                    continue
                steam_id = blind.attacker_steamid
                if steam_id not in self._players:
                    continue

                if blind.blind_duration >= EFFECTIVE_FLASH_DURATION:
                    self._players[steam_id].effective_flashes += 1
                else:
                    self._players[steam_id].ineffective_flashes += 1
                self._players[steam_id].total_blind_time += blind.blind_duration

        # Process utility damage (HE/Molotov)
        he_weapons = {"hegrenade", "he_grenade", "grenade_he", "hegrenade_projectile"}
        molly_weapons = {"molotov", "incgrenade", "inferno", "molotov_projectile", "incendiary"}

        if POLARS_AVAILABLE and self._damages_pl is not None:
            # HE damage
            he_dmg = (
                self._damages_pl.filter(
                    pl.col("weapon").str.to_lowercase().is_in(he_weapons)
                    & (pl.col("attacker_side") != pl.col("victim_side"))  # Exclude team damage
                )
                .group_by("attacker_steamid")
                .agg(pl.sum("damage").alias("he_damage"))
            )

            for row in he_dmg.iter_rows(named=True):
                steam_id = row["attacker_steamid"]
                if steam_id in self._players:
                    self._players[steam_id].he_damage = row["he_damage"]

            # Molotov damage
            molly_dmg = (
                self._damages_pl.filter(
                    pl.col("weapon").str.to_lowercase().is_in(molly_weapons)
                    & (pl.col("attacker_side") != pl.col("victim_side"))
                )
                .group_by("attacker_steamid")
                .agg(pl.sum("damage").alias("molotov_damage"))
            )

            for row in molly_dmg.iter_rows(named=True):
                steam_id = row["attacker_steamid"]
                if steam_id in self._players:
                    self._players[steam_id].molotov_damage = row["molotov_damage"]
        else:
            # Fallback
            for dmg in self.data.damages:
                if dmg.attacker_side == dmg.victim_side:  # Skip team damage
                    continue
                steam_id = dmg.attacker_steamid
                if steam_id not in self._players:
                    continue

                weapon_lower = dmg.weapon.lower()
                if weapon_lower in he_weapons:
                    self._players[steam_id].he_damage += dmg.damage
                elif weapon_lower in molly_weapons:
                    self._players[steam_id].molotov_damage += dmg.damage


# ============================================================================
# CROSSHAIR PLACEMENT ANALYSIS - Advanced Position Analysis
# ============================================================================


class CrosshairAnalyzer:
    """
    Advanced crosshair placement analysis.

    Calculates the "Vertical Adjustment Error" - how much the player
    had to adjust their aim vertically to hit the target.

    Low value = Perfect crosshair placement (head level)
    High value = Poor crosshair placement (had to flick)
    """

    def __init__(self, demo_data: DemoData):
        self.data = demo_data

    def analyze_vertical_adjustment(self, kill: KillEvent) -> float | None:
        """
        Calculate vertical adjustment error for a kill.

        Logic:
        1. Get attacker's view angle (pitch)
        2. Calculate ideal pitch to victim's head
        3. Return the absolute difference

        Returns degrees of vertical adjustment needed, or None if no position data.
        """
        if kill.attacker_x is None or kill.victim_x is None:
            return None
        if kill.attacker_pitch is None:
            return None

        # Attacker eye position
        att_pos = np.array(
            [kill.attacker_x, kill.attacker_y, (kill.attacker_z or 0.0) + EYE_HEIGHT_UNITS]
        )

        # Victim head position (approximate)
        vic_head_pos = np.array(
            [kill.victim_x, kill.victim_y, (kill.victim_z or 0.0) + EYE_HEIGHT_UNITS]
        )

        # Vector to victim
        to_victim = vic_head_pos - att_pos
        distance_2d = math.sqrt(to_victim[0] ** 2 + to_victim[1] ** 2)

        if distance_2d < 1.0:
            return 0.0  # Too close to calculate

        # Ideal pitch (negative because CS2 uses negative pitch for looking up)
        height_diff = to_victim[2]
        ideal_pitch = -math.degrees(math.atan2(height_diff, distance_2d))

        # Actual pitch from attacker
        actual_pitch = kill.attacker_pitch

        # Vertical adjustment error (absolute difference)
        vertical_error = abs(actual_pitch - ideal_pitch)

        return vertical_error

    def analyze_all_kills(self) -> dict[int, list[float]]:
        """
        Analyze vertical adjustment for all kills.

        Returns dict mapping steamid -> list of vertical errors.
        """
        results: dict[int, list[float]] = {}

        for kill in self.data.kills:
            error = self.analyze_vertical_adjustment(kill)
            if error is not None:
                steam_id = kill.attacker_steamid
                if steam_id not in results:
                    results[steam_id] = []
                results[steam_id].append(error)

        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def analyze_state(demo_data: DemoData) -> StateAnalysisResult:
    """
    Convenience function to run State Machine analysis.

    Usage:
        from opensight.state_machine import analyze_state
        result = analyze_state(demo_data)
    """
    machine = StateMachine(demo_data)
    return machine.analyze()


def get_kill_contexts(demo_data: DemoData) -> list[KillContext]:
    """
    Get contextualized kills (entry/trade/lurk classification).

    Usage:
        contexts = get_kill_contexts(demo_data)
        entries = [c for c in contexts if c.is_entry_kill]
        trades = [c for c in contexts if c.is_trade_kill]
    """
    result = analyze_state(demo_data)
    return result.kills_contextualized
