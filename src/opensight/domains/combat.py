"""
Combat Analysis Module for CS2 Demo Analysis

Implements advanced combat metrics:
- Trade kill detection (5-second window)
- Opening duel (first blood) statistics
- Clutch detection and success rates
- Multi-kill tracking (2k, 3k, 4k, ace)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from opensight.core.parser import DemoData, safe_int

logger = logging.getLogger(__name__)


# Trade window in milliseconds (standard is 5 seconds)
TRADE_WINDOW_MS = 5000
# Trade window in ticks (5 seconds at 64 tick rate)
TRADE_WINDOW_TICKS = 320
# Clutch scenarios
CLUTCH_SCENARIOS = ["1v1", "1v2", "1v3", "1v4", "1v5"]


class ClutchResult(Enum):
    """Result of a clutch situation."""

    WON = "won"
    LOST = "lost"
    IN_PROGRESS = "in_progress"


@dataclass
class TradeKill:
    """A trade kill event."""

    original_kill_tick: int
    trade_kill_tick: int
    time_delta_ms: float

    # Original kill info
    original_attacker_id: int
    original_attacker_name: str
    original_victim_id: int
    original_victim_name: str
    original_weapon: str

    # Trade kill info
    trader_id: int
    trader_name: str
    traded_player_id: int  # The player who got the original kill
    traded_player_name: str
    trade_weapon: str

    round_num: int


@dataclass
class OpeningDuel:
    """First engagement of a round."""

    round_num: int
    tick: int

    # Winner info
    winner_id: int
    winner_name: str
    winner_team: int
    weapon: str
    headshot: bool

    # Loser info
    loser_id: int
    loser_name: str
    loser_team: int

    # Did the opening kill team win the round?
    round_won: bool | None = None


@dataclass
class ClutchSituation:
    """A clutch scenario."""

    round_num: int
    start_tick: int
    end_tick: int

    clutcher_id: int
    clutcher_name: str
    clutcher_team: int

    scenario: str  # "1v1", "1v2", etc.
    enemies_alive: int

    result: ClutchResult
    kills_in_clutch: int
    damage_in_clutch: int

    # If bomb was involved
    bomb_planted: bool = False


@dataclass
class MultiKill:
    """Multiple kills in quick succession or same round."""

    round_num: int
    player_id: int
    player_name: str
    kill_count: int  # 2, 3, 4, 5 (ace)
    kills: list[dict] = field(default_factory=list)
    all_headshots: bool = False


@dataclass
class PlayerCombatStats:
    """Combat statistics for a single player."""

    steam_id: int
    name: str

    # Trade stats
    trades_given: int  # Times this player was traded by teammate
    trades_taken: int  # Times this player traded a teammate
    traded_deaths: int  # Deaths that were traded
    untraded_deaths: int  # Deaths that went untraded
    trade_kill_time_avg_ms: float  # Average time to trade

    # Opening duel stats
    opening_kills: int
    opening_deaths: int
    opening_kill_rate: float  # opening_kills / (opening_kills + opening_deaths)
    opening_kills_won_rounds: int  # Rounds won after getting opening kill

    # Clutch stats
    clutch_attempts: int
    clutch_wins: int
    clutch_win_rate: float
    clutches_by_scenario: dict[str, tuple[int, int]]  # scenario -> (attempts, wins)

    # Multi-kill stats
    double_kills: int
    triple_kills: int
    quad_kills: int
    aces: int


@dataclass
class CombatAnalysisResult:
    """Complete combat analysis for a match."""

    # All events
    trade_kills: list[TradeKill]
    opening_duels: list[OpeningDuel]
    clutch_situations: list[ClutchSituation]
    multi_kills: list[MultiKill]

    # Per-player stats
    player_stats: dict[int, PlayerCombatStats]

    # Team-level stats
    team_trade_rate: dict[int, float]  # team -> % of deaths traded
    team_opening_win_rate: dict[int, float]  # team -> opening duel win rate


class CombatAnalyzer:
    """Analyzer for combat metrics from parsed demo data."""

    def __init__(self, demo_data: DemoData):
        """
        Initialize the combat analyzer.

        Args:
            demo_data: Parsed demo data to analyze.
        """
        self.data = demo_data
        self.tick_rate = demo_data.tick_rate or 64

        self._trade_kills: list[TradeKill] = []
        self._opening_duels: list[OpeningDuel] = []
        self._clutch_situations: list[ClutchSituation] = []
        self._multi_kills: list[MultiKill] = []

    def analyze(self) -> CombatAnalysisResult:
        """
        Run full combat analysis on the demo data.

        Returns:
            CombatAnalysisResult containing all combat metrics.
        """
        logger.info("Starting combat analysis...")

        kills_df = self.data.kills_df
        if kills_df.empty:
            logger.warning("No kill data for combat analysis")
            return self._empty_result()

        # Find column names
        att_col = self._find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        vic_col = self._find_col(kills_df, ["user_steamid", "victim_steamid"])
        round_col = self._find_col(kills_df, ["total_rounds_played"])
        tick_col = self._find_col(kills_df, ["tick"])

        if not att_col or not vic_col:
            logger.warning("Missing required columns for combat analysis")
            return self._empty_result()

        # Run analyses
        self._analyze_trades(kills_df, att_col, vic_col, tick_col, round_col)
        self._analyze_opening_duels(kills_df, att_col, vic_col, tick_col, round_col)
        self._analyze_multi_kills(kills_df, att_col, vic_col, round_col)
        self._analyze_clutches(kills_df, att_col, vic_col, tick_col, round_col)

        # Build player stats
        player_stats = self._build_player_stats()

        # Build team stats
        team_trade_rate = self._calculate_team_trade_rates()
        team_opening_rate = self._calculate_team_opening_rates()

        logger.info(
            f"Combat analysis complete. {len(self._trade_kills)} trades, "
            f"{len(self._opening_duels)} opening duels, "
            f"{len(self._clutch_situations)} clutches"
        )

        return CombatAnalysisResult(
            trade_kills=self._trade_kills,
            opening_duels=self._opening_duels,
            clutch_situations=self._clutch_situations,
            multi_kills=self._multi_kills,
            player_stats=player_stats,
            team_trade_rate=team_trade_rate,
            team_opening_win_rate=team_opening_rate,
        )

    def _find_col(self, df: pd.DataFrame, options: list[str]) -> str | None:
        """Find first matching column name."""
        for col in options:
            if col in df.columns:
                return col
        return None

    def _empty_result(self) -> CombatAnalysisResult:
        """Return empty result when analysis cannot be performed."""
        return CombatAnalysisResult(
            trade_kills=[],
            opening_duels=[],
            clutch_situations=[],
            multi_kills=[],
            player_stats={},
            team_trade_rate={2: 0.0, 3: 0.0},
            team_opening_win_rate={2: 0.0, 3: 0.0},
        )

    def _ticks_to_ms(self, ticks: int) -> float:
        """Convert ticks to milliseconds."""
        return (ticks / self.tick_rate) * 1000

    def _same_team(self, player1_id: int, player2_id: int) -> bool:
        """Check if two players are on the same team."""
        team1 = self.data.player_teams.get(player1_id, 0)
        team2 = self.data.player_teams.get(player2_id, 0)
        return team1 == team2 and team1 != 0

    def _get_side(self, player_id: int, round_num: int = 0) -> str:
        """Get the side (CT/T) for a player."""
        team = self.data.player_teams.get(player_id, 0)
        if team == 2:
            return "T"
        elif team == 3:
            return "CT"
        return "Unknown"

    def _analyze_trades(
        self,
        kills_df: pd.DataFrame,
        att_col: str,
        vic_col: str,
        tick_col: str | None,
        round_col: str | None,
    ) -> None:
        """Detect trade kills within the trade window."""
        if tick_col is None:
            logger.warning("No tick column for trade analysis")
            return

        # Sort by tick
        sorted_kills = kills_df.sort_values(tick_col).reset_index(drop=True)
        trade_window_ticks = int((TRADE_WINDOW_MS / 1000) * self.tick_rate)

        # Find attacker/victim name columns
        self._find_col(kills_df, ["attacker_name"])
        self._find_col(kills_df, ["user_name", "victim_name"])
        weapon_col = self._find_col(kills_df, ["weapon"])

        for _i, kill in sorted_kills.iterrows():
            kill_tick = safe_int(kill[tick_col])
            victim_id = safe_int(kill[vic_col])
            attacker_id = safe_int(kill[att_col])

            if victim_id == 0 or attacker_id == 0:
                continue

            # Get team of victim
            victim_team = self.data.player_teams.get(victim_id, 0)

            # Look for a follow-up kill where:
            # - The victim's teammate kills the original attacker
            # - Within the trade window
            window_start = kill_tick
            window_end = kill_tick + trade_window_ticks

            # Get kills in window
            window_kills = sorted_kills[
                (sorted_kills[tick_col] > window_start) & (sorted_kills[tick_col] <= window_end)
            ]

            for _, trade_candidate in window_kills.iterrows():
                trade_attacker = safe_int(trade_candidate[att_col])
                trade_victim = safe_int(trade_candidate[vic_col])

                # Check if this is a trade: teammate of victim killed the attacker
                trade_attacker_team = self.data.player_teams.get(trade_attacker, 0)

                if trade_victim == attacker_id and trade_attacker_team == victim_team:
                    # This is a trade!
                    trade_tick = safe_int(trade_candidate[tick_col])
                    time_delta = self._ticks_to_ms(trade_tick - kill_tick)

                    round_num = safe_int(kill.get(round_col, 0)) if round_col else 0

                    trade = TradeKill(
                        original_kill_tick=kill_tick,
                        trade_kill_tick=trade_tick,
                        time_delta_ms=time_delta,
                        original_attacker_id=attacker_id,
                        original_attacker_name=self.data.player_names.get(attacker_id, "Unknown"),
                        original_victim_id=victim_id,
                        original_victim_name=self.data.player_names.get(victim_id, "Unknown"),
                        original_weapon=str(kill.get(weapon_col, "")) if weapon_col else "",
                        trader_id=trade_attacker,
                        trader_name=self.data.player_names.get(trade_attacker, "Unknown"),
                        traded_player_id=attacker_id,
                        traded_player_name=self.data.player_names.get(attacker_id, "Unknown"),
                        trade_weapon=str(trade_candidate.get(weapon_col, "")) if weapon_col else "",
                        round_num=round_num,
                    )
                    self._trade_kills.append(trade)
                    break  # Only count first trade

    def _analyze_opening_duels(
        self,
        kills_df: pd.DataFrame,
        att_col: str,
        vic_col: str,
        tick_col: str | None,
        round_col: str | None,
    ) -> None:
        """Detect first kills of each round."""
        if round_col is None:
            logger.warning("No round column for opening duel analysis")
            return

        weapon_col = self._find_col(kills_df, ["weapon"])
        hs_col = self._find_col(kills_df, ["headshot"])

        # Group by round and get first kill
        for round_num in kills_df[round_col].unique():
            round_kills = kills_df[kills_df[round_col] == round_num]

            if tick_col:
                round_kills = round_kills.sort_values(tick_col)

            if round_kills.empty:
                continue

            first_kill = round_kills.iloc[0]

            winner_id = safe_int(first_kill[att_col])
            loser_id = safe_int(first_kill[vic_col])

            if winner_id == 0 or loser_id == 0:
                continue

            opening = OpeningDuel(
                round_num=safe_int(round_num),
                tick=safe_int(first_kill.get(tick_col, 0)) if tick_col else 0,
                winner_id=winner_id,
                winner_name=self.data.player_names.get(winner_id, "Unknown"),
                winner_team=self.data.player_teams.get(winner_id, 0),
                weapon=str(first_kill.get(weapon_col, "")) if weapon_col else "",
                headshot=bool(first_kill.get(hs_col, False)) if hs_col else False,
                loser_id=loser_id,
                loser_name=self.data.player_names.get(loser_id, "Unknown"),
                loser_team=self.data.player_teams.get(loser_id, 0),
            )
            self._opening_duels.append(opening)

    def _analyze_multi_kills(
        self, kills_df: pd.DataFrame, att_col: str, vic_col: str, round_col: str | None
    ) -> None:
        """Detect multi-kills (2k, 3k, 4k, ace) per round."""
        if round_col is None:
            return

        weapon_col = self._find_col(kills_df, ["weapon"])
        hs_col = self._find_col(kills_df, ["headshot"])

        for round_num in kills_df[round_col].unique():
            round_kills = kills_df[kills_df[round_col] == round_num]

            # Count kills per player in this round
            kill_counts = round_kills[att_col].value_counts()

            for player_id, count in kill_counts.items():
                if count < 2:
                    continue

                player_id = safe_int(player_id)
                if player_id == 0:
                    continue

                # Get the kills
                player_round_kills = round_kills[round_kills[att_col] == player_id]
                kill_list = []
                all_hs = True

                for _, k in player_round_kills.iterrows():
                    kill_info = {
                        "victim_id": safe_int(k[vic_col]),
                        "victim_name": self.data.player_names.get(safe_int(k[vic_col]), "Unknown"),
                        "weapon": str(k.get(weapon_col, "")) if weapon_col else "",
                    }
                    kill_list.append(kill_info)

                    if hs_col and not k.get(hs_col, False):
                        all_hs = False

                multi = MultiKill(
                    round_num=safe_int(round_num),
                    player_id=player_id,
                    player_name=self.data.player_names.get(player_id, "Unknown"),
                    kill_count=int(count),
                    kills=kill_list,
                    all_headshots=all_hs,
                )
                self._multi_kills.append(multi)

    def _analyze_clutches(
        self,
        kills_df: pd.DataFrame,
        att_col: str,
        vic_col: str,
        tick_col: str | None,
        round_col: str | None,
    ) -> None:
        """Detect clutch situations (1vX scenarios)."""
        if round_col is None or tick_col is None:
            return

        for round_num in kills_df[round_col].unique():
            round_kills = kills_df[kills_df[round_col] == round_num].sort_values(tick_col)

            if len(round_kills) < 2:
                continue

            # Track alive players as we go through kills
            t_alive = {sid for sid, team in self.data.player_teams.items() if team == 2}
            ct_alive = {sid for sid, team in self.data.player_teams.items() if team == 3}

            for _, kill in round_kills.iterrows():
                victim_id = safe_int(kill[vic_col])
                safe_int(kill[att_col])

                # Remove victim from alive set
                if victim_id in t_alive:
                    t_alive.discard(victim_id)
                elif victim_id in ct_alive:
                    ct_alive.discard(victim_id)

                # Check for clutch situation (1 vs X)
                for team_alive, enemy_alive, team_num in [
                    (t_alive, ct_alive, 2),
                    (ct_alive, t_alive, 3),
                ]:
                    if len(team_alive) == 1 and len(enemy_alive) >= 1:
                        clutcher_id = list(team_alive)[0]
                        enemies = len(enemy_alive)

                        # Check if this clutch already tracked
                        existing = [
                            c
                            for c in self._clutch_situations
                            if c.round_num == safe_int(round_num) and c.clutcher_id == clutcher_id
                        ]
                        if existing:
                            continue

                        scenario = f"1v{enemies}"
                        start_tick = safe_int(kill[tick_col])

                        # Determine outcome by checking remaining kills
                        remaining_kills = round_kills[round_kills[tick_col] > start_tick]

                        clutcher_died = any(
                            safe_int(k[vic_col]) == clutcher_id
                            for _, k in remaining_kills.iterrows()
                        )

                        kills_by_clutcher = sum(
                            1
                            for _, k in remaining_kills.iterrows()
                            if safe_int(k[att_col]) == clutcher_id
                        )

                        if clutcher_died:
                            result = ClutchResult.LOST
                        elif kills_by_clutcher >= enemies:
                            result = ClutchResult.WON
                        else:
                            # Could be bomb explode/defuse win
                            result = (
                                ClutchResult.WON if kills_by_clutcher > 0 else ClutchResult.LOST
                            )

                        clutch = ClutchSituation(
                            round_num=safe_int(round_num),
                            start_tick=start_tick,
                            end_tick=safe_int(round_kills.iloc[-1][tick_col]),
                            clutcher_id=clutcher_id,
                            clutcher_name=self.data.player_names.get(clutcher_id, "Unknown"),
                            clutcher_team=team_num,
                            scenario=scenario,
                            enemies_alive=enemies,
                            result=result,
                            kills_in_clutch=kills_by_clutcher,
                            damage_in_clutch=0,  # Would need damage data per round
                        )
                        self._clutch_situations.append(clutch)

    def _build_player_stats(self) -> dict[int, PlayerCombatStats]:
        """Build per-player combat statistics."""
        stats: dict[int, PlayerCombatStats] = {}

        for steam_id, name in self.data.player_names.items():
            # Trade stats
            trades_given = sum(1 for t in self._trade_kills if t.original_victim_id == steam_id)
            trades_taken = sum(1 for t in self._trade_kills if t.trader_id == steam_id)

            # Count deaths
            kills_df = self.data.kills_df
            vic_col = self._find_col(kills_df, ["user_steamid", "victim_steamid"])
            total_deaths = (
                len(kills_df[kills_df[vic_col] == steam_id])
                if vic_col and not kills_df.empty
                else 0
            )
            traded_deaths = trades_given
            untraded_deaths = max(0, total_deaths - traded_deaths)

            # Average trade time
            player_trades = [t for t in self._trade_kills if t.trader_id == steam_id]
            trade_time_avg = (
                sum(t.time_delta_ms for t in player_trades) / len(player_trades)
                if player_trades
                else 0.0
            )

            # Opening duel stats
            opening_kills = sum(1 for o in self._opening_duels if o.winner_id == steam_id)
            opening_deaths = sum(1 for o in self._opening_duels if o.loser_id == steam_id)
            opening_total = opening_kills + opening_deaths
            opening_rate = opening_kills / opening_total if opening_total > 0 else 0.0

            # Clutch stats
            player_clutches = [c for c in self._clutch_situations if c.clutcher_id == steam_id]
            clutch_attempts = len(player_clutches)
            clutch_wins = sum(1 for c in player_clutches if c.result == ClutchResult.WON)
            clutch_rate = clutch_wins / clutch_attempts if clutch_attempts > 0 else 0.0

            clutches_by_scenario: dict[str, tuple[int, int]] = {}
            for scenario in CLUTCH_SCENARIOS:
                attempts = sum(1 for c in player_clutches if c.scenario == scenario)
                wins = sum(
                    1
                    for c in player_clutches
                    if c.scenario == scenario and c.result == ClutchResult.WON
                )
                if attempts > 0:
                    clutches_by_scenario[scenario] = (attempts, wins)

            # Multi-kill stats
            player_multis = [m for m in self._multi_kills if m.player_id == steam_id]
            double_kills = sum(1 for m in player_multis if m.kill_count == 2)
            triple_kills = sum(1 for m in player_multis if m.kill_count == 3)
            quad_kills = sum(1 for m in player_multis if m.kill_count == 4)
            aces = sum(1 for m in player_multis if m.kill_count >= 5)

            stats[steam_id] = PlayerCombatStats(
                steam_id=steam_id,
                name=name,
                trades_given=trades_given,
                trades_taken=trades_taken,
                traded_deaths=traded_deaths,
                untraded_deaths=untraded_deaths,
                trade_kill_time_avg_ms=trade_time_avg,
                opening_kills=opening_kills,
                opening_deaths=opening_deaths,
                opening_kill_rate=opening_rate,
                opening_kills_won_rounds=0,  # Would need round outcome data
                clutch_attempts=clutch_attempts,
                clutch_wins=clutch_wins,
                clutch_win_rate=clutch_rate,
                clutches_by_scenario=clutches_by_scenario,
                double_kills=double_kills,
                triple_kills=triple_kills,
                quad_kills=quad_kills,
                aces=aces,
            )

        return stats

    def _calculate_team_trade_rates(self) -> dict[int, float]:
        """Calculate trade success rate per team."""
        team_deaths: dict[int, int] = {2: 0, 3: 0}
        team_traded: dict[int, int] = {2: 0, 3: 0}

        for trade in self._trade_kills:
            victim_team = self.data.player_teams.get(trade.original_victim_id, 0)
            if victim_team in team_traded:
                team_traded[victim_team] += 1

        # Count total deaths per team
        kills_df = self.data.kills_df
        vic_col = self._find_col(kills_df, ["user_steamid", "victim_steamid"])
        if vic_col and not kills_df.empty:
            for _, kill in kills_df.iterrows():
                victim_id = safe_int(kill[vic_col])
                victim_team = self.data.player_teams.get(victim_id, 0)
                if victim_team in team_deaths:
                    team_deaths[victim_team] += 1

        return {
            team: (team_traded[team] / team_deaths[team] if team_deaths[team] > 0 else 0.0)
            for team in [2, 3]
        }

    def _calculate_team_opening_rates(self) -> dict[int, float]:
        """Calculate opening duel win rate per team."""
        team_wins: dict[int, int] = {2: 0, 3: 0}
        team_total: dict[int, int] = {2: 0, 3: 0}

        for opening in self._opening_duels:
            winner_team = opening.winner_team
            loser_team = opening.loser_team

            if winner_team in team_wins:
                team_wins[winner_team] += 1
                team_total[winner_team] += 1
            if loser_team in team_total:
                team_total[loser_team] += 1

        return {
            team: (team_wins[team] / team_total[team] if team_total[team] > 0 else 0.0)
            for team in [2, 3]
        }


def analyze_combat(demo_data: DemoData) -> CombatAnalysisResult:
    """
    Convenience function to analyze combat from demo data.

    Args:
        demo_data: Parsed demo data to analyze.

    Returns:
        CombatAnalysisResult containing all combat metrics.
    """
    analyzer = CombatAnalyzer(demo_data)
    return analyzer.analyze()
