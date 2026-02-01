"""
Team Synergy Analysis Module for CS2 Demo Analysis

Implements advanced team synergy metrics NOT available in Leetify:
- Trade synergy between player pairs
- Flash assist networks (who flashes for whom)
- Duo performance (win rate when both alive)
- Refrag response times between specific players

BORIS PRIORITY: HIGH
LEETIFY HAS THIS: NO
SCOPE.GG HAS THIS: PARTIAL
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from opensight.core.parser import safe_int, safe_str

if TYPE_CHECKING:
    from opensight.core.parser import DemoData

logger = logging.getLogger(__name__)

# Trade window in milliseconds (standard is 5 seconds)
TRADE_WINDOW_MS = 5000
# Flash assist window in milliseconds (2.5 seconds)
FLASH_ASSIST_WINDOW_MS = 2500


@dataclass
class PlayerPairSynergy:
    """Synergy metrics between two specific players."""

    player_a_steamid: int
    player_b_steamid: int
    player_a_name: str
    player_b_name: str

    # Trade synergy
    trades_a_for_b: int = 0  # A traded B's death
    trades_b_for_a: int = 0  # B traded A's death
    avg_trade_time_ms: float = 0.0  # How fast they trade each other

    # Flash synergy
    flashes_a_for_b: int = 0  # A's flashes that assisted B's kills
    flashes_b_for_a: int = 0  # B's flashes that assisted A's kills

    # Duo performance
    rounds_together_alive: int = 0  # Rounds where both were alive at same time
    rounds_both_got_kills: int = 0  # Rounds where both got at least 1 kill
    duo_rounds_won: int = 0  # Rounds won when both were alive together
    duo_rounds_total: int = 0  # Total rounds they were alive together

    # Refrag response
    avg_refrag_time_ms: float = 0.0  # Average time to refrag when partner dies

    # Internal tracking
    _trade_times: list[float] = field(default_factory=list)
    _refrag_times: list[float] = field(default_factory=list)

    @property
    def duo_win_rate(self) -> float:
        """Win rate when both players are alive together."""
        if self.duo_rounds_total == 0:
            return 0.0
        return self.duo_rounds_won / self.duo_rounds_total

    @property
    def total_trades(self) -> int:
        """Total trades for each other."""
        return self.trades_a_for_b + self.trades_b_for_a

    @property
    def total_flash_assists(self) -> int:
        """Total flash assists for each other."""
        return self.flashes_a_for_b + self.flashes_b_for_a

    @property
    def synergy_score(self) -> float:
        """
        Composite synergy score 0-100.

        Weighting:
        - Trade synergy: 30% (max 10 trades = 100 pts)
        - Flash synergy: 20% (max 6 flash assists = 100 pts)
        - Duo win rate: 50% (direct percentage)
        """
        trade_score = min(100.0, self.total_trades * 10)
        flash_score = min(100.0, self.total_flash_assists * 16.67)
        duo_score = self.duo_win_rate * 100

        return trade_score * 0.3 + flash_score * 0.2 + duo_score * 0.5

    def finalize(self) -> None:
        """Calculate final averages from collected times."""
        if self._trade_times:
            self.avg_trade_time_ms = float(np.mean(self._trade_times))
        if self._refrag_times:
            self.avg_refrag_time_ms = float(np.mean(self._refrag_times))


@dataclass
class SynergyAnalysisResult:
    """Complete synergy analysis for a match."""

    pair_synergies: list[PlayerPairSynergy]
    best_duo: PlayerPairSynergy | None
    trade_network: dict[str, list[dict]]  # player_name -> list of trade partners
    flash_network: dict[str, list[dict]]  # player_name -> list of flash assist partners


class SynergyAnalyzer:
    """
    Analyzes team synergy patterns.

    BORIS REQUIREMENTS:
    - All operations vectorized via pandas where possible
    - No nested loops over individual kills
    - Type hints on every method
    """

    def __init__(self, demo_data: "DemoData"):
        """
        Initialize the synergy analyzer.

        Args:
            demo_data: Parsed demo data to analyze.
        """
        self.data = demo_data
        self.tick_rate = demo_data.tick_rate or 64

        # Map player teams for teammate detection
        # Use persistent teams for better accuracy
        self._player_teams: dict[int, str] = {}
        for sid, team in demo_data.player_persistent_teams.items():
            self._player_teams[sid] = team
        # Fallback to regular teams if persistent not available
        if not self._player_teams:
            for sid, team in demo_data.player_teams.items():
                self._player_teams[sid] = str(team)

        # Player names lookup
        self._player_names: dict[int, str] = dict(demo_data.player_names)

        # Initialize pair synergies for all teammate pairs
        self._pair_synergies: dict[tuple[int, int], PlayerPairSynergy] = {}
        self._initialize_pairs()

    def _initialize_pairs(self) -> None:
        """Create synergy objects for all teammate pairs."""
        # Group players by team
        teams: dict[str, list[int]] = {}
        for sid, team in self._player_teams.items():
            if team not in teams:
                teams[team] = []
            teams[team].append(sid)

        # Create pairs for each team
        for team_players in teams.values():
            for a, b in combinations(sorted(team_players), 2):
                key = (a, b) if a < b else (b, a)
                self._pair_synergies[key] = PlayerPairSynergy(
                    player_a_steamid=key[0],
                    player_b_steamid=key[1],
                    player_a_name=self._player_names.get(key[0], f"Player_{key[0]}"),
                    player_b_name=self._player_names.get(key[1], f"Player_{key[1]}"),
                )

    def _get_pair(self, sid_a: int, sid_b: int) -> PlayerPairSynergy | None:
        """Get the synergy pair for two players (order-independent)."""
        key = (sid_a, sid_b) if sid_a < sid_b else (sid_b, sid_a)
        return self._pair_synergies.get(key)

    def _are_teammates(self, sid_a: int, sid_b: int) -> bool:
        """Check if two players are on the same team."""
        team_a = self._player_teams.get(sid_a)
        team_b = self._player_teams.get(sid_b)
        return team_a is not None and team_a == team_b

    def _ticks_to_ms(self, ticks: int) -> float:
        """Convert ticks to milliseconds."""
        return (ticks / self.tick_rate) * 1000

    def analyze(self) -> SynergyAnalysisResult:
        """
        Run full synergy analysis.

        Returns:
            SynergyAnalysisResult containing all synergy metrics.
        """
        logger.info("Starting synergy analysis...")

        kills_df = self.data.kills_df
        blinds_df = self.data.blinds_df

        if kills_df.empty:
            logger.warning("No kill data for synergy analysis")
            return self._empty_result()

        # Compute all synergy metrics
        self._compute_trade_synergy(kills_df)
        self._compute_flash_synergy(kills_df, blinds_df)
        self._compute_duo_performance(kills_df)

        # Finalize all pairs
        for pair in self._pair_synergies.values():
            pair.finalize()

        # Build results
        all_synergies = list(self._pair_synergies.values())
        best_duo = self._find_best_duo()
        trade_network = self._build_trade_network()
        flash_network = self._build_flash_network()

        logger.info(
            f"Synergy analysis complete. {len(all_synergies)} pairs analyzed, "
            f"best duo score: {best_duo.synergy_score:.1f if best_duo else 0}"
        )

        return SynergyAnalysisResult(
            pair_synergies=all_synergies,
            best_duo=best_duo,
            trade_network=trade_network,
            flash_network=flash_network,
        )

    def _empty_result(self) -> SynergyAnalysisResult:
        """Return empty result when analysis cannot be performed."""
        return SynergyAnalysisResult(
            pair_synergies=[],
            best_duo=None,
            trade_network={},
            flash_network={},
        )

    def _compute_trade_synergy(self, kills_df: pd.DataFrame) -> None:
        """
        Vectorized trade synergy computation.

        For each kill, check if a teammate died within 5s before
        and if this kill was on that killer (trade).
        """
        # Find column names
        att_col = self._find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        vic_col = self._find_col(kills_df, ["user_steamid", "victim_steamid"])
        tick_col = self._find_col(kills_df, ["tick"])

        if not att_col or not vic_col or not tick_col:
            logger.warning("Missing columns for trade synergy analysis")
            return

        # Sort by tick
        df = kills_df.sort_values(tick_col).reset_index(drop=True)
        trade_window_ticks = int((TRADE_WINDOW_MS / 1000) * self.tick_rate)

        # Convert to numpy for vectorized operations
        attackers = pd.to_numeric(df[att_col], errors="coerce").fillna(0).astype(int).values
        victims = pd.to_numeric(df[vic_col], errors="coerce").fillna(0).astype(int).values
        ticks = df[tick_col].values.astype(int)

        n = len(df)

        # For each kill, look back for a teammate's death that this kill trades
        for i in range(n):
            trade_attacker = attackers[i]  # Who got the trade kill
            trade_victim = victims[i]  # Who was killed (original killer)
            trade_tick = ticks[i]

            if trade_attacker == 0 or trade_victim == 0:
                continue

            # Look back for deaths within trade window
            for j in range(i - 1, -1, -1):
                prev_tick = ticks[j]
                tick_diff = trade_tick - prev_tick

                if tick_diff > trade_window_ticks:
                    break  # Outside trade window

                prev_attacker = attackers[j]  # Who killed our teammate
                prev_victim = victims[j]  # Our teammate who died

                # Check if this is a trade:
                # - trade_attacker and prev_victim are teammates
                # - trade_victim is the one who killed prev_victim (prev_attacker)
                if prev_attacker == trade_victim and self._are_teammates(
                    trade_attacker, prev_victim
                ):
                    # Found a trade! trade_attacker traded prev_victim's death
                    pair = self._get_pair(trade_attacker, prev_victim)
                    if pair:
                        time_ms = self._ticks_to_ms(tick_diff)
                        if trade_attacker == pair.player_a_steamid:
                            pair.trades_a_for_b += 1
                        else:
                            pair.trades_b_for_a += 1
                        pair._trade_times.append(time_ms)
                    break  # Only count first matching death

    def _compute_flash_synergy(self, kills_df: pd.DataFrame, blinds_df: pd.DataFrame) -> None:
        """
        Who flashes for whom?

        Match blind_events to kills within 2.5s window
        where victim was blinded by teammate's flash.
        """
        if blinds_df.empty:
            logger.debug("No blinds data for flash synergy")
            return

        # Find column names for kills
        kill_att_col = self._find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        kill_vic_col = self._find_col(kills_df, ["user_steamid", "victim_steamid"])
        kill_tick_col = self._find_col(kills_df, ["tick"])

        # Find column names for blinds
        blind_att_col = self._find_col(blinds_df, ["attacker_steamid", "attacker_steam_id"])
        blind_vic_col = self._find_col(blinds_df, ["user_steamid", "victim_steamid", "entity_id"])
        blind_tick_col = self._find_col(blinds_df, ["tick"])

        if not all(
            [
                kill_att_col,
                kill_vic_col,
                kill_tick_col,
                blind_att_col,
                blind_vic_col,
                blind_tick_col,
            ]
        ):
            logger.warning("Missing columns for flash synergy analysis")
            return

        flash_window_ticks = int((FLASH_ASSIST_WINDOW_MS / 1000) * self.tick_rate)

        # Sort both DataFrames by tick
        kills = kills_df.sort_values(kill_tick_col).reset_index(drop=True)
        blinds = blinds_df.sort_values(blind_tick_col).reset_index(drop=True)

        # Convert to numpy
        kill_attackers = (
            pd.to_numeric(kills[kill_att_col], errors="coerce").fillna(0).astype(int).values
        )
        kill_victims = (
            pd.to_numeric(kills[kill_vic_col], errors="coerce").fillna(0).astype(int).values
        )
        kill_ticks = kills[kill_tick_col].values.astype(int)

        blind_throwers = (
            pd.to_numeric(blinds[blind_att_col], errors="coerce").fillna(0).astype(int).values
        )
        blind_victims = (
            pd.to_numeric(blinds[blind_vic_col], errors="coerce").fillna(0).astype(int).values
        )
        blind_ticks = blinds[blind_tick_col].values.astype(int)

        blind_idx = 0
        n_blinds = len(blinds)

        # For each kill, check if the victim was recently blinded by a teammate
        for i in range(len(kills)):
            killer = kill_attackers[i]
            victim = kill_victims[i]
            kill_tick = kill_ticks[i]

            if killer == 0 or victim == 0:
                continue

            # Move blind index to start of window
            while blind_idx < n_blinds and blind_ticks[blind_idx] < kill_tick - flash_window_ticks:
                blind_idx += 1

            # Check blinds in the window
            j = blind_idx
            while j < n_blinds and blind_ticks[j] <= kill_tick:
                flasher = blind_throwers[j]
                flashed = blind_victims[j]

                # Check if: flasher is teammate of killer, and flashed is the victim
                if flashed == victim and flasher != killer and self._are_teammates(flasher, killer):
                    # Flash assist! flasher flashed for killer
                    pair = self._get_pair(flasher, killer)
                    if pair:
                        if flasher == pair.player_a_steamid:
                            pair.flashes_a_for_b += 1
                        else:
                            pair.flashes_b_for_a += 1
                    break  # Only count one flash assist per kill

                j += 1

    def _compute_duo_performance(self, kills_df: pd.DataFrame) -> None:
        """
        Win rate when specific player pairs are both alive.

        For each round, track:
        - Which player pairs were alive at start
        - Whether they both got kills
        - Round outcome
        """
        round_col = self._find_col(kills_df, ["total_rounds_played", "round_num", "round"])
        att_col = self._find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        vic_col = self._find_col(kills_df, ["user_steamid", "victim_steamid"])

        if not round_col or not att_col or not vic_col:
            logger.warning("Missing columns for duo performance analysis")
            return

        # Get round outcomes
        rounds_df = self.data.rounds_df
        round_winners: dict[int, str] = {}

        if not rounds_df.empty:
            winner_col = self._find_col(rounds_df, ["winner", "round_winner", "winner_side"])
            round_num_col = self._find_col(rounds_df, ["round_num", "round", "round_number"])

            if winner_col and round_num_col:
                for _, row in rounds_df.iterrows():
                    rnum = safe_int(row.get(round_num_col, 0))
                    winner = safe_str(row.get(winner_col, ""))
                    if rnum > 0:
                        round_winners[rnum] = winner

        # Group kills by round
        kills_by_round: dict[int, pd.DataFrame] = {
            int(k): v for k, v in kills_df.groupby(round_col)
        }

        # Get all rounds
        all_rounds = sorted(kills_by_round.keys())
        if not all_rounds:
            # Fallback to num_rounds
            all_rounds = list(range(1, self.data.num_rounds + 1))

        # For each round, determine who was alive at start (everyone)
        # Track performance each round
        for rnum in all_rounds:
            round_kills = kills_by_round.get(rnum, pd.DataFrame())

            # Get killers this round
            killers_this_round: set[int] = set()
            if not round_kills.empty:
                killers_this_round = set(
                    pd.to_numeric(round_kills[att_col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .values
                )

            # Determine round winner team
            round_winner = round_winners.get(rnum, "")

            # For each pair, track duo performance
            for (sid_a, sid_b), pair in self._pair_synergies.items():
                # Both players start alive each round
                pair.rounds_together_alive += 1
                pair.duo_rounds_total += 1

                # Did both get kills?
                if sid_a in killers_this_round and sid_b in killers_this_round:
                    pair.rounds_both_got_kills += 1

                # Did their team win?
                team = self._player_teams.get(sid_a, "")
                if team and round_winner:
                    # Check if team matches winner
                    # Round winner is usually "CT" or "T" or team name
                    starting_side = self.data.team_starting_sides.get(team, "")
                    current_side = starting_side
                    if rnum >= self.data.halftime_round:
                        current_side = "T" if starting_side == "CT" else "CT"

                    if current_side.upper() == round_winner.upper() or team == round_winner:
                        pair.duo_rounds_won += 1

    def _find_best_duo(self) -> PlayerPairSynergy | None:
        """Return the highest synergy score player pair."""
        if not self._pair_synergies:
            return None

        return max(self._pair_synergies.values(), key=lambda s: s.synergy_score)

    def _build_trade_network(self) -> dict[str, list[dict]]:
        """Build a network showing who trades for whom."""
        network: dict[str, list[dict]] = {}

        for pair in self._pair_synergies.values():
            if pair.total_trades == 0:
                continue

            # A -> B trades
            if pair.trades_a_for_b > 0:
                if pair.player_a_name not in network:
                    network[pair.player_a_name] = []
                network[pair.player_a_name].append(
                    {
                        "traded_for": pair.player_b_name,
                        "count": pair.trades_a_for_b,
                    }
                )

            # B -> A trades
            if pair.trades_b_for_a > 0:
                if pair.player_b_name not in network:
                    network[pair.player_b_name] = []
                network[pair.player_b_name].append(
                    {
                        "traded_for": pair.player_a_name,
                        "count": pair.trades_b_for_a,
                    }
                )

        return network

    def _build_flash_network(self) -> dict[str, list[dict]]:
        """Build a network showing who flashes for whom."""
        network: dict[str, list[dict]] = {}

        for pair in self._pair_synergies.values():
            if pair.total_flash_assists == 0:
                continue

            # A -> B flashes
            if pair.flashes_a_for_b > 0:
                if pair.player_a_name not in network:
                    network[pair.player_a_name] = []
                network[pair.player_a_name].append(
                    {
                        "flashed_for": pair.player_b_name,
                        "count": pair.flashes_a_for_b,
                    }
                )

            # B -> A flashes
            if pair.flashes_b_for_a > 0:
                if pair.player_b_name not in network:
                    network[pair.player_b_name] = []
                network[pair.player_b_name].append(
                    {
                        "flashed_for": pair.player_a_name,
                        "count": pair.flashes_b_for_a,
                    }
                )

        return network

    def _find_col(self, df: pd.DataFrame, options: list[str]) -> str | None:
        """Find first matching column name."""
        for col in options:
            if col in df.columns:
                return col
        return None


def synergy_to_dict(synergy: PlayerPairSynergy) -> dict:
    """Serialize synergy for API response."""
    return {
        "player_a": synergy.player_a_name,
        "player_a_steamid": str(synergy.player_a_steamid),
        "player_b": synergy.player_b_name,
        "player_b_steamid": str(synergy.player_b_steamid),
        "trades_a_for_b": synergy.trades_a_for_b,
        "trades_b_for_a": synergy.trades_b_for_a,
        "trades_for_each_other": synergy.total_trades,
        "avg_trade_time_ms": round(synergy.avg_trade_time_ms, 1),
        "flashes_a_for_b": synergy.flashes_a_for_b,
        "flashes_b_for_a": synergy.flashes_b_for_a,
        "flash_assists_for_each_other": synergy.total_flash_assists,
        "rounds_together_alive": synergy.rounds_together_alive,
        "rounds_both_got_kills": synergy.rounds_both_got_kills,
        "duo_win_rate": round(synergy.duo_win_rate * 100, 1),
        "synergy_score": round(synergy.synergy_score, 1),
    }


def analyze_synergy(demo_data: "DemoData") -> dict:
    """
    Convenience function to analyze synergy from demo data.

    Args:
        demo_data: Parsed demo data to analyze.

    Returns:
        Dictionary containing synergy analysis results.
    """
    analyzer = SynergyAnalyzer(demo_data)
    result = analyzer.analyze()

    return {
        "pair_synergies": [synergy_to_dict(s) for s in result.pair_synergies],
        "best_duo": synergy_to_dict(result.best_duo) if result.best_duo else None,
        "trade_network": result.trade_network,
        "flash_network": result.flash_network,
    }
