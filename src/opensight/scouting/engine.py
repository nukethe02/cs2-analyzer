"""
Scouting Engine - Multi-demo aggregation for opponent analysis.

Aggregates data across multiple demos to generate comprehensive
player profiles and team tendencies.
"""

import logging
from collections import defaultdict
from statistics import mean, stdev

import pandas as pd

from opensight.analysis.analytics import MatchAnalysis, PlayerMatchStats
from opensight.core.parser import DemoData
from opensight.scouting.models import (
    EconomyTendency,
    MapTendency,
    PlayerScoutProfile,
    PlayStyle,
    PositionTendency,
    TeamScoutReport,
    WeaponPreference,
)

logger = logging.getLogger(__name__)


class ScoutingEngine:
    """
    Multi-demo aggregation engine for scouting opponent teams.

    Usage:
        engine = ScoutingEngine()
        for demo, analysis in demos:
            engine.add_demo(demo, analysis)
        engine.set_opponent_team(opponent_steamids)
        report = engine.generate_report()
    """

    def __init__(self) -> None:
        """Initialize an empty scouting engine."""
        self._demos: list[DemoData] = []
        self._analyses: list[MatchAnalysis] = []
        self._opponent_steamids: set[int] = set()
        self._team_name: str = "Unknown Team"

        # Cached aggregations
        self._player_kills_cache: dict[int, pd.DataFrame] = {}
        self._player_deaths_cache: dict[int, pd.DataFrame] = {}

    def add_demo(self, demo_data: DemoData, analysis: MatchAnalysis) -> dict:
        """
        Add a parsed demo to the scouting pool.

        Args:
            demo_data: Parsed demo data from DemoParser
            analysis: Match analysis from DemoAnalyzer

        Returns:
            Dict with demo info and player list for team selection
        """
        self._demos.append(demo_data)
        self._analyses.append(analysis)

        # Clear caches when new demo added
        self._player_kills_cache.clear()
        self._player_deaths_cache.clear()

        # Extract player info for selection UI
        players = []
        for steamid, stats in analysis.players.items():
            players.append(
                {
                    "steamid": steamid,
                    "name": stats.name,
                    "team": stats.team,
                    "kills": stats.kills,
                    "deaths": stats.deaths,
                }
            )

        return {
            "demo_index": len(self._demos) - 1,
            "map_name": demo_data.map_name,
            "total_rounds": analysis.total_rounds,
            "players": players,
            "team1_name": analysis.team1_name,
            "team2_name": analysis.team2_name,
        }

    def set_opponent_team(self, steamids: list[int], team_name: str = "Opponent") -> None:
        """
        Specify which players are opponents to scout.

        Args:
            steamids: List of Steam IDs for opponent players
            team_name: Name of the opponent team
        """
        self._opponent_steamids = set(steamids)
        self._team_name = team_name

    @property
    def demo_count(self) -> int:
        """Number of demos loaded."""
        return len(self._demos)

    @property
    def maps_included(self) -> list[str]:
        """List of unique maps in the scouting pool."""
        return list({d.map_name for d in self._demos})

    def generate_report(self) -> TeamScoutReport:
        """
        Generate comprehensive scouting report for the opponent team.

        Returns:
            TeamScoutReport with player profiles, tendencies, and anti-strats
        """
        if not self._demos:
            raise ValueError("No demos loaded. Call add_demo() first.")

        if not self._opponent_steamids:
            raise ValueError("No opponent team specified. Call set_opponent_team() first.")

        # Build player profiles
        player_profiles = []
        for steamid in self._opponent_steamids:
            profile = self._build_player_profile(steamid)
            if profile:
                player_profiles.append(profile)

        # Sort by average performance (kills per round)
        player_profiles.sort(key=lambda p: p.avg_kills_per_round, reverse=True)

        # Build map tendencies
        map_tendencies = self._build_map_tendencies()

        # Analyze economy patterns
        economy_tendency, force_rate, eco_rate = self._analyze_economy()

        # Generate anti-strats
        from opensight.scouting.anti_strats import generate_anti_strats

        anti_strats = generate_anti_strats(player_profiles, map_tendencies, economy_tendency)

        # Calculate total rounds
        total_rounds = sum(a.total_rounds for a in self._analyses)

        # Determine confidence level
        if self.demo_count >= 4:
            confidence = "high"
        elif self.demo_count >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        return TeamScoutReport(
            team_name=self._team_name,
            demos_analyzed=self.demo_count,
            total_rounds=total_rounds,
            players=player_profiles,
            map_tendencies=map_tendencies,
            economy_tendency=economy_tendency,
            force_buy_rate=force_rate,
            eco_round_rate=eco_rate,
            anti_strats=anti_strats,
            confidence_level=confidence,
            maps_analyzed=self.maps_included,
        )

    def _build_player_profile(self, steamid: int) -> PlayerScoutProfile | None:
        """Build aggregated profile for a single player."""
        # Collect stats across all demos
        player_stats_list: list[PlayerMatchStats] = []
        player_name = "Unknown"

        for analysis in self._analyses:
            if steamid in analysis.players:
                stats = analysis.players[steamid]
                player_stats_list.append(stats)
                player_name = stats.name

        if not player_stats_list:
            logger.warning(f"No stats found for steamid {steamid}")
            return None

        # Aggregate basic metrics
        total_rounds = sum(s.rounds_played for s in player_stats_list)
        if total_rounds == 0:
            return None

        total_kills = sum(s.kills for s in player_stats_list)
        total_deaths = sum(s.deaths for s in player_stats_list)

        # Calculate per-round averages
        avg_kills_per_round = total_kills / total_rounds
        avg_deaths_per_round = total_deaths / total_rounds

        # ADR and KAST (weighted by rounds played)
        weighted_adr = sum(s.adr * s.rounds_played for s in player_stats_list) / total_rounds
        weighted_kast = (
            sum(s.kast_percentage * s.rounds_played for s in player_stats_list) / total_rounds
        )

        # Headshot rate
        total_headshots = sum(s.headshots for s in player_stats_list)
        hs_rate = total_headshots / total_kills if total_kills > 0 else 0

        # Entry/opening statistics
        total_entry_attempts = sum(
            s.entry_frags.total_entry_frags + s.entry_frags.total_entry_deaths
            for s in player_stats_list
        )
        total_entry_wins = sum(s.entry_frags.total_entry_frags for s in player_stats_list)
        entry_attempt_rate = total_entry_attempts / total_rounds if total_rounds > 0 else 0
        entry_success_rate = (
            total_entry_wins / total_entry_attempts if total_entry_attempts > 0 else 0
        )

        # Opening duel stats
        total_opening_wins = sum(s.opening_duels.wins for s in player_stats_list)
        total_opening_attempts = sum(s.opening_duels.attempts for s in player_stats_list)
        opening_win_rate = (
            total_opening_wins / total_opening_attempts if total_opening_attempts > 0 else 0
        )

        # AWP usage
        awp_rounds = 0
        awp_kills = 0
        for stats in player_stats_list:
            awp_kills_demo = stats.weapon_kills.get("AWP", 0)
            if awp_kills_demo > 0:
                awp_kills += awp_kills_demo
                # Rough estimate: if player got AWP kills, they had AWP
                awp_rounds += min(awp_kills_demo, stats.rounds_played // 3)

        awp_usage_rate = awp_rounds / total_rounds if total_rounds > 0 else 0
        awp_kpr = awp_kills / awp_rounds if awp_rounds > 0 else 0

        # Timing analysis (use TTD values if available)
        all_ttd = []
        for stats in player_stats_list:
            all_ttd.extend(stats.true_ttd_values)
        avg_first_kill_time = mean(all_ttd) / 1000 if all_ttd else 0  # Convert ms to seconds

        # Consistency score (inverse of stat variance)
        if len(player_stats_list) >= 2:
            kpr_values = [
                s.kills / s.rounds_played if s.rounds_played > 0 else 0 for s in player_stats_list
            ]
            kpr_stdev = stdev(kpr_values) if len(kpr_values) > 1 else 0
            consistency_score = max(0, 100 - (kpr_stdev * 200))  # Higher stdev = lower consistency
        else:
            consistency_score = 50  # Unknown with single demo

        # Calculate play style and aggression
        aggression_score = self._calculate_aggression_score(
            entry_attempt_rate, entry_success_rate, avg_kills_per_round, avg_deaths_per_round
        )

        if aggression_score >= 60:
            play_style = PlayStyle.AGGRESSIVE
        elif aggression_score <= 40:
            play_style = PlayStyle.PASSIVE
        else:
            play_style = PlayStyle.MIXED

        # Build weapon preferences
        weapon_kills_total: dict[str, int] = defaultdict(int)
        for stats in player_stats_list:
            for weapon, kills in stats.weapon_kills.items():
                weapon_kills_total[weapon] += kills

        weapon_prefs = []
        for weapon, kills in sorted(weapon_kills_total.items(), key=lambda x: -x[1])[:5]:
            weapon_prefs.append(
                WeaponPreference(
                    weapon_name=weapon,
                    usage_rate=kills / total_kills if total_kills > 0 else 0,
                    kills_with=kills,
                    accuracy=0.0,  # Would need shot data
                )
            )

        # Build position preferences (by map)
        favorite_positions = self._analyze_positions(steamid)

        # Clutch stats
        total_clutch_attempts = sum(s.clutches.total_situations for s in player_stats_list)
        total_clutch_wins = sum(s.clutches.total_wins for s in player_stats_list)

        return PlayerScoutProfile(
            steamid=str(steamid),
            name=player_name,
            play_style=play_style,
            aggression_score=aggression_score,
            consistency_score=consistency_score,
            avg_kills_per_round=avg_kills_per_round,
            avg_deaths_per_round=avg_deaths_per_round,
            avg_adr=weighted_adr,
            avg_kast=weighted_kast,
            headshot_rate=hs_rate,
            entry_attempt_rate=entry_attempt_rate,
            entry_success_rate=entry_success_rate,
            opening_duel_win_rate=opening_win_rate,
            awp_usage_rate=awp_usage_rate,
            awp_kills_per_awp_round=awp_kpr,
            avg_first_kill_time_seconds=avg_first_kill_time,
            avg_rotation_time_seconds=0,  # Would need position tracking
            favorite_positions=favorite_positions,
            weapon_preferences=weapon_prefs,
            clutch_attempts=total_clutch_attempts,
            clutch_wins=total_clutch_wins,
            demos_analyzed=len(player_stats_list),
            rounds_analyzed=total_rounds,
        )

    def _calculate_aggression_score(
        self,
        entry_rate: float,
        entry_success: float,
        kpr: float,
        dpr: float,
    ) -> float:
        """
        Calculate aggression score (0-100).

        Higher score = more aggressive playstyle.
        Factors:
        - Entry attempt rate (30%)
        - K/D ratio (20%)
        - Deaths per round (30%)
        - Entry success rate (20%)
        """
        # Entry rate component (high entry rate = aggressive)
        entry_component = min(entry_rate * 100, 100) * 0.30

        # K/D ratio component (high K/D with high kills = aggressive fragger)
        kd_ratio = kpr / dpr if dpr > 0 else kpr
        kd_component = min(kd_ratio * 30, 100) * 0.20

        # Deaths per round (high deaths = aggressive/risky)
        dpr_component = min(dpr * 100, 100) * 0.30

        # Entry success (successful aggression)
        success_component = entry_success * 100 * 0.20

        return entry_component + kd_component + dpr_component + success_component

    def _analyze_positions(self, steamid: int) -> dict[str, list[PositionTendency]]:
        """Analyze position tendencies by map for a player."""
        positions_by_map: dict[str, list[PositionTendency]] = {}

        for demo, _analysis in zip(self._demos, self._analyses, strict=True):
            map_name = demo.map_name
            if map_name not in positions_by_map:
                positions_by_map[map_name] = []
            # Zone analysis would need map coordinates - placeholder for now

        return positions_by_map

    def _build_map_tendencies(self) -> list[MapTendency]:
        """Build team tendencies grouped by map."""
        map_groups: dict[str, list[tuple[DemoData, MatchAnalysis]]] = defaultdict(list)

        for demo, analysis in zip(self._demos, self._analyses, strict=True):
            map_groups[demo.map_name].append((demo, analysis))

        tendencies = []
        for map_name, demo_pairs in map_groups.items():
            tendency = self._analyze_map_tendency(map_name, demo_pairs)
            if tendency:
                tendencies.append(tendency)

        return tendencies

    def _analyze_map_tendency(
        self, map_name: str, demo_pairs: list[tuple[DemoData, MatchAnalysis]]
    ) -> MapTendency | None:
        """Analyze team tendency on a specific map."""
        if not demo_pairs:
            return None

        total_rounds = sum(a.total_rounds for _, a in demo_pairs)

        # Analyze timing patterns from kills
        t_first_contact_times: list[float] = []
        t_execute_times: list[float] = []

        for demo, _analysis in demo_pairs:
            if not hasattr(demo, "kills_df") or demo.kills_df.empty:
                continue

            kills_df = demo.kills_df

            # Filter to opponent team kills on T side
            opponent_kills = kills_df[
                (kills_df["attacker_steamid"].isin(self._opponent_steamids))
                & (kills_df["attacker_side"].str.contains("T", case=False, na=False))
            ]

            if not opponent_kills.empty and "tick" in opponent_kills.columns:
                # Group by round and get first kill tick
                round_first_kills = opponent_kills.groupby("round")["tick"].min()
                # Convert ticks to seconds (approximate)
                tick_rate = demo.tick_rate if demo.tick_rate > 0 else 64
                for tick in round_first_kills:
                    t_first_contact_times.append(tick / tick_rate)

        avg_first_contact = mean(t_first_contact_times) if t_first_contact_times else 30.0
        avg_execute_time = mean(t_execute_times) if t_execute_times else 45.0

        # Calculate T-side aggression
        t_aggression = self._calculate_side_aggression("T", demo_pairs)
        ct_aggression = self._calculate_side_aggression("CT", demo_pairs)

        return MapTendency(
            map_name=map_name,
            demos_analyzed=len(demo_pairs),
            rounds_analyzed=total_rounds,
            t_side_default_setup="Standard default with lurk",  # Would need deeper analysis
            t_side_common_executes=["A execute", "B split"],  # Placeholder
            t_side_aggression=t_aggression,
            ct_side_default_setup="Standard 2-1-2",  # Placeholder
            ct_side_rotation_speed="medium",
            ct_side_aggression=ct_aggression,
            avg_execute_time_seconds=avg_execute_time,
            avg_first_contact_seconds=avg_first_contact,
        )

    def _calculate_side_aggression(
        self, side: str, demo_pairs: list[tuple[DemoData, MatchAnalysis]]
    ) -> float:
        """Calculate aggression level for a specific side."""
        entry_attempts = 0
        total_rounds = 0

        for _demo, analysis in demo_pairs:
            for steamid in self._opponent_steamids:
                if steamid in analysis.players:
                    stats = analysis.players[steamid]
                    # This is approximate - would need per-side entry stats
                    if side == "T":
                        entry_attempts += stats.entry_frags.total_entry_frags
                        total_rounds += (
                            stats.t_stats.rounds_played if hasattr(stats, "t_stats") else 0
                        )
                    else:
                        # CT aggression from pushes
                        total_rounds += (
                            stats.ct_stats.rounds_played if hasattr(stats, "ct_stats") else 0
                        )

        if total_rounds == 0:
            return 50.0

        # Entry rate * 100 gives rough aggression
        return min((entry_attempts / total_rounds) * 100, 100)

    def _analyze_economy(self) -> tuple[EconomyTendency, float, float]:
        """
        Analyze team economy patterns.

        Returns:
            Tuple of (tendency, force_buy_rate, eco_rate)
        """
        force_rounds = 0
        eco_rounds = 0
        total_rounds = 0

        for demo, analysis in zip(self._demos, self._analyses, strict=True):
            total_rounds += analysis.total_rounds

            # Analyze round economy from rounds data
            if hasattr(demo, "rounds_df") and not demo.rounds_df.empty:
                # Economy classification would go here
                # For now, use simplified heuristic based on equipment value if available
                pass

        # Default to balanced if can't determine
        if total_rounds == 0:
            return EconomyTendency.BALANCED, 0.2, 0.2

        force_rate = force_rounds / total_rounds if total_rounds > 0 else 0.2
        eco_rate = eco_rounds / total_rounds if total_rounds > 0 else 0.2

        if force_rate > 0.35:
            tendency = EconomyTendency.AGGRESSIVE
        elif eco_rate > 0.25:
            tendency = EconomyTendency.CONSERVATIVE
        else:
            tendency = EconomyTendency.BALANCED

        return tendency, force_rate, eco_rate


def create_scouting_engine() -> ScoutingEngine:
    """Factory function to create a new scouting engine."""
    return ScoutingEngine()
