"""
Utility Analysis Module for CS2 Demo Analysis

Implements utility (grenade) tracking and effectiveness:
- Flash effectiveness (enemies blinded)
- HE grenade damage
- Smoke and molotov usage
- Utility spending efficiency
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from opensight.core.parser import DemoData, safe_int

logger = logging.getLogger(__name__)


class GrenadeType(Enum):
    """Types of grenades in CS2."""

    FLASHBANG = "flashbang"
    HE_GRENADE = "hegrenade"
    SMOKE = "smokegrenade"
    MOLOTOV = "molotov"
    INCENDIARY = "incgrenade"
    DECOY = "decoy"


# Grenade costs
GRENADE_COSTS = {
    GrenadeType.FLASHBANG: 200,
    GrenadeType.HE_GRENADE: 300,
    GrenadeType.SMOKE: 300,
    GrenadeType.MOLOTOV: 400,
    GrenadeType.INCENDIARY: 600,
    GrenadeType.DECOY: 50,
}

# Weapon names that are grenades
GRENADE_WEAPONS = {
    "flashbang": GrenadeType.FLASHBANG,
    "hegrenade": GrenadeType.HE_GRENADE,
    "smokegrenade": GrenadeType.SMOKE,
    "molotov": GrenadeType.MOLOTOV,
    "incgrenade": GrenadeType.INCENDIARY,
    "inferno": GrenadeType.MOLOTOV,  # Molotov fire
    "decoy": GrenadeType.DECOY,
}


@dataclass
class GrenadeDamageEvent:
    """Damage dealt by a grenade."""

    tick: int
    round_num: int
    thrower_id: int
    thrower_name: str
    victim_id: int
    victim_name: str
    grenade_type: GrenadeType
    damage: int
    is_team_damage: bool


@dataclass
class PlayerUtilityStats:
    """Utility statistics for a single player."""

    steam_id: int
    name: str

    # Grenade counts (estimated from damage events)
    he_grenades_thrown: int
    he_damage_total: int
    he_damage_avg: float
    he_kills: int

    molotov_thrown: int
    molotov_damage_total: int
    molotov_ticks: int  # Damage instances (proxy for time burning)

    # Flash stats (would need flash events for full tracking)
    flashes_thrown: int  # Estimated if flash events available

    # Smoke stats
    smokes_thrown: int

    # Efficiency metrics
    utility_damage_per_round: float
    utility_cost_total: int
    damage_per_dollar: float

    # Team damage (bad utility)
    team_damage_total: int
    team_damage_instances: int


@dataclass
class UtilityAnalysisResult:
    """Complete utility analysis for a match."""

    grenade_damage_events: list[GrenadeDamageEvent]
    player_stats: dict[int, PlayerUtilityStats]

    # Team-level stats
    team_utility_damage: dict[int, int]  # team -> total utility damage
    team_utility_efficiency: dict[int, float]  # team -> damage per utility cost


class UtilityAnalyzer:
    """Analyzer for utility metrics from parsed demo data."""

    def __init__(self, demo_data: DemoData):
        """
        Initialize the utility analyzer.

        Args:
            demo_data: Parsed demo data to analyze.
        """
        self.data = demo_data
        self._grenade_events: list[GrenadeDamageEvent] = []

    def analyze(self) -> UtilityAnalysisResult:
        """
        Run full utility analysis on the demo data.

        Returns:
            UtilityAnalysisResult containing all utility metrics.
        """
        logger.info("Starting utility analysis...")

        damages_df = self.data.damages_df
        if damages_df is None or damages_df.empty:
            logger.warning("No damage data for utility analysis")
            return self._empty_result()

        # Find columns
        att_col = self._find_col(damages_df, ["attacker_steamid", "attacker_steam_id"])
        vic_col = self._find_col(damages_df, ["user_steamid", "victim_steamid"])
        dmg_col = self._find_col(damages_df, ["dmg_health", "damage", "dmg"])
        weapon_col = self._find_col(damages_df, ["weapon"])
        tick_col = self._find_col(damages_df, ["tick"])

        if not att_col or not vic_col or not weapon_col:
            logger.warning("Missing required columns for utility analysis")
            return self._empty_result()

        # Extract grenade damage events
        self._extract_grenade_damage(damages_df, att_col, vic_col, dmg_col, weapon_col, tick_col)

        # Build player stats
        player_stats = self._build_player_stats()

        # Build team stats
        team_damage = self._calculate_team_utility_damage()
        team_efficiency = self._calculate_team_efficiency()

        logger.info(f"Utility analysis complete. {len(self._grenade_events)} grenade damage events")

        return UtilityAnalysisResult(
            grenade_damage_events=self._grenade_events,
            player_stats=player_stats,
            team_utility_damage=team_damage,
            team_utility_efficiency=team_efficiency,
        )

    def _find_col(self, df: pd.DataFrame, options: list[str]) -> str | None:
        """Find first matching column name."""
        for col in options:
            if col in df.columns:
                return col
        return None

    def _empty_result(self) -> UtilityAnalysisResult:
        """Return empty result when analysis cannot be performed."""
        return UtilityAnalysisResult(
            grenade_damage_events=[],
            player_stats={},
            team_utility_damage={2: 0, 3: 0},
            team_utility_efficiency={2: 0.0, 3: 0.0},
        )

    def _extract_grenade_damage(
        self,
        damages_df: pd.DataFrame,
        att_col: str,
        vic_col: str,
        dmg_col: str | None,
        weapon_col: str,
        tick_col: str | None,
    ) -> None:
        """Extract all grenade damage events."""
        round_col = self._find_col(damages_df, ["total_rounds_played"])

        for _, row in damages_df.iterrows():
            weapon = str(row.get(weapon_col, "")).lower()

            if weapon not in GRENADE_WEAPONS:
                continue

            grenade_type = GRENADE_WEAPONS[weapon]
            thrower_id = safe_int(row[att_col])
            victim_id = safe_int(row[vic_col])

            if thrower_id == 0:
                continue

            # Check if team damage
            thrower_team = self.data.player_teams.get(thrower_id, 0)
            victim_team = self.data.player_teams.get(victim_id, 0)
            is_team_damage = thrower_team == victim_team and thrower_id != victim_id

            event = GrenadeDamageEvent(
                tick=safe_int(row.get(tick_col, 0)) if tick_col else 0,
                round_num=safe_int(row.get(round_col, 0)) if round_col else 0,
                thrower_id=thrower_id,
                thrower_name=self.data.player_names.get(thrower_id, "Unknown"),
                victim_id=victim_id,
                victim_name=self.data.player_names.get(victim_id, "Unknown"),
                grenade_type=grenade_type,
                damage=safe_int(row.get(dmg_col, 0)) if dmg_col else 0,
                is_team_damage=is_team_damage,
            )
            self._grenade_events.append(event)

    def _build_player_stats(self) -> dict[int, PlayerUtilityStats]:
        """Build per-player utility statistics."""
        stats: dict[int, PlayerUtilityStats] = {}
        num_rounds = max(self.data.num_rounds, 1)

        for steam_id, name in self.data.player_names.items():
            player_events = [e for e in self._grenade_events if e.thrower_id == steam_id]

            # HE grenade stats
            he_events = [e for e in player_events if e.grenade_type == GrenadeType.HE_GRENADE]
            he_damage = sum(e.damage for e in he_events if not e.is_team_damage)
            # Estimate grenades thrown by unique (round, victim) combinations
            he_instances = len({(e.round_num, e.tick) for e in he_events})
            he_thrown = max(he_instances, 1) if he_events else 0

            # Count HE kills from kills_df
            kills_df = self.data.kills_df
            weapon_col = self._find_col(kills_df, ["weapon"]) if not kills_df.empty else None
            att_col = (
                self._find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
                if not kills_df.empty
                else None
            )
            he_kills = 0
            if weapon_col and att_col and not kills_df.empty:
                player_kills = kills_df[kills_df[att_col] == steam_id]
                he_kills = len(player_kills[player_kills[weapon_col].str.lower() == "hegrenade"])

            # Molotov stats
            molotov_events = [
                e
                for e in player_events
                if e.grenade_type in [GrenadeType.MOLOTOV, GrenadeType.INCENDIARY]
            ]
            molotov_damage = sum(e.damage for e in molotov_events if not e.is_team_damage)
            molotov_ticks = len(molotov_events)
            molotov_thrown = len({e.round_num for e in molotov_events}) if molotov_events else 0

            # Total utility damage
            total_damage = sum(e.damage for e in player_events if not e.is_team_damage)

            # Team damage
            team_events = [e for e in player_events if e.is_team_damage]
            team_damage = sum(e.damage for e in team_events)

            # Estimate utility cost
            utility_cost = (
                he_thrown * GRENADE_COSTS[GrenadeType.HE_GRENADE]
                + molotov_thrown * GRENADE_COSTS[GrenadeType.MOLOTOV]
            )

            stats[steam_id] = PlayerUtilityStats(
                steam_id=steam_id,
                name=name,
                he_grenades_thrown=he_thrown,
                he_damage_total=he_damage,
                he_damage_avg=he_damage / he_thrown if he_thrown > 0 else 0.0,
                he_kills=he_kills,
                molotov_thrown=molotov_thrown,
                molotov_damage_total=molotov_damage,
                molotov_ticks=molotov_ticks,
                flashes_thrown=0,  # Would need flash events
                smokes_thrown=0,  # Would need smoke events
                utility_damage_per_round=total_damage / num_rounds,
                utility_cost_total=utility_cost,
                damage_per_dollar=total_damage / utility_cost if utility_cost > 0 else 0.0,
                team_damage_total=team_damage,
                team_damage_instances=len(team_events),
            )

        return stats

    def _calculate_team_utility_damage(self) -> dict[int, int]:
        """Calculate total utility damage per team."""
        team_damage: dict[int, int] = {2: 0, 3: 0}

        for event in self._grenade_events:
            if event.is_team_damage:
                continue

            thrower_team = self.data.player_teams.get(event.thrower_id, 0)
            if thrower_team in team_damage:
                team_damage[thrower_team] += event.damage

        return team_damage

    def _calculate_team_efficiency(self) -> dict[int, float]:
        """Calculate utility damage per dollar spent per team."""
        team_damage = self._calculate_team_utility_damage()
        team_cost: dict[int, int] = {2: 0, 3: 0}

        # Estimate cost per team
        for steam_id in self.data.player_names:
            team = self.data.player_teams.get(steam_id, 0)
            if team not in team_cost:
                continue

            player_events = [e for e in self._grenade_events if e.thrower_id == steam_id]
            he_count = len(
                {
                    (e.round_num, e.tick)
                    for e in player_events
                    if e.grenade_type == GrenadeType.HE_GRENADE
                }
            )
            molotov_count = len(
                {
                    e.round_num
                    for e in player_events
                    if e.grenade_type in [GrenadeType.MOLOTOV, GrenadeType.INCENDIARY]
                }
            )

            team_cost[team] += (
                he_count * GRENADE_COSTS[GrenadeType.HE_GRENADE]
                + molotov_count * GRENADE_COSTS[GrenadeType.MOLOTOV]
            )

        return {
            team: (team_damage[team] / team_cost[team] if team_cost[team] > 0 else 0.0)
            for team in [2, 3]
        }


def analyze_utility(demo_data: DemoData) -> UtilityAnalysisResult:
    """
    Convenience function to analyze utility from demo data.

    Args:
        demo_data: Parsed demo data to analyze.

    Returns:
        UtilityAnalysisResult containing all utility metrics.
    """
    analyzer = UtilityAnalyzer(demo_data)
    return analyzer.analyze()
