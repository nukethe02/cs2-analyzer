"""
Economy Analysis Module for CS2 Demo Analysis

Implements economy tracking and buy round classification:
- Equipment value calculation per player per round
- Buy round classification (eco, force, full buy)
- Team economy state tracking
- Economic efficiency metrics (damage per dollar, etc.)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from opensight.core.parser import DemoData, safe_int

logger = logging.getLogger(__name__)


class BuyType(Enum):
    """Classification of buy round types."""

    PISTOL = "pistol"  # First round or after half
    ECO = "eco"  # $0-1500 team spend
    FORCE = "force"  # $1500-3500 team spend (or partial buy)
    HALF_BUY = "half_buy"  # $3500-4500 team spend
    FULL_BUY = "full"  # $4500+ team spend (rifles/AWPs + util)
    UNKNOWN = "unknown"


# Equipment costs (CS2 values as of 2024)
WEAPON_COSTS = {
    # Pistols
    "glock": 0,
    "usp_silencer": 0,
    "hkp2000": 0,
    "p250": 300,
    "tec9": 500,
    "cz75a": 500,
    "fiveseven": 500,
    "elite": 400,
    "deagle": 700,
    "revolver": 600,
    # SMGs
    "mac10": 1050,
    "mp9": 1250,
    "mp7": 1500,
    "mp5sd": 1500,
    "ump45": 1200,
    "p90": 2350,
    "bizon": 1400,
    # Rifles
    "famas": 2050,
    "galilar": 1800,
    "m4a1": 3100,
    "m4a1_silencer": 2900,
    "ak47": 2700,
    "sg556": 3000,
    "aug": 3300,
    "ssg08": 1700,
    "awp": 4750,
    "g3sg1": 5000,
    "scar20": 5000,
    # Heavy
    "nova": 1050,
    "xm1014": 2000,
    "sawedoff": 1100,
    "mag7": 1300,
    "m249": 5200,
    "negev": 1700,
    # Equipment
    "vest": 650,
    "vesthelm": 1000,
    "defuser": 400,
    "taser": 200,
    # Grenades
    "flashbang": 200,
    "hegrenade": 300,
    "smokegrenade": 300,
    "molotov": 400,
    "incgrenade": 600,
    "decoy": 50,
}

# Team equipment thresholds (for 5-player team total)
ECO_THRESHOLD = 1500  # Below this = eco
FORCE_THRESHOLD = 3500  # Below this but above eco = force buy
HALF_BUY_THRESHOLD = 4500  # Below this but above force = half buy
# Above HALF_BUY_THRESHOLD = full buy

# Per-player thresholds
PLAYER_ECO_THRESHOLD = 300
PLAYER_FORCE_THRESHOLD = 700
PLAYER_FULL_THRESHOLD = 2500


@dataclass
class PlayerRoundEconomy:
    """Economy data for a single player in a single round."""

    steam_id: int
    round_num: int
    equipment_value: int
    start_money: int
    end_money: int
    spent: int
    weapon: str
    has_armor: bool
    has_helmet: bool
    has_defuser: bool
    grenade_count: int
    buy_type: BuyType


@dataclass
class TeamRoundEconomy:
    """Economy data for a team in a single round."""

    round_num: int
    team: int  # 2=T, 3=CT
    total_equipment: int
    avg_equipment: int
    total_money: int
    total_spent: int
    buy_type: BuyType
    player_economies: list[PlayerRoundEconomy] = field(default_factory=list)


@dataclass
class EconomyStats:
    """Comprehensive economy statistics for a match."""

    rounds_analyzed: int
    team_economies: dict[int, list[TeamRoundEconomy]]  # team -> rounds
    player_economies: dict[int, list[PlayerRoundEconomy]]  # steam_id -> rounds

    # Aggregated stats
    eco_round_win_rate: dict[int, float]  # team -> win rate on eco rounds
    force_buy_win_rate: dict[int, float]  # team -> win rate on force rounds
    full_buy_win_rate: dict[int, float]  # team -> win rate on full buy rounds
    avg_equipment_value: dict[int, float]  # team -> avg equipment
    damage_per_dollar: dict[int, float]  # player steam_id -> efficiency


@dataclass
class PlayerEconomyProfile:
    """Economy profile for a single player."""

    steam_id: int
    name: str

    # Round type counts
    eco_rounds: int
    force_rounds: int
    full_buy_rounds: int

    # Average values
    avg_equipment_value: float
    avg_spent_per_round: float

    # Efficiency
    damage_per_dollar: float
    kills_per_dollar: float

    # Utility usage
    avg_grenades_per_round: float
    utility_spend_ratio: float  # % of total spent on utility


def classify_buy_type(equipment_value: int, is_pistol_round: bool = False) -> BuyType:
    """
    Classify a buy round type based on equipment value.

    Args:
        equipment_value: Total equipment value for a player.
        is_pistol_round: Whether this is a pistol round.

    Returns:
        BuyType classification.
    """
    if is_pistol_round:
        return BuyType.PISTOL

    if equipment_value <= PLAYER_ECO_THRESHOLD:
        return BuyType.ECO
    elif equipment_value <= PLAYER_FORCE_THRESHOLD:
        return BuyType.FORCE
    elif equipment_value <= PLAYER_FULL_THRESHOLD:
        return BuyType.HALF_BUY
    else:
        return BuyType.FULL_BUY


def classify_team_buy(total_equipment: int, is_pistol_round: bool = False) -> BuyType:
    """
    Classify team buy type based on total equipment value.

    Args:
        total_equipment: Total equipment value for the team.
        is_pistol_round: Whether this is a pistol round.

    Returns:
        BuyType classification for the team.
    """
    if is_pistol_round:
        return BuyType.PISTOL

    if total_equipment <= ECO_THRESHOLD:
        return BuyType.ECO
    elif total_equipment <= FORCE_THRESHOLD:
        return BuyType.FORCE
    elif total_equipment <= HALF_BUY_THRESHOLD:
        return BuyType.HALF_BUY
    else:
        return BuyType.FULL_BUY


# Weapon name variations for better matching
WEAPON_NAME_ALIASES = {
    # M4 variations
    "m4a4": "m4a1",
    "m4a1-s": "m4a1_silencer",
    "m4a1s": "m4a1_silencer",
    "m4a1silencer": "m4a1_silencer",
    # AK variations
    "ak-47": "ak47",
    # Pistol variations
    "p2000": "hkp2000",
    "usps": "usp_silencer",
    "usp-s": "usp_silencer",
    "usp_s": "usp_silencer",
    "cz-75": "cz75a",
    "dual_berettas": "elite",
    "dualies": "elite",
    "r8": "revolver",
    # SMG variations
    "mp5": "mp5sd",
    "ppbizon": "bizon",
    # AWP/Scout
    "scout": "ssg08",
    "ssg08_silencer": "ssg08",
    # Shotguns
    "sawedoff": "sawedoff",
    "sawed-off": "sawedoff",
    # Grenades
    "flash": "flashbang",
    "he": "hegrenade",
    "smoke": "smokegrenade",
    "molly": "molotov",
    "inc": "incgrenade",
}

# Cache for weapon cost lookups
_weapon_cost_cache: dict[str, int] = {}


def estimate_weapon_cost(weapon_name: str) -> int:
    """
    Estimate the cost of a weapon by name with improved matching.

    Args:
        weapon_name: The weapon name (e.g., 'ak47', 'm4a1_silencer').

    Returns:
        Estimated cost in dollars (0-10000 range).
    """
    if not weapon_name:
        return 0

    # Normalize weapon name
    weapon = weapon_name.lower().replace(" ", "_").replace("-", "_")

    # Check cache first
    if weapon in _weapon_cost_cache:
        return _weapon_cost_cache[weapon]

    cost = 0

    # Direct lookup
    if weapon in WEAPON_COSTS:
        cost = WEAPON_COSTS[weapon]
    # Check aliases
    elif weapon in WEAPON_NAME_ALIASES:
        aliased = WEAPON_NAME_ALIASES[weapon]
        cost = WEAPON_COSTS.get(aliased, 0)
    else:
        # Try partial match
        for known_weapon, wcost in WEAPON_COSTS.items():
            if known_weapon in weapon or weapon in known_weapon:
                cost = wcost
                break

    # Validate cost is in reasonable range (0-10000)
    cost = max(0, min(cost, 10000))

    # Cache the result
    _weapon_cost_cache[weapon] = cost

    return cost


class EconomyAnalyzer:
    """Analyzer for computing economy metrics from parsed demo data."""

    def __init__(self, demo_data: DemoData):
        """
        Initialize the economy analyzer.

        Args:
            demo_data: Parsed demo data to analyze.
        """
        self.data = demo_data
        self._player_economies: dict[int, list[PlayerRoundEconomy]] = {}
        self._team_economies: dict[int, list[TeamRoundEconomy]] = {2: [], 3: []}

    def analyze(self) -> EconomyStats:
        """
        Run full economy analysis on the demo data.

        Returns:
            EconomyStats containing all economy metrics.
        """
        logger.info("Starting economy analysis...")

        # Try to get round-by-round data if available
        self._analyze_from_kills()

        # Build statistics
        stats = self._build_stats()

        logger.info(f"Economy analysis complete. {stats.rounds_analyzed} rounds analyzed.")
        return stats

    def _analyze_from_kills(self) -> None:
        """
        Analyze economy from kill events with improved estimation.

        Uses weapon data from kills to estimate equipment values per round.
        Also uses grenade events and damage data for more accurate estimates.
        """
        kills_df = self.data.kills_df
        if kills_df.empty:
            logger.warning("No kill data for economy analysis")
            return

        # Find column names
        def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
            for col in options:
                if col in df.columns:
                    return col
            return None

        att_col = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        weapon_col = find_col(kills_df, ["weapon"])
        round_col = find_col(kills_df, ["total_rounds_played", "round_num", "round"])

        if not att_col or not weapon_col:
            logger.warning("Missing columns for economy analysis")
            return

        # Determine pistol rounds from DemoData.rounds if available
        pistol_rounds = {1, 16}  # Default: first round and after halftime
        if hasattr(self.data, 'rounds') and self.data.rounds:
            pistol_rounds = {1}  # Round 1 is always pistol
            # MR15 format: round 16 is second half pistol
            # MR12 format: round 13 is second half pistol
            num_rounds = self.data.num_rounds
            if num_rounds >= 24:  # MR12
                pistol_rounds.add(13)
            else:  # MR15 or shorter
                pistol_rounds.add(16)

        # Pre-calculate grenade usage per player per round if available
        grenade_counts: dict[tuple[int, int], int] = {}  # (steam_id, round_num) -> count
        if hasattr(self.data, 'grenades') and self.data.grenades:
            for grenade in self.data.grenades:
                if grenade.event_type == 'thrown':
                    key = (grenade.player_steamid, grenade.round_num)
                    grenade_counts[key] = grenade_counts.get(key, 0) + 1

        # Pre-calculate armor info from damage events if available
        player_has_armor: dict[tuple[int, int], bool] = {}  # (steam_id, round_num) -> has_armor
        damages_df = self.data.damages_df
        if not damages_df.empty:
            vic_col = find_col(damages_df, ["user_steamid", "victim_steamid"])
            armor_col = find_col(damages_df, ["armor", "armor_remaining"])
            if vic_col and armor_col:
                for _, row in damages_df.iterrows():
                    vic_id = safe_int(row.get(vic_col))
                    round_num = safe_int(row.get(round_col, 0)) if round_col else 0
                    armor = safe_int(row.get(armor_col, 0))
                    if vic_id and armor > 0:
                        player_has_armor[(vic_id, round_num)] = True

        # Group kills by round and player
        for steam_id in self.data.player_names:
            player_rounds: list[PlayerRoundEconomy] = []
            team = self.data.player_teams.get(steam_id, 0)
            team_str = str(team) if isinstance(team, int) else team

            # Get this player's kills
            player_kills = kills_df[kills_df[att_col] == steam_id]

            if not player_kills.empty and weapon_col:
                # Group by round if available
                if round_col and round_col in player_kills.columns:
                    for round_num in player_kills[round_col].unique():
                        round_num_int = int(round_num)
                        round_kills = player_kills[player_kills[round_col] == round_num]
                        weapons = round_kills[weapon_col].unique()

                        # Estimate equipment value from weapons used
                        max_weapon_cost = max(
                            (estimate_weapon_cost(w) for w in weapons if w),
                            default=0
                        )

                        # Estimate armor from damage data or weapon cost
                        has_armor = player_has_armor.get((steam_id, round_num_int), max_weapon_cost >= 1800)
                        has_helmet = has_armor and max_weapon_cost >= 1500  # More likely helmet with better weapons

                        # Rough estimate: weapon + armor
                        equipment_estimate = max_weapon_cost
                        if has_helmet:
                            equipment_estimate += 1000  # Vest + helmet
                        elif has_armor:
                            equipment_estimate += 650  # Just vest

                        # Add grenade cost estimation
                        grenade_count = grenade_counts.get((steam_id, round_num_int), 0)
                        grenade_cost = min(grenade_count * 300, 1200)  # Estimate ~300 per grenade, max 4
                        equipment_estimate += grenade_cost

                        # Validate equipment value (0-10000 range)
                        equipment_estimate = max(0, min(equipment_estimate, 10000))

                        is_pistol = round_num_int in pistol_rounds
                        buy_type = classify_buy_type(equipment_estimate, is_pistol)

                        player_round = PlayerRoundEconomy(
                            steam_id=steam_id,
                            round_num=round_num_int,
                            equipment_value=equipment_estimate,
                            start_money=0,  # Unknown without full economy data
                            end_money=0,
                            spent=equipment_estimate,
                            weapon=str(weapons[0]) if len(weapons) > 0 else "",
                            has_armor=has_armor,
                            has_helmet=has_helmet,
                            has_defuser=team_str in ("3", "CT"),  # CT has defuser
                            grenade_count=grenade_count,
                            buy_type=buy_type,
                        )
                        player_rounds.append(player_round)
                else:
                    # No round info - create single summary
                    weapons = player_kills[weapon_col].unique()
                    max_weapon_cost = max(
                        (estimate_weapon_cost(w) for w in weapons if w),
                        default=0
                    )

                    equipment_estimate = max_weapon_cost
                    if max_weapon_cost >= 1800:
                        equipment_estimate += 1000

                    # Validate
                    equipment_estimate = max(0, min(equipment_estimate, 10000))

                    player_round = PlayerRoundEconomy(
                        steam_id=steam_id,
                        round_num=0,
                        equipment_value=equipment_estimate,
                        start_money=0,
                        end_money=0,
                        spent=equipment_estimate,
                        weapon=str(weapons[0]) if len(weapons) > 0 else "",
                        has_armor=max_weapon_cost >= 1800,
                        has_helmet=max_weapon_cost >= 1800,
                        has_defuser=team_str in ("3", "CT"),
                        grenade_count=0,
                        buy_type=classify_buy_type(equipment_estimate),
                    )
                    player_rounds.append(player_round)

            self._player_economies[steam_id] = player_rounds

        # Build team economies by aggregating player data
        self._build_team_economies()

    def _build_team_economies(self) -> None:
        """Build team-level economy data from player economies."""
        # Group players by team
        t_players = [sid for sid, team in self.data.player_teams.items() if team == 2]
        ct_players = [sid for sid, team in self.data.player_teams.items() if team == 3]

        # Get all round numbers
        all_rounds = set()
        for player_rounds in self._player_economies.values():
            for pr in player_rounds:
                all_rounds.add(pr.round_num)

        for round_num in sorted(all_rounds):
            for team, players in [(2, t_players), (3, ct_players)]:
                team_equipment = 0
                player_economies = []

                for steam_id in players:
                    if steam_id in self._player_economies:
                        for pr in self._player_economies[steam_id]:
                            if pr.round_num == round_num:
                                team_equipment += pr.equipment_value
                                player_economies.append(pr)
                                break

                if player_economies:
                    is_pistol = round_num in [1, 16]
                    team_round = TeamRoundEconomy(
                        round_num=round_num,
                        team=team,
                        total_equipment=team_equipment,
                        avg_equipment=team_equipment // len(player_economies),
                        total_money=0,  # Unknown
                        total_spent=team_equipment,
                        buy_type=classify_team_buy(team_equipment, is_pistol),
                        player_economies=player_economies,
                    )
                    self._team_economies[team].append(team_round)

    def _build_stats(self) -> EconomyStats:
        """Build final statistics from analyzed data."""
        # Calculate efficiency metrics
        damage_per_dollar: dict[int, float] = {}

        # Get damage data
        damages_df = self.data.damages_df
        if damages_df is not None and not damages_df.empty:

            def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
                for col in options:
                    if col in df.columns:
                        return col
                return None

            dmg_att_col = find_col(damages_df, ["attacker_steamid", "attacker_steam_id"])
            dmg_col = find_col(damages_df, ["dmg_health", "damage", "dmg"])

            if dmg_att_col and dmg_col:
                for steam_id in self._player_economies:
                    player_damage = damages_df[damages_df[dmg_att_col] == steam_id]
                    total_damage = (
                        safe_int(player_damage[dmg_col].sum()) if not player_damage.empty else 0
                    )

                    total_spent = sum(pr.spent for pr in self._player_economies.get(steam_id, []))

                    if total_spent > 0:
                        damage_per_dollar[steam_id] = total_damage / total_spent

        return EconomyStats(
            rounds_analyzed=self.data.num_rounds,
            team_economies=self._team_economies,
            player_economies=self._player_economies,
            eco_round_win_rate={2: 0.0, 3: 0.0},  # Would need round win data
            force_buy_win_rate={2: 0.0, 3: 0.0},
            full_buy_win_rate={2: 0.0, 3: 0.0},
            avg_equipment_value={
                2: sum(tr.avg_equipment for tr in self._team_economies[2])
                / max(len(self._team_economies[2]), 1),
                3: sum(tr.avg_equipment for tr in self._team_economies[3])
                / max(len(self._team_economies[3]), 1),
            },
            damage_per_dollar=damage_per_dollar,
        )

    def get_player_profile(self, steam_id: int) -> PlayerEconomyProfile | None:
        """
        Get economy profile for a specific player.

        Args:
            steam_id: The player's Steam ID.

        Returns:
            PlayerEconomyProfile or None if player not found.
        """
        if steam_id not in self._player_economies:
            return None

        player_rounds = self._player_economies[steam_id]
        if not player_rounds:
            return None

        # Count round types
        eco_rounds = sum(1 for pr in player_rounds if pr.buy_type == BuyType.ECO)
        force_rounds = sum(
            1 for pr in player_rounds if pr.buy_type in [BuyType.FORCE, BuyType.HALF_BUY]
        )
        full_buy_rounds = sum(1 for pr in player_rounds if pr.buy_type == BuyType.FULL_BUY)

        # Calculate averages
        avg_equipment = sum(pr.equipment_value for pr in player_rounds) / len(player_rounds)
        avg_spent = sum(pr.spent for pr in player_rounds) / len(player_rounds)
        avg_grenades = sum(pr.grenade_count for pr in player_rounds) / len(player_rounds)

        # Get efficiency metrics
        total_spent = sum(pr.spent for pr in player_rounds)

        # Calculate damage per dollar
        damage_per_dollar = 0.0
        kills_per_dollar = 0.0

        damages_df = self.data.damages_df
        if damages_df is not None and not damages_df.empty:

            def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
                for col in options:
                    if col in df.columns:
                        return col
                return None

            dmg_att_col = find_col(damages_df, ["attacker_steamid", "attacker_steam_id"])
            dmg_col = find_col(damages_df, ["dmg_health", "damage", "dmg"])

            if dmg_att_col and dmg_col:
                player_damage = damages_df[damages_df[dmg_att_col] == steam_id]
                total_damage = (
                    safe_int(player_damage[dmg_col].sum()) if not player_damage.empty else 0
                )

                if total_spent > 0:
                    damage_per_dollar = total_damage / total_spent

        # Get kill count
        kills_df = self.data.kills_df
        if not kills_df.empty and total_spent > 0:

            def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
                for col in options:
                    if col in df.columns:
                        return col
                return None

            att_col = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
            if att_col:
                kills = len(kills_df[kills_df[att_col] == steam_id])
                kills_per_dollar = kills / total_spent

        return PlayerEconomyProfile(
            steam_id=steam_id,
            name=self.data.player_names.get(steam_id, "Unknown"),
            eco_rounds=eco_rounds,
            force_rounds=force_rounds,
            full_buy_rounds=full_buy_rounds,
            avg_equipment_value=avg_equipment,
            avg_spent_per_round=avg_spent,
            damage_per_dollar=damage_per_dollar,
            kills_per_dollar=kills_per_dollar,
            avg_grenades_per_round=avg_grenades,
            utility_spend_ratio=0.0,  # Would need detailed buy data
        )


def analyze_economy(demo_data: DemoData) -> EconomyStats:
    """
    Convenience function to analyze economy from demo data.

    Args:
        demo_data: Parsed demo data to analyze.

    Returns:
        EconomyStats containing all economy metrics.
    """
    analyzer = EconomyAnalyzer(demo_data)
    return analyzer.analyze()
