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

# CS2 Loss Bonus System
# After a loss, teams receive increasing bonus money
# Formula: BASE_LOSS_BONUS + (consecutive_losses * LOSS_BONUS_INCREMENT)
BASE_LOSS_BONUS = 1400  # First loss bonus
LOSS_BONUS_INCREMENT = 500  # Additional per consecutive loss
MAX_LOSS_BONUS = 3400  # Cap at 5 consecutive losses (1400 + 4*500)
MAX_CONSECUTIVE_LOSSES = 5  # Bonus caps at 5 losses

# Loss bonus thresholds for buy decisions
LOW_LOSS_BONUS = 1900  # 1-2 consecutive losses - risky to force
HIGH_LOSS_BONUS = 2900  # 4+ consecutive losses - safer to force

# Team equipment thresholds for total team value
TEAM_FORCE_MIN = 12000  # Below this with 5 players isn't really a force
TEAM_FORCE_MAX = 20000  # Above this is closer to full buy


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

    # Loss bonus tracking
    loss_bonus: int = BASE_LOSS_BONUS  # Current loss bonus ($1400-$3400)
    consecutive_losses: int = 0  # Number of consecutive losses (0-5)

    # Buy decision quality
    is_bad_force: bool = False  # Force buy with low loss bonus
    is_good_force: bool = False  # Force buy with high loss bonus or broken enemy
    round_won: bool = False  # Did we win this round?

    # Per-round Economy IQ grading
    decision_flag: str = "ok"  # "ok", "bad_force", "full_save_error"
    decision_grade: str = "B"  # A-F per round
    loss_bonus_next: int = BASE_LOSS_BONUS  # Loss bonus if this round is lost


@dataclass
class BuyDecision:
    """Result of analyzing a team's buy decision for a single round."""

    grade: str  # A-F grade
    flag: str  # "ok", "bad_force", "bad_eco"
    reason: str
    loss_bonus: int
    spend_ratio: float


@dataclass
class EconomyTracker:
    """
    Stateful tracker for loss bonus across rounds.

    Tracks CT and T loss counters independently and provides
    methods to update state and analyze buy decisions.

    Usage:
        tracker = EconomyTracker()
        for round_info in rounds:
            ct_bonus = tracker.get_loss_bonus("CT")
            t_bonus = tracker.get_loss_bonus("T")
            # ... analyze buys ...
            tracker.process_round_result(round_info.winner)
    """

    ct_loss_counter: int = 0  # 0-4, capped at MAX_CONSECUTIVE_LOSSES
    t_loss_counter: int = 0  # 0-4, capped at MAX_CONSECUTIVE_LOSSES

    def process_round_result(self, winner: str) -> None:
        """
        Update loss counters after a round completes.

        Win = Reset to 0
        Loss = Increment (max 4)

        Args:
            winner: "CT" or "T"
        """
        if winner == "CT":
            self.ct_loss_counter = 0
            self.t_loss_counter = min(self.t_loss_counter + 1, MAX_CONSECUTIVE_LOSSES - 1)
        elif winner == "T":
            self.t_loss_counter = 0
            self.ct_loss_counter = min(self.ct_loss_counter + 1, MAX_CONSECUTIVE_LOSSES - 1)

    def get_loss_bonus(self, team: str) -> int:
        """
        Get current loss bonus for a team.

        Args:
            team: "CT" or "T"

        Returns:
            Loss bonus in dollars ($1400-$3400)
        """
        counter = self.ct_loss_counter if team == "CT" else self.t_loss_counter
        return calculate_loss_bonus(counter)

    def reset_half(self) -> None:
        """Reset both counters at half-time, match start, or OT start."""
        self.ct_loss_counter = 0
        self.t_loss_counter = 0


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

    # Buy decision quality stats
    bad_buy_count: dict[int, int] = field(default_factory=lambda: {2: 0, 3: 0})
    good_buy_count: dict[int, int] = field(default_factory=lambda: {2: 0, 3: 0})
    total_force_buys: dict[int, int] = field(default_factory=lambda: {2: 0, 3: 0})

    # Economy Grade (A-F)
    economy_grade: dict[int, str] = field(default_factory=lambda: {2: "C", 3: "C"})
    economy_grade_reason: dict[int, str] = field(default_factory=lambda: {2: "", 3: ""})


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


def calculate_loss_bonus(consecutive_losses: int) -> int:
    """
    Calculate loss bonus based on consecutive losses.

    CS2 Loss Bonus System (MR12):
    - 0 losses (just won): $1400
    - 1 loss: $1900
    - 2 losses: $2400
    - 3 losses: $2900
    - 4+ losses: $3400 (capped)

    Args:
        consecutive_losses: Number of consecutive round losses (0-4)

    Returns:
        Loss bonus amount in dollars ($1400-$3400)
    """
    # Clamp to max (4 losses = max bonus)
    losses = min(consecutive_losses, 4)

    # Formula: $1400 + (losses * $500)
    # 0 losses = $1400, 1 loss = $1900, 2 losses = $2400, etc.
    bonus = BASE_LOSS_BONUS + (losses * LOSS_BONUS_INCREMENT)
    return min(bonus, MAX_LOSS_BONUS)


def is_bad_force(buy_type: BuyType, loss_bonus: int) -> bool:
    """
    Determine if a force buy was a bad decision.

    Bad Force Criteria:
    - Team is force buying (FORCE or HALF_BUY type)
    - Loss bonus is low ($1400-$1900, meaning 1-2 consecutive losses)
    - Risk: If you lose, you'll have to double-eco

    Args:
        buy_type: The team's buy classification
        loss_bonus: Current loss bonus for the team

    Returns:
        True if this is a risky/bad force buy
    """
    if buy_type not in (BuyType.FORCE, BuyType.HALF_BUY):
        return False

    # Low loss bonus = bad time to force
    return loss_bonus <= LOW_LOSS_BONUS


def is_good_force(buy_type: BuyType, loss_bonus: int, enemy_economy_broken: bool = False) -> bool:
    """
    Determine if a force buy was a justified decision.

    Good Force Criteria:
    - Team is force buying
    - Loss bonus is high ($2900+, meaning 4+ consecutive losses) - max money anyway
    - OR enemy economy is broken (post-loss after their win streak)

    Args:
        buy_type: The team's buy classification
        loss_bonus: Current loss bonus for the team
        enemy_economy_broken: Whether enemy team just lost and has low money

    Returns:
        True if this force buy is justified
    """
    if buy_type not in (BuyType.FORCE, BuyType.HALF_BUY):
        return False

    # High loss bonus = might as well force (getting max money anyway)
    if loss_bonus >= HIGH_LOSS_BONUS:
        return True

    # Enemy is broke = good time to force
    if enemy_economy_broken:
        return True

    return False


# Full Save Error thresholds (used in analyze_round_buy and is_full_save_error)
FULL_SAVE_BANK_THRESHOLD = 10000  # Team has >$10k combined
FULL_SAVE_SPEND_RATIO = 0.10  # Spent <10% of available
FULL_SAVE_EQUIPMENT_MIN = 3000  # Equipment below this = saving


def analyze_round_buy(
    team_spend: int,
    team_bank: int,
    loss_bonus: int,
    is_pistol_round: bool = False,
) -> BuyDecision:
    """
    Judge a team's buy decision for a single round.

    Flags:
    - BAD_FORCE: Loss Bonus < $1900 AND Spend > 60% (risking reset)
    - BAD_ECO: Bank > $10k AND Spend < 10% (hoarding when rich)

    Args:
        team_spend: Total team equipment value / spending
        team_bank: Total team money at round start
        loss_bonus: Current loss bonus for the team
        is_pistol_round: Whether this is a pistol round

    Returns:
        BuyDecision with grade, flag, and reason
    """
    if is_pistol_round:
        return BuyDecision(
            grade="B",
            flag="ok",
            reason="Pistol round",
            loss_bonus=loss_bonus,
            spend_ratio=1.0,
        )

    spend_ratio = team_spend / max(team_bank, 1)

    # Check BAD_FORCE: Risky force with low loss bonus
    # Loss Bonus < $1900 means 0-1 consecutive losses - bad time to force
    if loss_bonus < LOW_LOSS_BONUS and spend_ratio > 0.60:
        return BuyDecision(
            grade="D",
            flag="bad_force",
            reason=f"Risky force at ${loss_bonus} bonus (spend {spend_ratio:.0%})",
            loss_bonus=loss_bonus,
            spend_ratio=spend_ratio,
        )

    # Check BAD_ECO: Hoarding money when rich
    if team_bank > FULL_SAVE_BANK_THRESHOLD and spend_ratio < FULL_SAVE_SPEND_RATIO:
        return BuyDecision(
            grade="D",
            flag="bad_eco",
            reason=f"Saving with ${team_bank:,} bank",
            loss_bonus=loss_bonus,
            spend_ratio=spend_ratio,
        )

    # Good decisions
    if spend_ratio > 0.80:
        grade = "A"
        reason = "Full buy"
    elif spend_ratio < 0.20:
        grade = "B" if loss_bonus < 2400 else "C"
        reason = "Eco round"
    elif loss_bonus >= HIGH_LOSS_BONUS:
        grade = "A"
        reason = f"Justified force at ${loss_bonus} bonus"
    else:
        grade = "B"
        reason = "Normal buy"

    return BuyDecision(
        grade=grade,
        flag="ok",
        reason=reason,
        loss_bonus=loss_bonus,
        spend_ratio=spend_ratio,
    )


def is_full_save_error(
    team_money: int,
    total_spent: int,
    total_equipment: int,
    is_pistol_round: bool = False,
) -> bool:
    """
    Detect unnecessary saving when team is rich.

    Full Save Error Criteria:
    - Team bank > $10,000 (rich enough to buy)
    - Spend ratio < 10% of available funds
    - Equipment < $3,000 (didn't buy meaningful items)
    - Not a pistol round

    This catches teams that save when they could force or full buy,
    which is a strategic mistake (wasting economic advantage).

    Args:
        team_money: Total team money at start of round
        total_spent: Amount team spent this round
        total_equipment: Total equipment value for team
        is_pistol_round: Whether this is a pistol round

    Returns:
        True if team saved unnecessarily when rich
    """
    if is_pistol_round:
        return False

    # Not rich enough to be an error
    if team_money < FULL_SAVE_BANK_THRESHOLD:
        return False

    # They did buy something meaningful
    if total_equipment >= FULL_SAVE_EQUIPMENT_MIN:
        return False

    # Check spend ratio
    spend_ratio = total_spent / max(team_money, 1)
    return spend_ratio < FULL_SAVE_SPEND_RATIO


def calculate_round_economy_grade(
    buy_type: BuyType,
    loss_bonus: int,
    is_bad_force_flag: bool,
    is_full_save_flag: bool,
    round_won: bool,
) -> tuple[str, str]:
    """
    Calculate per-round Economy Grade (A-F) based on the buy decision.

    Grading considers:
    - Was the buy appropriate for the loss bonus state?
    - Did the decision lead to a win?
    - Was it a clear mistake (bad force, unnecessary save)?

    Args:
        buy_type: Team's buy classification
        loss_bonus: Current loss bonus for the team
        is_bad_force_flag: Whether this was flagged as bad force
        is_full_save_flag: Whether this was flagged as full save error
        round_won: Whether the team won this round

    Returns:
        Tuple of (grade, reason)
    """
    # Clear mistakes get D/F
    if is_full_save_flag:
        grade = "D" if round_won else "F"
        return (grade, "Saved with $10k+ bank")

    if is_bad_force_flag:
        if round_won:
            return ("C", "Risky force paid off")
        else:
            return ("D", f"Bad force at ${loss_bonus} loss bonus")

    # Proper full buys
    if buy_type == BuyType.FULL_BUY:
        return ("A" if round_won else "B", "Full buy")

    # Pistol rounds are neutral
    if buy_type == BuyType.PISTOL:
        return ("B", "Pistol round")

    # Smart ecos
    if buy_type == BuyType.ECO:
        if round_won:
            return ("A", "Won eco round!")
        else:
            return ("B", "Proper eco")

    # Force buys with high loss bonus are fine
    if buy_type in (BuyType.FORCE, BuyType.HALF_BUY):
        if loss_bonus >= HIGH_LOSS_BONUS:
            return ("A" if round_won else "B", f"Justified force at ${loss_bonus}")
        else:
            return ("B" if round_won else "C", "Force buy")

    return ("C", "")


def calculate_economy_grade(
    force_win_rate: float,
    bad_buy_count: int,
    total_force_buys: int,
) -> tuple[str, str]:
    """
    Calculate Economy Grade (A-F) based on buying decisions.

    Grading Criteria:
    - A: High force win % (>40%) AND low bad buys (≤1)
    - B: Decent force win % (>30%) AND few bad buys (≤2)
    - C: Average performance
    - D: Low force win % (<25%) OR many bad buys (>2)
    - F: Very low force win % (<20%) AND many bad buys (>3)

    Args:
        force_win_rate: Win rate when force buying (0.0-1.0)
        bad_buy_count: Number of bad force buys
        total_force_buys: Total number of force buy rounds

    Returns:
        Tuple of (grade, reason)
    """
    # Convert to percentage for readability
    win_pct = force_win_rate * 100

    # Not enough data
    if total_force_buys < 2:
        return ("C", "Not enough force buy rounds to grade")

    # Grade F: Terrible economy management
    if win_pct < 20 and bad_buy_count > 3:
        return ("F", f"Poor force win rate ({win_pct:.0f}%) with {bad_buy_count} bad force buys")

    # Grade D: Below average
    if win_pct < 25 or bad_buy_count > 2:
        reasons = []
        if win_pct < 25:
            reasons.append(f"low force win rate ({win_pct:.0f}%)")
        if bad_buy_count > 2:
            reasons.append(f"{bad_buy_count} bad force buys")
        return ("D", " and ".join(reasons).capitalize())

    # Grade A: Excellent economy management
    if win_pct > 40 and bad_buy_count <= 1:
        return ("A", f"Strong force win rate ({win_pct:.0f}%) with disciplined buying")

    # Grade B: Good economy management
    if win_pct > 30 and bad_buy_count <= 2:
        return (
            "B",
            f"Solid force performance ({win_pct:.0f}%) with {bad_buy_count} questionable buys",
        )

    # Grade C: Average
    return ("C", f"Average economy ({win_pct:.0f}% force win rate)")


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

        # Determine pistol rounds using proper detection
        # Import is_pistol_round for proper OT handling
        try:
            from opensight.core.parser import is_pistol_round
        except ImportError:
            # Fallback if import fails
            def is_pistol_round(rn: int, rph: int = 12) -> bool:
                return rn == 1 or rn == rph + 1

        # Detect MR format: MR12 has max 24 regulation rounds, MR15 has max 30
        num_rounds = self.data.num_rounds
        rounds_per_half = 12 if num_rounds <= 30 else 15

        # Build set of pistol rounds (needed for set membership check)
        pistol_rounds = set()
        for rn in range(1, num_rounds + 10):  # Include potential OT rounds
            if is_pistol_round(rn, rounds_per_half):
                pistol_rounds.add(rn)

        # Pre-calculate grenade usage per player per round if available
        grenade_counts: dict[tuple[int, int], int] = {}  # (steam_id, round_num) -> count
        if hasattr(self.data, "grenades") and self.data.grenades:
            for grenade in self.data.grenades:
                if grenade.event_type == "thrown":
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
                            (estimate_weapon_cost(w) for w in weapons if w), default=0
                        )

                        # Estimate armor from damage data or weapon cost
                        has_armor = player_has_armor.get(
                            (steam_id, round_num_int), max_weapon_cost >= 1800
                        )
                        has_helmet = (
                            has_armor and max_weapon_cost >= 1500
                        )  # More likely helmet with better weapons

                        # Rough estimate: weapon + armor
                        equipment_estimate = max_weapon_cost
                        if has_helmet:
                            equipment_estimate += 1000  # Vest + helmet
                        elif has_armor:
                            equipment_estimate += 650  # Just vest

                        # Add grenade cost estimation
                        grenade_count = grenade_counts.get((steam_id, round_num_int), 0)
                        grenade_cost = min(
                            grenade_count * 300, 1200
                        )  # Estimate ~300 per grenade, max 4
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
                        (estimate_weapon_cost(w) for w in weapons if w), default=0
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
        # Import pistol round detection
        try:
            from opensight.core.parser import is_pistol_round
        except ImportError:

            def is_pistol_round(rn: int, rph: int = 12) -> bool:
                return rn == 1 or rn == rph + 1

        # Detect MR format
        num_rounds = self.data.num_rounds
        rounds_per_half = 12 if num_rounds <= 30 else 15

        # Group players by team
        t_players = [sid for sid, team in self.data.player_teams.items() if team == 2]
        ct_players = [sid for sid, team in self.data.player_teams.items() if team == 3]

        # Get all round numbers
        all_rounds = set()
        for player_rounds in self._player_economies.values():
            for pr in player_rounds:
                all_rounds.add(pr.round_num)

        # Build round winner lookup from DemoData.rounds
        round_winners: dict[int, str] = {}
        if hasattr(self.data, "rounds") and self.data.rounds:
            for round_info in self.data.rounds:
                round_winners[round_info.round_num] = round_info.winner

        # Track consecutive losses for loss bonus calculation
        ct_consecutive_losses = 0
        t_consecutive_losses = 0

        for round_num in sorted(all_rounds):
            # Get round winner for this round
            winner = round_winners.get(round_num, "Unknown")

            # Determine if each team won this round
            ct_won = winner == "CT"
            t_won = winner == "T"

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
                    is_pistol = is_pistol_round(round_num, rounds_per_half)
                    buy_type = classify_team_buy(team_equipment, is_pistol)

                    # Get loss bonus for this team at start of round
                    if team == 2:  # T side
                        consecutive_losses = t_consecutive_losses
                        team_won = t_won
                    else:  # CT side
                        consecutive_losses = ct_consecutive_losses
                        team_won = ct_won

                    loss_bonus = calculate_loss_bonus(consecutive_losses)

                    # Check enemy economy state (is enemy broken?)
                    # Enemy is "broken" if they just lost with high equipment (full buy loss)
                    enemy_broken = False
                    # Simple heuristic: if enemy has 0-1 consecutive losses after we won
                    if team == 2 and ct_consecutive_losses <= 1:
                        enemy_broken = ct_won  # CT just lost after winning
                    elif team == 3 and t_consecutive_losses <= 1:
                        enemy_broken = t_won  # T just lost after winning

                    # Judge buy decision quality
                    bad_force = is_bad_force(buy_type, loss_bonus)
                    good_force = is_good_force(buy_type, loss_bonus, enemy_broken)

                    # Check for full save error (rich team saving unnecessarily)
                    # Estimate team money from equipment (rough approximation)
                    est_team_money = team_equipment + 5000  # Base assumption
                    full_save_err = is_full_save_error(
                        est_team_money, team_equipment, team_equipment, is_pistol
                    )

                    # Determine decision flag
                    if bad_force:
                        decision_flag = "bad_force"
                    elif full_save_err:
                        decision_flag = "full_save_error"
                    else:
                        decision_flag = "ok"

                    # Calculate per-round economy grade
                    round_grade, _reason = calculate_round_economy_grade(
                        buy_type, loss_bonus, bad_force, full_save_err, team_won
                    )

                    # Calculate what loss bonus would be if this round is lost
                    loss_bonus_next = calculate_loss_bonus(
                        min(consecutive_losses + 1, MAX_CONSECUTIVE_LOSSES)
                    )

                    team_round = TeamRoundEconomy(
                        round_num=round_num,
                        team=team,
                        total_equipment=team_equipment,
                        avg_equipment=team_equipment // len(player_economies),
                        total_money=0,  # Unknown
                        total_spent=team_equipment,
                        buy_type=buy_type,
                        player_economies=player_economies,
                        loss_bonus=loss_bonus,
                        consecutive_losses=consecutive_losses,
                        is_bad_force=bad_force,
                        is_good_force=good_force,
                        round_won=team_won,
                        decision_flag=decision_flag,
                        decision_grade=round_grade,
                        loss_bonus_next=loss_bonus_next,
                    )
                    self._team_economies[team].append(team_round)

            # Update consecutive losses AFTER processing the round
            # (loss bonus applies to the CURRENT round, based on PREVIOUS losses)
            if ct_won:
                ct_consecutive_losses = 0
                t_consecutive_losses = min(t_consecutive_losses + 1, MAX_CONSECUTIVE_LOSSES)
            elif t_won:
                t_consecutive_losses = 0
                ct_consecutive_losses = min(ct_consecutive_losses + 1, MAX_CONSECUTIVE_LOSSES)

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

        # Calculate win rates by buy type for each team
        eco_round_win_rate: dict[int, float] = {2: 0.0, 3: 0.0}
        force_buy_win_rate: dict[int, float] = {2: 0.0, 3: 0.0}
        full_buy_win_rate: dict[int, float] = {2: 0.0, 3: 0.0}

        # Buy decision quality stats
        bad_buy_count: dict[int, int] = {2: 0, 3: 0}
        good_buy_count: dict[int, int] = {2: 0, 3: 0}
        total_force_buys: dict[int, int] = {2: 0, 3: 0}

        for team in [2, 3]:
            team_rounds = self._team_economies.get(team, [])

            # Count rounds and wins by buy type
            eco_total, eco_wins = 0, 0
            force_total, force_wins = 0, 0
            full_total, full_wins = 0, 0

            for tr in team_rounds:
                if tr.buy_type == BuyType.ECO:
                    eco_total += 1
                    if tr.round_won:
                        eco_wins += 1
                elif tr.buy_type in (BuyType.FORCE, BuyType.HALF_BUY):
                    force_total += 1
                    if tr.round_won:
                        force_wins += 1
                    # Track bad/good force buys
                    if tr.is_bad_force:
                        bad_buy_count[team] += 1
                    if tr.is_good_force:
                        good_buy_count[team] += 1
                elif tr.buy_type == BuyType.FULL_BUY:
                    full_total += 1
                    if tr.round_won:
                        full_wins += 1

            # Calculate win rates
            eco_round_win_rate[team] = eco_wins / eco_total if eco_total > 0 else 0.0
            force_buy_win_rate[team] = force_wins / force_total if force_total > 0 else 0.0
            full_buy_win_rate[team] = full_wins / full_total if full_total > 0 else 0.0
            total_force_buys[team] = force_total

        # Calculate Economy Grades
        economy_grade: dict[int, str] = {2: "C", 3: "C"}
        economy_grade_reason: dict[int, str] = {2: "", 3: ""}

        for team in [2, 3]:
            grade, reason = calculate_economy_grade(
                force_buy_win_rate[team],
                bad_buy_count[team],
                total_force_buys[team],
            )
            economy_grade[team] = grade
            economy_grade_reason[team] = reason

        return EconomyStats(
            rounds_analyzed=self.data.num_rounds,
            team_economies=self._team_economies,
            player_economies=self._player_economies,
            eco_round_win_rate=eco_round_win_rate,
            force_buy_win_rate=force_buy_win_rate,
            full_buy_win_rate=full_buy_win_rate,
            avg_equipment_value={
                2: sum(tr.avg_equipment for tr in self._team_economies[2])
                / max(len(self._team_economies[2]), 1),
                3: sum(tr.avg_equipment for tr in self._team_economies[3])
                / max(len(self._team_economies[3]), 1),
            },
            damage_per_dollar=damage_per_dollar,
            bad_buy_count=bad_buy_count,
            good_buy_count=good_buy_count,
            total_force_buys=total_force_buys,
            economy_grade=economy_grade,
            economy_grade_reason=economy_grade_reason,
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
