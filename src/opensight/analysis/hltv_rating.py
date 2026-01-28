"""
HLTV Rating 2.0 Calculator

Implements the HLTV 2.0 Rating formula used for professional CS2 statistics.
This module provides standalone functions for calculating HLTV ratings from
raw stats, making it easy to use in various contexts.

The HLTV 2.0 Rating formula:
Rating = 0.0073*KAST + 0.3591*KPR + (-0.5329)*DPR + 0.2372*Impact + 0.0032*ADR + 0.1587

Where:
- KAST: Kill/Assist/Survived/Traded percentage (0-100)
- KPR: Kills per round
- DPR: Deaths per round
- Impact: Impact rating (calculated from KPR, APR, and contribution)
- ADR: Average damage per round

Reference: https://www.hltv.org/news/20695/introducing-rating-20
"""

from dataclasses import dataclass
from typing import Any

# HLTV 2.0 Rating coefficients (approximated from public analysis)
HLTV_RATING_COEFFICIENTS = {
    "kast": 0.0073,
    "kpr": 0.3591,
    "dpr": -0.5329,
    "impact": 0.2372,
    "adr": 0.0032,
    "rmk": 0.1587,  # Round multi-kill bonus
}

# Impact rating sub-coefficients
IMPACT_COEFFICIENTS = {
    "kpr": 2.13,
    "apr": 0.42,
    "base": -0.41,
}


@dataclass
class HLTVRatingResult:
    """Result of HLTV Rating calculation with component breakdown."""

    rating: float
    impact: float
    kast: float
    kpr: float
    dpr: float
    adr: float
    rmk: float  # Round multi-kill percentage

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rating": self.rating,
            "impact": self.impact,
            "kast": self.kast,
            "kpr": self.kpr,
            "dpr": self.dpr,
            "adr": self.adr,
            "rmk": self.rmk,
        }


def calculate_impact(
    kpr: float,
    apr: float = 0.0,
    clutch_wins: int = 0,
    multi_kill_3k: int = 0,
    multi_kill_4k: int = 0,
    multi_kill_5k: int = 0,
) -> float:
    """
    Calculate Impact rating component for HLTV 2.0.

    Impact measures how impactful a player's frags are, considering
    multi-kills and clutch situations.

    Args:
        kpr: Kills per round
        apr: Assists per round
        clutch_wins: Number of clutch rounds won
        multi_kill_3k: Number of rounds with 3 kills
        multi_kill_4k: Number of rounds with 4 kills
        multi_kill_5k: Number of rounds with 5 kills

    Returns:
        Impact rating value
    """
    # Base impact from kills and assists
    base = (
        IMPACT_COEFFICIENTS["kpr"] * kpr
        + IMPACT_COEFFICIENTS["apr"] * apr
        + IMPACT_COEFFICIENTS["base"]
    )

    # Add clutch bonus (0.1 per clutch win)
    clutch_bonus = clutch_wins * 0.1

    # Add multi-kill bonus
    mk_bonus = multi_kill_3k * 0.1 + multi_kill_4k * 0.2 + multi_kill_5k * 0.3

    return round(base + clutch_bonus + mk_bonus, 3)


def calculate_hltv_rating(
    kills: int,
    deaths: int,
    assists: int,
    adr: float,
    kast_pct: float,
    rounds: int,
    clutch_wins: int = 0,
    multi_kill_2k: int = 0,
    multi_kill_3k: int = 0,
    multi_kill_4k: int = 0,
    multi_kill_5k: int = 0,
) -> float:
    """
    Calculate HLTV 2.0 Rating.

    This is the main function for calculating a player's HLTV rating from
    their match statistics.

    Args:
        kills: Total kills
        deaths: Total deaths
        assists: Total assists
        adr: Average damage per round
        kast_pct: KAST percentage (0-100)
        rounds: Total rounds played
        clutch_wins: Number of clutch rounds won
        multi_kill_2k: Number of rounds with 2 kills
        multi_kill_3k: Number of rounds with 3 kills
        multi_kill_4k: Number of rounds with 4 kills
        multi_kill_5k: Number of rounds with 5 kills

    Returns:
        HLTV 2.0 Rating (typically between 0.0 and 2.0+ for exceptional games)
    """
    if rounds <= 0:
        return 0.0

    # Calculate per-round metrics
    kpr = kills / rounds
    dpr = deaths / rounds
    apr = assists / rounds

    # Calculate round multi-kill rate (decimal rate of rounds with 2+ kills)
    # Note: The coefficient 0.1587 expects this as a decimal (0.0-1.0), not percentage
    total_multi_kills = multi_kill_2k + multi_kill_3k + multi_kill_4k + multi_kill_5k
    rmk = (total_multi_kills / rounds) if rounds > 0 else 0.0

    # Calculate impact
    impact = calculate_impact(
        kpr=kpr,
        apr=apr,
        clutch_wins=clutch_wins,
        multi_kill_3k=multi_kill_3k,
        multi_kill_4k=multi_kill_4k,
        multi_kill_5k=multi_kill_5k,
    )

    # Calculate final rating
    rating = (
        HLTV_RATING_COEFFICIENTS["kast"] * kast_pct
        + HLTV_RATING_COEFFICIENTS["kpr"] * kpr
        + HLTV_RATING_COEFFICIENTS["dpr"] * dpr
        + HLTV_RATING_COEFFICIENTS["impact"] * impact
        + HLTV_RATING_COEFFICIENTS["adr"] * adr
        + HLTV_RATING_COEFFICIENTS["rmk"] * rmk
    )

    return round(max(0.0, rating), 2)


def calculate_hltv_rating_detailed(
    kills: int,
    deaths: int,
    assists: int,
    adr: float,
    kast_pct: float,
    rounds: int,
    clutch_wins: int = 0,
    multi_kill_2k: int = 0,
    multi_kill_3k: int = 0,
    multi_kill_4k: int = 0,
    multi_kill_5k: int = 0,
) -> HLTVRatingResult:
    """
    Calculate HLTV 2.0 Rating with detailed component breakdown.

    Returns all intermediate values used in the calculation for
    transparency and debugging.

    Args:
        kills: Total kills
        deaths: Total deaths
        assists: Total assists
        adr: Average damage per round
        kast_pct: KAST percentage (0-100)
        rounds: Total rounds played
        clutch_wins: Number of clutch rounds won
        multi_kill_2k: Number of rounds with 2 kills
        multi_kill_3k: Number of rounds with 3 kills
        multi_kill_4k: Number of rounds with 4 kills
        multi_kill_5k: Number of rounds with 5 kills

    Returns:
        HLTVRatingResult with rating and all component values
    """
    if rounds <= 0:
        return HLTVRatingResult(
            rating=0.0, impact=0.0, kast=0.0, kpr=0.0, dpr=0.0, adr=0.0, rmk=0.0
        )

    # Calculate per-round metrics
    kpr = kills / rounds
    dpr = deaths / rounds
    apr = assists / rounds

    # Calculate round multi-kill rate (decimal for formula, percentage for display)
    total_multi_kills = multi_kill_2k + multi_kill_3k + multi_kill_4k + multi_kill_5k
    rmk_decimal = (total_multi_kills / rounds) if rounds > 0 else 0.0
    rmk_pct = rmk_decimal * 100  # For display

    # Calculate impact
    impact = calculate_impact(
        kpr=kpr,
        apr=apr,
        clutch_wins=clutch_wins,
        multi_kill_3k=multi_kill_3k,
        multi_kill_4k=multi_kill_4k,
        multi_kill_5k=multi_kill_5k,
    )

    # Calculate final rating (using decimal RMK)
    rating = (
        HLTV_RATING_COEFFICIENTS["kast"] * kast_pct
        + HLTV_RATING_COEFFICIENTS["kpr"] * kpr
        + HLTV_RATING_COEFFICIENTS["dpr"] * dpr
        + HLTV_RATING_COEFFICIENTS["impact"] * impact
        + HLTV_RATING_COEFFICIENTS["adr"] * adr
        + HLTV_RATING_COEFFICIENTS["rmk"] * rmk_decimal
    )

    return HLTVRatingResult(
        rating=round(max(0.0, rating), 2),
        impact=round(impact, 3),
        kast=round(kast_pct, 1),
        kpr=round(kpr, 2),
        dpr=round(dpr, 2),
        adr=round(adr, 1),
        rmk=round(rmk_pct, 1),  # Store as percentage for display
    )


def get_rating_tier(rating: float) -> str:
    """
    Get a descriptive tier for a given HLTV rating.

    Tiers based on professional player statistics:
    - Elite (Top 20 Pro): 1.15+
    - Very Good: 1.05 - 1.14
    - Good: 0.95 - 1.04
    - Average: 0.85 - 0.94
    - Below Average: 0.75 - 0.84
    - Poor: < 0.75

    Args:
        rating: HLTV 2.0 Rating value

    Returns:
        String description of the rating tier
    """
    if rating >= 1.30:
        return "Exceptional"
    if rating >= 1.15:
        return "Elite"
    if rating >= 1.05:
        return "Very Good"
    if rating >= 0.95:
        return "Good"
    if rating >= 0.85:
        return "Average"
    if rating >= 0.75:
        return "Below Average"
    return "Poor"


def get_rating_color(rating: float) -> str:
    """
    Get a color code for displaying the rating.

    Args:
        rating: HLTV 2.0 Rating value

    Returns:
        CSS color value
    """
    if rating >= 1.15:
        return "#10b981"  # Green (success)
    if rating >= 1.00:
        return "#00e5ff"  # Cyan (accent)
    if rating >= 0.85:
        return "#f59e0b"  # Yellow (warning)
    return "#ef4444"  # Red (danger)
