"""
OpenSight Persona Analysis Module

Determines player "Match Identity" personas based on their performance metrics.
Inspired by Leetify's Match Identity feature.

Personas are determined by analyzing player stats and identifying their
strongest performance characteristics.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Persona Definitions
# =============================================================================

PERSONAS = {
    "the_cleanup": {
        "name": "The Cleanup",
        "description": "Excels at trade kills and avenging teammates",
        "icon": "trade",
        "color": "#4CAF50",
        "check": lambda s: _calc_trade_rate(s) > 0.5 and s.get("trade_kill_success", 0) >= 2,
    },
    "the_opener": {
        "name": "The Opener",
        "description": "Leads the charge with entry frags",
        "icon": "entry",
        "color": "#FF5722",
        "check": lambda s: _calc_entry_rate(s) > 0.5 and s.get("entry_attempts", 0) >= 3,
    },
    "the_anchor": {
        "name": "The Anchor",
        "description": "Clutch master who excels in 1vX situations",
        "icon": "anchor",
        "color": "#2196F3",
        "check": lambda s: _calc_clutch_rate(s) > 0.3 and s.get("clutch_situations", 0) >= 2,
    },
    "the_utility_master": {
        "name": "The Utility Master",
        "description": "Uses grenades to maximum effect",
        "icon": "utility",
        "color": "#9C27B0",
        "check": lambda s: s.get("utility_rating", 0) > 65 or s.get("enemies_flashed", 0) >= 10,
    },
    "the_headhunter": {
        "name": "The Headhunter",
        "description": "Precision aimer with high headshot percentage",
        "icon": "headshot",
        "color": "#F44336",
        "check": lambda s: s.get("hs_pct", 0) > 50 and s.get("kills", 0) >= 5,
    },
    "the_survivor": {
        "name": "The Survivor",
        "description": "Stays alive and contributes consistently",
        "icon": "survival",
        "color": "#00BCD4",
        "check": lambda s: s.get("kast", 0) > 75 and s.get("deaths", 10) < s.get("kills", 0),
    },
    "the_damage_dealer": {
        "name": "The Damage Dealer",
        "description": "Maximizes damage output every round",
        "icon": "damage",
        "color": "#FF9800",
        "check": lambda s: s.get("adr", 0) > 85,
    },
    "the_flash_master": {
        "name": "The Flash Master",
        "description": "Blinds enemies to set up kills",
        "icon": "flash",
        "color": "#FFEB3B",
        "check": lambda s: s.get("flash_assists", 0) >= 3 or s.get("enemies_flashed", 0) >= 8,
    },
    "the_terminator": {
        "name": "The Terminator",
        "description": "High kill count with dominant performance",
        "icon": "kills",
        "color": "#E91E63",
        "check": lambda s: s.get("kills", 0) >= 25 and s.get("hltv_rating", 0) > 1.2,
    },
    "the_competitor": {
        "name": "The Competitor",
        "description": "Well-rounded player with solid fundamentals",
        "icon": "default",
        "color": "#607D8B",
        "check": lambda s: True,  # Default fallback
    },
}


# =============================================================================
# Stat Categories for Top 5 Stats
# =============================================================================

STAT_CATEGORIES = {
    "he_damage": ("HE Damage", "UTILITY"),
    "hs_pct": ("Headshot Accuracy", "AIM"),
    "cp_median_deg": ("Crosshair Placement", "AIM"),
    "ttd_median_ms": ("Time to Damage", "AIM"),
    "trade_kill_success": ("Trade Kills", "TRADES"),
    "enemies_flashed": ("Enemies Flashed", "UTILITY"),
    "flash_assists": ("Flash Assists", "UTILITY"),
    "entry_success": ("Entry Frags", "ENTRY"),
    "adr": ("ADR", "IMPACT"),
    "kills": ("Kills", "IMPACT"),
    "kast": ("KAST%", "IMPACT"),
    "clutch_wins": ("Clutch Wins", "CLUTCH"),
    "hltv_rating": ("HLTV Rating", "RATING"),
    "utility_rating": ("Utility Rating", "UTILITY"),
    "aim_rating": ("Aim Rating", "AIM"),
}


# Benchmarks for percentile calculation
STAT_BENCHMARKS = {
    "he_damage": {"min": 0, "avg": 50, "max": 200, "higher_is_better": True},
    "hs_pct": {"min": 0, "avg": 35, "max": 70, "higher_is_better": True},
    "cp_median_deg": {"min": 25, "avg": 12, "max": 3, "higher_is_better": False},  # Lower is better
    "ttd_median_ms": {"min": 500, "avg": 250, "max": 100, "higher_is_better": False},  # Lower is better
    "trade_kill_success": {"min": 0, "avg": 2, "max": 6, "higher_is_better": True},
    "enemies_flashed": {"min": 0, "avg": 5, "max": 15, "higher_is_better": True},
    "flash_assists": {"min": 0, "avg": 1, "max": 5, "higher_is_better": True},
    "entry_success": {"min": 0, "avg": 2, "max": 6, "higher_is_better": True},
    "adr": {"min": 40, "avg": 70, "max": 110, "higher_is_better": True},
    "kills": {"min": 5, "avg": 15, "max": 30, "higher_is_better": True},
    "kast": {"min": 50, "avg": 68, "max": 85, "higher_is_better": True},
    "clutch_wins": {"min": 0, "avg": 1, "max": 4, "higher_is_better": True},
    "hltv_rating": {"min": 0.6, "avg": 1.0, "max": 1.5, "higher_is_better": True},
    "utility_rating": {"min": 20, "avg": 50, "max": 80, "higher_is_better": True},
    "aim_rating": {"min": 30, "avg": 50, "max": 80, "higher_is_better": True},
}


# =============================================================================
# Helper Functions
# =============================================================================


def _calc_trade_rate(stats: dict[str, Any]) -> float:
    """Calculate trade kill success rate."""
    opportunities = stats.get("trade_kill_opportunities", 0)
    success = stats.get("trade_kill_success", 0)
    if opportunities == 0:
        return 0.0
    return success / opportunities


def _calc_entry_rate(stats: dict[str, Any]) -> float:
    """Calculate entry frag success rate."""
    attempts = stats.get("entry_attempts", 0)
    success = stats.get("entry_success", 0)
    if attempts == 0:
        return 0.0
    return success / attempts


def _calc_clutch_rate(stats: dict[str, Any]) -> float:
    """Calculate clutch win rate."""
    situations = stats.get("clutch_situations", 0)
    wins = stats.get("clutch_wins", 0)
    if situations == 0:
        return 0.0
    return wins / situations


def _calculate_percentile(value: float | None, benchmark: dict[str, Any]) -> float:
    """
    Calculate percentile score (0-100) for a stat value.

    Args:
        value: The stat value
        benchmark: Benchmark dict with min, avg, max, higher_is_better

    Returns:
        Percentile score from 0-100
    """
    if value is None:
        return 50.0  # Default to average if no value

    min_val = benchmark["min"]
    avg_val = benchmark["avg"]
    max_val = benchmark["max"]
    higher_is_better = benchmark["higher_is_better"]

    # Normalize value to 0-100 scale
    if higher_is_better:
        # Higher value = higher percentile
        if value <= min_val:
            return 0.0
        elif value >= max_val:
            return 100.0
        elif value <= avg_val:
            # 0 to 50 range
            return 50.0 * (value - min_val) / (avg_val - min_val)
        else:
            # 50 to 100 range
            return 50.0 + 50.0 * (value - avg_val) / (max_val - avg_val)
    else:
        # Lower value = higher percentile (inverted)
        if value >= min_val:
            return 0.0
        elif value <= max_val:
            return 100.0
        elif value >= avg_val:
            # 0 to 50 range
            return 50.0 * (min_val - value) / (min_val - avg_val)
        else:
            # 50 to 100 range
            return 50.0 + 50.0 * (avg_val - value) / (avg_val - max_val)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PersonaResult:
    """Result of persona determination."""

    id: str
    name: str
    description: str
    icon: str
    color: str
    confidence: float
    primary_trait: str | None
    secondary_trait: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "confidence": round(self.confidence, 2),
            "primary_trait": self.primary_trait,
            "secondary_trait": self.secondary_trait,
        }


@dataclass
class TopStatResult:
    """Result for a top stat entry."""

    stat: str
    name: str
    category: str
    value: float | int
    formatted_value: str
    percentile: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "stat": self.stat,
            "name": self.name,
            "category": self.category,
            "value": self.value,
            "formatted_value": self.formatted_value,
            "percentile": round(self.percentile, 1),
            "rank": self.rank,
        }


@dataclass
class ComparisonRow:
    """A row in the This Match vs Average comparison table."""

    metric: str
    label: str
    this_match: float | int
    average: float | int
    diff: float
    diff_percent: float | None
    is_better: bool
    formatted_this: str
    formatted_avg: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "metric": self.metric,
            "label": self.label,
            "this_match": self.this_match,
            "average": self.average,
            "diff": round(self.diff, 2),
            "diff_percent": round(self.diff_percent, 1) if self.diff_percent else None,
            "is_better": self.is_better,
            "formatted_this": self.formatted_this,
            "formatted_avg": self.formatted_avg,
        }


# =============================================================================
# Persona Analyzer Class
# =============================================================================


class PersonaAnalyzer:
    """Analyzes player stats to determine Match Identity persona."""

    def __init__(self, baselines: dict[str, dict] | None = None):
        """
        Initialize the analyzer.

        Args:
            baselines: Optional baseline data for percentile calculations
        """
        self.baselines = baselines or {}

    def determine_persona(self, stats: dict[str, Any]) -> PersonaResult:
        """
        Determine the most appropriate persona for a player based on their stats.

        Args:
            stats: Dictionary of player statistics

        Returns:
            PersonaResult with the determined persona
        """
        matches = []

        # Check each persona (except default)
        for persona_id, persona_def in PERSONAS.items():
            if persona_id == "the_competitor":
                continue

            try:
                if persona_def["check"](stats):
                    # Calculate confidence based on how strongly they match
                    confidence = self._calculate_confidence(persona_id, stats)
                    matches.append((persona_id, persona_def, confidence))
            except Exception as e:
                logger.debug(f"Error checking persona {persona_id}: {e}")
                continue

        # Sort by confidence
        matches.sort(key=lambda x: x[2], reverse=True)

        if matches:
            best_id, best_def, best_confidence = matches[0]
            secondary_trait = matches[1][0] if len(matches) > 1 else None
        else:
            # Default persona
            best_id = "the_competitor"
            best_def = PERSONAS["the_competitor"]
            best_confidence = 0.5
            secondary_trait = None

        return PersonaResult(
            id=best_id,
            name=best_def["name"],
            description=best_def["description"],
            icon=best_def["icon"],
            color=best_def["color"],
            confidence=best_confidence,
            primary_trait=best_id,
            secondary_trait=secondary_trait,
        )

    def _calculate_confidence(self, persona_id: str, stats: dict[str, Any]) -> float:
        """Calculate confidence score for a persona match."""
        # Base confidence mappings
        confidence_factors = {
            "the_cleanup": lambda s: min(1.0, _calc_trade_rate(s) * 1.5),
            "the_opener": lambda s: min(1.0, _calc_entry_rate(s) * 1.5),
            "the_anchor": lambda s: min(1.0, _calc_clutch_rate(s) * 2),
            "the_utility_master": lambda s: min(1.0, s.get("utility_rating", 50) / 80),
            "the_headhunter": lambda s: min(1.0, s.get("hs_pct", 0) / 60),
            "the_survivor": lambda s: min(1.0, s.get("kast", 0) / 85),
            "the_damage_dealer": lambda s: min(1.0, s.get("adr", 0) / 100),
            "the_flash_master": lambda s: min(1.0, s.get("flash_assists", 0) / 4),
            "the_terminator": lambda s: min(1.0, s.get("kills", 0) / 25),
        }

        factor = confidence_factors.get(persona_id)
        if factor:
            try:
                return factor(stats)
            except Exception:
                return 0.5
        return 0.5

    def calculate_top_5_stats(
        self,
        current_stats: dict[str, Any],
        baselines: dict[str, dict] | None = None,
    ) -> list[TopStatResult]:
        """
        Calculate the top 5 stats for a player compared to their baselines.

        Args:
            current_stats: Current match statistics
            baselines: Optional baseline data (uses self.baselines if not provided)

        Returns:
            List of top 5 TopStatResult objects, sorted by percentile
        """
        baselines = baselines or self.baselines
        rankings = []

        for stat_key, (display_name, category) in STAT_CATEGORIES.items():
            value = current_stats.get(stat_key)
            if value is None:
                continue

            # Calculate percentile using baselines or benchmarks
            if stat_key in baselines and baselines[stat_key].get("sample_count", 0) >= 3:
                # Use player's historical baselines
                baseline = baselines[stat_key]
                avg = baseline.get("avg", value)
                std = baseline.get("std", 1) or 1

                # Z-score based percentile
                z_score = (value - avg) / std
                percentile = min(100, max(0, 50 + z_score * 15))
            elif stat_key in STAT_BENCHMARKS:
                # Use global benchmarks
                percentile = _calculate_percentile(value, STAT_BENCHMARKS[stat_key])
            else:
                percentile = 50.0

            # Format value for display
            formatted = self._format_stat_value(stat_key, value)

            rankings.append(
                TopStatResult(
                    stat=stat_key,
                    name=display_name,
                    category=category,
                    value=value,
                    formatted_value=formatted,
                    percentile=percentile,
                    rank=0,  # Will be set after sorting
                )
            )

        # Sort by percentile descending
        rankings.sort(key=lambda x: x.percentile, reverse=True)

        # Assign ranks and return top 5
        top_5 = rankings[:5]
        for i, stat in enumerate(top_5):
            stat.rank = i + 1

        return top_5

    def _format_stat_value(self, stat_key: str, value: float | int) -> str:
        """Format a stat value for display."""
        if value is None:
            return "N/A"

        # Percentage stats
        if stat_key in ("hs_pct", "kast"):
            return f"{value:.0f}%"

        # Decimal stats
        if stat_key in ("hltv_rating",):
            return f"{value:.2f}"

        # Angle stats
        if stat_key in ("cp_median_deg",):
            return f"{value:.1f}°"

        # Time stats
        if stat_key in ("ttd_median_ms",):
            return f"{value:.0f}ms"

        # Float stats with one decimal
        if stat_key in ("adr", "aim_rating", "utility_rating"):
            return f"{value:.1f}"

        # Integer stats
        return str(int(value))

    def build_comparison_table(
        self,
        current_stats: dict[str, Any],
        baselines: dict[str, dict],
    ) -> list[ComparisonRow]:
        """
        Build the This Match vs Average comparison table.

        Args:
            current_stats: Current match statistics
            baselines: Player's baseline averages

        Returns:
            List of ComparisonRow objects
        """
        # Metrics to compare with labels and format specs
        metrics_config = [
            ("opensight_rating", "OpenSight Rating", True, ".1f"),
            ("hltv_rating", "HLTV Rating", True, ".2f"),
            ("aim_rating", "Aim Rating", True, ".0f"),
            ("utility_rating", "Utility Rating", True, ".0f"),
            ("trade_kill_opportunities", "Trade Kill Opps", True, ".0f"),
            ("kills", "Kills", True, ".0f"),
            ("adr", "ADR", True, ".1f"),
            ("kast", "KAST%", True, ".0f"),
            ("entry_attempts", "Opening Duel Attempts", None, ".0f"),  # None = neutral
            ("clutch_kills", "Clutch Kills", True, ".0f"),
        ]

        comparison = []

        for metric, label, higher_is_better, fmt in metrics_config:
            curr_val = current_stats.get(metric, 0)
            if curr_val is None:
                curr_val = 0

            # Get baseline average
            baseline = baselines.get(metric, {})
            avg_val = baseline.get("avg", curr_val)

            # Calculate difference
            diff = curr_val - avg_val
            diff_pct = (diff / avg_val * 100) if avg_val != 0 else None

            # Determine if better
            if higher_is_better is None:
                is_better = True  # Neutral
            elif higher_is_better:
                is_better = diff >= 0
            else:
                is_better = diff <= 0

            # Format values
            format_spec = f"{{:{fmt}}}"
            formatted_this = format_spec.format(curr_val)
            formatted_avg = format_spec.format(avg_val)

            # Add percentage symbol where needed
            if "%" in label or metric == "kast":
                formatted_this += "%"
                formatted_avg += "%"

            comparison.append(
                ComparisonRow(
                    metric=metric,
                    label=label,
                    this_match=curr_val,
                    average=round(avg_val, 2),
                    diff=diff,
                    diff_percent=diff_pct,
                    is_better=is_better,
                    formatted_this=formatted_this,
                    formatted_avg=formatted_avg,
                )
            )

        return comparison


# =============================================================================
# Convenience Functions
# =============================================================================


def determine_persona(stats: dict[str, Any]) -> PersonaResult:
    """Convenience function to determine persona without instantiating analyzer."""
    analyzer = PersonaAnalyzer()
    return analyzer.determine_persona(stats)


def calculate_top_5(
    stats: dict[str, Any],
    baselines: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Convenience function to calculate top 5 stats."""
    analyzer = PersonaAnalyzer(baselines)
    results = analyzer.calculate_top_5_stats(stats, baselines)
    return [r.to_dict() for r in results]


def build_comparison(
    current: dict[str, Any],
    baselines: dict[str, dict],
) -> list[dict[str, Any]]:
    """Convenience function to build comparison table."""
    analyzer = PersonaAnalyzer(baselines)
    results = analyzer.build_comparison_table(current, baselines)
    return [r.to_dict() for r in results]
