"""
OpenSight Player Profile and Trend Analysis System.

Provides advanced player analytics including:
- Performance trends over time
- Period comparisons
- Career milestones
- Statistical insights

All features are 100% FREE - no paid services required.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from opensight.infra.database import DatabaseManager, get_db

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Trend Analysis
# =============================================================================


@dataclass
class TrendResult:
    """Result of a trend analysis."""

    metric: str
    current_value: float
    previous_value: float
    change: float
    change_percent: float
    trend: str  # "improving", "declining", "stable"
    confidence: str  # "high", "medium", "low"
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "current": round(self.current_value, 2),
            "previous": round(self.previous_value, 2),
            "change": round(self.change, 2),
            "change_percent": round(self.change_percent, 1),
            "trend": self.trend,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
        }


@dataclass
class Milestone:
    """A player milestone or achievement."""

    name: str
    description: str
    achieved_date: datetime | None
    value: float | int
    category: str  # "kills", "rating", "clutches", etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "achieved_date": (self.achieved_date.isoformat() if self.achieved_date else None),
            "value": self.value,
            "category": self.category,
        }


@dataclass
class PlayerInsights:
    """Comprehensive insights about a player's performance."""

    steam_id: str
    player_name: str
    trends: list[TrendResult] = field(default_factory=list)
    milestones: list[Milestone] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    areas_to_improve: list[str] = field(default_factory=list)
    form_description: str = ""
    recommended_focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steam_id": self.steam_id,
            "player_name": self.player_name,
            "trends": [t.to_dict() for t in self.trends],
            "milestones": [m.to_dict() for m in self.milestones],
            "strengths": self.strengths,
            "areas_to_improve": self.areas_to_improve,
            "form_description": self.form_description,
            "recommended_focus": self.recommended_focus,
        }


# =============================================================================
# Profile Analytics Engine
# =============================================================================


class ProfileAnalyzer:
    """
    Analyzes player profiles for trends, milestones, and insights.

    All computations are done locally using FREE libraries.
    """

    # Thresholds for trend detection
    TREND_THRESHOLD = 5.0  # % change to be considered significant
    STABLE_THRESHOLD = 2.0  # % change within this is "stable"

    # Benchmarks for metric evaluation
    BENCHMARKS = {
        "hltv_rating": {"elite": 1.2, "good": 1.0, "average": 0.9, "poor": 0.8},
        "adr": {"elite": 90, "good": 75, "average": 65, "poor": 55},
        "kast": {"elite": 75, "good": 70, "average": 65, "poor": 55},
        "headshot_pct": {"elite": 55, "good": 45, "average": 35, "poor": 25},
        "opening_rate": {"elite": 60, "good": 52, "average": 48, "poor": 40},
        "clutch_rate": {"elite": 35, "good": 25, "average": 18, "poor": 10},
    }

    def __init__(self, db: DatabaseManager | None = None):
        """Initialize analyzer with database connection."""
        self.db = db or get_db()

    def analyze_player(self, steam_id: str) -> PlayerInsights | None:
        """
        Perform comprehensive analysis of a player.

        Args:
            steam_id: Player's Steam ID

        Returns:
            PlayerInsights object with trends, milestones, and recommendations
        """
        profile = self.db.get_player_profile(steam_id)
        if not profile:
            return None

        # Get match history for trend analysis
        matches = self.db.get_player_match_history(steam_id, limit=50)
        if len(matches) < 3:
            # Not enough data for meaningful analysis
            return PlayerInsights(
                steam_id=steam_id,
                player_name=profile.get("player_name", "Unknown"),
                form_description="Not enough matches for trend analysis (need 3+)",
            )

        insights = PlayerInsights(
            steam_id=steam_id,
            player_name=profile.get("player_name", "Unknown"),
        )

        # Analyze trends
        insights.trends = self._analyze_trends(matches)

        # Check milestones
        insights.milestones = self._check_milestones(profile, matches)

        # Identify strengths and weaknesses
        insights.strengths, insights.areas_to_improve = self._evaluate_performance(profile)

        # Describe current form
        insights.form_description = self._describe_form(matches)

        # Generate recommendations
        insights.recommended_focus = self._generate_recommendations(
            profile, insights.trends, insights.areas_to_improve
        )

        return insights

    def _analyze_trends(self, matches: list[dict]) -> list[TrendResult]:
        """Analyze performance trends from match history."""
        if len(matches) < 6:
            return []

        # Split into recent and previous periods
        mid = len(matches) // 2
        recent = matches[:mid]
        previous = matches[mid:]

        trends = []

        # Analyze key metrics
        metrics = [
            ("rating", "hltv_rating"),
            ("adr", "adr"),
            ("kills", "kills"),
            ("deaths", "deaths"),
        ]

        for metric_name, key in metrics:
            recent_avg = sum(m.get(key, 0) for m in recent) / len(recent)
            previous_avg = sum(m.get(key, 0) for m in previous) / len(previous)

            if previous_avg == 0:
                continue

            change = recent_avg - previous_avg
            change_pct = (change / previous_avg) * 100

            # Determine trend direction
            if abs(change_pct) < self.STABLE_THRESHOLD:
                trend = "stable"
            elif change_pct > 0:
                trend = "improving" if metric_name != "deaths" else "declining"
            else:
                trend = "declining" if metric_name != "deaths" else "improving"

            # Confidence based on sample size
            confidence = (
                "high" if len(matches) >= 20 else ("medium" if len(matches) >= 10 else "low")
            )

            trends.append(
                TrendResult(
                    metric=metric_name,
                    current_value=recent_avg,
                    previous_value=previous_avg,
                    change=change,
                    change_percent=change_pct,
                    trend=trend,
                    confidence=confidence,
                    sample_size=len(matches),
                )
            )

        return trends

    def _check_milestones(self, profile: dict, matches: list[dict]) -> list[Milestone]:
        """Check for achieved milestones."""
        milestones = []
        career = profile.get("career", {})
        highs = profile.get("career_highs", {})

        # Kill milestones
        total_kills = career.get("kills", 0)
        kill_thresholds = [100, 500, 1000, 5000, 10000]
        for threshold in kill_thresholds:
            if total_kills >= threshold:
                milestones.append(
                    Milestone(
                        name=f"{threshold} Career Kills",
                        description=f"Reached {threshold} total kills",
                        achieved_date=None,  # Would need to track when
                        value=total_kills,
                        category="kills",
                    )
                )

        # Match milestones
        total_matches = career.get("matches", 0)
        match_thresholds = [10, 25, 50, 100, 250, 500]
        for threshold in match_thresholds:
            if total_matches >= threshold:
                milestones.append(
                    Milestone(
                        name=f"{threshold} Matches Played",
                        description=f"Played {threshold} matches",
                        achieved_date=None,
                        value=total_matches,
                        category="matches",
                    )
                )

        # Rating milestones
        best_rating = highs.get("best_rating", 0)
        if best_rating >= 2.0:
            milestones.append(
                Milestone(
                    name="2.0+ Rating Game",
                    description="Achieved a 2.0+ HLTV rating in a match",
                    achieved_date=None,
                    value=best_rating,
                    category="rating",
                )
            )
        elif best_rating >= 1.5:
            milestones.append(
                Milestone(
                    name="1.5+ Rating Game",
                    description="Achieved a 1.5+ HLTV rating in a match",
                    achieved_date=None,
                    value=best_rating,
                    category="rating",
                )
            )

        # Clutch milestones
        clutches = profile.get("clutches", {})
        clutch_wins = clutches.get("wins", 0)
        if clutch_wins >= 10:
            milestones.append(
                Milestone(
                    name="Clutch Master",
                    description=f"Won {clutch_wins} clutch situations",
                    achieved_date=None,
                    value=clutch_wins,
                    category="clutches",
                )
            )

        return milestones

    def _evaluate_performance(self, profile: dict) -> tuple[list[str], list[str]]:
        """Identify strengths and areas to improve."""
        strengths = []
        weaknesses = []

        averages = profile.get("averages", {})
        clutches = profile.get("clutches", {})
        opening = profile.get("opening_duels", {})

        # Rating evaluation
        rating = averages.get("rating", 1.0)
        if rating >= self.BENCHMARKS["hltv_rating"]["elite"]:
            strengths.append("Elite overall performance (1.2+ rating)")
        elif rating >= self.BENCHMARKS["hltv_rating"]["good"]:
            strengths.append("Strong overall performance")
        elif rating < self.BENCHMARKS["hltv_rating"]["poor"]:
            weaknesses.append("Overall impact needs improvement")

        # ADR evaluation
        adr = averages.get("adr", 0)
        if adr >= self.BENCHMARKS["adr"]["elite"]:
            strengths.append("Excellent damage output")
        elif adr < self.BENCHMARKS["adr"]["poor"]:
            weaknesses.append("Damage per round below average")

        # KAST evaluation
        kast = averages.get("kast", 0)
        if kast >= self.BENCHMARKS["kast"]["elite"]:
            strengths.append("Exceptional round contribution (KAST)")
        elif kast < self.BENCHMARKS["kast"]["poor"]:
            weaknesses.append("Round contribution needs work")

        # Headshot percentage
        hs_pct = averages.get("hs_percentage", 0)
        if hs_pct >= self.BENCHMARKS["headshot_pct"]["elite"]:
            strengths.append("Elite aim precision (high HS%)")
        elif hs_pct < self.BENCHMARKS["headshot_pct"]["poor"]:
            weaknesses.append("Headshot percentage below average")

        # Opening duels
        opening_rate = opening.get("win_rate", 50)
        if opening_rate >= self.BENCHMARKS["opening_rate"]["elite"]:
            strengths.append("Dominant in opening duels")
        elif opening_rate < self.BENCHMARKS["opening_rate"]["poor"]:
            weaknesses.append("Opening duel win rate needs improvement")

        # Clutches
        clutch_rate = clutches.get("win_rate", 0)
        if clutch_rate >= self.BENCHMARKS["clutch_rate"]["elite"]:
            strengths.append("Clutch performer under pressure")
        elif (
            clutch_rate < self.BENCHMARKS["clutch_rate"]["poor"]
            and clutches.get("situations", 0) >= 5
        ):
            weaknesses.append("Clutch situations need work")

        return strengths, weaknesses

    def _describe_form(self, matches: list[dict]) -> str:
        """Generate a description of current form."""
        if len(matches) < 3:
            return "Insufficient data"

        recent_5 = matches[:5] if len(matches) >= 5 else matches
        recent_ratings = [m.get("rating", 1.0) for m in recent_5]
        avg_recent = sum(recent_ratings) / len(recent_ratings)

        wins = sum(1 for m in recent_5 if m.get("won"))
        win_rate = wins / len(recent_5) * 100

        # Describe form
        if avg_recent >= 1.2 and win_rate >= 60:
            return "🔥 Hot streak - performing exceptionally well"
        elif avg_recent >= 1.1 and win_rate >= 50:
            return "📈 Good form - playing above average"
        elif avg_recent >= 0.95:
            return "➡️ Stable form - consistent performance"
        elif avg_recent >= 0.85:
            return "📉 Below average - slight dip in performance"
        else:
            return "⚠️ Struggling - performance needs attention"

    def _generate_recommendations(
        self,
        profile: dict,
        trends: list[TrendResult],
        weaknesses: list[str],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Based on weaknesses
        for weakness in weaknesses[:2]:  # Top 2 weaknesses
            if "damage" in weakness.lower():
                recommendations.append("Focus on trading damage even when not getting kills")
            elif "headshot" in weakness.lower():
                recommendations.append("Practice crosshair placement at head level")
            elif "opening" in weakness.lower():
                recommendations.append("Work on positioning for favorable opening duels")
            elif "clutch" in weakness.lower():
                recommendations.append("Practice 1vX scenarios in aim trainers")
            elif "kast" in weakness.lower() or "contribution" in weakness.lower():
                recommendations.append("Focus on staying alive and trading teammates")

        # Based on declining trends
        for trend in trends:
            if trend.trend == "declining" and trend.confidence != "low":
                if trend.metric == "rating":
                    recommendations.append(
                        "Overall impact declining - review recent demos for mistakes"
                    )
                elif trend.metric == "adr":
                    recommendations.append("Damage output declining - be more aggressive in trades")

        # General recommendations if list is short
        if len(recommendations) < 2:
            recommendations.append("Review demos of matches with sub-1.0 rating")
            recommendations.append("Maintain consistent warmup routine")

        return recommendations[:4]  # Max 4 recommendations

    def compare_players(self, steam_id_a: str, steam_id_b: str) -> dict[str, Any] | None:
        """
        Compare two players head-to-head.

        Returns comparison data suitable for radar charts and tables.
        """
        profile_a = self.db.get_player_profile(steam_id_a)
        profile_b = self.db.get_player_profile(steam_id_b)

        if not profile_a or not profile_b:
            return None

        avg_a = profile_a.get("averages", {})
        avg_b = profile_b.get("averages", {})

        # Metrics to compare
        metrics = ["rating", "adr", "kast", "kills", "deaths", "hs_percentage"]

        comparison = {
            "player_a": {
                "steam_id": steam_id_a,
                "name": profile_a.get("player_name"),
                "matches": profile_a.get("career", {}).get("matches", 0),
            },
            "player_b": {
                "steam_id": steam_id_b,
                "name": profile_b.get("player_name"),
                "matches": profile_b.get("career", {}).get("matches", 0),
            },
            "metrics": {},
            "winner_by_metric": {},
        }

        for metric in metrics:
            val_a = avg_a.get(metric, 0)
            val_b = avg_b.get(metric, 0)

            comparison["metrics"][metric] = {
                "player_a": round(val_a, 2),
                "player_b": round(val_b, 2),
                "difference": round(val_a - val_b, 2),
            }

            # Determine winner (deaths: lower is better)
            if metric == "deaths":
                winner = "a" if val_a < val_b else ("b" if val_b < val_a else "tie")
            else:
                winner = "a" if val_a > val_b else ("b" if val_b > val_a else "tie")

            comparison["winner_by_metric"][metric] = winner

        return comparison

    def get_performance_over_time(self, steam_id: str, days: int = 30) -> list[dict]:
        """
        Get performance data points for time-series visualization.

        Args:
            steam_id: Player's Steam ID
            days: Number of days to look back

        Returns:
            List of data points with date and metrics
        """
        matches = self.db.get_player_match_history(steam_id, limit=100)

        # Filter by date
        cutoff = datetime.utcnow() - timedelta(days=days)
        data_points = []

        for match in matches:
            date_str = match.get("date")
            if date_str:
                try:
                    match_date = datetime.fromisoformat(date_str)
                    if match_date >= cutoff:
                        data_points.append(
                            {
                                "date": date_str,
                                "rating": match.get("rating", 1.0),
                                "adr": match.get("adr", 0),
                                "kills": match.get("kills", 0),
                                "deaths": match.get("deaths", 0),
                                "won": match.get("won"),
                            }
                        )
                except ValueError:
                    continue

        # Sort by date
        data_points.sort(key=lambda x: x["date"])

        return data_points


# =============================================================================
# Convenience Functions
# =============================================================================


def get_player_insights(steam_id: str) -> dict | None:
    """Get comprehensive player insights as a dictionary."""
    analyzer = ProfileAnalyzer()
    insights = analyzer.analyze_player(steam_id)
    return insights.to_dict() if insights else None


def compare_two_players(steam_id_a: str, steam_id_b: str) -> dict | None:
    """Compare two players and return comparison data."""
    analyzer = ProfileAnalyzer()
    return analyzer.compare_players(steam_id_a, steam_id_b)
