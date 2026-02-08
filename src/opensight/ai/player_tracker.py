"""
Cross-Match Player Development Tracking.

Tracks player metrics over time across multiple demos, identifies
improvement trends and regression, provides role-specific benchmarks
by competitive level, and generates practice recommendations.

Builds on top of the existing MatchHistory table and DatabaseManager
infrastructure rather than creating separate storage.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class MatchSnapshot:
    """Snapshot of a single match's key metrics for tracking.

    Can be constructed from orchestrator output or from MatchHistory DB rows.
    """

    # Identity
    steam_id: str
    demo_hash: str
    map_name: str | None = None
    result: str | None = None  # "win", "loss", "draw"
    analyzed_at: str | None = None

    # Core stats
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    adr: float = 0.0
    kast: float = 0.0
    hs_pct: float = 0.0
    rounds_played: int = 0

    # Ratings
    hltv_rating: float = 1.0
    aim_rating: float = 0.0
    utility_rating: float = 0.0

    # Advanced
    ttd_median_ms: float | None = None
    cp_median_deg: float | None = None

    # Duel stats
    entry_attempts: int = 0
    entry_success: int = 0
    clutch_situations: int = 0
    clutch_wins: int = 0
    trade_kill_success: int = 0
    trade_kill_attempts: int = 0

    # Utility
    enemies_flashed: int = 0
    flash_assists: int = 0
    he_damage: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "steam_id": self.steam_id,
            "demo_hash": self.demo_hash,
            "map_name": self.map_name,
            "result": self.result,
            "analyzed_at": self.analyzed_at,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "adr": round(self.adr, 1),
            "kast": round(self.kast, 1),
            "hs_pct": round(self.hs_pct, 1),
            "rounds_played": self.rounds_played,
            "hltv_rating": round(self.hltv_rating, 2),
            "aim_rating": round(self.aim_rating, 1),
            "utility_rating": round(self.utility_rating, 1),
            "ttd_median_ms": (
                round(self.ttd_median_ms, 1) if self.ttd_median_ms is not None else None
            ),
            "cp_median_deg": (
                round(self.cp_median_deg, 1) if self.cp_median_deg is not None else None
            ),
            "entry_attempts": self.entry_attempts,
            "entry_success": self.entry_success,
            "clutch_situations": self.clutch_situations,
            "clutch_wins": self.clutch_wins,
            "trade_kill_success": self.trade_kill_success,
            "trade_kill_attempts": self.trade_kill_attempts,
            "enemies_flashed": self.enemies_flashed,
            "flash_assists": self.flash_assists,
            "he_damage": self.he_damage,
        }


@dataclass
class TrendAnalysis:
    """Trend analysis for a single metric across match windows.

    Compares: current match vs recent (last 5) vs historical (all matches).
    """

    metric_name: str
    current_value: float
    recent_avg: float  # Last RECENT_WINDOW matches
    historical_avg: float  # All matches
    recent_std: float = 0.0
    historical_std: float = 0.0
    direction: str = "stable"  # "improving", "declining", "stable"
    change_pct: float = 0.0  # % change from historical to current
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "metric": self.metric_name,
            "current": round(self.current_value, 2),
            "recent_avg": round(self.recent_avg, 2),
            "historical_avg": round(self.historical_avg, 2),
            "recent_std": round(self.recent_std, 2),
            "historical_std": round(self.historical_std, 2),
            "direction": self.direction,
            "change_pct": round(self.change_pct, 1),
            "sample_count": self.sample_count,
        }


@dataclass
class RoleBenchmark:
    """Benchmark comparison for a metric against competitive levels."""

    metric_name: str
    player_value: float
    level: str  # "beginner", "intermediate", "advanced", "elite"
    level_avg: float
    level_low: float
    level_high: float
    percentile_in_level: float  # 0-100, where in this level the player falls
    verdict: str  # "below", "at", "above" relative to level avg

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "metric": self.metric_name,
            "player_value": round(self.player_value, 2),
            "level": self.level,
            "level_avg": round(self.level_avg, 2),
            "level_range": [round(self.level_low, 2), round(self.level_high, 2)],
            "percentile_in_level": round(self.percentile_in_level, 1),
            "verdict": self.verdict,
        }


@dataclass
class PracticeRecommendation:
    """A specific practice recommendation based on trend analysis."""

    area: str  # "aim", "utility", "positioning", "economy", "trading", "entry"
    priority: str  # "high", "medium", "low"
    description: str
    current_value: float
    target_value: float
    drill: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "area": self.area,
            "priority": self.priority,
            "description": self.description,
            "current_value": round(self.current_value, 2),
            "target_value": round(self.target_value, 2),
            "drill": self.drill,
        }


@dataclass
class DevelopmentReport:
    """Comprehensive player development report across matches."""

    steam_id: str
    match_count: int
    date_range: tuple[str, str] | None = None  # (earliest, latest) ISO dates
    current_snapshot: MatchSnapshot | None = None
    trends: list[TrendAnalysis] = field(default_factory=list)
    benchmarks: list[RoleBenchmark] = field(default_factory=list)
    recommendations: list[PracticeRecommendation] = field(default_factory=list)
    estimated_level: str = "intermediate"
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improvement_velocity: float = 0.0  # -1.0 to +1.0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "steam_id": self.steam_id,
            "match_count": self.match_count,
            "date_range": list(self.date_range) if self.date_range else None,
            "current_snapshot": self.current_snapshot.to_dict() if self.current_snapshot else None,
            "trends": [t.to_dict() for t in self.trends],
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "estimated_level": self.estimated_level,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_velocity": round(self.improvement_velocity, 2),
            "summary": self.summary,
        }


# =============================================================================
# Constants
# =============================================================================

# Metrics to track for trend analysis
# Maps: metric name -> (MatchSnapshot attr, higher_is_better)
TRACKED_METRICS: dict[str, tuple[str, bool]] = {
    "hltv_rating": ("hltv_rating", True),
    "adr": ("adr", True),
    "kast": ("kast", True),
    "hs_pct": ("hs_pct", True),
    "aim_rating": ("aim_rating", True),
    "utility_rating": ("utility_rating", True),
    "kills": ("kills", True),
    "deaths": ("deaths", False),  # Lower is better
}

# History column names to track (for DB-based trend analysis)
TRACKED_HISTORY_METRICS: list[str] = [
    "kills",
    "deaths",
    "adr",
    "kast",
    "hs_pct",
    "hltv_rating",
    "aim_rating",
    "utility_rating",
    "trade_kill_success",
    "trade_kill_attempts",
    "entry_success",
    "entry_attempts",
    "clutch_wins",
    "clutch_situations",
    "he_damage",
    "enemies_flashed",
    "flash_assists",
    "ttd_median_ms",
    "cp_median_deg",
]

# Metrics where lower is better
LOWER_IS_BETTER = {"deaths", "ttd_median_ms", "cp_median_deg"}

# Threshold for trend detection (5% relative change)
TREND_THRESHOLD = 0.05

# Recent window size for trend comparison
RECENT_WINDOW = 5

# Minimum matches required for meaningful analysis
MIN_MATCHES = 3

# Competitive level benchmarks for CS2
# Based on industry data: ESEA ranks, FACEIT levels, community averages
LEVEL_BENCHMARKS: dict[str, dict[str, dict[str, float]]] = {
    "beginner": {
        "hltv_rating": {"low": 0.0, "avg": 0.75, "high": 0.90},
        "adr": {"low": 0.0, "avg": 55.0, "high": 70.0},
        "kast": {"low": 0.0, "avg": 55.0, "high": 65.0},
        "hs_pct": {"low": 0.0, "avg": 30.0, "high": 40.0},
        "entry_win_rate": {"low": 0.0, "avg": 35.0, "high": 45.0},
        "clutch_win_rate": {"low": 0.0, "avg": 10.0, "high": 20.0},
        "trade_rate": {"low": 0.0, "avg": 30.0, "high": 40.0},
        "utility_rating": {"low": 0.0, "avg": 25.0, "high": 40.0},
    },
    "intermediate": {
        "hltv_rating": {"low": 0.80, "avg": 1.00, "high": 1.15},
        "adr": {"low": 60.0, "avg": 75.0, "high": 85.0},
        "kast": {"low": 60.0, "avg": 68.0, "high": 75.0},
        "hs_pct": {"low": 35.0, "avg": 42.0, "high": 50.0},
        "entry_win_rate": {"low": 40.0, "avg": 48.0, "high": 55.0},
        "clutch_win_rate": {"low": 15.0, "avg": 22.0, "high": 30.0},
        "trade_rate": {"low": 35.0, "avg": 45.0, "high": 55.0},
        "utility_rating": {"low": 30.0, "avg": 45.0, "high": 55.0},
    },
    "advanced": {
        "hltv_rating": {"low": 1.00, "avg": 1.15, "high": 1.30},
        "adr": {"low": 75.0, "avg": 85.0, "high": 95.0},
        "kast": {"low": 68.0, "avg": 73.0, "high": 80.0},
        "hs_pct": {"low": 42.0, "avg": 48.0, "high": 55.0},
        "entry_win_rate": {"low": 48.0, "avg": 52.0, "high": 58.0},
        "clutch_win_rate": {"low": 20.0, "avg": 28.0, "high": 35.0},
        "trade_rate": {"low": 45.0, "avg": 52.0, "high": 60.0},
        "utility_rating": {"low": 45.0, "avg": 55.0, "high": 65.0},
    },
    "elite": {
        "hltv_rating": {"low": 1.15, "avg": 1.30, "high": 1.50},
        "adr": {"low": 85.0, "avg": 92.0, "high": 105.0},
        "kast": {"low": 73.0, "avg": 78.0, "high": 85.0},
        "hs_pct": {"low": 48.0, "avg": 52.0, "high": 60.0},
        "entry_win_rate": {"low": 52.0, "avg": 55.0, "high": 62.0},
        "clutch_win_rate": {"low": 25.0, "avg": 32.0, "high": 40.0},
        "trade_rate": {"low": 50.0, "avg": 58.0, "high": 65.0},
        "utility_rating": {"low": 55.0, "avg": 65.0, "high": 75.0},
    },
}

# Persona ID -> role mapping (for practice recommendations)
PERSONA_ROLE_MAP: dict[str, str] = {
    "the_opener": "entry_fragger",
    "the_headhunter": "entry_fragger",
    "the_anchor": "anchor",
    "the_survivor": "anchor",
    "the_utility_master": "support",
    "the_flash_master": "support",
    "the_cleanup": "trader",
    "the_terminator": "trader",
    "the_lurker": "lurker",
    "the_damage_dealer": "fragger",
    "the_competitor": "fragger",
}

# Role-specific practice targets (what a good player in this role should hit)
ROLE_PRACTICE_TARGETS: dict[str, dict[str, float]] = {
    "entry_fragger": {
        "hs_pct": 55.0,
        "entry_success_rate": 55.0,
        "adr": 80.0,
        "hltv_rating": 1.10,
        "ttd_median_ms": 300.0,
        "cp_median_deg": 10.0,
    },
    "anchor": {
        "kast": 75.0,
        "clutch_success_rate": 30.0,
        "adr": 70.0,
        "hltv_rating": 1.05,
        "ttd_median_ms": 350.0,
        "cp_median_deg": 12.0,
    },
    "support": {
        "flash_assists": 3.0,
        "enemies_flashed": 8.0,
        "he_damage": 30.0,
        "kast": 72.0,
        "hltv_rating": 1.0,
        "adr": 65.0,
    },
    "trader": {
        "trade_success_rate": 55.0,
        "adr": 75.0,
        "kast": 72.0,
        "hltv_rating": 1.05,
        "ttd_median_ms": 320.0,
        "cp_median_deg": 11.0,
    },
    "lurker": {
        "adr": 80.0,
        "deaths": 15.0,  # lower is better
        "hltv_rating": 1.10,
        "kast": 72.0,
        "ttd_median_ms": 300.0,
        "cp_median_deg": 10.0,
    },
    "fragger": {
        "kills": 22.0,
        "adr": 85.0,
        "hltv_rating": 1.15,
        "hs_pct": 50.0,
        "ttd_median_ms": 300.0,
        "cp_median_deg": 10.0,
    },
}


# =============================================================================
# Player Tracker Engine
# =============================================================================


class PlayerTracker:
    """
    Cross-match player development tracker.

    Uses the existing DatabaseManager and MatchHistory table for storage.
    Provides snapshot extraction, trend analysis, level benchmarks,
    practice recommendations, and full development reports.
    """

    def __init__(self, db: Any | None = None) -> None:
        """Initialize with optional DatabaseManager instance.

        Args:
            db: DatabaseManager instance. If None, uses get_db().
        """
        self._db = db

    @property
    def db(self) -> Any:
        """Lazy-load database manager."""
        if self._db is None:
            from opensight.infra.database import get_db

            self._db = get_db()
        return self._db

    # =========================================================================
    # Snapshot Extraction
    # =========================================================================

    def extract_snapshot(
        self,
        orchestrator_result: dict[str, Any],
        steam_id: str,
    ) -> MatchSnapshot | None:
        """Extract a MatchSnapshot from an orchestrator result for a specific player.

        Args:
            orchestrator_result: Full orchestrator output dict
            steam_id: Player's Steam ID

        Returns:
            MatchSnapshot if player found, None otherwise
        """
        players = orchestrator_result.get("players", {})
        player = players.get(steam_id)
        if player is None:
            return None

        stats = player.get("stats", {})
        rating = player.get("rating", {})
        advanced = player.get("advanced", {})
        duels = player.get("duels", {})
        utility = player.get("utility", {})
        demo_info = orchestrator_result.get("demo_info", {})

        return MatchSnapshot(
            steam_id=steam_id,
            demo_hash=demo_info.get("demo_hash", ""),
            map_name=demo_info.get("map_name"),
            result=None,  # Determined after score comparison
            kills=stats.get("kills", 0),
            deaths=stats.get("deaths", 0),
            assists=stats.get("assists", 0),
            adr=stats.get("adr", 0.0),
            kast=rating.get("kast_percentage", 0.0),
            hs_pct=stats.get("headshot_pct", 0.0),
            rounds_played=stats.get("rounds_played", 0),
            hltv_rating=rating.get("hltv_rating", 1.0),
            aim_rating=rating.get("aim_rating", 0.0),
            utility_rating=rating.get("utility_rating", 0.0),
            ttd_median_ms=advanced.get("ttd_median_ms"),
            cp_median_deg=advanced.get("cp_median_error_deg"),
            entry_attempts=duels.get("opening_kills", 0) + duels.get("opening_deaths", 0),
            entry_success=duels.get("opening_kills", 0),
            clutch_situations=duels.get("clutch_attempts", 0),
            clutch_wins=duels.get("clutch_wins", 0),
            trade_kill_success=duels.get("trade_kills", 0),
            trade_kill_attempts=duels.get("trade_kill_opportunities", 0),
            enemies_flashed=utility.get("enemies_flashed", 0),
            flash_assists=utility.get("flash_assists", 0),
            he_damage=utility.get("he_damage", 0),
        )

    def snapshot_from_history(self, history_row: dict[str, Any]) -> MatchSnapshot:
        """Convert a get_player_history_full() row to a MatchSnapshot.

        Args:
            history_row: Dict from DatabaseManager.get_player_history_full()

        Returns:
            MatchSnapshot
        """
        return MatchSnapshot(
            steam_id=history_row.get("steam_id", ""),
            demo_hash=history_row.get("demo_hash", ""),
            map_name=history_row.get("map_name"),
            result=history_row.get("result"),
            analyzed_at=history_row.get("analyzed_at"),
            kills=history_row.get("kills", 0),
            deaths=history_row.get("deaths", 0),
            assists=history_row.get("assists", 0),
            adr=history_row.get("adr", 0.0),
            kast=history_row.get("kast", 0.0),
            hs_pct=history_row.get("hs_pct", 0.0),
            rounds_played=history_row.get("rounds_played", 0),
            hltv_rating=history_row.get("hltv_rating", 1.0),
            aim_rating=history_row.get("aim_rating", 0.0),
            utility_rating=history_row.get("utility_rating", 0.0),
            ttd_median_ms=history_row.get("ttd_median_ms"),
            cp_median_deg=history_row.get("cp_median_deg"),
            entry_attempts=history_row.get("entry_attempts", 0),
            entry_success=history_row.get("entry_success", 0),
            clutch_situations=history_row.get("clutch_situations", 0),
            clutch_wins=history_row.get("clutch_wins", 0),
            trade_kill_success=history_row.get("trade_kill_success", 0),
            trade_kill_attempts=history_row.get("trade_kill_attempts", 0),
            enemies_flashed=history_row.get("enemies_flashed", 0),
            flash_assists=history_row.get("flash_assists", 0),
            he_damage=history_row.get("he_damage", 0),
        )

    # =========================================================================
    # Trend Analysis
    # =========================================================================

    def analyze_trends(
        self,
        snapshots: list[MatchSnapshot],
        current: MatchSnapshot | None = None,
    ) -> list[TrendAnalysis]:
        """Analyze metric trends across match snapshots.

        Compares current match performance against:
        - Recent window (last RECENT_WINDOW matches)
        - Historical average (all matches)

        Args:
            snapshots: List of MatchSnapshots ordered oldest-first
            current: Optional current match snapshot (if not already in list)

        Returns:
            List of TrendAnalysis for each tracked metric
        """
        if not snapshots:
            return []

        all_snapshots = list(snapshots)
        if current is not None:
            all_snapshots.append(current)

        if len(all_snapshots) < MIN_MATCHES:
            return []

        trends = []
        for metric_name, (attr_name, higher_is_better) in TRACKED_METRICS.items():
            values = []
            for s in all_snapshots:
                v = getattr(s, attr_name, None)
                if v is not None:
                    values.append(float(v))

            if len(values) < MIN_MATCHES:
                continue

            current_value = values[-1]
            recent_values = values[-RECENT_WINDOW:]
            historical_values = values

            recent_avg = sum(recent_values) / len(recent_values)
            historical_avg = sum(historical_values) / len(historical_values)
            recent_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
            historical_std = (
                statistics.stdev(historical_values) if len(historical_values) > 1 else 0.0
            )

            direction = _compute_trend_direction(recent_avg, historical_avg, higher_is_better)

            change_pct = 0.0
            if historical_avg != 0:
                change_pct = ((current_value - historical_avg) / abs(historical_avg)) * 100

            trends.append(
                TrendAnalysis(
                    metric_name=metric_name,
                    current_value=current_value,
                    recent_avg=recent_avg,
                    historical_avg=historical_avg,
                    recent_std=recent_std,
                    historical_std=historical_std,
                    direction=direction,
                    change_pct=change_pct,
                    sample_count=len(historical_values),
                )
            )

        return trends

    def analyze_trends_from_db(
        self, steam_id: str, min_matches: int = MIN_MATCHES
    ) -> list[TrendAnalysis]:
        """Analyze trends directly from DB history (convenience method).

        Args:
            steam_id: Player's Steam ID
            min_matches: Minimum matches required

        Returns:
            List of TrendAnalysis
        """
        history_rows = self.db.get_player_history_full(steam_id, limit=30)
        if len(history_rows) < min_matches:
            return []

        # History is newest-first; reverse for chronological order
        snapshots = [self.snapshot_from_history(row) for row in reversed(history_rows)]
        return self.analyze_trends(snapshots)

    # =========================================================================
    # Level Estimation & Benchmarking
    # =========================================================================

    def estimate_level(self, snapshots: list[MatchSnapshot]) -> str:
        """Estimate a player's competitive level from their match history.

        Uses HLTV rating as the primary indicator:
        - <0.85: beginner
        - 0.85-1.05: intermediate
        - 1.05-1.20: advanced
        - >1.20: elite

        Args:
            snapshots: List of MatchSnapshots

        Returns:
            Level string: "beginner", "intermediate", "advanced", "elite"
        """
        if not snapshots:
            return "intermediate"

        ratings = [s.hltv_rating for s in snapshots if s.hltv_rating is not None]
        if not ratings:
            return "intermediate"

        avg_rating = sum(ratings) / len(ratings)

        if avg_rating < 0.85:
            return "beginner"
        elif avg_rating < 1.05:
            return "intermediate"
        elif avg_rating < 1.20:
            return "advanced"
        return "elite"

    def compute_benchmarks(
        self,
        snapshots: list[MatchSnapshot],
        level: str | None = None,
    ) -> list[RoleBenchmark]:
        """Compare player metrics against competitive level benchmarks.

        Args:
            snapshots: List of MatchSnapshots
            level: Competitive level to compare against.
                   If None, auto-estimated from snapshots.

        Returns:
            List of RoleBenchmark comparisons
        """
        if not snapshots:
            return []

        if level is None:
            level = self.estimate_level(snapshots)

        if level not in LEVEL_BENCHMARKS:
            level = "intermediate"

        level_benchmarks = LEVEL_BENCHMARKS[level]
        benchmarks = []

        player_avgs = _calculate_player_averages(snapshots)

        for metric_name, bench in level_benchmarks.items():
            player_value = player_avgs.get(metric_name)
            if player_value is None:
                continue

            level_low = bench["low"]
            level_avg = bench["avg"]
            level_high = bench["high"]

            # Calculate percentile within level range
            level_range = level_high - level_low
            if level_range > 0:
                percentile = ((player_value - level_low) / level_range) * 100
                percentile = max(0.0, min(100.0, percentile))
            else:
                percentile = 50.0

            # Determine verdict
            if player_value >= level_avg * 1.05:
                verdict = "above"
            elif player_value <= level_avg * 0.95:
                verdict = "below"
            else:
                verdict = "at"

            benchmarks.append(
                RoleBenchmark(
                    metric_name=metric_name,
                    player_value=player_value,
                    level=level,
                    level_avg=level_avg,
                    level_low=level_low,
                    level_high=level_high,
                    percentile_in_level=percentile,
                    verdict=verdict,
                )
            )

        return benchmarks

    # =========================================================================
    # Practice Recommendations
    # =========================================================================

    def generate_recommendations(
        self,
        steam_id: str,
        trends: list[TrendAnalysis] | None = None,
    ) -> list[PracticeRecommendation]:
        """Generate practice recommendations based on trends and role.

        Args:
            steam_id: Player's Steam ID
            trends: Pre-computed trends (if None, computed from DB)

        Returns:
            Sorted list of PracticeRecommendation
        """
        if trends is None:
            trends = self.analyze_trends_from_db(steam_id)

        if not trends:
            return []

        role = self._get_player_role(steam_id)
        targets = ROLE_PRACTICE_TARGETS.get(role, ROLE_PRACTICE_TARGETS["fragger"])

        # Also need DB averages for rate-based metrics
        history = self.db.get_player_history_full(steam_id, limit=30)
        averages = _compute_history_averages(history) if history else {}

        trend_map = {t.metric_name: t for t in trends}
        recs: list[PracticeRecommendation] = []

        # Rule 1: HS% below target
        hs = trend_map.get("hs_pct")
        target_hs = targets.get("hs_pct", 50.0)
        if hs and hs.recent_avg < target_hs * 0.85:
            recs.append(
                PracticeRecommendation(
                    area="aim",
                    priority="medium",
                    description="Headshot percentage below role target",
                    current_value=hs.recent_avg,
                    target_value=target_hs,
                    drill="Practice headshot-only deathmatch to build muscle memory",
                )
            )

        # Rule 2: KAST declining
        kast = trend_map.get("kast")
        if kast and kast.direction == "declining":
            recs.append(
                PracticeRecommendation(
                    area="positioning",
                    priority="high",
                    description="KAST is declining — less round impact",
                    current_value=kast.recent_avg,
                    target_value=targets.get("kast", 72.0),
                    drill="Focus on staying alive and getting at least one contribution per round",
                )
            )

        # Rule 3: ADR declining
        adr = trend_map.get("adr")
        if adr and adr.direction == "declining":
            recs.append(
                PracticeRecommendation(
                    area="aim",
                    priority="high",
                    description="Damage output is declining",
                    current_value=adr.recent_avg,
                    target_value=targets.get("adr", 80.0),
                    drill="Be more aggressive in engagements, use utility to deal damage",
                )
            )

        # Rule 4: Deaths increasing
        deaths = trend_map.get("deaths")
        if deaths and deaths.direction == "declining":  # "declining" = getting worse for deaths
            recs.append(
                PracticeRecommendation(
                    area="positioning",
                    priority="medium",
                    description="Dying more often — improve survival discipline",
                    current_value=deaths.recent_avg,
                    target_value=targets.get("deaths", 15.0),
                    drill="Focus on information gathering before peeking, use utility before engaging",
                )
            )

        # Rule 5: Entry success below target
        entry_rate = averages.get("entry_success_rate", 0)
        entry_target = targets.get("entry_success_rate", 55.0)
        if entry_rate > 0 and entry_rate < entry_target * 0.80:
            recs.append(
                PracticeRecommendation(
                    area="entry",
                    priority="medium",
                    description="Entry success rate below role target",
                    current_value=entry_rate,
                    target_value=entry_target,
                    drill="Practice entry routes with utility on specific maps",
                )
            )

        # Rule 6: Utility rating declining
        util = trend_map.get("utility_rating")
        if util and util.direction == "declining":
            recs.append(
                PracticeRecommendation(
                    area="utility",
                    priority="medium",
                    description="Utility effectiveness is declining",
                    current_value=util.recent_avg,
                    target_value=targets.get("utility_rating", 50.0)
                    if "utility_rating" in targets
                    else 50.0,
                    drill="Learn pop-flash lineups and HE/molotov spots for your most-played maps",
                )
            )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recs

    # =========================================================================
    # Development Report
    # =========================================================================

    def generate_report(
        self,
        steam_id: str,
        limit: int = 30,
    ) -> DevelopmentReport:
        """Generate a comprehensive development report for a player.

        Pulls match history from DB, analyzes trends, benchmarks against
        competitive levels, generates recommendations, and identifies
        strengths/weaknesses.

        Args:
            steam_id: Player's Steam ID (17 digits)
            limit: Max matches to analyze (default 30)

        Returns:
            DevelopmentReport
        """
        history_rows = self.db.get_player_history_full(steam_id, limit=limit)

        if not history_rows:
            return DevelopmentReport(steam_id=steam_id, match_count=0)

        # Convert to snapshots (history_rows are newest-first, reverse for oldest-first)
        snapshots = [self.snapshot_from_history(row) for row in reversed(history_rows)]

        current = snapshots[-1] if snapshots else None
        match_count = len(snapshots)

        # Date range
        latest = history_rows[0].get("analyzed_at", "unknown")
        earliest = history_rows[-1].get("analyzed_at", "unknown")
        date_range = (str(earliest), str(latest))

        # Analyze trends
        trends = self.analyze_trends(snapshots)

        # Estimate level and compute benchmarks
        level = self.estimate_level(snapshots)
        benchmarks = self.compute_benchmarks(snapshots, level)

        # Generate recommendations
        recommendations = self.generate_recommendations(steam_id, trends=trends)

        # Identify strengths and weaknesses
        strengths, weaknesses = _identify_strengths_weaknesses(benchmarks, trends)

        # Calculate improvement velocity
        improvement_velocity = _calculate_improvement_velocity(trends)

        # Build summary
        summary = _build_summary(trends, benchmarks, recommendations, match_count)

        return DevelopmentReport(
            steam_id=steam_id,
            match_count=match_count,
            date_range=date_range,
            current_snapshot=current,
            trends=trends,
            benchmarks=benchmarks,
            recommendations=recommendations,
            estimated_level=level,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_velocity=improvement_velocity,
            summary=summary,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_player_role(self, steam_id: str) -> str:
        """Get the player's role based on their persona."""
        try:
            persona = self.db.get_player_persona(steam_id)
            if persona:
                persona_id = persona.get("persona", "the_competitor")
                return PERSONA_ROLE_MAP.get(persona_id, "fragger")
        except Exception:
            logger.debug("Could not fetch persona for %s, defaulting to fragger", steam_id)
        return "fragger"


# =============================================================================
# Module-level Helpers
# =============================================================================


def _compute_trend_direction(
    recent_avg: float,
    historical_avg: float,
    higher_is_better: bool,
) -> str:
    """Determine trend direction by comparing recent vs historical average.

    Uses a 5% threshold relative to historical average to filter noise.
    """
    if historical_avg == 0:
        return "stable"

    pct_change = (recent_avg - historical_avg) / abs(historical_avg)

    if higher_is_better:
        if pct_change > TREND_THRESHOLD:
            return "improving"
        elif pct_change < -TREND_THRESHOLD:
            return "declining"
    else:
        # Lower is better (e.g., deaths)
        if pct_change < -TREND_THRESHOLD:
            return "improving"
        elif pct_change > TREND_THRESHOLD:
            return "declining"

    return "stable"


def _calculate_player_averages(
    snapshots: list[MatchSnapshot],
) -> dict[str, float]:
    """Calculate average metrics from snapshots for benchmarking."""
    if not snapshots:
        return {}

    n = len(snapshots)
    avgs: dict[str, float] = {}

    # Core averages
    avgs["hltv_rating"] = sum(s.hltv_rating for s in snapshots) / n
    avgs["adr"] = sum(s.adr for s in snapshots) / n
    avgs["kast"] = sum(s.kast for s in snapshots) / n
    avgs["hs_pct"] = sum(s.hs_pct for s in snapshots) / n

    # Utility rating (skip 0 values = missing data)
    util_values = [s.utility_rating for s in snapshots if s.utility_rating > 0]
    if util_values:
        avgs["utility_rating"] = sum(util_values) / len(util_values)

    # Entry win rate
    total_attempts = sum(s.entry_attempts for s in snapshots)
    total_success = sum(s.entry_success for s in snapshots)
    if total_attempts > 0:
        avgs["entry_win_rate"] = (total_success / total_attempts) * 100

    # Clutch win rate
    total_clutch_sit = sum(s.clutch_situations for s in snapshots)
    total_clutch_wins = sum(s.clutch_wins for s in snapshots)
    if total_clutch_sit > 0:
        avgs["clutch_win_rate"] = (total_clutch_wins / total_clutch_sit) * 100

    # Trade rate
    total_trade_attempts = sum(s.trade_kill_attempts for s in snapshots)
    total_trade_success = sum(s.trade_kill_success for s in snapshots)
    if total_trade_attempts > 0:
        avgs["trade_rate"] = (total_trade_success / total_trade_attempts) * 100

    return avgs


def _compute_history_averages(history: list[dict]) -> dict[str, float]:
    """Compute metric averages from match history dicts, including derived rates."""
    n = len(history) or 1
    totals: dict[str, float] = {}

    for h in history:
        for metric in TRACKED_HISTORY_METRICS:
            val = float(h.get(metric, 0) or 0)
            totals[metric] = totals.get(metric, 0) + val

    averages = {m: round(totals.get(m, 0) / n, 2) for m in TRACKED_HISTORY_METRICS}

    # Derived rates
    total_entry_attempts = totals.get("entry_attempts", 0)
    if total_entry_attempts > 0:
        averages["entry_success_rate"] = round(
            (totals.get("entry_success", 0) / total_entry_attempts) * 100, 1
        )
    else:
        averages["entry_success_rate"] = 0.0

    total_clutch_situations = totals.get("clutch_situations", 0)
    if total_clutch_situations > 0:
        averages["clutch_success_rate"] = round(
            (totals.get("clutch_wins", 0) / total_clutch_situations) * 100, 1
        )
    else:
        averages["clutch_success_rate"] = 0.0

    total_trade_attempts = totals.get("trade_kill_attempts", 0)
    if total_trade_attempts > 0:
        averages["trade_success_rate"] = round(
            (totals.get("trade_kill_success", 0) / total_trade_attempts) * 100, 1
        )
    else:
        averages["trade_success_rate"] = 0.0

    return averages


def _identify_strengths_weaknesses(
    benchmarks: list[RoleBenchmark],
    trends: list[TrendAnalysis],
) -> tuple[list[str], list[str]]:
    """Identify player strengths and weaknesses from benchmarks and trends."""
    strengths: list[str] = []
    weaknesses: list[str] = []

    metric_labels = {
        "hltv_rating": "HLTV Rating",
        "adr": "ADR",
        "kast": "KAST",
        "hs_pct": "Headshot %",
        "entry_win_rate": "Entry Duels",
        "clutch_win_rate": "Clutch Play",
        "trade_rate": "Trading",
        "utility_rating": "Utility Usage",
        "aim_rating": "Aim",
        "kills": "Kill Count",
        "deaths": "Survivability",
    }

    for b in benchmarks:
        label = metric_labels.get(b.metric_name, b.metric_name)
        if b.verdict == "above":
            strengths.append(
                f"{label} above {b.level} average ({b.player_value:.1f} vs {b.level_avg:.1f})"
            )
        elif b.verdict == "below":
            weaknesses.append(
                f"{label} below {b.level} average ({b.player_value:.1f} vs {b.level_avg:.1f})"
            )

    for t in trends:
        label = metric_labels.get(t.metric_name, t.metric_name)
        if t.direction == "improving" and abs(t.change_pct) > 5:
            strengths.append(f"{label} trending up ({t.change_pct:+.1f}%)")
        elif t.direction == "declining" and abs(t.change_pct) > 5:
            weaknesses.append(f"{label} trending down ({t.change_pct:+.1f}%)")

    return strengths, weaknesses


def _calculate_improvement_velocity(trends: list[TrendAnalysis]) -> float:
    """Calculate overall improvement velocity from trends.

    Returns:
        Float from -1.0 (all declining) to +1.0 (all improving)
    """
    if not trends:
        return 0.0

    direction_scores = []
    for t in trends:
        if t.direction == "improving":
            direction_scores.append(1.0)
        elif t.direction == "declining":
            direction_scores.append(-1.0)
        else:
            direction_scores.append(0.0)

    return sum(direction_scores) / len(direction_scores)


def _build_summary(
    trends: list[TrendAnalysis],
    benchmarks: list[RoleBenchmark],
    recommendations: list[PracticeRecommendation],
    match_count: int,
) -> str:
    """Build a human-readable summary string."""
    improving = [t for t in trends if t.direction == "improving"]
    declining = [t for t in trends if t.direction == "declining"]

    parts = [f"Based on {match_count} matches analyzed:"]

    if improving:
        names = ", ".join(t.metric_name.replace("_", " ") for t in improving[:3])
        parts.append(f"Improving in: {names}.")

    if declining:
        names = ", ".join(t.metric_name.replace("_", " ") for t in declining[:3])
        parts.append(f"Declining in: {names}.")

    if not improving and not declining:
        parts.append("Performance is stable across all metrics.")

    weak = [b for b in benchmarks if b.verdict == "below"]
    if weak:
        names = ", ".join(b.metric_name.replace("_", " ") for b in weak[:3])
        parts.append(f"Below benchmark in: {names}.")

    high_recs = [r for r in recommendations if r.priority == "high"]
    if high_recs:
        parts.append(f"{len(high_recs)} high-priority area(s) to focus on.")

    return " ".join(parts)


# =============================================================================
# Singleton Access
# =============================================================================

_tracker: PlayerTracker | None = None


def get_player_tracker() -> PlayerTracker:
    """Get the singleton PlayerTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PlayerTracker()
    return _tracker
