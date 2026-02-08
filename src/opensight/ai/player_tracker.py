"""Cross-match player development tracking.

Analyzes player performance across multiple demos to identify improvement
trends, compare against role benchmarks, and generate practice recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from opensight.infra.database import DatabaseManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricTrend:
    """Trend analysis for a single metric across matches."""

    metric: str
    values: list[float]
    direction: str  # "improving", "declining", "stable"
    change_pct: float
    recent_avg: float
    overall_avg: float


@dataclass
class RoleBenchmark:
    """Role-based benchmark comparison for a player."""

    role: str
    benchmarks: dict[str, float]  # metric -> target value
    percentiles: dict[str, float]  # metric -> player percentile (0-100)


@dataclass
class PracticeRecommendation:
    """A specific practice recommendation based on trend analysis."""

    area: str  # "aim", "utility", "positioning", "economy", "trading", "entry"
    priority: str  # "high", "medium", "low"
    description: str
    current_value: float
    target_value: float
    drill: str


@dataclass
class DevelopmentReport:
    """Full development report combining all analyses."""

    steam_id: str
    match_count: int
    date_range: tuple[str, str]  # (earliest, latest) ISO dates
    trends: list[MetricTrend]
    role_benchmark: RoleBenchmark | None
    recommendations: list[PracticeRecommendation]
    summary: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Metrics to track for trend analysis (must match match_history column names)
TRACKED_METRICS: list[str] = [
    "kills",
    "deaths",
    "adr",
    "kast",
    "hs_pct",
    "hltv_rating",
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

# Threshold for trend detection (5%)
TREND_THRESHOLD = 0.05

# Recent window size for trend comparison
RECENT_WINDOW = 5

# Minimum matches required for meaningful analysis
MIN_MATCHES = 5

# Persona ID -> role mapping
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

# Role benchmark targets (what a good player in this role should hit)
ROLE_BENCHMARKS: dict[str, dict[str, float]] = {
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


# ---------------------------------------------------------------------------
# PlayerTracker
# ---------------------------------------------------------------------------


class PlayerTracker:
    """Tracks player development across multiple matches."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def analyze_trends(self, steam_id: str, min_matches: int = MIN_MATCHES) -> list[MetricTrend]:
        """Analyze metric trends across a player's match history.

        Compares recent performance (last RECENT_WINDOW matches) against
        overall averages. Returns empty list if insufficient data.
        """
        history = self.db.get_player_history_full(steam_id, limit=30)
        if len(history) < min_matches:
            return []

        # History is ordered newest-first; reverse for chronological order
        history = list(reversed(history))

        trends: list[MetricTrend] = []
        for metric in TRACKED_METRICS:
            values = [float(h.get(metric, 0) or 0) for h in history]

            # Skip metrics that are all zero
            if all(v == 0 for v in values):
                continue

            overall_avg = sum(values) / len(values) if values else 0
            recent = values[-RECENT_WINDOW:]
            recent_avg = sum(recent) / len(recent) if recent else 0

            # Determine direction
            direction, change_pct = _compute_direction(metric, recent_avg, overall_avg)

            trends.append(
                MetricTrend(
                    metric=metric,
                    values=values,
                    direction=direction,
                    change_pct=round(change_pct, 1),
                    recent_avg=round(recent_avg, 2),
                    overall_avg=round(overall_avg, 2),
                )
            )

        return trends

    def get_role_benchmarks(self, steam_id: str) -> RoleBenchmark | None:
        """Compare player's metrics against their role's benchmarks.

        Uses the player's persona to determine role, then computes
        percentile rankings against those benchmarks.
        """
        history = self.db.get_player_history_full(steam_id, limit=30)
        if len(history) < MIN_MATCHES:
            return None

        # Get player persona
        role = self._get_player_role(steam_id)
        benchmarks = ROLE_BENCHMARKS.get(role, ROLE_BENCHMARKS["fragger"])

        # Compute player averages
        averages = _compute_averages(history)

        # Compute percentiles (how close player is to benchmark)
        percentiles: dict[str, float] = {}
        for metric, target in benchmarks.items():
            player_val = averages.get(metric, 0)
            if target == 0:
                percentiles[metric] = 100.0
            elif metric == "deaths":
                # Lower is better for deaths
                percentiles[metric] = round(min(100.0, (target / max(player_val, 0.1)) * 100), 1)
            elif metric in ("ttd_median_ms", "cp_median_deg"):
                # Lower is better for TTD and CP
                percentiles[metric] = round(min(100.0, (target / max(player_val, 0.1)) * 100), 1)
            else:
                # Higher is better for most metrics
                percentiles[metric] = round(min(100.0, (player_val / target) * 100), 1)

        return RoleBenchmark(
            role=role,
            benchmarks=benchmarks,
            percentiles=percentiles,
        )

    def generate_recommendations(self, steam_id: str) -> list[PracticeRecommendation]:
        """Generate practice recommendations based on trends and benchmarks."""
        trends = self.analyze_trends(steam_id)
        benchmark = self.get_role_benchmarks(steam_id)
        if not trends:
            return []

        trend_map = {t.metric: t for t in trends}
        bench_pcts = benchmark.percentiles if benchmark else {}
        bench_targets = benchmark.benchmarks if benchmark else {}

        recs: list[PracticeRecommendation] = []

        # Rule 1: CP declining → aim practice
        cp = trend_map.get("cp_median_deg")
        if cp and cp.direction == "declining":
            # For CP, "declining" means getting worse (higher degrees)
            recs.append(
                PracticeRecommendation(
                    area="aim",
                    priority="high",
                    description="Crosshair placement is getting worse — focus on pre-aiming common angles",
                    current_value=cp.recent_avg,
                    target_value=bench_targets.get("cp_median_deg", 10.0),
                    drill="Practice pre-aim routines on workshop maps (Aim Botz, Reflex)",
                )
            )

        # Rule 2: TTD declining → reaction time
        ttd = trend_map.get("ttd_median_ms")
        if ttd and ttd.direction == "declining":
            recs.append(
                PracticeRecommendation(
                    area="aim",
                    priority="high",
                    description="Time to damage is increasing — reactions are slowing down",
                    current_value=ttd.recent_avg,
                    target_value=bench_targets.get("ttd_median_ms", 300.0),
                    drill="Warm up with fast-paced aim trainers (Aim Lab, Kovaak's)",
                )
            )

        # Rule 3: HS% below role benchmark
        hs = trend_map.get("hs_pct")
        if hs and bench_pcts.get("hs_pct", 100) < 80:
            recs.append(
                PracticeRecommendation(
                    area="aim",
                    priority="medium",
                    description="Headshot percentage is below your role's benchmark",
                    current_value=hs.recent_avg,
                    target_value=bench_targets.get("hs_pct", 50.0),
                    drill="Practice headshot-only deathmatch to build muscle memory",
                )
            )

        # Rule 4: Trade kills declining
        tk = trend_map.get("trade_kill_success")
        if tk and tk.direction == "declining":
            recs.append(
                PracticeRecommendation(
                    area="trading",
                    priority="high",
                    description="Trade kill success is dropping — work on positioning near teammates",
                    current_value=tk.recent_avg,
                    target_value=bench_targets.get("trade_success_rate", 55.0),
                    drill="Focus on staying close enough to trade in executes and retakes",
                )
            )

        # Rule 5: Entry success below benchmark
        if bench_pcts.get("entry_success_rate", 100) < 75:
            entry = trend_map.get("entry_success")
            recs.append(
                PracticeRecommendation(
                    area="entry",
                    priority="medium",
                    description="Entry success rate is below your role's target",
                    current_value=entry.recent_avg if entry else 0,
                    target_value=bench_targets.get("entry_success_rate", 55.0),
                    drill="Practice entry routes with utility on specific maps",
                )
            )

        # Rule 6: Flash assists declining
        fa = trend_map.get("flash_assists")
        if fa and fa.direction == "declining":
            recs.append(
                PracticeRecommendation(
                    area="utility",
                    priority="medium",
                    description="Flash assists are declining — flashbangs are less effective",
                    current_value=fa.recent_avg,
                    target_value=bench_targets.get("flash_assists", 3.0),
                    drill="Learn pop-flash lineups for your most-played maps",
                )
            )

        # Rule 7: Utility damage below benchmark
        if bench_pcts.get("he_damage", 100) < 70:
            he = trend_map.get("he_damage")
            recs.append(
                PracticeRecommendation(
                    area="utility",
                    priority="low",
                    description="HE grenade damage is below your role's benchmark",
                    current_value=he.recent_avg if he else 0,
                    target_value=bench_targets.get("he_damage", 30.0),
                    drill="Learn HE lineups for common plant spots and chokepoints",
                )
            )

        # Rule 8: KAST declining
        kast = trend_map.get("kast")
        if kast and kast.direction == "declining":
            recs.append(
                PracticeRecommendation(
                    area="positioning",
                    priority="high",
                    description="KAST is declining — you're having less round impact",
                    current_value=kast.recent_avg,
                    target_value=bench_targets.get("kast", 72.0),
                    drill="Focus on staying alive and getting at least one contribution per round",
                )
            )

        # Rule 9: Clutch wins below benchmark
        if bench_pcts.get("clutch_success_rate", 100) < 70:
            cw = trend_map.get("clutch_wins")
            recs.append(
                PracticeRecommendation(
                    area="positioning",
                    priority="low",
                    description="Clutch win rate is below your role's target",
                    current_value=cw.recent_avg if cw else 0,
                    target_value=bench_targets.get("clutch_success_rate", 30.0),
                    drill="Review clutch round demos — focus on time management and info gathering",
                )
            )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recs

    def get_development_report(self, steam_id: str) -> DevelopmentReport | None:
        """Generate a complete development report for a player.

        Returns None if insufficient match history.
        """
        history = self.db.get_player_history_full(steam_id, limit=30)
        if len(history) < MIN_MATCHES:
            return None

        # Date range (history is newest-first)
        latest = history[0].get("analyzed_at", "unknown")
        earliest = history[-1].get("analyzed_at", "unknown")

        trends = self.analyze_trends(steam_id)
        benchmark = self.get_role_benchmarks(steam_id)
        recommendations = self.generate_recommendations(steam_id)

        summary = _build_summary(trends, benchmark, recommendations, len(history))

        return DevelopmentReport(
            steam_id=steam_id,
            match_count=len(history),
            date_range=(str(earliest), str(latest)),
            trends=trends,
            role_benchmark=benchmark,
            recommendations=recommendations,
            summary=summary,
        )

    def _get_player_role(self, steam_id: str) -> str:
        """Get the player's role based on their persona."""
        try:
            from opensight.infra.database import get_db

            db = get_db()
            session = db.get_session()
            try:
                from opensight.infra.database import PlayerPersona

                persona = (
                    session.query(PlayerPersona).filter(PlayerPersona.steam_id == steam_id).first()
                )
                if persona and persona.persona_id:
                    return PERSONA_ROLE_MAP.get(persona.persona_id, "fragger")
            finally:
                session.close()
        except Exception:
            logger.debug("Could not fetch persona for %s, defaulting to fragger", steam_id)
        return "fragger"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_direction(metric: str, recent_avg: float, overall_avg: float) -> tuple[str, float]:
    """Compute trend direction and change percentage.

    For metrics where lower is better (deaths, ttd, cp), the direction
    logic is inverted.
    """
    if overall_avg == 0:
        return ("stable", 0.0)

    change_pct = ((recent_avg - overall_avg) / abs(overall_avg)) * 100

    # Metrics where lower is better
    lower_is_better = {"deaths", "ttd_median_ms", "cp_median_deg"}

    if metric in lower_is_better:
        # For these, recent < overall = improving
        if change_pct < -TREND_THRESHOLD * 100:
            return ("improving", change_pct)
        elif change_pct > TREND_THRESHOLD * 100:
            return ("declining", change_pct)
    else:
        # Higher is better
        if change_pct > TREND_THRESHOLD * 100:
            return ("improving", change_pct)
        elif change_pct < -TREND_THRESHOLD * 100:
            return ("declining", change_pct)

    return ("stable", change_pct)


def _compute_averages(history: list[dict]) -> dict[str, float]:
    """Compute metric averages from match history, including derived rates."""
    n = len(history) or 1
    totals: dict[str, float] = {}

    for h in history:
        for metric in TRACKED_METRICS:
            val = float(h.get(metric, 0) or 0)
            totals[metric] = totals.get(metric, 0) + val

    averages = {m: round(totals.get(m, 0) / n, 2) for m in TRACKED_METRICS}

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


def _build_summary(
    trends: list[MetricTrend],
    benchmark: RoleBenchmark | None,
    recommendations: list[PracticeRecommendation],
    match_count: int,
) -> str:
    """Build a human-readable summary string."""
    improving = [t for t in trends if t.direction == "improving"]
    declining = [t for t in trends if t.direction == "declining"]

    parts = [f"Based on {match_count} matches analyzed:"]

    if improving:
        names = ", ".join(t.metric.replace("_", " ") for t in improving[:3])
        parts.append(f"Improving in: {names}.")

    if declining:
        names = ", ".join(t.metric.replace("_", " ") for t in declining[:3])
        parts.append(f"Declining in: {names}.")

    if not improving and not declining:
        parts.append("Performance is stable across all metrics.")

    if benchmark:
        weak = [m for m, p in benchmark.percentiles.items() if p < 70]
        if weak:
            names = ", ".join(m.replace("_", " ") for m in weak[:3])
            parts.append(f"Below role benchmark in: {names}.")

    if recommendations:
        high = [r for r in recommendations if r.priority == "high"]
        if high:
            parts.append(f"{len(high)} high-priority area(s) to focus on.")

    return " ".join(parts)
