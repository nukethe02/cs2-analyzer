"""Tests for cross-match player development tracking."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from opensight.ai.player_tracker import (
    LEVEL_BENCHMARKS,
    PERSONA_ROLE_MAP,
    ROLE_PRACTICE_TARGETS,
    DevelopmentReport,
    MatchSnapshot,
    PlayerTracker,
    PracticeRecommendation,
    RoleBenchmark,
    TrendAnalysis,
    _build_summary,
    _calculate_improvement_velocity,
    _calculate_player_averages,
    _compute_history_averages,
    _compute_trend_direction,
    _identify_strengths_weaknesses,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STEAM_ID = "76561198000000001"


def _make_match(
    idx: int,
    kills: int = 20,
    deaths: int = 15,
    adr: float = 75.0,
    kast: float = 70.0,
    hs_pct: float = 45.0,
    hltv_rating: float = 1.05,
    aim_rating: float = 60.0,
    utility_rating: float = 55.0,
    trade_kill_success: int = 3,
    trade_kill_attempts: int = 6,
    entry_success: int = 2,
    entry_attempts: int = 4,
    clutch_wins: int = 1,
    clutch_situations: int = 3,
    he_damage: int = 25,
    enemies_flashed: int = 5,
    flash_assists: int = 2,
    ttd_median_ms: float = 320.0,
    cp_median_deg: float = 12.0,
    rounds_played: int = 25,
) -> dict:
    """Build a match history dict matching get_player_history_full() output."""
    analyzed = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(days=idx)
    return {
        "id": idx,
        "steam_id": STEAM_ID,
        "demo_hash": f"hash_{idx:04d}",
        "analyzed_at": analyzed.isoformat(),
        "map_name": "de_mirage",
        "result": "win" if idx % 2 == 0 else "loss",
        "kills": kills,
        "deaths": deaths,
        "assists": 5,
        "adr": adr,
        "kast": kast,
        "hs_pct": hs_pct,
        "aim_rating": aim_rating,
        "utility_rating": utility_rating,
        "hltv_rating": hltv_rating,
        "opensight_rating": 55.0,
        "trade_kill_opportunities": 8,
        "trade_kill_attempts": trade_kill_attempts,
        "trade_kill_success": trade_kill_success,
        "entry_attempts": entry_attempts,
        "entry_success": entry_success,
        "clutch_situations": clutch_situations,
        "clutch_wins": clutch_wins,
        "clutch_kills": 2,
        "he_damage": he_damage,
        "enemies_flashed": enemies_flashed,
        "flash_assists": flash_assists,
        "ttd_median_ms": ttd_median_ms,
        "cp_median_deg": cp_median_deg,
        "rounds_played": rounds_played,
    }


def _make_improving_history(n: int = 10) -> list[dict]:
    """Create history where recent matches are better (improving trend)."""
    matches = []
    for i in range(n):
        matches.append(
            _make_match(
                idx=i,
                kills=15 + i * 2,  # 15 → 33
                deaths=20 - i,  # 20 → 11
                adr=60.0 + i * 4,  # 60 → 96
                kast=60.0 + i * 3,  # 60 → 87
                hs_pct=40.0 + i * 2,  # 40 → 58
                hltv_rating=0.85 + i * 0.05,  # 0.85 → 1.30
                ttd_median_ms=400.0 - i * 15,  # 400 → 265 (improving = lower)
                cp_median_deg=18.0 - i * 1.0,  # 18 → 9 (improving = lower)
            )
        )
    return matches


def _make_declining_history(n: int = 10) -> list[dict]:
    """Create history where recent matches are worse (declining trend)."""
    matches = []
    for i in range(n):
        matches.append(
            _make_match(
                idx=i,
                kills=30 - i * 2,  # 30 → 12
                deaths=10 + i,  # 10 → 19
                adr=95.0 - i * 4,  # 95 → 59
                kast=85.0 - i * 3,  # 85 → 58
                hs_pct=60.0 - i * 2,  # 60 → 42
                hltv_rating=1.35 - i * 0.05,  # 1.35 → 0.90
                trade_kill_success=6 - min(i, 5),  # 6 → 1
                flash_assists=5 - min(i, 4),  # 5 → 1
                ttd_median_ms=250.0 + i * 20,  # 250 → 430 (declining = higher)
                cp_median_deg=8.0 + i * 1.5,  # 8 → 21.5 (declining = higher)
            )
        )
    return matches


def _make_stable_history(n: int = 10) -> list[dict]:
    """Create history with minimal variation (stable trend)."""
    return [_make_match(idx=i) for i in range(n)]


def _history_to_snapshots(history: list[dict]) -> list[MatchSnapshot]:
    """Convert history dicts to MatchSnapshot list (oldest-first)."""
    tracker = PlayerTracker(db=MagicMock())
    return [tracker.snapshot_from_history(row) for row in history]


def _mock_tracker(history: list[dict]) -> PlayerTracker:
    """Create a PlayerTracker with mocked database.

    History is reversed to match real DB behavior (newest-first via ORDER BY DESC).
    """
    db = MagicMock()
    db.get_player_history_full.return_value = list(reversed(history))
    db.get_player_persona.return_value = None
    tracker = PlayerTracker(db=db)
    return tracker


# ===========================================================================
# TestMatchSnapshot
# ===========================================================================


class TestMatchSnapshot:
    """Test MatchSnapshot dataclass."""

    def test_construction(self):
        snap = MatchSnapshot(steam_id=STEAM_ID, demo_hash="abc123")
        assert snap.steam_id == STEAM_ID
        assert snap.kills == 0
        assert snap.hltv_rating == 1.0

    def test_to_dict(self):
        snap = MatchSnapshot(
            steam_id=STEAM_ID,
            demo_hash="abc123",
            kills=20,
            adr=75.5,
            hltv_rating=1.15,
        )
        d = snap.to_dict()
        assert d["steam_id"] == STEAM_ID
        assert d["kills"] == 20
        assert d["adr"] == 75.5
        assert d["hltv_rating"] == 1.15

    def test_optional_fields_none(self):
        snap = MatchSnapshot(steam_id=STEAM_ID, demo_hash="abc")
        d = snap.to_dict()
        assert d["ttd_median_ms"] is None
        assert d["cp_median_deg"] is None
        assert d["map_name"] is None


class TestSnapshotFromHistory:
    """Test converting DB history rows to MatchSnapshot."""

    def test_basic_conversion(self):
        row = _make_match(0, kills=25, adr=80.0, hltv_rating=1.10)
        tracker = PlayerTracker(db=MagicMock())
        snap = tracker.snapshot_from_history(row)

        assert snap.steam_id == STEAM_ID
        assert snap.kills == 25
        assert snap.adr == 80.0
        assert snap.hltv_rating == 1.10
        assert snap.map_name == "de_mirage"

    def test_missing_fields_default(self):
        row = {"steam_id": STEAM_ID, "demo_hash": "abc"}
        tracker = PlayerTracker(db=MagicMock())
        snap = tracker.snapshot_from_history(row)

        assert snap.kills == 0
        assert snap.hltv_rating == 1.0
        assert snap.adr == 0.0


class TestExtractSnapshot:
    """Test extracting snapshot from orchestrator result."""

    def test_basic_extraction(self):
        result = {
            "demo_info": {"demo_hash": "abc123", "map_name": "de_inferno"},
            "players": {
                STEAM_ID: {
                    "stats": {
                        "kills": 22,
                        "deaths": 14,
                        "assists": 5,
                        "adr": 82.0,
                        "headshot_pct": 48.0,
                        "rounds_played": 24,
                    },
                    "rating": {
                        "hltv_rating": 1.18,
                        "kast_percentage": 72.0,
                        "aim_rating": 65.0,
                        "utility_rating": 50.0,
                    },
                    "advanced": {"ttd_median_ms": 290.0, "cp_median_error_deg": 10.5},
                    "duels": {
                        "opening_kills": 3,
                        "opening_deaths": 2,
                        "clutch_wins": 1,
                        "clutch_attempts": 3,
                        "trade_kills": 4,
                        "trade_kill_opportunities": 8,
                    },
                    "utility": {"enemies_flashed": 6, "flash_assists": 3, "he_damage": 35},
                }
            },
        }
        tracker = PlayerTracker(db=MagicMock())
        snap = tracker.extract_snapshot(result, STEAM_ID)

        assert snap is not None
        assert snap.kills == 22
        assert snap.deaths == 14
        assert snap.adr == 82.0
        assert snap.hltv_rating == 1.18
        assert snap.kast == 72.0
        assert snap.cp_median_deg == 10.5
        assert snap.entry_attempts == 5  # 3 kills + 2 deaths
        assert snap.entry_success == 3
        assert snap.map_name == "de_inferno"

    def test_player_not_found(self):
        result = {"players": {}, "demo_info": {}}
        tracker = PlayerTracker(db=MagicMock())
        snap = tracker.extract_snapshot(result, STEAM_ID)
        assert snap is None

    def test_missing_sub_dicts(self):
        result = {
            "demo_info": {},
            "players": {STEAM_ID: {}},
        }
        tracker = PlayerTracker(db=MagicMock())
        snap = tracker.extract_snapshot(result, STEAM_ID)
        assert snap is not None
        assert snap.kills == 0
        assert snap.hltv_rating == 1.0


# ===========================================================================
# TestComputeTrendDirection
# ===========================================================================


class TestComputeTrendDirection:
    """Test the _compute_trend_direction helper."""

    def test_improving_higher_is_better(self):
        assert _compute_trend_direction(25.0, 20.0, True) == "improving"

    def test_declining_higher_is_better(self):
        assert _compute_trend_direction(15.0, 20.0, True) == "declining"

    def test_stable(self):
        assert _compute_trend_direction(20.0, 20.0, True) == "stable"

    def test_improving_lower_is_better(self):
        # Deaths: lower = better, so recent < overall = improving
        assert _compute_trend_direction(12.0, 18.0, False) == "improving"

    def test_declining_lower_is_better(self):
        # Deaths: lower = better, so recent > overall = declining
        assert _compute_trend_direction(22.0, 15.0, False) == "declining"

    def test_zero_historical(self):
        assert _compute_trend_direction(5.0, 0.0, True) == "stable"

    def test_within_threshold(self):
        # 4% change is within 5% threshold
        assert _compute_trend_direction(20.8, 20.0, True) == "stable"

    def test_exactly_at_threshold(self):
        # 5% exactly - just at the boundary
        assert _compute_trend_direction(21.0, 20.0, True) == "stable"

    def test_above_threshold(self):
        # 6% change > 5% threshold
        assert _compute_trend_direction(21.2, 20.0, True) == "improving"


# ===========================================================================
# TestTrendAnalysis
# ===========================================================================


class TestTrendAnalysis:
    """Test trend analysis via snapshots."""

    def test_improving_trends(self):
        history = _make_improving_history()
        snapshots = _history_to_snapshots(history)
        tracker = PlayerTracker(db=MagicMock())

        trends = tracker.analyze_trends(snapshots)
        assert len(trends) > 0

        kills_trend = next((t for t in trends if t.metric_name == "kills"), None)
        assert kills_trend is not None
        assert kills_trend.direction == "improving"
        assert kills_trend.change_pct > 0

    def test_declining_trends(self):
        history = _make_declining_history()
        snapshots = _history_to_snapshots(history)
        tracker = PlayerTracker(db=MagicMock())

        trends = tracker.analyze_trends(snapshots)
        kills_trend = next((t for t in trends if t.metric_name == "kills"), None)
        assert kills_trend is not None
        assert kills_trend.direction == "declining"

    def test_stable_trends(self):
        history = _make_stable_history()
        snapshots = _history_to_snapshots(history)
        tracker = PlayerTracker(db=MagicMock())

        trends = tracker.analyze_trends(snapshots)
        kills_trend = next((t for t in trends if t.metric_name == "kills"), None)
        assert kills_trend is not None
        assert kills_trend.direction == "stable"

    def test_insufficient_data(self):
        history = [_make_match(i) for i in range(2)]
        snapshots = _history_to_snapshots(history)
        tracker = PlayerTracker(db=MagicMock())

        trends = tracker.analyze_trends(snapshots)
        assert trends == []

    def test_current_snapshot_appended(self):
        history = _make_stable_history(5)
        snapshots = _history_to_snapshots(history)
        current = MatchSnapshot(
            steam_id=STEAM_ID,
            demo_hash="current",
            kills=50,
            deaths=5,
            adr=120.0,
            kast=90.0,
            hs_pct=70.0,
            hltv_rating=1.8,
            aim_rating=90.0,
            utility_rating=80.0,
        )
        tracker = PlayerTracker(db=MagicMock())
        trends = tracker.analyze_trends(snapshots, current=current)

        # Current value should be the appended snapshot
        kills_trend = next(t for t in trends if t.metric_name == "kills")
        assert kills_trend.current_value == 50.0

    def test_deaths_improving_when_lower(self):
        """Deaths should show 'improving' when recent values are lower."""
        history = _make_improving_history()  # deaths: 20 → 11
        snapshots = _history_to_snapshots(history)
        tracker = PlayerTracker(db=MagicMock())

        trends = tracker.analyze_trends(snapshots)
        deaths_trend = next((t for t in trends if t.metric_name == "deaths"), None)
        assert deaths_trend is not None
        assert deaths_trend.direction == "improving"

    def test_std_calculation(self):
        history = _make_improving_history()
        snapshots = _history_to_snapshots(history)
        tracker = PlayerTracker(db=MagicMock())

        trends = tracker.analyze_trends(snapshots)
        kills_trend = next(t for t in trends if t.metric_name == "kills")
        # With varying values, std should be > 0
        assert kills_trend.historical_std > 0
        assert kills_trend.recent_std >= 0

    def test_to_dict(self):
        t = TrendAnalysis(
            metric_name="kills",
            current_value=25.0,
            recent_avg=22.0,
            historical_avg=20.0,
            recent_std=3.5,
            historical_std=5.0,
            direction="improving",
            change_pct=25.0,
            sample_count=10,
        )
        d = t.to_dict()
        assert d["metric"] == "kills"
        assert d["current"] == 25.0
        assert d["direction"] == "improving"
        assert d["sample_count"] == 10


class TestAnalyzeTrendsFromDB:
    """Test the DB-based convenience method."""

    def test_from_db(self):
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        trends = tracker.analyze_trends_from_db(STEAM_ID)
        assert len(trends) > 0
        kills_trend = next(t for t in trends if t.metric_name == "kills")
        assert kills_trend.direction == "improving"

    def test_insufficient_data(self):
        history = [_make_match(0)]
        tracker = _mock_tracker(history)

        trends = tracker.analyze_trends_from_db(STEAM_ID)
        assert trends == []


# ===========================================================================
# TestLevelEstimation
# ===========================================================================


class TestLevelEstimation:
    """Test competitive level estimation."""

    def test_beginner(self):
        snapshots = _history_to_snapshots([_make_match(i, hltv_rating=0.70) for i in range(5)])
        tracker = PlayerTracker(db=MagicMock())
        assert tracker.estimate_level(snapshots) == "beginner"

    def test_intermediate(self):
        snapshots = _history_to_snapshots([_make_match(i, hltv_rating=0.95) for i in range(5)])
        tracker = PlayerTracker(db=MagicMock())
        assert tracker.estimate_level(snapshots) == "intermediate"

    def test_advanced(self):
        snapshots = _history_to_snapshots([_make_match(i, hltv_rating=1.12) for i in range(5)])
        tracker = PlayerTracker(db=MagicMock())
        assert tracker.estimate_level(snapshots) == "advanced"

    def test_elite(self):
        snapshots = _history_to_snapshots([_make_match(i, hltv_rating=1.35) for i in range(5)])
        tracker = PlayerTracker(db=MagicMock())
        assert tracker.estimate_level(snapshots) == "elite"

    def test_empty_snapshots(self):
        tracker = PlayerTracker(db=MagicMock())
        assert tracker.estimate_level([]) == "intermediate"

    def test_boundary_085(self):
        snapshots = _history_to_snapshots([_make_match(i, hltv_rating=0.85) for i in range(5)])
        tracker = PlayerTracker(db=MagicMock())
        assert tracker.estimate_level(snapshots) == "intermediate"

    def test_boundary_120(self):
        snapshots = _history_to_snapshots([_make_match(i, hltv_rating=1.20) for i in range(5)])
        tracker = PlayerTracker(db=MagicMock())
        assert tracker.estimate_level(snapshots) == "elite"


# ===========================================================================
# TestBenchmarks
# ===========================================================================


class TestBenchmarks:
    """Test competitive level benchmarking."""

    def test_basic_benchmarks(self):
        snapshots = _history_to_snapshots(_make_stable_history())
        tracker = PlayerTracker(db=MagicMock())

        benchmarks = tracker.compute_benchmarks(snapshots)
        assert len(benchmarks) > 0
        # All benchmarks should be for the estimated level
        levels = {b.level for b in benchmarks}
        assert len(levels) == 1

    def test_override_level(self):
        snapshots = _history_to_snapshots(_make_stable_history())
        tracker = PlayerTracker(db=MagicMock())

        benchmarks = tracker.compute_benchmarks(snapshots, level="elite")
        assert all(b.level == "elite" for b in benchmarks)

    def test_verdict_above(self):
        # Player with high rating (1.5) vs intermediate avg (1.0)
        snapshots = _history_to_snapshots(
            [_make_match(i, hltv_rating=1.5, adr=100.0) for i in range(5)]
        )
        tracker = PlayerTracker(db=MagicMock())

        benchmarks = tracker.compute_benchmarks(snapshots, level="intermediate")
        rating_bench = next(b for b in benchmarks if b.metric_name == "hltv_rating")
        assert rating_bench.verdict == "above"

    def test_verdict_below(self):
        # Player with low rating (0.6) vs intermediate avg (1.0)
        snapshots = _history_to_snapshots([_make_match(i, hltv_rating=0.6) for i in range(5)])
        tracker = PlayerTracker(db=MagicMock())

        benchmarks = tracker.compute_benchmarks(snapshots, level="intermediate")
        rating_bench = next(b for b in benchmarks if b.metric_name == "hltv_rating")
        assert rating_bench.verdict == "below"

    def test_percentile_clamped(self):
        # Very high value should cap at 100
        snapshots = _history_to_snapshots([_make_match(i, adr=200.0) for i in range(5)])
        tracker = PlayerTracker(db=MagicMock())

        benchmarks = tracker.compute_benchmarks(snapshots, level="beginner")
        adr_bench = next(b for b in benchmarks if b.metric_name == "adr")
        assert adr_bench.percentile_in_level == 100.0

    def test_empty_snapshots(self):
        tracker = PlayerTracker(db=MagicMock())
        assert tracker.compute_benchmarks([]) == []

    def test_invalid_level_defaults(self):
        snapshots = _history_to_snapshots(_make_stable_history())
        tracker = PlayerTracker(db=MagicMock())

        benchmarks = tracker.compute_benchmarks(snapshots, level="nonexistent")
        assert all(b.level == "intermediate" for b in benchmarks)

    def test_to_dict(self):
        b = RoleBenchmark(
            metric_name="adr",
            player_value=80.0,
            level="intermediate",
            level_avg=75.0,
            level_low=60.0,
            level_high=85.0,
            percentile_in_level=80.0,
            verdict="above",
        )
        d = b.to_dict()
        assert d["metric"] == "adr"
        assert d["level_range"] == [60.0, 85.0]
        assert d["verdict"] == "above"


# ===========================================================================
# TestPlayerAverages
# ===========================================================================


class TestPlayerAverages:
    """Test _calculate_player_averages."""

    def test_basic_averages(self):
        snapshots = _history_to_snapshots(
            [_make_match(0, kills=20, adr=70.0), _make_match(1, kills=30, adr=90.0)]
        )
        avgs = _calculate_player_averages(snapshots)
        assert avgs["adr"] == 80.0

    def test_entry_win_rate(self):
        snapshots = _history_to_snapshots(
            [
                _make_match(0, entry_success=3, entry_attempts=5),
                _make_match(1, entry_success=2, entry_attempts=5),
            ]
        )
        avgs = _calculate_player_averages(snapshots)
        assert avgs["entry_win_rate"] == 50.0

    def test_clutch_rate(self):
        snapshots = _history_to_snapshots(
            [
                _make_match(0, clutch_wins=2, clutch_situations=4),
                _make_match(1, clutch_wins=1, clutch_situations=6),
            ]
        )
        avgs = _calculate_player_averages(snapshots)
        assert avgs["clutch_win_rate"] == 30.0

    def test_trade_rate(self):
        snapshots = _history_to_snapshots(
            [
                _make_match(0, trade_kill_success=4, trade_kill_attempts=8),
                _make_match(1, trade_kill_success=3, trade_kill_attempts=7),
            ]
        )
        avgs = _calculate_player_averages(snapshots)
        assert abs(avgs["trade_rate"] - 46.7) < 0.1

    def test_zero_attempts(self):
        snapshots = _history_to_snapshots(
            [_make_match(0, entry_attempts=0, clutch_situations=0, trade_kill_attempts=0)]
        )
        avgs = _calculate_player_averages(snapshots)
        assert "entry_win_rate" not in avgs
        assert "clutch_win_rate" not in avgs
        assert "trade_rate" not in avgs

    def test_empty(self):
        assert _calculate_player_averages([]) == {}


class TestComputeHistoryAverages:
    """Test _compute_history_averages."""

    def test_basic(self):
        history = [_make_match(0, kills=20), _make_match(1, kills=30)]
        avgs = _compute_history_averages(history)
        assert avgs["kills"] == 25.0

    def test_derived_rates(self):
        history = [
            _make_match(0, entry_success=3, entry_attempts=5),
            _make_match(1, entry_success=2, entry_attempts=5),
        ]
        avgs = _compute_history_averages(history)
        assert avgs["entry_success_rate"] == 50.0


# ===========================================================================
# TestRecommendations
# ===========================================================================


class TestRecommendations:
    """Test practice recommendation generation."""

    def test_declining_kast_recommendation(self):
        history = _make_declining_history()
        tracker = _mock_tracker(history)

        recs = tracker.generate_recommendations(STEAM_ID)
        positioning_recs = [r for r in recs if r.area == "positioning"]
        assert len(positioning_recs) > 0

    def test_declining_adr_recommendation(self):
        history = _make_declining_history()
        tracker = _mock_tracker(history)

        recs = tracker.generate_recommendations(STEAM_ID)
        aim_recs = [r for r in recs if r.area == "aim"]
        assert len(aim_recs) > 0

    def test_no_recommendations_when_stable(self):
        """Stable metrics with good benchmarks should produce fewer recs."""
        history = [
            _make_match(
                i,
                kills=28,
                adr=90.0,
                hs_pct=55.0,
                hltv_rating=1.2,
                ttd_median_ms=280.0,
                cp_median_deg=8.0,
                trade_kill_success=5,
                flash_assists=4,
                he_damage=40,
                kast=78.0,
                clutch_wins=2,
                clutch_situations=4,
                entry_success=3,
                entry_attempts=5,
            )
            for i in range(10)
        ]
        tracker = _mock_tracker(history)

        recs = tracker.generate_recommendations(STEAM_ID)
        high_priority = [r for r in recs if r.priority == "high"]
        assert len(high_priority) == 0

    def test_priority_ordering(self):
        history = _make_declining_history()
        tracker = _mock_tracker(history)

        recs = tracker.generate_recommendations(STEAM_ID)
        if len(recs) >= 2:
            priorities = [r.priority for r in recs]
            order = {"high": 0, "medium": 1, "low": 2}
            assert all(
                order[priorities[i]] <= order[priorities[i + 1]] for i in range(len(priorities) - 1)
            )

    def test_insufficient_data_returns_empty(self):
        history = [_make_match(i) for i in range(2)]
        tracker = _mock_tracker(history)
        recs = tracker.generate_recommendations(STEAM_ID)
        assert recs == []

    def test_to_dict(self):
        rec = PracticeRecommendation(
            area="aim",
            priority="high",
            description="Test rec",
            current_value=10.0,
            target_value=20.0,
            drill="Do thing",
        )
        d = rec.to_dict()
        assert d["area"] == "aim"
        assert d["priority"] == "high"
        assert d["drill"] == "Do thing"


# ===========================================================================
# TestDevelopmentReport
# ===========================================================================


class TestDevelopmentReport:
    """Test full development report generation."""

    def test_full_report_structure(self):
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        report = tracker.generate_report(STEAM_ID)
        assert report.steam_id == STEAM_ID
        assert report.match_count == 10
        assert isinstance(report.trends, list)
        assert isinstance(report.benchmarks, list)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.strengths, list)
        assert isinstance(report.weaknesses, list)
        assert isinstance(report.summary, str)
        assert report.estimated_level in ("beginner", "intermediate", "advanced", "elite")

    def test_report_summary_content(self):
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        report = tracker.generate_report(STEAM_ID)
        assert "10 matches" in report.summary

    def test_report_date_range(self):
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        report = tracker.generate_report(STEAM_ID)
        assert report.date_range is not None
        earliest, latest = report.date_range
        assert "2026-01-01" in earliest
        assert "2026-01-10" in latest

    def test_report_empty_history(self):
        tracker = _mock_tracker([])
        report = tracker.generate_report(STEAM_ID)
        assert report.match_count == 0
        assert report.trends == []
        assert report.benchmarks == []

    def test_report_current_snapshot(self):
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        report = tracker.generate_report(STEAM_ID)
        assert report.current_snapshot is not None
        # Current should be the most recent match (highest kills)
        assert report.current_snapshot.kills == 33  # 15 + 9*2

    def test_to_dict(self):
        report = DevelopmentReport(
            steam_id=STEAM_ID,
            match_count=5,
            estimated_level="intermediate",
            strengths=["Good aim"],
            weaknesses=["Bad utility"],
            improvement_velocity=0.5,
            summary="Test summary",
        )
        d = report.to_dict()
        assert d["steam_id"] == STEAM_ID
        assert d["match_count"] == 5
        assert d["estimated_level"] == "intermediate"
        assert d["strengths"] == ["Good aim"]
        assert d["improvement_velocity"] == 0.5

    def test_improving_report_has_positive_velocity(self):
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        report = tracker.generate_report(STEAM_ID)
        assert report.improvement_velocity > 0

    def test_declining_report_has_negative_velocity(self):
        history = _make_declining_history()
        tracker = _mock_tracker(history)

        report = tracker.generate_report(STEAM_ID)
        assert report.improvement_velocity < 0


# ===========================================================================
# TestStrengthsWeaknesses
# ===========================================================================


class TestStrengthsWeaknesses:
    """Test strengths/weaknesses identification."""

    def test_above_benchmark_is_strength(self):
        benchmarks = [
            RoleBenchmark(
                metric_name="adr",
                player_value=95.0,
                level="intermediate",
                level_avg=75.0,
                level_low=60.0,
                level_high=85.0,
                percentile_in_level=100.0,
                verdict="above",
            )
        ]
        strengths, weaknesses = _identify_strengths_weaknesses(benchmarks, [])
        assert len(strengths) == 1
        assert "ADR" in strengths[0]
        assert "above" in strengths[0]

    def test_below_benchmark_is_weakness(self):
        benchmarks = [
            RoleBenchmark(
                metric_name="kast",
                player_value=55.0,
                level="intermediate",
                level_avg=68.0,
                level_low=60.0,
                level_high=75.0,
                percentile_in_level=0.0,
                verdict="below",
            )
        ]
        strengths, weaknesses = _identify_strengths_weaknesses(benchmarks, [])
        assert len(weaknesses) == 1
        assert "KAST" in weaknesses[0]

    def test_improving_trend_is_strength(self):
        trends = [
            TrendAnalysis(
                metric_name="kills",
                current_value=30.0,
                recent_avg=28.0,
                historical_avg=20.0,
                direction="improving",
                change_pct=50.0,
                sample_count=10,
            )
        ]
        strengths, _ = _identify_strengths_weaknesses([], trends)
        assert len(strengths) == 1
        assert "trending up" in strengths[0]

    def test_declining_trend_is_weakness(self):
        trends = [
            TrendAnalysis(
                metric_name="adr",
                current_value=60.0,
                recent_avg=62.0,
                historical_avg=80.0,
                direction="declining",
                change_pct=-25.0,
                sample_count=10,
            )
        ]
        _, weaknesses = _identify_strengths_weaknesses([], trends)
        assert len(weaknesses) == 1
        assert "trending down" in weaknesses[0]

    def test_small_change_not_reported(self):
        trends = [
            TrendAnalysis(
                metric_name="kills",
                current_value=20.5,
                recent_avg=20.3,
                historical_avg=20.0,
                direction="improving",
                change_pct=2.5,
                sample_count=10,
            )
        ]
        strengths, weaknesses = _identify_strengths_weaknesses([], trends)
        assert len(strengths) == 0  # 2.5% < 5% threshold


# ===========================================================================
# TestImprovementVelocity
# ===========================================================================


class TestImprovementVelocity:
    """Test improvement velocity calculation."""

    def test_all_improving(self):
        trends = [
            TrendAnalysis(
                metric_name="kills",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="improving",
            ),
            TrendAnalysis(
                metric_name="adr",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="improving",
            ),
        ]
        assert _calculate_improvement_velocity(trends) == 1.0

    def test_all_declining(self):
        trends = [
            TrendAnalysis(
                metric_name="kills",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="declining",
            ),
            TrendAnalysis(
                metric_name="adr",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="declining",
            ),
        ]
        assert _calculate_improvement_velocity(trends) == -1.0

    def test_mixed(self):
        trends = [
            TrendAnalysis(
                metric_name="kills",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="improving",
            ),
            TrendAnalysis(
                metric_name="adr",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="declining",
            ),
        ]
        assert _calculate_improvement_velocity(trends) == 0.0

    def test_all_stable(self):
        trends = [
            TrendAnalysis(
                metric_name="kills",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="stable",
            ),
        ]
        assert _calculate_improvement_velocity(trends) == 0.0

    def test_empty(self):
        assert _calculate_improvement_velocity([]) == 0.0


# ===========================================================================
# TestBuildSummary
# ===========================================================================


class TestBuildSummary:
    """Test summary generation."""

    def test_improving_summary(self):
        trends = [
            TrendAnalysis(
                metric_name="kills",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="improving",
            ),
        ]
        summary = _build_summary(trends, [], [], 10)
        assert "10 matches" in summary
        assert "Improving" in summary

    def test_declining_summary(self):
        trends = [
            TrendAnalysis(
                metric_name="adr",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="declining",
            ),
        ]
        summary = _build_summary(trends, [], [], 5)
        assert "Declining" in summary

    def test_stable_summary(self):
        trends = [
            TrendAnalysis(
                metric_name="kills",
                current_value=0,
                recent_avg=0,
                historical_avg=0,
                direction="stable",
            ),
        ]
        summary = _build_summary(trends, [], [], 5)
        assert "stable" in summary

    def test_benchmark_weakness_in_summary(self):
        benchmarks = [
            RoleBenchmark(
                metric_name="adr",
                player_value=50.0,
                level="intermediate",
                level_avg=75.0,
                level_low=60.0,
                level_high=85.0,
                percentile_in_level=0.0,
                verdict="below",
            )
        ]
        summary = _build_summary([], benchmarks, [], 5)
        assert "Below benchmark" in summary

    def test_high_priority_recs_in_summary(self):
        recs = [
            PracticeRecommendation(
                area="aim",
                priority="high",
                description="Test",
                current_value=0,
                target_value=0,
                drill="Test",
            )
        ]
        summary = _build_summary([], [], recs, 5)
        assert "high-priority" in summary


# ===========================================================================
# TestPersonaRoleMapping
# ===========================================================================


class TestPersonaRoleMapping:
    """Verify persona → role mapping coverage."""

    def test_all_personas_mapped(self):
        expected_personas = [
            "the_cleanup",
            "the_lurker",
            "the_opener",
            "the_anchor",
            "the_utility_master",
            "the_headhunter",
            "the_survivor",
            "the_damage_dealer",
            "the_flash_master",
            "the_terminator",
            "the_competitor",
        ]
        for persona in expected_personas:
            assert persona in PERSONA_ROLE_MAP, f"{persona} not in PERSONA_ROLE_MAP"

    def test_all_roles_have_practice_targets(self):
        roles = set(PERSONA_ROLE_MAP.values())
        for role in roles:
            assert role in ROLE_PRACTICE_TARGETS, f"{role} not in ROLE_PRACTICE_TARGETS"


class TestLevelBenchmarksCoverage:
    """Verify level benchmark data integrity."""

    def test_all_levels_present(self):
        assert "beginner" in LEVEL_BENCHMARKS
        assert "intermediate" in LEVEL_BENCHMARKS
        assert "advanced" in LEVEL_BENCHMARKS
        assert "elite" in LEVEL_BENCHMARKS

    def test_all_levels_have_same_metrics(self):
        metrics = set(LEVEL_BENCHMARKS["beginner"].keys())
        for level in LEVEL_BENCHMARKS.values():
            assert set(level.keys()) == metrics

    def test_all_benchmarks_have_low_avg_high(self):
        for level_name, level_data in LEVEL_BENCHMARKS.items():
            for metric_name, values in level_data.items():
                assert "low" in values, f"{level_name}.{metric_name} missing 'low'"
                assert "avg" in values, f"{level_name}.{metric_name} missing 'avg'"
                assert "high" in values, f"{level_name}.{metric_name} missing 'high'"
                assert values["low"] <= values["avg"] <= values["high"], (
                    f"{level_name}.{metric_name}: low <= avg <= high violated"
                )


# ===========================================================================
# TestTrackingAPI
# ===========================================================================


class TestTrackingAPI:
    """Test the player tracking API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from opensight.api import app

        self.client = TestClient(app)

    def test_invalid_steam_id_report(self):
        resp = self.client.get("/api/player-tracking/invalid/report")
        assert resp.status_code == 400

    def test_invalid_steam_id_trends(self):
        resp = self.client.get("/api/player-tracking/invalid/trends")
        assert resp.status_code == 400

    @patch("opensight.ai.player_tracker.get_player_tracker")
    def test_report_endpoint(self, mock_get):
        tracker = _mock_tracker(_make_improving_history())
        mock_get.return_value = tracker

        resp = self.client.get(f"/api/player-tracking/{STEAM_ID}/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["steam_id"] == STEAM_ID
        assert data["match_count"] == 10
        assert "trends" in data
        assert "benchmarks" in data
        assert "strengths" in data
        assert "weaknesses" in data
        assert "summary" in data

    @patch("opensight.ai.player_tracker.get_player_tracker")
    def test_trends_endpoint(self, mock_get):
        tracker = _mock_tracker(_make_improving_history())
        mock_get.return_value = tracker

        resp = self.client.get(f"/api/player-tracking/{STEAM_ID}/trends")
        assert resp.status_code == 200
        data = resp.json()
        assert data["steam_id"] == STEAM_ID
        assert "trends" in data
        assert data["count"] > 0

    @patch("opensight.ai.player_tracker.get_player_tracker")
    def test_trends_empty_history(self, mock_get):
        tracker = _mock_tracker([])
        mock_get.return_value = tracker

        resp = self.client.get(f"/api/player-tracking/{STEAM_ID}/trends")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    @patch("opensight.ai.player_tracker.get_player_tracker")
    def test_report_empty_history(self, mock_get):
        tracker = _mock_tracker([])
        mock_get.return_value = tracker

        resp = self.client.get(f"/api/player-tracking/{STEAM_ID}/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["match_count"] == 0

    def test_add_match_invalid_steam_id(self):
        resp = self.client.post(
            "/api/player-tracking/invalid/add-match?job_id=00000000-0000-0000-0000-000000000000"
        )
        assert resp.status_code == 400

    def test_add_match_job_not_found(self):
        resp = self.client.post(
            f"/api/player-tracking/{STEAM_ID}/add-match?job_id=00000000-0000-0000-0000-000000000000"
        )
        assert resp.status_code == 404
