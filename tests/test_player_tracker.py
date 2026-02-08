"""Tests for cross-match player development tracking."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from opensight.ai.player_tracker import (
    PERSONA_ROLE_MAP,
    ROLE_BENCHMARKS,
    PlayerTracker,
    _compute_averages,
    _compute_direction,
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
    """Build a match history dict matching database schema."""
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
        "aim_rating": 60.0,
        "utility_rating": 55.0,
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
        # Gradually increase kills, decrease deaths over time
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


def _mock_tracker(history: list[dict]) -> PlayerTracker:
    """Create a PlayerTracker with mocked database.

    History is reversed to match real DB behavior (newest-first via ORDER BY DESC).
    Fixture functions create data in chronological order (oldest-first).
    """
    db = MagicMock()
    db.get_player_history_full.return_value = list(reversed(history))
    tracker = PlayerTracker(db)
    return tracker


# ===========================================================================
# TestComputeDirection
# ===========================================================================


class TestComputeDirection:
    """Test the _compute_direction helper."""

    def test_improving_higher_is_better(self):
        direction, pct = _compute_direction("kills", 25.0, 20.0)
        assert direction == "improving"
        assert pct > 0

    def test_declining_higher_is_better(self):
        direction, pct = _compute_direction("kills", 15.0, 20.0)
        assert direction == "declining"
        assert pct < 0

    def test_stable(self):
        direction, pct = _compute_direction("kills", 20.0, 20.0)
        assert direction == "stable"

    def test_improving_lower_is_better(self):
        # TTD: lower = better, so recent < overall = improving
        direction, pct = _compute_direction("ttd_median_ms", 250.0, 350.0)
        assert direction == "improving"

    def test_declining_lower_is_better(self):
        # CP: lower = better, so recent > overall = declining
        direction, pct = _compute_direction("cp_median_deg", 18.0, 10.0)
        assert direction == "declining"

    def test_zero_overall(self):
        direction, pct = _compute_direction("kills", 5.0, 0.0)
        assert direction == "stable"
        assert pct == 0.0


# ===========================================================================
# TestMetricTrends
# ===========================================================================


class TestMetricTrends:
    """Test trend analysis."""

    def test_improving_trend(self):
        history = _make_improving_history()
        tracker = _mock_tracker(history)
        trends = tracker.analyze_trends(STEAM_ID)

        assert len(trends) > 0
        kills_trend = next((t for t in trends if t.metric == "kills"), None)
        assert kills_trend is not None
        assert kills_trend.direction == "improving"
        assert kills_trend.change_pct > 0

    def test_declining_trend(self):
        history = _make_declining_history()
        tracker = _mock_tracker(history)
        trends = tracker.analyze_trends(STEAM_ID)

        kills_trend = next((t for t in trends if t.metric == "kills"), None)
        assert kills_trend is not None
        assert kills_trend.direction == "declining"

    def test_stable_trend(self):
        history = _make_stable_history()
        tracker = _mock_tracker(history)
        trends = tracker.analyze_trends(STEAM_ID)

        kills_trend = next((t for t in trends if t.metric == "kills"), None)
        assert kills_trend is not None
        assert kills_trend.direction == "stable"

    def test_insufficient_data(self):
        history = [_make_match(i) for i in range(3)]  # Less than MIN_MATCHES
        tracker = _mock_tracker(history)
        trends = tracker.analyze_trends(STEAM_ID)
        assert trends == []

    def test_ttd_improving_when_lower(self):
        """TTD should show improving when recent values are lower."""
        history = _make_improving_history()
        tracker = _mock_tracker(history)
        trends = tracker.analyze_trends(STEAM_ID)

        ttd_trend = next((t for t in trends if t.metric == "ttd_median_ms"), None)
        assert ttd_trend is not None
        assert ttd_trend.direction == "improving"

    def test_cp_declining_when_higher(self):
        """CP should show declining when recent values are higher."""
        history = _make_declining_history()
        tracker = _mock_tracker(history)
        trends = tracker.analyze_trends(STEAM_ID)

        cp_trend = next((t for t in trends if t.metric == "cp_median_deg"), None)
        assert cp_trend is not None
        assert cp_trend.direction == "declining"

    def test_values_in_chronological_order(self):
        """Values should be in chronological order (oldest first)."""
        history = _make_improving_history()
        tracker = _mock_tracker(history)
        trends = tracker.analyze_trends(STEAM_ID)

        kills_trend = next(t for t in trends if t.metric == "kills")
        # First value should be smallest (oldest match)
        assert kills_trend.values[0] < kills_trend.values[-1]


# ===========================================================================
# TestRoleBenchmarks
# ===========================================================================


class TestRoleBenchmarks:
    """Test role benchmark comparison."""

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_entry_fragger_benchmarks(self, mock_role):
        mock_role.return_value = "entry_fragger"
        history = _make_stable_history()
        tracker = _mock_tracker(history)

        benchmark = tracker.get_role_benchmarks(STEAM_ID)
        assert benchmark is not None
        assert benchmark.role == "entry_fragger"
        assert "hs_pct" in benchmark.benchmarks
        assert "entry_success_rate" in benchmark.benchmarks

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_support_benchmarks(self, mock_role):
        mock_role.return_value = "support"
        history = _make_stable_history()
        tracker = _mock_tracker(history)

        benchmark = tracker.get_role_benchmarks(STEAM_ID)
        assert benchmark is not None
        assert benchmark.role == "support"
        assert "flash_assists" in benchmark.benchmarks
        assert "enemies_flashed" in benchmark.benchmarks

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_no_persona_defaults_to_fragger(self, mock_role):
        mock_role.return_value = "fragger"
        history = _make_stable_history()
        tracker = _mock_tracker(history)

        benchmark = tracker.get_role_benchmarks(STEAM_ID)
        assert benchmark is not None
        assert benchmark.role == "fragger"

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_percentile_calculation(self, mock_role):
        mock_role.return_value = "fragger"
        # Player with high kills should have good percentile
        history = [_make_match(i, kills=30, adr=95.0, hltv_rating=1.3) for i in range(10)]
        tracker = _mock_tracker(history)

        benchmark = tracker.get_role_benchmarks(STEAM_ID)
        assert benchmark is not None
        # 30 kills vs 22 target = 100+ capped at 100
        assert benchmark.percentiles["kills"] == 100.0
        # ADR 95 vs 85 target = ~112 capped at 100
        assert benchmark.percentiles["adr"] == 100.0

    def test_insufficient_data_returns_none(self):
        history = [_make_match(i) for i in range(3)]
        tracker = _mock_tracker(history)
        assert tracker.get_role_benchmarks(STEAM_ID) is None


# ===========================================================================
# TestRecommendations
# ===========================================================================


class TestRecommendations:
    """Test practice recommendation generation."""

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_declining_cp_recommendation(self, mock_role):
        mock_role.return_value = "fragger"
        history = _make_declining_history()
        tracker = _mock_tracker(history)

        recs = tracker.generate_recommendations(STEAM_ID)
        aim_recs = [r for r in recs if r.area == "aim"]
        assert len(aim_recs) > 0
        assert any("crosshair" in r.description.lower() for r in aim_recs)

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_declining_trade_recommendation(self, mock_role):
        mock_role.return_value = "fragger"
        history = _make_declining_history()
        tracker = _mock_tracker(history)

        recs = tracker.generate_recommendations(STEAM_ID)
        trade_recs = [r for r in recs if r.area == "trading"]
        assert len(trade_recs) > 0

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_no_recommendations_when_stable(self, mock_role):
        """Stable metrics with good benchmarks should produce fewer recs."""
        mock_role.return_value = "fragger"
        # High-performing stable player
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
        # A stable, high-performing player should have few or no recs
        high_priority = [r for r in recs if r.priority == "high"]
        assert len(high_priority) == 0

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_priority_ordering(self, mock_role):
        mock_role.return_value = "fragger"
        history = _make_declining_history()
        tracker = _mock_tracker(history)

        recs = tracker.generate_recommendations(STEAM_ID)
        if len(recs) >= 2:
            priorities = [r.priority for r in recs]
            order = {"high": 0, "medium": 1, "low": 2}
            assert all(
                order[priorities[i]] <= order[priorities[i + 1]] for i in range(len(priorities) - 1)
            )

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_multiple_recommendations(self, mock_role):
        mock_role.return_value = "fragger"
        history = _make_declining_history()
        tracker = _mock_tracker(history)

        recs = tracker.generate_recommendations(STEAM_ID)
        # Declining history should trigger multiple recs
        assert len(recs) >= 2

    def test_insufficient_data_returns_empty(self):
        history = [_make_match(i) for i in range(3)]
        tracker = _mock_tracker(history)
        recs = tracker.generate_recommendations(STEAM_ID)
        assert recs == []


# ===========================================================================
# TestDevelopmentReport
# ===========================================================================


class TestDevelopmentReport:
    """Test full development report generation."""

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_full_report_structure(self, mock_role):
        mock_role.return_value = "fragger"
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        report = tracker.get_development_report(STEAM_ID)
        assert report is not None
        assert report.steam_id == STEAM_ID
        assert report.match_count == 10
        assert isinstance(report.trends, list)
        assert isinstance(report.recommendations, list)
        assert report.role_benchmark is not None
        assert isinstance(report.summary, str)

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_report_summary_content(self, mock_role):
        mock_role.return_value = "fragger"
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        report = tracker.get_development_report(STEAM_ID)
        assert report is not None
        assert "10 matches" in report.summary
        assert "Improving" in report.summary

    @patch("opensight.ai.player_tracker.PlayerTracker._get_player_role")
    def test_report_date_range(self, mock_role):
        mock_role.return_value = "fragger"
        history = _make_improving_history()
        tracker = _mock_tracker(history)

        report = tracker.get_development_report(STEAM_ID)
        assert report is not None
        earliest, latest = report.date_range
        assert "2026-01-01" in earliest
        assert "2026-01-10" in latest

    def test_report_minimum_matches(self):
        history = [_make_match(i) for i in range(3)]
        tracker = _mock_tracker(history)

        report = tracker.get_development_report(STEAM_ID)
        assert report is None


# ===========================================================================
# TestComputeAverages
# ===========================================================================


class TestComputeAverages:
    """Test the _compute_averages helper."""

    def test_basic_averages(self):
        history = [_make_match(0, kills=20), _make_match(1, kills=30)]
        avgs = _compute_averages(history)
        assert avgs["kills"] == 25.0

    def test_derived_entry_rate(self):
        history = [
            _make_match(0, entry_success=3, entry_attempts=5),
            _make_match(1, entry_success=2, entry_attempts=5),
        ]
        avgs = _compute_averages(history)
        # 5 successes / 10 attempts = 50%
        assert avgs["entry_success_rate"] == 50.0

    def test_derived_clutch_rate(self):
        history = [
            _make_match(0, clutch_wins=2, clutch_situations=4),
            _make_match(1, clutch_wins=1, clutch_situations=6),
        ]
        avgs = _compute_averages(history)
        # 3 wins / 10 situations = 30%
        assert avgs["clutch_success_rate"] == 30.0

    def test_derived_trade_rate(self):
        history = [
            _make_match(0, trade_kill_success=4, trade_kill_attempts=8),
            _make_match(1, trade_kill_success=3, trade_kill_attempts=7),
        ]
        avgs = _compute_averages(history)
        # 7 success / 15 attempts = 46.7%
        assert abs(avgs["trade_success_rate"] - 46.7) < 0.1

    def test_zero_attempts_rates(self):
        history = [_make_match(0, entry_attempts=0, clutch_situations=0, trade_kill_attempts=0)]
        avgs = _compute_averages(history)
        assert avgs["entry_success_rate"] == 0.0
        assert avgs["clutch_success_rate"] == 0.0
        assert avgs["trade_success_rate"] == 0.0


# ===========================================================================
# TestTrackingAPI
# ===========================================================================


class TestTrackingAPI:
    """Test the tracking API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from opensight.api import app

        self.client = TestClient(app)

    def test_invalid_steam_id_trends(self):
        resp = self.client.get("/api/tracking/invalid/trends")
        assert resp.status_code == 400

    def test_invalid_steam_id_benchmarks(self):
        resp = self.client.get("/api/tracking/invalid/benchmarks")
        assert resp.status_code == 400

    def test_invalid_steam_id_recommendations(self):
        resp = self.client.get("/api/tracking/invalid/recommendations")
        assert resp.status_code == 400

    def test_invalid_steam_id_report(self):
        resp = self.client.get("/api/tracking/invalid/report")
        assert resp.status_code == 400

    @patch("opensight.api.routes_tracking._get_tracker")
    def test_trends_endpoint(self, mock_get):
        tracker = _mock_tracker(_make_improving_history())
        with patch.object(tracker, "_get_player_role", return_value="fragger"):
            mock_get.return_value = tracker
            resp = self.client.get(f"/api/tracking/{STEAM_ID}/trends")
            assert resp.status_code == 200
            data = resp.json()
            assert data["steam_id"] == STEAM_ID
            assert data["trend_count"] > 0
            assert isinstance(data["trends"], list)

    @patch("opensight.api.routes_tracking._get_tracker")
    def test_trends_insufficient_data(self, mock_get):
        tracker = _mock_tracker([_make_match(i) for i in range(3)])
        mock_get.return_value = tracker
        resp = self.client.get(f"/api/tracking/{STEAM_ID}/trends")
        assert resp.status_code == 400

    @patch("opensight.api.routes_tracking._get_tracker")
    def test_benchmarks_endpoint(self, mock_get):
        tracker = _mock_tracker(_make_stable_history())
        with patch.object(tracker, "_get_player_role", return_value="fragger"):
            mock_get.return_value = tracker
            resp = self.client.get(f"/api/tracking/{STEAM_ID}/benchmarks")
            assert resp.status_code == 200
            data = resp.json()
            assert data["steam_id"] == STEAM_ID
            assert data["role"] == "fragger"
            assert "benchmarks" in data
            assert "percentiles" in data

    @patch("opensight.api.routes_tracking._get_tracker")
    def test_recommendations_endpoint(self, mock_get):
        tracker = _mock_tracker(_make_declining_history())
        with patch.object(tracker, "_get_player_role", return_value="fragger"):
            mock_get.return_value = tracker
            resp = self.client.get(f"/api/tracking/{STEAM_ID}/recommendations")
            assert resp.status_code == 200
            data = resp.json()
            assert data["steam_id"] == STEAM_ID
            assert "recommendations" in data
            assert data["recommendation_count"] >= 0

    @patch("opensight.api.routes_tracking._get_tracker")
    def test_report_endpoint(self, mock_get):
        tracker = _mock_tracker(_make_improving_history())
        with patch.object(tracker, "_get_player_role", return_value="fragger"):
            mock_get.return_value = tracker
            resp = self.client.get(f"/api/tracking/{STEAM_ID}/report")
            assert resp.status_code == 200
            data = resp.json()
            assert data["steam_id"] == STEAM_ID
            assert data["match_count"] == 10
            assert "date_range" in data
            assert "trends" in data
            assert "role_benchmark" in data
            assert "recommendations" in data
            assert "summary" in data


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

    def test_all_roles_have_benchmarks(self):
        roles = set(PERSONA_ROLE_MAP.values())
        for role in roles:
            assert role in ROLE_BENCHMARKS, f"{role} not in ROLE_BENCHMARKS"
