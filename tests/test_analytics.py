"""Tests for the analytics module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import math
import numpy as np
import pandas as pd
import pytest

from opensight.analytics import (
    DemoAnalyzer,
    TTDResult,
    CrosshairPlacementResult,
    PlayerAnalytics,
    safe_float,
    analyze_demo,
)
from opensight.parser import DemoData, KillEvent, DamageEvent


class TestSafeFloat:
    """Tests for safe_float utility."""

    def test_safe_float_with_valid_float(self):
        """Return float when given valid float."""
        assert safe_float(3.14) == 3.14

    def test_safe_float_with_valid_int(self):
        """Return float when given valid int."""
        assert safe_float(42) == 42.0

    def test_safe_float_with_valid_string(self):
        """Return float when given valid string."""
        assert safe_float("3.14") == 3.14

    def test_safe_float_with_none(self):
        """Return default when given None."""
        assert safe_float(None) == 0.0
        assert safe_float(None, 99.9) == 99.9

    def test_safe_float_with_nan(self):
        """Return default when given NaN."""
        assert safe_float(float("nan")) == 0.0

    def test_safe_float_with_invalid_string(self):
        """Return default when given invalid string."""
        assert safe_float("not a number") == 0.0
        assert safe_float("not a number", 99.9) == 99.9


class TestTTDResult:
    """Tests for TTDResult dataclass."""

    def test_ttd_result_creation(self):
        """TTDResult can be created with required fields."""
        result = TTDResult(
            tick_spotted=1000,
            tick_damage=1100,
            ttd_ticks=100,
            ttd_ms=1562.5,
            attacker_steamid=12345,
            victim_steamid=67890,
            weapon="ak47",
            headshot=True,
            is_prefire=False,
        )
        assert result.tick_spotted == 1000
        assert result.ttd_ms == 1562.5
        assert result.is_prefire is False


class TestCrosshairPlacementResult:
    """Tests for CrosshairPlacementResult dataclass."""

    def test_cp_result_creation(self):
        """CrosshairPlacementResult can be created with required fields."""
        result = CrosshairPlacementResult(
            tick=1000,
            attacker_steamid=12345,
            victim_steamid=67890,
            angular_error_deg=5.5,
            pitch_error_deg=-1.2,
            yaw_error_deg=3.4,
        )
        assert result.tick == 1000
        assert result.angular_error_deg == 5.5


class TestPlayerAnalytics:
    """Tests for PlayerAnalytics dataclass."""

    def test_player_analytics_creation(self):
        """PlayerAnalytics can be created with required fields."""
        analytics = PlayerAnalytics(
            steam_id=12345,
            name="Player1",
            team="CT",
            kills=15,
            deaths=10,
            assists=5,
            adr=85.5,
            hs_percent=45.0,
            ttd_median_ms=350.0,
            ttd_mean_ms=380.0,
            ttd_min_ms=200.0,
            ttd_max_ms=800.0,
            ttd_std_ms=120.0,
            ttd_count=10,
            prefire_count=2,
            cp_median_error_deg=4.5,
            cp_mean_error_deg=5.2,
            cp_pitch_bias_deg=-0.5,
            weapon_kills={"ak47": 8, "awp": 5, "usp_silencer": 2},
            ttd_values=[300.0, 350.0, 400.0],
            cp_values=[3.0, 4.5, 6.0],
        )
        assert analytics.steam_id == 12345
        assert analytics.kills == 15
        assert analytics.adr == 85.5
        assert analytics.weapon_kills["ak47"] == 8


class TestDemoAnalyzer:
    """Tests for DemoAnalyzer class."""

    @pytest.fixture
    def sample_demo_data(self):
        """Create sample demo data for testing."""
        kills_df = pd.DataFrame({
            "tick": [1000, 2000, 3000],
            "attacker_steamid": [12345, 67890, 12345],
            "user_steamid": [67890, 12345, 67890],
            "attacker_name": ["Player1", "Player2", "Player1"],
            "user_name": ["Player2", "Player1", "Player2"],
            "weapon": ["ak47", "awp", "ak47"],
            "headshot": [True, False, True],
        })

        damages_df = pd.DataFrame({
            "tick": [990, 1000, 1990, 2000, 2990, 3000],
            "attacker_steamid": [12345, 12345, 67890, 67890, 12345, 12345],
            "user_steamid": [67890, 67890, 12345, 12345, 67890, 67890],
            "dmg_health": [27, 73, 108, 0, 27, 73],
            "weapon": ["ak47", "ak47", "awp", "awp", "ak47", "ak47"],
        })

        return DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=15,
            player_stats={
                12345: {
                    "name": "Player1",
                    "team": "CT",
                    "kills": 2,
                    "deaths": 1,
                    "assists": 0,
                    "adr": 80.0,
                    "hs_percent": 100.0,
                    "headshots": 2,
                    "total_damage": 200,
                    "weapon_kills": {"ak47": 2},
                },
                67890: {
                    "name": "Player2",
                    "team": "T",
                    "kills": 1,
                    "deaths": 2,
                    "assists": 0,
                    "adr": 72.0,
                    "hs_percent": 0.0,
                    "headshots": 0,
                    "total_damage": 108,
                    "weapon_kills": {"awp": 1},
                },
            },
            player_names={12345: "Player1", 67890: "Player2"},
            player_teams={12345: 3, 67890: 2},  # 3=CT, 2=T
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=damages_df,
            ticks_df=None,
        )

    def test_analyzer_initialization(self, sample_demo_data):
        """Analyzer initializes correctly with demo data."""
        analyzer = DemoAnalyzer(sample_demo_data)
        assert analyzer.data is sample_demo_data
        assert analyzer.TICK_RATE == 64

    def test_analyzer_computes_ttd(self, sample_demo_data):
        """Analyzer computes TTD from kill/damage data."""
        analyzer = DemoAnalyzer(sample_demo_data)
        analyzer._compute_ttd()

        # Should have TTD results for engagements
        assert len(analyzer._ttd_results) > 0

    def test_analyzer_builds_player_analytics(self, sample_demo_data):
        """Analyzer builds complete player analytics."""
        analyzer = DemoAnalyzer(sample_demo_data)
        results = analyzer.analyze()

        assert 12345 in results
        assert 67890 in results

        player1 = results[12345]
        assert player1.name == "Player1"
        assert player1.kills == 2
        assert player1.deaths == 1

    def test_analyze_with_empty_data(self):
        """Analyzer handles empty demo data gracefully."""
        empty_data = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=0.0,
            tick_rate=64,
            num_rounds=0,
            player_stats={},
            player_names={},
            player_teams={},
            kills=[],
            damages=[],
            kills_df=pd.DataFrame(),
            damages_df=pd.DataFrame(),
        )

        analyzer = DemoAnalyzer(empty_data)
        results = analyzer.analyze()

        assert results == {}


class TestAngularErrorCalculation:
    """Tests for angular error calculation in DemoAnalyzer."""

    def test_angular_error_forward_looking(self):
        """Zero error when looking directly at target."""
        analyzer = DemoAnalyzer(MagicMock())

        # Attacker at origin, looking forward (+X)
        attacker_pos = np.array([0.0, 0.0, 64.0])
        pitch = 0.0
        yaw = 0.0
        # Target directly ahead
        victim_pos = np.array([100.0, 0.0, 64.0])

        total, pitch_err, yaw_err = analyzer._calculate_angular_error(
            attacker_pos, pitch, yaw, victim_pos
        )

        assert total == pytest.approx(0.0, abs=0.1)
        assert pitch_err == pytest.approx(0.0, abs=0.1)
        assert yaw_err == pytest.approx(0.0, abs=0.1)

    def test_angular_error_90_degrees(self):
        """90 degree error when target is to the side."""
        analyzer = DemoAnalyzer(MagicMock())

        # Attacker at origin, looking forward (+X)
        attacker_pos = np.array([0.0, 0.0, 64.0])
        pitch = 0.0
        yaw = 0.0
        # Target to the left (+Y)
        victim_pos = np.array([0.0, 100.0, 64.0])

        total, pitch_err, yaw_err = analyzer._calculate_angular_error(
            attacker_pos, pitch, yaw, victim_pos
        )

        assert total == pytest.approx(90.0, abs=1.0)

    def test_angular_error_same_position(self):
        """Zero error when at same position."""
        analyzer = DemoAnalyzer(MagicMock())

        pos = np.array([100.0, 100.0, 64.0])
        total, pitch_err, yaw_err = analyzer._calculate_angular_error(
            pos, 0.0, 0.0, pos
        )

        assert total == 0.0
        assert pitch_err == 0.0
        assert yaw_err == 0.0


class TestAnalyzeDemoFunction:
    """Tests for analyze_demo convenience function."""

    def test_analyze_demo_creates_analyzer(self):
        """analyze_demo creates analyzer and runs analysis."""
        mock_data = MagicMock()
        mock_data.kills_df = pd.DataFrame()
        mock_data.damages_df = pd.DataFrame()
        mock_data.ticks_df = None
        mock_data.player_stats = {}
        mock_data.player_names = {}

        results = analyze_demo(mock_data)

        assert isinstance(results, dict)
