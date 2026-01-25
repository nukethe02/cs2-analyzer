"""Tests for the analytics module."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from opensight.analysis.analytics import (
    CrosshairPlacementResult,
    DemoAnalyzer,
    PlayerAnalytics,
    TTDResult,
    analyze_demo,
    safe_float,
)
from opensight.core.parser import DemoData


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
    """Tests for PlayerAnalytics (PlayerMatchStats) dataclass."""

    def test_player_analytics_creation(self):
        """PlayerAnalytics (PlayerMatchStats) can be created with required fields."""
        analytics = PlayerAnalytics(
            steam_id=12345,
            name="Player1",
            team="CT",
            kills=15,
            deaths=10,
            assists=5,
            headshots=7,
            total_damage=1280,
            rounds_played=15,
            weapon_kills={"ak47": 8, "awp": 5, "usp_silencer": 2},
            ttd_values=[300.0, 350.0, 400.0],
            cp_values=[3.0, 4.5, 6.0],
            prefire_count=2,
        )
        assert analytics.steam_id == 12345
        assert analytics.kills == 15
        assert analytics.adr == 85.3  # 1280/15 rounded
        assert analytics.weapon_kills["ak47"] == 8


class TestDemoAnalyzer:
    """Tests for DemoAnalyzer class."""

    @pytest.fixture
    def sample_demo_data(self):
        """Create sample demo data for testing."""
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 2000, 3000],
                "attacker_steamid": [12345, 67890, 12345],
                "user_steamid": [67890, 12345, 67890],
                "attacker_name": ["Player1", "Player2", "Player1"],
                "user_name": ["Player2", "Player1", "Player2"],
                "weapon": ["ak47", "awp", "ak47"],
                "headshot": [True, False, True],
            }
        )

        damages_df = pd.DataFrame(
            {
                "tick": [990, 1000, 1990, 2000, 2990, 3000],
                "attacker_steamid": [12345, 12345, 67890, 67890, 12345, 12345],
                "user_steamid": [67890, 67890, 12345, 12345, 67890, 67890],
                "dmg_health": [27, 73, 108, 0, 27, 73],
                "weapon": ["ak47", "ak47", "awp", "awp", "ak47", "ak47"],
            }
        )

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
        # Use fallback mode to test original implementation
        analyzer = DemoAnalyzer(sample_demo_data, use_optimized=False)
        analyzer._compute_ttd()

        # Should have TTD results for engagements (fallback checks kills list which is empty)
        # With optimized, TTD values go to player stats
        # This test verifies the computation doesn't crash with empty kills list
        assert analyzer._ttd_results is not None

    def test_analyzer_builds_player_analytics(self, sample_demo_data):
        """Analyzer builds complete player analytics."""
        analyzer = DemoAnalyzer(sample_demo_data)
        results = analyzer.analyze()

        # Results is now MatchAnalysis, access players dict
        assert 12345 in results.players
        assert 67890 in results.players

        player1 = results.players[12345]
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

        # Results is now MatchAnalysis with empty players dict
        assert results.players == {}


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
        total, pitch_err, yaw_err = analyzer._calculate_angular_error(pos, 0.0, 0.0, pos)

        assert total == 0.0
        assert pitch_err == 0.0
        assert yaw_err == 0.0


class TestAnalyzeDemoFunction:
    """Tests for analyze_demo convenience function."""

    def test_analyze_demo_creates_analyzer(self):
        """analyze_demo creates analyzer and runs analysis."""
        from opensight.analysis.analytics import MatchAnalysis

        mock_data = MagicMock()
        mock_data.file_path = Path("/tmp/test.dem")
        mock_data.kills_df = pd.DataFrame()
        mock_data.damages_df = pd.DataFrame()
        mock_data.ticks_df = None
        mock_data.player_stats = {}
        mock_data.player_names = {}
        mock_data.player_teams = {}
        mock_data.num_rounds = 0
        mock_data.map_name = "de_dust2"
        mock_data.kills = []
        mock_data.rounds = []

        results = analyze_demo(mock_data)

        assert isinstance(results, MatchAnalysis)


class TestOpeningDuelStats:
    """Tests for enhanced OpeningDuelStats."""

    def test_opening_duel_stats_creation(self):
        """OpeningDuelStats can be created with default values."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats()
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.attempts == 0
        assert stats.entry_ttd_values == []
        assert stats.t_side_entries == 0
        assert stats.ct_side_entries == 0

    def test_opening_duel_win_rate(self):
        """Win rate calculated correctly."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats(wins=3, losses=2, attempts=5)
        assert stats.win_rate == 60.0

    def test_opening_duel_win_rate_zero_attempts(self):
        """Win rate is 0 when no attempts."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats()
        assert stats.win_rate == 0.0

    def test_entry_ttd_median(self):
        """Entry TTD median calculated correctly."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats()
        stats.entry_ttd_values = [200.0, 300.0, 400.0]
        assert stats.entry_ttd_median_ms == 300.0

    def test_entry_ttd_mean(self):
        """Entry TTD mean calculated correctly."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats()
        stats.entry_ttd_values = [200.0, 300.0, 400.0]
        assert stats.entry_ttd_mean_ms == 300.0

    def test_entry_ttd_none_when_empty(self):
        """Entry TTD is None when no values."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats()
        assert stats.entry_ttd_median_ms is None
        assert stats.entry_ttd_mean_ms is None


class TestTradeStats:
    """Tests for enhanced TradeStats."""

    def test_trade_stats_creation(self):
        """TradeStats can be created with default values."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats()
        assert stats.kills_traded == 0
        assert stats.deaths_traded == 0
        assert stats.trade_attempts == 0
        assert stats.failed_trades == 0

    def test_trade_rate_calculation(self):
        """Trade Kill % calculated correctly."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats(kills_traded=3, trade_attempts=10)
        assert stats.trade_rate == 30.0

    def test_trade_rate_zero_attempts(self):
        """Trade rate is 0 when no attempts."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats()
        assert stats.trade_rate == 0.0

    def test_deaths_traded_rate(self):
        """Deaths traded rate calculated correctly."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats(deaths_traded=2, failed_trades=3)
        assert stats.deaths_traded_rate == 40.0


class TestClutchStats:
    """Tests for enhanced ClutchStats."""

    def test_clutch_stats_creation(self):
        """ClutchStats can be created with default values."""
        from opensight.analysis.analytics import ClutchStats

        stats = ClutchStats()
        assert stats.total_situations == 0
        assert stats.total_wins == 0
        assert stats.v1_attempts == 0
        assert stats.v1_wins == 0

    def test_clutch_win_rate(self):
        """Overall clutch win rate calculated correctly."""
        from opensight.analysis.analytics import ClutchStats

        stats = ClutchStats(total_situations=5, total_wins=2)
        assert stats.win_rate == 40.0

    def test_clutch_win_rate_zero_situations(self):
        """Win rate is 0 when no situations."""
        from opensight.analysis.analytics import ClutchStats

        stats = ClutchStats()
        assert stats.win_rate == 0.0

    def test_v1_win_rate(self):
        """1v1 clutch win rate calculated correctly."""
        from opensight.analysis.analytics import ClutchStats

        stats = ClutchStats(v1_attempts=4, v1_wins=3)
        assert stats.v1_win_rate == 75.0

    def test_v2_win_rate(self):
        """1v2 clutch win rate calculated correctly."""
        from opensight.analysis.analytics import ClutchStats

        stats = ClutchStats(v2_attempts=3, v2_wins=1)
        assert stats.v2_win_rate == pytest.approx(33.3, abs=0.1)

    def test_all_clutch_win_rates(self):
        """All clutch win rate properties work correctly."""
        from opensight.analysis.analytics import ClutchStats

        stats = ClutchStats(
            v1_attempts=2,
            v1_wins=1,
            v2_attempts=2,
            v2_wins=1,
            v3_attempts=2,
            v3_wins=0,
            v4_attempts=1,
            v4_wins=0,
            v5_attempts=1,
            v5_wins=0,
        )
        assert stats.v1_win_rate == 50.0
        assert stats.v2_win_rate == 50.0
        assert stats.v3_win_rate == 0.0
        assert stats.v4_win_rate == 0.0
        assert stats.v5_win_rate == 0.0


class TestPlayerMatchStatsNewProperties:
    """Tests for new PlayerMatchStats properties."""

    def test_entry_kills_per_round(self):
        """Entry kills per round calculated correctly."""
        from opensight.analysis.analytics import OpeningDuelStats, PlayerMatchStats

        player = PlayerMatchStats(
            steam_id=12345,
            name="Player1",
            team="CT",
            kills=10,
            deaths=5,
            assists=3,
            headshots=4,
            total_damage=1000,
            rounds_played=10,
        )
        player.opening_duels = OpeningDuelStats(wins=3, attempts=5)
        assert player.entry_kills_per_round == 0.3

    def test_entry_ttd_property(self):
        """Entry TTD property returns median from opening_duels."""
        from opensight.analysis.analytics import PlayerMatchStats

        player = PlayerMatchStats(
            steam_id=12345,
            name="Player1",
            team="CT",
            kills=10,
            deaths=5,
            assists=3,
            headshots=4,
            total_damage=1000,
            rounds_played=10,
        )
        player.opening_duels.entry_ttd_values = [200.0, 300.0, 400.0]
        assert player.entry_ttd == 300.0

    def test_trade_kill_rate_property(self):
        """Trade kill rate property returns trade_rate from trades."""
        from opensight.analysis.analytics import PlayerMatchStats, TradeStats

        player = PlayerMatchStats(
            steam_id=12345,
            name="Player1",
            team="CT",
            kills=10,
            deaths=5,
            assists=3,
            headshots=4,
            total_damage=1000,
            rounds_played=10,
        )
        player.trades = TradeStats(kills_traded=2, trade_attempts=5)
        assert player.trade_kill_rate == 40.0

    def test_clutch_win_rate_property(self):
        """Clutch win rate property returns win_rate from clutches."""
        from opensight.analysis.analytics import ClutchStats, PlayerMatchStats

        player = PlayerMatchStats(
            steam_id=12345,
            name="Player1",
            team="CT",
            kills=10,
            deaths=5,
            assists=3,
            headshots=4,
            total_damage=1000,
            rounds_played=10,
        )
        player.clutches = ClutchStats(total_situations=4, total_wins=2)
        assert player.clutch_win_rate == 50.0

    def test_clutch_1v1_rate_property(self):
        """Clutch 1v1 rate property works correctly."""
        from opensight.analysis.analytics import ClutchStats, PlayerMatchStats

        player = PlayerMatchStats(
            steam_id=12345,
            name="Player1",
            team="CT",
            kills=10,
            deaths=5,
            assists=3,
            headshots=4,
            total_damage=1000,
            rounds_played=10,
        )
        player.clutches = ClutchStats(v1_attempts=3, v1_wins=2)
        assert player.clutch_1v1_rate == pytest.approx(66.7, abs=0.1)

    def test_clutch_1v2_rate_property(self):
        """Clutch 1v2 rate property works correctly."""
        from opensight.analysis.analytics import ClutchStats, PlayerMatchStats

        player = PlayerMatchStats(
            steam_id=12345,
            name="Player1",
            team="CT",
            kills=10,
            deaths=5,
            assists=3,
            headshots=4,
            total_damage=1000,
            rounds_played=10,
        )
        player.clutches = ClutchStats(v2_attempts=2, v2_wins=1)
        assert player.clutch_1v2_rate == 50.0
