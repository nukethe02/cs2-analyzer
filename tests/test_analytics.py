"""Tests for the analytics module."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from opensight.analysis.analytics import (
    AnalysisValidator,
    CrosshairPlacementResult,
    DemoAnalyzer,
    EngagementResult,
    PlayerAnalytics,
    PlayerMatchStats,
    ValidationResult,
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
    """Tests for EngagementResult (legacy TTDResult) dataclass."""

    def test_ttd_result_creation(self):
        """EngagementResult can be created with required fields."""
        result = EngagementResult(
            tick_first_damage=1000,
            tick_kill=1100,
            duration_ticks=100,
            duration_ms=1562.5,
            attacker_steamid=12345,
            victim_steamid=67890,
            weapon="ak47",
            headshot=True,
        )
        assert result.tick_first_damage == 1000
        assert result.duration_ms == 1562.5
        assert result.tick_kill == 1100


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
            engagement_duration_values=[300.0, 350.0, 400.0],
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

    def test_death_counting_with_user_steamid_column(self):
        """Verify deaths are counted correctly with demoparser2's user_steamid column."""
        # demoparser2 uses 'user_steamid' for victim, NOT 'victim_steamid'
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 2000, 3000, 4000],
                "attacker_steamid": [111, 222, 111, 222],
                "user_steamid": [222, 111, 222, 111],  # demoparser2 convention
                "weapon": ["ak47", "awp", "m4a1", "deagle"],
                "headshot": [True, False, True, False],
            }
        )

        demo_data = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_mirage",
            duration_seconds=900.0,
            tick_rate=64,
            num_rounds=8,
            player_stats={},
            player_names={111: "PlayerA", 222: "PlayerB"},
            player_teams={111: "CT", 222: "T"},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = DemoAnalyzer(demo_data)
        results = analyzer.analyze()

        # PlayerA: 2 kills (tick 1000, 3000), 2 deaths (tick 2000, 4000)
        # PlayerB: 2 kills (tick 2000, 4000), 2 deaths (tick 1000, 3000)
        assert results.players[111].kills == 2
        assert results.players[111].deaths == 2, "Deaths must be counted from user_steamid column"
        assert results.players[222].kills == 2
        assert results.players[222].deaths == 2, "Deaths must be counted from user_steamid column"

    def test_death_counting_with_victim_steamid_column(self):
        """Verify deaths are counted correctly with awpy's victim_steamid column."""
        # awpy uses 'victim_steamid' for victim
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 2000, 3000],
                "attacker_steamid": [111, 222, 111],
                "victim_steamid": [222, 111, 222],  # awpy convention
                "weapon": ["ak47", "awp", "m4a1"],
                "headshot": [True, False, True],
            }
        )

        demo_data = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_mirage",
            duration_seconds=900.0,
            tick_rate=64,
            num_rounds=8,
            player_stats={},
            player_names={111: "PlayerA", 222: "PlayerB"},
            player_teams={111: "CT", 222: "T"},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = DemoAnalyzer(demo_data)
        results = analyzer.analyze()

        # PlayerA: 2 kills, 1 death | PlayerB: 1 kill, 2 deaths
        assert results.players[111].kills == 2
        assert results.players[111].deaths == 1, "Deaths must be counted from victim_steamid column"
        assert results.players[222].kills == 1
        assert results.players[222].deaths == 2, "Deaths must be counted from victim_steamid column"


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

    def test_zone_fields_default(self):
        """Zone-based fields have correct defaults."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats()
        assert stats.map_control_kills == 0
        assert stats.site_kills == 0
        assert stats.kill_zones == {}

    def test_map_control_rate(self):
        """Map control rate calculated correctly."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats(wins=10, map_control_kills=7, site_kills=3)
        assert stats.map_control_rate == 70.0

    def test_map_control_rate_zero_wins(self):
        """Map control rate is 0 when no wins."""
        from opensight.analysis.analytics import OpeningDuelStats

        stats = OpeningDuelStats()
        assert stats.map_control_rate == 0.0


class TestOpeningEngagementStats:
    """Tests for OpeningEngagementStats - damage-based engagement tracking."""

    def test_creation_with_defaults(self):
        """OpeningEngagementStats can be created with default values."""
        from opensight.analysis.analytics import OpeningEngagementStats

        stats = OpeningEngagementStats()
        assert stats.engagement_attempts == 0
        assert stats.engagement_wins == 0
        assert stats.engagement_losses == 0
        assert stats.first_damage_dealt == 0
        assert stats.first_damage_taken == 0
        assert stats.opening_damage_total == 0
        assert stats.opening_damage_values == []

    def test_engagement_win_rate(self):
        """Engagement win rate calculated correctly."""
        from opensight.analysis.analytics import OpeningEngagementStats

        stats = OpeningEngagementStats(engagement_attempts=10, engagement_wins=6)
        assert stats.engagement_win_rate == 60.0

    def test_engagement_win_rate_zero_attempts(self):
        """Engagement win rate is 0 when no attempts."""
        from opensight.analysis.analytics import OpeningEngagementStats

        stats = OpeningEngagementStats()
        assert stats.engagement_win_rate == 0.0

    def test_first_damage_rate(self):
        """First damage rate calculated correctly."""
        from opensight.analysis.analytics import OpeningEngagementStats

        stats = OpeningEngagementStats(engagement_attempts=10, first_damage_dealt=3)
        assert stats.first_damage_rate == 30.0

    def test_first_damage_rate_zero_attempts(self):
        """First damage rate is 0 when no attempts."""
        from opensight.analysis.analytics import OpeningEngagementStats

        stats = OpeningEngagementStats()
        assert stats.first_damage_rate == 0.0

    def test_opening_damage_avg(self):
        """Opening damage average calculated correctly."""
        from opensight.analysis.analytics import OpeningEngagementStats

        stats = OpeningEngagementStats()
        stats.opening_damage_values = [50, 100, 75]
        assert stats.opening_damage_avg == 75.0

    def test_opening_damage_avg_empty(self):
        """Opening damage average is 0 when no values."""
        from opensight.analysis.analytics import OpeningEngagementStats

        stats = OpeningEngagementStats()
        assert stats.opening_damage_avg == 0.0


class TestEntryFragStats:
    """Tests for EntryFragStats - zone-aware entry frag tracking."""

    def test_creation_with_defaults(self):
        """EntryFragStats can be created with default values."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats()
        assert stats.a_site_entries == 0
        assert stats.a_site_entry_deaths == 0
        assert stats.b_site_entries == 0
        assert stats.b_site_entry_deaths == 0
        assert stats.total_entry_frags == 0
        assert stats.total_entry_deaths == 0
        assert stats.entry_rounds_won == 0
        assert stats.entry_rounds_lost == 0

    def test_entry_frag_rate(self):
        """Entry frag rate calculated correctly."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats(total_entry_frags=7, total_entry_deaths=3)
        assert stats.entry_frag_rate == 70.0

    def test_entry_frag_rate_zero_total(self):
        """Entry frag rate is 0 when no entries."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats()
        assert stats.entry_frag_rate == 0.0

    def test_a_site_success_rate(self):
        """A site success rate calculated correctly."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats(a_site_entries=4, a_site_entry_deaths=1)
        assert stats.a_site_success_rate == 80.0

    def test_a_site_success_rate_zero(self):
        """A site success rate is 0 when no A site action."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats()
        assert stats.a_site_success_rate == 0.0

    def test_b_site_success_rate(self):
        """B site success rate calculated correctly."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats(b_site_entries=3, b_site_entry_deaths=2)
        assert stats.b_site_success_rate == 60.0

    def test_b_site_success_rate_zero(self):
        """B site success rate is 0 when no B site action."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats()
        assert stats.b_site_success_rate == 0.0

    def test_entry_round_win_rate(self):
        """Entry round win rate calculated correctly."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats(entry_rounds_won=8, entry_rounds_lost=2)
        assert stats.entry_round_win_rate == 80.0

    def test_entry_round_win_rate_zero(self):
        """Entry round win rate is 0 when no entry rounds."""
        from opensight.analysis.analytics import EntryFragStats

        stats = EntryFragStats()
        assert stats.entry_round_win_rate == 0.0


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
        """Trade Kill % calculated correctly (Leetify-style)."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats(trade_kill_opportunities=10, trade_kill_success=3, kills_traded=3)
        assert stats.trade_rate == 30.0

    def test_trade_rate_zero_attempts(self):
        """Trade rate is 0 when no opportunities."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats()
        assert stats.trade_rate == 0.0

    def test_deaths_traded_rate(self):
        """Deaths traded rate calculated correctly (Leetify-style)."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats(traded_death_opportunities=5, traded_death_success=2, deaths_traded=2)
        assert stats.deaths_traded_rate == 40.0

    def test_trade_kill_attempts_pct(self):
        """Trade Kill Attempts % calculated correctly."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats(trade_kill_opportunities=10, trade_kill_attempts=7)
        assert stats.trade_kill_attempts_pct == 70.0

    def test_trade_kill_success_pct(self):
        """Trade Kill Success % calculated correctly."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats(trade_kill_attempts=8, trade_kill_success=6)
        assert stats.trade_kill_success_pct == 75.0

    def test_traded_death_attempts_pct(self):
        """Traded Death Attempts % calculated correctly."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats(traded_death_opportunities=10, traded_death_attempts=8)
        assert stats.traded_death_attempts_pct == 80.0

    def test_traded_death_success_pct(self):
        """Traded Death Success % calculated correctly."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats(traded_death_attempts=5, traded_death_success=4)
        assert stats.traded_death_success_pct == 80.0

    def test_avg_time_to_trade_ms(self):
        """Average time to trade in milliseconds."""
        from opensight.analysis.analytics import TradeStats

        # 64 ticks = 1000 ms
        stats = TradeStats(time_to_trade_ticks=[64, 128, 192])  # 1000, 2000, 3000 ms
        avg = stats.avg_time_to_trade_ms
        assert avg is not None
        assert abs(avg - 2000.0) < 1  # Should be ~2000ms

    def test_median_time_to_trade_ms(self):
        """Median time to trade in milliseconds."""
        from opensight.analysis.analytics import TradeStats

        # 64 ticks = 1000 ms
        stats = TradeStats(time_to_trade_ticks=[64, 128, 192])  # 1000, 2000, 3000 ms
        median = stats.median_time_to_trade_ms
        assert median is not None
        assert abs(median - 2000.0) < 1  # Should be 2000ms (middle value)

    def test_time_to_trade_empty(self):
        """Time to trade is None when no trades."""
        from opensight.analysis.analytics import TradeStats

        stats = TradeStats()
        assert stats.avg_time_to_trade_ms is None
        assert stats.median_time_to_trade_ms is None


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
        player.trades = TradeStats(
            trade_kill_opportunities=5,
            trade_kill_success=2,
            kills_traded=2,
        )
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


class TestAimStats:
    """Tests for comprehensive aim statistics (Leetify style)."""

    def test_aim_stats_creation(self):
        """AimStats can be created with default values."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats()
        assert stats.shots_fired == 0
        assert stats.shots_hit == 0
        assert stats.spray_shots_fired == 0
        assert stats.counter_strafe_kills == 0

    def test_accuracy_all_calculation(self):
        """Overall accuracy calculated correctly."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(shots_fired=100, shots_hit=35)
        assert stats.accuracy_all == 35.0

    def test_accuracy_all_zero_shots(self):
        """Overall accuracy is 0 when no shots fired."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats()
        assert stats.accuracy_all == 0.0

    def test_head_accuracy_calculation(self):
        """Head accuracy calculated correctly."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(shots_hit=50, headshot_hits=15)
        assert stats.head_accuracy == 30.0

    def test_head_accuracy_zero_hits(self):
        """Head accuracy is 0 when no hits."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats()
        assert stats.head_accuracy == 0.0

    def test_hs_kill_pct_calculation(self):
        """HS kill percentage calculated correctly."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(total_kills=20, headshot_kills=10)
        assert stats.hs_kill_pct == 50.0

    def test_spray_accuracy_calculation(self):
        """Spray accuracy calculated correctly."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(spray_shots_fired=50, spray_shots_hit=20)
        assert stats.spray_accuracy == 40.0

    def test_spray_accuracy_zero_spray_shots(self):
        """Spray accuracy is 0 when no spray shots."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats()
        assert stats.spray_accuracy == 0.0

    def test_counter_strafe_pct_calculation(self):
        """Counter-strafing percentage calculated correctly (shot-based, Leetify parity)."""
        from opensight.analysis.analytics import AimStats

        # NEW: Shot-based tracking - 80 stationary shots out of 100 total = 80%
        stats = AimStats(shots_stationary=80, shots_with_velocity=100)
        assert stats.counter_strafe_pct == 80.0

    def test_counter_strafe_pct_zero_shots(self):
        """Counter-strafing percentage is 0 when no tracked shots."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats()
        assert stats.counter_strafe_pct == 0.0

    def test_ttd_rating_elite(self):
        """TTD rating is elite for <200ms."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(ttd_median_ms=180.0)
        assert stats._get_ttd_rating() == "elite"

    def test_ttd_rating_good(self):
        """TTD rating is good for 200-350ms."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(ttd_median_ms=275.0)
        assert stats._get_ttd_rating() == "good"

    def test_ttd_rating_average(self):
        """TTD rating is average for 350-500ms."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(ttd_median_ms=425.0)
        assert stats._get_ttd_rating() == "average"

    def test_ttd_rating_slow(self):
        """TTD rating is slow for >500ms."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(ttd_median_ms=550.0)
        assert stats._get_ttd_rating() == "slow"

    def test_ttd_rating_unknown(self):
        """TTD rating is unknown when None."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats()
        assert stats._get_ttd_rating() == "unknown"

    def test_cp_rating_elite(self):
        """CP rating is elite for <5 degrees."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(cp_median_deg=3.5)
        assert stats._get_cp_rating() == "elite"

    def test_cp_rating_good(self):
        """CP rating is good for 5-15 degrees."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(cp_median_deg=10.0)
        assert stats._get_cp_rating() == "good"

    def test_cp_rating_average(self):
        """CP rating is average for 15-25 degrees."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(cp_median_deg=20.0)
        assert stats._get_cp_rating() == "average"

    def test_cp_rating_needs_work(self):
        """CP rating is needs_work for >25 degrees."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(cp_median_deg=30.0)
        assert stats._get_cp_rating() == "needs_work"

    def test_aim_stats_to_dict(self):
        """AimStats to_dict serializes correctly."""
        from opensight.analysis.analytics import AimStats

        stats = AimStats(
            shots_fired=100,
            shots_hit=40,
            headshot_hits=12,
            spray_shots_fired=30,
            spray_shots_hit=10,
            # NEW: Shot-based counter-strafe tracking (Leetify parity)
            shots_stationary=50,
            shots_with_velocity=80,
            # DEPRECATED: Kill-based tracking (kept for backward compatibility)
            counter_strafe_kills=5,
            total_kills_for_cs=8,
            ttd_median_ms=280.0,
            cp_median_deg=8.5,
            total_kills=15,
            headshot_kills=6,
        )
        result = stats.to_dict()

        assert result["shots_fired"] == 100
        assert result["accuracy_all"] == 40.0
        assert result["head_accuracy"] == 30.0
        assert result["spray_accuracy"] == pytest.approx(33.3, abs=0.1)
        # NEW: Shot-based counter-strafe % (50/80 = 62.5%)
        assert result["counter_strafe_pct"] == 62.5
        # NEW: Raw shot counts in output
        assert result["shots_stationary"] == 50
        assert result["shots_with_velocity"] == 80
        assert result["time_to_damage_ms"] == 280.0
        assert result["crosshair_placement_deg"] == 8.5
        assert result["ttd_rating"] == "good"
        assert result["cp_rating"] == "good"


class TestPlayerMatchStatsAimProperties:
    """Tests for aim-related properties on PlayerMatchStats."""

    def test_spray_accuracy_property(self):
        """Spray accuracy property calculated correctly."""
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
        player.spray_shots_fired = 40
        player.spray_shots_hit = 16
        assert player.spray_accuracy == 40.0

    def test_counter_strafe_pct_property(self):
        """Counter-strafing percentage property calculated correctly (shot-based, Leetify parity)."""
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
        # NEW: Shot-based tracking - 60 stationary shots out of 80 total = 75%
        player.shots_stationary = 60
        player.shots_with_velocity = 80
        assert player.counter_strafe_pct == 75.0

    def test_aim_stats_property(self):
        """aim_stats property returns AimStats object."""
        from opensight.analysis.analytics import AimStats, PlayerMatchStats

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
        player.shots_fired = 100
        player.shots_hit = 35
        player.headshot_hits = 10
        player.spray_shots_fired = 30
        player.spray_shots_hit = 12
        # NEW: Shot-based counter-strafe tracking
        player.shots_stationary = 70
        player.shots_with_velocity = 100
        # DEPRECATED: Kill-based tracking (kept for backward compatibility)
        player.counter_strafe_kills = 7
        player.total_kills_with_velocity = 10
        player.engagement_duration_values = [250.0, 300.0, 350.0]
        player.cp_values = [5.0, 8.0, 10.0]

        aim_stats = player.aim_stats

        assert isinstance(aim_stats, AimStats)
        assert aim_stats.shots_fired == 100
        assert aim_stats.shots_hit == 35
        assert aim_stats.spray_shots_fired == 30
        assert aim_stats.spray_shots_hit == 12
        # NEW: Shot-based tracking
        assert aim_stats.shots_stationary == 70
        assert aim_stats.shots_with_velocity == 100
        # DEPRECATED: Kill-based tracking (still passed through for backward compat)
        assert aim_stats.counter_strafe_kills == 7
        assert aim_stats.total_kills_for_cs == 10
        assert aim_stats.total_kills == 10
        assert aim_stats.headshot_kills == 4


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result_is_truthy(self):
        """Valid result should evaluate to True."""
        result = ValidationResult(is_valid=True)
        assert result
        assert bool(result) is True

    def test_invalid_result_is_falsy(self):
        """Invalid result should evaluate to False."""
        result = ValidationResult(is_valid=False, errors=["Some error"])
        assert not result
        assert bool(result) is False


class TestAnalysisValidator:
    """Tests for the AnalysisValidator class."""

    def test_empty_players_is_invalid(self):
        """Empty players dict should return invalid result."""
        validator = AnalysisValidator()
        result = validator.validate({}, 30)
        assert not result.is_valid
        assert "No players" in result.errors[0]

    def test_valid_players_is_valid(self):
        """Normal players should pass validation."""
        validator = AnalysisValidator()
        players = {
            1: PlayerMatchStats(
                steam_id=1,
                name="Player1",
                team="CT",
                kills=10,
                deaths=8,
                assists=5,
                headshots=4,
                total_damage=2000,
                rounds_played=30,
            ),
            2: PlayerMatchStats(
                steam_id=2,
                name="Player2",
                team="T",
                kills=8,
                deaths=10,
                assists=3,
                headshots=3,
                total_damage=1800,
                rounds_played=30,
            ),
        }
        result = validator.validate(players, 30)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_negative_kills_is_error(self):
        """Negative kills should be flagged as error."""
        validator = AnalysisValidator()
        players = {
            1: PlayerMatchStats(
                steam_id=1,
                name="BadPlayer",
                team="CT",
                kills=-5,
                deaths=10,
                assists=0,
                headshots=0,
                total_damage=0,
                rounds_played=30,
            ),
        }
        result = validator.validate(players, 30)
        assert not result.is_valid
        assert any("Negative kills" in e for e in result.errors)

    def test_negative_deaths_is_error(self):
        """Negative deaths should be flagged as error."""
        validator = AnalysisValidator()
        players = {
            1: PlayerMatchStats(
                steam_id=1,
                name="BadPlayer",
                team="CT",
                kills=10,
                deaths=-5,
                assists=0,
                headshots=0,
                total_damage=0,
                rounds_played=30,
            ),
        }
        result = validator.validate(players, 30)
        assert not result.is_valid
        assert any("Negative deaths" in e for e in result.errors)

    def test_negative_rounds_survived_is_error(self):
        """More deaths than rounds played should be flagged."""
        validator = AnalysisValidator()
        players = {
            1: PlayerMatchStats(
                steam_id=1,
                name="BadPlayer",
                team="CT",
                kills=5,
                deaths=35,
                assists=0,
                headshots=0,
                total_damage=0,
                rounds_played=30,  # 30 - 35 = -5 survived
            ),
        }
        result = validator.validate(players, 30)
        assert not result.is_valid
        assert any("Negative rounds survived" in e for e in result.errors)

    def test_kill_death_imbalance_is_warning(self):
        """Significant kill/death imbalance should be a warning."""
        validator = AnalysisValidator()
        # 20 kills but only 10 deaths - big imbalance
        players = {
            1: PlayerMatchStats(
                steam_id=1,
                name="Player1",
                team="CT",
                kills=20,
                deaths=5,
                assists=0,
                headshots=0,
                total_damage=0,
                rounds_played=30,
            ),
            2: PlayerMatchStats(
                steam_id=2,
                name="Player2",
                team="T",
                kills=0,
                deaths=5,
                assists=0,
                headshots=0,
                total_damage=0,
                rounds_played=30,
            ),
        }
        result = validator.validate(players, 30)
        # Should be valid (imbalance is a warning, not error)
        assert result.is_valid
        assert any("imbalance" in w.lower() for w in result.warnings)

    def test_few_players_is_warning(self):
        """Less than 10 players should be a warning."""
        validator = AnalysisValidator()
        players = {
            1: PlayerMatchStats(
                steam_id=1,
                name="Player1",
                team="CT",
                kills=10,
                deaths=10,
                assists=0,
                headshots=0,
                total_damage=0,
                rounds_played=30,
            ),
        }
        result = validator.validate(players, 30)
        assert result.is_valid  # Valid but with warning
        assert any("Less than 10 players" in w for w in result.warnings)
