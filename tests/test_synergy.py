"""
Tests for Team Synergy Analysis Module

Tests the synergy analysis features:
- Trade synergy between player pairs
- Flash assist networks
- Duo performance metrics
- Synergy score calculations
"""

import pytest

from opensight.domains.synergy import (
    PlayerPairSynergy,
    SynergyAnalysisResult,
    SynergyAnalyzer,
    analyze_synergy,
    synergy_to_dict,
)


class TestPlayerPairSynergy:
    """Tests for the PlayerPairSynergy dataclass."""

    def test_synergy_score_calculation(self):
        """Synergy score should be in 0-100 range."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123456789,
            player_b_steamid=987654321,
            player_a_name="PlayerA",
            player_b_name="PlayerB",
            trades_a_for_b=3,
            trades_b_for_a=2,
            avg_trade_time_ms=1500.0,
            flashes_a_for_b=5,
            flashes_b_for_a=3,
            rounds_together_alive=20,
            rounds_both_got_kills=8,
            duo_rounds_won=13,
            duo_rounds_total=20,
            avg_refrag_time_ms=2000.0,
        )

        score = synergy.synergy_score
        assert 0 <= score <= 100
        assert score > 0  # Should have positive synergy with these stats

    def test_synergy_score_zero_with_no_data(self):
        """Synergy score should be 0 with no activity."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
        )

        assert synergy.synergy_score == 0
        assert synergy.duo_win_rate == 0
        assert synergy.total_trades == 0
        assert synergy.total_flash_assists == 0

    def test_duo_win_rate_calculation(self):
        """Duo win rate should be correctly calculated."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
            duo_rounds_won=15,
            duo_rounds_total=20,
        )

        assert synergy.duo_win_rate == 0.75  # 15/20

    def test_duo_win_rate_zero_division(self):
        """Duo win rate should handle zero total rounds."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
            duo_rounds_won=0,
            duo_rounds_total=0,
        )

        assert synergy.duo_win_rate == 0.0

    def test_total_trades_property(self):
        """Total trades should sum both directions."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
            trades_a_for_b=3,
            trades_b_for_a=5,
        )

        assert synergy.total_trades == 8

    def test_total_flash_assists_property(self):
        """Total flash assists should sum both directions."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
            flashes_a_for_b=2,
            flashes_b_for_a=4,
        )

        assert synergy.total_flash_assists == 6

    def test_finalize_calculates_averages(self):
        """Finalize should calculate average times from collected data."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
        )
        synergy._trade_times = [1000.0, 2000.0, 3000.0]
        synergy._refrag_times = [500.0, 1500.0]

        synergy.finalize()

        assert synergy.avg_trade_time_ms == 2000.0  # mean of 1000, 2000, 3000
        assert synergy.avg_refrag_time_ms == 1000.0  # mean of 500, 1500

    def test_finalize_handles_empty_times(self):
        """Finalize should handle empty time lists gracefully."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
        )

        synergy.finalize()

        assert synergy.avg_trade_time_ms == 0.0
        assert synergy.avg_refrag_time_ms == 0.0


class TestSynergyToDict:
    """Tests for the synergy_to_dict serialization function."""

    def test_serialization_includes_all_fields(self):
        """Serialized dict should include all relevant fields."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123456789,
            player_b_steamid=987654321,
            player_a_name="Alice",
            player_b_name="Bob",
            trades_a_for_b=3,
            trades_b_for_a=2,
            avg_trade_time_ms=1234.5,
            flashes_a_for_b=4,
            flashes_b_for_a=1,
            duo_rounds_won=10,
            duo_rounds_total=15,
        )

        result = synergy_to_dict(synergy)

        assert result["player_a"] == "Alice"
        assert result["player_b"] == "Bob"
        assert result["player_a_steamid"] == "123456789"
        assert result["player_b_steamid"] == "987654321"
        assert result["trades_a_for_b"] == 3
        assert result["trades_b_for_a"] == 2
        assert result["trades_for_each_other"] == 5
        assert result["flashes_a_for_b"] == 4
        assert result["flashes_b_for_a"] == 1
        assert result["flash_assists_for_each_other"] == 5
        assert result["avg_trade_time_ms"] == 1234.5
        assert result["duo_win_rate"] == pytest.approx(66.7, rel=0.1)  # 10/15 * 100
        assert "synergy_score" in result

    def test_serialization_rounds_values(self):
        """Serialization should round floating point values."""
        synergy = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
            avg_trade_time_ms=1234.56789,
            duo_rounds_won=7,
            duo_rounds_total=9,  # 77.777...%
        )

        result = synergy_to_dict(synergy)

        assert result["avg_trade_time_ms"] == 1234.6  # Rounded to 1 decimal
        assert result["duo_win_rate"] == 77.8  # Rounded to 1 decimal


class TestSynergyAnalysisResult:
    """Tests for the SynergyAnalysisResult dataclass."""

    def test_empty_result_creation(self):
        """Should be able to create an empty result."""
        result = SynergyAnalysisResult(
            pair_synergies=[],
            best_duo=None,
            trade_network={},
            flash_network={},
        )

        assert len(result.pair_synergies) == 0
        assert result.best_duo is None
        assert len(result.trade_network) == 0
        assert len(result.flash_network) == 0

    def test_result_with_data(self):
        """Should store synergy data correctly."""
        pair = PlayerPairSynergy(
            player_a_steamid=123,
            player_b_steamid=456,
            player_a_name="A",
            player_b_name="B",
        )

        result = SynergyAnalysisResult(
            pair_synergies=[pair],
            best_duo=pair,
            trade_network={"A": [{"traded_for": "B", "count": 2}]},
            flash_network={"B": [{"flashed_for": "A", "count": 1}]},
        )

        assert len(result.pair_synergies) == 1
        assert result.best_duo == pair
        assert "A" in result.trade_network
        assert "B" in result.flash_network


class TestAnalyzeSynergyFunction:
    """Tests for the analyze_synergy convenience function."""

    def test_returns_dict_with_expected_keys(self):
        """analyze_synergy should return dict with expected structure."""
        # Create a minimal mock DemoData
        from unittest.mock import MagicMock

        import pandas as pd

        mock_data = MagicMock()
        mock_data.kills_df = pd.DataFrame()
        mock_data.blinds_df = pd.DataFrame()
        mock_data.rounds_df = pd.DataFrame()
        mock_data.player_persistent_teams = {}
        mock_data.player_teams = {}
        mock_data.player_names = {}
        mock_data.num_rounds = 0
        mock_data.tick_rate = 64
        mock_data.team_starting_sides = {}
        mock_data.halftime_round = 13

        result = analyze_synergy(mock_data)

        assert "pair_synergies" in result
        assert "best_duo" in result
        assert "trade_network" in result
        assert "flash_network" in result
        assert isinstance(result["pair_synergies"], list)


class TestSynergyAnalyzer:
    """Tests for the SynergyAnalyzer class."""

    def test_initialization_creates_pairs_for_teammates(self):
        """Analyzer should create pairs for players on same team."""
        from unittest.mock import MagicMock

        import pandas as pd

        mock_data = MagicMock()
        mock_data.kills_df = pd.DataFrame()
        mock_data.blinds_df = pd.DataFrame()
        mock_data.rounds_df = pd.DataFrame()
        mock_data.player_persistent_teams = {
            100: "Team A",
            101: "Team A",
            102: "Team A",
            200: "Team B",
            201: "Team B",
        }
        mock_data.player_teams = {}
        mock_data.player_names = {
            100: "Player1",
            101: "Player2",
            102: "Player3",
            200: "Enemy1",
            201: "Enemy2",
        }
        mock_data.num_rounds = 20
        mock_data.tick_rate = 64
        mock_data.team_starting_sides = {"Team A": "CT", "Team B": "T"}
        mock_data.halftime_round = 13

        analyzer = SynergyAnalyzer(mock_data)

        # Team A has 3 players = 3 pairs (3 choose 2)
        # Team B has 2 players = 1 pair (2 choose 2)
        # Total: 4 pairs
        assert len(analyzer._pair_synergies) == 4

    def test_empty_kills_returns_empty_result(self):
        """Analyzer should return empty result with no kill data."""
        from unittest.mock import MagicMock

        import pandas as pd

        mock_data = MagicMock()
        mock_data.kills_df = pd.DataFrame()
        mock_data.blinds_df = pd.DataFrame()
        mock_data.rounds_df = pd.DataFrame()
        mock_data.player_persistent_teams = {"100": "Team A", "101": "Team A"}
        mock_data.player_teams = {}
        mock_data.player_names = {"100": "A", "101": "B"}
        mock_data.num_rounds = 0
        mock_data.tick_rate = 64
        mock_data.team_starting_sides = {}
        mock_data.halftime_round = 13

        analyzer = SynergyAnalyzer(mock_data)
        result = analyzer.analyze()

        assert len(result.pair_synergies) == 0
        assert result.best_duo is None

    def test_ticks_to_ms_conversion(self):
        """Should correctly convert ticks to milliseconds."""
        from unittest.mock import MagicMock

        import pandas as pd

        mock_data = MagicMock()
        mock_data.kills_df = pd.DataFrame()
        mock_data.blinds_df = pd.DataFrame()
        mock_data.rounds_df = pd.DataFrame()
        mock_data.player_persistent_teams = {}
        mock_data.player_teams = {}
        mock_data.player_names = {}
        mock_data.num_rounds = 0
        mock_data.tick_rate = 64
        mock_data.team_starting_sides = {}
        mock_data.halftime_round = 13

        analyzer = SynergyAnalyzer(mock_data)

        # 64 ticks = 1 second = 1000ms
        assert analyzer._ticks_to_ms(64) == 1000.0
        assert analyzer._ticks_to_ms(128) == 2000.0
        assert analyzer._ticks_to_ms(32) == 500.0

    def test_are_teammates_check(self):
        """Should correctly identify teammates."""
        from unittest.mock import MagicMock

        import pandas as pd

        mock_data = MagicMock()
        mock_data.kills_df = pd.DataFrame()
        mock_data.blinds_df = pd.DataFrame()
        mock_data.rounds_df = pd.DataFrame()
        mock_data.player_persistent_teams = {
            100: "Team A",
            101: "Team A",
            200: "Team B",
        }
        mock_data.player_teams = {}
        mock_data.player_names = {100: "A", 101: "B", 200: "C"}
        mock_data.num_rounds = 0
        mock_data.tick_rate = 64
        mock_data.team_starting_sides = {}
        mock_data.halftime_round = 13

        analyzer = SynergyAnalyzer(mock_data)

        assert analyzer._are_teammates(100, 101) is True
        assert analyzer._are_teammates(100, 200) is False
        assert analyzer._are_teammates(999, 100) is False  # Unknown player
