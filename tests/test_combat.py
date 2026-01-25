"""Tests for the combat analysis module."""

from pathlib import Path

import pandas as pd
import pytest

from opensight.core.parser import DemoData
from opensight.domains.combat import (
    CombatAnalysisResult,
    CombatAnalyzer,
    analyze_combat,
)


class TestTradeKillDetection:
    """Tests for trade kill detection."""

    @pytest.fixture
    def demo_with_trade(self):
        """Create demo data with a trade kill scenario."""
        # Player 12345 (CT) kills Player 67890 (T) at tick 1000
        # Player 11111 (T) kills Player 12345 (CT) at tick 1200 (trade)
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1200],
                "attacker_steamid": [12345, 11111],
                "user_steamid": [67890, 12345],
                "weapon": ["ak47", "awp"],
                "headshot": [True, False],
                "total_rounds_played": [1, 1],
            }
        )

        return DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=15,
            player_stats={},
            player_names={12345: "Player1", 67890: "Player2", 11111: "Player3"},
            player_teams={12345: 3, 67890: 2, 11111: 2},  # CT=3, T=2
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    def test_detects_trade_kill(self, demo_with_trade):
        """Trade kill is detected within window."""
        analyzer = CombatAnalyzer(demo_with_trade)
        result = analyzer.analyze()

        assert len(result.trade_kills) == 1
        trade = result.trade_kills[0]
        assert trade.original_victim_id == 67890
        assert trade.trader_id == 11111
        assert trade.traded_player_id == 12345

    def test_trade_time_calculated(self, demo_with_trade):
        """Trade time is correctly calculated."""
        analyzer = CombatAnalyzer(demo_with_trade)
        result = analyzer.analyze()

        trade = result.trade_kills[0]
        # 200 ticks at 64 tick rate = 3125ms
        expected_ms = (200 / 64) * 1000
        assert trade.time_delta_ms == pytest.approx(expected_ms, rel=0.01)

    def test_no_trade_outside_window(self):
        """Kills outside trade window are not counted as trades."""
        # 10 second gap (outside 5s window)
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1000 + (10 * 64)],  # 10 seconds apart
                "attacker_steamid": [12345, 11111],
                "user_steamid": [67890, 12345],
                "weapon": ["ak47", "awp"],
                "total_rounds_played": [1, 1],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=15,
            player_stats={},
            player_names={12345: "Player1", 67890: "Player2", 11111: "Player3"},
            player_teams={12345: 3, 67890: 2, 11111: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        assert len(result.trade_kills) == 0


class TestOpeningDuelDetection:
    """Tests for opening duel detection."""

    @pytest.fixture
    def demo_with_rounds(self):
        """Create demo data with multiple rounds."""
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1500, 3000, 3500],
                "attacker_steamid": [12345, 67890, 67890, 12345],
                "user_steamid": [67890, 12345, 12345, 67890],
                "weapon": ["ak47", "m4a1", "awp", "ak47"],
                "headshot": [True, False, True, False],
                "total_rounds_played": [1, 1, 2, 2],
            }
        )

        return DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=2,
            player_stats={},
            player_names={12345: "Player1", 67890: "Player2"},
            player_teams={12345: 3, 67890: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    def test_detects_opening_duels(self, demo_with_rounds):
        """Opening duel is detected for each round."""
        analyzer = CombatAnalyzer(demo_with_rounds)
        result = analyzer.analyze()

        assert len(result.opening_duels) == 2

    def test_opening_duel_is_first_kill(self, demo_with_rounds):
        """Opening duel is the first kill of each round."""
        analyzer = CombatAnalyzer(demo_with_rounds)
        result = analyzer.analyze()

        round1_opening = [o for o in result.opening_duels if o.round_num == 1][0]
        assert round1_opening.winner_id == 12345
        assert round1_opening.loser_id == 67890

        round2_opening = [o for o in result.opening_duels if o.round_num == 2][0]
        assert round2_opening.winner_id == 67890
        assert round2_opening.loser_id == 12345


class TestMultiKillDetection:
    """Tests for multi-kill detection."""

    @pytest.fixture
    def demo_with_ace(self):
        """Create demo data with an ace (5 kills)."""
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1100, 1200, 1300, 1400],
                "attacker_steamid": [12345, 12345, 12345, 12345, 12345],
                "user_steamid": [1, 2, 3, 4, 5],
                "weapon": ["ak47", "ak47", "ak47", "ak47", "ak47"],
                "headshot": [True, True, True, True, True],
                "total_rounds_played": [1, 1, 1, 1, 1],
            }
        )

        return DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={12345: "Ace Player", 1: "V1", 2: "V2", 3: "V3", 4: "V4", 5: "V5"},
            player_teams={12345: 3, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    def test_detects_ace(self, demo_with_ace):
        """Ace (5k) is detected."""
        analyzer = CombatAnalyzer(demo_with_ace)
        result = analyzer.analyze()

        assert len(result.multi_kills) == 1
        mk = result.multi_kills[0]
        assert mk.kill_count == 5
        assert mk.player_id == 12345

    def test_all_headshots_tracked(self, demo_with_ace):
        """All headshot aces are tracked."""
        analyzer = CombatAnalyzer(demo_with_ace)
        result = analyzer.analyze()

        mk = result.multi_kills[0]
        assert mk.all_headshots is True


class TestClutchDetection:
    """Tests for clutch detection."""

    @pytest.fixture
    def demo_with_clutch(self):
        """Create demo data with a 1v2 clutch."""
        # Round where all CTs die except one, then clutcher gets 2 kills
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1100, 1200, 1300, 2000, 2100],
                "attacker_steamid": [1, 1, 1, 1, 12345, 12345],
                "user_steamid": [101, 102, 103, 104, 1, 2],  # T kills 4 CTs, then CT 12345 clutches
                "weapon": ["ak47"] * 6,
                "headshot": [False] * 6,
                "total_rounds_played": [1, 1, 1, 1, 1, 1],
            }
        )

        return DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={
                12345: "Clutcher",
                101: "CT1",
                102: "CT2",
                103: "CT3",
                104: "CT4",
                1: "T1",
                2: "T2",
            },
            player_teams={
                12345: 3,
                101: 3,
                102: 3,
                103: 3,
                104: 3,
                1: 2,
                2: 2,
            },
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    def test_detects_clutch_situation(self, demo_with_clutch):
        """Clutch situation is detected."""
        analyzer = CombatAnalyzer(demo_with_clutch)
        result = analyzer.analyze()

        assert len(result.clutch_situations) >= 1

    def test_clutch_scenario_identified(self, demo_with_clutch):
        """Clutch scenario (1vX) is correctly identified."""
        analyzer = CombatAnalyzer(demo_with_clutch)
        result = analyzer.analyze()

        clutch = result.clutch_situations[0]
        assert clutch.clutcher_id == 12345
        assert "1v" in clutch.scenario


class TestPlayerCombatStats:
    """Tests for player combat statistics."""

    def test_player_stats_calculated(self):
        """Player stats are calculated correctly."""
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 2000],
                "attacker_steamid": [12345, 12345],
                "user_steamid": [67890, 11111],
                "weapon": ["ak47", "ak47"],
                "headshot": [True, False],
                "total_rounds_played": [1, 2],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=2,
            player_stats={},
            player_names={12345: "Player1", 67890: "Player2", 11111: "Player3"},
            player_teams={12345: 3, 67890: 2, 11111: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        assert 12345 in result.player_stats
        stats = result.player_stats[12345]
        assert stats.opening_kills == 2  # First kill in both rounds


class TestAnalyzeCombatFunction:
    """Tests for analyze_combat convenience function."""

    def test_analyze_combat_creates_analyzer(self):
        """analyze_combat creates analyzer and runs analysis."""
        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
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

        result = analyze_combat(demo)
        assert isinstance(result, CombatAnalysisResult)
