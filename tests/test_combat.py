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


class TestSprayTransferDetection:
    """Tests for spray transfer detection (2+ kills in one spray)."""

    @pytest.fixture
    def demo_with_double_spray(self):
        """Create demo data with a 2-kill spray transfer."""
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1100],  # ~156ms apart at 64 tick
                "attacker_steamid": [12345, 12345],
                "user_steamid": [1, 2],
                "user_name": ["Victim1", "Victim2"],
                "weapon": ["ak47", "ak47"],
                "headshot": [False, True],
                "total_rounds_played": [1, 1],
            }
        )

        return DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={12345: "Spray Master", 1: "Victim1", 2: "Victim2"},
            player_teams={12345: 3, 1: 2, 2: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    @pytest.fixture
    def demo_with_triple_spray(self):
        """Create demo data with a 3-kill spray transfer."""
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1080, 1160],  # All within 3 seconds
                "attacker_steamid": [12345, 12345, 12345],
                "user_steamid": [1, 2, 3],
                "user_name": ["V1", "V2", "V3"],
                "weapon": ["m4a1", "m4a1", "m4a1"],
                "headshot": [True, False, False],
                "total_rounds_played": [2, 2, 2],
            }
        )

        return DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=2,
            player_stats={},
            player_names={12345: "Triple Spray", 1: "V1", 2: "V2", 3: "V3"},
            player_teams={12345: 3, 1: 2, 2: 2, 3: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    def test_double_spray_detected(self, demo_with_double_spray):
        """Two kills within 3 seconds with same weapon = spray transfer."""
        analyzer = CombatAnalyzer(demo_with_double_spray)
        result = analyzer.analyze()

        assert len(result.spray_transfers) == 1
        st = result.spray_transfers[0]
        assert st.kills_in_spray == 2
        assert st.player_steamid == 12345
        assert st.weapon == "ak47"
        assert len(st.victims) == 2

    def test_triple_spray_detected(self, demo_with_triple_spray):
        """Three kills in one spray is detected."""
        analyzer = CombatAnalyzer(demo_with_triple_spray)
        result = analyzer.analyze()

        assert len(result.spray_transfers) == 1
        st = result.spray_transfers[0]
        assert st.kills_in_spray == 3
        assert st.is_triple_spray is True

    def test_no_spray_with_pistol(self):
        """Pistol kills don't count as spray transfers."""
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1100],
                "attacker_steamid": [12345, 12345],
                "user_steamid": [1, 2],
                "weapon": ["glock", "glock"],
                "total_rounds_played": [1, 1],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_names={12345: "Pistol Player"},
            player_teams={12345: 3, 1: 2, 2: 2},
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        assert len(result.spray_transfers) == 0

    def test_no_spray_timeout(self):
        """Kills more than 3 seconds apart don't count."""
        # 3 seconds = 192 ticks at 64 tick rate
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1300],  # 300 ticks = ~4.7 seconds
                "attacker_steamid": [12345, 12345],
                "user_steamid": [1, 2],
                "weapon": ["ak47", "ak47"],
                "total_rounds_played": [1, 1],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_names={12345: "Slow Sprayer"},
            player_teams={12345: 3, 1: 2, 2: 2},
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        assert len(result.spray_transfers) == 0

    def test_spray_time_calculated(self, demo_with_double_spray):
        """Spray time span is calculated correctly."""
        analyzer = CombatAnalyzer(demo_with_double_spray)
        result = analyzer.analyze()

        st = result.spray_transfers[0]
        # 100 ticks at 64 tick rate = 1562.5ms
        expected_ms = (100 / 64) * 1000
        assert abs(st.time_span_ms - expected_ms) < 1.0


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


class TestTradeChainDetection:
    """Tests for trade chain detection."""

    @pytest.fixture
    def demo_with_simple_chain(self):
        """Demo with a simple 2-kill chain (one trade).

        Scenario:
        - CT (12345) kills T (67890) at tick 1000
        - T (11111) kills CT (12345) at tick 1200 (trade, 200 ticks = 3.125s)
        """
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
            num_rounds=1,
            player_stats={},
            player_names={12345: "CT_Player", 67890: "T_Victim", 11111: "T_Trader"},
            player_teams={12345: 3, 67890: 2, 11111: 2},  # CT=3, T=2
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    @pytest.fixture
    def demo_with_extended_chain(self):
        """Demo with a 4-kill extended chain.

        Scenario (round 1):
        - A (CT, 1001) kills B (T, 2001) at tick 1000
        - C (T, 2002) kills A (CT, 1001) at tick 1200 (trade for B)
        - D (CT, 1002) kills C (T, 2002) at tick 1400 (trade for A)
        - E (T, 2003) kills D (CT, 1002) at tick 1600 (trade for C)
        """
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1200, 1400, 1600],
                "attacker_steamid": [1001, 2002, 1002, 2003],
                "user_steamid": [2001, 1001, 2002, 1002],
                "weapon": ["ak47", "m4a1", "awp", "ak47"],
                "headshot": [False, True, False, False],
                "total_rounds_played": [1, 1, 1, 1],
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
                1001: "CT_A",
                1002: "CT_D",
                2001: "T_B",
                2002: "T_C",
                2003: "T_E",
            },
            player_teams={
                1001: 3,  # CT
                1002: 3,  # CT
                2001: 2,  # T
                2002: 2,  # T
                2003: 2,  # T
            },
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    def test_detects_simple_chain(self, demo_with_simple_chain):
        """Simple trade is detected as a 2-kill chain."""
        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo_with_simple_chain)
        result = analyzer.analyze()

        assert len(result.trade_chains) == 1
        chain = result.trade_chains[0]
        assert chain.chain_length == 2
        assert not chain.is_extended  # 2 is not extended (need 3+)

    def test_detects_extended_chain(self, demo_with_extended_chain):
        """Extended trade chain (4 kills) is detected."""
        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo_with_extended_chain)
        result = analyzer.analyze()

        assert len(result.trade_chains) == 1
        chain = result.trade_chains[0]
        assert chain.chain_length == 4
        assert chain.is_extended  # 4 kills is extended

    def test_chain_winning_team(self, demo_with_extended_chain):
        """Winning team is the team of the final killer."""
        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo_with_extended_chain)
        result = analyzer.analyze()

        chain = result.trade_chains[0]
        # Last killer is T (2003), so T wins the chain
        assert chain.winning_team == "T"

    def test_chain_duration_calculated(self, demo_with_extended_chain):
        """Chain duration is calculated from first to last kill."""
        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo_with_extended_chain)
        result = analyzer.analyze()

        chain = result.trade_chains[0]
        # 600 ticks at 64 tick rate = 9375ms
        expected_ms = (600 / 64) * 1000
        assert abs(chain.duration_ms - expected_ms) < 1.0

    def test_animation_frames_generated(self, demo_with_simple_chain):
        """Animation frames are generated correctly."""
        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo_with_simple_chain)
        result = analyzer.analyze()

        chain = result.trade_chains[0]
        frames = chain.to_animation_frames()

        assert len(frames) == 2
        assert frames[0]["kill_type"] == "trigger"
        assert frames[0]["sequence"] == 0
        assert frames[1]["kill_type"] == "trade"
        assert frames[1]["sequence"] == 1
        assert "trade_time_ms" in frames[1]

    def test_no_chain_outside_window(self):
        """Kills outside trade window don't form chains."""
        # 10 second gap between kills (outside 5s window)
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1000 + (10 * 64)],  # 10 seconds apart
                "attacker_steamid": [12345, 11111],
                "user_steamid": [67890, 12345],
                "weapon": ["ak47", "awp"],
                "headshot": [False, False],
                "total_rounds_played": [1, 1],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={12345: "CT_Player", 67890: "T_Victim", 11111: "T_Trader"},
            player_teams={12345: 3, 67890: 2, 11111: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        # No chains should be detected (too far apart)
        assert len(result.trade_chains) == 0

    def test_chain_stats_aggregated(self, demo_with_extended_chain):
        """Chain statistics are correctly aggregated."""
        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo_with_extended_chain)
        result = analyzer.analyze()

        stats = result.trade_chain_stats
        assert stats is not None
        assert stats.total_chains == 1
        assert stats.max_chain_length == 4
        assert stats.avg_chain_length == 4.0
        assert stats.extended_chain_count == 1
        assert stats.longest_chain is not None

    def test_chain_to_dict_serialization(self, demo_with_simple_chain):
        """Chain can be serialized to dict for JSON."""
        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo_with_simple_chain)
        result = analyzer.analyze()

        chain = result.trade_chains[0]
        chain_dict = chain.to_dict()

        assert "chain_id" in chain_dict
        assert "round_num" in chain_dict
        assert "chain_length" in chain_dict
        assert "duration_ms" in chain_dict
        assert "winning_team" in chain_dict
        assert "frames" in chain_dict
        assert isinstance(chain_dict["frames"], list)

    def test_multiple_rounds_independent_chains(self):
        """Chains in different rounds are detected independently."""
        # Round 1: A kills B, C trades A
        # Round 2: D kills E, F trades D
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1200, 5000, 5200],
                "attacker_steamid": [1001, 2001, 1002, 2002],
                "user_steamid": [2001, 1001, 2002, 1002],
                "weapon": ["ak47", "m4a1", "awp", "ak47"],
                "headshot": [False, False, False, False],
                "total_rounds_played": [1, 1, 2, 2],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=2,
            player_stats={},
            player_names={
                1001: "CT_A",
                1002: "CT_D",
                2001: "T_B",
                2002: "T_E",
            },
            player_teams={1001: 3, 1002: 3, 2001: 2, 2002: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        # Should detect 2 separate chains
        assert len(result.trade_chains) == 2
        rounds = {c.round_num for c in result.trade_chains}
        assert rounds == {1, 2}
