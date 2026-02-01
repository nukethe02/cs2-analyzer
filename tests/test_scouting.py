"""Tests for the scouting module."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from opensight.scouting import (
    EconomyTendency,
    PlayerScoutProfile,
    PlayStyle,
    ScoutingEngine,
    TeamScoutReport,
    create_scouting_engine,
    generate_anti_strats,
)
from opensight.scouting.models import (
    MapTendency,
    PositionTendency,
    ScoutingSession,
    WeaponPreference,
)


class TestPlayStyle:
    """Tests for PlayStyle enum."""

    def test_playstyle_values(self):
        """PlayStyle enum has expected values."""
        assert PlayStyle.AGGRESSIVE.value == "aggressive"
        assert PlayStyle.PASSIVE.value == "passive"
        assert PlayStyle.MIXED.value == "mixed"


class TestEconomyTendency:
    """Tests for EconomyTendency enum."""

    def test_economy_tendency_values(self):
        """EconomyTendency enum has expected values."""
        assert EconomyTendency.AGGRESSIVE.value == "aggressive"
        assert EconomyTendency.CONSERVATIVE.value == "conservative"
        assert EconomyTendency.BALANCED.value == "balanced"


class TestPlayerScoutProfile:
    """Tests for PlayerScoutProfile dataclass."""

    @pytest.fixture
    def sample_profile(self):
        """Create a sample player profile."""
        return PlayerScoutProfile(
            steamid="76561198012345678",
            name="TestPlayer",
            play_style=PlayStyle.AGGRESSIVE,
            aggression_score=75.5,
            consistency_score=60.0,
            avg_kills_per_round=0.85,
            avg_deaths_per_round=0.65,
            avg_adr=85.5,
            avg_kast=72.3,
            headshot_rate=0.55,
            entry_attempt_rate=0.25,
            entry_success_rate=0.60,
            opening_duel_win_rate=0.52,
            awp_usage_rate=0.15,
            awp_kills_per_awp_round=1.2,
            avg_first_kill_time_seconds=25.5,
            avg_rotation_time_seconds=8.0,
            clutch_attempts=5,
            clutch_wins=2,
            demos_analyzed=3,
            rounds_analyzed=78,
        )

    def test_profile_creation(self, sample_profile):
        """PlayerScoutProfile can be created with required fields."""
        assert sample_profile.steamid == "76561198012345678"
        assert sample_profile.name == "TestPlayer"
        assert sample_profile.play_style == PlayStyle.AGGRESSIVE
        assert sample_profile.aggression_score == 75.5

    def test_kd_ratio_calculation(self, sample_profile):
        """K/D ratio is calculated correctly."""
        expected = 0.85 / 0.65
        assert abs(sample_profile.kd_ratio - expected) < 0.01

    def test_kd_ratio_zero_deaths(self):
        """K/D ratio handles zero deaths."""
        profile = PlayerScoutProfile(
            steamid="123",
            name="Test",
            play_style=PlayStyle.MIXED,
            aggression_score=50,
            consistency_score=50,
            avg_kills_per_round=1.0,
            avg_deaths_per_round=0.0,
            avg_adr=100,
            avg_kast=80,
            headshot_rate=0.5,
            entry_attempt_rate=0.2,
            entry_success_rate=0.5,
            opening_duel_win_rate=0.5,
            awp_usage_rate=0.1,
            awp_kills_per_awp_round=1.0,
            avg_first_kill_time_seconds=30,
            avg_rotation_time_seconds=10,
        )
        assert profile.kd_ratio == 1.0

    def test_clutch_win_rate(self, sample_profile):
        """Clutch win rate is calculated correctly."""
        assert sample_profile.clutch_win_rate == 2 / 5

    def test_clutch_win_rate_zero_attempts(self):
        """Clutch win rate handles zero attempts."""
        profile = PlayerScoutProfile(
            steamid="123",
            name="Test",
            play_style=PlayStyle.MIXED,
            aggression_score=50,
            consistency_score=50,
            avg_kills_per_round=0.8,
            avg_deaths_per_round=0.7,
            avg_adr=80,
            avg_kast=70,
            headshot_rate=0.5,
            entry_attempt_rate=0.2,
            entry_success_rate=0.5,
            opening_duel_win_rate=0.5,
            awp_usage_rate=0.1,
            awp_kills_per_awp_round=1.0,
            avg_first_kill_time_seconds=30,
            avg_rotation_time_seconds=10,
            clutch_attempts=0,
            clutch_wins=0,
        )
        assert profile.clutch_win_rate == 0.0

    def test_to_dict(self, sample_profile):
        """to_dict returns correct JSON-serializable dictionary."""
        d = sample_profile.to_dict()
        assert d["steamid"] == "76561198012345678"
        assert d["name"] == "TestPlayer"
        assert d["play_style"] == "aggressive"
        assert d["aggression_score"] == 75.5
        assert d["headshot_rate"] == 55.0  # Converted to percentage
        assert d["demos_analyzed"] == 3


class TestTeamScoutReport:
    """Tests for TeamScoutReport dataclass."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample team scout report."""
        player = PlayerScoutProfile(
            steamid="123",
            name="Player1",
            play_style=PlayStyle.AGGRESSIVE,
            aggression_score=70,
            consistency_score=60,
            avg_kills_per_round=0.9,
            avg_deaths_per_round=0.6,
            avg_adr=90,
            avg_kast=75,
            headshot_rate=0.5,
            entry_attempt_rate=0.3,
            entry_success_rate=0.6,
            opening_duel_win_rate=0.55,
            awp_usage_rate=0.1,
            awp_kills_per_awp_round=1.5,
            avg_first_kill_time_seconds=20,
            avg_rotation_time_seconds=7,
            demos_analyzed=2,
            rounds_analyzed=52,
        )
        return TeamScoutReport(
            team_name="Test Team",
            demos_analyzed=3,
            total_rounds=78,
            players=[player],
            economy_tendency=EconomyTendency.AGGRESSIVE,
            force_buy_rate=0.35,
            eco_round_rate=0.15,
            anti_strats=["Watch for aggressive pushes", "Smoke mid early"],
            confidence_level="medium",
            maps_analyzed=["de_dust2", "de_mirage"],
        )

    def test_report_creation(self, sample_report):
        """TeamScoutReport can be created with required fields."""
        assert sample_report.team_name == "Test Team"
        assert sample_report.demos_analyzed == 3
        assert len(sample_report.players) == 1
        assert sample_report.confidence_level == "medium"

    def test_to_dict(self, sample_report):
        """to_dict returns correct JSON-serializable dictionary."""
        d = sample_report.to_dict()
        assert d["team_name"] == "Test Team"
        assert d["demos_analyzed"] == 3
        assert d["total_rounds"] == 78
        assert len(d["players"]) == 1
        assert d["confidence_level"] == "medium"
        assert d["economy"]["tendency"] == "aggressive"
        assert d["economy"]["force_buy_rate"] == 35.0
        assert len(d["anti_strats"]) == 2


class TestMapTendency:
    """Tests for MapTendency dataclass."""

    def test_map_tendency_creation(self):
        """MapTendency can be created with required fields."""
        tendency = MapTendency(
            map_name="de_dust2",
            demos_analyzed=2,
            rounds_analyzed=52,
            t_side_default_setup="Standard 4-1 split",
            t_side_common_executes=["A long push", "B tunnels rush"],
            t_side_aggression=65.0,
            ct_side_default_setup="2-1-2",
            ct_side_rotation_speed="fast",
            ct_side_aggression=40.0,
            avg_execute_time_seconds=45.0,
            avg_first_contact_seconds=20.0,
        )
        assert tendency.map_name == "de_dust2"
        assert tendency.t_side_aggression == 65.0

    def test_map_tendency_to_dict(self):
        """MapTendency to_dict returns correct structure."""
        tendency = MapTendency(
            map_name="de_mirage",
            demos_analyzed=1,
            rounds_analyzed=26,
            t_side_default_setup="Default",
            t_side_common_executes=["A execute"],
            t_side_aggression=50.0,
            ct_side_default_setup="2-1-2",
            ct_side_rotation_speed="medium",
            ct_side_aggression=45.0,
            avg_execute_time_seconds=50.0,
            avg_first_contact_seconds=25.0,
        )
        d = tendency.to_dict()
        assert d["map_name"] == "de_mirage"
        assert d["t_side"]["aggression"] == 50.0
        assert d["ct_side"]["rotation_speed"] == "medium"


class TestWeaponPreference:
    """Tests for WeaponPreference dataclass."""

    def test_weapon_preference_creation(self):
        """WeaponPreference can be created."""
        pref = WeaponPreference(
            weapon_name="AK-47",
            usage_rate=0.45,
            kills_with=32,
            accuracy=0.22,
        )
        assert pref.weapon_name == "AK-47"
        assert pref.usage_rate == 0.45
        assert pref.kills_with == 32


class TestScoutingSession:
    """Tests for ScoutingSession dataclass."""

    def test_session_creation(self):
        """ScoutingSession can be created."""
        session = ScoutingSession(
            session_id="abc-123",
            created_at=1234567890.0,
        )
        assert session.session_id == "abc-123"
        assert session.demo_ids == []
        assert session.opponent_steamids == set()

    def test_session_to_dict(self):
        """ScoutingSession to_dict returns correct structure."""
        session = ScoutingSession(
            session_id="abc-123",
            created_at=1234567890.0,
            demo_ids=["demo1", "demo2"],
            team_name="Test Team",
        )
        session.opponent_steamids.add(12345)
        session.maps_included.add("de_dust2")

        d = session.to_dict()
        assert d["session_id"] == "abc-123"
        assert d["demo_count"] == 2
        assert d["opponent_count"] == 1
        assert "de_dust2" in d["maps_included"]


class TestScoutingEngine:
    """Tests for ScoutingEngine class."""

    def test_engine_creation(self):
        """ScoutingEngine can be created."""
        engine = ScoutingEngine()
        assert engine.demo_count == 0
        assert engine.maps_included == []

    def test_create_scouting_engine_factory(self):
        """create_scouting_engine factory function works."""
        engine = create_scouting_engine()
        assert isinstance(engine, ScoutingEngine)
        assert engine.demo_count == 0

    def test_set_opponent_team(self):
        """set_opponent_team stores steamids and team name."""
        engine = ScoutingEngine()
        engine.set_opponent_team([12345, 67890], "Enemy Team")
        assert engine._opponent_steamids == {12345, 67890}
        assert engine._team_name == "Enemy Team"

    def test_generate_report_no_demos(self):
        """generate_report raises when no demos loaded."""
        engine = ScoutingEngine()
        with pytest.raises(ValueError, match="No demos loaded"):
            engine.generate_report()

    def test_generate_report_no_opponents(self):
        """generate_report raises when no opponents set."""
        engine = ScoutingEngine()
        # Add a mock demo
        engine._demos.append(MagicMock())
        engine._analyses.append(MagicMock())

        with pytest.raises(ValueError, match="No opponent team specified"):
            engine.generate_report()


class TestAntiStrats:
    """Tests for anti-strat generation."""

    @pytest.fixture
    def aggressive_player(self):
        """Create an aggressive player profile."""
        return PlayerScoutProfile(
            steamid="123",
            name="AggroPlayer",
            play_style=PlayStyle.AGGRESSIVE,
            aggression_score=80,
            consistency_score=50,
            avg_kills_per_round=1.0,
            avg_deaths_per_round=0.8,
            avg_adr=95,
            avg_kast=70,
            headshot_rate=0.6,
            entry_attempt_rate=0.35,
            entry_success_rate=0.55,
            opening_duel_win_rate=0.58,
            awp_usage_rate=0.05,
            awp_kills_per_awp_round=0.5,
            avg_first_kill_time_seconds=15,
            avg_rotation_time_seconds=6,
            demos_analyzed=3,
            rounds_analyzed=78,
        )

    @pytest.fixture
    def passive_player(self):
        """Create a passive player profile."""
        return PlayerScoutProfile(
            steamid="456",
            name="PassivePlayer",
            play_style=PlayStyle.PASSIVE,
            aggression_score=25,
            consistency_score=75,
            avg_kills_per_round=0.65,
            avg_deaths_per_round=0.55,
            avg_adr=70,
            avg_kast=80,
            headshot_rate=0.45,
            entry_attempt_rate=0.08,
            entry_success_rate=0.4,
            opening_duel_win_rate=0.45,
            awp_usage_rate=0.02,
            awp_kills_per_awp_round=0.3,
            avg_first_kill_time_seconds=40,
            avg_rotation_time_seconds=12,
            demos_analyzed=3,
            rounds_analyzed=78,
        )

    @pytest.fixture
    def awp_player(self):
        """Create an AWP specialist profile."""
        return PlayerScoutProfile(
            steamid="789",
            name="AWPer",
            play_style=PlayStyle.MIXED,
            aggression_score=50,
            consistency_score=65,
            avg_kills_per_round=0.8,
            avg_deaths_per_round=0.6,
            avg_adr=75,
            avg_kast=72,
            headshot_rate=0.3,
            entry_attempt_rate=0.15,
            entry_success_rate=0.5,
            opening_duel_win_rate=0.5,
            awp_usage_rate=0.45,
            awp_kills_per_awp_round=1.3,
            avg_first_kill_time_seconds=25,
            avg_rotation_time_seconds=10,
            demos_analyzed=3,
            rounds_analyzed=78,
        )

    def test_aggressive_player_anti_strat(self, aggressive_player):
        """Anti-strats are generated for aggressive players."""
        strats = generate_anti_strats([aggressive_player], [], EconomyTendency.BALANCED)
        assert any("AGGRESSIVE" in s for s in strats)
        assert any("AggroPlayer" in s for s in strats)

    def test_passive_player_anti_strat(self, passive_player):
        """Anti-strats are generated for passive players."""
        strats = generate_anti_strats([passive_player], [], EconomyTendency.BALANCED)
        assert any("PASSIVE" in s for s in strats)
        assert any("PassivePlayer" in s for s in strats)

    def test_awp_player_anti_strat(self, awp_player):
        """Anti-strats are generated for AWP players."""
        strats = generate_anti_strats([awp_player], [], EconomyTendency.BALANCED)
        assert any("AWPer" in s for s in strats)
        assert any("AWP" in s for s in strats)

    def test_no_awper_anti_strat(self, aggressive_player, passive_player):
        """Anti-strat mentions no dedicated AWPer when none present."""
        strats = generate_anti_strats(
            [aggressive_player, passive_player], [], EconomyTendency.BALANCED
        )
        assert any("No dedicated AWPer" in s for s in strats)

    def test_economy_aggressive_anti_strat(self, aggressive_player):
        """Anti-strats mention force buys for aggressive economy."""
        strats = generate_anti_strats([aggressive_player], [], EconomyTendency.AGGRESSIVE)
        assert any("FORCE BUY" in s for s in strats)

    def test_economy_conservative_anti_strat(self, aggressive_player):
        """Anti-strats mention eco rounds for conservative economy."""
        strats = generate_anti_strats([aggressive_player], [], EconomyTendency.CONSERVATIVE)
        assert any("PROPER ECOS" in s for s in strats)

    def test_entry_fragger_detection(self, aggressive_player):
        """Entry fraggers are identified in anti-strats."""
        strats = generate_anti_strats([aggressive_player], [], EconomyTendency.BALANCED)
        # aggressive_player has high entry rate
        assert any("entry" in s.lower() for s in strats)

    def test_clutch_player_detection(self):
        """Clutch specialists are identified."""
        clutch_player = PlayerScoutProfile(
            steamid="111",
            name="ClutchMaster",
            play_style=PlayStyle.MIXED,
            aggression_score=55,
            consistency_score=70,
            avg_kills_per_round=0.75,
            avg_deaths_per_round=0.65,
            avg_adr=80,
            avg_kast=75,
            headshot_rate=0.48,
            entry_attempt_rate=0.12,
            entry_success_rate=0.45,
            opening_duel_win_rate=0.48,
            awp_usage_rate=0.08,
            awp_kills_per_awp_round=0.8,
            avg_first_kill_time_seconds=28,
            avg_rotation_time_seconds=9,
            clutch_attempts=10,
            clutch_wins=5,
            demos_analyzed=4,
            rounds_analyzed=104,
        )
        strats = generate_anti_strats([clutch_player], [], EconomyTendency.BALANCED)
        assert any("clutch" in s.lower() for s in strats)

    def test_map_tendencies_anti_strat(self, aggressive_player):
        """Map tendencies generate relevant anti-strats."""
        map_tendency = MapTendency(
            map_name="de_dust2",
            demos_analyzed=2,
            rounds_analyzed=52,
            t_side_default_setup="Standard",
            t_side_common_executes=["A long"],
            t_side_aggression=70.0,
            ct_side_default_setup="2-1-2",
            ct_side_rotation_speed="fast",
            ct_side_aggression=55.0,
            avg_execute_time_seconds=45.0,
            avg_first_contact_seconds=18.0,  # Early contact
        )
        strats = generate_anti_strats([aggressive_player], [map_tendency], EconomyTendency.BALANCED)
        assert any("Dust2" in s or "early contact" in s.lower() for s in strats)


class TestPositionTendency:
    """Tests for PositionTendency dataclass."""

    def test_position_tendency_creation(self):
        """PositionTendency can be created."""
        pos = PositionTendency(
            zone_name="A Site",
            frequency=0.35,
            success_rate=0.65,
            avg_time_held=45.0,
        )
        assert pos.zone_name == "A Site"
        assert pos.frequency == 0.35
        assert pos.success_rate == 0.65


class TestScoutingEngineIntegration:
    """Integration tests for ScoutingEngine with mock data."""

    @pytest.fixture
    def mock_demo_data(self):
        """Create mock DemoData."""
        demo = MagicMock()
        demo.map_name = "de_dust2"
        demo.tick_rate = 64
        demo.kills_df = pd.DataFrame(
            {
                "attacker_steamid": [12345, 12345, 67890],
                "attacker_side": ["T", "T", "CT"],
                "round": [1, 1, 2],
                "tick": [1000, 2000, 3000],
            }
        )
        demo.rounds_df = pd.DataFrame()
        return demo

    @pytest.fixture
    def mock_analysis(self):
        """Create mock MatchAnalysis."""
        # Create mock player stats
        player_stats = MagicMock()
        player_stats.name = "TestPlayer"
        player_stats.kills = 15
        player_stats.deaths = 10
        player_stats.rounds_played = 26
        player_stats.adr = 85.0
        player_stats.kast_percentage = 72.0
        player_stats.headshots = 8
        player_stats.entry_frags.attempts = 5
        player_stats.entry_frags.wins = 3
        player_stats.opening_duels.attempts = 4
        player_stats.opening_duels.wins = 2
        player_stats.t_stats.rounds = 13
        player_stats.ct_stats.rounds = 13
        player_stats.clutches.total = 3
        player_stats.clutches.won = 1
        player_stats.weapon_kills = {"AK-47": 10, "AWP": 3, "M4A4": 2}
        player_stats.true_ttd_values = [250, 300, 280]

        analysis = MagicMock()
        analysis.players = {12345: player_stats}
        analysis.total_rounds = 26
        analysis.team1_name = "Team A"
        analysis.team2_name = "Team B"

        return analysis

    def test_add_demo(self, mock_demo_data, mock_analysis):
        """add_demo correctly processes and stores demo."""
        engine = ScoutingEngine()
        result = engine.add_demo(mock_demo_data, mock_analysis)

        assert engine.demo_count == 1
        assert result["demo_index"] == 0
        assert result["map_name"] == "de_dust2"
        assert len(result["players"]) == 1

    def test_maps_included(self, mock_demo_data, mock_analysis):
        """maps_included returns correct list."""
        engine = ScoutingEngine()
        engine.add_demo(mock_demo_data, mock_analysis)

        # Add another demo with different map
        mock_demo_data2 = MagicMock()
        mock_demo_data2.map_name = "de_mirage"
        mock_demo_data2.kills_df = pd.DataFrame()
        mock_demo_data2.rounds_df = pd.DataFrame()

        engine.add_demo(mock_demo_data2, mock_analysis)

        maps = engine.maps_included
        assert "de_dust2" in maps
        assert "de_mirage" in maps
        assert len(maps) == 2
