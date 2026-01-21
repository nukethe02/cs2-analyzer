"""Tests for the economy analysis module."""

from pathlib import Path
import pandas as pd
import pytest

from opensight.domains.economy import (
    BuyType,
    classify_buy_type,
    classify_team_buy,
    estimate_weapon_cost,
    EconomyAnalyzer,
    PlayerRoundEconomy,
    TeamRoundEconomy,
    EconomyStats,
    PlayerEconomyProfile,
    analyze_economy,
    WEAPON_COSTS,
)
from opensight.core.parser import DemoData


class TestBuyTypeClassification:
    """Tests for buy type classification functions."""

    def test_pistol_round_classification(self):
        """Pistol rounds are correctly classified."""
        assert classify_buy_type(0, is_pistol_round=True) == BuyType.PISTOL
        assert classify_buy_type(800, is_pistol_round=True) == BuyType.PISTOL
        assert classify_buy_type(2000, is_pistol_round=True) == BuyType.PISTOL

    def test_eco_classification(self):
        """Eco rounds are correctly classified."""
        assert classify_buy_type(0) == BuyType.ECO
        assert classify_buy_type(200) == BuyType.ECO
        assert classify_buy_type(300) == BuyType.ECO

    def test_force_classification(self):
        """Force buy rounds are correctly classified."""
        assert classify_buy_type(400) == BuyType.FORCE
        assert classify_buy_type(600) == BuyType.FORCE
        assert classify_buy_type(700) == BuyType.FORCE

    def test_half_buy_classification(self):
        """Half buy rounds are correctly classified."""
        assert classify_buy_type(1000) == BuyType.HALF_BUY
        assert classify_buy_type(2000) == BuyType.HALF_BUY
        assert classify_buy_type(2500) == BuyType.HALF_BUY

    def test_full_buy_classification(self):
        """Full buy rounds are correctly classified."""
        assert classify_buy_type(3000) == BuyType.FULL_BUY
        assert classify_buy_type(5000) == BuyType.FULL_BUY
        assert classify_buy_type(10000) == BuyType.FULL_BUY


class TestTeamBuyClassification:
    """Tests for team buy classification."""

    def test_team_pistol_round(self):
        """Team pistol round is correctly classified."""
        assert classify_team_buy(2000, is_pistol_round=True) == BuyType.PISTOL

    def test_team_eco(self):
        """Team eco is correctly classified."""
        assert classify_team_buy(1000) == BuyType.ECO
        assert classify_team_buy(1500) == BuyType.ECO

    def test_team_force(self):
        """Team force buy is correctly classified."""
        assert classify_team_buy(2000) == BuyType.FORCE
        assert classify_team_buy(3000) == BuyType.FORCE
        assert classify_team_buy(3500) == BuyType.FORCE

    def test_team_half_buy(self):
        """Team half buy is correctly classified."""
        assert classify_team_buy(4000) == BuyType.HALF_BUY
        assert classify_team_buy(4500) == BuyType.HALF_BUY

    def test_team_full_buy(self):
        """Team full buy is correctly classified."""
        assert classify_team_buy(5000) == BuyType.FULL_BUY
        assert classify_team_buy(20000) == BuyType.FULL_BUY


class TestWeaponCostEstimation:
    """Tests for weapon cost estimation."""

    def test_known_weapons(self):
        """Known weapons return correct costs."""
        assert estimate_weapon_cost("ak47") == 2700
        assert estimate_weapon_cost("m4a1_silencer") == 2900
        assert estimate_weapon_cost("awp") == 4750
        assert estimate_weapon_cost("deagle") == 700

    def test_pistols_are_cheap(self):
        """Pistols have low costs."""
        assert estimate_weapon_cost("glock") == 0
        assert estimate_weapon_cost("usp_silencer") == 0
        assert estimate_weapon_cost("p250") == 300

    def test_rifles_are_expensive(self):
        """Rifles have high costs."""
        assert estimate_weapon_cost("ak47") >= 2500
        assert estimate_weapon_cost("m4a1") >= 2500
        assert estimate_weapon_cost("awp") >= 4500

    def test_unknown_weapon_returns_zero(self):
        """Unknown weapons return 0."""
        assert estimate_weapon_cost("unknown_weapon") == 0
        assert estimate_weapon_cost("") == 0

    def test_case_insensitive(self):
        """Weapon names are case insensitive."""
        assert estimate_weapon_cost("AK47") == 2700
        assert estimate_weapon_cost("AWP") == 4750


class TestDataClasses:
    """Tests for economy data classes."""

    def test_player_round_economy_creation(self):
        """PlayerRoundEconomy can be created."""
        economy = PlayerRoundEconomy(
            steam_id=12345,
            round_num=5,
            equipment_value=4500,
            start_money=5000,
            end_money=500,
            spent=4500,
            weapon="ak47",
            has_armor=True,
            has_helmet=True,
            has_defuser=False,
            grenade_count=3,
            buy_type=BuyType.FULL_BUY,
        )
        assert economy.steam_id == 12345
        assert economy.equipment_value == 4500
        assert economy.buy_type == BuyType.FULL_BUY

    def test_team_round_economy_creation(self):
        """TeamRoundEconomy can be created."""
        economy = TeamRoundEconomy(
            round_num=5,
            team=2,
            total_equipment=20000,
            avg_equipment=4000,
            total_money=25000,
            total_spent=20000,
            buy_type=BuyType.FULL_BUY,
        )
        assert economy.round_num == 5
        assert economy.team == 2
        assert economy.buy_type == BuyType.FULL_BUY


class TestEconomyAnalyzer:
    """Tests for EconomyAnalyzer class."""

    @pytest.fixture
    def sample_demo_data(self):
        """Create sample demo data for testing."""
        kills_df = pd.DataFrame({
            "tick": [1000, 2000, 3000, 4000],
            "attacker_steamid": [12345, 67890, 12345, 12345],
            "user_steamid": [67890, 12345, 67890, 67890],
            "weapon": ["ak47", "awp", "m4a1_silencer", "deagle"],
            "headshot": [True, False, True, False],
            "total_rounds_played": [1, 1, 5, 10],
        })

        damages_df = pd.DataFrame({
            "tick": [900, 1000, 1900, 2000],
            "attacker_steamid": [12345, 12345, 67890, 67890],
            "user_steamid": [67890, 67890, 12345, 12345],
            "dmg_health": [27, 73, 100, 8],
            "weapon": ["ak47", "ak47", "awp", "awp"],
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
                    "kills": 3,
                    "deaths": 1,
                    "assists": 0,
                    "adr": 80.0,
                    "hs_percent": 66.7,
                    "headshots": 2,
                    "total_damage": 200,
                    "weapon_kills": {"ak47": 1, "m4a1_silencer": 1, "deagle": 1},
                },
                67890: {
                    "name": "Player2",
                    "team": "T",
                    "kills": 1,
                    "deaths": 3,
                    "assists": 0,
                    "adr": 72.0,
                    "hs_percent": 0.0,
                    "headshots": 0,
                    "total_damage": 108,
                    "weapon_kills": {"awp": 1},
                },
            },
            player_names={12345: "Player1", 67890: "Player2"},
            player_teams={12345: 3, 67890: 2},  # CT=3, T=2
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=damages_df,
        )

    def test_analyzer_initialization(self, sample_demo_data):
        """Analyzer initializes correctly."""
        analyzer = EconomyAnalyzer(sample_demo_data)
        assert analyzer.data is sample_demo_data

    def test_analyzer_analyzes_demo(self, sample_demo_data):
        """Analyzer produces stats from demo data."""
        analyzer = EconomyAnalyzer(sample_demo_data)
        stats = analyzer.analyze()

        assert isinstance(stats, EconomyStats)
        assert stats.rounds_analyzed == 15

    def test_analyzer_estimates_equipment(self, sample_demo_data):
        """Analyzer estimates equipment from weapons used."""
        analyzer = EconomyAnalyzer(sample_demo_data)
        stats = analyzer.analyze()

        # Should have economy data for players
        assert len(stats.player_economies) > 0

    def test_analyzer_handles_empty_data(self):
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

        analyzer = EconomyAnalyzer(empty_data)
        stats = analyzer.analyze()

        assert stats.rounds_analyzed == 0

    def test_player_profile(self, sample_demo_data):
        """Analyzer can produce player economy profiles."""
        analyzer = EconomyAnalyzer(sample_demo_data)
        analyzer.analyze()

        profile = analyzer.get_player_profile(12345)

        assert profile is not None
        assert profile.steam_id == 12345
        assert profile.name == "Player1"

    def test_player_profile_unknown_player(self, sample_demo_data):
        """Analyzer returns None for unknown player."""
        analyzer = EconomyAnalyzer(sample_demo_data)
        analyzer.analyze()

        profile = analyzer.get_player_profile(99999)
        assert profile is None


class TestAnalyzeEconomyFunction:
    """Tests for analyze_economy convenience function."""

    def test_analyze_economy_creates_analyzer(self):
        """analyze_economy creates analyzer and runs analysis."""
        demo_data = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=15,
            player_stats={},
            player_names={},
            player_teams={},
            kills=[],
            damages=[],
            kills_df=pd.DataFrame(),
            damages_df=pd.DataFrame(),
        )

        stats = analyze_economy(demo_data)
        assert isinstance(stats, EconomyStats)
