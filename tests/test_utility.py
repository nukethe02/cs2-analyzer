"""Tests for the utility analysis module."""

from pathlib import Path

import pandas as pd
import pytest

from opensight.core.parser import DemoData
from opensight.domains.utility import (
    GRENADE_COSTS,
    GRENADE_WEAPONS,
    GrenadeType,
    UtilityAnalysisResult,
    UtilityAnalyzer,
    analyze_utility,
)


class TestGrenadeTypeMapping:
    """Tests for grenade type classification."""

    def test_he_grenade_recognized(self):
        """HE grenade is correctly classified."""
        assert GRENADE_WEAPONS.get("hegrenade") == GrenadeType.HE_GRENADE

    def test_molotov_recognized(self):
        """Molotov is correctly classified."""
        assert GRENADE_WEAPONS.get("molotov") == GrenadeType.MOLOTOV
        assert GRENADE_WEAPONS.get("inferno") == GrenadeType.MOLOTOV

    def test_flashbang_recognized(self):
        """Flashbang is correctly classified."""
        assert GRENADE_WEAPONS.get("flashbang") == GrenadeType.FLASHBANG

    def test_smoke_recognized(self):
        """Smoke grenade is correctly classified."""
        assert GRENADE_WEAPONS.get("smokegrenade") == GrenadeType.SMOKE

    def test_incendiary_recognized(self):
        """Incendiary grenade is correctly classified."""
        assert GRENADE_WEAPONS.get("incgrenade") == GrenadeType.INCENDIARY


class TestGrenadeCosts:
    """Tests for grenade cost values."""

    def test_flashbang_cost(self):
        """Flashbang costs $200."""
        assert GRENADE_COSTS[GrenadeType.FLASHBANG] == 200

    def test_he_cost(self):
        """HE grenade costs $300."""
        assert GRENADE_COSTS[GrenadeType.HE_GRENADE] == 300

    def test_smoke_cost(self):
        """Smoke grenade costs $300."""
        assert GRENADE_COSTS[GrenadeType.SMOKE] == 300

    def test_molotov_cost(self):
        """Molotov costs $400."""
        assert GRENADE_COSTS[GrenadeType.MOLOTOV] == 400

    def test_incendiary_cost(self):
        """Incendiary costs $600."""
        assert GRENADE_COSTS[GrenadeType.INCENDIARY] == 600


class TestUtilityAnalyzer:
    """Tests for UtilityAnalyzer class."""

    @pytest.fixture
    def demo_with_utility(self):
        """Create demo data with utility damage."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000, 1100, 2000, 2100, 2200],
                "attacker_steamid": [12345, 12345, 67890, 12345, 12345],
                "user_steamid": [67890, 67890, 12345, 67890, 11111],
                "dmg_health": [50, 30, 25, 40, 10],
                "weapon": ["hegrenade", "hegrenade", "molotov", "hegrenade", "hegrenade"],
                "total_rounds_played": [1, 1, 1, 2, 2],
            }
        )

        return DemoData(
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
            kills_df=pd.DataFrame(),
            damages_df=damages_df,
        )

    def test_extracts_grenade_events(self, demo_with_utility):
        """Grenade damage events are extracted."""
        analyzer = UtilityAnalyzer(demo_with_utility)
        result = analyzer.analyze()

        assert len(result.grenade_damage_events) == 5

    def test_classifies_grenade_types(self, demo_with_utility):
        """Grenade types are correctly classified."""
        analyzer = UtilityAnalyzer(demo_with_utility)
        result = analyzer.analyze()

        he_events = [
            e for e in result.grenade_damage_events if e.grenade_type == GrenadeType.HE_GRENADE
        ]
        molotov_events = [
            e for e in result.grenade_damage_events if e.grenade_type == GrenadeType.MOLOTOV
        ]

        assert len(he_events) == 4
        assert len(molotov_events) == 1

    def test_ignores_non_grenade_damage(self):
        """Non-grenade damage is ignored."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000, 2000],
                "attacker_steamid": [12345, 12345],
                "user_steamid": [67890, 67890],
                "dmg_health": [27, 100],
                "weapon": ["ak47", "awp"],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={12345: "Player1", 67890: "Player2"},
            player_teams={12345: 3, 67890: 2},
            kills=[],
            damages=[],
            kills_df=pd.DataFrame(),
            damages_df=damages_df,
        )

        analyzer = UtilityAnalyzer(demo)
        result = analyzer.analyze()

        assert len(result.grenade_damage_events) == 0


class TestTeamDamageDetection:
    """Tests for team damage detection."""

    def test_detects_team_damage(self):
        """Team damage is flagged."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000],
                "attacker_steamid": [12345],
                "user_steamid": [11111],  # Same team as attacker
                "dmg_health": [50],
                "weapon": ["hegrenade"],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={12345: "Player1", 11111: "Teammate"},
            player_teams={12345: 3, 11111: 3},  # Both CT
            kills=[],
            damages=[],
            kills_df=pd.DataFrame(),
            damages_df=damages_df,
        )

        analyzer = UtilityAnalyzer(demo)
        result = analyzer.analyze()

        assert len(result.grenade_damage_events) == 1
        assert result.grenade_damage_events[0].is_team_damage is True

    def test_enemy_damage_not_flagged_as_team(self):
        """Enemy damage is not flagged as team damage."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000],
                "attacker_steamid": [12345],
                "user_steamid": [67890],
                "dmg_health": [50],
                "weapon": ["hegrenade"],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={12345: "Player1", 67890: "Enemy"},
            player_teams={12345: 3, 67890: 2},  # Different teams
            kills=[],
            damages=[],
            kills_df=pd.DataFrame(),
            damages_df=damages_df,
        )

        analyzer = UtilityAnalyzer(demo)
        result = analyzer.analyze()

        assert result.grenade_damage_events[0].is_team_damage is False


class TestPlayerUtilityStats:
    """Tests for player utility statistics."""

    @pytest.fixture
    def demo_with_player_utility(self):
        """Create demo data with utility for specific player."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000, 1100, 2000],
                "attacker_steamid": [12345, 12345, 12345],
                "user_steamid": [67890, 67890, 67890],
                "dmg_health": [50, 30, 40],
                "weapon": ["hegrenade", "hegrenade", "hegrenade"],
                "total_rounds_played": [1, 1, 2],
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
            kills_df=pd.DataFrame(),
            damages_df=damages_df,
        )

    def test_calculates_he_damage(self, demo_with_player_utility):
        """HE damage total is calculated."""
        analyzer = UtilityAnalyzer(demo_with_player_utility)
        result = analyzer.analyze()

        stats = result.player_stats[12345]
        assert stats.he_damage_total == 120  # 50 + 30 + 40

    def test_calculates_damage_per_round(self, demo_with_player_utility):
        """Utility damage per round is calculated."""
        analyzer = UtilityAnalyzer(demo_with_player_utility)
        result = analyzer.analyze()

        stats = result.player_stats[12345]
        assert stats.utility_damage_per_round == 60.0  # 120 / 2 rounds


class TestTeamUtilityStats:
    """Tests for team-level utility statistics."""

    def test_team_damage_calculated(self):
        """Team utility damage totals are calculated."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000, 2000],
                "attacker_steamid": [12345, 67890],
                "user_steamid": [67890, 12345],
                "dmg_health": [50, 30],
                "weapon": ["hegrenade", "molotov"],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={12345: "CT", 67890: "T"},
            player_teams={12345: 3, 67890: 2},
            kills=[],
            damages=[],
            kills_df=pd.DataFrame(),
            damages_df=damages_df,
        )

        analyzer = UtilityAnalyzer(demo)
        result = analyzer.analyze()

        assert result.team_utility_damage[3] == 50  # CT
        assert result.team_utility_damage[2] == 30  # T


class TestAnalyzeUtilityFunction:
    """Tests for analyze_utility convenience function."""

    def test_analyze_utility_creates_analyzer(self):
        """analyze_utility creates analyzer and runs analysis."""
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

        result = analyze_utility(demo)
        assert isinstance(result, UtilityAnalysisResult)

    def test_handles_empty_data(self):
        """Empty damage data is handled gracefully."""
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

        result = analyze_utility(demo)
        assert result.grenade_damage_events == []
        assert result.player_stats == {}
