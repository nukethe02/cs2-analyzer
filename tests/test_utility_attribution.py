"""Tests for utility damage attribution.

Verifies that HE, molotov, and flash damage/assists are credited to the
correct player (the THROWER), not the victim or a random player.

Root cause being tested: VIC_SIDE_COLS was missing "user_team" so the
enemy/team damage split never worked when the only victim-side column
came from _enrich_with_team_info (which adds "user_team").
"""

from pathlib import Path

import pandas as pd
import pytest

from opensight.analysis.analytics import DemoAnalyzer
from opensight.analysis.compute_utility import (
    calculate_mistakes,
    calculate_utility_stats,
    compute_utility_metrics,
)
from opensight.core.parser import DemoData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_demo_data(
    damages_df: pd.DataFrame,
    kills_df: pd.DataFrame | None = None,
    player_names: dict[int, str] | None = None,
    player_teams: dict[int, str] | None = None,
    blinds: list | None = None,
    grenades: list | None = None,
) -> DemoData:
    """Build minimal DemoData for utility-attribution tests."""
    if kills_df is None:
        kills_df = pd.DataFrame()
    if player_names is None:
        player_names = {}
    if player_teams is None:
        player_teams = {}

    return DemoData(
        file_path=Path("/tmp/test.dem"),
        map_name="de_dust2",
        duration_seconds=1800.0,
        tick_rate=64,
        num_rounds=2,
        player_stats={
            sid: {
                "name": name,
                "team": player_teams.get(sid, "Unknown"),
                "kills": 0,
                "deaths": 0,
                "assists": 0,
                "headshots": 0,
                "hs_percent": 0.0,
                "total_damage": 0,
                "adr": 0.0,
                "weapon_kills": {},
            }
            for sid, name in player_names.items()
        },
        player_names=player_names,
        player_teams=player_teams,
        kills=[] if kills_df.empty else [],
        damages=[],
        kills_df=kills_df,
        damages_df=damages_df,
        blinds=blinds or [],
        grenades=grenades or [],
    )


def _run_utility_stats(demo: DemoData) -> DemoAnalyzer:
    """Create a DemoAnalyzer, init it, and run calculate_utility_stats."""
    analyzer = DemoAnalyzer(demo)
    analyzer._init_column_cache()
    analyzer._init_player_stats()
    analyzer._safe_calculate("basic_stats", analyzer._calculate_basic_stats)
    calculate_utility_stats(analyzer)
    return analyzer


# ---------------------------------------------------------------------------
# Test: HE grenade damage goes to the THROWER
# ---------------------------------------------------------------------------


class TestHEDamageAttribution:
    """HE grenade damage must be credited to the thrower, not the victim."""

    @pytest.fixture
    def he_demo(self):
        """Player A (CT) throws HE hitting Player B (T) and Player C (T)."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000, 1001],
                "attacker_steamid": [11111, 11111],
                "user_steamid": [22222, 33333],
                "attacker_name": ["PlayerA", "PlayerA"],
                "user_name": ["PlayerB", "PlayerC"],
                "dmg_health": [50, 30],
                "weapon": ["hegrenade", "hegrenade"],
                # Enrichment columns (as produced by _enrich_with_team_info)
                "attacker_team": [3, 3],  # CT
                "user_team": [2, 2],  # T
            }
        )
        return _make_demo_data(
            damages_df,
            player_names={11111: "PlayerA", 22222: "PlayerB", 33333: "PlayerC"},
            player_teams={11111: "CT", 22222: "T", 33333: "T"},
        )

    def test_thrower_gets_all_he_damage(self, he_demo):
        """Player A should get 80 HE damage (50 + 30)."""
        analyzer = _run_utility_stats(he_demo)
        player_a = analyzer._players[11111]
        assert player_a.utility.he_damage == 80

    def test_victims_get_no_he_damage_credit(self, he_demo):
        """Players B and C should NOT get HE damage credit."""
        analyzer = _run_utility_stats(he_demo)
        player_b = analyzer._players[22222]
        player_c = analyzer._players[33333]
        assert player_b.utility.he_damage == 0
        assert player_c.utility.he_damage == 0

    def test_team_he_damage_is_zero_for_enemy_hits(self, he_demo):
        """When all HE hits enemies, he_team_damage should be 0."""
        analyzer = _run_utility_stats(he_demo)
        player_a = analyzer._players[11111]
        assert player_a.utility.he_team_damage == 0


class TestHETeamDamageSplit:
    """HE damage to teammates must go to he_team_damage, not he_damage."""

    def test_he_team_damage_separated(self):
        """HE hitting a teammate should go to he_team_damage, not he_damage."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000, 1100],
                "attacker_steamid": [11111, 11111],
                "user_steamid": [22222, 33333],
                "attacker_name": ["PlayerA", "PlayerA"],
                "user_name": ["PlayerB", "Teammate"],
                "dmg_health": [50, 20],
                "weapon": ["hegrenade", "hegrenade"],
                # Player A (CT) hits Player B (T=enemy) and Teammate (CT=team)
                "attacker_team": [3, 3],
                "user_team": [2, 3],  # B is T, Teammate is CT
            }
        )
        demo = _make_demo_data(
            damages_df,
            player_names={11111: "PlayerA", 22222: "PlayerB", 33333: "Teammate"},
            player_teams={11111: "CT", 22222: "T", 33333: "CT"},
        )
        analyzer = _run_utility_stats(demo)
        player_a = analyzer._players[11111]
        assert player_a.utility.he_damage == 50  # enemy only
        assert player_a.utility.he_team_damage == 20  # teammate only


# ---------------------------------------------------------------------------
# Test: Molotov damage goes to the THROWER across all ticks
# ---------------------------------------------------------------------------


class TestMolotovDamageAttribution:
    """Molotov/inferno damage across multiple ticks must credit the thrower."""

    @pytest.fixture
    def molly_demo(self):
        """Player A throws molotov, Player B takes damage over 3 ticks."""
        damages_df = pd.DataFrame(
            {
                "tick": [2000, 2064, 2128],
                "attacker_steamid": [11111, 11111, 11111],
                "user_steamid": [22222, 22222, 22222],
                "attacker_name": ["PlayerA", "PlayerA", "PlayerA"],
                "user_name": ["PlayerB", "PlayerB", "PlayerB"],
                "dmg_health": [8, 8, 8],
                "weapon": ["inferno", "inferno", "inferno"],
                "attacker_team": [3, 3, 3],
                "user_team": [2, 2, 2],
            }
        )
        return _make_demo_data(
            damages_df,
            player_names={11111: "PlayerA", 22222: "PlayerB"},
            player_teams={11111: "CT", 22222: "T"},
        )

    def test_thrower_gets_all_molly_damage(self, molly_demo):
        """Player A should get 24 molotov damage (8 + 8 + 8)."""
        analyzer = _run_utility_stats(molly_demo)
        player_a = analyzer._players[11111]
        assert player_a.utility.molotov_damage == 24

    def test_victim_gets_no_molly_credit(self, molly_demo):
        """Player B (victim) should NOT get molotov damage credit."""
        analyzer = _run_utility_stats(molly_demo)
        player_b = analyzer._players[22222]
        assert player_b.utility.molotov_damage == 0

    def test_molly_team_damage_zero_for_enemy(self, molly_demo):
        """No team damage when molotov only hits enemies."""
        analyzer = _run_utility_stats(molly_demo)
        player_a = analyzer._players[11111]
        assert player_a.utility.molotov_team_damage == 0


class TestMolotovTeamDamageSplit:
    """Molotov damage to teammates must be tracked separately."""

    def test_molly_team_damage_separated(self):
        """Molotov hitting teammate goes to molotov_team_damage."""
        damages_df = pd.DataFrame(
            {
                "tick": [2000, 2064],
                "attacker_steamid": [11111, 11111],
                "user_steamid": [22222, 33333],
                "attacker_name": ["PlayerA", "PlayerA"],
                "user_name": ["Enemy", "Teammate"],
                "dmg_health": [8, 8],
                "weapon": ["inferno", "inferno"],
                "attacker_team": [3, 3],
                "user_team": [2, 3],  # Enemy=T, Teammate=CT
            }
        )
        demo = _make_demo_data(
            damages_df,
            player_names={11111: "PlayerA", 22222: "Enemy", 33333: "Teammate"},
            player_teams={11111: "CT", 22222: "T", 33333: "CT"},
        )
        analyzer = _run_utility_stats(demo)
        player_a = analyzer._players[11111]
        assert player_a.utility.molotov_damage == 8  # enemy
        assert player_a.utility.molotov_team_damage == 8  # teammate


# ---------------------------------------------------------------------------
# Test: Gun damage should NOT be counted as utility damage
# ---------------------------------------------------------------------------


class TestGunDamageExcluded:
    """Regular gun damage must not appear in utility damage fields."""

    def test_ak47_not_counted_as_utility(self):
        """AK-47 damage must not be counted as HE or molotov damage."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000, 1100, 1200],
                "attacker_steamid": [11111, 11111, 11111],
                "user_steamid": [22222, 22222, 22222],
                "attacker_name": ["PlayerA", "PlayerA", "PlayerA"],
                "user_name": ["PlayerB", "PlayerB", "PlayerB"],
                "dmg_health": [27, 27, 27],
                "weapon": ["ak47", "ak47", "ak47"],
                "attacker_team": [3, 3, 3],
                "user_team": [2, 2, 2],
            }
        )
        demo = _make_demo_data(
            damages_df,
            player_names={11111: "PlayerA", 22222: "PlayerB"},
            player_teams={11111: "CT", 22222: "T"},
        )
        analyzer = _run_utility_stats(demo)
        player_a = analyzer._players[11111]
        assert player_a.utility.he_damage == 0
        assert player_a.utility.molotov_damage == 0
        assert player_a.utility.total_utility_damage == 0


# ---------------------------------------------------------------------------
# Test: VIC_SIDE_COLS includes user_team (the root cause fix)
# ---------------------------------------------------------------------------


class TestVicSideColsIncludesUserTeam:
    """VIC_SIDE_COLS must include 'user_team' for _enrich_with_team_info compatibility."""

    def test_user_team_in_vic_side_cols(self):
        """'user_team' must be in VIC_SIDE_COLS."""
        assert "user_team" in DemoAnalyzer.VIC_SIDE_COLS

    def test_find_col_finds_user_team(self):
        """_find_col should find 'user_team' when it is the only victim side column."""
        df = pd.DataFrame(
            {
                "attacker_steamid": [11111],
                "user_steamid": [22222],
                "user_team": [2],
            }
        )
        demo = _make_demo_data(
            df,
            player_names={11111: "A", 22222: "B"},
        )
        analyzer = DemoAnalyzer(demo)
        result = analyzer._find_col(df, analyzer.VIC_SIDE_COLS)
        assert result == "user_team"


# ---------------------------------------------------------------------------
# Test: calculate_mistakes also uses VIC_SIDE_COLS correctly
# ---------------------------------------------------------------------------


class TestMistakesTeamDamage:
    """calculate_mistakes must detect team damage when user_team is the column."""

    def test_team_damage_detected_with_user_team_column(self):
        """Team damage should be detected using the user_team column."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000],
                "attacker_steamid": [11111],
                "user_steamid": [33333],
                "attacker_name": ["PlayerA"],
                "user_name": ["Teammate"],
                "dmg_health": [25],
                "weapon": ["hegrenade"],
                "attacker_team": [3],
                "user_team": [3],  # Same team = team damage
            }
        )
        kills_df = pd.DataFrame(
            {
                "tick": [2000],
                "attacker_steamid": [11111],
                "user_steamid": [22222],
                "attacker_name": ["PlayerA"],
                "user_name": ["Enemy"],
                "weapon": ["ak47"],
                "headshot": [True],
                "attacker_team": [3],
                "user_team": [2],
            }
        )
        demo = _make_demo_data(
            damages_df,
            kills_df=kills_df,
            player_names={11111: "PlayerA", 22222: "Enemy", 33333: "Teammate"},
            player_teams={11111: "CT", 22222: "T", 33333: "CT"},
        )
        analyzer = DemoAnalyzer(demo)
        analyzer._init_column_cache()
        analyzer._init_player_stats()
        analyzer._safe_calculate("basic_stats", analyzer._calculate_basic_stats)
        calculate_utility_stats(analyzer)
        calculate_mistakes(analyzer)
        player_a = analyzer._players[11111]
        assert player_a.mistakes.team_damage == 25


# ---------------------------------------------------------------------------
# Test: compute_utility_metrics standalone function
# ---------------------------------------------------------------------------


class TestComputeUtilityMetricsAttribution:
    """compute_utility_metrics (standalone) must also split enemy/team damage."""

    def test_standalone_he_damage_attributed_to_thrower(self):
        """Standalone function: HE damage credited to thrower."""
        damages_df = pd.DataFrame(
            {
                "tick": [1000],
                "attacker_steamid": [11111],
                "user_steamid": [22222],
                "dmg_health": [50],
                "weapon": ["hegrenade"],
                "attacker_team": [3],
                "user_team": [2],
            }
        )
        demo = _make_demo_data(
            damages_df,
            player_names={11111: "PlayerA", 22222: "PlayerB"},
            player_teams={11111: "CT", 22222: "T"},
        )
        result = compute_utility_metrics(demo)
        assert result["11111"].he_damage == 50
        assert result["22222"].he_damage == 0

    def test_standalone_molly_enemy_vs_team_split(self):
        """Standalone function: molotov enemy vs team damage split works."""
        damages_df = pd.DataFrame(
            {
                "tick": [2000, 2100],
                "attacker_steamid": [11111, 11111],
                "user_steamid": [22222, 33333],
                "dmg_health": [10, 10],
                "weapon": ["inferno", "inferno"],
                "attacker_team": [3, 3],
                "user_team": [2, 3],  # first=enemy, second=teammate
            }
        )
        demo = _make_demo_data(
            damages_df,
            player_names={11111: "PlayerA", 22222: "Enemy", 33333: "Teammate"},
            player_teams={11111: "CT", 22222: "T", 33333: "CT"},
        )
        result = compute_utility_metrics(demo)
        # Enemy molly damage only (not team)
        assert result["11111"].molotov_damage == 10


# ---------------------------------------------------------------------------
# Test: Flash assists go to the flash thrower
# ---------------------------------------------------------------------------


class TestFlashAssistAttribution:
    """Flash assists must be credited to the player who threw the flash."""

    def test_flash_assist_goes_to_thrower(self):
        """Flash assist counted for the player whose flash caused the blind."""
        kills_df = pd.DataFrame(
            {
                "tick": [3000],
                "attacker_steamid": [22222],
                "user_steamid": [33333],
                "attacker_name": ["Teammate"],
                "user_name": ["Enemy"],
                "weapon": ["ak47"],
                "headshot": [False],
                "assister_steamid": [11111],
                "flash_assist": [True],
                "attacker_team": [3],
                "user_team": [2],
            }
        )
        demo = _make_demo_data(
            pd.DataFrame(),
            kills_df=kills_df,
            player_names={11111: "FlashThrower", 22222: "Teammate", 33333: "Enemy"},
            player_teams={11111: "CT", 22222: "CT", 33333: "T"},
        )
        analyzer = _run_utility_stats(demo)
        flash_thrower = analyzer._players[11111]
        teammate = analyzer._players[22222]
        assert flash_thrower.utility.flash_assists == 1
        assert teammate.utility.flash_assists == 0
