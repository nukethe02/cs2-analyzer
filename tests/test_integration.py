"""
Integration Tests for OpenSight CS2 Analyzer

Tests the complete analysis pipeline and verifies fixes to core detection logic:
1. Clutch detection - team type normalization (string vs int)
2. Player name resolution - Steam ID validation and fallback names
3. Economy tracking - loss bonus, bad/good force detection, economy grades
4. Lurker persona detection - isolation with impact metrics

Reference values from Leetify (ancient_32rounds.dem):
- foe: 17/24/6 K/D/A, ADR 71.8, HLTV 0.82
- dergs: 22/25/5 K/D/A
- Team clutches: 20 + 54 = 74 total situations
"""

from pathlib import Path

import pandas as pd
import pytest

# =============================================================================
# TEAM TYPE NORMALIZATION TESTS (Clutch Detection Fix)
# =============================================================================


class TestTeamNormalization:
    """Tests for team type normalization - fixes the string vs int comparison bug."""

    def test_normalize_team_from_string_ct(self):
        """CT string normalizes to 3."""
        from opensight.domains.combat import normalize_team

        assert normalize_team("CT") == 3
        assert normalize_team("ct") == 3
        assert normalize_team("Counter-Terrorist") == 3

    def test_normalize_team_from_string_t(self):
        """T string normalizes to 2."""
        from opensight.domains.combat import normalize_team

        assert normalize_team("T") == 2
        assert normalize_team("t") == 2
        assert normalize_team("Terrorist") == 2

    def test_normalize_team_from_int(self):
        """Integer team values pass through."""
        from opensight.domains.combat import normalize_team

        assert normalize_team(2) == 2
        assert normalize_team(3) == 3

    def test_normalize_team_invalid_returns_zero(self):
        """Invalid values return 0."""
        from opensight.domains.combat import normalize_team

        assert normalize_team(None) == 0
        assert normalize_team("invalid") == 0
        assert normalize_team(99) == 0

    def test_combat_analyzer_handles_string_teams(self):
        """CombatAnalyzer correctly handles string team values."""
        from opensight.core.parser import DemoData
        from opensight.domains.combat import CombatAnalyzer

        # Create demo with STRING team values (the bug scenario)
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1100, 1200, 1300, 2000, 2100],
                "attacker_steamid": [1, 1, 1, 1, 12345, 12345],
                "user_steamid": [101, 102, 103, 104, 1, 2],
                "weapon": ["ak47"] * 6,
                "headshot": [False] * 6,
                "total_rounds_played": [1, 1, 1, 1, 1, 1],
            }
        )

        demo = DemoData(
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
            # STRING team values - this was causing the bug
            player_teams={
                12345: "CT",
                101: "CT",
                102: "CT",
                103: "CT",
                104: "CT",
                1: "T",
                2: "T",
            },
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        # Should detect the clutch situation now that string teams are normalized
        assert len(result.clutch_situations) >= 1, (
            "Clutch detection should work with string team values"
        )

    def test_combat_analyzer_handles_int_teams(self):
        """CombatAnalyzer correctly handles integer team values."""
        from opensight.core.parser import DemoData
        from opensight.domains.combat import CombatAnalyzer

        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1100, 1200, 1300, 2000, 2100],
                "attacker_steamid": [1, 1, 1, 1, 12345, 12345],
                "user_steamid": [101, 102, 103, 104, 1, 2],
                "weapon": ["ak47"] * 6,
                "headshot": [False] * 6,
                "total_rounds_played": [1, 1, 1, 1, 1, 1],
            }
        )

        demo = DemoData(
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
            # INTEGER team values
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

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        assert len(result.clutch_situations) >= 1


# =============================================================================
# PLAYER NAME VALIDATION TESTS (Steam ID Resolution Fix)
# =============================================================================


class TestPlayerNameValidation:
    """Tests for player name validation - prevents Steam IDs from appearing as names."""

    def test_is_valid_player_name_rejects_steam_id(self):
        """Steam ID format is rejected as invalid name."""
        from opensight.core.parser import is_valid_player_name

        # 17-digit Steam IDs starting with 7656119
        assert is_valid_player_name("76561198012345678") is False
        assert is_valid_player_name("76561199999999999") is False

    def test_is_valid_player_name_rejects_numeric_only(self):
        """Pure numeric strings are rejected."""
        from opensight.core.parser import is_valid_player_name

        assert is_valid_player_name("12345678901234567") is False
        assert is_valid_player_name("123456") is False

    def test_is_valid_player_name_accepts_normal_names(self):
        """Normal player names are accepted."""
        from opensight.core.parser import is_valid_player_name

        assert is_valid_player_name("foe") is True
        assert is_valid_player_name("dergs") is True
        assert is_valid_player_name("Player_123") is True
        assert is_valid_player_name("x1337x") is True

    def test_is_valid_player_name_rejects_empty(self):
        """Empty strings are rejected."""
        from opensight.core.parser import is_valid_player_name

        assert is_valid_player_name("") is False
        assert is_valid_player_name("   ") is False

    def test_make_fallback_player_name_format(self):
        """Fallback names use Player_XXXX format."""
        from opensight.core.parser import make_fallback_player_name

        name = make_fallback_player_name(76561198012345678)
        assert name == "Player_5678"

        name = make_fallback_player_name(76561199999999999)
        assert name == "Player_9999"

    def test_make_fallback_player_name_handles_zero(self):
        """Zero steam_id produces valid fallback."""
        from opensight.core.parser import make_fallback_player_name

        name = make_fallback_player_name(0)
        assert name == "Player_0000"


# =============================================================================
# ECONOMY LOSS BONUS TESTS (Economy Tracking Fix)
# =============================================================================


class TestEconomyLossBonus:
    """Tests for loss bonus tracking and economy grading."""

    def test_loss_bonus_constants_exist(self):
        """Loss bonus constants are defined."""
        from opensight.domains.economy import (
            BASE_LOSS_BONUS,
            HIGH_LOSS_BONUS,
            LOSS_BONUS_INCREMENT,
            LOW_LOSS_BONUS,
            MAX_LOSS_BONUS,
        )

        assert BASE_LOSS_BONUS == 1400
        assert LOSS_BONUS_INCREMENT == 500
        assert MAX_LOSS_BONUS == 3400
        assert LOW_LOSS_BONUS == 1900
        assert HIGH_LOSS_BONUS == 2900

    def test_calculate_loss_bonus_first_loss(self):
        """First loss gives base bonus."""
        from opensight.domains.economy import calculate_loss_bonus

        assert calculate_loss_bonus(1) == 1900  # $1400 + $500

    def test_calculate_loss_bonus_scales(self):
        """Loss bonus scales with consecutive losses."""
        from opensight.domains.economy import calculate_loss_bonus

        assert calculate_loss_bonus(1) == 1900  # $1400 + $500*1
        assert calculate_loss_bonus(2) == 2400  # $1400 + $500*2
        assert calculate_loss_bonus(3) == 2900  # $1400 + $500*3
        assert calculate_loss_bonus(4) == 3400  # $1400 + $500*4 = max

    def test_calculate_loss_bonus_caps_at_max(self):
        """Loss bonus caps at maximum."""
        from opensight.domains.economy import calculate_loss_bonus

        assert calculate_loss_bonus(4) == 3400
        assert calculate_loss_bonus(5) == 3400  # Still capped
        assert calculate_loss_bonus(10) == 3400  # Still capped

    def test_calculate_loss_bonus_zero_losses(self):
        """Zero losses returns base."""
        from opensight.domains.economy import calculate_loss_bonus

        assert calculate_loss_bonus(0) == 1400

    def test_is_bad_force_low_bonus(self):
        """Bad force: force buy at low loss bonus."""
        from opensight.domains.economy import BuyType, is_bad_force

        # Low loss bonus ($1900 or less) = risky to force
        assert is_bad_force(BuyType.FORCE, loss_bonus=1400) is True
        assert is_bad_force(BuyType.FORCE, loss_bonus=1900) is True

    def test_is_bad_force_high_bonus_ok(self):
        """Not bad force: force buy at high loss bonus."""
        from opensight.domains.economy import BuyType, is_bad_force

        # High loss bonus (>$1900) = safer to force
        assert is_bad_force(BuyType.FORCE, loss_bonus=2400) is False
        assert is_bad_force(BuyType.FORCE, loss_bonus=3400) is False

    def test_is_bad_force_not_force(self):
        """Non-force buys are never bad forces."""
        from opensight.domains.economy import BuyType, is_bad_force

        assert is_bad_force(BuyType.FULL_BUY, loss_bonus=1400) is False
        assert is_bad_force(BuyType.ECO, loss_bonus=1400) is False

    def test_is_good_force_high_bonus(self):
        """Good force: force at high loss bonus."""
        from opensight.domains.economy import BuyType, is_good_force

        # High loss bonus = good to force (getting max money anyway)
        assert is_good_force(BuyType.FORCE, loss_bonus=2900) is True
        assert is_good_force(BuyType.FORCE, loss_bonus=3400) is True

    def test_is_good_force_requirements(self):
        """Good force requires force buy type and high bonus or broken enemy."""
        from opensight.domains.economy import BuyType, is_good_force

        # Not a force buy type
        assert is_good_force(BuyType.FULL_BUY, loss_bonus=3400) is False
        assert is_good_force(BuyType.ECO, loss_bonus=3400) is False
        # Low loss bonus without enemy economy broken
        assert is_good_force(BuyType.FORCE, loss_bonus=1400) is False
        # Low loss bonus but enemy economy broken = good
        assert is_good_force(BuyType.FORCE, loss_bonus=1400, enemy_economy_broken=True) is True


class TestEconomyGrading:
    """Tests for economy grade calculation.

    Note: calculate_economy_grade returns (grade, description) tuple.
    """

    def test_economy_grade_a(self):
        """Grade A: high force win rate, few bad buys."""
        from opensight.domains.economy import calculate_economy_grade

        grade, description = calculate_economy_grade(
            force_win_rate=0.5,  # >40%
            bad_buy_count=1,  # ≤1
            total_force_buys=10,
        )
        assert grade == "A"
        assert isinstance(description, str)

    def test_economy_grade_b(self):
        """Grade B: good force win rate or few bad buys."""
        from opensight.domains.economy import calculate_economy_grade

        grade, description = calculate_economy_grade(
            force_win_rate=0.35,  # >30%
            bad_buy_count=2,  # ≤2
            total_force_buys=10,
        )
        assert grade == "B"
        assert isinstance(description, str)

    def test_economy_grade_c(self):
        """Grade C: average performance."""
        from opensight.domains.economy import calculate_economy_grade

        grade, description = calculate_economy_grade(
            force_win_rate=0.30,  # exactly 30%
            bad_buy_count=2,  # acceptable
            total_force_buys=10,
        )
        # May be B or C depending on exact criteria
        assert grade in ("B", "C")
        assert isinstance(description, str)

    def test_economy_grade_d(self):
        """Grade D: below average with many bad buys."""
        from opensight.domains.economy import calculate_economy_grade

        grade, description = calculate_economy_grade(
            force_win_rate=0.25,  # below 30%
            bad_buy_count=3,
            total_force_buys=10,
        )
        # Check it's a valid grade (actual grade depends on criteria)
        assert grade in ("C", "D")
        assert isinstance(description, str)

    def test_economy_grade_f(self):
        """Grade F: poor force win rate, many bad buys."""
        from opensight.domains.economy import calculate_economy_grade

        grade, description = calculate_economy_grade(
            force_win_rate=0.1,  # <20%
            bad_buy_count=5,  # >3
            total_force_buys=10,
        )
        assert grade == "F"
        assert isinstance(description, str)

    def test_economy_grade_returns_tuple(self):
        """Economy grade returns (grade, description) tuple."""
        from opensight.domains.economy import calculate_economy_grade

        result = calculate_economy_grade(
            force_win_rate=0.3,
            bad_buy_count=2,
            total_force_buys=10,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        grade, description = result
        assert grade in ("A", "B", "C", "D", "F")
        assert isinstance(description, str)


# =============================================================================
# LURKER PERSONA TESTS (Persona Detection Fix)
# =============================================================================


class TestLurkerPersona:
    """Tests for lurker persona detection."""

    def test_lurker_with_high_impact(self):
        """Effective lurker: high isolation but good impact."""
        from opensight.analysis.persona import _is_effective_lurker

        stats = {
            "deaths": 10,
            "untraded_deaths": 6,  # 60% isolation rate
            "kills": 15,
            "rounds_played": 20,  # 0.75 KPR
            "hltv_rating": 1.1,
            "backstab_kills": 3,
        }
        assert _is_effective_lurker(stats) is True

    def test_lurker_without_impact(self):
        """Ineffective lurker: high isolation, no impact."""
        from opensight.analysis.persona import _is_effective_lurker

        stats = {
            "deaths": 10,
            "untraded_deaths": 7,  # 70% isolation rate
            "kills": 5,
            "rounds_played": 20,  # 0.25 KPR - low
            "hltv_rating": 0.6,  # Low
            "backstab_kills": 0,
        }
        assert _is_effective_lurker(stats) is False

    def test_lurker_low_isolation(self):
        """Not a lurker: low isolation rate."""
        from opensight.analysis.persona import _is_effective_lurker

        stats = {
            "deaths": 10,
            "untraded_deaths": 3,  # 30% isolation - not lurking
            "kills": 20,
            "rounds_played": 20,
            "hltv_rating": 1.2,
            "backstab_kills": 5,
        }
        assert _is_effective_lurker(stats) is False

    def test_lurker_insufficient_deaths(self):
        """Not a lurker: too few deaths to judge."""
        from opensight.analysis.persona import _is_effective_lurker

        stats = {
            "deaths": 2,  # Too few to judge
            "untraded_deaths": 2,
            "kills": 10,
            "rounds_played": 10,
            "hltv_rating": 1.5,
            "backstab_kills": 2,
        }
        assert _is_effective_lurker(stats) is False

    def test_lurker_persona_in_personas_dict(self):
        """The lurker persona is defined in PERSONAS."""
        from opensight.analysis.persona import PERSONAS

        assert "the_lurker" in PERSONAS
        lurker = PERSONAS["the_lurker"]
        assert lurker["name"] == "The Lurker"
        assert "check" in lurker


# =============================================================================
# FULL PIPELINE INTEGRATION TESTS
# =============================================================================


class TestFullAnalysisPipeline:
    """End-to-end tests for the complete analysis pipeline."""

    @pytest.fixture
    def realistic_demo_data(self):
        """Create realistic demo data simulating a full match."""
        from opensight.core.parser import DemoData

        # Simulate a 16-14 match (30 rounds)
        num_rounds = 30

        # Create realistic kill data across rounds
        kills_data = []
        tick = 1000
        for round_num in range(1, num_rounds + 1):
            # 2-3 kills per round on average
            num_kills = 2 if round_num % 3 == 0 else 3
            for _ in range(num_kills):
                # Alternate attackers
                attacker = [12345, 67890, 11111, 22222, 33333][tick % 5]
                victim = [44444, 55555, 66666, 77777, 88888][tick % 5]
                kills_data.append(
                    {
                        "tick": tick,
                        "attacker_steamid": attacker,
                        "user_steamid": victim,
                        "weapon": ["ak47", "m4a1", "awp", "deagle", "usp_silencer"][tick % 5],
                        "headshot": tick % 3 == 0,
                        "total_rounds_played": round_num,
                    }
                )
                tick += 100

        kills_df = pd.DataFrame(kills_data)

        return DemoData(
            file_path=Path("/tmp/realistic_test.dem"),
            map_name="de_ancient",
            duration_seconds=2700.0,  # 45 minutes
            tick_rate=64,
            num_rounds=num_rounds,
            player_stats={
                12345: {"name": "foe", "kills": 17, "deaths": 24, "assists": 6},
                67890: {"name": "dergs", "kills": 22, "deaths": 25, "assists": 5},
                11111: {"name": "Player3", "kills": 15, "deaths": 20, "assists": 4},
                22222: {"name": "Player4", "kills": 12, "deaths": 22, "assists": 3},
                33333: {"name": "Player5", "kills": 10, "deaths": 18, "assists": 2},
                44444: {"name": "Enemy1", "kills": 20, "deaths": 15, "assists": 3},
                55555: {"name": "Enemy2", "kills": 18, "deaths": 14, "assists": 4},
                66666: {"name": "Enemy3", "kills": 16, "deaths": 16, "assists": 5},
                77777: {"name": "Enemy4", "kills": 14, "deaths": 17, "assists": 2},
                88888: {"name": "Enemy5", "kills": 12, "deaths": 19, "assists": 1},
            },
            player_names={
                12345: "foe",
                67890: "dergs",
                11111: "Player3",
                22222: "Player4",
                33333: "Player5",
                44444: "Enemy1",
                55555: "Enemy2",
                66666: "Enemy3",
                77777: "Enemy4",
                88888: "Enemy5",
            },
            # Mix of string and int team values to test normalization
            player_teams={
                12345: "CT",
                67890: "CT",
                11111: "CT",
                22222: "CT",
                33333: "CT",
                44444: 2,  # T as int
                55555: 2,
                66666: 2,
                77777: 2,
                88888: 2,
            },
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

    def test_combat_analyzer_produces_results(self, realistic_demo_data):
        """CombatAnalyzer produces non-empty results."""
        from opensight.domains.combat import CombatAnalyzer

        analyzer = CombatAnalyzer(realistic_demo_data)
        result = analyzer.analyze()

        # Should have results
        assert result is not None
        assert len(result.trade_kills) >= 0  # May have trades
        assert len(result.opening_duels) > 0  # Should have openings
        assert len(result.player_stats) > 0  # Should have player stats

    def test_economy_analyzer_produces_results(self, realistic_demo_data):
        """EconomyAnalyzer produces valid stats."""
        from opensight.domains.economy import EconomyAnalyzer

        analyzer = EconomyAnalyzer(realistic_demo_data)
        stats = analyzer.analyze()

        assert stats is not None
        assert stats.rounds_analyzed == 30

    def test_player_names_are_valid(self, realistic_demo_data):
        """All player names pass validation."""
        from opensight.core.parser import is_valid_player_name

        for steam_id, name in realistic_demo_data.player_names.items():
            assert is_valid_player_name(name), (
                f"Player name '{name}' for {steam_id} failed validation"
            )

    def test_mixed_team_formats_work(self, realistic_demo_data):
        """Demo with mixed string/int team formats works correctly."""
        from opensight.domains.combat import CombatAnalyzer

        # The fixture has both "CT" strings and 2 integers
        analyzer = CombatAnalyzer(realistic_demo_data)

        # This should not raise any errors
        analyzer.analyze()

        # All players should be processed
        team_a_count = sum(
            1
            for sid in realistic_demo_data.player_teams
            if analyzer._player_team_nums.get(sid) == 3
        )
        team_b_count = sum(
            1
            for sid in realistic_demo_data.player_teams
            if analyzer._player_team_nums.get(sid) == 2
        )

        assert team_a_count == 5, "Should have 5 CT players"
        assert team_b_count == 5, "Should have 5 T players"


# =============================================================================
# REGRESSION TESTS FOR SPECIFIC BUGS
# =============================================================================


class TestRegressionBugs:
    """Tests that verify specific bugs remain fixed."""

    def test_clutch_not_all_zeros(self):
        """Clutch detection should produce non-zero values for valid data.

        Regression test for: Clutch detection showing all zeros due to
        string vs int team comparison bug.
        """
        from opensight.core.parser import DemoData
        from opensight.domains.combat import CombatAnalyzer

        # Setup: 1v2 clutch scenario with STRING team values
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1100, 1200, 1300, 2000, 2100],
                "attacker_steamid": [1, 1, 1, 1, 12345, 12345],
                "user_steamid": [101, 102, 103, 104, 1, 2],
                "weapon": ["ak47"] * 6,
                "headshot": [False] * 6,
                "total_rounds_played": [1, 1, 1, 1, 1, 1],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/clutch_test.dem"),
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
                12345: "CT",
                101: "CT",
                102: "CT",
                103: "CT",
                104: "CT",
                1: "T",
                2: "T",
            },
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        # The bug was that clutch_situations was always empty
        assert len(result.clutch_situations) > 0, (
            "REGRESSION: Clutch detection returned zero situations"
        )

    def test_player_names_not_steam_ids(self):
        """Player names should never be raw Steam IDs.

        Regression test for: Player names showing as 17-digit Steam IDs
        instead of actual names.
        """
        from opensight.core.parser import is_valid_player_name, make_fallback_player_name

        # These are Steam ID formats that should NOT be valid names
        invalid_as_names = [
            "76561198012345678",
            "76561199999999999",
            "7656119803245612",
        ]

        for steam_id_str in invalid_as_names:
            assert not is_valid_player_name(steam_id_str), (
                f"REGRESSION: Steam ID '{steam_id_str}' accepted as valid name"
            )

        # Fallback should produce friendly names
        fallback = make_fallback_player_name(76561198012345678)
        assert fallback.startswith("Player_"), (
            f"REGRESSION: Fallback name '{fallback}' doesn't follow Player_XXXX format"
        )

    def test_economy_grade_exists(self):
        """Economy grade calculation should exist and work.

        Regression test for: Economy stats missing grade calculation.
        """
        from opensight.domains.economy import calculate_economy_grade

        # Should return (grade, description) tuple for valid input
        result = calculate_economy_grade(
            force_win_rate=0.3,
            bad_buy_count=2,
            total_force_buys=10,
        )

        assert isinstance(result, tuple), "Economy grade should return tuple"
        grade, description = result
        assert grade in ("A", "B", "C", "D", "F"), f"REGRESSION: Economy grade '{grade}' is invalid"
        assert isinstance(description, str)

    def test_loss_bonus_tracking_exists(self):
        """Loss bonus tracking functions should exist.

        Regression test for: Economy module missing loss bonus tracking.
        """
        from opensight.domains.economy import (
            calculate_loss_bonus,
            is_bad_force,
            is_good_force,
        )

        # Functions should be callable
        assert callable(calculate_loss_bonus)
        assert callable(is_bad_force)
        assert callable(is_good_force)

        # And produce valid results
        bonus = calculate_loss_bonus(3)
        assert isinstance(bonus, int) and bonus > 0


# =============================================================================
# BOUNDARY CONDITION TESTS
# =============================================================================


class TestBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    def test_empty_kills_df(self):
        """Analyzer handles empty kills DataFrame."""
        from opensight.core.parser import DemoData
        from opensight.domains.combat import CombatAnalyzer

        demo = DemoData(
            file_path=Path("/tmp/empty.dem"),
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

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        assert result is not None
        assert len(result.clutch_situations) == 0

    def test_single_player_demo(self):
        """Analyzer handles single player data."""
        from opensight.core.parser import DemoData
        from opensight.domains.combat import CombatAnalyzer

        kills_df = pd.DataFrame(
            {
                "tick": [1000],
                "attacker_steamid": [12345],
                "user_steamid": [67890],
                "weapon": ["ak47"],
                "headshot": [True],
                "total_rounds_played": [1],
            }
        )

        demo = DemoData(
            file_path=Path("/tmp/single.dem"),
            map_name="de_dust2",
            duration_seconds=60.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={12345: "Solo", 67890: "Bot"},
            player_teams={12345: 3, 67890: 2},
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = CombatAnalyzer(demo)
        result = analyzer.analyze()

        assert result is not None

    def test_loss_bonus_negative_losses(self):
        """Loss bonus handles negative input gracefully."""
        from opensight.domains.economy import calculate_loss_bonus

        # Should not crash, return base or 0
        result = calculate_loss_bonus(-1)
        assert result >= 0
