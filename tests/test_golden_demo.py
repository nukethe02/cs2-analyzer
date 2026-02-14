"""
Golden Master Regression Test - Team Identity & Rating Lock.

PURPOSE:
This test ensures that core analytics logic NEVER regresses. If any of these
values change by even 0.01, the test fails and blocks the build.

WHAT IT PROTECTS:
1. Team Identity - Players must always be assigned to the correct team
2. HLTV Rating - Rating calculations must produce exact, reproducible values
3. KAST% - Round-level metrics must be stable
4. Impact Rating - Impact calculations must not drift

HOW TO USE:
1. Set GOLDEN_DEMO_PATH to your reference demo file
2. Run the test once to establish baseline values
3. If values change legitimately, update GOLDEN_EXPECTATIONS with new values
4. Never disable this test without explicit sign-off

WHEN THIS FAILS:
- Someone changed the rating formula without updating the test
- Team swap logic was broken (Tr1d bug)
- Safe accessor behavior changed
- DataFrame column handling was modified

Author: Created as part of Boris Workflow - Campaign 2
"""

import os
from pathlib import Path

import pytest

# =============================================================================
# GOLDEN MASTER CONFIGURATION
# =============================================================================

# Path to the reference demo file - checked in order:
# 1. GOLDEN_DEMO_PATH env var (CI or custom location)
# 2. tests/fixtures/golden_master.dem (local dev standard location)
# 3. Original Downloads path (legacy fallback)
_FIXTURES_PATH = str(Path(__file__).parent / "fixtures" / "golden_master.dem")
_DOWNLOADS_PATH = "C:/Users/lglea/Downloads/1-7d912af3-b313-498a-83f0-9a29366420b3-1-1.dem"
GOLDEN_DEMO_PATH = os.environ.get("GOLDEN_DEMO_PATH") or (
    _FIXTURES_PATH if Path(_FIXTURES_PATH).exists() else _DOWNLOADS_PATH
)

# Expected values for the golden demo - UPDATE THESE WHEN LEGITIMATELY CHANGED
# Format: {"player_name": {"team": "...", "rating": X.XX, "kast": XX.X, ...}}
#
# NOTE: Team detection is now working (as of 2026-02-01).
# kast_percentage > 100% is a known issue - KAST calculation bug needs fixing.
#
GOLDEN_EXPECTATIONS = {
    "Docobo": {
        "team": "CT",
        "hltv_rating": 1.3,
        "kast_percentage": 71.0,
        "impact_rating": 2.03,
        "kills": 26,
        "deaths": 22,
        "adr": 105.5,
    },
    "Tr1d": {
        "team": "T",
        "hltv_rating": 1.43,
        "kast_percentage": 96.8,
        "impact_rating": 1.56,
        "kills": 24,
        "deaths": 21,
        "adr": 121.2,
    },
    "miasma": {
        "team": "T",
        "hltv_rating": 0.92,
        "kast_percentage": 58.1,
        "impact_rating": 1.42,
        "kills": 21,
        "deaths": 23,
        "adr": 85.2,
    },
    "DavidLithium": {
        "team": "CT",
        "hltv_rating": 1.0,
        "kast_percentage": 96.8,
        "impact_rating": 1.03,
        "kills": 17,
        "deaths": 22,
        "adr": 69.8,
    },
    "kix": {
        "team": "T",
        "hltv_rating": 1.01,
        "kast_percentage": 67.7,
        "impact_rating": 1.47,
        "kills": 25,
        "deaths": 28,
        "adr": 104.3,
    },
    "ina": {
        "team": "CT",
        "hltv_rating": 1.48,
        "kast_percentage": 71.0,
        "impact_rating": 2.49,
        "kills": 32,
        "deaths": 23,
        "adr": 112.7,
    },
    "dergs": {
        "team": "T",
        "hltv_rating": 0.9,
        "kast_percentage": 54.8,
        "impact_rating": 1.37,
        "kills": 22,
        "deaths": 25,
        "adr": 99.2,
    },
    "129310238324": {
        "team": "CT",
        "hltv_rating": 1.49,
        "kast_percentage": 77.4,
        "impact_rating": 2.0,
        "kills": 30,
        "deaths": 21,
        "adr": 129.2,
    },
    "snowing-": {
        "team": "CT",
        "hltv_rating": 0.86,
        "kast_percentage": 58.1,
        "impact_rating": 1.24,
        "kills": 17,
        "deaths": 23,
        "adr": 98.8,
    },
    "foe": {
        "team": "T",
        "hltv_rating": 0.69,
        "kast_percentage": 54.8,
        "impact_rating": 1.04,
        "kills": 17,
        "deaths": 24,
        "adr": 74.1,
    },
}

# Players who must ALWAYS be on specific teams (team identity lock)
# NOTE: Team detection is currently broken (all show "Unknown")
# Once fixed, add entries like:
# TEAM_IDENTITY_LOCK = {"mdma": "Team A", "4SkinsLittleBuddy": "Team B"}
TEAM_IDENTITY_LOCK = {
    # "Tr1d": "Team A",  # Tr1d must NEVER appear on Team B
}

# Tolerance for floating point comparisons - SET TO 0.0 FOR EXACT MATCH
RATING_TOLERANCE = 0.0  # No drift allowed
PERCENTAGE_TOLERANCE = 0.0  # No drift allowed


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def golden_demo_analysis():
    """
    Parse and analyze the golden master demo once per test module.

    This fixture is expensive (demo parsing takes time), so it's scoped
    to the module to avoid re-parsing for each test.
    """
    if not Path(GOLDEN_DEMO_PATH).exists():
        pytest.skip(
            f"Golden demo not found at {GOLDEN_DEMO_PATH}. "
            f"Set GOLDEN_DEMO_PATH environment variable or create fixtures/golden_master.dem"
        )

    try:
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.core.parser import DemoParser

        # Parse the demo
        parser = DemoParser(Path(GOLDEN_DEMO_PATH))
        data = parser.parse()

        # Analyze it
        analyzer = DemoAnalyzer(data)
        result = analyzer.analyze()

        return result

    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")
    except Exception as e:
        pytest.fail(f"Failed to parse golden demo: {e}")


@pytest.fixture
def player_stats_by_name(golden_demo_analysis):
    """Extract player stats indexed by player name for easy lookup."""
    # MatchAnalysis.players is dict[int, PlayerMatchStats] - convert to name-keyed dicts
    return {
        player.name: {
            "team": player.team,
            "hltv_rating": round(player.hltv_rating, 2),
            "kast_percentage": round(player.kast_percentage, 1),
            "impact_rating": round(player.impact_rating, 2),
            "kills": player.kills,
            "deaths": player.deaths,
            "adr": round(player.adr, 1),
        }
        for _steam_id, player in golden_demo_analysis.players.items()
    }


# =============================================================================
# TEAM IDENTITY TESTS - The "Tr1d Bug" Protection
# =============================================================================


class TestTeamIdentity:
    """
    Ensure players are ALWAYS assigned to the correct team.

    This protects against the "Tr1d Team Swap" bug where players
    were incorrectly assigned to the wrong team due to halftime
    swap logic errors.
    """

    @pytest.mark.skipif(
        not TEAM_IDENTITY_LOCK,
        reason="No team identity locks configured - add players to TEAM_IDENTITY_LOCK",
    )
    def test_locked_players_on_correct_team(self, player_stats_by_name):
        """
        CRITICAL: Players in TEAM_IDENTITY_LOCK must be on their locked team.

        If this fails, the team swap logic has regressed.
        """
        errors = []

        for player_name, expected_team in TEAM_IDENTITY_LOCK.items():
            if player_name not in player_stats_by_name:
                errors.append(f"Player '{player_name}' not found in analysis results")
                continue

            actual_team = player_stats_by_name[player_name].get("team", "MISSING")

            if actual_team != expected_team:
                errors.append(
                    f"TEAM SWAP BUG: '{player_name}' is on '{actual_team}' "
                    f"but should be on '{expected_team}'"
                )

        if errors:
            pytest.fail(
                "TEAM IDENTITY REGRESSION DETECTED:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def test_all_players_have_team_assigned(self, player_stats_by_name):
        """Every player must have a team assignment."""
        for name, stats in player_stats_by_name.items():
            team = stats.get("team")
            assert team is not None, f"Player '{name}' has no team assigned"
            assert team in (
                "Team A",
                "Team B",
                "CT",
                "T",
            ), f"Player '{name}' has invalid team: '{team}'"

    def test_teams_are_balanced(self, player_stats_by_name):
        """Each team should have roughly equal players (5v5)."""
        teams = {}
        for _name, stats in player_stats_by_name.items():
            team = stats.get("team", "Unknown")
            teams[team] = teams.get(team, 0) + 1

        # Standard competitive is 5v5
        team_sizes = list(teams.values())
        if len(team_sizes) >= 2:
            # Allow for players who disconnected (4-5 per team)
            for size in team_sizes:
                assert 4 <= size <= 5, f"Team sizes look wrong: {teams}"


# =============================================================================
# RATING REGRESSION TESTS - Exact Value Lock
# =============================================================================


class TestRatingRegression:
    """
    Ensure ratings produce EXACT expected values.

    Even a 0.01 change in rating indicates a regression in the formula
    or data pipeline. This catches subtle bugs before they ship.
    """

    @pytest.mark.skipif(
        not GOLDEN_EXPECTATIONS,
        reason="No golden expectations configured - add values to GOLDEN_EXPECTATIONS",
    )
    def test_hltv_ratings_exact_match(self, player_stats_by_name):
        """
        CRITICAL: HLTV ratings must match expected values EXACTLY.

        Formula: Rating = 0.0073*KAST + 0.3591*KPR - 0.5329*DPR +
                         0.2372*Impact + 0.0032*ADR + 0.1587*RMK
        """
        errors = []

        for player_name, expected in GOLDEN_EXPECTATIONS.items():
            if "hltv_rating" not in expected:
                continue

            if player_name not in player_stats_by_name:
                errors.append(f"Player '{player_name}' not found")
                continue

            actual = player_stats_by_name[player_name].get("hltv_rating")
            expected_rating = expected["hltv_rating"]

            if actual is None:
                errors.append(f"'{player_name}' has no HLTV rating (expected {expected_rating})")
                continue

            diff = abs(actual - expected_rating)
            if diff > RATING_TOLERANCE:
                errors.append(
                    f"'{player_name}' HLTV rating drift: "
                    f"expected {expected_rating}, got {actual} (diff={diff:.4f})"
                )

        if errors:
            pytest.fail(
                "HLTV RATING REGRESSION DETECTED:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @pytest.mark.skipif(
        not GOLDEN_EXPECTATIONS,
        reason="No golden expectations configured",
    )
    def test_kast_percentages_exact_match(self, player_stats_by_name):
        """KAST% must match expected values exactly."""
        errors = []

        for player_name, expected in GOLDEN_EXPECTATIONS.items():
            if "kast_percentage" not in expected:
                continue

            if player_name not in player_stats_by_name:
                continue

            actual = player_stats_by_name[player_name].get("kast_percentage")
            expected_kast = expected["kast_percentage"]

            if actual is None:
                errors.append(f"'{player_name}' has no KAST% (expected {expected_kast})")
                continue

            diff = abs(actual - expected_kast)
            if diff > PERCENTAGE_TOLERANCE:
                errors.append(
                    f"'{player_name}' KAST% drift: "
                    f"expected {expected_kast}, got {actual} (diff={diff:.2f})"
                )

        if errors:
            pytest.fail(
                "KAST PERCENTAGE REGRESSION DETECTED:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @pytest.mark.skipif(
        not GOLDEN_EXPECTATIONS,
        reason="No golden expectations configured",
    )
    def test_impact_ratings_exact_match(self, player_stats_by_name):
        """Impact ratings must match expected values exactly."""
        errors = []

        for player_name, expected in GOLDEN_EXPECTATIONS.items():
            if "impact_rating" not in expected:
                continue

            if player_name not in player_stats_by_name:
                continue

            actual = player_stats_by_name[player_name].get("impact_rating")
            expected_impact = expected["impact_rating"]

            if actual is None:
                errors.append(f"'{player_name}' has no impact rating (expected {expected_impact})")
                continue

            diff = abs(actual - expected_impact)
            if diff > RATING_TOLERANCE:
                errors.append(
                    f"'{player_name}' impact rating drift: "
                    f"expected {expected_impact}, got {actual} (diff={diff:.4f})"
                )

        if errors:
            pytest.fail(
                "IMPACT RATING REGRESSION DETECTED:\n" + "\n".join(f"  - {e}" for e in errors)
            )


# =============================================================================
# KILL/DEATH COUNT TESTS - Data Pipeline Integrity
# =============================================================================


class TestKillDeathCounts:
    """
    Ensure kill/death counts are EXACTLY correct.

    If these change, the demo parser or kill tracking logic has regressed.
    """

    @pytest.mark.skipif(
        not GOLDEN_EXPECTATIONS,
        reason="No golden expectations configured",
    )
    def test_kill_counts_exact(self, player_stats_by_name):
        """Kill counts must be exact integers."""
        errors = []

        for player_name, expected in GOLDEN_EXPECTATIONS.items():
            if "kills" not in expected:
                continue

            if player_name not in player_stats_by_name:
                continue

            actual = player_stats_by_name[player_name].get("kills")
            expected_kills = expected["kills"]

            if actual != expected_kills:
                errors.append(
                    f"'{player_name}' kill count: expected {expected_kills}, got {actual}"
                )

        if errors:
            pytest.fail("KILL COUNT REGRESSION DETECTED:\n" + "\n".join(f"  - {e}" for e in errors))

    @pytest.mark.skipif(
        not GOLDEN_EXPECTATIONS,
        reason="No golden expectations configured",
    )
    def test_death_counts_exact(self, player_stats_by_name):
        """Death counts must be exact integers."""
        errors = []

        for player_name, expected in GOLDEN_EXPECTATIONS.items():
            if "deaths" not in expected:
                continue

            if player_name not in player_stats_by_name:
                continue

            actual = player_stats_by_name[player_name].get("deaths")
            expected_deaths = expected["deaths"]

            if actual != expected_deaths:
                errors.append(
                    f"'{player_name}' death count: expected {expected_deaths}, got {actual}"
                )

        if errors:
            pytest.fail(
                "DEATH COUNT REGRESSION DETECTED:\n" + "\n".join(f"  - {e}" for e in errors)
            )


# =============================================================================
# FULL SNAPSHOT TEST - Nuclear Option
# =============================================================================


class TestFullSnapshot:
    """
    Compare entire player stat dictionaries against golden expectations.

    This is the "nuclear option" - if ANYTHING changes, it fails.
    """

    @pytest.mark.skipif(
        not GOLDEN_EXPECTATIONS,
        reason="No golden expectations configured",
    )
    def test_full_stats_snapshot(self, player_stats_by_name):
        """
        Every expected field must match exactly.

        This catches any drift in any metric, not just the main ones.
        """
        errors = []

        for player_name, expected in GOLDEN_EXPECTATIONS.items():
            if player_name not in player_stats_by_name:
                errors.append(f"Player '{player_name}' missing from results")
                continue

            actual = player_stats_by_name[player_name]

            for field, expected_value in expected.items():
                actual_value = actual.get(field)

                if actual_value is None:
                    errors.append(f"'{player_name}'.{field}: missing (expected {expected_value})")
                    continue

                # Compare based on type
                if isinstance(expected_value, float):
                    if abs(actual_value - expected_value) > RATING_TOLERANCE:
                        errors.append(
                            f"'{player_name}'.{field}: "
                            f"expected {expected_value}, got {actual_value}"
                        )
                elif actual_value != expected_value:
                    errors.append(
                        f"'{player_name}'.{field}: expected {expected_value}, got {actual_value}"
                    )

        if errors:
            pytest.fail(
                "FULL SNAPSHOT REGRESSION DETECTED:\n" + "\n".join(f"  - {e}" for e in errors)
            )


# =============================================================================
# UTILITY: Generate Golden Expectations from Demo
# =============================================================================


def generate_golden_expectations(demo_path: str | Path) -> dict:
    """
    Utility function to generate GOLDEN_EXPECTATIONS from a demo.

    Run this once to establish your baseline, then copy the output
    to GOLDEN_EXPECTATIONS at the top of this file.

    Usage:
        from tests.test_golden_demo import generate_golden_expectations
        expectations = generate_golden_expectations("path/to/demo.dem")
        print(expectations)
    """
    from opensight.analysis.analytics import DemoAnalyzer
    from opensight.core.parser import DemoParser

    parser = DemoParser(Path(demo_path))
    data = parser.parse()
    analyzer = DemoAnalyzer(data)
    result = analyzer.analyze()

    expectations = {}
    # MatchAnalysis.players is dict[int, PlayerMatchStats]
    for _steam_id, player in result.players.items():
        expectations[player.name] = {
            "team": player.team,
            "hltv_rating": round(player.hltv_rating, 2),
            "kast_percentage": round(player.kast_percentage, 1),
            "impact_rating": round(player.impact_rating, 2),
            "kills": player.kills,
            "deaths": player.deaths,
            "adr": round(player.adr, 1),
        }

    return expectations


if __name__ == "__main__":
    # When run directly, generate expectations from the golden demo
    import sys

    if len(sys.argv) > 1:
        demo_path = sys.argv[1]
    else:
        demo_path = GOLDEN_DEMO_PATH

    print(f"Generating golden expectations from: {demo_path}")
    expectations = generate_golden_expectations(demo_path)

    print("\n# Copy this to GOLDEN_EXPECTATIONS:\n")
    print("GOLDEN_EXPECTATIONS = {")
    for name, stats in expectations.items():
        print(f'    "{name}": {{')
        for key, value in stats.items():
            if isinstance(value, str):
                print(f'        "{key}": "{value}",')
            else:
                print(f'        "{key}": {value},')
        print("    },")
    print("}")
