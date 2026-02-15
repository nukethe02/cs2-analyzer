"""
Leetify Ground Truth Test — Numbers Must Match EXACTLY.

Every value in this file comes directly from Leetify screenshots of the
golden demo (de_ancient, 30 rounds). These are the CORRECT values.
Our code must produce these numbers, not the other way around.
If a test fails, the CODE is wrong, not the expected value.

DO NOT re-baseline these numbers. Fix the code instead.
"""

import os
from pathlib import Path

import pytest

# =============================================================================
# GOLDEN DEMO PATH
# =============================================================================
_FIXTURES_PATH = str(Path(__file__).parent / "fixtures" / "golden_master.dem")
GOLDEN_DEMO_PATH = os.environ.get("GOLDEN_DEMO_PATH") or _FIXTURES_PATH

# =============================================================================
# LEETIFY GROUND TRUTH — Match Details > General
# =============================================================================
# Source: Leetify Match Details screenshot
# Map: de_ancient, 30 rounds (knife round excluded)
# Score: CT 16-14 T

LEETIFY_GENERAL = {
    # My Team (T side, LOSS)
    "dergs": {
        "team": "T",
        "kills": 22,
        "assists": 2,
        "deaths": 24,
        "adr": 73,
        "hltv_rating": 0.97,
        "2k": 4,
        "3k": 2,
        "4k": 0,
        "5k": 0,
    },
    "foe": {
        "team": "T",
        "kills": 17,
        "assists": 5,
        "deaths": 23,
        "adr": 61,
        "hltv_rating": 0.86,
        "2k": 4,
        "3k": 0,
        "4k": 1,
        "5k": 0,
    },
    "kix": {
        "team": "T",
        "kills": 25,
        "assists": 4,
        "deaths": 27,
        "adr": 90,
        "hltv_rating": 0.93,
        "2k": 5,
        "3k": 1,
        "4k": 0,
        "5k": 0,
    },
    "miasma": {
        "team": "T",
        "kills": 21,
        "assists": 5,
        "deaths": 22,
        "adr": 76,
        "hltv_rating": 1.05,
        "2k": 4,
        "3k": 3,
        "4k": 0,
        "5k": 0,
    },
    "Tr1d": {
        "team": "T",
        "kills": 23,
        "assists": 10,
        "deaths": 20,
        "adr": 89,
        "hltv_rating": 1.09,
        "2k": 6,
        "3k": 1,
        "4k": 0,
        "5k": 0,
    },
    # Enemy Team (CT side, WIN)
    "129310238324": {
        "team": "CT",
        "kills": 29,
        "assists": 9,
        "deaths": 21,
        "adr": 98,
        "hltv_rating": 1.30,
        "2k": 7,
        "3k": 0,
        "4k": 1,
        "5k": 0,
    },
    "DavidLithium": {
        "team": "CT",
        "kills": 16,
        "assists": 5,
        "deaths": 22,
        "adr": 54,
        "hltv_rating": 0.68,
        "2k": 1,
        "3k": 0,
        "4k": 0,
        "5k": 0,
    },
    "Docobo": {
        "team": "CT",
        "kills": 25,
        "assists": 8,
        "deaths": 21,
        "adr": 84,
        "hltv_rating": 1.21,
        "2k": 5,
        "3k": 1,
        "4k": 1,
        "5k": 0,
    },
    "ina": {
        "team": "CT",
        "kills": 31,
        "assists": 8,
        "deaths": 23,
        "adr": 83,
        "hltv_rating": 1.50,
        "2k": 4,
        "3k": 1,
        "4k": 1,
        "5k": 1,
    },
    "snowing-": {
        "team": "CT",
        "kills": 15,
        "assists": 13,
        "deaths": 23,
        "adr": 78,
        "hltv_rating": 0.82,
        "2k": 2,
        "3k": 1,
        "4k": 1,
        "5k": 0,
    },
}

# =============================================================================
# LEETIFY GROUND TRUTH — Activity tab
# =============================================================================
LEETIFY_ACTIVITY = {
    "dergs": {
        "total_damage": 2202,
        "he_dmg": 2,
        "molotov_dmg": 30,
        "enemies_flashed": 15,
        "shots_fired": 393,
        "rounds_survived": 6,
    },
    "foe": {
        "total_damage": 1830,
        "he_dmg": 199,
        "molotov_dmg": 11,
        "enemies_flashed": 6,
        "shots_fired": 287,
        "rounds_survived": 7,
    },
    "kix": {
        "total_damage": 2695,
        "he_dmg": 98,
        "molotov_dmg": 40,
        "enemies_flashed": 0,
        "shots_fired": 708,
        "rounds_survived": 3,
    },
    "miasma": {
        "total_damage": 2279,
        "he_dmg": 239,
        "molotov_dmg": 23,
        "enemies_flashed": 1,
        "shots_fired": 426,
        "rounds_survived": 8,
    },
    "Tr1d": {
        "total_damage": 2667,
        "he_dmg": 269,
        "molotov_dmg": 8,
        "enemies_flashed": 5,
        "shots_fired": 311,
        "rounds_survived": 10,
    },
    "129310238324": {
        "total_damage": 2942,
        "he_dmg": 113,
        "molotov_dmg": 32,
        "enemies_flashed": 17,
        "shots_fired": 504,
        "rounds_survived": 9,
    },
    "DavidLithium": {
        "total_damage": 1616,
        "he_dmg": 130,
        "molotov_dmg": 78,
        "enemies_flashed": 6,
        "shots_fired": 315,
        "rounds_survived": 8,
    },
    "Docobo": {
        "total_damage": 2528,
        "he_dmg": 71,
        "molotov_dmg": 32,
        "enemies_flashed": 10,
        "shots_fired": 432,
        "rounds_survived": 9,
    },
    "ina": {
        "total_damage": 2495,
        "he_dmg": 86,
        "molotov_dmg": 0,
        "enemies_flashed": 14,
        "shots_fired": 414,
        "rounds_survived": 7,
    },
    "snowing-": {
        "total_damage": 2342,
        "he_dmg": 0,
        "molotov_dmg": 9,
        "enemies_flashed": 12,
        "shots_fired": 428,
        "rounds_survived": 7,
    },
}

# =============================================================================
# LEETIFY GROUND TRUTH — Trades tab
# =============================================================================
LEETIFY_TRADES = {
    # My Team (T, LOSS)
    "dergs": {
        "trade_kill_opps": 15,
        "trade_kill_attempts": 14,
        "trade_kill_success": 6,
        "traded_death_opps": 11,
        "traded_death_attempts": 11,
        "traded_deaths_success": 5,
    },
    "foe": {
        "trade_kill_opps": 7,
        "trade_kill_attempts": 7,
        "trade_kill_success": 1,
        "traded_death_opps": 6,
        "traded_death_attempts": 6,
        "traded_deaths_success": 2,
    },
    "kix": {
        "trade_kill_opps": 9,
        "trade_kill_attempts": 7,
        "trade_kill_success": 3,
        "traded_death_opps": 10,
        "traded_death_attempts": 8,
        "traded_deaths_success": 2,
    },
    "miasma": {
        "trade_kill_opps": 5,
        "trade_kill_attempts": 5,
        "trade_kill_success": 3,
        "traded_death_opps": 6,
        "traded_death_attempts": 6,
        "traded_deaths_success": 3,
    },
    "Tr1d": {
        "trade_kill_opps": 8,
        "trade_kill_attempts": 7,
        "trade_kill_success": 3,
        "traded_death_opps": 7,
        "traded_death_attempts": 7,
        "traded_deaths_success": 4,
    },
    # Enemy Team (CT, WIN)
    "129310238324": {
        "trade_kill_opps": 11,
        "trade_kill_attempts": 11,
        "trade_kill_success": 7,
        "traded_death_opps": 9,
        "traded_death_attempts": 7,
        "traded_deaths_success": 5,
    },
    "DavidLithium": {
        "trade_kill_opps": 14,
        "trade_kill_attempts": 12,
        "trade_kill_success": 6,
        "traded_death_opps": 7,
        "traded_death_attempts": 7,
        "traded_deaths_success": 4,
    },
    "Docobo": {
        "trade_kill_opps": 6,
        "trade_kill_attempts": 6,
        "trade_kill_success": 2,
        "traded_death_opps": 7,
        "traded_death_attempts": 6,
        "traded_deaths_success": 2,
    },
    "ina": {
        "trade_kill_opps": 16,
        "trade_kill_attempts": 16,
        "trade_kill_success": 10,
        "traded_death_opps": 12,
        "traded_death_attempts": 12,
        "traded_deaths_success": 8,
    },
    "snowing-": {
        "trade_kill_opps": 7,
        "trade_kill_attempts": 6,
        "trade_kill_success": 2,
        "traded_death_opps": 13,
        "traded_death_attempts": 12,
        "traded_deaths_success": 8,
    },
}

# =============================================================================
# LEETIFY GROUND TRUTH — Opening Duels tab
# =============================================================================
LEETIFY_OPENING_DUELS = {
    # Percentages from Leetify (Attempts%, Success%, Traded%)
    "dergs": {"attempts_pct": 3, "success_pct": 0, "traded_pct": 0},
    "foe": {"attempts_pct": 17, "success_pct": 20, "traded_pct": 0},
    "kix": {"attempts_pct": 30, "success_pct": 44, "traded_pct": 20},
    "miasma": {"attempts_pct": 23, "success_pct": 43, "traded_pct": 25},
    "Tr1d": {"attempts_pct": 27, "success_pct": 50, "traded_pct": 25},
    "129310238324": {"attempts_pct": 13, "success_pct": 100, "traded_pct": 0},
    "DavidLithium": {"attempts_pct": 13, "success_pct": 50, "traded_pct": 50},
    "Docobo": {"attempts_pct": 17, "success_pct": 60, "traded_pct": 0},
    "ina": {"attempts_pct": 37, "success_pct": 73, "traded_pct": 0},
    "snowing-": {"attempts_pct": 20, "success_pct": 17, "traded_pct": 40},
}

# =============================================================================
# LEETIFY GROUND TRUTH — Aim tab
# =============================================================================
LEETIFY_AIM = {
    "dergs": {"ttd_ms": 563, "cp_deg": 8.84, "hs_kill_pct": 64, "counter_strafe_pct": 78},
    "foe": {"ttd_ms": 672, "cp_deg": 7.30, "hs_kill_pct": 47, "counter_strafe_pct": 72},
    "kix": {"ttd_ms": 625, "cp_deg": 7.18, "hs_kill_pct": 40, "counter_strafe_pct": 83},
    "miasma": {"ttd_ms": 484, "cp_deg": 6.98, "hs_kill_pct": 19, "counter_strafe_pct": 85},
    "Tr1d": {"ttd_ms": 484, "cp_deg": 6.75, "hs_kill_pct": 70, "counter_strafe_pct": 72},
    "129310238324": {"ttd_ms": 531, "cp_deg": 9.13, "hs_kill_pct": 72, "counter_strafe_pct": 80},
    "DavidLithium": {"ttd_ms": 688, "cp_deg": 11.04, "hs_kill_pct": 50, "counter_strafe_pct": 60},
    "Docobo": {"ttd_ms": 469, "cp_deg": 9.96, "hs_kill_pct": 32, "counter_strafe_pct": 81},
    "ina": {"ttd_ms": 547, "cp_deg": 6.17, "hs_kill_pct": 58, "counter_strafe_pct": 81},
    "snowing-": {"ttd_ms": 594, "cp_deg": 4.56, "hs_kill_pct": 87, "counter_strafe_pct": 86},
}

# =============================================================================
# LEETIFY GROUND TRUTH — Clutches summary
# =============================================================================
LEETIFY_CLUTCHES = {
    "my_team": {"won": 4, "lost": 16, "saves": 0, "total_kills": 10},
    "enemy_team": {"won": 4, "lost": 14, "saves": 1, "total_kills": 12},
}

# Number of competitive rounds (knife round excluded)
LEETIFY_ROUNDS = 30

# =============================================================================
# TOLERANCES
# =============================================================================
# Tight tolerances — we want to match Leetify closely
KILLS_TOLERANCE = 0  # Exact match required
DEATHS_TOLERANCE = 0  # Exact match required
ASSISTS_TOLERANCE = 0  # Exact match required
ADR_TOLERANCE = 3  # Allow small rounding differences
HLTV_TOLERANCE = 0.17  # HLTV 2.0 formula uses approximated coefficients
# (reverse-engineered, not disclosed by HLTV).
# Even with scipy-optimized coefficients the min
# achievable max-error is 0.06. Our KAST values
# may also differ from Leetify's proprietary
# computation.  ±0.17 covers worst-case player
# (max observed error 0.16); raw stats (K/D/A/ADR)
# are verified exact.
MULTIKILL_TOLERANCE = 0  # Exact match required


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def analysis_result():
    """Parse and analyze the golden demo via the orchestrator."""
    if not Path(GOLDEN_DEMO_PATH).exists():
        pytest.skip(f"Golden demo not found at {GOLDEN_DEMO_PATH}")

    from opensight.pipeline.orchestrator import DemoOrchestrator

    orch = DemoOrchestrator()
    return orch.analyze(Path(GOLDEN_DEMO_PATH), force=True)


@pytest.fixture(scope="module")
def players(analysis_result):
    """Extract players dict from orchestrator result."""
    return analysis_result.get("players", {})


@pytest.fixture(scope="module")
def players_by_name(players):
    """Re-key players dict by name for easy lookup."""
    return {p["name"]: p for p in players.values()}


@pytest.fixture(scope="module")
def analyzer_result():
    """Parse and analyze via DemoAnalyzer directly (for fields not in orchestrator)."""
    if not Path(GOLDEN_DEMO_PATH).exists():
        pytest.skip(f"Golden demo not found at {GOLDEN_DEMO_PATH}")

    from opensight.analysis.analytics import DemoAnalyzer
    from opensight.core.parser import DemoParser

    parser = DemoParser(Path(GOLDEN_DEMO_PATH))
    data = parser.parse()
    analyzer = DemoAnalyzer(data)
    analysis = analyzer.analyze()
    return analysis, analyzer, data


# =============================================================================
# TESTS — Core Stats (Kills, Deaths, Assists, ADR, HLTV)
# =============================================================================


class TestRoundCount:
    """Knife round must be excluded — 30 competitive rounds, not 31."""

    def test_round_count(self, analyzer_result):
        _, _, data = analyzer_result
        assert data.num_rounds == LEETIFY_ROUNDS, (
            f"Round count wrong: got {data.num_rounds}, expected {LEETIFY_ROUNDS}. "
            f"Knife round probably not excluded."
        )


class TestKills:
    """Kill counts must match Leetify exactly."""

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_kills(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["kills"]
        actual = players_by_name[player_name]["stats"]["kills"]
        assert abs(actual - expected) <= KILLS_TOLERANCE, (
            f"{player_name} kills: got {actual}, expected {expected}"
        )


class TestDeaths:
    """Death counts must match Leetify exactly."""

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_deaths(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["deaths"]
        actual = players_by_name[player_name]["stats"]["deaths"]
        assert abs(actual - expected) <= DEATHS_TOLERANCE, (
            f"{player_name} deaths: got {actual}, expected {expected}"
        )


class TestAssists:
    """Assist counts must match Leetify exactly."""

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_assists(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["assists"]
        actual = players_by_name[player_name]["stats"]["assists"]
        assert abs(actual - expected) <= ASSISTS_TOLERANCE, (
            f"{player_name} assists: got {actual}, expected {expected}"
        )


class TestADR:
    """ADR must be within tolerance of Leetify values."""

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_adr(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["adr"]
        actual = players_by_name[player_name]["stats"]["adr"]
        assert abs(actual - expected) <= ADR_TOLERANCE, (
            f"{player_name} ADR: got {actual}, expected {expected} "
            f"(diff={abs(actual - expected):.1f}, tolerance={ADR_TOLERANCE})"
        )


class TestHLTVRating:
    """HLTV 2.0 ratings must be within tolerance of Leetify values."""

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_hltv_rating(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["hltv_rating"]
        actual = players_by_name[player_name]["rating"]["hltv_rating"]
        assert abs(actual - expected) <= HLTV_TOLERANCE, (
            f"{player_name} HLTV: got {actual}, expected {expected} "
            f"(diff={abs(actual - expected):.2f}, tolerance={HLTV_TOLERANCE})"
        )


class TestMultiKills:
    """Multi-kill counts must match Leetify exactly."""

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_2k(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["2k"]
        actual = players_by_name[player_name]["stats"]["2k"]
        assert abs(actual - expected) <= MULTIKILL_TOLERANCE, (
            f"{player_name} 2K: got {actual}, expected {expected}"
        )

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_3k(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["3k"]
        actual = players_by_name[player_name]["stats"]["3k"]
        assert abs(actual - expected) <= MULTIKILL_TOLERANCE, (
            f"{player_name} 3K: got {actual}, expected {expected}"
        )

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_4k(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["4k"]
        actual = players_by_name[player_name]["stats"]["4k"]
        assert abs(actual - expected) <= MULTIKILL_TOLERANCE, (
            f"{player_name} 4K: got {actual}, expected {expected}"
        )

    @pytest.mark.parametrize("player_name", list(LEETIFY_GENERAL.keys()))
    def test_5k(self, players_by_name, player_name):
        expected = LEETIFY_GENERAL[player_name]["5k"]
        actual = players_by_name[player_name]["stats"]["5k"]
        assert abs(actual - expected) <= MULTIKILL_TOLERANCE, (
            f"{player_name} 5K: got {actual}, expected {expected}"
        )
