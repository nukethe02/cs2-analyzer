"""Tests for TTD and CP pipeline connectivity.

Verifies that TTD (Time to Damage) and CP (Crosshair Placement) values
flow correctly from parser data through compute functions to model fields.

These tests focus on PIPELINE CONNECTIVITY, not unit-testing the math.
They verify:
1. Data flows from DemoData through DemoAnalyzer to PlayerMatchStats
2. Both optimized and fallback paths produce non-null results
3. Cross-round damage does not contaminate TTD values
4. Large Steam IDs (17 digits) survive the pipeline without precision loss
"""

from pathlib import Path

import pandas as pd
import pytest

from opensight.analysis.analytics import DemoAnalyzer
from opensight.analysis.metrics_optimized import compute_ttd_vectorized
from opensight.core.parser import DemoData, KillEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use realistic 17-digit Steam IDs to catch float precision bugs
P1_ID = 76561198073476793
P2_ID = 76561198154302841


def _make_demo_data(
    kills: list[KillEvent],
    kills_df: pd.DataFrame,
    damages_df: pd.DataFrame,
    num_rounds: int = 3,
) -> DemoData:
    """Build a minimal DemoData for testing aim pipeline."""
    return DemoData(
        file_path=Path("/tmp/test_aim.dem"),
        map_name="de_dust2",
        duration_seconds=1800.0,
        tick_rate=64,
        num_rounds=num_rounds,
        player_stats={
            P1_ID: {"name": "Player1", "team": "CT", "kills": 0, "deaths": 0},
            P2_ID: {"name": "Player2", "team": "T", "kills": 0, "deaths": 0},
        },
        player_names={P1_ID: "Player1", P2_ID: "Player2"},
        player_teams={P1_ID: 3, P2_ID: 2},
        kills=kills,
        damages=[],
        kills_df=kills_df,
        damages_df=damages_df,
        ticks_df=None,
    )


@pytest.fixture
def single_round_data():
    """Data where all kills and damages happen in one round (simplest case)."""
    kills = [
        KillEvent(
            tick=1000,
            round_num=1,
            attacker_steamid=P1_ID,
            attacker_name="Player1",
            attacker_side="CT",
            victim_steamid=P2_ID,
            victim_name="Player2",
            victim_side="T",
            weapon="ak47",
            headshot=True,
            attacker_x=100.0,
            attacker_y=200.0,
            attacker_z=0.0,
            attacker_pitch=0.0,
            attacker_yaw=0.0,
            victim_x=400.0,
            victim_y=200.0,
            victim_z=0.0,
        ),
    ]
    kills_df = pd.DataFrame(
        {
            "tick": [1000],
            "attacker_steamid": [P1_ID],
            "user_steamid": [P2_ID],
            "attacker_name": ["Player1"],
            "user_name": ["Player2"],
            "weapon": ["ak47"],
            "headshot": [True],
            "total_rounds_played": [1],
            "attacker_X": [100.0],
            "attacker_Y": [200.0],
            "attacker_Z": [0.0],
            "attacker_pitch": [0.0],
            "attacker_yaw": [0.0],
            "user_X": [400.0],
            "user_Y": [200.0],
            "user_Z": [0.0],
        }
    )
    # Damage 20 ticks before the kill
    damages_df = pd.DataFrame(
        {
            "tick": [980, 1000],
            "attacker_steamid": [P1_ID, P1_ID],
            "user_steamid": [P2_ID, P2_ID],
            "dmg_health": [27, 73],
            "weapon": ["ak47", "ak47"],
            "hitgroup": [2, 1],
        }
    )
    return _make_demo_data(kills, kills_df, damages_df, num_rounds=1)


@pytest.fixture
def cross_round_data():
    """Data where P1 damages P2 in round 1 and kills P2 in round 3.

    The vectorized TTD must NOT use round 1 damage for the round 3 kill.
    """
    kills = [
        KillEvent(
            tick=9000,
            round_num=3,
            attacker_steamid=P1_ID,
            attacker_name="Player1",
            attacker_side="CT",
            victim_steamid=P2_ID,
            victim_name="Player2",
            victim_side="T",
            weapon="ak47",
            headshot=True,
            attacker_x=200.0,
            attacker_y=400.0,
            attacker_z=0.0,
            attacker_pitch=2.0,
            attacker_yaw=0.0,
            victim_x=500.0,
            victim_y=400.0,
            victim_z=0.0,
        ),
    ]
    kills_df = pd.DataFrame(
        {
            "tick": [9000],
            "attacker_steamid": [P1_ID],
            "user_steamid": [P2_ID],
            "attacker_name": ["Player1"],
            "user_name": ["Player2"],
            "weapon": ["ak47"],
            "headshot": [True],
            "total_rounds_played": [3],
            "attacker_X": [200.0],
            "attacker_Y": [400.0],
            "attacker_Z": [0.0],
            "attacker_pitch": [2.0],
            "attacker_yaw": [0.0],
            "user_X": [500.0],
            "user_Y": [400.0],
            "user_Z": [0.0],
        }
    )
    # Round 1 damage (tick 500) is far away from the round 3 kill (tick 9000)
    # Round 3 damage (tick 8980) is the real engagement start
    damages_df = pd.DataFrame(
        {
            "tick": [500, 8980, 9000],
            "attacker_steamid": [P1_ID, P1_ID, P1_ID],
            "user_steamid": [P2_ID, P2_ID, P2_ID],
            "dmg_health": [27, 27, 73],
            "weapon": ["ak47", "ak47", "ak47"],
            "hitgroup": [2, 2, 1],
        }
    )
    return _make_demo_data(kills, kills_df, damages_df, num_rounds=3)


# ---------------------------------------------------------------------------
# TTD Pipeline Tests
# ---------------------------------------------------------------------------


class TestTTDPipeline:
    """Verify TTD flows from parser data to PlayerMatchStats."""

    def test_ttd_populated_fallback(self, single_round_data):
        """Non-optimized (fallback) path produces non-null TTD."""
        analyzer = DemoAnalyzer(single_round_data, use_optimized=False)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        assert player.engagement_duration_values, "engagement_duration_values should not be empty"
        assert player.ttd_median_ms is not None, "ttd_median_ms should not be None"
        assert player.ttd_median_ms > 0, "ttd_median_ms should be positive"

    def test_ttd_populated_optimized(self, single_round_data):
        """Optimized (vectorized) path produces non-null TTD."""
        analyzer = DemoAnalyzer(single_round_data, use_optimized=True)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        assert player.engagement_duration_values, "engagement_duration_values should not be empty"
        assert player.ttd_median_ms is not None, "ttd_median_ms should not be None"
        assert player.ttd_median_ms > 0, "ttd_median_ms should be positive"

    def test_ttd_optimized_matches_fallback(self, single_round_data):
        """Optimized and fallback paths produce similar TTD values."""
        analyzer_fb = DemoAnalyzer(single_round_data, use_optimized=False)
        result_fb = analyzer_fb.analyze()

        analyzer_opt = DemoAnalyzer(single_round_data, use_optimized=True)
        result_opt = analyzer_opt.analyze()

        ttd_fb = result_fb.players[P1_ID].ttd_median_ms
        ttd_opt = result_opt.players[P1_ID].ttd_median_ms

        assert ttd_fb is not None, "Fallback TTD should not be None"
        assert ttd_opt is not None, "Optimized TTD should not be None"
        # Both should be close (they may differ slightly due to different windowing)
        assert abs(ttd_fb - ttd_opt) < 100, f"TTD mismatch: fallback={ttd_fb}, optimized={ttd_opt}"

    def test_ttd_cross_round_not_contaminated(self, cross_round_data):
        """Cross-round damage does not inflate TTD values."""
        analyzer = DemoAnalyzer(cross_round_data, use_optimized=True)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        assert player.ttd_median_ms is not None, (
            "TTD should not be None even with cross-round damage"
        )
        # If cross-round contamination occurs, TTD would be ~132812ms (tick 500 to 9000).
        # The correct TTD is ~312ms (tick 8980 to 9000 = 20 ticks * 15.625ms).
        assert player.ttd_median_ms < 5000, (
            f"TTD={player.ttd_median_ms}ms is too large - cross-round contamination?"
        )

    def test_ttd_cross_round_fallback(self, cross_round_data):
        """Cross-round damage handled correctly in fallback path too."""
        analyzer = DemoAnalyzer(cross_round_data, use_optimized=False)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        assert player.ttd_median_ms is not None
        assert player.ttd_median_ms < 5000

    def test_ttd_steamid_precision_preserved(self):
        """17-digit Steam IDs survive float conversion without precision loss."""
        # This test catches the bug where int->float->int loses digits:
        # int(float(76561198073476793)) == 76561198073476800 (WRONG)
        kills_df = pd.DataFrame(
            {
                "tick": [1000],
                "attacker_steamid": [P1_ID],
                "user_steamid": [P2_ID],
            }
        )
        damages_df = pd.DataFrame(
            {
                "tick": [980],
                "attacker_steamid": [P1_ID],
                "user_steamid": [P2_ID],
                "dmg_health": [100],
            }
        )
        result = compute_ttd_vectorized(kills_df, damages_df, tick_rate=64)

        # The key in player_ttd_values must be the EXACT original steamid
        assert P1_ID in result.player_ttd_values, (
            f"Steam ID {P1_ID} not found in TTD results. "
            f"Keys: {list(result.player_ttd_values.keys())}. "
            f"Likely float precision loss."
        )


# ---------------------------------------------------------------------------
# CP Pipeline Tests
# ---------------------------------------------------------------------------


class TestCPPipeline:
    """Verify CP flows from parser data to PlayerMatchStats."""

    def test_cp_populated_fallback(self, single_round_data):
        """Non-optimized (fallback) path produces non-null CP."""
        analyzer = DemoAnalyzer(single_round_data, use_optimized=False)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        assert player.cp_values, "cp_values should not be empty"
        assert player.cp_median_error_deg is not None, "cp_median_error_deg should not be None"

    def test_cp_populated_optimized(self, single_round_data):
        """Optimized (vectorized) path produces non-null CP."""
        analyzer = DemoAnalyzer(single_round_data, use_optimized=True)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        assert player.cp_values, "cp_values should not be empty"
        assert player.cp_median_error_deg is not None, "cp_median_error_deg should not be None"

    def test_cp_optimized_matches_fallback(self, single_round_data):
        """Optimized and fallback paths produce similar CP values."""
        analyzer_fb = DemoAnalyzer(single_round_data, use_optimized=False)
        result_fb = analyzer_fb.analyze()

        analyzer_opt = DemoAnalyzer(single_round_data, use_optimized=True)
        result_opt = analyzer_opt.analyze()

        cp_fb = result_fb.players[P1_ID].cp_median_error_deg
        cp_opt = result_opt.players[P1_ID].cp_median_error_deg

        assert cp_fb is not None, "Fallback CP should not be None"
        assert cp_opt is not None, "Optimized CP should not be None"
        assert abs(cp_fb - cp_opt) < 1.0, f"CP mismatch: fallback={cp_fb}, optimized={cp_opt}"

    def test_cp_reasonable_range(self, single_round_data):
        """CP values should be in a reasonable range (0-180 degrees)."""
        analyzer = DemoAnalyzer(single_round_data, use_optimized=True)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        for cp_val in player.cp_values:
            assert 0 <= cp_val <= 180, f"CP value {cp_val} out of range [0, 180]"

    def test_cp_no_position_data_returns_none(self):
        """CP is None when kill events lack position data."""
        kills = [
            KillEvent(
                tick=1000,
                round_num=1,
                attacker_steamid=P1_ID,
                attacker_name="Player1",
                attacker_side="CT",
                victim_steamid=P2_ID,
                victim_name="Player2",
                victim_side="T",
                weapon="ak47",
                headshot=True,
                # No position data
            ),
        ]
        kills_df = pd.DataFrame(
            {
                "tick": [1000],
                "attacker_steamid": [P1_ID],
                "user_steamid": [P2_ID],
                "weapon": ["ak47"],
                "headshot": [True],
            }
        )
        damages_df = pd.DataFrame(
            {
                "tick": [980],
                "attacker_steamid": [P1_ID],
                "user_steamid": [P2_ID],
                "dmg_health": [100],
            }
        )
        data = _make_demo_data(kills, kills_df, damages_df, num_rounds=1)
        analyzer = DemoAnalyzer(data, use_optimized=False)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        # No position data means CP can't be computed
        assert player.cp_median_error_deg is None


# ---------------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------------


class TestSerializationPipeline:
    """Verify TTD/CP values survive serialization to dict."""

    def test_aim_stats_has_ttd_and_cp(self):
        """AimStats.to_dict() includes non-null TTD and CP."""
        # Use a kill where the attacker is NOT looking directly at the victim
        # so CP > 0 (avoids the "if x" vs "if x is not None" bug in AimStats.to_dict)
        kills = [
            KillEvent(
                tick=1000,
                round_num=1,
                attacker_steamid=P1_ID,
                attacker_name="Player1",
                attacker_side="CT",
                victim_steamid=P2_ID,
                victim_name="Player2",
                victim_side="T",
                weapon="ak47",
                headshot=True,
                attacker_x=100.0,
                attacker_y=200.0,
                attacker_z=0.0,
                attacker_pitch=0.0,
                attacker_yaw=45.0,  # Looking 45 degrees off from target
                victim_x=400.0,
                victim_y=200.0,
                victim_z=0.0,
            ),
        ]
        kills_df = pd.DataFrame(
            {
                "tick": [1000],
                "attacker_steamid": [P1_ID],
                "user_steamid": [P2_ID],
                "attacker_name": ["Player1"],
                "user_name": ["Player2"],
                "weapon": ["ak47"],
                "headshot": [True],
                "total_rounds_played": [1],
                "attacker_X": [100.0],
                "attacker_Y": [200.0],
                "attacker_Z": [0.0],
                "attacker_pitch": [0.0],
                "attacker_yaw": [45.0],
                "user_X": [400.0],
                "user_Y": [200.0],
                "user_Z": [0.0],
            }
        )
        damages_df = pd.DataFrame(
            {
                "tick": [980, 1000],
                "attacker_steamid": [P1_ID, P1_ID],
                "user_steamid": [P2_ID, P2_ID],
                "dmg_health": [27, 73],
                "weapon": ["ak47", "ak47"],
                "hitgroup": [2, 1],
            }
        )
        data = _make_demo_data(kills, kills_df, damages_df, num_rounds=1)
        analyzer = DemoAnalyzer(data, use_optimized=True)
        result = analyzer.analyze()

        player = result.players[P1_ID]
        aim_dict = player.aim_stats.to_dict()

        assert aim_dict["time_to_damage_ms"] is not None, "TTD should be in aim_stats dict"
        assert aim_dict["time_to_damage_ms"] > 0, "TTD should be positive"
        assert aim_dict["crosshair_placement_deg"] is not None, "CP should be in aim_stats dict"
        assert aim_dict["crosshair_placement_deg"] > 0, "CP should be positive (non-zero angle)"
        assert aim_dict["ttd_rating"] != "unknown", "TTD rating should not be unknown"
