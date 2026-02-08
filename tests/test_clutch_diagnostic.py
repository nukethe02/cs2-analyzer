"""Tests for clutch detection via compute_combat.detect_clutches.

Verifies the full production path (DemoAnalyzer → compute_combat.detect_clutches)
with realistic demoparser2-style DataFrame structures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opensight.analysis.analytics import DemoAnalyzer
from opensight.core.parser import DemoData, RoundInfo


def make_clutch_demo_data():
    """Create DemoData mimicking real demoparser2 output with a clear clutch scenario.

    Round 1: 4 CT deaths, then last CT (player 5) kills 2 Ts → 1v5 clutch (WON).
    Round 2: Normal round, no clutch.
    """
    ct_ids = [
        76561198000000001,
        76561198000000002,
        76561198000000003,
        76561198000000004,
        76561198000000005,
    ]
    t_ids = [
        76561198000000006,
        76561198000000007,
        76561198000000008,
        76561198000000009,
        76561198000000010,
    ]

    kills_df = pd.DataFrame(
        {
            "tick": [1000, 1100, 1200, 1300, 1400, 1500, 3000],
            "total_rounds_played": [1, 1, 1, 1, 1, 1, 2],
            "attacker_steamid": [
                t_ids[0],
                t_ids[1],
                t_ids[2],
                t_ids[3],
                ct_ids[4],
                ct_ids[4],
                t_ids[4],
            ],
            "user_steamid": [
                ct_ids[0],
                ct_ids[1],
                ct_ids[2],
                ct_ids[3],
                t_ids[0],
                t_ids[1],
                ct_ids[4],
            ],
            "attacker_name": [
                "T1",
                "T2",
                "T3",
                "T4",
                "CT5",
                "CT5",
                "T5",
            ],
            "user_name": ["CT1", "CT2", "CT3", "CT4", "T1", "T2", "CT5"],
            "attacker_team_num": [2, 2, 2, 2, 3, 3, 2],
            "user_team_num": [3, 3, 3, 3, 2, 2, 3],
            "weapon": ["ak47", "ak47", "ak47", "ak47", "m4a1", "m4a1", "ak47"],
            "headshot": [True, False, True, False, True, True, False],
        }
    )

    damages_df = pd.DataFrame(
        {
            "tick": [990, 1090, 1190, 1290, 1390, 1490, 2990],
            "attacker_steamid": [
                t_ids[0],
                t_ids[1],
                t_ids[2],
                t_ids[3],
                ct_ids[4],
                ct_ids[4],
                t_ids[4],
            ],
            "user_steamid": [
                ct_ids[0],
                ct_ids[1],
                ct_ids[2],
                ct_ids[3],
                t_ids[0],
                t_ids[1],
                ct_ids[4],
            ],
            "dmg_health": [100, 100, 100, 100, 100, 100, 100],
            "weapon": ["ak47", "ak47", "ak47", "ak47", "m4a1", "m4a1", "ak47"],
        }
    )

    player_names = {}
    player_teams = {}
    for i, sid in enumerate(ct_ids):
        player_names[sid] = f"CT{i + 1}"
        player_teams[sid] = 3
    for i, sid in enumerate(t_ids):
        player_names[sid] = f"T{i + 1}"
        player_teams[sid] = 2

    player_persistent_teams = dict.fromkeys(ct_ids, "Team A")
    player_persistent_teams.update(dict.fromkeys(t_ids, "Team B"))

    rounds = [
        RoundInfo(
            round_num=1,
            winner="CT",
            start_tick=0,
            end_tick=2000,
            freeze_end_tick=500,
            reason="ct_win",
        ),
        RoundInfo(
            round_num=2,
            winner="T",
            start_tick=2500,
            end_tick=4000,
            freeze_end_tick=2800,
            reason="t_win",
        ),
    ]

    return DemoData(
        file_path=Path("/tmp/test_clutch.dem"),
        map_name="de_dust2",
        duration_seconds=120.0,
        tick_rate=64,
        num_rounds=2,
        player_stats={},
        player_names=player_names,
        player_teams=player_teams,
        player_persistent_teams=player_persistent_teams,
        team_rosters={"Team A": set(ct_ids), "Team B": set(t_ids)},
        team_starting_sides={"Team A": "CT", "Team B": "T"},
        halftime_round=13,
        kills=[],
        damages=[],
        kills_df=kills_df,
        damages_df=damages_df,
        ticks_df=None,
        rounds=rounds,
    )


class TestClutchDetection:
    """Tests for clutch detection via the production compute_combat path."""

    def test_detects_clutch_situation(self):
        """detect_clutches finds a clutch when 4 teammates die."""
        data = make_clutch_demo_data()
        analyzer = DemoAnalyzer(data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        from opensight.analysis.compute_combat import detect_clutches

        detect_clutches(analyzer)

        clutcher = analyzer._players[76561198000000005]
        assert clutcher.clutches.total_situations == 1
        assert clutcher.clutches.total_wins == 1
        assert len(clutcher.clutches.clutches) == 1
        assert clutcher.clutches.clutches[0].outcome == "WON"

    def test_clutch_via_full_analyze_pipeline(self):
        """Clutch detection works through the full analyze() pipeline."""
        data = make_clutch_demo_data()
        result = DemoAnalyzer(data).analyze()

        clutcher = result.players[76561198000000005]
        assert clutcher.clutches.total_situations >= 1

    def test_non_clutch_players_have_zero(self):
        """Players who aren't in clutch situations have 0 clutch stats."""
        data = make_clutch_demo_data()
        result = DemoAnalyzer(data).analyze()

        # CT1 died early — not a clutcher
        assert result.players[76561198000000001].clutches.total_situations == 0


class TestNormalizeTeamNaN:
    """Tests for _normalize_team handling of NaN values (root cause of clutch bug)."""

    @pytest.fixture
    def analyzer(self):
        return DemoAnalyzer(make_clutch_demo_data())

    def test_handles_python_nan(self, analyzer):
        assert analyzer._normalize_team(float("nan")) == "Unknown"

    def test_handles_numpy_nan(self, analyzer):
        assert analyzer._normalize_team(np.nan) == "Unknown"

    def test_handles_numpy_float64_nan(self, analyzer):
        assert analyzer._normalize_team(np.float64(np.nan)) == "Unknown"

    def test_handles_none(self, analyzer):
        assert analyzer._normalize_team(None) == "Unknown"

    def test_normal_values_still_work(self, analyzer):
        assert analyzer._normalize_team(3) == "CT"
        assert analyzer._normalize_team(2) == "T"
        assert analyzer._normalize_team(3.0) == "CT"
        assert analyzer._normalize_team("CT") == "CT"
        assert analyzer._normalize_team("TERRORIST") == "T"

    def test_clutch_survives_nan_in_team_column(self):
        """Clutch detection doesn't crash when kill events have NaN team values."""
        data = make_clutch_demo_data()
        data.kills_df.loc[0, "user_team_num"] = np.nan

        analyzer = DemoAnalyzer(data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        from opensight.analysis.compute_combat import detect_clutches

        # Must not crash
        detect_clutches(analyzer)

        # Clutch should still be detected (NaN kill is processed via alive-set fallback)
        clutcher = analyzer._players[76561198000000005]
        assert clutcher.clutches.total_situations == 1
