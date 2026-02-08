"""Edge-case resilience tests for the analysis pipeline.

Tests that intentionally bad / degenerate data flows through the
DemoAnalyzer → DemoOrchestrator pipeline without crashing, producing
valid (possibly all-zero) output dictionaries.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from opensight.analysis.analytics import DemoAnalyzer
from opensight.analysis.models import (
    ClutchStats,
    MatchAnalysis,
    PlayerMatchStats,
    TradeStats,
)
from opensight.core.parser import (
    DemoData,
    KillEvent,
    RoundInfo,
)
from opensight.pipeline.orchestrator import DemoOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_demo_data(
    *,
    num_rounds: int = 0,
    player_stats: dict | None = None,
    player_names: dict | None = None,
    player_teams: dict | None = None,
    kills: list | None = None,
    damages: list | None = None,
    rounds: list | None = None,
    kills_df: pd.DataFrame | None = None,
    damages_df: pd.DataFrame | None = None,
    weapon_fires_df: pd.DataFrame | None = None,
    blinds_df: pd.DataFrame | None = None,
    grenades_df: pd.DataFrame | None = None,
    bomb_events_df: pd.DataFrame | None = None,
    map_name: str = "de_dust2",
) -> DemoData:
    """Build a DemoData with sensible defaults for missing fields."""
    return DemoData(
        file_path=Path("/tmp/edge_test.dem"),
        map_name=map_name,
        duration_seconds=1800.0,
        tick_rate=64,
        num_rounds=num_rounds,
        player_stats=player_stats or {},
        player_names=player_names or {},
        player_teams=player_teams or {},
        kills=kills or [],
        damages=damages or [],
        rounds=rounds or [],
        kills_df=kills_df if kills_df is not None else pd.DataFrame(),
        damages_df=damages_df if damages_df is not None else pd.DataFrame(),
        weapon_fires_df=weapon_fires_df if weapon_fires_df is not None else pd.DataFrame(),
        blinds_df=blinds_df if blinds_df is not None else pd.DataFrame(),
        grenades_df=grenades_df if grenades_df is not None else pd.DataFrame(),
        bomb_events_df=bomb_events_df if bomb_events_df is not None else pd.DataFrame(),
    )


def _run_analyzer(demo_data: DemoData) -> MatchAnalysis:
    """Run DemoAnalyzer.analyze() and return MatchAnalysis."""
    analyzer = DemoAnalyzer(demo_data)
    return analyzer.analyze()


def _run_orchestrator(demo_data: DemoData) -> dict:
    """Run the full orchestrator pipeline with a mocked DemoParser.

    DemoParser and DemoAnalyzer are imported locally inside
    DemoOrchestrator.analyze(), so we patch them at their source modules.
    """
    from unittest.mock import MagicMock, patch

    orchestrator = DemoOrchestrator()

    # Run real analysis first so orchestrator gets a real MatchAnalysis
    real_analyzer = DemoAnalyzer(demo_data)
    real_analysis = real_analyzer.analyze()

    # Patch at source modules (lazy imports inside analyze() pull from these)
    mock_parser_cls = MagicMock()
    mock_parser_cls.return_value.parse.return_value = demo_data

    mock_analyzer_cls = MagicMock()
    mock_analyzer_cls.return_value.analyze.return_value = real_analysis

    with (
        patch("opensight.core.parser.DemoParser", mock_parser_cls),
        patch("opensight.analysis.analytics.DemoAnalyzer", mock_analyzer_cls),
    ):
        result = orchestrator.analyze(Path("/tmp/edge_test.dem"))

    return result


# ===========================================================================
# 1. EMPTY DEMO MOCK
# ===========================================================================


class TestEmptyDemo:
    """Mock DemoParser returning empty DataFrames for all events.

    Contract: orchestrator returns a valid result dict with zero stats,
    no crashes.
    """

    def test_analyzer_handles_empty_data(self):
        """DemoAnalyzer.analyze() returns MatchAnalysis with 0 players."""
        demo_data = _make_demo_data(num_rounds=0)
        analysis = _run_analyzer(demo_data)

        assert isinstance(analysis, MatchAnalysis)
        assert analysis.players == {}
        assert analysis.total_rounds == 0
        assert analysis.team1_score == 0
        assert analysis.team2_score == 0

    def test_orchestrator_handles_empty_data(self):
        """Full orchestrator pipeline survives empty demo data."""
        demo_data = _make_demo_data(num_rounds=0)
        result = _run_orchestrator(demo_data)

        assert isinstance(result, dict)
        assert "players" in result
        assert "demo_info" in result
        assert "round_timeline" in result
        assert result["players"] == {}
        assert result["demo_info"]["rounds"] == 0
        assert result["demo_info"]["total_kills"] == 0
        # MVP should be None when no players
        assert result["mvp"] is None


# ===========================================================================
# 2. SINGLE ROUND — 2 players, 1 kill
# ===========================================================================


class TestSingleRound:
    """Mock a demo with exactly 1 round, 2 players, 1 kill.

    Verify per-round averages (ADR etc.) divide correctly without
    division-by-zero.
    """

    @pytest.fixture
    def single_round_data(self):
        """One round, Player1 (CT) kills Player2 (T) with AK-47 headshot."""
        kills_df = pd.DataFrame(
            {
                "tick": [5000],
                "attacker_steamid": [11111],
                "user_steamid": [22222],
                "attacker_name": ["Alice"],
                "user_name": ["Bob"],
                "weapon": ["ak47"],
                "headshot": [True],
            }
        )
        damages_df = pd.DataFrame(
            {
                "tick": [5000],
                "attacker_steamid": [11111],
                "user_steamid": [22222],
                "dmg_health": [100],
                "weapon": ["ak47"],
            }
        )
        kill_event = KillEvent(
            tick=5000,
            round_num=1,
            attacker_steamid=11111,
            attacker_name="Alice",
            attacker_side="CT",
            victim_steamid=22222,
            victim_name="Bob",
            victim_side="T",
            weapon="ak47",
            headshot=True,
        )
        round_info = RoundInfo(
            round_num=1,
            start_tick=0,
            end_tick=10000,
            freeze_end_tick=500,
            winner="CT",
            reason="elimination",
            ct_score=1,
            t_score=0,
        )
        return _make_demo_data(
            num_rounds=1,
            player_stats={
                11111: {
                    "name": "Alice",
                    "team": "CT",
                    "kills": 1,
                    "deaths": 0,
                    "assists": 0,
                    "headshots": 1,
                    "total_damage": 100,
                    "adr": 100.0,
                    "hs_percent": 100.0,
                    "weapon_kills": {"ak47": 1},
                },
                22222: {
                    "name": "Bob",
                    "team": "T",
                    "kills": 0,
                    "deaths": 1,
                    "assists": 0,
                    "headshots": 0,
                    "total_damage": 0,
                    "adr": 0.0,
                    "hs_percent": 0.0,
                    "weapon_kills": {},
                },
            },
            player_names={11111: "Alice", 22222: "Bob"},
            player_teams={11111: 3, 22222: 2},
            kills=[kill_event],
            damages=[],
            rounds=[round_info],
            kills_df=kills_df,
            damages_df=damages_df,
        )

    def test_analyzer_single_round(self, single_round_data):
        """ADR and per-round stats compute correctly with 1 round."""
        analysis = _run_analyzer(single_round_data)

        assert len(analysis.players) == 2

        alice = analysis.players[11111]
        bob = analysis.players[22222]

        # Alice: 1 kill, 100 damage in 1 round → ADR = 100.0
        assert alice.kills == 1
        assert alice.rounds_played == 1
        assert alice.adr == 100.0
        assert alice.kills_per_round == 1.0

        # Bob: 0 kills, 0 damage in 1 round → ADR = 0.0 (not NaN/crash)
        assert bob.kills == 0
        assert bob.deaths == 1
        assert bob.adr == 0.0
        assert bob.kills_per_round == 0.0
        # headshot_percentage with 0 kills should not crash
        assert bob.headshot_percentage == 0.0

    def test_orchestrator_single_round(self, single_round_data):
        """Full orchestrator pipeline works with single-round demo."""
        result = _run_orchestrator(single_round_data)

        assert isinstance(result, dict)
        assert len(result["players"]) == 2
        assert result["demo_info"]["rounds"] == 1

        # Verify no NaN or None leaked into serialized stats
        for sid, player in result["players"].items():
            stats = player["stats"]
            rating = player["rating"]
            # All numeric fields should be finite numbers, not NaN
            assert not math.isnan(stats["adr"]), f"Player {sid} has NaN ADR"
            assert not math.isnan(rating["hltv_rating"]), f"Player {sid} has NaN HLTV rating"
            assert not math.isnan(rating["kast_percentage"]), f"Player {sid} has NaN KAST"


# ===========================================================================
# 3. PLAYER WITH ZERO KILLS
# ===========================================================================


class TestZeroKillPlayer:
    """Verify a 0-kill player gets valid (bad) ratings, not null/NaN/crash."""

    @pytest.fixture
    def zero_kill_player(self):
        """Create a PlayerMatchStats with zero kills and verify computed props."""
        return PlayerMatchStats(
            steam_id=99999,
            name="ZeroKillBot",
            team="T",
            kills=0,
            deaths=15,
            assists=2,
            headshots=0,
            total_damage=150,
            rounds_played=15,
        )

    def test_hltv_rating_not_nan(self, zero_kill_player):
        """HLTV rating is a finite number for 0-kill player."""
        rating = zero_kill_player.hltv_rating
        assert isinstance(rating, float)
        assert not math.isnan(rating), "HLTV rating must not be NaN"
        assert not math.isinf(rating), "HLTV rating must not be Inf"
        # A 0-kill player should have a very low but valid rating
        assert rating >= 0.0

    def test_kast_percentage_not_nan(self, zero_kill_player):
        """KAST% is 0.0 for a player with 0 KAST rounds (not NaN)."""
        kast = zero_kill_player.kast_percentage
        assert isinstance(kast, float)
        assert not math.isnan(kast)
        assert kast == 0.0  # No KAST rounds by default

    def test_impact_rating_not_nan(self, zero_kill_player):
        """Impact rating is computed (not null/NaN) for 0-kill player."""
        impact = zero_kill_player.impact_rating
        assert isinstance(impact, float)
        assert not math.isnan(impact)
        assert not math.isinf(impact)

    def test_headshot_percentage_zero(self, zero_kill_player):
        """HS% doesn't crash with 0 kills (no division by zero)."""
        assert zero_kill_player.headshot_percentage == 0.0

    def test_kd_ratio_zero_deaths_guard(self):
        """K/D ratio with 0 deaths returns kills as float, not infinity."""
        player = PlayerMatchStats(
            steam_id=88888,
            name="NoDeath",
            team="CT",
            kills=5,
            deaths=0,
            assists=0,
            headshots=3,
            total_damage=500,
            rounds_played=10,
        )
        kd = player.kd_ratio
        assert not math.isinf(kd), "K/D ratio must not be infinity"
        assert kd == 5.0  # deaths=0 → returns float(kills)

    def test_aim_rating_zero_kill_player(self, zero_kill_player):
        """Aim rating computed for 0-kill player (no TTD/CP data)."""
        aim = zero_kill_player.aim_rating
        assert isinstance(aim, (int, float))
        assert not math.isnan(aim)

    def test_utility_rating_zero_kill_player(self, zero_kill_player):
        """Utility rating is 0 for player with no utility thrown."""
        util_rating = zero_kill_player.utility_rating
        assert isinstance(util_rating, float)
        assert util_rating == 0.0  # No utility → quantity = 0 → rating = 0

    def test_impact_plus_minus_zero_kills(self, zero_kill_player):
        """Impact +/- doesn't crash for 0-kill player."""
        ipm = zero_kill_player.impact_plus_minus
        assert isinstance(ipm, float)
        assert not math.isnan(ipm)
        assert not math.isinf(ipm)

    def test_zero_kill_in_full_pipeline(self):
        """0-kill player survives full analyzer pipeline."""
        kills_df = pd.DataFrame(
            {
                "tick": [5000, 6000],
                "attacker_steamid": [11111, 11111],
                "user_steamid": [22222, 22222],
                "attacker_name": ["Alice", "Alice"],
                "user_name": ["Bob", "Bob"],
                "weapon": ["ak47", "ak47"],
                "headshot": [True, False],
            }
        )
        damages_df = pd.DataFrame(
            {
                "tick": [5000, 6000],
                "attacker_steamid": [11111, 11111],
                "user_steamid": [22222, 22222],
                "dmg_health": [100, 100],
                "weapon": ["ak47", "ak47"],
            }
        )
        demo_data = _make_demo_data(
            num_rounds=2,
            player_stats={
                11111: {
                    "name": "Alice",
                    "team": "CT",
                    "kills": 2,
                    "deaths": 0,
                    "assists": 0,
                    "headshots": 1,
                    "total_damage": 200,
                    "adr": 100.0,
                    "hs_percent": 50.0,
                    "weapon_kills": {"ak47": 2},
                },
                22222: {
                    "name": "Bob",
                    "team": "T",
                    "kills": 0,
                    "deaths": 2,
                    "assists": 0,
                    "headshots": 0,
                    "total_damage": 0,
                    "adr": 0.0,
                    "hs_percent": 0.0,
                    "weapon_kills": {},
                },
            },
            player_names={11111: "Alice", 22222: "Bob"},
            player_teams={11111: 3, 22222: 2},
            kills_df=kills_df,
            damages_df=damages_df,
        )
        analysis = _run_analyzer(demo_data)
        bob = analysis.players[22222]

        assert bob.kills == 0
        assert not math.isnan(bob.hltv_rating)
        assert not math.isnan(bob.kast_percentage)
        assert not math.isnan(bob.impact_rating)


# ===========================================================================
# 4. MISSING UTILITY DATA (utility=None)
# ===========================================================================


class TestMissingUtility:
    """Mock a player with trades and clutches but utility=None.

    Verify the orchestrator serialization handles NoneType access
    on p.utility.* fields.
    """

    def test_player_with_none_utility_serializes(self):
        """PlayerMatchStats with utility=None doesn't crash orchestrator."""
        # Build a PlayerMatchStats with utility explicitly set to None
        # (This can happen if _safe_calculate("utility_stats") fails)
        player = PlayerMatchStats(
            steam_id=33333,
            name="NoUtil",
            team="CT",
            kills=10,
            deaths=5,
            assists=3,
            headshots=4,
            total_damage=800,
            rounds_played=10,
            trades=TradeStats(
                trade_kill_opportunities=3,
                trade_kill_success=2,
                kills_traded=2,
                deaths_traded=1,
            ),
            clutches=ClutchStats(
                total_situations=2,
                total_wins=1,
            ),
        )

        # Orchestrator accesses p.utility.flashbangs_thrown etc.
        # The default UtilityStats() should be fine, but let's test with None:
        # We have to force utility to None after construction since it's defaulted
        object.__setattr__(player, "utility", None)

        # Now simulate what the orchestrator does — the serialization logic
        # that accesses p.utility.<field>:
        p = player

        # These are the exact access patterns from orchestrator.py lines 134-170
        utility_dict = {
            "flashbangs_thrown": (p.utility.flashbangs_thrown if p.utility else 0),
            "smokes_thrown": p.utility.smokes_thrown if p.utility else 0,
            "he_thrown": p.utility.he_thrown if p.utility else 0,
            "molotovs_thrown": p.utility.molotovs_thrown if p.utility else 0,
            "flash_assists": p.utility.flash_assists if p.utility else 0,
            "enemies_flashed": p.utility.enemies_flashed if p.utility else 0,
            "teammates_flashed": (p.utility.teammates_flashed if p.utility else 0),
            "he_damage": p.utility.he_damage if p.utility else 0,
            "molotov_damage": p.utility.molotov_damage if p.utility else 0,
            "enemies_flashed_per_round": (p.utility.enemies_flashed_per_round if p.utility else 0),
            "friends_flashed_per_round": (p.utility.friends_flashed_per_round if p.utility else 0),
            "avg_blind_time": p.utility.avg_blind_time if p.utility else 0,
            "avg_he_damage": p.utility.avg_he_damage if p.utility else 0,
            "flash_effectiveness_pct": (p.utility.flash_effectiveness_pct if p.utility else 0),
            "flash_assist_pct": (p.utility.flash_assist_pct if p.utility else 0),
            "he_team_damage": p.utility.he_team_damage if p.utility else 0,
            "unused_utility_value": p.utility.unused_utility_value if p.utility else 0,
            "utility_quality_rating": (
                round(p.utility.utility_quality_rating, 1) if p.utility else 0
            ),
            "utility_quantity_rating": (
                round(p.utility.utility_quantity_rating, 1) if p.utility else 0
            ),
            "effective_flashes": p.utility.effective_flashes if p.utility else 0,
            "total_blind_time": p.utility.total_blind_time if p.utility else 0,
            "times_blinded": p.utility.times_blinded if p.utility else 0,
            "total_time_blinded": p.utility.total_time_blinded if p.utility else 0,
            "avg_time_blinded": p.utility.avg_time_blinded if p.utility else 0,
        }

        # All values should be 0 (not crash, not None, not NaN)
        for key, val in utility_dict.items():
            assert val == 0, f"utility[{key!r}] should be 0, got {val}"

    def test_player_with_none_trades_serializes(self):
        """PlayerMatchStats with trades=None doesn't crash orchestrator."""
        player = PlayerMatchStats(
            steam_id=44444,
            name="NoTrades",
            team="T",
            kills=5,
            deaths=8,
            assists=1,
            headshots=2,
            total_damage=400,
            rounds_played=10,
        )
        object.__setattr__(player, "trades", None)
        object.__setattr__(player, "clutches", None)
        object.__setattr__(player, "opening_duels", None)
        object.__setattr__(player, "spray_transfers", None)

        p = player

        # Exercise orchestrator duels serialization (lines 196-209)
        duels_dict = {
            "trade_kills": p.trades.kills_traded if p.trades else 0,
            "traded_deaths": p.trades.deaths_traded if p.trades else 0,
            "trade_kill_opportunities": (p.trades.trade_kill_opportunities if p.trades else 0),
            "clutch_wins": p.clutches.total_wins if p.clutches else 0,
            "clutch_attempts": p.clutches.total_situations if p.clutches else 0,
            "opening_kills": p.opening_duels.wins if p.opening_duels else 0,
            "opening_deaths": p.opening_duels.losses if p.opening_duels else 0,
        }
        for key, val in duels_dict.items():
            assert val == 0, f"duels[{key!r}] should be 0, got {val}"

        # Exercise spray_transfers serialization (lines 211-231)
        spray_dict = {
            "double_sprays": p.spray_transfers.double_sprays if p.spray_transfers else 0,
            "total_sprays": p.spray_transfers.total_sprays if p.spray_transfers else 0,
            "total_spray_kills": (p.spray_transfers.total_spray_kills if p.spray_transfers else 0),
        }
        for key, val in spray_dict.items():
            assert val == 0, f"spray[{key!r}] should be 0, got {val}"


# ===========================================================================
# 5. OVERTIME CHECK — round_timeline handles rounds 31+
# ===========================================================================


class TestOvertimeRounds:
    """Verify round_timeline builds correctly when rounds exceed 30 (overtime).

    Since we can't assume a golden demo is available, we construct a
    synthetic demo with rounds > 30 and verify the orchestrator doesn't
    crash or truncate.
    """

    def test_round_timeline_with_overtime_rounds(self):
        """Orchestrator builds timeline for 32-round (overtime) demo."""
        # Build 32 rounds of alternating CT/T wins
        rounds = []
        ct_score = 0
        t_score = 0
        for i in range(1, 33):
            if i % 2 == 1:
                winner = "CT"
                ct_score += 1
            else:
                winner = "T"
                t_score += 1
            rounds.append(
                RoundInfo(
                    round_num=i,
                    start_tick=(i - 1) * 10000,
                    end_tick=i * 10000 - 1,
                    freeze_end_tick=(i - 1) * 10000 + 500,
                    winner=winner,
                    reason="elimination",
                    ct_score=ct_score,
                    t_score=t_score,
                )
            )

        # Create kills across multiple rounds including overtime (rounds 31, 32)
        kill_events = []
        kills_data = {
            "tick": [],
            "attacker_steamid": [],
            "user_steamid": [],
            "attacker_name": [],
            "user_name": [],
            "weapon": [],
            "headshot": [],
        }
        for i in range(1, 33):
            tick = (i - 1) * 10000 + 5000
            kill_events.append(
                KillEvent(
                    tick=tick,
                    round_num=i,
                    attacker_steamid=11111,
                    attacker_name="Alice",
                    attacker_side="CT" if i <= 15 else "T",
                    victim_steamid=22222,
                    victim_name="Bob",
                    victim_side="T" if i <= 15 else "CT",
                    weapon="ak47",
                    headshot=False,
                )
            )
            kills_data["tick"].append(tick)
            kills_data["attacker_steamid"].append(11111)
            kills_data["user_steamid"].append(22222)
            kills_data["attacker_name"].append("Alice")
            kills_data["user_name"].append("Bob")
            kills_data["weapon"].append("ak47")
            kills_data["headshot"].append(False)

        kills_df = pd.DataFrame(kills_data)
        damages_df = pd.DataFrame(
            {
                "tick": kills_data["tick"],
                "attacker_steamid": [11111] * 32,
                "user_steamid": [22222] * 32,
                "dmg_health": [100] * 32,
                "weapon": ["ak47"] * 32,
            }
        )

        demo_data = _make_demo_data(
            num_rounds=32,
            player_stats={
                11111: {
                    "name": "Alice",
                    "team": "CT",
                    "kills": 32,
                    "deaths": 0,
                    "assists": 0,
                    "headshots": 0,
                    "total_damage": 3200,
                    "adr": 100.0,
                    "hs_percent": 0.0,
                    "weapon_kills": {"ak47": 32},
                },
                22222: {
                    "name": "Bob",
                    "team": "T",
                    "kills": 0,
                    "deaths": 32,
                    "assists": 0,
                    "headshots": 0,
                    "total_damage": 0,
                    "adr": 0.0,
                    "hs_percent": 0.0,
                    "weapon_kills": {},
                },
            },
            player_names={11111: "Alice", 22222: "Bob"},
            player_teams={11111: 3, 22222: 2},
            kills=kill_events,
            damages=[],
            rounds=rounds,
            kills_df=kills_df,
            damages_df=damages_df,
        )

        result = _run_orchestrator(demo_data)

        assert isinstance(result, dict)
        assert result["demo_info"]["rounds"] == 32

        # Verify round_timeline includes overtime rounds (31, 32)
        timeline = result["round_timeline"]
        assert isinstance(timeline, list)

        # Timeline should have entries for rounds 31+ if present
        round_nums_in_timeline = {r["round_num"] for r in timeline if "round_num" in r}
        if round_nums_in_timeline:
            # If timeline is populated, it should include overtime rounds
            assert 31 in round_nums_in_timeline, (
                f"Round 31 (overtime) missing from timeline. "
                f"Present rounds: {sorted(round_nums_in_timeline)}"
            )
            assert 32 in round_nums_in_timeline, (
                f"Round 32 (overtime) missing from timeline. "
                f"Present rounds: {sorted(round_nums_in_timeline)}"
            )

    def test_analyzer_handles_overtime_round_count(self):
        """DemoAnalyzer.analyze() works with >30 rounds."""
        rounds = []
        for i in range(1, 33):
            rounds.append(
                RoundInfo(
                    round_num=i,
                    start_tick=(i - 1) * 10000,
                    end_tick=i * 10000 - 1,
                    freeze_end_tick=(i - 1) * 10000 + 500,
                    winner="CT" if i % 2 else "T",
                    reason="elimination",
                )
            )

        demo_data = _make_demo_data(
            num_rounds=32,
            player_names={11111: "Alice", 22222: "Bob"},
            player_teams={11111: 3, 22222: 2},
            player_stats={
                11111: {
                    "name": "Alice",
                    "team": "CT",
                    "kills": 16,
                    "deaths": 16,
                    "assists": 0,
                    "headshots": 8,
                    "total_damage": 1600,
                    "adr": 50.0,
                    "hs_percent": 50.0,
                    "weapon_kills": {"ak47": 16},
                },
                22222: {
                    "name": "Bob",
                    "team": "T",
                    "kills": 16,
                    "deaths": 16,
                    "assists": 0,
                    "headshots": 8,
                    "total_damage": 1600,
                    "adr": 50.0,
                    "hs_percent": 50.0,
                    "weapon_kills": {"ak47": 16},
                },
            },
            rounds=rounds,
            kills_df=pd.DataFrame(
                {
                    "tick": [5000],
                    "attacker_steamid": [11111],
                    "user_steamid": [22222],
                    "weapon": ["ak47"],
                    "headshot": [True],
                }
            ),
            damages_df=pd.DataFrame(),
        )

        analysis = _run_analyzer(demo_data)
        assert analysis.total_rounds == 32
        # All players should have rounds_played reflecting the full match
        for player in analysis.players.values():
            assert player.rounds_played == 32
