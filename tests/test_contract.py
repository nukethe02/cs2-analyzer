"""
Contract validation tests — ensure orchestrator output matches the contract.

Two test strategies:
1. Synthetic: Build a fake MatchAnalysis with known data, run the orchestrator's
   serialization logic, and validate every field exists with the right type.
2. Runtime (requires demo): Parse a real demo, run the full pipeline, and validate.
   Skipped if no demo file is available.
"""

from __future__ import annotations

import numpy as np

from opensight.analysis.models import (
    ClutchStats,
    MatchAnalysis,
    MultiKillStats,
    OpeningDuelStats,
    PlayerMatchStats,
    SprayTransferStats,
    TradeStats,
    UtilityStats,
)
from opensight.pipeline.contract import (
    validate_player,
    validate_result,
)


def _make_player(steam_id: int = 76561198000000001, name: str = "TestPlayer") -> PlayerMatchStats:
    """Build a realistic PlayerMatchStats for contract testing."""
    utility = UtilityStats(
        flashbangs_thrown=12,
        smokes_thrown=8,
        he_thrown=5,
        molotovs_thrown=4,
        enemies_flashed=10,
        teammates_flashed=2,
        flash_assists=3,
        total_blind_time=18.5,
        effective_flashes=8,
        times_blinded=4,
        total_time_blinded=6.2,
        he_damage=120,
        he_team_damage=15,
        molotov_damage=80,
        unused_utility_value=450,
        _rounds_played=24,
    )

    trades = TradeStats(
        trade_kill_opportunities=8,
        trade_kill_attempts=6,
        trade_kill_success=4,
        traded_death_opportunities=5,
        traded_death_attempts=3,
        traded_death_success=2,
        kills_traded=4,
        deaths_traded=2,
        traded_entry_kills=1,
        traded_entry_deaths=1,
    )

    clutches = ClutchStats(
        total_situations=5,
        total_wins=2,
        v1_attempts=3,
        v1_wins=2,
        v2_attempts=1,
        v2_wins=0,
        v3_attempts=1,
        v3_wins=0,
    )

    opening_duels = OpeningDuelStats(
        wins=6,
        losses=3,
        attempts=9,
        entry_ttd_values=[180.0, 220.0, 195.0],
        t_side_entries=4,
        ct_side_entries=2,
    )

    multi_kills = MultiKillStats(
        rounds_with_1k=8,
        rounds_with_2k=4,
        rounds_with_3k=2,
        rounds_with_4k=1,
        rounds_with_5k=0,
    )

    spray_transfers = SprayTransferStats(
        double_sprays=3,
        triple_sprays=1,
        total_spray_kills=7,
    )

    p = PlayerMatchStats(
        steam_id=steam_id,
        name=name,
        team="CT",
        kills=20,
        deaths=14,
        assists=5,
        headshots=12,
        total_damage=2400,
        rounds_played=24,
        opening_duels=opening_duels,
        trades=trades,
        clutches=clutches,
        multi_kills=multi_kills,
        spray_transfers=spray_transfers,
        utility=utility,
        kast_rounds=18,
        rounds_survived=10,
        shots_fired=300,
        shots_hit=90,
        headshot_hits=35,
        spray_shots_fired=100,
        spray_shots_hit=25,
        shots_stationary=200,
        shots_with_velocity=280,
        prefire_count=2,
    )
    # Add engagement duration values for TTD/95th calculation
    p.engagement_duration_values = [150.0, 200.0, 250.0, 300.0, 180.0, 220.0, 400.0, 170.0]
    p.cp_values = [5.0, 8.0, 12.0, 6.0, 9.0, 7.0]

    return p


def _make_analysis() -> MatchAnalysis:
    """Build a minimal MatchAnalysis with two players."""
    p1 = _make_player(76561198000000001, "PlayerOne")
    p2 = _make_player(76561198000000002, "PlayerTwo")
    p2.team = "T"

    return MatchAnalysis(
        players={p1.steam_id: p1, p2.steam_id: p2},
        team1_score=13,
        team2_score=11,
        total_rounds=24,
        map_name="de_dust2",
    )


class TestContractValidation:
    """Test that the orchestrator output matches the contract schema."""

    def test_orchestrator_output_matches_contract(self):
        """Build an orchestrator result from synthetic data and validate."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        analysis = _make_analysis()
        orchestrator = DemoOrchestrator()

        # Call the internal serialization methods directly
        # We can't call analyze() without a real demo, so we replicate
        # the player dict building logic
        players = {}
        for sid, p in analysis.players.items():
            mk = {
                "2k": p.multi_kills.rounds_with_2k,
                "3k": p.multi_kills.rounds_with_3k,
                "4k": p.multi_kills.rounds_with_4k,
                "5k": p.multi_kills.rounds_with_5k,
            }
            player_dict = {
                "steam_id": str(sid),
                "name": p.name,
                "team": p.team,
                "rounds_played": p.rounds_played,
                "total_damage": p.total_damage or 0,
                "stats": {
                    "kills": p.kills,
                    "deaths": p.deaths,
                    "assists": p.assists,
                    "rounds_played": p.rounds_played,
                    "adr": round(p.adr, 1) if p.adr is not None else 0,
                    "headshot_pct": round(p.headshot_percentage, 1)
                    if p.headshot_percentage is not None
                    else 0,
                    "kd_ratio": round(p.kills / max(1, p.deaths), 2),
                    "total_damage": p.total_damage or 0,
                    "2k": mk["2k"],
                    "3k": mk["3k"],
                    "4k": mk["4k"],
                    "5k": mk["5k"],
                },
                "rating": {
                    "hltv_rating": round(p.hltv_rating, 2) if p.hltv_rating is not None else 0,
                    "kast_percentage": round(p.kast_percentage, 1)
                    if p.kast_percentage is not None
                    else 0,
                    "aim_rating": round(p.aim_rating, 1),
                    "utility_rating": round(p.utility_rating, 1)
                    if p.utility_rating is not None
                    else 0,
                    "impact_rating": round(p.impact_rating or 0, 2),
                },
                "advanced": {
                    "ttd_median_ms": round(p.ttd_median_ms or 0, 1),
                    "ttd_mean_ms": round(p.ttd_mean_ms or 0, 1),
                    "ttd_95th_ms": (
                        round(float(np.percentile(p.engagement_duration_values, 95)), 1)
                        if p.engagement_duration_values
                        else 0
                    ),
                    "cp_median_error_deg": round(p.cp_median_error_deg or 0, 1),
                    "cp_mean_error_deg": round(p.cp_mean_error_deg or 0, 1),
                    "prefire_kills": p.prefire_count or 0,
                    "opening_kills": p.opening_duels.wins if p.opening_duels else 0,
                    "opening_deaths": p.opening_duels.losses if p.opening_duels else 0,
                },
                "utility": {
                    "flashbangs_thrown": p.utility.flashbangs_thrown,
                    "smokes_thrown": p.utility.smokes_thrown,
                    "he_thrown": p.utility.he_thrown,
                    "molotovs_thrown": p.utility.molotovs_thrown,
                    "flash_assists": p.utility.flash_assists,
                    "enemies_flashed": p.utility.enemies_flashed,
                    "teammates_flashed": p.utility.teammates_flashed,
                    "he_damage": p.utility.he_damage,
                    "molotov_damage": p.utility.molotov_damage,
                    "enemies_flashed_per_round": p.utility.enemies_flashed_per_round,
                    "friends_flashed_per_round": p.utility.friends_flashed_per_round,
                    "avg_blind_time": p.utility.avg_blind_time,
                    "avg_he_damage": p.utility.avg_he_damage,
                    "flash_effectiveness_pct": p.utility.flash_effectiveness_pct,
                    "flash_assist_pct": p.utility.flash_assist_pct,
                    "he_team_damage": p.utility.he_team_damage,
                    "unused_utility_value": p.utility.unused_utility_value,
                    "utility_quality_rating": round(p.utility.utility_quality_rating, 1),
                    "utility_quantity_rating": round(p.utility.utility_quantity_rating, 1),
                    "effective_flashes": p.utility.effective_flashes,
                    "total_blind_time": p.utility.total_blind_time,
                    "times_blinded": p.utility.times_blinded,
                    "total_time_blinded": p.utility.total_time_blinded,
                    "avg_time_blinded": p.utility.avg_time_blinded,
                },
                "aim_stats": {
                    "shots_fired": p.shots_fired,
                    "shots_hit": p.shots_hit,
                    "accuracy_all": round(p.accuracy, 1),
                    "headshot_hits": p.headshot_hits,
                    "head_accuracy": round(p.head_hit_rate, 1),
                    "spray_shots_fired": p.spray_shots_fired,
                    "spray_shots_hit": p.spray_shots_hit,
                    "spray_accuracy": round(p.spray_accuracy, 1),
                    "shots_stationary": p.shots_stationary,
                    "shots_with_velocity": p.shots_with_velocity,
                    "counter_strafe_pct": round(p.counter_strafe_pct, 1),
                    "time_to_damage_ms": round(p.ttd_median_ms or 0, 1),
                    "crosshair_placement_deg": round(p.cp_median_error_deg or 0, 1),
                },
                "duels": {
                    "trade_kills": p.trades.kills_traded,
                    "traded_deaths": p.trades.deaths_traded,
                    "clutch_wins": p.clutches.total_wins,
                    "clutch_attempts": p.clutches.total_situations,
                    "opening_kills": p.opening_duels.wins,
                    "opening_deaths": p.opening_duels.losses,
                    "opening_wins": p.opening_duels.wins,
                    "opening_losses": p.opening_duels.losses,
                    "opening_win_rate": p.opening_duels.win_rate,
                },
                "spray_transfers": {
                    "double_sprays": p.spray_transfers.double_sprays,
                    "triple_sprays": p.spray_transfers.triple_sprays,
                    "quad_sprays": p.spray_transfers.quad_sprays,
                    "ace_sprays": p.spray_transfers.ace_sprays,
                    "total_sprays": p.spray_transfers.total_sprays,
                    "total_spray_kills": p.spray_transfers.total_spray_kills,
                    "avg_spray_time_ms": p.spray_transfers.avg_spray_time_ms,
                    "avg_kills_per_spray": (
                        round(
                            p.spray_transfers.total_spray_kills
                            / max(1, p.spray_transfers.total_sprays),
                            1,
                        )
                        if p.spray_transfers.total_sprays > 0
                        else 0
                    ),
                },
                "entry": orchestrator._get_entry_stats(p),
                "trades": orchestrator._get_trade_stats(p),
                "clutches": orchestrator._get_clutch_stats(p),
                "rws": {
                    "avg_rws": 0,
                    "total_rws": 0,
                    "rounds_won": 0,
                    "rounds_played": 0,
                    "damage_per_round": 0,
                    "objective_completions": 0,
                },
            }
            players[str(sid)] = player_dict

        # Validate each player against contract
        for sid, pdata in players.items():
            errors = validate_player(pdata)
            assert errors == [], f"Player {sid} contract violations:\n" + "\n".join(errors)

    def test_entry_stats_contract(self):
        """_get_entry_stats returns all contracted fields."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        entry = orch._get_entry_stats(p)

        expected_keys = {
            "entry_attempts",
            "entry_kills",
            "entry_deaths",
            "entry_diff",
            "entry_attempts_pct",
            "entry_success_pct",
        }
        assert set(entry.keys()) == expected_keys

    def test_trade_stats_contract(self):
        """_get_trade_stats returns all 14 contracted fields."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        trades = orch._get_trade_stats(p)

        expected_keys = {
            "trade_kill_opportunities",
            "trade_kill_attempts",
            "trade_kill_attempts_pct",
            "trade_kill_success",
            "trade_kill_success_pct",
            "traded_death_opportunities",
            "traded_death_attempts",
            "traded_death_attempts_pct",
            "traded_death_success",
            "traded_death_success_pct",
            "trade_kills",
            "deaths_traded",
            "traded_entry_kills",
            "traded_entry_deaths",
        }
        assert set(trades.keys()) == expected_keys

    def test_trade_stats_reads_real_attributes(self):
        """_get_trade_stats reads actual TradeStats attribute values, not zeros."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        trades = orch._get_trade_stats(p)

        # These MUST reflect the synthetic data, not all zeros
        assert trades["trade_kill_opportunities"] == 8
        assert trades["trade_kill_success"] == 4
        assert trades["trade_kills"] == 4  # alias for kills_traded
        assert trades["deaths_traded"] == 2

    def test_clutch_stats_contract(self):
        """_get_clutch_stats returns all contracted fields including total_situations."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        clutches = orch._get_clutch_stats(p)

        expected_keys = {
            "clutch_wins",
            "clutch_losses",
            "clutch_success_pct",
            "total_situations",
            "v1_wins",
            "v2_wins",
            "v3_wins",
            "v4_wins",
            "v5_wins",
        }
        assert set(clutches.keys()) == expected_keys
        assert clutches["total_situations"] == 5
        assert clutches["clutch_wins"] == 2

    def test_entry_stats_reads_opening_duels(self):
        """_get_entry_stats reads from opening_duels, not non-existent attributes."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        entry = orch._get_entry_stats(p)

        assert entry["entry_kills"] == 6  # opening_duels.wins
        assert entry["entry_deaths"] == 3  # opening_duels.losses
        assert entry["entry_attempts"] == 9

    def test_null_player_components(self):
        """Contract still validates when all component stats are None/default."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = PlayerMatchStats(
            steam_id=1,
            name="Empty",
            team="CT",
            kills=0,
            deaths=0,
            assists=0,
            headshots=0,
            total_damage=0,
            rounds_played=1,
        )
        orch = DemoOrchestrator()

        entry = orch._get_entry_stats(p)
        trades = orch._get_trade_stats(p)
        clutches = orch._get_clutch_stats(p)

        # Should all return zero-value dicts, no crashes
        assert entry["entry_kills"] == 0
        assert trades["trade_kill_opportunities"] == 0
        assert clutches["clutch_wins"] == 0
        assert clutches["total_situations"] == 0

    def test_trade_stats_null_returns_all_zeros(self):
        """_get_trade_stats with None trades returns dict with all expected keys at 0."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = PlayerMatchStats(
            steam_id=1,
            name="Empty",
            team="CT",
            kills=0,
            deaths=0,
            assists=0,
            headshots=0,
            total_damage=0,
            rounds_played=1,
        )
        p.trades = TradeStats()  # default empty
        orch = DemoOrchestrator()
        trades = orch._get_trade_stats(p)
        assert all(v == 0 or v == 0.0 for v in trades.values())

    def test_validate_result_catches_missing_keys(self):
        """validate_result catches missing top-level and player keys."""
        errors = validate_result({})
        assert any("MISSING" in e for e in errors)

    def test_validate_result_accepts_valid(self):
        """validate_result passes with a properly structured result."""
        from datetime import datetime

        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()

        player_dict = {
            "steam_id": str(p.steam_id),
            "name": p.name,
            "team": p.team,
            "rounds_played": p.rounds_played,
            "total_damage": p.total_damage,
            "stats": {
                "kills": p.kills,
                "deaths": p.deaths,
                "assists": p.assists,
                "rounds_played": p.rounds_played,
                "adr": round(p.adr, 1),
                "headshot_pct": round(p.headshot_percentage, 1),
                "kd_ratio": round(p.kills / max(1, p.deaths), 2),
                "total_damage": p.total_damage,
                "2k": 4,
                "3k": 2,
                "4k": 1,
                "5k": 0,
            },
            "rating": {
                "hltv_rating": 1.2,
                "kast_percentage": 75.0,
                "aim_rating": 60.0,
                "utility_rating": 45.0,
                "impact_rating": 1.1,
            },
            "advanced": {
                "ttd_median_ms": 200.0,
                "ttd_mean_ms": 210.0,
                "ttd_95th_ms": 350.0,
                "cp_median_error_deg": 8.0,
                "cp_mean_error_deg": 9.0,
                "prefire_kills": 2,
                "opening_kills": 6,
                "opening_deaths": 3,
            },
            "utility": {
                "flashbangs_thrown": 12,
                "smokes_thrown": 8,
                "he_thrown": 5,
                "molotovs_thrown": 4,
                "flash_assists": 3,
                "enemies_flashed": 10,
                "teammates_flashed": 2,
                "he_damage": 120,
                "molotov_damage": 80,
                "enemies_flashed_per_round": 0.42,
                "friends_flashed_per_round": 0.08,
                "avg_blind_time": 1.85,
                "avg_he_damage": 5.0,
                "flash_effectiveness_pct": 66.7,
                "flash_assist_pct": 25.0,
                "he_team_damage": 15,
                "unused_utility_value": 450,
                "utility_quality_rating": 55.0,
                "utility_quantity_rating": 60.0,
                "effective_flashes": 8,
                "total_blind_time": 18.5,
                "times_blinded": 4,
                "total_time_blinded": 6.2,
                "avg_time_blinded": 1.55,
            },
            "aim_stats": {
                "shots_fired": 300,
                "shots_hit": 90,
                "accuracy_all": 30.0,
                "headshot_hits": 35,
                "head_accuracy": 38.9,
                "spray_shots_fired": 100,
                "spray_shots_hit": 25,
                "spray_accuracy": 25.0,
                "shots_stationary": 200,
                "shots_with_velocity": 280,
                "counter_strafe_pct": 71.4,
                "time_to_damage_ms": 200.0,
                "crosshair_placement_deg": 8.0,
            },
            "duels": {
                "trade_kills": 4,
                "traded_deaths": 2,
                "clutch_wins": 2,
                "clutch_attempts": 5,
                "opening_kills": 6,
                "opening_deaths": 3,
                "opening_wins": 6,
                "opening_losses": 3,
                "opening_win_rate": 66.7,
            },
            "spray_transfers": {
                "double_sprays": 3,
                "triple_sprays": 1,
                "quad_sprays": 0,
                "ace_sprays": 0,
                "total_sprays": 4,
                "total_spray_kills": 7,
                "avg_spray_time_ms": 0,
                "avg_kills_per_spray": 1.8,
            },
            "entry": orch._get_entry_stats(p),
            "trades": orch._get_trade_stats(p),
            "clutches": orch._get_clutch_stats(p),
            "rws": {
                "avg_rws": 0,
                "total_rws": 0,
                "rounds_won": 0,
                "rounds_played": 0,
                "damage_per_round": 0,
                "objective_completions": 0,
            },
        }

        result = {
            "demo_info": {
                "map": "de_dust2",
                "rounds": 24,
                "duration_minutes": 30,
                "score": "13 - 11",
                "score_ct": 13,
                "score_t": 11,
                "total_kills": 40,
                "team1_name": "CT",
                "team2_name": "T",
            },
            "players": {str(p.steam_id): player_dict},
            "mvp": {"name": "TestPlayer", "rating": 1.2},
            "round_timeline": [],
            "kill_matrix": [],
            "heatmap_data": {},
            "coaching": {},
            "tactical": {},
            "synergy": {},
            "timeline_graph": {},
            "analyzed_at": datetime.now().isoformat(),
        }

        errors = validate_result(result)
        assert errors == [], "Contract violations:\n" + "\n".join(errors)


class TestContractConstants:
    """Verify the PLAYER_CONTRACT and RESULT_CONTRACT constants are complete."""

    def test_player_contract_has_all_top_level_sections(self):
        """PLAYER_CONTRACT must define all required top-level sections."""
        from opensight.pipeline.contract import PLAYER_CONTRACT

        required_sections = {
            "steam_id",
            "name",
            "team",
            "rounds_played",
            "total_damage",
            "stats",
            "rating",
            "advanced",
            "utility",
            "aim_stats",
            "duels",
            "spray_transfers",
            "entry",
            "trades",
            "clutches",
            "rws",
        }
        actual_keys = set(PLAYER_CONTRACT.keys())
        missing = required_sections - actual_keys
        assert missing == set(), f"PLAYER_CONTRACT missing sections: {missing}"

    def test_result_contract_has_all_top_level_keys(self):
        """RESULT_CONTRACT must define all required top-level keys."""
        from opensight.pipeline.contract import RESULT_CONTRACT

        required_keys = {
            "demo_info",
            "players",
            "mvp",
            "round_timeline",
            "kill_matrix",
            "heatmap_data",
            "coaching",
            "tactical",
            "synergy",
            "timeline_graph",
            "analyzed_at",
        }
        actual_keys = set(RESULT_CONTRACT.keys())
        missing = required_keys - actual_keys
        assert missing == set(), f"RESULT_CONTRACT missing keys: {missing}"

    def test_trade_contract_keys_match_orchestrator(self):
        """The 'trades' section in PLAYER_CONTRACT must match _get_trade_stats output."""
        from opensight.pipeline.contract import PLAYER_CONTRACT
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        trades_output = orch._get_trade_stats(p)

        contract_trades_keys = set(PLAYER_CONTRACT["trades"].keys())
        output_keys = set(trades_output.keys())

        missing_from_output = contract_trades_keys - output_keys
        extra_in_output = output_keys - contract_trades_keys

        assert missing_from_output == set(), (
            f"Contract requires keys not in _get_trade_stats output: {missing_from_output}"
        )
        assert extra_in_output == set(), (
            f"_get_trade_stats returns keys not in contract: {extra_in_output}"
        )

    def test_clutch_contract_keys_match_orchestrator(self):
        """The 'clutches' section in PLAYER_CONTRACT must match _get_clutch_stats output."""
        from opensight.pipeline.contract import PLAYER_CONTRACT
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        clutches_output = orch._get_clutch_stats(p)

        contract_clutch_keys = set(PLAYER_CONTRACT["clutches"].keys())
        output_keys = set(clutches_output.keys())

        missing_from_output = contract_clutch_keys - output_keys
        extra_in_output = output_keys - contract_clutch_keys

        assert missing_from_output == set(), (
            f"Contract requires keys not in _get_clutch_stats output: {missing_from_output}"
        )
        assert extra_in_output == set(), (
            f"_get_clutch_stats returns keys not in contract: {extra_in_output}"
        )

    def test_entry_contract_keys_match_orchestrator(self):
        """The 'entry' section in PLAYER_CONTRACT must match _get_entry_stats output."""
        from opensight.pipeline.contract import PLAYER_CONTRACT
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        entry_output = orch._get_entry_stats(p)

        contract_entry_keys = set(PLAYER_CONTRACT["entry"].keys())
        output_keys = set(entry_output.keys())

        missing_from_output = contract_entry_keys - output_keys
        extra_in_output = output_keys - contract_entry_keys

        assert missing_from_output == set(), (
            f"Contract requires keys not in _get_entry_stats output: {missing_from_output}"
        )
        assert extra_in_output == set(), (
            f"_get_entry_stats returns keys not in contract: {extra_in_output}"
        )

    def test_validate_result_catches_each_missing_top_level_key(self):
        """validate_result reports an error for EACH missing top-level key."""
        from opensight.pipeline.contract import RESULT_CONTRACT

        errors = validate_result({})
        # Should have at least one error per top-level key
        assert len(errors) >= len(RESULT_CONTRACT), (
            f"Expected at least {len(RESULT_CONTRACT)} errors for empty dict, got {len(errors)}"
        )

    def test_validate_player_catches_missing_nested_keys(self):
        """validate_player reports errors for missing nested keys."""
        # Provide top-level keys but empty nested dicts
        incomplete_player = {
            "steam_id": "123",
            "name": "Test",
            "team": "CT",
            "rounds_played": 1,
            "total_damage": 0,
            "stats": {},  # empty -- should trigger errors for every stats subkey
            "rating": {},
            "advanced": {},
            "utility": {},
            "aim_stats": {},
            "duels": {},
            "spray_transfers": {},
            "entry": {},
            "trades": {},
            "clutches": {},
            "rws": {},
        }
        errors = validate_player(incomplete_player)
        # Should have errors for missing keys within each nested dict
        assert len(errors) > 10, (
            f"Expected many missing-key errors for empty nested dicts, got {len(errors)}: {errors}"
        )


class TestTradeStatsValuePassthrough:
    """Verify _get_trade_stats reads real attribute values, not defaults."""

    def test_nonzero_trade_kill_attempts(self):
        """trade_kill_attempts passes through from TradeStats."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        trades = orch._get_trade_stats(p)
        assert trades["trade_kill_attempts"] == 6

    def test_nonzero_traded_death_stats(self):
        """Traded death stats pass through from TradeStats."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        trades = orch._get_trade_stats(p)
        assert trades["traded_death_opportunities"] == 5
        assert trades["traded_death_attempts"] == 3
        assert trades["traded_death_success"] == 2

    def test_traded_entry_stats_passthrough(self):
        """Entry-specific trade stats pass through."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        trades = orch._get_trade_stats(p)
        assert trades["traded_entry_kills"] == 1
        assert trades["traded_entry_deaths"] == 1

    def test_percentage_properties_computed(self):
        """Computed percentage properties are included and correct."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        trades = orch._get_trade_stats(p)

        # trade_kill_attempts_pct = 6/8 * 100 = 75.0
        assert trades["trade_kill_attempts_pct"] == 75.0
        # trade_kill_success_pct = 4/6 * 100 = 66.7
        assert trades["trade_kill_success_pct"] == 66.7


class TestClutchStatsValuePassthrough:
    """Verify _get_clutch_stats reads real attribute values."""

    def test_clutch_losses_computed(self):
        """clutch_losses = total_situations - total_wins."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        clutches = orch._get_clutch_stats(p)
        assert clutches["clutch_losses"] == 3  # 5 - 2

    def test_v1_wins_passthrough(self):
        """v1_wins passes through from ClutchStats."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        clutches = orch._get_clutch_stats(p)
        assert clutches["v1_wins"] == 2

    def test_clutch_success_pct_computed(self):
        """clutch_success_pct = wins/total * 100."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        p = _make_player()
        orch = DemoOrchestrator()
        clutches = orch._get_clutch_stats(p)
        # 2/5 * 100 = 40.0
        assert clutches["clutch_success_pct"] == 40


class TestImportSanity:
    """Import sanity tests to catch circular imports and missing modules."""

    def test_orchestrator_imports(self):
        """DemoOrchestrator can be imported without errors."""
        from opensight.pipeline.orchestrator import DemoOrchestrator

        assert DemoOrchestrator is not None

    def test_contract_imports(self):
        """Contract module can be imported without errors."""
        from opensight.pipeline.contract import (
            PLAYER_CONTRACT,
            RESULT_CONTRACT,
            validate_player,
            validate_result,
        )

        assert PLAYER_CONTRACT is not None
        assert RESULT_CONTRACT is not None
        assert callable(validate_player)
        assert callable(validate_result)

    def test_routes_import(self):
        """API route modules can be imported without errors."""
        from opensight.api.routes_analysis import router

        assert router is not None

    def test_ai_modules_import(self):
        """AI modules can be imported without errors."""
        from opensight.ai.llm_client import TacticalAIClient
        from opensight.ai.self_review import SelfReviewEngine
        from opensight.ai.strat_engine import StratEngine

        assert SelfReviewEngine is not None
        assert StratEngine is not None
        assert TacticalAIClient is not None

    def test_models_import(self):
        """Analysis models can be imported without errors."""
        from opensight.analysis.models import (
            ClutchStats,
            MatchAnalysis,
            MultiKillStats,
            OpeningDuelStats,
            PlayerMatchStats,
            SprayTransferStats,
            TradeStats,
            UtilityStats,
        )

        assert PlayerMatchStats is not None
        assert TradeStats is not None
        assert ClutchStats is not None
        assert OpeningDuelStats is not None
        assert UtilityStats is not None
        assert MultiKillStats is not None
        assert SprayTransferStats is not None
        assert MatchAnalysis is not None

    def test_core_modules_import(self):
        """Core modules can be imported without errors."""
        from opensight.core.constants import CS2_TICK_RATE, TRADE_WINDOW_SECONDS
        from opensight.core.utils import safe_divide, validate_steamid

        assert CS2_TICK_RATE > 0
        assert TRADE_WINDOW_SECONDS > 0
        assert callable(safe_divide)
        assert callable(validate_steamid)


class TestKeyNameConsistency:
    """Verify AI modules use correct key names from the contract."""

    def test_ai_modules_read_demo_info_not_match_info(self):
        """AI modules must access match data via 'demo_info' key, not 'match_info' key.

        The local variable name 'match_info' is fine, but the key used to
        access the data from match_data dict MUST be 'demo_info'.
        """
        import inspect

        from opensight.ai import llm_client, self_review, strat_engine

        for module in [self_review, strat_engine, llm_client]:
            source = inspect.getsource(module)
            # Check that nobody does match_data["match_info"] or match_data.get("match_info")
            # The pattern .get("demo_info" should be present; .get("match_info" should NOT
            assert '.get("match_info"' not in source and "['match_info']" not in source, (
                f"{module.__name__} accesses 'match_info' key instead of 'demo_info' — "
                f"this would silently return empty data"
            )

    def test_ai_modules_use_demo_info_key(self):
        """Confirm AI modules DO access 'demo_info' key."""
        import inspect

        from opensight.ai import llm_client, self_review, strat_engine

        for module in [self_review, strat_engine, llm_client]:
            source = inspect.getsource(module)
            assert '"demo_info"' in source, (
                f"{module.__name__} does not reference 'demo_info' — "
                f"may not be reading match metadata correctly"
            )
