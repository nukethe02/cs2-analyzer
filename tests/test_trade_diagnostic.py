"""Diagnostic test for trade detection pipeline.

This test creates a minimal but realistic DemoAnalyzer scenario
to identify why trade_kill_opportunities shows 0 for all players.
"""

from pathlib import Path

import pandas as pd

from opensight.core.parser import DemoData


def make_demo_data_with_trades() -> DemoData:
    """Create demo data with clear trade scenarios.

    Scenario (5v5, 2 rounds):
    Round 1:
      - T1 kills CT1 at tick 1000
      - CT2 kills T1 at tick 1200 (trade! 200 ticks = 3.125s at 64 tick)

    Round 2:
      - CT3 kills T2 at tick 5000
      - T3 kills CT3 at tick 5250 (trade! 250 ticks = 3.9s)
      - CT4 kills T3 at tick 5400 (trade! 150 ticks = 2.3s)
    """
    kills_df = pd.DataFrame(
        {
            "tick": [1000, 1200, 5000, 5250, 5400],
            "attacker_steamid": [201, 102, 103, 203, 104],
            "user_steamid": [101, 201, 202, 103, 203],
            "weapon": ["ak47", "m4a1", "m4a1", "ak47", "m4a1"],
            "headshot": [True, False, True, False, True],
            "total_rounds_played": [1, 1, 2, 2, 2],
            # Team columns (as if enrichment happened) - integers
            "attacker_team": [2, 3, 3, 2, 3],
            "user_team": [3, 2, 2, 3, 2],
        }
    )

    damages_df = pd.DataFrame(
        {
            "tick": [1000, 1200, 5000, 5250, 5400],
            "attacker_steamid": [201, 102, 103, 203, 104],
            "user_steamid": [101, 201, 202, 103, 203],
            "dmg_health": [100, 100, 100, 100, 100],
            "total_rounds_played": [1, 1, 2, 2, 2],
            "attacker_team": [2, 3, 3, 2, 3],
            "user_team": [3, 2, 2, 3, 2],
        }
    )

    # CT players: 101-105, T players: 201-205
    player_names = {
        101: "CT1",
        102: "CT2",
        103: "CT3",
        104: "CT4",
        105: "CT5",
        201: "T1",
        202: "T2",
        203: "T3",
        204: "T4",
        205: "T5",
    }

    # player_teams from _extract_players (string "CT"/"T")
    player_teams = {
        101: "CT",
        102: "CT",
        103: "CT",
        104: "CT",
        105: "CT",
        201: "T",
        202: "T",
        203: "T",
        204: "T",
        205: "T",
    }

    # Persistent team data
    player_persistent_teams = {
        101: "Team A",
        102: "Team A",
        103: "Team A",
        104: "Team A",
        105: "Team A",
        201: "Team B",
        202: "Team B",
        203: "Team B",
        204: "Team B",
        205: "Team B",
    }

    team_rosters = {
        "Team A": {101, 102, 103, 104, 105},
        "Team B": {201, 202, 203, 204, 205},
    }

    team_starting_sides = {"Team A": "CT", "Team B": "T"}

    return DemoData(
        file_path=Path("/tmp/test_trade.dem"),
        map_name="de_dust2",
        duration_seconds=1800.0,
        tick_rate=64,
        num_rounds=2,
        player_stats={},
        player_names=player_names,
        player_teams=player_teams,
        player_persistent_teams=player_persistent_teams,
        team_rosters=team_rosters,
        team_starting_sides=team_starting_sides,
        halftime_round=13,
        kills=[],
        damages=[],
        kills_df=kills_df,
        damages_df=damages_df,
    )


class TestTradeDetectionDiagnostic:
    """Diagnostic tests to find the root cause of all-zero trade stats."""

    def test_player_teams_set_correctly(self):
        """Step 1: Verify player.team is set to 'CT' or 'T' after init."""
        from opensight.analysis.analytics import DemoAnalyzer

        demo_data = make_demo_data_with_trades()
        analyzer = DemoAnalyzer(demo_data)

        # Call _init_player_stats manually
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        # Check that all players have valid teams
        for steam_id, player in analyzer._players.items():
            assert player.team in ("CT", "T"), (
                f"Player {player.name} (sid={steam_id}) has team='{player.team}', expected 'CT' or 'T'"
            )

        # Specifically check CT and T players
        assert analyzer._players[101].team == "CT"
        assert analyzer._players[201].team == "T"

    def test_detect_trades_produces_nonzero(self):
        """Step 2: Verify detect_trades produces non-zero values."""
        from opensight.analysis.analytics import DemoAnalyzer

        demo_data = make_demo_data_with_trades()
        analyzer = DemoAnalyzer(demo_data)

        # Run the full analysis
        result = analyzer.analyze()

        # Collect all trade stats
        total_opps = 0
        total_attempts = 0
        total_success = 0
        total_death_opps = 0
        total_death_success = 0

        for _steam_id, player in result.players.items():
            trades = player.trades
            total_opps += trades.trade_kill_opportunities
            total_attempts += trades.trade_kill_attempts
            total_success += trades.trade_kill_success
            total_death_opps += trades.traded_death_opportunities
            total_death_success += trades.traded_death_success

        # Print diagnostic info
        print("\n=== TRADE DETECTION DIAGNOSTIC ===")
        print(f"Total trade kill opportunities: {total_opps}")
        print(f"Total trade kill attempts: {total_attempts}")
        print(f"Total trade kill success: {total_success}")
        print(f"Total traded death opportunities: {total_death_opps}")
        print(f"Total traded death success: {total_death_success}")

        for _steam_id, player in result.players.items():
            t = player.trades
            print(
                f"  {player.name} ({player.team}): "
                f"opps={t.trade_kill_opportunities}, "
                f"attempts={t.trade_kill_attempts}, "
                f"success={t.trade_kill_success}, "
                f"death_opps={t.traded_death_opportunities}, "
                f"death_success={t.traded_death_success}"
            )

        # At minimum, round 1 should have trade opportunities
        # CT1 dies to T1, then CT2 kills T1 = trade
        # CT2 should have: trade_kill_success >= 1
        # CT1 should have: traded_death_success >= 1
        assert total_opps > 0, "No trade kill opportunities detected!"
        assert total_success > 0, "No trade kill successes detected!"
        assert total_death_opps > 0, "No traded death opportunities detected!"

    def test_detect_trades_direct_call(self):
        """Step 3: Call detect_trades directly and check for exceptions."""
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        demo_data = make_demo_data_with_trades()
        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        # Print pre-conditions
        print("\n=== PRE-CONDITIONS ===")
        print(f"_round_col: {analyzer._round_col}")
        print(f"_att_id_col: {analyzer._att_id_col}")
        print(f"_vic_id_col: {analyzer._vic_id_col}")
        print(f"Number of players: {len(analyzer._players)}")

        teams = {}
        for sid, p in analyzer._players.items():
            teams[sid] = p.team
        print(f"Player teams: {teams}")

        # Call detect_trades directly (no exception swallowing)
        detect_trades(analyzer)

        # Check results
        print("\n=== POST detect_trades ===")
        for _sid, player in analyzer._players.items():
            t = player.trades
            print(
                f"  {player.name}: opps={t.trade_kill_opportunities}, "
                f"attempts={t.trade_kill_attempts}, success={t.trade_kill_success}"
            )

        # Verify CT2 (102) traded for CT1 (101)
        ct2 = analyzer._players[102]
        assert ct2.trades.trade_kill_opportunities > 0, (
            f"CT2 should have trade opportunities, got {ct2.trades.trade_kill_opportunities}"
        )
        assert ct2.trades.trade_kill_success > 0, (
            f"CT2 should have trade success, got {ct2.trades.trade_kill_success}"
        )

    def test_player_teams_lookup_building(self):
        """Step 4: Check that player_teams_lookup is built correctly in detect_trades."""
        from opensight.analysis.analytics import DemoAnalyzer

        demo_data = make_demo_data_with_trades()
        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        # Replicate the team lookup building from detect_trades
        player_teams_lookup: dict[int, str] = {}
        for steam_id, player in analyzer._players.items():
            if player.team in ("CT", "T"):
                player_teams_lookup[steam_id] = player.team

        print("\n=== PLAYER TEAMS LOOKUP ===")
        print(f"Lookup size: {len(player_teams_lookup)}")
        for sid, team in sorted(player_teams_lookup.items()):
            print(f"  {sid}: {team}")

        # Must have all 10 players
        assert len(player_teams_lookup) == 10, (
            f"Expected 10 players in lookup, got {len(player_teams_lookup)}"
        )

        # Check specific teams
        assert player_teams_lookup[101] == "CT"
        assert player_teams_lookup[201] == "T"

    def test_no_team_enrichment_failure_mode(self):
        """Step 5: Simulate no team enrichment (ticks_df was empty).

        This is the most likely failure mode: demoparser2's player_death event
        doesn't include team columns natively. If ticks_df parsing fails,
        _enrich_with_team_info is skipped, leaving no team columns in kills_df.
        """
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        # kills_df WITHOUT team columns (simulating failed enrichment)
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1200, 5000, 5250, 5400],
                "attacker_steamid": [201, 102, 103, 203, 104],
                "user_steamid": [101, 201, 202, 103, 203],
                "weapon": ["ak47", "m4a1", "m4a1", "ak47", "m4a1"],
                "headshot": [True, False, True, False, True],
                "total_rounds_played": [1, 1, 2, 2, 2],
                # NO attacker_team or user_team columns!
            }
        )

        damages_df = pd.DataFrame()

        # NO persistent teams, NO player_teams (simulating complete enrichment failure)
        demo_data = DemoData(
            file_path=Path("/tmp/test_trade_no_teams.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=2,
            player_stats={},
            player_names={
                101: "CT1",
                102: "CT2",
                103: "CT3",
                104: "CT4",
                105: "CT5",
                201: "T1",
                202: "T2",
                203: "T3",
                204: "T4",
                205: "T5",
            },
            player_teams={},  # Empty! No team info
            player_persistent_teams={},  # Empty!
            team_rosters={"Team A": set(), "Team B": set()},
            team_starting_sides={"Team A": "CT", "Team B": "T"},
            halftime_round=13,
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=damages_df,
        )

        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        # Check: what team do players get?
        print("\n=== NO ENRICHMENT: Player teams ===")
        for _sid, p in analyzer._players.items():
            print(f"  {p.name}: team='{p.team}'")

        # Build team lookup like detect_trades does
        player_teams_lookup = {
            sid: p.team for sid, p in analyzer._players.items() if p.team in ("CT", "T")
        }
        print(f"Teams lookup size: {len(player_teams_lookup)} (should be 10, likely 0)")

        # Now call detect_trades
        detect_trades(analyzer)

        total_opps = sum(p.trades.trade_kill_opportunities for p in analyzer._players.values())
        print(f"Total trade opportunities: {total_opps}")

        # After fix: graph-based team inference should kick in and produce trades
        assert total_opps > 0, (
            "Trade detection should infer teams from kill relationships when team enrichment fails"
        )

    def test_with_player_teams_but_no_persistent(self):
        """Step 6: player_teams is set (from _extract_players) but persistent teams empty.

        This tests the fallback path in _init_player_stats.
        """
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1200, 5000, 5250, 5400],
                "attacker_steamid": [201, 102, 103, 203, 104],
                "user_steamid": [101, 201, 202, 103, 203],
                "weapon": ["ak47", "m4a1", "m4a1", "ak47", "m4a1"],
                "headshot": [True, False, True, False, True],
                "total_rounds_played": [1, 1, 2, 2, 2],
            }
        )

        demo_data = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=2,
            player_stats={},
            player_names={
                101: "CT1",
                102: "CT2",
                103: "CT3",
                104: "CT4",
                105: "CT5",
                201: "T1",
                202: "T2",
                203: "T3",
                204: "T4",
                205: "T5",
            },
            # player_teams IS populated (from _extract_players finding attacker_team cols)
            player_teams={
                101: "CT",
                102: "CT",
                103: "CT",
                104: "CT",
                105: "CT",
                201: "T",
                202: "T",
                203: "T",
                204: "T",
                205: "T",
            },
            player_persistent_teams={},  # Empty persistent teams
            team_rosters={"Team A": set(), "Team B": set()},
            team_starting_sides={"Team A": "CT", "Team B": "T"},
            halftime_round=13,
            kills=[],
            damages=[],
            kills_df=kills_df,
            damages_df=pd.DataFrame(),
        )

        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        print("\n=== WITH player_teams FALLBACK ===")
        for _sid, p in analyzer._players.items():
            print(f"  {p.name}: team='{p.team}'")

        detect_trades(analyzer)

        total_opps = sum(p.trades.trade_kill_opportunities for p in analyzer._players.values())
        total_success = sum(p.trades.trade_kill_success for p in analyzer._players.values())
        print(f"Total opportunities: {total_opps}, success: {total_success}")

        # With player_teams fallback, teams should be set correctly
        assert total_opps > 0, "Fallback should give us trade opportunities"
