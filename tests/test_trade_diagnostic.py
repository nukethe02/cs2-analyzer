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


class TestTradeProximityFiltering:
    """Tests for proximity-based trade opportunity filtering.

    Verifies that trade_kill_opportunities are only counted for teammates
    within TRADE_PROXIMITY_UNITS (1500 game units) of the death location,
    matching Leetify's methodology.
    """

    def _make_demo_with_positions(
        self,
        *,
        nearby_x: float = 100.0,
        nearby_y: float = 100.0,
        far_x: float = 5000.0,
        far_y: float = 5000.0,
    ) -> tuple:
        """Create demo data with position info for proximity testing.

        Scenario (Round 1):
          - T1(201) kills CT1(101) at tick 1000 at position (500, 500)
          - CT2(102) is at (nearby_x, nearby_y) — may be within proximity
          - CT3(103) is at (far_x, far_y) — should be outside proximity
          - CT2(102) kills T1(201) at tick 1200 — successful trade

        Returns (DemoData, ticks_df) where ticks_df has position data.
        """
        import numpy as np

        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1200],
                "attacker_steamid": [201, 102],
                "user_steamid": [101, 201],
                "weapon": ["ak47", "m4a1"],
                "headshot": [True, False],
                "total_rounds_played": [1, 1],
                "attacker_team": [2, 3],
                "user_team": [3, 2],
                # Death positions (victim position)
                "user_X": [500.0, 600.0],
                "user_Y": [500.0, 500.0],
            }
        )

        damages_df = pd.DataFrame(
            {
                "tick": [1000, 1200],
                "attacker_steamid": [201, 102],
                "user_steamid": [101, 201],
                "dmg_health": [100, 100],
                "total_rounds_played": [1, 1],
                "attacker_team": [2, 3],
                "user_team": [3, 2],
            }
        )

        # ticks_df with player positions at relevant ticks
        # Each player has position data near tick 1000
        ticks_data = []
        # CT1 (101) at death location
        ticks_data.append({"steamid": 101, "tick": 999, "X": 500.0, "Y": 500.0, "Z": 0.0})
        # CT2 (102) nearby or far depending on test
        ticks_data.append({"steamid": 102, "tick": 999, "X": nearby_x, "Y": nearby_y, "Z": 0.0})
        # CT3 (103) always far away
        ticks_data.append({"steamid": 103, "tick": 999, "X": far_x, "Y": far_y, "Z": 0.0})
        # CT4 (104) far away
        ticks_data.append({"steamid": 104, "tick": 999, "X": 4000.0, "Y": 4000.0, "Z": 0.0})
        # CT5 (105) far away
        ticks_data.append({"steamid": 105, "tick": 999, "X": 3500.0, "Y": 3500.0, "Z": 0.0})
        # T players
        ticks_data.append({"steamid": 201, "tick": 999, "X": 600.0, "Y": 500.0, "Z": 0.0})
        ticks_data.append({"steamid": 202, "tick": 999, "X": 700.0, "Y": 700.0, "Z": 0.0})
        ticks_data.append({"steamid": 203, "tick": 999, "X": 2000.0, "Y": 2000.0, "Z": 0.0})
        ticks_data.append({"steamid": 204, "tick": 999, "X": 3000.0, "Y": 3000.0, "Z": 0.0})
        ticks_data.append({"steamid": 205, "tick": 999, "X": 2500.0, "Y": 2500.0, "Z": 0.0})

        ticks_df = pd.DataFrame(ticks_data)
        # Ensure correct dtypes
        ticks_df["steamid"] = ticks_df["steamid"].astype(np.int64)
        ticks_df["tick"] = ticks_df["tick"].astype(np.int64)

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

        demo_data = DemoData(
            file_path=Path("/tmp/test_proximity.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
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
            ticks_df=ticks_df,
        )

        return demo_data

    def test_nearby_teammate_gets_opportunity(self):
        """Teammate within 1500 units of death gets a trade opportunity."""
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        # CT2 is at (600, 500), death at (500, 500) = 100 units away (nearby)
        demo_data = self._make_demo_with_positions(nearby_x=600.0, nearby_y=500.0)
        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        detect_trades(analyzer)

        ct2 = analyzer._players[102]
        assert ct2.trades.trade_kill_opportunities >= 1, (
            f"CT2 (100 units away) should get trade opportunity, got {ct2.trades.trade_kill_opportunities}"
        )
        # CT2 also successfully traded
        assert ct2.trades.trade_kill_success >= 1, (
            f"CT2 should have trade success, got {ct2.trades.trade_kill_success}"
        )

    def test_far_teammate_no_opportunity(self):
        """Teammate 5000+ units away should NOT get a trade opportunity."""
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        # CT2 at (600, 500) is nearby; CT3 at (5000, 5000) is far
        demo_data = self._make_demo_with_positions(nearby_x=600.0, nearby_y=500.0)
        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        detect_trades(analyzer)

        ct3 = analyzer._players[103]
        assert ct3.trades.trade_kill_opportunities == 0, (
            f"CT3 (5000+ units away) should NOT get trade opportunity, "
            f"got {ct3.trades.trade_kill_opportunities}"
        )

    def test_proximity_reduces_total_opportunities(self):
        """With proximity filtering, total opportunities should be much less than
        'all alive teammates' counting.

        Old behavior: 4 alive CT teammates each get +1 = 4 opportunities per death.
        New behavior: only 1 nearby teammate gets +1 = 1 opportunity per death.
        """
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        # Only CT2 is near the death (600, 500). CT3-CT5 are 3500-5000 units away.
        demo_data = self._make_demo_with_positions(nearby_x=600.0, nearby_y=500.0)
        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        detect_trades(analyzer)

        # Count total CT trade_kill_opportunities
        ct_opps = sum(
            analyzer._players[sid].trades.trade_kill_opportunities
            for sid in [102, 103, 104, 105]
            if sid in analyzer._players
        )

        # With proximity: should be ~1 (only CT2 nearby)
        # Without proximity: would be 4 (all alive CTs)
        assert ct_opps <= 2, (
            f"Expected at most 2 CT trade opportunities (proximity-filtered), got {ct_opps}. "
            f"This suggests proximity filtering is not working."
        )

    def test_traded_death_only_when_nearby_teammate(self):
        """A death is only 'tradeable' if a teammate was within proximity."""
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        # CT2 nearby, so CT1's death should be tradeable
        demo_data = self._make_demo_with_positions(nearby_x=600.0, nearby_y=500.0)
        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        detect_trades(analyzer)

        ct1 = analyzer._players[101]
        assert ct1.trades.traded_death_opportunities >= 1, (
            f"CT1 death should be tradeable (CT2 is nearby), "
            f"got {ct1.trades.traded_death_opportunities}"
        )

    def test_trade_outside_window_not_counted(self):
        """Kill outside 5-second window should not be a trade."""
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        # Create scenario where the "trade" kill is 6 seconds after death
        # 6 seconds * 64 ticks = 384 ticks
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1000 + 384],
                "attacker_steamid": [201, 102],
                "user_steamid": [101, 201],
                "weapon": ["ak47", "m4a1"],
                "headshot": [True, False],
                "total_rounds_played": [1, 1],
                "attacker_team": [2, 3],
                "user_team": [3, 2],
                "user_X": [500.0, 600.0],
                "user_Y": [500.0, 500.0],
            }
        )

        demo_data = DemoData(
            file_path=Path("/tmp/test_window.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={101: "CT1", 102: "CT2", 201: "T1", 202: "T2"},
            player_teams={101: "CT", 102: "CT", 201: "T", 202: "T"},
            player_persistent_teams={101: "Team A", 102: "Team A", 201: "Team B", 202: "Team B"},
            team_rosters={"Team A": {101, 102}, "Team B": {201, 202}},
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

        detect_trades(analyzer)

        ct2 = analyzer._players[102]
        assert ct2.trades.trade_kill_success == 0, (
            f"Kill at 6s (outside 5s window) should NOT be a trade, "
            f"got {ct2.trades.trade_kill_success} successes"
        )

    def test_trade_within_window_counted(self):
        """Kill within 5-second window should be a trade."""
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        # 3 seconds * 64 ticks = 192 ticks (within window)
        kills_df = pd.DataFrame(
            {
                "tick": [1000, 1192],
                "attacker_steamid": [201, 102],
                "user_steamid": [101, 201],
                "weapon": ["ak47", "m4a1"],
                "headshot": [True, False],
                "total_rounds_played": [1, 1],
                "attacker_team": [2, 3],
                "user_team": [3, 2],
                "user_X": [500.0, 600.0],
                "user_Y": [500.0, 500.0],
            }
        )

        demo_data = DemoData(
            file_path=Path("/tmp/test_window2.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=1,
            player_stats={},
            player_names={101: "CT1", 102: "CT2", 201: "T1", 202: "T2"},
            player_teams={101: "CT", 102: "CT", 201: "T", 202: "T"},
            player_persistent_teams={101: "Team A", 102: "Team A", 201: "Team B", 202: "Team B"},
            team_rosters={"Team A": {101, 102}, "Team B": {201, 202}},
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

        detect_trades(analyzer)

        ct2 = analyzer._players[102]
        assert ct2.trades.trade_kill_success >= 1, (
            f"Kill at 3s (within 5s window) should be a trade, "
            f"got {ct2.trades.trade_kill_success} successes"
        )

    def test_no_ticks_df_falls_back_gracefully(self):
        """Without ticks_df, proximity filtering falls back to all teammates."""
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.analysis.compute_combat import detect_trades

        # Use existing make_demo_data_with_trades which has no ticks_df
        demo_data = make_demo_data_with_trades()
        analyzer = DemoAnalyzer(demo_data)
        analyzer._init_column_cache()
        analyzer._init_player_stats()

        detect_trades(analyzer)

        # Should still detect trades (fallback = all alive teammates)
        total_success = sum(p.trades.trade_kill_success for p in analyzer._players.values())
        assert total_success > 0, (
            "Without ticks_df, trade detection should still work (fallback to all teammates)"
        )

    def test_get_nearby_teammates_helper(self):
        """Direct unit test for the _get_nearby_teammates helper function."""
        import numpy as np

        from opensight.analysis.compute_combat import _get_nearby_teammates

        # Create ticks_df with known positions
        ticks_df = pd.DataFrame(
            {
                "steamid": np.array([102, 103, 104], dtype=np.int64),
                "tick": np.array([1000, 1000, 1000], dtype=np.int64),
                "X": [600.0, 5000.0, 400.0],
                "Y": [500.0, 5000.0, 500.0],
            }
        )

        teammates_alive = [102, 103, 104]
        death_x = 500.0
        death_y = 500.0
        kill_tick = 1000

        nearby = _get_nearby_teammates(
            teammates_alive,
            death_x,
            death_y,
            kill_tick,
            ticks_df,
            "steamid",
            proximity=1500.0,
        )

        # 102 is 100 units away (nearby), 103 is ~6364 units away (far),
        # 104 is 100 units away (nearby)
        assert 102 in nearby, "Player 102 (100 units away) should be nearby"
        assert 104 in nearby, "Player 104 (100 units away) should be nearby"
        assert 103 not in nearby, "Player 103 (6364 units away) should NOT be nearby"

    def test_get_nearby_teammates_no_ticks_df(self):
        """Helper returns all teammates when ticks_df is None."""
        from opensight.analysis.compute_combat import _get_nearby_teammates

        teammates = [102, 103, 104]
        result = _get_nearby_teammates(
            teammates,
            500.0,
            500.0,
            1000,
            None,
            None,
        )
        assert result == teammates, "Should return all teammates when ticks_df is None"

    def test_get_nearby_teammates_nan_position(self):
        """Helper returns all teammates when death position is NaN."""
        import numpy as np

        from opensight.analysis.compute_combat import _get_nearby_teammates

        ticks_df = pd.DataFrame(
            {
                "steamid": np.array([102], dtype=np.int64),
                "tick": np.array([1000], dtype=np.int64),
                "X": [600.0],
                "Y": [500.0],
            }
        )

        teammates = [102]
        result = _get_nearby_teammates(
            teammates,
            float("nan"),
            float("nan"),
            1000,
            ticks_df,
            "steamid",
        )
        assert result == teammates, "Should return all teammates when death position is NaN"
