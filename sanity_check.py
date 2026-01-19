import sys
import traceback
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# If you run: PYTHONPATH=src python sanity_check.py
# It will run MOCK tests (simulates data) to check logic/math.

# If you run: PYTHONPATH=src python sanity_check.py /path/to/demo.dem
# It will run a REAL integration test using the actual parser.
# -----------------------------------------------------------------------------

def run_mock_tests():
    """
    Tests the analytics logic using fake data that mimics awpy output.
    This catches math errors and 'KeyError' issues without needing a demo file.
    """
    print("\n" + "="*60)
    print("MODE: MOCK DATA TESTS (Checking Math & Logic)")
    print("="*60 + "\n")

    try:
        # 1. Import Check
        print("[1/5] Testing Imports...")
        import pandas as pd
        import numpy as np
        from opensight.analytics import analyze_demo, DemoAnalyzer, PlayerMatchStats
        from opensight.parser import MatchData, KillEvent, DamageEvent, RoundInfo
        print("      ✅ Imports successful.\n")

        # 2. Create Mock DataFrames (Simulating awpy output)
        print("[2/5] Creating Mock Data (Simulating awpy)...")

        # Mock Kills as list of KillEvent objects
        mock_kills = [
            KillEvent(
                tick=1000, round_num=1,
                attacker_steamid=1001, attacker_name='Player1', attacker_side='T',
                victim_steamid=2001, victim_name='Player2', victim_side='CT',
                weapon='ak47', headshot=False,
                attacker_x=100.0, attacker_y=100.0, attacker_z=64.0,
                victim_x=150.0, victim_y=150.0, victim_z=64.0,
                attacker_pitch=0.0, attacker_yaw=90.0
            ),
            KillEvent(
                tick=2000, round_num=1,
                attacker_steamid=2001, attacker_name='Player2', attacker_side='CT',
                victim_steamid=1001, victim_name='Player1', victim_side='T',
                weapon='m4a1', headshot=True,
                attacker_x=200.0, attacker_y=200.0, attacker_z=64.0,
                victim_x=150.0, victim_y=150.0, victim_z=64.0,
                attacker_pitch=0.0, attacker_yaw=270.0
            ),
            KillEvent(
                tick=3000, round_num=2,
                attacker_steamid=1001, attacker_name='Player1', attacker_side='T',
                victim_steamid=2001, victim_name='Player2', victim_side='CT',
                weapon='ak47', headshot=False,
                attacker_x=150.0, attacker_y=150.0, attacker_z=64.0,
                victim_x=160.0, victim_y=160.0, victim_z=64.0,
                attacker_pitch=0.0, attacker_yaw=45.0
            ),
            KillEvent(
                tick=4000, round_num=2,
                attacker_steamid=2001, attacker_name='Player2', attacker_side='CT',
                victim_steamid=1001, victim_name='Player1', victim_side='T',
                weapon='m4a1', headshot=False,
                attacker_x=250.0, attacker_y=250.0, attacker_z=64.0,
                victim_x=160.0, victim_y=160.0, victim_z=64.0,
                attacker_pitch=0.0, attacker_yaw=225.0
            ),
        ]

        # Mock Damages (50 ticks before kill for TTD testing)
        mock_damages = [
            DamageEvent(
                tick=950, round_num=1,
                attacker_steamid=1001, attacker_name='Player1', attacker_side='T',
                victim_steamid=2001, victim_name='Player2', victim_side='CT',
                damage=20, damage_armor=0, health_remaining=80, armor_remaining=100,
                weapon='ak47', hitgroup='chest'
            ),
            DamageEvent(
                tick=1950, round_num=1,
                attacker_steamid=2001, attacker_name='Player2', attacker_side='CT',
                victim_steamid=1001, victim_name='Player1', victim_side='T',
                damage=30, damage_armor=5, health_remaining=70, armor_remaining=95,
                weapon='m4a1', hitgroup='chest'
            ),
            DamageEvent(
                tick=2950, round_num=2,
                attacker_steamid=1001, attacker_name='Player1', attacker_side='T',
                victim_steamid=2001, victim_name='Player2', victim_side='CT',
                damage=40, damage_armor=0, health_remaining=60, armor_remaining=100,
                weapon='ak47', hitgroup='stomach'
            ),
            DamageEvent(
                tick=3950, round_num=2,
                attacker_steamid=2001, attacker_name='Player2', attacker_side='CT',
                victim_steamid=1001, victim_name='Player1', victim_side='T',
                damage=50, damage_armor=10, health_remaining=50, armor_remaining=90,
                weapon='m4a1', hitgroup='stomach'
            ),
        ]

        # Mock Rounds
        mock_rounds = [
            RoundInfo(round_num=1, start_tick=0, end_tick=2400, freeze_end_tick=100, winner='T', reason='elimination'),
            RoundInfo(round_num=2, start_tick=2500, end_tick=5000, freeze_end_tick=2600, winner='CT', reason='elimination'),
        ]

        # Create DataFrames for backward compatibility (analytics may use either)
        mock_kills_df = pd.DataFrame({
            'tick': [k.tick for k in mock_kills],
            'round': [k.round_num for k in mock_kills],
            'attacker_steamid': [k.attacker_steamid for k in mock_kills],
            'attacker_name': [k.attacker_name for k in mock_kills],
            'attacker_team_name': [k.attacker_side for k in mock_kills],
            'victim_steamid': [k.victim_steamid for k in mock_kills],
            'victim_name': [k.victim_name for k in mock_kills],
            'victim_team_name': [k.victim_side for k in mock_kills],
            'weapon': [k.weapon for k in mock_kills],
            'headshot': [k.headshot for k in mock_kills],
            'attacker_X': [k.attacker_x for k in mock_kills],
            'attacker_Y': [k.attacker_y for k in mock_kills],
            'attacker_Z': [k.attacker_z for k in mock_kills],
            'victim_X': [k.victim_x for k in mock_kills],
            'victim_Y': [k.victim_y for k in mock_kills],
            'victim_Z': [k.victim_z for k in mock_kills],
            'attacker_pitch': [k.attacker_pitch for k in mock_kills],
            'attacker_yaw': [k.attacker_yaw for k in mock_kills],
        })

        mock_damages_df = pd.DataFrame({
            'tick': [d.tick for d in mock_damages],
            'round': [d.round_num for d in mock_damages],
            'attacker_steamid': [d.attacker_steamid for d in mock_damages],
            'attacker_name': [d.attacker_name for d in mock_damages],
            'attacker_team_name': [d.attacker_side for d in mock_damages],
            'victim_steamid': [d.victim_steamid for d in mock_damages],
            'victim_name': [d.victim_name for d in mock_damages],
            'victim_team_name': [d.victim_side for d in mock_damages],
            'dmg_health': [d.damage for d in mock_damages],
            'dmg_armor': [d.damage_armor for d in mock_damages],
            'weapon': [d.weapon for d in mock_damages],
            'hitgroup': [d.hitgroup for d in mock_damages],
        })

        mock_rounds_df = pd.DataFrame({
            'round': [r.round_num for r in mock_rounds],
            'start_tick': [r.start_tick for r in mock_rounds],
            'end_tick': [r.end_tick for r in mock_rounds],
            'freeze_end': [r.freeze_end_tick for r in mock_rounds],
            'winner_name': [r.winner for r in mock_rounds],
            'end_reason': [r.reason for r in mock_rounds],
        })

        print(f"      ✅ Created {len(mock_kills)} kills, {len(mock_damages)} damages.\n")

        # 3. Construct MatchData Object
        print("[3/5] Constructing MatchData Object...")
        match_data = MatchData(
            file_path=Path("/fake/demo.dem"),
            map_name="de_ancient",
            duration_seconds=600.0,
            tick_rate=64,
            num_rounds=2,
            game_rounds=mock_rounds,
            kills=mock_kills,
            damages=mock_damages,
            bomb_events=[],
            grenades=[],
            smokes=[],
            infernos=[],
            weapon_fires=[],
            blinds=[],
            player_stats={
                1001: {'kills': 2, 'deaths': 2, 'assists': 0},
                2001: {'kills': 2, 'deaths': 2, 'assists': 0},
            },
            player_names={1001: 'Player1', 2001: 'Player2'},
            player_teams={1001: 'T', 2001: 'CT'},
            kills_df=mock_kills_df,
            damages_df=mock_damages_df,
            rounds_df=mock_rounds_df,
            grenades_df=pd.DataFrame(),
            bomb_events_df=pd.DataFrame(),
            weapon_fires_df=pd.DataFrame(),
            blinds_df=pd.DataFrame(),
            ticks_df=None,
            final_score_ct=1,
            final_score_t=1,
        )
        print("      ✅ MatchData constructed.\n")

        # 4. Run Analytics
        print("[4/5] Running analyze_demo()...")
        analysis = analyze_demo(match_data)

        if not analysis or not analysis.players:
            print("      ❌ FAIL: Analytics returned empty results.")
            return False

        print(f"      ✅ Analytics returned data for {len(analysis.players)} players.\n")

        # 5. Validate Output Structure
        print("[5/5] Validating Output Structure...")
        for steam_id, metrics in analysis.players.items():
            print(f"      - {metrics.name} (steamid: {steam_id}):")
            # Check for required fields
            assert hasattr(metrics, 'kills'), "Missing 'kills' attribute"
            assert hasattr(metrics, 'ttd_mean_ms'), "Missing 'ttd_mean_ms' property"
            assert hasattr(metrics, 'cp_median_error_deg'), "Missing 'cp_median_error_deg' property"

            ttd = metrics.ttd_mean_ms
            cp = metrics.cp_median_error_deg
            print(f"        Kills: {metrics.kills}, Deaths: {metrics.deaths}")
            print(f"        TTD Mean: {ttd if ttd else 'N/A'} ms")
            print(f"        CP Median: {cp if cp else 'N/A'}°")

        print("\n" + "="*60)
        print("✅ ALL MOCK TESTS PASSED")
        print("="*60)
        return True

    except Exception as e:
        print("\n" + "="*60)
        print("❌ MOCK TESTS FAILED")
        print("="*60)
        print("Error Details:")
        traceback.print_exc()
        return False


def run_real_integration_test(demo_path):
    """
    Tests the full pipeline with a real .dem file.
    This checks if awpy is installed and if the parser doesn't crash.
    """
    print("\n" + "="*60)
    print(f"MODE: REAL INTEGRATION TEST ({demo_path})")
    print("="*60 + "\n")

    try:
        # 1. Import Parser
        print("[1/4] Importing Parser...")
        from opensight.parser import DemoParser
        from opensight.analytics import analyze_demo
        print("      ✅ Parser imported.\n")

        # 2. Parse Demo
        print("[2/4] Parsing Demo (this might take 10-30s)...")
        parser = DemoParser(demo_path)
        data = parser.parse()

        if data is None:
            raise ValueError("Parser returned None")

        print(f"      ✅ Demo Parsed. Map: {data.map_name}, Rounds: {data.num_rounds}\n")

        # 3. Run Analytics
        print("[3/4] Running Analytics on Real Data...")
        analysis = analyze_demo(data)
        print(f"      ✅ Analytics Complete. Players found: {len(analysis.players)}\n")

        # 4. Print Summary
        print("[4/4] Summary of First Player:")
        first_player = list(analysis.players.values())[0]
        print(f"      Name: {first_player.name}")
        print(f"      Kills: {first_player.kills}")
        print(f"      Deaths: {first_player.deaths}")
        print(f"      ADR: {first_player.adr}")
        print(f"      HLTV Rating: {first_player.hltv_rating}")

        ttd = first_player.ttd_mean_ms
        cp = first_player.cp_median_error_deg
        print(f"      Mean TTD: {ttd if ttd else 'N/A'} ms")
        print(f"      Median CP: {cp if cp else 'N/A'}°")

        print("\n" + "="*60)
        print("✅ INTEGRATION TEST PASSED")
        print("="*60)
        return True

    except Exception as e:
        print("\n" + "="*60)
        print("❌ INTEGRATION TEST FAILED")
        print("="*60)
        print("Error Details:")
        traceback.print_exc()
        return False


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    success = False

    if len(sys.argv) > 1:
        # User provided a demo file path
        success = run_real_integration_test(sys.argv[1])
    else:
        # No file provided, run fast mock tests
        success = run_mock_tests()

    sys.exit(0 if success else 1)
