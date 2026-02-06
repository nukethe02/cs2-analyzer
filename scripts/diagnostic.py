#!/usr/bin/env python3
"""
Diagnostic script: Test demo parsing outside the web app.

This script validates that demoparser2 can properly parse CS2 demo files
and extract all necessary data for analysis.

Run with: python scripts/diagnostic.py path/to/demo.dem

Exit codes:
    0 - All tests passed
    1 - Some tests failed
    2 - Critical failure (file not found, demoparser2 not installed)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnostic.py <path/to/demo.dem>")
        print("\nThis script tests demo parsing to diagnose issues.")
        sys.exit(2)

    demo_path = Path(sys.argv[1])
    if not demo_path.exists():
        print(f"ERROR: Demo file not found: {demo_path}")
        sys.exit(2)

    print("=== OpenSight Demo Diagnostic ===")
    print(f"Demo file: {demo_path}")
    print(f"File size: {demo_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # Test 1: Can demoparser2 load the file?
    print("=== TEST 1: demoparser2 Import ===")
    try:
        from demoparser2 import DemoParser as DP2

        print("OK: demoparser2 is installed")
    except ImportError as e:
        print(f"FAIL: demoparser2 not installed: {e}")
        print("Install with: pip install demoparser2")
        sys.exit(2)

    # Test 2: Can we create a parser instance?
    print("\n=== TEST 2: Parser Initialization ===")
    try:
        parser = DP2(str(demo_path))
        print("OK: Parser initialized successfully")
    except Exception as e:
        print(f"FAIL: Could not initialize parser: {e}")
        sys.exit(2)

    # Test 3: Parse header
    print("\n=== TEST 3: Demo Header ===")
    try:
        header = parser.parse_header()
        if isinstance(header, dict):
            print(f"  Map: {header.get('map_name', 'unknown')}")
            print(f"  Server: {header.get('server_name', 'unknown')}")
            print(f"  Duration: {header.get('playback_time', 0):.1f} seconds")
        else:
            print(f"  Header type: {type(header)}")
        print("OK: Header parsed")
    except Exception as e:
        print(f"WARN: Could not parse header: {e}")

    # Test 4: Parse kills (critical)
    print("\n=== TEST 4: Kill Events (CRITICAL) ===")
    try:
        kills = parser.parse_event("player_death")
        if kills is not None and len(kills) > 0:
            print(f"  Found {len(kills)} kills")
            print(f"  Columns: {list(kills.columns)[:10]}...")
            print("\n  First 3 kills:")
            for i, row in kills.head(3).iterrows():
                attacker = row.get("attacker_name", "Unknown")
                victim = row.get("user_name", "Unknown")
                weapon = row.get("weapon", "Unknown")
                headshot = row.get("headshot", False)
                hs_marker = " (HS)" if headshot else ""
                print(f"    {attacker} -> {victim} [{weapon}]{hs_marker}")
            print("OK: Kills parsed successfully")
        else:
            print("FAIL: No kills found - demo may be corrupted or warmup only")
    except Exception as e:
        print(f"FAIL: Could not parse kills: {e}")

    # Test 5: Parse rounds (critical)
    print("\n=== TEST 5: Round Events (CRITICAL) ===")
    try:
        rounds = parser.parse_event("round_end")
        if rounds is not None and len(rounds) > 0:
            print(f"  Found {len(rounds)} rounds")
            print(f"  Columns: {list(rounds.columns)}")
            print("OK: Rounds parsed successfully")
        else:
            print("FAIL: No round_end events found")
    except Exception as e:
        print(f"FAIL: Could not parse rounds: {e}")

    # Test 6: Parse damage events
    print("\n=== TEST 6: Damage Events ===")
    try:
        damage = parser.parse_event("player_hurt")
        if damage is not None and len(damage) > 0:
            print(f"  Found {len(damage)} damage events")
            total_dmg = damage["dmg_health"].sum() if "dmg_health" in damage.columns else 0
            print(f"  Total damage dealt: {total_dmg}")
            print("OK: Damage events parsed")
        else:
            print("WARN: No damage events found")
    except Exception as e:
        print(f"WARN: Could not parse damage events: {e}")

    # Test 7: Parse economy data
    print("\n=== TEST 7: Economy Data ===")
    try:
        econ = parser.parse_event(
            "round_freeze_end", player=["balance", "current_equip_value", "team_num"]
        )
        if econ is not None and len(econ) > 0:
            print(f"  Got {len(econ)} economy records")
            print(f"  Columns: {list(econ.columns)[:10]}...")
            print("OK: Economy data available")
        else:
            print("WARN: No economy data (round_freeze_end with player props)")
    except Exception as e:
        print(f"WARN: Could not parse economy data: {e}")

    # Test 8: Parse grenade events
    print("\n=== TEST 8: Grenade Events ===")
    grenade_counts = {}
    for event_name, label in [
        ("smokegrenade_detonate", "smokes"),
        ("flashbang_detonate", "flashes"),
        ("hegrenade_detonate", "HE grenades"),
        ("inferno_startburn", "molotovs"),
    ]:
        try:
            grenades = parser.parse_event(event_name)
            count = len(grenades) if grenades is not None else 0
            grenade_counts[label] = count
        except Exception:
            grenade_counts[label] = 0

    total = sum(grenade_counts.values())
    print(f"  Smokes: {grenade_counts['smokes']}")
    print(f"  Flashes: {grenade_counts['flashes']}")
    print(f"  HE grenades: {grenade_counts['HE grenades']}")
    print(f"  Molotovs: {grenade_counts['molotovs']}")
    if total > 0:
        print(f"OK: {total} total grenade events")
    else:
        print("WARN: No grenade events found")

    # Test 9: Parse player positions (first 100 ticks)
    print("\n=== TEST 9: Player Positions (tick data) ===")
    try:
        ticks = parser.parse_ticks(["X", "Y", "Z", "health", "team_num"], ticks=list(range(100)))
        if ticks is not None and len(ticks) > 0:
            print(f"  Got {len(ticks)} position records (first 100 ticks)")
            unique_players = ticks["steamid"].nunique() if "steamid" in ticks.columns else 0
            print(f"  Unique players: {unique_players}")
            print("OK: Tick data available")
        else:
            print("WARN: No tick data available")
    except Exception as e:
        print(f"WARN: Tick parsing failed: {e}")

    # Test 10: Flash blind events
    print("\n=== TEST 10: Flash Blind Events ===")
    try:
        blinds = parser.parse_event("player_blind")
        if blinds is not None and len(blinds) > 0:
            print(f"  Found {len(blinds)} blind events")
            if "blind_duration" in blinds.columns:
                avg_duration = blinds["blind_duration"].mean()
                print(f"  Average blind duration: {avg_duration:.2f}s")
            print("OK: Blind events parsed")
        else:
            print("WARN: No blind events found")
    except Exception as e:
        print(f"WARN: Could not parse blind events: {e}")

    # Test 11: Bomb events
    print("\n=== TEST 11: Bomb Events ===")
    bomb_counts = {}
    for event_name, label in [
        ("bomb_planted", "plants"),
        ("bomb_defused", "defuses"),
        ("bomb_exploded", "explosions"),
    ]:
        try:
            events = parser.parse_event(event_name)
            count = len(events) if events is not None else 0
            bomb_counts[label] = count
        except Exception:
            bomb_counts[label] = 0

    print(f"  Plants: {bomb_counts['plants']}")
    print(f"  Defuses: {bomb_counts['defuses']}")
    print(f"  Explosions: {bomb_counts['explosions']}")
    if sum(bomb_counts.values()) > 0:
        print("OK: Bomb events found")
    else:
        print("WARN: No bomb events (may be hostage map or warmup)")

    # Summary
    print("\n" + "=" * 50)
    print("=== DIAGNOSTIC SUMMARY ===")
    print("=" * 50)

    # Check critical metrics
    critical_passed = True
    try:
        kills = parser.parse_event("player_death")
        rounds = parser.parse_event("round_end")
        if kills is None or len(kills) == 0:
            print("CRITICAL: No kills found")
            critical_passed = False
        if rounds is None or len(rounds) == 0:
            print("CRITICAL: No rounds found")
            critical_passed = False
    except Exception:
        critical_passed = False

    if critical_passed:
        print("\nRESULT: Demo parsing is working correctly!")
        print("All critical data (kills, rounds) is available.")
        sys.exit(0)
    else:
        print("\nRESULT: Demo parsing has issues.")
        print("Check the warnings above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
