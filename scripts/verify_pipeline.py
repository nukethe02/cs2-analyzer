#!/usr/bin/env python3
"""
Verify the full parse → analyze pipeline works end-to-end.

This script tests the complete OpenSight analysis pipeline:
1. DemoParser parses the demo file
2. DemoAnalyzer computes metrics
3. CachedAnalyzer orchestrates everything
4. Final result has valid, non-zero data

Run with: python scripts/verify_pipeline.py [path/to/demo.dem]

If no demo is provided, it will look for test demos in test_demos/ directory.

Exit codes:
    0 - Pipeline working correctly
    1 - Pipeline has issues (check output for details)
    2 - Critical failure (missing dependencies, no demo file)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def find_test_demo() -> Path | None:
    """Find a test demo file."""
    project_root = Path(__file__).parent.parent

    # Check common locations
    locations = [
        project_root / "test_demos",
        project_root / "tests" / "fixtures",
        project_root / "demos",
        Path.home() / ".opensight" / "demos",
    ]

    for loc in locations:
        if loc.exists():
            demos = list(loc.glob("*.dem")) + list(loc.glob("*.dem.gz"))
            if demos:
                return demos[0]

    return None


def verify_parser(demo_path: Path) -> dict | None:
    """Test the DemoParser directly."""
    print("\n=== Step 1: DemoParser ===")

    try:
        from opensight.core.parser import DemoParser

        parser = DemoParser(demo_path)
        demo_data = parser.parse()

        print(f"  Map: {demo_data.map_name}")
        print(f"  Rounds: {demo_data.num_rounds}")
        print(f"  Players: {len(demo_data.player_names)}")
        print(f"  Kills: {len(demo_data.kills)}")
        print(f"  Duration: {demo_data.duration_seconds:.0f}s")

        # Validate critical fields
        issues = []
        if demo_data.num_rounds == 0:
            issues.append("num_rounds is 0")
        if len(demo_data.kills) == 0:
            issues.append("No kills parsed")
        if len(demo_data.player_names) == 0:
            issues.append("No players found")

        if issues:
            print(f"  ISSUES: {', '.join(issues)}")
            return None

        print("  OK: Parser working correctly")
        return {
            "demo_data": demo_data,
            "map": demo_data.map_name,
            "rounds": demo_data.num_rounds,
            "players": len(demo_data.player_names),
            "kills": len(demo_data.kills),
        }
    except Exception as e:
        print(f"  FAIL: Parser error: {e}")
        import traceback

        traceback.print_exc()
        return None


def verify_analyzer(demo_data) -> dict | None:
    """Test the DemoAnalyzer."""
    print("\n=== Step 2: DemoAnalyzer ===")

    try:
        from opensight.analysis.analytics import DemoAnalyzer

        analyzer = DemoAnalyzer(demo_data)
        analysis = analyzer.analyze()

        print(f"  Players analyzed: {len(analysis.players)}")

        # Check player stats
        zero_stat_players = []
        valid_players = 0

        for _steam_id, player in analysis.players.items():
            if player.kills == 0 and player.deaths == 0:
                zero_stat_players.append(player.name)
            else:
                valid_players += 1
                if valid_players <= 3:
                    print(
                        f"    {player.name}: {player.kills}K/{player.deaths}D, ADR: {player.adr:.1f}, Rating: {player.hltv_rating:.2f}"
                    )

        if zero_stat_players:
            print(
                f"  WARN: {len(zero_stat_players)} players with 0K/0D: {zero_stat_players[:3]}..."
            )

        if valid_players == 0:
            print("  FAIL: All players have 0 kills AND 0 deaths")
            return None

        print(f"  OK: {valid_players} players with valid stats")
        return {
            "analysis": analysis,
            "valid_players": valid_players,
        }
    except Exception as e:
        print(f"  FAIL: Analyzer error: {e}")
        import traceback

        traceback.print_exc()
        return None


def verify_cached_analyzer(demo_path: Path) -> dict | None:
    """Test the full CachedAnalyzer pipeline."""
    print("\n=== Step 3: CachedAnalyzer (Full Pipeline) ===")

    try:
        from opensight.infra.cache import CachedAnalyzer

        cached = CachedAnalyzer()
        result = cached.analyze(demo_path, force=True)  # Force fresh analysis

        if not result:
            print("  FAIL: CachedAnalyzer returned None/empty")
            return None

        # Check critical fields in result
        players = result.get("players", {})
        round_timeline = result.get("round_timeline", [])
        match_info = result.get("match_info", {})

        print(f"  Players in result: {len(players)}")
        print(f"  Rounds in timeline: {len(round_timeline)}")
        print(f"  Map: {match_info.get('map', 'unknown')}")

        # Validate player data
        issues = []
        zero_stat_count = 0

        for _steam_id, player in players.items():
            stats = player.get("stats", {})
            kills = stats.get("kills", 0)
            deaths = stats.get("deaths", 0)

            if kills == 0 and deaths == 0:
                zero_stat_count += 1

        if zero_stat_count > 0:
            issues.append(f"{zero_stat_count} players with 0K/0D")
        if len(round_timeline) == 0:
            issues.append("No round timeline")

        if issues:
            print(f"  ISSUES: {', '.join(issues)}")

        # Print sample player stats
        print("\n  Sample player stats:")
        for i, (_steam_id, player) in enumerate(players.items()):
            if i >= 5:
                break
            stats = player.get("stats", {})
            rating = player.get("rating", {})
            print(
                f"    {player.get('name', 'Unknown')}: {stats.get('kills', 0)}K/{stats.get('deaths', 0)}D, ADR: {stats.get('adr', 0)}, Rating: {rating.get('hltv_rating', 0)}"
            )

        # Check for zero-stat issues (the main bug we're looking for)
        valid_player_count = len(players) - zero_stat_count
        if valid_player_count == 0:
            print("  FAIL: All players have 0 kills AND 0 deaths - data is broken!")
            return None

        print(f"\n  OK: CachedAnalyzer pipeline working ({valid_player_count} valid players)")
        return {
            "result": result,
            "players": len(players),
            "rounds": len(round_timeline),
            "valid_players": valid_player_count,
        }
    except Exception as e:
        print(f"  FAIL: CachedAnalyzer error: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("OpenSight Pipeline Verification")
    print("=" * 60)

    # Find demo file
    if len(sys.argv) > 1:
        demo_path = Path(sys.argv[1])
    else:
        demo_path = find_test_demo()
        if demo_path is None:
            print("\nERROR: No demo file specified and no test demos found.")
            print("\nUsage: python scripts/verify_pipeline.py <path/to/demo.dem>")
            print("\nOr place demo files in:")
            print("  - test_demos/")
            print("  - tests/fixtures/")
            sys.exit(2)

    if not demo_path.exists():
        print(f"\nERROR: Demo file not found: {demo_path}")
        sys.exit(2)

    print(f"\nDemo file: {demo_path}")
    print(f"File size: {demo_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Run verification steps
    results = {}

    # Step 1: Parser
    parser_result = verify_parser(demo_path)
    results["parser"] = parser_result is not None

    # Step 2: Analyzer (if parser worked)
    if parser_result:
        analyzer_result = verify_analyzer(parser_result["demo_data"])
        results["analyzer"] = analyzer_result is not None
    else:
        results["analyzer"] = False

    # Step 3: CachedAnalyzer (full pipeline)
    cached_result = verify_cached_analyzer(demo_path)
    results["cached_analyzer"] = cached_result is not None

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for step, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {step}: {status}")

    if all_passed:
        print("\n SUCCESS: Full pipeline is working correctly!")
        print("Demo parsing and analysis are producing valid data.")
        sys.exit(0)
    else:
        print("\n ISSUES DETECTED: See above for details.")
        print("Some parts of the pipeline are not working correctly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
