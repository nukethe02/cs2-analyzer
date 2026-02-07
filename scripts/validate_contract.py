#!/usr/bin/env python3
"""
Runtime contract validator — parse a real demo and check output matches contract.

Usage:
    PYTHONPATH=src python scripts/validate_contract.py path/to/demo.dem

Run this before every push to catch field mismatches early.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_contract.py <demo.dem>")
        print("  Parses a real demo, runs full pipeline, validates output against contract.")
        return 1

    demo_path = Path(sys.argv[1])
    if not demo_path.exists():
        print(f"ERROR: Demo file not found: {demo_path}")
        return 1

    print(f"Parsing {demo_path.name}...")
    from opensight.pipeline.orchestrator import DemoOrchestrator

    orchestrator = DemoOrchestrator()
    result = orchestrator.analyze(demo_path)

    print(f"Analysis complete: {result['demo_info']['map']}, "
          f"{result['demo_info']['rounds']} rounds, "
          f"{len(result['players'])} players")

    from opensight.pipeline.contract import validate_result

    errors = validate_result(result)

    if errors:
        print(f"\nCONTRACT VIOLATIONS ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print(f"\n✓ Contract validated — all fields present and correctly typed")
    print(f"  Players: {len(result['players'])}")
    print(f"  Timeline rounds: {len(result.get('round_timeline', []))}")

    # Print a sample player for visual inspection
    first_player = next(iter(result["players"].values()))
    print(f"\n  Sample player: {first_player['name']}")
    print(f"    stats.kills={first_player['stats']['kills']}")
    print(f"    rating.hltv_rating={first_player['rating']['hltv_rating']}")
    print(f"    advanced.opening_kills={first_player['advanced']['opening_kills']}")
    print(f"    trades.trade_kill_success={first_player['trades']['trade_kill_success']}")
    print(f"    clutches.total_situations={first_player['clutches']['total_situations']}")
    print(f"    duels.opening_win_rate={first_player['duels']['opening_win_rate']}")
    print(f"    utility.he_team_damage={first_player['utility']['he_team_damage']}")
    print(f"    utility.unused_utility_value={first_player['utility']['unused_utility_value']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
