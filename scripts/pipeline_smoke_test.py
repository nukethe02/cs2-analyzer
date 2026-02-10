#!/usr/bin/env python3
"""
Pipeline Smoke Test Script.

USAGE:
    python scripts/pipeline_smoke_test.py path/to/demo.dem

PURPOSE:
Manual smoke test to verify the full pipeline works end-to-end after code changes.
This is faster than running the full test suite and provides immediate visual feedback.

WHAT IT CHECKS:
1. Field Coverage - Which metrics have data vs null/zero
2. Name Resolution - Are there raw steamids where player names should be?
3. Timeline - Do both teams have kill data?
4. Highlights - Are multi-kills/clutches detected?

OUTPUT:
Colored terminal output with ✅ PASS / ⚠️  PARTIAL / ❌ FAIL indicators.
Exit code 0 if all critical checks pass, 1 if any fail.

Author: Created as part of Preventive Infrastructure (Prompt 1)
"""

import sys
from pathlib import Path
from typing import Any


# =============================================================================
# COLOR CODES (ANSI)
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.GREEN}{text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.BLUE}{text}{Colors.RESET}"

    @staticmethod
    def bold(text: str) -> str:
        return f"{Colors.BOLD}{text}{Colors.RESET}"


# =============================================================================
# SMOKE TEST RUNNER
# =============================================================================

def run_smoke_test(demo_path: Path) -> int:
    """
    Run full pipeline on demo and report field coverage.
    Returns 0 if all critical checks pass, 1 if any fail.
    """
    if not demo_path.exists():
        print(Colors.error(f"❌ Demo file not found: {demo_path}"))
        return 1

    print(Colors.bold("╔══════════════════════════════════════════════════════════════╗"))
    print(Colors.bold("║                  PIPELINE SMOKE TEST                        ║"))
    print(Colors.bold("╠══════════════════════════════════════════════════════════════╣"))
    print(f"║ Demo: {demo_path.name:<54} ║")

    # Run the pipeline
    try:
        from opensight.pipeline.orchestrator import DemoOrchestrator

        orchestrator = DemoOrchestrator()
        print(Colors.info("║ Running analysis pipeline...                                ║"))
        result = orchestrator.analyze(demo_path)
    except Exception as e:
        print(Colors.bold("╚══════════════════════════════════════════════════════════════╝"))
        print(Colors.error(f"\n❌ PIPELINE FAILED: {e}"))
        import traceback
        traceback.print_exc()
        return 1

    players = result.get("players", {})
    demo_info = result.get("demo_info", {})
    timeline = result.get("timeline_graph", {})
    highlights = result.get("highlights", [])

    print(f"║ Players: {len(players):<2}  |  Rounds: {demo_info.get('rounds', 0):<2}  |  Total Kills: {demo_info.get('total_kills', 0):<4}       ║")
    print(Colors.bold("╠══════════════════════════════════════════════════════════════╣"))

    # Check field coverage
    print(Colors.bold("FIELD COVERAGE (per player):"))
    print("┌─────────────────────────────┬───────┬───────────┬───────────────────┐")
    print("│ Field                       │ Has   │ Has Not   │ Status            │")
    print("├─────────────────────────────┼───────┼───────────┼───────────────────┤")

    fields_to_check = [
        ("stats.kills", True),
        ("stats.deaths", True),
        ("stats.adr", True),
        ("rating.hltv_rating", True),
        ("rating.kast_percentage", True),
        ("rws.avg_rws", True),
        ("advanced.ttd_median_ms", False),
        ("advanced.cp_median_error_deg", False),
        ("utility.flash_assist_pct", False),
        ("stats.2k", False),
        ("duels.trade_kills", False),
        ("duels.clutch_wins", False),
        ("economy.avg_equipment_value", True),
        ("discipline.discipline_rating", True),
    ]

    failures = []
    for field_path, is_critical in fields_to_check:
        parts = field_path.split(".")
        has_count = 0
        for player in players.values():
            value = player
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            if value is not None and value != 0:
                has_count += 1
            elif is_critical and (value is None or value == 0):
                # For critical fields, even 0 counts if it's not None
                if value == 0:
                    has_count += 1

        total = len(players)
        has_not = total - has_count
        percentage = (has_count / total * 100) if total > 0 else 0

        # Determine status
        if is_critical and percentage < 100:
            status = Colors.error("❌ FAIL")
            failures.append(field_path)
        elif percentage == 0:
            status = Colors.error("❌ FAIL")
            failures.append(field_path)
        elif percentage < 50:
            status = Colors.warning(f"⚠️  PARTIAL ({percentage:.0f}%)")
        else:
            status = Colors.success("✅ PASS")

        # Format field name (truncate if too long)
        field_display = field_path[:27] if len(field_path) <= 27 else field_path[:24] + "..."

        print(f"│ {field_display:<27} │ {has_count:>2}/{total:<2} │ {has_not:>2}/{total:<2}     │ {status:<25} │")

    print("└─────────────────────────────┴───────┴───────────┴───────────────────┘")

    # Check name resolution
    print(Colors.bold("\nNAME RESOLUTION:"))
    print("┌─────────────────────────────┬───────────────────────────────────────┐")
    print("│ Check                       │ Status                                │")
    print("├─────────────────────────────┼───────────────────────────────────────┤")

    # Check player names
    raw_steamid_names = []
    for steam_id, player in players.items():
        name = player.get("name", "")
        if name.isdigit() and len(name) > 10:
            raw_steamid_names.append(name)

    if len(raw_steamid_names) == 0:
        status = Colors.success("✅ None found in player names")
    else:
        status = Colors.error(f"❌ {len(raw_steamid_names)} raw SteamID(s)")
        failures.append("player_names_contain_steamids")

    print(f"│ Raw SteamIDs                │ {status:<45} │")

    # Check kill matrix (it's a list of kill events)
    kill_matrix = result.get("kill_matrix", [])
    raw_in_matrix = []
    if isinstance(kill_matrix, list):
        for kill in kill_matrix:
            victim = kill.get("victim", "")
            killer = kill.get("killer", "")
            if victim and victim.isdigit() and len(victim) > 10:
                raw_in_matrix.append(victim)
            if killer and killer.isdigit() and len(killer) > 10:
                raw_in_matrix.append(killer)

    if len(raw_in_matrix) == 0:
        status = Colors.success("✅ None found in kill matrix")
    else:
        status = Colors.error(f"❌ {len(set(raw_in_matrix))} raw SteamID(s)")
        failures.append("kill_matrix_contains_steamids")

    print(f"│ Kill Matrix names           │ {status:<45} │")
    print("└─────────────────────────────┴───────────────────────────────────────┘")

    # Check timeline (uses players and round_scores)
    print(Colors.bold("\nTIMELINE:"))
    print("┌─────────────────────────────┬───────────────────────────────────────┐")
    print("│ Team                        │ Data                                  │")
    print("├─────────────────────────────┼───────────────────────────────────────┤")

    players_data = timeline.get("players", [])
    round_scores = timeline.get("round_scores", [])

    ct_players = [p for p in players_data if p.get("team") == "CT"]
    t_players = [p for p in players_data if p.get("team") == "T"]
    ct_kills = sum(r.get("ct_kills", 0) for r in round_scores)
    t_kills = sum(r.get("t_kills", 0) for r in round_scores)

    if len(ct_players) == 0:
        ct_status = Colors.error("❌ No CT players")
        failures.append("timeline_ct_no_players")
    elif ct_kills == 0:
        ct_status = Colors.error(f"❌ {len(ct_players)} players but ZERO kills")
        failures.append("timeline_ct_zero_kills")
    else:
        ct_status = Colors.success(f"{len(ct_players)} players, {ct_kills} kills")

    if len(t_players) == 0:
        t_status = Colors.error("❌ No T players")
        failures.append("timeline_t_no_players")
    elif t_kills == 0:
        t_status = Colors.error(f"❌ {len(t_players)} players but ZERO kills")
        failures.append("timeline_t_zero_kills")
    else:
        t_status = Colors.success(f"{len(t_players)} players, {t_kills} kills")

    print(f"│ CT                          │ {ct_status:<45} │")
    print(f"│ T                           │ {t_status:<45} │")
    print("└─────────────────────────────┴───────────────────────────────────────┘")

    # Check highlights
    print(Colors.bold("\nHIGHLIGHTS:"))
    print("┌─────────────────────────────┬───────────────────────────────────────┐")
    print("│ Type                        │ Count                                 │")
    print("├─────────────────────────────┼───────────────────────────────────────┤")

    multikills = [h for h in highlights if "multi" in h.get("highlight_type", "").lower()]
    clutches = [h for h in highlights if "clutch" in h.get("highlight_type", "").lower()]
    special = [h for h in highlights if h.get("highlight_type", "") in ["wallbang", "noscope", "collateral"]]

    rounds = demo_info.get("rounds", 0)
    if len(multikills) == 0 and rounds > 15:
        mk_status = Colors.warning(f"0 (expected: >0 in {rounds}-round match)")
    else:
        mk_status = Colors.success(str(len(multikills)))

    print(f"│ Multi-kills                 │ {mk_status:<45} │")
    print(f"│ Clutches                    │ {Colors.success(str(len(clutches))):<45} │")
    print(f"│ Special kills               │ {Colors.success(str(len(special))):<45} │")
    print("└─────────────────────────────┴───────────────────────────────────────┘")

    # Summary
    total_checks = len(fields_to_check) + 6  # fields + name checks (2) + timeline checks (3) + highlights (1)
    passed = total_checks - len(failures)
    print(Colors.bold(f"\nSUMMARY: {passed}/{total_checks} checks passed"))

    if len(failures) > 0:
        print(Colors.error(f"❌ {len(failures)} check(s) failed:"))
        for failure in failures:
            print(f"   - {failure}")
        return 1
    else:
        print(Colors.success("✅ All checks passed!"))
        return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/pipeline_smoke_test.py path/to/demo.dem")
        return 1

    demo_path = Path(sys.argv[1])
    exit_code = run_smoke_test(demo_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
