#!/usr/bin/env python3
"""
Tactical Analysis Demo - Shows how the new tactical review system works

This script demonstrates the complete tactical analysis pipeline without
needing to run the web server.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from opensight.analysis.tactical_service import TacticalAnalysisService
from opensight.core.parser import DemoParser


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def format_stat(label: str, value: str | float | int, suffix: str = "") -> None:
    """Format and print a stat."""
    if isinstance(value, float):
        value = f"{value:.1f}"
    print(f"  {label:<30} {value}{suffix}")


def demo_tactical_analysis():
    """Demonstrate tactical analysis system."""
    
    print("\n" + "="*70)
    print("  OPENSIGHT TACTICAL DEMO ANALYZER")
    print("  Showing new tactical review system capabilities")
    print("="*70 + "\n")
    
    # Find a demo file
    demo_path = None
    replays_dir = Path.home() / "AppData" / "Local" / "Valve" / "Steam" / "userdata"
    
    # Look for demo files
    for potential_path in [
        Path("replays"),
        Path("demos"),
        Path.home() / "replays",
    ]:
        if potential_path.exists():
            for dem_file in potential_path.glob("*.dem"):
                demo_path = dem_file
                break
        if demo_path:
            break
    
    if not demo_path:
        print("⚠️  No demo files found in standard locations.")
        print("\nTo test the tactical analyzer:")
        print("  1. Copy a CS2 demo to ./replays/ folder")
        print("  2. Run this script again")
        print("\nDemo files are typically in:")
        print("  C:\\Program Files (x86)\\Steam\\steamapps\\common\\Counter-Strike Global Offensive\\game\\csgo\\replays")
        print("\nOR")
        print("\n  Run the web interface:")
        print("  python -m opensight.web.app")
        return
    
    print(f"✓ Found demo: {demo_path.name}\n")
    
    try:
        # Parse demo
        print("Parsing demo file...")
        parser = DemoParser(demo_path)
        demo_data = parser.parse()
        print(f"✓ Demo parsed successfully\n")
        
        # Run tactical analysis
        print("Running tactical analysis...")
        service = TacticalAnalysisService(demo_data)
        summary = service.analyze()
        print(f"✓ Analysis complete\n")
        
        # Print results
        print_section("DEMO INFORMATION")
        format_stat("Map", demo_data.map_name)
        format_stat("Duration", f"{demo_data.duration_seconds / 60:.1f}", " minutes")
        format_stat("Players", len(demo_data.player_names))
        
        print_section("KEY INSIGHTS")
        for i, insight in enumerate(summary.key_insights, 1):
            print(f"  {i}. {insight}")
        
        print_section("TEAM OVERVIEW")
        print("\n  TERRORISTS")
        if summary.t_stats:
            format_stat("  Records", f"{summary.t_stats.get('rounds_won', 0)}-{summary.t_stats.get('rounds_lost', 0)}")
            format_stat("  Play Style", summary.t_stats.get('play_style', 'Unknown'))
            format_stat("  Utility Success", f"{summary.t_stats.get('utility_success', 0)}%")
            format_stat("  Coordination", f"{summary.t_stats.get('coordination_score', 0)}/100")
        
        print("\n  COUNTER-TERRORISTS")
        if summary.ct_stats:
            format_stat("  Records", f"{summary.ct_stats.get('rounds_won', 0)}-{summary.ct_stats.get('rounds_lost', 0)}")
            format_stat("  Defense Strategy", summary.ct_stats.get('defense_strategy', 'Unknown'))
            format_stat("  Anti-Exe Rate", f"{summary.ct_stats.get('anti_exe_rate', 0)}%")
            format_stat("  Retake Success", f"{summary.ct_stats.get('retake_rate', 0)}%")
        
        print_section("PLAY EXECUTION")
        if summary.t_executes:
            print("  Most Common Attacks:")
            for exec_type, count in summary.t_executes[:5]:
                print(f"    • {exec_type}: {count}x")
        
        if summary.buy_patterns:
            print("\n  Buy Patterns (Win Rate):")
            for buy_type, win_rate in summary.buy_patterns[:5]:
                status = "✓" if win_rate >= 60 else "~" if win_rate >= 40 else "✗"
                print(f"    {status} {buy_type}: {win_rate:.0f}%")
        
        print_section("KEY PLAYERS")
        for player in summary.key_players[:3]:
            print(f"\n  {player['name']} ({player['team']})")
            print(f"    Role: {player.get('primary_role', 'Unknown')}")
            print(f"    Opening Kills: {player.get('opening_kills', 0)}")
            print(f"    Trade Kills: {player.get('trade_kills', 0)}")
            print(f"    Impact Rating: {player.get('impact_rating', 0)}")
            
            if player.get('strengths'):
                print(f"    Strengths: {', '.join(player['strengths'][:2])}")
            if player.get('weaknesses'):
                print(f"    Weaknesses: {', '.join(player['weaknesses'][:2])}")
        
        print_section("ROUND-BY-ROUND (First 8 Rounds)")
        for play in summary.round_plays[:8]:
            result_symbol = "🟢" if play['round_winner'] == "T" else "🔵"
            print(
                f"  {result_symbol} R{play['round_num']:2d} | "
                f"{play['attack_type']:<20} | "
                f"First Kill: {play.get('first_kill_time', 0):.1f}s"
            )
        
        print_section("TEAM MATCHUP ANALYSIS")
        print(f"\n  Win Rates:")
        print(f"    T Side: {summary.t_win_rate:.1f}%")
        print(f"    CT Side: {summary.ct_win_rate:.1f}%")
        
        print(f"\n  T SIDE STRENGTHS:")
        for strength in summary.t_strengths:
            print(f"    ✓ {strength}")
        
        print(f"\n  T SIDE WEAKNESSES:")
        for weakness in summary.t_weaknesses:
            print(f"    ✗ {weakness}")
        
        print(f"\n  CT SIDE STRENGTHS:")
        for strength in summary.ct_strengths:
            print(f"    ✓ {strength}")
        
        print(f"\n  CT SIDE WEAKNESSES:")
        for weakness in summary.ct_weaknesses:
            print(f"    ✗ {weakness}")
        
        print_section("COACHING RECOMMENDATIONS")
        print("\n  TEAM LEVEL:")
        for i, rec in enumerate(summary.team_recommendations[:3], 1):
            print(f"    {i}. {rec}")
        
        print("\n  INDIVIDUAL FOCUS:")
        for i, rec in enumerate(summary.individual_recommendations[:3], 1):
            print(f"    {i}. {rec}")
        
        print("\n  PRACTICE DRILLS:")
        for i, drill in enumerate(summary.practice_drills[:3], 1):
            print(f"    {i}. {drill}")
        
        print("\n" + "="*70)
        print("  ✓ Tactical analysis complete!")
        print("="*70 + "\n")
        
        print("This is what the web interface shows when you upload a demo.")
        print("\nTo use the full web interface:")
        print("  python -m opensight.web.app")
        print("\nThen visit http://localhost:5000 and upload a demo.\n")
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_tactical_analysis()
