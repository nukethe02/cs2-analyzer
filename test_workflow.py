#!/usr/bin/env python3
"""
Test script to verify CS2 analyzer workflow.

Tests:
1. Demo parsing
2. TTD calculation
3. CP calculation
4. API response generation
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_demo_parsing():
    """Test demo file parsing."""
    from opensight.core.parser import DemoParser
    
    # Find a demo file to test with
    demo_path = Path.home() / "Counter-Strike Global Offensive" / "game" / "csgo" / "replays"
    
    dem_files = list(demo_path.glob("*.dem")) if demo_path.exists() else []
    if not dem_files:
        print("No .dem files found in default replay folder")
        print(f"Expected location: {demo_path}")
        return False
    
    demo_file = dem_files[0]
    print(f"\n✓ Testing with demo: {demo_file.name}")
    
    try:
        parser = DemoParser(demo_file)
        data = parser.parse()
        
        print(f"  - Map: {data.map_name}")
        print(f"  - Rounds: {data.num_rounds}")
        print(f"  - Kills: {len(data.kills)}")
        print(f"  - Damages: {len(data.damages)}")
        print(f"  - Duration: {data.duration_seconds:.1f}s")
        print(f"  - Players: {len(data.player_names)}")
        
        # Check position data
        kills_with_pos = sum(1 for k in data.kills if k.attacker_x is not None)
        kills_with_angles = sum(1 for k in data.kills if k.attacker_pitch is not None)
        print(f"  - Kills with position data: {kills_with_pos}/{len(data.kills)}")
        print(f"  - Kills with angles: {kills_with_angles}/{len(data.kills)}")
        
        if kills_with_pos < len(data.kills) * 0.5:
            print("  ⚠ WARNING: Less than 50% of kills have position data!")
        
        return data
        
    except Exception as e:
        print(f"  ✗ Error parsing demo: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_ttd_calculation(data):
    """Test TTD metric calculation."""
    from opensight.analysis.metrics import calculate_ttd
    
    if not data:
        return False
    
    print(f"\n✓ Testing TTD calculation...")
    try:
        ttd_results = calculate_ttd(data)
        
        if not ttd_results:
            print("  ⚠ No TTD results computed")
            return False
        
        for player_id, ttd in list(ttd_results.items())[:5]:
            print(f"  - {ttd.player_name}: {ttd.engagement_count} engagements, {ttd.median_ttd_ms:.0f}ms median")
        
        print(f"  ✓ Computed TTD for {len(ttd_results)} players")
        return ttd_results
        
    except Exception as e:
        print(f"  ✗ Error calculating TTD: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cp_calculation(data):
    """Test Crosshair Placement metric calculation."""
    from opensight.analysis.metrics import calculate_crosshair_placement
    
    if not data:
        return False
    
    print(f"\n✓ Testing Crosshair Placement calculation...")
    try:
        cp_results = calculate_crosshair_placement(data)
        
        if not cp_results:
            print("  ⚠ No CP results computed - ensure kills have position/angle data")
            return False
        
        for player_id, cp in list(cp_results.items())[:5]:
            print(f"  - {cp.player_name}: {cp.sample_count} samples, {cp.mean_angle_deg:.1f}° angle, {cp.placement_score:.1f} score")
        
        print(f"  ✓ Computed CP for {len(cp_results)} players")
        return cp_results
        
    except Exception as e:
        print(f"  ✗ Error calculating CP: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_api_response(data, ttd_results, cp_results):
    """Test full API response generation."""
    from opensight.analysis.analytics import DemoAnalyzer
    
    if not data:
        return False
    
    print(f"\n✓ Testing API response generation...")
    try:
        analyzer = DemoAnalyzer(data)
        analysis = analyzer.analyze()
        
        print(f"  - Map: {analysis.map_name}")
        print(f"  - Players analyzed: {len(analysis.players)}")
        print(f"  - Rounds: {analysis.total_rounds}")
        print(f"  - Score: {analysis.team1_score} - {analysis.team2_score}")
        
        # Get top player
        mvp = analysis.get_mvp()
        if mvp:
            print(f"  - MVP: {mvp.name} (Rating: {mvp.hltv_rating:.2f})")
        
        leaderboard = analysis.get_leaderboard()[:3]
        if leaderboard:
            print(f"  - Top 3 players:")
            for player in leaderboard:
                print(f"    • {player.name}: {player.kills}K-{player.deaths}D (Rating: {player.hltv_rating:.2f})")
        
        print(f"  ✓ Generated complete analysis")
        return True
        
    except Exception as e:
        print(f"  ✗ Error generating API response: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CS2 Analyzer Workflow Test")
    print("=" * 60)
    
    # Run tests
    data = test_demo_parsing()
    ttd_results = test_ttd_calculation(data) if data else None
    cp_results = test_cp_calculation(data) if data else None
    api_ok = test_api_response(data, ttd_results, cp_results) if data else False
    
    print("\n" + "=" * 60)
    if api_ok:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed - see errors above")
    print("=" * 60)
