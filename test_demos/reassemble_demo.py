#!/usr/bin/env python3
"""
Reassemble demo file from chunks and test parser with comprehensive analysis.
Usage: python reassemble_demo.py <output_filename>
"""

import sys
from pathlib import Path
from typing import Generator
import pandas as pd

def find_chunks(base_name: str) -> list[Path]:
    """Find all chunk files for a demo."""
    chunk_dir = Path(__file__).parent
    chunks = sorted(chunk_dir.glob(f"{base_name}.part*"))
    return chunks

def reassemble_demo(output_path: Path) -> bool:
    """Reassemble demo from chunks."""
    base_name = output_path.stem
    chunks = find_chunks(base_name)
    
    if not chunks:
        print(f"❌ No chunks found for {base_name}")
        return False
    
    print(f"📦 Found {len(chunks)} chunks for {base_name}")
    
    with open(output_path, 'wb') as out_f:
        for i, chunk_file in enumerate(chunks, 1):
            chunk_size = chunk_file.stat().st_size
            print(f"  ▸ Assembling chunk {i}/{len(chunks)} ({chunk_size / (1024*1024):.1f}MB): {chunk_file.name}")
            with open(chunk_file, 'rb') as in_f:
                out_f.write(in_f.read())
    
    total_size = output_path.stat().st_size
    print(f"✅ Reassembled: {output_path.name} ({total_size / (1024*1024):.1f}MB)")
    return True

def test_demo_parser(demo_path: Path) -> None:
    """Comprehensive parser test with detailed data analysis."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from opensight.core.parser import DemoParser
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"\n🔍 Testing parser on {demo_path.name}...")
    try:
        parser = DemoParser(demo_path)
        result = parser.parse(include_ticks=True, comprehensive=True)
        
        print(f"\n{'='*70}")
        print(f"📊 PARSE RESULTS")
        print(f"{'='*70}")
        print(f"  Map: {result.map_name}")
        print(f"  Server: {result.server_name}")
        print(f"  Demo Duration: {len(result.rounds)} rounds")
        
        print(f"\n{'='*70}")
        print(f"📈 CORE DATA EXTRACTION")
        print(f"{'='*70}")
        print(f"  Kills: {len(result.kills)}")
        print(f"  Damages: {len(result.damages)}")
        print(f"  Rounds: {len(result.rounds)}")
        
        print(f"\n{'='*70}")
        print(f"🎯 ADVANCED EVENT DATA")
        print(f"{'='*70}")
        weapon_fire_count = len(result.weapon_fire) if result.weapon_fire is not None and len(result.weapon_fire) > 0 else 0
        grenades_count = len(result.grenades) if result.grenades is not None and len(result.grenades) > 0 else 0
        blinds_count = len(result.blinds) if result.blinds is not None and len(result.blinds) > 0 else 0
        bomb_count = len(result.bomb) if result.bomb is not None and len(result.bomb) > 0 else 0
        
        print(f"  Weapon Fire Events: {weapon_fire_count}")
        print(f"  Grenade Events: {grenades_count}")
        print(f"  Blind Events: {blinds_count}")
        print(f"  Bomb Events: {bomb_count}")
        
        print(f"\n{'='*70}")
        print(f"💾 KILL DATA STRUCTURE")
        print(f"{'='*70}")
        print(f"  Total Columns: {len(result.kills.columns)}")
        print(f"  Kill Records: {len(result.kills)}")
        print(f"\n  Available Columns:")
        for i, col in enumerate(sorted(result.kills.columns), 1):
            sample_val = result.kills[col].iloc[0] if len(result.kills) > 0 else "N/A"
            print(f"    {i:2d}. {col:30s} | Sample: {sample_val}")
        
        if weapon_fire_count > 0:
            print(f"\n{'='*70}")
            print(f"💾 WEAPON FIRE DATA STRUCTURE")
            print(f"{'='*70}")
            print(f"  Total Columns: {len(result.weapon_fire.columns)}")
            print(f"  Fire Records: {weapon_fire_count}")
            print(f"\n  Available Columns:")
            for i, col in enumerate(sorted(result.weapon_fire.columns), 1):
                sample_val = result.weapon_fire[col].iloc[0] if weapon_fire_count > 0 else "N/A"
                print(f"    {i:2d}. {col:30s} | Sample: {sample_val}")
        
        if grenades_count > 0:
            print(f"\n{'='*70}")
            print(f"💾 GRENADE DATA STRUCTURE")
            print(f"{'='*70}")
            print(f"  Total Columns: {len(result.grenades.columns)}")
            print(f"  Grenade Records: {grenades_count}")
            print(f"\n  Available Columns:")
            for i, col in enumerate(sorted(result.grenades.columns), 1):
                sample_val = result.grenades[col].iloc[0] if grenades_count > 0 else "N/A"
                print(f"    {i:2d}. {col:30s} | Sample: {sample_val}")
        
        if blinds_count > 0:
            print(f"\n{'='*70}")
            print(f"💾 BLIND DATA STRUCTURE")
            print(f"{'='*70}")
            print(f"  Total Columns: {len(result.blinds.columns)}")
            print(f"  Blind Records: {blinds_count}")
            print(f"\n  Available Columns:")
            for i, col in enumerate(sorted(result.blinds.columns), 1):
                sample_val = result.blinds[col].iloc[0] if blinds_count > 0 else "N/A"
                print(f"    {i:2d}. {col:30s} | Sample: {sample_val}")
        
        print(f"\n{'='*70}")
        print(f"🎮 DATA INSIGHTS & OPTIMIZATION OPPORTUNITIES")
        print(f"{'='*70}")
        
        # Analyze what we can extract
        insights = []
        
        if weapon_fire_count > 0:
            insights.append("✅ Weapon Fire: Can extract accuracy stats, spray patterns, weapon usage")
        else:
            insights.append("⚠️  Weapon Fire: NO DATA - Accuracy metrics will be limited")
        
        if grenades_count > 0:
            insights.append("✅ Grenades: Can extract grenade economy, utility usage patterns, timing")
        else:
            insights.append("⚠️  Grenades: NO DATA - Utility analysis will be limited")
        
        if blinds_count > 0:
            insights.append("✅ Blinds: Can extract flash effectiveness, utility support rating")
        else:
            insights.append("⚠️  Blinds: NO DATA - Flash utility metrics unavailable")
        
        # Check for position data
        has_attacker_pos = 'attacker_X' in result.kills.columns
        has_victim_pos = 'victim_X' in result.kills.columns
        if has_attacker_pos and has_victim_pos:
            insights.append("✅ Position Data: Can calculate sightlines, map control, positioning metrics")
        
        # Check for tick data
        if result.ticks_df is not None:
            insights.append(f"✅ Tick-level Data: {len(result.ticks_df)} position records - enables real-time analysis")
        else:
            insights.append("⚠️  Tick Data: Not included - limits granular timing analysis")
        
        for insight in insights:
            print(f"  {insight}")
        
        print(f"\n{'='*70}")
        print(f"💡 RECOMMENDED OPTIMIZATIONS")
        print(f"{'='*70}")
        
        recommendations = [
            "1. Enable demoparser2 comprehensive mode to capture all event types",
            "2. Extract spray patterns from weapon_fire data for accuracy coaching",
            "3. Build grenade utility graphs showing economy and timing",
            "4. Calculate opening kill statistics with positioning",
            "5. Track utility support (flashes, smokes) per player per round",
            "6. Create heatmaps from position + weapon fire data",
            "7. Analyze crosshair placement efficiency with kill distance",
            "8. Build pre-round utility economy tracking",
        ]
        
        for rec in recommendations:
            print(f"  {rec}")
        
        return result
    except Exception as e:
        print(f"❌ Parser error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
    else:
        demo_name = "demo"
    
    output_path = Path(__file__).parent / f"{demo_name}.dem"
    
    # Reassemble
    success = reassemble_demo(output_path)
    
    # Test parser
    if success and output_path.exists():
        test_demo_parser(output_path)
