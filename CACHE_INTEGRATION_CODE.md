# Step-by-Step: Integrating Enhanced Parser into cache.py

This document shows the EXACT code changes needed to wire the enhanced parser into the existing cache system.

## Current cache.py structure (lines to find)

Find these sections in `src/opensight/infra/cache.py`:

```python
from opensight.analysis.analytics import DemoAnalyzer
```

## CHANGE 1: Add enhanced parser imports (after line ~20)

**Find this:**
```python
from opensight.analysis.analytics import DemoAnalyzer
```

**Replace with:**
```python
from opensight.analysis.analytics import DemoAnalyzer
from opensight.core.enhanced_parser import (
    ChunkedDemoParser,
    CoachingAnalysisEngine,
    MetricCalculator
)
```

---

## CHANGE 2: Update the analyze() method

This is the main integration point. Find the analyze method in cache.py (usually around line 80-120).

**Current code (approx):**
```python
def analyze(self, demo_path: str) -> dict:
    """Analyze a CS2 demo file."""
    cache_key = f"demo:{demo_path}"
    
    # Check cache
    cached = self.get(cache_key)
    if cached:
        return cached
    
    try:
        # Current implementation - basic analysis only
        analyzer = DemoAnalyzer(demo_path)
        analysis_result = analyzer.parse()
        
        # Cache results
        self.set(cache_key, analysis_result, ttl=86400)
        return analysis_result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
```

**New code (with enhanced parser):**
```python
def analyze(self, demo_path: str) -> dict:
    """Analyze a CS2 demo file with professional metrics."""
    cache_key = f"demo:{demo_path}"
    
    # Check cache
    cached = self.get(cache_key)
    if cached:
        return cached
    
    try:
        # Try enhanced analysis first (new professional metrics)
        logger.info(f"Starting enhanced analysis of {demo_path}")
        
        engine = CoachingAnalysisEngine()
        enhanced_metrics = engine.analyze_demo(demo_path)
        
        # Also get basic analysis for compatibility
        analyzer = DemoAnalyzer(demo_path)
        basic_analysis = analyzer.parse()
        
        # Merge both results - enhanced metrics override/supplement basic
        analysis_result = {
            **basic_analysis,
            **enhanced_metrics,
            'metrics_version': 'professional_v2'  # Track which version
        }
        
        logger.info(f"Enhanced analysis complete: {enhanced_metrics.get('total_rounds', '?')} rounds")
        
        # Cache results (longer TTL for complex analysis)
        self.set(cache_key, analysis_result, ttl=86400)
        return analysis_result
        
    except Exception as e:
        logger.warning(f"Enhanced analysis failed, falling back to basic: {e}")
        # Fallback to basic analysis if enhanced fails
        try:
            analyzer = DemoAnalyzer(demo_path)
            basic_result = analyzer.parse()
            basic_result['metrics_version'] = 'basic_fallback'
            self.set(cache_key, basic_result, ttl=86400)
            return basic_result
        except Exception as fallback_error:
            logger.error(f"All analysis failed: {fallback_error}")
            raise
```

---

## CHANGE 3: Add a new method to analyze without cache (optional but useful)

Add this method to the CacheManager class:

```python
def analyze_fresh(self, demo_path: str) -> dict:
    """Analyze demo bypassing cache - always fresh data."""
    cache_key = f"demo:{demo_path}"
    
    # Remove from cache to force fresh analysis
    self.delete(cache_key)
    
    # Perform fresh analysis
    return self.analyze(demo_path)
```

---

## CHANGE 4: Update the data structure to handle new metrics

When storing in database (if using cache with DB backend), ensure these fields exist:

```python
# In your cache backend (if database-backed):
analysis_result_schema = {
    'demo_path': str,
    'map_name': str,
    'duration_seconds': float,
    
    # Basic metrics (existing)
    'total_kills': int,
    'total_deaths': int,
    
    # Enhanced metrics (new - from professional parser)
    'entry_frags': dict,        # {steam_id: {attempts, kills, deaths}}
    'trade_kills': dict,        # {steam_id: {kills, deaths_traded}}
    'ttd_metrics': dict,        # {steam_id: {median_ms, mean_ms, 95th_ms}}
    'crosshair_placement': dict, # {steam_id: {median_error, mean_error}}
    'clutch_stats': dict,       # {steam_id: {wins, attempts, 1v1-1v5}}
    'total_rounds': int,
    'metrics_version': str      # 'professional_v2' or 'basic_fallback'
}
```

---

## CHANGE 5: Add logging for debugging

Add these log statements to track enhanced parser execution:

```python
# After the enhanced metrics are calculated (in analyze() method):

if 'entry_frags' in enhanced_metrics:
    total_entries = sum(
        p['entry_attempts'] 
        for p in enhanced_metrics.get('entry_frags', {}).values()
    )
    logger.info(f"  Entry frags detected: {total_entries} total")

if 'ttd_metrics' in enhanced_metrics:
    players_with_ttd = len(enhanced_metrics.get('ttd_metrics', {}))
    logger.info(f"  TTD metrics calculated for {players_with_ttd} players")

if 'crosshair_placement' in enhanced_metrics:
    players_with_cp = len(enhanced_metrics.get('crosshair_placement', {}))
    logger.info(f"  CP metrics calculated for {players_with_cp} players")

if 'clutch_stats' in enhanced_metrics:
    total_clutches = sum(
        p['clutch_attempts'] 
        for p in enhanced_metrics.get('clutch_stats', {}).values()
    )
    logger.info(f"  Clutch scenarios detected: {total_clutches} total")
```

---

## CHANGE 6: Add timing instrumentation (optional)

To measure performance:

```python
import time

def analyze(self, demo_path: str) -> dict:
    """Analyze a CS2 demo file with professional metrics."""
    cache_key = f"demo:{demo_path}"
    
    # Check cache
    cached = self.get(cache_key)
    if cached:
        return cached
    
    start_time = time.time()
    
    try:
        # Enhanced analysis
        logger.info(f"Starting enhanced analysis of {demo_path}")
        engine = CoachingAnalysisEngine()
        enhanced_metrics = engine.analyze_demo(demo_path)
        
        # Basic analysis
        analyzer = DemoAnalyzer(demo_path)
        basic_analysis = analyzer.parse()
        
        # Merge results
        analysis_result = {
            **basic_analysis,
            **enhanced_metrics,
            'metrics_version': 'professional_v2'
        }
        
        elapsed = time.time() - start_time
        logger.info(f"Enhanced analysis complete in {elapsed:.2f}s: "
                   f"{enhanced_metrics.get('total_rounds', '?')} rounds, "
                   f"{enhanced_metrics.get('entry_frags', {}).get('total', 0)} entry metrics")
        
        # Cache results
        self.set(cache_key, analysis_result, ttl=86400)
        return analysis_result
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning(f"Enhanced analysis failed after {elapsed:.2f}s, falling back: {e}")
        # Fallback...
```

---

## CHANGE 7: Testing the integration

After making these changes, test with:

```python
# test_cache_integration.py

from opensight.infra.cache import CacheManager

def test_enhanced_parser_integration():
    """Test that enhanced parser works in cache pipeline"""
    
    cache = CacheManager()
    demo_path = "/path/to/test_demo.dem"
    
    # Clear cache to force fresh analysis
    cache.delete(f"demo:{demo_path}")
    
    # Analyze - should use enhanced parser
    result = cache.analyze(demo_path)
    
    # Verify enhanced metrics are present
    print("\n=== Enhanced Metrics Verification ===")
    
    if 'entry_frags' in result:
        print("✓ Entry frags calculated")
        print(f"  Entries: {sum(p['entry_attempts'] for p in result['entry_frags'].values())}")
    
    if 'ttd_metrics' in result:
        print("✓ TTD metrics calculated")
        for steam_id, ttd in result['ttd_metrics'].items():
            print(f"  Player {steam_id}: {ttd['ttd_median_ms']:.0f}ms median")
    
    if 'crosshair_placement' in result:
        print("✓ CP metrics calculated")
        for steam_id, cp in result['crosshair_placement'].items():
            print(f"  Player {steam_id}: {cp['cp_median_error']:.1f}° error")
    
    if 'clutch_stats' in result:
        print("✓ Clutch stats calculated")
        print(f"  Total clutches: {sum(p['clutch_attempts'] for p in result['clutch_stats'].values())}")
    
    print(f"\nMetrics version: {result.get('metrics_version')}")
    
    return result

if __name__ == "__main__":
    result = test_enhanced_parser_integration()
    print("\n✓ Integration test complete!")
```

---

## CHANGE 8: Error Handling Edge Cases

Add these defensive checks in analyze():

```python
def analyze(self, demo_path: str) -> dict:
    """Analyze a CS2 demo file with professional metrics."""
    
    # Validate input
    if not demo_path:
        raise ValueError("demo_path cannot be empty")
    
    from pathlib import Path
    if not Path(demo_path).exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")
    
    cache_key = f"demo:{demo_path}"
    
    # ... rest of method ...
    
    try:
        engine = CoachingAnalysisEngine()
        enhanced_metrics = engine.analyze_demo(demo_path)
        
        # Validate enhanced metrics structure
        if not isinstance(enhanced_metrics, dict):
            raise ValueError("Enhanced metrics must return dict")
        
        if 'total_rounds' not in enhanced_metrics:
            logger.warning("Enhanced metrics missing total_rounds")
            enhanced_metrics['total_rounds'] = 0
        
        # Continue with merge...
```

---

## Summary of Changes

| File | Changes | Impact |
|------|---------|--------|
| cache.py | 5 import lines added, analyze() method updated | High - core integration |
| (optional) test file | New test for validation | Medium - verification |
| (optional) logging | Enhanced logging | Low - debugging |

**Total lines added**: ~50-70 lines
**Risk level**: Low (fallback to basic analysis if enhanced fails)
**Performance impact**: +5-10 seconds per demo (one-time, then cached)

---

## Verification Steps

After making changes:

1. ✅ Python syntax check
   ```bash
   python -m py_compile src/opensight/infra/cache.py
   ```

2. ✅ Import check
   ```python
   from opensight.infra.cache import CacheManager
   # Should not raise ImportError
   ```

3. ✅ Test with demo file
   ```python
   cache = CacheManager()
   result = cache.analyze("/path/to/demo.dem")
   # Should have 'entry_frags', 'ttd_metrics', etc.
   ```

4. ✅ Performance check
   - First run: ~30-60s for 500MB file
   - Subsequent runs: <1ms (cached)

5. ✅ Error handling
   - Test with corrupted file
   - Test with missing file
   - Verify fallback works

---

## Rollback Instructions

If you need to revert:

```python
# In cache.py, replace the analyze() method back to:

def analyze(self, demo_path: str) -> dict:
    """Analyze a CS2 demo file."""
    cache_key = f"demo:{demo_path}"
    
    cached = self.get(cache_key)
    if cached:
        return cached
    
    try:
        analyzer = DemoAnalyzer(demo_path)
        analysis_result = analyzer.parse()
        self.set(cache_key, analysis_result, ttl=86400)
        return analysis_result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
```

Remove the enhanced parser imports at the top.

---

## Next Steps After Integration

1. **Test with various demos** - Small, medium, large files
2. **Validate metric accuracy** - Compare to manual checks
3. **Monitor performance** - Ensure <60s for 500MB files
4. **Update API endpoints** - Return enhanced metrics to frontend
5. **Update web UI** - Display TTD, CP, entry/trade/clutch cards
6. **Gather feedback** - From users on metric quality

