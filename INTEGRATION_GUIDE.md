# Integration Guide: Enhanced Parser into Production System

## Current Status

✅ **Created:**
- `src/opensight/core/enhanced_parser.py` - Complete professional-grade parser
- `src/opensight/infra/enhanced_cache_integration.py` - Integration wrapper
- Enhanced `src/opensight/core/parser.py` - Full spatial data extraction

⏳ **Pending Integration:**
- Wire enhanced parser into cache.py analysis pipeline
- Update web UI to display new metrics
- Add database schema for TTD/CP storage
- Create coaching insights generator

## Integration Steps

### Step 1: Update cache.py to Use Enhanced Parser

**File:** `src/opensight/infra/cache.py`

**Current Flow:**
```
analyze() 
  → DemoAnalyzer() 
  → basic metrics (kills, deaths, KDA)
  → cache results
```

**New Flow:**
```
analyze() 
  → CoachingAnalysisEngine (enhanced parser)
  → professional metrics (TTD, CP, Entry, Trade, Clutch)
  → merge with basic metrics
  → cache results
```

**Changes Needed:**
```python
# At top of cache.py
from opensight.core.enhanced_parser import (
    ChunkedDemoParser, 
    CoachingAnalysisEngine,
    MetricCalculator
)

# In analyze() function (replace current analysis block)
def analyze(self, demo_path: str):
    # ... existing code ...
    
    try:
        # Use enhanced parser for professional metrics
        engine = CoachingAnalysisEngine()
        enhanced_metrics = engine.analyze_demo(demo_path)
        
        # Parse with standard parser for basic data
        analyzer = DemoAnalyzer(demo_path)
        basic_analysis = analyzer.parse()
        
        # Merge both results
        analysis_result = {
            **basic_analysis,
            **enhanced_metrics
        }
        
        # Cache results
        self.set(cache_key, analysis_result, ttl=86400)
        return analysis_result
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        # Fallback to basic analysis
        return self._basic_analysis_fallback(demo_path)
```

### Step 2: Add Metrics to Database Schema

**File:** `src/opensight/infra/database.py`

Add columns to player stats table:

```python
class PlayerStats(Base):
    # ... existing columns ...
    
    # Enhanced metrics
    ttd_median_ms: float = Column(Float, nullable=True)
    ttd_mean_ms: float = Column(Float, nullable=True)
    ttd_95th_ms: float = Column(Float, nullable=True)
    
    cp_median_error: float = Column(Float, nullable=True)
    cp_mean_error: float = Column(Float, nullable=True)
    
    entry_attempts: int = Column(Integer, default=0)
    entry_kills: int = Column(Integer, default=0)
    entry_deaths: int = Column(Integer, default=0)
    
    trade_kills: int = Column(Integer, default=0)
    deaths_traded: int = Column(Integer, default=0)
    
    clutch_wins: int = Column(Integer, default=0)
    clutch_attempts: int = Column(Integer, default=0)
    v1_wins: int = Column(Integer, default=0)
    v2_wins: int = Column(Integer, default=0)
    v3_wins: int = Column(Integer, default=0)
    v4_wins: int = Column(Integer, default=0)
    v5_wins: int = Column(Integer, default=0)
```

### Step 3: Update API Responses

**File:** `src/opensight/api.py`

Extend the player stats endpoint to include enhanced metrics:

```python
@app.get("/api/players/{player_id}/stats")
async def get_player_stats(player_id: str):
    stats = db.query(PlayerStats).filter_by(steam_id=player_id).first()
    
    return {
        # Basic stats
        "kills": stats.kills,
        "deaths": stats.deaths,
        "kda": stats.kills / max(stats.deaths, 1),
        
        # Enhanced metrics
        "timing": {
            "ttd_median_ms": stats.ttd_median_ms,
            "ttd_mean_ms": stats.ttd_mean_ms,
            "ttd_95th_ms": stats.ttd_95th_ms
        },
        "positioning": {
            "cp_median_error_deg": stats.cp_median_error,
            "cp_mean_error_deg": stats.cp_mean_error
        },
        "entries": {
            "attempts": stats.entry_attempts,
            "kills": stats.entry_kills,
            "deaths": stats.entry_deaths,
            "success_rate": stats.entry_kills / max(stats.entry_attempts, 1)
        },
        "trades": {
            "kills": stats.trade_kills,
            "deaths_traded": stats.deaths_traded
        },
        "clutches": {
            "wins": stats.clutch_wins,
            "attempts": stats.clutch_attempts,
            "win_rate": stats.clutch_wins / max(stats.clutch_attempts, 1),
            "breakdown": {
                "v1": stats.v1_wins,
                "v2": stats.v2_wins,
                "v3": stats.v3_wins,
                "v4": stats.v4_wins,
                "v5": stats.v5_wins
            }
        }
    }
```

### Step 4: Update Web UI

**File:** `src/opensight/static/index.html`

Add enhanced metrics sections to player card:

```html
<!-- Time to Damage Card -->
<div class="metric-card">
    <h4>Timing (TTD)</h4>
    <div class="metric-value" id="ttd-median">-</div>
    <span class="metric-label">Median</span>
    <div class="metric-detail">
        <span id="ttd-mean">-</span> mean | 
        <span id="ttd-95th">-</span> 95th percentile
    </div>
    <div class="metric-info">Lower is better. Pro range: 150-350ms</div>
</div>

<!-- Crosshair Placement Card -->
<div class="metric-card">
    <h4>Positioning (CP)</h4>
    <div class="metric-value" id="cp-median">-</div>
    <span class="metric-label">Angular Error (degrees)</span>
    <div class="metric-detail">
        <span id="cp-mean">-</span> mean
    </div>
    <div class="metric-info">Lower is better. Pro range: 3-8°</div>
</div>

<!-- Entry Frags Card -->
<div class="metric-card">
    <h4>Entry Frags</h4>
    <div class="metric-value" id="entry-sr">-</div>
    <span class="metric-label">Success Rate</span>
    <div class="metric-detail">
        <span id="entry-kills">-</span> kills / 
        <span id="entry-attempts">-</span> attempts
    </div>
</div>

<!-- Trade Kills Card -->
<div class="metric-card">
    <h4>Trade Kills</h4>
    <div class="metric-value" id="trade-kills">-</div>
    <span class="metric-label">Total Trades</span>
    <div class="metric-detail">
        <span id="trades-after-death">-</span> after teammate death
    </div>
</div>

<!-- Clutch Stats Card -->
<div class="metric-card">
    <h4>Clutch Performance</h4>
    <div class="metric-value" id="clutch-wr">-</div>
    <span class="metric-label">Win Rate</span>
    <div class="metric-detail">
        <span id="clutch-wins">-</span> wins / 
        <span id="clutch-attempts">-</span> attempts
    </div>
    <div class="clutch-breakdown">
        <span>1v1: <span id="v1">-</span></span>
        <span>1v2: <span id="v2">-</span></span>
        <span>1v3: <span id="v3">-</span></span>
        <span>1v4: <span id="v4">-</span></span>
        <span>1v5: <span id="v5">-</span></span>
    </div>
</div>
```

Add JavaScript to populate metrics:

```javascript
async function loadEnhancedMetrics(playerId) {
    const response = await fetch(`/api/players/${playerId}/stats`);
    const stats = await response.json();
    
    // Timing metrics
    document.getElementById('ttd-median').textContent = 
        (stats.timing.ttd_median_ms).toFixed(0) + 'ms';
    document.getElementById('ttd-mean').textContent = 
        (stats.timing.ttd_mean_ms).toFixed(0) + 'ms';
    document.getElementById('ttd-95th').textContent = 
        (stats.timing.ttd_95th_ms).toFixed(0) + 'ms';
    
    // Positioning metrics
    document.getElementById('cp-median').textContent = 
        (stats.positioning.cp_median_error_deg).toFixed(1) + '°';
    document.getElementById('cp-mean').textContent = 
        (stats.positioning.cp_mean_error_deg).toFixed(1) + '°';
    
    // Entry frags
    document.getElementById('entry-sr').textContent = 
        (stats.entries.success_rate * 100).toFixed(0) + '%';
    document.getElementById('entry-kills').textContent = stats.entries.kills;
    document.getElementById('entry-attempts').textContent = stats.entries.attempts;
    
    // Trade kills
    document.getElementById('trade-kills').textContent = stats.trades.kills;
    document.getElementById('trades-after-death').textContent = 
        stats.trades.deaths_traded;
    
    // Clutches
    document.getElementById('clutch-wr').textContent = 
        (stats.clutches.win_rate * 100).toFixed(0) + '%';
    document.getElementById('clutch-wins').textContent = stats.clutches.wins;
    document.getElementById('clutch-attempts').textContent = stats.clutches.attempts;
    
    // Clutch breakdown
    document.getElementById('v1').textContent = stats.clutches.breakdown.v1;
    document.getElementById('v2').textContent = stats.clutches.breakdown.v2;
    document.getElementById('v3').textContent = stats.clutches.breakdown.v3;
    document.getElementById('v4').textContent = stats.clutches.breakdown.v4;
    document.getElementById('v5').textContent = stats.clutches.breakdown.v5;
}
```

### Step 5: Add Coaching Insights

**File:** `src/opensight/ai/coaching.py` (new insights generator)

```python
def generate_coaching_insights(stats: PlayerStats) -> List[str]:
    """Generate actionable coaching feedback from metrics"""
    insights = []
    
    # TTD analysis
    if stats.ttd_median_ms < 150:
        insights.append("⚡ Exceptional timing - very fast to react")
    elif stats.ttd_median_ms > 400:
        insights.append("⚠️ Timing needs work - slow to open fire after sighting enemy")
    
    # CP analysis
    if stats.cp_median_error < 4:
        insights.append("✓ Excellent crosshair placement - positioning is strong")
    elif stats.cp_median_error > 8:
        insights.append("⚠️ Crosshair placement needs improvement - practice pre-aiming")
    
    # Entry frag analysis
    entry_sr = stats.entry_kills / max(stats.entry_attempts, 1)
    if entry_sr > 0.6:
        insights.append(f"✓ Strong entry fragging ({entry_sr*100:.0f}% - consider as lurker alternative")
    elif entry_sr < 0.3:
        insights.append(f"⚠️ Entry frag success low ({entry_sr*100:.0f}%) - needs team support or positioning adjustment")
    
    # Trade kill analysis
    if stats.trade_kills > stats.kills * 0.3:
        insights.append("✓ Excellent trading - strong team player")
    
    # Clutch analysis
    if stats.clutch_attempts > 0:
        clutch_wr = stats.clutch_wins / stats.clutch_attempts
        if clutch_wr > 0.4:
            insights.append(f"✓ Strong clutch player ({clutch_wr*100:.0f}% win rate)")
        elif clutch_wr < 0.2:
            insights.append("⚠️ Struggle in clutches - consider watching demo analysis")
    
    return insights
```

## Testing the Integration

### Test Script

```python
# test_enhanced_integration.py
import json
from opensight.infra.cache import CacheManager
from opensight.core.enhanced_parser import CoachingAnalysisEngine

def test_enhanced_metrics():
    """Test enhanced parser integration"""
    
    demo_path = "/path/to/test_demo.dem"
    
    # Test enhanced parser directly
    engine = CoachingAnalysisEngine()
    metrics = engine.analyze_demo(demo_path)
    
    print("Enhanced Metrics Extracted:")
    print(json.dumps(metrics, indent=2))
    
    # Verify key metrics present
    assert 'entry_frags' in metrics
    assert 'ttd_metrics' in metrics
    assert 'crosshair_placement' in metrics
    assert 'clutch_stats' in metrics
    
    print("✓ All enhanced metrics present")

def test_cache_integration():
    """Test enhanced parser in cache pipeline"""
    
    cache = CacheManager()
    demo_path = "/path/to/test_demo.dem"
    
    # Clear cache
    cache.delete(f"demo:{demo_path}")
    
    # Analyze with cache (should use enhanced parser)
    result = cache.analyze(demo_path)
    
    # Verify enhanced metrics
    assert 'entry_frags' in result
    assert 'ttd_metrics' in result
    
    print("✓ Enhanced metrics working in cache pipeline")

def test_memory_efficiency():
    """Test memory efficiency with large demos"""
    
    import tracemalloc
    tracemalloc.start()
    
    demo_path = "/path/to/large_demo.dem"  # 500MB+ file
    engine = CoachingAnalysisEngine()
    
    metrics = engine.analyze_demo(demo_path)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory used: {peak / 1024 / 1024:.1f}MB")
    
    # Should be <100MB even for large files
    assert peak < 100 * 1024 * 1024, "Memory usage exceeds limit"
    
    print("✓ Memory efficient processing confirmed")

if __name__ == "__main__":
    test_enhanced_metrics()
    test_cache_integration()
    test_memory_efficiency()
    print("\nAll integration tests passed! ✓")
```

## Verification Checklist

- [ ] Enhanced parser processes demo without errors
- [ ] TTD metrics reasonable (100-500ms range for normal players)
- [ ] CP metrics reasonable (2-15° range)
- [ ] Entry/Trade/Clutch detection working correctly
- [ ] Memory usage stays <100MB for any demo size
- [ ] API returns enhanced metrics
- [ ] Web UI displays metrics correctly
- [ ] Database stores all metric values
- [ ] Coaching insights generate appropriate feedback

## Performance Targets

| Metric | Target |
|--------|--------|
| TTD Median | 150-350ms (pro players) |
| TTD Accuracy | ±10ms from manual verification |
| CP Error | 3-8° (pro players) |
| Entry Frag Detection | 95%+ accuracy |
| Trade Kill Detection | 90%+ accuracy |
| Processing Time | <60s for 500MB file |
| Memory Usage | <100MB peak |

## Rollback Plan

If issues occur:

1. Comment out enhanced parser calls in cache.py
2. Revert to basic DemoAnalyzer for analysis
3. Run cache clear to refresh results
4. Debug enhanced parser independently

```python
# Temporary rollback
def analyze(self, demo_path: str):
    # Use basic analyzer only
    analyzer = DemoAnalyzer(demo_path)
    result = analyzer.parse()
    self.set(cache_key, result, ttl=86400)
    return result
```

## Next Steps After Integration

1. **Validate metrics accuracy** with known player data
2. **Add more coaching metrics**:
   - Spray control analysis
   - Economy efficiency
   - Utility usage patterns
   - Positioning heatmaps
3. **Create coaching dashboard** with tactical feedback
4. **Add performance tracking** over time (player improvement)
5. **Implement peer comparison** (compare to similar rank players)
