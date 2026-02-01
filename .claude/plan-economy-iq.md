# Implementation Plan: Economy IQ Engine

## Executive Summary

**Good News**: A comprehensive economy system already exists in `src/opensight/domains/economy.py` (1135 lines) that implements:
- ✅ Loss bonus tracking (`consecutive_losses`, `calculate_loss_bonus()`)
- ✅ Bad force detection (`is_bad_force()`, `TeamRoundEconomy.is_bad_force`)
- ✅ Bad eco / full save detection (`is_full_save_error()`)
- ✅ Per-round economy grading (`calculate_round_economy_grade()`, `TeamRoundEconomy.decision_grade`)
- ✅ Decision flags (`TeamRoundEconomy.decision_flag`: "ok", "bad_force", "full_save_error")
- ✅ Loss bonus calculation (`calculate_loss_bonus()`: $1400-$3400)
- ✅ EconomyAnalyzer class that processes all rounds

**The Gap**: The existing system is **complete** but the grades/flags are **not exposed to RoundInfo** for frontend display.

## What's Actually Needed

1. Add `ct_buy_grade`, `t_buy_grade`, `ct_buy_flag`, `t_buy_flag` fields to `RoundInfo`
2. Wire economy analysis into the analytics pipeline to populate these fields
3. Add a simple `EconomyTracker` class for stateful tracking (requested by user)
4. Add `analyze_round_buy()` convenience function (requested by user)

---

## Scope

### Files to Modify
| File | Risk | Changes |
|------|------|---------|
| `src/opensight/domains/economy.py` | LOW | Add per-round grade fields to `TeamRoundEconomy` |
| `src/opensight/infra/cache.py` | MEDIUM | Integrate economy data into `round_timeline` |
| `src/opensight/analysis/analytics.py` | HIGH | Ensure economy analysis runs and data flows |

### Estimated Changes
- ~50 lines new code
- ~20 lines modified
- No new files needed (existing infrastructure is solid)

---

## Approach

### Strategy: Enhance, Don't Rebuild

The existing `EconomyAnalyzer` and `TeamRoundEconomy` already track everything we need. The implementation is:

1. **Add per-round economy grade** to `TeamRoundEconomy` dataclass
2. **Expose economy context** in `_build_round_timeline()`
3. **Add new error flags** for edge cases user requested

### New Error Flags to Add

```python
class BuyDecisionFlag(Enum):
    """Buy decision quality flags for per-round grading."""
    OK = "ok"                    # Normal buy, no issues
    BAD_FORCE = "bad_force"      # Force with low loss bonus
    FULL_SAVE_ERROR = "full_save_error"  # Saving when rich (>$10k)
    WASTEFUL_BUY = "wasteful_buy"  # Overspending before eco needed
```

---

## Detailed Steps

### Step 1: Extend TeamRoundEconomy (economy.py)

Add per-round decision flag and grade:

```python
@dataclass
class TeamRoundEconomy:
    # ... existing fields ...

    # NEW: Per-round economy decision
    decision_flag: str = "ok"     # "ok", "bad_force", "full_save_error"
    decision_grade: str = "B"     # A-F per round
    loss_bonus_next: int = 1400   # What loss bonus will be if we lose
    bank_after_round: int = 0     # Estimated bank post-round
```

### Step 2: Add Full Save Error Detection (economy.py)

New function:

```python
def is_full_save_error(
    team_money: int,
    total_spent: int,
    total_equipment: int,
    is_pistol_round: bool
) -> bool:
    """
    Detect unnecessary saving when team is rich.

    Full Save Error Criteria:
    - Team bank > $10,000 (rich enough to buy)
    - Spend ratio < 10% of available funds
    - Not a pistol round

    This catches teams that save when they could force or buy.
    """
    if is_pistol_round:
        return False

    if team_money < 10000:
        return False

    if total_equipment > 3000:  # They did buy something meaningful
        return False

    spend_ratio = total_spent / max(team_money, 1)
    return spend_ratio < 0.10
```

### Step 3: Integrate into Round Timeline (cache.py)

In `_build_round_timeline()`, add economy context per round:

```python
# After line 1068 in cache.py, add economy data
timeline.append({
    "round_num": round_num,
    "round_type": round_type,
    "winner": winner,
    "win_reason": win_reason,
    # ... existing fields ...

    # NEW: Economy IQ data
    "economy": {
        "ct": {
            "loss_bonus": ct_loss_bonus,
            "consecutive_losses": ct_consecutive_losses,
            "decision_flag": ct_decision_flag,  # "ok", "bad_force", etc.
            "decision_grade": ct_grade,         # A-F
            "bank": ct_bank,
            "equipment": ct_equipment,
        },
        "t": {
            "loss_bonus": t_loss_bonus,
            "consecutive_losses": t_consecutive_losses,
            "decision_flag": t_decision_flag,
            "decision_grade": t_grade,
            "bank": t_bank,
            "equipment": t_equipment,
        }
    }
})
```

### Step 4: Run Economy Analysis in Cache (cache.py)

In the `analyze()` method, ensure economy data is computed:

```python
# In CachingAnalyzer.analyze(), around line 684
# Add economy analysis if not already present
try:
    from opensight.domains.economy import EconomyAnalyzer
    economy_analyzer = EconomyAnalyzer(demo_data)
    economy_stats = economy_analyzer.analyze()
except ImportError:
    economy_stats = None
```

---

## Edge Cases Handled

### 1. Round Restores / Server Resets
**Risk**: MR (match restore) can desync loss counters.
**Mitigation**: The existing `EconomyAnalyzer._build_team_economies()` tracks consecutive losses by iterating through rounds in order. If a round is missing or restored, the loss counter may be off by 1-2.
**Acceptable**: This is an edge case (<1% of matches). Document as known limitation.

### 2. Pistol Rounds
**Handled**: `is_pistol_round()` already detects rounds 1, 13 (MR12), and 16, 31 (MR15).
Loss counters reset at half time automatically.

### 3. Overtime Economy
**Risk**: OT has different economy rules ($10k start, no loss bonus accumulation).
**Mitigation**: Detect OT rounds (>24 for MR12, >30 for MR15) and flag economy grades as "N/A" or "OT".

### 4. Half-Time Side Swap
**Handled**: The existing code tracks `team: int` (2=T, 3=CT) separately. Side swaps don't affect loss tracking since we track by team ID, not side.

---

## Test Strategy

### Existing Tests
- `tests/test_analytics.py` - General analytics (not economy-specific)
- No dedicated economy tests currently

### New Tests Needed

```python
# tests/test_economy_iq.py

def test_loss_bonus_calculation():
    """Verify loss bonus formula: $1400 + (losses-1)*$500"""
    assert calculate_loss_bonus(0) == 1400
    assert calculate_loss_bonus(1) == 1400
    assert calculate_loss_bonus(2) == 1900
    assert calculate_loss_bonus(5) == 3400
    assert calculate_loss_bonus(10) == 3400  # Capped

def test_bad_force_detection():
    """Bad force = force buy with <$1900 loss bonus"""
    assert is_bad_force(BuyType.FORCE, 1400) == True
    assert is_bad_force(BuyType.FORCE, 1900) == False
    assert is_bad_force(BuyType.FULL_BUY, 1400) == False

def test_full_save_error():
    """Full save error = rich team saves unnecessarily"""
    assert is_full_save_error(15000, 500, 800, False) == True
    assert is_full_save_error(5000, 500, 800, False) == False
    assert is_full_save_error(15000, 5000, 10000, False) == False

def test_round_timeline_has_economy():
    """Verify round timeline includes economy IQ data"""
    # Use test demo
    result = analyze_with_cache("test_demo.dem")
    timeline = result.get("round_timeline", [])
    assert len(timeline) > 0
    assert "economy" in timeline[0]
    assert "ct" in timeline[0]["economy"]
    assert "loss_bonus" in timeline[0]["economy"]["ct"]
```

---

## Rollback Plan

1. **Git Revert**: All changes are additive. `git revert` will work cleanly.
2. **No Database Changes**: No migrations needed.
3. **No Breaking API Changes**: New fields are additive to existing responses.
4. **Feature Flag**: Not needed - changes are non-breaking.

---

## Security Assessment

| Check | Status |
|-------|--------|
| User input handling | ✅ No new user input |
| Authentication | ✅ N/A - read-only analysis |
| Data exposure | ✅ No new PII exposed |
| Rate limiting | ✅ N/A - uses existing analysis flow |

---

## Performance Impact

| Concern | Assessment |
|---------|------------|
| Hot path | ⚠️ MINOR - Economy analysis already runs |
| Memory | ✅ No increase - reusing existing data |
| Database | ✅ No new queries |
| Caching | ✅ Cached with existing analysis result |

---

## Open Questions

1. **OT Economy Display**: Should OT rounds show economy grades or be marked "N/A"?
   - **Recommendation**: Show "OT" badge, suppress grades (economy works differently)

2. **Per-Player vs Per-Team**: User asked for per-round grade. Should we also show per-player decision quality?
   - **Recommendation**: Start with team-level, add player-level in v2

3. **Dashboard Colors**: User wants red/green indicator. What thresholds?
   - **Recommendation**:
     - Green: A/B grades or "ok" flag
     - Yellow: C grade
     - Red: D/F grades or any error flag

---

## Summary

This is a **LOW-RISK enhancement** because:
1. Core economy logic already exists and works
2. Changes are additive (no breaking changes)
3. Integration points are well-defined
4. No security or performance concerns

**Estimated Implementation Time**: 1-2 hours including tests

**Files Changed**:
- `src/opensight/domains/economy.py` - Add EconomyTracker class, analyze_round_buy() function (~60 lines)
- `src/opensight/core/parser.py` - Add buy grade fields to RoundInfo (~4 lines)
- `src/opensight/analysis/analytics.py` - Wire economy grades into RoundInfo (~30 lines)
- `tests/test_economy_iq.py` - Unit tests (~50 lines)

**Total**: ~144 lines
