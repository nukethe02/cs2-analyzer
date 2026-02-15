# Implementation Plan: V4 Comprehensive Bug Fix (40 Bugs)

## Scope
- **Files**: ~15 files across parser, analytics, orchestrator, frontend, domains
- **Risk Level**: HIGH — touches core metrics pipeline, score calculation, damage math
- **Estimated Changes**: ~15 files, ~600-800 lines modified
- **Branch**: `claude/plan-issue-fixes-RFhB0`

## Root Cause → Bug Mapping

The 40 bugs stem from 8 root causes. Fixing root causes first eliminates cascading bugs efficiently.

| Root Cause | Bugs Fixed |
|-----------|-----------|
| RC1: Knife round not filtered | BUG-004,005,007,008,009,017,027,029,033,040 |
| RC2: Score calculation | BUG-001,002 |
| RC3: Team assignment | BUG-003 |
| RC4: Damage uses overkill (dmg_health_real) | BUG-006,014 |
| RC5: TTD/CP algorithm issues | BUG-010,011 |
| RC6: Trade proximity too loose | BUG-013 |
| RC7: Frontend rendering bugs | BUG-018,026,031,034,037 |
| RC8: Feature-level bugs | BUG-015,016,019,020,021,022,023,024,025,028,030,032,035,036,038,039 |

---

## Phase 1: Knife Round Filtering (ROOT CAUSE #1)

### Problem
The knife round detection EXISTS in `orchestrator.py:58-83` and `_strip_knife_round()` at lines 149-212, but it's FAILING for this specific demo. The audit shows 31 rounds, meaning the knife round was NOT stripped.

### Investigation Required
The `is_knife_round()` function at line 58 checks:
1. All kills are knife weapons → True
2. Round 1 with majority knife kills → True

**Why it might fail:**
- Round numbering might start at 0 instead of 1 (demoparser2 quirk)
- The knife round might use `round_num=0` and the check looks for `round_num=1`
- The weapons might not match the `KNIFE_WEAPONS` frozenset (e.g., different casing, missing variant)

### Files to Modify
1. `src/opensight/pipeline/orchestrator.py` — Lines 58-83, 111-122, 149-212

### Steps
1. **Read actual demo data** to check what round numbers and weapons appear in round 1/0
2. **Fix `_get_knife_round_num()`** (line 111-121): Check BOTH round 0 AND round 1
3. **Fix `is_knife_round()`** (line 58-83): Make weapon matching more robust (case-insensitive, handle weapon prefix variants)
4. **Add logging** to `_strip_knife_round()` to report what weapons are found in each candidate knife round
5. **Verify**: After fix, round count should be 30, kill count should be 224

### Specific Code Change
```python
# orchestrator.py line 111-121: Check round 0 AND round 1
def _get_knife_round_num(self, demo_data) -> int | None:
    all_kills = getattr(demo_data, "kills", [])
    # Check round 0 first (some demos number warmup as 0)
    for candidate_round in [0, 1]:
        round_kills = [k for k in all_kills if getattr(k, "round_num", -1) == candidate_round]
        if round_kills and is_knife_round(round_kills, round_num=candidate_round):
            self._knife_round_num_cache = candidate_round
            return candidate_round
    self._knife_round_num_cache = None
    return None
```

Also fix `is_knife_round()` to accept round 0 as a valid knife round:
```python
# Line 81: Accept round 0 OR round 1
if round_num <= 1 and knife_kills > len(weapons) / 2:
    return True
```

### Bugs Fixed
- BUG-004: 31→30 rounds
- BUG-005: 231→224 kills
- BUG-007: Per-player kills corrected (-1 to -2)
- BUG-008: Per-player deaths corrected (-1)
- BUG-009: Assists partially corrected
- BUG-017: Shots fired corrected (partially)
- BUG-027: Highlights no longer include knife round
- BUG-029: KAST denominator becomes 30
- BUG-033: 2D Replay K/D/A corrected
- BUG-040: Highlights count excludes knife round

---

## Phase 2: Score Calculation (ROOT CAUSE #2)

### Problem
Two separate score paths produce different wrong results:
- Overview: 15-15 TIE
- Match Details: 16-15

### Files to Modify
1. `src/opensight/pipeline/orchestrator.py` — Lines 123-147, 648-661

### Investigation
- Line 656-657: `score_ct: analysis.team1_score` and `score_t: analysis.team2_score` come from `DemoAnalyzer`, which may include the knife round in its score
- Line 655: `_format_score()` uses `_compute_real_score()` which DOES exclude knife round
- The inconsistency: `demo_info.score` uses the corrected score, but `score_ct`/`score_t` use the analyzer's uncorrected score

### Steps
1. **Fix `demo_info` dict** (line 648-661): Use `_compute_real_score()` for ALL score fields
2. **Verify `_compute_real_score()`** actually produces 16-14 for the test demo
3. **Ensure frontend reads the corrected fields** consistently

### Specific Code Change
```python
# orchestrator.py lines 648-661
ct_score, t_score, real_rounds = self._compute_real_score(demo_data)
result = {
    "demo_info": {
        "score": f"{ct_score} - {t_score}",
        "score_ct": ct_score,      # Was: analysis.team1_score
        "score_t": t_score,        # Was: analysis.team2_score
        "rounds": real_rounds,
        ...
    }
}
```

### Bugs Fixed
- BUG-001: Score shows correct 16-14
- BUG-002: Match Details shows correct 16-14

---

## Phase 3: Damage Calculation — Use Capped Damage (ROOT CAUSE #4)

### Problem
`_calculate_basic_stats()` at line 681 uses column priority `["dmg_health_real", "dmg_health", "damage", "dmg"]`. The `dmg_health_real` column includes OVERKILL damage (e.g., shooting a 10 HP player with an AWP records 100+ damage). Leetify uses CAPPED damage (max 100 per hit per player).

The RWS calculation at line 884-885 already caps at 100: `damage = min(damage, 100)`. But the ADR/total_damage calculation does NOT cap.

### Evidence
Total damage inflation of 20-41% is WAY too high for just a knife round. Example: Tr1d has 3758 (OS) vs 2670 (Leetify) = +41%. Even removing ~200 knife round damage, that's still ~3558 vs 2670 = +33%. The overkill damage is the dominant factor.

### Files to Modify
1. `src/opensight/analysis/analytics.py` — Line 681, 735

### Steps
1. **Change column priority** to prefer `dmg_health` over `dmg_health_real`
2. **Add damage cap** of 100 per event in the total damage summation
3. **Apply same fix** to ALL damage summation paths (search for `dmg_health_real`)

### Specific Code Change
```python
# analytics.py line 681: Remove dmg_health_real from priority list
dmg_col = (
    self._find_col(damages_df, ["dmg_health", "damage", "dmg"])
    if not damages_df.empty
    else None
)

# OR: Keep dmg_health_real but cap at 100 per event
# analytics.py line 735: Cap damage
player_dmg_values = player_dmg[dmg_col].clip(upper=100)
player.total_damage = int(player_dmg_values.sum())
```

The preferred approach is to use `dmg_health` (actual health removed) instead of `dmg_health_real` (theoretical damage before armor/HP cap). If `dmg_health` is not available, fall back and cap at 100.

### Bugs Fixed
- BUG-006: ADR corrected to match Leetify
- BUG-014: Total damage corrected

---

## Phase 4: TTD Calculation Fix (BUG-010)

### Problem
TTD values are 3-25x too low. The audit shows 23ms for one player, which is physically impossible. The algorithm at `compute_aim.py:343-344` computes:
```python
ttd_ms = (kill_tick - first_dmg_tick) * MS_PER_TICK
```

### Hypotheses (need investigation)
1. **MS_PER_TICK calculation error**: If tick rate is 64, MS_PER_TICK should be 1000/64 = 15.625ms. A value of 23ms suggests only 1-2 ticks between first damage and kill, which could happen if the damage cache is finding damage events TOO close to the kill (same engagement, not first damage in encounter).
2. **Engagement window too narrow**: The 5-second max engagement window might be correct, but the FIRST damage lookup might find damage from the SAME burst rather than the start of the encounter.
3. **TTD_MAX_MS too restrictive**: If set to ~1500ms, many real TTD values (500-700ms per Leetify) would be included, so this isn't the issue.

### Investigation Required
- Check `analyzer.MS_PER_TICK` value
- Check `analyzer.TTD_MIN_MS` and `analyzer.TTD_MAX_MS` values
- Check how `first_dmg_tick` is found — is it finding damage from the same burst?

### Files to Modify
1. `src/opensight/analysis/compute_aim.py` — TTD computation function
2. `src/opensight/core/constants.py` — TTD constants if needed

### Steps
1. **Read the TTD algorithm in detail** (compute_aim.py lines 216-465)
2. **Verify MS_PER_TICK** calculation matches tick rate
3. **Fix first damage detection**: The "first damage" should be the FIRST damage in an encounter (not a damage event from the same spray burst as the kill)
4. **Consider**: TTD may need to measure time from FIRST SIGHTING to first damage, not first damage to kill. Leetify's TTD is "time to damage" = time from being in LOS of enemy to dealing first damage.
5. **Test with demo data** to verify corrected values are in 200-700ms range

### Notes
This is the hardest fix because Leetify's exact TTD definition may differ from ours:
- **Leetify TTD**: Time from spotting an enemy to dealing first damage (reaction time + aim time)
- **Our TTD**: Time from first damage received to kill (engagement duration)

If these are fundamentally different metrics, we may need to rename ours or rewrite the algorithm. The CLAUDE.md already notes: "This is NOT reaction time. This measures how long it took to finish a kill after first dealing damage." But Leetify's metric IS closer to reaction time.

### Bugs Fixed
- BUG-010: TTD values in realistic range
- BUG-030: AI coaching based on realistic TTD

---

## Phase 5: CP Calculation Fix (BUG-011)

### Problem
CP values are 2-8x too low. Values like 1.2° suggest very tight aim when Leetify shows 9.13°.

### Hypotheses
1. **Distance filter too aggressive**: The MAX_DISTANCE of 2000 units (line 850) may filter out close-range kills where CP error is higher, biasing toward long-range kills with naturally better aim
2. **Eye height offset wrong**: Adding 64 units to Z (line 854) may not match demoparser2's position reference point
3. **Sampling bias**: Only kills WITH position data (~71%) are included, and those with position data may be biased toward certain engagement types
4. **Formula issue**: The arccos(dot) formula is mathematically correct for angular error, but the input data (attacker pitch/yaw) may be the FINAL aim position (at kill time) rather than aim position at first sight (which would show worse crosshair placement)

### Investigation Required
- **KEY INSIGHT**: The CP calculation uses kill-time angles, which are AFTER the player has already aimed at the victim. Leetify likely measures the crosshair angle BEFORE the engagement (initial crosshair placement when enemy first appears). Using kill-time angles would naturally produce LOWER errors because the player has already corrected their aim.

### Files to Modify
1. `src/opensight/analysis/compute_aim.py` — CP computation function

### Steps
1. **Verify the angle data source**: Are we using kill-time angles (post-correction) or initial engagement angles?
2. **If kill-time angles**: This is the wrong data point. We need pre-engagement crosshair position, which requires position data from ticks BEFORE the kill, not AT the kill.
3. **Fix**: Use attacker position/angles from tick data ~500ms-1000ms BEFORE the kill tick, not from the kill event itself
4. **Fallback**: If tick-level angle data isn't available, document this as a known limitation

### Bugs Fixed
- BUG-011: CP values in realistic range (5-11°)
- BUG-030: AI coaching based on realistic CP

---

## Phase 6: Trade Opportunity Inflation Fix (BUG-013)

### Problem
Trade opportunities are 2-5.75x higher than Leetify. Our detection uses:
- 5-second window (matches Leetify standard)
- 1500-unit proximity threshold

### Hypotheses
1. **Proximity too loose**: 1500 units may be too generous. Consider reducing to 1000-1200.
2. **Double-counting**: Each death may generate multiple "opportunities" if multiple teammates are nearby (one per nearby teammate), while Leetify counts one opportunity per death event.
3. **Position data timing**: Checking proximity at `kill_tick` but using tick data within 2 seconds (128 ticks) may pick up teammates who were far away but happened to be nearby 2 seconds later.

### Investigation Required
Read `compute_combat.py` lines 898-913 to check if opportunities are counted PER NEARBY TEAMMATE or PER DEATH EVENT.

### Files to Modify
1. `src/opensight/analysis/compute_combat.py` — Trade opportunity counting

### Steps
1. **Change opportunity counting**: Count one opportunity PER DEATH, not per nearby teammate
2. **Tighten proximity**: Reduce from 1500 to 1200 units
3. **Verify**: Trade opportunity counts should roughly match Leetify (within ~20%)

### Bugs Fixed
- BUG-013: Trade opportunities closer to Leetify
- BUG-028: Trades column "X/Y" format uses corrected denominator

---

## Phase 7: HLTV Rating Fix (BUG-022)

### Problem
HLTV ratings differ because they're computed from inflated inputs (inflated kills, deaths, ADR from knife round + overkill damage).

### Files to Modify
1. Primarily fixed by Phase 1 (knife round) and Phase 3 (damage cap)
2. `src/opensight/analysis/hltv_rating.py` — Verify formula coefficients

### Steps
1. After Phases 1+3, re-verify HLTV ratings match Leetify within ±0.05
2. If still off, check the formula coefficients against the latest published HLTV 2.0 analysis
3. Check RMK (rounds with multi-kills) is a decimal 0.0-1.0, not a percentage

### Bugs Fixed
- BUG-022: HLTV ratings corrected
- BUG-023: MVP correct (ina at 1.50)
- BUG-038: Rating rank order correct

---

## Phase 8: HE/Fire Damage Attribution (BUG-015, BUG-016)

### Problem
Tr1d shows 0 HE damage (should be 269), DavidLithium shows 0 fire damage (should be 78).

### Investigation Required
The HE damage code at `compute_utility.py:463-475` filters by weapon names:
```python
he_weapons = ["hegrenade", "he_grenade", "grenade_he", "hegrenade_projectile"]
```
If the demo uses a different weapon string (e.g., "HEGrenade" with different casing, or just "grenade"), it won't match.

### Files to Modify
1. `src/opensight/analysis/compute_utility.py` — HE/molotov weapon name matching

### Steps
1. **Log actual weapon names** from the damages_df for grenade damage events
2. **Add missing weapon variants** to the match lists
3. **Make matching case-insensitive** (use `.str.lower()` before comparison)
4. **Also check**: The attacker SteamID matching — could be float precision issue (Rule 6 from the AI coding rules: 17-digit SteamID through float64 loses precision)

### Bugs Fixed
- BUG-015: Tr1d HE damage = 269
- BUG-016: DavidLithium fire damage = 78

---

## Phase 9: Counter-Strafing Fix (BUG-012)

### Problem
Counter-strafing percentages are ~40-50% of Leetify values. Our threshold is 34 u/s for "stationary."

### Hypotheses
1. **Threshold too strict**: 34 u/s may be too low. Leetify may use a higher threshold (50-80 u/s) or a different metric entirely (deceleration pattern rather than absolute velocity)
2. **Different measurement point**: We measure at shot time; Leetify may measure PEAK velocity during the counter-strafe pattern
3. **Weapon-specific thresholds**: Rifles have different speed thresholds than SMGs for accuracy

### Files to Modify
1. `src/opensight/analysis/compute_aim.py` — Counter-strafe threshold and algorithm

### Steps
1. **Research Leetify's counter-strafe metric** (if publicly documented)
2. **Increase threshold** to match Leetify output. Try 64 u/s (half walk speed) or weapon-specific accurate speed thresholds from CS2
3. **Test with demo data** to verify values are in 60-86% range matching Leetify

### Bugs Fixed
- BUG-012: Counter-strafing closer to Leetify values

---

## Phase 10: Frontend Fixes

### BUG-026: Head to Head Missing 5 Matchups
**File**: `src/opensight/static/index.html` — Line 6616
**Fix**: Change `.slice(0, 20)` to `.slice(0, 25)` (10 players = 25 cross-team pairs max), or remove the limit entirely for cross-team pairs.

### BUG-031: Encoding "Â°" → "°"
**File**: `src/opensight/pipeline/orchestrator.py` — Lines 2667, 2679, 3691
**Fix**: Replace `Â°` with `°` in all three locations.

### BUG-034: Partial SteamIDs in Trends Dropdown
**File**: Find the Trends dropdown population code in index.html
**Fix**: Remove the SteamID suffix from player names in dropdown options.

### BUG-037: Kill Matrix Player Order
**File**: `src/opensight/static/index.html` or `orchestrator.py`
**Fix**: Sort players by team in the kill matrix rendering.

### BUG-018: Opening Duels "Most Killed/Most Died To" All N/A
**Investigation**: This feature may not be implemented at all (grep found no `most_killed` code).
**Fix Options**:
a) Implement opponent frequency tracking in `compute_combat.py` opening duel detection
b) Remove the UI column if the data doesn't exist (don't show N/A for unimplemented features)

### Bugs Fixed
- BUG-018, BUG-026, BUG-031, BUG-034, BUG-037

---

## Phase 11: Feature-Level Bug Fixes

### BUG-019: Clutch Detection Under-counting
**File**: `src/opensight/analysis/compute_combat.py` — Lines 1183-1198
**Issue**: Clutch win requires `enemies_killed >= 1` for 1vN (N≥2). But bomb explosion wins where the clutcher planted and hid (0 kills) should still count.
**Fix**: Remove the kill requirement for bomb plant wins (check round end reason).

### BUG-020: Spray Transfer Detection Non-functional
**File**: `src/opensight/domains/combat.py` — Lines 913-1008
**Issue**: The detection is in `domains/combat.py`, but need to verify it's actually CALLED from the pipeline.
**Fix**: Ensure `_analyze_spray_transfers()` is called and results are stored on player models.

### BUG-021: Synergy Shows No Data
**File**: `src/opensight/domains/synergy.py`
**Issue**: The `analyze_synergy()` function exists and `synergy_to_dict()` exists (line 596). The exception handler at orchestrator.py:611 silently catches errors.
**Fix**: Add logging to identify the actual exception. Likely a data structure mismatch or missing column.

### BUG-024: Heatmap Kills ≠ Deaths
**File**: `src/opensight/pipeline/orchestrator.py` — `_build_heatmap_data()`
**Fix**: Ensure kills and deaths use the same event filtering (both exclude knife round, both exclude suicides).

### BUG-025: Heatmap Zone Totals Don't Sum
**File**: `src/opensight/core/map_zones.py` or heatmap builder
**Fix**: Add an "Unclassified" zone for events outside defined zones.

### BUG-032: 2D Replay Blank
**File**: `src/opensight/visualization/replay.py` and frontend
**Fix**: Debug the replay data generation. Check if tick-level position data exists in the demo.

### BUG-036: AI Tabs Require API Key
**File**: `src/opensight/static/index.html` and `src/opensight/api/routes_analysis.py`
**Fix**: Check for API key before showing AI tabs. Show helpful message instead of raw env var name.

---

## Phase 12: Derived/Cascade Fixes (Auto-fixed by Earlier Phases)

These bugs are fixed automatically when root causes are addressed:

| Bug | Fixed By |
|-----|----------|
| BUG-009: Assists +1-3 | Phase 1 (knife round) — partial; residual may need investigation |
| BUG-017: Shots inflated | Phase 1 (knife round) |
| BUG-022: HLTV rating wrong | Phase 1 + Phase 3 |
| BUG-023: MVP wrong | Phase 7 |
| BUG-028: Trades X/Y format | Phase 6 |
| BUG-029: KAST on 31 rounds | Phase 1 |
| BUG-030: AI coaching wrong | Phase 4 + Phase 5 + Phase 1 + Phase 3 |
| BUG-033: Replay K/D/A | Phase 1 |
| BUG-035: Trends time period | Phase 10 (minor UX) |
| BUG-038: Rating rank wrong | Phase 7 |
| BUG-039: K/D values off | Phase 1 |
| BUG-040: Highlights count | Phase 1 |

---

## Test Strategy

### Existing Tests to Verify
```bash
PYTHONPATH=src pytest tests/test_api.py -v
PYTHONPATH=src pytest tests/test_analytics.py -v
PYTHONPATH=src pytest tests/test_metrics.py -v
PYTHONPATH=src pytest tests/test_e2e_pipeline.py -v
```

### New Tests Needed
1. **test_knife_round_filtering**: Verify knife round detection for round_num=0 AND round_num=1
2. **test_damage_cap**: Verify total_damage uses capped values (max 100 per hit)
3. **test_score_calculation**: Verify score excludes knife round
4. **test_trade_opportunity_counting**: Verify one opportunity per death, not per teammate

### Manual Verification
After all fixes, re-analyze the Leetify match demo and compare:
- [ ] Round count = 30
- [ ] Kill count = 224
- [ ] Score = 16-14 (CT wins)
- [ ] ADR values within ±5 of Leetify
- [ ] TTD values in 400-700ms range
- [ ] CP values in 4-11° range
- [ ] Trade opportunities within ±30% of Leetify
- [ ] HLTV ratings within ±0.05 of Leetify

---

## Risk Assessment

### Security
- [x] No user input handling changes
- [x] No auth changes
- [x] No data exposure
- [x] No rate limiting changes

### Performance
- [x] Damage cap adds negligible cost (`.clip(upper=100)`)
- [x] Knife round filtering already exists, just fixing detection
- [ ] Trade opportunity recount may change performance slightly (minor)

### Breaking Changes
- [x] API response values will change (correct values vs incorrect)
- [x] Cached results will show old incorrect values until cache is cleared
- [x] No schema changes
- [x] No config changes

### Cache Invalidation
After deploying, we should clear the analysis cache to prevent serving stale (incorrect) results:
```bash
curl -X POST http://localhost:7860/cache/clear
```

---

## Rollback Plan
- Git revert is sufficient (single branch, all changes committed atomically)
- No database migrations
- No feature flags needed
- Cache clear may be needed after rollback

---

## Implementation Order (Dependency-Aware)

```
Phase 1 (Knife Round) ─── MUST BE FIRST ───┐
Phase 2 (Score)                              │
Phase 3 (Damage Cap)                         ├──→ Phase 7 (HLTV Rating)
Phase 4 (TTD) ── independent ──              │         │
Phase 5 (CP) ── independent ──              │         ├──→ Phase 12 (Cascades)
Phase 6 (Trades) ── independent ──           │
Phase 8 (HE/Fire) ── independent ──         │
Phase 9 (Counter-strafe) ── independent ──  │
Phase 10 (Frontend) ── independent ──────────┘
Phase 11 (Features) ── independent ──
```

Phases 4-6, 8-11 can be done in parallel after Phase 1.

---

## Open Questions

1. **TTD metric definition**: Is our TTD (engagement duration: first damage to kill) the SAME metric as Leetify's TTD? Or is Leetify measuring something different (time from LOS to first damage)? This fundamentally changes the fix approach.

2. **CP measurement point**: Are we using kill-time angles or pre-engagement angles? Need to verify against tick data.

3. **Counter-strafe threshold**: What threshold does Leetify use? 34 u/s may be CS:GO era; CS2 may use different values.

4. **Damage column**: Does `dmg_health` exist in the demo's damages_df? If only `dmg_health_real` exists, we need to cap manually.

5. **Synergy failure**: Need to check the actual exception message (currently silently caught).

6. **Assists inflation (+3)**: After knife round fix, are assists still inflated? If so, the assist counting algorithm needs investigation.
