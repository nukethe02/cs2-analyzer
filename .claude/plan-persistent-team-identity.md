# Implementation Plan: Persistent Team Identity Fix

## Executive Summary

**Surprising Finding:** The "Tick Share" system is **already implemented** in `parser.py:1141-1288`.

The bug is **NOT** in the parser - it's in **analytics.py** which inconsistently uses the deprecated `player_teams` dict instead of the persistent team system.

---

## Current Architecture

### What Already Exists (parser.py)

| Component | Location | Status |
|-----------|----------|--------|
| `_resolve_persistent_teams()` | Lines 1141-1288 | ✅ **Already implemented** |
| `player_persistent_teams` | DemoData field | ✅ Works |
| `team_rosters` | DemoData field | ✅ Works |
| `team_starting_sides` | DemoData field | ✅ Works |
| `get_player_side_for_round()` | Lines 509-534 | ✅ Handles halftime swaps |
| `get_player_persistent_team()` | Lines 536-545 | ✅ Returns "Team A"/"Team B" |

### Current Tick Share Algorithm (Already Implemented)

```python
def _resolve_persistent_teams():
    # Step 1: Count appearances on team_num=2 vs team_num=3 across ALL events
    for df in [kills_df, damages_df]:
        for row in df:
            player_team_counts[steamid][team_num] += 1

    # Step 2: Assign each player to primary team_num (>50% appearances)
    primary_team_num = 2 if counts[2] > counts[3] else 3

    # Step 3: Group into rosters
    roster_2 = {players with primary_team_num == 2}
    roster_3 = {players with primary_team_num == 3}

    # Step 4: Determine starting sides from round 1 data
    # Team A = roster that started CT
    # Team B = roster that started T

    # Step 5: Detect MR12 vs MR15
    halftime_round = 16 if max_round > 24 else 13
```

---

## The Actual Bug: analytics.py Inconsistency

### Problem Locations (25 usages of deprecated system)

| Line | Code | Issue |
|------|------|-------|
| 2854 | `team = self.data.player_teams.get(steam_id, "Unknown")` | Uses deprecated dict at init |
| 2988-2992 | `player_starting_team[steam_id] = player.team` | Static team for clutch detection |
| 3630-3631 | `if player.team in ("CT", "T"): return player.team` | Entry team detection |
| 3670-3688 | `player_teams_lookup = {...}` | Trade detection team lookup |
| 3741-3755 | `victim_team = player_teams_lookup.get(...)` | Trade team matching |
| 4035-4036 | `vic_team = self.data.player_teams.get(victim_id)` | Flash stats |
| 4120-4151 | `player_teams = {sid: player.team for...}` | Trade detection rebuild |
| 5273 | `{steam_id: player.team for...}` | Team dict for zone stats |
| 6453 | `thrower_team = self.data.player_teams.get(...)` | Grenade thrower team |
| 6473 | `thrower_team = self.data.player_teams.get(...)` | Smoke stats |
| 6491 | `attacker_team = self.data.player_teams.get(...)` | Flash attacker team |
| 6727-6728 | `[p for p in players if p.team == "CT"]` | Team filtering |
| 6765 | `team = match_data.player_teams.get(steam_id)` | Utility metrics |

### Root Cause

`PlayerMatchStats.team` is a **static string** set once at init (line 2854):
```python
team = self.data.player_teams.get(steam_id, "Unknown")  # DEPRECATED!
```

But team side changes at halftime! Line 2854 should use:
```python
persistent_team = self.data.get_player_persistent_team(steam_id)  # "Team A" or "Team B"
```

And round-specific lookups should use:
```python
side = self.data.get_player_side_for_round(steam_id, round_num)  # "CT" or "T"
```

---

## Scope Analysis

### Files to Modify

| File | Lines | Risk | Changes |
|------|-------|------|---------|
| `analysis/analytics.py` | 7,882 | **HIGH** | ~25 fixes across 15 functions |
| `core/parser.py` | 1,600 | MEDIUM | Minor improvements (edge cases) |
| `api.py` | 3,500 | LOW | May need response structure update |

### Cross-Module Dependencies

```
parser.py (produces) → DemoData.player_persistent_teams
                     → DemoData.team_rosters
                     → DemoData.team_starting_sides
                     → DemoData.get_player_side_for_round()

analytics.py (consumes) → Currently uses DEPRECATED player_teams
                        → Should use persistent team methods

api.py (exposes) → Players with team field
                 → May need persistent_team field added
```

---

## Implementation Plan

### Phase 1: Improve Parser (Optional but Recommended)

**File:** `src/opensight/core/parser.py`

#### 1.1 Handle Overtime Edge Case

Current code uses max_round to detect halftime:
```python
halftime_round = 16 if max_round > 24 else 13
```

**Problem:** Overtime games can have 30+ rounds but still use MR12 (halftime at 13).

**Fix:** Count rounds per half, not max rounds:
```python
def _detect_match_format(self, kills_df, round_col) -> tuple[int, int]:
    """Detect MR12 vs MR15 by analyzing round score patterns."""
    # MR12: max 12 rounds per half (halftime at 13)
    # MR15: max 15 rounds per half (halftime at 16)
    # Overtime: Additional rounds after regulation

    # Look at score at round 12 vs round 15
    # If round 13 exists with team swap, it's MR12
    # If round 16 exists with team swap, it's MR15
```

#### 1.2 Handle Disconnects/Reconnects

Current algorithm counts event appearances. If player disconnects mid-half:
- They may have fewer appearances
- But majority rule still works (they appear on their team when connected)

**Verification:** Current algorithm handles this correctly - no change needed.

#### 1.3 Handle Substitute Players

Edge case: Player joins mid-match as a substitute.
- They only have appearances for the rounds they played
- Majority rule assigns them to correct team

**Verification:** Current algorithm handles this correctly - no change needed.

---

### Phase 2: Fix Analytics (Critical)

**File:** `src/opensight/analysis/analytics.py`

#### 2.1 Fix PlayerMatchStats Initialization (Line 2854)

**Before:**
```python
team = self.data.player_teams.get(steam_id, "Unknown")
self._players[steam_id] = PlayerMatchStats(
    ...
    team=team,  # Static "CT" or "T"
    ...
)
```

**After:**
```python
# Use persistent team identity for roster grouping
persistent_team = self.data.get_player_persistent_team(steam_id)
starting_side = self.data.team_starting_sides.get(persistent_team, "Unknown")

self._players[steam_id] = PlayerMatchStats(
    ...
    team=persistent_team,  # "Team A" or "Team B" - stable across halftime
    starting_side=starting_side,  # "CT" or "T" - their first-half side
    ...
)
```

**Note:** This changes `player.team` semantics from "CT/T" to "Team A/Team B".

#### 2.2 Add Round-Aware Side Accessor (New Helper Method)

```python
def _get_player_side(self, steam_id: int, round_num: int) -> str:
    """Get player's side (CT/T) for a specific round, handling halftime swaps."""
    return self.data.get_player_side_for_round(steam_id, round_num)
```

#### 2.3 Fix All Round-Specific Team Lookups

Replace all instances of:
```python
player.team  # or self.data.player_teams.get(steam_id)
```

With:
```python
self._get_player_side(steam_id, round_num)  # For round-specific analysis
# OR
player.team  # For persistent team grouping ("Team A"/"Team B")
```

**Specific Fixes:**

| Location | Current | Fix |
|----------|---------|-----|
| Line 2988-2992 | `player_starting_team[steam_id] = player.team` | Use `_get_player_side(steam_id, round_num)` |
| Line 3630-3631 | `return player.team` | Use `_get_player_side(steam_id, round_num)` |
| Line 3670-3688 | Build static team lookup | Build dynamic lookup per round |
| Line 4035-4036 | `self.data.player_teams.get(victim_id)` | Use `_get_player_side(victim_id, round_num)` |
| Line 4120 | `player.team for...` | Use `_get_player_side(sid, round_num)` |
| Line 6453 | `self.data.player_teams.get(grenade.player_steamid)` | Use `_get_player_side(steamid, round_num)` |
| Line 6473 | `self.data.player_teams.get(...)` | Use `_get_player_side(...)` |
| Line 6491 | `self.data.player_teams.get(...)` | Use `_get_player_side(...)` |
| Line 6727-6728 | `p.team == "CT"` | Use starting_side or round-specific |
| Line 6765 | `match_data.player_teams.get(steam_id)` | Use persistent team |

---

### Phase 3: Update API Response (Optional)

**File:** `src/opensight/api.py`

Add `persistent_team` to player response:
```python
def build_player_response(player):
    return {
        ...
        "team": player.team,  # "Team A" or "Team B"
        "starting_side": player.starting_side,  # "CT" or "T" (first half)
        ...
    }
```

---

## Edge Cases

### 1. Overtime Rounds

**Scenario:** Match goes to overtime (rounds 25+)

**Current Behavior:** Halftime detection uses `max_round > 24` → sets halftime to 16 (wrong for MR12 + OT)

**Fix:** Detect format by score pattern, not max rounds:
```python
# Check if round 13 had a team swap (score was 12-X or X-12)
# If yes: MR12 format (halftime at 13)
# If no: MR15 format (halftime at 16)
```

### 2. Player Disconnects Mid-Match

**Scenario:** Player disconnects in round 10, reconnects in round 18

**Current Behavior:** Counts appearances across all rounds they played

**Result:** Majority rule still works correctly
- Rounds 1-10: On team 2 (T)
- Rounds 18-24: On team 3 (CT) - but they're the SAME persistent team, just swapped sides

**Verification:** Algorithm correctly assigns them to "Team A" or "Team B" based on majority

### 3. Substitute Player

**Scenario:** Player X leaves, Player Y joins as substitute

**Current Behavior:** Player Y only has appearances from rounds they played

**Result:** Correctly assigned to the team that needed a substitute

### 4. Coach/Spectator Appears in Events

**Scenario:** Coach briefly appears in damage events

**Current Behavior:** May be assigned to a team with low consistency

**Fix:** Add consistency threshold - exclude players with <5 total appearances

### 5. Bot Takeover

**Scenario:** Player disconnects, bot takes over, player reconnects

**Current Behavior:** Bot has separate steamid (0 or bot-specific ID)

**Result:** Bot and player tracked separately - correct behavior

---

## Risk Assessment

### Security Implications
- [ ] User input handling? **NO** - Internal data processing only
- [ ] Authentication/authorization? **NO**
- [ ] Data exposure risks? **NO**
- [ ] Rate limiting needed? **NO**

### Performance Impact
- [ ] Hot path affected? **MINOR** - Adds one dict lookup per round-specific operation
- [ ] Database queries added? **NO**
- [ ] Memory usage increase? **NEGLIGIBLE** - Same data, different access pattern
- [ ] Caching implications? **NO**

### Breaking Changes
- [ ] API contract changes? **POSSIBLE** - `player.team` changes from "CT/T" to "Team A/B"
- [ ] Database schema changes? **NO**
- [ ] Configuration changes? **NO**
- [ ] Backward compatibility? **NEED MIGRATION** - Frontend may expect "CT"/"T"

---

## Test Strategy

### Existing Tests
- `tests/test_analytics.py` - Will need updates for new team semantics
- `tests/test_api.py` - May need response structure updates

### New Tests Needed

```python
class TestPersistentTeamIdentity:
    def test_player_team_consistent_across_halftime(self):
        """Player stays on same persistent team before and after halftime."""

    def test_player_side_swaps_at_halftime(self):
        """Player's side (CT/T) swaps at halftime round."""

    def test_overtime_halftime_detection(self):
        """Overtime games correctly detect MR12 vs MR15 halftime."""

    def test_disconnected_player_team_assignment(self):
        """Player who disconnects keeps correct team assignment."""

    def test_low_appearance_player_excluded(self):
        """Players with <5 appearances excluded from team assignment."""
```

### Manual Testing
1. Load a demo that went to halftime
2. Verify players show on correct team in dashboard
3. Check player stats are aggregated correctly per team
4. Verify round-by-round timeline shows correct sides

---

## Rollback Plan

1. **Git revert sufficient:** Yes - changes are code-only, no schema changes
2. **Feature flag needed:** No - but could add `USE_PERSISTENT_TEAMS` env var
3. **Database migration:** None required

---

## Implementation Steps

### Step 1: Add PlayerMatchStats.starting_side field
- Modify dataclass
- Update initialization

### Step 2: Add _get_player_side() helper
- Round-aware side lookup
- Delegates to DemoData.get_player_side_for_round()

### Step 3: Fix initialization (Line 2854)
- Use persistent team
- Set starting_side

### Step 4: Fix clutch detection (Lines 2988-2992)
- Use round-aware side lookup

### Step 5: Fix entry duel detection (Lines 3630-3631)
- Use round-aware side lookup

### Step 6: Fix trade detection (Lines 3670-3755, 4120-4151)
- Build dynamic team lookup per round

### Step 7: Fix utility stats (Lines 6453-6491, 6727-6765)
- Use round-aware side lookup

### Step 8: Update API response structure
- Add persistent_team field
- Document team field change

### Step 9: Update tests
- Fix expected values
- Add new edge case tests

### Step 10: Update frontend (if needed)
- Handle "Team A"/"Team B" team values
- Or map back to starting side for display

---

## Code: resolve_persistent_teams (Already Exists!)

The implementation already exists at `parser.py:1141-1288`. Here's the key algorithm:

```python
def _resolve_persistent_teams(self, kills_df, damages_df):
    """
    Tick Share System for Persistent Team Identity.

    Algorithm:
    1. Count each player's appearances on team_num=2 vs team_num=3
    2. Assign to primary team (>50% majority rule)
    3. Group into roster_2 and roster_3
    4. Determine starting sides from round 1 data
    5. Label: CT starters = "Team A", T starters = "Team B"
    """
    player_team_counts = defaultdict(lambda: {2: 0, 3: 0})

    # Count appearances across ALL events
    for df in [kills_df, damages_df]:
        for row in df:
            sid = row.steamid
            team_num = row.team_num  # 2 or 3
            player_team_counts[sid][team_num] += 1

    # Majority rule assignment
    for sid, counts in player_team_counts.items():
        total = counts[2] + counts[3]
        primary = 2 if counts[2] > counts[3] else 3
        player_primary_team_num[sid] = primary

    # Group into rosters
    roster_2 = {sid for sid, t in player_primary_team_num.items() if t == 2}
    roster_3 = {sid for sid, t in player_primary_team_num.items() if t == 3}

    # Determine starting sides from round 1
    # Count which roster had more CT appearances in round 1
    # Team A = whichever roster started CT

    return player_persistent_teams, team_rosters, team_starting_sides, halftime_round
```

---

## Open Questions

1. **API Breaking Change:** Should `player.team` return "Team A"/"Team B" or keep "CT"/"T"?
   - Option A: Change to "Team A"/"Team B" (cleaner, but breaking)
   - Option B: Add new `persistent_team` field, keep `team` as starting side

2. **Frontend Impact:** Does the dashboard need updates to handle new team format?
   - Need to check `index.html` for team-based styling

3. **Overtime Detection:** Should we improve halftime detection for overtime games?
   - Current: `max_round > 24` → MR15
   - Proposed: Analyze score patterns

---

## Summary

| Task | Effort | Risk |
|------|--------|------|
| Parser improvements | Low | Low |
| Analytics fixes (25 locations) | **High** | **Medium** |
| API updates | Low | Low |
| Test updates | Medium | Low |
| Frontend updates | Unknown | Medium |

**Recommended Approach:** Fix analytics.py to use the existing persistent team system. The parser already implements the tick share algorithm correctly.
