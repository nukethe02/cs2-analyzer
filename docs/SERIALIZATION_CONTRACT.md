# OpenSight Serialization Contract
## Model → Orchestrator → Frontend Field Mapping

**Last updated:** 2026-02-10
**Auto-verified by:** `tests/test_e2e_pipeline.py::TestSerializationCompleteness`

---

## Purpose

This document is the **single source of truth** for how data flows through the OpenSight pipeline:

```
Demo File → Parser → Analytics → PlayerMatchStats (models.py)
                                        ↓
                                  DemoOrchestrator (orchestrator.py)
                                        ↓
                                   JSON Response
                                        ↓
                                  Frontend JavaScript (index.html)
```

Every field on `PlayerMatchStats` and its nested dataclasses must either:
1. **Be serialized** and appear in orchestrator output, OR
2. **Be documented** as intentionally unserialized (with rationale)

This contract prevents the #1 recurring bug: **serialization gaps** where fields exist on the model but never reach the frontend.

---

## How to Use This Document

### When Adding a New Metric

1. **Add the field** to the appropriate dataclass in `src/opensight/analysis/models.py`
2. **Add the computation** in the appropriate `compute_*.py` module (or `analytics.py`)
3. **Check this contract** to find WHERE in `orchestrator.py` to add serialization
4. **Add serialization** to the orchestrator output dict using the key format documented here
5. **Add frontend code** to read the key in `src/opensight/static/index.html`
6. **Update this contract** with a new entry for the field
7. **Run tests:** `pytest tests/test_e2e_pipeline.py -v` (should pass with new field)

### When Debugging a Missing Field

1. **Search this contract** for the field name
2. **Check the Status column:**
   - ✅ WIRED → Field should work, check frontend code for bugs
   - ⚠️ WIRED (bug) → Known issue documented, check comment
   - ❌ NOT SERIALIZED → Field exists but orchestrator doesn't output it
   - 🔲 NO FRONTEND → Orchestrator outputs it but frontend doesn't read it
   - 🚫 INTENTIONALLY SKIPPED → Not meant for frontend (internal-only field)

---

## Contract Tables

### Legend

| Status | Meaning |
|--------|---------|
| ✅ WIRED | Field flows correctly: model → orchestrator → frontend |
| ⚠️ WIRED (bug) | Field is serialized but has a known issue (see comment) |
| ❌ NOT SERIALIZED | Field exists on model but NOT in orchestrator output |
| 🔲 NO FRONTEND | Orchestrator outputs it but frontend doesn't consume it |
| 🚫 INTENTIONALLY SKIPPED | Internal field, not meant for frontend (rationale required) |

---

## 1. Core Player Stats (PlayerMatchStats - Top Level)

### Identity & Basic Stats

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `steam_id` | `int` | `"steam_id"` (top-level) | orch:100 | `p.steam_id` | ✅ WIRED |
| `name` | `str` | `"name"` (top-level) | orch:101 | `p.name` | ✅ WIRED |
| `team` | `str` | `"team"` (top-level) | orch:102 | `p.team` | ✅ WIRED |
| `kills` | `int` | `"stats.kills"` | orch:108 | `p.stats.kills` | ✅ WIRED |
| `deaths` | `int` | `"stats.deaths"` | orch:109 | `p.stats.deaths` | ✅ WIRED |
| `assists` | `int` | `"stats.assists"` | orch:110 | `p.stats.assists` | ✅ WIRED |
| `headshots` | `int` | N/A | - | - | 🚫 SKIP: Use `headshot_hits` instead |
| `total_damage` | `int` | `"stats.total_damage"` + top-level | orch:105, 123 | `p.total_damage` | ✅ WIRED |
| `rounds_played` | `int` | `"stats.rounds_played"` + top-level | orch:104, 111 | `p.rounds_played` | ✅ WIRED |

**Notes:**
- `headshots` is a duplicate of `headshot_hits` from accuracy tracking. Frontend uses `p.stats.headshot_pct` instead.

### Weapon Stats

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `weapon_kills` | `dict` | `"weapon_kills"` (top-level) | orch:106 | `p.weapon_kills` | ✅ WIRED |

### KAST & Survival

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `kast_rounds` | `int` | N/A | - | - | 🚫 SKIP: Raw array, only `kast_percentage` serialized |
| `rounds_survived` | `int` | N/A | - | - | 🚫 SKIP: Raw array, only `survival_rate` computed |

### Economy Stats (Integrated from economy module)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `avg_equipment_value` | `float` | `"economy.avg_equipment_value"` | orch:281 | `p.economy.avg_equipment_value` | ✅ WIRED |
| `eco_rounds` | `int` | `"economy.eco_rounds"` | orch:282 | `p.economy.eco_rounds` | ✅ WIRED |
| `force_rounds` | `int` | `"economy.force_rounds"` | orch:283 | `p.economy.force_rounds` | ✅ WIRED |
| `full_buy_rounds` | `int` | `"economy.full_buy_rounds"` | orch:284 | `p.economy.full_buy_rounds` | ✅ WIRED |
| `damage_per_dollar` | `float` | `"economy.damage_per_dollar"` | orch:285 | `p.economy.damage_per_dollar` | ✅ WIRED |
| `kills_per_dollar` | `float` | `"economy.kills_per_dollar"` | orch:286 | `p.economy.kills_per_dollar` | ✅ WIRED |

### Discipline Stats

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `greedy_repeeks` | `int` | `"discipline.greedy_repeeks"` | orch:290 | `p.discipline.greedy_repeeks` | ✅ WIRED |
| `discipline_rating` | `float` | `"discipline.discipline_rating"` | orch:289 | `p.discipline.discipline_rating` | ✅ WIRED |

### Combat Stats

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `trade_kill_time_avg_ms` | `float` | N/A | - | - | 🚫 SKIP: Internal timing metric, not user-facing |
| `untraded_deaths` | `int` | `"duels.untraded_deaths"` | orch:235 | `p.duels.untraded_deaths` | ✅ WIRED |

### RWS (Round Win Shares)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `rws` | `float` | `"rws.avg_rws"` | orch:272 | `p.rws.avg_rws` | ✅ WIRED |
| `damage_in_won_rounds` | `int` | N/A | - | - | 🚫 SKIP: Internal tracking, `avg_rws` is user-facing |
| `rounds_won` | `int` | `"rws.rounds_won"` | orch:274 | `p.rws.rounds_won` | ✅ WIRED |

### State Machine Enhanced Stats

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `effective_flashes` | `int` | `"utility.effective_flashes"` | orch:200 | `p.utility.effective_flashes` | ✅ WIRED |
| `ineffective_flashes` | `int` | N/A | - | - | 🚫 SKIP: Computed in utility analysis, not top-level |
| `utility_adr` | `float` | N/A | - | - | 🚫 SKIP: Advanced metric, computed in utility module |

---

## 2. Aim & Timing Stats (PlayerMatchStats)

### Engagement Duration (Time from first damage to kill - measures spray control)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `engagement_duration_values` | `list` | N/A | - | - | 🚫 SKIP: Raw list, medians/percentiles serialized instead |
| `ttd_median_ms` | `@property` | `"advanced.ttd_median_ms"` | orch:145-146 | `p.advanced.ttd_median_ms` | ⚠️ WIRED (Wave B null bug) |
| `ttd_mean_ms` | `@property` | `"advanced.ttd_mean_ms"` | orch:148 | `p.advanced.ttd_mean_ms` | ⚠️ WIRED (Wave B null bug) |
| `ttd_95th_ms` | `@property` | `"advanced.ttd_95th_ms"` | orch:149-152 | - | 🔲 NO FRONTEND |

**Note:** TTD (Time to Damage) currently has 0/10 players with data in golden demo. Root cause: Wave B pipeline gap. Frontend shows "N/A" instead of values.

### True TTD (Reaction time: visibility to first damage)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `true_ttd_values` | `list` | N/A | - | - | 🚫 SKIP: Raw list, not serialized yet |
| `prefire_count` | `int` | `"advanced.prefire_kills"` | orch:164 | `p.advanced.prefire_kills` | ✅ WIRED |

**Note:** Model field is `prefire_count` but orchestrator renames to `prefire_kills` for clarity.

### Crosshair Placement

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `cp_values` | `list` | N/A | - | - | 🚫 SKIP: Raw list, medians serialized instead |
| `cp_median_error_deg` | `@property` | `"advanced.cp_median_error_deg"` | orch:155-158 | `p.advanced.cp_median_error_deg` | ✅ WIRED |
| `cp_mean_error_deg` | `@property` | `"advanced.cp_mean_error_deg"` | orch:160-161 | `p.advanced.cp_mean_error_deg` | ✅ WIRED |

### Accuracy Stats (Leetify style)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `shots_fired` | `int` | `"aim_stats.shots_fired"` | orch:208 | `p.aim_stats.shots_fired` | ✅ WIRED |
| `shots_hit` | `int` | `"aim_stats.shots_hit"` | orch:209 | `p.aim_stats.shots_hit` | ✅ WIRED |
| `accuracy` | `@property` | `"aim_stats.accuracy_all"` | orch:210 | `p.aim_stats.accuracy_all` | ✅ WIRED |
| `headshot_hits` | `int` | `"aim_stats.headshot_hits"` | orch:211 | `p.aim_stats.headshot_hits` | ✅ WIRED |
| `head_hit_rate` | `@property` | `"aim_stats.head_accuracy"` | orch:212 | `p.aim_stats.head_accuracy` | ✅ WIRED |

### Spray Accuracy (Hits after 4th bullet in burst)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `spray_shots_fired` | `int` | `"aim_stats.spray_shots_fired"` | orch:214 | `p.aim_stats.spray_shots_fired` | ✅ WIRED |
| `spray_shots_hit` | `int` | `"aim_stats.spray_shots_hit"` | orch:215 | `p.aim_stats.spray_shots_hit` | ✅ WIRED |
| `spray_accuracy` | `@property` | `"aim_stats.spray_accuracy"` | orch:216 | `p.aim_stats.spray_accuracy` | ✅ WIRED |

### Counter-Strafing (Leetify parity)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `shots_stationary` | `int` | `"aim_stats.shots_stationary"` | orch:218 | `p.aim_stats.shots_stationary` | ✅ WIRED |
| `shots_with_velocity` | `int` | `"aim_stats.shots_with_velocity"` | orch:219 | `p.aim_stats.shots_with_velocity` | ✅ WIRED |
| `counter_strafe_pct` | `@property` | `"aim_stats.counter_strafe_pct"` | orch:220 | `p.aim_stats.counter_strafe_pct` | ✅ WIRED |
| `counter_strafe_kills` | `int` | N/A | - | - | 🚫 SKIP: DEPRECATED, replaced by shot-based tracking |
| `total_kills_with_velocity` | `int` | N/A | - | - | 🚫 SKIP: DEPRECATED, internal tracking for legacy metric |

---

## 3. Nested Dataclasses

### 3.1 OpeningDuelStats (Opening Engagements & Entry Fragging)

**Serialized in:** `orchestrator._get_entry_stats()` → `"entry"` dict (lines 452-475)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `wins` | `int` | `"entry.entry_kills"` | orch:462 | `p.entry.entry_kills` | ✅ WIRED |
| `losses` | `int` | `"entry.entry_deaths"` | orch:463 | `p.entry.entry_deaths` | ✅ WIRED |
| `attempts` | `int` | `"entry.entry_attempts"` | orch:461 | `p.entry.entry_attempts` | ✅ WIRED |
| `win_rate` | `@property` | `"entry.entry_success_pct"` | orch:466 | `p.entry.entry_success_pct` | ✅ WIRED |
| `entry_diff` | (computed) | `"entry.entry_diff"` | orch:464 | `p.entry.entry_diff` | ✅ WIRED |
| `entry_ttd_values` | `list` | N/A | - | - | 🚫 SKIP: Raw list, median computed on frontend if needed |
| `t_side_entries` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `ct_side_entries` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `map_control_kills` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `site_kills` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `kill_zones` | `dict` | N/A | - | - | ❌ NOT SERIALIZED |
| `supported_entries` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `unsupported_entries` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `supported_deaths` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `unsupported_deaths` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `dry_peek_rate` | `@property` | N/A | - | - | ❌ NOT SERIALIZED |
| `dry_peek_death_rate` | `@property` | N/A | - | - | ❌ NOT SERIALIZED |
| `map_control_rate` | `@property` | N/A | - | - | ❌ NOT SERIALIZED |
| `entry_ttd_median_ms` | `@property` | N/A | - | - | ❌ NOT SERIALIZED |
| `entry_ttd_mean_ms` | `@property` | N/A | - | - | ❌ NOT SERIALIZED |

**Also serialized in:** `"duels.opening_kills"` and `"duels.opening_deaths"` (orch:238-239) - duplicate for frontend convenience

**Also serialized in:** `"advanced.opening_kills"` and `"advanced.opening_deaths"` (orch:165-166) - duplicate for frontend convenience

**Issue:** Many zone-based and dry-peek fields are computed but not serialized. These are valuable coaching insights that should be exposed.

---

### 3.2 TradeStats (Leetify-style Trade Analysis)

**Serialized in:** `orchestrator._get_trade_stats()` → `"trades"` dict (lines 477-527)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `trade_kill_opportunities` | `int` | `"trades.trade_kill_opportunities"` | orch:492 | `p.trades.trade_kill_opportunities` | ✅ WIRED |
| `trade_kill_attempts` | `int` | `"trades.trade_kill_attempts"` | orch:493 | `p.trades.trade_kill_attempts` | ✅ WIRED |
| `trade_kill_success` | `int` | `"trades.trade_kill_success"` | orch:495 | `p.trades.trade_kill_success` | ✅ WIRED |
| `trade_kill_attempts_pct` | `@property` | `"trades.trade_kill_attempts_pct"` | orch:494 | `p.trades.trade_kill_attempts_pct` | ✅ WIRED |
| `trade_kill_success_pct` | `@property` | `"trades.trade_kill_success_pct"` | orch:496 | `p.trades.trade_kill_success_pct` | ✅ WIRED |
| `traded_death_opportunities` | `int` | `"trades.traded_death_opportunities"` | orch:498 | `p.trades.traded_death_opportunities` | ✅ WIRED |
| `traded_death_attempts` | `int` | `"trades.traded_death_attempts"` | orch:499 | `p.trades.traded_death_attempts` | ✅ WIRED |
| `traded_death_success` | `int` | `"trades.traded_death_success"` | orch:501 | `p.trades.traded_death_success` | ✅ WIRED |
| `traded_death_attempts_pct` | `@property` | `"trades.traded_death_attempts_pct"` | orch:500 | `p.trades.traded_death_attempts_pct` | ✅ WIRED |
| `traded_death_success_pct` | `@property` | `"trades.traded_death_success_pct"` | orch:502 | `p.trades.traded_death_success_pct` | ✅ WIRED |
| `kills_traded` | `int` | `"trades.trade_kills"` (alias) | orch:506 | `p.trades.trade_kills` | ✅ WIRED |
| `deaths_traded` | `int` | `"trades.deaths_traded"` | orch:507 | `p.trades.deaths_traded` | ✅ WIRED |
| `trade_attempts` | `int` | N/A | - | - | 🚫 SKIP: Covered by `trade_kill_attempts` |
| `failed_trades` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `traded_entry_kills` | `int` | `"trades.traded_entry_kills"` | orch:508 | `p.trades.traded_entry_kills` | ✅ WIRED |
| `traded_entry_deaths` | `int` | `"trades.traded_entry_deaths"` | orch:509 | `p.trades.traded_entry_deaths` | ✅ WIRED |
| `time_to_trade_ticks` | `list` | N/A | - | - | 🚫 SKIP: Raw tick list, not user-facing |

**Also serialized in:** `"duels.trade_kills"` and `"duels.traded_deaths"` (orch:230-231) - duplicate for frontend convenience

---

### 3.3 ClutchStats (Clutch Performance)

**Serialized in:** `orchestrator._get_clutch_stats()` → `"clutches"` dict (lines 529-557)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `total_situations` | `int` | `"clutches.total_situations"` | orch:539 | `p.clutches.total_situations` | ✅ WIRED |
| `total_wins` | `int` | `"clutches.clutch_wins"` | orch:536 | `p.clutches.clutch_wins` | ✅ WIRED |
| `v1_attempts` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `v1_wins` | `int` | `"clutches.v1_wins"` | orch:540 | `p.clutches.v1_wins` | ✅ WIRED |
| `v2_attempts` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `v2_wins` | `int` | `"clutches.v2_wins"` | orch:541 | `p.clutches.v2_wins` | ✅ WIRED |
| `v3_attempts` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `v3_wins` | `int` | `"clutches.v3_wins"` | orch:542 | `p.clutches.v3_wins` | ✅ WIRED |
| `v4_attempts` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `v4_wins` | `int` | `"clutches.v4_wins"` | orch:543 | `p.clutches.v4_wins` | ✅ WIRED |
| `v5_attempts` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `v5_wins` | `int` | `"clutches.v5_wins"` | orch:544 | `p.clutches.v5_wins` | ✅ WIRED |
| `clutches` | `list` | N/A | - | - | 🚫 SKIP: Raw event list, not user-facing |
| `clutch_success_pct` | (computed) | `"clutches.clutch_success_pct"` | orch:538 | `p.clutches.clutch_success_pct` | ✅ WIRED |
| `clutch_losses` | (computed) | `"clutches.clutch_losses"` | orch:537 | `p.clutches.clutch_losses` | ✅ WIRED |

**Also serialized in:** `"duels.clutch_wins"` and `"duels.clutch_attempts"` (orch:236-237) - duplicate for frontend convenience

**Issue:** Clutch attempts per situation level (v1_attempts, v2_attempts, etc.) are not serialized. Frontend can only see wins, not success rates per level.

---

### 3.4 MultiKillStats (Multi-Kill Rounds)

**Serialized in:** `orchestrator.py` → `"stats"` dict (lines 124-127)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `rounds_with_1k` | `int` | N/A | - | - | 🚫 SKIP: Every round with a kill counts, not interesting |
| `rounds_with_2k` | `int` | `"stats.2k"` | orch:124 | `p.stats["2k"]` | ✅ WIRED |
| `rounds_with_3k` | `int` | `"stats.3k"` | orch:125 | `p.stats["3k"]` | ✅ WIRED |
| `rounds_with_4k` | `int` | `"stats.4k"` | orch:126 | `p.stats["4k"]` | ✅ WIRED |
| `rounds_with_5k` | `int` | `"stats.5k"` | orch:127 | `p.stats["5k"]` | ✅ WIRED |
| `total_multi_kill_rounds` | `@property` | N/A | - | - | 🔲 Computed client-side from 2k+3k+4k+5k |

**Note:** Multi-kill data comes from `multikills` dict passed to orchestrator, computed separately in analytics.

---

### 3.5 UtilityStats (Utility Usage & Effectiveness)

**Serialized in:** `orchestrator.py` → `"utility"` dict (lines 168-205)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `flashbangs_thrown` | `int` | `"utility.flashbangs_thrown"` | orch:170 | `p.utility.flashbangs_thrown` | ✅ WIRED |
| `smokes_thrown` | `int` | `"utility.smokes_thrown"` | orch:171 | `p.utility.smokes_thrown` | ✅ WIRED |
| `he_thrown` | `int` | `"utility.he_thrown"` | orch:172 | `p.utility.he_thrown` | ✅ WIRED |
| `molotovs_thrown` | `int` | `"utility.molotovs_thrown"` | orch:173 | `p.utility.molotovs_thrown` | ✅ WIRED |
| `enemies_flashed` | `int` | `"utility.enemies_flashed"` | orch:175 | `p.utility.enemies_flashed` | ✅ WIRED |
| `teammates_flashed` | `int` | `"utility.teammates_flashed"` | orch:176 | `p.utility.teammates_flashed` | ✅ WIRED |
| `flash_assists` | `int` | `"utility.flash_assists"` | orch:174 | `p.utility.flash_assists` | ✅ WIRED |
| `total_blind_time` | `float` | `"utility.total_blind_time"` | orch:201 | `p.utility.total_blind_time` | ✅ WIRED |
| `effective_flashes` | `int` | `"utility.effective_flashes"` | orch:200 | `p.utility.effective_flashes` | ✅ WIRED |
| `times_blinded` | `int` | `"utility.times_blinded"` | orch:202 | `p.utility.times_blinded` | ✅ WIRED |
| `total_time_blinded` | `float` | `"utility.total_time_blinded"` | orch:203 | `p.utility.total_time_blinded` | ✅ WIRED |
| `he_damage` | `int` | `"utility.he_damage"` | orch:177 | `p.utility.he_damage` | ✅ WIRED |
| `he_team_damage` | `int` | `"utility.he_team_damage"` | orch:192 | `p.utility.he_team_damage` | ✅ WIRED |
| `molotov_damage` | `int` | `"utility.molotov_damage"` | orch:178 | `p.utility.molotov_damage` | ✅ WIRED |
| `molotov_team_damage` | `int` | N/A | - | - | ❌ NOT SERIALIZED |
| `unused_utility_value` | `int` | `"utility.unused_utility_value"` | orch:193 | `p.utility.unused_utility_value` | ✅ WIRED |
| `enemies_flashed_per_round` | `@property` | `"utility.enemies_flashed_per_round"` | orch:180-181 | `p.utility.enemies_flashed_per_round` | ✅ WIRED |
| `friends_flashed_per_round` | `@property` | `"utility.friends_flashed_per_round"` | orch:183-184 | `p.utility.friends_flashed_per_round` | ✅ WIRED |
| `avg_blind_time` | `@property` | `"utility.avg_blind_time"` | orch:186 | `p.utility.avg_blind_time` | ✅ WIRED |
| `avg_he_damage` | `@property` | `"utility.avg_he_damage"` | orch:187 | `p.utility.avg_he_damage` | ✅ WIRED |
| `flash_effectiveness_pct` | `@property` | `"utility.flash_effectiveness_pct"` | orch:188-189 | `p.utility.flash_effectiveness_pct` | ✅ WIRED |
| `flash_assist_pct` | `@property` | `"utility.flash_assist_pct"` | orch:191 | `p.utility.flash_assist_pct` | ✅ WIRED |
| `avg_time_blinded` | `@property` | `"utility.avg_time_blinded"` | orch:204 | `p.utility.avg_time_blinded` | ✅ WIRED |
| `utility_quality_rating` | `@property` | `"utility.utility_quality_rating"` | orch:194-195 | `p.utility.utility_quality_rating` | ✅ WIRED |
| `utility_quantity_rating` | `@property` | `"utility.utility_quantity_rating"` | orch:197-198 | `p.utility.utility_quantity_rating` | ✅ WIRED |
| `_rounds_played` | `int` | N/A | - | - | 🚫 SKIP: Internal field for per-round calculations |

---

### 3.6 SprayTransferStats (Spray Transfer Kills)

**Serialized in:** `orchestrator.py` → `"spray_transfers"` dict (lines 244-265)

| Model Field | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|-------------|------|-----------------|-----------|-----------------|--------|
| `double_sprays` | `int` | `"spray_transfers.double_sprays"` | orch:245 | `p.spray_transfers.double_sprays` | ✅ WIRED |
| `triple_sprays` | `int` | `"spray_transfers.triple_sprays"` | orch:246 | `p.spray_transfers.triple_sprays` | ✅ WIRED |
| `quad_sprays` | `int` | `"spray_transfers.quad_sprays"` | orch:247 | `p.spray_transfers.quad_sprays` | ✅ WIRED |
| `ace_sprays` | `int` | `"spray_transfers.ace_sprays"` | orch:248 | `p.spray_transfers.ace_sprays` | ✅ WIRED |
| `total_spray_kills` | `int` | `"spray_transfers.total_spray_kills"` | orch:250-251 | `p.spray_transfers.total_spray_kills` | ✅ WIRED |
| `_spray_times_ms` | `list` | N/A | - | - | 🚫 SKIP: Raw list, avg computed |
| `total_sprays` | `@property` | `"spray_transfers.total_sprays"` | orch:249 | `p.spray_transfers.total_sprays` | ✅ WIRED |
| `avg_spray_time_ms` | `@property` | `"spray_transfers.avg_spray_time_ms"` | orch:253-254 | `p.spray_transfers.avg_spray_time_ms` | ✅ WIRED |
| `avg_kills_per_spray` | (computed) | `"spray_transfers.avg_kills_per_spray"` | orch:256-263 | `p.spray_transfers.avg_kills_per_spray` | ✅ WIRED |

---

### 3.7 SideStats (CT/T Side Performance)

**Status:** ❌ **NOT SERIALIZED** - Entire dataclass missing from orchestrator output

| Model Field | Type | Orchestrator Key | Status |
|-------------|------|-----------------|--------|
| `ct_stats` | `SideStats` | N/A | ❌ NOT SERIALIZED |
| `t_stats` | `SideStats` | N/A | ❌ NOT SERIALIZED |

**Issue:** Side-specific stats (CT vs T performance breakdown) are computed but never serialized. This is a significant gap for coaching insights.

---

### 3.8 MistakesStats (Scope.gg Style Error Tracking)

**Status:** ❌ **NOT SERIALIZED** - Entire dataclass missing from orchestrator output

| Model Field | Type | Orchestrator Key | Status |
|-------------|------|-----------------|--------|
| `mistakes` | `MistakesStats` | N/A | ❌ NOT SERIALIZED |

**Issue:** Mistake tracking (grenades through walls, friendly fire, etc.) is computed but not exposed to frontend.

---

### 3.9 LurkStats (Lurk Behavior Tracking)

**Status:** ❌ **NOT SERIALIZED** - Entire dataclass missing from orchestrator output

| Model Field | Type | Orchestrator Key | Status |
|-------------|------|-----------------|--------|
| `lurk` | `LurkStats` | N/A | ❌ NOT SERIALIZED |

**Issue:** Lurk detection and statistics are computed but not serialized.

---

### 3.10 OpeningEngagementStats (Pre-Kill Damage Tracking)

**Status:** ❌ **NOT SERIALIZED** - Entire dataclass missing from orchestrator output

| Model Field | Type | Orchestrator Key | Status |
|-------------|------|-----------------|--------|
| `opening_engagements` | `OpeningEngagementStats` | N/A | ❌ NOT SERIALIZED |

**Issue:** Opening engagement stats (damage before first kill) are computed but not serialized.

---

### 3.11 EntryFragStats (Legacy Entry Tracking)

**Status:** ❌ **NOT SERIALIZED** - Entire dataclass missing from orchestrator output

| Model Field | Type | Orchestrator Key | Status |
|-------------|------|-----------------|--------|
| `entry_frags` | `EntryFragStats` | N/A | ❌ NOT SERIALIZED |

**Note:** This is likely redundant with `opening_duels`. Consider deprecating.

---

## 4. Computed Properties (PlayerMatchStats)

These are `@property` methods on PlayerMatchStats that compute values on-demand:

| Property | Type | Orchestrator Key | Orch Line | Frontend Access | Status |
|----------|------|-----------------|-----------|-----------------|--------|
| `kd_ratio` | `float` | `"stats.kd_ratio"` | orch:122 | `p.stats.kd_ratio` | ✅ WIRED |
| `kd_diff` | `int` | N/A | - | - | 🔲 Computed client-side from kills-deaths |
| `adr` | `float` | `"stats.adr"` | orch:114 | `p.stats.adr` | ✅ WIRED |
| `headshot_percentage` | `float` | `"stats.headshot_pct"` | orch:117-118 | `p.stats.headshot_pct` | ✅ WIRED |
| `kast_percentage` | `float` | `"rating.kast_percentage"` | orch:131-132 | `p.rating.kast_percentage` | ✅ WIRED |
| `survival_rate` | `float` | N/A | - | - | 🔲 Computed client-side from rounds_survived |
| `kills_per_round` | `float` | N/A | - | - | 🔲 Computed client-side from kills/rounds |
| `deaths_per_round` | `float` | N/A | - | - | 🔲 Computed client-side from deaths/rounds |
| `assists_per_round` | `float` | N/A | - | - | 🔲 Computed client-side from assists/rounds |
| `multi_kill_round_rate` | `float` | N/A | - | - | 🔲 Computed client-side from multi-kills |
| `impact_rating` | `float` | `"rating.impact_rating"` | orch:139-140 | `p.rating.impact_rating` | ✅ WIRED |
| `hltv_rating` | `float` | `"rating.hltv_rating"` | orch:130 | `p.rating.hltv_rating` | ✅ WIRED |
| `impact_plus_minus` | `float` | N/A | - | - | ❌ NOT SERIALIZED |
| `aim_rating` | `float` | `"rating.aim_rating"` | orch:135 | `p.rating.aim_rating` | ✅ WIRED |
| `utility_rating` | `float` | `"rating.utility_rating"` | orch:136-137 | `p.rating.utility_rating` | ✅ WIRED |

---

## 5. Summary Statistics

### Total Field Count

| Category | Count |
|----------|-------|
| **PlayerMatchStats** direct fields | 53 |
| **Nested dataclasses** | 11 |
| **Total model fields** (including nested) | ~150+ |
| **Serialized fields** | ~100 |
| **Intentionally skipped** (documented) | ~40 |
| **Not serialized** (gaps) | ~10-15 |

### Status Breakdown

| Status | Count | % of Total |
|--------|-------|-----------|
| ✅ **WIRED** (working correctly) | ~90 | 60% |
| ⚠️ **WIRED** (with bugs) | 2 | 1% |
| ❌ **NOT SERIALIZED** (gaps) | ~15 | 10% |
| 🔲 **NO FRONTEND** (serialized but unused) | ~5 | 3% |
| 🚫 **INTENTIONALLY SKIPPED** (documented) | ~40 | 26% |

---

## 6. Known Issues & Gaps

### Critical Gaps (Should be serialized)

1. **SideStats** (`ct_stats`, `t_stats`) - Entire dataclass not serialized
   - Impact: Cannot show CT vs T performance breakdown
   - Severity: HIGH - valuable coaching insight missing

2. **MistakesStats** - Entire dataclass not serialized
   - Impact: Cannot show Scope.gg-style mistake tracking
   - Severity: MEDIUM - coaching insights missing

3. **LurkStats** - Entire dataclass not serialized
   - Impact: Cannot identify lurker playstyle
   - Severity: MEDIUM - persona detection incomplete

4. **OpeningDuelStats** zone/dry-peek fields - Computed but not serialized
   - Missing: `t_side_entries`, `ct_side_entries`, `map_control_kills`, `site_kills`, `dry_peek_rate`, etc.
   - Impact: Cannot show entry context analysis
   - Severity: MEDIUM - advanced entry insights missing

5. **impact_plus_minus** - Computed property not serialized
   - Impact: Leetify-style impact rating unavailable
   - Severity: LOW - alternative metrics exist

### Known Bugs

1. **TTD Null Bug (Wave B)**
   - Field: `ttd_median_ms`, `ttd_mean_ms`
   - Status: ⚠️ WIRED but returns None for all players in golden demo (0/10)
   - Root cause: Pipeline data gap in engagement duration tracking
   - Tracked in: `tests/test_e2e_pipeline.py::test_pipeline_produces_valid_output`

2. **Timeline Kill Counts**
   - Field: `timeline_graph.round_scores[].ct_kills`, `timeline_graph.round_scores[].t_kills`
   - Status: Always 0 in current implementation
   - Root cause: Timeline structure doesn't populate kill counts
   - Tracked in: `tests/test_e2e_pipeline.py::test_timeline_has_both_teams`

---

## 7. Maintenance

### Updating This Contract

When you add a new field:
1. Add entry to appropriate section
2. Document orchestrator line number
3. Verify frontend key with grep: `grep -n "field_name" src/opensight/static/index.html`
4. Run: `pytest tests/test_e2e_pipeline.py::TestSerializationCompleteness -v`
5. Update summary statistics

### Automated Verification

This contract is verified by `tests/test_e2e_pipeline.py`:
- `TestSerializationCompleteness` - Ensures all model fields are either serialized or documented as skipped
- `TestFrontendKeyAlignment` - Ensures frontend only reads keys that exist in orchestrator output
- `TestPipelineDataFlow` - Runs golden demo through pipeline and checks for null/missing data

Run tests after any serialization changes:
```bash
PYTHONPATH=src pytest tests/test_e2e_pipeline.py -v
```

---

## 8. Reference: Key Locations

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Model definitions | `src/opensight/analysis/models.py` | 1-2500 | All dataclass definitions |
| Main serialization | `src/opensight/pipeline/orchestrator.py` | 95-292 | Player dict building |
| Entry stats helper | `src/opensight/pipeline/orchestrator.py` | 452-475 | `_get_entry_stats()` |
| Trade stats helper | `src/opensight/pipeline/orchestrator.py` | 477-527 | `_get_trade_stats()` |
| Clutch stats helper | `src/opensight/pipeline/orchestrator.py` | 529-557 | `_get_clutch_stats()` |
| Frontend consumption | `src/opensight/static/index.html` | Various | JavaScript data access |
| E2E verification | `tests/test_e2e_pipeline.py` | 1-730 | Serialization contract tests |

---

**End of Contract**

*This document is the source of truth for OpenSight's data pipeline. Keep it updated as you add metrics.*
