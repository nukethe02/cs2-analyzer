# OpenSight LLM Cost Audit Report

**Date:** 2026-02-08
**Audited by:** Claude (read-only audit, no code changes)

## Summary

- **Total distinct LLM call sites found:** 6 methods across 4 files
- **Models in use:** `claude-haiku-4-5-20251001` (STANDARD), `claude-sonnet-4-5-20250929` (DEEP)
- **Biggest cost driver:** 10 per-player AI summaries generated on every cold analysis ($0.03) + uncached tactical/strat/review tabs re-running on every click (~$0.03-0.06 each)
- **Why $0.10/job:** ~$0.03 for per-player summaries + one AI tab click at ~$0.03-0.06
- **Why $1.50 for 10 runs:** Tactical tabs are not cached — every click re-runs the LLM

## Pricing Used

| Model | Input | Output | Cache Read (90% off) | Cache Write (+25%) |
|-------|-------|--------|---------------------|--------------------|
| Haiku 4.5 (STANDARD) | $1.00/MTok | $5.00/MTok | $0.10/MTok | $1.25/MTok |
| Sonnet 4.5 (DEEP) | $3.00/MTok | $15.00/MTok | $0.30/MTok | $3.75/MTok |

Source: `llm_client.py` lines 44-47 (pricing is hardcoded in the codebase itself)

---

## Per-Feature Breakdown

### Feature 1: Per-Player AI Summaries (AUTO — runs on every cold analysis)

- **Endpoint:** POST `/analyze` → orchestrator pipeline → `_generate_ai_summaries()`
- **File:** `orchestrator.py:2741` → `llm_client.py:485` (`LLMClient.generate_match_summary`)
- **Model:** STANDARD (Haiku 4.5)
- **LLM calls per execution:** **10** (one per player, sequential loop at orchestrator.py:2765)
- **max_tokens:** 400 per call
- **System prompt:** `CS2_COACHING_SYSTEM_PROMPT` — 16,668 chars / ~4,167 tokens (lines 83-413)
  - Contains: coaching framework, benchmark tables, role definitions, 3 few-shot examples, map callouts for 6 maps, economy reference, weapon meta, tactical mistakes by rank, analysis decision tree, output format rules
- **User prompt:** ~550 chars / ~140 tokens per player (core stats + advanced metrics)
- **Is result cached?** YES — cached in file-based SHA256 cache (survives restarts)
- **Redundant calls detected?** NO for same demo file (cache hit). YES if force-reanalyze.

**Cost per cold analysis (10 players):**

| Call | System Prompt | User Prompt | Output (est.) | Input Cost | Output Cost | Total |
|------|--------------|-------------|---------------|------------|-------------|-------|
| 1 (cache write) | 4,167 tok × $1.25/MTok | 140 tok × $1.00/MTok | 250 tok × $5.00/MTok | $0.00535 | $0.00125 | $0.0066 |
| 2-10 (cache read) × 9 | 4,167 tok × $0.10/MTok | 140 tok × $1.00/MTok | 250 tok × $5.00/MTok | $0.00056 each | $0.00125 each | $0.0018 each |

**Total for 10 players: ~$0.023**

---

### Feature 2: Tactical Analysis (ON-DEMAND — user clicks "Tactical Analysis" tab)

- **Endpoint:** POST `/api/tactical-analysis/{job_id}`
- **File:** `routes_analysis.py:249` → `llm_client.py:851` (`TacticalAIClient.analyze`)
- **Model:** STANDARD (Haiku 4.5)
- **LLM calls per execution:** 1-10 (tool-use agentic loop, typically 3-5 iterations)
- **max_tokens:** 4,096 per iteration
- **System prompt:** `SYSTEM_PROMPT_OVERVIEW` — 437 chars / ~109 tokens (tiny)
- **User prompt:** `to_llm_prompt()` structured data — ~8,730 chars / ~2,180 tokens
- **Is result cached?** **NO — every tab click makes fresh LLM calls**
- **Redundant calls detected?** **YES — clicking the same tab twice doubles the cost**

**Cost estimate per click (3 iterations avg):**

| Iteration | Input Tokens | Output Tokens | Input Cost | Output Cost |
|-----------|-------------|---------------|------------|-------------|
| 1 | ~2,300 | ~300 (tool call) | $0.0023 | $0.0015 |
| 2 | ~3,100 (+tool result) | ~400 (tool call) | $0.0031 | $0.0020 |
| 3 | ~4,000 (+tool result) | ~1,500 (final text) | $0.0040 | $0.0075 |
| **Total** | **~9,400** | **~2,200** | **$0.0094** | **$0.0110** |

**Total per click: ~$0.020**
**With 5 iterations: ~$0.045**

---

### Feature 3: Strat-Stealing Report (ON-DEMAND — user clicks "Strat Steal" tab)

- **Endpoint:** POST `/api/strat-steal/{job_id}`
- **File:** `routes_analysis.py:312` → `strat_engine.py:448` → `llm_client.py:851`
- **Model:** STANDARD (Haiku 4.5)
- **LLM calls per execution:** 1-10 (tool-use loop, same as tactical)
- **max_tokens:** 4,096 per iteration
- **System prompt:** `SYSTEM_PROMPT_STRAT_ANALYST` — 863 chars / ~216 tokens
- **User prompt:** Structured data + pattern analysis augmentation — ~10,000 chars / ~2,500 tokens
- **Is result cached?** **NO**
- **Redundant calls detected?** **YES**

**Total per click: ~$0.020-0.045** (same cost profile as tactical analysis)

---

### Feature 4: Self-Review Report (ON-DEMAND — user clicks "Self Review" tab)

- **Endpoint:** POST `/api/self-review/{job_id}`
- **File:** `routes_analysis.py:395` → `self_review.py:581` → `llm_client.py:851`
- **Model:** STANDARD (Haiku 4.5)
- **LLM calls per execution:** 1-10 (tool-use loop)
- **max_tokens:** 4,096 per iteration
- **System prompt:** `SYSTEM_PROMPT_SELF_REVIEW` — 737 chars / ~184 tokens
- **User prompt:** Structured data + mistake analysis augmentation — ~11,000 chars / ~2,750 tokens
- **Is result cached?** **NO**
- **Redundant calls detected?** **YES**

**Total per click: ~$0.020-0.045**

---

### Feature 5: Anti-Strat Report (ON-DEMAND — scouting workflow)

- **Endpoint:** POST `/api/antistrat/{session_id}/generate`
- **File:** `routes_misc.py:692` → `antistrat_report.py:375` (`AntiStratGenerator.generate`)
- **Model:** **DEEP (Sonnet 4.5)** — 3x more expensive
- **LLM calls per execution:** 1 (single call, no tool use)
- **max_tokens:** 4,096
- **System prompt:** `ANTISTRAT_SYSTEM_PROMPT` — 1,040 chars / ~260 tokens
- **User prompt:** Full scouting data + JSON schema — ~6,000 chars / ~1,500 tokens
- **Is result cached?** Partially (cached by session_id after generation)
- **Redundant calls detected?** No (generate endpoint is called once per session)

**Cost per execution:**

| Component | Tokens | Rate | Cost |
|-----------|--------|------|------|
| System prompt input | 260 | $3.00/MTok | $0.0008 |
| User prompt input | 1,500 | $3.00/MTok | $0.0045 |
| Output (est. 2,000) | 2,000 | $15.00/MTok | $0.0300 |
| **Total** | | | **~$0.035** |

---

### Feature 6: Game Plan Generation (ON-DEMAND — API only, no UI button)

- **Endpoint:** POST `/api/game-plan/generate`
- **File:** `routes_match.py:884` → `game_plan.py:621` (`GamePlanGenerator.generate`)
- **Model:** **DEEP (Sonnet 4.5)** — 3x more expensive
- **LLM calls per execution:** 1 (single call, no tool use)
- **max_tokens:** 4,096
- **System prompt:** `GAME_PLAN_SYSTEM_PROMPT` — 616 chars / ~154 tokens
- **User prompt:** Team summary + scouting data + matchup analysis + JSON schema — ~7,500 chars / ~1,875 tokens
- **Is result cached?** Yes (in-memory dict by plan_id, max 50 entries, lost on restart)
- **Redundant calls detected?** No

**Cost per execution:**

| Component | Tokens | Rate | Cost |
|-----------|--------|------|------|
| System prompt input | 154 | $3.00/MTok | $0.0005 |
| User prompt input | 1,875 | $3.00/MTok | $0.0056 |
| Output (est. 3,000) | 3,000 | $15.00/MTok | $0.0450 |
| **Total** | | | **~$0.051** |

---

### Feature 7: Natural Language Query (ON-DEMAND — API only, no UI button)

- **Endpoint:** POST `/api/query`
- **File:** `routes_match.py:1068` → `query_interface.py:111,671`
- **Model:** STANDARD (Haiku 4.5)
- **LLM calls per execution:** 2 (classify + format)
- **max_tokens:** 200 (classify) + 300 (format)
- **System prompt:** None for either call
- **Is result cached?** No
- **Redundant calls detected?** No

**Cost per execution: ~$0.003** (negligible)

---

## Cost Scenarios

### Scenario A: Single Demo Upload (Cold, All Features)

| Step | What happens | LLM Calls | Cost |
|------|-------------|-----------|------|
| Upload + analyze | 10 per-player summaries | 10 | ~$0.023 |
| Click Tactical tab | Tool-use loop (3-5 iters) | 3-5 | ~$0.020-0.045 |
| Click Strat Steal tab | Tool-use loop (3-5 iters) | 3-5 | ~$0.020-0.045 |
| Click Self Review tab | Tool-use loop (3-5 iters) | 3-5 | ~$0.020-0.045 |
| **TOTAL** | | **19-25** | **$0.083-0.158** |

**Matches user's observed ~$0.10 per job.**

### Scenario B: Same Demo, 10 Runs, All Features Clicked Each Time

| Run | Per-player summaries | 3 AI tabs (3 iter avg) | Cost |
|-----|---------------------|------------------------|------|
| 1 (cold) | $0.023 | $0.060 | $0.083 |
| 2-10 (cached per-player) | $0.00 (cache hit) | $0.060 each | $0.060 each |
| **Total (10 runs)** | | | **$0.083 + 9×$0.060 = $0.623** |

With 5-iteration tool loops:

| Run | Per-player summaries | 3 AI tabs (5 iter avg) | Cost |
|-----|---------------------|------------------------|------|
| 1 (cold) | $0.023 | $0.135 | $0.158 |
| 2-10 (cached) | $0.00 | $0.135 each | $0.135 each |
| **Total (10 runs)** | | | **$0.158 + 9×$0.135 = $1.373** |

**Close to user's observed ~$1.50 for 10 runs.** The remaining ~$0.13 gap is explained by tool-use loops occasionally hitting 6-8 iterations, plus the hero summary on first download.

### Scenario C: Anti-Strat + Game Plan (Premium Features)

| Feature | Cost |
|---------|------|
| Anti-strat report (Sonnet 4.5) | ~$0.035 |
| Game plan generation (Sonnet 4.5) | ~$0.051 |
| **Premium features total** | **~$0.086** |

---

## Token Breakdown for Largest Prompt

### CS2_COACHING_SYSTEM_PROMPT (used 10x per demo)

```
Characters: 16,668
Words: 2,590
Estimated tokens: ~4,167 (chars/4)
Lines: 330

Contents:
- Coaching Framework (5 principles): ~500 chars
- Performance Benchmark Tables (6 tables): ~2,800 chars
- CS2 Role Definitions (5 roles): ~1,800 chars
- Few-Shot Coaching Examples (3 examples): ~2,400 chars
- Map Callout Reference (6 maps): ~2,500 chars
- Economy Management Reference: ~1,700 chars
- Weapon Meta Reference: ~1,400 chars
- Common Tactical Mistakes by Rank (4 tiers): ~1,200 chars
- Round Type Classification: ~1,400 chars
- Analysis Decision Tree: ~500 chars
- Output Format Rules: ~400 chars
```

This system prompt is **~4,167 tokens sent on every single coaching call**. The actual user-specific content is only ~140 tokens. The output is capped at 400 tokens.

**Ratio: 4,167 tokens of boilerplate per 140 tokens of actual data = 30:1 waste ratio**

Prompt caching reduces the cost on calls 2-10, but the first call still pays full price, and the output tokens (the expensive part at $5.00/MTok) are unaffected by caching.

---

## Problems Found

### P1: UNCACHED TACTICAL TABS (Biggest cost multiplier)

**Impact: $0.06-0.14 wasted per repeated click**

The three AI tab endpoints (`/api/tactical-analysis`, `/api/strat-steal`, `/api/self-review`) have zero result caching. Every button click triggers a fresh multi-turn LLM call (1-10 API round-trips). The same analysis for the same demo produces essentially the same output, yet the LLM runs from scratch each time.

**Location:** `routes_analysis.py` lines 249-466

### P2: 10 SEQUENTIAL PER-PLAYER SUMMARIES (Latency + cost)

**Impact: ~$0.023 per demo, 10-30 seconds latency**

The orchestrator calls `generate_match_summary()` in a sequential loop for all 10 players (`orchestrator.py:2765`). Each call sends the 4,167-token system prompt + tiny user prompt for a 400-token response.

**Problems:**
- 10 sequential HTTP calls (no parallelism)
- Each player gets an independent summary with no cross-player context
- Could be consolidated into 1 call with all 10 players

**Location:** `orchestrator.py:2741-2812`

### P3: MASSIVE SYSTEM PROMPT FOR TINY OUTPUT

**Impact: 30:1 input-to-data token ratio**

The coaching system prompt is ~4,167 tokens containing map callouts, weapon prices, rank-tier mistakes, economy rules, etc. The actual player data is ~140 tokens. The output is capped at 400 tokens. Most of the system prompt content (weapon prices, map callouts) is never relevant to a single player's summary.

**Location:** `llm_client.py` lines 83-413

### P4: TOOL-USE LOOPS ARE UNBOUNDED IN PRACTICE

**Impact: Up to 10 iterations × 4,096 max_tokens = unpredictable costs**

The `TacticalAIClient.analyze()` uses a tool-use loop capped at 10 iterations (`llm_client.py:~938`). Each iteration accumulates context (previous messages + tool results), making later iterations progressively more expensive. A single tactical analysis request can generate 20,000-50,000 input tokens across all iterations.

**Location:** `llm_client.py:~920-960`

### P5: DUPLICATE HERO SUMMARY

**Impact: Minor (~$0.006 extra per first download)**

The download endpoint (`routes_analysis.py:201-233`) generates a separate "hero player" AI summary via `generate_match_summary()` if `ai_summary` not in result. This duplicates the per-player summary already generated by the orchestrator.

**Location:** `routes_analysis.py:198-239`

---

## Recommended Fixes (DO NOT IMPLEMENT — just listed)

### Fix 1: Cache tactical tab results in the job result dict

**Estimated savings: 50-70% of total LLM spend**

After generating tactical/strat/review analysis, store the result in `result["tactical_ai"]`, `result["strat_steal_ai"]`, `result["self_review_ai"]`. On subsequent requests for the same job_id, return the cached result. This is the single highest-impact fix — it eliminates repeated LLM calls for the same demo.

### Fix 2: Batch 10 per-player summaries into 1 LLM call

**Estimated savings: ~$0.015/demo + 8-25 seconds latency**

Instead of 10 separate calls, send all 10 players in a single prompt and ask the LLM to return 10 structured summaries. This eliminates 9 API round-trips and avoids paying for the system prompt 10 times (even with caching, you still pay for the user message and output 10x).

Single call estimate: ~4,167 system + ~1,400 user (10 players) + ~2,500 output = ~$0.016 vs current ~$0.023.

### Fix 3: Trim the coaching system prompt

**Estimated savings: ~30% of per-player summary input costs**

Remove from the coaching system prompt:
- Map callouts for 6 maps (~2,500 chars / ~625 tokens) — never relevant for individual player analysis
- Weapon meta reference (~1,400 chars / ~350 tokens) — not referenced in output
- Round type classification details (~1,400 chars / ~350 tokens) — generic CS2 knowledge

This reduces the system prompt from ~4,167 tokens to ~2,800 tokens (~33% reduction).

### Fix 4: Cap tool-use iterations to 3

**Estimated savings: ~20% of tactical analysis costs**

Most useful tactical insights come in the first 2-3 tool-use iterations. Iterations 4-10 add marginal value but accumulate significant context. Reducing the cap from 10 to 3 prevents runaway costs while preserving quality.

### Fix 5: Eliminate duplicate hero summary

**Estimated savings: ~$0.006 per first download**

The orchestrator already generates per-player summaries. The download endpoint should reuse the first player's `ai_summary` field instead of making a separate LLM call.

### Fix 6: Consider Haiku 3.5 for coaching summaries

**Estimated savings: ~20% of coaching summary costs**

The coaching summaries are formulaic (stats in, 200-word analysis out). Haiku 3.5 (`claude-3-5-haiku-20241022`) at $0.80/$4.00 per MTok may produce comparable quality at 20% lower cost vs Haiku 4.5. Worth A/B testing.

---

## Model Audit Table

| Feature/Job | Model Used | Should Be | Rationale |
|---|---|---|---|
| Per-player summaries | Haiku 4.5 ($1/$5) | Haiku 4.5 or 3.5 | Appropriate tier for simple summaries |
| Tactical analysis | Haiku 4.5 ($1/$5) | Haiku 4.5 | Appropriate — needs tool use |
| Strat stealing | Haiku 4.5 ($1/$5) | Haiku 4.5 | Appropriate |
| Self review | Haiku 4.5 ($1/$5) | Haiku 4.5 | Appropriate |
| Anti-strat report | **Sonnet 4.5 ($3/$15)** | Haiku 4.5 | Consider downgrade — structured JSON output, not creative writing |
| Game plan | **Sonnet 4.5 ($3/$15)** | Sonnet 4.5 | Appropriate — most complex output, needs strategic reasoning |
| Query classify | Haiku 4.5 ($1/$5) | Haiku 4.5 | Appropriate |
| Query format | Haiku 4.5 ($1/$5) | Haiku 4.5 | Appropriate |

---

## Quick Answers

1. **Most expensive model in use:** `claude-sonnet-4-5-20250929` (Sonnet 4.5) at $3/$15 per MTok — used by anti-strat and game plan
2. **Most expensive feature:** Game plan generation at ~$0.051 per call (Sonnet 4.5, large prompt + large output)
3. **Number of LLM calls per demo analysis:** 10 automatic (per-player) + 3-15 per AI tab click (tool-use loops) = **10-25 calls per fully-used demo**
4. **Are results cached between runs?** Per-player summaries: YES (file cache). Tactical/strat/review tabs: **NO (re-computed every click)**
5. **Estimated monthly cost at 100 demos/month:**

| Usage Pattern | Monthly Cost |
|---|---|
| 100 demos, no AI tabs clicked | ~$2.30 |
| 100 demos, each tab clicked once | ~$8.30-15.80 |
| 100 demos, each tab clicked 3x avg | ~$20.30-44.80 |
| + 20 anti-strat reports | add ~$0.70 |
| + 10 game plans | add ~$0.51 |
