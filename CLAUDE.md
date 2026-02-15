# OpenSight - CS2 Demo Analyzer

Local Counter-Strike 2 analytics framework providing Leetify/Scope.gg quality metrics without cloud dependencies. Works with Valve MM, FACEIT, ESEA, HLTV, scrims, and POV demos.

## THE RULE: NO IMAGINARY CODE

This codebase was damaged by AI inventing field names. Before touching ANY code that reads player/round/match data:

1. Read `analysis/models.py` — confirm the attribute exists on the dataclass
2. Read `pipeline/orchestrator.py` — confirm the serialized dict key matches
3. If a field isn't in models.py, IT DOES NOT EXIST
4. Trace: dataclass attr → orchestrator key → route response → frontend JS

## Known Landmines — VERIFY BEFORE TOUCHING

- Null checks: `if x is not None` NOT `if x` (0.0 is valid ADR)
- Trades: nested `p.trades.kills_traded` NOT flat `p.trade_kills`
- Clutches: nested `p.clutches.total_wins` NOT flat `p.clutch_wins`
- Players dict: keyed by steam_id string NOT a list
- Hero player: `player.get("stats", {}).get("kills")` NOT `player.get("kills")`
- AI modules: `"demo_info"` NOT `"match_info"`
- Round number: `"round_num"` NOT `"round_number"`
- Economy: nested `r["economy"]["ct"]["equipment"]` NOT flat `r["ct_team_money"]`
- Opening duels: `p.opening_duels.wins` / `.losses` NOT `p.entry_kills` / `p.entry_deaths`
- Trade stats: `p.trades.trade_kill_success_pct` NOT `_rate_pct`
- Utility stats: `p.utility.flashbangs_thrown` NOT `flashes_thrown`
- TTD fields: `ttd_median_ms` = engagement duration, NOT reaction time
- `cp_median_error_deg` on PlayerMatchStats, NOT `cp_median_deg`
- Damage: `dmg_health` is OVERKILL damage — capped at `user_health` in parser. Do NOT use `dmg_health_real` (doesn't exist)
- After ANY correction: update this section with the lesson learned

## Before Making Any Changes (CRITICAL)

**Always run these commands before committing:**

```bash
# 1. Format code
ruff format src/ tests/

# 2. Fix linting issues
ruff check --fix src/ tests/

# 3. Run tests (REQUIRED - do not skip)
PYTHONPATH=src pytest tests/ -v

# 4. If you modified a specific module, run its tests first:
PYTHONPATH=src pytest tests/test_api.py -v        # API changes
PYTHONPATH=src pytest tests/test_analytics.py -v  # Analytics changes
PYTHONPATH=src pytest tests/test_metrics.py -v    # Metric changes
```

**If tests fail, fix them before committing. Never push broken tests.**

## Quick Reference Commands

| Command | Description |
|---------|-------------|
| `pip install -e ".[dev]"` | Install with dev dependencies |
| `PYTHONPATH=src uvicorn opensight.api:app --port 7860 --reload` | Run API (dev mode) |
| `PYTHONPATH=src pytest tests/ -v` | Run all tests |
| `PYTHONPATH=src pytest tests/ -v -k "test_name"` | Run specific test |
| `ruff format src/ tests/ && ruff check --fix src/ tests/` | Format + lint |
| `PYTHONPATH=src python scripts/smoke_test.py` | Verify all module imports |
| `PYTHONPATH=src python scripts/check_deployment.py` | Pre-deployment checklist |
| `opensight analyze demo.dem` | CLI analysis |
| `opensight watch` | Watch replays folder |

## Architecture

```
src/opensight/
├── __init__.py            # Package init, version (importlib.metadata)
├── cli.py                 # Typer CLI
├── server.py              # Uvicorn server entry
│
├── api/                   # FastAPI web app (package)
│   ├── __init__.py        # App creation, middleware, routers, job_store
│   ├── shared.py          # Pydantic models, validation, rate limiting, JobStore
│   ├── routes_analysis.py # POST /analyze, GET /analyze/{job_id}, AI endpoints
│   ├── routes_auth.py     # POST /auth/register, /auth/login, GET /auth/me
│   ├── routes_export.py   # GET /api/export/{job_id}/json, csv
│   ├── routes_heatmap.py  # GET /api/heatmap/{job_id}/kills, grenades
│   ├── routes_maps.py     # GET /maps, /radar/transform, /hltv/*, /replay
│   ├── routes_match.py    # GET /api/your-match/*, /api/players/*
│   └── routes_misc.py     # /health, /readiness, /decode, /feedback, /scouting
│
├── core/                  # Demo parsing layer
│   ├── parser.py          # DemoParser class (demoparser2/awpy backends)
│   ├── constants.py       # TICK_RATE, TRADE_WINDOW_MS, HLTV coefficients
│   ├── config.py          # Configuration management
│   ├── schemas.py         # Shared schema definitions
│   ├── utils.py           # safe_int, safe_str, safe_float, safe_bool
│   └── map_zones.py       # Map area definitions (A site, B site, Mid, etc.)
│
├── analysis/              # Metrics and analytics
│   ├── analytics.py       # DemoAnalyzer - orchestration skeleton (~3200 lines)
│   ├── models.py          # 28 dataclasses (MatchAnalysis, PlayerMatchStats, etc.)
│   ├── compute_combat.py  # Combat: trades, clutches, opening duels, multi-kills
│   ├── compute_aim.py     # Aim: TTD, crosshair placement, accuracy stats
│   ├── compute_economy.py # Economy: KAST, side stats, economy history
│   ├── compute_utility.py # Utility: grenade stats, mistakes, utility metrics
│   ├── highlights.py      # Match highlights (clutches, aces, multi-kills)
│   ├── hltv_rating.py     # HLTV 2.0 Rating formula
│   ├── metrics_optimized.py # Vectorized computations (numpy)
│   ├── persona.py         # Player identity classification
│   ├── positioning.py     # Positioning analysis (map control)
│   ├── rotation.py        # CT rotation latency analysis
│   └── tactical_service.py # Tactical analysis service
│
├── pipeline/              # Analysis orchestration
│   ├── orchestrator.py    # DemoOrchestrator (parse → analyze → serialize)
│   └── store_events.py    # Event storage (kills, damage, grenades, bombs)
│
├── domains/               # Domain-specific analysis
│   ├── combat.py          # Combat analytics (refrag, trade chains)
│   ├── economy.py         # Economy analytics (force buy detection)
│   ├── synergy.py         # Team synergy metrics
│   └── utility.py         # Utility analytics
│
├── auth/                  # Authentication system
│   ├── passwords.py       # Password hashing with bcrypt
│   ├── jwt.py             # JWT token generation/validation
│   ├── middleware.py       # Auth middleware (get_current_user)
│   └── tiers.py           # User tier limits (Free, Pro, Team, Admin)
│
├── scouting/              # Opponent scouting
│   ├── engine.py          # Scouting report generation
│   ├── anti_strats.py     # Anti-stratting analysis
│   └── models.py          # Scouting data models
│
├── ai/                    # AI/LLM integrations
│   ├── llm_client.py      # Anthropic Claude API client
│   ├── self_review.py     # AI self-review of analyses
│   ├── strat_engine.py    # Strategy analysis engine
│   └── tactical.py        # Tactical analysis module
│
├── integrations/          # External services
│   ├── sharecode.py       # CS2 share code encode/decode
│   ├── hltv.py            # HLTV API (rankings, player data)
│   └── feedback.py        # Community feedback (SQLAlchemy-backed)
│
├── infra/                 # Infrastructure
│   ├── cache.py           # SHA256-based file caching (~540 lines)
│   ├── database.py        # SQLite (matches, players, events, jobs, feedback)
│   ├── job_store.py       # Persistent job tracking (PersistentJobStore)
│   └── parallel.py        # Batch demo processing
│
├── visualization/         # Output generation
│   ├── heatmaps.py        # Heatmap generation (kills, deaths, damage)
│   ├── exports.py         # Export to JSON, CSV, HTML
│   ├── replay.py          # 2D replay data extraction
│   └── radar.py           # Map coordinate transformation
│
└── static/                # Web UI
    ├── index.html         # Main interface (with Export dropdown, Highlights)
    ├── css/               # Stylesheets
    └── js/                # Frontend scripts

scripts/                   # Deployment & verification
├── smoke_test.py          # Import verification for all modules
├── check_deployment.py    # Pre-deployment checklist
└── worktree-setup.ps1     # Git worktree setup for parallel development
```

## API Endpoints

### Core Analysis
| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Upload demo, returns job_id (async) |
| GET | `/analyze/{job_id}` | Check job status |
| GET | `/analyze/{job_id}/download` | Download results when complete |
| GET | `/jobs` | List all analysis jobs |

### Your Match (Personal Dashboard)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/your-match/{demo_id}/{steam_id}` | Personalized match performance |
| POST | `/api/your-match/store` | Store match in player history |
| GET | `/api/your-match/baselines/{steam_id}` | Player 30-match averages |
| GET | `/api/your-match/history/{steam_id}` | Match history |
| GET | `/api/your-match/persona/{steam_id}` | Player identity persona |

### Maps & Replay
| Method | Path | Description |
|--------|------|-------------|
| GET | `/maps` | List available maps |
| GET | `/maps/{map_name}` | Map metadata and radar info |
| POST | `/radar/transform` | Game coords → radar pixels |
| POST | `/replay/generate` | Generate 2D replay data |

### HLTV Integration
| Method | Path | Description |
|--------|------|-------------|
| GET | `/hltv/rankings` | World team rankings |
| GET | `/hltv/map/{map_name}` | Map statistics |
| GET | `/hltv/player/search` | Search players by nickname |
| POST | `/hltv/enrich` | Enrich analysis with HLTV data |

### Utilities
| Method | Path | Description |
|--------|------|-------------|
| POST | `/decode` | Decode CS2 share code |
| GET | `/cache/stats` | Cache statistics |
| POST | `/cache/clear` | Clear analysis cache |
| POST | `/feedback` | Submit metric feedback |
| GET | `/health` | Health check |
| GET | `/readiness` | Orchestration readiness |
| GET | `/about` | API documentation |

## Key Metrics Implemented

### HLTV 2.0 Rating
```python
# Formula in hltv_rating.py
Rating = 0.0073*KAST + 0.3591*KPR - 0.5329*DPR + 0.2372*Impact + 0.0032*ADR + 0.1587*RMK
```

### Time to Damage (TTD)
Milliseconds from engagement start to damage dealt. Located in `analysis/compute_aim.py`.
- Elite: <200ms | Good: 200-350ms | Average: 350-500ms | Slow: >500ms

### Crosshair Placement (CP)
Angular error in degrees. Located in `analysis/compute_aim.py`.
- Elite: <5° | Good: 5-15° | Average: 15-25° | Needs work: >25°

### Trade Detection
5-second window (industry standard). Constants in `core/constants.py`:
```python
TRADE_WINDOW_MS = 5000
TRADE_WINDOW_TICKS = 320  # At 64 tick
```

### Player Personas (analysis/persona.py)
- **The Opener**: High entry success rate
- **The Anchor**: Clutch specialist
- **The Cleanup**: Trade kill expert
- **The Utility Master**: High utility effectiveness
- **The Fragger**: High K/D and impact

## Code Style Requirements

1. **Type hints on all functions** - Required for mypy
2. **Dataclasses for data structures** - Not dicts or tuples
3. **Safe accessors for demo data** - Never trust raw values:
   ```python
   from opensight.core.utils import safe_int, safe_str, safe_float, safe_bool

   kills = safe_int(row.get("kills"), default=0)
   name = safe_str(row.get("player_name"), default="Unknown")
   ```
4. **No wildcard imports** - Use named imports only
5. **Constants in UPPER_SNAKE_CASE** - Define at module level

## Error Handling Rules

**Never crash on malformed demo data.** All parsing must:
1. Use safe accessor functions
2. Wrap in try/except with sensible defaults
3. Log warnings, don't raise exceptions
4. Return partial results rather than failing completely

## Security Controls

| Control | Location | Notes |
|---------|----------|-------|
| CSP frame-ancestors | `api.py` middleware | Allows HF Spaces embedding |
| Rate limiting | `api.py` @rate_limit | 5/min uploads, 3/min replays |
| File size limit | `api.py` | 500MB max |
| Extension validation | `api.py` | Only .dem, .dem.gz |
| Input validation | `api.py` | validate_steam_id, validate_demo_id |
| XSS sanitization | `static/index.html` | escapeHtml() function |
| Temp file cleanup | `api.py` | Always in finally block |

## Database Schema (SQLite)

Located in `infra/database.py`. Tables:
- `match_history` - Per-match player stats
- `player_baselines` - Rolling 30-match averages
- `player_personas` - Current persona assignment
- `feedback` - User feedback on metrics

## Testing

```bash
# Full test suite
PYTHONPATH=src pytest tests/ -v

# With coverage
PYTHONPATH=src pytest tests/ --cov=opensight --cov-report=html

# Specific test file
PYTHONPATH=src pytest tests/test_api.py -v

# Specific test by name
PYTHONPATH=src pytest tests/ -v -k "test_hltv_rating"
```

### Test Files
| File | Tests |
|------|-------|
| `test_api.py` | API endpoints, security headers |
| `test_analytics.py` | DemoAnalyzer, metrics engine |
| `test_hltv_rating.py` | HLTV 2.0 formula accuracy |
| `test_metrics.py` | TTD, CP, utility calculations |
| `test_replay.py` | 2D replay generation |
| `test_sharecode.py` | Share code encode/decode |
| `test_your_match.py` | Personal dashboard features |

## Common Tasks

### Adding a New Metric
1. Add calculation in `analysis/analytics.py` or create new module
2. Add field to relevant dataclass (e.g., `PlayerMatchStats`)
3. Include in `player_stats_to_dict()` in `api.py`
4. Add to `build_player_response()` for structured output
5. Write test in `tests/`
6. Run `ruff format` and `pytest` before committing

### Adding a New API Endpoint
1. Add route in `api.py`
2. Add input validation (use existing `validate_*` functions)
3. Add rate limiting if it's resource-intensive
4. Add to this documentation
5. Write test in `test_api.py`

### Modifying Security Headers
1. Edit `security_headers_middleware()` in `api.py`
2. Update corresponding test in `test_api.py` `TestSecurityHeaders` class
3. Verify with: `PYTHONPATH=src pytest tests/test_api.py -v -k "security"`

## Deployment

### Hugging Face Spaces
- Dockerfile exposes port 7860
- Multi-stage build for optimization
- Non-root user execution
- Health check at `/health`
- Readiness check at `/readiness`

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENSIGHT_CACHE_DIR` | Cache directory | `~/.opensight/cache` |
| `OPENSIGHT_DB_PATH` | SQLite database path | `~/.opensight/opensight.db` |
| `OPENSIGHT_LOG_LEVEL` | Logging level | `INFO` |

## Troubleshooting

**Import Error: demoparser2**
```bash
pip install demoparser2
```

**Module not found: opensight**
```bash
export PYTHONPATH=src  # Unix
set PYTHONPATH=src     # Windows
# or install in dev mode:
pip install -e .
```

**Tests fail with import errors**
```bash
pip install -e ".[dev]"
```

**Port 7860 in use**
```bash
uvicorn opensight.api:app --port 8000
```

## File Structure Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package config, dependencies, ruff/mypy settings |
| `requirements.txt` | Pip dependencies for deployment |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff, mypy, bandit) |
| `Dockerfile` | Multi-stage build for HF Spaces |
| `README.md` | HF Spaces frontmatter + public docs |
| `CLAUDE.md` | This file - AI assistant instructions |

---

## Workflow Best Practices (Boris Cherny Method)

### Custom Skills Available

Use these slash commands for common tasks:

| Command | Description |
|---------|-------------|
| `/run-tests` | Run test suite with proper PYTHONPATH |
| `/format` | Format + lint with ruff |
| `/commit` | Create conventional commit |
| `/pr` | Create pull request |
| `/grill-me` | Ruthless self-review before PR |
| `/prove-it-works` | Verify changes with evidence |
| `/techdebt` | Scan for technical debt |
| `/plan-architect` | Staff engineer planning mode |
| `/plan-review` | Second opinion on plans |
| `/plan-reset` | Scrap and implement elegant solution |
| `/add-metric` | Guide for adding new metrics |
| `/add-endpoint` | Guide for adding API endpoints |
| `/fix-ci` | Diagnose and fix CI failures |
| `/security-audit` | OWASP Top 10 security review |
| `/deploy-hf` | Deploy to HuggingFace Spaces |
| `/check-logs` | Debug deployment issues |
| `/learn-mistake` | Capture lessons in CLAUDE.md |

### For Complex Tasks

1. Start with `/plan-architect` - pour energy into the plan
2. Run `/plan-review` to get second opinion
3. Only then implement - should be a 1-shot

### Before Every PR

1. `/grill-me` - critical self-review
2. `/prove-it-works` - verify behavior with evidence
3. Only create PR when both pass

### After Every Mistake

End corrections with: "Update CLAUDE.md so you don't make that mistake again"
Use `/learn-mistake` skill to capture the lesson.

### Daily Routine

- **Start**: Check worktrees, pick focus area
- **During**: Use skills, not raw commands
- **End**: `/techdebt` to find accumulated debt

### Parallel Development

Use 3-5 git worktrees for independent tracks:
```powershell
.\scripts\worktree-setup.ps1
```

Creates:
- `ai-coaching` - LLM features
- `api-endpoints` - Backend work
- `ui-frontend` - Visualization
- `bugfix-hotfix` - Quick fixes
- `performance` - Optimization

Each worktree = separate Claude session. Never block feature work with bugfixes.

---

## Lessons Learned (Auto-Updated)

### Demo Parsing Pitfalls
- [ ] ALWAYS use `safe_int()`, `safe_float()`, `safe_str()` from `core/utils.py`
- [ ] Demo files can have NaN positions - check before calculations
- [ ] Tick rates vary: 64 (MM), 128 (FACEIT) - use TICK_RATE constant

### API Development
- [ ] Rate limit decorator MUST come BEFORE @app.route
- [ ] Security headers in `security_headers_middleware()` - test with `test_api.py::TestSecurityHeaders`
- [ ] Every endpoint change needs `test_api.py` update

### Testing on Windows
- [ ] Use `set PYTHONPATH=src` not `export PYTHONPATH=src`
- [ ] Pytest needs `-v --tb=short` for readable output

### Common Mistakes
<!-- Claude: Add new lessons here when you make mistakes via /learn-mistake -->

---

## Module-Specific Gotchas

### api/ package (7 route modules + shared.py)
- Changes often need test_api.py updates
- Rate limiting decorator order matters
- Security headers tested in TestSecurityHeaders
- `from opensight.api import app, job_store, sharecode_cache` is the critical import

### analytics.py (~3,200 lines — delegation skeleton)
- Uses safe_* accessors throughout
- Metric changes cascade to PlayerMatchStats in models.py
- Delegates to compute_combat.py, compute_aim.py, compute_economy.py, compute_utility.py

### ai/llm_client.py
- Requires ANTHROPIC_API_KEY environment variable
- Uses Claude Sonnet for match analysis

### cli.py
- All metrics display functions now implemented (utility, trade, opening duels)
- Uses Rich library for formatted output

---

## Known Technical Debt

### TODOs
All major TODOs resolved as of 2026-02-01:
- ~~`cli.py:425` - Implement utility metrics display~~ DONE
- ~~`cli.py:432` - Implement trade metrics display~~ DONE
- ~~`cli.py:439` - Implement opening duel metrics display~~ DONE
- ~~`ai/coaching.py:886` - Implement recalculate_stats()~~ DONE

### Debug Artifacts
Cleaned up as of 2026-02-01:
- ~~`api.py:861-870` - Timeline debug logging~~ Converted to logger.debug()
- ~~`api.py:942-948` - Download endpoint debug logging~~ Converted to logger.debug()
- ~~`cache.py:686-695` - Round timeline debug logging~~ Converted to logger.debug()

Use `/techdebt` to scan for new technical debt.

---

## Pipeline Wiring Checklist

**CRITICAL:** Every time you add a new metric, field, or feature, complete this checklist BEFORE considering the task done:

- [ ] **1. MODEL**: Field added to appropriate dataclass in `analysis/models.py`
- [ ] **2. COMPUTE**: Computation implemented in appropriate `compute_*.py` module
- [ ] **3. COMPUTE → MODEL**: Computation result stored on the model instance
- [ ] **4. ORCHESTRATOR**: Field serialized in the player output dict in `pipeline/orchestrator.py`
  - Use `None` (not `0`) for missing/unavailable data
  - Verify the key name matches what the frontend expects
  - Check `docs/SERIALIZATION_CONTRACT.md` for key naming conventions
- [ ] **5. FRONTEND**: JavaScript reads the correct key and handles null properly
  - Use explicit null check: `value != null` (NOT `value ? ...`)
  - Display "N/A" for null values, not "-" or "0"
- [ ] **6. CONTRACT**: Entry added to `docs/SERIALIZATION_CONTRACT.md`
- [ ] **7. TEST**: `tests/test_e2e_pipeline.py` passes (or updated if needed)
- [ ] **8. SMOKE TEST**: `python scripts/pipeline_smoke_test.py <demo>` shows ✅ for new field

**If ANY step is skipped, the feature is NOT done.** A computed metric that isn't serialized is the same as an unimplemented metric from the user's perspective.

---

## Serialization Rules

### Rule 1: Never use 0 as a sentinel for missing data

**BAD:**
```python
"ttd_median_ms": round(val, 1) if val is not None else 0
```

**GOOD:**
```python
"ttd_median_ms": round(val, 1) if val is not None else None
```

**Why:** The frontend JavaScript treats `0` as falsy in conditionals (`if (value)`), making it indistinguishable from missing data. Use `None` for missing data and let the frontend display "N/A".

### Rule 2: Every output field must have model backing

- Do not add ad-hoc fields to the output dict without a corresponding model field or explicit computation
- If a field is computed (not stored on model), document it in the orchestrator as a comment

### Rule 3: Consult the serialization contract

Before adding a new field to orchestrator output:
1. Check `docs/SERIALIZATION_CONTRACT.md` for existing similar fields
2. Follow the established key naming convention (e.g., `stats.kills`, `advanced.ttd_median_ms`)
3. Add your new field to the contract after implementing it

### Rule 4: Nested vs flat structure

Follow existing patterns:
- **Basic stats**: `"stats": { "kills": ..., "deaths": ... }`
- **Ratings**: `"rating": { "hltv_rating": ..., "aim_rating": ... }`
- **Advanced metrics**: `"advanced": { "ttd_median_ms": ..., "cp_median_error_deg": ... }`
- **Utility**: `"utility": { "flashbangs_thrown": ..., "flash_assists": ... }`
- **Combat**: `"duels": { "trade_kills": ..., "clutch_wins": ... }`

Do NOT create new top-level keys unless there's a strong architectural reason.

---

## Name Resolution Rules

### Background

demoparser2 provides player names, but they can be invalid (empty, special characters, or raw SteamIDs). The parser generates fallback names like `Player_0810` for invalid names. However, other code paths (kill events, utility events) may use raw names without this validation.

### Rules

1. **ALWAYS validate player names** with `is_valid_player_name()` before displaying them
2. **Use fallback generation** if validation fails: `make_fallback_player_name(steamid)`
3. **NEVER use event names directly** in display output:
   - `kill.attacker_name` may contain raw SteamIDs
   - `kill.victim_name` may contain raw SteamIDs
   - `blind_event.attacker_name` may be invalid
4. **Canonical names are in `player_names` dict** (steamid → name)
   - Always try `player_names.get(steamid)` first
   - Only fall back to event names if player_names lookup fails
   - Then validate the fallback before display

### Example

**BAD:**
```python
killer_name = kill["attacker_name"]  # May be "76561197960287930" (raw SteamID)
```

**GOOD:**
```python
# Lookup canonical name first
killer_name = player_names.get(kill["attacker_steamid"])
if not killer_name:
    # Fallback to event name, but validate
    killer_name = kill.get("attacker_name", "Unknown")
    if not is_valid_player_name(killer_name):
        killer_name = make_fallback_player_name(kill["attacker_steamid"])
```

---

## SteamID Handling Rules

### Background

demoparser2 returns steamids in **different formats** depending on event context:
- **Player info events**: Steam64 format (17 digits, starts with `7656...`)
- **Kill events**: May use account ID (shorter format)
- **Grenade/blind events**: May use yet another format

This means `kill.attacker_steamid` and `player_info.steamid` for the same player **may not match directly**.

### Rules

1. **NEVER assume steamid formats match** across event types
2. **Prefer NAME-based matching** over steamid matching when correlating events:
   - Flash assist detection: Match by player name, not steamid
   - Trade detection: Match by player name, not steamid
   - Opening duel correlation: Match by player name, not steamid
3. **If steamid matching is required**, build a normalization function:
   ```python
   def normalize_steamid(steamid: int | str) -> str:
       """Convert various steamid formats to canonical Steam64 string."""
       # Implementation depends on observed formats in your data
       pass
   ```
4. **For cross-event correlation**, use RAW TICK proximity instead of steamid matching when possible

### Known Issue

In the golden demo:
- 1/10 kill event steamids matched `player_persistent_teams` (10% success rate)
- 10/10 grenade event steamids matched `player_names` (100% success rate)

**Lesson:** Grenade events have reliable steamids. Kill events do NOT. Always use name-based matching for kill event correlation.

---

## Deprecated Field Migration

### Current Deprecations

- **`player_teams`** (parser.py:531) is DEPRECATED → use **`player_persistent_teams`** instead
  - `player_teams` can change mid-match if players switch
  - `player_persistent_teams` uses the side they played more (stable)

### Migration Process

When you deprecate a field:

1. **Add a comment** at the field definition:
   ```python
   # DEPRECATED: use player_persistent_teams instead
   player_teams: dict = {}
   ```

2. **Search for ALL usages**:
   ```bash
   grep -rn "player_teams" src/
   ```

3. **Migrate ALL usages in the same commit**
   - Do NOT leave some code paths on the old field
   - Do NOT split migration across multiple commits (creates inconsistency window)

4. **Add a test** that verifies old field is no longer used:
   ```python
   def test_no_deprecated_field_usage():
       """Ensure deprecated player_teams is not used."""
       # Grep source code for player_teams (excluding the deprecation comment)
       pass
   ```

### When You Find Deprecated Field Usage

If you encounter code using a deprecated field:
1. **Check the comment** for the replacement field
2. **Migrate that usage immediately** (don't defer)
3. **Verify the new field has the same semantics** (or adapt logic if different)
4. **Test the migration** with a demo file

---

## Testing Requirements

### After ANY Code Change

**Run these tests BEFORE committing:**

```bash
# 1. E2E pipeline integrity test
PYTHONPATH=src pytest tests/test_e2e_pipeline.py -v

# 2. Module-specific tests (if you modified that module)
PYTHONPATH=src pytest tests/test_analytics.py -v  # Analytics changes
PYTHONPATH=src pytest tests/test_api.py -v        # API changes
PYTHONPATH=src pytest tests/test_metrics.py -v    # Metric changes
```

### After Adding a New Metric

**Run the smoke test on a real demo:**

```bash
PYTHONPATH=src python scripts/pipeline_smoke_test.py tests/fixtures/golden_master.dem
```

**Look for:**
- ✅ **PASS** on your new field (means data exists and is non-null)
- ⚠️ **PARTIAL** (some players have data, some don't — may be expected)
- ❌ **FAIL** (field is null/zero for all players — serialization gap)

**The smoke test takes <30 seconds** and gives immediate visual feedback on field coverage. A metric is NOT done until the smoke test shows ✅.

### Golden Demo Tests

The E2E pipeline test uses `tests/fixtures/golden_master.dem` to verify:
1. **Serialization completeness**: All model fields appear in output or are documented as skipped
2. **Frontend alignment**: Frontend only reads keys that exist in backend output
3. **No raw SteamIDs**: Player names and kill matrix contain valid names, not steamids
4. **Timeline integrity**: Both CT and T teams have data

If `test_e2e_pipeline.py` fails, it means you broke the data pipeline. Fix it before committing.

---

## Known demoparser2 Quirks

### 1. Team Numbers

Teams are **numeric, not strings**:
- `2` = Terrorist (T)
- `3` = CounterTerrorist (CT)

**Do NOT compare strings**: `if team == "T"` will always fail.

**Correct:**
```python
if team_number == 2:
    team_name = "T"
elif team_number == 3:
    team_name = "CT"
```

### 2. SteamID Format Variability

See "SteamID Handling Rules" section above. TL;DR: Prefer name-based matching over steamid matching.

### 3. Position Data Availability

Position data (`attacker_x`, `attacker_y`, `attacker_z`, `pitch`, `yaw`) may be **NaN for many kills**.

**Impact:**
- TTD (Time to Damage) computations may return `None` for ~29% of kills (unavoidable parser limitation)
- CP (Crosshair Placement) computations may return `None` for ~29% of kills

**This is EXPECTED, not a bug.** Display "N/A" for unavailable data, not "-" or "0".

**Coverage stats from Wave B findings:**
- ~71% of kills have position data (can compute TTD/CP)
- ~29% of kills lack position data (return `None`)

### 4. Flash Assist Detection

The `flash_assist` and `assister_steamid` columns **may not exist** in `kills_df` from demoparser2.

**Fallback strategy:**
1. First try reading `kills_df["flash_assist"]` column
2. If column doesn't exist, fall back to blind-event correlation:
   - Find blind events within 3 seconds before kill
   - Match by victim steamid
   - Use **NAME-based matching** for attacker (see SteamID Handling Rules)

### 5. Round Boundary Unreliability

`round_boundaries` from `build_round_boundaries()` can be unreliable:
- Negative time spans (end_tick < start_tick)
- Overlapping round starts
- Missing rounds in overtime

**Recommendation:** Use **RAW TICK proximity** instead of round-based matching when correlating cross-event data (e.g., "find grenades thrown near this kill").

### 6. Player Count Validation

Always validate player counts:
```python
if len(players) != 10:
    logger.warning(f"Expected 10 players, found {len(players)}")
```

Demos can have:
- Bots (may or may not have steamids)
- Spectators (may appear in some event types)
- Players who disconnected mid-match

Do NOT assume `len(players) == 10` without checking.

---

## Summary: How These Rules Prevent Bugs

| Bug Pattern | Prevented By | Example |
|-------------|--------------|---------|
| **Serialization gaps** | Pipeline Wiring Checklist step 4 + 7 | TTD computed but not in orchestrator output |
| **Name resolution failures** | Name Resolution Rules | Raw SteamIDs in kill matrix |
| **None→0→falsy display** | Serialization Rule 1 | Frontend shows "-" for 0 ADR vs missing ADR |
| **Deprecated field usage** | Deprecated Field Migration process | Code reads `player_teams` instead of `player_persistent_teams` |
| **Orphaned code** | Pipeline Wiring Checklist step 4 | `highlights.py` never imported in orchestrator |
| **SteamID format mismatch** | SteamID Handling Rules | Kill steamids don't match player info steamids |
| **Partial test coverage** | Testing Requirements | New metric breaks pipeline but no test catches it |

**The checklist is mandatory.** If you skip steps 4-7, you WILL create a serialization gap. If you skip step 8, you won't know until a user reports it.
