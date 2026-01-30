# OpenSight - CS2 Demo Analyzer

Local Counter-Strike 2 analytics framework providing professional-grade metrics without cloud dependencies. Works with Valve MM, FACEIT, ESEA, HLTV, scrims, and POV demos.

**RECENT FIXES**: TTD and Crosshair Placement now work without expensive tick data parsing. See `FIXES_DOCUMENTATION.md` for details.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web API (Hugging Face compatible)
PYTHONPATH=src uvicorn opensight.api:app --host 0.0.0.0 --port 7860

# Run CLI
pip install -e .
opensight analyze /path/to/demo.dem

# Test the workflow
python test_workflow.py
```

## Commands

| Command | Description |
|---------|-------------|
| `pip install -r requirements.txt` | Install all dependencies |
| `pip install -e .` | Install package in development mode |
| `pip install -e ".[dev]"` | Install with dev dependencies (pytest, ruff, mypy) |
| `PYTHONPATH=src uvicorn opensight.api:app --port 7860` | Run web API |
| `PYTHONPATH=src pytest tests/ -v` | Run tests |
| `python test_workflow.py` | Test complete workflow with local demo |
| `ruff check src/` | Run linting |
| `mypy src/` | Type checking |

## Architecture

```
src/opensight/
├── __init__.py        # Package exports (lazy imports for heavy deps)
├── api.py             # FastAPI web interface (port 7860)
├── cli.py             # Typer CLI (opensight command)
├── server.py          # Web server entry point
│── core/
│   ├── parser.py      # Demo parser using demoparser2 (optimized)
│   ├── constants.py   # Game constants
│   └── utils.py       # Utilities
├── analysis/
│   ├── analytics.py   # TTD and CP calculations (FIXED)
│   ├── metrics.py     # Metric calculation utilities (FIXED)
│   └── metrics_optimized.py  # Vectorized computations
├── integrations/
│   └── sharecode.py   # CS2 share code encode/decode
├── infra/
│   ├── watcher.py     # File system watcher for auto-processing
│   ├── cache.py       # Analysis caching
│   └── database.py    # SQLite history storage
└── static/
    └── index.html     # Web UI
```

## Key Files

- `pyproject.toml` - Package config, dependencies, tool settings
- `requirements.txt` - Pip dependencies for deployment
- `FIXES_DOCUMENTATION.md` - Details on recent performance fixes
- `test_workflow.py` - End-to-end test script
- `Dockerfile` - Container build (exposes 7860 for Hugging Face)
- `README.md` - Has Hugging Face YAML frontmatter for Spaces
- `tests/` - pytest test suite

## Dependencies

**Core:**
- `demoparser2>=0.9.0` - Rust-backed CS2 demo parser (fast)
- `pandas>=2.0.0` - DataFrame manipulation
- `numpy>=1.24.0` - Numerical calculations

**Web API:**
- `fastapi>=0.100.0` - Web framework
- `uvicorn>=0.23.0` - ASGI server
- `python-multipart>=0.0.6` - File uploads

**CLI:**
- `typer>=0.9.0` - CLI framework
- `rich>=13.0.0` - Terminal formatting
- `watchdog>=3.0.0` - File system monitoring

## Code Style

- Python 3.11+ with type hints on all function parameters
- Use dataclasses for data structures
- Detailed docstrings on public functions
- Named exports preferred, avoid wildcard imports
- Constants in UPPER_SNAKE_CASE at module level

## Error Handling

**Critical: Never crash on malformed demo data.**

Use safe accessor functions throughout:
```python
from opensight.parser import safe_int, safe_str, safe_bool
from opensight.analytics import safe_float

# Always use these when parsing demo events
value = safe_int(row.get("kills"), default=0)
name = safe_str(row.get("player_name"), default="Unknown")
```

All parsing code must use try/except and return sensible defaults.

## Security Measures

The codebase implements these security controls:

| Control | Location | Description |
|---------|----------|-------------|
| XSS Protection | `static/index.html` | `escapeHtml()` sanitizes player names |
| File Size Limit | `api.py` | 500MB max upload size |
| Extension Validation | `api.py` | Only `.dem` and `.dem.gz` accepted |
| Empty File Check | `api.py` | Rejects zero-byte uploads |
| Temp File Cleanup | `api.py` | Files deleted in finally block |
| Safe Accessors | `parser.py`, `analytics.py` | Prevent crashes on bad data |

## Metrics Implementation

### Time to Damage (TTD)
Measures latency from first damage to kill. Lower = faster reactions.
```python
ttd_ms = (kill_tick - first_damage_tick) * MS_PER_TICK
# MS_PER_TICK = 1000 / 64 ≈ 15.625ms
```
Benchmarks: <200ms elite | 200-350ms good | 350-500ms average | >500ms slow

### Crosshair Placement (CP)
Angular error between aim and enemy position at moment of kill.
```python
angular_error = math.degrees(math.acos(dot(view_vec, ideal_vec)))
```
Benchmarks: <5° elite | 5-15° good | 15-25° average | >25° needs work

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web interface (HTML) |
| GET | `/health` | Health check, returns version |
| GET | `/readiness` | Readiness check for orchestration (disk, temp dir, deps) |
| GET | `/about` | API documentation and metric descriptions |
| POST | `/decode` | Decode share code (JSON body: `{"code": "CSGO-..."}`) |
| POST | `/analyze` | Upload and analyze demo (multipart form) |

## Hugging Face Deployment

The project is configured for Hugging Face Spaces:

1. `README.md` has YAML frontmatter:
   ```yaml
   sdk: docker
   app_port: 7860
   ```

2. `Dockerfile` uses multi-stage build for optimization:
   - **Build stage**: Compiles Rust extensions (demoparser2), installs all dependencies
   - **Runtime stage**: Minimal image with only virtualenv and source files
   - Runs as non-root user for security
   - Includes Docker HEALTHCHECK for container orchestration

3. Uvicorn runs with production settings:
   - 1 worker (demo parsing is CPU-heavy)
   - Log level: warning (reduces verbosity)
   - Timeout keep-alive: 65s (above typical load balancer timeout)

4. Readiness endpoint (`/readiness`) checks:
   - Disk space (>100MB free)
   - Temp directory writable
   - Heavy dependencies importable (demoparser2, pandas, numpy)

5. To deploy: Push to a Hugging Face Space repository

## CLI Usage

```bash
# Analyze a demo file
opensight analyze /path/to/demo.dem

# Filter to specific player
opensight analyze demo.dem --player "PlayerName"

# Export to JSON
opensight analyze demo.dem --output results.json

# Decode a share code
opensight decode "CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx"

# Watch replays folder
opensight watch

# Check environment
opensight info
```

## Testing

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run specific test file
PYTHONPATH=src pytest tests/test_sharecode.py -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=opensight
```

Tests use mock data - no actual demo files required for unit tests.

## Before Committing (IMPORTANT)

**Always run these commands before committing any changes:**

```bash
# Format code with ruff
ruff format src/ tests/

# Fix linting issues
ruff check --fix src/ tests/

# Run tests to catch regressions
PYTHONPATH=src pytest tests/ -v
```

If any tests fail, fix them before committing. Do not push code that breaks tests.

## Common Development Tasks

### Adding a New Metric

1. Add calculation in `analytics.py` in `DemoAnalyzer` class
2. Add result field to `PlayerAnalytics` dataclass
3. Include in API response in `api.py`
4. Add display in `cli.py` if needed
5. Write test in `tests/`

### Adding New Event Parsing

1. Add parsing method in `parser.py` (e.g., `_parse_grenades`)
2. Add result to `DemoData` dataclass
3. Use safe accessors for all field access
4. Handle missing columns gracefully

## File Validation

The API only accepts these file types:
- `.dem` - CS2 demo files
- `.dem.gz` - Compressed demo files

Maximum file size: 500MB

## Known Limitations

1. POV demos have incomplete utility data
2. Clutch detection approximates from kill sequence
3. Economy tracking not yet implemented
4. Position heatmaps require tick-level parsing (slow)

## Troubleshooting

**Import Error: demoparser2**
```bash
pip install demoparser2
```

**Module not found: opensight**
```bash
export PYTHONPATH=src
# or
pip install -e .
```

**Port 7860 in use**
```bash
uvicorn opensight.api:app --port 8000
```
