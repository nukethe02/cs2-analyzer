# CS2 Demo Analyzer

Universal Counter-Strike 2 demo analysis tool for competitive teams. Works with Valve MM, FACEIT, ESEA, HLTV, scrims, and POV demos. Built for 67 Esports.

## Code Style

- Python 3.10+ with type hints on all function parameters
- Use dataclasses for configuration and data structures
- Detailed docstrings on every public function explaining purpose, params, and returns
- Named exports preferred, avoid wildcard imports
- Constants in UPPER_SNAKE_CASE at module level
- Error handling: never crash on bad data, collect warnings, return defaults for missing values
- Use safe accessors (`.get()`, try/except) when parsing demo events

## Commands

- `pip install -r requirements.txt` - Install dependencies
- `pip install -e ".[dev]"` - Install with dev dependencies
- `PYTHONPATH=src uvicorn cs2analyzer.api:app --port 7860` - Run web API
- `PYTHONPATH=src pytest tests/ -v` - Run all tests
- `ruff check src/` - Run linting
- `mypy src/` - Type checking

## Architecture

```
src/cs2analyzer/
├── __init__.py           # Package exports
├── api.py                # FastAPI web interface (port 7860)
├── parsers/
│   └── demo_parser.py    # Universal demo parser (auto-detects source)
├── analytics/
│   └── engine.py         # All metric calculations (HLTV rating, trades, etc.)
├── exporters/
│   └── formats.py        # JSON/CSV/SQLite export
└── utils/
    └── __init__.py       # Sharecode decoding, map names, weapon data
```

Other key files:
- `tests/` - pytest test suite
- `pyproject.toml` - Package config, dependencies, tool settings
- `Dockerfile` - Container build (exposes 7860)

## Key Dependencies

- **demoparser2** - Rust-backed CS2 demo parser (very fast)
- **pandas** - DataFrame manipulation for events
- **numpy** - Numerical calculations
- **fastapi** - Web API framework
- **uvicorn** - ASGI server

## Metrics Implementation

### HLTV 2.0 Rating Formula
```python
rating = (
    0.2448 * (kpr / 0.679) +
    0.2048 * (spr / 0.317) +
    0.1976 * (rmk / 0.317) +
    0.2336 * (impact / 0.317) +
    0.1192 * (adr / 80 / 0.317)
)
```
Benchmarks: <0.85 poor | 0.95-1.05 average | 1.10-1.25 good | >1.30 elite

### Trade Detection
5-second window. A trade occurs when a kill avenges a same-team death within the window.

### KAST%
Union of rounds where player had: Kill, Assist, Survived, or was Traded.
Benchmarks: <60% poor | 65-72% average | 73-80% good | >80% elite

## Demo Source Handling

Parser auto-detects source from server name and demo metadata:
- **Valve MM** (Premier, Competitive, Wingman) - Full data
- **FACEIT / ESEA** - Full data, server name contains platform ID
- **HLTV / Pro Matches** - Full GOTV data
- **POV Demos** - ~80% data, missing utility events, warn user about limitations

## Important Notes

- NEVER crash on malformed demo data - use try/except, return empty/zero defaults
- POV demos have incomplete utility data - always check `is_complete` flag
- Demo files can be 500MB+ - validate file size before processing
- Player names can contain injection characters - sanitize before display/export
- The tool runs offline with no network requests for security
- All file paths must be validated to prevent path traversal attacks
- Only accept `.dem` and `.dem.gz` file extensions

## Common Development Tasks

### Adding a New Metric
1. Add calculation in `analytics/engine.py` in `DemoAnalyzer` class
2. Add result field to `PlayerMatchStats` dataclass
3. Include in `JSONExporter.export_stats()`
4. Add to API response schema in `api.py`
5. Write test in `tests/test_analytics.py`

### Supporting a New Demo Source
1. Add detection logic in `_detect_source()` method
2. Add enum value to `DemoSource`
3. Handle any source-specific event differences

### Adding New Event Parsing
1. Add event name to `CORE_EVENTS` or `OPTIONAL_EVENTS` list
2. Create extraction method if complex parsing needed
3. Add DataFrame to `ParsedDemo` dataclass
4. Use in analytics calculations

## API Endpoints

- `GET /` - Web interface (HTML)
- `GET /health` - Health check, returns version and supported sources
- `POST /analyze` - Upload and analyze demo file (multipart form)
  - Query param: `include_ticks` (bool) - Include tick-by-tick position data (slower)

## Testing

Tests use pytest with `PYTHONPATH=src`. Mock demo data for unit tests, skip integration tests that require actual demo files with `@pytest.mark.skip`.

## Known Limitations

1. Clutch detection approximates from kill sequence, may miss edge cases
2. POV demos missing reliable utility data
3. Economy tracking not yet implemented (requires tick data)
4. Position heatmaps require `include_ticks=True` which is slower

## Future Improvements

See GitHub issues. Priority areas:
- Economy tracking (eco/force/full buy detection)
- Position heatmaps with map overlays
- Multi-demo aggregation for player profiles
- Team-level statistics
