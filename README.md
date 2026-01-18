---
title: OpenSight CS2 Analytics
emoji: 🎮
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OpenSight

**Local CS2 Analytics Framework** - Professional-grade metrics without cloud dependencies.

OpenSight is a locally-operated analytics system for Counter-Strike 2 that eliminates reliance on paid cloud services. Process replay files entirely on your computer with zero subscription costs and zero privacy compromises.

## Features

- **Demo Analysis**: Parse CS2 .dem files and extract tick-level game data
- **Share Code Decoding**: Decode match share codes to extract metadata
- **Replay Watching**: Monitor your replays folder for automatic processing
- **Professional Metrics**:
  - **Time to Damage (TTD)**: Latency between spotting an enemy and dealing damage
  - **Crosshair Placement (CP)**: Angular distance between aim and target position
  - Kill/Death statistics, headshot percentage, damage per round

## Installation

```bash
# Clone the repository
git clone https://github.com/nukethe02/cs2-analyzer.git
cd cs2-analyzer

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- [demoparser2](https://github.com/LaihoE/demoparser) - Rust-backed parser for CS2 demos
- [awpy](https://github.com/pnxenopoulos/awpy) - Geometric calculations (optional, for advanced visibility checks)

## Quick Start

### Analyze a Demo

```bash
# Basic analysis
opensight analyze /path/to/demo.dem

# Filter to specific player
opensight analyze /path/to/demo.dem --player "PlayerName"

# Export results to JSON
opensight analyze /path/to/demo.dem --output results.json

# Calculate specific metrics only
opensight analyze /path/to/demo.dem --metrics ttd
```

### Decode a Share Code

```bash
opensight decode "CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx"
```

### Watch for New Replays

```bash
# Watch default CS2 replays folder
opensight watch

# Watch custom folder
opensight watch --folder /path/to/replays

# Watch without auto-analysis
opensight watch --no-analyze
```

### Check Environment

```bash
opensight info
```

## Python API

```python
from opensight import DemoParser, calculate_ttd, calculate_crosshair_placement

# Parse a demo
parser = DemoParser("/path/to/demo.dem")
data = parser.parse()

print(f"Map: {data.map_name}")
print(f"Duration: {data.duration_seconds:.1f}s")

# Calculate metrics
ttd_results = calculate_ttd(data)
for steam_id, ttd in ttd_results.items():
    print(f"{ttd.player_name}: {ttd.mean_ttd_ms:.0f}ms average TTD")

cp_results = calculate_crosshair_placement(data)
for steam_id, cp in cp_results.items():
    print(f"{cp.player_name}: {cp.placement_score:.1f} placement score")
```

### Watch for Replays Programmatically

```python
from opensight import ReplayWatcher, DemoParser

watcher = ReplayWatcher()

@watcher.on_new_demo
def handle_demo(event):
    print(f"New demo: {event.file_path}")
    parser = DemoParser(event.file_path)
    data = parser.parse()
    # Process...

watcher.start(blocking=True)
```

### Decode Share Codes

```python
from opensight import decode_sharecode

info = decode_sharecode("CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx")
print(f"Match ID: {info.match_id}")
print(f"Outcome ID: {info.outcome_id}")
```

## Architecture

```
src/opensight/
├── __init__.py       # Package exports
├── cli.py            # Command-line interface (Typer)
├── sharecode.py      # Share code encoding/decoding
├── parser.py         # Demo file parsing (demoparser2 wrapper)
├── metrics.py        # TTD, Crosshair Placement calculations
└── watcher.py        # File system monitoring (watchdog)
```

## Metrics Explained

### Time to Damage (TTD)

Measures the latency between first seeing an enemy and dealing damage. Lower values indicate faster reactions and better aim.

**Calculation**: For each damage event, trace back through tick data to find when the attacker first had line-of-sight to the victim, then compute the time difference.

### Crosshair Placement (CP)

Measures how well a player keeps their crosshair positioned near potential enemy locations. Lower angles indicate better placement.

**Calculation**: At regular intervals, compute the angle between the player's view direction and the direction to the nearest visible enemy. The placement score is derived from the mean angle using exponential decay.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/

# Type checking
mypy src/
```

## How It Works

Unlike cloud-based analytics services, OpenSight:

1. **No Steam Bot Required**: Instead of automating demo downloads, users manually download replays in-game. The watchdog pattern monitors the local replays folder for new files.

2. **Local Processing**: All analysis happens on your machine. No data is sent to external servers.

3. **Zero Cost**: No subscriptions, no API limits, no usage tracking.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [demoparser2](https://github.com/LaihoE/demoparser) - Fast CS2 demo parsing
- [awpy](https://github.com/pnxenopoulos/awpy) - CS analytics library
- [watchdog](https://github.com/gorakhargosh/watchdog) - File system monitoring
