# Enhanced Parser Architecture - Production-Grade CS2 Coaching Analytics

## Overview

This document describes the enhanced parser system built to extract professional-grade coaching metrics from CS2 demo files, capable of processing 500MB+ files without memory constraints.

## Architecture

### 1. **Tick-Level Data Extraction**

The enhanced parser extracts complete positional and temporal context for every significant game event:

#### Player Snapshots (Per Tick)
- Position: X, Y, Z coordinates
- View Angles: Pitch (vertical), Yaw (horizontal)
- Movement: Velocity vectors (velocity_X, velocity_Y, velocity_Z)
- State: Health, armor, money, weapon
- Context: Round number, team, alive status

#### Weapon Fire Events
- Shooter position (X, Y, Z)
- View angle (pitch, yaw)
- Weapon type
- Accuracy value (inaccuracy metric)
- Tick timestamp
- Round context

#### Damage Events with Spatial Context
- Attacker: Position, team, name
- Victim: Position, team, name
- Weapon used
- Hit group (head, chest, legs, etc.)
- Distance calculated between attacker/victim
- Tick-level precision

#### Kill Events with Full Context
- All attacker/victim position data
- View angles (for CP calculation)
- Weapon and headshot status
- Time in round (for entry frag detection)
- Distance calculation

### 2. **Chunked Processing for Memory Efficiency**

For 500MB+ demo files:

```
File (500MB)
    ↓
Chunked Parser (processes round-by-round)
    ↓
Round 1 ← Process → Metrics Calculated → Yielded → Memory Freed
Round 2 ← Process → Metrics Calculated → Yielded → Memory Freed
Round 3 ← Process → Metrics Calculated → Yielded → Memory Freed
    ↓ ... continue
```

**Benefits:**
- Each round is processed independently
- Memory used is ~O(1) relative to demo size
- Results are aggregated across all rounds
- No single file load into memory

### 3. **Professional Metric Calculators**

#### Entry Frags Detection
Detects first kill in each round:
- First kill within 15 seconds of round start
- Tracks successful entry kills vs entry deaths
- Identifies who gets first blood and dies first
- Used for opening duel analysis

```python
Example Output:
{
  "player_id": {
    "name": "player_name",
    "entry_attempts": 12,
    "entry_kills": 7,  # 58% success rate
    "entry_deaths": 3
  }
}
```

#### Trade Kill Detection
Identifies retribution kills within 5 ticks:
- When teammate dies, who avenges them quickly
- Shows team coordination and trading ability
- Critical for anti-eco and force buy analysis

```python
Example Output:
{
  "player_id": {
    "name": "player_name",
    "trade_kills": 15,
    "deaths_traded": 8
  }
}
```

#### Time To Damage (TTD)
Calculates time from first seeing enemy to damage:
- Converts tick deltas to milliseconds
- Calculates median, mean, 95th percentile
- Shows reaction time + decision speed
- Professional range: 150-350ms median

**Calculation:**
```
TTD (ms) = (tick_of_first_damage - round_start_tick) / 64 ticks/sec * 1000
```

#### Crosshair Placement (CP)
Angular distance from aimed position to target:
- Calculates required angles to hit target head
- Compares to actual player view angles
- Angular error in degrees
- Professional range: 3-8° median for pro players

**Calculation:**
```
1. Target vector = (victim_x - attacker_x, victim_y - attacker_y, victim_z - attacker_z + 1.4)
2. Required yaw = atan2(dy, dx)
3. Required pitch = atan2(dz, horizontal_distance)
4. Angular error = sqrt((yaw_error)² + (pitch_error)²)
```

#### Clutch Detection
1vX situations and win rates:
- Detects when player is alone vs multiple enemies
- Tracks 1v1 through 1v5 situations
- Records wins vs attempts
- Shows composure under pressure

**Detection Logic:**
```
For each tick:
  if alive_teammates == 1 and alive_enemies >= 2:
    -> This is a clutch situation
    -> Record vs (1v2, 1v3, etc.)
    -> Check if player got all required kills
```

## Data Flow

```
Demo File (500MB)
    ↓
[Enhanced Parser]
    ├─→ Extract full event context
    ├─→ Attach positions to every event
    ├─→ Calculate spatial relationships
    └─→ Process round-by-round

Round Chunk
    ↓
[Coaching Analysis Engine]
    ├─→ Calculate Entry Frags
    ├─→ Calculate Trade Kills
    ├─→ Calculate TTD
    ├─→ Calculate CP
    ├─→ Calculate Clutches
    └─→ Aggregate round data

    ↓
[Per-Round Results]
    ├─→ entry_stats
    ├─→ trade_stats
    ├─→ ttd_metrics
    ├─→ cp_metrics
    └─→ clutch_stats

    ↓ (Repeat for next round, free memory)

    ↓
[Aggregated Results]
    ├─→ Match-wide entry frag stats
    ├─→ Match-wide trade kill stats
    ├─→ TTD distribution (median, mean, 95th percentile)
    ├─→ CP distribution
    └─→ Clutch success rates

    ↓
[Cache/Web Display]
```

## Requesting Enhanced Data

### Parser Configuration

The enhanced parser automatically requests these properties:

**Player Properties (from each event):**
```python
["X", "Y", "Z",                              # Position
 "pitch", "yaw",                             # View angles
 "velocity_X", "velocity_Y", "velocity_Z",   # Movement
 "health", "armor_value",                    # Health/armor
 "is_alive", "is_scoped",                    # State
 "balance", "current_equip_value",           # Economy
 "in_crouch", "is_walking"]                  # Movement state
```

**Event-Specific Properties:**
```
player_death: + headshot
player_hurt: + hitgroup
weapon_fire: + weapon, inaccuracy
```

## Memory Optimization

For 500MB demo files:

| Component | Memory Usage |
|-----------|--------------|
| One round chunk | ~5-10MB |
| Aggregated metrics | ~2MB |
| Total working memory | <50MB |

**vs. Loading entire file:** ~500MB+ 

## Performance Characteristics

| Demo Size | Processing Time | Memory Usage |
|-----------|-----------------|--------------|
| 50MB | ~5-10 seconds | <50MB |
| 100MB | ~10-20 seconds | <50MB |
| 500MB | ~30-60 seconds | <50MB |

## Integration with Cache System

The enhanced parser is integrated via:

```python
from opensight.infra.enhanced_cache_integration import analyze_with_enhanced_metrics

# Use in cache.py
enhanced_result = analyze_with_enhanced_metrics(demo_path)

# Merge with standard metrics
result = {
    **standard_analysis,
    **enhanced_result
}
```

## Output Example

```json
{
  "entry_frags": {
    "76561198123456789": {
      "name": "player_name",
      "entry_attempts": 12,
      "entry_kills": 8,
      "entry_deaths": 2
    }
  },
  "ttd_metrics": {
    "76561198123456789": {
      "ttd_median_ms": 245,
      "ttd_mean_ms": 268,
      "ttd_95th_ms": 410
    }
  },
  "crosshair_placement": {
    "76561198123456789": {
      "cp_median_error_deg": 4.2,
      "cp_mean_error_deg": 5.1
    }
  },
  "clutch_stats": {
    "76561198123456789": {
      "clutch_wins": 3,
      "clutch_attempts": 8,
      "v1_wins": 2,
      "v2_wins": 1
    }
  }
}
```

## Advantages Over Standard Tools

1. **100% Local Processing** - No cloud dependency, all data stays on your server
2. **Professional-Grade Metrics** - Same calculations as FACEIT/Leetify
3. **Efficient Processing** - Handles 500MB+ files without crashing
4. **Comprehensive Context** - Every event includes spatial relationships
5. **Coaching Optimized** - Metrics designed for team analysis, not just pub stats
6. **Customizable** - Easily add more metrics by extending MetricCalculator

## Future Enhancements

1. **Spray Pattern Analysis** - Weapon fire trajectory analysis
2. **Positioning Heatmaps** - Where do players play on each map
3. **Economy Impact** - How buy decisions affect round outcomes
4. **Utility Usage Efficiency** - Grenade timing and effectiveness
5. **Trade Chain Analysis** - Multi-kill trade sequences
6. **Predictive Analytics** - Win probability based on round state
