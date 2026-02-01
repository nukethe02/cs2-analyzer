---
description: Guide for adding a new metric to CS2 Analyzer
---

# Adding a New Metric to CS2 Analyzer

This is a multi-file change. Follow this checklist EXACTLY.

## Pre-Implementation Questions

Before starting, answer:
1. **What metric are you adding?** (name, description)
2. **What data inputs does it need?** (kills, damages, positions, etc.)
3. **What's the output type?** (float, int, dataclass)
4. **What's the benchmark?** (elite/good/average/poor thresholds)

## Implementation Checklist

### Step 1: Create Calculation (`analysis/metrics.py` or new module)

```python
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class YourMetricResult:
    """Result of your metric calculation."""
    value: float
    sample_size: int
    # Add relevant fields

def calculate_your_metric(
    kills_df: pd.DataFrame,
    damages_df: Optional[pd.DataFrame] = None,
) -> YourMetricResult:
    """
    Calculate your metric.

    Args:
        kills_df: DataFrame with kill events
        damages_df: Optional DataFrame with damage events

    Returns:
        YourMetricResult with calculated values
    """
    # ALWAYS use safe accessors
    from opensight.core.utils import safe_float, safe_int

    # Handle empty data
    if kills_df is None or len(kills_df) == 0:
        return YourMetricResult(value=0.0, sample_size=0)

    # Your calculation here
    # Handle NaN values explicitly

    return YourMetricResult(value=result, sample_size=count)
```

### Step 2: Integrate in Analytics (`analysis/analytics.py`)

1. Import the new metric function
2. Add field to `PlayerMatchStats` dataclass
3. Call calculation in the `analyze()` method
4. Store result in player stats

### Step 3: Expose in API (`api.py`)

1. Add to `player_stats_to_dict()` function:
```python
"your_metric": stats.your_metric,
```

2. Add to `build_player_response()` if structured output needed

3. Document in `/about` endpoint response

### Step 4: Add CLI Display (`cli.py`)

```python
def _display_your_metric_metrics(stats: PlayerMatchStats) -> None:
    """Display your metric in CLI output."""
    console = Console()
    table = Table(title="Your Metric")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Your Metric", f"{stats.your_metric:.2f}")
    console.print(table)
```

### Step 5: Write Tests (`tests/test_metrics.py`)

```python
def test_your_metric_valid_data():
    """Test metric with valid data."""
    kills_df = pd.DataFrame({...})
    result = calculate_your_metric(kills_df)
    assert result.value > 0
    assert result.sample_size > 0

def test_your_metric_empty_data():
    """Test metric with empty data."""
    result = calculate_your_metric(pd.DataFrame())
    assert result.value == 0.0
    assert result.sample_size == 0

def test_your_metric_nan_handling():
    """Test metric handles NaN values."""
    kills_df = pd.DataFrame({"value": [1.0, float('nan'), 3.0]})
    result = calculate_your_metric(kills_df)
    # Should not crash, should handle gracefully
```

### Step 6: Verify

```cmd
ruff format src/ tests/ && ruff check --fix src/ tests/
set PYTHONPATH=src && pytest tests/test_metrics.py -v
```

## Existing Metrics for Reference

| Metric | File | Calculation |
|--------|------|-------------|
| TTD (Time to Damage) | metrics.py:301+ | ms from engagement to damage |
| Crosshair Placement | metrics.py:500+ | Angular error in degrees |
| HLTV Rating | hltv_rating.py | Industry standard formula |
| Trade Detection | analytics.py | 5-second window |

## Common Pitfalls

- [ ] Forgetting to use safe_* accessors
- [ ] Not handling empty DataFrames
- [ ] Not handling NaN values in calculations
- [ ] Missing test for edge cases
- [ ] Forgetting to update api.py
