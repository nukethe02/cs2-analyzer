"""
Polars-Optimized Operations for CS2 Demo Analysis

This module provides high-performance DataFrame operations using Polars
when available, with automatic fallback to pandas.

Key features:
- LazyFrame support for large tick data (memory efficient)
- Optimized filtering and groupby operations
- Parallel execution for multi-core systems
- Efficient serialization to Parquet/Feather

Usage:
    from opensight.infra.polars_ops import PolarsAnalyzer, convert_demo_data_to_polars

    # Convert existing DemoData to use Polars DataFrames
    demo_data = convert_demo_data_to_polars(demo_data)

    # Use optimized analyzer
    analyzer = PolarsAnalyzer(demo_data)
    stats = analyzer.compute_player_stats()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Check for Polars availability
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

# pandas is always available
import pandas as pd

if TYPE_CHECKING:
    from opensight.core.parser import DemoData


def is_polars_dataframe(df: Any) -> bool:
    """Check if df is a Polars DataFrame or LazyFrame."""
    if not POLARS_AVAILABLE:
        return False
    return isinstance(df, (pl.DataFrame, pl.LazyFrame))


def is_pandas_dataframe(df: Any) -> bool:
    """Check if df is a pandas DataFrame."""
    return isinstance(df, pd.DataFrame)


def ensure_polars(df: pd.DataFrame | pl.DataFrame | None) -> pl.DataFrame | None:
    """Convert DataFrame to Polars if needed."""
    if df is None:
        return None
    if not POLARS_AVAILABLE:
        raise ImportError("Polars is required but not installed")
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Cannot convert {type(df)} to Polars DataFrame")


def ensure_pandas(df: pd.DataFrame | pl.DataFrame | None) -> pd.DataFrame | None:
    """Convert DataFrame to pandas if needed."""
    if df is None:
        return None
    if isinstance(df, pd.DataFrame):
        return df
    if POLARS_AVAILABLE:
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        if isinstance(df, pl.LazyFrame):
            return df.collect().to_pandas()
    raise TypeError(f"Cannot convert {type(df)} to pandas DataFrame")


# ============================================================================
# Optimized DataFrame Operations
# ============================================================================


class PolarsOps:
    """
    Polars-optimized DataFrame operations for CS2 analytics.

    These operations are designed to be significantly faster than pandas
    equivalents, especially for large tick data.
    """

    @staticmethod
    def filter_by_steamid(
        df: pl.DataFrame,
        column: str,
        steamid: int,
    ) -> pl.DataFrame:
        """Filter DataFrame by steamid, handling type coercion."""
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        # Handle potential float/int mismatch
        return df.filter(
            (pl.col(column).cast(pl.Float64) == float(steamid)) | (pl.col(column) == steamid)
        )

    @staticmethod
    def filter_by_round(
        df: pl.DataFrame,
        round_col: str,
        round_num: int,
    ) -> pl.DataFrame:
        """Filter DataFrame by round number."""
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")
        return df.filter(pl.col(round_col) == round_num)

    @staticmethod
    def group_kills_by_round(
        df: pl.DataFrame,
        attacker_col: str,
        round_col: str,
    ) -> pl.DataFrame:
        """
        Count kills per player per round (for multi-kill detection).

        Returns DataFrame with: steamid, round, kill_count
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        return (
            df.group_by([attacker_col, round_col])
            .agg(pl.len().alias("kill_count"))
            .sort([attacker_col, round_col])
        )

    @staticmethod
    def compute_damage_stats(
        df: pl.DataFrame,
        attacker_col: str,
        damage_col: str,
    ) -> pl.DataFrame:
        """
        Compute total damage per player.

        Returns DataFrame with: steamid, total_damage, damage_count
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        return df.group_by(attacker_col).agg(
            [
                pl.col(damage_col).sum().alias("total_damage"),
                pl.len().alias("damage_count"),
            ]
        )

    @staticmethod
    def find_first_kill_per_round(
        df: pl.DataFrame,
        round_col: str,
        tick_col: str = "tick",
    ) -> pl.DataFrame:
        """
        Find the first kill of each round (for opening duel detection).

        Returns DataFrame with first kill row from each round.
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        return df.sort(tick_col).group_by(round_col).first()

    @staticmethod
    def window_join_damages_to_kills(
        kills_df: pl.DataFrame,
        damages_df: pl.DataFrame,
        attacker_col: str,
        victim_col: str,
        window_ticks: int = 2000,
    ) -> pl.DataFrame:
        """
        Join damages to kills within a tick window (for TTD computation).

        This is a common pattern for computing Time to Damage:
        - For each kill, find the first damage event from same attacker to same victim
        - Within a reasonable tick window before the kill

        Returns kills_df with added first_damage_tick column.
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        # Use asof join for efficient temporal matching
        # Sort both by tick first
        kills_sorted = kills_df.sort("tick")
        damages_sorted = damages_df.sort("tick")

        # Create join keys
        kills_with_key = kills_sorted.with_columns(
            [
                (pl.col(attacker_col).cast(pl.Utf8) + "_" + pl.col(victim_col).cast(pl.Utf8)).alias(
                    "join_key"
                )
            ]
        )
        damages_with_key = damages_sorted.with_columns(
            [
                (pl.col(attacker_col).cast(pl.Utf8) + "_" + pl.col(victim_col).cast(pl.Utf8)).alias(
                    "join_key"
                )
            ]
        ).rename({"tick": "damage_tick"})

        # For each kill, find the earliest damage in the window
        # Using a grouped approach for better performance
        result_rows = []
        for key in kills_with_key["join_key"].unique():
            kill_rows = kills_with_key.filter(pl.col("join_key") == key)
            dmg_rows = damages_with_key.filter(pl.col("join_key") == key)

            for kill_row in kill_rows.iter_rows(named=True):
                kill_tick = kill_row["tick"]
                # Find first damage within window
                valid_dmg = dmg_rows.filter(
                    (pl.col("damage_tick") <= kill_tick)
                    & (pl.col("damage_tick") >= kill_tick - window_ticks)
                ).sort("damage_tick")

                first_dmg_tick = valid_dmg["damage_tick"][0] if len(valid_dmg) > 0 else None
                row_dict = dict(kill_row)
                row_dict["first_damage_tick"] = first_dmg_tick
                result_rows.append(row_dict)

        if result_rows:
            return pl.DataFrame(result_rows)
        return kills_df.with_columns(pl.lit(None).alias("first_damage_tick"))

    @staticmethod
    def compute_side_stats_vectorized(
        kills_df: pl.DataFrame,
        damages_df: pl.DataFrame,
        attacker_col: str,
        attacker_side_col: str,
        damage_col: str,
        steamid: int,
    ) -> dict[str, dict[str, int]]:
        """
        Compute CT-side and T-side stats for a player in a vectorized manner.

        Returns dict with CT and T stats (kills, deaths, damage).
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        result = {
            "CT": {"kills": 0, "deaths": 0, "damage": 0},
            "T": {"kills": 0, "deaths": 0, "damage": 0},
        }

        # Filter kills by player
        player_kills = PolarsOps.filter_by_steamid(kills_df, attacker_col, steamid)

        # Count by side
        for side in ["CT", "T"]:
            side_kills = player_kills.filter(
                pl.col(attacker_side_col).str.to_uppercase().str.contains(side)
            )
            result[side]["kills"] = len(side_kills)

        # Damage stats if available
        if damages_df is not None and len(damages_df) > 0:
            player_dmg = PolarsOps.filter_by_steamid(damages_df, attacker_col, steamid)
            for side in ["CT", "T"]:
                side_dmg = player_dmg.filter(
                    pl.col(attacker_side_col).str.to_uppercase().str.contains(side)
                )
                if damage_col in side_dmg.columns:
                    result[side]["damage"] = int(side_dmg[damage_col].sum() or 0)

        return result


# ============================================================================
# Lazy Operations for Large Tick Data
# ============================================================================


class PolarsLazyOps:
    """
    LazyFrame operations for memory-efficient processing of large tick data.

    LazyFrames allow query optimization and streaming execution,
    making it possible to process tick data larger than available RAM.
    """

    @staticmethod
    def scan_tick_data(path: Path) -> pl.LazyFrame:
        """Lazy-scan tick data from Parquet file."""
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")
        return pl.scan_parquet(path)

    @staticmethod
    def filter_player_ticks_lazy(
        lf: pl.LazyFrame,
        steamid_col: str,
        steamid: int,
        tick_range: tuple[int, int] | None = None,
    ) -> pl.LazyFrame:
        """
        Lazily filter tick data for a specific player.

        This uses predicate pushdown - the filter is applied during scan,
        avoiding loading unneeded data into memory.
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        expr = pl.col(steamid_col) == steamid
        if tick_range:
            expr = expr & (pl.col("tick") >= tick_range[0]) & (pl.col("tick") <= tick_range[1])

        return lf.filter(expr)

    @staticmethod
    def compute_positions_lazy(
        lf: pl.LazyFrame,
        x_col: str = "X",
        y_col: str = "Y",
        z_col: str = "Z",
    ) -> pl.LazyFrame:
        """
        Compute position-derived metrics lazily.

        Adds: distance_moved, velocity_magnitude
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        return lf.with_columns(
            [
                # Velocity magnitude
                (
                    pl.col("velocity_X").pow(2)
                    + pl.col("velocity_Y").pow(2)
                    + pl.col("velocity_Z").pow(2)
                )
                .sqrt()
                .alias("velocity_magnitude"),
            ]
        )

    @staticmethod
    def aggregate_player_positions_lazy(
        lf: pl.LazyFrame,
        steamid_col: str,
        x_col: str = "X",
        y_col: str = "Y",
    ) -> pl.LazyFrame:
        """
        Aggregate position data per player for heatmap generation.

        Returns: steamid, avg_x, avg_y, positions_count
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars required")

        return lf.group_by(steamid_col).agg(
            [
                pl.col(x_col).mean().alias("avg_x"),
                pl.col(y_col).mean().alias("avg_y"),
                pl.col(x_col).count().alias("positions_count"),
            ]
        )


# ============================================================================
# DemoData Conversion
# ============================================================================


def convert_demo_data_to_polars(demo_data: DemoData) -> DemoData:
    """
    Convert DemoData DataFrames from pandas to Polars.

    This enables using Polars-optimized operations for analysis.

    Args:
        demo_data: DemoData instance with pandas DataFrames

    Returns:
        Same DemoData instance with Polars DataFrames
    """
    if not POLARS_AVAILABLE:
        logger.warning("Polars not available, returning original DemoData")
        return demo_data

    # Convert each DataFrame attribute
    df_attrs = [
        "kills_df",
        "damages_df",
        "rounds_df",
        "weapon_fires_df",
        "blinds_df",
        "grenades_df",
        "bomb_events_df",
        "ticks_df",
    ]

    for attr in df_attrs:
        df = getattr(demo_data, attr, None)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            setattr(demo_data, attr, pl.from_pandas(df))
            logger.debug(f"Converted {attr} to Polars ({len(df)} rows)")

    return demo_data


def convert_demo_data_to_pandas(demo_data: DemoData) -> DemoData:
    """
    Convert DemoData DataFrames from Polars back to pandas.

    Args:
        demo_data: DemoData instance with Polars DataFrames

    Returns:
        Same DemoData instance with pandas DataFrames
    """
    if not POLARS_AVAILABLE:
        return demo_data

    df_attrs = [
        "kills_df",
        "damages_df",
        "rounds_df",
        "weapon_fires_df",
        "blinds_df",
        "grenades_df",
        "bomb_events_df",
        "ticks_df",
    ]

    for attr in df_attrs:
        df = getattr(demo_data, attr, None)
        if df is not None:
            if isinstance(df, pl.DataFrame):
                setattr(demo_data, attr, df.to_pandas())
                logger.debug(f"Converted {attr} to pandas")
            elif isinstance(df, pl.LazyFrame):
                setattr(demo_data, attr, df.collect().to_pandas())
                logger.debug(f"Converted {attr} LazyFrame to pandas")

    return demo_data


# ============================================================================
# Polars-Optimized Analyzer
# ============================================================================


@dataclass
class PolarsPlayerStats:
    """Player statistics computed using Polars."""

    steamid: int
    name: str = ""
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    headshots: int = 0
    total_damage: int = 0
    ct_kills: int = 0
    t_kills: int = 0
    kills_per_round: dict[int, int] = field(default_factory=dict)


class PolarsAnalyzer:
    """
    High-performance demo analyzer using Polars.

    This analyzer provides the same metrics as DemoAnalyzer but uses
    Polars for significantly faster computation on large datasets.
    """

    def __init__(self, demo_data: DemoData, convert_dataframes: bool = True):
        """
        Initialize the Polars analyzer.

        Args:
            demo_data: Parsed demo data
            convert_dataframes: If True, convert pandas DataFrames to Polars
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is required for PolarsAnalyzer")

        self.data = demo_data
        if convert_dataframes:
            self.data = convert_demo_data_to_polars(demo_data)

        self._kills_df = ensure_polars(self.data.kills_df)
        self._damages_df = ensure_polars(self.data.damages_df)
        self._ticks_df = (
            ensure_polars(self.data.ticks_df) if self.data.ticks_df is not None else None
        )

        # Detect column names
        self._att_col = self._find_col(self._kills_df, ["attacker_steamid", "attacker_steam_id"])
        self._vic_col = self._find_col(self._kills_df, ["victim_steamid", "user_steamid"])
        self._round_col = self._find_col(
            self._kills_df, ["round_num", "total_rounds_played", "round"]
        )
        self._att_side_col = self._find_col(self._kills_df, ["attacker_side", "attacker_team_name"])

    def _find_col(self, df: pl.DataFrame | None, options: list[str]) -> str | None:
        """Find first matching column."""
        if df is None:
            return None
        for col in options:
            if col in df.columns:
                return col
        return None

    def compute_player_stats(self) -> dict[int, PolarsPlayerStats]:
        """
        Compute basic player statistics using optimized Polars operations.

        This is significantly faster than pandas for large datasets.
        """
        stats: dict[int, PolarsPlayerStats] = {}

        if self._kills_df is None or len(self._kills_df) == 0:
            return stats

        # Initialize stats for all players
        for steamid, name in self.data.player_names.items():
            stats[steamid] = PolarsPlayerStats(steamid=steamid, name=name)

        # Vectorized kill counting
        if self._att_col:
            kill_counts = self._kills_df.group_by(self._att_col).agg(pl.len().alias("kills"))
            for row in kill_counts.iter_rows(named=True):
                sid = int(row[self._att_col])
                if sid in stats:
                    stats[sid].kills = row["kills"]

        # Vectorized death counting
        if self._vic_col:
            death_counts = self._kills_df.group_by(self._vic_col).agg(pl.len().alias("deaths"))
            for row in death_counts.iter_rows(named=True):
                sid = int(row[self._vic_col])
                if sid in stats:
                    stats[sid].deaths = row["deaths"]

        # Headshot counting
        if self._att_col and "headshot" in self._kills_df.columns:
            hs_counts = (
                self._kills_df.filter(pl.col("headshot"))
                .group_by(self._att_col)
                .agg(pl.len().alias("headshots"))
            )
            for row in hs_counts.iter_rows(named=True):
                sid = int(row[self._att_col])
                if sid in stats:
                    stats[sid].headshots = row["headshots"]

        # Multi-kill rounds
        if self._att_col and self._round_col:
            kills_per_round = PolarsOps.group_kills_by_round(
                self._kills_df, self._att_col, self._round_col
            )
            for row in kills_per_round.iter_rows(named=True):
                sid = int(row[self._att_col])
                if sid in stats:
                    round_num = row[self._round_col]
                    kill_count = row["kill_count"]
                    stats[sid].kills_per_round[round_num] = kill_count

        # Damage stats
        if self._damages_df is not None and len(self._damages_df) > 0:
            dmg_att_col = self._find_col(
                self._damages_df, ["attacker_steamid", "attacker_steam_id"]
            )
            dmg_col = self._find_col(self._damages_df, ["dmg_health", "damage", "dmg"])

            if dmg_att_col and dmg_col:
                damage_stats = PolarsOps.compute_damage_stats(
                    self._damages_df, dmg_att_col, dmg_col
                )
                for row in damage_stats.iter_rows(named=True):
                    sid = int(row[dmg_att_col])
                    if sid in stats:
                        stats[sid].total_damage = row["total_damage"]

        logger.info(f"Computed stats for {len(stats)} players using Polars")
        return stats

    def compute_opening_duels(self) -> dict[int, dict[str, int]]:
        """
        Detect opening duels (first kill of each round).

        Returns dict of steamid -> {wins: int, losses: int}
        """
        result: dict[int, dict[str, int]] = {}

        if self._kills_df is None or self._round_col is None:
            return result

        # Initialize for all players
        for steamid in self.data.player_names:
            result[steamid] = {"wins": 0, "losses": 0}

        # Get first kill per round
        first_kills = PolarsOps.find_first_kill_per_round(self._kills_df, self._round_col)

        for row in first_kills.iter_rows(named=True):
            if self._att_col and self._att_col in row:
                att_id = int(row[self._att_col])
                if att_id in result:
                    result[att_id]["wins"] += 1

            if self._vic_col and self._vic_col in row:
                vic_id = int(row[self._vic_col])
                if vic_id in result:
                    result[vic_id]["losses"] += 1

        return result

    def compute_ttd_fast(self, max_window_ticks: int = 2000) -> list[dict]:
        """
        Compute Time to Damage using optimized Polars operations.

        This is faster than the pandas-based TTD computation for large datasets.

        Returns list of TTD results with: attacker, victim, ttd_ms, tick
        """
        results = []

        if self._kills_df is None or self._damages_df is None:
            return results

        if len(self._kills_df) == 0 or len(self._damages_df) == 0:
            return results

        MS_PER_TICK = 1000 / 64  # CS2 tick rate

        # Use the optimized window join
        try:
            joined = PolarsOps.window_join_damages_to_kills(
                self._kills_df,
                self._damages_df,
                self._att_col or "attacker_steamid",
                self._vic_col or "victim_steamid",
                max_window_ticks,
            )

            for row in joined.iter_rows(named=True):
                first_dmg_tick = row.get("first_damage_tick")
                if first_dmg_tick is not None:
                    kill_tick = row["tick"]
                    ttd_ticks = kill_tick - first_dmg_tick
                    ttd_ms = ttd_ticks * MS_PER_TICK

                    results.append(
                        {
                            "attacker": row.get(self._att_col),
                            "victim": row.get(self._vic_col),
                            "ttd_ms": ttd_ms,
                            "ttd_ticks": ttd_ticks,
                            "kill_tick": kill_tick,
                            "first_damage_tick": first_dmg_tick,
                        }
                    )

            logger.info(f"Computed {len(results)} TTD values using Polars")

        except Exception as e:
            logger.warning(f"Polars TTD computation failed: {e}")

        return results


# ============================================================================
# Performance Comparison
# ============================================================================


def compare_performance(demo_data: DemoData, iterations: int = 5) -> dict[str, float]:
    """
    Compare pandas vs Polars performance for demo analysis.

    Args:
        demo_data: Parsed demo data
        iterations: Number of iterations for timing

    Returns:
        Dict with timing results
    """
    import time

    results = {}

    # Pandas timing
    pandas_data = convert_demo_data_to_pandas(demo_data)
    from opensight.analysis.analytics import DemoAnalyzer as PandasAnalyzer

    start = time.perf_counter()
    for _ in range(iterations):
        analyzer = PandasAnalyzer(pandas_data)
        analyzer._init_player_stats()
        analyzer._calculate_basic_stats()
    results["pandas_basic_stats_ms"] = (time.perf_counter() - start) / iterations * 1000

    # Polars timing
    if POLARS_AVAILABLE:
        polars_data = convert_demo_data_to_polars(demo_data)

        start = time.perf_counter()
        for _ in range(iterations):
            analyzer = PolarsAnalyzer(polars_data, convert_dataframes=False)
            _ = analyzer.compute_player_stats()
        results["polars_basic_stats_ms"] = (time.perf_counter() - start) / iterations * 1000

        if results.get("polars_basic_stats_ms", 0) > 0:
            results["speedup"] = results["pandas_basic_stats_ms"] / results["polars_basic_stats_ms"]

    return results
