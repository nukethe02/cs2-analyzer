"""
DataFrame Backend Abstraction for OpenSight

Provides a unified interface for DataFrame operations that works with both
pandas and Polars. Polars offers significant performance benefits for large
tick data (multi-gigabyte demos) through:

- LazyFrames for query optimization
- Better memory efficiency
- Parallel execution

Usage:
    from opensight.backend import get_backend, DataFrameBackend

    # Get the current backend (respects configuration)
    backend = get_backend()

    # Create a DataFrame
    df = backend.from_dict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Filter
    filtered = backend.filter(df, "col1", ">", 1)

    # Convert between backends
    pandas_df = backend.to_pandas(df)
    polars_df = backend.to_polars(pandas_df)

Configuration:
    Set OPENSIGHT_USE_POLARS=true or use_polars=True in config
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Lazy import flags
_POLARS_AVAILABLE: Optional[bool] = None
_PANDAS_AVAILABLE: Optional[bool] = None

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def _check_polars() -> bool:
    """Check if Polars is available (lazy check)."""
    global _POLARS_AVAILABLE
    if _POLARS_AVAILABLE is None:
        try:
            import polars  # noqa: F401
            _POLARS_AVAILABLE = True
        except ImportError:
            _POLARS_AVAILABLE = False
    return _POLARS_AVAILABLE


def _check_pandas() -> bool:
    """Check if pandas is available (lazy check)."""
    global _PANDAS_AVAILABLE
    if _PANDAS_AVAILABLE is None:
        try:
            import pandas  # noqa: F401
            _PANDAS_AVAILABLE = True
        except ImportError:
            _PANDAS_AVAILABLE = False
    return _PANDAS_AVAILABLE


@dataclass
class BackendConfig:
    """Configuration for DataFrame backend selection."""
    use_polars: bool = False
    lazy_mode: bool = True  # Use LazyFrames when possible
    cache_format: str = "parquet"  # 'parquet', 'feather', 'pickle'
    compression: str = "zstd"  # For Parquet: 'zstd', 'lz4', 'snappy', 'gzip'


# Global backend configuration
_backend_config: Optional[BackendConfig] = None


def get_backend_config() -> BackendConfig:
    """Get the global backend configuration."""
    global _backend_config
    if _backend_config is None:
        _backend_config = BackendConfig()
    return _backend_config


def set_backend_config(config: BackendConfig) -> None:
    """Set the global backend configuration."""
    global _backend_config
    _backend_config = config


def is_polars_available() -> bool:
    """Check if Polars is available for use."""
    return _check_polars()


def is_pandas_available() -> bool:
    """Check if pandas is available for use."""
    return _check_pandas()


class DataFrameBackend(ABC):
    """Abstract base class for DataFrame backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name ('pandas' or 'polars')."""
        pass

    @abstractmethod
    def from_dict(self, data: dict[str, list]) -> Any:
        """Create a DataFrame from a dictionary."""
        pass

    @abstractmethod
    def from_records(self, records: list[dict]) -> Any:
        """Create a DataFrame from a list of dictionaries."""
        pass

    @abstractmethod
    def empty(self, schema: Optional[dict[str, type]] = None) -> Any:
        """Create an empty DataFrame."""
        pass

    @abstractmethod
    def concat(self, dfs: list[Any], ignore_index: bool = True) -> Any:
        """Concatenate multiple DataFrames."""
        pass

    @abstractmethod
    def filter(self, df: Any, column: str, op: str, value: Any) -> Any:
        """Filter DataFrame by column condition."""
        pass

    @abstractmethod
    def groupby_agg(self, df: Any, by: list[str], aggs: dict[str, str]) -> Any:
        """Group by and aggregate."""
        pass

    @abstractmethod
    def sort(self, df: Any, by: list[str], descending: bool = False) -> Any:
        """Sort DataFrame by columns."""
        pass

    @abstractmethod
    def select(self, df: Any, columns: list[str]) -> Any:
        """Select specific columns."""
        pass

    @abstractmethod
    def to_pandas(self, df: Any) -> "pd.DataFrame":
        """Convert to pandas DataFrame."""
        pass

    @abstractmethod
    def to_polars(self, df: Any) -> "pl.DataFrame":
        """Convert to Polars DataFrame."""
        pass

    @abstractmethod
    def is_empty(self, df: Any) -> bool:
        """Check if DataFrame is empty."""
        pass

    @abstractmethod
    def len(self, df: Any) -> int:
        """Get number of rows."""
        pass

    @abstractmethod
    def columns(self, df: Any) -> list[str]:
        """Get column names."""
        pass

    @abstractmethod
    def iterrows(self, df: Any):
        """Iterate over rows (yields index, row as dict)."""
        pass

    @abstractmethod
    def get_column(self, df: Any, column: str) -> Any:
        """Get a single column as a series/list."""
        pass

    @abstractmethod
    def unique(self, df: Any, column: str) -> list:
        """Get unique values from a column."""
        pass

    # Serialization methods
    @abstractmethod
    def to_parquet(self, df: Any, path: Path, compression: str = "zstd") -> None:
        """Save DataFrame to Parquet format."""
        pass

    @abstractmethod
    def read_parquet(self, path: Path) -> Any:
        """Read DataFrame from Parquet format."""
        pass

    @abstractmethod
    def to_feather(self, df: Any, path: Path) -> None:
        """Save DataFrame to Feather/Arrow IPC format."""
        pass

    @abstractmethod
    def read_feather(self, path: Path) -> Any:
        """Read DataFrame from Feather/Arrow IPC format."""
        pass


class PandasBackend(DataFrameBackend):
    """pandas DataFrame backend implementation."""

    def __init__(self):
        if not _check_pandas():
            raise ImportError("pandas is required for PandasBackend")
        import pandas as pd
        self._pd = pd

    @property
    def name(self) -> str:
        return "pandas"

    def from_dict(self, data: dict[str, list]) -> "pd.DataFrame":
        return self._pd.DataFrame(data)

    def from_records(self, records: list[dict]) -> "pd.DataFrame":
        return self._pd.DataFrame.from_records(records)

    def empty(self, schema: Optional[dict[str, type]] = None) -> "pd.DataFrame":
        if schema:
            return self._pd.DataFrame({k: self._pd.Series(dtype=v) for k, v in schema.items()})
        return self._pd.DataFrame()

    def concat(self, dfs: list["pd.DataFrame"], ignore_index: bool = True) -> "pd.DataFrame":
        valid_dfs = [df for df in dfs if df is not None and not df.empty]
        if not valid_dfs:
            return self._pd.DataFrame()
        return self._pd.concat(valid_dfs, ignore_index=ignore_index)

    def filter(self, df: "pd.DataFrame", column: str, op: str, value: Any) -> "pd.DataFrame":
        if column not in df.columns:
            return df
        if op == "==":
            return df[df[column] == value]
        elif op == "!=":
            return df[df[column] != value]
        elif op == ">":
            return df[df[column] > value]
        elif op == ">=":
            return df[df[column] >= value]
        elif op == "<":
            return df[df[column] < value]
        elif op == "<=":
            return df[df[column] <= value]
        elif op == "in":
            return df[df[column].isin(value)]
        elif op == "contains":
            return df[df[column].astype(str).str.contains(str(value), na=False)]
        else:
            raise ValueError(f"Unknown operator: {op}")

    def groupby_agg(self, df: "pd.DataFrame", by: list[str], aggs: dict[str, str]) -> "pd.DataFrame":
        return df.groupby(by, as_index=False).agg(aggs)

    def sort(self, df: "pd.DataFrame", by: list[str], descending: bool = False) -> "pd.DataFrame":
        return df.sort_values(by=by, ascending=not descending)

    def select(self, df: "pd.DataFrame", columns: list[str]) -> "pd.DataFrame":
        existing = [c for c in columns if c in df.columns]
        return df[existing]

    def to_pandas(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df

    def to_polars(self, df: "pd.DataFrame") -> "pl.DataFrame":
        if not _check_polars():
            raise ImportError("polars is required to convert to Polars DataFrame")
        import polars as pl
        return pl.from_pandas(df)

    def is_empty(self, df: "pd.DataFrame") -> bool:
        return df is None or df.empty

    def len(self, df: "pd.DataFrame") -> int:
        return len(df) if df is not None else 0

    def columns(self, df: "pd.DataFrame") -> list[str]:
        return list(df.columns) if df is not None else []

    def iterrows(self, df: "pd.DataFrame"):
        if df is None or df.empty:
            return
        for idx, row in df.iterrows():
            yield idx, row.to_dict()

    def get_column(self, df: "pd.DataFrame", column: str) -> Any:
        if column in df.columns:
            return df[column]
        return None

    def unique(self, df: "pd.DataFrame", column: str) -> list:
        if column in df.columns:
            return df[column].dropna().unique().tolist()
        return []

    def to_parquet(self, df: "pd.DataFrame", path: Path, compression: str = "zstd") -> None:
        df.to_parquet(path, compression=compression, index=False)

    def read_parquet(self, path: Path) -> "pd.DataFrame":
        return self._pd.read_parquet(path)

    def to_feather(self, df: "pd.DataFrame", path: Path) -> None:
        df.to_feather(path)

    def read_feather(self, path: Path) -> "pd.DataFrame":
        return self._pd.read_feather(path)


class PolarsBackend(DataFrameBackend):
    """Polars DataFrame backend implementation with LazyFrame support."""

    def __init__(self, lazy_mode: bool = True):
        if not _check_polars():
            raise ImportError("polars is required for PolarsBackend")
        import polars as pl
        self._pl = pl
        self._lazy_mode = lazy_mode

    @property
    def name(self) -> str:
        return "polars"

    def from_dict(self, data: dict[str, list]) -> "pl.DataFrame":
        return self._pl.DataFrame(data)

    def from_records(self, records: list[dict]) -> "pl.DataFrame":
        if not records:
            return self._pl.DataFrame()
        return self._pl.DataFrame(records)

    def empty(self, schema: Optional[dict[str, type]] = None) -> "pl.DataFrame":
        if schema:
            pl_schema = {}
            type_map = {
                int: self._pl.Int64,
                float: self._pl.Float64,
                str: self._pl.Utf8,
                bool: self._pl.Boolean,
            }
            for k, v in schema.items():
                pl_schema[k] = type_map.get(v, self._pl.Utf8)
            return self._pl.DataFrame(schema=pl_schema)
        return self._pl.DataFrame()

    def concat(self, dfs: list["pl.DataFrame"], ignore_index: bool = True) -> "pl.DataFrame":
        valid_dfs = [df for df in dfs if df is not None and len(df) > 0]
        if not valid_dfs:
            return self._pl.DataFrame()
        return self._pl.concat(valid_dfs)

    def filter(self, df: "pl.DataFrame", column: str, op: str, value: Any) -> "pl.DataFrame":
        if column not in df.columns:
            return df
        col = self._pl.col(column)
        if op == "==":
            return df.filter(col == value)
        elif op == "!=":
            return df.filter(col != value)
        elif op == ">":
            return df.filter(col > value)
        elif op == ">=":
            return df.filter(col >= value)
        elif op == "<":
            return df.filter(col < value)
        elif op == "<=":
            return df.filter(col <= value)
        elif op == "in":
            return df.filter(col.is_in(value))
        elif op == "contains":
            return df.filter(col.cast(self._pl.Utf8).str.contains(str(value)))
        else:
            raise ValueError(f"Unknown operator: {op}")

    def groupby_agg(self, df: "pl.DataFrame", by: list[str], aggs: dict[str, str]) -> "pl.DataFrame":
        agg_exprs = []
        for col, agg_func in aggs.items():
            expr = self._pl.col(col)
            if agg_func == "sum":
                agg_exprs.append(expr.sum())
            elif agg_func == "mean":
                agg_exprs.append(expr.mean())
            elif agg_func == "count":
                agg_exprs.append(expr.count())
            elif agg_func == "min":
                agg_exprs.append(expr.min())
            elif agg_func == "max":
                agg_exprs.append(expr.max())
            elif agg_func == "first":
                agg_exprs.append(expr.first())
            elif agg_func == "last":
                agg_exprs.append(expr.last())
        return df.group_by(by).agg(agg_exprs)

    def sort(self, df: "pl.DataFrame", by: list[str], descending: bool = False) -> "pl.DataFrame":
        return df.sort(by, descending=descending)

    def select(self, df: "pl.DataFrame", columns: list[str]) -> "pl.DataFrame":
        existing = [c for c in columns if c in df.columns]
        return df.select(existing)

    def to_pandas(self, df: "pl.DataFrame") -> "pd.DataFrame":
        return df.to_pandas()

    def to_polars(self, df: Union["pl.DataFrame", "pd.DataFrame"]) -> "pl.DataFrame":
        if isinstance(df, self._pl.DataFrame):
            return df
        # Assume it's pandas
        return self._pl.from_pandas(df)

    def is_empty(self, df: "pl.DataFrame") -> bool:
        return df is None or len(df) == 0

    def len(self, df: "pl.DataFrame") -> int:
        return len(df) if df is not None else 0

    def columns(self, df: "pl.DataFrame") -> list[str]:
        return df.columns if df is not None else []

    def iterrows(self, df: "pl.DataFrame"):
        if df is None or len(df) == 0:
            return
        for idx, row in enumerate(df.iter_rows(named=True)):
            yield idx, row

    def get_column(self, df: "pl.DataFrame", column: str) -> Any:
        if column in df.columns:
            return df[column]
        return None

    def unique(self, df: "pl.DataFrame", column: str) -> list:
        if column in df.columns:
            return df[column].drop_nulls().unique().to_list()
        return []

    def to_parquet(self, df: "pl.DataFrame", path: Path, compression: str = "zstd") -> None:
        df.write_parquet(path, compression=compression)

    def read_parquet(self, path: Path) -> "pl.DataFrame":
        return self._pl.read_parquet(path)

    def to_feather(self, df: "pl.DataFrame", path: Path) -> None:
        df.write_ipc(path)

    def read_feather(self, path: Path) -> "pl.DataFrame":
        return self._pl.read_ipc(path)

    # Polars-specific: LazyFrame support
    def to_lazy(self, df: "pl.DataFrame") -> "pl.LazyFrame":
        """Convert DataFrame to LazyFrame for query optimization."""
        return df.lazy()

    def scan_parquet(self, path: Path) -> "pl.LazyFrame":
        """Lazy-scan a Parquet file for memory-efficient processing."""
        return self._pl.scan_parquet(path)

    def collect(self, lf: "pl.LazyFrame") -> "pl.DataFrame":
        """Collect a LazyFrame into a DataFrame."""
        return lf.collect()


class PolarsLazyBackend(PolarsBackend):
    """
    Polars backend optimized for large tick data using LazyFrames.

    LazyFrames provide:
    - Query optimization (predicate pushdown, projection pushdown)
    - Memory efficiency (streaming execution)
    - Better performance for complex operations on large data
    """

    def __init__(self):
        super().__init__(lazy_mode=True)

    @property
    def name(self) -> str:
        return "polars_lazy"

    def scan_parquet_lazy(self, path: Path) -> "pl.LazyFrame":
        """
        Lazy-scan a Parquet file for memory-efficient processing.

        This is ideal for large tick data files that don't fit in memory.
        The query is only executed when .collect() is called.
        """
        return self._pl.scan_parquet(path)

    def filter_lazy(self, lf: "pl.LazyFrame", column: str, op: str, value: Any) -> "pl.LazyFrame":
        """Filter a LazyFrame (optimized with predicate pushdown)."""
        col = self._pl.col(column)
        if op == "==":
            return lf.filter(col == value)
        elif op == "!=":
            return lf.filter(col != value)
        elif op == ">":
            return lf.filter(col > value)
        elif op == ">=":
            return lf.filter(col >= value)
        elif op == "<":
            return lf.filter(col < value)
        elif op == "<=":
            return lf.filter(col <= value)
        elif op == "in":
            return lf.filter(col.is_in(value))
        else:
            raise ValueError(f"Unknown operator: {op}")

    def groupby_agg_lazy(
        self, lf: "pl.LazyFrame", by: list[str], aggs: dict[str, str]
    ) -> "pl.LazyFrame":
        """Group by and aggregate on LazyFrame (optimized)."""
        agg_exprs = []
        for col, agg_func in aggs.items():
            expr = self._pl.col(col)
            if agg_func == "sum":
                agg_exprs.append(expr.sum())
            elif agg_func == "mean":
                agg_exprs.append(expr.mean())
            elif agg_func == "count":
                agg_exprs.append(expr.count())
            elif agg_func == "min":
                agg_exprs.append(expr.min())
            elif agg_func == "max":
                agg_exprs.append(expr.max())
        return lf.group_by(by).agg(agg_exprs)


# Singleton backend instances
_pandas_backend: Optional[PandasBackend] = None
_polars_backend: Optional[PolarsBackend] = None
_polars_lazy_backend: Optional[PolarsLazyBackend] = None


def get_backend(use_polars: Optional[bool] = None, lazy: bool = False) -> DataFrameBackend:
    """
    Get the appropriate DataFrame backend.

    Args:
        use_polars: Override the global config to force polars (True) or pandas (False).
                   If None, uses the global configuration.
        lazy: If True and using Polars, return the lazy backend optimized for large data.

    Returns:
        DataFrameBackend instance (pandas or polars)
    """
    global _pandas_backend, _polars_backend, _polars_lazy_backend

    config = get_backend_config()

    # Determine which backend to use
    should_use_polars = use_polars if use_polars is not None else config.use_polars

    if should_use_polars and _check_polars():
        if lazy:
            if _polars_lazy_backend is None:
                _polars_lazy_backend = PolarsLazyBackend()
            return _polars_lazy_backend
        else:
            if _polars_backend is None:
                _polars_backend = PolarsBackend()
            return _polars_backend
    else:
        if should_use_polars and not _check_polars():
            logger.warning("Polars requested but not available, falling back to pandas")
        if _pandas_backend is None:
            _pandas_backend = PandasBackend()
        return _pandas_backend


def convert_dataframe(
    df: Any,
    to_backend: str,
    from_backend: Optional[str] = None,
) -> Any:
    """
    Convert a DataFrame between backends.

    Args:
        df: The DataFrame to convert
        to_backend: Target backend ('pandas' or 'polars')
        from_backend: Source backend (auto-detected if None)

    Returns:
        Converted DataFrame
    """
    if df is None:
        return None

    # Auto-detect source backend
    if from_backend is None:
        if _check_polars():
            import polars as pl
            if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
                from_backend = "polars"
        if from_backend is None and _check_pandas():
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                from_backend = "pandas"

    if from_backend is None:
        raise ValueError("Could not detect DataFrame backend")

    # Convert
    if from_backend == to_backend:
        return df

    if to_backend == "pandas":
        backend = get_backend(use_polars=False)
        if from_backend == "polars":
            import polars as pl
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            return df.to_pandas()
        return df

    elif to_backend == "polars":
        if not _check_polars():
            raise ImportError("polars is required for conversion to Polars")
        import polars as pl
        if from_backend == "pandas":
            return pl.from_pandas(df)
        return df

    else:
        raise ValueError(f"Unknown target backend: {to_backend}")


# ============================================================================
# Serialization Utilities
# ============================================================================

def save_dataframe(
    df: Any,
    path: Path,
    format: Optional[str] = None,
    compression: Optional[str] = None,
) -> None:
    """
    Save a DataFrame to disk in an efficient format.

    Args:
        df: DataFrame to save (pandas or polars)
        path: Output path
        format: Format to use ('parquet', 'feather', 'csv'). Auto-detected from extension if None.
        compression: Compression for parquet ('zstd', 'lz4', 'snappy', 'gzip')

    Recommended formats:
    - Parquet: Best for archival and cross-language compatibility
    - Feather/Arrow IPC: Best for Python-to-Python IPC (fastest read/write)
    """
    config = get_backend_config()
    format = format or config.cache_format
    compression = compression or config.compression

    # Auto-detect format from extension
    if format is None:
        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            format = "parquet"
        elif suffix in (".feather", ".arrow", ".ipc"):
            format = "feather"
        elif suffix == ".csv":
            format = "csv"
        else:
            format = "parquet"  # Default

    # Determine backend
    backend = None
    if _check_polars():
        import polars as pl
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            backend = get_backend(use_polars=True)
            if isinstance(df, pl.LazyFrame):
                df = df.collect()

    if backend is None:
        backend = get_backend(use_polars=False)
        # Convert to pandas if needed
        if _check_polars():
            import polars as pl
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()

    # Save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        backend.to_parquet(df, path, compression=compression)
    elif format == "feather":
        backend.to_feather(df, path)
    elif format == "csv":
        if hasattr(df, "to_csv"):
            df.to_csv(path, index=False)
        elif hasattr(df, "write_csv"):
            df.write_csv(path)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.debug(f"Saved DataFrame to {path} ({format})")


def load_dataframe(
    path: Path,
    format: Optional[str] = None,
    use_polars: Optional[bool] = None,
    lazy: bool = False,
) -> Any:
    """
    Load a DataFrame from disk.

    Args:
        path: Path to the file
        format: Format ('parquet', 'feather', 'csv'). Auto-detected if None.
        use_polars: Force polars (True) or pandas (False). None uses config.
        lazy: If True and using polars, return a LazyFrame for large files.

    Returns:
        DataFrame (pandas or polars depending on config)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Auto-detect format
    if format is None:
        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            format = "parquet"
        elif suffix in (".feather", ".arrow", ".ipc"):
            format = "feather"
        elif suffix == ".csv":
            format = "csv"
        else:
            format = "parquet"

    backend = get_backend(use_polars=use_polars, lazy=lazy)

    # Load
    if format == "parquet":
        if lazy and hasattr(backend, "scan_parquet"):
            return backend.scan_parquet(path)
        return backend.read_parquet(path)
    elif format == "feather":
        return backend.read_feather(path)
    elif format == "csv":
        if backend.name == "polars":
            import polars as pl
            if lazy:
                return pl.scan_csv(path)
            return pl.read_csv(path)
        else:
            import pandas as pd
            return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown format: {format}")


# ============================================================================
# Performance Benchmarking
# ============================================================================

def benchmark_backends(df_size: int = 100000) -> dict[str, float]:
    """
    Benchmark pandas vs polars performance for common operations.

    Args:
        df_size: Number of rows to use in benchmark

    Returns:
        Dictionary with timing results
    """
    import time

    results = {}

    # Generate test data
    import numpy as np
    np.random.seed(42)
    data = {
        "tick": list(range(df_size)),
        "steamid": [np.random.randint(1, 11) for _ in range(df_size)],
        "x": [np.random.uniform(0, 1000) for _ in range(df_size)],
        "y": [np.random.uniform(0, 1000) for _ in range(df_size)],
        "health": [np.random.randint(0, 101) for _ in range(df_size)],
    }

    # Benchmark pandas
    if _check_pandas():
        pandas_backend = get_backend(use_polars=False)
        df_pd = pandas_backend.from_dict(data)

        start = time.perf_counter()
        for _ in range(10):
            _ = pandas_backend.filter(df_pd, "health", ">", 50)
            _ = pandas_backend.groupby_agg(df_pd, ["steamid"], {"health": "mean"})
            _ = pandas_backend.sort(df_pd, ["tick"])
        results["pandas_ops_time"] = (time.perf_counter() - start) / 10

    # Benchmark polars
    if _check_polars():
        polars_backend = get_backend(use_polars=True)
        df_pl = polars_backend.from_dict(data)

        start = time.perf_counter()
        for _ in range(10):
            _ = polars_backend.filter(df_pl, "health", ">", 50)
            _ = polars_backend.groupby_agg(df_pl, ["steamid"], {"health": "mean"})
            _ = polars_backend.sort(df_pl, ["tick"])
        results["polars_ops_time"] = (time.perf_counter() - start) / 10

    if "pandas_ops_time" in results and "polars_ops_time" in results:
        results["speedup"] = results["pandas_ops_time"] / results["polars_ops_time"]

    return results
