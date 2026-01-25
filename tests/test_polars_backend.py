"""
Tests for Polars Backend and DataFrame Operations

Tests cover:
- Backend abstraction (pandas and polars)
- DataFrame operations (filter, groupby, sort, etc.)
- Serialization (Parquet, Feather)
- Backend conversion
- Polars-optimized analytics
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

# Check if polars is available
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Check if pyarrow is available (needed for pandas parquet/feather)
try:
    import pyarrow as _  # noqa: F401

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


# ============================================================================
# Test Backend Module
# ============================================================================


class TestBackendAbstraction:
    """Test the DataFrame backend abstraction layer."""

    def test_pandas_backend_available(self):
        """Test that pandas backend is always available."""
        from opensight.infra.backend import PandasBackend, is_pandas_available

        assert is_pandas_available()
        backend = PandasBackend()
        assert backend.name == "pandas"

    def test_polars_backend_check(self):
        """Test Polars availability check."""
        from opensight.infra.backend import is_polars_available

        # Should match our local check
        assert is_polars_available() == POLARS_AVAILABLE

    def test_pandas_backend_from_dict(self):
        """Test creating DataFrame from dict using pandas backend."""
        from opensight.infra.backend import PandasBackend

        backend = PandasBackend()
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        df = backend.from_dict(data)

        assert backend.len(df) == 3
        assert "a" in backend.columns(df)
        assert "b" in backend.columns(df)

    def test_pandas_backend_filter(self):
        """Test filtering using pandas backend."""
        from opensight.infra.backend import PandasBackend

        backend = PandasBackend()
        data = {"value": [1, 5, 10, 15, 20], "name": ["a", "b", "c", "d", "e"]}
        df = backend.from_dict(data)

        # Filter value > 10
        filtered = backend.filter(df, "value", ">", 10)
        assert backend.len(filtered) == 2

        # Filter value == 5
        filtered = backend.filter(df, "value", "==", 5)
        assert backend.len(filtered) == 1

        # Filter in list
        filtered = backend.filter(df, "value", "in", [1, 10, 20])
        assert backend.len(filtered) == 3

    def test_pandas_backend_groupby(self):
        """Test groupby aggregation using pandas backend."""
        from opensight.infra.backend import PandasBackend

        backend = PandasBackend()
        data = {
            "player": [1, 1, 2, 2, 2],
            "damage": [100, 50, 75, 25, 100],
        }
        df = backend.from_dict(data)

        grouped = backend.groupby_agg(df, ["player"], {"damage": "sum"})
        assert backend.len(grouped) == 2

    def test_pandas_backend_sort(self):
        """Test sorting using pandas backend."""
        from opensight.infra.backend import PandasBackend

        backend = PandasBackend()
        data = {"tick": [300, 100, 200]}
        df = backend.from_dict(data)

        sorted_df = backend.sort(df, ["tick"])
        # Check first row is lowest tick
        for _idx, row in backend.iterrows(sorted_df):
            assert row["tick"] == 100
            break

    def test_pandas_backend_concat(self):
        """Test concatenation using pandas backend."""
        from opensight.infra.backend import PandasBackend

        backend = PandasBackend()
        df1 = backend.from_dict({"a": [1, 2]})
        df2 = backend.from_dict({"a": [3, 4]})
        df3 = backend.empty()  # Empty should be handled

        result = backend.concat([df1, df2, df3])
        assert backend.len(result) == 4

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_polars_backend_from_dict(self):
        """Test creating DataFrame from dict using Polars backend."""
        from opensight.infra.backend import PolarsBackend

        backend = PolarsBackend()
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        df = backend.from_dict(data)

        assert backend.name == "polars"
        assert backend.len(df) == 3
        assert "a" in backend.columns(df)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_polars_backend_filter(self):
        """Test filtering using Polars backend."""
        from opensight.infra.backend import PolarsBackend

        backend = PolarsBackend()
        data = {"value": [1, 5, 10, 15, 20], "name": ["a", "b", "c", "d", "e"]}
        df = backend.from_dict(data)

        # Filter value > 10
        filtered = backend.filter(df, "value", ">", 10)
        assert backend.len(filtered) == 2

        # Filter contains
        filtered = backend.filter(df, "name", "contains", "b")
        assert backend.len(filtered) == 1

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_polars_backend_groupby(self):
        """Test groupby aggregation using Polars backend."""
        from opensight.infra.backend import PolarsBackend

        backend = PolarsBackend()
        data = {
            "player": [1, 1, 2, 2, 2],
            "damage": [100, 50, 75, 25, 100],
        }
        df = backend.from_dict(data)

        grouped = backend.groupby_agg(df, ["player"], {"damage": "sum"})
        assert backend.len(grouped) == 2


class TestBackendConversion:
    """Test converting DataFrames between backends."""

    def test_pandas_to_pandas(self):
        """Test pandas DataFrame stays as pandas."""
        from opensight.infra.backend import convert_dataframe

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = convert_dataframe(df, "pandas")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_pandas_to_polars(self):
        """Test converting pandas to Polars."""
        from opensight.infra.backend import convert_dataframe

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = convert_dataframe(df, "polars")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    @pytest.mark.skipif(
        not PYARROW_AVAILABLE, reason="PyArrow not installed (needed for polars->pandas)"
    )
    def test_polars_to_pandas(self):
        """Test converting Polars to pandas."""
        from opensight.infra.backend import convert_dataframe

        df = pl.DataFrame({"a": [1, 2, 3]})
        result = convert_dataframe(df, "pandas")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestGetBackend:
    """Test backend selection logic."""

    def test_get_backend_default(self):
        """Test default backend selection."""
        from opensight.infra.backend import get_backend

        backend = get_backend()
        # Default should be pandas unless configured otherwise
        assert backend is not None
        assert backend.name in ["pandas", "polars"]

    def test_get_backend_force_pandas(self):
        """Test forcing pandas backend."""
        from opensight.infra.backend import get_backend

        backend = get_backend(use_polars=False)
        assert backend.name == "pandas"

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_get_backend_force_polars(self):
        """Test forcing Polars backend."""
        from opensight.infra.backend import get_backend

        backend = get_backend(use_polars=True)
        assert backend.name == "polars"


# ============================================================================
# Test Serialization
# ============================================================================


class TestSerialization:
    """Test DataFrame serialization (Parquet, Feather)."""

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
    def test_save_load_parquet_pandas(self):
        """Test saving/loading Parquet with pandas."""
        from opensight.infra.backend import load_dataframe, save_dataframe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

            save_dataframe(df, path, format="parquet")
            assert path.exists()

            loaded = load_dataframe(path, use_polars=False)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) == 3
            assert list(loaded["a"]) == [1, 2, 3]

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
    def test_save_load_feather_pandas(self):
        """Test saving/loading Feather with pandas."""
        from opensight.infra.backend import load_dataframe, save_dataframe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.feather"
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

            save_dataframe(df, path, format="feather")
            assert path.exists()

            loaded = load_dataframe(path, use_polars=False)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) == 3

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_save_load_parquet_polars(self):
        """Test saving/loading Parquet with Polars."""
        from opensight.infra.backend import load_dataframe, save_dataframe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

            save_dataframe(df, path, format="parquet")
            assert path.exists()

            loaded = load_dataframe(path, use_polars=True)
            assert isinstance(loaded, pl.DataFrame)
            assert len(loaded) == 3

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_lazy_load_parquet(self):
        """Test lazy loading Parquet with Polars."""
        from opensight.infra.backend import load_dataframe, save_dataframe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df = pl.DataFrame(
                {
                    "tick": list(range(1000)),
                    "value": [i * 0.5 for i in range(1000)],
                }
            )

            save_dataframe(df, path, format="parquet")

            # Load lazily
            lf = load_dataframe(path, use_polars=True, lazy=True)
            assert isinstance(lf, pl.LazyFrame)

            # Collect to verify
            result = lf.collect()
            assert len(result) == 1000


# ============================================================================
# Test Polars Operations
# ============================================================================


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
class TestPolarsOps:
    """Test Polars-optimized operations."""

    def test_filter_by_steamid(self):
        """Test steamid filtering with type coercion."""
        from opensight.infra.polars_ops import PolarsOps

        df = pl.DataFrame(
            {
                "attacker_steamid": [123, 456, 123, 789],
                "damage": [50, 100, 75, 25],
            }
        )

        filtered = PolarsOps.filter_by_steamid(df, "attacker_steamid", 123)
        assert len(filtered) == 2

    def test_group_kills_by_round(self):
        """Test multi-kill detection."""
        from opensight.infra.polars_ops import PolarsOps

        df = pl.DataFrame(
            {
                "attacker_steamid": [1, 1, 1, 2, 2],
                "round_num": [1, 1, 1, 1, 2],
                "tick": [100, 200, 300, 400, 500],
            }
        )

        result = PolarsOps.group_kills_by_round(df, "attacker_steamid", "round_num")

        # Player 1 has 3 kills in round 1
        p1_r1 = result.filter((pl.col("attacker_steamid") == 1) & (pl.col("round_num") == 1))
        assert p1_r1["kill_count"][0] == 3

    def test_compute_damage_stats(self):
        """Test damage aggregation."""
        from opensight.infra.polars_ops import PolarsOps

        df = pl.DataFrame(
            {
                "attacker_steamid": [1, 1, 2, 2],
                "dmg_health": [100, 50, 75, 25],
            }
        )

        result = PolarsOps.compute_damage_stats(df, "attacker_steamid", "dmg_health")

        p1_dmg = result.filter(pl.col("attacker_steamid") == 1)
        assert p1_dmg["total_damage"][0] == 150

    def test_find_first_kill_per_round(self):
        """Test opening duel detection."""
        from opensight.infra.polars_ops import PolarsOps

        df = pl.DataFrame(
            {
                "round_num": [1, 1, 1, 2, 2],
                "tick": [200, 100, 300, 150, 250],
                "attacker": ["a", "b", "c", "d", "e"],
            }
        )

        result = PolarsOps.find_first_kill_per_round(df, "round_num")

        # First kill in round 1 should be at tick 100 (player "b")
        r1_first = result.filter(pl.col("round_num") == 1)
        assert r1_first["tick"][0] == 100
        assert r1_first["attacker"][0] == "b"


# ============================================================================
# Test Polars Analyzer
# ============================================================================


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
class TestPolarsAnalyzer:
    """Test the Polars-optimized analyzer."""

    @pytest.fixture
    def mock_demo_data(self):
        """Create mock DemoData for testing."""

        # Create a minimal mock DemoData
        @dataclass
        class MockDemoData:
            kills_df: pd.DataFrame = field(default_factory=pd.DataFrame)
            damages_df: pd.DataFrame = field(default_factory=pd.DataFrame)
            ticks_df: pd.DataFrame | None = None
            player_names: dict = field(default_factory=dict)

        data = MockDemoData()
        data.kills_df = pd.DataFrame(
            {
                "attacker_steamid": [1, 1, 2, 2, 1],
                "victim_steamid": [2, 3, 1, 3, 2],
                "round_num": [1, 1, 1, 2, 2],
                "tick": [100, 200, 300, 400, 500],
                "headshot": [True, False, True, False, True],
            }
        )
        data.damages_df = pd.DataFrame(
            {
                "attacker_steamid": [1, 1, 2, 2, 1],
                "victim_steamid": [2, 3, 1, 3, 2],
                "dmg_health": [50, 100, 75, 25, 50],
                "tick": [90, 190, 290, 390, 490],
            }
        )
        data.player_names = {1: "Player1", 2: "Player2", 3: "Player3"}

        return data

    def test_polars_analyzer_init(self, mock_demo_data):
        """Test PolarsAnalyzer initialization."""
        from opensight.infra.polars_ops import PolarsAnalyzer

        analyzer = PolarsAnalyzer(mock_demo_data)
        assert analyzer._kills_df is not None
        assert isinstance(analyzer._kills_df, pl.DataFrame)

    def test_compute_player_stats(self, mock_demo_data):
        """Test basic player stats computation."""
        from opensight.infra.polars_ops import PolarsAnalyzer

        analyzer = PolarsAnalyzer(mock_demo_data)
        stats = analyzer.compute_player_stats()

        assert 1 in stats
        assert 2 in stats
        assert stats[1].kills == 3  # Player 1 has 3 kills
        assert stats[2].kills == 2  # Player 2 has 2 kills
        assert stats[1].headshots == 2  # Player 1 has 2 headshots

    def test_compute_opening_duels(self, mock_demo_data):
        """Test opening duel detection."""
        from opensight.infra.polars_ops import PolarsAnalyzer

        analyzer = PolarsAnalyzer(mock_demo_data)
        duels = analyzer.compute_opening_duels()

        # Player 1 got first kill in round 1 (tick 100)
        assert duels[1]["wins"] >= 1


# ============================================================================
# Test Configuration
# ============================================================================


class TestBackendConfig:
    """Test backend configuration."""

    def test_backend_config_defaults(self):
        """Test default backend configuration."""
        from opensight.infra.backend import BackendConfig

        config = BackendConfig()
        assert not config.use_polars
        assert config.lazy_mode
        assert config.cache_format == "parquet"
        assert config.compression == "zstd"

    def test_set_backend_config(self):
        """Test setting global backend config."""
        from opensight.infra.backend import BackendConfig, get_backend_config, set_backend_config

        config = BackendConfig(use_polars=True, cache_format="feather")
        set_backend_config(config)

        retrieved = get_backend_config()
        assert retrieved.use_polars
        assert retrieved.cache_format == "feather"

        # Reset to defaults
        set_backend_config(BackendConfig())

    def test_config_dataclass_in_opensight_config(self):
        """Test BackendConfig is included in OpenSightConfig."""
        from opensight.core.config import OpenSightConfig

        config = OpenSightConfig()
        assert hasattr(config, "backend")
        assert not config.backend.use_polars


# ============================================================================
# Test Lazy Imports
# ============================================================================


class TestLazyImports:
    """Test that backend is lazily imported."""

    def test_lazy_import_get_backend(self):
        """Test get_backend is accessible via lazy import."""
        from opensight import get_backend

        backend = get_backend()
        assert backend is not None

    def test_lazy_import_save_load(self):
        """Test serialization functions are accessible."""
        from opensight import load_dataframe, save_dataframe

        assert callable(save_dataframe)
        assert callable(load_dataframe)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_lazy_import_polars_backend(self):
        """Test PolarsBackend is accessible."""
        from opensight import PolarsBackend

        backend = PolarsBackend()
        assert backend.name == "polars"


# ============================================================================
# Test Performance (Optional - for local benchmarking)
# ============================================================================


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
class TestPerformance:
    """Performance comparison tests (may be slow, skip in CI)."""

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_benchmark_backends(self):
        """Benchmark pandas vs polars performance."""
        from opensight.infra.backend import benchmark_backends

        results = benchmark_backends(df_size=100000)

        print("\nBenchmark results:")
        print(f"  Pandas: {results.get('pandas_ops_time', 'N/A'):.4f}s")
        print(f"  Polars: {results.get('polars_ops_time', 'N/A'):.4f}s")
        print(f"  Speedup: {results.get('speedup', 'N/A'):.2f}x")

        # Polars should be faster for large datasets
        if "speedup" in results:
            assert results["speedup"] > 0.5  # At least not drastically slower
