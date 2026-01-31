"""
Configuration Management for OpenSight

Provides configuration loading from multiple sources:
- Default values
- Configuration files (YAML, TOML, JSON)
- Environment variables
- Command line arguments

Configuration precedence (highest to lowest):
1. Command line arguments
2. Environment variables (OPENSIGHT_*)
3. Configuration file
4. Default values
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class BackendConfig:
    """Configuration for DataFrame backend (pandas vs polars)."""

    # Use Polars for DataFrame operations (faster, better memory efficiency)
    use_polars: bool = False

    # Use LazyFrames for large tick data (memory efficient, query optimization)
    lazy_mode: bool = True

    # Cache format for intermediate results ('parquet', 'feather', 'pickle')
    # Parquet: Best for archival, cross-language compatibility
    # Feather: Fastest Python-to-Python IPC
    cache_format: str = "parquet"

    # Compression for Parquet files ('zstd', 'lz4', 'snappy', 'gzip', None)
    # zstd: Best balance of compression ratio and speed
    compression: str = "zstd"


@dataclass
class ParserConfig:
    """Configuration for demo parsing."""

    tick_fields: list[str] = field(
        default_factory=lambda: [
            "tick",
            "steamid",
            "X",
            "Y",
            "Z",
            "pitch",
            "yaw",
            "health",
            "armor_value",
            "team_num",
            "active_weapon_name",
        ]
    )
    parse_events: list[str] = field(
        default_factory=lambda: [
            "player_death",
            "player_hurt",
            "weapon_fire",
            "round_start",
            "round_end",
        ]
    )
    fallback_to_minimal: bool = True
    cache_parsed_demos: bool = True
    cache_directory: str | None = None

    # Use Polars for DataFrame storage in DemoData (otherwise pandas)
    use_polars: bool = False


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""

    # TTD settings
    ttd_max_lookback_ms: float = 2000.0
    ttd_min_visibility_ticks: int = 4

    # Crosshair placement settings (Leetify-style)
    cp_sample_interval_ticks: int = 16
    cp_min_distance: float = 100.0
    cp_max_distance: float = 2000.0
    cp_eye_height: float = 64.0
    # Pre-engagement window: measure CP this many ms BEFORE first damage
    # to capture pre-aim quality, not recoil/flick adjustments
    cp_pre_engagement_ms: float = 200.0
    # Max shots in engagement to count as "clean" placement (filters sprays)
    cp_max_shots_for_valid: int = 3

    # Economy settings
    eco_threshold: int = 1500
    force_buy_threshold: int = 3500

    # Trade settings
    trade_window_ms: float = 2000.0

    # Rating calculation weights
    rating_kd_weight: float = 0.30
    rating_cp_weight: float = 0.20
    rating_opening_weight: float = 0.10
    rating_trade_weight: float = 0.05


@dataclass
class WatcherConfig:
    """Configuration for replay watcher."""

    min_file_size_bytes: int = 1024 * 1024  # 1MB
    debounce_seconds: float = 2.0
    recursive: bool = False
    auto_analyze: bool = True
    custom_replays_folder: str | None = None


@dataclass
class ExportConfig:
    """Configuration for data export."""

    default_format: str = "json"
    json_indent: int = 2
    csv_delimiter: str = ","
    include_raw_values: bool = False
    timestamp_format: str = "%Y-%m-%d_%H-%M-%S"


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None
    file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5


@dataclass
class OpenSightConfig:
    """Main configuration container."""

    parser: ParserConfig = field(default_factory=ParserConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)

    # Version of the config format
    config_version: str = "1.0"


# ============================================================================
# Configuration Loading
# ============================================================================


def get_default_config_paths() -> list[Path]:
    """Get the default paths to search for configuration files."""
    paths = []

    # Current directory
    paths.append(Path.cwd() / "opensight.yaml")
    paths.append(Path.cwd() / "opensight.toml")
    paths.append(Path.cwd() / "opensight.json")
    paths.append(Path.cwd() / ".opensight.yaml")

    # User home directory
    home = Path.home()
    paths.append(home / ".config" / "opensight" / "config.yaml")
    paths.append(home / ".config" / "opensight" / "config.toml")
    paths.append(home / ".opensight.yaml")

    # XDG config directory
    xdg_config = os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))
    paths.append(Path(xdg_config) / "opensight" / "config.yaml")

    return paths


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed, cannot load YAML config")
        return {}


def load_toml_config(path: Path) -> dict[str, Any]:
    """Load configuration from a TOML file."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            logger.warning("tomllib/tomli not available, cannot load TOML config")
            return {}

    with open(path, "rb") as f:
        return tomllib.load(f)


def load_json_config(path: Path) -> dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_config_file(path: Path) -> dict[str, Any]:
    """Load configuration from a file, detecting format from extension."""
    if not path.exists():
        return {}

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return load_yaml_config(path)
    elif suffix == ".toml":
        return load_toml_config(path)
    elif suffix == ".json":
        return load_json_config(path)
    else:
        logger.warning(f"Unknown config file format: {suffix}")
        return {}


def load_env_config() -> dict[str, Any]:
    """Load configuration from environment variables."""
    config: dict[str, Any] = {}

    env_mappings = {
        "OPENSIGHT_LOG_LEVEL": ("logging", "level"),
        "OPENSIGHT_LOG_FILE": ("logging", "file"),
        "OPENSIGHT_REPLAYS_FOLDER": ("watcher", "custom_replays_folder"),
        "OPENSIGHT_AUTO_ANALYZE": ("watcher", "auto_analyze"),
        "OPENSIGHT_EXPORT_FORMAT": ("export", "default_format"),
        "OPENSIGHT_TTD_LOOKBACK_MS": ("metrics", "ttd_max_lookback_ms"),
        "OPENSIGHT_TRADE_WINDOW_MS": ("metrics", "trade_window_ms"),
        # Backend configuration
        "OPENSIGHT_USE_POLARS": ("backend", "use_polars"),
        "OPENSIGHT_LAZY_MODE": ("backend", "lazy_mode"),
        "OPENSIGHT_CACHE_FORMAT": ("backend", "cache_format"),
        "OPENSIGHT_COMPRESSION": ("backend", "compression"),
        # Parser backend configuration
        "OPENSIGHT_PARSER_USE_POLARS": ("parser", "use_polars"),
    }

    for env_var, (section, key) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in config:
                config[section] = {}

            # Type conversion
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            config[section][key] = value

    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def dict_to_config(data: dict[str, Any]) -> OpenSightConfig:
    """Convert a dictionary to OpenSightConfig."""
    config = OpenSightConfig()

    if "parser" in data:
        for key, value in data["parser"].items():
            if hasattr(config.parser, key):
                setattr(config.parser, key, value)

    if "metrics" in data:
        for key, value in data["metrics"].items():
            if hasattr(config.metrics, key):
                setattr(config.metrics, key, value)

    if "watcher" in data:
        for key, value in data["watcher"].items():
            if hasattr(config.watcher, key):
                setattr(config.watcher, key, value)

    if "export" in data:
        for key, value in data["export"].items():
            if hasattr(config.export, key):
                setattr(config.export, key, value)

    if "logging" in data:
        for key, value in data["logging"].items():
            if hasattr(config.logging, key):
                setattr(config.logging, key, value)

    if "backend" in data:
        for key, value in data["backend"].items():
            if hasattr(config.backend, key):
                setattr(config.backend, key, value)

    return config


def load_config(config_file: Path | None = None, include_env: bool = True) -> OpenSightConfig:
    """
    Load configuration from all sources.

    Args:
        config_file: Explicit path to a config file (optional)
        include_env: Whether to include environment variables

    Returns:
        Merged OpenSightConfig
    """
    config_data: dict[str, Any] = {}

    # Try to find and load a config file
    if config_file:
        config_data = load_config_file(config_file)
        logger.info(f"Loaded config from: {config_file}")
    else:
        for path in get_default_config_paths():
            if path.exists():
                config_data = load_config_file(path)
                logger.info(f"Loaded config from: {path}")
                break

    # Merge environment variables
    if include_env:
        env_config = load_env_config()
        config_data = merge_configs(config_data, env_config)

    return dict_to_config(config_data)


# ============================================================================
# Configuration Saving
# ============================================================================


def save_config(config: OpenSightConfig, path: Path) -> None:
    """
    Save configuration to a file.

    Args:
        config: Configuration to save
        path: Path to save to (format detected from extension)
    """
    data = config_to_dict(config)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml

            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML required to save YAML config")

    elif suffix == ".toml":
        try:
            import toml

            with open(path, "w") as f:
                toml.dump(data, f)
        except ImportError:
            raise ImportError("toml package required to save TOML config")

    elif suffix == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    else:
        raise ValueError(f"Unknown config format: {suffix}")

    logger.info(f"Saved config to: {path}")


def config_to_dict(config: OpenSightConfig) -> dict[str, Any]:
    """Convert OpenSightConfig to a dictionary."""
    from dataclasses import asdict

    return asdict(config)


# ============================================================================
# Global Configuration
# ============================================================================

_global_config: OpenSightConfig | None = None


def get_config() -> OpenSightConfig:
    """Get the global configuration, loading it if necessary."""
    global _global_config

    if _global_config is None:
        _global_config = load_config()

    return _global_config


def set_config(config: OpenSightConfig) -> None:
    """Set the global configuration."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _global_config
    _global_config = None


# ============================================================================
# Configuration Templates
# ============================================================================

DEFAULT_CONFIG_YAML = """# OpenSight Configuration
# https://github.com/yourusername/opensight

# Parser settings
parser:
  cache_parsed_demos: true
  fallback_to_minimal: true
  # use_polars: false  # Use Polars for DataFrame storage in DemoData

# Metrics calculation settings
metrics:
  ttd_max_lookback_ms: 2000.0
  cp_sample_interval_ticks: 16
  trade_window_ms: 2000.0

# Replay watcher settings
watcher:
  min_file_size_bytes: 1048576  # 1MB
  debounce_seconds: 2.0
  recursive: false
  auto_analyze: true
  # custom_replays_folder: /path/to/replays

# Export settings
export:
  default_format: json
  json_indent: 2
  csv_delimiter: ","
  include_raw_values: false

# Logging settings
logging:
  level: INFO
  # file: /path/to/opensight.log

# DataFrame backend settings
# Polars offers significant performance benefits for large tick data:
# - LazyFrames reduce memory usage
# - Better query optimization
# - Parallel execution
backend:
  use_polars: false  # Set to true to use Polars instead of pandas
  lazy_mode: true    # Use LazyFrames for large tick data (Polars only)
  cache_format: parquet  # parquet, feather, or pickle
  compression: zstd      # zstd, lz4, snappy, gzip (for parquet)
"""


def generate_default_config(path: Path) -> None:
    """Generate a default configuration file."""
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        path.write_text(DEFAULT_CONFIG_YAML)
    else:
        config = OpenSightConfig()
        save_config(config, path)

    logger.info(f"Generated default config at: {path}")
