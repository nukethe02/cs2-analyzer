"""
OpenSight - Professional CS2 Demo Analyzer

A locally-operated analytics system for Counter-Strike 2 that provides
professional-grade metrics without cloud dependencies.

Features:
- Parse CS2 demo files (.dem) locally
- Calculate professional-grade performance metrics
- Monitor replay folders for automatic analysis
- Export to multiple formats (JSON, CSV, Excel, HTML)
"""

__version__ = "0.2.0"
__author__ = "OpenSight Contributors"

# Core functionality
from opensight.sharecode import decode_sharecode, encode_sharecode, ShareCodeInfo
from opensight.parser import DemoParser, DemoData, parse_demo, PlayerState
from opensight.watcher import ReplayWatcher, DemoFileEvent, watch_replays

# Metrics
from opensight.metrics import (
    # Core metrics
    calculate_ttd,
    calculate_crosshair_placement,
    calculate_engagement_metrics,
    # Extended metrics
    calculate_economy_metrics,
    calculate_utility_metrics,
    calculate_positioning_metrics,
    calculate_trade_metrics,
    calculate_opening_metrics,
    calculate_comprehensive_metrics,
    # Result dataclasses
    TTDResult,
    CrosshairPlacementResult,
    EngagementMetrics,
    EconomyMetrics,
    UtilityMetrics,
    PositioningMetrics,
    TradeMetrics,
    OpeningDuelMetrics,
    ComprehensivePlayerMetrics,
)

# Configuration
from opensight.config import (
    OpenSightConfig,
    load_config,
    get_config,
    set_config,
)

# Export
from opensight.export import (
    export_to_json,
    export_to_excel,
    export_to_html,
    export_metrics_to_csv,
    export_analysis,
)

__all__ = [
    # Version
    "__version__",
    # Share code
    "decode_sharecode",
    "encode_sharecode",
    "ShareCodeInfo",
    # Parser
    "DemoParser",
    "DemoData",
    "parse_demo",
    "PlayerState",
    # Watcher
    "ReplayWatcher",
    "DemoFileEvent",
    "watch_replays",
    # Core metrics
    "calculate_ttd",
    "calculate_crosshair_placement",
    "calculate_engagement_metrics",
    # Extended metrics
    "calculate_economy_metrics",
    "calculate_utility_metrics",
    "calculate_positioning_metrics",
    "calculate_trade_metrics",
    "calculate_opening_metrics",
    "calculate_comprehensive_metrics",
    # Result types
    "TTDResult",
    "CrosshairPlacementResult",
    "EngagementMetrics",
    "EconomyMetrics",
    "UtilityMetrics",
    "PositioningMetrics",
    "TradeMetrics",
    "OpeningDuelMetrics",
    "ComprehensivePlayerMetrics",
    # Configuration
    "OpenSightConfig",
    "load_config",
    "get_config",
    "set_config",
    # Export
    "export_to_json",
    "export_to_excel",
    "export_to_html",
    "export_metrics_to_csv",
    "export_analysis",
]
