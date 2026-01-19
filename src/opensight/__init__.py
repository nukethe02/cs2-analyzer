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

    # Constants
    elif name == "DemoSource":
        from opensight.constants import DemoSource
        return DemoSource
    elif name == "GameMode":
        from opensight.constants import GameMode
        return GameMode
    elif name == "DemoType":
        from opensight.constants import DemoType
        return DemoType
    elif name == "CS2_TICK_RATE":
        from opensight.constants import CS2_TICK_RATE
        return CS2_TICK_RATE

    # Detection
    elif name == "DemoMetadata":
        from opensight.detection import DemoMetadata
        return DemoMetadata
    elif name == "detect_demo_metadata":
        from opensight.detection import detect_demo_metadata
        return detect_demo_metadata

    # Export
    elif name == "export_demo":
        from opensight.export import export_demo
        return export_demo
    elif name == "export_json":
        from opensight.export import export_json
        return export_json
    elif name == "export_csv":
        from opensight.export import export_csv
        return export_csv
    elif name == "export_excel":
        from opensight.export import export_excel
        return export_excel

    # Sharecode
    elif name == "decode_sharecode":
        from opensight.sharecode import decode_sharecode
        return decode_sharecode
    elif name == "ShareCodeInfo":
        from opensight.sharecode import ShareCodeInfo
        return ShareCodeInfo

    # State Machine (Pro-Level Analytics)
    elif name == "StateMachine":
        from opensight.state_machine import StateMachine
        return StateMachine
    elif name == "analyze_state":
        from opensight.state_machine import analyze_state
        return analyze_state
    elif name == "get_kill_contexts":
        from opensight.state_machine import get_kill_contexts
        return get_kill_contexts
    elif name == "KillContext":
        from opensight.state_machine import KillContext
        return KillContext
    elif name == "PlayerContextStats":
        from opensight.state_machine import PlayerContextStats
        return PlayerContextStats
    elif name == "StateAnalysisResult":
        from opensight.state_machine import StateAnalysisResult
        return StateAnalysisResult
    elif name == "CrosshairAnalyzer":
        from opensight.state_machine import CrosshairAnalyzer
        return CrosshairAnalyzer

    raise AttributeError(f"module 'opensight' has no attribute '{name}'")

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
    "KillEvent",
    "DamageEvent",
    # Analytics
    "DemoAnalyzer",
    "analyze_demo",
    "MatchAnalysis",
    "PlayerMatchStats",
    # State Machine (Pro-Level Analytics)
    "StateMachine",
    "analyze_state",
    "get_kill_contexts",
    "KillContext",
    "PlayerContextStats",
    "StateAnalysisResult",
    "CrosshairAnalyzer",
    # Constants
    "DemoSource",
    "GameMode",
    "DemoType",
    "CS2_TICK_RATE",
    # Detection
    "DemoMetadata",
    "detect_demo_metadata",
    # Export
    "export_to_json",
    "export_to_excel",
    "export_to_html",
    "export_metrics_to_csv",
    "export_analysis",
]
