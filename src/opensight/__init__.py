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


def __getattr__(name):
    """Lazy import for heavy dependencies."""
    # Parser
    if name == "DemoParser":
        from opensight.parser import DemoParser
        return DemoParser
    elif name == "DemoData":
        from opensight.parser import DemoData
        return DemoData
    elif name == "calculate_ttd":
        from opensight.metrics import calculate_ttd
        return calculate_ttd
    elif name == "calculate_crosshair_placement":
        from opensight.metrics import calculate_crosshair_placement
        return calculate_crosshair_placement
    elif name == "ReplayWatcher":
        from opensight.watcher import ReplayWatcher
        return ReplayWatcher
    elif name == "EconomyAnalyzer":
        from opensight.economy import EconomyAnalyzer
        return EconomyAnalyzer
    elif name == "analyze_economy":
        from opensight.economy import analyze_economy
        return analyze_economy
    elif name == "BuyType":
        from opensight.economy import BuyType
        return BuyType
    elif name == "CombatAnalyzer":
        from opensight.combat import CombatAnalyzer
        return CombatAnalyzer
    elif name == "analyze_combat":
        from opensight.combat import analyze_combat
        return analyze_combat
    elif name == "UtilityAnalyzer":
        from opensight.utility import UtilityAnalyzer
        return UtilityAnalyzer
    elif name == "analyze_utility":
        from opensight.utility import analyze_utility
        return analyze_utility
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
    "calculate_ttd",
    "calculate_crosshair_placement",
    "ReplayWatcher",
    "EconomyAnalyzer",
    "analyze_economy",
    "BuyType",
    "CombatAnalyzer",
    "analyze_combat",
    "UtilityAnalyzer",
    "analyze_utility",
]
