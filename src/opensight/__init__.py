"""
OpenSight - Professional CS2 Demo Analyzer

A universal Counter-Strike 2 demo analysis tool that works with any demo from any source.
Provides industry-standard metrics (HLTV 2.0 Rating, KAST%, ADR) and multiple output formats.

Usage:
    from opensight import parse_demo, analyze_demo

    demo = parse_demo("match.dem")
    analysis = analyze_demo(demo)

    for player in analysis.get_leaderboard():
        print(f"{player.name}: {player.hltv_rating:.2f}")
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


__all__ = [
    # Version
    "__version__",
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
