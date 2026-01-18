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
    elif name == "parse_demo":
        from opensight.parser import parse_demo
        return parse_demo
    elif name == "KillEvent":
        from opensight.parser import KillEvent
        return KillEvent
    elif name == "DamageEvent":
        from opensight.parser import DamageEvent
        return DamageEvent

    # Analytics
    elif name == "DemoAnalyzer":
        from opensight.analytics import DemoAnalyzer
        return DemoAnalyzer
    elif name == "analyze_demo":
        from opensight.analytics import analyze_demo
        return analyze_demo
    elif name == "MatchAnalysis":
        from opensight.analytics import MatchAnalysis
        return MatchAnalysis
    elif name == "PlayerMatchStats":
        from opensight.analytics import PlayerMatchStats
        return PlayerMatchStats

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

    raise AttributeError(f"module 'opensight' has no attribute '{name}'")


__all__ = [
    # Version
    "__version__",
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
    # Constants
    "DemoSource",
    "GameMode",
    "DemoType",
    "CS2_TICK_RATE",
    # Detection
    "DemoMetadata",
    "detect_demo_metadata",
    # Export
    "export_demo",
    "export_json",
    "export_csv",
    "export_excel",
    # Sharecode
    "decode_sharecode",
    "ShareCodeInfo",
]
