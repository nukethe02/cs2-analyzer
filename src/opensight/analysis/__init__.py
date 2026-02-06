"""
OpenSight Analysis - Core analytics and metrics calculation.

This module contains:
- analytics: Main analytics engine (HLTV rating, KAST, etc.)
- metrics: TTD, Crosshair Placement calculations
- metrics_optimized: Vectorized/optimized metric computations
- detection: Demo source and game mode detection
- rotation: CT rotation latency analysis
"""

from opensight.analysis.models import MatchAnalysis, PlayerMatchStats
from opensight.analysis.rotation import (
    CTRotationAnalyzer,
    PlayerRotationStats,
    RotationLatency,
    SiteContactEvent,
    TeamRotationStats,
    analyze_ct_rotations,
    get_rotation_summary,
)

__all__: list[str] = [
    "MatchAnalysis",
    "PlayerMatchStats",
    "CTRotationAnalyzer",
    "analyze_ct_rotations",
    "get_rotation_summary",
    "SiteContactEvent",
    "RotationLatency",
    "PlayerRotationStats",
    "TeamRotationStats",
]
