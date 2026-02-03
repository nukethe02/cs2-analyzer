"""
OpenSight Core - Foundation modules for demo parsing and analysis.

This module contains the fundamental components:
- constants: Game constants, enums, and configuration values
- config: Application configuration management
- utils: General utility functions
- parser: Demo file parsing using demoparser2
- schemas: Data contracts for module boundaries
"""

from opensight.core.constants import (
    ACTIVE_DUTY_MAPS,
    CLUTCH_TIME_THRESHOLD,
    COMPETITIVE_MAPS,
    CS2_TICK_RATE,
    FILENAME_PATTERNS,
    FLASH_ASSIST_MIN_DURATION,
    HLTV_RATING_COEFFICIENTS,
    IMPACT_COEFFICIENTS,
    RESERVE_MAPS,
    SOURCE_IDENTIFIERS,
    TICK_RATES,
    TRADE_WINDOW_SECONDS,
    WINGMAN_MAPS,
    DemoSource,
    DemoType,
    GameMode,
    RoundEndReason,
    Team,
)
from opensight.core.schemas import (
    AnalysisWarning,
    ClutchInfo,
    EconomyIQData,
    HeatmapData,
    HeatmapPosition,
    MatchInfo,
    MatchResult,
    PlayerPositionSnapshot,
    RoundEconomy,
    RoundTimelineEntry,
    TimelineBombEvent,
    TimelineKillEvent,
    TimelineUtilityEvent,
)

__all__ = [
    # Enums
    "DemoSource",
    "DemoType",
    "GameMode",
    "RoundEndReason",
    "Team",
    # Constants
    "ACTIVE_DUTY_MAPS",
    "CLUTCH_TIME_THRESHOLD",
    "COMPETITIVE_MAPS",
    "CS2_TICK_RATE",
    "FILENAME_PATTERNS",
    "FLASH_ASSIST_MIN_DURATION",
    "HLTV_RATING_COEFFICIENTS",
    "IMPACT_COEFFICIENTS",
    "RESERVE_MAPS",
    "SOURCE_IDENTIFIERS",
    "TICK_RATES",
    "TRADE_WINDOW_SECONDS",
    "WINGMAN_MAPS",
    # Schemas (data contracts)
    "AnalysisWarning",
    "ClutchInfo",
    "EconomyIQData",
    "HeatmapData",
    "HeatmapPosition",
    "MatchInfo",
    "MatchResult",
    "PlayerPositionSnapshot",
    "RoundEconomy",
    "RoundTimelineEntry",
    "TimelineBombEvent",
    "TimelineKillEvent",
    "TimelineUtilityEvent",
]
