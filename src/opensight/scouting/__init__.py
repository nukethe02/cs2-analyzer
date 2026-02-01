"""
Opponent Scouting Engine - Multi-demo analysis for team preparation.

This module provides tools for analyzing multiple demos of opponent teams
to generate player profiles, team tendencies, and anti-strat recommendations.
"""

from opensight.scouting.anti_strats import format_anti_strats_markdown, generate_anti_strats
from opensight.scouting.engine import ScoutingEngine, create_scouting_engine
from opensight.scouting.models import (
    EconomyTendency,
    MapTendency,
    PlayerScoutProfile,
    PlayStyle,
    PositionTendency,
    ScoutingSession,
    TeamScoutReport,
    WeaponPreference,
)

__all__ = [
    # Engine
    "ScoutingEngine",
    "create_scouting_engine",
    # Models
    "EconomyTendency",
    "MapTendency",
    "PlayerScoutProfile",
    "PlayStyle",
    "PositionTendency",
    "ScoutingSession",
    "TeamScoutReport",
    "WeaponPreference",
    # Anti-strats
    "generate_anti_strats",
    "format_anti_strats_markdown",
]
