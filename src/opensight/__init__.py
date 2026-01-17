"""
OpenSight - Local CS2 Analytics Framework

A locally-operated analytics system for Counter-Strike 2 that provides
professional-grade metrics without cloud dependencies.
"""

__version__ = "0.1.0"
__author__ = "OpenSight Contributors"

from opensight.sharecode import decode_sharecode, ShareCodeInfo
from opensight.parser import DemoParser, DemoData
from opensight.metrics import calculate_ttd, calculate_crosshair_placement
from opensight.watcher import ReplayWatcher

__all__ = [
    "decode_sharecode",
    "ShareCodeInfo",
    "DemoParser",
    "DemoData",
    "calculate_ttd",
    "calculate_crosshair_placement",
    "ReplayWatcher",
]
