"""
OpenSight - Professional CS2 Demo Analyzer

A locally-operated analytics system for Counter-Strike 2 that provides
professional-grade metrics without cloud dependencies.
"""

__version__ = "0.4.0"
__author__ = "OpenSight Contributors"

# Core exports - lazy loaded for fast startup
from opensight.analysis.analytics import DemoAnalyzer
from opensight.core.parser import DemoData, DemoParser
from opensight.infra.cache import CachedAnalyzer

__all__ = [
    "__version__",
    "DemoParser",
    "DemoData",
    "DemoAnalyzer",
    "CachedAnalyzer",
]
