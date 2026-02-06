"""
OpenSight - Professional CS2 Demo Analyzer

A locally-operated analytics system for Counter-Strike 2 that provides
professional-grade metrics without cloud dependencies.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("opensight")
except PackageNotFoundError:
    __version__ = "0.5.0"

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
