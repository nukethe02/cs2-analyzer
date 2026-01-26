"""
Enhanced Cache Integration

Bridges the enhanced parser with the caching system to provide
comprehensive coaching metrics.
"""

import logging
from pathlib import Path

from opensight.core.enhanced_parser import CoachingAnalysisEngine

logger = logging.getLogger(__name__)


def analyze_with_enhanced_metrics(demo_path: Path) -> dict:
    """
    Analyze demo with comprehensive professional metrics.

    This replaces the standard analysis with enhanced metrics extraction
    including TTD, CP, entry/trade/clutch detection, and chunked processing.

    Args:
        demo_path: Path to demo file

    Returns:
        Comprehensive analysis dict ready for web display
    """
    logger.info(f"Starting enhanced analysis of {demo_path.name}")

    engine = CoachingAnalysisEngine(demo_path)
    analysis_result = engine.analyze()

    logger.info(f"Enhanced analysis complete: {analysis_result['total_rounds']} rounds, "
               f"{analysis_result['total_kills']} kills")

    return analysis_result
