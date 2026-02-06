"""
OpenSight Pipeline - Demo analysis orchestration.

This module handles the complete demo processing pipeline:
- File validation
- Demo parsing (DemoParser)
- Analysis execution (DemoAnalyzer)
- Result serialization
"""

from opensight.pipeline.orchestrator import DemoOrchestrator, analyze_demo

__all__ = ["DemoOrchestrator", "analyze_demo"]
