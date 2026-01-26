"""
OpenSight - Professional CS2 Demo Analyzer

A locally-operated analytics system for Counter-Strike 2 that provides
professional-grade metrics without cloud dependencies.

Features:
- Parse CS2 demo files (.dem) locally
- Calculate professional-grade performance metrics
- Monitor replay folders for automatic analysis
- Export to multiple formats (JSON, CSV, Excel, HTML)
"""

__version__ = "0.3.0"
__author__ = "OpenSight Contributors"


def __getattr__(name):
    """Lazy import for heavy dependencies."""
    # Share code
    if name == "decode_sharecode":
        from opensight.integrations.sharecode import decode_sharecode

        return decode_sharecode
    elif name == "encode_sharecode":
        from opensight.integrations.sharecode import encode_sharecode

        return encode_sharecode
    elif name == "ShareCodeInfo":
        from opensight.integrations.sharecode import ShareCodeInfo

        return ShareCodeInfo

    # Parser
    elif name == "DemoParser":
        from opensight.core.parser import DemoParser

        return DemoParser
    elif name == "DemoData":
        from opensight.core.parser import DemoData

        return DemoData
    elif name == "calculate_ttd":
        from opensight.analysis.metrics import calculate_ttd

        return calculate_ttd
    elif name == "calculate_crosshair_placement":
        from opensight.analysis.metrics import calculate_crosshair_placement

        return calculate_crosshair_placement
    elif name == "ReplayWatcher":
        from opensight.infra.watcher import ReplayWatcher

        return ReplayWatcher
    elif name == "EconomyAnalyzer":
        from opensight.domains.economy import EconomyAnalyzer

        return EconomyAnalyzer
    elif name == "analyze_economy":
        from opensight.domains.economy import analyze_economy

        return analyze_economy
    elif name == "BuyType":
        from opensight.domains.economy import BuyType

        return BuyType
    elif name == "CombatAnalyzer":
        from opensight.domains.combat import CombatAnalyzer

        return CombatAnalyzer
    elif name == "analyze_combat":
        from opensight.domains.combat import analyze_combat

        return analyze_combat
    elif name == "UtilityAnalyzer":
        from opensight.domains.utility import UtilityAnalyzer

        return UtilityAnalyzer
    elif name == "analyze_utility":
        from opensight.domains.utility import analyze_utility

        return analyze_utility
    elif name == "UtilityMetrics":
        from opensight.analysis.analytics import UtilityMetrics

        return UtilityMetrics
    elif name == "compute_utility_metrics":
        from opensight.analysis.analytics import compute_utility_metrics

        return compute_utility_metrics
    elif name == "DemoAnalyzer":
        from opensight.analysis.analytics import DemoAnalyzer

        return DemoAnalyzer

    # Advanced AI Coaching
    elif name == "AdaptiveCoach":
        from opensight.ai.coaching import AdaptiveCoach

        return AdaptiveCoach
    elif name == "generate_coaching_insights":
        from opensight.ai.coaching import generate_coaching_insights

        return generate_coaching_insights

    # Temporal Pattern Analysis
    elif name == "PatternAggregator":
        from opensight.ai.patterns import PatternAggregator

        return PatternAggregator
    elif name == "analyze_demo_patterns":
        from opensight.ai.patterns import analyze_demo_patterns

        return analyze_demo_patterns

    # Opponent Modeling
    elif name == "OpponentModeler":
        from opensight.ai.opponent import OpponentModeler

        return OpponentModeler
    elif name == "get_scouting_report":
        from opensight.ai.opponent import get_scouting_report

        return get_scouting_report

    # Team Playbook
    elif name == "PlaybookGenerator":
        from opensight.ai.playbook import PlaybookGenerator

        return PlaybookGenerator
    elif name == "analyze_team_demo":
        from opensight.ai.playbook import analyze_team_demo

        return analyze_team_demo

    # Real-time Coaching
    elif name == "RealtimeCoachingSession":
        from opensight.ai.realtime import RealtimeCoachingSession

        return RealtimeCoachingSession
    elif name == "create_coaching_session":
        from opensight.ai.realtime import create_coaching_session

        return create_coaching_session

    # Sentiment Analysis
    elif name == "CommunicationAnalyzer":
        from opensight.ai.sentiment import CommunicationAnalyzer

        return CommunicationAnalyzer
    elif name == "analyze_team_morale":
        from opensight.ai.sentiment import analyze_team_morale

        return analyze_team_morale

    # Custom Metrics
    elif name == "CustomMetricBuilder":
        from opensight.ai.custom_metrics import CustomMetricBuilder

        return CustomMetricBuilder
    elif name == "create_custom_metric":
        from opensight.ai.custom_metrics import create_custom_metric

        return create_custom_metric

    # Collaboration
    elif name == "CollaborationManager":
        from opensight.integrations.collaboration import CollaborationManager

        return CollaborationManager
    elif name == "create_collaboration_session":
        from opensight.integrations.collaboration import create_collaboration_session

        return create_collaboration_session

    # Profiling
    elif name == "TimingCollector":
        from opensight.infra.profiling import TimingCollector

        return TimingCollector
    elif name == "SlowJobLogger":
        from opensight.infra.profiling import SlowJobLogger

        return SlowJobLogger
    elif name == "Profiler":
        from opensight.infra.profiling import Profiler

        return Profiler
    elif name == "stage_timer":
        from opensight.infra.profiling import stage_timer

        return stage_timer
    elif name == "create_timing_context":
        from opensight.infra.profiling import create_timing_context

        return create_timing_context

    # New modules
    elif name == "ParallelDemoAnalyzer":
        from opensight.infra.parallel import ParallelDemoAnalyzer

        return ParallelDemoAnalyzer
    elif name == "analyze_batch":
        from opensight.infra.parallel import analyze_batch

        return analyze_batch
    elif name == "CoordinateTransformer":
        from opensight.visualization.radar import CoordinateTransformer

        return CoordinateTransformer
    elif name == "RadarDataGenerator":
        from opensight.visualization.radar import RadarDataGenerator

        return RadarDataGenerator
    elif name == "ReplayGenerator":
        from opensight.visualization.replay import ReplayGenerator

        return ReplayGenerator
    elif name == "HLTVClient":
        from opensight.integrations.hltv import HLTVClient

        return HLTVClient
    elif name == "MatchEnricher":
        from opensight.integrations.hltv import MatchEnricher

        return MatchEnricher
    elif name == "DemoCache":
        from opensight.infra.cache import DemoCache

        return DemoCache
    elif name == "CachedAnalyzer":
        from opensight.infra.cache import CachedAnalyzer

        return CachedAnalyzer
    elif name == "FeedbackDatabase":
        from opensight.integrations.feedback import FeedbackDatabase

        return FeedbackDatabase

    # DataFrame Backend (Polars optimization)
    elif name == "get_backend":
        from opensight.infra.backend import get_backend

        return get_backend
    elif name == "DataFrameBackend":
        from opensight.infra.backend import DataFrameBackend

        return DataFrameBackend
    elif name == "PandasBackend":
        from opensight.infra.backend import PandasBackend

        return PandasBackend
    elif name == "PolarsBackend":
        from opensight.infra.backend import PolarsBackend

        return PolarsBackend
    elif name == "PolarsLazyBackend":
        from opensight.infra.backend import PolarsLazyBackend

        return PolarsLazyBackend
    elif name == "save_dataframe":
        from opensight.infra.backend import save_dataframe

        return save_dataframe
    elif name == "load_dataframe":
        from opensight.infra.backend import load_dataframe

        return load_dataframe
    elif name == "convert_dataframe":
        from opensight.infra.backend import convert_dataframe

        return convert_dataframe
    elif name == "is_polars_available":
        from opensight.infra.backend import is_polars_available

        return is_polars_available
    elif name == "benchmark_backends":
        from opensight.infra.backend import benchmark_backends

        return benchmark_backends
    elif name == "BackendConfig":
        from opensight.infra.backend import BackendConfig

        return BackendConfig
    elif name == "set_backend_config":
        from opensight.infra.backend import set_backend_config

        return set_backend_config
    elif name == "get_backend_config":
        from opensight.infra.backend import get_backend_config

        return get_backend_config

    # Game State Tracking
    elif name == "GameStateTracker":
        from opensight.analysis.game_state import GameStateTracker

        return GameStateTracker
    elif name == "track_game_state":
        from opensight.analysis.game_state import track_game_state

        return track_game_state
    elif name == "GameState":
        from opensight.analysis.game_state import GameState

        return GameState

    # Player Behavior Analysis
    elif name == "PlayerBehaviorAnalyzer":
        from opensight.analysis.player_behavior import PlayerBehaviorAnalyzer

        return PlayerBehaviorAnalyzer
    elif name == "analyze_player_behavior":
        from opensight.analysis.player_behavior import analyze_player_behavior

        return analyze_player_behavior
    elif name == "PlayerBehaviorProfile":
        from opensight.analysis.player_behavior import PlayerBehaviorProfile

        return PlayerBehaviorProfile

    # FACEIT Integration
    elif name == "FACEITClient":
        from opensight.integrations.faceit import FACEITClient

        return FACEITClient
    elif name == "get_faceit_player":
        from opensight.integrations.faceit import get_faceit_player

        return get_faceit_player
    elif name == "get_faceit_match_history":
        from opensight.integrations.faceit import get_faceit_match_history

        return get_faceit_match_history

    raise AttributeError(f"module 'opensight' has no attribute '{name}'")


# Export (imported here after lazy loading setup)
from opensight.visualization.export import (  # noqa: E402
    export_analysis,
    export_metrics_to_csv,
    export_to_excel,
    export_to_html,
    export_to_json,
)

__all__ = [
    # Version
    "__version__",
    # Share code
    "decode_sharecode",
    "encode_sharecode",
    "ShareCodeInfo",
    # Parser
    "DemoParser",
    "DemoData",
    "DemoAnalyzer",
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
    "UtilityMetrics",
    "compute_utility_metrics",
    # Advanced AI Coaching
    "AdaptiveCoach",
    "generate_coaching_insights",
    # Temporal Pattern Analysis
    "PatternAggregator",
    "analyze_demo_patterns",
    # Opponent Modeling
    "OpponentModeler",
    "get_scouting_report",
    # Team Playbook
    "PlaybookGenerator",
    "analyze_team_demo",
    # Real-time Coaching
    "RealtimeCoachingSession",
    "create_coaching_session",
    # Sentiment Analysis
    "CommunicationAnalyzer",
    "analyze_team_morale",
    # Custom Metrics
    "CustomMetricBuilder",
    "create_custom_metric",
    # Collaboration
    "CollaborationManager",
    "create_collaboration_session",
    # DataFrame Backend (Polars optimization)
    "get_backend",
    "DataFrameBackend",
    "PandasBackend",
    "PolarsBackend",
    "PolarsLazyBackend",
    "save_dataframe",
    "load_dataframe",
    "convert_dataframe",
    "is_polars_available",
    "benchmark_backends",
    "BackendConfig",
    "set_backend_config",
    "get_backend_config",
    # Export functions
    "export_analysis",
    "export_metrics_to_csv",
    "export_to_excel",
    "export_to_html",
    "export_to_json",
    # Game State Tracking
    "GameStateTracker",
    "track_game_state",
    "GameState",
    # Player Behavior Analysis
    "PlayerBehaviorAnalyzer",
    "analyze_player_behavior",
    "PlayerBehaviorProfile",
    # FACEIT Integration
    "FACEITClient",
    "get_faceit_player",
    "get_faceit_match_history",
]


# Backwards-compatible module alias for tests that import opensight.sharecode
try:
    from opensight.integrations import sharecode as sharecode  # type: ignore
except Exception:
    sharecode = None
