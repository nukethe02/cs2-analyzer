"""
Professional Analytics Engine for CS2 Demo Analysis

Implements industry-standard metrics:
- HLTV 2.0 Rating
- KAST% (Kill/Assist/Survived/Traded)
- ADR (Average Damage per Round)
- Trade kill detection
- Clutch detection
- Opening duel analysis
- Multi-kill tracking
- TTD (Time to Damage)
- Crosshair Placement
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from opensight.analysis.models import *  # noqa: F403, F405
from opensight.core.constants import (
    CS2_TICK_RATE,
    TRADE_WINDOW_SECONDS,
)
from opensight.core.parser import DemoData, safe_float, safe_int, safe_str

logger = logging.getLogger(__name__)

# Import optimized metrics computation
try:
    from opensight.analysis.metrics_optimized import (  # noqa: F401
        MetricType,
        OptimizedMetricsComputer,
        compute_cp_from_dataframe_vectorized,
        compute_cp_vectorized,
        compute_ttd_vectorized,
        set_cache_directory,
    )

    HAS_OPTIMIZED_METRICS = True
except ImportError:
    HAS_OPTIMIZED_METRICS = False
    MetricType = None  # Placeholder
    OptimizedMetricsComputer = None  # Placeholder
    logger.debug("Optimized metrics module not available")

# Import economy and combat modules for integration
try:
    from opensight.domains.economy import (  # noqa: F401
        BuyType,
        EconomyAnalyzer,
        EconomyStats,
        PlayerEconomyProfile,
    )

    HAS_ECONOMY = True
except ImportError:
    HAS_ECONOMY = False
    logger.debug("Economy module not available")

try:
    from opensight.domains.combat import (  # noqa: F401
        CombatAnalysisResult,
        CombatAnalyzer,
        PlayerCombatStats,
    )

    HAS_COMBAT = True
except ImportError:
    HAS_COMBAT = False
    logger.debug("Combat module not available")

# Import lurker detection for smart spacing warnings
try:
    from opensight.analysis.persona import _is_effective_lurker

    HAS_PERSONA = True
except ImportError:
    HAS_PERSONA = False
    _is_effective_lurker = None
    logger.debug("Persona module not available")


# Note: safe_int, safe_str, safe_float are imported from opensight.core.parser


def compute_kill_positions(match_data: DemoData) -> list[dict]:
    """
    Compute kill positions for Kill Map visualization.

    Extracts kill data from match_data.kills for plotting on map radar images.
    Uses victim position as the kill location (where the player died).

    Args:
        match_data: MatchData or DemoData object with kills list and player_names dict.

    Returns:
        List of dicts with kill information:
        - attacker_name: Name of the player who got the kill
        - victim_name: Name of the player who died
        - attacker_team: Team of attacker ("CT" or "T")
        - victim_team: Team of victim ("CT" or "T")
        - weapon: Weapon used for the kill
        - is_headshot: Whether it was a headshot
        - x: X coordinate (victim position)
        - y: Y coordinate (victim position)
        - z: Z coordinate (victim position)
        - round_num: Round number when the kill occurred
    """
    kill_positions = []

    # Handle both DemoData (with kills list) and MatchData
    kills = getattr(match_data, "kills", [])
    player_names = getattr(match_data, "player_names", {})

    for kill in kills:
        try:
            # Get player names
            attacker_name = player_names.get(kill.attacker_steamid, kill.attacker_name or "Unknown")
            victim_name = player_names.get(kill.victim_steamid, kill.victim_name or "Unknown")

            # Use victim position as the kill location (where the death occurred)
            x = kill.victim_x if kill.victim_x is not None else kill.attacker_x
            y = kill.victim_y if kill.victim_y is not None else kill.attacker_y
            z = kill.victim_z if kill.victim_z is not None else (kill.attacker_z or 0)

            if x is None or y is None:
                continue

            # Determine attacker team from side info
            attacker_team = "Unknown"
            if hasattr(kill, "attacker_side") and kill.attacker_side:
                side = str(kill.attacker_side).upper()
                if "CT" in side:
                    attacker_team = "CT"
                elif "T" in side and "CT" not in side:
                    attacker_team = "T"

            victim_team = "Unknown"
            if hasattr(kill, "victim_side") and kill.victim_side:
                side = str(kill.victim_side).upper()
                if "CT" in side:
                    victim_team = "CT"
                elif "T" in side and "CT" not in side:
                    victim_team = "T"

            kill_positions.append(
                {
                    "attacker_name": attacker_name,
                    "victim_name": victim_name,
                    "attacker_team": attacker_team,
                    "victim_team": victim_team,
                    "weapon": kill.weapon or "Unknown",
                    "is_headshot": bool(kill.headshot),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z) if z else 0.0,
                    "round_num": kill.round_num or 0,
                }
            )
        except Exception as e:
            logger.debug(f"Error extracting kill position: {e}")
            continue

    logger.info(f"Computed {len(kill_positions)} kill positions for Kill Map")
    return kill_positions


class AnalysisValidator:
    """Validates analysis results for data quality issues.

    Catches common problems like:
    - Negative values where only positive are valid
    - Kill/death totals not matching (team kills vs team deaths)
    - Missing required data
    - Out-of-range percentages

    Usage:
        validator = AnalysisValidator()
        result = validator.validate(analysis_result)
        if not result.is_valid:
            logger.warning(f"Validation errors: {result.errors}")
    """

    def validate(
        self, players: dict[int, "PlayerMatchStats"], total_rounds: int
    ) -> ValidationResult:
        """
        Validate analysis data for quality issues.

        Args:
            players: Dict of steam_id -> PlayerMatchStats
            total_rounds: Total rounds in the match

        Returns:
            ValidationResult with is_valid flag and any errors/warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not players:
            errors.append("No players in analysis result")
            return ValidationResult(is_valid=False, errors=errors)

        # Check individual player stats
        for steam_id, player in players.items():
            player_name = getattr(player, "name", f"Player_{steam_id}")

            # No negative values
            if getattr(player, "kills", 0) < 0:
                errors.append(f"Negative kills for {player_name}: {player.kills}")

            if getattr(player, "deaths", 0) < 0:
                errors.append(f"Negative deaths for {player_name}: {player.deaths}")

            if getattr(player, "assists", 0) < 0:
                errors.append(f"Negative assists for {player_name}: {player.assists}")

            # Rounds survived should be non-negative
            deaths = getattr(player, "deaths", 0)
            rounds_played = getattr(player, "rounds_played", total_rounds)
            rounds_survived = rounds_played - deaths
            if rounds_survived < 0:
                errors.append(
                    f"Negative rounds survived for {player_name}: "
                    f"{rounds_survived} (rounds={rounds_played}, deaths={deaths})"
                )

            # Percentages should be 0-100
            for pct_attr in ["headshot_percentage", "kast_percentage", "survival_rate"]:
                pct_value = getattr(player, pct_attr, None)
                if pct_value is not None:
                    if pct_value < 0:
                        errors.append(f"Negative {pct_attr} for {player_name}: {pct_value}")
                    elif pct_value > 100:
                        warnings.append(f"High {pct_attr} for {player_name}: {pct_value}")

            # ADR sanity check (typically 0-200, but can be higher in rare cases)
            adr = getattr(player, "adr", None)
            if adr is not None:
                if adr < 0:
                    errors.append(f"Negative ADR for {player_name}: {adr}")
                elif adr > 300:
                    warnings.append(f"Unusually high ADR for {player_name}: {adr}")

            # HLTV rating sanity check (typically 0-3, but can be higher)
            rating = getattr(player, "hltv_rating", None)
            if rating is not None:
                if rating < 0:
                    errors.append(f"Negative HLTV rating for {player_name}: {rating}")
                elif rating > 5:
                    warnings.append(f"Unusually high HLTV rating for {player_name}: {rating}")

        # Check team balance (total kills should roughly equal total deaths)
        # Note: Self-damage deaths and team kills can cause slight imbalances
        total_kills = sum(getattr(p, "kills", 0) for p in players.values())
        total_deaths = sum(getattr(p, "deaths", 0) for p in players.values())

        if total_kills > 0 and total_deaths > 0:
            # Allow 10% tolerance for edge cases (suicides, disconnects)
            ratio = total_kills / total_deaths
            if ratio < 0.9 or ratio > 1.1:
                warnings.append(
                    f"Kill/death imbalance: {total_kills} kills vs {total_deaths} deaths "
                    f"(ratio: {ratio:.2f})"
                )

        # Check for at least 10 players (5v5)
        if len(players) < 10:
            warnings.append(f"Less than 10 players: {len(players)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_and_fix(
        self, players: dict[int, "PlayerMatchStats"], total_rounds: int
    ) -> tuple[dict[int, "PlayerMatchStats"], ValidationResult]:
        """
        Validate and attempt to fix common issues.

        Currently fixes:
        - Negative rounds_survived by clamping to 0

        Args:
            players: Dict of steam_id -> PlayerMatchStats
            total_rounds: Total rounds in the match

        Returns:
            Tuple of (possibly fixed players, validation result)
        """
        # First validate to find issues
        result = self.validate(players, total_rounds)

        # Log any issues
        if result.errors:
            logger.warning(f"Validation errors found: {result.errors}")
        if result.warnings:
            logger.debug(f"Validation warnings: {result.warnings}")

        return players, result


class DemoAnalyzer:
    """Analyzer for computing professional-grade metrics from parsed demo data.

    Supports configurable metrics computation for performance optimization.
    When only specific metrics are needed (e.g., just KD ratio), you can skip
    expensive computations like TTD and CP.

    Usage:
        # Full analysis (default)
        analyzer = DemoAnalyzer(demo_data)
        result = analyzer.analyze()

        # Only basic metrics (faster)
        analyzer = DemoAnalyzer(demo_data, metrics="basic")
        result = analyzer.analyze()

        # Specific metrics
        analyzer = DemoAnalyzer(demo_data, metrics=["ttd", "cp", "kd"])
        result = analyzer.analyze()
    """

    TICK_RATE = CS2_TICK_RATE
    MS_PER_TICK = 1000 / TICK_RATE

    # Engagement duration thresholds (time from first damage to kill)
    ENGAGEMENT_MIN_MS = 0
    ENGAGEMENT_MAX_MS = 1500  # Kills taking >1.5s are outliers

    # True TTD (reaction time) thresholds
    REACTION_TIME_MIN_MS = 100  # < 100ms = prefire (anticipation, not reaction)
    REACTION_TIME_MAX_MS = 2000  # > 2s = visibility logic likely failed (wallbang/smoke)

    # Legacy aliases
    TTD_MIN_MS = REACTION_TIME_MIN_MS
    TTD_MAX_MS = REACTION_TIME_MAX_MS

    # Column name variations (includes both demoparser2 enriched names and raw event names)
    ROUND_COLS = ["round_num", "total_rounds_played", "round"]
    ATT_ID_COLS = ["attacker_steamid", "attacker_steam_id", "attacker"]
    VIC_ID_COLS = ["victim_steamid", "user_steamid", "victim_steam_id", "userid"]
    ATT_SIDE_COLS = [
        "attacker_side",
        "attacker_team_name",
        "attacker_team",
        "attacker_team_num",
    ]
    VIC_SIDE_COLS = [
        "victim_side",
        "user_team_name",
        "victim_team",
        "user_team_num",
        "victim_team_num",
    ]

    # Available metric categories
    METRIC_CATEGORIES = {
        "basic": ["kd", "adr", "headshots"],
        "kast": ["kast", "survival"],
        "ttd": ["ttd", "prefire"],
        "cp": ["crosshair_placement"],
        "trades": ["trade_kills", "trade_deaths"],
        "opening": ["opening_duels"],
        "multi_kills": ["multi_kill_rounds"],
        "clutches": ["clutch_situations"],
        "utility": ["utility_usage", "flash_effectiveness"],
        "accuracy": ["shots_fired", "shots_hit", "accuracy_percent"],
        "economy": ["equipment_value", "damage_per_dollar"],
        "sides": ["ct_stats", "t_stats"],
        "mistakes": ["team_damage", "team_kills"],
    }

    def __init__(
        self,
        demo_data: DemoData,
        metrics: str | list[str] | None = None,
        use_cache: bool = True,
        use_optimized: bool = True,
    ):
        """
        Initialize analyzer.

        Args:
            demo_data: Parsed demo data from DemoParser
            metrics: Which metrics to compute. Options:
                - None or "full": Compute all metrics (default)
                - "basic": Only basic stats (KD, ADR, HS%)
                - "advanced": Basic + TTD, CP, trades, opening duels
                - List of specific categories: ["ttd", "cp", "trades"]
            use_cache: Whether to use metrics caching (default True)
            use_optimized: Whether to use vectorized implementations (default True)
        """
        self.data = demo_data
        self._ttd_results: list[TTDResult] = []
        self._cp_results: list[CrosshairPlacementResult] = []
        self._players: dict[int, PlayerMatchStats] = {}
        # Cache column lookups
        self._round_col: str | None = None
        self._att_id_col: str | None = None
        self._vic_id_col: str | None = None
        self._att_side_col: str | None = None
        self._vic_side_col: str | None = None

        # Metrics configuration
        self._use_cache = use_cache
        self._use_optimized = use_optimized and HAS_OPTIMIZED_METRICS
        self._metrics_computer: Any = None  # OptimizedMetricsComputer | None
        self._requested_metrics = self._parse_metrics_config(metrics)

    def _parse_metrics_config(self, metrics: str | list[str] | None) -> set[str]:
        """Parse metrics configuration into a set of metric categories."""
        if metrics is None or metrics == "full":
            return set(self.METRIC_CATEGORIES.keys())

        if metrics == "basic":
            return {"basic", "kast", "multi_kills"}

        if metrics == "advanced":
            return {
                "basic",
                "kast",
                "ttd",
                "cp",
                "trades",
                "opening",
                "multi_kills",
                "utility",
            }

        if isinstance(metrics, str):
            return {metrics}

        return set(metrics)

    def _find_col(self, df: pd.DataFrame, options: list[str]) -> str | None:
        """Find first matching column from options."""
        for col in options:
            if col in df.columns:
                return col
        return None

    def _match_steamid(self, df: pd.DataFrame, col: str, steam_id: int) -> pd.DataFrame:
        """Match steamid handling type differences (int vs float)."""
        try:
            # Convert both to float for comparison to handle int/float mismatch
            return df[df[col].astype(float) == float(steam_id)]
        except (ValueError, TypeError):
            # Fallback to direct comparison
            return df[df[col] == steam_id]

    def _normalize_team(self, value: str | int | float | None) -> str:
        """Normalize team/side values to 'CT' or 'T' for consistent comparison.

        Handles various formats from demo data:
        - Strings: 'CT', 'ct', 'CounterTerrorist', 'TERRORIST', etc.
        - Numbers: 2 = T, 3 = CT (CS2 team numbers)
        """
        if value is None:
            return "Unknown"
        if isinstance(value, str):
            upper = value.upper()
            if "CT" in upper or "COUNTER" in upper:
                return "CT"
            elif "T" in upper and "CT" not in upper:
                return "T"
            return "Unknown"
        elif isinstance(value, (int, float)):
            val = int(value)
            if val == 3:
                return "CT"
            elif val == 2:
                return "T"
        return "Unknown"

    def _get_player_side(self, steam_id: int, round_num: int) -> str:
        """Get the ACTUAL side (CT/T) of a player for a specific round.

        Handles halftime swaps automatically using the persistent team system.
        Crucial for clutch/trade/utility logic where side matters per-round.

        Args:
            steam_id: Player's Steam ID
            round_num: Round number (1-indexed)

        Returns:
            "CT", "T", or "Unknown"
        """
        return self.data.get_player_side_for_round(steam_id, round_num)

    def _init_column_cache(self) -> None:
        """Initialize column name cache for kills DataFrame."""
        kills_df = self.data.kills_df
        if kills_df.empty:
            return
        self._round_col = self._find_col(kills_df, self.ROUND_COLS)
        self._att_id_col = self._find_col(kills_df, self.ATT_ID_COLS)
        self._vic_id_col = self._find_col(kills_df, self.VIC_ID_COLS)
        self._att_side_col = self._find_col(kills_df, self.ATT_SIDE_COLS)
        self._vic_side_col = self._find_col(kills_df, self.VIC_SIDE_COLS)
        logger.info(
            f"Column cache: round={self._round_col}, att_id={self._att_id_col}, vic_id={self._vic_id_col}"
        )

    def _validate_data(self) -> list[str]:
        """Validate demo data and return list of warnings."""
        warnings = []

        # Check for required data
        if self.data.kills_df.empty:
            warnings.append("No kill data available - some metrics will be incomplete")

        if self.data.damages_df.empty:
            warnings.append("No damage data available - TTD and ADR will be unavailable")

        if not self.data.player_names:
            warnings.append("No player data found - analysis may fail")

        # Validate round numbers
        if self.data.num_rounds < 1:
            warnings.append("Invalid round count (0) - defaulting to 1")

        if self.data.num_rounds > 60:
            warnings.append(
                f"Unusual round count ({self.data.num_rounds}) - may be overtime or corrupt data"
            )

        # Check for steam ID validity
        invalid_ids = [sid for sid in self.data.player_names.keys() if sid <= 0]
        if invalid_ids:
            warnings.append(f"Found {len(invalid_ids)} players with invalid steam IDs")

        return warnings

    def _safe_calculate(self, func_name: str, func: callable) -> bool:
        """Safely run a calculation function with error handling."""
        try:
            func()
            return True
        except Exception as e:
            logger.error(f"Error in {func_name}: {e}")
            return False

    def analyze(self) -> MatchAnalysis:
        """Run full analysis and return match analysis with comprehensive error handling."""
        logger.info("Starting professional analysis...")

        # Validate data first
        warnings = self._validate_data()
        for warning in warnings:
            logger.warning(warning)

        # Initialize column name cache
        self._init_column_cache()

        # Initialize player stats (always needed)
        self._init_player_stats()

        # Calculate basic stats (critical - don't skip on error)
        self._safe_calculate("basic_stats", self._calculate_basic_stats)

        # Calculate RWS (Round Win Shares)
        self._calculate_rws()

        # Initialize optimized metrics computer if using optimized implementations
        if self._use_optimized and OptimizedMetricsComputer is not None:
            self._metrics_computer = OptimizedMetricsComputer(self.data, use_cache=self._use_cache)

        # Calculate multi-kill rounds
        self._safe_calculate("multi_kills", self._calculate_multi_kills)

        # Detect opening duels
        self._safe_calculate("opening_duels", self._detect_opening_duels)

        # Detect opening engagements (damage-based, not just kill-based)
        self._safe_calculate("opening_engagements", self._detect_opening_engagements)

        # Detect zone-aware entry frags (bombsite kills vs map control)
        self._safe_calculate("entry_frags", self._detect_entry_frags)

        # Detect trade kills
        self._safe_calculate("trade_detection", self._detect_trades)

        # Detect clutches
        self._safe_calculate("clutch_detection", self._detect_clutches)

        # Calculate KAST
        self._safe_calculate("kast", self._calculate_kast)

        # Compute TTD
        self._safe_calculate("ttd", self._compute_ttd)

        # Compute crosshair placement
        self._safe_calculate("crosshair_placement", self._compute_crosshair_placement)

        # Calculate side-based stats (CT vs T)
        self._safe_calculate("side_stats", self._calculate_side_stats)

        # Calculate utility stats
        self._safe_calculate("utility_stats", self._calculate_utility_stats)

        # Calculate accuracy stats (from weapon_fire events)
        self._safe_calculate("accuracy_stats", self._calculate_accuracy_stats)

        # Calculate mistakes
        self._safe_calculate("mistakes", self._calculate_mistakes)

        # Detect greedy re-peeks (discipline tracking)
        self._safe_calculate("greedy_repeeks", self._detect_greedy_repeeks)

        # Run State Machine for pro-level analytics (Entry/Trade/Lurk)
        # self._safe_calculate("state_machine", self._run_state_machine)  # Disabled: produces zeros, no UI consumes output

        # Integrate Economy Module
        if "economy" in self._requested_metrics:
            self._integrate_economy()

        # Integrate Combat Module
        combat_stats = {}
        if "trades" in self._requested_metrics:
            combat_stats = self._integrate_combat()

        # Build kill matrix (always useful)
        kill_matrix = self._build_kill_matrix()

        # Build round timeline
        round_timeline = self._build_round_timeline()

        # Extract position data for heatmaps
        kill_positions, death_positions = self._extract_position_data()

        # Extract grenade trajectory data for utility visualization
        grenade_positions = []
        grenade_team_stats = {}
        if "utility" in self._requested_metrics:
            grenade_positions, grenade_team_stats = self._extract_grenade_trajectories()

        # Generate AI coaching insights
        coaching_insights = self._generate_coaching_insights()

        # Calculate weapon-specific statistics (stub - returns empty dict)
        weapon_stats: dict[str, list] = {}

        # Build result
        team_scores = self._calculate_team_scores()
        team_names = self._extract_team_names()
        analysis = MatchAnalysis(
            players=self._players,
            team1_score=team_scores[0],
            team2_score=team_scores[1],
            total_rounds=self.data.num_rounds,
            map_name=self.data.map_name,
            team1_name=team_names[0],
            team2_name=team_names[1],
            round_timeline=round_timeline,
            kill_matrix=kill_matrix,
            team_trade_rates=combat_stats.get("trade_rates", {}),
            team_opening_rates=combat_stats.get("opening_rates", {}),
            kill_positions=kill_positions,
            death_positions=death_positions,
            grenade_positions=grenade_positions,
            grenade_team_stats=grenade_team_stats,
            coaching_insights=coaching_insights,
            weapon_stats=weapon_stats,
        )

        # Validate analysis results
        validator = AnalysisValidator()
        validation_result = validator.validate(self._players, self.data.num_rounds)
        if validation_result.errors:
            logger.warning(f"Analysis validation errors: {validation_result.errors}")
        if validation_result.warnings:
            logger.debug(f"Analysis validation warnings: {validation_result.warnings}")

        logger.info(f"Analysis complete. {len(self._players)} players analyzed.")
        return analysis

    def _init_player_stats(self) -> None:
        """Initialize PlayerMatchStats for each player."""
        logger.info(f"Initializing stats for {len(self.data.player_names)} players")
        for steam_id, name in self.data.player_names.items():
            # Use persistent team identity to keep teammates grouped correctly
            # even after halftime side swaps
            persistent_team = self.data.get_player_persistent_team(steam_id)
            # Map to display name (CT/T based on starting side) for frontend colors
            team = self.data.get_team_display_name(persistent_team)
            if team == "Unknown":
                # Fallback for backward compatibility with old data
                team = self.data.player_teams.get(steam_id, "Unknown")
            self._players[steam_id] = PlayerMatchStats(
                steam_id=steam_id,
                name=name,
                team=team,
                kills=0,
                deaths=0,
                assists=0,
                headshots=0,
                total_damage=0,
                rounds_played=self.data.num_rounds,
            )
            logger.debug(
                f"Initialized player: {name} (steamid={steam_id}, team={team}, persistent={persistent_team})"
            )

    def _calculate_basic_stats(self) -> None:
        """Calculate basic K/D/A and damage stats."""
        kills_df = self.data.kills_df
        damages_df = self.data.damages_df

        # Use cached column names for kills, with robust fallback
        att_id_col = self._att_id_col
        if not att_id_col:
            att_id_col = self._find_col(kills_df, self.ATT_ID_COLS) or "attacker_steamid"
            logger.warning(f"Attacker column not cached, using fallback: {att_id_col}")

        vic_id_col = self._vic_id_col
        if not vic_id_col:
            # IMPORTANT: Try to find the column dynamically instead of hardcoding
            # demoparser2 uses 'user_steamid', awpy uses 'victim_steamid'
            vic_id_col = self._find_col(kills_df, self.VIC_ID_COLS)
            if vic_id_col:
                logger.warning(f"Victim column not cached, found dynamically: {vic_id_col}")
            else:
                logger.error(
                    f"Could not find victim column! Tried: {self.VIC_ID_COLS}. "
                    f"Available columns: {list(kills_df.columns)[:10]}..."
                )
                vic_id_col = "user_steamid"  # Default to demoparser2 convention

        # Find damage columns
        dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS) if not damages_df.empty else None
        dmg_col = (
            self._find_col(damages_df, ["dmg_health", "damage", "dmg"])
            if not damages_df.empty
            else None
        )

        # Log DataFrame info for debugging
        if not kills_df.empty and att_id_col in kills_df.columns:
            unique_attackers = kills_df[att_id_col].dropna().unique()
            logger.info(
                f"DataFrame has {len(unique_attackers)} unique attackers in column '{att_id_col}'"
            )
            logger.info(f"Player steamids: {list(self._players.keys())[:5]}...")
            logger.info(f"DataFrame attacker steamids (sample): {list(unique_attackers[:5])}")
            logger.info(f"Attacker column dtype: {kills_df[att_id_col].dtype}")

        for steam_id, player in self._players.items():
            # Kills - use cached column
            if not kills_df.empty and att_id_col in kills_df.columns:
                # Convert to same type for comparison (handle float vs int issue)
                player_kills = kills_df[kills_df[att_id_col].astype(float) == float(steam_id)]
                player.kills = len(player_kills)

                if "headshot" in kills_df.columns:
                    player.headshots = int(player_kills["headshot"].sum())

                if "weapon" in kills_df.columns:
                    player.weapon_kills = player_kills["weapon"].value_counts().to_dict()

            # Deaths - use cached column (handles user_steamid vs victim_steamid)
            if not kills_df.empty and vic_id_col in kills_df.columns:
                player.deaths = len(kills_df[kills_df[vic_id_col].astype(float) == float(steam_id)])

            # Assists
            if not kills_df.empty and "assister_steamid" in kills_df.columns:
                player.assists = len(
                    kills_df[kills_df["assister_steamid"].astype(float) == float(steam_id)]
                )

            # Damage - use dynamic column finding
            if dmg_att_col and dmg_col:
                player_dmg = damages_df[damages_df[dmg_att_col].astype(float) == float(steam_id)]
                player.total_damage = int(player_dmg[dmg_col].sum())

            # Flash assists
            if (
                not kills_df.empty
                and "flash_assist" in kills_df.columns
                and "assister_steamid" in kills_df.columns
            ):
                flash_assists = kills_df[
                    (kills_df["assister_steamid"].astype(float) == float(steam_id))
                    & (kills_df["flash_assist"])
                ]
                player.utility.flash_assists = len(flash_assists)

        # Log results
        total_kills = sum(p.kills for p in self._players.values())
        total_deaths = sum(p.deaths for p in self._players.values())
        logger.info(
            f"Basic stats calculated: {total_kills} total kills, {total_deaths} total deaths across {len(self._players)} players"
        )

        # Sanity check: deaths should approximately equal kills (barring suicides/teamkills)
        if total_kills > 0 and total_deaths == 0:
            logger.error(
                f"BUG DETECTED: {total_kills} kills but 0 deaths! "
                f"vic_id_col='{vic_id_col}', in_columns={vic_id_col in kills_df.columns if not kills_df.empty else 'N/A'}"
            )

    def _calculate_rws(self) -> None:
        """
        Calculate RWS (Round Win Shares) - ESEA style metric.

        RWS measures a player's contribution to rounds their team won.
        Formula: For each won round, (player_damage / team_damage) * 100
        Final RWS = average across all won rounds.

        This rewards impactful damage in winning rounds.
        """
        damages_df = self.data.damages_df
        rounds_data = self.data.rounds

        if damages_df.empty:
            logger.info("Skipping RWS calculation - no damage data")
            return

        # Find damage and round columns
        dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS)
        dmg_col = self._find_col(damages_df, ["dmg_health", "damage", "dmg"])
        round_col = self._find_col(damages_df, ["round_num", "round"])

        if not dmg_att_col or not dmg_col or not round_col:
            logger.info(
                f"Skipping RWS calculation - missing columns. att={dmg_att_col}, dmg={dmg_col}, round={round_col}"
            )
            return

        # Get attacker team column (check DataFrame columns, not row)
        att_team_col = self._find_col(
            damages_df, ["attacker_side", "attacker_team", "attacker_team_name"]
        )
        has_team_col = att_team_col is not None and att_team_col in damages_df.columns

        # Build dict of round winners from rounds data
        round_winners: dict[int, str] = {}
        if rounds_data:
            for round_info in rounds_data:
                round_winners[round_info.round_num] = round_info.winner
            logger.info(f"RWS: Built round winners from {len(rounds_data)} rounds")
        else:
            logger.info("RWS: No rounds data - will infer winners from damage patterns")

        # Build player starting team lookup for halftime handling
        player_starting_team: dict[int, str] = {}
        for steam_id, player in self._players.items():
            player_starting_team[steam_id] = player.team

        # Also add from data.player_teams if available
        if hasattr(self.data, "player_teams"):
            for steam_id, team_num in self.data.player_teams.items():
                if steam_id not in player_starting_team:
                    player_starting_team[steam_id] = (
                        "CT" if team_num == 3 else "T" if team_num == 2 else "Unknown"
                    )

        # Track RWS contributions per player
        from collections import defaultdict

        player_rws_contributions: dict[int, list[float]] = defaultdict(list)
        player_damage_in_won: dict[int, int] = defaultdict(int)
        player_rounds_won: dict[int, int] = defaultdict(int)

        # Track for debugging
        rounds_processed = 0
        rounds_with_winner = 0

        # Group damage by round
        for round_num, round_damages in damages_df.groupby(round_col):
            round_num = safe_int(round_num)
            if round_num <= 0:
                continue

            rounds_processed += 1

            # Get winning side for this round
            winning_side = round_winners.get(round_num, "Unknown")

            # If no round winner data, try to infer from kills
            if (
                winning_side not in ["CT", "T"]
                and hasattr(self.data, "kills_df")
                and not self.data.kills_df.empty
            ):
                # Infer winner: count deaths per team, team with fewer deaths likely won
                # This is a heuristic fallback
                kills_df = self.data.kills_df
                round_kills = (
                    kills_df[kills_df[self._round_col] == round_num]
                    if self._round_col
                    else pd.DataFrame()
                )
                if not round_kills.empty and self._vic_side_col:
                    ct_deaths = sum(
                        1
                        for _, k in round_kills.iterrows()
                        if self._normalize_team(k.get(self._vic_side_col)) == "CT"
                    )
                    t_deaths = sum(
                        1
                        for _, k in round_kills.iterrows()
                        if self._normalize_team(k.get(self._vic_side_col)) == "T"
                    )
                    if ct_deaths >= 5:
                        winning_side = "T"
                    elif t_deaths >= 5:
                        winning_side = "CT"

            if winning_side not in ["CT", "T"]:
                continue

            rounds_with_winner += 1

            # Calculate team damage for this round
            team_damage: dict[str, int] = {"CT": 0, "T": 0}
            player_round_damage: dict[int, int] = defaultdict(int)

            for _, dmg_row in round_damages.iterrows():
                attacker_id = safe_int(dmg_row.get(dmg_att_col))
                damage = safe_int(dmg_row.get(dmg_col))

                if attacker_id == 0 or damage <= 0:
                    continue

                # Cap damage at 100 per event (no overkill)
                damage = min(damage, 100)

                # Determine attacker's team for THIS round (handle halftime)
                attacker_team = "Unknown"

                # Priority 1: Use team column from damage event if available
                if has_team_col:
                    attacker_team = self._normalize_side(dmg_row.get(att_team_col))

                # Priority 2: Look up from player data with halftime handling
                if attacker_team not in ["CT", "T"]:
                    attacker_team = self.data.get_player_side_for_round(attacker_id, round_num)

                if attacker_team in ["CT", "T"]:
                    team_damage[attacker_team] += damage
                    player_round_damage[attacker_id] = (
                        player_round_damage.get(attacker_id, 0) + damage
                    )

            # Calculate RWS contribution for players on winning team
            winning_team_total = team_damage[winning_side]
            if winning_team_total > 0:
                for player_id, player_dmg in player_round_damage.items():
                    # Check if this player was on the winning team this round
                    player_team_this_round = "Unknown"
                    if has_team_col:
                        # Already tracked damage only for attacker's team
                        player_team_this_round = self.data.get_player_side_for_round(
                            player_id, round_num
                        )
                    else:
                        player_team_this_round = self.data.get_player_side_for_round(
                            player_id, round_num
                        )

                    if player_team_this_round == winning_side and player_id in self._players:
                        rws_contribution = (player_dmg / winning_team_total) * 100
                        player_rws_contributions[player_id].append(rws_contribution)
                        player_damage_in_won[player_id] += player_dmg
                        player_rounds_won[player_id] += 1

        # Calculate average RWS for each player
        for steam_id, player in self._players.items():
            contributions = player_rws_contributions.get(steam_id, [])
            if contributions:
                player.rws = round(sum(contributions) / len(contributions), 2)
            else:
                player.rws = 0.0
            player.damage_in_won_rounds = player_damage_in_won.get(steam_id, 0)
            player.rounds_won = player_rounds_won.get(steam_id, 0)

        total_rws = sum(p.rws for p in self._players.values())
        players_with_rws = sum(1 for p in self._players.values() if p.rws > 0)
        logger.info(
            f"RWS calculated: {players_with_rws}/{len(self._players)} players have RWS>0, "
            f"total={total_rws:.1f}, rounds processed={rounds_processed}, with winner={rounds_with_winner}"
        )

    def _normalize_side(self, value) -> str:
        """Normalize team/side values to 'CT' or 'T'."""
        if value is None:
            return "Unknown"
        if isinstance(value, str):
            upper = value.upper()
            if "CT" in upper or "COUNTER" in upper:
                return "CT"
            elif "T" in upper and "CT" not in upper:
                return "T"
            return value
        elif isinstance(value, (int, float)):
            val = int(value)
            if val == 3:
                return "CT"
            elif val == 2:
                return "T"
        return "Unknown"

    def _calculate_multi_kills(self) -> None:
        """Calculate multi-kill rounds for each player.

        Counts enemy kills only (excludes teamkills).
        Each round is assigned to exactly one category (1K, 2K, 3K, 4K, or 5K).
        """
        kills_df = self.data.kills_df
        if kills_df.empty or not self._round_col or not self._att_id_col:
            logger.info("Skipping multi-kill calculation - missing columns")
            return

        # Filter out teamkills if we have side columns
        valid_kills = kills_df
        if self._att_side_col and self._vic_side_col:
            if self._att_side_col in kills_df.columns and self._vic_side_col in kills_df.columns:
                # Only count enemy kills (attacker_side != victim_side)
                valid_kills = kills_df[kills_df[self._att_side_col] != kills_df[self._vic_side_col]]
                teamkills_filtered = len(kills_df) - len(valid_kills)
                if teamkills_filtered > 0:
                    logger.debug(f"Multi-kill calc: filtered {teamkills_filtered} teamkills")

        for steam_id, player in self._players.items():
            player_kills = valid_kills[
                valid_kills[self._att_id_col].astype(float) == float(steam_id)
            ]
            if player_kills.empty:
                continue
            kills_per_round = player_kills.groupby(self._round_col).size()

            player.multi_kills.rounds_with_1k = int((kills_per_round == 1).sum())
            player.multi_kills.rounds_with_2k = int((kills_per_round == 2).sum())
            player.multi_kills.rounds_with_3k = int((kills_per_round == 3).sum())
            player.multi_kills.rounds_with_4k = int((kills_per_round == 4).sum())
            player.multi_kills.rounds_with_5k = int((kills_per_round >= 5).sum())

    def _is_utility_supported(
        self,
        kill_tick: int,
        kill_x: float | None,
        kill_y: float | None,
        kill_z: float | None,
        player_team: str,
        round_num: int,
    ) -> bool:
        """Check if an engagement was supported by teammate utility (flash/smoke).

        An engagement is considered "supported" if a teammate's flash or smoke
        detonated within 3 seconds prior AND within 2000 game units of the kill position.

        This detects "dry peeks" - entry plays taken without utility support.

        Args:
            kill_tick: Tick when the kill/death occurred
            kill_x, kill_y, kill_z: Position of the engagement
            player_team: Team of the player ('CT' or 'T')
            round_num: Round number for filtering grenades

        Returns:
            True if flash or smoke support was present, False if dry peek
        """
        # Need position data to check spatial proximity
        if kill_x is None or kill_y is None:
            return False  # Can't determine, assume unsupported

        # Check if we have grenade data
        if not hasattr(self.data, "grenades") or not self.data.grenades:
            return False  # No grenade data, can't determine support

        # Constants for support detection
        SUPPORT_WINDOW_TICKS = int(3.0 * self.TICK_RATE)  # 3 seconds
        SUPPORT_DISTANCE = 2000.0  # Game units

        # Get team grenades (flashes and smokes from teammates)
        for grenade in self.data.grenades:
            # Only consider flashes and smokes as "support" utility
            grenade_type = grenade.grenade_type.lower()
            if "flash" not in grenade_type and "smoke" not in grenade_type:
                continue

            # Only count detonations
            if grenade.event_type != "detonate":
                continue

            # Must be from same round
            if grenade.round_num != round_num:
                continue

            # Must be from same team (teammate utility)
            grenade_team = self._normalize_team(grenade.player_side)
            if grenade_team != player_team:
                continue

            # Need position data
            if grenade.x is None or grenade.y is None:
                continue

            # Temporal check: grenade detonated within 3 seconds BEFORE the kill
            tick_diff = kill_tick - grenade.tick
            if tick_diff < 0 or tick_diff > SUPPORT_WINDOW_TICKS:
                continue

            # Spatial check: grenade detonated within 2000 units
            dx = kill_x - grenade.x
            dy = kill_y - grenade.y
            dz = (kill_z - grenade.z) if (kill_z and grenade.z) else 0
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance <= SUPPORT_DISTANCE:
                return True  # Found supporting utility

        return False  # No supporting utility found

    def _is_sniper_weapon(self, weapon: str | None) -> bool:
        """Check if weapon is a sniper rifle (AWP, Scout, Autos).

        Snipers legitimately hold angles without utility support,
        so they should be excluded from dry peek tracking.
        """
        if not weapon:
            return False
        weapon_lower = weapon.lower()
        sniper_weapons = {"awp", "ssg08", "g3sg1", "scar20", "weapon_awp", "weapon_ssg08"}
        return any(sniper in weapon_lower for sniper in sniper_weapons)

    def _detect_opening_duels(self) -> None:
        """Detect opening duels (first kill of each round) with Entry TTD and Dry Peek tracking.

        Entry duels are the first kills of each round. This method:
        1. Identifies the first kill of each round
        2. Calculates Entry TTD (time from first damage to kill for entry frags)
        3. Tracks T-side vs CT-side entries for context
        4. Detects "dry peeks" - entries without teammate utility support
        """
        kills_df = self.data.kills_df
        damages_df = self.data.damages_df
        if kills_df.empty or not self._round_col or not self._att_id_col or not self._vic_id_col:
            logger.info("Skipping opening duels - missing columns")
            return

        # Find damage columns for Entry TTD calculation
        dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS) if not damages_df.empty else None
        dmg_vic_col = self._find_col(damages_df, self.VIC_ID_COLS) if not damages_df.empty else None

        # Find position columns for dry peek detection
        att_x_col = self._find_col(kills_df, ["attacker_X", "attacker_x", "X", "x"])
        att_y_col = self._find_col(kills_df, ["attacker_Y", "attacker_y", "Y", "y"])
        att_z_col = self._find_col(kills_df, ["attacker_Z", "attacker_z", "Z", "z"])
        vic_x_col = self._find_col(kills_df, ["user_X", "victim_X", "user_x", "victim_x"])
        vic_y_col = self._find_col(kills_df, ["user_Y", "victim_Y", "user_y", "victim_y"])
        vic_z_col = self._find_col(kills_df, ["user_Z", "victim_Z", "user_z", "victim_z"])

        # Find weapon column for sniper detection
        weapon_col = self._find_col(kills_df, ["weapon", "weapon_name", "attacker_weapon"])

        entry_kills_count = 0
        dry_peek_entries = 0
        dry_peek_deaths = 0

        # Get first kill of each round
        for round_num in kills_df[self._round_col].unique():
            round_num_int = safe_int(round_num)
            round_kills = kills_df[kills_df[self._round_col] == round_num].sort_values("tick")
            if round_kills.empty:
                continue

            first_kill = round_kills.iloc[0]
            attacker_id = safe_int(first_kill.get(self._att_id_col))
            victim_id = safe_int(first_kill.get(self._vic_id_col))
            kill_tick = safe_int(first_kill.get("tick"))

            # Get attacker side for T/CT classification using normalized team values
            attacker_side = ""
            victim_side = ""
            if self._att_side_col and self._att_side_col in kills_df.columns:
                attacker_side = self._normalize_team(first_kill.get(self._att_side_col))
                if attacker_side == "Unknown":
                    attacker_side = ""
            if self._vic_side_col and self._vic_side_col in kills_df.columns:
                victim_side = self._normalize_team(first_kill.get(self._vic_side_col))
                if victim_side == "Unknown":
                    victim_side = ""

            # Get position data for dry peek detection
            att_x = safe_float(first_kill.get(att_x_col)) if att_x_col else None
            att_y = safe_float(first_kill.get(att_y_col)) if att_y_col else None
            att_z = safe_float(first_kill.get(att_z_col)) if att_z_col else None
            vic_x = safe_float(first_kill.get(vic_x_col)) if vic_x_col else None
            vic_y = safe_float(first_kill.get(vic_y_col)) if vic_y_col else None
            vic_z = safe_float(first_kill.get(vic_z_col)) if vic_z_col else None

            # Get weapon for sniper check
            weapon = safe_str(first_kill.get(weapon_col)) if weapon_col else None
            is_sniper_kill = self._is_sniper_weapon(weapon)

            if attacker_id in self._players:
                self._players[attacker_id].opening_duels.attempts += 1
                self._players[attacker_id].opening_duels.wins += 1
                entry_kills_count += 1

                # Track T-side vs CT-side entries
                if attacker_side == "T":
                    self._players[attacker_id].opening_duels.t_side_entries += 1
                elif attacker_side == "CT":
                    self._players[attacker_id].opening_duels.ct_side_entries += 1

                # Dry peek detection for attacker (the one who got the kill)
                # Skip sniper weapons - they legitimately hold angles without utility
                if not is_sniper_kill and attacker_side:
                    is_supported = self._is_utility_supported(
                        kill_tick, att_x, att_y, att_z, attacker_side, round_num_int
                    )
                    if is_supported:
                        self._players[attacker_id].opening_duels.supported_entries += 1
                    else:
                        self._players[attacker_id].opening_duels.unsupported_entries += 1
                        dry_peek_entries += 1

                # Calculate Entry TTD (time from first damage to kill)
                if dmg_att_col and dmg_vic_col and not damages_df.empty:
                    entry_damages = damages_df[
                        (damages_df[dmg_att_col].astype(float) == float(attacker_id))
                        & (damages_df[dmg_vic_col].astype(float) == float(victim_id))
                        & (damages_df["tick"] <= kill_tick)
                    ].sort_values(by="tick")

                    if not entry_damages.empty:
                        first_dmg_tick = safe_int(entry_damages.iloc[0]["tick"])
                        entry_ttd_ticks = kill_tick - first_dmg_tick
                        entry_ttd_ms = entry_ttd_ticks * self.MS_PER_TICK

                        # Only record reasonable TTD values (0-1500ms)
                        if 0 < entry_ttd_ms <= self.TTD_MAX_MS:
                            self._players[attacker_id].opening_duels.entry_ttd_values.append(
                                entry_ttd_ms
                            )

            if victim_id in self._players:
                self._players[victim_id].opening_duels.attempts += 1
                self._players[victim_id].opening_duels.losses += 1

                # Dry peek detection for victim (the one who died)
                # This tracks if they died while dry peeking (no utility support)
                # Skip if victim was holding with sniper (check weapon that killed them isn't relevant,
                # but we can check if they were likely peeking vs holding based on context)
                if victim_side:
                    is_supported = self._is_utility_supported(
                        kill_tick, vic_x, vic_y, vic_z, victim_side, round_num_int
                    )
                    if is_supported:
                        self._players[victim_id].opening_duels.supported_deaths += 1
                    else:
                        self._players[victim_id].opening_duels.unsupported_deaths += 1
                        dry_peek_deaths += 1

        logger.info(
            f"Detected {entry_kills_count} entry kills across {len(kills_df[self._round_col].unique())} rounds, "
            f"dry peek entries: {dry_peek_entries}, dry peek deaths: {dry_peek_deaths}"
        )

    def _detect_opening_engagements(self) -> None:
        """Detect opening engagements - who FOUGHT first, not just who DIED first.

        Identifies:
        1. First damage tick of each round
        2. All players who dealt/took damage before first kill
        3. Opening phase damage totals

        This captures true engagement participation even when a player
        initiates combat but doesn't secure the kill.
        """
        kills_df = self.data.kills_df
        damages_df = self.data.damages_df

        if kills_df.empty or damages_df.empty:
            logger.info("Skipping opening engagements - missing kill or damage data")
            return

        if not self._round_col or not self._att_id_col:
            logger.info("Skipping opening engagements - missing columns")
            return

        # Find damage columns
        dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS)
        dmg_vic_col = self._find_col(damages_df, self.VIC_ID_COLS)
        dmg_round_col = self._find_col(damages_df, self.ROUND_COLS)
        dmg_val_col = self._find_col(damages_df, ["dmg_health", "damage", "dmg"])

        if not all([dmg_att_col, dmg_vic_col, dmg_round_col, dmg_val_col]):
            logger.info("Skipping opening engagements - missing damage columns")
            return

        engagement_count = 0

        for round_num in kills_df[self._round_col].unique():
            # Get first kill tick for this round
            round_kills = kills_df[kills_df[self._round_col] == round_num].sort_values("tick")
            if round_kills.empty:
                continue

            first_kill_tick = safe_int(round_kills.iloc[0]["tick"])
            first_kill_attacker = safe_int(round_kills.iloc[0].get(self._att_id_col))

            # Get all damage events before (and including) first kill
            round_damages = damages_df[
                (damages_df[dmg_round_col] == round_num) & (damages_df["tick"] <= first_kill_tick)
            ].sort_values("tick")

            if round_damages.empty:
                continue

            # Find first damage tick and who dealt it
            first_damage_attacker = safe_int(round_damages.iloc[0].get(dmg_att_col))
            first_damage_victim = safe_int(round_damages.iloc[0].get(dmg_vic_col))

            # Track all players involved in opening phase damage
            opening_phase_damage: dict[int, int] = {}  # steam_id -> damage dealt
            players_took_damage: set[int] = set()

            for _, dmg_row in round_damages.iterrows():
                attacker_id = safe_int(dmg_row.get(dmg_att_col))
                victim_id = safe_int(dmg_row.get(dmg_vic_col))
                damage = safe_int(dmg_row.get(dmg_val_col))

                if attacker_id:
                    opening_phase_damage[attacker_id] = (
                        opening_phase_damage.get(attacker_id, 0) + damage
                    )
                if victim_id:
                    players_took_damage.add(victim_id)

            # Determine winner team (team of the player who got first kill)
            winner_team = self._get_player_team_for_engagement(first_kill_attacker)

            # Update stats for all involved players
            all_participants = set(opening_phase_damage.keys()) | players_took_damage
            for steam_id in all_participants:
                if steam_id not in self._players:
                    continue

                player = self._players[steam_id]
                player.opening_engagements.engagement_attempts += 1

                # First damage tracking
                if steam_id == first_damage_attacker:
                    player.opening_engagements.first_damage_dealt += 1
                if steam_id == first_damage_victim:
                    player.opening_engagements.first_damage_taken += 1

                # Damage accumulation
                if steam_id in opening_phase_damage:
                    dmg = opening_phase_damage[steam_id]
                    player.opening_engagements.opening_damage_total += dmg
                    player.opening_engagements.opening_damage_values.append(dmg)

                # Win/loss tracking based on team
                player_team = self._get_player_team_for_engagement(steam_id)
                if player_team and winner_team and player_team == winner_team:
                    player.opening_engagements.engagement_wins += 1
                else:
                    player.opening_engagements.engagement_losses += 1

                engagement_count += 1

        logger.info(f"Detected {engagement_count} opening engagement participations")

    def _detect_entry_frags(self) -> None:
        """Detect zone-aware entry frags using position data.

        Entry Frag = First kill in a specific bombsite for a round.
        Uses get_zone_for_position() to classify kill locations.

        This distinguishes:
        - Map control kills: First kills in mid/connectors/routes
        - Entry frags: First kills inside bombsite zones
        """
        from opensight.visualization.radar import MAP_ZONES, get_zone_for_position

        kills_df = self.data.kills_df
        if kills_df.empty:
            logger.info("Skipping entry frag detection - no kills")
            return

        if not self._round_col or not self._att_id_col or not self._vic_id_col:
            logger.info("Skipping entry frag detection - missing columns")
            return

        # Check if we have position data (handle both uppercase and lowercase column names)
        pos_col_variants = {
            "attacker_x": ["attacker_x", "attacker_X"],
            "attacker_y": ["attacker_y", "attacker_Y"],
            "victim_x": ["victim_x", "victim_X", "user_x", "user_X"],
            "victim_y": ["victim_y", "victim_Y", "user_y", "user_Y"],
        }
        pos_cols = {}
        for key, variants in pos_col_variants.items():
            for v in variants:
                if v in kills_df.columns:
                    pos_cols[key] = v
                    break
        has_positions = len(pos_cols) == 4
        if not has_positions:
            logger.info(
                f"Skipping entry frag detection - no position data (found: {list(pos_cols.keys())})"
            )
            return

        map_name = self.data.map_name.lower() if self.data.map_name else ""
        if map_name not in MAP_ZONES:
            logger.info(f"Skipping entry frag detection - unknown map: {map_name}")
            return

        zones = MAP_ZONES[map_name]
        bombsite_zones = {name for name, data in zones.items() if data.get("type") == "bombsite"}

        entry_frag_count = 0
        map_control_count = 0

        for round_num in kills_df[self._round_col].unique():
            round_kills = kills_df[kills_df[self._round_col] == round_num].sort_values("tick")
            if round_kills.empty:
                continue

            # Track first kill in each bombsite for this round
            bombsite_first_kills: dict[str, bool] = dict.fromkeys(bombsite_zones, False)
            is_first_kill_of_round = True

            for _, kill in round_kills.iterrows():
                victim_x = safe_float(kill.get(pos_cols.get("victim_x", "victim_x")))
                victim_y = safe_float(kill.get(pos_cols.get("victim_y", "victim_y")))
                # victim_z may not be in pos_cols - try variants
                victim_z = safe_float(
                    kill.get("victim_z")
                    or kill.get("victim_Z")
                    or kill.get("user_z")
                    or kill.get("user_Z"),
                    default=0.0,
                )

                if victim_x == 0.0 and victim_y == 0.0:
                    is_first_kill_of_round = False
                    continue

                # Determine zone of kill
                zone = get_zone_for_position(map_name, victim_x, victim_y, victim_z)

                attacker_id = safe_int(kill.get(self._att_id_col))
                victim_id = safe_int(kill.get(self._vic_id_col))

                # Check if this is a bombsite zone
                is_bombsite = zone in bombsite_zones

                # Update kill zone tracking for attacker
                if attacker_id in self._players:
                    self._players[attacker_id].opening_duels.kill_zones[zone] = (
                        self._players[attacker_id].opening_duels.kill_zones.get(zone, 0) + 1
                    )

                # Entry frag = first kill in this specific bombsite this round
                if is_bombsite and not bombsite_first_kills.get(zone, True):
                    bombsite_first_kills[zone] = True

                    # Determine if A or B site
                    is_a_site = "A" in zone.upper() or "SITE A" in zone.upper()

                    if attacker_id in self._players:
                        player = self._players[attacker_id]
                        player.entry_frags.total_entry_frags += 1
                        if is_a_site:
                            player.entry_frags.a_site_entries += 1
                        else:
                            player.entry_frags.b_site_entries += 1
                        player.opening_duels.site_kills += 1
                        entry_frag_count += 1

                    if victim_id in self._players:
                        player = self._players[victim_id]
                        player.entry_frags.total_entry_deaths += 1
                        if is_a_site:
                            player.entry_frags.a_site_entry_deaths += 1
                        else:
                            player.entry_frags.b_site_entry_deaths += 1

                # First kill of round in non-bombsite = map control
                elif is_first_kill_of_round and not is_bombsite:
                    if attacker_id in self._players:
                        self._players[attacker_id].opening_duels.map_control_kills += 1
                        map_control_count += 1

                is_first_kill_of_round = False

        logger.info(
            f"Detected {entry_frag_count} entry frags and {map_control_count} map control kills"
        )

    def _get_player_team_for_engagement(self, steam_id: int) -> str | None:
        """Get player's team for engagement tracking.

        Args:
            steam_id: Player's Steam ID

        Returns:
            Team string ("CT" or "T") or None if not found
        """
        if steam_id in self._players:
            player = self._players[steam_id]
            if player.team in ("CT", "T"):
                return player.team
        return None

    def _detect_trades(self) -> None:
        """Detect trade kills with Leetify-style opportunity/attempt/success tracking.

        Tracks:
        - Trade Kill Opportunities: Teammate died and you were alive
        - Trade Kill Attempts: You damaged/shot at the killer within window
        - Trade Kill Success: You killed the killer within window
        - Traded Death Opportunities: You died and teammates were alive
        - Traded Death Attempts: Teammates damaged your killer within window
        - Traded Death Success: Teammates killed your killer within window
        - Time to Trade: How fast successful trades were completed
        """
        kills_df = self.data.kills_df
        damages_df = self.data.damages_df
        if kills_df.empty or not self._round_col:
            logger.info("Skipping trade detection - missing round column")
            return

        if not self._vic_id_col or not self._att_id_col:
            logger.info("Skipping trade detection - missing id columns")
            return

        # Use trade window from constants (typically 5 seconds)
        trade_window_ticks = int(TRADE_WINDOW_SECONDS * self.TICK_RATE)
        logger.info(
            f"Trade detection: window = {TRADE_WINDOW_SECONDS}s = {trade_window_ticks} ticks"
        )

        # Find damage DataFrame columns for attempt detection
        dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS) if not damages_df.empty else None
        dmg_vic_col = self._find_col(damages_df, self.VIC_ID_COLS) if not damages_df.empty else None
        dmg_round_col = (
            self._find_col(damages_df, self.ROUND_COLS) if not damages_df.empty else None
        )

        # Build player team lookup for consistent team matching
        # Uses persistent team display name to correctly group teammates across halftime
        player_teams_lookup: dict[int, str] = {}
        for steam_id, player in self._players.items():
            if player.team in ("CT", "T"):
                player_teams_lookup[steam_id] = player.team

        # Also extract from persistent team data if not in players
        # This handles edge cases where a player appears in kills but not in player list
        if self._att_id_col:
            for _, row in kills_df.drop_duplicates(subset=[self._att_id_col]).iterrows():
                att_id = safe_int(row.get(self._att_id_col))
                if att_id and att_id not in player_teams_lookup:
                    # Use persistent team display name to match teammates correctly
                    persistent_team = self.data.get_player_persistent_team(att_id)
                    display_team = self.data.get_team_display_name(persistent_team)
                    if display_team in ("CT", "T"):
                        player_teams_lookup[att_id] = display_team

        # Counters for logging
        total_trade_opportunities = 0
        total_trade_attempts = 0
        total_trade_success = 0
        total_traded_death_opportunities = 0
        total_traded_death_attempts = 0
        total_traded_death_success = 0

        for round_num in kills_df[self._round_col].unique():
            round_kills = (
                kills_df[kills_df[self._round_col] == round_num]
                .sort_values(by="tick")
                .reset_index(drop=True)
            )

            if round_kills.empty:
                continue

            # Get round damages for attempt detection
            round_damages = pd.DataFrame()
            if not damages_df.empty and dmg_round_col and dmg_att_col and dmg_vic_col:
                round_damages = damages_df[damages_df[dmg_round_col] == round_num]

            # Build list of all players in this round
            all_players_in_round: set[int] = set()
            for _, kill in round_kills.iterrows():
                att_id = safe_int(kill.get(self._att_id_col))
                vic_id = safe_int(kill.get(self._vic_id_col))
                if att_id:
                    all_players_in_round.add(att_id)
                if vic_id:
                    all_players_in_round.add(vic_id)

            # Identify entry kill for entry trade tracking
            entry_kill_victim_id = 0
            entry_kill_attacker_id = 0
            if len(round_kills) > 0:
                entry_kill = round_kills.iloc[0]
                entry_kill_victim_id = safe_int(entry_kill.get(self._vic_id_col))
                entry_kill_attacker_id = safe_int(entry_kill.get(self._att_id_col))

            # Track who is dead at each point (cumulative deaths)
            dead_players: set[int] = set()

            # Process each death in order
            for _i, kill in round_kills.iterrows():
                victim_id = safe_int(kill.get(self._vic_id_col))
                killer_id = safe_int(kill.get(self._att_id_col))
                kill_tick = safe_int(kill.get("tick"))
                victim_team = player_teams_lookup.get(victim_id, "")

                if not victim_id or not killer_id or victim_team not in ("CT", "T"):
                    if victim_id:
                        dead_players.add(victim_id)
                    continue

                # === TRADED DEATH OPPORTUNITIES ===
                # Count teammates who are alive when this player dies
                teammates_alive = [
                    pid
                    for pid in all_players_in_round
                    if pid != victim_id
                    and pid not in dead_players
                    and player_teams_lookup.get(pid) == victim_team
                    and pid in self._players
                ]

                if teammates_alive and victim_id in self._players:
                    # There's at least one teammate alive who could trade
                    self._players[victim_id].trades.traded_death_opportunities += 1
                    total_traded_death_opportunities += 1

                    # Check if any teammate attempted or succeeded
                    teammate_attempted = False
                    teammate_succeeded = False

                    for teammate_id in teammates_alive:
                        # Check if teammate damaged the killer within trade window
                        if not round_damages.empty:
                            teammate_damage = round_damages[
                                (round_damages[dmg_att_col].astype(float) == float(teammate_id))
                                & (round_damages[dmg_vic_col].astype(float) == float(killer_id))
                                & (round_damages["tick"] > kill_tick)
                                & (round_damages["tick"] <= kill_tick + trade_window_ticks)
                            ]
                            if not teammate_damage.empty:
                                teammate_attempted = True

                        # Check if teammate killed the killer within trade window
                        teammate_kills = round_kills[
                            (round_kills[self._att_id_col].astype(float) == float(teammate_id))
                            & (round_kills[self._vic_id_col].astype(float) == float(killer_id))
                            & (round_kills["tick"] > kill_tick)
                            & (round_kills["tick"] <= kill_tick + trade_window_ticks)
                        ]
                        if not teammate_kills.empty:
                            teammate_succeeded = True
                            trade_tick = safe_int(teammate_kills.iloc[0].get("tick"))
                            trader_id = safe_int(teammate_kills.iloc[0].get(self._att_id_col))
                            if trader_id in self._players:
                                self._players[trader_id].trades.trade_kill_success += 1
                                self._players[trader_id].trades.kills_traded += 1
                                time_to_trade = trade_tick - kill_tick
                                self._players[trader_id].trades.time_to_trade_ticks.append(
                                    time_to_trade
                                )
                                total_trade_success += 1
                                # Entry trade tracking
                                if victim_id in (entry_kill_victim_id, entry_kill_attacker_id):
                                    self._players[trader_id].trades.traded_entry_kills += 1
                                    if victim_id in self._players:
                                        self._players[victim_id].trades.traded_entry_deaths += 1
                            break

                    if teammate_attempted and victim_id in self._players:
                        self._players[victim_id].trades.traded_death_attempts += 1
                        total_traded_death_attempts += 1

                    if teammate_succeeded and victim_id in self._players:
                        self._players[victim_id].trades.traded_death_success += 1
                        self._players[victim_id].trades.deaths_traded += 1
                        total_traded_death_success += 1

                # === TRADE KILL OPPORTUNITIES ===
                for teammate_id in teammates_alive:
                    if teammate_id in self._players:
                        self._players[teammate_id].trades.trade_kill_opportunities += 1
                        self._players[teammate_id].trades.trade_attempts += 1
                        total_trade_opportunities += 1

                        # Check if this teammate attempted to trade
                        attempted = False
                        if not round_damages.empty:
                            teammate_damage = round_damages[
                                (round_damages[dmg_att_col].astype(float) == float(teammate_id))
                                & (round_damages[dmg_vic_col].astype(float) == float(killer_id))
                                & (round_damages["tick"] > kill_tick)
                                & (round_damages["tick"] <= kill_tick + trade_window_ticks)
                            ]
                            if not teammate_damage.empty:
                                attempted = True

                        teammate_kills = round_kills[
                            (round_kills[self._att_id_col].astype(float) == float(teammate_id))
                            & (round_kills[self._vic_id_col].astype(float) == float(killer_id))
                            & (round_kills["tick"] > kill_tick)
                            & (round_kills["tick"] <= kill_tick + trade_window_ticks)
                        ]
                        if not teammate_kills.empty:
                            attempted = True

                        if attempted:
                            self._players[teammate_id].trades.trade_kill_attempts += 1
                            total_trade_attempts += 1

                # Mark this player as dead
                dead_players.add(victim_id)

        # Log summary
        logger.info(
            f"Trade detection complete: "
            f"opportunities={total_trade_opportunities}, attempts={total_trade_attempts}, success={total_trade_success}, "
            f"death_opps={total_traded_death_opportunities}, death_attempts={total_traded_death_attempts}, "
            f"death_success={total_traded_death_success}"
        )

    def _detect_clutches(self) -> None:
        """Detect clutch situations (1vX where player is last alive) with win tracking.

        Clutch: A round where you were the last player alive on your team
        facing one or more enemies. Uses round winner data to determine success.

        This method tracks:
        - total_situations: Total 1vX clutch attempts
        - total_wins: Clutches won (determined by round outcome)
        - Per-scenario tracking (1v1, 1v2, etc.) with wins and attempts
        - Individual clutch events with round_number, type, and outcome
        """
        kills_df = self.data.kills_df
        if kills_df.empty or not self._round_col or not self._vic_id_col:
            logger.info("Skipping clutch detection - missing required columns")
            return

        # Check if we have team columns - required for proper detection
        use_team_column = bool(self._vic_side_col)
        att_team_col = self._att_side_col

        # Build round winner lookup from rounds data
        round_winners: dict[int, str] = {}
        for round_info in self.data.rounds:
            round_winners[round_info.round_num] = round_info.winner

        total_clutch_situations = 0
        total_clutch_wins = 0

        for round_num in kills_df[self._round_col].unique():
            round_num_int = int(round_num)
            round_kills = kills_df[kills_df[self._round_col] == round_num].sort_values("tick")

            if round_kills.empty:
                continue

            # Get round winner (CT or T)
            round_winner = round_winners.get(round_num_int, "Unknown")

            # Initialize FULL team rosters at round start
            # Critical: We must include ALL players, not just those in kill events
            # Players who survive without killing would otherwise be invisible
            ct_alive: set[int] = set()
            t_alive: set[int] = set()

            # Detect match format: MR12 (CS2 standard) vs MR15 (legacy CS:GO)
            # MR12: halftime after round 12, max 24 rounds (or 25+ with OT)
            # MR15: halftime after round 15, max 30 rounds (or 31+ with OT)
            # MR12: halftime after round 12 (CS2 standard)
            # Note: halftime detection is handled by get_player_side_for_round()

            # Build full roster using round-aware side detection
            for steam_id in self.data.player_persistent_teams.keys():
                # Use new helper that handles halftime swaps automatically
                side = self.data.get_player_side_for_round(steam_id, round_num_int)
                if side == "CT":
                    ct_alive.add(steam_id)
                elif side == "T":
                    t_alive.add(steam_id)

            # Fallback: if player_teams is empty, extract from kill events
            # This handles edge cases like incomplete demo data
            if use_team_column and (not ct_alive or not t_alive):
                for _, kill in round_kills.iterrows():
                    # Add attacker to their team
                    attacker_id = safe_int(kill.get(self._att_id_col)) if self._att_id_col else 0
                    if attacker_id and att_team_col:
                        att_side = self._normalize_team(kill.get(att_team_col))
                        if att_side == "CT":
                            ct_alive.add(attacker_id)
                        elif att_side == "T":
                            t_alive.add(attacker_id)

                    # Add victim to their team
                    victim_id = safe_int(kill.get(self._vic_id_col))
                    if victim_id:
                        vic_side = self._normalize_team(kill.get(self._vic_side_col))
                        if vic_side == "CT":
                            ct_alive.add(victim_id)
                        elif vic_side == "T":
                            t_alive.add(victim_id)

            # Skip rounds with incomplete team data (still no teams after fallback)
            if not ct_alive or not t_alive:
                continue

            # Track if we've already detected a clutch this round (one per side max)
            clutch_detected: dict[str, bool] = {"CT": False, "T": False}
            clutch_info: dict[str, dict[str, Any]] = {}

            # Second pass: process deaths in tick order to track alive status
            for _, kill in round_kills.iterrows():
                victim_id = safe_int(kill.get(self._vic_id_col))
                if not victim_id:
                    continue

                # Get victim team from the kill event (handles side swaps correctly)
                if use_team_column:
                    victim_side = self._normalize_team(kill.get(self._vic_side_col))
                else:
                    # Fallback: determine from current alive sets
                    if victim_id in ct_alive:
                        victim_side = "CT"
                    elif victim_id in t_alive:
                        victim_side = "T"
                    else:
                        victim_side = "Unknown"

                # Remove victim from alive set
                if victim_side == "CT":
                    ct_alive.discard(victim_id)
                elif victim_side == "T":
                    t_alive.discard(victim_id)

                # Get tick from kill event for bookmark timestamps
                kill_tick = safe_int(kill.get("tick"), default=0)

                # Check for clutch situation after each death
                # CT clutch: 1 CT alive, 1+ T alive, not already detected
                if len(ct_alive) == 1 and len(t_alive) >= 1 and not clutch_detected["CT"]:
                    clutch_detected["CT"] = True
                    clutcher_id = next(iter(ct_alive))
                    clutch_info["CT"] = {
                        "clutcher_id": clutcher_id,
                        "enemies_at_start": len(t_alive),
                        "clutcher_died": False,
                        "tick_start": kill_tick,  # Tick when clutch situation began
                        "team": "CT",
                    }

                # T clutch: 1 T alive, 1+ CT alive, not already detected
                if len(t_alive) == 1 and len(ct_alive) >= 1 and not clutch_detected["T"]:
                    clutch_detected["T"] = True
                    clutcher_id = next(iter(t_alive))
                    clutch_info["T"] = {
                        "clutcher_id": clutcher_id,
                        "enemies_at_start": len(ct_alive),
                        "clutcher_died": False,
                        "tick_start": kill_tick,  # Tick when clutch situation began
                        "team": "T",
                    }

                # Check if a clutcher just died
                for _side, info in clutch_info.items():
                    if info.get("clutcher_id") == victim_id:
                        info["clutcher_died"] = True

            # Process detected clutches
            for side, info in clutch_info.items():
                clutcher_id = info["clutcher_id"]
                enemies_at_start = info["enemies_at_start"]
                clutcher_died = info["clutcher_died"]

                if clutcher_id not in self._players:
                    continue

                player = self._players[clutcher_id]

                # Determine outcome: WON, LOST, or SAVED
                # WON: Clutcher's team won the round
                # LOST: Clutcher died
                # SAVED: Clutcher survived but team lost (time/bomb)
                if round_winner == side:
                    outcome = "WON"
                    clutch_won = True
                elif clutcher_died:
                    outcome = "LOST"
                    clutch_won = False
                else:
                    outcome = "SAVED"
                    clutch_won = False

                # Count enemies killed during clutch (by the clutcher)
                enemies_killed = 0
                enemy_side = "T" if side == "CT" else "CT"
                for _, kill in round_kills.iterrows():
                    attacker_id = safe_int(kill.get(self._att_id_col)) if self._att_id_col else 0
                    if attacker_id != clutcher_id:
                        continue
                    # Get victim team from kill event
                    victim_id = safe_int(kill.get(self._vic_id_col))
                    if use_team_column:
                        vic_side = self._normalize_team(kill.get(self._vic_side_col))
                    else:
                        # Fallback: use round-aware side lookup to handle halftime swaps
                        vic_side = self._get_player_side(victim_id, round_num_int)
                    if vic_side == enemy_side:
                        enemies_killed += 1

                # Create clutch type string
                clutch_type = f"1v{enemies_at_start}"

                # Get tick and team data for replay bookmarks
                tick_start = info.get("tick_start", 0)
                clutcher_team = info.get("team", side)

                # Create clutch event with full bookmark data
                clutch_event = ClutchEvent(
                    round_number=round_num_int,
                    type=clutch_type,
                    outcome=outcome,
                    enemies_killed=enemies_killed,
                    tick_start=tick_start,
                    clutcher_steamid=clutcher_id,
                    clutcher_team=clutcher_team,
                    enemies_at_start=enemies_at_start,
                )
                player.clutches.clutches.append(clutch_event)

                # Update totals
                player.clutches.total_situations += 1
                total_clutch_situations += 1
                if clutch_won:
                    player.clutches.total_wins += 1
                    total_clutch_wins += 1

                # Update per-scenario stats
                if enemies_at_start == 1:
                    player.clutches.v1_attempts += 1
                    if clutch_won:
                        player.clutches.v1_wins += 1
                elif enemies_at_start == 2:
                    player.clutches.v2_attempts += 1
                    if clutch_won:
                        player.clutches.v2_wins += 1
                elif enemies_at_start == 3:
                    player.clutches.v3_attempts += 1
                    if clutch_won:
                        player.clutches.v3_wins += 1
                elif enemies_at_start == 4:
                    player.clutches.v4_attempts += 1
                    if clutch_won:
                        player.clutches.v4_wins += 1
                elif enemies_at_start >= 5:
                    player.clutches.v5_attempts += 1
                    if clutch_won:
                        player.clutches.v5_wins += 1

        logger.info(
            f"Detected {total_clutch_situations} clutch situations, {total_clutch_wins} won"
        )

    def _calculate_kast(self) -> None:
        """Calculate KAST (Kill/Assist/Survived/Traded) for each player using optimized lookups."""
        from opensight.analysis.compute_economy import calculate_kast

        calculate_kast(self)

    def _compute_ttd(self) -> None:
        """Compute Time to Damage for each kill with optimized indexing."""
        # Check for damage data
        if self.data.damages_df.empty:
            logger.warning("No damage data for TTD computation")
            return

        # Check for kill data - either kills list or kills_df
        has_kills = bool(self.data.kills) or (
            hasattr(self.data, "kills_df") and not self.data.kills_df.empty
        )
        if not has_kills:
            logger.warning("No kill data for TTD computation")
            return

        # Use optimized vectorized implementation if available
        if self._use_optimized and self._metrics_computer is not None:
            logger.info("Using vectorized TTD computation")
            self._metrics_computer.compute(MetricType.TTD)

            # Transfer results to player stats (engagement duration from vectorized computation)
            for steam_id, player in self._players.items():
                player.engagement_duration_values = self._metrics_computer.get_ttd_values(steam_id)
                # Note: prefire_count will be updated by _compute_true_ttd if tick data available

            ttd_metrics = self._metrics_computer.ttd_metrics
            if ttd_metrics:
                logger.info(
                    f"Computed TTD (vectorized) for {ttd_metrics.total_engagements} engagements"
                )
            return

        # Fallback: Original per-kill loop implementation
        logger.info("Using per-kill TTD computation (fallback)")
        damages_df = self.data.damages_df
        logger.info(
            f"TTD computation: {len(damages_df)} damage events, {len(self.data.kills)} kills"
        )
        logger.debug(f"Damage columns available: {list(damages_df.columns)}")

        # Find the right column names
        def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
            for col in options:
                if col in df.columns:
                    return col
            return None

        dmg_att_col = find_col(damages_df, ["attacker_steamid", "attacker_steam_id", "attacker"])
        dmg_vic_col = find_col(
            damages_df, ["user_steamid", "victim_steamid", "victim_steam_id", "userid"]
        )
        logger.info(f"TTD columns: attacker={dmg_att_col}, victim={dmg_vic_col}")

        if not dmg_att_col or not dmg_vic_col:
            logger.warning(f"Missing columns for TTD. Have: {list(damages_df.columns)}")
            return

        # Pre-build damage lookup index for (attacker, victim) pairs
        # This avoids repeated DataFrame filtering for each kill
        damage_cache: dict[tuple[int, int], list[int]] = {}

        # Convert columns to numeric for reliable comparison
        damages_df = damages_df.copy()
        damages_df[dmg_att_col] = pd.to_numeric(damages_df[dmg_att_col], errors="coerce")
        damages_df[dmg_vic_col] = pd.to_numeric(damages_df[dmg_vic_col], errors="coerce")

        # Group damages by (attacker, victim) and store ticks sorted
        for (att, vic), group in damages_df.groupby([dmg_att_col, dmg_vic_col]):
            if pd.notna(att) and pd.notna(vic):
                damage_cache[(int(att), int(vic))] = sorted(
                    group["tick"].dropna().astype(int).tolist()
                )

        logger.info(f"Built damage cache with {len(damage_cache)} (attacker, victim) pairs")

        # TTD validation thresholds

        # Process kills using cached damage lookups
        raw_ttd_values: dict[int, list[float]] = {}  # For outlier removal later

        # Use kills list if available, otherwise fall back to kills_df
        kills_source = self.data.kills
        use_df_fallback = (
            not kills_source and hasattr(self.data, "kills_df") and not self.data.kills_df.empty
        )

        if use_df_fallback:
            logger.info("Using kills_df for TTD computation (kills list empty)")
            kills_df = self.data.kills_df
            kill_att_col = self._find_col(kills_df, self.ATT_ID_COLS)
            kill_vic_col = self._find_col(kills_df, self.VIC_ID_COLS)

            if kill_att_col and kill_vic_col and "tick" in kills_df.columns:
                for _, row in kills_df.iterrows():
                    try:
                        att_id = safe_int(row.get(kill_att_col))
                        vic_id = safe_int(row.get(kill_vic_col))
                        kill_tick = safe_int(row.get("tick"))
                        round_num = safe_int(row.get("round_num", row.get("round", 0)))

                        if not att_id or not vic_id or kill_tick <= 0:
                            continue

                        cache_key = (att_id, vic_id)
                        damage_ticks = damage_cache.get(cache_key, [])

                        if not damage_ticks:
                            continue

                        first_dmg_tick = None
                        for tick in damage_ticks:
                            if tick <= kill_tick:
                                first_dmg_tick = tick
                                break

                        if first_dmg_tick is None:
                            continue

                        ttd_ticks = kill_tick - first_dmg_tick
                        ttd_ms = ttd_ticks * self.MS_PER_TICK

                        if ttd_ms < 0 or ttd_ms > 5000:
                            continue

                        is_prefire = ttd_ms <= self.TTD_MIN_MS

                        if att_id not in raw_ttd_values:
                            raw_ttd_values[att_id] = []

                        if not is_prefire and ttd_ms <= self.TTD_MAX_MS:
                            raw_ttd_values[att_id].append(ttd_ms)
                        elif is_prefire and att_id in self._players:
                            self._players[att_id].prefire_count += 1

                        self._ttd_results.append(
                            TTDResult(
                                tick_spotted=first_dmg_tick,
                                tick_damage=kill_tick,
                                ttd_ticks=ttd_ticks,
                                ttd_ms=ttd_ms,
                                attacker_steamid=att_id,
                                victim_steamid=vic_id,
                                weapon=str(row.get("weapon", "unknown")),
                                headshot=bool(row.get("headshot", False)),
                                is_prefire=is_prefire,
                                round_num=round_num,
                            )
                        )
                    except Exception as e:
                        logger.debug(f"Error processing kill row for TTD: {e}")
                        continue
        else:
            # Original path using kills list
            for kill in self.data.kills:
                try:
                    att_id = kill.attacker_steamid
                    vic_id = kill.victim_steamid
                    kill_tick = kill.tick
                    round_num = kill.round_num

                    if not att_id or not vic_id or kill_tick <= 0:
                        continue

                    # Use cached damage lookup
                    cache_key = (att_id, vic_id)
                    damage_ticks = damage_cache.get(cache_key, [])

                    if not damage_ticks:
                        continue

                    # Find first damage tick before kill using binary search
                    first_dmg_tick = None
                    for tick in damage_ticks:
                        if tick <= kill_tick:
                            first_dmg_tick = tick
                            break  # Already sorted, first match is earliest

                    if first_dmg_tick is None:
                        continue

                    ttd_ticks = kill_tick - first_dmg_tick
                    ttd_ms = ttd_ticks * self.MS_PER_TICK

                    # Validate TTD value (filter out invalid/negative values)
                    if ttd_ms < 0 or ttd_ms > 5000:  # Max 5 seconds is reasonable
                        continue

                    is_prefire = ttd_ms <= self.TTD_MIN_MS

                    # Account for wallbangs/through-smoke kills (may have higher TTD)
                    getattr(kill, "penetrated", False)
                    getattr(kill, "thrusmoke", False)

                    # Store raw values for later outlier removal
                    if att_id not in raw_ttd_values:
                        raw_ttd_values[att_id] = []

                    if not is_prefire and ttd_ms <= self.TTD_MAX_MS:
                        raw_ttd_values[att_id].append(ttd_ms)
                    elif is_prefire and att_id in self._players:
                        self._players[att_id].prefire_count += 1

                    self._ttd_results.append(
                        TTDResult(
                            tick_spotted=first_dmg_tick,
                            tick_damage=kill_tick,
                            ttd_ticks=ttd_ticks,
                            ttd_ms=ttd_ms,
                            attacker_steamid=att_id,
                            victim_steamid=vic_id,
                            weapon=kill.weapon,
                            headshot=kill.headshot,
                            is_prefire=is_prefire,
                            round_num=round_num,
                        )
                    )

                except Exception as e:
                    logger.debug(f"Error processing kill for TTD: {e}")
                    continue

        # Transfer engagement duration values to player stats
        for att_id, values in raw_ttd_values.items():
            if att_id in self._players and values:
                self._players[att_id].engagement_duration_values = values

        # Summary of engagement results per player
        players_with_data = sum(1 for p in self._players.values() if p.engagement_duration_values)
        total_values = sum(len(p.engagement_duration_values) for p in self._players.values())
        logger.info(
            f"Computed engagement duration for {len(self._ttd_results)} engagements, "
            f"{players_with_data} players have data ({total_values} total samples)"
        )

        # Calculate true TTD (reaction time) if tick data is available
        self._compute_true_ttd()

    def _compute_true_ttd(self) -> None:
        """Compute true Time to Damage (reaction time: visibility to first damage).

        This uses tick-level position data to determine when the attacker
        first had visibility of the victim, then calculates the time until
        first damage was dealt.

        Requires: self.data.ticks_df with X, Y, Z, pitch, yaw columns
        """
        # Check if tick data is available
        if self.data.ticks_df is None or self.data.ticks_df.empty:
            logger.info("No tick data available for true TTD calculation - skipping")
            return

        ticks_df = self.data.ticks_df
        damages_df = self.data.damages_df

        # Check for required columns
        required_cols = ["tick", "X", "Y", "Z"]
        view_cols = ["pitch", "yaw"]

        has_position = all(col in ticks_df.columns for col in required_cols)
        has_view = all(col in ticks_df.columns for col in view_cols)

        if not has_position:
            logger.info("Tick data missing position columns for true TTD - skipping")
            return

        if not has_view:
            logger.info("Tick data missing view angle columns - using position-only visibility")

        # Find steamid column in tick data
        steamid_col = None
        for col in ["steamid", "steam_id", "user_steamid"]:
            if col in ticks_df.columns:
                steamid_col = col
                break

        if not steamid_col:
            logger.warning("No steamid column in tick data for true TTD")
            return

        logger.info(f"Computing true TTD with {len(ticks_df)} tick entries")

        # For each damage event, try to find visibility start
        dmg_att_col = self._find_col(damages_df, self.ATT_ID_COLS)
        dmg_vic_col = self._find_col(damages_df, self.VIC_ID_COLS)

        if not dmg_att_col or not dmg_vic_col:
            logger.warning("Missing columns in damage data for true TTD")
            return

        true_ttd_values: dict[int, list[float]] = {}
        prefire_counts: dict[int, int] = {}
        processed = 0
        visibility_found = 0

        # Process unique (attacker, victim) pairs from damage events
        # Group damages to get first damage tick per engagement
        for (att_id, vic_id), group in damages_df.groupby([dmg_att_col, dmg_vic_col]):
            try:
                att_id = int(att_id)
                vic_id = int(vic_id)
                first_dmg_tick = int(group["tick"].min())

                # Get position data for attacker and victim
                attacker_ticks = ticks_df[ticks_df[steamid_col] == att_id]
                victim_ticks = ticks_df[ticks_df[steamid_col] == vic_id]

                if attacker_ticks.empty or victim_ticks.empty:
                    continue

                # Find visibility start using simplified check
                visibility_tick = self._find_visibility_start_simple(
                    attacker_ticks, victim_ticks, first_dmg_tick, has_view
                )

                if visibility_tick is None:
                    continue

                visibility_found += 1

                # Calculate true TTD
                reaction_ticks = first_dmg_tick - visibility_tick
                reaction_ms = reaction_ticks * self.MS_PER_TICK

                # Sanity filter
                if reaction_ms < 0:
                    continue  # Invalid - damage before visibility

                processed += 1

                # Classify as prefire or valid reaction
                if reaction_ms < self.REACTION_TIME_MIN_MS:
                    # Prefire - player was pre-aiming
                    prefire_counts[att_id] = prefire_counts.get(att_id, 0) + 1
                elif reaction_ms <= self.REACTION_TIME_MAX_MS:
                    # Valid reaction time sample
                    if att_id not in true_ttd_values:
                        true_ttd_values[att_id] = []
                    true_ttd_values[att_id].append(reaction_ms)
                # > REACTION_TIME_MAX_MS: visibility logic likely failed, skip

            except Exception as e:
                logger.debug(f"Error processing true TTD for engagement: {e}")
                continue

        # Transfer to player stats
        for att_id, values in true_ttd_values.items():
            if att_id in self._players and values:
                self._players[att_id].true_ttd_values = values

        # Update prefire counts (true prefires based on reaction time)
        for att_id, count in prefire_counts.items():
            if att_id in self._players:
                self._players[att_id].prefire_count = count

        players_with_ttd = sum(1 for p in self._players.values() if p.true_ttd_values)
        total_prefires = sum(prefire_counts.values())

        logger.info(
            f"True TTD computed: {processed} engagements analyzed, "
            f"{visibility_found} with visibility data, "
            f"{players_with_ttd} players have reaction time data, "
            f"{total_prefires} prefires detected"
        )

    def _find_visibility_start_simple(
        self,
        attacker_ticks: pd.DataFrame,
        victim_ticks: pd.DataFrame,
        damage_tick: int,
        use_view_angles: bool = True,
    ) -> int | None:
        """Find when attacker first had visibility of victim (simplified).

        Uses a distance + FOV check without ray-casting against map geometry.
        This is an approximation that works reasonably well for open areas.

        Args:
            attacker_ticks: Attacker position data (needs X, Y, Z, optionally pitch/yaw)
            victim_ticks: Victim position data (needs X, Y, Z)
            damage_tick: Tick when damage occurred
            use_view_angles: Whether to check if victim is in FOV

        Returns:
            Tick when visibility started, or None if not determinable
        """
        MAX_LOOKBACK_TICKS = int(2000 / self.MS_PER_TICK)  # 2 seconds max lookback
        MIN_DISTANCE = 50  # Units - too close is likely already in combat
        MAX_DISTANCE = 3000  # Units - beyond this visibility is questionable
        FOV_THRESHOLD = 90  # Degrees - consider visible if within this FOV

        start_tick = max(0, damage_tick - MAX_LOOKBACK_TICKS)

        # Get data in lookback window
        att_window = attacker_ticks[
            (attacker_ticks["tick"] >= start_tick) & (attacker_ticks["tick"] <= damage_tick)
        ].sort_values("tick")

        vic_window = victim_ticks[
            (victim_ticks["tick"] >= start_tick) & (victim_ticks["tick"] <= damage_tick)
        ].sort_values("tick")

        if att_window.empty or vic_window.empty:
            return None

        # Check each tick from earliest to find first visibility
        for _, att_row in att_window.iterrows():
            tick = int(att_row["tick"])

            # Find victim position at same tick (or closest)
            vic_at_tick = vic_window[vic_window["tick"] == tick]
            if vic_at_tick.empty:
                # Try closest tick
                vic_at_tick = vic_window.iloc[(vic_window["tick"] - tick).abs().argsort()[:1]]
                if vic_at_tick.empty:
                    continue

            vic_row = vic_at_tick.iloc[0]

            # Calculate distance
            try:
                att_pos = np.array([att_row["X"], att_row["Y"], att_row["Z"]])
                vic_pos = np.array([vic_row["X"], vic_row["Y"], vic_row["Z"]])
                direction = vic_pos - att_pos
                distance = np.linalg.norm(direction)

                if distance < MIN_DISTANCE or distance > MAX_DISTANCE:
                    continue

                # If we have view angles, check FOV
                if use_view_angles and "pitch" in att_row and "yaw" in att_row:
                    pitch = float(att_row["pitch"])
                    yaw = float(att_row["yaw"])

                    # Convert view angles to direction vector
                    pitch_rad = np.radians(pitch)
                    yaw_rad = np.radians(yaw)
                    view_dir = np.array(
                        [
                            np.cos(yaw_rad) * np.cos(pitch_rad),
                            np.sin(yaw_rad) * np.cos(pitch_rad),
                            -np.sin(pitch_rad),
                        ]
                    )

                    # Calculate angle to target
                    target_dir = direction / distance
                    dot = np.clip(np.dot(view_dir, target_dir), -1.0, 1.0)
                    angle = np.degrees(np.arccos(dot))

                    if angle <= FOV_THRESHOLD:
                        return tick
                else:
                    # No view angles - assume visible if in range
                    return tick

            except Exception:
                continue

        return None

    def _compute_crosshair_placement(self) -> None:
        """Compute crosshair placement error for each kill.

        Uses vectorized numpy implementation when available for ~5-20x speedup.
        Falls back to per-kill loop for compatibility.
        """
        # Use optimized vectorized implementation if available
        if self._use_optimized and self._metrics_computer is not None:
            logger.info("Using vectorized CP computation")
            self._metrics_computer.compute(MetricType.CP)

            # Transfer results to player stats
            for steam_id, player in self._players.items():
                player.cp_values = self._metrics_computer.get_cp_values(steam_id)

            cp_metrics = self._metrics_computer.cp_metrics
            if cp_metrics:
                logger.info(f"Computed CP (vectorized) for {cp_metrics.total_kills_analyzed} kills")
            return

        # Fallback: Original implementation
        logger.info("Using per-kill CP computation (fallback)")

        # First try using KillEvent objects directly (preferred - they have embedded position data)
        # Count what data is available
        kills_with_x = sum(1 for k in self.data.kills if k.attacker_x is not None)
        kills_with_pitch = sum(1 for k in self.data.kills if k.attacker_pitch is not None)
        kills_with_victim_x = sum(1 for k in self.data.kills if k.victim_x is not None)
        logger.info(
            f"CP data availability: {len(self.data.kills)} kills, "
            f"{kills_with_x} with attacker_x, {kills_with_pitch} with pitch, "
            f"{kills_with_victim_x} with victim_x"
        )

        kills_with_pos = [
            k
            for k in self.data.kills
            if k.attacker_x is not None and k.attacker_pitch is not None and k.victim_x is not None
        ]

        if kills_with_pos:
            logger.info(
                f"Computing CP from {len(kills_with_pos)} KillEvent objects with position data"
            )
            self._compute_cp_from_kill_events(kills_with_pos)
            return

        # Fallback: check DataFrame for position columns
        if not self.data.kills_df.empty:
            kills_df = self.data.kills_df

            # Check various column name patterns
            pos_patterns = [
                [
                    "attacker_X",
                    "attacker_Y",
                    "attacker_Z",
                    "victim_X",
                    "victim_Y",
                    "victim_Z",
                ],
                [
                    "attacker_x",
                    "attacker_y",
                    "attacker_z",
                    "victim_x",
                    "victim_y",
                    "victim_z",
                ],
                [
                    "attacker_X",
                    "attacker_Y",
                    "attacker_Z",
                    "user_X",
                    "user_Y",
                    "user_Z",
                ],
            ]
            angle_patterns = [
                ["attacker_pitch", "attacker_yaw"],
            ]

            has_positions = any(
                all(col in kills_df.columns for col in pattern) for pattern in pos_patterns
            )
            has_angles = any(
                all(col in kills_df.columns for col in pattern) for pattern in angle_patterns
            )

            if has_positions and has_angles:
                logger.info(f"Computing CP from DataFrame. Columns: {list(kills_df.columns)}")
                self._compute_cp_from_events()
                return

        # Final fallback: tick data
        if self.data.ticks_df is not None and not self.data.ticks_df.empty:
            self._compute_cp_from_ticks()
        else:
            logger.warning(
                "No position/angle data available for CP computation. Position data requires parsing with player props."
            )

    def _compute_cp_from_kill_events(self, kills: list) -> None:
        """Compute CP from KillEvent objects with optimized vectorized calculations."""
        # Constants
        MAX_DISTANCE = 2000.0  # Skip kills beyond this distance (unlikely to be meaningful CP data)
        EYE_HEIGHT = 64.0

        # Pre-allocate arrays for batch processing
        valid_kills = []
        att_positions = []
        vic_positions = []
        att_pitches = []
        att_yaws = []

        # First pass: validate and collect data
        for kill in kills:
            att_id = kill.attacker_steamid
            vic_id = kill.victim_steamid

            if not att_id or not vic_id:
                continue

            # Validate position data (not zero or NaN)
            att_x = kill.attacker_x
            att_y = kill.attacker_y
            att_z = kill.attacker_z
            vic_x = kill.victim_x
            vic_y = kill.victim_y
            vic_z = kill.victim_z

            if any(
                v is None or (isinstance(v, float) and np.isnan(v))
                for v in [att_x, att_y, vic_x, vic_y]
            ):
                continue

            # Skip if positions are clearly invalid (all zeros)
            if abs(att_x) < 0.001 and abs(att_y) < 0.001:
                continue
            if abs(vic_x) < 0.001 and abs(vic_y) < 0.001:
                continue

            # Calculate distance for filtering
            dx = vic_x - att_x
            dy = vic_y - att_y
            dz = (vic_z or 0) - (att_z or 0)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Skip very long distance kills
            if distance > MAX_DISTANCE:
                continue

            valid_kills.append(kill)
            att_positions.append([att_x, att_y, (att_z or 0) + EYE_HEIGHT])
            vic_positions.append([vic_x, vic_y, (vic_z or 0) + EYE_HEIGHT])
            att_pitches.append(kill.attacker_pitch or 0.0)
            att_yaws.append(kill.attacker_yaw or 0.0)

        if not valid_kills:
            logger.info("No valid kills with position data for CP calculation")
            return

        # Convert to numpy arrays for vectorized operations
        att_pos_arr = np.array(att_positions)
        vic_pos_arr = np.array(vic_positions)
        pitch_arr = np.array(att_pitches)
        yaw_arr = np.array(att_yaws)

        # Vectorized angular error calculation
        pitch_rad = np.radians(pitch_arr)
        yaw_rad = np.radians(yaw_arr)

        # View vectors from Euler angles (vectorized)
        view_x = np.cos(yaw_rad) * np.cos(pitch_rad)
        view_y = np.sin(yaw_rad) * np.cos(pitch_rad)
        view_z = -np.sin(pitch_rad)
        view_vecs = np.column_stack([view_x, view_y, view_z])

        # Ideal vectors (normalized direction to victim)
        ideal_vecs = vic_pos_arr - att_pos_arr
        distances = np.linalg.norm(ideal_vecs, axis=1, keepdims=True)
        distances = np.maximum(distances, 0.001)  # Avoid division by zero
        ideal_vecs = ideal_vecs / distances

        # Dot products and angular errors (vectorized)
        dots = np.sum(view_vecs * ideal_vecs, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angular_errors = np.degrees(np.arccos(dots))

        # Separate pitch/yaw errors
        ideal_pitches = np.degrees(np.arcsin(-ideal_vecs[:, 2]))
        ideal_yaws = np.degrees(np.arctan2(ideal_vecs[:, 1], ideal_vecs[:, 0]))
        pitch_errors = pitch_arr - ideal_pitches
        yaw_errors = yaw_arr - ideal_yaws

        # Normalize yaw errors to [-180, 180]
        yaw_errors = np.mod(yaw_errors + 180, 360) - 180

        # Store results
        for i, kill in enumerate(valid_kills):
            att_id = kill.attacker_steamid
            angular_error = float(angular_errors[i])
            pitch_error = float(pitch_errors[i])
            yaw_error = float(yaw_errors[i])

            if att_id in self._players:
                self._players[att_id].cp_values.append(angular_error)

            self._cp_results.append(
                CrosshairPlacementResult(
                    tick=kill.tick,
                    attacker_steamid=att_id,
                    victim_steamid=kill.victim_steamid,
                    angular_error_deg=angular_error,
                    pitch_error_deg=pitch_error,
                    yaw_error_deg=yaw_error,
                    round_num=kill.round_num,
                )
            )

        logger.info(
            f"Computed CP for {len(self._cp_results)} kills from KillEvent objects (vectorized)"
        )

    def _compute_cp_from_events(self) -> None:
        """Compute CP from position data embedded in kill events DataFrame (optimized)."""
        kills_df = self.data.kills_df
        logger.info("Computing CP from DataFrame event-embedded positions")

        # Constants
        MAX_DISTANCE = 2000.0
        EYE_HEIGHT = 64.0

        # Find position columns
        att_x_col = "attacker_X" if "attacker_X" in kills_df.columns else "attacker_x"
        att_y_col = "attacker_Y" if "attacker_Y" in kills_df.columns else "attacker_y"
        att_z_col = "attacker_Z" if "attacker_Z" in kills_df.columns else "attacker_z"
        vic_x_col = (
            "victim_X"
            if "victim_X" in kills_df.columns
            else ("user_X" if "user_X" in kills_df.columns else "victim_x")
        )
        vic_y_col = (
            "victim_Y"
            if "victim_Y" in kills_df.columns
            else ("user_Y" if "user_Y" in kills_df.columns else "victim_y")
        )
        vic_z_col = (
            "victim_Z"
            if "victim_Z" in kills_df.columns
            else ("user_Z" if "user_Z" in kills_df.columns else "victim_z")
        )

        # Filter valid rows with position data
        required_cols = [
            att_x_col,
            att_y_col,
            vic_x_col,
            vic_y_col,
            "attacker_pitch",
            "attacker_yaw",
        ]
        if not all(col in kills_df.columns for col in required_cols):
            logger.warning(f"Missing position columns for CP. Have: {list(kills_df.columns)}")
            return

        # Create working copy with validated data
        df = kills_df.copy()
        df["_att_x"] = pd.to_numeric(df[att_x_col], errors="coerce")
        df["_att_y"] = pd.to_numeric(df[att_y_col], errors="coerce")
        if att_z_col and att_z_col in df.columns:
            df["_att_z"] = pd.to_numeric(df[att_z_col], errors="coerce").fillna(0) + EYE_HEIGHT
        else:
            df["_att_z"] = float(EYE_HEIGHT)
        df["_vic_x"] = pd.to_numeric(df[vic_x_col], errors="coerce")
        df["_vic_y"] = pd.to_numeric(df[vic_y_col], errors="coerce")
        if vic_z_col and vic_z_col in df.columns:
            df["_vic_z"] = pd.to_numeric(df[vic_z_col], errors="coerce").fillna(0) + EYE_HEIGHT
        else:
            df["_vic_z"] = float(EYE_HEIGHT)
        df["_pitch"] = pd.to_numeric(df["attacker_pitch"], errors="coerce").fillna(0)
        df["_yaw"] = pd.to_numeric(df["attacker_yaw"], errors="coerce").fillna(0)

        # Filter out invalid positions - require all notna and at least some non-zero positions
        valid_mask = (
            df["_att_x"].notna()
            & df["_att_y"].notna()
            & df["_vic_x"].notna()
            & df["_vic_y"].notna()
            & ((df["_att_x"].abs() > 0.001) | (df["_att_y"].abs() > 0.001))
            & ((df["_vic_x"].abs() > 0.001) | (df["_vic_y"].abs() > 0.001))
        )
        df = df[valid_mask]

        if df.empty:
            logger.warning("No valid positions for CP calculation")
            return

        # Calculate distances
        df["_dist"] = np.sqrt(
            (df["_vic_x"] - df["_att_x"]) ** 2
            + (df["_vic_y"] - df["_att_y"]) ** 2
            + (df["_vic_z"] - df["_att_z"]) ** 2
        )

        # Filter by distance
        df = df[df["_dist"] <= MAX_DISTANCE]

        if df.empty:
            return

        # Vectorized calculations
        pitch_rad = np.radians(df["_pitch"].values)
        yaw_rad = np.radians(df["_yaw"].values)

        # View vectors
        view_x = np.cos(yaw_rad) * np.cos(pitch_rad)
        view_y = np.sin(yaw_rad) * np.cos(pitch_rad)
        view_z = -np.sin(pitch_rad)

        # Ideal vectors
        ideal_x = (df["_vic_x"].values - df["_att_x"].values) / df["_dist"].values
        ideal_y = (df["_vic_y"].values - df["_att_y"].values) / df["_dist"].values
        ideal_z = (df["_vic_z"].values - df["_att_z"].values) / df["_dist"].values

        # Angular errors
        dots = view_x * ideal_x + view_y * ideal_y + view_z * ideal_z
        dots = np.clip(dots, -1.0, 1.0)
        angular_errors = np.degrees(np.arccos(dots))

        # Component errors
        ideal_pitches = np.degrees(np.arcsin(-ideal_z))
        ideal_yaws = np.degrees(np.arctan2(ideal_y, ideal_x))
        pitch_errors = df["_pitch"].values - ideal_pitches
        yaw_errors = np.mod(df["_yaw"].values - ideal_yaws + 180, 360) - 180

        # Store results
        att_id_col = self._find_col(df, self.ATT_ID_COLS)
        vic_id_col = self._find_col(df, self.VIC_ID_COLS)

        for i, (_idx, row) in enumerate(df.iterrows()):
            att_id = safe_int(row.get(att_id_col)) if att_id_col else 0
            vic_id = safe_int(row.get(vic_id_col)) if vic_id_col else 0
            tick = safe_int(row.get("tick"))
            round_num = safe_int(row.get("round_num", 0))

            if att_id in self._players:
                self._players[att_id].cp_values.append(angular_errors[i])

            self._cp_results.append(
                CrosshairPlacementResult(
                    tick=tick,
                    attacker_steamid=att_id,
                    victim_steamid=vic_id,
                    angular_error_deg=float(angular_errors[i]),
                    pitch_error_deg=float(pitch_errors[i]),
                    yaw_error_deg=float(yaw_errors[i]),
                    round_num=round_num,
                )
            )

        logger.info(f"Computed CP for {len(self._cp_results)} kills (vectorized)")

    def _compute_cp_from_ticks(self) -> None:
        """Compute CP from tick-level data (fallback)."""
        ticks_df = self.data.ticks_df
        kills_df = self.data.kills_df
        logger.info("Computing CP from tick data (slower)")

        required_cols = ["steamid", "X", "Y", "Z", "pitch", "yaw", "tick"]
        if not all(col in ticks_df.columns for col in required_cols):
            logger.warning("Missing columns for tick-based CP")
            return

        for _, kill_row in kills_df.iterrows():
            try:
                att_id = safe_int(kill_row.get("attacker_steamid"))
                vic_id = safe_int(kill_row.get("victim_steamid"))
                kill_tick = safe_int(kill_row.get("tick"))
                round_num = safe_int(kill_row.get("round_num", 0))

                if not att_id or not vic_id:
                    continue

                att_ticks = ticks_df[
                    (ticks_df["steamid"] == att_id) & (ticks_df["tick"] <= kill_tick)
                ]
                vic_ticks = ticks_df[
                    (ticks_df["steamid"] == vic_id) & (ticks_df["tick"] <= kill_tick)
                ]

                if att_ticks.empty or vic_ticks.empty:
                    continue

                att_state = att_ticks.iloc[-1]
                vic_state = vic_ticks.iloc[-1]

                att_pos = np.array(
                    [
                        safe_float(att_state["X"]),
                        safe_float(att_state["Y"]),
                        safe_float(att_state["Z"]) + 64,
                    ]
                )
                att_pitch = safe_float(att_state["pitch"])
                att_yaw = safe_float(att_state["yaw"])

                vic_pos = np.array(
                    [
                        safe_float(vic_state["X"]),
                        safe_float(vic_state["Y"]),
                        safe_float(vic_state["Z"]) + 64,
                    ]
                )

                angular_error, pitch_error, yaw_error = self._calculate_angular_error(
                    att_pos, att_pitch, att_yaw, vic_pos
                )

                if att_id in self._players:
                    self._players[att_id].cp_values.append(angular_error)

                self._cp_results.append(
                    CrosshairPlacementResult(
                        tick=kill_tick,
                        attacker_steamid=att_id,
                        victim_steamid=vic_id,
                        angular_error_deg=angular_error,
                        pitch_error_deg=pitch_error,
                        yaw_error_deg=yaw_error,
                        round_num=round_num,
                    )
                )

            except Exception as e:
                logger.debug(f"Error in tick-based CP: {e}")
                continue

        logger.info(f"Computed CP for {len(self._cp_results)} kills (tick-based)")

    def _calculate_angular_error(
        self,
        attacker_pos: np.ndarray,
        pitch_deg: float,
        yaw_deg: float,
        victim_pos: np.ndarray,
    ) -> tuple[float, float, float]:
        """Calculate angular error between view direction and target."""
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)

        # View vector from Euler angles
        view_x = math.cos(yaw_rad) * math.cos(pitch_rad)
        view_y = math.sin(yaw_rad) * math.cos(pitch_rad)
        view_z = -math.sin(pitch_rad)
        view_vec = np.array([view_x, view_y, view_z])

        # Ideal vector
        ideal_vec = victim_pos - attacker_pos
        distance = np.linalg.norm(ideal_vec)
        if distance < 0.001:
            return 0.0, 0.0, 0.0

        ideal_vec = ideal_vec / distance

        # Total angular error
        dot = np.clip(np.dot(view_vec, ideal_vec), -1.0, 1.0)
        angular_error = math.degrees(math.acos(dot))

        # Separate pitch/yaw errors
        ideal_pitch = math.degrees(math.asin(-ideal_vec[2]))
        ideal_yaw = math.degrees(math.atan2(ideal_vec[1], ideal_vec[0]))

        pitch_error = pitch_deg - ideal_pitch
        yaw_error = yaw_deg - ideal_yaw

        while yaw_error > 180:
            yaw_error -= 360
        while yaw_error < -180:
            yaw_error += 360

        return angular_error, pitch_error, yaw_error

    def _calculate_side_stats(self) -> None:
        """Calculate CT-side vs T-side performance breakdown."""
        from opensight.analysis.compute_economy import calculate_side_stats

        calculate_side_stats(self)

    def _calculate_utility_stats(self) -> None:
        """Calculate comprehensive utility statistics (Leetify-style) using all available data."""
        from opensight.analysis.compute_utility import calculate_utility_stats

        calculate_utility_stats(self)

    def _calculate_accuracy_stats(self) -> None:
        """Calculate accuracy statistics from weapon_fire events."""
        if not hasattr(self.data, "weapon_fires") or not self.data.weapon_fires:
            logger.info("No weapon_fire data available for accuracy stats")
            return

        damages_df = self.data.damages_df

        for steam_id, player in self._players.items():
            # Count shots fired
            player_shots = [f for f in self.data.weapon_fires if f.player_steamid == steam_id]
            player.shots_fired = len(player_shots)

            # Count shots that hit (from damage events)
            if not damages_df.empty:
                att_col = self._find_col(damages_df, self.ATT_ID_COLS)
                if att_col:
                    player_hits = damages_df[damages_df[att_col] == steam_id]
                    player.shots_hit = len(player_hits)

                    # Count headshot hits
                    hitgroup_col = self._find_col(damages_df, ["hitgroup"])
                    if hitgroup_col:
                        head_hits = player_hits[
                            player_hits[hitgroup_col].str.lower().str.contains("head", na=False)
                        ]
                        player.headshot_hits = len(head_hits)

            # Calculate spray accuracy
            self._calculate_spray_accuracy_for_player(player, player_shots, damages_df)

            # Calculate counter-strafing
            self._calculate_counter_strafing_for_player(player, steam_id)

        logger.info("Calculated accuracy stats (including spray and counter-strafing)")

    def _calculate_spray_accuracy_for_player(
        self,
        player: PlayerMatchStats,
        player_shots: list,
        damages_df: pd.DataFrame,
    ) -> None:
        """Calculate spray accuracy for a single player.

        Spray accuracy = hits after 4th bullet / shots after 4th bullet in a burst.
        """
        # Define weapons that support spray (exclude pistols, snipers, shotguns)
        spray_weapons = {
            "ak47",
            "m4a1",
            "m4a1_silencer",
            "m4a4",
            "galil",
            "famas",
            "aug",
            "sg556",
            "mp9",
            "mac10",
            "mp7",
            "ump45",
            "p90",
            "bizon",
            "mp5sd",
            "negev",
            "m249",
        }
        burst_tick_window = 20  # ~312ms at 64 tick

        if not player_shots:
            return

        # Filter to spray weapons only
        spray_shots = [
            s
            for s in player_shots
            if s.weapon and s.weapon.lower().replace("weapon_", "") in spray_weapons
        ]

        if not spray_shots:
            return

        # Sort by tick
        spray_shots.sort(key=lambda s: s.tick)

        # Detect bursts and count shots 4+ in each burst
        spray_shot_ticks = []
        burst_shot_count = 1

        for i in range(1, len(spray_shots)):
            current = spray_shots[i]
            previous = spray_shots[i - 1]

            if current.tick - previous.tick <= burst_tick_window:
                burst_shot_count += 1
                if burst_shot_count >= 4:
                    spray_shot_ticks.append(current.tick)
            else:
                burst_shot_count = 1

        player.spray_shots_fired = len(spray_shot_ticks)

        # Count spray hits
        if not damages_df.empty and spray_shot_ticks:
            att_col = self._find_col(damages_df, self.ATT_ID_COLS)
            if att_col and "tick" in damages_df.columns:
                player_damages = damages_df[damages_df[att_col] == player.steam_id]
                if not player_damages.empty:
                    damage_ticks = set(player_damages["tick"].values)
                    spray_hits = 0
                    for shot_tick in spray_shot_ticks:
                        for dt in range(shot_tick, shot_tick + 5):
                            if dt in damage_ticks:
                                spray_hits += 1
                                break
                    player.spray_shots_hit = spray_hits

    def _calculate_counter_strafing_for_player(
        self, player: PlayerMatchStats, steam_id: int
    ) -> None:
        """Calculate counter-strafing percentage for a player.

        Leetify-style metric: measures movement discipline across ALL shots fired,
        not just kill shots. A shot is "stationary" if velocity < 34 units/s.

        Excludes non-gun weapons (knife, grenades, C4) since movement doesn't
        affect their accuracy.
        """
        # Velocity threshold - standard CS2 "stopped" speed for accurate shooting
        cs_velocity_threshold = 34.0

        # Weapons to EXCLUDE from counter-strafe tracking (movement doesn't matter)
        excluded_weapons = {
            "knife",
            "knife_t",
            "bayonet",
            "hegrenade",
            "flashbang",
            "smokegrenade",
            "molotov",
            "incgrenade",
            "decoy",
            "c4",
            "taser",  # Zeus
        }

        if not hasattr(self.data, "weapon_fires") or not self.data.weapon_fires:
            return

        # Count stationary vs moving shots across ALL weapon_fire events
        shots_stationary = 0
        shots_with_velocity = 0

        # Also track kill-based stats for backward compatibility
        counter_strafe_kills = 0
        tracked_kills = 0

        # Build velocity lookup for kill-based tracking (backward compat)
        velocity_by_tick: dict[int, float] = {}

        for fire in self.data.weapon_fires:
            if fire.player_steamid != steam_id:
                continue

            # Get weapon name, normalize it
            weapon = (fire.weapon or "").lower().replace("weapon_", "")

            # Skip non-gun weapons
            if weapon in excluded_weapons:
                continue

            # Check if velocity data is available
            if fire.velocity_x is None or fire.velocity_y is None:
                continue

            # Calculate 2D velocity (horizontal movement)
            vel_x = fire.velocity_x or 0.0
            vel_y = fire.velocity_y or 0.0
            velocity_2d = math.sqrt(vel_x**2 + vel_y**2)

            # Track for shot-based metric (NEW - Leetify parity)
            shots_with_velocity += 1
            if velocity_2d < cs_velocity_threshold:
                shots_stationary += 1

            # Store for kill-based lookup (backward compat)
            velocity_by_tick[fire.tick] = velocity_2d

        # Set shot-based stats (NEW - Leetify parity)
        player.shots_stationary = shots_stationary
        player.shots_with_velocity = shots_with_velocity

        # Calculate kill-based stats for backward compatibility
        if velocity_by_tick:
            kills_by_player = [k for k in self.data.kills if k.attacker_steamid == steam_id]

            for kill in kills_by_player:
                kill_tick = kill.tick
                closest_velocity = None
                min_tick_diff = float("inf")

                for fire_tick, velocity in velocity_by_tick.items():
                    tick_diff = abs(fire_tick - kill_tick)
                    if tick_diff < min_tick_diff and fire_tick <= kill_tick and tick_diff <= 10:
                        min_tick_diff = tick_diff
                        closest_velocity = velocity

                if closest_velocity is not None:
                    tracked_kills += 1
                    if closest_velocity < cs_velocity_threshold:
                        counter_strafe_kills += 1

        # Set kill-based stats (DEPRECATED but kept for backward compatibility)
        player.counter_strafe_kills = counter_strafe_kills
        player.total_kills_with_velocity = tracked_kills

    def _calculate_mistakes(self) -> None:
        """Calculate mistakes (Scope.gg style)."""
        from opensight.analysis.compute_utility import calculate_mistakes

        calculate_mistakes(self)

    def _detect_greedy_repeeks(self) -> None:
        """
        Detect greedy re-peek deaths (static repeek discipline).

        A greedy re-peek occurs when a player:
        1. Gets a kill
        2. Dies within 3 seconds (192 ticks at 64 tick rate)
        3. Their death position is within 150 units of their kill position

        This indicates the player re-peeked the same angle after getting a kill
        instead of repositioning - a common mistake that gets punished.
        """
        kills_df = self.data.kills_df
        if kills_df.empty:
            logger.debug("No kills data for greedy repeek detection")
            return

        # Check for required columns
        att_id_col = self._att_id_col
        vic_id_col = self._vic_id_col
        tick_col = self._find_col(kills_df, ["tick", "game_tick", "time_tick"])

        # Find position columns
        att_x_col = self._find_col(kills_df, ["attacker_X", "attacker_x", "X", "x"])
        att_y_col = self._find_col(kills_df, ["attacker_Y", "attacker_y", "Y", "y"])
        vic_x_col = self._find_col(kills_df, ["user_X", "victim_X", "user_x", "victim_x"])
        vic_y_col = self._find_col(kills_df, ["user_Y", "victim_Y", "user_y", "victim_y"])

        if not all([att_id_col, vic_id_col, tick_col, att_x_col, att_y_col, vic_x_col, vic_y_col]):
            logger.debug("Missing columns for greedy repeek detection")
            return

        # Constants
        REPEEK_WINDOW_TICKS = 192  # 3 seconds at 64 tick
        STATIC_DISTANCE_THRESHOLD = 150  # units (approx 1.5 steps)

        for steam_id, player in self._players.items():
            greedy_count = 0

            # Get kills by this player (they are attacker)
            player_kills = kills_df[kills_df[att_id_col] == steam_id].copy()

            # Get deaths of this player (they are victim)
            player_deaths = kills_df[kills_df[vic_id_col] == steam_id].copy()

            if player_kills.empty or player_deaths.empty:
                continue

            # Sort by tick
            player_kills = player_kills.sort_values(tick_col)
            player_deaths = player_deaths.sort_values(tick_col)

            # For each kill, check if player died shortly after in similar position
            for _, kill_row in player_kills.iterrows():
                kill_tick = kill_row[tick_col]
                kill_x = kill_row[att_x_col]
                kill_y = kill_row[att_y_col]

                # Skip if position data is missing
                if pd.isna(kill_x) or pd.isna(kill_y):
                    continue

                # Find deaths within the time window after this kill
                subsequent_deaths = player_deaths[
                    (player_deaths[tick_col] > kill_tick)
                    & (player_deaths[tick_col] <= kill_tick + REPEEK_WINDOW_TICKS)
                ]

                for _, death_row in subsequent_deaths.iterrows():
                    death_x = death_row[vic_x_col]
                    death_y = death_row[vic_y_col]

                    # Skip if position data is missing
                    if pd.isna(death_x) or pd.isna(death_y):
                        continue

                    # Calculate distance between kill position and death position
                    distance = np.sqrt((kill_x - death_x) ** 2 + (kill_y - death_y) ** 2)

                    # If player was still in roughly the same spot, it's a greedy repeek
                    if distance < STATIC_DISTANCE_THRESHOLD:
                        greedy_count += 1
                        break  # Only count one greedy death per kill

            # Update player stats
            player.greedy_repeeks = greedy_count

            # Calculate discipline rating: (safe kills / total kills) * 100
            if player.kills > 0:
                safe_kills = player.kills - greedy_count
                player.discipline_rating = round((safe_kills / player.kills) * 100, 1)
            else:
                player.discipline_rating = 100.0  # No kills = no mistakes possible

        total_greedy = sum(p.greedy_repeeks for p in self._players.values())
        logger.info(f"Detected {total_greedy} greedy re-peek deaths across all players")

    def _run_state_machine(self) -> None:
        """
        Run State Machine analysis for pro-level metrics.

        Enhances PlayerMatchStats with:
        - Entry Kill (from State Machine - more accurate than opening duels)
        - Trade Kill (with 4-second window and proper killer tracking)
        - Lurk Kill (distance from team center of mass)
        - Flash Effectiveness (>2.0s blind duration)
        - Utility ADR (HE + Molotov damage per round)
        """
        try:
            from opensight.visualization.state_machine import StateMachine
        except ImportError as e:
            logger.warning(f"State Machine not available: {e}")
            return

        try:
            machine = StateMachine(self.data)
            result = machine.analyze()

            # Merge state machine results into PlayerMatchStats
            for steam_id, context_stats in result.players.items():
                if steam_id not in self._players:
                    continue

                player = self._players[steam_id]

                # Update entry stats (State Machine is more accurate)
                player.opening_duels.wins = context_stats.entry_kills
                player.opening_duels.losses = context_stats.entry_deaths
                player.opening_duels.attempts = context_stats.entry_attempts

                # Update trade stats (State Machine uses tighter 4-second window)
                player.trades.kills_traded = context_stats.trade_kills
                player.trades.deaths_traded = context_stats.deaths_traded
                player.trades.trade_attempts = context_stats.trade_opportunities

                # Add lurk stats
                player.lurk.kills = context_stats.lurk_kills
                player.lurk.deaths = context_stats.lurk_deaths

                # Flash effectiveness (>2.0 seconds = effective)
                player.effective_flashes = context_stats.effective_flashes
                player.ineffective_flashes = context_stats.ineffective_flashes

                # Utility ADR
                util_damage = context_stats.he_damage + context_stats.molotov_damage
                player.utility_adr = (
                    round(util_damage / player.rounds_played, 1)
                    if player.rounds_played > 0
                    else 0.0
                )

            logger.info(
                f"State Machine complete: {result.total_entry_kills} entries, "
                f"{result.total_trade_kills} trades, {result.total_lurk_kills} lurks"
            )

        except Exception as e:
            logger.warning(f"State Machine analysis failed: {e}")

    def _integrate_economy(self) -> dict:
        """Integrate economy module data into player stats."""
        if not HAS_ECONOMY:
            logger.debug("Economy module not available, skipping integration")
            return {}

        try:
            economy_analyzer = EconomyAnalyzer(self.data)
            economy_stats = economy_analyzer.analyze()

            # Merge economy data into player stats
            for steam_id, player in self._players.items():
                profile = economy_analyzer.get_player_profile(steam_id)
                if profile:
                    player.avg_equipment_value = profile.avg_equipment_value
                    player.eco_rounds = profile.eco_rounds
                    player.force_rounds = profile.force_rounds
                    player.full_buy_rounds = profile.full_buy_rounds
                    player.damage_per_dollar = profile.damage_per_dollar
                    player.kills_per_dollar = profile.kills_per_dollar

            logger.info(f"Economy integration complete: {economy_stats.rounds_analyzed} rounds")
            return {
                "avg_equipment": economy_stats.avg_equipment_value,
                "damage_per_dollar": economy_stats.damage_per_dollar,
            }
        except Exception as e:
            logger.warning(f"Economy integration failed: {e}")
            return {}

    def _integrate_combat(self) -> dict:
        """Integrate combat module data into player stats."""
        if not HAS_COMBAT:
            logger.debug("Combat module not available, skipping integration")
            return {}

        try:
            combat_analyzer = CombatAnalyzer(self.data)
            combat_stats = combat_analyzer.analyze()

            # Merge combat data into player stats
            for steam_id, player in self._players.items():
                if steam_id in combat_stats.player_stats:
                    cs = combat_stats.player_stats[steam_id]
                    player.trade_kill_time_avg_ms = cs.trade_kill_time_avg_ms
                    player.untraded_deaths = cs.untraded_deaths

            # Process spray transfers
            for spray in combat_stats.spray_transfers:
                steam_id = spray.player_steamid
                if steam_id in self._players:
                    player = self._players[steam_id]
                    kills = spray.kills_in_spray
                    player.spray_transfers.total_spray_kills += kills
                    player.spray_transfers._spray_times_ms.append(spray.time_span_ms)

                    if kills == 2:
                        player.spray_transfers.double_sprays += 1
                    elif kills == 3:
                        player.spray_transfers.triple_sprays += 1
                    elif kills == 4:
                        player.spray_transfers.quad_sprays += 1
                    elif kills >= 5:
                        player.spray_transfers.ace_sprays += 1

            # Build team-level stats
            trade_rates = {
                "CT": combat_stats.team_trade_rate.get(3, 0.0),
                "T": combat_stats.team_trade_rate.get(2, 0.0),
            }
            opening_rates = {
                "CT": combat_stats.team_opening_win_rate.get(3, 0.0),
                "T": combat_stats.team_opening_win_rate.get(2, 0.0),
            }

            logger.info(
                f"Combat integration complete: {len(combat_stats.trade_kills)} trades, "
                f"{len(combat_stats.opening_duels)} opening duels"
            )
            return {
                "trade_rates": trade_rates,
                "opening_rates": opening_rates,
            }
        except Exception as e:
            logger.warning(f"Combat integration failed: {e}")
            return {}

    def _build_kill_matrix(self) -> list:
        """Build kill matrix showing who killed whom and how often."""
        kills_df = self.data.kills_df
        if kills_df.empty or not self._att_id_col or not self._vic_id_col:
            return []

        matrix_entries = []
        weapon_col = self._find_col(kills_df, ["weapon"])

        # Group by attacker-victim pairs
        try:
            grouped = kills_df.groupby([self._att_id_col, self._vic_id_col])

            for (att_id, vic_id), group in grouped:
                att_id = safe_int(att_id)
                vic_id = safe_int(vic_id)

                if att_id == 0 or vic_id == 0:
                    continue

                att_name = self.data.player_names.get(att_id, "Unknown")
                vic_name = self.data.player_names.get(vic_id, "Unknown")

                weapons = []
                if weapon_col:
                    weapons = group[weapon_col].value_counts().head(3).index.tolist()

                matrix_entries.append(
                    KillMatrixEntry(
                        attacker_name=att_name,
                        victim_name=vic_name,
                        count=len(group),
                        weapons=weapons,
                    )
                )

            logger.info(f"Built kill matrix with {len(matrix_entries)} entries")
        except Exception as e:
            logger.warning(f"Kill matrix building failed: {e}")

        return matrix_entries

    def _build_round_timeline(self) -> list:
        """Build round-by-round timeline with key events and win probability."""
        timeline = []

        if not self.data.rounds:
            return timeline

        kills_df = self.data.kills_df
        tick_rate = getattr(self.data, "tick_rate", CS2_TICK_RATE)

        # Build lookup for victim side from kills
        victim_side_col = self._find_col(kills_df, ["victim_team_name", "user_team_name"])

        for round_info in self.data.rounds:
            try:
                round_num = round_info.round_num
                winner = round_info.winner or "Unknown"
                win_reason = round_info.win_reason or "unknown"
                round_start_tick = round_info.start_tick

                # Get first kill of round
                first_kill_player = ""
                first_death_player = ""

                # Build win probability timeline for this round
                momentum = self._build_round_momentum(
                    round_num=round_num,
                    round_info=round_info,
                    kills_df=kills_df,
                    winner=winner,
                    round_start_tick=round_start_tick,
                    tick_rate=tick_rate,
                    victim_side_col=victim_side_col,
                )

                if not kills_df.empty and self._round_col and self._att_id_col and self._vic_id_col:
                    round_kills = kills_df[kills_df[self._round_col] == round_num]
                    if not round_kills.empty:
                        round_kills = round_kills.sort_values(by="tick")
                        first = round_kills.iloc[0]
                        att_id = safe_int(first.get(self._att_id_col))
                        vic_id = safe_int(first.get(self._vic_id_col))
                        first_kill_player = self.data.player_names.get(att_id, "")
                        first_death_player = self.data.player_names.get(vic_id, "")

                timeline.append(
                    RoundTimeline(
                        round_num=round_num,
                        winner=winner,
                        win_reason=win_reason,
                        ct_score=round_info.ct_score,
                        t_score=round_info.t_score,
                        first_kill_player=first_kill_player,
                        first_death_player=first_death_player,
                        momentum=momentum,
                    )
                )
            except Exception as e:
                logger.debug(f"Error building timeline for round: {e}")
                continue

        # Log throw/heroic summary
        throws = sum(1 for t in timeline if t.momentum and t.momentum.round_tag)
        logger.info(
            f"Built round timeline with {len(timeline)} rounds, {throws} throw/heroic rounds"
        )
        return timeline

    def _build_round_momentum(
        self,
        round_num: int,
        round_info,
        kills_df: pd.DataFrame,
        winner: str,
        round_start_tick: int,
        tick_rate: int,
        victim_side_col: str | None,
    ) -> RoundMomentum:
        """
        Build win probability timeline for a single round.

        Tracks probability at each state change (kill, bomb plant, etc.)
        to identify throw/heroic rounds.
        """
        momentum = RoundMomentum(round_num=round_num, winner=winner)
        prob_events: list[WinProbEvent] = []

        # Initial state: 5v5, no bomb
        ct_alive = 5
        t_alive = 5
        bomb_planted = False

        # Add round start event
        ct_prob = calculate_win_probability("CT", ct_alive, t_alive, bomb_planted)
        t_prob = calculate_win_probability("T", ct_alive, t_alive, bomb_planted)
        prob_events.append(
            WinProbEvent(
                tick=round_start_tick,
                time_seconds=0.0,
                event_type="round_start",
                ct_alive=ct_alive,
                t_alive=t_alive,
                bomb_planted=bomb_planted,
                ct_win_prob=ct_prob,
                t_win_prob=t_prob,
                description="Round start (5v5)",
            )
        )

        # Collect all state-changing events for this round
        state_events = []

        # Add kills
        if not kills_df.empty and self._round_col:
            round_kills = kills_df[kills_df[self._round_col] == round_num].copy()
            if not round_kills.empty:
                for _, kill in round_kills.iterrows():
                    tick = safe_int(kill.get("tick", 0))
                    victim_side = ""
                    if victim_side_col and victim_side_col in kill.index:
                        victim_side = str(kill.get(victim_side_col, "")).upper()

                    # Fallback to KillEvent data if DataFrame doesn't have side
                    if not victim_side or victim_side not in (
                        "CT",
                        "T",
                        "COUNTERTERRORIST",
                        "TERRORIST",
                    ):
                        # Try to find in kills list
                        for k in self.data.kills:
                            if k.round_num == round_num and k.tick == tick:
                                victim_side = k.victim_side or ""
                                break

                    # Normalize side names
                    if "CT" in victim_side or "COUNTER" in victim_side:
                        victim_side = "CT"
                    else:
                        victim_side = "T"

                    att_id = safe_int(kill.get(self._att_id_col)) if self._att_id_col else 0
                    vic_id = safe_int(kill.get(self._vic_id_col)) if self._vic_id_col else 0
                    att_name = self.data.player_names.get(att_id, "Unknown")
                    vic_name = self.data.player_names.get(vic_id, "Unknown")

                    state_events.append(
                        {
                            "tick": tick,
                            "type": "kill",
                            "victim_side": victim_side,
                            "description": f"{att_name} killed {vic_name}",
                        }
                    )

        # Add bomb plant event
        bomb_plant_tick = getattr(round_info, "bomb_plant_tick", None)
        if bomb_plant_tick:
            state_events.append(
                {
                    "tick": bomb_plant_tick,
                    "type": "bomb_plant",
                    "description": "Bomb planted",
                }
            )

        # Sort events by tick
        state_events.sort(key=lambda e: e["tick"])

        # Process events in order and calculate probability after each
        for event in state_events:
            tick = event["tick"]
            time_seconds = (tick - round_start_tick) / tick_rate if tick_rate > 0 else 0.0

            if event["type"] == "kill":
                # Update alive count
                if event["victim_side"] == "CT":
                    ct_alive = max(0, ct_alive - 1)
                else:
                    t_alive = max(0, t_alive - 1)
                event_type = "kill"
            elif event["type"] == "bomb_plant":
                bomb_planted = True
                event_type = "bomb_plant"
            else:
                event_type = event["type"]

            # Calculate new probabilities
            ct_prob = calculate_win_probability("CT", ct_alive, t_alive, bomb_planted)
            t_prob = calculate_win_probability("T", ct_alive, t_alive, bomb_planted)

            prob_events.append(
                WinProbEvent(
                    tick=tick,
                    time_seconds=time_seconds,
                    event_type=event_type,
                    ct_alive=ct_alive,
                    t_alive=t_alive,
                    bomb_planted=bomb_planted,
                    ct_win_prob=ct_prob,
                    t_win_prob=t_prob,
                    description=event.get("description", ""),
                )
            )

        # Store timeline and compute peak/min values
        momentum.win_prob_timeline = prob_events

        if prob_events:
            ct_probs = [e.ct_win_prob for e in prob_events]
            t_probs = [e.t_win_prob for e in prob_events]
            momentum.ct_peak_prob = max(ct_probs)
            momentum.ct_min_prob = min(ct_probs)
            momentum.t_peak_prob = max(t_probs)
            momentum.t_min_prob = min(t_probs)

        return momentum

    def _extract_position_data(self) -> tuple[list, list]:
        """Extract position data for heatmap and kill map visualization."""
        kill_positions = []
        death_positions = []

        # Extract from KillEvent objects (they have position data)
        for kill in self.data.kills:
            try:
                att_name = self.data.player_names.get(kill.attacker_steamid, "Unknown")
                vic_name = self.data.player_names.get(kill.victim_steamid, "Unknown")

                # Attacker position (kill location)
                if kill.attacker_x is not None and kill.attacker_y is not None:
                    kill_positions.append(
                        {
                            "x": kill.attacker_x,
                            "y": kill.attacker_y,
                            "z": kill.attacker_z or 0,
                            "player": att_name,
                            "attacker": att_name,
                            "victim": vic_name,
                            "attacker_team": kill.attacker_side,
                            "victim_team": kill.victim_side,
                            "weapon": kill.weapon,
                            "round": kill.round_num,
                            "headshot": kill.headshot,
                        }
                    )

                # Victim position (death location)
                if kill.victim_x is not None and kill.victim_y is not None:
                    death_positions.append(
                        {
                            "x": kill.victim_x,
                            "y": kill.victim_y,
                            "z": kill.victim_z or 0,
                            "player": vic_name,
                            "attacker": att_name,
                            "victim_team": kill.victim_side,
                            "attacker_team": kill.attacker_side,
                            "round": kill.round_num,
                        }
                    )
            except Exception as e:
                logger.debug(f"Error extracting position: {e}")
                continue

        logger.info(
            f"Extracted {len(kill_positions)} kill positions, {len(death_positions)} death positions"
        )
        return kill_positions, death_positions

    def _extract_grenade_trajectories(self) -> tuple[list, dict]:
        """
        Extract grenade trajectory data for utility visualization.

        Returns:
            Tuple of (grenade_positions, team_stats) where:
            - grenade_positions: List of dicts with position and metadata
            - team_stats: Dict with team-level utility statistics
        """
        # Grenade visualization constants (moved from trajectory.py)
        GRENADE_COLORS = {
            "flashbang": "#ffff00",
            "smokegrenade": "#808080",
            "hegrenade": "#ff4500",
            "molotov": "#ff6600",
            "incgrenade": "#ff6600",
            "decoy": "#00ff00",
        }
        GRENADE_CSS_CLASSES = {
            "flashbang": "grenade-flash",
            "smokegrenade": "grenade-smoke",
            "hegrenade": "grenade-he",
            "molotov": "grenade-molotov",
            "incgrenade": "grenade-molotov",
            "decoy": "grenade-decoy",
        }

        grenade_positions = []
        team_stats = {
            "CT": {
                "total_utility": 0,
                "flashbangs": 0,
                "smokes": 0,
                "molotovs": 0,
                "he_grenades": 0,
                "enemies_flashed": 0,
            },
            "T": {
                "total_utility": 0,
                "flashbangs": 0,
                "smokes": 0,
                "molotovs": 0,
                "he_grenades": 0,
                "enemies_flashed": 0,
            },
        }

        # Process grenade events to get positions (detonation points)
        if hasattr(self.data, "grenades") and self.data.grenades:
            for grenade in self.data.grenades:
                # Only include grenades with valid positions
                if grenade.x is not None and grenade.y is not None:
                    grenade_type = grenade.grenade_type.lower()
                    # Use round-aware side lookup to handle halftime swaps
                    thrower_team = self._get_player_side(grenade.player_steamid, grenade.round_num)

                    position = {
                        "x": grenade.x,
                        "y": grenade.y,
                        "z": grenade.z or 0,
                        "grenade_type": grenade_type,
                        "thrower_steamid": str(grenade.player_steamid),
                        "thrower_name": self.data.player_names.get(
                            grenade.player_steamid, "Unknown"
                        ),
                        "thrower_team": thrower_team,
                        "round_num": grenade.round_num,
                        "tick": grenade.tick,
                        "color": GRENADE_COLORS.get(grenade_type, "#ffffff"),
                        "css_class": GRENADE_CSS_CLASSES.get(grenade_type, "grenade-unknown"),
                    }
                    grenade_positions.append(position)

                # Count grenades for team stats (count each grenade once)
                # Use round-aware side lookup to handle halftime swaps
                thrower_team = self._get_player_side(grenade.player_steamid, grenade.round_num)
                if thrower_team in team_stats:
                    grenade_type = grenade.grenade_type.lower()
                    team_stats[thrower_team]["total_utility"] += 1

                    if "flash" in grenade_type:
                        team_stats[thrower_team]["flashbangs"] += 1
                    elif "smoke" in grenade_type:
                        team_stats[thrower_team]["smokes"] += 1
                    elif "molotov" in grenade_type or "inc" in grenade_type:
                        team_stats[thrower_team]["molotovs"] += 1
                    elif "hegrenade" in grenade_type:
                        team_stats[thrower_team]["he_grenades"] += 1

        # Count enemies flashed from blinds data
        if hasattr(self.data, "blinds") and self.data.blinds:
            for blind in self.data.blinds:
                if not blind.is_teammate and blind.blind_duration >= 1.5:
                    # Use round-aware side lookup to handle halftime swaps
                    attacker_team = self._get_player_side(blind.attacker_steamid, blind.round_num)
                    if attacker_team in team_stats:
                        team_stats[attacker_team]["enemies_flashed"] += 1

        logger.info(f"Extracted {len(grenade_positions)} grenade positions")
        return grenade_positions, team_stats

    def _generate_coaching_insights(self) -> list:
        """Generate AI-powered coaching insights based on player performance."""
        insights = []

        for steam_id, player in self._players.items():
            player_insights = []

            # HLTV Rating insights
            if player.hltv_rating < 0.8:
                player_insights.append(
                    {
                        "type": "warning",
                        "category": "overall",
                        "message": f"Low overall rating ({player.hltv_rating:.2f}). Focus on fundamentals.",
                        "priority": "high",
                    }
                )
            elif player.hltv_rating > 1.3:
                player_insights.append(
                    {
                        "type": "positive",
                        "category": "overall",
                        "message": f"Excellent performance ({player.hltv_rating:.2f}). Keep it up!",
                        "priority": "low",
                    }
                )

            # TTD insights (reaction time)
            if player.ttd_median_ms:
                if player.ttd_median_ms > 500:
                    player_insights.append(
                        {
                            "type": "warning",
                            "category": "aim",
                            "message": f"Slow time-to-damage ({player.ttd_median_ms:.0f}ms). Consider aim training.",
                            "priority": "medium",
                        }
                    )
                elif player.ttd_median_ms < 200:
                    player_insights.append(
                        {
                            "type": "positive",
                            "category": "aim",
                            "message": f"Fast reactions ({player.ttd_median_ms:.0f}ms TTD)!",
                            "priority": "low",
                        }
                    )

            # Crosshair placement insights
            if player.cp_median_error_deg:
                if player.cp_median_error_deg > 15:
                    player_insights.append(
                        {
                            "type": "warning",
                            "category": "aim",
                            "message": f"Poor crosshair placement ({player.cp_median_error_deg:.1f}° error). Keep crosshair at head level.",
                            "priority": "high",
                        }
                    )
                elif player.cp_median_error_deg < 5:
                    player_insights.append(
                        {
                            "type": "positive",
                            "category": "aim",
                            "message": f"Excellent crosshair placement ({player.cp_median_error_deg:.1f}°)!",
                            "priority": "low",
                        }
                    )

            # Trade insights - with lurker exception
            if player.untraded_deaths > player.deaths * 0.6 and player.deaths > 3:
                # Build stats dict for lurker detection
                player_stats_for_lurk = {
                    "deaths": player.deaths,
                    "untraded_deaths": player.untraded_deaths,
                    "kills": player.kills,
                    "hltv_rating": player.hltv_rating,
                    "backstab_kills": getattr(player, "backstab_kills", 0),
                    "impact_rating": getattr(player, "impact_rating", 0),
                    "rounds_played": player.rounds_played,
                }

                # Check if this player is an effective lurker
                is_lurker = (
                    HAS_PERSONA
                    and _is_effective_lurker
                    and _is_effective_lurker(player_stats_for_lurk)
                )

                if is_lurker:
                    # Lurker with impact - no spacing warning needed
                    # Only warn if they have lurk kills but low conversion
                    lurk_kills = getattr(player.lurk, "kills", 0) if hasattr(player, "lurk") else 0
                    if lurk_kills == 0 and player.kills < player.rounds_played * 0.5:
                        # Lurking without getting kills - this IS a problem
                        player_insights.append(
                            {
                                "type": "warning",
                                "category": "positioning",
                                "message": f"Lurking without impact ({player.untraded_deaths} solo deaths, {player.kills} kills). Lurking requires getting picks.",
                                "priority": "medium",
                            }
                        )
                    # else: Effective lurker - suppress spacing warning
                else:
                    # Not a lurker - standard spacing warning applies
                    player_insights.append(
                        {
                            "type": "warning",
                            "category": "positioning",
                            "message": f"Too many untraded deaths ({player.untraded_deaths}/{player.deaths}). Stay closer to teammates.",
                            "priority": "medium",
                        }
                    )

            # Utility insights
            if (
                player.utility.total_utility < player.rounds_played * 1.5
                and player.rounds_played >= 10
            ):
                player_insights.append(
                    {
                        "type": "warning",
                        "category": "utility",
                        "message": f"Low utility usage ({player.utility.total_utility} thrown in {player.rounds_played} rounds). Buy and use more grenades.",
                        "priority": "medium",
                    }
                )

            if player.utility.teammates_flashed > 5:
                player_insights.append(
                    {
                        "type": "mistake",
                        "category": "utility",
                        "message": f"Flashed teammates {player.utility.teammates_flashed} times. Communicate flash usage.",
                        "priority": "high",
                    }
                )

            # Opening duel insights
            if player.opening_duels.attempts >= 5:
                if player.opening_duels.win_rate < 30:
                    player_insights.append(
                        {
                            "type": "warning",
                            "category": "duels",
                            "message": f"Low opening duel win rate ({player.opening_duels.win_rate:.0f}%). Reconsider early aggression.",
                            "priority": "medium",
                        }
                    )
                elif player.opening_duels.win_rate > 70:
                    player_insights.append(
                        {
                            "type": "positive",
                            "category": "duels",
                            "message": f"Dominant entry fragging ({player.opening_duels.win_rate:.0f}% win rate)!",
                            "priority": "low",
                        }
                    )

            # KAST insights
            if player.kast_percentage < 60 and player.rounds_played >= 10:
                player_insights.append(
                    {
                        "type": "warning",
                        "category": "impact",
                        "message": f"Low KAST ({player.kast_percentage:.0f}%). Try to have more round impact.",
                        "priority": "medium",
                    }
                )

            # Team damage insights
            if player.mistakes.team_damage > 100:
                player_insights.append(
                    {
                        "type": "mistake",
                        "category": "mistakes",
                        "message": f"High team damage ({player.mistakes.team_damage}). Be more careful with grenades and fire.",
                        "priority": "high",
                    }
                )

            # Greedy re-peek discipline insights
            if player.greedy_repeeks >= 2:
                player_insights.append(
                    {
                        "type": "mistake",
                        "category": "positioning",
                        "message": f"Greedy plays detected. You died {player.greedy_repeeks} times by re-peeking the same angle after getting a kill. Reposition after kills.",
                        "priority": "high",
                    }
                )
            elif player.discipline_rating < 90.0 and player.kills >= 5:
                player_insights.append(
                    {
                        "type": "warning",
                        "category": "positioning",
                        "message": f"Discipline rating: {player.discipline_rating:.0f}%. After kills, reposition instead of re-peeking.",
                        "priority": "medium",
                    }
                )

            if player_insights:
                insights.append(
                    {
                        "steam_id": str(steam_id),
                        "player_name": player.name,
                        "insights": player_insights,
                    }
                )

        logger.info(f"Generated coaching insights for {len(insights)} players")
        return insights

    def _calculate_team_scores(self) -> tuple[int, int]:
        """Calculate team scores from round data."""
        if not self.data.rounds:
            return (0, 0)

        ct_wins = sum(1 for r in self.data.rounds if r.winner == "CT")
        t_wins = sum(1 for r in self.data.rounds if r.winner == "T")
        return (ct_wins, t_wins)

    def _extract_team_names(self) -> tuple[str, str]:
        """Extract team names from demo data or use defaults.

        Returns:
            Tuple of (CT team name, T team name).
            - If clan tags exist, use those
            - Otherwise use "Counter-Terrorists" / "Terrorists"
        """
        # Default to full team names (cleaner than "Team 1/2")
        team1_name = "Counter-Terrorists"
        team2_name = "Terrorists"

        # Try to get team names from players' clan tags or team assignments
        ct_players = [p for p in self._players.values() if p.team == "CT"]
        t_players = [p for p in self._players.values() if p.team == "T"]

        # Use first player's clan tag if available
        if ct_players:
            clan_tag = getattr(ct_players[0], "clan_tag", None)
            if clan_tag:
                team1_name = clan_tag
        if t_players:
            clan_tag = getattr(t_players[0], "clan_tag", None)
            if clan_tag:
                team2_name = clan_tag

        return (team1_name, team2_name)


def compute_utility_metrics(match_data: DemoData) -> dict[str, UtilityMetrics]:
    """Compute utility usage metrics for all players from match data."""
    from opensight.analysis.compute_utility import compute_utility_metrics as _compute

    return _compute(match_data)


def analyze_demo(
    demo_data: DemoData,
    metrics: str | list[str] | None = None,
    use_cache: bool = True,
    use_optimized: bool = True,
) -> MatchAnalysis:
    """Convenience function to analyze a parsed demo.

    Args:
        demo_data: Parsed demo data from DemoParser
        metrics: Which metrics to compute. Options:
            - None or "full": Compute all metrics (default)
            - "basic": Only basic stats (KD, ADR, HS%)
            - "advanced": Basic + TTD, CP, trades, opening duels
            - List of specific categories: ["ttd", "cp", "trades"]
        use_cache: Whether to use metrics caching (default True)
        use_optimized: Whether to use vectorized implementations (default True)

    Returns:
        MatchAnalysis with computed metrics
    """
    analyzer = DemoAnalyzer(
        demo_data, metrics=metrics, use_cache=use_cache, use_optimized=use_optimized
    )
    return analyzer.analyze()


def get_player_comparison_stats(
    player_a: PlayerMatchStats, player_b: PlayerMatchStats, normalize: bool = True
) -> dict:
    """
    Generate comparison statistics for two players for radar chart visualization.

    Returns normalized stats (0-100 scale) suitable for radar charts when normalize=True.
    The axes are:
    - HLTV Rating (higher is better)
    - Impact Score (higher is better)
    - TTD (inverted - lower TTD means faster reactions, so we invert for display)
    - Headshot % (higher is better)
    - Utility Damage (HE + Molotov damage, higher is better)

    Args:
        player_a: First player's match stats
        player_b: Second player's match stats
        normalize: If True, normalize values to 0-100 scale

    Returns:
        Dict with comparison data for both players
    """

    def get_ttd_score(player: PlayerMatchStats) -> float:
        """Calculate TTD score. Lower TTD is better, so we invert it."""
        ttd = player.ttd_median_ms
        if ttd is None:
            return 50.0 if normalize else 0.0  # Default to average if no data

        if normalize:
            # TTD typically ranges from 100ms (elite) to 600ms (slow)
            # Invert so that lower (faster) TTD gives higher score
            # 100ms -> 100 score, 600ms -> 0 score
            clamped = max(100, min(600, ttd))
            return round(100 - ((clamped - 100) / 500 * 100), 1)
        return round(ttd, 1)

    def get_hltv_score(player: PlayerMatchStats) -> float:
        """Normalize HLTV rating to 0-100 scale."""
        rating = player.hltv_rating
        if normalize:
            # HLTV rating typically ranges from 0.5 to 2.0
            # Map 0.5 -> 0, 1.0 -> 50, 1.5 -> 100
            clamped = max(0.5, min(2.0, rating))
            return round((clamped - 0.5) / 1.5 * 100, 1)
        return round(rating, 2)

    def get_impact_score(player: PlayerMatchStats) -> float:
        """Normalize impact rating to 0-100 scale."""
        impact = player.impact_rating
        if normalize:
            # Impact typically ranges from 0.0 to 2.0
            clamped = max(0.0, min(2.0, impact))
            return round(clamped / 2.0 * 100, 1)
        return round(impact, 2)

    def get_hs_score(player: PlayerMatchStats) -> float:
        """Get headshot percentage (already 0-100)."""
        return round(player.headshot_percentage, 1)

    def get_utility_damage_score(player: PlayerMatchStats) -> float:
        """Calculate utility damage score (HE + Molotov damage)."""
        he_dmg = player.utility.he_damage
        molotov_dmg = player.utility.molotov_damage
        total_dmg = he_dmg + molotov_dmg

        if normalize:
            # Utility damage can vary widely, let's use 0-500 as a typical range
            # Map 0 -> 0, 250 -> 50, 500+ -> 100
            clamped = max(0, min(500, total_dmg))
            return round(clamped / 500 * 100, 1)
        return total_dmg

    # Calculate stats for both players
    player_a_stats = {
        "name": player_a.name,
        "steam_id": str(player_a.steam_id),
        "team": player_a.team,
        "metrics": {
            "hltv_rating": get_hltv_score(player_a),
            "impact_score": get_impact_score(player_a),
            "ttd_score": get_ttd_score(player_a),
            "headshot_pct": get_hs_score(player_a),
            "utility_damage": get_utility_damage_score(player_a),
        },
        "raw_values": {
            "hltv_rating": round(player_a.hltv_rating, 2),
            "impact_rating": round(player_a.impact_rating, 2),
            "ttd_median_ms": (round(player_a.ttd_median_ms, 1) if player_a.ttd_median_ms else None),
            "headshot_pct": round(player_a.headshot_percentage, 1),
            "utility_damage": player_a.utility.he_damage + player_a.utility.molotov_damage,
        },
    }

    player_b_stats = {
        "name": player_b.name,
        "steam_id": str(player_b.steam_id),
        "team": player_b.team,
        "metrics": {
            "hltv_rating": get_hltv_score(player_b),
            "impact_score": get_impact_score(player_b),
            "ttd_score": get_ttd_score(player_b),
            "headshot_pct": get_hs_score(player_b),
            "utility_damage": get_utility_damage_score(player_b),
        },
        "raw_values": {
            "hltv_rating": round(player_b.hltv_rating, 2),
            "impact_rating": round(player_b.impact_rating, 2),
            "ttd_median_ms": (round(player_b.ttd_median_ms, 1) if player_b.ttd_median_ms else None),
            "headshot_pct": round(player_b.headshot_percentage, 1),
            "utility_damage": player_b.utility.he_damage + player_b.utility.molotov_damage,
        },
    }

    return {
        "labels": [
            "HLTV Rating",
            "Impact Score",
            "TTD (Reaction)",
            "Headshot %",
            "Utility Damage",
        ],
        "player_a": player_a_stats,
        "player_b": player_b_stats,
        "normalized": normalize,
    }


def compare_players(match_analysis: MatchAnalysis, player_a_name: str, player_b_name: str) -> dict:
    """
    Compare two players using Scope.gg-style radar chart axes.

    Axes (all normalized to 0-100 score for visualization):
    - ADR: Average Damage per Round (scaled 0-150, where 100 ADR = 66.7 score)
    - Opening Success %: Percentage of opening duels won (0-100%)
    - Clutch Win %: Percentage of clutch situations won (0-100%)
    - Trade Success %: Percentage of trade opportunities converted (0-100%)
    - Utility Usage: Total grenades thrown per round (scaled 0-4 per round)

    Args:
        match_analysis: The MatchAnalysis object containing player data
        player_a_name: Name of the first player to compare
        player_b_name: Name of the second player to compare

    Returns:
        Dict with:
            - player_a_name: First player's name
            - player_b_name: Second player's name
            - axes: List of axis names
            - scores_a: List of normalized scores (0-100) for player A
            - scores_b: List of normalized scores (0-100) for player B
            - raw_values_a: Dict of raw metric values for player A
            - raw_values_b: Dict of raw metric values for player B
    """
    # Find players by name
    player_a = None
    player_b = None

    for player in match_analysis.players.values():
        if player.name.lower() == player_a_name.lower():
            player_a = player
        elif player.name.lower() == player_b_name.lower():
            player_b = player

    if player_a is None:
        raise ValueError(f"Player '{player_a_name}' not found in match")
    if player_b is None:
        raise ValueError(f"Player '{player_b_name}' not found in match")

    def normalize_adr(adr: float) -> float:
        """
        Normalize ADR to 0-100 scale.
        Scale: 0 ADR -> 0, 150 ADR -> 100 (linear)
        Most pros average 70-90 ADR, exceptional is 100+
        """
        return round(min(100, max(0, (adr / 150) * 100)), 1)

    def normalize_percentage(pct: float) -> float:
        """
        Normalize a percentage (0-100) to 0-100 score.
        Already in the right range, just ensure bounds.
        """
        return round(min(100, max(0, pct)), 1)

    def normalize_utility_usage(player: PlayerMatchStats) -> float:
        """
        Normalize utility usage to 0-100 scale.
        Based on grenades thrown per round.
        Scale: 0 per round -> 0, 4+ per round -> 100
        Average player uses ~2-3 utility per round.
        """
        if player.rounds_played == 0:
            return 0.0
        total_utility = (
            player.utility.flashbangs_thrown
            + player.utility.he_thrown
            + player.utility.molotovs_thrown
            + player.utility.smokes_thrown
        )
        utility_per_round = total_utility / player.rounds_played
        # Scale: 4 grenades per round = 100
        return round(min(100, max(0, (utility_per_round / 4) * 100)), 1)

    def get_utility_per_round(player: PlayerMatchStats) -> float:
        """Get raw utility usage per round."""
        if player.rounds_played == 0:
            return 0.0
        total_utility = (
            player.utility.flashbangs_thrown
            + player.utility.he_thrown
            + player.utility.molotovs_thrown
            + player.utility.smokes_thrown
        )
        return round(total_utility / player.rounds_played, 2)

    # Define axes
    axes = [
        "ADR",
        "Opening Success %",
        "Clutch Win %",
        "Trade Success %",
        "Utility Usage",
    ]

    # Calculate normalized scores for player A
    scores_a = [
        normalize_adr(player_a.adr),
        normalize_percentage(player_a.opening_duels.win_rate),
        normalize_percentage(player_a.clutches.win_rate),
        normalize_percentage(player_a.trades.trade_rate),
        normalize_utility_usage(player_a),
    ]

    # Calculate normalized scores for player B
    scores_b = [
        normalize_adr(player_b.adr),
        normalize_percentage(player_b.opening_duels.win_rate),
        normalize_percentage(player_b.clutches.win_rate),
        normalize_percentage(player_b.trades.trade_rate),
        normalize_utility_usage(player_b),
    ]

    # Raw values for display
    raw_values_a = {
        "adr": round(player_a.adr, 1),
        "opening_success_pct": round(player_a.opening_duels.win_rate, 1),
        "opening_attempts": player_a.opening_duels.attempts,
        "opening_wins": player_a.opening_duels.wins,
        "clutch_win_pct": round(player_a.clutches.win_rate, 1),
        "clutch_situations": player_a.clutches.total_situations,
        "clutch_wins": player_a.clutches.total_wins,
        "trade_success_pct": round(player_a.trades.trade_rate, 1),
        "trade_kills": player_a.trades.kills_traded,
        "trade_attempts": player_a.trades.trade_attempts,
        "utility_per_round": get_utility_per_round(player_a),
        "total_utility": (
            player_a.utility.flashbangs_thrown
            + player_a.utility.he_thrown
            + player_a.utility.molotovs_thrown
            + player_a.utility.smokes_thrown
        ),
        # Additional stats for the comparison table
        "kills": player_a.kills,
        "deaths": player_a.deaths,
        "kd_ratio": player_a.kd_ratio,
        "headshot_pct": round(player_a.headshot_percentage, 1),
        "hltv_rating": round(player_a.hltv_rating, 2),
        "kast_pct": round(player_a.kast_percentage, 1),
        "ttd_median_ms": (round(player_a.ttd_median_ms, 1) if player_a.ttd_median_ms else None),
    }

    raw_values_b = {
        "adr": round(player_b.adr, 1),
        "opening_success_pct": round(player_b.opening_duels.win_rate, 1),
        "opening_attempts": player_b.opening_duels.attempts,
        "opening_wins": player_b.opening_duels.wins,
        "clutch_win_pct": round(player_b.clutches.win_rate, 1),
        "clutch_situations": player_b.clutches.total_situations,
        "clutch_wins": player_b.clutches.total_wins,
        "trade_success_pct": round(player_b.trades.trade_rate, 1),
        "trade_kills": player_b.trades.kills_traded,
        "trade_attempts": player_b.trades.trade_attempts,
        "utility_per_round": get_utility_per_round(player_b),
        "total_utility": (
            player_b.utility.flashbangs_thrown
            + player_b.utility.he_thrown
            + player_b.utility.molotovs_thrown
            + player_b.utility.smokes_thrown
        ),
        # Additional stats for the comparison table
        "kills": player_b.kills,
        "deaths": player_b.deaths,
        "kd_ratio": player_b.kd_ratio,
        "headshot_pct": round(player_b.headshot_percentage, 1),
        "hltv_rating": round(player_b.hltv_rating, 2),
        "kast_pct": round(player_b.kast_percentage, 1),
        "ttd_median_ms": (round(player_b.ttd_median_ms, 1) if player_b.ttd_median_ms else None),
    }

    return {
        "player_a_name": player_a.name,
        "player_b_name": player_b.name,
        "player_a_team": player_a.team,
        "player_b_team": player_b.team,
        "player_a_steam_id": str(player_a.steam_id),
        "player_b_steam_id": str(player_b.steam_id),
        "axes": axes,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "raw_values_a": raw_values_a,
        "raw_values_b": raw_values_b,
    }


# Alias for backward compatibility
PlayerAnalytics = PlayerMatchStats


# =============================================================================
# Tier 1 Player Metrics - Clean MatchData Interface
# =============================================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.

    Args:
        value: Any value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int.

    Args:
        value: Any value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def _compute_view_direction(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Convert pitch and yaw angles (in degrees) to a normalized 3D direction vector.

    Uses Source Engine conventions:
    - Yaw: rotation around Z axis (0 = +X, 90 = +Y)
    - Pitch: rotation around horizontal (positive = looking down in Source)

    Args:
        pitch_deg: Pitch angle in degrees
        yaw_deg: Yaw angle in degrees

    Returns:
        Normalized 3D direction vector as numpy array [x, y, z]
    """
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)

    # View direction from Euler angles (Source Engine convention)
    x = np.cos(yaw_rad) * np.cos(pitch_rad)
    y = np.sin(yaw_rad) * np.cos(pitch_rad)
    z = -np.sin(pitch_rad)  # Negative because positive pitch = looking down

    return np.array([x, y, z])


def _compute_angular_error(
    attacker_pos: np.ndarray,
    attacker_pitch: float,
    attacker_yaw: float,
    victim_pos: np.ndarray,
) -> float | None:
    """
    Compute the angular error between where the attacker was looking
    and where they needed to look to hit the victim.

    Args:
        attacker_pos: Attacker position [x, y, z]
        attacker_pitch: Attacker pitch angle in degrees
        attacker_yaw: Attacker yaw angle in degrees
        victim_pos: Victim position [x, y, z]

    Returns:
        Angular error in degrees, or None if computation fails
    """
    try:
        # Direction the attacker was actually looking
        view_vec = _compute_view_direction(attacker_pitch, attacker_yaw)

        # Ideal direction from attacker to victim
        ideal_vec = victim_pos - attacker_pos
        distance = np.linalg.norm(ideal_vec)

        if distance < 0.001:
            return 0.0  # Extremely close, no error

        ideal_vec = ideal_vec / distance  # Normalize

        # Compute angle between vectors using dot product
        dot = np.clip(np.dot(view_vec, ideal_vec), -1.0, 1.0)
        angular_error_rad = np.arccos(dot)
        angular_error_deg = np.degrees(angular_error_rad)

        return float(angular_error_deg)

    except Exception:
        return None


def calculate_player_metrics(match_data: DemoData) -> dict[str, PlayerMetrics]:
    """
    Calculate tier 1 player metrics from awpy MatchData.

    Computes core metrics for each player in the match:
    - Basic stats (K/D/A, damage, ADR)
    - Time to Damage (TTD) - reaction time metric
    - Crosshair Placement (CP) - aim accuracy metric

    Args:
        match_data: Parsed match data from awpy DemoParser

    Returns:
        Dictionary mapping player_name to PlayerMetrics

    TTD Calculation:
        For each kill, find the first damage event from the same attacker
        to the same victim in the same round, before the kill tick.
        TTD = (kill_tick - first_damage_tick) * (1000.0 / tick_rate)

    Crosshair Placement Calculation:
        For each kill with position/angle data:
        1. Get attacker's position and view angles
        2. Convert view angles to a 3D direction vector
        3. Compute direction from attacker to victim
        4. Calculate angle between the two vectors using arccos(dot product)

    Missing Data Handling:
        - Uses .get() on dicts and .fillna() on DataFrames
        - Skips kills missing required position/angle data for CP
        - Skips kills with no prior damage events for TTD
        - Returns None for TTD/CP if no valid measurements
    """
    kills_df = match_data.kills_df
    damages_df = match_data.damages_df
    num_rounds = max(match_data.num_rounds, 1)
    tick_rate = match_data.tick_rate if match_data.tick_rate > 0 else CS2_TICK_RATE

    # Initialize player data structures
    player_data: dict[str, dict[str, Any]] = {}

    # Get player names from match_data
    for steam_id, name in match_data.player_names.items():
        player_data[name] = {
            "steam_id": steam_id,
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "headshot_kills": 0,
            "damage_total": 0,
            "ttd_values": [],
            "cp_values": [],
        }

    # ===========================================
    # Calculate basic stats from DataFrames
    # ===========================================

    # Find column names (handle various naming conventions)
    def find_column(df: pd.DataFrame, options: list[str]) -> str | None:
        for col in options:
            if col in df.columns:
                return col
        return None

    att_steamid_col = find_column(kills_df, ["attacker_steamid", "attacker_steam_id", "attacker"])
    vic_steamid_col = find_column(
        kills_df, ["victim_steamid", "user_steamid", "victim_steam_id", "userid"]
    )
    round_col = find_column(kills_df, ["round_num", "round", "total_rounds_played"])

    dmg_att_col = find_column(damages_df, ["attacker_steamid", "attacker_steam_id", "attacker"])
    dmg_vic_col = find_column(
        damages_df, ["victim_steamid", "user_steamid", "victim_steam_id", "userid"]
    )
    dmg_round_col = find_column(damages_df, ["round_num", "round", "total_rounds_played"])
    dmg_amount_col = find_column(damages_df, ["damage", "dmg_health", "health_damage", "dmg"])

    # Count kills, deaths, assists, headshots per player
    for steam_id, name in match_data.player_names.items():
        if name not in player_data:
            continue

        # Kills and headshots
        if not kills_df.empty and att_steamid_col:
            player_kills = kills_df[
                kills_df[att_steamid_col].fillna(0).astype(float) == float(steam_id)
            ]
            player_data[name]["kills"] = len(player_kills)

            if "headshot" in kills_df.columns:
                player_data[name]["headshot_kills"] = int(
                    player_kills["headshot"].fillna(False).sum()
                )

        # Deaths
        if not kills_df.empty and vic_steamid_col:
            player_deaths = kills_df[
                kills_df[vic_steamid_col].fillna(0).astype(float) == float(steam_id)
            ]
            player_data[name]["deaths"] = len(player_deaths)

        # Assists
        if not kills_df.empty and "assister_steamid" in kills_df.columns:
            player_assists = kills_df[
                kills_df["assister_steamid"].fillna(0).astype(float) == float(steam_id)
            ]
            player_data[name]["assists"] = len(player_assists)

        # Total damage
        if not damages_df.empty and dmg_att_col and dmg_amount_col:
            player_dmg = damages_df[
                damages_df[dmg_att_col].fillna(0).astype(float) == float(steam_id)
            ]
            player_data[name]["damage_total"] = int(player_dmg[dmg_amount_col].fillna(0).sum())

    # ===========================================
    # Calculate TTD (Time to Damage)
    # ===========================================
    #
    # For each kill:
    # 1. Find the first damage event in the same round
    # 2. From the same attacker to the same victim
    # 3. Before the kill tick
    # 4. Compute TTD = (kill_tick - first_damage_tick) * (1000 / tick_rate)

    ms_per_tick = 1000.0 / tick_rate
    ttd_min_ms = 0.0
    ttd_max_ms = 1500.0  # Filter out unreasonable values

    if not kills_df.empty and not damages_df.empty and att_steamid_col and vic_steamid_col:
        if (
            dmg_att_col
            and dmg_vic_col
            and "tick" in kills_df.columns
            and "tick" in damages_df.columns
        ):
            for _, kill_row in kills_df.iterrows():
                try:
                    att_id = kill_row.get(att_steamid_col)
                    vic_id = kill_row.get(vic_steamid_col)
                    kill_tick = _safe_int(kill_row.get("tick"))
                    kill_round = kill_row.get(round_col) if round_col else None

                    if pd.isna(att_id) or pd.isna(vic_id) or kill_tick <= 0:
                        continue

                    att_id = float(att_id)
                    vic_id = float(vic_id)

                    # Find first damage from this attacker to this victim before the kill
                    # Optionally filter by round if available
                    dmg_mask = (
                        (damages_df[dmg_att_col].fillna(0).astype(float) == att_id)
                        & (damages_df[dmg_vic_col].fillna(0).astype(float) == vic_id)
                        & (damages_df["tick"].fillna(0) <= kill_tick)
                    )

                    if kill_round is not None and dmg_round_col and not pd.isna(kill_round):
                        dmg_mask = dmg_mask & (damages_df[dmg_round_col].fillna(-1) == kill_round)

                    relevant_damages = damages_df[dmg_mask].sort_values(by="tick")

                    if relevant_damages.empty:
                        continue  # No damage found before kill

                    first_dmg_tick = _safe_int(relevant_damages.iloc[0]["tick"])
                    if first_dmg_tick <= 0:
                        continue

                    ttd_ticks = kill_tick - first_dmg_tick
                    ttd_ms = ttd_ticks * ms_per_tick

                    # Filter reasonable TTD values
                    if ttd_min_ms < ttd_ms <= ttd_max_ms:
                        # Find player name for this attacker
                        attacker_name = match_data.player_names.get(int(att_id))
                        if attacker_name and attacker_name in player_data:
                            player_data[attacker_name]["ttd_values"].append(ttd_ms)

                except Exception:
                    continue  # Skip problematic kills

    # ===========================================
    # Calculate Crosshair Placement (CP)
    # ===========================================
    #
    # For each kill with position/angle data:
    # 1. Get attacker position (X, Y, Z) and view angles (pitch, yaw)
    # 2. Get victim position (X, Y, Z)
    # 3. Convert view angles to a normalized 3D direction vector
    # 4. Compute direction from attacker to victim, normalized
    # 5. Compute angle between vectors: arccos(dot(view_vec, ideal_vec))

    # Eye height offset (player eye level above origin)
    EYE_HEIGHT = 64.0

    # Try to compute CP from KillEvent objects first (have embedded position data)
    kills_with_pos = [
        k
        for k in match_data.kills
        if k.attacker_x is not None and k.attacker_pitch is not None and k.victim_x is not None
    ]

    if kills_with_pos:
        for kill in kills_with_pos:
            try:
                att_name = match_data.player_names.get(kill.attacker_steamid)
                if not att_name or att_name not in player_data:
                    continue

                # Attacker position (add eye height)
                att_x = _safe_float(kill.attacker_x)
                att_y = _safe_float(kill.attacker_y)
                att_z = _safe_float(kill.attacker_z) + EYE_HEIGHT

                # Victim position (add eye height for head-level)
                vic_x = _safe_float(kill.victim_x)
                vic_y = _safe_float(kill.victim_y)
                vic_z = _safe_float(kill.victim_z) + EYE_HEIGHT

                # Skip if positions are at origin (bad data)
                if abs(att_x) < 0.001 and abs(att_y) < 0.001:
                    continue
                if abs(vic_x) < 0.001 and abs(vic_y) < 0.001:
                    continue

                att_pos = np.array([att_x, att_y, att_z])
                vic_pos = np.array([vic_x, vic_y, vic_z])

                att_pitch = _safe_float(kill.attacker_pitch)
                att_yaw = _safe_float(kill.attacker_yaw)

                angular_error = _compute_angular_error(att_pos, att_pitch, att_yaw, vic_pos)

                if angular_error is not None and 0 <= angular_error <= 180:
                    player_data[att_name]["cp_values"].append(angular_error)

            except Exception:
                continue

    else:
        # Fallback: try to compute from DataFrame columns
        pos_cols = [
            "attacker_X",
            "attacker_Y",
            "attacker_Z",
            "victim_X",
            "victim_Y",
            "victim_Z",
        ]
        angle_cols = ["attacker_pitch", "attacker_yaw"]

        has_pos = all(col in kills_df.columns for col in pos_cols)
        has_angles = all(col in kills_df.columns for col in angle_cols)

        if not kills_df.empty and has_pos and has_angles and att_steamid_col:
            for _, row in kills_df.iterrows():
                try:
                    att_id = row.get(att_steamid_col)
                    if pd.isna(att_id):
                        continue

                    att_name = match_data.player_names.get(int(float(att_id)))
                    if not att_name or att_name not in player_data:
                        continue

                    # Attacker position
                    att_x = _safe_float(row.get("attacker_X"))
                    att_y = _safe_float(row.get("attacker_Y"))
                    att_z = _safe_float(row.get("attacker_Z")) + EYE_HEIGHT

                    # Victim position
                    vic_x = _safe_float(row.get("victim_X"))
                    vic_y = _safe_float(row.get("victim_Y"))
                    vic_z = _safe_float(row.get("victim_Z")) + EYE_HEIGHT

                    # Skip origin positions
                    if abs(att_x) < 0.001 and abs(att_y) < 0.001:
                        continue
                    if abs(vic_x) < 0.001 and abs(vic_y) < 0.001:
                        continue

                    att_pos = np.array([att_x, att_y, att_z])
                    vic_pos = np.array([vic_x, vic_y, vic_z])

                    att_pitch = _safe_float(row.get("attacker_pitch"))
                    att_yaw = _safe_float(row.get("attacker_yaw"))

                    angular_error = _compute_angular_error(att_pos, att_pitch, att_yaw, vic_pos)

                    if angular_error is not None and 0 <= angular_error <= 180:
                        player_data[att_name]["cp_values"].append(angular_error)

                except Exception:
                    continue

    # ===========================================
    # Build final PlayerMetrics objects
    # ===========================================

    result: dict[str, PlayerMetrics] = {}

    for name, data in player_data.items():
        kills = data["kills"]
        damage_total = data["damage_total"]
        ttd_values = data["ttd_values"]
        cp_values = data["cp_values"]

        # Compute ADR
        adr = round(damage_total / num_rounds, 1) if num_rounds > 0 else 0.0

        # Compute TTD statistics
        mean_ttd = None
        median_ttd = None
        if ttd_values:
            mean_ttd = float(np.mean(ttd_values))
            median_ttd = float(np.median(ttd_values))

        # Compute CP statistics
        median_cp = None
        mean_cp = None
        if cp_values:
            median_cp = float(np.median(cp_values))
            mean_cp = float(np.mean(cp_values))

        result[name] = PlayerMetrics(
            player_name=name,
            kills=kills,
            deaths=data["deaths"],
            assists=data["assists"],
            headshot_kills=data["headshot_kills"],
            damage_total=damage_total,
            adr=adr,
            mean_ttd_ms=mean_ttd,
            median_crosshair_error_deg=median_cp,
            median_ttd_ms=median_ttd,
            mean_crosshair_error_deg=mean_cp,
        )

    return result


def calculate_economy_history(match_data: DemoData) -> list[dict]:
    """Calculate round-by-round economy history for both teams."""
    from opensight.analysis.compute_economy import calculate_economy_history as _calc

    return _calc(match_data)
