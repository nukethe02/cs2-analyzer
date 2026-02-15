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
from typing import Any

import pandas as pd

from opensight.analysis.models import (  # noqa: F401
    AimStats,
    ClutchEvent,
    ClutchStats,
    CrosshairPlacementResult,
    EngagementResult,
    EntryFragStats,
    KillMatrixEntry,
    MatchAnalysis,
    OpeningDuelStats,
    OpeningEngagementStats,
    PlayerMatchStats,
    RoundMomentum,
    RoundTimeline,
    TradeStats,
    TTDResult,
    UtilityMetrics,
    UtilityStats,
    ValidationResult,
    WinProbEvent,
    calculate_win_probability,
)
from opensight.core.constants import (
    CS2_TICK_RATE,
)
from opensight.core.parser import DemoData, safe_int

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
        "user_team",
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
        - NaN/None: Returns 'Unknown'
        """
        if value is None:
            return "Unknown"
        # Handle NaN (pandas/numpy float NaN passes isinstance check but crashes int())
        try:
            if pd.isna(value):
                return "Unknown"
        except (TypeError, ValueError):
            pass
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
        # Prefer dmg_health_real (theoretical damage per hit) — Leetify/HLTV use this
        # for ADR. We then clip at 100 per hit to cap overkill (e.g. AWP headshot = 400+
        # becomes 100). Do NOT use dmg_health here: it caps at victim's remaining HP,
        # which double-caps and produces ADR values lower than Leetify.
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

        # Pre-compute flash assist column for assist filtering
        fa_col = self._find_col(kills_df, ["assistedflash", "flash_assist", "is_flash_assist"])

        for steam_id, player in self._players.items():
            # Kills - use cached column
            if not kills_df.empty and att_id_col in kills_df.columns:
                # Convert to same type for comparison (handle float vs int issue)
                player_kills = kills_df[kills_df[att_id_col].astype(float) == float(steam_id)]
                # Exclude suicides (attacker == victim, e.g. fall damage / world kills)
                if vic_id_col in kills_df.columns:
                    player_kills = player_kills[
                        player_kills[att_id_col].astype(float)
                        != player_kills[vic_id_col].astype(float)
                    ]
                player.kills = len(player_kills)

                if "headshot" in kills_df.columns:
                    player.headshots = int(player_kills["headshot"].sum())

                if "weapon" in kills_df.columns:
                    player.weapon_kills = player_kills["weapon"].value_counts().to_dict()

            # Deaths - use cached column (handles user_steamid vs victim_steamid)
            if not kills_df.empty and vic_id_col in kills_df.columns:
                player.deaths = len(kills_df[kills_df[vic_id_col].astype(float) == float(steam_id)])

            # Assists — exclude flash assists (Leetify counts damage-only assists)
            if not kills_df.empty and "assister_steamid" in kills_df.columns:
                assister_mask = kills_df["assister_steamid"].astype(float) == float(steam_id)
                if fa_col and fa_col in kills_df.columns:
                    # Damage assists only: assister matches AND NOT a flash assist
                    player.assists = len(kills_df[assister_mask & ~kills_df[fa_col]])
                else:
                    player.assists = len(kills_df[assister_mask])

            # Damage - use dynamic column finding (filter self/team damage for accurate ADR)
            if dmg_att_col and dmg_col:
                player_dmg = damages_df[damages_df[dmg_att_col].astype(float) == float(steam_id)]
                # Filter out self-damage
                dmg_vic_col = self._find_col(damages_df, self.VIC_ID_COLS)
                if dmg_vic_col and dmg_vic_col in damages_df.columns:
                    player_dmg = player_dmg[
                        player_dmg[dmg_vic_col].astype(float) != float(steam_id)
                    ]
                # Filter out team damage (attacker_team == victim_team)
                dmg_att_team_col = self._find_col(damages_df, self.ATT_SIDE_COLS)
                dmg_vic_team_col = self._find_col(damages_df, self.VIC_SIDE_COLS)
                if dmg_att_team_col and dmg_vic_team_col:
                    player_dmg = player_dmg[
                        player_dmg[dmg_att_team_col] != player_dmg[dmg_vic_team_col]
                    ]
                # Cap damage at 100 per hit (no overkill) to match Leetify/HLTV ADR
                player.total_damage = int(player_dmg[dmg_col].clip(upper=100).sum())

            # Flash assists (handle demoparser2 column name variants)
            fa_col = self._find_col(kills_df, ["flash_assist", "assistedflash", "is_flash_assist"])
            if not kills_df.empty and fa_col and "assister_steamid" in kills_df.columns:
                flash_assists = kills_df[
                    (kills_df["assister_steamid"].astype(float) == float(steam_id))
                    & (kills_df[fa_col])
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
        from opensight.analysis.compute_combat import calculate_multi_kills

        calculate_multi_kills(self)

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
        from opensight.analysis.compute_combat import detect_opening_duels

        detect_opening_duels(self)

    def _detect_opening_engagements(self) -> None:
        """Detect opening engagements - who FOUGHT first, not just who DIED first.

        Identifies:
        1. First damage tick of each round
        2. All players who dealt/took damage before first kill
        3. Opening phase damage totals

        This captures true engagement participation even when a player
        initiates combat but doesn't secure the kill.
        """
        from opensight.analysis.compute_combat import detect_opening_engagements

        detect_opening_engagements(self)

    def _detect_entry_frags(self) -> None:
        """Detect zone-aware entry frags using position data.

        Entry Frag = First kill in a specific bombsite for a round.
        Uses get_zone_for_position() to classify kill locations.

        This distinguishes:
        - Map control kills: First kills in mid/connectors/routes
        - Entry frags: First kills inside bombsite zones
        """
        from opensight.analysis.compute_combat import detect_entry_frags

        detect_entry_frags(self)

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
        from opensight.analysis.compute_combat import detect_trades

        detect_trades(self)

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
        from opensight.analysis.compute_combat import detect_clutches

        detect_clutches(self)

    def _calculate_kast(self) -> None:
        """Calculate KAST (Kill/Assist/Survived/Traded) for each player using optimized lookups."""
        from opensight.analysis.compute_economy import calculate_kast

        calculate_kast(self)

    def _compute_ttd(self) -> None:
        """Compute Time to Damage for each kill with optimized indexing."""
        from opensight.analysis.compute_aim import compute_ttd

        compute_ttd(self)

    def _compute_crosshair_placement(self) -> None:
        """Compute crosshair placement error for each kill.

        Uses vectorized numpy implementation when available for ~5-20x speedup.
        Falls back to per-kill loop for compatibility.
        """
        from opensight.analysis.compute_aim import compute_crosshair_placement

        compute_crosshair_placement(self)

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
        from opensight.analysis.compute_aim import calculate_accuracy_stats

        calculate_accuracy_stats(self)

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
        from opensight.analysis.compute_combat import detect_greedy_repeeks

        detect_greedy_repeeks(self)

    def _integrate_economy(self) -> dict:
        """Integrate economy module data into player stats."""
        if not HAS_ECONOMY:
            logger.debug("Economy module not available, skipping integration")
            return {}

        try:
            # NOTE(perf): single-instantiation here; orchestrator creates a separate
            # instance for timeline enrichment (see orchestrator.py _build_round_timeline).
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
            # NOTE(perf): single-instantiation here; orchestrator creates a separate
            # instance for timeline enrichment (see orchestrator.py _build_round_timeline).
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


# Alias for backward compatibility
PlayerAnalytics = PlayerMatchStats


def calculate_economy_history(match_data: DemoData) -> list[dict]:
    """Calculate round-by-round economy history for both teams."""
    from opensight.analysis.compute_economy import calculate_economy_history as _calc

    return _calc(match_data)
