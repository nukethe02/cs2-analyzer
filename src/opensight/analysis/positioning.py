"""
Per-player positional heatmap analysis.

This module provides individual player positioning analysis that goes beyond
Leetify's team-aggregate heatmaps. Features include:
- Per-player presence heatmaps (where they spend time)
- Kill/death position heatmaps
- Early vs late round positioning split
- Favorite/danger zone detection
- Player positioning comparison

Author: OpenSight Team
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from opensight.core.parser import DemoData

logger = logging.getLogger(__name__)

# Type aliases for clarity
Position3D = tuple[float, float, float]
Position4D = tuple[float, float, float, int]  # x, y, z, tick
HeatmapGrid = list[list[float]]
ZoneDistribution = dict[str, float]
ZoneCounts = dict[str, int]

# Return type for JSON serialization
HeatmapDict = dict[str, str | int | float | list[str] | HeatmapGrid | ZoneDistribution | ZoneCounts]
ComparisonDict = dict[str, HeatmapDict | float | list[str]]


@dataclass
class PlayerHeatmapData:
    """Heatmap data for a single player."""

    steamid: int
    player_name: str
    map_name: str

    # 64x64 heatmaps (normalized 0-1)
    presence_heatmap: list[list[float]] = field(default_factory=list)
    kills_heatmap: list[list[float]] = field(default_factory=list)
    deaths_heatmap: list[list[float]] = field(default_factory=list)
    early_round_heatmap: list[list[float]] = field(default_factory=list)
    late_round_heatmap: list[list[float]] = field(default_factory=list)

    # Zone analysis
    favorite_zones: list[str] = field(default_factory=list)
    danger_zones: list[str] = field(default_factory=list)
    zone_time_distribution: dict[str, float] = field(default_factory=dict)
    zone_kill_distribution: dict[str, int] = field(default_factory=dict)
    zone_death_distribution: dict[str, int] = field(default_factory=dict)

    # Stats
    total_positions: int = 0
    total_kills: int = 0
    total_deaths: int = 0

    def to_dict(self) -> HeatmapDict:
        """Convert to JSON-serializable dict."""
        return {
            "steamid": str(self.steamid),
            "player_name": self.player_name,
            "map_name": self.map_name,
            "presence_heatmap": self.presence_heatmap,
            "kills_heatmap": self.kills_heatmap,
            "deaths_heatmap": self.deaths_heatmap,
            "early_round_heatmap": self.early_round_heatmap,
            "late_round_heatmap": self.late_round_heatmap,
            "favorite_zones": self.favorite_zones,
            "danger_zones": self.danger_zones,
            "zone_time_distribution": self.zone_time_distribution,
            "zone_kill_distribution": self.zone_kill_distribution,
            "zone_death_distribution": self.zone_death_distribution,
            "total_positions": self.total_positions,
            "total_kills": self.total_kills,
            "total_deaths": self.total_deaths,
        }


@dataclass
class PlayerComparison:
    """Comparison of two players' positioning."""

    player_a: PlayerHeatmapData
    player_b: PlayerHeatmapData
    overlap_score: float  # 0-100, how similar their positioning
    unique_zones_a: list[str]  # Zones where A is but B isn't
    unique_zones_b: list[str]  # Zones where B is but A isn't
    shared_zones: list[str]  # Zones where both spend significant time

    def to_dict(self) -> ComparisonDict:
        """Convert to JSON-serializable dict."""
        return {
            "player_a": self.player_a.to_dict(),
            "player_b": self.player_b.to_dict(),
            "overlap_score": round(self.overlap_score, 1),
            "unique_zones_a": self.unique_zones_a,
            "unique_zones_b": self.unique_zones_b,
            "shared_zones": self.shared_zones,
        }


class PositioningAnalyzer:
    """
    Analyze per-player positioning patterns.

    Creates heatmaps showing:
    - Where a player spends time (presence)
    - Where they get kills
    - Where they die
    - Early vs late round positioning

    This is more useful than Leetify's team-aggregate heatmaps
    for scouting specific opponents.
    """

    # Heatmap configuration
    HEATMAP_RESOLUTION = 64  # 64x64 grid
    HEATMAP_MAX_VALUE = 1.0  # Normalized max value

    # Time thresholds
    EARLY_ROUND_SECONDS = 30  # First 30 seconds = "early round"
    TICK_RATE = 64  # Standard tick rate
    EARLY_ROUND_TICKS = EARLY_ROUND_SECONDS * TICK_RATE

    # Zone significance threshold (% of time to be considered "favorite")
    ZONE_SIGNIFICANCE_THRESHOLD = 0.05  # 5% of time
    TOP_ZONES_COUNT = 5  # Number of top zones to report

    # DataFrame column name variants
    STEAMID_COLUMNS = ("steamid", "steam_id", "user_steamid")

    # Default map bounds (used when metadata unavailable)
    DEFAULT_MAP_BOUNDS = {"x_min": -3000, "x_max": 3000, "y_min": -3000, "y_max": 3000}

    def __init__(self, demo_data: DemoData):
        self.data = demo_data
        self.map_name = getattr(demo_data, "map_name", "").lower()
        self._map_bounds = self._get_map_bounds()
        self._player_names = getattr(demo_data, "player_names", {})

        # Import zone detection
        try:
            from opensight.visualization.radar import MAP_ZONES, get_zone_for_position

            self._get_zone = get_zone_for_position
            self._has_zones = self.map_name in MAP_ZONES
        except ImportError:
            self._get_zone = lambda m, x, y, z=None: "World"
            self._has_zones = False

    def _get_map_bounds(self) -> dict[str, float]:
        """Get map bounds for coordinate normalization."""
        try:
            from opensight.visualization.radar import MAP_METADATA

            if self.map_name in MAP_METADATA:
                meta = MAP_METADATA[self.map_name]
                # Calculate approximate bounds from radar metadata
                # Typical CS2 map is ~4000x4000 units
                pos_x = meta.get("pos_x", -2000)
                pos_y = meta.get("pos_y", 2000)
                scale = meta.get("scale", 4.0)
                # Radar is typically 1024x1024 pixels
                width = 1024 * scale
                return {
                    "x_min": pos_x,
                    "x_max": pos_x + width,
                    "y_min": pos_y - width,  # Y is inverted in CS2
                    "y_max": pos_y,
                }
        except ImportError:
            pass

        # Default bounds for unknown maps
        return {"x_min": -3000, "x_max": 3000, "y_min": -3000, "y_max": 3000}

    def analyze_player(self, steamid: int) -> PlayerHeatmapData:
        """
        Generate comprehensive position analysis for a single player.

        Args:
            steamid: Player's Steam ID

        Returns:
            PlayerHeatmapData with all heatmaps and zone analysis
        """
        player_name = self._player_names.get(steamid, f"Player_{steamid}")

        # Collect positions
        all_positions = self._get_player_positions(steamid)
        kill_positions = self._get_player_kill_positions(steamid)
        death_positions = self._get_player_death_positions(steamid)
        early_positions, late_positions = self._split_early_late(all_positions)

        # Generate heatmaps (vectorized numpy operations)
        presence_hm = self._create_heatmap([p[:2] for p in all_positions])
        kills_hm = self._create_heatmap([p[:2] for p in kill_positions])
        deaths_hm = self._create_heatmap([p[:2] for p in death_positions])
        early_hm = self._create_heatmap([p[:2] for p in early_positions])
        late_hm = self._create_heatmap([p[:2] for p in late_positions])

        # Zone analysis
        zone_time = self._calculate_zone_distribution(all_positions)
        zone_kills = self._calculate_zone_counts(kill_positions)
        zone_deaths = self._calculate_zone_counts(death_positions)

        # Find favorite and danger zones
        favorite_zones = self._find_top_zones(zone_time, top_n=5)
        danger_zones = self._find_top_zones(zone_deaths, top_n=5)

        return PlayerHeatmapData(
            steamid=steamid,
            player_name=player_name,
            map_name=self.map_name,
            presence_heatmap=presence_hm,
            kills_heatmap=kills_hm,
            deaths_heatmap=deaths_hm,
            early_round_heatmap=early_hm,
            late_round_heatmap=late_hm,
            favorite_zones=favorite_zones,
            danger_zones=danger_zones,
            zone_time_distribution=zone_time,
            zone_kill_distribution=zone_kills,
            zone_death_distribution=zone_deaths,
            total_positions=len(all_positions),
            total_kills=len(kill_positions),
            total_deaths=len(death_positions),
        )

    def compare_players(self, steamid_a: int, steamid_b: int) -> PlayerComparison:
        """
        Compare positioning of two players.

        Useful for:
        - Scouting opponent tendencies
        - Comparing teammate positioning overlap
        - Finding unique spots each player uses

        Args:
            steamid_a: First player's Steam ID
            steamid_b: Second player's Steam ID

        Returns:
            PlayerComparison with overlap analysis
        """
        data_a = self.analyze_player(steamid_a)
        data_b = self.analyze_player(steamid_b)

        # Calculate overlap score
        overlap = self._calculate_heatmap_overlap(data_a.presence_heatmap, data_b.presence_heatmap)

        # Find unique and shared zones
        unique_a, unique_b, shared = self._find_zone_differences(
            data_a.zone_time_distribution, data_b.zone_time_distribution
        )

        return PlayerComparison(
            player_a=data_a,
            player_b=data_b,
            overlap_score=overlap,
            unique_zones_a=unique_a,
            unique_zones_b=unique_b,
            shared_zones=shared,
        )

    def analyze_all_players(self) -> dict[int, PlayerHeatmapData]:
        """Analyze positioning for all players in the demo."""
        results = {}
        for steamid in self._player_names:
            try:
                results[steamid] = self.analyze_player(steamid)
            except Exception as e:
                logger.warning(f"Failed to analyze player {steamid}: {e}")
        return results

    # =========================================================================
    # Position Collection Methods
    # =========================================================================

    def _get_player_positions(self, steamid: int) -> list[Position4D]:
        """
        Get all positions for a player from tick data.

        Returns list of (x, y, z, tick) tuples.
        Fully vectorized - no iterrows().
        """
        positions: list[Position4D] = []

        # Try tick-level data first (most granular)
        ticks_df = getattr(self.data, "ticks_df", None)
        if ticks_df is not None and not ticks_df.empty:
            # Find steamid column
            steamid_col = None
            for col in self.STEAMID_COLUMNS:
                if col in ticks_df.columns:
                    steamid_col = col
                    break

            if steamid_col and "X" in ticks_df.columns and "Y" in ticks_df.columns:
                # Filter to player's ticks
                player_mask = ticks_df[steamid_col] == steamid
                player_ticks = ticks_df.loc[player_mask]

                if not player_ticks.empty:
                    # Vectorized extraction using numpy arrays
                    x_arr = player_ticks["X"].to_numpy()
                    y_arr = player_ticks["Y"].to_numpy()
                    z_arr = (
                        player_ticks["Z"].to_numpy()
                        if "Z" in player_ticks.columns
                        else np.zeros(len(x_arr))
                    )
                    tick_arr = (
                        player_ticks["tick"].to_numpy()
                        if "tick" in player_ticks.columns
                        else np.zeros(len(x_arr), dtype=int)
                    )

                    # Filter out NaN values (vectorized)
                    valid_mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
                    x_valid = x_arr[valid_mask]
                    y_valid = y_arr[valid_mask]
                    z_valid = np.nan_to_num(z_arr[valid_mask], nan=0.0)
                    tick_valid = np.nan_to_num(tick_arr[valid_mask], nan=0).astype(int)

                    # Convert to list of tuples (vectorized zip is faster than iterrows)
                    positions = list(
                        zip(
                            x_valid.astype(float),
                            y_valid.astype(float),
                            z_valid.astype(float),
                            tick_valid,
                            strict=False,
                        )
                    )

        # Fallback: Extract positions from kill/death events
        if not positions:
            positions = self._get_positions_from_events(steamid)

        return positions

    def _get_positions_from_events(self, steamid: int) -> list[Position4D]:
        """Extract approximate positions from kill/death events when tick data unavailable."""
        positions: list[Position4D] = []
        kills = getattr(self.data, "kills", [])

        for kill in kills:
            # Attacker position
            if getattr(kill, "attacker_steamid", None) == steamid:
                x = getattr(kill, "attacker_x", None)
                y = getattr(kill, "attacker_y", None)
                z = getattr(kill, "attacker_z", 0)
                tick = getattr(kill, "tick", 0)
                if x is not None and y is not None:
                    positions.append((float(x), float(y), float(z or 0), int(tick or 0)))

            # Victim position
            if getattr(kill, "victim_steamid", None) == steamid:
                x = getattr(kill, "victim_x", None)
                y = getattr(kill, "victim_y", None)
                z = getattr(kill, "victim_z", 0)
                tick = getattr(kill, "tick", 0)
                if x is not None and y is not None:
                    positions.append((float(x), float(y), float(z or 0), int(tick or 0)))

        return positions

    def _get_player_kill_positions(self, steamid: int) -> list[Position3D]:
        """Get positions where player got kills."""
        positions: list[Position3D] = []
        kills = getattr(self.data, "kills", [])

        for kill in kills:
            if getattr(kill, "attacker_steamid", None) == steamid:
                x = getattr(kill, "attacker_x", None)
                y = getattr(kill, "attacker_y", None)
                z = getattr(kill, "attacker_z", 0)
                if x is not None and y is not None:
                    positions.append((float(x), float(y), float(z or 0)))

        return positions

    def _get_player_death_positions(self, steamid: int) -> list[Position3D]:
        """Get positions where player died."""
        positions: list[Position3D] = []
        kills = getattr(self.data, "kills", [])

        for kill in kills:
            if getattr(kill, "victim_steamid", None) == steamid:
                x = getattr(kill, "victim_x", None)
                y = getattr(kill, "victim_y", None)
                z = getattr(kill, "victim_z", 0)
                if x is not None and y is not None:
                    positions.append((float(x), float(y), float(z or 0)))

        return positions

    def _split_early_late(
        self, positions: list[Position4D]
    ) -> tuple[list[Position4D], list[Position4D]]:
        """Split positions into early round (first 30s) and late round."""
        # Get round start ticks
        rounds = getattr(self.data, "rounds", [])
        round_starts = {}
        for r in rounds:
            round_num = getattr(r, "round_num", 0)
            start_tick = getattr(r, "start_tick", 0)
            if round_num and start_tick:
                round_starts[round_num] = start_tick

        early = []
        late = []

        for x, y, z, tick in positions:
            # Find which round this tick belongs to
            round_start = 0
            for _rn, start in round_starts.items():
                if start <= tick:
                    round_start = max(round_start, start)

            # Calculate time into round
            ticks_into_round = tick - round_start

            if ticks_into_round <= self.EARLY_ROUND_TICKS:
                early.append((x, y, z, tick))
            else:
                late.append((x, y, z, tick))

        return early, late

    # =========================================================================
    # Heatmap Generation (Vectorized)
    # =========================================================================

    def _create_heatmap(self, positions: list[tuple[float, float]]) -> HeatmapGrid:
        """
        Create a 64x64 heatmap from positions using numpy histogram2d.

        Fully vectorized - no Python loops.
        """
        if not positions:
            return [[0.0] * self.HEATMAP_RESOLUTION for _ in range(self.HEATMAP_RESOLUTION)]

        # Convert to numpy array
        pts = np.array(positions, dtype=np.float64)

        # Extract bounds
        x_min, x_max = self._map_bounds["x_min"], self._map_bounds["x_max"]
        y_min, y_max = self._map_bounds["y_min"], self._map_bounds["y_max"]

        # Normalize coordinates to heatmap grid (0 to resolution-1)
        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range <= 0 or y_range <= 0:
            return [[0.0] * self.HEATMAP_RESOLUTION for _ in range(self.HEATMAP_RESOLUTION)]

        x_norm = (pts[:, 0] - x_min) / x_range * (self.HEATMAP_RESOLUTION - 1)
        y_norm = (pts[:, 1] - y_min) / y_range * (self.HEATMAP_RESOLUTION - 1)

        # Clip to valid range
        x_norm = np.clip(x_norm, 0, self.HEATMAP_RESOLUTION - 1)
        y_norm = np.clip(y_norm, 0, self.HEATMAP_RESOLUTION - 1)

        # Create 2D histogram
        heatmap, _, _ = np.histogram2d(
            x_norm,
            y_norm,
            bins=self.HEATMAP_RESOLUTION,
            range=[[0, self.HEATMAP_RESOLUTION], [0, self.HEATMAP_RESOLUTION]],
        )

        # Normalize to 0-1
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.tolist()

    # =========================================================================
    # Zone Analysis
    # =========================================================================

    def _calculate_zone_distribution(self, positions: list[Position4D]) -> ZoneDistribution:
        """Calculate percentage of time spent in each zone."""
        if not positions:
            return {}

        zone_counts: dict[str, int] = defaultdict(int)
        total = len(positions)

        for x, y, z, _ in positions:
            zone = self._get_zone(self.map_name, x, y, z)
            zone_counts[zone] += 1

        # Convert to percentages
        return {zone: round(count / total * 100, 1) for zone, count in zone_counts.items()}

    def _calculate_zone_counts(self, positions: list[Position3D]) -> ZoneCounts:
        """Count events in each zone."""
        zone_counts: dict[str, int] = defaultdict(int)

        for x, y, z in positions:
            zone = self._get_zone(self.map_name, x, y, z)
            zone_counts[zone] += 1

        return dict(zone_counts)

    def _find_top_zones(
        self, zone_data: ZoneDistribution | ZoneCounts, top_n: int | None = None
    ) -> list[str]:
        """Find top N zones by value."""
        if top_n is None:
            top_n = self.TOP_ZONES_COUNT
        sorted_zones = sorted(zone_data.items(), key=lambda x: x[1], reverse=True)
        return [zone for zone, _ in sorted_zones[:top_n] if zone != "World"]

    def _find_zone_differences(
        self, zones_a: ZoneDistribution, zones_b: ZoneDistribution
    ) -> tuple[list[str], list[str], list[str]]:
        """Find unique zones for each player and shared zones."""
        threshold = self.ZONE_SIGNIFICANCE_THRESHOLD * 100  # Convert to percentage

        # Significant zones for each player
        sig_a = {z for z, pct in zones_a.items() if pct >= threshold and z != "World"}
        sig_b = {z for z, pct in zones_b.items() if pct >= threshold and z != "World"}

        unique_a = list(sig_a - sig_b)
        unique_b = list(sig_b - sig_a)
        shared = list(sig_a & sig_b)

        return unique_a, unique_b, shared

    # =========================================================================
    # Heatmap Comparison
    # =========================================================================

    def _calculate_heatmap_overlap(self, heatmap_a: HeatmapGrid, heatmap_b: HeatmapGrid) -> float:
        """
        Calculate overlap score between two heatmaps (0-100).

        Uses cosine similarity on flattened heatmaps.
        """
        if not heatmap_a or not heatmap_b:
            return 0.0

        # Convert to numpy arrays
        a = np.array(heatmap_a).flatten()
        b = np.array(heatmap_b).flatten()

        # Handle zero vectors
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(a, b) / (norm_a * norm_b)

        # Convert to 0-100 scale
        return round(float(similarity) * 100, 1)


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_player_positioning(demo_data: DemoData, steamid: int) -> HeatmapDict:
    """Convenience function to analyze a single player's positioning."""
    analyzer = PositioningAnalyzer(demo_data)
    result = analyzer.analyze_player(steamid)
    return result.to_dict()


def compare_player_positioning(
    demo_data: DemoData, steamid_a: int, steamid_b: int
) -> ComparisonDict:
    """Convenience function to compare two players' positioning."""
    analyzer = PositioningAnalyzer(demo_data)
    result = analyzer.compare_players(steamid_a, steamid_b)
    return result.to_dict()


def get_all_player_heatmaps(demo_data: DemoData) -> dict[str, HeatmapDict]:
    """Get heatmap data for all players in the demo."""
    analyzer = PositioningAnalyzer(demo_data)
    results = analyzer.analyze_all_players()
    return {str(sid): data.to_dict() for sid, data in results.items()}
