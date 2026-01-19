"""
Grenade Trajectory Analysis Module for CS2 Demo Analysis

Provides:
- Grenade trajectory tracking and analysis
- Utility usage metrics per player
- Flashbang efficiency calculation
- Grenade landing position data for visualization

Color coding for visualization:
- Purple (#9b59b6): Smoke grenades
- Yellow (#f1c40f): Flashbangs
- Orange (#e67e22): Molotov/Incendiary
- Red (#e74c3c): HE grenades
- Gray (#95a5a6): Decoy
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from opensight.parser import DemoData

logger = logging.getLogger(__name__)


class GrenadeType(Enum):
    """Types of grenades in CS2 with display colors."""
    FLASHBANG = "flashbang"
    HE_GRENADE = "hegrenade"
    SMOKE = "smokegrenade"
    MOLOTOV = "molotov"
    INCENDIARY = "incgrenade"
    DECOY = "decoy"


# Grenade colors for visualization (hex colors)
GRENADE_COLORS = {
    "flashbang": "#f1c40f",      # Yellow
    "smokegrenade": "#9b59b6",   # Purple
    "molotov": "#e67e22",        # Orange
    "incgrenade": "#e67e22",     # Orange (same as molotov)
    "hegrenade": "#e74c3c",      # Red
    "decoy": "#95a5a6",          # Gray
}

# Grenade CSS classes for styling
GRENADE_CSS_CLASSES = {
    "flashbang": "grenade-flash",
    "smokegrenade": "grenade-smoke",
    "molotov": "grenade-molotov",
    "incgrenade": "grenade-molotov",
    "hegrenade": "grenade-he",
    "decoy": "grenade-decoy",
}


@dataclass
class GrenadePosition:
    """A grenade landing/detonation position for visualization."""
    x: float  # Radar pixel X
    y: float  # Radar pixel Y
    game_x: float  # Game X coordinate
    game_y: float  # Game Y coordinate
    game_z: float  # Game Z coordinate
    grenade_type: str
    thrower_steamid: int
    thrower_name: str
    thrower_team: str
    round_num: int
    tick: int
    event_type: str  # 'thrown', 'detonate', 'expire'
    color: str = ""  # Hex color for visualization
    css_class: str = ""  # CSS class for styling

    def __post_init__(self):
        """Set color and CSS class based on grenade type."""
        if not self.color:
            self.color = GRENADE_COLORS.get(self.grenade_type.lower(), "#ffffff")
        if not self.css_class:
            self.css_class = GRENADE_CSS_CLASSES.get(self.grenade_type.lower(), "grenade-unknown")


@dataclass
class PlayerUtilityUsage:
    """Utility usage statistics for a single player."""
    steam_id: int
    name: str
    team: str

    # Grenade counts
    flashbangs_thrown: int = 0
    smokes_thrown: int = 0
    he_grenades_thrown: int = 0
    molotovs_thrown: int = 0
    decoys_thrown: int = 0

    # Flash effectiveness
    enemies_flashed: int = 0
    teammates_flashed: int = 0
    total_blind_duration: float = 0.0  # Total seconds enemies were blinded

    # HE grenade effectiveness
    he_damage: int = 0
    he_enemies_hit: int = 0
    he_teammates_hit: int = 0

    # Molotov effectiveness
    molotov_damage: int = 0

    # Grenade positions for visualization
    grenade_positions: list[GrenadePosition] = field(default_factory=list)

    @property
    def total_utility(self) -> int:
        """Total utility thrown."""
        return (
            self.flashbangs_thrown +
            self.smokes_thrown +
            self.he_grenades_thrown +
            self.molotovs_thrown +
            self.decoys_thrown
        )

    @property
    def flashbang_efficiency(self) -> float:
        """
        Flashbang efficiency: enemies flashed per flashbang thrown.
        Higher is better. Pro average is ~0.5-0.7.
        """
        if self.flashbangs_thrown <= 0:
            return 0.0
        return round(self.enemies_flashed / self.flashbangs_thrown, 2)

    @property
    def flashbang_team_ratio(self) -> float:
        """
        Ratio of teammates flashed vs enemies flashed.
        Lower is better. Above 0.5 means flashing teammates more than enemies (bad).
        """
        if self.enemies_flashed <= 0:
            return 0.0 if self.teammates_flashed == 0 else float('inf')
        return round(self.teammates_flashed / self.enemies_flashed, 2)

    @property
    def avg_blind_duration(self) -> float:
        """Average blind duration per enemy flashed (seconds)."""
        if self.enemies_flashed <= 0:
            return 0.0
        return round(self.total_blind_duration / self.enemies_flashed, 2)

    @property
    def he_damage_per_nade(self) -> float:
        """Average HE grenade damage."""
        if self.he_grenades_thrown <= 0:
            return 0.0
        return round(self.he_damage / self.he_grenades_thrown, 1)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "steam_id": str(self.steam_id),
            "name": self.name,
            "team": self.team,
            "utility_usage": {
                "total": self.total_utility,
                "flashbangs": self.flashbangs_thrown,
                "smokes": self.smokes_thrown,
                "he_grenades": self.he_grenades_thrown,
                "molotovs": self.molotovs_thrown,
                "decoys": self.decoys_thrown,
            },
            "flash_effectiveness": {
                "enemies_flashed": self.enemies_flashed,
                "teammates_flashed": self.teammates_flashed,
                "efficiency": self.flashbang_efficiency,
                "team_ratio": self.flashbang_team_ratio,
                "avg_blind_duration": self.avg_blind_duration,
            },
            "he_effectiveness": {
                "damage": self.he_damage,
                "enemies_hit": self.he_enemies_hit,
                "teammates_hit": self.he_teammates_hit,
                "damage_per_nade": self.he_damage_per_nade,
            },
            "molotov_damage": self.molotov_damage,
        }


@dataclass
class GrenadeTrajectoryAnalysis:
    """Complete grenade trajectory analysis for a match."""
    player_stats: dict[int, PlayerUtilityUsage]
    all_positions: list[GrenadePosition]

    # Team-level stats
    team_stats: dict[str, dict] = field(default_factory=dict)

    # Summary stats
    total_grenades: int = 0
    total_flashes: int = 0
    total_smokes: int = 0
    total_molotovs: int = 0
    total_he: int = 0

    def get_positions_for_player(self, steam_id: int) -> list[GrenadePosition]:
        """Get grenade positions for a specific player."""
        return [p for p in self.all_positions if p.thrower_steamid == steam_id]

    def get_positions_by_type(self, grenade_type: str) -> list[GrenadePosition]:
        """Get grenade positions filtered by type."""
        return [p for p in self.all_positions if p.grenade_type.lower() == grenade_type.lower()]

    def get_positions_for_round(self, round_num: int) -> list[GrenadePosition]:
        """Get grenade positions for a specific round."""
        return [p for p in self.all_positions if p.round_num == round_num]

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "summary": {
                "total_grenades": self.total_grenades,
                "flashbangs": self.total_flashes,
                "smokes": self.total_smokes,
                "molotovs": self.total_molotovs,
                "he_grenades": self.total_he,
            },
            "player_stats": {
                str(sid): stats.to_dict()
                for sid, stats in self.player_stats.items()
            },
            "team_stats": self.team_stats,
            "positions": [
                {
                    "x": p.x,
                    "y": p.y,
                    "game_x": round(p.game_x, 1),
                    "game_y": round(p.game_y, 1),
                    "grenade_type": p.grenade_type,
                    "thrower_steamid": str(p.thrower_steamid),
                    "thrower_name": p.thrower_name,
                    "round_num": p.round_num,
                    "color": p.color,
                    "css_class": p.css_class,
                }
                for p in self.all_positions
            ],
        }


class GrenadeTrajectoryAnalyzer:
    """Analyzer for grenade trajectory and utility effectiveness."""

    def __init__(self, demo_data: DemoData):
        """
        Initialize the grenade trajectory analyzer.

        Args:
            demo_data: Parsed demo data containing grenade and blind events.
        """
        self.data = demo_data
        self._player_stats: dict[int, PlayerUtilityUsage] = {}
        self._positions: list[GrenadePosition] = []

    def analyze(self, transformer=None) -> GrenadeTrajectoryAnalysis:
        """
        Run full grenade trajectory analysis.

        Args:
            transformer: Optional CoordinateTransformer for radar positions.
                        If None, radar coordinates will be 0.

        Returns:
            GrenadeTrajectoryAnalysis with all utility metrics.
        """
        logger.info("Starting grenade trajectory analysis...")

        # Initialize player stats
        self._init_player_stats()

        # Process grenade events (thrown and detonations)
        self._process_grenade_events(transformer)

        # Process blind events for flash effectiveness
        self._process_blind_events()

        # Calculate team stats
        team_stats = self._calculate_team_stats()

        # Calculate totals
        total_flashes = sum(p.flashbangs_thrown for p in self._player_stats.values())
        total_smokes = sum(p.smokes_thrown for p in self._player_stats.values())
        total_molotovs = sum(p.molotovs_thrown for p in self._player_stats.values())
        total_he = sum(p.he_grenades_thrown for p in self._player_stats.values())
        total_decoys = sum(p.decoys_thrown for p in self._player_stats.values())

        result = GrenadeTrajectoryAnalysis(
            player_stats=self._player_stats,
            all_positions=self._positions,
            team_stats=team_stats,
            total_grenades=total_flashes + total_smokes + total_molotovs + total_he + total_decoys,
            total_flashes=total_flashes,
            total_smokes=total_smokes,
            total_molotovs=total_molotovs,
            total_he=total_he,
        )

        logger.info(
            f"Grenade analysis complete: {result.total_grenades} grenades, "
            f"{len(self._positions)} positions tracked"
        )

        return result

    def _init_player_stats(self) -> None:
        """Initialize utility stats for each player."""
        for steam_id, name in self.data.player_names.items():
            team = self.data.player_teams.get(steam_id, "Unknown")
            self._player_stats[steam_id] = PlayerUtilityUsage(
                steam_id=steam_id,
                name=name,
                team=team,
            )

    def _process_grenade_events(self, transformer=None) -> None:
        """Process grenade thrown and detonation events."""
        grenades = self.data.grenades
        if not grenades:
            logger.info("No grenade events to process")
            return

        for grenade in grenades:
            thrower_id = grenade.player_steamid
            grenade_type = grenade.grenade_type.lower()

            # Update player counts based on event type
            if grenade.event_type == "thrown":
                if thrower_id in self._player_stats:
                    player = self._player_stats[thrower_id]
                    if "flash" in grenade_type:
                        player.flashbangs_thrown += 1
                    elif "smoke" in grenade_type:
                        player.smokes_thrown += 1
                    elif "molotov" in grenade_type or "inc" in grenade_type:
                        player.molotovs_thrown += 1
                    elif "hegrenade" in grenade_type or grenade_type == "hegrenade":
                        player.he_grenades_thrown += 1
                    elif "decoy" in grenade_type:
                        player.decoys_thrown += 1

            # Create position data for detonation events (where grenade landed)
            if grenade.event_type == "detonate" and grenade.x is not None and grenade.y is not None:
                radar_x = 0.0
                radar_y = 0.0

                if transformer:
                    try:
                        pos = transformer.game_to_radar(
                            grenade.x,
                            grenade.y,
                            grenade.z or 0
                        )
                        if pos.is_valid:
                            radar_x = pos.x
                            radar_y = pos.y
                    except Exception as e:
                        logger.debug(f"Error transforming grenade position: {e}")

                position = GrenadePosition(
                    x=radar_x,
                    y=radar_y,
                    game_x=grenade.x,
                    game_y=grenade.y,
                    game_z=grenade.z or 0,
                    grenade_type=grenade_type,
                    thrower_steamid=thrower_id,
                    thrower_name=self.data.player_names.get(thrower_id, "Unknown"),
                    thrower_team=self.data.player_teams.get(thrower_id, "Unknown"),
                    round_num=grenade.round_num,
                    tick=grenade.tick,
                    event_type=grenade.event_type,
                )

                self._positions.append(position)

                # Add to player's positions
                if thrower_id in self._player_stats:
                    self._player_stats[thrower_id].grenade_positions.append(position)

                # Track HE damage
                if "hegrenade" in grenade_type and thrower_id in self._player_stats:
                    self._player_stats[thrower_id].he_damage += grenade.damage_dealt
                    self._player_stats[thrower_id].he_enemies_hit += grenade.enemies_hit
                    self._player_stats[thrower_id].he_teammates_hit += grenade.teammates_hit

        logger.info(f"Processed {len(grenades)} grenade events, {len(self._positions)} positions")

    def _process_blind_events(self) -> None:
        """Process player blind events for flash effectiveness."""
        blinds = self.data.blinds
        if not blinds:
            logger.info("No blind events to process")
            return

        # Minimum blind duration to count (>1.1 seconds is meaningful)
        MIN_BLIND_DURATION = 1.1

        for blind in blinds:
            attacker_id = blind.attacker_steamid
            if attacker_id not in self._player_stats:
                continue

            player = self._player_stats[attacker_id]

            # Only count meaningful blinds
            if blind.blind_duration >= MIN_BLIND_DURATION:
                if blind.is_teammate:
                    player.teammates_flashed += 1
                else:
                    player.enemies_flashed += 1
                    player.total_blind_duration += blind.blind_duration

        logger.info(f"Processed {len(blinds)} blind events")

    def _calculate_team_stats(self) -> dict[str, dict]:
        """Calculate team-level utility statistics."""
        team_stats: dict[str, dict] = {
            "CT": {
                "total_utility": 0,
                "flashbangs": 0,
                "smokes": 0,
                "molotovs": 0,
                "he_grenades": 0,
                "enemies_flashed": 0,
                "avg_flash_efficiency": 0.0,
            },
            "T": {
                "total_utility": 0,
                "flashbangs": 0,
                "smokes": 0,
                "molotovs": 0,
                "he_grenades": 0,
                "enemies_flashed": 0,
                "avg_flash_efficiency": 0.0,
            },
        }

        for player in self._player_stats.values():
            team = player.team
            if team not in team_stats:
                continue

            team_stats[team]["total_utility"] += player.total_utility
            team_stats[team]["flashbangs"] += player.flashbangs_thrown
            team_stats[team]["smokes"] += player.smokes_thrown
            team_stats[team]["molotovs"] += player.molotovs_thrown
            team_stats[team]["he_grenades"] += player.he_grenades_thrown
            team_stats[team]["enemies_flashed"] += player.enemies_flashed

        # Calculate team-level flash efficiency
        for team in ["CT", "T"]:
            if team_stats[team]["flashbangs"] > 0:
                team_stats[team]["avg_flash_efficiency"] = round(
                    team_stats[team]["enemies_flashed"] / team_stats[team]["flashbangs"],
                    2
                )

        return team_stats


def analyze_grenade_trajectories(
    demo_data: DemoData, transformer=None
) -> GrenadeTrajectoryAnalysis:
    """
    Convenience function to analyze grenade trajectories from demo data.

    Args:
        demo_data: Parsed demo data to analyze.
        transformer: Optional CoordinateTransformer for radar positions.

    Returns:
        GrenadeTrajectoryAnalysis containing all utility metrics and positions.
    """
    analyzer = GrenadeTrajectoryAnalyzer(demo_data)
    return analyzer.analyze(transformer)
