"""
Radar Map Module for CS2 Demo Visualization

Provides:
- CS2 map radar images (from official/community sources)
- Coordinate transformation from game units to radar pixels
- Position plotting on radar
- Grenade trajectory visualization

Map coordinate systems:
- CS2 uses Source 2 coordinates (X, Y, Z in game units)
- Radar images have their own coordinate systems defined by pos_x, pos_y, scale
- Transformation: pixel = (game_coord - pos) / scale
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import urllib.request
import os

logger = logging.getLogger(__name__)

# ============================================================================
# MAP METADATA - Coordinate transformation parameters
# These values are from CS2 map files (or derived from AWPY/other sources)
# pos_x, pos_y = coordinates of top-left corner of radar
# scale = units per pixel
# ============================================================================

MAP_DATA = {
    "de_dust2": {
        "name": "Dust II",
        "pos_x": -2476,
        "pos_y": 3239,
        "scale": 4.4,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_dust2_radar.png",
        "z_cutoff": None,  # No vertical split
    },
    "de_mirage": {
        "name": "Mirage",
        "pos_x": -3230,
        "pos_y": 1713,
        "scale": 5.0,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_mirage_radar.png",
        "z_cutoff": None,
    },
    "de_inferno": {
        "name": "Inferno",
        "pos_x": -2087,
        "pos_y": 3870,
        "scale": 4.9,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_inferno_radar.png",
        "z_cutoff": None,
    },
    "de_nuke": {
        "name": "Nuke",
        "pos_x": -3453,
        "pos_y": 2887,
        "scale": 7.0,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_nuke_radar.png",
        "z_cutoff": -495,  # Split between upper and lower
    },
    "de_overpass": {
        "name": "Overpass",
        "pos_x": -4831,
        "pos_y": 1781,
        "scale": 5.2,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_overpass_radar.png",
        "z_cutoff": None,
    },
    "de_vertigo": {
        "name": "Vertigo",
        "pos_x": -3168,
        "pos_y": 1762,
        "scale": 4.0,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_vertigo_radar.png",
        "z_cutoff": 11700,  # Split between floors
    },
    "de_ancient": {
        "name": "Ancient",
        "pos_x": -2953,
        "pos_y": 2164,
        "scale": 5.0,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_ancient_radar.png",
        "z_cutoff": None,
    },
    "de_anubis": {
        "name": "Anubis",
        "pos_x": -2796,
        "pos_y": 3328,
        "scale": 5.22,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_anubis_radar.png",
        "z_cutoff": None,
    },
    "cs_office": {
        "name": "Office",
        "pos_x": -1838,
        "pos_y": 1858,
        "scale": 4.1,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/cs_office_radar.png",
        "z_cutoff": None,
    },
    "cs_italy": {
        "name": "Italy",
        "pos_x": -2647,
        "pos_y": 2592,
        "scale": 4.6,
        "radar_url": "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/cs_italy_radar.png",
        "z_cutoff": None,
    },
}

# Fallback radar URLs (alternative sources)
FALLBACK_RADAR_URLS = {
    "de_dust2": "https://raw.githubusercontent.com/akiver/CSGO-Demos-Manager/master/Manager/resources/images/maps/radar/de_dust2.png",
    "de_mirage": "https://raw.githubusercontent.com/akiver/CSGO-Demos-Manager/master/Manager/resources/images/maps/radar/de_mirage.png",
    "de_inferno": "https://raw.githubusercontent.com/akiver/CSGO-Demos-Manager/master/Manager/resources/images/maps/radar/de_inferno.png",
}


@dataclass
class RadarPosition:
    """A position on the radar in pixel coordinates."""
    x: float  # Pixel X (0 = left)
    y: float  # Pixel Y (0 = top)
    z: float = 0.0  # Original Z coordinate (for level detection)
    is_valid: bool = True

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class MapMetadata:
    """Metadata for a CS2 map."""
    name: str
    internal_name: str
    pos_x: float
    pos_y: float
    scale: float
    radar_url: str
    z_cutoff: Optional[float] = None

    # Radar image dimensions (set after loading)
    radar_width: int = 1024
    radar_height: int = 1024


class CoordinateTransformer:
    """
    Transforms CS2 game coordinates to radar pixel coordinates.

    CS2 coordinate system:
    - X increases going East
    - Y increases going North
    - Z increases going Up

    Radar coordinate system:
    - X increases going right (pixel columns)
    - Y increases going down (pixel rows)
    """

    def __init__(self, map_name: str):
        """
        Initialize transformer for a specific map.

        Args:
            map_name: Internal map name (e.g., "de_dust2")
        """
        self.map_name = map_name.lower()

        if self.map_name not in MAP_DATA:
            logger.warning(f"Unknown map: {map_name}, using default transform")
            self.metadata = MapMetadata(
                name=map_name,
                internal_name=map_name,
                pos_x=-3000,
                pos_y=3000,
                scale=5.0,
                radar_url="",
            )
        else:
            data = MAP_DATA[self.map_name]
            self.metadata = MapMetadata(
                name=data["name"],
                internal_name=self.map_name,
                pos_x=data["pos_x"],
                pos_y=data["pos_y"],
                scale=data["scale"],
                radar_url=data["radar_url"],
                z_cutoff=data.get("z_cutoff"),
            )

    def game_to_radar(self, x: float, y: float, z: float = 0.0) -> RadarPosition:
        """
        Transform game coordinates to radar pixel coordinates.

        Args:
            x: Game X coordinate
            y: Game Y coordinate
            z: Game Z coordinate (for level detection)

        Returns:
            RadarPosition in pixel coordinates
        """
        try:
            # Transform using map parameters
            # pixel_x = (game_x - pos_x) / scale
            # pixel_y = (pos_y - game_y) / scale (Y is inverted)
            pixel_x = (x - self.metadata.pos_x) / self.metadata.scale
            pixel_y = (self.metadata.pos_y - y) / self.metadata.scale

            # Clamp to radar bounds
            pixel_x = max(0, min(pixel_x, self.metadata.radar_width - 1))
            pixel_y = max(0, min(pixel_y, self.metadata.radar_height - 1))

            return RadarPosition(x=pixel_x, y=pixel_y, z=z, is_valid=True)

        except Exception as e:
            logger.debug(f"Coordinate transform failed: {e}")
            return RadarPosition(x=0, y=0, z=z, is_valid=False)

    def radar_to_game(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """
        Transform radar pixel coordinates back to game coordinates.

        Args:
            pixel_x: Radar pixel X
            pixel_y: Radar pixel Y

        Returns:
            Tuple of (game_x, game_y)
        """
        game_x = (pixel_x * self.metadata.scale) + self.metadata.pos_x
        game_y = self.metadata.pos_y - (pixel_y * self.metadata.scale)
        return (game_x, game_y)

    def is_upper_level(self, z: float) -> bool:
        """Check if Z coordinate is on upper level (for multi-level maps)."""
        if self.metadata.z_cutoff is None:
            return True  # Single level map
        return z >= self.metadata.z_cutoff


@dataclass
class PlayerRadarPosition:
    """Player position for radar display."""
    steam_id: int
    name: str
    team: str  # "CT" or "T"
    x: float  # Radar pixel X
    y: float  # Radar pixel Y
    yaw: float  # View angle (0-360)
    health: int
    is_alive: bool


@dataclass
class GrenadeRadarPosition:
    """Grenade position for radar display."""
    grenade_type: str  # "flashbang", "smoke", "he", "molotov", "decoy"
    x: float
    y: float
    thrower_steam_id: int
    is_detonated: bool = False


@dataclass
class RadarFrame:
    """A single frame of radar data."""
    tick: int
    round_num: int
    players: list[PlayerRadarPosition]
    grenades: list[GrenadeRadarPosition] = field(default_factory=list)
    bomb_x: Optional[float] = None
    bomb_y: Optional[float] = None
    bomb_planted: bool = False


class RadarDataGenerator:
    """
    Generates radar visualization data from demo data.

    Converts player positions and grenade positions to radar coordinates
    for visualization.
    """

    def __init__(self, map_name: str):
        """
        Initialize the radar data generator.

        Args:
            map_name: Internal map name
        """
        self.transformer = CoordinateTransformer(map_name)
        self.map_name = map_name

    def generate_kill_heatmap_data(
        self,
        kills: list,
        player_names: dict[int, str],
    ) -> dict:
        """
        Generate heatmap data for kills on the radar.

        Args:
            kills: List of KillEvent objects
            player_names: Dict mapping steam_id to name

        Returns:
            Dict with radar coordinates for kills and deaths
        """
        kill_positions = []
        death_positions = []

        for kill in kills:
            try:
                # Attacker position (kill)
                if kill.attacker_x is not None and kill.attacker_y is not None:
                    pos = self.transformer.game_to_radar(
                        kill.attacker_x,
                        kill.attacker_y,
                        kill.attacker_z or 0,
                    )
                    if pos.is_valid:
                        kill_positions.append({
                            "x": round(pos.x, 1),
                            "y": round(pos.y, 1),
                            "player": player_names.get(kill.attacker_steamid, "Unknown"),
                            "weapon": kill.weapon,
                            "headshot": kill.headshot,
                        })

                # Victim position (death)
                if kill.victim_x is not None and kill.victim_y is not None:
                    pos = self.transformer.game_to_radar(
                        kill.victim_x,
                        kill.victim_y,
                        kill.victim_z or 0,
                    )
                    if pos.is_valid:
                        death_positions.append({
                            "x": round(pos.x, 1),
                            "y": round(pos.y, 1),
                            "player": player_names.get(kill.victim_steamid, "Unknown"),
                        })

            except Exception as e:
                logger.debug(f"Error processing kill for heatmap: {e}")
                continue

        return {
            "map_name": self.map_name,
            "radar_width": self.transformer.metadata.radar_width,
            "radar_height": self.transformer.metadata.radar_height,
            "kill_positions": kill_positions,
            "death_positions": death_positions,
        }

    def generate_grenade_data(
        self,
        grenades: list,
        player_names: dict[int, str],
    ) -> list[dict]:
        """
        Generate grenade position data for radar visualization.

        Args:
            grenades: List of GrenadeEvent objects
            player_names: Dict mapping steam_id to name

        Returns:
            List of grenade data dicts
        """
        grenade_data = []

        for grenade in grenades:
            try:
                if grenade.x is None or grenade.y is None:
                    continue

                pos = self.transformer.game_to_radar(
                    grenade.x,
                    grenade.y,
                    grenade.z or 0,
                )

                if pos.is_valid:
                    grenade_data.append({
                        "type": grenade.grenade_type,
                        "x": round(pos.x, 1),
                        "y": round(pos.y, 1),
                        "thrower": player_names.get(grenade.thrower_steamid, "Unknown"),
                        "round": grenade.round_num,
                        "tick": grenade.tick,
                    })

            except Exception as e:
                logger.debug(f"Error processing grenade: {e}")
                continue

        return grenade_data


class RadarImageManager:
    """
    Manages radar image downloading and caching.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the radar image manager.

        Args:
            cache_dir: Directory to cache radar images
        """
        self.cache_dir = cache_dir or Path.home() / ".opensight" / "radar_images"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_radar_path(self, map_name: str) -> Optional[Path]:
        """
        Get path to radar image, downloading if needed.

        Args:
            map_name: Internal map name

        Returns:
            Path to radar image, or None if unavailable
        """
        map_name = map_name.lower()
        cached_path = self.cache_dir / f"{map_name}_radar.png"

        # Return cached if exists
        if cached_path.exists():
            return cached_path

        # Try to download
        if map_name not in MAP_DATA:
            logger.warning(f"No radar URL for map: {map_name}")
            return None

        url = MAP_DATA[map_name]["radar_url"]

        try:
            logger.info(f"Downloading radar image for {map_name} from {url}")
            urllib.request.urlretrieve(url, cached_path)
            logger.info(f"Saved radar image to {cached_path}")
            return cached_path

        except Exception as e:
            logger.warning(f"Failed to download radar for {map_name}: {e}")

            # Try fallback URL
            if map_name in FALLBACK_RADAR_URLS:
                try:
                    fallback_url = FALLBACK_RADAR_URLS[map_name]
                    logger.info(f"Trying fallback URL: {fallback_url}")
                    urllib.request.urlretrieve(fallback_url, cached_path)
                    return cached_path
                except Exception as e2:
                    logger.error(f"Fallback download also failed: {e2}")

            return None

    def get_radar_url(self, map_name: str) -> Optional[str]:
        """Get the URL for a map's radar image."""
        map_name = map_name.lower()
        if map_name in MAP_DATA:
            return MAP_DATA[map_name]["radar_url"]
        return None

    def list_available_maps(self) -> list[str]:
        """List all maps with available radar images."""
        return list(MAP_DATA.keys())

    def get_map_info(self, map_name: str) -> Optional[dict]:
        """Get metadata for a map."""
        map_name = map_name.lower()
        if map_name in MAP_DATA:
            return MAP_DATA[map_name].copy()
        return None


# Convenience functions

def get_radar_positions(
    kills: list,
    map_name: str,
    player_names: dict[int, str],
) -> dict:
    """
    Convert kill positions to radar coordinates.

    Args:
        kills: List of KillEvent objects
        map_name: Map name
        player_names: Player name mapping

    Returns:
        Dict with radar coordinate data
    """
    generator = RadarDataGenerator(map_name)
    return generator.generate_kill_heatmap_data(kills, player_names)


def get_map_metadata(map_name: str) -> Optional[MapMetadata]:
    """Get metadata for a map."""
    map_name = map_name.lower()
    if map_name not in MAP_DATA:
        return None

    data = MAP_DATA[map_name]
    return MapMetadata(
        name=data["name"],
        internal_name=map_name,
        pos_x=data["pos_x"],
        pos_y=data["pos_y"],
        scale=data["scale"],
        radar_url=data["radar_url"],
        z_cutoff=data.get("z_cutoff"),
    )
