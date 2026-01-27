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
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

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
        "radar_url": raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_dust2_radar.png"
        ),
        "z_cutoff": None,  # No vertical split
    },
    "de_mirage": {
        "name": "Mirage",
        "pos_x": -3230,
        "pos_y": 1713,
        "scale": 5.0,
        "radar_url":
        ),
        "z_cutoff": None,
    },
    "de_inferno": {
        "name": "Inferno",
        "pos_x": -2087,
        "pos_y": 3870,
        "scale": 4.9,
        "radar_url": (
            "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_inferno_radar.png"
        ),one,
    },
    "de_nuke": {
        "name": "Nuke",
        "pos_x": -3453,
        "pos_y": 2887,
        "scale": 7.0,
        "radar_url": (
            "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_nuke_radar.png"
        ),
        "z_cutoff": -
    "de_overpass": {
        "name": "Overpass",
        "pos_x": -4831,
        "pos_y": 1781,
        "scale": 5.2,
        "radar_url": (
            "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_overpass_radar.png"
        ),
        "z_cutoff": None,
    },
        "name": "Vertigo",
        "pos_x": -3168,
        "pos_y": 1762,
        "scale": 4.0,
        "radar_url": (
            "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_vertigo_radar.png"
        ),
        "z_cutoff": 11700,  # Split between floors
    },
    "de_ancient": {ent",
        "pos_x": -2953,
        "pos_y": 2164,
        "scale": 5.0,
        "radar_url": (
            "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_ancient_radar.png"
        ),
        "z_cutoff": None,
    },
    "de_anubis": {
        "name": "Anub6,
        "pos_y": 3328,
        "scale": 5.22,
        "radar_url": (
            "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_anubis_radar.png"
        ),
        "z_cutoff": None,
    },
    "cs_office": {
        "name": "Office",
        "pos_x": -183,
        "scale": 4.1,
        "radar_url": (
            "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/cs_office_radar.png"
        ),
        "z_cutoff": None,
    },
    "cs_italy": {
        "name": "Italy",
        "pos_x": -2647,
        "pos_y": 2592
        "radar_url": (
            "https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/cs_italy_radar.png"
        ),
        "z_cutoff": None,
    },
}

# Fallback radar URLs (alternative sources)
FALLBACK_RADAR_URLS = {
    "de_dust2": (githubusercontent.com/akiver/CSGO-Demos-Manager/master/Manager/resources/images/maps/radar/de_dust2.png"
    ),
    "de_mirage": (
        "https://raw.githubusercontent.com/akiver/CSGO-Demos-Manager/master/Manager/resources/images/maps/radar/de_mirage.png"
    ),
    "de_inferno": (
        "https://raw.githubusercontent.com/akiver/CSGO-Demos-Manager/master/Manager/resources/images/maps/radar/de_inferno.png"
    ),
}
# Coordinates are in game units (not radar pixels)
# Each zone: {"bounds": [[x1,y1], [x2,y2], ...], "type": "bombsite|route|spawn|mid|choke|position"}
# ============================================================================

MAP_ZONES: dict[str, dict[str, dict]] = {
    "de_dust2": {
        "T Spawn": {
            "bounds": [[-710, -1780], [540, -1780], [540, -2820], [-710, -2820]],
            "type": "spawn",
        },
        "CT Spawn": {
            "bounds": [[800, 2430], [1850, 2430], [1850, 3120], [800, 3120]],
            "type": "spawn",
        },
        "Long A": {
            "bounds": [[180, -510], [1380, -510], [1380, 280], [180, 280]],
            "type": "route",
        },
        "Long Doors": {
            "bounds": [[-100, -510], [180, -510], [180, -100], [-100, -100]],
            "type": "choke",
        },
        "A Site": {
            "bounds": [[670, 1510], [1480, 1510], [1480, 2430], [670, 2430]],
            "type": "bombsite",
        },
        "A Ramp": {
            "bounds": [[470, 280], [1050, 280], [1050, 1510], [470, 1510]],
            "type": "route",
        },
        "Short A": {
            "bounds": [[-1350, 1550], [-650, 1550], [-650, 2250], [-1350, 2250]],
            "type": "route",
        },
        "Catwalk": {
            "bounds": [[-650, 1550], [470, 1550], [470, 2250], [-650, 2250]],
            "type": "route",
        },
        "Mid": {
            "bounds": [[-1480, -20], [-400, -20], [-400, 1200], [-1480, 1200]],
            "type": "mid",
        },
        "Mid Doors": {
            "bounds": [[-1480, -580], [-400, -580], [-400, -20], [-1480, -20]],
            "type": "choke",
        },
        "Lower Tunnels": {
            "bounds": [[-2040, -1180], [-1480, -1180], [-1480, -580], [-2040, -580]],
            "type": "route",
        },
        "Upper Tunnels": {
            "bounds": [[-1900, 280], [-1500, 280], [-1500, 960], [-1900, 960]],
            "type": "route",
        },
        "B Site": {
            "bounds": [[-2350, 1200], [-1500, 1200], [-1500, 2480], [-2350, 2480]],
            "type": "bombsite",
        },
        "B Window": {
            "bounds": [[-1500, 1700], [-1150, 1700], [-1150, 2150], [-1500, 2150]],
            "type": "position",
        },
        "Pit": {
            "bounds": [[1480, 750], [1900, 750], [1900, 1510], [1480, 1510]],
            "type": "position",
        },
        "Goose": {
            "bounds": [[1480, 2430], [1900, 2430], [1900, 2850], [1480, 2850]],
            "type": "position",
        },
    },
    "de_mirage": {
        "T Spawn": {
            "bounds": [[-2900, -2200], [-2100, -2200], [-2100, -1500], [-2900, -1500]],
            "type": "spawn",
        },
        "CT Spawn": {
            "bounds": [[600, 1100], [1400, 1100], [1400, 1700], [600, 1700]],
            "type": "spawn",
        },
        "A Site": {
            "bounds": [[-400, 800], [600, 800], [600, 1600], [-400, 1600]],
            "type": "bombsite",
        },
        "A Ramp": {
            "bounds": [[-1200, 400], [-400, 400], [-400, 1000], [-1200, 1000]],
            "type": "route",
        },
        "Palace": {
            "bounds": [[-1800, 600], [-1200, 600], [-1200, 1200], [-1800, 1200]],
            "type": "route",
        },
        "Tetris": {
            "bounds": [[-400, 300], [200, 300], [200, 800], [-400, 800]],
            "type": "position",
        },
        "Jungle": {
            "bounds": [[200, 300], [800, 300], [800, 800], [200, 800]],
            "type": "position",
        },
        "Connector": {
            "bounds": [[200, -400], [800, -400], [800, 300], [200, 300]],
            "type": "route",
        },
        "Mid": {
            "bounds": [[-600, -1000], [200, -1000], [200, -200], [-600, -200]],
            "type": "mid",
        },
        "Top Mid": {
            "bounds": [[-1200, -1200], [-600, -1200], [-600, -600], [-1200, -600]],
            "type": "mid",
        },
        "Underpass": {
            "bounds": [[200, -1200], [800, -1200], [800, -600], [200, -600]],
            "type": "route",
        },
        "B Short": {
            "bounds": [[-1200, -600], [-600, -600], [-600, -200], [-1200, -200]],
            "type": "route",
        },
        "B Site": {
            "bounds": [[-2200, -800], [-1400, -800], [-1400, 0], [-2200, 0]],
            "type": "bombsite",
        },
        "B Apartments": {
            "bounds": [[-2800, -1200], [-2200, -1200], [-2200, -400], [-2800, -400]],
            "type": "route",
        },
        "Market": {
            "bounds": [[800, -200], [1400, -200], [1400, 400], [800, 400]],
            "type": "route",
        },
        "Window": {
            "bounds": [[600, 400], [1000, 400], [1000, 800], [600, 800]],
            "type": "position",
        },
    },
    "de_ancient": {
        "T Spawn": {
            "bounds": [[-1800, -600], [-1200, -600], [-1200, 0], [-1800, 0]],
            "type": "spawn",
        },
        "CT Spawn": {
            "bounds": [[700, 300], [1400, 300], [1400, 900], [700, 900]],
            "type": "spawn",
        },
        "A Site": {
            "bounds": [[-500, 750], [250, 750], [250, 1350], [-500, 1350]],
            "type": "bombsite",
        },
        "A Main": {
            "bounds": [[-1200, 250], [-500, 250], [-500, 750], [-1200, 750]],
            "type": "route",
        },
        "Donut": {
            "bounds": [[-500, -200], [200, -200], [200, 250], [-500, 250]],
            "type": "route",
        },
        "Mid": {
            "bounds": [[-200, -700], [600, -700], [600, -200], [-200, -200]],
            "type": "mid",
        },
        "B Site": {
            "bounds": [[900, -400], [1600, -400], [1600, 300], [900, 300]],
            "type": "bombsite",
        },
        "B Main": {
            "bounds": [[600, -1100], [1200, -1100], [1200, -400], [600, -400]],
            "type": "route",
        },
        "Cave": {
            "bounds": [[200, 750], [700, 750], [700, 1200], [200, 1200]],
            "type": "route",
        },
        "Temple": {
            "bounds": [[-1200, -200], [-500, -200], [-500, 250], [-1200, 250]],
            "type": "route",
        },
        "Ramp": {
            "bounds": [[200, -200], [600, -200], [600, 300], [200, 300]],
            "type": "route",
        },
        "Water": {
            "bounds": [[-200, -1100], [400, -1100], [400, -700], [-200, -700]],
            "type": "route",
        },
    },
    "de_inferno": {
        "T Spawn": {
            "bounds": [[-400, -1400], [400, -1400], [400, -800], [-400, -800]],
            "type": "spawn",
        },
        "CT Spawn": {
            "bounds": [[2000, 1400], [2700, 1400], [2700, 2000], [2000, 2000]],
            "type": "spawn",
        },
        "A Site": {
            "bounds": [[1700, 200], [2400, 200], [2400, 900], [1700, 900]],
            "type": "bombsite",
        },
        "A Long": {
            "bounds": [[1200, -200], [1800, -200], [1800, 400], [1200, 400]],
            "type": "route",
        },
        "Apartments": {
            "bounds": [[600, 0], [1200, 0], [1200, 600], [600, 600]],
            "type": "route",
        },
        "Pit": {
            "bounds": [[2400, 200], [2900, 200], [2900, 700], [2400, 700]],
            "type": "position",
        },
        "Library": {
            "bounds": [[1700, 900], [2200, 900], [2200, 1400], [1700, 1400]],
            "type": "position",
        },
        "Arch": {
            "bounds": [[1200, 900], [1700, 900], [1700, 1400], [1200, 1400]],
            "type": "route",
        },
        "Mid": {
            "bounds": [[400, -600], [1000, -600], [1000, 0], [400, 0]],
            "type": "mid",
        },
        "Top Mid": {
            "bounds": [[-200, -600], [400, -600], [400, 0], [-200, 0]],
            "type": "mid",
        },
        "Banana": {
            "bounds": [[-800, 600], [-200, 600], [-200, 1400], [-800, 1400]],
            "type": "route",
        },
        "B Site": {
            "bounds": [[-1200, 1800], [-400, 1800], [-400, 2600], [-1200, 2600]],
            "type": "bombsite",
        },
        "CT": {
            "bounds": [[1000, 1400], [1600, 1400], [1600, 2000], [1000, 2000]],
            "type": "route",
        },
        "Second Mid": {
            "bounds": [[400, 0], [1000, 0], [1000, 600], [400, 600]],
            "type": "mid",
        },
    },
    "de_anubis": {
        "T Spawn": {
            "bounds": [[-2400, -200], [-1600, -200], [-1600, 600], [-2400, 600]],
            "type": "spawn",
        },
        "CT Spawn": {
            "bounds": [[1000, 1200], [1800, 1200], [1800, 2000], [1000, 2000]],
            "type": "spawn",
        },
        "A Site": {
            "bounds": [[-400, 1400], [400, 1400], [400, 2200], [-400, 2200]],
            "type": "bombsite",
        },
        "A Main": {
            "bounds": [[-1200, 800], [-400, 800], [-400, 1400], [-1200, 1400]],
            "type": "route",
        },
        "A Long": {
            "bounds": [[-1800, 600], [-1200, 600], [-1200, 1200], [-1800, 1200]],
            "type": "route",
        },
        "Mid": {
            "bounds": [[-600, 0], [200, 0], [200, 800], [-600, 800]],
            "type": "mid",
        },
        "Connector": {
            "bounds": [[200, 400], [800, 400], [800, 1000], [200, 1000]],
            "type": "route",
        },
        "B Site": {
            "bounds": [[800, -400], [1600, -400], [1600, 400], [800, 400]],
            "type": "bombsite",
        },
        "B Main": {
            "bounds": [[-200, -800], [600, -800], [600, -200], [-200, -200]],
            "type": "route",
        },
        "Canal": {
            "bounds": [[600, -1200], [1200, -1200], [1200, -600], [600, -600]],
            "type": "route",
        },
        "Palace": {
            "bounds": [[400, 1000], [1000, 1000], [1000, 1600], [400, 1600]],
            "type": "route",
        },
        "Water": {
            "bounds": [[-1200, 0], [-600, 0], [-600, 600], [-1200, 600]],
            "type": "route",
        },
    },
    "de_nuke": {
        "T Spawn": {
            "bounds": [[-1800, -1600], [-1000, -1600], [-1000, -800], [-1800, -800]],
            "type": "spawn",
        },
        "CT Spawn": {
            "bounds": [[400, -400], [1200, -400], [1200, 400], [400, 400]],
            "type": "spawn",
        },
        "A Site": {
            "bounds": [[-800, 400], [200, 400], [200, 1200], [-800, 1200]],
            "type": "bombsite",
        },
        "Hut": {
            "bounds": [[-1200, 800], [-800, 800], [-800, 1200], [-1200, 1200]],
            "type": "position",
        },
        "Heaven": {
            "bounds": [[-400, 1200], [400, 1200], [400, 1800], [-400, 1800]],
            "type": "position",
        },
        "Outside": {
            "bounds": [[-2200, -400], [-1400, -400], [-1400, 400], [-2200, 400]],
            "type": "route",
        },
        "Secret": {
            "bounds": [[-2200, 400], [-1400, 400], [-1400, 1200], [-2200, 1200]],
            "type": "route",
        },
        "Ramp": {
            "bounds": [[-400, -400], [200, -400], [200, 400], [-400, 400]],
            "type": "route",
        },
        "B Site": {
            "bounds": [[-800, 400], [200, 400], [200, 1200], [-800, 1200]],
            "type": "bombsite",
        },
        "Lobby": {
            "bounds": [[-600, -800], [200, -800], [200, -200], [-600, -200]],
            "type": "route",
        },
        "Squeaky": {
            "bounds": [[-1000, 0], [-600, 0], [-600, 400], [-1000, 400]],
            "type": "choke",
        },
    },
    "de_vertigo": {
        "T Spawn": {
            "bounds": [[-2000, -800], [-1200, -800], [-1200, 0], [-2000, 0]],
            "type": "spawn",
        },
        "CT Spawn": {
            "bounds": [[200, 600], [1000, 600], [1000, 1400], [200, 1400]],
            "type": "spawn",
        },
        "A Site": {
            "bounds": [[-600, 800], [200, 800], [200, 1600], [-600, 1600]],
            "type": "bombsite",
        },
        "A Ramp": {
            "bounds": [[-1200, 400], [-600, 400], [-600, 1000], [-1200, 1000]],
            "type": "route",
        },
        "Mid": {
            "bounds": [[-800, -200], [0, -200], [0, 600], [-800, 600]],
            "type": "mid",
        },
        "B Site": {
            "bounds": [[-1800, 400], [-1200, 400], [-1200, 1200], [-1800, 1200]],
            "type": "bombsite",
        },
        "B Stairs": {
            "bounds": [[-1400, -200], [-800, -200], [-800, 400], [-1400, 400]],
            "type": "route",
        },
        "Generator": {
            "bounds": [[0, 0], [600, 0], [600, 600], [0, 600]],
            "type": "position",
        },
        "Elevator": {
            "bounds": [[-400, -600], [200, -600], [200, -200], [-400, -200]],
            "type": "route",
        },
        "Scaffolding": {
            "bounds": [[-600, 1200], [0, 1200], [0, 1800], [-600, 1800]],
            "type": "route",
        },
    },
    "de_overpass": {
        "T Spawn": {
            "bounds": [[-3800, -2600], [-3000, -2600], [-3000, -1800], [-3800, -1800]],
            "type": "spawn",
        },
        "CT Spawn": {
            "bounds": [[-1800, 400], [-1000, 400], [-1000, 1200], [-1800, 1200]],
            "type": "spawn",
        },
        "A Site": {
            "bounds": [[-2200, -400], [-1400, -400], [-1400, 400], [-2200, 400]],
            "type": "bombsite",
        },
        "A Long": {
            "bounds": [[-3000, -1200], [-2200, -1200], [-2200, -400], [-3000, -400]],
            "type": "route",
        },
        "Toilets": {
            "bounds": [[-1400, -800], [-800, -800], [-800, -200], [-1400, -200]],
            "type": "route",
        },
        "Party": {
            "bounds": [[-2600, 0], [-2000, 0], [-2000, 600], [-2600, 600]],
            "type": "position",
        },
        "Connector": {
            "bounds": [[-2800, -1800], [-2200, -1800], [-2200, -1200], [-2800, -1200]],
            "type": "route",
        },
        "B Site": {
            "bounds": [[-3400, -800], [-2600, -800], [-2600, 0], [-3400, 0]],
            "type": "bombsite",
        },
        "B Short": {
            "bounds": [[-2800, -1200], [-2200, -1200], [-2200, -600], [-2800, -600]],
            "type": "route",
        },
        "Monster": {
            "bounds": [[-3800, -1000], [-3200, -1000], [-3200, -400], [-3800, -400]],
            "type": "route",
        },
        "Water": {
            "bounds": [[-3000, 0], [-2400, 0], [-2400, 600], [-3000, 600]],
            "type": "route",
        },
        "Heaven": {
            "bounds": [[-1800, -200], [-1200, -200], [-1200, 400], [-1800, 400]],
            "type": "position",
        },
    },
}


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        x: Point X coordinate
        y: Point Y coordinate
        polygon: List of [x, y] vertices defining the polygon

    Returns:
        True if point is inside the polygon
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def get_zone_for_position(
    map_name: str, x: float, y: float, z: float | None = None
) -> str:
    """
    Determine which zone a position is in.

    Args:
        map_name: Internal map name (e.g., "de_dust2")
        x: Game X coordinate
        y: Game Y coordinate
        z: Game Z coordinate (for multi-level maps like Nuke/Vertigo)

    Returns:
        Zone name or "Unknown" if not in any defined zone
    """
    map_name = map_name.lower()
    if map_name not in MAP_ZONES:
        return "Unknown"

    zones = MAP_ZONES[map_name]

    for zone_name, zone_def in zones.items():
        bounds = zone_def.get("bounds", [])
        if len(bounds) < 3:
            continue

        if _point_in_polygon(x, y, bounds):
            return zone_name

    return "Unknown"


def classify_round_economy(equipment_value: int, is_pistol_round: bool) -> str:
    """
    Classify round type based on equipment value.

    Args:
        equipment_value: Total equipment value for the team/player
        is_pistol_round: Whether this is round 1 or 13

    Returns:
        Round type: 'pistol', 'eco', 'semi_eco', 'force', or 'full_buy'
    """
    if is_pistol_round:
        return "pistol"
    if equipment_value < 5000:
        return "eco"
    elif equipment_value < 10000:
        return "semi_eco"
    elif equipment_value < 20000:
        return "force"
    else:
        return "full_buy"


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
    z_cutoff: float | None = None

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
    bomb_x: float | None = None
    bomb_y: float | None = None
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
                        kill_positions.append(
                            {
                                "x": round(pos.x, 1),
                                "y": round(pos.y, 1),
                                "player": player_names.get(kill.attacker_steamid, "Unknown"),
                                "weapon": kill.weapon,
                                "headshot": kill.headshot,
                            }
                        )

                # Victim position (death)
                if kill.victim_x is not None and kill.victim_y is not None:
                    pos = self.transformer.game_to_radar(
                        kill.victim_x,
                        kill.victim_y,
                        kill.victim_z or 0,
                    )
                    if pos.is_valid:
                        death_positions.append(
                            {
                                "x": round(pos.x, 1),
                                "y": round(pos.y, 1),
                                "player": player_names.get(kill.victim_steamid, "Unknown"),
                            }
                        )

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
                    grenade_data.append(
                        {
                            "type": grenade.grenade_type,
                            "x": round(pos.x, 1),
                            "y": round(pos.y, 1),
                            "thrower": player_names.get(grenade.thrower_steamid, "Unknown"),
                            "round": grenade.round_num,
                            "tick": grenade.tick,
                        }
                    )

            except Exception as e:
                logger.debug(f"Error processing grenade: {e}")
                continue

        return grenade_data


class RadarImageManager:
    """
    Manages radar image downloading and caching.
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the radar image manager.

        Args:
            cache_dir: Directory to cache radar images
        """
        self.cache_dir = cache_dir or Path.home() / ".opensight" / "radar_images"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_radar_path(self, map_name: str) -> Path | None:
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

    def get_radar_url(self, map_name: str) -> str | None:
        """Get the URL for a map's radar image."""
        map_name = map_name.lower()
        if map_name in MAP_DATA:
            return MAP_DATA[map_name]["radar_url"]
        return None

    def list_available_maps(self) -> list[str]:
        """List all maps with available radar images."""
        return list(MAP_DATA.keys())

    def get_map_info(self, map_name: str) -> dict | None:
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


def get_map_metadata(map_name: str) -> MapMetadata | None:
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
