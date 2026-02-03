"""
Map Zone Definitions for CS2 Competitive Maps.

Defines coordinate boundaries for common callouts on each map.
Used to translate X/Y coordinates into human-readable callout names.

Coordinate system: CS2 uses game units where:
- X increases going east
- Y increases going north
- Z is vertical (height)
"""

from dataclasses import dataclass


@dataclass
class MapZone:
    """A named zone on a map with coordinate boundaries."""

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float = -10000  # Default to any height
    z_max: float = 10000


# =============================================================================
# MIRAGE
# =============================================================================

MIRAGE_ZONES = [
    # A Site and surroundings
    MapZone("A Site", -250, 500, -2200, -1500),
    MapZone("A Ramp", -400, 200, -1500, -900),
    MapZone("Palace", -900, -200, -2200, -1600),
    MapZone("Jungle", 200, 700, -1300, -800),
    MapZone("Connector", 700, 1400, -1200, -500),
    MapZone("Stairs", 300, 700, -2100, -1700),
    MapZone("Firebox", -200, 200, -1900, -1600),
    MapZone("Sandwich", 200, 500, -1900, -1600),
    MapZone("CT", 500, 1000, -1900, -1400),
    # Mid
    MapZone("Mid", -600, 200, -100, 800),
    MapZone("Top Mid", -500, 100, 500, 1000),
    MapZone("Window", 0, 400, 200, 700),
    MapZone("Short", 100, 600, 700, 1200),
    MapZone("Underpass", -200, 300, -100, 400),
    MapZone("Ladder", 400, 700, 0, 400),
    # B Site and surroundings
    MapZone("B Site", -2100, -1400, -500, 200),
    MapZone("B Apartments", -2500, -1800, -1200, -400),
    MapZone("B Short", -1800, -1200, -400, 200),
    MapZone("Van", -1900, -1500, 0, 400),
    MapZone("Bench", -1700, -1400, -200, 200),
    MapZone("Market", -1800, -1200, 200, 700),
    MapZone("Market Window", -1400, -1000, 200, 600),
    # Spawns
    MapZone("T Spawn", -1500, -500, 1000, 1800),
    MapZone("CT Spawn", 800, 1500, -1000, -200),
]

# =============================================================================
# INFERNO
# =============================================================================

INFERNO_ZONES = [
    # A Site
    MapZone("A Site", 1800, 2500, 400, 1100),
    MapZone("Pit", 1500, 2000, 1100, 1700),
    MapZone("Graveyard", 2000, 2500, 1100, 1600),
    MapZone("Library", 2200, 2700, 600, 1100),
    MapZone("Arch", 1600, 2100, 100, 600),
    MapZone("Boiler", 2000, 2400, -100, 400),
    MapZone("Balcony", 1800, 2300, 800, 1200, z_min=200),
    # Mid and Apartments
    MapZone("Mid", 500, 1200, -500, 200),
    MapZone("Second Mid", 1000, 1600, -200, 400),
    MapZone("Apartments", 100, 700, 200, 900),
    MapZone("Bedroom", 200, 600, 600, 1000),
    MapZone("Porch", 500, 900, 100, 500),
    # B Site and Banana
    MapZone("B Site", -400, 400, -1800, -1200),
    MapZone("Banana", -200, 600, -800, -200),
    MapZone("Top Banana", -100, 500, -500, 100),
    MapZone("Bottom Banana", -100, 500, -1000, -500),
    MapZone("Car", 100, 500, -1600, -1200),
    MapZone("Coffins", -300, 100, -1700, -1300),
    MapZone("CT", 0, 600, -2000, -1600),
    MapZone("Construction", -600, 0, -1500, -1000),
    MapZone("Fountain", -200, 400, -1400, -1000),
    MapZone("New Box", -100, 300, -1500, -1200),
    # Spawns
    MapZone("T Spawn", -1200, -400, 600, 1200),
    MapZone("CT Spawn", 500, 1200, -2200, -1600),
    # Other
    MapZone("Alley", -500, 100, -200, 400),
    MapZone("Long", 1200, 1800, -200, 600),
]

# =============================================================================
# DUST2
# =============================================================================

DUST2_ZONES = [
    # A Site
    MapZone("A Site", 500, 1400, 2000, 2800),
    MapZone("A Long", -200, 800, 500, 1500),
    MapZone("A Short", 200, 900, 1500, 2100),
    MapZone("Pit", -300, 300, 2400, 3000),
    MapZone("Car", 800, 1200, 2400, 2800),
    MapZone("Goose", 1100, 1500, 2200, 2600),
    MapZone("Elevator", 600, 1000, 2200, 2500),
    MapZone("A Ramp", 400, 900, 1800, 2200),
    # Mid
    MapZone("Mid", -800, 200, 0, 800),
    MapZone("Top Mid", -600, 100, 600, 1200),
    MapZone("Mid Doors", -400, 200, -200, 400),
    MapZone("CT Mid", 0, 600, 800, 1400),
    MapZone("Catwalk", 100, 600, 1200, 1800),
    MapZone("Xbox", -200, 300, 200, 600),
    # B Site
    MapZone("B Site", -1800, -1000, 1800, 2600),
    MapZone("B Tunnels", -2000, -1200, 0, 1000),
    MapZone("Upper Tunnels", -1800, -1200, 800, 1400),
    MapZone("Lower Tunnels", -2000, -1400, -200, 600),
    MapZone("B Platform", -1400, -800, 2000, 2500),
    MapZone("B Window", -1200, -800, 2200, 2600),
    MapZone("B Back", -1500, -1000, 2400, 2900),
    MapZone("B Closet", -1200, -800, 2000, 2300),
    MapZone("B Car", -1700, -1300, 2000, 2400),
    # Spawns
    MapZone("T Spawn", -800, 0, -1200, -400),
    MapZone("CT Spawn", 200, 1000, 1600, 2200),
    # Other
    MapZone("Outside Long", -600, 200, -400, 300),
    MapZone("Blue", -300, 300, 0, 500),
]

# =============================================================================
# ANCIENT
# =============================================================================

ANCIENT_ZONES = [
    # A Site
    MapZone("A Site", -800, 0, -1800, -1000),
    MapZone("A Main", -1500, -600, -1000, -200),
    MapZone("Donut", -400, 200, -1200, -700),
    MapZone("A Elbow", -1200, -600, -700, -100),
    MapZone("Temple", -600, 200, -600, 0),
    MapZone("Red", -200, 400, -1500, -1100),
    # Mid
    MapZone("Mid", -600, 200, 0, 800),
    MapZone("Cave", -1200, -400, 200, 800),
    MapZone("House", -400, 200, 400, 1000),
    # B Site
    MapZone("B Site", 400, 1200, 200, 1000),
    MapZone("B Main", 0, 600, -600, 200),
    MapZone("B Ramp", 200, 800, -200, 400),
    MapZone("Alley", 600, 1200, -400, 200),
    # Spawns
    MapZone("T Spawn", -2000, -1200, -400, 400),
    MapZone("CT Spawn", 600, 1400, 800, 1600),
]

# =============================================================================
# ANUBIS
# =============================================================================

ANUBIS_ZONES = [
    # A Site
    MapZone("A Site", -400, 400, 1200, 2000),
    MapZone("A Main", -1200, -400, 600, 1400),
    MapZone("A Connector", -400, 200, 600, 1200),
    MapZone("Palace", -800, 0, 1600, 2200),
    # Mid
    MapZone("Mid", -600, 400, -200, 600),
    MapZone("Canal", -200, 600, 0, 800),
    # B Site
    MapZone("B Site", 800, 1600, 0, 800),
    MapZone("B Main", 600, 1400, -800, 0),
    MapZone("B Heaven", 1000, 1500, 400, 1000, z_min=100),
    # Spawns
    MapZone("T Spawn", -1600, -800, -800, 0),
    MapZone("CT Spawn", 400, 1200, 1400, 2200),
]

# =============================================================================
# NUKE
# =============================================================================

NUKE_ZONES = [
    # A Site (Upper)
    MapZone("A Site", -600, 600, -1000, 200),
    MapZone("Hut", -200, 400, -1200, -600),
    MapZone("Heaven", 0, 600, -600, 200, z_min=0),
    MapZone("A Main", -1200, -400, -800, 0),
    MapZone("Squeaky", -1000, -400, -400, 200),
    # B Site (Lower)
    MapZone("B Site", -600, 600, -1000, 200, z_max=-200),
    MapZone("Ramp", 200, 1000, -400, 400),
    MapZone("Secret", 600, 1200, 0, 600),
    MapZone("Vents", -400, 200, 200, 800),
    # Outside
    MapZone("Outside", 800, 2000, -1600, -400),
    MapZone("Silo", 1200, 1800, -1400, -800),
    MapZone("Yard", 600, 1400, -2000, -1400),
    # Spawns
    MapZone("T Spawn", 1400, 2200, -2400, -1600),
    MapZone("CT Spawn", -800, 0, 400, 1200),
]

# =============================================================================
# VERTIGO
# =============================================================================

VERTIGO_ZONES = [
    # A Site
    MapZone("A Site", -1000, -200, 1000, 1800),
    MapZone("A Ramp", -1400, -800, 600, 1200),
    MapZone("Elevator", -600, 0, 800, 1400),
    MapZone("Headshot", -400, 200, 1400, 1800),
    # Mid
    MapZone("Mid", -600, 200, 0, 800),
    # B Site
    MapZone("B Site", 400, 1200, 600, 1400),
    MapZone("B Stairs", 200, 800, 200, 800),
    MapZone("CT", 600, 1200, 1200, 1800),
    # Spawns
    MapZone("T Spawn", -1400, -600, -800, 0),
    MapZone("CT Spawn", 400, 1200, 1600, 2400),
]

# =============================================================================
# OVERPASS
# =============================================================================

OVERPASS_ZONES = [
    # A Site
    MapZone("A Site", -1800, -1000, 0, 800),
    MapZone("A Long", -2400, -1600, -600, 200),
    MapZone("Toilets", -2200, -1600, -200, 400),
    MapZone("Bank", -1600, -1000, 400, 1000),
    MapZone("Truck", -1200, -600, 200, 800),
    # B Site
    MapZone("B Site", 400, 1200, -800, 0),
    MapZone("B Short", 0, 600, 0, 600),
    MapZone("Monster", -400, 400, -400, 200),
    MapZone("Connector", -200, 600, 400, 1000),
    MapZone("Water", 200, 1000, -1400, -600),
    MapZone("Heaven", 600, 1200, -600, 0, z_min=100),
    # Mid
    MapZone("Playground", -1200, -400, -400, 400),
    # Spawns
    MapZone("T Spawn", -2800, -2000, -1200, -400),
    MapZone("CT Spawn", 0, 800, 800, 1600),
]

# =============================================================================
# TRAIN
# =============================================================================

TRAIN_ZONES = [
    # A Site
    MapZone("A Site", 200, 1000, 600, 1400),
    MapZone("A Main", -400, 400, -200, 600),
    MapZone("Ivy", -800, 0, -600, 200),
    MapZone("Pop Dog", 400, 800, 200, 600),
    # B Site
    MapZone("B Site", 200, 1000, -1200, -400),
    MapZone("B Ramp", -400, 400, -800, -200),
    MapZone("Upper B", 400, 1000, -800, -200),
    MapZone("Lower B", -200, 600, -1400, -800),
    # Mid/Connector
    MapZone("Connector", 800, 1400, -200, 600),
    MapZone("Z Connector", 1000, 1600, 200, 800),
    # Spawns
    MapZone("T Spawn", -1400, -600, -600, 200),
    MapZone("CT Spawn", 1200, 2000, -200, 600),
]

# =============================================================================
# MAP REGISTRY
# =============================================================================

MAP_ZONES = {
    "de_mirage": MIRAGE_ZONES,
    "de_inferno": INFERNO_ZONES,
    "de_dust2": DUST2_ZONES,
    "de_ancient": ANCIENT_ZONES,
    "de_anubis": ANUBIS_ZONES,
    "de_nuke": NUKE_ZONES,
    "de_vertigo": VERTIGO_ZONES,
    "de_overpass": OVERPASS_ZONES,
    "de_train": TRAIN_ZONES,
}


def get_callout(map_name: str, x: float, y: float, z: float = 0) -> str:
    """
    Get the callout name for a position on a map.

    Args:
        map_name: Map name (e.g., "de_mirage")
        x: X coordinate
        y: Y coordinate
        z: Z coordinate (optional, for height-based zones)

    Returns:
        Callout name or "Unknown" if no matching zone
    """
    # Normalize map name
    map_key = map_name.lower()
    if not map_key.startswith("de_"):
        map_key = f"de_{map_key}"

    zones = MAP_ZONES.get(map_key, [])
    if not zones:
        return "Unknown"

    for zone in zones:
        if (
            zone.x_min <= x <= zone.x_max
            and zone.y_min <= y <= zone.y_max
            and zone.z_min <= z <= zone.z_max
        ):
            return zone.name

    return "Unknown"


def get_side_for_position(map_name: str, x: float, y: float) -> str:
    """
    Determine which side of the map a position is on.

    Returns "T side", "CT side", or "Mid" based on position.
    """
    callout = get_callout(map_name, x, y)

    # Simple heuristics based on common callout patterns
    t_side_keywords = ["T Spawn", "Apartments", "Long", "Tunnels", "Main", "Banana"]
    ct_side_keywords = ["CT", "Spawn", "Site", "Heaven", "Library"]
    mid_keywords = ["Mid", "Connector", "Short"]

    callout_lower = callout.lower()

    for keyword in mid_keywords:
        if keyword.lower() in callout_lower:
            return "Mid"

    for keyword in t_side_keywords:
        if keyword.lower() in callout_lower:
            return "T side"

    for keyword in ct_side_keywords:
        if keyword.lower() in callout_lower:
            return "CT side"

    return "Unknown"
