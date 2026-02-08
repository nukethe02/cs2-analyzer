"""CS2 map metadata for coordinate transformation.

To convert game coordinates to radar pixel coordinates:
    pixel_x = (game_x - pos_x) / scale
    pixel_y = (pos_y - game_y) / scale  # Y is inverted

Radar images are 1024x1024 pixels.
"""

MAP_METADATA = {
    "de_ancient": {"pos_x": -2953, "pos_y": 2164, "scale": 5.0},
    "de_mirage": {"pos_x": -3230, "pos_y": 1713, "scale": 5.0},
    "de_inferno": {"pos_x": -2087, "pos_y": 3870, "scale": 4.9},
    "de_dust2": {"pos_x": -2476, "pos_y": 3239, "scale": 4.4},
    "de_anubis": {"pos_x": -2796, "pos_y": 3328, "scale": 5.22},
    "de_nuke": {"pos_x": -3453, "pos_y": 2887, "scale": 7.0},
    "de_overpass": {"pos_x": -4831, "pos_y": 1781, "scale": 5.2},
    "de_vertigo": {"pos_x": -3168, "pos_y": 1762, "scale": 4.0},
    "de_train": {"pos_x": -2477, "pos_y": 2392, "scale": 4.7},
}

# Image dimensions (all CS2 radar images are 1024x1024)
RADAR_IMAGE_SIZE = 1024


def get_map_metadata(map_name: str) -> dict | None:
    """Get coordinate transformation metadata for a map.

    Args:
        map_name: Map name with or without 'de_' prefix

    Returns:
        Dict with pos_x, pos_y, scale, or None if unknown map
    """
    # Normalize map name
    if not map_name.startswith("de_"):
        map_name = f"de_{map_name}"
    map_name = map_name.lower().strip()

    return MAP_METADATA.get(map_name)


def game_to_pixel(
    game_x: float, game_y: float, map_name: str
) -> tuple[float, float] | None:
    """Convert game world coordinates to radar pixel coordinates.

    Args:
        game_x: X position in game world units
        game_y: Y position in game world units
        map_name: CS2 map name

    Returns:
        Tuple of (pixel_x, pixel_y) or None if unknown map
    """
    meta = get_map_metadata(map_name)
    if meta is None:
        return None

    pixel_x = (game_x - meta["pos_x"]) / meta["scale"]
    pixel_y = (meta["pos_y"] - game_y) / meta["scale"]  # Y inverted

    return (pixel_x, pixel_y)
