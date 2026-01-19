"""
Demo Source and Game Mode Detection

Automatically detects the source (Valve, FACEIT, ESEA, etc.) and game mode
(Competitive, Premier, Wingman, etc.) from demo file metadata.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

from opensight.constants import (
    DemoSource,
    GameMode,
    DemoType,
    CS2_TICK_RATE,
    TICK_RATES,
    SOURCE_IDENTIFIERS,
    FILENAME_PATTERNS,
    COMPETITIVE_MAPS,
    WINGMAN_MAPS,
)

logger = logging.getLogger(__name__)


@dataclass
class DemoMetadata:
    """Metadata about a parsed demo file."""
    source: DemoSource          # Detected platform
    game_mode: GameMode         # Detected game mode
    demo_type: DemoType         # GOTV or POV
    tick_rate: int              # Always 64 for CS2
    map_name: str               # e.g., "de_mirage"
    server_name: str            # Server identifier
    client_name: str            # Recording client
    duration_seconds: float     # Match length
    total_rounds: int           # Rounds played
    player_count: int           # Players detected
    is_complete: bool           # Did match complete normally?
    team1_score: int
    team2_score: int
    match_id: Optional[str] = None
    share_code: Optional[str] = None


def detect_source_from_server_name(server_name: str) -> DemoSource:
    """Detect demo source from server name patterns."""
    if not server_name:
        return DemoSource.UNKNOWN

    server_upper = server_name.upper()

    for source, identifiers in SOURCE_IDENTIFIERS.items():
        for identifier in identifiers:
            if identifier.upper() in server_upper:
                logger.debug(f"Detected source {source} from server name: {server_name}")
                return source

    # Check for Valve MM patterns
    if "valve" in server_name.lower() or server_name.startswith("="):
        return DemoSource.VALVE

    return DemoSource.UNKNOWN


def detect_source_from_filename(filename: str) -> DemoSource:
    """Detect demo source from filename patterns."""
    if not filename:
        return DemoSource.UNKNOWN

    filename_lower = filename.lower()

    for source, patterns in FILENAME_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                logger.debug(f"Detected source {source} from filename: {filename}")
                return source

    return DemoSource.UNKNOWN


def detect_game_mode(
    player_count: int,
    total_rounds: int,
    map_name: str,
    server_name: str = ""
) -> GameMode:
    """
    Detect game mode from match characteristics.

    Args:
        player_count: Number of unique players in the demo
        total_rounds: Total rounds played
        map_name: Map name (e.g., "de_mirage")
        server_name: Server name for additional hints
    """
    # Wingman: 2v2 (4 players), MR8 (16 rounds max)
    if player_count <= 4:
        if map_name in WINGMAN_MAPS or total_rounds <= 16:
            return GameMode.WINGMAN

    # Deathmatch: No real round structure, high player counts
    if total_rounds <= 1 and player_count > 10:
        return GameMode.DEATHMATCH

    # Arms Race: Many rounds (guns cycle), moderate players
    if total_rounds >= 30 and player_count <= 10:
        return GameMode.ARMS_RACE

    # Retakes: Very high round count
    if total_rounds >= 50:
        return GameMode.RETAKES

    # Casual: More than 10 players
    if player_count > 10:
        return GameMode.CASUAL

    # Check server name for Premier hints
    if server_name:
        server_lower = server_name.lower()
        if "premier" in server_lower:
            return GameMode.PREMIER
        if "scrimmage" in server_lower or "unranked" in server_lower:
            return GameMode.SCRIMMAGE

    # Default: Competitive 5v5
    if player_count >= 8 and player_count <= 12:
        if map_name in COMPETITIVE_MAPS:
            return GameMode.COMPETITIVE

    # Could be Premier if it's on Active Duty
    if total_rounds >= 12:
        return GameMode.COMPETITIVE

    return GameMode.UNKNOWN


def detect_demo_type(header: dict) -> DemoType:
    """Detect if demo is GOTV or POV recording."""
    # Check for POV indicators
    client_name = header.get("client_name", "")
    if client_name and "GOTV" not in client_name.upper():
        # If client name is a player name, it's likely POV
        if not any(x in client_name.upper() for x in ["GOTV", "TV", "HLTV"]):
            return DemoType.POV

    # Default to GOTV
    return DemoType.GOTV


def detect_demo_metadata(
    header: dict,
    player_count: int,
    total_rounds: int,
    duration_seconds: float,
    team_scores: tuple[int, int] = (0, 0),
    filename: str = ""
) -> DemoMetadata:
    """
    Detect all metadata about a demo file.

    Args:
        header: Parsed demo header dictionary
        player_count: Number of unique players
        total_rounds: Number of rounds
        duration_seconds: Match duration in seconds
        team_scores: Tuple of (team1_score, team2_score)
        filename: Original filename for pattern matching

    Returns:
        DemoMetadata with all detected information
    """
    map_name = header.get("map_name", "unknown")
    server_name = header.get("server_name", "")
    client_name = header.get("client_name", "")

    # Detect source (try server name first, then filename)
    source = detect_source_from_server_name(server_name)
    if source == DemoSource.UNKNOWN and filename:
        source = detect_source_from_filename(filename)

    # Detect game mode
    game_mode = detect_game_mode(
        player_count=player_count,
        total_rounds=total_rounds,
        map_name=map_name,
        server_name=server_name
    )

    # Detect demo type
    demo_type = detect_demo_type(header)

    # Get tick rate (always 64 for CS2)
    tick_rate = TICK_RATES.get(source, CS2_TICK_RATE)

    # Check if match completed normally
    is_complete = (
        total_rounds >= 12 and  # At least half a match
        (team_scores[0] >= 13 or team_scores[1] >= 13 or  # Someone won
         team_scores[0] == 12 and team_scores[1] == 12)   # Or went to OT
    )

    return DemoMetadata(
        source=source,
        game_mode=game_mode,
        demo_type=demo_type,
        tick_rate=tick_rate,
        map_name=map_name,
        server_name=server_name,
        client_name=client_name,
        duration_seconds=duration_seconds,
        total_rounds=total_rounds,
        player_count=player_count,
        is_complete=is_complete,
        team1_score=team_scores[0],
        team2_score=team_scores[1],
    )
