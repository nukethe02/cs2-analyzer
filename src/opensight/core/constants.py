"""
OpenSight CS2 Demo Analyzer - Constants

Defines demo sources, game modes, and other constants for universal demo analysis.
Based on industry standards from CS Demo Manager and professional parsing tools.
"""

from enum import Enum, StrEnum


class DemoSource(StrEnum):
    """
    Demo source/platform detection.

    Based on CS Demo Manager's supported sources:
    https://github.com/akiver/cs-demo-analyzer
    """

    UNKNOWN = "unknown"
    VALVE = "valve"  # Valve Matchmaking (Premier, Competitive, Wingman, etc.)
    FACEIT = "faceit"  # FACEIT platform
    ESEA = "esea"  # ESEA league/pugs
    ESL = "esl"  # ESL tournaments
    ESPORTAL = "esportal"  # Esportal
    CHALLENGERMODE = "challengermode"
    POPFLASH = "popflash"  # PopFlash
    FASTCUP = "fastcup"  # FastCup
    FIVEPLAY = "5eplay"  # 5EPlay (China)
    PERFECTWORLD = "perfectworld"  # Perfect World (China)
    EBOT = "ebot"  # eBot match servers
    CEVO = "cevo"  # CEVO
    HLTV = "hltv"  # HLTV professional demos
    CUSTOM = "custom"  # Custom/private servers
    POV = "pov"  # POV recording (not GOTV)


class GameMode(StrEnum):
    """
    CS2 game modes.

    Different modes have different round structures and player counts.
    """

    UNKNOWN = "unknown"
    COMPETITIVE = "competitive"  # 5v5, MR12 (24 rounds max + OT)
    PREMIER = "premier"  # 5v5, MR12 with pick/ban
    WINGMAN = "wingman"  # 2v2, MR8 (16 rounds max)
    CASUAL = "casual"  # Up to 10v10, relaxed rules
    DEATHMATCH = "deathmatch"  # FFA/Team DM, no rounds
    ARMS_RACE = "arms_race"  # Gun game
    DANGER_ZONE = "danger_zone"  # Battle royale
    RETAKES = "retakes"  # Retake practice
    SCRIMMAGE = "scrimmage"  # Unranked competitive
    CUSTOM = "custom"  # Custom server modes


class DemoType(StrEnum):
    """
    Demo recording type.

    GOTV demos contain full match data from all perspectives.
    POV demos contain only one player's perspective.
    """

    GOTV = "gotv"  # Full match recording (SourceTV)
    POV = "pov"  # Single player perspective


class Team(int, Enum):
    """CS2 team numbers."""

    UNASSIGNED = 0
    SPECTATOR = 1
    TERRORIST = 2
    CT = 3


class RoundEndReason(int, Enum):
    """
    Round end reasons from CS2.

    These are the official game event values.
    """

    TARGET_BOMBED = 1  # Terrorists bombed the target
    VIP_ESCAPED = 2  # (Legacy)
    VIP_KILLED = 3  # (Legacy)
    TERRORISTS_ESCAPED = 4  # (Legacy)
    CT_STOPPED_ESCAPE = 5  # (Legacy)
    TERRORIST_STOPPED = 6  # (Legacy)
    BOMB_DEFUSED = 7  # CTs defused the bomb
    CT_WIN = 8  # CTs eliminated terrorists
    TERRORIST_WIN = 9  # Terrorists eliminated CTs
    ROUND_DRAW = 10  # Draw
    ALL_HOSTAGES_RESCUED = 11  # CTs rescued hostages
    TARGET_SAVED = 12  # Time ran out, bomb not planted
    HOSTAGES_NOT_RESCUED = 13  # Terrorists won hostage round
    TERRORISTS_SURRENDER = 14  # Terrorists surrendered
    CT_SURRENDER = 15  # CTs surrendered
    TERRORISTS_PLANTED = 16  # (Technical)
    CTS_REACHED_HOSTAGE = 17  # (Technical)


# Map pool groupings (Updated January 2026)
# Active Duty pool changes frequently - check Valve release notes
ACTIVE_DUTY_MAPS = {
    "de_mirage",
    "de_inferno",
    "de_nuke",
    "de_dust2",
    "de_ancient",
    "de_train",
    "de_overpass",
}

# Reserve pool maps (can be played in Competitive but not Premier)
RESERVE_MAPS = {"de_anubis", "de_vertigo"}

# All competitive maps (Active Duty + Reserve)
COMPETITIVE_MAPS = ACTIVE_DUTY_MAPS | RESERVE_MAPS

WINGMAN_MAPS = {
    "de_inferno",
    "de_overpass",
    "de_vertigo",
    "de_nuke",
    "de_memento",
    "de_assembly",
    "de_rooftop",
    "de_transit",
}

# Server identifiers for source detection
SOURCE_IDENTIFIERS = {
    DemoSource.FACEIT: ["FACEIT", "faceit.com"],
    DemoSource.ESEA: ["ESEA", "esea.net", "play.esea.net"],
    DemoSource.ESL: ["ESL", "esl.gg", "ESL_"],
    DemoSource.ESPORTAL: ["Esportal", "esportal.com"],
    DemoSource.CHALLENGERMODE: ["Challengermode", "challengermode.com"],
    DemoSource.POPFLASH: ["PopFlash", "popflash.site"],
    DemoSource.FASTCUP: ["FastCup", "fastcup.net"],
    DemoSource.FIVEPLAY: ["5EPlay", "5ewin", "5eplay.com"],
    DemoSource.PERFECTWORLD: ["Perfect World", "完美世界"],
    DemoSource.EBOT: ["eBot"],
    DemoSource.CEVO: ["CEVO"],
}

# Filename patterns for source detection
FILENAME_PATTERNS = {
    DemoSource.VALVE: ["match730_"],
    DemoSource.FACEIT: ["faceit_", "faceit-"],
    DemoSource.ESEA: ["esea_"],
    DemoSource.ESL: ["esl_"],
}

# Tick rates by source
# NOTE: CS2 uses 64 tick UNIVERSALLY with the subtick system
# This is different from CS:GO which had 64 tick (Valve) and 128 tick (FACEIT/ESEA)
# The subtick system timestamps actions precisely between ticks, making 64 tick
# effectively as responsive as 128 tick was in CS:GO
CS2_TICK_RATE = 64  # All CS2 servers use 64 tick

TICK_RATES = {
    DemoSource.VALVE: 64,  # Valve MM - 64 tick + subtick
    DemoSource.FACEIT: 64,  # FACEIT - 64 tick + subtick (NOT 128 like CS:GO)
    DemoSource.ESEA: 64,  # ESEA - 64 tick + subtick (NOT 128 like CS:GO)
    DemoSource.ESL: 64,  # ESL - 64 tick + subtick
    DemoSource.ESPORTAL: 64,  # Esportal - 64 tick + subtick
    DemoSource.HLTV: 64,  # Pro matches - 64 tick + subtick
    DemoSource.UNKNOWN: 64,  # Default to 64 tick
}

# Trade window in seconds (industry standard from Leetify/Stratbook)
TRADE_WINDOW_SECONDS = 5.0

# Trade proximity threshold in game units.
# Only teammates within this distance of a death are counted as having
# a "trade opportunity". Reduced from 1500 to 1000 to match Leetify's
# tighter definition — ~1000 units = close enough to realistically trade
# without requiring a major rotation.
TRADE_PROXIMITY_UNITS = 1000.0

# Flash assist minimum duration (seconds)
FLASH_ASSIST_MIN_DURATION = 0.5

# Clutch timing (last player alive threshold in seconds)
CLUTCH_TIME_THRESHOLD = 60.0

# HLTV 2.0 Rating coefficients (approximated from public analysis)
HLTV_RATING_COEFFICIENTS = {
    "kast": 0.0073,
    "kpr": 0.3591,
    "dpr": -0.5329,
    "impact": 0.2372,
    "adr": 0.0032,
    "rmk": 0.1587,
}

# Impact rating sub-coefficients
IMPACT_COEFFICIENTS = {
    "kpr": 2.13,
    "apr": 0.42,
    "base": -0.41,
}
