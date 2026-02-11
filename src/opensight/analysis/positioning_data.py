"""
Positional Coordinate Extraction System for CS2 Demo Analysis.

Extracts 5 categories of position data from parsed demos:
1. Death positions (where players died)
2. Defensive setup positions (CT positions at freeze-time end)
3. Engagement positions (attacker location at time of kill)
4. Grenade landing positions (utility impact locations)
5. Firing velocity (player speed at moment of each shot)

Uses map_zones.py for zone classification (e.g., "Banana", "A Site").
"""

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from opensight.core.map_zones import get_callout
from opensight.core.parser import safe_float

if TYPE_CHECKING:
    from opensight.analysis.analytics import DemoAnalyzer

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class PositionEvent:
    """A single position event extracted from a demo."""

    tick: int
    steam_id: str
    x: float
    y: float
    z: float
    event_type: str  # "death", "kill", "grenade", "freeze_end", "shot"
    zone: str  # Resolved callout name from map_zones
    round_num: int
    metadata: dict = field(default_factory=dict)  # Extra context per event type


@dataclass
class ZoneFrequency:
    """Aggregated zone frequency for a player."""

    zone: str
    count: int
    percentage: float


@dataclass
class PlayerPositionStats:
    """Per-player position analysis summary."""

    steam_id: str
    death_zones: list[ZoneFrequency] = field(default_factory=list)
    kill_zones: list[ZoneFrequency] = field(default_factory=list)
    top_death_zone: str = "Unknown"
    top_kill_zone: str = "Unknown"
    ct_setup_zones: list[str] = field(default_factory=list)  # Most common CT setup positions
    avg_firing_velocity: float | None = None
    shots_stationary_pct: float | None = None
    total_positions_extracted: int = 0


@dataclass
class PositionAnalysis:
    """Complete position analysis for a match."""

    death_positions: list[PositionEvent] = field(default_factory=list)
    engagement_positions: list[PositionEvent] = field(default_factory=list)
    defensive_setups: list[PositionEvent] = field(default_factory=list)
    grenade_positions: list[PositionEvent] = field(default_factory=list)
    firing_velocity_events: list[PositionEvent] = field(default_factory=list)
    player_stats: dict[str, PlayerPositionStats] = field(default_factory=dict)
    map_name: str = ""


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================


def extract_positions(analyzer: "DemoAnalyzer") -> PositionAnalysis:
    """Extract all position data from a parsed demo.

    Args:
        analyzer: DemoAnalyzer instance with parsed data

    Returns:
        PositionAnalysis with all extracted position events
    """
    map_name = getattr(analyzer.data, "map_name", "") or ""
    result = PositionAnalysis(map_name=map_name)

    # Extract each category
    result.death_positions = _extract_death_positions(analyzer, map_name)
    result.engagement_positions = _extract_engagement_positions(analyzer, map_name)
    result.defensive_setups = _extract_defensive_setups(analyzer, map_name)
    result.grenade_positions = _extract_grenade_positions(analyzer, map_name)
    result.firing_velocity_events = _extract_firing_velocity(analyzer, map_name)

    # Build per-player aggregated stats
    result.player_stats = _build_player_position_stats(result, analyzer)

    total = (
        len(result.death_positions)
        + len(result.engagement_positions)
        + len(result.defensive_setups)
        + len(result.grenade_positions)
        + len(result.firing_velocity_events)
    )
    logger.info(
        f"Position extraction complete: {total} events "
        f"({len(result.death_positions)} deaths, "
        f"{len(result.engagement_positions)} engagements, "
        f"{len(result.defensive_setups)} setups, "
        f"{len(result.grenade_positions)} grenades, "
        f"{len(result.firing_velocity_events)} velocity samples)"
    )

    return result


def _extract_death_positions(analyzer: "DemoAnalyzer", map_name: str) -> list[PositionEvent]:
    """Extract death positions from kill events."""
    events = []
    kills = getattr(analyzer.data, "kills", [])

    for kill in kills:
        try:
            vx = safe_float(getattr(kill, "victim_x", None))
            vy = safe_float(getattr(kill, "victim_y", None))
            vz = safe_float(getattr(kill, "victim_z", None))

            if vx is None or vy is None or math.isnan(vx) or math.isnan(vy):
                continue

            vz = vz if (vz is not None and not math.isnan(vz)) else 0.0

            zone = get_callout(map_name, vx, vy, vz)
            steam_id = str(getattr(kill, "victim_steamid", ""))

            events.append(
                PositionEvent(
                    tick=getattr(kill, "tick", 0),
                    steam_id=steam_id,
                    x=vx,
                    y=vy,
                    z=vz,
                    event_type="death",
                    zone=zone,
                    round_num=getattr(kill, "round_num", 0),
                    metadata={
                        "weapon": str(getattr(kill, "weapon", "unknown")),
                        "headshot": bool(getattr(kill, "headshot", False)),
                        "attacker": str(getattr(kill, "attacker_steamid", "")),
                    },
                )
            )
        except Exception as e:
            logger.debug(f"Error extracting death position: {e}")
            continue

    logger.info(f"Extracted {len(events)} death positions from {len(kills)} kills")
    return events


def _extract_engagement_positions(analyzer: "DemoAnalyzer", map_name: str) -> list[PositionEvent]:
    """Extract attacker positions at time of kill."""
    events = []
    kills = getattr(analyzer.data, "kills", [])

    for kill in kills:
        try:
            ax = safe_float(getattr(kill, "attacker_x", None))
            ay = safe_float(getattr(kill, "attacker_y", None))
            az = safe_float(getattr(kill, "attacker_z", None))

            if ax is None or ay is None or math.isnan(ax) or math.isnan(ay):
                continue

            az = az if (az is not None and not math.isnan(az)) else 0.0

            zone = get_callout(map_name, ax, ay, az)
            steam_id = str(getattr(kill, "attacker_steamid", ""))

            events.append(
                PositionEvent(
                    tick=getattr(kill, "tick", 0),
                    steam_id=steam_id,
                    x=ax,
                    y=ay,
                    z=az,
                    event_type="kill",
                    zone=zone,
                    round_num=getattr(kill, "round_num", 0),
                    metadata={
                        "weapon": str(getattr(kill, "weapon", "unknown")),
                        "headshot": bool(getattr(kill, "headshot", False)),
                        "victim": str(getattr(kill, "victim_steamid", "")),
                        "distance": safe_float(getattr(kill, "distance", None)),
                    },
                )
            )
        except Exception as e:
            logger.debug(f"Error extracting engagement position: {e}")
            continue

    logger.info(f"Extracted {len(events)} engagement positions")
    return events


def _extract_defensive_setups(analyzer: "DemoAnalyzer", map_name: str) -> list[PositionEvent]:
    """Extract CT positions at freeze-time end for defensive setup analysis.

    Uses tick data (ticks_df) to find player positions at round start.
    Falls back to first kill of round if tick data unavailable.
    """
    events = []
    ticks_df = getattr(analyzer.data, "ticks_df", None)
    rounds = getattr(analyzer.data, "rounds", [])

    if ticks_df is None or (hasattr(ticks_df, "empty") and ticks_df.empty):
        logger.info("No tick data for defensive setup extraction — skipping")
        return events

    try:
        for rnd in rounds:
            freeze_end_tick = getattr(rnd, "freeze_end_tick", None)
            round_num = getattr(rnd, "round_num", 0)

            if freeze_end_tick is None or freeze_end_tick <= 0:
                continue

            # Get player positions at freeze end tick (±2 ticks tolerance)
            mask = (ticks_df["tick"] >= freeze_end_tick - 2) & (
                ticks_df["tick"] <= freeze_end_tick + 2
            )
            freeze_positions = ticks_df[mask]

            if freeze_positions.empty:
                continue

            # Group by steamid, take closest to freeze_end_tick
            for steam_id, group in freeze_positions.groupby("steamid"):
                closest = group.iloc[(group["tick"] - freeze_end_tick).abs().argsort()[:1]]
                if closest.empty:
                    continue

                row = closest.iloc[0]
                x = safe_float(row.get("X", None))
                y = safe_float(row.get("Y", None))
                z = safe_float(row.get("Z", None))

                if x is None or y is None or math.isnan(x) or math.isnan(y):
                    continue

                z = z if (z is not None and not math.isnan(z)) else 0.0

                # Only include CT-side players (team 3)
                team = row.get("team_num", 0)
                if team != 3:
                    continue

                zone = get_callout(map_name, x, y, z)

                events.append(
                    PositionEvent(
                        tick=int(freeze_end_tick),
                        steam_id=str(steam_id),
                        x=x,
                        y=y,
                        z=z,
                        event_type="freeze_end",
                        zone=zone,
                        round_num=round_num,
                        metadata={"team": "CT"},
                    )
                )
    except Exception as e:
        logger.warning(f"Error extracting defensive setups: {e}")

    logger.info(f"Extracted {len(events)} defensive setup positions")
    return events


def _extract_grenade_positions(analyzer: "DemoAnalyzer", map_name: str) -> list[PositionEvent]:
    """Extract grenade landing/detonation positions."""
    events = []
    grenades = getattr(analyzer.data, "grenades", [])

    for nade in grenades:
        try:
            x = safe_float(getattr(nade, "x", None))
            y = safe_float(getattr(nade, "y", None))
            z = safe_float(getattr(nade, "z", None))

            if x is None or y is None or math.isnan(x) or math.isnan(y):
                continue

            z = z if (z is not None and not math.isnan(z)) else 0.0

            zone = get_callout(map_name, x, y, z)
            steam_id = str(getattr(nade, "thrower_steamid", "") or getattr(nade, "steamid", ""))
            nade_type = str(getattr(nade, "grenade_type", "") or getattr(nade, "weapon", "unknown"))

            events.append(
                PositionEvent(
                    tick=getattr(nade, "tick", 0),
                    steam_id=steam_id,
                    x=x,
                    y=y,
                    z=z,
                    event_type="grenade",
                    zone=zone,
                    round_num=getattr(nade, "round_num", 0),
                    metadata={"grenade_type": nade_type},
                )
            )
        except Exception as e:
            logger.debug(f"Error extracting grenade position: {e}")
            continue

    logger.info(f"Extracted {len(events)} grenade positions from {len(grenades)} events")
    return events


def _extract_firing_velocity(analyzer: "DemoAnalyzer", map_name: str) -> list[PositionEvent]:
    """Extract player velocity at time of shots/kills for counter-strafe analysis."""
    events = []
    kills = getattr(analyzer.data, "kills", [])

    for kill in kills:
        try:
            ax = safe_float(getattr(kill, "attacker_x", None))
            ay = safe_float(getattr(kill, "attacker_y", None))

            if ax is None or ay is None or math.isnan(ax) or math.isnan(ay):
                continue

            # Get velocity components if available
            vel_x = safe_float(getattr(kill, "attacker_vel_x", None)) or 0.0
            vel_y = safe_float(getattr(kill, "attacker_vel_y", None)) or 0.0
            vel_z = safe_float(getattr(kill, "attacker_vel_z", None)) or 0.0

            # Handle NaN velocities
            if math.isnan(vel_x):
                vel_x = 0.0
            if math.isnan(vel_y):
                vel_y = 0.0
            if math.isnan(vel_z):
                vel_z = 0.0

            speed = math.sqrt(vel_x**2 + vel_y**2)  # Horizontal speed only
            is_stationary = speed < 34.0  # CS2 threshold for standing accuracy

            zone = get_callout(map_name, ax, ay, 0)
            steam_id = str(getattr(kill, "attacker_steamid", ""))

            events.append(
                PositionEvent(
                    tick=getattr(kill, "tick", 0),
                    steam_id=steam_id,
                    x=ax,
                    y=ay,
                    z=0,
                    event_type="shot",
                    zone=zone,
                    round_num=getattr(kill, "round_num", 0),
                    metadata={
                        "speed": round(speed, 1),
                        "is_stationary": is_stationary,
                        "vel_x": round(vel_x, 1),
                        "vel_y": round(vel_y, 1),
                    },
                )
            )
        except Exception as e:
            logger.debug(f"Error extracting firing velocity: {e}")
            continue

    logger.info(f"Extracted {len(events)} firing velocity samples")
    return events


# =============================================================================
# AGGREGATION
# =============================================================================


def _count_zones(events: list[PositionEvent]) -> list[ZoneFrequency]:
    """Count zone frequency from a list of events."""
    zone_counts: dict[str, int] = {}
    for e in events:
        zone_counts[e.zone] = zone_counts.get(e.zone, 0) + 1

    total = len(events) or 1
    return sorted(
        [
            ZoneFrequency(zone=z, count=c, percentage=round(c / total * 100, 1))
            for z, c in zone_counts.items()
        ],
        key=lambda x: x.count,
        reverse=True,
    )


def _build_player_position_stats(
    analysis: PositionAnalysis, analyzer: "DemoAnalyzer"
) -> dict[str, PlayerPositionStats]:
    """Build per-player position summaries."""
    stats: dict[str, PlayerPositionStats] = {}
    all_steam_ids = set()

    # Collect all steam IDs
    for ev_list in [
        analysis.death_positions,
        analysis.engagement_positions,
        analysis.grenade_positions,
        analysis.firing_velocity_events,
    ]:
        for ev in ev_list:
            if ev.steam_id:
                all_steam_ids.add(ev.steam_id)

    for sid in all_steam_ids:
        ps = PlayerPositionStats(steam_id=sid)

        # Death zones
        player_deaths = [e for e in analysis.death_positions if e.steam_id == sid]
        ps.death_zones = _count_zones(player_deaths)
        if ps.death_zones:
            ps.top_death_zone = ps.death_zones[0].zone

        # Kill zones
        player_kills = [e for e in analysis.engagement_positions if e.steam_id == sid]
        ps.kill_zones = _count_zones(player_kills)
        if ps.kill_zones:
            ps.top_kill_zone = ps.kill_zones[0].zone

        # CT setup zones
        player_setups = [e for e in analysis.defensive_setups if e.steam_id == sid]
        setup_zones = _count_zones(player_setups)
        ps.ct_setup_zones = [z.zone for z in setup_zones[:3]]  # Top 3 setup zones

        # Firing velocity
        player_shots = [e for e in analysis.firing_velocity_events if e.steam_id == sid]
        if player_shots:
            speeds = [e.metadata.get("speed", 0) for e in player_shots]
            stationary = [e for e in player_shots if e.metadata.get("is_stationary")]
            ps.avg_firing_velocity = round(sum(speeds) / len(speeds), 1) if speeds else None
            ps.shots_stationary_pct = (
                round(len(stationary) / len(player_shots) * 100, 1) if player_shots else None
            )

        ps.total_positions_extracted = (
            len(player_deaths) + len(player_kills) + len(player_setups) + len(player_shots)
        )

        stats[sid] = ps

    return stats


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================


def serialize_position_analysis(analysis: PositionAnalysis) -> dict:
    """Serialize PositionAnalysis for JSON output to frontend."""
    return {
        "map_name": analysis.map_name,
        "summary": {
            "total_death_positions": len(analysis.death_positions),
            "total_engagement_positions": len(analysis.engagement_positions),
            "total_defensive_setups": len(analysis.defensive_setups),
            "total_grenade_positions": len(analysis.grenade_positions),
            "total_velocity_samples": len(analysis.firing_velocity_events),
        },
        "heatmap_data": {
            "deaths": [
                {"x": e.x, "y": e.y, "z": e.z, "zone": e.zone, "round": e.round_num}
                for e in analysis.death_positions
            ],
            "engagements": [
                {"x": e.x, "y": e.y, "z": e.z, "zone": e.zone, "round": e.round_num}
                for e in analysis.engagement_positions
            ],
            "grenades": [
                {
                    "x": e.x,
                    "y": e.y,
                    "z": e.z,
                    "zone": e.zone,
                    "round": e.round_num,
                    "type": e.metadata.get("grenade_type", "unknown"),
                }
                for e in analysis.grenade_positions
            ],
        },
    }


def serialize_player_positions(stats: PlayerPositionStats | None) -> dict | None:
    """Serialize a single player's position stats for JSON output."""
    if stats is None:
        return None
    return {
        "top_death_zone": stats.top_death_zone,
        "top_kill_zone": stats.top_kill_zone,
        "death_zones": [
            {"zone": z.zone, "count": z.count, "pct": z.percentage} for z in stats.death_zones[:5]
        ],
        "kill_zones": [
            {"zone": z.zone, "count": z.count, "pct": z.percentage} for z in stats.kill_zones[:5]
        ],
        "ct_setup_zones": stats.ct_setup_zones,
        "avg_firing_velocity": stats.avg_firing_velocity,
        "shots_stationary_pct": stats.shots_stationary_pct,
        "total_positions": stats.total_positions_extracted,
    }
