"""
Maps, radar, replay, and rotation route handlers.

Endpoints:
- GET /maps — list available maps
- GET /maps/{map_name} — map metadata and radar info
- POST /radar/transform — game coords to radar pixels
- GET /hltv/rankings — world team rankings
- GET /hltv/map/{map_name} — map statistics
- GET /hltv/player/search — search players by nickname
- POST /hltv/enrich — enrich analysis with HLTV data
- POST /replay/generate — generate 2D replay data
- POST /tactical/rotations — CT rotation latency analysis
"""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from fastapi import APIRouter, Body, File, HTTPException, Query, Request, UploadFile

from opensight.api.shared import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    RATE_LIMIT_API,
    RATE_LIMIT_REPLAY,
    RadarRequest,
    rate_limit,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["maps"])


# =============================================================================
# Radar Map Endpoints
# =============================================================================


@router.get("/maps")
async def list_maps() -> dict[str, Any]:
    """List all available maps with radar support."""
    try:
        from opensight.visualization.radar import MAP_DATA

        return {
            "maps": [
                {
                    "internal_name": name,
                    "display_name": data["name"],
                    "has_radar": True,
                }
                for name, data in MAP_DATA.items()
            ]
        }
    except ImportError:
        return {"maps": [], "error": "Radar module not available"}


@router.get("/maps/{map_name}")
async def get_map_info(map_name: str) -> dict[str, Any]:
    """Get map metadata and radar information."""
    try:
        from opensight.visualization.radar import RadarImageManager, get_map_metadata

        metadata = get_map_metadata(map_name)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Map not found: {map_name}")

        manager = RadarImageManager()
        radar_url = manager.get_radar_url(map_name)

        return {
            "name": metadata.name,
            "internal_name": metadata.internal_name,
            "pos_x": metadata.pos_x,
            "pos_y": metadata.pos_y,
            "scale": metadata.scale,
            "radar_url": radar_url,
            "z_cutoff": metadata.z_cutoff,
            "has_multiple_levels": metadata.z_cutoff is not None,
        }
    except ImportError as e:
        logger.warning("Radar module not available: %s", e)
        raise HTTPException(status_code=503, detail="Radar module not available.") from e


@router.post("/radar/transform")
async def transform_coordinates(request: RadarRequest) -> dict[str, Any]:
    """Transform game coordinates to radar pixel coordinates."""
    try:
        from opensight.visualization.radar import CoordinateTransformer

        transformer = CoordinateTransformer(request.map_name)
        results = []

        for pos in request.positions:
            x = pos.get("x", 0.0)
            y = pos.get("y", 0.0)
            z = pos.get("z", 0.0)
            radar_pos = transformer.game_to_radar(x, y, z)
            results.append(
                {
                    "game": {"x": x, "y": y, "z": z},
                    "radar": {"x": round(radar_pos.x, 1), "y": round(radar_pos.y, 1)},
                    "is_upper_level": transformer.is_upper_level(z),
                }
            )

        return {
            "map_name": request.map_name,
            "positions": results,
        }
    except ImportError as e:
        logger.warning("Radar module not available: %s", e)
        raise HTTPException(status_code=503, detail="Radar module not available.") from e


# =============================================================================
# HLTV Integration Endpoints
# =============================================================================


@router.get("/hltv/rankings")
async def get_hltv_rankings(
    top_n: Annotated[int, Query(le=30)] = 10,
) -> dict[str, Any]:
    """Get current world team rankings (cached data)."""
    try:
        from opensight.integrations.hltv import HLTVClient

        client = HLTVClient()
        return {"rankings": client.get_world_rankings(top_n)}
    except ImportError as e:
        logger.warning("HLTV module not available: %s", e)
        raise HTTPException(status_code=503, detail="HLTV module not available.") from e


@router.get("/hltv/map/{map_name}")
async def get_hltv_map_stats(map_name: str) -> dict[str, Any]:
    """Get map statistics from HLTV data."""
    try:
        from opensight.integrations.hltv import get_map_statistics

        stats = get_map_statistics(map_name)
        if not stats:
            raise HTTPException(status_code=404, detail=f"No stats for map: {map_name}")
        return stats
    except ImportError as e:
        logger.warning("HLTV module not available: %s", e)
        raise HTTPException(status_code=503, detail="HLTV module not available.") from e


@router.get("/hltv/player/search")
async def search_hltv_player(
    nickname: Annotated[str, Query(..., min_length=2)],
) -> dict[str, Any]:
    """Search for a player by nickname."""
    try:
        from opensight.integrations.hltv import HLTVClient

        client = HLTVClient()
        return {"results": client.search_player(nickname)}
    except ImportError as e:
        logger.warning("HLTV module not available: %s", e)
        raise HTTPException(status_code=503, detail="HLTV module not available.") from e


@router.post("/hltv/enrich")
async def enrich_analysis(
    analysis_data: Annotated[dict[str, Any], Body(...)],
) -> dict[str, Any]:
    """Enrich analysis data with HLTV information."""
    try:
        from opensight.integrations.hltv import enrich_match_analysis

        return enrich_match_analysis(analysis_data)
    except ImportError as e:
        logger.warning("HLTV module not available: %s", e)
        raise HTTPException(status_code=503, detail="HLTV module not available.") from e


# =============================================================================
# 2D Replay Data Endpoints
# =============================================================================


@router.post("/replay/generate")
@rate_limit(RATE_LIMIT_REPLAY)
async def generate_replay_data(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    sample_rate: Annotated[int, Query(ge=1, le=128, description="Extract every Nth tick")] = 16,
) -> dict[str, Any]:
    """Generate 2D replay data from a demo file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="File must be a .dem or .dem.gz file")

    try:
        from opensight.core.parser import DemoParser
        from opensight.visualization.radar import CoordinateTransformer
        from opensight.visualization.replay import ReplayGenerator
    except ImportError as e:
        logger.warning("Replay module not available: %s", e)
        raise HTTPException(status_code=503, detail="Replay module not available.") from e

    tmp_path = None
    try:
        content = await file.read()
        file_size_bytes = len(content)

        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")

        if file_size_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        suffix = ".dem.gz" if filename_lower.endswith(".dem.gz") else ".dem"
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Direct parsing — this endpoint is not job-based, it accepts ad-hoc uploads
        # for 2D replay visualization. Replay generation requires raw tick-level
        # positional data (player coords, bomb state per-frame) that is NOT stored
        # in cached orchestrator results. Not an orchestrator bypass.
        parser = DemoParser(tmp_path)
        data = parser.parse()

        generator = ReplayGenerator(data, sample_rate=sample_rate)
        replay = generator.generate_full_replay()

        if not replay.rounds:
            raise HTTPException(
                status_code=422,
                detail="No replay data available. This demo may not contain tick-level position data required for 2D replay.",
            )

        transformer = CoordinateTransformer(data.map_name)

        all_replay_frames = [frame for r in replay.rounds for frame in r.frames]

        total_ticks = 0
        if replay.rounds:
            total_ticks = replay.rounds[-1].end_tick - replay.rounds[0].start_tick

        frames = []
        for frame in all_replay_frames[:10000]:
            frame_data: dict[str, Any] = {
                "tick": frame.tick,
                "round": frame.round_num,
                "time_in_round": round(frame.game_time, 2),
                "players": [],
                "bomb": None,
            }

            for player in frame.players:
                radar_pos = transformer.game_to_radar(player.x, player.y, player.z)
                frame_data["players"].append(
                    {
                        "steam_id": str(player.steam_id),
                        "name": player.name,
                        "team": player.team,
                        "x": round(radar_pos.x, 1),
                        "y": round(radar_pos.y, 1),
                        "yaw": round(player.yaw, 1),
                        "health": player.health,
                        "armor": player.armor,
                        "is_alive": player.is_alive,
                        "weapon": player.active_weapon,
                        "money": player.money,
                        "equipment_value": player.equipment_value,
                    }
                )

            if frame.bomb:
                bomb_pos = transformer.game_to_radar(frame.bomb.x, frame.bomb.y, frame.bomb.z)
                bomb_state = (
                    frame.bomb.state.value
                    if hasattr(frame.bomb.state, "value")
                    else frame.bomb.state
                )
                frame_data["bomb"] = {
                    "x": round(bomb_pos.x, 1),
                    "y": round(bomb_pos.y, 1),
                    "state": bomb_state,
                }

            if frame.events:
                frame_data["events"] = frame.events

            frames.append(frame_data)

        # Build kill events per round with radar coords
        rounds_data = []
        for r in replay.rounds:
            round_info: dict[str, Any] = {
                "round_num": r.round_num,
                "start_tick": r.start_tick,
                "end_tick": r.end_tick,
                "winner": r.winner,
            }
            if r.kills:
                round_info["kills"] = [
                    {
                        "tick": k.tick,
                        "attacker": k.attacker_name,
                        "victim": k.victim_name,
                        "weapon": k.weapon,
                        "headshot": k.headshot,
                        "x": round(transformer.game_to_radar(k.x, k.y).x, 1),
                        "y": round(transformer.game_to_radar(k.x, k.y).y, 1),
                    }
                    for k in r.kills
                ]
            rounds_data.append(round_info)

        return {
            "map_name": replay.map_name,
            "total_ticks": total_ticks,
            "tick_rate": replay.tick_rate,
            "sample_rate": sample_rate,
            "total_frames": len(all_replay_frames),
            "frames_returned": len(frames),
            "rounds": rounds_data,
            "frames": frames,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Replay generation failed")
        raise HTTPException(
            status_code=500, detail="Replay generation failed. Check server logs."
        ) from e
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# =============================================================================
# Tactical Analysis - CT Rotation Latency
# =============================================================================


@router.post("/tactical/rotations")
@rate_limit(RATE_LIMIT_API)
async def analyze_rotations(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """Analyze CT rotation latency from a demo file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="File must be a .dem or .dem.gz file")

    try:
        from opensight.analysis.rotation import CTRotationAnalyzer, get_rotation_summary
        from opensight.core.parser import DemoParser
    except ImportError as e:
        logger.warning("Rotation module not available: %s", e)
        raise HTTPException(status_code=503, detail="Rotation module not available.") from e

    tmp_path = None
    try:
        content = await file.read()
        file_size_bytes = len(content)

        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")

        if file_size_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        suffix = ".dem.gz" if filename_lower.endswith(".dem.gz") else ".dem"
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Direct parsing — this endpoint is not job-based, it accepts ad-hoc uploads
        # for CT rotation latency analysis. The CTRotationAnalyzer requires raw
        # tick-level positional data (player movement over time) that is NOT stored
        # in cached orchestrator results. Not an orchestrator bypass.
        parser = DemoParser(tmp_path)
        data = parser.parse()

        analyzer = CTRotationAnalyzer(data, data.map_name)
        team_stats = analyzer.analyze()
        advice = analyzer.get_rotation_advice()
        summary = get_rotation_summary(data, data.map_name)

        player_stats_list = []
        for player in team_stats.player_stats.values():
            player_stats_list.append(player.to_dict())

        total_rotations = (
            team_stats.total_over_rotations
            + team_stats.total_balanced_rotations
            + team_stats.total_slow_rotations
        )

        return {
            "map_name": data.map_name,
            "team_stats": {
                "team_avg_reaction_sec": team_stats.avg_team_reaction_time,
                "team_avg_travel_sec": team_stats.avg_team_travel_time,
                "total_rotations": total_rotations,
                "total_over_rotations": team_stats.total_over_rotations,
                "total_balanced_rotations": team_stats.total_balanced_rotations,
                "total_slow_rotations": team_stats.total_slow_rotations,
                "rounds_analyzed": team_stats.rounds_analyzed,
            },
            "player_stats": player_stats_list,
            "advice": advice,
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Rotation analysis failed")
        raise HTTPException(
            status_code=500, detail="Rotation analysis failed. Check server logs."
        ) from e
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
