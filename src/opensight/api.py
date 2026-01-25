"""
OpenSight Web API

FastAPI application for CS2 demo analysis with professional-grade metrics.

Provides:
- Demo analysis with HLTV 2.0 Rating, KAST%, TTD, Crosshair Placement
- Batch analysis with parallel processing
- 2D replay data generation
- Radar map coordinate transformation
- HLTV integration for pro player detection
- Caching for faster repeated analysis
- Community feedback system
"""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

__version__ = "0.3.0"

# Security constants
MAX_FILE_SIZE_MB = 500  # Maximum demo file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = (".dem", ".dem.gz")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenSight API",
    description=(
        "CS2 demo analyzer - professional-grade metrics including "
        "HLTV 2.0 Rating, KAST%, TTD, and Crosshair Placement"
    ),
    version=__version__,
)


class ShareCodeRequest(BaseModel):
    code: str


class ShareCodeResponse(BaseModel):
    match_id: int
    outcome_id: int
    token: int


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    demo_hash: str = Field(..., description="Hash of the demo file")
    player_steam_id: str = Field(..., description="Steam ID of the player")
    metric_name: str = Field(..., description="Name of the metric being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: str | None = Field(None, description="Optional comment")
    correction_value: float | None = Field(None, description="Optional corrected value")


class CoachingFeedbackRequest(BaseModel):
    """Request model for coaching feedback."""
    demo_hash: str
    insight_id: str
    was_helpful: bool
    correction: str | None = None


class RadarRequest(BaseModel):
    """Request model for radar coordinate transformation."""
    map_name: str
    positions: list[dict[str, float]] = Field(..., description="List of {x, y, z} game coordinates")


# Get the static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the main web interface."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>OpenSight</h1><p>Web interface not found.</p>",
        status_code=200,
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.post("/decode", response_model=ShareCodeResponse)
async def decode_share_code(request: ShareCodeRequest) -> dict[str, int]:
    """Decode a CS2 share code to extract match metadata."""
    try:
        from opensight.integrations.sharecode import decode_sharecode
        info = decode_sharecode(request.code)
        return {
            "match_id": info.match_id,
            "outcome_id": info.outcome_id,
            "token": info.token,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Module not available: {e!s}"
        ) from e


@app.post("/analyze")
async def analyze_demo(
    file: Annotated[UploadFile, File(...)],
) -> JSONResponse:
    """
    Analyze an uploaded CS2 demo file.

    Returns comprehensive player stats including:
    - Basic stats: Kills, Deaths, Assists, K/D, ADR, HS%
    - Professional metrics: HLTV 2.0 Rating, KAST%, Impact
    - Advanced metrics: TTD (Time to Damage), Crosshair Placement
    - Weapon breakdown

    Accepts .dem and .dem.gz files up to 500MB.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a .dem or .dem.gz file. Got: {file.filename}"
        )

    try:
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.core.parser import DemoParser
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Demo analysis not available. Missing: {e!s}"
        ) from e

    tmp_path = None
    try:
        # Read and validate file size
        content = await file.read()
        file_size_bytes = len(content)
        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: {file_size_mb:.1f}MB. "
                    f"Maximum allowed: {MAX_FILE_SIZE_MB}MB"
                ),
            )

        if file_size_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Analyzing demo: {file.filename} ({file_size_mb:.1f} MB)")

        # Save uploaded file temporarily
        # Use appropriate suffix based on file type
        suffix = ".dem.gz" if filename_lower.endswith(".dem.gz") else ".dem"
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Parse the demo
            parser = DemoParser(tmp_path)
            data = parser.parse()

            # Run advanced analytics
            analyzer = DemoAnalyzer(data)
            analysis = analyzer.analyze()
        finally:
            # Ensure temp file is cleaned up even if parsing fails
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            tmp_path = None

        # Build response
        result: dict[str, Any] = {
            "demo_info": {
                "map": analysis.map_name,
                "duration_seconds": round(data.duration_seconds, 1),
                "duration_minutes": (
                    round(data.duration_seconds / 60, 1) if data.duration_seconds else 0
                ),
                "tick_rate": data.tick_rate,
                "rounds": analysis.total_rounds,
                "score": f"{analysis.team1_score} - {analysis.team2_score}",
                "player_count": len(analysis.players),
                "total_kills": len(data.kills),
                "total_damage_events": len(data.damages),
            },
            "mvp": None,
            "players": {},
        }

        # Add MVP
        mvp = analysis.get_mvp()
        if mvp:
            result["mvp"] = {
                "steam_id": str(mvp.steam_id),
                "name": mvp.name,
                "rating": mvp.hltv_rating,
            }

        # Add player stats with advanced metrics (sorted by rating)
        for player in analysis.get_leaderboard():
            steam_id = player.steam_id

            # Build weapon stats
            weapon_stats = []
            for weapon, count in sorted(player.weapon_kills.items(), key=lambda x: -x[1]):
                weapon_stats.append({"weapon": weapon, "kills": count})

            result["players"][str(steam_id)] = {
                "name": player.name,
                "team": player.team,
                "stats": {
                    # Basic stats
                    "kills": player.kills,
                    "deaths": player.deaths,
                    "assists": player.assists,
                    "kd_ratio": player.kd_ratio,
                    "kd_diff": player.kd_diff,
                    "headshots": player.headshots,
                    "headshot_pct": player.headshot_percentage,
                    "total_damage": player.total_damage,
                    "adr": player.adr,
                },
                "rating": {
                    # HLTV 2.0 Rating components
                    "hltv_rating": player.hltv_rating,
                    "impact_rating": player.impact_rating,
                    "kast_percentage": player.kast_percentage,
                    "kills_per_round": player.kills_per_round,
                    "deaths_per_round": player.deaths_per_round,
                    "survival_rate": player.survival_rate,
                    # Leetify-style composite ratings
                    "aim_rating": player.aim_rating,
                    "utility_rating": player.utility_rating,
                    "entry_success_rate": player.entry_success_rate,
                },
                "duels": {
                    # Opening duels
                    "opening_attempts": player.opening_duels.attempts,
                    "opening_wins": player.opening_duels.wins,
                    "opening_losses": player.opening_duels.losses,
                    "opening_win_rate": player.opening_duels.win_rate,
                    # Trades
                    "kills_traded": player.trades.kills_traded,
                    "deaths_traded": player.trades.deaths_traded,
                },
                "clutches": {
                    "total_situations": player.clutches.total_situations,
                    "total_wins": player.clutches.total_wins,
                    "1v1": {
                        "attempts": player.clutches.situations_1v1,
                        "wins": player.clutches.wins_1v1,
                    },
                    "1v2": {
                        "attempts": player.clutches.situations_1v2,
                        "wins": player.clutches.wins_1v2,
                    },
                    "1v3": {
                        "attempts": player.clutches.situations_1v3,
                        "wins": player.clutches.wins_1v3,
                    },
                    "1v4": {
                        "attempts": player.clutches.situations_1v4,
                        "wins": player.clutches.wins_1v4,
                    },
                    "1v5": {
                        "attempts": player.clutches.situations_1v5,
                        "wins": player.clutches.wins_1v5,
                    },
                },
                "multi_kills": {
                    "rounds_with_2k": player.multi_kills.rounds_with_2k,
                    "rounds_with_3k": player.multi_kills.rounds_with_3k,
                    "rounds_with_4k": player.multi_kills.rounds_with_4k,
                    "rounds_with_5k": player.multi_kills.rounds_with_5k,
                },
                "advanced": {
                    # TTD Stats
                    "ttd_median_ms": (
                        round(player.ttd_median_ms, 1) if player.ttd_median_ms else None
                    ),
                    "ttd_mean_ms": (
                        round(player.ttd_mean_ms, 1) if player.ttd_mean_ms else None
                    ),
                    "ttd_samples": len(player.ttd_values),
                    "prefire_kills": player.prefire_count,
                    # Crosshair Placement Stats
                    "cp_median_error_deg": (
                        round(player.cp_median_error_deg, 1)
                        if player.cp_median_error_deg
                        else None
                    ),
                    "cp_mean_error_deg": (
                        round(player.cp_mean_error_deg, 1)
                        if player.cp_mean_error_deg
                        else None
                    ),
                    "cp_samples": len(player.cp_values),
                },
                "utility": {
                    # Flash stats (Leetify style)
                    "flash_assists": player.utility.flash_assists,
                    "flashbangs_thrown": player.utility.flashbangs_thrown,
                    "enemies_flashed": player.utility.enemies_flashed,
                    "teammates_flashed": player.utility.teammates_flashed,
                    "enemies_flashed_per_flash": round(
                        player.utility.enemies_flashed_per_flash, 2
                    ),
                    "avg_blind_time": round(player.utility.avg_blind_time, 2),
                    # HE stats
                    "he_thrown": player.utility.he_thrown,
                    "he_damage": player.utility.he_damage,
                    "he_team_damage": player.utility.he_team_damage,
                    "he_damage_per_nade": round(
                        player.utility.he_damage_per_nade, 1
                    ),
                    # Molotov stats
                    "molotov_thrown": player.utility.molotov_thrown,
                    "molotov_damage": player.utility.molotov_damage,
                    # Ratings
                    "utility_quantity_rating": player.utility_quantity_rating,
                    "utility_quality_rating": player.utility_quality_rating,
                },
                "side_stats": {
                    # CT-side performance
                    "ct": {
                        "kills": player.ct_stats.kills,
                        "deaths": player.ct_stats.deaths,
                        "kd_ratio": player.ct_stats.kd_ratio,
                        "adr": player.ct_stats.adr,
                        "rounds_played": player.ct_stats.rounds_played,
                    },
                    # T-side performance
                    "t": {
                        "kills": player.t_stats.kills,
                        "deaths": player.t_stats.deaths,
                        "kd_ratio": player.t_stats.kd_ratio,
                        "adr": player.t_stats.adr,
                        "rounds_played": player.t_stats.rounds_played,
                    },
                },
                "mistakes": {
                    "team_kills": player.mistakes.team_kills,
                    "team_damage": player.mistakes.team_damage,
                    "teammates_flashed": player.mistakes.teammates_flashed,
                    "total_mistakes": player.mistakes.total_mistakes,
                },
                "economy": {
                    "avg_equipment_value": round(player.avg_equipment_value, 0),
                    "eco_rounds": player.eco_rounds,
                    "force_rounds": player.force_rounds,
                    "full_buy_rounds": player.full_buy_rounds,
                    "damage_per_dollar": (
                        round(player.damage_per_dollar, 4)
                        if player.damage_per_dollar
                        else 0
                    ),
                    "kills_per_dollar": (
                        round(player.kills_per_dollar, 6)
                        if player.kills_per_dollar
                        else 0
                    ),
                },
                "weapons": weapon_stats,
            }

        # Add enhanced match-level data
        result["round_timeline"] = [
            {
                "round_num": r.round_num,
                "winner": r.winner,
                "win_reason": r.win_reason,
                "ct_score": r.ct_score,
                "t_score": r.t_score,
                "first_kill": r.first_kill_player,
                "first_death": r.first_death_player,
            }
            for r in analysis.round_timeline
        ]

        result["kill_matrix"] = [
            {
                "attacker": e.attacker_name,
                "victim": e.victim_name,
                "count": e.count,
                "weapons": e.weapons,
            }
            for e in analysis.kill_matrix
        ]

        result["team_stats"] = {
            "trade_rates": analysis.team_trade_rates,
            "opening_rates": analysis.team_opening_rates,
        }

        # Position data for heatmaps (only first 500 to avoid huge response)
        result["heatmap_data"] = {
            "kill_positions": analysis.kill_positions[:500],
            "death_positions": analysis.death_positions[:500],
        }

        # AI Coaching insights
        result["coaching"] = analysis.coaching_insights

        logger.info(
            f"Analysis complete: {len(result['players'])} players, "
            f"{analysis.total_rounds} rounds"
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {e!s}"
        ) from e
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                # Ignore cleanup errors - temp file will be cleaned up by OS
                pass


@app.get("/about")
async def about() -> dict[str, Any]:
    """Information about the API and metrics."""
    return {
        "name": "OpenSight",
        "version": __version__,
        "description": (
            "Local CS2 analytics framework - "
            "Leetify/Scope.gg style professional-grade metrics"
        ),
        "metrics": {
            "basic": {
                "kills": "Total eliminations",
                "deaths": "Times eliminated",
                "assists": "Kill assists",
                "kd_ratio": "Kills divided by deaths",
                "adr": "Average Damage per Round",
                "headshot_pct": "Percentage of kills that were headshots",
            },
            "rating": {
                "hltv_rating": (
                    "HLTV 2.0 Rating - industry standard performance metric "
                    "(1.0 = average)"
                ),
                "impact_rating": (
                    "Impact component - measures round-winning contributions"
                ),
                "kast_percentage": (
                    "KAST% - rounds with Kill, Assist, Survived, or Traded"
                ),
                "aim_rating": (
                    "Leetify-style Aim Rating (0-100, 50 = average) - "
                    "based on TTD, CP, HS%"
                ),
                "utility_rating": (
                    "Leetify-style Utility Rating (0-100) - "
                    "geometric mean of quantity and quality"
                ),
                "entry_success_rate": "Percentage of opening duels won",
            },
            "advanced": {
                "ttd_median_ms": (
                    "Time to Damage (median) - "
                    "milliseconds from engagement start to damage dealt"
                ),
                "ttd_mean_ms": "Time to Damage (mean)",
                "cp_median_error_deg": (
                    "Crosshair Placement error (median) - "
                    "degrees off-target when engaging"
                ),
                "prefire_kills": (
                    "Kills where damage was dealt before/instantly "
                    "upon visibility (prediction shots)"
                ),
            },
            "duels": {
                "opening_wins": "First kills of the round won",
                "opening_losses": "First deaths of the round",
                "kills_traded": "Times you avenged a teammate within 5 seconds",
                "deaths_traded": (
                    "Times a teammate avenged your death within 5 seconds"
                ),
            },
            "utility": {
                "flash_assists": "Kills on enemies you flashed",
                "enemies_flashed": "Total enemies blinded >1.1s",
                "teammates_flashed": "Times you blinded teammates (mistake)",
                "enemies_flashed_per_flash": (
                    "Average enemies blinded per flashbang (Leetify metric)"
                ),
                "avg_blind_time": "Average enemy blind duration per flash",
                "he_damage": "Total HE grenade damage to enemies",
                "he_damage_per_nade": "Average damage per HE grenade",
                "molotov_damage": "Total molotov/incendiary damage",
                "utility_quantity_rating": (
                    "How much utility thrown vs expected (3/round)"
                ),
                "utility_quality_rating": "Flash/HE effectiveness composite",
            },
            "side_stats": {
                "ct_kills": "Kills while playing CT side",
                "ct_deaths": "Deaths while playing CT side",
                "ct_adr": "ADR on CT side",
                "t_kills": "Kills while playing T side",
                "t_deaths": "Deaths while playing T side",
                "t_adr": "ADR on T side",
            },
            "mistakes": {
                "team_kills": "Friendly fire kills (bad)",
                "team_damage": "Total damage dealt to teammates",
                "teammates_flashed": "Times you blinded your teammates",
                "total_mistakes": "Composite mistake score",
            },
        },
        "rating_interpretation": {
            "hltv_rating": {
                "below_0.8": "Below average",
                "0.8_to_1.0": "Average",
                "1.0_to_1.2": "Above average",
                "1.2_to_1.5": "Excellent",
                "above_1.5": "Exceptional",
            },
            "aim_rating": {
                "0_to_30": "Poor",
                "30_to_45": "Below average",
                "45_to_55": "Average",
                "55_to_70": "Good",
                "70_to_100": "Excellent",
            },
            "utility_rating": {
                "0_to_20": "Rarely uses utility",
                "20_to_40": "Below average",
                "40_to_60": "Average",
                "60_to_80": "Good utility player",
                "80_to_100": "Utility specialist",
            },
        },
        "methodology": {
            "hltv_rating": (
                "Rating = 0.0073*KAST + 0.3591*KPR - 0.5329*DPR + "
                "0.2372*Impact + 0.0032*ADR + 0.1587*RMK"
            ),
            "trade_window": (
                "5.0 seconds (industry standard from Leetify/Stratbook)"
            ),
            "ttd": (
                "TTD measures reaction time from first damage to kill. "
                "Lower is better. Values under 200ms indicate fast reactions."
            ),
            "crosshair_placement": (
                "CP measures how far your crosshair was from the enemy "
                "when engaging. Lower is better. Under 5 degrees is elite."
            ),
            "aim_rating": (
                "Composite of TTD, CP, HS%, and prefire rate. "
                "50 = average, adjusted by component performance."
            ),
            "utility_rating": (
                "Geometric mean of Quantity (utility thrown) and "
                "Quality (effectiveness). Leetify methodology."
            ),
            "enemies_flashed_threshold": (
                "Only counts enemies blinded for >1.1 seconds "
                "(excludes half-blinds)."
            ),
        },
        "comparisons": {
            "leetify": (
                "Aim Rating, Utility Rating, and detailed flash stats "
                "follow Leetify methodology"
            ),
            "scope_gg": (
                "Mistakes tracking and side-based stats "
                "follow Scope.gg methodology"
            ),
            "hltv": "HLTV 2.0 Rating formula and KAST% calculation",
        },
    }


# =============================================================================
# Radar Map Endpoints
# =============================================================================

@app.get("/maps")
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


@app.get("/maps/{map_name}")
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
        raise HTTPException(
            status_code=503, detail=f"Radar module not available: {e}"
        ) from e


@app.post("/radar/transform")
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
            results.append({
                "game": {"x": x, "y": y, "z": z},
                "radar": {"x": round(radar_pos.x, 1), "y": round(radar_pos.y, 1)},
                "is_upper_level": transformer.is_upper_level(z),
            })

        return {
            "map_name": request.map_name,
            "positions": results,
        }
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Radar module not available: {e}"
        ) from e


# =============================================================================
# HLTV Integration Endpoints
# =============================================================================

@app.get("/hltv/rankings")
async def get_hltv_rankings(
    top_n: Annotated[int, Query(le=30)] = 10,
) -> dict[str, Any]:
    """Get current world team rankings (cached data)."""
    try:
        from opensight.integrations.hltv import HLTVClient
        client = HLTVClient()
        return {"rankings": client.get_world_rankings(top_n)}
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"HLTV module not available: {e}"
        ) from e


@app.get("/hltv/map/{map_name}")
async def get_hltv_map_stats(map_name: str) -> dict[str, Any]:
    """Get map statistics from HLTV data."""
    try:
        from opensight.integrations.hltv import get_map_statistics
        stats = get_map_statistics(map_name)
        if not stats:
            raise HTTPException(
                status_code=404, detail=f"No stats for map: {map_name}"
            )
        return stats
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"HLTV module not available: {e}"
        ) from e


@app.get("/hltv/player/search")
async def search_hltv_player(
    nickname: Annotated[str, Query(..., min_length=2)],
) -> dict[str, Any]:
    """Search for a player by nickname."""
    try:
        from opensight.integrations.hltv import HLTVClient
        client = HLTVClient()
        return {"results": client.search_player(nickname)}
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"HLTV module not available: {e}"
        ) from e


@app.post("/hltv/enrich")
async def enrich_analysis(
    analysis_data: Annotated[dict[str, Any], Body(...)],
) -> dict[str, Any]:
    """Enrich analysis data with HLTV information."""
    try:
        from opensight.integrations.hltv import enrich_match_analysis
        return enrich_match_analysis(analysis_data)
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"HLTV module not available: {e}"
        ) from e


# =============================================================================
# Cache Management Endpoints
# =============================================================================

@app.get("/cache/stats")
async def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    try:
        from opensight.infra.cache import get_cache_stats
        return get_cache_stats()
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Cache module not available: {e}"
        ) from e


@app.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Clear all cached analysis data."""
    try:
        from opensight.infra.cache import clear_cache
        clear_cache()
        return {"status": "ok", "message": "Cache cleared"}
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Cache module not available: {e}"
        ) from e


# =============================================================================
# Community Feedback Endpoints
# =============================================================================

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict[str, Any]:
    """Submit feedback on analysis accuracy."""
    try:
        from datetime import datetime

        from opensight.integrations.feedback import FeedbackDatabase, FeedbackEntry
        db = FeedbackDatabase()
        metadata: dict[str, Any] = (
            {"correction_value": request.correction_value}
            if request.correction_value
            else {}
        )
        feedback = FeedbackEntry(
            id=None,
            demo_hash=request.demo_hash,
            user_id=request.player_steam_id,
            rating=request.rating,
            category=request.metric_name,
            comment=request.comment or "",
            analysis_version=__version__,
            created_at=datetime.now(),
            metadata=metadata,
        )
        entry_id = db.add_feedback(feedback)
        return {"status": "ok", "feedback_id": entry_id}
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Feedback module not available: {e}"
        ) from e


@app.post("/feedback/coaching")
async def submit_coaching_feedback(
    request: CoachingFeedbackRequest,
) -> dict[str, Any]:
    """Submit feedback on coaching insights."""
    try:
        from datetime import datetime

        from opensight.integrations.feedback import CoachingFeedback, FeedbackDatabase
        db = FeedbackDatabase()
        feedback = CoachingFeedback(
            id=None,
            demo_hash=request.demo_hash,
            player_steam_id="",  # Not provided in request
            insight_category="coaching",
            insight_message=request.insight_id,
            was_helpful=request.was_helpful,
            user_correction=request.correction,
            created_at=datetime.now(),
        )
        entry_id = db.add_coaching_feedback(feedback)
        return {"status": "ok", "feedback_id": entry_id}
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Feedback module not available: {e}"
        ) from e


@app.get("/feedback/stats")
async def get_feedback_stats() -> Any:
    """Get feedback statistics for model improvement."""
    try:
        from opensight.integrations.feedback import FeedbackDatabase
        db = FeedbackDatabase()
        return db.get_stats()
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Feedback module not available: {e}"
        ) from e


# =============================================================================
# Parallel Batch Analysis Endpoints
# =============================================================================

@app.get("/parallel/status")
async def get_parallel_status() -> dict[str, Any]:
    """Get parallel processing capabilities."""
    try:
        import multiprocessing

        from opensight.infra.parallel import DEFAULT_WORKERS, MAX_WORKERS

        return {
            "available": True,
            "cpu_count": multiprocessing.cpu_count(),
            "default_workers": DEFAULT_WORKERS,
            "max_workers": MAX_WORKERS,
        }
    except ImportError as e:
        return {
            "available": False,
            "error": str(e),
        }


# =============================================================================
# 2D Replay Data Endpoints
# =============================================================================

@app.post("/replay/generate")
async def generate_replay_data(
    file: Annotated[UploadFile, File(...)],
    sample_rate: Annotated[
        int, Query(ge=1, le=128, description="Extract every Nth tick")
    ] = 16,
) -> dict[str, Any]:
    """
    Generate 2D replay data from a demo file.

    This extracts player positions and game state at regular intervals
    for use in 2D replay visualization.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="File must be a .dem or .dem.gz file"
        )

    try:
        from opensight.core.parser import DemoParser
        from opensight.visualization.radar import CoordinateTransformer
        from opensight.visualization.replay import ReplayGenerator
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Replay module not available: {e}"
        ) from e

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

        # Parse demo
        parser = DemoParser(tmp_path)
        data = parser.parse()

        # Generate replay data
        generator = ReplayGenerator(data, sample_rate=sample_rate)
        replay = generator.generate_full_replay()

        # Get coordinate transformer for radar positions
        transformer = CoordinateTransformer(data.map_name)

        # Collect all frames from all rounds
        all_replay_frames = [
            frame for r in replay.rounds for frame in r.frames
        ]

        # Calculate total ticks from rounds
        total_ticks = 0
        if replay.rounds:
            total_ticks = (
                replay.rounds[-1].end_tick - replay.rounds[0].start_tick
            )

        # Convert to response format (limit frames to prevent huge response)
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
                frame_data["players"].append({
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
                })

            if frame.bomb:
                bomb_pos = transformer.game_to_radar(
                    frame.bomb.x, frame.bomb.y, frame.bomb.z
                )
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

            frames.append(frame_data)

        return {
            "map_name": replay.map_name,
            "total_ticks": total_ticks,
            "tick_rate": replay.tick_rate,
            "sample_rate": sample_rate,
            "total_frames": len(all_replay_frames),
            "frames_returned": len(frames),
            "rounds": [
                {
                    "round_num": r.round_num,
                    "start_tick": r.start_tick,
                    "end_tick": r.end_tick,
                    "winner": r.winner,
                }
                for r in replay.rounds
            ],
            "frames": frames,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Replay generation failed")
        raise HTTPException(
            status_code=500, detail=f"Replay generation failed: {e!s}"
        ) from e
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
