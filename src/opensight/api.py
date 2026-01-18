"""
OpenSight Web API

FastAPI application for CS2 demo analysis with professional-grade metrics.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

__version__ = "0.2.0"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenSight API",
    description="CS2 demo analyzer - professional-grade metrics including HLTV 2.0 Rating, KAST%, TTD, and Crosshair Placement",
    version=__version__,
)


class ShareCodeRequest(BaseModel):
    code: str


class ShareCodeResponse(BaseModel):
    match_id: int
    outcome_id: int
    token: int


# Get the static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>OpenSight</h1><p>Web interface not found.</p>", status_code=200)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.post("/decode", response_model=ShareCodeResponse)
async def decode_share_code(request: ShareCodeRequest):
    """Decode a CS2 share code to extract match metadata."""
    try:
        from opensight.sharecode import decode_sharecode
        info = decode_sharecode(request.code)
        return {
            "match_id": info.match_id,
            "outcome_id": info.outcome_id,
            "token": info.token,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Module not available: {str(e)}")


@app.post("/analyze")
async def analyze_demo(file: UploadFile = File(...)):
    """
    Analyze an uploaded CS2 demo file.

    Returns comprehensive player stats including:
    - Basic stats: Kills, Deaths, Assists, K/D, ADR, HS%
    - Professional metrics: HLTV 2.0 Rating, KAST%, Impact
    - Advanced metrics: TTD (Time to Damage), Crosshair Placement
    - Detailed breakdowns: Opening duels, trades, clutches, multi-kills
    """
    if not file.filename or not file.filename.endswith(".dem"):
        raise HTTPException(status_code=400, detail="File must be a .dem file")

    try:
        from opensight.parser import DemoParser
        from opensight.analytics import DemoAnalyzer
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Demo analysis not available. Missing: {str(e)}"
        )

    tmp_path = None
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"Analyzing demo: {file.filename} ({file_size_mb:.1f} MB)")

        # Parse the demo
        parser = DemoParser(tmp_path)
        data = parser.parse()

        # Run advanced analytics
        analyzer = DemoAnalyzer(data)
        analysis = analyzer.analyze()

        # Build response
        result = {
            "demo_info": {
                "map": analysis.map_name,
                "duration_seconds": round(data.duration_seconds, 1),
                "duration_minutes": round(data.duration_seconds / 60, 1),
                "tick_rate": data.tick_rate,
                "rounds": analysis.total_rounds,
                "score": f"{analysis.team1_score} - {analysis.team2_score}",
                "player_count": len(analysis.players),
                "total_kills": len(data.kills),
                "total_damage_events": len(data.damages),
            },
            "mvp": None,
            "players": {}
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
                    "1v1": {"attempts": player.clutches.situations_1v1, "wins": player.clutches.wins_1v1},
                    "1v2": {"attempts": player.clutches.situations_1v2, "wins": player.clutches.wins_1v2},
                    "1v3": {"attempts": player.clutches.situations_1v3, "wins": player.clutches.wins_1v3},
                    "1v4": {"attempts": player.clutches.situations_1v4, "wins": player.clutches.wins_1v4},
                    "1v5": {"attempts": player.clutches.situations_1v5, "wins": player.clutches.wins_1v5},
                },
                "multi_kills": {
                    "rounds_with_2k": player.multi_kills.rounds_with_2k,
                    "rounds_with_3k": player.multi_kills.rounds_with_3k,
                    "rounds_with_4k": player.multi_kills.rounds_with_4k,
                    "rounds_with_5k": player.multi_kills.rounds_with_5k,
                },
                "advanced": {
                    # TTD Stats
                    "ttd_median_ms": round(player.ttd_median_ms, 1) if player.ttd_median_ms else None,
                    "ttd_mean_ms": round(player.ttd_mean_ms, 1) if player.ttd_mean_ms else None,
                    "ttd_samples": len(player.ttd_values),
                    "prefire_kills": player.prefire_count,
                    # Crosshair Placement Stats
                    "cp_median_error_deg": round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None,
                    "cp_mean_error_deg": round(player.cp_mean_error_deg, 1) if player.cp_mean_error_deg else None,
                    "cp_samples": len(player.cp_values),
                },
                "utility": {
                    "flash_assists": player.utility.flash_assists,
                },
                "weapons": weapon_stats,
            }

        logger.info(f"Analysis complete: {len(result['players'])} players, {analysis.total_rounds} rounds")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass


@app.get("/about")
async def about():
    """Information about the API and metrics."""
    return {
        "name": "OpenSight",
        "version": __version__,
        "description": "Local CS2 analytics framework - professional-grade metrics",
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
                "hltv_rating": "HLTV 2.0 Rating - industry standard performance metric (1.0 = average)",
                "impact_rating": "Impact component - measures round-winning contributions",
                "kast_percentage": "KAST% - rounds with Kill, Assist, Survived, or Traded",
            },
            "advanced": {
                "ttd_median_ms": "Time to Damage (median) - milliseconds from engagement start to damage dealt",
                "ttd_mean_ms": "Time to Damage (mean)",
                "cp_median_error_deg": "Crosshair Placement error (median) - degrees off-target when engaging",
                "prefire_kills": "Kills where damage was dealt before/instantly upon visibility (prediction shots)",
            },
            "duels": {
                "opening_wins": "First kills of the round won",
                "opening_losses": "First deaths of the round",
                "kills_traded": "Times you avenged a teammate within 5 seconds",
                "deaths_traded": "Times a teammate avenged your death within 5 seconds",
            },
        },
        "rating_interpretation": {
            "below_0.8": "Below average",
            "0.8_to_1.0": "Average",
            "1.0_to_1.2": "Above average",
            "1.2_to_1.5": "Excellent",
            "above_1.5": "Exceptional",
        },
        "methodology": {
            "hltv_rating": "Rating = 0.0073*KAST + 0.3591*KPR - 0.5329*DPR + 0.2372*Impact + 0.0032*ADR + 0.1587*RMK",
            "trade_window": "5.0 seconds (industry standard from Leetify/Stratbook)",
            "ttd": "TTD measures reaction time from first damage to kill. Lower is better. Values under 200ms indicate fast reactions.",
            "crosshair_placement": "CP measures how far your crosshair was from the enemy when engaging. Lower is better. Under 5 degrees is elite-level.",
        }
    }
