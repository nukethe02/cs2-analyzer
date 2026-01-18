"""
OpenSight Web API

FastAPI application for CS2 demo analysis with advanced metrics.
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
    description="CS2 demo analyzer - professional-grade metrics including TTD and Crosshair Placement",
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
    - Advanced metrics: TTD (Time to Damage), Crosshair Placement
    - Weapon breakdown
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
        player_analytics = analyzer.analyze()

        # Build response
        result = {
            "demo_info": {
                "map": data.map_name,
                "duration_seconds": round(data.duration_seconds, 1),
                "duration_minutes": round(data.duration_seconds / 60, 1),
                "tick_rate": data.tick_rate,
                "rounds": data.num_rounds,
                "player_count": len(data.player_stats),
                "total_kills": len(data.kills),
                "total_damage_events": len(data.damages),
            },
            "players": {}
        }

        # Add player stats with advanced metrics
        for steam_id, analytics in player_analytics.items():
            # Build weapon stats
            weapon_stats = []
            for weapon, count in sorted(analytics.weapon_kills.items(), key=lambda x: -x[1]):
                weapon_stats.append({"weapon": weapon, "kills": count})

            result["players"][str(steam_id)] = {
                "name": analytics.name,
                "team": analytics.team,
                "stats": {
                    # Basic stats
                    "kills": analytics.kills,
                    "deaths": analytics.deaths,
                    "assists": analytics.assists,
                    "kd_ratio": round(analytics.kills / max(analytics.deaths, 1), 2),
                    "headshots": data.player_stats[steam_id]["headshots"],
                    "headshot_pct": analytics.hs_percent,
                    "total_damage": data.player_stats[steam_id]["total_damage"],
                    "adr": analytics.adr,
                },
                "advanced": {
                    # TTD Stats
                    "ttd_median_ms": round(analytics.ttd_median_ms, 1) if analytics.ttd_median_ms else None,
                    "ttd_mean_ms": round(analytics.ttd_mean_ms, 1) if analytics.ttd_mean_ms else None,
                    "ttd_min_ms": round(analytics.ttd_min_ms, 1) if analytics.ttd_min_ms else None,
                    "ttd_max_ms": round(analytics.ttd_max_ms, 1) if analytics.ttd_max_ms else None,
                    "ttd_std_ms": round(analytics.ttd_std_ms, 1) if analytics.ttd_std_ms else None,
                    "ttd_samples": analytics.ttd_count,
                    "prefire_kills": analytics.prefire_count,
                    # Crosshair Placement Stats
                    "cp_median_error_deg": round(analytics.cp_median_error_deg, 1) if analytics.cp_median_error_deg else None,
                    "cp_mean_error_deg": round(analytics.cp_mean_error_deg, 1) if analytics.cp_mean_error_deg else None,
                    "cp_pitch_bias_deg": round(analytics.cp_pitch_bias_deg, 1) if analytics.cp_pitch_bias_deg else None,
                },
                "weapons": weapon_stats,
            }

        logger.info(f"Analysis complete: {len(result['players'])} players, {data.num_rounds} rounds")
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
            except OSError:
                # Ignore cleanup errors - temp file will be cleaned up by OS
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
            "advanced": {
                "ttd_median_ms": "Time to Damage (median) - milliseconds from engagement start to damage dealt",
                "ttd_mean_ms": "Time to Damage (mean)",
                "cp_median_error_deg": "Crosshair Placement error (median) - degrees off-target when engaging",
                "cp_pitch_bias_deg": "Vertical aim bias - negative means aiming too low",
                "prefire_kills": "Kills where damage was dealt before/instantly upon visibility (prediction shots)",
            }
        },
        "methodology": {
            "ttd": "TTD measures reaction time from first damage to kill. Lower is better. Values under 200ms indicate fast reactions.",
            "crosshair_placement": "CP measures how far your crosshair was from the enemy when engaging. Lower is better. Under 5 degrees is elite-level.",
        }
    }
