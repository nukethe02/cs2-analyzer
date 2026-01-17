"""
OpenSight Web API

Simple FastAPI application exposing OpenSight functionality for web deployment.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# Version without importing heavy dependencies
__version__ = "0.1.0"

app = FastAPI(
    title="OpenSight API",
    description="Local CS2 analytics framework - professional-grade metrics",
    version=__version__,
)


class ShareCodeRequest(BaseModel):
    code: str


class ShareCodeResponse(BaseModel):
    match_id: int
    outcome_id: int
    token: int


class HealthResponse(BaseModel):
    status: str
    version: str


# Get the static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>OpenSight</h1><p>Web interface not found.</p>", status_code=200)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.post("/decode", response_model=ShareCodeResponse)
async def decode_share_code(request: ShareCodeRequest):
    """Decode a CS2 share code."""
    try:
        # Lazy import
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
async def analyze_demo(
    file: UploadFile = File(...),
    player_filter: Optional[str] = None,
):
    """
    Analyze an uploaded demo file.

    Returns engagement metrics, TTD, and crosshair placement data.
    """
    if not file.filename.endswith(".dem"):
        raise HTTPException(status_code=400, detail="File must be a .dem file")

    # Lazy imports for heavy dependencies
    try:
        from opensight.parser import DemoParser
        from opensight.metrics import calculate_engagement_metrics, calculate_ttd, calculate_crosshair_placement
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Demo analysis not available on this server. Missing dependency: {str(e)}"
        )

    # Save uploaded file temporarily
    tmp_path = None
    try:
        with NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Parse the demo
        parser = DemoParser(tmp_path)
        data = parser.parse()

        # Filter by player if specified
        steam_id = None
        if player_filter:
            try:
                steam_id = int(player_filter)
            except ValueError:
                for sid, name in data.player_names.items():
                    if player_filter.lower() in name.lower():
                        steam_id = sid
                        break

        # Calculate metrics
        engagement = calculate_engagement_metrics(data, steam_id)
        ttd = calculate_ttd(data, steam_id)
        cp = calculate_crosshair_placement(data, steam_id)

        # Build response
        result = {
            "demo_info": {
                "map": data.map_name,
                "duration_seconds": data.duration_seconds,
                "tick_rate": data.tick_rate,
                "player_count": len(data.player_names),
            },
            "players": {}
        }

        for sid, name in data.player_names.items():
            player_data = {
                "name": name,
                "team": data.teams.get(sid, "Unknown"),
            }

            if sid in engagement:
                m = engagement[sid]
                player_data["stats"] = {
                    "kills": m.total_kills,
                    "deaths": m.total_deaths,
                    "headshot_pct": round(m.headshot_percentage, 1),
                    "damage_per_round": round(m.damage_per_round, 1),
                }

            if sid in ttd:
                t = ttd[sid]
                player_data["ttd"] = {
                    "mean_ms": round(t.mean_ttd_ms, 0),
                    "median_ms": round(t.median_ttd_ms, 0),
                    "engagements": t.engagement_count,
                }

            if sid in cp:
                c = cp[sid]
                player_data["crosshair_placement"] = {
                    "mean_angle": round(c.mean_angle_deg, 1),
                    "score": round(c.placement_score, 1),
                    "samples": c.sample_count,
                }

            result["players"][str(sid)] = player_data

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temp file
        if tmp_path:
            try:
                tmp_path.unlink()
            except:
                pass
