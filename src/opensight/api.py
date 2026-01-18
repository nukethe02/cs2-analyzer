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

    Returns player stats including kills, deaths, assists, ADR, headshot %.
    """
    if not file.filename.endswith(".dem"):
        raise HTTPException(status_code=400, detail="File must be a .dem file")

    # Lazy imports for heavy dependencies
    try:
        from opensight.parser import DemoParser
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

        # Get number of rounds
        num_rounds = len(data.rounds_df) if hasattr(data, 'rounds_df') and not data.rounds_df.empty else 1

        # Build response using player_stats from parser
        result = {
            "demo_info": {
                "map": data.map_name,
                "duration_seconds": round(data.duration_seconds, 1),
                "duration_minutes": round(data.duration_seconds / 60, 1),
                "tick_rate": data.tick_rate,
                "player_count": len(data.player_names),
                "rounds": num_rounds,
            },
            "players": {}
        }

        # Use the pre-calculated player_stats
        for steam_id, stats in data.player_stats.items():
            result["players"][str(steam_id)] = {
                "name": stats["name"],
                "team": stats["team"],
                "stats": {
                    "kills": stats["kills"],
                    "deaths": stats["deaths"],
                    "assists": stats["assists"],
                    "kd_ratio": stats["kd_ratio"],
                    "headshots": stats["headshots"],
                    "headshot_pct": stats["hs_percent"],
                    "total_damage": stats["total_damage"],
                    "adr": stats["adr"],
                }
            }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temp file
        if tmp_path:
            try:
                tmp_path.unlink()
            except:
                pass
