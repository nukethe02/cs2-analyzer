"""
OpenSight Web API

FastAPI application for CS2 demo analysis.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenSight API",
    description="CS2 demo analyzer - professional-grade metrics",
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
    """Decode a CS2 share code."""
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

    Returns player stats including kills, deaths, assists, ADR, headshot %.
    """
    if not file.filename or not file.filename.endswith(".dem"):
        raise HTTPException(status_code=400, detail="File must be a .dem file")

    try:
        from opensight.parser import DemoParser
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

        logger.info(f"Analyzing demo: {file.filename} ({len(content)} bytes)")

        # Parse the demo
        parser = DemoParser(tmp_path)
        data = parser.parse()

        # Build response
        result = {
            "demo_info": {
                "map": data.map_name,
                "duration_seconds": round(data.duration_seconds, 1),
                "duration_minutes": round(data.duration_seconds / 60, 1),
                "rounds": data.num_rounds,
                "player_count": len(data.player_stats),
            },
            "players": {}
        }

        # Add player stats
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

        logger.info(f"Analysis complete: {len(result['players'])} players found")
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
