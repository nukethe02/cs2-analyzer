"""
Scouting Engine API Routes — Multi-demo opponent analysis.

Temporary location: will be moved to src/opensight/api/routes_scouting.py after merge.

Endpoints:
    POST   /api/scouting/session                       — Create session
    POST   /api/scouting/session/{id}/add-demo          — Add demo to session
    POST   /api/scouting/session/{id}/report             — Generate scouting report
    GET    /api/scouting/session/{id}                    — Get session state
    DELETE /api/scouting/session/{id}                    — Delete session
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scouting"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCOUTING_SESSION_TTL = 3600  # 1 hour
MAX_SCOUTING_SESSIONS = 50
MAX_DEMOS_PER_SESSION = 10
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = (".dem", ".dem.gz")

# ---------------------------------------------------------------------------
# In-memory session storage
# ---------------------------------------------------------------------------
_scouting_sessions: dict[str, dict] = {}
_scouting_sessions_lock = threading.Lock()


def _cleanup_expired_scouting_sessions() -> None:
    """Remove expired scouting sessions."""
    now = time.time()
    with _scouting_sessions_lock:
        expired = [
            sid
            for sid, sess in _scouting_sessions.items()
            if now - sess["created_at"] > SCOUTING_SESSION_TTL
        ]
        for sid in expired:
            del _scouting_sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired scouting sessions")


def _enforce_max_scouting_sessions() -> None:
    """Remove oldest sessions if max limit exceeded."""
    with _scouting_sessions_lock:
        if len(_scouting_sessions) >= MAX_SCOUTING_SESSIONS:
            sorted_sessions = sorted(_scouting_sessions.items(), key=lambda x: x[1]["created_at"])
            to_remove = len(_scouting_sessions) - MAX_SCOUTING_SESSIONS + 1
            for sid, _ in sorted_sessions[:to_remove]:
                del _scouting_sessions[sid]
            logger.info(f"Removed {to_remove} oldest scouting sessions")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ScoutingReportRequest(BaseModel):
    """Request body for generating scouting report."""

    opponent_steamids: list[int] = Field(..., description="Steam IDs of opponent players to scout")
    team_name: str = Field(default="Opponent", description="Name of the opponent team")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/api/scouting/session")
async def create_scouting_session(request: Request) -> dict[str, Any]:
    """
    Create a new scouting session for multi-demo opponent analysis.

    Returns:
        session_id: Unique ID for this scouting session
    """
    _cleanup_expired_scouting_sessions()
    _enforce_max_scouting_sessions()

    session_id = str(uuid.uuid4())

    try:
        from opensight.scouting import ScoutingEngine
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Scouting engine not available. Check installation.",
        )

    with _scouting_sessions_lock:
        _scouting_sessions[session_id] = {
            "created_at": time.time(),
            "engine": ScoutingEngine(),
            "demos": [],
            "player_list": [],
        }

    logger.info(f"Created scouting session: {session_id}")
    return {"session_id": session_id}


@router.post("/api/scouting/session/{session_id}/add-demo")
async def add_demo_to_scouting_session(
    request: Request,
    session_id: str,
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """
    Add a demo to an existing scouting session.

    Parses the demo and adds it to the scouting engine for aggregation.

    Returns:
        demo_index: Index of this demo in the session
        map_name: Map name from the demo
        players: List of players in the demo for team selection
    """
    # Validate session exists
    with _scouting_sessions_lock:
        session = _scouting_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Scouting session not found")

        if len(session["demos"]) >= MAX_DEMOS_PER_SESSION:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_DEMOS_PER_SESSION} demos per session",
            )

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be .dem or .dem.gz")

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum {MAX_FILE_SIZE_MB}MB")

    # Parse demo
    suffix = ".dem.gz" if file.filename.endswith(".gz") else ".dem"
    temp_file = None
    try:
        temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.flush()
        temp_path = Path(temp_file.name)
        temp_file.close()

        # Parse demo
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.core.parser import DemoParser

        parser = DemoParser()
        demo_data = parser.parse(temp_path)
        if not demo_data:
            raise HTTPException(status_code=400, detail="Failed to parse demo file")

        # Analyze demo
        analyzer = DemoAnalyzer()
        analysis = analyzer.analyze(demo_data)
        if not analysis:
            raise HTTPException(status_code=400, detail="Failed to analyze demo")

        # Add to scouting engine
        with _scouting_sessions_lock:
            session = _scouting_sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session expired")

            engine = session["engine"]
            demo_info = engine.add_demo(demo_data, analysis)

            # Store demo reference
            session["demos"].append(
                {
                    "filename": file.filename,
                    "map_name": demo_data.map_name,
                    "rounds": analysis.total_rounds,
                }
            )

            # Merge player list (dedupe by steamid)
            existing_steamids = {p["steamid"] for p in session["player_list"]}
            for player in demo_info["players"]:
                if player["steamid"] not in existing_steamids:
                    session["player_list"].append(player)
                    existing_steamids.add(player["steamid"])

        logger.info(f"Added demo to session {session_id}: {file.filename}")

        return {
            "demo_index": demo_info["demo_index"],
            "map_name": demo_info["map_name"],
            "total_rounds": demo_info["total_rounds"],
            "players": demo_info["players"],
            "session_demos": len(session["demos"]),
            "all_players": session["player_list"],
        }

    finally:
        if temp_file:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass


@router.post("/api/scouting/session/{session_id}/report")
async def generate_scouting_report(
    request: Request,
    session_id: str,
    body: ScoutingReportRequest,
) -> dict[str, Any]:
    """
    Generate a scouting report for the specified opponent players.

    Aggregates data across all demos in the session and generates:
    - Player profiles with playstyle classification
    - Team tendencies by map
    - Anti-strat recommendations

    Returns:
        Complete TeamScoutReport as JSON
    """
    with _scouting_sessions_lock:
        session = _scouting_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Scouting session not found")

        engine = session["engine"]

    if not body.opponent_steamids:
        raise HTTPException(status_code=400, detail="No opponent players specified")

    try:
        # Set opponent team
        engine.set_opponent_team(body.opponent_steamids, body.team_name)

        # Generate report
        report = engine.generate_report()

        logger.info(
            f"Generated scouting report for session {session_id}: "
            f"{len(report.players)} players, {report.demos_analyzed} demos"
        )

        return {
            "success": True,
            "report": report.to_dict(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to generate scouting report")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e!s}") from e


@router.get("/api/scouting/session/{session_id}")
async def get_scouting_session(request: Request, session_id: str) -> dict[str, Any]:
    """
    Get current state of a scouting session.

    Returns:
        Session info including demos loaded and player list
    """
    with _scouting_sessions_lock:
        session = _scouting_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Scouting session not found")

        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "demos": session["demos"],
            "demo_count": len(session["demos"]),
            "players": session["player_list"],
            "maps_included": list(session["engine"].maps_included),
        }


@router.delete("/api/scouting/session/{session_id}")
async def delete_scouting_session(request: Request, session_id: str) -> dict[str, Any]:
    """
    Delete a scouting session and free resources.
    """
    with _scouting_sessions_lock:
        if session_id in _scouting_sessions:
            del _scouting_sessions[session_id]
            logger.info(f"Deleted scouting session: {session_id}")
            return {"success": True, "deleted": session_id}
        else:
            raise HTTPException(status_code=404, detail="Scouting session not found")
