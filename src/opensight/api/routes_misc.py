"""
Miscellaneous route handlers.

Endpoints:
- GET /health — health check
- GET /readiness — container orchestration readiness
- GET /about — API documentation
- POST /decode — decode CS2 share code
- GET /cache/stats — cache statistics
- POST /cache/clear — clear analysis cache
- POST /feedback — submit metric feedback
- POST /feedback/coaching — submit coaching feedback
- GET /feedback/stats — get feedback statistics
- GET /parallel/status — parallel processing capabilities
- POST /api/scouting/session — create scouting session
- POST /api/scouting/session/{session_id}/add-demo — add demo to scouting session
- POST /api/scouting/session/{session_id}/report — generate scouting report
- GET /api/scouting/session/{session_id} — get scouting session state
- DELETE /api/scouting/session/{session_id} — delete scouting session
"""

import logging
import threading
import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from opensight.api.shared import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    RATE_LIMIT_API,
    RATE_LIMIT_UPLOAD,
    CoachingFeedbackRequest,
    FeedbackRequest,
    ScoutingReportRequest,
    ShareCodeRequest,
    ShareCodeResponse,
    __version__,
    _get_job_store,
    rate_limit,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["misc"])


def _get_sharecode_cache():
    """Lazy import to avoid circular dependency."""
    from opensight.api import sharecode_cache

    return sharecode_cache


# =============================================================================
# Health & Info
# =============================================================================


@router.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@router.get("/readiness")
async def readiness() -> dict[str, Any]:
    """Readiness check for container orchestration (Hugging Face Spaces)."""
    import shutil
    import tempfile

    from fastapi.responses import JSONResponse

    checks = {}

    try:
        disk = shutil.disk_usage("/tmp")
        free_mb = disk.free / (1024 * 1024)
        checks["disk_space"] = {"ok": free_mb > 100, "free_mb": round(free_mb, 1)}
    except Exception as e:
        checks["disk_space"] = {"ok": False, "error": str(e)}

    try:
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(b"test")
        checks["temp_dir"] = {"ok": True}
    except Exception as e:
        checks["temp_dir"] = {"ok": False, "error": str(e)}

    try:
        import demoparser2  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401

        checks["dependencies"] = {"ok": True}
    except ImportError as e:
        checks["dependencies"] = {"ok": False, "error": str(e)}

    all_ok = all(c.get("ok", False) for c in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={"ready": all_ok, "checks": checks, "version": __version__},
        status_code=status_code,
    )


@router.get("/about")
async def about() -> dict[str, Any]:
    """Information about the API and metrics."""
    return {
        "name": "OpenSight",
        "version": __version__,
        "description": (
            "Local CS2 analytics framework - Leetify/Scope.gg style professional-grade metrics"
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
                    "HLTV 2.0 Rating - industry standard performance metric (1.0 = average)"
                ),
                "impact_rating": ("Impact component - measures round-winning contributions"),
                "kast_percentage": ("KAST% - rounds with Kill, Assist, Survived, or Traded"),
                "aim_rating": (
                    "Leetify-style Aim Rating (0-100, 50 = average) - based on TTD, CP, HS%"
                ),
                "utility_rating": (
                    "Leetify-style Utility Rating (0-100) - geometric mean of quantity and quality"
                ),
                "entry_success_rate": "Percentage of opening duels won",
            },
            "advanced": {
                "reaction_time_median_ms": (
                    "True TTD / Reaction Time (median) - ms from seeing enemy to first damage"
                ),
                "reaction_time_mean_ms": "Reaction Time (mean)",
                "engagement_duration_median_ms": (
                    "Duel Duration (median) - ms from first damage to kill (spray/tracking)"
                ),
                "engagement_duration_mean_ms": "Duel Duration (mean)",
                "cp_median_error_deg": (
                    "Crosshair Placement error (median) - degrees off-target when engaging"
                ),
                "prefire_count": (
                    "Kills with reaction time < 100ms - pre-aimed/anticipated enemy position"
                ),
                "prefire_percentage": "Percentage of engagements that were prefires",
            },
            "detailed_accuracy": {
                "spotted_accuracy": "Accuracy when enemy is spotted (shots within damage window)",
                "spray_accuracy": "Accuracy during spray/burst fire (consecutive rapid shots)",
                "spray_headshot_rate": "Headshot rate during spray fire",
                "first_bullet_accuracy": "First bullet accuracy (crucial for duels)",
                "first_bullet_hs_rate": "First bullet headshot rate",
                "counter_strafe_rating": "Counter-strafe rating (0-100) - how well player stops before shooting",
                "movement_accuracy_penalty": "% of shots fired while moving (higher = worse)",
                "overall_accuracy": "Total accuracy (shots hit / shots fired)",
                "head_accuracy": "% of hits that landed on head",
                "hs_kill_percentage": "% of kills that were headshots",
            },
            "duels": {
                "opening_wins": "First kills of the round won",
                "opening_losses": "First deaths of the round",
                "kills_traded": "Times you avenged a teammate within 5 seconds",
                "deaths_traded": ("Times a teammate avenged your death within 5 seconds"),
            },
            "utility": {
                "flash_assists": "Kills on enemies you flashed",
                "enemies_flashed": "Total enemies blinded >1.1s",
                "teammates_flashed": "Times you blinded teammates (mistake)",
                "enemies_flashed_per_flash": (
                    "Average enemies blinded per flashbang (Leetify metric)"
                ),
                "avg_blind_time": "Average enemy blind duration per flash",
                "effective_flashes": "Unique flashbangs with >= 1 significant enemy blind",
                "avg_enemies_per_flash": "Average enemies blinded per effective flash",
                "flash_effectiveness_pct": "% of flashes that were effective",
                "times_blinded": "Times you were blinded by enemies",
                "avg_time_blinded": "Average time you were blinded (Leetify Avg Blind Time)",
                "he_damage": "Total HE grenade damage to enemies",
                "he_damage_per_nade": "Average damage per HE grenade",
                "molotov_damage": "Total molotov/incendiary damage",
                "utility_quantity_rating": ("How much utility thrown vs expected (3/round)"),
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
            "trade_window": ("5.0 seconds (industry standard from Leetify/Stratbook)"),
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
                "Only counts enemies blinded for >1.1 seconds (excludes half-blinds)."
            ),
        },
        "comparisons": {
            "leetify": (
                "Aim Rating, Utility Rating, and detailed flash stats follow Leetify methodology"
            ),
            "scope_gg": ("Mistakes tracking and side-based stats follow Scope.gg methodology"),
            "hltv": "HLTV 2.0 Rating formula and KAST% calculation",
        },
        "api_optimization": {
            "async_analysis": "Use the /analyze async job endpoints to upload and poll status",
            "caching": "Results are cached server-side to speed up repeated analyses",
        },
    }


# =============================================================================
# Decode
# =============================================================================


@router.post("/decode", response_model=ShareCodeResponse)
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
        raise HTTPException(status_code=500, detail=f"Module not available: {e!s}") from e


# =============================================================================
# Cache Management
# =============================================================================


@router.get("/cache/stats")
async def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    sharecode_cache = _get_sharecode_cache()
    job_store = _get_job_store()

    try:
        from opensight.infra.cache import get_cache_stats as infra_get_cache_stats

        stats = infra_get_cache_stats()

        stats_wrapped: dict[str, Any] = {
            "demo_cache": stats,
            "sharecode_cache": {"maxsize": getattr(sharecode_cache, "maxsize", None)},
            "job_store": {"total_jobs": len(job_store.list_jobs())},
        }

        return stats_wrapped
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Cache module not available: {e}") from e


@router.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Clear all cached analysis data."""
    sharecode_cache = _get_sharecode_cache()

    try:
        from opensight.infra.cache import clear_cache as infra_clear_cache

        infra_clear_cache()

        try:
            sharecode_cache.clear()
        except Exception as e:
            logger.warning(f"Failed to clear sharecode cache: {e}")

        return {"status": "ok", "message": "Cache cleared"}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Cache module not available: {e}") from e


# =============================================================================
# Community Feedback
# =============================================================================


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict[str, Any]:
    """Submit feedback on analysis accuracy."""
    try:
        from datetime import datetime

        from opensight.integrations.feedback import FeedbackDatabase, FeedbackEntry

        db = FeedbackDatabase()
        metadata: dict[str, Any] = (
            {"correction_value": request.correction_value} if request.correction_value else {}
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
        raise HTTPException(status_code=503, detail=f"Feedback module not available: {e}") from e


@router.post("/feedback/coaching")
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
            player_steam_id="",
            insight_category="coaching",
            insight_message=request.insight_id,
            was_helpful=request.was_helpful,
            user_correction=request.correction,
            created_at=datetime.now(),
        )
        entry_id = db.add_coaching_feedback(feedback)
        return {"status": "ok", "feedback_id": entry_id}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Feedback module not available: {e}") from e


@router.get("/feedback/stats")
async def get_feedback_stats() -> Any:
    """Get feedback statistics for model improvement."""
    try:
        from opensight.integrations.feedback import FeedbackDatabase

        db = FeedbackDatabase()
        return db.get_stats()
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Feedback module not available: {e}") from e


# =============================================================================
# Parallel Batch Analysis
# =============================================================================


@router.get("/parallel/status")
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
# Scouting Engine Endpoints
# =============================================================================

SCOUTING_SESSION_TTL = 3600
MAX_SCOUTING_SESSIONS = 50
MAX_DEMOS_PER_SESSION = 10

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


@router.post("/api/scouting/session")
@rate_limit(RATE_LIMIT_API)
async def create_scouting_session(request: Request) -> dict[str, Any]:
    """Create a new scouting session for multi-demo opponent analysis."""
    _cleanup_expired_scouting_sessions()
    _enforce_max_scouting_sessions()

    session_id = str(uuid.uuid4())

    from opensight.scouting import ScoutingEngine

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
@rate_limit(RATE_LIMIT_UPLOAD)
async def add_demo_to_scouting_session(
    request: Request,
    session_id: str,
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """Add a demo to an existing scouting session."""
    with _scouting_sessions_lock:
        session = _scouting_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Scouting session not found")

        if len(session["demos"]) >= MAX_DEMOS_PER_SESSION:
            raise HTTPException(
                status_code=400, detail=f"Maximum {MAX_DEMOS_PER_SESSION} demos per session"
            )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be .dem or .dem.gz")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum {MAX_FILE_SIZE_MB}MB")

    suffix = ".dem.gz" if file.filename.endswith(".gz") else ".dem"
    temp_file = None
    try:
        temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.flush()
        temp_path = Path(temp_file.name)
        temp_file.close()

        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.core.parser import DemoParser

        parser = DemoParser(temp_path)
        demo_data = parser.parse()
        if not demo_data:
            raise HTTPException(status_code=400, detail="Failed to parse demo file")

        analyzer = DemoAnalyzer(demo_data)
        analysis = analyzer.analyze()
        if not analysis:
            raise HTTPException(status_code=400, detail="Failed to analyze demo")

        with _scouting_sessions_lock:
            session = _scouting_sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session expired")

            engine = session["engine"]
            demo_info = engine.add_demo(demo_data, analysis)

            session["demos"].append(
                {
                    "filename": file.filename,
                    "map_name": demo_data.map_name,
                    "rounds": analysis.total_rounds,
                }
            )

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
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file.name}: {e}")


@router.post("/api/scouting/session/{session_id}/report")
@rate_limit(RATE_LIMIT_API)
async def generate_scouting_report(
    request: Request,
    session_id: str,
    body: ScoutingReportRequest,
) -> dict[str, Any]:
    """Generate a scouting report for the specified opponent players."""
    with _scouting_sessions_lock:
        session = _scouting_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Scouting session not found")

        engine = session["engine"]

    if not body.opponent_steamids:
        raise HTTPException(status_code=400, detail="No opponent players specified")

    try:
        engine.set_opponent_team(body.opponent_steamids, body.team_name)
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
@rate_limit(RATE_LIMIT_API)
async def get_scouting_session(request: Request, session_id: str) -> dict[str, Any]:
    """Get current state of a scouting session."""
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
@rate_limit(RATE_LIMIT_API)
async def delete_scouting_session(request: Request, session_id: str) -> dict[str, Any]:
    """Delete a scouting session and free resources."""
    with _scouting_sessions_lock:
        if session_id in _scouting_sessions:
            del _scouting_sessions[session_id]
            logger.info(f"Deleted scouting session: {session_id}")
            return {"success": True, "deleted": session_id}
        else:
            raise HTTPException(status_code=404, detail="Scouting session not found")
