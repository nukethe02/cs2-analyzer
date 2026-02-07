"""
Analysis route handlers.

Endpoints:
- POST /analyze — demo upload and background analysis
- GET /analyze/{job_id} — check job status
- GET /analyze/{job_id}/download — download results
- POST /api/tactical-analysis/{job_id} — Claude-powered tactical analysis
- POST /api/strat-steal/{job_id} — strat-stealing report
- POST /api/self-review/{job_id} — team self-review report
- GET /jobs — list all jobs
"""

import logging
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from opensight.api.shared import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    RATE_LIMIT_API,
    RATE_LIMIT_UPLOAD,
    JobStatus,
    rate_limit,
    validate_demo_id,
    validate_job_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


def _get_job_store():
    """Lazy import to avoid circular dependency.

    NOTE(DRY-intentional): This identical pattern exists in all 5 route modules
    (routes_analysis, routes_heatmap, routes_match, routes_export, routes_misc).
    The duplication is INTENTIONAL — each module needs its own lazy import to
    break the circular dependency between route modules and api/__init__.py.
    Do NOT consolidate into shared.py; that would re-introduce the circular import.
    """
    # To use persistent jobs (survives restart):
    # from opensight.infra.job_store import PersistentJobStore
    # job_store = PersistentJobStore()
    from opensight.api import job_store

    return job_store


@router.post("/analyze", status_code=202)
@rate_limit(RATE_LIMIT_UPLOAD)
async def analyze_demo(request: Request, file: UploadFile = File(...)):
    """
    Submit a CS2 demo file for analysis.

    Returns a job ID immediately (202 Accepted). Poll GET /analyze/{job_id}
    to check status and retrieve results when complete.

    Accepts .dem and .dem.gz files up to 500MB.
    """
    job_store = _get_job_store()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a .dem or .dem.gz file. Got: {file.filename}",
        )

    try:
        # Write upload to disk in chunks to avoid loading entire 500MB demo into memory
        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        try:
            tmp = NamedTemporaryFile(suffix=".dem", delete=False)
            file_size_bytes = 0
            try:
                while chunk := await file.read(CHUNK_SIZE):
                    file_size_bytes += len(chunk)
                    if file_size_bytes > MAX_FILE_SIZE_BYTES:
                        tmp.close()
                        Path(tmp.name).unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large: {file_size_bytes / (1024 * 1024):.1f}MB",
                        )
                    tmp.write(chunk)
                tmp.flush()
                demo_path = Path(tmp.name)
            finally:
                tmp.close()
        except HTTPException:
            raise
        except Exception:
            logger.exception("Failed to write uploaded demo to temp file")
            raise

        if file_size_bytes == 0:
            demo_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        job = job_store.create_job(file.filename, file_size_bytes)

        try:

            def _process_job(jid: str, demo_path: Path):
                try:
                    job_store.set_status(jid, JobStatus.PROCESSING)
                    from opensight.infra.cache import analyze_with_cache

                    result = analyze_with_cache(demo_path)

                    timeline = result.get("round_timeline", [])
                    logger.debug(f"Analysis complete: {len(timeline)} rounds in timeline")

                    j = job_store.get_job(jid)
                    if j:
                        j.result = result
                        job_store.set_status(jid, JobStatus.COMPLETED)
                except Exception as ex:
                    logger.exception("Job processing failed")
                    j = job_store.get_job(jid)
                    if j:
                        error_type = type(ex).__name__
                        if "parse" in str(ex).lower() or "demo" in str(ex).lower():
                            safe_error = "Demo file could not be parsed. The file may be corrupted or in an unsupported format."
                        elif "memory" in str(ex).lower():
                            safe_error = "Analysis failed due to resource constraints. Try a smaller demo file."
                        else:
                            safe_error = f"Analysis failed ({error_type}). Please try again or contact support."
                        j.result = {"error": safe_error}
                        job_store.set_status(jid, JobStatus.FAILED)
                finally:
                    try:
                        demo_path.unlink(missing_ok=True)
                    except Exception:
                        logger.warning("Failed to clean up temp demo file: %s", demo_path)

            threading.Thread(target=_process_job, args=(job.job_id, demo_path), daemon=True).start()
        except Exception:
            logger.exception("Failed to start analysis thread for job %s", job.job_id)
            job_store.set_status(job.job_id, JobStatus.FAILED)

        base = f"/analyze/{job.job_id}"
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job.job_id,
                "status": job.status,
                "status_url": base,
                "download_url": f"{base}/download",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to queue analysis job")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {e!s}") from e


@router.get("/analyze/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Get the status of an analysis job."""
    # TODO(DRY): extract "validate + get_job + 404 check" to shared utility
    # in shared.py — this pattern repeats 6+ times across route modules.
    # See also _get_completed_result() in routes_heatmap.py / routes_export.py
    # which partially consolidates this for the "completed job + result" variant.
    validate_job_id(job_id)
    job_store = _get_job_store()

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "filename": job.filename,
        "size": job.size,
    }


@router.get("/analyze/{job_id}/download")
async def download_job_result(job_id: str):
    """Download the results of a completed analysis job."""
    validate_job_id(job_id)
    job_store = _get_job_store()

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")

    result = job.result or {}
    logger.debug(f"Returning analysis result for job {job_id}")

    # --- AI COACHING INTEGRATION ---
    # NOTE: AI summaries are cached in the result dict after first generation.
    # Subsequent downloads of the same job reuse the cached summary.
    if "ai_summary" not in result:
        players = result.get("players", {})
        if players and len(players) > 0:
            try:
                from opensight.ai.llm_client import generate_match_summary

                hero_player = players[0] if isinstance(players, list) else list(players.values())[0]

                player_stats = {
                    "kills": hero_player.get("stats", {}).get("kills", 0),
                    "deaths": hero_player.get("stats", {}).get("deaths", 0),
                    "assists": hero_player.get("stats", {}).get("assists", 0),
                    "hltv_rating": hero_player.get("rating", {}).get("hltv_rating", 0.0),
                    "adr": hero_player.get("stats", {}).get("adr", 0.0),
                    "headshot_pct": hero_player.get("stats", {}).get("headshot_pct", 0.0),
                    "kast_percentage": hero_player.get("rating", {}).get("kast_percentage", 0.0),
                    "ttd_median_ms": hero_player.get("advanced", {}).get("ttd_median_ms", 0),
                    "cp_median_error_deg": hero_player.get("advanced", {}).get(
                        "cp_median_error_deg", 0.0
                    ),
                    "entry_kills": hero_player.get("entry", {}).get("entry_kills", 0),
                    "entry_deaths": hero_player.get("entry", {}).get("entry_deaths", 0),
                    "trade_kill_success": hero_player.get("trades", {}).get(
                        "trade_kill_success", 0
                    ),
                    "trade_kill_opportunities": hero_player.get("trades", {}).get(
                        "trade_kill_opportunities", 0
                    ),
                    "clutch_wins": hero_player.get("clutches", {}).get("clutch_wins", 0),
                    "clutch_attempts": hero_player.get("clutches", {}).get("clutch_wins", 0)
                    + hero_player.get("clutches", {}).get("clutch_losses", 0),
                }

                ai_insight_text = generate_match_summary(player_stats)
                result["ai_summary"] = ai_insight_text
                logger.info(f"AI summary generated for job {job_id}")

            except Exception as e:
                logger.warning(f"AI coaching unavailable: {e}")
                result["ai_summary"] = "Tactical Analysis unavailable (Check ANTHROPIC_API_KEY)."

    return JSONResponse(content=result)


# =============================================================================
# TACTICAL AI ANALYSIS ENDPOINTS
# =============================================================================


@router.post("/api/tactical-analysis/{job_id}")
@rate_limit(RATE_LIMIT_API)
async def tactical_analysis(job_id: str, request: Request) -> dict[str, Any]:
    """Generate Claude-powered tactical analysis for a completed demo."""
    validate_job_id(job_id)
    job_store = _get_job_store()

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")
    if not job.result:
        raise HTTPException(status_code=400, detail="No analysis result available")

    try:
        body = await request.json()
    except Exception:
        logger.debug("No JSON body provided for tactical analysis, using defaults")
        body = {}

    analysis_type = body.get("type", "overview")
    focus = body.get("focus", None)

    valid_types = ["overview", "strat-steal", "self-review", "scout", "quick"]
    if analysis_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid analysis type. Must be one of: {valid_types}",
        )

    try:
        from opensight.ai.llm_client import get_tactical_ai_client

        ai = get_tactical_ai_client()
        analysis = ai.analyze(
            match_data=job.result,
            analysis_type=analysis_type,
            focus=focus,
        )
        logger.info(f"Tactical analysis generated for job {job_id}: type={analysis_type}")
        return {"analysis": analysis, "type": analysis_type}

    except ValueError as e:
        logger.warning(f"Tactical analysis unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Tactical AI not configured. Set ANTHROPIC_API_KEY environment variable.",
        ) from e
    except ImportError as e:
        logger.warning(f"Anthropic library not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Anthropic library not installed. Install with: pip install anthropic",
        ) from e
    except Exception as e:
        logger.exception(f"Tactical analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Tactical analysis failed: {type(e).__name__}",
        ) from e


@router.post("/api/strat-steal/{job_id}")
@rate_limit(RATE_LIMIT_API)
async def steal_strats(job_id: str, request: Request) -> dict[str, Any]:
    """Generate a strat-stealing report from a parsed demo."""
    validate_job_id(job_id)
    job_store = _get_job_store()

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")
    if not job.result:
        raise HTTPException(status_code=400, detail="No analysis result available")

    try:
        body = await request.json()
    except Exception:
        logger.debug("No JSON body provided for strat-steal, using defaults")
        body = {}

    team_focus = body.get("team", None)
    side_focus = body.get("side", None)

    if side_focus and side_focus not in ["T", "CT"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid side. Must be 'T', 'CT', or omitted for both.",
        )

    try:
        from opensight.ai.strat_engine import get_strat_engine

        engine = get_strat_engine()
        report = engine.generate_strat_report(
            match_data=job.result,
            team_focus=team_focus,
            side_focus=side_focus,
        )

        analysis = engine.analyze(job.result, team_focus, side_focus)
        patterns_summary = {
            "defaults_detected": len(analysis.defaults),
            "executes_detected": len(analysis.executes),
            "economy_patterns": len(analysis.economy),
            "player_tendencies": len(analysis.tendencies),
        }

        logger.info(
            f"Strat report generated for job {job_id}: team={team_focus}, side={side_focus}"
        )
        return {
            "report": report,
            "patterns": patterns_summary,
            "team": analysis.team_name,
            "map": analysis.map_name,
        }

    except ValueError as e:
        logger.warning(f"Strat stealing unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Tactical AI not configured. Set ANTHROPIC_API_KEY environment variable.",
        ) from e
    except ImportError as e:
        logger.warning(f"Required library not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Required library not installed. Check installation.",
        ) from e
    except Exception as e:
        logger.exception(f"Strat stealing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Strat analysis failed: {type(e).__name__}",
        ) from e


# =============================================================================
# SELF-REVIEW ENDPOINTS
# =============================================================================


@router.post("/api/self-review/{job_id}")
async def self_review_analysis(
    job_id: str,
    request: Request,
) -> dict[str, Any]:
    """Generate team self-review report analyzing mistakes and improvement areas."""
    if not validate_demo_id(job_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid job_id format. Use alphanumeric, dash, underscore only.",
        )

    job_store = _get_job_store()

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed yet. Status: {job.status}",
        )

    if not job.result:
        raise HTTPException(
            status_code=500,
            detail=f"Job {job_id} completed but has no result data",
        )

    try:
        our_team: str | None = None
        try:
            body = await request.json()
            our_team = body.get("our_team")
        except Exception:
            logger.debug("No JSON body provided for self-review, using defaults")

        from opensight.ai.self_review import get_self_review_engine

        engine = get_self_review_engine()
        report = engine.generate_review_report(job.result, our_team=our_team)

        logger.info(f"Generated self-review report for job {job_id}")
        return {
            "job_id": job_id,
            "report": report,
            "our_team": our_team or "auto-detected",
            "status": "success",
        }

    except Exception as e:
        logger.exception(f"Self-review analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Self-review analysis failed: {type(e).__name__}: {str(e)}",
        ) from e


@router.get("/jobs")
async def list_jobs() -> dict[str, Any]:
    job_store = _get_job_store()
    jobs = job_store.list_jobs()
    return {
        "jobs": [{"job_id": j.job_id, "status": j.status, "filename": j.filename} for j in jobs]
    }
