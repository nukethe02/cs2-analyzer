"""
AI / Tactical Analysis API Routes.

Temporary location: will be moved to src/opensight/api/routes_ai.py after merge.

Endpoints:
    POST /api/tactical-analysis/{job_id}  — Claude-powered tactical analysis
    POST /api/strat-steal/{job_id}        — Strat-stealing report
    POST /api/self-review/{job_id}        — Team self-review report
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ai"])

# ---------------------------------------------------------------------------
# Validation helpers (duplicated minimally to stay standalone)
# ---------------------------------------------------------------------------
JOB_ID_PATTERN = re.compile(
    r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"
)
DEMO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_job_id(job_id: str) -> str:
    """Validate job_id UUID format."""
    if not job_id or not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id: must be a valid UUID")
    return job_id


def _validate_demo_id(demo_id: str) -> bool:
    """Validate demo_id format."""
    return bool(demo_id and DEMO_ID_PATTERN.match(demo_id))


def _get_job_store():
    """Lazy import of the global job_store from api.py."""
    from opensight.api import job_store

    return job_store


def _get_job_status_completed() -> str:
    """Return the completed status string."""
    from opensight.api import JobStatus

    return JobStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/api/tactical-analysis/{job_id}")
async def tactical_analysis(job_id: str, request: Request) -> dict[str, Any]:
    """
    Generate Claude-powered tactical analysis for a completed demo.

    Request body:
        - type: Analysis type (overview, strat-steal, self-review, scout, quick)
        - focus: Optional focus area (specific round, player, or side)

    Returns:
        - analysis: Markdown-formatted tactical report
    """
    _validate_job_id(job_id)

    job_store = _get_job_store()
    completed = _get_job_status_completed()

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != completed:
        raise HTTPException(status_code=400, detail="Job not completed")
    if not job.result:
        raise HTTPException(status_code=400, detail="No analysis result available")

    try:
        body = await request.json()
    except Exception:
        body = {}

    analysis_type = body.get("type", "overview")
    focus = body.get("focus", None)

    # Validate analysis type
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
async def steal_strats(job_id: str, request: Request) -> dict[str, Any]:
    """
    Generate a strat-stealing report from a parsed demo.

    This is the core feature: upload a pro team's demo, get a structured
    tactical breakdown that an IGL can immediately use in their stratbook.

    Request body:
        - team: Which team in the demo to analyze (optional)
        - side: "T", "CT", or None for both (optional)

    Returns:
        - report: Markdown-formatted strat-stealing report
        - patterns: Summary of detected patterns
    """
    _validate_job_id(job_id)

    job_store = _get_job_store()
    completed = _get_job_status_completed()

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != completed:
        raise HTTPException(status_code=400, detail="Job not completed")
    if not job.result:
        raise HTTPException(status_code=400, detail="No analysis result available")

    try:
        body = await request.json()
    except Exception:
        body = {}

    team_focus = body.get("team", None)
    side_focus = body.get("side", None)

    # Validate side
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

        # Also return the raw pattern analysis for the UI
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


@router.post("/api/self-review/{job_id}")
async def self_review_analysis(
    job_id: str,
    request: Request,
) -> dict[str, Any]:
    """
    Generate team self-review report analyzing mistakes and improvement areas.

    Requires a completed analysis job. This endpoint analyzes your team's
    mistakes, generates player report cards, and recommends practice priorities.

    Request body (optional):
        - our_team: Team name to focus on (defaults to auto-detect)

    Returns:
        - report: Markdown-formatted self-review report
        - patterns_detected: Breakdown of detected mistakes
        - player_report_cards: Individual player grades
        - practice_priorities: Recommended focus areas
    """
    if not _validate_demo_id(job_id):
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
        # Parse optional our_team from request body
        our_team: str | None = None
        try:
            body = await request.json()
            our_team = body.get("our_team")
        except Exception:
            pass

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
