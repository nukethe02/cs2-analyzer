"""
Export route handlers.

Endpoints:
- GET /api/export/{job_id}/json — full match JSON download
- GET /api/export/{job_id}/players-csv — player stats CSV download
- GET /api/export/{job_id}/rounds-csv — round-by-round CSV download
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from opensight.api.shared import JobStatus, validate_job_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["export"])


def _get_job_store():
    """Lazy import to avoid circular dependency."""
    from opensight.api import job_store

    return job_store


def _get_completed_result(job_id: str) -> dict[str, Any]:
    """Validate job_id, fetch job, ensure completed, return result dict."""
    validate_job_id(job_id)
    job_store = _get_job_store()

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    result = job.result
    if not result or not isinstance(result, dict):
        raise HTTPException(status_code=404, detail="No analysis result available")

    return result


@router.get("/api/export/{job_id}/json")
async def export_match_json(job_id: str) -> Response:
    """Download the full match analysis as a JSON file."""
    result = _get_completed_result(job_id)

    try:
        from opensight.visualization.exports import export_match_json as _export_json

        json_str = _export_json(result)
    except Exception as e:
        logger.exception(f"JSON export failed for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {type(e).__name__}",
        ) from e

    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="match_{job_id[:8]}.json"'},
    )


@router.get("/api/export/{job_id}/players-csv")
async def export_players_csv(job_id: str) -> Response:
    """Download player stats as a CSV file."""
    result = _get_completed_result(job_id)

    players_raw = result.get("players", {})
    if not players_raw:
        raise HTTPException(status_code=404, detail="No player data in analysis result")

    # Convert dict → list for CSV export (orchestrator outputs dict keyed by steam_id)
    players = list(players_raw.values()) if isinstance(players_raw, dict) else players_raw

    try:
        from opensight.visualization.exports import export_player_stats_csv

        csv_str = export_player_stats_csv(players)
    except Exception as e:
        logger.exception(f"Player CSV export failed for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {type(e).__name__}",
        ) from e

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="players_{job_id[:8]}.csv"'},
    )


@router.get("/api/export/{job_id}/rounds-csv")
async def export_rounds_csv(job_id: str) -> Response:
    """Download round-by-round data as a CSV file."""
    result = _get_completed_result(job_id)

    rounds = result.get("rounds", result.get("round_timeline", []))
    if not rounds:
        raise HTTPException(status_code=404, detail="No round data in analysis result")

    try:
        from opensight.visualization.exports import export_rounds_csv as _export_rounds

        csv_str = _export_rounds(rounds)
    except Exception as e:
        logger.exception(f"Rounds CSV export failed for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {type(e).__name__}",
        ) from e

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="rounds_{job_id[:8]}.csv"'},
    )
