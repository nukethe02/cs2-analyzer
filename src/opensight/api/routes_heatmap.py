"""
Heatmap route handlers.

Endpoints:
- GET /api/heatmap/{job_id}/kills — kill/death heatmap from completed analysis
- GET /api/heatmap/{job_id}/grenades — grenade landing heatmap from completed analysis
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from opensight.api.shared import JobStatus, _get_job_store, validate_job_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["heatmaps"])


def _get_completed_result(job_id: str) -> dict[str, Any]:
    """Validate job_id, fetch job, ensure completed, return result dict.

    TODO(DRY): extract to shared.py — identical copy exists in routes_export.py.
    """
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


@router.get("/api/heatmap/{job_id}/kills")
async def get_kill_heatmap(
    job_id: str,
    steam_id: str | None = Query(None, description="Filter by player steam ID"),
    side: str | None = Query(None, description="Filter by side (CT/T)"),
    weapon: str | None = Query(None, description="Filter by weapon name"),
    round_min: int | None = Query(None, description="Minimum round number"),
    round_max: int | None = Query(None, description="Maximum round number"),
) -> dict[str, Any]:
    """Generate kill/death heatmap from a completed analysis job."""
    result = _get_completed_result(job_id)

    kills = result.get("kills", [])
    map_name = result.get("map_name", result.get("map", "de_dust2"))

    if not kills:
        return {"map_name": map_name, "points": [], "total": 0}

    filters: dict[str, Any] = {}
    if steam_id is not None:
        filters["steam_id"] = steam_id
    if side is not None:
        filters["side"] = side
    if weapon is not None:
        filters["weapon"] = weapon
    if round_min is not None and round_max is not None:
        filters["round_range"] = (round_min, round_max)

    try:
        from opensight.visualization.heatmaps import generate_kill_heatmap

        return generate_kill_heatmap(kills, map_name, filters or None)
    except Exception as e:
        logger.exception(f"Heatmap generation failed for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Heatmap generation failed: {type(e).__name__}",
        ) from e


@router.get("/api/heatmap/{job_id}/grenades")
async def get_grenade_heatmap(
    job_id: str,
    grenade_type: str | None = Query(
        None, description="Filter by grenade type (flashbang, smoke, hegrenade, molotov)"
    ),
) -> dict[str, Any]:
    """Generate grenade landing heatmap from a completed analysis job."""
    result = _get_completed_result(job_id)

    grenades = result.get("grenades", [])
    map_name = result.get("map_name", result.get("map", "de_dust2"))

    if not grenades:
        return {"map_name": map_name, "points": [], "total": 0}

    try:
        from opensight.visualization.heatmaps import generate_grenade_heatmap

        return generate_grenade_heatmap(grenades, map_name, grenade_type)
    except Exception as e:
        logger.exception(f"Grenade heatmap generation failed for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Grenade heatmap generation failed: {type(e).__name__}",
        ) from e
