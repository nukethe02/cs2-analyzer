"""Player development tracking API endpoints."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException

from opensight.api.shared import validate_steam_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tracking"])


def _get_tracker():
    """Lazy import to avoid circular dependencies."""
    from opensight.ai.player_tracker import PlayerTracker
    from opensight.infra.database import get_db

    return PlayerTracker(get_db())


@router.get("/api/tracking/{steam_id}/trends")
async def get_player_trends(steam_id: str) -> dict[str, Any]:
    """Get metric trend analysis for a player."""
    if not validate_steam_id(steam_id):
        raise HTTPException(status_code=400, detail="Invalid steam_id format")

    try:
        tracker = _get_tracker()
        trends = tracker.analyze_trends(steam_id)
        if not trends:
            raise HTTPException(
                status_code=400,
                detail="Insufficient match history (minimum 5 matches required)",
            )
        return {
            "steam_id": steam_id,
            "trend_count": len(trends),
            "trends": [asdict(t) for t in trends],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error analyzing trends for %s", steam_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/tracking/{steam_id}/benchmarks")
async def get_player_benchmarks(steam_id: str) -> dict[str, Any]:
    """Get role benchmark comparison for a player."""
    if not validate_steam_id(steam_id):
        raise HTTPException(status_code=400, detail="Invalid steam_id format")

    try:
        tracker = _get_tracker()
        benchmark = tracker.get_role_benchmarks(steam_id)
        if benchmark is None:
            raise HTTPException(
                status_code=400,
                detail="Insufficient match history (minimum 5 matches required)",
            )
        return {
            "steam_id": steam_id,
            "role": benchmark.role,
            "benchmarks": benchmark.benchmarks,
            "percentiles": benchmark.percentiles,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing benchmarks for %s", steam_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/tracking/{steam_id}/recommendations")
async def get_player_recommendations(steam_id: str) -> dict[str, Any]:
    """Get practice recommendations for a player."""
    if not validate_steam_id(steam_id):
        raise HTTPException(status_code=400, detail="Invalid steam_id format")

    try:
        tracker = _get_tracker()
        recs = tracker.generate_recommendations(steam_id)
        return {
            "steam_id": steam_id,
            "recommendation_count": len(recs),
            "recommendations": [asdict(r) for r in recs],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating recommendations for %s", steam_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/tracking/{steam_id}/report")
async def get_development_report(steam_id: str) -> dict[str, Any]:
    """Get full development report for a player."""
    if not validate_steam_id(steam_id):
        raise HTTPException(status_code=400, detail="Invalid steam_id format")

    try:
        tracker = _get_tracker()
        report = tracker.get_development_report(steam_id)
        if report is None:
            raise HTTPException(
                status_code=400,
                detail="Insufficient match history (minimum 5 matches required)",
            )
        return {
            "steam_id": report.steam_id,
            "match_count": report.match_count,
            "date_range": {
                "earliest": report.date_range[0],
                "latest": report.date_range[1],
            },
            "trends": [asdict(t) for t in report.trends],
            "role_benchmark": asdict(report.role_benchmark) if report.role_benchmark else None,
            "recommendations": [asdict(r) for r in report.recommendations],
            "summary": report.summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating report for %s", steam_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
