"""
Match and player route handlers.

Endpoints:
- GET /api/your-match/{demo_id}/{steam_id} — personalized match performance
- POST /api/your-match/store — store match in player history
- GET /api/your-match/baselines/{steam_id} — player 30-match averages
- GET /api/your-match/history/{steam_id} — match history
- GET /api/your-match/persona/{steam_id} — player identity persona
- GET /api/your-match/trends/{steam_id} — performance trends
- GET /api/players/{steam_id}/metrics — professional metrics
- GET /api/positioning/{job_id}/{steam_id} — player positioning heatmaps
- GET /api/positioning/{job_id}/compare/{steam_id_a}/{steam_id_b} — compare positioning
- GET /api/positioning/{job_id}/all — all player positioning
- GET /api/trade-chains/{job_id} — trade chain analysis
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from opensight.api.shared import (
    validate_demo_id,
    validate_steam_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["match"])


def _get_job_store():
    """Lazy import to avoid circular dependency."""
    from opensight.api import job_store

    return job_store


# NOTE: Static path routes must come BEFORE parameterized routes to avoid conflicts


@router.post("/api/your-match/store")
async def store_match_for_player(
    steam_id: str = Body(..., embed=True),
    demo_hash: str = Body(..., embed=True),
    player_stats: dict[str, Any] = Body(..., embed=True),
    map_name: str | None = Body(None, embed=True),
    result: str | None = Body(None, embed=True),
) -> dict[str, Any]:
    """Store a match in player's history and update baselines."""
    validate_steam_id(steam_id)

    try:
        from opensight.infra.database import get_db

        db = get_db()

        entry = db.save_match_history_entry(
            steam_id=steam_id,
            demo_hash=demo_hash,
            player_stats=player_stats,
            map_name=map_name,
            result=result,
        )

        if entry is None:
            return {"status": "duplicate", "message": "Match already recorded"}

        baselines = db.update_player_baselines(steam_id)

        return {
            "status": "ok",
            "match_id": entry.id,
            "baselines_updated": len(baselines),
        }

    except Exception as e:
        logger.exception("Failed to store match")
        raise HTTPException(status_code=500, detail=f"Failed to store match: {e!s}") from e


@router.get("/api/your-match/baselines/{steam_id}")
async def get_player_baselines_endpoint(steam_id: str) -> dict[str, Any]:
    """Get a player's baseline statistics."""
    validate_steam_id(steam_id)

    try:
        from opensight.infra.database import get_db

        db = get_db()
        baselines = db.get_player_baselines(steam_id)

        return {
            "steam_id": steam_id,
            "baselines": baselines,
            "metric_count": len(baselines),
        }

    except Exception as e:
        logger.exception("Failed to get baselines")
        raise HTTPException(status_code=500, detail=f"Failed to get baselines: {e!s}") from e


@router.get("/api/your-match/history/{steam_id}")
async def get_player_match_history_endpoint(
    steam_id: str, limit: int = Query(default=30, le=100)
) -> dict[str, Any]:
    """Get a player's match history for the Your Match feature."""
    validate_steam_id(steam_id)

    try:
        from opensight.infra.database import get_db

        db = get_db()
        history = db.get_player_history(steam_id, limit=limit)

        return {
            "steam_id": steam_id,
            "matches": history,
            "count": len(history),
        }

    except Exception as e:
        logger.exception("Failed to get match history")
        raise HTTPException(status_code=500, detail=f"Failed to get match history: {e!s}") from e


@router.get("/api/your-match/persona/{steam_id}")
async def get_player_persona_endpoint(steam_id: str) -> dict[str, Any]:
    """Get a player's current persona based on their match history."""
    validate_steam_id(steam_id)

    try:
        from opensight.analysis.persona import PersonaAnalyzer
        from opensight.infra.database import get_db

        db = get_db()
        history = db.get_player_history(steam_id, limit=10)

        if not history:
            return {
                "steam_id": steam_id,
                "persona": {
                    "id": "the_competitor",
                    "name": "The Competitor",
                    "description": "Play more matches to determine your identity",
                    "confidence": 0.0,
                },
                "match_count": 0,
            }

        aggregated: dict[str, Any] = {}
        count = len(history)

        metrics_to_avg = [
            "kills",
            "deaths",
            "adr",
            "kast",
            "hs_pct",
            "hltv_rating",
            "aim_rating",
            "utility_rating",
            "trade_kill_success",
            "entry_success",
            "clutch_wins",
            "enemies_flashed",
        ]

        for metric in metrics_to_avg:
            values = [m.get(metric, 0) for m in history if m.get(metric) is not None]
            if values:
                aggregated[metric] = sum(values) / len(values)

        aggregated["trade_kill_opportunities"] = sum(
            m.get("trade_kill_opportunities", 0) for m in history
        )
        aggregated["clutch_situations"] = sum(m.get("clutch_situations", 0) for m in history)
        aggregated["entry_attempts"] = sum(m.get("entry_attempts", 0) for m in history)

        analyzer = PersonaAnalyzer()
        persona = analyzer.determine_persona(aggregated)

        db.update_player_persona(
            steam_id=steam_id,
            persona_id=persona.id,
            confidence=persona.confidence,
            primary_trait=persona.primary_trait,
            secondary_trait=persona.secondary_trait,
        )

        return {
            "steam_id": steam_id,
            "persona": persona.to_dict(),
            "match_count": count,
        }

    except Exception as e:
        logger.exception("Failed to get persona")
        raise HTTPException(status_code=500, detail=f"Failed to get persona: {e!s}") from e


@router.get("/api/your-match/trends/{steam_id}")
async def get_player_trends_endpoint(steam_id: str, days: int = 30) -> dict[str, Any]:
    """Get performance trends for a player over time."""
    validate_steam_id(steam_id)

    days = max(7, min(days, 365))

    try:
        from opensight.infra.database import get_db

        db = get_db()
        trends = db.get_player_trends(steam_id, days=days)

        if not trends:
            return {
                "steam_id": steam_id,
                "period_days": days,
                "matches_analyzed": 0,
                "message": "Insufficient match history. Play more matches to see trends.",
                "rating": {"history": [], "trend": "stable", "change_pct": 0},
                "adr": {"history": [], "trend": "stable", "change_pct": 0},
                "winrate": {"history": [], "current": 0},
                "slump": {"detected": False, "severity": None},
                "improvement_areas": [],
                "map_performance": {},
            }

        return trends

    except Exception as e:
        logger.exception("Failed to get player trends")
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {e!s}") from e


# Parameterized route MUST come AFTER static routes
@router.get("/api/your-match/{demo_id}/{steam_id}")
async def get_your_match(demo_id: str, steam_id: str) -> dict[str, Any]:
    """Get personalized match performance data (Leetify-style "Your Match" feature)."""
    validate_demo_id(demo_id)
    validate_steam_id(steam_id)

    job_store = _get_job_store()

    try:
        from opensight.analysis.persona import PersonaAnalyzer
        from opensight.infra.database import get_db

        db = get_db()

        current_stats = None
        job = job_store.get_job(demo_id)

        if job and job.result:
            players = job.result.get("players", {})
            # players is a dict keyed by steam_id string
            if isinstance(players, dict):
                current_stats = players.get(steam_id)
            else:
                # Legacy list format fallback
                for player in players:
                    if str(player.get("steam_id")) == steam_id:
                        current_stats = player
                        break

        if not current_stats:
            history = db.get_player_history(steam_id, limit=1)
            if history:
                current_stats = history[0]
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"No stats found for player {steam_id} in demo {demo_id}",
                )

        baselines = db.get_player_baselines(steam_id)
        analyzer = PersonaAnalyzer(baselines)
        persona = analyzer.determine_persona(current_stats)
        top_5 = analyzer.calculate_top_5_stats(current_stats, baselines)
        comparison = analyzer.build_comparison_table(current_stats, baselines)

        match_count = 0
        if baselines:
            first_baseline = next(iter(baselines.values()), {})
            match_count = first_baseline.get("sample_count", 0)

        return {
            "persona": persona.to_dict(),
            "top_5": [s.to_dict() for s in top_5],
            "comparison": [c.to_dict() for c in comparison],
            "match_count": match_count,
            "steam_id": steam_id,
            "demo_id": demo_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Your Match data retrieval failed")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve Your Match data: {e!s}"
        ) from e


# ============================================================================
# Professional Metrics Endpoints
# ============================================================================


@router.get("/api/players/{steam_id}/metrics")
async def get_player_metrics(steam_id: str, demo_id: str = Query(None, max_length=64)) -> dict:
    """Get professional metrics for a player."""
    validate_steam_id(steam_id)

    if demo_id:
        validate_demo_id(demo_id)

    try:
        from opensight.infra.cache import DemoCache

        DemoCache()

        if demo_id:
            pass

        return {
            "steam_id": steam_id,
            "metrics": {
                "timing": {
                    "ttd_median_ms": 0,
                    "ttd_mean_ms": 0,
                    "ttd_95th_ms": 0,
                },
                "positioning": {
                    "cp_median_error_deg": 0,
                    "cp_mean_error_deg": 0,
                },
                "entries": {
                    "attempts": 0,
                    "kills": 0,
                    "deaths": 0,
                    "success_rate": 0.0,
                },
                "trades": {
                    "trade_kill_opportunities": 0,
                    "trade_kill_attempts": 0,
                    "trade_kill_attempts_pct": 0.0,
                    "trade_kill_success": 0,
                    "trade_kill_success_pct": 0.0,
                    "traded_death_opportunities": 0,
                    "traded_death_attempts": 0,
                    "traded_death_attempts_pct": 0.0,
                    "traded_death_success": 0,
                    "traded_death_success_pct": 0.0,
                    "avg_time_to_trade_ms": None,
                    "median_time_to_trade_ms": None,
                    "traded_entry_kills": 0,
                    "traded_entry_deaths": 0,
                    "kills_traded": 0,
                    "deaths_traded": 0,
                    "trade_rate": 0.0,
                },
                "clutches": {
                    "wins": 0,
                    "attempts": 0,
                    "win_rate": 0.0,
                    "breakdown": {
                        "v1": 0,
                        "v2": 0,
                        "v3": 0,
                        "v4": 0,
                        "v5": 0,
                    },
                },
            },
        }
    except Exception as e:
        logger.exception("Failed to retrieve player metrics")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {e!s}") from e


# =============================================================================
# Per-Player Positioning Heatmaps
# =============================================================================


@router.get("/api/positioning/{job_id}/{steam_id}")
async def get_player_positioning(job_id: str, steam_id: str) -> dict[str, object]:
    """Get per-player positioning heatmap data for a completed analysis."""
    validate_demo_id(job_id)
    validate_steam_id(steam_id)

    job_store = _get_job_store()
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job.status}")

    try:
        from opensight.analysis.positioning import PositioningAnalyzer
        from opensight.core.parser import DemoParser

        demo_path = job.result.get("demo_path") if job.result else None
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        parser = DemoParser(Path(demo_path))
        data = parser.parse()
        analyzer = PositioningAnalyzer(data)
        result = analyzer.analyze_player(int(steam_id))

        return result.to_dict()

    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Positioning module not available: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid steam_id: {e}") from e
    except Exception as e:
        logger.exception("Positioning analysis failed")
        raise HTTPException(status_code=500, detail=f"Positioning analysis failed: {e!s}") from e


@router.get("/api/positioning/{job_id}/compare/{steam_id_a}/{steam_id_b}")
async def compare_player_positioning(
    job_id: str, steam_id_a: str, steam_id_b: str
) -> dict[str, object]:
    """Compare positioning of two players from a completed analysis."""
    validate_demo_id(job_id)
    validate_steam_id(steam_id_a)
    validate_steam_id(steam_id_b)

    job_store = _get_job_store()
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job.status}")

    try:
        from opensight.analysis.positioning import PositioningAnalyzer
        from opensight.core.parser import DemoParser

        demo_path = job.result.get("demo_path") if job.result else None
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        parser = DemoParser(Path(demo_path))
        data = parser.parse()
        analyzer = PositioningAnalyzer(data)
        result = analyzer.compare_players(int(steam_id_a), int(steam_id_b))

        return result.to_dict()

    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Positioning module not available: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid steam_id: {e}") from e
    except Exception as e:
        logger.exception("Positioning comparison failed")
        raise HTTPException(status_code=500, detail=f"Positioning comparison failed: {e!s}") from e


@router.get("/api/positioning/{job_id}/all")
async def get_all_player_positioning(job_id: str) -> dict[str, object]:
    """Get positioning heatmaps for all players in a completed analysis."""
    validate_demo_id(job_id)

    job_store = _get_job_store()
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job.status}")

    try:
        from opensight.analysis.positioning import PositioningAnalyzer
        from opensight.core.parser import DemoParser

        demo_path = job.result.get("demo_path") if job.result else None
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        parser = DemoParser(Path(demo_path))
        data = parser.parse()
        analyzer = PositioningAnalyzer(data)
        results = analyzer.analyze_all_players()

        return {
            "map_name": data.map_name,
            "player_count": len(results),
            "players": {str(sid): data.to_dict() for sid, data in results.items()},
        }

    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Positioning module not available: {e}") from e
    except Exception as e:
        logger.exception("All players positioning analysis failed")
        raise HTTPException(status_code=500, detail=f"Positioning analysis failed: {e!s}") from e


# =============================================================================
# Trade Chain Visualization
# =============================================================================


@router.get("/api/trade-chains/{job_id}")
async def get_trade_chains(
    job_id: str,
    round_num: int | None = Query(None, description="Filter by round number"),
    min_chain_length: int = Query(2, ge=2, description="Minimum chain length"),
) -> dict[str, object]:
    """Get trade chain analysis for a completed demo."""
    validate_demo_id(job_id)

    job_store = _get_job_store()
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job.status}")

    try:
        from opensight.core.parser import DemoParser
        from opensight.domains.combat import CombatAnalyzer

        demo_path = job.result.get("demo_path") if job.result else None
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        parser = DemoParser(Path(demo_path))
        data = parser.parse()
        analyzer = CombatAnalyzer(data)
        result = analyzer.analyze()

        chains = result.trade_chains

        if round_num is not None:
            chains = [c for c in chains if c.round_num == round_num]

        chains = [c for c in chains if c.chain_length >= min_chain_length]

        chains_data = [c.to_dict() for c in chains]

        return {
            "job_id": job_id,
            "map_name": data.map_name,
            "total_rounds": data.num_rounds,
            "chains": chains_data,
            "stats": result.trade_chain_stats.to_dict() if result.trade_chain_stats else None,
            "filter_applied": {
                "round_num": round_num,
                "min_chain_length": min_chain_length,
            },
        }

    except Exception as e:
        logger.exception("Trade chain analysis failed")
        raise HTTPException(status_code=500, detail=f"Trade chain analysis failed: {e!s}") from e
