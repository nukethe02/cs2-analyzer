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
- POST /api/game-plan/generate — generate game plan from demos + scouting
- GET /api/game-plan/{plan_id} — retrieve cached game plan
- POST /api/veto/analyze — map veto analysis and recommendations
- POST /api/query — natural language query interface
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from opensight.api.shared import (
    _get_job_store,
    validate_demo_id,
    validate_steam_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["match"])


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
        raise HTTPException(
            status_code=500, detail="Failed to store match. Check server logs."
        ) from e


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
        raise HTTPException(
            status_code=500, detail="Failed to get baselines. Check server logs."
        ) from e


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
        raise HTTPException(
            status_code=500, detail="Failed to get match history. Check server logs."
        ) from e


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
        raise HTTPException(
            status_code=500, detail="Failed to get persona. Check server logs."
        ) from e


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
        raise HTTPException(
            status_code=500, detail="Failed to get trends. Check server logs."
        ) from e


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
            status_code=500, detail="Failed to retrieve Your Match data. Check server logs."
        ) from e


# ============================================================================
# Professional Metrics Endpoints
# ============================================================================


@router.get("/api/players/{steam_id}/metrics")
async def get_player_metrics(steam_id: str, demo_id: str = Query(None, max_length=64)) -> dict:
    """Get professional metrics for a player.

    If demo_id is provided, looks up the job results and extracts real metrics.
    If no demo_id, falls back to database history.
    Returns null for unavailable metrics (not zero — zero means 'player got zero').
    """
    validate_steam_id(steam_id)

    if demo_id:
        validate_demo_id(demo_id)

    try:
        player_data: dict[str, Any] | None = None

        # Strategy 1: Look up from job results if demo_id provided
        if demo_id:
            job_store = _get_job_store()
            job = job_store.get_job(demo_id)
            if job and job.result:
                players = job.result.get("players", {})
                if isinstance(players, dict):
                    player_data = players.get(steam_id)
                else:
                    # Legacy list format fallback
                    for p in players:
                        if str(p.get("steam_id")) == steam_id:
                            player_data = p
                            break

        # Strategy 2: Fall back to database history
        if not player_data:
            try:
                from opensight.infra.database import get_db

                db = get_db()
                history = db.get_player_history(steam_id, limit=1)
                if history:
                    # Database history has a flat structure; wrap into expected shape
                    h = history[0]
                    player_data = {
                        "stats": {
                            "kills": h.get("kills"),
                            "deaths": h.get("deaths"),
                            "adr": h.get("adr"),
                        },
                        "rating": {
                            "hltv_rating": h.get("hltv_rating"),
                        },
                        "advanced": {
                            "ttd_median_ms": h.get("ttd_median_ms"),
                            "ttd_mean_ms": h.get("ttd_mean_ms"),
                            "ttd_95th_ms": h.get("ttd_95th_ms"),
                            "cp_median_error_deg": h.get("cp_median_error_deg"),
                            "cp_mean_error_deg": h.get("cp_mean_error_deg"),
                        },
                        "entry": {
                            "entry_attempts": h.get("entry_attempts"),
                            "entry_kills": h.get("entry_kills"),
                            "entry_deaths": h.get("entry_deaths"),
                            "entry_success_pct": h.get("entry_success_pct"),
                        },
                        "trades": h.get("trades", {}),
                        "clutches": h.get("clutches", {}),
                    }
            except Exception:
                logger.exception("Failed to fetch player history from database")

        if not player_data:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics found for player {steam_id}"
                + (
                    f" in demo {demo_id}"
                    if demo_id
                    else ". Provide a demo_id or play more matches."
                ),
            )

        # Extract metrics from nested player data structure
        advanced = player_data.get("advanced", {})
        entry = player_data.get("entry", {})
        trades = player_data.get("trades", {})
        clutches = player_data.get("clutches", {})

        return {
            "steam_id": steam_id,
            "demo_id": demo_id,
            "metrics": {
                "timing": {
                    "ttd_median_ms": advanced.get("ttd_median_ms"),
                    "ttd_mean_ms": advanced.get("ttd_mean_ms"),
                    "ttd_95th_ms": advanced.get("ttd_95th_ms"),
                },
                "positioning": {
                    "cp_median_error_deg": advanced.get("cp_median_error_deg"),
                    "cp_mean_error_deg": advanced.get("cp_mean_error_deg"),
                },
                "entries": {
                    "attempts": entry.get("entry_attempts"),
                    "kills": entry.get("entry_kills"),
                    "deaths": entry.get("entry_deaths"),
                    "success_rate": entry.get("entry_success_pct"),
                },
                "trades": {
                    "trade_kill_opportunities": trades.get("trade_kill_opportunities"),
                    "trade_kill_attempts": trades.get("trade_kill_attempts"),
                    "trade_kill_attempts_pct": trades.get("trade_kill_attempts_pct"),
                    "trade_kill_success": trades.get("trade_kill_success"),
                    "trade_kill_success_pct": trades.get("trade_kill_success_pct"),
                    "traded_death_opportunities": trades.get("traded_death_opportunities"),
                    "traded_death_attempts": trades.get("traded_death_attempts"),
                    "traded_death_attempts_pct": trades.get("traded_death_attempts_pct"),
                    "traded_death_success": trades.get("traded_death_success"),
                    "traded_death_success_pct": trades.get("traded_death_success_pct"),
                    "avg_time_to_trade_ms": trades.get("avg_time_to_trade_ms"),
                    "median_time_to_trade_ms": trades.get("median_time_to_trade_ms"),
                    "traded_entry_kills": trades.get("traded_entry_kills"),
                    "traded_entry_deaths": trades.get("traded_entry_deaths"),
                    "kills_traded": trades.get("trade_kills"),
                    "deaths_traded": trades.get("deaths_traded"),
                    "trade_rate": trades.get("trade_rate"),
                },
                "clutches": {
                    "wins": clutches.get("clutch_wins"),
                    "attempts": clutches.get("total_situations"),
                    "win_rate": clutches.get("clutch_success_pct"),
                    "breakdown": {
                        "v1": clutches.get("v1_wins"),
                        "v2": clutches.get("v2_wins"),
                        "v3": clutches.get("v3_wins"),
                        "v4": clutches.get("v4_wins"),
                        "v5": clutches.get("v5_wins"),
                    },
                },
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to retrieve player metrics")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve metrics. Check server logs."
        ) from e


# =============================================================================
# Per-Player Positioning Heatmaps
# =============================================================================

# IMPORTANT: Static/literal path segments (/all, /compare) must come BEFORE
# parameterized segments (/{steam_id}) to prevent FastAPI from capturing
# "all" or "compare" as a steam_id value.


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

        # NOTE: This endpoint requires raw tick-level position data not stored in orchestrator output.
        # Re-parsing is intentional here. PositioningAnalyzer needs DemoData with per-tick coordinates.
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
            "players": {str(sid): pos_data.to_dict() for sid, pos_data in results.items()},
        }

    except HTTPException:
        raise
    except ImportError as e:
        logger.warning("Positioning module not available: %s", e)
        raise HTTPException(status_code=503, detail="Positioning module not available.") from e
    except Exception as e:
        logger.exception("All players positioning analysis failed for job %s", job_id)
        raise HTTPException(
            status_code=500, detail="Positioning analysis failed. Check server logs."
        ) from e


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

        # NOTE: This endpoint requires raw tick-level position data not stored in orchestrator output.
        # Re-parsing is intentional here. PositioningAnalyzer needs DemoData with per-tick coordinates.
        demo_path = job.result.get("demo_path") if job.result else None
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        parser = DemoParser(Path(demo_path))
        data = parser.parse()
        analyzer = PositioningAnalyzer(data)
        result = analyzer.compare_players(int(steam_id_a), int(steam_id_b))

        return result.to_dict()

    except HTTPException:
        raise
    except ImportError as e:
        logger.warning("Positioning module not available: %s", e)
        raise HTTPException(status_code=503, detail="Positioning module not available.") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid steam_id format.") from e
    except Exception as e:
        logger.exception("Positioning comparison failed for job %s", job_id)
        raise HTTPException(
            status_code=500, detail="Positioning comparison failed. Check server logs."
        ) from e


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

        # NOTE: This endpoint requires raw tick-level position data not stored in orchestrator output.
        # Re-parsing is intentional here. PositioningAnalyzer needs DemoData with per-tick coordinates.
        demo_path = job.result.get("demo_path") if job.result else None
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        parser = DemoParser(Path(demo_path))
        data = parser.parse()
        analyzer = PositioningAnalyzer(data)
        result = analyzer.analyze_player(int(steam_id))

        return result.to_dict()

    except HTTPException:
        raise
    except ImportError as e:
        logger.warning("Positioning module not available: %s", e)
        raise HTTPException(status_code=503, detail="Positioning module not available.") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid steam_id format.") from e
    except Exception as e:
        logger.exception("Positioning analysis failed for job %s, player %s", job_id, steam_id)
        raise HTTPException(
            status_code=500, detail="Positioning analysis failed. Check server logs."
        ) from e


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

    if not job.result:
        raise HTTPException(status_code=400, detail="Job has no result data")

    try:
        import pandas as pd

        from opensight.core.parser import DemoData
        from opensight.domains.combat import CombatAnalyzer

        # Build kills DataFrame from cached round_timeline instead of re-parsing
        # the full demo. CombatAnalyzer only needs kills_df, player_names,
        # player_teams, and tick_rate — all available in job.result.
        kills_rows: list[dict] = []
        for rd in job.result.get("round_timeline", []):
            rn = rd.get("round_num", 0)
            for k in rd.get("kills", []):
                kills_rows.append(
                    {
                        "attacker_steamid": int(k.get("killer_steamid", 0)),
                        "attacker_name": k.get("killer", "Unknown"),
                        "victim_steamid": int(k.get("victim_steamid", 0)),
                        "victim_name": k.get("victim", "Unknown"),
                        "round_num": rn,
                        "tick": k.get("tick", 0),
                        "weapon": k.get("weapon", ""),
                        "headshot": k.get("headshot", False),
                        "attacker_x": k.get("killer_x"),
                        "attacker_y": k.get("killer_y"),
                        "victim_x": k.get("victim_x"),
                        "victim_y": k.get("victim_y"),
                    }
                )

        kills_df = pd.DataFrame(kills_rows) if kills_rows else pd.DataFrame()

        # Build name→team mapping from players dict (authoritative initial teams).
        # Kill-event steamids may differ from players-dict keys, so we bridge
        # via player name to get correct team assignments.
        name_to_team: dict[str, str] = {}
        for pdata in job.result.get("players", {}).values():
            name_to_team[pdata.get("name", "")] = pdata.get("team", "Unknown")

        # Build steamid→name and steamid→team from kill events so steamids
        # are consistent with kills_df. Use name bridge for team lookup.
        player_names: dict[int, str] = {}
        player_teams: dict[int, str] = {}
        for rd in job.result.get("round_timeline", []):
            for k in rd.get("kills", []):
                for sid_key, name_key in [
                    ("killer_steamid", "killer"),
                    ("victim_steamid", "victim"),
                ]:
                    sid = int(k.get(sid_key, 0))
                    name = k.get(name_key, "Unknown")
                    if sid and sid not in player_names:
                        player_names[sid] = name
                        player_teams[sid] = name_to_team.get(name, "Unknown")

        demo_info = job.result.get("demo_info", {})

        # Construct minimal DemoData from cached results (no demo re-parse)
        demo_data = DemoData(
            file_path=Path(job.result.get("demo_path", "")),
            map_name=demo_info.get("map", "unknown"),
            duration_seconds=0.0,
            tick_rate=64,
            num_rounds=demo_info.get("rounds", 0),
            player_names=player_names,
            player_teams=player_teams,
            kills_df=kills_df,
        )

        analyzer = CombatAnalyzer(demo_data)
        combat_result = analyzer.analyze()

        chains = combat_result.trade_chains

        if round_num is not None:
            chains = [c for c in chains if c.round_num == round_num]

        chains = [c for c in chains if c.chain_length >= min_chain_length]

        chains_data = [c.to_dict() for c in chains]

        return {
            "job_id": job_id,
            "map_name": demo_data.map_name,
            "total_rounds": demo_data.num_rounds,
            "chains": chains_data,
            "stats": combat_result.trade_chain_stats.to_dict()
            if combat_result.trade_chain_stats
            else None,
            "filter_applied": {
                "round_num": round_num,
                "min_chain_length": min_chain_length,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Trade chain analysis failed for job %s", job_id)
        raise HTTPException(
            status_code=500, detail="Trade chain analysis failed. Check server logs."
        ) from e


# =============================================================================
# Player Development Tracking
# =============================================================================


@router.get("/api/player-tracking/{steam_id}/report")
async def get_player_development_report(
    steam_id: str,
    limit: int = Query(default=30, le=100, description="Max matches to analyze"),
) -> dict[str, Any]:
    """Get comprehensive player development report with trends, benchmarks, and recommendations."""
    validate_steam_id(steam_id)

    try:
        from opensight.ai.player_tracker import get_player_tracker

        tracker = get_player_tracker()
        report = tracker.generate_report(steam_id, limit=limit)
        return report.to_dict()

    except Exception as e:
        logger.exception("Failed to generate development report for %s", steam_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate development report. Check server logs.",
        ) from e


@router.get("/api/player-tracking/{steam_id}/trends")
async def get_player_development_trends(
    steam_id: str,
) -> dict[str, Any]:
    """Get metric trend analysis for a player across their match history."""
    validate_steam_id(steam_id)

    try:
        from opensight.ai.player_tracker import get_player_tracker

        tracker = get_player_tracker()
        trends = tracker.analyze_trends_from_db(steam_id)

        return {
            "steam_id": steam_id,
            "trends": [t.to_dict() for t in trends],
            "count": len(trends),
        }

    except Exception as e:
        logger.exception("Failed to get development trends for %s", steam_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to get development trends. Check server logs.",
        ) from e


@router.post("/api/player-tracking/{steam_id}/add-match")
async def add_match_to_tracking(
    steam_id: str,
    job_id: str = Query(..., description="Job ID of completed analysis"),
) -> dict[str, Any]:
    """Extract a match snapshot from a completed analysis and store it for tracking."""
    validate_steam_id(steam_id)
    validate_demo_id(job_id)

    job_store = _get_job_store()
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.result:
        raise HTTPException(status_code=400, detail="Job not completed or has no results")

    try:
        from opensight.ai.player_tracker import get_player_tracker

        tracker = get_player_tracker()
        snapshot = tracker.extract_snapshot(job.result, steam_id)

        if not snapshot:
            raise HTTPException(
                status_code=404,
                detail=f"Player {steam_id} not found in analysis results",
            )

        # Store via existing DB infrastructure
        from opensight.infra.database import get_db

        db = get_db()
        entry = db.save_match_history_entry(
            steam_id=steam_id,
            demo_hash=snapshot.demo_hash,
            player_stats={
                "kills": snapshot.kills,
                "deaths": snapshot.deaths,
                "assists": snapshot.assists,
                "adr": snapshot.adr,
                "kast_percentage": snapshot.kast,
                "headshot_percentage": snapshot.hs_pct,
                "hltv_rating": snapshot.hltv_rating,
                "aim_rating": snapshot.aim_rating,
                "utility_rating": snapshot.utility_rating,
                "ttd_median_ms": snapshot.ttd_median_ms,
                "cp_median_error_deg": snapshot.cp_median_deg,
                "opening_duel_attempts": snapshot.entry_attempts,
                "opening_duel_wins": snapshot.entry_success,
                "clutch_situations": snapshot.clutch_situations,
                "clutch_wins": snapshot.clutch_wins,
                "kills_traded": snapshot.trade_kill_success,
                "enemies_flashed": snapshot.enemies_flashed,
                "flash_assists": snapshot.flash_assists,
                "he_damage": snapshot.he_damage,
                "rounds_played": snapshot.rounds_played,
            },
            map_name=snapshot.map_name,
            result=snapshot.result,
        )

        if entry is None:
            return {"status": "duplicate", "message": "Match already tracked"}

        # Update baselines
        db.update_player_baselines(steam_id)

        return {
            "status": "ok",
            "match_id": entry.id,
            "snapshot": snapshot.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to add match to tracking for %s", steam_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to add match to tracking. Check server logs.",
        ) from e


# =============================================================================
# Game Plan Generation
# =============================================================================


@router.post("/api/game-plan/generate")
async def generate_game_plan(
    your_team_job_ids: list[str] = Body(..., embed=True),
    opponent_scouting: dict[str, Any] = Body(..., embed=True),
    map_name: str = Body(..., embed=True, alias="map"),
    roster: dict[str, str] = Body(..., embed=True),
) -> dict[str, Any]:
    """
    Generate a complete game plan from your team's demos + opponent scouting data.

    Body:
        your_team_job_ids: List of completed job IDs for your recent matches
        opponent_scouting: Opponent scouting data (TeamScoutReport.to_dict() format)
        map: Map name (e.g., "de_ancient")
        roster: Player name → role mapping (e.g., {"foe": "igl", "kix": "entry"})
    """
    from opensight.ai.game_plan import cache_plan, get_game_plan_generator
    from opensight.api.shared import JobStatus

    if not your_team_job_ids:
        raise HTTPException(status_code=400, detail="At least one job ID is required")

    if not roster:
        raise HTTPException(status_code=400, detail="Roster cannot be empty")

    if not map_name:
        raise HTTPException(status_code=400, detail="Map name is required")

    # Collect orchestrator results from completed jobs
    job_store = _get_job_store()
    your_demos: list[dict] = []

    for job_id in your_team_job_ids:
        job = job_store.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail=f"Job {job_id} not found or expired"
            )
        if job.status != JobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed (status: {job.status})",
            )
        if job.result:
            your_demos.append(job.result)

    if not your_demos:
        raise HTTPException(
            status_code=400,
            detail="No completed job results found for the provided job IDs",
        )

    try:
        generator = get_game_plan_generator()
        plan = generator.generate(
            your_team_demos=your_demos,
            opponent_scouting=opponent_scouting,
            map_name=map_name,
            roster=roster,
        )

        # Cache the plan for retrieval
        cache_plan(plan)

        return plan.to_dict()

    except Exception as e:
        logger.exception("Failed to generate game plan")
        raise HTTPException(
            status_code=500,
            detail=f"Game plan generation failed: {type(e).__name__}",
        ) from e


@router.get("/api/game-plan/{plan_id}")
async def get_game_plan(plan_id: str) -> dict[str, Any]:
    """Retrieve a cached game plan by ID."""
    from opensight.ai.game_plan import get_cached_plan

    plan = get_cached_plan(plan_id)
    if plan is None:
        raise HTTPException(
            status_code=404,
            detail=f"Game plan {plan_id} not found or expired",
        )
    return plan.to_dict()


# =============================================================================
# Veto Analysis
# =============================================================================


@router.post("/api/veto/analyze")
async def analyze_veto(
    your_job_ids: list[str] = Body(..., embed=True),
    opponent_job_ids: list[str] | None = Body(None, embed=True),
    opponent_scouting: dict[str, Any] | None = Body(None, embed=True),
    format: str = Body("bo1", embed=True),
) -> dict[str, Any]:
    """
    Analyze map veto options for an upcoming match.

    Provide your team's job IDs and either opponent job IDs or scouting data.

    Body:
        your_job_ids: List of completed job IDs for your recent matches
        opponent_job_ids: Optional list of completed job IDs for opponent matches
        opponent_scouting: Optional opponent scouting data (TeamScoutReport.to_dict())
        format: "bo1" or "bo3"
    """
    from opensight.ai.veto_optimizer import get_veto_optimizer
    from opensight.api.shared import JobStatus

    if not your_job_ids:
        raise HTTPException(status_code=400, detail="At least one job ID is required")

    if format not in ("bo1", "bo3"):
        raise HTTPException(status_code=400, detail="Format must be 'bo1' or 'bo3'")

    if not opponent_job_ids and not opponent_scouting:
        raise HTTPException(
            status_code=400,
            detail="Either opponent_job_ids or opponent_scouting is required",
        )

    job_store = _get_job_store()

    # Collect your demos
    your_demos: list[dict] = []
    for job_id in your_job_ids:
        job = job_store.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail=f"Job {job_id} not found or expired"
            )
        if job.status != JobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed (status: {job.status})",
            )
        if job.result:
            your_demos.append(job.result)

    # Collect opponent data
    opponent_data: list[dict] | dict
    if opponent_job_ids:
        opp_demos: list[dict] = []
        for job_id in opponent_job_ids:
            job = job_store.get_job(job_id)
            if job is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Opponent job {job_id} not found or expired",
                )
            if job.status != JobStatus.COMPLETED.value:
                raise HTTPException(
                    status_code=400,
                    detail=f"Opponent job {job_id} is not completed (status: {job.status})",
                )
            if job.result:
                opp_demos.append(job.result)
        opponent_data = opp_demos
    else:
        opponent_data = opponent_scouting  # type: ignore[assignment]

    try:
        optimizer = get_veto_optimizer()
        analysis = optimizer.analyze(
            your_demos=your_demos,
            opponent_data=opponent_data,
            format=format,
        )
        return analysis.to_dict()

    except Exception as e:
        logger.exception("Failed to generate veto analysis")
        raise HTTPException(
            status_code=500,
            detail=f"Veto analysis failed: {type(e).__name__}",
        ) from e


# =============================================================================
# Natural Language Query Interface
# =============================================================================


@router.post("/api/query")
async def natural_language_query(
    question: str = Body(..., embed=True),
    job_id: str = Body(..., embed=True),
) -> dict[str, Any]:
    """
    Answer a natural language question about a completed demo analysis.

    Body:
        question: Natural language question (e.g., "How often does kix win opening duels?")
        job_id: Job ID of a completed analysis
    """
    from opensight.ai.query_interface import get_query_interface
    from opensight.api.shared import JobStatus

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    validate_demo_id(job_id)

    job_store = _get_job_store()
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Job {job_id} not found or expired"
        )
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed (status: {job.status})",
        )
    if not job.result:
        raise HTTPException(
            status_code=400, detail="Job has no result data"
        )

    try:
        qi = get_query_interface()
        result = qi.query(question=question.strip(), available_data=job.result)
        return result.to_dict()

    except Exception as e:
        logger.exception("Natural language query failed")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {type(e).__name__}",
        ) from e
