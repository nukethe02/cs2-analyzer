"""
OpenSight Web API

FastAPI application for CS2 demo analysis with professional-grade metrics.

Provides:
- Demo analysis with HLTV 2.0 Rating, KAST%, TTD, Crosshair Placement
- Batch analysis with parallel processing
- 2D replay data generation
- Radar map coordinate transformation
- HLTV integration for pro player detection
- Caching for faster repeated analysis
- Community feedback system
- Background job processing for large demos
- Response compression (GZip)
- Share code caching
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
import logging
from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import threading
from functools import lru_cache
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import Response
from pydantic import BaseModel, Field

from opensight.profiling import (
    TimingCollector,
    SlowJobLogger,
    set_timing_collector,
    DEFAULT_SLOW_THRESHOLD_SECONDS,
)

__version__ = "0.3.0"

# Security constants
MAX_FILE_SIZE_MB = 500  # Maximum demo file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = (".dem", ".dem.gz")

# Job processing constants
MAX_CONCURRENT_JOBS = 4
JOB_TTL_HOURS = 24  # Jobs expire after 24 hours
JOB_CLEANUP_INTERVAL_SECONDS = 3600  # Clean up old jobs every hour

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Job Status and Store for Background Processing
# =============================================================================

class JobStatus(str, Enum):
    """Status of an analysis job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalysisJob:
    """Represents an analysis job with its metadata and results."""
    job_id: str
    status: JobStatus
    filename: str
    file_size_bytes: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100 percentage

    def to_status_dict(self) -> Dict[str, Any]:
        """Return status information without full results."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "filename": self.filename,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 1),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "error": self.error,
            "has_result": self.result is not None,
        }


class JobStore:
    """Thread-safe store for analysis jobs with automatic cleanup."""

    def __init__(self, ttl_hours: int = JOB_TTL_HOURS):
        self._jobs: Dict[str, AnalysisJob] = {}
        self._lock = threading.RLock()
        self._ttl = timedelta(hours=ttl_hours)
        self._last_cleanup = datetime.now()

    def create_job(self, filename: str, file_size_bytes: int) -> AnalysisJob:
        """Create a new pending job."""
        job = AnalysisJob(
            job_id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            filename=filename,
            file_size_bytes=file_size_bytes,
            created_at=datetime.now(),
        )
        with self._lock:
            self._maybe_cleanup()
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        progress: Optional[int] = None,
    ) -> Optional[AnalysisJob]:
        """Update a job's status and/or result."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            if status is not None:
                job.status = status
                if status == JobStatus.PROCESSING and job.started_at is None:
                    job.started_at = datetime.now()
                elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job.completed_at = datetime.now()

            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if progress is not None:
                job.progress = min(100, max(0, progress))

            return job

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent jobs (status only, not full results)."""
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True
            )[:limit]
            return [j.to_status_dict() for j in jobs]

    def _maybe_cleanup(self) -> None:
        """Remove expired jobs if enough time has passed."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() < JOB_CLEANUP_INTERVAL_SECONDS:
            return

        self._last_cleanup = now
        cutoff = now - self._ttl
        expired = [
            job_id for job_id, job in self._jobs.items()
            if job.created_at < cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired jobs")


# Global job store and thread pool
job_store = JobStore()
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="analysis_")


# =============================================================================
# Share Code Cache with TTL
# =============================================================================

class ShareCodeCache:
    """LRU cache for share code decoding with TTL."""

    def __init__(self, maxsize: int = 1000, ttl_minutes: int = 60):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._ttl_seconds = ttl_minutes * 60

    def get(self, code: str) -> Optional[Any]:
        """Get a cached result if not expired."""
        with self._lock:
            if code in self._cache:
                result, timestamp = self._cache[code]
                if time.time() - timestamp < self._ttl_seconds:
                    return result
                else:
                    del self._cache[code]
            return None

    def set(self, code: str, result: Any) -> None:
        """Cache a result with current timestamp."""
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._maxsize:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[code] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            now = time.time()
            valid_count = sum(
                1 for _, (_, ts) in self._cache.items()
                if now - ts < self._ttl_seconds
            )
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_count,
                "maxsize": self._maxsize,
                "ttl_minutes": self._ttl_seconds // 60,
            }


# Global share code cache
sharecode_cache = ShareCodeCache(maxsize=1000, ttl_minutes=60)


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="OpenSight API",
    description="CS2 demo analyzer - professional-grade metrics including HLTV 2.0 Rating, KAST%, TTD, and Crosshair Placement",
    version=__version__,
)

# Enable GZip compression for responses > 1KB
# This significantly reduces bandwidth for large JSON responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


class ShareCodeRequest(BaseModel):
    code: str


class ShareCodeResponse(BaseModel):
    match_id: int
    outcome_id: int
    token: int


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    demo_hash: str = Field(..., description="Hash of the demo file")
    player_steam_id: str = Field(..., description="Steam ID of the player")
    metric_name: str = Field(..., description="Name of the metric being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, description="Optional comment")
    correction_value: Optional[float] = Field(None, description="Optional corrected value")


class CoachingFeedbackRequest(BaseModel):
    """Request model for coaching feedback."""
    demo_hash: str
    insight_id: str
    was_helpful: bool
    correction: Optional[str] = None


class RadarRequest(BaseModel):
    """Request model for radar coordinate transformation."""
    map_name: str
    positions: List[dict] = Field(..., description="List of {x, y, z} game coordinates")


# Get the static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface with cache headers."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        content = html_file.read_text()
        return HTMLResponse(
            content=content,
            status_code=200,
            headers={
                # Cache for 5 minutes, revalidate after
                "Cache-Control": "public, max-age=300, must-revalidate",
                "Vary": "Accept-Encoding",
            }
        )
    return HTMLResponse(content="<h1>OpenSight</h1><p>Web interface not found.</p>", status_code=200)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.get("/readiness")
async def readiness():
    """
    Readiness check endpoint for Kubernetes/container orchestration.

    Checks:
    - Disk space: Sufficient free space in temp directory (>100MB)
    - Temp dir writable: Can create and delete temporary files
    - Heavy deps: Core dependencies (demoparser2, pandas, numpy) importable

    Returns 200 if all checks pass, 503 if any check fails.
    This prevents routing traffic to containers that aren't fully ready.
    """
    import tempfile
    import shutil
    import os

    checks = {
        "disk_space": {"status": "unknown", "details": None},
        "temp_writable": {"status": "unknown", "details": None},
        "dependencies": {"status": "unknown", "details": {}},
    }
    all_ready = True

    # Check 1: Disk space (need at least 100MB free for temp files during demo processing)
    try:
        temp_dir = tempfile.gettempdir()
        disk_usage = shutil.disk_usage(temp_dir)
        free_mb = disk_usage.free / (1024 * 1024)
        required_mb = 100

        if free_mb >= required_mb:
            checks["disk_space"]["status"] = "ok"
            checks["disk_space"]["details"] = f"{free_mb:.0f}MB free"
        else:
            checks["disk_space"]["status"] = "fail"
            checks["disk_space"]["details"] = f"Only {free_mb:.0f}MB free, need {required_mb}MB"
            all_ready = False
    except Exception as e:
        checks["disk_space"]["status"] = "fail"
        checks["disk_space"]["details"] = str(e)
        all_ready = False

    # Check 2: Temp directory writable
    try:
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"opensight_readiness_test_{os.getpid()}.tmp")

        # Try to write and read back
        with open(test_file, "w") as f:
            f.write("test")
        with open(test_file, "r") as f:
            content = f.read()
        os.remove(test_file)

        if content == "test":
            checks["temp_writable"]["status"] = "ok"
            checks["temp_writable"]["details"] = temp_dir
        else:
            checks["temp_writable"]["status"] = "fail"
            checks["temp_writable"]["details"] = "Read/write verification failed"
            all_ready = False
    except Exception as e:
        checks["temp_writable"]["status"] = "fail"
        checks["temp_writable"]["details"] = str(e)
        all_ready = False

    # Check 3: Heavy dependencies importable
    heavy_deps = ["demoparser2", "pandas", "numpy"]
    dep_status = {}

    for dep in heavy_deps:
        try:
            __import__(dep)
            dep_status[dep] = "ok"
        except ImportError as e:
            dep_status[dep] = f"fail: {str(e)}"
            all_ready = False

    checks["dependencies"]["details"] = dep_status
    checks["dependencies"]["status"] = "ok" if all(v == "ok" for v in dep_status.values()) else "fail"

    # Build response
    response = {
        "ready": all_ready,
        "version": __version__,
        "checks": checks,
    }

    if all_ready:
        return JSONResponse(content=response, status_code=200)
    else:
        return JSONResponse(content=response, status_code=503)


@app.post("/decode", response_model=ShareCodeResponse)
async def decode_share_code(request: ShareCodeRequest):
    """Decode a CS2 share code to extract match metadata (cached)."""
    # Check cache first
    cached = sharecode_cache.get(request.code)
    if cached is not None:
        return cached

    try:
        from opensight.sharecode import decode_sharecode
        info = decode_sharecode(request.code)
        result = {
            "match_id": info.match_id,
            "outcome_id": info.outcome_id,
            "token": info.token,
        }
        # Cache the result
        sharecode_cache.set(request.code, result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Module not available: {str(e)}")


# =============================================================================
# Background Analysis Worker
# =============================================================================

def _run_analysis(job_id: str, tmp_path: Path, filename: str) -> None:
    """
    Background worker function that performs demo analysis.

    This runs in a thread pool to avoid blocking the event loop.
    """
    try:
        from opensight.parser import DemoParser
        from opensight.analytics import DemoAnalyzer, compute_kill_positions

        job_store.update_job(job_id, status=JobStatus.PROCESSING, progress=10)
        logger.info(f"Job {job_id}: Starting analysis of {filename}")

        # Parse the demo
        parser = DemoParser(tmp_path)
        data = parser.parse()
        # Log parsing progress for debugging
        logger.info(f"Job {job_id}: Parsed {len(data.kills)} kills, {len(data.damages)} damages, {data.num_rounds} rounds")
        job_store.update_job(job_id, progress=40)

        # Run advanced analytics
        analyzer = DemoAnalyzer(data)
        analysis = analyzer.analyze()
        job_store.update_job(job_id, progress=70)

        # Build round-by-round data
        rounds_data = []
        for round_info in data.rounds:
            round_kills = [k for k in data.kills if k.round_num == round_info.round_num]
            rounds_data.append({
                "round_num": round_info.round_num,
                "winner": round_info.winner,
                "reason": round_info.reason,
                "ct_score": round_info.ct_score,
                "t_score": round_info.t_score,
                "kills": len(round_kills),
                "round_type": round_info.round_type or "unknown",
            })

        # Build response
        result = {
            "demo_info": {
                "map": analysis.map_name,
                "duration_seconds": round(data.duration_seconds, 1),
                "duration_minutes": round(data.duration_seconds / 60, 1),
                "tick_rate": data.tick_rate,
                "rounds": analysis.total_rounds,
                "score": f"{analysis.team1_score} - {analysis.team2_score}",
                "ct_score": analysis.team1_score,
                "t_score": analysis.team2_score,
                "player_count": len(analysis.players),
                "total_kills": len(data.kills),
                "total_damage_events": len(data.damages),
                # Parsing metadata
                "parser": "awpy",  # Using awpy library for robust parsing
                "parse_mode": "comprehensive",  # awpy always parses comprehensively
            },
            "rounds": rounds_data,
            "mvp": None,
            "players": {}
        }

        # Add MVP
        mvp = analysis.get_mvp()
        if mvp:
            result["mvp"] = {
                "steam_id": str(mvp.steam_id),
                "name": mvp.name,
                "rating": mvp.hltv_rating,
            }

        job_store.update_job(job_id, progress=80)

        # Add player stats with advanced metrics (sorted by rating)
        for player in analysis.get_leaderboard():
            steam_id = player.steam_id

            # Build weapon stats
            weapon_stats = []
            for weapon, count in sorted(player.weapon_kills.items(), key=lambda x: -x[1]):
                weapon_stats.append({"weapon": weapon, "kills": count})

            result["players"][str(steam_id)] = {
                "name": player.name,
                "team": player.team,
                "stats": {
                    "kills": player.kills,
                    "deaths": player.deaths,
                    "assists": player.assists,
                    "kd_ratio": player.kd_ratio,
                    "kd_diff": player.kd_diff,
                    "headshots": player.headshots,
                    "headshot_pct": player.headshot_percentage,
                    "total_damage": player.total_damage,
                    "adr": player.adr,
                },
                "rating": {
                    "hltv_rating": player.hltv_rating,
                    "impact_rating": player.impact_rating,
                    "kast_percentage": player.kast_percentage,
                    "kills_per_round": player.kills_per_round,
                    "deaths_per_round": player.deaths_per_round,
                    "survival_rate": player.survival_rate,
                    "aim_rating": player.aim_rating,
                    "utility_rating": player.utility_rating,
                    "entry_success_rate": player.entry_success_rate,
                },
                "duels": {
                    "opening_attempts": player.opening_duels.attempts,
                    "opening_wins": player.opening_duels.wins,
                    "opening_losses": player.opening_duels.losses,
                    "opening_win_rate": player.opening_duels.win_rate,
                    "kills_traded": player.trades.kills_traded,
                    "deaths_traded": player.trades.deaths_traded,
                },
                "clutches": {
                    "total_situations": player.clutches.total_situations,
                    "total_wins": player.clutches.total_wins,
                    "1v1": {"attempts": player.clutches.situations_1v1, "wins": player.clutches.wins_1v1},
                    "1v2": {"attempts": player.clutches.situations_1v2, "wins": player.clutches.wins_1v2},
                    "1v3": {"attempts": player.clutches.situations_1v3, "wins": player.clutches.wins_1v3},
                    "1v4": {"attempts": player.clutches.situations_1v4, "wins": player.clutches.wins_1v4},
                    "1v5": {"attempts": player.clutches.situations_1v5, "wins": player.clutches.wins_1v5},
                },
                "multi_kills": {
                    "rounds_with_2k": player.multi_kills.rounds_with_2k,
                    "rounds_with_3k": player.multi_kills.rounds_with_3k,
                    "rounds_with_4k": player.multi_kills.rounds_with_4k,
                    "rounds_with_5k": player.multi_kills.rounds_with_5k,
                },
                "advanced": {
                    "ttd_median_ms": round(player.ttd_median_ms, 1) if player.ttd_median_ms else None,
                    "ttd_mean_ms": round(player.ttd_mean_ms, 1) if player.ttd_mean_ms else None,
                    "ttd_samples": len(player.ttd_values),
                    "prefire_kills": player.prefire_count,
                    "cp_median_error_deg": round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None,
                    "cp_mean_error_deg": round(player.cp_mean_error_deg, 1) if player.cp_mean_error_deg else None,
                    "cp_samples": len(player.cp_values),
                },
                "utility": {
                    "flash_assists": player.utility.flash_assists,
                    "flashbangs_thrown": player.utility.flashbangs_thrown,
                    "enemies_flashed": player.utility.enemies_flashed,
                    "teammates_flashed": player.utility.teammates_flashed,
                    "enemies_flashed_per_flash": round(player.utility.enemies_flashed_per_flash, 2),
                    "avg_blind_time": round(player.utility.avg_blind_time, 2),
                    "he_thrown": player.utility.he_thrown,
                    "he_damage": player.utility.he_damage,
                    "he_team_damage": player.utility.he_team_damage,
                    "he_damage_per_nade": round(player.utility.he_damage_per_nade, 1),
                    "molotov_thrown": player.utility.molotov_thrown,
                    "molotov_damage": player.utility.molotov_damage,
                    "utility_quantity_rating": player.utility_quantity_rating,
                    "utility_quality_rating": player.utility_quality_rating,
                },
                "side_stats": {
                    "ct": {
                        "kills": player.ct_stats.kills,
                        "deaths": player.ct_stats.deaths,
                        "kd_ratio": player.ct_stats.kd_ratio,
                        "adr": player.ct_stats.adr,
                        "rounds_played": player.ct_stats.rounds_played,
                    },
                    "t": {
                        "kills": player.t_stats.kills,
                        "deaths": player.t_stats.deaths,
                        "kd_ratio": player.t_stats.kd_ratio,
                        "adr": player.t_stats.adr,
                        "rounds_played": player.t_stats.rounds_played,
                    },
                },
                "mistakes": {
                    "team_kills": player.mistakes.team_kills,
                    "team_damage": player.mistakes.team_damage,
                    "teammates_flashed": player.mistakes.teammates_flashed,
                    "total_mistakes": player.mistakes.total_mistakes,
                },
                "economy": {
                    "avg_equipment_value": round(player.avg_equipment_value, 0),
                    "eco_rounds": player.eco_rounds,
                    "force_rounds": player.force_rounds,
                    "full_buy_rounds": player.full_buy_rounds,
                    "damage_per_dollar": round(player.damage_per_dollar, 4) if player.damage_per_dollar else 0,
                    "kills_per_dollar": round(player.kills_per_dollar, 6) if player.kills_per_dollar else 0,
                },
                "weapons": weapon_stats,
            }

        job_store.update_job(job_id, progress=90)

        # Add enhanced match-level data
        result["round_timeline"] = [
            {
                "round_num": r.round_num,
                "winner": r.winner,
                "win_reason": r.win_reason,
                "ct_score": r.ct_score,
                "t_score": r.t_score,
                "first_kill": r.first_kill_player,
                "first_death": r.first_death_player,
            }
            for r in analysis.round_timeline
        ]

        result["kill_matrix"] = [
            {
                "attacker": e.attacker_name,
                "victim": e.victim_name,
                "count": e.count,
                "weapons": e.weapons,
            }
            for e in analysis.kill_matrix
        ]

        result["team_stats"] = {
            "trade_rates": analysis.team_trade_rates,
            "opening_rates": analysis.team_opening_rates,
        }

        # Position data for heatmaps (only first 500 to avoid huge response)
        result["heatmap_data"] = {
            "kill_positions": analysis.kill_positions[:500],
            "death_positions": analysis.death_positions[:500],
        }

        # Kill Map data for radar visualization (detailed kill positions)
        result["kill_map"] = compute_kill_positions(data)

        # Grenade trajectory data for utility visualization (limit to 1000 positions)
        result["grenade_data"] = {
            "positions": analysis.grenade_positions[:1000],
            "team_stats": analysis.grenade_team_stats,
        }

        # AI Coaching insights
        result["coaching"] = analysis.coaching_insights

        # Mark job as completed
        job_store.update_job(job_id, status=JobStatus.COMPLETED, result=result, progress=100)
        logger.info(f"Job {job_id}: Analysis complete - {len(result['players'])} players, {analysis.total_rounds} rounds")

    except Exception as e:
        # Log full traceback for debugging in production
        tb_str = traceback.format_exc()
        logger.exception(f"Job {job_id}: Analysis failed")
        # Include traceback in error details for API response
        error_detail = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{tb_str}"
        job_store.update_job(job_id, status=JobStatus.FAILED, error=error_detail)

    finally:
        # Clean up temp file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@app.post("/analyze", status_code=202)
async def analyze_demo(file: UploadFile = File(...)):
    """
    Submit a CS2 demo file for analysis.

    Returns a job ID immediately (202 Accepted). Poll GET /analyze/{job_id}
    to check status and retrieve results when complete.

    Benefits:
    - No request timeouts on large demos
    - Server can rate-limit/queue jobs under load
    - Progress tracking available via status endpoint

    Accepts .dem and .dem.gz files up to 500MB.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a .dem or .dem.gz file. Got: {file.filename}"
        )

    # Verify analysis modules are available
    try:
        from opensight.parser import DemoParser
        from opensight.analytics import DemoAnalyzer
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Demo analysis not available. Missing: {str(e)}"
        )

    # Read and validate file
    content = await file.read()
    file_size_bytes = len(content)
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB"
        )

    if file_size_bytes == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Create job
    job = job_store.create_job(file.filename, file_size_bytes)
    logger.info(f"Created job {job.job_id} for {file.filename} ({file_size_mb:.1f} MB)")

    # Save file to temp location
    suffix = ".dem.gz" if filename_lower.endswith(".dem.gz") else ".dem"
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Submit to thread pool
    executor.submit(_run_analysis, job.job_id, tmp_path, file.filename)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status.value,
            "message": "Analysis job submitted. Poll GET /analyze/{job_id} for status.",
            "status_url": f"/analyze/{job.job_id}",
            "download_url": f"/analyze/{job.job_id}/download",
        }
    )


@app.get("/analyze/{job_id}")
async def get_analysis_status(job_id: str, include_result: bool = Query(default=True)):
    """
    Get the status and result of an analysis job.

    Args:
        job_id: The job ID returned from POST /analyze
        include_result: If True (default), include full results when completed.
                       Set to False for status-only polling.

    Returns:
        - Job status (pending/processing/completed/failed)
        - Progress percentage
        - Full results when completed (if include_result=True)
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    response = job.to_status_dict()

    if include_result and job.status == JobStatus.COMPLETED and job.result:
        response["result"] = job.result

    return JSONResponse(content=response)


@app.get("/analyze/{job_id}/download")
async def download_analysis_result(job_id: str, format: str = Query(default="json")):
    """
    Download analysis results as a file.

    Useful for large results that may timeout in normal responses.
    Supports JSON format with optional NDJSON for streaming.

    Args:
        job_id: The job ID
        format: Output format - 'json' (default) or 'ndjson' (newline-delimited)
    """
    import json

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status.value}"
        )

    if not job.result:
        raise HTTPException(status_code=500, detail="Job completed but no result available")

    filename_base = Path(job.filename).stem

    if format == "ndjson":
        # Stream as newline-delimited JSON (useful for very large results)
        def generate_ndjson():
            # Emit demo_info first
            yield json.dumps({"type": "demo_info", "data": job.result.get("demo_info", {})}) + "\n"
            # Emit each player as separate line
            for steam_id, player_data in job.result.get("players", {}).items():
                yield json.dumps({"type": "player", "steam_id": steam_id, "data": player_data}) + "\n"
            # Emit rounds
            for round_data in job.result.get("rounds", []):
                yield json.dumps({"type": "round", "data": round_data}) + "\n"

        return StreamingResponse(
            generate_ndjson(),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="{filename_base}_analysis.ndjson"'
            }
        )
    else:
        # Standard JSON download
        content = json.dumps(job.result, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename_base}_analysis.json"'
            }
        )


@app.get("/jobs")
async def list_analysis_jobs(limit: int = Query(default=50, le=100)):
    """
    List recent analysis jobs.

    Returns status info for jobs (not full results).
    Use GET /analyze/{job_id} to retrieve full results.
    """
    jobs = job_store.list_jobs(limit=limit)
    return JSONResponse(content={"jobs": jobs, "count": len(jobs)})


@app.get("/about")
async def about():
    """Information about the API and metrics."""
    return {
        "name": "OpenSight",
        "version": __version__,
        "description": "Local CS2 analytics framework - Leetify/Scope.gg style professional-grade metrics",
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
                "hltv_rating": "HLTV 2.0 Rating - industry standard performance metric (1.0 = average)",
                "impact_rating": "Impact component - measures round-winning contributions",
                "kast_percentage": "KAST% - rounds with Kill, Assist, Survived, or Traded",
                "aim_rating": "Leetify-style Aim Rating (0-100, 50 = average) - based on TTD, CP, HS%",
                "utility_rating": "Leetify-style Utility Rating (0-100) - geometric mean of quantity and quality",
                "entry_success_rate": "Percentage of opening duels won",
            },
            "advanced": {
                "ttd_median_ms": "Time to Damage (median) - milliseconds from engagement start to damage dealt",
                "ttd_mean_ms": "Time to Damage (mean)",
                "cp_median_error_deg": "Crosshair Placement error (median) - degrees off-target when engaging",
                "prefire_kills": "Kills where damage was dealt before/instantly upon visibility (prediction shots)",
            },
            "duels": {
                "opening_wins": "First kills of the round won",
                "opening_losses": "First deaths of the round",
                "kills_traded": "Times you avenged a teammate within 5 seconds",
                "deaths_traded": "Times a teammate avenged your death within 5 seconds",
            },
            "utility": {
                "flash_assists": "Kills on enemies you flashed",
                "enemies_flashed": "Total enemies blinded >1.1s",
                "teammates_flashed": "Times you blinded teammates (mistake)",
                "enemies_flashed_per_flash": "Average enemies blinded per flashbang (Leetify metric)",
                "avg_blind_time": "Average enemy blind duration per flash",
                "he_damage": "Total HE grenade damage to enemies",
                "he_damage_per_nade": "Average damage per HE grenade",
                "molotov_damage": "Total molotov/incendiary damage",
                "utility_quantity_rating": "How much utility thrown vs expected (3/round)",
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
            "hltv_rating": "Rating = 0.0073*KAST + 0.3591*KPR - 0.5329*DPR + 0.2372*Impact + 0.0032*ADR + 0.1587*RMK",
            "trade_window": "5.0 seconds (industry standard from Leetify/Stratbook)",
            "ttd": "TTD measures reaction time from first damage to kill. Lower is better. Values under 200ms indicate fast reactions.",
            "crosshair_placement": "CP measures how far your crosshair was from the enemy when engaging. Lower is better. Under 5 degrees is elite-level.",
            "aim_rating": "Composite of TTD, CP, HS%, and prefire rate. 50 = average, adjusted by component performance.",
            "utility_rating": "Geometric mean of Quantity (utility thrown) and Quality (effectiveness). Leetify methodology.",
            "enemies_flashed_threshold": "Only counts enemies blinded for >1.1 seconds (excludes half-blinds).",
        },
        "comparisons": {
            "leetify": "Aim Rating, Utility Rating, and detailed flash stats follow Leetify methodology",
            "scope_gg": "Mistakes tracking and side-based stats follow Scope.gg methodology",
            "hltv": "HLTV 2.0 Rating formula and KAST% calculation",
        },
        "advanced_features": {
            "coaching": "/coaching/* - Adaptive AI coaching with reinforcement learning",
            "patterns": "/patterns/* - Temporal pattern analysis for recurring mistakes",
            "opponent": "/opponent/* - Opponent modeling with HLTV integration",
            "playbook": "/playbook/* - Team playbook generation",
            "realtime": "/realtime/* - Real-time coaching mode with WebSocket",
            "sentiment": "/sentiment/* - Sentiment analysis for voice comms",
            "metrics": "/metrics/* - Custom metric builder",
            "collaboration": "/collab/* - Multi-user collaborative analysis",
        },
        "api_optimization": {
            "async_analysis": {
                "description": "Large demo analysis runs in background threads to avoid timeouts",
                "submit": "POST /analyze - Returns 202 Accepted with job_id",
                "poll": "GET /analyze/{job_id} - Poll for status (pending/processing/completed/failed)",
                "download": "GET /analyze/{job_id}/download - Download results as JSON or NDJSON",
                "list": "GET /jobs - List recent analysis jobs",
            },
            "compression": "GZip middleware enabled for responses >1KB",
            "caching": {
                "sharecode": "Share code decoding cached with 60-minute TTL (1000 entries max)",
                "static_assets": "Cache-Control headers on static files (5 min cache)",
                "stats": "GET /cache/stats - View cache statistics",
                "clear": "POST /cache/clear - Clear all caches",
            },
            "streaming": "NDJSON format available for large result downloads",
        }
    }


# ============================================================================
# Request/Response Models for Advanced Features
# ============================================================================

class PlayerProfileRequest(BaseModel):
    steamid: str
    name: Optional[str] = None
    rank: Optional[str] = None
    role: Optional[str] = None
    map_pool: Optional[List[str]] = None


class CoachingInsightsRequest(BaseModel):
    steamid: str
    player_stats: dict
    map_name: Optional[str] = None


class PatternAnalysisRequest(BaseModel):
    steamid: str
    demo_id: str
    demo_data: dict
    player_stats: dict


class OpponentAnalysisRequest(BaseModel):
    steamid: str
    demo_data: dict
    player_stats: dict


class PlaybookRequest(BaseModel):
    team_name: str
    team_steamids: List[str]
    demo_data: dict


class RealtimeSessionRequest(BaseModel):
    focus_player: Optional[str] = None
    focus_team: str = "ct"


class GameStateUpdate(BaseModel):
    session_id: str
    state_update: dict


class SentimentAnalysisRequest(BaseModel):
    messages: List[dict]
    team_steamids: List[str]
    demo_data: dict


class SingleMessageRequest(BaseModel):
    text: str


class CustomMetricRequest(BaseModel):
    name: str
    formula: str
    description: Optional[str] = ""
    metric_type: Optional[str] = "scalar"
    unit: Optional[str] = ""
    higher_is_better: Optional[bool] = True
    benchmarks: Optional[List[float]] = None


class MetricCalculationRequest(BaseModel):
    metric_id: Optional[str] = None
    player_stats: dict


class CollabSessionRequest(BaseModel):
    demo_id: str
    demo_name: str
    map_name: str
    creator_id: str
    creator_name: str
    title: Optional[str] = ""
    description: Optional[str] = ""
    is_public: Optional[bool] = False
    password: Optional[str] = None


class JoinSessionRequest(BaseModel):
    session_id: str
    user_id: str
    username: str
    password: Optional[str] = None


class AnnotationRequest(BaseModel):
    session_id: str
    user_id: str
    annotation_type: str
    category: str
    tick: int
    round_num: int
    text: Optional[str] = ""
    target_player: Optional[str] = None
    position: Optional[List[float]] = None
    drawing_data: Optional[dict] = None
    tags: Optional[List[str]] = None
    is_private: Optional[bool] = False


# ============================================================================
# Adaptive AI Coaching Endpoints
# ============================================================================

@app.post("/coaching/profile")
async def update_player_profile(request: PlayerProfileRequest):
    """Update or create a player profile for coaching."""
    try:
        from opensight.coaching import get_coach, PlayerRank, PlayerRole

        coach = get_coach()

        rank = None
        role = None

        if request.rank:
            try:
                rank = PlayerRank[request.rank.upper()]
            except KeyError:
                pass

        if request.role:
            try:
                role = PlayerRole(request.role.lower())
            except ValueError:
                pass

        profile = coach.update_profile(
            steamid=request.steamid,
            name=request.name or "",
            rank=rank,
            role=role,
            map_pool=request.map_pool
        )

        return profile.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/coaching/profile/{steamid}")
async def get_player_profile(steamid: str):
    """Get a player's coaching profile."""
    try:
        from opensight.coaching import get_coach

        coach = get_coach()
        profile = coach.get_profile(steamid)
        return profile.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coaching/insights")
async def generate_coaching_insights(request: CoachingInsightsRequest):
    """Generate personalized coaching insights for a player."""
    try:
        from opensight.coaching import generate_coaching_insights

        insights = generate_coaching_insights(
            player_stats=request.player_stats,
            steamid=request.steamid,
            map_name=request.map_name or ""
        )

        return {"insights": insights}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/coaching/practice-plan/{steamid}")
async def get_practice_plan(steamid: str, duration_minutes: int = 30):
    """Generate a personalized practice plan."""
    try:
        from opensight.coaching import get_practice_plan

        plan = get_practice_plan(steamid, duration_minutes)
        return plan

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coaching/suggest-role")
async def suggest_player_role(request: CoachingInsightsRequest):
    """Suggest optimal role based on player statistics."""
    try:
        from opensight.coaching import suggest_player_role, PlayerRank

        rank = PlayerRank.GOLD_NOVA_MASTER
        if "rank" in request.player_stats:
            try:
                rank = PlayerRank[request.player_stats["rank"].upper()]
            except (KeyError, AttributeError):
                pass

        suggestion = suggest_player_role(request.player_stats, rank)
        return suggestion

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Temporal Pattern Analysis Endpoints
# ============================================================================

@app.post("/patterns/analyze")
async def analyze_patterns(request: PatternAnalysisRequest):
    """Analyze a demo for mistake patterns."""
    try:
        from opensight.patterns import analyze_demo_patterns

        mistakes = analyze_demo_patterns(
            steamid=request.steamid,
            demo_id=request.demo_id,
            demo_data=request.demo_data,
            player_stats=request.player_stats
        )

        return {"mistakes_detected": mistakes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns/{steamid}")
async def get_recurring_patterns(steamid: str, min_occurrences: int = 3):
    """Get recurring patterns for a player."""
    try:
        from opensight.patterns import get_player_patterns

        patterns = get_player_patterns(steamid, min_occurrences)
        return {"patterns": patterns}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns/report/{steamid}")
async def get_pattern_report(steamid: str):
    """Get a comprehensive pattern report for a player."""
    try:
        from opensight.patterns import get_pattern_report

        report = get_pattern_report(steamid)
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Opponent Modeling Endpoints
# ============================================================================

@app.post("/opponent/analyze")
async def analyze_opponent(request: OpponentAnalysisRequest):
    """Analyze an opponent from demo data."""
    try:
        from opensight.opponent import analyze_opponent

        profile = analyze_opponent(
            steamid=request.steamid,
            demo_data=request.demo_data,
            player_stats=request.player_stats
        )

        return profile

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/opponent/scouting/{steamid}")
async def get_scouting_report(steamid: str):
    """Get a scouting report for an opponent."""
    try:
        from opensight.opponent import get_scouting_report

        report = get_scouting_report(steamid)
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/opponent/counter-tactics/{steamid}")
async def get_counter_tactics(steamid: str, map_name: Optional[str] = None):
    """Get counter-tactics for an opponent."""
    try:
        from opensight.opponent import get_counter_tactics

        tactics = get_counter_tactics(steamid, map_name)
        return {"tactics": tactics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Team Playbook Generation Endpoints
# ============================================================================

@app.post("/playbook/analyze")
async def analyze_team_demo(request: PlaybookRequest):
    """Analyze a team demo and add to playbook."""
    try:
        from opensight.playbook import analyze_team_demo

        result = analyze_team_demo(
            demo_data=request.demo_data,
            team_steamids=request.team_steamids,
            team_name=request.team_name
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/playbook/{team_name}")
async def get_playbook(team_name: str):
    """Get team playbook."""
    try:
        from opensight.playbook import get_playbook

        playbook = get_playbook(team_name)
        return playbook

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/playbook/report/{team_name}")
async def get_playbook_report(team_name: str):
    """Get comprehensive playbook report."""
    try:
        from opensight.playbook import get_playbook_report

        report = get_playbook_report(team_name)
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/playbook/export/{team_name}")
async def export_playbook(team_name: str, format: str = "json"):
    """Export playbook in specified format."""
    try:
        from opensight.playbook import export_playbook

        content = export_playbook(team_name, format)

        if format == "markdown":
            return HTMLResponse(content=f"<pre>{content}</pre>")

        return JSONResponse(content={"content": content})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Real-Time Coaching Endpoints
# ============================================================================

@app.post("/realtime/session")
async def create_realtime_session(request: RealtimeSessionRequest):
    """Create a new real-time coaching session."""
    try:
        from opensight.realtime import create_coaching_session

        session_info = create_coaching_session(
            focus_player=request.focus_player,
            focus_team=request.focus_team
        )

        return session_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/realtime/update")
async def update_realtime_state(request: GameStateUpdate):
    """Update game state and get alerts."""
    try:
        from opensight.realtime import update_game_state

        alerts = update_game_state(
            session_id=request.session_id,
            state_update=request.state_update
        )

        return {"alerts": alerts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/session/{session_id}")
async def get_realtime_session(session_id: str):
    """Get real-time session information."""
    try:
        from opensight.realtime import get_session_info

        info = get_session_info(session_id)
        if not info:
            raise HTTPException(status_code=404, detail="Session not found")

        return info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/realtime/session/{session_id}")
async def end_realtime_session(session_id: str):
    """End a real-time coaching session."""
    try:
        from opensight.realtime import end_coaching_session

        stats = end_coaching_session(session_id)
        return stats or {"status": "not_found"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Sentiment Analysis Endpoints
# ============================================================================

@app.post("/sentiment/analyze")
async def analyze_team_sentiment(request: SentimentAnalysisRequest):
    """Analyze team morale from voice communications."""
    try:
        from opensight.sentiment import analyze_team_morale

        report = analyze_team_morale(
            messages=request.messages,
            team_steamids=request.team_steamids,
            demo_data=request.demo_data
        )

        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/message")
async def analyze_single_message(request: SingleMessageRequest):
    """Analyze sentiment of a single message."""
    try:
        from opensight.sentiment import analyze_single_message

        result = analyze_single_message(request.text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/chat")
async def analyze_chat_log(request: SentimentAnalysisRequest):
    """Analyze text chat log for sentiment."""
    try:
        from opensight.sentiment import analyze_chat_log

        result = analyze_chat_log(
            chat_messages=request.messages,
            team_steamids=request.team_steamids
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Custom Metric Builder Endpoints
# ============================================================================

@app.post("/metrics/create")
async def create_custom_metric(request: CustomMetricRequest):
    """Create a new custom metric."""
    try:
        from opensight.custom_metrics import create_custom_metric, MetricType

        benchmarks = None
        if request.benchmarks and len(request.benchmarks) == 3:
            benchmarks = tuple(request.benchmarks)

        metric = create_custom_metric(
            name=request.name,
            formula=request.formula,
            description=request.description or "",
            metric_type=MetricType(request.metric_type or "scalar"),
            unit=request.unit or "",
            higher_is_better=request.higher_is_better if request.higher_is_better is not None else True,
            benchmarks=benchmarks
        )

        return metric

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/list")
async def list_custom_metrics():
    """List all custom metrics."""
    try:
        from opensight.custom_metrics import list_custom_metrics

        metrics = list_custom_metrics()
        return {"metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/variables")
async def list_metric_variables():
    """List all available variables for formulas."""
    try:
        from opensight.custom_metrics import list_available_variables

        variables = list_available_variables()
        return {"variables": variables}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/calculate")
async def calculate_metrics(request: MetricCalculationRequest):
    """Calculate custom metrics for a player."""
    try:
        from opensight.custom_metrics import calculate_custom_metrics, calculate_metric

        if request.metric_id:
            result = calculate_metric(request.metric_id, request.player_stats)
            return result
        else:
            results = calculate_custom_metrics(request.player_stats)
            return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/validate")
async def validate_formula(formula: str):
    """Validate a metric formula."""
    try:
        from opensight.custom_metrics import validate_formula

        result = validate_formula(formula)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/install-presets")
async def install_preset_metrics():
    """Install all preset metrics."""
    try:
        from opensight.custom_metrics import install_preset_metrics

        metrics = install_preset_metrics()
        return {"installed": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/metrics/{metric_id}")
async def delete_custom_metric(metric_id: str):
    """Delete a custom metric."""
    try:
        from opensight.custom_metrics import get_builder

        builder = get_builder()
        success = builder.delete_metric(metric_id)

        if not success:
            raise HTTPException(status_code=404, detail="Metric not found")

        return {"status": "deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Radar Map Endpoints
# =============================================================================

@app.get("/maps")
async def list_maps():
    """List all available maps with radar support."""
    try:
        from opensight.radar import MAP_DATA
        return {
            "maps": [
                {
                    "internal_name": name,
                    "display_name": data["name"],
                    "has_radar": True,
                }
                for name, data in MAP_DATA.items()
            ]
        }
    except ImportError:
        return {"maps": [], "error": "Radar module not available"}


@app.get("/maps/{map_name}")
async def get_map_info(map_name: str):
    """Get map metadata and radar information."""
    try:
        from opensight.radar import get_map_metadata, RadarImageManager

        metadata = get_map_metadata(map_name)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Map not found: {map_name}")

        manager = RadarImageManager()
        radar_url = manager.get_radar_url(map_name)

        return {
            "name": metadata.name,
            "internal_name": metadata.internal_name,
            "pos_x": metadata.pos_x,
            "pos_y": metadata.pos_y,
            "scale": metadata.scale,
            "radar_url": radar_url,
            "z_cutoff": metadata.z_cutoff,
            "has_multiple_levels": metadata.z_cutoff is not None,
        }
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Radar module not available: {e}")


@app.post("/radar/transform")
async def transform_coordinates(request: RadarRequest):
    """Transform game coordinates to radar pixel coordinates."""
    try:
        from opensight.radar import CoordinateTransformer

        transformer = CoordinateTransformer(request.map_name)
        results = []

        for pos in request.positions:
            x = pos.get("x", 0)
            y = pos.get("y", 0)
            z = pos.get("z", 0)
            radar_pos = transformer.game_to_radar(x, y, z)
            results.append({
                "game": {"x": x, "y": y, "z": z},
                "radar": {"x": round(radar_pos.x, 1), "y": round(radar_pos.y, 1)},
                "is_upper_level": transformer.is_upper_level(z),
            })

        return {
            "map_name": request.map_name,
            "positions": results,
        }
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Radar module not available: {e}")


# =============================================================================
# HLTV Integration Endpoints
# =============================================================================

@app.get("/hltv/rankings")
async def get_hltv_rankings(top_n: int = Query(default=10, le=30)):
    """Get current world team rankings (cached data)."""
    try:
        from opensight.hltv import HLTVClient
        client = HLTVClient()
        return {"rankings": client.get_world_rankings(top_n)}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"HLTV module not available: {e}")


@app.get("/hltv/map/{map_name}")
async def get_hltv_map_stats(map_name: str):
    """Get map statistics from HLTV data."""
    try:
        from opensight.hltv import get_map_statistics
        stats = get_map_statistics(map_name)
        if not stats:
            raise HTTPException(status_code=404, detail=f"No stats for map: {map_name}")
        return stats
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"HLTV module not available: {e}")


@app.get("/hltv/player/search")
async def search_hltv_player(nickname: str = Query(..., min_length=2)):
    """Search for a player by nickname."""
    try:
        from opensight.hltv import HLTVClient
        client = HLTVClient()
        return {"results": client.search_player(nickname)}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"HLTV module not available: {e}")


@app.post("/hltv/enrich")
async def enrich_analysis(analysis_data: dict = Body(...)):
    """Enrich analysis data with HLTV information."""
    try:
        from opensight.hltv import enrich_match_analysis
        return enrich_match_analysis(analysis_data)
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"HLTV module not available: {e}")


# =============================================================================
# Cache Management Endpoints
# =============================================================================

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics including sharecode cache."""
    result = {
        "sharecode_cache": sharecode_cache.stats(),
        "job_store": {
            "active_jobs": len(job_store.list_jobs(limit=1000)),
        }
    }

    try:
        from opensight.cache import get_cache_stats
        result["analysis_cache"] = get_cache_stats()
    except ImportError:
        result["analysis_cache"] = {"available": False}

    return result


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached data including sharecode cache."""
    # Clear sharecode cache
    sharecode_cache.clear()

    try:
        from opensight.cache import clear_cache as clear_analysis_cache
        clear_analysis_cache()
    except ImportError:
        pass  # Analysis cache module not available

    return {"status": "ok", "message": "All caches cleared"}


# =============================================================================
# Community Feedback Endpoints
# =============================================================================

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on analysis accuracy."""
    try:
        from opensight.feedback import FeedbackDatabase
        db = FeedbackDatabase()
        entry_id = db.add_feedback(
            demo_hash=request.demo_hash,
            player_steam_id=request.player_steam_id,
            metric_name=request.metric_name,
            rating=request.rating,
            comment=request.comment,
            correction_value=request.correction_value,
        )
        return {"status": "ok", "feedback_id": entry_id}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Feedback module not available: {e}")


@app.post("/feedback/coaching")
async def submit_coaching_feedback(request: CoachingFeedbackRequest):
    """Submit feedback on coaching insights."""
    try:
        from opensight.feedback import FeedbackDatabase
        db = FeedbackDatabase()
        entry_id = db.add_coaching_feedback(
            demo_hash=request.demo_hash,
            insight_id=request.insight_id,
            was_helpful=request.was_helpful,
            correction=request.correction,
        )
        return {"status": "ok", "feedback_id": entry_id}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Feedback module not available: {e}")


@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics for model improvement."""
    try:
        from opensight.feedback import FeedbackDatabase
        db = FeedbackDatabase()
        return db.get_stats()
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Feedback module not available: {e}")


# =============================================================================
# Parallel Batch Analysis Endpoints
# =============================================================================

@app.get("/parallel/status")
async def get_parallel_status():
    """Get parallel processing capabilities."""
    try:
        from opensight.parallel import get_system_info, DEFAULT_WORKERS, MAX_WORKERS
        import multiprocessing

        return {
            "available": True,
            "cpu_count": multiprocessing.cpu_count(),
            "default_workers": DEFAULT_WORKERS,
            "max_workers": MAX_WORKERS,
            "system_info": get_system_info(),
        }
    except ImportError as e:
        return {
            "available": False,
            "error": str(e),
        }


# =============================================================================
# 2D Replay Data Endpoints
# =============================================================================

@app.post("/replay/generate")
async def generate_replay_data(
    file: UploadFile = File(...),
    sample_rate: int = Query(default=16, ge=1, le=128, description="Extract every Nth tick"),
):
    """
    Generate 2D replay data from a demo file.

    This extracts player positions and game state at regular intervals
    for use in 2D replay visualization.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a .dem or .dem.gz file"
        )

    try:
        from opensight.parser import DemoParser
        from opensight.replay import ReplayGenerator
        from opensight.radar import CoordinateTransformer
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Replay module not available: {e}")

    tmp_path = None
    try:
        content = await file.read()
        file_size_bytes = len(content)

        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")

        if file_size_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        suffix = ".dem.gz" if filename_lower.endswith(".dem.gz") else ".dem"
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Parse demo
        parser = DemoParser(tmp_path)
        data = parser.parse()

        # Generate replay data
        generator = ReplayGenerator(data, sample_rate=sample_rate)
        replay = generator.generate_full_replay()

        # Get coordinate transformer for radar positions
        transformer = CoordinateTransformer(data.map_name)

        # Convert to response format
        frames = []
        for frame in replay.frames[:10000]:  # Limit frames to prevent huge response
            frame_data = {
                "tick": frame.tick,
                "round": frame.round_num,
                "time_in_round": round(frame.time_in_round, 2),
                "players": [],
                "bomb": None,
            }

            for player in frame.players:
                radar_pos = transformer.game_to_radar(player.x, player.y, player.z)
                frame_data["players"].append({
                    "steam_id": str(player.steam_id),
                    "name": player.name,
                    "team": player.team,
                    "x": round(radar_pos.x, 1),
                    "y": round(radar_pos.y, 1),
                    "yaw": round(player.yaw, 1),
                    "health": player.health,
                    "armor": player.armor,
                    "is_alive": player.is_alive,
                    "weapon": player.active_weapon,
                })

            if frame.bomb:
                bomb_pos = transformer.game_to_radar(frame.bomb.x, frame.bomb.y, frame.bomb.z)
                frame_data["bomb"] = {
                    "x": round(bomb_pos.x, 1),
                    "y": round(bomb_pos.y, 1),
                    "state": frame.bomb.state.value if hasattr(frame.bomb.state, 'value') else frame.bomb.state,
                }

            frames.append(frame_data)

        return {
            "map_name": replay.map_name,
            "total_ticks": replay.total_ticks,
            "tick_rate": replay.tick_rate,
            "sample_rate": sample_rate,
            "total_frames": len(replay.frames),
            "frames_returned": len(frames),
            "rounds": [
                {
                    "round_num": r.round_num,
                    "start_tick": r.start_tick,
                    "end_tick": r.end_tick,
                    "winner": r.winner,
                }
                for r in replay.rounds
            ],
            "frames": frames,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Collaborative Analysis Endpoints
# ============================================================================

@app.post("/collab/session")
async def create_collab_session(request: CollabSessionRequest):
    """Create a new collaborative analysis session."""
    try:
        from opensight.collaboration import create_collaboration_session

        session = create_collaboration_session(
            demo_id=request.demo_id,
            demo_name=request.demo_name,
            map_name=request.map_name,
            creator_id=request.creator_id,
            creator_name=request.creator_name,
            title=request.title or "",
            description=request.description or "",
            is_public=request.is_public or False,
            password=request.password
        )

        return session

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collab/join")
async def join_collab_session(request: JoinSessionRequest):
    """Join a collaborative session."""
    try:
        from opensight.collaboration import join_collaboration_session

        result = join_collaboration_session(
            session_id=request.session_id,
            user_id=request.user_id,
            username=request.username,
            password=request.password
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collab/leave/{session_id}/{user_id}")
async def leave_collab_session(session_id: str, user_id: str):
    """Leave a collaborative session."""
    try:
        from opensight.collaboration import get_manager

        manager = get_manager()
        success = manager.leave_session(session_id, user_id)

        return {"status": "left" if success else "not_found"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collab/annotate")
async def add_collab_annotation(request: AnnotationRequest):
    """Add an annotation to a collaborative session."""
    try:
        from opensight.collaboration import add_annotation

        position = tuple(request.position) if request.position else None

        result = add_annotation(
            session_id=request.session_id,
            user_id=request.user_id,
            annotation_type=request.annotation_type,
            category=request.category,
            tick=request.tick,
            round_num=request.round_num,
            text=request.text or "",
            target_player=request.target_player,
            position=position,
            drawing_data=request.drawing_data,
            tags=request.tags,
            is_private=request.is_private or False
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collab/annotations/{session_id}/{user_id}")
async def get_collab_annotations(session_id: str, user_id: str, round_num: Optional[int] = None):
    """Get annotations from a collaborative session."""
    try:
        from opensight.collaboration import get_session_annotations

        annotations = get_session_annotations(session_id, user_id, round_num)
        return {"annotations": annotations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collab/sessions")
async def list_collab_sessions(user_id: Optional[str] = None):
    """List available collaborative sessions."""
    try:
        from opensight.collaboration import list_sessions

        sessions = list_sessions(user_id)
        return {"sessions": sessions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collab/session/{session_id}")
async def get_collab_session(session_id: str):
    """Get a collaborative session by ID."""
    try:
        from opensight.collaboration import get_manager

        manager = get_manager()
        session = manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return session.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collab/export/{session_id}")
async def export_collab_session(session_id: str, format: str = "json"):
    """Export collaborative session annotations."""
    try:
        from opensight.collaboration import export_session

        content = export_session(session_id, format)

        if not content:
            raise HTTPException(status_code=404, detail="Session not found")

        if format == "markdown":
            return HTMLResponse(content=f"<pre>{content}</pre>")

        return JSONResponse(content={"content": content})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
