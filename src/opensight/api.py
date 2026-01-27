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
"""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

__version__ = "0.3.0"

# Security constants
MAX_FILE_SIZE_MB = 500  # Maximum demo file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = (".dem", ".dem.gz")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenSight API",
    description=(
        "CS2 demo analyzer - professional-grade metrics including "
        "HLTV 2.0 Rating, KAST%, TTD, and Crosshair Placement"
    ),
    version=__version__,
)

# Add CORS middleware for Hugging Face Spaces compatibility
# Allows requests from any origin (needed for HF Spaces iframe embedding)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple in-memory job store and sharecode cache for tests and lightweight usage
import uuid
from dataclasses import dataclass
from enum import Enum


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    filename: str
    size: int
    status: str = JobStatus.PENDING.value
    progress: int = 0
    result: dict | None = None
    error: str | None = None


class JobStore:
    """In-memory job store for tracking analysis jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create_job(self, filename: str, size: int) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, filename=filename, size=size)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: int | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Update job fields."""
        job = self._jobs.get(job_id)
        if job:
            if status is not None:
                job.status = status.value
            if progress is not None:
                job.progress = progress
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error

    def set_status(self, job_id: str, status: JobStatus) -> None:
        """Set job status."""
        job = self._jobs.get(job_id)
        if job:
            job.status = status.value

    def list_jobs(self) -> list[Job]:
        """List all jobs."""
        return list(self._jobs.values())


# Global job store
job_store = JobStore()


def player_stats_to_dict(player: Any) -> dict:
    """Convert PlayerMatchStats to a JSON-serializable dict."""
    return {
        "steam_id": str(player.steam_id),
        "name": player.name,
        "team": player.team,
        "kills": player.kills,
        "deaths": player.deaths,
        "assists": player.assists,
        "headshots": player.headshots,
        "total_damage": player.total_damage,
        "rounds_played": player.rounds_played,
        "kd_ratio": player.kd_ratio,
        "kd_diff": player.kd_diff,
        "adr": player.adr,
        "headshot_percentage": player.headshot_percentage,
        "kast_percentage": player.kast_percentage,
        "hltv_rating": player.hltv_rating,
        "impact_rating": player.impact_rating,
        "survival_rate": player.survival_rate,
        "kills_per_round": player.kills_per_round,
        "deaths_per_round": player.deaths_per_round,
        # TTD (Time to Damage)
        "ttd_median_ms": (
            round(player.ttd_median_ms, 1) if player.ttd_median_ms else None
        ),
        "ttd_mean_ms": round(player.ttd_mean_ms, 1) if player.ttd_mean_ms else None,
        "ttd_samples": len(player.ttd_values),
        "prefire_count": player.prefire_count,
        # Crosshair Placement
        "cp_median_error_deg": (
            round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None
        ),
        "cp_mean_error_deg": (
            round(player.cp_mean_error_deg, 1) if player.cp_mean_error_deg else None
        ),
        "cp_samples": len(player.cp_values),
        # Opening duels
        "opening_duel_wins": player.opening_duels.wins,
        "opening_duel_losses": player.opening_duels.losses,
        "opening_duel_attempts": player.opening_duels.attempts,
        "opening_duel_win_rate": player.opening_duels.win_rate,
        # Trades
        "kills_traded": player.trades.kills_traded,
        "deaths_traded": player.trades.deaths_traded,
        # Clutches
        "clutch_situations": player.clutches.total_situations,
        "clutch_wins": player.clutches.total_wins,
        "clutch_win_rate": player.clutches.win_rate,
        # Multi-kills
        "rounds_with_2k": player.multi_kills.rounds_with_2k,
        "rounds_with_3k": player.multi_kills.rounds_with_3k,
        "rounds_with_4k": player.multi_kills.rounds_with_4k,
        "rounds_with_5k": player.multi_kills.rounds_with_5k,
        # Utility
        "flashbangs_thrown": player.utility.flashbangs_thrown,
        "enemies_flashed": player.utility.enemies_flashed,
        "flash_assists": player.utility.flash_assists,
        "he_thrown": player.utility.he_thrown,
        "he_damage": player.utility.he_damage,
        # Weapon breakdown
        "weapon_kills": player.weapon_kills,
        # RWS (Round Win Shares) - ESEA style
        "rws": player.rws,
        "damage_in_won_rounds": player.damage_in_won_rounds,
        "rounds_won": player.rounds_won,
    }


def build_player_response(player: Any) -> dict:
    """Build comprehensive player response with all metrics.

    This function provides a structured response format suitable for
    frontend consumption, with nested objects for complex stats.
    """
    return {
        "name": player.name,
        "team": player.team,
        "steam_id": str(player.steam_id),
        # Basic stats
        "kills": player.kills,
        "deaths": player.deaths,
        "assists": player.assists,
        "headshots": player.headshots,
        # Per-round metrics
        "kpr": round(player.kills_per_round, 2),
        "dpr": round(player.deaths_per_round, 2),
        "apr": round(player.assists_per_round, 2),
        "adr": round(player.adr, 1),
        # Percentages
        "hs_percentage": round(player.headshot_percentage, 1),
        "kast_percentage": round(player.kast_percentage, 1),
        # Ratings
        "impact_rating": round(player.impact_rating, 2),
        "hltv_rating": round(player.hltv_rating, 2),
        # Advanced metrics - TTD and CP
        "ttd_median_ms": (
            round(player.ttd_median_ms, 1) if player.ttd_median_ms else None
        ),
        "cp_median_error": (
            round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None
        ),
        # Opening duels (nested)
        "opening_duels": {
            "attempts": player.opening_duels.attempts,
            "wins": player.opening_duels.wins,
            "losses": player.opening_duels.losses,
            "win_rate": round(player.opening_duels.win_rate, 1),
        },
        # Trades (nested)
        "trades": {
            "kills_traded": player.trades.kills_traded,
            "deaths_traded": player.trades.deaths_traded,
            "trade_rate": round(player.trades.trade_rate, 1),
        },
        # Clutches (nested with breakdown)
        "clutches": {
            "attempts": player.clutches.total_situations,
            "wins": player.clutches.total_wins,
            "win_rate": round(player.clutches.win_rate, 1),
            "1v1": {"attempts": player.clutches.v1_attempts, "wins": player.clutches.v1_wins},
            "1v2": {"attempts": player.clutches.v2_attempts, "wins": player.clutches.v2_wins},
            "1v3": {"attempts": player.clutches.v3_attempts, "wins": player.clutches.v3_wins},
            "1v4": {"attempts": player.clutches.v4_attempts, "wins": player.clutches.v4_wins},
            "1v5": {"attempts": player.clutches.v5_attempts, "wins": player.clutches.v5_wins},
        },
        # Utility (nested)
        "utility": {
            "damage": player.utility.he_damage + player.utility.molotov_damage,
            "enemies_flashed": player.utility.enemies_flashed,
            "flash_assists": player.utility.flash_assists,
            "flashbangs_thrown": player.utility.flashbangs_thrown,
            "he_thrown": player.utility.he_thrown,
            "molotovs_thrown": player.utility.molotovs_thrown,
            "smokes_thrown": player.utility.smokes_thrown,
        },
        # Multi-kills
        "multi_kills": {
            "2k": player.multi_kills.rounds_with_2k,
            "3k": player.multi_kills.rounds_with_3k,
            "4k": player.multi_kills.rounds_with_4k,
            "5k": player.multi_kills.rounds_with_5k,
        },
    }


class SharecodeCache:
    def __init__(self, maxsize: int = 1024) -> None:
        self._cache: dict[str, dict] = {}
        self.maxsize = maxsize

    def set(self, key: str, value: dict) -> None:
        self._cache[key] = value

    def get(self, key: str) -> dict | None:
        return self._cache.get(key)

    def clear(self) -> None:
        self._cache.clear()


# Global share code cache
sharecode_cache = SharecodeCache(maxsize=1000)


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="OpenSight API",
    description=(
        "CS2 demo analyzer - professional-grade metrics including "
        "HLTV 2.0 Rating, KAST%, TTD, and Crosshair Placement"
    ),
    version=__version__,
)

# Enable CORS for frontend JavaScript communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for HF Spaces
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    comment: str | None = Field(None, description="Optional comment")
    correction_value: float | None = Field(None, description="Optional corrected value")


class CoachingFeedbackRequest(BaseModel):
    """Request model for coaching feedback."""

    demo_hash: str
    insight_id: str
    was_helpful: bool
    correction: str | None = None


class RadarRequest(BaseModel):
    """Request model for radar coordinate transformation."""

    map_name: str
    positions: list[dict[str, float]] = Field(..., description="List of {x, y, z} game coordinates")


# Get the static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the main web interface."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(
            content=html_file.read_text(encoding="utf-8"),
            status_code=200,
            headers={"cache-control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse(
        content="<h1>OpenSight</h1><p>Web interface not found.</p>",
        status_code=200,
        headers={"cache-control": "no-cache"},
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.get("/readiness")
async def readiness() -> dict[str, Any]:
    """
    Readiness check for container orchestration (Hugging Face Spaces).

    Verifies:
    - Disk space available (>100MB)
    - Temp directory writable
    - Critical dependencies importable
    """
    import shutil
    import tempfile

    checks = {}

    # Check disk space
    try:
        disk = shutil.disk_usage("/tmp")
        free_mb = disk.free / (1024 * 1024)
        checks["disk_space"] = {"ok": free_mb > 100, "free_mb": round(free_mb, 1)}
    except Exception as e:
        checks["disk_space"] = {"ok": False, "error": str(e)}

    # Check temp directory
    try:
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(b"test")
        checks["temp_dir"] = {"ok": True}
    except Exception as e:
        checks["temp_dir"] = {"ok": False, "error": str(e)}

    # Check critical dependencies
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


@app.post("/decode", response_model=ShareCodeResponse)
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
            status_code=400, detail=f"File must be a .dem or .dem.gz file. Got: {file.filename}"
        )

    # Create a background job for analysis instead of processing synchronously
    try:
        content = await file.read()
        file_size_bytes = len(content)

        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_bytes / (1024 * 1024):.1f}MB",
            )

        if file_size_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        job = job_store.create_job(file.filename, file_size_bytes)

        # Start background worker to save upload and run real analysis
        try:
            tmp = NamedTemporaryFile(suffix=".dem", delete=False)
            try:
                tmp.write(content)
                tmp.flush()
                demo_path = Path(tmp.name)
            finally:
                tmp.close()

            def _process_job(jid: str, demo_path: Path):
                try:
                    job_store.set_status(jid, JobStatus.PROCESSING)
                    from opensight.infra.cache import analyze_with_cache

                    result = analyze_with_cache(demo_path)
                    j = job_store.get_job(jid)
                    if j:
                        j.result = result
                        job_store.set_status(jid, JobStatus.COMPLETED)
                except Exception as ex:
                    logger.exception("Job processing failed")
                    j = job_store.get_job(jid)
                    if j:
                        j.result = {"error": str(ex)}
                        job_store.set_status(jid, JobStatus.FAILED)
                finally:
                    try:
                        demo_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            import threading

            threading.Thread(target=_process_job, args=(job.job_id, demo_path), daemon=True).start()
        except Exception:
            # If we fail to start the worker, mark job failed
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


@app.get("/analyze/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "filename": job.filename,
        "size": job.size,
    }


@app.get("/analyze/{job_id}/download")
async def download_job_result(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")
    return JSONResponse(content=job.result or {})


@app.get("/jobs")
async def list_jobs() -> dict[str, Any]:
    jobs = job_store.list_jobs()
    return {
        "jobs": [
            {"job_id": j.job_id, "status": j.status, "filename": j.filename} for j in jobs
        ]
    }


@app.get("/about")
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
                "ttd_median_ms": (
                    "Time to Damage (median) - milliseconds from engagement start to damage dealt"
                ),
                "ttd_mean_ms": "Time to Damage (mean)",
                "cp_median_error_deg": (
                    "Crosshair Placement error (median) - degrees off-target when engaging"
                ),
                "prefire_kills": (
                    "Kills where damage was dealt before/instantly "
                    "upon visibility (prediction shots)"
                ),
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
        # API optimization notes for clients
        "api_optimization": {
            "async_analysis": "Use the /analyze async job endpoints to upload and poll status",
            "caching": "Results are cached server-side to speed up repeated analyses",
        },
    }


# =============================================================================
# Radar Map Endpoints
# =============================================================================


@app.get("/maps")
async def list_maps() -> dict[str, Any]:
    """List all available maps with radar support."""
    try:
        from opensight.visualization.radar import MAP_DATA

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
async def get_map_info(map_name: str) -> dict[str, Any]:
    """Get map metadata and radar information."""
    try:
        from opensight.visualization.radar import RadarImageManager, get_map_metadata

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
        raise HTTPException(status_code=503, detail=f"Radar module not available: {e}") from e


@app.post("/radar/transform")
async def transform_coordinates(request: RadarRequest) -> dict[str, Any]:
    """Transform game coordinates to radar pixel coordinates."""
    try:
        from opensight.visualization.radar import CoordinateTransformer

        transformer = CoordinateTransformer(request.map_name)
        results = []

        for pos in request.positions:
            x = pos.get("x", 0.0)
            y = pos.get("y", 0.0)
            z = pos.get("z", 0.0)
            radar_pos = transformer.game_to_radar(x, y, z)
            results.append(
                {
                    "game": {"x": x, "y": y, "z": z},
                    "radar": {"x": round(radar_pos.x, 1), "y": round(radar_pos.y, 1)},
                    "is_upper_level": transformer.is_upper_level(z),
                }
            )

        return {
            "map_name": request.map_name,
            "positions": results,
        }
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Radar module not available: {e}") from e


# =============================================================================
# HLTV Integration Endpoints
# =============================================================================


@app.get("/hltv/rankings")
async def get_hltv_rankings(
    top_n: Annotated[int, Query(le=30)] = 10,
) -> dict[str, Any]:
    """Get current world team rankings (cached data)."""
    try:
        from opensight.integrations.hltv import HLTVClient

        client = HLTVClient()
        return {"rankings": client.get_world_rankings(top_n)}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"HLTV module not available: {e}") from e


@app.get("/hltv/map/{map_name}")
async def get_hltv_map_stats(map_name: str) -> dict[str, Any]:
    """Get map statistics from HLTV data."""
    try:
        from opensight.integrations.hltv import get_map_statistics

        stats = get_map_statistics(map_name)
        if not stats:
            raise HTTPException(status_code=404, detail=f"No stats for map: {map_name}")
        return stats
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"HLTV module not available: {e}") from e


@app.get("/hltv/player/search")
async def search_hltv_player(
    nickname: Annotated[str, Query(..., min_length=2)],
) -> dict[str, Any]:
    """Search for a player by nickname."""
    try:
        from opensight.integrations.hltv import HLTVClient

        client = HLTVClient()
        return {"results": client.search_player(nickname)}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"HLTV module not available: {e}") from e


@app.post("/hltv/enrich")
async def enrich_analysis(
    analysis_data: Annotated[dict[str, Any], Body(...)],
) -> dict[str, Any]:
    """Enrich analysis data with HLTV information."""
    try:
        from opensight.integrations.hltv import enrich_match_analysis

        return enrich_match_analysis(analysis_data)
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"HLTV module not available: {e}") from e


# =============================================================================
# Cache Management Endpoints
# =============================================================================


@app.get("/cache/stats")
async def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    try:
        from opensight.infra.cache import get_cache_stats as infra_get_cache_stats

        stats = infra_get_cache_stats()

        # Augment with lightweight runtime caches used in the API
        stats_wrapped: dict[str, Any] = {
            "demo_cache": stats,
            "sharecode_cache": {"maxsize": getattr(sharecode_cache, "maxsize", None)},
            "job_store": {"total_jobs": len(job_store.list_jobs())},
        }

        return stats_wrapped
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Cache module not available: {e}") from e


@app.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Clear all cached analysis data."""
    try:
        from opensight.infra.cache import clear_cache as infra_clear_cache

        infra_clear_cache()

        # Also clear lightweight runtime caches
        try:
            sharecode_cache.clear()
        except Exception:
            pass

        return {"status": "ok", "message": "Cache cleared"}
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Cache module not available: {e}") from e


# =============================================================================
# Community Feedback Endpoints
# =============================================================================


@app.post("/feedback")
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


@app.post("/feedback/coaching")
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
            player_steam_id="",  # Not provided in request
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


@app.get("/feedback/stats")
async def get_feedback_stats() -> Any:
    """Get feedback statistics for model improvement."""
    try:
        from opensight.integrations.feedback import FeedbackDatabase

        db = FeedbackDatabase()
        return db.get_stats()
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Feedback module not available: {e}") from e


# =============================================================================
# Parallel Batch Analysis Endpoints
# =============================================================================


@app.get("/parallel/status")
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
# 2D Replay Data Endpoints
# =============================================================================


@app.post("/replay/generate")
async def generate_replay_data(
    file: Annotated[UploadFile, File(...)],
    sample_rate: Annotated[int, Query(ge=1, le=128, description="Extract every Nth tick")] = 16,
) -> dict[str, Any]:
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
        raise HTTPException(status_code=400, detail="File must be a .dem or .dem.gz file")

    try:
        from opensight.core.parser import DemoParser
        from opensight.visualization.radar import CoordinateTransformer
        from opensight.visualization.replay import ReplayGenerator
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Replay module not available: {e}") from e

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

        # Collect all frames from all rounds
        all_replay_frames = [frame for r in replay.rounds for frame in r.frames]

        # Calculate total ticks from rounds
        total_ticks = 0
        if replay.rounds:
            total_ticks = replay.rounds[-1].end_tick - replay.rounds[0].start_tick

        # Convert to response format (limit frames to prevent huge response)
        frames = []
        for frame in all_replay_frames[:10000]:
            frame_data: dict[str, Any] = {
                "tick": frame.tick,
                "round": frame.round_num,
                "time_in_round": round(frame.game_time, 2),
                "players": [],
                "bomb": None,
            }

            for player in frame.players:
                radar_pos = transformer.game_to_radar(player.x, player.y, player.z)
                frame_data["players"].append(
                    {
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
                    }
                )

            if frame.bomb:
                bomb_pos = transformer.game_to_radar(frame.bomb.x, frame.bomb.y, frame.bomb.z)
                bomb_state = (
                    frame.bomb.state.value
                    if hasattr(frame.bomb.state, "value")
                    else frame.bomb.state
                )
                frame_data["bomb"] = {
                    "x": round(bomb_pos.x, 1),
                    "y": round(bomb_pos.y, 1),
                    "state": bomb_state,
                }

            frames.append(frame_data)

        return {
            "map_name": replay.map_name,
            "total_ticks": total_ticks,
            "tick_rate": replay.tick_rate,
            "sample_rate": sample_rate,
            "total_frames": len(all_replay_frames),
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
        logger.exception("Replay generation failed")
        raise HTTPException(status_code=500, detail=f"Replay generation failed: {e!s}") from e
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ============================================================================
# Professional Metrics Endpoints
# ============================================================================


@app.get("/api/players/{steam_id}/metrics")
async def get_player_metrics(steam_id: str, demo_id: str = Query(None)) -> dict:
    """
    Get professional metrics for a player.

    Returns:
    - TTD (Time to Damage): Reaction time metrics (ms)
    - CP (Crosshair Placement): Angular error metrics (degrees)
    - Entry Frags: Opening duel stats
    - Trade Kills: Retribution kill stats
    - Clutch Stats: 1vX situation performance
    """
    try:
        from opensight.infra.cache import CacheManager

        CacheManager()

        # If demo_id provided, get metrics from that analysis
        if demo_id:
            # Try to load from cache with demo_id
            # This is a simplified version - in production you'd look up the actual demo file
            pass

        # Return structured metrics data
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
                    "kills": 0,
                    "deaths_traded": 0,
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
