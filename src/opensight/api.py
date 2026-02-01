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
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import Response

from opensight.ai.llm_client import generate_match_summary

__version__ = "0.3.0"

# Security constants
MAX_FILE_SIZE_MB = 500  # Maximum demo file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = (".dem", ".dem.gz")

# =============================================================================
# Input Validation Patterns
# =============================================================================

# Demo ID: alphanumeric, max 64 characters (UUIDs, hashes, etc.)
DEMO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
# Steam ID: exactly 17 digits (64-bit Steam ID)
STEAM_ID_PATTERN = re.compile(r"^\d{17}$")
# Job ID: UUID format (from job_store)
JOB_ID_PATTERN = re.compile(
    r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"
)


def validate_demo_id(demo_id: str) -> str:
    """Validate demo_id format. Raises HTTPException if invalid."""
    if not demo_id or not DEMO_ID_PATTERN.match(demo_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid demo_id: must be alphanumeric, 1-64 characters",
        )
    return demo_id


def validate_steam_id(steam_id: str) -> str:
    """Validate steam_id format. Raises HTTPException if invalid."""
    if not steam_id or not STEAM_ID_PATTERN.match(steam_id):
        raise HTTPException(status_code=400, detail="Invalid steam_id: must be exactly 17 digits")
    return steam_id


def validate_job_id(job_id: str) -> str:
    """Validate job_id UUID format. Raises HTTPException if invalid."""
    if not job_id or not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id: must be a valid UUID")
    return job_id


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

# =============================================================================
# CORS Configuration - Security hardened
# =============================================================================
# Allowed origins for CORS - restrict to known trusted domains
ALLOWED_ORIGINS = [
    "https://huggingface.co",
    "https://*.huggingface.co",
    "https://*.hf.space",
    "http://localhost:7860",  # Local development
    "http://localhost:3000",  # Local frontend dev
    "http://127.0.0.1:7860",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.(huggingface\.co|hf\.space)",
    allow_credentials=False,  # Disable credentials for cross-origin (more secure)
    allow_methods=["GET", "POST"],  # Only methods we actually use
    allow_headers=["Content-Type", "Accept"],  # Only headers we need
)


# Simple in-memory job store and sharecode cache for tests and lightweight usage
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

# Job TTL configuration (seconds)
JOB_TTL_SECONDS = 3600  # Jobs expire after 1 hour
JOB_CLEANUP_INTERVAL = 300  # Run cleanup every 5 minutes
MAX_JOBS = 100  # Maximum jobs to keep in memory


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
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: int = JOB_TTL_SECONDS) -> bool:
        """Check if job has expired based on TTL."""
        return time.time() - self.created_at > ttl_seconds


class JobStore:
    """In-memory job store for tracking analysis jobs with TTL cleanup."""

    def __init__(self, ttl_seconds: int = JOB_TTL_SECONDS, max_jobs: int = MAX_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds
        self._max_jobs = max_jobs
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start background thread for periodic job cleanup."""

        def cleanup_loop():
            while True:
                time.sleep(JOB_CLEANUP_INTERVAL)
                self._cleanup_expired_jobs()

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def _cleanup_expired_jobs(self) -> None:
        """Remove expired jobs to prevent memory exhaustion."""
        with self._lock:
            expired_ids = [
                jid for jid, job in self._jobs.items() if job.is_expired(self._ttl_seconds)
            ]
            for jid in expired_ids:
                del self._jobs[jid]
            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired jobs")

    def _enforce_max_jobs(self) -> None:
        """Remove oldest jobs if max limit exceeded."""
        if len(self._jobs) >= self._max_jobs:
            # Sort by created_at and remove oldest
            sorted_jobs = sorted(self._jobs.items(), key=lambda x: x[1].created_at)
            jobs_to_remove = len(self._jobs) - self._max_jobs + 1
            for jid, _ in sorted_jobs[:jobs_to_remove]:
                del self._jobs[jid]
            logger.info(f"Removed {jobs_to_remove} oldest jobs (max limit reached)")

    def create_job(self, filename: str, size: int) -> Job:
        """Create a new job and return it."""
        with self._lock:
            self._enforce_max_jobs()
            job_id = str(uuid.uuid4())
            job = Job(job_id=job_id, filename=filename, size=size)
            self._jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        with self._lock:
            job = self._jobs.get(job_id)
            # Don't return expired jobs
            if job and job.is_expired(self._ttl_seconds):
                del self._jobs[job_id]
                return None
            return job

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: int | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Update job fields."""
        with self._lock:
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
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status.value

    def list_jobs(self) -> list[Job]:
        """List all non-expired jobs."""
        with self._lock:
            # Filter out expired jobs
            valid_jobs = [
                job for job in self._jobs.values() if not job.is_expired(self._ttl_seconds)
            ]
            return valid_jobs


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
        "impact_plus_minus": player.impact_plus_minus,
        "survival_rate": player.survival_rate,
        "kills_per_round": player.kills_per_round,
        "deaths_per_round": player.deaths_per_round,
        # Engagement Duration (time from first damage to kill - spray/tracking skill)
        "engagement_duration_median_ms": (
            round(player.engagement_duration_median_ms, 1)
            if player.engagement_duration_median_ms
            else None
        ),
        "engagement_duration_mean_ms": (
            round(player.engagement_duration_mean_ms, 1)
            if player.engagement_duration_mean_ms
            else None
        ),
        "engagement_samples": len(player.engagement_duration_values),
        # True TTD / Reaction Time (visibility to first damage)
        "reaction_time_median_ms": (
            round(player.reaction_time_median_ms, 1) if player.reaction_time_median_ms else None
        ),
        "reaction_time_mean_ms": (
            round(player.reaction_time_mean_ms, 1) if player.reaction_time_mean_ms else None
        ),
        "reaction_time_samples": len(player.true_ttd_values),
        "prefire_count": player.prefire_count,
        "prefire_percentage": round(player.prefire_percentage, 1),
        # Legacy TTD aliases (now returns engagement_duration for backwards compatibility)
        "ttd_median_ms": (round(player.ttd_median_ms, 1) if player.ttd_median_ms else None),
        "ttd_mean_ms": round(player.ttd_mean_ms, 1) if player.ttd_mean_ms else None,
        "ttd_samples": len(player.engagement_duration_values),
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
        # Dry peek tracking
        "dry_peek_rate": player.opening_duels.dry_peek_rate,
        "unsupported_entries": player.opening_duels.unsupported_entries,
        "unsupported_deaths": player.opening_duels.unsupported_deaths,
        "supported_entries": player.opening_duels.supported_entries,
        "supported_deaths": player.opening_duels.supported_deaths,
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
        # Utility (Leetify-style comprehensive)
        "utility": player.utility.to_dict(),
        # Legacy utility fields for backwards compatibility
        "flashbangs_thrown": player.utility.flashbangs_thrown,
        "enemies_flashed": player.utility.enemies_flashed,
        "effective_flashes": player.utility.effective_flashes,
        "avg_enemies_per_flash": player.utility.avg_enemies_per_flash,
        "flash_effectiveness_pct": player.utility.flash_effectiveness_pct,
        "flash_assists": player.utility.flash_assists,
        "flash_assist_pct": player.utility.flash_assist_pct,
        "enemies_flashed_per_round": player.utility.enemies_flashed_per_round,
        "friends_flashed_per_round": player.utility.friends_flashed_per_round,
        "avg_blind_time": player.utility.avg_blind_time,
        # Victim-side blind metrics (Leetify "Avg Blind Time")
        "times_blinded": player.utility.times_blinded,
        "total_time_blinded": player.utility.total_time_blinded,
        "avg_time_blinded": player.utility.avg_time_blinded,
        "he_thrown": player.utility.he_thrown,
        "he_damage": player.utility.he_damage,
        "he_team_damage": player.utility.he_team_damage,
        "avg_he_damage": player.utility.avg_he_damage,
        "smokes_thrown": player.utility.smokes_thrown,
        "molotovs_thrown": player.utility.molotovs_thrown,
        "molotov_damage": player.utility.molotov_damage,
        "utility_quality_rating": player.utility.utility_quality_rating,
        "utility_quantity_rating": player.utility.utility_quantity_rating,
        # Weapon breakdown
        "weapon_kills": player.weapon_kills,
        # RWS (Round Win Shares) - ESEA style
        "rws": player.rws,
        "damage_in_won_rounds": player.damage_in_won_rounds,
        "rounds_won": player.rounds_won,
        # Comprehensive aim stats (Leetify style)
        "accuracy_all": round(player.accuracy, 1),
        "head_accuracy": round(player.head_hit_rate, 1),
        "spray_accuracy": round(player.spray_accuracy, 1),
        "counter_strafe_pct": round(player.counter_strafe_pct, 1),
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
        "survival_rate": round(player.survival_rate, 1)
        if hasattr(player, "survival_rate") and player.survival_rate
        else 0.0,
        # Ratings
        "impact_rating": round(player.impact_rating, 2),
        "hltv_rating": round(player.hltv_rating, 2),
        "impact_plus_minus": round(player.impact_plus_minus, 2),
        # Advanced metrics - TTD and CP
        "ttd_median_ms": (round(player.ttd_median_ms, 1) if player.ttd_median_ms else None),
        "cp_median_error": (
            round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None
        ),
        # Opening duels (nested) with entry TTD, dry peek stats, and zone classification
        "opening_duels": {
            "attempts": player.opening_duels.attempts,
            "wins": player.opening_duels.wins,
            "losses": player.opening_duels.losses,
            "win_rate": round(player.opening_duels.win_rate, 1),
            "t_side_entries": player.opening_duels.t_side_entries,
            "ct_side_entries": player.opening_duels.ct_side_entries,
            # Entry-specific TTD (reaction time on first contact)
            "entry_ttd_median_ms": (
                round(player.opening_duels.entry_ttd_median_ms, 1)
                if player.opening_duels.entry_ttd_median_ms
                else None
            ),
            "entry_ttd_mean_ms": (
                round(player.opening_duels.entry_ttd_mean_ms, 1)
                if player.opening_duels.entry_ttd_mean_ms
                else None
            ),
            # Dry peek tracking (entries without utility support)
            "supported_entries": player.opening_duels.supported_entries,
            "unsupported_entries": player.opening_duels.unsupported_entries,
            "supported_deaths": player.opening_duels.supported_deaths,
            "unsupported_deaths": player.opening_duels.unsupported_deaths,
            "dry_peek_rate": round(player.opening_duels.dry_peek_rate, 1),
            "dry_peek_death_rate": round(player.opening_duels.dry_peek_death_rate, 1),
            # Zone-based classification (map control vs site kills)
            "map_control_kills": player.opening_duels.map_control_kills,
            "site_kills": player.opening_duels.site_kills,
            "map_control_rate": round(player.opening_duels.map_control_rate, 1),
            "kill_zones": player.opening_duels.kill_zones,
        },
        # Opening engagements - who FOUGHT first (damage-based tracking)
        "opening_engagements": {
            "engagement_attempts": player.opening_engagements.engagement_attempts,
            "engagement_wins": player.opening_engagements.engagement_wins,
            "engagement_losses": player.opening_engagements.engagement_losses,
            "engagement_win_rate": round(player.opening_engagements.engagement_win_rate, 1),
            "first_damage_dealt": player.opening_engagements.first_damage_dealt,
            "first_damage_taken": player.opening_engagements.first_damage_taken,
            "first_damage_rate": round(player.opening_engagements.first_damage_rate, 1),
            "opening_damage_total": player.opening_engagements.opening_damage_total,
            "opening_damage_avg": round(player.opening_engagements.opening_damage_avg, 1),
        },
        # Zone-aware entry frags (first kills INTO bombsites)
        "entry_frags": {
            "total_entry_frags": player.entry_frags.total_entry_frags,
            "total_entry_deaths": player.entry_frags.total_entry_deaths,
            "entry_frag_rate": round(player.entry_frags.entry_frag_rate, 1),
            "a_site_entries": player.entry_frags.a_site_entries,
            "a_site_entry_deaths": player.entry_frags.a_site_entry_deaths,
            "a_site_success_rate": round(player.entry_frags.a_site_success_rate, 1),
            "b_site_entries": player.entry_frags.b_site_entries,
            "b_site_entry_deaths": player.entry_frags.b_site_entry_deaths,
            "b_site_success_rate": round(player.entry_frags.b_site_success_rate, 1),
            "entry_round_win_rate": round(player.entry_frags.entry_round_win_rate, 1),
        },
        # Trades (Leetify-style nested stats)
        "trades": {
            # Trade Kill stats (you trading for teammates)
            "trade_kill_opportunities": player.trades.trade_kill_opportunities,
            "trade_kill_attempts": player.trades.trade_kill_attempts,
            "trade_kill_attempts_pct": round(player.trades.trade_kill_attempts_pct, 1),
            "trade_kill_success": player.trades.trade_kill_success,
            "trade_kill_success_pct": round(player.trades.trade_kill_success_pct, 1),
            # Traded Death stats (teammates trading for you)
            "traded_death_opportunities": player.trades.traded_death_opportunities,
            "traded_death_attempts": player.trades.traded_death_attempts,
            "traded_death_attempts_pct": round(player.trades.traded_death_attempts_pct, 1),
            "traded_death_success": player.trades.traded_death_success,
            "traded_death_success_pct": round(player.trades.traded_death_success_pct, 1),
            # Time to trade analysis
            "avg_time_to_trade_ms": player.trades.avg_time_to_trade_ms,
            "median_time_to_trade_ms": player.trades.median_time_to_trade_ms,
            # Entry trades
            "traded_entry_kills": player.trades.traded_entry_kills,
            "traded_entry_deaths": player.trades.traded_entry_deaths,
            # Legacy fields for backwards compatibility
            "kills_traded": player.trades.kills_traded,
            "deaths_traded": player.trades.deaths_traded,
            "trade_rate": round(player.trades.trade_rate, 1),
        },
        # Clutches (nested with breakdown)
        "clutches": {
            "attempts": player.clutches.total_situations,
            "wins": player.clutches.total_wins,
            "win_rate": round(player.clutches.win_rate, 1),
            "1v1": {
                "attempts": player.clutches.v1_attempts,
                "wins": player.clutches.v1_wins,
            },
            "1v2": {
                "attempts": player.clutches.v2_attempts,
                "wins": player.clutches.v2_wins,
            },
            "1v3": {
                "attempts": player.clutches.v3_attempts,
                "wins": player.clutches.v3_wins,
            },
            "1v4": {
                "attempts": player.clutches.v4_attempts,
                "wins": player.clutches.v4_wins,
            },
            "1v5": {
                "attempts": player.clutches.v5_attempts,
                "wins": player.clutches.v5_wins,
            },
            "details": [
                {
                    "round_number": c.round_number,
                    "type": c.type,
                    "outcome": c.outcome,
                    "enemies_killed": c.enemies_killed,
                }
                for c in player.clutches.clutches
            ],
        },
        # Utility (nested, Leetify-style comprehensive)
        "utility": player.utility.to_dict(),
        # Multi-kills
        "multi_kills": {
            "2k": player.multi_kills.rounds_with_2k,
            "3k": player.multi_kills.rounds_with_3k,
            "4k": player.multi_kills.rounds_with_4k,
            "5k": player.multi_kills.rounds_with_5k,
        },
        # Spray transfers (unique OpenSight metric)
        "spray_transfers": (
            player.spray_transfers.to_dict()
            if hasattr(player, "spray_transfers")
            else {
                "double_sprays": 0,
                "triple_sprays": 0,
                "quad_sprays": 0,
                "ace_sprays": 0,
                "total_sprays": 0,
                "total_spray_kills": 0,
                "avg_spray_time_ms": 0,
            }
        ),
        # Comprehensive aim stats (Leetify style)
        "aim_stats": {
            # Raw counts
            "shots_fired": player.shots_fired,
            "shots_hit": player.shots_hit,
            "headshot_hits": player.headshot_hits,
            "spray_shots_fired": player.spray_shots_fired,
            "spray_shots_hit": player.spray_shots_hit,
            "counter_strafe_kills": player.counter_strafe_kills,
            "total_kills_with_velocity": player.total_kills_with_velocity,
            # Computed percentages (Leetify format)
            "accuracy_all": round(player.accuracy, 1),
            "head_accuracy": round(player.head_hit_rate, 1),
            "hs_kill_pct": round(player.headshot_percentage, 1),
            "spray_accuracy": round(player.spray_accuracy, 1),
            "counter_strafe_pct": round(player.counter_strafe_pct, 1),
            # TTD and CP
            "time_to_damage_ms": (round(player.ttd_median_ms, 1) if player.ttd_median_ms else None),
            "crosshair_placement_deg": (
                round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None
            ),
            # Prefire percentage (game sense indicator)
            "prefire_percentage": player.prefire_percentage,
        },
        # Side-specific stats (Leetify CT/T breakdown)
        "side_stats": {
            "ct": player.ct_stats.to_dict(),
            "t": player.t_stats.to_dict(),
        },
        # Lurk statistics (State Machine tracked)
        "lurk_stats": {
            "kills": player.lurk.kills if hasattr(player, "lurk") and player.lurk else 0,
            "deaths": player.lurk.deaths if hasattr(player, "lurk") and player.lurk else 0,
            "rounds_lurking": player.lurk.rounds_lurking
            if hasattr(player, "lurk") and player.lurk
            else 0,
        },
        # Mistake tracking (Scope.gg style)
        "mistakes": {
            "team_kills": player.mistakes.team_kills
            if hasattr(player, "mistakes") and player.mistakes
            else 0,
            "team_damage": player.mistakes.team_damage
            if hasattr(player, "mistakes") and player.mistakes
            else 0,
            "teammates_flashed": player.mistakes.teammates_flashed
            if hasattr(player, "mistakes") and player.mistakes
            else 0,
            "suicides": player.mistakes.suicides
            if hasattr(player, "mistakes") and player.mistakes
            else 0,
            "total_score": player.mistakes.total_mistakes
            if hasattr(player, "mistakes") and player.mistakes
            else 0,
        },
        # Economy efficiency (from domains/economy)
        "economy": {
            "avg_equipment_value": round(player.avg_equipment_value, 0)
            if hasattr(player, "avg_equipment_value") and player.avg_equipment_value
            else 0,
            "damage_per_dollar": round(player.damage_per_dollar, 2)
            if hasattr(player, "damage_per_dollar") and player.damage_per_dollar
            else 0.0,
            "kills_per_dollar": round(player.kills_per_dollar, 4)
            if hasattr(player, "kills_per_dollar") and player.kills_per_dollar
            else 0.0,
            "eco_rounds": player.eco_rounds if hasattr(player, "eco_rounds") else 0,
            "force_rounds": player.force_rounds if hasattr(player, "force_rounds") else 0,
            "full_buy_rounds": player.full_buy_rounds if hasattr(player, "full_buy_rounds") else 0,
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


# =============================================================================
# Security Middleware
# =============================================================================


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next) -> Response:
    """Add comprehensive security headers to all responses."""
    response = await call_next(request)

    # =============================================================================
    # HSTS - Enforce HTTPS (1 year, include subdomains)
    # =============================================================================
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # =============================================================================
    # Content Security Policy - Comprehensive protection
    # =============================================================================
    csp_directives = [
        "default-src 'self'",
        # Scripts: self + inline (needed for embedded JS) + HF CDN
        "script-src 'self' 'unsafe-inline' https://*.huggingface.co",
        # Styles: self + inline (needed for dynamic styles)
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        # Fonts
        "font-src 'self' https://fonts.gstatic.com data:",
        # Images: self + data URIs (for canvas) + blob (for generated content)
        "img-src 'self' data: blob: https:",
        # Connect: API endpoints
        "connect-src 'self' https://*.huggingface.co https://*.hf.space",
        # Frame ancestors: HF Spaces embedding
        "frame-ancestors 'self' https://*.huggingface.co https://huggingface.co https://*.hf.space",
        # Object/embed: none (security)
        "object-src 'none'",
        # Base URI: self only
        "base-uri 'self'",
        # Form action: self only
        "form-action 'self'",
    ]
    response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

    # =============================================================================
    # Other Security Headers
    # =============================================================================
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # XSS protection for older browsers
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Referrer policy - send origin only for cross-origin requests
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Permissions policy - disable unnecessary features
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=()"
    )
    # Prevent caching of sensitive responses
    if "/api/" in request.url.path:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"

    return response


# =============================================================================
# Rate Limiting Configuration
# =============================================================================
import os

# Check if we're in production (HF Spaces sets SPACE_ID env var)
IS_PRODUCTION = os.getenv("SPACE_ID") is not None or os.getenv("PRODUCTION") == "true"

# Rate limiting configuration:
# - DISABLED by default for local development
# - ENABLED automatically in production (SPACE_ID or PRODUCTION=true)
# - Can force enable with ENABLE_RATE_LIMITING=true
# - Can force disable with DISABLE_RATE_LIMITING=true (not recommended in production)
FORCE_ENABLE = os.getenv("ENABLE_RATE_LIMITING", "").lower() == "true"
FORCE_DISABLE = os.getenv("DISABLE_RATE_LIMITING", "").lower() == "true"

# Configurable rate limits via environment variables
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "30/minute")  # Default: 30 uploads/min
RATE_LIMIT_REPLAY = os.getenv("RATE_LIMIT_REPLAY", "60/minute")  # Default: 60 replays/min
RATE_LIMIT_API = os.getenv("RATE_LIMIT_API", "120/minute")  # Default: 120 API calls/min

# Determine if rate limiting should be enabled
SHOULD_ENABLE_RATE_LIMITING = (IS_PRODUCTION or FORCE_ENABLE) and not FORCE_DISABLE

if not SHOULD_ENABLE_RATE_LIMITING:
    RATE_LIMITING_ENABLED = False
    limiter = None
    logger.info("Rate limiting disabled (development mode)")
else:
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        RATE_LIMITING_ENABLED = True
        logger.info(
            f"Rate limiting enabled (upload: {RATE_LIMIT_UPLOAD}, replay: {RATE_LIMIT_REPLAY}, api: {RATE_LIMIT_API})"
        )
    except ImportError as e:
        if IS_PRODUCTION:
            # CRITICAL: Rate limiting is required in production
            raise RuntimeError(
                "SECURITY ERROR: slowapi is required for rate limiting in production. "
                "Install with: pip install slowapi"
            ) from e
        else:
            # Development mode - warn but continue
            RATE_LIMITING_ENABLED = False
            limiter = None
            logger.warning(
                "⚠️  SECURITY WARNING: slowapi not installed - rate limiting DISABLED. "
                "This is acceptable for development but MUST be enabled in production."
            )


def rate_limit(limit_string: str):
    """Decorator for rate limiting. No-op if slowapi not available (dev only)."""
    if RATE_LIMITING_ENABLED and limiter:
        return limiter.limit(limit_string)

    # Return identity decorator if rate limiting disabled (dev mode only)
    def identity(func):
        return func

    return identity


# =============================================================================
# Global Exception Handler
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler to prevent information disclosure.

    Logs the full error internally but returns a generic message to clients.
    """
    # Log the full exception for debugging (server-side only)
    logger.exception(f"Unhandled exception for {request.method} {request.url.path}")

    # Return generic error to client (no internal details)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
        },
    )


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

# Mount static file directories for CSS and JS
if (STATIC_DIR / "css").exists():
    app.mount("/static/css", StaticFiles(directory=STATIC_DIR / "css"), name="css")
if (STATIC_DIR / "js").exists():
    app.mount("/static/js", StaticFiles(directory=STATIC_DIR / "js"), name="js")


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
@rate_limit(RATE_LIMIT_UPLOAD)  # Configurable via RATE_LIMIT_UPLOAD env var
async def analyze_demo(request: Request, file: UploadFile = File(...)):
    """
    Submit a CS2 demo file for analysis.

    Returns a job ID immediately (202 Accepted). Poll GET /analyze/{job_id}
    to check status and retrieve results when complete.

    Benefits:
    - No request timeouts on large demos
    - Server can rate-limit/queue jobs under load
    - Progress tracking available via status endpoint

    Accepts .dem and .dem.gz files up to 500MB.

    Rate limit: 5 requests per minute per IP address.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a .dem or .dem.gz file. Got: {file.filename}",
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

                    # Log timeline summary at debug level
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
                        # Sanitize error message - don't expose internal details
                        # Log full error server-side, return generic message to client
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
    """Get the status of an analysis job."""
    # Validate job_id format
    validate_job_id(job_id)

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
    """Download the results of a completed analysis job."""
    # Validate job_id format
    validate_job_id(job_id)

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")

    result = job.result or {}
    logger.debug(f"Returning analysis result for job {job_id}")

    # --- AI COACHING INTEGRATION ---
    # Generate AI-powered match summary for the top player
    players = result.get("players", [])
    if players and len(players) > 0:
        try:
            # Pick the MVP (first player in sorted list = highest rating)
            hero_player = players[0] if isinstance(players, list) else list(players.values())[0]

            # Extract stats for the AI summary
            player_stats = {
                "kills": hero_player.get("kills", 0),
                "deaths": hero_player.get("deaths", 0),
                "assists": hero_player.get("assists", 0),
                "hltv_rating": hero_player.get("rating", {}).get("hltv_rating", 0.0),
                "adr": hero_player.get("adr", 0.0),
                "headshot_pct": hero_player.get("headshot_pct", 0.0),
                "kast_percentage": hero_player.get("kast_percentage", 0.0),
                "ttd_median_ms": hero_player.get("aim_stats", {}).get("time_to_damage_ms", 0),
                "cp_median_error_deg": hero_player.get("aim_stats", {}).get(
                    "crosshair_placement_deg", 0.0
                ),
                "entry_kills": hero_player.get("opening_duels", {}).get("wins", 0),
                "entry_deaths": hero_player.get("opening_duels", {}).get("losses", 0),
                "trade_kill_success": hero_player.get("trades", {}).get("trade_kill_success", 0),
                "trade_kill_opportunities": hero_player.get("trades", {}).get(
                    "trade_kill_opportunities", 0
                ),
                "clutch_wins": hero_player.get("clutches", {}).get("wins", 0),
                "clutch_attempts": hero_player.get("clutches", {}).get("attempts", 0),
            }

            # Call Anthropic Claude for AI coaching insights
            ai_insight_text = generate_match_summary(player_stats)
            result["ai_summary"] = ai_insight_text
            logger.info(f"AI summary generated for job {job_id}")

        except Exception as e:
            logger.warning(f"AI coaching unavailable: {e}")
            result["ai_summary"] = "Tactical Analysis unavailable (Check ANTHROPIC_API_KEY)."

    return JSONResponse(content=result)


@app.get("/jobs")
async def list_jobs() -> dict[str, Any]:
    jobs = job_store.list_jobs()
    return {
        "jobs": [{"job_id": j.job_id, "status": j.status, "filename": j.filename} for j in jobs]
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
@rate_limit(RATE_LIMIT_REPLAY)  # Configurable via RATE_LIMIT_REPLAY env var
async def generate_replay_data(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    sample_rate: Annotated[int, Query(ge=1, le=128, description="Extract every Nth tick")] = 16,
) -> dict[str, Any]:
    """
    Generate 2D replay data from a demo file.

    This extracts player positions and game state at regular intervals
    for use in 2D replay visualization.

    Rate limit: 3 requests per minute per IP address.
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


# =============================================================================
# Per-Player Positioning Heatmaps
# =============================================================================


@app.get("/api/positioning/{job_id}/{steam_id}")
async def get_player_positioning(job_id: str, steam_id: str) -> dict[str, object]:
    """
    Get per-player positioning heatmap data for a completed analysis.

    Returns:
    - presence_heatmap: 64x64 grid showing where player spends time
    - kills_heatmap: Where they get kills
    - deaths_heatmap: Where they die
    - early_round_heatmap: First 30 seconds positioning
    - late_round_heatmap: After 30 seconds
    - favorite_zones: Top 5 zones by presence
    - danger_zones: Top 5 zones where they die most

    This is more useful than Leetify's team-aggregate heatmaps
    for scouting specific opponents.
    """
    validate_demo_id(job_id)
    validate_steam_id(steam_id)

    # Check job exists and is completed
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job['status']}")

    try:
        from opensight.analysis.positioning import PositioningAnalyzer
        from opensight.core.parser import DemoParser

        # Get demo path from job
        demo_path = job.get("demo_path")
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        # Parse and analyze
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


@app.get("/api/positioning/{job_id}/compare/{steam_id_a}/{steam_id_b}")
async def compare_player_positioning(
    job_id: str, steam_id_a: str, steam_id_b: str
) -> dict[str, object]:
    """
    Compare positioning of two players from a completed analysis.

    Returns:
    - player_a, player_b: Full heatmap data for each player
    - overlap_score: 0-100, how similar their positioning
    - unique_zones_a: Zones where A is but B isn't
    - unique_zones_b: Zones where B is but A isn't
    - shared_zones: Zones where both spend significant time

    Useful for:
    - Scouting opponent tendencies
    - Comparing teammate positioning overlap
    - Finding unique spots each player uses
    """
    validate_demo_id(job_id)
    validate_steam_id(steam_id_a)
    validate_steam_id(steam_id_b)

    # Check job exists and is completed
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job['status']}")

    try:
        from opensight.analysis.positioning import PositioningAnalyzer
        from opensight.core.parser import DemoParser

        # Get demo path from job
        demo_path = job.get("demo_path")
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        # Parse and analyze
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


@app.get("/api/positioning/{job_id}/all")
async def get_all_player_positioning(job_id: str) -> dict[str, object]:
    """
    Get positioning heatmaps for all players in a completed analysis.

    Returns a dictionary keyed by steam_id with each player's full heatmap data.
    """
    validate_demo_id(job_id)

    # Check job exists and is completed
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job['status']}")

    try:
        from opensight.analysis.positioning import PositioningAnalyzer
        from opensight.core.parser import DemoParser

        # Get demo path from job
        demo_path = job.get("demo_path")
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        # Parse and analyze
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
# Trade Chain Visualization - Unique to OpenSight
# =============================================================================


@app.get("/api/trade-chains/{job_id}")
async def get_trade_chains(
    job_id: str,
    round_num: int | None = Query(None, description="Filter by round number"),
    min_chain_length: int = Query(2, ge=2, description="Minimum chain length"),
) -> dict[str, object]:
    """
    Get trade chain analysis for a completed demo.

    Trade chains are sequences of linked trade kills:
    - A kills B (trigger)
    - C kills A (trade for B) within 5 seconds
    - D kills C (trade for A) within 5 seconds
    - ... continues until no trade occurs

    Returns:
    - chains: List of trade chain data with animation frames
    - stats: Aggregate statistics (avg length, max length, team win rates)
    - map_name: Map name for visualization

    This is UNIQUE to OpenSight - no other tool visualizes trade chains.
    """
    validate_demo_id(job_id)

    # Check job exists and is completed
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job['status']}")

    try:
        from opensight.core.parser import DemoParser
        from opensight.domains.combat import CombatAnalyzer

        # Get demo path from job
        demo_path = job.get("demo_path")
        if not demo_path or not Path(demo_path).exists():
            raise HTTPException(status_code=404, detail="Demo file no longer available")

        # Parse and analyze
        parser = DemoParser(Path(demo_path))
        data = parser.parse()
        analyzer = CombatAnalyzer(data)
        result = analyzer.analyze()

        # Get chains and apply filters
        chains = result.trade_chains

        # Filter by round if specified
        if round_num is not None:
            chains = [c for c in chains if c.round_num == round_num]

        # Filter by minimum length
        chains = [c for c in chains if c.chain_length >= min_chain_length]

        # Convert to JSON-serializable format
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


# =============================================================================
# Your Match - Personal Performance Dashboard
# =============================================================================


class YourMatchResponse(BaseModel):
    """Response model for Your Match data."""

    persona: dict[str, Any] = Field(..., description="Match identity persona")
    top_5: list[dict[str, Any]] = Field(..., description="Top 5 stats with rankings")
    comparison: list[dict[str, Any]] = Field(..., description="This Match vs Average comparison")
    match_count: int = Field(..., description="Number of matches in baseline")


# NOTE: Static path routes must come BEFORE parameterized routes to avoid conflicts
# FastAPI matches routes in order, so /api/your-match/baselines/{steam_id}
# must be defined before /api/your-match/{demo_id}/{steam_id}


@app.post("/api/your-match/store")
async def store_match_for_player(
    steam_id: str = Body(..., embed=True),
    demo_hash: str = Body(..., embed=True),
    player_stats: dict[str, Any] = Body(..., embed=True),
    map_name: str | None = Body(None, embed=True),
    result: str | None = Body(None, embed=True),
) -> dict[str, Any]:
    """
    Store a match in player's history and update baselines.

    This should be called after analyzing a demo to track the player's
    performance over time.
    """
    validate_steam_id(steam_id)

    try:
        from opensight.infra.database import get_db

        db = get_db()

        # Store match history
        entry = db.save_match_history_entry(
            steam_id=steam_id,
            demo_hash=demo_hash,
            player_stats=player_stats,
            map_name=map_name,
            result=result,
        )

        if entry is None:
            return {"status": "duplicate", "message": "Match already recorded"}

        # Update baselines
        baselines = db.update_player_baselines(steam_id)

        return {
            "status": "ok",
            "match_id": entry.id,
            "baselines_updated": len(baselines),
        }

    except Exception as e:
        logger.exception("Failed to store match")
        raise HTTPException(status_code=500, detail=f"Failed to store match: {e!s}") from e


@app.get("/api/your-match/baselines/{steam_id}")
async def get_player_baselines_endpoint(steam_id: str) -> dict[str, Any]:
    """
    Get a player's baseline statistics.

    Returns rolling averages for each metric over the last 30 matches.
    """
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


@app.get("/api/your-match/history/{steam_id}")
async def get_player_match_history_endpoint(
    steam_id: str, limit: int = Query(default=30, le=100)
) -> dict[str, Any]:
    """
    Get a player's match history for the Your Match feature.

    Returns recent matches with all tracked metrics.
    """
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


@app.get("/api/your-match/persona/{steam_id}")
async def get_player_persona_endpoint(steam_id: str) -> dict[str, Any]:
    """
    Get a player's current persona based on their match history.

    Analyzes recent performance to determine their playstyle identity.
    """
    validate_steam_id(steam_id)

    try:
        from opensight.analysis.persona import PersonaAnalyzer
        from opensight.infra.database import get_db

        db = get_db()

        # Get recent match history
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

        # Aggregate stats from recent matches
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

        # Also track totals for certain metrics
        aggregated["trade_kill_opportunities"] = sum(
            m.get("trade_kill_opportunities", 0) for m in history
        )
        aggregated["clutch_situations"] = sum(m.get("clutch_situations", 0) for m in history)
        aggregated["entry_attempts"] = sum(m.get("entry_attempts", 0) for m in history)

        # Determine persona
        analyzer = PersonaAnalyzer()
        persona = analyzer.determine_persona(aggregated)

        # Update stored persona
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


@app.get("/api/your-match/trends/{steam_id}")
async def get_player_trends_endpoint(steam_id: str, days: int = 30) -> dict[str, Any]:
    """
    Get performance trends for a player over time.

    Calculates rating/ADR/winrate trends, detects slumps, and identifies
    improvement areas. This feature is 100% FREE - Leetify charges for it.

    Args:
        steam_id: Player's Steam ID (17 digits)
        days: Number of days to analyze (default 30, max 365)

    Returns:
        Performance trend data including:
        - rating_history: List of (date, rating) points
        - rating_trend: "improving" | "declining" | "stable"
        - slump_detected: bool
        - slump_severity: "minor" | "major" | null
        - improvement_areas: List of areas to work on
        - map_performance: Per-map stats
    """
    validate_steam_id(steam_id)

    # Clamp days to reasonable range
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
@app.get("/api/your-match/{demo_id}/{steam_id}")
async def get_your_match(demo_id: str, steam_id: str) -> dict[str, Any]:
    """
    Get personalized match performance data (Leetify-style "Your Match" feature).

    Returns:
    - Match Identity persona
    - Top 5 Stats with progress bars
    - This Match vs Your 30 Match Average comparison

    Args:
        demo_id: Demo hash or job ID (alphanumeric, max 64 chars)
        steam_id: Player's Steam ID (17 digits)
    """
    # Validate inputs
    validate_demo_id(demo_id)
    validate_steam_id(steam_id)

    try:
        from opensight.analysis.persona import PersonaAnalyzer
        from opensight.infra.database import get_db

        db = get_db()

        # Get current match stats from job store
        current_stats = None
        job = job_store.get_job(demo_id)

        if job and job.result:
            # Extract player stats from job result
            players = job.result.get("players", [])
            for player in players:
                if str(player.get("steam_id")) == steam_id:
                    current_stats = player
                    break

        if not current_stats:
            # Try to find in match history
            history = db.get_player_history(steam_id, limit=1)
            if history:
                current_stats = history[0]
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"No stats found for player {steam_id} in demo {demo_id}",
                )

        # Get player baselines
        baselines = db.get_player_baselines(steam_id)

        # Initialize persona analyzer
        analyzer = PersonaAnalyzer(baselines)

        # Determine persona
        persona = analyzer.determine_persona(current_stats)

        # Calculate top 5 stats
        top_5 = analyzer.calculate_top_5_stats(current_stats, baselines)

        # Build comparison table
        comparison = analyzer.build_comparison_table(current_stats, baselines)

        # Get match count from baselines
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


@app.get("/api/players/{steam_id}/metrics")
async def get_player_metrics(steam_id: str, demo_id: str = Query(None, max_length=64)) -> dict:
    """
    Get professional metrics for a player.

    Returns:
    - TTD (Time to Damage): Reaction time metrics (ms)
    - CP (Crosshair Placement): Angular error metrics (degrees)
    - Entry Frags: Opening duel stats
    - Trade Kills: Retribution kill stats
    - Clutch Stats: 1vX situation performance
    """
    # Validate steam_id format
    validate_steam_id(steam_id)

    # Validate demo_id if provided
    if demo_id:
        validate_demo_id(demo_id)

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
# Scouting Engine Endpoints - Multi-demo opponent analysis
# =============================================================================

# In-memory scouting session storage with TTL
SCOUTING_SESSION_TTL = 3600  # 1 hour
MAX_SCOUTING_SESSIONS = 50
MAX_DEMOS_PER_SESSION = 10

# Session storage
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


@app.post("/api/scouting/session")
@rate_limit(RATE_LIMIT_API)
async def create_scouting_session(request: Request) -> dict[str, Any]:
    """
    Create a new scouting session for multi-demo opponent analysis.

    Returns:
        session_id: Unique ID for this scouting session
    """
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


@app.post("/api/scouting/session/{session_id}/add-demo")
@rate_limit(RATE_LIMIT_UPLOAD)
async def add_demo_to_scouting_session(
    request: Request,
    session_id: str,
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """
    Add a demo to an existing scouting session.

    Parses the demo and adds it to the scouting engine for aggregation.

    Returns:
        demo_index: Index of this demo in the session
        map_name: Map name from the demo
        players: List of players in the demo for team selection
    """
    # Validate session exists
    with _scouting_sessions_lock:
        session = _scouting_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Scouting session not found")

        if len(session["demos"]) >= MAX_DEMOS_PER_SESSION:
            raise HTTPException(
                status_code=400, detail=f"Maximum {MAX_DEMOS_PER_SESSION} demos per session"
            )

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be .dem or .dem.gz")

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum {MAX_FILE_SIZE_MB}MB")

    # Parse demo
    suffix = ".dem.gz" if file.filename.endswith(".gz") else ".dem"
    temp_file = None
    try:
        temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.flush()
        temp_path = Path(temp_file.name)
        temp_file.close()

        # Parse demo
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.core.parser import DemoParser

        parser = DemoParser()
        demo_data = parser.parse(temp_path)
        if not demo_data:
            raise HTTPException(status_code=400, detail="Failed to parse demo file")

        # Analyze demo
        analyzer = DemoAnalyzer()
        analysis = analyzer.analyze(demo_data)
        if not analysis:
            raise HTTPException(status_code=400, detail="Failed to analyze demo")

        # Add to scouting engine
        with _scouting_sessions_lock:
            session = _scouting_sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session expired")

            engine = session["engine"]
            demo_info = engine.add_demo(demo_data, analysis)

            # Store demo reference
            session["demos"].append(
                {
                    "filename": file.filename,
                    "map_name": demo_data.map_name,
                    "rounds": analysis.total_rounds,
                }
            )

            # Merge player list (dedupe by steamid)
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
            except Exception:
                pass


class ScoutingReportRequest(BaseModel):
    """Request body for generating scouting report."""

    opponent_steamids: list[int] = Field(..., description="Steam IDs of opponent players to scout")
    team_name: str = Field(default="Opponent", description="Name of the opponent team")


@app.post("/api/scouting/session/{session_id}/report")
@rate_limit(RATE_LIMIT_API)
async def generate_scouting_report(
    request: Request,
    session_id: str,
    body: ScoutingReportRequest,
) -> dict[str, Any]:
    """
    Generate a scouting report for the specified opponent players.

    Aggregates data across all demos in the session and generates:
    - Player profiles with playstyle classification
    - Team tendencies by map
    - Anti-strat recommendations

    Returns:
        Complete TeamScoutReport as JSON
    """
    with _scouting_sessions_lock:
        session = _scouting_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Scouting session not found")

        engine = session["engine"]

    if not body.opponent_steamids:
        raise HTTPException(status_code=400, detail="No opponent players specified")

    try:
        # Set opponent team
        engine.set_opponent_team(body.opponent_steamids, body.team_name)

        # Generate report
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


@app.get("/api/scouting/session/{session_id}")
@rate_limit(RATE_LIMIT_API)
async def get_scouting_session(request: Request, session_id: str) -> dict[str, Any]:
    """
    Get current state of a scouting session.

    Returns:
        Session info including demos loaded and player list
    """
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


@app.delete("/api/scouting/session/{session_id}")
@rate_limit(RATE_LIMIT_API)
async def delete_scouting_session(request: Request, session_id: str) -> dict[str, Any]:
    """
    Delete a scouting session and free resources.
    """
    with _scouting_sessions_lock:
        if session_id in _scouting_sessions:
            del _scouting_sessions[session_id]
            logger.info(f"Deleted scouting session: {session_id}")
            return {"success": True, "deleted": session_id}
        else:
            raise HTTPException(status_code=404, detail="Scouting session not found")


# =============================================================================
# Tactical Analysis Endpoints - CT Rotation Latency
# =============================================================================


@app.post("/tactical/rotations")
@rate_limit(RATE_LIMIT_API)  # Configurable via RATE_LIMIT_API env var
async def analyze_rotations(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """
    Analyze CT rotation latency from a demo file.

    Returns rotation timing metrics for each player:
    - Reaction time: How fast they start moving after contact
    - Travel time: How fast they reach the opposite site
    - Classification: over_rotator, balanced, slow_rotator, anchor

    This helps IGLs identify rotation tendencies and improve team coordination.

    Rate limit: 3 requests per minute per IP address.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="File must be a .dem or .dem.gz file")

    try:
        from opensight.analysis.rotation import CTRotationAnalyzer, get_rotation_summary
        from opensight.core.parser import DemoParser
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Rotation module not available: {e}") from e

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

        # Parse demo with tick data for position tracking
        parser = DemoParser(tmp_path)
        data = parser.parse()

        # Run rotation analysis
        analyzer = CTRotationAnalyzer(data)
        team_stats = analyzer.analyze()
        advice = analyzer.get_rotation_advice()
        summary = get_rotation_summary(team_stats)

        # Build response
        player_stats = []
        for player in team_stats.player_stats:
            player_stats.append(
                {
                    "steam_id": str(player.steam_id),
                    "name": player.name,
                    "rotation_count": player.rotation_count,
                    "avg_reaction_time_ms": round(player.avg_reaction_time_ms, 0)
                    if player.avg_reaction_time_ms
                    else None,
                    "avg_travel_time_ms": round(player.avg_travel_time_ms, 0)
                    if player.avg_travel_time_ms
                    else None,
                    "fastest_rotation_ms": round(player.fastest_rotation_ms, 0)
                    if player.fastest_rotation_ms
                    else None,
                    "slowest_rotation_ms": round(player.slowest_rotation_ms, 0)
                    if player.slowest_rotation_ms
                    else None,
                    "classification": player.classification.value
                    if player.classification
                    else "unknown",
                    "over_rotations": player.over_rotations,
                    "late_rotations": player.late_rotations,
                }
            )

        contact_events = []
        for event in team_stats.contact_events[:50]:  # Limit to 50 events
            contact_events.append(
                {
                    "round_num": event.round_num,
                    "site": event.site,
                    "trigger": event.trigger.value,
                    "tick": event.tick,
                    "t_players_present": event.t_players_present,
                }
            )

        return {
            "map_name": data.map_name,
            "team_stats": {
                "team_avg_reaction_ms": round(team_stats.team_avg_reaction_ms, 0)
                if team_stats.team_avg_reaction_ms
                else None,
                "team_avg_travel_ms": round(team_stats.team_avg_travel_ms, 0)
                if team_stats.team_avg_travel_ms
                else None,
                "total_rotations": team_stats.total_rotations,
                "successful_rotations": team_stats.successful_rotations,
            },
            "player_stats": player_stats,
            "contact_events": contact_events,
            "advice": advice,
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Rotation analysis failed")
        raise HTTPException(status_code=500, detail=f"Rotation analysis failed: {e!s}") from e
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
