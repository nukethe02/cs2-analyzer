"""
Shared utilities for OpenSight API.

Contains validation, security constants, rate limiting, request/response models,
and the JobStore/SharecodeCache classes used across all route modules.
"""

import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

__version__ = "0.3.0"

logger = logging.getLogger(__name__)

# =============================================================================
# Security Constants
# =============================================================================

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = (".dem", ".dem.gz")

# =============================================================================
# Input Validation Patterns
# =============================================================================

DEMO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
STEAM_ID_PATTERN = re.compile(r"^\d{17}$")
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


# =============================================================================
# Job Store
# =============================================================================

JOB_TTL_SECONDS = 3600
JOB_CLEANUP_INTERVAL = 300
MAX_JOBS = 100


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
            valid_jobs = [
                job for job in self._jobs.values() if not job.is_expired(self._ttl_seconds)
            ]
            return valid_jobs


# =============================================================================
# Sharecode Cache
# =============================================================================


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


# =============================================================================
# Lazy Job Store Accessor
# =============================================================================


def _get_job_store():
    """Lazy import of job_store to avoid circular dependency.

    api/__init__.py creates the JobStore instance, and route modules are
    imported by api/__init__.py at module load time. This lazy accessor
    breaks the circular import by deferring the import to call time.
    """
    from opensight.api import job_store

    return job_store


# =============================================================================
# Player Stats Serialization
# =============================================================================


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
        # Engagement Duration
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
        # True TTD / Reaction Time
        "reaction_time_median_ms": (
            round(player.reaction_time_median_ms, 1) if player.reaction_time_median_ms else None
        ),
        "reaction_time_mean_ms": (
            round(player.reaction_time_mean_ms, 1) if player.reaction_time_mean_ms else None
        ),
        "reaction_time_samples": len(player.true_ttd_values),
        "prefire_count": player.prefire_count,
        "prefire_percentage": round(player.prefire_percentage, 1),
        # Legacy TTD aliases
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
        # Utility
        "utility": player.utility.to_dict(),
        # Legacy utility fields
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
        # Victim-side blind metrics
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
        # RWS
        "rws": player.rws,
        "damage_in_won_rounds": player.damage_in_won_rounds,
        "rounds_won": player.rounds_won,
        # Comprehensive aim stats
        "accuracy_all": round(player.accuracy, 1),
        "head_accuracy": round(player.head_hit_rate, 1),
        "spray_accuracy": round(player.spray_accuracy, 1),
        "counter_strafe_pct": round(player.counter_strafe_pct, 1),
    }


def build_player_response(player: Any) -> dict:
    """Build comprehensive player response with all metrics."""
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
        # Advanced metrics
        "ttd_median_ms": (round(player.ttd_median_ms, 1) if player.ttd_median_ms else None),
        "cp_median_error": (
            round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None
        ),
        # Opening duels
        "opening_duels": {
            "attempts": player.opening_duels.attempts,
            "wins": player.opening_duels.wins,
            "losses": player.opening_duels.losses,
            "win_rate": round(player.opening_duels.win_rate, 1),
            "t_side_entries": player.opening_duels.t_side_entries,
            "ct_side_entries": player.opening_duels.ct_side_entries,
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
            "supported_entries": player.opening_duels.supported_entries,
            "unsupported_entries": player.opening_duels.unsupported_entries,
            "supported_deaths": player.opening_duels.supported_deaths,
            "unsupported_deaths": player.opening_duels.unsupported_deaths,
            "dry_peek_rate": round(player.opening_duels.dry_peek_rate, 1),
            "dry_peek_death_rate": round(player.opening_duels.dry_peek_death_rate, 1),
            "map_control_kills": player.opening_duels.map_control_kills,
            "site_kills": player.opening_duels.site_kills,
            "map_control_rate": round(player.opening_duels.map_control_rate, 1),
            "kill_zones": player.opening_duels.kill_zones,
        },
        # Opening engagements
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
        # Entry frags
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
        # Trades
        "trades": {
            "trade_kill_opportunities": player.trades.trade_kill_opportunities,
            "trade_kill_attempts": player.trades.trade_kill_attempts,
            "trade_kill_attempts_pct": round(player.trades.trade_kill_attempts_pct, 1),
            "trade_kill_success": player.trades.trade_kill_success,
            "trade_kill_success_pct": round(player.trades.trade_kill_success_pct, 1),
            "traded_death_opportunities": player.trades.traded_death_opportunities,
            "traded_death_attempts": player.trades.traded_death_attempts,
            "traded_death_attempts_pct": round(player.trades.traded_death_attempts_pct, 1),
            "traded_death_success": player.trades.traded_death_success,
            "traded_death_success_pct": round(player.trades.traded_death_success_pct, 1),
            "avg_time_to_trade_ms": player.trades.avg_time_to_trade_ms,
            "median_time_to_trade_ms": player.trades.median_time_to_trade_ms,
            "traded_entry_kills": player.trades.traded_entry_kills,
            "traded_entry_deaths": player.trades.traded_entry_deaths,
            "kills_traded": player.trades.kills_traded,
            "deaths_traded": player.trades.deaths_traded,
            "trade_rate": round(player.trades.trade_rate, 1),
        },
        # Clutches
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
        # Utility
        "utility": player.utility.to_dict(),
        # Multi-kills
        "multi_kills": {
            "2k": player.multi_kills.rounds_with_2k,
            "3k": player.multi_kills.rounds_with_3k,
            "4k": player.multi_kills.rounds_with_4k,
            "5k": player.multi_kills.rounds_with_5k,
        },
        # Spray transfers
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
        # Comprehensive aim stats
        "aim_stats": {
            "shots_fired": player.shots_fired,
            "shots_hit": player.shots_hit,
            "headshot_hits": player.headshot_hits,
            "spray_shots_fired": player.spray_shots_fired,
            "spray_shots_hit": player.spray_shots_hit,
            "counter_strafe_kills": player.counter_strafe_kills,
            "total_kills_with_velocity": player.total_kills_with_velocity,
            "accuracy_all": round(player.accuracy, 1),
            "head_accuracy": round(player.head_hit_rate, 1),
            "hs_kill_pct": round(player.headshot_percentage, 1),
            "spray_accuracy": round(player.spray_accuracy, 1),
            "counter_strafe_pct": round(player.counter_strafe_pct, 1),
            "time_to_damage_ms": (round(player.ttd_median_ms, 1) if player.ttd_median_ms else None),
            "crosshair_placement_deg": (
                round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None
            ),
            "prefire_percentage": player.prefire_percentage,
        },
        # Side-specific stats
        "side_stats": {
            "ct": player.ct_stats.to_dict(),
            "t": player.t_stats.to_dict(),
        },
        # Lurk statistics
        "lurk_stats": {
            "kills": player.lurk.kills if hasattr(player, "lurk") and player.lurk else 0,
            "deaths": player.lurk.deaths if hasattr(player, "lurk") and player.lurk else 0,
            "rounds_lurking": player.lurk.rounds_lurking
            if hasattr(player, "lurk") and player.lurk
            else 0,
        },
        # Mistake tracking
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
        # Economy efficiency
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


# =============================================================================
# Pydantic Request/Response Models
# =============================================================================


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


class YourMatchResponse(BaseModel):
    """Response model for Your Match data."""

    persona: dict[str, Any] = Field(..., description="Match identity persona")
    top_5: list[dict[str, Any]] = Field(..., description="Top 5 stats with rankings")
    comparison: list[dict[str, Any]] = Field(..., description="This Match vs Average comparison")
    match_count: int = Field(..., description="Number of matches in baseline")


class ScoutingReportRequest(BaseModel):
    """Request body for generating scouting report."""

    opponent_steamids: list[int] = Field(..., description="Steam IDs of opponent players to scout")
    team_name: str = Field(default="Opponent", description="Name of the opponent team")


# =============================================================================
# Rate Limiting Configuration
# =============================================================================

IS_PRODUCTION = os.getenv("SPACE_ID") is not None or os.getenv("PRODUCTION") == "true"

RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "30/minute")
RATE_LIMIT_REPLAY = os.getenv("RATE_LIMIT_REPLAY", "60/minute")
RATE_LIMIT_API = os.getenv("RATE_LIMIT_API", "120/minute")

FORCE_ENABLE = os.getenv("ENABLE_RATE_LIMITING", "").lower() == "true"
FORCE_DISABLE = os.getenv("DISABLE_RATE_LIMITING", "").lower() == "true"

SHOULD_ENABLE_RATE_LIMITING = (IS_PRODUCTION or FORCE_ENABLE) and not FORCE_DISABLE


def get_real_client_ip(request: Request) -> str:
    """
    Get real client IP from X-Forwarded-For header (for reverse proxies like HF Spaces).
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
        if client_ip:
            return client_ip

    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


# Rate limiter instance (set up in __init__.py where app exists)
RATE_LIMITING_ENABLED = False
limiter = None


def rate_limit(limit_string: str):
    """Decorator for rate limiting. No-op if slowapi not available (dev only)."""
    if RATE_LIMITING_ENABLED and limiter:
        return limiter.limit(limit_string)

    def identity(func):
        return func

    return identity
