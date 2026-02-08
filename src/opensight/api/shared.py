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


# Legacy serialization functions (player_stats_to_dict, build_player_response)
# removed — orchestrator handles all player serialization via contract.py.

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
