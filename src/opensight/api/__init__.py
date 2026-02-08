"""
OpenSight Web API

FastAPI application for CS2 demo analysis with professional-grade metrics.

This package exposes:
- app: The FastAPI application (used by uvicorn, Dockerfile, server.py)
- job_store: In-memory job store (used by test_api.py)
- sharecode_cache: Share code cache (used by test_api.py)
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from opensight.api.shared import (
    ALLOWED_EXTENSIONS,  # noqa: F401
    IS_PRODUCTION,
    MAX_FILE_SIZE_BYTES,  # noqa: F401
    SHOULD_ENABLE_RATE_LIMITING,
    CoachingFeedbackRequest,  # noqa: F401
    FeedbackRequest,  # noqa: F401
    Job,
    JobStatus,  # noqa: F401
    JobStore,
    SharecodeCache,
    ShareCodeRequest,  # noqa: F401
    ShareCodeResponse,  # noqa: F401
    __version__,
    get_real_client_ip,
    rate_limit,  # noqa: F401
    validate_demo_id,  # noqa: F401
    validate_steam_id,  # noqa: F401
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI App Creation
# =============================================================================

app = FastAPI(
    title="OpenSight API",
    description=(
        "CS2 demo analyzer - professional-grade metrics including "
        "HLTV 2.0 Rating, KAST%, TTD, and Crosshair Placement"
    ),
    version=__version__,
)

# =============================================================================
# CORS Configuration
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global Instances (test_api.py imports these)
# =============================================================================


class _PersistentJobStoreAdapter:
    """Adapts PersistentJobStore (dict-based) to the in-memory JobStore interface.

    Route modules expect get_job() to return shared.Job dataclass objects with
    .job_id, .status, .result attributes.  PersistentJobStore returns plain dicts
    and uses update_status() instead of update_job()/set_status().
    This thin adapter bridges both interfaces so routes work unchanged.
    """

    def __init__(self, persistent_store):
        self._store = persistent_store

    def create_job(self, filename: str, size: int) -> Job:
        result = self._store.create_job(filename, size)
        return Job(
            job_id=result["job_id"],
            filename=filename,
            size=size,
            status=result["status"],
        )

    def get_job(self, job_id: str) -> Job | None:
        result = self._store.get_job(job_id)
        if result is None:
            return None
        job = Job(
            job_id=result["job_id"],
            filename=result.get("filename", ""),
            size=result.get("file_size", 0),
            status=result.get("status", "pending"),
        )
        if "result" in result and result["result"] is not None:
            job.result = result["result"]
        if "error" in result:
            job.error = result["error"]
        return job

    def set_status(self, job_id: str, status: JobStatus) -> None:
        self._store.update_status(job_id, status=status.value)

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: int | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        self._store.update_status(
            job_id,
            status=status.value if status is not None else None,
            result=result,
            error=error,
        )

    def list_jobs(self) -> list[Job]:
        results = self._store.list_jobs()
        return [
            Job(
                job_id=r["job_id"],
                filename=r.get("filename", ""),
                size=r.get("file_size", 0),
                status=r.get("status", "pending"),
            )
            for r in results
        ]


def _create_job_store():
    """Try PersistentJobStore (database-backed), fall back to in-memory."""
    try:
        from opensight.infra.job_store import PersistentJobStore

        persistent = PersistentJobStore()
        # Smoke-test: verify DB connection works
        persistent.list_jobs(limit=1)
        logger.info("Using persistent job store (database-backed)")
        return _PersistentJobStoreAdapter(persistent)
    except Exception as exc:
        logger.warning("Persistent job store unavailable (%s), using in-memory fallback", exc)
        return JobStore()


job_store = _create_job_store()
sharecode_cache = SharecodeCache(maxsize=1000)

# =============================================================================
# GZip Middleware
# =============================================================================

app.add_middleware(GZipMiddleware, minimum_size=1000)

# =============================================================================
# Security Middleware
# =============================================================================


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next) -> Response:
    """Add comprehensive security headers to all responses."""
    response = await call_next(request)

    # HSTS
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Content Security Policy
    csp_directives = [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://*.huggingface.co",
        "script-src-elem 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://*.huggingface.co",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com data:",
        "img-src 'self' data: blob: https:",
        "connect-src 'self' https://cdn.jsdelivr.net https://*.huggingface.co https://*.hf.space",
        "frame-ancestors 'self' https://*.huggingface.co https://huggingface.co https://*.hf.space",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
    response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

    # Other Security Headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=()"
    )
    if "/api/" in request.url.path:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"

    return response


# =============================================================================
# Rate Limiting Setup
# =============================================================================

import opensight.api.shared as _shared  # noqa: E402

if not SHOULD_ENABLE_RATE_LIMITING:
    _shared.RATE_LIMITING_ENABLED = False
    _shared.limiter = None
    logger.info("Rate limiting disabled (development mode)")
else:
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded

        _limiter = Limiter(key_func=get_real_client_ip)
        app.state.limiter = _limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        _shared.RATE_LIMITING_ENABLED = True
        _shared.limiter = _limiter
        logger.info(
            f"Rate limiting enabled with X-Forwarded-For support "
            f"(upload: {_shared.RATE_LIMIT_UPLOAD}, replay: {_shared.RATE_LIMIT_REPLAY}, api: {_shared.RATE_LIMIT_API})"
        )
    except ImportError as e:
        if IS_PRODUCTION:
            raise RuntimeError(
                "SECURITY ERROR: slowapi is required for rate limiting in production. "
                "Install with: pip install slowapi"
            ) from e
        else:
            _shared.RATE_LIMITING_ENABLED = False
            _shared.limiter = None
            logger.warning(
                "SECURITY WARNING: slowapi not installed - rate limiting DISABLED. "
                "This is acceptable for development but MUST be enabled in production."
            )

# =============================================================================
# Global Exception Handler
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler to prevent information disclosure."""
    logger.exception(f"Unhandled exception for {request.method} {request.url.path}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
        },
    )


# =============================================================================
# Static Files & Root
# =============================================================================

STATIC_DIR = Path(__file__).parent.parent / "static"

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


# =============================================================================
# Include Route Modules
# =============================================================================

from opensight.api.routes_analysis import router as analysis_router  # noqa: E402
from opensight.api.routes_auth import router as auth_router  # noqa: E402
from opensight.api.routes_export import router as export_router  # noqa: E402
from opensight.api.routes_heatmap import router as heatmap_router  # noqa: E402
from opensight.api.routes_maps import router as maps_router  # noqa: E402
from opensight.api.routes_match import router as match_router  # noqa: E402
from opensight.api.routes_misc import router as misc_router  # noqa: E402

app.include_router(analysis_router)
app.include_router(auth_router)
app.include_router(export_router)
app.include_router(heatmap_router)
app.include_router(match_router)
app.include_router(maps_router)
app.include_router(misc_router)
