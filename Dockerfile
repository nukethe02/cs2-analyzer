# =============================================================================
# OpenSight CS2 Analyzer - Multi-Stage Docker Build
# Optimized for Hugging Face Spaces deployment
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build Stage
# Install dev tools, compile Rust extensions (demoparser2), build dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for compiling Rust extensions and native modules
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cargo \
    rustc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements and install dependencies
# This layer is cached unless requirements.txt changes
COPY requirements.txt .

# Install core dependencies first (these have Rust/native extensions)
# demoparser2 has Rust bindings that need compilation
RUN pip install --no-cache-dir demoparser2>=0.9.0 || echo "demoparser2 compilation failed, will retry"

# Install awpy (includes demoparser2 as dependency, may provide pre-built wheels)
RUN pip install --no-cache-dir awpy>=2.0.0 || echo "awpy installation failed, continuing"

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    "uvicorn[standard]>=0.23.0" \
    python-multipart>=0.0.6 \
    websockets>=11.0.0 \
    slowapi>=0.1.9 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    polars>=0.20.0 \
    rich>=13.0.0 \
    typer>=0.9.0 \
    watchdog>=3.0.0 \
    pyyaml>=6.0 \
    openpyxl>=3.1.0 \
    || true

# Retry demoparser2 if first attempt failed
RUN pip show demoparser2 > /dev/null 2>&1 || pip install --no-cache-dir demoparser2>=0.9.0 || echo "demoparser2 unavailable"

# =============================================================================
# CRITICAL: Verify essential packages are installed (fail build if missing)
# =============================================================================
RUN echo "Verifying critical packages..." && \
    pip show awpy || (echo "CRITICAL: awpy not installed!" && exit 1) && \
    pip show fastapi || (echo "CRITICAL: fastapi not installed!" && exit 1) && \
    pip show uvicorn || (echo "CRITICAL: uvicorn not installed!" && exit 1) && \
    pip show pandas || (echo "CRITICAL: pandas not installed!" && exit 1) && \
    echo "✅ All critical packages verified"

# -----------------------------------------------------------------------------
# Stage 2: Runtime Stage
# Minimal image with only runtime dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only runtime system dependencies (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy only necessary source files (not tests, docs, etc.)
COPY src/ ./src/

# Create temp directory with proper permissions
RUN mkdir -p /tmp/opensight && chmod 777 /tmp/opensight

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app /tmp/opensight
USER appuser

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run uvicorn with optimized settings:
# - 1 worker: Demo parsing is CPU-heavy, multiple workers would compete for CPU
# - log-level warning: Reduce verbose access logs in production
# - timeout-keep-alive 65: Slightly above typical load balancer timeout (60s)
CMD ["uvicorn", "opensight.api:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "warning", \
     "--timeout-keep-alive", "65"]
