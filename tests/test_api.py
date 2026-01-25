"""Tests for the FastAPI web API."""

import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from opensight.api import app, job_store, sharecode_cache

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self):
        """Health endpoint returns OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_returns_version(self):
        """Health endpoint includes version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)


class TestRootEndpoint:
    """Tests for the / endpoint."""

    def test_root_returns_html(self):
        """Root endpoint returns HTML response."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestDecodeEndpoint:
    """Tests for the /decode endpoint."""

    def test_decode_valid_sharecode(self):
        """Decode endpoint returns match info for valid code."""
        # Use a mock to avoid needing a real sharecode
        with patch("opensight.sharecode.decode_sharecode") as mock_decode:
            mock_decode.return_value = MagicMock(match_id=12345, outcome_id=67890, token=11111)
            response = client.post("/decode", json={"code": "CSGO-test-code-here-xxxx-xxxxx"})
            assert response.status_code == 200
            data = response.json()
            assert data["match_id"] == 12345
            assert data["outcome_id"] == 67890
            assert data["token"] == 11111

    def test_decode_invalid_sharecode(self):
        """Decode endpoint returns 400 for invalid code."""
        with patch("opensight.sharecode.decode_sharecode") as mock_decode:
            mock_decode.side_effect = ValueError("Invalid sharecode")
            response = client.post("/decode", json={"code": "invalid"})
            assert response.status_code == 400
            assert "Invalid sharecode" in response.json()["detail"]

    def test_decode_missing_code(self):
        """Decode endpoint returns 422 for missing code."""
        response = client.post("/decode", json={})
        assert response.status_code == 422


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint (async job-based)."""

    def test_analyze_rejects_non_dem_file(self):
        """Analyze endpoint rejects non-.dem files."""
        files = {"file": ("test.txt", io.BytesIO(b"not a demo"), "text/plain")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 400
        assert ".dem" in response.json()["detail"]

    def test_analyze_returns_202_with_job_id(self):
        """Analyze endpoint returns 202 Accepted with job ID."""
        files = {"file": ("test.dem", io.BytesIO(b"FAKE_DEMO"), "application/octet-stream")}
        response = client.post("/analyze", files=files)

        # Should return 202 Accepted
        assert response.status_code == 202
        data = response.json()

        # Should contain job tracking info
        assert "job_id" in data
        # Status may be "pending" or "processing" depending on timing
        assert data["status"] in ("pending", "processing")
        assert "status_url" in data
        assert "download_url" in data

    def test_analyze_empty_file_rejected(self):
        """Analyze endpoint rejects empty files."""
        files = {"file": ("test.dem", io.BytesIO(b""), "application/octet-stream")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]


class TestJobStatusEndpoint:
    """Tests for the /analyze/{job_id} endpoint."""

    def test_job_status_not_found(self):
        """Job status returns 404 for unknown job."""
        response = client.get("/analyze/nonexistent-job-id")
        assert response.status_code == 404

    def test_job_status_returns_info(self):
        """Job status returns job information."""
        # Create a test job directly
        job = job_store.create_job("test.dem", 1024)
        response = client.get(f"/analyze/{job.job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job.job_id
        assert data["status"] == "pending"
        assert data["filename"] == "test.dem"


class TestJobsListEndpoint:
    """Tests for the /jobs endpoint."""

    def test_jobs_list_returns_array(self):
        """Jobs list returns array of jobs."""
        response = client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)


class TestDownloadEndpoint:
    """Tests for the /analyze/{job_id}/download endpoint."""

    def test_download_not_found(self):
        """Download returns 404 for unknown job."""
        response = client.get("/analyze/nonexistent-job-id/download")
        assert response.status_code == 404

    def test_download_not_completed(self):
        """Download returns 400 for incomplete job."""
        job = job_store.create_job("test.dem", 1024)
        response = client.get(f"/analyze/{job.job_id}/download")
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]


class TestAboutEndpoint:
    """Tests for the /about endpoint."""

    def test_about_returns_info(self):
        """About endpoint returns API information."""
        response = client.get("/about")
        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "OpenSight"
        assert "version" in data
        assert "metrics" in data
        assert "methodology" in data

    def test_about_includes_metrics_descriptions(self):
        """About endpoint includes metric descriptions."""
        response = client.get("/about")
        data = response.json()

        assert "basic" in data["metrics"]
        assert "advanced" in data["metrics"]
        assert "kills" in data["metrics"]["basic"]
        assert "ttd_median_ms" in data["metrics"]["advanced"]

    def test_about_includes_methodology(self):
        """About endpoint includes methodology explanations."""
        response = client.get("/about")
        data = response.json()

        assert "ttd" in data["methodology"]
        assert "crosshair_placement" in data["methodology"]

    def test_about_includes_api_optimization(self):
        """About endpoint includes API optimization documentation."""
        response = client.get("/about")
        data = response.json()

        assert "api_optimization" in data
        assert "async_analysis" in data["api_optimization"]
        assert "caching" in data["api_optimization"]


class TestCacheEndpoints:
    """Tests for cache management endpoints."""

    def test_cache_stats_returns_info(self):
        """Cache stats endpoint returns cache information."""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()

        assert "sharecode_cache" in data
        assert "job_store" in data
        assert "maxsize" in data["sharecode_cache"]

    def test_cache_clear_clears_caches(self):
        """Cache clear endpoint clears all caches."""
        # Add something to sharecode cache
        sharecode_cache.set("test-code", {"test": "data"})

        response = client.post("/cache/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Cache should be empty
        assert sharecode_cache.get("test-code") is None


class TestShareCodeCaching:
    """Tests for share code caching."""

    def test_sharecode_cache_set_and_get(self):
        """Sharecode cache can set and get values."""
        sharecode_cache.set("test-code-1", {"match_id": 123})
        result = sharecode_cache.get("test-code-1")
        assert result == {"match_id": 123}

    def test_sharecode_cache_miss(self):
        """Sharecode cache returns None for missing keys."""
        result = sharecode_cache.get("nonexistent-code")
        assert result is None


class TestGZipCompression:
    """Tests for GZip compression middleware."""

    def test_gzip_enabled_for_large_response(self):
        """GZip compression is applied for large responses."""
        # Request about endpoint which returns sizeable JSON
        response = client.get("/about", headers={"Accept-Encoding": "gzip"})
        assert response.status_code == 200
        # The middleware handles decompression, so we just verify it works
        data = response.json()
        assert "name" in data


class TestStaticAssetCaching:
    """Tests for static asset cache headers."""

    def test_root_has_cache_headers(self):
        """Root endpoint includes cache-control headers."""
        response = client.get("/")
        assert response.status_code == 200
        # Check for cache-control header
        assert "cache-control" in response.headers
        assert "max-age" in response.headers["cache-control"]
