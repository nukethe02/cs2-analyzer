"""Tests for the FastAPI web API."""

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import io
import pytest

from fastapi.testclient import TestClient

from opensight.api import app


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
            mock_decode.return_value = MagicMock(
                match_id=12345,
                outcome_id=67890,
                token=11111
            )
            response = client.post(
                "/decode",
                json={"code": "CSGO-test-code-here-xxxx-xxxxx"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["match_id"] == 12345
            assert data["outcome_id"] == 67890
            assert data["token"] == 11111

    def test_decode_invalid_sharecode(self):
        """Decode endpoint returns 400 for invalid code."""
        with patch("opensight.sharecode.decode_sharecode") as mock_decode:
            mock_decode.side_effect = ValueError("Invalid sharecode")
            response = client.post(
                "/decode",
                json={"code": "invalid"}
            )
            assert response.status_code == 400
            assert "Invalid sharecode" in response.json()["detail"]

    def test_decode_missing_code(self):
        """Decode endpoint returns 422 for missing code."""
        response = client.post("/decode", json={})
        assert response.status_code == 422


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""

    def test_analyze_rejects_non_dem_file(self):
        """Analyze endpoint rejects non-.dem files."""
        files = {"file": ("test.txt", io.BytesIO(b"not a demo"), "text/plain")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 400
        assert "must be a .dem file" in response.json()["detail"]

    def test_analyze_with_valid_demo(self):
        """Analyze endpoint processes valid demo file."""
        # Create mock DemoData
        mock_demo_data = MagicMock()
        mock_demo_data.map_name = "de_dust2"
        mock_demo_data.duration_seconds = 1800.0
        mock_demo_data.tick_rate = 64
        mock_demo_data.num_rounds = 15
        mock_demo_data.player_stats = {
            12345: {
                "name": "Player1",
                "headshots": 5,
                "total_damage": 1200,
            }
        }
        mock_demo_data.kills = []
        mock_demo_data.damages = []

        # Create mock PlayerAnalytics
        mock_analytics = MagicMock()
        mock_analytics.name = "Player1"
        mock_analytics.team = "CT"
        mock_analytics.kills = 10
        mock_analytics.deaths = 5
        mock_analytics.assists = 3
        mock_analytics.adr = 80.0
        mock_analytics.hs_percent = 50.0
        mock_analytics.ttd_median_ms = 350.0
        mock_analytics.ttd_mean_ms = 380.0
        mock_analytics.ttd_min_ms = 200.0
        mock_analytics.ttd_max_ms = 800.0
        mock_analytics.ttd_std_ms = 120.0
        mock_analytics.ttd_count = 10
        mock_analytics.prefire_count = 2
        mock_analytics.cp_median_error_deg = 4.5
        mock_analytics.cp_mean_error_deg = 5.2
        mock_analytics.cp_pitch_bias_deg = -0.5
        mock_analytics.weapon_kills = {"ak47": 8, "awp": 2}

        with patch("opensight.parser.DemoParser") as mock_parser_cls, \
             patch("opensight.analytics.DemoAnalyzer") as mock_analyzer_cls:

            mock_parser = MagicMock()
            mock_parser.parse.return_value = mock_demo_data
            mock_parser_cls.return_value = mock_parser

            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = {12345: mock_analytics}
            mock_analyzer_cls.return_value = mock_analyzer

            files = {"file": ("test.dem", io.BytesIO(b"FAKE_DEMO"), "application/octet-stream")}
            response = client.post("/analyze", files=files)

            assert response.status_code == 200
            data = response.json()

            # Check demo info
            assert data["demo_info"]["map"] == "de_dust2"
            assert data["demo_info"]["rounds"] == 15

            # Check player data
            assert "12345" in data["players"]
            player = data["players"]["12345"]
            assert player["name"] == "Player1"
            assert player["stats"]["kills"] == 10
            assert player["advanced"]["ttd_median_ms"] == 350.0

    def test_analyze_handles_parser_error(self):
        """Analyze endpoint handles parser errors gracefully."""
        with patch("opensight.parser.DemoParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.parse.side_effect = Exception("Parse error")
            mock_parser_cls.return_value = mock_parser

            files = {"file": ("test.dem", io.BytesIO(b"FAKE_DEMO"), "application/octet-stream")}
            response = client.post("/analyze", files=files)

            assert response.status_code == 500
            assert "Analysis failed" in response.json()["detail"]


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
