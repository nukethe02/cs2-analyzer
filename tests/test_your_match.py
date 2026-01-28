"""
Tests for OpenSight Your Match feature.

Tests persona analysis, baseline calculation, match history storage,
and API endpoints for the Leetify-style personal performance dashboard.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Persona Analysis Tests
# =============================================================================


class TestPersonaAnalyzer:
    """Tests for persona determination logic."""

    def test_determine_persona_cleanup(self):
        """Test The Cleanup persona detection for trade kills."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {
            "trade_kill_opportunities": 10,
            "trade_kill_success": 6,  # 60% success rate
            "kills": 15,
            "deaths": 10,
            "adr": 70,
        }

        analyzer = PersonaAnalyzer()
        result = analyzer.determine_persona(stats)

        assert result.id == "the_cleanup"
        assert result.name == "The Cleanup"
        assert result.confidence > 0.5

    def test_determine_persona_opener(self):
        """Test The Opener persona detection for entry frags."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {
            "entry_attempts": 8,
            "entry_success": 5,  # 62.5% success rate
            "kills": 18,
            "deaths": 12,
            "adr": 75,
        }

        analyzer = PersonaAnalyzer()
        result = analyzer.determine_persona(stats)

        assert result.id == "the_opener"
        assert result.name == "The Opener"

    def test_determine_persona_anchor(self):
        """Test The Anchor persona detection for clutches."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {
            "clutch_situations": 5,
            "clutch_wins": 2,  # 40% clutch rate
            "kills": 12,
            "deaths": 8,
            "adr": 65,
        }

        analyzer = PersonaAnalyzer()
        result = analyzer.determine_persona(stats)

        assert result.id == "the_anchor"
        assert result.name == "The Anchor"

    def test_determine_persona_headhunter(self):
        """Test The Headhunter persona detection for high HS%."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {
            "hs_pct": 55,
            "kills": 20,
            "deaths": 10,
            "adr": 80,
        }

        analyzer = PersonaAnalyzer()
        result = analyzer.determine_persona(stats)

        assert result.id == "the_headhunter"
        assert result.name == "The Headhunter"

    def test_determine_persona_default(self):
        """Test default Competitor persona for average stats."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {
            "kills": 10,
            "deaths": 10,
            "adr": 60,
            "hs_pct": 30,
        }

        analyzer = PersonaAnalyzer()
        result = analyzer.determine_persona(stats)

        assert result.id == "the_competitor"
        assert result.name == "The Competitor"

    def test_determine_persona_empty_stats(self):
        """Test persona determination with empty stats."""
        from opensight.analysis.persona import PersonaAnalyzer

        analyzer = PersonaAnalyzer()
        result = analyzer.determine_persona({})

        assert result.id == "the_competitor"

    def test_persona_to_dict(self):
        """Test PersonaResult serialization."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {"hs_pct": 55, "kills": 10}
        analyzer = PersonaAnalyzer()
        result = analyzer.determine_persona(stats)
        data = result.to_dict()

        assert "id" in data
        assert "name" in data
        assert "description" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)


class TestTop5Stats:
    """Tests for top 5 stats calculation."""

    def test_calculate_top_5_basic(self):
        """Test basic top 5 stats calculation."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {
            "kills": 25,
            "adr": 90,
            "hs_pct": 50,
            "enemies_flashed": 10,
            "kast": 75,
            "hltv_rating": 1.3,
        }

        analyzer = PersonaAnalyzer()
        top_5 = analyzer.calculate_top_5_stats(stats)

        assert len(top_5) == 5
        # Should be sorted by percentile
        for i in range(len(top_5) - 1):
            assert top_5[i].percentile >= top_5[i + 1].percentile

    def test_calculate_top_5_with_baselines(self):
        """Test top 5 calculation with player baselines."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {"kills": 20, "adr": 85}

        baselines = {
            "kills": {"avg": 15, "std": 5, "sample_count": 10},
            "adr": {"avg": 70, "std": 10, "sample_count": 10},
        }

        analyzer = PersonaAnalyzer(baselines)
        top_5 = analyzer.calculate_top_5_stats(stats, baselines)

        assert len(top_5) <= 5
        # Kills should have high percentile (above avg)
        kills_stat = next((s for s in top_5 if s.stat == "kills"), None)
        if kills_stat:
            assert kills_stat.percentile > 50

    def test_top_stat_result_to_dict(self):
        """Test TopStatResult serialization."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {"kills": 20, "adr": 80}
        analyzer = PersonaAnalyzer()
        top_5 = analyzer.calculate_top_5_stats(stats)

        for stat in top_5:
            data = stat.to_dict()
            assert "stat" in data
            assert "name" in data
            assert "category" in data
            assert "value" in data
            assert "percentile" in data
            assert "rank" in data


class TestComparisonTable:
    """Tests for This Match vs Average comparison."""

    def test_build_comparison_basic(self):
        """Test basic comparison table building."""
        from opensight.analysis.persona import PersonaAnalyzer

        current = {
            "kills": 20,
            "adr": 85,
            "kast": 72,
            "hltv_rating": 1.2,
        }

        baselines = {
            "kills": {"avg": 15},
            "adr": {"avg": 70},
            "kast": {"avg": 68},
            "hltv_rating": {"avg": 1.0},
        }

        analyzer = PersonaAnalyzer(baselines)
        comparison = analyzer.build_comparison_table(current, baselines)

        assert len(comparison) > 0
        # Find kills comparison
        kills_row = next((r for r in comparison if r.metric == "kills"), None)
        if kills_row:
            assert kills_row.this_match == 20
            assert kills_row.average == 15
            assert kills_row.diff == 5
            assert kills_row.is_better is True

    def test_comparison_row_worse(self):
        """Test comparison row when performance is worse."""
        from opensight.analysis.persona import PersonaAnalyzer

        current = {"kills": 10, "adr": 60}
        baselines = {"kills": {"avg": 15}, "adr": {"avg": 75}}

        analyzer = PersonaAnalyzer(baselines)
        comparison = analyzer.build_comparison_table(current, baselines)

        kills_row = next((r for r in comparison if r.metric == "kills"), None)
        if kills_row:
            assert kills_row.is_better is False
            assert kills_row.diff < 0

    def test_comparison_row_to_dict(self):
        """Test ComparisonRow serialization."""
        from opensight.analysis.persona import PersonaAnalyzer

        current = {"kills": 20}
        baselines = {"kills": {"avg": 15}}

        analyzer = PersonaAnalyzer(baselines)
        comparison = analyzer.build_comparison_table(current, baselines)

        for row in comparison:
            data = row.to_dict()
            assert "metric" in data
            assert "label" in data
            assert "this_match" in data
            assert "average" in data
            assert "diff" in data
            assert "is_better" in data


# =============================================================================
# Database Tests
# =============================================================================


class TestMatchHistoryDatabase:
    """Tests for match history database operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        from opensight.infra.database import DatabaseManager

        # Use a temp file that we manage manually
        tmp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_file.close()
        db_path = Path(tmp_file.name)

        db = DatabaseManager(db_path)
        yield db

        # Cleanup: close all sessions and dispose engine
        try:
            db.engine.dispose()
        except Exception:
            pass

        # Try to remove the file
        try:
            db_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows may still have a lock

    def test_save_match_history_entry(self, temp_db):
        """Test saving a match history entry."""
        steam_id = "12345678901234567"
        demo_hash = "abc123def456"
        player_stats = {
            "kills": 20,
            "deaths": 10,
            "adr": 80,
            "hltv_rating": 1.2,
        }

        entry = temp_db.save_match_history_entry(
            steam_id=steam_id,
            demo_hash=demo_hash,
            player_stats=player_stats,
            map_name="de_dust2",
            result="win",
        )

        assert entry is not None

        # Verify by fetching from history
        history = temp_db.get_player_history(steam_id, limit=1)
        assert len(history) == 1
        assert history[0]["kills"] == 20
        assert history[0]["adr"] == 80

    def test_save_match_history_duplicate(self, temp_db):
        """Test that duplicate entries are rejected."""
        steam_id = "12345678901234567"
        demo_hash = "abc123def456"
        player_stats = {"kills": 20}

        # First save
        entry1 = temp_db.save_match_history_entry(
            steam_id=steam_id,
            demo_hash=demo_hash,
            player_stats=player_stats,
        )
        assert entry1 is not None

        # Duplicate save
        entry2 = temp_db.save_match_history_entry(
            steam_id=steam_id,
            demo_hash=demo_hash,
            player_stats=player_stats,
        )
        assert entry2 is None

    def test_get_player_history(self, temp_db):
        """Test retrieving player match history."""
        steam_id = "12345678901234567"

        # Save multiple matches
        for i in range(5):
            temp_db.save_match_history_entry(
                steam_id=steam_id,
                demo_hash=f"hash_{i}",
                player_stats={"kills": 10 + i},
            )

        history = temp_db.get_player_history(steam_id, limit=10)

        assert len(history) == 5

    def test_update_player_baselines(self, temp_db):
        """Test baseline calculation from match history."""
        steam_id = "12345678901234567"

        # Save matches with varying stats
        kills_values = [10, 15, 20, 25, 30]
        for i, kills in enumerate(kills_values):
            temp_db.save_match_history_entry(
                steam_id=steam_id,
                demo_hash=f"hash_{i}",
                player_stats={"kills": kills, "adr": 70.0},
            )

        baselines = temp_db.update_player_baselines(steam_id)

        assert "kills" in baselines
        assert baselines["kills"]["avg"] == 20  # Average of 10,15,20,25,30
        assert baselines["kills"]["sample_count"] == 5

    def test_get_player_baselines(self, temp_db):
        """Test retrieving player baselines."""
        steam_id = "12345678901234567"

        # Create some data first
        temp_db.save_match_history_entry(
            steam_id=steam_id,
            demo_hash="hash_1",
            player_stats={"kills": 20, "adr": 80.0},
        )
        temp_db.update_player_baselines(steam_id)

        baselines = temp_db.get_player_baselines(steam_id)

        assert isinstance(baselines, dict)
        assert "kills" in baselines

    def test_update_player_persona(self, temp_db):
        """Test updating player persona."""
        steam_id = "12345678901234567"

        persona = temp_db.update_player_persona(
            steam_id=steam_id,
            persona_id="the_headhunter",
            confidence=0.85,
            primary_trait="the_headhunter",
            secondary_trait="the_opener",
        )

        assert persona is not None

        # Verify by fetching persona
        result = temp_db.get_player_persona(steam_id)
        assert result is not None
        assert result["persona"] == "the_headhunter"
        assert result["confidence"] == 0.85

    def test_get_player_persona(self, temp_db):
        """Test retrieving player persona."""
        steam_id = "12345678901234567"

        # Create persona first
        temp_db.update_player_persona(
            steam_id=steam_id,
            persona_id="the_anchor",
            confidence=0.7,
        )

        persona = temp_db.get_player_persona(steam_id)

        assert persona is not None
        assert persona["persona"] == "the_anchor"


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestYourMatchAPI:
    """Tests for Your Match API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked database."""
        from fastapi.testclient import TestClient

        from opensight.api import app

        return TestClient(app)

    def test_get_your_match_invalid_steam_id(self, client):
        """Test validation of invalid Steam ID."""
        response = client.get("/api/your-match/abc123/invalid_steam_id")

        assert response.status_code == 400
        assert "steam_id" in response.json()["detail"].lower()

    def test_get_your_match_invalid_demo_id(self, client):
        """Test validation of invalid demo ID."""
        response = client.get(
            "/api/your-match/invalid!demo!id/12345678901234567"
        )

        assert response.status_code == 400
        assert "demo_id" in response.json()["detail"].lower()

    def test_get_baselines_endpoint(self, client):
        """Test baselines endpoint returns correct structure."""
        response = client.get("/api/your-match/baselines/12345678901234567")

        # Should return 200 (new player gets empty baselines)
        assert response.status_code == 200
        data = response.json()
        assert "steam_id" in data
        assert "baselines" in data
        assert data["steam_id"] == "12345678901234567"

    def test_get_history_endpoint(self, client):
        """Test history endpoint returns correct structure."""
        response = client.get("/api/your-match/history/12345678901234567")

        # Should return 200 (new player gets empty history)
        assert response.status_code == 200
        data = response.json()
        assert "steam_id" in data
        assert "matches" in data
        assert isinstance(data["matches"], list)

    def test_get_persona_endpoint_no_history(self, client):
        """Test persona endpoint returns default for new player."""
        response = client.get("/api/your-match/persona/12345678901234567")

        # Should return 200 with default persona for new players
        assert response.status_code == 200
        data = response.json()
        assert "persona" in data
        # New player should get default competitor persona
        assert data["persona"]["id"] == "the_competitor"
        assert data["match_count"] == 0

    def test_store_match_invalid_steam_id(self, client):
        """Test store match validation."""
        response = client.post(
            "/api/your-match/store",
            json={
                "steam_id": "invalid",
                "demo_hash": "abc123",
                "player_stats": {"kills": 20},
            },
        )

        assert response.status_code == 400


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation security."""

    def test_steam_id_validation_valid(self):
        """Test valid Steam ID passes validation."""
        from opensight.api import validate_steam_id

        # Should not raise
        result = validate_steam_id("12345678901234567")
        assert result == "12345678901234567"

    def test_steam_id_validation_invalid_length(self):
        """Test Steam ID with wrong length is rejected."""
        from fastapi import HTTPException

        from opensight.api import validate_steam_id

        with pytest.raises(HTTPException) as exc:
            validate_steam_id("123456789")  # Too short

        assert exc.value.status_code == 400

    def test_steam_id_validation_non_numeric(self):
        """Test Steam ID with non-numeric chars is rejected."""
        from fastapi import HTTPException

        from opensight.api import validate_steam_id

        with pytest.raises(HTTPException) as exc:
            validate_steam_id("1234567890123456a")  # Has letter

        assert exc.value.status_code == 400

    def test_demo_id_validation_valid(self):
        """Test valid demo ID passes validation."""
        from opensight.api import validate_demo_id

        result = validate_demo_id("abc123-def456_xyz")
        assert result == "abc123-def456_xyz"

    def test_demo_id_validation_invalid_chars(self):
        """Test demo ID with invalid characters is rejected."""
        from fastapi import HTTPException

        from opensight.api import validate_demo_id

        with pytest.raises(HTTPException) as exc:
            validate_demo_id("abc!@#$%")  # Has special chars

        assert exc.value.status_code == 400

    def test_demo_id_validation_too_long(self):
        """Test demo ID that's too long is rejected."""
        from fastapi import HTTPException

        from opensight.api import validate_demo_id

        with pytest.raises(HTTPException) as exc:
            validate_demo_id("a" * 100)  # Too long

        assert exc.value.status_code == 400


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_determine_persona_function(self):
        """Test module-level determine_persona function."""
        from opensight.analysis.persona import determine_persona

        result = determine_persona({"hs_pct": 55, "kills": 15})

        assert result.id is not None
        assert result.name is not None

    def test_calculate_top_5_function(self):
        """Test module-level calculate_top_5 function."""
        from opensight.analysis.persona import calculate_top_5

        result = calculate_top_5({"kills": 20, "adr": 80})

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)
            assert "stat" in item

    def test_build_comparison_function(self):
        """Test module-level build_comparison function."""
        from opensight.analysis.persona import build_comparison

        result = build_comparison(
            {"kills": 20},
            {"kills": {"avg": 15}},
        )

        assert isinstance(result, list)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_persona_with_none_values(self):
        """Test persona determination handles None values."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {
            "kills": None,
            "deaths": 10,
            "adr": None,
        }

        analyzer = PersonaAnalyzer()
        result = analyzer.determine_persona(stats)

        # Should not raise and return default
        assert result.id == "the_competitor"

    def test_top_5_with_missing_stats(self):
        """Test top 5 calculation with minimal stats."""
        from opensight.analysis.persona import PersonaAnalyzer

        stats = {"kills": 10}  # Only one stat

        analyzer = PersonaAnalyzer()
        top_5 = analyzer.calculate_top_5_stats(stats)

        assert len(top_5) >= 1

    def test_comparison_with_zero_baseline(self):
        """Test comparison handles zero baseline gracefully."""
        from opensight.analysis.persona import PersonaAnalyzer

        current = {"kills": 10}
        baselines = {"kills": {"avg": 0}}

        analyzer = PersonaAnalyzer(baselines)
        comparison = analyzer.build_comparison_table(current, baselines)

        # Should not raise division by zero
        kills_row = next((r for r in comparison if r.metric == "kills"), None)
        if kills_row:
            assert kills_row.diff == 10

    def test_percentile_calculation_inverted_metric(self):
        """Test percentile calculation for lower-is-better metrics."""
        from opensight.analysis.persona import _calculate_percentile

        benchmark = {"min": 500, "avg": 250, "max": 100, "higher_is_better": False}

        # Lower TTD should have higher percentile
        good_ttd = _calculate_percentile(150, benchmark)
        bad_ttd = _calculate_percentile(400, benchmark)

        assert good_ttd > bad_ttd
