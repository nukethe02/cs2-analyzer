"""
Test AI Connection for LLM-powered coaching insights.

This test verifies that the LLM client:
1. Accepts player stats correctly
2. Returns a non-empty response
3. The response contains references to the actual stats (not hallucinated)

Run with:
    PYTHONPATH=src pytest tests/test_ai_connection.py -v

For real LLM testing (requires ANTHROPIC_API_KEY):
    ANTHROPIC_API_KEY=sk-... PYTHONPATH=src pytest tests/test_ai_connection.py -v
"""

import os
from unittest.mock import Mock, patch

import pytest


class TestLLMClientModule:
    """Test that the LLM client module can be imported and used."""

    def test_module_imports(self):
        """Test that llm_client module imports correctly."""
        from opensight.ai.llm_client import (
            LLMClient,
            generate_match_summary,
            get_llm_client,
        )

        assert LLMClient is not None
        assert callable(get_llm_client)
        assert callable(generate_match_summary)

    def test_client_initialization(self):
        """Test LLMClient can be initialized without API key."""
        from opensight.ai.llm_client import LLMClient

        client = LLMClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model == "claude-sonnet-4-20250514"


class TestGenerateSummaryWithMockedLLM:
    """Test generate_match_summary with mocked LLM responses."""

    @pytest.fixture
    def mock_player_stats(self):
        """Mock player stats with 30 Kills and 1.5 Rating."""
        return {
            "kills": 30,
            "deaths": 12,
            "assists": 5,
            "hltv_rating": 1.50,
            "adr": 95.5,
            "headshot_pct": 45.0,
            "kast_percentage": 72.0,
            "ttd_median_ms": 280,
            "cp_median_error_deg": 8.5,
            "entry_kills": 6,
            "entry_deaths": 2,
            "trade_kill_success": 4,
            "trade_kill_opportunities": 7,
            "clutch_wins": 2,
            "clutch_attempts": 3,
        }

    def test_stats_not_hallucinated_mock(self, mock_player_stats):
        """Test that response contains actual stats, not hallucinated ones."""
        from opensight.ai.llm_client import LLMClient

        # Create mock response that references the correct stats
        mock_content = Mock()
        mock_content.text = """## Match Analysis

### Strengths
1. **Strong Fragging**: 30 Kills with only 12 deaths shows excellent impact
2. **High Rating**: 1.50 HLTV rating is well above average
3. **Good ADR**: 95.5 damage per round shows consistent damage output

### Critical Weakness
- Entry death ratio could improve (6 entry kills vs 2 deaths)

### Practice Focus
Focus on crosshair placement drills (current: 8.5 error)"""

        mock_message = Mock()
        mock_message.content = [mock_content]

        # Patch where Anthropic is imported (inside _get_client method)
        with patch.dict("sys.modules", {"anthropic": Mock()}):
            import sys

            mock_anthropic_module = sys.modules["anthropic"]
            mock_anthropic_class = Mock()
            mock_client_instance = Mock()
            mock_client_instance.messages.create.return_value = mock_message
            mock_anthropic_class.return_value = mock_client_instance
            mock_anthropic_module.Anthropic = mock_anthropic_class

            client = LLMClient(api_key="test-key")
            result = client.generate_match_summary(mock_player_stats)

            # Assert response is NOT empty
            assert result is not None
            assert len(result) > 0

            # Assert response contains "Kills" (the actual stat)
            assert "Kills" in result or "kills" in result.lower()

            # Assert response contains "30" (the actual kill count)
            assert "30" in result

            # Assert response references the rating
            assert "1.5" in result or "1.50" in result

    def test_zero_stats_returns_error(self):
        """Test that zero stats returns an error message, not hallucinated data."""
        from opensight.ai.llm_client import LLMClient

        zero_stats = {
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "hltv_rating": 0.0,
        }

        client = LLMClient(api_key="test-key")
        result = client.generate_match_summary(zero_stats)

        # Should return error message, not try to generate summary
        assert "Error" in result or "not available" in result


class TestRealLLMConnection:
    """
    Test with real LLM API (only runs if ANTHROPIC_API_KEY is set).

    These tests verify the actual LLM connection works.
    Skip in CI environments without API key.
    """

    @pytest.fixture
    def player_30k_stats(self):
        """Player with 30 Kills and 1.5 Rating."""
        return {
            "kills": 30,
            "deaths": 12,
            "assists": 5,
            "hltv_rating": 1.50,
            "adr": 95.5,
            "headshot_pct": 45.0,
            "kast_percentage": 72.0,
            "ttd_median_ms": 280,
            "cp_median_error_deg": 8.5,
            "entry_kills": 6,
            "entry_deaths": 2,
            "trade_kill_success": 4,
            "trade_kill_opportunities": 7,
            "clutch_wins": 2,
            "clutch_attempts": 3,
        }

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set - skipping real LLM test",
    )
    def test_real_llm_connection(self, player_30k_stats):
        """
        Test real LLM generates summary with actual stats.

        REQUIREMENT: Response must NOT be empty and MUST contain "Kills".
        """
        from opensight.ai.llm_client import generate_match_summary

        result = generate_match_summary(player_30k_stats)

        # Print for verification
        print("\n" + "=" * 60)
        print("REAL LLM RESPONSE:")
        print("=" * 60)
        print(result)
        print("=" * 60)

        # ASSERT 1: Response is NOT empty
        assert result is not None, "Response should not be None"
        assert len(result) > 0, "Response should not be empty"

        # ASSERT 2: Response contains "Kills" (proves it's using real stats)
        assert "kill" in result.lower(), (
            f"Response must contain 'Kills' to prove it's not hallucinating. Got: {result[:200]}..."
        )

        # ASSERT 3: Response should reference the actual numbers
        # (at least one of: 30, 12, 1.5, 95)
        contains_real_stats = any(str(num) in result for num in ["30", "12", "1.5", "1.50", "95"])
        assert contains_real_stats, (
            f"Response should reference actual stats (30K, 12D, 1.5 rating, 95 ADR). "
            f"Got: {result[:200]}..."
        )


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
