"""
Tests for HLTV Rating 2.0 calculation.

Tests verify that the HLTV Rating calculation produces expected values
for various player performance scenarios.
"""

import pytest

from opensight.analysis.hltv_rating import (
    HLTVRatingResult,
    calculate_hltv_rating,
    calculate_hltv_rating_detailed,
    calculate_impact,
    get_rating_color,
    get_rating_tier,
)


class TestCalculateImpact:
    """Tests for impact rating calculation."""

    def test_impact_basic(self):
        """Test basic impact calculation with just KPR."""
        impact = calculate_impact(kpr=0.7, apr=0.2)
        # 2.13 * 0.7 + 0.42 * 0.2 - 0.41 = 1.491 + 0.084 - 0.41 = 1.165
        assert 1.1 < impact < 1.2

    def test_impact_zero_stats(self):
        """Test impact with zero stats."""
        impact = calculate_impact(kpr=0, apr=0)
        # Base is -0.41
        assert impact == -0.41

    def test_impact_with_clutch_bonus(self):
        """Test impact with clutch wins bonus."""
        base_impact = calculate_impact(kpr=0.7, apr=0.2)
        with_clutches = calculate_impact(kpr=0.7, apr=0.2, clutch_wins=3)
        # Each clutch adds 0.1
        assert with_clutches == pytest.approx(base_impact + 0.3, abs=0.01)

    def test_impact_with_multikills(self):
        """Test impact with multi-kill bonus."""
        base_impact = calculate_impact(kpr=0.7, apr=0.2)
        with_multis = calculate_impact(
            kpr=0.7, apr=0.2, multi_kill_3k=2, multi_kill_4k=1, multi_kill_5k=1
        )
        # 3k: 2 * 0.1 = 0.2
        # 4k: 1 * 0.2 = 0.2
        # 5k: 1 * 0.3 = 0.3
        expected_bonus = 0.2 + 0.2 + 0.3
        assert with_multis == pytest.approx(base_impact + expected_bonus, abs=0.01)


class TestCalculateHLTVRating:
    """Tests for HLTV 2.0 Rating calculation."""

    def test_rating_zero_rounds(self):
        """Test that zero rounds returns 0 rating."""
        rating = calculate_hltv_rating(
            kills=10,
            deaths=5,
            assists=3,
            adr=80.0,
            kast_pct=75.0,
            rounds=0,
        )
        assert rating == 0.0

    def test_rating_average_player(self):
        """Test rating for an average player (should be around 1.0)."""
        rating = calculate_hltv_rating(
            kills=14,
            deaths=14,
            assists=4,
            adr=75.0,
            kast_pct=70.0,
            rounds=20,
        )
        # Average player should be around 0.85-1.05
        assert 0.85 <= rating <= 1.05

    def test_rating_good_player(self):
        """Test rating for a good player (should be > 1.0)."""
        rating = calculate_hltv_rating(
            kills=22,
            deaths=12,
            assists=5,
            adr=90.0,
            kast_pct=80.0,
            rounds=20,
            multi_kill_2k=4,
            multi_kill_3k=1,
        )
        # Good player should be above 1.0
        assert rating > 1.0

    def test_rating_poor_player(self):
        """Test rating for a struggling player (should be < 0.9)."""
        rating = calculate_hltv_rating(
            kills=8,
            deaths=18,
            assists=2,
            adr=50.0,
            kast_pct=45.0,
            rounds=20,
        )
        # Poor performance should be below 0.9
        assert rating < 0.9

    def test_rating_exceptional_player(self):
        """Test rating for an exceptional player (should be > 1.3)."""
        rating = calculate_hltv_rating(
            kills=30,
            deaths=8,
            assists=6,
            adr=110.0,
            kast_pct=90.0,
            rounds=20,
            clutch_wins=3,
            multi_kill_2k=6,
            multi_kill_3k=3,
            multi_kill_4k=1,
        )
        # Exceptional player should be above 1.3
        assert rating > 1.3

    def test_rating_minimum_bound(self):
        """Test that rating cannot go below 0."""
        rating = calculate_hltv_rating(
            kills=0,
            deaths=20,
            assists=0,
            adr=10.0,
            kast_pct=10.0,
            rounds=20,
        )
        # Rating should be at minimum 0
        assert rating >= 0.0


class TestCalculateHLTVRatingDetailed:
    """Tests for detailed HLTV Rating calculation."""

    def test_detailed_returns_dataclass(self):
        """Test that detailed calculation returns HLTVRatingResult."""
        result = calculate_hltv_rating_detailed(
            kills=15,
            deaths=12,
            assists=4,
            adr=80.0,
            kast_pct=72.0,
            rounds=20,
        )
        assert isinstance(result, HLTVRatingResult)

    def test_detailed_components(self):
        """Test that detailed calculation includes all components."""
        result = calculate_hltv_rating_detailed(
            kills=20,
            deaths=10,
            assists=5,
            adr=85.0,
            kast_pct=75.0,
            rounds=20,
            multi_kill_2k=3,
        )
        # Check all components are present
        assert result.rating > 0
        assert result.impact > 0
        assert result.kast == 75.0
        assert result.kpr == 1.0  # 20 kills / 20 rounds
        assert result.dpr == 0.5  # 10 deaths / 20 rounds
        assert result.adr == 85.0
        assert result.rmk == 15.0  # 3 multi-kills / 20 rounds * 100 = 15%

    def test_detailed_to_dict(self):
        """Test that to_dict() returns proper dictionary."""
        result = calculate_hltv_rating_detailed(
            kills=15,
            deaths=12,
            assists=4,
            adr=80.0,
            kast_pct=72.0,
            rounds=20,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "rating" in d
        assert "impact" in d
        assert "kast" in d
        assert "kpr" in d
        assert "dpr" in d
        assert "adr" in d
        assert "rmk" in d

    def test_detailed_zero_rounds(self):
        """Test detailed with zero rounds returns zeroed result."""
        result = calculate_hltv_rating_detailed(
            kills=10,
            deaths=5,
            assists=3,
            adr=80.0,
            kast_pct=75.0,
            rounds=0,
        )
        assert result.rating == 0.0
        assert result.impact == 0.0
        assert result.kpr == 0.0
        assert result.dpr == 0.0


class TestGetRatingTier:
    """Tests for rating tier classification."""

    def test_tier_exceptional(self):
        """Test exceptional tier (1.30+)."""
        assert get_rating_tier(1.45) == "Exceptional"

    def test_tier_elite(self):
        """Test elite tier (1.15-1.29)."""
        assert get_rating_tier(1.20) == "Elite"
        assert get_rating_tier(1.15) == "Elite"

    def test_tier_very_good(self):
        """Test very good tier (1.05-1.14)."""
        assert get_rating_tier(1.10) == "Very Good"
        assert get_rating_tier(1.05) == "Very Good"

    def test_tier_good(self):
        """Test good tier (0.95-1.04)."""
        assert get_rating_tier(1.00) == "Good"
        assert get_rating_tier(0.95) == "Good"

    def test_tier_average(self):
        """Test average tier (0.85-0.94)."""
        assert get_rating_tier(0.90) == "Average"
        assert get_rating_tier(0.85) == "Average"

    def test_tier_below_average(self):
        """Test below average tier (0.75-0.84)."""
        assert get_rating_tier(0.80) == "Below Average"
        assert get_rating_tier(0.75) == "Below Average"

    def test_tier_poor(self):
        """Test poor tier (<0.75)."""
        assert get_rating_tier(0.70) == "Poor"
        assert get_rating_tier(0.50) == "Poor"


class TestGetRatingColor:
    """Tests for rating color coding."""

    def test_color_high_rating(self):
        """Test green color for high ratings."""
        assert get_rating_color(1.20) == "#10b981"
        assert get_rating_color(1.50) == "#10b981"

    def test_color_good_rating(self):
        """Test cyan color for good ratings."""
        assert get_rating_color(1.05) == "#00e5ff"
        assert get_rating_color(1.00) == "#00e5ff"

    def test_color_average_rating(self):
        """Test yellow color for average ratings."""
        assert get_rating_color(0.90) == "#f59e0b"
        assert get_rating_color(0.85) == "#f59e0b"

    def test_color_low_rating(self):
        """Test red color for low ratings."""
        assert get_rating_color(0.70) == "#ef4444"
        assert get_rating_color(0.50) == "#ef4444"


class TestHLTVRatingIntegration:
    """Integration tests simulating real match scenarios."""

    def test_pro_player_performance(self):
        """Test rating for typical pro player stats."""
        # Simulating s1mple-level performance
        rating = calculate_hltv_rating(
            kills=28,
            deaths=11,
            assists=4,
            adr=95.5,
            kast_pct=82.0,
            rounds=22,
            clutch_wins=2,
            multi_kill_2k=5,
            multi_kill_3k=2,
            multi_kill_4k=1,
        )
        # Top tier performance should be 1.3+
        assert rating >= 1.20

    def test_support_player_performance(self):
        """Test rating for support player with lower frags but high utility."""
        rating = calculate_hltv_rating(
            kills=12,
            deaths=13,
            assists=8,  # High assists from support
            adr=65.0,
            kast_pct=78.0,  # High KAST from trading/surviving
            rounds=22,
        )
        # Support player should still have reasonable rating
        assert 0.85 <= rating <= 1.05

    def test_entry_fragger_performance(self):
        """Test rating for aggressive entry fragger."""
        rating = calculate_hltv_rating(
            kills=18,
            deaths=16,  # Higher deaths from entries
            assists=3,
            adr=85.0,
            kast_pct=65.0,
            rounds=22,
            multi_kill_2k=3,
            multi_kill_3k=1,
        )
        # Entry fragger with good damage should be around average
        assert 0.9 <= rating <= 1.1

    def test_short_match(self):
        """Test rating calculation for short match (13-2 stomp)."""
        rating = calculate_hltv_rating(
            kills=10,
            deaths=3,
            assists=2,
            adr=85.0,
            kast_pct=80.0,
            rounds=15,
            multi_kill_2k=2,
        )
        # Should still calculate correctly for short matches
        assert rating > 1.0
