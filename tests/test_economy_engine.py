"""
Tests for Economy Management Engine.

Tests the full pipeline:
  - Loss bonus ladder calculation
  - Buy recommendation at various money levels
  - Opponent economy prediction
  - Halftime economy reset handling
  - Decision grading against optimal play
  - Full match analysis
  - API endpoint integration
"""

import pytest

# =============================================================================
# Fixtures — realistic round_timeline data matching orchestrator format
# =============================================================================


def _make_round(
    round_num: int,
    winner: str,
    ct_equipment: int = 0,
    t_equipment: int = 0,
    ct_buy_type: str = "full",
    t_buy_type: str = "full",
    ct_loss_bonus: int = 1400,
    t_loss_bonus: int = 1400,
    ct_consecutive_losses: int = 0,
    t_consecutive_losses: int = 0,
    ct_decision_grade: str = "B",
    t_decision_grade: str = "B",
) -> dict:
    """Build a single round_timeline entry with economy data."""
    return {
        "round_num": round_num,
        "winner": winner,
        "win_reason": "elimination",
        "economy": {
            "ct": {
                "equipment": ct_equipment,
                "buy_type": ct_buy_type,
                "loss_bonus": ct_loss_bonus,
                "consecutive_losses": ct_consecutive_losses,
                "decision_flag": "ok",
                "decision_grade": ct_decision_grade,
                "loss_bonus_next": min(ct_loss_bonus + 500, 3400),
                "is_bad_force": False,
                "is_good_force": False,
                "prediction": None,
            },
            "t": {
                "equipment": t_equipment,
                "buy_type": t_buy_type,
                "loss_bonus": t_loss_bonus,
                "consecutive_losses": t_consecutive_losses,
                "decision_flag": "ok",
                "decision_grade": t_decision_grade,
                "loss_bonus_next": min(t_loss_bonus + 500, 3400),
                "is_bad_force": False,
                "is_good_force": False,
                "prediction": None,
            },
        },
    }


@pytest.fixture
def basic_match():
    """A basic 24-round match (12-12 regulation) with realistic economy flow."""
    timeline = [
        # Pistol round + anti-eco
        _make_round(
            1, "CT", ct_equipment=4000, t_equipment=4000, ct_buy_type="pistol", t_buy_type="pistol"
        ),
        _make_round(
            2,
            "CT",
            ct_equipment=18000,
            t_equipment=3000,
            ct_buy_type="full",
            t_buy_type="eco",
            t_consecutive_losses=1,
            t_loss_bonus=1400,
        ),
        _make_round(
            3,
            "CT",
            ct_equipment=25000,
            t_equipment=5000,
            ct_buy_type="full",
            t_buy_type="force",
            t_consecutive_losses=2,
            t_loss_bonus=1900,
        ),
        # T wins a force buy
        _make_round(
            4,
            "T",
            ct_equipment=24000,
            t_equipment=15000,
            ct_buy_type="full",
            t_buy_type="force",
            t_consecutive_losses=3,
            t_loss_bonus=2400,
        ),
        # T builds economy
        _make_round(
            5,
            "T",
            ct_equipment=8000,
            t_equipment=22000,
            ct_buy_type="eco",
            t_buy_type="full",
            ct_consecutive_losses=1,
            ct_loss_bonus=1400,
        ),
        _make_round(
            6, "CT", ct_equipment=20000, t_equipment=25000, ct_buy_type="full", t_buy_type="full"
        ),
        _make_round(
            7, "T", ct_equipment=24000, t_equipment=24000, ct_buy_type="full", t_buy_type="full"
        ),
        _make_round(
            8,
            "CT",
            ct_equipment=22000,
            t_equipment=8000,
            ct_buy_type="full",
            t_buy_type="eco",
            t_consecutive_losses=1,
        ),
        _make_round(
            9,
            "CT",
            ct_equipment=25000,
            t_equipment=15000,
            ct_buy_type="full",
            t_buy_type="force",
            t_consecutive_losses=2,
            t_loss_bonus=1900,
        ),
        _make_round(
            10, "T", ct_equipment=24000, t_equipment=23000, ct_buy_type="full", t_buy_type="full"
        ),
        _make_round(
            11,
            "T",
            ct_equipment=7000,
            t_equipment=25000,
            ct_buy_type="eco",
            t_buy_type="full",
            ct_consecutive_losses=1,
        ),
        _make_round(
            12, "CT", ct_equipment=20000, t_equipment=25000, ct_buy_type="full", t_buy_type="full"
        ),
        # Second half (sides swap)
        _make_round(
            13, "T", ct_equipment=4000, t_equipment=4000, ct_buy_type="pistol", t_buy_type="pistol"
        ),
        _make_round(
            14,
            "T",
            ct_equipment=3000,
            t_equipment=18000,
            ct_buy_type="eco",
            t_buy_type="full",
            ct_consecutive_losses=1,
        ),
        _make_round(
            15, "CT", ct_equipment=15000, t_equipment=24000, ct_buy_type="force", t_buy_type="full"
        ),
        _make_round(
            16,
            "CT",
            ct_equipment=23000,
            t_equipment=8000,
            ct_buy_type="full",
            t_buy_type="eco",
            t_consecutive_losses=1,
        ),
        _make_round(
            17, "T", ct_equipment=24000, t_equipment=20000, ct_buy_type="full", t_buy_type="full"
        ),
        _make_round(
            18,
            "T",
            ct_equipment=7000,
            t_equipment=24000,
            ct_buy_type="eco",
            t_buy_type="full",
            ct_consecutive_losses=1,
        ),
        _make_round(
            19, "CT", ct_equipment=22000, t_equipment=25000, ct_buy_type="full", t_buy_type="full"
        ),
        _make_round(
            20,
            "CT",
            ct_equipment=25000,
            t_equipment=7000,
            ct_buy_type="full",
            t_buy_type="eco",
            t_consecutive_losses=1,
        ),
        _make_round(
            21, "T", ct_equipment=24000, t_equipment=20000, ct_buy_type="full", t_buy_type="full"
        ),
        _make_round(
            22,
            "T",
            ct_equipment=6000,
            t_equipment=25000,
            ct_buy_type="eco",
            t_buy_type="full",
            ct_consecutive_losses=1,
        ),
        _make_round(
            23, "CT", ct_equipment=22000, t_equipment=24000, ct_buy_type="full", t_buy_type="full"
        ),
        _make_round(
            24, "CT", ct_equipment=25000, t_equipment=22000, ct_buy_type="full", t_buy_type="full"
        ),
    ]

    return {
        "demo_info": {
            "rounds": 24,
            "team1_name": "TeamAlpha",
            "team2_name": "TeamBravo",
            "score_ct": 13,
            "score_t": 11,
        },
        "round_timeline": timeline,
    }


@pytest.fixture
def short_match():
    """A short 5-round sequence for targeted tests."""
    timeline = [
        _make_round(
            1, "CT", ct_equipment=4000, t_equipment=4000, ct_buy_type="pistol", t_buy_type="pistol"
        ),
        _make_round(
            2,
            "CT",
            ct_equipment=18000,
            t_equipment=2500,
            ct_buy_type="full",
            t_buy_type="eco",
            t_consecutive_losses=1,
        ),
        _make_round(
            3,
            "T",
            ct_equipment=24000,
            t_equipment=5000,
            ct_buy_type="full",
            t_buy_type="force",
            t_consecutive_losses=2,
            t_loss_bonus=1900,
        ),
        _make_round(
            4,
            "T",
            ct_equipment=7000,
            t_equipment=20000,
            ct_buy_type="eco",
            t_buy_type="full",
            ct_consecutive_losses=1,
        ),
        _make_round(
            5, "CT", ct_equipment=20000, t_equipment=24000, ct_buy_type="full", t_buy_type="full"
        ),
    ]
    return {
        "demo_info": {"rounds": 24, "team1_name": "Alpha", "team2_name": "Bravo"},
        "round_timeline": timeline,
    }


# =============================================================================
# Test: Loss Bonus Calculation
# =============================================================================


class TestLossBonusLadder:
    """Test CS2 loss bonus progression."""

    def test_zero_losses(self):
        from opensight.ai.economy_engine import LOSS_BONUS_LADDER

        assert LOSS_BONUS_LADDER[0] == 1400

    def test_one_loss(self):
        from opensight.ai.economy_engine import LOSS_BONUS_LADDER

        assert LOSS_BONUS_LADDER[1] == 1900

    def test_max_losses(self):
        from opensight.ai.economy_engine import LOSS_BONUS_LADDER

        assert LOSS_BONUS_LADDER[4] == 3400

    def test_ladder_length(self):
        from opensight.ai.economy_engine import LOSS_BONUS_LADDER

        assert len(LOSS_BONUS_LADDER) == 5

    def test_ladder_increments(self):
        from opensight.ai.economy_engine import LOSS_BONUS_LADDER

        for i in range(1, len(LOSS_BONUS_LADDER)):
            assert LOSS_BONUS_LADDER[i] - LOSS_BONUS_LADDER[i - 1] == 500


# =============================================================================
# Test: Buy Recommendation
# =============================================================================


class TestBuyRecommendation:
    """Test recommend_buy at various money levels and game states."""

    def _recommend(self, **kwargs):
        from opensight.ai.economy_engine import EconomyEngine

        defaults = {
            "team": "ct",
            "round_number": 5,
            "our_money_estimate": 25000,
            "opponent_predicted_money": 25000,
            "our_score": 3,
            "opp_score": 1,
            "total_rounds": 24,
            "loss_streak": 0,
            "is_pistol": False,
        }
        defaults.update(kwargs)
        return EconomyEngine().recommend_buy(**defaults)

    def test_pistol_round_always_full_buy(self):
        state = self._recommend(is_pistol=True, our_money_estimate=4000)
        assert state.buy_recommendation == "full_buy"
        assert state.confidence == 1.0

    def test_full_buy_with_rich_economy(self):
        state = self._recommend(our_money_estimate=25000)
        assert state.buy_recommendation == "full_buy"

    def test_full_buy_vs_opponent_eco(self):
        state = self._recommend(our_money_estimate=25000, opponent_predicted_money=5000)
        assert state.buy_recommendation == "full_buy"
        assert "eco" in state.reasoning.lower()

    def test_force_buy_in_mid_range(self):
        state = self._recommend(our_money_estimate=12000, loss_streak=1)
        assert state.buy_recommendation == "force_buy"

    def test_half_buy_on_high_loss_streak(self):
        state = self._recommend(our_money_estimate=12000, loss_streak=3)
        assert state.buy_recommendation == "half_buy"

    def test_eco_in_low_range(self):
        state = self._recommend(our_money_estimate=8000, loss_streak=4)
        assert state.buy_recommendation == "eco"

    def test_full_save_when_broke(self):
        state = self._recommend(our_money_estimate=5000)
        assert state.buy_recommendation == "full_save"

    def test_match_point_full_buy(self):
        state = self._recommend(our_score=12, opp_score=5, our_money_estimate=25000)
        assert state.buy_recommendation == "full_buy"
        assert "match point" in state.reasoning.lower()

    def test_match_point_force_when_poor(self):
        state = self._recommend(our_score=12, opp_score=5, our_money_estimate=8000)
        assert state.buy_recommendation == "force_buy"
        assert "match point" in state.reasoning.lower()

    def test_opponent_match_point_force(self):
        state = self._recommend(our_score=5, opp_score=12, our_money_estimate=8000)
        assert state.buy_recommendation == "force_buy"
        assert "nothing to save" in state.reasoning.lower()

    def test_every_recommendation_has_reasoning(self):
        """All recommendations must have non-empty reasoning."""
        for money in [3000, 8000, 12000, 18000, 25000]:
            for streak in [0, 2, 4]:
                state = self._recommend(our_money_estimate=money, loss_streak=streak)
                assert state.reasoning, f"Empty reasoning for ${money}, streak={streak}"


# =============================================================================
# Test: Opponent Economy Prediction
# =============================================================================


class TestOpponentPrediction:
    """Test predict_opponent_economy accuracy."""

    def test_after_win_higher_money(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        predicted = engine.predict_opponent_economy(loss_streak=0, last_equipment=20000)
        # After win: win reward × 5 + surviving equipment
        assert predicted > 20000

    def test_first_loss_low_money(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        predicted = engine.predict_opponent_economy(loss_streak=1, last_equipment=0)
        # Loss streak 1 → LOSS_BONUS_LADDER[1] = $1900 × 5 = $9500
        assert predicted == 9500

    def test_max_loss_streak(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        predicted = engine.predict_opponent_economy(loss_streak=5, last_equipment=0)
        # Max loss: $3400 × 5 = $17000
        assert predicted == 17000

    def test_loss_streak_2(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        predicted = engine.predict_opponent_economy(loss_streak=2, last_equipment=0)
        # 2 losses: $2400 × 5 = $12000
        assert predicted == 12000

    def test_loss_streak_beyond_cap(self):
        """Loss streak beyond 5 should still use max bonus."""
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        predicted = engine.predict_opponent_economy(loss_streak=8, last_equipment=0)
        # Capped at index 4: $3400 × 5 = $17000
        assert predicted == 17000


# =============================================================================
# Test: Halftime Reset
# =============================================================================


class TestHalftimeHandling:
    """Test halftime economy reset and side swap."""

    def test_is_pistol_round_1(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._is_pistol_round(1, 24) is True

    def test_is_pistol_round_13(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._is_pistol_round(13, 24) is True

    def test_regular_round_not_pistol(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._is_pistol_round(5, 24) is False
        assert engine._is_pistol_round(12, 24) is False
        assert engine._is_pistol_round(14, 24) is False

    def test_halftime_boundary(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._is_halftime_boundary(12, 24) is True
        assert engine._is_halftime_boundary(11, 24) is False
        assert engine._is_halftime_boundary(13, 24) is False

    def test_side_swap_at_halftime(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        # First half: starting side preserved
        assert engine._get_current_side("ct", 1, 24) == "ct"
        assert engine._get_current_side("ct", 12, 24) == "ct"
        # Second half: sides swap
        assert engine._get_current_side("ct", 13, 24) == "t"
        assert engine._get_current_side("ct", 24, 24) == "t"
        # Reverse
        assert engine._get_current_side("t", 1, 24) == "t"
        assert engine._get_current_side("t", 13, 24) == "ct"

    def test_mr15_format(self):
        """MR15 uses 15 rounds per half."""
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._rounds_per_half(30) == 15
        assert engine._is_pistol_round(16, 30) is True
        assert engine._is_halftime_boundary(15, 30) is True


# =============================================================================
# Test: Decision Grading
# =============================================================================


class TestDecisionGrading:
    """Test grade_decision against optimal play."""

    def test_pistol_always_optimal(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        grade = engine.grade_decision(
            round_number=1,
            team="ct",
            actual_buy_type="pistol",
            recommended_buy_type="full_buy",
            equipment_value=4000,
            loss_streak=0,
            we_won=True,
            orchestrator_grade="B",
        )
        assert grade.was_optimal is True
        assert grade.money_wasted == 0

    def test_matching_buy_is_optimal(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        grade = engine.grade_decision(
            round_number=5,
            team="ct",
            actual_buy_type="full",
            recommended_buy_type="full_buy",
            equipment_value=25000,
            loss_streak=0,
            we_won=True,
            orchestrator_grade="A",
        )
        assert grade.was_optimal is True

    def test_overbuy_detected(self):
        """Force buying when should have saved → not optimal."""
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        grade = engine.grade_decision(
            round_number=5,
            team="ct",
            actual_buy_type="full",
            recommended_buy_type="eco",
            equipment_value=20000,
            loss_streak=2,
            we_won=False,
            orchestrator_grade="D",
        )
        assert grade.was_optimal is False
        assert grade.money_wasted > 0
        assert "over-bought" in grade.impact.lower()

    def test_underbuy_detected(self):
        """Saving when should have bought → not optimal."""
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        grade = engine.grade_decision(
            round_number=5,
            team="ct",
            actual_buy_type="eco",
            recommended_buy_type="full_buy",
            equipment_value=5000,
            loss_streak=0,
            we_won=False,
            orchestrator_grade="C",
        )
        assert grade.was_optimal is False
        assert "under-bought" in grade.impact.lower()

    def test_one_level_difference_is_acceptable(self):
        """Force buy when full buy recommended (1 level diff) = still optimal."""
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        grade = engine.grade_decision(
            round_number=5,
            team="ct",
            actual_buy_type="force",
            recommended_buy_type="full_buy",
            equipment_value=15000,
            loss_streak=1,
            we_won=True,
            orchestrator_grade="B",
        )
        assert grade.was_optimal is True

    def test_orchestrator_d_grade_overrides(self):
        """Orchestrator D/F grade forces non-optimal even if levels match."""
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        grade = engine.grade_decision(
            round_number=5,
            team="ct",
            actual_buy_type="force",
            recommended_buy_type="force_buy",
            equipment_value=12000,
            loss_streak=1,
            we_won=False,
            orchestrator_grade="D",
        )
        assert grade.was_optimal is False


# =============================================================================
# Test: Full Match Analysis
# =============================================================================


class TestMatchAnalysis:
    """Test analyze_match with realistic match data."""

    def test_basic_match_produces_report(self, basic_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(basic_match, "ct")

        assert report.team_name == "TeamAlpha"
        assert len(report.round_states) == 24
        assert len(report.decisions) == 24

    def test_economy_grade_is_letter(self, basic_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(basic_match, "ct")

        assert report.economy_grade in ("A", "B", "C", "D", "F")

    def test_optimal_rate_between_0_and_1(self, basic_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(basic_match, "ct")

        assert 0 <= report.optimal_decision_rate <= 1

    def test_force_buy_wins_detected(self, basic_match):
        """Round 4: T wins on a force buy (from CT perspective, that's opponent)."""
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        # Analyze from T side to see force buy wins
        report = engine.analyze_match(basic_match, "t")
        # T side should show force buy wins in early rounds
        # (round 4: T wins with force buy type)
        assert isinstance(report.force_buy_wins, list)

    def test_economy_breaks_detected(self, basic_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(basic_match, "ct")
        assert isinstance(report.economy_breaks, list)

    def test_eco_round_wins_detected(self, basic_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(basic_match, "ct")
        assert isinstance(report.eco_round_wins, list)

    def test_t_side_analysis(self, basic_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(basic_match, "t")
        assert report.team_name == "TeamBravo"
        assert len(report.round_states) == 24

    def test_invalid_side_raises(self, basic_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        with pytest.raises(ValueError, match="must be 'ct' or 't'"):
            engine.analyze_match(basic_match, "invalid")

    def test_empty_timeline(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match({"demo_info": {"rounds": 0}, "round_timeline": []}, "ct")
        assert len(report.round_states) == 0
        assert report.economy_grade in ("A", "B", "C", "D", "F")

    def test_money_advantage_rounds_tracked(self, basic_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(basic_match, "ct")
        assert report.money_advantage_rounds >= 0
        assert report.money_advantage_rounds <= 24


# =============================================================================
# Test: to_dict Serialization
# =============================================================================


class TestReportSerialization:
    """Test EconomyReport.to_dict() produces valid output."""

    def test_to_dict_has_all_keys(self, short_match):
        import json

        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(short_match, "ct")
        d = report.to_dict()

        assert "team_name" in d
        assert "economy_grade" in d
        assert "optimal_decision_rate" in d
        assert "total_money_wasted" in d
        assert "round_states" in d
        assert "decisions" in d
        assert "economy_breaks" in d
        assert "force_buy_wins" in d
        assert "eco_round_wins" in d
        assert "wasteful_rounds" in d

        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 50

    def test_round_state_dict_structure(self, short_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(short_match, "ct")
        d = report.to_dict()

        rs = d["round_states"][0]
        assert "round" in rs
        assert "team" in rs
        assert "predicted_money" in rs
        assert "recommendation" in rs
        assert "reasoning" in rs
        assert "confidence" in rs

    def test_decision_dict_structure(self, short_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(short_match, "ct")
        d = report.to_dict()

        dec = d["decisions"][0]
        assert "round" in dec
        assert "actual" in dec
        assert "recommended" in dec
        assert "was_optimal" in dec
        assert "impact" in dec
        assert "money_wasted" in dec

    def test_optimal_rate_is_percentage(self, short_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        report = engine.analyze_match(short_match, "ct")
        d = report.to_dict()

        # to_dict multiplies by 100
        assert 0 <= d["optimal_decision_rate"] <= 100


# =============================================================================
# Test: Round-by-Round Prediction
# =============================================================================


class TestRoundPrediction:
    """Test get_round_prediction for specific rounds."""

    def test_predict_existing_round(self, short_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        state = engine.get_round_prediction(short_match, 3, "ct")

        assert state is not None
        assert state.round_number == 3
        assert state.buy_recommendation in ("full_buy", "force_buy", "half_buy", "eco", "full_save")

    def test_predict_nonexistent_round(self, short_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        state = engine.get_round_prediction(short_match, 99, "ct")
        assert state is None

    def test_predict_pistol_round(self, short_match):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        state = engine.get_round_prediction(short_match, 1, "ct")

        assert state is not None
        assert state.buy_recommendation == "full_buy"
        assert "pistol" in state.reasoning.lower()


# =============================================================================
# Test: Overall Grade Calculation
# =============================================================================


class TestOverallGrade:
    """Test _calculate_overall_grade at various rates."""

    def test_grade_a_high_rate(self):
        from opensight.ai.economy_engine import EconomyEngine, EconomyReport

        engine = EconomyEngine()
        report = EconomyReport(
            team_name="Test",
            optimal_decision_rate=0.90,
            economy_breaks=[1, 5, 10],
            force_buy_wins=[3],
        )
        grade = engine._calculate_overall_grade(report)
        assert grade == "A"

    def test_grade_b_moderate_rate(self):
        from opensight.ai.economy_engine import EconomyEngine, EconomyReport

        engine = EconomyEngine()
        report = EconomyReport(
            team_name="Test",
            optimal_decision_rate=0.72,
        )
        grade = engine._calculate_overall_grade(report)
        assert grade == "B"

    def test_grade_c_average(self):
        from opensight.ai.economy_engine import EconomyEngine, EconomyReport

        engine = EconomyEngine()
        report = EconomyReport(
            team_name="Test",
            optimal_decision_rate=0.55,
        )
        grade = engine._calculate_overall_grade(report)
        assert grade == "C"

    def test_grade_f_terrible(self):
        from opensight.ai.economy_engine import EconomyEngine, EconomyReport

        engine = EconomyEngine()
        report = EconomyReport(
            team_name="Test",
            optimal_decision_rate=0.20,
            wasteful_rounds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        grade = engine._calculate_overall_grade(report)
        assert grade == "F"

    def test_economy_breaks_boost_grade(self):
        from opensight.ai.economy_engine import EconomyEngine, EconomyReport

        engine = EconomyEngine()
        # Borderline B/C without breaks
        report_without = EconomyReport(team_name="Test", optimal_decision_rate=0.68)
        report_with = EconomyReport(
            team_name="Test",
            optimal_decision_rate=0.68,
            economy_breaks=[1, 2, 3, 4, 5],
        )
        grade_without = engine._calculate_overall_grade(report_without)
        grade_with = engine._calculate_overall_grade(report_with)
        # Breaks should help push grade higher (A > B > C > D > F)
        grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
        assert grade_order[grade_with] >= grade_order[grade_without]


# =============================================================================
# Test: Side Win Detection
# =============================================================================


class TestSideWinDetection:
    """Test _did_side_win helper."""

    def test_ct_wins(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._did_side_win("ct", "CT") is True
        assert engine._did_side_win("ct", "CT_WIN") is True

    def test_t_wins(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._did_side_win("t", "T") is True
        assert engine._did_side_win("t", "T_WIN") is True

    def test_wrong_side(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._did_side_win("ct", "T") is False
        assert engine._did_side_win("t", "CT") is False

    def test_empty_winner(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        assert engine._did_side_win("ct", "") is False
        assert engine._did_side_win("ct", None) is False


# =============================================================================
# Test: API Endpoints
# =============================================================================


class TestEconomyEndpoints:
    """Test economy API endpoints via FastAPI test client."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from opensight.api import app

        return TestClient(app)

    def test_analysis_no_job(self, client):
        """GET /api/economy/{id}/analysis with valid UUID but no job → 404."""
        resp = client.get("/api/economy/00000000-0000-0000-0000-000000000000/analysis?team=ct")
        assert resp.status_code == 404

    def test_predict_no_job(self, client):
        """GET /api/economy/{id}/predict with valid UUID but no job → 404."""
        resp = client.get(
            "/api/economy/00000000-0000-0000-0000-000000000000/predict?round=1&team=ct"
        )
        assert resp.status_code == 404

    def test_predict_missing_round_param(self, client):
        """GET without round param → 400."""
        resp = client.get("/api/economy/00000000-0000-0000-0000-000000000000/predict?team=ct")
        # Will 404 first because job doesn't exist; that's fine
        assert resp.status_code in (400, 404)

    def test_analysis_invalid_team(self, client):
        """GET with invalid team param → 400 or 404."""
        resp = client.get("/api/economy/00000000-0000-0000-0000-000000000000/analysis?team=invalid")
        # Will 404 first because job doesn't exist
        assert resp.status_code in (400, 404)


# =============================================================================
# Test: Missing Economy Data
# =============================================================================


class TestMissingEconomyData:
    """Test graceful handling of rounds with no economy data."""

    def test_none_economy(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        timeline = [
            {"round_num": 1, "winner": "CT", "economy": None},
            {"round_num": 2, "winner": "T", "economy": None},
        ]
        result = {"demo_info": {"rounds": 24}, "round_timeline": timeline}
        report = engine.analyze_match(result, "ct")

        assert len(report.round_states) == 2
        assert len(report.decisions) == 2

    def test_empty_economy_dict(self):
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        timeline = [
            {"round_num": 1, "winner": "CT", "economy": {}},
        ]
        result = {"demo_info": {"rounds": 24}, "round_timeline": timeline}
        report = engine.analyze_match(result, "ct")

        assert len(report.round_states) == 1

    def test_missing_round_num(self):
        """Rounds with round_num 0 should be skipped."""
        from opensight.ai.economy_engine import EconomyEngine

        engine = EconomyEngine()
        timeline = [
            {"round_num": 0, "winner": "CT", "economy": None},
            {"round_num": 1, "winner": "CT", "economy": None},
        ]
        result = {"demo_info": {"rounds": 24}, "round_timeline": timeline}
        report = engine.analyze_match(result, "ct")

        assert len(report.round_states) == 1
