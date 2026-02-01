"""
Tests for Economy IQ features.

Tests the new economy grading system including:
- Per-round economy grades (A-F)
- Full save error detection
- Bad force detection with loss bonus context
- Integration with round timeline
"""

from opensight.domains.economy import (
    BASE_LOSS_BONUS,
    HIGH_LOSS_BONUS,
    LOW_LOSS_BONUS,
    MAX_CONSECUTIVE_LOSSES,
    MAX_LOSS_BONUS,
    BuyDecision,
    BuyType,
    EconomyTracker,
    TeamRoundEconomy,
    analyze_round_buy,
    calculate_loss_bonus,
    calculate_round_economy_grade,
    is_bad_force,
    is_full_save_error,
    is_good_force,
)


class TestLossBonusCalculation:
    """Test loss bonus formula accuracy."""

    def test_zero_losses(self):
        """Zero losses should return base bonus."""
        assert calculate_loss_bonus(0) == BASE_LOSS_BONUS
        assert calculate_loss_bonus(0) == 1400

    def test_one_loss(self):
        """One loss should return $1900."""
        assert calculate_loss_bonus(1) == 1900

    def test_two_losses(self):
        """Two losses should return $2400."""
        assert calculate_loss_bonus(2) == 2400

    def test_three_losses(self):
        """Three losses should return $2900."""
        assert calculate_loss_bonus(3) == 2900

    def test_four_losses(self):
        """Four losses should return max $3400."""
        assert calculate_loss_bonus(4) == 3400

    def test_five_losses(self):
        """Five losses should return max $3400."""
        assert calculate_loss_bonus(5) == MAX_LOSS_BONUS
        assert calculate_loss_bonus(5) == 3400

    def test_more_than_max_losses_caps(self):
        """Losses beyond 5 should still return max $3400."""
        assert calculate_loss_bonus(6) == MAX_LOSS_BONUS
        assert calculate_loss_bonus(10) == MAX_LOSS_BONUS
        assert calculate_loss_bonus(100) == MAX_LOSS_BONUS


class TestBadForceDetection:
    """Test bad force buy detection."""

    def test_bad_force_with_low_loss_bonus(self):
        """Force buy at $1400 loss bonus is bad."""
        assert is_bad_force(BuyType.FORCE, 1400) is True

    def test_bad_force_at_low_threshold(self):
        """Force buy at or below LOW_LOSS_BONUS is bad."""
        assert is_bad_force(BuyType.FORCE, LOW_LOSS_BONUS) is True

    def test_not_bad_force_with_higher_loss_bonus(self):
        """Force buy above LOW_LOSS_BONUS is not bad."""
        assert is_bad_force(BuyType.FORCE, 2400) is False
        assert is_bad_force(BuyType.FORCE, 2900) is False

    def test_full_buy_never_bad_force(self):
        """Full buys are never flagged as bad force."""
        assert is_bad_force(BuyType.FULL_BUY, 1400) is False
        assert is_bad_force(BuyType.FULL_BUY, 1900) is False

    def test_eco_never_bad_force(self):
        """Ecos are never flagged as bad force."""
        assert is_bad_force(BuyType.ECO, 1400) is False

    def test_half_buy_can_be_bad_force(self):
        """Half buys can be flagged as bad force."""
        assert is_bad_force(BuyType.HALF_BUY, 1400) is True


class TestGoodForceDetection:
    """Test good force buy detection."""

    def test_good_force_with_high_loss_bonus(self):
        """Force buy at high loss bonus is good (getting max money anyway)."""
        assert is_good_force(BuyType.FORCE, HIGH_LOSS_BONUS) is True
        assert is_good_force(BuyType.FORCE, MAX_LOSS_BONUS) is True

    def test_good_force_when_enemy_broken(self):
        """Force buy when enemy is broken is good."""
        assert is_good_force(BuyType.FORCE, 1400, enemy_economy_broken=True) is True

    def test_not_good_force_low_bonus_enemy_not_broken(self):
        """Force buy with low bonus and enemy not broken is not good."""
        assert is_good_force(BuyType.FORCE, 1400, enemy_economy_broken=False) is False

    def test_full_buy_never_good_force(self):
        """Full buys are never flagged as good force (they're full buys)."""
        assert is_good_force(BuyType.FULL_BUY, HIGH_LOSS_BONUS) is False


class TestFullSaveErrorDetection:
    """Test full save error detection."""

    def test_full_save_error_when_rich_and_saving(self):
        """Rich team ($10k+) saving with low equipment is an error."""
        # Team has $15k, spent only $500, equipment $800
        assert is_full_save_error(15000, 500, 800, False) is True

    def test_not_full_save_error_when_poor(self):
        """Team with <$10k saving is not an error."""
        assert is_full_save_error(5000, 500, 800, False) is False
        assert is_full_save_error(9999, 500, 800, False) is False

    def test_not_full_save_error_when_bought_meaningful(self):
        """Team that bought meaningful equipment ($3k+) is not saving."""
        assert is_full_save_error(15000, 5000, 10000, False) is False
        assert is_full_save_error(15000, 3000, 3500, False) is False

    def test_not_full_save_error_on_pistol_round(self):
        """Pistol rounds never flag as full save error."""
        assert is_full_save_error(15000, 500, 800, True) is False

    def test_full_save_error_at_threshold(self):
        """Team at exactly $10k threshold triggers check."""
        # Threshold is >= $10k, so exactly $10k triggers (just barely)
        assert is_full_save_error(10000, 500, 800, False) is True
        # But $9999 does not trigger
        assert is_full_save_error(9999, 500, 800, False) is False


class TestRoundEconomyGrade:
    """Test per-round economy grading."""

    def test_full_buy_win_is_grade_a(self):
        """Full buy that wins is grade A."""
        grade, _ = calculate_round_economy_grade(BuyType.FULL_BUY, 1400, False, False, True)
        assert grade == "A"

    def test_full_buy_loss_is_grade_b(self):
        """Full buy that loses is grade B."""
        grade, _ = calculate_round_economy_grade(BuyType.FULL_BUY, 1400, False, False, False)
        assert grade == "B"

    def test_bad_force_loss_is_grade_d(self):
        """Bad force that loses is grade D."""
        grade, reason = calculate_round_economy_grade(BuyType.FORCE, 1400, True, False, False)
        assert grade == "D"
        assert "bad force" in reason.lower() or "loss bonus" in reason.lower()

    def test_bad_force_win_is_grade_c(self):
        """Bad force that wins is grade C (risky but paid off)."""
        grade, reason = calculate_round_economy_grade(BuyType.FORCE, 1400, True, False, True)
        assert grade == "C"
        assert "paid off" in reason.lower()

    def test_full_save_error_is_grade_d_or_f(self):
        """Full save error is grade D (if won) or F (if lost)."""
        grade_loss, _ = calculate_round_economy_grade(BuyType.ECO, 1400, False, True, False)
        assert grade_loss == "F"

        grade_win, _ = calculate_round_economy_grade(BuyType.ECO, 1400, False, True, True)
        assert grade_win == "D"

    def test_eco_win_is_grade_a(self):
        """Winning an eco round is grade A."""
        grade, reason = calculate_round_economy_grade(BuyType.ECO, 1400, False, False, True)
        assert grade == "A"
        assert "won eco" in reason.lower()

    def test_eco_loss_is_grade_b(self):
        """Losing an eco round properly is grade B."""
        grade, _ = calculate_round_economy_grade(BuyType.ECO, 1400, False, False, False)
        assert grade == "B"

    def test_justified_force_at_high_loss_bonus(self):
        """Force at high loss bonus is grade A/B."""
        grade_win, _ = calculate_round_economy_grade(
            BuyType.FORCE, HIGH_LOSS_BONUS, False, False, True
        )
        assert grade_win == "A"

        grade_loss, _ = calculate_round_economy_grade(
            BuyType.FORCE, HIGH_LOSS_BONUS, False, False, False
        )
        assert grade_loss == "B"


class TestTeamRoundEconomyNewFields:
    """Test new fields in TeamRoundEconomy dataclass."""

    def test_default_values(self):
        """Test default values for new fields."""
        tr = TeamRoundEconomy(
            round_num=1,
            team=2,
            total_equipment=5000,
            avg_equipment=1000,
            total_money=10000,
            total_spent=5000,
            buy_type=BuyType.FULL_BUY,
        )
        assert tr.decision_flag == "ok"
        assert tr.decision_grade == "B"
        assert tr.loss_bonus_next == BASE_LOSS_BONUS

    def test_custom_values(self):
        """Test custom values for new fields."""
        tr = TeamRoundEconomy(
            round_num=5,
            team=3,
            total_equipment=2000,
            avg_equipment=400,
            total_money=3000,
            total_spent=2000,
            buy_type=BuyType.FORCE,
            decision_flag="bad_force",
            decision_grade="D",
            loss_bonus_next=1900,
        )
        assert tr.decision_flag == "bad_force"
        assert tr.decision_grade == "D"
        assert tr.loss_bonus_next == 1900


class TestConstants:
    """Test economy constants are correctly defined."""

    def test_loss_bonus_constants(self):
        """Verify loss bonus constants match CS2 values."""
        assert BASE_LOSS_BONUS == 1400
        assert MAX_LOSS_BONUS == 3400
        assert MAX_CONSECUTIVE_LOSSES == 5
        assert LOW_LOSS_BONUS == 1900  # 1-2 losses
        assert HIGH_LOSS_BONUS == 2900  # 4+ losses

    def test_loss_bonus_formula(self):
        """Verify loss bonus follows CS2 formula: $1400 + losses*$500."""
        # losses=0: $1400 (just won)
        # losses=1: $1400 + 1*500 = $1900
        # losses=2: $1400 + 2*500 = $2400
        # losses=3: $1400 + 3*500 = $2900
        # losses=4: $1400 + 4*500 = $3400 (cap)
        for i in range(0, 6):
            expected = min(BASE_LOSS_BONUS + i * 500, MAX_LOSS_BONUS)
            assert calculate_loss_bonus(i) == expected


class TestEconomyTracker:
    """Tests for the stateful EconomyTracker class."""

    def test_initial_state(self):
        """Tracker should start with zero losses for both teams."""
        tracker = EconomyTracker()
        assert tracker.ct_loss_counter == 0
        assert tracker.t_loss_counter == 0

    def test_ct_win_resets_ct_increments_t(self):
        """CT win should reset CT losses and increment T losses."""
        tracker = EconomyTracker()
        tracker.ct_loss_counter = 3
        tracker.t_loss_counter = 2

        tracker.process_round_result("CT")

        assert tracker.ct_loss_counter == 0
        assert tracker.t_loss_counter == 3

    def test_t_win_resets_t_increments_ct(self):
        """T win should reset T losses and increment CT losses."""
        tracker = EconomyTracker()
        tracker.ct_loss_counter = 1
        tracker.t_loss_counter = 2

        tracker.process_round_result("T")

        assert tracker.t_loss_counter == 0
        assert tracker.ct_loss_counter == 2

    def test_loss_counter_caps_at_max(self):
        """Loss counter should cap at MAX_CONSECUTIVE_LOSSES - 1 (4)."""
        tracker = EconomyTracker()
        tracker.t_loss_counter = 4

        tracker.process_round_result("CT")  # T loses again

        assert tracker.t_loss_counter == 4  # Still 4, capped

    def test_get_loss_bonus_ct(self):
        """get_loss_bonus should return correct bonus for CT."""
        tracker = EconomyTracker()
        tracker.ct_loss_counter = 3

        bonus = tracker.get_loss_bonus("CT")

        assert bonus == calculate_loss_bonus(3)

    def test_get_loss_bonus_t(self):
        """get_loss_bonus should return correct bonus for T."""
        tracker = EconomyTracker()
        tracker.t_loss_counter = 2

        bonus = tracker.get_loss_bonus("T")

        assert bonus == calculate_loss_bonus(2)

    def test_reset_half(self):
        """reset_half should reset both counters to zero."""
        tracker = EconomyTracker()
        tracker.ct_loss_counter = 4
        tracker.t_loss_counter = 3

        tracker.reset_half()

        assert tracker.ct_loss_counter == 0
        assert tracker.t_loss_counter == 0

    def test_full_match_simulation(self):
        """Simulate a series of rounds to test tracker state."""
        tracker = EconomyTracker()

        # T wins first 3 rounds
        tracker.process_round_result("T")
        assert tracker.ct_loss_counter == 1
        assert tracker.t_loss_counter == 0

        tracker.process_round_result("T")
        assert tracker.ct_loss_counter == 2

        tracker.process_round_result("T")
        assert tracker.ct_loss_counter == 3

        # CT wins round 4 (eco break)
        tracker.process_round_result("CT")
        assert tracker.ct_loss_counter == 0
        assert tracker.t_loss_counter == 1

        # Check loss bonuses
        assert tracker.get_loss_bonus("CT") == 1400  # 0 losses
        assert tracker.get_loss_bonus("T") == 1900  # 1 loss


class TestAnalyzeRoundBuy:
    """Tests for the analyze_round_buy convenience function."""

    def test_pistol_round_returns_ok(self):
        """Pistol rounds should always return 'ok' flag and 'B' grade."""
        result = analyze_round_buy(
            team_spend=1000,
            team_bank=800,
            loss_bonus=1400,
            is_pistol_round=True,
        )

        assert result.flag == "ok"
        assert result.grade == "B"
        assert "Pistol" in result.reason

    def test_bad_force_detected(self):
        """High spend with low loss bonus should trigger bad_force."""
        result = analyze_round_buy(
            team_spend=3000,
            team_bank=4000,  # 75% spend ratio
            loss_bonus=1400,  # 0-1 consecutive losses
        )

        assert result.flag == "bad_force"
        assert result.grade == "D"
        assert "Risky" in result.reason

    def test_bad_eco_detected(self):
        """Low spend with rich bank should trigger bad_eco."""
        result = analyze_round_buy(
            team_spend=500,
            team_bank=12000,  # ~4% spend ratio with $12k bank
            loss_bonus=2400,
        )

        assert result.flag == "bad_eco"
        assert result.grade == "D"
        assert "Saving" in result.reason

    def test_full_buy_gets_a_grade(self):
        """Full buy (>80% spend) with adequate loss bonus should get A grade."""
        # Note: With low loss bonus ($1400) AND high spend ratio (>60%),
        # the function flags it as bad_force. Use a higher loss bonus
        # to test the full buy path without triggering bad_force check.
        result = analyze_round_buy(
            team_spend=18000,
            team_bank=20000,  # 90% spend ratio
            loss_bonus=2400,  # Not low enough to trigger bad_force
        )

        assert result.flag == "ok"
        assert result.grade == "A"
        assert "Full buy" in result.reason

    def test_justified_force_with_high_bonus(self):
        """Force buy with high loss bonus should be justified."""
        result = analyze_round_buy(
            team_spend=3000,
            team_bank=5000,  # 60% spend
            loss_bonus=HIGH_LOSS_BONUS,
        )

        assert result.flag == "ok"
        assert result.grade == "A"
        assert "Justified" in result.reason

    def test_eco_round_detection(self):
        """Low spend should be classified as eco round."""
        result = analyze_round_buy(
            team_spend=500,
            team_bank=5000,  # 10% spend, not rich enough for bad_eco
            loss_bonus=1400,
        )

        assert result.flag == "ok"
        assert result.grade == "B"  # B for eco with low loss bonus
        assert "Eco" in result.reason

    def test_spend_ratio_calculation(self):
        """Verify spend ratio is correctly calculated."""
        result = analyze_round_buy(
            team_spend=2500,
            team_bank=5000,
            loss_bonus=2000,
        )

        assert result.spend_ratio == 0.5
        assert result.loss_bonus == 2000


class TestBuyDecisionDataclass:
    """Tests for the BuyDecision dataclass."""

    def test_dataclass_fields(self):
        """BuyDecision should have all required fields."""
        decision = BuyDecision(
            grade="A",
            flag="ok",
            reason="Full buy",
            loss_bonus=1400,
            spend_ratio=0.9,
        )

        assert decision.grade == "A"
        assert decision.flag == "ok"
        assert decision.reason == "Full buy"
        assert decision.loss_bonus == 1400
        assert decision.spend_ratio == 0.9


# =============================================================================
# ECONOMY PREDICTION ENGINE TESTS
# =============================================================================


from opensight.domains.economy import (
    EconomyPrediction,
    EconomyPredictor,
)


class TestEconomyPredictionDataclass:
    """Tests for the EconomyPrediction dataclass."""

    def test_prediction_fields(self):
        """EconomyPrediction should have all required fields."""
        pred = EconomyPrediction(
            round_num=5,
            team="CT",
            predicted_buy="full",
            confidence=0.85,
            reasoning="High economy",
            estimated_team_money=25000,
            estimated_avg_loadout=4500,
            loss_bonus=1400,
            consecutive_losses=0,
        )

        assert pred.round_num == 5
        assert pred.team == "CT"
        assert pred.predicted_buy == "full"
        assert pred.confidence == 0.85
        assert pred.estimated_team_money == 25000


class TestEconomyPredictor:
    """Tests for the EconomyPredictor class."""

    def test_pistol_round_prediction(self):
        """Pistol rounds should always predict 'pistol' with 100% confidence."""
        predictor = EconomyPredictor()

        pred = predictor.predict_next_round(1, "CT", [])
        assert pred.predicted_buy == "pistol"
        assert pred.confidence == 1.0

        # Halftime pistol (round 13 in MR12)
        pred = predictor.predict_next_round(13, "T", [])
        assert pred.predicted_buy == "pistol"
        assert pred.confidence == 1.0

    def test_after_pistol_loss_predicts_eco(self):
        """After losing pistol, expect eco (low money)."""
        predictor = EconomyPredictor()

        # Simulate losing pistol round
        predictor.record_round_result("CT")  # T lost

        # Create minimal history showing T lost round 1
        history = [
            TeamRoundEconomy(
                round_num=1,
                team=2,
                total_equipment=4000,
                avg_equipment=800,
                total_money=4000,
                total_spent=4000,
                buy_type=BuyType.PISTOL,
                round_won=False,
            )
        ]

        pred = predictor.predict_next_round(2, "T", history)
        assert pred.predicted_buy == "eco"
        assert pred.confidence >= 0.65

    def test_high_money_predicts_full_buy(self):
        """With high team money, expect full buy."""
        predictor = EconomyPredictor()

        # Create history showing team has lots of money (won previous rounds)
        history = [
            TeamRoundEconomy(
                round_num=i,
                team=3,
                total_equipment=20000,
                avg_equipment=4000,
                total_money=25000,
                total_spent=20000,
                buy_type=BuyType.FULL_BUY,
                round_won=True,
            )
            for i in range(1, 5)
        ]

        pred = predictor.predict_next_round(5, "CT", history)
        assert pred.predicted_buy == "full"
        assert pred.confidence >= 0.80

    def test_consecutive_losses_tracked(self):
        """Predictor should track consecutive losses correctly."""
        predictor = EconomyPredictor()

        # T loses 3 rounds in a row
        for _ in range(3):
            predictor.record_round_result("CT")

        assert predictor._tracker.t_loss_counter == 3
        assert predictor._tracker.ct_loss_counter == 0

        # CT loses once
        predictor.record_round_result("T")

        assert predictor._tracker.t_loss_counter == 0  # Reset after win
        assert predictor._tracker.ct_loss_counter == 1

    def test_halftime_reset(self):
        """Loss counters should reset at halftime."""
        predictor = EconomyPredictor()

        # Build up losses
        predictor.record_round_result("CT")
        predictor.record_round_result("CT")
        predictor.record_round_result("CT")

        assert predictor._tracker.t_loss_counter == 3

        # Reset at halftime
        predictor.reset_half()

        assert predictor._tracker.t_loss_counter == 0
        assert predictor._tracker.ct_loss_counter == 0

    def test_high_loss_bonus_favors_force(self):
        """With high loss bonus, prediction should favor force buy."""
        predictor = EconomyPredictor()

        # Simulate 4 consecutive losses (max loss bonus)
        for _ in range(4):
            predictor.record_round_result("CT")

        # Create history with medium economy (force range)
        history = [
            TeamRoundEconomy(
                round_num=i,
                team=2,
                total_equipment=3000,
                avg_equipment=600,
                total_money=15000,
                total_spent=3000,
                buy_type=BuyType.ECO,
                round_won=False,
            )
            for i in range(1, 5)
        ]

        pred = predictor.predict_next_round(5, "T", history)
        # With $3400 loss bonus and medium economy, should predict force
        assert pred.predicted_buy in ("force", "eco")
        assert pred.loss_bonus == 3400  # Max loss bonus


class TestPredictionAccuracy:
    """Tests for prediction accuracy calculation."""

    def test_accuracy_with_no_predictions(self):
        """Empty predictor should return 0% accuracy."""
        predictor = EconomyPredictor()
        accuracy = predictor.get_accuracy()

        assert accuracy.total_predictions == 0
        assert accuracy.accuracy_pct == 0.0

    def test_accuracy_calculation(self):
        """Accuracy should be correctly calculated."""
        predictor = EconomyPredictor()

        # Make some predictions
        predictor._predictions = [
            EconomyPrediction(
                round_num=1,
                team="CT",
                predicted_buy="pistol",
                confidence=1.0,
                reasoning="",
                estimated_team_money=4000,
                estimated_avg_loadout=800,
                loss_bonus=1400,
                consecutive_losses=0,
            ),
            EconomyPrediction(
                round_num=2,
                team="CT",
                predicted_buy="eco",
                confidence=0.9,
                reasoning="",
                estimated_team_money=8000,
                estimated_avg_loadout=800,
                loss_bonus=1900,
                consecutive_losses=1,
            ),
            EconomyPrediction(
                round_num=3,
                team="CT",
                predicted_buy="full",
                confidence=0.85,
                reasoning="",
                estimated_team_money=25000,
                estimated_avg_loadout=4500,
                loss_bonus=1400,
                consecutive_losses=0,
            ),
        ]

        # Record actuals
        predictor._actuals[(1, "CT")] = "pistol"  # Correct
        predictor._actuals[(2, "CT")] = "eco"  # Correct
        predictor._actuals[(3, "CT")] = "force"  # Wrong

        accuracy = predictor.get_accuracy()

        assert accuracy.total_predictions == 3
        assert accuracy.correct_predictions == 2
        assert accuracy.accuracy_pct == 66.7  # 2/3 = 66.7%
