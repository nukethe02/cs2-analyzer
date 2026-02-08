"""
Economy Management Engine for CS2.

Analyzes economy decisions, predicts opponent economy, recommends buy/save/force
decisions, and grades past economy choices. Entirely heuristic-based — no LLM needed.

Uses orchestrator round_timeline economy data which already contains per-round
equipment values, buy types, loss bonus tracking, and decision grades from
the domains/economy.py analysis.

This module adds the INTELLIGENCE layer:
  - Opponent economy prediction based on round results
  - Buy recommendation engine (full_buy / force / half_buy / eco / full_save)
  - Decision grading against optimal play
  - Full-match economy report with grade A-F
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# CS2 Economy Constants
# =============================================================================

# Loss bonus ladder: $1400 → $1900 → $2400 → $2900 → $3400 (cap)
LOSS_BONUS_LADDER = [1400, 1900, 2400, 2900, 3400]

# Round rewards
WIN_REWARD = 3250  # base round win reward
LOSS_REWARD_BASE = 1400  # first loss (same as LOSS_BONUS_LADDER[0])
BOMB_PLANT_BONUS = 800  # T-side gets $800 for planting (even on loss)
DEFUSE_KIT_BONUS = 300  # CT bonus for defusing (varies, approximation)

# Kill rewards by weapon category
KILL_REWARDS = {
    "rifle": 300,
    "sniper": 100,  # AWP, Scout
    "smg": 600,
    "shotgun": 900,
    "pistol": 300,
    "knife": 1500,
    "grenade": 300,
    "default": 300,
}

# Starting money
PISTOL_ROUND_MONEY = 800
HALF_TIME_MONEY = 800  # MR12 format

# Buy thresholds (per player)
FULL_BUY_THRESHOLD = 4750  # rifle + kevlar + utility
FORCE_BUY_MIN = 2000  # minimum for a meaningful force
ECO_THRESHOLD = 1500  # below this, save everything

# Team thresholds (5 players)
TEAM_FULL_BUY = 23750  # 5 × $4750
TEAM_FORCE_MIN = 10000  # minimum for team force
TEAM_ECO_MAX = 7500  # below this, full save

# Match point / critical round logic
MR12_TOTAL = 24  # regulation rounds in MR12


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class EconomyState:
    """Predicted economy state for one team at a given round."""

    team: str  # "ct" or "t"
    round_number: int
    predicted_money: int
    loss_streak: int
    buy_recommendation: str  # "full_buy", "force_buy", "half_buy", "eco", "full_save"
    confidence: float  # 0-1, how confident in the prediction
    reasoning: str  # why this recommendation


@dataclass
class EconomyDecisionGrade:
    """Evaluation of an actual economy decision made in the demo."""

    round_number: int
    team: str
    actual_buy_type: str  # from orchestrator: "pistol", "eco", "force", "half_buy", "full"
    recommended_buy_type: str
    equipment_value: int
    was_optimal: bool
    impact: str  # explanation of the decision's impact
    money_wasted: int  # if suboptimal, estimated waste


@dataclass
class EconomyReport:
    """Full economy analysis for a match."""

    team_name: str

    # Per-round state
    round_states: list[EconomyState] = field(default_factory=list)

    # Decision grades
    decisions: list[EconomyDecisionGrade] = field(default_factory=list)
    optimal_decision_rate: float = 0.0  # 0-1

    # Key moments
    economy_breaks: list[int] = field(default_factory=list)
    force_buy_wins: list[int] = field(default_factory=list)
    eco_round_wins: list[int] = field(default_factory=list)
    wasteful_rounds: list[int] = field(default_factory=list)

    # Summary
    money_advantage_rounds: int = 0
    total_money_wasted: int = 0
    economy_grade: str = "C"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "team_name": self.team_name,
            "economy_grade": self.economy_grade,
            "optimal_decision_rate": round(self.optimal_decision_rate * 100, 1),
            "total_money_wasted": self.total_money_wasted,
            "money_advantage_rounds": self.money_advantage_rounds,
            "economy_breaks": self.economy_breaks,
            "force_buy_wins": self.force_buy_wins,
            "eco_round_wins": self.eco_round_wins,
            "wasteful_rounds": self.wasteful_rounds,
            "round_states": [
                {
                    "round": rs.round_number,
                    "team": rs.team,
                    "predicted_money": rs.predicted_money,
                    "loss_streak": rs.loss_streak,
                    "recommendation": rs.buy_recommendation,
                    "confidence": round(rs.confidence, 2),
                    "reasoning": rs.reasoning,
                }
                for rs in self.round_states
            ],
            "decisions": [
                {
                    "round": d.round_number,
                    "team": d.team,
                    "actual": d.actual_buy_type,
                    "recommended": d.recommended_buy_type,
                    "equipment_value": d.equipment_value,
                    "was_optimal": d.was_optimal,
                    "impact": d.impact,
                    "money_wasted": d.money_wasted,
                }
                for d in self.decisions
            ],
        }


# =============================================================================
# Economy Engine
# =============================================================================


class EconomyEngine:
    """
    Analyzes economy decisions and generates buy recommendations.

    This is entirely heuristic-based — no LLM needed.
    Works with orchestrator output (round_timeline + demo_info).
    """

    def analyze_match(self, orchestrator_result: dict, team_side: str) -> EconomyReport:
        """
        Analyze economy decisions for a specific team across the match.

        Args:
            orchestrator_result: Full orchestrator output dict
            team_side: "ct" or "t" — which starting side to analyze

        Returns:
            EconomyReport with grades, recommendations, and key moments
        """
        team_side = team_side.lower()
        if team_side not in ("ct", "t"):
            raise ValueError(f"team_side must be 'ct' or 't', got '{team_side}'")

        timeline = orchestrator_result.get("round_timeline", [])
        demo_info = orchestrator_result.get("demo_info", {})
        total_rounds = demo_info.get("rounds", len(timeline))
        team_name = self._get_team_name(demo_info, team_side)

        if not timeline:
            return EconomyReport(team_name=team_name)

        report = EconomyReport(team_name=team_name)

        # Track state across rounds
        our_loss_streak = 0
        opp_loss_streak = 0
        our_score = 0
        opp_score = 0

        for rdata in timeline:
            round_num = rdata.get("round_num", 0)
            if round_num == 0:
                continue

            winner = (rdata.get("winner") or "").upper()
            economy = rdata.get("economy") or {}

            # After halftime, sides swap: the team that started CT is now T
            current_side = self._get_current_side(team_side, round_num, total_rounds)
            opp_side = "ct" if current_side == "t" else "t"

            our_econ = economy.get(current_side) or {}
            opp_econ = economy.get(opp_side) or {}

            our_equipment = our_econ.get("equipment", 0)
            opp_equipment = opp_econ.get("equipment", 0)
            our_buy_type = our_econ.get("buy_type", "unknown")
            our_grade = our_econ.get("decision_grade", "")

            # Determine if we won this round
            we_won = self._did_side_win(current_side, winner)

            # Track score
            if we_won:
                our_score += 1
            elif winner:  # opponent won (not a draw/unknown)
                opp_score += 1

            # Equipment advantage tracking
            if our_equipment > opp_equipment:
                report.money_advantage_rounds += 1

            # Predict opponent economy for this round
            opp_predicted_money = self.predict_opponent_economy(opp_loss_streak, opp_equipment)

            # Generate buy recommendation
            is_pistol = self._is_pistol_round(round_num, total_rounds)
            state = self.recommend_buy(
                team=current_side,
                round_number=round_num,
                our_money_estimate=our_equipment,
                opponent_predicted_money=opp_predicted_money,
                our_score=our_score,
                opp_score=opp_score,
                total_rounds=total_rounds,
                loss_streak=our_loss_streak,
                is_pistol=is_pistol,
            )
            report.round_states.append(state)

            # Grade the actual decision
            decision = self.grade_decision(
                round_number=round_num,
                team=current_side,
                actual_buy_type=our_buy_type,
                recommended_buy_type=state.buy_recommendation,
                equipment_value=our_equipment,
                loss_streak=our_loss_streak,
                we_won=we_won,
                orchestrator_grade=our_grade,
            )
            report.decisions.append(decision)

            if not decision.was_optimal:
                report.wasteful_rounds.append(round_num)
                report.total_money_wasted += decision.money_wasted

            # Detect key economy moments
            if we_won and our_buy_type == "force":
                report.force_buy_wins.append(round_num)
            if we_won and our_buy_type == "eco":
                report.eco_round_wins.append(round_num)

            # Economy break: opponent had full buy, we won
            opp_buy = opp_econ.get("buy_type", "")
            if we_won and opp_buy == "full":
                report.economy_breaks.append(round_num)

            # Update loss streaks
            if we_won:
                our_loss_streak = 0
                opp_loss_streak += 1
            elif winner:
                our_loss_streak += 1
                opp_loss_streak = 0

            # Halftime resets
            if self._is_halftime_boundary(round_num, total_rounds):
                our_loss_streak = 0
                opp_loss_streak = 0

        # Calculate summary stats
        gradeable = [d for d in report.decisions if d.actual_buy_type != "pistol"]
        if gradeable:
            optimal_count = sum(1 for d in gradeable if d.was_optimal)
            report.optimal_decision_rate = optimal_count / len(gradeable)

        report.economy_grade = self._calculate_overall_grade(report)

        return report

    def predict_opponent_economy(self, loss_streak: int, last_equipment: int) -> int:
        """
        Predict opponent's available money for the current round.

        Uses loss bonus ladder and last known equipment value as proxy.

        Args:
            loss_streak: Opponent's consecutive losses going into this round
            last_equipment: Opponent's last round equipment value

        Returns:
            Predicted total team money (approximate)
        """
        # Loss bonus from ladder
        capped_streak = min(loss_streak, len(LOSS_BONUS_LADDER) - 1)
        loss_bonus = LOSS_BONUS_LADDER[capped_streak] if loss_streak > 0 else 0

        if loss_streak == 0:
            # They won last round — estimate from win reward + surviving equipment
            # Rough estimate: winners keep ~60% of equipment + win reward
            predicted = WIN_REWARD * 5 + int(last_equipment * 0.6)
        else:
            # They lost — loss bonus × 5 players
            predicted = loss_bonus * 5

        return predicted

    def recommend_buy(
        self,
        team: str,
        round_number: int,
        our_money_estimate: int,
        opponent_predicted_money: int,
        our_score: int,
        opp_score: int,
        total_rounds: int,
        loss_streak: int,
        is_pistol: bool,
    ) -> EconomyState:
        """
        Recommend buy type for a round.

        Decision tree:
        1. Pistol round → always buy (armor + pistol upgrade)
        2. Match point for us → full buy if possible, force otherwise
        3. Match point for opponent → force buy (nothing to save for)
        4. Money ≥ full buy threshold → full buy
        5. Force buy zone + high loss bonus → force (building bonus)
        6. Force buy zone + low loss bonus → eco/save to guarantee next round
        7. Below eco threshold → full save
        """
        if is_pistol:
            return EconomyState(
                team=team,
                round_number=round_number,
                predicted_money=PISTOL_ROUND_MONEY * 5,
                loss_streak=0,
                buy_recommendation="full_buy",
                confidence=1.0,
                reasoning="Pistol round — buy armor + upgraded pistol",
            )

        rounds_per_half = self._rounds_per_half(total_rounds)
        max_score = rounds_per_half + 1  # MR12 = 13 to win

        # Match point scenarios
        is_our_match_point = our_score == max_score - 1
        is_opp_match_point = opp_score == max_score - 1

        if is_our_match_point:
            if our_money_estimate >= TEAM_FULL_BUY:
                return EconomyState(
                    team=team,
                    round_number=round_number,
                    predicted_money=our_money_estimate,
                    loss_streak=loss_streak,
                    buy_recommendation="full_buy",
                    confidence=0.95,
                    reasoning=f"Match point (score {our_score}-{opp_score}). Full buy to close it out.",
                )
            return EconomyState(
                team=team,
                round_number=round_number,
                predicted_money=our_money_estimate,
                loss_streak=loss_streak,
                buy_recommendation="force_buy",
                confidence=0.90,
                reasoning=f"Match point (score {our_score}-{opp_score}). Force buy — must win this round.",
            )

        if is_opp_match_point:
            return EconomyState(
                team=team,
                round_number=round_number,
                predicted_money=our_money_estimate,
                loss_streak=loss_streak,
                buy_recommendation="force_buy",
                confidence=0.90,
                reasoning=f"Opponent match point ({opp_score}-{our_score}). Force — nothing to save for.",
            )

        # Full buy zone
        if our_money_estimate >= TEAM_FULL_BUY:
            # Check if opponent is on eco — consider anti-eco setup
            if opponent_predicted_money < TEAM_ECO_MAX:
                return EconomyState(
                    team=team,
                    round_number=round_number,
                    predicted_money=our_money_estimate,
                    loss_streak=loss_streak,
                    buy_recommendation="full_buy",
                    confidence=0.90,
                    reasoning="Full buy vs predicted opponent eco. Consider SMGs for $600 kill reward.",
                )
            return EconomyState(
                team=team,
                round_number=round_number,
                predicted_money=our_money_estimate,
                loss_streak=loss_streak,
                buy_recommendation="full_buy",
                confidence=0.95,
                reasoning="Sufficient economy for rifles + utility + armor.",
            )

        # Force buy zone ($10K-$23.75K team)
        if our_money_estimate >= TEAM_FORCE_MIN:
            # High loss bonus means saving builds money faster
            capped = min(loss_streak, len(LOSS_BONUS_LADDER) - 1)
            current_bonus = LOSS_BONUS_LADDER[capped] if loss_streak > 0 else 0

            if loss_streak >= 3:
                # High loss bonus — can afford to save for guaranteed full buy
                return EconomyState(
                    team=team,
                    round_number=round_number,
                    predicted_money=our_money_estimate,
                    loss_streak=loss_streak,
                    buy_recommendation="half_buy",
                    confidence=0.70,
                    reasoning=f"Loss streak {loss_streak} (bonus ${current_bonus}). "
                    f"Half buy — loss bonus building, full save guarantees full buy next round.",
                )
            # Low loss bonus — force to try to break the cycle
            return EconomyState(
                team=team,
                round_number=round_number,
                predicted_money=our_money_estimate,
                loss_streak=loss_streak,
                buy_recommendation="force_buy",
                confidence=0.65,
                reasoning=f"Force buy zone (${our_money_estimate} team). "
                f"Loss streak {loss_streak} — force to break opponent economy.",
            )

        # Eco zone ($7.5K-$10K team)
        if our_money_estimate >= TEAM_ECO_MAX:
            if loss_streak >= 4:
                return EconomyState(
                    team=team,
                    round_number=round_number,
                    predicted_money=our_money_estimate,
                    loss_streak=loss_streak,
                    buy_recommendation="eco",
                    confidence=0.80,
                    reasoning=f"Eco round — save for guaranteed full buy. Loss bonus at max (${LOSS_BONUS_LADDER[-1]}).",
                )
            return EconomyState(
                team=team,
                round_number=round_number,
                predicted_money=our_money_estimate,
                loss_streak=loss_streak,
                buy_recommendation="eco",
                confidence=0.75,
                reasoning=f"Eco round (${our_money_estimate} team). Save for full buy next round.",
            )

        # Full save zone (< $7.5K team)
        return EconomyState(
            team=team,
            round_number=round_number,
            predicted_money=our_money_estimate,
            loss_streak=loss_streak,
            buy_recommendation="full_save",
            confidence=0.85,
            reasoning=f"Full save (${our_money_estimate} team). "
            f"Build economy through loss bonus (streak: {loss_streak}).",
        )

    def grade_decision(
        self,
        round_number: int,
        team: str,
        actual_buy_type: str,
        recommended_buy_type: str,
        equipment_value: int,
        loss_streak: int,
        we_won: bool,
        orchestrator_grade: str,
    ) -> EconomyDecisionGrade:
        """
        Grade an actual economy decision against the optimal recommendation.

        Uses both the recommendation engine and the orchestrator's own grade.
        """
        # Pistol rounds are always "optimal" — no decision to grade
        if actual_buy_type == "pistol":
            return EconomyDecisionGrade(
                round_number=round_number,
                team=team,
                actual_buy_type=actual_buy_type,
                recommended_buy_type="full_buy",
                equipment_value=equipment_value,
                was_optimal=True,
                impact="Pistol round — standard buy.",
                money_wasted=0,
            )

        # Map buy types to severity levels for comparison
        buy_levels = {
            "full_save": 0,
            "eco": 1,
            "half_buy": 2,
            "force_buy": 3,
            "force": 3,
            "full_buy": 4,
            "full": 4,
        }

        actual_level = buy_levels.get(actual_buy_type, 2)
        rec_level = buy_levels.get(recommended_buy_type, 2)

        # Decision is optimal if within 1 level of recommendation
        level_diff = actual_level - rec_level
        was_optimal = abs(level_diff) <= 1

        # Use orchestrator grade as tiebreaker
        if orchestrator_grade in ("D", "F"):
            was_optimal = False

        # Calculate money wasted
        money_wasted = 0
        impact = ""

        if level_diff > 1:
            # Over-bought (forced when should have saved)
            money_wasted = self._estimate_waste_overbuy(
                equipment_value, recommended_buy_type, loss_streak
            )
            impact = (
                f"Over-bought: {actual_buy_type} when {recommended_buy_type} was optimal. "
                f"~${money_wasted} wasted that could have guaranteed a full buy."
            )
        elif level_diff < -1:
            # Under-bought (saved when should have bought)
            impact = (
                f"Under-bought: {actual_buy_type} when {recommended_buy_type} was recommended. "
                f"Missed opportunity to invest in this round."
            )
        elif was_optimal:
            impact = f"Good decision: {actual_buy_type} aligned with recommendation."
        else:
            # Orchestrator flagged it bad
            impact = (
                f"Suboptimal: {actual_buy_type} (grade: {orchestrator_grade}). "
                f"Recommended: {recommended_buy_type}."
            )
            money_wasted = self._estimate_waste_overbuy(
                equipment_value, recommended_buy_type, loss_streak
            )

        return EconomyDecisionGrade(
            round_number=round_number,
            team=team,
            actual_buy_type=actual_buy_type,
            recommended_buy_type=recommended_buy_type,
            equipment_value=equipment_value,
            was_optimal=was_optimal,
            impact=impact,
            money_wasted=money_wasted,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_team_name(self, demo_info: dict, starting_side: str) -> str:
        """Get team name from demo_info based on starting side."""
        if starting_side == "ct":
            return demo_info.get("team1_name", "Counter-Terrorists")
        return demo_info.get("team2_name", "Terrorists")

    def _get_current_side(self, starting_side: str, round_num: int, total_rounds: int) -> str:
        """Determine current side accounting for halftime swap."""
        rounds_per_half = self._rounds_per_half(total_rounds)

        if round_num <= rounds_per_half:
            return starting_side
        # After halftime, sides swap
        return "t" if starting_side == "ct" else "ct"

    def _rounds_per_half(self, total_rounds: int) -> int:
        """Get rounds per half based on match format."""
        if total_rounds <= 24:
            return 12  # MR12
        return 15  # MR15

    def _is_pistol_round(self, round_num: int, total_rounds: int) -> bool:
        """Check if round is a pistol round (start of half or OT half)."""
        rounds_per_half = self._rounds_per_half(total_rounds)

        # First round of each half
        if round_num == 1 or round_num == rounds_per_half + 1:
            return True

        # Overtime pistol rounds (MR12: OT starts at round 25, every 3 rounds)
        regulation_rounds = rounds_per_half * 2
        if round_num > regulation_rounds:
            ot_round = round_num - regulation_rounds
            # OT is MR3 in CS2: rounds 1, 4, 7, 10, ... are pistol
            if (ot_round - 1) % 6 == 0 or (ot_round - 1) % 6 == 3:
                return True

        return False

    def _is_halftime_boundary(self, round_num: int, total_rounds: int) -> bool:
        """Check if this round is the last round of a half."""
        rounds_per_half = self._rounds_per_half(total_rounds)
        return round_num == rounds_per_half

    def _did_side_win(self, side: str, winner: str) -> bool:
        """Check if the given side won the round."""
        if not winner:
            return False
        winner_upper = winner.upper().strip()
        side_upper = side.upper().strip()
        # Must NOT match "T" inside "CT" — use exact or prefix matching
        if side_upper == "T":
            return winner_upper == "T" or winner_upper.startswith("T_")
        # CT: starts with "CT"
        return winner_upper.startswith(side_upper)

    def _estimate_waste_overbuy(
        self, equipment_value: int, recommended_buy: str, loss_streak: int
    ) -> int:
        """Estimate money wasted when over-buying."""
        # If recommended eco/save but bought, waste = equipment spent
        if recommended_buy in ("full_save", "eco"):
            return min(equipment_value, 15000)  # Cap at reasonable amount
        if recommended_buy == "half_buy":
            # Wasted the difference between full and half buy
            return min(equipment_value // 3, 5000)
        return 0

    def _calculate_overall_grade(self, report: EconomyReport) -> str:
        """Calculate overall economy grade A-F."""
        rate = report.optimal_decision_rate

        # Bonus for economy breaks (good play)
        break_bonus = min(len(report.economy_breaks) * 0.02, 0.10)
        # Penalty for wasteful rounds
        waste_penalty = min(len(report.wasteful_rounds) * 0.01, 0.10)
        # Bonus for force buy wins (clutch economic plays)
        force_bonus = min(len(report.force_buy_wins) * 0.03, 0.10)

        adjusted = rate + break_bonus + force_bonus - waste_penalty

        if adjusted >= 0.85:
            return "A"
        if adjusted >= 0.70:
            return "B"
        if adjusted >= 0.55:
            return "C"
        if adjusted >= 0.40:
            return "D"
        return "F"

    def get_round_prediction(
        self, orchestrator_result: dict, round_num: int, team_side: str
    ) -> EconomyState | None:
        """
        Get economy prediction and recommendation for a specific round.

        Args:
            orchestrator_result: Full orchestrator output dict
            round_num: Round number to predict for
            team_side: "ct" or "t" — starting side

        Returns:
            EconomyState for the round, or None if round not found
        """
        timeline = orchestrator_result.get("round_timeline", [])
        demo_info = orchestrator_result.get("demo_info", {})
        total_rounds = demo_info.get("rounds", len(timeline))

        team_side = team_side.lower()
        our_score = 0
        opp_score = 0
        our_loss_streak = 0
        opp_loss_streak = 0

        for rdata in timeline:
            rn = rdata.get("round_num", 0)
            if rn == 0:
                continue

            winner = (rdata.get("winner") or "").upper()
            economy = rdata.get("economy") or {}
            current_side = self._get_current_side(team_side, rn, total_rounds)
            opp_side = "ct" if current_side == "t" else "t"

            if rn == round_num:
                our_econ = economy.get(current_side) or {}
                opp_econ = economy.get(opp_side) or {}
                our_equipment = our_econ.get("equipment", 0)
                opp_equipment = opp_econ.get("equipment", 0)

                opp_predicted = self.predict_opponent_economy(opp_loss_streak, opp_equipment)
                is_pistol = self._is_pistol_round(rn, total_rounds)

                return self.recommend_buy(
                    team=current_side,
                    round_number=rn,
                    our_money_estimate=our_equipment,
                    opponent_predicted_money=opp_predicted,
                    our_score=our_score,
                    opp_score=opp_score,
                    total_rounds=total_rounds,
                    loss_streak=our_loss_streak,
                    is_pistol=is_pistol,
                )

            # Track state for rounds before the target
            we_won = self._did_side_win(current_side, winner)
            if we_won:
                our_score += 1
                our_loss_streak = 0
                opp_loss_streak += 1
            elif winner:
                opp_score += 1
                our_loss_streak += 1
                opp_loss_streak = 0

            if self._is_halftime_boundary(rn, total_rounds):
                our_loss_streak = 0
                opp_loss_streak = 0

        return None
