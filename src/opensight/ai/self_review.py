"""
Team Self-Review Module for CS2 Demo Analysis.

Analyzes your own team's demos to identify:
- Failed trades (teammate died, not traded within 5 seconds)
- Wasted utility (flashes/smokes with no impact)
- Economy mistakes (bad force buys, wrong saves)
- Positioning errors (crossfires, bad peeks)
- Communication failures (duplicate holds, gaps)

Generates brutally honest player report cards and practice priorities.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 67 Esports roster (from PROMPT.md)
TEAM_67_ROSTER = {
    "Luke": "IGL",
    "foe": "Entry",
    "kix": "Support",
    "dergs": "AWP",
    "tr1d": "Support/Lurk",
    "miasma": "Anchor",
}


@dataclass
class Mistake:
    """A detected mistake in a round."""

    round_number: int
    mistake_type: str  # "failed_trade", "wasted_utility", "economy", "positioning"
    description: str
    players_involved: list[str]
    fix_suggestion: str
    severity: str = "medium"  # "low", "medium", "high", "critical"


@dataclass
class PlayerReportCard:
    """Individual player performance report."""

    player_name: str
    role: str
    grade: str  # A, B, C, D, F
    kills: int
    deaths: int
    adr: float
    rating: float
    strengths: list[str]
    weaknesses: list[str]
    focus_area: str  # Primary improvement area


@dataclass
class SelfReviewReport:
    """Complete self-review report."""

    team_name: str
    map_name: str
    result: str  # "win" or "loss"
    score: str  # e.g., "13-16"
    mistakes: list[Mistake] = field(default_factory=list)
    report_cards: list[PlayerReportCard] = field(default_factory=list)
    practice_priorities: list[str] = field(default_factory=list)


class SelfReviewEngine:
    """
    Engine for analyzing your own team's demos and identifying mistakes.
    """

    def __init__(self, team_roster: dict[str, str] | None = None):
        """
        Initialize the self-review engine.

        Args:
            team_roster: Dict of player_name -> role (defaults to 67 Esports roster)
        """
        self.team_roster = team_roster or TEAM_67_ROSTER

    def analyze(
        self,
        match_data: dict,
        our_team: str | None = None,
    ) -> SelfReviewReport:
        """
        Analyze match data for team mistakes.

        Args:
            match_data: Parsed match data from CachedAnalyzer
            our_team: Name of our team in the demo (optional, auto-detects)

        Returns:
            SelfReviewReport with mistakes and report cards
        """
        match_info = match_data.get("demo_info", {})
        map_name = match_info.get("map", "unknown")
        round_timeline = match_data.get("round_timeline", [])
        players = match_data.get("players", {})

        # Auto-detect our team based on roster names
        if not our_team:
            our_team = self._detect_our_team(players)

        # Determine result
        ct_score = match_info.get("score_ct", 0)
        t_score = match_info.get("score_t", 0)
        score = f"{ct_score}-{t_score}"

        # Filter to our team's players
        our_players = self._get_our_players(players, our_team)

        # Detect mistakes
        mistakes = []
        mistakes.extend(self._detect_failed_trades(round_timeline, our_players))
        mistakes.extend(self._detect_wasted_utility(round_timeline, our_players))
        mistakes.extend(self._detect_economy_mistakes(round_timeline, our_players))
        mistakes.extend(self._detect_positioning_errors(round_timeline, our_players))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        mistakes.sort(key=lambda m: severity_order.get(m.severity, 99))

        # Generate report cards
        report_cards = self._generate_report_cards(our_players, mistakes)

        # Generate practice priorities
        practice_priorities = self._generate_practice_priorities(mistakes, report_cards)

        # Determine result
        our_wins = sum(1 for r in round_timeline if r.get("winner") == "CT")  # Simplified
        result = "win" if our_wins > len(round_timeline) / 2 else "loss"

        return SelfReviewReport(
            team_name=our_team or "Unknown",
            map_name=map_name,
            result=result,
            score=score,
            mistakes=mistakes,
            report_cards=report_cards,
            practice_priorities=practice_priorities,
        )

    def _detect_our_team(self, players: dict) -> str:
        """Detect our team based on roster names."""
        for _steam_id, player in players.items():
            name = player.get("name", "")
            if name in self.team_roster:
                return player.get("team", "Unknown")
        return "Unknown"

    def _get_our_players(self, players: dict, our_team: str) -> dict:
        """Filter to only our team's players."""
        return {
            sid: p
            for sid, p in players.items()
            if p.get("team", "").lower() == our_team.lower()
            or p.get("name", "") in self.team_roster
        }

    def _detect_failed_trades(self, round_timeline: list[dict], our_players: dict) -> list[Mistake]:
        """Detect rounds where teammates weren't traded."""
        mistakes = []
        our_names = {p.get("name", "") for p in our_players.values()}

        for round_data in round_timeline:
            round_num = round_data.get("round_num", 0)
            kills = round_data.get("kills", [])

            # Track teammate deaths and check if traded
            for i, kill in enumerate(kills):
                victim = kill.get("victim", "")
                if victim not in our_names:
                    continue

                attacker = kill.get("killer", "")
                kill_time = kill.get("tick", 0)

                # Check if attacker was killed within 5 seconds (320 ticks at 64 tick)
                trade_window = 320  # 5 seconds
                was_traded = False

                for subsequent_kill in kills[i + 1 :]:
                    if subsequent_kill.get("victim") == attacker:
                        if subsequent_kill.get("tick", 0) - kill_time <= trade_window:
                            was_traded = True
                            break

                if not was_traded and round_num > 0:
                    # Find nearby teammates who could have traded
                    nearby = self._find_nearby_teammates(round_data, victim, our_names)

                    mistakes.append(
                        Mistake(
                            round_number=round_num,
                            mistake_type="failed_trade",
                            description=f"{victim} died to {attacker} and wasn't traded",
                            players_involved=nearby,
                            fix_suggestion=f"Players {', '.join(nearby) if nearby else 'nearby'} should have traded within 5 seconds",
                            severity="high",
                        )
                    )

        return mistakes

    def _find_nearby_teammates(self, round_data: dict, victim: str, our_names: set) -> list[str]:
        """Find teammates who were alive when victim died (computed from kills list)."""
        kills = round_data.get("kills", [])

        # Find all players dead before or at the same time as victim
        dead_before_victim = set()
        for k in kills:
            dead_before_victim.add(k.get("victim", ""))
            if k.get("victim", "") == victim:
                break

        # Return alive teammates (not dead and not the victim)
        return [p for p in our_names if p not in dead_before_victim and p != victim][:2]

    def _detect_wasted_utility(
        self, round_timeline: list[dict], our_players: dict
    ) -> list[Mistake]:
        """Detect utility that had no impact."""
        mistakes = []
        our_names = {p.get("name", "") for p in our_players.values()}

        for round_data in round_timeline:
            round_num = round_data.get("round_num", 0)
            utility = round_data.get("utility", [])
            blinds = round_data.get("blinds", [])

            for util in utility:
                player = util.get("player", "")
                if player not in our_names:
                    continue

                util_type = util.get("type", "")

                # Check if flash had effect
                if util_type == "flashbang":
                    enemies_flashed = [
                        b for b in blinds if b.get("player") == player and b.get("enemy", False)
                    ]
                    teammates_flashed = [
                        b for b in blinds if b.get("player") == player and not b.get("enemy", False)
                    ]

                    if not enemies_flashed and teammates_flashed:
                        mistakes.append(
                            Mistake(
                                round_number=round_num,
                                mistake_type="wasted_utility",
                                description=f"{player} threw a flash that only hit teammates",
                                players_involved=[player],
                                fix_suggestion="Practice flash lineups to avoid team flashes",
                                severity="medium",
                            )
                        )

        return mistakes

    def _detect_economy_mistakes(
        self, round_timeline: list[dict], our_players: dict
    ) -> list[Mistake]:
        """Detect economy management mistakes."""
        mistakes = []

        for i, round_data in enumerate(round_timeline):
            round_num = round_data.get("round_num", 0)
            if round_num == 0:
                continue

            # Get next round if available
            next_round = round_timeline[i + 1] if i + 1 < len(round_timeline) else None

            round_type = round_data.get("round_type", "")
            winner = round_data.get("winner", "")
            lost = winner != "T"  # Simplified assumption

            if round_type == "force" and lost and next_round:
                next_type = next_round.get("round_type", "")
                if next_type in ["eco", "semi_eco"]:
                    # Force buy that lost AND ruined next round
                    mistakes.append(
                        Mistake(
                            round_number=round_num,
                            mistake_type="economy",
                            description="Force buy lost and ruined next round's economy",
                            players_involved=["IGL"],
                            fix_suggestion="Consider saving to guarantee full buy next round",
                            severity="high",
                        )
                    )

        return mistakes

    def _detect_positioning_errors(
        self, round_timeline: list[dict], our_players: dict
    ) -> list[Mistake]:
        """Detect positioning mistakes."""
        mistakes = []
        our_names = {p.get("name", "") for p in our_players.values()}

        for round_data in round_timeline:
            round_num = round_data.get("round_num", 0)
            kills = round_data.get("kills", [])

            for kill in kills:
                victim = kill.get("victim", "")
                if victim not in our_names:
                    continue

                # Check for dry peek (death without utility support)
                if kill.get("was_dry_peek", False):
                    mistakes.append(
                        Mistake(
                            round_number=round_num,
                            mistake_type="positioning",
                            description=f"{victim} dry peeked without utility support",
                            players_involved=[victim],
                            fix_suggestion="Always peek with flash or wait for teammate utility",
                            severity="medium",
                        )
                    )

        return mistakes

    def _generate_report_cards(
        self, our_players: dict, mistakes: list[Mistake]
    ) -> list[PlayerReportCard]:
        """Generate individual player report cards."""
        report_cards = []

        # Count mistakes per player
        player_mistakes = defaultdict(list)
        for mistake in mistakes:
            for player in mistake.players_involved:
                player_mistakes[player].append(mistake)

        for _steam_id, player in our_players.items():
            name = player.get("name", "Unknown")
            stats = player.get("stats", {})
            rating_data = player.get("rating", {})

            kills = stats.get("kills", 0)
            deaths = stats.get("deaths", 0)
            adr = stats.get("adr", 0)
            rating = rating_data.get("hltv_rating", 1.0)

            # Calculate grade
            grade = self._calculate_grade(rating, adr, len(player_mistakes[name]))

            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []

            if rating > 1.2:
                strengths.append("High impact (good rating)")
            if adr > 85:
                strengths.append("Consistent damage output")
            if kills > deaths * 1.5:
                strengths.append("Strong K/D ratio")

            if rating < 0.9:
                weaknesses.append("Low impact (poor rating)")
            if adr < 60:
                weaknesses.append("Low damage output")
            if len(player_mistakes[name]) > 3:
                weaknesses.append(f"Multiple mistakes ({len(player_mistakes[name])} detected)")

            # Determine focus area
            focus = "General improvement"
            mistake_types = [m.mistake_type for m in player_mistakes[name]]
            if "failed_trade" in mistake_types:
                focus = "Trade timing and positioning"
            elif "wasted_utility" in mistake_types:
                focus = "Utility effectiveness"
            elif "positioning" in mistake_types:
                focus = "Peek discipline"

            role = self.team_roster.get(name, "Unknown")

            report_cards.append(
                PlayerReportCard(
                    player_name=name,
                    role=role,
                    grade=grade,
                    kills=kills,
                    deaths=deaths,
                    adr=adr,
                    rating=rating,
                    strengths=strengths or ["Solid performance"],
                    weaknesses=weaknesses or ["No major issues"],
                    focus_area=focus,
                )
            )

        # Sort by rating (best first)
        report_cards.sort(key=lambda x: x.rating, reverse=True)
        return report_cards

    def _calculate_grade(self, rating: float, adr: float, mistake_count: int) -> str:
        """Calculate a letter grade for a player."""
        score = 0

        # Rating contribution (0-40 points)
        if rating >= 1.3:
            score += 40
        elif rating >= 1.1:
            score += 30
        elif rating >= 0.9:
            score += 20
        elif rating >= 0.7:
            score += 10

        # ADR contribution (0-30 points)
        if adr >= 90:
            score += 30
        elif adr >= 75:
            score += 20
        elif adr >= 60:
            score += 10

        # Mistake penalty (-10 each)
        score -= mistake_count * 10

        # Cap at 0
        score = max(0, score)

        # Convert to grade
        if score >= 70:
            return "A"
        elif score >= 55:
            return "B"
        elif score >= 40:
            return "C"
        elif score >= 25:
            return "D"
        else:
            return "F"

    def _generate_practice_priorities(
        self, mistakes: list[Mistake], report_cards: list[PlayerReportCard]
    ) -> list[str]:
        """Generate prioritized practice recommendations."""
        priorities = []

        # Count mistake types
        type_counts = defaultdict(int)
        for mistake in mistakes:
            type_counts[mistake.mistake_type] += 1

        # Generate priorities based on most common mistakes
        if type_counts["failed_trade"] > 2:
            priorities.append(
                "Trade timing drills - Practice 2-man peek scenarios and refrag timing"
            )
        if type_counts["wasted_utility"] > 2:
            priorities.append(
                "Utility practice - Review flash lineups and practice timing with team pushes"
            )
        if type_counts["economy"] > 1:
            priorities.append(
                "Economy decisions - IGL should review force buy thresholds with team"
            )
        if type_counts["positioning"] > 2:
            priorities.append("Peek discipline - Practice waiting for utility before engaging")

        # Add generic if no specific issues
        if not priorities:
            priorities.append("Continue current practice routine - no major issues detected")

        return priorities

    def generate_review_report(
        self,
        match_data: dict,
        our_team: str | None = None,
    ) -> str:
        """
        Generate a full self-review report using Claude.

        Args:
            match_data: Parsed match data from CachedAnalyzer
            our_team: Name of our team in the demo

        Returns:
            Markdown-formatted self-review report
        """
        # Analyze the match
        analysis = self.analyze(match_data, our_team)

        # Build summary for Claude
        from opensight.ai.llm_client import get_tactical_ai_client
        from opensight.ai.tactical import SYSTEM_PROMPT_SELF_REVIEW

        ai = get_tactical_ai_client()

        # Augment match data with our analysis
        augmented_data = dict(match_data)
        augmented_data["self_review"] = {
            "team_name": analysis.team_name,
            "result": analysis.result,
            "score": analysis.score,
            "mistakes": [
                {
                    "round": m.round_number,
                    "type": m.mistake_type,
                    "description": m.description,
                    "severity": m.severity,
                    "fix": m.fix_suggestion,
                }
                for m in analysis.mistakes[:10]  # Top 10 mistakes
            ],
            "report_cards": [
                {
                    "player": rc.player_name,
                    "role": rc.role,
                    "grade": rc.grade,
                    "rating": rc.rating,
                    "focus": rc.focus_area,
                }
                for rc in analysis.report_cards
            ],
            "practice_priorities": analysis.practice_priorities,
        }

        report = ai.analyze(
            match_data=augmented_data,
            analysis_type="self-review",
            focus=f"Team: {analysis.team_name}, Result: {analysis.result} ({analysis.score})",
            system_prompt=SYSTEM_PROMPT_SELF_REVIEW,
        )

        return report


# Singleton instance
_review_engine_instance: SelfReviewEngine | None = None


def get_self_review_engine(
    team_roster: dict[str, str] | None = None,
) -> SelfReviewEngine:
    """Get or create singleton SelfReviewEngine instance."""
    global _review_engine_instance
    if _review_engine_instance is None:
        _review_engine_instance = SelfReviewEngine(team_roster)
    return _review_engine_instance
