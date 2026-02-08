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
from collections import Counter, defaultdict
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
            if round_num == 0:
                continue
            kills = round_data.get("kills") or []

            # Track our team's deaths as we iterate through kills in order
            team_dead_so_far: set[str] = set()
            first_team_death_seen = False

            for i, kill in enumerate(kills):
                victim = kill.get("victim", "")
                attacker = kill.get("killer", "")
                kill_time = kill.get("tick", 0)

                # Track our team deaths before any checks (order matters)
                is_our_death = victim in our_names
                if is_our_death:
                    team_dead_so_far.add(victim)

                # Only care about our team's deaths
                if not is_our_death:
                    continue

                # Skip suicides and world kills (no attacker to trade)
                if not attacker or attacker == victim:
                    continue

                # Skip the first death on our team each round (entry frag —
                # dying on the opening duel is part of the role, not a
                # failed trade)
                if not first_team_death_seen:
                    first_team_death_seen = True
                    continue

                # Check if attacker was killed within 5 seconds (320 ticks at 64 tick)
                trade_window = 320
                was_traded = False

                for subsequent_kill in kills[i + 1 :]:
                    if subsequent_kill.get("victim") == attacker:
                        if subsequent_kill.get("tick", 0) - kill_time <= trade_window:
                            was_traded = True
                            break

                if was_traded:
                    continue

                # Find teammates who were alive when victim died
                alive_teammates = [p for p in our_names if p not in team_dead_so_far]

                # Skip if no teammates alive to trade (last-alive / clutch situation)
                if not alive_teammates:
                    continue

                nearby = alive_teammates[:2]

                mistakes.append(
                    Mistake(
                        round_number=round_num,
                        mistake_type="failed_trade",
                        description=f"{victim} died to {attacker} and wasn't traded",
                        players_involved=nearby,
                        fix_suggestion=f"Players {', '.join(nearby)} should have traded within 5 seconds",
                        severity="high",
                    )
                )

        return mistakes

    def _find_nearby_teammates(self, round_data: dict, victim: str, our_names: set) -> list[str]:
        """Find teammates who were alive when victim died (computed from kills list)."""
        kills = round_data.get("kills") or []

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
            utility = round_data.get("utility") or []
            blinds = round_data.get("blinds") or []

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
        """Detect positioning mistakes.

        Note: ``was_dry_peek`` is computed by the orchestrator based on
        whether friendly utility was used within 192 ticks of the kill
        event.  If the field is absent we silently skip dry-peek
        detection rather than crashing.
        """
        our_names = {p.get("name", "") for p in our_players.values()}

        # Pre-check: if was_dry_peek is True on >80% of kills, the field
        # is unreliable (upstream detection broken) — skip entirely.
        all_kills = [k for r in round_timeline for k in (r.get("kills") or [])]
        dry_peek_count = sum(1 for k in all_kills if k.get("was_dry_peek") is True)
        if len(all_kills) > 10 and dry_peek_count / len(all_kills) > 0.8:
            logger.warning(
                "was_dry_peek=True on %d/%d kills (>80%%) — field unreliable, "
                "skipping positioning errors",
                dry_peek_count,
                len(all_kills),
            )
            return []

        mistakes = []
        for round_data in round_timeline:
            round_num = round_data.get("round_num", 0)
            kills = round_data.get("kills") or []
            round_positioning = 0

            for kill in kills:
                victim = kill.get("victim", "")
                if victim not in our_names:
                    continue

                dry_peek = kill.get("was_dry_peek")
                if dry_peek is None:
                    continue
                if dry_peek:
                    # Cap at 2 per round — more than that is repetitive
                    round_positioning += 1
                    if round_positioning > 2:
                        continue
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
        player_mistakes: dict[str, list[Mistake]] = defaultdict(list)
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
            rounds_played = stats.get("rounds_played", 1) or 1

            my_mistakes = player_mistakes[name]
            grade = self._calculate_grade(rating, adr, len(my_mistakes), rounds_played)

            # --- Strengths (stat-based) ---
            strengths: list[str] = []
            if rating > 1.3:
                strengths.append("Star player (elite rating)")
            elif rating > 1.1:
                strengths.append("High impact (good rating)")
            if adr > 100:
                strengths.append("Dominant damage output")
            elif adr > 85:
                strengths.append("Consistent damage output")
            if kills > deaths * 1.5:
                strengths.append("Strong K/D ratio")
            kast = rating_data.get("kast_percentage", 0)
            if kast > 75:
                strengths.append(f"High KAST ({kast:.0f}%)")

            # --- Weaknesses (stat + mistake-based) ---
            weaknesses: list[str] = []
            if rating < 0.8:
                weaknesses.append("Low impact (poor rating)")
            elif rating < 0.9:
                weaknesses.append("Below average impact")
            if adr < 50:
                weaknesses.append("Very low damage output")
            elif adr < 65:
                weaknesses.append("Low damage output")
            if deaths > kills * 1.5:
                weaknesses.append("Dying too often")

            # Add per-type weakness labels for frequent mistakes
            type_counts = Counter(m.mistake_type for m in my_mistakes)
            for mtype, count in type_counts.most_common(2):
                if count >= 2:
                    type_labels = {
                        "failed_trade": f"Failed to trade {count} times",
                        "wasted_utility": f"Wasted utility {count} times",
                        "economy": f"Economy errors ({count})",
                        "positioning": f"Positioning mistakes ({count})",
                    }
                    weaknesses.append(type_labels.get(mtype, f"{mtype} ({count})"))

            # --- Focus area: player's MOST COMMON mistake type ---
            # Sub-differentiate within failed_trade using player stats so
            # not everyone gets the same generic focus.
            focus_map = {
                "wasted_utility": "Utility effectiveness",
                "economy": "Economy decision-making",
                "positioning": "Peek discipline and angles",
            }
            if type_counts:
                most_common_type = type_counts.most_common(1)[0][0]
                if most_common_type == "failed_trade":
                    if rating > 1.2:
                        # Star player survives but teammates die untraded
                        focus = "Trade execution — position closer for refrags"
                    elif deaths > kills * 1.2:
                        focus = "Peek discipline — take fewer isolated fights"
                    else:
                        focus = "Trade timing and positioning"
                else:
                    focus = focus_map.get(most_common_type, "General improvement")
            elif rating < 0.9:
                focus = "Impact and damage output"
            else:
                focus = "Maintain current form"

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

    def _calculate_grade(
        self, rating: float, adr: float, mistake_count: int, rounds_played: int
    ) -> str:
        """Calculate a letter grade for a player.

        Normalizes mistake penalty by rounds played so a handful of
        mistakes across a long match doesn't tank an otherwise good grade.
        """
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

        # Mistake penalty: normalize by rounds, cap at -30
        mistakes_per_round = mistake_count / max(1, rounds_played)
        penalty = min(30, int(mistakes_per_round * 40))
        score -= penalty

        score = max(0, score)

        if score >= 60:
            return "A"
        elif score >= 45:
            return "B"
        elif score >= 30:
            return "C"
        elif score >= 15:
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
