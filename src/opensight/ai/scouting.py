"""
Opponent Scouting Module for CS2 Demo Analysis.

Aggregates patterns across multiple demos from the same opponent
to build comprehensive scouting reports for pre-match preparation.

Features:
- Multi-demo aggregation
- Pattern frequency analysis
- Cross-demo tendency identification
- Anti-strat recommendations
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScoutedPattern:
    """A pattern aggregated across multiple demos."""

    pattern_type: str  # "default", "execute", "economy", "tendency"
    description: str
    frequency: float  # 0-1, how often this appears across demos
    demo_count: int  # How many demos this was seen in
    round_examples: list[dict] = field(default_factory=list)  # Example rounds


@dataclass
class PlayerProfile:
    """Aggregated profile for an opponent player."""

    player_name: str
    roles: list[str]  # e.g., ["AWP", "Entry"]
    tendencies: list[str]  # Specific habits
    avg_rating: float
    avg_adr: float
    signature_plays: list[str]  # Notable plays across demos


@dataclass
class ScoutReport:
    """Complete scouting report for an opponent."""

    opponent_team: str
    map_name: str | None  # If filtered by map
    demos_analyzed: int
    patterns: list[ScoutedPattern] = field(default_factory=list)
    player_profiles: list[PlayerProfile] = field(default_factory=list)
    anti_strat_recommendations: list[str] = field(default_factory=list)
    key_rounds_to_watch: list[dict] = field(default_factory=list)


class OpponentScout:
    """
    Aggregates analysis across multiple demos from the same opponent.

    Used to build comprehensive scouting reports before matches.
    """

    def __init__(self):
        """Initialize the scout."""
        self.demos: list[dict] = []  # Parsed demo results
        self.team_name: str = ""

    def add_demo(self, demo_result: dict, opponent_team: str) -> None:
        """
        Add a parsed demo to the scouting database.

        Args:
            demo_result: Parsed match data from CachedAnalyzer
            opponent_team: Name of the opponent team to focus on
        """
        self.demos.append(
            {
                "data": demo_result,
                "opponent_team": opponent_team,
            }
        )
        self.team_name = opponent_team
        logger.info(f"Added demo to scout: team={opponent_team}, total={len(self.demos)}")

    def clear(self) -> None:
        """Clear all scouted demos."""
        self.demos = []
        self.team_name = ""

    def generate_scout_report(self, map_name: str | None = None) -> str:
        """
        Generate comprehensive scouting report from all added demos.

        Aggregates patterns across all demos, identifies consistent
        tendencies, and feeds to Claude for the final report.

        Args:
            map_name: Optional map filter (only analyze demos on this map)

        Returns:
            Markdown-formatted scouting report
        """
        if not self.demos:
            return "**No demos added for scouting.**\n\nAdd demos using the /api/scout/add-demo endpoint first."

        # Filter by map if specified
        filtered_demos = self.demos
        if map_name:
            filtered_demos = [
                d
                for d in self.demos
                if d["data"].get("match_info", {}).get("map", "").lower() == map_name.lower()
                or d["data"].get("match_info", {}).get("map", "").lower()
                == f"de_{map_name}".lower()
            ]
            if not filtered_demos:
                return (
                    f"**No demos found for map: {map_name}**\n\nAvailable demos are on other maps."
                )

        # Analyze each demo with the strat engine
        from opensight.ai.strat_engine import StratEngine

        engine = StratEngine()
        all_analyses = []

        for demo in filtered_demos:
            demo_data = demo["data"]
            opponent = demo["opponent_team"]

            try:
                analysis = engine.analyze(demo_data, team_focus=opponent)
                all_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze demo for scouting: {e}")
                continue

        if not all_analyses:
            return "**Analysis failed for all demos.**\n\nCheck demo data integrity."

        # Aggregate patterns
        report = self._aggregate_patterns(all_analyses, map_name)

        # Generate Claude report
        report_text = self._generate_claude_report(report, filtered_demos)

        return report_text

    def _aggregate_patterns(self, analyses: list, map_name: str | None) -> ScoutReport:
        """Aggregate patterns across multiple demo analyses."""
        # Count default setups
        default_counter = Counter()
        for analysis in analyses:
            for default in analysis.defaults:
                key = f"{default.side}:{default.setup_type}"
                default_counter[key] += 1

        # Count executes
        execute_counter = Counter()
        execute_success = defaultdict(list)
        for analysis in analyses:
            for execute in analysis.executes:
                key = f"{execute.site} execute"
                execute_counter[key] += 1
                execute_success[key].append(execute.success)

        # Build patterns
        patterns = []
        total_demos = len(analyses)

        # Add default patterns
        for key, count in default_counter.most_common(5):
            side, setup = key.split(":")
            patterns.append(
                ScoutedPattern(
                    pattern_type="default",
                    description=f"{side} default: {setup} distribution",
                    frequency=count / total_demos,
                    demo_count=count,
                )
            )

        # Add execute patterns
        for key, count in execute_counter.most_common(5):
            successes = execute_success[key]
            win_rate = sum(successes) / len(successes) if successes else 0
            patterns.append(
                ScoutedPattern(
                    pattern_type="execute",
                    description=f"{key} (success rate: {win_rate:.0%})",
                    frequency=count / total_demos,
                    demo_count=count,
                )
            )

        # Aggregate player profiles
        player_stats = defaultdict(
            lambda: {
                "ratings": [],
                "adrs": [],
                "awp_kills": 0,
                "entry_kills": 0,
            }
        )

        for analysis in analyses:
            for tendency in analysis.tendencies:
                name = tendency.player_name
                if tendency.tendency_type == "awp_player":
                    player_stats[name]["awp_kills"] += 1
                elif tendency.tendency_type == "entry_fragger":
                    player_stats[name]["entry_kills"] += 1

        player_profiles = []
        for name, stats in player_stats.items():
            roles = []
            if stats["awp_kills"] > 0:
                roles.append("AWP")
            if stats["entry_kills"] > 0:
                roles.append("Entry")

            if roles:
                player_profiles.append(
                    PlayerProfile(
                        player_name=name,
                        roles=roles,
                        tendencies=[],
                        avg_rating=sum(stats["ratings"]) / len(stats["ratings"])
                        if stats["ratings"]
                        else 0,
                        avg_adr=sum(stats["adrs"]) / len(stats["adrs"]) if stats["adrs"] else 0,
                        signature_plays=[],
                    )
                )

        return ScoutReport(
            opponent_team=self.team_name,
            map_name=map_name,
            demos_analyzed=total_demos,
            patterns=patterns,
            player_profiles=player_profiles,
        )

    def _generate_claude_report(self, report: ScoutReport, demos: list[dict]) -> str:
        """Generate the final scouting report using Claude."""
        from opensight.ai.llm_client import get_tactical_ai_client
        from opensight.ai.tactical import SYSTEM_PROMPT_SCOUT

        # Build aggregated match data for Claude
        aggregated_data = {
            "match_info": {
                "opponent_team": report.opponent_team,
                "map": report.map_name or "multiple maps",
                "demos_analyzed": report.demos_analyzed,
            },
            "scouting_patterns": {
                "defaults": [
                    {"type": p.description, "frequency": p.frequency, "demos": p.demo_count}
                    for p in report.patterns
                    if p.pattern_type == "default"
                ],
                "executes": [
                    {"type": p.description, "frequency": p.frequency, "demos": p.demo_count}
                    for p in report.patterns
                    if p.pattern_type == "execute"
                ],
                "player_profiles": [
                    {"name": p.player_name, "roles": p.roles} for p in report.player_profiles
                ],
            },
            "players": {},
            "round_timeline": [],
        }

        # Add round data from first demo for context
        if demos:
            first_demo = demos[0]["data"]
            aggregated_data["players"] = first_demo.get("players", {})
            aggregated_data["round_timeline"] = first_demo.get("round_timeline", [])[:10]

        try:
            ai = get_tactical_ai_client()
            report_text = ai.analyze(
                match_data=aggregated_data,
                analysis_type="scout",
                focus=f"Team: {report.opponent_team}, Demos: {report.demos_analyzed}",
                system_prompt=SYSTEM_PROMPT_SCOUT,
            )
            return report_text
        except Exception as e:
            logger.error(f"Claude report generation failed: {e}")
            # Return a basic report without AI
            return self._generate_basic_report(report)

    def _generate_basic_report(self, report: ScoutReport) -> str:
        """Generate a basic report without AI (fallback)."""
        lines = [
            f"# {report.opponent_team} Scouting Report",
            f"**Map:** {report.map_name or 'Multiple Maps'}",
            f"**Demos Analyzed:** {report.demos_analyzed}",
            "",
            "## Patterns Detected",
            "",
        ]

        for pattern in report.patterns:
            lines.append(f"- **{pattern.description}**")
            lines.append(f"  - Frequency: {pattern.frequency:.0%} ({pattern.demo_count} demos)")
            lines.append("")

        if report.player_profiles:
            lines.append("## Key Players")
            lines.append("")
            for player in report.player_profiles:
                roles = ", ".join(player.roles) if player.roles else "Unknown"
                lines.append(f"- **{player.player_name}**: {roles}")
            lines.append("")

        lines.append("---")
        lines.append("*Note: AI analysis unavailable. This is a basic pattern summary.*")

        return "\n".join(lines)


# Singleton instance
_scout_instance: OpponentScout | None = None


def get_opponent_scout() -> OpponentScout:
    """Get or create singleton OpponentScout instance."""
    global _scout_instance
    if _scout_instance is None:
        _scout_instance = OpponentScout()
    return _scout_instance
