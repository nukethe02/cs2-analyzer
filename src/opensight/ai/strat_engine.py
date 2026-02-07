"""
Strat-Stealing Engine for CS2 Demo Analysis.

Analyzes demo data to detect:
- Default setups (player positions in first 30 seconds)
- Executes (coordinated site takes with utility)
- Economy patterns (buy behaviors, force thresholds)
- Player tendencies (AWP positions, entry patterns)

Then feeds these patterns to Claude for tactical report generation.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DefaultSetup:
    """A detected default setup pattern."""

    round_number: int
    side: str  # "T" or "CT"
    player_positions: dict[str, str]  # player_name -> callout
    setup_type: str  # e.g., "2-1-2" for distribution across map areas
    timestamp: float  # seconds into round when snapshot taken


@dataclass
class Execute:
    """A detected execute pattern."""

    round_number: int
    site: str  # "A" or "B"
    utility_sequence: list[dict]  # [{type, player, time}]
    entry_point: str  # Primary entry location
    success: bool  # Did they take the site?
    timing: float  # Seconds into round when execute started


@dataclass
class EconomyPattern:
    """Detected economy behavior pattern."""

    pattern_type: str  # "pistol", "force", "eco", "full_buy"
    round_numbers: list[int]
    average_equipment: float
    win_rate: float
    description: str


@dataclass
class PlayerTendency:
    """A player-specific tendency."""

    player_name: str
    tendency_type: str  # "awp_position", "entry_timing", "lurk_spot"
    description: str
    frequency: float  # 0-1, how often this occurs
    round_numbers: list[int]


@dataclass
class StratAnalysis:
    """Complete strat analysis result."""

    map_name: str
    team_name: str
    side: str  # "T", "CT", or "both"
    defaults: list[DefaultSetup] = field(default_factory=list)
    executes: list[Execute] = field(default_factory=list)
    economy: list[EconomyPattern] = field(default_factory=list)
    tendencies: list[PlayerTendency] = field(default_factory=list)


class StratEngine:
    """
    Engine for detecting tactical patterns from demo data.

    Extracts defaults, executes, economy patterns, and player tendencies
    from parsed match data.
    """

    def __init__(self):
        """Initialize the strat engine."""
        pass

    def analyze(
        self,
        match_data: dict,
        team_focus: str | None = None,
        side_focus: str | None = None,
    ) -> StratAnalysis:
        """
        Analyze match data for tactical patterns.

        Args:
            match_data: Parsed match data from CachedAnalyzer
            team_focus: Specific team name to focus on (optional)
            side_focus: "T", "CT", or None for both

        Returns:
            StratAnalysis with detected patterns
        """
        match_info = match_data.get("demo_info", {})
        map_name = match_info.get("map", "unknown")
        round_timeline = match_data.get("round_timeline", [])
        players = match_data.get("players", {})

        logger.info(
            f"Analyzing strats: map={map_name}, rounds={len(round_timeline)}, team={team_focus}, side={side_focus}"
        )

        # Detect team name if not specified
        if not team_focus:
            # Use the first team we find
            team_names = set()
            for p in players.values():
                team = p.get("team", "Unknown")
                if team and team not in ["Unknown", ""]:
                    team_names.add(team)
            team_focus = list(team_names)[0] if team_names else "Unknown"

        # Detect patterns
        defaults = self._detect_defaults(round_timeline, side_focus)
        executes = self._detect_executes(round_timeline, side_focus)
        economy = self._detect_economy_patterns(round_timeline, players)
        tendencies = self._detect_player_tendencies(round_timeline, players)

        return StratAnalysis(
            map_name=map_name,
            team_name=team_focus,
            side=side_focus or "both",
            defaults=defaults,
            executes=executes,
            economy=economy,
            tendencies=tendencies,
        )

    def _detect_defaults(
        self, round_timeline: list[dict], side_focus: str | None
    ) -> list[DefaultSetup]:
        """Detect default setup patterns from round data."""
        defaults = []

        for round_data in round_timeline:
            round_num = round_data.get("round_num", 0)
            if round_num == 0:
                continue

            # Get player positions at start of round (schema: list of PlayerPositionSnapshot)
            positions = round_data.get("player_positions", [])
            if not positions:
                continue

            # Analyze T-side default (rounds 1-12, 16-27)
            if side_focus in [None, "T"]:
                # Filter to T-side positions and convert to dict by player_name
                t_positions = {
                    pos.get("player_name", "Unknown"): pos
                    for pos in positions
                    if pos.get("side") == "T" and pos.get("is_alive", True)
                }
                if t_positions:
                    setup_type = self._classify_setup(t_positions)
                    defaults.append(
                        DefaultSetup(
                            round_number=round_num,
                            side="T",
                            player_positions={
                                name: pos.get("zone", "Unknown")  # Schema uses "zone" not "callout"
                                for name, pos in t_positions.items()
                            },
                            setup_type=setup_type,
                            timestamp=30.0,  # Snapshot at 30 seconds
                        )
                    )

            # Analyze CT-side default
            if side_focus in [None, "CT"]:
                ct_positions = {
                    pos.get("player_name", "Unknown"): pos
                    for pos in positions
                    if pos.get("side") == "CT" and pos.get("is_alive", True)
                }
                if ct_positions:
                    setup_type = self._classify_setup(ct_positions)
                    defaults.append(
                        DefaultSetup(
                            round_number=round_num,
                            side="CT",
                            player_positions={
                                name: pos.get("zone", "Unknown")  # Schema uses "zone" not "callout"
                                for name, pos in ct_positions.items()
                            },
                            setup_type=setup_type,
                            timestamp=30.0,
                        )
                    )

        return defaults

    def _classify_setup(self, positions: dict) -> str:
        """Classify a setup into a distribution pattern like 2-1-2."""
        # Group positions by general area
        areas = defaultdict(int)
        for _name, pos in positions.items():
            # Schema uses "zone" field for callout/position name
            zone = pos.get("zone", "Unknown") if isinstance(pos, dict) else pos
            # Simplify zone to area
            if "A" in zone.upper():
                areas["A"] += 1
            elif "B" in zone.upper():
                areas["B"] += 1
            elif "MID" in zone.upper():
                areas["Mid"] += 1
            else:
                areas["Other"] += 1

        # Build setup string
        counts = sorted(areas.values(), reverse=True)
        return "-".join(str(c) for c in counts)

    def _detect_executes(self, round_timeline: list[dict], side_focus: str | None) -> list[Execute]:
        """Detect execute patterns from round data."""
        executes = []

        for round_data in round_timeline:
            round_num = round_data.get("round_num", 0)
            if round_num == 0:
                continue

            # Only look at T-side rounds for executes
            if side_focus == "CT":
                continue

            # Check for coordinated utility usage followed by kills
            utility = round_data.get("utility", [])
            kills = round_data.get("kills", [])

            if len(utility) < 2 or len(kills) < 1:
                continue

            # Detect A site execute based on kill positions (victim_zone)
            # Utility events don't have a "zone" field, so we detect executes
            # by checking if T-side kills happened at a specific site after utility
            a_kills = [
                k
                for k in kills
                if "A" in (k.get("victim_zone") or "").upper() and k.get("killer_team") == "T"
            ]
            if len(a_kills) >= 1 and len(utility) >= 2:
                # Utility thrown before the kills is likely part of the execute
                first_kill_tick = a_kills[0].get("tick", 0)
                pre_kill_utility = [u for u in utility if u.get("tick", 0) <= first_kill_tick]
                if len(pre_kill_utility) >= 2:
                    executes.append(
                        Execute(
                            round_number=round_num,
                            site="A",
                            utility_sequence=[
                                {
                                    "type": u.get("type"),
                                    "player": u.get("player"),
                                    "time": u.get("time_seconds", 0),
                                }
                                for u in pre_kill_utility[:5]
                            ],
                            entry_point=a_kills[0].get("victim_zone", "A Site"),
                            success=round_data.get("winner") == "T",
                            timing=pre_kill_utility[0].get("time_seconds", 45.0),
                        )
                    )

            # Detect B site execute based on kill positions (victim_zone)
            b_kills = [
                k
                for k in kills
                if "B" in (k.get("victim_zone") or "").upper() and k.get("killer_team") == "T"
            ]
            if len(b_kills) >= 1 and len(utility) >= 2:
                first_kill_tick = b_kills[0].get("tick", 0)
                pre_kill_utility = [u for u in utility if u.get("tick", 0) <= first_kill_tick]
                if len(pre_kill_utility) >= 2:
                    executes.append(
                        Execute(
                            round_number=round_num,
                            site="B",
                            utility_sequence=[
                                {
                                    "type": u.get("type"),
                                    "player": u.get("player"),
                                    "time": u.get("time_seconds", 0),
                                }
                                for u in pre_kill_utility[:5]
                            ],
                            entry_point=b_kills[0].get("victim_zone", "B Site"),
                            success=round_data.get("winner") == "T",
                            timing=pre_kill_utility[0].get("time_seconds", 45.0),
                        )
                    )

        return executes

    def _detect_economy_patterns(
        self, round_timeline: list[dict], players: dict
    ) -> list[EconomyPattern]:
        """Detect economy behavior patterns."""
        patterns = []

        # Track round types
        pistol_rounds = []
        eco_rounds = []
        force_rounds = []
        full_buy_rounds = []

        for round_data in round_timeline:
            round_num = round_data.get("round_num", 0)
            round_type = round_data.get("round_type", "")
            winner = round_data.get("winner", "")

            if round_type == "pistol":
                pistol_rounds.append({"round": round_num, "won": winner == "T"})
            elif round_type == "eco":
                eco_rounds.append({"round": round_num, "won": winner == "T"})
            elif round_type in ["force", "semi_eco"]:
                force_rounds.append({"round": round_num, "won": winner == "T"})
            elif round_type == "full_buy":
                full_buy_rounds.append({"round": round_num, "won": winner == "T"})

        # Generate patterns
        if pistol_rounds:
            wins = sum(1 for r in pistol_rounds if r["won"])
            patterns.append(
                EconomyPattern(
                    pattern_type="pistol",
                    round_numbers=[r["round"] for r in pistol_rounds],
                    average_equipment=800,
                    win_rate=wins / len(pistol_rounds) if pistol_rounds else 0,
                    description=f"Pistol rounds: {wins}/{len(pistol_rounds)} won",
                )
            )

        if force_rounds:
            wins = sum(1 for r in force_rounds if r["won"])
            patterns.append(
                EconomyPattern(
                    pattern_type="force",
                    round_numbers=[r["round"] for r in force_rounds],
                    average_equipment=2500,
                    win_rate=wins / len(force_rounds) if force_rounds else 0,
                    description=f"Force buy rounds: {wins}/{len(force_rounds)} won",
                )
            )

        if eco_rounds:
            wins = sum(1 for r in eco_rounds if r["won"])
            patterns.append(
                EconomyPattern(
                    pattern_type="eco",
                    round_numbers=[r["round"] for r in eco_rounds],
                    average_equipment=1000,
                    win_rate=wins / len(eco_rounds) if eco_rounds else 0,
                    description=f"Eco rounds: {wins}/{len(eco_rounds)} won",
                )
            )

        return patterns

    def _detect_player_tendencies(
        self, round_timeline: list[dict], players: dict
    ) -> list[PlayerTendency]:
        """Detect player-specific tendencies."""
        tendencies = []

        # Track AWP usage
        awp_players = Counter()
        entry_players = Counter()
        clutch_players = Counter()

        for round_data in round_timeline:
            kills = round_data.get("kills", [])

            for kill in kills:
                weapon = kill.get("weapon", "").lower()
                killer = kill.get("killer", "")  # Schema uses "killer" not "attacker_name"

                if "awp" in weapon:
                    awp_players[killer] += 1

                # First kill of round = entry
                if kill == kills[0]:
                    entry_players[killer] += 1

            # Check for clutch situations
            clutches = round_data.get("clutches", [])
            for clutch in clutches:
                clutch_players[clutch.get("player", "")] += 1

        # Generate tendencies for top AWP player
        if awp_players:
            top_awp = awp_players.most_common(1)[0]
            total_rounds = len(round_timeline)
            tendencies.append(
                PlayerTendency(
                    player_name=top_awp[0],
                    tendency_type="awp_player",
                    description=f"Primary AWP with {top_awp[1]} AWP kills",
                    frequency=top_awp[1] / total_rounds if total_rounds > 0 else 0,
                    round_numbers=[],
                )
            )

        # Generate tendencies for top entry fragger
        if entry_players:
            top_entry = entry_players.most_common(1)[0]
            total_rounds = len(round_timeline)
            tendencies.append(
                PlayerTendency(
                    player_name=top_entry[0],
                    tendency_type="entry_fragger",
                    description=f"Entry fragger with {top_entry[1]} opening kills",
                    frequency=top_entry[1] / total_rounds if total_rounds > 0 else 0,
                    round_numbers=[],
                )
            )

        return tendencies

    def generate_strat_report(
        self,
        match_data: dict,
        team_focus: str | None = None,
        side_focus: str | None = None,
    ) -> str:
        """
        Generate a full tactical report using Claude.

        Analyzes the match data for patterns, then feeds them to Claude
        with the strat-analyst system prompt.

        Args:
            match_data: Parsed match data from CachedAnalyzer
            team_focus: Specific team name to focus on
            side_focus: "T", "CT", or None for both

        Returns:
            Markdown-formatted tactical report
        """
        # First, run pattern detection
        analysis = self.analyze(match_data, team_focus, side_focus)

        # Build summary of detected patterns for Claude
        pattern_summary = self._build_pattern_summary(analysis)

        # Now call Claude for the tactical report
        from opensight.ai.llm_client import get_tactical_ai_client
        from opensight.ai.tactical import SYSTEM_PROMPT_STRAT_ANALYST

        ai = get_tactical_ai_client()

        # Augment match_data with our pattern analysis
        augmented_data = dict(match_data)
        augmented_data["pattern_analysis"] = {
            "defaults_detected": len(analysis.defaults),
            "executes_detected": len(analysis.executes),
            "economy_patterns": len(analysis.economy),
            "player_tendencies": len(analysis.tendencies),
            "summary": pattern_summary,
        }

        report = ai.analyze(
            match_data=augmented_data,
            analysis_type="strat-steal",
            focus=f"Team: {team_focus}, Side: {side_focus or 'both'}",
            system_prompt=SYSTEM_PROMPT_STRAT_ANALYST,
        )

        return report

    def _build_pattern_summary(self, analysis: StratAnalysis) -> str:
        """Build a text summary of detected patterns."""
        lines = []

        lines.append(f"## Detected Patterns for {analysis.team_name} on {analysis.map_name}")
        lines.append("")

        # Defaults
        if analysis.defaults:
            lines.append(f"### Default Setups ({len(analysis.defaults)} detected)")
            setup_types = Counter(d.setup_type for d in analysis.defaults)
            for setup, count in setup_types.most_common(3):
                lines.append(f"- {setup} distribution: {count} times")
            lines.append("")

        # Executes
        if analysis.executes:
            lines.append(f"### Executes ({len(analysis.executes)} detected)")
            a_execs = [e for e in analysis.executes if e.site == "A"]
            b_execs = [e for e in analysis.executes if e.site == "B"]
            if a_execs:
                wins = sum(1 for e in a_execs if e.success)
                lines.append(f"- A site: {len(a_execs)} executes, {wins} successful")
            if b_execs:
                wins = sum(1 for e in b_execs if e.success)
                lines.append(f"- B site: {len(b_execs)} executes, {wins} successful")
            lines.append("")

        # Economy
        if analysis.economy:
            lines.append(f"### Economy Patterns ({len(analysis.economy)} detected)")
            for pattern in analysis.economy:
                lines.append(f"- {pattern.description}")
            lines.append("")

        # Tendencies
        if analysis.tendencies:
            lines.append(f"### Player Tendencies ({len(analysis.tendencies)} detected)")
            for tendency in analysis.tendencies:
                lines.append(f"- {tendency.player_name}: {tendency.description}")

        return "\n".join(lines)


# Singleton instance
_strat_engine_instance: StratEngine | None = None


def get_strat_engine() -> StratEngine:
    """Get or create singleton StratEngine instance."""
    global _strat_engine_instance
    if _strat_engine_instance is None:
        _strat_engine_instance = StratEngine()
    return _strat_engine_instance
