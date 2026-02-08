"""
AI Anti-Strat Report Generator — Structured tactical plans from scouting data.

This is OpenSight's highest-value premium feature. It takes opponent demo data
from the ScoutingEngine and generates a structured tactical report an IGL can
use for match preparation.

Architecture:
  - Input: TeamScoutReport (from ScoutingEngine.generate_report())
  - Processing: Converts scouting data → structured LLM prompt
  - Model: ModelTier.DEEP (Sonnet 4.5) — this is the premium feature
  - Output: AntiStratReport dataclass with structured sections
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Report Dataclasses
# =============================================================================


@dataclass
class PlayerThreat:
    """Scouting-derived threat assessment for a single opponent player."""

    name: str
    steamid: str
    role: str  # "entry", "awper", "support", "lurker", "star", "igl"
    threat_level: str  # "high", "medium", "low"

    # Key stats that ground the assessment
    avg_kills_per_round: float
    entry_success_rate: float  # 0-1
    awp_usage_rate: float  # 0-1
    clutch_win_rate: float  # 0-1
    headshot_rate: float  # 0-1
    play_style: str  # "aggressive", "passive", "mixed"

    # LLM-generated fields (populated after generation)
    counter_strategy: str = ""  # How to neutralize this player
    key_weakness: str = ""  # Exploitable weakness


@dataclass
class MapCounter:
    """Counter-strategy for a specific map tendency."""

    map_name: str
    opponent_tendency: str  # What they do (data-grounded)
    counter_plan: str  # What to do about it (LLM-generated)
    required_utility: list[str] = field(default_factory=list)  # Specific util needed
    key_positions: list[str] = field(default_factory=list)  # Positions to hold/take


@dataclass
class EconomyExploit:
    """Exploitable economy pattern."""

    pattern: str  # What the opponent does
    exploit: str  # How to exploit it
    trigger_rounds: str  # When this applies (e.g., "after pistol loss")


@dataclass
class AntiStratReport:
    """
    Structured anti-strat report for IGL match preparation.

    Every recommendation is grounded in scouting data statistics.
    """

    team_name: str
    maps_analyzed: list[str]
    demos_analyzed: int
    confidence_level: str  # "low", "medium", "high"

    # Structured sections
    player_threats: list[PlayerThreat] = field(default_factory=list)
    map_counters: list[MapCounter] = field(default_factory=list)
    economy_exploits: list[EconomyExploit] = field(default_factory=list)

    # LLM-generated tactical sections
    t_side_game_plan: str = ""  # Full T-side plan
    ct_side_game_plan: str = ""  # Full CT-side plan
    pistol_round_plan: str = ""  # Pistol round strategy
    anti_eco_plan: str = ""  # Anti-eco round plan
    veto_recommendation: str = ""  # Map veto advice

    # Metadata
    model_used: str = ""
    generation_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "team_name": self.team_name,
            "maps_analyzed": self.maps_analyzed,
            "demos_analyzed": self.demos_analyzed,
            "confidence_level": self.confidence_level,
            "player_threats": [
                {
                    "name": pt.name,
                    "steamid": pt.steamid,
                    "role": pt.role,
                    "threat_level": pt.threat_level,
                    "stats": {
                        "avg_kills_per_round": round(pt.avg_kills_per_round, 2),
                        "entry_success_rate": round(pt.entry_success_rate * 100, 1),
                        "awp_usage_rate": round(pt.awp_usage_rate * 100, 1),
                        "clutch_win_rate": round(pt.clutch_win_rate * 100, 1),
                        "headshot_rate": round(pt.headshot_rate * 100, 1),
                        "play_style": pt.play_style,
                    },
                    "counter_strategy": pt.counter_strategy,
                    "key_weakness": pt.key_weakness,
                }
                for pt in self.player_threats
            ],
            "map_counters": [
                {
                    "map_name": mc.map_name,
                    "opponent_tendency": mc.opponent_tendency,
                    "counter_plan": mc.counter_plan,
                    "required_utility": mc.required_utility,
                    "key_positions": mc.key_positions,
                }
                for mc in self.map_counters
            ],
            "economy_exploits": [
                {
                    "pattern": ee.pattern,
                    "exploit": ee.exploit,
                    "trigger_rounds": ee.trigger_rounds,
                }
                for ee in self.economy_exploits
            ],
            "game_plans": {
                "t_side": self.t_side_game_plan,
                "ct_side": self.ct_side_game_plan,
                "pistol_round": self.pistol_round_plan,
                "anti_eco": self.anti_eco_plan,
            },
            "veto_recommendation": self.veto_recommendation,
            "model_used": self.model_used,
            "generation_error": self.generation_error,
        }


# =============================================================================
# Scouting Data → LLM Prompt Conversion
# =============================================================================

ANTISTRAT_SYSTEM_PROMPT = """You are a professional CS2 IGL and tactical analyst preparing an anti-strat report.

Your job: take opponent scouting data and produce ACTIONABLE counter-strategies that an IGL can use directly in match preparation.

## Rules

1. **Every recommendation must be grounded in a specific statistic from the scouting data.**
   Bad: "They like to rush B." Good: "Their entry fragger attempts 35% of opening duels — set up crossfire on B site to punish."

2. **Be specific about utility, positions, and timing.**
   Bad: "Use smokes." Good: "Smoke B main at 1:25 when their entry peaks. Hold off-angle at car with MP9."

3. **Prioritize high-impact counters.**
   Focus on exploiting the opponent's BIGGEST weaknesses, not minor tendencies.

4. **Account for confidence level.**
   With 1 demo, trends may be coincidental. With 4+ demos, patterns are reliable.

5. **Structure your response as valid JSON matching the schema provided.**
   This output goes directly into a structured report, not a chat response.

## CS2 Tactical Knowledge

- Entry fraggers peek first — counter with off-angles, utility, or jiggle bait.
- AWPers hold predictable angles — smoke them out, or double-peek to trade.
- Aggressive CT players can be baited with fakes and utility.
- Economy tendencies predict force-buy rounds — prepare anti-eco setups.
- Rotation speed determines how fast you need to execute.
- Clutch specialists need to be traded out — never give 1vX situations.
"""


def _build_scouting_prompt(report_dict: dict[str, Any]) -> str:
    """
    Convert a TeamScoutReport.to_dict() into a structured LLM prompt.

    This is the critical data-grounding step. Every stat cited in the prompt
    comes directly from the scouting engine output.
    """
    lines: list[str] = []

    team_name = report_dict.get("team_name", "Unknown")
    demos = report_dict.get("demos_analyzed", 0)
    confidence = report_dict.get("confidence_level", "low")
    maps = report_dict.get("maps_analyzed", [])

    lines.append(f"# Opponent Scouting Data: {team_name}")
    lines.append(f"Demos analyzed: {demos} | Confidence: {confidence}")
    lines.append(f"Maps: {', '.join(maps) if maps else 'Unknown'}")
    lines.append("")

    # Player profiles
    players = report_dict.get("players", [])
    if players:
        lines.append("## Player Profiles")
        for p in players:
            name = p.get("name", "Unknown")
            lines.append(f"\n### {name} (steamid: {p.get('steamid', '?')})")
            lines.append(f"- Play Style: {p.get('play_style', '?')}")
            lines.append(f"- Aggression Score: {p.get('aggression_score', 0)}/100")
            lines.append(f"- Consistency: {p.get('consistency_score', 0)}/100")
            lines.append(
                f"- KPR: {p.get('avg_kills_per_round', 0):.2f} | "
                f"DPR: {p.get('avg_deaths_per_round', 0):.2f} | "
                f"K/D: {p.get('kd_ratio', 0):.2f}"
            )
            lines.append(f"- ADR: {p.get('avg_adr', 0):.1f} | KAST: {p.get('avg_kast', 0):.1f}%")
            lines.append(f"- HS Rate: {p.get('headshot_rate', 0):.1f}%")
            lines.append(
                f"- Entry Attempt Rate: {p.get('entry_attempt_rate', 0):.1f}% | "
                f"Entry Success: {p.get('entry_success_rate', 0):.1f}%"
            )
            lines.append(f"- Opening Duel Win Rate: {p.get('opening_duel_win_rate', 0):.1f}%")
            lines.append(
                f"- AWP Usage: {p.get('awp_usage_rate', 0):.1f}% | "
                f"AWP KPR: {p.get('awp_kills_per_awp_round', 0):.2f}"
            )
            clutch_attempts = p.get("clutch_attempts", 0)
            clutch_wins = p.get("clutch_wins", 0)
            clutch_rate = p.get("clutch_win_rate", 0)
            lines.append(f"- Clutch: {clutch_wins}/{clutch_attempts} ({clutch_rate:.1f}%)")

            # Weapon preferences
            weapons = p.get("weapon_preferences", [])
            if weapons:
                wp_str = ", ".join(
                    f"{w['weapon']} ({w['kills']}K, {w['usage_rate']:.0f}%)" for w in weapons[:3]
                )
                lines.append(f"- Top Weapons: {wp_str}")

            lines.append(
                f"- Demos/Rounds analyzed: {p.get('demos_analyzed', 0)}/{p.get('rounds_analyzed', 0)}"
            )

    # Map tendencies
    map_tendencies = report_dict.get("map_tendencies", [])
    if map_tendencies:
        lines.append("\n## Map Tendencies")
        for mt in map_tendencies:
            map_name = mt.get("map_name", "Unknown")
            lines.append(f"\n### {map_name}")
            t_side = mt.get("t_side", {})
            ct_side = mt.get("ct_side", {})
            timing = mt.get("timing", {})

            lines.append(f"- T-Side Default: {t_side.get('default_setup', '?')}")
            lines.append(f"- T-Side Executes: {', '.join(t_side.get('common_executes', []))}")
            lines.append(f"- T-Side Aggression: {t_side.get('aggression', 0):.1f}/100")
            lines.append(f"- CT-Side Default: {ct_side.get('default_setup', '?')}")
            lines.append(f"- CT Rotation Speed: {ct_side.get('rotation_speed', '?')}")
            lines.append(f"- CT Aggression: {ct_side.get('aggression', 0):.1f}/100")
            lines.append(f"- Avg First Contact: {timing.get('avg_first_contact', 0):.1f}s")
            lines.append(f"- Avg Execute Time: {timing.get('avg_execute_time', 0):.1f}s")

    # Economy
    economy = report_dict.get("economy", {})
    if economy:
        lines.append("\n## Economy Patterns")
        lines.append(f"- Tendency: {economy.get('tendency', '?')}")
        lines.append(f"- Force Buy Rate: {economy.get('force_buy_rate', 0):.1f}%")
        lines.append(f"- Eco Round Rate: {economy.get('eco_round_rate', 0):.1f}%")

    return "\n".join(lines)


def _build_antistrat_user_prompt(scouting_text: str, maps: list[str]) -> str:
    """Build the user prompt requesting structured anti-strat output."""
    map_str = ", ".join(maps) if maps else "all analyzed maps"

    return f"""{scouting_text}

---

Based on the scouting data above, generate a structured anti-strat report as JSON.

The JSON must have these exact keys:
{{
  "player_assessments": [
    {{
      "name": "player name",
      "role": "entry|awper|support|lurker|star|igl",
      "threat_level": "high|medium|low",
      "counter_strategy": "specific counter-strategy grounded in their stats",
      "key_weakness": "their most exploitable weakness"
    }}
  ],
  "map_counters": [
    {{
      "map_name": "de_mapname",
      "opponent_tendency": "what they do (cite stats)",
      "counter_plan": "what to do about it",
      "required_utility": ["smoke X", "flash Y"],
      "key_positions": ["position1", "position2"]
    }}
  ],
  "economy_exploits": [
    {{
      "pattern": "what they do economically",
      "exploit": "how to exploit it",
      "trigger_rounds": "when this applies"
    }}
  ],
  "t_side_game_plan": "comprehensive T-side plan for {map_str}",
  "ct_side_game_plan": "comprehensive CT-side plan for {map_str}",
  "pistol_round_plan": "pistol round strategy based on their tendencies",
  "anti_eco_plan": "anti-eco approach based on their force buy tendencies",
  "veto_recommendation": "which maps to pick/ban based on their map tendencies"
}}

IMPORTANT:
- Every recommendation MUST reference specific stats from the scouting data.
- Be specific about utility, positions, and timing.
- Output ONLY valid JSON, no markdown fencing or extra text."""


# =============================================================================
# AntiStratGenerator
# =============================================================================


class AntiStratGenerator:
    """
    Generates structured anti-strat reports from scouting data using LLM.

    Uses ModelTier.DEEP (Sonnet 4.5) because this is the premium feature
    worth paying for — complex tactical reasoning over multi-demo data.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the anti-strat generator.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        import os

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=90,  # Long timeout — complex generation
                )
            except ImportError as e:
                raise ImportError(
                    "Anthropic library not installed. Install with: pip install anthropic"
                ) from e
        return self._client

    def generate(self, scout_report_dict: dict[str, Any]) -> AntiStratReport:
        """
        Generate a structured anti-strat report from scouting data.

        Args:
            scout_report_dict: Output of TeamScoutReport.to_dict()

        Returns:
            AntiStratReport with structured tactical recommendations
        """
        from opensight.ai.llm_client import ModelTier, _build_cached_system, _log_usage

        team_name = scout_report_dict.get("team_name", "Unknown")
        maps = scout_report_dict.get("maps_analyzed", [])
        demos = scout_report_dict.get("demos_analyzed", 0)
        confidence = scout_report_dict.get("confidence_level", "low")

        # Build player threats from raw scouting data (data-grounded, no LLM needed)
        player_threats = self._extract_player_threats(scout_report_dict)

        # Build the report shell
        report = AntiStratReport(
            team_name=team_name,
            maps_analyzed=maps,
            demos_analyzed=demos,
            confidence_level=confidence,
            player_threats=player_threats,
            model_used=ModelTier.DEEP.value,
        )

        # If no API key, return data-only report
        if not self.api_key:
            report.generation_error = (
                "ANTHROPIC_API_KEY not configured. "
                "Report contains data-grounded player threats but no LLM-generated plans."
            )
            return report

        # Generate LLM-powered tactical plans
        try:
            scouting_text = _build_scouting_prompt(scout_report_dict)
            user_prompt = _build_antistrat_user_prompt(scouting_text, maps)

            client = self._get_client()

            logger.info(
                "Generating anti-strat report: team=%s, demos=%d, maps=%s, confidence=%s",
                team_name,
                demos,
                maps,
                confidence,
            )

            message = client.messages.create(
                model=ModelTier.DEEP.value,
                max_tokens=4096,
                system=_build_cached_system(ANTISTRAT_SYSTEM_PROMPT),
                messages=[{"role": "user", "content": user_prompt}],
            )

            _log_usage(ModelTier.DEEP, message.usage)

            # Parse the JSON response
            response_text = message.content[0].text
            self._populate_report_from_llm(report, response_text)

            logger.info(
                "Anti-strat report generated: %d player threats, %d map counters, %d economy exploits",
                len(report.player_threats),
                len(report.map_counters),
                len(report.economy_exploits),
            )

        except Exception as e:
            logger.error(f"Anti-strat LLM generation failed: {e}")
            report.generation_error = f"LLM generation failed: {type(e).__name__}: {e}"

        return report

    def _extract_player_threats(self, scout_report_dict: dict[str, Any]) -> list[PlayerThreat]:
        """
        Extract player threat assessments from scouting data.

        This is pure data processing — no LLM needed. Every field comes
        directly from PlayerScoutProfile.to_dict().
        """
        threats = []
        players = scout_report_dict.get("players", [])

        for p in players:
            name = p.get("name", "Unknown")
            steamid = p.get("steamid", "")

            # Determine role from stats
            role = self._classify_role(p)

            # Determine threat level from impact metrics
            threat_level = self._assess_threat_level(p)

            # Stats come directly from PlayerScoutProfile.to_dict()
            # Note: to_dict() already multiplies rates by 100
            threats.append(
                PlayerThreat(
                    name=name,
                    steamid=steamid,
                    role=role,
                    threat_level=threat_level,
                    avg_kills_per_round=p.get("avg_kills_per_round", 0),
                    entry_success_rate=p.get("entry_success_rate", 0) / 100,
                    awp_usage_rate=p.get("awp_usage_rate", 0) / 100,
                    clutch_win_rate=p.get("clutch_win_rate", 0) / 100,
                    headshot_rate=p.get("headshot_rate", 0) / 100,
                    play_style=p.get("play_style", "mixed"),
                )
            )

        return threats

    def _classify_role(self, player_dict: dict[str, Any]) -> str:
        """Classify player role from scouting stats."""
        # Values from to_dict() are already percentages (multiplied by 100)
        awp_rate = player_dict.get("awp_usage_rate", 0)
        entry_rate = player_dict.get("entry_attempt_rate", 0)
        entry_success = player_dict.get("entry_success_rate", 0)
        kpr = player_dict.get("avg_kills_per_round", 0)

        # AWPer: high AWP usage
        if awp_rate > 20:
            return "awper"

        # Entry: high entry attempt rate
        if entry_rate > 15:
            return "entry"

        # Star: high KPR but not entry/AWP
        if kpr > 0.85:
            return "star"

        # Lurker: passive play style with decent KPR
        if player_dict.get("play_style") == "passive" and kpr > 0.6:
            return "lurker"

        # Support: low entry, moderate KPR (fallback)
        if entry_rate < 8 and kpr < 0.7:
            return "support"

        return "support"

    def _assess_threat_level(self, player_dict: dict[str, Any]) -> str:
        """Assess threat level from scouting stats."""
        kd = player_dict.get("kd_ratio", 1.0)
        kpr = player_dict.get("avg_kills_per_round", 0)
        entry_success = player_dict.get("entry_success_rate", 0)  # already %

        # High threat: strong K/D + high impact
        if kd > 1.2 and kpr > 0.8:
            return "high"
        if entry_success > 55 and kpr > 0.7:
            return "high"

        # Low threat: poor stats
        if kd < 0.9 and kpr < 0.6:
            return "low"

        return "medium"

    def _populate_report_from_llm(self, report: AntiStratReport, response_text: str) -> None:
        """Parse LLM JSON response and populate the report."""
        # Try to extract JSON from the response (handle markdown fencing)
        json_text = response_text.strip()
        if json_text.startswith("```"):
            # Strip markdown code fencing
            lines = json_text.split("\n")
            # Remove first line (```json) and last line (```)
            start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            json_text = "\n".join(lines[start:end])

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse anti-strat JSON response: {e}")
            report.generation_error = f"JSON parse error: {e}"
            # Fall back to raw text in game plan fields
            report.t_side_game_plan = response_text
            return

        # Populate player counter-strategies
        assessments = data.get("player_assessments", [])
        for assessment in assessments:
            # Match by name to existing PlayerThreat objects
            name = assessment.get("name", "")
            for pt in report.player_threats:
                if pt.name.lower() == name.lower():
                    pt.counter_strategy = assessment.get("counter_strategy", "")
                    pt.key_weakness = assessment.get("key_weakness", "")
                    # Update role if LLM has a better classification
                    llm_role = assessment.get("role", "")
                    if llm_role in ("entry", "awper", "support", "lurker", "star", "igl"):
                        pt.role = llm_role
                    break

        # Populate map counters
        for mc_data in data.get("map_counters", []):
            report.map_counters.append(
                MapCounter(
                    map_name=mc_data.get("map_name", ""),
                    opponent_tendency=mc_data.get("opponent_tendency", ""),
                    counter_plan=mc_data.get("counter_plan", ""),
                    required_utility=mc_data.get("required_utility", []),
                    key_positions=mc_data.get("key_positions", []),
                )
            )

        # Populate economy exploits
        for ee_data in data.get("economy_exploits", []):
            report.economy_exploits.append(
                EconomyExploit(
                    pattern=ee_data.get("pattern", ""),
                    exploit=ee_data.get("exploit", ""),
                    trigger_rounds=ee_data.get("trigger_rounds", ""),
                )
            )

        # Populate game plans
        report.t_side_game_plan = data.get("t_side_game_plan", "")
        report.ct_side_game_plan = data.get("ct_side_game_plan", "")
        report.pistol_round_plan = data.get("pistol_round_plan", "")
        report.anti_eco_plan = data.get("anti_eco_plan", "")
        report.veto_recommendation = data.get("veto_recommendation", "")


# =============================================================================
# Module-level convenience
# =============================================================================

_generator_instance: AntiStratGenerator | None = None


def get_antistrat_generator() -> AntiStratGenerator:
    """Get or create singleton AntiStratGenerator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = AntiStratGenerator()
    return _generator_instance


def generate_antistrat_report(scout_report_dict: dict[str, Any]) -> AntiStratReport:
    """
    Convenience function to generate an anti-strat report.

    Args:
        scout_report_dict: Output of TeamScoutReport.to_dict()

    Returns:
        AntiStratReport with structured tactical recommendations
    """
    generator = get_antistrat_generator()
    return generator.generate(scout_report_dict)
