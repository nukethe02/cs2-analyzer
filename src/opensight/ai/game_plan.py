"""
AI Game Plan Generator.

Combines your team's strengths with opponent weaknesses to produce
a structured stratbook for match preparation.

Design principle: HUMAN IN THE LOOP.
Every recommendation includes confidence level + statistical evidence.
The IGL decides — the AI suggests.

Architecture:
  - Input: Your team's recent demos (orchestrator results) + opponent scouting data
  - Processing: Heuristic matchup analysis → LLM tactical generation
  - Model: ModelTier.DEEP (Sonnet 4.5) — premium capstone feature
  - Output: GamePlan dataclass with structured sections
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class StratCall:
    """A specific tactical call with evidence."""

    name: str  # "A Split through Squeaky + Main"
    description: str  # step-by-step execution
    utility_sequence: list[str] = field(default_factory=list)
    player_assignments: dict[str, str] = field(default_factory=dict)
    when_to_call: str = ""  # "Default after winning pistol"
    confidence: float = 0.5  # 0.0-1.0
    evidence: str = ""  # statistical evidence
    expected_success_rate: str = ""  # "~60% based on ..."

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "utility_sequence": self.utility_sequence,
            "player_assignments": self.player_assignments,
            "when_to_call": self.when_to_call,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "expected_success_rate": self.expected_success_rate,
        }


@dataclass
class EconomyPlan:
    """Round-type-specific economy guidance."""

    pistol_round_buy: str = "kevlar + utility"
    anti_eco_setup: str = "SMGs + hold angles, play for exit kills"
    force_buy_threshold: int = 3200
    save_triggers: list[str] = field(default_factory=list)
    double_eco_into_full: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pistol_round_buy": self.pistol_round_buy,
            "anti_eco_setup": self.anti_eco_setup,
            "force_buy_threshold": self.force_buy_threshold,
            "save_triggers": self.save_triggers,
            "double_eco_into_full": self.double_eco_into_full,
        }


@dataclass
class GamePlan:
    """Complete stratbook for an upcoming match."""

    plan_id: str = ""
    opponent: str = ""
    map_name: str = ""
    generated_at: str = ""
    confidence_overall: float = 0.5

    # Executive brief
    executive_summary: str = ""
    key_advantage: str = ""
    key_risk: str = ""

    # CT-side plan
    ct_default: StratCall | None = None
    ct_adjustments: list[StratCall] = field(default_factory=list)
    ct_retake_priorities: dict[str, str] = field(default_factory=dict)

    # T-side plan
    t_default: StratCall | None = None
    t_executes: list[StratCall] = field(default_factory=list)
    t_read_based: list[StratCall] = field(default_factory=list)

    # Economy
    economy_plan: EconomyPlan = field(default_factory=EconomyPlan)

    # Player assignments
    player_roles: dict[str, str] = field(default_factory=dict)
    player_matchups: list[str] = field(default_factory=list)

    # Situational
    if_losing: list[str] = field(default_factory=list)
    if_winning: list[str] = field(default_factory=list)
    timeout_triggers: list[str] = field(default_factory=list)

    # Anti-strat
    opponent_exploits: list[str] = field(default_factory=list)

    # Metadata
    model_used: str = ""
    generation_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "plan_id": self.plan_id,
            "opponent": self.opponent,
            "map_name": self.map_name,
            "generated_at": self.generated_at,
            "confidence_overall": round(self.confidence_overall, 2),
            "executive_summary": self.executive_summary,
            "key_advantage": self.key_advantage,
            "key_risk": self.key_risk,
            "ct_side": {
                "default": self.ct_default.to_dict() if self.ct_default else None,
                "adjustments": [s.to_dict() for s in self.ct_adjustments],
                "retake_priorities": self.ct_retake_priorities,
            },
            "t_side": {
                "default": self.t_default.to_dict() if self.t_default else None,
                "executes": [s.to_dict() for s in self.t_executes],
                "read_based": [s.to_dict() for s in self.t_read_based],
            },
            "economy_plan": self.economy_plan.to_dict(),
            "player_roles": self.player_roles,
            "player_matchups": self.player_matchups,
            "situational": {
                "if_losing": self.if_losing,
                "if_winning": self.if_winning,
                "timeout_triggers": self.timeout_triggers,
            },
            "opponent_exploits": self.opponent_exploits,
            "model_used": self.model_used,
            "generation_error": self.generation_error,
        }


# =============================================================================
# Matchup Analysis (heuristic, no LLM)
# =============================================================================


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return numerator / denominator if denominator > 0 else default


def _compute_team_averages(orchestrator_results: list[dict]) -> dict[str, float]:
    """
    Compute average team stats from multiple orchestrator results.

    Returns dict with averaged stats across all players and matches.
    """
    totals: dict[str, float] = {
        "kills": 0,
        "deaths": 0,
        "adr": 0,
        "kast": 0,
        "rating": 0,
        "hs_pct": 0,
        "opening_duel_wins": 0,
        "opening_duel_total": 0,
        "clutch_wins": 0,
        "clutch_total": 0,
        "trade_kills": 0,
        "trade_opps": 0,
        "flash_assists": 0,
        "he_damage": 0,
        "utility_thrown": 0,
        "rounds_played": 0,
    }
    player_count = 0

    for result in orchestrator_results:
        players = result.get("players") or {}
        for _sid, pdata in players.items():
            stats = pdata.get("stats") or {}
            rating = pdata.get("rating") or {}
            duels = pdata.get("duels") or {}
            utility = pdata.get("utility") or {}

            totals["kills"] += stats.get("kills", 0)
            totals["deaths"] += stats.get("deaths", 0)
            totals["adr"] += stats.get("adr", 0.0)
            totals["kast"] += rating.get("kast_percentage", 0.0)
            totals["rating"] += rating.get("hltv_rating", 0.0)
            totals["hs_pct"] += stats.get("headshot_pct", 0.0)
            totals["opening_duel_wins"] += duels.get("opening_kills", 0)
            totals["opening_duel_total"] += (
                duels.get("opening_kills", 0) + duels.get("opening_deaths", 0)
            )
            totals["clutch_wins"] += duels.get("clutch_wins", 0)
            totals["clutch_total"] += duels.get("clutch_attempts", 0)
            totals["trade_kills"] += duels.get("trade_kills", 0)
            totals["trade_opps"] += duels.get("trade_kill_opportunities", 0)
            totals["flash_assists"] += utility.get("flash_assists", 0)
            totals["he_damage"] += utility.get("he_damage", 0)

            flashes = utility.get("flashbangs_thrown", 0)
            smokes = utility.get("smokes_thrown", 0)
            he = utility.get("he_thrown", 0)
            molotovs = utility.get("molotovs_thrown", 0)
            totals["utility_thrown"] += flashes + smokes + he + molotovs

            totals["rounds_played"] += stats.get("rounds_played", 0)
            player_count += 1

    if player_count == 0:
        return totals

    # Average per player
    return {
        "avg_kills": totals["kills"] / player_count,
        "avg_deaths": totals["deaths"] / player_count,
        "avg_adr": totals["adr"] / player_count,
        "avg_kast": totals["kast"] / player_count,
        "avg_rating": totals["rating"] / player_count,
        "avg_hs_pct": totals["hs_pct"] / player_count,
        "opening_duel_win_rate": _safe_div(
            totals["opening_duel_wins"], totals["opening_duel_total"]
        ),
        "clutch_win_rate": _safe_div(totals["clutch_wins"], totals["clutch_total"]),
        "trade_success_rate": _safe_div(totals["trade_kills"], totals["trade_opps"]),
        "avg_flash_assists": totals["flash_assists"] / player_count,
        "avg_utility_damage": totals["he_damage"] / player_count,
        "avg_utility_thrown": totals["utility_thrown"] / player_count,
        "total_rounds": totals["rounds_played"] / max(player_count, 1),
    }


def _extract_opponent_stats(opponent_scouting: dict) -> dict[str, float]:
    """
    Extract averaged stats from opponent scouting data (TeamScoutReport.to_dict()).
    """
    players = opponent_scouting.get("players") or []
    if not players:
        return {}

    totals: dict[str, float] = {
        "kpr": 0,
        "adr": 0,
        "kast": 0,
        "hs_rate": 0,
        "entry_success": 0,
        "entry_attempts": 0,
        "clutch_wins": 0,
        "clutch_attempts": 0,
        "force_buy_rate": 0,
    }

    for p in players:
        totals["kpr"] += p.get("avg_kills_per_round", 0)
        totals["adr"] += p.get("avg_adr", 0)
        totals["kast"] += p.get("avg_kast", 0)
        totals["hs_rate"] += p.get("headshot_rate", 0)
        totals["entry_success"] += p.get("entry_success_rate", 0)
        totals["entry_attempts"] += p.get("entry_attempt_rate", 0)
        totals["clutch_wins"] += p.get("clutch_wins", 0)
        totals["clutch_attempts"] += p.get("clutch_attempts", 0)

    n = len(players)
    economy = opponent_scouting.get("economy") or {}

    return {
        "avg_kpr": totals["kpr"] / n,
        "avg_adr": totals["adr"] / n,
        "avg_kast": totals["kast"] / n,
        "avg_hs_rate": totals["hs_rate"] / n,
        "avg_entry_success": totals["entry_success"] / n,
        "avg_entry_attempts": totals["entry_attempts"] / n,
        "total_clutch_wins": totals["clutch_wins"],
        "total_clutch_attempts": totals["clutch_attempts"],
        "force_buy_rate": economy.get("force_buy_rate", 0),
        "eco_round_rate": economy.get("eco_round_rate", 0),
    }


def build_matchup_analysis(
    your_data: list[dict],
    opponent_data: dict,
) -> str:
    """
    Compare team profiles to find exploitable matchups.

    Returns structured text for LLM context — no LLM call here.
    """
    your_stats = _compute_team_averages(your_data)
    opp_stats = _extract_opponent_stats(opponent_data)

    if not your_stats.get("avg_rating") or not opp_stats:
        return "<matchup_analysis>\nInsufficient data for matchup analysis.\n</matchup_analysis>"

    lines: list[str] = ["<matchup_analysis>"]

    # Opening duel comparison
    your_od = your_stats.get("opening_duel_win_rate", 0)
    opp_entry = opp_stats.get("avg_entry_success", 0) / 100  # scouting is %
    od_advantage = your_od - opp_entry
    if od_advantage > 0.05:
        lines.append(
            f"ADVANTAGE — Opening duels: Your win rate {your_od:.0%} "
            f"vs their entry success {opp_entry:.0%} (+{od_advantage:.0%})"
        )
    elif od_advantage < -0.05:
        lines.append(
            f"DISADVANTAGE — Opening duels: Your win rate {your_od:.0%} "
            f"vs their entry success {opp_entry:.0%} ({od_advantage:.0%})"
        )
    else:
        lines.append(
            f"EVEN — Opening duels: Your {your_od:.0%} vs their {opp_entry:.0%}"
        )

    # ADR / firepower comparison
    your_adr = your_stats.get("avg_adr", 0)
    opp_adr = opp_stats.get("avg_adr", 0)
    adr_diff = your_adr - opp_adr
    if abs(adr_diff) > 5:
        tag = "ADVANTAGE" if adr_diff > 0 else "DISADVANTAGE"
        lines.append(
            f"{tag} — Firepower: Your ADR {your_adr:.1f} vs their {opp_adr:.1f} "
            f"(diff: {adr_diff:+.1f})"
        )

    # Trade discipline
    your_trade = your_stats.get("trade_success_rate", 0)
    lines.append(f"Trade discipline: Your trade success rate {your_trade:.0%}")

    # Clutch comparison
    your_clutch = your_stats.get("clutch_win_rate", 0)
    opp_clutch_w = opp_stats.get("total_clutch_wins", 0)
    opp_clutch_a = opp_stats.get("total_clutch_attempts", 0)
    opp_clutch_rate = _safe_div(opp_clutch_w, opp_clutch_a)
    lines.append(
        f"Clutch: Your win rate {your_clutch:.0%} vs their "
        f"{opp_clutch_w}/{opp_clutch_a} ({opp_clutch_rate:.0%})"
    )

    # Economy discipline
    opp_force = opp_stats.get("force_buy_rate", 0)
    if opp_force > 30:
        lines.append(
            f"EXPLOIT — Economy: Opponent force-buys {opp_force:.0f}% of rounds "
            f"(aggressive, punishable)"
        )
    elif opp_force < 15:
        lines.append(
            f"NOTE — Economy: Opponent is disciplined (force rate {opp_force:.0f}%) "
            f"— harder to break economically"
        )

    # Utility effectiveness
    your_flash = your_stats.get("avg_flash_assists", 0)
    your_util_dmg = your_stats.get("avg_utility_damage", 0)
    lines.append(
        f"Utility: Your avg flash assists {your_flash:.1f}/player, "
        f"avg utility damage {your_util_dmg:.0f}/player"
    )

    # KAST comparison
    your_kast = your_stats.get("avg_kast", 0)
    opp_kast = opp_stats.get("avg_kast", 0)
    kast_diff = your_kast - opp_kast
    if abs(kast_diff) > 3:
        tag = "ADVANTAGE" if kast_diff > 0 else "DISADVANTAGE"
        lines.append(
            f"{tag} — Consistency: Your KAST {your_kast:.0f}% vs their {opp_kast:.0f}% "
            f"(diff: {kast_diff:+.0f}%)"
        )

    # Player-specific threats from opponent
    players = opponent_data.get("players") or []
    high_threats = [
        p for p in players if p.get("avg_kills_per_round", 0) > 0.8
    ]
    if high_threats:
        lines.append("")
        lines.append("HIGH-THREAT PLAYERS:")
        for p in high_threats:
            name = p.get("name", "Unknown")
            kpr = p.get("avg_kills_per_round", 0)
            style = p.get("play_style", "unknown")
            lines.append(f"  - {name}: {kpr:.2f} KPR, {style} style")

    lines.append("</matchup_analysis>")
    return "\n".join(lines)


def _build_economy_plan(
    your_data: list[dict],
    opponent_data: dict,
) -> EconomyPlan:
    """
    Generate economy plan from heuristic analysis (no LLM needed).
    """
    opp_economy = opponent_data.get("economy") or {}
    opp_force_rate = opp_economy.get("force_buy_rate", 20)

    save_triggers = [
        "Team money < $10,000 combined AND loss bonus building",
        "After losing pistol round — save for round 4 full buy",
        "Score is close (within 2 rounds) — preserve economy for crucial rounds",
    ]

    # Adjust force threshold based on opponent tendencies
    force_threshold = 3200
    if opp_force_rate > 30:
        # Opponent force-buys a lot — we can be more conservative
        save_triggers.append(
            f"Opponent force-buys {opp_force_rate:.0f}% — hold saves, they'll give you free rounds"
        )
    elif opp_force_rate < 15:
        # Opponent saves discipline — we need to be more aggressive on anti-ecos
        force_threshold = 2800

    # Anti-eco advice based on opponent eco round rate
    opp_eco_rate = opp_economy.get("eco_round_rate", 20)
    if opp_eco_rate > 25:
        anti_eco = (
            f"Opponent ecos {opp_eco_rate:.0f}% of rounds — expect frequent eco rushes. "
            "SMGs + utility, hold disciplined angles, deny exit kills."
        )
    else:
        anti_eco = "Standard anti-eco: SMGs for $600 kill reward, hold angles, use utility to slow pushes."

    return EconomyPlan(
        pistol_round_buy="kevlar + utility (smoke + flash preferred)",
        anti_eco_setup=anti_eco,
        force_buy_threshold=force_threshold,
        save_triggers=save_triggers,
        double_eco_into_full=(
            "After pistol loss: eco rounds 2-3, guarantee rifles + full utility round 4. "
            "Loss bonus builds to $2400+ by round 4 — combined with eco savings, "
            "full buy is guaranteed. Never break this with a solo force."
        ),
    )


# =============================================================================
# LLM Prompt Construction
# =============================================================================

GAME_PLAN_SYSTEM_PROMPT = """You are a professional CS2 IGL and tactical analyst generating a structured game plan.

Your output will be parsed as JSON and fed directly into a match preparation tool.
Every recommendation MUST include:
1. Statistical evidence from the data provided
2. A confidence level (high/medium/low mapped to 0.8+/0.5-0.8/below 0.5)
3. A fallback plan if the primary strategy fails

## Rules
- Be SPECIFIC about positions, utility, and timing. Not "play aggressive" but
  "push B main with flash from kix, foe holds connector for rotation."
- Use actual player names from the roster in assignments.
- Every strategy must cite evidence from the matchup analysis or scouting data.
- CT default and T default are mandatory — adjustments are optional.
- Limit to 3-5 T-side executes — quality over quantity.
- Economy plan must follow CS2 economy rules (loss bonus ladder, etc.).
"""


def _build_game_plan_prompt(
    matchup_analysis: str,
    opponent_scouting_text: str,
    your_team_text: str,
    map_name: str,
    roster: dict[str, str],
) -> str:
    """
    Build the LLM prompt for game plan generation.
    """
    roster_str = "\n".join(f"  - {name}: {role}" for name, role in roster.items())

    return f"""<your_team>
{your_team_text}

Roster:
{roster_str}
</your_team>

<opponent>
{opponent_scouting_text}
</opponent>

{matchup_analysis}

<task>
Generate a complete game plan for {map_name}.

Output ONLY valid JSON with these exact keys:
{{
  "executive_summary": "3-4 sentence brief readable in 30 seconds",
  "key_advantage": "your team's biggest advantage over this opponent",
  "key_risk": "biggest risk to manage in this match",
  "ct_default": {{
    "name": "CT default setup name",
    "description": "step-by-step default positions and roles",
    "utility_sequence": ["utility1", "utility2"],
    "player_assignments": {{"player_name": "assignment"}},
    "when_to_call": "when to use this",
    "confidence": 0.8,
    "evidence": "statistical evidence",
    "expected_success_rate": "estimated rate"
  }},
  "ct_adjustments": [
    {{same format as ct_default, "when_to_call": "trigger condition"}}
  ],
  "ct_retake_priorities": {{"A": "retake plan for A", "B": "retake plan for B"}},
  "t_default": {{same format as ct_default}},
  "t_executes": [{{same format, 3-5 specific execute calls}}],
  "t_read_based": [{{same format, mid-round reads}}],
  "player_roles": {{"player_name": "role assignment"}},
  "player_matchups": ["specific player-vs-player matchup advice"],
  "if_losing": ["adjustment when down 0-3", "adjustment when losing on CT"],
  "if_winning": ["how to maintain lead"],
  "timeout_triggers": ["when to call timeout"],
  "opponent_exploits": ["specific weakness to exploit with evidence"]
}}

IMPORTANT:
- Use the actual roster names: {', '.join(roster.keys())}
- Every strategy must cite evidence from the data above
- Be specific about positions, utility, and timing for {map_name}
- Output ONLY valid JSON, no markdown fencing
</task>"""


def _build_your_team_summary(your_data: list[dict]) -> str:
    """Build a text summary of your team's stats for LLM context."""
    lines: list[str] = []
    avgs = _compute_team_averages(your_data)

    lines.append("Team Performance Summary:")
    lines.append(f"  Average Rating: {avgs.get('avg_rating', 0):.2f}")
    lines.append(f"  Average ADR: {avgs.get('avg_adr', 0):.1f}")
    lines.append(f"  Average KAST: {avgs.get('avg_kast', 0):.0f}%")
    lines.append(f"  Opening Duel Win Rate: {avgs.get('opening_duel_win_rate', 0):.0%}")
    lines.append(f"  Trade Success Rate: {avgs.get('trade_success_rate', 0):.0%}")
    lines.append(f"  Clutch Win Rate: {avgs.get('clutch_win_rate', 0):.0%}")
    lines.append(f"  Avg Flash Assists: {avgs.get('avg_flash_assists', 0):.1f}/player")

    # Per-player breakdown from most recent demo
    if your_data:
        latest = your_data[-1]
        players = latest.get("players") or {}
        if players:
            lines.append("\nPlayer Breakdown (most recent match):")
            for sid, pdata in players.items():
                name = pdata.get("name", sid[:8])
                stats = pdata.get("stats") or {}
                rating_data = pdata.get("rating") or {}
                lines.append(
                    f"  {name}: {stats.get('kills', 0)}K/{stats.get('deaths', 0)}D, "
                    f"Rating {rating_data.get('hltv_rating', 0):.2f}, "
                    f"ADR {stats.get('adr', 0):.1f}"
                )

    return "\n".join(lines)


# =============================================================================
# GamePlanGenerator
# =============================================================================


class GamePlanGenerator:
    """
    Generates complete game plans from team data + opponent scouting.

    Uses ModelTier.DEEP (Sonnet 4.5) — this is the capstone premium feature.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the game plan generator.

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
                    timeout=120,  # Long timeout for complex generation
                )
            except ImportError as e:
                raise ImportError(
                    "Anthropic library not installed. Install with: pip install anthropic"
                ) from e
        return self._client

    def generate(
        self,
        your_team_demos: list[dict],
        opponent_scouting: dict,
        map_name: str,
        roster: dict[str, str],
    ) -> GamePlan:
        """
        Generate complete game plan.

        Uses ModelTier.DEEP — this is the premium feature.
        Combines structured data analysis with LLM narrative generation.

        Args:
            your_team_demos: Orchestrator results from your recent matches
            opponent_scouting: Scouting engine data on opponent (TeamScoutReport.to_dict())
            map_name: Map name (e.g., "de_ancient")
            roster: Player name → role mapping (e.g., {"foe": "igl", "kix": "entry"})

        Returns:
            GamePlan with all sections populated
        """
        from opensight.ai.antistrat_report import _build_scouting_prompt
        from opensight.ai.llm_client import ModelTier, _build_cached_system, _log_usage

        plan_id = str(uuid.uuid4())
        opponent_name = opponent_scouting.get("team_name", "Unknown")
        now = datetime.now(UTC).isoformat()

        # Compute data quality confidence
        demos_analyzed = opponent_scouting.get("demos_analyzed", 0)
        confidence = self._compute_confidence(
            len(your_team_demos), demos_analyzed
        )

        plan = GamePlan(
            plan_id=plan_id,
            opponent=opponent_name,
            map_name=map_name,
            generated_at=now,
            confidence_overall=confidence,
            model_used=ModelTier.DEEP.value,
        )

        # 1. Build economy plan (heuristic, no LLM)
        plan.economy_plan = _build_economy_plan(your_team_demos, opponent_scouting)

        # 2. Build matchup analysis (heuristic, no LLM)
        matchup_text = build_matchup_analysis(your_team_demos, opponent_scouting)

        # 3. Build team summary
        your_team_text = _build_your_team_summary(your_team_demos)

        # 4. Build opponent scouting text
        opponent_text = _build_scouting_prompt(opponent_scouting)

        # 5. Set roster
        plan.player_roles = dict(roster)

        # If no API key, return data-only plan
        if not self.api_key:
            plan.generation_error = (
                "ANTHROPIC_API_KEY not configured. "
                "Plan contains economy guidance and matchup analysis "
                "but no LLM-generated tactical sections."
            )
            plan.executive_summary = (
                f"Game plan for {map_name} vs {opponent_name}. "
                f"Data confidence: {confidence:.0%}. "
                "LLM generation skipped — see economy plan and matchup analysis."
            )
            return plan

        # 6. Build LLM prompt
        user_prompt = _build_game_plan_prompt(
            matchup_analysis=matchup_text,
            opponent_scouting_text=opponent_text,
            your_team_text=your_team_text,
            map_name=map_name,
            roster=roster,
        )

        # 7. Call LLM
        try:
            client = self._get_client()

            logger.info(
                "Generating game plan: opponent=%s, map=%s, roster=%d players, confidence=%.0f%%",
                opponent_name,
                map_name,
                len(roster),
                confidence * 100,
            )

            message = client.messages.create(
                model=ModelTier.DEEP.value,
                max_tokens=4096,
                system=_build_cached_system(GAME_PLAN_SYSTEM_PROMPT),
                messages=[{"role": "user", "content": user_prompt}],
            )

            _log_usage(ModelTier.DEEP, message.usage)

            response_text = message.content[0].text
            self._populate_plan_from_llm(plan, response_text, roster)

            logger.info(
                "Game plan generated: %d T executes, %d CT adjustments, %d exploits",
                len(plan.t_executes),
                len(plan.ct_adjustments),
                len(plan.opponent_exploits),
            )

        except Exception as e:
            logger.error("Game plan LLM generation failed: %s", e)
            plan.generation_error = f"LLM generation failed: {type(e).__name__}: {e}"

        return plan

    def _compute_confidence(self, your_demos: int, opponent_demos: int) -> float:
        """
        Compute overall confidence based on data availability.

        More demos = higher confidence.
        """
        # Each side contributes up to 0.5
        your_conf = min(your_demos / 5, 1.0) * 0.5
        opp_conf = min(opponent_demos / 4, 1.0) * 0.5
        return your_conf + opp_conf

    def _populate_plan_from_llm(
        self, plan: GamePlan, response_text: str, roster: dict[str, str]
    ) -> None:
        """Parse LLM JSON response and populate the GamePlan."""
        # Strip markdown fencing if present
        json_text = response_text.strip()
        if json_text.startswith("```"):
            lines = json_text.split("\n")
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
            logger.warning("Failed to parse game plan JSON: %s", e)
            plan.generation_error = f"JSON parse error: {e}"
            plan.executive_summary = response_text[:500]
            return

        # Executive brief
        plan.executive_summary = data.get("executive_summary", "")
        plan.key_advantage = data.get("key_advantage", "")
        plan.key_risk = data.get("key_risk", "")

        # CT-side
        ct_default_data = data.get("ct_default")
        if ct_default_data:
            plan.ct_default = self._parse_strat_call(ct_default_data)

        for adj in data.get("ct_adjustments", []):
            plan.ct_adjustments.append(self._parse_strat_call(adj))

        plan.ct_retake_priorities = data.get("ct_retake_priorities", {})

        # T-side
        t_default_data = data.get("t_default")
        if t_default_data:
            plan.t_default = self._parse_strat_call(t_default_data)

        for exe in data.get("t_executes", []):
            plan.t_executes.append(self._parse_strat_call(exe))

        for read in data.get("t_read_based", []):
            plan.t_read_based.append(self._parse_strat_call(read))

        # Player assignments
        plan.player_roles = data.get("player_roles", dict(roster))
        plan.player_matchups = data.get("player_matchups", [])

        # Situational
        plan.if_losing = data.get("if_losing", [])
        plan.if_winning = data.get("if_winning", [])
        plan.timeout_triggers = data.get("timeout_triggers", [])

        # Anti-strat
        plan.opponent_exploits = data.get("opponent_exploits", [])

    def _parse_strat_call(self, data: dict) -> StratCall:
        """Parse a StratCall from LLM JSON output."""
        return StratCall(
            name=data.get("name", ""),
            description=data.get("description", ""),
            utility_sequence=data.get("utility_sequence", []),
            player_assignments=data.get("player_assignments", {}),
            when_to_call=data.get("when_to_call", ""),
            confidence=float(data.get("confidence", 0.5)),
            evidence=data.get("evidence", ""),
            expected_success_rate=data.get("expected_success_rate", ""),
        )


# =============================================================================
# In-memory plan cache
# =============================================================================

_plan_cache: dict[str, GamePlan] = {}


def cache_plan(plan: GamePlan) -> None:
    """Store a generated plan in the cache."""
    _plan_cache[plan.plan_id] = plan
    # Keep cache bounded
    if len(_plan_cache) > 50:
        oldest_key = next(iter(_plan_cache))
        del _plan_cache[oldest_key]


def get_cached_plan(plan_id: str) -> GamePlan | None:
    """Retrieve a cached plan by ID."""
    return _plan_cache.get(plan_id)


# =============================================================================
# Module-level convenience
# =============================================================================

_generator_instance: GamePlanGenerator | None = None


def get_game_plan_generator() -> GamePlanGenerator:
    """Get or create singleton GamePlanGenerator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = GamePlanGenerator()
    return _generator_instance
