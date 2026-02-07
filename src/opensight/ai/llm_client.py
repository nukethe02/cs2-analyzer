"""
LLM Client for AI Coaching Summaries

Integrates Anthropic API (Claude 3.5 Sonnet) to generate
personalized match analysis and coaching insights.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with Anthropic Claude API.

    Supports:
    - Anthropic Claude 3.5 Sonnet (default)
    - Other Claude models via model parameter
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        timeout: int = 30,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-20250514)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self.timeout = timeout

        # Lazy import to avoid requiring anthropic if not used
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except ImportError as e:
                raise ImportError(
                    "Anthropic library not installed. Install with: pip install anthropic"
                ) from e

        return self._client

    def generate_match_summary(
        self, player_stats: dict[str, Any], match_context: dict[str, Any] | None = None
    ) -> str:
        """
        Generate AI-powered match summary and coaching insights.

        Args:
            player_stats: Player statistics dictionary with:
                - kills, deaths, assists
                - hltv_rating
                - adr (average damage per round)
                - headshot_pct
                - ttd_median_ms (time to damage)
                - cp_median_error_deg (crosshair placement)
                - kast_percentage
                - entry_kills, entry_deaths
                - trade_kill_success, trade_kill_opportunities
                - clutch_wins, clutch_attempts
            match_context: Optional context (map, opponent, team performance)

        Returns:
            Markdown-formatted coaching summary

        Raises:
            ValueError: If API key not configured
            Exception: If LLM call fails
        """
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not configured. Set environment variable or pass api_key to constructor."
            )

        # Extract key stats with safe fallbacks
        kills = player_stats.get("kills", 0)
        deaths = player_stats.get("deaths", 0)
        assists = player_stats.get("assists", 0)
        rating = player_stats.get("hltv_rating", 0.0)
        adr = player_stats.get("adr", 0.0)
        hs_pct = player_stats.get("headshot_pct", 0.0)
        kast = player_stats.get("kast_percentage", 0.0)

        # Advanced stats
        ttd = player_stats.get("ttd_median_ms", 0)
        cp = player_stats.get("cp_median_error_deg", 0.0)
        entry_kills = player_stats.get("entry_kills", 0)
        entry_deaths = player_stats.get("entry_deaths", 0)
        trade_success = player_stats.get("trade_kill_success", 0)
        trade_opps = player_stats.get("trade_kill_opportunities", 0)
        clutch_wins = player_stats.get("clutch_wins", 0)
        clutch_attempts = player_stats.get("clutch_attempts", 0)

        # Validate stats are not all zero (would indicate empty/uninitialized data)
        if kills == 0 and deaths == 0 and rating == 0.0:
            logger.warning("Player stats appear to be uninitialized (all zeros)")
            return (
                "**Error**: Unable to generate summary. Player statistics are not available. "
                "Ensure the demo has been fully analyzed before requesting AI insights."
            )

        # Build context string
        context_str = ""
        if match_context:
            map_name = match_context.get("map_name", "")
            rounds = match_context.get("total_rounds", 0)
            result = match_context.get("result", "")
            if map_name:
                context_str = f"Map: {map_name}"
            if rounds:
                context_str += f", Rounds: {rounds}"
            if result:
                context_str += f", Result: {result}"

        # Construct system prompt (Tier 1 CS2 Coach persona)
        system_prompt = """You are a Tier 1 CS2 Coach with deep expertise in professional Counter-Strike.
Analyze player statistics with brutal honesty but constructive feedback.
Focus on actionable improvements backed by specific metrics.
Never hallucinate stats - only reference the exact numbers provided."""

        # Construct user prompt with stats
        user_prompt = f"""Analyze this CS2 match performance:

**Core Stats:**
- Kills: {kills}
- Deaths: {deaths}
- Assists: {assists}
- K/D Ratio: {kills / max(deaths, 1):.2f}
- HLTV 2.0 Rating: {rating:.2f}
- ADR: {adr:.1f}
- Headshot %: {hs_pct:.0f}%
- KAST%: {kast:.0f}%

**Advanced Metrics:**
- Time to Damage (TTD): {ttd:.0f}ms
- Crosshair Placement: {cp:.1f} error
- Entry Kills: {entry_kills} | Entry Deaths: {entry_deaths}
- Trade Kill Success: {trade_success} / {trade_opps} opportunities
- Clutches Won: {clutch_wins} / {clutch_attempts} attempts

{context_str if context_str else ""}

Provide a concise analysis with:
1. **3 Strengths**: What they did well (be specific with numbers)
2. **1 Critical Weakness**: The #1 area to improve immediately
3. **Actionable Advice**: One concrete drill or practice focus

Format in markdown. Be harsh but fair. Keep it under 200 words."""

        try:
            client = self._get_client()

            logger.info(
                f"Generating LLM summary for player with {kills}K/{deaths}D, Rating={rating:.2f}"
            )

            message = client.messages.create(
                model=self.model,
                max_tokens=400,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )

            summary = message.content[0].text
            logger.info(f"LLM summary generated successfully ({len(summary)} chars)")
            return summary

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return fallback summary on error
            return f"""**AI Coaching Unavailable**

Unable to generate personalized insights (Error: {type(e).__name__}).

**Quick Stats:**
- {kills}K / {deaths}D / {assists}A
- HLTV Rating: {rating:.2f}
- ADR: {adr:.1f}

Please check your ANTHROPIC_API_KEY configuration or try again later."""


# Singleton instance for reuse
_llm_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client instance."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance


def generate_match_summary(
    player_stats: dict[str, Any], match_context: dict[str, Any] | None = None
) -> str:
    """
    Convenience function to generate match summary.

    Args:
        player_stats: Player statistics dictionary
        match_context: Optional match context (map, result, etc.)

    Returns:
        Markdown-formatted coaching summary
    """
    client = get_llm_client()
    return client.generate_match_summary(player_stats, match_context)


# =============================================================================
# TacticalAIClient - Claude-powered tactical analysis with tool-use
# =============================================================================


class TacticalAIClient:
    """
    Claude-powered tactical analysis for CS2 demos.

    Uses Claude's tool-use (function calling) to query match data
    and generate comprehensive tactical reports.
    """

    # Tools Claude can call to query match data
    ANALYSIS_TOOLS = [
        {
            "name": "get_round_data",
            "description": "Get detailed data for a specific round including kills, economy, utility usage",
            "input_schema": {
                "type": "object",
                "properties": {
                    "round_number": {
                        "type": "integer",
                        "description": "Round number (1-30+)",
                    },
                },
                "required": ["round_number"],
            },
        },
        {
            "name": "get_player_stats",
            "description": "Get a player's full statistics for the match",
            "input_schema": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Player name"},
                },
                "required": ["player_name"],
            },
        },
        {
            "name": "get_economy_timeline",
            "description": "Get team economy state across all rounds",
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "enum": ["CT", "T"],
                        "description": "Team to get economy for",
                    },
                },
                "required": ["team"],
            },
        },
        {
            "name": "get_kills_by_round",
            "description": "Get all kills in a specific round with positions and weapons",
            "input_schema": {
                "type": "object",
                "properties": {
                    "round_number": {
                        "type": "integer",
                        "description": "Round number",
                    },
                },
                "required": ["round_number"],
            },
        },
        {
            "name": "get_utility_usage",
            "description": "Get all utility (grenade) usage for a round or entire match",
            "input_schema": {
                "type": "object",
                "properties": {
                    "round_number": {
                        "type": "integer",
                        "description": "Round number (omit for all rounds)",
                    },
                },
            },
        },
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        """
        Initialize TacticalAIClient.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-5-20250929)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=60,  # Longer timeout for complex analysis
                )
            except ImportError as e:
                raise ImportError(
                    "Anthropic library not installed. Install with: pip install anthropic"
                ) from e
        return self._client

    def _execute_tool(self, tool_name: str, tool_input: dict, match_data: dict) -> str:
        """Execute a tool call with real match data."""
        import json

        if tool_name == "get_round_data":
            round_num = tool_input.get("round_number", 1)
            timeline = match_data.get("round_timeline", [])
            for r in timeline:
                if r.get("round_num") == round_num:
                    return json.dumps(r, indent=2, default=str)
            return json.dumps({"error": f"Round {round_num} not found"})

        elif tool_name == "get_player_stats":
            name = tool_input.get("player_name", "").lower()
            players = match_data.get("players", {})
            for _sid, player in players.items():
                if player.get("name", "").lower() == name:
                    return json.dumps(player, indent=2, default=str)
            return json.dumps({"error": f"Player '{name}' not found"})

        elif tool_name == "get_economy_timeline":
            team = tool_input.get("team", "CT")
            timeline = match_data.get("round_timeline", [])
            economy = []
            team_key = "ct" if team == "CT" else "t"
            for r in timeline:
                rn = r.get("round_num", 0)
                econ = r.get("economy") or {}
                team_econ = econ.get(team_key) or {}
                economy.append(
                    {
                        "round": rn,
                        "equipment_value": team_econ.get("equipment", 0),
                        "round_type": team_econ.get("buy_type", "unknown"),
                        "loss_bonus": team_econ.get("loss_bonus", 0),
                        "decision_grade": team_econ.get("decision_grade", ""),
                    }
                )
            return json.dumps(economy, indent=2, default=str)

        elif tool_name == "get_kills_by_round":
            round_num = tool_input.get("round_number", 1)
            timeline = match_data.get("round_timeline", [])
            for r in timeline:
                if r.get("round_num") == round_num:
                    kills = r.get("kills", [])
                    return json.dumps(kills, indent=2, default=str)
            return json.dumps({"error": f"Round {round_num} not found"})

        elif tool_name == "get_utility_usage":
            # Extract utility from round timeline
            round_num = tool_input.get("round_number")
            timeline = match_data.get("round_timeline", [])
            if round_num:
                for r in timeline:
                    if r.get("round_num") == round_num:
                        return json.dumps(r.get("utility", []), indent=2, default=str)
                return json.dumps({"error": f"Round {round_num} not found"})
            # Return all utility across all rounds
            all_utility = {}
            for r in timeline:
                rn = r.get("round_num", 0)
                util = r.get("utility", [])
                if util:
                    all_utility[f"round_{rn}"] = util
            return json.dumps(all_utility, indent=2, default=str)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def analyze(
        self,
        match_data: dict,
        analysis_type: str = "overview",
        focus: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate tactical analysis using Claude with tool-use.

        Args:
            match_data: Parsed match data from CachedAnalyzer
            analysis_type: Type of analysis (overview, strat-steal, self-review, scout)
            focus: Optional focus (specific round, player, or side)
            system_prompt: Optional custom system prompt

        Returns:
            Markdown-formatted tactical report
        """

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not configured. "
                "Set environment variable or pass api_key to constructor."
            )

        # Import system prompts
        from opensight.ai.tactical import get_system_prompt

        # Get appropriate system prompt
        if system_prompt is None:
            system_prompt = get_system_prompt(analysis_type)

        # Build initial context
        match_info = match_data.get("demo_info", {})
        map_name = match_info.get("map", "unknown")
        total_rounds = match_info.get("rounds", 0)
        players = match_data.get("players", {})
        player_names = [p.get("name", "Unknown") for p in players.values()]

        # Build user prompt
        focus_str = f" Focus on: {focus}." if focus else ""
        user_prompt = f"""Analyze this CS2 match:

**Match Info:**
- Map: {map_name}
- Total Rounds: {total_rounds}
- Players: {", ".join(player_names[:10])}

**Analysis Type:** {analysis_type}
{focus_str}

Use the tools available to query specific round data, player stats, and economy timeline.
Then generate a comprehensive tactical report in markdown format."""

        try:
            client = self._get_client()

            # Initial message
            messages = [{"role": "user", "content": user_prompt}]

            logger.info(
                f"Starting tactical analysis: type={analysis_type}, map={map_name}, rounds={total_rounds}"
            )

            # Tool-use loop (max 10 iterations to prevent infinite loops)
            iterations_count = 0
            for _ in range(10):
                iterations_count += 1
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=self.ANALYSIS_TOOLS,
                    messages=messages,
                )

                # Check if we got tool calls
                if response.stop_reason == "tool_use":
                    # Collect all tool calls from the response
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            logger.debug(f"Tool call: {block.name}({block.input})")
                            result = self._execute_tool(block.name, block.input, match_data)
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result,
                                }
                            )

                    # Add assistant's response and tool results to messages
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # No more tool calls, extract final text
                    break

            # Extract final text response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            logger.info(
                f"Tactical analysis complete ({len(final_text)} chars, {iterations_count} iterations)"
            )
            return final_text

        except Exception as e:
            logger.error(f"Tactical analysis failed: {e}")
            return f"""**Tactical Analysis Error**

Unable to generate analysis: {type(e).__name__}

Please check your ANTHROPIC_API_KEY configuration or try again later."""


# Singleton instance for TacticalAIClient
_tactical_ai_instance: TacticalAIClient | None = None


def get_tactical_ai_client() -> TacticalAIClient:
    """Get or create singleton TacticalAIClient instance."""
    global _tactical_ai_instance
    if _tactical_ai_instance is None:
        _tactical_ai_instance = TacticalAIClient()
    return _tactical_ai_instance
