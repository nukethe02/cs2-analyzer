"""
LLM Client for AI Coaching Summaries

Integrates OpenAI API (or compatible endpoints) to generate
personalized match analysis and coaching insights.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with LLM APIs (OpenAI-compatible).

    Supports:
    - OpenAI API
    - Azure OpenAI
    - Local OpenAI-compatible endpoints (LM Studio, Ollama, etc.)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: int = 30,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL (for Azure, local LLMs, etc.)
            model: Model to use (default: gpt-4o-mini for cost efficiency)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout

        # Lazy import to avoid requiring openai if not used
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError as e:
                raise ImportError(
                    "OpenAI library not installed. Install with: pip install openai"
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
                "OPENAI_API_KEY not configured. Set environment variable or pass api_key to constructor."
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
- Crosshair Placement: {cp:.1f}° error
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

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
                temperature=0.7,
            )

            summary = response.choices[0].message.content
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

Please check your OPENAI_API_KEY configuration or try again later."""


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
