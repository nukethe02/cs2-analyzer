"""
Two-Tier LLM Client for AI Coaching and Tactical Analysis.

Architecture:
  - STANDARD tier (Haiku 4.5): Fast, cheap — 90% of calls
  - DEEP tier (Sonnet 4.5): Complex analysis — 10% of calls

Prompt caching: System prompt is >4096 tokens with cache_control
header so repeated calls within 5 minutes pay 90% less on input tokens.

Cost tracking: Every API call logs model, tokens, estimated cost,
and cache hit status.
"""

import logging
import os
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Model Tier Configuration
# =============================================================================


class ModelTier(StrEnum):
    """Two-tier model selection for cost optimization."""

    STANDARD = "claude-haiku-4-5-20251001"  # Fast, cheap, 90% of calls
    DEEP = "claude-sonnet-4-5-20250929"  # Complex analysis, 10%


def _get_default_tier() -> ModelTier:
    """Get default tier from environment or fall back to STANDARD."""
    env_tier = os.getenv("LLM_DEFAULT_TIER", "").lower().strip()
    if env_tier == "deep":
        return ModelTier.DEEP
    return ModelTier.STANDARD


# Pricing per million tokens (USD)
_PRICING = {
    ModelTier.STANDARD: {"input": 1.0, "output": 5.0, "cache_read": 0.1},
    ModelTier.DEEP: {"input": 3.0, "output": 15.0, "cache_read": 0.3},
}


def _log_usage(tier: ModelTier, usage: Any) -> None:
    """Log token usage and estimated cost after each API call."""
    prices = _PRICING[tier]
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0)

    # Non-cached input tokens = total input - cache_read - cache_creation
    regular_input = max(0, input_tokens - cache_read - cache_creation)

    input_cost = regular_input * prices["input"] / 1_000_000
    output_cost = output_tokens * prices["output"] / 1_000_000
    cache_read_cost = cache_read * prices["cache_read"] / 1_000_000
    # Cache creation costs 25% more than regular input
    cache_create_cost = cache_creation * prices["input"] * 1.25 / 1_000_000
    total_cost = input_cost + output_cost + cache_read_cost + cache_create_cost

    logger.info(
        "LLM call: model=%s in_tok=%d out_tok=%d cache_read=%d cache_create=%d cost=$%.4f",
        tier.value,
        input_tokens,
        output_tokens,
        cache_read,
        cache_creation,
        total_cost,
    )


# =============================================================================
# Expanded System Prompt (>4096 tokens for prompt caching)
# =============================================================================

CS2_COACHING_SYSTEM_PROMPT = """You are a Tier 1 CS2 Coach with deep expertise in professional Counter-Strike 2.
Analyze player statistics with brutal honesty but constructive feedback.
Focus on actionable improvements backed by specific metrics.
Never hallucinate stats — only reference the exact numbers provided.

## Coaching Framework

Good CS2 coaching feedback is:
1. SPECIFIC — cite exact numbers, rounds, and situations. "Your ADR of 62.3
   is below the 75+ threshold for your role" is better than "Your damage is low."
2. ACTIONABLE — every critique must include a concrete drill, habit change, or
   practice routine. "Workshop 15 minutes of prefire peek practice on Mirage
   A-ramp daily" beats "Improve your aim."
3. GROUNDED IN CONTEXT — a 0.85 rating as entry fragger on a 16-14 loss is
   very different from a 0.85 rating as support on a 16-3 stomp. Always consider
   the match context, role, and scoreline.
4. PRIORITIZED — identify the ONE thing that would have the biggest impact if
   fixed. Players can only focus on improving one skill at a time.
5. HONEST — do not soften bad performances. A 0.65 rating is terrible. Say so.
   But also explain the path to improvement so the feedback is constructive.

## Performance Benchmarks by Skill Level

### HLTV 2.0 Rating
| Level              | Rating Range | Description                          |
|--------------------|-------------|---------------------------------------|
| Elite (Level 10+)  | 1.30+       | Star player, carrying the team        |
| Advanced (Main)    | 1.10-1.29   | Consistent positive impact            |
| Intermediate (IM)  | 0.90-1.09   | Average — breaking even               |
| Developing (Open)  | 0.70-0.89   | Below average, needs improvement      |
| Struggling         | <0.70       | Significantly underperforming         |

### ADR (Average Damage per Round)
| Level     | ADR    | What It Means                              |
|-----------|--------|--------------------------------------------|
| Elite     | 90+    | Consistently winning duels and dealing chip |
| Good      | 75-89  | Pulling your weight                        |
| Average   | 60-74  | Need to find more engagements              |
| Low       | <60    | Not engaging enough or losing most duels   |

### Time to Damage (TTD) — engagement duration in milliseconds
| Level     | TTD (ms) | What It Means                            |
|-----------|----------|------------------------------------------|
| Elite     | <200     | Reacting and hitting almost instantly     |
| Good      | 200-350  | Solid reaction time and crosshair work    |
| Average   | 350-500  | Room to improve crosshair placement       |
| Slow      | >500     | Likely getting caught off-guard often      |

### Crosshair Placement (CP) — angular error in degrees
| Level     | Error (deg) | What It Means                         |
|-----------|-------------|---------------------------------------|
| Elite     | <5          | Crosshair is nearly on-target pre-aim |
| Good      | 5-15        | Solid crosshair discipline            |
| Average   | 15-25       | Needs crosshair placement practice    |
| Poor      | >25         | Significant crosshair placement issue |

### KAST% (Kill/Assist/Survive/Trade percentage)
| Level     | KAST%  | What It Means                              |
|-----------|--------|--------------------------------------------|
| Elite     | 80+    | Contributing meaningfully nearly every round |
| Good      | 70-79  | Solid round-by-round contribution          |
| Average   | 60-69  | Inconsistent round impact                  |
| Low       | <60    | Too many rounds with zero contribution     |

### Trade Kill Success Rate
| Level     | Rate   | What It Means                              |
|-----------|--------|--------------------------------------------|
| Elite     | 70+%   | Almost always trading fallen teammates     |
| Good      | 50-69% | Decent trade discipline                    |
| Average   | 30-49% | Need better positioning for refrags        |
| Poor      | <30%   | Major trading discipline problem           |

## CS2 Role Definitions and Key Metrics

### Entry Fragger
Primary responsibility: Be first into the site, create space, get opening kills.
Key metrics: Opening duel win rate (>55% is good), flash-assisted entries, ADR.
Common mistakes: Dry peeking without utility, inconsistent timing, not
communicating what they see before dying.

### AWPer
Primary responsibility: Hold angles, get picks, control map areas.
Key metrics: AWP kill efficiency, opening picks, impact rating, deaths with
AWP (losing the $4750 investment).
Common mistakes: Over-peeking when team needs them alive, not repositioning
after a kill, taking bad AWP duels.

### Support
Primary responsibility: Flash for entries, throw utility, trade kills, play for team.
Key metrics: Flash assists, KAST%, trade kill success rate, utility damage.
Common mistakes: Throwing utility too early/late, not being in position to
trade, holding utility too long.

### Lurker
Primary responsibility: Create pressure away from the team, punish rotations,
gather information.
Key metrics: Clutch win rate, impact kills during rotations, information
gathered (enemy positions revealed).
Common mistakes: Going for hero plays instead of info, being too far from
the team to trade, getting caught in no-mans-land.

### IGL (In-Game Leader)
Primary responsibility: Call strats, manage economy, read the opponent, adapt.
Key metrics: Team round-win rate on called strats, economy management grade,
mid-round adaptation success, KAST%.
Common mistakes: Calling too late (team already committed), not adapting after
opponent adjusts, micromanaging instead of letting players play.

## Few-Shot Coaching Examples

### Example 1: Strong Entry Fragger with Economy Issues
Player: 25K/16D, 1.18 Rating, 82 ADR, 58% opening duel win rate
Round 14: Force-bought deagle+kevlar at $2400 when 3-round loss bonus was
building to $3400. Full save would have guaranteed AK/M4+utility round 15.
Instead, lost the force AND the follow-up eco, turning a 2-round deficit
into a 4-round deficit.
COACHING: "Your fragging impact is strong (1.18 rating, 58% opening duels).
But your force buy in round 14 was the turning point. At $2400 with loss
bonus building, the save gives you a guaranteed full buy worth $5700+ next
round. That one decision likely cost 2 extra rounds. Rule of thumb: never
force when you can full buy in 1 round with loss bonus."

### Example 2: Low-Impact Support Player
Player: 11K/18D, 0.68 Rating, 52 ADR, 61% KAST, 2/7 trades
COACHING: "0.68 rating with 52 ADR means you are getting eliminated without
enough impact. As support, your KAST of 61% is too low — you should be
contributing to 70%+ of rounds through flashes, trades, or survival. Most
critically, 2/7 trade attempts (29%) means your teammates are dying and you
are not punishing the enemy. Fix: in your next 5 pugs, focus ONLY on
positioning yourself within 5 seconds of your entry fragger. If they die,
you should see the enemy within 1 second."

### Example 3: Inconsistent AWPer
Player: 19K/14D, 1.05 Rating, 71 ADR, 4 AWP kills, 3 opening picks, died
with AWP 6 times
COACHING: "1.05 rating is passable but not what your team needs from the
$4750 investment. You died holding the AWP 6 times — that is $28,500 in
lost equipment across the match. 3 opening picks is decent but 4 total AWP
kills means you got only 1 non-opening kill. After getting your pick, you
need to reposition and look for a second kill, not hold the same angle.
Drill: Play 10 rounds of retake servers focusing on quick-scoping and
repositioning between shots."

## Map Callout Reference (Active Duty Pool)

### Mirage
T-Side: T Spawn, T Ramp, Underpass, Top Mid, Mid Boxes, Palace, A Ramp,
Tetris, Stairs, A Site, CT Spawn, Jungle, Connector, Window, Short/Catwalk,
B Apartments, B Site, Van, Bench, Market/Kitchen, B Short.
CT-Side: CT Spawn, Ticket Booth, Jungle, Connector, Window Room, A Site,
Stairs, Firebox, Triple, Under Palace, B Site, Van, Short, Market Door,
Bench, Cat, Kitchen.

### Inferno
T-Side: T Spawn, T Ramp, Alt Mid, Second Mid, Banana, Car (Banana),
Logs, Half Wall, A Apartments (Apps), A Short/Boiler, A Long, A Site,
Pit, Graveyard, Library, Arch, B Site, First Oranges, Second Oranges,
Dark/Spools, New Box, CT Spawn.
CT-Side: CT Spawn, Arch, Library, Pit, Moto, Site, Balcony, Graveyard,
B Site, Coffins, First Oranges, Dark, New Box, Construction, Banana.

### Nuke
T-Side: T Spawn, Outside, T Roof, Lobby, Squeaky, Hut, Main/Mustang,
A Site (Heaven), Hell, Rafters, Mini/Mini-Ramp, Ramp, B Site (Basement),
Vents, Secret, Decon, CT Red Box, Yard.
CT-Side: CT Spawn, Heaven, Hell, Rafters, Trophy, Control Room, Ramp,
B Site, Decon, Secret, Dark, Silo, Garage, Outside.

### Ancient
T-Side: T Spawn, T Ramp, Mid, Donut, A Main, A Short, A Link, A Site,
Elbow, Alley, B Ramp, B Main, B Site, B Short, Ruins, Cave.
CT-Side: CT Spawn, CT, A Site, A Short, Elbow, Alley, Temple, B Site,
B Ramp, Tunnel, Waterfall, Cave.

### Anubis
T-Side: T Spawn, Mid, T Bridge, Canal, Connector, A Main, A Site,
A Long, Palace, B Main, B Site, B Short, B Long, Walkway, Ruins.
CT-Side: CT Spawn, A Site, Heaven, Boat, Bridge, B Site, B Short,
B Long, Alley, Street, Water.

### Dust2
T-Side: T Spawn, T Long, Long Doors, Long Corner, Blue (Pit), A Long,
A Site, A Short/Catwalk, Mid, Mid Doors (Xbox), Lower Tunnels, Upper
Tunnels, B Tunnels, B Site, B Window, B Doors, B Platform.
CT-Side: CT Spawn, CT Mid, A Site, A Short, A Car, A Platform, Goose,
Elevator, B Site, B Window, B Doors, B Back Platform, B Car.

## Economy Management Reference

### Buy Round Thresholds (per player)
| Round Type    | Equipment Value | When to Use                           |
|---------------|----------------|---------------------------------------|
| Full Buy      | $5000-5700+    | AK/M4 + full utility + armor+helmet   |
| Force Buy     | $2500-4000     | When full save doesn't change outcome  |
| Semi-Eco      | $1500-2500     | Deagle+armor or SMG+armor             |
| Eco/Save      | <$1000         | Save for next round full buy           |
| Pistol Round  | $800           | Default or upgraded pistol + armor     |

### Key Economy Rules
- Loss bonus resets after a win: $1400, $1900, $2400, $2900, $3400 (max)
- Kill rewards: Rifle $300, SMG $600, Shotgun $900, Knife $1500, AWP $100
- Team money should be tracked collectively — one player force buying when
  4 teammates save breaks the team's economy for 2+ rounds
- The "$4750 rule": never force buy when a full save guarantees rifles next round
- Pistol round wins are worth ~3-4 rounds of advantage (economy snowball)

### Economy Decision Grading
- A: Correct buy decision for the situation (team-coordinated full buy/save)
- B: Slightly suboptimal but defensible (semi-force when close to loss bonus max)
- C: Poor decision (scattered buys, no team coordination)
- D: Damaging (force buy that ruins 2+ subsequent rounds)
- F: Critical error (AWP force on eco, dropping weapons to wrong teammates)

## Weapon Meta Reference (CS2 2025)

### Rifles (Primary Weapons)
- AK-47 ($2700): T-side default, one-shot headshot, spray pattern mastery critical
- M4A4 ($3100): CT-side default, higher fire rate, no one-shot headshot
- M4A1-S ($2900): CT alternative, silenced, tighter spray, 20-round magazine
- AWP ($4750): One-shot body kill, $100 kill reward, huge economy risk if lost
- SG 553 ($3000): Scoped rifle, niche pick for long-range angles

### SMGs (Anti-Eco)
- MP9 ($1250): CT anti-eco default, $600 kill reward, high mobility
- MAC-10 ($1050): T anti-eco default, $600 kill reward, very cheap
- MP7 ($1500): Versatile SMG, good against light armor

### Pistols
- Desert Eagle ($700): One-shot headshot at range, high skill ceiling
- USP-S (CT default): Silenced, accurate first shot, 12 rounds
- Glock (T default): Burst fire viable, poor armor penetration
- P250 ($300): Budget upgrade, decent armor penetration

### Utility ($200-400 each, $1000 total for full set)
- Smoke Grenade ($300): 18-second smoke, blocks vision, extinguishes molotovs
- Flashbang ($200, max 2): Blinds enemies, key for entry support
- HE Grenade ($300): 50-100 damage depending on distance and armor
- Molotov/Incendiary ($400/$600): Area denial, forces position changes
- Decoy ($50): Mimics gunfire, rarely useful outside pistol rounds

## Common Tactical Mistakes by Rank

### Silver-Gold Nova (Low Rank)
- Not buying armor on pistol round
- Force buying every round
- Using utility randomly (flashing no one, smoking own team)
- Rushing the same site every T round
- Not watching minimap

### Master Guardian-Distinguished Master Guardian (Mid Rank)
- Poor trade positioning (too far from teammates)
- Using all utility in first 30 seconds
- Not adapting economy to team money
- Predictable A/B site splits
- Forgetting to check common angles

### Supreme-Global Elite / FACEIT Level 7+ (High Rank)
- Dry peeking when utility is available
- Over-rotating on CT side (leaving sites empty)
- Not punishing opponent economy patterns
- Inconsistent communication mid-round
- Ego peeking when playing for time/info

### ESEA Main+ / Semi-Pro (Advanced)
- Not varying default setups enough (predictable)
- Economy mistakes in crucial rounds (13-12, 14-14)
- Utility timing off by 1-2 seconds on executes
- Not anti-stratting opponent's known patterns
- Individual play overriding team structure

## Round Type Classification

### Pistol Rounds (Rounds 1 and 13)
These rounds set the economic trajectory for 3-4 rounds. Winning pistol with
a 3-round win streak is worth approximately $12,000-15,000 in total economic
advantage. Losing pistol means you must survive 1-2 eco rounds before you can
buy. Pistol round performance should be evaluated differently from gun rounds:
- Did the player buy armor? (Failing to buy armor on pistol is almost always wrong)
- Did they use utility effectively? (A well-placed smoke or flash on pistol
  is worth more than on a full buy round because utility is scarce)
- Did they play for trades? (1-for-1 trades on pistol favor the team with
  better economy, which is equal at round start)

### Anti-Eco Rounds (Rounds 2-3, 14-15 after winning pistol)
Expected win rate: 85%+. Losing an anti-eco is a critical mistake because:
- You have rifles vs pistols — the firepower advantage is enormous
- The enemy has nothing to lose — they expect to lose this round
- Losing an anti-eco gives the opponent a free rifle round (your dropped weapons)
Anti-eco losses usually come from: poor positioning (getting rushed), not
using utility to slow pushes, taking unnecessary duels at close range where
pistols are lethal.

### Force Buy Rounds
A force buy should only happen when: (a) you cannot afford a full buy next
round regardless of saving, or (b) it is match point or a critical round
where one more loss means elimination. The most common economy mistake in
competitive CS2 is the "hope force" — buying rifles without utility when a
full save would guarantee a complete buy next round.

### Full Buy Rounds
Expected win rate: 50-55% (CT side advantage). On full buy rounds, individual
mechanical skill matters less and team coordination matters more. If a player
has poor stats on full buy rounds, look at their positioning and utility usage
rather than their aim.

## Analysis Decision Tree

When analyzing a player's performance, follow this priority order:
1. Check HLTV Rating first — is this player performing above or below expectations?
2. Check ADR — is damage output consistent with their role?
3. Check KAST% — are they contributing every round or disappearing?
4. Check opening duels — are entry kills/deaths appropriate for their role?
5. Check trades — are they trading teammates and getting traded?
6. Check economy — any bad force buys or missed save rounds?
7. Check utility — flash assists, wasted utility, team flashes?
8. Check TTD and CP — mechanical aim issues or just decision-making?

Always start with the MOST IMPACTFUL issue. A player with 0.65 rating doesn't
need crosshair placement tips — they need to understand WHY they're not getting
kills (positioning? utility? timing? aim? all of the above?).

## Output Format Rules

1. Always structure output with markdown headers (##, ###)
2. Bold (**) key stats and numbers for scanability
3. Use bullet points for lists of strengths/weaknesses
4. Keep total response under 200 words for match summaries, 800 for tactical
5. Cite specific round numbers when discussing economy or key moments
6. End every analysis with ONE clear action item the player should focus on
7. Never start with generic praise — lead with the most important finding
"""


def _build_cached_system(prompt_text: str) -> list[dict[str, Any]]:
    """Wrap a system prompt string in the Anthropic cache_control format."""
    return [
        {
            "type": "text",
            "text": prompt_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


# =============================================================================
# Batch summary prompt — trimmed essentials only, no map callouts/weapon meta
# =============================================================================

_BATCH_SYSTEM_PROMPT = """You are a Tier 1 CS2 Coach. Analyze each player's statistics with brutal honesty but constructive feedback. Focus on actionable improvements backed by specific metrics. Never hallucinate stats.

## Benchmarks
HLTV Rating: Elite 1.30+, Advanced 1.10-1.29, Average 0.90-1.09, Below 0.70-0.89, Struggling <0.70
ADR: Elite 90+, Good 75-89, Average 60-74, Low <60
KAST%: Elite 80+, Good 70-79, Average 60-69, Low <60
Trade Kill Rate: Elite 70%+, Good 50-69%, Average 30-49%, Poor <30%
TTD (engagement speed): Elite <200ms, Good 200-350ms, Average 350-500ms, Slow >500ms
CP (crosshair error): Elite <5 deg, Good 5-15, Average 15-25, Poor >25

## Roles
Entry: Opening duel win rate >55% is good. Support: KAST >70%, trade kills, flash assists.
AWPer: Impact picks, don't die with AWP. Lurker: Clutch wins, rotation punishes.

## Output Rules
For EACH player, write 3-5 sentences:
1. Lead with the most important finding (best or worst stat)
2. Cite 2-3 specific numbers
3. End with ONE concrete action item
Be harsh on bad performances. A 0.65 rating is terrible — say so.
Format each as markdown. Bold key stats."""


# =============================================================================
# LLMClient — simple single-call coaching summaries
# =============================================================================


class LLMClient:
    """
    Client for generating AI coaching summaries.

    Uses two-tier model selection:
      - STANDARD (Haiku 4.5): default for match summaries
      - DEEP (Sonnet 4.5): for complex/important analyses
    """

    def __init__(
        self,
        api_key: str | None = None,
        tier: ModelTier | None = None,
        timeout: int = 30,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            tier: Model tier to use (defaults to LLM_DEFAULT_TIER env var or STANDARD)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_tier = tier or _get_default_tier()
        self.timeout = timeout

        # Backward compat: expose model string
        self.model = self.default_tier.value

        # Lazy import to avoid requiring anthropic if not used
        self._client = None

        # Cache the system prompt
        self.system_prompt = CS2_COACHING_SYSTEM_PROMPT

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
        self,
        player_stats: dict[str, Any],
        match_context: dict[str, Any] | None = None,
        tier: ModelTier | None = None,
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
            tier: Override model tier for this call

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

        use_tier = tier or self.default_tier

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
                "Generating LLM summary: tier=%s, %dK/%dD, Rating=%.2f",
                use_tier.value,
                kills,
                deaths,
                rating,
            )

            message = client.messages.create(
                model=use_tier.value,
                max_tokens=400,
                system=_build_cached_system(self.system_prompt),
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )

            _log_usage(use_tier, message.usage)

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

    def generate_batch_summaries(
        self,
        all_player_stats: list[dict[str, Any]],
        match_context: dict[str, Any] | None = None,
        tier: ModelTier | None = None,
    ) -> dict[str, str]:
        """
        Generate coaching summaries for ALL players in a single LLM call.

        Args:
            all_player_stats: List of player stat dicts (same format as generate_match_summary)
            match_context: Optional match context (map, rounds, scores)
            tier: Override model tier

        Returns:
            Dict mapping player name to markdown summary string
        """
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not configured. Set environment variable or pass api_key to constructor."
            )

        if not all_player_stats:
            return {}

        use_tier = tier or self.default_tier

        # Build context header
        context_str = ""
        if match_context:
            map_name = match_context.get("map_name", "")
            rounds = match_context.get("total_rounds", 0)
            t1 = match_context.get("team1_score", 0)
            t2 = match_context.get("team2_score", 0)
            context_str = f"Match: {map_name}, Score: {t1}-{t2}, Rounds: {rounds}\n\n"

        # Build all players into one prompt
        players_text = ""
        player_names = []
        for ps in all_player_stats:
            name = ps.get("name", "Unknown")
            player_names.append(name)
            kills = ps.get("kills", 0)
            deaths = ps.get("deaths", 0)
            assists = ps.get("assists", 0)
            rating = ps.get("hltv_rating", 0.0)
            adr = ps.get("adr", 0.0)
            hs_pct = ps.get("headshot_pct", 0.0)
            kast = ps.get("kast_percentage", 0.0)
            ttd = ps.get("ttd_median_ms", 0)
            cp = ps.get("cp_median_error_deg", 0.0)
            entry_k = ps.get("entry_kills", 0)
            entry_d = ps.get("entry_deaths", 0)
            trade_s = ps.get("trade_kill_success", 0)
            trade_o = ps.get("trade_kill_opportunities", 0)
            clutch_w = ps.get("clutch_wins", 0)
            clutch_a = ps.get("clutch_attempts", 0)

            players_text += f"### {name}\n"
            players_text += (
                f"K/D/A: {kills}/{deaths}/{assists} | "
                f"Rating: {rating:.2f} | ADR: {adr:.1f} | "
                f"HS: {hs_pct:.0f}% | KAST: {kast:.0f}%\n"
            )
            players_text += (
                f"TTD: {ttd:.0f}ms | CP: {cp:.1f}deg | "
                f"Entry: {entry_k}K/{entry_d}D | "
                f"Trades: {trade_s}/{trade_o} | "
                f"Clutches: {clutch_w}/{clutch_a}\n\n"
            )

        user_prompt = f"""{context_str}Analyze each player below. Respond with a JSON object where keys are EXACT player names and values are the markdown coaching summary string.

{players_text}
Respond ONLY with valid JSON. Example format:
{{"PlayerName": "**1.23 rating** with ...", "Player2": "..."}}"""

        try:
            client = self._get_client()

            logger.info(
                "Generating batched LLM summaries: tier=%s, players=%d",
                use_tier.value,
                len(all_player_stats),
            )

            message = client.messages.create(
                model=use_tier.value,
                max_tokens=2500,
                system=_build_cached_system(_BATCH_SYSTEM_PROMPT),
                messages=[{"role": "user", "content": user_prompt}],
            )

            _log_usage(use_tier, message.usage)

            raw = message.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

            import json

            summaries = json.loads(raw)

            logger.info(f"Batched summaries generated for {len(summaries)} players")
            return summaries

        except Exception as e:
            logger.error(f"Batched LLM generation failed: {e}")
            # Return empty — caller will use per-player fallback
            return {}


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
    Convenience function to generate match summary for a single player.

    Args:
        player_stats: Player statistics dictionary
        match_context: Optional match context (map, result, etc.)

    Returns:
        Markdown-formatted coaching summary
    """
    client = get_llm_client()
    return client.generate_match_summary(player_stats, match_context)


def generate_batch_summaries(
    all_player_stats: list[dict[str, Any]],
    match_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Convenience function to generate summaries for all players in one call.

    Returns:
        Dict mapping player name to summary string. Empty dict on failure.
    """
    client = get_llm_client()
    return client.generate_batch_summaries(all_player_stats, match_context)


# =============================================================================
# TacticalAIClient - Claude-powered tactical analysis with tool-use
# =============================================================================


class TacticalAIClient:
    """
    Claude-powered tactical analysis for CS2 demos.

    Uses Claude's tool-use (function calling) to query match data
    and generate comprehensive tactical reports.

    Default tier: STANDARD (Haiku 4.5) for most analyses.
    Use DEEP tier for anti-strat generation and game plans.
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
        tier: ModelTier | None = None,
    ):
        """
        Initialize TacticalAIClient.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            tier: Model tier (defaults to LLM_DEFAULT_TIER env var or STANDARD)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_tier = tier or _get_default_tier()
        self.model = self.default_tier.value
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
        """Execute a tool call with real match data.

        Returns compressed summaries instead of raw dicts to reduce token usage.
        Full player dicts can be 4-8K tokens; compressed versions are ~300-500 tokens.
        """
        import json

        if tool_name == "get_round_data":
            round_num = tool_input.get("round_number", 1)
            timeline = match_data.get("round_timeline", [])
            for r in timeline:
                if r.get("round_num") == round_num:
                    return json.dumps(self._compress_round(r), default=str)
            return json.dumps({"error": f"Round {round_num} not found"})

        elif tool_name == "get_player_stats":
            name = tool_input.get("player_name", "").lower()
            players = match_data.get("players", {})
            for _sid, player in players.items():
                if player.get("name", "").lower() == name:
                    return json.dumps(self._compress_player(player), default=str)
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
            return json.dumps(economy, default=str)

        elif tool_name == "get_kills_by_round":
            round_num = tool_input.get("round_number", 1)
            timeline = match_data.get("round_timeline", [])
            for r in timeline:
                if r.get("round_num") == round_num:
                    kills = r.get("kills", [])
                    return json.dumps([self._compress_kill(k) for k in kills], default=str)
            return json.dumps({"error": f"Round {round_num} not found"})

        elif tool_name == "get_utility_usage":
            round_num = tool_input.get("round_number")
            timeline = match_data.get("round_timeline", [])
            if round_num:
                for r in timeline:
                    if r.get("round_num") == round_num:
                        utils = r.get("utility", [])
                        return json.dumps([self._compress_utility(u) for u in utils], default=str)
                return json.dumps({"error": f"Round {round_num} not found"})
            # Summary per round instead of dumping every event
            summary = {}
            for r in timeline:
                rn = r.get("round_num", 0)
                utils = r.get("utility", [])
                if utils:
                    by_type: dict[str, int] = {}
                    for u in utils:
                        t = u.get("type", "unknown")
                        by_type[t] = by_type.get(t, 0) + 1
                    summary[f"R{rn}"] = by_type
            return json.dumps(summary, default=str)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    @staticmethod
    def _compress_player(player: dict) -> dict:
        """Compress a full player dict (~4-8K tokens) to essentials (~300 tokens)."""
        stats = player.get("stats", {})
        rating = player.get("rating", {})
        adv = player.get("advanced", {})
        entry = player.get("entry", {})
        trades = player.get("trades", {})
        clutches = player.get("clutches", {})
        util = player.get("utility", {})
        duels = player.get("duels", {})

        return {
            "name": player.get("name"),
            "team": player.get("team"),
            "kills": stats.get("kills", 0),
            "deaths": stats.get("deaths", 0),
            "assists": stats.get("assists", 0),
            "adr": stats.get("adr", 0),
            "hs_pct": stats.get("headshot_pct", 0),
            "hltv_rating": rating.get("hltv_rating", 0),
            "kast": rating.get("kast_percentage", 0),
            "aim_rating": rating.get("aim_rating", 0),
            "utility_rating": rating.get("utility_rating", 0),
            "impact_rating": rating.get("impact_rating", 0),
            "ttd_ms": adv.get("ttd_median_ms"),
            "cp_deg": adv.get("cp_median_error_deg"),
            "entry_kills": entry.get("entry_kills", 0),
            "entry_deaths": entry.get("entry_deaths", 0),
            "entry_success_pct": entry.get("entry_success_pct", 0),
            "trade_kill_success": trades.get("trade_kill_success", 0),
            "trade_kill_opps": trades.get("trade_kill_opportunities", 0),
            "trade_kill_pct": trades.get("trade_kill_success_pct", 0),
            "untraded_deaths": trades.get("untraded_deaths", 0),
            "clutch_wins": clutches.get("clutch_wins", 0),
            "clutch_total": clutches.get("total_situations", 0),
            "opening_kills": duels.get("opening_kills", 0),
            "opening_deaths": duels.get("opening_deaths", 0),
            "flash_assists": util.get("flash_assists", 0),
            "enemies_flashed": util.get("enemies_flashed", 0),
            "he_damage": util.get("he_damage", 0),
            "molotov_damage": util.get("molotov_damage", 0),
            "util_thrown": (
                util.get("flashbangs_thrown", 0)
                + util.get("smokes_thrown", 0)
                + util.get("he_thrown", 0)
                + util.get("molotovs_thrown", 0)
            ),
            "multi_kills": {
                "2k": stats.get("2k", 0),
                "3k": stats.get("3k", 0),
                "4k": stats.get("4k", 0),
                "5k": stats.get("5k", 0),
            },
        }

    @staticmethod
    def _compress_round(r: dict) -> dict:
        """Compress a full round dict to essentials. Drops coordinates/positions."""
        kills = r.get("kills", [])
        compressed_kills = [
            {
                "killer": k.get("killer"),
                "victim": k.get("victim"),
                "weapon": k.get("weapon"),
                "headshot": k.get("headshot"),
                "killer_team": k.get("killer_team"),
            }
            for k in kills
        ]
        econ = r.get("economy") or {}
        ct_econ = econ.get("ct") or {}
        t_econ = econ.get("t") or {}

        return {
            "round_num": r.get("round_num"),
            "winner": r.get("winner"),
            "win_reason": r.get("win_reason"),
            "round_type": r.get("round_type"),
            "first_kill": r.get("first_kill"),
            "first_death": r.get("first_death"),
            "ct_kills": r.get("ct_kills", 0),
            "t_kills": r.get("t_kills", 0),
            "kills": compressed_kills,
            "economy": {
                "ct_buy": ct_econ.get("buy_type", "unknown"),
                "ct_equip": ct_econ.get("equipment", 0),
                "t_buy": t_econ.get("buy_type", "unknown"),
                "t_equip": t_econ.get("equipment", 0),
            },
            "clutches": r.get("clutches", []),
        }

    @staticmethod
    def _compress_kill(k: dict) -> dict:
        """Compress a kill event — drop coordinates, keep tactical info."""
        return {
            "killer": k.get("killer"),
            "victim": k.get("victim"),
            "weapon": k.get("weapon"),
            "headshot": k.get("headshot"),
            "killer_team": k.get("killer_team"),
            "is_trade": k.get("is_trade"),
            "is_first_kill": k.get("is_first_kill"),
        }

    @staticmethod
    def _compress_utility(u: dict) -> dict:
        """Compress a utility event — drop coordinates, keep type and player."""
        return {
            "type": u.get("type"),
            "player": u.get("player"),
            "team": u.get("team"),
        }

    def analyze(
        self,
        match_data: dict,
        analysis_type: str = "overview",
        focus: str | None = None,
        system_prompt: str | None = None,
        tier: ModelTier | None = None,
    ) -> str:
        """
        Generate tactical analysis using Claude with tool-use.

        Args:
            match_data: Parsed match data from CachedAnalyzer
            analysis_type: Type of analysis (overview, strat-steal, self-review, scout)
            focus: Optional focus (specific round, player, or side)
            system_prompt: Optional custom system prompt
            tier: Override model tier for this call

        Returns:
            Markdown-formatted tactical report
        """

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not configured. "
                "Set environment variable or pass api_key to constructor."
            )

        use_tier = tier or self.default_tier

        # Import system prompts
        from opensight.ai.tactical import get_system_prompt

        # Get appropriate system prompt
        if system_prompt is None:
            system_prompt = get_system_prompt(analysis_type)

        # Pre-process match data into structured summary for better LLM context
        from opensight.ai.data_prep import preprocess_match, to_llm_prompt

        match_summary = preprocess_match(match_data)

        # Map analysis_type to data_prep focus
        focus_map = {
            "overview": "coaching",
            "strat-steal": "scouting",
            "self-review": "coaching",
            "scout": "scouting",
            "quick": "coaching",
        }
        prep_focus = focus_map.get(analysis_type, "coaching")
        structured_data = to_llm_prompt(match_summary, focus=prep_focus)

        map_name = match_summary.map_name
        total_rounds = match_summary.total_rounds

        # Build user prompt with preprocessed data + tool instructions
        focus_str = f" Focus on: {focus}." if focus else ""
        user_prompt = f"""Analyze this CS2 match:

{structured_data}

**Analysis Type:** {analysis_type}
{focus_str}

Use the tools available to query specific round data, player stats, and economy
timeline for deeper investigation. The structured data above gives you the overview —
use tools to drill into specific rounds or players that stand out.
Generate a comprehensive tactical report in markdown format."""

        try:
            client = self._get_client()

            # Initial message
            messages = [{"role": "user", "content": user_prompt}]

            logger.info(
                "Starting tactical analysis: tier=%s, type=%s, map=%s, rounds=%d",
                use_tier.value,
                analysis_type,
                map_name,
                total_rounds,
            )

            # Tool-use loop (max 10 iterations to prevent infinite loops)
            iterations_count = 0
            response = None
            for _ in range(10):
                iterations_count += 1
                response = client.messages.create(
                    model=use_tier.value,
                    max_tokens=4096,
                    system=_build_cached_system(system_prompt),
                    tools=self.ANALYSIS_TOOLS,
                    messages=messages,
                )

                _log_usage(use_tier, response.usage)

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
            if response is not None:
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
