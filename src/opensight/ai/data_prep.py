"""
Pre-processes orchestrator output into LLM-ready structured summaries.

Computes derived metrics so the LLM explains insights rather than
calculating them.  Reduces token count by ~60-80% vs sending raw JSON.

Usage:
    from opensight.ai.data_prep import preprocess_match, to_llm_prompt

    summary = preprocess_match(orchestrator_result)
    prompt  = to_llm_prompt(summary, focus="coaching")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class PlayerSummary:
    """Pre-computed player metrics for LLM consumption."""

    name: str = ""
    steam_id: str = ""
    team: str = ""
    role_guess: str = "unknown"

    # Core stats
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    adr: float = 0.0
    kast_pct: float = 0.0
    headshot_pct: float = 0.0
    rating: float = 0.0

    # Opening duels
    opening_duel_attempts: int = 0
    opening_duel_wins: int = 0
    opening_duel_win_rate: float = 0.0

    # Clutches
    clutch_attempts: int = 0
    clutch_wins: int = 0
    clutch_win_rate: float = 0.0

    # Trades
    trades_given: int = 0  # times this player traded for a teammate
    trades_missed: int = 0  # untraded deaths
    trade_rate: float = 0.0  # traded deaths / total deaths

    # Utility
    utility_damage: float = 0.0
    utility_thrown: int = 0
    flash_assists: int = 0

    # Positioning
    dry_peek_deaths: int = 0
    total_deaths_with_peek_data: int = 0
    dry_peek_rate: float = 0.0

    # Side stats
    ct_kills: int = 0
    t_kills: int = 0

    # Economy context
    eco_frags: int = 0
    full_buy_kills: int = 0
    awp_kills: int = 0


@dataclass
class RoundSummary:
    """Pre-computed round metrics."""

    round_number: int = 0
    winner: str = ""
    win_method: str = ""
    score_after_ct: int = 0
    score_after_t: int = 0

    # Economy
    ct_buy_type: str = ""
    t_buy_type: str = ""
    ct_equipment_value: int = 0
    t_equipment_value: int = 0

    # Key events
    first_kill_player: str = ""
    first_kill_victim: str = ""
    first_kill_side: str = ""
    clutch_player: str | None = None
    clutch_situation: str | None = None
    clutch_won: bool = False

    # Tactical
    bomb_site: str | None = None
    utility_used_before_entry: int = 0
    total_kills: int = 0


@dataclass
class MatchSummary:
    """Complete pre-processed match data for LLM input."""

    map_name: str = ""
    final_score: str = ""
    total_rounds: int = 0
    winner: str = ""
    overtime: bool = False

    team_a_name: str = ""
    team_b_name: str = ""
    team_a_players: list[PlayerSummary] = field(default_factory=list)
    team_b_players: list[PlayerSummary] = field(default_factory=list)
    rounds: list[RoundSummary] = field(default_factory=list)

    # Team-level derived stats
    team_a_pistol_wins: int = 0
    team_b_pistol_wins: int = 0
    team_a_ct_rounds: int = 0
    team_a_t_rounds: int = 0
    team_b_ct_rounds: int = 0
    team_b_t_rounds: int = 0
    team_a_first_kill_rate: float = 0.0
    team_b_first_kill_rate: float = 0.0

    # Economy patterns
    eco_round_wins_a: int = 0
    eco_round_total_a: int = 0
    eco_round_wins_b: int = 0
    eco_round_total_b: int = 0
    force_buy_wins_a: int = 0
    force_buy_total_a: int = 0
    force_buy_wins_b: int = 0
    force_buy_total_b: int = 0

    # Momentum
    longest_win_streak_a: int = 0
    longest_win_streak_b: int = 0
    comeback_rounds: list[int] = field(default_factory=list)


# =============================================================================
# preprocess_match()
# =============================================================================


def preprocess_match(result: dict) -> MatchSummary:
    """
    Convert raw orchestrator output to a MatchSummary.

    Args:
        result: Orchestrator result dict with keys: demo_info, players,
                round_timeline, etc.

    Returns:
        MatchSummary with all derived metrics pre-computed.
    """
    demo_info = result.get("demo_info") or {}
    players_raw = result.get("players") or {}
    timeline = result.get("round_timeline") or []

    # --- Match metadata ---
    score_ct = demo_info.get("score_ct", 0)
    score_t = demo_info.get("score_t", 0)
    total_rounds = demo_info.get("rounds", 0) or len(timeline)

    summary = MatchSummary(
        map_name=demo_info.get("map", "unknown"),
        final_score=f"{score_ct}-{score_t}",
        total_rounds=total_rounds,
        winner="CT" if score_ct > score_t else ("T" if score_t > score_ct else "Draw"),
        overtime=total_rounds > 30,
        team_a_name=demo_info.get("team1_name", "Team A"),
        team_b_name=demo_info.get("team2_name", "Team B"),
    )

    # --- Build player summaries ---
    team_a_names: set[str] = set()
    team_b_names: set[str] = set()

    for steam_id, pdata in players_raw.items():
        ps = _build_player_summary(steam_id, pdata, timeline)
        team = pdata.get("team", "")
        # First team encountered = team_a (usually CT first half)
        if not team_a_names and not team_b_names:
            team_a_names.add(ps.name)
            summary.team_a_players.append(ps)
        elif ps.name in team_a_names or (
            team and any(p.team == team for p in summary.team_a_players)
        ):
            team_a_names.add(ps.name)
            summary.team_a_players.append(ps)
        elif ps.name in team_b_names or (
            team and any(p.team == team for p in summary.team_b_players)
        ):
            team_b_names.add(ps.name)
            summary.team_b_players.append(ps)
        else:
            # Assign to team_b if team_a already has 5
            if len(summary.team_a_players) < 5:
                team_a_names.add(ps.name)
                summary.team_a_players.append(ps)
            else:
                team_b_names.add(ps.name)
                summary.team_b_players.append(ps)

    # Sort players by rating descending within each team
    summary.team_a_players.sort(key=lambda p: p.rating, reverse=True)
    summary.team_b_players.sort(key=lambda p: p.rating, reverse=True)

    # --- Build round summaries and team-level stats ---
    ct_score_running = 0
    t_score_running = 0
    streak_a = 0
    streak_b = 0
    prev_winner = ""
    team_a_first_kills = 0
    team_b_first_kills = 0
    rounds_with_first_kill = 0

    for rdata in timeline:
        rs = _build_round_summary(rdata)

        winner = rs.winner
        rnum = rs.round_number

        # Running score
        if winner == "CT":
            ct_score_running += 1
        elif winner == "T":
            t_score_running += 1
        rs.score_after_ct = ct_score_running
        rs.score_after_t = t_score_running

        summary.rounds.append(rs)

        # --- Team-level aggregation ---

        # Determine which "team" (a or b) won this round.
        # Team A = first-half CT, Team B = first-half T
        # After halftime (round 13+), sides swap.
        is_first_half = rnum <= 12
        team_a_is_ct = is_first_half
        team_a_won = (winner == "CT" and team_a_is_ct) or (winner == "T" and not team_a_is_ct)

        # Side-specific round wins
        if team_a_won:
            if team_a_is_ct:
                summary.team_a_ct_rounds += 1
            else:
                summary.team_a_t_rounds += 1
        else:
            if team_a_is_ct:
                summary.team_b_t_rounds += 1
            else:
                summary.team_b_ct_rounds += 1

        # Pistol rounds
        if rnum == 1:
            if team_a_won:
                summary.team_a_pistol_wins += 1
            else:
                summary.team_b_pistol_wins += 1
        elif rnum == 13:
            if team_a_won:
                summary.team_a_pistol_wins += 1
            else:
                summary.team_b_pistol_wins += 1

        # First kill tracking
        if rs.first_kill_side:
            rounds_with_first_kill += 1
            fk_is_team_a = (rs.first_kill_side == "CT" and team_a_is_ct) or (
                rs.first_kill_side == "T" and not team_a_is_ct
            )
            if fk_is_team_a:
                team_a_first_kills += 1
            else:
                team_b_first_kills += 1

        # Economy patterns — use the LOSING team's buy type
        if team_a_won:
            # Team B lost — check their buy type
            b_buy = rs.t_buy_type if team_a_is_ct else rs.ct_buy_type
            a_buy = rs.ct_buy_type if team_a_is_ct else rs.t_buy_type
        else:
            # Team A lost — check their buy type
            a_buy = rs.ct_buy_type if team_a_is_ct else rs.t_buy_type
            b_buy = rs.t_buy_type if team_a_is_ct else rs.ct_buy_type

        # Track eco/force rounds for each team
        if a_buy in ("eco", "semi_eco"):
            summary.eco_round_total_a += 1
            if team_a_won:
                summary.eco_round_wins_a += 1
        elif a_buy in ("force", "half_buy"):
            summary.force_buy_total_a += 1
            if team_a_won:
                summary.force_buy_wins_a += 1

        if b_buy in ("eco", "semi_eco"):
            summary.eco_round_total_b += 1
            if not team_a_won:
                summary.eco_round_wins_b += 1
        elif b_buy in ("force", "half_buy"):
            summary.force_buy_total_b += 1
            if not team_a_won:
                summary.force_buy_wins_b += 1

        # Streak tracking
        if team_a_won:
            if prev_winner == "a":
                streak_a += 1
            else:
                # Team A broke Team B's streak
                if streak_b >= 3:
                    summary.comeback_rounds.append(rnum)
                streak_a = 1
                streak_b = 0
            prev_winner = "a"
        else:
            if prev_winner == "b":
                streak_b += 1
            else:
                if streak_a >= 3:
                    summary.comeback_rounds.append(rnum)
                streak_b = 1
                streak_a = 0
            prev_winner = "b"

        summary.longest_win_streak_a = max(summary.longest_win_streak_a, streak_a)
        summary.longest_win_streak_b = max(summary.longest_win_streak_b, streak_b)

    # First kill rates
    if rounds_with_first_kill > 0:
        summary.team_a_first_kill_rate = team_a_first_kills / rounds_with_first_kill
        summary.team_b_first_kill_rate = team_b_first_kills / rounds_with_first_kill

    return summary


# =============================================================================
# Player summary builder
# =============================================================================


def _build_player_summary(steam_id: str, pdata: dict, timeline: list[dict]) -> PlayerSummary:
    """Build a PlayerSummary from a single player's orchestrator data."""
    stats = pdata.get("stats") or {}
    rating_data = pdata.get("rating") or {}
    utility = pdata.get("utility") or {}
    entry = pdata.get("entry") or {}
    trades = pdata.get("trades") or {}
    clutches = pdata.get("clutches") or {}
    weapon_kills = pdata.get("weapon_kills") or {}

    kills = stats.get("kills", 0)
    deaths = stats.get("deaths", 0)
    name = pdata.get("name", "Unknown")

    # Opening duels
    od_attempts = entry.get("entry_attempts", 0)
    od_wins = entry.get("entry_kills", 0)

    # Trades
    trade_given = trades.get("trade_kill_success", 0)
    untraded = trades.get("untraded_deaths", 0)
    traded_deaths_success = trades.get("traded_death_success", 0)

    # Clutches
    cl_attempts = clutches.get("total_situations", 0)
    cl_wins = clutches.get("clutch_wins", 0)

    # Utility totals
    flashes = utility.get("flashbangs_thrown", 0)
    smokes = utility.get("smokes_thrown", 0)
    he = utility.get("he_thrown", 0)
    molotovs = utility.get("molotovs_thrown", 0)
    util_thrown = flashes + smokes + he + molotovs
    util_dmg = utility.get("he_damage", 0) + utility.get("molotov_damage", 0)

    # AWP kills
    awp_kills = weapon_kills.get("awp", 0) + weapon_kills.get("AWP", 0)

    # Dry peek rate from round timeline kills
    dry_deaths = 0
    total_deaths_peeked = 0
    ct_kills = 0
    t_kills = 0
    for rdata in timeline:
        for kill in rdata.get("kills") or []:
            # Count side-specific kills
            if kill.get("killer", "") == name:
                if kill.get("killer_team") == "CT":
                    ct_kills += 1
                else:
                    t_kills += 1
            # Count dry peek deaths
            if kill.get("victim", "") == name:
                dp = kill.get("was_dry_peek")
                if dp is not None:
                    total_deaths_peeked += 1
                    if dp:
                        dry_deaths += 1

    # Role guess based on stats
    role = _guess_role(
        kills=kills,
        awp_kills=awp_kills,
        od_wins=od_wins,
        od_attempts=od_attempts,
        flash_assists=utility.get("flash_assists", 0),
        cl_wins=cl_wins,
        rating_val=rating_data.get("hltv_rating", 0.0),
    )

    return PlayerSummary(
        name=name,
        steam_id=steam_id,
        team=pdata.get("team", ""),
        role_guess=role,
        kills=kills,
        deaths=deaths,
        assists=stats.get("assists", 0),
        adr=stats.get("adr", 0.0),
        kast_pct=rating_data.get("kast_percentage", 0.0),
        headshot_pct=stats.get("headshot_pct", 0.0),
        rating=rating_data.get("hltv_rating", 0.0),
        opening_duel_attempts=od_attempts,
        opening_duel_wins=od_wins,
        opening_duel_win_rate=(od_wins / od_attempts * 100) if od_attempts > 0 else 0.0,
        clutch_attempts=cl_attempts,
        clutch_wins=cl_wins,
        clutch_win_rate=(cl_wins / cl_attempts * 100) if cl_attempts > 0 else 0.0,
        trades_given=trade_given,
        trades_missed=untraded,
        trade_rate=((traded_deaths_success / deaths * 100) if deaths > 0 else 0.0),
        utility_damage=util_dmg,
        utility_thrown=util_thrown,
        flash_assists=utility.get("flash_assists", 0),
        dry_peek_deaths=dry_deaths,
        total_deaths_with_peek_data=total_deaths_peeked,
        dry_peek_rate=(
            (dry_deaths / total_deaths_peeked * 100) if total_deaths_peeked > 0 else 0.0
        ),
        ct_kills=ct_kills,
        t_kills=t_kills,
        awp_kills=awp_kills,
    )


def _guess_role(
    *,
    kills: int,
    awp_kills: int,
    od_wins: int,
    od_attempts: int,
    flash_assists: int,
    cl_wins: int,
    rating_val: float,
) -> str:
    """Guess player role from stats. Returns one of:
    awper, entry, support, lurker, igl_candidate, unknown
    """
    if kills > 0 and awp_kills / kills > 0.30:
        return "awper"
    if od_attempts >= 5 and od_wins / max(od_attempts, 1) > 0.50:
        return "entry"
    if flash_assists >= 4:
        return "support"
    if cl_wins >= 2 and rating_val < 1.0:
        return "lurker"
    if rating_val < 1.0 and flash_assists >= 2:
        return "igl_candidate"
    return "rifler"


# =============================================================================
# Round summary builder
# =============================================================================


def _build_round_summary(rdata: dict) -> RoundSummary:
    """Build a RoundSummary from a single round_timeline entry."""
    kills = rdata.get("kills") or []
    economy = rdata.get("economy") or {}
    ct_econ = economy.get("ct") or {}
    t_econ = economy.get("t") or {}
    clutches = rdata.get("clutches") or []

    # First kill
    first_kill_player = ""
    first_kill_victim = ""
    first_kill_side = ""
    if kills:
        fk = kills[0]
        first_kill_player = fk.get("killer", "")
        first_kill_victim = fk.get("victim", "")
        first_kill_side = fk.get("killer_team", "")

    # Clutch info
    clutch_player = None
    clutch_situation = None
    clutch_won = False
    if clutches:
        cl = clutches[0]  # take first clutch in round
        clutch_player = cl.get("player")
        clutch_situation = cl.get("scenario")
        clutch_won = cl.get("won", False)

    # Bomb site — check bomb_plant events
    bomb_site = None
    for ev in rdata.get("events") or []:
        if ev.get("type") == "bomb_plant":
            bomb_site = ev.get("site")
            break

    # Utility before first kill
    util_before_entry = 0
    if kills:
        first_kill_tick = kills[0].get("tick", 0)
        for util in rdata.get("utility") or []:
            if util.get("tick", 0) <= first_kill_tick:
                util_before_entry += 1

    return RoundSummary(
        round_number=rdata.get("round_num", 0),
        winner=rdata.get("winner", ""),
        win_method=rdata.get("win_reason", ""),
        ct_buy_type=ct_econ.get("buy_type", ""),
        t_buy_type=t_econ.get("buy_type", ""),
        ct_equipment_value=ct_econ.get("equipment", 0),
        t_equipment_value=t_econ.get("equipment", 0),
        first_kill_player=first_kill_player,
        first_kill_victim=first_kill_victim,
        first_kill_side=first_kill_side,
        clutch_player=clutch_player,
        clutch_situation=clutch_situation,
        clutch_won=clutch_won,
        bomb_site=bomb_site,
        utility_used_before_entry=util_before_entry,
        total_kills=len(kills),
    )


# =============================================================================
# to_llm_prompt()
# =============================================================================


def to_llm_prompt(summary: MatchSummary, focus: str = "coaching") -> str:
    """
    Convert a MatchSummary to a structured text prompt optimized for
    LLM consumption.  Uses XML-style tags for clear section boundaries.

    Args:
        summary: Pre-processed MatchSummary.
        focus: One of "coaching", "scouting", "economy", "antistrat".

    Returns:
        Structured text string ready to embed in an LLM prompt.
    """
    parts: list[str] = []

    # --- Match context ---
    parts.append("<match_context>")
    parts.append(
        f"Map: {summary.map_name} | Score: {summary.final_score} | Winner: {summary.winner}"
    )
    parts.append(
        f"Overtime: {'Yes' if summary.overtime else 'No'} | Rounds: {summary.total_rounds}"
    )
    parts.append(
        f"Pistol rounds: {summary.team_a_name} {summary.team_a_pistol_wins}/2, "
        f"{summary.team_b_name} {summary.team_b_pistol_wins}/2"
    )
    parts.append(
        f"First kill rate: {summary.team_a_name} {summary.team_a_first_kill_rate:.0%}, "
        f"{summary.team_b_name} {summary.team_b_first_kill_rate:.0%}"
    )
    parts.append(
        f"Longest streak: {summary.team_a_name} {summary.longest_win_streak_a}, "
        f"{summary.team_b_name} {summary.longest_win_streak_b}"
    )
    parts.append("</match_context>")
    parts.append("")

    # --- Team sections ---
    for _label, team_name, players in [
        ("team_a", summary.team_a_name, summary.team_a_players),
        ("team_b", summary.team_b_name, summary.team_b_players),
    ]:
        parts.append(f'<team name="{team_name}">')
        for p in players:
            parts.append(
                f'  <player name="{p.name}" role="{p.role_guess}" rating="{p.rating:.2f}">'
            )
            parts.append(
                f"    K/D: {p.kills}/{p.deaths} | ADR: {p.adr:.1f} "
                f"| KAST: {p.kast_pct:.0f}% | HS%: {p.headshot_pct:.0f}%"
            )
            if p.opening_duel_attempts > 0:
                parts.append(
                    f"    Opening duels: {p.opening_duel_attempts} attempted, "
                    f"{p.opening_duel_wins} won ({p.opening_duel_win_rate:.0f}%)"
                )
            if p.deaths > 0:
                parts.append(
                    f"    Trade rate: {p.trade_rate:.0f}% of deaths traded "
                    f"| Trades given: {p.trades_given} | Untraded deaths: {p.trades_missed}"
                )
            if p.clutch_attempts > 0:
                parts.append(
                    f"    Clutches: {p.clutch_wins}/{p.clutch_attempts} ({p.clutch_win_rate:.0f}%)"
                )
            if p.utility_thrown > 0:
                parts.append(
                    f"    Utility: {p.utility_thrown} thrown, "
                    f"{p.flash_assists} flash assists, "
                    f"{p.utility_damage:.0f} util damage"
                )
            if p.total_deaths_with_peek_data > 0:
                parts.append(f"    Dry peek rate: {p.dry_peek_rate:.0f}%")
            if p.awp_kills > 0:
                parts.append(f"    AWP kills: {p.awp_kills}")
            parts.append("  </player>")
        parts.append("</team>")
        parts.append("")

    # --- Round flow (compact) ---
    if focus in ("coaching", "scouting", "antistrat"):
        parts.append("<round_flow>")
        for rs in summary.rounds:
            label_parts = []
            if rs.round_number in (1, 13):
                label_parts.append("PISTOL")
            if rs.first_kill_player:
                label_parts.append(f"FK: {rs.first_kill_player} -> {rs.first_kill_victim}")
            if rs.clutch_player and rs.clutch_won:
                label_parts.append(f"CLUTCH {rs.clutch_situation} by {rs.clutch_player}")
            if rs.bomb_site:
                label_parts.append(f"Site {rs.bomb_site}")

            detail = " | ".join(label_parts) if label_parts else ""
            score_str = f"{rs.score_after_ct}-{rs.score_after_t}"
            buy_str = f"[CT:{rs.ct_buy_type or '?'} T:{rs.t_buy_type or '?'}]"

            parts.append(
                f"  R{rs.round_number}: {rs.winner} wins "
                f"({rs.win_method}) {score_str} {buy_str}" + (f" — {detail}" if detail else "")
            )
        parts.append("</round_flow>")
        parts.append("")

    # --- Economy patterns ---
    if focus in ("coaching", "economy", "antistrat"):
        parts.append("<economy_patterns>")
        if summary.eco_round_total_a > 0:
            parts.append(
                f"  {summary.team_a_name} eco conversion: "
                f"{summary.eco_round_wins_a}/{summary.eco_round_total_a}"
            )
        if summary.eco_round_total_b > 0:
            parts.append(
                f"  {summary.team_b_name} eco conversion: "
                f"{summary.eco_round_wins_b}/{summary.eco_round_total_b}"
            )
        if summary.force_buy_total_a > 0:
            parts.append(
                f"  {summary.team_a_name} force buy success: "
                f"{summary.force_buy_wins_a}/{summary.force_buy_total_a}"
            )
        if summary.force_buy_total_b > 0:
            parts.append(
                f"  {summary.team_b_name} force buy success: "
                f"{summary.force_buy_wins_b}/{summary.force_buy_total_b}"
            )
        if summary.comeback_rounds:
            parts.append(
                f"  Streak-breaking rounds: {', '.join(f'R{r}' for r in summary.comeback_rounds)}"
            )
        parts.append("</economy_patterns>")
        parts.append("")

    # --- Focus-specific instructions ---
    parts.append("<analysis_instructions>")
    if focus == "coaching":
        parts.append(
            "Analyze each player's performance. Identify the #1 improvement "
            "area per player. Prioritize team-level issues (failed trades, "
            "economy mistakes) over individual aim problems. Be specific "
            "with round numbers and stats."
        )
    elif focus == "scouting":
        parts.append(
            "Identify exploitable patterns in this team's play. Focus on: "
            "default setups, execute timings, economy tendencies, and individual "
            "player habits. For each pattern, suggest a counter-strategy."
        )
    elif focus == "economy":
        parts.append(
            "Analyze economy decision-making across the match. Grade each "
            "team's buy decisions. Identify rounds where bad economy choices "
            "directly cost round wins. Quantify the economic impact."
        )
    elif focus == "antistrat":
        parts.append(
            "Generate specific counter-strategies for the patterns observed. "
            "For each tendency, provide: the pattern, the counter, required "
            "utility, and which player should execute the counter."
        )
    parts.append("</analysis_instructions>")

    return "\n".join(parts)
