"""
Anti-Strat Generation - Actionable recommendations based on scouting data.

Analyzes player profiles and team tendencies to generate specific
counter-strategies for match preparation.
"""

from opensight.scouting.models import (
    EconomyTendency,
    MapTendency,
    PlayerScoutProfile,
    PlayStyle,
)


def generate_anti_strats(
    players: list[PlayerScoutProfile],
    map_tendencies: list[MapTendency],
    economy_tendency: EconomyTendency,
) -> list[str]:
    """
    Generate actionable anti-strat recommendations.

    Args:
        players: List of opponent player profiles
        map_tendencies: Team tendencies by map
        economy_tendency: Overall economy approach

    Returns:
        List of anti-strat recommendation strings
    """
    strats: list[str] = []

    # Player-specific anti-strats
    strats.extend(_player_anti_strats(players))

    # Map-specific anti-strats
    strats.extend(_map_anti_strats(map_tendencies))

    # Economy anti-strats
    strats.extend(_economy_anti_strats(economy_tendency, players))

    # AWP player handling
    strats.extend(_awp_anti_strats(players))

    # Entry/opening strategies
    strats.extend(_entry_anti_strats(players))

    return strats


def _player_anti_strats(players: list[PlayerScoutProfile]) -> list[str]:
    """Generate anti-strats for individual players."""
    strats = []

    for player in players:
        # Aggressive player detection
        if player.play_style == PlayStyle.AGGRESSIVE:
            strats.append(
                f"⚔️ **{player.name}** plays AGGRESSIVE (score: {player.aggression_score:.0f}). "
                f"Set up crossfires to punish early pushes. Expect them to peek first."
            )

            if player.entry_success_rate > 0.5:
                strats.append(
                    f"🎯 **{player.name}** wins {player.entry_success_rate * 100:.0f}% of entries. "
                    f"Consider playing retake setups instead of holding direct angles."
                )

        # Passive player detection
        elif player.play_style == PlayStyle.PASSIVE:
            strats.append(
                f"🛡️ **{player.name}** plays PASSIVE. Expect them to anchor sites and hold late. "
                f"Use utility to flush them from common positions."
            )

        # Clutch player detection
        if player.clutch_attempts >= 5 and player.clutch_win_rate > 0.4:
            strats.append(
                f"🏆 **{player.name}** is a clutch specialist ({player.clutch_wins}/{player.clutch_attempts} = "
                f"{player.clutch_win_rate * 100:.0f}%). Trade kills aggressively and don't give 1v1s."
            )

        # Headshot machine
        if player.headshot_rate > 0.55:
            strats.append(
                f"🎯 **{player.name}** has {player.headshot_rate * 100:.0f}% HS rate. "
                f"Expect precise aim - avoid wide swings and use flashes."
            )

        # Inconsistent player
        if player.consistency_score < 40 and player.demos_analyzed >= 2:
            strats.append(
                f"📊 **{player.name}** is INCONSISTENT (score: {player.consistency_score:.0f}). "
                f"Performance varies between matches - they may be streaky."
            )

    return strats


def _map_anti_strats(map_tendencies: list[MapTendency]) -> list[str]:
    """Generate anti-strats based on map tendencies."""
    strats = []

    for mt in map_tendencies:
        map_name = mt.map_name.replace("de_", "").title()

        # T-side timing
        if mt.avg_first_contact_seconds < 25:
            strats.append(
                f"🗺️ On **{map_name}** (T-side): They make early contact at ~{mt.avg_first_contact_seconds:.0f}s. "
                f"Expect aggressive map control and early picks."
            )
        elif mt.avg_first_contact_seconds > 40:
            strats.append(
                f"🗺️ On **{map_name}** (T-side): They play slow with contact at ~{mt.avg_first_contact_seconds:.0f}s. "
                f"Don't overrotate early - wait for confirmed executes."
            )

        # T-side aggression
        if mt.t_side_aggression > 60:
            strats.append(
                f"⚔️ On **{map_name}**: High T-side aggression ({mt.t_side_aggression:.0f}%). "
                f"Set up early crossfires and be ready for rushes."
            )

        # CT-side aggression (pushes)
        if mt.ct_side_aggression > 50:
            strats.append(
                f"🛡️ On **{map_name}**: They push as CT ({mt.ct_side_aggression:.0f}% aggression). "
                f"Clear angles carefully on T-side and punish overaggression."
            )

        # Execute timing
        if mt.avg_execute_time_seconds > 60:
            strats.append(
                f"⏱️ On **{map_name}**: Late executes around {mt.avg_execute_time_seconds:.0f}s. "
                f"Stay patient and don't force rotations early."
            )

    return strats


def _economy_anti_strats(
    economy_tendency: EconomyTendency, players: list[PlayerScoutProfile]
) -> list[str]:
    """Generate anti-strats based on economy patterns."""
    strats = []

    if economy_tendency == EconomyTendency.AGGRESSIVE:
        strats.append(
            "💰 They FORCE BUY often. Expect upgraded pistols and SMGs on eco rounds. "
            "Play safer angles and don't give easy trades."
        )
        strats.append(
            "💰 After winning pistol: They may force second round. Don't assume easy anti-eco."
        )

    elif economy_tendency == EconomyTendency.CONSERVATIVE:
        strats.append(
            "💰 They play PROPER ECOS. Expect full saves on broken economy. "
            "Push for info on eco rounds - they won't have utility."
        )

    return strats


def _awp_anti_strats(players: list[PlayerScoutProfile]) -> list[str]:
    """Generate anti-strats for AWP players."""
    strats = []

    awpers = [p for p in players if p.awp_usage_rate > 0.2]
    if not awpers:
        strats.append(
            "🔫 No dedicated AWPer detected. They may share the AWP or not prioritize it. "
            "Consider aggressive AWP plays on CT side."
        )
        return strats

    primary_awper = max(awpers, key=lambda p: p.awp_usage_rate)

    strats.append(
        f"🔫 **{primary_awper.name}** is their primary AWPer ({primary_awper.awp_usage_rate * 100:.0f}% rounds). "
        f"Gets {primary_awper.awp_kills_per_awp_round:.1f} kills/AWP round."
    )

    if primary_awper.awp_kills_per_awp_round > 1.0:
        strats.append(
            f"⚠️ **{primary_awper.name}** is DANGEROUS with AWP. "
            f"Smoke common angles and use utility before peeking."
        )

    # Multiple AWPers
    if len(awpers) >= 2:
        second_awper = sorted(awpers, key=lambda p: p.awp_usage_rate, reverse=True)[1]
        strats.append(
            f"🔫 Secondary AWP: **{second_awper.name}** ({second_awper.awp_usage_rate * 100:.0f}% rounds). "
            f"Watch for double AWP setups."
        )

    return strats


def _entry_anti_strats(players: list[PlayerScoutProfile]) -> list[str]:
    """Generate anti-strats for entry fraggers."""
    strats = []

    # Find entry players
    entry_players = [p for p in players if p.entry_attempt_rate > 0.15]
    if not entry_players:
        return strats

    primary_entry = max(entry_players, key=lambda p: p.entry_attempt_rate)

    strats.append(
        f"🚪 **{primary_entry.name}** is their entry fragger "
        f"({primary_entry.entry_attempt_rate * 100:.0f}% entry rate, "
        f"{primary_entry.entry_success_rate * 100:.0f}% success). "
    )

    if primary_entry.entry_success_rate > 0.5:
        strats.append(
            f"⚠️ **{primary_entry.name}** wins most entries. "
            f"Consider off-angles or jiggle-peeking to bait shots."
        )
    else:
        strats.append(
            f"✅ **{primary_entry.name}** loses entries often. "
            f"Hold standard angles and trade quickly if they do get a kill."
        )

    # Opening duel analysis
    openers = [p for p in players if p.opening_duel_win_rate > 0.55 and p.entry_attempt_rate > 0.1]
    for opener in openers:
        if opener.name != primary_entry.name:
            strats.append(
                f"🎯 **{opener.name}** wins {opener.opening_duel_win_rate * 100:.0f}% of opening duels. "
                f"Strong aim - avoid dry peeking."
            )

    return strats


def format_anti_strats_markdown(anti_strats: list[str]) -> str:
    """Format anti-strats list as markdown."""
    if not anti_strats:
        return (
            "No specific anti-strats generated. Need more demo data for reliable recommendations."
        )

    sections = {
        "Player Targeting": [],
        "Map Tendencies": [],
        "Economy": [],
        "AWP Handling": [],
        "Entry/Opening": [],
    }

    for strat in anti_strats:
        if "⚔️" in strat or "🛡️" in strat or "🏆" in strat or "📊" in strat or "🎯" in strat[:5]:
            sections["Player Targeting"].append(strat)
        elif "🗺️" in strat or "⏱️" in strat:
            sections["Map Tendencies"].append(strat)
        elif "💰" in strat:
            sections["Economy"].append(strat)
        elif "🔫" in strat:
            sections["AWP Handling"].append(strat)
        elif "🚪" in strat:
            sections["Entry/Opening"].append(strat)
        else:
            sections["Player Targeting"].append(strat)

    output = []
    for section, strats in sections.items():
        if strats:
            output.append(f"### {section}")
            for strat in strats:
                output.append(f"- {strat}")
            output.append("")

    return "\n".join(output)
