"""
Extracted combat computation functions from DemoAnalyzer.

These functions implement combat-related analytics:
- Multi-kill detection
- Opening duels and engagements
- Entry frags (zone-aware)
- Trade kill detection
- Clutch detection
- Greedy re-peek detection

Each public function takes a DemoAnalyzer instance as its first parameter
and is called via delegation from the corresponding DemoAnalyzer method.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from opensight.analysis.analytics import ClutchEvent
from opensight.core.constants import TRADE_WINDOW_SECONDS
from opensight.core.parser import safe_float, safe_int, safe_str

if TYPE_CHECKING:
    from opensight.analysis.analytics import DemoAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers (only used by the combat functions below)
# ---------------------------------------------------------------------------


def _is_sniper_weapon(weapon: str | None) -> bool:
    """Check if weapon is a sniper rifle (AWP, Scout, Autos).

    Snipers legitimately hold angles without utility support,
    so they should be excluded from dry peek tracking.
    """
    if not weapon:
        return False
    weapon_lower = weapon.lower()
    sniper_weapons = {"awp", "ssg08", "g3sg1", "scar20", "weapon_awp", "weapon_ssg08"}
    return any(sniper in weapon_lower for sniper in sniper_weapons)


def _is_utility_supported(
    analyzer: DemoAnalyzer,
    kill_tick: int,
    kill_x: float | None,
    kill_y: float | None,
    kill_z: float | None,
    player_team: str,
    round_num: int,
) -> bool:
    """Check if an engagement was supported by teammate utility (flash/smoke).

    An engagement is considered "supported" if a teammate's flash or smoke
    detonated within 3 seconds prior AND within 2000 game units of the kill position.

    This detects "dry peeks" - entry plays taken without utility support.
    """
    if kill_x is None or kill_y is None:
        return False

    if not hasattr(analyzer.data, "grenades") or not analyzer.data.grenades:
        return False

    SUPPORT_WINDOW_TICKS = int(3.0 * analyzer.TICK_RATE)
    SUPPORT_DISTANCE = 2000.0

    for grenade in analyzer.data.grenades:
        grenade_type = grenade.grenade_type.lower()
        if "flash" not in grenade_type and "smoke" not in grenade_type:
            continue

        if grenade.event_type != "detonate":
            continue

        if grenade.round_num != round_num:
            continue

        grenade_team = analyzer._normalize_team(grenade.player_side)
        if grenade_team != player_team:
            continue

        if grenade.x is None or grenade.y is None:
            continue

        tick_diff = kill_tick - grenade.tick
        if tick_diff < 0 or tick_diff > SUPPORT_WINDOW_TICKS:
            continue

        dx = kill_x - grenade.x
        dy = kill_y - grenade.y
        dz = (kill_z - grenade.z) if (kill_z and grenade.z) else 0
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance <= SUPPORT_DISTANCE:
            return True

    return False


def _get_player_team_for_engagement(analyzer: DemoAnalyzer, steam_id: int) -> str | None:
    """Get player's team for engagement tracking."""
    if steam_id in analyzer._players:
        player = analyzer._players[steam_id]
        if player.team in ("CT", "T"):
            return player.team
    return None


# ---------------------------------------------------------------------------
# Public combat functions (delegated from DemoAnalyzer)
# ---------------------------------------------------------------------------


def calculate_multi_kills(analyzer: DemoAnalyzer) -> None:
    """Calculate multi-kill rounds for each player.

    Counts enemy kills only (excludes teamkills).
    Each round is assigned to exactly one category (1K, 2K, 3K, 4K, or 5K).
    """
    kills_df = analyzer.data.kills_df
    if kills_df.empty or not analyzer._round_col or not analyzer._att_id_col:
        logger.info("Skipping multi-kill calculation - missing columns")
        return

    # Filter out teamkills if we have side columns
    valid_kills = kills_df
    if analyzer._att_side_col and analyzer._vic_side_col:
        if (
            analyzer._att_side_col in kills_df.columns
            and analyzer._vic_side_col in kills_df.columns
        ):
            # Only count enemy kills (attacker_side != victim_side)
            valid_kills = kills_df[
                kills_df[analyzer._att_side_col] != kills_df[analyzer._vic_side_col]
            ]
            teamkills_filtered = len(kills_df) - len(valid_kills)
            if teamkills_filtered > 0:
                logger.debug(f"Multi-kill calc: filtered {teamkills_filtered} teamkills")

    for steam_id, player in analyzer._players.items():
        player_kills = valid_kills[
            valid_kills[analyzer._att_id_col].astype(float) == float(steam_id)
        ]
        if player_kills.empty:
            continue
        kills_per_round = player_kills.groupby(analyzer._round_col).size()

        player.multi_kills.rounds_with_1k = int((kills_per_round == 1).sum())
        player.multi_kills.rounds_with_2k = int((kills_per_round == 2).sum())
        player.multi_kills.rounds_with_3k = int((kills_per_round == 3).sum())
        player.multi_kills.rounds_with_4k = int((kills_per_round == 4).sum())
        player.multi_kills.rounds_with_5k = int((kills_per_round >= 5).sum())


def detect_opening_duels(analyzer: DemoAnalyzer) -> None:
    """Detect opening duels (first kill of each round) with Entry TTD and Dry Peek tracking.

    Entry duels are the first kills of each round. This method:
    1. Identifies the first kill of each round
    2. Calculates Entry TTD (time from first damage to kill for entry frags)
    3. Tracks T-side vs CT-side entries for context
    4. Detects "dry peeks" - entries without teammate utility support
    """
    kills_df = analyzer.data.kills_df
    damages_df = analyzer.data.damages_df
    if (
        kills_df.empty
        or not analyzer._round_col
        or not analyzer._att_id_col
        or not analyzer._vic_id_col
    ):
        logger.info("Skipping opening duels - missing columns")
        return

    # Find damage columns for Entry TTD calculation
    dmg_att_col = (
        analyzer._find_col(damages_df, analyzer.ATT_ID_COLS) if not damages_df.empty else None
    )
    dmg_vic_col = (
        analyzer._find_col(damages_df, analyzer.VIC_ID_COLS) if not damages_df.empty else None
    )

    # Find position columns for dry peek detection
    att_x_col = analyzer._find_col(kills_df, ["attacker_X", "attacker_x", "X", "x"])
    att_y_col = analyzer._find_col(kills_df, ["attacker_Y", "attacker_y", "Y", "y"])
    att_z_col = analyzer._find_col(kills_df, ["attacker_Z", "attacker_z", "Z", "z"])
    vic_x_col = analyzer._find_col(kills_df, ["user_X", "victim_X", "user_x", "victim_x"])
    vic_y_col = analyzer._find_col(kills_df, ["user_Y", "victim_Y", "user_y", "victim_y"])
    vic_z_col = analyzer._find_col(kills_df, ["user_Z", "victim_Z", "user_z", "victim_z"])

    # Find weapon column for sniper detection
    weapon_col = analyzer._find_col(kills_df, ["weapon", "weapon_name", "attacker_weapon"])

    entry_kills_count = 0
    dry_peek_entries = 0
    dry_peek_deaths = 0

    # Get first kill of each round
    for round_num in kills_df[analyzer._round_col].unique():
        round_num_int = safe_int(round_num)
        round_kills = kills_df[kills_df[analyzer._round_col] == round_num].sort_values("tick")
        if round_kills.empty:
            continue

        first_kill = round_kills.iloc[0]
        attacker_id = safe_int(first_kill.get(analyzer._att_id_col))
        victim_id = safe_int(first_kill.get(analyzer._vic_id_col))
        kill_tick = safe_int(first_kill.get("tick"))

        # Get attacker side for T/CT classification using normalized team values
        attacker_side = ""
        victim_side = ""
        if analyzer._att_side_col and analyzer._att_side_col in kills_df.columns:
            attacker_side = analyzer._normalize_team(first_kill.get(analyzer._att_side_col))
            if attacker_side == "Unknown":
                attacker_side = ""
        if analyzer._vic_side_col and analyzer._vic_side_col in kills_df.columns:
            victim_side = analyzer._normalize_team(first_kill.get(analyzer._vic_side_col))
            if victim_side == "Unknown":
                victim_side = ""

        # Get position data for dry peek detection
        att_x = safe_float(first_kill.get(att_x_col)) if att_x_col else None
        att_y = safe_float(first_kill.get(att_y_col)) if att_y_col else None
        att_z = safe_float(first_kill.get(att_z_col)) if att_z_col else None
        vic_x = safe_float(first_kill.get(vic_x_col)) if vic_x_col else None
        vic_y = safe_float(first_kill.get(vic_y_col)) if vic_y_col else None
        vic_z = safe_float(first_kill.get(vic_z_col)) if vic_z_col else None

        # Get weapon for sniper check
        weapon = safe_str(first_kill.get(weapon_col)) if weapon_col else None
        is_sniper_kill = _is_sniper_weapon(weapon)

        if attacker_id in analyzer._players:
            analyzer._players[attacker_id].opening_duels.attempts += 1
            analyzer._players[attacker_id].opening_duels.wins += 1
            entry_kills_count += 1

            # Track T-side vs CT-side entries
            if attacker_side == "T":
                analyzer._players[attacker_id].opening_duels.t_side_entries += 1
            elif attacker_side == "CT":
                analyzer._players[attacker_id].opening_duels.ct_side_entries += 1

            # Dry peek detection for attacker (the one who got the kill)
            # Skip sniper weapons - they legitimately hold angles without utility
            if not is_sniper_kill and attacker_side:
                is_supported = _is_utility_supported(
                    analyzer, kill_tick, att_x, att_y, att_z, attacker_side, round_num_int
                )
                if is_supported:
                    analyzer._players[attacker_id].opening_duels.supported_entries += 1
                else:
                    analyzer._players[attacker_id].opening_duels.unsupported_entries += 1
                    dry_peek_entries += 1

            # Calculate Entry TTD (time from first damage to kill)
            if dmg_att_col and dmg_vic_col and not damages_df.empty:
                entry_damages = damages_df[
                    (damages_df[dmg_att_col].astype(float) == float(attacker_id))
                    & (damages_df[dmg_vic_col].astype(float) == float(victim_id))
                    & (damages_df["tick"] <= kill_tick)
                ].sort_values(by="tick")

                if not entry_damages.empty:
                    first_dmg_tick = safe_int(entry_damages.iloc[0]["tick"])
                    entry_ttd_ticks = kill_tick - first_dmg_tick
                    entry_ttd_ms = entry_ttd_ticks * analyzer.MS_PER_TICK

                    # Only record reasonable TTD values (0-1500ms)
                    if 0 < entry_ttd_ms <= analyzer.TTD_MAX_MS:
                        analyzer._players[attacker_id].opening_duels.entry_ttd_values.append(
                            entry_ttd_ms
                        )

        if victim_id in analyzer._players:
            analyzer._players[victim_id].opening_duels.attempts += 1
            analyzer._players[victim_id].opening_duels.losses += 1

            # Dry peek detection for victim (the one who died)
            if victim_side:
                is_supported = _is_utility_supported(
                    analyzer, kill_tick, vic_x, vic_y, vic_z, victim_side, round_num_int
                )
                if is_supported:
                    analyzer._players[victim_id].opening_duels.supported_deaths += 1
                else:
                    analyzer._players[victim_id].opening_duels.unsupported_deaths += 1
                    dry_peek_deaths += 1

    logger.info(
        f"Detected {entry_kills_count} entry kills across "
        f"{len(kills_df[analyzer._round_col].unique())} rounds, "
        f"dry peek entries: {dry_peek_entries}, dry peek deaths: {dry_peek_deaths}"
    )


def detect_opening_engagements(analyzer: DemoAnalyzer) -> None:
    """Detect opening engagements - who FOUGHT first, not just who DIED first.

    Identifies:
    1. First damage tick of each round
    2. All players who dealt/took damage before first kill
    3. Opening phase damage totals

    This captures true engagement participation even when a player
    initiates combat but doesn't secure the kill.
    """
    kills_df = analyzer.data.kills_df
    damages_df = analyzer.data.damages_df

    if kills_df.empty or damages_df.empty:
        logger.info("Skipping opening engagements - missing kill or damage data")
        return

    if not analyzer._round_col or not analyzer._att_id_col:
        logger.info("Skipping opening engagements - missing columns")
        return

    # Find damage columns
    dmg_att_col = analyzer._find_col(damages_df, analyzer.ATT_ID_COLS)
    dmg_vic_col = analyzer._find_col(damages_df, analyzer.VIC_ID_COLS)
    dmg_round_col = analyzer._find_col(damages_df, analyzer.ROUND_COLS)
    dmg_val_col = analyzer._find_col(damages_df, ["dmg_health", "damage", "dmg"])

    if not all([dmg_att_col, dmg_vic_col, dmg_round_col, dmg_val_col]):
        logger.info("Skipping opening engagements - missing damage columns")
        return

    engagement_count = 0

    for round_num in kills_df[analyzer._round_col].unique():
        # Get first kill tick for this round
        round_kills = kills_df[kills_df[analyzer._round_col] == round_num].sort_values("tick")
        if round_kills.empty:
            continue

        first_kill_tick = safe_int(round_kills.iloc[0]["tick"])
        first_kill_attacker = safe_int(round_kills.iloc[0].get(analyzer._att_id_col))

        # Get all damage events before (and including) first kill
        round_damages = damages_df[
            (damages_df[dmg_round_col] == round_num) & (damages_df["tick"] <= first_kill_tick)
        ].sort_values("tick")

        if round_damages.empty:
            continue

        # Find first damage tick and who dealt it
        first_damage_attacker = safe_int(round_damages.iloc[0].get(dmg_att_col))
        first_damage_victim = safe_int(round_damages.iloc[0].get(dmg_vic_col))

        # Track all players involved in opening phase damage
        opening_phase_damage: dict[int, int] = {}  # steam_id -> damage dealt
        players_took_damage: set[int] = set()

        for _, dmg_row in round_damages.iterrows():
            attacker_id = safe_int(dmg_row.get(dmg_att_col))
            victim_id = safe_int(dmg_row.get(dmg_vic_col))
            damage = safe_int(dmg_row.get(dmg_val_col))

            if attacker_id:
                opening_phase_damage[attacker_id] = (
                    opening_phase_damage.get(attacker_id, 0) + damage
                )
            if victim_id:
                players_took_damage.add(victim_id)

        # Determine winner team (team of the player who got first kill)
        winner_team = _get_player_team_for_engagement(analyzer, first_kill_attacker)

        # Update stats for all involved players
        all_participants = set(opening_phase_damage.keys()) | players_took_damage
        for steam_id in all_participants:
            if steam_id not in analyzer._players:
                continue

            player = analyzer._players[steam_id]
            player.opening_engagements.engagement_attempts += 1

            # First damage tracking
            if steam_id == first_damage_attacker:
                player.opening_engagements.first_damage_dealt += 1
            if steam_id == first_damage_victim:
                player.opening_engagements.first_damage_taken += 1

            # Damage accumulation
            if steam_id in opening_phase_damage:
                dmg = opening_phase_damage[steam_id]
                player.opening_engagements.opening_damage_total += dmg
                player.opening_engagements.opening_damage_values.append(dmg)

            # Win/loss tracking based on team
            player_team = _get_player_team_for_engagement(analyzer, steam_id)
            if player_team and winner_team and player_team == winner_team:
                player.opening_engagements.engagement_wins += 1
            else:
                player.opening_engagements.engagement_losses += 1

            engagement_count += 1

    logger.info(f"Detected {engagement_count} opening engagement participations")


def detect_entry_frags(analyzer: DemoAnalyzer) -> None:
    """Detect zone-aware entry frags using position data.

    Entry Frag = First kill in a specific bombsite for a round.
    Uses get_zone_for_position() to classify kill locations.

    This distinguishes:
    - Map control kills: First kills in mid/connectors/routes
    - Entry frags: First kills inside bombsite zones
    """
    from opensight.visualization.radar import MAP_ZONES, get_zone_for_position

    kills_df = analyzer.data.kills_df
    if kills_df.empty:
        logger.info("Skipping entry frag detection - no kills")
        return

    if not analyzer._round_col or not analyzer._att_id_col or not analyzer._vic_id_col:
        logger.info("Skipping entry frag detection - missing columns")
        return

    # Check if we have position data (handle both uppercase and lowercase column names)
    pos_col_variants = {
        "attacker_x": ["attacker_x", "attacker_X"],
        "attacker_y": ["attacker_y", "attacker_Y"],
        "victim_x": ["victim_x", "victim_X", "user_x", "user_X"],
        "victim_y": ["victim_y", "victim_Y", "user_y", "user_Y"],
    }
    pos_cols = {}
    for key, variants in pos_col_variants.items():
        for v in variants:
            if v in kills_df.columns:
                pos_cols[key] = v
                break
    has_positions = len(pos_cols) == 4
    if not has_positions:
        logger.info(
            f"Skipping entry frag detection - no position data (found: {list(pos_cols.keys())})"
        )
        return

    map_name = analyzer.data.map_name.lower() if analyzer.data.map_name else ""
    if map_name not in MAP_ZONES:
        logger.info(f"Skipping entry frag detection - unknown map: {map_name}")
        return

    zones = MAP_ZONES[map_name]
    bombsite_zones = {name for name, data in zones.items() if data.get("type") == "bombsite"}

    entry_frag_count = 0
    map_control_count = 0

    for round_num in kills_df[analyzer._round_col].unique():
        round_kills = kills_df[kills_df[analyzer._round_col] == round_num].sort_values("tick")
        if round_kills.empty:
            continue

        # Track first kill in each bombsite for this round
        bombsite_first_kills: dict[str, bool] = dict.fromkeys(bombsite_zones, False)
        is_first_kill_of_round = True

        for _, kill in round_kills.iterrows():
            victim_x = safe_float(kill.get(pos_cols.get("victim_x", "victim_x")))
            victim_y = safe_float(kill.get(pos_cols.get("victim_y", "victim_y")))
            # victim_z may not be in pos_cols - try variants
            victim_z = safe_float(
                kill.get("victim_z")
                or kill.get("victim_Z")
                or kill.get("user_z")
                or kill.get("user_Z"),
                default=0.0,
            )

            if victim_x == 0.0 and victim_y == 0.0:
                is_first_kill_of_round = False
                continue

            # Determine zone of kill
            zone = get_zone_for_position(map_name, victim_x, victim_y, victim_z)

            attacker_id = safe_int(kill.get(analyzer._att_id_col))
            victim_id = safe_int(kill.get(analyzer._vic_id_col))

            # Check if this is a bombsite zone
            is_bombsite = zone in bombsite_zones

            # Update kill zone tracking for attacker
            if attacker_id in analyzer._players:
                analyzer._players[attacker_id].opening_duels.kill_zones[zone] = (
                    analyzer._players[attacker_id].opening_duels.kill_zones.get(zone, 0) + 1
                )

            # Entry frag = first kill in this specific bombsite this round
            if is_bombsite and not bombsite_first_kills.get(zone, True):
                bombsite_first_kills[zone] = True

                # Determine if A or B site
                is_a_site = "A" in zone.upper() or "SITE A" in zone.upper()

                if attacker_id in analyzer._players:
                    player = analyzer._players[attacker_id]
                    player.entry_frags.total_entry_frags += 1
                    if is_a_site:
                        player.entry_frags.a_site_entries += 1
                    else:
                        player.entry_frags.b_site_entries += 1
                    player.opening_duels.site_kills += 1
                    entry_frag_count += 1

                if victim_id in analyzer._players:
                    player = analyzer._players[victim_id]
                    player.entry_frags.total_entry_deaths += 1
                    if is_a_site:
                        player.entry_frags.a_site_entry_deaths += 1
                    else:
                        player.entry_frags.b_site_entry_deaths += 1

            # First kill of round in non-bombsite = map control
            elif is_first_kill_of_round and not is_bombsite:
                if attacker_id in analyzer._players:
                    analyzer._players[attacker_id].opening_duels.map_control_kills += 1
                    map_control_count += 1

            is_first_kill_of_round = False

    logger.info(
        f"Detected {entry_frag_count} entry frags and {map_control_count} map control kills"
    )


def detect_trades(analyzer: DemoAnalyzer) -> None:
    """Detect trade kills with Leetify-style opportunity/attempt/success tracking.

    Tracks:
    - Trade Kill Opportunities: Teammate died and you were alive
    - Trade Kill Attempts: You damaged/shot at the killer within window
    - Trade Kill Success: You killed the killer within window
    - Traded Death Opportunities: You died and teammates were alive
    - Traded Death Attempts: Teammates damaged your killer within window
    - Traded Death Success: Teammates killed your killer within window
    - Time to Trade: How fast successful trades were completed
    """
    kills_df = analyzer.data.kills_df
    damages_df = analyzer.data.damages_df
    if kills_df.empty or not analyzer._round_col:
        logger.info("Skipping trade detection - missing round column")
        return

    if not analyzer._vic_id_col or not analyzer._att_id_col:
        logger.info("Skipping trade detection - missing id columns")
        return

    # Use trade window from constants (typically 5 seconds)
    trade_window_ticks = int(TRADE_WINDOW_SECONDS * analyzer.TICK_RATE)
    logger.info(f"Trade detection: window = {TRADE_WINDOW_SECONDS}s = {trade_window_ticks} ticks")

    # Find damage DataFrame columns for attempt detection
    dmg_att_col = (
        analyzer._find_col(damages_df, analyzer.ATT_ID_COLS) if not damages_df.empty else None
    )
    dmg_vic_col = (
        analyzer._find_col(damages_df, analyzer.VIC_ID_COLS) if not damages_df.empty else None
    )
    dmg_round_col = (
        analyzer._find_col(damages_df, analyzer.ROUND_COLS) if not damages_df.empty else None
    )

    # Build player team lookup for consistent team matching
    player_teams_lookup: dict[int, str] = {}
    for steam_id, player in analyzer._players.items():
        if player.team in ("CT", "T"):
            player_teams_lookup[steam_id] = player.team

    # Also extract from persistent team data if not in players
    if analyzer._att_id_col:
        for _, row in kills_df.drop_duplicates(subset=[analyzer._att_id_col]).iterrows():
            att_id = safe_int(row.get(analyzer._att_id_col))
            if att_id and att_id not in player_teams_lookup:
                persistent_team = analyzer.data.get_player_persistent_team(att_id)
                display_team = analyzer.data.get_team_display_name(persistent_team)
                if display_team in ("CT", "T"):
                    player_teams_lookup[att_id] = display_team

    # Counters for logging
    total_trade_opportunities = 0
    total_trade_attempts = 0
    total_trade_success = 0
    total_traded_death_opportunities = 0
    total_traded_death_attempts = 0
    total_traded_death_success = 0

    for round_num in kills_df[analyzer._round_col].unique():
        round_kills = (
            kills_df[kills_df[analyzer._round_col] == round_num]
            .sort_values(by="tick")
            .reset_index(drop=True)
        )

        if round_kills.empty:
            continue

        # Get round damages for attempt detection
        round_damages = pd.DataFrame()
        if not damages_df.empty and dmg_round_col and dmg_att_col and dmg_vic_col:
            round_damages = damages_df[damages_df[dmg_round_col] == round_num]

        # Build list of all players in this round
        all_players_in_round: set[int] = set()
        for _, kill in round_kills.iterrows():
            att_id = safe_int(kill.get(analyzer._att_id_col))
            vic_id = safe_int(kill.get(analyzer._vic_id_col))
            if att_id:
                all_players_in_round.add(att_id)
            if vic_id:
                all_players_in_round.add(vic_id)

        # Identify entry kill for entry trade tracking
        entry_kill_victim_id = 0
        entry_kill_attacker_id = 0
        if len(round_kills) > 0:
            entry_kill = round_kills.iloc[0]
            entry_kill_victim_id = safe_int(entry_kill.get(analyzer._vic_id_col))
            entry_kill_attacker_id = safe_int(entry_kill.get(analyzer._att_id_col))

        # Track who is dead at each point (cumulative deaths)
        dead_players: set[int] = set()

        # Process each death in order
        for _i, kill in round_kills.iterrows():
            victim_id = safe_int(kill.get(analyzer._vic_id_col))
            killer_id = safe_int(kill.get(analyzer._att_id_col))
            kill_tick = safe_int(kill.get("tick"))
            victim_team = player_teams_lookup.get(victim_id, "")

            if not victim_id or not killer_id or victim_team not in ("CT", "T"):
                if victim_id:
                    dead_players.add(victim_id)
                continue

            # === TRADED DEATH OPPORTUNITIES ===
            # Count teammates who are alive when this player dies
            teammates_alive = [
                pid
                for pid in all_players_in_round
                if pid != victim_id
                and pid not in dead_players
                and player_teams_lookup.get(pid) == victim_team
                and pid in analyzer._players
            ]

            if teammates_alive and victim_id in analyzer._players:
                # There's at least one teammate alive who could trade
                analyzer._players[victim_id].trades.traded_death_opportunities += 1
                total_traded_death_opportunities += 1

                # Check if any teammate attempted or succeeded
                teammate_attempted = False
                teammate_succeeded = False

                for teammate_id in teammates_alive:
                    # Check if teammate damaged the killer within trade window
                    if not round_damages.empty:
                        teammate_damage = round_damages[
                            (round_damages[dmg_att_col].astype(float) == float(teammate_id))
                            & (round_damages[dmg_vic_col].astype(float) == float(killer_id))
                            & (round_damages["tick"] > kill_tick)
                            & (round_damages["tick"] <= kill_tick + trade_window_ticks)
                        ]
                        if not teammate_damage.empty:
                            teammate_attempted = True

                    # Check if teammate killed the killer within trade window
                    teammate_kills = round_kills[
                        (round_kills[analyzer._att_id_col].astype(float) == float(teammate_id))
                        & (round_kills[analyzer._vic_id_col].astype(float) == float(killer_id))
                        & (round_kills["tick"] > kill_tick)
                        & (round_kills["tick"] <= kill_tick + trade_window_ticks)
                    ]
                    if not teammate_kills.empty:
                        teammate_succeeded = True
                        trade_tick = safe_int(teammate_kills.iloc[0].get("tick"))
                        trader_id = safe_int(teammate_kills.iloc[0].get(analyzer._att_id_col))
                        if trader_id in analyzer._players:
                            analyzer._players[trader_id].trades.trade_kill_success += 1
                            analyzer._players[trader_id].trades.kills_traded += 1
                            time_to_trade = trade_tick - kill_tick
                            analyzer._players[trader_id].trades.time_to_trade_ticks.append(
                                time_to_trade
                            )
                            total_trade_success += 1
                            # Entry trade tracking
                            if victim_id in (entry_kill_victim_id, entry_kill_attacker_id):
                                analyzer._players[trader_id].trades.traded_entry_kills += 1
                                if victim_id in analyzer._players:
                                    analyzer._players[victim_id].trades.traded_entry_deaths += 1
                        break

                if teammate_attempted and victim_id in analyzer._players:
                    analyzer._players[victim_id].trades.traded_death_attempts += 1
                    total_traded_death_attempts += 1

                if teammate_succeeded and victim_id in analyzer._players:
                    analyzer._players[victim_id].trades.traded_death_success += 1
                    analyzer._players[victim_id].trades.deaths_traded += 1
                    total_traded_death_success += 1

            # === TRADE KILL OPPORTUNITIES ===
            for teammate_id in teammates_alive:
                if teammate_id in analyzer._players:
                    analyzer._players[teammate_id].trades.trade_kill_opportunities += 1
                    analyzer._players[teammate_id].trades.trade_attempts += 1
                    total_trade_opportunities += 1

                    # Check if this teammate attempted to trade
                    attempted = False
                    if not round_damages.empty:
                        teammate_damage = round_damages[
                            (round_damages[dmg_att_col].astype(float) == float(teammate_id))
                            & (round_damages[dmg_vic_col].astype(float) == float(killer_id))
                            & (round_damages["tick"] > kill_tick)
                            & (round_damages["tick"] <= kill_tick + trade_window_ticks)
                        ]
                        if not teammate_damage.empty:
                            attempted = True

                    teammate_kills = round_kills[
                        (round_kills[analyzer._att_id_col].astype(float) == float(teammate_id))
                        & (round_kills[analyzer._vic_id_col].astype(float) == float(killer_id))
                        & (round_kills["tick"] > kill_tick)
                        & (round_kills["tick"] <= kill_tick + trade_window_ticks)
                    ]
                    if not teammate_kills.empty:
                        attempted = True

                    if attempted:
                        analyzer._players[teammate_id].trades.trade_kill_attempts += 1
                        total_trade_attempts += 1

            # Mark this player as dead
            dead_players.add(victim_id)

    # Log summary
    logger.info(
        f"Trade detection complete: "
        f"opportunities={total_trade_opportunities}, attempts={total_trade_attempts}, "
        f"success={total_trade_success}, "
        f"death_opps={total_traded_death_opportunities}, "
        f"death_attempts={total_traded_death_attempts}, "
        f"death_success={total_traded_death_success}"
    )


def detect_clutches(analyzer: DemoAnalyzer) -> None:
    """Detect clutch situations (1vX where player is last alive) with win tracking.

    Clutch: A round where you were the last player alive on your team
    facing one or more enemies. Uses round winner data to determine success.

    This method tracks:
    - total_situations: Total 1vX clutch attempts
    - total_wins: Clutches won (determined by round outcome)
    - Per-scenario tracking (1v1, 1v2, etc.) with wins and attempts
    - Individual clutch events with round_number, type, and outcome
    """
    kills_df = analyzer.data.kills_df
    if kills_df.empty or not analyzer._round_col or not analyzer._vic_id_col:
        logger.info("Skipping clutch detection - missing required columns")
        return

    # Check if we have team columns - required for proper detection
    use_team_column = bool(analyzer._vic_side_col)
    att_team_col = analyzer._att_side_col

    # Build round winner lookup from rounds data
    round_winners: dict[int, str] = {}
    for round_info in analyzer.data.rounds:
        round_winners[round_info.round_num] = round_info.winner

    total_clutch_situations = 0
    total_clutch_wins = 0

    for round_num in kills_df[analyzer._round_col].unique():
        round_num_int = int(round_num)
        round_kills = kills_df[kills_df[analyzer._round_col] == round_num].sort_values("tick")

        if round_kills.empty:
            continue

        # Get round winner (CT or T)
        round_winner = round_winners.get(round_num_int, "Unknown")

        # Initialize FULL team rosters at round start
        ct_alive: set[int] = set()
        t_alive: set[int] = set()

        # Build full roster using round-aware side detection
        for steam_id in analyzer.data.player_persistent_teams.keys():
            side = analyzer.data.get_player_side_for_round(steam_id, round_num_int)
            if side == "CT":
                ct_alive.add(steam_id)
            elif side == "T":
                t_alive.add(steam_id)

        # Fallback: if player_teams is empty, extract from kill events
        if use_team_column and (not ct_alive or not t_alive):
            for _, kill in round_kills.iterrows():
                # Add attacker to their team
                attacker_id = (
                    safe_int(kill.get(analyzer._att_id_col)) if analyzer._att_id_col else 0
                )
                if attacker_id and att_team_col:
                    att_side = analyzer._normalize_team(kill.get(att_team_col))
                    if att_side == "CT":
                        ct_alive.add(attacker_id)
                    elif att_side == "T":
                        t_alive.add(attacker_id)

                # Add victim to their team
                victim_id = safe_int(kill.get(analyzer._vic_id_col))
                if victim_id:
                    vic_side = analyzer._normalize_team(kill.get(analyzer._vic_side_col))
                    if vic_side == "CT":
                        ct_alive.add(victim_id)
                    elif vic_side == "T":
                        t_alive.add(victim_id)

        # Skip rounds with incomplete team data (still no teams after fallback)
        if not ct_alive or not t_alive:
            continue

        # Track if we've already detected a clutch this round (one per side max)
        clutch_detected: dict[str, bool] = {"CT": False, "T": False}
        clutch_info: dict[str, dict[str, Any]] = {}

        # Second pass: process deaths in tick order to track alive status
        for _, kill in round_kills.iterrows():
            victim_id = safe_int(kill.get(analyzer._vic_id_col))
            if not victim_id:
                continue

            # Get victim team from the kill event (handles side swaps correctly)
            if use_team_column:
                victim_side = analyzer._normalize_team(kill.get(analyzer._vic_side_col))
            else:
                # Fallback: determine from current alive sets
                if victim_id in ct_alive:
                    victim_side = "CT"
                elif victim_id in t_alive:
                    victim_side = "T"
                else:
                    victim_side = "Unknown"

            # Remove victim from alive set
            if victim_side == "CT":
                ct_alive.discard(victim_id)
            elif victim_side == "T":
                t_alive.discard(victim_id)

            # Get tick from kill event for bookmark timestamps
            kill_tick = safe_int(kill.get("tick"), default=0)

            # Check for clutch situation after each death
            # CT clutch: 1 CT alive, 1+ T alive, not already detected
            if len(ct_alive) == 1 and len(t_alive) >= 1 and not clutch_detected["CT"]:
                clutch_detected["CT"] = True
                clutcher_id = next(iter(ct_alive))
                clutch_info["CT"] = {
                    "clutcher_id": clutcher_id,
                    "enemies_at_start": len(t_alive),
                    "clutcher_died": False,
                    "tick_start": kill_tick,
                    "team": "CT",
                }

            # T clutch: 1 T alive, 1+ CT alive, not already detected
            if len(t_alive) == 1 and len(ct_alive) >= 1 and not clutch_detected["T"]:
                clutch_detected["T"] = True
                clutcher_id = next(iter(t_alive))
                clutch_info["T"] = {
                    "clutcher_id": clutcher_id,
                    "enemies_at_start": len(ct_alive),
                    "clutcher_died": False,
                    "tick_start": kill_tick,
                    "team": "T",
                }

            # Check if a clutcher just died
            for _side, info in clutch_info.items():
                if info.get("clutcher_id") == victim_id:
                    info["clutcher_died"] = True

        # Process detected clutches
        for side, info in clutch_info.items():
            clutcher_id = info["clutcher_id"]
            enemies_at_start = info["enemies_at_start"]
            clutcher_died = info["clutcher_died"]

            if clutcher_id not in analyzer._players:
                continue

            player = analyzer._players[clutcher_id]

            # Determine outcome: WON, LOST, or SAVED
            if round_winner == side:
                outcome = "WON"
                clutch_won = True
            elif clutcher_died:
                outcome = "LOST"
                clutch_won = False
            else:
                outcome = "SAVED"
                clutch_won = False

            # Count enemies killed during clutch (by the clutcher)
            enemies_killed = 0
            enemy_side = "T" if side == "CT" else "CT"
            for _, kill in round_kills.iterrows():
                attacker_id = (
                    safe_int(kill.get(analyzer._att_id_col)) if analyzer._att_id_col else 0
                )
                if attacker_id != clutcher_id:
                    continue
                # Get victim team from kill event
                victim_id = safe_int(kill.get(analyzer._vic_id_col))
                if use_team_column:
                    vic_side = analyzer._normalize_team(kill.get(analyzer._vic_side_col))
                else:
                    # Fallback: use round-aware side lookup to handle halftime swaps
                    vic_side = analyzer._get_player_side(victim_id, round_num_int)
                if vic_side == enemy_side:
                    enemies_killed += 1

            # Create clutch type string
            clutch_type = f"1v{enemies_at_start}"

            # Get tick and team data for replay bookmarks
            tick_start = info.get("tick_start", 0)
            clutcher_team = info.get("team", side)

            # Create clutch event with full bookmark data
            clutch_event = ClutchEvent(
                round_number=round_num_int,
                type=clutch_type,
                outcome=outcome,
                enemies_killed=enemies_killed,
                tick_start=tick_start,
                clutcher_steamid=clutcher_id,
                clutcher_team=clutcher_team,
                enemies_at_start=enemies_at_start,
            )
            player.clutches.clutches.append(clutch_event)

            # Update totals
            player.clutches.total_situations += 1
            total_clutch_situations += 1
            if clutch_won:
                player.clutches.total_wins += 1
                total_clutch_wins += 1

            # Update per-scenario stats
            if enemies_at_start == 1:
                player.clutches.v1_attempts += 1
                if clutch_won:
                    player.clutches.v1_wins += 1
            elif enemies_at_start == 2:
                player.clutches.v2_attempts += 1
                if clutch_won:
                    player.clutches.v2_wins += 1
            elif enemies_at_start == 3:
                player.clutches.v3_attempts += 1
                if clutch_won:
                    player.clutches.v3_wins += 1
            elif enemies_at_start == 4:
                player.clutches.v4_attempts += 1
                if clutch_won:
                    player.clutches.v4_wins += 1
            elif enemies_at_start >= 5:
                player.clutches.v5_attempts += 1
                if clutch_won:
                    player.clutches.v5_wins += 1

    logger.info(f"Detected {total_clutch_situations} clutch situations, {total_clutch_wins} won")


def detect_greedy_repeeks(analyzer: DemoAnalyzer) -> None:
    """Detect greedy re-peek deaths (static repeek discipline).

    A greedy re-peek occurs when a player:
    1. Gets a kill
    2. Dies within 3 seconds (192 ticks at 64 tick rate)
    3. Their death position is within 150 units of their kill position

    This indicates the player re-peeked the same angle after getting a kill
    instead of repositioning - a common mistake that gets punished.
    """
    kills_df = analyzer.data.kills_df
    if kills_df.empty:
        logger.debug("No kills data for greedy repeek detection")
        return

    # Check for required columns
    att_id_col = analyzer._att_id_col
    vic_id_col = analyzer._vic_id_col
    tick_col = analyzer._find_col(kills_df, ["tick", "game_tick", "time_tick"])

    # Find position columns
    att_x_col = analyzer._find_col(kills_df, ["attacker_X", "attacker_x", "X", "x"])
    att_y_col = analyzer._find_col(kills_df, ["attacker_Y", "attacker_y", "Y", "y"])
    vic_x_col = analyzer._find_col(kills_df, ["user_X", "victim_X", "user_x", "victim_x"])
    vic_y_col = analyzer._find_col(kills_df, ["user_Y", "victim_Y", "user_y", "victim_y"])

    if not all([att_id_col, vic_id_col, tick_col, att_x_col, att_y_col, vic_x_col, vic_y_col]):
        logger.debug("Missing columns for greedy repeek detection")
        return

    # Constants
    REPEEK_WINDOW_TICKS = 192  # 3 seconds at 64 tick
    STATIC_DISTANCE_THRESHOLD = 150  # units (approx 1.5 steps)

    for steam_id, player in analyzer._players.items():
        greedy_count = 0

        # Get kills by this player (they are attacker)
        player_kills = kills_df[kills_df[att_id_col] == steam_id].copy()

        # Get deaths of this player (they are victim)
        player_deaths = kills_df[kills_df[vic_id_col] == steam_id].copy()

        if player_kills.empty or player_deaths.empty:
            continue

        # Sort by tick
        player_kills = player_kills.sort_values(tick_col)
        player_deaths = player_deaths.sort_values(tick_col)

        # For each kill, check if player died shortly after in similar position
        for _, kill_row in player_kills.iterrows():
            kill_tick = kill_row[tick_col]
            kill_x = kill_row[att_x_col]
            kill_y = kill_row[att_y_col]

            # Skip if position data is missing
            if pd.isna(kill_x) or pd.isna(kill_y):
                continue

            # Find deaths within the time window after this kill
            subsequent_deaths = player_deaths[
                (player_deaths[tick_col] > kill_tick)
                & (player_deaths[tick_col] <= kill_tick + REPEEK_WINDOW_TICKS)
            ]

            for _, death_row in subsequent_deaths.iterrows():
                death_x = death_row[vic_x_col]
                death_y = death_row[vic_y_col]

                # Skip if position data is missing
                if pd.isna(death_x) or pd.isna(death_y):
                    continue

                # Calculate distance between kill position and death position
                distance = np.sqrt((kill_x - death_x) ** 2 + (kill_y - death_y) ** 2)

                # If player was still in roughly the same spot, it's a greedy repeek
                if distance < STATIC_DISTANCE_THRESHOLD:
                    greedy_count += 1
                    break  # Only count one greedy death per kill

        # Update player stats
        player.greedy_repeeks = greedy_count

        # Calculate discipline rating: (safe kills / total kills) * 100
        if player.kills > 0:
            safe_kills = player.kills - greedy_count
            player.discipline_rating = round((safe_kills / player.kills) * 100, 1)
        else:
            player.discipline_rating = 100.0  # No kills = no mistakes possible

    total_greedy = sum(p.greedy_repeeks for p in analyzer._players.values())
    logger.info(f"Detected {total_greedy} greedy re-peek deaths across all players")
