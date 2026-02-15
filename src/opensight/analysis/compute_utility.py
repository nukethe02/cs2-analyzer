"""
Utility and mistake computation methods extracted from DemoAnalyzer.

Contains:
- calculate_utility_stats: Flash, HE, molotov, smoke statistics
- calculate_mistakes: Team damage, team kills, teammates flashed
- compute_utility_metrics: Standalone utility metrics (Scope.gg style)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from opensight.core.parser import DemoData

if TYPE_CHECKING:
    from opensight.analysis.analytics import DemoAnalyzer, UtilityMetrics

logger = logging.getLogger(__name__)

# CS2 grenade costs for unused utility calculation (BUG-14)
GRENADE_COSTS = {
    "weapon_flashbang": 200,
    "weapon_smokegrenade": 300,
    "weapon_hegrenade": 300,
    "weapon_molotov": 400,
    "weapon_incgrenade": 600,
    "weapon_decoy": 50,
    "flashbang": 200,
    "smokegrenade": 300,
    "hegrenade": 300,
    "molotov": 400,
    "incgrenade": 600,
    "decoy": 50,
}


def _build_name_to_steamid_map(analyzer: DemoAnalyzer) -> dict[str, int]:
    """Build player_name → steam_id lookup from analyzer._players."""
    return {player.name: sid for sid, player in analyzer._players.items() if player.name}


def _regroup_blinds_by_name(
    blinds: list,
    name_to_sid: dict[str, int],
) -> dict[int, list]:
    """Re-group blind events using attacker NAME when steamid matching failed."""
    blinds_by_attacker: dict[int, list] = {}
    matched = 0
    for blind in blinds:
        sid = name_to_sid.get(blind.attacker_name)
        if sid is not None:
            blinds_by_attacker.setdefault(sid, []).append(blind)
            matched += 1
    logger.info(f"Name-based blind matching: {matched}/{len(blinds)} events matched")
    return blinds_by_attacker


def _regroup_grenades_by_name(
    grenades: list,
    name_to_sid: dict[str, int],
) -> dict[int, list]:
    """Re-group grenade events using player NAME when steamid matching failed."""
    grenades_by_player: dict[int, list] = {}
    matched = 0
    for grenade in grenades:
        sid = name_to_sid.get(grenade.player_name)
        if sid is not None:
            grenades_by_player.setdefault(sid, []).append(grenade)
            matched += 1
    logger.info(f"Name-based grenade matching: {matched}/{len(grenades)} events matched")
    return grenades_by_player


def _count_grenades_for_player(player_grenades: list) -> tuple[int, int, int, int]:
    """Count grenades by type, avoiding double-counting thrown+detonate pairs.

    Returns (flashes, smokes, he_count, molly_count).
    """
    # Prefer "thrown" events; only use "detonate" if no thrown events exist for that type
    thrown = [g for g in player_grenades if g.event_type == "thrown"]
    detonate = [g for g in player_grenades if g.event_type != "thrown"]

    # Count from thrown events first
    flashes, smokes, he_count, molly_count = 0, 0, 0, 0
    has_thrown_type: set[str] = set()

    for g in thrown:
        gt = g.grenade_type.lower()
        if "smoke" in gt:
            smokes += 1
            has_thrown_type.add("smoke")
        elif "hegrenade" in gt or "he_grenade" in gt:
            he_count += 1
            has_thrown_type.add("he")
        elif "molotov" in gt or "incendiary" in gt or "inferno" in gt:
            molly_count += 1
            has_thrown_type.add("molly")
        elif "flash" in gt:
            flashes += 1
            has_thrown_type.add("flash")

    # Supplement with detonate events ONLY for grenade types with no thrown events
    for g in detonate:
        gt = g.grenade_type.lower()
        if "smoke" in gt and "smoke" not in has_thrown_type:
            smokes += 1
        elif ("hegrenade" in gt or "he_grenade" in gt) and "he" not in has_thrown_type:
            he_count += 1
        elif (
            "molotov" in gt or "incendiary" in gt or "inferno" in gt
        ) and "molly" not in has_thrown_type:
            molly_count += 1
        elif "flash" in gt and "flash" not in has_thrown_type:
            flashes += 1

    return flashes, smokes, he_count, molly_count


def calculate_utility_stats(analyzer: DemoAnalyzer) -> None:
    """Calculate comprehensive utility statistics (Leetify-style) using all available data."""

    # Early return if no utility data available
    has_blinds = hasattr(analyzer.data, "blinds") and analyzer.data.blinds
    has_grenades = hasattr(analyzer.data, "grenades") and analyzer.data.grenades
    has_damages = not analyzer.data.damages_df.empty
    has_blinds_df = (
        hasattr(analyzer.data, "blinds_df")
        and analyzer.data.blinds_df is not None
        and not analyzer.data.blinds_df.empty
    )
    has_grenades_df = (
        hasattr(analyzer.data, "grenades_df")
        and analyzer.data.grenades_df is not None
        and not analyzer.data.grenades_df.empty
    )

    # Log utility data availability
    logger.info(
        f"Utility data check: blinds={len(analyzer.data.blinds) if has_blinds else 0}, "
        f"grenades={len(analyzer.data.grenades) if has_grenades else 0}, "
        f"has_damages={has_damages}, "
        f"blinds_df={len(analyzer.data.blinds_df) if has_blinds_df else 0}, "
        f"grenades_df={len(analyzer.data.grenades_df) if has_grenades_df else 0}"
    )

    if (
        not has_blinds
        and not has_grenades
        and not has_damages
        and not has_blinds_df
        and not has_grenades_df
    ):
        logger.warning("No utility data available, skipping utility stats")
        return

    # Set _rounds_played for per-round metrics calculation
    for _steam_id, player in analyzer._players.items():
        player.utility._rounds_played = player.rounds_played

    # Build name→steamid map for fallback matching
    name_to_sid = _build_name_to_steamid_map(analyzer)

    # Constants for validation
    MIN_BLIND_DURATION = 0.0
    MAX_BLIND_DURATION = 10.0  # Max 10 seconds is reasonable
    SIGNIFICANT_BLIND_THRESHOLD = 1.5  # Significant blind threshold (full direct hit)

    # ===========================================
    # Use BLINDS data for accurate flash stats
    # ===========================================
    blinds_by_attacker: dict[int, list] = {}
    if has_blinds:
        logger.info(f"Using {len(analyzer.data.blinds)} blind events for flash stats")

        # Group blinds by attacker steamid
        for blind in analyzer.data.blinds:
            if (
                blind.blind_duration < MIN_BLIND_DURATION
                or blind.blind_duration > MAX_BLIND_DURATION
            ):
                continue
            att_id = blind.attacker_steamid
            blinds_by_attacker.setdefault(att_id, []).append(blind)

        # Check if steamid matching failed (all grouped under steamid 0 or no match)
        matched_any = any(sid in analyzer._players for sid in blinds_by_attacker if sid != 0)
        if not matched_any and blinds_by_attacker:
            logger.warning(
                "Blind steamid matching failed (all attacker_steamid=0 or no match). "
                "Falling back to name-based matching."
            )
            valid_blinds = [
                b
                for b in analyzer.data.blinds
                if MIN_BLIND_DURATION <= b.blind_duration <= MAX_BLIND_DURATION
            ]
            blinds_by_attacker = _regroup_blinds_by_name(valid_blinds, name_to_sid)

        for steam_id, player in analyzer._players.items():
            player_blinds = blinds_by_attacker.get(steam_id, [])
            if not player_blinds:
                continue

            # Separate enemy vs teammate blinds
            enemy_blinds = [b for b in player_blinds if not b.is_teammate]
            team_blinds = [b for b in player_blinds if b.is_teammate]

            # Only count blinds > 1.5 seconds as "significant" (full direct hit)
            significant_enemy_blinds = [
                b for b in enemy_blinds if b.blind_duration >= SIGNIFICANT_BLIND_THRESHOLD
            ]
            # Apply same threshold to teammates (don't shame for 0.1s glances)
            significant_team_blinds = [
                b for b in team_blinds if b.blind_duration >= SIGNIFICANT_BLIND_THRESHOLD
            ]

            player.utility.enemies_flashed = len(significant_enemy_blinds)
            player.utility.teammates_flashed = len(significant_team_blinds)
            player.utility.total_blind_time = sum(b.blind_duration for b in enemy_blinds)

            # Count unique flashbangs (group blinds by tick proximity)
            blind_ticks = sorted({b.tick for b in player_blinds})
            if blind_ticks:
                # Group ticks within 10 ticks as same flash
                flash_count = 1
                prev_tick = blind_ticks[0]
                for tick in blind_ticks[1:]:
                    if tick - prev_tick > 10:
                        flash_count += 1
                    prev_tick = tick
                player.utility.flashbangs_thrown = flash_count

            # Calculate effective_flashes: unique flashbangs with >= 1 significant enemy blind
            sig_blind_ticks = sorted({b.tick for b in significant_enemy_blinds})
            if sig_blind_ticks:
                effective_count = 1
                prev_tick = sig_blind_ticks[0]
                for tick in sig_blind_ticks[1:]:
                    if tick - prev_tick > 10:
                        effective_count += 1
                    prev_tick = tick
                player.utility.effective_flashes = effective_count
            else:
                player.utility.effective_flashes = 0

        # ===========================================
        # Victim-side blind metrics (Leetify "Avg Blind Time")
        # ===========================================
        blinds_by_victim: dict[int, list] = {}
        for blind in analyzer.data.blinds:
            if (
                blind.blind_duration < MIN_BLIND_DURATION
                or blind.blind_duration > MAX_BLIND_DURATION
            ):
                continue
            vic_id = blind.victim_steamid
            blinds_by_victim.setdefault(vic_id, []).append(blind)

        # Check if victim steamid matching also failed
        matched_victims = any(sid in analyzer._players for sid in blinds_by_victim if sid != 0)
        if not matched_victims and blinds_by_victim:
            logger.warning("Victim blind steamid matching failed, using name fallback.")
            victim_name_map = {
                player.name: sid for sid, player in analyzer._players.items() if player.name
            }
            blinds_by_victim = {}
            for blind in analyzer.data.blinds:
                if MIN_BLIND_DURATION <= blind.blind_duration <= MAX_BLIND_DURATION:
                    sid = victim_name_map.get(blind.victim_name)
                    if sid is not None:
                        blinds_by_victim.setdefault(sid, []).append(blind)

        for steam_id, player in analyzer._players.items():
            victim_blinds = blinds_by_victim.get(steam_id, [])
            if not victim_blinds:
                continue

            # Only count blinds from enemies (not self-flashes or teammate flashes)
            enemy_blinds_received = [
                b for b in victim_blinds if b.attacker_steamid != steam_id and not b.is_teammate
            ]

            player.utility.times_blinded = len(enemy_blinds_received)
            player.utility.total_time_blinded = sum(b.blind_duration for b in enemy_blinds_received)

    # ===========================================
    # BLINDS DataFrame fallback (when structured blinds list is empty)
    # ===========================================
    elif has_blinds_df:
        logger.info(
            f"Using blinds_df fallback ({len(analyzer.data.blinds_df)} rows) — "
            "structured blinds list was empty"
        )
        bdf = analyzer.data.blinds_df
        att_col = analyzer._find_col(bdf, ["attacker_steamid", "attacker_steam_id", "attacker"])
        dur_col = analyzer._find_col(bdf, ["blind_duration", "duration"])

        if att_col and dur_col:
            for steam_id, player in analyzer._players.items():
                player_blinds = bdf[pd.to_numeric(bdf[att_col], errors="coerce") == float(steam_id)]
                if player_blinds.empty:
                    continue
                durations = pd.to_numeric(player_blinds[dur_col], errors="coerce").dropna()
                valid = durations[
                    (durations >= MIN_BLIND_DURATION) & (durations <= MAX_BLIND_DURATION)
                ]
                significant = valid[valid >= SIGNIFICANT_BLIND_THRESHOLD]
                player.utility.enemies_flashed = len(significant)
                player.utility.total_blind_time = float(valid.sum())
                # Estimate flash count from unique ticks
                if "tick" in player_blinds.columns:
                    ticks = sorted(player_blinds["tick"].dropna().unique())
                    if ticks:
                        count = 1
                        prev = ticks[0]
                        for t in ticks[1:]:
                            if t - prev > 10:
                                count += 1
                            prev = t
                        player.utility.flashbangs_thrown = count

    # ===========================================
    # Use GRENADES data for accurate grenade counts
    # ===========================================
    if has_grenades:
        logger.info(f"Using {len(analyzer.data.grenades)} grenade events")

        # Group grenades by player steamid
        grenades_by_player: dict[int, list] = {}
        for grenade in analyzer.data.grenades:
            player_id = grenade.player_steamid
            grenades_by_player.setdefault(player_id, []).append(grenade)

        # Check if steamid matching failed (all grouped under steamid 0 or no match)
        matched_any = any(sid in analyzer._players for sid in grenades_by_player if sid != 0)
        if not matched_any and grenades_by_player:
            logger.warning(
                "Grenade steamid matching failed (all player_steamid=0 or no match). "
                "Falling back to name-based matching."
            )
            grenades_by_player = _regroup_grenades_by_name(analyzer.data.grenades, name_to_sid)

        for steam_id, player in analyzer._players.items():
            player_grenades = grenades_by_player.get(steam_id, [])
            if not player_grenades:
                continue

            # Count by type, avoiding thrown+detonate double-counting
            flash_count, smokes, he_count, molly_count = _count_grenades_for_player(player_grenades)

            player.utility.smokes_thrown = smokes
            if he_count > 0:
                player.utility.he_thrown = he_count
            if molly_count > 0:
                player.utility.molotovs_thrown = molly_count
            if flash_count > 0 and player.utility.flashbangs_thrown == 0:
                player.utility.flashbangs_thrown = flash_count

    # ===========================================
    # GRENADES DataFrame fallback (when structured grenades list is empty)
    # ===========================================
    elif has_grenades_df:
        logger.info(
            f"Using grenades_df fallback ({len(analyzer.data.grenades_df)} rows) — "
            "structured grenades list was empty"
        )
        gdf = analyzer.data.grenades_df
        player_col = analyzer._find_col(gdf, ["user_steamid", "player_steamid", "steamid"])
        weapon_col = analyzer._find_col(gdf, ["weapon", "grenade_type", "grenade"])
        event_type_col = "event_type" if "event_type" in gdf.columns else None

        if player_col and weapon_col:
            for steam_id, player in analyzer._players.items():
                player_nades = gdf[
                    pd.to_numeric(gdf[player_col], errors="coerce") == float(steam_id)
                ]
                if player_nades.empty:
                    continue
                # Prefer thrown events to avoid double-counting
                if event_type_col:
                    thrown = player_nades[player_nades[event_type_col] == "thrown"]
                    if not thrown.empty:
                        player_nades = thrown

                weapons = player_nades[weapon_col].str.lower()
                flashes = int(weapons.str.contains("flash", na=False).sum())
                smokes = int(weapons.str.contains("smoke", na=False).sum())
                hes = int(
                    (
                        weapons.str.contains("hegrenade", na=False)
                        | weapons.str.contains("he_grenade", na=False)
                    ).sum()
                )
                mollys = int(
                    (
                        weapons.str.contains("molotov", na=False)
                        | weapons.str.contains("incendiary", na=False)
                        | weapons.str.contains("inferno", na=False)
                        | weapons.str.contains("incgrenade", na=False)
                    ).sum()
                )

                if flashes > 0 and player.utility.flashbangs_thrown == 0:
                    player.utility.flashbangs_thrown = flashes
                if smokes > 0 and player.utility.smokes_thrown == 0:
                    player.utility.smokes_thrown = smokes
                if hes > 0 and player.utility.he_thrown == 0:
                    player.utility.he_thrown = hes
                if mollys > 0 and player.utility.molotovs_thrown == 0:
                    player.utility.molotovs_thrown = mollys

    # ===========================================
    # Use DAMAGES data for HE/Molly damage (fallback and supplement)
    # ===========================================
    damages_df = analyzer.data.damages_df
    if not damages_df.empty:
        logger.debug(f"Damage DF columns: {list(damages_df.columns)}")
        att_col = analyzer._find_col(damages_df, analyzer.ATT_ID_COLS)
        att_side = analyzer._find_col(damages_df, analyzer.ATT_SIDE_COLS)
        vic_side = analyzer._find_col(damages_df, analyzer.VIC_SIDE_COLS)
        weapon_col = analyzer._find_col(damages_df, ["weapon", "weapon_name", "attacker_weapon"])
        dmg_col = analyzer._find_col(damages_df, ["dmg_health", "damage", "dmg", "health_damage"])
        logger.debug(f"Utility damage cols: att={att_col}, weapon={weapon_col}, dmg={dmg_col}")

        if not att_col or not weapon_col or not dmg_col:
            logger.warning(
                f"Missing columns for utility damage calculation: "
                f"att_col={att_col}, weapon_col={weapon_col}, dmg_col={dmg_col}. "
                f"Available columns: {list(damages_df.columns)}"
            )

        if att_col and weapon_col and dmg_col:
            he_weapons = [
                "hegrenade",
                "he_grenade",
                "grenade_he",
                "hegrenade_projectile",
                "weapon_hegrenade",
            ]
            molly_weapons = [
                "molotov",
                "incgrenade",
                "inferno",
                "molotov_projectile",
                "incendiary",
                "weapon_molotov",
                "weapon_incgrenade",
                "incgrenade_projectile",
            ]

            # Log weapon distribution in damage events
            if not damages_df.empty:
                weapon_counts = damages_df[weapon_col].value_counts().head(10)
                logger.debug(f"Top weapons in damages: {weapon_counts.to_dict()}")

            for steam_id, player in analyzer._players.items():
                # Match steamid handling type differences
                # Use string comparison to avoid float64 precision loss on 17-digit IDs
                steam_id_str = str(steam_id)
                att_values = damages_df[att_col].astype(str).str.split(".").str[0]
                player_dmg = damages_df[att_values == steam_id_str]

                # HE damage
                he_dmg = player_dmg[player_dmg[weapon_col].str.lower().isin(he_weapons)]
                if not he_dmg.empty:
                    if att_side and vic_side:
                        enemy_he = he_dmg[he_dmg[att_side] != he_dmg[vic_side]]
                        team_he = he_dmg[he_dmg[att_side] == he_dmg[vic_side]]
                        player.utility.he_damage = int(enemy_he[dmg_col].sum())
                        player.utility.he_team_damage = int(team_he[dmg_col].sum())
                    else:
                        player.utility.he_damage = int(he_dmg[dmg_col].sum())
                    if player.utility.he_thrown == 0:
                        player.utility.he_thrown = max(1, len(he_dmg[dmg_col].unique()))

                # Molotov damage
                molly_dmg = player_dmg[player_dmg[weapon_col].str.lower().isin(molly_weapons)]
                if not molly_dmg.empty:
                    if att_side and vic_side:
                        enemy_molly = molly_dmg[molly_dmg[att_side] != molly_dmg[vic_side]]
                        team_molly = molly_dmg[molly_dmg[att_side] == molly_dmg[vic_side]]
                        player.utility.molotov_damage = int(enemy_molly[dmg_col].sum())
                        player.utility.molotov_team_damage = int(team_molly[dmg_col].sum())
                    else:
                        player.utility.molotov_damage = int(molly_dmg[dmg_col].sum())
                    if player.utility.molotovs_thrown == 0:
                        player.utility.molotovs_thrown = max(
                            1,
                            (len(set(molly_dmg["tick"])) if "tick" in molly_dmg.columns else 1),
                        )

    # ===========================================
    # Flash assists from kills
    # Primary: use flash_assist field from kills
    # Fallback: correlate blinds with kills (within 3 seconds / ~192 ticks)
    # ===========================================
    kills_df = analyzer.data.kills_df
    FLASH_ASSIST_WINDOW_TICKS = 192  # ~3 seconds at 64 tick

    # Try native flash_assist field first (handle demoparser2 column name variants)
    flash_assist_col = analyzer._find_col(
        kills_df, ["flash_assist", "assistedflash", "is_flash_assist"]
    )
    if not kills_df.empty and "assister_steamid" in kills_df.columns and flash_assist_col:
        # Convert assister_steamid to numeric once (demoparser2 returns strings;
        # steam_id keys are int, so direct comparison fails silently)
        assister_numeric = pd.to_numeric(kills_df["assister_steamid"], errors="coerce")
        for steam_id, player in analyzer._players.items():
            flash_assists = kills_df[
                (assister_numeric == float(steam_id)) & (kills_df[flash_assist_col])
            ]
            player.utility.flash_assists = len(flash_assists)

    # Fallback: calculate from blind events and kills correlation
    elif has_blinds and not kills_df.empty and "tick" in kills_df.columns:
        logger.info("Calculating flash assists from blind/kill correlation (name-based matching)")
        # Use name columns instead of steamid to avoid demoparser2 steamid quirk
        att_name_col = analyzer._find_col(kills_df, ["attacker_name"])
        vic_name_col = analyzer._find_col(kills_df, ["victim_name", "user_name"])

        if att_name_col and vic_name_col:
            for steam_id, player in analyzer._players.items():
                player_blinds = blinds_by_attacker.get(steam_id, [])
                if not player_blinds:
                    continue

                flash_assist_count = 0
                player_name = player.name  # Get the thrower's name

                # For each enemy blind, check if a teammate got a kill on that enemy
                for blind in player_blinds:
                    if blind.is_teammate:
                        continue

                    victim_name = blind.victim_name  # Use name instead of steamid
                    blind_tick = blind.tick
                    blind_end_tick = blind_tick + int(blind.blind_duration * 64)

                    # Check if any teammate killed this blinded enemy within window
                    victim_kills = kills_df[
                        (kills_df[vic_name_col] == victim_name)
                        & (kills_df["tick"] >= blind_tick)
                        & (kills_df["tick"] <= blind_end_tick + FLASH_ASSIST_WINDOW_TICKS)
                    ]

                    # Count kills by teammates (not by the flash thrower)
                    for _, kill in victim_kills.iterrows():
                        killer_name = kill.get(att_name_col)
                        if killer_name and killer_name != player_name:
                            flash_assist_count += 1
                            break  # Only count once per blind

                player.utility.flash_assists = flash_assist_count

    # ===========================================
    # FALLBACK: Count grenades from weapon_fire events if still zero
    # This handles cases where grenade_thrown/player_blind events are empty
    # ===========================================
    if hasattr(analyzer.data, "weapon_fires") and analyzer.data.weapon_fires:
        # Check if we need the fallback (any player with 0 total utility)
        needs_fallback = any(
            p.utility.flashbangs_thrown == 0
            and p.utility.smokes_thrown == 0
            and p.utility.he_thrown == 0
            and p.utility.molotovs_thrown == 0
            for p in analyzer._players.values()
        )

        if needs_fallback:
            logger.info(
                f"Using weapon_fire fallback for utility counts "
                f"({len(analyzer.data.weapon_fires)} weapon_fire events)"
            )

            # Grenade weapon names in weapon_fire events
            FLASH_WEAPONS = ["flashbang", "weapon_flashbang"]
            SMOKE_WEAPONS = ["smokegrenade", "weapon_smokegrenade"]
            HE_WEAPONS = ["hegrenade", "weapon_hegrenade"]
            MOLLY_WEAPONS = [
                "molotov",
                "weapon_molotov",
                "incgrenade",
                "weapon_incgrenade",
            ]

            # Count by player
            for steam_id, player in analyzer._players.items():
                player_fires = [
                    f for f in analyzer.data.weapon_fires if f.player_steamid == steam_id
                ]

                flash_count = 0
                smoke_count = 0
                he_count = 0
                molly_count = 0

                for fire in player_fires:
                    weapon = fire.weapon.lower() if fire.weapon else ""
                    if weapon in FLASH_WEAPONS or "flash" in weapon:
                        flash_count += 1
                    elif weapon in SMOKE_WEAPONS or "smoke" in weapon:
                        smoke_count += 1
                    elif weapon in HE_WEAPONS or "hegrenade" in weapon:
                        he_count += 1
                    elif weapon in MOLLY_WEAPONS or "molotov" in weapon or "incendiary" in weapon:
                        molly_count += 1

                # Only update if player has 0 and we found some
                if flash_count > 0 and player.utility.flashbangs_thrown == 0:
                    player.utility.flashbangs_thrown = flash_count
                if smoke_count > 0 and player.utility.smokes_thrown == 0:
                    player.utility.smokes_thrown = smoke_count
                if he_count > 0 and player.utility.he_thrown == 0:
                    player.utility.he_thrown = he_count
                if molly_count > 0 and player.utility.molotovs_thrown == 0:
                    player.utility.molotovs_thrown = molly_count

    # ---- Unused utility value ----
    # NOTE: demoparser2 player_death events do NOT include victim inventory/equipment.
    # KillEvent has no victim_equipment/victim_inventory fields, so we cannot determine
    # what grenades a player was holding when they died. This metric requires parsing
    # entity props (m_hMyWeapons) at the death tick, which is not yet implemented.
    # Set to None (unknown) rather than 0 (falsely claims no waste).
    for _steam_id, player in analyzer._players.items():
        player.utility.unused_utility_value = None

    # Log final utility stats summary
    total_flashes = sum(p.utility.flashbangs_thrown for p in analyzer._players.values())
    total_smokes = sum(p.utility.smokes_thrown for p in analyzer._players.values())
    total_he = sum(p.utility.he_thrown for p in analyzer._players.values())
    total_molly = sum(p.utility.molotovs_thrown for p in analyzer._players.values())
    logger.info(
        f"Utility stats complete: {total_flashes} flashes, {total_smokes} smokes, "
        f"{total_he} HE, {total_molly} molotovs across all players"
    )


def calculate_mistakes(analyzer: DemoAnalyzer) -> None:
    """Calculate mistakes (Scope.gg style)."""
    kills_df = analyzer.data.kills_df
    damages_df = analyzer.data.damages_df

    # Team kills (friendly fire deaths)
    if (
        not kills_df.empty
        and analyzer._att_id_col
        and analyzer._vic_id_col
        and analyzer._att_side_col
        and analyzer._vic_side_col
    ):
        for steam_id, player in analyzer._players.items():
            # Check for team kills (attacker and victim same team)
            team_kills = kills_df[
                (kills_df[analyzer._att_id_col] == steam_id)
                & (kills_df[analyzer._att_side_col] == kills_df[analyzer._vic_side_col])
            ]
            player.mistakes.team_kills = len(team_kills)

    # Team damage
    if not damages_df.empty:
        att_col = analyzer._find_col(damages_df, analyzer.ATT_ID_COLS)
        att_side = analyzer._find_col(damages_df, analyzer.ATT_SIDE_COLS)
        vic_side = analyzer._find_col(damages_df, analyzer.VIC_SIDE_COLS)
        dmg_col = analyzer._find_col(damages_df, ["dmg_health", "damage", "dmg"])

        if att_col and att_side and vic_side and dmg_col:
            for steam_id, player in analyzer._players.items():
                team_dmg = damages_df[
                    (damages_df[att_col] == steam_id)
                    & (damages_df[att_side] == damages_df[vic_side])
                ]
                player.mistakes.team_damage = int(team_dmg[dmg_col].sum())

                # Teammates flashed (from utility stats)
                player.mistakes.teammates_flashed = player.utility.teammates_flashed

    logger.info("Calculated mistakes")


def compute_utility_metrics(match_data: DemoData) -> dict[str, UtilityMetrics]:
    """
    Compute utility usage metrics for all players from match data.

    This function provides a standalone way to extract utility statistics
    similar to Scope.gg's nade stats, using awpy's grenade, smoke, inferno,
    and blind data.

    Args:
        match_data: Parsed demo data (MatchData/DemoData) from DemoParser

    Returns:
        Dictionary mapping steam_id (as string) to UtilityMetrics for each player

    Example:
        >>> from opensight.core.parser import parse_demo
        >>> from opensight.analysis.compute_utility import compute_utility_metrics
        >>> data = parse_demo("match.dem")
        >>> utility_stats = compute_utility_metrics(data)
        >>> for steam_id, metrics in utility_stats.items():
        ...     print(f"{metrics.player_name}: {metrics.total_utility_thrown} grenades")
    """
    from opensight.analysis.analytics import UtilityMetrics as _UtilityMetrics

    result: dict[str, _UtilityMetrics] = {}

    # Initialize metrics for all known players
    for steam_id, name in match_data.player_names.items():
        # Use persistent team display name to correctly group teammates across halftime
        persistent_team = match_data.get_player_persistent_team(steam_id)
        team = match_data.get_team_display_name(persistent_team)
        if team == "Unknown":
            # Fallback for backward compatibility with old data
            team = match_data.player_teams.get(steam_id, "Unknown")
        result[str(steam_id)] = _UtilityMetrics(
            player_name=name,
            player_steamid=steam_id,
            team=team,
        )

    # ===========================================
    # Count grenades from grenades list
    # ===========================================
    if hasattr(match_data, "grenades") and match_data.grenades:
        for grenade in match_data.grenades:
            steam_id = str(grenade.player_steamid)
            if steam_id not in result:
                # Player not in player_names, add them
                result[steam_id] = _UtilityMetrics(
                    player_name=grenade.player_name,
                    player_steamid=grenade.player_steamid,
                    team=grenade.player_side,
                )

            grenade_type = grenade.grenade_type.lower()

            # Count by grenade type (awpy uses grenade_type field)
            if "smoke" in grenade_type:
                result[steam_id].smokes_thrown += 1
            elif "flash" in grenade_type:
                result[steam_id].flashes_thrown += 1
            elif "hegrenade" in grenade_type or "he_grenade" in grenade_type:
                result[steam_id].he_thrown += 1
            elif (
                "molotov" in grenade_type
                or "incgrenade" in grenade_type
                or "incendiary" in grenade_type
            ):
                result[steam_id].molotovs_thrown += 1

    # ===========================================
    # Count smokes from smokes list (more accurate count)
    # ===========================================
    if hasattr(match_data, "smokes") and match_data.smokes:
        # Reset smoke counts and use smoke events for more accurate tracking
        for steam_id in result:
            result[steam_id].smokes_thrown = 0

        for smoke in match_data.smokes:
            steam_id = str(smoke.thrower_steamid)
            if steam_id in result:
                result[steam_id].smokes_thrown += 1

    # ===========================================
    # Count molotovs from infernos list (more accurate count)
    # ===========================================
    if hasattr(match_data, "infernos") and match_data.infernos:
        # Reset molotov counts and use inferno events for more accurate tracking
        for steam_id in result:
            result[steam_id].molotovs_thrown = 0

        for inferno in match_data.infernos:
            steam_id = str(inferno.thrower_steamid)
            if steam_id in result:
                result[steam_id].molotovs_thrown += 1

    # ===========================================
    # Process blind events for flash effectiveness
    # ===========================================
    if hasattr(match_data, "blinds") and match_data.blinds:
        for blind in match_data.blinds:
            steam_id = str(blind.attacker_steamid)
            if steam_id not in result:
                continue

            # Only count significant blinds (>1.5 seconds for full direct hit)
            if blind.blind_duration >= 1.5:
                if blind.is_teammate:
                    result[steam_id].flashes_teammates_total += 1
                else:
                    result[steam_id].flashes_enemies_total += 1

            # Accumulate total blind time for enemies only
            if not blind.is_teammate:
                result[steam_id].total_blind_time += blind.blind_duration

    # ===========================================
    # Calculate utility damage from damages DataFrame
    # ===========================================
    damages_df = match_data.damages_df
    if not damages_df.empty:
        # Find column names (different parsers use different names)
        att_col = None
        for col in ["attacker_steamid", "attacker", "att_steamid"]:
            if col in damages_df.columns:
                att_col = col
                break

        weapon_col = "weapon" if "weapon" in damages_df.columns else None
        dmg_col = None
        for col in ["dmg_health", "damage", "dmg"]:
            if col in damages_df.columns:
                dmg_col = col
                break

        att_side_col = None
        for col in ["attacker_side", "attacker_team"]:
            if col in damages_df.columns:
                att_side_col = col
                break

        vic_side_col = None
        for col in ["victim_side", "victim_team", "user_team"]:
            if col in damages_df.columns:
                vic_side_col = col
                break

        if att_col and weapon_col and dmg_col:
            he_weapons = [
                "hegrenade",
                "he_grenade",
                "grenade_he",
                "hegrenade_projectile",
                "weapon_hegrenade",
            ]
            molly_weapons = [
                "molotov",
                "incgrenade",
                "inferno",
                "molotov_projectile",
                "incendiary",
                "weapon_molotov",
                "weapon_incgrenade",
                "incgrenade_projectile",
            ]

            for steam_id, metrics in result.items():
                steam_id_int = int(steam_id)
                player_dmg = damages_df[damages_df[att_col] == steam_id_int]

                # HE damage
                he_dmg = player_dmg[player_dmg[weapon_col].str.lower().isin(he_weapons)]
                if not he_dmg.empty:
                    if att_side_col and vic_side_col:
                        # Only count enemy damage
                        enemy_he = he_dmg[he_dmg[att_side_col] != he_dmg[vic_side_col]]
                        metrics.he_damage = int(enemy_he[dmg_col].sum())
                    else:
                        metrics.he_damage = int(he_dmg[dmg_col].sum())

                # Molotov damage
                molly_dmg = player_dmg[player_dmg[weapon_col].str.lower().isin(molly_weapons)]
                if not molly_dmg.empty:
                    if att_side_col and vic_side_col:
                        # Only count enemy damage
                        enemy_molly = molly_dmg[molly_dmg[att_side_col] != molly_dmg[vic_side_col]]
                        metrics.molotov_damage = int(enemy_molly[dmg_col].sum())
                    else:
                        metrics.molotov_damage = int(molly_dmg[dmg_col].sum())

                # Total utility damage
                metrics.total_utility_damage = float(metrics.he_damage + metrics.molotov_damage)

    # ===========================================
    # Count flash assists from kills DataFrame
    # ===========================================
    kills_df = match_data.kills_df
    # Handle demoparser2 column name variants for flash assists
    fa_col = None
    for _col in ["flash_assist", "assistedflash", "is_flash_assist"]:
        if _col in kills_df.columns:
            fa_col = _col
            break
    if not kills_df.empty and "assister_steamid" in kills_df.columns and fa_col:
        # Convert assister_steamid to numeric once (string vs int mismatch)
        assister_numeric = pd.to_numeric(kills_df["assister_steamid"], errors="coerce")
        for steam_id, metrics in result.items():
            flash_assists = kills_df[(assister_numeric == float(steam_id)) & (kills_df[fa_col])]
            metrics.flash_assists = len(flash_assists)

    logger.info(f"Computed utility metrics for {len(result)} players")
    return result
