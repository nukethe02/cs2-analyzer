"""
Event Storage Pipeline - Store detailed match events in database.

Extracts events from parsed demo DataFrames and stores them in the database
for persistent access and advanced queries.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from opensight.core.parser import safe_bool, safe_float, safe_int, safe_str
from opensight.infra.database import BombEvent, DamageEvent, GrenadeEvent, Kill

logger = logging.getLogger(__name__)


def _get_col(df: pd.DataFrame, row_idx: int, col_variants: list[str], default: Any = None) -> Any:
    """
    Safely get a column value from a DataFrame row, trying multiple column name variants.

    Args:
        df: DataFrame to read from
        row_idx: Row index
        col_variants: List of possible column names to try (in order)
        default: Default value if column not found or value is NaN

    Returns:
        Column value or default
    """
    for col in col_variants:
        if col in df.columns:
            val = df.at[row_idx, col]
            if pd.notna(val):
                return val
    return default


def store_match_events(match_id: int, data, session: Session) -> dict[str, int]:
    """
    Store detailed match events from parsed demo data into database.

    Args:
        match_id: Database match ID to associate events with
        data: DemoData object from parser with kills_df, damages_df, grenades_df, bomb_events_df
        session: Active database session

    Returns:
        Dictionary with counts of stored events: {"kills": N, "damage_events": N, ...}
    """
    counts = {"kills": 0, "damage_events": 0, "grenades": 0, "bombs": 0}

    # Store kill events
    if hasattr(data, "kills_df") and not data.kills_df.empty:
        kills_df = data.kills_df
        kill_records = []

        for idx in range(len(kills_df)):
            try:
                kill_records.append(
                    {
                        "match_id": match_id,
                        "tick": safe_int(_get_col(kills_df, idx, ["tick"])),
                        "round_num": safe_int(
                            _get_col(kills_df, idx, ["total_rounds_played", "round", "round_num"])
                        ),
                        "attacker_steam_id": safe_str(
                            _get_col(kills_df, idx, ["attacker_steamid", "attacker_steam_id"])
                        ),
                        "attacker_name": safe_str(
                            _get_col(kills_df, idx, ["attacker_name", "attacker"])
                        ),
                        "attacker_team": safe_str(
                            _get_col(kills_df, idx, ["attacker_team", "attacker_side"])
                        ),
                        "attacker_x": safe_float(
                            _get_col(kills_df, idx, ["attacker_X", "attacker_x"])
                        ),
                        "attacker_y": safe_float(
                            _get_col(kills_df, idx, ["attacker_Y", "attacker_y"])
                        ),
                        "attacker_z": safe_float(
                            _get_col(kills_df, idx, ["attacker_Z", "attacker_z"])
                        ),
                        "victim_steam_id": safe_str(
                            _get_col(
                                kills_df, idx, ["victim_steamid", "user_steamid", "victim_steam_id"]
                            )
                        ),
                        "victim_name": safe_str(
                            _get_col(kills_df, idx, ["victim_name", "user_name", "victim"])
                        ),
                        "victim_team": safe_str(
                            _get_col(kills_df, idx, ["victim_team", "user_team", "victim_side"])
                        ),
                        "victim_x": safe_float(
                            _get_col(kills_df, idx, ["victim_X", "user_X", "victim_x"])
                        ),
                        "victim_y": safe_float(
                            _get_col(kills_df, idx, ["victim_Y", "user_Y", "victim_y"])
                        ),
                        "victim_z": safe_float(
                            _get_col(kills_df, idx, ["victim_Z", "user_Z", "victim_z"])
                        ),
                        "weapon": safe_str(_get_col(kills_df, idx, ["weapon"])),
                        "headshot": safe_bool(_get_col(kills_df, idx, ["headshot", "is_headshot"])),
                        "penetrated": safe_bool(
                            _get_col(kills_df, idx, ["penetrated", "is_penetrated"])
                        ),
                        "noscope": safe_bool(_get_col(kills_df, idx, ["noscope", "is_noscope"])),
                        "thrusmoke": safe_bool(
                            _get_col(kills_df, idx, ["thrusmoke", "is_thrusmoke", "attackerblind"])
                        ),
                        "attackerblind": safe_bool(
                            _get_col(kills_df, idx, ["attackerblind", "is_attackerblind"])
                        ),
                        "distance": safe_float(_get_col(kills_df, idx, ["distance"])),
                        "assister_steam_id": safe_str(
                            _get_col(kills_df, idx, ["assister_steamid", "assister_steam_id"])
                        ),
                        "assister_name": safe_str(
                            _get_col(kills_df, idx, ["assister_name", "assister"])
                        ),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract kill event at index {idx}: {e}")
                continue

        if kill_records:
            session.bulk_insert_mappings(Kill, kill_records)
            counts["kills"] = len(kill_records)
            logger.info(f"Stored {counts['kills']} kill events for match {match_id}")

    # Store damage events
    if hasattr(data, "damages_df") and not data.damages_df.empty:
        damages_df = data.damages_df
        damage_records = []

        for idx in range(len(damages_df)):
            try:
                damage_records.append(
                    {
                        "match_id": match_id,
                        "tick": safe_int(_get_col(damages_df, idx, ["tick"])),
                        "round_num": safe_int(
                            _get_col(damages_df, idx, ["total_rounds_played", "round", "round_num"])
                        ),
                        "attacker_steam_id": safe_str(
                            _get_col(damages_df, idx, ["attacker_steamid", "attacker_steam_id"])
                        ),
                        "victim_steam_id": safe_str(
                            _get_col(
                                damages_df,
                                idx,
                                ["victim_steamid", "user_steamid", "victim_steam_id"],
                            )
                        ),
                        "dmg_health": safe_int(
                            _get_col(damages_df, idx, ["dmg_health", "damage_health", "dmg"])
                        ),
                        "dmg_armor": safe_int(
                            _get_col(damages_df, idx, ["dmg_armor", "damage_armor"])
                        ),
                        "hitgroup": safe_str(_get_col(damages_df, idx, ["hitgroup", "hit_group"])),
                        "weapon": safe_str(_get_col(damages_df, idx, ["weapon"])),
                        "attacker_x": safe_float(
                            _get_col(damages_df, idx, ["attacker_X", "attacker_x"])
                        ),
                        "attacker_y": safe_float(
                            _get_col(damages_df, idx, ["attacker_Y", "attacker_y"])
                        ),
                        "attacker_z": safe_float(
                            _get_col(damages_df, idx, ["attacker_Z", "attacker_z"])
                        ),
                        "victim_x": safe_float(
                            _get_col(damages_df, idx, ["victim_X", "user_X", "victim_x"])
                        ),
                        "victim_y": safe_float(
                            _get_col(damages_df, idx, ["victim_Y", "user_Y", "victim_y"])
                        ),
                        "victim_z": safe_float(
                            _get_col(damages_df, idx, ["victim_Z", "user_Z", "victim_z"])
                        ),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract damage event at index {idx}: {e}")
                continue

        if damage_records:
            session.bulk_insert_mappings(DamageEvent, damage_records)
            counts["damage_events"] = len(damage_records)
            logger.info(f"Stored {counts['damage_events']} damage events for match {match_id}")

    # Store grenade events
    if hasattr(data, "grenades_df") and not data.grenades_df.empty:
        grenades_df = data.grenades_df
        grenade_records = []

        for idx in range(len(grenades_df)):
            try:
                grenade_records.append(
                    {
                        "match_id": match_id,
                        "tick": safe_int(_get_col(grenades_df, idx, ["tick"])),
                        "round_num": safe_int(
                            _get_col(
                                grenades_df, idx, ["total_rounds_played", "round", "round_num"]
                            )
                        ),
                        "thrower_steam_id": safe_str(
                            _get_col(
                                grenades_df,
                                idx,
                                ["thrower_steamid", "user_steamid", "thrower_steam_id"],
                            )
                        ),
                        "thrower_name": safe_str(
                            _get_col(grenades_df, idx, ["thrower_name", "user_name", "thrower"])
                        ),
                        "grenade_type": safe_str(
                            _get_col(grenades_df, idx, ["grenade_type", "weapon", "grenade"])
                        ),
                        "event_type": safe_str(_get_col(grenades_df, idx, ["event_type", "event"])),
                        "x": safe_float(_get_col(grenades_df, idx, ["X", "x"])),
                        "y": safe_float(_get_col(grenades_df, idx, ["Y", "y"])),
                        "z": safe_float(_get_col(grenades_df, idx, ["Z", "z"])),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract grenade event at index {idx}: {e}")
                continue

        if grenade_records:
            session.bulk_insert_mappings(GrenadeEvent, grenade_records)
            counts["grenades"] = len(grenade_records)
            logger.info(f"Stored {counts['grenades']} grenade events for match {match_id}")

    # Store bomb events
    if hasattr(data, "bomb_events_df") and not data.bomb_events_df.empty:
        bomb_events_df = data.bomb_events_df
        bomb_records = []

        for idx in range(len(bomb_events_df)):
            try:
                bomb_records.append(
                    {
                        "match_id": match_id,
                        "tick": safe_int(_get_col(bomb_events_df, idx, ["tick"])),
                        "round_num": safe_int(
                            _get_col(
                                bomb_events_df, idx, ["total_rounds_played", "round", "round_num"]
                            )
                        ),
                        "player_steam_id": safe_str(
                            _get_col(
                                bomb_events_df,
                                idx,
                                ["user_steamid", "player_steamid", "player_steam_id"],
                            )
                        ),
                        "player_name": safe_str(
                            _get_col(bomb_events_df, idx, ["user_name", "player_name", "player"])
                        ),
                        "event_type": safe_str(
                            _get_col(bomb_events_df, idx, ["event_type", "event"])
                        ),
                        "site": safe_str(_get_col(bomb_events_df, idx, ["site"])),
                        "x": safe_float(_get_col(bomb_events_df, idx, ["X", "x"])),
                        "y": safe_float(_get_col(bomb_events_df, idx, ["Y", "y"])),
                        "z": safe_float(_get_col(bomb_events_df, idx, ["Z", "z"])),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract bomb event at index {idx}: {e}")
                continue

        if bomb_records:
            session.bulk_insert_mappings(BombEvent, bomb_records)
            counts["bombs"] = len(bomb_records)
            logger.info(f"Stored {counts['bombs']} bomb events for match {match_id}")

    return counts
