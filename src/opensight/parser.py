"""
Demo Parser Wrapper for CS2 Replay Files

Uses demoparser2 to extract game events and player stats from .dem files.
API Reference: https://github.com/LaihoE/demoparser
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import logging

import pandas as pd

try:
    from demoparser2 import DemoParser as Demoparser2
except ImportError:
    Demoparser2 = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class DemoData:
    """Parsed demo data container."""
    file_path: Path
    map_name: str
    duration_seconds: float
    player_stats: dict[int, dict]
    num_rounds: int
    kills_df: pd.DataFrame
    damages_df: pd.DataFrame


class DemoParser:
    """Parser for CS2 demo files using demoparser2."""

    def __init__(self, demo_path: str | Path):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")

        self._data: Optional[DemoData] = None

    def parse(self) -> DemoData:
        """Parse the demo file and extract all relevant data."""
        if self._data is not None:
            return self._data

        if Demoparser2 is None:
            raise ImportError("demoparser2 is required. Install with: pip install demoparser2")

        logger.info(f"Parsing demo: {self.demo_path}")
        parser = Demoparser2(str(self.demo_path))

        # Get header info
        try:
            header = parser.parse_header()
            map_name = header.get("map_name", "unknown") if isinstance(header, dict) else "unknown"
        except Exception as e:
            logger.warning(f"Failed to parse header: {e}")
            map_name = "unknown"

        # Parse player_death events - this gives us kills
        # Columns returned: tick, user_steamid, user_name, attacker_steamid, attacker_name,
        #                   assister_steamid, weapon, headshot, etc.
        kills_df = pd.DataFrame()
        try:
            kills_df = parser.parse_event("player_death")
            if kills_df is not None and not kills_df.empty:
                logger.info(f"Found {len(kills_df)} kill events")
                logger.info(f"Kill columns: {list(kills_df.columns)}")
        except Exception as e:
            logger.warning(f"Failed to parse player_death: {e}")

        # Parse player_hurt events - this gives us damage
        damages_df = pd.DataFrame()
        try:
            damages_df = parser.parse_event("player_hurt")
            if damages_df is not None and not damages_df.empty:
                logger.info(f"Found {len(damages_df)} damage events")
                logger.info(f"Damage columns: {list(damages_df.columns)}")
        except Exception as e:
            logger.warning(f"Failed to parse player_hurt: {e}")

        # Parse round_end events
        num_rounds = 0
        try:
            rounds_df = parser.parse_event("round_end")
            if rounds_df is not None and not rounds_df.empty:
                num_rounds = len(rounds_df)
                logger.info(f"Found {num_rounds} rounds")
        except Exception as e:
            logger.warning(f"Failed to parse round_end: {e}")

        # Get duration from ticks
        duration_seconds = 0.0
        try:
            ticks_df = parser.parse_ticks(["tick"])
            if ticks_df is not None and not ticks_df.empty:
                max_tick = int(ticks_df["tick"].max())
                duration_seconds = max_tick / 64  # Assume 64 tick
        except Exception as e:
            logger.warning(f"Failed to get duration: {e}")

        # Calculate player stats from the DataFrames
        player_stats = self._calculate_stats(kills_df, damages_df, num_rounds)

        self._data = DemoData(
            file_path=self.demo_path,
            map_name=map_name,
            duration_seconds=duration_seconds,
            player_stats=player_stats,
            num_rounds=max(num_rounds, 1),
            kills_df=kills_df,
            damages_df=damages_df,
        )

        logger.info(f"Parsed: {map_name}, {duration_seconds:.0f}s, {len(player_stats)} players, {num_rounds} rounds")
        return self._data

    def _calculate_stats(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        num_rounds: int
    ) -> dict[int, dict]:
        """Calculate player stats from event DataFrames."""
        stats: dict[int, dict] = {}
        num_rounds = max(num_rounds, 1)

        # Find the correct column names for steam IDs
        attacker_col = None
        victim_col = None
        assister_col = None
        attacker_name_col = None
        victim_name_col = None

        if not kills_df.empty:
            cols = kills_df.columns.tolist()
            logger.info(f"Available kill columns: {cols}")

            # Try different possible column names
            for col in ["attacker_steamid", "attacker_steam_id", "attacker"]:
                if col in cols:
                    attacker_col = col
                    break

            for col in ["user_steamid", "user_steam_id", "userid", "victim_steamid"]:
                if col in cols:
                    victim_col = col
                    break

            for col in ["assister_steamid", "assister_steam_id", "assister"]:
                if col in cols:
                    assister_col = col
                    break

            for col in ["attacker_name", "attacker"]:
                if col in cols and "steamid" not in col.lower():
                    attacker_name_col = col
                    break

            for col in ["user_name", "user", "victim_name"]:
                if col in cols and "steamid" not in col.lower():
                    victim_name_col = col
                    break

        # Get unique players from kills
        all_players = set()
        player_names = {}

        if not kills_df.empty:
            if attacker_col and attacker_col in kills_df.columns:
                for sid in kills_df[attacker_col].dropna().unique():
                    if sid and sid != 0:
                        all_players.add(int(sid))

            if victim_col and victim_col in kills_df.columns:
                for sid in kills_df[victim_col].dropna().unique():
                    if sid and sid != 0:
                        all_players.add(int(sid))

            # Get player names
            if attacker_col and attacker_name_col:
                for _, row in kills_df.drop_duplicates(subset=[attacker_col]).iterrows():
                    sid = row.get(attacker_col)
                    name = row.get(attacker_name_col)
                    if sid and sid != 0 and name:
                        player_names[int(sid)] = str(name)

            if victim_col and victim_name_col:
                for _, row in kills_df.drop_duplicates(subset=[victim_col]).iterrows():
                    sid = row.get(victim_col)
                    name = row.get(victim_name_col)
                    if sid and sid != 0 and name:
                        player_names[int(sid)] = str(name)

        # Calculate stats for each player
        for steam_id in all_players:
            kills = 0
            deaths = 0
            assists = 0
            headshots = 0

            if not kills_df.empty:
                # Count kills
                if attacker_col:
                    player_kills = kills_df[kills_df[attacker_col] == steam_id]
                    kills = len(player_kills)
                    if "headshot" in kills_df.columns:
                        headshots = int(player_kills["headshot"].sum())

                # Count deaths
                if victim_col:
                    deaths = len(kills_df[kills_df[victim_col] == steam_id])

                # Count assists
                if assister_col:
                    assists = len(kills_df[kills_df[assister_col] == steam_id])

            # Calculate damage
            total_damage = 0
            damage_col = None
            damage_attacker_col = None

            if not damages_df.empty:
                damage_cols = damages_df.columns.tolist()

                for col in ["dmg_health", "damage", "dmg"]:
                    if col in damage_cols:
                        damage_col = col
                        break

                for col in ["attacker_steamid", "attacker_steam_id", "attacker"]:
                    if col in damage_cols:
                        damage_attacker_col = col
                        break

                if damage_col and damage_attacker_col:
                    player_damage = damages_df[damages_df[damage_attacker_col] == steam_id]
                    total_damage = int(player_damage[damage_col].sum())

            # Calculate derived stats
            adr = total_damage / num_rounds
            kd_ratio = kills / deaths if deaths > 0 else float(kills)
            hs_percent = (headshots / kills * 100) if kills > 0 else 0.0

            stats[steam_id] = {
                "name": player_names.get(steam_id, f"Player_{steam_id}"),
                "team": "Unknown",  # Would need team data from ticks
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "headshots": headshots,
                "hs_percent": round(hs_percent, 1),
                "total_damage": total_damage,
                "adr": round(adr, 1),
                "kd_ratio": round(kd_ratio, 2),
            }

        return stats


def parse_demo(demo_path: str | Path) -> DemoData:
    """Convenience function to parse a demo file."""
    parser = DemoParser(demo_path)
    return parser.parse()
