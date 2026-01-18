"""
Demo Parser Wrapper for CS2 Replay Files

Wraps demoparser2 to extract tick-level game data from .dem files,
providing structured access to player positions, events, and game state.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import logging

import pandas as pd
import numpy as np

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
    tick_rate: int
    duration_ticks: int
    duration_seconds: float

    # Player data
    player_names: dict[int, str]  # steam_id -> name
    teams: dict[int, str]  # steam_id -> team

    # DataFrames
    kills_df: pd.DataFrame
    damages_df: pd.DataFrame
    rounds_df: pd.DataFrame

    # Aggregated stats per player
    player_stats: dict[int, dict]  # steam_id -> stats dict


class DemoParser:
    """
    Parser for CS2 demo files using demoparser2.
    """

    def __init__(self, demo_path: str | Path):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        if not self.demo_path.suffix.lower() == ".dem":
            raise ValueError(f"Expected .dem file, got: {self.demo_path.suffix}")

        self._parser: Optional[Any] = None
        self._data: Optional[DemoData] = None

    def parse(self) -> DemoData:
        """Parse the demo file and extract all relevant data."""
        if self._data is not None:
            return self._data

        if Demoparser2 is None:
            raise ImportError(
                "demoparser2 is required but not installed. "
                "Install with: pip install demoparser2"
            )

        logger.info(f"Parsing demo: {self.demo_path}")
        self._parser = Demoparser2(str(self.demo_path))

        # Get header info
        header = self._parser.parse_header()
        map_name = header.get("map_name", "unknown") if isinstance(header, dict) else "unknown"
        tick_rate = 64

        # Parse player info
        player_names = {}
        teams = {}

        try:
            # Get player info from ticks - this is more reliable
            player_df = self._parser.parse_ticks(["steamid", "name", "team_num"])
            if isinstance(player_df, pd.DataFrame) and not player_df.empty:
                # Get unique players
                for steamid in player_df["steamid"].unique():
                    if steamid and steamid != 0:
                        player_data = player_df[player_df["steamid"] == steamid].iloc[-1]
                        player_names[int(steamid)] = str(player_data.get("name", f"Player_{steamid}"))
                        team_num = player_data.get("team_num", 0)
                        teams[int(steamid)] = "CT" if team_num == 3 else "T" if team_num == 2 else "Spec"
        except Exception as e:
            logger.warning(f"Failed to parse player info from ticks: {e}")

        # Parse kills (player_death events)
        kills_df = pd.DataFrame()
        try:
            kills_df = self._parser.parse_event("player_death")
            if isinstance(kills_df, pd.DataFrame) and not kills_df.empty:
                logger.info(f"Found {len(kills_df)} kill events")
                # Rename columns for consistency
                kills_df = kills_df.rename(columns={
                    "attacker_steamid": "attacker_id",
                    "user_steamid": "victim_id",
                    "assister_steamid": "assister_id",
                })
        except Exception as e:
            logger.warning(f"Failed to parse kills: {e}")
            kills_df = pd.DataFrame()

        # Parse damage (player_hurt events)
        damages_df = pd.DataFrame()
        try:
            damages_df = self._parser.parse_event("player_hurt")
            if isinstance(damages_df, pd.DataFrame) and not damages_df.empty:
                logger.info(f"Found {len(damages_df)} damage events")
                damages_df = damages_df.rename(columns={
                    "attacker_steamid": "attacker_id",
                    "user_steamid": "victim_id",
                    "dmg_health": "damage",
                    "dmg_armor": "armor_damage",
                })
        except Exception as e:
            logger.warning(f"Failed to parse damage: {e}")
            damages_df = pd.DataFrame()

        # Parse rounds
        rounds_df = pd.DataFrame()
        try:
            round_end = self._parser.parse_event("round_end")
            if isinstance(round_end, pd.DataFrame) and not round_end.empty:
                rounds_df = round_end
                logger.info(f"Found {len(rounds_df)} rounds")
        except Exception as e:
            logger.warning(f"Failed to parse rounds: {e}")

        # Calculate duration from ticks
        duration_ticks = 0
        duration_seconds = 0.0
        try:
            tick_df = self._parser.parse_ticks(["tick"])
            if isinstance(tick_df, pd.DataFrame) and not tick_df.empty:
                duration_ticks = int(tick_df["tick"].max())
                duration_seconds = duration_ticks / tick_rate
        except Exception as e:
            logger.warning(f"Failed to get duration: {e}")

        # Calculate per-player stats
        player_stats = self._calculate_player_stats(
            player_names, teams, kills_df, damages_df, rounds_df
        )

        self._data = DemoData(
            file_path=self.demo_path,
            map_name=map_name,
            tick_rate=tick_rate,
            duration_ticks=duration_ticks,
            duration_seconds=duration_seconds,
            player_names=player_names,
            teams=teams,
            kills_df=kills_df,
            damages_df=damages_df,
            rounds_df=rounds_df,
            player_stats=player_stats,
        )

        logger.info(f"Parsed demo: {map_name}, {duration_seconds:.0f}s, {len(player_names)} players")
        return self._data

    def _calculate_player_stats(
        self,
        player_names: dict[int, str],
        teams: dict[int, str],
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        rounds_df: pd.DataFrame,
    ) -> dict[int, dict]:
        """Calculate aggregated stats for each player."""
        stats = {}
        num_rounds = len(rounds_df) if not rounds_df.empty else 1

        for steam_id, name in player_names.items():
            # Count kills
            kills = 0
            headshots = 0
            if not kills_df.empty and "attacker_id" in kills_df.columns:
                player_kills = kills_df[kills_df["attacker_id"] == steam_id]
                kills = len(player_kills)
                if "headshot" in kills_df.columns:
                    headshots = int(player_kills["headshot"].sum())

            # Count deaths
            deaths = 0
            if not kills_df.empty and "victim_id" in kills_df.columns:
                deaths = len(kills_df[kills_df["victim_id"] == steam_id])

            # Count assists
            assists = 0
            if not kills_df.empty and "assister_id" in kills_df.columns:
                assists = len(kills_df[kills_df["assister_id"] == steam_id])

            # Calculate damage
            total_damage = 0
            if not damages_df.empty and "attacker_id" in damages_df.columns:
                player_damage = damages_df[damages_df["attacker_id"] == steam_id]
                if "damage" in damages_df.columns:
                    total_damage = int(player_damage["damage"].sum())

            # Calculate ADR (Average Damage per Round)
            adr = total_damage / num_rounds if num_rounds > 0 else 0

            # Calculate K/D ratio
            kd_ratio = kills / deaths if deaths > 0 else kills

            # Calculate headshot percentage
            hs_percent = (headshots / kills * 100) if kills > 0 else 0

            stats[steam_id] = {
                "name": name,
                "team": teams.get(steam_id, "Unknown"),
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
