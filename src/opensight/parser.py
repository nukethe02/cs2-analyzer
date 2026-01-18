"""
Demo Parser for CS2 Replay Files

Uses awpy (built on demoparser2) for reliable, structured demo parsing.
Awpy provides consistent column names and pre-built DataFrames for kills, damages, rounds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import logging

import pandas as pd
import numpy as np

try:
    from awpy import Demo
    AWPY_AVAILABLE = True
except ImportError:
    AWPY_AVAILABLE = False

try:
    from demoparser2 import DemoParser as Demoparser2
    DEMOPARSER2_AVAILABLE = True
except ImportError:
    DEMOPARSER2_AVAILABLE = False

logger = logging.getLogger(__name__)


# Safe type conversion helpers
def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    """Safely convert a value to string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert a value to bool."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return bool(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


@dataclass
class KillEvent:
    """A kill event with timing and position data."""
    tick: int
    round_num: int
    attacker_steamid: int
    attacker_name: str
    attacker_side: str  # "CT" or "T"
    victim_steamid: int
    victim_name: str
    victim_side: str
    weapon: str
    headshot: bool
    assister_steamid: Optional[int] = None
    assister_name: Optional[str] = None
    flash_assist: bool = False
    # Positions (if available)
    attacker_x: Optional[float] = None
    attacker_y: Optional[float] = None
    attacker_z: Optional[float] = None
    victim_x: Optional[float] = None
    victim_y: Optional[float] = None
    victim_z: Optional[float] = None


@dataclass
class DamageEvent:
    """A damage event."""
    tick: int
    round_num: int
    attacker_steamid: int
    attacker_name: str
    attacker_side: str
    victim_steamid: int
    victim_name: str
    victim_side: str
    damage: int
    weapon: str
    hitgroup: str  # 'head', 'chest', 'generic', etc.


@dataclass
class RoundInfo:
    """Information about a round."""
    round_num: int
    start_tick: int
    end_tick: int
    winner: str  # "CT" or "T"
    reason: str  # "bomb_exploded", "ct_killed", etc.
    bomb_planted: bool
    bomb_site: Optional[str] = None


@dataclass
class DemoData:
    """Complete parsed demo data."""
    file_path: Path
    map_name: str
    duration_seconds: float
    tick_rate: int
    num_rounds: int

    # Player info
    player_stats: dict[int, dict]
    player_names: dict[int, str]
    player_teams: dict[int, str]  # Now stores "CT"/"T" strings

    # Events
    kills: list[KillEvent]
    damages: list[DamageEvent]
    rounds: list[RoundInfo]

    # DataFrames for detailed analysis
    kills_df: pd.DataFrame
    damages_df: pd.DataFrame
    rounds_df: pd.DataFrame
    ticks_df: Optional[pd.DataFrame] = None


class DemoParser:
    """Parser for CS2 demo files using awpy (preferred) or demoparser2 (fallback)."""

    def __init__(self, demo_path: str | Path):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        self._data: Optional[DemoData] = None

    def parse(self, include_ticks: bool = False) -> DemoData:
        """Parse the demo file and extract all relevant data.

        Args:
            include_ticks: If True, also parse tick-level position data (slower).
        """
        if self._data is not None:
            return self._data

        if AWPY_AVAILABLE:
            logger.info("Using awpy parser (recommended)")
            return self._parse_with_awpy(include_ticks)
        elif DEMOPARSER2_AVAILABLE:
            logger.info("Using demoparser2 fallback")
            return self._parse_with_demoparser2(include_ticks)
        else:
            raise ImportError("No parser available. Install awpy: pip install awpy")

    def _parse_with_awpy(self, include_ticks: bool = False) -> DemoData:
        """Parse using awpy - provides clean, consistent data structures."""
        logger.info(f"Parsing demo with awpy: {self.demo_path}")

        # Parse the demo
        demo = Demo(str(self.demo_path), verbose=False)

        # Parse with optional tick data
        if include_ticks:
            demo.parse(player_props=["X", "Y", "Z", "pitch", "yaw", "health"])
        else:
            demo.parse()

        # Get header info
        header = demo.header
        map_name = header.get("map_name", "unknown") if header else "unknown"

        # Convert Polars to Pandas for compatibility
        kills_df = demo.kills.to_pandas() if demo.kills is not None else pd.DataFrame()
        damages_df = demo.damages.to_pandas() if demo.damages is not None else pd.DataFrame()
        rounds_df = demo.rounds.to_pandas() if demo.rounds is not None else pd.DataFrame()
        ticks_df = demo.ticks.to_pandas() if include_ticks and demo.ticks is not None else None

        logger.info(f"Parsed: {len(kills_df)} kills, {len(damages_df)} damages, {len(rounds_df)} rounds")
        if not kills_df.empty:
            logger.info(f"Kill columns: {list(kills_df.columns)}")
        if not damages_df.empty:
            logger.info(f"Damage columns: {list(damages_df.columns)}")

        # Calculate duration
        tick_rate = 64
        max_tick = 0
        if not kills_df.empty and "tick" in kills_df.columns:
            max_tick = max(max_tick, int(kills_df["tick"].max()))
        if not rounds_df.empty and "end" in rounds_df.columns:
            max_tick = max(max_tick, int(rounds_df["end"].max()))
        duration_seconds = max_tick / tick_rate

        # Extract player info
        player_names, player_teams = self._extract_players_awpy(kills_df, damages_df)

        # Calculate stats
        num_rounds = len(rounds_df) if not rounds_df.empty else 1
        player_stats = self._calculate_stats_awpy(kills_df, damages_df, player_names, player_teams, num_rounds)

        # Build structured events
        kills = self._build_kills_awpy(kills_df)
        damages = self._build_damages_awpy(damages_df)
        rounds = self._build_rounds_awpy(rounds_df)

        self._data = DemoData(
            file_path=self.demo_path,
            map_name=map_name,
            duration_seconds=duration_seconds,
            tick_rate=tick_rate,
            num_rounds=num_rounds,
            player_stats=player_stats,
            player_names=player_names,
            player_teams=player_teams,
            kills=kills,
            damages=damages,
            rounds=rounds,
            kills_df=kills_df,
            damages_df=damages_df,
            rounds_df=rounds_df,
            ticks_df=ticks_df,
        )

        logger.info(f"Parsed: {map_name}, {duration_seconds:.0f}s, {len(player_stats)} players, {num_rounds} rounds")
        return self._data

    def _extract_players_awpy(self, kills_df: pd.DataFrame, damages_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
        """Extract player names and teams from awpy DataFrames."""
        names: dict[int, str] = {}
        teams: dict[int, str] = {}

        # Extract from kills
        if not kills_df.empty:
            for col_id, col_name, col_side in [
                ("attacker_steamid", "attacker_name", "attacker_side"),
                ("victim_steamid", "victim_name", "victim_side"),
            ]:
                if col_id in kills_df.columns and col_name in kills_df.columns:
                    for _, row in kills_df.drop_duplicates(subset=[col_id]).iterrows():
                        sid = safe_int(row.get(col_id))
                        name = safe_str(row.get(col_name))
                        side = safe_str(row.get(col_side)) if col_side in kills_df.columns else ""
                        if sid and name:
                            names[sid] = name
                            if side:
                                teams[sid] = side

        # Also check damages for any missing players
        if not damages_df.empty:
            for col_id, col_name, col_side in [
                ("attacker_steamid", "attacker_name", "attacker_side"),
                ("victim_steamid", "victim_name", "victim_side"),
            ]:
                if col_id in damages_df.columns and col_name in damages_df.columns:
                    for _, row in damages_df.drop_duplicates(subset=[col_id]).iterrows():
                        sid = safe_int(row.get(col_id))
                        if sid and sid not in names:
                            names[sid] = safe_str(row.get(col_name))
                            if col_side in damages_df.columns:
                                teams[sid] = safe_str(row.get(col_side))

        return names, teams

    def _calculate_stats_awpy(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        player_names: dict[int, str],
        player_teams: dict[int, str],
        num_rounds: int
    ) -> dict[int, dict]:
        """Calculate player statistics from awpy DataFrames."""
        stats: dict[int, dict] = {}
        num_rounds = max(num_rounds, 1)

        for steam_id, name in player_names.items():
            kills = 0
            deaths = 0
            assists = 0
            headshots = 0
            total_damage = 0
            weapon_kills: dict[str, int] = {}

            if not kills_df.empty:
                # Kills (attacker)
                if "attacker_steamid" in kills_df.columns:
                    player_kills = kills_df[kills_df["attacker_steamid"] == steam_id]
                    kills = len(player_kills)

                    # Headshots
                    if "headshot" in kills_df.columns and kills > 0:
                        headshots = int(player_kills["headshot"].sum())

                    # Weapon breakdown
                    if "weapon" in kills_df.columns and kills > 0:
                        weapon_kills = player_kills["weapon"].value_counts().to_dict()

                # Deaths (victim)
                if "victim_steamid" in kills_df.columns:
                    deaths = len(kills_df[kills_df["victim_steamid"] == steam_id])

                # Assists
                if "assister_steamid" in kills_df.columns:
                    assists = len(kills_df[kills_df["assister_steamid"] == steam_id])

            # Damage
            if not damages_df.empty and "attacker_steamid" in damages_df.columns and "damage" in damages_df.columns:
                player_dmg = damages_df[damages_df["attacker_steamid"] == steam_id]
                total_damage = int(player_dmg["damage"].sum())

            # Derived stats
            kd_ratio = round(kills / deaths, 2) if deaths > 0 else float(kills)
            hs_percent = round((headshots / kills * 100), 1) if kills > 0 else 0.0
            adr = round(total_damage / num_rounds, 1)

            team = player_teams.get(steam_id, "Unknown")

            stats[steam_id] = {
                "name": name,
                "team": team,
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "headshots": headshots,
                "hs_percent": hs_percent,
                "total_damage": total_damage,
                "adr": adr,
                "kd_ratio": kd_ratio,
                "weapon_kills": weapon_kills,
            }

        return stats

    def _build_kills_awpy(self, kills_df: pd.DataFrame) -> list[KillEvent]:
        """Build KillEvent list from awpy kills DataFrame."""
        kills = []
        if kills_df.empty:
            return kills

        for _, row in kills_df.iterrows():
            kill = KillEvent(
                tick=safe_int(row.get("tick")),
                round_num=safe_int(row.get("round_num")),
                attacker_steamid=safe_int(row.get("attacker_steamid")),
                attacker_name=safe_str(row.get("attacker_name")),
                attacker_side=safe_str(row.get("attacker_side")),
                victim_steamid=safe_int(row.get("victim_steamid")),
                victim_name=safe_str(row.get("victim_name")),
                victim_side=safe_str(row.get("victim_side")),
                weapon=safe_str(row.get("weapon")),
                headshot=safe_bool(row.get("headshot")),
                assister_steamid=safe_int(row.get("assister_steamid")) if row.get("assister_steamid") else None,
                assister_name=safe_str(row.get("assister_name")) if row.get("assister_name") else None,
                flash_assist=safe_bool(row.get("flash_assist")),
                # Positions if available
                attacker_x=safe_float(row.get("attacker_X")) if "attacker_X" in row else None,
                attacker_y=safe_float(row.get("attacker_Y")) if "attacker_Y" in row else None,
                attacker_z=safe_float(row.get("attacker_Z")) if "attacker_Z" in row else None,
                victim_x=safe_float(row.get("victim_X")) if "victim_X" in row else None,
                victim_y=safe_float(row.get("victim_Y")) if "victim_Y" in row else None,
                victim_z=safe_float(row.get("victim_Z")) if "victim_Z" in row else None,
            )
            kills.append(kill)

        return kills

    def _build_damages_awpy(self, damages_df: pd.DataFrame) -> list[DamageEvent]:
        """Build DamageEvent list from awpy damages DataFrame."""
        damages = []
        if damages_df.empty:
            return damages

        for _, row in damages_df.iterrows():
            dmg = DamageEvent(
                tick=safe_int(row.get("tick")),
                round_num=safe_int(row.get("round_num")),
                attacker_steamid=safe_int(row.get("attacker_steamid")),
                attacker_name=safe_str(row.get("attacker_name")),
                attacker_side=safe_str(row.get("attacker_side")),
                victim_steamid=safe_int(row.get("victim_steamid")),
                victim_name=safe_str(row.get("victim_name")),
                victim_side=safe_str(row.get("victim_side")),
                damage=safe_int(row.get("damage")),
                weapon=safe_str(row.get("weapon")),
                hitgroup=safe_str(row.get("hitgroup")),
            )
            damages.append(dmg)

        return damages

    def _build_rounds_awpy(self, rounds_df: pd.DataFrame) -> list[RoundInfo]:
        """Build RoundInfo list from awpy rounds DataFrame."""
        rounds = []
        if rounds_df.empty:
            return rounds

        for _, row in rounds_df.iterrows():
            round_info = RoundInfo(
                round_num=safe_int(row.get("round_num")),
                start_tick=safe_int(row.get("start")),
                end_tick=safe_int(row.get("end")),
                winner=safe_str(row.get("winner")),
                reason=safe_str(row.get("reason")),
                bomb_planted=row.get("bomb_plant") is not None and not pd.isna(row.get("bomb_plant")),
                bomb_site=safe_str(row.get("bomb_site")) if row.get("bomb_site") != "not_planted" else None,
            )
            rounds.append(round_info)

        return rounds

    def _parse_with_demoparser2(self, include_ticks: bool = False) -> DemoData:
        """Fallback parser using raw demoparser2."""
        logger.info(f"Parsing demo with demoparser2 (fallback): {self.demo_path}")
        parser = Demoparser2(str(self.demo_path))

        # Parse header
        map_name = "unknown"
        try:
            header = parser.parse_header()
            if isinstance(header, dict):
                map_name = header.get("map_name", "unknown")
        except Exception as e:
            logger.warning(f"Failed to parse header: {e}")

        # Parse events
        kills_df = pd.DataFrame()
        damages_df = pd.DataFrame()
        rounds_df = pd.DataFrame()

        try:
            kills_df = parser.parse_event("player_death")
            if kills_df is not None:
                logger.info(f"Found {len(kills_df)} kills. Columns: {list(kills_df.columns)}")
        except Exception as e:
            logger.warning(f"Failed to parse kills: {e}")

        try:
            damages_df = parser.parse_event("player_hurt")
            if damages_df is not None:
                logger.info(f"Found {len(damages_df)} damages. Columns: {list(damages_df.columns)}")
        except Exception as e:
            logger.warning(f"Failed to parse damages: {e}")

        try:
            rounds_df = parser.parse_event("round_end")
            if rounds_df is not None:
                logger.info(f"Found {len(rounds_df)} rounds")
        except Exception as e:
            logger.warning(f"Failed to parse rounds: {e}")

        # Ensure DataFrames
        kills_df = kills_df if kills_df is not None else pd.DataFrame()
        damages_df = damages_df if damages_df is not None else pd.DataFrame()
        rounds_df = rounds_df if rounds_df is not None else pd.DataFrame()

        # Calculate stats (simplified for fallback)
        tick_rate = 64
        max_tick = 0
        if not kills_df.empty and "tick" in kills_df.columns:
            max_tick = int(kills_df["tick"].max())
        duration_seconds = max_tick / tick_rate

        # Extract players
        player_names: dict[int, str] = {}
        player_teams: dict[int, str] = {}

        def find_col(df, options):
            for col in options:
                if col in df.columns:
                    return col
            return None

        if not kills_df.empty:
            att_id = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
            att_name = find_col(kills_df, ["attacker_name"])
            vic_id = find_col(kills_df, ["user_steamid", "victim_steamid"])
            vic_name = find_col(kills_df, ["user_name", "victim_name"])

            if att_id and att_name:
                for _, row in kills_df.drop_duplicates(subset=[att_id]).iterrows():
                    sid = safe_int(row.get(att_id))
                    if sid:
                        player_names[sid] = safe_str(row.get(att_name))
                        player_teams[sid] = "Unknown"

            if vic_id and vic_name:
                for _, row in kills_df.drop_duplicates(subset=[vic_id]).iterrows():
                    sid = safe_int(row.get(vic_id))
                    if sid and sid not in player_names:
                        player_names[sid] = safe_str(row.get(vic_name))
                        player_teams[sid] = "Unknown"

        num_rounds = len(rounds_df) if not rounds_df.empty else 1
        player_stats = self._calculate_stats_fallback(kills_df, damages_df, player_names, num_rounds)

        self._data = DemoData(
            file_path=self.demo_path,
            map_name=map_name,
            duration_seconds=duration_seconds,
            tick_rate=tick_rate,
            num_rounds=num_rounds,
            player_stats=player_stats,
            player_names=player_names,
            player_teams=player_teams,
            kills=[],  # Simplified for fallback
            damages=[],
            rounds=[],
            kills_df=kills_df,
            damages_df=damages_df,
            rounds_df=rounds_df,
            ticks_df=None,
        )

        return self._data

    def _calculate_stats_fallback(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        player_names: dict[int, str],
        num_rounds: int
    ) -> dict[int, dict]:
        """Calculate stats for demoparser2 fallback."""
        stats: dict[int, dict] = {}
        num_rounds = max(num_rounds, 1)

        def find_col(df, options):
            for col in options:
                if col in df.columns:
                    return col
            return None

        att_col = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        vic_col = find_col(kills_df, ["user_steamid", "victim_steamid"])
        hs_col = find_col(kills_df, ["headshot"])
        weapon_col = find_col(kills_df, ["weapon"])
        dmg_att_col = find_col(damages_df, ["attacker_steamid", "attacker_steam_id"])
        dmg_col = find_col(damages_df, ["dmg_health", "damage", "dmg"])

        for steam_id, name in player_names.items():
            kills = 0
            deaths = 0
            headshots = 0
            total_damage = 0
            weapon_kills = {}

            if not kills_df.empty and att_col:
                player_kills = kills_df[kills_df[att_col] == steam_id]
                kills = len(player_kills)
                if hs_col and kills > 0:
                    try:
                        headshots = int(player_kills[hs_col].sum())
                    except:
                        headshots = 0
                if weapon_col and kills > 0:
                    weapon_kills = player_kills[weapon_col].value_counts().to_dict()

            if not kills_df.empty and vic_col:
                deaths = len(kills_df[kills_df[vic_col] == steam_id])

            if not damages_df.empty and dmg_att_col and dmg_col:
                try:
                    total_damage = int(damages_df[damages_df[dmg_att_col] == steam_id][dmg_col].sum())
                except:
                    total_damage = 0

            stats[steam_id] = {
                "name": name,
                "team": "Unknown",
                "kills": kills,
                "deaths": deaths,
                "assists": 0,
                "headshots": headshots,
                "hs_percent": round((headshots / kills * 100), 1) if kills > 0 else 0.0,
                "total_damage": total_damage,
                "adr": round(total_damage / num_rounds, 1),
                "kd_ratio": round(kills / deaths, 2) if deaths > 0 else float(kills),
                "weapon_kills": weapon_kills,
            }

        return stats


def parse_demo(demo_path: str | Path, include_ticks: bool = False) -> DemoData:
    """Convenience function to parse a demo file."""
    parser = DemoParser(demo_path)
    return parser.parse(include_ticks=include_ticks)
