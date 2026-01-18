"""
Demo Parser for CS2 Replay Files

Uses demoparser2 to extract game events and tick-level player state data.
Provides data for TTD and Crosshair Placement analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import logging
import math

import pandas as pd
import numpy as np

try:
    from demoparser2 import DemoParser as Demoparser2
except ImportError:
    Demoparser2 = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class PlayerState:
    """Player state at a specific tick."""
    tick: int
    steam_id: int
    name: str
    team: int  # 2=T, 3=CT
    position: np.ndarray  # X, Y, Z
    eye_angles: np.ndarray  # Pitch, Yaw
    health: int
    is_alive: bool


@dataclass
class KillEvent:
    """A kill event with timing and position data."""
    tick: int
    attacker_steamid: int
    attacker_name: str
    victim_steamid: int
    victim_name: str
    weapon: str
    headshot: bool
    attacker_position: Optional[np.ndarray] = None
    victim_position: Optional[np.ndarray] = None
    attacker_angles: Optional[np.ndarray] = None


@dataclass
class DamageEvent:
    """A damage event."""
    tick: int
    attacker_steamid: int
    victim_steamid: int
    damage: int
    weapon: str
    hitgroup: str  # 'head', 'chest', 'generic', etc.


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int, returning default on failure."""
    if value is None:
        return default
    try:
        # Handle pandas NA/NaN
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
    player_teams: dict[int, int]

    # Events
    kills: list[KillEvent]
    damages: list[DamageEvent]

    # DataFrames for detailed analysis
    kills_df: pd.DataFrame
    damages_df: pd.DataFrame
    ticks_df: Optional[pd.DataFrame] = None

    # Round data
    round_wins: list[dict] = field(default_factory=list)


class DemoParser:
    """Parser for CS2 demo files using demoparser2."""

    # Properties to extract at tick level for advanced analysis
    TICK_PROPS = [
        "X", "Y", "Z",  # Position
        "pitch", "yaw",  # View angles
        "health",
        "team_num",
        "is_alive",
    ]

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

        # Parse header for map name
        map_name = self._parse_header(parser)

        # Parse events
        kills_df = self._parse_kills(parser)
        damages_df = self._parse_damages(parser)
        round_ends = self._parse_rounds(parser)

        # Parse tick-level data for position/angle analysis
        ticks_df = self._parse_ticks(parser)

        # Calculate duration and tick rate
        duration_seconds, tick_rate = self._get_duration(ticks_df, kills_df)

        # Extract player info and calculate stats
        player_names, player_teams = self._extract_players(kills_df, damages_df, ticks_df)
        player_stats = self._calculate_stats(kills_df, damages_df, player_names, player_teams, len(round_ends))

        # Build structured kill/damage events
        kills = self._build_kill_events(kills_df, ticks_df)
        damages = self._build_damage_events(damages_df)

        self._data = DemoData(
            file_path=self.demo_path,
            map_name=map_name,
            duration_seconds=duration_seconds,
            tick_rate=tick_rate,
            num_rounds=max(len(round_ends), 1),
            player_stats=player_stats,
            player_names=player_names,
            player_teams=player_teams,
            kills=kills,
            damages=damages,
            kills_df=kills_df,
            damages_df=damages_df,
            ticks_df=ticks_df,
            round_wins=[],
        )

        logger.info(f"Parsed: {map_name}, {duration_seconds:.0f}s, {len(player_stats)} players, {len(round_ends)} rounds")
        return self._data

    def _parse_header(self, parser: Demoparser2) -> str:
        """Extract map name from header."""
        try:
            header = parser.parse_header()
            if isinstance(header, dict):
                return header.get("map_name", "unknown")
        except Exception as e:
            logger.warning(f"Failed to parse header: {e}")
        return "unknown"

    def _parse_kills(self, parser: Demoparser2) -> pd.DataFrame:
        """Parse player_death events."""
        try:
            df = parser.parse_event("player_death")
            if df is not None and not df.empty:
                logger.info(f"Found {len(df)} kill events. Columns: {list(df.columns)}")
                return df
        except Exception as e:
            logger.warning(f"Failed to parse player_death: {e}")
        return pd.DataFrame()

    def _parse_damages(self, parser: Demoparser2) -> pd.DataFrame:
        """Parse player_hurt events."""
        try:
            df = parser.parse_event("player_hurt")
            if df is not None and not df.empty:
                logger.info(f"Found {len(df)} damage events. Columns: {list(df.columns)}")
                return df
        except Exception as e:
            logger.warning(f"Failed to parse player_hurt: {e}")
        return pd.DataFrame()

    def _parse_rounds(self, parser: Demoparser2) -> pd.DataFrame:
        """Parse round_end events."""
        try:
            df = parser.parse_event("round_end")
            if df is not None and not df.empty:
                logger.info(f"Found {len(df)} rounds")
                return df
        except Exception as e:
            logger.warning(f"Failed to parse round_end: {e}")
        return pd.DataFrame()

    def _parse_ticks(self, parser: Demoparser2) -> Optional[pd.DataFrame]:
        """Parse tick-level player state data."""
        try:
            # Try to get tick data with positions and angles
            df = parser.parse_ticks(self.TICK_PROPS)
            if df is not None and not df.empty:
                logger.info(f"Parsed {len(df)} tick records. Columns: {list(df.columns)}")
                return df
        except Exception as e:
            logger.warning(f"Failed to parse ticks: {e}")
        return None

    def _get_duration(self, ticks_df: Optional[pd.DataFrame], kills_df: pd.DataFrame) -> tuple[float, int]:
        """Calculate demo duration and tick rate."""
        tick_rate = 64  # Default CS2 tick rate
        max_tick = 0

        if ticks_df is not None and not ticks_df.empty and "tick" in ticks_df.columns:
            max_tick = int(ticks_df["tick"].max())
        elif not kills_df.empty and "tick" in kills_df.columns:
            max_tick = int(kills_df["tick"].max())

        duration = max_tick / tick_rate if max_tick > 0 else 0
        return duration, tick_rate

    def _extract_players(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        ticks_df: Optional[pd.DataFrame]
    ) -> tuple[dict[int, str], dict[int, int]]:
        """Extract player names and teams from events."""
        names: dict[int, str] = {}
        teams: dict[int, int] = {}

        # Helper to find column
        def find_col(df: pd.DataFrame, options: list[str]) -> Optional[str]:
            for col in options:
                if col in df.columns:
                    return col
            return None

        # Extract from kills
        if not kills_df.empty:
            att_id = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
            att_name = find_col(kills_df, ["attacker_name"])
            vic_id = find_col(kills_df, ["user_steamid", "victim_steamid"])
            vic_name = find_col(kills_df, ["user_name", "victim_name"])

            if att_id and att_name:
                for _, row in kills_df.drop_duplicates(subset=[att_id]).iterrows():
                    sid = row.get(att_id)
                    name = row.get(att_name)
                    if sid and sid != 0 and name:
                        names[int(sid)] = str(name)

            if vic_id and vic_name:
                for _, row in kills_df.drop_duplicates(subset=[vic_id]).iterrows():
                    sid = row.get(vic_id)
                    name = row.get(vic_name)
                    if sid and sid != 0 and name:
                        names[int(sid)] = str(name)

        # Extract teams from ticks if available
        if ticks_df is not None and not ticks_df.empty:
            steamid_col = find_col(ticks_df, ["steamid", "steam_id"])
            team_col = find_col(ticks_df, ["team_num", "team"])

            if steamid_col and team_col:
                for sid in names.keys():
                    player_ticks = ticks_df[ticks_df[steamid_col] == sid]
                    if not player_ticks.empty:
                        team = player_ticks[team_col].mode()
                        if len(team) > 0:
                            teams[sid] = safe_int(team.iloc[0])

        # Default teams based on kill patterns if not found
        if not teams and not kills_df.empty:
            # Try to infer from attacker/victim patterns (same team = teamkill)
            for sid in names.keys():
                teams[sid] = 0  # Unknown

        return names, teams

    def _calculate_stats(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        player_names: dict[int, str],
        player_teams: dict[int, int],
        num_rounds: int
    ) -> dict[int, dict]:
        """Calculate comprehensive player statistics."""
        stats: dict[int, dict] = {}
        num_rounds = max(num_rounds, 1)

        # Column finders
        def find_col(df: pd.DataFrame, options: list[str]) -> Optional[str]:
            for col in options:
                if col in df.columns:
                    return col
            return None

        # Kill columns
        att_col = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"]) if not kills_df.empty else None
        vic_col = find_col(kills_df, ["user_steamid", "victim_steamid"]) if not kills_df.empty else None
        assist_col = find_col(kills_df, ["assister_steamid", "assister_steam_id"]) if not kills_df.empty else None
        hs_col = find_col(kills_df, ["headshot"]) if not kills_df.empty else None
        weapon_col = find_col(kills_df, ["weapon"]) if not kills_df.empty else None

        # Damage columns
        dmg_att_col = find_col(damages_df, ["attacker_steamid", "attacker_steam_id"]) if not damages_df.empty else None
        dmg_col = find_col(damages_df, ["dmg_health", "damage", "dmg"]) if not damages_df.empty else None

        for steam_id, name in player_names.items():
            kills = 0
            deaths = 0
            assists = 0
            headshots = 0
            total_damage = 0
            weapon_kills: dict[str, int] = {}

            if not kills_df.empty:
                # Kills
                if att_col:
                    player_kills_df = kills_df[kills_df[att_col] == steam_id]
                    kills = len(player_kills_df)

                    # Headshots
                    if hs_col and kills > 0:
                        try:
                            headshots = int(player_kills_df[hs_col].sum())
                        except (ValueError, TypeError):
                            # Handle case where headshot is bool column
                            headshots = player_kills_df[hs_col].apply(lambda x: 1 if x else 0).sum()

                    # Weapon breakdown
                    if weapon_col and kills > 0:
                        weapon_kills = player_kills_df[weapon_col].value_counts().to_dict()

                # Deaths
                if vic_col:
                    deaths = len(kills_df[kills_df[vic_col] == steam_id])

                # Assists
                if assist_col:
                    assists = len(kills_df[kills_df[assist_col] == steam_id])

            # Damage
            if not damages_df.empty and dmg_att_col and dmg_col:
                player_dmg = damages_df[damages_df[dmg_att_col] == steam_id]
                try:
                    total_damage = int(player_dmg[dmg_col].sum())
                except (ValueError, TypeError):
                    total_damage = 0

            # Derived stats
            kd_ratio = round(kills / deaths, 2) if deaths > 0 else float(kills)
            hs_percent = round((headshots / kills * 100), 1) if kills > 0 else 0.0
            adr = round(total_damage / num_rounds, 1)

            # Team name
            team_num = player_teams.get(steam_id, 0)
            team_name = {2: "T", 3: "CT"}.get(team_num, "Unknown")

            stats[steam_id] = {
                "name": name,
                "team": team_name,
                "team_num": team_num,
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

    def _build_kill_events(self, kills_df: pd.DataFrame, ticks_df: Optional[pd.DataFrame]) -> list[KillEvent]:
        """Build structured kill events with position data if available."""
        kills = []
        if kills_df.empty:
            return kills

        def find_col(df: pd.DataFrame, options: list[str]) -> Optional[str]:
            for col in options:
                if col in df.columns:
                    return col
            return None

        att_col = find_col(kills_df, ["attacker_steamid", "attacker_steam_id"])
        att_name_col = find_col(kills_df, ["attacker_name"])
        vic_col = find_col(kills_df, ["user_steamid", "victim_steamid"])
        vic_name_col = find_col(kills_df, ["user_name", "victim_name"])
        weapon_col = find_col(kills_df, ["weapon"])
        hs_col = find_col(kills_df, ["headshot"])
        tick_col = find_col(kills_df, ["tick"])

        for _, row in kills_df.iterrows():
            tick = safe_int(row.get(tick_col)) if tick_col else 0
            att_id = safe_int(row.get(att_col)) if att_col else 0
            vic_id = safe_int(row.get(vic_col)) if vic_col else 0

            kill = KillEvent(
                tick=tick,
                attacker_steamid=att_id,
                attacker_name=safe_str(row.get(att_name_col)) if att_name_col else "",
                victim_steamid=vic_id,
                victim_name=safe_str(row.get(vic_name_col)) if vic_name_col else "",
                weapon=safe_str(row.get(weapon_col)) if weapon_col else "",
                headshot=safe_bool(row.get(hs_col)) if hs_col else False,
            )
            kills.append(kill)

        return kills

    def _build_damage_events(self, damages_df: pd.DataFrame) -> list[DamageEvent]:
        """Build structured damage events."""
        damages = []
        if damages_df.empty:
            return damages

        def find_col(df: pd.DataFrame, options: list[str]) -> Optional[str]:
            for col in options:
                if col in df.columns:
                    return col
            return None

        att_col = find_col(damages_df, ["attacker_steamid", "attacker_steam_id"])
        vic_col = find_col(damages_df, ["user_steamid", "victim_steamid"])
        dmg_col = find_col(damages_df, ["dmg_health", "damage", "dmg"])
        weapon_col = find_col(damages_df, ["weapon"])
        hitgroup_col = find_col(damages_df, ["hitgroup"])
        tick_col = find_col(damages_df, ["tick"])

        for _, row in damages_df.iterrows():
            dmg = DamageEvent(
                tick=safe_int(row.get(tick_col)) if tick_col else 0,
                attacker_steamid=safe_int(row.get(att_col)) if att_col else 0,
                victim_steamid=safe_int(row.get(vic_col)) if vic_col else 0,
                damage=safe_int(row.get(dmg_col)) if dmg_col else 0,
                weapon=safe_str(row.get(weapon_col)) if weapon_col else "",
                hitgroup=safe_str(row.get(hitgroup_col)) if hitgroup_col else "",
            )
            damages.append(dmg)

        return damages


def parse_demo(demo_path: str | Path) -> DemoData:
    """Convenience function to parse a demo file."""
    parser = DemoParser(demo_path)
    return parser.parse()
