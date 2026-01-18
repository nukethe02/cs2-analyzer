"""
Demo Parser for CS2 Replay Files

Uses demoparser2 directly for reliable data extraction with position/angle data.
Awpy is used as a fallback for basic parsing.

Key insight: To get position and angle data for crosshair placement analysis,
we must explicitly request player props when parsing events.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import logging

import pandas as pd
import numpy as np

# Try demoparser2 first (more control over what data we extract)
try:
    from demoparser2 import DemoParser as Demoparser2
    DEMOPARSER2_AVAILABLE = True
except ImportError:
    DEMOPARSER2_AVAILABLE = False

# awpy as fallback
try:
    from awpy import Demo
    AWPY_AVAILABLE = True
except ImportError:
    AWPY_AVAILABLE = False

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
    """A kill event with timing, position, and angle data."""
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
    # Attacker position and view angles
    attacker_x: Optional[float] = None
    attacker_y: Optional[float] = None
    attacker_z: Optional[float] = None
    attacker_pitch: Optional[float] = None
    attacker_yaw: Optional[float] = None
    # Victim position
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
    reason: str
    ct_score: int = 0
    t_score: int = 0


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
    player_teams: dict[int, str]

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
    """Parser for CS2 demo files using demoparser2 with position/angle data."""

    # Player properties to extract for position and angle data
    PLAYER_PROPS = ["X", "Y", "Z", "pitch", "yaw", "health", "is_alive"]

    def __init__(self, demo_path: str | Path):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        self._data: Optional[DemoData] = None

    def parse(self, include_ticks: bool = False) -> DemoData:
        """Parse the demo file and extract all relevant data."""
        if self._data is not None:
            return self._data

        if DEMOPARSER2_AVAILABLE:
            logger.info("Using demoparser2 with player props for position data")
            return self._parse_with_demoparser2(include_ticks)
        elif AWPY_AVAILABLE:
            logger.info("Using awpy parser (fallback)")
            return self._parse_with_awpy(include_ticks)
        else:
            raise ImportError("No parser available. Install demoparser2: pip install demoparser2")

    def _parse_with_demoparser2(self, include_ticks: bool = False) -> DemoData:
        """Parse using demoparser2 with explicit player prop extraction."""
        logger.info(f"Parsing demo: {self.demo_path}")
        parser = Demoparser2(str(self.demo_path))

        # Parse header
        map_name = "unknown"
        try:
            header = parser.parse_header()
            if isinstance(header, dict):
                map_name = header.get("map_name", "unknown")
                logger.info(f"Map: {map_name}")
        except Exception as e:
            logger.warning(f"Failed to parse header: {e}")

        # Parse kills WITH position and angle data
        kills_df = pd.DataFrame()
        try:
            # Request player props to get position/angle data embedded in events
            kills_df = parser.parse_event(
                "player_death",
                player=["X", "Y", "Z", "pitch", "yaw"],
                other=["total_rounds_played"]
            )
            if kills_df is not None and not kills_df.empty:
                logger.info(f"Parsed {len(kills_df)} kills with columns: {list(kills_df.columns)}")
            else:
                kills_df = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to parse kills with props: {e}")
            # Fallback to basic parsing
            try:
                kills_df = parser.parse_event("player_death")
                if kills_df is not None:
                    logger.info(f"Fallback: {len(kills_df)} kills, columns: {list(kills_df.columns)}")
            except Exception as e2:
                logger.warning(f"Failed basic kill parsing: {e2}")
                kills_df = pd.DataFrame()

        # Parse damages
        damages_df = pd.DataFrame()
        try:
            damages_df = parser.parse_event("player_hurt")
            if damages_df is not None and not damages_df.empty:
                logger.info(f"Parsed {len(damages_df)} damage events")
            else:
                damages_df = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to parse damages: {e}")
            damages_df = pd.DataFrame()

        # Parse round ends
        rounds_df = pd.DataFrame()
        try:
            rounds_df = parser.parse_event("round_end")
            if rounds_df is not None and not rounds_df.empty:
                logger.info(f"Parsed {len(rounds_df)} rounds")
            else:
                rounds_df = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to parse rounds: {e}")
            rounds_df = pd.DataFrame()

        # Parse ticks if requested (for detailed position tracking)
        ticks_df = None
        if include_ticks:
            try:
                ticks_df = parser.parse_ticks(self.PLAYER_PROPS)
                if ticks_df is not None and not ticks_df.empty:
                    logger.info(f"Parsed {len(ticks_df)} tick entries")
            except Exception as e:
                logger.warning(f"Failed to parse ticks: {e}")

        # Calculate duration
        tick_rate = 64
        max_tick = 0
        if not kills_df.empty and "tick" in kills_df.columns:
            max_tick = max(max_tick, int(kills_df["tick"].max()))
        duration_seconds = max_tick / tick_rate

        # Determine round count
        num_rounds = 1
        if not rounds_df.empty:
            num_rounds = len(rounds_df)
        elif not kills_df.empty:
            round_col = self._find_column(kills_df, ["total_rounds_played", "round", "round_num"])
            if round_col:
                num_rounds = int(kills_df[round_col].max())

        # Extract player info and calculate stats
        player_names, player_teams = self._extract_players(kills_df, damages_df)
        player_stats = self._calculate_stats(kills_df, damages_df, player_names, player_teams, num_rounds)

        # Build structured events
        kills = self._build_kills(kills_df)
        damages = self._build_damages(damages_df)
        rounds = self._build_rounds(rounds_df)

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

        logger.info(f"Parsing complete: {len(player_stats)} players, {num_rounds} rounds, {len(kills)} kills")
        return self._data

    def _find_column(self, df: pd.DataFrame, options: list[str]) -> Optional[str]:
        """Find first matching column from options."""
        for col in options:
            if col in df.columns:
                return col
        return None

    def _extract_players(self, kills_df: pd.DataFrame, damages_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
        """Extract player names and teams from DataFrames."""
        names: dict[int, str] = {}
        teams: dict[int, str] = {}

        # Column name variations
        att_id_cols = ["attacker_steamid", "attacker_steam_id"]
        att_name_cols = ["attacker_name"]
        att_team_cols = ["attacker_team_name", "attacker_side", "attacker_team"]
        vic_id_cols = ["user_steamid", "victim_steamid", "victim_steam_id"]
        vic_name_cols = ["user_name", "victim_name"]
        vic_team_cols = ["user_team_name", "victim_side", "victim_team"]

        def extract_from_df(df, id_cols, name_cols, team_cols):
            if df.empty:
                return
            id_col = self._find_column(df, id_cols)
            name_col = self._find_column(df, name_cols)
            team_col = self._find_column(df, team_cols)

            if id_col and name_col:
                for _, row in df.drop_duplicates(subset=[id_col]).iterrows():
                    sid = safe_int(row.get(id_col))
                    if sid and sid not in names:
                        names[sid] = safe_str(row.get(name_col))
                        if team_col:
                            team_val = row.get(team_col)
                            # Handle various team formats
                            if isinstance(team_val, str):
                                if "CT" in team_val.upper():
                                    teams[sid] = "CT"
                                elif "T" in team_val.upper() and "CT" not in team_val.upper():
                                    teams[sid] = "T"
                                else:
                                    teams[sid] = team_val
                            elif isinstance(team_val, (int, float)):
                                teams[sid] = "CT" if int(team_val) == 3 else "T" if int(team_val) == 2 else "Unknown"
                            else:
                                teams[sid] = "Unknown"

        # Extract from kills (attackers)
        extract_from_df(kills_df, att_id_cols, att_name_cols, att_team_cols)
        # Extract from kills (victims)
        extract_from_df(kills_df, vic_id_cols, vic_name_cols, vic_team_cols)
        # Extract from damages
        extract_from_df(damages_df, att_id_cols, att_name_cols, att_team_cols)
        extract_from_df(damages_df, vic_id_cols, vic_name_cols, vic_team_cols)

        return names, teams

    def _calculate_stats(
        self,
        kills_df: pd.DataFrame,
        damages_df: pd.DataFrame,
        player_names: dict[int, str],
        player_teams: dict[int, str],
        num_rounds: int
    ) -> dict[int, dict]:
        """Calculate player statistics."""
        stats: dict[int, dict] = {}
        num_rounds = max(num_rounds, 1)

        att_col = self._find_column(kills_df, ["attacker_steamid", "attacker_steam_id"])
        vic_col = self._find_column(kills_df, ["user_steamid", "victim_steamid", "victim_steam_id"])
        hs_col = self._find_column(kills_df, ["headshot"])
        weapon_col = self._find_column(kills_df, ["weapon"])
        assist_col = self._find_column(kills_df, ["assister_steamid", "assister_steam_id"])
        dmg_att_col = self._find_column(damages_df, ["attacker_steamid", "attacker_steam_id"])
        dmg_col = self._find_column(damages_df, ["dmg_health", "damage", "dmg"])

        for steam_id, name in player_names.items():
            kills = 0
            deaths = 0
            assists = 0
            headshots = 0
            total_damage = 0
            weapon_kills: dict[str, int] = {}

            if not kills_df.empty and att_col:
                player_kills_df = kills_df[kills_df[att_col] == steam_id]
                kills = len(player_kills_df)
                if hs_col and kills > 0:
                    headshots = int(player_kills_df[hs_col].sum())
                if weapon_col and kills > 0:
                    weapon_kills = player_kills_df[weapon_col].value_counts().to_dict()

            if not kills_df.empty and vic_col:
                deaths = len(kills_df[kills_df[vic_col] == steam_id])

            if not kills_df.empty and assist_col:
                assists = len(kills_df[kills_df[assist_col] == steam_id])

            if not damages_df.empty and dmg_att_col and dmg_col:
                player_dmg = damages_df[damages_df[dmg_att_col] == steam_id]
                total_damage = int(player_dmg[dmg_col].sum())

            stats[steam_id] = {
                "name": name,
                "team": player_teams.get(steam_id, "Unknown"),
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "headshots": headshots,
                "hs_percent": round((headshots / kills * 100), 1) if kills > 0 else 0.0,
                "total_damage": total_damage,
                "adr": round(total_damage / num_rounds, 1),
                "kd_ratio": round(kills / deaths, 2) if deaths > 0 else float(kills),
                "weapon_kills": weapon_kills,
            }

        return stats

    def _build_kills(self, kills_df: pd.DataFrame) -> list[KillEvent]:
        """Build KillEvent list from kills DataFrame."""
        kills = []
        if kills_df.empty:
            return kills

        # Find columns
        att_id = self._find_column(kills_df, ["attacker_steamid", "attacker_steam_id"])
        att_name = self._find_column(kills_df, ["attacker_name"])
        att_team = self._find_column(kills_df, ["attacker_team_name", "attacker_side"])
        vic_id = self._find_column(kills_df, ["user_steamid", "victim_steamid"])
        vic_name = self._find_column(kills_df, ["user_name", "victim_name"])
        vic_team = self._find_column(kills_df, ["user_team_name", "victim_side"])
        round_col = self._find_column(kills_df, ["total_rounds_played", "round", "round_num"])

        # Position columns (from player props)
        att_x = self._find_column(kills_df, ["attacker_X", "attacker_x"])
        att_y = self._find_column(kills_df, ["attacker_Y", "attacker_y"])
        att_z = self._find_column(kills_df, ["attacker_Z", "attacker_z"])
        att_pitch = self._find_column(kills_df, ["attacker_pitch"])
        att_yaw = self._find_column(kills_df, ["attacker_yaw"])
        vic_x = self._find_column(kills_df, ["user_X", "victim_X", "user_x", "victim_x"])
        vic_y = self._find_column(kills_df, ["user_Y", "victim_Y", "user_y", "victim_y"])
        vic_z = self._find_column(kills_df, ["user_Z", "victim_Z", "user_z", "victim_z"])

        logger.info(f"Position columns found - attacker: X={att_x}, Y={att_y}, Z={att_z}, pitch={att_pitch}, yaw={att_yaw}")
        logger.info(f"Position columns found - victim: X={vic_x}, Y={vic_y}, Z={vic_z}")

        for _, row in kills_df.iterrows():
            # Get team values
            att_side = "Unknown"
            if att_team:
                att_side_val = row.get(att_team)
                if isinstance(att_side_val, str):
                    att_side = "CT" if "CT" in att_side_val.upper() else "T" if "T" in att_side_val.upper() else att_side_val
                elif isinstance(att_side_val, (int, float)):
                    att_side = "CT" if int(att_side_val) == 3 else "T" if int(att_side_val) == 2 else "Unknown"

            vic_side = "Unknown"
            if vic_team:
                vic_side_val = row.get(vic_team)
                if isinstance(vic_side_val, str):
                    vic_side = "CT" if "CT" in vic_side_val.upper() else "T" if "T" in vic_side_val.upper() else vic_side_val
                elif isinstance(vic_side_val, (int, float)):
                    vic_side = "CT" if int(vic_side_val) == 3 else "T" if int(vic_side_val) == 2 else "Unknown"

            kill = KillEvent(
                tick=safe_int(row.get("tick")),
                round_num=safe_int(row.get(round_col)) if round_col else 0,
                attacker_steamid=safe_int(row.get(att_id)) if att_id else 0,
                attacker_name=safe_str(row.get(att_name)) if att_name else "",
                attacker_side=att_side,
                victim_steamid=safe_int(row.get(vic_id)) if vic_id else 0,
                victim_name=safe_str(row.get(vic_name)) if vic_name else "",
                victim_side=vic_side,
                weapon=safe_str(row.get("weapon", "")),
                headshot=safe_bool(row.get("headshot")),
                assister_steamid=safe_int(row.get("assister_steamid")) if row.get("assister_steamid") else None,
                assister_name=safe_str(row.get("assister_name")) if row.get("assister_name") else None,
                flash_assist=safe_bool(row.get("flash_assist")),
                # Position data
                attacker_x=safe_float(row.get(att_x)) if att_x and row.get(att_x) is not None else None,
                attacker_y=safe_float(row.get(att_y)) if att_y and row.get(att_y) is not None else None,
                attacker_z=safe_float(row.get(att_z)) if att_z and row.get(att_z) is not None else None,
                attacker_pitch=safe_float(row.get(att_pitch)) if att_pitch and row.get(att_pitch) is not None else None,
                attacker_yaw=safe_float(row.get(att_yaw)) if att_yaw and row.get(att_yaw) is not None else None,
                victim_x=safe_float(row.get(vic_x)) if vic_x and row.get(vic_x) is not None else None,
                victim_y=safe_float(row.get(vic_y)) if vic_y and row.get(vic_y) is not None else None,
                victim_z=safe_float(row.get(vic_z)) if vic_z and row.get(vic_z) is not None else None,
            )
            kills.append(kill)

        # Log position data availability
        kills_with_pos = sum(1 for k in kills if k.attacker_x is not None)
        logger.info(f"Built {len(kills)} kill events, {kills_with_pos} have position data")

        return kills

    def _build_damages(self, damages_df: pd.DataFrame) -> list[DamageEvent]:
        """Build DamageEvent list from damages DataFrame."""
        damages = []
        if damages_df.empty:
            return damages

        att_id = self._find_column(damages_df, ["attacker_steamid", "attacker_steam_id"])
        att_name = self._find_column(damages_df, ["attacker_name"])
        att_team = self._find_column(damages_df, ["attacker_team_name", "attacker_side"])
        vic_id = self._find_column(damages_df, ["user_steamid", "victim_steamid"])
        vic_name = self._find_column(damages_df, ["user_name", "victim_name"])
        vic_team = self._find_column(damages_df, ["user_team_name", "victim_side"])
        dmg_col = self._find_column(damages_df, ["dmg_health", "damage", "dmg"])
        round_col = self._find_column(damages_df, ["total_rounds_played", "round", "round_num"])

        for _, row in damages_df.iterrows():
            att_side = "Unknown"
            if att_team:
                att_side_val = row.get(att_team)
                if isinstance(att_side_val, str):
                    att_side = "CT" if "CT" in att_side_val.upper() else "T" if "T" in att_side_val.upper() else att_side_val

            vic_side = "Unknown"
            if vic_team:
                vic_side_val = row.get(vic_team)
                if isinstance(vic_side_val, str):
                    vic_side = "CT" if "CT" in vic_side_val.upper() else "T" if "T" in vic_side_val.upper() else vic_side_val

            dmg = DamageEvent(
                tick=safe_int(row.get("tick")),
                round_num=safe_int(row.get(round_col)) if round_col else 0,
                attacker_steamid=safe_int(row.get(att_id)) if att_id else 0,
                attacker_name=safe_str(row.get(att_name)) if att_name else "",
                attacker_side=att_side,
                victim_steamid=safe_int(row.get(vic_id)) if vic_id else 0,
                victim_name=safe_str(row.get(vic_name)) if vic_name else "",
                victim_side=vic_side,
                damage=safe_int(row.get(dmg_col)) if dmg_col else 0,
                weapon=safe_str(row.get("weapon", "")),
                hitgroup=safe_str(row.get("hitgroup", "generic")),
            )
            damages.append(dmg)

        return damages

    def _build_rounds(self, rounds_df: pd.DataFrame) -> list[RoundInfo]:
        """Build RoundInfo list from rounds DataFrame."""
        rounds = []
        if rounds_df.empty:
            return rounds

        for idx, row in rounds_df.iterrows():
            # Determine winner from reason or winner column
            winner = "Unknown"
            reason = safe_str(row.get("reason", ""))
            winner_col = row.get("winner")

            if winner_col is not None:
                if isinstance(winner_col, str):
                    winner = "CT" if "CT" in winner_col.upper() else "T" if "T" in winner_col.upper() else winner_col
                elif isinstance(winner_col, (int, float)):
                    winner = "CT" if int(winner_col) == 3 else "T" if int(winner_col) == 2 else "Unknown"
            elif reason:
                # Infer from reason
                ct_reasons = ["bomb_defused", "ct_win", "target_saved"]
                t_reasons = ["target_bombed", "terrorist_win", "t_win"]
                reason_lower = reason.lower()
                if any(r in reason_lower for r in ct_reasons):
                    winner = "CT"
                elif any(r in reason_lower for r in t_reasons):
                    winner = "T"

            round_info = RoundInfo(
                round_num=idx + 1,
                start_tick=safe_int(row.get("start_tick", 0)),
                end_tick=safe_int(row.get("tick", row.get("end_tick", 0))),
                winner=winner,
                reason=reason,
                ct_score=safe_int(row.get("ct_score", 0)),
                t_score=safe_int(row.get("t_score", 0)),
            )
            rounds.append(round_info)

        return rounds

    def _parse_with_awpy(self, include_ticks: bool = False) -> DemoData:
        """Fallback parser using awpy."""
        logger.info(f"Parsing demo with awpy: {self.demo_path}")

        demo = Demo(str(self.demo_path), verbose=False)

        # Parse with player props if we want tick data
        if include_ticks:
            demo.parse(player_props=self.PLAYER_PROPS)
        else:
            demo.parse()

        header = demo.header or {}
        map_name = header.get("map_name", "unknown")

        # Convert Polars to Pandas
        kills_df = demo.kills.to_pandas() if demo.kills is not None else pd.DataFrame()
        damages_df = demo.damages.to_pandas() if demo.damages is not None else pd.DataFrame()
        rounds_df = demo.rounds.to_pandas() if demo.rounds is not None else pd.DataFrame()
        ticks_df = demo.ticks.to_pandas() if include_ticks and demo.ticks is not None else None

        logger.info(f"awpy parsed: {len(kills_df)} kills, {len(damages_df)} damages, {len(rounds_df)} rounds")
        if not kills_df.empty:
            logger.info(f"Kill columns: {list(kills_df.columns)}")

        # Calculate stats
        tick_rate = 64
        max_tick = 0
        if not kills_df.empty and "tick" in kills_df.columns:
            max_tick = int(kills_df["tick"].max())
        duration_seconds = max_tick / tick_rate

        num_rounds = len(rounds_df) if not rounds_df.empty else 1

        player_names, player_teams = self._extract_players(kills_df, damages_df)
        player_stats = self._calculate_stats(kills_df, damages_df, player_names, player_teams, num_rounds)
        kills = self._build_kills(kills_df)
        damages = self._build_damages(damages_df)
        rounds = self._build_rounds(rounds_df)

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

        return self._data


def parse_demo(demo_path: str | Path, include_ticks: bool = False) -> DemoData:
    """Convenience function to parse a demo file."""
    parser = DemoParser(demo_path)
    return parser.parse(include_ticks=include_ticks)
