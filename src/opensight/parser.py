"""
Demo Parser Wrapper for CS2 Replay Files

Wraps demoparser2 to extract tick-level game data from .dem files,
providing structured access to player positions, events, and game state.
"""

from dataclasses import dataclass, field
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
class PlayerState:
    """Player state at a specific tick."""

    steam_id: int
    name: str
    team: str
    position: tuple[float, float, float]  # x, y, z
    view_angles: tuple[float, float]  # pitch, yaw
    health: int
    armor: int
    is_alive: bool
    weapon: str
    tick: int


@dataclass
class GameEvent:
    """A game event extracted from the demo."""

    event_type: str
    tick: int
    data: dict[str, Any]


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

    # Tick-level data as DataFrames
    player_positions: pd.DataFrame  # tick, steam_id, x, y, z, pitch, yaw
    player_health: pd.DataFrame  # tick, steam_id, health, armor
    shots_fired: pd.DataFrame  # tick, steam_id, weapon, hit, target_id
    damage_events: pd.DataFrame  # tick, attacker_id, victim_id, damage, weapon, hitgroup
    kill_events: pd.DataFrame  # tick, attacker_id, victim_id, weapon, headshot

    # Round data
    round_starts: list[int]  # ticks where rounds start
    round_ends: list[int]  # ticks where rounds end

    @property
    def tick_interval(self) -> float:
        """Seconds per tick."""
        return 1.0 / self.tick_rate if self.tick_rate > 0 else 0.0

    def get_player_state_at_tick(self, steam_id: int, tick: int) -> Optional[PlayerState]:
        """Get a player's state at a specific tick."""
        pos_row = self.player_positions[
            (self.player_positions["tick"] == tick) &
            (self.player_positions["steam_id"] == steam_id)
        ]
        health_row = self.player_health[
            (self.player_health["tick"] == tick) &
            (self.player_health["steam_id"] == steam_id)
        ]

        if pos_row.empty:
            return None

        row = pos_row.iloc[0]
        health = health_row.iloc[0]["health"] if not health_row.empty else 0
        armor = health_row.iloc[0]["armor"] if not health_row.empty else 0

        return PlayerState(
            steam_id=steam_id,
            name=self.player_names.get(steam_id, "Unknown"),
            team=self.teams.get(steam_id, "Unknown"),
            position=(row["x"], row["y"], row["z"]),
            view_angles=(row.get("pitch", 0), row.get("yaw", 0)),
            health=health,
            armor=armor,
            is_alive=health > 0,
            weapon=row.get("weapon", "unknown"),
            tick=tick,
        )


class DemoParser:
    """
    Parser for CS2 demo files.

    Wraps demoparser2 for efficient tick-level data extraction with
    a high-level interface for analytics calculations.
    """

    def __init__(self, demo_path: str | Path):
        """
        Initialize the parser with a demo file path.

        Args:
            demo_path: Path to the .dem file
        """
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        if not self.demo_path.suffix.lower() == ".dem":
            raise ValueError(f"Expected .dem file, got: {self.demo_path.suffix}")

        self._parser: Optional[Any] = None
        self._data: Optional[DemoData] = None

    def parse(self) -> DemoData:
        """
        Parse the demo file and extract all relevant data.

        Returns:
            DemoData containing all extracted information
        """
        if self._data is not None:
            return self._data

        if Demoparser2 is None:
            raise ImportError(
                "demoparser2 is required but not installed. "
                "Install with: pip install demoparser2"
            )

        logger.info(f"Parsing demo: {self.demo_path}")
        self._parser = Demoparser2(str(self.demo_path))

        # Extract header info
        header = self._parser.parse_header()
        # Header might be a dict or have different structure
        if isinstance(header, dict):
            map_name = header.get("map_name", "unknown")
            tick_rate = int(header.get("tickrate", 64))
        else:
            # Try to access as attributes or default
            map_name = getattr(header, "map_name", "unknown") if header else "unknown"
            tick_rate = int(getattr(header, "tickrate", 64)) if header else 64

        # Parse player info - might be DataFrame or list
        player_info = self._parser.parse_player_info()
        player_names = {}
        if isinstance(player_info, pd.DataFrame):
            for _, row in player_info.iterrows():
                sid = row.get("steamid") or row.get("steam_id") or 0
                name = row.get("name", "Unknown")
                if sid:
                    player_names[int(sid)] = str(name)
        elif isinstance(player_info, list):
            for p in player_info:
                if isinstance(p, dict):
                    sid = p.get("steamid") or p.get("steam_id") or 0
                    name = p.get("name", "Unknown")
                    if sid:
                        player_names[int(sid)] = str(name)
        elif hasattr(player_info, 'iterrows'):
            for _, row in player_info.iterrows():
                sid = row.get("steamid") or row.get("steam_id") or 0
                name = row.get("name", "Unknown")
                if sid:
                    player_names[int(sid)] = str(name)

        # Parse tick-level data
        # Request specific fields for efficiency
        tick_fields = [
            "tick",
            "steamid",
            "X",
            "Y",
            "Z",
            "pitch",
            "yaw",
            "health",
            "armor_value",
            "team_num",
            "active_weapon_name",
        ]

        try:
            ticks_df = self._parser.parse_ticks(tick_fields)
        except Exception as e:
            logger.warning(f"Failed to parse ticks with all fields: {e}")
            # Try with minimal fields
            ticks_df = self._parser.parse_ticks(["tick", "steamid", "X", "Y", "Z"])

        # Normalize column names
        ticks_df = ticks_df.rename(columns={
            "steamid": "steam_id",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "armor_value": "armor",
            "team_num": "team",
            "active_weapon_name": "weapon",
        })

        # Extract teams mapping
        teams = {}
        if "team" in ticks_df.columns:
            team_mapping = ticks_df.groupby("steam_id")["team"].last().to_dict()
            teams = {k: "CT" if v == 3 else "T" if v == 2 else "Unknown" for k, v in team_mapping.items()}

        # Build position DataFrame
        pos_cols = ["tick", "steam_id", "x", "y", "z"]
        if "pitch" in ticks_df.columns:
            pos_cols.extend(["pitch", "yaw"])
        player_positions = ticks_df[pos_cols].copy()

        # Build health DataFrame
        health_cols = ["tick", "steam_id"]
        if "health" in ticks_df.columns:
            health_cols.append("health")
        if "armor" in ticks_df.columns:
            health_cols.append("armor")
        player_health = ticks_df[health_cols].copy() if len(health_cols) > 2 else pd.DataFrame()

        # Parse game events
        try:
            events = self._parser.parse_events(["player_death", "player_hurt", "weapon_fire"])
            # Events might be a dict of DataFrames or dict of lists
            if isinstance(events, dict):
                for key in events:
                    if isinstance(events[key], pd.DataFrame):
                        events[key] = events[key].to_dict('records')
        except Exception as e:
            logger.warning(f"Failed to parse events: {e}")
            events = {}

        # Process damage events
        damage_events = self._process_damage_events(events.get("player_hurt", []))

        # Process kill events
        kill_events = self._process_kill_events(events.get("player_death", []))

        # Process shots fired
        shots_fired = self._process_shots(events.get("weapon_fire", []))

        # Parse round events
        round_starts = []
        round_ends = []
        try:
            round_events = self._parser.parse_events(["round_start", "round_end"])
            if isinstance(round_events, dict):
                rs = round_events.get("round_start", [])
                re = round_events.get("round_end", [])
                # Convert DataFrames if needed
                if isinstance(rs, pd.DataFrame):
                    rs = rs.to_dict('records')
                if isinstance(re, pd.DataFrame):
                    re = re.to_dict('records')
                round_starts = [e.get("tick", 0) for e in rs if isinstance(e, dict)]
                round_ends = [e.get("tick", 0) for e in re if isinstance(e, dict)]
        except Exception as e:
            logger.warning(f"Failed to parse round events: {e}")

        # Calculate duration
        max_tick = int(ticks_df["tick"].max()) if not ticks_df.empty else 0
        duration_seconds = max_tick / tick_rate if tick_rate > 0 else 0

        self._data = DemoData(
            file_path=self.demo_path,
            map_name=map_name,
            tick_rate=tick_rate,
            duration_ticks=max_tick,
            duration_seconds=duration_seconds,
            player_names=player_names,
            teams=teams,
            player_positions=player_positions,
            player_health=player_health,
            shots_fired=shots_fired,
            damage_events=damage_events,
            kill_events=kill_events,
            round_starts=round_starts,
            round_ends=round_ends,
        )

        logger.info(
            f"Parsed {max_tick} ticks ({duration_seconds:.1f}s) on {map_name}"
        )
        return self._data

    def _process_damage_events(self, events: list[dict]) -> pd.DataFrame:
        """Process player_hurt events into a DataFrame."""
        if not events:
            return pd.DataFrame(columns=[
                "tick", "attacker_id", "victim_id", "damage", "weapon", "hitgroup"
            ])

        records = []
        for event in events:
            records.append({
                "tick": event.get("tick", 0),
                "attacker_id": event.get("attacker_steamid", 0),
                "victim_id": event.get("userid_steamid", 0),
                "damage": event.get("dmg_health", 0),
                "weapon": event.get("weapon", "unknown"),
                "hitgroup": event.get("hitgroup", 0),
            })
        return pd.DataFrame(records)

    def _process_kill_events(self, events: list[dict]) -> pd.DataFrame:
        """Process player_death events into a DataFrame."""
        if not events:
            return pd.DataFrame(columns=[
                "tick", "attacker_id", "victim_id", "weapon", "headshot"
            ])

        records = []
        for event in events:
            records.append({
                "tick": event.get("tick", 0),
                "attacker_id": event.get("attacker_steamid", 0),
                "victim_id": event.get("userid_steamid", 0),
                "weapon": event.get("weapon", "unknown"),
                "headshot": event.get("headshot", False),
            })
        return pd.DataFrame(records)

    def _process_shots(self, events: list[dict]) -> pd.DataFrame:
        """Process weapon_fire events into a DataFrame."""
        if not events:
            return pd.DataFrame(columns=["tick", "steam_id", "weapon"])

        records = []
        for event in events:
            records.append({
                "tick": event.get("tick", 0),
                "steam_id": event.get("userid_steamid", 0),
                "weapon": event.get("weapon", "unknown"),
            })
        return pd.DataFrame(records)

    def get_player_positions_between(
        self,
        start_tick: int,
        end_tick: int,
        steam_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get player positions between two ticks.

        Args:
            start_tick: Starting tick (inclusive)
            end_tick: Ending tick (inclusive)
            steam_id: Optional filter for specific player

        Returns:
            DataFrame with position data
        """
        data = self.parse()
        mask = (data.player_positions["tick"] >= start_tick) & \
               (data.player_positions["tick"] <= end_tick)

        if steam_id is not None:
            mask &= data.player_positions["steam_id"] == steam_id

        return data.player_positions[mask].copy()


def parse_demo(demo_path: str | Path) -> DemoData:
    """
    Convenience function to parse a demo file.

    Args:
        demo_path: Path to the .dem file

    Returns:
        DemoData containing all extracted information
    """
    parser = DemoParser(demo_path)
    return parser.parse()
