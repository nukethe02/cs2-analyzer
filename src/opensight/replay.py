"""
2D Replay Viewer Module for CS2 Demo Visualization

Provides:
- Tick-level player position extraction
- Frame-by-frame replay data generation
- Round-based replay segments
- Event markers (kills, grenades, bomb)
- Player POV tracking
- Multi-POV synchronization data

The replay viewer generates JSON data that can be rendered by a frontend
canvas/WebGL renderer for interactive 2D replay playback.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Tick sampling rate (extract every Nth tick for performance)
DEFAULT_TICK_SAMPLE_RATE = 8  # 64 tick / 8 = 8 frames per second
HIGH_QUALITY_SAMPLE_RATE = 4  # 16 fps
FULL_QUALITY_SAMPLE_RATE = 1  # 64 fps (full data, large file)


@dataclass
class PlayerFrame:
    """Player state at a single tick."""

    steam_id: int
    name: str
    team: str  # "CT" or "T"
    x: float
    y: float
    z: float
    yaw: float  # View direction (0-360)
    pitch: float
    health: int
    armor: int
    is_alive: bool
    is_scoped: bool
    is_crouching: bool
    active_weapon: str = ""
    money: int = 0
    equipment_value: int = 0

    def to_dict(self) -> dict:
        return {
            "sid": self.steam_id,
            "n": self.name,
            "t": self.team,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "z": round(self.z, 1),
            "yaw": round(self.yaw, 1),
            "hp": self.health,
            "alive": self.is_alive,
            "scoped": self.is_scoped,
            "crouch": self.is_crouching,
            "weapon": self.active_weapon,
        }


@dataclass
class GrenadeFrame:
    """Grenade state at a single tick."""

    grenade_id: int
    grenade_type: str  # "flashbang", "smoke", "he", "molotov", "decoy"
    x: float
    y: float
    z: float
    thrower_steam_id: int
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.grenade_id,
            "type": self.grenade_type,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "z": round(self.z, 1),
            "thrower": self.thrower_steam_id,
            "active": self.is_active,
        }


@dataclass
class BombFrame:
    """Bomb state at a single tick."""

    x: float
    y: float
    z: float
    state: str  # "carried", "dropped", "planting", "planted", "defusing", "exploded", "defused"
    carrier_steam_id: int | None = None
    plant_progress: float = 0.0  # 0-100%
    defuse_progress: float = 0.0  # 0-100%
    time_remaining: float = 0.0  # Seconds until explosion

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "state": self.state,
            "carrier": self.carrier_steam_id,
            "plant_pct": round(self.plant_progress, 1),
            "defuse_pct": round(self.defuse_progress, 1),
            "timer": round(self.time_remaining, 1),
        }


@dataclass
class KillEvent:
    """A kill event for replay markers."""

    tick: int
    round_num: int
    attacker_steam_id: int
    attacker_name: str
    victim_steam_id: int
    victim_name: str
    weapon: str
    headshot: bool
    x: float  # Position where kill happened
    y: float

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "attacker": self.attacker_name,
            "victim": self.victim_name,
            "weapon": self.weapon,
            "hs": self.headshot,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
        }


@dataclass
class ReplayFrame:
    """A single frame of replay data."""

    tick: int
    round_num: int
    game_time: float  # Seconds since round start
    players: list[PlayerFrame]
    grenades: list[GrenadeFrame] = field(default_factory=list)
    bomb: BombFrame | None = None
    events: list[dict] = field(default_factory=list)  # Kill events, etc.

    def to_dict(self) -> dict:
        result = {
            "tick": self.tick,
            "round": self.round_num,
            "time": round(self.game_time, 2),
            "players": [p.to_dict() for p in self.players],
        }

        if self.grenades:
            result["grenades"] = [g.to_dict() for g in self.grenades]

        if self.bomb:
            result["bomb"] = self.bomb.to_dict()

        if self.events:
            result["events"] = self.events

        return result


@dataclass
class RoundReplay:
    """Replay data for a single round."""

    round_num: int
    start_tick: int
    end_tick: int
    winner: str  # "CT" or "T"
    win_reason: str
    ct_score: int
    t_score: int
    frames: list[ReplayFrame] = field(default_factory=list)
    kills: list[KillEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "round_num": self.round_num,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "ct_score": self.ct_score,
            "t_score": self.t_score,
            "frame_count": len(self.frames),
            "kills": [k.to_dict() for k in self.kills],
            "frames": [f.to_dict() for f in self.frames],
        }

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def duration_seconds(self) -> float:
        return self.duration_ticks / 64  # Assuming 64 tick


@dataclass
class MatchReplay:
    """Complete replay data for a match."""

    map_name: str
    tick_rate: int
    sample_rate: int
    total_rounds: int
    team1_score: int
    team2_score: int
    player_names: dict[int, str]
    player_teams: dict[int, str]
    rounds: list[RoundReplay] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "map_name": self.map_name,
            "tick_rate": self.tick_rate,
            "sample_rate": self.sample_rate,
            "fps": self.tick_rate // self.sample_rate,
            "total_rounds": self.total_rounds,
            "score": f"{self.team1_score} - {self.team2_score}",
            "players": self.player_names,
            "teams": self.player_teams,
            "rounds": [r.to_dict() for r in self.rounds],
        }

    def get_round(self, round_num: int) -> RoundReplay | None:
        """Get replay data for a specific round."""
        for r in self.rounds:
            if r.round_num == round_num:
                return r
        return None


class ReplayGenerator:
    """
    Generates 2D replay data from parsed demo data.

    Requires tick-level player state data from the parser.
    """

    def __init__(
        self,
        demo_data,  # DemoData from parser
        sample_rate: int = DEFAULT_TICK_SAMPLE_RATE,
    ):
        """
        Initialize the replay generator.

        Args:
            demo_data: Parsed demo data with tick-level player states
            sample_rate: Extract every Nth tick (higher = smaller file, lower fps)
        """
        self.data = demo_data
        self.sample_rate = sample_rate
        self.tick_rate = demo_data.tick_rate or 64

        # Index player ticks by round for fast lookup
        self._player_ticks_by_round: dict[int, list] = defaultdict(list)
        self._index_player_ticks()

        # Index kills by round
        self._kills_by_round: dict[int, list] = defaultdict(list)
        self._index_kills()

    def _index_player_ticks(self):
        """Index player tick data by round number."""
        if not hasattr(self.data, 'ticks_df') or self.data.ticks_df is None or self.data.ticks_df.empty:
            logger.warning("No tick-level player data available for replay")
            return

        for _, tick_data in self.data.ticks_df.iterrows():
            round_num = tick_data.get('round_num', tick_data.get('round', 0))
            self._player_ticks_by_round[round_num].append(tick_data)

        # Sort by tick
        for round_num in self._player_ticks_by_round:
            self._player_ticks_by_round[round_num].sort(key=lambda t: t.get('tick', 0))

        logger.info(f"Indexed {len(self.data.ticks_df)} player ticks across {len(self._player_ticks_by_round)} rounds")

    def _index_kills(self):
        """Index kills by round number."""
        for kill in self.data.kills:
            self._kills_by_round[kill.round_num].append(kill)

    def generate_full_replay(self) -> MatchReplay:
        """
        Generate complete replay data for the match.

        Returns:
            MatchReplay with all rounds
        """
        logger.info(f"Generating full replay with sample rate {self.sample_rate}")

        replay = MatchReplay(
            map_name=self.data.map_name,
            tick_rate=self.tick_rate,
            sample_rate=self.sample_rate,
            total_rounds=self.data.num_rounds,
            team1_score=0,
            team2_score=0,
            player_names=self.data.player_names.copy(),
            player_teams={
                sid: "CT" if team == 3 else "T" for sid, team in self.data.player_teams.items()
            },
        )

        # Generate each round
        for round_info in self.data.rounds:
            round_replay = self._generate_round_replay(round_info)
            if round_replay:
                replay.rounds.append(round_replay)

                # Update scores
                if round_replay.winner == "CT":
                    replay.team1_score += 1
                else:
                    replay.team2_score += 1

        logger.info(f"Generated replay with {len(replay.rounds)} rounds")
        return replay

    def generate_round_replay(self, round_num: int) -> RoundReplay | None:
        """
        Generate replay data for a specific round.

        Args:
            round_num: Round number to generate

        Returns:
            RoundReplay or None if round not found
        """
        for round_info in self.data.rounds:
            if round_info.round_num == round_num:
                return self._generate_round_replay(round_info)
        return None

    def _generate_round_replay(self, round_info) -> RoundReplay | None:
        """Generate replay for a single round."""
        round_num = round_info.round_num

        # Get tick data for this round
        tick_data = self._player_ticks_by_round.get(round_num, [])
        if not tick_data:
            logger.debug(f"No tick data for round {round_num}")
            return None

        # Get kills for this round
        round_kills = self._kills_by_round.get(round_num, [])

        # Create replay
        replay = RoundReplay(
            round_num=round_num,
            start_tick=round_info.start_tick,
            end_tick=round_info.end_tick,
            winner=round_info.winner or "Unknown",
            win_reason=round_info.reason or "unknown",
            ct_score=round_info.ct_score,
            t_score=round_info.t_score,
        )

        # Add kill events
        for kill in round_kills:
            replay.kills.append(
                KillEvent(
                    tick=kill.tick,
                    round_num=round_num,
                    attacker_steam_id=kill.attacker_steamid,
                    attacker_name=self.data.player_names.get(kill.attacker_steamid, "Unknown"),
                    victim_steam_id=kill.victim_steamid,
                    victim_name=self.data.player_names.get(kill.victim_steamid, "Unknown"),
                    weapon=kill.weapon,
                    headshot=kill.headshot,
                    x=kill.victim_x or 0,
                    y=kill.victim_y or 0,
                )
            )

        # Group tick data by tick number
        ticks_by_number: dict[int, list] = defaultdict(list)
        for td in tick_data:
            ticks_by_number[td.tick].append(td)

        # Sample ticks and create frames
        sorted_ticks = sorted(ticks_by_number.keys())
        round_start_tick = round_info.start_tick

        for i, tick in enumerate(sorted_ticks):
            # Only include sampled ticks
            if i % self.sample_rate != 0:
                continue

            players_at_tick = ticks_by_number[tick]
            player_frames = []

            for pd in players_at_tick:
                pf = PlayerFrame(
                    steam_id=pd.steamid,
                    name=pd.name,
                    team="CT" if pd.side == "CT" else "T",
                    x=pd.x,
                    y=pd.y,
                    z=pd.z,
                    yaw=pd.yaw,
                    pitch=pd.pitch,
                    health=pd.health,
                    armor=pd.armor,
                    is_alive=pd.is_alive,
                    is_scoped=pd.is_scoped,
                    is_crouching=pd.is_crouching,
                    money=pd.money,
                    equipment_value=pd.equipment_value,
                )
                player_frames.append(pf)

            # Calculate game time
            game_time = (tick - round_start_tick) / self.tick_rate

            # Check for events at this tick (kills)
            events = []
            for kill in round_kills:
                if abs(kill.tick - tick) < self.sample_rate:
                    events.append(
                        {
                            "type": "kill",
                            "attacker": self.data.player_names.get(
                                kill.attacker_steamid, "Unknown"
                            ),
                            "victim": self.data.player_names.get(kill.victim_steamid, "Unknown"),
                            "weapon": kill.weapon,
                            "headshot": kill.headshot,
                        }
                    )

            frame = ReplayFrame(
                tick=tick,
                round_num=round_num,
                game_time=game_time,
                players=player_frames,
                events=events,
            )
            replay.frames.append(frame)

        logger.debug(f"Generated {len(replay.frames)} frames for round {round_num}")
        return replay


class ReplayExporter:
    """
    Exports replay data to various formats.
    """

    @staticmethod
    def to_json(replay: MatchReplay, pretty: bool = False) -> str:
        """Export replay to JSON string."""
        import json

        return json.dumps(
            replay.to_dict(),
            indent=2 if pretty else None,
            ensure_ascii=False,
        )

    @staticmethod
    def to_file(replay: MatchReplay, path: Path, pretty: bool = False):
        """Export replay to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(replay.to_dict(), f, indent=2 if pretty else None)
        logger.info(f"Exported replay to {path}")

    @staticmethod
    def to_compressed(replay: MatchReplay, path: Path):
        """Export replay to compressed JSON file."""
        import gzip
        import json

        with gzip.open(path, "wt") as f:
            json.dump(replay.to_dict(), f)
        logger.info(f"Exported compressed replay to {path}")


class POVTracker:
    """
    Tracks player POV for replay viewing.

    Provides data for:
    - Following a specific player
    - Auto-switching to action (kills, clutches)
    - Multi-POV synchronization
    """

    def __init__(self, replay: MatchReplay):
        """
        Initialize POV tracker.

        Args:
            replay: Match replay data
        """
        self.replay = replay
        self.current_pov: int | None = None  # Steam ID

    def set_pov(self, steam_id: int):
        """Set POV to follow a specific player."""
        if steam_id in self.replay.player_names:
            self.current_pov = steam_id
            logger.debug(f"POV set to {self.replay.player_names[steam_id]}")

    def get_player_frames(self, round_num: int) -> list[ReplayFrame]:
        """Get frames with focus on current POV player."""
        round_data = self.replay.get_round(round_num)
        if not round_data:
            return []

        if self.current_pov is None:
            return round_data.frames

        # Filter/annotate frames for POV
        pov_frames = []
        for frame in round_data.frames:
            # Find POV player
            for p in frame.players:
                if p.steam_id == self.current_pov:
                    break

            # Create frame with POV info
            pov_frame = ReplayFrame(
                tick=frame.tick,
                round_num=frame.round_num,
                game_time=frame.game_time,
                players=frame.players,
                grenades=frame.grenades,
                bomb=frame.bomb,
                events=frame.events,
            )
            pov_frames.append(pov_frame)

        return pov_frames

    def find_action_moments(self, round_num: int) -> list[dict]:
        """
        Find interesting moments in a round (kills, clutches, etc.).

        Returns:
            List of moments with tick and description
        """
        round_data = self.replay.get_round(round_num)
        if not round_data:
            return []

        moments = []

        for kill in round_data.kills:
            moments.append(
                {
                    "tick": kill.tick,
                    "time": (kill.tick - round_data.start_tick) / self.replay.tick_rate,
                    "type": "kill",
                    "description": f"{kill.attacker_name} killed {kill.victim_name}",
                    "headshot": kill.headshot,
                }
            )

        # Sort by tick
        moments.sort(key=lambda m: m["tick"])
        return moments


# Convenience functions


def generate_replay(demo_data, sample_rate: int = DEFAULT_TICK_SAMPLE_RATE) -> MatchReplay:
    """
    Generate replay data from parsed demo.

    Args:
        demo_data: Parsed demo data (must include tick-level data)
        sample_rate: Tick sampling rate

    Returns:
        MatchReplay object
    """
    generator = ReplayGenerator(demo_data, sample_rate)
    return generator.generate_full_replay()


def generate_round_replay(demo_data, round_num: int) -> RoundReplay | None:
    """Generate replay for a single round."""
    generator = ReplayGenerator(demo_data)
    return generator.generate_round_replay(round_num)
