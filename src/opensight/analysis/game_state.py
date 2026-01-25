"""
OpenSight Game State Tracker

Unified game state tracking for CS2 demo analysis.
Consolidates game mode, map, round state, and match context information.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from opensight.core.constants import DemoSource, GameMode, Team
from opensight.core.parser import DemoData

logger = logging.getLogger(__name__)


class RoundPhase(str, Enum):
    """Current phase within a round."""

    WARMUP = "warmup"
    FREEZE_TIME = "freeze_time"
    LIVE = "live"
    BOMB_PLANTED = "bomb_planted"
    POST_ROUND = "post_round"
    HALFTIME = "halftime"
    OVERTIME = "overtime"


class MatchPhase(str, Enum):
    """Overall match phase."""

    WARMUP = "warmup"
    FIRST_HALF = "first_half"
    HALFTIME = "halftime"
    SECOND_HALF = "second_half"
    OVERTIME = "overtime"
    FINISHED = "finished"


@dataclass
class PlayerState:
    """Current state of a player."""

    steamid: int
    name: str
    team: Team
    is_alive: bool = True
    health: int = 100
    armor: int = 0
    has_helmet: bool = False
    has_defuser: bool = False
    money: int = 800
    equipment_value: int = 0
    position: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] | None = None
    view_angles: tuple[float, float] | None = None  # pitch, yaw


@dataclass
class RoundState:
    """State of the current round."""

    round_num: int
    phase: RoundPhase
    time_remaining: float = 0.0
    bomb_planted: bool = False
    bomb_site: str | None = None  # "A" or "B"
    bomb_time_remaining: float = 0.0
    ct_alive: int = 5
    t_alive: int = 5
    ct_score: int = 0
    t_score: int = 0


@dataclass
class GameState:
    """Complete game state at a point in time."""

    tick: int
    map_name: str
    game_mode: GameMode
    demo_source: DemoSource
    match_phase: MatchPhase
    round_state: RoundState
    players: dict[int, PlayerState] = field(default_factory=dict)

    # Match metadata
    match_id: str | None = None
    server_name: str | None = None
    tick_rate: int = 64


class GameStateTracker:
    """
    Tracks and manages game state throughout demo analysis.

    Provides a unified interface for querying game state at any point
    in the demo, including round phase, player states, and match context.

    Example:
        >>> from opensight.analysis.game_state import GameStateTracker
        >>> from opensight.core.parser import DemoParser
        >>>
        >>> parser = DemoParser()
        >>> demo_data = parser.parse("match.dem")
        >>> tracker = GameStateTracker(demo_data)
        >>>
        >>> # Get state at specific tick
        >>> state = tracker.get_state_at_tick(10000)
        >>> print(f"Round {state.round_state.round_num}, Phase: {state.round_state.phase}")
        >>>
        >>> # Get round summary
        >>> summary = tracker.get_round_summary(5)
    """

    def __init__(self, demo_data: DemoData):
        """
        Initialize the game state tracker.

        Args:
            demo_data: Parsed demo data from DemoParser
        """
        self.demo_data = demo_data
        self._round_states: dict[int, list[GameState]] = {}
        self._current_state: GameState | None = None
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize game state from demo data."""
        from opensight.analysis.detection import detect_demo_source, detect_game_mode

        # Detect demo source and game mode
        source = detect_demo_source(self.demo_data)
        mode = detect_game_mode(self.demo_data)

        # Initialize round state
        initial_round = RoundState(round_num=1, phase=RoundPhase.WARMUP, ct_score=0, t_score=0)

        # Create initial game state
        self._current_state = GameState(
            tick=0,
            map_name=self.demo_data.map_name,
            game_mode=mode,
            demo_source=source,
            match_phase=MatchPhase.WARMUP,
            round_state=initial_round,
            tick_rate=self.demo_data.tick_rate,
        )

        # Initialize player states
        self._initialize_players()

        logger.info(
            f"GameStateTracker initialized: {self.demo_data.map_name}, "
            f"Mode: {mode.value}, Source: {source.value}"
        )

    def _initialize_players(self) -> None:
        """Initialize player states from demo data."""
        if not self._current_state:
            return

        for steamid, name in self.demo_data.player_names.items():
            team_num = self.demo_data.player_teams.get(steamid, 0)
            team = Team(team_num) if team_num in [t.value for t in Team] else Team.UNASSIGNED

            self._current_state.players[steamid] = PlayerState(
                steamid=steamid, name=name, team=team
            )

    def get_state_at_tick(self, tick: int) -> GameState:
        """
        Get the game state at a specific tick.

        Args:
            tick: The tick number to query

        Returns:
            GameState at the specified tick
        """
        if not self._current_state:
            raise ValueError("Game state not initialized")

        # Find the round for this tick
        round_num = self._get_round_for_tick(tick)
        round_info = self._get_round_info(round_num)

        # Determine round phase
        phase = self._determine_phase(tick, round_info)

        # Determine match phase
        match_phase = self._determine_match_phase(round_num)

        # Build round state
        round_state = RoundState(
            round_num=round_num,
            phase=phase,
            ct_score=round_info.get("ct_score", 0) if round_info else 0,
            t_score=round_info.get("t_score", 0) if round_info else 0,
        )

        # Count alive players at this tick
        ct_alive, t_alive = self._count_alive_at_tick(tick, round_num)
        round_state.ct_alive = ct_alive
        round_state.t_alive = t_alive

        # Build complete state
        state = GameState(
            tick=tick,
            map_name=self.demo_data.map_name,
            game_mode=self._current_state.game_mode,
            demo_source=self._current_state.demo_source,
            match_phase=match_phase,
            round_state=round_state,
            players=self._get_player_states_at_tick(tick),
            tick_rate=self.demo_data.tick_rate,
        )

        return state

    def _get_round_for_tick(self, tick: int) -> int:
        """Determine which round a tick belongs to."""
        if not hasattr(self.demo_data, "rounds") or not self.demo_data.rounds:
            # Estimate from kills
            for kill in reversed(self.demo_data.kills):
                if kill.tick <= tick:
                    return kill.round_num
            return 1

        for round_info in self.demo_data.rounds:
            if round_info.start_tick <= tick <= round_info.end_tick:
                return round_info.round_num

        return 1

    def _get_round_info(self, round_num: int) -> dict[str, Any] | None:
        """Get round information by round number."""
        if not hasattr(self.demo_data, "rounds") or not self.demo_data.rounds:
            return None

        for round_info in self.demo_data.rounds:
            if round_info.round_num == round_num:
                return {
                    "round_num": round_info.round_num,
                    "start_tick": round_info.start_tick,
                    "end_tick": round_info.end_tick,
                    "freeze_end_tick": round_info.freeze_end_tick,
                    "winner": round_info.winner,
                    "ct_score": round_info.ct_score,
                    "t_score": round_info.t_score,
                }
        return None

    def _determine_phase(self, tick: int, round_info: dict | None) -> RoundPhase:
        """Determine the round phase at a given tick."""
        if not round_info:
            return RoundPhase.LIVE

        if tick < round_info.get("freeze_end_tick", 0):
            return RoundPhase.FREEZE_TIME
        elif tick > round_info.get("end_tick", float("inf")):
            return RoundPhase.POST_ROUND
        else:
            return RoundPhase.LIVE

    def _determine_match_phase(self, round_num: int) -> MatchPhase:
        """Determine the overall match phase."""
        if round_num <= 0:
            return MatchPhase.WARMUP
        elif round_num <= 12:
            return MatchPhase.FIRST_HALF
        elif round_num == 13:
            return MatchPhase.HALFTIME
        elif round_num <= 24:
            return MatchPhase.SECOND_HALF
        else:
            return MatchPhase.OVERTIME

    def _count_alive_at_tick(self, tick: int, round_num: int) -> tuple[int, int]:
        """Count alive players on each team at a specific tick."""
        ct_alive = 5
        t_alive = 5

        for kill in self.demo_data.kills:
            if kill.round_num == round_num and kill.tick <= tick:
                if kill.victim_side == "CT":
                    ct_alive = max(0, ct_alive - 1)
                elif kill.victim_side == "T":
                    t_alive = max(0, t_alive - 1)

        return ct_alive, t_alive

    def _get_player_states_at_tick(self, tick: int) -> dict[int, PlayerState]:
        """Get player states at a specific tick."""
        if not self._current_state:
            return {}

        # Start with base player states
        states = {}
        for steamid, base_state in self._current_state.players.items():
            states[steamid] = PlayerState(
                steamid=base_state.steamid,
                name=base_state.name,
                team=base_state.team,
                is_alive=True,
            )

        # Update based on kills up to this tick
        round_num = self._get_round_for_tick(tick)
        for kill in self.demo_data.kills:
            if kill.round_num == round_num and kill.tick <= tick:
                if kill.victim_steamid in states:
                    states[kill.victim_steamid].is_alive = False

        # Update positions if tick data available
        if hasattr(self.demo_data, "ticks_df") and self.demo_data.ticks_df is not None:
            tick_data = self.demo_data.ticks_df[self.demo_data.ticks_df["tick"] == tick]
            for _, row in tick_data.iterrows():
                steamid = row.get("steamid")
                if steamid in states:
                    states[steamid].position = (row.get("X", 0), row.get("Y", 0), row.get("Z", 0))

        return states

    def get_round_summary(self, round_num: int) -> dict[str, Any]:
        """
        Get a summary of a specific round.

        Args:
            round_num: The round number to summarize

        Returns:
            Dictionary with round summary information
        """
        round_info = self._get_round_info(round_num)

        # Get kills in this round
        round_kills = [k for k in self.demo_data.kills if k.round_num == round_num]

        # Get damages in this round
        round_damages = [d for d in self.demo_data.damages if d.round_num == round_num]

        return {
            "round_num": round_num,
            "winner": round_info.get("winner") if round_info else None,
            "ct_score": round_info.get("ct_score", 0) if round_info else 0,
            "t_score": round_info.get("t_score", 0) if round_info else 0,
            "total_kills": len(round_kills),
            "total_damage": sum(d.damage for d in round_damages),
            "first_kill": round_kills[0] if round_kills else None,
            "kills_by_team": {
                "CT": len([k for k in round_kills if k.attacker_side == "CT"]),
                "T": len([k for k in round_kills if k.attacker_side == "T"]),
            },
        }

    def get_match_summary(self) -> dict[str, Any]:
        """
        Get a summary of the entire match.

        Returns:
            Dictionary with match summary information
        """
        if not self._current_state:
            return {}

        # Calculate final score
        final_ct_score = 0
        final_t_score = 0

        if hasattr(self.demo_data, "rounds") and self.demo_data.rounds:
            last_round = self.demo_data.rounds[-1]
            final_ct_score = last_round.ct_score
            final_t_score = last_round.t_score

        return {
            "map": self.demo_data.map_name,
            "game_mode": self._current_state.game_mode.value,
            "demo_source": self._current_state.demo_source.value,
            "duration_seconds": self.demo_data.duration_seconds,
            "total_rounds": self.demo_data.num_rounds,
            "final_score": {"CT": final_ct_score, "T": final_t_score},
            "total_kills": len(self.demo_data.kills),
            "total_players": len(self.demo_data.player_names),
        }


def track_game_state(demo_data: DemoData) -> GameStateTracker:
    """
    Convenience function to create a GameStateTracker.

    Args:
        demo_data: Parsed demo data

    Returns:
        Initialized GameStateTracker
    """
    return GameStateTracker(demo_data)
