"""
Team Playbook Generation Module for CS2 Demo Analyzer.

Automatically generates shareable playbooks from team demos,
highlighting default setups, executes, and retake protocols.
"""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# Playbook Data Types
# ============================================================================


class PlayType(Enum):
    """Types of plays/strategies."""

    DEFAULT = "default"
    EXECUTE = "execute"
    SPLIT = "split"
    RUSH = "rush"
    FAKE = "fake"
    PICK = "pick"
    SLOW = "slow"
    RETAKE = "retake"
    STACK = "stack"
    ANTI_ECO = "anti_eco"


class BombSite(Enum):
    """Bomb sites."""

    A = "A"
    B = "B"
    MID = "mid"
    UNKNOWN = "unknown"


class Side(Enum):
    """Team sides."""

    T = "T"
    CT = "CT"


@dataclass
class PlayerPosition:
    """Player position at a specific time."""

    steamid: str
    name: str
    x: float
    y: float
    z: float
    pitch: float = 0.0
    yaw: float = 0.0
    role_in_play: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steamid": self.steamid,
            "name": self.name,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "z": round(self.z, 1),
            "pitch": round(self.pitch, 1),
            "yaw": round(self.yaw, 1),
            "role_in_play": self.role_in_play,
        }


@dataclass
class UtilityUsage:
    """Utility usage in a play."""

    player_steamid: str
    player_name: str
    grenade_type: str  # smoke, flash, molotov, he
    x: float
    y: float
    z: float
    tick: int
    description: str = ""
    lineup_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_steamid": self.player_steamid,
            "player_name": self.player_name,
            "grenade_type": self.grenade_type,
            "position": {"x": round(self.x, 1), "y": round(self.y, 1), "z": round(self.z, 1)},
            "tick": self.tick,
            "description": self.description,
            "lineup_name": self.lineup_name,
        }


@dataclass
class Play:
    """A single play/strategy extracted from demos."""

    play_id: str
    name: str
    play_type: PlayType
    side: Side
    target_site: BombSite
    map_name: str

    # Description
    description: str = ""
    success_rate: float = 0.0

    # Timing
    avg_execute_time: float = 0.0  # Seconds into round
    time_variance: float = 0.0

    # Positions
    initial_positions: list[PlayerPosition] = field(default_factory=list)
    execute_positions: list[PlayerPosition] = field(default_factory=list)
    final_positions: list[PlayerPosition] = field(default_factory=list)

    # Utility
    utility_sequence: list[UtilityUsage] = field(default_factory=list)

    # Movement paths
    movement_paths: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)

    # Statistics
    times_run: int = 0
    rounds_won: int = 0
    first_kill_rate: float = 0.0

    # Demo sources
    source_demos: list[str] = field(default_factory=list)
    example_rounds: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "play_id": self.play_id,
            "name": self.name,
            "play_type": self.play_type.value,
            "side": self.side.value,
            "target_site": self.target_site.value,
            "map_name": self.map_name,
            "description": self.description,
            "success_rate": round(self.success_rate, 2),
            "avg_execute_time": round(self.avg_execute_time, 1),
            "time_variance": round(self.time_variance, 1),
            "initial_positions": [p.to_dict() for p in self.initial_positions],
            "execute_positions": [p.to_dict() for p in self.execute_positions],
            "final_positions": [p.to_dict() for p in self.final_positions],
            "utility_sequence": [u.to_dict() for u in self.utility_sequence],
            "movement_paths": {
                k: [(round(x, 1), round(y, 1), round(z, 1)) for x, y, z in v]
                for k, v in self.movement_paths.items()
            },
            "times_run": self.times_run,
            "rounds_won": self.rounds_won,
            "first_kill_rate": round(self.first_kill_rate, 2),
            "source_demos": self.source_demos,
            "example_rounds": self.example_rounds[:3],
        }


@dataclass
class DefaultSetup:
    """A default defensive setup for CT side."""

    setup_id: str
    name: str
    map_name: str

    # Player positions
    positions: list[PlayerPosition] = field(default_factory=list)

    # Coverage areas
    a_players: int = 0
    b_players: int = 0
    mid_players: int = 0

    # Utility setup
    smoke_positions: list[tuple[float, float, float]] = field(default_factory=list)
    molotov_triggers: list[str] = field(default_factory=list)

    # Statistics
    times_used: int = 0
    rounds_won: int = 0

    # Notes
    description: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "name": self.name,
            "map_name": self.map_name,
            "positions": [p.to_dict() for p in self.positions],
            "a_players": self.a_players,
            "b_players": self.b_players,
            "mid_players": self.mid_players,
            "smoke_positions": [
                (round(x, 1), round(y, 1), round(z, 1)) for x, y, z in self.smoke_positions
            ],
            "molotov_triggers": self.molotov_triggers,
            "times_used": self.times_used,
            "rounds_won": self.rounds_won,
            "success_rate": round(self.rounds_won / max(1, self.times_used), 2),
            "description": self.description,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


@dataclass
class RetakeProtocol:
    """A retake strategy for specific site."""

    protocol_id: str
    name: str
    map_name: str
    site: BombSite

    # Retake positions
    approach_positions: list[PlayerPosition] = field(default_factory=list)

    # Utility sequence
    utility_sequence: list[UtilityUsage] = field(default_factory=list)

    # Priority targets
    clear_order: list[str] = field(default_factory=list)

    # Statistics
    times_attempted: int = 0
    times_succeeded: int = 0

    # Notes
    description: str = ""
    key_callouts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "name": self.name,
            "map_name": self.map_name,
            "site": self.site.value,
            "approach_positions": [p.to_dict() for p in self.approach_positions],
            "utility_sequence": [u.to_dict() for u in self.utility_sequence],
            "clear_order": self.clear_order,
            "times_attempted": self.times_attempted,
            "times_succeeded": self.times_succeeded,
            "success_rate": round(self.times_succeeded / max(1, self.times_attempted), 2),
            "description": self.description,
            "key_callouts": self.key_callouts,
        }


@dataclass
class Playbook:
    """Complete team playbook."""

    playbook_id: str
    team_name: str
    created_at: str
    last_updated: str

    # Team members
    players: list[dict[str, str]] = field(default_factory=list)  # steamid, name, role

    # Plays organized by map
    t_side_plays: dict[str, list[Play]] = field(default_factory=dict)  # map -> plays
    ct_setups: dict[str, list[DefaultSetup]] = field(default_factory=dict)  # map -> setups
    retakes: dict[str, list[RetakeProtocol]] = field(default_factory=dict)  # map -> protocols

    # Map pool
    map_pool: list[str] = field(default_factory=list)
    map_win_rates: dict[str, float] = field(default_factory=dict)

    # Analysis stats
    demos_analyzed: int = 0
    rounds_analyzed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook_id": self.playbook_id,
            "team_name": self.team_name,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "players": self.players,
            "t_side_plays": {
                m: [p.to_dict() for p in plays] for m, plays in self.t_side_plays.items()
            },
            "ct_setups": {m: [s.to_dict() for s in setups] for m, setups in self.ct_setups.items()},
            "retakes": {
                m: [r.to_dict() for r in protocols] for m, protocols in self.retakes.items()
            },
            "map_pool": self.map_pool,
            "map_win_rates": {k: round(v, 2) for k, v in self.map_win_rates.items()},
            "demos_analyzed": self.demos_analyzed,
            "rounds_analyzed": self.rounds_analyzed,
        }


# ============================================================================
# Play Detection and Classification
# ============================================================================


class PlayClassifier:
    """
    Classifies rounds into play types based on positioning and timing.
    """

    # Map-specific site locations (approximate centers)
    SITE_LOCATIONS = {
        "de_dust2": {
            BombSite.A: (-1424, 2496, 96),
            BombSite.B: (-1536, 2592, 96),
            BombSite.MID: (-700, 1000, 0),
        },
        "de_mirage": {
            BombSite.A: (-300, -1700, -160),
            BombSite.B: (-2000, 400, -160),
            BombSite.MID: (-400, 0, -100),
        },
        "de_inferno": {
            BombSite.A: (2080, 480, 96),
            BombSite.B: (250, 2800, 160),
            BombSite.MID: (1280, 800, 0),
        },
        "de_nuke": {
            BombSite.A: (640, -780, -415),
            BombSite.B: (624, -700, -750),
            BombSite.MID: (0, 0, 0),
        },
        "de_ancient": {
            BombSite.A: (-450, -1300, 0),
            BombSite.B: (750, 0, 100),
            BombSite.MID: (100, -600, 50),
        },
        "de_anubis": {
            BombSite.A: (-1200, 200, 0),
            BombSite.B: (1000, 400, 0),
            BombSite.MID: (0, -200, 0),
        },
        "de_vertigo": {
            BombSite.A: (-1500, -400, 11776),
            BombSite.B: (-2200, -1300, 11776),
            BombSite.MID: (-1800, -1000, 11776),
        },
    }

    # Time thresholds (seconds into round)
    RUSH_THRESHOLD = 25
    FAST_EXECUTE_THRESHOLD = 45
    SLOW_THRESHOLD = 75

    def classify_round(
        self, round_data: dict[str, Any], map_name: str
    ) -> tuple[PlayType, BombSite]:
        """
        Classify a round into a play type.

        Args:
            round_data: Round data including positions, kills, bomb events
            map_name: Map name

        Returns:
            Tuple of (PlayType, target BombSite)
        """
        # Get key events
        bomb_plant = round_data.get("bomb_plant")
        first_kill = round_data.get("first_kill")
        t_positions = round_data.get("t_positions", [])

        # Determine target site from bomb plant
        if bomb_plant:
            site = self._get_bomb_site(bomb_plant, map_name)
        else:
            site = self._predict_site_from_positions(t_positions, map_name)

        # Determine play type from timing and movement
        plant_time = bomb_plant.get("round_time", 999) if bomb_plant else 999
        first_kill_time = first_kill.get("round_time", 999) if first_kill else 999

        # Classify based on timing
        if first_kill_time < self.RUSH_THRESHOLD:
            play_type = PlayType.RUSH
        elif plant_time < self.FAST_EXECUTE_THRESHOLD:
            play_type = PlayType.EXECUTE
        elif plant_time < self.SLOW_THRESHOLD:
            play_type = PlayType.DEFAULT
        else:
            play_type = PlayType.SLOW

        # Check for split (players spread across map)
        if self._is_split(t_positions, map_name):
            play_type = PlayType.SPLIT

        # Check for fake (utility at one site, plant at other)
        if self._is_fake(round_data, site, map_name):
            play_type = PlayType.FAKE

        return play_type, site

    def _get_bomb_site(self, bomb_event: dict[str, Any], map_name: str) -> BombSite:
        """Determine bomb site from plant location."""
        if not bomb_event:
            return BombSite.UNKNOWN

        # Use site field if available
        if bomb_event.get("site"):
            site_str = bomb_event["site"].upper()
            if site_str == "A":
                return BombSite.A
            elif site_str == "B":
                return BombSite.B

        # Calculate from position
        plant_pos = (bomb_event.get("x", 0), bomb_event.get("y", 0), bomb_event.get("z", 0))

        sites = self.SITE_LOCATIONS.get(map_name, {})
        if not sites:
            return BombSite.UNKNOWN

        closest_site = BombSite.UNKNOWN
        min_dist = float("inf")

        for site, site_pos in sites.items():
            if site == BombSite.MID:
                continue
            dist = self._distance(plant_pos, site_pos)
            if dist < min_dist:
                min_dist = dist
                closest_site = site

        return closest_site

    def _predict_site_from_positions(
        self, positions: list[dict[str, Any]], map_name: str
    ) -> BombSite:
        """Predict target site from T positions."""
        if not positions:
            return BombSite.UNKNOWN

        sites = self.SITE_LOCATIONS.get(map_name, {})
        if not sites:
            return BombSite.UNKNOWN

        # Average T position
        avg_x = sum(p.get("x", 0) for p in positions) / len(positions)
        avg_y = sum(p.get("y", 0) for p in positions) / len(positions)
        avg_z = sum(p.get("z", 0) for p in positions) / len(positions)
        avg_pos = (avg_x, avg_y, avg_z)

        closest_site = BombSite.UNKNOWN
        min_dist = float("inf")

        for site, site_pos in sites.items():
            if site == BombSite.MID:
                continue
            dist = self._distance(avg_pos, site_pos)
            if dist < min_dist:
                min_dist = dist
                closest_site = site

        return closest_site

    def _is_split(self, positions: list[dict[str, Any]], map_name: str) -> bool:
        """Check if T players are split across the map."""
        if len(positions) < 4:
            return False

        # Calculate position spread
        xs = [p.get("x", 0) for p in positions]
        ys = [p.get("y", 0) for p in positions]

        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)

        # Consider split if spread is large
        return x_spread > 1500 or y_spread > 1500

    def _is_fake(self, round_data: dict[str, Any], actual_site: BombSite, map_name: str) -> bool:
        """Check if round involved a fake."""
        grenades = round_data.get("grenades", [])
        if not grenades:
            return False

        # Check if utility was used at opposite site
        other_site = BombSite.B if actual_site == BombSite.A else BombSite.A
        sites = self.SITE_LOCATIONS.get(map_name, {})

        if other_site not in sites:
            return False

        other_site_pos = sites[other_site]
        utility_at_other = 0

        for g in grenades:
            g_pos = (g.get("x", 0), g.get("y", 0), g.get("z", 0))
            if self._distance(g_pos, other_site_pos) < 1000:
                utility_at_other += 1

        # Consider fake if 2+ utility at other site
        return utility_at_other >= 2

    def _distance(self, p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
        """Calculate 3D distance."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# ============================================================================
# Playbook Generator
# ============================================================================


class PlaybookGenerator:
    """
    Generates playbooks from analyzed team demos.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path.home() / ".opensight" / "playbooks"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = PlayClassifier()
        self.playbooks: dict[str, Playbook] = {}

    def get_playbook(self, team_name: str) -> Playbook:
        """Get or create a playbook for a team."""
        if team_name not in self.playbooks:
            playbook_path = (
                self.data_dir
                / f"playbook_{hashlib.md5(team_name.encode(), usedforsecurity=False).hexdigest()[:8]}.json"
            )
            if playbook_path.exists():
                try:
                    with open(playbook_path) as f:
                        data = json.load(f)
                    self.playbooks[team_name] = self._from_dict(data)
                except (OSError, json.JSONDecodeError):
                    self.playbooks[team_name] = self._create_playbook(team_name)
            else:
                self.playbooks[team_name] = self._create_playbook(team_name)

        return self.playbooks[team_name]

    def save_playbook(self, playbook: Playbook) -> None:
        """Save playbook to disk."""
        self.playbooks[playbook.team_name] = playbook
        playbook_path = (
            self.data_dir
            / f"playbook_{hashlib.md5(playbook.team_name.encode(), usedforsecurity=False).hexdigest()[:8]}.json"
        )
        try:
            with open(playbook_path, "w") as f:
                json.dump(playbook.to_dict(), f, indent=2)
        except OSError:
            pass

    def _create_playbook(self, team_name: str) -> Playbook:
        """Create a new empty playbook."""
        return Playbook(
            playbook_id=hashlib.md5(
                f"{team_name}_{datetime.now().isoformat()}".encode(),
                usedforsecurity=False,
            ).hexdigest()[:12],
            team_name=team_name,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )

    def _from_dict(self, data: dict[str, Any]) -> Playbook:
        """Reconstruct playbook from dictionary."""
        playbook = Playbook(
            playbook_id=data.get("playbook_id", ""),
            team_name=data.get("team_name", "Unknown"),
            created_at=data.get("created_at", ""),
            last_updated=data.get("last_updated", ""),
            players=data.get("players", []),
            map_pool=data.get("map_pool", []),
            map_win_rates=data.get("map_win_rates", {}),
            demos_analyzed=data.get("demos_analyzed", 0),
            rounds_analyzed=data.get("rounds_analyzed", 0),
        )

        # Reconstruct plays (simplified - would need full reconstruction in production)
        for map_name, plays_data in data.get("t_side_plays", {}).items():
            playbook.t_side_plays[map_name] = []
            for p_data in plays_data:
                try:
                    play = Play(
                        play_id=p_data.get("play_id", ""),
                        name=p_data.get("name", "Unknown"),
                        play_type=PlayType(p_data.get("play_type", "default")),
                        side=Side(p_data.get("side", "T")),
                        target_site=BombSite(p_data.get("target_site", "unknown")),
                        map_name=map_name,
                        description=p_data.get("description", ""),
                        success_rate=p_data.get("success_rate", 0),
                        avg_execute_time=p_data.get("avg_execute_time", 0),
                        times_run=p_data.get("times_run", 0),
                        rounds_won=p_data.get("rounds_won", 0),
                        first_kill_rate=p_data.get("first_kill_rate", 0),
                        source_demos=p_data.get("source_demos", []),
                    )
                    playbook.t_side_plays[map_name].append(play)
                except (KeyError, ValueError):
                    continue

        return playbook

    def analyze_demo(
        self, demo_data: dict[str, Any], team_steamids: list[str], team_name: str
    ) -> dict[str, Any]:
        """
        Analyze a demo and add plays to the playbook.

        Args:
            demo_data: Complete demo data
            team_steamids: Steam IDs of team members
            team_name: Team name

        Returns:
            Summary of extracted plays
        """
        playbook = self.get_playbook(team_name)
        map_name = demo_data.get("map_name", "unknown")

        if map_name not in playbook.map_pool:
            playbook.map_pool.append(map_name)

        # Initialize map-specific structures
        if map_name not in playbook.t_side_plays:
            playbook.t_side_plays[map_name] = []
        if map_name not in playbook.ct_setups:
            playbook.ct_setups[map_name] = []
        if map_name not in playbook.retakes:
            playbook.retakes[map_name] = []

        plays_extracted = []
        setups_extracted = []
        retakes_extracted = []

        rounds = demo_data.get("rounds", [])

        for round_data in rounds:
            round_data.get("round_num", 0)

            # Determine team side this round
            team_side = self._get_team_side(round_data, team_steamids)

            if team_side == Side.T:
                # Extract T-side play
                play = self._extract_t_play(
                    round_data, team_steamids, map_name, demo_data.get("demo_id", "")
                )
                if play:
                    plays_extracted.append(play)
                    self._add_play_to_playbook(playbook, play, map_name)

            else:
                # Extract CT setup
                setup = self._extract_ct_setup(round_data, team_steamids, map_name)
                if setup:
                    setups_extracted.append(setup)
                    self._add_setup_to_playbook(playbook, setup, map_name)

                # Check for retake situation
                retake = self._extract_retake(round_data, team_steamids, map_name)
                if retake:
                    retakes_extracted.append(retake)
                    self._add_retake_to_playbook(playbook, retake, map_name)

        # Update statistics
        playbook.demos_analyzed += 1
        playbook.rounds_analyzed += len(rounds)
        playbook.last_updated = datetime.now().isoformat()

        # Update map win rate
        team_wins = sum(1 for r in rounds if self._team_won_round(r, team_steamids))
        if len(rounds) > 0:
            current_wr = playbook.map_win_rates.get(map_name, 0.5)
            new_wr = team_wins / len(rounds)
            playbook.map_win_rates[map_name] = (current_wr + new_wr) / 2

        self.save_playbook(playbook)

        return {
            "map": map_name,
            "rounds_analyzed": len(rounds),
            "t_plays_extracted": len(plays_extracted),
            "ct_setups_extracted": len(setups_extracted),
            "retakes_extracted": len(retakes_extracted),
        }

    def _get_team_side(self, round_data: dict[str, Any], team_steamids: list[str]) -> Side:
        """Determine which side the team is on this round."""
        # Check from round info or player positions
        t_side_players = round_data.get("t_side_players", [])

        team_on_t = sum(1 for sid in team_steamids if sid in t_side_players)
        if team_on_t >= 3:
            return Side.T
        return Side.CT

    def _team_won_round(self, round_data: dict[str, Any], team_steamids: list[str]) -> bool:
        """Check if team won this round."""
        winner_side = round_data.get("winner")
        team_side = self._get_team_side(round_data, team_steamids)
        return winner_side == team_side.value

    def _extract_t_play(
        self, round_data: dict[str, Any], team_steamids: list[str], map_name: str, demo_id: str
    ) -> Play | None:
        """Extract a T-side play from round data."""
        # Classify the round
        play_type, target_site = self.classifier.classify_round(round_data, map_name)

        # Get positions and utility
        positions = round_data.get("positions", {})
        grenades = round_data.get("grenades", [])
        bomb_plant = round_data.get("bomb_plant")
        round_won = round_data.get("winner") == "T"

        # Calculate execute time
        execute_time = 0
        if bomb_plant:
            execute_time = bomb_plant.get("round_time", 0)
        elif round_data.get("first_kill"):
            execute_time = round_data["first_kill"].get("round_time", 0)

        # Extract player positions at execute time
        execute_positions = []
        for steamid in team_steamids:
            if steamid in positions:
                pos = positions[steamid].get("execute", positions[steamid].get("initial", {}))
                if pos:
                    execute_positions.append(
                        PlayerPosition(
                            steamid=steamid,
                            name=pos.get("name", "Player"),
                            x=pos.get("x", 0),
                            y=pos.get("y", 0),
                            z=pos.get("z", 0),
                        )
                    )

        # Extract utility sequence
        utility_sequence = []
        team_grenades = [g for g in grenades if g.get("player_steamid") in team_steamids]
        for g in sorted(team_grenades, key=lambda x: x.get("tick", 0)):
            utility_sequence.append(
                UtilityUsage(
                    player_steamid=g.get("player_steamid", ""),
                    player_name=g.get("player_name", "Player"),
                    grenade_type=g.get("grenade_type", "unknown"),
                    x=g.get("x", 0),
                    y=g.get("y", 0),
                    z=g.get("z", 0),
                    tick=g.get("tick", 0),
                )
            )

        play = Play(
            play_id=hashlib.md5(
                f"{demo_id}_{round_data.get('round_num', 0)}".encode(), usedforsecurity=False
            ).hexdigest()[:8],
            name=f"{play_type.value.title()} {target_site.value}",
            play_type=play_type,
            side=Side.T,
            target_site=target_site,
            map_name=map_name,
            avg_execute_time=execute_time,
            execute_positions=execute_positions,
            utility_sequence=utility_sequence,
            times_run=1,
            rounds_won=1 if round_won else 0,
            source_demos=[demo_id],
            example_rounds=[
                {"demo_id": demo_id, "round_num": round_data.get("round_num", 0), "won": round_won}
            ],
        )

        return play

    def _extract_ct_setup(
        self, round_data: dict[str, Any], team_steamids: list[str], map_name: str
    ) -> DefaultSetup | None:
        """Extract CT default setup from round data."""
        positions = round_data.get("positions", {})
        round_won = round_data.get("winner") == "CT"

        # Get initial positions
        player_positions = []
        for steamid in team_steamids:
            if steamid in positions:
                pos = positions[steamid].get("initial", {})
                if pos:
                    player_positions.append(
                        PlayerPosition(
                            steamid=steamid,
                            name=pos.get("name", "Player"),
                            x=pos.get("x", 0),
                            y=pos.get("y", 0),
                            z=pos.get("z", 0),
                        )
                    )

        if len(player_positions) < 4:
            return None

        # Count players per site
        sites = PlayClassifier.SITE_LOCATIONS.get(map_name, {})
        a_players = 0
        b_players = 0

        for pos in player_positions:
            pos_tuple = (pos.x, pos.y, pos.z)

            if BombSite.A in sites:
                a_dist = self._distance(pos_tuple, sites[BombSite.A])
                b_dist = (
                    self._distance(pos_tuple, sites[BombSite.B])
                    if BombSite.B in sites
                    else float("inf")
                )

                if a_dist < b_dist and a_dist < 2000:
                    a_players += 1
                elif b_dist < 2000:
                    b_players += 1

        setup = DefaultSetup(
            setup_id=hashlib.md5(
                f"{map_name}_{a_players}_{b_players}".encode(), usedforsecurity=False
            ).hexdigest()[:8],
            name=f"{a_players}A-{b_players}B Setup",
            map_name=map_name,
            positions=player_positions,
            a_players=a_players,
            b_players=b_players,
            mid_players=5 - a_players - b_players,
            times_used=1,
            rounds_won=1 if round_won else 0,
        )

        return setup

    def _extract_retake(
        self, round_data: dict[str, Any], team_steamids: list[str], map_name: str
    ) -> RetakeProtocol | None:
        """Extract retake protocol if applicable."""
        bomb_plant = round_data.get("bomb_plant")
        if not bomb_plant:
            return None

        # Check if CT won after bomb plant (successful retake)
        round_won = round_data.get("winner") == "CT"

        site = BombSite.A if bomb_plant.get("site", "").upper() == "A" else BombSite.B

        # Get grenades used after bomb plant
        grenades = round_data.get("grenades", [])
        plant_tick = bomb_plant.get("tick", 0)
        retake_utility = []

        team_grenades = [
            g
            for g in grenades
            if g.get("player_steamid") in team_steamids and g.get("tick", 0) > plant_tick
        ]

        for g in sorted(team_grenades, key=lambda x: x.get("tick", 0)):
            retake_utility.append(
                UtilityUsage(
                    player_steamid=g.get("player_steamid", ""),
                    player_name=g.get("player_name", "Player"),
                    grenade_type=g.get("grenade_type", "unknown"),
                    x=g.get("x", 0),
                    y=g.get("y", 0),
                    z=g.get("z", 0),
                    tick=g.get("tick", 0),
                )
            )

        if not retake_utility:
            return None

        protocol = RetakeProtocol(
            protocol_id=hashlib.md5(
                f"{map_name}_{site.value}_retake".encode(), usedforsecurity=False
            ).hexdigest()[:8],
            name=f"{site.value} Site Retake",
            map_name=map_name,
            site=site,
            utility_sequence=retake_utility,
            times_attempted=1,
            times_succeeded=1 if round_won else 0,
        )

        return protocol

    def _add_play_to_playbook(self, playbook: Playbook, play: Play, map_name: str) -> None:
        """Add or merge play into playbook."""
        existing = playbook.t_side_plays.get(map_name, [])

        # Check for similar existing play
        for _i, existing_play in enumerate(existing):
            if self._plays_similar(play, existing_play):
                # Merge into existing
                existing_play.times_run += 1
                existing_play.rounds_won += play.rounds_won
                existing_play.success_rate = existing_play.rounds_won / existing_play.times_run
                existing_play.source_demos.extend(play.source_demos)
                existing_play.example_rounds.extend(play.example_rounds)
                # Average execute time
                existing_play.avg_execute_time = (
                    existing_play.avg_execute_time * (existing_play.times_run - 1)
                    + play.avg_execute_time
                ) / existing_play.times_run
                return

        # Add as new play
        playbook.t_side_plays[map_name].append(play)

    def _add_setup_to_playbook(
        self, playbook: Playbook, setup: DefaultSetup, map_name: str
    ) -> None:
        """Add or merge CT setup into playbook."""
        existing = playbook.ct_setups.get(map_name, [])

        for existing_setup in existing:
            if (
                existing_setup.a_players == setup.a_players
                and existing_setup.b_players == setup.b_players
            ):
                # Merge
                existing_setup.times_used += 1
                existing_setup.rounds_won += setup.rounds_won
                return

        playbook.ct_setups[map_name].append(setup)

    def _add_retake_to_playbook(
        self, playbook: Playbook, retake: RetakeProtocol, map_name: str
    ) -> None:
        """Add or merge retake protocol into playbook."""
        existing = playbook.retakes.get(map_name, [])

        for existing_retake in existing:
            if existing_retake.site == retake.site:
                # Merge
                existing_retake.times_attempted += 1
                existing_retake.times_succeeded += retake.times_succeeded
                return

        playbook.retakes[map_name].append(retake)

    def _plays_similar(self, play1: Play, play2: Play) -> bool:
        """Check if two plays are similar enough to merge."""
        if play1.play_type != play2.play_type:
            return False
        if play1.target_site != play2.target_site:
            return False
        if abs(play1.avg_execute_time - play2.avg_execute_time) > 15:
            return False
        return True

    def _distance(self, p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
        """Calculate 3D distance."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def generate_playbook_report(self, team_name: str) -> dict[str, Any]:
        """
        Generate a comprehensive playbook report.

        Args:
            team_name: Team name

        Returns:
            Full playbook report
        """
        playbook = self.get_playbook(team_name)

        report = {
            "team_name": playbook.team_name,
            "created_at": playbook.created_at,
            "last_updated": playbook.last_updated,
            "summary": {
                "demos_analyzed": playbook.demos_analyzed,
                "rounds_analyzed": playbook.rounds_analyzed,
                "map_pool": playbook.map_pool,
                "total_t_plays": sum(len(plays) for plays in playbook.t_side_plays.values()),
                "total_ct_setups": sum(len(setups) for setups in playbook.ct_setups.values()),
                "total_retakes": sum(len(retakes) for retakes in playbook.retakes.values()),
            },
            "map_analysis": {},
        }

        # Per-map analysis
        for map_name in playbook.map_pool:
            t_plays = playbook.t_side_plays.get(map_name, [])
            ct_setups = playbook.ct_setups.get(map_name, [])
            retakes = playbook.retakes.get(map_name, [])

            # Find best plays
            best_t_plays = sorted(t_plays, key=lambda x: x.success_rate, reverse=True)[:5]
            best_setups = sorted(
                ct_setups, key=lambda x: x.rounds_won / max(1, x.times_used), reverse=True
            )[:3]

            # Play type distribution
            play_types = defaultdict(int)
            for play in t_plays:
                play_types[play.play_type.value] += play.times_run

            report["map_analysis"][map_name] = {
                "win_rate": playbook.map_win_rates.get(map_name, 0),
                "t_side": {
                    "total_plays": len(t_plays),
                    "play_type_distribution": dict(play_types),
                    "best_plays": [p.to_dict() for p in best_t_plays],
                },
                "ct_side": {
                    "total_setups": len(ct_setups),
                    "best_setups": [s.to_dict() for s in best_setups],
                },
                "retakes": {
                    "a_site": next((r.to_dict() for r in retakes if r.site == BombSite.A), None),
                    "b_site": next((r.to_dict() for r in retakes if r.site == BombSite.B), None),
                },
            }

        return report

    def export_playbook(self, team_name: str, format: str = "json") -> str:
        """
        Export playbook in shareable format.

        Args:
            team_name: Team name
            format: Export format ("json", "markdown")

        Returns:
            Exported playbook content
        """
        playbook = self.get_playbook(team_name)

        if format == "markdown":
            return self._export_markdown(playbook)
        else:
            return json.dumps(playbook.to_dict(), indent=2)

    def _export_markdown(self, playbook: Playbook) -> str:
        """Export playbook as Markdown."""
        md = []
        md.append(f"# {playbook.team_name} Playbook")
        md.append(f"\n*Last updated: {playbook.last_updated}*")
        md.append(f"\n**Demos analyzed:** {playbook.demos_analyzed}")
        md.append(f"\n**Map pool:** {', '.join(playbook.map_pool)}")

        for map_name in playbook.map_pool:
            md.append(f"\n## {map_name.replace('de_', '').title()}")

            # T-side plays
            t_plays = playbook.t_side_plays.get(map_name, [])
            if t_plays:
                md.append("\n### T-Side Plays")
                for play in sorted(t_plays, key=lambda x: x.success_rate, reverse=True)[:5]:
                    wr = round(play.success_rate * 100)
                    md.append(f"\n**{play.name}** (Win rate: {wr}%, Run: {play.times_run}x)")
                    if play.description:
                        md.append(f"\n_{play.description}_")
                    md.append(f"\n- Execute time: ~{play.avg_execute_time}s")
                    if play.utility_sequence:
                        md.append(f"- Utility: {len(play.utility_sequence)} grenades")

            # CT setups
            ct_setups = playbook.ct_setups.get(map_name, [])
            if ct_setups:
                md.append("\n### CT Setups")
                for setup in ct_setups[:3]:
                    wr = round(setup.rounds_won / max(1, setup.times_used) * 100)
                    md.append(f"\n**{setup.name}** (Win rate: {wr}%)")
                    md.append(
                        f"- A: {setup.a_players} | B: {setup.b_players} | Mid: {setup.mid_players}"
                    )

            # Retakes
            retakes = playbook.retakes.get(map_name, [])
            if retakes:
                md.append("\n### Retake Protocols")
                for retake in retakes:
                    wr = round(retake.times_succeeded / max(1, retake.times_attempted) * 100)
                    md.append(f"\n**{retake.name}** (Success: {wr}%)")

        return "\n".join(md)


# ============================================================================
# Convenience Functions
# ============================================================================

_default_generator: PlaybookGenerator | None = None


def get_generator() -> PlaybookGenerator:
    """Get or create the default playbook generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = PlaybookGenerator()
    return _default_generator


def analyze_team_demo(
    demo_data: dict[str, Any], team_steamids: list[str], team_name: str
) -> dict[str, Any]:
    """
    Analyze a team demo and add to playbook.

    Args:
        demo_data: Demo data
        team_steamids: Steam IDs of team members
        team_name: Team name

    Returns:
        Analysis summary
    """
    return get_generator().analyze_demo(demo_data, team_steamids, team_name)


def get_playbook(team_name: str) -> dict[str, Any]:
    """Get team playbook as dictionary."""
    return get_generator().get_playbook(team_name).to_dict()


def get_playbook_report(team_name: str) -> dict[str, Any]:
    """Get comprehensive playbook report."""
    return get_generator().generate_playbook_report(team_name)


def export_playbook(team_name: str, format: str = "json") -> str:
    """Export playbook in specified format."""
    return get_generator().export_playbook(team_name, format)
