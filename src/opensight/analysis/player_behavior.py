"""
OpenSight Player Behavior Analyzer

Analyzes player movement patterns, counter-strafing, positioning,
and mechanical skill indicators from CS2 demo data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
import math
import logging

import numpy as np
import pandas as pd

from opensight.core.parser import DemoData
from opensight.core.constants import CS2_TICK_RATE

logger = logging.getLogger(__name__)

# Movement constants
WALK_SPEED = 130.0  # Units per second when walking
RUN_SPEED = 250.0   # Units per second when running (knife)
CROUCH_SPEED = 85.0 # Units per second when crouching

# Counter-strafe thresholds
COUNTER_STRAFE_WINDOW_TICKS = 8  # ~125ms window
VELOCITY_STOP_THRESHOLD = 34.0  # Considered "stopped" for accurate shooting


class MovementType(str, Enum):
    """Type of movement detected."""
    STATIONARY = "stationary"
    WALKING = "walking"
    RUNNING = "running"
    CROUCHING = "crouching"
    JUMPING = "jumping"
    FALLING = "falling"


class StrafeDirection(str, Enum):
    """Direction of strafe movement."""
    LEFT = "left"
    RIGHT = "right"
    FORWARD = "forward"
    BACKWARD = "backward"
    NONE = "none"


@dataclass
class MovementSample:
    """A single movement sample from tick data."""
    tick: int
    steamid: int
    position: Tuple[float, float, float]  # X, Y, Z
    velocity: Tuple[float, float, float]  # vX, vY, vZ
    view_angles: Tuple[float, float]  # pitch, yaw
    speed: float
    movement_type: MovementType
    is_on_ground: bool = True


@dataclass
class StrafeAnalysis:
    """Analysis of a strafe sequence."""
    start_tick: int
    end_tick: int
    direction: StrafeDirection
    max_speed: float
    entry_speed: float
    exit_speed: float
    duration_ticks: int
    distance_traveled: float
    counter_strafed: bool = False
    counter_strafe_efficiency: float = 0.0  # 0-1, how quickly speed was nullified


@dataclass
class CounterStrafeMetrics:
    """Metrics for counter-strafing ability."""
    total_engagements: int = 0
    counter_strafed_engagements: int = 0
    avg_velocity_at_shot: float = 0.0
    avg_counter_strafe_time_ms: float = 0.0
    perfect_counter_strafes: int = 0  # velocity < 5 at shot
    counter_strafe_rate: float = 0.0


@dataclass
class PositioningMetrics:
    """Metrics for positioning behavior."""
    avg_engagement_distance: float = 0.0
    preferred_angles: List[str] = field(default_factory=list)  # Common holding angles
    time_in_open: float = 0.0  # Percentage of time in open areas
    crosshair_placement_avg: float = 0.0  # Average angular error to head level
    pre_aim_accuracy: float = 0.0  # How often crosshair was near enemy before seeing them


@dataclass
class PlayerBehaviorProfile:
    """Complete behavioral profile for a player."""
    steamid: int
    name: str
    movement_samples: int
    avg_speed: float
    max_speed: float
    time_moving_pct: float
    time_crouching_pct: float
    time_jumping_pct: float
    counter_strafe_metrics: CounterStrafeMetrics
    positioning_metrics: PositioningMetrics
    strafe_sequences: List[StrafeAnalysis] = field(default_factory=list)

    # Derived skill indicators
    movement_skill_score: float = 0.0  # 0-100
    positioning_skill_score: float = 0.0  # 0-100


class PlayerBehaviorAnalyzer:
    """
    Analyzes player movement patterns and mechanical skills.

    Examines tick-level data to identify:
    - Movement patterns (strafing, counter-strafing)
    - Positioning tendencies
    - Mechanical skill indicators

    Example:
        >>> from opensight.analysis.player_behavior import PlayerBehaviorAnalyzer
        >>> from opensight.core.parser import DemoParser
        >>>
        >>> parser = DemoParser()
        >>> demo_data = parser.parse("match.dem", parse_ticks=True)
        >>> analyzer = PlayerBehaviorAnalyzer(demo_data)
        >>>
        >>> # Analyze specific player
        >>> profile = analyzer.analyze_player(76561198012345678)
        >>> print(f"Counter-strafe rate: {profile.counter_strafe_metrics.counter_strafe_rate:.1%}")
        >>>
        >>> # Get all player profiles
        >>> profiles = analyzer.analyze_all_players()
    """

    def __init__(self, demo_data: DemoData):
        """
        Initialize the behavior analyzer.

        Args:
            demo_data: Parsed demo data (must include tick data for full analysis)
        """
        self.demo_data = demo_data
        self._tick_rate = demo_data.tick_rate or CS2_TICK_RATE
        self._ms_per_tick = 1000 / self._tick_rate

        # Check if tick data is available
        self._has_tick_data = (
            hasattr(demo_data, 'ticks_df') and
            demo_data.ticks_df is not None and
            len(demo_data.ticks_df) > 0
        )

        if not self._has_tick_data:
            logger.warning(
                "No tick data available. Movement analysis will be limited. "
                "Re-parse with parse_ticks=True for full analysis."
            )

    def analyze_player(self, steamid: int) -> PlayerBehaviorProfile:
        """
        Analyze behavior for a specific player.

        Args:
            steamid: Player's Steam ID

        Returns:
            PlayerBehaviorProfile with movement and skill analysis
        """
        name = self.demo_data.player_names.get(steamid, "Unknown")

        if not self._has_tick_data:
            return self._create_limited_profile(steamid, name)

        # Get player's tick data
        player_ticks = self._get_player_ticks(steamid)

        if player_ticks.empty:
            return self._create_limited_profile(steamid, name)

        # Extract movement samples
        samples = self._extract_movement_samples(steamid, player_ticks)

        # Analyze strafing patterns
        strafe_sequences = self._analyze_strafes(samples)

        # Analyze counter-strafing at engagements
        counter_strafe_metrics = self._analyze_counter_strafing(steamid, samples)

        # Analyze positioning
        positioning_metrics = self._analyze_positioning(steamid, samples)

        # Calculate aggregate metrics
        speeds = [s.speed for s in samples]
        movement_types = [s.movement_type for s in samples]

        profile = PlayerBehaviorProfile(
            steamid=steamid,
            name=name,
            movement_samples=len(samples),
            avg_speed=np.mean(speeds) if speeds else 0.0,
            max_speed=max(speeds) if speeds else 0.0,
            time_moving_pct=self._calc_moving_pct(movement_types),
            time_crouching_pct=self._calc_type_pct(movement_types, MovementType.CROUCHING),
            time_jumping_pct=self._calc_type_pct(movement_types, MovementType.JUMPING),
            counter_strafe_metrics=counter_strafe_metrics,
            positioning_metrics=positioning_metrics,
            strafe_sequences=strafe_sequences[:50]  # Limit stored sequences
        )

        # Calculate skill scores
        profile.movement_skill_score = self._calculate_movement_skill(profile)
        profile.positioning_skill_score = self._calculate_positioning_skill(profile)

        return profile

    def analyze_all_players(self) -> Dict[int, PlayerBehaviorProfile]:
        """
        Analyze behavior for all players in the demo.

        Returns:
            Dictionary mapping steamid to PlayerBehaviorProfile
        """
        profiles = {}
        for steamid in self.demo_data.player_names.keys():
            try:
                profiles[steamid] = self.analyze_player(steamid)
            except Exception as e:
                logger.error(f"Error analyzing player {steamid}: {e}")
        return profiles

    def _get_player_ticks(self, steamid: int) -> pd.DataFrame:
        """Get tick data for a specific player."""
        if not self._has_tick_data:
            return pd.DataFrame()

        df = self.demo_data.ticks_df
        return df[df['steamid'] == steamid].copy()

    def _extract_movement_samples(
        self,
        steamid: int,
        player_ticks: pd.DataFrame
    ) -> List[MovementSample]:
        """Extract movement samples from tick data."""
        samples = []

        for _, row in player_ticks.iterrows():
            # Get position
            pos = (
                float(row.get('X', 0)),
                float(row.get('Y', 0)),
                float(row.get('Z', 0))
            )

            # Get velocity (if available)
            vel = (
                float(row.get('vX', 0)),
                float(row.get('vY', 0)),
                float(row.get('vZ', 0))
            )

            # Get view angles
            view = (
                float(row.get('pitch', 0)),
                float(row.get('yaw', 0))
            )

            # Calculate horizontal speed
            speed = math.sqrt(vel[0]**2 + vel[1]**2)

            # Determine movement type
            movement_type = self._classify_movement(speed, vel[2], row)

            # Check if on ground
            is_on_ground = vel[2] == 0 or abs(vel[2]) < 10

            samples.append(MovementSample(
                tick=int(row.get('tick', 0)),
                steamid=steamid,
                position=pos,
                velocity=vel,
                view_angles=view,
                speed=speed,
                movement_type=movement_type,
                is_on_ground=is_on_ground
            ))

        return samples

    def _classify_movement(
        self,
        speed: float,
        vertical_vel: float,
        row: pd.Series
    ) -> MovementType:
        """Classify the type of movement based on speed and state."""
        # Check for jumping/falling
        if vertical_vel > 50:
            return MovementType.JUMPING
        elif vertical_vel < -50:
            return MovementType.FALLING

        # Check for crouching (if data available)
        if row.get('is_ducking', False) or row.get('ducking', False):
            return MovementType.CROUCHING

        # Classify by speed
        if speed < 5:
            return MovementType.STATIONARY
        elif speed < WALK_SPEED + 20:
            return MovementType.WALKING
        else:
            return MovementType.RUNNING

    def _analyze_strafes(self, samples: List[MovementSample]) -> List[StrafeAnalysis]:
        """Analyze strafe sequences from movement samples."""
        strafes = []
        if len(samples) < 3:
            return strafes

        current_strafe: Optional[Dict] = None

        for i in range(1, len(samples)):
            prev = samples[i - 1]
            curr = samples[i]

            # Detect direction change
            direction = self._get_strafe_direction(prev, curr)

            if direction != StrafeDirection.NONE:
                if current_strafe is None:
                    # Start new strafe
                    current_strafe = {
                        'start_tick': curr.tick,
                        'direction': direction,
                        'max_speed': curr.speed,
                        'entry_speed': prev.speed,
                        'positions': [curr.position]
                    }
                elif current_strafe['direction'] == direction:
                    # Continue strafe
                    current_strafe['max_speed'] = max(
                        current_strafe['max_speed'],
                        curr.speed
                    )
                    current_strafe['positions'].append(curr.position)
                else:
                    # Direction changed - end current strafe
                    if len(current_strafe['positions']) >= 2:
                        strafe = self._finalize_strafe(current_strafe, prev)
                        strafes.append(strafe)

                    # Start new strafe
                    current_strafe = {
                        'start_tick': curr.tick,
                        'direction': direction,
                        'max_speed': curr.speed,
                        'entry_speed': prev.speed,
                        'positions': [curr.position]
                    }
            else:
                # No strafe - end current if exists
                if current_strafe and len(current_strafe['positions']) >= 2:
                    strafe = self._finalize_strafe(current_strafe, prev)
                    strafes.append(strafe)
                current_strafe = None

        return strafes

    def _get_strafe_direction(
        self,
        prev: MovementSample,
        curr: MovementSample
    ) -> StrafeDirection:
        """Determine strafe direction from velocity change."""
        if not prev.is_on_ground or not curr.is_on_ground:
            return StrafeDirection.NONE

        # Get velocity components relative to view direction
        yaw_rad = math.radians(curr.view_angles[1])
        forward = (math.cos(yaw_rad), math.sin(yaw_rad))
        right = (-forward[1], forward[0])

        # Project velocity onto forward/right
        vel = curr.velocity[:2]
        forward_vel = vel[0] * forward[0] + vel[1] * forward[1]
        right_vel = vel[0] * right[0] + vel[1] * right[1]

        # Determine dominant direction
        if abs(right_vel) > abs(forward_vel) and abs(right_vel) > 30:
            return StrafeDirection.RIGHT if right_vel > 0 else StrafeDirection.LEFT
        elif abs(forward_vel) > 30:
            return StrafeDirection.FORWARD if forward_vel > 0 else StrafeDirection.BACKWARD

        return StrafeDirection.NONE

    def _finalize_strafe(
        self,
        strafe_data: Dict,
        final_sample: MovementSample
    ) -> StrafeAnalysis:
        """Finalize a strafe sequence analysis."""
        positions = strafe_data['positions']

        # Calculate distance
        distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance += math.sqrt(dx**2 + dy**2)

        # Check for counter-strafe
        counter_strafed = final_sample.speed < VELOCITY_STOP_THRESHOLD
        efficiency = 1.0 - (final_sample.speed / max(strafe_data['max_speed'], 1))

        return StrafeAnalysis(
            start_tick=strafe_data['start_tick'],
            end_tick=final_sample.tick,
            direction=strafe_data['direction'],
            max_speed=strafe_data['max_speed'],
            entry_speed=strafe_data['entry_speed'],
            exit_speed=final_sample.speed,
            duration_ticks=final_sample.tick - strafe_data['start_tick'],
            distance_traveled=distance,
            counter_strafed=counter_strafed,
            counter_strafe_efficiency=max(0, efficiency)
        )

    def _analyze_counter_strafing(
        self,
        steamid: int,
        samples: List[MovementSample]
    ) -> CounterStrafeMetrics:
        """Analyze counter-strafing at engagement moments."""
        metrics = CounterStrafeMetrics()

        # Get kills by this player
        player_kills = [
            k for k in self.demo_data.kills
            if k.attacker_steamid == steamid
        ]

        if not player_kills or not samples:
            return metrics

        # Build tick -> sample lookup
        tick_to_sample = {s.tick: s for s in samples}

        velocities_at_shot = []
        counter_strafe_times = []

        for kill in player_kills:
            kill_tick = kill.tick
            metrics.total_engagements += 1

            # Find sample at or near kill tick
            sample = tick_to_sample.get(kill_tick)
            if not sample:
                # Find closest
                closest = min(samples, key=lambda s: abs(s.tick - kill_tick))
                if abs(closest.tick - kill_tick) < 10:
                    sample = closest

            if sample:
                velocities_at_shot.append(sample.speed)

                if sample.speed < VELOCITY_STOP_THRESHOLD:
                    metrics.counter_strafed_engagements += 1

                if sample.speed < 5:
                    metrics.perfect_counter_strafes += 1

                # Calculate time to stop (look backwards)
                stop_time = self._calc_time_to_stop(
                    samples,
                    kill_tick,
                    tick_to_sample
                )
                if stop_time is not None:
                    counter_strafe_times.append(stop_time)

        if velocities_at_shot:
            metrics.avg_velocity_at_shot = np.mean(velocities_at_shot)

        if counter_strafe_times:
            metrics.avg_counter_strafe_time_ms = np.mean(counter_strafe_times)

        if metrics.total_engagements > 0:
            metrics.counter_strafe_rate = (
                metrics.counter_strafed_engagements / metrics.total_engagements
            )

        return metrics

    def _calc_time_to_stop(
        self,
        samples: List[MovementSample],
        target_tick: int,
        tick_to_sample: Dict[int, MovementSample]
    ) -> Optional[float]:
        """Calculate time taken to stop before a shot."""
        # Look backwards from target tick
        for i in range(COUNTER_STRAFE_WINDOW_TICKS):
            check_tick = target_tick - i
            sample = tick_to_sample.get(check_tick)

            if sample and sample.speed > RUN_SPEED * 0.7:
                # Found when player was moving fast
                return i * self._ms_per_tick

        return None

    def _analyze_positioning(
        self,
        steamid: int,
        samples: List[MovementSample]
    ) -> PositioningMetrics:
        """Analyze positioning behavior."""
        metrics = PositioningMetrics()

        # Get kills involving this player
        player_kills = [
            k for k in self.demo_data.kills
            if k.attacker_steamid == steamid or k.victim_steamid == steamid
        ]

        # Calculate engagement distances
        distances = []
        for kill in player_kills:
            if (hasattr(kill, 'attacker_x') and kill.attacker_x is not None and
                hasattr(kill, 'victim_x') and kill.victim_x is not None):
                dx = kill.attacker_x - kill.victim_x
                dy = kill.attacker_y - kill.victim_y
                dz = (kill.attacker_z or 0) - (kill.victim_z or 0)
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                distances.append(dist)

        if distances:
            metrics.avg_engagement_distance = np.mean(distances)

        return metrics

    def _calc_moving_pct(self, types: List[MovementType]) -> float:
        """Calculate percentage of time moving."""
        if not types:
            return 0.0
        moving = sum(1 for t in types if t != MovementType.STATIONARY)
        return moving / len(types)

    def _calc_type_pct(self, types: List[MovementType], target: MovementType) -> float:
        """Calculate percentage of time in a specific movement type."""
        if not types:
            return 0.0
        count = sum(1 for t in types if t == target)
        return count / len(types)

    def _calculate_movement_skill(self, profile: PlayerBehaviorProfile) -> float:
        """Calculate overall movement skill score (0-100)."""
        score = 50.0  # Base score

        # Counter-strafe rate (up to +25)
        score += profile.counter_strafe_metrics.counter_strafe_rate * 25

        # Perfect counter-strafes bonus (up to +10)
        if profile.counter_strafe_metrics.total_engagements > 0:
            perfect_rate = (
                profile.counter_strafe_metrics.perfect_counter_strafes /
                profile.counter_strafe_metrics.total_engagements
            )
            score += perfect_rate * 10

        # Low average velocity at shot (up to +10)
        if profile.counter_strafe_metrics.avg_velocity_at_shot < 20:
            score += 10
        elif profile.counter_strafe_metrics.avg_velocity_at_shot < 50:
            score += 5

        # Penalize excessive crouching (-5 to 0)
        if profile.time_crouching_pct > 0.3:
            score -= 5

        return max(0, min(100, score))

    def _calculate_positioning_skill(self, profile: PlayerBehaviorProfile) -> float:
        """Calculate positioning skill score (0-100)."""
        score = 50.0  # Base score

        # Good engagement distances (medium range preferred)
        avg_dist = profile.positioning_metrics.avg_engagement_distance
        if 500 < avg_dist < 1500:
            score += 15
        elif 300 < avg_dist < 2000:
            score += 10

        return max(0, min(100, score))

    def _create_limited_profile(self, steamid: int, name: str) -> PlayerBehaviorProfile:
        """Create a limited profile when tick data is unavailable."""
        return PlayerBehaviorProfile(
            steamid=steamid,
            name=name,
            movement_samples=0,
            avg_speed=0.0,
            max_speed=0.0,
            time_moving_pct=0.0,
            time_crouching_pct=0.0,
            time_jumping_pct=0.0,
            counter_strafe_metrics=CounterStrafeMetrics(),
            positioning_metrics=PositioningMetrics()
        )


def analyze_player_behavior(demo_data: DemoData) -> Dict[int, PlayerBehaviorProfile]:
    """
    Convenience function to analyze all player behavior.

    Args:
        demo_data: Parsed demo data

    Returns:
        Dictionary mapping steamid to PlayerBehaviorProfile
    """
    analyzer = PlayerBehaviorAnalyzer(demo_data)
    return analyzer.analyze_all_players()
