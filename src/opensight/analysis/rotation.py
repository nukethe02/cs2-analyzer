"""CT Rotation Latency Analysis Module.

Measures how fast CTs rotate when the opposite site is hit.
Provides IGL-level insights into macro efficiency and rotation patterns.

Key Metrics:
- Reaction Time: Time from site contact to leaving hold position
- Travel Time: Time from leaving position to arriving at contact site
- Rotation Efficiency: Combined score based on speed and safety
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from opensight.core.parser import DemoData, RoundInfo

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Sprint speed threshold (units/second) - CS2 run speed is ~250 u/s
SPRINT_SPEED_THRESHOLD = 200

# Rotation timing thresholds (seconds)
FAST_ROTATION_THRESHOLD = 2.0  # < 2s = over-rotation risk
SLOW_ROTATION_THRESHOLD = 10.0  # > 10s = too passive

# Contact detection thresholds
MIN_T_PLAYERS_FOR_PRESENCE = 2  # Minimum T players in site to trigger presence contact

# CS2 tick rate
TICK_RATE = 64
MS_PER_TICK = 1000 / TICK_RATE


class ContactTriggerType(Enum):
    """Type of event that triggered site contact detection."""

    KILL = "kill"  # First kill in bombsite
    PRESENCE = "presence"  # 2+ T players in bombsite
    BOMB_PLANT = "bomb_plant"  # Bomb planted (backup trigger)
    BOMB_SPOTTED = "bomb_spotted"  # Bomb carrier spotted in site


class RotationOutcome(Enum):
    """Outcome of a rotation attempt."""

    ARRIVED = "arrived"  # CT reached the site
    DIED_ROTATING = "died_rotating"  # CT died while rotating
    ROUND_ENDED = "round_ended"  # Round ended before arrival
    STAYED = "stayed"  # CT never left their position


class RotationClassification(Enum):
    """Classification of rotation behavior."""

    OVER_ROTATOR = "over_rotator"  # Reacts too fast (< 2s)
    BALANCED = "balanced"  # Good timing (2-10s)
    SLOW_ROTATOR = "slow_rotator"  # Reacts too slow (> 10s)
    ANCHOR = "anchor"  # Stayed at position (not a rotator)


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class SiteContactEvent:
    """Represents the moment when significant contact occurs at a bombsite.

    This is the "stimulus" that should trigger CT rotations.
    """

    tick: int
    round_num: int
    site: str  # "A" or "B"
    trigger_type: ContactTriggerType
    trigger_player_steamid: int | None = None  # Who caused the trigger (killer/first T)
    trigger_player_name: str = ""
    time_in_round_seconds: float = 0.0

    def __post_init__(self):
        """Convert trigger_type string to enum if needed."""
        if isinstance(self.trigger_type, str):
            self.trigger_type = ContactTriggerType(self.trigger_type)


@dataclass
class RotationLatency:
    """Tracks a single CT's rotation response to a site contact event.

    Measures both reaction time (hesitation) and travel time (efficiency).
    """

    # Player info
    ct_steamid: int
    ct_name: str
    round_num: int

    # Contact info
    contact_tick: int
    contact_site: str  # Site being attacked ("A" or "B")

    # Starting position
    starting_zone: str  # Where CT was when contact occurred
    starting_x: float = 0.0
    starting_y: float = 0.0

    # Timing measurements
    departure_tick: int | None = None  # When CT left their zone
    arrival_tick: int | None = None  # When CT reached contact site

    # Calculated metrics (in seconds)
    reaction_time_seconds: float | None = None  # Time to start moving
    travel_time_seconds: float | None = None  # Time from departure to arrival
    total_time_seconds: float | None = None  # Total time from contact to arrival

    # Outcome
    outcome: RotationOutcome = RotationOutcome.STAYED
    classification: RotationClassification = RotationClassification.ANCHOR

    # Death info (if died rotating)
    death_tick: int | None = None
    death_zone: str = ""

    def __post_init__(self):
        """Convert enums from strings if needed."""
        if isinstance(self.outcome, str):
            self.outcome = RotationOutcome(self.outcome)
        if isinstance(self.classification, str):
            self.classification = RotationClassification(self.classification)


@dataclass
class PlayerRotationStats:
    """Aggregated rotation statistics for a single player across all rounds."""

    steamid: int
    name: str
    team: str  # Should be "CT" for meaningful stats

    # Rotation counts
    total_rotation_opportunities: int = 0  # Times they could have rotated
    rotations_attempted: int = 0  # Times they actually rotated
    rotations_completed: int = 0  # Arrived at site
    rotations_died: int = 0  # Died while rotating

    # Timing aggregates (in seconds)
    reaction_times: list[float] = field(default_factory=list)
    travel_times: list[float] = field(default_factory=list)
    total_times: list[float] = field(default_factory=list)

    # Classification counts
    over_rotations: int = 0  # Reaction < 2s
    balanced_rotations: int = 0  # 2s <= Reaction <= 10s
    slow_rotations: int = 0  # Reaction > 10s
    anchor_rounds: int = 0  # Stayed in position

    @property
    def avg_reaction_time(self) -> float | None:
        """Average reaction time in seconds."""
        if not self.reaction_times:
            return None
        return round(float(np.mean(self.reaction_times)), 2)

    @property
    def avg_travel_time(self) -> float | None:
        """Average travel time in seconds."""
        if not self.travel_times:
            return None
        return round(float(np.mean(self.travel_times)), 2)

    @property
    def avg_total_time(self) -> float | None:
        """Average total rotation time in seconds."""
        if not self.total_times:
            return None
        return round(float(np.mean(self.total_times)), 2)

    @property
    def rotation_rate(self) -> float:
        """Percentage of opportunities where player rotated."""
        if self.total_rotation_opportunities == 0:
            return 0.0
        return round(self.rotations_attempted / self.total_rotation_opportunities * 100, 1)

    @property
    def rotation_success_rate(self) -> float:
        """Percentage of rotations that completed successfully."""
        if self.rotations_attempted == 0:
            return 0.0
        return round(self.rotations_completed / self.rotations_attempted * 100, 1)

    @property
    def primary_classification(self) -> RotationClassification:
        """Most common rotation behavior for this player."""
        counts = {
            RotationClassification.OVER_ROTATOR: self.over_rotations,
            RotationClassification.BALANCED: self.balanced_rotations,
            RotationClassification.SLOW_ROTATOR: self.slow_rotations,
            RotationClassification.ANCHOR: self.anchor_rounds,
        }
        return max(counts, key=counts.get)

    @property
    def efficiency_rating(self) -> float:
        """Rotation efficiency score (0-100).

        Based on:
        - Fast reaction time (40 points max)
        - Successful arrivals (40 points max)
        - Not over-rotating (20 points max)
        """
        if self.total_rotation_opportunities == 0:
            return 50.0  # Default/unknown

        score = 50.0  # Start at average

        # Reaction time component (-20 to +20)
        if self.avg_reaction_time is not None:
            if self.avg_reaction_time < 3.0:
                score += 20
            elif self.avg_reaction_time < 5.0:
                score += 10
            elif self.avg_reaction_time > 10.0:
                score -= 20
            elif self.avg_reaction_time > 7.0:
                score -= 10

        # Success rate component (-20 to +20)
        if self.rotations_attempted > 0:
            success_rate = self.rotation_success_rate
            if success_rate >= 80:
                score += 20
            elif success_rate >= 60:
                score += 10
            elif success_rate < 40:
                score -= 10
            elif success_rate < 20:
                score -= 20

        # Over-rotation penalty (-10 to 0)
        if self.rotations_attempted > 0:
            over_rotation_rate = self.over_rotations / self.rotations_attempted
            if over_rotation_rate > 0.5:
                score -= 10
            elif over_rotation_rate > 0.3:
                score -= 5

        return round(max(0, min(100, score)), 1)

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "steamid": str(self.steamid),
            "name": self.name,
            "total_rotation_opportunities": self.total_rotation_opportunities,
            "rotations_attempted": self.rotations_attempted,
            "rotations_completed": self.rotations_completed,
            "rotations_died": self.rotations_died,
            "avg_reaction_time_sec": self.avg_reaction_time,
            "avg_travel_time_sec": self.avg_travel_time,
            "avg_total_time_sec": self.avg_total_time,
            "rotation_rate_pct": self.rotation_rate,
            "rotation_success_rate_pct": self.rotation_success_rate,
            "over_rotations": self.over_rotations,
            "balanced_rotations": self.balanced_rotations,
            "slow_rotations": self.slow_rotations,
            "anchor_rounds": self.anchor_rounds,
            "primary_classification": self.primary_classification.value,
            "efficiency_rating": self.efficiency_rating,
        }


@dataclass
class TeamRotationStats:
    """Team-level rotation statistics."""

    team: str  # "CT"
    rounds_analyzed: int = 0

    # Site-specific
    a_site_contacts: int = 0
    b_site_contacts: int = 0

    # Aggregate timing
    avg_team_reaction_time: float | None = None
    avg_team_travel_time: float | None = None

    # Classification counts
    total_over_rotations: int = 0
    total_balanced_rotations: int = 0
    total_slow_rotations: int = 0

    # Player stats
    player_stats: dict[int, PlayerRotationStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "team": self.team,
            "rounds_analyzed": self.rounds_analyzed,
            "a_site_contacts": self.a_site_contacts,
            "b_site_contacts": self.b_site_contacts,
            "avg_team_reaction_time_sec": self.avg_team_reaction_time,
            "avg_team_travel_time_sec": self.avg_team_travel_time,
            "total_over_rotations": self.total_over_rotations,
            "total_balanced_rotations": self.total_balanced_rotations,
            "total_slow_rotations": self.total_slow_rotations,
            "player_stats": {str(sid): stats.to_dict() for sid, stats in self.player_stats.items()},
        }


# =============================================================================
# CT ROTATION ANALYZER
# =============================================================================


class CTRotationAnalyzer:
    """Analyzes CT rotation patterns and latency.

    Usage:
        analyzer = CTRotationAnalyzer(demo_data, map_name)
        results = analyzer.analyze()
    """

    def __init__(self, demo_data: DemoData, map_name: str):
        """Initialize the analyzer.

        Args:
            demo_data: Parsed demo data with positions, kills, rounds
            map_name: Map name for zone lookups (e.g., "de_mirage")
        """
        self.data = demo_data
        self.map_name = map_name.lower()

        # Results storage
        self.contact_events: list[SiteContactEvent] = []
        self.rotation_latencies: list[RotationLatency] = []
        self.team_stats: TeamRotationStats | None = None

        # Import zone detection
        try:
            from opensight.visualization.radar import MAP_ZONES, get_zone_for_position

            self._get_zone = get_zone_for_position
            self._map_zones = MAP_ZONES
        except ImportError:
            logger.warning("Could not import zone detection - rotation analysis disabled")
            self._get_zone = None
            self._map_zones = {}

    def analyze(self) -> TeamRotationStats:
        """Run full rotation analysis.

        Returns:
            TeamRotationStats with per-player and aggregate metrics
        """
        if self._get_zone is None:
            logger.warning("Zone detection not available")
            return TeamRotationStats(team="CT")

        if self.map_name not in self._map_zones:
            logger.warning(f"No zone data for map: {self.map_name}")
            return TeamRotationStats(team="CT")

        # Check for required data
        if self.data.ticks_df is None or self.data.ticks_df.empty:
            logger.warning("No tick data available for rotation analysis")
            return TeamRotationStats(team="CT")

        logger.info(f"Starting CT rotation analysis for {self.map_name}")

        # Step 1: Detect site contact events for each round
        self._detect_all_contact_events()

        # Step 2: Calculate rotation latencies for each contact
        self._calculate_all_rotations()

        # Step 3: Aggregate into player and team stats
        self._aggregate_stats()

        logger.info(
            f"Rotation analysis complete: {len(self.contact_events)} contacts, "
            f"{len(self.rotation_latencies)} rotation opportunities"
        )

        return self.team_stats

    def _detect_all_contact_events(self) -> None:
        """Detect site contact events for all rounds."""
        self.contact_events = []

        for round_info in self.data.rounds:
            # Skip warmup/knife rounds
            if round_info.round_num <= 0:
                continue

            contacts = self._detect_site_contact_for_round(round_info)
            self.contact_events.extend(contacts)

        logger.info(f"Detected {len(self.contact_events)} site contact events")

    def _detect_site_contact_for_round(self, round_info: RoundInfo) -> list[SiteContactEvent]:
        """Detect site contact events for a single round.

        Checks three triggers (in order of priority):
        1. First kill in a bombsite
        2. 2+ T players in a bombsite
        3. Bomb plant

        Args:
            round_info: Round information

        Returns:
            List of SiteContactEvent (usually 0-2, one per site)
        """
        contacts = []
        round_num = round_info.round_num
        round_start = round_info.freeze_end_tick or round_info.start_tick
        round_end = round_info.end_tick

        # Track which sites we've already detected contact for
        contacted_sites: dict[str, SiteContactEvent] = {}

        # --- Trigger 1: First kill in bombsite ---
        kills_in_round = [
            k
            for k in self.data.kills
            if k.round_num == round_num and round_start <= k.tick <= round_end
        ]

        for kill in sorted(kills_in_round, key=lambda k: k.tick):
            # Get zone where kill occurred (victim position)
            zone = self._get_zone(self.map_name, kill.victim_x, kill.victim_y, kill.victim_z)

            if zone and self._is_bombsite_zone(zone):
                site = self._get_site_from_zone(zone)
                if site and site not in contacted_sites:
                    time_in_round = (kill.tick - round_start) * MS_PER_TICK / 1000

                    contacted_sites[site] = SiteContactEvent(
                        tick=kill.tick,
                        round_num=round_num,
                        site=site,
                        trigger_type=ContactTriggerType.KILL,
                        trigger_player_steamid=kill.attacker_steamid,
                        trigger_player_name=kill.attacker_name,
                        time_in_round_seconds=time_in_round,
                    )

        # --- Trigger 2: T presence in bombsite (2+ players) ---
        presence_contacts = self._detect_presence_contacts(
            round_info, round_start, round_end, contacted_sites
        )

        for site, contact in presence_contacts.items():
            if site not in contacted_sites or contact.tick < contacted_sites[site].tick:
                contacted_sites[site] = contact

        # --- Trigger 3: Bomb plant (backup) ---
        if round_info.bomb_plant_tick and round_info.bomb_site:
            site = round_info.bomb_site.upper()
            if len(site) == 1 and site in ["A", "B"]:
                if site not in contacted_sites:
                    time_in_round = (round_info.bomb_plant_tick - round_start) * MS_PER_TICK / 1000

                    contacted_sites[site] = SiteContactEvent(
                        tick=round_info.bomb_plant_tick,
                        round_num=round_num,
                        site=site,
                        trigger_type=ContactTriggerType.BOMB_PLANT,
                        time_in_round_seconds=time_in_round,
                    )

        contacts = list(contacted_sites.values())
        return contacts

    def _detect_presence_contacts(
        self,
        round_info: RoundInfo,
        round_start: int,
        round_end: int,
        existing_contacts: dict[str, SiteContactEvent],
    ) -> dict[str, SiteContactEvent]:
        """Detect when 2+ T players enter a bombsite.

        Args:
            round_info: Round information
            round_start: Round start tick
            round_end: Round end tick
            existing_contacts: Already detected contacts (for comparison)

        Returns:
            Dict mapping site to SiteContactEvent
        """
        contacts: dict[str, SiteContactEvent] = {}
        ticks_df = self.data.ticks_df

        if ticks_df is None or ticks_df.empty:
            return contacts

        # Find steamid column
        steamid_col = None
        for col in ["steamid", "steam_id", "user_steamid"]:
            if col in ticks_df.columns:
                steamid_col = col
                break

        if not steamid_col:
            return contacts

        # Get T-side players
        t_players = set()
        for sid, team in self.data.player_teams.items():
            if team == 2:  # T side
                t_players.add(sid)

        if not t_players:
            return contacts

        # Filter ticks to this round
        round_ticks = ticks_df[(ticks_df["tick"] >= round_start) & (ticks_df["tick"] <= round_end)]

        if round_ticks.empty:
            return contacts

        # Sample ticks (every 64 ticks = 1 second for performance)
        unique_ticks = sorted(round_ticks["tick"].unique())
        sample_ticks = unique_ticks[::64]  # Every second

        for tick in sample_ticks:
            tick_data = round_ticks[round_ticks["tick"] == tick]

            # Count T players in each bombsite
            site_counts: dict[str, list[int]] = {"A": [], "B": []}

            for _, row in tick_data.iterrows():
                player_id = int(row[steamid_col])
                if player_id not in t_players:
                    continue

                x, y, z = row.get("X", 0), row.get("Y", 0), row.get("Z", 0)
                zone = self._get_zone(self.map_name, x, y, z)

                if zone and self._is_bombsite_zone(zone):
                    site = self._get_site_from_zone(zone)
                    if site:
                        site_counts[site].append(player_id)

            # Check if 2+ T players in any site
            for site, players in site_counts.items():
                if len(players) >= MIN_T_PLAYERS_FOR_PRESENCE:
                    if site not in contacts:
                        # Only use if earlier than existing contact
                        if site in existing_contacts and tick >= existing_contacts[site].tick:
                            continue

                        time_in_round = (tick - round_start) * MS_PER_TICK / 1000

                        contacts[site] = SiteContactEvent(
                            tick=tick,
                            round_num=round_info.round_num,
                            site=site,
                            trigger_type=ContactTriggerType.PRESENCE,
                            trigger_player_steamid=players[0] if players else None,
                            time_in_round_seconds=time_in_round,
                        )

        return contacts

    def _calculate_all_rotations(self) -> None:
        """Calculate rotation latencies for all contact events."""
        self.rotation_latencies = []

        for contact in self.contact_events:
            rotations = self._calculate_rotations_for_contact(contact)
            self.rotation_latencies.extend(rotations)

    def _calculate_rotations_for_contact(self, contact: SiteContactEvent) -> list[RotationLatency]:
        """Calculate rotation latencies for CTs responding to a site contact.

        Args:
            contact: The site contact event

        Returns:
            List of RotationLatency for each CT who could rotate
        """
        rotations = []
        ticks_df = self.data.ticks_df

        if ticks_df is None or ticks_df.empty:
            return rotations

        # Find round info
        round_info = None
        for r in self.data.rounds:
            if r.round_num == contact.round_num:
                round_info = r
                break

        if not round_info:
            return rotations

        round_end = round_info.end_tick

        # Find steamid column
        steamid_col = None
        for col in ["steamid", "steam_id", "user_steamid"]:
            if col in ticks_df.columns:
                steamid_col = col
                break

        if not steamid_col:
            return rotations

        # Get CT players
        ct_players = {}
        for sid, team in self.data.player_teams.items():
            if team == 3:  # CT side
                name = self.data.player_names.get(sid, f"Player_{sid}")
                ct_players[sid] = name

        # Get CT positions at contact tick
        contact_tick_data = ticks_df[ticks_df["tick"] == contact.tick]

        if contact_tick_data.empty:
            # Try to find nearest tick
            closest_idx = (ticks_df["tick"] - contact.tick).abs().idxmin()
            contact_tick_data = ticks_df.loc[[closest_idx]]

        # For each CT, determine if they should rotate
        for ct_id, ct_name in ct_players.items():
            ct_data = contact_tick_data[contact_tick_data[steamid_col] == ct_id]

            if ct_data.empty:
                continue

            ct_row = ct_data.iloc[0]
            x, y, z = ct_row.get("X", 0), ct_row.get("Y", 0), ct_row.get("Z", 0)
            starting_zone = self._get_zone(self.map_name, x, y, z) or "World"

            # Skip if already at contact site (they're the anchor)
            if self._is_zone_at_site(starting_zone, contact.site):
                continue

            # Create rotation tracking entry
            rotation = RotationLatency(
                ct_steamid=ct_id,
                ct_name=ct_name,
                round_num=contact.round_num,
                contact_tick=contact.tick,
                contact_site=contact.site,
                starting_zone=starting_zone,
                starting_x=x,
                starting_y=y,
            )

            # Track this CT's movement after contact
            self._track_rotation_movement(rotation, ticks_df, steamid_col, round_end)

            rotations.append(rotation)

        return rotations

    def _track_rotation_movement(
        self,
        rotation: RotationLatency,
        ticks_df: pd.DataFrame,
        steamid_col: str,
        round_end: int,
    ) -> None:
        """Track a CT's movement from contact until arrival/death/round end.

        Updates the rotation object in place with timing data.

        Args:
            rotation: RotationLatency to update
            ticks_df: Tick data
            steamid_col: Column name for steam ID
            round_end: End tick of the round
        """
        # Get ticks for this player after contact
        player_ticks = ticks_df[
            (ticks_df[steamid_col] == rotation.ct_steamid)
            & (ticks_df["tick"] >= rotation.contact_tick)
            & (ticks_df["tick"] <= round_end)
        ].sort_values("tick")

        if player_ticks.empty:
            rotation.outcome = RotationOutcome.ROUND_ENDED
            return

        # Check if player died
        deaths_in_round = [
            k
            for k in self.data.kills
            if k.victim_steamid == rotation.ct_steamid
            and k.round_num == rotation.round_num
            and k.tick >= rotation.contact_tick
        ]

        death_tick = deaths_in_round[0].tick if deaths_in_round else None

        # Track zone transitions and velocity
        left_zone = False
        current_zone = rotation.starting_zone

        for _, row in player_ticks.iterrows():
            tick = int(row["tick"])

            # Stop if player died
            if death_tick and tick >= death_tick:
                rotation.death_tick = death_tick
                rotation.outcome = RotationOutcome.DIED_ROTATING
                death_zone = self._get_zone(
                    self.map_name,
                    row.get("X", 0),
                    row.get("Y", 0),
                    row.get("Z", 0),
                )
                rotation.death_zone = death_zone or "World"
                break

            # Get current position and zone
            x, y, z = row.get("X", 0), row.get("Y", 0), row.get("Z", 0)
            zone = self._get_zone(self.map_name, x, y, z) or "World"

            # Calculate velocity
            vx = row.get("velocity_X", row.get("velocity_x", 0)) or 0
            vy = row.get("velocity_Y", row.get("velocity_y", 0)) or 0
            speed = np.sqrt(vx**2 + vy**2)

            # Detect departure (left initial zone while sprinting)
            if not left_zone and zone != current_zone and speed >= SPRINT_SPEED_THRESHOLD:
                left_zone = True
                rotation.departure_tick = tick

            # Detect arrival at contact site
            if self._is_zone_at_site(zone, rotation.contact_site):
                rotation.arrival_tick = tick
                rotation.outcome = RotationOutcome.ARRIVED
                break

        # If loop completed without arrival or death
        if rotation.outcome == RotationOutcome.STAYED:
            if left_zone:
                rotation.outcome = RotationOutcome.ROUND_ENDED
            # else stays as STAYED (never left position)

        # Calculate timing metrics
        if rotation.departure_tick:
            rotation.reaction_time_seconds = (
                (rotation.departure_tick - rotation.contact_tick) * MS_PER_TICK / 1000
            )

            if rotation.arrival_tick:
                rotation.travel_time_seconds = (
                    (rotation.arrival_tick - rotation.departure_tick) * MS_PER_TICK / 1000
                )
                rotation.total_time_seconds = (
                    (rotation.arrival_tick - rotation.contact_tick) * MS_PER_TICK / 1000
                )

        # Classify rotation behavior
        rotation.classification = self._classify_rotation(rotation)

    def _classify_rotation(self, rotation: RotationLatency) -> RotationClassification:
        """Classify a rotation based on reaction time.

        Args:
            rotation: RotationLatency to classify

        Returns:
            RotationClassification
        """
        if rotation.outcome == RotationOutcome.STAYED:
            return RotationClassification.ANCHOR

        if rotation.reaction_time_seconds is None:
            return RotationClassification.ANCHOR

        if rotation.reaction_time_seconds < FAST_ROTATION_THRESHOLD:
            return RotationClassification.OVER_ROTATOR
        elif rotation.reaction_time_seconds > SLOW_ROTATION_THRESHOLD:
            return RotationClassification.SLOW_ROTATOR
        else:
            return RotationClassification.BALANCED

    def _aggregate_stats(self) -> None:
        """Aggregate rotation data into player and team statistics."""
        self.team_stats = TeamRotationStats(team="CT")
        self.team_stats.rounds_analyzed = len({c.round_num for c in self.contact_events})

        # Count site contacts
        for contact in self.contact_events:
            if contact.site == "A":
                self.team_stats.a_site_contacts += 1
            elif contact.site == "B":
                self.team_stats.b_site_contacts += 1

        # Aggregate by player
        all_reaction_times = []
        all_travel_times = []

        for rotation in self.rotation_latencies:
            sid = rotation.ct_steamid

            # Initialize player stats if needed
            if sid not in self.team_stats.player_stats:
                self.team_stats.player_stats[sid] = PlayerRotationStats(
                    steamid=sid,
                    name=rotation.ct_name,
                    team="CT",
                )

            pstats = self.team_stats.player_stats[sid]
            pstats.total_rotation_opportunities += 1

            # Count based on outcome
            if rotation.outcome in [RotationOutcome.ARRIVED, RotationOutcome.DIED_ROTATING]:
                pstats.rotations_attempted += 1

                if rotation.outcome == RotationOutcome.ARRIVED:
                    pstats.rotations_completed += 1
                else:
                    pstats.rotations_died += 1

            # Record timing
            if rotation.reaction_time_seconds is not None:
                pstats.reaction_times.append(rotation.reaction_time_seconds)
                all_reaction_times.append(rotation.reaction_time_seconds)

            if rotation.travel_time_seconds is not None:
                pstats.travel_times.append(rotation.travel_time_seconds)
                all_travel_times.append(rotation.travel_time_seconds)

            if rotation.total_time_seconds is not None:
                pstats.total_times.append(rotation.total_time_seconds)

            # Count classifications
            if rotation.classification == RotationClassification.OVER_ROTATOR:
                pstats.over_rotations += 1
                self.team_stats.total_over_rotations += 1
            elif rotation.classification == RotationClassification.BALANCED:
                pstats.balanced_rotations += 1
                self.team_stats.total_balanced_rotations += 1
            elif rotation.classification == RotationClassification.SLOW_ROTATOR:
                pstats.slow_rotations += 1
                self.team_stats.total_slow_rotations += 1
            elif rotation.classification == RotationClassification.ANCHOR:
                pstats.anchor_rounds += 1

        # Calculate team averages
        if all_reaction_times:
            self.team_stats.avg_team_reaction_time = round(float(np.mean(all_reaction_times)), 2)

        if all_travel_times:
            self.team_stats.avg_team_travel_time = round(float(np.mean(all_travel_times)), 2)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _is_bombsite_zone(self, zone_name: str) -> bool:
        """Check if a zone is a bombsite."""
        zone_lower = zone_name.lower()
        return "site" in zone_lower or zone_lower in ["a", "b"]

    def _get_site_from_zone(self, zone_name: str) -> str | None:
        """Extract site letter (A/B) from zone name."""
        zone_lower = zone_name.lower()

        if "a site" in zone_lower or "a_site" in zone_lower or zone_lower == "a":
            return "A"
        elif "b site" in zone_lower or "b_site" in zone_lower or zone_lower == "b":
            return "B"
        elif zone_lower.startswith("bombsite"):
            if "a" in zone_lower:
                return "A"
            elif "b" in zone_lower:
                return "B"

        return None

    def _is_zone_at_site(self, zone_name: str, site: str) -> bool:
        """Check if a zone is at the specified bombsite."""
        zone_site = self._get_site_from_zone(zone_name)
        return zone_site == site

    def get_rotation_advice(self) -> list[dict]:
        """Generate IGL advice based on rotation patterns.

        Returns:
            List of advice items with player-specific recommendations
        """
        advice = []

        if not self.team_stats:
            return advice

        for sid, pstats in self.team_stats.player_stats.items():
            player_advice = {
                "steamid": str(sid),
                "name": pstats.name,
                "classification": pstats.primary_classification.value,
                "avg_reaction_time": pstats.avg_reaction_time,
                "efficiency_rating": pstats.efficiency_rating,
                "issues": [],
                "recommendations": [],
            }

            # Over-rotator advice
            if pstats.over_rotations > pstats.balanced_rotations:
                player_advice["issues"].append(
                    f"Over-rotates on first contact ({pstats.over_rotations} times)"
                )
                player_advice["recommendations"].append(
                    "Wait for secondary info before rotating. Hold for 2-3 seconds."
                )

            # Slow rotator advice
            if pstats.slow_rotations > pstats.balanced_rotations:
                player_advice["issues"].append(
                    f"Rotates too slowly ({pstats.avg_reaction_time:.1f}s average)"
                )
                player_advice["recommendations"].append(
                    "Improve map awareness. Rotate within 5-7 seconds of confirmed contact."
                )

            # Died while rotating
            if pstats.rotations_died > pstats.rotations_completed:
                player_advice["issues"].append(
                    f"Dies frequently while rotating ({pstats.rotations_died} deaths)"
                )
                player_advice["recommendations"].append(
                    "Use safer rotation routes. Smoke/flash before crossing dangerous areas."
                )

            # Good performer
            if (
                pstats.efficiency_rating >= 70
                and pstats.balanced_rotations >= pstats.over_rotations
            ):
                player_advice["recommendations"].append(
                    "Good rotation discipline. Continue current approach."
                )

            advice.append(player_advice)

        return advice


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def analyze_ct_rotations(demo_data: DemoData, map_name: str) -> TeamRotationStats:
    """Analyze CT rotation patterns in a demo.

    Args:
        demo_data: Parsed demo data
        map_name: Map name (e.g., "de_mirage")

    Returns:
        TeamRotationStats with rotation analysis
    """
    analyzer = CTRotationAnalyzer(demo_data, map_name)
    return analyzer.analyze()


def get_rotation_summary(demo_data: DemoData, map_name: str) -> dict:
    """Get a summary of CT rotation analysis.

    Args:
        demo_data: Parsed demo data
        map_name: Map name

    Returns:
        Dictionary with rotation summary for API response
    """
    analyzer = CTRotationAnalyzer(demo_data, map_name)
    stats = analyzer.analyze()
    advice = analyzer.get_rotation_advice()

    return {
        "team_stats": stats.to_dict(),
        "advice": advice,
        "contact_events": [
            {
                "round": c.round_num,
                "site": c.site,
                "trigger": c.trigger_type.value,
                "time_in_round": c.time_in_round_seconds,
            }
            for c in analyzer.contact_events
        ],
    }
