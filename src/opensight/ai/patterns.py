"""
Temporal Pattern Analysis Module for CS2 Demo Analyzer.

Identifies recurring mistakes and behavioral patterns across multiple demos.
Aggregates these patterns to highlight chronic issues that need attention.
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
# Pattern Types and Definitions
# ============================================================================


class PatternCategory(Enum):
    """Categories of behavioral patterns."""

    POSITIONING = "positioning"
    TIMING = "timing"
    AIM = "aim"
    UTILITY = "utility"
    ECONOMY = "economy"
    COMMUNICATION = "communication"
    DECISION_MAKING = "decision_making"
    MECHANICAL = "mechanical"


class PatternSeverity(Enum):
    """Severity levels for identified patterns."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MistakeType(Enum):
    """Types of common mistakes to track."""

    # Positioning mistakes
    WIDE_PEEK = "wide_peek"
    EXPOSED_POSITION = "exposed_position"
    SAME_ANGLE_HOLD = "same_angle_hold"
    OVER_ROTATION = "over_rotation"
    UNDER_ROTATION = "under_rotation"
    LATE_ROTATION = "late_rotation"

    # Timing mistakes
    DRY_PEEK = "dry_peek"
    PEEKING_ALONE = "peeking_alone"
    LATE_TRADE = "late_trade"
    PREMATURE_ROTATE = "premature_rotate"
    SLOW_EXECUTE = "slow_execute"

    # Aim mistakes
    POOR_SPRAY_CONTROL = "poor_spray_control"
    OVERAIMING = "overaiming"
    UNDERAIMING = "underaiming"
    WRONG_CROSSHAIR_HEIGHT = "wrong_crosshair_height"
    SLOW_ADJUSTMENT = "slow_adjustment"

    # Utility mistakes
    WASTED_FLASH = "wasted_flash"
    TEAM_FLASH = "team_flash"
    EARLY_SMOKE = "early_smoke"
    LATE_SMOKE = "late_smoke"
    MISSED_MOLLY = "missed_molly"
    NO_UTILITY_BUY = "no_utility_buy"

    # Economy mistakes
    FORCE_BUY_BROKE_TEAM = "force_buy_broke_team"
    NO_SAVE = "no_save"
    WEAPON_LOSS = "weapon_loss"
    OVERBUYING = "overbuying"

    # Decision mistakes
    UNNECESSARY_AGGRESSION = "unnecessary_aggression"
    TOO_PASSIVE = "too_passive"
    WRONG_BOMBSITE = "wrong_bombsite"
    BAD_RETAKE_TIMING = "bad_retake_timing"


@dataclass
class MistakeInstance:
    """Single instance of a detected mistake."""

    mistake_type: MistakeType
    demo_id: str
    round_num: int
    tick: int
    timestamp_ms: float
    map_name: str
    position: tuple[float, float, float] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    severity: PatternSeverity = PatternSeverity.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "mistake_type": self.mistake_type.value,
            "demo_id": self.demo_id,
            "round_num": self.round_num,
            "tick": self.tick,
            "timestamp_ms": self.timestamp_ms,
            "map_name": self.map_name,
            "position": self.position,
            "context": self.context,
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MistakeInstance":
        return cls(
            mistake_type=MistakeType(data["mistake_type"]),
            demo_id=data["demo_id"],
            round_num=data["round_num"],
            tick=data["tick"],
            timestamp_ms=data["timestamp_ms"],
            map_name=data["map_name"],
            position=tuple(data["position"]) if data.get("position") else None,
            context=data.get("context", {}),
            severity=PatternSeverity(data.get("severity", "medium")),
        )


@dataclass
class RecurringPattern:
    """A pattern identified across multiple demos."""

    pattern_id: str
    mistake_type: MistakeType
    category: PatternCategory
    frequency: int  # Total occurrences
    demos_count: int  # Number of demos where it appeared
    maps_affected: list[str]
    positions_common: list[tuple[float, float, float]]  # Common positions

    # Statistical analysis
    avg_per_demo: float
    trend: str  # "improving", "worsening", "stable"
    first_seen: str
    last_seen: str

    # Contextual data
    common_contexts: list[dict[str, Any]]
    instances: list[MistakeInstance] = field(default_factory=list)

    # Coaching
    description: str = ""
    root_cause: str = ""
    improvement_suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "mistake_type": self.mistake_type.value,
            "category": self.category.value,
            "frequency": self.frequency,
            "demos_count": self.demos_count,
            "maps_affected": self.maps_affected,
            "positions_common": self.positions_common,
            "avg_per_demo": round(self.avg_per_demo, 2),
            "trend": self.trend,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "common_contexts": self.common_contexts,
            "description": self.description,
            "root_cause": self.root_cause,
            "improvement_suggestions": self.improvement_suggestions,
            "instances_count": len(self.instances),
            "recent_instances": [i.to_dict() for i in self.instances[-5:]],
        }


# ============================================================================
# Pattern Detection Engine
# ============================================================================


class PatternDetector:
    """
    Detects specific mistake patterns from demo analysis data.
    """

    def __init__(self):
        self.detection_functions = {
            MistakeType.WIDE_PEEK: self._detect_wide_peek,
            MistakeType.SAME_ANGLE_HOLD: self._detect_same_angle,
            MistakeType.DRY_PEEK: self._detect_dry_peek,
            MistakeType.PEEKING_ALONE: self._detect_solo_peek,
            MistakeType.LATE_TRADE: self._detect_late_trade,
            MistakeType.TEAM_FLASH: self._detect_team_flash,
            MistakeType.WASTED_FLASH: self._detect_wasted_flash,
            MistakeType.POOR_SPRAY_CONTROL: self._detect_poor_spray,
            MistakeType.WRONG_CROSSHAIR_HEIGHT: self._detect_crosshair_error,
            MistakeType.UNNECESSARY_AGGRESSION: self._detect_overaggression,
            MistakeType.TOO_PASSIVE: self._detect_passive_play,
            MistakeType.NO_UTILITY_BUY: self._detect_no_utility,
            MistakeType.FORCE_BUY_BROKE_TEAM: self._detect_bad_force,
        }

    def detect_mistakes(
        self, demo_data: dict[str, Any], player_stats: dict[str, Any], steamid: str, demo_id: str
    ) -> list[MistakeInstance]:
        """
        Detect all mistake instances from a single demo.

        Args:
            demo_data: Raw demo data including events
            player_stats: Analyzed player statistics
            steamid: Target player's Steam ID
            demo_id: Unique identifier for this demo

        Returns:
            List of detected mistake instances
        """
        mistakes = []
        map_name = demo_data.get("map_name", "unknown")

        for _mistake_type, detector_func in self.detection_functions.items():
            try:
                detected = detector_func(demo_data, player_stats, steamid, demo_id, map_name)
                mistakes.extend(detected)
            except Exception:
                # Don't crash on detection errors
                pass

        return mistakes

    def _detect_wide_peek(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect wide peeks that resulted in death."""
        mistakes = []

        # Analyze deaths where player was moving fast
        deaths = demo_data.get("deaths", [])
        for death in deaths:
            if death.get("victim_steamid") != steamid:
                continue

            # Check velocity at death
            velocity = death.get("victim_velocity", 0)
            if velocity > 200:  # Running peek threshold
                tick = death.get("tick", 0)
                round_num = death.get("round_num", 0)
                position = (
                    death.get("victim_x", 0),
                    death.get("victim_y", 0),
                    death.get("victim_z", 0),
                )

                mistakes.append(
                    MistakeInstance(
                        mistake_type=MistakeType.WIDE_PEEK,
                        demo_id=demo_id,
                        round_num=round_num,
                        tick=tick,
                        timestamp_ms=tick * 15.625,
                        map_name=map_name,
                        position=position,
                        context={
                            "velocity": velocity,
                            "weapon": death.get("weapon", "unknown"),
                            "attacker": death.get("attacker_name", "unknown"),
                        },
                        severity=PatternSeverity.MEDIUM,
                    )
                )

        return mistakes

    def _detect_same_angle(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect holding the same angle repeatedly and dying."""
        mistakes = []

        deaths = demo_data.get("deaths", [])
        death_positions = []

        for death in deaths:
            if death.get("victim_steamid") != steamid:
                continue

            position = (
                death.get("victim_x", 0),
                death.get("victim_y", 0),
                death.get("victim_z", 0),
            )
            death_positions.append(
                {
                    "position": position,
                    "tick": death.get("tick", 0),
                    "round_num": death.get("round_num", 0),
                }
            )

        # Check for deaths at similar positions
        for i, death1 in enumerate(death_positions):
            similar_count = 0
            for death2 in death_positions[i + 1 :]:
                dist = self._position_distance(death1["position"], death2["position"])
                if dist < 200:  # Within 200 units
                    similar_count += 1

            if similar_count >= 2:  # Died at same spot 3+ times
                mistakes.append(
                    MistakeInstance(
                        mistake_type=MistakeType.SAME_ANGLE_HOLD,
                        demo_id=demo_id,
                        round_num=death1["round_num"],
                        tick=death1["tick"],
                        timestamp_ms=death1["tick"] * 15.625,
                        map_name=map_name,
                        position=death1["position"],
                        context={
                            "deaths_at_position": similar_count + 1,
                            "note": "Repeating same defensive position",
                        },
                        severity=PatternSeverity.HIGH,
                    )
                )
                break  # Only report once per position cluster

        return mistakes

    def _detect_dry_peek(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect peeking without utility support."""
        mistakes = []

        deaths = demo_data.get("deaths", [])
        grenades = demo_data.get("grenades", [])

        # Group grenades by round
        grenades_by_round = defaultdict(list)
        for g in grenades:
            if g.get("player_steamid") == steamid:
                grenades_by_round[g.get("round_num", 0)].append(g)

        # Check deaths without preceding utility
        for death in deaths:
            if death.get("victim_steamid") != steamid:
                continue

            round_num = death.get("round_num", 0)
            death_tick = death.get("tick", 0)

            # Check if any utility was used in 5 seconds before death
            had_utility = False
            for g in grenades_by_round.get(round_num, []):
                g_tick = g.get("tick", 0)
                if death_tick - 320 < g_tick < death_tick:  # 5 seconds = ~320 ticks
                    had_utility = True
                    break

            if not had_utility:
                position = (
                    death.get("victim_x", 0),
                    death.get("victim_y", 0),
                    death.get("victim_z", 0),
                )
                mistakes.append(
                    MistakeInstance(
                        mistake_type=MistakeType.DRY_PEEK,
                        demo_id=demo_id,
                        round_num=round_num,
                        tick=death_tick,
                        timestamp_ms=death_tick * 15.625,
                        map_name=map_name,
                        position=position,
                        context={
                            "note": "Died without using utility before peek",
                            "weapon": death.get("weapon", "unknown"),
                        },
                        severity=PatternSeverity.MEDIUM,
                    )
                )

        return mistakes

    def _detect_solo_peek(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect peeking alone without teammate support."""
        mistakes = []

        deaths = demo_data.get("deaths", [])
        positions = demo_data.get("player_positions", {})

        for death in deaths:
            if death.get("victim_steamid") != steamid:
                continue

            round_num = death.get("round_num", 0)
            death_tick = death.get("tick", 0)
            victim_pos = (
                death.get("victim_x", 0),
                death.get("victim_y", 0),
                death.get("victim_z", 0),
            )

            # Check teammate positions at time of death
            teammates_nearby = 0
            round_positions = positions.get(round_num, {})
            for player_id, pos_data in round_positions.items():
                if player_id == steamid:
                    continue
                # Same team check would go here
                pos = pos_data.get(death_tick, {})
                if pos:
                    teammate_pos = (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
                    # 750 units = ~3 seconds run distance at 250 units/s
                    # Aligns with 5-second trade window (TRADE_WINDOW_SECONDS)
                    if self._position_distance(victim_pos, teammate_pos) < 750:
                        teammates_nearby += 1

            if teammates_nearby == 0:
                mistakes.append(
                    MistakeInstance(
                        mistake_type=MistakeType.PEEKING_ALONE,
                        demo_id=demo_id,
                        round_num=round_num,
                        tick=death_tick,
                        timestamp_ms=death_tick * 15.625,
                        map_name=map_name,
                        position=victim_pos,
                        context={"note": "Died while isolated from team", "teammates_nearby": 0},
                        severity=PatternSeverity.MEDIUM,
                    )
                )

        return mistakes

    def _detect_late_trade(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect late or missed trade opportunities."""
        mistakes = []

        kills = demo_data.get("kills", [])
        deaths = demo_data.get("deaths", [])

        # Group by round
        deaths_by_round = defaultdict(list)
        for d in deaths:
            deaths_by_round[d.get("round_num", 0)].append(d)

        kills_by_round = defaultdict(list)
        for k in kills:
            if k.get("attacker_steamid") == steamid:
                kills_by_round[k.get("round_num", 0)].append(k)

        # Check if player traded teammate deaths
        for round_num, round_deaths in deaths_by_round.items():
            teammate_deaths = [d for d in round_deaths if d.get("victim_steamid") != steamid]

            for td in teammate_deaths:
                death_tick = td.get("tick", 0)
                killer_steamid = td.get("attacker_steamid")

                # Check if player killed that enemy within 5 seconds
                traded = False
                for k in kills_by_round.get(round_num, []):
                    if k.get("victim_steamid") == killer_steamid:
                        kill_tick = k.get("tick", 0)
                        if death_tick < kill_tick < death_tick + 320:
                            traded = True
                            (kill_tick - death_tick) * 15.625
                            break

                if not traded:
                    mistakes.append(
                        MistakeInstance(
                            mistake_type=MistakeType.LATE_TRADE,
                            demo_id=demo_id,
                            round_num=round_num,
                            tick=death_tick,
                            timestamp_ms=death_tick * 15.625,
                            map_name=map_name,
                            context={
                                "teammate_killed": td.get("victim_name", "teammate"),
                                "enemy": td.get("attacker_name", "enemy"),
                                "note": "Failed to trade teammate death",
                            },
                            severity=PatternSeverity.HIGH,
                        )
                    )

        return mistakes

    def _detect_team_flash(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect flashing teammates."""
        mistakes = []

        blinds = demo_data.get("blinds", [])

        for blind in blinds:
            if blind.get("attacker_steamid") != steamid:
                continue

            if blind.get("is_teammate", False):
                tick = blind.get("tick", 0)
                mistakes.append(
                    MistakeInstance(
                        mistake_type=MistakeType.TEAM_FLASH,
                        demo_id=demo_id,
                        round_num=blind.get("round_num", 0),
                        tick=tick,
                        timestamp_ms=tick * 15.625,
                        map_name=map_name,
                        context={
                            "victim": blind.get("victim_name", "teammate"),
                            "blind_duration": blind.get("blind_duration", 0),
                        },
                        severity=PatternSeverity.MEDIUM,
                    )
                )

        return mistakes

    def _detect_wasted_flash(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect flashes that didn't blind any enemies."""
        mistakes = []

        grenades = demo_data.get("grenades", [])
        blinds = demo_data.get("blinds", [])

        # Get all flash throws
        flashes = [
            g
            for g in grenades
            if g.get("grenade_type") == "flashbang" and g.get("player_steamid") == steamid
        ]

        # Get blind events caused by player
        flash_blinds = defaultdict(list)
        for b in blinds:
            if b.get("attacker_steamid") == steamid:
                flash_blinds[b.get("round_num", 0)].append(b)

        for flash in flashes:
            round_num = flash.get("round_num", 0)
            tick = flash.get("tick", 0)

            # Check if any enemy was blinded
            enemies_blinded = 0
            for b in flash_blinds.get(round_num, []):
                if not b.get("is_teammate", True) and abs(b.get("tick", 0) - tick) < 64:
                    enemies_blinded += 1

            if enemies_blinded == 0:
                position = (flash.get("x", 0), flash.get("y", 0), flash.get("z", 0))
                mistakes.append(
                    MistakeInstance(
                        mistake_type=MistakeType.WASTED_FLASH,
                        demo_id=demo_id,
                        round_num=round_num,
                        tick=tick,
                        timestamp_ms=tick * 15.625,
                        map_name=map_name,
                        position=position,
                        context={"note": "Flashbang blinded no enemies"},
                        severity=PatternSeverity.LOW,
                    )
                )

        return mistakes

    def _detect_poor_spray(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect poor spray control from damage patterns."""
        mistakes = []

        damages = demo_data.get("damages", [])

        # Group damages by engagement (close tick proximity)
        engagements = []
        current_engagement = []

        sorted_damages = sorted(
            [d for d in damages if d.get("attacker_steamid") == steamid],
            key=lambda x: x.get("tick", 0),
        )

        for dmg in sorted_damages:
            if not current_engagement:
                current_engagement.append(dmg)
            elif dmg.get("tick", 0) - current_engagement[-1].get("tick", 0) < 64:
                current_engagement.append(dmg)
            else:
                if len(current_engagement) > 3:
                    engagements.append(current_engagement)
                current_engagement = [dmg]

        if len(current_engagement) > 3:
            engagements.append(current_engagement)

        # Check each engagement for spray issues
        for engagement in engagements:
            total_damage = sum(d.get("damage", 0) for d in engagement)
            shots = len(engagement)
            avg_damage = total_damage / shots if shots > 0 else 0

            # Poor spray if many shots but low average damage
            if shots >= 5 and avg_damage < 15:
                tick = engagement[0].get("tick", 0)
                mistakes.append(
                    MistakeInstance(
                        mistake_type=MistakeType.POOR_SPRAY_CONTROL,
                        demo_id=demo_id,
                        round_num=engagement[0].get("round_num", 0),
                        tick=tick,
                        timestamp_ms=tick * 15.625,
                        map_name=map_name,
                        context={
                            "shots_fired": shots,
                            "total_damage": total_damage,
                            "avg_damage_per_shot": round(avg_damage, 1),
                            "target": engagement[0].get("victim_name", "enemy"),
                        },
                        severity=PatternSeverity.MEDIUM,
                    )
                )

        return mistakes

    def _detect_crosshair_error(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect consistently poor crosshair placement."""
        mistakes = []

        cp_error = player_stats.get("cp_median_error", 0)
        if cp_error > 20:  # More than 20 degrees average error
            mistakes.append(
                MistakeInstance(
                    mistake_type=MistakeType.WRONG_CROSSHAIR_HEIGHT,
                    demo_id=demo_id,
                    round_num=0,
                    tick=0,
                    timestamp_ms=0,
                    map_name=map_name,
                    context={
                        "average_error_deg": cp_error,
                        "note": "Crosshair placement needs significant improvement",
                    },
                    severity=PatternSeverity.HIGH,
                )
            )

        return mistakes

    def _detect_overaggression(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect unnecessarily aggressive plays."""
        mistakes = []

        deaths = player_stats.get("deaths", 0)
        rounds_played = player_stats.get("rounds_played", 1)
        kast = player_stats.get("kast_percentage", 100)
        survival_rate = player_stats.get("survival_rate", 100)

        if deaths / rounds_played > 0.8 and kast < 60 and survival_rate < 30:
            mistakes.append(
                MistakeInstance(
                    mistake_type=MistakeType.UNNECESSARY_AGGRESSION,
                    demo_id=demo_id,
                    round_num=0,
                    tick=0,
                    timestamp_ms=0,
                    map_name=map_name,
                    context={
                        "deaths_per_round": round(deaths / rounds_played, 2),
                        "kast": kast,
                        "survival_rate": survival_rate,
                        "note": "Dying too often without impact",
                    },
                    severity=PatternSeverity.HIGH,
                )
            )

        return mistakes

    def _detect_passive_play(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect overly passive play style."""
        mistakes = []

        kills = player_stats.get("kills", 0)
        rounds_played = player_stats.get("rounds_played", 1)
        adr = player_stats.get("adr", 0)
        opening_attempts = player_stats.get("opening_duel_attempts", 0)

        kpr = kills / rounds_played if rounds_played > 0 else 0

        if kpr < 0.4 and adr < 50 and opening_attempts < 2:
            mistakes.append(
                MistakeInstance(
                    mistake_type=MistakeType.TOO_PASSIVE,
                    demo_id=demo_id,
                    round_num=0,
                    tick=0,
                    timestamp_ms=0,
                    map_name=map_name,
                    context={
                        "kpr": round(kpr, 2),
                        "adr": adr,
                        "opening_attempts": opening_attempts,
                        "note": "Not taking enough fights or impact plays",
                    },
                    severity=PatternSeverity.MEDIUM,
                )
            )

        return mistakes

    def _detect_no_utility(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect rounds where no utility was purchased/used."""
        mistakes = []

        utility_stats = player_stats.get("utility_stats", {})
        rounds_played = player_stats.get("rounds_played", 1)

        total_utility = (
            utility_stats.get("flashbangs_thrown", 0)
            + utility_stats.get("smokes_thrown", 0)
            + utility_stats.get("he_thrown", 0)
            + utility_stats.get("molotovs_thrown", 0)
        )

        utility_per_round = total_utility / rounds_played if rounds_played > 0 else 0

        if utility_per_round < 0.5:
            mistakes.append(
                MistakeInstance(
                    mistake_type=MistakeType.NO_UTILITY_BUY,
                    demo_id=demo_id,
                    round_num=0,
                    tick=0,
                    timestamp_ms=0,
                    map_name=map_name,
                    context={
                        "utility_per_round": round(utility_per_round, 2),
                        "total_utility": total_utility,
                        "note": "Severely underusing utility",
                    },
                    severity=PatternSeverity.HIGH,
                )
            )

        return mistakes

    def _detect_bad_force(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        demo_id: str,
        map_name: str,
    ) -> list[MistakeInstance]:
        """Detect force buys that hurt team economy."""
        # This would require economy tracking data
        return []

    def _position_distance(
        self, pos1: tuple[float, float, float], pos2: tuple[float, float, float]
    ) -> float:
        """Calculate 3D distance between positions."""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2
        )


# ============================================================================
# Pattern Aggregation and Analysis
# ============================================================================


class PatternAggregator:
    """
    Aggregates mistake instances across multiple demos to identify patterns.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path.home() / ".opensight" / "patterns"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.detector = PatternDetector()

    def add_demo_analysis(
        self, steamid: str, demo_id: str, demo_data: dict[str, Any], player_stats: dict[str, Any]
    ) -> list[MistakeInstance]:
        """
        Analyze a demo and add detected mistakes to the pattern database.

        Args:
            steamid: Player's Steam ID
            demo_id: Unique demo identifier
            demo_data: Raw demo data
            player_stats: Analyzed statistics

        Returns:
            List of detected mistakes from this demo
        """
        # Detect mistakes
        mistakes = self.detector.detect_mistakes(demo_data, player_stats, steamid, demo_id)

        # Load existing pattern data
        pattern_file = self.data_dir / f"patterns_{steamid}.json"
        pattern_data = self._load_pattern_data(pattern_file)

        # Add new mistakes
        for mistake in mistakes:
            mistake_type = mistake.mistake_type.value
            if mistake_type not in pattern_data["mistakes"]:
                pattern_data["mistakes"][mistake_type] = []
            pattern_data["mistakes"][mistake_type].append(mistake.to_dict())

        # Update demo list
        if demo_id not in pattern_data["demos_analyzed"]:
            pattern_data["demos_analyzed"].append(demo_id)
            pattern_data["analysis_dates"].append(datetime.now().isoformat())

        # Save updated data
        self._save_pattern_data(pattern_file, pattern_data)

        return mistakes

    def get_recurring_patterns(
        self, steamid: str, min_occurrences: int = 3
    ) -> list[RecurringPattern]:
        """
        Get identified recurring patterns for a player.

        Args:
            steamid: Player's Steam ID
            min_occurrences: Minimum occurrences to consider a pattern

        Returns:
            List of recurring patterns sorted by severity
        """
        pattern_file = self.data_dir / f"patterns_{steamid}.json"
        pattern_data = self._load_pattern_data(pattern_file)

        recurring = []

        for mistake_type_str, instances_data in pattern_data["mistakes"].items():
            if len(instances_data) < min_occurrences:
                continue

            instances = [MistakeInstance.from_dict(i) for i in instances_data]
            mistake_type = MistakeType(mistake_type_str)

            # Calculate statistics
            maps = list({i.map_name for i in instances})
            demos = list({i.demo_id for i in instances})
            positions = [i.position for i in instances if i.position]

            # Find common positions
            common_positions = self._find_common_positions(positions)

            # Determine trend
            trend = self._calculate_trend(instances, pattern_data["analysis_dates"])

            # Get category and description
            category = self._get_category(mistake_type)
            description, root_cause, suggestions = self._get_coaching_info(mistake_type)

            pattern = RecurringPattern(
                pattern_id=hashlib.md5(
                    f"{steamid}_{mistake_type_str}".encode(), usedforsecurity=False
                ).hexdigest()[:12],
                mistake_type=mistake_type,
                category=category,
                frequency=len(instances),
                demos_count=len(demos),
                maps_affected=maps,
                positions_common=common_positions[:5],  # Top 5 positions
                avg_per_demo=len(instances) / max(1, len(demos)),
                trend=trend,
                first_seen=instances[0].demo_id if instances else "",
                last_seen=instances[-1].demo_id if instances else "",
                common_contexts=self._get_common_contexts(instances),
                instances=instances,
                description=description,
                root_cause=root_cause,
                improvement_suggestions=suggestions,
            )

            recurring.append(pattern)

        # Sort by frequency and severity
        recurring.sort(
            key=lambda p: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}.get(
                    self._get_pattern_severity(p).value, 0
                ),
                p.frequency,
            ),
            reverse=True,
        )

        return recurring

    def get_pattern_summary(self, steamid: str) -> dict[str, Any]:
        """
        Get a summary of all patterns for a player.

        Args:
            steamid: Player's Steam ID

        Returns:
            Summary dictionary with key statistics
        """
        patterns = self.get_recurring_patterns(steamid, min_occurrences=2)
        pattern_file = self.data_dir / f"patterns_{steamid}.json"
        pattern_data = self._load_pattern_data(pattern_file)

        # Count by category
        by_category = defaultdict(int)
        for p in patterns:
            by_category[p.category.value] += p.frequency

        # Count by severity
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for p in patterns:
            sev = self._get_pattern_severity(p).value
            by_severity[sev] += 1

        # Most problematic areas
        top_issues = patterns[:5]

        # Improvement tracking
        improving = [p for p in patterns if p.trend == "improving"]
        worsening = [p for p in patterns if p.trend == "worsening"]

        return {
            "steamid": steamid,
            "demos_analyzed": len(pattern_data["demos_analyzed"]),
            "total_patterns": len(patterns),
            "total_mistakes": sum(p.frequency for p in patterns),
            "by_category": dict(by_category),
            "by_severity": by_severity,
            "top_issues": [
                {
                    "type": p.mistake_type.value,
                    "frequency": p.frequency,
                    "trend": p.trend,
                    "description": p.description,
                }
                for p in top_issues
            ],
            "improving_areas": [p.mistake_type.value for p in improving],
            "worsening_areas": [p.mistake_type.value for p in worsening],
            "recommendations": self._generate_recommendations(patterns),
        }

    def clear_history(self, steamid: str) -> bool:
        """Clear pattern history for a player."""
        pattern_file = self.data_dir / f"patterns_{steamid}.json"
        try:
            if pattern_file.exists():
                pattern_file.unlink()
            return True
        except OSError:
            return False

    def _load_pattern_data(self, filepath: Path) -> dict[str, Any]:
        """Load pattern data from file."""
        if filepath.exists():
            try:
                with open(filepath) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        return {"mistakes": {}, "demos_analyzed": [], "analysis_dates": []}

    def _save_pattern_data(self, filepath: Path, data: dict[str, Any]) -> None:
        """Save pattern data to file."""
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _find_common_positions(
        self, positions: list[tuple[float, float, float]], threshold: float = 200
    ) -> list[tuple[float, float, float]]:
        """Find clusters of common positions."""
        if not positions:
            return []

        clusters = []
        used = set()

        for i, pos1 in enumerate(positions):
            if i in used:
                continue

            cluster = [pos1]
            used.add(i)

            for j, pos2 in enumerate(positions):
                if j in used:
                    continue

                dist = math.sqrt(
                    (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2
                )
                if dist < threshold:
                    cluster.append(pos2)
                    used.add(j)

            if len(cluster) >= 2:
                # Average position of cluster
                avg_pos = (
                    sum(p[0] for p in cluster) / len(cluster),
                    sum(p[1] for p in cluster) / len(cluster),
                    sum(p[2] for p in cluster) / len(cluster),
                )
                clusters.append(avg_pos)

        return sorted(
            clusters,
            key=lambda x: len([p for p in positions if self._dist(p, x) < threshold]),
            reverse=True,
        )

    def _dist(self, p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
        """Calculate distance between positions."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _calculate_trend(self, instances: list[MistakeInstance], dates: list[str]) -> str:
        """Calculate if the pattern is improving, worsening, or stable."""
        if len(dates) < 3:
            return "stable"

        # Split into halves by time
        mid = len(instances) // 2
        if mid == 0:
            return "stable"

        early_count = mid
        late_count = len(instances) - mid

        # Normalize by demos analyzed
        if len(dates) >= 2:
            early_demos = len(dates) // 2
            late_demos = len(dates) - early_demos

            early_rate = early_count / max(1, early_demos)
            late_rate = late_count / max(1, late_demos)

            if late_rate < early_rate * 0.7:
                return "improving"
            elif late_rate > early_rate * 1.3:
                return "worsening"

        return "stable"

    def _get_category(self, mistake_type: MistakeType) -> PatternCategory:
        """Map mistake type to category."""
        mapping = {
            MistakeType.WIDE_PEEK: PatternCategory.POSITIONING,
            MistakeType.EXPOSED_POSITION: PatternCategory.POSITIONING,
            MistakeType.SAME_ANGLE_HOLD: PatternCategory.POSITIONING,
            MistakeType.OVER_ROTATION: PatternCategory.POSITIONING,
            MistakeType.UNDER_ROTATION: PatternCategory.POSITIONING,
            MistakeType.DRY_PEEK: PatternCategory.TIMING,
            MistakeType.PEEKING_ALONE: PatternCategory.TIMING,
            MistakeType.LATE_TRADE: PatternCategory.TIMING,
            MistakeType.LATE_ROTATION: PatternCategory.TIMING,
            MistakeType.POOR_SPRAY_CONTROL: PatternCategory.AIM,
            MistakeType.OVERAIMING: PatternCategory.AIM,
            MistakeType.UNDERAIMING: PatternCategory.AIM,
            MistakeType.WRONG_CROSSHAIR_HEIGHT: PatternCategory.AIM,
            MistakeType.WASTED_FLASH: PatternCategory.UTILITY,
            MistakeType.TEAM_FLASH: PatternCategory.UTILITY,
            MistakeType.EARLY_SMOKE: PatternCategory.UTILITY,
            MistakeType.LATE_SMOKE: PatternCategory.UTILITY,
            MistakeType.NO_UTILITY_BUY: PatternCategory.UTILITY,
            MistakeType.FORCE_BUY_BROKE_TEAM: PatternCategory.ECONOMY,
            MistakeType.NO_SAVE: PatternCategory.ECONOMY,
            MistakeType.OVERBUYING: PatternCategory.ECONOMY,
            MistakeType.UNNECESSARY_AGGRESSION: PatternCategory.DECISION_MAKING,
            MistakeType.TOO_PASSIVE: PatternCategory.DECISION_MAKING,
            MistakeType.WRONG_BOMBSITE: PatternCategory.DECISION_MAKING,
        }
        return mapping.get(mistake_type, PatternCategory.MECHANICAL)

    def _get_coaching_info(self, mistake_type: MistakeType) -> tuple[str, str, list[str]]:
        """Get coaching information for a mistake type."""
        info = {
            MistakeType.WIDE_PEEK: (
                "You consistently swing wide when peeking corners",
                "Likely over-extending or not using cover effectively",
                ["Practice shoulder peeking", "Use counter-strafe", "Pre-aim common angles"],
            ),
            MistakeType.SAME_ANGLE_HOLD: (
                "You're holding the same position repeatedly and dying",
                "Predictable positioning makes you easy to prefire",
                ["Change positions after getting kills", "Use off-angles", "Rotate defensively"],
            ),
            MistakeType.DRY_PEEK: (
                "You often peek without using utility first",
                "Not creating advantages before taking fights",
                [
                    "Flash or smoke before peeking",
                    "Have teammate flash for you",
                    "Buy utility every round",
                ],
            ),
            MistakeType.PEEKING_ALONE: (
                "You take fights without teammate support nearby",
                "Isolating yourself makes trades impossible",
                ["Stay closer to teammates", "Wait for support before peeking", "Use buddy system"],
            ),
            MistakeType.LATE_TRADE: (
                "You're not trading teammates quickly enough",
                "Hesitation or poor positioning for trades",
                ["Stay closer to entry player", "Be ready to swing on death", "Communicate trades"],
            ),
            MistakeType.TEAM_FLASH: (
                "You're flashing teammates too often",
                "Poor flash timing or trajectory",
                ["Communicate flash timing", "Use popflashes", "Practice flash lineups"],
            ),
            MistakeType.WASTED_FLASH: (
                "Many of your flashes don't blind enemies",
                "Incorrect timing or positioning of flashbangs",
                ["Learn popflash techniques", "Flash for teammates", "Time flashes with pushes"],
            ),
            MistakeType.POOR_SPRAY_CONTROL: (
                "Your spray control needs improvement",
                "Not compensating for recoil patterns",
                ["Practice spray patterns", "Burst fire at range", "Use workshop maps"],
            ),
            MistakeType.WRONG_CROSSHAIR_HEIGHT: (
                "Your crosshair placement is consistently off",
                "Not keeping crosshair at head level",
                ["Practice head-level pre-aims", "Use prefire maps", "Watch pro demos"],
            ),
            MistakeType.UNNECESSARY_AGGRESSION: (
                "You're dying too often without impact",
                "Taking unnecessary fights or bad positioning",
                ["Play for trades", "Value your life more", "Focus on survival in KAST"],
            ),
            MistakeType.TOO_PASSIVE: (
                "You're not taking enough impactful fights",
                "Playing too safely or baiting teammates",
                ["Look for opening opportunities", "Support entry players", "Be first to trade"],
            ),
            MistakeType.NO_UTILITY_BUY: (
                "You're severely underusing grenades",
                "Not buying or saving utility",
                [
                    "Buy smokes/flashes every full buy",
                    "Learn basic lineups",
                    "Use utility before fights",
                ],
            ),
        }

        default = (
            f"Recurring issue: {mistake_type.value}",
            "This mistake is occurring frequently",
            ["Review demo footage", "Practice specific scenarios", "Focus on fundamentals"],
        )

        return info.get(mistake_type, default)

    def _get_common_contexts(self, instances: list[MistakeInstance]) -> list[dict[str, Any]]:
        """Extract common contextual patterns from instances."""
        contexts = []

        # Aggregate context data
        weapons = defaultdict(int)
        notes = defaultdict(int)

        for inst in instances:
            if "weapon" in inst.context:
                weapons[inst.context["weapon"]] += 1
            if "note" in inst.context:
                notes[inst.context["note"]] += 1

        if weapons:
            top_weapons = sorted(weapons.items(), key=lambda x: x[1], reverse=True)[:3]
            contexts.append({"type": "common_weapons", "data": dict(top_weapons)})

        if notes:
            top_notes = sorted(notes.items(), key=lambda x: x[1], reverse=True)[:3]
            contexts.append({"type": "common_situations", "data": dict(top_notes)})

        return contexts

    def _get_pattern_severity(self, pattern: RecurringPattern) -> PatternSeverity:
        """Determine overall severity of a pattern."""
        # High frequency = higher severity
        freq_score = min(pattern.frequency / 10, 1.0)

        # Worsening trend = higher severity
        trend_score = {"worsening": 0.3, "stable": 0.0, "improving": -0.2}.get(pattern.trend, 0)

        # Some mistake types are more severe
        type_scores = {
            MistakeType.TEAM_FLASH: 0.2,
            MistakeType.LATE_TRADE: 0.3,
            MistakeType.DRY_PEEK: 0.2,
            MistakeType.UNNECESSARY_AGGRESSION: 0.3,
            MistakeType.WRONG_CROSSHAIR_HEIGHT: 0.3,
        }
        type_score = type_scores.get(pattern.mistake_type, 0.1)

        total = freq_score + trend_score + type_score

        if total > 0.8:
            return PatternSeverity.CRITICAL
        elif total > 0.5:
            return PatternSeverity.HIGH
        elif total > 0.3:
            return PatternSeverity.MEDIUM
        else:
            return PatternSeverity.LOW

    def _generate_recommendations(self, patterns: list[RecurringPattern]) -> list[str]:
        """Generate prioritized recommendations based on patterns."""
        recommendations = []

        # Count issues by category
        category_counts = defaultdict(int)
        for p in patterns:
            category_counts[p.category] += p.frequency

        # Most problematic category
        if category_counts:
            worst_category = max(category_counts.items(), key=lambda x: x[1])
            cat_name = worst_category[0].value.replace("_", " ").title()
            recommendations.append(f"Primary focus area: {cat_name}")

        # Specific recommendations for top patterns
        for p in patterns[:3]:
            if p.improvement_suggestions:
                recommendations.append(p.improvement_suggestions[0])

        # Trend-based recommendations
        worsening = [p for p in patterns if p.trend == "worsening"]
        if worsening:
            recommendations.append(
                f"Urgent: {worsening[0].mistake_type.value.replace('_', ' ')} is getting worse"
            )

        return recommendations[:5]


# ============================================================================
# Convenience Functions
# ============================================================================

_default_aggregator: PatternAggregator | None = None


def get_aggregator() -> PatternAggregator:
    """Get or create the default pattern aggregator."""
    global _default_aggregator
    if _default_aggregator is None:
        _default_aggregator = PatternAggregator()
    return _default_aggregator


def analyze_demo_patterns(
    steamid: str, demo_id: str, demo_data: dict[str, Any], player_stats: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Analyze a demo for patterns and return detected mistakes.

    Args:
        steamid: Player's Steam ID
        demo_id: Unique demo identifier
        demo_data: Raw demo data
        player_stats: Analyzed statistics

    Returns:
        List of detected mistakes as dictionaries
    """
    aggregator = get_aggregator()
    mistakes = aggregator.add_demo_analysis(steamid, demo_id, demo_data, player_stats)
    return [m.to_dict() for m in mistakes]


def get_player_patterns(steamid: str, min_occurrences: int = 3) -> list[dict[str, Any]]:
    """
    Get recurring patterns for a player.

    Args:
        steamid: Player's Steam ID
        min_occurrences: Minimum occurrences to consider

    Returns:
        List of recurring patterns as dictionaries
    """
    aggregator = get_aggregator()
    patterns = aggregator.get_recurring_patterns(steamid, min_occurrences)
    return [p.to_dict() for p in patterns]


def get_pattern_report(steamid: str) -> dict[str, Any]:
    """
    Get a comprehensive pattern report for a player.

    Args:
        steamid: Player's Steam ID

    Returns:
        Pattern summary report
    """
    return get_aggregator().get_pattern_summary(steamid)
