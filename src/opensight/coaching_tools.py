"""
AI-Powered Coaching Tools Module

Provides personalized coaching based on player performance metrics:
- Identifies specific areas of improvement
- Suggests targeted strategies
- Offers structured practice exercises with workshop map recommendations
- Provides warm-up routines

Integrates with player_performance_metrics for data-driven coaching.
"""

from dataclasses import dataclass, field
from enum import Enum

from opensight.player_performance_metrics import (
    PlayerPerformanceMetrics,
    compute_player_performance_metrics,
)


class ImprovementArea(Enum):
    """Areas where players can improve."""

    AIM_PRECISION = "aim_precision"
    CROSSHAIR_PLACEMENT = "crosshair_placement"
    MOVEMENT = "movement"
    POSITIONING = "positioning"
    UTILITY_USAGE = "utility_usage"
    TRADING = "trading"
    OPENING_DUELS = "opening_duels"
    CLUTCHING = "clutching"
    SURVIVAL = "survival"
    AGGRESSION = "aggression"
    GAME_SENSE = "game_sense"
    SPRAY_CONTROL = "spray_control"
    AWP_SKILL = "awp_skill"
    ECONOMY_MANAGEMENT = "economy_management"


class Priority(Enum):
    """Priority levels for improvement areas."""

    CRITICAL = 1  # Fix immediately
    HIGH = 2  # Important to address
    MEDIUM = 3  # Should improve
    LOW = 4  # Nice to have


@dataclass
class PracticeExercise:
    """A specific practice exercise or drill."""

    name: str
    description: str
    duration_minutes: int
    workshop_maps: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)  # Console commands
    tips: list[str] = field(default_factory=list)
    difficulty: str = "beginner"  # beginner, intermediate, advanced

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "duration_minutes": self.duration_minutes,
            "workshop_maps": self.workshop_maps,
            "commands": self.commands,
            "tips": self.tips,
            "difficulty": self.difficulty,
        }


@dataclass
class ImprovementPlan:
    """A personalized improvement plan for a player."""

    area: ImprovementArea
    priority: Priority
    current_level: str  # Assessment of current skill
    target_level: str  # Goal to reach
    reasoning: str  # Why this was identified
    strategies: list[str] = field(default_factory=list)
    exercises: list[PracticeExercise] = field(default_factory=list)
    metrics_to_track: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "area": self.area.value,
            "priority": self.priority.name.lower(),
            "current_level": self.current_level,
            "target_level": self.target_level,
            "reasoning": self.reasoning,
            "strategies": self.strategies,
            "exercises": [e.to_dict() for e in self.exercises],
            "metrics_to_track": self.metrics_to_track,
        }


@dataclass
class WarmupRoutine:
    """Pre-game warmup routine."""

    name: str
    total_duration_minutes: int
    exercises: list[PracticeExercise] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_duration_minutes": self.total_duration_minutes,
            "exercises": [e.to_dict() for e in self.exercises],
            "focus_areas": self.focus_areas,
        }


@dataclass
class CoachingReport:
    """Complete coaching report for a player."""

    player_name: str
    steam_id: int
    overall_assessment: str
    rank_estimate: str  # Estimated skill tier
    improvement_plans: list[ImprovementPlan] = field(default_factory=list)
    warmup_routine: WarmupRoutine | None = None
    quick_tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "steam_id": str(self.steam_id),
            "overall_assessment": self.overall_assessment,
            "rank_estimate": self.rank_estimate,
            "improvement_plans": [p.to_dict() for p in self.improvement_plans],
            "warmup_routine": self.warmup_routine.to_dict() if self.warmup_routine else None,
            "quick_tips": self.quick_tips,
        }


# =============================================================================
# Practice Exercise Database
# =============================================================================

PRACTICE_EXERCISES: dict[ImprovementArea, list[PracticeExercise]] = {
    ImprovementArea.AIM_PRECISION: [
        PracticeExercise(
            name="Aim_Botz Flicking",
            description="Practice flick shots on static and moving targets",
            duration_minutes=10,
            workshop_maps=["Aim_Botz", "aim_botz"],
            commands=["sv_cheats 1", "mp_warmup_end"],
            tips=[
                "Start slow, focus on accuracy over speed",
                "Gradually increase target spawn speed",
                "Practice both left and right flicks",
            ],
            difficulty="beginner",
        ),
        PracticeExercise(
            name="Reflex Training",
            description="Train reaction time with sudden target appearances",
            duration_minutes=15,
            workshop_maps=["Fast Aim / Reflex Training", "training_aim_csgo2"],
            tips=[
                "Focus on first shot accuracy",
                "Don't spray, take deliberate shots",
                "Track your average reaction time",
            ],
            difficulty="intermediate",
        ),
        PracticeExercise(
            name="Tracking Practice",
            description="Follow moving targets smoothly",
            duration_minutes=10,
            workshop_maps=["Aim_Botz", "Yprac Practice Maps"],
            tips=[
                "Keep crosshair on target, not ahead",
                "Practice with both AK and M4",
                "Work on smooth mouse movements",
            ],
            difficulty="intermediate",
        ),
    ],
    ImprovementArea.CROSSHAIR_PLACEMENT: [
        PracticeExercise(
            name="Prefire Practice",
            description="Learn common angles and practice prefiring them",
            duration_minutes=15,
            workshop_maps=["Yprac Mirage", "Yprac Inferno", "Yprac Dust2"],
            tips=[
                "Memorize exact head height at each angle",
                "Practice the timing, not just placement",
                "Clear angles in sequence",
            ],
            difficulty="beginner",
        ),
        PracticeExercise(
            name="Crosshair Placement Walkthrough",
            description="Walk through maps focusing only on crosshair position",
            duration_minutes=20,
            workshop_maps=["Any competitive map"],
            commands=["sv_cheats 1", "bot_add_ct", "bot_stop 1"],
            tips=[
                "Never look at the ground or sky",
                "Pre-aim every corner before peeking",
                "Keep crosshair at head level at all times",
            ],
            difficulty="beginner",
        ),
    ],
    ImprovementArea.SPRAY_CONTROL: [
        PracticeExercise(
            name="Recoil Master",
            description="Master spray patterns for all rifles",
            duration_minutes=15,
            workshop_maps=["Recoil Master - Spray Training", "UltraLow Recoil"],
            tips=[
                "Start with AK-47, then M4A4/M4A1-S",
                "Practice first 10 bullets most",
                "Pull down and slightly left for AK",
            ],
            difficulty="beginner",
        ),
        PracticeExercise(
            name="Spray Transfer",
            description="Practice transferring spray between targets",
            duration_minutes=10,
            workshop_maps=["Aim_Botz - set to 2+ targets"],
            tips=[
                "Kill first target with 3-5 bullets, then transfer",
                "Practice horizontal transfers first",
                "Add vertical variation once comfortable",
            ],
            difficulty="advanced",
        ),
    ],
    ImprovementArea.MOVEMENT: [
        PracticeExercise(
            name="Counter-Strafe Training",
            description="Practice stopping accurately before shooting",
            duration_minutes=10,
            workshop_maps=["aim_training_csgo2"],
            tips=[
                "Press opposite movement key to stop instantly",
                "Shoot only when fully stopped",
                "Practice A-D-A-D strafing with shots",
            ],
            difficulty="intermediate",
        ),
        PracticeExercise(
            name="Jiggle Peeking",
            description="Learn to peek and gather info without committing",
            duration_minutes=10,
            workshop_maps=["Any map with walls"],
            commands=["sv_cheats 1", "bot_add_t", "bot_stop 1"],
            tips=[
                "Expose minimum of your body",
                "Use to bait AWP shots",
                "Practice both left and right peeks",
            ],
            difficulty="intermediate",
        ),
        PracticeExercise(
            name="Bunny Hop Practice",
            description="Learn basic bunny hopping for faster rotations",
            duration_minutes=10,
            workshop_maps=["Bhop maps", "kz_ maps"],
            commands=["sv_autobunnyhopping 0"],
            tips=[
                "Jump at the exact moment you land",
                "Strafe in air for speed gain",
                "Don't hold W while bhopping",
            ],
            difficulty="advanced",
        ),
    ],
    ImprovementArea.POSITIONING: [
        PracticeExercise(
            name="Off-Angle Practice",
            description="Learn and practice uncommon angles",
            duration_minutes=15,
            workshop_maps=["Yprac maps", "Any competitive map"],
            tips=[
                "Find angles enemies don't pre-aim",
                "Change positions after getting kills",
                "Have an escape route planned",
            ],
            difficulty="intermediate",
        ),
        PracticeExercise(
            name="Trade Position Practice",
            description="Learn positions that allow teammates to trade",
            duration_minutes=15,
            workshop_maps=["Any competitive map"],
            tips=[
                "Stay close enough for refrag but not too close",
                "2-3 second gap between you and entry",
                "Call out when flashing for teammate",
            ],
            difficulty="intermediate",
        ),
    ],
    ImprovementArea.UTILITY_USAGE: [
        PracticeExercise(
            name="Smoke Lineups",
            description="Learn essential smoke lineups for each map",
            duration_minutes=20,
            workshop_maps=["Yprac Practice Maps", "Smoke Training maps"],
            commands=["sv_cheats 1", "sv_infinite_ammo 1", "sv_grenade_trajectory 1"],
            tips=[
                "Start with 5 essential smokes per map",
                "Practice until you can throw them quickly",
                "Learn both 64 and 128 tick versions",
            ],
            difficulty="beginner",
        ),
        PracticeExercise(
            name="Pop Flash Practice",
            description="Learn pop flashes for common positions",
            duration_minutes=15,
            workshop_maps=["Yprac Practice Maps"],
            commands=["sv_cheats 1", "sv_infinite_ammo 1"],
            tips=[
                "Flash should pop as it enters enemy FOV",
                "Practice self-flashes for entries",
                "Time your flash with teammate's peek",
            ],
            difficulty="intermediate",
        ),
        PracticeExercise(
            name="Molotov Lineups",
            description="Learn molotovs to clear common positions",
            duration_minutes=15,
            workshop_maps=["Yprac Practice Maps"],
            tips=[
                "Focus on positions that are hard to smoke",
                "Time molotovs with team executes",
                "Learn to bounce molotovs around corners",
            ],
            difficulty="intermediate",
        ),
    ],
    ImprovementArea.OPENING_DUELS: [
        PracticeExercise(
            name="Entry Fragging Practice",
            description="Practice taking map control as entry fragger",
            duration_minutes=15,
            workshop_maps=["Yprac Prefire Practice Maps"],
            tips=[
                "Clear angles in order of threat",
                "Use utility to isolate 1v1 fights",
                "Call out when you die for trade",
            ],
            difficulty="intermediate",
        ),
        PracticeExercise(
            name="Peek Timing",
            description="Learn when to peek based on sound cues",
            duration_minutes=10,
            workshop_maps=["Any competitive map"],
            tips=[
                "Peek when enemy is reloading",
                "Peek right after your flash pops",
                "Don't dry peek AWPers",
            ],
            difficulty="intermediate",
        ),
    ],
    ImprovementArea.CLUTCHING: [
        PracticeExercise(
            name="Clutch Practice",
            description="Practice 1vX scenarios with bots",
            duration_minutes=20,
            workshop_maps=["Clutch Map Workshop", "Any competitive map"],
            commands=["bot_add_ct 3", "mp_limitteams 0", "mp_autoteambalance 0"],
            tips=[
                "Isolate 1v1 fights",
                "Use sound to track enemy positions",
                "Don't panic, play slow",
                "Use utility to delay enemies",
            ],
            difficulty="advanced",
        ),
    ],
    ImprovementArea.SURVIVAL: [
        PracticeExercise(
            name="Positioning for Survival",
            description="Learn safe positions and when to rotate",
            duration_minutes=15,
            workshop_maps=["Any competitive map - watch pro demos"],
            tips=[
                "Always have an escape route",
                "Don't overextend for kills",
                "Rotate early when site is lost",
                "Play for trades, not hero plays",
            ],
            difficulty="intermediate",
        ),
    ],
    ImprovementArea.AWP_SKILL: [
        PracticeExercise(
            name="AWP Flick Training",
            description="Practice AWP flick shots",
            duration_minutes=15,
            workshop_maps=["Aim_Botz", "AWP Training maps"],
            tips=[
                "Don't hardscope, use quick scopes",
                "Practice both standing and moving targets",
                "Work on quick repositioning after shots",
            ],
            difficulty="intermediate",
        ),
        PracticeExercise(
            name="AWP Positioning",
            description="Learn common AWP angles and rotations",
            duration_minutes=20,
            workshop_maps=["Watch pro AWPer demos"],
            tips=[
                "Never peek the same angle twice",
                "Have a plan for after the first shot",
                "Know when to save vs. aggressive peek",
            ],
            difficulty="advanced",
        ),
    ],
}


# =============================================================================
# Warmup Routines
# =============================================================================

STANDARD_WARMUP = WarmupRoutine(
    name="Standard Pre-Game Warmup",
    total_duration_minutes=20,
    exercises=[
        PracticeExercise(
            name="Deathmatch",
            description="Free-for-all deathmatch to warm up mechanics",
            duration_minutes=10,
            workshop_maps=["Valve FFA DM", "Community DM servers"],
            tips=["Focus on headshots only", "Don't worry about K/D", "Practice counter-strafing"],
            difficulty="beginner",
        ),
        PracticeExercise(
            name="Aim_Botz Spray",
            description="Quick spray control refresher",
            duration_minutes=5,
            workshop_maps=["Aim_Botz"],
            tips=["Do 2-3 full sprays per gun", "Reset muscle memory"],
            difficulty="beginner",
        ),
        PracticeExercise(
            name="Prefire Angles",
            description="Run through prefire angles on your map",
            duration_minutes=5,
            workshop_maps=["Yprac Practice Maps"],
            tips=["Focus on the map you're about to play"],
            difficulty="beginner",
        ),
    ],
    focus_areas=["aim", "spray control", "crosshair placement"],
)

QUICK_WARMUP = WarmupRoutine(
    name="Quick 10-Minute Warmup",
    total_duration_minutes=10,
    exercises=[
        PracticeExercise(
            name="Fast Aim Training",
            description="Quick aim warmup with bots",
            duration_minutes=5,
            workshop_maps=["Aim_Botz"],
            tips=["100 kills, headshots only"],
            difficulty="beginner",
        ),
        PracticeExercise(
            name="DM Headshots",
            description="Quick DM focusing on headshots",
            duration_minutes=5,
            workshop_maps=["HS Only DM servers"],
            tips=["Don't spray, tap or burst only"],
            difficulty="beginner",
        ),
    ],
    focus_areas=["aim", "reaction time"],
)


# =============================================================================
# Coaching Analysis Functions
# =============================================================================


def analyze_and_coach(metrics: PlayerPerformanceMetrics) -> CoachingReport:
    """
    Analyze player performance and generate personalized coaching.

    Args:
        metrics: Player performance metrics from compute_player_performance_metrics

    Returns:
        CoachingReport with personalized improvement plans
    """
    improvement_plans: list[ImprovementPlan] = []
    quick_tips: list[str] = []

    # Analyze aim precision (headshot %)
    hs_pct = metrics.kda.headshot_percentage
    if hs_pct < 25:
        improvement_plans.append(
            ImprovementPlan(
                area=ImprovementArea.AIM_PRECISION,
                priority=Priority.CRITICAL,
                current_level=f"{hs_pct}% headshots",
                target_level="40%+ headshots",
                reasoning="Low headshot percentage indicates aim is not landing on heads consistently",
                strategies=[
                    "Lower your sensitivity if too high",
                    "Focus on crosshair placement at head level",
                    "Practice in aim trainers daily",
                    "Watch your crosshair, not the enemy model",
                ],
                exercises=PRACTICE_EXERCISES[ImprovementArea.AIM_PRECISION],
                metrics_to_track=["headshot_percentage", "average_kill_distance"],
            )
        )
        quick_tips.append("Focus on crosshair placement - keep it at head level always")
    elif hs_pct < 40:
        improvement_plans.append(
            ImprovementPlan(
                area=ImprovementArea.CROSSHAIR_PLACEMENT,
                priority=Priority.MEDIUM,
                current_level=f"{hs_pct}% headshots",
                target_level="50%+ headshots",
                reasoning="Headshot percentage could be improved with better crosshair placement",
                strategies=[
                    "Pre-aim common angles before peeking",
                    "Practice prefire maps to learn exact head positions",
                ],
                exercises=PRACTICE_EXERCISES[ImprovementArea.CROSSHAIR_PLACEMENT],
                metrics_to_track=["headshot_percentage"],
            )
        )

    # Analyze opening duels
    first_kills = metrics.kda.first_kills
    first_deaths = metrics.kda.first_deaths
    opening_total = first_kills + first_deaths
    if opening_total >= 3:
        opening_wr = metrics.kda.opening_duel_win_rate
        if opening_wr < 35:
            improvement_plans.append(
                ImprovementPlan(
                    area=ImprovementArea.OPENING_DUELS,
                    priority=Priority.HIGH,
                    current_level=f"{opening_wr}% win rate ({first_kills}W-{first_deaths}L)",
                    target_level="50%+ win rate",
                    reasoning="Losing opening duels puts your team at a disadvantage",
                    strategies=[
                        "Use utility before dry peeking",
                        "Trade flash with teammate for entries",
                        "Check off-angles more carefully",
                        "Consider playing second instead of entry",
                    ],
                    exercises=PRACTICE_EXERCISES[ImprovementArea.OPENING_DUELS],
                    metrics_to_track=["opening_duel_win_rate", "first_kills", "first_deaths"],
                )
            )
            quick_tips.append(
                "You're losing too many opening duels - use utility or let teammate entry"
            )

    # Analyze survival
    survival = metrics.time_alive.survival_rate
    early_death = metrics.time_alive.early_death_rate
    if survival < 25:
        improvement_plans.append(
            ImprovementPlan(
                area=ImprovementArea.SURVIVAL,
                priority=Priority.HIGH,
                current_level=f"{survival}% survival, {early_death}% early deaths",
                target_level="35%+ survival rate",
                reasoning="Low survival hurts your economy and utility availability",
                strategies=[
                    "Don't overpeek - gather info and retreat",
                    "Play for trades, not hero plays",
                    "Rotate early when outnumbered",
                    "Value your life in eco rounds",
                ],
                exercises=PRACTICE_EXERCISES[ImprovementArea.SURVIVAL],
                metrics_to_track=["survival_rate", "early_death_rate"],
            )
        )
        quick_tips.append("You're dying too often - play safer positions and use utility")

    if early_death > 60:
        quick_tips.append(f"{early_death}% of deaths are early in rounds - slow down your plays")

    # Analyze kill distance patterns
    close_pct = metrics.distance_stats.close_range_percentage
    long_pct = metrics.distance_stats.long_range_percentage
    if close_pct > 60 and metrics.kda.kills >= 5:
        quick_tips.append("Most kills at close range - consider practicing mid/long range fights")
    elif long_pct > 50 and metrics.kda.kills >= 5:
        quick_tips.append("Strong at long range - consider AWPing or holding angles")

    # Analyze K/D
    kd = metrics.kda.kd_ratio
    if kd < 0.7 and metrics.rounds_played >= 10:
        improvement_plans.append(
            ImprovementPlan(
                area=ImprovementArea.GAME_SENSE,
                priority=Priority.HIGH,
                current_level=f"{kd} K/D ratio",
                target_level="1.0+ K/D ratio",
                reasoning="Negative K/D indicates fundamental issues with engagements",
                strategies=[
                    "Pick your fights more carefully",
                    "Avoid 1vX situations without utility",
                    "Play with teammates for trades",
                    "Focus on positioning over raw aim",
                ],
                exercises=PRACTICE_EXERCISES.get(ImprovementArea.POSITIONING, []),
                metrics_to_track=["kd_ratio", "deaths_per_round"],
            )
        )

    # Check for nemesis
    nemesis = metrics.nemesis_stats.nemesis
    if nemesis and nemesis[1] >= 4:
        quick_tips.append(
            f"{nemesis[0]} killed you {nemesis[1]} times - avoid their positions or peek differently"
        )

    # Generate overall assessment
    if kd >= 1.5 and hs_pct >= 45:
        overall = "Strong fragging performance with good aim mechanics"
        rank_estimate = "MGE - Global Elite range"
    elif kd >= 1.0 and hs_pct >= 35:
        overall = "Solid performance with room for improvement in aim"
        rank_estimate = "Gold Nova - MGE range"
    elif kd >= 0.8:
        overall = "Average performance - focus on fundamentals"
        rank_estimate = "Silver Elite - Gold Nova range"
    else:
        overall = "Struggling with fundamentals - prioritize aim and positioning practice"
        rank_estimate = "Silver range"

    # Sort improvement plans by priority
    improvement_plans.sort(key=lambda p: p.priority.value)

    # Select appropriate warmup
    warmup = QUICK_WARMUP if len(improvement_plans) <= 2 else STANDARD_WARMUP

    return CoachingReport(
        player_name=metrics.name,
        steam_id=metrics.steam_id,
        overall_assessment=overall,
        rank_estimate=rank_estimate,
        improvement_plans=improvement_plans[:5],  # Top 5 priorities
        warmup_routine=warmup,
        quick_tips=quick_tips[:5],  # Top 5 quick tips
    )


def generate_coaching_for_match(match_data) -> dict[int, CoachingReport]:
    """
    Generate coaching reports for all players in a match.

    Args:
        match_data: MatchData from DemoParser

    Returns:
        Dictionary mapping steam_id to CoachingReport
    """
    metrics = compute_player_performance_metrics(match_data)
    reports = {}

    for steam_id, player_metrics in metrics.items():
        reports[steam_id] = analyze_and_coach(player_metrics)

    return reports


def get_practice_exercises_for_area(area: ImprovementArea) -> list[PracticeExercise]:
    """Get all practice exercises for a specific improvement area."""
    return PRACTICE_EXERCISES.get(area, [])


def get_all_warmup_routines() -> list[WarmupRoutine]:
    """Get all available warmup routines."""
    return [STANDARD_WARMUP, QUICK_WARMUP]
