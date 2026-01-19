"""
Adaptive AI Coaching Module for CS2 Demo Analyzer.

Provides personalized coaching insights based on player rank, role, and map pool.
Uses reinforcement learning to prioritize the most impactful mistakes for each user.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import json
import math
import hashlib
from pathlib import Path


# ============================================================================
# Player Profile Enums and Types
# ============================================================================

class PlayerRank(Enum):
    """CS2 competitive ranks."""
    UNRANKED = 0
    SILVER_1 = 1
    SILVER_2 = 2
    SILVER_3 = 3
    SILVER_4 = 4
    SILVER_ELITE = 5
    SILVER_ELITE_MASTER = 6
    GOLD_NOVA_1 = 7
    GOLD_NOVA_2 = 8
    GOLD_NOVA_3 = 9
    GOLD_NOVA_MASTER = 10
    MASTER_GUARDIAN_1 = 11
    MASTER_GUARDIAN_2 = 12
    MASTER_GUARDIAN_ELITE = 13
    DISTINGUISHED_MASTER_GUARDIAN = 14
    LEGENDARY_EAGLE = 15
    LEGENDARY_EAGLE_MASTER = 16
    SUPREME_MASTER_FIRST_CLASS = 17
    GLOBAL_ELITE = 18
    # FACEIT/ESEA ranks
    FACEIT_1 = 101
    FACEIT_2 = 102
    FACEIT_3 = 103
    FACEIT_4 = 104
    FACEIT_5 = 105
    FACEIT_6 = 106
    FACEIT_7 = 107
    FACEIT_8 = 108
    FACEIT_9 = 109
    FACEIT_10 = 110


class PlayerRole(Enum):
    """Standard CS2 player roles."""
    ENTRY_FRAGGER = "entry"
    AWP = "awp"
    SUPPORT = "support"
    LURKER = "lurker"
    IGL = "igl"
    RIFLER = "rifler"
    FLEX = "flex"
    UNKNOWN = "unknown"


class SkillArea(Enum):
    """Skill areas for coaching focus."""
    AIM_MECHANICS = "aim_mechanics"
    CROSSHAIR_PLACEMENT = "crosshair_placement"
    MOVEMENT = "movement"
    UTILITY_USAGE = "utility_usage"
    POSITIONING = "positioning"
    TRADING = "trading"
    CLUTCHING = "clutching"
    ECONOMY = "economy"
    COMMUNICATION = "communication"
    GAME_SENSE = "game_sense"
    ENTRY_FRAGGING = "entry_fragging"
    AWP_USAGE = "awp_usage"
    MAP_KNOWLEDGE = "map_knowledge"
    RETAKES = "retakes"
    POST_PLANT = "post_plant"


# ============================================================================
# Player Profile Data Structures
# ============================================================================

@dataclass
class PlayerProfile:
    """Complete player profile for coaching personalization."""
    steamid: str
    name: str = "Unknown"
    rank: PlayerRank = PlayerRank.UNRANKED
    primary_role: PlayerRole = PlayerRole.UNKNOWN
    secondary_role: Optional[PlayerRole] = None
    map_pool: list[str] = field(default_factory=list)
    hours_played: float = 0.0
    demos_analyzed: int = 0

    # Skill ratings (0-100 scale)
    skill_ratings: dict[str, float] = field(default_factory=dict)

    # Historical performance averages
    avg_hltv_rating: float = 0.0
    avg_adr: float = 0.0
    avg_kast: float = 0.0
    avg_ttd_ms: float = 0.0
    avg_cp_error: float = 0.0

    # Coaching state
    focus_areas: list[SkillArea] = field(default_factory=list)
    coaching_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "steamid": self.steamid,
            "name": self.name,
            "rank": self.rank.name,
            "primary_role": self.primary_role.value,
            "secondary_role": self.secondary_role.value if self.secondary_role else None,
            "map_pool": self.map_pool,
            "hours_played": self.hours_played,
            "demos_analyzed": self.demos_analyzed,
            "skill_ratings": self.skill_ratings,
            "avg_hltv_rating": self.avg_hltv_rating,
            "avg_adr": self.avg_adr,
            "avg_kast": self.avg_kast,
            "avg_ttd_ms": self.avg_ttd_ms,
            "avg_cp_error": self.avg_cp_error,
            "focus_areas": [a.value for a in self.focus_areas],
            "coaching_history": self.coaching_history[-50:]  # Keep last 50 entries
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerProfile":
        """Create profile from dictionary."""
        profile = cls(
            steamid=data.get("steamid", ""),
            name=data.get("name", "Unknown"),
            hours_played=data.get("hours_played", 0.0),
            demos_analyzed=data.get("demos_analyzed", 0),
            skill_ratings=data.get("skill_ratings", {}),
            avg_hltv_rating=data.get("avg_hltv_rating", 0.0),
            avg_adr=data.get("avg_adr", 0.0),
            avg_kast=data.get("avg_kast", 0.0),
            avg_ttd_ms=data.get("avg_ttd_ms", 0.0),
            avg_cp_error=data.get("avg_cp_error", 0.0),
            map_pool=data.get("map_pool", []),
            coaching_history=data.get("coaching_history", [])
        )

        # Parse rank
        try:
            profile.rank = PlayerRank[data.get("rank", "UNRANKED")]
        except KeyError:
            profile.rank = PlayerRank.UNRANKED

        # Parse roles
        try:
            profile.primary_role = PlayerRole(data.get("primary_role", "unknown"))
        except ValueError:
            profile.primary_role = PlayerRole.UNKNOWN

        if data.get("secondary_role"):
            try:
                profile.secondary_role = PlayerRole(data["secondary_role"])
            except ValueError:
                pass

        # Parse focus areas
        for area in data.get("focus_areas", []):
            try:
                profile.focus_areas.append(SkillArea(area))
            except ValueError:
                pass

        return profile


@dataclass
class CoachingInsight:
    """A single coaching insight/recommendation."""
    skill_area: SkillArea
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    specific_examples: list[str] = field(default_factory=list)
    improvement_tips: list[str] = field(default_factory=list)
    practice_drills: list[str] = field(default_factory=list)
    priority_score: float = 0.0  # RL-computed priority
    confidence: float = 1.0  # How confident the system is in this insight

    # Tracking
    times_shown: int = 0
    times_acknowledged: int = 0
    improvement_observed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "skill_area": self.skill_area.value,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "specific_examples": self.specific_examples,
            "improvement_tips": self.improvement_tips,
            "practice_drills": self.practice_drills,
            "priority_score": round(self.priority_score, 3),
            "confidence": round(self.confidence, 3),
            "times_shown": self.times_shown,
            "times_acknowledged": self.times_acknowledged,
            "improvement_observed": self.improvement_observed
        }


# ============================================================================
# Reinforcement Learning Priority System
# ============================================================================

@dataclass
class RLState:
    """State representation for RL priority learning."""
    # Player context
    rank_tier: int  # 0=low, 1=mid, 2=high, 3=pro
    role_id: int
    map_familiarity: float  # 0-1

    # Recent performance
    performance_trend: float  # -1 to 1
    consistency: float  # 0-1

    # Insight context
    skill_area_id: int
    severity_id: int
    times_shown: int

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        return [
            self.rank_tier / 3.0,
            self.role_id / 7.0,
            self.map_familiarity,
            (self.performance_trend + 1) / 2.0,
            self.consistency,
            self.skill_area_id / 15.0,
            self.severity_id / 3.0,
            min(self.times_shown / 10.0, 1.0)
        ]


class AdaptivePriorityLearner:
    """
    Reinforcement learning system for prioritizing coaching insights.
    Uses a simple Q-learning approach with function approximation.
    """

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Weight vectors for each action (show/hide insight)
        # State vector has 8 features
        self.num_features = 8
        self.weights_show = [0.0] * self.num_features
        self.weights_hide = [0.0] * self.num_features

        # Reward history for tracking
        self.total_reward = 0.0
        self.num_updates = 0

        # Initialize with prior knowledge
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights with domain knowledge."""
        # Higher weight for severity
        self.weights_show[5] = 0.5  # severity_id importance
        self.weights_show[6] = 0.3  # skill_area importance

        # Penalize showing same insight repeatedly
        self.weights_show[7] = -0.2  # times_shown penalty

        # Consider rank when prioritizing
        self.weights_show[0] = 0.1  # rank affects what to show

    def get_q_value(self, state: RLState, action: str) -> float:
        """Compute Q-value for state-action pair."""
        features = state.to_vector()
        weights = self.weights_show if action == "show" else self.weights_hide
        return sum(f * w for f, w in zip(features, weights))

    def compute_priority(self, state: RLState) -> float:
        """Compute priority score for showing an insight."""
        q_show = self.get_q_value(state, "show")
        q_hide = self.get_q_value(state, "hide")

        # Softmax to get probability
        max_q = max(q_show, q_hide)
        exp_show = math.exp(q_show - max_q)
        exp_hide = math.exp(q_hide - max_q)

        priority = exp_show / (exp_show + exp_hide)
        return priority

    def update(self, state: RLState, action: str, reward: float,
               next_state: Optional[RLState] = None) -> None:
        """Update weights based on observed reward."""
        features = state.to_vector()

        # Compute current Q-value
        current_q = self.get_q_value(state, action)

        # Compute target
        if next_state is not None:
            next_q_show = self.get_q_value(next_state, "show")
            next_q_hide = self.get_q_value(next_state, "hide")
            target = reward + self.discount_factor * max(next_q_show, next_q_hide)
        else:
            target = reward

        # TD error
        td_error = target - current_q

        # Update weights
        weights = self.weights_show if action == "show" else self.weights_hide
        for i in range(self.num_features):
            weights[i] += self.learning_rate * td_error * features[i]

        self.total_reward += reward
        self.num_updates += 1

    def get_reward(self, insight: CoachingInsight, acknowledged: bool,
                   improvement_observed: bool) -> float:
        """Compute reward for showing an insight."""
        reward = 0.0

        # Base reward for acknowledgment
        if acknowledged:
            reward += 0.3

        # Big reward for actual improvement
        if improvement_observed:
            reward += 1.0

        # Penalty for repeated unacknowledged insights
        if insight.times_shown > 3 and not acknowledged:
            reward -= 0.2

        # Severity bonus - more important insights get higher reward
        severity_bonus = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.0}
        reward += severity_bonus.get(insight.severity, 0.0)

        return reward

    def to_dict(self) -> dict[str, Any]:
        """Serialize learner state."""
        return {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "weights_show": self.weights_show,
            "weights_hide": self.weights_hide,
            "total_reward": self.total_reward,
            "num_updates": self.num_updates
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptivePriorityLearner":
        """Deserialize learner state."""
        learner = cls(
            learning_rate=data.get("learning_rate", 0.1),
            discount_factor=data.get("discount_factor", 0.95),
            exploration_rate=data.get("exploration_rate", 0.1)
        )
        learner.weights_show = data.get("weights_show", learner.weights_show)
        learner.weights_hide = data.get("weights_hide", learner.weights_hide)
        learner.total_reward = data.get("total_reward", 0.0)
        learner.num_updates = data.get("num_updates", 0)
        return learner


# ============================================================================
# Role-Specific Benchmarks and Thresholds
# ============================================================================

# Benchmarks by rank tier (index: 0=silver, 1=gold, 2=mg, 3=le+, 4=faceit7+)
RANK_BENCHMARKS = {
    "ttd_ms": [400, 350, 300, 250, 200],
    "cp_error_deg": [25, 20, 15, 10, 5],
    "adr": [50, 60, 70, 80, 90],
    "kast": [50, 60, 65, 70, 75],
    "hltv_rating": [0.7, 0.85, 0.95, 1.05, 1.15],
    "hs_percent": [30, 35, 40, 45, 50],
    "utility_per_round": [0.5, 1.0, 1.5, 2.0, 2.5],
    "opening_duel_wr": [35, 40, 45, 50, 55],
    "trade_rate": [20, 30, 40, 50, 60]
}

# Role-specific focus areas
ROLE_FOCUS_AREAS = {
    PlayerRole.ENTRY_FRAGGER: [
        SkillArea.AIM_MECHANICS,
        SkillArea.CROSSHAIR_PLACEMENT,
        SkillArea.ENTRY_FRAGGING,
        SkillArea.MOVEMENT
    ],
    PlayerRole.AWP: [
        SkillArea.AWP_USAGE,
        SkillArea.POSITIONING,
        SkillArea.GAME_SENSE,
        SkillArea.ECONOMY
    ],
    PlayerRole.SUPPORT: [
        SkillArea.UTILITY_USAGE,
        SkillArea.TRADING,
        SkillArea.COMMUNICATION,
        SkillArea.POSITIONING
    ],
    PlayerRole.LURKER: [
        SkillArea.GAME_SENSE,
        SkillArea.POSITIONING,
        SkillArea.CLUTCHING,
        SkillArea.MAP_KNOWLEDGE
    ],
    PlayerRole.IGL: [
        SkillArea.GAME_SENSE,
        SkillArea.COMMUNICATION,
        SkillArea.ECONOMY,
        SkillArea.MAP_KNOWLEDGE
    ],
    PlayerRole.RIFLER: [
        SkillArea.AIM_MECHANICS,
        SkillArea.CROSSHAIR_PLACEMENT,
        SkillArea.TRADING,
        SkillArea.UTILITY_USAGE
    ],
    PlayerRole.FLEX: [
        SkillArea.AIM_MECHANICS,
        SkillArea.UTILITY_USAGE,
        SkillArea.GAME_SENSE,
        SkillArea.TRADING
    ],
    PlayerRole.UNKNOWN: [
        SkillArea.AIM_MECHANICS,
        SkillArea.CROSSHAIR_PLACEMENT,
        SkillArea.UTILITY_USAGE,
        SkillArea.POSITIONING
    ]
}

# Map-specific tips
MAP_TIPS = {
    "de_dust2": {
        SkillArea.POSITIONING: ["Control long doors as CT", "Hold catwalk angles"],
        SkillArea.UTILITY_USAGE: ["Practice A site smokes", "Learn B tunnels mollies"],
        SkillArea.MAP_KNOWLEDGE: ["Understand mid control importance", "Learn boost spots"]
    },
    "de_mirage": {
        SkillArea.POSITIONING: ["Window room control is crucial", "Connector presence"],
        SkillArea.UTILITY_USAGE: ["A site execute smokes", "Jungle/connector mollies"],
        SkillArea.MAP_KNOWLEDGE: ["Underpass timings", "Palace control"]
    },
    "de_inferno": {
        SkillArea.POSITIONING: ["Banana control defines rounds", "Apps presence"],
        SkillArea.UTILITY_USAGE: ["Banana utility essential", "A site smokes"],
        SkillArea.MAP_KNOWLEDGE: ["Timing-based plays", "Second mid control"]
    },
    "de_nuke": {
        SkillArea.POSITIONING: ["Vertical play understanding", "Outside control"],
        SkillArea.UTILITY_USAGE: ["Heaven smokes", "Secret area utility"],
        SkillArea.MAP_KNOWLEDGE: ["Vent usage", "Ramp control importance"]
    },
    "de_ancient": {
        SkillArea.POSITIONING: ["Mid control variations", "Elbow holds"],
        SkillArea.UTILITY_USAGE: ["A site executes", "Mid smokes"],
        SkillArea.MAP_KNOWLEDGE: ["Cave timings", "Donut area usage"]
    },
    "de_anubis": {
        SkillArea.POSITIONING: ["Canal control", "Connector angles"],
        SkillArea.UTILITY_USAGE: ["B site executes", "Mid control smokes"],
        SkillArea.MAP_KNOWLEDGE: ["Palace timings", "Water area usage"]
    },
    "de_vertigo": {
        SkillArea.POSITIONING: ["Ramp control", "Mid presence"],
        SkillArea.UTILITY_USAGE: ["A ramp smokes", "B site utility"],
        SkillArea.MAP_KNOWLEDGE: ["Elevator usage", "Scaffold plays"]
    }
}


def get_rank_tier(rank: PlayerRank) -> int:
    """Convert rank to tier index (0-4)."""
    if rank.value <= 6:  # Silver
        return 0
    elif rank.value <= 10:  # Gold Nova
        return 1
    elif rank.value <= 13:  # MG
        return 2
    elif rank.value <= 18:  # LE+
        return 3
    elif rank.value >= 107:  # FACEIT 7+
        return 4
    else:
        return 2  # Default to MG tier


# ============================================================================
# Adaptive Coaching Engine
# ============================================================================

class AdaptiveCoach:
    """
    Main coaching engine that generates personalized insights.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".opensight" / "coaching"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Player profiles cache
        self.profiles: dict[str, PlayerProfile] = {}

        # RL learner
        self.learner = AdaptivePriorityLearner()
        self._load_learner()

        # Insight templates
        self.insight_templates = self._build_insight_templates()

    def _load_learner(self) -> None:
        """Load RL learner state from disk."""
        learner_path = self.data_dir / "learner_state.json"
        if learner_path.exists():
            try:
                with open(learner_path, "r") as f:
                    data = json.load(f)
                self.learner = AdaptivePriorityLearner.from_dict(data)
            except (json.JSONDecodeError, IOError):
                pass

    def _save_learner(self) -> None:
        """Save RL learner state to disk."""
        learner_path = self.data_dir / "learner_state.json"
        try:
            with open(learner_path, "w") as f:
                json.dump(self.learner.to_dict(), f, indent=2)
        except IOError:
            pass

    def get_profile(self, steamid: str) -> PlayerProfile:
        """Get or create player profile."""
        if steamid not in self.profiles:
            profile_path = self.data_dir / f"profile_{steamid}.json"
            if profile_path.exists():
                try:
                    with open(profile_path, "r") as f:
                        data = json.load(f)
                    self.profiles[steamid] = PlayerProfile.from_dict(data)
                except (json.JSONDecodeError, IOError):
                    self.profiles[steamid] = PlayerProfile(steamid=steamid)
            else:
                self.profiles[steamid] = PlayerProfile(steamid=steamid)
        return self.profiles[steamid]

    def save_profile(self, profile: PlayerProfile) -> None:
        """Save player profile to disk."""
        self.profiles[profile.steamid] = profile
        profile_path = self.data_dir / f"profile_{profile.steamid}.json"
        try:
            with open(profile_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)
        except IOError:
            pass

    def update_profile(self, steamid: str, name: str = "",
                       rank: Optional[PlayerRank] = None,
                       role: Optional[PlayerRole] = None,
                       map_pool: Optional[list[str]] = None) -> PlayerProfile:
        """Update player profile settings."""
        profile = self.get_profile(steamid)

        if name:
            profile.name = name
        if rank is not None:
            profile.rank = rank
        if role is not None:
            profile.primary_role = role
        if map_pool is not None:
            profile.map_pool = map_pool

        self.save_profile(profile)
        return profile

    def _build_insight_templates(self) -> dict[SkillArea, list[dict[str, Any]]]:
        """Build insight templates for each skill area."""
        templates = {
            SkillArea.AIM_MECHANICS: [
                {
                    "title": "Slow Time to Damage",
                    "condition": lambda stats, bench: stats.get("ttd_median_ms", 999) > bench["ttd_ms"],
                    "severity": "high",
                    "description": "Your reaction time from first damage to kill is slower than expected for your rank.",
                    "tips": [
                        "Practice aim trainers like Aim Lab or Kovaak's",
                        "Focus on pre-aiming common angles",
                        "Work on crosshair placement to reduce adjustment time"
                    ],
                    "drills": [
                        "Yprac aim training maps",
                        "Deathmatch with specific weapon focus",
                        "1v1 arena maps"
                    ]
                },
                {
                    "title": "Low Headshot Percentage",
                    "condition": lambda stats, bench: stats.get("hs_percent", 0) < bench["hs_percent"],
                    "severity": "medium",
                    "description": "Your headshot percentage is below average. Focus on head-level aim.",
                    "tips": [
                        "Keep crosshair at head level",
                        "Practice head-level pre-aims for common positions",
                        "Slow down and aim for heads in deathmatch"
                    ],
                    "drills": [
                        "Headshot-only deathmatch servers",
                        "Aim_botz with head-level targets",
                        "Prefire practice maps"
                    ]
                }
            ],
            SkillArea.CROSSHAIR_PLACEMENT: [
                {
                    "title": "High Crosshair Angle Error",
                    "condition": lambda stats, bench: stats.get("cp_median_error", 999) > bench["cp_error_deg"],
                    "severity": "high",
                    "description": "Your crosshair is often far from where enemies appear. This costs valuable milliseconds.",
                    "tips": [
                        "Pre-aim common angles when moving",
                        "Keep crosshair at head level always",
                        "Clear angles systematically instead of randomly"
                    ],
                    "drills": [
                        "Prefire practice workshop maps",
                        "Slow-paced deathmatch focusing on placement",
                        "VOD review of professional players"
                    ]
                }
            ],
            SkillArea.UTILITY_USAGE: [
                {
                    "title": "Low Utility Usage",
                    "condition": lambda stats, bench: stats.get("utility_per_round", 0) < bench["utility_per_round"],
                    "severity": "medium",
                    "description": "You're not using enough utility each round. Grenades can win rounds.",
                    "tips": [
                        "Buy utility every round you can",
                        "Learn 2-3 smokes per map for executes",
                        "Use flashes for entry and support"
                    ],
                    "drills": [
                        "Smoke/flash/molly practice maps",
                        "Execute practice in offline server",
                        "Watch pro utility usage"
                    ]
                },
                {
                    "title": "Flashing Teammates",
                    "condition": lambda stats, bench: stats.get("teammates_flashed", 0) > 3,
                    "severity": "high",
                    "description": "You're flashing your teammates too often. This loses rounds.",
                    "tips": [
                        "Communicate flash timing",
                        "Use pop flashes that don't blind teammates",
                        "Wait for teammates to be ready"
                    ],
                    "drills": [
                        "Practice flash lineups in offline server",
                        "Learn popflash techniques",
                        "Coordinate with team on utility"
                    ]
                }
            ],
            SkillArea.TRADING: [
                {
                    "title": "Low Trade Rate",
                    "condition": lambda stats, bench: stats.get("trade_rate", 0) < bench["trade_rate"],
                    "severity": "high",
                    "description": "You're not trading teammates quickly enough when they die.",
                    "tips": [
                        "Stay closer to teammates",
                        "Be ready to swing immediately after teammate dies",
                        "Use utility to help with trades"
                    ],
                    "drills": [
                        "Practice buddy system in retakes",
                        "Work on peek timing drills",
                        "Review demos for trade opportunities missed"
                    ]
                }
            ],
            SkillArea.ENTRY_FRAGGING: [
                {
                    "title": "Low Opening Duel Win Rate",
                    "condition": lambda stats, bench: stats.get("opening_duel_wr", 0) < bench["opening_duel_wr"],
                    "severity": "medium",
                    "description": "You're losing too many opening duels as an entry fragger.",
                    "tips": [
                        "Use utility before entering",
                        "Work on prefire timing",
                        "Consider changing entry spots"
                    ],
                    "drills": [
                        "Prefire maps for common spots",
                        "Flash and entry practice",
                        "1v1 arena for raw dueling"
                    ]
                }
            ],
            SkillArea.POSITIONING: [
                {
                    "title": "Overaggressive Positioning",
                    "condition": lambda stats, bench: stats.get("deaths_per_round", 1) > 0.8 and stats.get("kast", 0) < 60,
                    "severity": "medium",
                    "description": "You're dying too often without impact. Consider safer positions.",
                    "tips": [
                        "Hold angles instead of pushing",
                        "Stay alive for retakes",
                        "Position where you can be traded"
                    ],
                    "drills": [
                        "Study CT positioning in pro matches",
                        "Practice holding specific angles",
                        "Work on when to fall back"
                    ]
                }
            ],
            SkillArea.ECONOMY: [
                {
                    "title": "Poor Buy Decisions",
                    "condition": lambda stats, bench: stats.get("force_buy_rate", 0) > 40,
                    "severity": "low",
                    "description": "You're force buying too often. This hurts team economy.",
                    "tips": [
                        "Follow team's buy calls",
                        "Save for full buys when appropriate",
                        "Understand eco round strategy"
                    ],
                    "drills": [
                        "Practice eco round strategies",
                        "Learn force buy situations",
                        "Review economy management"
                    ]
                }
            ],
            SkillArea.CLUTCHING: [
                {
                    "title": "Low Clutch Win Rate",
                    "condition": lambda stats, bench: stats.get("clutch_win_rate", 0) < 30 and stats.get("clutch_attempts", 0) > 5,
                    "severity": "low",
                    "description": "Your clutch conversion rate is low. Work on 1vX situations.",
                    "tips": [
                        "Take your time in clutches",
                        "Use sound to track enemies",
                        "Isolate 1v1 duels"
                    ],
                    "drills": [
                        "Retake servers",
                        "1v1/1v2 clutch scenarios",
                        "Decision-making practice"
                    ]
                }
            ],
            SkillArea.GAME_SENSE: [
                {
                    "title": "Poor Timing",
                    "condition": lambda stats, bench: stats.get("deaths_while_rotating", 0) > 5,
                    "severity": "medium",
                    "description": "You're dying during rotations too often. Work on timing.",
                    "tips": [
                        "Wait for information before rotating",
                        "Use utility to cover rotations",
                        "Understand timing windows"
                    ],
                    "drills": [
                        "Watch demos for timing patterns",
                        "Practice map timings",
                        "Learn callout timing"
                    ]
                }
            ],
            SkillArea.AWP_USAGE: [
                {
                    "title": "Aggressive AWP Dies Often",
                    "condition": lambda stats, bench: stats.get("awp_deaths", 0) > stats.get("awp_kills", 0) * 0.7,
                    "severity": "high",
                    "description": "You're dying too much with the AWP. It's a $4750 investment.",
                    "tips": [
                        "Hold angles instead of peeking",
                        "Reposition after shots",
                        "Stay alive - AWP is expensive"
                    ],
                    "drills": [
                        "AWP positioning maps",
                        "Peek timing practice",
                        "Movement with AWP"
                    ]
                }
            ]
        }
        return templates

    def generate_insights(self, player_stats: dict[str, Any],
                          steamid: str,
                          map_name: str = "") -> list[CoachingInsight]:
        """
        Generate personalized coaching insights for a player.

        Args:
            player_stats: Statistics from demo analysis
            steamid: Player's Steam ID
            map_name: Optional map name for map-specific tips

        Returns:
            List of prioritized coaching insights
        """
        profile = self.get_profile(steamid)
        rank_tier = get_rank_tier(profile.rank)

        # Get benchmarks for this rank tier
        benchmarks = {k: v[min(rank_tier, len(v) - 1)] for k, v in RANK_BENCHMARKS.items()}

        insights = []
        role_focus = ROLE_FOCUS_AREAS.get(profile.primary_role,
                                          ROLE_FOCUS_AREAS[PlayerRole.UNKNOWN])

        # Check each skill area
        for skill_area, templates in self.insight_templates.items():
            for template in templates:
                # Check if condition is met
                if template["condition"](player_stats, benchmarks):
                    insight = CoachingInsight(
                        skill_area=skill_area,
                        severity=template["severity"],
                        title=template["title"],
                        description=template["description"],
                        improvement_tips=template["tips"],
                        practice_drills=template["drills"]
                    )

                    # Add map-specific tips if available
                    if map_name and map_name in MAP_TIPS:
                        map_tips = MAP_TIPS[map_name].get(skill_area, [])
                        insight.specific_examples.extend(map_tips)

                    # Compute RL priority
                    state = RLState(
                        rank_tier=rank_tier,
                        role_id=list(PlayerRole).index(profile.primary_role),
                        map_familiarity=1.0 if map_name in profile.map_pool else 0.5,
                        performance_trend=self._compute_trend(profile),
                        consistency=self._compute_consistency(profile),
                        skill_area_id=list(SkillArea).index(skill_area),
                        severity_id=["low", "medium", "high", "critical"].index(insight.severity),
                        times_shown=self._get_times_shown(profile, skill_area)
                    )

                    insight.priority_score = self.learner.compute_priority(state)

                    # Boost priority for role-relevant skills
                    if skill_area in role_focus:
                        insight.priority_score *= 1.2

                    insights.append(insight)

        # Sort by priority
        insights.sort(key=lambda x: x.priority_score, reverse=True)

        # Record in profile history
        profile.coaching_history.append({
            "map": map_name,
            "insights_generated": len(insights),
            "top_issues": [i.title for i in insights[:3]]
        })
        self.save_profile(profile)

        return insights

    def _compute_trend(self, profile: PlayerProfile) -> float:
        """Compute recent performance trend (-1 to 1)."""
        if profile.demos_analyzed < 3:
            return 0.0

        # Check coaching history for improvement signals
        recent = profile.coaching_history[-5:] if profile.coaching_history else []
        if not recent:
            return 0.0

        # Trend based on number of issues decreasing
        if len(recent) >= 2:
            early_issues = sum(h.get("insights_generated", 0) for h in recent[:len(recent)//2])
            late_issues = sum(h.get("insights_generated", 0) for h in recent[len(recent)//2:])

            if early_issues > 0:
                trend = (early_issues - late_issues) / early_issues
                return max(-1.0, min(1.0, trend))

        return 0.0

    def _compute_consistency(self, profile: PlayerProfile) -> float:
        """Compute consistency score (0 to 1)."""
        if profile.demos_analyzed < 2:
            return 0.5

        # Base consistency on skill rating variance
        if not profile.skill_ratings:
            return 0.5

        ratings = list(profile.skill_ratings.values())
        if len(ratings) < 2:
            return 0.5

        avg = sum(ratings) / len(ratings)
        variance = sum((r - avg) ** 2 for r in ratings) / len(ratings)
        std_dev = math.sqrt(variance)

        # Normalize - higher std_dev = lower consistency
        consistency = max(0, 1 - std_dev / 50)
        return consistency

    def _get_times_shown(self, profile: PlayerProfile, skill_area: SkillArea) -> int:
        """Get how many times insights for this skill area have been shown."""
        count = 0
        for entry in profile.coaching_history:
            if skill_area.value in str(entry.get("top_issues", [])):
                count += 1
        return count

    def record_feedback(self, steamid: str, insight_title: str,
                        acknowledged: bool, improvement_observed: bool = False) -> None:
        """
        Record user feedback on an insight for RL learning.

        Args:
            steamid: Player's Steam ID
            insight_title: Title of the insight
            acknowledged: Whether user acknowledged the insight
            improvement_observed: Whether improvement was observed in next demo
        """
        profile = self.get_profile(steamid)
        rank_tier = get_rank_tier(profile.rank)

        # Find the insight in templates to get skill area
        skill_area = SkillArea.AIM_MECHANICS  # default
        severity = "medium"

        for area, templates in self.insight_templates.items():
            for template in templates:
                if template["title"] == insight_title:
                    skill_area = area
                    severity = template["severity"]
                    break

        # Create state for RL update
        state = RLState(
            rank_tier=rank_tier,
            role_id=list(PlayerRole).index(profile.primary_role),
            map_familiarity=0.5,
            performance_trend=self._compute_trend(profile),
            consistency=self._compute_consistency(profile),
            skill_area_id=list(SkillArea).index(skill_area),
            severity_id=["low", "medium", "high", "critical"].index(severity),
            times_shown=self._get_times_shown(profile, skill_area)
        )

        # Compute reward
        insight = CoachingInsight(
            skill_area=skill_area,
            severity=severity,
            title=insight_title,
            description="",
            times_shown=self._get_times_shown(profile, skill_area)
        )
        reward = self.learner.get_reward(insight, acknowledged, improvement_observed)

        # Update RL learner
        self.learner.update(state, "show", reward)
        self._save_learner()

    def get_practice_plan(self, steamid: str,
                          duration_minutes: int = 30) -> dict[str, Any]:
        """
        Generate a personalized practice plan based on identified weaknesses.

        Args:
            steamid: Player's Steam ID
            duration_minutes: Target practice duration

        Returns:
            Structured practice plan
        """
        profile = self.get_profile(steamid)

        # Get recent insights
        recent_issues = []
        for entry in profile.coaching_history[-10:]:
            recent_issues.extend(entry.get("top_issues", []))

        # Count frequency of issues
        issue_freq = {}
        for issue in recent_issues:
            issue_freq[issue] = issue_freq.get(issue, 0) + 1

        # Build practice plan
        plan = {
            "player": profile.name,
            "rank": profile.rank.name,
            "role": profile.primary_role.value,
            "duration_minutes": duration_minutes,
            "focus_areas": [],
            "warmup": [],
            "main_practice": [],
            "cooldown": []
        }

        # Determine focus areas from most frequent issues
        sorted_issues = sorted(issue_freq.items(), key=lambda x: x[1], reverse=True)

        # Map issues to practice activities
        time_remaining = duration_minutes

        # Warmup (10% of time)
        warmup_time = max(5, int(duration_minutes * 0.1))
        plan["warmup"] = [
            {"activity": "Aim trainer warmup", "duration": warmup_time // 2},
            {"activity": "Deathmatch warmup", "duration": warmup_time // 2}
        ]
        time_remaining -= warmup_time

        # Cooldown (10% of time)
        cooldown_time = max(5, int(duration_minutes * 0.1))
        plan["cooldown"] = [
            {"activity": "Review one recent demo round", "duration": cooldown_time}
        ]
        time_remaining -= cooldown_time

        # Main practice based on issues
        for issue, _ in sorted_issues[:3]:
            if time_remaining <= 0:
                break

            activity_time = time_remaining // 3

            if "TTD" in issue or "Aim" in issue:
                plan["main_practice"].append({
                    "activity": "Aim training (tracking and flicking)",
                    "duration": activity_time,
                    "details": "Focus on headshot level tracking"
                })
                plan["focus_areas"].append("aim_mechanics")
            elif "Crosshair" in issue:
                plan["main_practice"].append({
                    "activity": "Prefire practice maps",
                    "duration": activity_time,
                    "details": "Work on pre-aiming common angles"
                })
                plan["focus_areas"].append("crosshair_placement")
            elif "Utility" in issue or "Flash" in issue:
                plan["main_practice"].append({
                    "activity": "Utility lineups practice",
                    "duration": activity_time,
                    "details": f"Learn 3 new lineups for {profile.map_pool[0] if profile.map_pool else 'dust2'}"
                })
                plan["focus_areas"].append("utility_usage")
            elif "Trade" in issue:
                plan["main_practice"].append({
                    "activity": "Retake servers",
                    "duration": activity_time,
                    "details": "Practice trading and buddy system"
                })
                plan["focus_areas"].append("trading")
            else:
                plan["main_practice"].append({
                    "activity": "Deathmatch focused practice",
                    "duration": activity_time,
                    "details": "General mechanics improvement"
                })

            time_remaining -= activity_time

        # Fill remaining time with general practice if needed
        if time_remaining > 5:
            plan["main_practice"].append({
                "activity": "Competitive match or retakes",
                "duration": time_remaining,
                "details": "Apply learned skills in live environment"
            })

        return plan

    def compare_to_role_benchmarks(self, player_stats: dict[str, Any],
                                    role: PlayerRole,
                                    rank: PlayerRank) -> dict[str, Any]:
        """
        Compare player stats to role-specific benchmarks.

        Args:
            player_stats: Player statistics from analysis
            role: Player's role
            rank: Player's rank

        Returns:
            Comparison results with percentile rankings
        """
        rank_tier = get_rank_tier(rank)

        comparisons = {}

        # Define role-specific expected values (multipliers on base benchmarks)
        role_multipliers = {
            PlayerRole.ENTRY_FRAGGER: {
                "adr": 1.1, "opening_duel_wr": 1.2, "ttd_ms": 0.9,
                "trade_rate": 0.8, "kast": 0.95
            },
            PlayerRole.AWP: {
                "adr": 0.9, "opening_duel_wr": 1.1, "ttd_ms": 1.1,
                "kast": 1.0, "trade_rate": 0.7
            },
            PlayerRole.SUPPORT: {
                "adr": 0.9, "utility_per_round": 1.5, "trade_rate": 1.3,
                "kast": 1.05, "ttd_ms": 1.0
            },
            PlayerRole.LURKER: {
                "adr": 0.95, "kast": 0.9, "clutch_win_rate": 1.2,
                "trade_rate": 0.6, "ttd_ms": 1.0
            },
            PlayerRole.IGL: {
                "adr": 0.85, "kast": 1.0, "utility_per_round": 1.2,
                "trade_rate": 1.0, "ttd_ms": 1.1
            }
        }

        multipliers = role_multipliers.get(role, {})

        for metric, base_values in RANK_BENCHMARKS.items():
            base_value = base_values[min(rank_tier, len(base_values) - 1)]
            expected = base_value * multipliers.get(metric, 1.0)
            actual = player_stats.get(metric, 0)

            # Determine if higher or lower is better
            higher_is_better = metric not in ["ttd_ms", "cp_error_deg"]

            if higher_is_better:
                performance = (actual / expected * 100) if expected > 0 else 0
            else:
                performance = (expected / actual * 100) if actual > 0 else 0

            comparisons[metric] = {
                "actual": actual,
                "expected_for_role": round(expected, 2),
                "performance_percent": round(performance, 1),
                "status": "above" if performance > 100 else "below" if performance < 100 else "at"
            }

        return {
            "role": role.value,
            "rank": rank.name,
            "comparisons": comparisons,
            "overall_fit": sum(c["performance_percent"] for c in comparisons.values()) / len(comparisons)
        }

    def suggest_role(self, player_stats: dict[str, Any],
                     rank: PlayerRank) -> dict[str, Any]:
        """
        Suggest optimal role based on player statistics.

        Args:
            player_stats: Player statistics
            rank: Player's rank

        Returns:
            Role suggestions with fit scores
        """
        role_scores = {}

        for role in PlayerRole:
            if role == PlayerRole.UNKNOWN:
                continue

            comparison = self.compare_to_role_benchmarks(player_stats, role, rank)
            role_scores[role] = comparison["overall_fit"]

        # Sort by fit
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "recommended_role": sorted_roles[0][0].value,
            "recommended_fit": round(sorted_roles[0][1], 1),
            "secondary_role": sorted_roles[1][0].value if len(sorted_roles) > 1 else None,
            "all_roles": {r.value: round(s, 1) for r, s in sorted_roles}
        }


# ============================================================================
# Coaching Session Management
# ============================================================================

@dataclass
class CoachingSession:
    """Represents an active coaching session."""
    session_id: str
    steamid: str
    started_at: str
    insights_shown: list[str] = field(default_factory=list)
    feedback_received: dict[str, bool] = field(default_factory=dict)
    demos_analyzed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "steamid": self.steamid,
            "started_at": self.started_at,
            "insights_shown": self.insights_shown,
            "feedback_received": self.feedback_received,
            "demos_analyzed": self.demos_analyzed
        }


class CoachingSessionManager:
    """Manages coaching sessions for continuity."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".opensight" / "sessions"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions: dict[str, CoachingSession] = {}

    def create_session(self, steamid: str) -> CoachingSession:
        """Create a new coaching session."""
        from datetime import datetime

        session_id = hashlib.md5(f"{steamid}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        session = CoachingSession(
            session_id=session_id,
            steamid=steamid,
            started_at=datetime.now().isoformat()
        )
        self.active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[CoachingSession]:
        """Get an existing session."""
        return self.active_sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """End a session and return summary."""
        session = self.active_sessions.pop(session_id, None)
        if session:
            # Save session data
            session_path = self.data_dir / f"session_{session_id}.json"
            try:
                with open(session_path, "w") as f:
                    json.dump(session.to_dict(), f, indent=2)
            except IOError:
                pass

            return {
                "session_id": session_id,
                "demos_analyzed": len(session.demos_analyzed),
                "insights_shown": len(session.insights_shown),
                "feedback_rate": len(session.feedback_received) / max(1, len(session.insights_shown))
            }
        return None


# ============================================================================
# Module-level convenience functions
# ============================================================================

_default_coach: Optional[AdaptiveCoach] = None


def get_coach() -> AdaptiveCoach:
    """Get or create the default coach instance."""
    global _default_coach
    if _default_coach is None:
        _default_coach = AdaptiveCoach()
    return _default_coach


def generate_coaching_insights(player_stats: dict[str, Any],
                               steamid: str,
                               map_name: str = "",
                               rank: Optional[PlayerRank] = None,
                               role: Optional[PlayerRole] = None) -> list[dict[str, Any]]:
    """
    Convenience function to generate coaching insights.

    Args:
        player_stats: Statistics dictionary from demo analysis
        steamid: Player's Steam ID
        map_name: Optional map name
        rank: Optional rank (will be fetched from profile if not provided)
        role: Optional role (will be fetched from profile if not provided)

    Returns:
        List of insight dictionaries
    """
    coach = get_coach()

    # Update profile if rank/role provided
    if rank or role:
        coach.update_profile(steamid, rank=rank, role=role)

    insights = coach.generate_insights(player_stats, steamid, map_name)
    return [i.to_dict() for i in insights]


def get_practice_plan(steamid: str, duration_minutes: int = 30) -> dict[str, Any]:
    """Generate a practice plan for the player."""
    return get_coach().get_practice_plan(steamid, duration_minutes)


def suggest_player_role(player_stats: dict[str, Any],
                        rank: PlayerRank = PlayerRank.GOLD_NOVA_MASTER) -> dict[str, Any]:
    """Suggest optimal role for player based on stats."""
    return get_coach().suggest_role(player_stats, rank)
