"""
Opponent Modeling Module for CS2 Demo Analyzer.

Cross-references opponent tendencies from public databases (HLTV, FACEIT)
to predict strategies and suggest counter-tactics.
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# Data Types and Enums
# ============================================================================


class PlayStyle(Enum):
    """Player playstyle classifications."""

    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    BALANCED = "balanced"
    TACTICAL = "tactical"
    AIM_HEAVY = "aim_heavy"
    UTILITY_HEAVY = "utility_heavy"


class StrategyType(Enum):
    """Team strategy types."""

    FAST_EXECUTE = "fast_execute"
    SLOW_DEFAULT = "slow_default"
    SPLIT = "split"
    FAKE = "fake"
    RUSH = "rush"
    PICK_PLAY = "pick_play"
    RETAKE = "retake"
    STACK = "stack"


class TendencyCategory(Enum):
    """Categories of player tendencies."""

    POSITIONING = "positioning"
    TIMING = "timing"
    WEAPON_CHOICE = "weapon_choice"
    UTILITY_USAGE = "utility_usage"
    AGGRESSION = "aggression"
    ECONOMY = "economy"
    MAP_SPECIFIC = "map_specific"


@dataclass
class PlayerTendency:
    """A single behavioral tendency of a player."""

    category: TendencyCategory
    description: str
    frequency: float  # 0-1 how often this happens
    confidence: float  # 0-1 how confident we are
    map_specific: str | None = None
    side_specific: str | None = None  # "ct", "t", or None for both
    examples: list[str] = field(default_factory=list)
    counter_tactics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "frequency": round(self.frequency, 2),
            "confidence": round(self.confidence, 2),
            "map_specific": self.map_specific,
            "side_specific": self.side_specific,
            "examples": self.examples,
            "counter_tactics": self.counter_tactics,
        }


@dataclass
class OpponentProfile:
    """Complete opponent profile with analyzed tendencies."""

    steamid: str
    name: str
    team: str = ""
    country: str = ""

    # External data sources
    hltv_id: int | None = None
    faceit_id: str | None = None
    esea_id: int | None = None

    # Statistics from external sources
    hltv_rating: float = 0.0
    maps_played: int = 0
    win_rate: float = 0.0

    # Calculated playstyle
    playstyle: PlayStyle = PlayStyle.BALANCED
    playstyle_confidence: float = 0.0

    # Role prediction
    predicted_role: str = "rifler"
    role_confidence: float = 0.0

    # Weapon preferences
    weapon_preferences: dict[str, float] = field(default_factory=dict)

    # Map-specific data
    map_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Identified tendencies
    tendencies: list[PlayerTendency] = field(default_factory=list)

    # Local data from demos
    local_demos_count: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steamid": self.steamid,
            "name": self.name,
            "team": self.team,
            "country": self.country,
            "hltv_id": self.hltv_id,
            "faceit_id": self.faceit_id,
            "esea_id": self.esea_id,
            "hltv_rating": round(self.hltv_rating, 2),
            "maps_played": self.maps_played,
            "win_rate": round(self.win_rate, 2),
            "playstyle": self.playstyle.value,
            "playstyle_confidence": round(self.playstyle_confidence, 2),
            "predicted_role": self.predicted_role,
            "role_confidence": round(self.role_confidence, 2),
            "weapon_preferences": self.weapon_preferences,
            "map_stats": self.map_stats,
            "tendencies": [t.to_dict() for t in self.tendencies],
            "local_demos_count": self.local_demos_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpponentProfile":
        profile = cls(
            steamid=data.get("steamid", ""),
            name=data.get("name", "Unknown"),
            team=data.get("team", ""),
            country=data.get("country", ""),
            hltv_id=data.get("hltv_id"),
            faceit_id=data.get("faceit_id"),
            esea_id=data.get("esea_id"),
            hltv_rating=data.get("hltv_rating", 0.0),
            maps_played=data.get("maps_played", 0),
            win_rate=data.get("win_rate", 0.0),
            predicted_role=data.get("predicted_role", "rifler"),
            role_confidence=data.get("role_confidence", 0.0),
            weapon_preferences=data.get("weapon_preferences", {}),
            map_stats=data.get("map_stats", {}),
            local_demos_count=data.get("local_demos_count", 0),
            last_updated=data.get("last_updated", ""),
        )

        try:
            profile.playstyle = PlayStyle(data.get("playstyle", "balanced"))
        except ValueError:
            profile.playstyle = PlayStyle.BALANCED

        profile.playstyle_confidence = data.get("playstyle_confidence", 0.0)

        for t_data in data.get("tendencies", []):
            try:
                tendency = PlayerTendency(
                    category=TendencyCategory(t_data["category"]),
                    description=t_data["description"],
                    frequency=t_data["frequency"],
                    confidence=t_data["confidence"],
                    map_specific=t_data.get("map_specific"),
                    side_specific=t_data.get("side_specific"),
                    examples=t_data.get("examples", []),
                    counter_tactics=t_data.get("counter_tactics", []),
                )
                profile.tendencies.append(tendency)
            except (KeyError, ValueError):
                continue

        return profile


@dataclass
class TeamProfile:
    """Team-level analysis and tendencies."""

    team_name: str
    players: list[str]  # Steam IDs

    # Team statistics
    avg_rating: float = 0.0
    maps_played: int = 0
    win_rate: float = 0.0

    # Strategy tendencies
    t_side_strategies: dict[str, float] = field(default_factory=dict)  # Strategy -> frequency
    ct_side_strategies: dict[str, float] = field(default_factory=dict)

    # Default setups by map
    default_setups: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Tendencies
    tendencies: list[str] = field(default_factory=list)

    # Predicted bans/picks
    predicted_map_bans: list[str] = field(default_factory=list)
    predicted_map_picks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_name": self.team_name,
            "players": self.players,
            "avg_rating": round(self.avg_rating, 2),
            "maps_played": self.maps_played,
            "win_rate": round(self.win_rate, 2),
            "t_side_strategies": self.t_side_strategies,
            "ct_side_strategies": self.ct_side_strategies,
            "default_setups": self.default_setups,
            "tendencies": self.tendencies,
            "predicted_map_bans": self.predicted_map_bans,
            "predicted_map_picks": self.predicted_map_picks,
        }


@dataclass
class CounterTactic:
    """A suggested counter-tactic against opponent."""

    title: str
    description: str
    targets_tendency: str
    effectiveness: float  # 0-1 estimated effectiveness
    difficulty: str  # "easy", "medium", "hard"
    map_specific: str | None = None
    side: str | None = None
    required_utility: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "targets_tendency": self.targets_tendency,
            "effectiveness": round(self.effectiveness, 2),
            "difficulty": self.difficulty,
            "map_specific": self.map_specific,
            "side": self.side,
            "required_utility": self.required_utility,
            "steps": self.steps,
        }


# ============================================================================
# HLTV Data Integration
# ============================================================================


class HLTVClient:
    """
    Client for fetching data from HLTV.
    Note: HLTV doesn't have a public API, so this uses web scraping with caching.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".opensight" / "hltv_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = 24  # Cache data for 24 hours

    def search_player(self, name: str) -> dict[str, Any] | None:
        """
        Search for a player on HLTV by name.

        Args:
            name: Player name to search

        Returns:
            Player data if found
        """
        cache_key = f"search_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # HLTV search URL pattern
        # In production, would need proper web scraping or unofficial API
        # For now, return mock structure that can be populated
        result = {
            "name": name,
            "hltv_id": None,
            "team": None,
            "country": None,
            "rating_2_0": None,
            "maps_played": 0,
            "source": "search",
        }

        self._set_cached(cache_key, result)
        return result

    def get_player_stats(self, hltv_id: int) -> dict[str, Any] | None:
        """
        Get detailed stats for a player by HLTV ID.

        Args:
            hltv_id: HLTV player ID

        Returns:
            Player statistics
        """
        cache_key = f"player_{hltv_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Would fetch from HLTV stats page
        # https://www.hltv.org/stats/players/{id}/{name}
        result = {
            "hltv_id": hltv_id,
            "rating_2_0": 0.0,
            "dpr": 0.0,
            "kast": 0.0,
            "impact": 0.0,
            "adr": 0.0,
            "kpr": 0.0,
            "maps_played": 0,
            "rounds_played": 0,
            "opening_kills": 0,
            "opening_deaths": 0,
            "rounds_with_kills": 0,
            "clutches_won": 0,
            "clutches_total": 0,
            "source": "hltv",
        }

        self._set_cached(cache_key, result)
        return result

    def get_player_matches(self, hltv_id: int, limit: int = 20) -> list[dict[str, Any]]:
        """
        Get recent matches for a player.

        Args:
            hltv_id: HLTV player ID
            limit: Maximum matches to return

        Returns:
            List of recent match data
        """
        cache_key = f"matches_{hltv_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Would fetch from HLTV matches page
        result = []
        self._set_cached(cache_key, result)
        return result

    def get_team_stats(self, team_name: str) -> dict[str, Any] | None:
        """
        Get team statistics from HLTV.

        Args:
            team_name: Team name

        Returns:
            Team statistics
        """
        cache_key = f"team_{hashlib.md5(team_name.encode()).hexdigest()[:8]}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = {
            "team_name": team_name,
            "world_ranking": None,
            "avg_player_age": None,
            "maps_played": 0,
            "wins": 0,
            "losses": 0,
            "map_pool": [],
            "source": "hltv",
        }

        self._set_cached(cache_key, result)
        return result

    def _get_cached(self, key: str) -> Any | None:
        """Get cached data if not expired."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                # Check expiry
                cached_time = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
                if (datetime.now() - cached_time).total_seconds() < self.cache_ttl_hours * 3600:
                    return data.get("data")
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        return None

    def _set_cached(self, key: str, data: Any) -> None:
        """Cache data with timestamp."""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"_cached_at": datetime.now().isoformat(), "data": data}, f, indent=2)
        except OSError:
            pass


# ============================================================================
# Tendency Analysis Engine
# ============================================================================


class TendencyAnalyzer:
    """
    Analyzes demo data to identify player tendencies.
    """

    def __init__(self):
        self.counter_tactic_templates = self._build_counter_tactics()

    def analyze_player(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        steamid: str,
        existing_profile: OpponentProfile | None = None,
    ) -> OpponentProfile:
        """
        Analyze a player from demo data to build/update their profile.

        Args:
            demo_data: Raw demo data
            player_stats: Analyzed statistics
            steamid: Player's Steam ID
            existing_profile: Existing profile to update

        Returns:
            Updated opponent profile
        """
        profile = existing_profile or OpponentProfile(steamid=steamid)

        # Update basic info
        profile.name = player_stats.get("name", profile.name or "Unknown")
        profile.local_demos_count += 1
        profile.last_updated = datetime.now().isoformat()

        # Analyze playstyle
        self._analyze_playstyle(demo_data, player_stats, profile)

        # Analyze weapon preferences
        self._analyze_weapons(demo_data, player_stats, profile)

        # Analyze tendencies
        self._analyze_tendencies(demo_data, player_stats, profile)

        # Update map-specific stats
        map_name = demo_data.get("map_name", "unknown")
        self._update_map_stats(demo_data, player_stats, map_name, profile)

        # Predict role
        self._predict_role(player_stats, profile)

        return profile

    def _analyze_playstyle(
        self, demo_data: dict[str, Any], player_stats: dict[str, Any], profile: OpponentProfile
    ) -> None:
        """Determine player's playstyle from stats."""
        # Aggression metrics
        opening_attempts = player_stats.get("opening_duel_attempts", 0)
        rounds = player_stats.get("rounds_played", 1)
        adr = player_stats.get("adr", 0)
        deaths_per_round = player_stats.get("deaths", 0) / max(rounds, 1)
        utility_used = player_stats.get("utility_per_round", 0)

        # Calculate playstyle scores (protect against division by zero)
        aggression_score = min(
            1.0, (opening_attempts / max(rounds, 1)) * 2 + deaths_per_round * 0.5
        )
        utility_score = min(1.0, utility_used / 3)
        aim_score = min(1.0, adr / 100)

        # Determine primary playstyle
        if aggression_score > 0.6 and aim_score > 0.5:
            profile.playstyle = PlayStyle.AGGRESSIVE
            profile.playstyle_confidence = aggression_score
        elif utility_score > 0.7:
            profile.playstyle = PlayStyle.UTILITY_HEAVY
            profile.playstyle_confidence = utility_score
        elif aggression_score < 0.3:
            profile.playstyle = PlayStyle.PASSIVE
            profile.playstyle_confidence = 1 - aggression_score
        elif aim_score > 0.7 and utility_score < 0.3:
            profile.playstyle = PlayStyle.AIM_HEAVY
            profile.playstyle_confidence = aim_score
        else:
            profile.playstyle = PlayStyle.BALANCED
            profile.playstyle_confidence = 0.5

    def _analyze_weapons(
        self, demo_data: dict[str, Any], player_stats: dict[str, Any], profile: OpponentProfile
    ) -> None:
        """Analyze weapon preferences."""
        weapon_kills = player_stats.get("weapon_kills", {})
        total_kills = sum(weapon_kills.values()) if weapon_kills else 1

        for weapon, kills in weapon_kills.items():
            current = profile.weapon_preferences.get(weapon, 0)
            new_rate = kills / total_kills
            # Exponential moving average
            profile.weapon_preferences[weapon] = current * 0.7 + new_rate * 0.3

    def _analyze_tendencies(
        self, demo_data: dict[str, Any], player_stats: dict[str, Any], profile: OpponentProfile
    ) -> None:
        """Identify specific behavioral tendencies."""
        map_name = demo_data.get("map_name", "unknown")
        new_tendencies = []

        # Positioning tendencies
        deaths = demo_data.get("deaths", [])
        kills = demo_data.get("kills", [])

        # Check for aggressive positioning
        early_deaths = sum(
            1
            for d in deaths
            if d.get("victim_steamid") == profile.steamid and d.get("round_time", 999) < 30
        )
        if early_deaths > 3:
            new_tendencies.append(
                PlayerTendency(
                    category=TendencyCategory.AGGRESSION,
                    description="Tends to push early in rounds",
                    frequency=early_deaths / max(len(deaths), 1),
                    confidence=0.7,
                    map_specific=map_name,
                    counter_tactics=[
                        "Hold angles and let them push into you",
                        "Set up utility traps for early aggression",
                    ],
                )
            )

        # Check for AWP usage
        awp_kills = sum(
            1
            for k in kills
            if k.get("attacker_steamid") == profile.steamid and k.get("weapon") in ["awp", "ssg08"]
        )
        total_kills = sum(1 for k in kills if k.get("attacker_steamid") == profile.steamid)
        if total_kills > 0 and awp_kills / total_kills > 0.4:
            new_tendencies.append(
                PlayerTendency(
                    category=TendencyCategory.WEAPON_CHOICE,
                    description="Primary AWPer - relies heavily on AWP",
                    frequency=awp_kills / total_kills,
                    confidence=0.8,
                    counter_tactics=[
                        "Force close engagements",
                        "Use smokes to deny AWP angles",
                        "Rush and trade quickly",
                    ],
                )
            )

        # Check for same-spot holds
        death_positions = defaultdict(int)
        for d in deaths:
            if d.get("victim_steamid") == profile.steamid:
                pos = (
                    round(d.get("victim_x", 0), -1),  # Round to 10 units
                    round(d.get("victim_y", 0), -1),
                    round(d.get("victim_z", 0), -1),
                )
                death_positions[pos] += 1

        repeat_spots = [pos for pos, count in death_positions.items() if count >= 2]
        if repeat_spots:
            new_tendencies.append(
                PlayerTendency(
                    category=TendencyCategory.POSITIONING,
                    description="Predictable positioning - holds same spots repeatedly",
                    frequency=len(repeat_spots) / max(len(death_positions), 1),
                    confidence=0.6,
                    map_specific=map_name,
                    examples=[f"Position near {pos}" for pos in repeat_spots[:3]],
                    counter_tactics=["Prefire these positions", "Use utility to flush them out"],
                )
            )

        # Check utility usage patterns
        grenades = demo_data.get("grenades", [])
        player_nades = [g for g in grenades if g.get("player_steamid") == profile.steamid]

        smoke_count = sum(1 for g in player_nades if g.get("grenade_type") == "smoke")
        sum(1 for g in player_nades if g.get("grenade_type") == "flashbang")

        if smoke_count < 3 and profile.local_demos_count > 1:
            new_tendencies.append(
                PlayerTendency(
                    category=TendencyCategory.UTILITY_USAGE,
                    description="Rarely uses smokes",
                    frequency=1 - (smoke_count / max(player_stats.get("rounds_played", 10), 1)),
                    confidence=0.6,
                    counter_tactics=[
                        "They won't smoke key angles - use them",
                        "Expect dry takes without utility cover",
                    ],
                )
            )

        # Merge with existing tendencies
        existing_cats = {t.category for t in profile.tendencies}
        for nt in new_tendencies:
            if nt.category not in existing_cats:
                profile.tendencies.append(nt)
            else:
                # Update existing tendency
                for i, et in enumerate(profile.tendencies):
                    if et.category == nt.category:
                        # Average the frequencies
                        profile.tendencies[i].frequency = (et.frequency + nt.frequency) / 2
                        profile.tendencies[i].confidence = min(1.0, et.confidence + 0.1)
                        break

    def _update_map_stats(
        self,
        demo_data: dict[str, Any],
        player_stats: dict[str, Any],
        map_name: str,
        profile: OpponentProfile,
    ) -> None:
        """Update map-specific statistics."""
        if map_name not in profile.map_stats:
            profile.map_stats[map_name] = {
                "games": 0,
                "kills": 0,
                "deaths": 0,
                "rating_sum": 0.0,
                "ct_rounds": 0,
                "t_rounds": 0,
                "favorite_positions": [],
            }

        stats = profile.map_stats[map_name]
        stats["games"] += 1
        stats["kills"] += player_stats.get("kills", 0)
        stats["deaths"] += player_stats.get("deaths", 0)
        stats["rating_sum"] += player_stats.get("hltv_rating", 1.0)
        stats["ct_rounds"] += player_stats.get("ct_rounds", 0)
        stats["t_rounds"] += player_stats.get("t_rounds", 0)

    def _predict_role(self, player_stats: dict[str, Any], profile: OpponentProfile) -> None:
        """Predict player's role from statistics."""
        # Role indicators
        scores = {
            "entry": 0.0,
            "awp": 0.0,
            "support": 0.0,
            "lurker": 0.0,
            "igl": 0.0,
            "rifler": 0.0,
        }

        # Entry indicators
        opening_wr = player_stats.get("opening_duel_win_rate", 0)
        opening_attempts = player_stats.get("opening_duel_attempts", 0)
        rounds = player_stats.get("rounds_played", 1)

        if opening_attempts / max(rounds, 1) > 0.3:
            scores["entry"] += 0.4
        if opening_wr > 50:
            scores["entry"] += 0.3

        # AWP indicators
        awp_preference = profile.weapon_preferences.get("awp", 0)
        if awp_preference > 0.3:
            scores["awp"] += 0.7

        # Support indicators
        utility_per_round = player_stats.get("utility_per_round", 0)
        trade_rate = player_stats.get("trade_rate", 0)
        if utility_per_round > 2:
            scores["support"] += 0.4
        if trade_rate > 50:
            scores["support"] += 0.3

        # Lurker indicators
        if profile.playstyle == PlayStyle.PASSIVE:
            scores["lurker"] += 0.3
        survival = player_stats.get("survival_rate", 0)
        if survival > 40:
            scores["lurker"] += 0.2

        # IGL indicators (hard to detect from stats)
        scores["igl"] = 0.1  # Default low

        # Rifler is default
        scores["rifler"] = 0.3

        # Determine role
        best_role = max(scores.items(), key=lambda x: x[1])
        profile.predicted_role = best_role[0]
        profile.role_confidence = best_role[1]

    def _build_counter_tactics(self) -> dict[str, list[CounterTactic]]:
        """Build counter-tactic templates for common tendencies."""
        return {
            "aggressive": [
                CounterTactic(
                    title="Bait the Aggression",
                    description="Let them push into prepared positions",
                    targets_tendency="early_aggression",
                    effectiveness=0.7,
                    difficulty="medium",
                    required_utility=["smoke", "molotov"],
                    steps=[
                        "Hold passive angles initially",
                        "Let them commit to the push",
                        "Trade them when they overextend",
                        "Use utility to punish aggression",
                    ],
                ),
                CounterTactic(
                    title="Counter-Flash",
                    description="Flash as they peek for easy kills",
                    targets_tendency="dry_peeks",
                    effectiveness=0.6,
                    difficulty="easy",
                    required_utility=["flashbang"],
                    steps=[
                        "Listen for movement",
                        "Throw popflash as they peek",
                        "Hold angle and get easy kill",
                    ],
                ),
            ],
            "awp_heavy": [
                CounterTactic(
                    title="Smoke and Execute",
                    description="Deny AWP angles with smokes",
                    targets_tendency="awp_reliance",
                    effectiveness=0.8,
                    difficulty="easy",
                    required_utility=["smoke", "smoke", "flashbang"],
                    steps=[
                        "Smoke known AWP positions",
                        "Execute through smokes",
                        "Force close-range fights",
                    ],
                ),
                CounterTactic(
                    title="Rush and Trade",
                    description="Overwhelm with numbers before AWP can impact",
                    targets_tendency="awp_reliance",
                    effectiveness=0.6,
                    difficulty="medium",
                    steps=[
                        "Rush together as a unit",
                        "Be prepared to trade first death",
                        "Don't give AWP time to reposition",
                    ],
                ),
            ],
            "passive": [
                CounterTactic(
                    title="Take Map Control",
                    description="Gain info advantage against passive defense",
                    targets_tendency="passive_play",
                    effectiveness=0.7,
                    difficulty="easy",
                    steps=[
                        "Take map control methodically",
                        "Gather information on rotations",
                        "Execute when you have numbers advantage",
                    ],
                )
            ],
            "predictable_positions": [
                CounterTactic(
                    title="Prefire Common Spots",
                    description="Prefire their predictable positions",
                    targets_tendency="same_positions",
                    effectiveness=0.75,
                    difficulty="medium",
                    steps=[
                        "Note their common positions",
                        "Prefire these spots on entry",
                        "Coordinate with flash support",
                    ],
                )
            ],
        }

    def get_counter_tactics(
        self, profile: OpponentProfile, map_name: str | None = None
    ) -> list[CounterTactic]:
        """
        Get recommended counter-tactics for an opponent.

        Args:
            profile: Opponent profile
            map_name: Optional map filter

        Returns:
            List of recommended counter-tactics
        """
        tactics = []

        # Get tactics based on playstyle
        if profile.playstyle == PlayStyle.AGGRESSIVE:
            tactics.extend(self.counter_tactic_templates.get("aggressive", []))

        if profile.weapon_preferences.get("awp", 0) > 0.3:
            tactics.extend(self.counter_tactic_templates.get("awp_heavy", []))

        if profile.playstyle == PlayStyle.PASSIVE:
            tactics.extend(self.counter_tactic_templates.get("passive", []))

        # Get tactics based on tendencies
        for tendency in profile.tendencies:
            if tendency.category == TendencyCategory.POSITIONING:
                tactics.extend(self.counter_tactic_templates.get("predictable_positions", []))

            # Add tendency-specific counters
            for ct in tendency.counter_tactics:
                tactics.append(
                    CounterTactic(
                        title=f"Counter: {tendency.description[:30]}",
                        description=ct,
                        targets_tendency=tendency.category.value,
                        effectiveness=tendency.confidence,
                        difficulty="medium",
                    )
                )

        # Filter by map if specified
        if map_name:
            tactics = [t for t in tactics if t.map_specific is None or t.map_specific == map_name]

        # Remove duplicates and sort by effectiveness
        seen = set()
        unique_tactics = []
        for t in tactics:
            if t.title not in seen:
                seen.add(t.title)
                unique_tactics.append(t)

        unique_tactics.sort(key=lambda x: x.effectiveness, reverse=True)

        return unique_tactics[:10]  # Top 10 tactics


# ============================================================================
# Opponent Modeling Engine
# ============================================================================


class OpponentModeler:
    """
    Main engine for opponent modeling and analysis.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path.home() / ".opensight" / "opponents"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.hltv_client = HLTVClient()
        self.analyzer = TendencyAnalyzer()

        # Cache of opponent profiles
        self.profiles: dict[str, OpponentProfile] = {}

    def get_profile(self, steamid: str) -> OpponentProfile:
        """Get or create opponent profile."""
        if steamid not in self.profiles:
            profile_path = self.data_dir / f"opponent_{steamid}.json"
            if profile_path.exists():
                try:
                    with open(profile_path) as f:
                        data = json.load(f)
                    self.profiles[steamid] = OpponentProfile.from_dict(data)
                except (OSError, json.JSONDecodeError):
                    self.profiles[steamid] = OpponentProfile(steamid=steamid)
            else:
                self.profiles[steamid] = OpponentProfile(steamid=steamid)
        return self.profiles[steamid]

    def save_profile(self, profile: OpponentProfile) -> None:
        """Save opponent profile to disk."""
        self.profiles[profile.steamid] = profile
        profile_path = self.data_dir / f"opponent_{profile.steamid}.json"
        try:
            with open(profile_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)
        except OSError:
            pass

    def analyze_opponent(
        self, steamid: str, demo_data: dict[str, Any], player_stats: dict[str, Any]
    ) -> OpponentProfile:
        """
        Analyze an opponent from demo data.

        Args:
            steamid: Opponent's Steam ID
            demo_data: Raw demo data
            player_stats: Analyzed statistics

        Returns:
            Updated opponent profile
        """
        existing = self.get_profile(steamid)
        updated = self.analyzer.analyze_player(demo_data, player_stats, steamid, existing)
        self.save_profile(updated)
        return updated

    def analyze_demo_opponents(
        self, demo_data: dict[str, Any], my_steamid: str
    ) -> list[OpponentProfile]:
        """
        Analyze all opponents from a demo.

        Args:
            demo_data: Raw demo data with all players
            my_steamid: Your Steam ID to exclude

        Returns:
            List of opponent profiles
        """
        profiles = []

        # Get all player stats from demo
        player_stats = demo_data.get("player_stats", {})

        # Determine my team
        my_team = None
        for sid, stats in player_stats.items():
            if sid == my_steamid:
                my_team = stats.get("team")
                break

        # Analyze opponents (players on other team)
        for sid, stats in player_stats.items():
            if sid == my_steamid:
                continue

            player_team = stats.get("team")
            if player_team and my_team and player_team != my_team:
                profile = self.analyze_opponent(sid, demo_data, stats)
                profiles.append(profile)

        return profiles

    def enrich_with_hltv(self, profile: OpponentProfile) -> OpponentProfile:
        """
        Enrich opponent profile with HLTV data if available.

        Args:
            profile: Opponent profile to enrich

        Returns:
            Enriched profile
        """
        if not profile.name or profile.name == "Unknown":
            return profile

        # Search HLTV for player
        hltv_data = self.hltv_client.search_player(profile.name)
        if hltv_data and hltv_data.get("hltv_id"):
            profile.hltv_id = hltv_data["hltv_id"]
            profile.team = hltv_data.get("team", profile.team)
            profile.country = hltv_data.get("country", profile.country)

            # Get detailed stats
            stats = self.hltv_client.get_player_stats(profile.hltv_id)
            if stats:
                profile.hltv_rating = stats.get("rating_2_0", 0.0)
                profile.maps_played = stats.get("maps_played", 0)

        self.save_profile(profile)
        return profile

    def get_counter_tactics(
        self, steamid: str, map_name: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get counter-tactics for an opponent.

        Args:
            steamid: Opponent's Steam ID
            map_name: Optional map filter

        Returns:
            List of counter-tactic dictionaries
        """
        profile = self.get_profile(steamid)
        tactics = self.analyzer.get_counter_tactics(profile, map_name)
        return [t.to_dict() for t in tactics]

    def get_scouting_report(self, steamid: str) -> dict[str, Any]:
        """
        Generate a comprehensive scouting report for an opponent.

        Args:
            steamid: Opponent's Steam ID

        Returns:
            Scouting report dictionary
        """
        profile = self.get_profile(steamid)

        return {
            "player": {
                "name": profile.name,
                "steamid": profile.steamid,
                "team": profile.team,
                "country": profile.country,
                "hltv_rating": profile.hltv_rating,
            },
            "playstyle": {
                "type": profile.playstyle.value,
                "confidence": profile.playstyle_confidence,
                "description": self._describe_playstyle(profile.playstyle),
            },
            "role": {"predicted": profile.predicted_role, "confidence": profile.role_confidence},
            "weapons": {
                "primary": self._get_primary_weapon(profile),
                "preferences": dict(
                    sorted(profile.weapon_preferences.items(), key=lambda x: x[1], reverse=True)[:5]
                ),
            },
            "tendencies": [t.to_dict() for t in profile.tendencies[:5]],
            "map_performance": self._summarize_map_stats(profile),
            "counter_tactics": self.get_counter_tactics(steamid)[:5],
            "data_quality": {
                "demos_analyzed": profile.local_demos_count,
                "has_hltv_data": profile.hltv_id is not None,
                "last_updated": profile.last_updated,
            },
        }

    def _describe_playstyle(self, playstyle: PlayStyle) -> str:
        """Get human-readable playstyle description."""
        descriptions = {
            PlayStyle.AGGRESSIVE: "Plays aggressively, often taking early fights and pushing",
            PlayStyle.PASSIVE: "Plays passively, preferring to hold angles and wait",
            PlayStyle.BALANCED: "Balanced playstyle, adapts to situations",
            PlayStyle.TACTICAL: "Tactical player, focuses on strategy over aim",
            PlayStyle.AIM_HEAVY: "Relies heavily on aim, less utility usage",
            PlayStyle.UTILITY_HEAVY: "Uses lots of utility, team-oriented player",
        }
        return descriptions.get(playstyle, "Unknown playstyle")

    def _get_primary_weapon(self, profile: OpponentProfile) -> str:
        """Determine primary weapon from preferences."""
        if not profile.weapon_preferences:
            return "ak47"

        return max(profile.weapon_preferences.items(), key=lambda x: x[1])[0]

    def _summarize_map_stats(self, profile: OpponentProfile) -> dict[str, Any]:
        """Summarize map performance."""
        summary = {}
        for map_name, stats in profile.map_stats.items():
            games = stats.get("games", 1)
            summary[map_name] = {
                "games": games,
                "avg_kills": stats.get("kills", 0) / games,
                "avg_deaths": stats.get("deaths", 0) / games,
                "avg_rating": stats.get("rating_sum", 1.0) / games,
            }
        return summary

    def build_team_profile(self, team_name: str, player_steamids: list[str]) -> TeamProfile:
        """
        Build a team profile from individual player profiles.

        Args:
            team_name: Team name
            player_steamids: List of player Steam IDs

        Returns:
            Team profile
        """
        team = TeamProfile(team_name=team_name, players=player_steamids)

        # Aggregate player data
        ratings = []
        for steamid in player_steamids:
            profile = self.get_profile(steamid)
            if profile.hltv_rating > 0:
                ratings.append(profile.hltv_rating)

        team.avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

        # Try to get team data from HLTV
        hltv_team = self.hltv_client.get_team_stats(team_name)
        if hltv_team:
            team.maps_played = hltv_team.get("maps_played", 0)
            if hltv_team.get("wins") and hltv_team.get("losses"):
                total = hltv_team["wins"] + hltv_team["losses"]
                team.win_rate = hltv_team["wins"] / total if total > 0 else 0

        return team


# ============================================================================
# Convenience Functions
# ============================================================================

_default_modeler: OpponentModeler | None = None


def get_modeler() -> OpponentModeler:
    """Get or create the default opponent modeler."""
    global _default_modeler
    if _default_modeler is None:
        _default_modeler = OpponentModeler()
    return _default_modeler


def analyze_opponent(
    steamid: str, demo_data: dict[str, Any], player_stats: dict[str, Any]
) -> dict[str, Any]:
    """
    Analyze an opponent and return their profile.

    Args:
        steamid: Opponent's Steam ID
        demo_data: Raw demo data
        player_stats: Analyzed statistics

    Returns:
        Opponent profile dictionary
    """
    modeler = get_modeler()
    profile = modeler.analyze_opponent(steamid, demo_data, player_stats)
    return profile.to_dict()


def get_scouting_report(steamid: str) -> dict[str, Any]:
    """
    Get a scouting report for an opponent.

    Args:
        steamid: Opponent's Steam ID

    Returns:
        Scouting report dictionary
    """
    return get_modeler().get_scouting_report(steamid)


def get_counter_tactics(steamid: str, map_name: str | None = None) -> list[dict[str, Any]]:
    """
    Get counter-tactics for an opponent.

    Args:
        steamid: Opponent's Steam ID
        map_name: Optional map filter

    Returns:
        List of counter-tactics
    """
    return get_modeler().get_counter_tactics(steamid, map_name)
