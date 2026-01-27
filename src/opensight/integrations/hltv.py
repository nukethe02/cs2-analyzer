"""
HLTV Stats Integration Module

Provides:
- Player career stats lookup
- Team rankings and info
- Map statistics
- Match metadata enrichment
- Pro player detection

Note: HLTV does not have a public API. This module scrapes publicly available
data or uses cached community-maintained datasets. For production use,
consider using official APIs when available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache duration in hours
CACHE_DURATION_HOURS = 24

# User agent for requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class HLTVPlayerStats:
    """HLTV player statistics."""

    player_id: int
    nickname: str
    full_name: str = ""
    country: str = ""
    team_name: str = ""
    rating_2_0: float = 0.0
    maps_played: int = 0
    kills_per_round: float = 0.0
    deaths_per_round: float = 0.0
    headshot_percentage: float = 0.0
    kast_percentage: float = 0.0
    impact_rating: float = 0.0
    adr: float = 0.0

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "nickname": self.nickname,
            "full_name": self.full_name,
            "country": self.country,
            "team": self.team_name,
            "rating": self.rating_2_0,
            "maps": self.maps_played,
            "kpr": self.kills_per_round,
            "dpr": self.deaths_per_round,
            "hs_pct": self.headshot_percentage,
            "kast": self.kast_percentage,
            "impact": self.impact_rating,
            "adr": self.adr,
        }


@dataclass
class HLTVTeamInfo:
    """HLTV team information."""

    team_id: int
    name: str
    country: str = ""
    world_ranking: int = 0
    weeks_in_top_30: int = 0
    roster: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "country": self.country,
            "ranking": self.world_ranking,
            "roster": self.roster,
        }


@dataclass
class MapStatistics:
    """Map-specific statistics."""

    map_name: str
    ct_win_rate: float = 50.0
    t_win_rate: float = 50.0
    avg_round_time_seconds: float = 90.0
    common_positions: list[str] = field(default_factory=list)
    meta_weapons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "map": self.map_name,
            "ct_win_pct": self.ct_win_rate,
            "t_win_pct": self.t_win_rate,
            "avg_round_time": self.avg_round_time_seconds,
            "common_positions": self.common_positions,
            "meta_weapons": self.meta_weapons,
        }


# Cached pro player database (community-maintained)
# This is a small subset - full database would be fetched/updated
PRO_PLAYERS_DB = {
    # Format: steam_id -> {"name": ..., "team": ..., "hltv_id": ...}
    76561198034202275: {"name": "s1mple", "team": "NAVI", "hltv_id": 7998},
    76561198066693739: {"name": "ZywOo", "team": "Vitality", "hltv_id": 11893},
    76561198073611610: {"name": "NiKo", "team": "G2", "hltv_id": 3741},
    76561198079326014: {"name": "dev1ce", "team": "Astralis", "hltv_id": 7592},
    76561198004854956: {"name": "Twistzz", "team": "FaZe", "hltv_id": 10394},
    76561198152618007: {"name": "m0NESY", "team": "G2", "hltv_id": 18987},
    76561198087135745: {"name": "ropz", "team": "FaZe", "hltv_id": 11816},
    76561198258562177: {"name": "donk", "team": "Spirit", "hltv_id": 20978},
}

# Map statistics (average values from pro matches)
MAP_STATS_DB = {
    "de_dust2": MapStatistics(
        map_name="Dust II",
        ct_win_rate=49.5,
        t_win_rate=50.5,
        avg_round_time_seconds=95,
        common_positions=["Long A", "Short A", "B Tunnels", "Mid Doors", "CT Spawn"],
        meta_weapons=["AK-47", "AWP", "M4A1-S"],
    ),
    "de_mirage": MapStatistics(
        map_name="Mirage",
        ct_win_rate=52.1,
        t_win_rate=47.9,
        avg_round_time_seconds=92,
        common_positions=["A Ramp", "Palace", "Connector", "B Apartments", "Window"],
        meta_weapons=["AK-47", "M4A4", "AWP"],
    ),
    "de_inferno": MapStatistics(
        map_name="Inferno",
        ct_win_rate=53.2,
        t_win_rate=46.8,
        avg_round_time_seconds=88,
        common_positions=["Banana", "Apartments", "A Long", "Pit", "Library"],
        meta_weapons=["AK-47", "M4A1-S", "AWP", "Molotov"],
    ),
    "de_nuke": MapStatistics(
        map_name="Nuke",
        ct_win_rate=55.8,
        t_win_rate=44.2,
        avg_round_time_seconds=85,
        common_positions=["Ramp", "Outside", "Secret", "Heaven", "Vent"],
        meta_weapons=["M4A1-S", "AK-47", "P250"],
    ),
    "de_overpass": MapStatistics(
        map_name="Overpass",
        ct_win_rate=51.5,
        t_win_rate=48.5,
        avg_round_time_seconds=90,
        common_positions=["Connector", "B Long", "A Long", "Playground", "Monster"],
        meta_weapons=["AK-47", "M4A4", "AWP"],
    ),
    "de_vertigo": MapStatistics(
        map_name="Vertigo",
        ct_win_rate=52.8,
        t_win_rate=47.2,
        avg_round_time_seconds=82,
        common_positions=["A Ramp", "B Stairs", "Mid", "Elevator", "Generator"],
        meta_weapons=["AK-47", "M4A1-S", "MAC-10"],
    ),
    "de_ancient": MapStatistics(
        map_name="Ancient",
        ct_win_rate=50.3,
        t_win_rate=49.7,
        avg_round_time_seconds=94,
        common_positions=["A Main", "B Ramp", "Mid", "Donut", "Cave"],
        meta_weapons=["AK-47", "M4A4", "AWP"],
    ),
    "de_anubis": MapStatistics(
        map_name="Anubis",
        ct_win_rate=51.2,
        t_win_rate=48.8,
        avg_round_time_seconds=91,
        common_positions=["A Main", "B Main", "Mid", "Canal", "Connector"],
        meta_weapons=["AK-47", "M4A1-S", "AWP"],
    ),
}


class HLTVCache:
    """
    Cache for HLTV data to avoid excessive requests.
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir or Path.home() / ".opensight" / "hltv_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        hash_key = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()
        return self.cache_dir / f"{hash_key}.json"

    def get(self, key: str) -> dict | None:
        """Get cached data if not expired."""
        path = self._get_cache_path(key)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            # Check expiration
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if datetime.now() - cached_at > timedelta(hours=CACHE_DURATION_HOURS):
                return None

            return data.get("value")

        except Exception:
            return None

    def set(self, key: str, value: dict):
        """Cache data."""
        path = self._get_cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump(
                    {
                        "cached_at": datetime.now().isoformat(),
                        "value": value,
                    },
                    f,
                )
        except Exception as e:
            logger.warning(f"Failed to cache: {e}")

    def clear(self):
        """Clear all cached data."""
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
            except Exception:
                pass


class HLTVClient:
    """
    Client for fetching HLTV data.

    Note: HLTV has no public API. This uses publicly available data
    and caching to minimize requests.
    """

    def __init__(self, cache: HLTVCache | None = None):
        """
        Initialize the HLTV client.

        Args:
            cache: Optional cache instance
        """
        self.cache = cache or HLTVCache()
        self.base_url = "https://www.hltv.org"

    def _make_request(self, url: str) -> str | None:
        """Make HTTP request with error handling."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode("utf-8")
        except Exception as e:
            logger.warning(f"HLTV request failed: {e}")
            return None

    def is_pro_player(self, steam_id: int) -> bool:
        """
        Check if a steam ID belongs to a known pro player.

        Args:
            steam_id: Player's Steam ID

        Returns:
            True if pro player
        """
        return steam_id in PRO_PLAYERS_DB

    def get_pro_player_info(self, steam_id: int) -> dict | None:
        """
        Get info about a pro player.

        Args:
            steam_id: Player's Steam ID

        Returns:
            Player info dict or None
        """
        if steam_id in PRO_PLAYERS_DB:
            return PRO_PLAYERS_DB[steam_id].copy()
        return None

    def get_map_stats(self, map_name: str) -> MapStatistics | None:
        """
        Get statistics for a map.

        Args:
            map_name: Map name (e.g., "de_dust2")

        Returns:
            MapStatistics or None
        """
        map_name = map_name.lower()
        return MAP_STATS_DB.get(map_name)

    def search_player(self, nickname: str) -> list[dict]:
        """
        Search for a player by nickname.

        Note: This uses cached data. Live search would require API access.

        Args:
            nickname: Player nickname to search

        Returns:
            List of matching players
        """
        nickname_lower = nickname.lower()
        results = []

        for steam_id, info in PRO_PLAYERS_DB.items():
            if nickname_lower in info["name"].lower():
                results.append(
                    {
                        "steam_id": steam_id,
                        **info,
                    }
                )

        return results

    def get_world_rankings(self, top_n: int = 30) -> list[dict]:
        """
        Get current world team rankings.

        Note: This uses cached/static data. Live rankings would require scraping.

        Returns:
            List of teams with rankings
        """
        # Static top teams (would be updated periodically)
        static_rankings = [
            {"rank": 1, "name": "G2 Esports", "points": 1000},
            {"rank": 2, "name": "Team Vitality", "points": 950},
            {"rank": 3, "name": "NAVI", "points": 900},
            {"rank": 4, "name": "FaZe Clan", "points": 850},
            {"rank": 5, "name": "Team Spirit", "points": 800},
            {"rank": 6, "name": "Astralis", "points": 750},
            {"rank": 7, "name": "Cloud9", "points": 700},
            {"rank": 8, "name": "Heroic", "points": 650},
            {"rank": 9, "name": "MOUZ", "points": 600},
            {"rank": 10, "name": "Complexity", "points": 550},
        ]

        return static_rankings[:top_n]


class MatchEnricher:
    """
    Enriches match analysis with external HLTV data.
    """

    def __init__(self, hltv_client: HLTVClient | None = None):
        """
        Initialize the enricher.

        Args:
            hltv_client: HLTV client instance
        """
        self.hltv = hltv_client or HLTVClient()

    def enrich_analysis(self, analysis_data: dict) -> dict:
        """
        Enrich match analysis with HLTV data.

        Args:
            analysis_data: Analysis dict from DemoAnalyzer

        Returns:
            Enriched analysis dict
        """
        enriched = analysis_data.copy()

        # Add map statistics
        map_name = analysis_data.get("map_name", "")
        map_stats = self.hltv.get_map_stats(map_name)
        if map_stats:
            enriched["map_statistics"] = map_stats.to_dict()

        # Check for pro players
        pro_players = []
        players = analysis_data.get("players", {})
        for steam_id_str, player_data in players.items():
            try:
                steam_id = int(steam_id_str)
                if self.hltv.is_pro_player(steam_id):
                    pro_info = self.hltv.get_pro_player_info(steam_id)
                    if pro_info:
                        pro_players.append(
                            {
                                "steam_id": steam_id,
                                "match_name": player_data.get("name", ""),
                                "pro_name": pro_info.get("name", ""),
                                "team": pro_info.get("team", ""),
                            }
                        )
            except (ValueError, TypeError):
                continue

        if pro_players:
            enriched["pro_players_detected"] = pro_players

        # Add context insights based on map
        if map_stats:
            insights = []

            # Side advantage insight
            ct_wr = map_stats.ct_win_rate
            if ct_wr > 52:
                insights.append(
                    {
                        "type": "map_meta",
                        "message": f"{map_stats.map_name} is CT-sided ({ct_wr:.1f}% CT win rate in pro play)",
                    }
                )
            elif ct_wr < 48:
                insights.append(
                    {
                        "type": "map_meta",
                        "message": f"{map_stats.map_name} is T-sided ({100 - ct_wr:.1f}% T win rate in pro play)",
                    }
                )

            # Add meta weapons
            if map_stats.meta_weapons:
                insights.append(
                    {
                        "type": "meta",
                        "message": f"Meta weapons on {map_stats.map_name}: {', '.join(map_stats.meta_weapons)}",
                    }
                )

            enriched["external_insights"] = insights

        return enriched

    def get_opponent_tendencies(self, player_names: list[str]) -> dict:
        """
        Get tendencies for known players.

        Args:
            player_names: List of player nicknames

        Returns:
            Dict with player tendencies
        """
        tendencies = {}

        for name in player_names:
            matches = self.hltv.search_player(name)
            if matches:
                best_match = matches[0]
                tendencies[name] = {
                    "pro_player": True,
                    "team": best_match.get("team", "Unknown"),
                    "typical_role": "Unknown",  # Would need more data
                }

        return tendencies


# Convenience functions


def enrich_match_analysis(analysis_data: dict) -> dict:
    """
    Convenience function to enrich analysis with HLTV data.

    Args:
        analysis_data: Analysis dict

    Returns:
        Enriched analysis dict
    """
    enricher = MatchEnricher()
    return enricher.enrich_analysis(analysis_data)


def get_map_statistics(map_name: str) -> dict | None:
    """Get statistics for a map."""
    stats = MAP_STATS_DB.get(map_name.lower())
    return stats.to_dict() if stats else None


def check_pro_players(steam_ids: list[int]) -> list[dict]:
    """Check which steam IDs are known pro players."""
    client = HLTVClient()
    results = []
    for sid in steam_ids:
        info = client.get_pro_player_info(sid)
        if info:
            results.append({"steam_id": sid, **info})
    return results
