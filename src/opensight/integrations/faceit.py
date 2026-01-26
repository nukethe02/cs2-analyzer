"""
OpenSight FACEIT API Integration

Provides integration with FACEIT's public API to retrieve
match history, player statistics, and ELO information.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# FACEIT API base URL
FACEIT_API_BASE = "https://open.faceit.com/data/v4"

# Rate limiting
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds


@dataclass
class FACEITPlayer:
    """FACEIT player profile information."""

    player_id: str
    nickname: str
    country: str = ""
    avatar: str = ""
    steam_id: str = ""
    faceit_elo: int = 0
    skill_level: int = 0  # 1-10
    games_played: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_kd: float = 0.0
    avg_hs_pct: float = 0.0
    recent_results: list[str] = field(default_factory=list)  # W/L strings


@dataclass
class FACEITMatch:
    """FACEIT match information."""

    match_id: str
    game: str = "cs2"
    region: str = ""
    competition_type: str = ""  # "5v5", "Wingman", etc.
    map_name: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: str = ""  # "finished", "ongoing", "cancelled"
    demo_url: str | None = None

    # Teams
    team1_name: str = ""
    team1_score: int = 0
    team1_players: list[str] = field(default_factory=list)

    team2_name: str = ""
    team2_score: int = 0
    team2_players: list[str] = field(default_factory=list)

    winner: str = ""  # "team1" or "team2"


@dataclass
class FACEITMatchStats:
    """Player statistics from a FACEIT match."""

    match_id: str
    player_id: str
    nickname: str
    team: str
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    headshots: int = 0
    headshot_pct: float = 0.0
    kd_ratio: float = 0.0
    kr_ratio: float = 0.0  # Kills per round
    mvps: int = 0
    triple_kills: int = 0
    quad_kills: int = 0
    aces: int = 0


class FACEITClient:
    """
    Client for interacting with the FACEIT API.

    Requires a FACEIT API key which can be obtained from:
    https://developers.faceit.com/

    Example:
        >>> from opensight.integrations.faceit import FACEITClient
        >>>
        >>> client = FACEITClient(api_key="your-api-key")
        >>>
        >>> # Get player by nickname
        >>> player = client.get_player_by_nickname("s1mple")
        >>> print(f"ELO: {player.faceit_elo}, Level: {player.skill_level}")
        >>>
        >>> # Get match history
        >>> matches = client.get_player_matches(player.player_id, limit=10)
        >>> for match in matches:
        ...     print(f"{match.map_name}: {match.team1_score}-{match.team2_score}")
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the FACEIT client.

        Args:
            api_key: FACEIT API key. If not provided, will try to read from
                     FACEIT_API_KEY environment variable.
        """
        import os

        self.api_key = api_key or os.environ.get("FACEIT_API_KEY")

        if not self.api_key:
            logger.warning(
                "No FACEIT API key provided. Set FACEIT_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._request_times: list[float] = []
        self._session = None

    def _get_session(self):
        """Get or create requests session."""
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
                if self.api_key:
                    self._session.headers.update(
                        {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
                    )
            except ImportError:
                raise ImportError(
                    "requests library required for FACEIT API. Install with: pip install requests"
                )
        return self._session

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        now = time.time()
        # Remove old requests from window
        self._request_times = [t for t in self._request_times if now - t < RATE_LIMIT_WINDOW]

        if len(self._request_times) >= RATE_LIMIT_REQUESTS:
            # Wait until oldest request is outside window
            sleep_time = RATE_LIMIT_WINDOW - (now - self._request_times[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        self._request_times.append(time.time())

    def _make_request(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Make a request to the FACEIT API."""
        if not self.api_key:
            logger.error("FACEIT API key required")
            return None

        self._check_rate_limit()

        session = self._get_session()
        url = f"{FACEIT_API_BASE}{endpoint}"

        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"FACEIT API request failed: {e}")
            return None

    def get_player_by_nickname(self, nickname: str) -> FACEITPlayer | None:
        """
        Get player profile by FACEIT nickname.

        Args:
            nickname: FACEIT username

        Returns:
            FACEITPlayer profile or None if not found
        """
        data = self._make_request(f"/players?nickname={nickname}")
        if not data:
            return None

        return self._parse_player(data)

    def get_player_by_steam_id(self, steam_id: str) -> FACEITPlayer | None:
        """
        Get player profile by Steam ID.

        Args:
            steam_id: Steam ID (64-bit format)

        Returns:
            FACEITPlayer profile or None if not found
        """
        data = self._make_request("/players", params={"game": "cs2", "game_player_id": steam_id})
        if not data:
            return None

        return self._parse_player(data)

    def get_player_stats(self, player_id: str) -> dict[str, Any] | None:
        """
        Get player statistics for CS2.

        Args:
            player_id: FACEIT player ID

        Returns:
            Dictionary with player statistics
        """
        data = self._make_request(f"/players/{player_id}/stats/cs2")
        if not data:
            return None

        return data

    def get_player_matches(
        self, player_id: str, limit: int = 20, offset: int = 0
    ) -> list[FACEITMatch]:
        """
        Get player's match history.

        Args:
            player_id: FACEIT player ID
            limit: Maximum matches to return (default 20, max 100)
            offset: Offset for pagination

        Returns:
            List of FACEITMatch objects
        """
        data = self._make_request(
            f"/players/{player_id}/history",
            params={"game": "cs2", "limit": min(limit, 100), "offset": offset},
        )
        if not data or "items" not in data:
            return []

        matches = []
        for item in data["items"]:
            match = self._parse_match(item)
            if match:
                matches.append(match)

        return matches

    def get_match_details(self, match_id: str) -> FACEITMatch | None:
        """
        Get detailed information about a specific match.

        Args:
            match_id: FACEIT match ID

        Returns:
            FACEITMatch with full details
        """
        data = self._make_request(f"/matches/{match_id}")
        if not data:
            return None

        return self._parse_match(data)

    def get_match_stats(self, match_id: str) -> list[FACEITMatchStats]:
        """
        Get player statistics from a specific match.

        Args:
            match_id: FACEIT match ID

        Returns:
            List of FACEITMatchStats for each player
        """
        data = self._make_request(f"/matches/{match_id}/stats")
        if not data or "rounds" not in data:
            return []

        stats = []
        for round_data in data.get("rounds", []):
            for team in round_data.get("teams", []):
                for player in team.get("players", []):
                    stat = self._parse_match_stats(match_id, player, team.get("team_id", ""))
                    if stat:
                        stats.append(stat)

        return stats

    def get_match_demo_url(self, match_id: str) -> str | None:
        """
        Get the demo download URL for a match.

        Args:
            match_id: FACEIT match ID

        Returns:
            Demo download URL or None if not available
        """
        data = self._make_request(f"/matches/{match_id}")
        if not data:
            return None

        return data.get("demo_url")

    def _parse_player(self, data: dict) -> FACEITPlayer:
        """Parse player data from API response."""
        games = data.get("games", {})
        cs2_data = games.get("cs2", {})

        return FACEITPlayer(
            player_id=data.get("player_id", ""),
            nickname=data.get("nickname", ""),
            country=data.get("country", ""),
            avatar=data.get("avatar", ""),
            steam_id=data.get("steam_id_64", ""),
            faceit_elo=cs2_data.get("faceit_elo", 0),
            skill_level=cs2_data.get("skill_level", 0),
        )

    def _parse_match(self, data: dict) -> FACEITMatch | None:
        """Parse match data from API response."""
        try:
            # Parse timestamps
            started_at = None
            finished_at = None
            if data.get("started_at"):
                started_at = datetime.fromtimestamp(data["started_at"])
            if data.get("finished_at"):
                finished_at = datetime.fromtimestamp(data["finished_at"])

            # Parse teams
            teams = data.get("teams", {})
            team1 = teams.get("faction1", {})
            team2 = teams.get("faction2", {})

            # Get results
            results = data.get("results", {})
            score = results.get("score", {})

            # Determine winner
            winner = ""
            if results.get("winner") == "faction1":
                winner = "team1"
            elif results.get("winner") == "faction2":
                winner = "team2"

            return FACEITMatch(
                match_id=data.get("match_id", ""),
                game=data.get("game", "cs2"),
                region=data.get("region", ""),
                competition_type=data.get("competition_type", ""),
                map_name=self._extract_map(data),
                started_at=started_at,
                finished_at=finished_at,
                status=data.get("status", ""),
                demo_url=data.get("demo_url"),
                team1_name=team1.get("name", "Team 1"),
                team1_score=score.get("faction1", 0),
                team1_players=self._extract_players(team1),
                team2_name=team2.get("name", "Team 2"),
                team2_score=score.get("faction2", 0),
                team2_players=self._extract_players(team2),
                winner=winner,
            )
        except Exception as e:
            logger.error(f"Error parsing match data: {e}")
            return None

    def _extract_map(self, data: dict) -> str:
        """Extract map name from match data."""
        voting = data.get("voting", {})
        map_data = voting.get("map", {})
        if isinstance(map_data, dict):
            pick = map_data.get("pick", [])
            if pick:
                return pick[0] if isinstance(pick, list) else pick
        return data.get("map", {}).get("name", "")

    def _extract_players(self, team_data: dict) -> list[str]:
        """Extract player nicknames from team data."""
        roster = team_data.get("roster", [])
        return [p.get("nickname", "") for p in roster]

    def _parse_match_stats(
        self, match_id: str, player_data: dict, team: str
    ) -> FACEITMatchStats | None:
        """Parse player match statistics."""
        try:
            stats = player_data.get("player_stats", {})

            return FACEITMatchStats(
                match_id=match_id,
                player_id=player_data.get("player_id", ""),
                nickname=player_data.get("nickname", ""),
                team=team,
                kills=int(stats.get("Kills", 0)),
                deaths=int(stats.get("Deaths", 0)),
                assists=int(stats.get("Assists", 0)),
                headshots=int(stats.get("Headshots", 0)),
                headshot_pct=float(stats.get("Headshots %", 0)),
                kd_ratio=float(stats.get("K/D Ratio", 0)),
                kr_ratio=float(stats.get("K/R Ratio", 0)),
                mvps=int(stats.get("MVPs", 0)),
                triple_kills=int(stats.get("Triple Kills", 0)),
                quad_kills=int(stats.get("Quadro Kills", 0)),
                aces=int(stats.get("Penta Kills", 0)),
            )
        except Exception as e:
            logger.error(f"Error parsing match stats: {e}")
            return None


def get_faceit_player(nickname: str, api_key: str | None = None) -> FACEITPlayer | None:
    """
    Convenience function to get a FACEIT player by nickname.

    Args:
        nickname: FACEIT username
        api_key: Optional API key (uses env var if not provided)

    Returns:
        FACEITPlayer or None
    """
    client = FACEITClient(api_key=api_key)
    return client.get_player_by_nickname(nickname)


def get_faceit_match_history(
    nickname: str, limit: int = 20, api_key: str | None = None
) -> list[FACEITMatch]:
    """
    Convenience function to get a player's FACEIT match history.

    Args:
        nickname: FACEIT username
        limit: Maximum matches to return
        api_key: Optional API key

    Returns:
        List of FACEITMatch objects
    """
    client = FACEITClient(api_key=api_key)
    player = client.get_player_by_nickname(nickname)
    if not player:
        return []
    return client.get_player_matches(player.player_id, limit=limit)
