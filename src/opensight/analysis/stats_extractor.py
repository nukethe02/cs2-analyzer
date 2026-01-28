"""
Stats Extractor for Enhanced Stats Tables

Extracts and formats player statistics for the enhanced frontend stats tables.
Provides data in the format expected by stats-tables.js with:
- Team groupings with win/loss indicators
- Personal performance (+/-) calculation
- Best-in-match identification
- All columns needed for Leetify-style display
"""

from dataclasses import dataclass, field
from typing import Any

from opensight.analysis.hltv_rating import (
    calculate_hltv_rating,
)


@dataclass
class PlayerStatsRow:
    """Player stats formatted for table display."""

    # Identity
    steam_id: str
    name: str
    team: str
    avatar_initial: str

    # Core stats
    kills: int = 0
    assists: int = 0
    deaths: int = 0
    kd: float = 0.0
    kd_diff: int = 0
    adr: float = 0.0

    # Multi-kills
    multi_2k: int = 0
    multi_3k: int = 0
    multi_4k: int = 0
    multi_5k: int = 0

    # Ratings
    hltv_rating: float = 0.0
    impact_rating: float = 0.0
    performance: float = 0.0  # Personal Performance (+/-)

    # Percentages
    hs_percentage: float = 0.0
    kast_percentage: float = 0.0

    # Advanced
    entry_win_rate: float = 0.0
    trade_rate: float = 0.0
    clutch_win_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "steamid": self.steam_id,
            "name": self.name,
            "team": self.team,
            "avatar_initial": self.avatar_initial,
            "kills": self.kills,
            "assists": self.assists,
            "deaths": self.deaths,
            "kd": self.kd,
            "kd_diff": self.kd_diff,
            "adr": self.adr,
            "multi_2k": self.multi_2k,
            "multi_3k": self.multi_3k,
            "multi_4k": self.multi_4k,
            "multi_5k": self.multi_5k,
            "hltv_rating": self.hltv_rating,
            "impact_rating": self.impact_rating,
            "performance": self.performance,
            "hs_percentage": self.hs_percentage,
            "kast_percentage": self.kast_percentage,
            "entry_win_rate": self.entry_win_rate,
            "trade_rate": self.trade_rate,
            "clutch_win_rate": self.clutch_win_rate,
        }


@dataclass
class TeamStatsGroup:
    """Team statistics with win/loss indicator."""

    team_name: str
    result: str  # "WIN" or "LOSS"
    score: int
    players: list[PlayerStatsRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "team_name": self.team_name,
            "result": self.result,
            "score": self.score,
            "players": [p.to_dict() for p in self.players],
        }


@dataclass
class EnhancedStatsData:
    """Complete stats data for enhanced tables."""

    my_team: TeamStatsGroup
    enemy_team: TeamStatsGroup
    best_values: dict[str, Any] = field(default_factory=dict)
    column_definitions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "my_team": self.my_team.to_dict(),
            "enemy_team": self.enemy_team.to_dict(),
            "best_values": self.best_values,
            "column_definitions": self.column_definitions,
        }


def get_column_definitions() -> list[dict[str, Any]]:
    """
    Get column definitions for the stats table.

    Returns column configuration including:
    - key: Data key to access value
    - label: Display header text
    - numeric: Whether the column should be sorted numerically
    - format: Formatting type (decimal, plusminus, percent, none)
    - sortable: Whether column is sortable
    - highlight: Whether to highlight best value
    """
    return [
        {
            "key": "name",
            "label": "Player",
            "numeric": False,
            "format": "none",
            "sortable": False,
            "highlight": False,
        },
        {
            "key": "kills",
            "label": "K",
            "numeric": True,
            "format": "none",
            "sortable": True,
            "highlight": True,
        },
        {
            "key": "assists",
            "label": "A",
            "numeric": True,
            "format": "none",
            "sortable": True,
            "highlight": False,
        },
        {
            "key": "deaths",
            "label": "D",
            "numeric": True,
            "format": "none",
            "sortable": True,
            "highlight": False,
            "invert": True,  # Lower is better
        },
        {
            "key": "kd",
            "label": "K/D",
            "numeric": True,
            "format": "decimal",
            "sortable": True,
            "highlight": True,
            "colorCode": True,  # Green if > 1, red if < 1
        },
        {
            "key": "adr",
            "label": "ADR",
            "numeric": True,
            "format": "decimal",
            "sortable": True,
            "highlight": True,
        },
        {
            "key": "multi_2k",
            "label": "2K",
            "numeric": True,
            "format": "none",
            "sortable": True,
            "highlight": False,
        },
        {
            "key": "multi_3k",
            "label": "3K",
            "numeric": True,
            "format": "none",
            "sortable": True,
            "highlight": True,
        },
        {
            "key": "multi_4k",
            "label": "4K",
            "numeric": True,
            "format": "none",
            "sortable": True,
            "highlight": True,
        },
        {
            "key": "multi_5k",
            "label": "5K",
            "numeric": True,
            "format": "none",
            "sortable": True,
            "highlight": True,
        },
        {
            "key": "hltv_rating",
            "label": "Rating",
            "numeric": True,
            "format": "decimal",
            "sortable": True,
            "highlight": True,
        },
        {
            "key": "performance",
            "label": "+/-",
            "numeric": True,
            "format": "plusminus",
            "sortable": True,
            "highlight": False,
            "colorCode": True,  # Green if positive, red if negative
        },
    ]


def calculate_personal_performance(player_stats: dict[str, Any]) -> float:
    """
    Calculate Personal Performance (+/-) for a player.

    This metric shows how much the player performed above or below
    their expected performance based on their HLTV rating.

    A simplified version: actual K/D diff vs expected based on rounds.

    Args:
        player_stats: Player statistics dictionary

    Returns:
        Personal performance value (positive = above expected, negative = below)
    """
    kills = player_stats.get("kills", 0)
    deaths = player_stats.get("deaths", 0)
    assists = player_stats.get("assists", 0)
    rounds_played = player_stats.get("rounds_played", 1)

    if rounds_played <= 0:
        rounds_played = 1

    # Expected kills per round for average player is ~0.7
    expected_kills = rounds_played * 0.7
    # Expected deaths per round is ~0.7
    expected_deaths = rounds_played * 0.7
    # Assists contribution (weighted at 0.3)
    assist_value = assists * 0.3

    # Performance = (actual - expected) contribution
    kill_diff = kills - expected_kills
    death_diff = expected_deaths - deaths  # Positive if you died less than expected

    performance = (kill_diff + death_diff * 0.5 + assist_value) / rounds_played * 10

    return round(performance, 2)


def extract_player_stats_row(player: Any, my_team_steam_ids: set[str]) -> PlayerStatsRow:
    """
    Extract a PlayerStatsRow from a player object.

    Handles both dictionary and object formats for player data.

    Args:
        player: Player data (dict or object with attributes)
        my_team_steam_ids: Set of steam IDs for "my team" identification

    Returns:
        PlayerStatsRow with extracted stats
    """
    # Helper to get value from dict or object
    def get_val(key: str, default: Any = 0) -> Any:
        if isinstance(player, dict):
            return player.get(key, default)
        return getattr(player, key, default)

    steam_id = str(get_val("steam_id", ""))
    name = get_val("name", "Unknown")

    # Core stats
    kills = int(get_val("kills", 0))
    deaths = int(get_val("deaths", 0))
    assists = int(get_val("assists", 0))
    rounds_played = int(get_val("rounds_played", 1)) or 1

    # Calculate K/D
    kd = round(kills / deaths, 2) if deaths > 0 else float(kills)
    kd_diff = kills - deaths

    # ADR
    total_damage = get_val("total_damage", 0)
    adr = round(total_damage / rounds_played, 1) if rounds_played > 0 else 0.0

    # Multi-kills - handle both nested and flat formats
    multi_kills = get_val("multi_kills", {})
    if isinstance(multi_kills, dict):
        multi_2k = multi_kills.get("rounds_with_2k", 0) or multi_kills.get("2k", 0)
        multi_3k = multi_kills.get("rounds_with_3k", 0) or multi_kills.get("3k", 0)
        multi_4k = multi_kills.get("rounds_with_4k", 0) or multi_kills.get("4k", 0)
        multi_5k = multi_kills.get("rounds_with_5k", 0) or multi_kills.get("5k", 0)
    else:
        multi_2k = get_val("rounds_with_2k", 0)
        multi_3k = get_val("rounds_with_3k", 0)
        multi_4k = get_val("rounds_with_4k", 0)
        multi_5k = get_val("rounds_with_5k", 0)

    # Percentages
    hs_percentage = float(get_val("hs_percentage", 0) or get_val("headshot_percentage", 0))
    kast_percentage = float(get_val("kast_percentage", 0))

    # Get or calculate HLTV rating
    hltv_rating = float(get_val("hltv_rating", 0))
    if hltv_rating == 0 and rounds_played > 0:
        # Calculate if not provided
        clutch_wins = 0
        clutches = get_val("clutches", {})
        if isinstance(clutches, dict):
            clutch_wins = clutches.get("total_wins", 0) or clutches.get("wins", 0)

        hltv_rating = calculate_hltv_rating(
            kills=kills,
            deaths=deaths,
            assists=assists,
            adr=adr,
            kast_pct=kast_percentage,
            rounds=rounds_played,
            clutch_wins=clutch_wins,
            multi_kill_2k=multi_2k,
            multi_kill_3k=multi_3k,
            multi_kill_4k=multi_4k,
            multi_kill_5k=multi_5k,
        )

    # Impact rating
    impact_rating = float(get_val("impact_rating", 0))

    # Calculate personal performance
    player_dict = player if isinstance(player, dict) else {
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "rounds_played": rounds_played,
    }
    performance = calculate_personal_performance(player_dict)

    # Advanced stats
    opening_duels = get_val("opening_duels", {})
    entry_win_rate = 0.0
    if isinstance(opening_duels, dict):
        entry_win_rate = float(opening_duels.get("win_rate", 0))

    trades = get_val("trades", {})
    trade_rate = 0.0
    if isinstance(trades, dict):
        trade_rate = float(trades.get("trade_rate", 0))

    clutches = get_val("clutches", {})
    clutch_win_rate = 0.0
    if isinstance(clutches, dict):
        clutch_win_rate = float(clutches.get("win_rate", 0))

    # Team identification
    team = get_val("team", "Unknown")

    return PlayerStatsRow(
        steam_id=steam_id,
        name=name,
        team=team,
        avatar_initial=name[0].upper() if name else "?",
        kills=kills,
        assists=assists,
        deaths=deaths,
        kd=kd,
        kd_diff=kd_diff,
        adr=adr,
        multi_2k=multi_2k,
        multi_3k=multi_3k,
        multi_4k=multi_4k,
        multi_5k=multi_5k,
        hltv_rating=hltv_rating,
        impact_rating=impact_rating,
        performance=performance,
        hs_percentage=hs_percentage,
        kast_percentage=kast_percentage,
        entry_win_rate=entry_win_rate,
        trade_rate=trade_rate,
        clutch_win_rate=clutch_win_rate,
    )


def calculate_best_values(all_players: list[PlayerStatsRow]) -> dict[str, Any]:
    """
    Calculate the best value for each numeric column across all players.

    Args:
        all_players: List of all player stats

    Returns:
        Dictionary mapping column key to best value
    """
    if not all_players:
        return {}

    columns = get_column_definitions()
    best = {}

    for col in columns:
        if col.get("numeric") and col.get("highlight"):
            key = col["key"]
            values = [getattr(p, key, 0) for p in all_players]
            if col.get("invert"):
                # Lower is better (e.g., deaths)
                best[key] = min(values) if values else 0
            else:
                best[key] = max(values) if values else 0

    return best


def extract_enhanced_stats(
    players: list[Any],
    my_team_steam_ids: list[str],
    my_team_score: int,
    enemy_team_score: int,
    my_team_name: str = "My Team",
    enemy_team_name: str = "Enemy Team",
) -> EnhancedStatsData:
    """
    Extract enhanced stats data for the stats tables.

    Args:
        players: List of player data (dict or objects)
        my_team_steam_ids: Steam IDs for the player's team
        my_team_score: Score for the player's team
        enemy_team_score: Score for the enemy team
        my_team_name: Display name for player's team
        enemy_team_name: Display name for enemy team

    Returns:
        EnhancedStatsData with all formatted data for the tables
    """
    my_team_set = {str(sid) for sid in my_team_steam_ids}

    # Extract all player rows
    all_rows = [extract_player_stats_row(p, my_team_set) for p in players]

    # Split by team
    my_team_players = [p for p in all_rows if p.steam_id in my_team_set]
    enemy_team_players = [p for p in all_rows if p.steam_id not in my_team_set]

    # Determine win/loss
    my_result = "WIN" if my_team_score > enemy_team_score else "LOSS"
    enemy_result = "WIN" if enemy_team_score > my_team_score else "LOSS"

    # Handle ties
    if my_team_score == enemy_team_score:
        my_result = "TIE"
        enemy_result = "TIE"

    # Calculate best values across ALL players
    best_values = calculate_best_values(all_rows)

    return EnhancedStatsData(
        my_team=TeamStatsGroup(
            team_name=my_team_name,
            result=my_result,
            score=my_team_score,
            players=my_team_players,
        ),
        enemy_team=TeamStatsGroup(
            team_name=enemy_team_name,
            result=enemy_result,
            score=enemy_team_score,
            players=enemy_team_players,
        ),
        best_values=best_values,
        column_definitions=get_column_definitions(),
    )
