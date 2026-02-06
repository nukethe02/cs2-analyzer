"""
OpenSight Match History Database System.

Provides persistent storage for match history, player statistics,
and performance tracking across multiple matches.

Uses SQLite (free, no external dependencies) with SQLAlchemy ORM
for clean database abstraction.

All features are 100% FREE - no paid services required.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(UTC)


logger = logging.getLogger(__name__)

# Database configuration
DEFAULT_DB_PATH = Path.home() / ".opensight" / "history.db"
Base = declarative_base()


# =============================================================================
# Database Models
# =============================================================================


class Match(Base):
    """Match metadata and summary."""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    demo_hash = Column(String(64), unique=True, nullable=False, index=True)
    map_name = Column(String(50), nullable=False, index=True)
    date_played = Column(DateTime, index=True)
    date_analyzed = Column(DateTime, default=_utc_now)

    # Score
    score_ct = Column(Integer, default=0)
    score_t = Column(Integer, default=0)
    total_rounds = Column(Integer, default=0)

    # Match info
    duration_seconds = Column(Float)
    tick_rate = Column(Integer)
    server_name = Column(String(200))

    # Demo file info
    demo_filename = Column(String(500))
    demo_size_mb = Column(Float)

    # Relationships
    player_matches = relationship(
        "PlayerMatch", back_populates="match", cascade="all, delete-orphan"
    )
    rounds = relationship("Round", back_populates="match", cascade="all, delete-orphan")

    # Indexes for common queries
    __table_args__ = (
        Index("idx_match_date_map", "date_played", "map_name"),
        Index("idx_match_analyzed", "date_analyzed"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "demo_hash": self.demo_hash,
            "map_name": self.map_name,
            "date_played": self.date_played.isoformat() if self.date_played else None,
            "date_analyzed": (self.date_analyzed.isoformat() if self.date_analyzed else None),
            "score": f"{self.score_ct}-{self.score_t}",
            "total_rounds": self.total_rounds,
            "duration_seconds": self.duration_seconds,
            "tick_rate": self.tick_rate,
        }


class PlayerMatch(Base):
    """Player performance in a specific match."""

    __tablename__ = "player_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)
    steam_id = Column(String(20), nullable=False, index=True)
    player_name = Column(String(100), nullable=False)
    team = Column(String(20))  # CT, T, or team name

    # Core stats
    kills = Column(Integer, default=0)
    deaths = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    headshots = Column(Integer, default=0)
    headshot_percentage = Column(Float, default=0.0)

    # Damage
    total_damage = Column(Integer, default=0)
    adr = Column(Float, default=0.0)

    # Ratings
    hltv_rating = Column(Float, default=1.0, index=True)
    kast_percentage = Column(Float, default=0.0)

    # Advanced metrics
    ttd_mean_ms = Column(Float)
    ttd_median_ms = Column(Float)
    cp_mean_deg = Column(Float)
    cp_median_deg = Column(Float)

    # Utility stats
    flashbangs_thrown = Column(Integer, default=0)
    smokes_thrown = Column(Integer, default=0)
    he_thrown = Column(Integer, default=0)
    molotovs_thrown = Column(Integer, default=0)
    enemies_flashed = Column(Integer, default=0)
    flash_assists = Column(Integer, default=0)

    # Economy
    total_spent = Column(Integer, default=0)
    avg_equipment_value = Column(Float, default=0.0)

    # Duels
    opening_kills = Column(Integer, default=0)
    opening_deaths = Column(Integer, default=0)
    opening_attempts = Column(Integer, default=0)
    opening_win_rate = Column(Float, default=0.0)

    # Clutches
    clutch_situations = Column(Integer, default=0)
    clutch_wins = Column(Integer, default=0)
    clutch_win_rate = Column(Float, default=0.0)

    # Trades
    kills_traded = Column(Integer, default=0)
    deaths_traded = Column(Integer, default=0)
    trade_kill_rate = Column(Float, default=0.0)

    # Round participation
    rounds_played = Column(Integer, default=0)
    rounds_with_kill = Column(Integer, default=0)
    rounds_with_damage = Column(Integer, default=0)

    # Match result for this player
    won_match = Column(Boolean)

    # Relationships
    match = relationship("Match", back_populates="player_matches")

    # Indexes
    __table_args__ = (
        Index("idx_player_steam_rating", "steam_id", "hltv_rating"),
        Index("idx_player_match_team", "match_id", "team"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "steam_id": self.steam_id,
            "player_name": self.player_name,
            "team": self.team,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "headshot_percentage": round(self.headshot_percentage, 1),
            "adr": round(self.adr, 1),
            "hltv_rating": round(self.hltv_rating, 2),
            "kast_percentage": round(self.kast_percentage, 1),
            "ttd_median_ms": (round(self.ttd_median_ms, 1) if self.ttd_median_ms else None),
            "cp_median_deg": (round(self.cp_median_deg, 1) if self.cp_median_deg else None),
            "opening_kills": self.opening_kills,
            "opening_deaths": self.opening_deaths,
            "clutch_wins": self.clutch_wins,
            "clutch_situations": self.clutch_situations,
            "won_match": self.won_match,
        }


class Round(Base):
    """Round-by-round data for detailed analysis."""

    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)
    round_num = Column(Integer, nullable=False)

    # Winner
    winner = Column(String(20))  # CT, T
    win_type = Column(String(30))  # bomb_defused, elimination, time, bomb_exploded

    # First kill info
    first_kill_steam_id = Column(String(20))
    first_death_steam_id = Column(String(20))
    first_kill_weapon = Column(String(50))

    # Economy
    ct_equipment_value = Column(Integer)
    t_equipment_value = Column(Integer)
    ct_buy_type = Column(String(20))  # full, eco, force, etc.
    t_buy_type = Column(String(20))

    # Duration
    duration_seconds = Column(Float)

    # Relationships
    match = relationship("Match", back_populates="rounds")

    __table_args__ = (Index("idx_round_match_num", "match_id", "round_num"),)


class MatchHistory(Base):
    """Individual match history for 'Your Match' personal performance tracking."""

    __tablename__ = "match_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    steam_id = Column(String(20), nullable=False, index=True)
    demo_hash = Column(String(64), nullable=False, index=True)
    analyzed_at = Column(DateTime, default=_utc_now)

    # Match metadata
    map_name = Column(String(50))
    result = Column(String(10))  # 'win', 'loss', 'draw'

    # Core stats
    kills = Column(Integer, default=0)
    deaths = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    adr = Column(Float, default=0.0)
    kast = Column(Float, default=0.0)
    hs_pct = Column(Float, default=0.0)

    # Ratings (0.0 = missing data, actual values 1-100)
    aim_rating = Column(Float, default=0.0)
    utility_rating = Column(Float, default=0.0)
    hltv_rating = Column(Float, default=1.0)
    opensight_rating = Column(Float, default=50.0)

    # Trade stats
    trade_kill_opportunities = Column(Integer, default=0)
    trade_kill_attempts = Column(Integer, default=0)
    trade_kill_success = Column(Integer, default=0)

    # Entry stats
    entry_attempts = Column(Integer, default=0)
    entry_success = Column(Integer, default=0)

    # Clutch stats
    clutch_situations = Column(Integer, default=0)
    clutch_wins = Column(Integer, default=0)
    clutch_kills = Column(Integer, default=0)

    # Utility stats
    he_damage = Column(Integer, default=0)
    enemies_flashed = Column(Integer, default=0)
    flash_assists = Column(Integer, default=0)

    # Advanced metrics
    ttd_median_ms = Column(Float)
    cp_median_deg = Column(Float)

    # Rounds
    rounds_played = Column(Integer, default=0)

    __table_args__ = (
        Index("idx_match_history_steam_analyzed", "steam_id", "analyzed_at"),
        # SQLite doesn't support UNIQUE in table_args the same way, use UniqueConstraint
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "steam_id": self.steam_id,
            "demo_hash": self.demo_hash,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "map_name": self.map_name,
            "result": self.result,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "adr": round(self.adr, 1) if self.adr else 0,
            "kast": round(self.kast, 1) if self.kast else 0,
            "hs_pct": round(self.hs_pct, 1) if self.hs_pct else 0,
            # aim_rating: 0 = missing data, actual values 1-100
            "aim_rating": round(self.aim_rating, 1) if self.aim_rating is not None else 0,
            "utility_rating": round(self.utility_rating, 1)
            if self.utility_rating is not None
            else 0,
            "hltv_rating": round(self.hltv_rating, 2) if self.hltv_rating else 1.0,
            "opensight_rating": round(self.opensight_rating, 1) if self.opensight_rating else 50,
            "trade_kill_opportunities": self.trade_kill_opportunities,
            "clutch_kills": self.clutch_kills,
            "entry_attempts": self.entry_attempts,
            "entry_success": self.entry_success,
        }


class PlayerBaseline(Base):
    """Rolling baselines for player metrics comparison."""

    __tablename__ = "player_baselines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    steam_id = Column(String(20), nullable=False, index=True)
    metric = Column(String(50), nullable=False)

    # Statistical values
    avg_value = Column(Float, default=0.0)
    std_value = Column(Float, default=0.0)
    min_value = Column(Float)
    max_value = Column(Float)

    # Sample tracking
    sample_count = Column(Integer, default=0)
    window_size = Column(Integer, default=30)

    # Last update timestamp
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now)

    __table_args__ = (Index("idx_player_baselines_steam_metric", "steam_id", "metric"),)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "steam_id": self.steam_id,
            "metric": self.metric,
            "avg": round(self.avg_value, 2) if self.avg_value else 0,
            "std": round(self.std_value, 2) if self.std_value else 0,
            "min": round(self.min_value, 2) if self.min_value else None,
            "max": round(self.max_value, 2) if self.max_value else None,
            "sample_count": self.sample_count,
        }


class PlayerPersona(Base):
    """Player persona tracking for Match Identity feature."""

    __tablename__ = "player_personas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    steam_id = Column(String(20), nullable=False, unique=True, index=True)

    # Current persona
    current_persona = Column(String(50), default="the_competitor")
    persona_confidence = Column(Float, default=0.0)

    # Persona history (JSON array)
    persona_history_json = Column(Text)

    # Top traits
    primary_trait = Column(String(50))
    secondary_trait = Column(String(50))

    # Calculation timestamp
    calculated_at = Column(DateTime, default=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        history = json.loads(self.persona_history_json) if self.persona_history_json else []
        return {
            "steam_id": self.steam_id,
            "persona": self.current_persona,
            "confidence": round(self.persona_confidence, 2) if self.persona_confidence else 0,
            "primary_trait": self.primary_trait,
            "secondary_trait": self.secondary_trait,
            "history": history[-5:],  # Last 5 personas
        }


class PlayerProfile(Base):
    """Aggregated player statistics across all matches."""

    __tablename__ = "player_profiles"

    steam_id = Column(String(20), primary_key=True)
    player_name = Column(String(100), nullable=False)
    last_updated = Column(DateTime, default=_utc_now, onupdate=_utc_now)

    # Career totals
    total_matches = Column(Integer, default=0)
    total_wins = Column(Integer, default=0)
    total_losses = Column(Integer, default=0)
    total_kills = Column(Integer, default=0)
    total_deaths = Column(Integer, default=0)
    total_assists = Column(Integer, default=0)
    total_headshots = Column(Integer, default=0)
    total_damage = Column(Integer, default=0)
    total_rounds_played = Column(Integer, default=0)

    # Average metrics (updated on each match)
    avg_kills = Column(Float, default=0.0)
    avg_deaths = Column(Float, default=0.0)
    avg_adr = Column(Float, default=0.0)
    avg_rating = Column(Float, default=1.0)
    avg_kast = Column(Float, default=0.0)
    avg_hs_percentage = Column(Float, default=0.0)

    # Best performances (career highs)
    best_rating = Column(Float, default=0.0)
    best_kills = Column(Integer, default=0)
    best_adr = Column(Float, default=0.0)
    best_kast = Column(Float, default=0.0)

    # Clutch/duel stats
    total_clutch_situations = Column(Integer, default=0)
    total_clutch_wins = Column(Integer, default=0)
    total_opening_duels = Column(Integer, default=0)
    total_opening_kills = Column(Integer, default=0)

    # Map performance (JSON blob)
    map_stats_json = Column(Text)  # {"de_dust2": {"matches": 5, "wins": 3, ...}}

    # Recent form (last N matches rating)
    recent_form_json = Column(Text)  # [1.2, 0.9, 1.1, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        map_stats = json.loads(self.map_stats_json) if self.map_stats_json else {}
        recent_form = json.loads(self.recent_form_json) if self.recent_form_json else []

        return {
            "steam_id": self.steam_id,
            "player_name": self.player_name,
            "last_updated": (self.last_updated.isoformat() if self.last_updated else None),
            "career": {
                "matches": self.total_matches,
                "wins": self.total_wins,
                "losses": self.total_losses,
                "win_rate": (round(self.total_wins / max(self.total_matches, 1) * 100, 1)),
                "kills": self.total_kills,
                "deaths": self.total_deaths,
                "kd_ratio": round(self.total_kills / max(self.total_deaths, 1), 2),
                "headshots": self.total_headshots,
                "total_damage": self.total_damage,
                "rounds_played": self.total_rounds_played,
            },
            "averages": {
                "kills": round(self.avg_kills, 1),
                "deaths": round(self.avg_deaths, 1),
                "adr": round(self.avg_adr, 1),
                "rating": round(self.avg_rating, 2),
                "kast": round(self.avg_kast, 1),
                "hs_percentage": round(self.avg_hs_percentage, 1),
            },
            "career_highs": {
                "best_rating": round(self.best_rating, 2),
                "best_kills": self.best_kills,
                "best_adr": round(self.best_adr, 1),
                "best_kast": round(self.best_kast, 1),
            },
            "clutches": {
                "situations": self.total_clutch_situations,
                "wins": self.total_clutch_wins,
                "win_rate": round(
                    self.total_clutch_wins / max(self.total_clutch_situations, 1) * 100,
                    1,
                ),
            },
            "opening_duels": {
                "total": self.total_opening_duels,
                "wins": self.total_opening_kills,
                "win_rate": round(
                    self.total_opening_kills / max(self.total_opening_duels, 1) * 100, 1
                ),
            },
            "map_stats": map_stats,
            "recent_form": recent_form[-10:],  # Last 10 matches
        }


class Kill(Base):
    """Individual kill event storage for detailed analysis."""

    __tablename__ = "kills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    tick = Column(Integer)
    round_num = Column(Integer, index=True)

    attacker_steam_id = Column(String(20), index=True)
    attacker_name = Column(String(100))
    attacker_team = Column(String(20))
    attacker_x = Column(Float)
    attacker_y = Column(Float)
    attacker_z = Column(Float)

    victim_steam_id = Column(String(20), index=True)
    victim_name = Column(String(100))
    victim_team = Column(String(20))
    victim_x = Column(Float)
    victim_y = Column(Float)
    victim_z = Column(Float)

    weapon = Column(String(50))
    headshot = Column(Boolean, default=False)
    penetrated = Column(Boolean, default=False)
    noscope = Column(Boolean, default=False)
    thrusmoke = Column(Boolean, default=False)
    attackerblind = Column(Boolean, default=False)
    distance = Column(Float)

    assister_steam_id = Column(String(20))
    assister_name = Column(String(100))

    __table_args__ = (
        Index("idx_kill_match_round", "match_id", "round_num"),
        Index("idx_kill_attacker", "attacker_steam_id"),
        Index("idx_kill_victim", "victim_steam_id"),
    )


class DamageEvent(Base):
    """Damage event storage for TTD and engagement analysis."""

    __tablename__ = "damage_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    tick = Column(Integer)
    round_num = Column(Integer, index=True)

    attacker_steam_id = Column(String(20), index=True)
    victim_steam_id = Column(String(20), index=True)

    dmg_health = Column(Integer, default=0)
    dmg_armor = Column(Integer, default=0)
    hitgroup = Column(String(20))
    weapon = Column(String(50))

    attacker_x = Column(Float)
    attacker_y = Column(Float)
    attacker_z = Column(Float)

    victim_x = Column(Float)
    victim_y = Column(Float)
    victim_z = Column(Float)

    __table_args__ = (
        Index("idx_damage_match_round", "match_id", "round_num"),
        Index("idx_damage_attacker", "attacker_steam_id"),
    )


class GrenadeEvent(Base):
    """Grenade throw/detonate events for utility analysis."""

    __tablename__ = "grenade_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    tick = Column(Integer)
    round_num = Column(Integer, index=True)

    thrower_steam_id = Column(String(20), index=True)
    thrower_name = Column(String(100))

    grenade_type = Column(String(20))
    event_type = Column(String(20))

    x = Column(Float)
    y = Column(Float)
    z = Column(Float)

    __table_args__ = (
        Index("idx_grenade_match_round", "match_id", "round_num"),
        Index("idx_grenade_thrower", "thrower_steam_id"),
    )


class BombEvent(Base):
    """Bomb plant/defuse/explode events."""

    __tablename__ = "bomb_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    tick = Column(Integer)
    round_num = Column(Integer, index=True)

    player_steam_id = Column(String(20), index=True)
    player_name = Column(String(100))

    event_type = Column(String(20))

    site = Column(String(10))

    x = Column(Float)
    y = Column(Float)
    z = Column(Float)

    __table_args__ = (
        Index("idx_bomb_match_round", "match_id", "round_num"),
        Index("idx_bomb_player", "player_steam_id"),
    )


class Job(Base):
    """Persistent job tracking for async analysis."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    filename = Column(String(500), nullable=False)
    file_size = Column(Integer)

    status = Column(String(20), default="queued", index=True)

    created_at = Column(DateTime, default=_utc_now)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    result_json = Column(Text)
    error_message = Column(Text)

    demo_hash = Column(String(64), index=True)

    __table_args__ = (
        Index("idx_job_status", "status"),
        Index("idx_job_created", "created_at"),
    )


# =============================================================================
# Database Manager
# =============================================================================


class DatabaseManager:
    """
    Manages database connections and operations.

    All operations are FREE - uses SQLite with no external dependencies.
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize database connection."""
        if db_path is None:
            db_path = os.environ.get("OPENSIGHT_DB_PATH", DEFAULT_DB_PATH)

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with SQLite
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        logger.info(f"Database initialized at: {self.db_path}")

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    # =========================================================================
    # Match Operations
    # =========================================================================

    def save_match(
        self,
        demo_hash: str,
        map_name: str,
        player_stats: list[dict],
        match_info: dict | None = None,
        round_data: list[dict] | None = None,
    ) -> Match | None:
        """
        Save a match and all player statistics.

        Args:
            demo_hash: SHA256 hash of the demo file (for deduplication)
            map_name: Map name
            player_stats: List of player statistics dictionaries
            match_info: Optional match metadata
            round_data: Optional round-by-round data

        Returns:
            Match object if saved, None if already exists
        """
        session = self.get_session()
        try:
            # Check for duplicate
            existing = session.query(Match).filter(Match.demo_hash == demo_hash).first()
            if existing:
                logger.info(f"Match already exists: {demo_hash[:16]}...")
                return None

            # Create match record
            match_info = match_info or {}
            match = Match(
                demo_hash=demo_hash,
                map_name=map_name,
                date_played=match_info.get("date_played"),
                score_ct=match_info.get("score_ct", 0),
                score_t=match_info.get("score_t", 0),
                total_rounds=match_info.get("total_rounds", 0),
                duration_seconds=match_info.get("duration_seconds"),
                tick_rate=match_info.get("tick_rate"),
                server_name=match_info.get("server_name"),
                demo_filename=match_info.get("demo_filename"),
                demo_size_mb=match_info.get("demo_size_mb"),
            )
            session.add(match)
            session.flush()  # Get match ID

            # Determine winner
            ct_won = match.score_ct > match.score_t

            # Save player stats
            for ps in player_stats:
                player_won = None
                if ps.get("team"):
                    if ps["team"].upper() == "CT":
                        player_won = ct_won
                    elif ps["team"].upper() in ("T", "TERRORIST"):
                        player_won = not ct_won

                pm = PlayerMatch(
                    match_id=match.id,
                    steam_id=str(ps.get("steam_id", "")),
                    player_name=ps.get("name", "Unknown"),
                    team=ps.get("team"),
                    kills=ps.get("kills", 0),
                    deaths=ps.get("deaths", 0),
                    assists=ps.get("assists", 0),
                    headshots=ps.get("headshots", 0),
                    headshot_percentage=ps.get("headshot_percentage", 0),
                    total_damage=ps.get("total_damage", 0),
                    adr=ps.get("adr", 0),
                    hltv_rating=ps.get("hltv_rating", 1.0),
                    kast_percentage=ps.get("kast_percentage", 0),
                    ttd_mean_ms=ps.get("ttd_mean_ms"),
                    ttd_median_ms=ps.get("ttd_median_ms"),
                    cp_mean_deg=ps.get("cp_mean_deg"),
                    cp_median_deg=ps.get("cp_median_deg"),
                    flashbangs_thrown=ps.get("flashbangs_thrown", 0),
                    smokes_thrown=ps.get("smokes_thrown", 0),
                    he_thrown=ps.get("he_thrown", 0),
                    molotovs_thrown=ps.get("molotovs_thrown", 0),
                    enemies_flashed=ps.get("enemies_flashed", 0),
                    opening_kills=ps.get("opening_kills", 0),
                    opening_deaths=ps.get("opening_deaths", 0),
                    opening_attempts=ps.get("opening_attempts", 0),
                    opening_win_rate=ps.get("opening_win_rate", 0),
                    clutch_situations=ps.get("clutch_situations", 0),
                    clutch_wins=ps.get("clutch_wins", 0),
                    clutch_win_rate=ps.get("clutch_win_rate", 0),
                    kills_traded=ps.get("kills_traded", 0),
                    deaths_traded=ps.get("deaths_traded", 0),
                    rounds_played=ps.get("rounds_played", 0),
                    won_match=player_won,
                )
                session.add(pm)

                # Update player profile
                self._update_player_profile(session, pm, map_name)

            # Save round data if provided
            if round_data:
                for rd in round_data:
                    round_obj = Round(
                        match_id=match.id,
                        round_num=rd.get("round_num", 0),
                        winner=rd.get("winner"),
                        win_type=rd.get("win_type"),
                        first_kill_steam_id=rd.get("first_kill_steam_id"),
                        first_death_steam_id=rd.get("first_death_steam_id"),
                        ct_equipment_value=rd.get("ct_equipment_value"),
                        t_equipment_value=rd.get("t_equipment_value"),
                        ct_buy_type=rd.get("ct_buy_type"),
                        t_buy_type=rd.get("t_buy_type"),
                        duration_seconds=rd.get("duration_seconds"),
                    )
                    session.add(round_obj)

            session.commit()
            logger.info(f"Saved match {demo_hash[:16]}... with {len(player_stats)} players")
            return match

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save match: {e}")
            raise
        finally:
            session.close()

    def _update_player_profile(self, session: Session, pm: PlayerMatch, map_name: str) -> None:
        """Update or create player profile with new match data."""
        profile = session.query(PlayerProfile).filter(PlayerProfile.steam_id == pm.steam_id).first()

        if not profile:
            profile = PlayerProfile(
                steam_id=pm.steam_id,
                player_name=pm.player_name,
            )
            session.add(profile)

        # Update totals
        profile.player_name = pm.player_name  # Use latest name
        profile.total_matches += 1
        if pm.won_match is True:
            profile.total_wins += 1
        elif pm.won_match is False:
            profile.total_losses += 1
        profile.total_kills += pm.kills
        profile.total_deaths += pm.deaths
        profile.total_assists += pm.assists
        profile.total_headshots += pm.headshots
        profile.total_damage += pm.total_damage
        profile.total_rounds_played += pm.rounds_played
        profile.total_clutch_situations += pm.clutch_situations
        profile.total_clutch_wins += pm.clutch_wins
        profile.total_opening_duels += pm.opening_attempts
        profile.total_opening_kills += pm.opening_kills

        # Update averages
        n = profile.total_matches
        profile.avg_kills = profile.total_kills / n
        profile.avg_deaths = profile.total_deaths / n
        profile.avg_adr = profile.total_damage / max(profile.total_rounds_played, 1)
        # Running average for rating
        profile.avg_rating = (profile.avg_rating * (n - 1) + pm.hltv_rating) / n
        profile.avg_kast = (profile.avg_kast * (n - 1) + pm.kast_percentage) / n
        profile.avg_hs_percentage = (
            profile.avg_hs_percentage * (n - 1) + pm.headshot_percentage
        ) / n

        # Update career highs
        if pm.hltv_rating > profile.best_rating:
            profile.best_rating = pm.hltv_rating
        if pm.kills > profile.best_kills:
            profile.best_kills = pm.kills
        if pm.adr > profile.best_adr:
            profile.best_adr = pm.adr
        if pm.kast_percentage > profile.best_kast:
            profile.best_kast = pm.kast_percentage

        # Update map stats
        map_stats = json.loads(profile.map_stats_json) if profile.map_stats_json else {}
        if map_name not in map_stats:
            map_stats[map_name] = {"matches": 0, "wins": 0, "avg_rating": 0}
        map_stats[map_name]["matches"] += 1
        if pm.won_match:
            map_stats[map_name]["wins"] += 1
        old_avg = map_stats[map_name]["avg_rating"]
        old_n = map_stats[map_name]["matches"] - 1
        map_stats[map_name]["avg_rating"] = (old_avg * old_n + pm.hltv_rating) / map_stats[
            map_name
        ]["matches"]
        profile.map_stats_json = json.dumps(map_stats)

        # Update recent form
        recent_form = json.loads(profile.recent_form_json) if profile.recent_form_json else []
        recent_form.append(round(pm.hltv_rating, 2))
        if len(recent_form) > 20:  # Keep last 20
            recent_form = recent_form[-20:]
        profile.recent_form_json = json.dumps(recent_form)

    def get_match_by_hash(self, demo_hash: str) -> Match | None:
        """Get a match by its demo hash."""
        session = self.get_session()
        try:
            return session.query(Match).filter(Match.demo_hash == demo_hash).first()
        finally:
            session.close()

    def get_match_history(
        self,
        limit: int = 50,
        offset: int = 0,
        map_name: str | None = None,
        steam_id: str | None = None,
    ) -> list[dict]:
        """
        Get match history with optional filters.

        Args:
            limit: Maximum matches to return
            offset: Pagination offset
            map_name: Filter by map
            steam_id: Filter by player Steam ID

        Returns:
            List of match dictionaries
        """
        session = self.get_session()
        try:
            query = session.query(Match)

            if map_name:
                query = query.filter(Match.map_name == map_name)

            if steam_id:
                query = query.join(PlayerMatch).filter(PlayerMatch.steam_id == steam_id)

            matches = query.order_by(Match.date_analyzed.desc()).offset(offset).limit(limit).all()

            result = []
            for m in matches:
                match_dict = m.to_dict()
                # Include player count
                match_dict["player_count"] = len(m.player_matches)
                result.append(match_dict)

            return result
        finally:
            session.close()

    def get_match_details(self, match_id: int) -> dict | None:
        """Get full match details including all player stats."""
        session = self.get_session()
        try:
            match = session.query(Match).filter(Match.id == match_id).first()
            if not match:
                return None

            return {
                "match": match.to_dict(),
                "players": [pm.to_dict() for pm in match.player_matches],
                "rounds": [
                    {
                        "round_num": r.round_num,
                        "winner": r.winner,
                        "win_type": r.win_type,
                        "ct_equipment": r.ct_equipment_value,
                        "t_equipment": r.t_equipment_value,
                    }
                    for r in sorted(match.rounds, key=lambda x: x.round_num)
                ],
            }
        finally:
            session.close()

    # =========================================================================
    # Player Profile Operations
    # =========================================================================

    def get_player_profile(self, steam_id: str) -> dict | None:
        """Get a player's career profile."""
        session = self.get_session()
        try:
            profile = (
                session.query(PlayerProfile).filter(PlayerProfile.steam_id == steam_id).first()
            )
            return profile.to_dict() if profile else None
        finally:
            session.close()

    def get_player_match_history(self, steam_id: str, limit: int = 20) -> list[dict]:
        """Get a player's recent matches."""
        session = self.get_session()
        try:
            player_matches = (
                session.query(PlayerMatch)
                .join(Match)
                .filter(PlayerMatch.steam_id == steam_id)
                .order_by(Match.date_analyzed.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "match_id": pm.match_id,
                    "map_name": pm.match.map_name,
                    "date": (
                        pm.match.date_analyzed.isoformat() if pm.match.date_analyzed else None
                    ),
                    "score": f"{pm.match.score_ct}-{pm.match.score_t}",
                    "team": pm.team,
                    "kills": pm.kills,
                    "deaths": pm.deaths,
                    "rating": round(pm.hltv_rating, 2),
                    "adr": round(pm.adr, 1),
                    "won": pm.won_match,
                }
                for pm in player_matches
            ]
        finally:
            session.close()

    def search_players(self, name_query: str, limit: int = 10) -> list[dict]:
        """Search for players by name."""
        session = self.get_session()
        try:
            profiles = (
                session.query(PlayerProfile)
                .filter(PlayerProfile.player_name.ilike(f"%{name_query}%"))
                .order_by(PlayerProfile.total_matches.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "steam_id": p.steam_id,
                    "name": p.player_name,
                    "matches": p.total_matches,
                    "avg_rating": round(p.avg_rating, 2),
                }
                for p in profiles
            ]
        finally:
            session.close()

    def get_leaderboard(
        self,
        metric: str = "avg_rating",
        min_matches: int = 5,
        limit: int = 20,
    ) -> list[dict]:
        """Get player leaderboard by metric."""
        session = self.get_session()
        try:
            valid_metrics = {
                "avg_rating": PlayerProfile.avg_rating,
                "avg_adr": PlayerProfile.avg_adr,
                "avg_kast": PlayerProfile.avg_kast,
                "total_kills": PlayerProfile.total_kills,
                "total_matches": PlayerProfile.total_matches,
            }

            order_col = valid_metrics.get(metric, PlayerProfile.avg_rating)

            profiles = (
                session.query(PlayerProfile)
                .filter(PlayerProfile.total_matches >= min_matches)
                .order_by(order_col.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "rank": i + 1,
                    "steam_id": p.steam_id,
                    "name": p.player_name,
                    "matches": p.total_matches,
                    "value": round(getattr(p, metric, 0), 2),
                }
                for i, p in enumerate(profiles)
            ]
        finally:
            session.close()

    # =========================================================================
    # Your Match - Match History Operations
    # =========================================================================

    def save_match_history_entry(
        self,
        steam_id: str,
        demo_hash: str,
        player_stats: dict[str, Any],
        map_name: str | None = None,
        result: str | None = None,
    ) -> MatchHistory | None:
        """
        Save a match history entry for a player.

        Args:
            steam_id: Player's Steam ID (17 digits)
            demo_hash: SHA256 hash of the demo file
            player_stats: Dictionary of player statistics
            map_name: Map name
            result: 'win', 'loss', or 'draw'

        Returns:
            MatchHistory object if saved, None if duplicate
        """
        session = self.get_session()
        try:
            # Check for duplicate
            existing = (
                session.query(MatchHistory)
                .filter(MatchHistory.steam_id == steam_id, MatchHistory.demo_hash == demo_hash)
                .first()
            )
            if existing:
                logger.debug(f"Match history already exists for {steam_id[:8]}...")
                return None

            entry = MatchHistory(
                steam_id=steam_id,
                demo_hash=demo_hash,
                map_name=map_name,
                result=result,
                kills=player_stats.get("kills", 0),
                deaths=player_stats.get("deaths", 0),
                assists=player_stats.get("assists", 0),
                adr=player_stats.get("adr", 0.0),
                kast=player_stats.get("kast_percentage", player_stats.get("kast", 0.0)),
                hs_pct=player_stats.get("headshot_percentage", player_stats.get("hs_pct", 0.0)),
                # 0.0 = missing data, actual values 1-100
                aim_rating=player_stats.get("aim_rating", 0.0),
                utility_rating=player_stats.get("utility_rating", 0.0),
                hltv_rating=player_stats.get("hltv_rating", 1.0),
                opensight_rating=player_stats.get("opensight_rating", 50.0),
                trade_kill_opportunities=player_stats.get("trade_kill_opportunities", 0),
                trade_kill_attempts=player_stats.get("trade_kill_attempts", 0),
                trade_kill_success=player_stats.get(
                    "kills_traded", player_stats.get("trade_kill_success", 0)
                ),
                entry_attempts=player_stats.get(
                    "opening_duel_attempts", player_stats.get("entry_attempts", 0)
                ),
                entry_success=player_stats.get(
                    "opening_duel_wins", player_stats.get("entry_success", 0)
                ),
                clutch_situations=player_stats.get("clutch_situations", 0),
                clutch_wins=player_stats.get("clutch_wins", 0),
                clutch_kills=player_stats.get("clutch_kills", 0),
                he_damage=player_stats.get("he_damage", 0),
                enemies_flashed=player_stats.get("enemies_flashed", 0),
                flash_assists=player_stats.get("flash_assists", 0),
                ttd_median_ms=player_stats.get("ttd_median_ms"),
                cp_median_deg=player_stats.get(
                    "cp_median_error_deg", player_stats.get("cp_median_deg")
                ),
                rounds_played=player_stats.get("rounds_played", 0),
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            logger.info(f"Saved match history for {steam_id[:8]}...")
            return entry
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save match history: {e}")
            raise
        finally:
            session.close()

    def get_player_history(self, steam_id: str, limit: int = 30) -> list[dict]:
        """Get a player's match history for baseline calculation."""
        session = self.get_session()
        try:
            entries = (
                session.query(MatchHistory)
                .filter(MatchHistory.steam_id == steam_id)
                .order_by(MatchHistory.analyzed_at.desc())
                .limit(limit)
                .all()
            )
            return [e.to_dict() for e in entries]
        finally:
            session.close()

    def get_player_baselines(self, steam_id: str) -> dict[str, dict]:
        """Get all baselines for a player."""
        session = self.get_session()
        try:
            baselines = (
                session.query(PlayerBaseline).filter(PlayerBaseline.steam_id == steam_id).all()
            )
            return {b.metric: b.to_dict() for b in baselines}
        finally:
            session.close()

    def update_player_baselines(self, steam_id: str, window_size: int = 30) -> dict[str, dict]:
        """
        Recalculate and update baselines for a player from their match history.

        Args:
            steam_id: Player's Steam ID
            window_size: Number of recent matches to include (default 30)

        Returns:
            Updated baselines dictionary
        """
        import statistics

        session = self.get_session()
        try:
            # Get recent matches
            history = (
                session.query(MatchHistory)
                .filter(MatchHistory.steam_id == steam_id)
                .order_by(MatchHistory.analyzed_at.desc())
                .limit(window_size)
                .all()
            )

            if not history:
                return {}

            # Metrics to track
            metrics_config = {
                "kills": lambda h: h.kills,
                "deaths": lambda h: h.deaths,
                "adr": lambda h: h.adr,
                "kast": lambda h: h.kast,
                "hs_pct": lambda h: h.hs_pct,
                "aim_rating": lambda h: h.aim_rating,
                "utility_rating": lambda h: h.utility_rating,
                "hltv_rating": lambda h: h.hltv_rating,
                "opensight_rating": lambda h: h.opensight_rating,
                "trade_kill_opportunities": lambda h: h.trade_kill_opportunities,
                "trade_kill_success": lambda h: h.trade_kill_success,
                "entry_attempts": lambda h: h.entry_attempts,
                "entry_success": lambda h: h.entry_success,
                "clutch_situations": lambda h: h.clutch_situations,
                "clutch_wins": lambda h: h.clutch_wins,
                "clutch_kills": lambda h: h.clutch_kills,
                "he_damage": lambda h: h.he_damage,
                "enemies_flashed": lambda h: h.enemies_flashed,
                "flash_assists": lambda h: h.flash_assists,
                "ttd_median_ms": lambda h: h.ttd_median_ms,
                "cp_median_deg": lambda h: h.cp_median_deg,
            }

            updated_baselines = {}

            for metric_name, extractor in metrics_config.items():
                values = [extractor(h) for h in history if extractor(h) is not None]

                if not values:
                    continue

                avg_val = sum(values) / len(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                min_val = min(values)
                max_val = max(values)

                # Upsert baseline
                baseline = (
                    session.query(PlayerBaseline)
                    .filter(
                        PlayerBaseline.steam_id == steam_id,
                        PlayerBaseline.metric == metric_name,
                    )
                    .first()
                )

                if baseline:
                    baseline.avg_value = avg_val
                    baseline.std_value = std_val
                    baseline.min_value = min_val
                    baseline.max_value = max_val
                    baseline.sample_count = len(values)
                    baseline.window_size = window_size
                else:
                    baseline = PlayerBaseline(
                        steam_id=steam_id,
                        metric=metric_name,
                        avg_value=avg_val,
                        std_value=std_val,
                        min_value=min_val,
                        max_value=max_val,
                        sample_count=len(values),
                        window_size=window_size,
                    )
                    session.add(baseline)

                updated_baselines[metric_name] = {
                    "avg": round(avg_val, 2),
                    "std": round(std_val, 2),
                    "min": round(min_val, 2),
                    "max": round(max_val, 2),
                    "sample_count": len(values),
                }

            session.commit()
            logger.info(f"Updated {len(updated_baselines)} baselines for {steam_id[:8]}...")
            return updated_baselines
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update baselines: {e}")
            raise
        finally:
            session.close()

    def get_player_persona(self, steam_id: str) -> dict | None:
        """Get a player's current persona."""
        session = self.get_session()
        try:
            persona = (
                session.query(PlayerPersona).filter(PlayerPersona.steam_id == steam_id).first()
            )
            return persona.to_dict() if persona else None
        finally:
            session.close()

    def update_player_persona(
        self,
        steam_id: str,
        persona_id: str,
        confidence: float,
        primary_trait: str | None = None,
        secondary_trait: str | None = None,
    ) -> PlayerPersona:
        """Update or create a player's persona."""
        session = self.get_session()
        try:
            persona = (
                session.query(PlayerPersona).filter(PlayerPersona.steam_id == steam_id).first()
            )

            if persona:
                # Update existing
                old_history = (
                    json.loads(persona.persona_history_json) if persona.persona_history_json else []
                )
                old_history.append(persona.current_persona)
                if len(old_history) > 10:
                    old_history = old_history[-10:]

                persona.current_persona = persona_id
                persona.persona_confidence = confidence
                persona.persona_history_json = json.dumps(old_history)
                persona.primary_trait = primary_trait
                persona.secondary_trait = secondary_trait
                persona.calculated_at = _utc_now()
            else:
                # Create new
                persona = PlayerPersona(
                    steam_id=steam_id,
                    current_persona=persona_id,
                    persona_confidence=confidence,
                    persona_history_json=json.dumps([]),
                    primary_trait=primary_trait,
                    secondary_trait=secondary_trait,
                )
                session.add(persona)

            session.commit()
            return persona
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update persona: {e}")
            raise
        finally:
            session.close()

    # =========================================================================
    # Performance Trend Analysis
    # =========================================================================

    def get_player_trends(
        self,
        steam_id: str,
        days: int = 30,
    ) -> dict[str, Any] | None:
        """
        Get performance trends for a player over time.

        Calculates rating/ADR/winrate trends, detects slumps, and identifies
        improvement areas. This feature is 100% FREE - Leetify charges for it.

        Args:
            steam_id: Player's Steam ID (17 digits)
            days: Number of days to analyze (default 30)

        Returns:
            Performance trend data or None if insufficient history
        """
        from datetime import timedelta

        session = self.get_session()
        try:
            # Get match history within date range
            cutoff = _utc_now() - timedelta(days=days)
            history = (
                session.query(MatchHistory)
                .filter(
                    MatchHistory.steam_id == steam_id,
                    MatchHistory.analyzed_at >= cutoff,
                )
                .order_by(MatchHistory.analyzed_at.asc())
                .all()
            )

            if not history or len(history) < 2:
                return None

            # Build trend data
            rating_history = [
                {
                    "date": h.analyzed_at.isoformat() if h.analyzed_at else None,
                    "value": round(h.hltv_rating, 2) if h.hltv_rating else 1.0,
                    "map": h.map_name,
                }
                for h in history
            ]
            adr_history = [
                {
                    "date": h.analyzed_at.isoformat() if h.analyzed_at else None,
                    "value": round(h.adr, 1) if h.adr else 0,
                }
                for h in history
            ]

            # Calculate rolling 5-match winrate
            winrate_history = self._calculate_rolling_winrate(history, window=5)

            # Extract just values for trend/slump detection
            rating_values = [h.hltv_rating or 1.0 for h in history]
            adr_values = [h.adr or 0 for h in history]

            # Detect trends
            rating_trend = self._detect_trend(rating_values)
            adr_trend = self._detect_trend(adr_values)

            # Calculate change percentages
            rating_change = self._calculate_change_pct(rating_values)
            adr_change = self._calculate_change_pct(adr_values)

            # Detect slump
            slump_detected, slump_severity = self._detect_slump(rating_values)

            # Identify improvement areas
            improvement_areas = self._identify_improvement_areas(history)

            # Calculate map performance
            map_stats = self._calculate_map_performance(history)

            # Get player name from most recent match
            player_name = history[-1].map_name if history else "Unknown"
            # Try to get from profile
            profile = (
                session.query(PlayerProfile).filter(PlayerProfile.steam_id == steam_id).first()
            )
            if profile:
                player_name = profile.player_name

            return {
                "steam_id": steam_id,
                "player_name": player_name,
                "period_days": days,
                "matches_analyzed": len(history),
                "rating": {
                    "history": rating_history,
                    "trend": rating_trend,
                    "change_pct": round(rating_change * 100, 1),
                    "current_avg": round(sum(rating_values[-5:]) / min(5, len(rating_values)), 2),
                },
                "adr": {
                    "history": adr_history,
                    "trend": adr_trend,
                    "change_pct": round(adr_change * 100, 1),
                    "current_avg": round(sum(adr_values[-5:]) / min(5, len(adr_values)), 1),
                },
                "winrate": {
                    "history": winrate_history,
                    "current": winrate_history[-1]["value"] if winrate_history else 0,
                },
                "slump": {
                    "detected": slump_detected,
                    "severity": slump_severity,
                },
                "improvement_areas": improvement_areas,
                "map_performance": map_stats,
            }
        finally:
            session.close()

    def _detect_trend(self, values: list[float]) -> str:
        """
        Detect if values are trending up, down, or stable.

        Uses simple linear regression slope to determine trend direction.
        """
        if len(values) < 3:
            return "stable"

        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Normalize slope by the mean value to get relative change
        relative_slope = slope / (y_mean if y_mean != 0 else 1)

        if relative_slope > 0.02:  # More than 2% improvement per match
            return "improving"
        elif relative_slope < -0.02:  # More than 2% decline per match
            return "declining"
        return "stable"

    def _calculate_change_pct(self, values: list[float]) -> float:
        """Calculate percentage change between first and last values."""
        if len(values) < 2:
            return 0.0

        first_val = values[0] if values[0] != 0 else 0.001
        last_val = values[-1]

        return (last_val - first_val) / abs(first_val)

    def _detect_slump(self, rating_values: list[float]) -> tuple[bool, str | None]:
        """
        Detect if player is in a performance slump.

        Slump is defined as recent 5-match average being significantly
        below historical average.
        """
        if len(rating_values) < 5:
            return False, None

        recent_5 = rating_values[-5:]
        avg_recent = sum(recent_5) / 5

        # Compare to their historical average
        historical_avg = sum(rating_values) / len(rating_values)

        if historical_avg == 0:
            return False, None

        drop = (historical_avg - avg_recent) / historical_avg

        if drop > 0.20:  # 20%+ drop
            return True, "major"
        elif drop > 0.10:  # 10%+ drop
            return True, "minor"

        return False, None

    def _calculate_rolling_winrate(
        self, history: list[MatchHistory], window: int = 5
    ) -> list[dict]:
        """Calculate rolling winrate over match history."""
        result = []

        for i in range(len(history)):
            start_idx = max(0, i - window + 1)
            window_matches = history[start_idx : i + 1]

            wins = sum(1 for m in window_matches if m.result == "win")
            total = len(window_matches)
            winrate = (wins / total * 100) if total > 0 else 0

            result.append(
                {
                    "date": (
                        history[i].analyzed_at.isoformat() if history[i].analyzed_at else None
                    ),
                    "value": round(winrate, 1),
                    "window_size": total,
                }
            )

        return result

    def _identify_improvement_areas(self, history: list[MatchHistory]) -> list[dict]:
        """Identify areas where player could improve based on their stats."""
        areas = []
        total = len(history)
        if total == 0:
            return areas

        # Calculate averages
        avg_rating = sum(h.hltv_rating or 0 for h in history) / total
        avg_adr = sum(h.adr or 0 for h in history) / total
        avg_kast = sum(h.kast or 0 for h in history) / total
        avg_hs_pct = sum(h.hs_pct or 0 for h in history) / total

        # Trade stats
        total_trade_opps = sum(h.trade_kill_opportunities or 0 for h in history)
        total_trade_success = sum(h.trade_kill_success or 0 for h in history)
        trade_rate = (total_trade_success / total_trade_opps * 100) if total_trade_opps > 0 else 0

        # Entry stats
        total_entries = sum(h.entry_attempts or 0 for h in history)
        total_entry_wins = sum(h.entry_success or 0 for h in history)
        entry_rate = (total_entry_wins / total_entries * 100) if total_entries > 0 else 0

        # Clutch stats
        total_clutches = sum(h.clutch_situations or 0 for h in history)
        total_clutch_wins = sum(h.clutch_wins or 0 for h in history)
        clutch_rate = (total_clutch_wins / total_clutches * 100) if total_clutches > 0 else 0

        # Check thresholds and add improvement suggestions
        if avg_rating < 0.9:
            areas.append(
                {
                    "area": "Overall Impact",
                    "metric": "Rating",
                    "current": round(avg_rating, 2),
                    "target": 1.0,
                    "suggestion": "Focus on staying alive longer and getting more kills per round",
                }
            )

        if avg_adr < 70:
            areas.append(
                {
                    "area": "Damage Output",
                    "metric": "ADR",
                    "current": round(avg_adr, 1),
                    "target": 80,
                    "suggestion": "Be more aggressive in engagements, use utility to deal damage",
                }
            )

        if avg_kast < 60:
            areas.append(
                {
                    "area": "Round Involvement",
                    "metric": "KAST",
                    "current": round(avg_kast, 1),
                    "target": 70,
                    "suggestion": "Participate more in rounds - get kills, assists, or trade deaths",
                }
            )

        if avg_hs_pct < 35:
            areas.append(
                {
                    "area": "Headshot Accuracy",
                    "metric": "HS%",
                    "current": round(avg_hs_pct, 1),
                    "target": 45,
                    "suggestion": "Practice crosshair placement and aim for head level",
                }
            )

        if trade_rate < 40 and total_trade_opps >= 10:
            areas.append(
                {
                    "area": "Trading",
                    "metric": "Trade Rate",
                    "current": round(trade_rate, 1),
                    "target": 50,
                    "suggestion": "Stay closer to teammates to trade their deaths quickly",
                }
            )

        if entry_rate < 40 and total_entries >= 10:
            areas.append(
                {
                    "area": "Entry Fragging",
                    "metric": "Entry Win Rate",
                    "current": round(entry_rate, 1),
                    "target": 50,
                    "suggestion": "Work on pre-aiming common angles and using utility before entries",
                }
            )

        if clutch_rate < 20 and total_clutches >= 5:
            areas.append(
                {
                    "area": "Clutch Situations",
                    "metric": "Clutch Win Rate",
                    "current": round(clutch_rate, 1),
                    "target": 30,
                    "suggestion": "Focus on time management and isolating 1v1 duels in clutches",
                }
            )

        return areas

    def _calculate_map_performance(self, history: list[MatchHistory]) -> dict[str, dict]:
        """Calculate performance statistics per map."""
        map_data: dict[str, dict] = {}

        for h in history:
            map_name = h.map_name or "unknown"
            if map_name not in map_data:
                map_data[map_name] = {
                    "matches": 0,
                    "wins": 0,
                    "losses": 0,
                    "ratings": [],
                }

            map_data[map_name]["matches"] += 1
            if h.result == "win":
                map_data[map_name]["wins"] += 1
            elif h.result == "loss":
                map_data[map_name]["losses"] += 1
            map_data[map_name]["ratings"].append(h.hltv_rating or 1.0)

        # Calculate averages and win rates
        result = {}
        for map_name, data in map_data.items():
            ratings = data["ratings"]
            result[map_name] = {
                "matches": data["matches"],
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(data["wins"] / data["matches"] * 100, 1)
                if data["matches"] > 0
                else 0,
                "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else 1.0,
            }

        return result

    # =========================================================================
    # Statistics Operations
    # =========================================================================

    def get_global_stats(self) -> dict:
        """Get global statistics across all matches."""
        session = self.get_session()
        try:
            total_matches = session.query(func.count(Match.id)).scalar() or 0
            total_players = session.query(func.count(PlayerProfile.steam_id)).scalar() or 0
            total_rounds = session.query(func.sum(Match.total_rounds)).scalar() or 0

            # Map distribution
            map_counts = (
                session.query(Match.map_name, func.count(Match.id)).group_by(Match.map_name).all()
            )

            return {
                "total_matches": total_matches,
                "total_players": total_players,
                "total_rounds": total_rounds,
                "maps": dict(map_counts),
            }
        finally:
            session.close()


# =============================================================================
# Utility Functions
# =============================================================================


def compute_demo_hash(file_path: Path | str) -> str:
    """Compute SHA256 hash of a demo file for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


# Global database instance (lazy initialization)
_db_manager: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
