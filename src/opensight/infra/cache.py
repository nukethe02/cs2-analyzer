"""
Incremental Caching Module for Demo Analysis

Provides:
- File-based caching of analysis results
- Content-addressable storage (hash-based)
- Incremental updates (only re-analyze changed portions)
- Cache invalidation and cleanup
- Memory-efficient streaming
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_default_cache_dir() -> Path:
    """Get the default cache directory, compatible with Hugging Face Spaces."""
    # Check for explicit cache dir environment variable
    if env_cache := os.environ.get("OPENSIGHT_CACHE_DIR"):
        return Path(env_cache)

    # Try home directory first (works locally)
    home_cache = Path.home() / ".opensight" / "cache"
    try:
        home_cache.mkdir(parents=True, exist_ok=True)
        # Test if we can write
        test_file = home_cache / ".write_test"
        test_file.touch()
        test_file.unlink()
        return home_cache
    except (OSError, PermissionError):
        pass

    # Fallback to /tmp for Hugging Face Spaces and other restricted environments
    tmp_cache = Path("/tmp/opensight/cache")
    try:
        tmp_cache.mkdir(parents=True, exist_ok=True)
        return tmp_cache
    except (OSError, PermissionError):
        pass

    # Last resort: current working directory
    cwd_cache = Path.cwd() / ".opensight_cache"
    cwd_cache.mkdir(parents=True, exist_ok=True)
    return cwd_cache


# Default cache directory (HF Spaces compatible)
DEFAULT_CACHE_DIR = _get_default_cache_dir()

# Cache entry max age in days
DEFAULT_MAX_AGE_DAYS = 30

# Maximum cache size in MB
DEFAULT_MAX_CACHE_SIZE_MB = 1000


@dataclass
class CacheEntry:
    """A cached analysis result."""

    key: str
    file_hash: str
    file_path: str
    file_size: int
    created_at: datetime
    accessed_at: datetime
    analysis_version: str
    compressed: bool = True

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "file_hash": self.file_hash,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "analysis_version": self.analysis_version,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheEntry:
        return cls(
            key=data["key"],
            file_hash=data["file_hash"],
            file_path=data["file_path"],
            file_size=data["file_size"],
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            analysis_version=data["analysis_version"],
            compressed=data.get("compressed", True),
        )


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int
    total_size_bytes: int
    oldest_entry: datetime | None
    newest_entry: datetime | None
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0.0

    @property
    def size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "total_size_mb": round(self.size_mb, 2),
            "oldest_entry": (self.oldest_entry.isoformat() if self.oldest_entry else None),
            "newest_entry": (self.newest_entry.isoformat() if self.newest_entry else None),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_pct": round(self.hit_rate, 1),
        }


def compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Read chunk size in bytes

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute hash of string content."""
    return hashlib.sha256(content.encode()).hexdigest()


class DemoCache:
    """
    File-based cache for demo analysis results.

    Features:
    - Content-addressable storage (cache key = file hash)
    - Compressed storage (gzip)
    - Automatic cleanup of old entries
    - Thread-safe operations
    """

    # Analysis version - increment when analysis code changes significantly
    ANALYSIS_VERSION = "2.0.0"

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_size_mb: int = DEFAULT_MAX_CACHE_SIZE_MB,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in MB
            max_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_age = timedelta(days=max_age_days)

        self.data_dir = self.cache_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.index_path = self.cache_dir / "index.json"
        self._index: dict[str, CacheEntry] = {}

        self._hit_count = 0
        self._miss_count = 0

        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    data = json.load(f)
                self._index = {
                    k: CacheEntry.from_dict(v) for k, v in data.get("entries", {}).items()
                }
                self._hit_count = data.get("hit_count", 0)
                self._miss_count = data.get("miss_count", 0)
                logger.debug(f"Loaded cache index with {len(self._index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}

    def _save_index(self):
        """Save cache index to disk."""
        try:
            data = {
                "entries": {k: v.to_dict() for k, v in self._index.items()},
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
            }
            with open(self.index_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def _get_data_path(self, key: str) -> Path:
        """Get path for cached data file."""
        return self.data_dir / f"{key}.json.gz"

    def get_cache_key(self, demo_path: Path) -> str:
        """
        Get cache key for a demo file.

        The key is based on file hash + analysis version.
        """
        file_hash = compute_file_hash(demo_path)
        return f"{file_hash[:16]}_{self.ANALYSIS_VERSION}"

    def has(self, demo_path: Path) -> bool:
        """Check if analysis is cached for a demo."""
        key = self.get_cache_key(demo_path)
        return key in self._index and self._get_data_path(key).exists()

    def get(self, demo_path: Path) -> dict | None:
        """
        Get cached analysis for a demo.

        Args:
            demo_path: Path to demo file

        Returns:
            Cached analysis dict or None if not cached
        """
        # Ensure cache directory and index exist (first-run safety)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._miss_count += 1
            return None

        key = self.get_cache_key(demo_path)

        with open(self.index_path) as f:  # Add lock for thread safety
            if key not in self._index:
                self._miss_count += 1
                return None

            entry = self._index[key]
            data_path = self._get_data_path(key)

            if not data_path.exists():
                # Stale index entry - use lock when modifying
                del self._index[key]
                self._miss_count += 1
                self._save_index()
                return None

        try:
            with gzip.open(data_path, "rt") as f:
                data = json.load(f)

            # Update access time
            entry.accessed_at = datetime.now()
            self._hit_count += 1
            self._save_index()

            logger.debug(f"Cache hit for {demo_path.name}")
            return data

        except Exception as e:
            logger.warning(f"Failed to read cached data: {e}")
            self._miss_count += 1
            return None

    def put(self, demo_path: Path, analysis_data: dict):
        """
        Cache analysis for a demo.

        Args:
            demo_path: Path to demo file
            analysis_data: Analysis result dict
        """
        key = self.get_cache_key(demo_path)
        data_path = self._get_data_path(key)

        try:
            # Write compressed data
            with gzip.open(data_path, "wt") as f:
                json.dump(analysis_data, f)

            # Create index entry
            entry = CacheEntry(
                key=key,
                file_hash=key.split("_")[0],
                file_path=str(demo_path),
                file_size=demo_path.stat().st_size,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                analysis_version=self.ANALYSIS_VERSION,
                compressed=True,
            )
            self._index[key] = entry
            self._save_index()

            logger.debug(f"Cached analysis for {demo_path.name}")

            # Check if cleanup needed
            self._maybe_cleanup()

        except Exception as e:
            logger.warning(f"Failed to cache analysis: {e}")

    def invalidate(self, demo_path: Path):
        """Invalidate cache for a demo."""
        key = self.get_cache_key(demo_path)

        if key in self._index:
            del self._index[key]

        data_path = self._get_data_path(key)
        if data_path.exists():
            data_path.unlink()

        self._save_index()
        logger.debug(f"Invalidated cache for {demo_path.name}")

    def clear(self):
        """Clear all cached data."""
        self._index.clear()
        self._hit_count = 0
        self._miss_count = 0

        # Remove all data files
        for path in self.data_dir.glob("*.json.gz"):
            try:
                path.unlink()
            except Exception:
                pass

        self._save_index()
        logger.info("Cache cleared")

    def _maybe_cleanup(self):
        """Run cleanup if needed."""
        stats = self.get_stats()

        # Cleanup if over size limit
        if stats.total_size_bytes > self.max_size_bytes:
            self._cleanup_by_size()

        # Cleanup old entries
        self._cleanup_by_age()

    def _cleanup_by_size(self):
        """Remove old entries to get under size limit."""
        # Sort by access time (oldest first)
        sorted_entries = sorted(
            self._index.values(),
            key=lambda e: e.accessed_at,
        )

        current_size = sum(
            self._get_data_path(e.key).stat().st_size
            for e in self._index.values()
            if self._get_data_path(e.key).exists()
        )

        removed = 0
        for entry in sorted_entries:
            if current_size <= self.max_size_bytes * 0.8:  # Target 80%
                break

            data_path = self._get_data_path(entry.key)
            if data_path.exists():
                try:
                    size = data_path.stat().st_size
                    data_path.unlink()
                    current_size -= size
                    del self._index[entry.key]
                    removed += 1
                except Exception:
                    pass

        if removed:
            self._save_index()
            logger.info(f"Cleaned up {removed} cache entries by size")

    def _cleanup_by_age(self):
        """Remove entries older than max age."""
        now = datetime.now()
        removed = 0

        for key in list(self._index.keys()):
            entry = self._index[key]
            if now - entry.accessed_at > self.max_age:
                data_path = self._get_data_path(key)
                if data_path.exists():
                    try:
                        data_path.unlink()
                    except Exception:
                        pass
                del self._index[key]
                removed += 1

        if removed:
            self._save_index()
            logger.info(f"Cleaned up {removed} old cache entries")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_size = 0
        oldest = None
        newest = None

        for entry in self._index.values():
            data_path = self._get_data_path(entry.key)
            if data_path.exists():
                total_size += data_path.stat().st_size

            if oldest is None or entry.created_at < oldest:
                oldest = entry.created_at
            if newest is None or entry.created_at > newest:
                newest = entry.created_at

        return CacheStats(
            total_entries=len(self._index),
            total_size_bytes=total_size,
            oldest_entry=oldest,
            newest_entry=newest,
            hit_count=self._hit_count,
            miss_count=self._miss_count,
        )

    def list_entries(self) -> list[dict]:
        """List all cache entries."""
        return [e.to_dict() for e in self._index.values()]


class CachedAnalyzer:
    """
    Analyzer wrapper that uses caching.

    Automatically checks cache before analyzing and stores results.
    """

    def __init__(self, cache: DemoCache | None = None):
        """
        Initialize the cached analyzer.

        Args:
            cache: Cache instance (creates default if None)
        """
        self.cache = cache or DemoCache()

    def analyze(self, demo_path: Path, force: bool = False) -> dict:
        """
        Analyze a demo with caching.

        Args:
            demo_path: Path to demo file
            force: Force re-analysis even if cached

        Returns:
            Comprehensive analysis result dict including tactical data
        """
        # Check cache first
        if not force:
            cached = self.cache.get(demo_path)
            if cached:
                logger.info(f"Using cached analysis for {demo_path.name}")
                return cached

        # Run analysis
        logger.info(f"Analyzing {demo_path.name}")
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.core.enhanced_parser import CoachingAnalysisEngine
        from opensight.core.parser import DemoParser

        parser = DemoParser(demo_path)
        demo_data = parser.parse()

        analyzer = DemoAnalyzer(demo_data)
        analysis = analyzer.analyze()

        # Calculate professional metrics using enhanced parser
        try:
            logger.info("Calculating professional metrics (TTD, CP, Entry/Trade/Clutch)")
            engine = CoachingAnalysisEngine(demo_path)
            enhanced_metrics = engine.analyze()
        except Exception as e:
            logger.warning(f"Enhanced metrics calculation failed, using basic analysis only: {e}")
            enhanced_metrics = {}

        # Calculate RWS using direct method (more reliable)
        rws_data = self._calculate_rws_direct(demo_data)

        # Calculate multi-kills (2K, 3K, 4K, 5K)
        multikills = self._calculate_multikills(demo_data)

        # Build timeline graph data
        timeline_graph = self._build_timeline_graph_data(demo_data)

        # Build comprehensive player data
        players = {}
        for sid, p in analysis.players.items():
            mk = multikills.get(sid, {"2k": 0, "3k": 0, "4k": 0, "5k": 0})
            players[str(sid)] = {
                "name": p.name,
                "team": p.team,
                "stats": {
                    "kills": p.kills,
                    "deaths": p.deaths,
                    "assists": p.assists,
                    "adr": round(p.adr, 1) if p.adr else 0,
                    "headshot_pct": (
                        round(p.headshot_percentage, 1) if p.headshot_percentage else 0
                    ),
                    "kd_ratio": round(p.kills / max(1, p.deaths), 2),
                    "2k": mk["2k"],
                    "3k": mk["3k"],
                    "4k": mk["4k"],
                    "5k": mk["5k"],
                },
                "rating": {
                    "hltv_rating": round(p.hltv_rating, 2) if p.hltv_rating else 0,
                    "kast_percentage": (round(p.kast_percentage, 1) if p.kast_percentage else 0),
                    "aim_rating": round(p.aim_rating, 1) if p.aim_rating else 50,
                    "utility_rating": (round(p.utility_rating, 1) if p.utility_rating else 50),
                    "impact_rating": round(getattr(p, "impact_rating", None) or 0, 2),
                },
                "advanced": {
                    # From enhanced parser (TTD - Time to Damage)
                    "ttd_median_ms": round(getattr(p, "ttd_median_ms", None) or 0, 1),
                    "ttd_mean_ms": round(getattr(p, "ttd_mean_ms", None) or 0, 1),
                    "ttd_95th_ms": round(getattr(p, "ttd_95th_ms", None) or 0, 1),
                    # From enhanced parser (CP - Crosshair Placement)
                    "cp_median_error_deg": round(getattr(p, "cp_median_error_deg", None) or 0, 1),
                    "cp_mean_error_deg": round(getattr(p, "cp_mean_error_deg", None) or 0, 1),
                    # Other advanced stats
                    "prefire_kills": getattr(p, "prefire_kills", None) or 0,
                    "opening_kills": getattr(p, "opening_kills", None) or 0,
                    "opening_deaths": getattr(p, "opening_deaths", None) or 0,
                },
                "utility": {
                    "flashbangs_thrown": (p.utility.flashbangs_thrown if p.utility else 0),
                    "smokes_thrown": p.utility.smokes_thrown if p.utility else 0,
                    "he_thrown": p.utility.he_thrown if p.utility else 0,
                    "molotovs_thrown": p.utility.molotovs_thrown if p.utility else 0,
                    "flash_assists": p.utility.flash_assists if p.utility else 0,
                    "enemies_flashed": p.utility.enemies_flashed if p.utility else 0,
                    "teammates_flashed": (p.utility.teammates_flashed if p.utility else 0),
                    "he_damage": p.utility.he_damage if p.utility else 0,
                    "molotov_damage": p.utility.molotov_damage if p.utility else 0,
                },
                "duels": {
                    "trade_kills": getattr(p, "trade_kills", None) or 0,
                    "traded_deaths": getattr(p, "traded_deaths", None) or 0,
                    "clutch_wins": getattr(p, "clutch_wins", None) or 0,
                    "clutch_attempts": getattr(p, "clutch_attempts", None) or 0,
                },
                "entry": self._get_entry_stats(p),
                "trades": self._get_trade_stats(p),
                "clutches": self._get_clutch_stats(p),
                "rws": rws_data.get(
                    sid,
                    {
                        "avg_rws": 0,
                        "total_rws": 0,
                        "rounds_won": 0,
                        "rounds_played": 0,
                        "damage_per_round": 0,
                        "objective_completions": 0,
                    },
                ),
            }

        # Merge enhanced metrics from professional parser
        # NOTE: enhanced_metrics use integer steam IDs, but players dict uses string keys
        if enhanced_metrics and "entry_frags" in enhanced_metrics:
            for steam_id, entry_data in enhanced_metrics.get("entry_frags", {}).items():
                sid_str = str(steam_id)
                if sid_str in players:
                    if "entry" not in players[sid_str]:
                        players[sid_str]["entry"] = {}
                    players[sid_str]["entry"].update(
                        {
                            "entry_attempts": entry_data.get("entry_attempts", 0),
                            "entry_kills": entry_data.get("entry_kills", 0),
                            "entry_deaths": entry_data.get("entry_deaths", 0),
                        }
                    )

        # Merge TTD metrics
        if enhanced_metrics and "ttd_metrics" in enhanced_metrics:
            for steam_id, ttd_data in enhanced_metrics.get("ttd_metrics", {}).items():
                sid_str = str(steam_id)
                if sid_str in players:
                    players[sid_str]["advanced"].update(
                        {
                            "ttd_median_ms": ttd_data.get("ttd_median_ms", 0),
                            "ttd_mean_ms": ttd_data.get("ttd_mean_ms", 0),
                            "ttd_95th_ms": ttd_data.get("ttd_95th_ms", 0),
                        }
                    )

        # Merge CP metrics
        if enhanced_metrics and "crosshair_placement" in enhanced_metrics:
            for steam_id, cp_data in enhanced_metrics.get("crosshair_placement", {}).items():
                sid_str = str(steam_id)
                if sid_str in players:
                    players[sid_str]["advanced"].update(
                        {
                            "cp_median_error_deg": cp_data.get("cp_median_error_deg", 0),
                            "cp_mean_error_deg": cp_data.get("cp_mean_error_deg", 0),
                        }
                    )

        # Merge Trade Kill metrics
        if enhanced_metrics and "trade_kills" in enhanced_metrics:
            for steam_id, trade_data in enhanced_metrics.get("trade_kills", {}).items():
                sid_str = str(steam_id)
                if sid_str in players:
                    players[sid_str]["duels"].update(
                        {
                            "trade_kills": trade_data.get("trade_kills", 0),
                            "deaths_traded": trade_data.get("deaths_traded", 0),
                        }
                    )

        # Merge Clutch metrics (enhanced_parser uses "clutch_statistics")
        if enhanced_metrics and "clutch_statistics" in enhanced_metrics:
            for steam_id, clutch_data in enhanced_metrics.get("clutch_statistics", {}).items():
                sid_str = str(steam_id)
                if sid_str in players:
                    players[sid_str]["duels"].update(
                        {
                            "clutch_wins": clutch_data.get("clutch_wins", 0),
                            "clutch_attempts": clutch_data.get("clutch_attempts", 0),
                        }
                    )
                    # Add breakdown by variant
                    if "clutches" not in players[sid_str]:
                        players[sid_str]["clutches"] = {}
                    players[sid_str]["clutches"].update(
                        {
                            "v1_wins": clutch_data.get("v1_wins", 0),
                            "v2_wins": clutch_data.get("v2_wins", 0),
                            "v3_wins": clutch_data.get("v3_wins", 0),
                            "v4_wins": clutch_data.get("v4_wins", 0),
                            "v5_wins": clutch_data.get("v5_wins", 0),
                        }
                    )

        # Build round timeline
        round_timeline = self._build_round_timeline(demo_data, analysis)

        # DEBUG: Log timeline details
        logger.info(f"[DEBUG] round_timeline length: {len(round_timeline)}")
        if round_timeline:
            sample_round = round_timeline[0]
            logger.info(f"[DEBUG] Sample round keys: {list(sample_round.keys())}")
            logger.info(f"[DEBUG] Sample round events count: {len(sample_round.get('events', []))}")
            # Log first 3 rounds' event counts
            for _i, r in enumerate(round_timeline[:3]):
                events = r.get("events", [])
                logger.info(f"[DEBUG] Round {r.get('round_num')}: {len(events)} events")

        # Build kill matrix
        kill_matrix = self._build_kill_matrix(demo_data)

        # Build heatmap data
        heatmap_data = self._build_heatmap_data(demo_data)

        # Generate coaching insights
        coaching = self._generate_coaching_insights(demo_data, analysis, players)

        # Generate AI-powered match summaries for each player
        self._generate_ai_summaries(players, analysis)

        # Get tactical summary
        tactical = self._get_tactical_summary(demo_data, analysis)

        # Find MVP
        mvp = None
        if players:
            mvp_data = max(players.values(), key=lambda x: x["rating"]["hltv_rating"])
            mvp = {
                "name": mvp_data["name"],
                "rating": mvp_data["rating"]["hltv_rating"],
            }

        # Sort players by HLTV rating (descending) before returning
        players_sorted = sorted(
            players.values(), key=lambda p: p["rating"]["hltv_rating"], reverse=True
        )

        # Convert to dict for caching
        result = {
            "demo_info": {
                "map": analysis.map_name,
                "rounds": analysis.total_rounds,
                "duration_minutes": getattr(analysis, "duration_minutes", 30),
                "score": f"{analysis.team1_score} - {analysis.team2_score}",
                "total_kills": sum(p["stats"]["kills"] for p in players.values()),
                "team1_name": getattr(analysis, "team1_name", "Team 1"),
                "team2_name": getattr(analysis, "team2_name", "Team 2"),
            },
            "players": players_sorted,
            "mvp": mvp,
            "round_timeline": round_timeline,
            "kill_matrix": kill_matrix,
            "heatmap_data": heatmap_data,
            "coaching": coaching,
            "tactical": tactical,
            "timeline_graph": timeline_graph,
            "analyzed_at": datetime.now().isoformat(),
        }

        # Cache result
        self.cache.put(demo_path, result)

        return result

    def _get_entry_stats(self, player) -> dict:
        """Get comprehensive entry/opening duel stats like FACEIT."""
        opening = getattr(player, "opening_duels", None)
        if opening:
            attempts = getattr(opening, "attempts", 0) or 0
            wins = getattr(opening, "wins", 0) or 0
            losses = getattr(opening, "losses", 0) or 0
            rounds = getattr(player, "rounds_played", 0) or 1
            return {
                "entry_attempts": attempts,
                "entry_kills": wins,
                "entry_deaths": losses,
                "entry_diff": wins - losses,
                "entry_attempts_pct": (round(attempts / rounds * 100, 0) if rounds > 0 else 0),
                "entry_success_pct": (round(wins / attempts * 100, 0) if attempts > 0 else 0),
            }
        return {
            "entry_attempts": 0,
            "entry_kills": 0,
            "entry_deaths": 0,
            "entry_diff": 0,
            "entry_attempts_pct": 0,
            "entry_success_pct": 0,
        }

    def _get_trade_stats(self, player) -> dict:
        """Get comprehensive trade stats like FACEIT."""
        trades = getattr(player, "trades", None)
        if trades:
            return {
                "trade_kills": getattr(trades, "kills_traded", 0) or 0,
                "deaths_traded": getattr(trades, "deaths_traded", 0) or 0,
                "traded_entry_kills": getattr(trades, "traded_entry_kills", 0) or 0,
                "traded_entry_deaths": getattr(trades, "traded_entry_deaths", 0) or 0,
            }
        return {
            "trade_kills": 0,
            "deaths_traded": 0,
            "traded_entry_kills": 0,
            "traded_entry_deaths": 0,
        }

    def _get_clutch_stats(self, player) -> dict:
        """Get comprehensive clutch stats like FACEIT."""
        clutches = getattr(player, "clutches", None)
        if clutches:
            total = getattr(clutches, "total_situations", 0) or 0
            wins = getattr(clutches, "total_wins", 0) or 0
            return {
                "clutch_wins": wins,
                "clutch_losses": total - wins,
                "clutch_success_pct": round(wins / total * 100, 0) if total > 0 else 0,
                "v1_wins": getattr(clutches, "v1_wins", 0) or 0,
                "v2_wins": getattr(clutches, "v2_wins", 0) or 0,
                "v3_wins": getattr(clutches, "v3_wins", 0) or 0,
                "v4_wins": getattr(clutches, "v4_wins", 0) or 0,
                "v5_wins": getattr(clutches, "v5_wins", 0) or 0,
            }
        return {
            "clutch_wins": 0,
            "clutch_losses": 0,
            "clutch_success_pct": 0,
            "v1_wins": 0,
            "v2_wins": 0,
            "v3_wins": 0,
            "v4_wins": 0,
            "v5_wins": 0,
        }

    def _get_rws_for_player(self, steam_id: int, rws_data: dict) -> dict:
        """Get RWS data for a specific player."""
        if steam_id in rws_data:
            rws = rws_data[steam_id]
            return {
                "avg_rws": round(rws.avg_rws, 2),
                "total_rws": round(rws.total_rws, 1),
                "rounds_won": rws.rounds_won,
                "rounds_played": rws.rounds_played,
                "damage_per_round": round(rws.damage_per_round, 1),
                "objective_completions": rws.objective_completions,
            }
        return {
            "avg_rws": 0.0,
            "total_rws": 0.0,
            "rounds_won": 0,
            "rounds_played": 0,
            "damage_per_round": 0.0,
            "objective_completions": 0,
        }

    def _build_round_timeline(self, demo_data, analysis) -> list[dict]:
        """Build round-by-round timeline data with detailed events and win probability."""
        # Import win probability calculation
        try:
            from opensight.analysis.analytics import calculate_win_probability
        except ImportError:
            calculate_win_probability = None

        timeline = []
        kills = getattr(demo_data, "kills", [])
        rounds_data = getattr(demo_data, "rounds", [])
        player_names = getattr(demo_data, "player_names", {})

        logger.info(
            f"Building timeline: {len(kills)} kills, {len(rounds_data)} rounds, "
            f"{len(player_names)} players"
        )

        # Build round boundaries for tick-based inference
        round_boundaries = {}  # round_num -> (start_tick, end_tick)
        round_info = {}  # round_num -> round data
        for r in rounds_data:
            round_num = getattr(r, "round_num", 0)
            start_tick = getattr(r, "start_tick", 0)
            end_tick = getattr(r, "end_tick", 0)
            if round_num:
                round_boundaries[round_num] = (start_tick, end_tick)
                round_info[round_num] = r

        def infer_round(tick: int) -> int:
            """Infer round number from tick."""
            for rn, (st, et) in round_boundaries.items():
                if st <= tick <= et:
                    return rn
            # Fallback: find closest round
            if round_boundaries:
                for rn in sorted(round_boundaries.keys(), reverse=True):
                    st, et = round_boundaries[rn]
                    if tick > et:
                        return rn
                return 1
            return 1

        def tick_to_round_time(tick: int, round_start: int) -> float:
            """Convert tick to seconds from round start."""
            tick_rate = 64  # CS2 default
            return max(0, (tick - round_start) / tick_rate)

        # Check if kills have valid round data
        has_round_data = any(getattr(k, "round_num", 0) > 0 for k in kills[:10])

        # Group kills and events by round
        round_events: dict[int, list] = {}
        round_stats: dict[int, dict] = {}

        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            tick = getattr(kill, "tick", 0)

            # Infer round from tick if needed
            if round_num == 0:
                if round_boundaries and not has_round_data:
                    round_num = infer_round(tick)
                else:
                    round_num = 1

            if round_num not in round_events:
                round_events[round_num] = []
                round_stats[round_num] = {"ct_kills": 0, "t_kills": 0}

            # Get round start tick for time calculation
            round_start = round_boundaries.get(round_num, (0, 0))[0]
            time_seconds = tick_to_round_time(tick, round_start)

            attacker_name = getattr(kill, "attacker_name", "") or player_names.get(
                getattr(kill, "attacker_steamid", 0), "Unknown"
            )
            victim_name = getattr(kill, "victim_name", "") or player_names.get(
                getattr(kill, "victim_steamid", 0), "Unknown"
            )
            attacker_side = str(getattr(kill, "attacker_side", "")).upper()
            victim_side = str(getattr(kill, "victim_side", "")).upper()

            # Normalize team names
            attacker_team = "CT" if "CT" in attacker_side else "T"
            victim_team = "CT" if "CT" in victim_side else "T"

            # Track kill counts
            if attacker_team == "CT":
                round_stats[round_num]["ct_kills"] += 1
            else:
                round_stats[round_num]["t_kills"] += 1

            # Create kill event
            round_events[round_num].append(
                {
                    "tick": tick,
                    "time_seconds": round(time_seconds, 1),
                    "type": "kill",
                    "killer": attacker_name,
                    "killer_team": attacker_team,
                    "victim": victim_name,
                    "victim_team": victim_team,
                    "weapon": getattr(kill, "weapon", "unknown"),
                    "headshot": bool(getattr(kill, "headshot", False)),
                    "is_first_kill": len(round_events[round_num]) == 0,
                }
            )

        # Add bomb events from round data
        for round_num, r in round_info.items():
            if round_num not in round_events:
                round_events[round_num] = []
                round_stats[round_num] = {"ct_kills": 0, "t_kills": 0}

            round_start = round_boundaries.get(round_num, (0, 0))[0]

            # Check for bomb plant
            bomb_plant_tick = getattr(r, "bomb_plant_tick", None)
            if bomb_plant_tick:
                time_seconds = tick_to_round_time(bomb_plant_tick, round_start)
                round_events[round_num].append(
                    {
                        "tick": bomb_plant_tick,
                        "time_seconds": round(time_seconds, 1),
                        "type": "bomb_plant",
                        "player": getattr(r, "bomb_planter", "Unknown"),
                        "site": getattr(r, "bomb_site", "?"),
                    }
                )

            # Check for bomb defuse
            win_reason = str(getattr(r, "win_reason", "")).lower()
            if "defuse" in win_reason:
                defuse_tick = getattr(r, "end_tick", 0)
                time_seconds = tick_to_round_time(defuse_tick, round_start)
                round_events[round_num].append(
                    {
                        "tick": defuse_tick,
                        "time_seconds": round(time_seconds, 1),
                        "type": "bomb_defuse",
                        "player": getattr(r, "bomb_defuser", "Unknown"),
                    }
                )

            # Check for bomb explosion
            if "explod" in win_reason:
                explode_tick = getattr(r, "end_tick", 0)
                time_seconds = tick_to_round_time(explode_tick, round_start)
                round_events[round_num].append(
                    {
                        "tick": explode_tick,
                        "time_seconds": round(time_seconds, 1),
                        "type": "bomb_explode",
                    }
                )

        # Use actual round data if available, otherwise use analysis total_rounds
        total_rounds = (
            getattr(analysis, "total_rounds", 0) or len(round_boundaries) or len(round_events) or 30
        )

        # Build timeline entries for all rounds
        for round_num in range(1, total_rounds + 1):
            events = round_events.get(round_num, [])
            stats = round_stats.get(round_num, {"ct_kills": 0, "t_kills": 0})

            # Sort events by tick
            events.sort(key=lambda e: e.get("tick", 0))

            # Mark first kill
            kill_events = [e for e in events if e.get("type") == "kill"]
            if kill_events:
                kill_events[0]["is_first_kill"] = True
                for e in kill_events[1:]:
                    e["is_first_kill"] = False

            # Get winner from rounds data if available
            winner = "CT" if stats["ct_kills"] > stats["t_kills"] else "T"
            win_reason = "Elimination"
            round_type = "full_buy"

            if round_num in round_info:
                r = round_info[round_num]
                winner = str(getattr(r, "winner", winner)).upper()
                if "CT" not in winner and "T" not in winner:
                    winner = "CT" if stats["ct_kills"] > stats["t_kills"] else "T"
                win_reason = getattr(r, "win_reason", win_reason) or win_reason

                # Use round_type from parser (populated from equipment values)
                stored_round_type = getattr(r, "round_type", "")
                if stored_round_type:
                    round_type = stored_round_type

            # Fallback pistol detection if round_type not set from parser
            # Use is_pistol_round() for proper OT handling
            if round_type == "full_buy":
                from opensight.core.parser import is_pistol_round

                # Detect MR format from total rounds
                rounds_per_half = 12 if total_rounds <= 30 else 15
                if is_pistol_round(round_num, rounds_per_half):
                    round_type = "pistol"

            # Get first kill/death info
            first_kill = None
            first_death = None
            if kill_events:
                first_kill = kill_events[0].get("killer")
                first_death = kill_events[0].get("victim")

            # Calculate win probability timeline for this round
            momentum = None
            if calculate_win_probability is not None:
                momentum = self._calculate_round_momentum(events, winner, calculate_win_probability)
                # Add win probability to each event
                if momentum:
                    prob_by_tick = {p["tick"]: p for p in momentum.get("timeline", [])}
                    for event in events:
                        tick = event.get("tick", 0)
                        if tick in prob_by_tick:
                            event["ct_prob"] = prob_by_tick[tick]["ct_prob"]
                            event["t_prob"] = prob_by_tick[tick]["t_prob"]

            timeline.append(
                {
                    "round_num": round_num,
                    "round_type": round_type,
                    "winner": winner,
                    "win_reason": win_reason,
                    "first_kill": first_kill,
                    "first_death": first_death,
                    "ct_kills": stats["ct_kills"],
                    "t_kills": stats["t_kills"],
                    "events": events,
                    "momentum": momentum,
                }
            )

        # Log timeline generation stats
        total_events = sum(len(r.get("events", [])) for r in timeline)
        rounds_with_events = sum(1 for r in timeline if r.get("events"))
        throws = sum(1 for r in timeline if r.get("momentum", {}).get("round_tag"))
        logger.info(
            f"Built round timeline: {len(timeline)} rounds, "
            f"{rounds_with_events} with events, {total_events} total events, "
            f"{throws} throw/heroic rounds"
        )

        return timeline

    def _calculate_round_momentum(self, events: list[dict], winner: str, calc_prob_fn) -> dict:
        """
        Calculate win probability timeline for a single round.

        Tracks probability at each state change (kill, bomb plant) to identify
        "throw" rounds (had >=80% prob, lost) and "heroic" rounds (had <=20%, won).

        Args:
            events: List of round events (kills, bomb plants, etc.)
            winner: Round winner ("CT" or "T")
            calc_prob_fn: Win probability calculation function

        Returns:
            Dict with momentum data including timeline, peak/min probs, and flags
        """
        prob_timeline = []

        # Initial state: 5v5, no bomb planted
        ct_alive = 5
        t_alive = 5
        bomb_planted = False

        # Add round start
        ct_prob = calc_prob_fn("CT", ct_alive, t_alive, bomb_planted)
        t_prob = calc_prob_fn("T", ct_alive, t_alive, bomb_planted)
        prob_timeline.append(
            {
                "tick": 0,
                "time": 0.0,
                "event": "round_start",
                "ct_alive": ct_alive,
                "t_alive": t_alive,
                "bomb_planted": bomb_planted,
                "ct_prob": round(ct_prob, 2),
                "t_prob": round(t_prob, 2),
                "desc": "Round start (5v5)",
            }
        )

        # Process each event in order
        for event in events:
            event_type = event.get("type", "")
            tick = event.get("tick", 0)
            time_sec = event.get("time_seconds", 0.0)

            if event_type == "kill":
                # Update alive count
                victim_team = event.get("victim_team", "")
                if victim_team == "CT":
                    ct_alive = max(0, ct_alive - 1)
                elif victim_team == "T":
                    t_alive = max(0, t_alive - 1)

                desc = f"{event.get('killer', 'Unknown')} killed {event.get('victim', 'Unknown')}"

            elif event_type == "bomb_plant":
                bomb_planted = True
                desc = "Bomb planted"

            elif event_type == "bomb_defuse":
                bomb_planted = False
                desc = "Bomb defused"

            elif event_type == "bomb_explode":
                desc = "Bomb exploded"

            else:
                continue  # Skip unknown events

            # Calculate new probabilities
            ct_prob = calc_prob_fn("CT", ct_alive, t_alive, bomb_planted)
            t_prob = calc_prob_fn("T", ct_alive, t_alive, bomb_planted)

            prob_timeline.append(
                {
                    "tick": tick,
                    "time": round(time_sec, 1),
                    "event": event_type,
                    "ct_alive": ct_alive,
                    "t_alive": t_alive,
                    "bomb_planted": bomb_planted,
                    "ct_prob": round(ct_prob, 2),
                    "t_prob": round(t_prob, 2),
                    "desc": desc,
                }
            )

        # Calculate peak/min probabilities
        ct_probs = [p["ct_prob"] for p in prob_timeline]
        t_probs = [p["t_prob"] for p in prob_timeline]

        ct_peak = max(ct_probs) if ct_probs else 0.5
        ct_min = min(ct_probs) if ct_probs else 0.5
        t_peak = max(t_probs) if t_probs else 0.5
        t_min = min(t_probs) if t_probs else 0.5

        # Determine throw/heroic status
        ct_is_throw = ct_peak >= 0.80 and winner == "T"
        ct_is_heroic = ct_min <= 0.20 and winner == "CT"
        t_is_throw = t_peak >= 0.80 and winner == "CT"
        t_is_heroic = t_min <= 0.20 and winner == "T"

        # Determine round tag
        if ct_is_throw:
            round_tag = "CT_THROW"
        elif t_is_throw:
            round_tag = "T_THROW"
        elif ct_is_heroic:
            round_tag = "CT_HEROIC"
        elif t_is_heroic:
            round_tag = "T_HEROIC"
        else:
            round_tag = ""

        return {
            "winner": winner,
            "ct_peak_prob": ct_peak,
            "ct_min_prob": ct_min,
            "t_peak_prob": t_peak,
            "t_min_prob": t_min,
            "ct_is_throw": ct_is_throw,
            "ct_is_heroic": ct_is_heroic,
            "t_is_throw": t_is_throw,
            "t_is_heroic": t_is_heroic,
            "round_tag": round_tag,
            "timeline": prob_timeline,
        }

    def _calculate_multikills(self, demo_data) -> dict[int, dict]:
        """Calculate multi-kill counts (2K, 3K, 4K, 5K) per player per round."""
        kills = getattr(demo_data, "kills", [])

        # Count kills per player per round
        player_round_kills: dict[int, dict[int, int]] = {}  # steam_id -> {round_num -> kill_count}

        for kill in kills:
            attacker_id = getattr(kill, "attacker_steamid", 0)
            round_num = getattr(kill, "round_num", 0)

            if attacker_id and round_num:
                if attacker_id not in player_round_kills:
                    player_round_kills[attacker_id] = {}
                if round_num not in player_round_kills[attacker_id]:
                    player_round_kills[attacker_id][round_num] = 0
                player_round_kills[attacker_id][round_num] += 1

        # Count 2K, 3K, 4K, 5K for each player
        result: dict[int, dict] = {}
        for steam_id, round_kills in player_round_kills.items():
            counts = {"2k": 0, "3k": 0, "4k": 0, "5k": 0}
            for _, kill_count in round_kills.items():
                if kill_count == 2:
                    counts["2k"] += 1
                elif kill_count == 3:
                    counts["3k"] += 1
                elif kill_count == 4:
                    counts["4k"] += 1
                elif kill_count >= 5:
                    counts["5k"] += 1
            result[steam_id] = counts

        return result

    def _build_timeline_graph_data(self, demo_data) -> dict:
        """Build round-by-round data for Leetify-style timeline graphs.

        Tracks per-round cumulative stats for all players:
        - kills, deaths, damage, awp_kills, enemies_flashed
        - Team information for grouping (CT/T)
        """
        kills = getattr(demo_data, "kills", [])
        damages = getattr(demo_data, "damages", [])
        blinds = getattr(demo_data, "blinds", [])
        player_names = getattr(demo_data, "player_names", {})
        player_teams = getattr(demo_data, "player_teams", {})
        rounds = getattr(demo_data, "rounds", [])

        # Build round boundaries from round events (for inferring round_num from tick)
        round_boundaries = []  # List of (round_num, start_tick, end_tick)
        for r in rounds:
            round_num = getattr(r, "round_num", 0)
            start_tick = getattr(r, "start_tick", 0)
            end_tick = getattr(r, "end_tick", 0)
            if round_num and end_tick > 0:
                round_boundaries.append((round_num, start_tick, end_tick))
        round_boundaries.sort(key=lambda x: x[1])  # Sort by start_tick

        def infer_round_from_tick(tick: int) -> int:
            """Infer round number from tick using round boundaries."""
            for round_num, start_tick, end_tick in round_boundaries:
                if start_tick <= tick <= end_tick:
                    return round_num
            if round_boundaries:
                for round_num, _start_tick, end_tick in reversed(round_boundaries):
                    if tick > end_tick:
                        return round_num
                return 1
            return 1

        # Check if kills have valid round data
        has_round_data = any(getattr(k, "round_num", 0) > 0 for k in kills[:10])

        # Initialize per-player round data with all metrics
        # steam_id -> {round_num -> {kills, deaths, damage, awp_kills, enemies_flashed}}
        player_round_data: dict[int, dict[int, dict]] = {}

        # Get max rounds from demo_data or round boundaries
        max_round = getattr(demo_data, "num_rounds", 0) or len(round_boundaries) or 1

        def ensure_player_round(steam_id: int, round_num: int) -> None:
            """Ensure player and round entry exists with all metric fields."""
            if steam_id not in player_round_data:
                player_round_data[steam_id] = {}
            if round_num not in player_round_data[steam_id]:
                player_round_data[steam_id][round_num] = {
                    "kills": 0,
                    "deaths": 0,
                    "damage": 0,
                    "awp_kills": 0,
                    "enemies_flashed": 0,
                }

        # Process kills - track both attacker (kills) and victim (deaths)
        for kill in kills:
            attacker_id = getattr(kill, "attacker_steamid", 0)
            victim_id = getattr(kill, "victim_steamid", 0)
            round_num = getattr(kill, "round_num", 0)
            weapon = str(getattr(kill, "weapon", "")).lower()

            # Infer round if missing
            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(kill, "tick", 0)
                    round_num = infer_round_from_tick(tick)
                else:
                    round_num = 1

            max_round = max(max_round, round_num)

            # Track kills for attacker
            if attacker_id:
                ensure_player_round(attacker_id, round_num)
                player_round_data[attacker_id][round_num]["kills"] += 1
                # Track AWP kills
                if "awp" in weapon:
                    player_round_data[attacker_id][round_num]["awp_kills"] += 1

            # Track deaths for victim
            if victim_id:
                ensure_player_round(victim_id, round_num)
                player_round_data[victim_id][round_num]["deaths"] += 1

        # Process damage
        for dmg in damages:
            attacker_id = getattr(dmg, "attacker_steamid", 0)
            round_num = getattr(dmg, "round_num", 0)
            damage_val = getattr(dmg, "damage", 0)

            if not attacker_id:
                continue

            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(dmg, "tick", 0)
                    round_num = infer_round_from_tick(tick)
                else:
                    round_num = 1

            max_round = max(max_round, round_num)
            ensure_player_round(attacker_id, round_num)
            player_round_data[attacker_id][round_num]["damage"] += damage_val

        # Process blinds - count enemies flashed (duration > 0.5s, not teammate)
        for blind in blinds:
            attacker_id = getattr(blind, "attacker_steamid", 0)
            round_num = getattr(blind, "round_num", 0)
            duration = getattr(blind, "blind_duration", 0.0)
            is_teammate = getattr(blind, "is_teammate", False)

            # Only count enemy flashes with meaningful duration
            if not attacker_id or duration < 0.5 or is_teammate:
                continue

            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(blind, "tick", 0)
                    round_num = infer_round_from_tick(tick)
                else:
                    round_num = 1

            max_round = max(max_round, round_num)
            ensure_player_round(attacker_id, round_num)
            player_round_data[attacker_id][round_num]["enemies_flashed"] += 1

        # Build cumulative data for graphs
        players_timeline = []
        for steam_id, round_data in player_round_data.items():
            player_name = player_names.get(steam_id, f"Player {steam_id}")
            # Get team - prefer from player_teams dict, fallback to inferring from kills
            team = player_teams.get(steam_id, "Unknown")
            if team == "Unknown":
                # Try to infer from kills
                for kill in kills:
                    if getattr(kill, "attacker_steamid", 0) == steam_id:
                        side = str(getattr(kill, "attacker_side", "")).upper()
                        if "CT" in side:
                            team = "CT"
                        elif "T" in side:
                            team = "T"
                        break
                    if getattr(kill, "victim_steamid", 0) == steam_id:
                        side = str(getattr(kill, "victim_side", "")).upper()
                        if "CT" in side:
                            team = "CT"
                        elif "T" in side:
                            team = "T"
                        break

            # Build cumulative stats per round
            cumulative = {
                "kills": 0,
                "deaths": 0,
                "damage": 0,
                "awp_kills": 0,
                "enemies_flashed": 0,
            }
            rounds_list = []

            for r in range(1, max_round + 1):
                rd = round_data.get(
                    r,
                    {
                        "kills": 0,
                        "deaths": 0,
                        "damage": 0,
                        "awp_kills": 0,
                        "enemies_flashed": 0,
                    },
                )
                cumulative["kills"] += rd["kills"]
                cumulative["deaths"] += rd["deaths"]
                cumulative["damage"] += rd["damage"]
                cumulative["awp_kills"] += rd["awp_kills"]
                cumulative["enemies_flashed"] += rd["enemies_flashed"]

                rounds_list.append(
                    {
                        "round": r,
                        "kills": cumulative["kills"],
                        "deaths": cumulative["deaths"],
                        "damage": cumulative["damage"],
                        "awp_kills": cumulative["awp_kills"],
                        "enemies_flashed": cumulative["enemies_flashed"],
                        # Per-round values for tooltips
                        "round_kills": rd["kills"],
                        "round_deaths": rd["deaths"],
                        "round_damage": rd["damage"],
                    }
                )

            players_timeline.append(
                {
                    "steam_id": steam_id,
                    "name": player_name,
                    "team": team,
                    "rounds": rounds_list,
                }
            )

        # Sort players by team for better grouping
        players_timeline.sort(key=lambda p: (p["team"] != "CT", p["name"]))

        # Build round scores from rounds data (for Round Difference chart)
        round_scores = []
        ct_score = 0
        t_score = 0
        for r in range(1, max_round + 1):
            round_info = None
            for rd in rounds:
                if getattr(rd, "round_num", 0) == r:
                    round_info = rd
                    break

            if round_info:
                winner = str(getattr(round_info, "winner", "")).upper()
                if "CT" in winner:
                    ct_score += 1
                elif "T" in winner:
                    t_score += 1
                else:
                    # Infer from kill differential if no winner
                    ct_kills = sum(
                        1
                        for k in kills
                        if getattr(k, "round_num", 0) == r
                        and "CT" in str(getattr(k, "attacker_side", "")).upper()
                    )
                    t_kills = sum(
                        1
                        for k in kills
                        if getattr(k, "round_num", 0) == r
                        and "T" in str(getattr(k, "attacker_side", "")).upper()
                    )
                    if ct_kills > t_kills:
                        ct_score += 1
                    elif t_kills > ct_kills:
                        t_score += 1

            round_scores.append(
                {
                    "round": r,
                    "ct_score": ct_score,
                    "t_score": t_score,
                    "diff": ct_score - t_score,  # Positive = CT leading
                }
            )

        return {
            "max_rounds": max_round,
            "players": players_timeline,
            "round_scores": round_scores,
        }

    def _calculate_rws_direct(self, demo_data) -> dict[int, dict]:
        """Calculate RWS directly from demo data with better team handling."""
        kills = getattr(demo_data, "kills", [])
        damages = getattr(demo_data, "damages", [])
        rounds = getattr(demo_data, "rounds", [])
        player_names = getattr(demo_data, "player_names", {})

        if not rounds or not kills:
            return {}

        # Build player teams from FIRST HALF kills only (rounds 1-12)
        # to establish starting side, then handle halftime swap
        player_starting_teams: dict[int, str] = {}
        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            # Only use first half kills to determine starting team
            if round_num > 12:
                continue

            att_id = getattr(kill, "attacker_steamid", 0)
            att_side = str(getattr(kill, "attacker_side", "")).upper()
            vic_id = getattr(kill, "victim_steamid", 0)
            vic_side = str(getattr(kill, "victim_side", "")).upper()

            if att_id and att_id not in player_starting_teams:
                if "CT" in att_side:
                    player_starting_teams[att_id] = "CT"
                elif "T" in att_side:
                    player_starting_teams[att_id] = "T"
            if vic_id and vic_id not in player_starting_teams:
                if "CT" in vic_side:
                    player_starting_teams[vic_id] = "CT"
                elif "T" in vic_side:
                    player_starting_teams[vic_id] = "T"

        # If no first-half kills found, fall back to any kill data
        if not player_starting_teams:
            for kill in kills:
                att_id = getattr(kill, "attacker_steamid", 0)
                att_side = str(getattr(kill, "attacker_side", "")).upper()
                vic_id = getattr(kill, "victim_steamid", 0)
                vic_side = str(getattr(kill, "victim_side", "")).upper()

                if att_id and att_id not in player_starting_teams:
                    if "CT" in att_side:
                        player_starting_teams[att_id] = "CT"
                    elif "T" in att_side:
                        player_starting_teams[att_id] = "T"
                if vic_id and vic_id not in player_starting_teams:
                    if "CT" in vic_side:
                        player_starting_teams[vic_id] = "CT"
                    elif "T" in vic_side:
                        player_starting_teams[vic_id] = "T"

        # Group damage by round with attacker's side for that specific event
        round_damages: dict[int, dict[int, int]] = {}  # round_num -> {steam_id -> damage}
        round_player_sides: dict[int, dict[int, str]] = {}  # round_num -> {steam_id -> side}
        for dmg in damages:
            round_num = getattr(dmg, "round_num", 0)
            attacker_id = getattr(dmg, "attacker_steamid", 0)
            damage_val = getattr(dmg, "damage", 0)
            attacker_side = str(getattr(dmg, "attacker_side", "")).upper()
            victim_side = str(getattr(dmg, "victim_side", "")).upper()

            # Only count damage to enemies
            is_enemy_damage = (
                "CT" in attacker_side and "T" in victim_side and "CT" not in victim_side
            ) or ("T" in attacker_side and "CT" not in attacker_side and "CT" in victim_side)

            if attacker_id and round_num and is_enemy_damage:
                if round_num not in round_damages:
                    round_damages[round_num] = {}
                    round_player_sides[round_num] = {}
                if attacker_id not in round_damages[round_num]:
                    round_damages[round_num][attacker_id] = 0
                round_damages[round_num][attacker_id] += damage_val
                # Track the side for this player in this round
                if "CT" in attacker_side:
                    round_player_sides[round_num][attacker_id] = "CT"
                elif "T" in attacker_side:
                    round_player_sides[round_num][attacker_id] = "T"

        # Initialize player stats
        player_stats: dict[int, dict] = {}
        for pid in player_names:
            player_stats[pid] = {
                "rounds_played": 0,
                "rounds_won": 0,
                "total_rws": 0.0,
                "total_damage": 0,
            }

        # Calculate RWS for each round
        for round_info in rounds:
            round_num = getattr(round_info, "round_num", 0)
            winner = str(getattr(round_info, "winner", "")).upper()

            if not winner or winner == "UNKNOWN":
                continue

            round_dmg = round_damages.get(round_num, {})
            round_sides = round_player_sides.get(round_num, {})

            # Find winning players based on their side for THIS round
            winning_players = []
            for pid in player_stats:
                # First try to get side from damage events in this round (most accurate)
                player_side = round_sides.get(pid, "")
                # Fall back to calculated side based on starting team + halftime
                if not player_side:
                    player_side = demo_data.get_player_side_for_round(pid, round_num)

                if player_side in ["CT", "T"]:
                    player_stats[pid]["rounds_played"] += 1
                    # Check if player is on winning team
                    if player_side == winner:
                        winning_players.append(pid)

            if not winning_players:
                continue

            # Calculate total damage by winning team
            winning_team_damage = sum(round_dmg.get(pid, 0) for pid in winning_players)

            # Distribute 100 RWS among winning team based on damage
            for pid in winning_players:
                player_stats[pid]["rounds_won"] += 1
                player_damage = round_dmg.get(pid, 0)
                player_stats[pid]["total_damage"] += player_damage

                if winning_team_damage > 0:
                    damage_share = player_damage / winning_team_damage
                    rws_this_round = damage_share * 100
                else:
                    # Equal share if no damage recorded
                    rws_this_round = 100 / len(winning_players)

                player_stats[pid]["total_rws"] += rws_this_round

        # Build results
        results = {}
        for pid, stats in player_stats.items():
            rounds_played = max(stats["rounds_played"], 1)
            results[pid] = {
                "avg_rws": round(stats["total_rws"] / rounds_played, 2),
                "total_rws": round(stats["total_rws"], 1),
                "rounds_won": stats["rounds_won"],
                "rounds_played": stats["rounds_played"],
                "damage_per_round": round(stats["total_damage"] / rounds_played, 1),
                "objective_completions": 0,
            }

        return results

    def _build_kill_matrix(self, demo_data) -> list[dict]:
        """Build kill matrix showing who killed who."""
        kills = getattr(demo_data, "kills", [])
        player_names = getattr(demo_data, "player_names", {})

        matrix = {}
        for kill in kills:
            attacker_id = getattr(kill, "attacker_steamid", 0)
            victim_id = getattr(kill, "victim_steamid", 0)
            attacker_name = player_names.get(attacker_id, getattr(kill, "attacker_name", "Unknown"))
            victim_name = player_names.get(victim_id, getattr(kill, "victim_name", "Unknown"))

            key = (attacker_name, victim_name)
            matrix[key] = matrix.get(key, 0) + 1

        return [{"attacker": k[0], "victim": k[1], "count": v} for k, v in matrix.items()]

    def _build_heatmap_data(self, demo_data) -> dict:
        """Build comprehensive position data for heatmap visualization.

        Includes zone detection, side info, phase (pre/post plant), and economy context.
        """
        kills = getattr(demo_data, "kills", [])
        rounds = getattr(demo_data, "rounds", [])
        player_names = getattr(demo_data, "player_names", {})
        map_name = getattr(demo_data, "map_name", "").lower()

        # Build round lookup for bomb plant and economy data
        round_info = {}
        for r in rounds:
            round_num = getattr(r, "round_num", 0)
            round_info[round_num] = {
                "bomb_plant_tick": getattr(r, "bomb_plant_tick", None),
                "bomb_site": getattr(r, "bomb_site", ""),
                "ct_equipment": getattr(r, "ct_equipment_value", 0),
                "t_equipment": getattr(r, "t_equipment_value", 0),
                "round_type": getattr(r, "round_type", ""),
            }

        # Import zone detection function
        try:
            from opensight.visualization.radar import (
                MAP_ZONES,
                classify_round_economy,
                get_zone_for_position,
            )

            has_zones = map_name in MAP_ZONES
        except ImportError:
            has_zones = False
            MAP_ZONES: dict = {}  # type: ignore[no-redef]

            def get_zone_for_position(
                map_name: str, x: float, y: float, z: float | None = None
            ) -> str:
                _ = map_name, x, y, z  # Unused in fallback
                return "Unknown"

            def classify_round_economy(equipment_value: int, is_pistol_round: bool) -> str:
                _ = equipment_value, is_pistol_round  # Unused in fallback
                return "unknown"

        # Import pistol round detection
        try:
            from opensight.core.parser import is_pistol_round as check_pistol
        except ImportError:
            # Simple fallback if import fails
            def check_pistol(rn: int, rph: int = 12) -> bool:
                return rn == 1 or rn == rph + 1

        # Detect MR format from total rounds
        total_rounds = len(rounds)
        rounds_per_half = 12 if total_rounds <= 30 else 15

        kill_positions = []
        death_positions = []
        zone_stats: dict[str, dict] = {}

        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            tick = getattr(kill, "tick", 0)
            r_info = round_info.get(round_num, {})  # type: ignore[arg-type]

            # Determine phase (pre-plant vs post-plant)
            bomb_plant_tick = r_info.get("bomb_plant_tick", 0)  # type: ignore[union-attr]
            phase = "pre_plant"
            if bomb_plant_tick and tick >= int(bomb_plant_tick or 0):  # type: ignore[arg-type]
                phase = "post_plant"

            # Determine economy round type
            attacker_side = getattr(kill, "attacker_side", "") or ""
            is_pistol = check_pistol(round_num, rounds_per_half)
            eq_key = "t_equipment" if "T" in attacker_side.upper() else "ct_equipment"
            eq_raw = r_info.get(eq_key, 0)  # type: ignore[union-attr]
            eq_value = int(eq_raw) if eq_raw else 0  # type: ignore[arg-type]
            round_type_raw = r_info.get("round_type", "")  # type: ignore[union-attr]
            stored_round_type = str(round_type_raw) if round_type_raw else ""  # type: ignore[arg-type]
            if stored_round_type:
                round_type = str(stored_round_type)
            elif has_zones:
                round_type = classify_round_economy(eq_value, is_pistol)
            else:
                round_type = "pistol" if is_pistol else "unknown"

            # Kill position (attacker)
            ax = getattr(kill, "attacker_x", None)
            ay = getattr(kill, "attacker_y", None)
            if ax is not None and ay is not None:
                az = getattr(kill, "attacker_z", 0) or 0
                zone = get_zone_for_position(map_name, ax, ay, az) if has_zones else "Unknown"
                kill_positions.append(
                    {
                        "x": ax,
                        "y": ay,
                        "z": az,
                        "zone": zone,
                        "side": attacker_side,
                        "phase": phase,
                        "round_type": round_type,
                        "round_num": round_num,
                        "player_name": player_names.get(
                            getattr(kill, "attacker_steamid", 0), "Unknown"
                        ),
                        "player_steamid": getattr(kill, "attacker_steamid", 0),
                        "weapon": getattr(kill, "weapon", ""),
                        "headshot": getattr(kill, "headshot", False),
                    }
                )

                # Update zone stats for kills
                if zone not in zone_stats:
                    zone_stats[zone] = {
                        "kills": 0,
                        "deaths": 0,
                        "ct_kills": 0,
                        "t_kills": 0,
                    }
                zone_stats[zone]["kills"] += 1
                if "CT" in attacker_side.upper():
                    zone_stats[zone]["ct_kills"] += 1
                else:
                    zone_stats[zone]["t_kills"] += 1

            # Death position (victim)
            vx = getattr(kill, "victim_x", None)
            vy = getattr(kill, "victim_y", None)
            if vx is not None and vy is not None:
                vz = getattr(kill, "victim_z", 0) or 0
                victim_side = getattr(kill, "victim_side", "") or ""
                zone = get_zone_for_position(map_name, vx, vy, vz) if has_zones else "Unknown"
                death_positions.append(
                    {
                        "x": vx,
                        "y": vy,
                        "z": vz,
                        "zone": zone,
                        "side": victim_side,
                        "phase": phase,
                        "round_type": round_type,
                        "round_num": round_num,
                        "player_name": player_names.get(
                            getattr(kill, "victim_steamid", 0), "Unknown"
                        ),
                        "player_steamid": getattr(kill, "victim_steamid", 0),
                    }
                )

                # Update zone stats for deaths
                if zone not in zone_stats:
                    zone_stats[zone] = {
                        "kills": 0,
                        "deaths": 0,
                        "ct_kills": 0,
                        "t_kills": 0,
                    }
                zone_stats[zone]["deaths"] += 1

        # Calculate zone K/D ratios and percentages
        total_kills = len(kill_positions)
        for _zone, stats in zone_stats.items():
            k = stats["kills"]
            d = stats["deaths"]
            stats["kd_ratio"] = round(k / max(d, 1), 2)
            stats["kill_pct"] = round(k / max(total_kills, 1) * 100, 1)

        # Get zone definitions for frontend if available
        zone_definitions = {}
        if has_zones:
            try:
                zone_definitions = MAP_ZONES.get(map_name, {})
            except Exception:
                pass

        return {
            "map_name": map_name,
            "kill_positions": kill_positions,
            "death_positions": death_positions,
            "zone_stats": zone_stats,
            "zone_definitions": zone_definitions,
        }

    def _generate_coaching_insights(self, demo_data, analysis, players: dict) -> list[dict]:
        """
        Generate comprehensive, data-driven coaching insights for each player.

        Inspired by Leetify's detailed coaching system, this provides:
        - Specific metrics with actual numbers and comparisons
        - Role-specific analysis
        - Comparative insights vs teammates and match averages
        - Player identity/archetype detection
        - Actionable improvement recommendations
        """
        coaching = []

        # Calculate match-wide statistics for comparison
        match_stats = self._calculate_match_averages(players)

        # Find top performers in each category for comparative insights
        top_performers = self._find_top_performers(players)

        for steam_id, player in players.items():
            insights = []
            name = player["name"]
            stats = player["stats"]
            rating = player["rating"]
            advanced = player["advanced"]
            utility = player["utility"]
            duels = player["duels"]
            entry = player.get("entry", {})
            trades = player.get("trades", {})
            clutches = player.get("clutches", {})
            rws = player.get("rws", {})

            # Detect role with more detail
            role, role_confidence = self._detect_role_detailed(player, match_stats)

            # Determine player identity/archetype (like Leetify's "The Cleanup")
            identity = self._determine_player_identity(player, match_stats, top_performers)

            # ==========================================
            # AIM & MECHANICS INSIGHTS
            # ==========================================

            # Time to Damage (TTD) - reaction time analysis
            ttd = advanced.get("ttd_median_ms", 0)
            if ttd > 0:
                ttd_rating = self._get_ttd_rating(ttd)
                avg_ttd = match_stats.get("avg_ttd", 350)
                diff = ttd - avg_ttd
                if diff > 50:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"TTD {ttd:.0f}ms is {diff:.0f}ms slower than match average ({avg_ttd:.0f}ms) - practice pre-aiming common angles",
                            "category": "Aim",
                            "metric": "ttd_ms",
                            "value": ttd,
                            "benchmark": avg_ttd,
                            "severity": "high" if diff > 100 else "medium",
                        }
                    )
                elif diff < -50:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Fast reactions: {ttd:.0f}ms TTD ({abs(diff):.0f}ms faster than match avg) - {ttd_rating}",
                            "category": "Aim",
                            "metric": "ttd_ms",
                            "value": ttd,
                            "benchmark": avg_ttd,
                        }
                    )

            # Crosshair Placement (CP) - angle accuracy
            cp = advanced.get("cp_median_error_deg", 0)
            if cp > 0:
                cp_rating = self._get_cp_rating(cp)
                avg_cp = match_stats.get("avg_cp", 8)
                diff = cp - avg_cp
                if cp > 10:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Crosshair placement: {cp:.1f}° error ({diff:+.1f}° vs avg). Pre-aim head level at common angles",
                            "category": "Aim",
                            "metric": "cp_error_deg",
                            "value": cp,
                            "benchmark": avg_cp,
                            "severity": "high" if cp > 15 else "medium",
                        }
                    )
                elif cp < 5:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Excellent crosshair placement: {cp:.1f}° error - {cp_rating}",
                            "category": "Aim",
                            "metric": "cp_error_deg",
                            "value": cp,
                        }
                    )

            # Headshot percentage analysis
            hs_pct = stats.get("headshot_pct", 0)
            if hs_pct > 0:
                avg_hs = match_stats.get("avg_hs_pct", 35)
                if hs_pct >= 50:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Elite headshot %: {hs_pct:.0f}% ({hs_pct - avg_hs:+.0f}% vs match avg) - precision aiming",
                            "category": "Aim",
                            "metric": "headshot_pct",
                            "value": hs_pct,
                        }
                    )
                elif hs_pct < 25:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Low HS%: {hs_pct:.0f}% (avg: {avg_hs:.0f}%). Focus on head-level crosshair, less spraying",
                            "category": "Aim",
                            "metric": "headshot_pct",
                            "value": hs_pct,
                            "benchmark": avg_hs,
                        }
                    )

            # ==========================================
            # OPENING DUEL / ENTRY INSIGHTS
            # ==========================================

            entry_attempts = entry.get("entry_attempts", 0)
            entry_kills = entry.get("entry_kills", 0)
            entry_deaths = entry.get("entry_deaths", 0)
            entry_success = entry.get("entry_success_pct", 0)

            if entry_attempts >= 3:
                avg_entry_success = match_stats.get("avg_entry_success", 50)

                if entry_success < 35:
                    insights.append(
                        {
                            "type": "mistake",
                            "message": f"Opening duels: {entry_success:.0f}% success ({entry_kills}W-{entry_deaths}L). Use utility before peeking or change entry spots",
                            "category": "Entry",
                            "metric": "opening_duel_success",
                            "value": entry_success,
                            "benchmark": avg_entry_success,
                            "severity": "high",
                        }
                    )
                elif entry_success >= 65:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Dominant entry fragging: {entry_success:.0f}% ({entry_kills}W-{entry_deaths}L) - {entry_success - avg_entry_success:+.0f}% vs match avg",
                            "category": "Entry",
                            "metric": "opening_duel_success",
                            "value": entry_success,
                        }
                    )

                # Entry attempt rate analysis
                entry_rate = entry.get("entry_attempts_pct", 0)
                if role == "entry" and entry_rate < 20:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Low entry rate for entry role: {entry_rate:.0f}% of rounds. Lead more site takes with utility",
                            "category": "Role",
                            "metric": "entry_attempt_rate",
                            "value": entry_rate,
                        }
                    )

            # ==========================================
            # TRADE KILL INSIGHTS
            # ==========================================

            trade_kills = trades.get("trade_kills", 0) or duels.get("trade_kills", 0)
            deaths_traded = trades.get("deaths_traded", 0) or duels.get("traded_deaths", 0)
            total_deaths = stats.get("deaths", 1)

            trade_rate = (deaths_traded / max(1, total_deaths)) * 100 if total_deaths > 0 else 0
            avg_trade_rate = match_stats.get("avg_trade_rate", 40)

            # Find best trader on team for comparison
            best_trader = top_performers.get("trade_kills", {})

            if trade_kills >= 5:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"Strong trader: {trade_kills} trade kills - reliable teammate support",
                        "category": "Trading",
                        "metric": "trade_kills",
                        "value": trade_kills,
                    }
                )

            if total_deaths >= 8 and trade_rate < 30:
                best_trader_name = best_trader.get("name", "teammate")
                insights.append(
                    {
                        "type": "warning",
                        "message": f"Only {trade_rate:.0f}% of deaths traded ({deaths_traded}/{total_deaths}). Stay closer to teammates, especially {best_trader_name}",
                        "category": "Trading",
                        "metric": "deaths_traded_rate",
                        "value": trade_rate,
                        "benchmark": avg_trade_rate,
                        "severity": "medium",
                    }
                )

            # ==========================================
            # CLUTCH INSIGHTS
            # ==========================================

            clutch_wins = clutches.get("clutch_wins", 0) or duels.get("clutch_wins", 0)
            clutch_attempts = clutches.get("clutch_wins", 0) + clutches.get("clutch_losses", 0)
            if clutch_attempts == 0:
                clutch_attempts = duels.get("clutch_attempts", 0)

            if clutch_attempts >= 3:
                clutch_pct = clutches.get("clutch_success_pct", 0)
                if clutch_pct == 0 and clutch_attempts > 0:
                    clutch_pct = (clutch_wins / clutch_attempts) * 100

                # Check for impressive clutches (1v3, 1v4, 1v5)
                v3_plus = (
                    clutches.get("v3_wins", 0)
                    + clutches.get("v4_wins", 0)
                    + clutches.get("v5_wins", 0)
                )

                if v3_plus >= 1:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Clutch master: Won 1v3+ situations {v3_plus} time(s) - ice cold under pressure",
                            "category": "Clutch",
                            "metric": "difficult_clutches",
                            "value": v3_plus,
                        }
                    )
                elif clutch_pct >= 40 and clutch_attempts >= 3:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Reliable clutcher: {clutch_pct:.0f}% success rate ({clutch_wins}/{clutch_attempts})",
                            "category": "Clutch",
                            "metric": "clutch_success",
                            "value": clutch_pct,
                        }
                    )
                elif clutch_pct < 20 and clutch_attempts >= 4:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Clutch struggles: {clutch_pct:.0f}% ({clutch_wins}/{clutch_attempts}). Play for info, use utility to isolate fights",
                            "category": "Clutch",
                            "metric": "clutch_success",
                            "value": clutch_pct,
                            "severity": "medium",
                        }
                    )

            # ==========================================
            # UTILITY USAGE INSIGHTS
            # ==========================================

            flashes = utility.get("flashbangs_thrown", 0)
            smokes = utility.get("smokes_thrown", 0)
            he_thrown = utility.get("he_thrown", 0)
            molotovs = utility.get("molotovs_thrown", 0)
            total_util = flashes + smokes + he_thrown + molotovs
            enemies_flashed = utility.get("enemies_flashed", 0)
            flash_assists = utility.get("flash_assists", 0)
            he_damage = utility.get("he_damage", 0)

            rounds_played = rws.get("rounds_played", 0) or 30  # Approximate if missing
            util_per_round = total_util / max(1, rounds_played)
            avg_util_per_round = match_stats.get("avg_util_per_round", 2.5)

            if flashes > 0:
                flash_effectiveness = (enemies_flashed / max(1, flashes)) * 100
                if flash_effectiveness < 30 and flashes >= 5:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Flash effectiveness: {flash_effectiveness:.0f}% ({enemies_flashed} blinds from {flashes} flashes). Learn pop-flashes for common angles",
                            "category": "Utility",
                            "metric": "flash_effectiveness",
                            "value": flash_effectiveness,
                            "severity": "medium",
                        }
                    )
                elif flash_effectiveness >= 60 and flash_assists >= 2:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Quality flashbangs: {enemies_flashed} enemies blinded, {flash_assists} flash assists - great support play",
                            "category": "Utility",
                            "metric": "flash_effectiveness",
                            "value": flash_effectiveness,
                        }
                    )

            if util_per_round < 1.5 and rounds_played >= 15:
                insights.append(
                    {
                        "type": "warning",
                        "message": f"Low utility usage: {util_per_round:.1f}/round (avg: {avg_util_per_round:.1f}). Buy and use more grenades",
                        "category": "Utility",
                        "metric": "utility_per_round",
                        "value": util_per_round,
                        "benchmark": avg_util_per_round,
                    }
                )

            if he_damage >= 150:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"Effective HE grenades: {he_damage} damage total - valuable chip damage",
                        "category": "Utility",
                        "metric": "he_damage",
                        "value": he_damage,
                    }
                )

            # ==========================================
            # IMPACT & DAMAGE INSIGHTS
            # ==========================================

            adr = stats.get("adr", 0)
            avg_adr = match_stats.get("avg_adr", 75)
            kast = rating.get("kast_percentage", 0)
            hltv = rating.get("hltv_rating", 0)
            kd_ratio = stats.get("kd_ratio", 1)

            # Multi-kill analysis
            multikills = stats.get("3k", 0) + stats.get("4k", 0) + stats.get("5k", 0)

            if adr >= 90:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"High impact: {adr:.0f} ADR ({adr - avg_adr:+.0f} vs avg) - consistent round damage",
                        "category": "Impact",
                        "metric": "adr",
                        "value": adr,
                    }
                )
            elif adr < 60:
                insights.append(
                    {
                        "type": "warning",
                        "message": f"Low ADR: {adr:.0f} (avg: {avg_adr:.0f}). Find more engagements, use utility to create opportunities",
                        "category": "Impact",
                        "metric": "adr",
                        "value": adr,
                        "benchmark": avg_adr,
                        "severity": "high" if adr < 50 else "medium",
                    }
                )

            if multikills >= 3:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"Round-winning plays: {multikills} multi-kills (3K+) - clutch performer in key rounds",
                        "category": "Impact",
                        "metric": "multikills",
                        "value": multikills,
                    }
                )

            if kast < 55 and rounds_played >= 15:
                insights.append(
                    {
                        "type": "warning",
                        "message": f"Low KAST: {kast:.0f}% - dying without impact too often. Focus on trading and staying alive",
                        "category": "Consistency",
                        "metric": "kast",
                        "value": kast,
                        "severity": "high",
                    }
                )
            elif kast >= 80:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"Exceptional consistency: {kast:.0f}% KAST - contributing in nearly every round",
                        "category": "Consistency",
                        "metric": "kast",
                        "value": kast,
                    }
                )

            # ==========================================
            # RWS (Round Win Share) INSIGHTS
            # ==========================================

            avg_rws = rws.get("avg_rws", 0)
            if avg_rws > 0:
                if avg_rws >= 12:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"High impact in won rounds: {avg_rws:.1f} RWS - key contributor to team victories",
                            "category": "Impact",
                            "metric": "rws",
                            "value": avg_rws,
                        }
                    )
                elif avg_rws < 6:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Low RWS: {avg_rws:.1f} - limited impact in rounds your team wins. Be more aggressive in won rounds",
                            "category": "Impact",
                            "metric": "rws",
                            "value": avg_rws,
                        }
                    )

            # ==========================================
            # ROLE-SPECIFIC INSIGHTS
            # ==========================================

            if role == "entry":
                if entry_success < 45 and entry_attempts >= 4:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Entry role struggling ({entry_success:.0f}% success). Coordinate utility with teammates before taking fights",
                            "category": "Role",
                            "severity": "medium",
                        }
                    )
            elif role == "support":
                if flash_assists < 2 and enemies_flashed < 5:
                    insights.append(
                        {
                            "type": "warning",
                            "message": "Support role needs more flash impact. Focus on enabling teammates with utility",
                            "category": "Role",
                        }
                    )
            elif role == "awp":
                awp_kd = kd_ratio  # Simplified - ideally would track AWP-specific K/D
                if awp_kd < 1.0:
                    insights.append(
                        {
                            "type": "warning",
                            "message": "AWPer dying too often - hold angles, avoid aggressive peeks. $4750 value at risk each death",
                            "category": "Role",
                            "severity": "high",
                        }
                    )

            # ==========================================
            # SORT AND PRIORITIZE INSIGHTS
            # ==========================================

            # Sort by severity and type (mistakes first, then warnings, then positives)
            type_priority = {"mistake": 0, "warning": 1, "positive": 2}
            severity_priority = {"high": 0, "medium": 1, "low": 2}

            insights.sort(
                key=lambda x: (
                    type_priority.get(x.get("type"), 2),
                    severity_priority.get(x.get("severity", "low"), 2),
                )
            )

            # Ensure at least one insight
            if not insights:
                if hltv >= 1.0:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Solid match performance: {hltv:.2f} rating, {kast:.0f}% KAST",
                            "category": "Overall",
                        }
                    )
                else:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Below average match ({hltv:.2f} rating). Focus on survival and trading with teammates",
                            "category": "Overall",
                        }
                    )

            coaching.append(
                {
                    "player_name": name,
                    "steam_id": steam_id,
                    "role": role,
                    "role_confidence": role_confidence,
                    "identity": identity,
                    "stats_summary": {
                        "rating": round(hltv, 2),
                        "adr": round(adr, 0),
                        "kast": round(kast, 0),
                        "kd": round(kd_ratio, 2),
                        "ttd_ms": round(advanced.get("ttd_median_ms", 0), 0),
                        "cp_deg": round(advanced.get("cp_median_error_deg", 0), 1),
                        "entry_success": round(entry_success, 0),
                        "trade_rate": round(trade_rate, 0),
                        "clutch_pct": round(clutches.get("clutch_success_pct", 0), 0),
                    },
                    "insights": insights[:8],  # Top 8 insights per player
                }
            )

        return coaching

    def _calculate_match_averages(self, players: dict) -> dict:
        """Calculate match-wide statistics for comparison benchmarks."""
        if not players:
            return {}

        ttd_values = []
        cp_values = []
        adr_values = []
        hs_values = []
        entry_success_values = []
        trade_rates = []
        util_per_round_values = []

        for p in players.values():
            ttd = p.get("advanced", {}).get("ttd_median_ms", 0)
            if ttd > 0:
                ttd_values.append(ttd)

            cp = p.get("advanced", {}).get("cp_median_error_deg", 0)
            if cp > 0:
                cp_values.append(cp)

            adr = p.get("stats", {}).get("adr", 0)
            if adr > 0:
                adr_values.append(adr)

            hs = p.get("stats", {}).get("headshot_pct", 0)
            if hs > 0:
                hs_values.append(hs)

            entry_success = p.get("entry", {}).get("entry_success_pct", 0)
            entry_attempts = p.get("entry", {}).get("entry_attempts", 0)
            if entry_attempts >= 2:
                entry_success_values.append(entry_success)

            deaths = p.get("stats", {}).get("deaths", 0)
            traded = p.get("trades", {}).get("deaths_traded", 0) or p.get("duels", {}).get(
                "traded_deaths", 0
            )
            if deaths >= 5:
                trade_rates.append((traded / deaths) * 100)

            util = p.get("utility", {})
            total_util = sum(
                [
                    util.get("flashbangs_thrown", 0),
                    util.get("smokes_thrown", 0),
                    util.get("he_thrown", 0),
                    util.get("molotovs_thrown", 0),
                ]
            )
            rounds = p.get("rws", {}).get("rounds_played", 0) or 30
            if rounds > 0:
                util_per_round_values.append(total_util / rounds)

        return {
            "avg_ttd": sum(ttd_values) / len(ttd_values) if ttd_values else 350,
            "avg_cp": sum(cp_values) / len(cp_values) if cp_values else 8,
            "avg_adr": sum(adr_values) / len(adr_values) if adr_values else 75,
            "avg_hs_pct": sum(hs_values) / len(hs_values) if hs_values else 35,
            "avg_entry_success": (
                sum(entry_success_values) / len(entry_success_values)
                if entry_success_values
                else 50
            ),
            "avg_trade_rate": (sum(trade_rates) / len(trade_rates) if trade_rates else 40),
            "avg_util_per_round": (
                sum(util_per_round_values) / len(util_per_round_values)
                if util_per_round_values
                else 2.5
            ),
        }

    def _find_top_performers(self, players: dict) -> dict:
        """Find top performers in each stat category for comparative insights."""
        top = {}

        # Best trader
        best_trade_kills = 0
        for sid, p in players.items():
            tk = p.get("trades", {}).get("trade_kills", 0) or p.get("duels", {}).get(
                "trade_kills", 0
            )
            if tk > best_trade_kills:
                best_trade_kills = tk
                top["trade_kills"] = {"steam_id": sid, "name": p["name"], "value": tk}

        # Best entry
        best_entry_pct = 0
        for sid, p in players.items():
            entry_pct = p.get("entry", {}).get("entry_success_pct", 0)
            entry_attempts = p.get("entry", {}).get("entry_attempts", 0)
            if entry_attempts >= 3 and entry_pct > best_entry_pct:
                best_entry_pct = entry_pct
                top["entry"] = {"steam_id": sid, "name": p["name"], "value": entry_pct}

        # Best clutcher
        best_clutch_pct = 0
        for sid, p in players.items():
            clutch_pct = p.get("clutches", {}).get("clutch_success_pct", 0)
            clutch_attempts = p.get("clutches", {}).get("clutch_wins", 0) + p.get(
                "clutches", {}
            ).get("clutch_losses", 0)
            if clutch_attempts >= 2 and clutch_pct > best_clutch_pct:
                best_clutch_pct = clutch_pct
                top["clutch"] = {
                    "steam_id": sid,
                    "name": p["name"],
                    "value": clutch_pct,
                }

        # Best aim (lowest CP error)
        best_cp = 999
        for sid, p in players.items():
            cp = p.get("advanced", {}).get("cp_median_error_deg", 0)
            if 0 < cp < best_cp:
                best_cp = cp
                top["aim"] = {"steam_id": sid, "name": p["name"], "value": cp}

        return top

    def _get_ttd_rating(self, ttd_ms: float) -> str:
        """Get descriptive rating for TTD (Time to Damage)."""
        if ttd_ms < 200:
            return "Elite reaction time"
        elif ttd_ms < 300:
            return "Fast reactions"
        elif ttd_ms < 400:
            return "Average reactions"
        elif ttd_ms < 500:
            return "Slow reactions"
        else:
            return "Very slow - needs work"

    def _get_cp_rating(self, cp_deg: float) -> str:
        """Get descriptive rating for Crosshair Placement."""
        if cp_deg < 3:
            return "Pro-level placement"
        elif cp_deg < 6:
            return "Excellent placement"
        elif cp_deg < 10:
            return "Good placement"
        elif cp_deg < 15:
            return "Average placement"
        else:
            return "Needs improvement"

    def _detect_role_detailed(self, player: dict, match_stats: dict) -> tuple[str, str]:
        """
        Detect player role with confidence level using behavioral scoring.

        Uses the unified Role Scoring Engine approach:
        1. AWPer check (35%+ kills with sniper) - overrides everything
        2. Entry score (high opening attempts + first contact patterns)
        3. Support score (high utility + low first contact)
        4. Lurker score (high isolation + impact)
        5. Rifler (default high-frag role)

        Returns (role, confidence) where confidence is 'high', 'medium', or 'low'.
        """
        advanced = player.get("advanced", {})
        utility = player.get("utility", {})
        stats = player.get("stats", {})
        entry = player.get("entry", {})
        trades = player.get("trades", {})
        weapons = player.get("weapons", {})

        # Extract metrics
        kills = stats.get("kills", 0)
        deaths = stats.get("deaths", 0)
        rounds_played = stats.get("rounds_played", 1) or 1
        hs_pct = stats.get("headshot_pct", 0)
        adr = stats.get("adr", 0)

        entry_attempts = entry.get("entry_attempts", 0)
        entry_kills = entry.get("entry_kills", 0)
        entry_success_pct = entry.get("entry_success_pct", 0)

        flashes_thrown = utility.get("flashbangs_thrown", 0)
        smokes_thrown = utility.get("smokes_thrown", 0)
        he_thrown = utility.get("he_thrown", 0)
        molotovs_thrown = utility.get("molotovs_thrown", 0)
        flash_assists = utility.get("flash_assists", 0)
        effective_flashes = utility.get("effective_flashes", 0)

        untraded_deaths = (
            trades.get("untraded_deaths", 0) or advanced.get("untraded_deaths", 0) or 0
        )

        # AWP kills from weapon breakdown
        awp_kills = weapons.get("awp", 0) + weapons.get("AWP", 0)
        ssg_kills = weapons.get("ssg08", 0) + weapons.get("SSG08", 0)
        sniper_kills = awp_kills + ssg_kills

        # Score each role (0-100 scale)
        scores = {
            "entry": 0.0,
            "support": 0.0,
            "rifler": 0.0,
            "awper": 0.0,
            "lurker": 0.0,
        }

        # =====================================================================
        # STEP 1: AWPer Detection (Highest Priority)
        # If 35%+ of kills are with AWP/SSG08, this defines the player's role
        # =====================================================================
        if kills > 0:
            awp_kill_pct = (sniper_kills / kills) * 100
            if awp_kill_pct >= 35:
                scores["awper"] = 85.0 + min(awp_kill_pct - 35, 15)  # 85-100

        # =====================================================================
        # STEP 2: Entry Score
        # Based on: opening duel attempts (NOT just kills), aggression patterns
        # =====================================================================
        entry_score = 0.0

        # Opening duel ATTEMPTS (shows aggression, not just success)
        attempts_per_round = entry_attempts / rounds_played
        entry_score += min(attempts_per_round / 0.3, 1.0) * 40  # Up to 40 points

        # Entry success rate (reward winning, not just attempting)
        if entry_attempts >= 3:
            entry_score += min(entry_success_pct / 50, 1.0) * 25  # Up to 25 points

        # Opening kills as proxy for first contact
        opening_kill_rate = entry_kills / rounds_played * 100
        entry_score += min(opening_kill_rate / 25, 1.0) * 20  # Up to 20 points

        # Bonus: High effective flashes (utility-supported entries)
        if effective_flashes > 3:
            entry_score += 10

        scores["entry"] = min(entry_score, 100)

        # =====================================================================
        # STEP 3: Support Score
        # Based on: utility effectiveness, flash assists, passive positioning
        # =====================================================================
        support_score = 0.0

        # Effective flashes (shows intentional team support)
        support_score += min(effective_flashes / 8, 1.0) * 25  # Up to 25 points

        # Flash assists (direct team contribution)
        support_score += min(flash_assists / 4, 1.0) * 25  # Up to 25 points

        # Total utility usage
        total_utility = flashes_thrown + smokes_thrown + he_thrown + molotovs_thrown
        utility_per_round = total_utility / rounds_played
        support_score += min(utility_per_round / 3, 1.0) * 20  # Up to 20 points

        # LOW entry attempts = passive player (support indicator)
        if attempts_per_round < 0.15:
            support_score += 15

        # Trade attempt rate (being in position to support)
        trade_opps = trades.get("trade_kill_opportunities", 0)
        trade_attempts = trades.get("trade_kill_attempts", 0)
        if trade_opps > 0:
            trade_attempt_rate = trade_attempts / trade_opps
            support_score += trade_attempt_rate * 15  # Up to 15 points

        scores["support"] = min(support_score, 100)

        # =====================================================================
        # STEP 4: Lurker Score
        # Based on: high isolation (untraded deaths), but WITH IMPACT
        # =====================================================================
        lurker_score = 0.0

        # Isolation indicator: untraded deaths
        if deaths > 3:
            isolation_rate = untraded_deaths / deaths
            if isolation_rate > 0.5:
                lurker_score += (isolation_rate - 0.5) * 2 * 35  # Up to 35 points

        # LOW entry attempts = not taking first contact
        if entry_attempts <= 2 and rounds_played >= 10:
            lurker_score += 20

        # CRITICAL: Must have IMPACT to be a lurker, not a feeder
        kpr = kills / rounds_played
        hltv_rating = stats.get("hltv_rating", 0) or player.get("rating", {}).get("hltv_rating", 0)
        has_impact = kpr >= 0.5 or hltv_rating >= 0.9

        if has_impact:
            lurker_score += 25  # Impact bonus
        elif lurker_score > 30:
            lurker_score *= 0.3  # Penalize if no impact (feeding, not lurking)

        # Multi-kills (lurkers often catch rotations)
        multikills = (
            stats.get("2k", 0) + stats.get("3k", 0) + stats.get("4k", 0) + stats.get("5k", 0)
        )
        if multikills >= 3:
            lurker_score += 15

        scores["lurker"] = min(lurker_score, 100)

        # =====================================================================
        # STEP 5: Rifler Score (Default High-Frag Role)
        # Based on: high kills, high HS%, consistent damage
        # =====================================================================
        rifler_score = 0.0

        # Kill rate
        rifler_score += min(kpr / 0.8, 1.0) * 30  # Up to 30 points for 0.8+ KPR

        # ADR (consistent damage)
        rifler_score += min(adr / 85, 1.0) * 25  # Up to 25 points

        # Headshot percentage
        rifler_score += min(hs_pct / 50, 1.0) * 20  # Up to 20 points

        # Multi-kills
        mk_score = (
            multikills
            if "multikills" in dir()
            else (
                stats.get("2k", 0)
                + stats.get("3k", 0) * 2
                + stats.get("4k", 0) * 3
                + stats.get("5k", 0) * 4
            )
        )
        rifler_score += min(mk_score / 8, 1.0) * 15  # Up to 15 points

        # HLTV rating bonus
        if hltv_rating >= 1.15:
            rifler_score += 10

        scores["rifler"] = min(rifler_score, 100)

        # =====================================================================
        # Determine primary role
        # =====================================================================
        best_role = max(scores, key=scores.get)
        best_score = scores[best_role]

        # Confidence based on score
        if best_score >= 70:
            confidence = "high"
        elif best_score >= 40:
            confidence = "medium"
        else:
            confidence = "low"

        # Check for tie (within 10% of each other) - default to flex
        second_best_score = sorted(scores.values(), reverse=True)[1]
        if best_score > 0 and best_score < 20:
            best_role = "flex"
            confidence = "low"
        elif best_score > 0 and (best_score - second_best_score) / best_score < 0.10:
            best_role = "flex"
            confidence = "medium"

        return best_role, confidence

    def _determine_player_identity(
        self, player: dict, match_stats: dict, top_performers: dict
    ) -> dict:
        """
        Determine player identity/archetype like Leetify's system.
        Returns identity name and top stats.
        """
        stats = player.get("stats", {})
        advanced = player.get("advanced", {})
        rating = player.get("rating", {})
        entry = player.get("entry", {})
        utility = player.get("utility", {})
        clutches = player.get("clutches", {})
        trades = player.get("trades", {})

        # Collect notable stats
        notable_stats = []

        # ADR
        adr = stats.get("adr", 0)
        if adr >= 85:
            notable_stats.append(("damage_dealer", adr, f"{adr:.0f} ADR"))

        # Entry kills
        entry_kills = entry.get("entry_kills", 0)
        if entry_kills >= 5:
            notable_stats.append(("entry_fragger", entry_kills, f"{entry_kills} Opening Kills"))

        # Clutches
        clutch_wins = clutches.get("clutch_wins", 0)
        if clutch_wins >= 2:
            notable_stats.append(("clutch_player", clutch_wins, f"{clutch_wins} Clutches Won"))

        # Trade kills
        trade_kills = trades.get("trade_kills", 0) or player.get("duels", {}).get("trade_kills", 0)
        if trade_kills >= 5:
            notable_stats.append(("team_player", trade_kills, f"{trade_kills} Trade Kills"))

        # Flash assists
        flash_assists = utility.get("flash_assists", 0)
        if flash_assists >= 3:
            notable_stats.append(
                ("support_master", flash_assists, f"{flash_assists} Flash Assists")
            )

        # HS%
        hs_pct = stats.get("headshot_pct", 0)
        if hs_pct >= 50:
            notable_stats.append(("headshot_machine", hs_pct, f"{hs_pct:.0f}% HS"))

        # Multi-kills
        multikills = stats.get("3k", 0) + stats.get("4k", 0) + stats.get("5k", 0)
        if multikills >= 3:
            notable_stats.append(("round_winner", multikills, f"{multikills} Multi-kills"))

        # KAST
        kast = rating.get("kast_percentage", 0)
        if kast >= 80:
            notable_stats.append(("consistent", kast, f"{kast:.0f}% KAST"))

        # TTD (lower is better)
        ttd = advanced.get("ttd_median_ms", 0)
        if 0 < ttd < 250:
            notable_stats.append(("fast_reactions", 1000 - ttd, f"{ttd:.0f}ms TTD"))

        # CP (lower is better)
        cp = advanced.get("cp_median_error_deg", 0)
        if 0 < cp < 5:
            notable_stats.append(("precise_aim", 100 - cp, f"{cp:.1f}° CP"))

        # Sort by value (highest first) and pick top identity
        notable_stats.sort(key=lambda x: x[1], reverse=True)

        # Identity mapping
        identity_names = {
            "damage_dealer": "The Damage Dealer",
            "entry_fragger": "The Entry Fragger",
            "clutch_player": "The Clutch Master",
            "team_player": "The Team Player",
            "support_master": "The Support",
            "headshot_machine": "The Headshot Machine",
            "round_winner": "The Round Winner",
            "consistent": "The Consistent One",
            "fast_reactions": "The Reactor",
            "precise_aim": "The Precise",
        }

        if notable_stats:
            top_identity = notable_stats[0][0]
            return {
                "name": identity_names.get(top_identity, "The Player"),
                "top_stats": [{"label": s[2], "category": s[0]} for s in notable_stats[:5]],
            }
        else:
            return {
                "name": "The Contributor",
                "top_stats": [],
            }

    def _detect_role(self, player: dict) -> str:
        """
        Detect player role from stats using behavioral scoring.

        Priority:
        1. AWPer (35%+ kills with sniper)
        2. Entry (high opening attempts)
        3. Support (high utility + flash assists)
        4. Rifler (high kills + HS%)
        5. Flex (default)
        """
        utility = player.get("utility", {})
        stats = player.get("stats", {})
        entry = player.get("entry", {})
        weapons = player.get("weapons", {})

        kills = stats.get("kills", 0)
        rounds_played = stats.get("rounds_played", 1) or 1

        # AWPer check first (highest priority)
        awp_kills = weapons.get("awp", 0) + weapons.get("AWP", 0)
        ssg_kills = weapons.get("ssg08", 0) + weapons.get("SSG08", 0)
        if kills > 0 and (awp_kills + ssg_kills) / kills >= 0.35:
            return "awper"

        # Entry: high opening attempts (behavior, not just kills)
        entry_attempts = entry.get("entry_attempts", 0)
        if entry_attempts >= 4 or (entry_attempts >= 3 and entry.get("entry_success_pct", 0) >= 50):
            return "entry"

        # Support: high utility + flash assists (team contribution)
        flash_assists = utility.get("flash_assists", 0)
        effective_flashes = utility.get("effective_flashes", 0)
        total_utility = (
            utility.get("flashbangs_thrown", 0)
            + utility.get("smokes_thrown", 0)
            + utility.get("he_thrown", 0)
            + utility.get("molotovs_thrown", 0)
        )
        if (
            flash_assists >= 3
            or effective_flashes >= 5
            or (total_utility / rounds_played >= 2.5 and entry_attempts <= 2)
        ):
            return "support"

        # Rifler: high kills + consistent fragging
        if kills >= 15 and stats.get("headshot_pct", 0) >= 40:
            return "rifler"

        return "flex"

    def _get_tactical_summary(self, demo_data, analysis) -> dict:
        """Get tactical analysis summary."""
        try:
            from opensight.analysis.tactical_service import TacticalAnalysisService

            service = TacticalAnalysisService(demo_data)
            summary = service.analyze()

            # Serialize team analysis to dict
            team1_dict = {
                "team_name": summary.team1_analysis.team_name,
                "team_side": summary.team1_analysis.team_side,
                "key_insights": summary.team1_analysis.key_insights,
                "recommendations": summary.team1_analysis.recommendations,
                "strengths": summary.team1_analysis.strengths,
                "weaknesses": summary.team1_analysis.weaknesses,
                "star_player": summary.team1_analysis.star_player,
                "star_player_role": summary.team1_analysis.star_player_role,
                "coordination_score": summary.team1_analysis.coordination_score,
            }

            team2_dict = {
                "team_name": summary.team2_analysis.team_name,
                "team_side": summary.team2_analysis.team_side,
                "key_insights": summary.team2_analysis.key_insights,
                "recommendations": summary.team2_analysis.recommendations,
                "strengths": summary.team2_analysis.strengths,
                "weaknesses": summary.team2_analysis.weaknesses,
                "star_player": summary.team2_analysis.star_player,
                "star_player_role": summary.team2_analysis.star_player_role,
                "coordination_score": summary.team2_analysis.coordination_score,
            }

            return {
                "key_insights": summary.key_insights,
                "t_stats": summary.t_stats,
                "ct_stats": summary.ct_stats,
                "t_executes": summary.t_executes,
                "buy_patterns": summary.buy_patterns,
                "t_strengths": summary.t_strengths,
                "t_weaknesses": summary.t_weaknesses,
                "ct_strengths": summary.ct_strengths,
                "ct_weaknesses": summary.ct_weaknesses,
                "team_recommendations": summary.team_recommendations,
                "practice_drills": summary.practice_drills,
                "team1_analysis": team1_dict,
                "team2_analysis": team2_dict,
            }
        except Exception as e:
            logger.warning(f"Tactical analysis failed: {e}")
            return {
                "key_insights": ["Demo analysis complete"],
                "team_recommendations": ["Review round-by-round for specific improvements"],
            }


# Convenience functions


def get_cached_analysis(demo_path: Path) -> dict | None:
    """Get cached analysis if available."""
    cache = DemoCache()
    return cache.get(demo_path)


def analyze_with_cache(demo_path: Path, force: bool = False) -> dict:
    """Analyze a demo with caching."""
    analyzer = CachedAnalyzer()
    return analyzer.analyze(demo_path, force=force)


def get_cache_stats() -> dict:
    """Get cache statistics."""
    cache = DemoCache()
    return cache.get_stats().to_dict()


def clear_cache() -> None:
    """Clear all cached data."""
    cache = DemoCache()
    cache.clear()
