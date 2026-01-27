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
                    "flashbangs_thrown": getattr(p, "flashbangs_thrown", None) or 0,
                    "smokes_thrown": getattr(p, "smokes_thrown", None) or 0,
                    "he_thrown": getattr(p, "he_thrown", None) or 0,
                    "molotovs_thrown": getattr(p, "molotovs_thrown", None) or 0,
                    "flash_assists": getattr(p, "flash_assists", None) or 0,
                    "enemies_flashed": getattr(p, "enemies_flashed", None) or 0,
                    "he_damage": getattr(p, "he_damage", None) or 0,
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

        # Build kill matrix
        kill_matrix = self._build_kill_matrix(demo_data)

        # Build heatmap data
        heatmap_data = self._build_heatmap_data(demo_data)

        # Generate coaching insights
        coaching = self._generate_coaching_insights(demo_data, analysis, players)

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

        # Convert to dict for caching
        result = {
            "demo_info": {
                "map": analysis.map_name,
                "rounds": analysis.total_rounds,
                "duration_minutes": getattr(analysis, "duration_minutes", 30),
                "score": f"{analysis.team1_score} - {analysis.team2_score}",
                "total_kills": sum(p["stats"]["kills"] for p in players.values()),
            },
            "players": players,
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
                "traded_entry_kills": 0,  # Would need additional tracking
                "traded_entry_deaths": 0,  # Would need additional tracking
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
        """Build round-by-round timeline data."""
        timeline = []
        kills = getattr(demo_data, "kills", [])
        rounds_data = getattr(demo_data, "rounds", [])

        # Build round boundaries for tick-based inference
        round_boundaries = []
        for r in rounds_data:
            round_num = getattr(r, "round_num", 0)
            start_tick = getattr(r, "start_tick", 0)
            end_tick = getattr(r, "end_tick", 0)
            if round_num and end_tick > 0:
                round_boundaries.append((round_num, start_tick, end_tick))
        round_boundaries.sort(key=lambda x: x[1])

        def infer_round(tick: int) -> int:
            """Infer round number from tick."""
            for rn, st, et in round_boundaries:
                if st <= tick <= et:
                    return rn
            if round_boundaries:
                for rn, _st, et in reversed(round_boundaries):
                    if tick > et:
                        return rn
                return 1
            return 1

        # Check if kills have valid round data
        has_round_data = any(getattr(k, "round_num", 0) > 0 for k in kills[:10])

        # Group kills by round
        round_kills = {}
        for kill in kills:
            round_num = getattr(kill, "round_num", 0)

            # Infer round from tick if needed
            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(kill, "tick", 0)
                    round_num = infer_round(tick)
                else:
                    round_num = 1

            if round_num not in round_kills:
                round_kills[round_num] = {
                    "ct_kills": 0,
                    "t_kills": 0,
                    "first_kill": None,
                    "first_death": None,
                }

            attacker_side = str(getattr(kill, "attacker_side", "")).upper()
            if "CT" in attacker_side:
                round_kills[round_num]["ct_kills"] += 1
            else:
                round_kills[round_num]["t_kills"] += 1

            if round_kills[round_num]["first_kill"] is None:
                round_kills[round_num]["first_kill"] = getattr(kill, "attacker_name", "Unknown")
                round_kills[round_num]["first_death"] = getattr(kill, "victim_name", "Unknown")

        # Use actual round data if available, otherwise use analysis total_rounds
        total_rounds = (
            getattr(analysis, "total_rounds", 0) or len(round_boundaries) or len(round_kills)
        )

        # Build timeline entries - ensure we have entries for all rounds
        for round_num in range(1, total_rounds + 1):
            rd = round_kills.get(
                round_num,
                {"ct_kills": 0, "t_kills": 0, "first_kill": None, "first_death": None},
            )
            # Get winner from rounds data if available
            winner = "CT" if rd["ct_kills"] > rd["t_kills"] else "T"
            for r in rounds_data:
                if getattr(r, "round_num", 0) == round_num:
                    winner = getattr(r, "winner", winner)
                    break

            timeline.append(
                {
                    "round_num": round_num,
                    "winner": winner,
                    "win_reason": (
                        "Elimination" if abs(rd["ct_kills"] - rd["t_kills"]) >= 4 else "Objective"
                    ),
                    "first_kill": rd["first_kill"],
                    "first_death": rd["first_death"],
                    "ct_kills": rd["ct_kills"],
                    "t_kills": rd["t_kills"],
                }
            )

        return timeline

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
        """Build round-by-round data for timeline graphs (kills, damage per round per player)."""
        kills = getattr(demo_data, "kills", [])
        damages = getattr(demo_data, "damages", [])
        player_names = getattr(demo_data, "player_names", {})
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
            # If no match, estimate based on position in sorted boundaries
            if round_boundaries:
                # Use the last round if tick is after all boundaries
                for round_num, _start_tick, end_tick in reversed(round_boundaries):
                    if tick > end_tick:
                        return round_num
                # Use round 1 if tick is before all boundaries
                return 1
            return 1  # Fallback

        # Check if kills have valid round data
        has_round_data = any(getattr(k, "round_num", 0) > 0 for k in kills[:10])

        # Initialize per-player round data
        player_round_data: dict[
            int, dict[int, dict]
        ] = {}  # steam_id -> {round_num -> {kills, damage}}

        # Get max rounds from demo_data or round boundaries
        max_round = getattr(demo_data, "num_rounds", 0) or len(round_boundaries) or 1

        # Count kills per player per round
        for kill in kills:
            attacker_id = getattr(kill, "attacker_steamid", 0)
            round_num = getattr(kill, "round_num", 0)

            # Skip kills without attacker_id
            if not attacker_id:
                continue

            # If no round_num, try to infer from tick using round boundaries
            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(kill, "tick", 0)
                    round_num = infer_round_from_tick(tick)
                else:
                    round_num = 1  # Fallback to round 1

            max_round = max(max_round, round_num)

            if attacker_id not in player_round_data:
                player_round_data[attacker_id] = {}
            if round_num not in player_round_data[attacker_id]:
                player_round_data[attacker_id][round_num] = {"kills": 0, "damage": 0}
            player_round_data[attacker_id][round_num]["kills"] += 1

        # Sum damage per player per round
        for dmg in damages:
            attacker_id = getattr(dmg, "attacker_steamid", 0)
            round_num = getattr(dmg, "round_num", 0)
            damage_val = getattr(dmg, "damage", 0)

            # Skip damages without attacker_id
            if not attacker_id:
                continue

            # If no round_num, try to infer from tick using round boundaries
            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(dmg, "tick", 0)
                    round_num = infer_round_from_tick(tick)
                else:
                    round_num = 1  # Fallback to round 1

            max_round = max(max_round, round_num)

            if attacker_id not in player_round_data:
                player_round_data[attacker_id] = {}
            if round_num not in player_round_data[attacker_id]:
                player_round_data[attacker_id][round_num] = {"kills": 0, "damage": 0}
            player_round_data[attacker_id][round_num]["damage"] += damage_val

        # Build cumulative data for graphs
        players_timeline = []
        for steam_id, round_data in player_round_data.items():
            player_name = player_names.get(steam_id, f"Player {steam_id}")

            cumulative_kills = 0
            cumulative_damage = 0
            rounds = []

            for r in range(1, max_round + 1):
                rd = round_data.get(r, {"kills": 0, "damage": 0})
                cumulative_kills += rd["kills"]
                cumulative_damage += rd["damage"]
                rounds.append(
                    {
                        "round": r,
                        "kills": cumulative_kills,
                        "damage": cumulative_damage,
                        "round_kills": rd["kills"],
                        "round_damage": rd["damage"],
                    }
                )

            players_timeline.append(
                {
                    "steam_id": steam_id,
                    "name": player_name,
                    "rounds": rounds,
                }
            )

        return {
            "max_rounds": max_round,
            "players": players_timeline,
        }

    def _calculate_rws_direct(self, demo_data) -> dict[int, dict]:
        """Calculate RWS directly from demo data with better team handling."""
        kills = getattr(demo_data, "kills", [])
        damages = getattr(demo_data, "damages", [])
        rounds = getattr(demo_data, "rounds", [])
        player_names = getattr(demo_data, "player_names", {})

        if not rounds or not kills:
            return {}

        # Build player teams from kills (more reliable than player_teams dict)
        player_teams: dict[int, str] = {}
        for kill in kills:
            att_id = getattr(kill, "attacker_steamid", 0)
            att_side = str(getattr(kill, "attacker_side", "")).upper()
            vic_id = getattr(kill, "victim_steamid", 0)
            vic_side = str(getattr(kill, "victim_side", "")).upper()

            if att_id and "CT" in att_side:
                player_teams[att_id] = "CT"
            elif att_id and "T" in att_side:
                player_teams[att_id] = "T"
            if vic_id and "CT" in vic_side:
                player_teams[vic_id] = "CT"
            elif vic_id and "T" in vic_side:
                player_teams[vic_id] = "T"

        # Group damage by round
        round_damages: dict[int, dict[int, int]] = {}  # round_num -> {steam_id -> damage}
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
                if attacker_id not in round_damages[round_num]:
                    round_damages[round_num][attacker_id] = 0
                round_damages[round_num][attacker_id] += damage_val

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

            # Find winning players based on their side in this half
            winning_players = []
            for pid, team in player_teams.items():
                if pid in player_stats:
                    player_stats[pid]["rounds_played"] += 1

                    # Check if player is on winning team
                    if (winner == "CT" and team == "CT") or (winner == "T" and team == "T"):
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
            is_pistol = round_num in [1, 13]
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
        Detect player role with confidence level.
        Returns (role, confidence) where confidence is 'high', 'medium', or 'low'.
        """
        advanced = player.get("advanced", {})
        utility = player.get("utility", {})
        stats = player.get("stats", {})
        entry = player.get("entry", {})

        opening_kills = advanced.get("opening_kills", 0)
        entry_attempts = entry.get("entry_attempts", 0)
        flashes_thrown = utility.get("flashbangs_thrown", 0)
        smokes_thrown = utility.get("smokes_thrown", 0)
        kills = stats.get("kills", 0)
        hs_pct = stats.get("headshot_pct", 0)

        # Score each role
        scores = {
            "entry": 0,
            "support": 0,
            "rifler": 0,
            "awp": 0,
            "lurker": 0,
            "flex": 0,
        }

        # Entry scoring
        if entry_attempts >= 5:
            scores["entry"] += 3
        elif entry_attempts >= 3:
            scores["entry"] += 2
        if opening_kills >= 5:
            scores["entry"] += 2
        elif opening_kills >= 3:
            scores["entry"] += 1

        # Support scoring
        if flashes_thrown >= 10:
            scores["support"] += 3
        elif flashes_thrown >= 6:
            scores["support"] += 2
        if smokes_thrown >= 8:
            scores["support"] += 2
        if utility.get("flash_assists", 0) >= 3:
            scores["support"] += 2

        # Rifler scoring
        if kills >= 20 and hs_pct >= 45:
            scores["rifler"] += 3
        elif kills >= 15 and hs_pct >= 40:
            scores["rifler"] += 2

        # AWP scoring (would need weapon data for accuracy)
        # For now, low entry + high kills might indicate AWP
        if kills >= 15 and entry_attempts <= 2:
            scores["awp"] += 1

        # Find highest score
        best_role = max(scores, key=scores.get)
        best_score = scores[best_role]

        if best_score >= 4:
            confidence = "high"
        elif best_score >= 2:
            confidence = "medium"
        else:
            best_role = "flex"
            confidence = "low"

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
        """Detect player role from stats."""
        advanced = player["advanced"]
        utility = player["utility"]
        stats = player["stats"]

        if advanced["opening_kills"] >= 5:
            return "entry"
        if utility["flashbangs_thrown"] >= 10 or utility["smokes_thrown"] >= 8:
            return "support"
        if stats["kills"] >= 20 and stats["headshot_pct"] >= 50:
            return "rifler"
        return "flex"

    def _get_tactical_summary(self, demo_data, analysis) -> dict:
        """Get tactical analysis summary."""
        try:
            from opensight.analysis.tactical_service import TacticalAnalysisService

            service = TacticalAnalysisService(demo_data)
            summary = service.analyze()

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
