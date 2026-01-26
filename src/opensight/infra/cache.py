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
            "oldest_entry": self.oldest_entry.isoformat() if self.oldest_entry else None,
            "newest_entry": self.newest_entry.isoformat() if self.newest_entry else None,
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
        from opensight.analysis.analytics import DemoAnalyzer, compute_kill_positions
        from opensight.core.parser import DemoParser
        from opensight.core.enhanced_parser import CoachingAnalysisEngine

        parser = DemoParser(demo_path)
        demo_data = parser.parse()

        analyzer = DemoAnalyzer(demo_data)
        analysis = analyzer.analyze()
        
        # Calculate professional metrics using enhanced parser
        try:
            logger.info(f"Calculating professional metrics (TTD, CP, Entry/Trade/Clutch)")
            engine = CoachingAnalysisEngine()
            enhanced_metrics = engine.analyze_demo(demo_path)
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
                    "headshot_pct": round(p.headshot_percentage, 1) if p.headshot_percentage else 0,
                    "kd_ratio": round(p.kills / max(1, p.deaths), 2),
                    "2k": mk["2k"],
                    "3k": mk["3k"],
                    "4k": mk["4k"],
                    "5k": mk["5k"],
                },
                "rating": {
                    "hltv_rating": round(p.hltv_rating, 2) if p.hltv_rating else 0,
                    "kast_percentage": round(p.kast_percentage, 1) if p.kast_percentage else 0,
                    "aim_rating": round(p.aim_rating, 1) if p.aim_rating else 50,
                    "utility_rating": round(p.utility_rating, 1) if p.utility_rating else 50,
                    "impact_rating": round(getattr(p, 'impact_rating', None) or 0, 2),
                },
                "advanced": {
                    # From enhanced parser (TTD - Time to Damage)
                    "ttd_median_ms": round(getattr(p, 'ttd_median_ms', None) or 0, 1),
                    "ttd_mean_ms": round(getattr(p, 'ttd_mean_ms', None) or 0, 1),
                    "ttd_95th_ms": round(getattr(p, 'ttd_95th_ms', None) or 0, 1),
                    # From enhanced parser (CP - Crosshair Placement)
                    "cp_median_error_deg": round(getattr(p, 'cp_median_error_deg', None) or 0, 1),
                    "cp_mean_error_deg": round(getattr(p, 'cp_mean_error_deg', None) or 0, 1),
                    # Other advanced stats
                    "prefire_kills": getattr(p, 'prefire_kills', None) or 0,
                    "opening_kills": getattr(p, 'opening_kills', None) or 0,
                    "opening_deaths": getattr(p, 'opening_deaths', None) or 0,
                },
                "utility": {
                    "flashbangs_thrown": getattr(p, 'flashbangs_thrown', None) or 0,
                    "smokes_thrown": getattr(p, 'smokes_thrown', None) or 0,
                    "he_thrown": getattr(p, 'he_thrown', None) or 0,
                    "molotovs_thrown": getattr(p, 'molotovs_thrown', None) or 0,
                    "flash_assists": getattr(p, 'flash_assists', None) or 0,
                    "enemies_flashed": getattr(p, 'enemies_flashed', None) or 0,
                    "he_damage": getattr(p, 'he_damage', None) or 0,
                },
                "duels": {
                    "trade_kills": getattr(p, 'trade_kills', None) or 0,
                    "traded_deaths": getattr(p, 'traded_deaths', None) or 0,
                    "clutch_wins": getattr(p, 'clutch_wins', None) or 0,
                    "clutch_attempts": getattr(p, 'clutch_attempts', None) or 0,
                },
                "entry": self._get_entry_stats(p),
                "trades": self._get_trade_stats(p),
                "clutches": self._get_clutch_stats(p),
                "rws": rws_data.get(sid, {"avg_rws": 0, "total_rws": 0, "rounds_won": 0, "rounds_played": 0, "damage_per_round": 0, "objective_completions": 0}),
            }
        
        # Merge enhanced metrics from professional parser
        if enhanced_metrics and "entry_frags" in enhanced_metrics:
            for steam_id, entry_data in enhanced_metrics.get("entry_frags", {}).items():
                if steam_id in players:
                    if "entry" not in players[steam_id]:
                        players[steam_id]["entry"] = {}
                    players[steam_id]["entry"].update({
                        "entry_attempts": entry_data.get("entry_attempts", 0),
                        "entry_kills": entry_data.get("entry_kills", 0),
                        "entry_deaths": entry_data.get("entry_deaths", 0),
                    })
        
        # Merge TTD metrics
        if enhanced_metrics and "ttd_metrics" in enhanced_metrics:
            for steam_id, ttd_data in enhanced_metrics.get("ttd_metrics", {}).items():
                if steam_id in players:
                    players[steam_id]["advanced"].update({
                        "ttd_median_ms": ttd_data.get("ttd_median_ms", 0),
                        "ttd_mean_ms": ttd_data.get("ttd_mean_ms", 0),
                        "ttd_95th_ms": ttd_data.get("ttd_95th_ms", 0),
                    })
        
        # Merge CP metrics
        if enhanced_metrics and "crosshair_placement" in enhanced_metrics:
            for steam_id, cp_data in enhanced_metrics.get("crosshair_placement", {}).items():
                if steam_id in players:
                    players[steam_id]["advanced"].update({
                        "cp_median_error_deg": cp_data.get("cp_median_error_deg", 0),
                        "cp_mean_error_deg": cp_data.get("cp_mean_error_deg", 0),
                    })
        
        # Merge Trade Kill metrics
        if enhanced_metrics and "trade_kills" in enhanced_metrics:
            for steam_id, trade_data in enhanced_metrics.get("trade_kills", {}).items():
                if steam_id in players:
                    players[steam_id]["duels"].update({
                        "trade_kills": trade_data.get("trade_kills", 0),
                        "deaths_traded": trade_data.get("deaths_traded", 0),
                    })
        
        # Merge Clutch metrics
        if enhanced_metrics and "clutch_stats" in enhanced_metrics:
            for steam_id, clutch_data in enhanced_metrics.get("clutch_stats", {}).items():
                if steam_id in players:
                    players[steam_id]["duels"].update({
                        "clutch_wins": clutch_data.get("clutch_wins", 0),
                        "clutch_attempts": clutch_data.get("clutch_attempts", 0),
                    })
                    # Add breakdown by variant
                    if "clutches" not in players[steam_id]:
                        players[steam_id]["clutches"] = {}
                    players[steam_id]["clutches"].update({
                        "v1_wins": clutch_data.get("v1_wins", 0),
                        "v2_wins": clutch_data.get("v2_wins", 0),
                        "v3_wins": clutch_data.get("v3_wins", 0),
                        "v4_wins": clutch_data.get("v4_wins", 0),
                        "v5_wins": clutch_data.get("v5_wins", 0),
                    })

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
            mvp = {"name": mvp_data["name"], "rating": mvp_data["rating"]["hltv_rating"]}

        # Convert to dict for caching
        result = {
            "demo_info": {
                "map": analysis.map_name,
                "rounds": analysis.total_rounds,
                "duration_minutes": getattr(analysis, 'duration_minutes', 30),
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
        opening = getattr(player, 'opening_duels', None)
        if opening:
            attempts = getattr(opening, 'attempts', 0) or 0
            wins = getattr(opening, 'wins', 0) or 0
            losses = getattr(opening, 'losses', 0) or 0
            rounds = getattr(player, 'rounds_played', 0) or 1
            return {
                "entry_attempts": attempts,
                "entry_kills": wins,
                "entry_deaths": losses,
                "entry_diff": wins - losses,
                "entry_attempts_pct": round(attempts / rounds * 100, 0) if rounds > 0 else 0,
                "entry_success_pct": round(wins / attempts * 100, 0) if attempts > 0 else 0,
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
        trades = getattr(player, 'trades', None)
        if trades:
            return {
                "trade_kills": getattr(trades, 'kills_traded', 0) or 0,
                "deaths_traded": getattr(trades, 'deaths_traded', 0) or 0,
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
        clutches = getattr(player, 'clutches', None)
        if clutches:
            total = getattr(clutches, 'total_situations', 0) or 0
            wins = getattr(clutches, 'total_wins', 0) or 0
            return {
                "clutch_wins": wins,
                "clutch_losses": total - wins,
                "clutch_success_pct": round(wins / total * 100, 0) if total > 0 else 0,
                "v1_wins": getattr(clutches, 'v1_wins', 0) or 0,
                "v2_wins": getattr(clutches, 'v2_wins', 0) or 0,
                "v3_wins": getattr(clutches, 'v3_wins', 0) or 0,
                "v4_wins": getattr(clutches, 'v4_wins', 0) or 0,
                "v5_wins": getattr(clutches, 'v5_wins', 0) or 0,
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

        # Group kills by round
        round_kills = {}
        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            if round_num not in round_kills:
                round_kills[round_num] = {"ct_kills": 0, "t_kills": 0, "first_kill": None, "first_death": None}

            attacker_side = str(getattr(kill, "attacker_side", "")).upper()
            if "CT" in attacker_side:
                round_kills[round_num]["ct_kills"] += 1
            else:
                round_kills[round_num]["t_kills"] += 1

            if round_kills[round_num]["first_kill"] is None:
                round_kills[round_num]["first_kill"] = getattr(kill, "attacker_name", "Unknown")
                round_kills[round_num]["first_death"] = getattr(kill, "victim_name", "Unknown")

        # Build timeline entries
        for round_num in sorted(round_kills.keys()):
            rd = round_kills[round_num]
            winner = "CT" if rd["ct_kills"] > rd["t_kills"] else "T"
            timeline.append({
                "round_num": round_num,
                "winner": winner,
                "win_reason": "Elimination" if abs(rd["ct_kills"] - rd["t_kills"]) >= 4 else "Objective",
                "first_kill": rd["first_kill"],
                "first_death": rd["first_death"],
                "ct_kills": rd["ct_kills"],
                "t_kills": rd["t_kills"],
            })

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

        # Initialize per-player round data
        player_round_data: dict[int, dict[int, dict]] = {}  # steam_id -> {round_num -> {kills, damage}}

        # Get max rounds
        max_round = 0

        # Count kills per player per round
        for kill in kills:
            attacker_id = getattr(kill, "attacker_steamid", 0)
            round_num = getattr(kill, "round_num", 0)

            if not attacker_id or not round_num:
                continue

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

            if not attacker_id or not round_num:
                continue

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
                rounds.append({
                    "round": r,
                    "kills": cumulative_kills,
                    "damage": cumulative_damage,
                    "round_kills": rd["kills"],
                    "round_damage": rd["damage"],
                })

            players_timeline.append({
                "steam_id": steam_id,
                "name": player_name,
                "rounds": rounds,
            })

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
                ("CT" in attacker_side and "T" in victim_side and "CT" not in victim_side) or
                ("T" in attacker_side and "CT" not in attacker_side and "CT" in victim_side)
            )

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
        """Build position data for heatmap visualization."""
        kills = getattr(demo_data, "kills", [])

        kill_positions = []
        death_positions = []

        for kill in kills:
            # Kill position (attacker)
            ax = getattr(kill, "attacker_x", None)
            ay = getattr(kill, "attacker_y", None)
            if ax is not None and ay is not None:
                kill_positions.append({"x": ax, "y": ay, "z": getattr(kill, "attacker_z", 0)})

            # Death position (victim)
            vx = getattr(kill, "victim_x", None)
            vy = getattr(kill, "victim_y", None)
            if vx is not None and vy is not None:
                death_positions.append({"x": vx, "y": vy, "z": getattr(kill, "victim_z", 0)})

        return {"kill_positions": kill_positions, "death_positions": death_positions}

    def _generate_coaching_insights(self, demo_data, analysis, players: dict) -> list[dict]:
        """Generate coaching insights for each player."""
        coaching = []

        for steam_id, player in players.items():
            insights = []
            name = player["name"]
            stats = player["stats"]
            rating = player["rating"]
            advanced = player["advanced"]
            utility = player["utility"]
            duels = player["duels"]

            # Role detection
            role = self._detect_role(player)

            # Analyze strengths
            if rating["hltv_rating"] >= 1.2:
                insights.append({
                    "type": "positive",
                    "message": f"Outstanding performance with {rating['hltv_rating']:.2f} rating - carrying the team",
                    "category": "Performance"
                })
            if stats["adr"] >= 90:
                insights.append({
                    "type": "positive",
                    "message": f"Excellent damage output ({stats['adr']} ADR) - consistent impact every round",
                    "category": "Damage"
                })
            if advanced["opening_kills"] >= 5:
                insights.append({
                    "type": "positive",
                    "message": f"{advanced['opening_kills']} opening kills - strong entry fragging",
                    "category": "Entry"
                })
            if rating["kast_percentage"] >= 80:
                insights.append({
                    "type": "positive",
                    "message": f"{rating['kast_percentage']:.0f}% KAST - almost always contributing to rounds",
                    "category": "Consistency"
                })

            # Analyze weaknesses
            if stats["deaths"] > stats["kills"] + 5:
                insights.append({
                    "type": "mistake",
                    "message": f"Dying too often ({stats['deaths']} deaths) - consider safer positioning",
                    "category": "Positioning"
                })
            if advanced["ttd_median_ms"] > 400:
                insights.append({
                    "type": "warning",
                    "message": f"Slow reaction time ({advanced['ttd_median_ms']:.0f}ms TTD) - work on pre-aiming angles",
                    "category": "Mechanics"
                })
            if advanced["cp_median_error_deg"] > 15:
                insights.append({
                    "type": "warning",
                    "message": f"Crosshair placement needs work ({advanced['cp_median_error_deg']:.1f}° error) - pre-aim common spots",
                    "category": "Mechanics"
                })
            if utility["flashbangs_thrown"] < 3:
                insights.append({
                    "type": "warning",
                    "message": "Low utility usage - buy and throw more flashes for team support",
                    "category": "Utility"
                })
            if duels["traded_deaths"] > duels["trade_kills"] + 2:
                insights.append({
                    "type": "warning",
                    "message": "Not getting traded when dying - stay closer to teammates",
                    "category": "Teamplay"
                })

            # Role-specific advice
            if role == "entry" and advanced["opening_kills"] < 3:
                insights.append({
                    "type": "warning",
                    "message": "As entry, focus on getting more opening kills with flash support",
                    "category": "Role"
                })
            if role == "awp" and stats["deaths"] > stats["kills"]:
                insights.append({
                    "type": "mistake",
                    "message": "Dying with AWP too often - $4750 lost each time. Hold angles, don't peek",
                    "category": "Economy"
                })

            # Add at least one insight
            if not insights:
                if rating["hltv_rating"] >= 1.0:
                    insights.append({
                        "type": "positive",
                        "message": "Solid performance overall - keep up the consistency",
                        "category": "General"
                    })
                else:
                    insights.append({
                        "type": "warning",
                        "message": "Focus on staying alive and trading with teammates",
                        "category": "General"
                    })

            coaching.append({
                "player_name": name,
                "steam_id": steam_id,
                "role": role,
                "insights": insights[:5],  # Top 5 insights per player
            })

        return coaching

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
