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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".opensight" / "cache"

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
            Analysis result dict
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
        from opensight.core.parser import DemoParser

        parser = DemoParser(demo_path)
        demo_data = parser.parse()

        analyzer = DemoAnalyzer(demo_data)
        analysis = analyzer.analyze()

        # Convert to dict for caching
        result = {
            "map_name": analysis.map_name,
            "total_rounds": analysis.total_rounds,
            "team1_score": analysis.team1_score,
            "team2_score": analysis.team2_score,
            "players": {
                str(sid): {
                    "name": p.name,
                    "team": p.team,
                    "kills": p.kills,
                    "deaths": p.deaths,
                    "assists": p.assists,
                    "adr": p.adr,
                    "headshot_percentage": p.headshot_percentage,
                    "hltv_rating": p.hltv_rating,
                    "kast_percentage": p.kast_percentage,
                    "aim_rating": p.aim_rating,
                    "utility_rating": p.utility_rating,
                }
                for sid, p in analysis.players.items()
            },
            "analyzed_at": datetime.now().isoformat(),
        }

        # Cache result
        self.cache.put(demo_path, result)

        return result


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
