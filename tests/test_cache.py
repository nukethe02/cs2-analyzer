"""Tests for the cache module — hash computation, cache hit/miss, index persistence."""

from __future__ import annotations

import pytest

from opensight.infra.cache import (
    CacheEntry,
    CacheStats,
    DemoCache,
    compute_content_hash,
    compute_file_hash,
)


class TestHashComputation:
    """Test SHA256 hash computation functions."""

    def test_compute_file_hash_consistency(self, tmp_path):
        """Hashing the same file twice produces the same result."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world content for hashing")

        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest is 64 chars

    def test_compute_file_hash_different_files(self, tmp_path):
        """Different file contents produce different hashes."""
        file_a = tmp_path / "a.bin"
        file_b = tmp_path / "b.bin"
        file_a.write_bytes(b"content A")
        file_b.write_bytes(b"content B")

        assert compute_file_hash(file_a) != compute_file_hash(file_b)

    def test_compute_content_hash_consistency(self):
        """Hashing the same string twice produces the same result."""
        h1 = compute_content_hash("test string")
        h2 = compute_content_hash("test string")
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64

    def test_compute_content_hash_different_strings(self):
        """Different strings produce different hashes."""
        assert compute_content_hash("alpha") != compute_content_hash("beta")


class TestCacheEntry:
    """Test CacheEntry serialization and deserialization."""

    def test_round_trip(self):
        """CacheEntry can be serialized to dict and back."""
        from datetime import datetime

        entry = CacheEntry(
            key="abc123_2.0.0",
            file_hash="abc123",
            file_path="/path/to/demo.dem",
            file_size=1024,
            created_at=datetime(2025, 1, 1, 12, 0, 0),
            accessed_at=datetime(2025, 1, 2, 12, 0, 0),
            analysis_version="2.0.0",
            compressed=True,
        )
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        assert restored.key == entry.key
        assert restored.file_hash == entry.file_hash
        assert restored.file_size == entry.file_size
        assert restored.compressed == entry.compressed
        assert restored.analysis_version == entry.analysis_version


class TestCacheStats:
    """Test CacheStats properties."""

    def test_hit_rate_zero_total(self):
        """hit_rate returns 0 when no hits or misses."""
        stats = CacheStats(
            total_entries=0,
            total_size_bytes=0,
            oldest_entry=None,
            newest_entry=None,
            hit_count=0,
            miss_count=0,
        )
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """hit_rate computes correctly."""
        stats = CacheStats(
            total_entries=5,
            total_size_bytes=1000,
            oldest_entry=None,
            newest_entry=None,
            hit_count=3,
            miss_count=7,
        )
        assert stats.hit_rate == 30.0  # 3/10 * 100

    def test_size_mb(self):
        """size_mb converts bytes to megabytes."""
        stats = CacheStats(
            total_entries=1,
            total_size_bytes=1024 * 1024,
            oldest_entry=None,
            newest_entry=None,
        )
        assert stats.size_mb == 1.0

    def test_to_dict(self):
        """to_dict returns all expected keys."""
        stats = CacheStats(
            total_entries=2,
            total_size_bytes=500,
            oldest_entry=None,
            newest_entry=None,
            hit_count=1,
            miss_count=1,
        )
        d = stats.to_dict()
        assert "total_entries" in d
        assert "total_size_mb" in d
        assert "hit_rate_pct" in d
        assert d["total_entries"] == 2


class TestDemoCache:
    """Test DemoCache put/get/miss behavior using a temp directory."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a DemoCache with a temporary directory."""
        return DemoCache(cache_dir=tmp_path / "cache")

    @pytest.fixture
    def demo_file(self, tmp_path):
        """Create a fake demo file for testing."""
        demo = tmp_path / "test_demo.dem"
        demo.write_bytes(b"fake demo content for testing")
        return demo

    def test_cache_miss(self, cache, demo_file):
        """get() returns None for uncached demo."""
        result = cache.get(demo_file)
        assert result is None

    def test_has_returns_false_for_uncached(self, cache, demo_file):
        """has() returns False for uncached demo."""
        assert cache.has(demo_file) is False

    def test_put_and_get(self, cache, demo_file):
        """put() stores data, get() retrieves it."""
        analysis = {"map": "de_dust2", "rounds": 24, "players": {"1": {"kills": 20}}}
        cache.put(demo_file, analysis)
        assert cache.has(demo_file) is True

        cached = cache.get(demo_file)
        assert cached is not None
        assert cached["map"] == "de_dust2"
        assert cached["rounds"] == 24

    def test_cache_key_deterministic(self, cache, demo_file):
        """get_cache_key returns the same key for the same file."""
        key1 = cache.get_cache_key(demo_file)
        key2 = cache.get_cache_key(demo_file)
        assert key1 == key2

    def test_invalidate(self, cache, demo_file):
        """invalidate() removes cached data."""
        cache.put(demo_file, {"test": True})
        assert cache.has(demo_file) is True

        cache.invalidate(demo_file)
        assert cache.has(demo_file) is False
        assert cache.get(demo_file) is None

    def test_clear(self, cache, demo_file):
        """clear() removes all cached data."""
        cache.put(demo_file, {"test": True})
        cache.clear()
        assert cache.has(demo_file) is False
        stats = cache.get_stats()
        assert stats.total_entries == 0

    def test_stats_after_put(self, cache, demo_file):
        """get_stats reflects cached entries."""
        cache.put(demo_file, {"test": True})
        stats = cache.get_stats()
        assert stats.total_entries == 1
        assert stats.total_size_bytes > 0

    def test_hit_miss_counters(self, cache, demo_file):
        """Hit and miss counters increment properly."""
        # Miss
        cache.get(demo_file)
        stats = cache.get_stats()
        assert stats.miss_count >= 1

        # Put then hit
        cache.put(demo_file, {"data": 1})
        cache.get(demo_file)
        stats = cache.get_stats()
        assert stats.hit_count >= 1

    def test_corrupted_index_recovery(self, tmp_path):
        """Cache recovers gracefully from corrupted index file."""
        cache_dir = tmp_path / "corrupt_cache"
        cache_dir.mkdir()
        index_path = cache_dir / "index.json"
        # Write invalid JSON
        index_path.write_text("{invalid json content!!")

        # Should not crash, should start with empty index
        cache = DemoCache(cache_dir=cache_dir)
        assert cache.get_stats().total_entries == 0

    def test_list_entries(self, cache, demo_file):
        """list_entries returns serialized cache entries."""
        cache.put(demo_file, {"test": True})
        entries = cache.list_entries()
        assert len(entries) == 1
        assert "key" in entries[0]
        assert "file_hash" in entries[0]
